# Chapter 18: NVIDIA GPU 后端——PTX 代码生成

> **学习目标**：
> - 理解 Triton NVIDIA 后端的完整代码生成路径（TritonGPU → LLVM → PTX → cubin）
> - 掌握 Tensor Core 指令映射（mma.sync.aligned.m16n8k8, ldmatrix）
> - 理解 Warp 级操作（shfl_sync, ballot_sync）在 Triton 归约中的使用
> - 掌握 PTX 代码生成过程与 Occupancy 优化
> - 了解 Triton 与 cuBLAS 的性能差距分析

---

## 18.1 NVIDIA 后端架构概览

### 18.1.1 Triton NVIDIA 后端目录结构

Triton 的 NVIDIA 后端代码主要分布在 `third_party/nvidia/` 和 `lib/Target/LLVMIR/` 中，负责将高层的 TritonGPU IR 最终转化为 NVIDIA GPU 可执行的 PTX 和 cubin 代码。

```
triton NVIDIA 后端代码结构
├── third_party/nvidia/
│   ├── lib/
│   │   ├── Dialect/NVMGM/
│   │   │   └── NVMMGMDialect.cpp          # NVMMGM Dialect 定义
│   │   ├── Dialect/NVGPU/
│   │   │   ├── NVGPUDialect.cpp           # NVGPU Dialect（warp 级操作）
│   │   │   └── NVGPUOps.cpp               # NVGPU 操作定义
│   │   ├── Dialect/TritonNVIDIAGPU/
│   │   │   ├── TritonNVIDIAGPUOps.cpp     # NVIDIA 专用 Triton 操作
│   │   │   └── Transform/                 # NVIDIA 特定 Pass
│   │   └── Transformation/               # TritonGPU → NVGPU 转换
│   ├── Dialect/
│   │   └── NVVMToLLVM/                   # NVVM → LLVM 转换
│   └── include/
│       └── triton/Dialect/NVGPU/         # NVGPU 头文件
├── lib/Target/LLVMIR/
│   ├── LLVMIRTranslation.cpp             # LLVM Dialect → LLVM IR
│   ├── LLVMIRAccelerationConversion.cpp  # Tensor Core 代码生成
│   └── PipelineLinearLayoutTranslation.cpp # TMA Pipeline 代码生成
├── lib/Target/PTX/
│   └── PTXTranslation.cpp                # PTX 相关辅助
└── lib/Conversion/
    ├── TritonGPUToLLVMPass.cpp            # TritonGPU → LLVM 总入口
    └── NVGPUToLLVM.cpp                    # NVGPU → LLVM 转换
```

### 18.1.2 完整编译路径

Triton 编写的 kernel 到最终 GPU 可执行代码，经历以下编译阶段：

```
Triton 后端完整编译路径

  Python Kernel
       │
       ▼
  ┌─────────────┐
  │  Triton IR   │  tt.load, tt.dot, tt.store
  │  (TTIR)      │  纯抽象 tile 操作
  └──────┬──────┘
         │  语言优化 Pass + TritonGPU Conversion
         ▼
  ┌─────────────┐
  │ TritonGPU   │  triton_gpu.load, triton_gpu.dot
  │  (TTGIR)    │  带有 Encoding 信息的 tile 操作
  └──────┬──────┘
         │  TritonGPU 优化 Pass（软件流水线, 共享内存优化等）
         │  TritonGPUToLLVM Conversion Pass
         ▼
  ┌─────────────┐
  │ LLVM Dialect │  llvm.load, llvm.store, llvm.fadd
  │  (LLVMIR)   │  平坦的标量操作，无 tile 概念
  └──────┬──────┘
         │  LLVM Dialect → NVVM Dialect
         │  LLVMDialectTranslation
         ▼
  ┌─────────────┐
  │  NVVM Dialect│  nvvm.mma.sync, nvvm.ldmatrix
  │             │  NVIDIA 硬件语义
  └──────┬──────┘
         │  NVVM → PTX Translation
         ▼
  ┌─────────────┐
  │    PTX      │  mma.sync.aligned.m16n8k8, ldmatrix.sync
  │  Assembly   │  字符串形式的 GPU 指令
  └──────┬──────┘
         │  PTXAS (NVIDIA 编译器)
         ▼
  ┌─────────────┐
  │   cubin     │  GPU 二进制可执行文件
  │  (SASS)     │  最终的 GPU 机器码
  └──────┬──────┘
         │
         ▼
    GPU Kernel 启动
```

### 18.1.3 各阶段的 IR 表示

```python
# 阶段 1: Python 源码
@triton.jit
def matmul_kernel(A, B, C, M, N, K, ...):
    pid = tl.program_id(0)
    # ... tile 计算逻辑
    c = tl.dot(a, b)
    tl.store(c_ptr + offs, c)
```

```mlir
// 阶段 2: Triton IR (TTIR)
tt.func @matmul_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>,
                        %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32, %arg5: i32) {
  %0 = tt.load %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}
      : tensor<128x64xf16>
  %1 = tt.load %arg1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}
      : tensor<64x128xf16>
  %2 = tt.dot %0, %1, %cst : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
  tt.store %arg2, %2 : tensor<128x128xf32>
  tt.return
}
```

```mlir
// 阶段 3: TritonGPU Dialect (TTGIR)
// 此时每个 tensor 都带有 Encoding 信息
%0 = triton_gpu.load %arg0
    : tensor<128x64xf16, #triton_gpu.blocked<{sizePerThread=[1, 1], threadsPerWarp=[32, 1], warpsPerCta=[4, 1], order=[0, 1]}>>
%1 = triton_gpu.dot %0, %1, %cst
    {allowTF32 = true, maxNumImpreciseAcc = 0 : i32}
    : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #triton_gpu.mma<{version = 2, warpsPerCta = [4, 1]}>}>>
    * tensor<64x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #triton_gpu.mma<{version = 2, warpsPerCta = [4, 1]}>}>>
    -> tensor<128x128xf32, #triton_gpu.mma<{version = 2, warpsPerCta = [4, 1]}>>
```

```llvm
// 阶段 4: LLVM IR (经过 NVVM Dialect 翻译后)
%0 = call { i32, i32, i32, i32 } @llvm.nvvm.mma.sync.m16n8k8.row.col.f32.f16.f16.f32(
    %a0, %a1, %a2, %a3, %b0, %b1, %c0, %c1, %c2, %c3)
    {leftLayouts = ...}
    : (v4f16, v4f16, v4f16, v4f16, v4f16, v4f16, v4f32, v4f32, v4f32, v4f32)
    -> { i32, i32, i32, i32 }
```

```ptx
// 阶段 5: PTX Assembly
mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
    {aDesc, bDesc}, %a0, %a1, %a2, %a3, %b0, %b1, {%c0, %c1, %c2, %c3};

shfl.sync.bfly.b32 %r0, %r1, 0x10, 0xffffffff;

st.shared.v4.b32 [%addr], {%r0, %r1, %r2, %r3};
```

### 18.1.4 TritonGPUToLLVM Conversion Pass 入口

TritonGPUToLLVM 是将高层 TritonGPU 操作转化为 LLVM Dialect 的关键转换。这个 Pass 负责将每个 `triton_gpu.*` 操作替换为对应的 LLVM 指令组合。

```cpp
// lib/Conversion/TritonGPUToLLVMPass.cpp (简化)
class TritonGPUToLLVM : public TritonGPUToLLVMBase<TritonGPUToLLVM> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    // 1. 准备 Conversion Target
    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<NVVM::NVVMDialect>();
    target.addIllegalDialect<TritonGPU::TritonGPUDialect>();

    // 2. 准备类型转换器
    TritonGPUTypeConverter typeConverter(numWarps, computeCapability);

    // 3. 添加所有合法化 Pattern
    RewritePatternSet patterns(context);
    mlir::triton::populateTritonGPUToLLVMConversionPatterns(
        typeConverter, patterns, target, ...);

    // 4. 添加 NVGPU → LLVM 转换
    mlir::triton::populateNVGPUTOLLVMConversionPatterns(
        typeConverter, patterns);

    // 5. 执行转换
    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      signalPassFailure();
  }
};
```

### 18.1.5 转换模式注册

Triton 通过 `populate*ConversionPatterns` 函数注册了大量的转换模式，每个模式负责将一种 TritonGPU 操作转化为 LLVM IR：

```cpp
// populateTritonGPUToLLVMConversionPatterns 注册的模式（部分）

// ─── Load/Store 相关 ───
patterns.add<LoadOpConversion>(typeConverter, patternBenefit);    // tt.load → GEP + load
patterns.add<StoreOpConversion>(typeConverter, patternBenefit);   // tt.store → store
patterns.add<AtomicCASOpConversion>(...);                          // atomic CAS
patterns.add<AtomicRMWOpConversion>(...);                          // atomic RMW
patterns.add<ExternElementwiseOpConversion>(...);                  // extern elementwise

// ─── 视图与转换 ───
patterns.add<ExpandDimsOpConversion>(...);    // tt.expand_dims → 重塑
patterns.add<SplatOpConversion>(...);          // tt.splat → broadcast
patterns.add<ArithConstantSplatOpConversion>(...); // 常量 splat
patterns.add<GetProgramIdOpConversion>(...);   // tl.program_id → nvvm.read
patterns.add<GetNumProgramsOpConversion>(...); // tl.num_programs → nvvm.read

// ─── Dot (Tensor Core) ───
patterns.add<DotOpConversion>(...);            // triton_gpu.dot → mma.sync
patterns.add<TransOpConversion>(...);          // 转置操作

// ─── 归约 ───
patterns.add<ReduceOpConversion>(...);         // triton_gpu.reduce → shfl_sync
patterns.add<AsyncCopyOpConversion>(...);      // cp.async
patterns.add<BarrierOpConversion>(...);        // mbarrier

// ─── Layout 转换 ───
patterns.add<ConvertLayoutOpConversion>(...);  // convert_layout
patterns.add<LocalLoadOpConversion>(...);      // 从共享内存加载
patterns.add<LocalStoreOpConversion>(...);     // 存储到共享内存
```

---

## 18.2 TTGIR → LLVM Dialect：操作降低

### 18.2.1 降低框架

从 TTGIR 到 LLVM Dialect 的转换是 Triton 编译过程中最复杂的阶段。每个操作需要被分解为 LLVM 层面的内存操作、算术运算和控制流。

```
TTGIR → LLVM 降低流程

┌──────────────────────────────────────────────────────────────┐
│                      TTGIR Module                            │
│  triton_gpu.dot, triton_gpu.load, triton_gpu.reduce ...     │
└───────────┬──────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────────┐
│                    Type Conversion                           │
│  tensor<128x64xf16, #blocked>                                │
│    → llvm.mlir.undef : !llvm.struct<(f16, f16, ...)>        │
│  每个 tensor 根据 Encoding 展开为具体的标量值序列               │
└───────────┬──────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────────┐
│                   Operation Conversion                       │
│  每个 triton_gpu.op 被 Pattern 匹配并替换为对应的              │
│  LLVM 指令序列（GEP, load, store, fadd, fmul, ...）         │
└───────────┬──────────────────────────────────────────────────┘
            │
            ▼
┌──────────────────────────────────────────────────────────────┐
│                      LLVM Dialect Module                     │
│  llvm.getelementptr, llvm.load, llvm.fadd, llvm.fmul ...    │
│  llvm.call @llvm.nvvm.mma.sync (Tensor Core intrinsics)     │
└──────────────────────────────────────────────────────────────┘
```

### 18.2.2 类型转换：Tensor 到标量

当 TTGIR 中的 tensor 进入 LLVM 层时，每个 tensor 根据其 Encoding 属性被展平为一个 LLVM 结构体（struct），其中每个元素对应一个线程持有的标量值。

```
Tensor → LLVM struct 展开示例

输入 tensor（BlockedEncoding）：
  tensor<128x64xf16, #blocked<{
    sizePerThread = [1, 4],
    threadsPerWarp = [32, 1],
    warpsPerCta = [4, 1]
  }>>

每个 CTA 有 4 个 warp × 32 个线程 = 128 个线程
每个线程持有一个 1×4 的小 tile（sizePerThread=[1,4]）
每个 CTA 处理 128×128 的数据（128 × 64 × 4 / 128 = ...）

LLVM struct: !llvm.struct<(f16, f16, f16, f16)>
  其中 struct 的每个元素 = 一个线程持有的 4 个 f16 值
  → 实际为 4 个独立的 f16 标量
```

```mlir
// 类型转换前后的对比
// TTGIR 层
%tile : tensor<128x64xf16, #blocked<{...}>

// LLVM 层（每个线程持有 4 个 f16 值）
%v0 : f16   // 线程持有的第 0 个元素
%v1 : f16   // 线程持有的第 1 个元素
%v2 : f16   // 线程持有的第 2 个元素
%v3 : f16   // 线程持有的第 3 个元素
```

### 18.2.2 LoadOp 降低

`triton_gpu.load` 被降低为 LLVM 的地址计算（GEP）+ 内存加载操作。根据编码信息，编译器生成正确的地址偏移计算，使得每个线程加载它需要的数据。

```cpp
// LoadOp 降低逻辑（简化）
class LoadOpConversion : public ConvertTritonGPUOp<LoadOp> {
  matchAndRewrite(LoadOp op, ...) {
    // 1. 获取指针和偏移
    auto basePtr = adaptor.getPtr();
    auto offsets = computeOffsets(op, rewriter);

    // 2. 计算每个线程的 GEP 地址
    for (int i = 0; i < numElems; ++i) {
      auto ptr = rewriter.create<LLVM::GEPOp>(loc, ptrType,
          basePtr, ArrayRef<Value>{offsets[i]});

      // 3. 执行内存加载
      auto load = rewriter.create<LLVM::LoadOp>(loc, elemType, ptr,
          /*alignment=*/16, /*volatile=*/false);
      loadedValues.push_back(load);
    }
    // 4. 组装为结构体
    rewriter.replaceOp(op, packValues(loadedValues, rewriter));
  }
};
```

```
LoadOp 降低中的地址计算

TTGIR: %val = triton_gpu.load %ptr : tensor<128x64xf16, #blocked<...>>

地址计算示意：
  线程 t (x 方向线程 ID)
  warp w (warp ID)
  
  row_base = w * 32 + t_x         // 每个线程负责的起始行
  col_base = (t_x // threadsPerWarp[0]) * sizePerThread[1]
  
  对于每个线程的 4 个元素:
    elem_addr[i] = base_ptr + (row_base + row_offset) * stride_row
                                + (col_base + i) * stride_col
```

### 18.2.3 StoreOp 降低

StoreOp 的降低与 LoadOp 类似，但方向相反：每个线程将其持有的标量值通过地址计算和 store 操作写回全局内存。

```cpp
class StoreOpConversion : public ConvertTritonGPUOp<StoreOp> {
  matchAndRewrite(StoreOp op, ...) {
    // 1. 计算目标地址
    auto ptrs = computePtrs(op, rewriter);
    // 2. 获取要存储的值
    auto vals = unpackValues(op.getVal());
    // 3. 执行 store
    for (int i = 0; i < numElems; ++i) {
      rewriter.create<LLVM::StoreOp>(loc, vals[i], ptrs[i]);
    }
    rewriter.eraseOp(op);
  }
};
```

### 18.2.4 Arith Operations 降低

Arith dialect 的操作（add, mul, cmp 等）直接对应到 LLVM 的同名操作，转换相对简单：

```mlir
// TTGIR: 每个操作在 tensor 级别
%a = triton_gpu.load %ptr_a : tensor<128x64xf16, #blocked<...>>
%b = triton_gpu.load %ptr_b : tensor<128x64xf16, #blocked<...>>

// LLVM: 展开为逐元素操作
%a0, %a1, %a2, %a3 = load ptr_a : f16, f16, f16, f16
%b0, %b1, %b2, %b3 = load ptr_b : f16, f16, f16, f16
%c0 = fadd %a0, %b0 : f16
%c1 = fadd %a1, %b1 : f16
%c2 = fadd %a2, %b2 : f16
%c3 = fadd %a3, %b3 : f16
```

---

## 18.3 Tensor Core 指令映射

### 18.3.1 Tensor Core 硬件背景

NVIDIA Tensor Core 是专用于矩阵乘累加（MMA）的硬件单元，每个 Tensor Core 在一个时钟周期内执行一次小规模的矩阵运算。

```
NVIDIA Tensor Core 各版本能力

┌────────────┬────────────────┬──────────────┬─────────────┐
│ 架构       │ 支持的数据类型  │ MMA 操作      │ 峰值 TFLOPS  │
├────────────┼────────────────┼──────────────┼─────────────┤
│ Volta      │ FP16×FP16+FP32 │ m16n16k16    │ 125 (V100)  │
│ Turing     │ INT8×INT8+INT32│ m16n16k16    │ 130 (RTX2080)│
│ Ampere     │ BF16×BF16+FP32 │ m16n8k16     │ 312 (A100)  │
│            │ FP64×FP64+FP64 │ m16n8k4      │ 19.5 (A100) │
│            │ TF32×TF32+FP32 │ m16n8k8      │ 156 (A100)  │
│ Hopper     │ FP8×FP8+FP32   │ m16n8k16     │ 3958 (H100) │
│            │ FP8×FP8+FP16   │ m16n8k32     │ 1979 (H100) │
│            │ INT8×INT8+INT32│ m16n8k32     │ 3958 (H100) │
└────────────┴────────────────┴──────────────┴─────────────┘

MMA 指令命名规则:
  mma.sync.aligned.m16n8k8.{row|col}.{dtype_a}{dtype_b}.{dtype_c}
       │         │      │      │
       │         │      │      └── 数据布局: row-major 或 col-major
       │         │      └── 矩阵维度: M×N×K
       │         └── 对齐的 (所有参与线程同步执行)
       └── 同步的 (需要 __syncwarp 保证)
```

### 18.3.2 mma.sync.aligned.m16n8k8 指令详解

以 Ampere (SM80) 上的 FP16 为例，`mma.sync.aligned.m16n8k8` 指令是最常用的 Tensor Core 操作之一。

```
mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32

语义: D[M×N] = A[M×K] × B[K×N] + C[M×N]
      其中 M=16, N=8, K=8
      A 是 row-major, B 是 col-major
      输入: f16, 累加: f32

硬件执行:
  一个 warp (32 threads) 协同执行一次 MMA
  每个线程持有一部分 A, B, C/D 的数据

  线程分配（A矩阵, row-major）:
  ┌────┬────┬────┬────┬────┬────┬────┬────┐
  │ t0 │ t1 │ t2 │ t3 │ t4 │ t5 │ t6 │ t7 │ row 0
  ├────┼────┼────┼────┼────┼────┼────┼────┤
  │ t8 │ t9 │t10 │t11 │t12 │t13 │t14 │t15 │ row 1
  ├────┼────┼────┼────┼────┼────┼────┼────┤
  │t16 │t17 │t18 │t19 │t20 │t21 │t22 │t23 │ row 2
  ├────┼────┼────┼────┼────┼────┼────┼────┤
  │t24 │t25 │t26 │t27 │t28 │t29 │t30 │t31 │ row 3
  └────┴────┴────┴────┴────┴────┴────┴────┘
  每个线程持有 4 个 f16 元素（两列）

  线程分配（B矩阵, col-major）:
  每个线程持有 2 个 f16 元素（两行）
  
  线程分配（D/C矩阵）:
  每个线程持有 4 个 f32 元素
```

```
mma.sync.aligned.m16n8k8 操作分解

输入:
  A: 16×8 矩阵, row-major, f16
     每个线程持有 4 个元素 = 32 × 4 = 128 = 16 × 8 ✓
  B: 8×8 矩阵, col-major, f16
     每个线程持有 2 个元素 = 32 × 2 = 64 = 8 × 8 ✓
  C: 16×8 矩阵, f32 (累加器)
     每个线程持有 4 个元素 = 32 × 4 = 128 = 16 × 8 ✓

输出:
  D: 16×8 矩阵, f32
     D = A × B + C

PTX 语法:
  mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {a0,a1,a2,a3}, {b0,b1}, {c0,c1,c2,c3};

  其中:
    {a0,a1,a2,a3} : 线程持有的 A 矩阵片段 (v4f16)
    {b0,b1}       : 线程持有的 B 矩阵片段 (v4f16, 实际2个)
    {c0,c1,c2,c3} : 线程持有的 C 累加器 (v4f32)
    输出覆盖 c0,c1,c2,c3
```

### 18.3.3 Triton DotOp → MMA 降低过程

`triton_gpu.dot` 操作到 `mma.sync` 的降低是 NVIDIA 后端最核心的代码生成路径。

```
Triton DotOp → MMA 降低步骤

Step 1: 分析 DotOp 的 Encoding
  ┌──────────────────────────────────────────────────────────────┐
  │ %d = triton_gpu.dot %a, %b, %c                              │
  │   {layoutA = #dot_op<{opIdx=0, parent=#mma{...}}>           │
  │    layoutB = #dot_op<{opIdx=1, parent=#mma{...}}>           │
  │    layoutC = #mma<{version=2, warpsPerCta=[4,1]}>}         │
  │                                                              │
  │ → 确认使用 MMA v2 (Ampere), warpsPerCta = [4, 1]           │
  │ → 选择 mma.sync.aligned.m16n8k8 指令                        │
  └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 2: 从寄存器布局提取 MMA 片段
  ┌──────────────────────────────────────────────────────────────┐
  │ 每个线程从其持有的 f16 标量值中，组装成 v4f16 向量           │
  │                                                              │
  │ a_thread = {a_elem0, a_elem1, a_elem2, a_elem3} : v4f16    │
  │ b_thread = {b_elem0, b_elem1} : v4f16 (实际用 v4f16 打包) │
  │ c_thread = {c_elem0, c_elem1, c_elem2, c_elem3} : v4f32    │
  └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 3: 调用 NVVM MMA intrinsic
  ┌──────────────────────────────────────────────────────────────┐
  │ %d0, %d1, %d2, %d3 = @llvm.nvvm.mma.sync.m16n8k8.row.col   │
  │     .f32.f16.f16.f32(                                       │
  │       %a0, %a1, %a2, %a3,    // A 片段                      │
  │       %b0, %b1,              // B 片段                       │
  │       %c0, %c1, %c2, %c3    // C 累加器                     │
  │     ) : (v4f16, v4f16, v4f16, v4f16, v4f16, v4f16,         │
  │           v4f32, v4f32, v4f32, v4f32)                        │
  │     -> (v4f32, v4f32, v4f32, v4f32)                          │
  └──────────────────────────────────────────────────────────────┘
```

```cpp
// DotOpConversion 的核心逻辑（简化）
class DotOpConversion : public ConvertTritonGPUOp<triton::gpu::DotOp> {
  matchAndRewrite(DotOp op, ...) {
    // 1. 获取 encoding 信息
    auto aEncoding = op.getA().getType().getEncoding();
    auto bEncoding = op.getB().getType().getEncoding();
    auto dEncoding = op.getD().getType().getEncoding();

    // 2. 确定使用哪种 Tensor Core 指令
    auto mmaEncoding = dEncoding.cast<MmaEncodingAttr>();
    auto versionMajor = mmaEncoding.getVersionMajor();

    // 3. 提取 A, B, C/D 线程持有的值
    auto aVals = unpackThreadValues(op.getA());
    auto bVals = unpackThreadValues(op.getB());
    auto cVals = unpackThreadValues(op.getC());

    // 4. 根据矩阵布局组装向量片段
    //    对于 row-major A: {v0, v1, v2, v3} 组装为 v4f16
    Value aVec = packAsVector(aVals, rewriter);  // v4f16
    Value bVec = packAsVector(bVals, rewriter);  // v4f16
    Value cVec = packAsVector(cVals, rewriter);  // v4f32

    // 5. 调用 NVVM MMA intrinsic
    if (versionMajor == 2) {
      // Ampere MMA
      auto mmaOp = rewriter.create<NVVM::MmaSyncOp>(
          loc, typeConverter.convertType(op.getType()),
          aVec, bVec, cVec,
          /*aCol=*/false, /*bCol=*/true,
          /*m=*/16, /*n=*/8, /*k=*/8);
    }

    rewriter.replaceOp(op, mmaOp);
  }
};
```

### 18.3.4 ldmatrix 指令

`ldmatrix.sync.aligned.m8n8.x4` 用于从共享内存高效加载 Tensor Core 所需的数据片段。它通过矩阵转置和重排，直接生成 MMA 指令所需的寄存器布局。

```
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 语义

从共享内存加载 4 个 8×8 的小矩阵片段
每个线程提供一个 128-bit 的共享内存地址
硬件自动完成:
  1. 从共享内存加载 4×8 矩阵
  2. 执行转置操作
  3. 按 MMA 要求的格式放入寄存器

  共享内存布局 (每个线程持有一个 16-byte 地址):
  ┌────────────────────────────────────┐
  │  thread 0: addr_0                  │  → 8×8 矩阵片段 0
  │  thread 1: addr_1                  │
  │  ...                              │
  │  thread 31: addr_31                │
  │  thread 32: addr_32                │  → 8×8 矩阵片段 1
  │  ...                              │
  │  thread 63: addr_63                │
  │  thread 64: addr_64                │  → 8×8 矩阵片段 2
  │  ...                              │
  │  thread 95: addr_95                │
  │  thread 96: addr_96                │  → 8×8 矩阵片段 3
  │  ...                              │
  │  thread 127: addr_127              │
  └────────────────────────────────────┘

  输出: 每个线程获得 4 个 v2f16 寄存器
  (共 4 个 8×8 矩阵, 每个线程持有其中的部分)
```

```
ldmatrix vs 手动加载对比

方式 1: 手动从共享内存加载
  每个线程需要:
  - 16 次 shared memory load (128 bytes)
  - 16 次标量到向量组装
  - 可能产生 bank conflict

方式 2: ldmatrix
  每个线程需要:
  - 1 次 ldmatrix.x4 指令 (128 bytes)
  - 硬件自动转置和重排
  - 保证无 bank conflict (通过硬件仲裁)

  性能差异: ldmatrix 比手动加载快 2-4x
```

### 18.3.5 MMA 降低的完整 PTX 输出

以一个 128×128 的 matrix multiplication 为例，展示 DotOp 降低后生成的关键 PTX 指令：

```ptx
// ══════════════════════════════════════════════════════════════
// DotOp 降低后的 PTX 片段 (matmul 128×128×64, Ampere A100)
// ══════════════════════════════════════════════════════════════

// --- 从共享内存加载 A 矩阵片段 (ldmatrix) ---
ldmatrix.sync.aligned.m8n8.x4.shared.b16  {%r0, %r1}, [%addr_a];
// 生成 4 个 8×8 的矩阵片段到寄存器

// --- 从共享内存加载 B 矩阵片段 (ldmatrix) ---
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16  {%r4, %r5}, [%addr_b];
// 生成 4 个 8×8 的矩阵片段 (转置) 到寄存器

// --- 执行 Tensor Core 累加 ---
// 对于每个 K 块 (k=0..7):
mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
    {%rd0, %rd1, %rd2, %rd3},   // 输出 D
    {%ra0, %ra1, %ra2, %ra3},   // A 片段
    {%rb0, %rb1},               // B 片段
    {%rc0, %rc1, %rc2, %rc3};   // C 累加器
```

### 18.3.6 多版本 MMA 支持

Triton 支持不同 GPU 架构的 MMA 指令版本：

```cpp
// MMA 版本选择逻辑
enum class MmaVersion { V1 = 1, V2 = 2, V3 = 3, V4 = 4 };

MmaVersion getMmaVersion(int computeCapability) {
  if (computeCapability >= 90) return MmaVersion::V4;  // Hopper
  if (computeCapability >= 80) return MmaVersion::V3;  // Ampere
  if (computeCapability >= 75) return MmaVersion::V2;  // Turing
  return MmaVersion::V1;                                // Volta
}

// 不同版本的 MMA 操作
V1: mma.sync.aligned.m16n16k16.row.col.f16.f16.f32.f32   // Volta
V2: mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32     // Turing/Ampere
V3: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32  // Ampere BF16
V4: mma.sync.aligned.m16n8k16.row.col.f32.fp8.fp8.f32     // Hopper FP8
```

---

## 18.4 Warp 级操作

### 18.4.1 Warp 基础概念

```
NVIDIA GPU Warp 架构

GPU 执行模型:
  Grid (网格)
  ├── CTA (Cooperative Thread Array) = Block
  │   ├── Warp 0  (thread 0-31)
  │   ├── Warp 1  (thread 32-63)
  │   ├── Warp 2  (thread 64-95)
  │   └── Warp 3  (thread 96-127)
  │
  Warp 是 GPU 调度的基本单位
  一个 warp = 32 个线程, 执行 SIMD 模式
  同一 warp 内的线程可以进行 warp 级操作:
    - shuffle (数据交换)
    - vote/ballot (投票)
    - match (匹配)
    - reduce (归约)
```

### 18.4.2 __shfl_sync: Warp 内数据交换

`__shfl_sync` 允许 warp 内线程之间直接交换数据，无需通过共享内存。这是实现快速归约和扫描操作的关键。

```
__shfl_sync.mask.b32 语义

__shfl_sync(mask, val, srcLane, width)
  mask    : 活跃线程掩码 (0xFFFFFFFF = 所有 32 线程)
  val     : 要交换的值
  srcLane : 源线程 ID
  width   : 子 warp 宽度 (1, 2, 4, 8, 16, 32)

返回值: 从 srcLane 线程获取的值

变体:
  __shfl_up_sync    : 从 srcLane = laneId - delta 获取
  __shfl_down_sync  : 从 srcLane = laneId + delta 获取
  __shfl_xor_sync   : 从 srcLane = laneId ^ laneMask 获取
```

```
__shfl_xor_sync 用于 Butterfly Reduction

初始状态: 每个线程有一个值
  thread: 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
  value:  v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15

Step 1: xor 1 (交换相邻线程, stride=1)
  (0,1) (2,3) (4,5) (6,7) (8,9) (10,11) (12,13) (14,15)
  → v0+v1, v2+v3, v4+v5, v6+v7, v8+v9, v10+v11, v12+v13, v14+v15

Step 2: xor 2 (交换间隔 2 的线程)
  (0,2) (1,3) (4,6) (5,7) (8,10) (9,11) (12,14) (13,15)
  → [v0+v1+v2+v3], [v4+v5+v6+v7], [v8+v9+v10+v11], [v12+v13+v14+v15]

Step 3: xor 4 (交换间隔 4 的线程)
  (0,4) (1,5) (2,6) (3,7) (8,12) (9,13) (10,14) (11,15)
  → [v0+v1+...+v7], [v8+v9+...+v15]

Step 4: xor 8 (交换间隔 8 的线程)
  (0,8) (1,9) (2,10) (3,11) (4,12) (5,13) (6,14) (7,15)
  → [v0+v1+...+v15]   ← 归约结果在 thread 0

总步数: log2(32) = 5 步, 每步一条指令
```

```
CUDA 代码: Warp Butterfly Reduction

__device__ float warpReduceMax(float val) {
  // mask=0xffffffff 表示所有 32 个线程参与
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other = __shfl_xor_sync(0xffffffff, val, offset, 32);
    val = fmaxf(val, other);
  }
  return val;  // 结果在所有线程中 (可通过 shfl 广播)
}
```

### 18.4.3 __ballot_sync: 活跃线程投票

`__ballot_sync` 用于确定哪些线程满足某个条件。它返回一个 32 位掩码，其中每个 bit 对应一个线程。

```
__ballot_sync 语义

uint32_t __ballot_sync(unsigned mask, int pred)
  mask : 活跃线程掩码
  pred : 每个线程的谓词 (0 或 1)

返回值: 32-bit 掩码, bit[i] = (mask 中线程 i 的 pred 值)

示例:
  线程:    0  1  2  3  4  5  6  7
  pred:    1  0  1  1  0  0  1  0
  ballot:  0b10110101 = 0xB5 = 181

  → 线程 0, 2, 3, 6 的 pred 为真
```

### 18.4.4 Triton ReduceOp 降低为 Warp Shuffle

Triton 的 `triton_gpu.reduce` 操作在 NVIDIA 后端被降低为 warp 级的 shuffle 操作序列。

```
Triton Reduce 降低流程

TTGIR: %result = triton_gpu.reduce %val {axis = 0}
       // 在轴 0 上执行 max 归约

降低步骤:
┌──────────────────────────────────────────────────────────────────┐
│ Step 1: 识别归约模式                                             │
│   - axis = 0, 在 warp 内归约                                     │
│   - 选择 warp shuffle reduction 策略                             │
└──────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 2: 生成 butterfly reduction 代码                            │
│   for (int offset = 16; offset > 0; offset >>= 1):              │
│     other = __shfl_xor_sync(0xffffffff, val, offset, 32);       │
│     val = max(val, other);                                       │
└──────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 3: 跨 warp 归约 (如果需要)                                  │
│   - 通过共享内存进行 warp 间归约                                  │
│   - 或使用 atomicAdd 做全局归约                                   │
└──────────────────────────────────────────────────────────────────┘
```

```cpp
// ReduceOp 降低代码（简化）
class ReduceOpConversion : public ConvertTritonGPUOp<triton::gpu::ReduceOp> {
  matchAndRewrite(ReduceOp op, ...) {
    auto axis = op.getAxis();
    auto reduceOp = op.getCombineFn();  // max, min, add, ...

    // 1. 判断是否可以使用 warp shuffle
    if (canUseWarpShuffle(op)) {
      // Butterfly reduction with __shfl_xor_sync
      auto mod = op->getParentOfType<ModuleOp>();
      auto loc = op.getLoc();

      // 对于每个归约 lane，生成 shuffle 代码
      for (int i = 0; i < numLanes; ++i) {
        Value val = getLaneValue(op, i);
        for (int offset = numLanes / 2; offset > 0; offset >>= 1) {
          Value other = rewriter.create<NVVM::ShflOp>(
              loc, valType,
              /*srcLane=*/val,
              /*offset=*/offset,
              /*mask=*/i32_all,
              /*width=*/numLanes);
          val = applyReduction(reduceOp, val, other);
        }
        setLaneResult(op, i, val);
      }
    } else {
      // 跨 warp 归约: 使用共享内存
      lowerToSharedMemReduction(op, rewriter);
    }
  }
};
```

### 18.4.5 Warp Shuffle 在 Softmax 中的应用

Softmax 的行归约是 warp shuffle 的经典应用场景：

```
Softmax 的 exp + sum 归约

输入: 一行 32 个浮点数, 每个线程持有一个元素

Step 1: 找到最大值 (warp reduce max)
  for offset in [16, 8, 4, 2, 1]:
    max_val = max(max_val, __shfl_xor_sync(mask, max_val, offset))

Step 2: 计算 exp(x - max) (逐元素)
  exp_val = exp(local_val - max_val)

Step 3: 求和 (warp reduce sum)
  for offset in [16, 8, 4, 2, 1]:
    sum_val = sum_val + __shfl_xor_sync(mask, sum_val, offset)

Step 4: 归一化
  result = exp_val / sum_val

PTX 输出:
  // Step 1: Warp reduce max
  shfl.sync.bfly.b32 %max1, %max0, 16, -1;
  max.f32 %max0, %max0, %max1;
  shfl.sync.bfly.b32 %max2, %max0, 8, -1;
  max.f32 %max0, %max0, %max2;
  shfl.sync.bfly.b32 %max3, %max0, 4, -1;
  max.f32 %max0, %max0, %max3;
  shfl.sync.bfly.b32 %max4, %max0, 2, -1;
  max.f32 %max0, %max0, %max4;
  shfl.sync.bfly.b32 %max5, %max0, 1, -1;
  max.f32 %max0, %max0, %max5;

  // Step 2: 计算 exp
  sub.f32 %diff, %local_val, %max0;
  ex2.approx.f32 %exp_val, %diff;

  // Step 3: Warp reduce sum (同样的 butterfly)
  // ... (与 max 相同的结构, 但使用 add)
```

### 18.4.6 Warp 级同步

`__syncwarp` 保证同一 warp 内所有线程到达同一执行点，这对于 MMA 指令的正确执行至关重要：

```ptx
// mma.sync 之前需要确保所有线程的寄存器值已就绪
bar.sync 0;                              // Block-level 同步
ldmatrix.sync.aligned.m8n8.x4.shared.b16 ...;  // 加载 A 矩阵
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 ...;  // 加载 B 矩阵
bar.sync 1;                              // 确保数据加载完成
// mma.sync 执行时，硬件保证同一 warp 内的线程同步
mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 ...;
```

---

## 18.5 PTX 代码生成详解

### 18.5.1 NVVM Dialect 到 PTX 的翻译

NVVM Dialect 是 NVIDIA GPU 专用的 LLVM 方言，包含了 MMA、warp shuffle、共享内存等 NVIDIA 特有操作的 MLIR 表示。NVVM Dialect 被直接翻译为 PTX 指令。

```
NVVM Dialect → PTX 翻译对应关系

NVVM Dialect                        PTX 指令
─────────────────────────────────────────────────────────────
NVVM::MmaSyncOp                     mma.sync.aligned.m16n8k8.row.col...
NVVM::LdMatrixOp                    ldmatrix.sync.aligned.m8n8.x4.shared...
NVVM::ShflOp (bfly)                 shfl.sync.bfly.b32
NVVM::ShflOp (up)                   shfl.sync.up.b32
NVVM::ShflOp (down)                 shfl.sync.down.b32
NVVM::VoteOp (ballot)               vote.sync.ballot.b32
NVVM::VoteOp (any)                  vote.sync.any.b32
NVVM::VoteOp (all)                  vote.sync.all.b32
NVVM::MemBarOp                      mbarrier.init / cp.async.commit_group
NVVM::CpAsyncOp                     cp.async.ca.shared.global
LLVM::LoadOp (.shared addr)         ld.shared.b32
LLVM::StoreOp (.shared addr)        st.shared.b32
LLVM::LoadOp (.global addr)         ld.global.b32
LLVM::StoreOp (.global addr)        st.global.b32
LLVM::FAddOp                        add.f32
LLVM::FMulOp                        mul.f32
LLVM::FMAOp                         fma.rn.f32
```

### 18.5.2 完整 Matmul Kernel 的 PTX 输出

以一个简化的 64×64×64 matrix multiplication 为例，展示 Triton 生成的完整 PTX：

```ptx
// ══════════════════════════════════════════════════════════════════
// Triton 生成的 Matmul PTX (简化版, SM80, FP16)
// 64×64 矩阵乘法, 每个 CTA 计算 64×64 的输出
// ══════════════════════════════════════════════════════════════════

.version 7.8
.target sm_80
.address_size 64

.entry matmul_kernel(
    .param .u64 A,
    .param .u64 B,
    .param .u64 C,
    .param .u32 M,
    .param .u32 N,
    .param .u32 K
)
{
    .reg .pred  %p<4>;
    .reg .f32   %f<32>;
    .reg .f16   %h<64>;
    .reg .b32   %r<128>;
    .reg .b64   %rd<32>;
    .shared .align 16 .b8 smem[16384];  // 16KB 共享内存

    // ─── 线程与 CTA 索引 ───
    mov.u32     %r1, %ctaid.x;            // CTA ID (program_id)
    mov.u32     %r2, %tid.x;              // 线程 ID (threadIdx.x)
    mov.u32     %r3, %ntid.x;             // 每 CTA 线程数 (blockDim.x)
    mov.u32     %r4, %r2;                 // 复制线程 ID

    // ─── 地址计算 ───
    cvt.u64.u32 %rd1, %r1;                // CTA ID → 64-bit
    mul.wide.u32 %rd2, %rd1, %r3;         // CTA 偏移 × blockDim
    add.u64      %rd3, %rd2, %rd4;        // 全局线程 ID

    // ─── 共享内存指针 ───
    cvta.to.shared.u64 %rd10, smem;        // 共享内存地址空间转换

    // ─── K 维度循环 ───
    mov.u32 %r20, 0;                       // k = 0
    mov.u32 %r21, 64;                      // K = 64 (循环上界)

.LBB0_1:  // K 循环体
    setp.ge.u32 %p1, %r20, %r21;          // k >= K?
    @%p1 bra.uni .LBB0_3;                 // 退出循环

    // ─── 从全局内存加载 A 矩阵 tile 到共享内存 ───
    // 每个线程加载 4 个 f16 值 (16 bytes)
    ld.global.v4.b16 {%h0, %h1, %h2, %h3}, [%rd_addr_a];
    st.shared.v4.b16 [%smem_a_addr], {%h0, %h1, %h2, %h3};

    // ─── 从全局内存加载 B 矩阵 tile 到共享内存 ───
    ld.global.v4.b16 {%h4, %h5, %h6, %h7}, [%rd_addr_b];
    st.shared.v4.b16 [%smem_b_addr], {%h4, %h5, %h6, %h7};

    // ─── 同步: 确保所有线程完成共享内存写入 ───
    bar.sync 0;

    // ─── 从共享内存加载到寄存器 (ldmatrix) ───
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r10, %r11, %r12, %r13}, [%smem_a_load];
    ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r20, %r21, %r22, %r23}, [%smem_b_load];

    // ─── Tensor Core MMA 操作 ───
    mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
        {%rd0, %rd1, %rd2, %rd3},         // 输出 D (16×8, f32)
        {%r10, %r11, %r12, %r13},         // A 片段 (16×8, f16)
        {%r20, %r21},                      // B 片段 (8×8, f16)
        {%rd0, %rd1, %rd2, %rd3};         // C 累加器

    // ─── 同步: 确保 MMA 完成后再使用共享内存 ───
    bar.sync 0;

    // ─── 下一轮 K 的地址偏移 ───
    add.u32 %r20, %r20, 8;                // k += 8 (K=8 per mma)
    bra.uni .LBB0_1;                       // 回到循环顶部

.LBB0_3:  // K 循环结束
    // ─── 将结果从寄存器写回全局内存 ───
    // 每个线程存储 4 个 f32 值
    st.global.v4.f32 [%rd_out_addr], {%rd0, %rd1, %rd2, %rd3};

    ret;
}
```

### 18.5.3 PTX 关键指令解析

```
PTX 指令详解

1. 地址空间转换
   cvta.to.shared.u64 %rd, smem;     // 将通用地址转为共享内存地址
   cvta.to.global.u64 %rd, %ptr;     // 将共享内存地址转为全局地址
   cvta.shared.to.global.u64 %rd, %rs; // 共享 → 全局地址转换

2. 内存加载模式
   ld.global.v4.b16 {h0,h1,h2,h3}, [addr];    // 向量加载 4×16-bit
   ld.shared.b32 %r, [addr];                    // 共享内存加载 32-bit
   ld.shared.v4.b32 {r0,r1,r2,r3}, [addr];     // 共享内存向量加载

3. 内存存储模式
   st.global.v4.f32 [addr], {f0,f1,f2,f3};     // 全局内存向量存储
   st.shared.v4.b32 [addr], {r0,r1,r2,r3};     // 共享内存向量存储

4. 同步原语
   bar.sync 0;            // Block-level barrier, 所有线程同步
   bar.sync 1, 32;       // 带计数的 barrier (32 个线程参与)
   mbarrier.init [addr], 1;  // 初始化 memory barrier

5. Tensor Core
   mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 ...;
   // .sync      : warp 内所有线程同步执行
   // .aligned   : 输入数据按 MMA 要求对齐
   // .m16n8k8   : 矩阵维度 M=16, N=8, K=8
   // .row.col   : A 是行主序, B 是列主序
   // .f32.f16.f16.f32 : 输入 f16, 累加 f32

6. Warp Shuffle
   shfl.sync.bfly.b32 %rd, %rs, 16, -1;
   // .bfly      : butterfly 模式 (与 offset 线程交换)
   // .b32       : 32-bit 数据
   // offset=16  : 与 offset 为 16 的线程交换
   // -1         : mask = 0xFFFFFFFF (所有线程)

   shfl.sync.up.b32 %rd, %rs, 1, 0, -1;
   // .up        : 从 laneId - delta 获取数据
   // delta=1    : 从前一个线程获取
   // clamp=0    : 不 clamp, 超出范围返回自身
```

### 18.5.4 PTX 指令调度与延迟

```
Ampere (SM80) 指令延迟与吞吐

┌─────────────────────────┬──────────┬──────────┬───────────┐
│ 指令类型                 │ 延迟(cyc)│ 吞吐/cyc  │ 说明       │
├─────────────────────────┼──────────┼──────────┼───────────┤
│ FP32 算术               │ 4-8      │ 每 SM 多  │ 标量运算   │
│ FP16 算术               │ 4-8      │ 2×FP32   │ 半精度运算 │
│ FMA (乘加)              │ 4        │ 1/SM/cyc  │ 单指令乘加 │
│ FP32 Load (L2)          │ ~200     │ -        │ L2 cache  │
│ FP32 Load (L1)          │ ~30      │ -        │ L1 cache  │
│ Shared Memory Load      │ ~20-30   │ 1/cyc/SM │ bank 特定 │
│ Shared Memory Store     │ ~20-30   │ 1/cyc/SM │ bank 特定 │
│ Global Memory Load      │ ~300-600 │ -        │ HBM2e    │
│ Warp Shuffle            │ ~5-10    │ 1/cyc/SM │ 寄存器 → 寄存器 │
│ MMA (Tensor Core)       │ ~12-16   │ 1/cyc/SM │ 16×8×8   │
│ Barrier                 │ 可变     │ -        │ 全局同步   │
│ LDMATRIX                │ ~20-30   │ 1/cyc/SM │ 共享→寄存器│
└─────────────────────────┴──────────┴──────────┴───────────┘

关键优化原则:
  - 指令级并行 (ILP): 发射多条独立指令隐藏延迟
  - 内存级并行 (MLP): 同时发起多个内存请求
  - 寄存器压力: 平衡寄存器使用和 occupancy
```

---

## 18.6 共享内存管理

### 18.6.1 共享内存声明与布局

```ptx
// PTX 共享内存声明
.shared .align 16 .b8 smem[16384];         // 静态共享内存, 16KB, 16字节对齐
.shared .align 16 .b8 smem_dynamic[];       // 动态共享内存 (大小由 host 传入)

// 共享内存地址空间
// 通用地址 → 共享内存地址:
cvta.to.shared.u64 %rd_shmem, %ptr;

// 共享内存地址 → 通用地址:
cvta.shared.to.global.u64 %rd_global, %rd_shmem;
```

```
共享内存布局示意

Triton 将共享内存划分为不同区域:
┌──────────────────────────────────────────────────────┐
│                  共享内存 (Shared Memory)              │
├──────────────────────────────────────────────────────┤
│  ┌────────────────────┐  ┌────────────────────┐      │
│  │   A Matrix Tile    │  │   B Matrix Tile    │      │
│  │   (行优先布局)      │  │   (列优先布局)     │      │
│  │   8KB              │  │   8KB              │      │
│  └────────────────────┘  └────────────────────┘      │
│                                                      │
│  或使用 padding 避免 bank conflict:                   │
│  ┌──────────────────────┐  ┌────────────────────┐    │
│  │ A Tile + 8B padding  │  │ B Tile + 8B padding│    │
│  │                      │  │                    │    │
│  └──────────────────────┘  └────────────────────┘    │
└──────────────────────────────────────────────────────┘

Bank Conflict 说明:
  共享内存有 32 个 bank, 每个 bank 宽度 4 bytes
  同一 warp 内不同线程访问同一 bank → conflict
  解决方案:
    1. Padding: 在每行末尾添加 8 字节 (2 个 bank) 偏移
    2. 转置布局: 改变数据分布避免冲突
    3. 向量化访问: 同一线程访问连续地址
```

### 18.6.2 共享内存分配策略

```python
# Triton 中控制共享内存使用
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(A, B, C, M, N, K, stride_am, stride_ak, ...):
    # Triton 自动管理共享内存分配
    # BLOCK_M × BLOCK_K × 2 bytes (f16) + BLOCK_N × BLOCK_K × 2 bytes
    # = 128 × 32 × 2 + 128 × 32 × 2 = 16KB
    ...
```

```
共享内存大小计算

对于 matmul kernel:
  A tile: BLOCK_M × BLOCK_K × sizeof(elem)
  B tile: BLOCK_N × BLOCK_K × sizeof(elem)
  Padding (可选): 8-16 bytes per row

示例配置:
  BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, elem=f16 (2 bytes)
  A tile: 128 × 32 × 2 = 8,192 bytes (8KB)
  B tile: 128 × 32 × 2 = 8,192 bytes (8KB)
  Padding: 128 × 2 × 8 = 2,048 bytes (2KB)
  总计: ~18KB (A100 有 192KB/SM, 足够)

  BLOCK_M=256, BLOCK_N=256, BLOCK_K=32, elem=f16
  A tile: 256 × 32 × 2 = 16,384 bytes (16KB)
  B tile: 256 × 32 × 2 = 16,384 bytes (16KB)
  总计: ~32KB (仍然足够)

  超出共享内存限制时:
  A100: 192KB/SM (可配置为 164KB shared + 28KB L1)
  H100: 228KB/SM (可配置为 228KB shared)
```

### 18.6.3 Bank Conflict 分析与避免

```
Bank Conflict 模式分析

共享内存: 32 banks, 每个 bank 4 bytes 宽

Case 1: 无 Bank Conflict
  线程 0 访问 bank 0, 线程 1 访问 bank 1, ...
  每个线程访问不同的 bank
  → 1 cycle 完成

Case 2: 2-way Bank Conflict
  线程 0, 16 访问同一 bank (相差 16 × 4 = 64 bytes)
  → 2 cycles 完成

Case 3: 4-way Bank Conflict
  线程 0, 8, 16, 24 访问同一 bank
  → 4 cycles 完成

避免 Bank Conflict 的方法:

方法 1: Padding
  原始布局 (每行 32×4 = 128 bytes):
  Bank: 0 1 2 ... 31 | 0 1 2 ... 31 | ...
  
  Padding 后 (每行 128 + 8 = 136 bytes):
  Bank: 0 1 2 ... 31 | 0 1 2 3 ... 27 | 28 29 30 31 0 1 ...
       └── row 0 ──┘  └── row 1 ──┘    └── row 2 ──┘
  每行偏移 8 bytes (2 banks), 消除冲突

方法 2: 向量化访问
  使用 v4.b32 加载, 每次加载 16 bytes (4 banks)
  4 个 bank 同时读取, 无冲突

方法 3: 数据重排
  重新排列数据的内存布局, 使同时访问的数据分布在不同 bank
```

---

## 18.7 Occupancy 优化

### 18.7.1 Occupancy 概念

```
Occupancy = 活跃 Warp 数 / SM 最大 Warp 数

A100 SM 资源限制:
  最大线程数: 2,048 threads/SM
  最大 Warp 数: 64 warps/SM
  最大 Shared Memory: 192 KB/SM
  最大寄存器: 65,536 registers/SM

Occupancy 的影响:
  100% Occupancy → 最大化延迟隐藏
  低 Occupancy → 更多寄存器可用于每线程, 更大 tile
```

### 18.7.2 寄存器数与 Occupancy 关系

```
寄存器使用 vs Occupancy (A100, SM80)

┌────────────────────┬─────────────────┬──────────────────┐
│ 每线程寄存器数      │ 每 SM 最大线程数 │ Occupancy        │
├────────────────────┼─────────────────┼──────────────────┤
│ 32                 │ 2,048           │ 100% (64 warps) │
│ 40                 │ 2,048           │ 100% (64 warps) │
│ 48                 │ 1,706           │ 83% (53 warps)  │
│ 56                 │ 1,463           │ 71% (46 warps)  │
│ 64                 │ 1,280           │ 62% (40 warps)  │
│ 72                 │ 1,137           │ 55% (36 warps)  │
│ 80                 │ 1,024           │ 50% (32 warps)  │
│ 96                 │ 853             │ 41% (27 warps)  │
│ 128                │ 640             │ 31% (20 warps)  │
│ 255                │ 320             │ 15% (10 warps)  │
└────────────────────┴─────────────────┴──────────────────┘

计算公式:
  max_warps = floor(65536 / (registers_per_thread × 32))
  occupancy = max_warps / 64
```

```
maxnreg PTX 限制

.maxnreg 64     // 限制每个线程最多使用 64 个寄存器
                 // 编译器会将溢出的寄存器 spill 到共享内存
                 // 提高 occupancy 但增加内存访问

使用场景:
  - 当寄存器压力是瓶颈时
  - 当 occupancy 太低导致延迟隐藏不足时
  - Triton autotuning 会自动测试不同配置

Trade-off:
  高 occupancy (低寄存器):
    + 更好延迟隐藏
    + 更高 GPU 利用率
    - 每线程寄存器少 → 更小 tile
    - 更多 shared memory spill

  低 occupancy (高寄存器):
    + 每线程更多寄存器 → 更大 tile
    + 减少 shared memory 使用
    - 延迟隐藏差
    - GPU 可能未充分利用
```

### 18.7.3 num_warps 参数影响

```python
# num_warps 控制每个 CTA 的 warp 数
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=32),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(...):
    pid = tl.program_id(0)
    ...
```

```
num_warps 与 Tile 大小的关系

num_warps = 4 (128 threads):
  每个线程负责 128×128 / 128 = 128 个元素
  寄存器需求: ~64-80 per thread
  Shared memory: 128×32×2 + 128×32×2 = 16KB
  → 适合小规模矩阵, 高 occupancy

num_warps = 8 (256 threads):
  每个线程负责 128×128 / 256 = 64 个元素
  寄存器需求: ~48-64 per thread
  Shared memory: 128×32×2 + 128×32×2 = 16KB
  → 中等规模矩阵, 平衡配置

num_warps = 32 (1024 threads):
  每个线程负责 256×256 / 1024 = 64 个元素
  寄存器需求: ~40-64 per thread
  Shared memory: 256×32×2 + 256×32×2 = 32KB
  → 大规模矩阵, 可能降低 occupancy

Autotuning 策略:
  Triton 会尝试多种 num_warps 值, 选择最快的配置
  影响因素:
    - 矩阵大小 (小矩阵需要更多 warp 并行)
    - 硬件资源 (寄存器、shared memory)
    - 计算密度 (更多 warp 更好隐藏延迟)
```

### 18.7.4 Triton Autotuning 中的资源管理

```python
# Autotuning 自动探索最优资源分配
@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8},
            num_warps=8,    # 控制 occupancy
            num_stages=3,   # 控制软件流水线深度
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 4},
            num_warps=4,
            num_stages=2,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(...):
    ...
```

```
Autotuning 搜索空间

┌──────────────┬──────────────────────────────────┐
│ 参数          │ 搜索范围                          │
├──────────────┼──────────────────────────────────┤
│ BLOCK_M      │ {64, 128, 256}                   │
│ BLOCK_N      │ {64, 128, 256}                   │
│ BLOCK_K      │ {16, 32, 64}                     │
│ num_warps    │ {4, 8, 16, 32}                   │
│ num_stages   │ {1, 2, 3, 4}                     │
│ GROUP_SIZE_M │ {1, 2, 4, 8, 16, 32}             │
│ matrix_instr_nonkdim │ {0, 1}                   │
│ allow_tf32   │ {True, False}                    │
└──────────────┴──────────────────────────────────┘

约束条件:
  - BLOCK_M × BLOCK_K × 2 + BLOCK_N × BLOCK_K × 2 ≤ Shared Memory
  - num_warps × 32 ≤ Max Threads per SM
  - BLOCK_M % (num_warps × 16) == 0 (MMA 维度对齐)
  - BLOCK_N % 8 == 0 (MMA N 维度对齐)
```

### 18.7.5 Shared Memory 容量与 Tile 大小

```
A100 (SM80) 资源配置方案

配置 1: 大共享内存 (训练)
  Shared Memory: 164 KB
  L1 Cache: 28 KB
  → 可容纳更大 tile, 适合大矩阵

配置 2: 大 L1 Cache (推理)
  Shared Memory: 100 KB
  L1 Cache: 128 KB (含 texture cache)
  → 更好缓存, 适合小矩阵

H100 (SM90):
  Shared Memory: 228 KB (可配置)
  更灵活的内存分配

Triton 中配置共享内存:
  在 kernel launch 时:
  kernel[grid](..., num_warps=8, num_stages=3)
  
  Triton 会自动:
  1. 计算所需 shared memory 大小
  2. 调用 cudaFuncSetAttribute 设置
  3. 根据硬件限制调整 tile 大小
```

---

## 18.8 Triton 与 cuBLAS 性能对比

### 18.8.1 cuBLAS 实现策略

```
cuBLAS Matmul 实现层次

cuBLAS 有多种内核实现, 根据矩阵大小和形状自动选择:

┌─────────────────────────────────────────────────────────────┐
│                      cuBLAS 策略选择                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  小矩阵 (M,N,K < 32):                                      │
│    → 使用 CUDA Core 实现, 避免 Tensor Core 启动开销         │
│                                                             │
│  中等矩阵:                                                   │
│    → Split-K: 将 K 维度分片, 并行计算后归约                  │
│    → Strided Batched: 批量矩阵乘法优化                      │
│                                                             │
│  大矩阵 (M,N ≥ 256):                                       │
│    → Tile-based MMA: 使用 Tensor Core + 分块策略            │
│    → 多级 tiling: CTA tile → Warp tile → MMA tile           │
│                                                             │
│  超大矩阵:                                                   │
│    → 多 CTA 协作: 每个 CTA 处理子块, atomicAdd 归约         │
│    → Pipeline: 软件流水线重叠计算和搬运                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 18.8.2 性能对比数据

```
Triton vs cuBLAS Matmul Performance (A100, FP16)

矩阵大小        cuBLAS (TFLOPS)  Triton (TFLOPS)  差距
──────────────────────────────────────────────────────
64×64×64         8.2              7.1             -13%
128×128×128      35.6             31.2            -12%
256×256×256      82.4             74.5            -10%
512×512×512      156.3            141.2           -10%
1024×1024×1024   231.8            210.5           -9%
2048×2048×2048   278.6            255.3           -8%
4096×4096×4096   302.1            278.9           -8%
8192×8192×8192   310.5            292.1           -6%

A100 峰值 FP16 Tensor Core: 312 TFLOPS

性能差距分析:
  1. 小矩阵: Triton 较差 (启动开销, tile 选择受限)
  2. 大矩阵: 差距缩小 (cuBLAS 的优化空间有限)
  3. 非方形矩阵: Triton 可能更好 (灵活 tile 选择)

cuBLAS 的优势:
  1. 更精细的手写汇编优化 (SASS 级别)
  2. 更成熟的 autotuning (搜索空间更大)
  3. 针对不同矩阵形状的专用内核
  4. 更好的 pipeline 调度

Triton 的优势:
  1. 编译时确定最优配置 (无运行时搜索)
  2. 用户可定制 kernel 逻辑
  3. 跨平台 (NVIDIA, AMD, Intel)
  4. 代码可读性和可维护性更好
```

### 18.8.3 Roofline 分析

```
A100 Roofline 模型

峰值算力 (FP16 Tensor Core): 312 TFLOPS
显存带宽 (HBM2e): 2,039 GB/s
L2 Cache 带宽: 12 TB/s (理论)

Roofline 等式:
  Performance = min(Peak Compute, Arithmetic Intensity × Memory Bandwidth)

  Arithmetic Intensity = FLOPs / Bytes = 2×M×N×K / (M×K×2 + K×N×2 + M×N×2)
  = 2×M×N×K / (2×M×K + 2×K×N + 2×M×N)

对于方形矩阵 M=N=K:
  AI = 2×N³ / (6×N²) = N/3

  N=128:  AI = 42.7 FLOPs/byte → Compute bound (312 TFLOPS)
  N=256:  AI = 85.3 FLOPs/byte → Compute bound
  N=1024: AI = 341 FLOPs/byte → Deeply compute bound

Roofline 图:

  Performance
  (TFLOPS)
    312 ────────────────────────────────────────── ← Peak Compute
         │                    ╱
         │                 ╱
         │              ╱
         │           ╱
         │        ╱
         │     ╱          ← Memory Bound Region
         │  ╱
         │╱
         └──────────────────────────────────────── AI (FLOPs/byte)
              43     85     170    341

  对于 N ≥ 128 的方形矩阵, Triton 通常能达到 compute-bound 区域
  对于非方形矩阵或小矩阵, 可能受限于 memory bandwidth
```

### 18.8.4 Triton 优化空间

```
Triton 可以进一步优化的方向

1. Hand-tuned SASS 优化
   cuBLAS 在 SASS (GPU assembly) 级别做了精细优化
   Triton 在 PTX 级别停止, 依赖 ptxas 进行寄存器分配和指令调度
   → 可以通过 LLVM 后端的 ptxas 调优来缩小差距

2. 更激进的 Pipeline
   cuBLAS 使用更长的软件流水线 (4-6 stage)
   Triton 默认 2-3 stage
   → 增加 num_stages 可以提高 throughput

3. 更灵活的 Tiling
   cuBLAS 针对特定矩阵形状有专用内核
   Triton 使用通用 tile 配置
   → 自定义 tile 大小可针对特定应用优化

4. 多 CTA 协作
   cuBLAS 使用多个 CTA 协作处理大矩阵
   Triton 的 CTA 级别归约较简单
   → 实现更复杂的 CTA 协作策略

5. 异步拷贝
   cuBLAS 大量使用 cp.async 进行异步数据搬运
   Triton 的异步拷贝支持在改进中
   → 更好的计算-搬运重叠

6. 分块矩阵乘法策略
   Split-K, Split-M/N 等策略
   Triton 支持 Split-K (via tl.program_id)
   → 但不如 cuBLAS 的策略丰富
```

---

## 18.9 调试与分析

### 18.9.1 PTX 编译调试

```bash
# 获取 Triton kernel 的 PTX 输出
# 方法 1: 使用 triton 保存中间结果
import triton
import triton.tools.driver as drv

# 启用 PTX 保存
@triton.jit
def my_kernel(...):
    ...
# 编译后在 _dump 目录查看 PTX 文件

# 方法 2: 使用 Triton 的 dump 功能
import os
os.environ['TRITON_PRINT_AUTOTUNING'] = '1'
os.environ['TRITON_CACHE_DIR'] = '/tmp/triton_cache'

# 查看编译的 PTX 和 cubin
ls /tmp/triton_cache/
```

```bash
# 使用 ptxas 手动编译和调试 PTX
ptxas -arch=sm_80 -v kernel.ptx -o kernel.cubin

# ptxas 输出信息:
# ptxas info: Used 128 registers, 16384 bytes smem, 360 bytes cmem[0]
# ptxas info: Spill details: 0 regs, 0 smem
# ptxas info: Function uses 0 bytes stack, 0 bytes cmem[2]

# 关键指标:
# registers: 每线程寄存器数 (影响 occupancy)
# smem: 共享内存使用量
# spill: 寄存器溢出到共享内存
```

### 18.9.2 NSight Compute 分析

```bash
# 使用 NSight Compute 分析 kernel 性能
ncu --set full -o report ./my_program

# 关键性能指标:
# ── Compute ──
# SM [cycles]                      : SM 活跃周期数
# Inst Executed                   : 执行的指令数
# Warp Instructions               : Warp 级指令数
# Avg. Threads Executed / Warp    : 平均每 warp 活跃线程数
#
# ── Memory ──
# DRAM Throughput                 : HBM 带宽使用率
# L1 Hit Rate                     : L1 缓存命中率
# L2 Hit Rate                     : L2 缓存命中率
# Shared Memory Throughput        : 共享内存带宽
# Shared Memory Bank Conflicts    : Bank Conflict 次数
#
# ── Occupancy ──
# Achieved Occupancy              : 实际达到的 Occupancy
# Theoretical Occupancy           : 理论最大 Occupancy
# Register Usage                  : 寄存器使用量
# Block Limit Registers           : 寄存器限制的最大 block 数
#
# ── Bottleneck ──
# Compute (SM) Throughput         : 计算吞吐量瓶颈
# Memory Throughput               : 内存吞吐量瓶颈
# Launch Overhead                 : Kernel 启动开销
```

```
NSight Compute 分析报告解读示例

═══════════════════════════════════════════════════════════════
Kernel: matmul_kernel
Grid: (128, 1, 1)   Block: (256, 1, 1)
═══════════════════════════════════════════════════════════════

Duration:              0.156 ms
Grid Size:             128
Block Size:            256

═══════════════════════════════════════════════════════════════
Section: Compute
═══════════════════════════════════════════════════════════════
  Compute (SM) Throughput:     78.2%
  FLOP Count FP32:             1.074 GFLOP
  FLOP Count FP16:             2.147 GFLOP
  FLOP Rate FP16:              13.8 TFLOPS

  → 13.8 / 312 = 4.4% (远低于峰值, 可能有优化空间)

═══════════════════════════════════════════════════════════════
Section: Memory
═══════════════════════════════════════════════════════════════
  DRAM Throughput:              45.3%
  DRAM Bytes:                   1.2 GB
  L2 Hit Rate:                  87.2%
  Shared Memory Bank Conflicts: 12,345

  → Bank Conflicts 存在, 可以通过 padding 优化

═══════════════════════════════════════════════════════════════
Section: Occupancy
═══════════════════════════════════════════════════════════════
  Achieved Occupancy:           82.5%
  Theoretical Occupancy:        100%
  Register Usage:               72 per thread
  Shared Memory Usage:          16,384 bytes

  → 寄存器使用 72, 理论 occupancy 55%, 实际达到 82.5%
  → 可能通过降低寄存器使用来提高 occupancy
```

### 18.9.3 Compute-Sanitizer 内存检查

```bash
# 使用 compute-sanitizer 检查内存错误
compute-sanitizer --tool memcheck ./my_program

# 输出示例:
========= Invalid __global__ read of size 4 bytes
=========     at 0x00000148 in matmul_kernel
=========     by thread (3,0,,0) in block (0,0,0)
=========     Address 0x7f1234567890 is out of bounds

# 常见错误类型:
# 1. 越界访问 (Out of bounds)
# 2. 未对齐访问 (Unaligned access)
# 3. 竞态条件 (Race condition)
# 4. 共享内存冲突 (Shared memory conflict)

# 使用 racecheck 检查竞态条件
compute-sanitizer --tool racecheck ./my_program

# 使用 synccheck 检查同步问题
compute-sanitizer --tool synccheck ./my_program

# 使用 memcheck 检查内存泄漏
compute-sanitizer --tool memcheck --leak-check full ./my_program
```

### 18.9.4 PTX 级别问题定位

```
常见 PTX 级别问题

问题 1: 寄存器溢出 (Register Spill)
  症状: ptxas 输出 "Spill details: N regs, M smem"
  原因: 寄存器压力过大, 部分变量溢出到共享内存
  解决:
    - 减小 tile 大小
    - 减少 num_warps
    - 使用 #unroll 减少临时变量

问题 2: Bank Conflict
  症状: Shared Memory Throughput 低
  原因: 多个线程同时访问同一 bank
  解决:
    - 添加 padding (在 Triton 中通过 @triton.autotune 测试)
    - 改变数据布局
    - 使用向量化访问

问题 3: 指令延迟未隐藏
  症状: Compute throughput 低
  原因: Occupancy 不足, 或指令并行度不够
  解决:
    - 增加 num_warps
    - 减少每线程工作量
    - 添加 #unroll 提高 ILP

问题 4: Memory Bound (内存瓶颈)
  症状: DRAM Throughput 高, Compute throughput 低
  原因: 矩阵太小或 AI 太低
  解决:
    - 增大 tile 大小
    - 使用更小的数据类型 (FP16 → INT8)
    - 增加计算密度 (融合更多操作)

问题 5: 不正确的 MMA 对齐
  症状: 非法指令错误
  原因: 寄存器或共享内存地址未按 MMA 要求对齐
  解决:
    - 确保 BLOCK_M % 16 == 0 (MMA M 维度)
    - 确保 BLOCK_N % 8 == 0 (MMA N 维度)
    - 确保 BLOCK_K % 8 == 0 (MMA K 维度)
```

### 18.9.5 Triton 内置调试工具

```python
# Triton 内置的调试功能

# 1. 打印编译的 PTX
import triton
triton.rcParams["debug"] = True

# 2. 打印 autotuning 信息
import os
os.environ['TRITON_PRINT_AUTOTUNING'] = '1'

# 3. 检查生成的汇编
@triton.jit
def my_kernel(...):
    # 使用 tl.static_assert 进行编译时检查
    tl.static_assert(BLOCK_K % 8 == 0, "BLOCK_K must be multiple of 8 for MMA")
    ...

# 4. 打印 kernel 参数
@triton.autotune(configs=[...], key=['M'])
@triton.jit
def my_kernel(..., M, BLOCK_M: tl.constexpr):
    print(f"BLOCK_M = {BLOCK_M}")  # 编译时打印
    ...

# 5. 使用 Triton 的 profiling 工具
import triton.tools.driver as drv
drv.init()
# ... launch kernel ...
drv.synchronize()
```

---

## 18.10 NVIDIA 后端源码走读

### 18.10.1 核心文件概览

```
NVIDIA 后端源码走读路线

1. 转换入口
   lib/Conversion/TritonGPUToLLVMPass.cpp
   → 主转换 Pass, 调度所有 Pattern

2. 操作转换
   lib/Conversion/TritonGPUToLLVM/   (NVIDIA 子目录)
   ├── DotOpToLLVM.cpp        → DotOp → MMA
   ├── LoadOpToLLVM.cpp       → LoadOp → GEP + load
   ├── StoreOpToLLVM.cpp      → StoreOp → store
   ├── ReduceOpToLLVM.cpp     → ReduceOp → shfl_sync
   ├── ConvertLayoutToLLVM.cpp → convert_layout → shuffle
   └── ViewOpsToLLVM.cpp      → view, expand_dims 等

3. NVGPU Dialect
   third_party/nvidia/lib/Dialect/NVGPU/
   ├── NVGPUOps.cpp           → shfl, ballot, mma 等操作
   └── NVGPUDialect.cpp       → Dialect 注册

4. NVVM 转换
   third_party/nvidia/lib/Dialect/NVVMToLLVM/
   └── NVVMToLLVM.cpp         → NVVM intrinsic → PTX

5. LLVM 翻译
   lib/Target/LLVMIR/LLVMIRTranslation.cpp
   → LLVM Dialect → LLVM IR
```

### 18.10.2 DotOp 转换源码分析

```cpp
// DotOpToLLVM.cpp 核心逻辑

LogicalResult convertDot(TritonGPUToLLVMTypeConverter *typeConverter,
                         triton::gpu::DotOp op, ...) {

  // 1. 获取 encoding
  auto aEncoding = op.getA().getType().getEncoding();
  auto bEncoding = op.getB().getType().getEncoding();
  auto dEncoding = op.getD().getType().getEncoding();

  // 2. 检查是否是 MMA encoding
  if (!dEncoding.isa<MmaEncodingAttr>())
    return failure();

  auto mmaEncoding = dEncoding.cast<MmaEncodingAttr>();

  // 3. 获取 warpsPerCta
  auto warpsPerCta = mmaEncoding.getWarpsPerCTA();

  // 4. 计算每个 warp 的 MMA tiles
  int mmaShapeM = 16, mmaShapeN = 8, mmaShapeK = 8;
  int numMmaM = BLOCK_M / mmaShapeM;
  int numMmaN = BLOCK_N / mmaShapeN;

  // 5. 为每个 warp 生成 MMA 操作
  for (int warp = 0; warp < numWarps; ++warp) {
    for (int i = 0; i < numMmaM; ++i) {
      for (int j = 0; j < numMmaN; ++j) {
        // 提取 A 矩阵片段
        auto aFragments = extractMmaFragmentsA(op.getA(), warp, i, k);
        // 提取 B 矩阵片段
        auto bFragments = extractMmaFragmentsB(op.getB(), warp, j, k);

        // 组装 v4f16 向量
        Value aVec = packFragments(aFragments, rewriter);
        Value bVec = packFragments(bFragments, rewriter);

        // 调用 NVVM MMA intrinsic
        auto mmaOp = rewriter.create<NVVM::MmaSyncOp>(
            loc, dType, aVec, bVec, cVec,
            /*aCol=*/false, /*bCol=*/true,
            mmaShapeM, mmaShapeN, mmaShapeK);

        // 更新累加器
        cVec = mmaOp.getResult();
      }
    }
  }
}
```

### 18.10.3 ConvertLayout 转换源码分析

```cpp
// ConvertLayoutToLLVM.cpp 核心逻辑
// convert_layout 用于在不同 Encoding 之间转换数据布局

LogicalResult convertLayout(triton::gpu::ConvertLayoutOp op, ...) {
  auto srcEncoding = op.getSrc().getType().getEncoding();
  auto dstEncoding = op.getResult().getType().getEncoding();

  // Case 1: Blocked → Blocked (简单的 shuffle)
  if (srcEncoding.isa<BlockedEncodingAttr>() &&
      dstEncoding.isa<BlockedEncodingAttr>()) {
    // 使用 warp shuffle 进行数据交换
    return convertBlockedToBlocked(op, rewriter);
  }

  // Case 2: Blocked → Shared (存储到共享内存)
  if (srcEncoding.isa<BlockedEncodingAttr>() &&
      dstEncoding.isa<SharedEncodingAttr>()) {
    return convertBlockedToShared(op, rewriter);
  }

  // Case 3: Shared → DotOperand (从共享内存加载到寄存器)
  if (srcEncoding.isa<SharedEncodingAttr>() &&
      dstEncoding.isa<DotOperandEncodingAttr>()) {
    // 使用 ldmatrix 加载
    return convertSharedToDotOperand(op, rewriter);
  }

  // Case 4: DotOperand → Mma (直接传递)
  if (srcEncoding.isa<DotOperandEncodingAttr>() &&
      dstEncoding.isa<MmaEncodingAttr>()) {
    // 不需要额外操作, 数据已在寄存器中
    rewriter.replaceOp(op, op.getSrc());
    return success();
  }

  return failure();
}
```

---

## 18.11 Hopper (SM90) 新特性

### 18.11.1 Hopper 架构新功能

```
Hopper (H100) 新特性

1. TMA (Tensor Memory Accelerator)
   - 硬件加速的张量数据搬运
   - 异步、多阶段的内存访问
   - 与 shared memory 直接交互

2. TCG (Tensor Core Generation)
   - 更大维度的 MMA: m16n8k32 (FP8)
   - 支持 FP8 数据类型
   - 更高的峰值算力: 3958 TFLOPS (FP8)

3. 异步 warpgroup 执行
   - warpgroup = 4 warps = 128 threads
   - 硬件调度的异步执行
   - 更好的资源利用

4. 更大的 Shared Memory
   - 最大 228 KB/SM (可配置)
   - 支持更长的 pipeline 深度
```

```
Hopper TMA 操作示例

Triton 中的 TMA 支持:
  tl.load(pointer, boundary_check=False, padding_option=0,
          cache_modifier='ca', volatile=False)

Hopper PTX TMA 指令:
  cp.async.bulk.tensor.4d.shared.global.m128.n8
    {tensor_desc}, [smem_addr], [global_addr];
  
  // 异步将 128×8 的张量从全局内存搬到共享内存
  // 硬件自动处理地址计算和边界检查
```

### 18.11.2 Triton 对 Hopper 的支持

```python
# Triton Hopper 支持 (SM90)
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},
            num_warps=8, num_stages=4,
        ),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_hopper_kernel(A, B, C, M, N, K, ...):
    # Hopper 优化策略:
    # 1. 更长的 pipeline (num_stages=4)
    # 2. 更大的 BLOCK_K (64) 增加计算密度
    # 3. 使用 TMA 异步搬运
    ...
```

```
Hopper vs Ampere 性能对比

矩阵大小        A100 (TFLOPS)  H100 (TFLOPS)  提升
─────────────────────────────────────────────────────
1024×1024×1024   231.8          456.2          97%
2048×2048×2048   278.6          712.5          156%
4096×4096×4096   302.1          1,023.4        239%
8192×8192×8192   310.5          1,456.7        369%

H100 FP8 峰值: 3,958 TFLOPS
H100 BF16 峰值: 1,979 TFLOPS
A100 FP16 峰值: 312 TFLOPS

提升主要来自:
  1. 更高时钟频率 (1.83 GHz → 1.41 GHz × 更多 SM)
  2. FP8 支持 (2× 吞吐)
  3. 更大的 shared memory (更长 pipeline)
  4. TMA 硬件加速 (更好的数据搬运)
  5. Warpgroup 异步执行 (更好的并行度)
```

---

## 18.12 高级主题

### 18.12.1 异步拷贝与 Pipeline

```
Triton 软件流水线在 NVIDIA 后端的实现

CP (Copy Pipeline) 概念:
  Stage 0: 从全局内存加载 tile 0 → 共享内存 A
  Stage 1: 计算 tile 0, 同时加载 tile 1 → 共享内存 B
  Stage 2: 计算 tile 1, 同时加载 tile 2 → 共享内存 A
  ...

Triton PTX 实现:
  // 初始化 mbarrier
  mbarrier.init [mbar_addr], num_threads;

  // 异步拷贝 (cp.async)
  cp.async.ca.shared.global [smem_dst], [global_src], 16;
  cp.async.commit_group;  // 提交异步拷贝组

  // 等待拷贝完成
  cp.async.wait_group 0;  // 等待所有异步拷贝完成
  mbarrier.try_wait.parity [mbar_addr], ...;

  // 使用数据
  ldmatrix.sync.aligned.m8n8.x4.shared.b16 ...;
  mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 ...;
```

```
Pipeline 深度与性能

num_stages = 1 (无流水线):
  Load → Compute → Load → Compute → ...
  延迟: Load_time + Compute_time
  GPU 利用率低

num_stages = 2 (双缓冲):
  Load0 → Load1+Compute0 → Load2+Compute1 → ...
  延迟: max(Load_time, Compute_time)
  GPU 利用率提高

num_stages = 3 (三缓冲):
  Load0 → Load1+Compute0 → Load2+Compute1+... → ...
  延迟: max(Load_time, Compute_time)
  更好的重叠

A100 推荐 num_stages:
  - 小 tile (< 64KB shared memory): num_stages = 4
  - 中等 tile: num_stages = 3
  - 大 tile (接近 shared memory 限制): num_stages = 2
```

### 18.12.2 CTA 协作策略

```
多 CTA 协作 Matmul 策略

Strategy 1: 简单 CTA 映射 (Triton 默认)
  每个 CTA 处理一个 tile
  CTA 之间无协作
  Grid: ceil(M/BLOCK_M) × ceil(N/BLOCK_N)

Strategy 2: Split-K
  将 K 维度分片到多个 CTA
  每个 CTA 计算部分和
  使用 atomicAdd 或 reduction 归约
  Grid: ceil(M/BLOCK_M) × ceil(N/BLOCK_N) × SplitFactor

Strategy 3: CTA Cluster (Hopper)
  一组 CTA 组成 cluster
  共享 shared memory
  通过 TMA 协作搬运数据
  Grid: cluster 数量

Triton Split-K 示例:
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)
  pid_k = tl.program_id(2)  # Split-K 维度
  
  # 每个 CTA 处理 K 维度的一个分片
  offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
  # ... 计算部分和 ...
  
  # atomicAdd 归约到全局内存
  tl.atomic_add(c_ptr + offs, acc)
```

### 18.12.3 自定义 PTX 注入

Triton 允许用户通过 `tl.inline_asm_elementwise` 直接注入 PTX 代码：

```python
@triton.jit
def custom_mma_kernel(A, B, C, M, N, K, ...):
    # 使用自定义 PTX 代码
    # 例如: 使用特殊的 warp shuffle 实现自定义归约
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a = tl.load(A + ...)
        b = tl.load(B + ...)
        
        # 自定义 PTX: 使用 ldmatrix 加载 A
        a_packed = tl.inline_asm_elementwise(
            asm="$0 = ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$1, $2, $3, $4};",
            constraints=("r,r,r,r,r"),  # 输出, 输入
            args=[a], ...)
        
        acc += tl.dot(a_packed, b)
    
    tl.store(C + ..., acc)
```

```ptx
# 自定义 PTX 的典型用例

# 1. 使用 PTX 实现 Triton 尚未支持的操作
# 2. 优化特定的内存访问模式
# 3. 使用硬件特定的指令 (如 Hopper 的 TMA)
# 4. 实现自定义的数据类型转换

# 注意事项:
# - PTX 代码依赖于特定 GPU 架构 (sm_80, sm_90)
# - 需要正确处理寄存器分配
# - 可能破坏 Triton 的优化 (如软件流水线)
# - 仅在必要时使用, 优先使用 Triton 原生操作
```

---

## 本章小结

本章深入探讨了 Triton NVIDIA 后端的完整代码生成路径。关键要点如下：

1. **编译路径**: Triton 的编译路径为 `Triton IR → TritonGPU IR → LLVM Dialect → NVVM Dialect → PTX → cubin`。每一层逐步增加硬件信息，最终生成可在 NVIDIA GPU 上执行的二进制代码。

2. **Tensor Core 映射**: `triton_gpu.dot` 操作通过 `DotOpConversion` 被降低为 `mma.sync.aligned.m16n8k8` 等 Tensor Core 指令。每个 warp 的 32 个线程协同执行一次 MMA，每个线程持有 A、B 矩阵的一部分。

3. **ldmatrix 指令**: 从共享内存加载 Tensor Core 数据时，使用 `ldmatrix.sync.aligned.m8n8.x4` 可以高效地完成数据重排和转置，比手动加载快 2-4 倍。

4. **Warp 级操作**: `__shfl_sync` 和 `__ballot_sync` 是实现快速归约和投票的关键原语。Triton 的 `reduce` 操作被降低为 butterfly reduction，只需 `log2(32)` 步即可完成 warp 内归约。

5. **共享内存管理**: 共享内存通过 `.shared .align 16` 声明，需要避免 bank conflict。Padding 和向量化访问是两种有效的优化策略。

6. **Occupancy 优化**: 寄存器数和共享内存使用量共同决定了 occupancy。Triton 的 autotuning 会自动搜索最优的 `num_warps` 和 tile 大小配置。

7. **与 cuBLAS 的差距**: Triton 在大矩阵上通常能达到 cuBLAS 90-95% 的性能，主要差距在于 cuBLAS 的手写 SASS 优化和更成熟的 autotuning。

8. **调试工具链**: `ptxas`、`nsight compute`、`compute-sanitizer` 是 NVIDIA 后端调试的核心工具。理解 PTX 指令和寄存器使用情况对于性能优化至关重要。

9. **Hopper 新特性**: SM90 引入了 TMA、更大的 shared memory 和 warpgroup 异步执行，Triton 正在积极支持这些新特性。

---

## 思考题

1. **Tensor Core 数据分布**: 在 `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` 中，一个 warp 的 32 个线程如何分配 A 矩阵（16×8, row-major）的 128 个元素？请画出线程到元素的映射表。

2. **Bank Conflict 分析**: 假设共享内存有 32 个 bank，每个 bank 4 bytes。一个 warp 的 32 个线程同时读取共享内存地址 `base + thread_id × 4`，是否会产生 bank conflict？如果地址改为 `base + thread_id × 16` 呢？

3. **Occupancy Trade-off**: 对于一个 1024×1024×1024 的 FP16 matmul，配置 A 使用 64 寄存器/线程（occupancy 62%），配置 B 使用 48 寄存器/线程（occupancy 83%）。请分析哪种配置可能更快，并解释原因。

4. **Pipeline 深度**: Triton 的 `num_stages` 参数控制软件流水线深度。对于 A100（192KB shared memory），为什么不能无限增加 `num_stages`？请从 shared memory 容量和延迟隐藏两个角度分析。

5. **Split-K 策略**: 当矩阵形状为 M=1024, N=1024, K=10240 时，使用 Split-K 策略（将 K 维度分片到多个 CTA）比简单 CTA 映射有什么优势？Split-K 的缺点是什么？

6. **Hopper TMA**: 解释 Hopper 的 Tensor Memory Accelerator (TMA) 如何改进数据搬运效率。与传统的 `cp.async` 相比，TMA 的主要优势是什么？

7. **跨平台对比**: Triton 的 NVIDIA 后端使用 `mma.sync` 指令，AMD 后端使用 MFMA 指令。这两种指令在数据类型支持、矩阵维度和线程分配上有什么异同？

8. **PTX vs SASS**: 为什么 cuBLAS 的性能通常优于 Triton？从 PTX 到 SASS 的编译过程（由 ptxas 完成）中，哪些优化是 Triton 无法控制的？Triton 未来如何缩小这个差距？

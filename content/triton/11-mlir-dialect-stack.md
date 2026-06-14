---
title: "Chapter 11: MLIR 方言栈与 Triton Dialect"
description: "深入理解 MLIR 编译框架的设计哲学、方言（Dialect）层次结构、Triton 的多层方言栈（tt → ttg → triton_gpu → llvm）、Type/Attribute 系统、Dialect Conversion 以及 triton-opt/triton-translate 工具链"
date: "2026-06-11"
---

# Chapter 11: MLIR 方言栈与 Triton Dialect

> **学习目标**：
> - 理解 MLIR 的设计哲学：多层方言（Multi-Level Dialect）、渐进式 lowering、SSA 形式
> - 掌握 MLIR 的四个核心概念：Dialect、Operation、Type、Attribute
> - 了解 Triton 的四层方言栈架构：tt → ttg → triton_gpu → llvm，每一层的职责与边界
> - 熟悉 Triton Dialect (tt) 的 ODS 定义和核心 Operation 类型
> - 理解 TritonGPU Dialect (ttg) 中的 LayoutEncoding 机制：BlockedEncoding、SharedEncoding、DotOperandEncoding、MmaEncoding
> - 掌握 Triton 的类型系统：tt.ptr<T>、tensor<NxT>、ttg.memdesc<T> 及类型转换规则
> - 理解 Attribute 系统中 layout attribute 的定义、语义与传播规则
> - 了解 Dialect Conversion 的工作原理：tt → ttg 的布局推断、ttg → llvm 的指令选择
> - 能够使用 triton-opt、triton-translate 工具查看和分析 MLIR IR

---

## 11.1 MLIR 基础

### 11.1.1 MLIR 的设计哲学

MLIR（Multi-Level Intermediate Representation）是 LLVM 项目中的新一代编译器基础设施。与传统的单一 IR 设计不同，MLIR 的核心创新在于**可扩展的方言（Dialect）系统**——允许在同一编译流程中使用多个不同抽象层次的中间表示。

传统编译器面临一个根本矛盾：

```
┌─────────────────────────────────────────────────────────────────┐
│                  传统编译器的抽象鸿沟                              │
│                                                                 │
│   高级语言（Python / Julia / Swift）                              │
│       │                                                         │
│       │  ← 巨大的语义鸿沟（语义丢失严重）                          │
│       │                                                         │
│       ▼                                                         │
│   LLVM IR（低级 SSA，面向通用 CPU/GPU）                           │
│       │                                                         │
│       │                                                         │
│       ▼                                                         │
│   机器码                                                          │
└─────────────────────────────────────────────────────────────────┘
```

问题在于：从高级语言到 LLVM IR 的转换是一步完成的"大跳"（big step），在这个过程中大量高层语义信息丢失。例如：

- 矩阵乘法 `C = A @ B` 被展开为一系列 load/store/multiply/add
- 循环结构被展平为基本块和分支
- 并行语义（如 GPU kernel 的 thread block）消失

MLIR 通过**渐进式 lowering**（Progressive Lowering）解决这个问题：

```
┌─────────────────────────────────────────────────────────────────┐
│                  MLIR 的渐进式 Lowering                           │
│                                                                 │
│   高层方言（Tensor / Linalg / Triton）                            │
│       │  ──── Dialect Conversion ────                            │
│       ▼                                                         │
│   中层方言（Affine / SCF / TritonGPU）                           │
│       │  ──── Dialect Conversion ────                            │
│       ▼                                                         │
│   低层方言（LLVM Dialect / NVVM / ROCDL）                        │
│       │  ──── LLVM Translation ────                              │
│       ▼                                                         │
│   LLVM IR → 机器码                                                │
└─────────────────────────────────────────────────────────────────┘
```

每一层 lowering 都是**局部的、可组合的**转换，保留尽可能多的语义信息，直到最后一步才丢弃。

### 11.1.2 核心概念：Dialect

**Dialect**（方言）是 MLIR 中的核心组织单元。每个 Dialect 定义了一组相关的 Operation、Type 和 Attribute。可以将 Dialect 理解为一个"命名空间"或"方言"——它为特定领域的操作提供了词汇表。

```
// MLIR 中常见的 Dialect 及其领域
┌──────────────────┬──────────────────────────────────┐
│ Dialect           │ 职责                              │
├──────────────────┼──────────────────────────────────┤
│ func             │ 函数定义与调用                      │
│ arith            │ 算术运算（add, mul, cmp 等）        │
│ memref           │ 内存引用（load, store, alloc）      │
│ tensor           │ 张量操作（extract_slice 等）        │
│ linalg           │ 线性代数（matmul, conv2d 等）       │
│ scf              │ 结构化控制流（for, while, if）      │
│ affine           │ 仿射循环与多面体优化                │
│ llvm             │ LLVM IR 的镜像                     │
│ gpu              │ GPU 操作（launch, barrier）        │
│ tt (triton)      │ Triton 核心张量操作                │
│ ttg (triton_gpu) │ Triton GPU 内存布局                │
└──────────────────┴──────────────────────────────────┘
```

### 11.1.3 核心概念：Operation

**Operation** 是 MLIR 中的基本计算单元。每个 Operation 有一个名称、零或多个操作数（operand）、零或多个结果（result）、可选的属性（attribute）和可选的区域（region）。

```
// Operation 的通用语法格式
%result = "dialect.operation_name"(%operand0, %operand1) {
  attr_name = value
} : (type_of_operand0, type_of_operand0) -> type_of_result

// 简化的通用语法（更常见）
%result = dialect.operation_name %operand0, %operand1 {
  attr_name = value
} : (type0, type1) -> result_type
```

让我们通过一个具体的例子来理解：

```mlir
// Triton Dialect 中的一个 load 操作
%ptrs = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256x!tt.ptr<f32>>
%mask = arith.cmpi slt, %ptrs, %bound : tensor<256xi32>
%values = tt.load %ptrs, %mask {
  cache = 1 : i32,
  evict = 1 : i32,
  isVolatile = false
} : tensor<256xf32>
```

在这个例子中：
- `%ptrs`, `%mask`, `%values` 是 SSA 值（Value）
- `tt.make_range`, `arith.cmpi`, `tt.load` 是 Operation
- `cache = 1 : i32` 等是 Attribute
- `tensor<256xf32>` 等是 Type

### 11.1.4 核心概念：Type 与 Attribute

**Type** 描述了值的类型信息。MLIR 中的 Type 系统是可扩展的——每个 Dialect 可以定义自己的类型。

```
// MLIR 内置类型
i32, i64, f16, f32, f64        // 标量类型
tensor<128x64xf16>              // 张量类型
memref<128x64xf32>              // 内存引用类型

// Triton 方言定义的类型
!tt.ptr<f32>                    // Triton 指针类型
!tt.ptr<tensor<128x64xf16>>     // 块指针类型
tensor<128x64x!tt.ptr<f32>>     // 指针张量

// TritonGPU 方言定义的类型
!ttg.memdesc<128x64xf16, ...>   // 共享内存描述符
```

**Attribute** 是编译时已知的常量数据，用于配置 Operation 的行为。与 Type 类似，Attribute 也是可扩展的。

```mlir
// 内置 Attribute
42 : i32                        // 整数常量
3.14 : f32                      // 浮点常量
[dense<1.0> : tensor<4xf32>]    // Dense 元素属性

// TritonGPU Layout Attribute（后面详细讲解）
#ttg.blocked<{sizePerThread = [1, 4],
              threadsPerWarp = [8, 4],
              warpsPerCTA = [4, 1],
              order = [1, 0]}>
```

### 11.1.5 MLIR 的 SSA 形式

MLIR 采用**静态单赋值**（Static Single Assignment, SSA）形式。每个值只能被定义一次，但可以被使用多次。这使得数据流分析变得简单而高效。

```mlir
// SSA 形式的例子
func.func @example(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
  // 每个 %name 只定义一次
  %0 = arith.addf %arg0, %arg1 : tensor<128xf32>
  %1 = arith.mulf %0, %0 : tensor<128xf32>      // %0 被使用两次
  %2 = math.exp %1 : tensor<128xf32>
  return %2 : tensor<128xf32>
}
```

对于控制流中的多定义值（如循环变量），MLIR 使用 Block Argument 而非 Phi 节点：

```mlir
// 循环变量通过 block argument 传递
%init = arith.constant 0.0 : f32
%lb = arith.constant 0 : index
%ub = arith.constant 128 : index
%step = arith.constant 1 : index

%result = scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> f32 {
  %val = memref.load %buf[%i] : memref<128xf32>
  %new_acc = arith.addf %acc, %val : f32
  scf.yield %new_acc : f32    // 将 %new_acc 传递给下一次迭代的 %acc
}
```

---

## 11.2 Triton 方言栈全景

### 11.2.1 为什么 Triton 需要多层方言

Triton 的编译过程需要将高级的张量操作逐步降低到具体的 GPU 指令。这个过程不可能一步完成——不同层次需要不同的抽象：

```
┌─────────────────────────────────────────────────────────────────┐
│                  Triton 编译的抽象层次                            │
│                                                                 │
│   Python (@tl.jit)                                              │
│     tl.load, tl.store, tl.dot, tl.where ...                    │
│       │                                                         │
│       │  ──── AST → IR (Triton IR Builder) ────                  │
│       ▼                                                         │
│   tt (Triton Dialect)     ← "做什么"：张量级操作                  │
│       │                                                         │
│       │  ──── Pass: tritongpu-coalesce ────                      │
│       │  ──── Pass: tritongpu-remove-layout-conversions ────     │
│       │  ──── Pass: tritongpu-materialize ────                   │
│       ▼                                                         │
│   ttg (TritonGPU Dialect) ← "怎么放"：内存布局与并行策略          │
│       │                                                         │
│       │  ──── Pass: tritongpu-to-llvm ────                       │
│       ▼                                                         │
│   llvm (LLVM Dialect)     ← "怎么做"：具体指令选择               │
│       │                                                         │
│       │  ──── LLVM Translation ────                              │
│       ▼                                                         │
│   LLVM IR → PTX / ROCm → GPU 机器码                              │
└─────────────────────────────────────────────────────────────────┘
```

### 11.2.2 四层方言栈详解

Triton 的方言栈可以分为四个主要层次：

```
┌─────────────────────────────────────────────────────────────────┐
│                     Triton 方言栈                                │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  tt (Triton Dialect)                                      │  │
│  │  ─────────────────────────────────────────────────────── │  │
│  │  • 硬件无关的张量操作                                      │  │
│  │  • tt.load, tt.store, tt.dot, tt.expand_dims, ...        │  │
│  │  • 用户直接编写的操作的直接映射                              │  │
│  │  • 类型: tensor<NxT>, !tt.ptr<T>                         │  │
│  └───────────────────────┬───────────────────────────────────┘  │
│                          │ tritongpu-coalesce                   │
│                          │ tritongpu-materialize                │
│                          ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  ttg (TritonGPU Dialect)                                  │  │
│  │  ─────────────────────────────────────────────────────── │  │
│  │  • GPU 内存布局与数据分块                                   │  │
│  │  • ttg.local_load, ttg.local_store, ttg.global_load      │  │
│  │  • Layout Encoding: Blocked, Shared, DotOperand, Mma     │  │
│  │  • 类型: !ttg.memdesc<T, encoding>                       │  │
│  └───────────────────────┬───────────────────────────────────┘  │
│                          │ tritongpu-to-llvm                    │
│                          ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  triton_gpu (硬件相关方言)                                 │  │
│  │  ─────────────────────────────────────────────────────── │  │
│  │  • 具体硬件指令选择                                        │  │
│  │  • MMA (Tensor Core) 指令映射                              │  │
│  │  • Shared Memory 地址计算                                  │  │
│  └───────────────────────┬───────────────────────────────────┘  │
│                          │                                      │
│                          ▼                                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  llvm (LLVM Dialect)                                      │  │
│  │  ─────────────────────────────────────────────────────── │  │
│  │  • LLVM IR 的 MLIR 镜像                                    │  │
│  │  • llvm.load, llvm.store, llvm.inline_asm                │  │
│  │  • 最终通过 LLVM Translation 生成 LLVM IR                  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 11.2.3 各层职责对比

| 层次 | Dialect | 关注点 | 核心问题 | 类比 |
|------|---------|--------|---------|------|
| L0 | `tt` | 算法语义 | "做什么"（What） | NumPy/PyTorch API |
| L1 | `ttg` | 内存布局 | "怎么放"（Where） | CUDA 的 thread/block 映射 |
| L2 | `triton_gpu` | 硬件指令 | "怎么算"（How） | PTX/SASS 指令 |
| L3 | `llvm` | 机器表示 | "怎么跑"（Run） | LLVM IR |

### 11.2.4 一个完整的 Lowering 示例

让我们跟踪一个简单的向量加法从 tt 到 llvm 的完整 lowering 过程：

**原始 Python 代码：**

```python
@triton.jit
def add_kernel(X, Y, Z, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    z = x + y
    tl.store(Z + offsets, z, mask=mask)
```

**L0 — tt 方言（Triton Dialect）：**

```mlir
// 这是 Triton 编译器在 AST-to-IR 阶段生成的 tt 方言
// 关注点：纯张量语义，没有内存布局信息
func.func @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>,
                       %arg2: !tt.ptr<f32>, %arg3: i32) {
  // 获取 program_id
  %pid = tt.get_program_id x : i32

  // 计算偏移量: pid * BLOCK +.arange(0, BLOCK)
  %cst_block = arith.constant 256 : i32
  %pid_offset = arith.muli %pid, %cst_block : i32
  %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
  %offsets = arith.addi %pid_offset, %range : tensor<256xi32>

  // 构造 mask: offsets < N
  %mask = arith.cmpi slt, %offsets, %arg3 : tensor<256xi32>

  // 构造指针张量
  %x_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %y_ptrs = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %z_ptrs = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %x_off = tt.addptr %x_ptrs, %offsets : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  %y_off = tt.addptr %y_ptrs, %offsets : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  %z_off = tt.addptr %z_ptrs, %offsets : tensor<256x!tt.ptr<f32>>, tensor<256xi32>

  // Load（张量级操作，不涉及具体的内存层次）
  %x = tt.load %x_off, %mask {cache = 1 : i32, evict = 1 : i32, isVolatile = false}
       : tensor<256xf32>
  %y = tt.load %y_off, %mask {cache = 1 : i32, evict = 1 : i32, isVolatile = false}
       : tensor<256xf32>

  // 向量加法（张量级操作）
  %z = arith.addf %x, %y : tensor<256xf32>

  // Store
  tt.store %z_off, %z, %mask {cache = 1 : i32, evict = 1 : i32}
    : tensor<256xf32>

  tt.return
}
```

**L1 — ttg 方言（TritonGPU Dialect）：**

```mlir
// 经过 tritongpu-coalesce 和 tritongpu-materialize pass 后
// 关注点：每个值都有了明确的内存布局 Encoding
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32],
                         warpsPerCTA = [8], order = [0]}>
#shared = #ttg.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

func.func @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>,
                       %arg2: !tt.ptr<f32>, %arg3: i32) {
  %pid = tt.get_program_id x : i32

  // 注意：tensor 类型现在带上了 #blocked 布局
  // tensor<256xf32, #blocked> 表示数据按 blocked 方式分布在各线程上
  %cst_block = arith.constant 256 : i32
  %pid_offset = arith.muli %pid, %cst_block : i32
  %range = tt.make_range {end = 256 : i32, start = 0 : i32}
           : tensor<256xi32, #blocked>
  %offsets = arith.addi %pid_offset, %range : tensor<256xi32, #blocked>

  %mask = arith.cmpi slt, %offsets, %arg3 : tensor<256xi32, #blocked>

  %x_ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
  %y_ptrs = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
  %z_ptrs = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
  %x_off = tt.addptr %x_ptrs, %offsets
           : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
  %y_off = tt.addptr %y_ptrs, %offsets
           : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
  %z_off = tt.addptr %z_ptrs, %offsets
           : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>

  // ttg.global_load 将全局内存加载转换为带布局的形式
  %x = ttg.global_load %x_off, %mask
       : tensor<256xf32, #blocked>
  %y = ttg.global_load %y_off, %mask
       : tensor<256xf32, #blocked>

  %z = arith.addf %x, %y : tensor<256xf32, #blocked>

  ttg.global_store %z_off, %z, %mask
    : tensor<256xf32, #blocked>

  tt.return
}
```

**L2+L3 — llvm 方言（LLVM Dialect）：**

```mlir
// 最终 lowered 到 LLVM Dialect
// 所有张量操作被展开为标量的 load/store/compute
// 注意：tensor<256xf32, #blocked> 被展开为多个 i32/f32 标量操作
// 每个线程处理 sizePerThread[0]=4 个元素

func.func @add_kernel(%arg0: !llvm.ptr, %arg1: !llvm.ptr,
                       %arg2: !llvm.ptr, %arg3: i32) {
  // program_id → nvvm.read.ptx.sreg.ctaid.x
  %pid = nvvm.read.ptx.sreg.ctaid.x : i32

  // 循环展开：每个线程处理 4 个元素
  // ... （大量展开的 load/add/store 操作）

  // PTX 内联汇编：ld.global.f32, add.f32, st.global.f32
  // 最终生成的 PTX 代码类似：
  //   @%p ld.global.v4.f32 {%f0,%f1,%f2,%f3}, [%rd0];
  //   @%p ld.global.v4.f32 {%f4,%f5,%f6,%f7}, [%rd1];
  //   add.f32 %f8, %f0, %f4;
  //   ...
  //   @%p st.global.v4.f32 [%rd2], {%f8,%f9,%f10,%f11};

  llvm.return
}
```

<div data-component="dialect-lowering-diagram">

```
向量加法的 Lowering 路径：

Python                    tl.load, tl.store, +
  │
  ▼
tt (Triton Dialect)       tt.load, tt.store, arith.addf
  │                       tensor<256xf32>（无布局信息）
  │
  │ tritongpu-coalesce
  │ tritongpu-materialize
  ▼
ttg (TritonGPU)           ttg.global_load, ttg.global_store
  │                       tensor<256xf32, #blocked>（带布局）
  │
  │ tritongpu-to-llvm
  ▼
llvm (LLVM Dialect)       llvm.load, llvm.store, llvm.fadd
  │                       展开为标量操作
  │
  │ LLVM Translation
  ▼
PTX / LLVM IR             ld.global, st.global, add.f32
```

[组件：方言 lowering 路径图]

</div>

---

## 11.3 Triton Dialect (tt)

### 11.3.1 ODS 定义体系

Triton Dialect 的 Operation 使用 MLIR 的 **ODS**（Operation Definition Specification）框架定义。ODS 是一种声明式语言，用于定义 Operation 的签名、约束和行为。

核心定义文件位于 Triton 源码中：

```
include/triton/Dialect/Triton/IR/
├── TritonOps.td          ← Operation 定义（核心）
├── TritonDialect.td      ← Dialect 定义
├── TritonTypes.td        ← Type 定义
├── TritonAttrDefs.td     ← Attribute 定义
└── TritonInterfaces.td   ← Interface 定义
```

一个典型的 ODS Operation 定义如下（以 `tt.load` 为例）：

```tablegen
// include/triton/Dialect/Triton/IR/TritonOps.td
// (简化版本，突出核心结构)

def TT_LoadOp : TT_Op<"load", [
    DeclareOpInterfaceMethods<OpInterfaces::MemorySideEffectInterface>,
    SameOperandsAndResultShape,
    TritonVerifyTensorLayoutInterface
]> {
    let summary = "Load a tensor from memory pointed to by `ptr`";
    let description = [{
        Load a tensor of data from memory locations pointed to by `ptr`,
        using `mask` and `other` to handle boundary conditions.

        If `mask` is provided, only elements where mask is true are loaded.
        Elements where mask is false get the value from `other` (or undef).
    }];

    let arguments = (ins
        TT_PtrType:$ptr,        // 指针张量
        DefaultValuedAttr<I32Attr, "1">:$cache,   // 缓存提示
        DefaultValuedAttr<I32Attr, "1">:$evict,    // 驱逐策略
        DefaultValuedAttr<BoolAttr, "false">:$isVolatile,
        OptionalAttr<I32Attr>:$width,              // 向量宽度
        Variadic<AnyType>:$mask,                   // 可选的 mask
        Variadic<AnyType>:$other                   // 可选的默认值
    );

    let results = (outs
        AnyType:$result         // 加载结果张量
    );

    let assemblyFormat = [{
        $ptr (`,` $mask^)? (`,` $other^)? attr-dict `:` type($ptr) `->` type($result)
    }];
}
```

### 11.3.2 tt 核心 Operation 类型

Triton Dialect 定义了一组精简的张量操作。这些操作的设计遵循 "够用就好" 的原则——覆盖常见的 GPU 计算模式，但不过度抽象。

#### 内存操作

```mlir
// tt.load — 从全局内存加载张量
// 语法: tt.load %ptr, %mask, %other : tensor<Nx!tt.ptr<T>> -> tensor<NxT>
%x = tt.load %ptrs, %mask, %other {
  cache = 1 : i32,    // CA = .ca (cache all)
  evict = 1 : i32,    // EVICT_NORMAL
  isVolatile = false
} : tensor<256x!tt.ptr<f32>> -> tensor<256xf32>

// tt.store — 将张量存储到全局内存
// 语法: tt.store %ptr, %value, %mask : tensor<Nx!tt.ptr<T>>, tensor<NxT>
tt.store %ptrs, %values, %mask {
  cache = 1 : i32,
  evict = 1 : i32
} : tensor<256x!tt.ptr<f32>>, tensor<256xf32>

// tt.addptr — 指针偏移计算
// 语法: tt.addptr %base, %offset : tensor<Nx!tt.ptr<T>>, tensor<Nxi32>
%new_ptrs = tt.addptr %base_ptrs, %offsets
  : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
```

#### 指针与寻址

```mlir
// tt.splat — 将标量广播为张量
// 将一个标量指针广播为指针张量
%ptrs = tt.splat %base_ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>

// tt.make_range — 创建索引张量
// 语法: tt.make_range {start, end} : tensor<Nxi32>
%range = tt.make_range {end = 256 : i32, start = 0 : i32}
  : tensor<256xi32>

// tt.make_tensor_ptr — 创建块指针（Block Pointer）
// 用于高级的张量内存访问模式
%block_ptr = tt.make_tensor_ptr %base[%c0, %c0] [%M, %N] [%stride_m, %stride_n]
  : !tt.ptr<tensor<128x64xf16>>
```

#### 计算操作

```mlir
// tt.dot — 矩阵乘法（最重要的计算操作）
// 语法: tt.dot %a, %b, %c, %allow_tf32 : tensor<MxKxT>, tensor<KxNxT>, tensor<MxNxT>
%result = tt.dot %a, %b, %c %allow_tf32 {
  maxNumImpreciseAcc = 0 : i32
} : tensor<128x64xf16> * tensor<64x64xf16> -> tensor<128x64xf32>

// tt.expand_dims — 扩展维度
// 语法: tt.expand_dims %tensor : tensor<NxT> -> tensor<1xNxT> 或 tensor<Nx1xT>
%expanded = tt.expand_dims %tensor {axis = 1 : i32}
  : tensor<256xf32> -> tensor<256x1xf32>

// tt.extern_elementwise — 调用外部数学函数
// 用于 math.exp, math.log 等
%y = tt.extern_elementwise %x {libname = "", libpath = "", pure = true,
  symbol = "__nv_expf"} : (tensor<256xf32>) -> tensor<256xf32>
```

#### 控制流

```mlir
// tt.get_program_id — 获取程序 ID（对应 CUDA 的 blockIdx）
%pid = tt.get_program_id x : i32
%pid_y = tt.get_program_id y : i32

// tt.get_num_programs — 获取程序数量（对应 CUDA 的 gridDim）
%nprocs = tt.get_num_programs x : i32

// tt.assert — 断言
tt.assert %cond, "message" : tensor<256xi1>
```

### 11.3.3 tt.dot 的语义详解

`tt.dot` 是 Triton 中最重要的操作——它映射到 GPU 的 Tensor Core 指令。让我们深入理解它的语义：

```mlir
// tt.dot 计算: D = A × B + C
// A: tensor<MxKxf16>  — 左矩阵
// B: tensor<KxNxf16>  — 右矩阵
// C: tensor<MxNxf32>  — 累加器（通常是 f32）
// D: tensor<MxNxf32>  — 结果

%c_zero = arith.constant dense<0.0> : tensor<128x64xf32>
%d = tt.dot %a, %b, %c_zero {allowTF32 = true}
  : tensor<128x32xf16> * tensor<32x64xf16> -> tensor<128x64xf32>

// 关键特性：
// 1. 输入 A/B 通常是 f16 或 bf16（半精度）
// 2. 累加器 C 和结果 D 是 f32（全精度）
// 3. allowTF32 控制是否允许 TF32 格式的 Tensor Core 计算
// 4. 实际映射到 mma.sync 或 wmma 指令
```

### 11.3.4 tt 中的 Attribute 详解

```mlir
// cache 属性：控制加载时的缓存行为
// 0 = CA (cache all levels)
// 1 = CG (cache global, bypass L1)
// 2 = CS (cache streaming)
// 3 = CV (cache volatile, no caching)

// evict 属性：控制缓存驱逐策略
// 0 = EVICT_FIRST (优先驱逐)
// 1 = EVICT_NORMAL (正常)
// 2 = EVICT_LAST (保留缓存)

// 指针类型属性
!tt.ptr<f32>           // 指向 f32 的普通指针
!tt.ptr<tensor<128x64xf16>>  // 指向张量的块指针（用于 tt.make_tensor_ptr）
```

---

## 11.4 TritonGPU Dialect (ttg)

### 11.4.1 ttg 的设计动机

tt 方言中的 `tensor<256xf32>` 只告诉我们数据的**逻辑形状**，没有告诉我们数据在**物理上如何分布**。在一个 warp（32 个线程）上，这 256 个元素是如何分配到各个线程的？每个线程持有多少个元素？元素在内存中的排列顺序是什么？

这些问题在 ttg 方言中通过 **Layout Encoding** 来回答。

```
┌─────────────────────────────────────────────────────────────────┐
│     从 tt 到 ttg：从"做什么"到"怎么放"                            │
│                                                                 │
│   tt 方言:  tensor<256xf32>                                     │
│   "有 256 个 f32 元素的张量"                                      │
│                                                                 │
│   ttg 方言: tensor<256xf32, #blocked>                           │
│   "256 个 f32 元素，按 blocked 方式分布"                          │
│   #blocked = {                                                 │
│     sizePerThread = [4],     // 每个线程持有 4 个元素              │
│     threadsPerWarp = [32],   // 每个 warp 有 32 个线程            │
│     warpsPerCTA = [8],       // 每个 CTA 有 8 个 warp            │
│     order = [0]              // 维度遍历顺序                      │
│   }                                                            │
│   → 总元素 = 4 × 32 × 8 = 1024 ... (需要处理不整除的情况)        │
└─────────────────────────────────────────────────────────────────┘
```

### 11.4.2 ttg 核心 Operation

```mlir
// ttg.global_load — 从全局内存加载到寄存器（带布局）
%x = ttg.global_load %ptr, %mask, %other
  : tensor<256xf32, #blocked>

// ttg.global_store — 从寄存器存储到全局内存（带布局）
ttg.global_store %ptr, %value, %mask
  : tensor<256xf32, #blocked>

// ttg.local_load — 从共享内存加载到寄存器
// 通常在布局转换时使用：shared → blocked 或 shared → dot_op
%x = ttg.local_load %memdesc
  : !ttg.memdesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>

// ttg.local_store — 从寄存器存储到共享内存
// 通常在布局转换时使用：blocked → shared
ttg.local_store %value, %memdesc
  : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared>

// ttg.local_alloc — 分配共享内存
%smem = ttg.local_alloc
  : !ttg.memdesc<128x64xf16, #shared, #smem>

// ttg.local_dealloc — 释放共享内存
ttg.local_dealloc %smem
  : !ttg.memdesc<128x64xf16, #shared, #smem>

// ttg.memdesc_subview — 共享内存子视图
%sub = ttg.memdesc_subview %smem[%c0, %c0]
  : !ttg.memdesc<128x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared>

// ttg.convert_layout — 转换数据布局
// 在不同 Encoding 之间转换，可能经过共享内存
%converted = ttg.convert_layout %value
  : tensor<256xf32, #blocked> -> tensor<256xf32, #mma>
```

### 11.4.3 Layout Encoding 概览

Layout Encoding 是 ttg 方言的核心概念。它描述了张量数据在线程束（warp）中的物理分布方式。Triton 定义了以下主要的 Encoding 类型：

```
┌─────────────────────────────────────────────────────────────────┐
│                    Layout Encoding 层次                          │
│                                                                 │
│  #ttg.blocked                                                │
│  ├── 通用的寄存器布局                                          │
│  ├── 每个线程持有 sizePerThread 个元素                          │
│  └── 用于 load/store 后的初始布局                               │
│                                                                 │
│  #ttg.shared                                                 │
│  ├── 共享内存布局                                              │
│  ├── 支持 bank conflict-free 的数据排列                        │
│  └── 用于 layout conversion 的中间态                           │
│                                                                 │
│  #ttg.dot_op                                                 │
│  ├── dot 操作的输入布局                                        │
│  ├── 针对 Tensor Core 的数据分布优化                            │
│  └── 与特定 MMA 指令的 operand 寄存器布局匹配                   │
│                                                                 │
│  #ttg.mma (或 NvidiaMmaEncodingAttr)                          │
│  ├── Tensor Core (MMA) 的输出布局                              │
│  ├── 映射到 mma.sync / wmma 指令                               │
│  └── 包含版本信息（如 mma.v2, mma.v3）                          │
│                                                                 │
│  #ttg.slice                                                  │
│  ├── 高维张量的一个切片布局                                     │
│  └── 用于 expand_dims / reduce 等操作                          │
└─────────────────────────────────────────────────────────────────┘
```

### 11.4.4 BlockedEncodingAttr 详解

`#ttg.blocked` 是最基础的 Layout Encoding，描述数据在寄存器中的基本分布：

```mlir
#blocked = #ttg.blocked<{
  sizePerThread    = [1, 4],    // 每个线程在每个维度上持有多少元素
  threadsPerWarp   = [8, 4],    // 每个 warp 在每个维度上有多少线程
  warpsPerCTA      = [4, 1],    // 每个 CTA 在每个维度上有多少 warp
  order            = [1, 0]     // 维度遍历优先级（C 行主序 = [1,0]）
}>
```

让我们详细计算一个例子：

```
// 给定 #blocked 编码：
// sizePerThread  = [1, 4]
// threadsPerWarp = [8, 4]
// warpsPerCTA    = [4, 1]
// order          = [1, 0]
//
// 对于 tensor<32x64xf32, #blocked>：
//
// 维度 0 (行):
//   每个线程持有 1 个元素
//   每个 warp 有 8 个线程 → 每个 warp 覆盖 8 行
//   4 个 warp → CTA 覆盖 32 行 ✓
//
// 维度 1 (列):
//   每个线程持有 4 个元素
//   每个 warp 有 4 个线程 → 每个 warp 覆盖 16 列
//   1 个 warp → CTA 覆盖 16 列
//   但张量有 64 列 → 需要多次访问（wrap around）
//
// 每个 warp 负责的区域:
//   warp 0: rows [0,8),   cols [0,16), [16,32), [32,48), [48,64)
//   warp 1: rows [8,16),  cols [0,16), [16,32), [32,48), [48,64)
//   warp 2: rows [16,24), cols [0,16), [16,32), [32,48), [48,64)
//   warp 3: rows [24,32), cols [0,16), [16,32), [32,48), [48,64)
//
// 每个线程持有的元素（以 warp 0 中的 thread 0 为例）:
//   线程 0 在 warp 中的位置: row_idx=0, col_idx=0
//   持有元素: (0,0), (0,1), (0,2), (0,3) — 4 个 f32 元素
//   以及 wrap around: (0,16), (0,17), (0,18), (0,19)
//                     (0,32), (0,33), (0,34), (0,35)
//                     (0,48), (0,49), (0,50), (0,51)
```

<div data-component="blocked-layout-diagram">

```
tensor<32x64xf32, #blocked> 的物理分布：

#blocked = #ttg.blocked<{
  sizePerThread    = [1, 4],
  threadsPerWarp   = [8, 4],
  warpsPerCTA      = [4, 1],
  order            = [1, 0]
}>

         Warp 0 (rows 0-7)              Warp 1 (rows 8-15)    ...
      Thread  Thread  Thread  Thread
        T0      T1      T2      T3      ...
Row 0: [e00,e01,e02,e03] [e04,...,e07] [e08,...] [e12,...]
       [e16,e17,e18,e19] [e20,...,e23] [e24,...] [e28,...]
       [e32,e33,e34,e35] [e36,...,e39] [e40,...] [e44,...]
       [e48,e49,e50,e51] [e52,...,e55] [e56,...] [e60,...]
Row 1: [e64,e65,e66,e67] ...
  ...

每个线程在列方向持有 4 个连续元素
线程按列优先排列（order=[1,0] 表示列维度优先）
```

[组件：BlockedEncoding 数据分布示意图]

</div>

### 11.4.5 SharedEncodingAttr 详解

`#ttg.shared` 描述数据在共享内存中的布局。共享内存的关键目标是**避免 bank conflict**。

```mlir
#shared = #ttg.shared<{
  vec        = 4,       // 每次访问的向量宽度
  perPhase   = 1,       // 每阶段的行数
  maxPhase   = 8,       // 最大阶段数
  order      = [1, 0],  // 维度遍历顺序
  hasLeadingOffset = false  // 是否有前导偏移（Hopper TMA 用）
}>
```

共享内存的 bank conflict-free 排列策略：

```
// 共享内存有 32 个 bank，每个 bank 4 字节
// 如果连续的线程访问连续的地址，且地址跨 bank 对齐
// → 无 bank conflict
//
// #shared 的 perPhase 和 maxPhase 控制行的交错排列：
//
// 普通行主序（有 bank conflict）：
//   Row 0: [bank0, bank1, bank2, ..., bank31]
//   Row 1: [bank0, bank1, bank2, ..., bank31]  ← 列访问冲突！
//
// 交错排列（#shared 控制）：
//   Row 0:  [bank0,  bank1,  bank2,  ..., bank31]
//   Row 1:  [bank4,  bank5,  bank6,  ..., bank3, bank4, ...]  ← 偏移
//   Row 2:  [bank8,  bank9,  bank10, ..., bank7, bank8, ...]
//   → 列方向访问无 bank conflict
```

### 11.4.6 DotOperandEncodingAttr 与 MmaEncodingAttr

Tensor Core 的矩阵乘法指令要求输入数据按照特定的格式分布在寄存器中。

```mlir
// MmaEncodingAttr — 描述 Tensor Core MMA 指令的输出布局
#mma = #ttg.nvidia_mma<{
  versionMajor = 2,        // MMA 指令版本（V100=1, A100=2, H100=3）
  versionMinor = 0,
  warpsPerCTA = [4, 1],    // warp 分配
  CTAsPerCGA = [1, 1],     // CTA 分组
  instrShape = [16, 8],    // 单条 MMA 指令的形状
  isMfma = false           // 是否为 AMD MFMA
}>

// DotOperandEncodingAttr — 描述 dot 操作的输入 operand 布局
#dot_op_0 = #ttg.dot_op<{
  opIdx = 0,               // 第一个 operand（A 矩阵）
  parent = #mma,           // 关联的 MMA 编码
  kWidth = 4               // 每次 MMA 指令消耗的 K 维度元素数
}>

#dot_op_1 = #ttg.dot_op<{
  opIdx = 1,               // 第二个 operand（B 矩阵）
  parent = #mma,
  kWidth = 4
}>
```

Tensor Core 的寄存器数据分布：

```
// 以 A100 的 mma.sync.aligned.m16n8k16 指令为例
// 输入 A: 16×16 矩阵，分给 32 个线程
// 每个线程持有 8 个元素（来自 4 行，每行 2 个）
//
// 线程布局（简化）:
//   Thread 0:  A[0][0:2], A[1][0:2], A[2][0:2], A[3][0:2]
//   Thread 1:  A[0][2:4], A[1][2:4], A[2][2:4], A[3][2:4]
//   ...
//   Thread 31: A[12][14:16], A[13][14:16], A[14][14:16], A[15][14:16]
//
// #dot_op{kWidth=4} 表示每次从共享内存加载 4 个连续元素
// 这与 Tensor Core 指令的 operand 寄存器布局直接对应
```

### 11.4.7 ttg.memdesc 类型

`!ttg.memdesc` 是共享内存的描述符类型，它不是指向共享内存的指针，而是对共享内存区域的**类型化视图**：

```mlir
// !ttg.memdesc 的语法
!ttg.memdesc<SHAPE, ENCODING, ADDRESS_SPACE>

// 例子
!ttg.memdesc<128x64xf16, #shared, #smem>
//   SHAPE:         128x64xf16
//   ENCODING:      #shared（共享内存布局）
//   ADDRESS_SPACE: #smem（共享内存地址空间）

// 分配
%smem = ttg.local_alloc : !ttg.memdesc<128x64xf16, #shared, #smem>

// 子视图（用于 tiling）
%sub = ttg.memdesc_subview %smem[%row_offset, %col_offset]
  : !ttg.memdesc<128x64xf16, #shared, #smem>
  -> !ttg.memdesc<64x64xf16, #shared, #smem>

// 存入共享内存
ttg.local_store %value, %smem
  : tensor<128x64xf16, #blocked> -> !ttg.memdesc<128x64xf16, #shared, #smem>

// 从共享内存读出
%result = ttg.local_load %smem
  : !ttg.memdesc<128x64xf16, #shared, #smem> -> tensor<128x64xf16, #dot_op>
```

---

## 11.5 Type 系统

### 11.5.1 Triton 的类型层次

Triton 的类型系统跨越多个方言层次：

```
┌─────────────────────────────────────────────────────────────────┐
│                    Triton Type 系统层次                          │
│                                                                 │
│  MLIR 内置类型                                                   │
│  ├── i1, i8, i16, i32, i64           整数类型                    │
│  ├── f16, bf16, f32, f64             浮点类型                    │
│  ├── index                            索引类型                    │
│  └── tensor<NxMxT>                   张量类型                    │
│                                                                 │
│  tt 方言类型                                                     │
│  ├── !tt.ptr<T>                      标量指针                    │
│  └── tensor<Nx!tt.ptr<T>>            指针张量                    │
│                                                                 │
│  ttg 方言类型                                                    │
│  ├── tensor<NxT, #encoding>          带布局的张量                 │
│  └── !ttg.memdesc<Shape, Enc, AS>    共享内存描述符              │
│                                                                 │
│  gpu 方言类型                                                    │
│  └── !gpu.barrier                    GPU barrier                 │
└─────────────────────────────────────────────────────────────────┘
```

### 11.5.2 tt.ptr<T> 类型

`!tt.ptr<T>` 是 Triton 的指针类型。它可以指向全局内存、共享内存或常量内存，具体的地址空间信息由后端添加。

```mlir
// 标量指针
!tt.ptr<f32>                    // 指向单个 f32 的指针
!tt.ptr<f16>                    // 指向单个 f16 的指针
!tt.ptr<i32>                    // 指向单个 i32 的指针

// 块指针（用于 tt.make_tensor_ptr / tl.make_block_ptr）
!tt.ptr<tensor<128x64xf16>>     // 指向 128x64 张量块的指针

// 指针张量（每个元素一个指针，用于 tt.load/tt.store）
tensor<256x!tt.ptr<f32>>        // 256 个 f32 指针

// 指针算术
%new_ptr = tt.addptr %base_ptr, %offset : !tt.ptr<f32>, i32
%new_ptrs = tt.addptr %base_ptrs, %offsets
  : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
```

### 11.5.3 tensor<NxT> vs tensor<NxT, #encoding>

同一个逻辑张量在不同方言中有不同的表示：

```mlir
// tt 方言：无布局信息
%t0 = arith.constant dense<1.0> : tensor<128x64xf16>
// 含义：128x64 个 f16 元素的张量，不关心物理分布

// ttg 方言：带布局信息
#blocked = #ttg.blocked<{sizePerThread=[1,4], threadsPerWarp=[8,4],
                         warpsPerCTA=[4,1], order=[1,0]}>
%t1 = arith.constant dense<1.0> : tensor<128x64xf16, #blocked>
// 含义：128x64 个 f16 元素，按 #blocked 方式分布在各线程的寄存器中

// 注意：两种表示的"值"是相同的，但物理分布不同
// ttg.convert_layout 可以改变布局：
%t2 = ttg.convert_layout %t1
  : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #mma>
```

### 11.5.4 类型转换规则

在方言转换过程中，类型会按照以下规则转换：

```
┌─────────────────────────────────────────────────────────────────┐
│                    类型转换规则                                   │
│                                                                 │
│  tt → ttg 转换:                                                │
│  ┌────────────────────────┬─────────────────────────────────┐  │
│  │ tt 类型                 │ ttg 类型                         │  │
│  ├────────────────────────┼─────────────────────────────────┤  │
│  │ tensor<NxT>            │ tensor<NxT, #blocked>           │  │
│  │ !tt.ptr<T>             │ !tt.ptr<T>  (不变)              │  │
│  │ tensor<Nx!tt.ptr<T>>   │ tensor<Nx!tt.ptr<T>, #blocked> │  │
│  │ i32, f32, ...          │ 不变                            │  │
│  └────────────────────────┴─────────────────────────────────┘  │
│                                                                 │
│  ttg → llvm 转换:                                              │
│  ┌────────────────────────┬─────────────────────────────────┐  │
│  │ ttg 类型                │ llvm 类型                        │  │
│  ├────────────────────────┼─────────────────────────────────┤  │
│  │ tensor<NxT, #enc>      │ 展开为多个标量寄存器              │  │
│  │ !ttg.memdesc<...>      │ !llvm.ptr (共享内存指针)         │  │
│  │ !tt.ptr<T>             │ !llvm.ptr                       │  │
│  │ i32                    │ i32                             │  │
│  └────────────────────────┴─────────────────────────────────┘  │
│                                                                 │
│  张量展开规则:                                                   │
│  tensor<256xf32, #blocked{sizePerThread=[4], ...}>             │
│  → 每个线程持有 4 个 f32 寄存器                                   │
│  → 展开为 4 个独立的 f32 值                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 11.6 Attribute 系统

### 11.6.1 Layout Attribute 的定义

Layout Attribute 是 ttg 方言的核心。每个 Layout Attribute 都是 MLIR 的一个 Dialect Attribute，定义在 `include/triton/Dialect/TritonGPU/IR/Attrs.td` 中。

```tablegen
// include/triton/Dialect/TritonGPU/IR/Attrs.td (简化)
// (基于 Triton 源码结构)

// 定义 BlockedEncodingAttr
def BlockedEncodingAttr : TritonGPU_Attr<"BlockedEncoding"> {
    let mnemonic = "ttg.blocked";

    let parameters = (ins
        "SmallVector<unsigned>":$sizePerThread,
        "SmallVector<unsigned>":$threadsPerWarp,
        "SmallVector<unsigned>":$warpsPerCTA,
        "SmallVector<unsigned>":$CTAsPerCGA,
        "SmallVector<unsigned>":$order,
        "SmallVector<unsigned>":$instOrder
    );
    // ... verifier, printer, parser
}

// 定义 SharedEncodingAttr
def SharedEncodingAttr : TritonGPU_Attr<"SharedEncoding"> {
    let mnemonic = "ttg.shared";

    let parameters = (ins
        "unsigned":$vec,
        "unsigned":$perPhase,
        "unsigned":$maxPhase,
        "SmallVector<unsigned>":$order,
        "bool":$hasLeadingOffset
    );
}

// 定义 NvidiaMmaEncodingAttr
def NvidiaMmaEncodingAttr : TritonGPU_Attr<"NvidiaMmaEncoding"> {
    let mnemonic = "ttg.nvidia_mma";

    let parameters = (ins
        "unsigned":$versionMajor,
        "unsigned":$versionMinor,
        "SmallVector<unsigned>":$warpsPerCTA,
        "SmallVector<unsigned>":$CTAsPerCGA,
        "SmallVector<unsigned>":$instrShape,
        "bool":$isMfma
    );
}

// 定义 DotOperandEncodingAttr
def DotOperandEncodingAttr : TritonGPU_Attr<"DotOperandEncoding"> {
    let mnemonic = "ttg.dot_op";

    let parameters = (ins
        "unsigned":$opIdx,
        "Attribute":$parent,
        "unsigned":$kWidth
    );
}
```

### 11.6.2 Layout Attribute 的语义

每个 Layout Attribute 描述了一种"线程到数据"的映射关系。让我们用数学语言来描述：

```
// 对于 #blocked 编码：
//
// 给定张量 T 的形状为 [d0, d1, ..., d_{n-1}]
// 和编码参数 sizePerThread = [s0, s1, ..., s_{n-1}]
//                threadsPerWarp = [t0, t1, ..., t_{n-1}]
//                warpsPerCTA = [w0, w1, ..., w_{n-1}]
//
// 对于张量中的位置 (i0, i1, ..., i_{n-1}):
//
// 1. warp_id_k = i_k / (s_k * t_k)
//    → 确定该元素属于哪个 warp
//
// 2. thread_id_k = (i_k / s_k) % t_k
//    → 确定该元素在 warp 中属于哪个线程
//
// 3. register_id_k = i_k % s_k
//    → 确定该元素在线程的寄存器中的位置
//
// 约束：s_k * t_k * w_k >= d_k（所有元素都要被覆盖）
```

### 11.6.3 Layout Attribute 的传播

在编译过程中，Layout Attribute 会按照以下规则传播：

```
┌─────────────────────────────────────────────────────────────────┐
│                 Layout Attribute 传播规则                         │
│                                                                 │
│  1. Load/Store: 输入的 layout 由使用者（consumer）决定            │
│     %x = tt.load %ptr  →  %x 的 layout 由使用 %x 的操作决定       │
│                                                                 │
│  2. Elementwise (add, mul, exp, ...): 继承使用者的 layout        │
│     %z = arith.addf %x, %y  →  %z 的 layout = %x 的 layout     │
│     (前提是 %x 和 %y 有相同的 layout)                             │
│                                                                 │
│  3. Dot: 输出是 #mma，输入是 #dot_op                             │
│     %d = tt.dot %a, %b, %c                                      │
│     → %d 的 layout = #mma                                       │
│     → %a 的 layout = #dot_op{opIdx=0, parent=#mma}              │
│     → %b 的 layout = #dot_op{opIdx=1, parent=#mma}              │
│                                                                 │
│  4. Convert layout: 显式转换                                      │
│     %y = ttg.convert_layout %x                                  │
│     → %y 的 layout 指定，%x 的 layout 由其使用者决定              │
│                                                                 │
│  5. Reduce: 沿某个维度 reduce，layout 变化                       │
│     %r = "tt.reduce"(%x) {axis=0}                               │
│     → 输出 layout 丢失维度 0 的 encoding                         │
│                                                                 │
│  6. Expand_dims: 增加维度，layout 扩展                           │
│     %e = tt.expand_dims %x {axis=1}                             │
│     → 输出 layout 在 axis=1 处插入新的 encoding 维度              │
└─────────────────────────────────────────────────────────────────┘
```

### 11.6.4 Layout Conversion 路径

当两个操作的 layout 不兼容时，需要进行 layout conversion。Triton 通常通过共享内存作为中介：

```
┌─────────────────────────────────────────────────────────────────┐
│                 Layout Conversion 路径                           │
│                                                                 │
│  路径 1: blocked → blocked (相同 encoding，不同 shape)           │
│  → 直接寄存器重排，无需共享内存                                   │
│                                                                 │
│  路径 2: blocked → shared → dot_op (最常见)                      │
│  ttg.local_store: blocked → shared                              │
│  ttg.local_load:  shared  → dot_op                              │
│                                                                 │
│  路径 3: blocked → mma (通过共享内存或直接转换)                   │
│  如果 encoding 兼容，可以直接在寄存器中重排                       │
│                                                                 │
│  路径 4: dot_op → mma (Tensor Core 输出)                        │
│  这是 MMA 指令的自然输出，不需要额外转换                          │
│                                                                 │
│  完整的 GEMM 数据流:                                             │
│  A: global → blocked → shared → dot_op_0 → MMA                  │
│  B: global → blocked → shared → dot_op_1 → MMA                  │
│  C: blocked → mma (accumulator)                                 │
│  D: mma → shared → blocked → global                             │
└─────────────────────────────────────────────────────────────────┘
```

### 11.6.5 不同硬件的 Encoding 变体

不同 GPU 架构支持不同的 MMA 指令，因此 Encoding 的参数也不同：

| GPU 架构 | MMA 版本 | 典型 instrShape | 特性 |
|---------|---------|----------------|------|
| V100 | mma.v1 | [8,8,4] | FP16 Tensor Core |
| A100 | mma.v2 | [16,8,16] | TF32/BF16/FP16/INT8 |
| H100 | mma.v3 | [16,8,32] | WGMMA (Warp Group MMA) |
| MI250 | mfma | [32,32,8] 或 [16,16,16] | AMD Matrix Core |

```mlir
// A100 的 MMA encoding
#mma_v2 = #ttg.nvidia_mma<{
  versionMajor = 2, versionMinor = 0,
  warpsPerCTA = [4, 1],
  instrShape = [16, 8]
}>

// H100 的 MMA encoding（WGMMA）
#mma_v3 = #ttg.nvidia_mma<{
  versionMajor = 3, versionMinor = 0,
  warpsPerCTA = [4, 1],
  warpsPerCTA = [1, 1, 4],  // Hopper WGMMA 用 3D warp 分配
  instrShape = [16, 8, 32]
}>

// AMD MI250 的 MFMA encoding
#mfma = #ttg.amd_mfma<{
  versionMajor = 2, versionMinor = 0,
  warpsPerCTA = [4, 1],
  instrShape = [32, 32],
  isTranspose = false
}>
```

---

## 11.7 Dialect Conversion

### 11.7.1 MLIR Dialect Conversion 框架

MLIR 提供了 `DialectConversion` 框架来实现方言间的转换。这个框架的核心思想是**类型转换**和 **Operation 转换**的解耦：

```
┌─────────────────────────────────────────────────────────────────┐
│              MLIR Dialect Conversion 框架                        │
│                                                                 │
│  1. Type Converter: 定义类型如何从源方言映射到目标方言             │
│     TypeConverter::addConversion(                               │
│       [](TensorType type) -> Type {                             │
│         return tensor_with_encoding(type, blockedEncoding);     │
│       }                                                         │
│     )                                                           │
│                                                                 │
│  2. Conversion Patterns: 定义每个 Operation 如何被转换            │
│     struct LoadOpConversion : OpConversionPattern<tt::LoadOp> { │
│       matchAndRewrite(...) {                                    │
│         // 将 tt.load 转换为 ttg.global_load                     │
│         // 更新类型，添加 layout attribute                       │
│       }                                                         │
│     }                                                           │
│                                                                 │
│  3. Conversion Target: 定义哪些 Operation 是合法的               │
│     target.addLegalOp<ttg::GlobalLoadOp>();                     │
│     target.addIllegalOp<tt::LoadOp>();                          │
│                                                                 │
│  4. applyPartialConversion: 执行转换                             │
│     // 框架自动处理：                                            │
│     // - 类型转换的传播                                          │
│     // - Operation 的递归转换                                    │
│     // - 非法 Operation 的报告                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 11.7.2 tt → ttg 的布局推断

从 tt 到 ttg 的转换（`tritongpu-materialize` pass）是最关键的一步——它需要为每个张量值推断出合适的 Layout Encoding。

布局推断的算法大致如下：

```
布局推断算法（简化版）：

1. 收集约束
   - 遍历所有 Operation
   - 对于每个 Operation，根据其类型确定 layout 约束：
     - tt.dot 的输出 → 必须是 #mma
     - tt.dot 的 operand → 必须是 #dot_op
     - tt.load/tt.store → 可以是 #blocked
     - 其他 elementwise → 继承使用者的 layout

2. 传播 layout
   - 从 sink（输出）向 source（输入）反向传播
   - 如果一个值有多个使用者且要求不同的 layout
     → 在该点插入 ttg.convert_layout

3. 选择初始 layout
   - 对于 #mma，根据目标 GPU 选择版本和参数
   - 对于 #blocked，根据张量 shape 和 warp 数选择参数
   - 对于 #shared，根据访问模式选择 bank conflict-free 参数

4. 插入 layout conversion
   - 在 layout 不匹配的 Operation 之间插入 ttg.convert_layout
   - 优化：尝试找到最优的 conversion 路径
```

让我们看一个具体的例子：

```mlir
// 输入 (tt 方言):
func.func @gemm(%a: tensor<128x32xf16>, %b: tensor<32x64xf16>,
                 %c: tensor<128x64xf32>) -> tensor<128x64xf32> {
  %d = tt.dot %a, %b, %c {allowTF32 = true}
    : tensor<128x32xf16> * tensor<32x64xf16> -> tensor<128x64xf32>
  tt.return %d : tensor<128x64xf32>
}

// 输出 (ttg 方言, 布局推断后):
#mma = #ttg.nvidia_mma<{versionMajor=2, versionMinor=0,
                         warpsPerCTA=[4,1], instrShape=[16,8]}>
#dot_op_0 = #ttg.dot_op<{opIdx=0, parent=#mma, kWidth=4}>
#dot_op_1 = #ttg.dot_op<{opIdx=1, parent=#mma, kWidth=4}>

func.func @gemm(%a: tensor<128x32xf16, #dot_op_0>,
                 %b: tensor<32x64xf16, #dot_op_1>,
                 %c: tensor<128x64xf32, #mma>) -> tensor<128x64xf32, #mma> {
  %d = tt.dot %a, %b, %c {allowTF32 = true}
    : tensor<128x32xf16, #dot_op_0> * tensor<32x64xf16, #dot_op_1>
    -> tensor<128x64xf32, #mma>
  tt.return %d : tensor<128x64xf32, #mma>
}
```

### 11.7.3 ttg → llvm 的指令选择

从 ttg 到 llvm 的转换是"最后一步"——将带有布局信息的张量操作降低到具体的 GPU 指令。

```mlir
// 输入 (ttg 方言):
#mma = #ttg.nvidia_mma<{versionMajor=2, versionMinor=0,
                         warpsPerCTA=[4,1], instrShape=[16,8]}>
#shared = #ttg.shared<{vec=4, perPhase=1, maxPhase=4, order=[1,0]}>

func.func @gemm(...) {
  // ... 分配共享内存
  %smem_a = ttg.local_alloc : !ttg.memdesc<128x32xf16, #shared, #smem>
  %smem_b = ttg.local_alloc : !ttg.memdesc<32x64xf16, #shared, #smem>

  // 从全局内存加载到寄存器
  %a_blocked = ttg.global_load %a_ptrs, %mask
    : tensor<128x32xf16, #blocked>
  %b_blocked = ttg.global_load %b_ptrs, %mask
    : tensor<32x64xf16, #blocked>

  // 存入共享内存
  ttg.local_store %a_blocked, %smem_a
    : tensor<128x32xf16, #blocked> -> !ttg.memdesc<128x32xf16, #shared, #smem>
  ttg.local_store %b_blocked, %smem_b
    : tensor<32x64xf16, #blocked> -> !ttg.memdesc<32x64xf16, #shared, #smem>

  // 从共享内存加载到 dot_op 布局
  %a_dotop = ttg.local_load %smem_a
    : !ttg.memdesc<128x32xf16, #shared, #smem> -> tensor<128x32xf16, #dot_op_0>
  %b_dotop = ttg.local_load %smem_b
    : !ttg.memdesc<32x64xf16, #shared, #smem> -> tensor<32x64xf16, #dot_op_1>

  // Tensor Core 矩阵乘法
  %d = tt.dot %a_dotop, %b_dotop, %c_mma
    : tensor<128x32xf16, #dot_op_0> * tensor<32x64xf16, #dot_op_1>
    -> tensor<128x64xf32, #mma>

  tt.return
}

// 输出 (llvm 方言, 指令选择后):
// 每个张量操作被展开为具体的 GPU 指令序列：

func.func @gemm(...) {
  // ... 共享内存分配 → llvm.alloca 或 nvvm 的 shared memory 声明

  // 全局加载 → 展开为多个 ld.global 指令
  // 每个线程加载 sizePerThread 个元素
  %v0 = nvvm.ld.global.f16 [%ptr0]
  %v1 = nvvm.ld.global.f16 [%ptr1]
  // ...

  // 共享内存写入 → st.shared 指令
  nvvm.st.shared.f16 [%smem_addr0], %v0
  nvvm.st.shared.f16 [%smem_addr1], %v1

  // 共享内存读取 → ld.shared 指令
  // + register shuffling（如果需要 layout 转换）
  %a0 = nvvm.ld.shared.v4.f16 [%smem_a_addr]
  %b0 = nvvm.ld.shared.v4.f16 [%smem_b_addr]

  // Tensor Core MMA → mma.sync 指令
  // 一条 mma.sync 指令完成一个 16x8x16 的矩阵乘累加
  nvvm.mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {%d0, %d1, %d2, %d3},
    {%a0, %a1, %a2, %a3},
    {%b0, %b1},
    {%c0, %c1, %c2, %c3}

  llvm.return
}
```

### 11.7.4 Conversion Patterns 详解

Triton 的 Dialect Conversion 由多个 Pass 组成，每个 Pass 包含一组 Conversion Patterns：

```
include/triton/Dialect/TritonGPU/Transforms/
├── Passes.td                    ← Pass 定义
├── Pipeline.cpp                 ← Pass pipeline 管理
├── TritonGPUConversion.cpp      ← tt → ttg 转换
├── Coalesce.cpp                 ← 合并内存访问
├── RemoveLayoutConversions.cpp  ← 消除不必要的 layout conversion
├── OptimizeDotOperands.cpp      ← 优化 dot 操作的 operand
├── Materialize.cpp              ← 分配实际的 layout
├── DecomposeConversions.cpp     ← 分解复杂的 layout conversion
└── ReduceDataDuplication.cpp    ← 减少数据重复

lib/Dialect/TritonGPU/Transforms/
├── TritonGPUToLLVM/             ← ttg → llvm 转换
│   ├── ConvertLayoutOpToLLVM.cpp    ← layout conversion 的 LLVM 实现
│   ├── DotOpToLLVM.cpp              ← dot 操作的 LLVM 实现
│   ├── LoadStoreOpToLLVM.cpp        ← load/store 的 LLVM 实现
│   ├── BarrierOpToLLVM.cpp          ← barrier 的 LLVM 实现
│   └── ElementwiseOpToLLVM.cpp      ← elementwise 的 LLVM 实现
```

### 11.7.5 完整的 Pass Pipeline

Triton 的编译 Pass Pipeline 按以下顺序执行：

```
┌─────────────────────────────────────────────────────────────────┐
│              Triton 编译 Pass Pipeline                           │
│                                                                 │
│  1. tt (Triton Dialect)                                        │
│     │                                                          │
│     ├─ tritongpu-coalesce                                      │
│     │  合并标量 load/store 为向量 load/store                    │
│     │                                                          │
│     ├─ tritongpu-remove-layout-conversions                     │
│     │  消除冗余的 layout conversion                             │
│     │                                                          │
│     ├─ tritongpu-optimize-dot-operands                         │
│     │  优化 dot 操作的 operand 布局                             │
│     │                                                          │
│     ├─ tritongpu-materialize                                    │
│     │  为所有张量值分配实际的 layout encoding                    │
│     │                                                          │
│     │  2. ttg (TritonGPU Dialect)                               │
│     │                                                          │
│     ├─ tritongpu-decompose-conversions                          │
│     │  将复杂的 layout conversion 分解为基本操作                 │
│     │                                                          │
│     ├─ tritongpu-reduce-data-duplication                        │
│     │  减少共享内存中的数据重复                                  │
│     │                                                          │
│     ├─ tritongpu-accelerate-matmul                              │
│     │  将 tt.dot 转换为 Tensor Core 的 MMA 操作                 │
│     │                                                          │
│     ├─ tritongpu-optimize-dot-operands (再次)                   │
│     │                                                          │
│     │  3. ttg → llvm 转换                                       │
│     │                                                          │
│     └─ tritongpu-to-llvm                                        │
│        将所有 ttg 操作降低到 LLVM Dialect                        │
│                                                                 │
│  4. llvm (LLVM Dialect)                                        │
│     │                                                          │
│     └─ LLVM Translation → LLVM IR → PTX/ROCm                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 11.8 MLIR 工具链

### 11.8.1 triton-opt

`triton-opt` 是 Triton 的主要调试工具，类似于 MLIR 的 `mlir-opt`。它可以加载 Triton IR 文件（.mlir），运行指定的 Pass，并输出结果。

```bash
# 基本用法
triton-opt input.mlir --pass-name

# 查看所有可用的 Pass
triton-opt --help | grep triton

# 从 Python 导出 Triton IR 并分析
# 方法 1: 使用环境变量
TRITON_PRINT_AUTOTUNING=1 python my_kernel.py

# 方法 2: 使用 triton.compile 导出
import triton
import triton.compiler as compiler

src = compiler.ASTSource(fn=my_kernel, signature={...}, constexprs={...})
compiled = compiler.compile(src, options=...)
```

### 11.8.2 常用 triton-opt 命令

```bash
# 1. 查看 tt 方言（原始 Triton IR）
triton-opt input.mlir --tritongpu-print-module
# 输出: 带有 tt 操作的 MLIR 模块

# 2. 执行 coalesce pass（合并内存访问）
triton-opt input.mlir --tritongpu-coalesce
# 将标量 load/store 合并为向量 load/store

# 3. 执行完整的 tt → ttg 转换
triton-opt input.mlir \
  --tritongpu-coalesce \
  --tritongpu-remove-layout-conversions \
  --tritongpu-materialize \
  --tritongpu-optimize-dot-operands
# 输出: 带有 layout encoding 的 ttg 方言

# 4. 执行完整的 ttg → llvm 转换
triton-opt input.mlir \
  --tritongpu-coalesce \
  --tritongpu-materialize \
  --tritongpu-decompose-conversions \
  --tritongpu-accelerate-matmul \
  --tritongpu-to-llvm
# 输出: LLVM Dialect

# 5. 查看特定 pass 的效果
triton-opt input.mlir --tritongpu-accelerate-matmul -mlir-print-ir-after-all
# -mlir-print-ir-after-all: 打印每个 pass 后的 IR

# 6. 只运行特定 pass 并打印
triton-opt input.mlir \
  --tritongpu-materialize \
  -mlir-print-ir-before=tritongpu-materialize \
  -mlir-print-ir-after=tritongpu-materialize
```

### 11.8.3 triton-translate

`triton-translate` 将 Triton 的 MLIR IR 转换为目标平台的代码（PTX 或 ROCm）。

```bash
# 将 MLIR 转换为 PTX
triton-opt input.mlir \
  --tritongpu-to-llvm \
  --llvm-to-ptx \
  -o output.ptx

# 等价的一步到位命令
triton-translate input.mlir --target=nvidia --gpu-arch=sm_80

# 查看生成的 PTX
cat output.ptx

# 将 MLIR 转换为 LLVM IR
triton-translate input.mlir --target=llvmir -o output.ll

# 查看 NVIDIA GPU 相关的 PTX
triton-translate input.mlir --target=nvidia --gpu-arch=sm_90
# sm_90 = H100, sm_80 = A100, sm_70 = V100
```

### 11.8.4 使用 triton-opt 分析一个完整例子

让我们通过一个实际的 Triton kernel 来演示完整的分析流程。

**Step 1: Python 代码**

```python
# matmul_kernel.py
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_n = N // BLOCK_N
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    tl.store(C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn),
             accumulator)
```

**Step 2: 导出 Triton IR**

```bash
# 运行 Python 脚本，设置环境变量导出 IR
TRITON_PRINT_AUTOTUNING=1 \
TRITON_PRINT_IR=1 \
python matmul_kernel.py
```

**Step 3: 分析 IR 变换**

```bash
# 将导出的 IR 保存为 .mlir 文件后逐步分析
# 1. 原始 tt 方言
cat matmul_kernel.mlir
```

输出（tt 方言，简化）：

```mlir
// matmul_kernel 在 tt 方言中的表示
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1],
                         warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32} {
  tt.func @matmul_kernel(
    %arg0: !tt.ptr<f16>,  // A
    %arg1: !tt.ptr<f16>,  // B
    %arg2: !tt.ptr<f32>,  // C
    %arg3: i32,  // M
    %arg4: i32,  // N
    %arg5: i32,  // K
    %arg6: i32,  // stride_am
    %arg7: i32,  // stride_ak
    %arg8: i32,  // stride_bk
    %arg9: i32,  // stride_bn
    %arg10: i32, // stride_cm
    %arg11: i32  // stride_cn
  ) {
    // program_id
    %pid = tt.get_program_id x : i32

    // 计算 pid_m, pid_n
    %cst_num_pid_n = arith.constant 4 : i32
    %pid_n = arith.remsi %pid, %cst_num_pid_n : i32
    %pid_m = arith.divsi %pid, %cst_num_pid_n : i32

    // 计算偏移
    %offs_m_base = arith.muli %pid_m, %cst_block_m : i32
    %range_m = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %offs_m = arith.addi %offs_m_base, %range_m : tensor<128xi32>

    %offs_n_base = arith.muli %pid_n, %cst_block_n : i32
    %range_n = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %offs_n = arith.addi %offs_n_base, %range_n : tensor<64xi32>

    %range_k = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>

    // 构造指针张量（通过 expand_dims + broadcast）
    %offs_m_2d = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %offs_n_2d = tt.expand_dims %offs_n {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %offs_k_row = tt.expand_dims %range_k {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %offs_k_col = tt.expand_dims %range_k {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>

    // ... 构造 A 和 B 的指针张量
    // %a_ptrs: tensor<128x32x!tt.ptr<f16>>
    // %b_ptrs: tensor<32x64x!tt.ptr<f16>>

    // 初始化累加器
    %cst_zero = arith.constant dense<0.0> : tensor<128x64xf32>
    %cst_k = arith.constant 0 : i32
    %cst_k_end = arith.constant 256 : i32
    %cst_k_step = arith.constant 32 : i32

    // 循环
    %result = scf.for %k = %cst_k to %cst_k_end step %cst_k_step
      iter_args(%acc = %cst_zero) -> tensor<128x64xf32> {

      // 加载 A 和 B 的一个块
      %a = tt.load %a_ptrs_k, %mask_k : tensor<128x32xf16>
      %b = tt.load %b_ptrs_k, %mask_k : tensor<32x64xf16>

      // 矩阵乘法累加
      %acc_new = tt.dot %a, %b, %acc {allowTF32 = true}
        : tensor<128x32xf16> * tensor<32x64xf16> -> tensor<128x64xf32>

      // 更新指针
      %a_ptrs_next = tt.addptr %a_ptrs_k, %k_step_offset
      %b_ptrs_next = tt.addptr %b_ptrs_k, %k_step_offset

      scf.yield %acc_new : tensor<128x64xf32>
    }

    // 存储结果
    tt.store %c_ptrs, %result : tensor<128x64x!tt.ptr<f32>>, tensor<128x64xf32>

    tt.return
  }
}
```

```bash
# 2. 运行 layout 推断 pass
triton-opt matmul_kernel.mlir --tritongpu-materialize -o ttg_output.mlir
```

输出（ttg 方言，带 layout）：

```mlir
// 布局推断后的 IR
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0,
                         warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#dot_op_0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dot_op_1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1],
                         warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.shared<{vec = 4, perPhase = 1, maxPhase = 4, order = [1, 0]}>

module attributes {"triton_gpu.num-warps" = 4 : i32} {
  tt.func @matmul_kernel(...) {
    // ... 加载 A 和 B 到 #blocked 布局
    %a_blocked = ttg.global_load %a_ptrs, %mask
      : tensor<128x32xf16, #blocked>
    %b_blocked = ttg.global_load %b_ptrs, %mask
      : tensor<32x64xf16, #blocked>

    // 存入共享内存
    %smem_a = ttg.local_alloc : !ttg.memdesc<128x32xf16, #shared, #smem>
    %smem_b = ttg.local_alloc : !ttg.memdesc<32x64xf16, #shared, #smem>
    ttg.local_store %a_blocked, %smem_a
      : tensor<128x32xf16, #blocked> -> !ttg.memdesc<128x32xf16, #shared, #smem>
    ttg.local_store %b_blocked, %smem_b
      : tensor<32x64xf16, #blocked> -> !ttg.memdesc<32x64xf16, #shared, #smem>

    // 从共享内存加载到 dot_op 布局
    %a_dotop = ttg.local_load %smem_a
      : !ttg.memdesc<128x32xf16, #shared, #smem> -> tensor<128x32xf16, #dot_op_0>
    %b_dotop = ttg.local_load %smem_b
      : !ttg.memdesc<32x64xf16, #shared, #smem> -> tensor<32x64xf16, #dot_op_1>

    // 矩阵乘法（现在操作数有正确的 layout）
    %acc_new = tt.dot %a_dotop, %b_dotop, %acc
      : tensor<128x32xf16, #dot_op_0> * tensor<32x64xf16, #dot_op_1>
      -> tensor<128x64xf32, #mma>

    // ... 存储结果（需要 layout conversion: #mma → #blocked → global）
    // ttg.convert_layout %result : tensor<128x64xf32, #mma> -> tensor<128x64xf32, #blocked>
    // ttg.global_store %c_ptrs, %result_blocked
  }
}
```

### 11.8.5 自定义 Pass 的添加方法

如果你想为 Triton 添加自定义的编译优化 Pass，需要以下步骤：

**Step 1: 在 Passes.td 中定义 Pass**

```tablegen
// include/triton/Dialect/TritonGPU/Transforms/Passes.td
def MyCustomPass : Pass<"tritongpu-my-custom"> {
  let summary = "My custom optimization pass";
  let description = [{
    This pass performs my custom optimization on the TritonGPU IR.
  }];

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect"
  ];

  let options = [
    Option<"myParam", "my-param", "int",
           /*default=*/"42", "My custom parameter">
  ];
}
```

**Step 2: 实现 Pass**

```cpp
// lib/Dialect/TritonGPU/Transforms/MyCustom.cpp
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_MYCUSTOMPASS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

struct MyCustomPass : public impl::MyCustomPassBase<MyCustomPass> {
  using impl::MyCustomPassBase<MyCustomPass>::MyCustomPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](Operation *op) {
      // 你的自定义逻辑
      if (auto dotOp = dyn_cast<triton::DotOp>(op)) {
        // 分析 dot 操作
        auto aType = dotOp.getA().getType().cast<RankedTensorType>();
        auto bType = dotOp.getB().getType().cast<RankedTensorType>();

        // 获取 layout encoding
        auto aEncoding = triton::gpu::getEncoding(aType);
        auto bEncoding = triton::gpu::getEncoding(bType);

        // 应用你的优化
        // ...
      }
    });
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
```

**Step 3: 注册 Pass**

```cpp
// 在 Pipeline.cpp 或适当的位置注册
void registerMyCustomPass() {
  PassRegistration<MyCustomPass>();
}
```

**Step 4: 测试 Pass**

```bash
# 使用 triton-opt 测试
triton-opt input.mlir --tritongpu-my-custom -mlir-print-ir-after-all

# 使用 lit 测试
# test/my_custom_pass.mlir:
// RUN: triton-opt %s --tritongpu-my-custom | FileCheck %s
// CHECK: expected_ir_pattern
```

### 11.8.6 调试技巧

```bash
# 1. 打印每个 pass 后的 IR（最常用的调试手段）
triton-opt input.mlir --tritongpu-materialize -mlir-print-ir-after-all

# 2. 只打印特定 pass 的前后 IR
triton-opt input.mlir --tritongpu-materialize \
  -mlir-print-ir-before=tritongpu-materialize \
  -mlir-print-ir-after=tritongpu-materialize

# 3. 将 IR 输出到文件
triton-opt input.mlir --tritongpu-materialize \
  -mlir-print-ir-after-all 2> ir_dump.mlir

# 4. 使用 MLIR 的 IR 打印控制
triton-opt input.mlir --tritongpu-to-llvm \
  -mlir-print-ir-after-all \
  -mlir-elide-elementsattrs-if-larger=16
  # -mlir-elide-elementsattrs-if-larger: 省略大的 dense 属性

# 5. 从 Python 中获取 IR
import os
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
os.environ["MLIR_ENABLE_DUMP"] = "1"  # 打印所有中间 IR

# 6. 使用 triton.compile 获取 IR
from triton.compiler import compile, ASTSource

src = ASTSource(fn=my_kernel, signature=..., constexprs=...)
compiled = compile(src)
# compiled.asm  包含 PTX 代码
# compiled.metadata  包含编译元信息
```

---

## 11.9 MLIR 在 Triton 编译流程中的位置

### 11.9.1 端到端编译流程

```python
# Triton 的端到端编译流程（从 Python 到 GPU 执行）

# 1. Python 前端
@triton.jit
def my_kernel(X, Y, Z, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets)
    y = tl.load(Y + offsets)
    tl.store(Z + offsets, x + y)

# 2. 调用 kernel
my_kernel[grid](x, y, z, BLOCK=256)

# 内部编译流程:
# ┌─────────────────────────────────────────────────────┐
# │  Python AST                                          │
# │    │                                                 │
# │    │  TritonASTToIR (triton/python/src/ir.cc)        │
# │    ▼                                                 │
# │  tt 方言 (Triton Dialect)                             │
# │    │                                                 │
# │    │  Pass Pipeline                                  │
# │    │  ├── tritongpu-coalesce                         │
# │    │  ├── tritongpu-materialize                      │
# │    │  ├── tritongpu-accelerate-matmul                │
# │    │  ├── tritongpu-optimize-dot-operands            │
# │    │  ├── tritongpu-decompose-conversions            │
# │    │  └── tritongpu-to-llvm                          │
# │    ▼                                                 │
# │  llvm 方言 (LLVM Dialect)                             │
# │    │                                                 │
# │    │  LLVM Translation                               │
# │    ▼                                                 │
# │  LLVM IR                                             │
# │    │                                                 │
# │    │  LLVM Backend (NVPTX / AMDGPU)                  │
# │    ▼                                                 │
# │  PTX / ROCm                                          │
# │    │                                                 │
# │    │  Driver (cuModuleLoad / hipModuleLoad)           │
# │    ▼                                                 │
# │  GPU 执行                                             │
# └─────────────────────────────────────────────────────┘
```

### 11.9.2 源码目录结构

```
triton/
├── include/triton/
│   ├── Dialect/
│   │   ├── Triton/              ← tt 方言定义
│   │   │   ├── IR/
│   │   │   │   ├── TritonOps.td         ← Operation 定义
│   │   │   │   ├── TritonDialect.td     ← Dialect 定义
│   │   │   │   ├── TritonTypes.td       ← Type 定义
│   │   │   │   └── TritonAttrDefs.td    ← Attribute 定义
│   │   │   └── Transform/
│   │   │       └── Passes.td            ← tt 级别的 Pass
│   │   ├── TritonGPU/           ← ttg 方言定义
│   │   │   ├── IR/
│   │   │   │   ├── Attrs.td             ← Layout Encoding 定义
│   │   │   │   ├── Ops.td               ← ttg Operation 定义
│   │   │   │   └── Types.td             ← ttg Type 定义
│   │   │   └── Transform/
│   │   │       └── Passes.td            ← ttg 级别的 Pass
│   │   └── TritonNvidiaGPU/     ← NVIDIA 特定方言
│   └── Conversion/
│       └── TritonGPUToLLVM/     ← ttg → llvm 转换 patterns
│
├── lib/
│   ├── Dialect/
│   │   ├── Triton/
│   │   │   ├── IR/
│   │   │   │   ├── Ops.cpp              ← Operation 实现
│   │   │   │   ├── Dialect.cpp          ← Dialect 注册
│   │   │   │   └── Types.cpp            ← Type 实现
│   │   │   └── Transform/
│   │   │       └── *.cpp                ← Pass 实现
│   │   ├── TritonGPU/
│   │   │   ├── IR/
│   │   │   │   ├── Attrs.cpp            ← Encoding 实现
│   │   │   │   ├── Ops.cpp              ← ttg Operation 实现
│   │   │   │   └── Dialect.cpp          ← Dialect 注册
│   │   │   └── Transform/
│   │   │       ├── Coalesce.cpp         ← 内存合并 pass
│   │   │       ├── Materialize.cpp      ← 布局推断 pass
│   │   │       ├── Pipeline.cpp         ← Pass pipeline
│   │   │       └── ...
│   │   └── TritonNvidiaGPU/
│   └── Conversion/
│       └── TritonGPUToLLVM/
│           ├── ConvertLayoutOpToLLVM.cpp  ← layout conversion
│           ├── DotOpToLLVM.cpp            ← MMA 指令选择
│           ├── LoadStoreOpToLLVM.cpp      ← load/store 指令选择
│           └── ...
│
└── bin/
    ├── triton-opt               ← MLIR IR 分析工具
    └── triton-translate         ← IR → PTX/ROCm 翻译工具
```

### 11.9.3 关键接口文件

```cpp
// include/triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h
// 定义了 Encoding 之间的关键接口

// SharedEncodingAttr 需要实现的接口
class SharedEncodingAttr : public ... {
  // 获取 bank conflict-free 的地址偏移
  unsigned getVec() const;
  unsigned getPerPhase() const;
  unsigned getMaxPhase() const;
};

// Encoding 之间的转换接口
// 从 source encoding 到 target encoding 的转换是否需要经过 shared memory
bool requiresSharedMemory(Type srcType, Type dstType);

// 获取寄存器布局信息
SmallVector<unsigned> getSizePerThread(Attribute encoding);
SmallVector<unsigned> getThreadsPerWarp(Attribute encoding);
SmallVector<unsigned> getWarpsPerCTA(Attribute encoding);
```

---

## 11.10 实战：用 MLIR IR 分析 Triton 编译决策

### 11.10.1 案例：理解为什么 Triton 选择特定的 BlockedEncoding

```python
# 假设我们有一个简单的向量加法 kernel
@triton.jit
def add_kernel(X, Y, Z, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask)
    y = tl.load(Y + offsets, mask=mask)
    tl.store(Z + offsets, x + y, mask=mask)
```

让我们分析 Triton 如何为这个 kernel 选择 `sizePerThread`：

```
分析过程：

输入:
  BLOCK = 256
  num_warps = 4

约束:
  1. 每个 warp 处理 256/4 = 64 个元素
  2. 每个 warp 有 32 个线程
  3. 每个线程需要处理 64/32 = 2 个元素

但是 Triton 倾向于使用向量加载（如 v4）来提高内存带宽利用率，
所以它可能选择 sizePerThread=4（每个线程加载 4 个元素）。

这意味着:
  每个线程: 4 个元素
  每个 warp: 4 × 32 = 128 个元素
  4 个 warp: 128 × 4 = 512 个元素

但是 BLOCK = 256，所以需要 wrap around:
  每个线程实际只用到 2 个元素（而不是 4 个）

Triton 的实际选择可能是:
  #blocked = #ttg.blocked<{
    sizePerThread = [4],   // 向量宽度为 4（利于 coalesce）
    threadsPerWarp = [32], // 每个 warp 32 个线程
    warpsPerCTA = [4],     // 4 个 warp
    order = [0]
  }>

  // coalesce pass 会将标量 load 合并为 v4 load
  // 但只有 256 个元素，每个线程 4 个，32×4×4=512 > 256
  // 实际加载时会用 mask 屏蔽多余的元素
```

### 11.10.2 案例：GEMM 中的 Layout 转换链

一个典型的 GEMM kernel 的数据流和 layout 转换：

```
┌─────────────────────────────────────────────────────────────────┐
│                GEMM 数据流的 Layout 转换链                        │
│                                                                 │
│  A 矩阵 (128×32xf16):                                           │
│  ┌───────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐ │
│  │ Global │───>│ blocked  │───>│ shared   │───>│ dot_op_0     │ │
│  │ (HBM)  │load│ (寄存器) │smem│ (SMEM)   │load│ (MMA operand)│ │
│  └───────┘    └──────────┘    └──────────┘    └──────────────┘ │
│                                                                 │
│  B 矩阵 (32×64xf16):                                           │
│  ┌───────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐ │
│  │ Global │───>│ blocked  │───>│ shared   │───>│ dot_op_1     │ │
│  │ (HBM)  │load│ (寄存器) │smem│ (SMEM)   │load│ (MMA operand)│ │
│  └───────┘    └──────────┘    └──────────┘    └──────────────┘ │
│                                                                 │
│  C/D 矩阵 (128×64xf32):                                        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │ blocked  │───>│ mma      │───>│ shared   │───>│ blocked  │ │
│  │ (初始化) │acc │ (TensorCore)│smem│ (SMEM)   │load│ (写回)    │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│                                                     │          │
│                                                     ▼          │
│                                                 ┌──────────┐  │
│                                                 │ Global   │  │
│                                                 │ (HBM)    │  │
│                                                 └──────────┘  │
│                                                                 │
│  转换次数统计:                                                    │
│  A: 1 load + 1 smem_store + 1 local_load = 3 操作               │
│  B: 1 load + 1 smem_store + 1 local_load = 3 操作               │
│  C: 1 smem_store + 1 local_load + 1 store = 3 操作              │
│  总共: 9 次数据搬运                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 本章小结

本章深入介绍了 MLIR 编译框架在 Triton 中的应用。核心要点如下：

1. **MLIR 设计哲学**：MLIR 通过可扩展的方言（Dialect）系统实现了多层中间表示。与传统编译器的"一步到位"不同，MLIR 采用渐进式 lowering（Progressive Lowering），每一层保留尽可能多的语义信息。核心概念包括 Dialect（方言）、Operation（操作）、Type（类型）和 Attribute（属性）。

2. **Triton 的四层方言栈**：Triton 构建了 tt → ttg → triton_gpu → llvm 四层方言栈。tt 层描述"做什么"（硬件无关的张量操作），ttg 层描述"怎么放"（内存布局与数据分块），triton_gpu 层描述"怎么算"（硬件指令选择），llvm 层描述"怎么跑"（机器表示）。

3. **tt 方言核心操作**：Triton Dialect 定义了一组精简的张量操作——`tt.load`/`tt.store`（内存访问）、`tt.dot`（矩阵乘法）、`tt.addptr`（指针算术）、`tt.make_range`（索引生成）等。这些操作直接对应 Python API 中的 `tl.load`/`tl.store`/`tl.dot`。

4. **Layout Encoding 机制**：ttg 方言的核心创新是 Layout Encoding。`#blocked` 描述寄存器中的基本数据分布，`#shared` 描述共享内存的 bank conflict-free 布局，`#dot_op` 和 `#mma` 描述 Tensor Core 的 operand/accumulator 布局。每个 Encoding 由 `sizePerThread`、`threadsPerWarp`、`warpsPerCTA` 等参数定义。

5. **类型系统**：`!tt.ptr<T>` 是 Triton 的指针类型，`tensor<NxT>` 在 tt 中无布局信息、在 ttg 中带有 Encoding，`!ttg.memdesc<T, encoding>` 是共享内存描述符。类型在方言转换过程中按照确定的规则映射。

6. **Attribute 系统**：Layout Attribute 通过 ODS 框架定义，描述线程到数据的映射关系。Layout 的传播遵循"使用者决定"原则——tt.dot 的输出必须是 `#mma`，operand 必须是 `#dot_op`，其他操作继承使用者的 layout。

7. **Dialect Conversion**：tt → ttg 的转换通过 `tritongpu-materialize` pass 完成布局推断；ttg → llvm 的转换通过 `tritongpu-to-llvm` pass 完成指令选择。复杂的 layout conversion 通过共享内存作为中介完成。

8. **工具链**：`triton-opt` 用于查看和分析 MLIR IR 的中间状态，`triton-translate` 用于将 MLIR IR 转换为 PTX/ROCm。环境变量 `TRITON_PRINT_AUTOTUNING=1` 和 `MLIR_ENABLE_DUMP=1` 是最常用的调试手段。

---

## 思考题

### 概念理解题

1. **方言层次的必要性**：为什么 Triton 需要 tt 和 ttg 两层方言，而不是直接从 Python 生成带 layout 的 IR？如果合并为一层会有什么问题？

2. **Layout Encoding 的抽象层级**：`#blocked`、`#shared`、`#dot_op` 和 `#mma` 分别描述了数据在什么物理介质上的布局？它们之间的转换为什么需要通过共享内存？

3. **sizePerThread 的选择**：对于一个 `tensor<1024xf32>` 的张量，在 4 个 warp 的 GPU 上，如果 `sizePerThread=[4]`、`threadsPerWarp=[32]`、`warpsPerCTA=[4]`，每个线程实际需要处理多少个元素？多余的元素如何处理？

4. **tt.dot 的语义**：`tt.dot` 操作在 tt 方言中是"做什么"，在 ttg 方言中被映射为 `#dot_op` + `#mma`。这两种表示的关键区别是什么？为什么不能在 tt 方言中直接指定 layout？

5. **Bank Conflict**：共享内存有 32 个 bank，如果 `#shared` 的 `perPhase=1, maxPhase=1`，在什么情况下会出现 bank conflict？`perPhase` 和 `maxPhase` 如何解决这个问题？

### 实践题

6. **IR 分析**：使用 `TRITON_PRINT_AUTOTUNING=1` 运行一个简单的 Triton kernel（如向量加法），导出 MLIR IR 并用 `triton-opt` 分析每个 pass 的效果。记录 IR 在每个阶段的变化。

7. **Encoding 参数调优**：对于一个 `tensor<64x64xf16>` 的矩阵乘法，尝试不同的 `sizePerThread` 和 `threadsPerWarp` 组合，分析它们对寄存器使用量和性能的影响。

8. **添加自定义 Pass**：基于 11.8.5 节的框架，实现一个简单的 Pass：遍历所有 `tt.dot` 操作，打印其 operand 的 layout encoding 和输出的 layout encoding。使用 `triton-opt` 测试。

9. **Layout Conversion 开销**：编写一个包含多个 elementwise 操作的 kernel，然后分析 IR 中 `ttg.convert_layout` 的数量。尝试优化 kernel 减少 layout conversion 的次数。

### 设计思考题

10. **新 Dialect 的设计**：如果要为 Transformer 的 Attention 操作设计一个新的方言层（介于 tt 和 ttg 之间），你会在其中定义哪些 Operation？这个方言层需要包含哪些 Layout Encoding？

11. **Layout 自动推断**：Triton 的 layout 推断算法是启发式的。设计一个更系统化的推断算法，能够全局优化 layout conversion 的总开销。需要考虑哪些约束和目标？

12. **跨硬件可移植性**：NVIDIA 的 Tensor Core 和 AMD 的 Matrix Core 有不同的指令形状和寄存器布局。Triton 如何通过 `#mma` 和 `#mfma` Encoding 的抽象来实现跨硬件可移植性？如果要支持 Intel 的 XMX，需要做哪些扩展？

13. **MLIR 与 CUDA 的对比**：MLIR 的方言栈和 CUDA 的编译流程（nvcc → ptxas → cubin）有什么异同？MLIR 的优势在哪里？在什么场景下 CUDA 的方式可能更合适？

### 进阶题

14. **编译时间分析**：Triton 的多层 lowering 需要多次遍历 IR，这会增加编译时间。分析 Triton 的 Pass Pipeline 中哪些 Pass 最耗时，以及如何通过 pass 融合（pass fusion）减少编译时间。

15. **内存占用优化**：在 LLM 推理中，KV Cache 的内存布局对性能至关重要。设计一个基于 Layout Encoding 的 KV Cache 内存布局方案，使得 Attention 计算中的 layout conversion 最少。

16. **Triton 编译器扩展**：假设你要为一个新的 AI 加速器（非 NVIDIA/AMD GPU）移植 Triton，你需要定义哪些新的 Dialect、Type 和 Attribute？列出需要修改的关键文件和添加的新组件。

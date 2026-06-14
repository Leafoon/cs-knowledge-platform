---
title: "Chapter 13: TritonGPU Dialect 与硬件映射"
description: "深入理解 Triton 编译器中 TritonGPU Dialect 的设计原理，包括 Encoding 属性体系（Blocked、MMA、Shared、DotOperand）、线程到数据的映射关系、布局转换机制，以及从抽象 tile 操作到具体硬件指令的完整映射过程"
date: "2026-06-11"
---

# Chapter 13: TritonGPU Dialect 与硬件映射

> **学习目标**：
> - 理解为什么 Triton 需要 TritonGPU Dialect 作为从抽象 tile 操作到硬件指令之间的中间表示层
> - 掌握 Encoding 属性体系的设计：BlockedEncodingAttr、MmaEncodingAttr、SharedEncodingAttr、DotOperandEncodingAttr 各自的职责与参数
> - 深入理解 BlockedEncoding 的线程到数据映射关系：sizePerThread、threadsPerWarp、warpsPerCta、order 的协同工作方式
> - 了解 MmaEncoding 如何将 Triton 的 dot 操作映射到 NVIDIA Tensor Core 的 mma.sync 指令和 AMD MFMA 指令
> - 理解 SharedEncoding 的共享内存布局优化策略，包括 bank conflict 避免和 vector width 设计
> - 掌握 triton_gpu.convert_layout 操作的原理与开销，理解不同 Encoding 之间的转换机制

---

## 13.1 为什么需要 TritonGPU Dialect

### 13.1.1 从抽象到具体的鸿沟

在前面的章节中，我们学习了如何用 Triton 编写高性能 kernel。当你写下 `tl.dot(a, b)` 时，这个操作是高度抽象的——它描述的是"对两个 tile 做矩阵乘法"，但完全没有指定：

- 数据如何分布到各个线程？
- 使用哪种硬件指令（CUDA Core 还是 Tensor Core）？
- 中间结果存储在寄存器还是共享内存？
- 如何处理 bank conflict？

这就是 **TritonGPU Dialect** 存在的意义。它是 Triton 编译器中连接"用户意图"和"硬件执行"的桥梁层。

```
Triton 编译管线中的 Dialect 演进:

┌─────────────────────────────────────────────────────────────┐
│                    Triton Source (Python)                    │
│   tl.dot(a, b)  ← 用户写的抽象 tile 操作                    │
└────────────────────────┬────────────────────────────────────┘
                         │  Frontend (triton.tools.aot / AST → IR)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Triton Dialect (TTIR)                     │
│   tt.dot %a, %b, %c  ← 仍抽象，无硬件信息                    │
│   tt.load %ptr, ...   ← 不知数据在线程间如何分布              │
└────────────────────────┬────────────────────────────────────┘
                         │  TritonGPU Conversion Pass
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                TritonGPU Dialect (TTGIR)                     │
│   triton_gpu.dot %a, %b, %c                                 │
│     {layoutA = #mma, layoutB = #blocked}                    │
│   triton_gpu.convert_layout %x                              │
│     {from = #blocked, to = #shared}                         │
│   ← 每个 SSA value 都带有 encoding 属性                      │
└────────────────────────┬────────────────────────────────────┘
                         │  Lower to LLVM / NVVM
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLVM Dialect / NVVM                       │
│   nvvm.mma.sync  ← 真实的硬件指令                            │
│   llvm.load       ← 带有正确的地址计算                        │
└─────────────────────────────────────────────────────────────┘
```

### 13.1.2 Encoding 的核心概念

**Encoding** 是 TritonGPU Dialect 的灵魂。它是一个 MLIR Attribute，附加在每个 SSA value 上，描述该 value 的数据如何分布到 GPU 的线程、warp 和 CTA 中。

```mlir
// 一个带有 BlockedEncoding 的 tensor 类型
// 表示一个 [128, 64] 的 tensor，按特定方式分布到线程中
%a = ... : tensor<128x64xf16, #blocked>

// 同样的 tensor，但使用 MMA encoding（用于 Tensor Core）
%b = ... : tensor<128x64xf16, #mma>

// 同样的 tensor，存储在共享内存中
%c = ... : tensor<128x64xf16, #shared>
```

没有 Encoding 的 tensor 类型（如 `tensor<128x64xf16>`）在 TritonGPU Dialect 中是非法的——每个 tensor 必须明确其数据布局。

### 13.1.3 Encoding 解决了什么问题

| 问题 | 没有 Encoding | 有 Encoding |
|------|-------------|------------|
| 数据如何分布到线程？ | 编译器猜测 | 明确声明 |
| 哪些线程持有哪些元素？ | 不确定 | 由编码规则确定 |
| 何时需要数据搬运？ | 运行时才知道 | 编译时可分析 |
| 如何选择硬件指令？ | 无法指定 | Encoding 类型决定 |

让我们通过一个具体的例子来感受 Encoding 的作用：

```mlir
// 没有 Encoding 的 IR（Triton Dialect）——抽象的
func.func @matmul(%a: tensor<128x64xf16>, %b: tensor<64x128xf16>,
                   %c: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %d = tt.dot %a, %b, %c : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
  return %d : tensor<128x128xf32>
}

// 有 Encoding 的 IR（TritonGPU Dialect）——具体的
// 每个操作数都有明确的硬件映射
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4],
                                 warpsPerCta = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCta = [4, 1]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>

func.func @matmul(%a: tensor<128x64xf16, #blocked>,
                   %b: tensor<64x128xf16, #blocked>,
                   %c: tensor<128x128xf32, #mma>) -> tensor<128x128xf32, #mma> {
  // 加载到共享内存
  %a_shared = triton_gpu.convert_layout %a : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #shared>
  %b_shared = triton_gpu.convert_layout %b : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #shared>

  // 从共享内存转换到 dot operand 布局
  %a_dot = triton_gpu.convert_layout %a_shared : tensor<128x64xf16, #shared> -> tensor<128x64xf16, #dot_operand_a>
  %b_dot = triton_gpu.convert_layout %b_shared : tensor<64x128xf16, #shared> -> tensor<64x128xf16, #dot_operand_b>

  // 执行矩阵乘法——操作数布局已匹配 Tensor Core 要求
  %d = tt.dot %a_dot, %b_dot, %c : tensor<128x64xf16, #dot_operand_a> * tensor<64x128xf16, #dot_operand_b> -> tensor<128x128xf32, #mma>

  return %d : tensor<128x128xf32, #mma>
}
```

<div data-component="EncodingVisualizer"></div>

[组件：EncodingVisualizer - 可视化展示同一数据在不同 Encoding 下的线程分布方式]

---

## 13.2 Encoding 属性体系总览

### 13.2.1 四大 Encoding 类型

TritonGPU Dialect 定义了四种核心 Encoding 属性，每种对应硬件的一个层次：

```mlir
// TritonGPU 的 Encoding 属性定义在:
// include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td

// 1. BlockedEncoding — 通用的分块布局，映射到寄存器
#blocked = #triton_gpu.blocked<{
  sizePerThread = [4, 4],      // 每个线程处理的元素数
  threadsPerWarp = [8, 4],     // 每个 warp 中的线程排列
  warpsPerCta = [4, 1],        // 每个 CTA 中的 warp 排列
  order = [1, 0]               // 维度优先级（列优先）
}>

// 2. MmaEncoding — Tensor Core / MFMA 矩阵乘法布局
#mma = #triton_gpu.nvidia_mma<{
  versionMajor = 2,            // mma.sync 版本
  versionMinor = 0,            // 子版本
  warpsPerCta = [4, 1],        // warp 在 CTA 中的分布
  CTAsPerCGA = [1, 1],         // CTA 在 CGA 中的分布（Hopper+）
  CTASplitNum = [1, 1],        // CTA 的分割方式
  CTAOrder = [1, 0]            // CTA 的排列顺序
}>

// 3. SharedEncoding — 共享内存布局
#shared = #triton_gpu.shared<{
  vec = 16,                    // 向量化访问宽度
  perPhase = 1,                // 每阶段的行数
  maxPhase = 8,                // 最大阶段数
  order = [1, 0],              // 维度优先级
  hasLeadingOffset = false      // 是否有前导偏移（Hopper+）
}>

// 4. DotOperandEncoding — 点积操作数布局
#dot_op = #triton_gpu.dot_op<{
  opIdx = 0,                   // 操作数索引（0=A, 1=B）
  parent = #mma,               // 父 Encoding（通常是 MmaEncoding）
  kWidth = 2                   // K 维度的向量宽度
}>
```

### 13.2.2 Encoding 的继承与组合关系

四种 Encoding 之间存在明确的层级关系：

```
Encoding 属性层级:

                ┌─────────────────┐
                │  DotOperand     │──引用──┐
                │  EncodingAttr   │        │
                └─────────────────┘        ▼
                                    ┌─────────────┐
                ┌─────────────────┐ │  Mma         │
                │  Shared         │ │  EncodingAttr│
                │  EncodingAttr   │ └─────────────┘
                └─────────────────┘
                        ▲
                        │ 数据从 shared 加载到 register
                        │
                ┌─────────────────┐
                │  Blocked        │
                │  EncodingAttr   │
                └─────────────────┘

典型数据流:
  Global Memory → Blocked (register) → Shared (smem) → DotOperand (register) → MMA unit
```

### 13.2.3 Encoding 属性的源码定义位置

```
Encoding 属性的 MLIR TableGen 定义:
├── include/triton/Dialect/TritonGPU/IR/
│   ├── TritonGPUAttrDefs.td        ← 属性的 TableGen 定义
│   ├── TritonGPUBase.td            ← 基础类型定义
│   └── TritonGPUOps.td             ← 操作定义
├── lib/Dialect/TritonGPU/IR/
│   ├── TritonGPUAttrDefs.cpp       ← 属性的 C++ 实现
│   ├── TritonGPUDialect.cpp        ← Dialect 注册
│   └── TritonGPUOps.cpp            ← 操作实现
└── lib/Dialect/TritonGPU/Transforms/
    ├── AccelerateMatmul.cpp         ← 注入 MmaEncoding
    ├── Coalesce.cpp                 ← 优化 BlockedEncoding
    ├── OptimizeDotOperands.cpp      ← 优化 DotOperand 布局
    └── Pipeline.cpp                 ← 软件流水线优化
```

<div data-component="EncodingHierarchy"></div>

[组件：EncodingHierarchy - 交互式展示四种 Encoding 的参数、约束和转换关系]

---

## 13.3 BlockedEncoding：通用分块布局

### 13.3.1 核心参数详解

BlockedEncoding 是最基础的 Encoding 类型，它定义了 tensor 数据如何分布到 GPU 的线程网格中。理解它需要理解四个核心参数：

```mlir
#blocked = #triton_gpu.blocked<{
  sizePerThread    = [4, 4],    // S: 每个线程连续处理的元素数
  threadsPerWarp   = [8, 4],    // T: 每个 warp 中线程的二维排列
  warpsPerCta      = [4, 1],    // W: 每个 CTA 中 warp 的二维排列
  order            = [1, 0]     // 维度遍历顺序（[1,0] = 列优先）
}>
```

让我们逐一分析每个参数的含义。

### 13.3.2 sizePerThread（每线程元素数）

`sizePerThread` 指定每个线程"持有"多少个连续元素。这直接决定了：
- 每个线程需要多少寄存器来存储数据
- 向量化内存访问的宽度

```
sizePerThread = [4, 2] 的含义:

原始 tensor [8, 8]:
  ┌───┬───┬───┬───┬───┬───┬───┬───┐
  │0,0│0,1│0,2│0,3│0,4│0,5│0,6│0,7│  行 0
  ├───┼───┼───┼───┼───┼───┼───┼───┤
  │1,0│1,1│1,2│1,3│1,4│1,5│1,6│1,7│  行 1
  ├───┼───┼───┼───┼───┼───┼───┼───┤
  │2,0│2,1│2,2│2,3│2,4│2,5│2,6│2,7│  行 2
  ├───┼───┼───┼───┼───┼───┼───┼───┤
  │...│...│...│...│...│...│...│...│
  └───┴───┴───┴───┴───┴───┴───┴───┘

sizePerThread = [4, 2] 表示:
  - 每个线程在"行"方向持有 4 个连续元素
  - 每个线程在"列"方向持有 2 个连续元素
  - 每个线程总共持有 4 × 2 = 8 个元素
  - 需要 8 个寄存器（per value，具体取决于数据类型）
```

### 13.3.3 threadsPerWarp（每 warp 线程数）

`threadsPerWarp` 定义了 warp 内 32 个线程的二维排列方式。其乘积必须等于 32（NVIDIA GPU）或 64（AMD GPU）。

```
threadsPerWarp = [8, 4] 的含义:

Warp 内的线程排列:
  ┌────┬────┬────┬────┐
  │ T0 │ T1 │ T2 │ T3 │   线程行 0
  ├────┼────┼────┼────┤
  │ T4 │ T5 │ T6 │ T7 │   线程行 1
  ├────┼────┼────┼────┤
  │ T8 │ T9 │T10 │T11 │   线程行 2
  ├────┼────┼────┼────┤
  │T12 │T13 │T14 │T15 │   线程行 3
  ├────┼────┼────┼────┤
  │T16 │T17 │T18 │T19 │   线程行 4
  ├────┼────┼────┼────┤
  │T20 │T21 │T22 │T23 │   线程行 5
  ├────┼────┼────┼────┤
  │T24 │T25 │T26 │T27 │   线程行 6
  ├────┼────┼────┼────┤
  │T28 │T29 │T30 │T31 │   线程行 7
  └────┴────┴────┴────┘
  8 行 × 4 列 = 32 线程 ✓
```

### 13.3.4 warpsPerCta（每 CTA warp 数）

`warpsPerCta` 定义了 CTA 内多个 warp 的二维排列。其乘积等于 CTA 中的 warp 总数。

```
warpsPerCta = [4, 2] 的含义:

CTA 内的 warp 排列:
  ┌───────┬───────┐
  │ Warp0 │ Warp1 │    Warp 行 0
  ├───────┼───────┤
  │ Warp2 │ Warp3 │    Warp 行 1
  ├───────┼───────┤
  │ Warp4 │ Warp5 │    Warp 行 2
  ├───────┼───────┤
  │ Warp6 │ Warp7 │    Warp 行 3
  └───────┴───────┘
  4 行 × 2 列 = 8 warps per CTA
```

### 13.3.5 综合映射：线程到数据的完整关系

将三个参数组合起来，我们可以计算出完整的线程→数据映射：

```
完整映射: sizePerThread=[4,2], threadsPerWarp=[8,4], warpsPerCta=[4,2]

Tensor 形状计算:
  dim0 = sizePerThread[0] × threadsPerWarp[0] × warpsPerCta[0]
       = 4 × 8 × 4 = 128

  dim1 = sizePerThread[1] × threadsPerWarp[1] × warpsPerCta[1]
       = 2 × 4 × 2 = 16

  → 可表示 128×16 的 tensor

单个线程负责的元素 (以线程 T5 为例, warp 内位置 = [1,1]):
  线程在 warp 内的坐标: row=1, col=1
  线程在 CTA 内的坐标 (假设 warp0):
    dim0_base = warp_row × threadsPerWarp[0] + thread_row
              = 0 × 8 + 1 = 1
    dim1_base = warp_col × threadsPerWarp[1] + thread_col
              = 0 × 4 + 1 = 1

  线程 T5 持有的元素:
    dim0: [1, 5, 9, 13]  (每 8 行一个周期, sizePerThread[0]=4 个)
    dim1: [1, 3]          (每 4 列一个周期, sizePerThread[1]=2 个)

  → T5 持有 (1,1), (1,3), (5,1), (5,3), (9,1), (9,3), (13,1), (13,3)
     共 8 个元素
```

### 13.3.6 order 参数的影响

`order` 参数控制维度的优先级，类似于 NumPy 的 memory layout：

```mlir
// 列优先 (order = [1, 0]) — 数据按列连续存储
#blocked_col = #triton_gpu.blocked<{
  sizePerThread = [4, 4], threadsPerWarp = [8, 4],
  warpsPerCta = [4, 1], order = [1, 0]
}>

// 行优先 (order = [0, 1]) — 数据按行连续存储
#blocked_row = #triton_gpu.blocked<{
  sizePerThread = [4, 4], threadsPerWarp = [8, 4],
  warpsPerCta = [4, 1], order = [0, 1]
}>
```

```
order = [1, 0] (列优先) 的数据分布:

线程 T0 持有的元素按列排列:
  (0,0) (0,1) (0,2) (0,3)
  (8,0) (8,1) (8,2) (8,3)
  (16,0) (16,1) (16,2) (16,3)
  (24,0) (24,1) (24,2) (24,3)

order = [0, 1] (行优先) 的数据分布:

线程 T0 持有的元素按行排列:
  (0,0) (0,1) (0,2) (0,3)
  (1,0) (1,1) (1,2) (1,3)
  (2,0) (2,1) (2,2) (2,3)
  (3,0) (3,1) (3,2) (3,3)

→ order 决定了连续元素在哪个维度上"步进"
```

### 13.3.7 BlockedEncoding 的实际 IR 示例

```mlir
// 定义一个 BlockedEncoding
#blocked = #triton_gpu.blocked<{
  sizePerThread = [1, 4],
  threadsPerWarp = [8, 4],
  warpsPerCta = [4, 1],
  order = [1, 0]
}>

// 使用该 Encoding 的函数
func.func @elementwise_add(%a: tensor<128x64xf16, #blocked>,
                             %b: tensor<128x64xf16, #blocked>)
                             -> tensor<128x64xf16, #blocked> {
  // tt.add 操作要求两个操作数的 encoding 相同
  %c = arith.addf %a, %b : tensor<128x64xf16, #blocked>
  return %c : tensor<128x64xf16, #blocked>
}
```

<div data-component="BlockedEncodingMapper"></div>

[组件：BlockedEncodingMapper - 交互式可视化：调整四个参数，观察线程到数据映射的变化]

---

## 13.4 MmaEncoding：Tensor Core 硬件映射

### 13.4.1 NVIDIA Tensor Core 的 mma.sync 指令

NVIDIA 的 Tensor Core 是专用的矩阵乘法硬件单元。从 Volta 架构开始引入，每代都有不同的指令和约束：

| 架构 | 计算能力 | 指令 | 支持的数据类型 | 典型形状 |
|------|---------|------|-------------|---------|
| Volta | 7.0 | wmma.mma.sync | FP16 | 16×16×16 |
| Turing | 7.5 | wmma.mma.sync | FP16/INT8 | 16×16×16 |
| Ampere | 8.0 | mma.sync | FP16/BF16/TF32/INT8 | 16×8×16, 16×8×8 |
| Hopper | 9.0 | wgmma.mma | FP16/BF16/FP8 | 64×N×16 |

### 13.4.2 MmaEncodingAttr 的参数

```mlir
// NVIDIA MMA Encoding
#mma = #triton_gpu.nvidia_mma<{
  versionMajor = 2,          // Tensor Core 版本 (1=Volta, 2=Ampere, 3=Hopper)
  versionMinor = 0,          // 子版本
  warpsPerCta = [4, 1],      // warp 在 CTA 中的分布
  CTAsPerCGA = [1, 1],       // CTA 在 CGA 中的分布 (Hopper+)
  CTASplitNum = [1, 1],      // CTA 分割数
  CTAOrder = [1, 0]          // CTA 排列顺序
}>
```

**versionMajor 的含义：**

```mlir
// Version 1: Volta 时代的 wmma 指令
//   - 每个 warp 独立执行一个 16×16×16 的矩阵乘法
//   - 操作数需要按特定模式分布在 warp 的 32 个线程中
#mma_v1 = #triton_gpu.nvidia_mma<{versionMajor = 1, versionMinor = 0,
                                    warpsPerCta = [4, 1]}>

// Version 2: Ampere 的 mma.sync 指令
//   - 支持更多数据类型和形状
//   - 指令粒度更灵活: 16×8×8, 16×8×16, 16×8×32 等
#mma_v2 = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0,
                                    warpsPerCta = [4, 1]}>

// Version 3: Hopper 的 wgmma (Warp Group MMA) 指令
//   - 以 warp group (4 个 warp) 为执行单位
//   - 异步执行，支持更深层次的流水线
//   - 支持 FP8 数据类型
#mma_v3 = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0,
                                    warpsPerCta = [4, 1]}>
```

### 13.4.3 mma.sync 指令的操作数布局

以 Ampere 的 `mma.sync.aligned.m16n8k16` 指令为例，操作数需要按特定模式分布到 warp 的 32 个线程中：

```
mma.sync m16n8k16.f16.f16 的操作数布局:

操作数 A (16×16, FP16):
  每个线程持有 8 个元素 (2 个寄存器，每寄存器 4 个 FP16)
  线程 0-7:   持有行 0-1 的元素
  线程 8-15:  持有行 2-3 的元素
  线程 16-23: 持有行 4-5 的元素（实际映射到行 8-9）
  线程 24-31: 持有行 6-7 的元素（实际映射到行 10-11）

  具体映射 (线程 → 元素):
  ┌─────┬──────────────────────────────────┐
  │ T0  │ a[0][0:4], a[1][0:4]            │
  │ T1  │ a[0][4:8], a[1][4:8]            │
  │ T2  │ a[0][8:12], a[1][8:12]          │
  │ T3  │ a[0][12:16], a[1][12:16]        │
  │ T4  │ a[2][0:4], a[3][0:4]            │
  │ ... │ ...                              │
  │ T31 │ a[14][12:16], a[15][12:16]      │
  └─────┴──────────────────────────────────┘

操作数 B (16×8, FP16):
  每个线程持有 4 个元素
  线程 0-3:   持有列 0-1 的元素
  线程 4-7:   持有列 2-3 的元素
  ...（重复模式）

输出 C/D (16×8, FP32):
  每个线程持有 4 个 FP32 元素
  分布模式与 B 类似
```

### 13.4.4 MmaEncoding 的 IR 表示

```mlir
// 定义 MmaEncoding
#mma = #triton_gpu.nvidia_mma<{
  versionMajor = 2,
  versionMinor = 0,
  warpsPerCta = [4, 1]
}>

// 使用 MmaEncoding 的 tensor 类型
// 注意: MmaEncoding 通常用于 dot 操作的输出
func.func @matmul_with_mma(
    %a: tensor<128x32xf16, #dot_operand_a>,  // 操作数 A
    %b: tensor<32x128xf16, #dot_operand_b>,  // 操作数 B
    %c: tensor<128x128xf32, #mma>            // 累加器（MMA 布局）
) -> tensor<128x128xf32, #mma> {
  // tt.dot 的输出自动获得 MMA encoding
  %d = tt.dot %a, %b, %c
    : tensor<128x32xf16, #dot_operand_a>
    * tensor<32x128xf16, #dot_operand_b>
    -> tensor<128x128xf32, #mma>
  return %d : tensor<128x128xf32, #mma>
}
```

### 13.4.5 AMD MFMA Encoding

AMD GPU 使用 MFMA (Matrix Fused Multiply-Add) 指令，对应不同的参数：

```mlir
// AMD MFMA Encoding
#mfma = #triton_gpu.amd_mma<{
  versionMajor = 2,          // MFMA 版本
  versionMinor = 0,
  warpsPerCta = [4, 1],
  isTranspose = false         // 是否转置操作数布局
}>

// MFMA 指令的典型形状:
//   MFMA_32x32x8_F16:  32×32 输出, K=8
//   MFMA_16x16x16_F16: 16×16 输出, K=16
//   MFMA_32x32x16_F16: 32×32 输出, K=16 (CDNA 2+)
```

AMD MFMA 与 NVIDIA MMA 的关键差异：

```
NVIDIA mma.sync vs AMD MFMA:

┌────────────────────┬───────────────────────┬───────────────────────┐
│ 特性               │ NVIDIA mma.sync       │ AMD MFMA              │
├────────────────────┼───────────────────────┼───────────────────────┤
│ 执行粒度           │ Warp (32 threads)     │ Wavefront (64 threads)│
│ 典型输出形状       │ 16×8                  │ 32×32 或 16×16        │
│ K 维度             │ 16 或 32              │ 8 或 16               │
│ 操作数布局         │ 复杂的线程映射         │ 相对简单的线程映射     │
│ 数据类型支持       │ FP16/BF16/TF32/INT8   │ FP16/BF16/FP32/INT8   │
│ 异步执行           │ Volta+ (wmma)         │ CDNA 2+ (MFMA.async)  │
└────────────────────┴───────────────────────┴───────────────────────┘

MFMA_32x32x8 的操作数布局 (AMD, 64 threads):
  操作数 A: 每个线程持有 4 个 FP16 元素
    线程 0-31: 持有 A 矩阵的前 32 行
    线程 32-63: 持有 A 矩阵的后 32 行

  操作数 B: 每个线程持有 4 个 FP16 元素
    线程 0-31: 持有 B 矩阵的前 32 列
    线程 32-63: 持有 B 矩阵的后 32 列

  输出 C: 每个线程持有 16 个 FP32 元素
    32×32 输出 / 64 线程 = 16 元素/线程
```

### 13.4.6 Tensor Core 与 CUDA Core 的对比

| 特性 | CUDA Core | Tensor Core |
|------|-----------|------------|
| 计算单元 | 标量 ALU | 矩阵乘法单元 |
| 操作 | 逐元素 | 矩阵块乘法 |
| 吞吐量 (A100) | 19.5 TFLOPS (FP32) | 312 TFLOPS (FP16) |
| 数据布局 | 无特殊要求 | 严格的线程到数据映射 |
| 编程模型 | 简单 | 需要 Encoding 配合 |

<div data-component="TensorCoreLayout"></div>

[组件：TensorCoreLayout - 可视化 mma.sync 指令的操作数分布模式，展示线程如何持有和传递操作数]

---

## 13.5 SharedEncoding：共享内存布局

### 13.5.1 共享内存的 Bank 结构

GPU 的共享内存被组织为 32 个 bank（NVIDIA），每个 bank 4 字节宽。当 32 个线程同时访问不同的 bank 时，可以并行完成；但如果有多个线程访问同一个 bank，就会发生 **bank conflict**，导致串行化。

```
共享内存 Bank 结构 (NVIDIA GPU):

Bank 0: [0x00-0x03] [0x80-0x83] [0x100-0x103] ...
Bank 1: [0x04-0x07] [0x84-0x87] [0x104-0x107] ...
Bank 2: [0x08-0x0B] [0x88-0x8B] [0x108-0x10B] ...
...
Bank 31: [0x7C-0x7F] [0xFC-0xFF] [0x17C-0x17F] ...

Bank conflict 示例 (3-way conflict):
  线程 0 → Bank 0, addr 0x00
  线程 1 → Bank 0, addr 0x80  ← 同一 bank!
  线程 2 → Bank 0, addr 0x100 ← 同一 bank!
  → 需要 3 个周期完成，吞吐降为 1/3
```

### 13.5.2 SharedEncodingAttr 的参数

```mlir
#shared = #triton_gpu.shared<{
  vec = 8,              // 向量化访问的元素数
  perPhase = 1,         // 每个 phase 覆盖的行数
  maxPhase = 8,         // 总 phase 数
  order = [1, 0],       // 维度优先级
  hasLeadingOffset = false  // 是否使用 leading offset (Hopper+)
}>
```

**参数详解：**

```
vec (向量宽度):
  - 控制一次共享内存事务访问的连续元素数
  - 较大的 vec → 更高的带宽利用率
  - 受限于数据类型: vec × sizeof(element) ≤ 16 bytes
  - FP16 数据: 最大 vec = 8 (8 × 2 = 16 bytes)

perPhase (每阶段行数):
  - 控制 bank 偏移的"周期"
  - 每 perPhase 行使用相同的 bank 偏移
  - 典型值: 1, 2, 4

maxPhase (最大阶段数):
  - bank 偏移循环的总阶段数
  - 偏移模式: phase_id = (row / perPhase) % maxPhase
  - bank_offset = phase_id × (vec / bank_width)

order (维度优先级):
  - [1, 0]: 列优先，连续元素沿列方向排列
  - [0, 1]: 行优先，连续元素沿行方向排列
  - 应与数据的访问模式匹配
```

### 13.5.3 Bank Conflict 避免策略

SharedEncoding 通过精心设计的偏移模式来避免 bank conflict：

```
以 FP16 数据, shape=[8, 8], vec=8, perPhase=1, maxPhase=8, order=[1,0] 为例:

行 0: offset = 0  → Bank 0-7
行 1: offset = 1  → Bank 2-9  (偏移 1 个 bank)
行 2: offset = 2  → Bank 4-11
行 3: offset = 3  → Bank 6-13
行 4: offset = 4  → Bank 8-15
行 5: offset = 5  → Bank 10-17
行 6: offset = 6  → Bank 12-19
行 7: offset = 7  → Bank 14-21

→ 每行的起始 bank 不同，当 warp 中的线程按列访问时
  （每个线程访问不同行的同一列位置），
  不同线程访问不同的 bank → 无 conflict！

无 bank conflict 的条件:
  对于 warp 中 32 个线程同时访问共享内存的场景，
  所有被访问的地址必须落在不同的 bank 中。
```

### 13.5.4 hasLeadingOffset 属性（Hopper+）

在 Hopper 架构上，共享内存支持 `ldmatrix` 指令的 leading offset 模式：

```mlir
// 没有 leading offset (Ampere 及更早)
#shared_no_offset = #triton_gpu.shared<{
  vec = 8, perPhase = 1, maxPhase = 8,
  order = [1, 0], hasLeadingOffset = false
}>

// 有 leading offset (Hopper+)
//   - 支持更高效的数据重排
//   - 与 wgmma 指令配合使用
#shared_with_offset = #triton_gpu.shared<{
  vec = 8, perPhase = 1, maxPhase = 8,
  order = [1, 0], hasLeadingOffset = true
}>
```

### 13.5.5 SharedEncoding 的 IR 示例

```mlir
#shared = #triton_gpu.shared<{
  vec = 8, perPhase = 1, maxPhase = 8,
  order = [1, 0]
}>

func.func @shared_memory_example(
    %a: tensor<128x64xf16, #blocked>
) -> tensor<128x64xf16, #shared> {
  // convert_layout: 从寄存器布局转到共享内存布局
  // 编译器会生成: store → __syncthreads() → load
  %a_shared = triton_gpu.convert_layout %a
    : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #shared>
  return %a_shared : tensor<128x64xf16, #shared>
}

// 编译器生成的伪 PTX:
//   // 写入共享内存
//   st.shared.v4.b32 [smem_addr], {r0, r1, r2, r3}
//   bar.sync 0                          // __syncthreads()
//   // 从共享内存读取（带 bank 偏移）
//   ld.shared.v4.b32 {r4, r5, r6, r7}, [smem_addr + offset]
```

<div data-component="BankConflictVisualizer"></div>

[组件：BankConflictVisualizer - 交互式展示不同 SharedEncoding 参数下的 bank conflict 情况]

---

## 13.6 DotOperandEncoding：点积操作数布局

### 13.6.1 为什么需要 DotOperandEncoding

Tensor Core 的 mma.sync 指令对操作数的布局有严格要求。操作数不能以任意布局传入，必须按照硬件规定的线程到数据映射来排列。DotOperandEncoding 就是描述这种特定布局的 Encoding。

```mlir
#dot_operand_a = #triton_gpu.dot_op<{
  opIdx = 0,        // 操作数索引: 0 = A, 1 = B
  parent = #mma,     // 父 Encoding: 必须是 MmaEncoding
  kWidth = 2         // K 维度上的向量宽度
}>
```

### 13.6.2 opIdx 的含义

```
矩阵乘法 C = A × B 的操作数分配:

  A (M×K)  ─┐
             ├──→  Tensor Core  ──→  C (M×N)
  B (K×N)  ─┘

  opIdx = 0: 操作数 A (左侧矩阵)
    - A 的列数 (K) 需要与 B 的行数匹配
    - A 按 M 行和 K 列分布到线程

  opIdx = 1: 操作数 B (右侧矩阵)
    - B 按 K 行和 N 列分布到线程
    - 布局与 A 不同（因为矩阵乘法不对称）
```

### 13.6.3 parent 属性

DotOperandEncoding 的 `parent` 属性指向它所服务的 MmaEncoding。这建立了明确的依赖关系：

```mlir
// MmaEncoding 定义了硬件能力
#mma = #triton_gpu.nvidia_mma<{
  versionMajor = 2, versionMinor = 0,
  warpsPerCta = [4, 1]
}>

// DotOperandEncoding 引用 MmaEncoding 作为父节点
#dot_a = #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>
#dot_b = #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>

// 在 dot 操作中使用
%d = tt.dot %a, %b, %c
  : tensor<128x32xf16, #dot_a>    // A: parent=#mma, opIdx=0
  * tensor<32x128xf16, #dot_b>    // B: parent=#mma, opIdx=1
  -> tensor<128x128xf32, #mma>    // C: 使用 #mma 本身
```

### 13.6.4 kWidth 参数

`kWidth` 指定 K 维度上的向量宽度，影响操作数在 K 维度上的分块方式：

```mlir
// kWidth = 2: 每次处理 K 维度上的 2 个元素
// 适用于 FP16 (2 × FP16 = 4 bytes = 一个 32-bit 寄存器)
#dot_a_k2 = #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>

// kWidth = 4: 每次处理 K 维度上的 4 个元素
// 适用于 INT8 (4 × INT8 = 4 bytes = 一个 32-bit 寄存器)
#dot_a_k4 = #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
```

kWidth 的选择与数据类型密切相关：

| 数据类型 | sizeof(element) | 推荐 kWidth | kWidth × sizeof |
|---------|----------------|------------|----------------|
| FP32 | 4 | 1 | 4 |
| FP16 / BF16 | 2 | 2 | 4 |
| INT8 | 1 | 4 | 4 |
| FP8 | 1 | 4 | 4 |

### 13.6.5 DotOperandEncoding 与其他 Encoding 的关系

```
数据流中的 DotOperandEncoding:

  Global Memory
       │
       ▼
  ┌──────────────┐
  │   Blocked     │  ← tt.load 加载数据到寄存器
  │   Encoding    │     每个线程按 blocked 布局持有数据
  └──────┬───────┘
         │  convert_layout: Blocked → Shared
         ▼
  ┌──────────────┐
  │   Shared      │  ← 数据写入共享内存
  │   Encoding    │     按 shared 布局存储，避免 bank conflict
  └──────┬───────┘
         │  convert_layout: Shared → DotOperand
         ▼
  ┌──────────────┐
  │ DotOperand    │  ← 从共享内存加载到寄存器
  │ Encoding      │     按 Tensor Core 要求的布局排列
  └──────┬───────┘
         │  tt.dot
         ▼
  ┌──────────────┐
  │   Mma         │  ← Tensor Core 执行矩阵乘法
  │   Encoding    │     输出按 MMA 布局存储
  └──────────────┘
```

### 13.6.6 DotOperandEncoding 的 IR 完整示例

```mlir
#blocked = #triton_gpu.blocked<{
  sizePerThread = [1, 4], threadsPerWarp = [8, 4],
  warpsPerCta = [4, 1], order = [1, 0]
}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0,
                                 warpsPerCta = [4, 1]}>
#shared = #triton_gpu.shared<{
  vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]
}>
#dot_a = #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dot_b = #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

func.func @dot_operand_flow(
    %A: tensor<128x64xf16, #blocked>,
    %B: tensor<64x128xf16, #blocked>,
    %C: tensor<128x128xf32, #mma>
) -> tensor<128x128xf32, #mma> {
  // Step 1: Blocked → Shared
  %A_smem = triton_gpu.convert_layout %A
    : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #shared>
  %B_smem = triton_gpu.convert_layout %B
    : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #shared>

  // Step 2: Shared → DotOperand
  //   这一步使用 ldmatrix 指令将数据从共享内存加载到寄存器
  //   同时按 Tensor Core 要求的模式重新排列
  %A_dot = triton_gpu.convert_layout %A_smem
    : tensor<128x64xf16, #shared> -> tensor<128x64xf16, #dot_a>
  %B_dot = triton_gpu.convert_layout %B_smem
    : tensor<64x128xf16, #shared> -> tensor<64x128xf16, #dot_b>

  // Step 3: 执行 Tensor Core 矩阵乘法
  %D = tt.dot %A_dot, %B_dot, %C
    : tensor<128x64xf16, #dot_a> * tensor<64x128xf16, #dot_b>
    -> tensor<128x128xf32, #mma>

  return %D : tensor<128x128xf32, #mma>
}
```

<div data-component="DotOperandMapping"></div>

[组件：DotOperandMapping - 展示 Shared → DotOperand 转换中数据的重新排列过程]

---

## 13.7 Layout 转换：convert_layout 操作

### 13.7.1 convert_layout 概述

`triton_gpu.convert_layout` 是 TritonGPU Dialect 中最重要的操作之一。它在不同的 Encoding 之间转换数据布局，是连接各个硬件层次的关键纽带。

```mlir
// 基本语法
%result = triton_gpu.convert_layout %source
  : tensor<MxNxT, #src_encoding> -> tensor<MxNxT, #dst_encoding>
```

### 13.7.2 转换的分类与开销

不同类型的布局转换，其开销差异巨大：

```
布局转换的开销分类:

1. 零开销转换 (纯元数据操作):
   - 当 src 和 dst 的线程分布完全相同
   - 或者转换可以通过 reinterpret 完成
   示例: Blocked → Blocked (相同参数)

2. 寄存器重排转换 (仅在寄存器内):
   - 当数据在同一线程内重新排列
   - 使用 shuffle 指令 (warp 内) 或 shared memory (warp 间)
   示例: 不同 BlockedEncoding 之间的转换

3. 共享内存转换 (需要写入/读出 shared memory):
   - 需要 store → barrier → load
   - 有共享内存带宽开销和同步开销
   示例: Blocked → Shared, Shared → DotOperand

4. 跨 CTA 转换 (需要全局内存中转):
   - 最昂贵的转换
   - 需要 store 到 global → barrier → load from global
   通常由编译器尽量避免
```

### 13.7.3 常见的转换路径

```mlir
// 路径 1: Blocked → Shared (用于数据搬运)
// 开销: 写共享内存 + __syncthreads
%1 = triton_gpu.convert_layout %x
  : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #shared>

// 路径 2: Shared → DotOperand (用于 Tensor Core)
// 开销: 从共享内存加载，使用 ldmatrix 指令
%2 = triton_gpu.convert_layout %x
  : tensor<128x64xf16, #shared> -> tensor<128x64xf16, #dot_a>

// 路径 3: Blocked → DotOperand (直接转换，编译器可能优化)
// 编译器可能将其分解为 Blocked → Shared → DotOperand
// 或者直接使用寄存器重排
%3 = triton_gpu.convert_layout %x
  : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #dot_a>

// 路径 4: Mma → Blocked (Tensor Core 输出到通用布局)
// 开销: 寄存器重排 (warp 内 shuffle)
%4 = triton_gpu.convert_layout %x
  : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
```

### 13.7.4 Shared → DotOperand 的详细机制

这是最关键的转换之一，因为它直接决定了 Tensor Core 的效率：

```
Shared → DotOperand 转换 (使用 ldmatrix 指令):

共享内存中的数据布局 (Shared Encoding):
  ┌────────────────────────────────────┐
  │  行 0: [e00 e01 e02 ... e0f]      │
  │  行 1: [e10 e11 e12 ... e1f]      │
  │  ...                               │
  │  行 M: [eM0 eM1 eM2 ... eMf]      │
  └────────────────────────────────────┘

ldmatrix 指令:
  - 从共享内存加载 4 个 8×8 的矩阵片段到寄存器
  - 每个线程贡献 8 个字节的共享内存地址
  - 加载后，数据自动按 MMA 布局排列

寄存器中的数据布局 (DotOperand Encoding):
  线程 T0: [A[0][0:4], A[1][0:4]]  ← 4 个 FP16 元素
  线程 T1: [A[0][4:8], A[1][4:8]]
  ...
  (完全匹配 mma.sync 指令的操作数格式)

对应 PTX:
  ldmatrix.sync.aligned.m8n8.x4.shared.b16
    {r0, r1, r2, r3}, [smem_ptr];
```

### 13.7.5 转换开销分析

```
转换开销量化 (以 A100 为例):

1. Blocked → Shared:
   - 写入: 128×64×2 bytes = 16 KB
   - 同步: __syncthreads ≈ 0.1 μs
   - 总开销 ≈ 写入带宽 + 同步

2. Shared → DotOperand:
   - ldmatrix 指令: 每条加载 4 个 8×8 矩阵
   - 对于 128×64 矩阵: 需要 128/8 × 64/8 / 4 = 32 条 ldmatrix
   - 每条 ldmatrix 约 4 个周期 → 32 × 4 = 128 周期

3. Blocked → Shared → DotOperand (完整路径):
   - 总开销 ≈ 写 smem + barrier + ldmatrix
   - 通常在 1-5 μs 内完成

对比: 如果不用共享内存，直接寄存器重排
   - 需要 warp shuffle (跨线程数据交换)
   - 对于复杂的重排模式，可能需要多轮 shuffle
   - 开销取决于源和目标布局的差异程度
```

### 13.7.6 转换模式的 PTX 级实现

```
编译器将 convert_layout 降级为具体的 PTX 指令序列:

1. Blocked → Shared:
   // 对应 PTX 代码 (简化)
   // 每个线程将寄存器中的数据写入共享内存
   st.shared.v4.b32 [smem + offset0], {r0, r1, r2, r3}
   st.shared.v4.b32 [smem + offset1], {r4, r5, r6, r7}
   // 同步屏障
   bar.sync 0

2. Shared → DotOperand (使用 ldmatrix):
   // 对应 PTX 代码
   ldmatrix.sync.aligned.m8n8.x4.shared.b16
     {r0, r1, r2, r3}, [smem + addr0];
   ldmatrix.sync.aligned.m8n8.x4.shared.b16
     {r4, r5, r6, r7}, [smem + addr1];
   // ldmatrix 自动完成数据重排，无需额外指令

3. Blocked → Mma (直接寄存器重排):
   // 如果源和目标的 warp 内线程分布兼容
   // 可以使用 shuffle 指令直接重排
   shfl.sync.bfly r0, r0, 0x10, 0x1f
   shfl.sync.bfly r1, r1, 0x10, 0x1f
   // 或者使用 shared memory 中转 (更通用但更慢)
```

### 13.7.7 RemoveLayoutConversions Pass

Triton 编译器的 `RemoveLayoutConversions` pass 会分析数据流图，尝试消除冗余的布局转换：

```mlir
// 优化前: 多余的转换
%a_blocked = tt.load %ptr : tensor<128x64xf16, #blocked>
%a_shared = triton_gpu.convert_layout %a_blocked  // Blocked → Shared
  : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #shared>
%a_dot = triton_gpu.convert_layout %a_shared      // Shared → DotOperand
  : tensor<128x64xf16, #shared> -> tensor<128x64xf16, #dot_a>

// 优化后: 直接从 Blocked 到 DotOperand (如果布局兼容)
// 编译器的 OptimizeDotOperands pass 会尝试合并转换路径
%a_blocked = tt.load %ptr : tensor<128x64xf16, #blocked>
%a_dot = triton_gpu.convert_layout %a_blocked      // 直接转换
  : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #dot_a>
```

<div data-component="LayoutConversionCost"></div>

[组件：LayoutConversionCost - 交互式展示不同布局转换路径的开销对比]

---

## 13.8 硬件映射完整示例：矩阵乘法的 Encoding 注入

### 13.8.1 从 Triton IR 到 TritonGPU Dialect

让我们通过一个完整的矩阵乘法示例，展示 Triton 编译器如何将抽象的 Triton IR 转换为带有硬件 Encoding 的 TritonGPU Dialect。

**原始 Triton Python 代码：**

```python
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    tl.store(C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn), accumulator)
```

### 13.8.2 第一步：Triton Dialect (TTIR)

Frontend 将 Python 代码转换为 Triton Dialect。此时 IR 是完全抽象的，没有任何硬件信息：

```mlir
// 阶段 1: Triton Dialect (TTIR) — 抽象的 tile 操作
// 没有 Encoding，tensor 类型不含布局信息

module {
  tt.func @matmul_kernel(
    %arg0: !tt.ptr<f16>,    // A 指针
    %arg1: !tt.ptr<f16>,    // B 指针
    %arg2: !tt.ptr<f32>,    // C 指针
    %arg3: i32,             // M
    %arg4: i32,             // N
    %arg5: i32,             // K
    %arg6: i32,             // stride_am
    %arg7: i32,             // stride_ak
    %arg8: i32,             // stride_bk
    %arg9: i32,             // stride_bn
    %arg10: i32,            // stride_cm
    %arg11: i32             // stride_cn
  ) {
    // 获取程序 ID
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32

    // 计算 pid_m, pid_n
    %c128_i32 = arith.constant 128 : i32   // BLOCK_M
    %c64_i32 = arith.constant 64 : i32     // BLOCK_N
    %c32_i32 = arith.constant 32 : i32     // BLOCK_K
    %1 = arith.divsi %arg4, %c64_i32 : i32  // num_pid_n
    %2 = arith.divsi %0, %1 : i32           // pid_m
    %3 = arith.remsi %0, %1 : i32           // pid_n

    // 计算偏移
    %4 = arith.muli %2, %c128_i32 : i32    // pid_m * BLOCK_M
    %5 = arith.muli %3, %c64_i32 : i32     // pid_n * BLOCK_N

    // 创建 range
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %7 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %8 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>

    // 广播和指针计算 (省略中间步骤) ...
    // %a_ptrs, %b_ptrs: 指针 tensor

    // 初始化累加器
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32>

    // 循环
    %c1_i32 = arith.constant 1 : i32
    %loop_count = arith.divsi %arg5, %c32_i32 : i32

    %result = scf.for %k = %c0_i32 to %loop_count step %c1_i32
      iter_args(%acc = %cst) -> tensor<128x64xf32> {

      // 加载 A 和 B (无 encoding)
      %a_mask = ... // 省略 mask 计算
      %a = tt.load %a_ptrs, %a_mask : tensor<128x32xf16>
      %b = tt.load %b_ptrs, %b_mask : tensor<32x64xf16>

      // 矩阵乘法 (无 encoding)
      %acc_new = tt.dot %a, %b, %acc
        : tensor<128x32xf16> * tensor<32x64xf16> -> tensor<128x64xf32>

      // 更新指针
      // ...

      scf.yield %acc_new : tensor<128x64xf32>
    }

    // 存储结果
    tt.store %c_ptrs, %result : tensor<128x64xf32>

    tt.return
  }
}
```

### 13.8.3 第二步：TritonGPU Conversion — 注入 Encoding

编译器运行 `TritonGPUConversionPass`，为每个 tensor 类型注入适当的 Encoding。这个过程涉及多个子 pass：

```mlir
// 阶段 2: TritonGPU Dialect (TTGIR) — 带硬件 Encoding
// 每个 tensor 类型都有明确的布局信息

// 定义 Encoding 属性
#blocked_A = #triton_gpu.blocked<{
  sizePerThread = [1, 4],       // 每线程: 1 行 × 4 列
  threadsPerWarp = [8, 4],      // 每 warp: 8 行 × 4 列 = 32 线程
  warpsPerCta = [4, 1],         // 每 CTA: 4 行 × 1 列 = 4 warps
  order = [1, 0]                // 列优先
}>

#blocked_B = #triton_gpu.blocked<{
  sizePerThread = [4, 1],       // 每线程: 4 行 × 1 列
  threadsPerWarp = [4, 8],      // 每 warp: 4 行 × 8 列 = 32 线程
  warpsPerCta = [1, 4],         // 每 CTA: 1 行 × 4 列 = 4 warps
  order = [0, 1]                // 行优先 (因为 B 矩阵要转置访问)
}>

#mma = #triton_gpu.nvidia_mma<{
  versionMajor = 2, versionMinor = 0,
  warpsPerCta = [4, 1]
}>

#shared_A = #triton_gpu.shared<{
  vec = 8, perPhase = 1, maxPhase = 8,
  order = [1, 0]
}>

#shared_B = #triton_gpu.shared<{
  vec = 8, perPhase = 1, maxPhase = 8,
  order = [0, 1]
}>

#dot_operand_a = #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

module {
  tt.func @matmul_kernel(
    %arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f32>,
    // ... 其他参数不变
  ) {
    %0 = tt.get_program_id x : i32

    // 范围计算 — 结果带有 BlockedEncoding
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32}
      : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked_A}>>
    %7 = tt.make_range {end = 64 : i32, start = 0 : i32}
      : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked_B}>>
    %8 = tt.make_range {end = 32 : i32, start = 0 : i32}
      : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked_A}>>

    // 指针计算 — slice encoding 通过 expand_dims 和广播传播
    // ... (省略中间步骤)

    // 初始化累加器 — MMA encoding (Tensor Core 输出布局)
    %cst = arith.constant dense<0.000000e+00>
      : tensor<128x64xf32, #mma>

    %loop_count = arith.divsi %arg5, %c32_i32 : i32

    // 循环 — 每次迭代的布局转换
    %result = scf.for %k = %c0_i32 to %loop_count step %c1_i32
      iter_args(%acc = %cst) -> tensor<128x64xf32, #mma> {

      // 加载 A — 结果使用 BlockedEncoding
      %a = tt.load %a_ptrs, %a_mask
        : tensor<128x32xf16, #blocked_A>

      // 加载 B — 结果使用 BlockedEncoding
      %b = tt.load %b_ptrs, %b_mask
        : tensor<32x64xf16, #blocked_B>

      // === 布局转换路径: Blocked → Shared → DotOperand ===

      // Step 2a: Blocked → Shared
      %a_shared = triton_gpu.convert_layout %a
        : tensor<128x32xf16, #blocked_A> -> tensor<128x32xf16, #shared_A>
      %b_shared = triton_gpu.convert_layout %b
        : tensor<32x64xf16, #blocked_B> -> tensor<32x64xf16, #shared_B>

      // Step 2b: Shared → DotOperand (ldmatrix)
      %a_dot = triton_gpu.convert_layout %a_shared
        : tensor<128x32xf16, #shared_A> -> tensor<128x32xf16, #dot_operand_a>
      %b_dot = triton_gpu.convert_layout %b_shared
        : tensor<32x64xf16, #shared_B> -> tensor<32x64xf16, #dot_operand_b>

      // Step 3: Tensor Core 矩阵乘法
      %acc_new = tt.dot %a_dot, %b_dot, %acc
        : tensor<128x32xf16, #dot_operand_a>
        * tensor<32x64xf16, #dot_operand_b>
        -> tensor<128x64xf32, #mma>

      scf.yield %acc_new : tensor<128x64xf32, #mma>
    }

    // 存储结果 — 需要从 MMA 布局转回 Blocked 布局
    %result_blocked = triton_gpu.convert_layout %result
      : tensor<128x64xf32, #mma> -> tensor<128x64xf32, #blocked_A>

    tt.store %c_ptrs, %result_blocked : tensor<128x64xf32, #blocked_A>

    tt.return
  }
}
```

### 13.8.4 第三步：优化 Pass 后的 IR

编译器运行多个优化 pass 后，IR 会进一步优化：

```mlir
// 阶段 3: 优化后的 TTGIR
// 主要优化:
//   1. Coalesce pass: 优化 BlockedEncoding 使内存访问更合并
//   2. OptimizeDotOperands: 优化布局转换路径
//   3. Pipeline: 插入软件流水线（异步加载）

// 优化后的编码可能被调整
#blocked_A_opt = #triton_gpu.blocked<{
  sizePerThread = [1, 8],        // 优化: 增大向量宽度以改善合并访问
  threadsPerWarp = [16, 2],      // 优化: 调整线程排列
  warpsPerCta = [4, 1],
  order = [1, 0]
}>

// 在 Hopper 架构上，可能使用 wgmma 和 async copy:
//   - ttg.async_copy_global_to_local: 异步从全局内存加载到共享内存
//   - wgmma.mma_async: 异步 Warp Group 矩阵乘法
//   - ttg.async_commit_group / ttg.async_wait_group: 流水线同步
```

### 13.8.5 Encoding 注入的完整流程图

```
Encoding 注入的 Pass 序列:

┌─────────────────────────────────────────────────────┐
│ 1. TritonGPUConversion                              │
│    为所有 tensor 类型添加基础 Encoding (Blocked)       │
└───────────────────────┬─────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│ 2. Coalesce                                         │
│    优化 BlockedEncoding 的 sizePerThread              │
│    使内存访问更好地合并 (coalesced)                    │
└───────────────────────┬─────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│ 3. AccelerateMatmul                                 │
│    将 tt.dot 的输出 Encoding 从 Blocked → Mma         │
│    为 dot 操作数创建 DotOperandEncoding               │
└───────────────────────┬─────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│ 4. OptimizeDotOperands                              │
│    优化 dot 操作数的布局转换路径                       │
│    可能插入 SharedEncoding 作为中间步骤                │
└───────────────────────┬─────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│ 5. RemoveLayoutConversions (可选)                    │
│    移除冗余的 convert_layout 操作                     │
└───────────────────────┬─────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│ 6. Pipeline                                         │
│    插入软件流水线，将加载与计算重叠                     │
│    可能引入新的 SharedEncoding 用于双缓冲              │
└───────────────────────┬─────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│ 7. DecomposeConversions                             │
│    将复杂的 convert_layout 分解为基本操作             │
│    如: Blocked → DotOperand = Blocked → Shared → Dot │
└───────────────────────┬─────────────────────────────┘
                        ▼
        最终的 TTGIR → LLVM Dialect → PTX/SASS
```

### 13.8.6 完整的 Encoding 转换时序

```
矩阵乘法 kernel 中的数据布局生命周期:

时间 ──────────────────────────────────────────────────►

A 矩阵:
  Global Mem  →  [load: Blocked]  →  [cvtl: Shared]  →  [cvtl: DotOp]  →  [dot: consumed]
                  每线程 1×4         写入 smem           ldmatrix 加载      Tensor Core 使用
                  合并访问            无 bank conflict     匹配 mma 格式

B 矩阵:
  Global Mem  →  [load: Blocked]  →  [cvtl: Shared]  →  [cvtl: DotOp]  →  [dot: consumed]
                  每线程 4×1         写入 smem           ldmatrix 加载      Tensor Core 使用

C 矩阵 (累加器):
                  [init: Mma]  ──────────────────────────────────────────  [store: → Blocked]
                  零初始化             Tensor Core 输出格式                   转为通用格式存储

数据布局转换图:
  ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
  │ Blocked │───►│ Shared │───►│ DotOp  │───►│  Mma   │
  │ (load)  │    │ (smem) │    │ (ldmx) │    │ (dot)  │
  └────────┘    └────────┘    └────────┘    └────────┘
  寄存器布局      共享内存        寄存器布局      Tensor Core
  合并访问       bank 优化       匹配硬件        矩阵乘法
```

<div data-component="MatmulEncodingFlow"></div>

[组件：MatmulEncodingFlow - 交互式展示矩阵乘法从 TTIR 到 TTGIR 的完整转换过程，包括每个步骤的 Encoding 变化]

---

## 13.9 Encoding 的高级主题

### 13.9.1 SliceEncodingAttr

SliceEncoding 是一种"降维"编码，用于表示高维 tensor 的切片：

```mlir
// 当对一个 3D tensor 取切片得到 2D tensor 时
// 使用 SliceEncoding 保持与父 tensor 的布局一致性

#blocked_3d = #triton_gpu.blocked<{
  sizePerThread = [1, 4, 2],
  threadsPerWarp = [4, 4, 2],
  warpsPerCta = [2, 2, 1],
  order = [2, 1, 0]
}>

// 在 dim=0 上切片
#slice = #triton_gpu.slice<{dim = 0, parent = #blocked_3d}>

// 切片后的 tensor 使用 SliceEncoding
%x_slice = ... : tensor<64x32xf16, #slice>
```

### 13.9.2 Encoding 的传播规则

当操作的输入和输出需要不同的 Encoding 时，编译器需要决定如何传播：

```mlir
// 规则 1: 逐元素操作 — 输出继承输入的 Encoding
%a: tensor<128x64xf16, #blocked>
%b = arith.addf %a, %a : tensor<128x64xf16, #blocked>  // 输出也是 #blocked

// 规则 2: dot 操作 — 输出使用 MmaEncoding
%a: tensor<128x32xf16, #dot_a>
%b: tensor<32x64xf16, #dot_b>
%c: tensor<128x64xf32, #mma>
%d = tt.dot %a, %b, %c : ... -> tensor<128x64xf32, #mma>

// 规则 3: reduce 操作 — 切片维度被去除
%a: tensor<128x64xf16, #blocked>
%b = tt.reduce %a, dim=0 : tensor<128x64xf16, #blocked> -> tensor<64xf16, #slice>

// 规则 4: expand_dims — 添加新的切片维度
%a: tensor<64xf16, #slice>
%b = tt.expand_dims %a, dim=0 : tensor<64xf16, #slice> -> tensor<1x64xf16, #blocked>
```

### 13.9.3 多 CTA 与 CGA（Cooperative Grid Array）

在 Hopper 架构上，多个 CTA 可以协作完成一个大矩阵乘法：

```mlir
// Hopper+ 的 CTA 协作配置
#mma_hopper = #triton_gpu.nvidia_mma<{
  versionMajor = 3,
  versionMinor = 0,
  warpsPerCta = [4, 1],
  CTAsPerCGA = [2, 2],       // 2×2 = 4 个 CTA 协作
  CTASplitNum = [2, 2],       // 每个维度由 2 个 CTA 分担
  CTAOrder = [1, 0]
}>

// CTAsPerCGA = [2, 2] 的含义:
// ┌─────────┬─────────┐
// │  CTA(0,0)│  CTA(0,1)│   M 维度由 2 个 CTA 分担
// ├─────────┼─────────┤
// │  CTA(1,0)│  CTA(1,1)│   N 维度由 2 个 CTA 分担
// └─────────┴─────────┘
// 每个 CTA 计算结果的 1/4，最后通过分布式共享内存或 TMA 合并
```

### 13.9.4 Encoding 验证

TritonGPU Dialect 对 Encoding 有严格的验证规则：

```mlir
// 验证规则 1: dot 操作的操作数必须使用 DotOperandEncoding
// 错误: 操作数使用 BlockedEncoding
%d = tt.dot %a, %b, %c
  : tensor<128x32xf16, #blocked>    // ✗ 应该是 #dot_a
  * tensor<32x64xf16, #blocked>     // ✗ 应该是 #dot_b
  -> tensor<128x64xf32, #mma>

// 验证规则 2: convert_layout 的 src 和 dst 必须兼容
// 错误: 直接从 Blocked 转到 Mma (需要经过 Shared/DotOperand)
%x = triton_gpu.convert_layout %a
  : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #mma>  // ✗ 不合法

// 验证规则 3: threadsPerWarp 的乘积必须等于 warp size
// NVIDIA: 32, AMD: 64
#bad = #triton_gpu.blocked<{
  threadsPerWarp = [5, 5],  // 5×5=25 ≠ 32 ✗
  // ...
}>
```

---

## 13.10 源码导读

### 13.10.1 关键源码文件

```
TritonGPU Dialect 核心源码:

include/triton/Dialect/TritonGPU/IR/
├── TritonGPUAttrDefs.td          ← Encoding 属性的 TableGen 定义
│   ├── BlockedEncodingAttr       ← 第 ~150 行
│   ├── MmaEncodingAttr           ← 第 ~300 行 (含 NvidiaMmaEncodingAttr, AmdMmaEncodingAttr)
│   ├── SharedEncodingAttr        ← 第 ~450 行
│   └── DotOperandEncodingAttr    ← 第 ~550 行
├── TritonGPUBase.td              ← 基础类型和枚举定义
├── TritonGPUOps.td               ← 操作定义 (convert_layout 等)
└── TritonGPUDialect.td           ← Dialect 级别的定义

lib/Dialect/TritonGPU/IR/
├── TritonGPUAttrDefs.cpp         ← Encoding 的 C++ 实现
│   ├── BlockedEncodingAttr::getElemsPerThread()
│   ├── MmaEncodingAttr::getElemsPerThread()
│   └── SharedEncodingAttr::getMutableMmaLayout()
├── TritonGPUOps.cpp              ← 操作实现
│   └── ConvertLayoutOp::verify() ← convert_layout 的验证逻辑
└── TritonGPUDialect.cpp          ← Dialect 注册和类型解析

lib/Dialect/TritonGPU/Transforms/
├── AccelerateMatmul.cpp          ← 注入 MmaEncoding 的核心 pass
│   └── getMmaVersion()           ← 根据硬件选择 MMA 版本
├── Coalesce.cpp                  ← 优化 BlockedEncoding
├── OptimizeDotOperands.cpp       ← 优化 dot 操作数布局
├── RemoveLayoutConversions.cpp   ← 移除冗余的布局转换
├── DecomposeConversions.cpp      ← 分解复杂的布局转换
├── Pipeline.cpp                  ← 软件流水线优化
└── Utility.cpp                   ← 共用工具函数
    ├── getSharedEncoding()
    └── getDotOperandEncoding()
```

### 13.10.2 Encoding 属性的核心方法

每种 Encoding 都需要实现以下核心方法：

```cpp
// 每个 Encoding 必须实现的接口 (简化版)

class EncodingAttr {
public:
  // 获取每个线程持有的元素数
  virtual SmallVector<unsigned> getElemsPerThread(Type type) const = 0;

  // 获取每个 warp 的线程数
  virtual SmallVector<unsigned> getThreadsPerWarp() const = 0;

  // 获取每个 CTA 的 warp 数
  virtual SmallVector<unsigned> getWarpsPerCTA() const = 0;

  // 获取维度优先级
  virtual SmallVector<unsigned> getOrder() const = 0;

  // 验证 Encoding 的合法性
  virtual LogicalResult verify() const = 0;
};

// BlockedEncodingAttr 的实现示例
SmallVector<unsigned> BlockedEncodingAttr::getElemsPerThread(Type type) const {
  auto shape = cast<RankedTensorType>(type).getShape();
  SmallVector<unsigned> elemsPerThread;
  for (int d = 0; d < shape.size(); d++) {
    elemsPerThread.push_back(
      getSizePerThread()[d] * getThreadsPerWarp()[d] * getWarpsPerCTA()[d]
    );
  }
  return elemsPerThread;
}
```

<div data-component="SourceCodeBrowser"></div>

[组件：SourceCodeBrowser - 可视化浏览 TritonGPU Dialect 源码结构，支持跳转到关键定义]

---

## 13.11 性能影响与编码选择策略

### 13.11.1 Encoding 对性能的影响

Encoding 的选择直接影响 kernel 的执行效率。不同 Encoding 导致不同的数据访问模式，进而影响内存合并度、bank conflict 和指令吞吐。

```
Encoding 选择对性能的影响:

┌─────────────────────┬──────────────────────────────────────────┐
│  Encoding 类型       │  性能影响因素                             │
├─────────────────────┼──────────────────────────────────────────┤
│  BlockedEncoding    │  - sizePerThread 影响向量化访问宽度        │
│                     │  - order 影响合并访问程度                  │
│                     │  - threadsPerWarp 影响 warp 内协作效率     │
├─────────────────────┼──────────────────────────────────────────┤
│  MmaEncoding        │  - version 决定可用指令集                  │
│                     │  - warpsPerCta 影响 warp 级并行度          │
│                     │  - 操作数布局直接映射硬件格式              │
├─────────────────────┼──────────────────────────────────────────┤
│  SharedEncoding     │  - vec 影响共享内存带宽利用率              │
│                     │  - perPhase/maxPhase 影响 bank conflict   │
│                     │  - order 影响访问局部性                    │
├─────────────────────┼──────────────────────────────────────────┤
│  DotOperandEncoding │  - kWidth 影响 K 维度分块粒度              │
│                     │  - parent 决定与哪个 MmaEncoding 配对     │
└─────────────────────┴──────────────────────────────────────────┘
```

### 13.11.2 BlockedEncoding 的最优配置原则

选择 BlockedEncoding 参数时，需要平衡多个因素：

```mlir
// 原则 1: sizePerThread 应匹配向量化加载宽度
// FP16 数据, 128-bit 加载: sizePerThread[dim] 应为 8 (8×2=16 bytes)
#good_vec = #triton_gpu.blocked<{
  sizePerThread = [1, 8],     // 列方向 8 个 FP16 = 16 bytes = 128 bits ✓
  threadsPerWarp = [32, 1],   // 32 线程沿行方向
  warpsPerCta = [4, 1],
  order = [1, 0]
}>

// 原则 2: threadsPerWarp 的 order 应匹配 sizePerThread 的 order
// 确保连续线程访问连续地址
#good_coalesced = #triton_gpu.blocked<{
  sizePerThread = [1, 8],     // 连续 8 个元素在 dim1
  threadsPerWarp = [4, 8],    // 连续 8 个线程在 dim1
  warpsPerCta = [4, 1],
  order = [1, 0]              // dim1 优先 → 连续线程访问连续地址 ✓
}>

// 原则 3: warpsPerCta 应匹配问题规模
// 大规模矩阵: 更多 warp 在 M 维度
// 小规模矩阵: 均匀分布 warp
#large_matmul = #triton_gpu.blocked<{
  sizePerThread = [1, 8],
  threadsPerWarp = [16, 2],
  warpsPerCta = [8, 1],       // 8 个 warp 全在 M 维度
  order = [1, 0]
}>
```

### 13.11.3 Encoding 选择的决策树

```
Encoding 选择决策树:

                    操作类型?
                    /       \
               逐元素操作    矩阵乘法
                 |              |
           使用 Blocked     需要 Tensor Core?
                            /          \
                          是            否
                          |              |
                    使用 Mma       使用 Blocked
                    + DotOperand    + 高 sizePerThread
                          |
                    数据需要经过 Shared?
                    /              \
                  是                否
                  |                  |
          Blocked → Shared       Blocked → DotOperand
          → DotOperand          (如果布局兼容)
                  |
          bank conflict 优化?
          /              \
        有               无
        |                |
  调整 SharedEncoding   保持默认
  (perPhase, maxPhase)
```

### 13.11.4 常见的性能陷阱

```
陷阱 1: sizePerThread 与数据类型不匹配

  // 错误: FP16 数据使用 sizePerThread=4, 但 vec 计算为 4×2=8 bytes
  // 不足以充分利用 128-bit 加载
  #bad = #triton_gpu.blocked<{
    sizePerThread = [1, 4],  // 4 × sizeof(FP16) = 8 bytes < 128 bits
    ...
  }>

  // 正确: 使用 8 以匹配 128-bit 加载
  #good = #triton_gpu.blocked<{
    sizePerThread = [1, 8],  // 8 × sizeof(FP16) = 16 bytes = 128 bits ✓
    ...
  }>

陷阱 2: order 与数据访问模式不匹配

  // 矩阵按行存储 (row-major)，但 order 设为列优先
  // 导致连续线程访问不连续的地址
  #bad_order = #triton_gpu.blocked<{
    order = [1, 0],  // 列优先，但数据是行存储
    ...
  }>

  // 正确: order 应匹配数据的存储顺序
  #good_order = #triton_gpu.blocked<{
    order = [0, 1],  // 行优先，匹配行存储
    ...
  }>

陷阱 3: SharedEncoding 的 vec 过大导致 bank conflict

  // vec=16 可能导致所有线程访问同一 bank 的同一行
  #risky = #triton_gpu.shared<{
    vec = 16,        // 可能导致 2-way bank conflict
    perPhase = 1,
    maxPhase = 8,
    ...
  }>

  // 降低 vec 或调整 perPhase 来避免
  #safe = #triton_gpu.shared<{
    vec = 8,         // 更安全的向量宽度
    perPhase = 2,    // 增大 perPhase 改变偏移模式
    maxPhase = 8,
    ...
  }>
```

### 13.11.5 性能对比：不同 Encoding 配置的 benchmark

```
矩阵乘法 4096×4096×4096, FP16, A100 GPU:

配置 1: 朴素 BlockedEncoding (不使用 Tensor Core)
  #blocked = #triton_gpu.blocked<{
    sizePerThread = [4, 4], threadsPerWarp = [8, 4],
    warpsPerCta = [4, 1], order = [1, 0]
  }>
  性能: ~15 TFLOPS (使用 CUDA Core)

配置 2: 使用 Tensor Core, 但布局不优
  #blocked = #triton_gpu.blocked<{
    sizePerThread = [1, 4], threadsPerWarp = [8, 4],
    warpsPerCta = [4, 1], order = [1, 0]
  }>
  #mma = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCta = [4, 1]}>
  性能: ~180 TFLOPS (Tensor Core, 但数据搬运开销大)

配置 3: 优化后的布局
  #blocked = #triton_gpu.blocked<{
    sizePerThread = [1, 8], threadsPerWarp = [16, 2],
    warpsPerCta = [4, 1], order = [1, 0]
  }>
  #mma = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCta = [4, 1]}>
  #shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
  性能: ~280 TFLOPS (接近 cuBLAS 的 312 TFLOPS)

配置 4: Hopper 优化 (wgmma + TMA)
  #mma = #triton_gpu.nvidia_mma<{versionMajor = 3, warpsPerCta = [4, 1]}>
  #shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8,
                                  order = [1, 0], hasLeadingOffset = true}>
  性能: ~600+ TFLOPS (H100, 使用 wgmma 异步指令)
```

<div data-component="EncodingBenchmark"></div>

[组件：EncodingBenchmark - 交互式展示不同 Encoding 配置对矩阵乘法性能的影响]

---

## 13.12 Encoding 调试与可视化

### 13.12.1 使用 Triton 的 IR dump 功能

Triton 编译器提供了 IR dump 功能，可以查看每个 pass 后的 IR 状态：

```bash
# 启用 IR dump，查看 Encoding 注入过程
TRITON_PRINT_AUTOTUNING=1 \
TRITON_ALWAYS_COMPILE=1 \
python your_kernel.py

# 或者使用更详细的 dump
TRITON_PRINT_AUTOTUNING=1 \
TRITON_ALWAYS_COMPILE=1 \
TRITON_DUMP_DIR=/tmp/triton_dump \
python your_kernel.py
```

### 13.12.2 查看 Encoding 属性

```bash
# 在 dump 的 IR 中搜索 Encoding 定义
grep -n "triton_gpu.blocked" /tmp/triton_dump/*.ttir
grep -n "triton_gpu.nvidia_mma" /tmp/triton_dump/*.ttir
grep -n "triton_gpu.shared" /tmp/triton_dump/*.ttir
grep -n "convert_layout" /tmp/triton_dump/*.ttir
```

### 13.12.3 常见 Encoding 错误诊断

```
错误 1: "op requires the same encoding for all operands"
原因: tt.dot 的两个操作数 Encoding 不匹配
修复: 确保 %a 和 %b 都使用兼容的 DotOperandEncoding

错误 2: "mismatching encoding"
原因: convert_layout 的源和目标 Encoding 不兼容
修复: 检查转换路径是否合法 (如 Blocked → Shared → DotOperand)

错误 3: "sizePerThread * threadsPerWarp * warpsPerCta != tensor shape"
原因: Encoding 参数的乘积与 tensor 形状不匹配
修复: 调整 sizePerThread 或 threadsPerWarp 使其乘积等于 tensor 维度

错误 4: "threadsPerWarp product must be warp size"
原因: threadsPerWarp 的乘积不等于 32 (NVIDIA) 或 64 (AMD)
修复: 调整 threadsPerWarp 使其乘积正确
```

### 13.12.4 Encoding 可视化工具

```python
# 自定义工具: 打印线程到数据的映射
def print_thread_mapping(encoding, tensor_shape):
    """
    打印给定 Encoding 下每个线程负责的元素
    """
    size_per_thread = encoding.size_per_thread
    threads_per_warp = encoding.threads_per_warp
    warps_per_cta = encoding.warps_per_cta

    print(f"Encoding: {encoding}")
    print(f"Tensor shape: {tensor_shape}")
    print(f"Size per thread: {size_per_thread}")
    print(f"Threads per warp: {threads_per_warp}")
    print(f"Warps per CTA: {warps_per_cta}")
    print()

    # 计算每个线程的元素范围
    for warp_id in range(warps_per_cta[0] * warps_per_cta[1]):
        warp_row = warp_id // warps_per_cta[1]
        warp_col = warp_id % warps_per_cta[1]

        for thread_id in range(32):
            thread_row = thread_id // threads_per_warp[1]
            thread_col = thread_id % threads_per_warp[1]

            # 计算全局坐标
            global_row = (warp_row * threads_per_warp[0] + thread_row) * size_per_thread[0]
            global_col = (warp_col * threads_per_warp[1] + thread_col) * size_per_thread[1]

            print(f"Warp {warp_id}, Thread {thread_id}: "
                  f"row=[{global_row}, {global_row + size_per_thread[0]}), "
                  f"col=[{global_col}, {global_col + size_per_thread[1]})")
```

<div data-component="EncodingDebugger"></div>

[组件：EncodingDebugger - 交互式调试工具：输入 Encoding 参数和 tensor 形状，可视化线程到数据的映射]

---

## 13.13 实战练习：从零设计 Encoding

### 13.13.1 练习 1: 为自定义操作设计 Encoding

假设你要实现一个自定义的 2D 卷积 kernel，输入 tensor 形状为 `[N, C, H, W]`，需要设计合适的 Encoding。

```python
# 问题: 2D 卷积的 Encoding 设计
# 输入: [N, C, H, W] = [16, 64, 56, 56]
# 卷积核: [K, C, R, S] = [128, 64, 3, 3]
# 输出: [N, K, H', W'] = [16, 128, 56, 56]

# 方案 1: 使用通用 BlockedEncoding
# 将 N 和 C 合并为一个维度，H 和 W 合并为另一个维度
blocked_input = #triton_gpu.blocked<{
  sizePerThread = [1, 8],      // 每线程: 1 (N×C) × 8 (H×W)
  threadsPerWarp = [4, 8],     // 每 warp: 4 × 8 = 32
  warpsPerCta = [4, 1],        // 每 CTA: 4 warps
  order = [1, 0]               // H×W 优先
}>

# 方案 2: 使用分块 Encoding，按 C 维度分块
# 更适合卷积的计算模式
blocked_channel = #triton_gpu.blocked<{
  sizePerThread = [4, 1, 1, 8],  // N, C, H, W
  threadsPerWarp = [2, 8, 1, 2], // 每 warp 32 线程
  warpsPerCta = [1, 1, 2, 2],    // 每 CTA 4 warps
  order = [3, 2, 1, 0]           // W, H, C, N 优先
}>
```

### 13.13.2 练习 2: 优化已有 Kernel 的 Encoding

```python
# 原始 Kernel (性能不佳)
@triton.jit
def bad_matmul(A, B, C, M, N, K, ...):
    # BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
    # 问题: 使用了不优的 BlockedEncoding

# 优化步骤:
# 1. 分析当前 Encoding 的数据访问模式
# 2. 识别性能瓶颈 (bank conflict, 非合并访问等)
# 3. 调整 Encoding 参数
# 4. 验证优化效果

# 优化后的 Encoding:
optimized_blocked = #triton_gpu.blocked<{
  sizePerThread = [1, 8],      // 增大向量宽度
  threadsPerWarp = [16, 2],    // 调整线程排列
  warpsPerCta = [4, 1],
  order = [1, 0]
}>
```

### 13.13.3 练习 3: 跨架构 Encoding 迁移

```
任务: 将 Ampere (A100) 的 Encoding 迁移到 Hopper (H100)

Ampere 配置:
  #mma_ampere = #triton_gpu.nvidia_mma<{
    versionMajor = 2, versionMinor = 0,
    warpsPerCta = [4, 1]
  }>
  #shared_ampere = #triton_gpu.shared<{
    vec = 8, perPhase = 1, maxPhase = 8,
    order = [1, 0], hasLeadingOffset = false
  }>

Hopper 配置:
  #mma_hopper = #triton_gpu.nvidia_mma<{
    versionMajor = 3, versionMinor = 0,
    warpsPerCta = [4, 1],
    CTAsPerCGA = [2, 2],       // 新增: CTA 协作
    CTASplitNum = [2, 2],
    CTAOrder = [1, 0]
  }>
  #shared_hopper = #triton_gpu.shared<{
    vec = 8, perPhase = 1, maxPhase = 8,
    order = [1, 0], hasLeadingOffset = true  // 修改: 启用 leading offset
  }>

迁移要点:
  1. MMA version 从 2 升级到 3
  2. 添加 CGA 相关参数
  3. 启用 SharedEncoding 的 hasLeadingOffset
  4. 考虑使用 TMA (Tensor Memory Accelerator) 替代手动加载
```

<div data-component="EncodingExercise"></div>

[组件：EncodingExercise - 交互式练习：设计和优化不同场景下的 Encoding 配置]

---

## 本章小结

本章深入探讨了 TritonGPU Dialect 的设计原理和实现细节。以下是关键要点：

1. **TritonGPU Dialect 的必要性**：它是连接抽象 tile 操作和具体硬件指令的桥梁层。没有它，编译器无法知道数据应该如何分布到线程中，也无法选择合适的硬件指令。

2. **四大 Encoding 属性**：
   - **BlockedEncoding**：通用的分块布局，用于寄存器中的数据分布。由 sizePerThread、threadsPerWarp、warpsPerCta、order 四个参数控制。
   - **MmaEncoding**：Tensor Core / MFMA 的专用布局。通过 versionMajor/minor 区分不同硬件代次的指令集。
   - **SharedEncoding**：共享内存的布局优化。通过 vec、perPhase、maxPhase 等参数避免 bank conflict。
   - **DotOperandEncoding**：点积操作数的专用布局，引用 MmaEncoding 作为父节点，确保操作数格式匹配 Tensor Core 要求。

3. **布局转换**：`triton_gpu.convert_layout` 是连接不同 Encoding 的关键操作。常见的路径是 Blocked → Shared → DotOperand，每一步都有明确的硬件操作对应（写 smem → barrier → ldmatrix）。

4. **编译器优化**：Triton 编译器通过多个 pass（Coalesce、AccelerateMatmul、OptimizeDotOperands、Pipeline 等）自动完成 Encoding 的注入和优化，用户无需手动管理。

5. **硬件映射**：最终，每个 tensor 操作都被映射到具体的硬件操作——加载使用合并访问的 load 指令，矩阵乘法使用 Tensor Core 的 mma.sync/wgmma 指令，数据搬运通过共享内存和 ldmatrix 指令完成。

---

## 思考题

1. **BlockedEncoding 设计**：假设你有一个形状为 `[256, 128]` 的 tensor，目标 GPU 的 warp size 为 32。请设计一个 BlockedEncoding 参数组合（sizePerThread、threadsPerWarp、warpsPerCta），使得：
   - 每个 CTA 使用 8 个 warp
   - 每个线程持有 16 个元素
   - 内存访问是合并的（order = [1, 0]）

2. **Bank Conflict 分析**：给定 SharedEncoding `{vec = 4, perPhase = 2, maxPhase = 4, order = [1, 0]}`，分析当 warp 中 32 个线程同时访问共享内存的同一列时，是否会发生 bank conflict？如果会，如何调整参数来避免？

3. **Encoding 转换开销**：在一个矩阵乘法 kernel 中，如果 BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32，数据类型为 FP16，估算一次循环迭代中所有 convert_layout 操作的总开销（以字节为单位的共享内存读写量）。

4. **MmaEncoding 版本选择**：解释为什么在 Ampere GPU 上使用 `versionMajor = 2` 而在 Hopper GPU 上使用 `versionMajor = 3`。这两个版本在操作数布局上有什么根本性差异？

5. **编译器优化推理**：假设编译器发现一个 dot 操作的两个操作数已经按 DotOperandEncoding 布局存储在共享内存中（例如上一次 dot 操作的输出被转存后复用）。编译器应该如何优化这个场景？能否跳过某些 convert_layout 步骤？

6. **跨架构移植**：如果要将一个为 NVIDIA GPU 编写的 Triton kernel 移植到 AMD GPU，哪些 Encoding 需要修改？MmaEncoding 和 SharedEncoding 的参数会有什么变化？

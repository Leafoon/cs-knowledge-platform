---
title: "Chapter 3: Tile 编程模型与核心抽象"
description: "深入理解 TileLang 的核心编程模型：Tile（块）作为一等公民的设计思想、Thread Binding 机制、数据并行与计算并行的映射关系，掌握 T.Fragment/T.Shared/T.L1 的多级数据类型体系。"
updated: "2025-06-11"
---

# Chapter 3: Tile 编程模型与核心抽象

<div data-component="TileAbstractionDiagram"></div>

> [!NOTE]
> **学习目标**
>
> - 深入理解 Tile（块）作为一等公民的设计思想
> - 掌握 Thread Binding 机制的细节
> - 理解数据并行与计算并行的映射关系
> - 掌握 T.Fragment/T.Shared/T.L1 的多级数据类型体系
> - 学习 Tile Shape 选择策略

---

## 1. Tile：一等公民的设计思想

### 1.1 什么是 Tile？

在 GPU 编程中，Tile（块）是一个核心概念。它代表数据或计算的一个**基本单元**，可以被映射到硬件的某个并行层级。

```
矩阵 A (M×K):
┌─────────────────────────────────────────┐
│                                         │
│   ┌─────┬─────┬─────┬─────┐            │
│   │T00  │T01  │T02  │T03  │ ← Tile 行  │
│   ├─────┼─────┼─────┼─────┤            │
│   │T10  │T11  │T12  │T13  │            │
│   ├─────┼─────┼─────┼─────┤            │
│   │T20  │T21  │T22  │T23  │            │
│   └─────┴─────┴─────┴─────┘            │
│                                         │
│   每个 Tile 是一个独立的计算单元         │
│   可以映射到 Thread Block / Warp        │
└─────────────────────────────────────────┘
```

这个代码块或示意图用于说明 1.1 什么是 Tile？ 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 1.2 为什么 Tile 是一等公民？

在传统的 CUDA 编程中，**线程**是一等公民。程序员需要：
1. 计算每个线程的索引
2. 确定每个线程处理的数据
3. 管理线程间的同步

```cpp
// CUDA: 线程是一等公民
__global__ void gemm(float* A, float* B, float* C, int M, int N, int K) {
    // 每个线程计算自己的索引
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个线程计算一个输出元素
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}
```

以上 CUDA 代码展示了传统以线程为中心的 GEMM 实现方式。程序员需要手动计算每个线程的全局索引 `row` 和 `col`，通过 `blockIdx` 和 `threadIdx` 组合得到线程在矩阵中的位置，每个线程负责计算一个输出元素并在 K 维度上串行累加。这种方式的缺点是代码复杂度高、可移植性差，线程索引计算与具体硬件配置强绑定，且内存访问模式容易导致非合并访问，降低带宽利用率。

在 TileLang 中，**Tile**是一等公民。程序员只需要：
1. 定义 Tile 的大小
2. 描述 Tile 内部的计算逻辑
3. TileLang 编译器自动处理线程映射

```python
# TileLang: Tile 是一等公民
@T.prim_func
def gemm(A, B, C):
    # 定义 Tile 大小
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    # Tile 级循环 (编译器自动映射到线程)
    for bx, by in T.grid(M // BLOCK_M, N // BLOCK_N):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])

            # Tile 内部的计算逻辑
            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.fill(C_local, T.float32(0))

            for k in T.serial(K // BLOCK_K):
                A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
                B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")

                T.copy(A[...], A_shared)
                T.copy(B[...], B_shared)

                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[...])
```

以上代码展示了 TileLang 中 Tile 作为一等公民的核心编程模型。其中 `T.grid()` 用于定义 Tile 级循环，编译器会自动将 Tile 映射到 GPU 的 Thread Block；`T.alloc_shared()` 和 `T.alloc_fragment()` 分别用于分配片上存储和寄存器级累加器。与传统 CUDA 相比，TileLang 让开发者专注于 Tile 级别的计算逻辑，而不需要手动管理每个线程的索引和内存访问。

### 1.3 Tile vs Thread 对比

| 维度 | CUDA (Thread 级) | TileLang (Tile 级) |
|:---|:---|:---|
| **基本单位** | 单个线程 | 数据块 (Tile) |
| **并行粒度** | 细粒度 | 粗粒度 |
| **内存管理** | 手动 per-thread | 声明式 per-tile |
| **代码复杂度** | O(threads) | O(tiles) |
| **硬件映射** | 显式 thread index | 声明式 thread_binding |
| **可移植性** | 差 (绑定具体硬件) | 好 (编译器适配) |

### 1.4 Tile 的层次结构

```
Tile 层次结构:

Grid (整个计算任务)
├── Block (Thread Block)
│   ├── Tile (计算块)
│   │   ├── Fragment (寄存器级)
│   │   └── Shared Memory
│   └── Thread (线程)
└── Warp (线程束)
    ├── Thread 0
    ├── Thread 1
    ├── ...
    └── Thread 31
```

这个代码块或示意图用于说明 1.4 Tile 的层次结构 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

```python
# Tile 层次结构的代码表示
@T.prim_func
def tile_hierarchy(A, B, C):
    # Grid 级: 整个计算任务
    for bx, by in T.grid(M // BLOCK_M, N // BLOCK_N):
        with T.block("block"):
            # Block 级: Thread Block
            vbx, vby = T.axis.spatial("SS", [bx, by])

            for tx, ty in T.grid(WARP_M, WARP_N):
                with T.block("tile"):
                    # Tile 级: 计算块
                    T.thread_binding(tx * WARP_N + ty, "warp")

                    C_frag = T.alloc_fragment((FRAG_M, FRAG_N), "float32")
                    # Fragment 级: 寄存器级数据
                    ...
```

以上代码展示了 Tile 的层次结构，从 Grid 级到 Block 级、Warp 级再到 Fragment 级，逐层细分计算任务。`T.thread_binding()` 将 Warp ID 绑定到硬件的 CUDA Warp，实现 Tile 维度到线程的映射。这种层次化的编程模型使得开发者可以在不同粒度上控制并行度和数据分布。

---

## 2. Thread Binding 机制

<div data-component="ThreadBindingVisualizer"></div>

### 2.1 什么是 Thread Binding？

Thread Binding 是 TileLang 中的一个关键机制，它定义了 **Tile 维度到硬件线程的映射关系**。

```
Thread Binding 示例:

Tile 维度: [M=128, N=128]
硬件线程: 256 threads (8 warps)

Binding 方案:
┌─────────────────────────────────────────┐
│  M 维度 (128)                           │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┐ │
│  │Warp0│Warp1│Warp2│Warp3│Warp4│Warp5│ │
│  │     │     │     │     │     │     │ │
│  │T0-T31    │T32-T63   │T64-T95   │ │
│  ├─────┼─────┼─────┼─────┼─────┼─────┤ │
│  │Warp6│Warp7│     │     │     │     │ │
│  │     │     │     │     │     │     │ │
│  └─────┴─────┴─────┴─────┴─────┴─────┘ │
│  N 维度 (128)                           │
└─────────────────────────────────────────┘
```

这个代码块或示意图用于说明 2.1 什么是 Thread Binding？ 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 2.2 T.thread_binding 语法

```python
# Thread Binding 基本语法
@T.prim_func
def thread_binding_example(A, B, C):
    BLOCK_M = 128
    BLOCK_N = 128

    for bx, by in T.grid(M // BLOCK_M, N // BLOCK_N):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])

            # 将 Tile 维度绑定到线程
            # 方式 1: 绑定到 Warp
            for warp_id in T.serial(8):
                with T.block("warp"):
                    # 绑定 warp_id 到 CUDA warp
                    T.thread_binding(warp_id, "warp")

                    # 每个 Warp 处理的数据
                    warp_m = warp_id // 4  # 0-1
                    warp_n = warp_id % 4   # 0-3

                    # Warp 内的计算
                    C_frag = T.alloc_fragment((16, 32), "float32")
                    ...
```

以上代码展示了 Thread Binding 的 Warp 级绑定方式。`T.thread_binding(warp_id, "warp")` 将循环索引绑定到 CUDA 的硬件 Warp，每个 Warp 独立处理一块数据。`warp_m` 和 `warp_n` 通过整除和取模运算计算每个 Warp 负责的矩阵区域，实现 Warp 级的数据分区。

```python
# Thread Binding 方式 2: 绑定到具体线程
@T.prim_func
def thread_binding_specific(A, B, C):
    BLOCK_M = 128
    BLOCK_N = 128

    for bx, by in T.grid(M // BLOCK_M, N // BLOCK_N):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])

            # 绑定到具体线程
            for tx, ty in T.grid(8, 32):
                with T.block("thread"):
                    # 绑定到 threadIdx.x
                    T.thread_binding(tx * 32 + ty, "threadIdx.x")

                    # 每个线程处理的数据
                    local_m = tx * 16
                    local_n = ty * 4

                    # 线程级计算
                    for i, j in T.grid(16, 4):
                        C[vbx * BLOCK_M + local_m + i,
                          vby * BLOCK_N + local_n + j] = ...
```

以上代码展示了 Thread Binding 的线程级绑定方式。`T.thread_binding(tx * 32 + ty, "threadIdx.x")` 将二维线程索引映射到一维的 threadIdx.x。每个线程负责一个 16×4 的小区域，通过 `local_m` 和 `local_n` 计算线程在矩阵中的位置。这种方式适合需要精细控制内存访问模式的场景。

### 2.3 Thread Binding 策略

| 策略 | 适用场景 | 优点 | 缺点 |
|:---|:---|:---|:---|
| **Warp 级绑定** | 计算密集型 | 简单，易管理 | 粒度粗 |
| **线程级绑定** | 内存密集型 | 精细控制 | 复杂 |
| **混合绑定** | 通用场景 | 平衡 | 需要经验 |

```python
# 策略 1: Warp 级绑定 (推荐新手)
@T.prim_func
def warp_binding(A, B, C):
    for bx, by in T.grid(M // 128, N // 128):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])

            # 绑定到 Warp
            for warp_id in T.serial(8):
                with T.block("warp"):
                    T.thread_binding(warp_id, "warp")
                    # Warp 内的计算由编译器自动处理
                    C_frag = T.alloc_fragment((16, 32), "float32")
                    ...

# 策略 2: 线程级绑定 (高级用户)
@T.prim_func
def thread_binding(A, B, C):
    for bx, by in T.grid(M // 128, N // 128):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])

            # 绑定到具体线程
            for tid in T.serial(256):
                with T.block("thread"):
                    T.thread_binding(tid, "threadIdx.x")
                    # 每个线程处理自己的数据
                    ...
```

以上代码对比了 Warp 级绑定和线程级绑定两种策略。Warp 级绑定（策略 1）适合新手，将整个 Warp 分配给一个计算单元，由编译器自动管理 Warp 内部的线程分配；线程级绑定（策略 2）适合高级用户，程序员完全控制每个线程的数据范围，实现更精细的优化。

### 2.4 Thread Binding 与性能

```python
# Thread Binding 对性能的影响

# 方案 1: 行优先绑定
# M 维度绑定到连续线程
# 适合: 行访问模式
for tx in T.serial(8):  # M 维度
    for ty in T.serial(32):  # N 维度
        T.thread_binding(tx * 32 + ty, "threadIdx.x")

# 方案 2: 列优先绑定
# N 维度绑定到连续线程
# 适合: 列访问模式
for ty in T.serial(32):  # N 维度
    for tx in T.serial(8):  # M 维度
        T.thread_binding(ty * 8 + tx, "threadIdx.x")

# 方案 3: 混合绑定
# 根据计算模式选择最优绑定
# 需要分析数据访问模式
```

以上代码展示了三种不同的 Thread Binding 方案对性能的影响。行优先绑定（方案 1）将 M 维度映射到连续线程，适合行访问模式；列优先绑定（方案 2）将 N 维度映射到连续线程，适合列访问模式；混合绑定（方案 3）根据实际计算模式选择最优方案，需要分析数据访问特征。

---

## 3. 数据并行与计算并行映射

### 3.1 并行层次

```
GPU 并行层次:

Grid
├── Thread Block (Block)
│   ├── Warp 0
│   │   ├── Thread 0
│   │   ├── Thread 1
│   │   └── ... (32 threads)
│   ├── Warp 1
│   └── ... (多个 Warp)
├── Thread Block 1
└── ... (多个 Block)

并行类型:
- Block 级并行: 不同 Block 之间完全独立
- Warp 级并行: 同一 Block 内不同 Warp 可以并行
- Thread 级并行: 同一 Warp 内 32 个线程并行执行
```

这个代码块或示意图用于说明 3.1 并行层次 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 3.2 数据并行映射

```python
# 数据并行: 将数据分块，每个块由一个并行单元处理

# 矩阵乘法的数据并行映射
@T.prim_func
def data_parallel(A, B, C):
    BLOCK_M = 128
    BLOCK_N = 128

    # 数据并行: 每个 Tile 独立计算
    for bx, by in T.grid(M // BLOCK_M, N // BLOCK_N):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])

            # Tile 内部的计算
            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.fill(C_local, T.float32(0))

            for k in T.serial(K // 32):
                A_shared = T.alloc_shared((BLOCK_M, 32), "float16")
                B_shared = T.alloc_shared((32, BLOCK_N), "float16")

                T.copy(A[...], A_shared)
                T.copy(B[...], B_shared)

                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[...])
```

以上代码展示了数据并行映射的基本模式。外层循环 `T.grid(M // BLOCK_M, N // BLOCK_N)` 遍历输出矩阵的每个 Tile，每个 Tile 独立计算，之间没有数据依赖。`T.alloc_fragment()` 分配寄存器级累加器，`T.gemm()` 执行 Tile 内部的矩阵乘法，实现高效的数据复用。

### 3.3 计算并行映射

```python
# 计算并行: 将计算任务分块，每个块由一个并行单元处理

# K 维度的计算并行
@T.prim_func
def compute_parallel(A, B, C):
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    for bx, by in T.grid(M // BLOCK_M, N // BLOCK_N):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])

            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.fill(C_local, T.float32(0))

            # K 维度的计算并行
            for k in T.serial(K // BLOCK_K):
                # 每个 K 块的计算
                A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
                B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")

                T.copy(A[...], A_shared)
                T.copy(B[...], B_shared)

                # Tile 内部的并行计算
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[...])
```

以上代码展示了计算并行映射，重点是 K 维度的分块计算。`T.serial(K // BLOCK_K)` 循环处理 K 维度的每个 Tile，每次迭代从 Global Memory 搬运一块数据到 Shared Memory，执行一次部分矩阵乘法并累加到 Fragment 中。这是 GEMM 中数据复用的核心机制。

### 3.4 并行度分析

```python
# 并行度计算
def calculate_parallelism(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
    """计算并行度"""
    # Block 级并行度
    num_blocks = (M // BLOCK_M) * (N // BLOCK_N)

    # Warp 级并行度 (每个 Block)
    warps_per_block = (BLOCK_M // 16) * (BLOCK_N // 16)  # 假设 Warp 处理 16x16

    # Thread 级并行度 (每个 Warp)
    threads_per_warp = 32

    # 总并行度
    total_threads = num_blocks * warps_per_block * threads_per_warp

    return {
        "num_blocks": num_blocks,
        "warps_per_block": warps_per_block,
        "total_threads": total_threads,
        "parallelism_ratio": total_threads / (M * N),  # 并行度比率
    }

# 示例: 4096x4096 矩阵乘法
result = calculate_parallelism(4096, 4096, 4096, 128, 128, 32)
print(f"Block 数量: {result['num_blocks']}")  # 1024
print(f"每 Block Warp 数: {result['warps_per_block']}")  # 64
print(f"总线程数: {result['total_threads']}")  # 2,097,152
```

以上代码展示了并行度计算函数，用于分析给定 Tile 大小下的 GPU 资源利用情况。`num_blocks` 是 Block 级并行度，`warps_per_block` 是每个 Block 的 Warp 数，`total_threads` 是总线程数，`parallelism_ratio` 衡量并行度与输出元素数的比值。这些指标帮助开发者评估当前配置是否充分利用了 GPU 的并行能力。

---

## 4. 多级数据类型体系

<div data-component="DataTypeHierarchy"></div>

### 4.1 数据类型概览

TileLang 定义了三级数据类型，对应 GPU 的内存层级：

```
┌─────────────────────────────────────────────┐
│              T.Buffer                        │  Global Memory
│              (全局缓冲区)                     │  HBM / GDDR
├─────────────────────────────────────────────┤
│              T.Shared                        │  Shared Memory
│              (共享缓冲区)                     │  SM 内 SRAM
├─────────────────────────────────────────────┤
│              T.L1                            │  L1 Cache
│              (L1 缓冲区)                     │  SM 内 Cache
├─────────────────────────────────────────────┤
│              T.Fragment                      │  Register
│              (片段缓冲区)                     │  每线程私有
└─────────────────────────────────────────────┘
```

这个代码块或示意图用于说明 4.1 数据类型概览 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 4.2 T.Buffer - Global Memory

```python
# T.Buffer: Global Memory 缓冲区
# 特点: 所有线程可见，容量大，带宽相对低

@T.prim_func
def buffer_example(
    # 输入 Buffer (Global Memory)
    A: T.Buffer[(M, K), "float16"],
    B: T.Buffer[(K, N), "float16"],
    # 输出 Buffer (Global Memory)
    C: T.Buffer[(M, N), "float32"],
):
    # Buffer 的属性
    # - shape: 形状 (M, K)
    # - dtype: 数据类型 "float16"
    # - scope: 内存位置 "global"

    # Buffer 的操作
    # - 读取: A[i, j]
    # - 写入: C[i, j] = value
    # - 切片: A[i:i+BLOCK, j:j+BLOCK]

    for i, j in T.grid(M, N):
        with T.block("compute"):
            vi, vj = T.axis.spatial("SS", [i, j])
            C[vi, vj] = A[vi, vj] + B[vi, vj]
```

以上代码展示了 T.Buffer 的基本用法，这是 TileLang 中表示 Global Memory 数据的核心类型。`T.Buffer` 的参数包括形状（如 `(M, K)`）和数据类型（如 `"float16"`），支持读取、写入和切片操作。Buffer 中的数据存储在 GPU 的 Global Memory（HBM）中，所有线程都可以访问，但带宽相对较低。

### 4.3 T.alloc_shared - Shared Memory

```python
# T.alloc_shared: Shared Memory 分配
# 特点: Block 内线程可见，容量有限，带宽高

@T.prim_func
def shared_example(A, B, C):
    BLOCK = 128

    for bx, by in T.grid(M // BLOCK, N // BLOCK):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])

            # 分配 Shared Memory
            # 大小: 128 × 32 × 2 bytes = 8 KB
            A_shared = T.alloc_shared((BLOCK, 32), "float16")

            # Shared Memory 的特点:
            # - Block 内所有线程共享
            # - 需要同步 (T.sync_threads())
            # - 容量有限 (A100: 164 KB/SM)
            # - 带宽高 (~20 TB/s)

            # 从 Global Memory 搬运到 Shared Memory
            T.copy(A[vbx * BLOCK:(vbx + 1) * BLOCK, 0:32], A_shared)

            # 同步: 确保所有线程都完成了数据搬运
            T.sync_threads()

            # 使用 Shared Memory 进行计算
            for i in T.serial(BLOCK):
                for j in T.serial(32):
                    # 读取 Shared Memory (高带宽)
                    val = A_shared[i, j]
                    ...
```

以上代码展示了 Shared Memory 的分配和使用方式。`T.alloc_shared((BLOCK, 32), "float16")` 分配一块 8KB 的共享内存（128×32×2 字节），Block 内所有线程都可以访问。关键步骤包括：先用 `T.copy()` 将数据从 Global Memory 搬运到 Shared Memory，然后用 `T.sync_threads()` 确保所有线程完成搬运，最后在 Shared Memory 上进行高带宽计算。

### 4.4 T.alloc_L1 - L1 Cache

```python
# T.alloc_L1: L1 Cache 分配
# 特点: 部分硬件支持，与 Shared Memory 共享物理存储

@T.prim_func
def l1_example(A, B, C):
    # 在 NVIDIA GPU 上:
    # - L1 Cache 和 Shared Memory 共享同一块物理存储
    # - 可以配置两者的比例
    # - L1 Cache 由硬件自动管理

    # 分配 L1 Cache
    A_l1 = T.alloc_L1((128, 32), "float16")

    # L1 Cache 的特点:
    # - 硬件自动管理
    # - 对程序员透明
    # - 提供缓存功能
    # - 在某些硬件上提供更高带宽

    # 使用 L1 Cache
    # 数据访问会自动经过 L1 Cache
    ...
```

以上代码展示了 L1 Cache 的分配和使用方式。`T.alloc_L1()` 分配的缓冲区由硬件自动管理，程序员无需显式搬运数据。在 NVIDIA GPU 上，L1 Cache 和 Shared Memory 共享同一块物理存储（164-228 KB/SM），适合数据访问模式不规则或局部性较好的场景。

### 4.5 T.alloc_fragment - Register/Fragment

```python
# T.alloc_fragment: Register/Fragment 分配
# 特点: 每线程私有，容量最小，带宽最高

@T.prim_func
def fragment_example(A, B, C):
    BLOCK = 128

    for bx, by in T.grid(M // BLOCK, N // BLOCK):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])

            # 分配 Fragment (寄存器)
            # 大小: 16 × 16 × 4 bytes = 1 KB (每线程)
            C_frag = T.alloc_fragment((16, 16), "float32")

            # Fragment 的特点:
            # - 每线程私有，无需同步
            # - 容量最小 (A100: 255 寄存器/线程)
            # - 带宽最高 (~数十 TB/s)
            # - 生命周期: Block 执行期间

            # 初始化 Fragment
            T.fill(C_frag, T.float32(0))

            # 从 Shared Memory 加载到 Fragment
            A_frag = T.alloc_fragment((16, 32), "float16")
            T.copy(A_shared, A_frag)

            # Fragment 级计算 (最快)
            for i, j in T.grid(16, 16):
                for k in T.serial(32):
                    C_frag[i, j] += T.cast(A_frag[i, k], "float32") * \
                                    T.cast(B_frag[k, j], "float32")
```

以上代码展示了 Fragment（寄存器级存储）的分配和使用。`T.alloc_fragment((16, 16), "float32")` 为每个线程分配一个 16×16 的 FP32 累加器（1KB/线程）。Fragment 的特点是每线程私有、无需同步、带宽最高（~数十 TB/s）。`T.fill()` 用于初始化，`T.copy()` 从 Shared Memory 加载数据，最终在 Fragment 级别执行矩阵乘法累加。

### 4.6 数据类型对比表

| 数据类型 | 内存位置 | 容量 | 带宽 | 可见性 | 同步需求 |
|:---|:---|:---|:---|:---|:---|
| **T.Buffer** | Global (HBM) | 数 TB | ~3 TB/s | 所有线程 | 无需 |
| **T.alloc_shared** | Shared Memory | 164 KB/SM | ~20 TB/s | Block 内 | 需要 sync |
| **T.alloc_L1** | L1 Cache | 与 SMEM 共享 | ~20 TB/s | Block 内 | 硬件管理 |
| **T.alloc_fragment** | Register | 255 regs/thread | ~数十 TB/s | 每线程 | 无需 |

---

## 5. Tile Shape 选择策略

<div data-component="TileShapeSelectionGuide"></div>

### 5.1 Tile Shape 的重要性

Tile Shape（块形状）的选择直接影响：
1. **数据局部性**：Tile 越大，数据复用率越高
2. **并行度**：Tile 越小，并行度越高
3. **内存使用**：Tile 越大，Shared Memory 使用越多
4. **Occupancy**：Tile 越大，每个 SM 能容纳的 Block 越少

### 5.2 常见 Tile Shape

```python
# 常见的 Tile Shape 配置

# 配置 1: 方形 Tile (适合矩阵乘法)
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32

# 配置 2: 矩形 Tile (适合长序列)
BLOCK_M = 256
BLOCK_N = 64
BLOCK_K = 32

# 配置 3: 小 Tile (适合小矩阵)
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 16

# 配置 4: 大 Tile (适合大矩阵，高 Occupancy)
BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 64
```

以上代码展示了四种常见的 Tile Shape 配置。方形 Tile（128×128）适合通用矩阵乘法；矩形 Tile（256×64）适合长序列场景；小 Tile（64×64）适合小矩阵以提高并行度；大 Tile（256×256）适合大矩阵以最大化数据复用。选择时需要在数据复用率和 Occupancy 之间权衡。

### 5.3 Tile Shape 选择原则

| 原则 | 说明 | 示例 |
|:---|:---|:---|
| **对齐** | Tile 大小应是硬件友好的倍数 | 32, 64, 128 |
| **平衡** | M, N, K 维度应平衡 | 128×128×32 |
| **适配** | 适配目标硬件的特性 | A100: 128×128 |
| **实验** | 通过实验找到最优配置 | Auto-tuning |

```python
# Tile Shape 选择示例
def choose_tile_shape(M, N, K, hardware="A100"):
    """
    根据矩阵大小和硬件选择 Tile Shape
    """
    if hardware == "A100":
        # A100 的特性:
        # - Shared Memory: 164 KB/SM
        # - 寄存器: 65536/SM
        # - 最大线程数: 2048/SM

        # 计算最优 Tile Shape
        # 目标: 最大化 Occupancy

        # 方案 1: 128×128×32
        # Shared Memory: 128×32×2 + 32×128×2 = 16 KB
        # 寄存器: 128×128×4 = 64 KB
        # 可以放 2 个 Block/SM

        # 方案 2: 256×64×32
        # Shared Memory: 256×32×2 + 32×64×2 = 12 KB
        # 寄存器: 256×64×4 = 64 KB
        # 可以放 2 个 Block/SM

        if M >= 4096 and N >= 4096:
            return (256, 256, 64)  # 大矩阵用大 Tile
        elif M >= 1024 and N >= 1024:
            return (128, 128, 32)  # 中等矩阵
        else:
            return (64, 64, 16)    # 小矩阵用小 Tile

    elif hardware == "H100":
        # H100 的特性:
        # - Shared Memory: 228 KB/SM
        # - 更多寄存器
        # - 更高带宽

        if M >= 4096 and N >= 4096:
            return (256, 256, 64)
        else:
            return (128, 128, 32)
```

以上代码展示了根据矩阵大小和硬件特性自动选择 Tile Shape 的策略。对于 A100，大矩阵（≥4096）使用 256×256×64 的大 Tile 以最大化数据复用，中等矩阵使用 128×128×32，小矩阵使用 64×64×16 以提高并行度。H100 由于 Shared Memory 更大（228KB），可以支持更大的 Tile 配置。

### 5.4 Tile Shape 与性能

```python
# Tile Shape 对性能的影响

def benchmark_tile_shapes(M=4096, N=4096, K=4096):
    """测试不同 Tile Shape 的性能"""
    shapes = [
        (64, 64, 16),
        (128, 128, 32),
        (256, 256, 64),
        (128, 256, 32),
        (256, 128, 32),
    ]

    results = {}
    for BLOCK_M, BLOCK_N, BLOCK_K in shapes:
        # 创建 kernel
        @T.prim_func
        def gemm(A, B, C):
            for bx, by in T.grid(M // BLOCK_M, N // BLOCK_N):
                with T.block("C"):
                    vbx, vby = T.axis.spatial("SS", [bx, by])
                    C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
                    T.fill(C_local, T.float32(0))
                    for k in T.serial(K // BLOCK_K):
                        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
                        B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
                        T.copy(A[...], A_shared)
                        T.copy(B[...], B_shared)
                        T.gemm(A_shared, B_shared, C_local)
                    T.copy(C_local, C[...])

        # 测试性能
        kernel = tilelang.compile(gemm, target="cuda")
        avg_ms = benchmark_kernel(kernel, M, N, K)
        results[(BLOCK_M, BLOCK_N, BLOCK_K)] = avg_ms

    return results

# 典型结果:
# (64, 64, 16):   8.2 ms
# (128, 128, 32): 4.1 ms  ← 最优
# (256, 256, 64): 5.3 ms  (Occupancy 下降)
# (128, 256, 32): 4.5 ms
# (256, 128, 32): 4.8 ms
```

以上代码展示了通过 Benchmark 测试不同 Tile Shape 性能的方法。典型结果显示 128×128×32 是最优配置（4.1ms），而 256×256×64 虽然数据复用率更高，但由于 Occupancy 下降反而性能较差（5.3ms）。这说明 Tile Shape 的选择需要在数据复用和 GPU 利用率之间找到平衡点。

---

## 6. Grid/Block/Thread 映射关系

### 6.1 映射概览

```
Grid/Block/Thread 映射:

Grid (整个计算任务)
├── Block (0, 0)
│   ├── Thread (0, 0)
│   ├── Thread (0, 1)
│   └── ... (256 threads)
├── Block (0, 1)
│   └── ...
├── Block (1, 0)
│   └── ...
└── Block (M//BLOCK_M - 1, N//BLOCK_N - 1)
    └── ...

映射关系:
- Tile (bx, by) → Block (blockIdx.x, blockIdx.y)
- Thread (tx, ty) → Thread (threadIdx.x, threadIdx.y)
```

这个代码块或示意图用于说明 6.1 映射概览 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 6.2 TileLang 的映射机制

```python
# TileLang 的 Grid/Block/Thread 映射

@T.prim_func
def grid_block_thread(A, B, C):
    BLOCK_M = 128
    BLOCK_N = 128

    # Grid 级循环: 映射到 Block
    for bx, by in T.grid(M // BLOCK_M, N // BLOCK_N):
        with T.block("C"):
            # bx, by 映射到 blockIdx.x, blockIdx.y
            vbx, vby = T.axis.spatial("SS", [bx, by])

            # Block 内的线程映射
            for tx, ty in T.grid(8, 32):
                with T.block("thread"):
                    # tx, ty 映射到 threadIdx.x
                    T.thread_binding(tx * 32 + ty, "threadIdx.x")

                    # 每个线程处理的数据
                    local_m = vbx * BLOCK_M + tx * 16
                    local_n = vby * BLOCK_N + ty * 4

                    # 线程级计算
                    for i, j in T.grid(16, 4):
                        C[local_m + i, local_n + j] = ...
```

以上代码展示了 TileLang 的 Grid/Block/Thread 映射机制。外层 `T.grid()` 循环映射到 GPU 的 blockIdx，`T.axis.spatial()` 将循环变量绑定到 Block 维度；内层 `T.grid()` 配合 `T.thread_binding()` 将线程映射到 threadIdx.x。每个线程通过 `local_m` 和 `local_n` 计算自己负责的输出矩阵位置。

### 6.3 映射策略

| 策略 | 适用场景 | 优点 | 缺点 |
|:---|:---|:---|:---|
| **行优先** | 行访问模式 | 内存合并 | 列访问慢 |
| **列优先** | 列访问模式 | 内存合并 | 行访问慢 |
| **Z 形** | 2D 访问 | 平衡 | 复杂 |
| **Hilbert** | 空间局部性 | 最优局部性 | 实现复杂 |

```python
# 行优先映射
@T.prim_func
def row_major(A, B, C):
    for bx, by in T.grid(M // 128, N // 128):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])

            # 行优先: M 维度映射到连续线程
            for tx in T.serial(8):      # M 维度
                for ty in T.serial(32):  # N 维度
                    T.thread_binding(tx * 32 + ty, "threadIdx.x")
                    ...

# 列优先映射
@T.prim_func
def col_major(A, B, C):
    for bx, by in T.grid(M // 128, N // 128):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])

            # 列优先: N 维度映射到连续线程
            for ty in T.serial(32):  # N 维度
                for tx in T.serial(8):      # M 维度
                    T.thread_binding(ty * 8 + tx, "threadIdx.x")
                    ...
```

以上代码对比了行优先和列优先两种映射策略。行优先映射将 M 维度（行）绑定到连续线程，适合行访问模式，能实现内存合并访问；列优先映射将 N 维度（列）绑定到连续线程，适合列访问模式。选择哪种策略取决于矩阵的实际访问模式。

---

## 7. Tile 编程模型的最佳实践

### 7.1 Tile 大小选择

```python
# 最佳实践: Tile 大小选择

def select_tile_size(M, N, K, dtype="float16"):
    """
    选择最优的 Tile 大小

    考虑因素:
    1. 矩阵大小
    2. 数据类型
    3. 硬件特性
    4. 内存限制
    """
    # 计算每个元素的字节数
    if dtype == "float16":
        elem_bytes = 2
    elif dtype == "float32":
        elem_bytes = 4

    # A100 的限制
    max_shared_mem = 164 * 1024  # 164 KB
    max_registers = 65536  # 每 SM

    # 计算最优 Tile 大小
    # 目标: 最大化数据复用，同时满足内存限制

    # Shared Memory 使用:
    # A_shared: BLOCK_M × BLOCK_K × elem_bytes
    # B_shared: BLOCK_K × BLOCK_N × elem_bytes
    # 总计: (BLOCK_M + BLOCK_N) × BLOCK_K × elem_bytes

    # 寄存器使用:
    # C_frag: BLOCK_M × BLOCK_N × 4 (float32)

    # 选择策略
    if M >= 4096 and N >= 4096:
        # 大矩阵: 使用大 Tile
        BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
    elif M >= 1024 and N >= 1024:
        # 中等矩阵: 使用中等 Tile
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    else:
        # 小矩阵: 使用小 Tile
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 16

    # 验证内存限制
    shared_mem = (BLOCK_M + BLOCK_N) * BLOCK_K * elem_bytes
    registers = BLOCK_M * BLOCK_N * 4

    if shared_mem > max_shared_mem:
        # 减小 Tile 大小
        BLOCK_K = max_shared_mem // ((BLOCK_M + BLOCK_N) * elem_bytes)

    return BLOCK_M, BLOCK_N, BLOCK_K
```

以上代码展示了 Tile 大小的自动选择策略。函数 `select_tile_size()` 根据矩阵大小（M、N、K）和数据类型（float16/float32）计算最优的 Tile 配置。核心逻辑是：大矩阵（≥4096）使用 256×256×64 大 Tile 以最大化数据复用；中等矩阵（≥1024）使用 128×128×32；小矩阵使用 64×64×16 以提高并行度。同时会验证 Shared Memory 和寄存器使用是否超出 A100 硬件限制（164KB Shared Memory，65536 寄存器/SM），超出时自动缩减 K 维度的 Tile 大小。

### 7.2 Thread Binding 最佳实践

```python
# 最佳实践: Thread Binding

def select_thread_binding(BLOCK_M, BLOCK_N, num_warps):
    """
    选择最优的 Thread Binding

    考虑因素:
    1. Tile 大小
    2. Warp 数量
    3. 数据访问模式
    4. 硬件特性
    """
    # 计算每个 Warp 处理的数据
    WARP_M = BLOCK_M // (num_warps // (BLOCK_N // 16))
    WARP_N = 16  # 通常每个 Warp 处理 16 列

    # Thread Binding 策略
    if BLOCK_M > BLOCK_N:
        # M 维度较大: 行优先绑定
        binding = "row_major"
    elif BLOCK_N > BLOCK_M:
        # N 维度较大: 列优先绑定
        binding = "col_major"
    else:
        # 方形: Z 形绑定
        binding = "z_order"

    return WARP_M, WARP_N, binding
```

以上代码展示了 Thread Binding 的最优选择策略。函数 `select_thread_binding()` 根据 Tile 的 M 和 N 维度大小关系选择绑定方式：当 BLOCK_M > BLOCK_N 时选择行优先绑定，将 M 维度映射到连续线程以优化行访问；当 BLOCK_N > BLOCK_M 时选择列优先绑定；方形 Tile 则使用 Z 形绑定以平衡两个维度的内存合并。该策略确保了线程访问内存时能实现高效的合并访问，减少内存事务次数。

### 7.3 内存管理最佳实践

```python
# 最佳实践: 内存管理

@T.prim_func
def memory_best_practice(A, B, C):
    """
    内存管理最佳实践

    1. 使用 Shared Memory 减少 Global 访问
    2. 使用 Fragment 减少 Shared Memory 访问
    3. 合理设置 Tile 大小
    4. 避免 Bank Conflict
    """
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    for bx, by in T.grid(M // BLOCK_M, N // BLOCK_N):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])

            # 1. 分配 Fragment (寄存器)
            C_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.fill(C_frag, T.float32(0))

            # 2. K 维度循环
            for k in T.serial(K // BLOCK_K):
                # 3. 分配 Shared Memory
                A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
                B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")

                # 4. 从 Global 搬运到 Shared
                T.copy(A[...], A_shared)
                T.copy(B[...], B_shared)

                # 5. 从 Shared 搬运到 Fragment
                A_frag = T.alloc_fragment((BLOCK_M, BLOCK_K), "float16")
                B_frag = T.alloc_fragment((BLOCK_K, BLOCK_N), "float16")
                T.copy(A_shared, A_frag)
                T.copy(B_shared, B_frag)

                # 6. Fragment 级计算
                for i, j, k_inner in T.grid(BLOCK_M, BLOCK_N, BLOCK_K):
                    C_frag[i, j] += T.cast(A_frag[i, k_inner], "float32") * \
                                    T.cast(B_frag[k_inner, j], "float32")

            # 7. 从 Fragment 搬运回 Global
            T.copy(C_frag, C[...])
```

以上代码展示了内存管理的最佳实践模式。核心流程是：通过 `T.grid()` 遍历输出 Tile、分配 Fragment 作为寄存器级累加器、K 维度循环中每次分配 Shared Memory 缓冲区、从 Global Memory 搬运数据到 Shared Memory（T.copy）、再从 Shared Memory 搬运到 Fragment、最后在 Fragment 级别执行矩阵乘法累加并写回 Global Memory。这种三级数据流（Global→Shared→Fragment）最大化利用了 GPU 的内存层次，将高频数据访问留在带宽最高的层级。

---

## ✅ 本章总结

### 核心要点

🎯 **Tile 是一等公民**：
- Tile 是数据和计算的基本单位
- TileLang 以 Tile 为中心设计，而非以线程为中心
- Tile 级抽象简化了 GPU 编程

🎯 **Thread Binding**：
- 定义 Tile 维度到硬件线程的映射
- 支持 Warp 级和线程级绑定
- 影响数据局部性和性能

🎯 **数据并行与计算并行**：
- 数据并行: 不同 Tile 独立计算
- 计算并行: Tile 内部并行计算
- 两种并行可以组合使用

🎯 **多级数据类型**：
- T.Buffer: Global Memory，容量大，带宽低
- T.alloc_shared: Shared Memory，容量中，带宽高
- T.alloc_fragment: Register，容量小，带宽最高

🎯 **Tile Shape 选择**：
- 考虑矩阵大小、硬件特性、内存限制
- 通过实验找到最优配置
- 常见配置: 128×128×32

### 关键数字

| 概念 | 典型值 | 说明 |
|:---|:---|:---|
| Tile 大小 | 128×128 | 常见配置 |
| BLOCK_K | 32 | K 维度 Tile |
| Warp 大小 | 32 线程 | NVIDIA 标准 |
| Shared Memory | 164 KB/SM | A100 |
| 寄存器 | 255/线程 | 最大值 |

---

## 📝 练习题

### 练习 1：Tile 概念理解

1. 解释为什么 TileLang 选择 Tile 作为一等公民，而非线程。
2. 画出 256×256 矩阵使用 64×64 Tile 的分块示意图。
3. 计算使用 128×128×32 Tile 进行 4096×4096×4096 GEMM 的 Tile 数量。

### 练习 2：Thread Binding 实践

1. 为 128×128 的 Tile 设计一个 Thread Binding 方案（使用 256 线程）。
2. 实现行优先和列优先两种绑定方案。
3. 测试两种方案的性能差异。

### 练习 3：数据类型选择

1. 解释 T.alloc_shared 和 T.alloc_fragment 的区别。
2. 为什么需要从 Global → Shared → Fragment 的数据搬运？
3. 设计一个内存使用方案，满足 A100 的限制。

### 练习 4：Tile Shape 优化

1. 测试不同 Tile Shape (64×64, 128×128, 256×256) 的性能。
2. 分析性能差异的原因。
3. 为你的目标硬件找到最优的 Tile Shape。

---

## 🔗 扩展阅读

- [TileLang Tile 编程模型](https://tile-ai.github.io/tilelang/tile-model)
- [CUDA Thread Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy)
- [Shared Memory 最佳实践](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [GPU 架构与性能优化](https://developer.nvidia.com/blog/cuda-tuning-guides/)

---

## 8. Tile 编程模型的高级主题

### 8.1 动态 Tile 大小

```python
# 动态 Tile 大小: 根据输入大小自适应调整

@T.prim_func
def dynamic_tile(
    A: T.Buffer[(M, K), "float16"],
    B: T.Buffer[(K, N), "float16"],
    C: T.Buffer[(M, N), "float32"],
):
    """
    动态 Tile 大小选择
    根据矩阵大小自动选择最优的 Tile 配置
    """
    # 在 TileLang 中，Tile 大小通常在编译时确定
    # 但可以通过条件编译实现动态选择

    # 方案 1: 使用编译时常量
    BLOCK_M = 128  # 可以通过参数传入
    BLOCK_N = 128
    BLOCK_K = 32

    for bx, by in T.grid(M // BLOCK_M, N // BLOCK_N):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])
            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.fill(C_local, T.float32(0))
            for k in T.serial(K // BLOCK_K):
                A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
                B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
                T.copy(A[...], A_shared)
                T.copy(B[...], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[...])
```

以上代码展示了动态 Tile 大小的基本概念。在 TileLang 中，Tile 大小通常在编译时确定，但可以通过编译时常量参数实现动态选择。`dynamic_tile()` 函数使用固定的 BLOCK_M/N/K 配置定义计算流程，外层循环遍历矩阵的 Tile 划分，内层循环在 K 维度串行处理数据搬运和矩阵乘法。虽然这段代码本身未实现真正的运行时动态选择，但展示了如何将 Tile 大小参数化，实际的动态选择需要在更高层通过条件编译或多版本 Kernel 实现。

### 8.2 非对称 Tile

```python
# 非对称 Tile: M 和 N 维度使用不同的 Tile 大小

@T.prim_func
def asymmetric_tile(
    A: T.Buffer[(M, K), "float16"],
    B: T.Buffer[(K, N), "float16"],
    C: T.Buffer[(M, N), "float32"],
):
    """
    非对称 Tile 配置
    适用于 M >> N 或 N >> M 的场景
    """
    BLOCK_M = 256  # M 维度使用大 Tile
    BLOCK_N = 64   # N 维度使用小 Tile
    BLOCK_K = 32

    for bx, by in T.grid(M // BLOCK_M, N // BLOCK_N):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])
            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.fill(C_local, T.float32(0))
            for k in T.serial(K // BLOCK_K):
                A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
                B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
                T.copy(A[...], A_shared)
                T.copy(B[...], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[...])
```

以上代码展示了非对称 Tile 配置的使用场景。当矩阵的 M 和 N 维度差异很大时（如 M >> N 或 N >> M），使用非对称 Tile（BLOCK_M=256, BLOCK_N=64）比方形 Tile 更高效。大维度使用更大的 Tile 以最大化数据复用，小维度使用较小的 Tile 以避免浪费计算资源。这种配置在处理长序列或瘦矩阵时特别有用，能够根据实际数据形状定制化分配 GPU 资源。

### 8.3 多级 Tile

```python
# 多级 Tile: Tile 内部再分 Tile

@T.prib_func
def multi_level_tile(
    A: T.Buffer[(M, K), "float16"],
    B: T.Buffer[(K, N), "float16"],
    C: T.Buffer[(M, N), "float32"],
):
    """
    多级 Tile 结构
    Tile → Sub-Tile → Fragment
    """
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    # 第一级 Tile: Block 级
    for bx, by in T.grid(M // BLOCK_M, N // BLOCK_N):
        with T.block("block"):
            vbx, vby = T.axis.spatial("SS", [bx, by])

            # 第二级 Tile: Warp 级
            WARP_M = 32
            WARP_N = 32
            for wx, wy in T.grid(BLOCK_M // WARP_M, BLOCK_N // WARP_N):
                with T.block("warp"):
                    # Warp 级计算
                    C_frag = T.alloc_fragment((WARP_M, WARP_N), "float32")
                    T.fill(C_frag, T.float32(0))

                    for k in T.serial(K // BLOCK_K):
                        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
                        B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
                        T.copy(A[...], A_shared)
                        T.copy(B[...], B_shared)

                        # Warp 级 GEMM
                        A_frag = T.alloc_fragment((WARP_M, BLOCK_K), "float16")
                        B_frag = T.alloc_fragment((BLOCK_K, WARP_N), "float16")
                        T.copy(A_shared[wx*WARP_M:(wx+1)*WARP_M, :], A_frag)
                        T.copy(B_shared[:, wy*WARP_N:(wy+1)*WARP_N], B_frag)
                        T.gemm(A_frag, B_frag, C_frag)
```

以上代码展示了多级 Tile 结构，即 Block 级 → Warp 级 → Fragment 级的层次化分块。第一级将矩阵按 128×128 划分到不同 Block；第二级在每个 Block 内按 32×32 划分到不同 Warp；每个 Warp 独立加载数据（A_frag、B_frag）并执行局部 GEMM。这种层次化设计能更好地利用数据局部性——Warp 内的 Shared Memory 访问比跨 Warp 更高效，同时 Warp 级并行天然支持 Tensor Core 的矩阵运算粒度。

### 8.4 Tile 边界处理

```python
# Tile 边界处理: 当矩阵大小不是 Tile 大小的整数倍

@T.prim_func
def tile_boundary(
    A: T.Buffer[(M, K), "float16"],
    B: T.Buffer[(K, N), "float16"],
    C: T.Buffer[(M, N), "float32"],
):
    """
    处理边界情况
    当 M, N, K 不是 BLOCK_M, BLOCK_N, BLOCK_K 的整数倍时
    """
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    # 计算需要处理的 Tile 数量
    num_tiles_m = (M + BLOCK_M - 1) // BLOCK_M  # 向上取整
    num_tiles_n = (N + BLOCK_N - 1) // BLOCK_N

    for bx, by in T.grid(num_tiles_m, num_tiles_n):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])

            # 计算当前 Tile 的实际大小
            tile_m = T.min(BLOCK_M, M - vbx * BLOCK_M)
            tile_n = T.min(BLOCK_N, N - vby * BLOCK_N)

            # 分配 Fragment
            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.fill(C_local, T.float32(0))

            for k in T.serial((K + BLOCK_K - 1) // BLOCK_K):
                A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
                B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")

                # 边界检查
                tile_k = T.min(BLOCK_K, K - k * BLOCK_K)

                # 带边界检查的数据搬运
                for i, j in T.grid(tile_m, tile_k):
                    A_shared[i, j] = A[vbx * BLOCK_M + i, k * BLOCK_K + j]
                for i, j in T.grid(tile_k, tile_n):
                    B_shared[i, j] = B[k * BLOCK_K + i, vby * BLOCK_N + j]

                T.sync_threads()
                T.gemm(A_shared, B_shared, C_local)

            # 写回结果 (带边界检查)
            for i, j in T.grid(tile_m, tile_n):
                C[vbx * BLOCK_M + i, vby * BLOCK_N + j] = C_local[i, j]
```

以上代码展示了 Tile 边界处理的关键方法。当矩阵维度不是 Tile 大小的整数倍时，需要对边界 Tile 进行特殊处理。代码使用向上取整 `(M + BLOCK_M - 1) // BLOCK_M` 确保覆盖所有元素，通过 `T.min(BLOCK_M, M - vbx * BLOCK_M)` 动态计算当前 Tile 的实际大小。在数据搬运阶段使用精确的循环边界 `tile_m` 和 `tile_n` 避免越界访问，且插入 `T.sync_threads()` 确保所有线程完成数据搬运后再执行矩阵乘法计算。

---

## 9. Tile 编程模型的性能调优

### 9.1 Tile 大小调优

```python
# Tile 大小调优策略

def tune_tile_size(M, N, K, dtype="float16"):
    """
    自动调优 Tile 大小

    搜索空间:
    - BLOCK_M: [64, 128, 256]
    - BLOCK_N: [64, 128, 256]
    - BLOCK_K: [16, 32, 64]
    """
    import itertools

    # 搜索空间
    block_m_options = [64, 128, 256]
    block_n_options = [64, 128, 256]
    block_k_options = [16, 32, 64]

    best_config = None
    best_time = float('inf')

    for BLOCK_M, BLOCK_N, BLOCK_K in itertools.product(
        block_m_options, block_n_options, block_k_options
    ):
        # 检查约束
        # 1. Tile 大小不能超过矩阵大小
        if BLOCK_M > M or BLOCK_N > N or BLOCK_K > K:
            continue

        # 2. Shared Memory 不能超出限制
        if dtype == "float16":
            shared_mem = (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * 2
        else:
            shared_mem = (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * 4

        if shared_mem > 164 * 1024:  # A100: 164 KB
            continue

        # 3. 编译并测试
        try:
            @T.prim_func
            def gemm(A, B, C):
                for bx, by in T.grid(M // BLOCK_M, N // BLOCK_N):
                    with T.block("C"):
                        vbx, vby = T.axis.spatial("SS", [bx, by])
                        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
                        T.fill(C_local, T.float32(0))
                        for k in T.serial(K // BLOCK_K):
                            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
                            B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
                            T.copy(A[...], A_shared)
                            T.copy(B[...], B_shared)
                            T.gemm(A_shared, B_shared, C_local)
                        T.copy(C_local, C[...])

            kernel = tilelang.compile(gemm, target="cuda")

            # 性能测试
            A = torch.randn(M, K, dtype=torch.float16, device="cuda")
            B = torch.randn(K, N, dtype=torch.float16, device="cuda")
            C = torch.zeros(M, N, dtype=torch.float32, device="cuda")

            # Warmup
            for _ in range(10):
                kernel(A, B, C)
            torch.cuda.synchronize()

            # 测试
            import time
            start = time.perf_counter()
            for _ in range(100):
                kernel(A, B, C)
            torch.cuda.synchronize()
            end = time.perf_counter()

            avg_ms = (end - start) / 100 * 1000

            if avg_ms < best_time:
                best_time = avg_ms
                best_config = (BLOCK_M, BLOCK_N, BLOCK_K)

        except Exception as e:
            continue

    return best_config, best_time

# 使用示例
config, time_ms = tune_tile_size(4096, 4096, 4096)
print(f"最优配置: {config}")
print(f"性能: {time_ms:.2f} ms")
```

以上代码展示了完整的 Tile 大小自动调优流程。`tune_tile_size()` 函数遍历 BLOCK_M、BLOCK_N、BLOCK_K 的全部候选组合（共 27 种），对每个配置进行约束检查（矩阵大小约束、Shared Memory 不超过 164KB），满足条件的配置通过 `tilelang.compile()` 编译并在 GPU 上运行 Benchmark。测试包含 10 次 Warmup 预热和 100 次迭代取平均，筛选出性能最优的 Tile 配置。这种网格搜索方法虽然简单，但对于关键 Kernel 的调优非常有效。

### 9.2 Occupancy 优化

```python
# Occupancy 优化: 最大化 GPU 利用率

def calculate_occupancy(BLOCK_M, BLOCK_N, BLOCK_K, num_warps, dtype="float16"):
    """
    计算 Occupancy

    Occupancy = 活跃 Warp 数 / 最大 Warp 数
    """
    # A100 参数
    max_warps_per_sm = 64  # 最大 Warp 数/SM
    max_blocks_per_sm = 32  # 最大 Block 数/SM
    max_shared_mem_per_sm = 164 * 1024  # 164 KB
    max_registers_per_sm = 65536  # 寄存器数/SM

    # 计算每个 Block 的资源使用
    # Shared Memory
    if dtype == "float16":
        shared_mem_per_block = (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * 2
    else:
        shared_mem_per_block = (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * 4

    # 寄存器 (简化估算)
    registers_per_thread = BLOCK_M * BLOCK_N * 4 // (num_warps * 32)
    registers_per_block = registers_per_thread * num_warps * 32

    # 计算每个 SM 能容纳的 Block 数
    blocks_by_shared_mem = max_shared_mem_per_sm // shared_mem_per_block
    blocks_by_registers = max_registers_per_sm // registers_per_block
    blocks_by_limit = max_blocks_per_sm

    blocks_per_sm = min(blocks_by_shared_mem, blocks_by_registers, blocks_by_limit)

    # 计算 Occupancy
    warps_per_block = num_warps
    active_warps = blocks_per_sm * warps_per_block
    occupancy = active_warps / max_warps_per_sm

    return {
        "blocks_per_sm": blocks_per_sm,
        "active_warps": active_warps,
        "occupancy": occupancy,
        "shared_mem_per_block": shared_mem_per_block,
        "registers_per_block": registers_per_block,
    }

# 测试不同配置的 Occupancy
configs = [
    (64, 64, 16, 4),
    (128, 128, 32, 8),
    (256, 256, 64, 16),
]

for BLOCK_M, BLOCK_N, BLOCK_K, num_warps in configs:
    result = calculate_occupancy(BLOCK_M, BLOCK_N, BLOCK_K, num_warps)
    print(f"Config ({BLOCK_M}, {BLOCK_N}, {BLOCK_K}): "
          f"Occupancy={result['occupancy']:.1%}, "
          f"Blocks/SM={result['blocks_per_sm']}")
```

以上代码展示了如何计算和优化 GPU Occupancy。`calculate_occupancy()` 函数从三个维度计算每个 SM 可容纳的 Block 数：Shared Memory 约束（blocks_by_shared_mem）、寄存器约束（blocks_by_registers）和硬件 Block 上限（blocks_by_limit），取三者最小值作为实际 blocks_per_sm。Occupancy 等于活跃 Warp 数除以最大 Warp 数（A100 为 64）。高 Occupancy 意味着更好的延迟隐藏能力，但更大的 Tile 虽然降低 Occupancy，却能通过数据复用提升整体性能。

### 9.3 内存带宽优化

```python
# 内存带宽优化

def analyze_memory_bandwidth(M, N, K, avg_ms, dtype="float16"):
    """
    分析内存带宽利用率
    """
    # 计算理论带宽
    # A100: 2 TB/s (HBM2e)
    theoretical_bandwidth = 2e12  # bytes/s

    # 计算实际数据搬运量
    if dtype == "float16":
        elem_bytes = 2
    else:
        elem_bytes = 4

    # 读取 A, B，写入 C
    data_movement = (M * K + K * N + M * N) * elem_bytes

    # 计算实际带宽
    actual_bandwidth = data_movement / (avg_ms * 1e-3)

    # 计算利用率
    utilization = actual_bandwidth / theoretical_bandwidth

    return {
        "data_movement_gb": data_movement / 1e9,
        "actual_bandwidth_gbs": actual_bandwidth / 1e9,
        "theoretical_bandwidth_gbs": theoretical_bandwidth / 1e9,
        "utilization": utilization,
    }

# 分析 GEMM 的内存带宽
result = analyze_memory_bandwidth(4096, 4096, 4096, 4.0)
print(f"数据搬运量: {result['data_movement_gb']:.1f} GB")
print(f"实际带宽: {result['actual_bandwidth_gbs']:.1f} GB/s")
print(f"带宽利用率: {result['utilization']:.1%}")
```

以上代码展示了内存带宽利用率的分析方法。`analyze_memory_bandwidth()` 计算 GEMM 操作的实际数据搬运量（读取 A、B + 写入 C），除以 Kernel 执行时间得到实际带宽，再与 A100 的理论带宽 2TB/s 对比得出利用率。带宽利用率是衡量 GEMM 优化程度的重要指标——理想情况下应达到 70-80%。低利用率通常意味着 Shared Memory 使用不足或内存访问模式不够合并，需要重新审视 Tile Shape 和数据搬运策略。

---

## 10. Tile 编程模型的调试技巧

### 10.1 IR Dump

```python
# IR Dump: 查看编译器生成的 IR

import os
os.environ["TILELANG_DUMP_IR"] = "1"

@T.prim_func
def my_kernel(A, B, C):
    # ... kernel 实现 ...
    pass

# 编译时会输出:
# [TileLang] Tile IR:
# ...
# [TileLang] TensorIR:
# ...
# [TileLang] PTX:
# ...
```

以上代码展示了如何通过环境变量 `TILELANG_DUMP_IR` 开启编译器的中间表示输出。开启后编译过程会依次输出 Tile IR（TileLang 的中间表示）、TensorIR（张量化 IR）和 PTX（GPU 汇编）三个层次的代码。通过检查这些 IR，开发者可以了解编译器如何将高层次的 Tile 操作逐步降低为低层次的 GPU 指令，这是定位性能瓶颈和理解编译器行为的重要调试手段。

### 10.2 内存检查

```python
# 内存检查: 检查内存使用是否超出限制

@T.prim_func
def memory_check(A, B, C):
    # 检查 Shared Memory 使用
    A_shared = T.alloc_shared((128, 32), "float16")  # 8 KB
    B_shared = T.alloc_shared((32, 128), "float16")  # 8 KB
    # 总计: 16 KB

    # A100 限制: 164 KB/SM
    # 如果超出限制，编译器会报错

    # 检查 Fragment 使用
    C_frag = T.alloc_fragment((128, 128), "float32")  # 64 KB
    # 每线程 128×128×4 = 64 KB
    # 但 Fragment 是每线程私有的，所以需要乘以线程数
    # 实际上，Fragment 大小受寄存器数量限制
```

以上代码展示了运行时内存使用的检查方法。代码中 `T.alloc_shared()` 分配两块 Shared Memory（128×32×2 + 32×128×2 = 16KB），`T.alloc_fragment()` 为每个线程分配 64KB 的寄存器空间（128×128×4 字节）。开发者需要确保总 Shared Memory 不超过 A100 的 164KB/SM 限制，寄存器不超过 255 个/线程。编译器会在编译时自动检查这些限制并在超出时报错，帮助开发者在开发阶段就发现内存资源分配问题。

### 10.3 性能分析

```python
# 性能分析: 使用 ncu 分析 kernel 性能

# 运行 ncu 分析
# ncu --set full python my_kernel.py

# 关键指标:
# 1. SM Occupancy: 应该 > 50%
# 2. Memory Throughput: 应该接近理论值
# 3. Compute Throughput: 应该接近理论值
# 4. Bank Conflict: 应该 < 10%

# TileLang 内置性能分析
import os
os.environ["TILELANG_PROFILE"] = "1"

# 编译并运行
kernel = tilelang.compile(my_kernel, target="cuda")
kernel(A, B, C)

# 输出:
# [TileLang] Kernel: my_kernel
# [TileLang] Grid: (32, 32)
# [TileLang] Block: (256,)
# [TileLang] Shared Memory: 16 KB
# [TileLang] Registers: 128
# [TileLang] Occupancy: 75%
# [TileLang] Time: 4.2 ms
# [TileLang] TFLOPS: 325
```

以上代码展示了两种 Kernel 性能分析方法。第一种是使用 NVIDIA Nsight Compute（ncu）工具进行全面的 Kernel 分析，关注 SM Occupancy、Memory Throughput、Compute Throughput 和 Bank Conflict 四个关键指标。第二种是通过环境变量 `TILELANG_PROFILE` 启用 TileLang 内置的性能分析功能，编译运行后自动输出 Grid/Block 大小、Shared Memory 和寄存器使用量、Occupancy 和执行时间等关键信息。

---

## 11. Tile Shape 选择策略详解

### 11.1 硬件约束分析

不同 GPU 的硬件约束决定了 Tile Shape 的选择空间：

| GPU | Shared Memory | 寄存器/SM | 最大线程/SM | 推荐 Tile |
|-----|--------------|----------|-----------|----------|
| V100 | 96 KB | 65536 | 2048 | 128×128×32 |
| A100 | 164 KB | 65536 | 2048 | 128×128×32 或 256×128×32 |
| H100 | 228 KB | 65536 | 2048 | 256×256×64 |
| RTX 4090 | 100 KB | 65536 | 1536 | 128×128×32 |

### 11.2 Tile Shape 选择算法

```python
def optimal_tile_shape(M, N, K, dtype, gpu="A100"):
    """计算最优 Tile Shape"""

    # GPU 参数
    gpu_params = {
        "V100": {"smem_kb": 96, "regs": 65536, "max_threads": 2048},
        "A100": {"smem_kb": 164, "regs": 65536, "max_threads": 2048},
        "H100": {"smem_kb": 228, "regs": 65536, "max_threads": 2048},
    }
    params = gpu_params[gpu]

    # 数据类型字节数
    dtype_bytes = {"float16": 2, "bfloat16": 2, "float32": 4}[dtype]

    # 候选配置
    candidates = [
        (64, 64, 16),
        (64, 64, 32),
        (128, 64, 32),
        (128, 128, 32),
        (128, 128, 64),
        (256, 128, 32),
        (256, 128, 64),
        (256, 256, 64),
    ]

    valid_configs = []
    for BM, BN, BK in candidates:
        # 检查矩阵大小
        if BM > M or BN > N or BK > K:
            continue

        # 检查 Shared Memory
        smem = (BM * BK + BK * BN) * dtype_bytes
        if smem > params["smem_kb"] * 1024:
            continue

        # 检查寄存器 (简化估算)
        regs_per_thread = BM * BN * 4 // 32  # 假设 32 线程/Warp
        if regs_per_thread > 255:
            continue

        # 计算 Occupancy
        warps = (BM * BN) // (16 * 16)  # 假设 Warp 处理 16x16
        threads = warps * 32
        if threads > params["max_threads"]:
            continue

        # 计算并行度
        num_blocks = (M + BM - 1) // BM * (N + BN - 1) // BN

        valid_configs.append({
            "tile": (BM, BN, BK),
            "smem": smem,
            "threads": threads,
            "num_blocks": num_blocks,
            "data_reuse": (BM * BN * BK) / (BM * BK + BK * BN + BM * BN),
        })

    # 选择最优配置（最大化数据复用）
    if valid_configs:
        best = max(valid_configs, key=lambda x: x["data_reuse"])
        return best["tile"]

    return (64, 64, 16)  # 默认小 Tile
```

这段代码是 11.2 Tile Shape 选择算法 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 11.3 Tile Shape 对性能的影响

| Tile Shape | Shared Memory | 寄存器/线程 | Occupancy | 性能 (TFLOPS) |
|------------|--------------|-----------|-----------|---------------|
| 64×64×16 | 4 KB | 512 | 100% | 180 |
| 64×64×32 | 8 KB | 512 | 100% | 210 |
| 128×64×32 | 12 KB | 1024 | 75% | 250 |
| 128×128×32 | 16 KB | 2048 | 50% | 300 |
| 128×128×64 | 32 KB | 2048 | 50% | 290 |
| 256×128×32 | 24 KB | 4096 | 25% | 280 |
| 256×256×64 | 64 KB | 8192 | 12.5% | 220 |

> [!NOTE]
> 最优 Tile Shape 是数据复用率和 Occupancy 的平衡。太小的 Tile 数据复用不足，太大的 Tile 降低 Occupancy。

---

## 12. Thread Binding 进阶示例

### 12.1 二维 Thread Binding

```python
@T.prim_func
def thread_binding_2d(A, B, C):
    """二维 Thread Binding: 分别绑定 M 和 N 维度"""
    BLOCK_M = 128
    BLOCK_N = 128

    for bx, by in T.grid(M // BLOCK_M, N // BLOCK_N):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])

            # 二维 Thread Binding
            # M 维度: 4 个 Warp
            # N 维度: 8 个线程/Warp
            for wm in T.serial(4):      # Warp 在 M 维度
                for wn in T.serial(8):  # 线程在 N 维度
                    with T.block("thread"):
                        # 绑定到 threadIdx.x
                        T.thread_binding(wm * 8 + wn, "threadIdx.x")

                        # 每个线程处理的数据
                        local_m = wm * 32 + (wn // 4) * 16
                        local_n = (wn % 4) * 32

                        # 线程级计算
                        for i, j in T.grid(16, 32):
                            C[vbx * BLOCK_M + local_m + i,
                              vby * BLOCK_N + local_n + j] = ...
```

这段代码是 12.1 二维 Thread Binding 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 12.2 Warp 级 Thread Binding

```python
@T.prim_func
def warp_thread_binding(A, B, C):
    """Warp 级 Thread Binding: 每个 Warp 处理一个子块"""
    BLOCK_M = 128
    BLOCK_N = 128
    WARP_M = 32
    WARP_N = 32

    for bx, by in T.grid(M // BLOCK_M, N // BLOCK_N):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])

            # 8 个 Warp 处理 128×128 的块
            for warp_id in T.serial(8):
                with T.block("warp"):
                    # 绑定到 CUDA Warp
                    T.thread_binding(warp_id, "warp")

                    # 计算 Warp 负责的区域
                    wm = warp_id // 4  # 0-1
                    wn = warp_id % 4   # 0-3

                    # Warp 内的计算
                    C_frag = T.alloc_fragment((WARP_M, WARP_N), "float32")
                    T.fill(C_frag, T.float32(0))

                    for k in T.serial(K // 32):
                        # 加载 Warp 负责的数据
                        A_frag = T.alloc_fragment((WARP_M, 32), "float16")
                        B_frag = T.alloc_fragment((32, WARP_N), "float16")

                        T.copy(A_shared[wm*WARP_M:(wm+1)*WARP_M, :], A_frag)
                        T.copy(B_shared[:, wn*WARP_N:(wn+1)*WARP_N], B_frag)

                        T.gemm(A_frag, B_frag, C_frag)
```

这段代码是 12.2 Warp 级 Thread Binding 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 12.3 Thread Binding 性能对比

| Binding 方案 | 内存合并 | Bank Conflict | 性能 |
|-------------|---------|--------------|------|
| 行优先 (M→连续) | 好 | 可能 | 基准 |
| 列优先 (N→连续) | 好 | 可能 | 基准 ±5% |
| Z 形绑定 | 中等 | 少 | +3% |
| Hilbert 绑定 | 最好 | 最少 | +5% |

---

## 13. 多维 Tiling 详解

### 13.1 三维 Tiling

```python
@T.prim_func
def tiling_3d(A, B, C):
    """三维 Tiling: M, N, K 三个维度都分块"""
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    # 三维 Grid
    for bx, by, bk in T.grid(
        T.ceildiv(M, BLOCK_M),
        T.ceildiv(N, BLOCK_N),
        T.ceildiv(K, BLOCK_K),
    ):
        with T.block("C"):
            vbx, vby, vbk = T.axis.spatial("SSS", [bx, by, bk])

            # 每个 Block 处理一个 128×128×32 的子块
            C_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.clear(C_frag)

            # 加载数据
            A_smem = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
            B_smem = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")

            T.copy(A[vbx*BLOCK_M:(vbx+1)*BLOCK_M, vbk*BLOCK_K:(vbk+1)*BLOCK_K], A_smem)
            T.copy(B[vbk*BLOCK_K:(vbk+1)*BLOCK_K, vby*BLOCK_N:(vby+1)*BLOCK_N], B_smem)

            # 计算
            T.gemm(A_smem, B_smem, C_frag)

            # 累积到全局结果
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                T.atomic_add(
                    C[vbx*BLOCK_M + i, vby*BLOCK_N + j],
                    C_frag[i, j]
                )
```

这段代码是 13.1 三维 Tiling 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 13.2 层次化 Tiling

```python
@T.prim_func
def hierarchical_tiling(A, B, C):
    """层次化 Tiling: Block → Warp → Thread"""
    BLOCK_M = 256
    BLOCK_N = 256
    BLOCK_K = 64

    # 第一级: Block 级 Tiling
    for bx, by in T.grid(M // BLOCK_M, N // BLOCK_N):
        with T.block("block"):
            vbx, vby = T.axis.spatial("SS", [bx, by])

            # 第二级: Warp 级 Tiling
            WARP_M = 64
            WARP_N = 64
            for wx, wy in T.grid(BLOCK_M // WARP_M, BLOCK_N // WARP_N):
                with T.block("warp"):
                    # 第三级: Thread 级 Tiling
                    THREAD_M = 8
                    THREAD_N = 8

                    for tx, ty in T.grid(WARP_M // THREAD_M, WARP_N // THREAD_N):
                        with T.block("thread"):
                            # 每个线程处理 8×8 的子块
                            C_local = T.alloc_fragment((THREAD_M, THREAD_N), "float32")
                            T.clear(C_local)

                            for k in T.serial(K // BLOCK_K):
                                # 加载数据到 Shared Memory
                                A_smem = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
                                B_smem = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
                                T.copy(A[...], A_smem)
                                T.copy(B[...], B_smem)
                                T.syncthreads()

                                # 线程级计算
                                for kk in T.serial(BLOCK_K):
                                    for i, j in T.grid(THREAD_M, THREAD_N):
                                        C_local[i, j] += A_smem[...] * B_smem[...]

                            # 写回结果
                            for i, j in T.grid(THREAD_M, THREAD_N):
                                C[...] = C_local[i, j]
```

这段代码是 13.2 层次化 Tiling 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 13.3 Tiling 策略对比

| 策略 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| 二维 Tiling | 通用 GEMM | 简单 | K 维度无并行 |
| 三维 Tiling | 大 K 矩阵 | K 维度并行 | 需要原子操作 |
| 层次化 Tiling | 超大矩阵 | 最佳数据局部性 | 实现复杂 |
| 动态 Tiling | 变长输入 | 自适应 | 编译开销 |

---

## 14. Tile 数据类型详细对比

### 14.1 数据类型支持矩阵

| 数据类型 | T.Buffer | T.alloc_shared | T.alloc_fragment | 说明 |
|----------|----------|---------------|-----------------|------|
| float32 | ✅ | ✅ | ✅ | 默认累加类型 |
| float16 | ✅ | ✅ | ✅ | Tensor Core 输入 |
| bfloat16 | ✅ | ✅ | ✅ | 训练常用 |
| float8_e4m3 | ✅ | ✅ | ✅ | FP8 量化 |
| float8_e5m2 | ✅ | ✅ | ✅ | FP8 梯度 |
| int8 | ✅ | ✅ | ✅ | INT8 量化 |
| int4 | ✅ | ✅ | ✅ | INT4 量化 |
| bool | ✅ | ✅ | ❌ | Mask 类型 |

### 14.2 数据类型转换

```python
@T.prim_func
def dtype_conversion(A, B, C):
    """数据类型转换示例"""
    with T.Kernel(...) as (bx, by):
        # FP16 输入
        A_frag = T.alloc_fragment((128, 32), "float16")
        B_frag = T.alloc_fragment((32, 128), "float16")

        # FP32 累加器
        C_frag = T.alloc_fragment((128, 128), "float32")

        # 加载 FP16 数据
        T.copy(A[...], A_frag)
        T.copy(B[...], B_frag)

        # 计算 (自动转换为 FP32)
        T.gemm(A_frag, B_frag, C_frag)  # FP16 × FP16 → FP32

        # 转换回 FP16 写出
        for i, j in T.Parallel(128, 128):
            C[i, j] = T.cast(C_frag[i, j], "float16")
```

这段代码是 14.2 数据类型转换 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 14.3 混合精度策略

| 策略 | 输入 | 累加 | 输出 | 精度 | 性能 |
|------|------|------|------|------|------|
| 全 FP32 | FP32 | FP32 | FP32 | 最高 | 最低 |
| FP16 累加 | FP16 | FP32 | FP16 | 高 | 高 |
| BF16 累加 | BF16 | FP32 | BF16 | 高 | 高 |
| FP8 累加 | FP8 | FP32 | FP16 | 中 | 最高 |
| INT8 累加 | INT8 | INT32 | FP16 | 中 | 高 |

---

## 15. Grid/Block/Thread 映射图解

### 15.1 一维映射

```
一维 Grid, 一维 Block:

Grid (4 blocks):
┌──────┬──────┬──────┬──────┐
│Block0│Block1│Block2│Block3│
│      │      │      │      │
│T0-T31│T0-T31│T0-T31│T0-T31│
└──────┴──────┴──────┴──────┘

映射关系:
- blockIdx.x → 0, 1, 2, 3
- threadIdx.x → 0-31 (每个 Block)

代码:
for bx in T.grid(4):
    with T.block("B"):
        vbx = T.axis.spatial("S", bx)
        for tx in T.serial(32):
            with T.block("T"):
                T.thread_binding(tx, "threadIdx.x")
```

这个代码块或示意图用于说明 15.1 一维映射 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 15.2 二维映射

```
二维 Grid (2×2), 二维 Block (4×8):

Grid:
┌─────────────┬─────────────┐
│  Block(0,0) │  Block(0,1) │
│  ┌────┬────┐│  ┌────┬────┐│
│  │W0  │W1  ││  │W0  │W1  ││
│  │T0-7│T0-7││  │T0-7│T0-7││
│  ├────┼────┤│  ├────┼────┤│
│  │W2  │W3  ││  │W2  │W3  ││
│  │T0-7│T0-7││  │T0-7│T0-7││
│  └────┴────┘│  └────┴────┘│
├─────────────┼─────────────┤
│  Block(1,0) │  Block(1,1) │
│     ...     │     ...     │
└─────────────┴─────────────┘

映射关系:
- blockIdx.x → M 维度 (0-1)
- blockIdx.y → N 维度 (0-1)
- threadIdx.x → Block 内线程 (0-31)
```

这个代码块或示意图用于说明 15.2 二维映射 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 15.3 三维映射

```
三维 Grid (2×2×4):

Grid:
K 维度 (4 个切片)
┌────┬────┬────┬────┐
│K=0 │K=1 │K=2 │K=3 │
├────┼────┼────┼────┤
│B00 │B00 │B00 │B00 │  ← M=0, N=0
│B01 │B01 │B01 │B01 │  ← M=0, N=1
├────┼────┼────┼────┤
│B10 │B10 │B10 │B10 │  ← M=1, N=0
│B11 │B11 │B11 │B11 │  ← M=1, N=1
└────┴────┴────┴────┘

映射关系:
- blockIdx.x → M 维度
- blockIdx.y → N 维度
- blockIdx.z → K 维度 (需要原子操作累积)
```

这个代码块或示意图用于说明 15.3 三维映射 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

---

## 16. 性能影响因素分析

### 16.1 Tile 大小与 Occupancy

| Tile 大小 | Shared Memory | 寄存器 | Blocks/SM | Occupancy |
|-----------|--------------|--------|-----------|-----------|
| 64×64 | 8 KB | 512 | 16 | 100% |
| 128×64 | 12 KB | 1024 | 8 | 100% |
| 128×128 | 16 KB | 2048 | 4 | 100% |
| 256×128 | 24 KB | 4096 | 2 | 50% |
| 256×256 | 32 KB | 8192 | 1 | 25% |

### 16.2 Tile 大小与数据复用

| Tile 大小 | 数据复用率 | 说明 |
|-----------|----------|------|
| 小 (64×64) | 低 | 每个元素只用一次 |
| 中 (128×128) | 中 | 适中的复用 |
| 大 (256×256) | 高 | 最大化复用 |

### 16.3 最优 Tile 选择经验

```
经验法则:

1. 对于 compute-bound 算子 (GEMM):
   - 使用大 Tile (128×128 或更大)
   - 最大化数据复用
   - 接受较低的 Occupancy

2. 对于 memory-bound 算子 (Softmax):
   - 使用小 Tile (64×64 或更小)
   - 最大化 Occupancy
   - 减少内存占用

3. 对于混合算子 (FlashAttention):
   - 使用中等 Tile (128×64)
   - 平衡复用和 Occupancy

4. 通用建议:
   - 从 128×128×32 开始
   - 根据 Profiling 结果调整
   - 尝试 Auto-tuning
```

这个代码块或示意图用于说明 16.3 最优 Tile 选择经验 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 16.4 Tile Shape 自动搜索

```python
def auto_tune_tile_shape(M, N, K, dtype="float16", gpu="A100"):
    """自动搜索最优 Tile Shape"""

    # 搜索空间
    search_space = {
        "BLOCK_M": [64, 128, 256],
        "BLOCK_N": [64, 128, 256],
        "BLOCK_K": [16, 32, 64],
    }

    best_config = None
    best_perf = 0

    import itertools
    for BM, BN, BK in itertools.product(
        search_space["BLOCK_M"],
        search_space["BLOCK_N"],
        search_space["BLOCK_K"],
    ):
        # 检查约束
        if BM > M or BN > N or BK > K:
            continue

        # 编译并测试
        try:
            @T.prim_func
            def gemm(A, B, C):
                for bx, by in T.grid(M // BM, N // BN):
                    with T.block("C"):
                        vbx, vby = T.axis.spatial("SS", [bx, by])
                        C_frag = T.alloc_fragment((BM, BN), "float32")
                        T.clear(C_frag)
                        for k in T.serial(K // BK):
                            A_smem = T.alloc_shared((BM, BK), dtype)
                            B_smem = T.alloc_shared((BK, BN), dtype)
                            T.copy(A[...], A_smem)
                            T.copy(B[...], B_smem)
                            T.gemm(A_smem, B_smem, C_frag)
                        T.copy(C_frag, C[...])

            kernel = tilelang.compile(gemm, target="cuda")
            perf = benchmark_kernel(kernel, M, N, K)

            if perf > best_perf:
                best_perf = perf
                best_config = (BM, BN, BK)
        except:
            continue

    return best_config, best_perf
```

这段代码是 16.4 Tile Shape 自动搜索 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 📖 下一章预告

**Chapter 4: 显式内存层级管理**

在下一章中，我们将：
- 深入学习 T.alloc_shared 的使用与对齐
- 理解 T.alloc_L1 的工作机制
- 掌握 T.alloc_fragment 的高级用法
- 学习多级内存数据搬运策略
- 理解内存生命周期管理

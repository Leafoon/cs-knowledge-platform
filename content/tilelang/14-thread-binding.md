---
title: "Chapter 14: Thread Binding 与硬件线程映射"
description: "深入理解 Thread Binding 概念、CUDA/AMD/昇腾线程映射、性能影响及最优策略，包含源码走读"
updated: 2026-06-11
---

# Chapter 14: Thread Binding 与硬件线程映射

> **Learning Objectives**
>
> 1. 理解 Thread Binding 的核心概念：Tile 维度到硬件线程的映射
> 2. 掌握 `T.thread_binding` 注解的使用方法
> 3. 理解 CUDA Thread Block/Warp/Thread 的层次映射
> 4. 学习 AMD Wavefront/Thread 映射机制
> 5. 了解华为昇腾 AI Core 的线程映射方式
> 6. 分析 Thread Binding 对性能的影响
> 7. 掌握最优 Thread Binding 策略的选择方法
> 8. 阅读 Thread Binding Pass 的源码实现

---

## 14.1 Thread Binding 概念

### 14.1.1 什么是 Thread Binding

<div data-component="ThreadBindingDiagram"></div>

Thread Binding 是将程序中的逻辑循环维度映射到物理硬件线程的过程。在 GPU 编程中，这意味着将 tile 的行/列维度映射到 Thread Block、Warp、Thread 等硬件层次。

```
Thread Binding 的本质：

逻辑视图 (TileLang):
┌─────────────────────────┐
│  for i in range(128):   │  ← Tile 行维度
│    for j in range(64):  │  ← Tile 列维度
│      compute(i, j)      │
└─────────────────────────┘
          │
          │ Thread Binding
          ▼
物理视图 (GPU):
┌─────────────────────────┐
│  Block (bx, by)         │  ← 映射到 grid
│    Warp (warp_id)       │  ← 映射到 warp
│      Thread (tx, ty)    │  ← 映射到 thread
└─────────────────────────┘

目标：让逻辑循环高效地映射到硬件执行
```

上述图示展示了 Thread Binding 的核心概念：将 TileLang 中的逻辑循环维度（行维度 i 和列维度 j）映射到 GPU 的物理硬件层次（Grid → Block → Warp → Thread）。这种映射直接决定了数据访问模式、线程利用率和整体性能。TileLang 的 `T.thread_binding` 注解就是实现这一映射的关键机制，它允许开发者精确控制循环维度与硬件线程的对应关系。

### 14.1.2 为什么 Thread Binding 重要

Thread Binding 直接影响以下性能因素：

| 因素 | 说明 | 影响 |
|------|------|------|
| 内存合并 | 相邻线程访问相邻地址 | 高 |
| Warp 利用率 | Warp 内线程活跃比例 | 高 |
| Bank Conflict | 共享内存访问冲突 | 中 |
| 寄存器压力 | 每个线程的寄存器使用 | 中 |
| 占用率 | 活跃 Warp 数量 | 中 |

### 14.1.3 Thread Binding 的层次模型

```
GPU 硬件层次：

┌─────────────────────────────────────────────────────────┐
│                      GPU                                │
│  ┌───────────────────────────────────────────────────┐  │
│  │                 Grid                              │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────┐  │  │
│  │  │  Block (0,0)│  │  Block (1,0)│  │  ...     │  │  │
│  │  │  ┌───────┐  │  │  ┌───────┐  │  │          │  │  │
│  │  │  │Warp 0 │  │  │  │Warp 0 │  │  │          │  │  │
│  │  │  │Thread │  │  │  │Thread │  │  │          │  │  │
│  │  │  │0-31   │  │  │  │0-31   │  │  │          │  │  │
│  │  │  ├───────┤  │  │  ├───────┤  │  │          │  │  │
│  │  │  │Warp 1 │  │  │  │Warp 1 │  │  │          │  │  │
│  │  │  │Thread │  │  │  │Thread │  │  │          │  │  │
│  │  │  │32-63  │  │  │  │32-63  │  │  │          │  │  │
│  │  │  ├───────┤  │  │  ├───────┤  │  │          │  │  │
│  │  │  │ ...   │  │  │  │ ...   │  │  │          │  │  │
│  │  │  └───────┘  │  │  └───────┘  │  │          │  │  │
│  │  └─────────────┘  └─────────────┘  └──────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘

对应到 TileLang:
  Grid   → T.Kernel(grid_m, grid_n)
  Block  → Warp 级别循环
  Thread → Thread 级别循环
```

这个代码块或示意图用于说明 14.1.3 Thread Binding 的层次模型 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

---

## 14.2 T.thread_binding 注解详解

### 14.2.1 基本语法

```python
@T.prim_func
def kernel_with_binding(
    A: T.Buffer[(M, N), "float16"],
    B: T.Buffer[(M, N), "float16"],
    C: T.Buffer[(M, N), "float16"],
):
    with T.Kernel(grid_m, grid_n, threads=256) as (bx, by):
        # 方式 1: 隐式绑定（自动）
        # TileLang 自动将循环绑定到线程
        for i in T.serial(TILE_M):
            for j in T.serial(TILE_N):
                C[bx * TILE_M + i, by * TILE_N + j] = A[bx * TILE_M + i, by * TILE_N + j] + B[bx * TILE_M + i, by * TILE_N + j]

        # 方式 2: 显式绑定
        # 使用 T.thread_binding 显式指定绑定
        for i in T.serial(TILE_M):
            for j in T.thread_binding(TILE_N, thread="threadIdx.x"):
                C[bx * TILE_M + i, by * TILE_N + j] = A[bx * TILE_M + i, by * TILE_N + j] + B[bx * TILE_M + i, by * TILE_N + j]
```

上述代码展示了 Thread Binding 的两种模式：隐式绑定由编译器自动将循环维度映射到最近可用的硬件线程，适合快速原型验证；显式绑定则通过 `T.thread_binding(thread="threadIdx.x")` 精确控制 `j` 维度映射到 `threadIdx.x`。显式绑定的核心优势在于程序员可以主动设计内存访问模式——当内层循环绑定到 `threadIdx.x` 时，相邻线程访问相邻列地址，实现完美的合并访问。但需注意，显式绑定的循环范围必须与 `T.Kernel(threads=...)` 参数匹配，否则编译器会报错或生成错误的线程索引。

### 14.2.2 绑定类型

```python
# Thread Binding 类型

# 1. Grid 绑定 (Block 级别)
for bx in T.thread_binding(grid_m, thread="blockIdx.x"):
    for by in T.thread_binding(grid_n, thread="blockIdx.y"):
        pass

# 2. Thread 绑定
for tx in T.thread_binding(256, thread="threadIdx.x"):
    pass

# 3. 多维 Thread 绑定
for tx in T.thread_binding(32, thread="threadIdx.x"):
    for ty in T.thread_binding(8, thread="threadIdx.y"):
        pass  # 32 * 8 = 256 threads

# 4. Warp 绑定
for warp_id in T.thread_binding(8, thread="warp"):
    pass  # 8 warps = 256 threads
```

这段代码是 14.2.2 绑定类型 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 14.2.3 绑定策略示例

```python
# 策略 1: 行优先绑定
# 将行维度绑定到 threadIdx.x
"""
for i in T.serial(TILE_M):
    for j in T.thread_binding(TILE_N, thread="threadIdx.x"):
        # 相邻线程访问相邻列 → 合并访问
        C[i, j] = A[i, j] + B[i, j]
"""

# 策略 2: 列优先绑定
# 将列维度绑定到 threadIdx.x
"""
for j in T.serial(TILE_N):
    for i in T.thread_binding(TILE_M, thread="threadIdx.x"):
        # 相邻线程访问相邻行 → 可能不合并
        C[i, j] = A[i, j] + B[i, j]
"""

# 策略 3: 分块绑定
# 将线程分组，每组处理一个子块
"""
for i_outer in T.serial(TILE_M // 32):
    for j_outer in T.serial(TILE_N // 32):
        for i_inner in T.thread_binding(32, thread="threadIdx.y"):
            for j_inner in T.thread_binding(32, thread="threadIdx.x"):
                i = i_outer * 32 + i_inner
                j = j_outer * 32 + j_inner
                C[i, j] = A[i, j] + B[i, j]
"""
```

这段代码是 14.2.3 绑定策略示例 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 14.3 CUDA Thread Block/Warp/Thread 映射

<div data-component="CUDAThreadMapping"></div>

### 14.3.1 CUDA 线程层次

```
CUDA 线程层次：

Grid
├── Block (0, 0, 0)
│   ├── Warp 0: Thread 0-31
│   ├── Warp 1: Thread 32-63
│   ├── Warp 2: Thread 64-95
│   └── Warp 3: Thread 96-127
├── Block (1, 0, 0)
│   ├── Warp 0: Thread 0-31
│   └── ...
└── ...

硬件约束：
  - Block 最大线程数: 1024
  - Warp 大小: 32
  - Block 维度: (x, y, z)
  - Grid 维度: (x, y, z)
```

这个代码块或示意图用于说明 14.3.1 CUDA 线程层次 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 14.3.2 CUDA Thread 索引计算

```python
# CUDA 线程索引

# 1D Thread Block
thread_id = threadIdx.x

# 2D Thread Block
thread_id = threadIdx.y * blockDim.x + threadIdx.x

# 3D Thread Block
thread_id = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x

# Warp ID
warp_id = thread_id // 32

# Lane ID (warp 内线程 ID)
lane_id = thread_id % 32
```

这段代码是 14.3.2 CUDA Thread 索引计算 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 14.3.3 TileLang 到 CUDA 的映射

```python
# TileLang Thread Binding 到 CUDA 的映射

"""
TileLang:
  with T.Kernel(grid_m, grid_n, threads=256) as (bx, by):
    for i in T.serial(128):
      for j in T.thread_binding(64, thread="threadIdx.x"):
        ...

CUDA:
  __global__ void kernel() {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;  // 0..63

    for (int i = 0; i < 128; i++) {
      // j = tx，直接映射
      int j = tx;
      ...
    }
  }
"""

"""
TileLang:
  with T.Kernel(grid_m, grid_n, threads=256) as (bx, by):
    for i in T.thread_binding(128, thread="threadIdx.y"):
      for j in T.thread_binding(64, thread="threadIdx.x"):
        ...

CUDA:
  __global__ void kernel() {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int ty = threadIdx.y;  // 0..127 (注意: 需要 blockDim.y = 128)
    int tx = threadIdx.x;  // 0..63 (注意: 需要 blockDim.x = 64)

    // i = ty, j = tx
    int i = ty;
    int j = tx;
    ...
  }
"""
```

这段代码是 14.3.3 TileLang 到 CUDA 的映射 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 14.3.4 Warp 级别操作

```python
# Warp 级别的 Thread Binding

@T.prim_func
def warp_level_kernel(A, B, C):
    with T.Kernel(grid_m, grid_n, threads=256) as (bx, by):
        # 获取 warp 和 lane 信息
        tx = T.get_thread_id()  # 0..255
        warp_id = tx // 32
        lane_id = tx % 32

        # Warp 级别的绑定
        for i in T.serial(TILE_M // 8):  # 8 个 warp
            for j in T.serial(TILE_N):
                # 每个 warp 处理 TILE_M/8 行
                row = warp_id * (TILE_M // 8) + i
                col = j

                # Warp 内使用 shuffle 进行通信
                val = A[row, col]
                # Warp 内归约
                for offset in [16, 8, 4, 2, 1]:
                    val += T.warp_shuffle_down(val, offset)

                if lane_id == 0:
                    C[row // 32, col] = val
```

这段代码是 14.3.4 Warp 级别操作 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 14.4 AMD Wavefront/Thread 映射

<div data-component="AMDWavefrontMapping"></div>

### 14.4.1 AMD GPU 线程层次

```
AMD GPU (RDNA / CDNA) 线程层次：

Workgroup (对应 CUDA Block)
├── Wavefront 0: Thread 0-63    ← 注意: 64 线程，不是 32
├── Wavefront 1: Thread 64-127
├── Wavefront 2: Thread 128-191
└── Wavefront 3: Thread 192-255

硬件约束：
  - Wavefront 大小: 64 (CDNA) 或 32 (RDNA)
  - Workgroup 最大线程数: 1024
  - VGPR (向量寄存器): 每 CU 有大量
  - SGPR (标量寄存器): 每 CU 有少量
```

这个代码块或示意图用于说明 14.4.1 AMD GPU 线程层次 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 14.4.2 AMD vs CUDA 差异

```
AMD vs CUDA 关键差异：

┌────────────────────┬─────────────┬──────────────┐
│  特性              │  CUDA       │  AMD (ROCm)  │
├────────────────────┼─────────────┼──────────────┤
│  Warp 大小         │  32         │  64 (CDNA)   │
│  Block 名称        │  Block      │  Workgroup   │
│  Warp 名称         │  Warp       │  Wavefront   │
│  共享内存          │  Shared Mem │  LDS (Local  │
│                    │             │  Data Share) │
│  线程 ID           │  threadIdx  │  __lane_id   │
│  Warp Shuffle      │  __shfl     │  __shfl      │
│  Warp Vote         │  __any/all  │  __any/all   │
└────────────────────┴─────────────┴──────────────┘
```

这个代码块或示意图用于说明 14.4.2 AMD vs CUDA 差异 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 14.4.3 TileLang 的 AMD 适配

```python
# TileLang 自动处理 AMD 和 CUDA 的差异

@T.prim_func
def portable_kernel(A, B, C):
    """
    同一份 TileLang 代码可以编译到 NVIDIA 和 AMD GPU
    TileLang 编译器自动处理底层差异
    """
    with T.Kernel(grid_m, grid_n, threads=256) as (bx, by):
        # 这段代码在 NVIDIA GPU 上：
        #   - 每个 warp 32 线程
        #   - 8 个 warp
        #
        # 在 AMD GPU 上：
        #   - 每个 wavefront 64 线程
        #   - 4 个 wavefront

        tx = T.get_thread_id()  # 0..255

        for i in T.serial(TILE_M):
            for j in T.serial(TILE_N // 256):
                col = tx * (TILE_N // 256) + j
                C[bx * TILE_M + i, by * TILE_N + col] = \
                    A[bx * TILE_M + i, by * TILE_N + col] + \
                    B[bx * TILE_M + i, by * TILE_N + col]

# 编译到不同目标
nvidia_compiled = tilelang.compile(portable_kernel, target="cuda")
amd_compiled = tilelang.compile(portable_kernel, target="rocm")
```

这段代码是 14.4.3 TileLang 的 AMD 适配 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 14.4.4 Wavefront 级别操作

```python
# Wavefront 级别操作 (AMD 特有)

@T.prim_func
def wavefront_reduce(A, Output):
    """
    Wavefront 级别归约
    AMD 的 wavefront 有 64 个线程
    """
    with T.Kernel(grid_m, threads=256) as (bx,):
        tx = T.get_thread_id()
        wave_id = tx // 64    # 在 AMD 上是 64
        lane_id = tx % 64

        # 每个 wavefront 归约 64 个元素
        val = A[bx * 256 + tx]

        # Wavefront 内归约
        for offset in [32, 16, 8, 4, 2, 1]:
            val += T.warp_shuffle_down(val, offset)

        # 第一个线程写结果
        if lane_id == 0:
            Output[bx * 4 + wave_id] = val
```

这段代码是 14.4.4 Wavefront 级别操作 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 14.5 昇腾 AI Core 映射

<div data-component="AscendAICoreMapping"></div>

### 14.5.1 昇腾 NPU 架构

```
昇腾 910B AI Core 架构：

AI Core
├── Vector Unit (向量计算单元)
│   ├── 3 个 Cube Unit (矩阵计算)
│   └── 2 个 Vector Unit (向量计算)
├── Memory Hierarchy
│   ├── L1 Buffer (1MB)
│   ├── Unified Buffer (UB, 256KB)
│   └── L0A/L0B/L0C Buffer
├── Data Agent (DA)
│   ├── MTE (Memory Transfer Engine)
│   └── Scalar Unit
└── Control Unit
    ├── Instruction Buffer
    └── Program Counter

关键区别：
  - 不是传统的 SIMT 模型
  - 使用 AI Core 而非 CUDA Core
  - 数据搬运由 DA (Data Agent) 控制
  - 矩阵计算由 Cube Unit 完成
```

这个代码块或示意图用于说明 14.5.1 昇腾 NPU 架构 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 14.5.2 昇腾线程模型

```python
# 昇腾 NPU 的线程模型

"""
昇腾使用 AI Core 而非传统的 Thread Block

TileLang 映射:
  T.Kernel(grid_m, grid_n) → AI Core Grid
  每个 AI Core → 一个独立的计算单元

与 CUDA 的区别：
  CUDA: 大量轻量级线程 (数千个)
  昇腾: 少量重量级 AI Core (数百个)

内存层次：
  GM (Global Memory) → 对应 CUDA Global Memory
  L1 Buffer → 对应 CUDA L2 Cache
  UB (Unified Buffer) → 对应 CUDA Shared Memory
  L0A/L0B/L0C → 对应 CUDA Register File
"""

@T.prim_func
def ascend_kernel(A, B, C):
    """
    昇腾 NPU kernel
    """
    with T.Kernel(grid_m, grid_n, target="ascend") as (bx, by):
        # 数据搬运: GM → UB
        A_ub = T.alloc_buffer([TILE_M, TILE_K], scope="UB")
        B_ub = T.alloc_buffer([TILE_K, TILE_N], scope="UB")

        # 使用 Data Agent 搬运数据
        T.copy(A[bx * TILE_M:..., by * TILE_K:...], A_ub)
        T.copy(B[by * TILE_K:..., by * TILE_N:...], B_ub)

        # 矩阵计算: 使用 Cube Unit
        C_ub = T.alloc_buffer([TILE_M, TILE_N], scope="UB")
        T.gemm(A_ub, B_ub, C_ub, unit="cube")

        # 数据搬运: UB → GM
        T.copy(C_ub, C[bx * TILE_M:..., by * TILE_N:...])
```

这段代码是 14.5.2 昇腾线程模型 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 14.5.3 昇腾特有优化

```python
# 昇腾 NPU 特有的优化策略

class AscendOptimizer:
    """
    昇腾 NPU 优化器
    """

    def optimize(self, kernel):
        # 1. 数据对齐优化
        # 昇腾要求 32 字节对齐
        kernel = self._align_data(kernel, alignment=32)

        # 2. 数据搬运优化
        # 使用 MTE (Memory Transfer Engine) 流水线
        kernel = self._pipeline_data_movement(kernel)

        # 3. Cube Unit 利用率优化
        # 确保矩阵维度是 16 的倍数
        kernel = self._pad_for_cube(kernel, granularity=16)

        # 4. L1 Buffer 优化
        # 合理分配 L1 Buffer 空间
        kernel = self._optimize_l1_usage(kernel)

        return kernel

    def _pipeline_data_movement(self, kernel):
        """
        数据搬运流水线化

        在计算当前块时，预取下一个块的数据
        """
        # GM → L1 → UB 流水线
        # Stage 1: 预取到 L1
        # Stage 2: 从 L1 搬运到 UB
        # Stage 3: 计算
        pass
```

这段代码是 14.5.3 昇腾特有优化 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 14.5b NVIDIA vs AMD vs Ascend 映射对比

### 14.5b.1 三种硬件的线程模型详细对比

```
三种硬件线程模型对比：

┌─────────────────┬──────────────┬──────────────┬──────────────┐
│  特性           │  NVIDIA CUDA │  AMD ROCm    │  华为 Ascend │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│  最小执行单位   │  Thread      │  Thread      │  AI Core     │
│  线程组         │  Warp (32)   │  Wavefront   │  N/A         │
│                 │              │  (64/32)     │              │
│  线程块         │  Block       │  Workgroup   │  AI Core Grid│
│  共享内存       │  Shared Mem  │  LDS         │  UB/L1       │
│  矩阵单元       │  Tensor Core │  MFMA/WMMA   │  Cube Unit   │
│  向量单元       │  CUDA Core   │  VALU        │  Vector Unit │
│  最大线程/块    │  1024        │  1024        │  N/A         │
│  最大块/SM      │  32          │  40          │  1 (per Core)│
│  Warp 调度      │  GTO/RR      │  RR          │  硬件调度    │
└─────────────────┴──────────────┴──────────────┴──────────────┘
```

这个代码块或示意图用于说明 14.5b.1 三种硬件的线程模型详细对比 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 14.5b.2 Warp 调度策略详解

```
GPU Warp 调度策略：

1. GTO (Greedy Then Oldest)
   - 优先执行同一个 Warp 直到阻塞
   - 然后切换到最老的就绪 Warp
   - 优点：缓存局部性好
   - 缺点：延迟隐藏差

2. RR (Round Robin)
   - 轮流执行所有就绪 Warp
   - 优点：延迟隐藏好
   - 缺点：缓存局部性差

3. 两步调度 (Two-Level Scheduler)
   - 组内 GTO，组间 RR
   - 平衡缓存局部性和延迟隐藏

NVIDIA Ampere+: 两步调度
AMD CDNA: Round Robin
Ascend: 硬件自动调度（不暴露给程序员）

对 Thread Binding 的影响:
  - GTO: 倾向于让同一个 Warp 连续执行，适合 Warp Tiled 策略
  - RR: 倾向于公平调度，适合行优先策略
  - 两步: 需要在 Warp 间平衡工作量
```

这个代码块或示意图用于说明 14.5b.2 Warp 调度策略详解 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 14.5b.3 Occupancy 计算器

```python
def calculate_gpu_occupancy(
    gpu_type: str,
    threads_per_block: int,
    regs_per_thread: int,
    smem_per_block: int,
    blocks_per_sm_limit: int = None,
):
    """
    计算 GPU Occupancy

    支持 NVIDIA、AMD、华为昇腾三种硬件
    """
    specs = {
        "A100": {
            "max_regs_per_sm": 65536,
            "max_smem_per_sm": 164 * 1024,
            "max_warps_per_sm": 64,
            "warp_size": 32,
            "max_blocks_per_sm": 32,
        },
        "H100": {
            "max_regs_per_sm": 65536,
            "max_smem_per_sm": 228 * 1024,
            "max_warps_per_sm": 64,
            "warp_size": 32,
            "max_blocks_per_sm": 32,
        },
        "MI300X": {
            "max_regs_per_sm": 65536,
            "max_smem_per_sm": 64 * 1024,
            "max_warps_per_sm": 40,
            "warp_size": 64,
            "max_blocks_per_sm": 40,
        },
        "Ascend910B": {
            "max_regs_per_sm": None,  # 不适用
            "max_smem_per_sm": 256 * 1024,  # UB
            "max_warps_per_sm": None,  # 不适用
            "warp_size": None,
            "max_blocks_per_sm": 1,  # 每个 AI Core 一个任务
        },
    }

    if gpu_type not in specs:
        raise ValueError(f"Unsupported GPU: {gpu_type}")

    spec = specs[gpu_type]

    if gpu_type == "Ascend910B":
        # 昇腾使用不同的计算模型
        return {
            "occupancy": 1.0,  # 每个 AI Core 独立运行
            "active_tasks": 1,
            "bottleneck": "compute",
        }

    # NVIDIA 和 AMD 的计算
    warps_per_block = (threads_per_block + spec["warp_size"] - 1) // spec["warp_size"]
    regs_per_block = regs_per_thread * threads_per_block

    blocks_by_regs = spec["max_regs_per_sm"] // max(regs_per_block, 1)
    blocks_by_smem = spec["max_smem_per_sm"] // max(smem_per_block, 1) if smem_per_block > 0 else 1000
    blocks_by_warps = spec["max_warps_per_sm"] // warps_per_block
    blocks_by_limit = spec["max_blocks_per_sm"]
    if blocks_per_sm_limit:
        blocks_by_limit = min(blocks_by_limit, blocks_per_sm_limit)

    active_blocks = min(blocks_by_regs, blocks_by_smem, blocks_by_warps, blocks_by_limit)
    active_warps = active_blocks * warps_per_block
    occupancy = active_warps / spec["max_warps_per_sm"]

    bottleneck = "regs" if blocks_by_regs == active_blocks else \
                 "smem" if blocks_by_smem == active_blocks else \
                 "warps" if blocks_by_warps == active_blocks else "limit"

    return {
        "active_blocks": active_blocks,
        "active_warps": active_warps,
        "max_warps": spec["max_warps_per_sm"],
        "occupancy": occupancy,
        "bottleneck": bottleneck,
        "breakdown": {
            "by_regs": blocks_by_regs,
            "by_smem": blocks_by_smem,
            "by_warps": blocks_by_warps,
            "by_limit": blocks_by_limit,
        },
    }

# 使用示例
for gpu in ["A100", "H100", "MI300X"]:
    occ = calculate_gpu_occupancy(
        gpu_type=gpu,
        threads_per_block=256,
        regs_per_thread=128,
        smem_per_block=48 * 1024,
    )
    print(f"{gpu}: Occupancy={occ['occupancy']:.1%}, Bottleneck={occ['bottleneck']}")
```

这段代码是 14.5b.3 Occupancy 计算器 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 14.5b.4 跨平台 Thread Binding 策略

```python
@T.prim_func
def cross_platform_binding(A, B, C):
    """
    跨平台 Thread Binding 策略

    设计原则：
    1. 使用 1D Thread Binding（最通用）
    2. 线程数为 256（所有平台都支持）
    3. Warp 大小无关的算法
    """
    with T.Kernel(grid_m, grid_n, threads=256) as (bx, by):
        tx = T.get_thread_id()  # 0..255

        # 每个线程处理 TILE_N/256 个元素
        elements_per_thread = TILE_N // 256

        for i in T.serial(TILE_M):
            for j in T.serial(elements_per_thread):
                col = tx * elements_per_thread + j
                C[bx * TILE_M + i, by * TILE_N + col] = \
                    A[bx * TILE_M + i, by * TILE_N + col] + \
                    B[bx * TILE_M + i, by * TILE_N + col]

# 编译到不同平台
# NVIDIA: 每个 Warp 32 线程，8 个 Warp
# AMD: 每个 Wavefront 64 线程，4 个 Wavefront
# 昇腾: 1 个 AI Core，内部并行
```

这段代码是 14.5b.4 跨平台 Thread Binding 策略 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 14.5c 性能影响详细分析

### 14.5c.1 内存合并访问的量化分析

```python
def analyze_coalescing_efficiency(access_pattern, warp_size=32):
    """
    量化分析内存合并访问效率

    参数:
        access_pattern: 列表，每个元素是一个 warp 内线程的访问地址
        warp_size: warp 大小

    返回:
        efficiency: 合并效率 (0-1)
        transactions: 需要的内存事务数
    """
    # 假设 128 字节的内存事务
    transaction_size = 128  # bytes
    element_size = 2  # float16 = 2 bytes

    # 计算每个线程访问的地址
    addresses = [access_pattern(tid) for tid in range(warp_size)]

    # 计算访问的 cache line
    cache_lines = set()
    for addr in addresses:
        cache_line = (addr * element_size) // transaction_size
        cache_lines.add(cache_line)

    # 理想情况：所有线程访问同一个 cache line
    ideal_transactions = 1
    actual_transactions = len(cache_lines)

    efficiency = ideal_transactions / actual_transactions

    return {
        "efficiency": efficiency,
        "transactions": actual_transactions,
        "ideal_transactions": ideal_transactions,
        "cache_lines_accessed": sorted(cache_lines),
    }

# 示例 1: 完美合并
def perfect_coalescing(tid):
    return tid  # 线程 0 访问地址 0，线程 1 访问地址 1，...

result = analyze_coalescing_efficiency(perfect_coalescing)
print(f"完美合并: 效率={result['efficiency']:.1%}, 事务数={result['transactions']}")

# 示例 2: 跨步访问
def strided_access(tid, stride=32):
    return tid * stride  # 线程 0 访问地址 0，线程 1 访问地址 32，...

result = analyze_coalescing_efficiency(lambda tid: strided_access(tid, 32))
print(f"跨步访问: 效率={result['efficiency']:.1%}, 事务数={result['transactions']}")
```

这段代码是 14.5c.1 内存合并访问的量化分析 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 14.5c.2 Bank Conflict 的性能影响

```python
def analyze_bank_conflict_impact(access_pattern, num_banks=32, bank_width=4):
    """
    分析 Bank Conflict 对性能的影响

    参数:
        access_pattern: 函数，输入线程 ID，输出 Shared Memory 地址
        num_banks: bank 数量
        bank_width: 每个 bank 的宽度（字节）

    返回:
        conflict_degree: 冲突程度（最大冲突数）
        serialization_factor: 序列化因子
    """
    warp_size = 32

    # 计算每个线程访问的 bank
    banks = []
    for tid in range(warp_size):
        addr = access_pattern(tid)
        bank_id = (addr // bank_width) % num_banks
        banks.append(bank_id)

    # 计算每个 bank 的访问次数
    bank_counts = {}
    for bank_id in banks:
        bank_counts[bank_id] = bank_counts.get(bank_id, 0) + 1

    # 最大冲突数
    conflict_degree = max(bank_counts.values())

    # 序列化因子：无冲突时为 1，有冲突时大于 1
    serialization_factor = conflict_degree

    return {
        "conflict_degree": conflict_degree,
        "serialization_factor": serialization_factor,
        "performance_impact": 1.0 / serialization_factor,
        "bank_distribution": bank_counts,
    }

# 示例 1: 无冲突
def no_conflict(tid):
    return tid * 4  # 每个线程访问不同的 bank

result = analyze_bank_conflict_impact(no_conflict)
print(f"无冲突: 冲突度={result['conflict_degree']}, 性能影响={result['performance_impact']:.1%}")

# 示例 2: 严重冲突
def severe_conflict(tid):
    return tid * 128  # 多个线程访问同一个 bank

result = analyze_bank_conflict_impact(severe_conflict)
print(f"严重冲突: 冲突度={result['conflict_degree']}, 性能影响={result['performance_impact']:.1%}")
```

这段代码是 14.5c.2 Bank Conflict 的性能影响 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 14.5c.3 Warp 利用率分析

```python
def analyze_warp_utilization(active_threads_per_warp, warp_size=32):
    """
    分析 Warp 利用率

    Warp 利用率 = 活跃线程数 / Warp 大小

    影响 Warp 利用率的因素：
    1. 分支发散（Branch Divergence）
    2. 边界条件（Boundary Conditions）
    3. 循环展开不完全
    """
    utilization = active_threads_per_warp / warp_size

    # 性能影响
    # Warp 内所有线程必须执行相同的指令
    # 如果只有部分线程活跃，其他线程会空转
    performance_ratio = utilization

    return {
        "utilization": utilization,
        "performance_ratio": performance_ratio,
        "idle_threads": warp_size - active_threads_per_warp,
    }

# 不同场景的 Warp 利用率
scenarios = {
    "全活跃": 32,
    "边界条件 (256 % 32 = 0)": 32,
    "边界条件 (300 % 32 = 28)": 28,
    "分支发散 (50%)": 16,
    "分支发散 (25%)": 8,
    "最差情况": 1,
}

for name, active in scenarios.items():
    result = analyze_warp_utilization(active)
    print(f"{name}: 利用率={result['utilization']:.1%}, 性能={result['performance_ratio']:.1%}")
```

这段代码是 14.5c.3 Warp 利用率分析 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 14.6 Thread Binding 对性能的影响

<div data-component="ThreadBindingPerformanceImpact"></div>

### 14.6.1 内存合并访问

```python
# 内存合并访问的影响

# 好的 Thread Binding: 相邻线程访问相邻地址
"""
Thread 0: addr[0]
Thread 1: addr[1]
Thread 2: addr[2]
...
Thread 31: addr[31]

→ 合并为一次 128 字节的内存事务
"""

# 差的 Thread Binding: 相邻线程访问跨步地址
"""
Thread 0: addr[0]
Thread 1: addr[32]
Thread 2: addr[64]
...
Thread 31: addr[31*32]

→ 需要 32 次独立的内存事务
"""

@T.prim_func
def good_binding(A, C):
    """好的 binding: 合并访问"""
    with T.Kernel(grid_m, threads=256) as (bx,):
        for i in T.serial(TILE_M):
            for j in T.thread_binding(TILE_N, thread="threadIdx.x"):
                # 相邻线程 (tx) 访问相邻列 (j)
                C[bx * TILE_M + i, j] = A[bx * TILE_M + i, j]

@T.prim_func
def bad_binding(A, C):
    """差的 binding: 非合并访问"""
    with T.Kernel(grid_m, threads=256) as (bx,):
        for j in T.serial(TILE_N):
            for i in T.thread_binding(TILE_M, thread="threadIdx.x"):
                # 相邻线程 (tx) 访问相邻行 (i)，但列是固定的
                # 不同线程访问同一列的不同行 → 不合并
                C[bx * TILE_M + i, j] = A[bx * TILE_M + i, j]
```

这段代码是 14.6.1 内存合并访问 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 14.6.2 Bank Conflict 分析

```python
# Bank Conflict 的影响

"""
共享内存被分为 32 个 bank (NVIDIA)
每个 bank 宽度 4 字节

Bank 冲突: 多个线程同时访问同一 bank 的不同地址

好的 binding:
  Thread 0 → Bank 0
  Thread 1 → Bank 1
  Thread 2 → Bank 2
  ...
  → 无冲突

差的 binding:
  Thread 0 → Bank 0
  Thread 1 → Bank 0  (冲突!)
  Thread 2 → Bank 0  (冲突!)
  ...
  → 32-way bank 冲突
"""

def analyze_bank_conflict(shared_mem_access, num_threads=256):
    """
    分析 bank 冲突
    """
    # 假设 32 个 bank，每个 4 字节
    num_banks = 32
    bank_width = 4

    conflicts = 0
    for thread_id in range(num_threads):
        addr = shared_mem_access(thread_id)
        bank_id = (addr // bank_width) % num_banks
        # 检查同一 bank 是否有其他线程访问
        for other_id in range(thread_id + 1, num_threads):
            other_addr = shared_mem_access(other_id)
            other_bank = (other_addr // bank_width) % num_banks
            if bank_id == other_bank:
                conflicts += 1

    return conflicts
```

这段代码是 14.6.2 Bank Conflict 分析 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 14.6.3 Warp 利用率

```python
# Warp 利用率的影响

"""
Warp 利用率 = 活跃线程数 / Warp 大小

好的情况:
  Warp 内所有 32 个线程都在执行
  利用率 = 100%

差的情况:
  只有 1 个线程在执行（其他在等待分支）
  利用率 = 3.125%
"""

@T.prim_func
def high_warp_utilization(A, B, C):
    """高 Warp 利用率"""
    with T.Kernel(grid_m, threads=256) as (bx,):
        tx = T.get_thread_id()
        # 所有线程执行相同的计算
        for i in T.serial(TILE_M):
            C[bx * TILE_M + i, tx] = A[bx * TILE_M + i, tx] + B[bx * TILE_M + i, tx]

@T.prim_func
def low_warp_utilization(A, B, C):
    """低 Warp 利用率"""
    with T.Kernel(grid_m, threads=256) as (bx,):
        tx = T.get_thread_id()
        # 分支导致大部分线程空闲
        if tx < 32:  # 只有第一个 warp 工作
            for i in T.serial(TILE_M):
                C[bx * TILE_M + i, tx] = A[bx * TILE_M + i, tx] + B[bx * TILE_M + i, tx]
```

这段代码是 14.6.3 Warp 利用率 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 14.6.4 性能对比实验

```python
def benchmark_thread_binding_strategies():
    """
    对比不同 Thread Binding 策略的性能
    """
    import time
    import torch

    M, N = 1024, 1024
    A = torch.randn(M, N, dtype=torch.float16, device="cuda")
    B = torch.randn(M, N, dtype=torch.float16, device="cuda")

    strategies = {
        "row_major": row_major_binding,
        "col_major": col_major_binding,
        "block_2d": block_2d_binding,
        "warp_tiled": warp_tiled_binding,
    }

    results = {}
    for name, kernel in strategies.items():
        compiled = tilelang.compile(kernel, target="cuda")

        # Warmup
        for _ in range(10):
            compiled(A, B)

        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            compiled(A, B)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 100

        results[name] = elapsed * 1000  # ms
        print(f"{name}: {elapsed*1000:.2f} ms")

    return results

# 预期结果:
# row_major:   0.12 ms (最佳)
# col_major:   0.45 ms (3.75x 慢)
# block_2d:    0.15 ms (1.25x 慢)
# warp_tiled:  0.13 ms (1.08x 慢)
```

这段代码是 14.6.4 性能对比实验 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 14.7 最优 Thread Binding 策略

### 14.7.1 策略选择指南

```
Thread Binding 策略选择：

┌────────────────────────────────────────────────────────────┐
│  场景                      │  推荐策略          │  原因     │
├────────────────────────────┼──────────────────┼──────────┤
│  逐元素操作               │  行优先           │  合并访问  │
│  矩阵乘法                 │  2D Block Tiling  │  平衡负载  │
│  卷积                     │  Channel Last     │  合并访问  │
│  注意力                   │  Head 并行        │  独立头    │
│  归约                     │  Warp 级别归约    │  高效通信  │
│  Scan/Prefix Sum          │  分块 + 通信      │  依赖链    │
└────────────────────────────┴──────────────────┴──────────┘
```

这个代码块或示意图用于说明 14.7.1 策略选择指南 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 14.7.2 策略 1: 行优先绑定

```python
@T.prim_func
def row_major_binding(A, B, C):
    """
    行优先绑定策略

    适用: 逐元素操作、行访问密集的操作
    特点: 相邻线程访问相邻列，实现内存合并
    """
    with T.Kernel(grid_m, threads=256) as (bx,):
        for i in T.serial(TILE_M):
            for j in T.thread_binding(TILE_N, thread="threadIdx.x"):
                C[bx * TILE_M + i, j] = A[bx * TILE_M + i, j] + B[bx * TILE_M + i, j]
```

这段代码是 14.7.2 策略 1: 行优先绑定 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 14.7.3 策略 2: 2D Block Tiling

```python
@T.prim_func
def block_2d_binding(A, B, C):
    """
    2D Block Tiling 策略

    适用: 矩阵乘法、2D 卷积
    特点: 将线程组织为 2D 网格，平衡计算和访存
    """
    with T.Kernel(grid_m, grid_n, threads=256) as (bx, by):
        # 16x16 = 256 线程
        tx = T.thread_binding(16, thread="threadIdx.x")
        ty = T.thread_binding(16, thread="threadIdx.y")

        for k in T.serial(K // TILE_K):
            # 加载 A 和 B 的 tile
            A_tile = A[bx * TILE_M + ty, k * TILE_K + tx]
            B_tile = B[k * TILE_K + ty, by * TILE_N + tx]

            # 计算
            C[bx * TILE_M + ty, by * TILE_N + tx] += A_tile * B_tile
```

这段代码是 14.7.3 策略 2: 2D Block Tiling 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 14.7.4 策略 3: Warp Tiled Binding

```python
@T.prim_func
def warp_tiled_binding(A, B, C):
    """
    Warp Tiled 绑定策略

    适用: 需要 warp 级别协作的场景
    特点: 每个 warp 处理一个子块，warp 内线程协作
    """
    with T.Kernel(grid_m, grid_n, threads=256) as (bx, by):
        tx = T.get_thread_id()
        warp_id = tx // 32
        lane_id = tx % 32

        # 4 个 warp，每个 warp 处理 TILE_M/4 行
        rows_per_warp = TILE_M // 4

        for i in T.serial(rows_per_warp):
            row = warp_id * rows_per_warp + i

            # Warp 内 32 个线程协作处理 TILE_N 列
            for j in T.serial(TILE_N // 32):
                col = lane_id + j * 32
                C[bx * TILE_M + row, by * TILE_N + col] = \
                    A[bx * TILE_M + row, by * TILE_N + col] + \
                    B[bx * TILE_M + row, by * TILE_N + col]
```

这段代码是 14.7.4 策略 3: Warp Tiled Binding 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 14.7.5 策略 4: Pipeline Binding

```python
@T.prim_func
def pipeline_binding(A, B, C):
    """
    流水线绑定策略

    适用: 计算和访存可以重叠的场景
    特点: 将线程分为计算组和访存组，重叠执行
    """
    with T.Kernel(grid_m, grid_n, threads=256) as (bx, by):
        # 将线程分为两组
        tx = T.get_thread_id()

        # 双缓冲
        A_buf = T.alloc_shared([2, TILE_M, TILE_K], "float16")
        B_buf = T.alloc_shared([2, TILE_K, TILE_N], "float16")

        # Stage 0: 预取第一块数据
        if tx < 128:
            # 线程 0-127 负责加载 A
            T.copy(A[bx * TILE_M:..., 0:TILE_K], A_buf[0])
        else:
            # 线程 128-255 负责加载 B
            T.copy(B[0:TILE_K, by * TILE_N:...], B_buf[0])

        T.sync_threads()

        for k in T.serial(K // TILE_K):
            curr = k % 2
            next = (k + 1) % 2

            # 流水线: 计算当前块 + 预取下一块
            # 计算组
            T.gemm(A_buf[curr], B_buf[curr], C_local)

            # 预取组 (与计算重叠)
            if k < K // TILE_K - 1:
                if tx < 128:
                    T.copy(A[bx * TILE_M:..., (k+1)*TILE_K:...], A_buf[next])
                else:
                    T.copy(B[(k+1)*TILE_K:..., by * TILE_N:...], B_buf[next])

            T.sync_threads()
```

这段代码是 14.7.5 策略 4: Pipeline Binding 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 14.8 自动 Thread Binding

### 14.8.1 自动调优框架

```python
import tilelang
from tilelang import autotune

@autotune(
    configs=[
        # 策略 1: 行优先
        {
            "binding": "row_major",
            "threads": 256,
            "tile_m": 128,
            "tile_n": 128,
        },
        # 策略 2: 2D Block
        {
            "binding": "block_2d",
            "threads": 256,
            "block_dim": (16, 16),
            "tile_m": 128,
            "tile_n": 128,
        },
        # 策略 3: Warp Tiled
        {
            "binding": "warp_tiled",
            "threads": 256,
            "warps_per_row": 2,
            "tile_m": 128,
            "tile_n": 128,
        },
    ],
    keys=["binding", "threads", "tile_m", "tile_n"],
    warmup=10,
    rep=100,
)
def auto_binding_kernel(binding="row_major", threads=256, tile_m=128, tile_n=128):
    @T.prim_func
    def kernel(A, B, C):
        with T.Kernel(grid_m, grid_n, threads=threads) as (bx, by):
            if binding == "row_major":
                # 行优先绑定
                for i in T.serial(tile_m):
                    for j in T.thread_binding(tile_n, thread="threadIdx.x"):
                        C[bx * tile_m + i, by * tile_n + j] = A[bx * tile_m + i, by * tile_n + j] + B[bx * tile_m + i, by * tile_n + j]
            elif binding == "block_2d":
                # 2D Block 绑定
                tx = T.thread_binding(16, thread="threadIdx.x")
                ty = T.thread_binding(16, thread="threadIdx.y")
                # ...
            elif binding == "warp_tiled":
                # Warp Tiled 绑定
                # ...
                pass
    return kernel
```

这段代码是 14.8.1 自动调优框架 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 14.8.2 基于性能模型的选择

```python
class ThreadBindingAdvisor:
    """
    基于性能模型的 Thread Binding 建议器
    """

    def __init__(self, target_gpu):
        self.target = target_gpu
        self.specs = self._get_gpu_specs(target_gpu)

    def recommend(self, kernel_type, problem_size):
        """
        根据 kernel 类型和问题大小推荐最优 binding
        """
        if kernel_type == "elementwise":
            return self._recommend_elementwise(problem_size)
        elif kernel_type == "gemm":
            return self._recommend_gemm(problem_size)
        elif kernel_type == "attention":
            return self._recommend_attention(problem_size)
        elif kernel_type == "reduce":
            return self._recommend_reduce(problem_size)

    def _recommend_gemm(self, problem_size):
        """
        矩阵乘法的 binding 推荐
        """
        M, N, K = problem_size

        # 基于 SM 数量和占用率计算最优配置
        num_sms = self.specs["num_sms"]
        max_threads = self.specs["max_threads_per_block"]
        shared_memory = self.specs["shared_memory_per_block"]

        # 计算最优 tile 大小
        # 目标: 最大化 SM 占用率
        tile_m = self._find_optimal_tile(M, num_sms)
        tile_n = self._find_optimal_tile(N, num_sms)
        tile_k = min(K, 32)  # 通常 32

        # 计算线程配置
        threads = min(tile_m * tile_n // 4, max_threads)  # 每个线程处理 4 个元素

        return {
            "binding": "block_2d",
            "tile_m": tile_m,
            "tile_n": tile_n,
            "tile_k": tile_k,
            "threads": threads,
        }
```

这段代码是 14.8.2 基于性能模型的选择 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 14.9 源码走读：Thread Binding Pass 实现

### 14.9.1 Pass 总览

```python
# 文件: tilelang/ir/transform/thread_binding.py

class ThreadBindingPass(tvm.tir.PrimFuncPass):
    """
    Thread Binding Pass

    将 TileLang 的 thread_binding 注解转换为实际的线程绑定

    主要步骤：
    1. 分析 T.thread_binding 注解
    2. 计算线程索引映射
    3. 插入线程索引计算代码
    4. 绑定到硬件线程
    """

    def transform_function(self, func, mod, ctx):
        # 1. 查找所有 thread_binding 注解
        bindings = self._find_thread_bindings(func)

        # 2. 分析绑定关系
        binding_info = self._analyze_bindings(bindings)

        # 3. 生成线程索引计算
        func = self._generate_thread_indices(func, binding_info)

        # 4. 应用硬件绑定
        func = self._apply_hardware_binding(func, binding_info)

        return func
```

这段代码是 14.9.1 Pass 总览 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 14.9.2 查找 Thread Binding 注解

```python
def _find_thread_bindings(self, func):
    """
    查找函数中所有的 thread_binding 注解

    示例输入:
      for i in T.thread_binding(128, thread="threadIdx.x"):
        ...

    返回:
      [ThreadBinding(loop_var="i", extent=128, thread="threadIdx.x")]
    """
    bindings = []

    class BindingFinder(tvm.tir.PyExprVisitor):
        def visit_for(self, loop):
            # 检查是否有 thread_binding 注解
            if hasattr(loop, 'thread_binding'):
                bindings.append(ThreadBinding(
                    loop_var=loop.loop_var,
                    extent=loop.extent,
                    thread=loop.thread_binding,
                ))
            # 继续遍历
            super().visit_for(loop)

    finder = BindingFinder()
    finder.visit_expr(func.body)

    return bindings
```

这段代码是 14.9.2 查找 Thread Binding 注解 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 14.9.3 分析绑定关系

```python
def _analyze_bindings(self, bindings):
    """
    分析 thread_binding 的关系

    检查：
    1. 是否有冲突的绑定
    2. 线程总数是否合法
    3. 绑定维度是否匹配
    """
    info = BindingInfo()

    for binding in bindings:
        if binding.thread == "threadIdx.x":
            if info.x_extent is not None:
                raise CompilationError("Multiple threadIdx.x bindings")
            info.x_extent = binding.extent
            info.x_loop_var = binding.loop_var

        elif binding.thread == "threadIdx.y":
            if info.y_extent is not None:
                raise CompilationError("Multiple threadIdx.y bindings")
            info.y_extent = binding.extent
            info.y_loop_var = binding.loop_var

        elif binding.thread == "threadIdx.z":
            if info.z_extent is not None:
                raise CompilationError("Multiple threadIdx.z bindings")
            info.z_extent = binding.extent
            info.z_loop_var = binding.loop_var

    # 验证线程总数
    total_threads = (info.x_extent or 1) * (info.y_extent or 1) * (info.z_extent or 1)
    if total_threads > 1024:
        raise CompilationError(f"Total threads {total_threads} exceeds 1024")

    info.total_threads = total_threads

    return info
```

这段代码是 14.9.3 分析绑定关系 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 14.9.4 生成线程索引

```python
def _generate_thread_indices(self, func, binding_info):
    """
    生成线程索引计算代码

    将 thread_binding 循环替换为线程索引访问
    """
    # 创建 threadIdx 变量
    tx_var = tvm.tir.Var("threadIdx_x", "int32")
    ty_var = tvm.tir.Var("threadIdx_y", "int32")
    tz_var = tvm.tir.Var("threadIdx_z", "int32")

    # 替换规则
    replacements = {}
    if binding_info.x_loop_var:
        replacements[binding_info.x_loop_var] = tx_var
    if binding_info.y_loop_var:
        replacements[binding_info.y_loop_var] = ty_var
    if binding_info.z_loop_var:
        replacements[binding_info.z_loop_var] = tz_var

    # 应用替换
    func = tvm.tir.subst(func.body, replacements)

    # 添加线程索引声明
    func = self._add_thread_declarations(func, tx_var, ty_var, tz_var)

    return func
```

这段代码是 14.9.4 生成线程索引 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 14.9.5 应用硬件绑定

```python
def _apply_hardware_binding(self, func, binding_info):
    """
    应用硬件线程绑定

    生成 CUDA: threadIdx.x, threadIdx.y, threadIdx.z
    生成 ROCm: __lane_id, etc.
    """
    if self.target.kind.name == "cuda":
        return self._apply_cuda_binding(func, binding_info)
    elif self.target.kind.name == "rocm":
        return self._apply_rocm_binding(func, binding_info)
    elif self.target.kind.name == "ascend":
        return self._apply_ascend_binding(func, binding_info)

def _apply_cuda_binding(self, func, binding_info):
    """
    应用 CUDA 线程绑定

    生成:
      int tx = threadIdx.x;
      int ty = threadIdx.y;
      int tz = threadIdx.z;
    """
    # 创建 CUDA intrinsic 调用
    tx_call = tvm.tir.call_extern("int32", "threadIdx_x")
    ty_call = tvm.tir.call_extern("int32", "threadIdx_y")
    tz_call = tvm.tir.call_extern("int32", "threadIdx_z")

    # 绑定到变量
    bindings = {}
    if binding_info.x_loop_var:
        bindings[binding_info.x_loop_var] = tx_call
    if binding_info.y_loop_var:
        bindings[binding_info.y_loop_var] = ty_call
    if binding_info.z_loop_var:
        bindings[binding_info.z_loop_var] = tz_call

    # 应用绑定
    func = tvm.tir.subst(func.body, bindings)

    return func
```

这段代码是 14.9.5 应用硬件绑定 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 14.9b Thread Binding 与 Warp Scheduling 的交互

### 14.9b.0 Warp Scheduling 详细分析

```
Warp Scheduler 工作原理：

每个 SM 有多个 Warp Scheduler（A100 有 4 个）
每个 Scheduler 每周期可以发射 1-2 条指令

调度流程：
1. 从就绪 Warp 队列中选择一个 Warp
2. 检查该 Warp 的下一条指令
3. 如果操作数就绪，发射指令
4. 否则，选择另一个就绪 Warp

Thread Binding 对调度的影响：

1. 行优先绑定
   - 相邻线程访问相邻地址
   - 内存请求可以合并
   - Warp 可以快速完成内存操作
   - 调度器可以快速切换到下一个 Warp

2. 列优先绑定
   - 相邻线程访问不同行
   - 内存请求分散
   - Warp 需要等待更多内存操作
   - 调度器可能频繁切换 Warp

3. 2D Block 绑定
   - 平衡内存访问和计算
   - 调度器可以有效利用所有 Warp
   - 适合大多数场景
```

这个代码块或示意图用于说明 14.9b.0 Warp Scheduling 详细分析 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 14.9b.1 GEMM 的最优 Thread Binding

### 14.9b.1 GEMM 的最优 Thread Binding

```python
@T.prim_func
def gemm_optimal_binding(
    A: T.Tensor([M, K], "float16"),
    B: T.Tensor([K, N], "float16"),
    C: T.Tensor([M, N], "float16"),
):
    """GEMM 的最优 Thread Binding

    策略：2D Block Tiling + Warp Tiled 内部结构

    Block 级别：
    - 每个 Block 处理 BM × BN 的输出 Tile
    - 使用 256 个线程

    Warp 级别：
    - 8 个 Warp，每个 Warp 处理 BM/8 行
    - Warp 内 32 个线程协作处理 BN 列

    Thread 级别：
    - 每个线程处理 TN 个元素
    - 相邻线程访问相邻列（合并访问）
    """
    with T.Kernel(T.ceildiv(M, BM), T.ceildiv(N, BN), threads=256) as (bx, by):
        tx = T.get_thread_id()
        warp_id = tx // 32
        lane_id = tx % 32

        # 每个 Warp 处理的行范围
        rows_per_warp = BM // 8
        warp_row_start = warp_id * rows_per_warp

        # 每个线程处理的列范围
        cols_per_thread = BN // 32

        # 累加器
        C_frag = T.alloc_fragment([rows_per_warp, BN], "float32")
        T.clear(C_frag)

        # 主循环
        for k in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
            # 加载 A 和 B 到 Shared Memory
            A_smem = T.alloc_shared([BM, BK], "float16")
            B_smem = T.alloc_shared([BK, BN], "float16")

            for i, j in T.Parallel(BM, BK):
                A_smem[i, j] = A[bx * BM + i, k * BK + j]
            for i, j in T.Parallel(BK, BN):
                B_smem[i, j] = B[k * BK + i, by * BN + j]

            # Warp 内计算
            for kk in range(BK):
                for i in range(rows_per_warp):
                    for j in range(cols_per_thread):
                        col = lane_id * cols_per_thread + j
                        C_frag[i, col] += T.cast(A_smem[warp_row_start + i, kk], "float32") * \
                                          T.cast(B_smem[kk, col], "float32")

        # 写回结果
        for i in range(rows_per_warp):
            for j in range(cols_per_thread):
                col = lane_id * cols_per_thread + j
                C[bx * BM + warp_row_start + i, by * BN + col] = T.cast(C_frag[i, col], "float16")
```

这段代码是 14.9b.1 GEMM 的最优 Thread Binding 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 14.9b.2 Attention 的 Thread Binding

```python
@T.prim_func
def attention_thread_binding(
    Q: T.Tensor([batch, seq_len, n_heads, d], "float16"),
    K: T.Tensor([batch, seq_len, n_heads, d], "float16"),
    V: T.Tensor([batch, seq_len, n_heads, d], "float16"),
    Output: T.Tensor([batch, seq_len, n_heads, d], "float16"),
):
    """Attention 的 Thread Binding 策略

    策略：Head 并行 + 序列维度分块

    Block 级别：
    - blockIdx.x → 序列块
    - blockIdx.y → Head

    Warp 级别：
    - 每个 Warp 处理 Query 的一部分行
    - Warp 内协作计算注意力分数

    Thread 级别：
    - 每个线程处理 d 维度的一部分
    """
    with T.Kernel(
        T.ceildiv(seq_len, BLOCK_M),
        n_heads,
        batch,
        threads=256
    ) as (bx, by, bz):
        tx = T.get_thread_id()
        warp_id = tx // 32
        lane_id = tx % 32

        # 每个 Warp 处理的 Query 行
        rows_per_warp = BLOCK_M // 8
        warp_row_start = warp_id * rows_per_warp

        # 加载 Q 块
        Q_smem = T.alloc_shared([BLOCK_M, d], "float16")
        for i in T.Parallel(BLOCK_M):
            for j in T.serial(d // 32):
                col = lane_id + j * 32
                Q_smem[i, col] = Q[bz, bx * BLOCK_M + i, by, col]

        # 初始化输出
        O_frag = T.alloc_fragment([rows_per_warp, d], "float32")
        T.clear(O_frag)
        m_prev = T.alloc_fragment([rows_per_warp], "float32")
        l_prev = T.alloc_fragment([rows_per_warp], "float32")
        for i in T.Parallel(rows_per_warp):
            m_prev[i] = -T.inf("float32")
            l_prev[i] = 0.0

        # 遍历 KV 块
        for kv_block in T.serial(T.ceildiv(seq_len, BLOCK_N)):
            K_smem = T.alloc_shared([BLOCK_N, d], "float16")
            V_smem = T.alloc_shared([BLOCK_N, d], "float16")

            # 加载 K, V
            for i, j in T.Parallel(BLOCK_N, d):
                K_smem[i, j] = K[bz, kv_block * BLOCK_N + i, by, j]
                V_smem[i, j] = V[bz, kv_block * BLOCK_N + i, by, j]

            # 计算 Q @ K^T
            S_frag = T.alloc_fragment([rows_per_warp, BLOCK_N], "float32")
            T.gemm(Q_smem[warp_row_start:warp_row_start + rows_per_warp, :],
                   K_smem, S_frag, transpose_B=True)
            S_frag /= T.sqrt(d)

            # Online Softmax
            m_new = T.alloc_fragment([rows_per_warp], "float32")
            T.reduce_max(S_frag, m_new, dim=1)
            m_max = T.maximum(m_prev, m_new)

            P_frag = T.alloc_fragment([rows_per_warp, BLOCK_N], "float16")
            for i, j in T.Parallel(rows_per_warp, BLOCK_N):
                P_frag[i, j] = T.exp(S_frag[i, j] - m_max[i])

            l_new = T.alloc_fragment([rows_per_warp], "float32")
            T.reduce_sum(P_frag, l_new, dim=1)
            correction = T.exp(m_prev - m_max)
            l_prev = l_prev * correction + l_new

            for i in T.Parallel(rows_per_warp):
                O_frag[i, :] *= correction[i]
            T.gemm(P_frag, V_smem, O_frag)

            m_prev = m_max

        # 写回
        for i in T.Parallel(rows_per_warp):
            O_frag[i, :] /= l_prev[i]
        for i, j in T.Parallel(rows_per_warp, d):
            Output[bz, bx * BLOCK_M + warp_row_start + i, by, j] = T.cast(O_frag[i, j], "float16")
```

这段代码是 14.9b.2 Attention 的 Thread Binding 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 14.9b.3 Reduction 的 Thread Binding

```python
@T.prim_func
def reduction_thread_binding(
    Input: T.Tensor([M, N], "float32"),
    Output: T.Tensor([M], "float32"),
):
    """Reduction 的 Thread Binding 策略

    策略：分块 + Warp Shuffle + 跨 Warp 通信

    1. 每个 Warp 处理 N/32 个元素
    2. Warp 内使用 Shuffle 归约
    3. 跨 Warp 使用 Shared Memory 归约
    """
    with T.Kernel(M, threads=256) as (bx,):
        tx = T.get_thread_id()
        warp_id = tx // 32
        lane_id = tx % 32

        # 每个线程的部分归约
        local_sum = T.float32(0)
        for j in T.serial(N // 256):
            idx = tx + j * 256
            local_sum += Input[bx, idx]

        # Warp 内归约（使用 Shuffle）
        for offset in [16, 8, 4, 2, 1]:
            local_sum += T.warp_shuffle_down(local_sum, offset)

        # 第一个线程写结果到 Shared Memory
        warp_sum_smem = T.alloc_shared([8], "float32")  # 8 个 Warp
        if lane_id == 0:
            warp_sum_smem[warp_id] = local_sum
        T.sync_threads()

        # 第一个 Warp 归约所有 Warp 的结果
        if warp_id == 0:
            final_sum = T.float32(0)
            if lane_id < 8:
                final_sum = warp_sum_smem[lane_id]
            for offset in [4, 2, 1]:
                final_sum += T.warp_shuffle_down(final_sum, offset)
            if lane_id == 0:
                Output[bx] = final_sum
```

这段代码是 14.9b.3 Reduction 的 Thread Binding 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 14.9c Thread Binding 调试技巧

### 14.9c.1 使用 Nsight Compute 分析 Thread Binding

```bash
# 分析内存合并效率
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum \
    --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum \
    ./benchmark

# 分析 Bank Conflict
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    ./benchmark

# 分析 Warp 利用率
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
    ./benchmark
```

这段命令对应 14.9c.1 使用 Nsight Compute 分析 Thread Binding 中的实际操作步骤，重点在于把环境、工具链或性能诊断流程拆成可验证的阶段。阅读时应关注每条命令的输入、输出和依赖关系，而不是机械复制；例如路径、版本、设备权限和环境变量都会影响最终结果。工程实践中建议每执行完一个阶段就做一次最小验证，这样能把问题定位在安装、编译、运行或 profiling 的具体环节。对于性能测试命令，还要注意预热、同步和重复测量，否则得到的数字很容易被缓存或异步执行干扰。

### 14.9c.2 常见 Thread Binding 问题诊断

```
问题诊断流程：

1. 性能低于预期
   ├── 检查内存合并效率
   │   ├── 效率 < 80% → 优化 Thread Binding
   │   └── 效率 > 80% → 检查其他因素
   │
   ├── 检查 Bank Conflict
   │   ├── 冲突严重 → 使用 Swizzled Layout
   │   └── 无冲突 → 检查其他因素
   │
   └── 检查 Warp 利用率
       ├── 利用率低 → 减少分支发散
       └── 利用率高 → 检查其他因素

2. 占用率低
   ├── 寄存器压力 → 减小 Tile 大小
   ├── Shared Memory → 减少 Buffer 数量
   └── 线程数 → 调整 Block 大小

3. 数值不正确
   ├── 同步问题 → 检查 T.syncthreads()
   ├── 边界条件 → 检查边界处理
   └── 数据竞争 → 检查内存访问
```

这个代码块或示意图用于说明 14.9c.2 常见 Thread Binding 问题诊断 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

---

## 14.10 高级话题

### 14.10.1 动态 Thread Binding

```python
@T.prim_func
def dynamic_binding(A, B, C, M: T.int32, N: T.int32):
    """
    动态 Thread Binding

    根据运行时输入大小动态调整 binding 策略
    """
    with T.Kernel(dynamic=True) as (bx, by):
        # 动态计算最优线程配置
        tile_m = T.min(M, 128)
        tile_n = T.min(N, 256)
        threads = T.min(tile_m * tile_n, 1024)

        # 动态绑定
        tx = T.thread_binding(threads, thread="threadIdx.x")

        # 计算
        for i in T.serial(tile_m):
            j = tx % tile_n
            if bx * tile_m + i < M and by * tile_n + j < N:
                C[bx * tile_m + i, by * tile_n + j] = \
                    A[bx * tile_m + i, by * tile_n + j] + \
                    B[bx * tile_m + i, by * tile_n + j]
```

上述动态 Thread Binding 示例展示了如何根据运行时输入大小动态计算最优的线程配置。与静态绑定不同，动态绑定在 kernel 启动时才确定 tile 维度和线程数，这使得同一份代码可以适应多种输入规模。然而，动态绑定的代价是编译器在编译期无法进行某些静态优化（如循环展开和寄存器分配），因此通常在输入尺寸不确定但对性能要求不极端的场景下使用。实际部署时，更常见的做法是编译多个静态变体并通过 dispatch 选择。

### 14.10.2 跨 Block 通信

```python
@T.prim_func
def cross_block_communication(A, B, C):
    """
    跨 Block 通信的 Thread Binding

    使用全局内存进行 Block 间通信
    """
    with T.Kernel(grid_m, threads=256) as (bx,):
        # 本地计算
        local_sum = T.alloc_fragment([TILE_M], "float32")

        for i in T.serial(TILE_M):
            for j in T.thread_binding(TILE_N, thread="threadIdx.x"):
                local_sum[i] += A[bx * TILE_M + i, j]

        # 写中间结果到全局内存
        temp = T.alloc_buffer([grid_m, TILE_M], "float32", scope="global")
        T.copy(local_sum, temp[bx, :])

        T.global_sync()  # 全局同步

        # 跨 Block 归约
        if bx == 0:
            for i in T.serial(TILE_M):
                for b in T.serial(grid_m):
                    C[i] += temp[b, i]
```

跨 Block 通信是 Thread Binding 中最复杂也最具挑战性的场景之一。在上述代码中，每个 Block 首先利用 threadIdx 维度的绑定完成局部归约，将部分和写入全局内存的临时缓冲区，然后通过 T.global_sync() 进行全局栅栏同步，最后由 Block 0 收集所有 Block 的部分结果完成最终归约。这种模式虽然功能强大，但全局同步的开销极大（通常需要通过 kernel 拆分实现），且临时缓冲区的分配会增加全局内存压力。在实际优化中，通常只在归约维度极大且无法在单个 Block 内完成时才使用跨 Block 通信。

### 14.10.3 异构计算绑定

```python
@T.prim_func
def heterogeneous_binding(A, B, C):
    """
    异构计算绑定

    同时利用 GPU 的不同计算单元
    """
    with T.Kernel(grid_m, threads=256) as (bx,):
        tx = T.get_thread_id()

        # 前 128 个线程使用向量单元
        if tx < 128:
            for i in T.serial(TILE_M):
                for j in T.serial(TILE_N // 128):
                    col = tx * (TILE_N // 128) + j
                    C[bx * TILE_M + i, col] = A[bx * TILE_M + i, col] + B[bx * TILE_M + i, col]

        # 后 128 个线程使用矩阵单元 (Tensor Core)
        else:
            # 使用 Tensor Core 进行矩阵计算
            T.tensor_core_gemm(A_tile, B_tile, C_tile)
```

上述异构计算绑定示例展示了如何在同一个 Kernel 中同时调度向量单元和矩阵单元完成不同类型的计算任务。通过线程 ID 的分段控制（前 128 个与后 128 个线程），可以实现计算资源的精细化分配。这种策略在实际硬件上尤为关键：NVIDIA 的 Tensor Core 和 CUDA Core 虽然共享线程资源但执行路径截然不同，AMD 的 MFMA 指令与普通 VALU 指令也有不同的发射约束。合理利用异构绑定可以最大化硬件利用率，但需要注意分支发散带来的 Warp 利用率下降问题。

在现代 GPU 编程中，Thread Binding 绝非简单的“映射关系”，而是连接算法逻辑与硬件物理执行的核心桥梁。通过本章从基础概念到源码实现的完整学习，我们应当建立起对循环维度到硬件线程映射的直觉，并能够在不同硬件平台和不同计算场景下做出最优的绑定策略选择。

---

## 14.11 总结

### Thread Binding 策略速查

<div data-component="ThreadBindingDiagram"></div>

```
Thread Binding 速查表：

┌──────────────────────────────────────────────────────────────┐
│  策略              │  适用场景           │  内存合并 │ 占用率  │
├────────────────────┼────────────────────┼──────────┼────────┤
│  行优先            │  逐元素、行访问     │  ✅ 高   │  ✅ 高  │
│  列优先            │  列访问             │  ❌ 低   │  ✅ 高  │
│  2D Block          │  GEMM、2D 卷积     │  ✅ 高   │  ✅ 高  │
│  Warp Tiled        │  Warp 协作          │  ✅ 高   │  ⚠️ 中  │
│  Pipeline          │  计算-访存重叠      │  ✅ 高   │  ⚠️ 中  │
│  Head 并行         │  多头注意力         │  ✅ 高   │  ✅ 高  │
└────────────────────┴────────────────────┴──────────┴────────┘

选择原则:
1. 优先保证内存合并访问
2. 次要考虑 Warp 利用率
3. 最后考虑占用率
```

这张速查表总结了本章讨论的六种核心 Thread Binding 策略及其在不同维度上的表现。在选择策略时，内存合并访问应当是首要考量——换行优先绑定和 2D Block Tiling 之所以在大多数场景下表现最佳，正是因为它们天然地保证了相邻线程访问相邻地址。占用率虽然重要，但通常可以通过调整 Block 大小来优化，不应以牺牲内存合并为代价。

### 硬件映射总结

```
硬件映射总结：

CUDA (NVIDIA):
  Grid → BlockIdx
  Block → Warp (32 threads)
  Thread → threadIdx

ROCm (AMD):
  Grid → WorkgroupId
  Workgroup → Wavefront (64 threads)
  Thread → __lane_id

Ascend (华为):
  Grid → AI Core
  AI Core → Cube/Vector Unit
  线程 → Data Agent 控制
```

上述硬件映射总结展示了三大平台的线程模型如何统一映射到 TileLang 的抽象层：CUDA 的 Block/Warp/Thread 三级层次、ROCm 的 Workgroup/Wavefront 两级以及昇腾的 AI Core 单级模型。理解这些映射差异至关重要，因为它决定了 Thread Binding 时线程粒度的选择——例如在 CUDA 上绑定到 Warp 级别可以获得高效的 Shuffle 通信，而在昇腾上则更适合以数据流的方式组织计算任务，让 Data Agent 自动管理数据搬运与计算的并行执行。

---

## 练习

### 基础练习

1. **内存合并**：编写一个 kernel，分别使用行优先和列优先 binding，对比带宽利用率。

2. **Warp 利用率**：编写一个有分支的 kernel，分析不同 binding 策略对 warp 利用率的影响。

3. **Bank Conflict**：分析以下代码的 bank conflict 情况，并提出优化方案：
   ```python
    for i in T.serial(32):
        A_smem[i, threadIdx.x] = ...
    ```

上述代码展示了共享内存访问的一种典型模式：循环维度 i 控制对共享内存的行写入，而 threadIdx.x 控制列索引。如果 A_smem 的列维度与 threadIdx.x 对齐，那么相邻线程会访问相邻的共享内存地址，从而避免 Bank Conflict。在实际优化中，可以通过调整 Thread Binding 的顺序或引入列维度偏移（如对地址进行 XOR 混洗）来消除 Bank Conflict，确保所有线程在单周期内完成共享内存访问。

### 进阶练习

4. **自动调优**：实现一个自动 Thread Binding 调优器，搜索最优的 binding 策略。

5. **跨平台**：编写一份 TileLang 代码，使其在 NVIDIA、AMD 和华为 NPU 上都能高效执行。

6. **性能分析**：使用 profiling 工具分析不同 binding 策略的性能瓶颈。

---

## 思考题

1. **设计思考**：为什么 GPU 使用 Warp/Wavefront 这种 SIMT 模型？这对 Thread Binding 有什么影响？

2. **硬件思考**：未来的 GPU 可能有哪些新的线程模型？Thread Binding 策略需要如何调整？

3. **抽象思考**：TileLang 的 Thread Binding 抽象是否足够？如果要支持更多硬件，需要哪些扩展？

---

## 扩展阅读

1. **CUDA Programming Guide**：深入理解 CUDA 线程模型和内存层次
2. **ROCm Documentation**：AMD GPU 的编程模型和优化指南
3. **Ascend C Programming Guide**：华为 NPU 的编程模型
4. **GPU Optimization Guide**：各种 GPU 的优化最佳实践
5. **Warp-level Programming**：理解 warp/wavefront 级别的编程技巧

---

## 课程总结

经过本章的学习，我们已经完整地掌握了 TileLang 的核心知识：

1. **FlashMLA 算子实现**：从算法到代码
2. **IR 与 TensorIR**：底层表示与编译原理
3. **TVM/Relax 集成**：系统级编译管线
4. **编译管线全景**：从 Python 到机器码
5. **Thread Binding**：硬件线程映射与性能优化

这些知识构成了 TileLang 的完整技术栈，从高层算法设计到底层硬件优化，为编写高性能 GPU kernel 提供了全面的工具和方法。

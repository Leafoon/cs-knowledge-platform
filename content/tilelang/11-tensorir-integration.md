---
title: "Chapter 11: TileLang IR 与 TensorIR 的关系"
description: "深入理解 TileLang IR 设计、Buffer/Region 抽象、Block/Loop 结构、与 TensorIR 的映射规则及 Lowering 过程"
updated: 2026-06-11
---

# Chapter 11: TileLang IR 与 TensorIR 的关系

> **Learning Objectives**
>
> 1. 理解 TileLang IR 的设计理念：基于 TVM TensorIR 的领域特定扩展
> 2. 掌握 Buffer/Region 抽象的核心概念与使用方法
> 3. 理解 Block/Loop 结构在 TileLang 中的特殊语义
> 4. 学习 TileLang IR 到 TensorIR 的完整映射规则
> 5. 掌握 Lowering 过程的每个阶段
> 6. 了解 TileLang 在 TVM 生态中的定位
> 7. 学会使用 IR dump 工具进行调试

---

## 11.1 TileLang IR 设计概述

### 11.1.1 为什么需要 TileLang IR

在 GPU 编程中，开发者需要手动管理大量的底层细节：线程映射、共享内存分配、数据搬运、同步屏障等。这些细节虽然重要，但它们掩盖了算法的核心逻辑。

TileLang IR 的设计目标是：

```
抽象层次对比：

传统 CUDA:
  算法逻辑 + 线程管理 + 内存管理 + 同步 + 指令选择
  (所有细节混合在一起)

TileLang:
  算法逻辑 (TileLang IR)
      │
      ▼ Lowering
  线程管理 + 内存管理 + 同步 + 指令选择 (TensorIR/TIR)
      │
      ▼ CodeGen
  PTX / HSACO / Ascend C

开发者只需关注上层，底层由编译器自动处理
```

这段 ASCII 图表清晰地展示了 TileLang 的抽象层次设计思路。在传统的 CUDA 编程模型中，开发者必须同时处理算法逻辑、线程管理、内存分配、同步操作以及底层指令选择等所有细节，这导致代码复杂度极高且难以维护。TileLang 通过引入 IR 层次结构，将这些关注点进行了分离：上层开发者只需要编写算法逻辑（TileLang IR），然后通过 Lowering 过程自动完成线程管理、内存管理、同步和指令选择等底层工作（TensorIR/TIR），最终通过 CodeGen 生成目标平台的机器码（PTX/HSACO/Ascend C）。这种分层设计大幅降低了 GPU kernel 开发的门槛，同时保留了对底层硬件的精细控制能力。从性能角度来看，编译器能够在 Lowering 阶段进行全局优化，如循环融合、内存访问合并等，这些优化在手写代码中往往难以实现。

### 11.1.2 TileLang IR 与 TensorIR 的关系

TileLang IR 是建立在 TVM TensorIR 之上的领域特定扩展。它不是一个全新的 IR，而是为张量计算（特别是 GPU kernel）提供了更高层次的抽象：

```
IR 层次结构：

┌─────────────────────────────────────────┐
│           TileLang DSL (Python)         │  ← 用户编写的 @T.prim_func
├─────────────────────────────────────────┤
│           TileLang IR                   │  ← Tile/Region/Block 抽象
├─────────────────────────────────────────┤
│           TensorIR (TVM)                │  ← Buffer/Block/Loop
├─────────────────────────────────────────┤
│           TIR (Low-level)               │  ← 标准 TIR 表示
├─────────────────────────────────────────┤
  │           LLVM IR / PTX / HSACO         │  ← 目标代码
└─────────────────────────────────────────┘
```

这个层次结构图详细描述了从用户代码到最终机器码的完整 IR 转换路径。最顶层是用户编写的 TileLang DSL（Python 代码），通过 `@T.prim_func` 装饰器定义 kernel 函数。第二层是 TileLang IR，使用 Tile/Region/Block 等高层抽象来表示计算逻辑，开发者可以在此层次上进行算子融合和数据流分析。第三层是 TensorIR，这是 TVM 的标准 IR 表示，使用 Buffer/Block/Loop 结构来描述计算和内存访问。第四层是 TIR（低级 TIR），经过各种优化 Pass 后生成的低级中间表示，已经完成了循环展开、向量化等优化。最底层是目标代码，根据目标硬件平台生成 PTX（NVIDIA GPU）、HSACO（AMD GPU）或 Ascend C（华为 NPU）等特定指令集。每一层的抽象都屏蔽了下层的复杂性，同时为上层提供更高层次的编程接口。

### 11.1.3 设计原则

TileLang IR 遵循以下设计原则：

1. **Tile-centric**：以 tile（块）为核心抽象，而非单个元素
2. **Memory-aware**：显式管理内存层次（寄存器、共享内存、全局内存）
3. **Hardware-abstract**：隐藏硬件线程映射细节
4. **Composable**：支持算子融合和组合
5. **Compatible**：与 TVM 生态完全兼容

---

## 11.2 Buffer 与 Region 抽象

<div data-component="BufferRegionDiagram"></div>

### 11.2.1 Buffer 概念

在 TileLang IR 中，`Buffer` 是对内存区域的抽象。它描述了一块连续内存的形状、数据类型和存储位置：

```python
# TileLang 中的 Buffer 声明
@T.prim_func
def kernel(
    A: T.Buffer[(M, N), "float16"],      # 输入 Buffer
    B: T.Buffer[(N, K), "float16"],      # 输入 Buffer
    C: T.Buffer[(M, K), "float32"],      # 输出 Buffer
):
    # Buffer 的属性:
    # - shape: (M, N), (N, K), (M, K)
    # - dtype: float16, float32
    # - scope: global (默认)
    pass
```

这段代码展示了 TileLang 中 Buffer 的基本声明方式。函数参数 A、B、C 都是 `T.Buffer` 类型，分别表示输入矩阵和输出矩阵。每个 Buffer 声明包含两个核心属性：shape（形状）和 dtype（数据类型）。例如 `A: T.Buffer[(M, N), "float16"]` 表示 A 是一个 M×N 的 float16 矩阵。注释中提到的 `scope: global (默认)` 表示这些 Buffer 默认位于全局内存（HBM）中。在 GPU 编程中，Buffer 的声明是 kernel 编写的第一步，它决定了数据在内存中的布局和访问方式。TileLang 通过 Buffer 抽象隐藏了底层的内存管理细节，开发者无需手动分配和释放 GPU 内存，编译器会自动处理这些操作。此外，Buffer 的 shape 可以是动态的（使用 T.Var），也可以是静态的（使用常量），这为不同场景下的 kernel 编写提供了灵活性。

Buffer 的三种存储作用域：

```python
# 1. Global Buffer (HBM/Global Memory)
A_global = T.Buffer[(M, N), "float16", scope="global"]

# 2. Shared Buffer (Shared Memory)
A_shared = T.Buffer[(tile_m, tile_n), "float16", scope="shared"]

# 3. Local Buffer (Registers)
A_local = T.Buffer[(tile_m, tile_n), "float16", scope="local"]
```

这段代码展示了 TileLang 中三种不同存储作用域的 Buffer 声明方式。第一种 `scope="global"` 表示全局内存（HBM），这是 GPU 上最大的内存池，容量可达数十 GB，但访问延迟较高。第二种 `scope="shared"` 表示共享内存（Shared Memory），这是位于 SM（流多处理器）上的高速 SRAM，容量较小（通常 16-228 KB），但访问延迟极低，且支持同一线程块内所有线程的高速访问。第三种 `scope="local"` 表示寄存器（Registers），这是 GPU 上速度最快的存储介质，容量极小但延迟几乎为零。在 GPU kernel 优化中，合理的内存层次管理是关键：将数据从全局内存加载到共享内存，再从共享内存加载到寄存器进行计算，可以显著提升性能。TileLang 通过 Buffer 的 scope 参数让开发者显式指定数据的存储位置，编译器会自动处理不同层次间的数据搬运和同步操作。这种设计既保留了对内存层次的精细控制，又避免了手动管理共享内存的复杂性。

### 11.2.2 Region 抽象

`Region` 是 TileLang 的核心创新之一。它描述了对 Buffer 的一个"区域"访问，支持 tile 级别的操作：

```python
# Region 的基本形式
region = T.Region(buffer[start:end, start:end])

# 示例：访问矩阵 A 的一个 tile
with T.Kernel(...) as (bx, by):
    # 定义 tile 大小
    TILE_M = 128
    TILE_N = 64

    # Region: A 的一个 128x64 子块
    A_tile = A[bx * TILE_M : (bx + 1) * TILE_M,
               by * TILE_N : (by + 1) * TILE_N]

    # 这个 Region 可以被:
    # 1. 加载到共享内存
    T.copy(A_tile, A_shared)
    # 2. 用于 GEMM 计算
    T.gemm(A_shared, B_shared, C_local)
```

这段代码展示了 Region 抽象的核心用法。Region 是 TileLang 的核心创新之一，它描述了对 Buffer 的一个"区域"访问，支持 tile 级别的操作。在这个例子中，`A_tile = A[bx * TILE_M : (bx + 1) * TILE_M, by * TILE_N : (by + 1) * TILE_N]` 创建了一个指向矩阵 A 的子块的 Region，大小为 TILE_M × TILE_N。Region 的关键特性是它只是一个"视图"（view），不涉及数据复制。当我们执行 `T.copy(A_tile, A_shared)` 时，数据才会从全局内存实际搬运到共享内存。这种设计使得 tile 级别的操作变得非常自然：开发者可以像操作普通矩阵一样操作 Region，编译器会自动处理底层的内存访问和边界检查。在 GEMM 计算中，Region 使得矩阵分块操作变得直观，开发者可以清晰地看到数据的分块策略和计算逻辑。从性能角度看，Region 的引入使得编译器能够更好地分析数据访问模式，从而进行更有效的内存访问优化和循环变换。

### 11.2.3 Buffer 与 Region 的关系

```
Buffer vs Region:

Buffer (完整矩阵):
┌───────────────────────────────────────┐
│                                       │
│           A: [M × N]                  │
│                                       │
└───────────────────────────────────────┘

Region (子块访问):
┌───────────────────────────────────────┐
│      ┌─────────┐                      │
│      │ A_tile  │ ← Region            │
│      │ [128×64]│   [bx*128:(bx+1)*128│
│      └─────────┘    , by*64:(by+1)*64]
│                                       │
└───────────────────────────────────────┘

Region 是 Buffer 的一个"视图"，不复制数据
```

这个图表直观地展示了 Buffer 和 Region 之间的关系。上方的图示表示一个完整的 Buffer（矩阵 A），包含 M×N 个元素，对应 GPU 上一块连续的全局内存区域。下方的图示表示一个 Region（A_tile），它是 Buffer 的一个子块，大小为 128×64，通过切片操作 `[bx*128:(bx+1)*128, by*64:(by+1)*64]` 从原始 Buffer 中提取。Region 的核心特性是零拷贝——它只是维护了指向原始 Buffer 的指针和偏移量信息，不涉及任何数据复制操作。这种设计在 GPU 编程中至关重要，因为数据复制会带来显著的性能开销。Region 使得 tile 级别的操作变得高效：当需要加载一个 tile 到共享内存时，Region 提供了精确的内存地址信息，编译器可以生成高效的内存访问指令。此外，Region 的引入使得算子融合成为可能：多个操作可以共享同一个 Region，避免中间结果的内存分配和复制。

### 11.2.4 多级 Buffer 映射

```python
@T.prim_func
def gemm_with_memory_hierarchy(
    A: T.Buffer[(M, K), "float16"],    # Global
    B: T.Buffer[(K, N), "float16"],    # Global
    C: T.Buffer[(M, N), "float32"],    # Global
):
    with T.Kernel(grid_m, grid_n, threads=256) as (bx, by):
        # 共享内存 Buffer
        A_smem = T.alloc_shared([TILE_M, TILE_K], "float16")
        B_smem = T.alloc_shared([TILE_K, TILE_N], "float16")

        # 寄存器 Buffer
        C_reg = T.alloc_fragment([TILE_M, TILE_N], "float32")

        # 数据流: Global → Shared → Local → Shared → Global
        for k in range(K // TILE_K):
            # Global → Shared
            T.copy(A[bx*TILE_M:(bx+1)*TILE_M, k*TILE_K:(k+1)*TILE_K], A_smem)
            T.copy(B[k*TILE_K:(k+1)*TILE_K, by*TILE_N:(by+1)*TILE_N], B_smem)

            # Shared → Local (通过 GEMM)
            T.gemm(A_smem, B_smem, C_reg)

        # Local → Global
        T.copy(C_reg, C[bx*TILE_M:(bx+1)*TILE_M, by*TILE_N:(by+1)*TILE_N])
```

这段代码展示了 TileLang 中多级内存层次的完整使用模式。函数定义了三个全局 Buffer（A、B、C），然后在 T.Kernel 内部分配了共享内存 Buffer（A_smem、B_smem）和寄存器 Buffer（C_reg）。数据流路径为：Global → Shared → Local → Shared → Global。首先，`T.copy` 将全局内存中的 tile 加载到共享内存，这一步利用了共享内存的高带宽特性来隐藏全局内存的访问延迟。然后，`T.gemm` 执行共享内存到寄存器的矩阵乘法计算，利用寄存器的超低延迟来加速计算。最后，`T.copy` 将结果从寄存器写回全局内存。这种多级内存管理策略是 GPU kernel 优化的核心：通过合理地在不同内存层次间搬运数据，可以最大化内存带宽利用率和计算吞吐量。TileLang 通过 `T.alloc_shared` 和 `T.alloc_fragment` 让开发者显式管理内存层次，而编译器会自动处理内存分配、同步和数据搬运的细节。在实际应用中，tile 大小的选择需要在共享内存容量、寄存器压力和计算强度之间找到平衡。

---

## 11.3 Block 与 Loop 结构

<div data-component="BlockLoopStructure"></div>

### 11.3.1 Block 概念

Block 是 TileLang IR 中的计算单元。每个 Block 包含：
- **读区域**（Read Region）：Block 读取的 Buffer 区域
- **写区域**（Write Region）：Block 写入的 Buffer 区域
- **计算体**（Body）：实际的计算逻辑

```python
# Block 的概念表示
"""
block compute_tile [bx, by]:
    reads: A[bx*TM:(bx+1)*TM, :], B[:, by*TN:(by+1)*TN]
    writes: C[bx*TM:(bx+1)*TM, by*TN:(by+1)*TN]

    body:
        for k in range(K // TK):
            A_tile = A[bx*TM:(bx+1)*TM, k*TK:(k+1)*TK]
            B_tile = B[k*TK:(k+1)*TK, by*TN:(by+1)*TN]
            C[bx*TM:...] += A_tile @ B_tile
"""
```

### 11.3.2 Loop 结构

TileLang 中的循环结构有特殊语义：

```python
# 1. T.serial - 串行循环
for i in T.serial(N):
    body[i]

# 2. T.parallel - 并行循环（映射到线程块）
for i in T.parallel(grid_m):
    body[i]

# 3. T.vectorize - 向量化循环
for i in T.vectorize(4):
    body[i]  # 映射到 SIMD 指令

# 4. T.unroll - 展开循环
for i in T.unroll(8):
    body[i]  # 编译时展开
```

### 11.3.3 Kernel 启动与 Grid 映射

```python
@T.prim_func
def kernel_with_grid(
    A: T.Buffer[(M, N), "float16"],
    B: T.Buffer[(M, N), "float16"],
    C: T.Buffer[(M, N), "float16"],
):
    # T.Kernel 定义 GPU kernel 的启动配置
    # grid_dim = (num_blocks_x, num_blocks_y, num_blocks_z)
    # block_dim = (threads_per_block,)
    with T.Kernel(grid_m, grid_n, threads=256) as (bx, by):
        # bx, by 是线程块索引
        # 在 CUDA 中对应 blockIdx.x, blockIdx.y

        # 分配共享内存
        A_tile = T.alloc_shared([TILE_M, TILE_N], "float16")

        # 加载数据
        T.copy(A[bx*TILE_M:(bx+1)*TILE_M, by*TILE_N:(by+1)*TILE_N], A_tile)

        # 计算
        for i in T.serial(TILE_M):
            for j in T.serial(TILE_N):
                C[bx*TILE_M+i, by*TILE_N+j] = A_tile[i, j] * 2.0
```

### 11.3.4 Thread Binding

```python
@T.prim_func
def kernel_with_thread_binding(
    A: T.Buffer[(M, N), "float16"],
    B: T.Buffer[(M, N), "float16"],
    C: T.Buffer[(M, N), "float16"],
):
    with T.Kernel(grid_m, grid_n, threads=256) as (bx, by):
        # 获取线程 ID
        tx = T.get_thread_id()  # 0..255

        # 手动线程绑定（高级用法）
        for i in T.serial(TILE_M):
            # 将 i 循环绑定到 tx
            tid = tx % TILE_M
            if tid == i:
                for j in T.serial(TILE_N):
                    C[bx*TILE_M+i, by*TILE_N+j] = A[bx*TILE_M+i, by*TILE_N+j]
```

---

## 11.4 TileLang IR 到 TensorIR 的映射

<div data-component="TileLangIRToTensorIRFlow"></div>

### 11.4.1 映射规则概述

TileLang IR 到 TensorIR 的 Lowering 过程遵循一组系统化的映射规则：

```
TileLang IR → TensorIR 映射规则：

┌─────────────────┬──────────────────────────────────────┐
│  TileLang       │  TensorIR                            │
├─────────────────┼──────────────────────────────────────┤
│  T.Buffer       │  tir.Buffer                          │
│  T.Region       │  tir.BufferRegion                    │
│  T.Kernel       │  tir.block + tir.For (grid)          │
│  T.alloc_shared │  tir.alloc_buffer(scope="shared")    │
│  T.alloc_frag   │  tir.alloc_buffer(scope="local")     │
│  T.copy         │  tir.block (load/compute)            │
│  T.gemm         │  tir.block (matmul compute)          │
│  T.fill         │  tir.block (初始化)                   │
│  T.serial       │  tir.ForKind.serial                  │
│  T.parallel     │  tir.ForKind.parallel                │
│  T.vectorize    │  tir.ForKind.vectorized              │
│  T.unroll       │  tir.ForKind.unrolled                │
└─────────────────┴──────────────────────────────────────┘
```

### 11.4.2 Buffer 映射详解

```python
# TileLang 源码
@T.prim_func
def add_vectors(
    A: T.Buffer[(N,), "float32"],
    B: T.Buffer[(N,), "float32"],
    C: T.Buffer[(N,), "float32"],
):
    with T.Kernel(1, threads=256) as (bx,):
        for i in T.serial(N):
            C[i] = A[i] + B[i]

# 映射后的 TensorIR
"""
@T.prim_func
def add_vectors(A: T.handle, B: T.handle, C: T.handle):
    N = T.int32()
    A_buf = T.match_buffer(A, (N,), dtype="float32")
    B_buf = T.match_buffer(B, (N,), dtype="float32")
    C_buf = T.match_buffer(C, (N,), dtype="float32")

    for bx in T.thread_binding(1, thread="blockIdx.x"):
        for i in T.serial(N):
            with T.block("compute"):
                C_buf[i] = A_buf[i] + B_buf[i]
"""
```

### 11.4.3 Kernel 到 Block 的映射

```python
# TileLang: T.Kernel 创建一个 GPU kernel
"""
with T.Kernel(grid_m, grid_n, threads=256) as (bx, by):
    body
"""

# TensorIR: 映射为嵌套的 For 循环 + Block
"""
# Grid 循环 (映射到 blockIdx)
for bx in T.thread_binding(grid_m, thread="blockIdx.x"):
    for by in T.thread_binding(grid_n, thread="blockIdx.y"):
        with T.block("kernel"):
            # Thread 循环 (映射到 threadIdx)
            for tx in T.thread_binding(256, thread="threadIdx.x"):
                body
"""
```

### 11.4.4 内存分配映射

```python
# TileLang 中的内存分配
"""
A_smem = T.alloc_shared([128, 64], "float16")   # 共享内存
C_reg = T.alloc_fragment([128, 64], "float32")   # 寄存器
"""

# 映射到 TensorIR
"""
# 共享内存
A_smem = T.alloc_buffer((128, 64), dtype="float16", scope="shared")

# 寄存器 (local)
C_reg = T.alloc_buffer((128, 64), dtype="float32", scope="local")
"""
```

### 11.4.5 GEMM 操作映射

```python
# TileLang 中的 GEMM
"""
T.gemm(A_smem, B_smem, C_reg, transpose_B=False)
"""

# 映射到 TensorIR (展开为嵌套循环)
"""
for i in T.serial(128):
    for j in T.serial(64):
        for k in T.serial(32):
            with T.block("gemm"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                T.reads(A_smem[vi, vk], B_smem[vk, vj])
                T.writes(C_reg[vi, vj])
                with T.init():
                    C_reg[vi, vj] = T.float32(0)
                C_reg[vi, vj] += A_smem[vi, vk] * B_smem[vk, vj]
"""
```

---

## 11.5 Lowering 过程详解

### 11.5.1 Lowering Pipeline 总览

```
完整的 Lowering 流程：

TileLang DSL (Python)
       │
       ▼
┌──────────────────┐
│  TileLang Parser │  解析 @T.prim_func 装饰器
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  TileLang IR     │  Tile/Region/Block 高层表示
└────────┬─────────┘
         │
         ▼ TileLang → TensorIR Pass
┌──────────────────┐
│  TensorIR        │  Buffer/Block/Loop 标准表示
└────────┬─────────┘
         │
         ▼ TIR Lowering Passes
┌──────────────────┐
│  TIR             │  低级 TIR 表示
└────────┬─────────┘
         │
         ▼ CodeGen
┌──────────────────┐
│  Target Code     │  PTX/HSACO/LLVM IR
└──────────────────┘
```

### 11.5.2 Pass 1: TileLangLowerTileOp

这个 Pass 将 TileLang 的高层 tile 操作降低到标准的循环结构：

```python
# 输入 (TileLang IR)
"""
with T.Kernel(4, 8, threads=256) as (bx, by):
    A_smem = T.alloc_shared([128, 64], "float16")
    T.copy(A[bx*128:(bx+1)*128, by*64:(by+1)*64], A_smem)
"""

# 输出 (TensorIR)
"""
for bx in T.thread_binding(4, thread="blockIdx.x"):
    for by in T.thread_binding(8, thread="blockIdx.y"):
        A_smem = T.alloc_buffer((128, 64), dtype="float16", scope="shared")
        for i in T.serial(128):
            for j in T.serial(64):
                with T.block("copy"):
                    A_smem[i, j] = A[bx * 128 + i, by * 64 + j]
"""
```

### 11.5.3 Pass 2: TileLangInferLayout

这个 Pass 推断数据的内存布局，决定如何将逻辑 tile 映射到物理内存：

```python
# 推断前
"""
A_smem = T.alloc_buffer((128, 64), dtype="float16", scope="shared")
"""

# 推断后 (添加 swizzle 信息以避免 bank conflict)
"""
A_smem = T.alloc_buffer((128, 64), dtype="float16", scope="shared",
                         layout="swizzle<3, 0, 4>")
"""
```

### 11.5.4 Pass 3: TileLegalizeLegacy

这个 Pass 处理各种合法性检查和转换：

```python
# 检查和转换：
# 1. 共享内存大小是否超出限制
# 2. 线程数量是否合法
# 3. 数据类型转换
# 4. 边界条件处理
```

### 11.5.5 Pass 4: LowerThreadAllreduce

将高层的归约操作降低为 warp/block 级别的归约原语：

```python
# 输入
"""
result = T.sum(local_values, axis=0)
"""

# 输出 (CUDA)
"""
# Warp-level reduction
for offset in T.unroll([16, 8, 4, 2, 1]):
    local_values[tx] += T.shuffle_down(local_values[tx], offset)

# Block-level reduction (通过 shared memory)
if tx % 32 == 0:
    smem[tx // 32] = local_values[tx]
T.sync_threads()
if tx < 32:
    # Warp-level reduction of partial sums
    ...
"""
```

---

## 11.6 TileLang 与 TVM 编译管线协同

### 11.6.1 TVM 编译管线概览

```
TVM 编译管线中 TileLang 的位置：

Python Model (PyTorch/ONNX)
       │
       ▼
┌──────────────────┐
│  Relax (Graph IR)│  高层计算图
└────────┬─────────┘
         │
         ▼ Fuse/Partition
┌──────────────────┐
│  TensorIR        │  算子级别 IR
└────────┬─────────┘
         │
         ▼ TileLang Integration
┌──────────────────┐
│  TileLang IR     │  TileLang 调度
└────────┬─────────┘
         │
         ▼ TIR Passes
┌──────────────────┐
│  TIR (Lowered)   │  低级 IR
└────────┬─────────┘
         │
         ▼ CodeGen
┌──────────────────┐
│  LLVM/PTX/HSACO  │  目标代码
└──────────────────┘
```

### 11.6.2 算子注册与调度绑定

```python
import tvm
from tvm import relax
import tilelang

# 定义 TileLang kernel
@T.prim_func
def my_gemm(A, B, C):
    with T.Kernel(...) as (bx, by):
        # ... TileLang GEMM 实现 ...
        pass

# 注册到 TVM Relax
@relax.expr_functor.register("my_gemm")
class MyGEMMFunctor:
    def create(self, builder, call):
        return builder.call_te(my_gemm, *call.args)

# 或者通过调度原语绑定
def schedule_gemm(sch, block):
    """使用 TileLang 调度原语"""
    tilelang.schedule.AutoInline(sch, block)
    tilelang.schedule.Tile(sch, block, [128, 128, 32])
```

### 11.6.3 TileLang Kernel 嵌入 Relax 图

```python
import tilelang
from tilelang import tvm as tvm

# 定义完整的模型
def build_model_with_tilelang():
    # 1. 定义计算图
    bb = relax.BlockBuilder()

    with bb.function("main"):
        # 标准 TVM Relax 算子
        x = bb.emit(relax.op.nn.linear(input, weight, bias))

        # 嵌入 TileLang kernel
        tilelang_output = bb.emit_te(
            flash_mla_kernel,  # TileLang 定义的 kernel
            Q=x, c_KV=kv_cache, W_UK=w_uk, W_UV=w_uv
        )

        # 继续使用标准算子
        output = bb.emit(relax.op.nn.linear(tilelang_output, out_weight, out_bias))
        bb.emit_func_output(output)

    # 2. 编译
    mod = bb.get()
    ex = relax.build(mod, target="cuda")
    return ex
```

---

## 11.7 TileLang 在 TVM 生态中的定位

### 11.7.1 生态系统全景

```
TVM 生态系统中的 TileLang：

                    ┌─────────────────────────────┐
                    │        用户模型              │
                    │  (PyTorch / ONNX / JAX)      │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │     Relax (Graph IR)         │
                    │  - 算子融合                  │
                    │  - 内存规划                  │
                    │  - 数据流优化                │
                    └─────────────┬───────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
  ┌─────────▼─────────┐ ┌────────▼────────┐ ┌─────────▼─────────┐
  │   标准 TVM 调度    │ │   TileLang      │ │   CUTLASS/ROCm    │
  │   (AutoTVM/       │ │   (手写调度)     │ │   (外部库)         │
  │    MetaSchedule)  │ │                 │ │                   │
  └─────────┬─────────┘ └────────┬────────┘ └─────────┬─────────┘
            │                     │                     │
            └─────────────────────┼─────────────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │     TIR Lowering             │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │     CodeGen                  │
                    │  (LLVM / PTX / HSACO)        │
                    └─────────────────────────────┘
```

### 11.7.2 TileLang vs AutoTVM vs MetaSchedule

| 特性 | AutoTVM | MetaSchedule | TileLang |
|------|---------|--------------|----------|
| 调度方式 | 模板 + 搜索 | 搜索 + 学习 | 手写 + 自动优化 |
| 开发效率 | 中 | 高 | 高 |
| 性能上限 | 中 | 中-高 | 高 |
| 学习曲线 | 陡峭 | 中等 | 平缓 |
| 适用场景 | 通用算子 | 通用算子 | GPU kernel |
| 硬件适配 | 需要模板 | 自动探索 | 显式指定 |

### 11.7.3 混合使用策略

```python
# 策略：对关键算子使用 TileLang，其他使用标准调度

def build_optimized_model(model):
    bb = relax.BlockBuilder()

    with bb.function("main"):
        # 标准算子使用 TVM 默认调度
        x = bb.emit(relax.op.nn.embedding(tokens, emb_table))

        # 关键注意力算子使用 TileLang
        attn_out = bb.emit_te(
            flash_mla_kernel,  # TileLang 优化
            Q=x, c_KV=kv_cache, ...
        )

        # FFN 使用 MetaSchedule 自动调优
        ffn_out = bb.emit(relax.op.nn.linear(attn_out, ffn_weight))

        # 另一个关键算子使用 TileLang
        moe_out = bb.emit_te(
            moe_routing_kernel,  # TileLang 优化
            x=ffn_out, experts=expert_weights
        )

        output = bb.emit(relax.op.nn.linear(moe_out, out_weight))
        bb.emit_func_output(output)

    return bb.get()
```

---

## 11.8 IR Dump 与调试方法

<div data-component="IRDumperVisualizer"></div>

### 11.8.1 IR Dump 工具

TileLang 提供了多种 IR dump 工具来帮助调试：

```python
import tilelang
from tilelang import T, ir

@T.prim_func
def my_kernel(A, B, C):
    with T.Kernel(...) as (bx, by):
        # ... kernel 实现 ...
        pass

# 1. 打印 TileLang IR
print("=== TileLang IR ===")
print(ir.dump_tilelang(my_kernel))

# 2. 打印 TensorIR (Lowered)
print("\n=== TensorIR ===")
lowered = ir.lower_tilelang(my_kernel)
print(lowered)

# 3. 打印 TIR (最终)
print("\n=== TIR ===")
tir = ir.lower_to_tir(lowered)
print(tir)

# 4. 打印目标代码
print("\n=== PTX ===")
ptx = ir.codegen(tir, target="cuda")
print(ptx)
```

### 11.8.2 IR Dump 输出示例

```
=== TileLang IR ===
@T.prim_func
def flash_mla(Q: Buffer[(32, 128, 128), "float16"],
              c_KV: Buffer[(32, 4096, 512), "float16"],
              W_UK: Buffer[(128, 512, 128), "float16"],
              Output: Buffer[(32, 128, 128), "float16"]):
    with T.Kernel(128, 32, threads=256) as (hid, bid):
        Q_smem = T.alloc_shared([64, 128], "float16")
        cKV_smem = T.alloc_shared([64, 512], "float16")
        UK_smem = T.alloc_shared([512, 128], "float16")
        K_tile = T.alloc_shared([64, 128], "float16")
        S_local = T.alloc_fragment([64, 64], "float32")
        O_local = T.alloc_fragment([64, 128], "float32")

        T.copy(Q[bid, hid, :], Q_smem)
        T.copy(W_UK[hid, :, :], UK_smem)

        for j in T.serial(64):
            T.copy(c_KV[bid, j*64:(j+1)*64, :], cKV_smem)
            T.gemm(cKV_smem, UK_smem, K_tile)
            T.gemm(Q_smem, K_tile, S_local, transpose_B=True)
            T.online_softmax(S_local, O_local, ...)

=== TensorIR (Lowered) ===
@T.prim_func
def flash_mla(A: T.handle, B: T.handle, ...):
    # ... 降低后的标准 TensorIR 表示 ...
    pass
```

### 11.8.3 可视化调试

```python
import tilelang.debug as debug

# 生成计算图可视化
debug.visualize_computation_graph(my_kernel, output_file="kernel_graph.html")

# 生成内存访问模式可视化
debug.visualize_memory_access(my_kernel, output_file="memory_access.html")

# 生成线程映射可视化
debug.visualize_thread_mapping(my_kernel, output_file="thread_map.html")
```

### 11.8.4 常见 IR 级别的问题诊断

```python
# 问题 1: Bank Conflict
# 诊断：检查共享内存访问模式
"""
有问题的 IR:
  for i in T.serial(32):
    A_smem[i, tx % 32] = ...  # 连续线程访问同一 bank

修复后:
  for i in T.serial(32):
    A_smem[i, (tx + i * 3) % 32] = ...  # 交错访问
"""

# 问题 2: 内存合并失败
# 诊断：检查全局内存访问模式
"""
有问题的 IR:
  for i in T.serial(128):
    A[tx, i] = ...  # 跨步访问

修复后:
  for i in T.serial(128):
    A[i, tx] = ...  # 连续访问
"""

# 问题 3: 寄存器溢出
# 诊断：检查寄存器使用量
"""
使用 debug 工具:
  debug.check_register_usage(my_kernel)
  # 输出: Warning: 128 registers used, may cause spilling
  # 建议: 减小 tile 大小或使用 __launch_bounds__
"""
```

---

## 11.9 高级 IR 特性

### 11.9.1 自定义 Block 属性

```python
@T.prim_func
def kernel_with_block_attrs(A, B, C):
    with T.Kernel(...) as (bx, by):
        with T.block("compute"):
            # 自定义属性
            T.block_attr({"scheduler": "tilelang", "priority": 1})

            # 计算逻辑
            for i in T.serial(TILE_M):
                for j in T.serial(TILE_N):
                    C[i, j] = A[i, j] + B[i, j]
```

### 11.9.2 Annotation 与 Hint

```python
@T.prim_func
def kernel_with_hints(A, B, C):
    with T.Kernel(...) as (bx, by):
        # 提示编译器进行特定优化
        T.annotate("pragma_unroll", 4)
        T.annotate("pragma_vectorize", True)

        for k in T.serial(K // TILE_K):
            # 提示：这个循环可以被流水线化
            T.annotate("pipeline_stage", 2)
            T.copy(A_tile, A_smem)
```

### 11.9.3 类型系统扩展

```python
# TileLang 扩展的类型系统

# 1. 向量类型
float16x8 = T.vector("float16", 8)  # 8 个 float16 的向量

# 2. 矩阵片段类型 (用于 Tensor Core)
mma_fragment = T.mma_fragment("float16", 16, 16, 16)

# 3. 条件类型
pred_t = T.PrimType("bool")  # 谓词类型，用于 masked 操作
```

---

## 11.10 源码走读：Lowering 实现

### 11.10.1 TileLangLowerTileOp 实现

```python
# 文件: tilelang/ir/transform/lower_tile_op.py

class TileLangLowerTileOp(tvm.tir.PrimFuncPass):
    """
    将 TileLang 的高层 tile 操作降低到标准 TensorIR
    """

    def transform_function(self, func, mod, ctx):
        # 1. 分析 T.Kernel 块
        kernel_blocks = self._find_kernel_blocks(func)

        # 2. 将 T.Kernel 映射到 For 循环
        for block in kernel_blocks:
            grid_loops = self._create_grid_loops(block)
            thread_binding = self._create_thread_binding(block)

            # 3. 重写计算体
            new_body = self._rewrite_body(block.body, thread_binding)

            # 4. 替换原始 block
            func = self._replace_block(func, block, grid_loops, new_body)

        return func

    def _create_grid_loops(self, kernel_block):
        """创建 grid 维度的 For 循环"""
        grid_dims = kernel_block.grid_dims
        loops = []
        for i, dim in enumerate(grid_dims):
            loop = tvm.tir.For(
                loop_var=tvm.tir.Var(f"bx{i}", "int32"),
                min_val=0,
                extent=dim,
                kind=tvm.tir.ForKind.SERIAL,  # 后续会绑定到 blockIdx
                body=None,
            )
            loops.append(loop)
        return loops

    def _rewrite_body(self, body, thread_binding):
        """重写计算体，添加线程绑定"""
        # 1. 查找所有 serial 循环
        serial_loops = self._find_serial_loops(body)

        # 2. 根据 tile 大小分配线程
        for loop in serial_loops:
            if self._should_bind_to_thread(loop):
                loop = self._bind_to_thread(loop, thread_binding)

        # 3. 处理共享内存分配
        body = self._lower_shared_memory(body)

        # 4. 处理 copy 操作
        body = self._lower_copy_operations(body)

        return body
```

### 11.10.2 TileLangInferLayout 实现

```python
# 文件: tilelang/ir/transform/infer_layout.py

class TileLangInferLayout(tvm.tir.PrimFuncPass):
    """
    推断和优化内存布局
    """

    def transform_function(self, func, mod, ctx):
        # 1. 收集所有共享内存 buffer
        shared_buffers = self._collect_shared_buffers(func)

        # 2. 分析访问模式
        for buf in shared_buffers:
            access_pattern = self._analyze_access_pattern(func, buf)

            # 3. 选择最优布局
            layout = self._choose_layout(buf, access_pattern)

            # 4. 应用布局变换
            if layout.needs_swizzle:
                func = self._apply_swizzle(func, buf, layout)
            if layout.needs_padding:
                func = self._apply_padding(func, buf, layout)

        return func

    def _choose_layout(self, buffer, access_pattern):
        """
        选择最优的内存布局
        考虑因素：
        1. Bank conflict 避免
        2. 合并访问优化
        3. 对齐要求
        """
        if access_pattern.is_column_major:
            return SwizzleLayout(offset=3, stride=0, granularity=4)
        elif access_pattern.has_bank_conflict:
            return PaddingLayout(padding=1)
        else:
            return DefaultLayout()
```

---

## 11.11 实验与验证

### 11.11.1 IR Dump 实验

```python
import tilelang
from tilelang import T, ir

# 实验 1: 观察 TileLang IR 的结构
@T.prim_func
def simple_add(A: T.Buffer[(1024,), "float32"],
               B: T.Buffer[(1024,), "float32"],
               C: T.Buffer[(1024,), "float32"]):
    with T.Kernel(4, threads=256) as (bx,):
        for i in T.serial(256):
            C[bx * 256 + i] = A[bx * 256 + i] + B[bx * 256 + i]

# 查看 IR
print("TileLang IR:")
print(ir.dump_tilelang(simple_add))

print("\nTensorIR (Lowered):")
lowered = ir.lower_tilelang(simple_add)
print(lowered)

# 实验 2: 观察 GEMM 的 IR 变换
@T.prim_func
def simple_gemm(A: T.Buffer[(128, 128), "float16"],
                B: T.Buffer[(128, 128), "float16"],
                C: T.Buffer[(128, 128), "float32"]):
    with T.Kernel(1, threads=256) as (bx,):
        A_smem = T.alloc_shared([128, 128], "float16")
        B_smem = T.alloc_shared([128, 128], "float16")
        C_reg = T.alloc_fragment([128, 128], "float32")

        T.copy(A, A_smem)
        T.copy(B, B_smem)
        T.gemm(A_smem, B_smem, C_reg)
        T.copy(C_reg, C)

print("\nGEMM TileLang IR:")
print(ir.dump_tilelang(simple_gemm))
```

### 11.11.2 验证 Lowering 正确性

```python
import numpy as np

def verify_lowering(tilelang_func, reference_func, inputs):
    """
    验证 Lowering 后的功能正确性
    """
    # 1. 运行 TileLang 版本
    tilelang_output = tilelang_func(*inputs)

    # 2. 运行参考实现
    reference_output = reference_func(*inputs)

    # 3. 比较结果
    np.testing.assert_allclose(
        tilelang_output, reference_output,
        rtol=1e-3, atol=1e-3,
        err_msg="Lowering verification failed!"
    )
    print("Lowering verification passed!")

# 测试
A = np.random.randn(128, 128).astype(np.float16)
B = np.random.randn(128, 128).astype(np.float16)

verify_lowering(
    tilelang_func=tilelang.compile(simple_gemm),
    reference_func=lambda a, b: a @ b,
    inputs=[A, B]
)
```

---

## 11.12 总结

### 核心要点

<div data-component="TileLangIRToTensorIRFlow"></div>

1. **TileLang IR** 是基于 TVM TensorIR 的领域特定扩展，以 tile 为核心抽象
2. **Buffer/Region** 抽象提供了内存区域的高层视图，支持多级内存管理
3. **Block/Loop** 结构将计算逻辑与线程映射分离，提高代码可读性
4. **Lowering 过程**通过一系列 Pass 将高层 IR 降低到标准 TensorIR
5. **与 TVM 协同**：TileLang 可以无缝嵌入 TVM 的编译管线
6. **调试工具**：IR dump 和可视化工具帮助理解编译过程

### IR 映射速查表

```
TileLang → TensorIR 映射速查：

T.Buffer          → tir.Buffer / tir.match_buffer
T.Region          → tir.BufferRegion (slice notation)
T.Kernel          → tir.For (grid) + tir.block
T.alloc_shared    → tir.alloc_buffer(scope="shared")
T.alloc_fragment  → tir.alloc_buffer(scope="local")
T.copy            → tir.block (element-wise copy)
T.gemm            → tir.block (nested loop matmul)
T.fill            → tir.block (init block)
T.serial          → tir.ForKind.SERIAL
T.parallel        → tir.ForKind.PARALLEL
T.vectorize       → tir.ForKind.VECTORIZED
T.unroll          → tir.ForKind.UNROLLED
T.sync_threads    → tir.Evaluate(tvm.tir.call_extern(...))
```

---

## 练习

### 基础练习

1. **IR 观察**：编写一个简单的 TileLang kernel，使用 `ir.dump_tilelang()` 和 `ir.lower_tilelang()` 分别查看 IR，并分析两者的关系。

2. **Buffer 理解**：解释 `T.alloc_shared` 和 `T.alloc_fragment` 在生成的 TensorIR 中有什么区别。

3. **映射练习**：将以下 TileLang 代码手动翻译为 TensorIR：
   ```python
   with T.Kernel(4, 8, threads=128) as (bx, by):
       A_smem = T.alloc_shared([32, 32], "float16")
       T.copy(A[bx*32:(bx+1)*32, by*32:(by+1)*32], A_smem)
       for i in T.serial(32):
           C[bx*32+i, by*32+i] = A_smem[i, i]
   ```

### 进阶练习

4. **Pass 实现**：实现一个简单的 IR Pass，将所有的 `T.serial` 循环替换为 `T.unroll`（当循环次数 ≤ 4 时）。

5. **布局分析**：编写一个函数，分析给定 kernel 的共享内存访问模式，检测是否存在 bank conflict。

6. **调试工具**：实现一个 IR 可视化工具，将 TileLang IR 渲染为 SVG 图形，显示数据流和内存层次。

---

## 思考题

1. **设计思考**：为什么 TileLang 选择扩展 TensorIR 而不是设计一个全新的 IR？这种选择有什么优劣？

2. **抽象层次**：TileLang 的抽象层次是否合适？如果要支持更多的硬件（如 NPU），IR 层面需要哪些改变？

3. **编译器设计**：TileLang 的 Pass 设计遵循了什么原则？如果要添加一个新的硬件后端，需要修改哪些 Pass？

---

## 11.13 IR 变换的数学基础

### 11.13.1 多面体模型

TileLang IR 的循环变换基于多面体模型（Polyhedral Model）：

```
多面体模型基本概念：

迭代空间: 所有循环索引的集合
  例: for i in range(M): for j in range(N)
  迭代空间 = {(i, j) | 0 ≤ i < M, 0 ≤ j < N}

访问函数: 描述内存访问的映射
  例: A[i, j] → 访问函数 f(i, j) = (i, j)

依赖关系: 描述计算之间的依赖
  例: S1: C[i] = A[i] + B[i]
      S2: D[i] = C[i] * 2
  S2 依赖 S1 (RAW - Read After Write)
```

### 11.13.2 循环变换操作

```python
# 循环变换操作

# 1. 循环交换 (Loop Interchange)
"""
原始:
  for i in range(M):
    for j in range(N):
      A[i][j] = ...

交换后:
  for j in range(N):
    for i in range(M):
      A[i][j] = ...

影响: 改变内存访问模式
"""

# 2. 循环分块 (Loop Tiling)
"""
原始:
  for i in range(M):
    for j in range(N):
      A[i][j] = ...

分块后:
  for i_outer in range(M // TILE):
    for j_outer in range(N // TILE):
      for i_inner in range(TILE):
        for j_inner in range(TILE):
          i = i_outer * TILE + i_inner
          j = j_outer * TILE + j_inner
          A[i][j] = ...

影响: 提高数据局部性
"""

# 3. 循环融合 (Loop Fusion)
"""
原始:
  for i in range(N):
    B[i] = A[i] * 2
  for i in range(N):
    C[i] = B[i] + 1

融合后:
  for i in range(N):
    B[i] = A[i] * 2
    C[i] = B[i] + 1

影响: 减少循环开销，提高缓存利用率
"""
```

### 11.13.3 仿射变换矩阵

```python
# 使用仿射变换矩阵表示循环变换

def affine_transform(iteration_space, transform_matrix):
    """
    应用仿射变换

    transform_matrix: 变换矩阵 T
    新的迭代空间 = T * 原始迭代空间
    """
    import numpy as np

    # 原始迭代空间
    original = np.array(iteration_space)

    # 应用变换
    transformed = transform_matrix @ original

    return transformed

# 示例: 循环交换
# 原始: (i, j)
# 交换后: (j, i)
# 变换矩阵: [[0, 1], [1, 0]]
T_swap = np.array([[0, 1], [1, 0]])

# 示例: 循环分块
# 原始: (i, j)
# 分块后: (i_outer, j_outer, i_inner, j_inner)
# 需要更复杂的变换
```

---

## 11.14 高级 IR 特性详解

### 11.14.1 动态形状支持

```python
# TileLang 支持动态形状

@T.prim_func
def dynamic_shape_kernel(
    A: T.Buffer[(T.Var("M"), T.Var("N")), "float16"],
    B: T.Buffer[(T.Var("M"), T.Var("N")), "float16"],
    C: T.Buffer[(T.Var("M"), T.Var("N")), "float16"],
    M: T.int32,
    N: T.int32,
):
    """
    支持动态 M 和 N 的 kernel
    """
    with T.Kernel(dynamic=True) as (bx, by):
        # 动态计算 grid 大小
        grid_m = (M + TILE_M - 1) // TILE_M
        grid_n = (N + TILE_N - 1) // TILE_N

        # 动态边界检查
        for i in T.serial(TILE_M):
            for j in T.serial(TILE_N):
                row = bx * TILE_M + i
                col = by * TILE_N + j
                if row < M and col < N:
                    C[row, col] = A[row, col] + B[row, col]
```

### 11.14.2 条件执行

```python
@T.prim_func
def conditional_kernel(A, B, C, use_relu: T.bool):
    """
    支持条件执行的 kernel
    """
    with T.Kernel(grid_m, threads=256) as (bx,):
        tx = T.get_thread_id()

        val = A[bx * 256 + tx] + B[bx * 256 + tx]

        # 条件执行
        if use_relu:
            val = T.max(val, 0.0)

        C[bx * 256 + tx] = val
```

### 11.14.3 自定义 Intrinsic

```python
# 注册自定义 intrinsic

@T.register_intrinsic("my_custom_op")
def my_custom_op(a, b, c):
    """
    自定义操作的定义
    """
    return T.call_extern("float32", "my_custom_op", a, b, c)

# 使用自定义 intrinsic
@T.prim_func
def kernel_with_custom_op(A, B, C):
    with T.Kernel(...) as (bx,):
        for i in T.serial(N):
            C[i] = my_custom_op(A[i], B[i], 1.0)
```

---

## 11.15 IR 优化的理论分析

### 11.15.1 优化效果量化

```python
def analyze_optimization_impact(original_ir, optimized_ir):
    """
    分析 IR 优化的效果
    """
    metrics = {}

    # 1. 操作数统计
    original_ops = count_operations(original_ir)
    optimized_ops = count_operations(optimized_ir)
    metrics["op_reduction"] = (original_ops - optimized_ops) / original_ops

    # 2. 内存访问统计
    original_mem = count_memory_accesses(original_ir)
    optimized_mem = count_memory_accesses(optimized_ir)
    metrics["mem_reduction"] = (original_mem - optimized_mem) / original_mem

    # 3. 循环嵌套深度
    original_depth = max_loop_depth(original_ir)
    optimized_depth = max_loop_depth(optimized_ir)
    metrics["loop_depth_change"] = optimized_depth - original_depth

    # 4. 并行度
    original_parallelism = compute_parallelism(original_ir)
    optimized_parallelism = compute_parallelism(optimized_ir)
    metrics["parallelism_improvement"] = (
        optimized_parallelism / original_parallelism
    )

    return metrics
```

### 11.15.2 最优性分析

```
IR 优化的最优性分析：

1. 循环分块的最优性
   - 问题: 选择最优的 tile 大小
   - 目标: 最小化 cache miss
   - 约束: 共享内存大小限制

2. 循环交换的最优性
   - 问题: 选择最优的循环顺序
   - 目标: 最大化内存访问局部性
   - 约束: 数据依赖关系

3. 循环融合的最优性
   - 问题: 决定哪些循环应该融合
   - 目标: 最小化数据传输
   - 约束: 寄存器压力
```

### 11.15.3 复杂度分析

```python
def complexity_analysis(ir):
    """
    分析 IR 的计算和内存复杂度
    """
    # 计算复杂度 (FLOPS)
    compute_complexity = analyze_compute_complexity(ir)
    # 例: GEMM 的计算复杂度 O(M * N * K)

    # 内存复杂度 (字节)
    memory_complexity = analyze_memory_complexity(ir)
    # 例: GEMM 的内存复杂度 O(M*K + K*N + M*N)

    # 算术强度 (FLOPS / Byte)
    arithmetic_intensity = compute_complexity / memory_complexity
    # 例: GEMM 的算术强度 = 2*M*N*K / (M*K + K*N + M*N) * 2

    # Roofline 分析
    peak_compute = get_peak_compute()  # TFLOPS
    peak_bandwidth = get_peak_bandwidth()  # GB/s

    if arithmetic_intensity < peak_compute / peak_bandwidth:
        # 带宽受限
        bottleneck = "memory"
        expected_performance = arithmetic_intensity * peak_bandwidth
    else:
        # 计算受限
        bottleneck = "compute"
        expected_performance = peak_compute

    return {
        "compute_complexity": compute_complexity,
        "memory_complexity": memory_complexity,
        "arithmetic_intensity": arithmetic_intensity,
        "bottleneck": bottleneck,
        "expected_performance": expected_performance,
    }
```

---

## 11.16 IR Dump 与可视化工具详解

### 11.16.1 完整的 IR Dump 工具

```python
import tilelang
from tilelang import ir
import json
import graphviz

class IRDumper:
    """
    完整的 IR Dump 工具
    支持多种输出格式
    """

    def __init__(self, output_dir="./ir_dump"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def dump_all_stages(self, kernel_func):
        """
        Dump 所有阶段的 IR
        """
        stages = {}

        # 1. TileLang IR
        tilelang_ir = ir.dump_tilelang(kernel_func)
        stages["tilelang_ir"] = tilelang_ir
        self._save_text("01_tilelang_ir.txt", tilelang_ir)

        # 2. TensorIR
        tensorir = ir.lower_tilelang(kernel_func)
        stages["tensorir"] = str(tensorir)
        self._save_text("02_tensorir.txt", str(tensorir))

        # 3. 优化后的 TIR
        tir = ir.optimize(tensorir)
        stages["tir_optimized"] = str(tir)
        self._save_text("03_tir_optimized.txt", str(tir))

        # 4. PTX 代码
        ptx = ir.codegen(tir, target="cuda")
        stages["ptx"] = ptx
        self._save_text("04_ptx.sass", ptx)

        # 5. 生成可视化
        self._generate_visualization(stages)

        return stages

    def _generate_visualization(self, stages):
        """
        生成 IR 可视化
        """
        # 生成数据流图
        dot = graphviz.Digraph(comment='IR Data Flow')

        for stage_name, stage_ir in stages.items():
            dot.node(stage_name, stage_name)

        # 添加转换边
        dot.edge("tilelang_ir", "tensorir", label="Lowering")
        dot.edge("tensorir", "tir_optimized", label="Optimization")
        dot.edge("tir_optimized", "ptx", label="CodeGen")

        dot.render(os.path.join(self.output_dir, "ir_flow"), format="png")
```

### 11.16.2 IR Diff 工具

```python
class IRDiffTool:
    """
    IR Diff 工具
    比较两个 IR 的差异
    """

    def diff(self, ir1, ir2):
        """
        比较两个 IR 的差异
        """
        # 1. 结构差异
        structural_diff = self._structural_diff(ir1, ir2)

        # 2. 语义差异
        semantic_diff = self._semantic_diff(ir1, ir2)

        # 3. 性能差异
        perf_diff = self._performance_diff(ir1, ir2)

        return {
            "structural": structural_diff,
            "semantic": semantic_diff,
            "performance": perf_diff,
        }

    def _structural_diff(self, ir1, ir2):
        """
        比较结构差异
        """
        diff = []

        # 比较循环结构
        loops1 = self._extract_loops(ir1)
        loops2 = self._extract_loops(ir2)

        if len(loops1) != len(loops2):
            diff.append(f"Loop count: {len(loops1)} -> {len(loops2)}")

        # 比较 Block 结构
        blocks1 = self._extract_blocks(ir1)
        blocks2 = self._extract_blocks(ir2)

        if len(blocks1) != len(blocks2):
            diff.append(f"Block count: {len(blocks1)} -> {len(blocks2)}")

        return diff
```

### 11.16.3 性能预估工具

```python
class PerformanceEstimator:
    """
    基于 IR 的性能预估工具
    """

    def estimate(self, ir, target_gpu):
        """
        预估 kernel 的性能
        """
        # 1. 计算 FLOPS
        flops = self._count_flops(ir)

        # 2. 计算内存访问量
        memory_bytes = self._count_memory_accesses(ir)

        # 3. 计算理论带宽和计算峰值
        peak_bandwidth = self._get_peak_bandwidth(target_gpu)  # GB/s
        peak_compute = self._get_peak_compute(target_gpu)  # TFLOPS

        # 4. Roofline 分析
        arithmetic_intensity = flops / memory_bytes  # FLOPS/Byte

        if arithmetic_intensity < peak_compute * 1e3 / peak_bandwidth:
            # 带宽受限
            estimated_time = memory_bytes / (peak_bandwidth * 1e9)  # 秒
            bottleneck = "memory"
        else:
            # 计算受限
            estimated_time = flops / (peak_compute * 1e12)  # 秒
            bottleneck = "compute"

        return {
            "flops": flops,
            "memory_bytes": memory_bytes,
            "arithmetic_intensity": arithmetic_intensity,
            "estimated_time_ms": estimated_time * 1000,
            "bottleneck": bottleneck,
        }
```

---

## 11.17 与其他 IR 系统的对比

### 11.17.1 TileLang IR vs MLIR

```
TileLang IR vs MLIR:

相似之处:
  - 都是多级 IR 系统
  - 都支持 dialect 概念
  - 都有 lowering 过程

不同之处:
  - TileLang 专注于张量计算
  - MLIR 是通用编译器框架
  - TileLang 更高层抽象
  - MLIR 更灵活但更复杂

选择建议:
  - GPU kernel 优化: TileLang
  - 通用编译器: MLIR
  - 硬件综合: MLIR
```

### 11.17.2 TileLang IR vs Triton IR

```
TileLang IR vs Triton IR:

相似之处:
  - 都是 GPU kernel DSL
  - 都有 tile 抽象
  - 都支持自动优化

不同之处:
  - TileLang 基于 TVM 生态
  - Triton 独立的 IR 系统
  - TileLang 更底层控制
  - Triton 更易用

选择建议:
  - 快速开发: Triton
  - 深度优化: TileLang
  - TVM 集成: TileLang
```

### 11.17.3 TileLang IR vs XLA HLO

```
TileLang IR vs XLA HLO:

相似之处:
  - 都是计算图 IR
  - 都支持优化

不同之处:
  - TileLang 面向 kernel 开发
  - XLA HLO 面向模型编译
  - TileLang 更细粒度
  - XLA HLO 更高层

选择建议:
  - 自定义 kernel: TileLang
  - 模型编译: XLA
  - JAX/TensorFlow: XLA
```

---

## 11.18 实战案例：IR 分析

### 11.18.1 分析 FlashMLA 的 IR

```python
def analyze_flash_mla_ir():
    """
    分析 FlashMLA 的 IR 变换过程
    """
    from tilelang.flash_mla import flash_mla_kernel

    dumper = IRDumper(output_dir="./flash_mla_ir")

    # Dump 所有阶段
    stages = dumper.dump_all_stages(flash_mla_kernel)

    # 分析每个阶段的特点
    print("=== IR 分析 ===")

    # 1. TileLang IR 特点
    print("\n1. TileLang IR:")
    print("  - 使用 T.Kernel 定义 GPU kernel")
    print("  - 使用 T.alloc_shared 分配共享内存")
    print("  - 使用 T.gemm 进行矩阵乘法")
    print("  - 使用 T.copy 进行数据搬运")

    # 2. TensorIR 特点
    print("\n2. TensorIR:")
    print("  - 转换为标准的 Buffer/Block/Loop 结构")
    print("  - 共享内存映射为 alloc_buffer(scope='shared')")
    print("  - GEMM 展开为嵌套循环")

    # 3. TIR 优化后的特点
    print("\n3. TIR (优化后):")
    print("  - 循环展开和向量化")
    print("  - 常量折叠")
    print("  - 内存访问优化")

    # 4. PTX 特点
    print("\n4. PTX 代码:")
    print("  - 使用 GPU 指令")
    print("  - 共享内存声明")
    print("  - 线程同步指令")

    return stages
```

### 11.18.2 IR 优化效果验证

```python
def verify_optimization_effects():
    """
    验证 IR 优化的效果
    """
    import torch

    # 创建测试数据
    M, N, K = 1024, 1024, 1024
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")

    # 1. 未优化的 kernel
    @T.prim_func
    def unoptimized_gemm(A, B, C):
        with T.Kernel(1, threads=256) as (bx,):
            for i in T.serial(M):
                for j in T.serial(N):
                    for k in T.serial(K):
                        C[i, j] += A[i, k] * B[k, j]

    # 2. 优化后的 kernel
    @T.prim_func
    def optimized_gemm(A, B, C):
        with T.Kernel(grid_m, grid_n, threads=256) as (bx, by):
            A_smem = T.alloc_shared([TILE_M, TILE_K], "float16")
            B_smem = T.alloc_shared([TILE_K, TILE_N], "float16")
            C_reg = T.alloc_fragment([TILE_M, TILE_N], "float32")

            for k in range(K // TILE_K):
                T.copy(A[bx*TILE_M:..., k*TILE_K:...], A_smem)
                T.copy(B[k*TILE_K:..., by*TILE_N:...], B_smem)
                T.gemm(A_smem, B_smem, C_reg)

            T.copy(C_reg, C[bx*TILE_M:..., by*TILE_N:...])

    # 3. 性能对比
    compiled_unopt = tilelang.compile(unoptimized_gemm)
    compiled_opt = tilelang.compile(optimized_gemm)

    C_unopt = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    C_opt = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    # Benchmark
    import time

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        compiled_unopt(A, B, C_unopt)
    torch.cuda.synchronize()
    unopt_time = (time.time() - start) / 10

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        compiled_opt(A, B, C_opt)
    torch.cuda.synchronize()
    opt_time = (time.time() - start) / 10

    print(f"未优化: {unopt_time*1000:.2f} ms")
    print(f"优化后: {opt_time*1000:.2f} ms")
    print(f"加速比: {unopt_time/opt_time:.2f}x")

    # 验证正确性
    torch.testing.assert_close(C_unopt, C_opt, rtol=1e-3, atol=1e-3)
    print("正确性验证通过!")
```

---

## 11.19 IR 级别的调试技巧

### 11.19.1 IR Dump 的实用工作流

```python
def practical_ir_debug_workflow(kernel_func, test_inputs):
    """
    实用的 IR 调试工作流

    步骤:
    1. 检查 TileLang IR 确认逻辑正确
    2. 检查 TensorIR 确认 lowering 正确
    3. 检查 TIR 确认优化正确
    4. 对比各阶段 IR 找出问题
    """
    import tilelang
    from tilelang import ir

    # Step 1: TileLang IR
    print("=" * 60)
    print("Step 1: TileLang IR")
    print("=" * 60)
    tilelang_ir = ir.dump_tilelang(kernel_func)
    print(tilelang_ir)

    # 检查点:
    # - Buffer 声明是否正确
    # - T.Kernel 配置是否合理
    # - 计算逻辑是否正确

    # Step 2: TensorIR
    print("\n" + "=" * 60)
    print("Step 2: TensorIR (Lowered)")
    print("=" * 60)
    tensorir = ir.lower_tilelang(kernel_func)
    print(tensorir)

    # 检查点:
    # - Buffer 映射是否正确
    # - Block/Loop 结构是否合理
    # - 共享内存分配是否正确

    # Step 3: 运行验证
    print("\n" + "=" * 60)
    print("Step 3: 运行验证")
    print("=" * 60)
    compiled = tilelang.compile(kernel_func, target="cuda")
    output = compiled(*test_inputs)
    print(f"输出形状: {output.shape}")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    print(f"包含 NaN: {torch.isnan(output).any()}")
```

### 11.19.2 常见 IR 问题诊断

```python
def diagnose_ir_issues(kernel_func):
    """
    诊断常见 IR 问题

    常见问题:
    1. Bank conflict: 共享内存访问模式不当
    2. 内存合并失败: 全局内存访问不连续
    3. 寄存器溢出: tile 太大
    4. 同步缺失: 缺少 T.syncthreads()
    """
    from tilelang import ir

    issues = []

    # 获取 TensorIR
    tensorir = ir.lower_tilelang(kernel_func)
    tir_str = str(tensorir)

    # 检查共享内存大小
    if "alloc_buffer" in tir_str and "scope='shared'" in tir_str:
        # 提取共享内存大小
        import re
        shared_allocs = re.findall(
            r"alloc_buffer\((\d+),\s*(\d+)\).*scope='shared'",
            tir_str
        )
        total_shared = sum(int(w) * int(h) * 2 for w, h in shared_allocs)  # FP16
        if total_shared > 48 * 1024:
            issues.append({
                "type": "shared_memory_exceeded",
                "description": f"共享内存使用 {total_shared/1024:.1f} KB，超过 48 KB 限制",
                "fix": "减小 tile 大小或使用 FP8 量化",
            })

    # 检查循环结构
    if "ForKind.SERIAL" in tir_str:
        serial_count = tir_str.count("ForKind.SERIAL")
        if serial_count > 6:
            issues.append({
                "type": "deep_loop_nesting",
                "description": f"循环嵌套深度 {serial_count}，可能导致寄存器压力过大",
                "fix": "考虑循环融合或减少嵌套",
            })

    return issues
```

### 11.19.3 IR 性能预估

```python
def estimate_kernel_performance(kernel_func, target_gpu="a100"):
    """
    基于 IR 预估 kernel 性能

    不需要实际运行 kernel，通过分析 IR 估算:
    1. 计算 FLOPS
    2. 内存访问量
    3. 理论执行时间
    """
    from tilelang import ir

    # GPU 规格
    gpu_specs = {
        "a100": {
            "fp16_tflops": 312,
            "bandwidth_gbs": 2039,
            "sm_count": 108,
            "shared_mem_per_sm": 164 * 1024,
        },
        "h100": {
            "fp16_tflops": 989,
            "bandwidth_gbs": 3350,
            "sm_count": 132,
            "shared_mem_per_sm": 228 * 1024,
        },
    }

    spec = gpu_specs.get(target_gpu, gpu_specs["a100"])

    # 分析 IR
    tensorir = ir.lower_tilelang(kernel_func)
    tir_str = str(tensorir)

    # 估算 FLOPS (简化)
    # 查找 GEMM 操作
    gemm_count = tir_str.count("gemm")
    flops_per_gemm = 2 * 128 * 128 * 32  # 假设 tile 大小
    total_flops = gemm_count * flops_per_gemm

    # 估算内存访问 (简化)
    copy_count = tir_str.count("copy")
    bytes_per_copy = 128 * 32 * 2  # 假设 FP16
    total_bytes = copy_count * bytes_per_copy

    # Roofline 分析
    arithmetic_intensity = total_flops / total_bytes if total_bytes > 0 else float('inf')
    ridge_point = spec["fp16_tflops"] * 1000 / spec["bandwidth_gbs"]

    if arithmetic_intensity < ridge_point:
        bottleneck = "memory"
        estimated_time_us = total_bytes / (spec["bandwidth_gbs"] * 1e9) * 1e6
    else:
        bottleneck = "compute"
        estimated_time_us = total_flops / (spec["fp16_tflops"] * 1e12) * 1e6

    return {
        "total_flops": total_flops,
        "total_bytes": total_bytes,
        "arithmetic_intensity": arithmetic_intensity,
        "bottleneck": bottleneck,
        "estimated_time_us": estimated_time_us,
    }
```

---

## 11.20 IR 与硬件映射的深入分析

### 11.20.1 GPU 内存层次的 IR 表示

```
GPU 内存层次在 IR 中的表示:

┌─────────────────────────────────────────────────────┐
│  内存层次        │  IR 表示                          │
├──────────────────┼──────────────────────────────────┤
│  寄存器           │  alloc_buffer(scope="local")      │
│  共享内存         │  alloc_buffer(scope="shared")     │
│  L1 Cache        │  (自动管理，IR 中不可见)           │
│  L2 Cache        │  (自动管理，IR 中不可见)           │
│  全局内存         │  Buffer (默认 scope)              │
│  常量内存         │  alloc_buffer(scope="global.const")│
└──────────────────┴──────────────────────────────────┘

数据流在 IR 中的表示:
  Global → Shared:  T.copy(A_global, A_shared)
  Shared → Local:   T.copy(A_shared, A_local) 或 T.gemm
  Local → Shared:   T.copy(A_local, A_shared)
  Shared → Global:  T.copy(A_shared, A_global)
```

### 11.20.2 IR 优化 Pass 的硬件感知

```python
"""
IR 优化 Pass 的硬件感知

不同的硬件需要不同的优化策略:

NVIDIA GPU:
  - 共享内存 bank conflict: 32 bank, 4 字节 stride
  - Warp 大小: 32 线程
  - Tensor Core: m16n8k16 (FP16)
  - 异步拷贝: cp.async

AMD GPU:
  - 共享内存 bank conflict: 32 bank, 4 字节 stride
  - Wavefront 大小: 64 线程
  - MFMA: 16x16x16 (FP16)
  - 异步拷贝: buffer_load

TileLang 的硬件感知优化:
  1. 自动选择合适的 tile 大小
  2. 自动处理 bank conflict (swizzle/padding)
  3. 自动选择正确的 Tensor Core/MFMA 指令
  4. 自动生成异步拷贝指令
"""

def apply_hardware_aware_optimization(ir, target):
    """
    应用硬件感知的 IR 优化
    """
    if target == "cuda":
        # NVIDIA 特定优化
        ir = apply_swizzle_for_bank_conflict(ir)
        ir = apply_cp_async(ir)
        ir = apply_warp_specialization(ir)
    elif target == "rocm":
        # AMD 特定优化
        ir = apply_wave64_optimization(ir)
        ir = apply_mfma_layout(ir)

    return ir
```

---

## 11.21 IR 扩展与自定义

### 11.21.1 自定义 IR 操作

```python
# TileLang 允许用户定义自定义 IR 操作

@T.register_intrinsic("my_custom_reduce")
def my_custom_reduce(x, axis):
    """
    自定义归约操作

    可以映射到特定的硬件指令
    """
    return T.call_extern(
        "float32",
        "my_custom_reduce",
        x,
        axis,
        # 可以传递额外的属性
    )

# 使用自定义操作
@T.prim_func
def kernel_with_custom_reduce(A, B):
    with T.Kernel(...) as (bx,):
        local = T.alloc_fragment((128,), "float32")
        # ... 加载数据 ...
        result = my_custom_reduce(local, axis=0)
        B[bx] = result
```

### 11.21.2 IR Pass 插件

```python
class MyCustomPass(tvm.tir.PrimFuncPass):
    """
    自定义 IR Pass

    可以在 lowering 流程中插入自定义优化
    """

    def transform_function(self, func, mod, ctx):
        # 分析 IR
        analysis = self._analyze(func)

        # 应用变换
        if self._should_transform(analysis):
            func = self._apply_transform(func)

        return func

    def _analyze(self, func):
        """分析 IR 特征"""
        return {
            "has_bank_conflict": self._check_bank_conflict(func),
            "register_pressure": self._estimate_register_pressure(func),
            "loop_depth": self._get_loop_depth(func),
        }

# 注册 Pass
@tvm.transform.register_pass("my_custom_pass", level=100)
def my_custom_pass():
    return MyCustomPass()
```

这个自定义 IR Pass 展示了 TileLang 的可扩展性。通过继承 `tvm.tir.PrimFuncPass` 并实现 `transform_function` 方法，开发者可以创建自己的 IR 优化 Pass，在 Lowering 流程中插入自定义优化逻辑。接口中的 `_analyze` 方法可用于分析 IR 特征（如 bank conflict、寄存器压力、循环深度等），然后根据分析结果决定是否应用变换。通过 `@tvm.transform.register_pass` 装饰器可以将自定义 Pass 注册到 TVM 的 Pass 管理器中，并指定其执行优先级。这种插件化的设计使得 TileLang 能够灵活地适应不同的硬件后端的优化需求。

---

## 扩展阅读

1. **TVM TensorIR 论文**：理解 TensorIR 的设计理念和形式化定义
2. **TVM 文档**：深入学习 TVM 的编译管线和 Pass 管理
3. **MLIR 文档**：了解 MLIR 的多级 IR 设计思想，与 TileLang 的设计理念有相似之处
4. **Halide 语言**：TileLang 的调度思想部分源于 Halide 的计算-调度分离
5. **Polyhedral Compilation**：多面体编译模型，理解循环变换的理论基础
6. **TVM Pass Infrastructure**：深入了解 TVM 的 Pass 管理框架

---

## 下一章预告

> **Chapter 12: TVM/Relax 编译管线集成**
>
> 在理解了 TileLang IR 与 TensorIR 的关系之后，下一章我们将学习如何将 TileLang kernel 完整地集成到 TVM Relax 编译管线中，实现从 Python 模型到优化机器码的端到端编译。我们将涵盖：
>
> - TVM Relax 框架概述
> - TileLang kernel 嵌入 Relax 计算图
> - 算子注册与调度绑定
> - 端到端编译流程
> - Relax VM 执行
> - 与 torch.compile 集成

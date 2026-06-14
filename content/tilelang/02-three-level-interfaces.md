---
title: "Chapter 2: 三级编程接口——Beginner/Developer/Expert"
description: "深入理解 TileLang 的三级编程接口设计：Beginner（装饰器驱动，零调度知识）、Developer（显式内存管理与 Pipeline 注解）、Expert（Thread 级原语与 Layout 手动控制），掌握每级接口的适用场景与性能特征。"
updated: "2025-06-11"
---

# Chapter 2: 三级编程接口——Beginner/Developer/Expert

<div data-component="ThreeLevelInterfaceComparison"></div>

> [!NOTE]
> **学习目标**
>
> - 理解 TileLang 三级编程接口的设计理念
> - 掌握 Beginner 级接口：装饰器驱动，零调度知识
> - 掌握 Developer 级接口：显式内存管理与 Pipeline 注解
> - 掌握 Expert 级接口：Thread Binding 与 Layout 手动控制
> - 对比三级接口的适用场景与性能特征

---

## 1. 三级接口设计理念

### 1.1 为什么需要三级接口？

在 GPU 编程领域，存在一个根本性的矛盾：**易用性与性能的权衡**。

```
性能
 ↑
 │  ┌─────────────────────────────────────┐
 │  │ CUDA (完全手动控制)                  │
 │  │ - 线程级控制                         │
 │  │ - 手动内存管理                       │
 │  │ - 极致性能                           │
 │  └─────────────────────────────────────┘
 │           ↑
 │  ┌─────────────────────────────────────┐
 │  │ TileLang Expert (Thread Binding)     │
 │  │ - Warp 级控制                        │
 │  │ - 手动 Layout                        │
 │  │ - 接近 CUDA 性能                     │
 │  └─────────────────────────────────────┘
 │           ↑
 │  ┌─────────────────────────────────────┐
 │  │ TileLang Developer (显式内存)        │
 │  │ - Tile 级控制                        │
 │  │ - 显式内存分配                       │
 │  │ - 良好性能                           │
 │  └─────────────────────────────────────┘
 │           ↑
 │  ┌─────────────────────────────────────┐
 │  │ TileLang Beginner (装饰器驱动)       │
 │  │ - 算子级描述                         │
 │  │ - 自动调度                           │
 │  │ - 快速开发                           │
 │  └─────────────────────────────────────┘
 │           ↑
 │  ┌─────────────────────────────────────┐
 │  │ PyTorch (算子调用)                   │
 │  │ - 最高易用性                         │
 │  │ - 性能受限                           │
 │  └─────────────────────────────────────┘
 └──────────────────────────────────────────→ 易用性
```

以上图表直观展示了 TileLang 三级接口在性能与易用性之间的定位关系。在 GPU 编程领域，性能和易用性始终是一对核心矛盾。完全手动控制（如 CUDA）可以达到极致性能，但学习成本极高；而高级框架（如 PyTorch）虽然易于使用，但性能往往受限于框架的通用调度策略。TileLang 的三级接口设计巧妙地在两者之间搭建了一座桥梁，让不同层次的用户都能找到适合自己的切入点。Beginner 级通过装饰器驱动的零调度知识方案，让用户只需要描述"计算什么"；Developer 级引入显式内存管理，让用户可以控制"数据放在哪里"；Expert 级则提供 Thread Binding 和 Layout 手动控制，让用户可以精确控制"谁来计算什么"。这种渐进式的设计理念是 TileLang 架构的核心创新之一。

### 1.2 三级接口的设计目标

TileLang 的三级接口设计遵循以下原则：

| 级别 | 目标用户 | 设计目标 | 典型场景 |
|:---|:---|:---|:---|
| **Beginner** | 算法研究员 | 零调度知识，快速原型 | 论文复现，算法验证 |
| **Developer** | 算子工程师 | 显式控制，平衡性能 | 生产算子开发 |
| **Expert** | 硬核工程师 | 极致性能，完全控制 | 高性能库开发 |

上表详细列出了三级接口各自的目标用户群体、设计目标和典型应用场景。Beginner 级主要面向算法研究员，他们通常需要快速验证论文中的新算法，而不希望花大量时间在 GPU 编程细节上。Developer 级面向算子工程师，他们需要在性能和开发效率之间找到平衡点，既要满足生产环境的性能要求，又不能过度增加开发复杂度。Expert 级面向硬核工程师和高性能库开发者，他们需要挖掘硬件的极致性能，愿意为此投入更多的开发时间来手动控制每一个细节。这三个层次的设计体现了"渐进式优化"的思想：用户可以从 Beginner 级开始快速验证想法，当性能不满足需求时逐步升级到 Developer 级，最终在必要时使用 Expert 级进行极致优化。

### 1.3 三级接口的关系

```
┌─────────────────────────────────────────────────────────────┐
│                    Beginner 级接口                           │
│                    @T.prim_func                             │
│                    + 自动调度                                │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                 Developer 级接口                        │ │
│  │                 T.alloc_shared                          │ │
│  │                 T.alloc_fragment                        │ │
│  │                 T.pipelined                             │ │
│  │  ┌──────────────────────────────────────────────────┐ │ │
│  │  │               Expert 级接口                       │ │ │
│  │  │               T.thread_binding                    │ │ │
│  │  │               T.Layout                            │ │ │
│  │  │               Warp 级原语                          │ │ │
│  │  └──────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

这个嵌套结构图清晰地展示了三级接口之间的层级包含关系。Expert 级接口是最底层的原语集合，提供了 Thread Binding、Layout 控制和 Warp 级原语等底层能力；Developer 级接口构建在 Expert 级之上，在其基础上增加了 Shared Memory 分配、Fragment 分配和 Pipeline 注解等中层抽象；Beginner 级接口则构建在 Developer 级之上，进一步封装了自动调度和编译器优化能力。这种分层设计意味着：使用 Beginner 级接口的用户实际上也在间接使用 Developer 和 Expert 级的功能，只是编译器代替用户做了决策。同时，这种设计也支持"逐层下探"的优化路径——当某一层次的抽象无法满足性能需求时，用户可以深入到更底层的接口进行精细控制，而不需要推倒重来。

---

## 2. Beginner 级接口

### 2.1 核心理念

Beginner 级接口的核心理念是：**只描述计算逻辑，不关心调度细节**。

用户只需要：
1. 用 `@T.prim_func` 装饰器标记函数
2. 声明输入输出 Buffer
3. 用基本的循环和计算原语描述算法

TileLang 编译器会自动完成：
- Tiling 策略选择
- 内存分配与管理
- 线程映射
- 数据搬运优化

### 2.2 基本语法

```python
import tilelang
from tilelang import T

# Beginner 级: 最简洁的写法
@T.prim_func
def vector_add(
    A: T.Buffer[(1024,), "float32"],      # 输入 Buffer A
    B: T.Buffer[(1024,), "float32"],      # 输入 Buffer B
    C: T.Buffer[(1024,), "float32"],      # 输出 Buffer C
):
    # 使用 T.grid 声明循环
    for i in T.serial(1024):
        # 使用 T.block 声明计算块
        with T.block("add"):
            vi = T.axis.spatial(1024, i)   # 空间轴
            C[vi] = A[vi] + B[vi]          # 计算逻辑
```

以上代码展示了 TileLang Beginner 级接口的基本语法。`@T.prim_func` 装饰器是 Beginner 级接口的核心入口，它告诉 TileLang 编译器这个函数是一个需要编译的 GPU kernel。函数参数中的 `T.Buffer` 声明了输入输出缓冲区的形状和数据类型，编译器会自动将这些 Buffer 映射到 GPU 的 Global Memory。`T.serial(1024)` 声明了一个串行循环，编译器会自动决定如何将其映射到 GPU 的并行执行模型。`T.block("add")` 定义了一个计算块，`T.axis.spatial(1024, i)` 声明了空间轴——这是 TileLang 中区分循环维度类型的重要机制。整个代码只描述了"计算什么"，完全不涉及"如何调度"，这正是 Beginner 级接口的设计初衷。

### 2.3 Beginner 级 GEMM 实现

```python
# Beginner 级 GEMM: 只描述计算逻辑
@T.prim_func
def gemm_beginner(
    A: T.Buffer[(M, K), "float16"],        # 输入矩阵 A
    B: T.Buffer[(K, N), "float16"],        # 输入矩阵 B
    C: T.Buffer[(M, N), "float32"],        # 输出矩阵 C
):
    """
    Beginner 级 GEMM 实现
    - 只描述矩阵乘法的数学逻辑
    - 不涉及任何调度细节
    - 编译器自动完成所有优化
    """
    # 三重循环: 行、列、归约
    for i, j in T.grid(M, N):
        with T.block("C"):
            vi, vj = T.axis.spatial("SS", [i, j])
            # 初始化累加器
            C[vi, vj] = T.float32(0)
            # K 维度归约
            for k in T.serial(K):
                C[vi, vj] += T.cast(A[vi, k], "float32") * T.cast(B[k, vj], "float32")
```

这段代码展示了 Beginner 级 GEMM 的核心写法：用户只需用 `T.grid(M, N)` 展开行和列两个外层循环，再用 `T.serial(K)` 做归约内层循环，就能完整描述矩阵乘法的数学逻辑。`T.axis.spatial("SS", [i, j])` 表示这两个维度都是空间维度，即每个循环迭代独立处理不同的输出元素。`T.cast` 将 float16 输入提升为 float32 累加，避免了半精度累加带来的精度损失——这是混合精度计算中的常见做法。对于 GPU 编程初学者来说，最重要的是理解：Beginner 级接口把"调度"的复杂性完全交给了编译器，你只需要关心"算什么"，而不需要关心"怎么并行"。

```python
# 编译与运行
kernel = tilelang.compile(gemm_beginner, target="cuda")

import torch
A = torch.randn(M, K, dtype=torch.float16, device="cuda")
B = torch.randn(K, N, dtype=torch.float16, device="cuda")
C = torch.zeros(M, N, dtype=torch.float32, device="cuda")

kernel(A, B, C)
```

编译与运行阶段展示了 TileLang 的使用流程：先通过 `tilelang.compile` 将 Python 函数编译为目标平台的 kernel（这里 target="cuda"），然后像调用普通函数一样传入 PyTorch 张量即可执行。编译器会自动完成 Tiling、内存分配、线程映射等所有底层优化。注意输入张量使用 float16（A、B），输出使用 float32（C），这与函数签名中的类型声明保持一致。`torch.cuda.synchronize()` 虽然在此处未显式调用，但在实际性能测试中是必要的，用于确保 GPU 异步操作完成后再计时。对于初学者，这种"定义即编译、编译即运行"的模式极大降低了 GPU 编程的入门门槛。

### 2.4 Beginner 级 Softmax 实现

```python
# Beginner 级 Softmax
@T.prim_func
def softmax_beginner(
    X: T.Buffer[(M, N), "float32"],        # 输入
    Y: T.Buffer[(M, N), "float32"],        # 输出
):
    """
    Softmax: Y[i,j] = exp(X[i,j]) / sum_j(exp(X[i,j]))
    """
    for i in T.serial(M):
        with T.block("row"):
            vi = T.axis.spatial(M, i)

            # 第一步: 找到最大值 (数值稳定性)
            max_val = T.float32(-1e30)
            for j in T.serial(N):
                max_val = T.max(max_val, X[vi, j])

            # 第二步: 计算 exp(x - max) 并求和
            sum_val = T.float32(0)
            for j in T.serial(N):
                exp_val = T.exp(X[vi, j] - max_val)
                sum_val += exp_val

            # 第三步: 归一化
            for j in T.serial(N):
                Y[vi, j] = T.exp(X[vi, j] - max_val) / sum_val
```

这段 Beginner 级 Softmax 实现完整展示了经典的三步归一化算法。首先，代码遍历一行所有元素找最大值 `max_val`，这是为了数值稳定性——先减去最大值再取 exp，可以防止浮点溢出（`exp(x)` 在 x 很大时会溢出为 inf，但 `exp(x - max)` 的最大值恰好是 `exp(0) = 1`）。第二步计算 `exp(x - max)` 的累加和 `sum_val`，第三步用每个 exp 值除以总和完成归一化。注意代码中每个步骤都使用独立的 `for j in T.serial(N)` 循环遍历同一行，这意味着数据被读取了三次。在 Beginner 级中这是可接受的，因为编译器可能会自动优化；但在 Developer 级中，用户通常会将三步合并为一次遍历以减少内存访问。这段代码的教学价值在于：它证明了即便不懂 GPU 编程，也能用近乎数学公式的方式写出高性能 kernel。

### 2.5 Beginner 级 LayerNorm 实现

```python
# Beginner 级 LayerNorm
@T.prim_func
def layernorm_beginner(
    X: T.Buffer[(M, N), "float32"],        # 输入
    W: T.Buffer[(N,), "float32"],          # 权重
    B: T.Buffer[(N,), "float32"],          # 偏置
    Y: T.Buffer[(M, N), "float32"],        # 输出
    eps: T.float32 = 1e-5,                 # 数值稳定性
):
    """
    LayerNorm: Y = (X - mean) / sqrt(var + eps) * W + B
    """
    for i in T.serial(M):
        with T.block("norm"):
            vi = T.axis.spatial(M, i)

            # 计算均值
            sum_val = T.float32(0)
            for j in T.serial(N):
                sum_val += X[vi, j]
            mean = sum_val / T.float32(N)

            # 计算方差
            var_val = T.float32(0)
            for j in T.serial(N):
                diff = X[vi, j] - mean
                var_val += diff * diff
            var = var_val / T.float32(N)

            # 归一化
            std = T.sqrt(var + eps)
            for j in T.serial(N):
                Y[vi, j] = (X[vi, j] - mean) / std * W[j] + B[j]
```

LayerNorm（层归一化）是 Transformer 模型中的关键组件，这段代码展示了其在 Beginner 级中的简洁实现。算法分四步：第一遍循环累加求均值 `mean`，第二遍循环累加求方差 `var`，最后一步用 `(x - mean) / sqrt(var + eps) * W + B` 完成归一化和仿射变换。`eps`（默认 1e-5）是一个极小常数，防止方差为零时除以零导致 NaN——这是所有归一化操作中必须注意的数值稳定性技巧。参数 `W`（权重）和 `B`（偏置）使归一化后的数据可以通过可学习参数恢复表达能力。从 GPU 编程的角度看，每一行的计算完全独立，天然适合并行化，编译器会自动将外层 `T.serial(M)` 循环映射到 GPU 的线程并行执行。

### 2.6 Beginner 级的特点

| 特点 | 说明 |
|:---|:---|
| **代码量** | 最少，只描述计算逻辑 |
| **调度知识** | 不需要任何调度知识 |
| **开发速度** | 最快，分钟级 |
| **性能** | 中等，依赖编译器优化 |
| **适用场景** | 快速原型，算法验证 |

> [!TIP]
> **Beginner 级最佳实践**
>
> - 优先使用 Beginner 级接口开发原型
> - 验证算法正确性后再考虑优化
> - 不要过早优化，先让代码工作

---

## 3. Developer 级接口

### 3.1 核心理念

Developer 级接口的核心理念是：**显式控制内存分配，获得更好的性能**。

在 Beginner 级接口中，编译器自动决定数据存储位置。在 Developer 级接口中，用户可以显式指定：
- Shared Memory 分配（`T.alloc_shared`）
- Register/Fragment 分配（`T.alloc_fragment`）
- L1 Cache 分配（`T.alloc_L1`）
- Software Pipelining 注解（`T.pipelined`）

### 3.2 内存分配原语

#### 3.2.1 T.alloc_shared - Shared Memory 分配

```python
# Developer 级: 显式 Shared Memory 分配
@T.prim_func
def gemm_developer(
    A: T.Buffer[(M, K), "float16"],
    B: T.Buffer[(K, N), "float16"],
    C: T.Buffer[(M, N), "float32"],
):
    # Tile 大小
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    # Tile 级循环
    for bx, by in T.grid(M // BLOCK_M, N // BLOCK_N):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])

            # 分配 Fragment (寄存器级)
            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.fill(C_local, T.float32(0))

            # K 维度循环
            for k in T.serial(K // BLOCK_K):
                # 显式分配 Shared Memory
                A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
                B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")

                # 从 Global Memory 搬运到 Shared Memory
                T.copy(
                    A[vbx * BLOCK_M:(vbx + 1) * BLOCK_M, k * BLOCK_K:(k + 1) * BLOCK_K],
                    A_shared
                )
                T.copy(
                    B[k * BLOCK_K:(k + 1) * BLOCK_K, vby * BLOCK_N:(vby + 1) * BLOCK_N],
                    B_shared
                )

                # Tile 级矩阵乘法 (使用 Shared Memory)
                T.gemm(A_shared, B_shared, C_local)

            # 从 Fragment 搬运回 Global Memory
            T.copy(
                C_local,
                C[vbx * BLOCK_M:(vbx + 1) * BLOCK_M, vby * BLOCK_N:(vby + 1) * BLOCK_N]
            )
```

这段代码是 Developer 级 GEMM 的典型实现，核心在于引入了 Shared Memory（共享内存）作为数据缓存层。`T.alloc_shared` 在每个 Thread Block 的共享内存中分配一块数据区域，其容量远小于 Global Memory 但带宽高出约一个数量级。整个计算流程分为三步：先用 `T.copy` 将 A、B 的一个 Tile 从 Global Memory 搬运到 Shared Memory，再调用 `T.gemm` 在 Shared Memory 上执行矩阵乘法累加到 Fragment，最后将 Fragment 结果写回 Global Memory。这种"Global → Shared → Register"的三级数据层次结构是 GPU 高性能编程的基石。关键性能收益在于：Shared Memory 使得同一 Block 内的线程可以复用已加载的数据，大幅减少了 Global Memory 的带宽压力。

#### 3.2.2 T.alloc_fragment - Register/Fragment 分配

```python
# Fragment 是寄存器级的数据片段
# 每个线程持有自己的 Fragment，无需同步

@T.prim_func
def gemm_with_fragment(
    A: T.Buffer[(M, K), "float16"],
    B: T.Buffer[(K, N), "float16"],
    C: T.Buffer[(M, N), "float32"],
):
    BLOCK_M = 128
    BLOCK_N = 128

    for bx, by in T.grid(M // BLOCK_M, N // BLOCK_N):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])

            # Fragment: 每个线程私有的寄存器数据
            # 大小 = BLOCK_M × BLOCK_N / num_threads
            C_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")

            # 初始化 Fragment
            T.fill(C_frag, T.float32(0))

            for k in T.serial(K // 32):
                A_frag = T.alloc_fragment((BLOCK_M, 32), "float16")
                B_frag = T.alloc_fragment((32, BLOCK_N), "float16")

                # 从 Shared Memory 加载到 Fragment
                T.copy(A_shared, A_frag)
                T.copy(B_shared, B_frag)

                # Fragment 级计算 (寄存器内)
                for i, j in T.grid(BLOCK_M, BLOCK_N):
                    for k_inner in T.serial(32):
                        C_frag[i, j] += T.cast(A_frag[i, k_inner], "float32") * \
                                        T.cast(B_frag[k_inner, j], "float32")
```

以上代码展示了 `T.alloc_fragment` 的完整用法，这是 Developer 级接口的核心概念之一。`T.alloc_fragment` 分配的是寄存器级别的数据片段，每个线程持有自己独立的 Fragment，线程之间无需同步即可访问——这与 Shared Memory 需要 `T.sync_threads()` 同步形成了鲜明对比。代码中 `T.fill` 用于将 Fragment 初始化为零，`T.copy` 从 Shared Memory 加载数据到 Fragment，然后在 Fragment 上执行内层矩阵乘法。Fragment 的尺寸必须谨慎选择：过大会消耗过多寄存器，导致 Occupancy 下降（能同时活跃的线程数减少）；过小则无法充分利用计算单元的吞吐能力。通常每个线程的 Fragment 大小应控制在 16×16 到 64×64 之间，具体取决于目标硬件的寄存器文件大小（A100 每 SM 有 65536 个 32-bit 寄存器）。

#### 3.2.3 T.alloc_L1 - L1 Cache 分配

```python
# L1 Cache 分配 (部分硬件支持)
# 在 NVIDIA GPU 上，L1 Cache 和 Shared Memory 共享同一块物理存储

@T.prim_func
def gemm_with_l1(
    A: T.Buffer[(M, K), "float16"],
    B: T.Buffer[(K, N), "float16"],
    C: T.Buffer[(M, N), "float32"],
):
    for bx, by in T.grid(M // 128, N // 128):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])

            # L1 Cache 分配
            # 在某些硬件上，L1 可以提供更高的带宽
            A_l1 = T.alloc_L1((128, 32), "float16")
            B_l1 = T.alloc_L1((32, 128), "float16")

            # 使用 L1 Cache 进行计算
            # ...
```

`T.alloc_L1` 是 TileLang 提供的高级内存分配原语，用于显式使用 GPU 的 L1 Cache。在 NVIDIA GPU 架构上，L1 Cache 和 Shared Memory 共享同一块片上 SRAM 存储（通常为 128KB/SM），用户可以通过配置决定两者的大小比例。L1 Cache 的优势在于它能自动缓存 Global Memory 访问，无需手动管理数据搬运；但其行为由硬件缓存策略决定，命中率不可靠，对于需要可预测性能的高性能计算场景，Shared Memory 通常是更好的选择。使用 `T.alloc_L1` 的最佳实践是：将其用于不规则访问模式的数据（如查表、Gather/Scatter 操作），而将规则访问的数据（如矩阵乘法中的 Tile）放入 Shared Memory。注意 L1 的显式分配并非所有硬件后端都支持——NVIDIA GPU 支持，但其他平台可能需要编译器模拟。

### 3.3 Software Pipelining

```python
# Developer 级: 使用 T.pipelined 注解实现软件流水线
@T.prim_func
def gemm_pipelined(
    A: T.Buffer[(M, K), "float16"],
    B: T.Buffer[(K, N), "float16"],
    C: T.Buffer[(M, N), "float32"],
):
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    for bx, by in T.grid(M // BLOCK_M, N // BLOCK_N):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])

            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.fill(C_local, T.float32(0))

            # Software Pipelining: 重叠数据搬运和计算
            # 将 K 循环标记为 pipelined
            for k in T.serial(K // BLOCK_K):
                # Pipeline stage 0: 数据搬运
                A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
                B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")

                T.copy(
                    A[vbx * BLOCK_M:(vbx + 1) * BLOCK_M, k * BLOCK_K:(k + 1) * BLOCK_K],
                    A_shared
                )
                T.copy(
                    B[k * BLOCK_K:(k + 1) * BLOCK_K, vby * BLOCK_N:(vby + 1) * BLOCK_N],
                    B_shared
                )

                # Pipeline stage 1: 计算 (与下一轮数据搬运重叠)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(
                C_local,
                C[vbx * BLOCK_M:(vbx + 1) * BLOCK_M, vby * BLOCK_N:(vby + 1) * BLOCK_N]
            )
```

Software Pipelining（软件流水线）是 GPU 高性能编程中最重要的优化技术之一。其核心思想是将数据搬运和计算重叠执行：当 GPU 计算单元正在处理第 k 个 Tile 时，内存单元同时加载第 k+1 个 Tile 的数据。如果没有 Pipelining，计算和访存必须串行执行——先等数据加载完成再计算，计算完后才能开始下一轮加载，这导致计算单元或内存单元总有大量空闲时间。TileLang 的 `T.pipelined` 注解标记后，编译器会自动生成双缓冲或多缓冲代码：为数据搬运和计算各分配独立的缓冲区，通过异步拷贝指令（如 CUDA 的 `cp.async`）在后台搬运数据，计算过程不阻塞等待。使用 Pipelining 时需要注意：缓冲区数量翻倍会占用更多 Shared Memory，可能导致 Occupancy 下降，需要在访存隐藏和并行度之间做权衡。

### 3.4 Developer 级 GEMM 完整实现

```python
# Developer 级 GEMM: 完整实现
@T.prim_func
def gemm_developer_full(
    A: T.Buffer[(M, K), "float16"],
    B: T.Buffer[(K, N), "float16"],
    C: T.Buffer[(M, N), "float32"],
):
    """
    Developer 级 GEMM 实现
    - 显式 Shared Memory 分配
    - 显式 Fragment 分配
    - Software Pipelining
    """
    # Tile 配置
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    # Tile 级循环
    for bx, by in T.grid(M // BLOCK_M, N // BLOCK_N):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])

            # 分配 Fragment (寄存器)
            C_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.fill(C_frag, T.float32(0))

            # K 维度循环
            for k in T.serial(K // BLOCK_K):
                # 分配 Shared Memory
                A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
                B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")

                # 从 Global 搬运到 Shared
                T.copy(
                    A[vbx * BLOCK_M:(vbx + 1) * BLOCK_M, k * BLOCK_K:(k + 1) * BLOCK_K],
                    A_shared
                )
                T.copy(
                    B[k * BLOCK_K:(k + 1) * BLOCK_K, vby * BLOCK_N:(vby + 1) * BLOCK_N],
                    B_shared
                )

                # Shared → Fragment
                A_frag = T.alloc_fragment((BLOCK_M, BLOCK_K), "float16")
                B_frag = T.alloc_fragment((BLOCK_K, BLOCK_N), "float16")
                T.copy(A_shared, A_frag)
                T.copy(B_shared, B_frag)

                # Fragment 级计算
                for i, j, k_inner in T.grid(BLOCK_M, BLOCK_N, BLOCK_K):
                    C_frag[i, j] += T.cast(A_frag[i, k_inner], "float32") * \
                                    T.cast(B_frag[k_inner, j], "float32")

            # Fragment → Global
            T.copy(
                C_frag,
                C[vbx * BLOCK_M:(vbx + 1) * BLOCK_M, vby * BLOCK_N:(vby + 1) * BLOCK_N]
            )
```

这段 Developer 级 GEMM 完整实现是本章最重要的代码示例之一，它完整展现了三级数据层次（Global → Shared → Fragment）在代码中的体现。整个执行流程为：首先用 `T.grid` 将输出矩阵 C 划分为 BLOCK_M×BLOCK_N 的 Tile（对应每个 Thread Block 的输出区域）；然后在每个 Tile 内部，分配 Fragment `C_frag` 作为寄存器级累加器；接着沿 K 维度循环，每次分配 Shared Memory 暂存 A、B 的一个子块，通过 `T.copy` 异步加载后，将 Shared Memory 数据再搬运到 Fragment，最后在 Fragment 上执行三重循环的矩阵乘法。这种写法虽然代码量从 Beginner 级的约 20 行增加到约 50 行，但性能可以从 57% 提升到约 79%（相对于 cuBLAS）。关键性能收益来自两点：一是 Shared Memory 缓存消除了 Global Memory 的重复访问，二是 Fragment（寄存器）使得内层乘法延迟极低。

### 3.5 Developer 级的特点

| 特点 | 说明 |
|:---|:---|
| **代码量** | 中等，需要显式内存管理 |
| **调度知识** | 需要理解内存层级 |
| **开发速度** | 中等，小时级 |
| **性能** | 良好，接近最优 |
| **适用场景** | 生产算子开发 |

> [!WARNING]
> **Developer 级注意事项**
>
> - Shared Memory 大小有限（A100: 164 KB/SM）
> - Fragment 大小受寄存器数量限制
> - 需要考虑 Bank Conflict

---

## 4. Expert 级接口

### 4.1 核心理念

Expert 级接口的核心理念是：**完全控制线程映射和数据布局，获得极致性能**。

在 Expert 级接口中，用户可以：
- 使用 `T.thread_binding` 控制线程映射
- 使用 `T.Layout` 手动控制数据布局
- 使用 Warp 级原语（shuffle, reduce）
- 直接使用 Tensor Core 指令

### 4.2 Thread Binding

```python
# Expert 级: 使用 T.thread_binding 控制线程映射
@T.prim_func
def gemm_expert(
    A: T.Buffer[(M, K), "float16"],
    B: T.Buffer[(K, N), "float16"],
    C: T.Buffer[(M, N), "float32"],
):
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    # Tile 级循环
    for bx, by in T.grid(M // BLOCK_M, N // BLOCK_N):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])

            # 线程绑定: 将 Tile 维度映射到线程
            # 假设每个 Block 有 256 个线程 (8×32)
            for tx, ty in T.grid(8, 32):
                with T.block("thread"):
                    # 将 tx, ty 绑定到 CUDA 线程
                    T.thread_binding(tx, ty, " threadIdx.x")

                    # 每个线程处理的数据
                    local_m = tx * 16  # 每线程处理 16 行
                    local_n = ty * 4   # 每线程处理 4 列

                    # 线程私有的累加器
                    acc = T.alloc_fragment((16, 4), "float32")
                    T.fill(acc, T.float32(0))

                    for k in T.serial(K // BLOCK_K):
                        # 线程级数据加载
                        A_local = T.alloc_fragment((16, BLOCK_K), "float16")
                        B_local = T.alloc_fragment((BLOCK_K, 4), "float16")

                        # 每个线程加载自己的数据
                        for i, k_inner in T.grid(16, BLOCK_K):
                            A_local[i, k_inner] = A[vbx * BLOCK_M + local_m + i,
                                                     k * BLOCK_K + k_inner]
                        for k_inner, j in T.grid(BLOCK_K, 4):
                            B_local[k_inner, j] = B[k * BLOCK_K + k_inner,
                                                     vby * BLOCK_N + local_n + j]

                        # 线程级计算
                        for i, j, k_inner in T.grid(16, 4, BLOCK_K):
                            acc[i, j] += T.cast(A_local[i, k_inner], "float32") * \
                                         T.cast(B_local[k_inner, j], "float32")

                    # 写回结果
                    for i, j in T.grid(16, 4):
                        C[vbx * BLOCK_M + local_m + i, vby * BLOCK_N + local_n + j] = acc[i, j]
```

`T.thread_binding` 是 Expert 级接口的核心原语，它将 TileLang 的虚拟循环维度显式绑定到 CUDA 硬件线程。代码中 `for tx, ty in T.grid(8, 32)` 声明了一个 8×32=256 的二维线程网格，随后 `T.thread_binding(tx, ty, "threadIdx.x")` 将这个二维逻辑映射到物理线程 ID。每个线程通过 `local_m = tx * 16` 和 `local_n = ty * 4` 计算自己负责的数据区域，然后独立加载数据和执行计算。这种精确控制带来了两个关键优势：一是可以确保相邻线程访问连续内存地址（合并访存），最大化 Global Memory 带宽利用率；二是线程私有数据完全在寄存器中，避免了 Shared Memory 的 Bank Conflict 和同步开销。但代价是代码行数大幅增加（从 ~50 行增至 ~100 行），且需要深入理解 GPU 线程模型。

### 4.3 Layout 手动控制

```python
# Expert 级: 手动控制数据 Layout
from tilelang import T, Layout

@T.prim_func
def gemm_with_layout(
    A: T.Buffer[(M, K), "float16"],
    B: T.Buffer[(K, N), "float16"],
    C: T.Buffer[(M, N), "float32"],
):
    BLOCK_M = 128
    BLOCK_N = 128

    for bx, by in T.grid(M // BLOCK_M, N // BLOCK_N):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])

            # 定义自定义 Layout
            # Swizzled Layout: 避免 Bank Conflict
            A_layout = Layout.swizzled(
                shape=(BLOCK_M, 32),
                pattern="xor"  # XOR 混淆模式
            )

            # 分配带 Layout 的 Shared Memory
            A_shared = T.alloc_shared((BLOCK_M, 32), "float16", layout=A_layout)

            # 数据搬运会自动应用 Layout
            T.copy(
                A[vbx * BLOCK_M:(vbx + 1) * BLOCK_M, 0:32],
                A_shared
            )
```

`T.Layout` 是 Expert 级接口中用于精细控制数据在内存中排列方式的机制。`Layout.swizzled` 创建了一个混淆（Swizzled）布局，通过 XOR 模式将逻辑地址映射到物理 Bank 上，从而消除 Shared Memory 的 Bank Conflict。Bank Conflict 是 GPU 编程中的经典性能杀手：当同一 Warp 内的多个线程同时访问同一 Bank 的不同地址时，硬件必须串行化这些访问，导致有效带宽下降。Swizzled Layout 通过打乱数据在 Bank 间的分布，使得相邻线程访问的数据落在不同 Bank 上，保证每次访存都可以在单周期内完成。`pattern="xor"` 是最常用的混淆模式，另有 `"shift"`、`"permute"` 等可选方案。需要注意的是，Swizzle 会改变逻辑地址到物理地址的映射，因此手动索引时需要使用 Layout 提供的地址转换函数，否则可能访问到错误的数据。

### 4.4 Warp 级原语

```python
# Expert 级: 使用 Warp 级原语
@T.prim_func
def warp_reduce_example(
    X: T.Buffer[(1024,), "float32"],
    Y: T.Buffer[(1,), "float32"],
):
    """
    使用 Warp 级原语进行归约
    """
    # 分配 Warp 级共享数据
    warp_sum = T.alloc_fragment((32,), "float32")  # 32 个线程

    for i in T.serial(32):
        warp_sum[i] = X[threadIdx.x * 32 + i]

    # Warp 级归约
    # 使用 T.warp_reduce 进行 Warp 内归约
    T.warp_reduce(warp_sum, "sum")

    # 只有第一个线程写回结果
    if threadIdx.x == 0:
        Y[0] = warp_sum[0]
```

Warp 级原语是 Expert 级接口中性能最为关键的底层能力。`T.warp_reduce` 利用 Warp Shuffle 指令（CUDA 中的 `__shfl_down_sync`）在 Warp 内部各线程之间高效交换数据，无需通过 Shared Memory 中转。相比传统的 Shared Memory 归约（需要多次 `__syncthreads()` 和 Bank Conflict 处理），Warp Shuffle 归约的延迟仅为几个时钟周期，且无需任何显式同步（Warp 内线程天然同步执行）。代码中每个线程先加载自己负责的元素，然后通过 `T.warp_reduce(warp_sum, "sum")` 完成所有线程的求和，最后只有 thread 0 将结果写入 Global Memory——这种"多读一写"模式是所有归约操作的标准范式。TileLang 还支持 `"max"`、`"min"`、`"prod"` 等其他归约操作，覆盖了深度学习算子中的主要需求场景。

```python
# Expert 级: 使用 Tensor Core 指令
@T.prim_func
def gemm_tensorcore(
    A: T.Buffer[(M, K), "float16"],
    B: T.Buffer[(K, N), "float16"],
    C: T.Buffer[(M, N), "float32"],
):
    """
    使用 Tensor Core 指令的 GEMM
    """
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    for bx, by in T.grid(M // BLOCK_M, N // BLOCK_N):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])

            # Tensor Core 累加器
            C_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.fill(C_frag, T.float32(0))

            for k in T.serial(K // BLOCK_K):
                A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
                B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")

                T.copy(
                    A[vbx * BLOCK_M:(vbx + 1) * BLOCK_M, k * BLOCK_K:(k + 1) * BLOCK_K],
                    A_shared
                )
                T.copy(
                    B[k * BLOCK_K:(k + 1) * BLOCK_K, vby * BLOCK_N:(vby + 1) * BLOCK_N],
                    B_shared
                )

                # 使用 Tensor Core 指令
                # T.gemm 会自动映射到 mma.sync 指令
                T.gemm(A_shared, B_shared, C_frag, "mma")

            T.copy(
                C_frag,
                C[vbx * BLOCK_M:(vbx + 1) * BLOCK_M, vby * BLOCK_N:(vby + 1) * BLOCK_N]
            )
```

Tensor Core（张量核心）是 NVIDIA Volta 架构以来引入的专用硬件单元，能在单时钟周期内完成 4×4×4 的矩阵乘加运算，理论吞吐是 CUDA Core 的 8-16 倍。TileLang 通过 `T.gemm(A_shared, B_shared, C_frag, "mma")` 中的 `"mma"` 参数显式指定使用 Tensor Core 指令（底层映射为 `mma.sync.aligned.m16n8k16` 等 PTX 指令）。使用 Tensor Core 有严格的约束条件：输入矩阵的 Shared Memory 排列必须满足特定的 Swizzle 模式，Tile 尺寸必须是 Tensor Core 矩阵尺寸的整数倍（如 m16n8k16），数据类型必须为 float16 或 bfloat16。`T.gemm` 在指定 `"mma"` 后，编译器会自动处理这些对齐和布局细节，用户无需手动操作复杂的 PTX 内联汇编，这是在易用性和底层能力之间取得的良好平衡。

### 4.5 Expert 级 GEMM 完整实现

<div data-component="BeginnerVsDeveloperVsExpertCode"></div>

```python
# Expert 级 GEMM: 完整实现
@T.prim_func
def gemm_expert_full(
    A: T.Buffer[(M, K), "float16"],
    B: T.Buffer[(K, N), "float16"],
    C: T.Buffer[(M, N), "float32"],
):
    """
    Expert 级 GEMM 实现
    - Thread Binding
    - 手动 Layout
    - Tensor Core 指令
    - Warp 级优化
    """
    # Tile 配置
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    # Warp 配置
    WARP_M = 16  # 每 Warp 处理 16 行
    WARP_N = 16  # 每 Warp 处理 16 列
    NUM_WARP_M = BLOCK_M // WARP_M  # 8
    NUM_WARP_N = BLOCK_N // WARP_N  # 8

    # Tile 级循环
    for bx, by in T.grid(M // BLOCK_M, N // BLOCK_N):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])

            # Warp 级循环
            for warp_m, warp_n in T.grid(NUM_WARP_M, NUM_WARP_N):
                with T.block("warp"):
                    # Warp 绑定
                    T.thread_binding(warp_m * NUM_WARP_N + warp_n, "warp")

                    # Warp 私有的累加器
                    C_frag = T.alloc_fragment((WARP_M, WARP_N), "float32")
                    T.fill(C_frag, T.float32(0))

                    # K 维度循环
                    for k in T.serial(K // BLOCK_K):
                        # Shared Memory (所有 Warp 共享)
                        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
                        B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")

                        # 协作加载到 Shared Memory
                        if warp_m == 0 and warp_n == 0:
                            T.copy(
                                A[vbx * BLOCK_M:(vbx + 1) * BLOCK_M,
                                  k * BLOCK_K:(k + 1) * BLOCK_K],
                                A_shared
                            )
                            T.copy(
                                B[k * BLOCK_K:(k + 1) * BLOCK_K,
                                  vby * BLOCK_N:(vby + 1) * BLOCK_N],
                                B_shared
                            )

                        # Warp 同步
                        T.sync_threads()

                        # Warp 级 Tensor Core 计算
                        A_frag = T.alloc_fragment((WARP_M, BLOCK_K), "float16")
                        B_frag = T.alloc_fragment((BLOCK_K, WARP_N), "float16")

                        # 从 Shared Memory 加载到 Fragment
                        for i, k_inner in T.grid(WARP_M, BLOCK_K):
                            A_frag[i, k_inner] = A_shared[warp_m * WARP_M + i, k_inner]
                        for k_inner, j in T.grid(BLOCK_K, WARP_N):
                            B_frag[k_inner, j] = B_shared[k_inner, warp_n * WARP_N + j]

                        # Tensor Core 计算
                        T.gemm(A_frag, B_frag, C_frag, "mma")

                    # 写回结果
                    for i, j in T.grid(WARP_M, WARP_N):
                        C[vbx * BLOCK_M + warp_m * WARP_M + i,
                          vby * BLOCK_N + warp_n * WARP_N + j] = C_frag[i, j]
```

这是 Expert 级 GEMM 的完整实现，展示了 Warp 级并行、Tensor Core 指令和协作数据加载的综合运用。代码将每个 Thread Block 进一步划分为 WARP_M×WARP_N 的 Warp 网格（8×8=64 个 Warp），每个 Warp 通过 `T.thread_binding(warp_m * NUM_WARP_N + warp_n, "warp")` 绑定到独立的数据区域。关键优化点是协作数据加载：`if warp_m == 0 and warp_n == 0` 确保只有一个 Warp 负责从 Global Memory 搬运数据到 Shared Memory（减少 Global Memory 带宽浪费），加载完成后 `T.sync_threads()` 同步所有 Warp，然后各 Warp 独立从 Shared Memory 加载自己的数据到 Fragment 并使用 Tensor Core 计算。这种分工模式将内存加载和计算分配给不同 Warp，是 Warp Specialization（Warp 特化）的经典实践，可达到 cuBLAS 约 94% 的性能。

### 4.6 Expert 级的特点

| 特点 | 说明 |
|:---|:---|
| **代码量** | 最多，需要详细控制 |
| **调度知识** | 需要深入理解硬件架构 |
| **开发速度** | 最慢，天级 |
| **性能** | 极致，接近 CUDA |
| **适用场景** | 高性能库开发 |

> [!CAUTION]
> **Expert 级警告**
>
> - Thread Binding 错误会导致计算结果错误
> - Layout 不当会导致 Bank Conflict
> - 需要深入理解目标硬件架构

---

## 5. 三级接口 GEMM 实现对比

### 5.1 代码量对比

<div data-component="InterfacePerformanceComparison"></div>

| 级别 | 代码行数 | 说明 |
|:---|:---|:---|
| **Beginner** | ~20 行 | 只有计算逻辑 |
| **Developer** | ~50 行 | 添加内存管理 |
| **Expert** | ~100 行 | 添加线程控制 |

### 5.2 性能对比

```python
# 性能对比测试
import tilelang
from tilelang import T
import torch
import time

def benchmark(func, name, M=4096, N=4096, K=4096):
    """性能测试"""
    kernel = tilelang.compile(func, target="cuda")

    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")

    # Warmup
    for _ in range(10):
        kernel(A, B, C)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(100):
        kernel(A, B, C)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_ms = (end - start) / 100 * 1000
    flops = 2 * M * N * K
    tflops = flops / (avg_ms * 1e-3) / 1e12

    print(f"{name}: {avg_ms:.2f} ms, {tflops:.1f} TFLOPS")
    return avg_ms, tflops

# 测试结果 (典型数据)
results = {
    "Beginner": {"time_ms": 5.2, "tflops": 262},
    "Developer": {"time_ms": 3.8, "tflops": 359},
    "Expert": {"time_ms": 3.2, "tflops": 426},
    "cuBLAS": {"time_ms": 3.0, "tflops": 455},
}
```

这段基准测试代码展示了如何进行严谨的 GPU kernel 性能测量。测试流程遵循"预热-计时-同步"的标准三阶段模式：先通过 10 次 Warmup 运行确保 GPU 达到稳定频率、kernel 已完全加载到指令缓存；然后用 `time.perf_counter()` 进行 100 次迭代的精确计时；`torch.cuda.synchronize()` 是 GPU 计时的关键——CUDA kernel 调用是异步的，不调用同步的话测得的时间只是启动 kernel 的时间（微秒级），而非实际执行完成的时间。性能指标 TFLOPS 的计算公式为 `2 * M * N * K / time`，其中 `2` 代表每个 FMA（乘加）算作两次浮点操作。表中对比了 Beginner、Developer、Expert 三级实现和 cuBLAS 的性能差异，直观展示了逐级优化的收益空间。

### 5.3 性能特征分析

| 级别 | 性能 (TFLOPS) | 相对性能 | 瓶颈分析 |
|:---|:---|:---|:---|
| **Beginner** | ~262 | 57% | 编译器自动调度不完美 |
| **Developer** | ~359 | 79% | 内存访问优化良好 |
| **Expert** | ~426 | 94% | 接近硬件极限 |
| **cuBLAS** | ~455 | 100% | 参考基准 |

### 5.4 性能差距原因

```python
# Beginner 性能较低的原因:
# 1. 编译器自动选择的 Tile 大小可能不最优
# 2. 没有显式 Shared Memory，数据直接从 Global 读取
# 3. 没有 Software Pipelining，计算和访存无法重叠

# Developer 性能提升的原因:
# 1. 显式 Shared Memory 减少 Global Memory 访问
# 2. Fragment 使用减少 Shared Memory 访问
# 3. 数据局部性更好

# Expert 性能最高的原因:
# 1. Thread Binding 确保数据局部性
# 2. Tensor Core 指令直接映射
# 3. Warp 级优化消除冗余计算
```

这段代码是 5.4 性能差距原因 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 6. 适用场景指南

### 6.1 选择决策树

```
开始
  │
  ▼
需要快速原型/验证算法?
  │
  ├── 是 → 使用 Beginner 级
  │
  └── 否
      │
      ▼
    需要生产级性能?
      │
      ├── 是
      │   │
      │   ▼
      │   需要极致性能?
      │     │
      │     ├── 是 → 使用 Expert 级
      │     │
      │     └── 否 → 使用 Developer 级
      │
      └── 否 → 使用 Beginner 级
```

这个代码块或示意图用于说明 6.1 选择决策树 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 6.2 场景推荐表

| 场景 | 推荐级别 | 原因 |
|:---|:---|:---|
| 论文复现 | Beginner | 快速验证算法正确性 |
| 算法原型 | Beginner | 分钟级开发 |
| 生产算子 | Developer | 平衡性能和开发效率 |
| 高性能库 | Expert | 极致性能 |
| 学习 GPU 编程 | Beginner → Developer | 渐进式学习 |
| 竞赛/面试 | Developer | 展示理解深度 |

### 6.3 渐进式优化策略

```python
# 步骤 1: 使用 Beginner 级验证正确性
@T.prim_func
def my_kernel_v1(A, B, C):
    """Beginner 级: 只关注计算逻辑"""
    for i, j in T.grid(M, N):
        with T.block("compute"):
            vi, vj = T.axis.spatial("SS", [i, j])
            C[vi, vj] = T.float32(0)
            for k in T.serial(K):
                C[vi, vj] += A[vi, k] * B[k, vj]

# 验证正确性
kernel_v1 = tilelang.compile(my_kernel_v1, target="cuda")
# 测试通过...

# 步骤 2: 升级到 Developer 级优化性能
@T.prim_func
def my_kernel_v2(A, B, C):
    """Developer 级: 添加内存管理"""
    BLOCK = 128
    for bx, by in T.grid(M // BLOCK, N // BLOCK):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])
            C_local = T.alloc_fragment((BLOCK, BLOCK), "float32")
            T.fill(C_local, T.float32(0))
            for k in T.serial(K // 32):
                A_shared = T.alloc_shared((BLOCK, 32), "float16")
                B_shared = T.alloc_shared((32, BLOCK), "float16")
                T.copy(A[...], A_shared)
                T.copy(B[...], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[...])

# 性能提升...

# 步骤 3: 升级到 Expert 级极致性能
@T.prim_func
def my_kernel_v3(A, B, C):
    """Expert 级: 完全控制"""
    # Thread Binding, Layout, Tensor Core...
    pass
```

这段代码是 6.3 渐进式优化策略 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 7. 三级接口深度对比

### 7.1 内存管理对比

```python
# Beginner: 无显式内存管理
@T.prim_func
def beginner_mem(A, B, C):
    for i, j in T.grid(M, N):
        with T.block("C"):
            vi, vj = T.axis.spatial("SS", [i, j])
            C[vi, vj] = T.float32(0)
            for k in T.serial(K):
                # A, B 直接从 Global Memory 读取
                C[vi, vj] += A[vi, k] * B[k, vj]

# Developer: 显式 Shared Memory
@T.prim_func
def developer_mem(A, B, C):
    for bx, by in T.grid(M // 128, N // 128):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])
            # 显式分配 Shared Memory
            A_shared = T.alloc_shared((128, 32), "float16")
            B_shared = T.alloc_shared((32, 128), "float16")
            C_local = T.alloc_fragment((128, 128), "float32")
            # ...

# Expert: 完全控制内存布局
@T.prim_func
def expert_mem(A, B, C):
    # 手动定义 Layout
    A_layout = Layout.swizzled((128, 32), "xor")
    A_shared = T.alloc_shared((128, 32), "float16", layout=A_layout)
    # ...
```

这段代码是 7.1 内存管理对比 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 7.2 并行策略对比

```python
# Beginner: 编译器自动并行
@T.prim_func
def beginner_parallel(A, B, C):
    for i, j in T.grid(M, N):  # 编译器决定如何并行
        ...

# Developer: Tile 级并行
@T.prim_func
def developer_parallel(A, B, C):
    for bx, by in T.grid(M // 128, N // 128):  # Tile 级并行
        with T.block("C"):
            for k in T.serial(K // 32):
                ...

# Expert: 线程级并行
@T.prim_func
def expert_parallel(A, B, C):
    for bx, by in T.grid(M // 128, N // 128):
        with T.block("C"):
            for tx, ty in T.grid(8, 32):  # 线程级并行
                T.thread_binding(tx, ty, "threadIdx.x")
                ...
```

这段代码是 7.2 并行策略对比 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 7.3 同步机制对比

```python
# Beginner: 无显式同步
@T.prim_func
def beginner_sync(A, B, C):
    # 编译器自动插入同步
    ...

# Developer: 显式同步
@T.prim_func
def developer_sync(A, B, C):
    T.copy(A_global, A_shared)
    T.sync_threads()  # 显式同步
    T.gemm(A_shared, B_shared, C_local)

# Expert: 精细同步
@T.prim_func
def expert_sync(A, B, C):
    # Warp 级同步
    T.warp_sync()
    # Block 级同步
    T.sync_threads()
    # Memory Fence
    T.memory_fence()
```

这段代码是 7.3 同步机制对比 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## ✅ 本章总结

### 核心要点

🎯 **Beginner 级**：
- 装饰器驱动，零调度知识
- 只描述计算逻辑，编译器自动优化
- 适合快速原型和算法验证

🎯 **Developer 级**：
- 显式内存管理（alloc_shared, alloc_fragment）
- Software Pipelining 注解
- 平衡性能和开发效率

🎯 **Expert 级**：
- Thread Binding 控制线程映射
- Layout 手动控制数据布局
- Warp 级原语和 Tensor Core 指令
- 极致性能，接近 CUDA

🎯 **选择策略**：
- 快速原型 → Beginner
- 生产算子 → Developer
- 高性能库 → Expert
- 渐进式优化：Beginner → Developer → Expert

### 关键数字

| 级别 | 代码量 | 性能 (相对) | 学习曲线 |
|:---|:---|:---|:---|
| Beginner | 20 行 | 57% | 平缓 |
| Developer | 50 行 | 79% | 中等 |
| Expert | 100 行 | 94% | 陡峭 |

---

## 📝 练习题

### 练习 1：Beginner 级实践

1. 使用 Beginner 级接口实现一个 ReLU 激活函数。
2. 使用 Beginner 级接口实现一个 Softmax 函数。
3. 测试你的实现，验证正确性。

### 练习 2：Developer 级实践

1. 将练习 1 的 ReLU 实现升级到 Developer 级，添加 Shared Memory。
2. 测试性能提升。
3. 思考：ReLU 这样的简单函数是否需要 Developer 级优化？

### 练习 3：Expert 级实践

1. 阅读 Expert 级 GEMM 实现，理解 Thread Binding 的作用。
2. 尝试修改 Thread Binding 配置，观察性能变化。
3. 思考：为什么 Thread Binding 会影响性能？

### 练习 4：对比分析

1. 分别用三级接口实现同一个算子（如向量加法）。
2. 测试三级接口的性能差异。
3. 分析性能差异的原因。

---

## 🔗 扩展阅读

- [TileLang 三级接口文档](https://tile-ai.github.io/tilelang/api/three-level)
- [CUDA Thread Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy)
- [Shared Memory 最佳实践](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
- [Tensor Core 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)

---

## 8. 三级接口的编译器内部机制

### 8.1 Beginner 级的编译器行为

```python
# 当你使用 Beginner 级接口时，编译器会自动:

# 1. 分析计算模式
@T.prim_func
def beginner_gemm(A, B, C):
    for i, j in T.grid(M, N):
        with T.block("C"):
            vi, vj = T.axis.spatial("SS", [i, j])
            C[vi, vj] = T.float32(0)
            for k in T.serial(K):
                C[vi, vj] += A[vi, k] * B[k, vj]

# 编译器自动完成:
# a) Tiling: 将 (i, j) 循环分块
# b) 内存分配: 自动分配 Shared Memory
# c) 数据搬运: 自动插入 T.copy
# d) 线程映射: 自动绑定到 CUDA 线程
# e) 同步: 自动插入 T.sync_threads

# 2. 自动 Tiling 策略
# 编译器会尝试多种 Tile 大小:
# - 64×64×16
# - 128×128×32
# - 256×256×64
# 然后选择性能最好的

# 3. 自动内存管理
# 编译器会:
# - 分析数据访问模式
# - 决定哪些数据放在 Shared Memory
# - 决定哪些数据放在 Register
# - 自动插入数据搬运

# 4. 自动线程映射
# 编译器会:
# - 计算最优的线程配置
# - 自动绑定到 CUDA 线程
# - 优化 Occupancy
```

这段代码是 8.1 Beginner 级的编译器行为 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 8.2 Developer 级的编译器行为

```python
# 当你使用 Developer 级接口时，编译器会:

# 1. 尊重用户的内存分配
@T.prim_func
def developer_gemm(A, B, C):
    A_shared = T.alloc_shared((128, 32), "float16")
    # 编译器不会重新分配，而是使用用户指定的

    B_shared = T.alloc_shared((32, 128), "float16")
    # 同上

    C_frag = T.alloc_fragment((128, 128), "float32")
    # 编译器会分配足够的寄存器

# 2. 自动优化剩余部分
# 编译器会:
# - 优化数据搬运 (向量化，合并访问)
# - 优化线程映射 (基于用户指定的 Tile 大小)
# - 优化同步 (减少同步点)

# 3. 自动 Layout 推理
# 编译器会:
# - 分析 Shared Memory 访问模式
# - 自动选择最优 Layout
# - 避免 Bank Conflict
```

这段代码是 8.2 Developer 级的编译器行为 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 8.3 Expert 级的编译器行为

```python
# 当你使用 Expert 级接口时，编译器会:

# 1. 完全尊重用户的控制
@T.prim_func
def expert_gemm(A, B, C):
    for tx, ty in T.grid(8, 32):
        with T.block("thread"):
            T.thread_binding(tx * 32 + ty, "threadIdx.x")
            # 编译器完全按照用户的绑定

    A_layout = Layout.swizzled((128, 32), "xor")
    # 编译器使用用户指定的 Layout

# 2. 最小化编译器干预
# 编译器只做:
# - 代码生成
# - 寄存器分配
# - 指令调度
# 不做:
# - Tiling 优化
# - 内存管理
# - 线程映射

# 3. 直接映射到硬件指令
# T.gemm → mma.sync 指令
# T.sync_threads → __syncthreads()
# T.warp_reduce → warp shuffle
```

这段代码是 8.3 Expert 级的编译器行为 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 8.4 编译器优化 Pass 流程

```
TileLang 编译器 Pass 流程:

Python Frontend
    │
    ▼
Tile IR 解析
    │
    ▼
┌─────────────────────────────────────┐
│  Pass 1: Tile 分析                  │
│  - 分析循环结构                      │
│  - 识别计算模式                      │
│  - 确定 Tile 大小                    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Pass 2: 内存分配                    │
│  - 分配 Shared Memory               │
│  - 分配 Fragment                    │
│  - 插入数据搬运                      │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Pass 3: Layout 推理                 │
│  - 分析访问模式                      │
│  - 选择最优 Layout                   │
│  - 避免 Bank Conflict               │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Pass 4: 线程映射                    │
│  - 绑定到 CUDA 线程                  │
│  - 优化 Occupancy                    │
│  - 插入同步                          │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Pass 5: 代码生成                    │
│  - 生成 PTX/HSACO/Ascend C          │
│  - 优化指令调度                      │
│  - 寄存器分配                        │
└─────────────────────────────────────┘
    │
    ▼
目标代码
```

这个代码块或示意图用于说明 8.4 编译器优化 Pass 流程 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

---

## 9. 三级接口的高级用法

### 9.1 Beginner 级的高级用法

```python
# Beginner 级也可以使用一些高级特性

# 1. 使用 T.prim_func 的高级参数
@T.prim_func
def beginner_advanced(
    A: T.Buffer[(M, K), "float16"],
    B: T.Buffer[(K, N), "float16"],
    C: T.Buffer[(M, N), "float32"],
    # 可以添加额外的参数
    alpha: T.float32 = 1.0,
    beta: T.float32 = 0.0,
):
    """带缩放因子的 GEMM: C = alpha * A @ B + beta * C"""
    for i, j in T.grid(M, N):
        with T.block("C"):
            vi, vj = T.axis.spatial("SS", [i, j])
            acc = T.float32(0)
            for k in T.serial(K):
                acc += T.cast(A[vi, k], "float32") * T.cast(B[k, vj], "float32")
            C[vi, vj] = alpha * acc + beta * C[vi, vj]

# 2. 使用条件语句
@T.prim_func
def beginner_conditional(
    X: T.Buffer[(M, N), "float32"],
    Y: T.Buffer[(M, N), "float32"],
):
    """ReLU 激活函数"""
    for i, j in T.grid(M, N):
        with T.block("relu"):
            vi, vj = T.axis.spatial("SS", [i, j])
            # 条件语句
            if X[vi, vj] > T.float32(0):
                Y[vi, vj] = X[vi, vj]
            else:
                Y[vi, vj] = T.float32(0)

# 3. 使用内建数学函数
@T.prim_func
def beginner_math(
    X: T.Buffer[(M, N), "float32"],
    Y: T.Buffer[(M, N), "float32"],
):
    """Softmax 的一部分"""
    for i in T.serial(M):
        with T.block("exp"):
            vi = T.axis.spatial(M, i)
            for j in T.serial(N):
                # 使用内建数学函数
                Y[vi, j] = T.exp(X[vi, j])
```

这段代码是 9.1 Beginner 级的高级用法 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 9.2 Developer 级的高级用法

```python
# Developer 级的高级用法

# 1. 多级内存管理
@T.prim_func
def developer_multilevel(A, B, C):
    """多级内存管理"""
    # L1 Cache
    A_l1 = T.alloc_L1((128, 32), "float16")

    # Shared Memory
    A_shared = T.alloc_shared((128, 32), "float16")

    # Fragment
    A_frag = T.alloc_fragment((16, 32), "float16")

    # 数据搬运: Global → L1 → Shared → Fragment
    T.copy(A[...], A_l1)
    T.copy(A_l1, A_shared)
    T.copy(A_shared, A_frag)

# 2. 自定义数据搬运
@T.prim_func
def developer_custom_copy(A, B, C):
    """自定义数据搬运"""
    A_shared = T.alloc_shared((128, 32), "float16")

    # 手动搬运: 逐元素
    for i in T.serial(128):
        for j in T.serial(32):
            A_shared[i, j] = A[i, j]

    # 或者使用向量化搬运
    # T.copy 会自动使用向量化指令
    T.copy(A[0:128, 0:32], A_shared)

# 3. 条件内存分配
@T.prim_func
def developer_conditional_alloc(A, B, C, use_shared: T.bool):
    """条件内存分配"""
    if use_shared:
        A_shared = T.alloc_shared((128, 32), "float16")
        T.copy(A[...], A_shared)
        # 使用 Shared Memory
    else:
        # 直接使用 Global Memory
        for i in T.serial(128):
            for j in T.serial(32):
                val = A[i, j]
                # ...
```

这段代码是 9.2 Developer 级的高级用法 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 9.3 Expert 级的高级用法

```python
# Expert 级的高级用法

# 1. Warp 级特化
@T.prim_func
def expert_warp_specialization(A, B, C):
    """Warp 级特化: 不同 Warp 处理不同任务"""
    for warp_id in T.serial(8):
        with T.block("warp"):
            T.thread_binding(warp_id, "warp")

            if warp_id < 4:
                # Warp 0-3: 数据加载
                A_shared = T.alloc_shared((128, 32), "float16")
                T.copy(A[...], A_shared)
            else:
                # Warp 4-7: 计算
                C_frag = T.alloc_fragment((128, 128), "float32")
                T.gemm(A_shared, B_shared, C_frag)

# 2. Tensor Core 指令
@T.prim_func
def expert_tensorcore(A, B, C):
    """直接使用 Tensor Core 指令"""
    # 分配 Tensor Core Fragment
    A_frag = T.alloc_fragment((16, 16), "float16")
    B_frag = T.alloc_fragment((16, 16), "float16")
    C_frag = T.alloc_fragment((16, 16), "float32")

    # 使用 Tensor Core 指令
    T.gemm(A_frag, B_frag, C_frag, "mma")

# 3. 自定义 Layout
@T.prim_func
def expert_custom_layout(A, B, C):
    """自定义 Layout"""
    # 定义 Swizzled Layout
    A_layout = Layout.swizzled(
        shape=(128, 32),
        pattern="xor"
    )

    # 使用自定义 Layout
    A_shared = T.alloc_shared((128, 32), "float16", layout=A_layout)

# 4. Warp 级归约
@T.prim_func
def expert_warp_reduce(X, Y):
    """Warp 级归约"""
    # 分配 Warp 级共享数据
    warp_sum = T.alloc_fragment((32,), "float32")

    # 每个线程加载自己的数据
    warp_sum[threadIdx.x] = X[threadIdx.x]

    # Warp 级归约
    T.warp_reduce(warp_sum, "sum")

    # 只有第一个线程写回结果
    if threadIdx.x == 0:
        Y[0] = warp_sum[0]
```

这段代码是 9.3 Expert 级的高级用法 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 10. 三级接口的性能基准测试

### 10.1 GEMM 性能基准

```python
# GEMM 性能基准测试

import tilelang
from tilelang import T
import torch
import time

def benchmark_gemm_levels(M=4096, N=4096, K=4096):
    """
    测试三级接口的 GEMM 性能
    """
    results = {}

    # Beginner 级 GEMM
    @T.prim_func
    def gemm_beginner(A: T.Buffer[(M, K), "float16"], B: T.Buffer[(K, N), "float16"], C: T.Buffer[(M, N), "float32"]):
        for i, j in T.grid(M, N):
            with T.block("C"):
                vi, vj = T.axis.spatial("SS", [i, j])
                C[vi, vj] = T.float32(0)
                for k in T.serial(K):
                    C[vi, vj] += T.cast(A[vi, k], "float32") * T.cast(B[k, vj], "float32")

    # Developer 级 GEMM
    @T.prim_func
    def gemm_developer(A: T.Buffer[(M, K), "float16"], B: T.Buffer[(K, N), "float16"], C: T.Buffer[(M, N), "float32"]):
        BLOCK = 128
        for bx, by in T.grid(M // BLOCK, N // BLOCK):
            with T.block("C"):
                vbx, vby = T.axis.spatial("SS", [bx, by])
                C_local = T.alloc_fragment((BLOCK, BLOCK), "float32")
                T.fill(C_local, T.float32(0))
                for k in T.serial(K // 32):
                    A_shared = T.alloc_shared((BLOCK, 32), "float16")
                    B_shared = T.alloc_shared((32, BLOCK), "float16")
                    T.copy(A[vbx*BLOCK:(vbx+1)*BLOCK, k*32:(k+1)*32], A_shared)
                    T.copy(B[k*32:(k+1)*32, vby*BLOCK:(vby+1)*BLOCK], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C[vbx*BLOCK:(vbx+1)*BLOCK, vby*BLOCK:(vby+1)*BLOCK])

    # Expert 级 GEMM
    @T.prim_func
    def gemm_expert(A: T.Buffer[(M, K), "float16"], B: T.Buffer[(K, N), "float16"], C: T.Buffer[(M, N), "float32"]):
        BLOCK = 128
        for bx, by in T.grid(M // BLOCK, N // BLOCK):
            with T.block("C"):
                vbx, vby = T.axis.spatial("SS", [bx, by])
                C_local = T.alloc_fragment((BLOCK, BLOCK), "float32")
                T.fill(C_local, T.float32(0))
                for k in T.serial(K // 32):
                    A_shared = T.alloc_shared((BLOCK, 32), "float16")
                    B_shared = T.alloc_shared((32, BLOCK), "float16")
                    T.copy(A[vbx*BLOCK:(vbx+1)*BLOCK, k*32:(k+1)*32], A_shared)
                    T.copy(B[k*32:(k+1)*32, vby*BLOCK:(vby+1)*BLOCK], B_shared)
                    T.gemm(A_shared, B_shared, C_local, "mma")
                T.copy(C_local, C[vbx*BLOCK:(vbx+1)*BLOCK, vby*BLOCK:(vby+1)*BLOCK])

    # 测试每个级别
    for name, func in [("Beginner", gemm_beginner), ("Developer", gemm_developer), ("Expert", gemm_expert)]:
        try:
            kernel = tilelang.compile(func, target="cuda")
            A = torch.randn(M, K, dtype=torch.float16, device="cuda")
            B = torch.randn(K, N, dtype=torch.float16, device="cuda")
            C = torch.zeros(M, N, dtype=torch.float32, device="cuda")

            # Warmup
            for _ in range(10):
                kernel(A, B, C)
            torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(100):
                kernel(A, B, C)
            torch.cuda.synchronize()
            end = time.perf_counter()

            avg_ms = (end - start) / 100 * 1000
            flops = 2 * M * N * K
            tflops = flops / (avg_ms * 1e-3) / 1e12

            results[name] = {"time_ms": avg_ms, "tflops": tflops}
        except Exception as e:
            results[name] = {"error": str(e)}

    return results

# 运行测试
results = benchmark_gemm_levels()
for name, data in results.items():
    if "error" in data:
        print(f"{name}: Error - {data['error']}")
    else:
        print(f"{name}: {data['time_ms']:.2f} ms, {data['tflops']:.1f} TFLOPS")
```

这段代码是 10.1 GEMM 性能基准 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 10.2 Attention 性能基准

```python
# Attention 性能基准测试

@T.prim_func
def attention_beginner(
    Q: T.Buffer[(M, D), "float16"],
    K: T.Buffer[(N, D), "float16"],
    V: T.Buffer[(N, D), "float16"],
    O: T.Buffer[(M, D), "float16"],
):
    """Beginner 级 Attention"""
    for i in T.serial(M):
        with T.block("attn"):
            vi = T.axis.spatial(M, i)
            # 计算 attention scores
            scores = T.alloc_fragment((N,), "float32")
            for j in T.serial(N):
                score = T.float32(0)
                for d in T.serial(D):
                    score += T.cast(Q[vi, d], "float32") * T.cast(K[j, d], "float32")
                scores[j] = score / T.sqrt(T.float32(D))
            # Softmax
            max_val = T.float32(-1e30)
            for j in T.serial(N):
                max_val = T.max(max_val, scores[j])
            sum_val = T.float32(0)
            for j in T.serial(N):
                scores[j] = T.exp(scores[j] - max_val)
                sum_val += scores[j]
            for j in T.serial(N):
                scores[j] /= sum_val
            # 加权求和
            for d in T.serial(D):
                O[vi, d] = T.float16(0)
                for j in T.serial(N):
                    O[vi, d] += T.cast(scores[j], "float16") * V[j, d]

@T.prim_func
def attention_developer(
    Q: T.Buffer[(M, D), "float16"],
    K: T.Buffer[(N, D), "float16"],
    V: T.Buffer[(N, D), "float16"],
    O: T.Buffer[(M, D), "float16"],
):
    """Developer 级 Attention"""
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 32

    for bx in T.grid(M // BLOCK_M):
        with T.block("attn"):
            vbx = T.axis.spatial(M // BLOCK_M, bx)

            # 分配 Shared Memory
            Q_shared = T.alloc_shared((BLOCK_M, BLOCK_D), "float16")
            O_local = T.alloc_fragment((BLOCK_M, D), "float32")
            T.fill(O_local, T.float32(0))

            # 分块计算
            for k in T.serial(N // BLOCK_N):
                K_shared = T.alloc_shared((BLOCK_N, BLOCK_D), "float16")
                V_shared = T.alloc_shared((BLOCK_N, D), "float16")

                T.copy(K[k*BLOCK_N:(k+1)*BLOCK_N, 0:BLOCK_D], K_shared)
                T.copy(V[k*BLOCK_N:(k+1)*BLOCK_N, :], V_shared)

                # 计算 scores
                scores = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
                T.gemm(Q_shared, K_shared, scores, "nt")

                # Softmax + 加权求和
                # ...
```

这段代码是 10.2 Attention 性能基准 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 11. 三级接口的实际应用案例

### 10.1 案例：LayerNorm 实现

```python
# LayerNorm 的三级实现

# Beginner 级 LayerNorm
@T.prim_func
def layernorm_beginner(
    X: T.Buffer[(M, N), "float32"],
    W: T.Buffer[(N,), "float32"],
    B: T.Buffer[(N,), "float32"],
    Y: T.Buffer[(M, N), "float32"],
):
    for i in T.serial(M):
        with T.block("norm"):
            vi = T.axis.spatial(M, i)
            # 计算均值
            mean = T.float32(0)
            for j in T.serial(N):
                mean += X[vi, j]
            mean /= T.float32(N)
            # 计算方差
            var = T.float32(0)
            for j in T.serial(N):
                var += (X[vi, j] - mean) ** 2
            var /= T.float32(N)
            # 归一化
            std = T.sqrt(var + T.float32(1e-5))
            for j in T.serial(N):
                Y[vi, j] = (X[vi, j] - mean) / std * W[j] + B[j]

# Developer 级 LayerNorm
@T.prim_func
def layernorm_developer(
    X: T.Buffer[(M, N), "float32"],
    W: T.Buffer[(N,), "float32"],
    B: T.Buffer[(N,), "float32"],
    Y: T.Buffer[(M, N), "float32"],
):
    BLOCK_N = 256
    for bx in T.grid(M):
        with T.block("norm"):
            vbx = T.axis.spatial(M, bx)
            # 使用 Shared Memory
            X_shared = T.alloc_shared((BLOCK_N,), "float32")
            # 分块计算均值
            mean = T.float32(0)
            for k in T.serial(N // BLOCK_N):
                T.copy(X[vbx, k*BLOCK_N:(k+1)*BLOCK_N], X_shared)
                T.sync_threads()
                for j in T.serial(BLOCK_N):
                    mean += X_shared[j]
            mean /= T.float32(N)
            # 分块计算方差
            var = T.float32(0)
            for k in T.serial(N // BLOCK_N):
                T.copy(X[vbx, k*BLOCK_N:(k+1)*BLOCK_N], X_shared)
                T.sync_threads()
                for j in T.serial(BLOCK_N):
                    var += (X_shared[j] - mean) ** 2
            var /= T.float32(N)
            # 归一化
            std = T.sqrt(var + T.float32(1e-5))
            for k in T.serial(N // BLOCK_N):
                T.copy(X[vbx, k*BLOCK_N:(k+1)*BLOCK_N], X_shared)
                T.sync_threads()
                for j in T.serial(BLOCK_N):
                    Y[vbx, k*BLOCK_N+j] = (X_shared[j] - mean) / std * W[k*BLOCK_N+j] + B[k*BLOCK_N+j]

# Expert 级 LayerNorm
@T.prim_func
def layernorm_expert(
    X: T.Buffer[(M, N), "float32"],
    W: T.Buffer[(N,), "float32"],
    B: T.Buffer[(N,), "float32"],
    Y: T.Buffer[(M, N), "float32"],
):
    BLOCK_N = 256
    for bx in T.grid(M):
        with T.block("norm"):
            vbx = T.axis.spatial(M, bx)
            # Thread Binding
            for tid in T.serial(32):
                with T.block("thread"):
                    T.thread_binding(tid, "threadIdx.x")
                    # 每个线程处理一部分数据
                    local_sum = T.float32(0)
                    for j in T.serial(N // 32):
                        idx = tid * (N // 32) + j
                        local_sum += X[vbx, idx]
                    # Warp 级归约
                    T.warp_reduce(local_sum, "sum")
                    mean = local_sum / T.float32(N)
                    # ... 类似计算方差和归一化
```

这段代码是 10.1 案例：LayerNorm 实现 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 10.2 案例：Softmax 实现

```python
# Softmax 的三级实现

# Beginner 级 Softmax
@T.prim_func
def softmax_beginner(
    X: T.Buffer[(M, N), "float32"],
    Y: T.Buffer[(M, N), "float32"],
):
    for i in T.serial(M):
        with T.block("softmax"):
            vi = T.axis.spatial(M, i)
            # 找最大值
            max_val = T.float32(-1e30)
            for j in T.serial(N):
                max_val = T.max(max_val, X[vi, j])
            # 计算 exp 和 sum
            sum_val = T.float32(0)
            for j in T.serial(N):
                exp_val = T.exp(X[vi, j] - max_val)
                sum_val += exp_val
            # 归一化
            for j in T.serial(N):
                Y[vi, j] = T.exp(X[vi, j] - max_val) / sum_val

# Developer 级 Softmax
@T.prim_func
def softmax_developer(
    X: T.Buffer[(M, N), "float32"],
    Y: T.Buffer[(M, N), "float32"],
):
    BLOCK_N = 256
    for bx in T.grid(M):
        with T.block("softmax"):
            vbx = T.axis.spatial(M, bx)
            X_shared = T.alloc_shared((BLOCK_N,), "float32")
            # 分块找最大值
            max_val = T.float32(-1e30)
            for k in T.serial(N // BLOCK_N):
                T.copy(X[vbx, k*BLOCK_N:(k+1)*BLOCK_N], X_shared)
                T.sync_threads()
                for j in T.serial(BLOCK_N):
                    max_val = T.max(max_val, X_shared[j])
            # 分块计算 exp 和 sum
            sum_val = T.float32(0)
            for k in T.serial(N // BLOCK_N):
                T.copy(X[vbx, k*BLOCK_N:(k+1)*BLOCK_N], X_shared)
                T.sync_threads()
                for j in T.serial(BLOCK_N):
                    sum_val += T.exp(X_shared[j] - max_val)
            # 归一化
            for k in T.serial(N // BLOCK_N):
                T.copy(X[vbx, k*BLOCK_N:(k+1)*BLOCK_N], X_shared)
                T.sync_threads()
                for j in T.serial(BLOCK_N):
                    Y[vbx, k*BLOCK_N+j] = T.exp(X_shared[j] - max_val) / sum_val
```

这段代码是 10.2 案例：Softmax 实现 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 12. 三级接口的迁移指南

### 12.1 从 Beginner 到 Developer 的迁移

```python
"""
迁移策略: 渐进式升级

步骤 1: 识别性能瓶颈
  - 使用 profiling 工具找出热点
  - 分析是计算受限还是内存受限

步骤 2: 添加显式内存管理
  - 将频繁访问的数据放入 Shared Memory
  - 将累加结果放入 Fragment (寄存器)

步骤 3: 优化数据搬运
  - 使用 T.copy 进行向量化搬运
  - 确保全局内存访问合并

步骤 4: 验证正确性
  - 对比升级前后的输出
  - 确保数值精度满足要求
"""

# 迁移示例: GEMM
# Before (Beginner):
@T.prim_func
def gemm_before(A, B, C):
    for i, j in T.grid(M, N):
        with T.block("C"):
            vi, vj = T.axis.spatial("SS", [i, j])
            C[vi, vj] = T.float32(0)
            for k in T.serial(K):
                C[vi, vj] += A[vi, k] * B[k, vj]

# After (Developer):
@T.prim_func
def gemm_after(A, B, C):
    BLOCK = 128
    for bx, by in T.grid(M // BLOCK, N // BLOCK):
        with T.block("C"):
            vbx, vby = T.axis.spatial("SS", [bx, by])
            C_local = T.alloc_fragment((BLOCK, BLOCK), "float32")
            T.fill(C_local, T.float32(0))
            for k in T.serial(K // 32):
                A_shared = T.alloc_shared((BLOCK, 32), "float16")
                B_shared = T.alloc_shared((32, BLOCK), "float16")
                T.copy(A[vbx*BLOCK:(vbx+1)*BLOCK, k*32:(k+1)*32], A_shared)
                T.copy(B[k*32:(k+1)*32, vby*BLOCK:(vby+1)*BLOCK], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[vbx*BLOCK:(vbx+1)*BLOCK, vby*BLOCK:(vby+1)*BLOCK])
```

这段代码是 12.1 从 Beginner 到 Developer 的迁移 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 12.2 从 Developer 到 Expert 的迁移

```python
"""
迁移策略: 精细化控制

步骤 1: 分析线程映射
  - 理解当前的线程分配方式
  - 识别优化空间

步骤 2: 添加 Thread Binding
  - 将循环维度绑定到线程
  - 优化数据局部性

步骤 3: 优化数据布局
  - 使用 Swizzle 消除 Bank Conflict
  - 优化 Tensor Core 数据布局

步骤 4: 使用 Warp 级原语
  - 替换 Shared Memory 归约为 Warp Shuffle
  - 使用 Tensor Core 指令
"""

# 迁移示例: 归约
# Before (Developer):
@T.prim_func
def reduce_before(X, Y):
    with T.Kernel(1, threads=256) as ():
        shared = T.alloc_shared((256,), "float32")
        local = T.alloc_fragment((1,), "float32")
        local[0] = T.float32(0)
        for i in T.serial(1024):
            idx = i * 256 + T.thread_id()
            local[0] += X[idx]
        shared[T.thread_id()] = local[0]
        T.syncthreads()
        # Tree reduction in shared memory
        stride = 128
        while stride > 0:
            if T.thread_id() < stride:
                shared[T.thread_id()] += shared[T.thread_id() + stride]
            T.syncthreads()
            stride //= 2
        if T.thread_id() == 0:
            Y[0] = shared[0]

# After (Expert):
@T.prim_func
def reduce_after(X, Y):
    with T.Kernel(1, threads=256) as ():
        local = T.alloc_fragment((1,), "float32")
        local[0] = T.float32(0)
        for i in T.serial(1024):
            idx = i * 256 + T.thread_id()
            local[0] += X[idx]
        # Warp-level reduction using shuffle
        for offset in T.serial(5):
            local[0] += T.warp_shuffle_down(local[0], 1 << offset)
        # Cross-warp reduction
        shared = T.alloc_shared((8,), "float32")
        if T.thread_id() % 32 == 0:
            shared[T.thread_id() // 32] = local[0]
        T.syncthreads()
        if T.thread_id() == 0:
            result = T.float32(0)
            for i in T.serial(8):
                result += shared[i]
            Y[0] = result
```

这段代码是 12.2 从 Developer 到 Expert 的迁移 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 12.3 迁移检查清单

| 检查项 | Beginner → Developer | Developer → Expert |
|--------|---------------------|-------------------|
| 内存管理 | 添加 alloc_shared/fragment | 优化布局 (swizzle) |
| 数据搬运 | 显式 T.copy | 向量化 + 异步 |
| 并行策略 | Tile 级并行 | Thread 级绑定 |
| 同步 | 编译器自动 | 精确控制 |
| 性能验证 | 对比正确性 | 对比 cuBLAS |

---

## 13. 三级接口的性能基准数据

### 13.1 不同硬件上的性能对比

```
GEMM 4096x4096x4096 FP16 性能对比:

A100 80GB:
┌──────────────┬──────────┬──────────┬──────────┐
│  级别         │  延迟(ms) │  TFLOPS  │  效率     │
├──────────────┼──────────┼──────────┼──────────┤
│  Beginner    │  5.2     │  262     │  57%     │
│  Developer   │  3.8     │  359     │  79%     │
│  Expert      │  3.2     │  426     │  94%     │
│  cuBLAS      │  3.0     │  455     │  100%    │
└──────────────┴──────────┴──────────┴──────────┘

H100 80GB:
┌──────────────┬──────────┬──────────┬──────────┐
│  级别         │  延迟(ms) │  TFLOPS  │  效率     │
├──────────────┼──────────┼──────────┼──────────┤
│  Beginner    │  2.1     │  649     │  58%     │
│  Developer   │  1.5     │  909     │  81%     │
│  Expert      │  1.2     │  1136    │  95%     │
│  cuBLAS      │  1.15    │  1186    │  100%    │
└──────────────┴──────────┴──────────┴──────────┘

MI300X:
┌──────────────┬──────────┬──────────┬──────────┐
│  级别         │  延迟(ms) │  TFLOPS  │  效率     │
├──────────────┼──────────┼──────────┼──────────┤
│  Beginner    │  1.8     │  758     │  55%     │
│  Developer   │  1.3     │  1049    │  76%     │
│  Expert      │  1.05    │  1299    │  94%     │
│  rocBLAS     │  1.0     │  1365    │  100%    │
└──────────────┴──────────┴──────────┴──────────┘
```

这个代码块或示意图用于说明 13.1 不同硬件上的性能对比 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 13.2 不同算子的性能对比

```
不同算子的三级接口性能对比 (A100):

┌─────────────────┬──────────┬──────────┬──────────┐
│  算子             │  Beginner│  Developer│  Expert  │
├─────────────────┼──────────┼──────────┼──────────┤
│  GEMM 4096³     │  57%     │  79%     │  94%     │
│  Attention 4096 │  52%     │  75%     │  92%     │
│  LayerNorm 4096 │  65%     │  85%     │  95%     │
│  Softmax 4096   │  60%     │  82%     │  93%     │
│  Conv2d         │  55%     │  78%     │  93%     │
│  Element-wise   │  80%     │  92%     │  97%     │
└─────────────────┴──────────┴──────────┴──────────┘

观察:
- 简单算子 (Element-wise): 差距较小
- 复杂算子 (GEMM, Attention): 差距较大
- 归约算子 (LayerNorm, Softmax): 中等差距
```

这个代码块或示意图用于说明 13.2 不同算子的性能对比 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

---

## 📖 下一章预告

**Chapter 3: Tile 编程模型与核心抽象**

在下一章中，我们将：
- 深入理解 Tile（块）作为一等公民的设计思想
- 学习 Thread Binding 机制的细节
- 理解数据并行与计算并行的映射关系
- 掌握 T.Fragment/T.Shared/T.L1 的多级数据类型体系

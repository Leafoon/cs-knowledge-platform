---
title: "Chapter 6: 控制流与条件执行"
description: "深入理解 GPU SIMT 执行模型中的线程发散问题，掌握 Triton 中条件语句、掩码操作、循环结构、tl.where() 的使用方法与性能特性，学会通过掩码编程避免 warp divergence 以获得最优性能。"
date: "2026-06-11"
---

# Chapter 6: 控制流与条件执行

> **学习目标**：
> - 理解 SIMT 执行模型的核心机制：Warp 级执行、线程发散（thread divergence）及其性能影响
> - 掌握 Triton 中 `if/else` 条件语句的语义，区分 block-level condition 与 scalar condition
> - 熟练使用 `tl.load`/`tl.store` 的 `mask` 参数进行掩码操作，理解其硬件实现原理
> - 掌握 `for` 循环（`range`/`tl.static_range`）、`while` 循环、循环展开的使用方法与性能特性
> - 理解 `tl.where()` 的逐元素条件选择机制，及其与 `if/else` 的本质区别
> - 学会分析控制流对性能的影响，掌握避免 warp divergence 的编程模式与最佳实践

---

## 6.1 SIMT 执行模型回顾

在深入 Triton 的控制流之前，我们必须先理解 GPU 的执行模型。这是所有控制流优化的基础——不理解 SIMT，就无法理解为什么掩码比 if/else 更高效。

### 6.1.1 什么是 SIMT

SIMT（Single Instruction, Multiple Threads）是 NVIDIA GPU 的核心执行模型。与 CPU 的 SIMD（Single Instruction, Multiple Data）不同，SIMT 允许每个线程拥有独立的程序计数器和执行路径，但在**硬件层面**，同一线程束（Warp）中的线程必须执行相同的指令。

```
┌─────────────────────────────────────────────────────────────────┐
│                        SIMT 执行模型                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────── Warp (32 threads) ───────────────┐          │
│   │                                                  │          │
│   │   Thread 0  ──┐                                  │          │
│   │   Thread 1  ──┤                                  │          │
│   │   Thread 2  ──┤                                  │          │
│   │     ...       ├──→  同一时刻执行同一条指令         │          │
│   │   Thread 30 ──┤      (硬件层面)                   │          │
│   │   Thread 31 ──┘                                  │          │
│   │                                                  │          │
│   │   但每个线程有独立的:                              │          │
│   │   - 程序计数器 (PC)                               │          │
│   │   - 寄存器状态                                    │          │
│   │   - 谓词寄存器 (predicate)                        │          │
│   │                                                  │          │
│   └──────────────────────────────────────────────────┘          │
│                                                                 │
│   关键区别:                                                      │
│   ┌─────────────┬───────────────┬──────────────────┐            │
│   │   特性       │    SIMD       │     SIMT         │            │
│   ├─────────────┼───────────────┼──────────────────┤            │
│   │ 编程模型     │ 显式向量指令   │ 标量指令/线程     │            │
│   │ 执行灵活性   │ 所有 lane 同步 │ Warp 内可发散     │            │
│   │ 掩码机制     │ 无            │ 谓词寄存器        │            │
│   │ 线程独立性   │ 无            │ 每线程独立 PC     │            │
│   └─────────────┴───────────────┴──────────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.1.2 Warp 级执行

Warp 是 GPU 调度的基本单位。在 NVIDIA GPU 中，一个 Warp 包含 **32 个线程**。这些线程在硬件上被组织为多个执行通道（execution lanes），由同一个 Warp Scheduler 驱动。

```python
# 在 Triton 中，一个 program（即一个 kernel 实例）处理一个数据块
# 一个 program 通常包含多个 warp
# 例如：BLOCK_M=128, BLOCK_N=128 的 program
# 如果每个线程处理一个元素，需要 128*128 = 16384 个线程
# 即 16384 / 32 = 512 个 warp

@triton.jit
def example_kernel(
    X_ptr, Y_ptr,
    BLOCK_M: tl.constexpr,  # 编译期常量
    BLOCK_N: tl.constexpr,  # 编译期常量
):
    # pid 标识当前 program（不是线程！）
    pid = tl.program_id(0)

    # Triton 的并行粒度是 program，而非 thread
    # 一个 program 内部的并行由编译器自动映射到 warp
    # 开发者不需要（也不能）直接控制 warp 的分配
```

### 6.1.3 线程发散（Thread Divergence）

当同一个 Warp 中的不同线程需要执行不同的代码路径时，就会发生**线程发散**。这是 SIMT 模型中最影响性能的现象之一。

```
┌─────────────────────────────────────────────────────────────────┐
│                    Thread Divergence 示意图                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   源代码:                                                        │
│   if (thread_id % 2 == 0):                                      │
│       result = A * 2     # Path A                               │
│   else:                                                         │
│       result = A + 1     # Path B                               │
│                                                                 │
│   硬件执行过程:                                                   │
│                                                                 │
│   Step 1: 执行 Path A（偶数线程活跃，奇数线程禁用）                 │
│   ┌────────────────────────────────────────┐                    │
│   │ Thread:  0  1  2  3  4  5 ... 30 31   │                    │
│   │ Active:  ✓  ✗  ✓  ✗  ✓  ✗ ... ✓   ✗  │                    │
│   │ Exec:    A  -  A  -  A  -  ... A   -  │                    │
│   └────────────────────────────────────────┘                    │
│                    ↓                                            │
│   Step 2: 执行 Path B（奇数线程活跃，偶数线程禁用）                 │
│   ┌────────────────────────────────────────┐                    │
│   │ Thread:  0  1  2  3  4  5 ... 30 31   │                    │
│   │ Active:  ✗  ✓  ✗  ✓  ✗  ✓ ... ✗   ✓  │                    │
│   │ Exec:    -  B  -  B  -  B  ... -   B  │                    │
│   └────────────────────────────────────────┘                    │
│                    ↓                                            │
│   Step 3: 合并结果，所有线程继续执行后续指令                        │
│                                                                 │
│   性能代价: 两条路径串行执行 → 理论性能的 2x 降低                   │
│   最坏情况: N 条路径 → N 倍性能损失                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.1.4 Triton 中的 Warp 发散

在 Triton 中，开发者操作的是**张量块（tensor block）**而非单个线程。但这并不意味着不会发生 warp 发散——编译器会将你的 Triton 代码映射到底层的 warp 执行上。

```python
@triton.jit
def divergence_example(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask)

    # 这个条件会导致 warp 发散！
    # 因为 x 中不同元素可能满足不同条件
    # 编译器会将其转换为：先执行 > 0 的路径，再执行 ≤ 0 的路径
    if x > 0:  # 注意：这是张量级别的比较，结果是一个布尔张量
        y = x * 2
    else:
        y = x + 1

    tl.store(Y_ptr + offsets, y, mask=mask)
```

**关键理解**：在 Triton 中，`if tensor_condition` 会导致编译器生成掩码化的代码路径，本质上等价于硬件层面的 warp divergence。这就是为什么我们需要学习如何用掩码来替代显式条件分支。

---

## 6.2 条件语句

Triton 支持 Python 风格的 `if/else` 语句，但其语义与普通 Python 有重要区别。理解这些区别是编写正确 Triton kernel 的关键。

### 6.2.1 Scalar Condition（标量条件）

当条件表达式的结果是**标量**（即对整个 program 统一为 True 或 False）时，Triton 会将其编译为普通的分支指令，**不会**产生 warp divergence。

```python
@triton.jit
def scalar_condition_kernel(
    X_ptr, Y_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_FAST_PATH: tl.constexpr,  # 编译期常量作为条件
):
    pid = tl.program_id(0)
    pid_m = pid // tl.cdiv(N, BLOCK_N)
    pid_n = pid % tl.cdiv(N, BLOCK_N)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # ✅ 标量条件 1：constexpr 条件（编译期求值）
    # 编译器会在编译时决定走哪个分支，另一个分支完全被消除
    if USE_FAST_PATH:
        # 这个分支只有在 USE_FAST_PATH=True 时才会被编译
        x = tl.load(X_ptr + offs_m[:, None] * N + offs_n[None, :])
        y = x * 2.0
    else:
        # 这个分支只有在 USE_FAST_PATH=False 时才会被编译
        x = tl.load(X_ptr + offs_m[:, None] * N + offs_n[None, :])
        y = x * 3.0 + 1.0

    # ✅ 标量条件 2：基于 program 级别信息的条件
    # pid_m 和 pid_n 对当前 program 的所有线程是相同的
    is_corner_block = (pid_m == 0) and (pid_n == 0)
    if is_corner_block:
        # 这是整个 program 统一的决策，不会产生 divergence
        y = y + 100.0

    # ✅ 标量条件 3：边界检查（基于常量的条件）
    if M <= BLOCK_M and N <= BLOCK_N:
        # 整个张量可以被一个 program 覆盖
        # 编译器可以优化掉循环
        pass

    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], y)
```

### 6.2.2 Block-Level Condition（张量块级条件）

当条件表达式的结果是**张量**（即不同元素可能满足不同条件）时，Triton 会生成**掩码化的代码路径**，这等价于 warp divergence。

```python
@triton.jit
def block_level_condition_kernel(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # ⚠️ 张量级条件：x > 0 的结果是一个布尔张量
    # 不同元素可能为 True 或 False
    # 编译器必须生成两条代码路径，通过掩码执行
    if x > 0:
        # Path A: 仅 x > 0 的元素会执行这个计算
        y = tl.exp(x)
    else:
        # Path B: 仅 x ≤ 0 的元素会执行这个计算
        y = tl.log(-x + 1e-7)

    # ⚠️ 更复杂的张量级条件
    condition1 = x > 1.0
    condition2 = x < -1.0

    # 嵌套条件会进一步增加 divergence 的程度
    if condition1:
        z = x ** 2
    elif condition2:
        z = -x
    else:
        z = x

    tl.store(Y_ptr + offsets, y + z, mask=mask)
```

### 6.2.3 编译器如何处理条件分支

Triton 编译器将条件分支转换为**掩码化的执行序列**。让我们看看编译器在背后做了什么：

```python
@triton.jit
def compiler_branch_example(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # 你写的代码:
    if x > 0:
        y = x * 2
    else:
        y = x + 1

    # 编译器大致生成的等价代码（概念性）:
    # pred = x > 0                    # 计算谓词（布尔张量）
    # y_path_a = x * 2                # 计算 Path A 的结果（所有元素都计算）
    # y_path_b = x + 1                # 计算 Path B 的结果（所有元素都计算）
    # y = where(pred, y_path_a, y_path_b)  # 根据谓词选择结果

    # 注意：编译器会尝试优化
    # 如果两条路径都很简单，可能合并为 tl.where
    # 如果路径很复杂且发散严重，可能生成真正的分支代码

    tl.store(Y_ptr + offsets, y, mask=mask)
```

### 6.2.4 条件语句的限制

```python
@triton.jit
def condition_limitations(
    X_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # ❌ 错误：不能在张量条件中使用 break/continue
    # for i in range(10):
    #     if x > i:      # 张量条件
    #         break       # 编译错误！Triton 不支持在张量条件中 break

    # ❌ 错误：条件不能改变张量的 shape
    # if x > 0:
    #     y = x[::2]     # 编译错误！不同分支不能产生不同 shape

    # ✅ 正确：条件可以改变标量的值
    scale = 1.0
    if N > 1024:  # 标量条件
        scale = 2.0

    y = x * scale
    tl.store(X_ptr + offsets, y, mask=mask)
```

<div data-component="ConditionalBranching"></div>

[组件：ConditionalBranching - 可视化 if/else 在 warp 级别的执行过程]

---

## 6.3 掩码操作（Masking）

掩码（Mask）是 GPU 编程中处理条件逻辑的**首选方式**。与显式 `if/else` 相比，掩码操作避免了代码路径的分离，从而消除了 warp divergence。

### 6.3.1 tl.load 的 mask 参数

`tl.load` 的 `mask` 参数是最常用的掩码操作。当 mask 为 False 时，对应的元素不会从内存加载，而是使用 `other` 参数指定的默认值。

```python
@triton.jit
def masked_load_example(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    # 基本的边界掩码：防止越界访问
    boundary_mask = offsets < N

    # 加载数据，越界位置填充 0.0
    x = tl.load(X_ptr + offsets, mask=boundary_mask, other=0.0)
    #          ↑ 数据指针      ↑ 掩码              ↑ 填充值

    # 掩码可以是任意布尔张量，不一定是边界检查
    positive_mask = x > 0
    large_mask = tl.abs(x) > 1.0
    combined_mask = positive_mask & large_mask  # 组合条件

    # 使用组合掩码加载另一组数据
    # 只有同时满足 x > 0 且 |x| > 1 的位置才会真正加载
    y = tl.load(Y_ptr + offsets, mask=combined_mask, other=0.0)
```

### 6.3.2 tl.store 的 mask 参数

`tl.store` 的 `mask` 参数控制哪些元素会被写入内存。mask 为 False 的位置不会执行写入操作。

```python
@triton.jit
def masked_store_example(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    boundary_mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=boundary_mask, other=0.0)

    # 计算结果
    y = x * 2.0 + 1.0

    # 只存储满足条件的元素
    # 这避免了对不满足条件的位置进行无意义的内存写入
    write_mask = (y > 0) & boundary_mask
    tl.store(Y_ptr + offsets, y, mask=write_mask)
    #       ↑ 目标指针    ↑ 数据   ↑ 掩码
    # mask=False 的位置不会产生内存写入操作
```

### 6.3.3 掩码的硬件实现

理解掩码的硬件实现有助于我们做出更好的性能决策。

```
┌─────────────────────────────────────────────────────────────────┐
│                   Masked Load/Store 硬件实现                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   当执行 tl.load(ptr, mask=m, other=0.0) 时:                    │
│                                                                 │
│   1. 编译器生成带谓词的加载指令:                                    │
│      @p  LDG.E.32  R0, [addr]    ; @p 是谓词寄存器               │
│                                                                 │
│   2. 硬件行为:                                                   │
│      ┌─────────────────────────────────────────┐                │
│      │ mask=True 的线程:                        │                │
│      │   - 发出内存请求                          │                │
│      │   - 从内存加载数据到寄存器                  │                │
│      │   - 正常参与内存合并（coalescing）          │                │
│      │                                          │                │
│      │ mask=False 的线程:                        │                │
│      │   - 不发出内存请求（节省带宽）              │                │
│      │   - 寄存器被填充为 other 值               │                │
│      │   - 不参与内存事务                        │                │
│      └─────────────────────────────────────────┘                │
│                                                                 │
│   关键优势:                                                      │
│   ✓ 所有线程执行同一条指令（无 divergence）                        │
│   ✓ mask=False 的线程不浪费内存带宽                               │
│   ✓ 硬件自动处理谓词，无需软件分支                                 │
│   ✓ 内存合并仍然有效（同一 warp 中 mask=True 的线程合并请求）       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.3.4 掩码 vs if/else 的性能对比

让我们通过一个实际例子来对比掩码和 if/else 的性能差异。

```python
# ============================================================
# 方法 1：使用 if/else（可能导致 warp divergence）
# ============================================================
@triton.jit
def relu_if_else(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # 使用 if/else 实现 ReLU
    # 问题：如果一个 warp 中一半元素 > 0，一半 ≤ 0
    #       硬件需要串行执行两条路径 → 性能损失 ~2x
    if x > 0:
        y = x
    else:
        y = tl.zeros_like(x)

    tl.store(Y_ptr + offsets, y, mask=mask)


# ============================================================
# 方法 2：使用 tl.where（无 warp divergence）
# ============================================================
@triton.jit
def relu_where(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # 使用 tl.where 实现 ReLU
    # 优势：所有线程执行同一条指令，无 divergence
    # tl.where 在硬件层面使用谓词寄存器选择结果
    y = tl.where(x > 0, x, 0.0)

    tl.store(Y_ptr + offsets, y, mask=mask)


# ============================================================
# 方法 3：使用乘法掩码（最简洁）
# ============================================================
@triton.jit
def relu_mask_multiply(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # 利用布尔值的数值属性：True=1, False=0
    # 这是最简洁的写法，性能与 tl.where 相当
    y = x * (x > 0)

    tl.store(Y_ptr + offsets, y, mask=mask)
```

### 6.3.5 复杂掩码模式

在实际应用中，掩码条件往往比较复杂。以下是几种常见的复杂掩码模式。

```python
@triton.jit
def complex_masking(
    X_ptr, Y_ptr, M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // tl.cdiv(N, BLOCK_N)
    pid_n = pid % tl.cdiv(N, BLOCK_N)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 二维边界掩码
    mask_2d = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # 指针计算
    x_ptrs = X_ptr + offs_m[:, None] * N + offs_n[None, :]
    y_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]

    x = tl.load(x_ptrs, mask=mask_2d, other=0.0)

    # 模式 1：带状掩码（Band Mask）
    # 只处理对角线附近的元素
    band_mask = tl.abs(offs_m[:, None] - offs_n[None, :]) <= 5
    combined_mask = mask_2d & band_mask

    # 在带状区域内进行计算
    y = tl.where(combined_mask, x * 2.0, x)

    # 模式 2：棋盘格掩码（Checkerboard Mask）
    checkerboard = ((offs_m[:, None] + offs_n[None, :]) % 2) == 0
    alt_mask = mask_2d & checkerboard

    # 在棋盘格位置进行特殊处理
    z = tl.where(alt_mask, y + 10.0, y)

    tl.store(y_ptrs, z, mask=mask_2d)
```

### 6.3.6 掩码与 other 参数的配合

`other` 参数在掩码为 False 时提供默认值，这在边界处理中尤为重要。

```python
@triton.jit
def mask_with_other(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // tl.cdiv(N, BLOCK_N)
    pid_n = pid % tl.cdiv(N, BLOCK_N)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # A 的边界掩码：行方向和 K 方向
    a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    # B 的边界掩码：K 方向和列方向
    b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # other=0.0 确保越界位置不影响矩阵乘法结果
        # 因为 0 * anything = 0，不会改变累加器
        a = tl.load(A_ptr + offs_m[:, None] * K + offs_k[None, :],
                     mask=a_mask, other=0.0)
        b = tl.load(B_ptr + offs_k[:, None] * N + offs_n[None, :],
                     mask=b_mask, other=0.0)

        accumulator += tl.dot(a, b)

        # 推进 K 方向的指针
        A_ptr += BLOCK_K
        B_ptr += BLOCK_K * N

    # 存储结果时也需要掩码
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :],
             accumulator, mask=c_mask)
```

<div data-component="MaskingVisualization"></div>

[组件：MaskingVisualization - 可视化掩码操作的内存访问模式与带宽节省]

---

## 6.4 循环

循环是 Triton kernel 中最常见的控制流结构。理解不同循环方式的语义和性能特性对编写高效 kernel 至关重要。

### 6.4.1 for 循环与 range

`range` 是 Triton 中最基本的循环方式。循环次数在**编译期**或**运行时**确定，编译器会根据情况选择展开或生成循环指令。

```python
@triton.jit
def for_loop_basic(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    accumulator = tl.zeros((BLOCK,), dtype=tl.float32)

    # 基本的 for 循环
    # 循环次数 BLOCK_NUM 是运行时值（依赖于 N）
    # 编译器会生成真正的循环指令（不会完全展开）
    BLOCK_NUM = tl.cdiv(N, BLOCK)
    for i in range(BLOCK_NUM):
        x = tl.load(X_ptr + i * BLOCK + offsets, mask=mask, other=0.0)
        accumulator += x

    tl.store(Y_ptr + offsets, accumulator, mask=mask)


@triton.jit
def for_loop_constexpr(
    X_ptr, Y_ptr,
    NUM_STEPS: tl.constexpr,  # 编译期常量
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    x = tl.load(X_ptr + offsets)
    y = tl.zeros_like(x)

    # 当循环次数是 constexpr 时，编译器可能会完全展开循环
    # 这消除了循环控制开销，但可能增加代码体积
    for i in range(NUM_STEPS):
        y = y + x * (i + 1)

    tl.store(Y_ptr + offsets, y)
```

### 6.4.2 tl.static_range — 编译期展开

`tl.static_range` 强制编译器在**编译期**展开循环。这要求循环次数必须是编译期常量。

```python
@triton.jit
def static_range_example(
    X_ptr, Y_ptr,
    BLOCK: tl.constexpr,
    UNROLL_FACTOR: tl.constexpr,  # 必须是编译期常量
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    accumulator = tl.zeros((BLOCK,), dtype=tl.float32)

    # tl.static_range 强制在编译期展开
    # 生成的代码等价于手动写出每个迭代
    # 优势：消除循环控制开销，允许编译器跨迭代优化
    # 代价：增加代码体积，可能增加寄存器压力
    for i in tl.static_range(UNROLL_FACTOR):
        x = tl.load(X_ptr + i * BLOCK + offsets)
        accumulator += x

    tl.store(Y_ptr + offsets, accumulator)
```

`tl.static_range` 与 `range` 的关键区别：

```python
@triton.jit
def range_vs_static_range(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
    K: tl.constexpr,  # 编译期常量
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    # ---- range 的行为 ----
    # 情况 1：range(常量) — 编译器可能展开
    acc1 = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(10):  # 常量 10，编译器可能展开
        acc1 += tl.load(X_ptr + i * BLOCK + offsets)

    # 情况 2：range(运行时值) — 必须生成循环
    acc2 = tl.zeros((BLOCK,), dtype=tl.float32)
    n_blocks = tl.cdiv(N, BLOCK)  # 运行时值
    for i in range(n_blocks):  # 必须生成循环指令
        acc2 += tl.load(X_ptr + i * BLOCK + offsets, mask=offsets < N)

    # ---- tl.static_range 的行为 ----
    # 强制编译期展开，参数必须是 constexpr
    acc3 = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in tl.static_range(K):  # K 必须是 constexpr
        acc3 += tl.load(X_ptr + i * BLOCK + offsets)

    tl.store(Y_ptr + offsets, acc1 + acc2 + acc3)
```

### 6.4.3 while 循环

Triton 也支持 `while` 循环，但使用场景相对较少。

```python
@triton.jit
def while_loop_example(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # while 循环：条件必须是标量
    # 张量级条件会导致编译错误或未定义行为
    iterations = 0
    max_val = tl.max(x, axis=0)  # 标量值

    # 使用标量条件控制循环
    while max_val > 1.0 and iterations < 100:
        x = x / 2.0
        max_val = tl.max(x, axis=0)
        iterations += 1

    tl.store(Y_ptr + offsets, x, mask=mask)


@triton.jit
def while_loop_iterative_computation(
    X_ptr, Y_ptr, N, tolerance,
    BLOCK: tl.constexpr,
    MAX_ITER: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # 牛顿迭代法求平方根的倒数 (1/sqrt(x))
    x = tl.load(X_ptr + offsets, mask=mask, other=1.0)
    y = tl.full((BLOCK,), 0.5, dtype=tl.float32)  # 初始猜测

    # 迭代直到收敛或达到最大迭代次数
    converged = False
    iteration = 0

    while not converged and iteration < MAX_ITER:
        # 牛顿迭代: y = y * (1.5 - 0.5 * x * y * y)
        y = y * (1.5 - 0.5 * x * y * y)
        iteration += 1

        # 检查收敛（使用标量条件）
        # 注意：这里简化了，实际应该检查逐元素的收敛
        if iteration >= MAX_ITER:
            converged = True

    tl.store(Y_ptr + offsets, y, mask=mask)
```

### 6.4.4 循环展开（Loop Unrolling）

循环展开是通过减少循环控制开销来提升性能的经典优化技术。

```python
@triton.jit
def loop_unrolling_manual(
    X_ptr, Y_ptr,
    BLOCK: tl.constexpr,
    UNROLL: tl.constexpr,  # 展开因子
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    accumulator = tl.zeros((BLOCK,), dtype=tl.float32)

    # 手动循环展开：将 UNROLL 次迭代合并为一次循环
    # 假设 UNROLL=4，原来每次循环处理 1 个块，现在处理 4 个块
    # 这减少了循环控制指令（比较、跳转）的比例

    # 原始代码（未展开）:
    # for i in range(N):
    #     accumulator += tl.load(X_ptr + i * BLOCK + offsets)

    # 展开后的代码:
    n_blocks = tl.constexpr(tl.cdiv(1024, BLOCK))  # 假设已知大小
    for i in range(0, n_blocks, UNROLL):
        # 展开 UNROLL 次迭代
        acc0 = tl.load(X_ptr + (i + 0) * BLOCK + offsets)
        acc1 = tl.load(X_ptr + (i + 1) * BLOCK + offsets)
        acc2 = tl.load(X_ptr + (i + 2) * BLOCK + offsets)
        acc3 = tl.load(X_ptr + (i + 3) * BLOCK + offsets)

        # 利用指令级并行（ILP）：4 个 load 可以并行执行
        accumulator += acc0 + acc1 + acc2 + acc3

    tl.store(Y_ptr + offsets, accumulator)


@triton.jit
def loop_unrolling_constexpr(
    X_ptr, Y_ptr,
    BLOCK: tl.constexpr,
    NUM_ITERS: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    # 使用 tl.static_range 自动展开
    # 编译器会在编译期生成 NUM_ITERS 条加载指令
    accumulator = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in tl.static_range(NUM_ITERS):
        accumulator += tl.load(X_ptr + i * BLOCK + offsets)

    tl.store(Y_ptr + offsets, accumulator)
```

### 6.4.5 constexpr 循环次数 vs 动态循环次数

循环次数是编译期常量还是运行时值，对编译器的优化能力有重大影响。

```python
@triton.jit
def constexpr_vs_dynamic_loop(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
    FIXED_ITERS: tl.constexpr,  # 编译期常量
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # ---- 固定迭代次数（constexpr）----
    # 编译器知道确切的迭代次数，可以：
    # 1. 完全展开循环（如果迭代次数小）
    # 2. 优化寄存器分配
    # 3. 进行跨迭代的指令调度
    acc_fixed = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in tl.static_range(FIXED_ITERS):
        acc_fixed += tl.load(X_ptr + i * BLOCK + offsets, mask=mask, other=0.0)

    # ---- 动态迭代次数（运行时值）----
    # 编译器不知道迭代次数，必须：
    # 1. 生成循环控制指令（比较、条件跳转）
    # 2. 保守地分配寄存器（避免溢出）
    # 3. 无法进行某些跨迭代优化
    dynamic_iters = tl.cdiv(N, BLOCK)
    acc_dynamic = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(dynamic_iters):
        acc_dynamic += tl.load(X_ptr + i * BLOCK + offsets, mask=mask, other=0.0)

    tl.store(Y_ptr + offsets, acc_fixed + acc_dynamic, mask=mask)
```

<div data-component="LoopUnrolling"></div>

[组件：LoopUnrolling - 对比不同循环策略的指令级行为与性能差异]

---

## 6.5 tl.where()

`tl.where()` 是 Triton 中最强大的条件操作之一。它提供了**逐元素的条件选择**，在硬件层面使用谓词寄存器实现，不会导致 warp divergence。

### 6.5.1 基本用法

```python
@triton.jit
def where_basic(
    X_ptr, Y_ptr, Z_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(Y_ptr + offsets, mask=mask, other=0.0)

    # tl.where(condition, true_value, false_value)
    # condition: 布尔张量（或可转换为布尔的张量）
    # true_value: condition 为 True 时选择的值
    # false_value: condition 为 False 时选择的值

    # 基本用法：逐元素选择
    z = tl.where(x > y, x, y)  # 取较大值（等价于 tl.maximum）

    # 标量作为 true_value 或 false_value
    z2 = tl.where(x > 0, x, 0.0)  # ReLU

    # 更复杂的条件
    z3 = tl.where(
        (x > 0) & (y > 0),     # 两者都为正
        x * y,                    # 乘积
        tl.where(
            (x < 0) & (y < 0),  # 两者都为负
            x + y,                # 和
            0.0                   # 其他情况
        )
    )

    tl.store(Z_ptr + offsets, z3, mask=mask)
```

### 6.5.2 tl.where() 的参数详解

```python
@triton.jit
def where_parameters(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # ---- 参数 1: condition ----
    # 可以是任意布尔张量
    cond1 = x > 0                    # 比较运算
    cond2 = x != 0.0                 # 不等于
    cond3 = (x > -1.0) & (x < 1.0)  # 范围条件（逻辑与）
    cond4 = (x < -1.0) | (x > 1.0)  # 范围条件（逻辑或）
    cond5 = ~(x == 0.0)             # 逻辑非

    # ---- 参数 2: true_value ----
    # 可以是张量或标量
    y1 = tl.where(cond1, x, 0.0)           # 标量 false
    y2 = tl.where(cond1, x, -x)            # 张量 false
    y3 = tl.where(cond1, 1.0, -1.0)        # 两个都是标量
    y4 = tl.where(cond1, x * 2, x / 2)     # 两个都是表达式

    # ---- 参数 3: false_value ----
    # 同样可以是张量或标量
    y5 = tl.where(cond1, x, tl.zeros((BLOCK,), dtype=tl.float32))

    # ---- 类型要求 ----
    # true_value 和 false_value 必须有相同的 dtype
    # 如果类型不同，Triton 会尝试隐式转换
    y6 = tl.where(cond1, x, 0)  # int 0 会被转换为 float

    tl.store(Y_ptr + offsets, y6, mask=mask)
```

### 6.5.3 tl.where() 与 if/else 的本质区别

这是理解 Triton 控制流的关键。`tl.where()` 和 `if/else` 看似做同样的事情，但在底层实现上有本质区别。

```python
@triton.jit
def where_vs_if_else(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # ============================================================
    # 方法 1：使用 tl.where（推荐）
    # ============================================================
    # 硬件行为：
    # 1. 计算 condition = x > 0（所有线程同时执行）
    # 2. 计算 true_branch = x * 2（所有线程同时执行）
    # 3. 计算 false_branch = -x（所有线程同时执行）
    # 4. 使用谓词寄存器选择结果（所有线程同时执行）
    #
    # 关键：所有线程始终执行相同的指令，无 divergence
    y_where = tl.where(x > 0, x * 2, -x)

    # ============================================================
    # 方法 2：使用 if/else
    # ============================================================
    # 硬件行为（当 condition 是张量时）：
    # 1. 计算 condition = x > 0
    # 2. 设置活动掩码：仅 condition=True 的线程活跃
    # 3. 执行 true_branch（False 线程被禁用，但仍在硬件层面"等待"）
    # 4. 切换活动掩码：仅 condition=False 的线程活跃
    # 5. 执行 false_branch（True 线程被禁用）
    # 6. 恢复所有线程的活动状态
    #
    # 关键：两条路径串行执行，产生 warp divergence
    if x > 0:
        y_if = x * 2
    else:
        y_if = -x

    # 两种方法的结果完全相同
    # 但 tl.where 通常更快（无 divergence）
    tl.store(Y_ptr + offsets, y_where + y_if, mask=mask)
```

```
┌─────────────────────────────────────────────────────────────────┐
│              tl.where() vs if/else 执行对比                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   tl.where(x > 0, a, b):                                       │
│   ┌─────────────────────────────────────────┐                   │
│   │ Cycle 1: pred = x > 0        (all)      │                   │
│   │ Cycle 2: result = select(pred, a, b)     │                   │
│   │ Total: 2 cycles                         │                   │
│   └─────────────────────────────────────────┘                   │
│                                                                 │
│   if x > 0:                                                     │
│       result = a                                                │
│   else:                                                         │
│       result = b                                                │
│                                                                 │
│   ┌─────────────────────────────────────────┐                   │
│   │ Cycle 1: pred = x > 0        (all)      │                   │
│   │ Cycle 2: result_a = a        (pred=T)    │ ← pred=F 的线程   │
│   │ Cycle 3: result_b = b        (pred=F)    │ ← pred=T 的线程   │
│   │ Cycle 4: result = merge                 │   空闲             │
│   │ Total: 4 cycles (worst case)            │                   │
│   └─────────────────────────────────────────┘                   │
│                                                                 │
│   注意：编译器可能会优化简单的 if/else 为 tl.where                 │
│   但复杂的 if/else 块无法自动优化                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.5.4 tl.where() 在掩码计算中的典型用法

`tl.where()` 在实际 kernel 中最常见的用途是处理边界情况和实现条件计算。

```python
@triton.jit
def where_masking_patterns(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # 模式 1：边界安全的除法
    # 避免除以零的错误
    denominator = tl.load(X_ptr + offsets + 1, mask=offsets + 1 < N, other=1.0)
    # 将零值替换为 1.0（或其他安全值）
    safe_denominator = tl.where(denominator == 0.0, 1.0, denominator)
    result = x / safe_denominator

    # 模式 2：Clamp 操作（值裁剪）
    # 将值限制在 [min_val, max_val] 范围内
    min_val = -1.0
    max_val = 1.0
    clamped = tl.where(x < min_val, min_val,
                       tl.where(x > max_val, max_val, x))

    # 模式 3：条件累加
    # 只对正数进行累加
    positive_sum = tl.sum(tl.where(x > 0, x, 0.0), axis=0)

    # 模式 4：带条件的归约
    # 计算正数的平均值
    positive_count = tl.sum(tl.where(x > 0, 1.0, 0.0), axis=0)
    positive_avg = tl.where(
        positive_count > 0,
        positive_sum / positive_count,
        0.0
    )

    tl.store(Y_ptr + offsets, result + clamped, mask=mask)
```

### 6.5.5 嵌套 tl.where()

对于多条件选择，可以嵌套使用 `tl.where()`。当条件较多时，这比 if/else 链更高效。

```python
@triton.jit
def nested_where(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # 实现分段函数：
    #   x < -2:     y = -2
    #   -2 ≤ x < 0: y = x
    #   0 ≤ x < 2:  y = x * x
    #   x ≥ 2:      y = 4

    # 使用嵌套 tl.where
    y = tl.where(
        x < -2.0,
        -2.0,                           # x < -2
        tl.where(
            x < 0.0,
            x,                           # -2 ≤ x < 0
            tl.where(
                x < 2.0,
                x * x,                   # 0 ≤ x < 2
                4.0                      # x ≥ 2
            )
        )
    )

    # 等价的 if/else 写法（会有 divergence）：
    # if x < -2.0:
    #     y = -2.0
    # elif x < 0.0:
    #     y = x
    # elif x < 2.0:
    #     y = x * x
    # else:
    #     y = 4.0

    tl.store(Y_ptr + offsets, y, mask=mask)
```

<div data-component="WhereVsIfElse"></div>

[组件：WhereVsIfElse - 交互式对比 tl.where 与 if/else 的执行行为与性能]

---

## 6.6 控制流对性能的影响

理解控制流如何影响性能是编写高效 Triton kernel 的核心技能。本节深入分析 warp divergence 的性能代价及其优化策略。

### 6.6.1 Warp Divergence 的性能代价

```
┌─────────────────────────────────────────────────────────────────┐
│                  Warp Divergence 性能分析                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   理论模型:                                                      │
│   - 无 divergence: T_base = N 条指令 × 1 cycle/指令              │
│   - 有 divergence (k 条路径): T_div = k × T_base                │
│   - 性能损失: kx (最坏情况)                                      │
│                                                                 │
│   实际影响因素:                                                   │
│   ┌──────────────────────────────────────────────────────┐      │
│   │ 1. 路径数量                                          │      │
│   │    - 2 条路径 (if/else): 最多 2x 损失                 │      │
│   │    - 3 条路径 (if/elif/else): 最多 3x 损失            │      │
│   │    - N 条路径: 最多 Nx 损失                           │      │
│   │                                                      │      │
│   │ 2. 路径长度差异                                       │      │
│   │    - 短路径 vs 长路径: 短路径必须"等待"长路径完成       │      │
│   │    - 理想情况: 所有路径长度相近                        │      │
│   │                                                      │      │
│   │ 3. 线程分布                                           │      │
│   │    - 均匀分布: 所有路径都被等比例执行                   │      │
│   │    - 极端分布: 99% 走路径 A，1% 走路径 B               │      │
│   │      → 路径 B 的开销被分摊，实际损失较小               │      │
│   │                                                      │      │
│   │ 4. 编译器优化                                         │      │
│   │    - 简单的 if/else 可能被自动转换为 tl.where          │      │
│   │    - 编译器可能重排指令以减少 divergence 影响          │      │
│   └──────────────────────────────────────────────────────┘      │
│                                                                 │
│   量化分析示例:                                                   │
│   假设一个 warp 处理 32 个元素:                                   │
│   - 无 divergence: 16 条指令 × 1 cycle = 16 cycles              │
│   - if/else (50/50 分布):                                       │
│     Path A: 8 条指令 × 1 cycle = 8 cycles                       │
│     Path B: 8 条指令 × 1 cycle = 8 cycles                       │
│     合并: 8 + 8 = 16 cycles (2x 损失)                           │
│   - tl.where: 16 条指令 × 1 cycle = 16 cycles (无损失)          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.6.2 如何通过掩码避免 Divergence

让我们看一个实际的性能优化案例：Softmax 实现。

```python
# ============================================================
# 实现 1：朴素的 Softmax（有潜在 divergence）
# ============================================================
@triton.jit
def softmax_naive(
    X_ptr, Y_ptr,
    stride_xm, stride_ym,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # 加载一行数据
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :]
    mask = (offs_m[:, None] < BLOCK_M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))

    # 计算最大值（数值稳定性）
    max_val = tl.max(x, axis=1)[:, None]

    # ⚠️ 这里使用 if/else 会导致问题
    # 因为 max_val 对不同行可能不同
    # 但这是行级别的操作，不是元素级别的
    # 实际上这里不会产生 warp divergence（因为是行级操作）

    # 减去最大值
    x = x - max_val

    # 计算指数
    numerator = tl.exp(x)

    # 求和
    denominator = tl.sum(numerator, axis=1)[:, None]

    # ⚠️ 潜在的除零问题
    # 如果 denominator 为 0，会产生 inf 或 nan
    # 使用 if/else 处理会导致 divergence
    if denominator == 0:  # 这是标量条件，不会 divergence
        result = tl.zeros_like(x)
    else:
        result = numerator / denominator

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :]
    tl.store(y_ptrs, result, mask=mask)


# ============================================================
# 实现 2：使用掩码的 Softmax（无 divergence）
# ============================================================
@triton.jit
def softmax_masked(
    X_ptr, Y_ptr,
    stride_xm, stride_ym,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :]
    mask = (offs_m[:, None] < BLOCK_M) & (offs_n[None, :] < N)

    # 使用 -inf 作为填充值，确保不影响 max 计算
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))

    # 数值稳定的 softmax
    max_val = tl.max(x, axis=1)[:, None]
    x = x - max_val
    numerator = tl.exp(x)
    denominator = tl.sum(numerator, axis=1)[:, None]

    # 使用 tl.where 处理除零，避免 divergence
    # 当 denominator = 0 时，numerator 也必然为 0（因为 e^(-inf) = 0）
    # 所以结果应该是 0
    result = tl.where(denominator > 0, numerator / denominator, 0.0)

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :]
    tl.store(y_ptrs, result, mask=mask)
```

### 6.6.3 Triton 编译器的 Divergence 优化

Triton 编译器会尝试自动优化某些模式的 divergence，但不能依赖它。

```python
@triton.jit
def compiler_optimizations(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # ============================================================
    # 编译器可能优化的情况
    # ============================================================

    # 情况 1：简单的 if/else 可能被转换为 tl.where
    # 编译器检测到两条路径都很短且无副作用
    if x > 0:
        y1 = x * 2      # 简单表达式
    else:
        y1 = -x          # 简单表达式
    # 编译器可能将其优化为: y1 = tl.where(x > 0, x * 2, -x)

    # 情况 2：单分支 if（无 else）
    y2 = x
    if x > 0:
        y2 = x * 2       # 编译器可能使用谓词存储
    # 等价于: y2 = tl.where(x > 0, x * 2, x)

    # ============================================================
    # 编译器无法优化的情况
    # ============================================================

    # 情况 3：复杂的分支体（包含函数调用、内存操作）
    # 编译器必须保留真正的分支
    y3 = tl.zeros_like(x)
    if x > 0:
        # 包含内存加载的复杂分支
        y3 = tl.load(X_ptr + offsets + 1, mask=offsets + 1 < N, other=0.0)
    else:
        y3 = x * 0.5

    # 情况 4：嵌套条件
    y4 = tl.zeros_like(x)
    if x > 0:
        if x > 10:
            y4 = x ** 2
        else:
            y4 = x
    else:
        y4 = 0.0
    # 编译器通常无法优化嵌套的张量级条件

    tl.store(Y_ptr + offsets, y1 + y2 + y3 + y4, mask=mask)
```

### 6.6.4 性能测量与分析

```python
import triton
import triton.language as tl
import torch
import time

# 性能对比测试
def benchmark_kernels():
    N = 1024 * 1024
    BLOCK = 1024
    x = torch.randn(N, device='cuda')
    y = torch.empty_like(x)

    # 预热
    for _ in range(10):
        relu_if_else[(N // BLOCK,)](x, y, N, BLOCK)
        relu_where[(N // BLOCK,)](x, y, N, BLOCK)

    # 测量 if/else 版本
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        relu_if_else[(N // BLOCK,)](x, y, N, BLOCK)
    torch.cuda.synchronize()
    time_if_else = time.time() - start

    # 测量 tl.where 版本
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        relu_where[(N // BLOCK,)](x, y, N, BLOCK)
    torch.cuda.synchronize()
    time_where = time.time() - start

    print(f"if/else: {time_if_else*1000:.2f} ms")
    print(f"tl.where: {time_where*1000:.2f} ms")
    print(f"Speedup: {time_if_else/time_where:.2f}x")
```

<div data-component="DivergenceAnalyzer"></div>

[组件：DivergenceAnalyzer - 可视化 warp divergence 的发生过程与性能影响]

---

## 6.7 高级控制流

本节讨论更复杂的控制流模式，包括嵌套条件、early return、break/continue 等。

### 6.7.1 嵌套条件

嵌套条件在实际 kernel 中很常见，但需要特别注意 divergence 的累积效应。

```python
@triton.jit
def nested_conditions(
    X_ptr, Y_ptr, Z_ptr, W_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(Y_ptr + offsets, mask=mask, other=0.0)

    # 嵌套条件（使用 tl.where 避免 divergence）
    # 第一层：x 的符号
    # 第二层：y 的大小
    # 第三层：组合条件

    # 等价于：
    # if x > 0:
    #     if y > 0:
    #         z = x + y
    #     else:
    #         z = x - y
    # else:
    #     if y > 0:
    #         z = y - x
    #     else:
    #         z = -(x + y)

    z = tl.where(
        x > 0,
        tl.where(y > 0, x + y, x - y),
        tl.where(y > 0, y - x, -(x + y))
    )

    # 更复杂的嵌套：三层条件
    w = tl.where(
        x > 10,
        tl.where(
            y > 10,
            x * y,           # x > 10, y > 10
            tl.where(
                y > 0,
                x + y,       # x > 10, 0 < y ≤ 10
                x            # x > 10, y ≤ 0
            )
        ),
        tl.where(
            x > 0,
            tl.where(
                y > 0,
                x * 0.5,     # 0 < x ≤ 10, y > 0
                y * 0.5      # 0 < x ≤ 10, y ≤ 0
            ),
            tl.where(
                y > 0,
                y,           # x ≤ 0, y > 0
                0.0          # x ≤ 0, y ≤ 0
            )
        )
    )

    tl.store(Z_ptr + offsets, z, mask=mask)
    tl.store(W_ptr + offsets, w, mask=mask)
```

### 6.7.2 Early Return

Early return 可以避免不必要的计算，但在 Triton 中需要小心使用。

```python
@triton.jit
def early_return_scalar(
    X_ptr, Y_ptr, N, threshold,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # ✅ 标量条件的 early return：安全且高效
    # 如果整个块的最大值都小于阈值，跳过所有计算
    max_val = tl.max(x, axis=0)
    if max_val < threshold:
        # 直接返回，不执行后续计算
        tl.store(Y_ptr + offsets, tl.zeros_like(x), mask=mask)
        return

    # 只有当 max_val >= threshold 时才执行到这里
    y = tl.exp(x - max_val)  # 数值稳定的 exp
    y = y / tl.sum(y, axis=0)

    tl.store(Y_ptr + offsets, y, mask=mask)


@triton.jit
def early_return_with_mask(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # ⚠️ 张量条件不能直接用于 early return
    # 因为不同元素可能满足不同条件
    # 错误示例（编译错误）：
    # if x > 0:
    #     return  # 不允许！张量条件不能控制 return

    # ✅ 正确做法：使用掩码处理不同条件
    # 本质上，"early return" 对于张量条件就是掩码操作
    positive_mask = x > 0
    large_mask = x > 100

    # 对不同条件的元素进行不同处理
    result = tl.where(
        large_mask,
        tl.exp(x),           # x > 100: 指数
        tl.where(
            positive_mask,
            x * 2,            # 0 < x ≤ 100: 线性
            0.0               # x ≤ 0: 零
        )
    )

    tl.store(Y_ptr + offsets, result, mask=mask)
```

### 6.7.3 Break 与 Continue

Triton 对 `break` 和 `continue` 的支持有限，需要特别注意使用条件。

```python
@triton.jit
def break_example(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
    MAX_ITER: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    result = tl.zeros_like(x)

    # ✅ 在标量条件下使用 break
    # 这里 converged 是标量，所有线程同时 break
    for i in tl.static_range(MAX_ITER):
        # 计算新的值
        new_result = result + x / (i + 1)

        # 检查收敛（使用标量归约）
        max_change = tl.max(tl.abs(new_result - result), axis=0)
        result = new_result

        # 标量条件 break：所有线程同时退出
        if max_change < 1e-6:
            break

    tl.store(Y_ptr + offsets, result, mask=mask)


@triton.jit
def continue_example(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # ✅ continue 在标量条件下使用
    # 计算前 N 个非零元素的累加和
    accumulator = tl.zeros((BLOCK,), dtype=tl.float32)
    count = 0

    for i in range(N):
        x = tl.load(X_ptr + i, mask=mask, other=0.0)

        # 标量条件 continue：跳过当前迭代
        # 注意：这里 x 是张量，但 tl.load 单个元素得到标量
        # 实际上这个例子不太正确，因为我们加载的是块
        # 更好的例子见下面

    # 更实用的 continue 示例：处理稀疏数据
    # 假设我们有一个索引数组，只处理非零索引
    indices = tl.load(X_ptr + offsets, mask=mask, other=0)
    values = tl.load(Y_ptr + offsets, mask=mask, other=0.0)

    # 使用掩码实现 "continue" 的效果
    valid_mask = (indices >= 0) & mask
    processed = tl.where(valid_mask, values * 2.0, 0.0)

    tl.store(Y_ptr + offsets, processed, mask=mask)
```

### 6.7.4 模拟复杂的控制流模式

某些复杂的控制流模式需要通过组合基本操作来实现。

```python
@triton.jit
def state_machine_pattern(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
    NUM_STATES: tl.constexpr,
):
    """模拟一个简单的状态机，每个元素可能处于不同状态"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # 状态编码：0=初始, 1=处理中, 2=完成, 3=错误
    state = tl.zeros((BLOCK,), dtype=tl.int32)  # 初始状态 0
    result = tl.zeros_like(x)

    # 模拟状态转换
    for step in tl.static_range(NUM_STATES):
        # 状态 0 → 状态 1：检查输入有效性
        to_state1 = (state == 0) & (x > -100) & (x < 100)
        state = tl.where(to_state1, 1, state)
        result = tl.where(to_state1, x, result)

        # 状态 1 → 状态 2 或 3：处理数据
        processing = state == 1
        success = processing & (tl.abs(result) < 50)
        error = processing & (tl.abs(result) >= 50)
        state = tl.where(success, 2, tl.where(error, 3, state))
        result = tl.where(success, result * 2, tl.where(error, -999.0, result))

        # 状态 2：完成，可以进行下一步处理
        done = state == 2
        result = tl.where(done, result + 1, result)

    # 状态 3 的元素被标记为错误值 -999
    tl.store(Y_ptr + offsets, result, mask=mask)


@triton.jit
def binary_search_pattern(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
    TABLE_SIZE: tl.constexpr,
):
    """在有序表中二分查找"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # 要查找的值
    query = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # 二分查找的上下界
    low = tl.zeros((BLOCK,), dtype=tl.int32)
    high = tl.full((BLOCK,), TABLE_SIZE - 1, dtype=tl.int32)

    # 二分查找循环（迭代次数是 constexpr，可以展开）
    for _ in tl.static_range(32):  # 最多 32 次迭代（支持 2^32 的表）
        # 计算中间点
        mid = (low + high) // 2

        # 加载中间值（需要特殊处理，因为 mid 是张量）
        # 这里简化处理，实际需要更复杂的内存访问模式
        mid_val = tl.load(Y_ptr + mid, mask=mask, other=0.0)

        # 更新上下界
        go_right = query > mid_val
        low = tl.where(go_right, mid + 1, low)
        high = tl.where(go_right, high, mid - 1)

        # 检查是否找到或搜索完毕
        found = (mid_val == query) | (low > high)

    # 存储结果（找到的索引或 -1）
    result = tl.where(found, mid, -1)
    tl.store(Y_ptr + offsets, result, mask=mask)
```

<div data-component="AdvancedControlFlow"></div>

[组件：AdvancedControlFlow - 可视化状态机与二分查找的执行过程]

---

## 6.8 最佳实践

本节总结控制流编程的最佳实践，帮助你编写既正确又高效的 Triton kernel。

### 6.8.1 优先使用掩码而非 if/else

```python
# ❌ 不推荐：使用 if/else 处理元素级条件
@triton.jit
def bad_relu(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    if x > 0:      # 张量条件 → warp divergence
        y = x
    else:
        y = 0.0

    tl.store(Y_ptr + offsets, y, mask=mask)


# ✅ 推荐：使用 tl.where
@triton.jit
def good_relu(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    y = tl.where(x > 0, x, 0.0)  # 无 divergence

    tl.store(Y_ptr + offsets, y, mask=mask)


# ✅ 最简洁：利用布尔值的数值属性
@triton.jit
def concise_relu(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    y = x * (x > 0)  # True=1, False=0

    tl.store(Y_ptr + offsets, y, mask=mask)
```

### 6.8.2 constexpr 循环优于动态循环

```python
# ❌ 不推荐：动态循环次数
@triton.jit
def bad_loop(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    num_blocks = tl.cdiv(N, BLOCK)  # 运行时值
    for i in range(num_blocks):      # 编译器无法展开
        acc += tl.load(X_ptr + i * BLOCK + offsets)

    tl.store(Y_ptr + offsets, acc)


# ✅ 推荐：使用 constexpr 循环次数
@triton.jit
def good_loop(X_ptr, Y_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    # 当 N 和 BLOCK 都是 constexpr 时，tl.cdiv(N, BLOCK) 也是 constexpr
    for i in tl.static_range(tl.cdiv(N, BLOCK)):
        acc += tl.load(X_ptr + i * BLOCK + offsets)

    tl.store(Y_ptr + offsets, acc)


# ✅ 最佳：手动展开小循环
@triton.jit
def unrolled_loop(X_ptr, Y_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    # 已知只有 4 次迭代，手动展开
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    acc += tl.load(X_ptr + 0 * BLOCK + offsets)
    acc += tl.load(X_ptr + 1 * BLOCK + offsets)
    acc += tl.load(X_ptr + 2 * BLOCK + offsets)
    acc += tl.load(X_ptr + 3 * BLOCK + offsets)

    tl.store(Y_ptr + offsets, acc)
```

### 6.8.3 避免 Warp Divergence 的编程模式

```python
@triton.jit
def divergence_free_patterns(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # ============================================================
    # 模式 1：用掩码替代条件分支
    # ============================================================
    # ❌ 分支版本
    # if x > 0:
    #     y = x * 2
    # else:
    #     y = -x

    # ✅ 掩码版本
    y = tl.where(x > 0, x * 2, -x)

    # ============================================================
    # 模式 2：用选择替代多分支
    # ============================================================
    # ❌ 多分支版本
    # if x > 10:
    #     category = 3
    # elif x > 5:
    #     category = 2
    # elif x > 0:
    #     category = 1
    # else:
    #     category = 0

    # ✅ 嵌套 tl.where
    category = tl.where(
        x > 10, 3,
        tl.where(x > 5, 2, tl.where(x > 0, 1, 0))
    )

    # ============================================================
    # 模式 3：用数学运算替代条件
    # ============================================================
    # ❌ 条件版本
    # if x > 0:
    #     sign = 1
    # else:
    #     sign = -1

    # ✅ 数学版本（无分支）
    sign = 2 * (x > 0) - 1  # True→1, False→-1

    # ============================================================
    # 模式 4：用累加替代条件赋值
    # ============================================================
    # 计算满足条件的元素数量
    # ❌ 条件版本
    # count = 0
    # for i in range(N):
    #     if x[i] > 0:
    #         count += 1

    # ✅ 掩码累加版本
    count = tl.sum(tl.where(x > 0, 1.0, 0.0), axis=0)

    tl.store(Y_ptr + offsets, y + category + sign, mask=mask)
```

### 6.8.4 实战案例：优化 Softmax 的控制流

```python
@triton.jit
def optimized_softmax(
    X_ptr, Y_ptr,
    stride_xm, stride_ym,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    优化的 Softmax 实现，展示控制流最佳实践。

    关键优化点：
    1. 使用 -inf 作为填充值，确保不影响 max 计算
    2. 使用 tl.where 处理除零，避免 if/else
    3. 使用 constexpr 循环（如果可能）
    """
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # 边界掩码
    mask = (offs_m[:, None] < BLOCK_M) & (offs_n[None, :] < N)

    # 加载数据，越界位置用 -inf 填充
    # 这样 tl.max 会自动忽略越界位置
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :]
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))

    # Step 1: 数值稳定性 - 减去最大值
    row_max = tl.max(x, axis=1)[:, None]
    x = x - row_max

    # Step 2: 计算指数
    numerator = tl.exp(x)

    # Step 3: 求和
    row_sum = tl.sum(numerator, axis=1)[:, None]

    # Step 4: 归一化（使用 tl.where 避免除零）
    # 当整行都是 -inf 时（越界），row_sum = 0
    # 此时 numerator 也全为 0，结果应该为 0
    result = tl.where(row_sum > 0, numerator / row_sum, 0.0)

    # 存储结果
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :]
    tl.store(y_ptrs, result, mask=mask)
```

### 6.8.5 控制流选择决策树

```
┌─────────────────────────────────────────────────────────────────┐
│                   控制流选择决策树                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   需要条件执行?                                                  │
│   │                                                             │
│   ├─ 条件是标量（整个 program 统一）?                             │
│   │  └─ YES → 使用 if/else（无 divergence）                      │
│   │                                                             │
│   ├─ 条件是张量（不同元素可能不同）?                               │
│   │  ├─ 只需要选择两个值之一?                                     │
│   │  │  └─ YES → 使用 tl.where()（首选）                         │
│   │  │                                                          │
│   │  ├─ 需要执行复杂计算?                                        │
│   │  │  └─ YES → 使用 if/else（编译器可能优化）                    │
│   │  │         → 或拆分为多个 tl.where                            │
│   │  │                                                          │
│   │  └─ 需要修改控制流（break/return）?                           │
│   │     └─ 使用掩码 + tl.where 模拟                              │
│   │                                                             │
│   └─ 循环次数是编译期常量?                                        │
│      ├─ YES → 使用 tl.static_range（展开循环）                    │
│      └─ NO  → 使用 range（生成循环指令）                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 本章小结

本章深入探讨了 Triton 中的控制流与条件执行机制，涵盖了从基础概念到高级优化的完整知识体系。

### 核心概念回顾

1. **SIMT 执行模型**：GPU 以 Warp（32 线程）为单位执行，同一 Warp 中的线程在硬件层面必须执行相同指令。当线程需要走不同路径时，硬件通过串行化执行来模拟并行，这就是 Warp Divergence。

2. **条件语句的两种语义**：
   - **标量条件**：结果对整个 program 统一，使用 if/else 即可，无 divergence
   - **张量条件**：不同元素可能满足不同条件，会导致 warp divergence

3. **掩码操作**：通过 `mask` 参数控制 load/store 的元素选择，在硬件层面使用谓词寄存器实现，不会导致 divergence。`other` 参数为被掩码排除的元素提供默认值。

4. **循环结构**：
   - `range`：支持运行时值，编译器生成循环指令
   - `tl.static_range`：要求 constexpr 参数，编译器展开循环
   - 循环展开可减少控制开销，但增加代码体积和寄存器压力

5. **tl.where()**：逐元素条件选择，硬件层面使用谓词实现，无 warp divergence。是替代简单 if/else 的首选方案。

6. **性能影响**：Warp Divergence 会导致 N 路径串行执行，理论性能损失可达 Nx。通过掩码编程可以完全避免这种损失。

### 关键要点

| 场景 | 推荐方案 | 原因 |
|:---|:---|:---|
| 标量条件分支 | `if/else` | 无 divergence |
| 张量条件选择 | `tl.where()` | 无 divergence |
| 复杂条件逻辑 | `if/else` + 编译器优化 | 代码清晰 |
| 边界处理 | `mask` + `other` | 安全且高效 |
| 固定次数循环 | `tl.static_range` | 编译器展开 |
| 动态次数循环 | `range` | 灵活性 |

### 性能优化原则

1. **优先使用掩码**：掩码操作是 GPU 上处理条件逻辑的原生方式，性能最优
2. **constexpr 为王**：尽可能将循环次数、条件判断变为编译期常量
3. **避免不必要的分支**：每个分支都可能引入 divergence，能用数学表达式替代就替代
4. **理解编译器行为**：简单的 if/else 可能被自动优化，但不要依赖它

---

## 思考题

### 基础题

1. **SIMT vs SIMD**：解释 SIMT 和 SIMD 的主要区别。为什么 GPU 选择 SIMT 模型而不是 SIMD？

2. **Warp Divergence 计算**：假设一个 Warp（32 线程）执行以下代码，如果 50% 的线程满足 `x > 0`，计算理论上的性能损失比例。

```python
if x > 0:
    y = expensive_function_a(x)
else:
    y = expensive_function_b(x)
```

3. **掩码加载**：解释 `tl.load(ptr, mask=m, other=0.0)` 中 `mask=False` 的元素在硬件层面发生了什么。为什么这比 if/else 更高效？

### 进阶题

4. **tl.where 优化**：将以下使用 if/else 的代码重写为使用 tl.where 的版本，并解释为什么重写后的版本性能更好。

```python
@triton.jit
def complex_branch(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    if x > 10:
        y = x ** 2
    elif x > 0:
        y = x * 2
    elif x > -10:
        y = -x
    else:
        y = 0.0

    tl.store(Y_ptr + offsets, y, mask=mask)
```

5. **循环优化**：分析以下代码中的性能问题，并提出至少两种优化方案。

```python
@triton.jit
def slow_loop(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    result = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(N):
        x = tl.load(X_ptr + i, mask=offsets < N, other=0.0)
        if x > 0:
            result += tl.exp(x)
        else:
            result += tl.log(-x + 1e-7)

    tl.store(Y_ptr + offsets, result)
```

### 挑战题

6. **设计题**：设计一个 Triton kernel 实现分段线性函数（piecewise linear function），函数由 N 个断点和 N+1 个斜率定义。要求：
   - 使用 tl.where 而非 if/else
   - 处理边界情况
   - 分析时间复杂度和可能的性能瓶颈

7. **优化题**：实现一个数值稳定的 LogSoftmax kernel，要求：
   - 使用掩码处理边界
   - 使用 tl.where 处理数值稳定性（避免 log(0)）
   - 对比使用 if/else 的实现，分析性能差异

8. **分析题**：阅读以下 Triton 编译器生成的 PTX 代码片段，解释每条指令的作用，并指出哪些部分与 warp divergence 相关。

```ptx
@p  ld.global.f32  %f1, [%rd1];
@!p ld.global.f32  %f2, [%rd2];
setp.gt.f32  %p1, %f1, 0f00000000;
@%p1  fma.rn.f32  %f3, %f1, %f1, %f2;
@!%p1 fma.rn.f32  %f3, %f2, %f2, %f1;
```

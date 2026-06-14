---
title: "Chapter 5: 程序结构与 Grid/Block 映射"
description: "深入理解 Triton kernel 的基本结构、program_id 与 num_programs 机制、Grid 的定义与启动、Triton Grid 与 CUDA Grid/Block 的映射关系、多维索引计算、分块策略、Grid stride loop 以及 Launch 参数详解"
date: "2026-06-11"
---

# Chapter 5: 程序结构与 Grid/Block 映射

> **学习目标**：
> - 掌握 Triton kernel 的基本结构：`@triton.jit` 装饰器、函数签名、指针参数、标量参数与 `tl.constexpr` 参数的分类与语义
> - 理解 `tl.program_id(axis)` 与 `tl.num_programs(axis)` 的完整语义，以及 Grid 维度（x/y/z）的含义
> - 掌握 Grid 的定义方式：tuple 形式与 callable lambda 形式，以及 1D/2D/3D Grid 的选择策略
> - 深入理解 Triton Grid 与 CUDA Grid/Block 的映射关系，理解 Triton 如何隐藏硬件复杂性
> - 熟练使用 `//` 和 `%` 运算从 1D program_id 推导多维索引，掌握常见的多维索引映射模式
> - 掌握数据分块（tiling）策略、BLOCK_SIZE 选择、非对齐维度处理、Grid stride loop 以及 num_warps/num_stages 等 Launch 参数的调优方法

---

## 5.1 Triton Kernel 的基本结构

### 5.1.1 @triton.jit 装饰器

每一个 Triton kernel 都以 `@triton.jit` 装饰器开头。这个装饰器告诉 Triton 编译器：这个 Python 函数是一个 GPU kernel，需要被编译成 PTX/CUBIN 代码并在 GPU 上执行。

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(
    # 参数列表：分为指针参数、标量参数、constexpr 参数
):
    # kernel 函数体
    pass
```

**`@triton.jit` 的核心作用**：

| 作用 | 说明 |
|:---|:---|
| **标记 GPU kernel** | 告诉 Triton 编译器将函数编译为 GPU 代码 |
| **参数类型推断** | 根据调用时传入的参数自动推断类型 |
| **constexpr 特化** | 为不同的 constexpr 值生成不同的编译版本 |
| **禁止 Python 特性** | kernel 内不能使用 list、dict、print 等 Python 运行时特性 |

**`@triton.jit` 可选参数**：

```python
@triton.jit(
    # 指定调试模式：生成更易读的 IR，但性能下降
    debug=True,

    # 指定最大寄存器数（影响 occupancy）
    maxnreg=128,

    # 指定每个 program 的 warp 数（也可以在 launch 时指定）
    num_warps=4,

    # 指定 software pipeline 的 stage 数
    num_stages=3,
)
def my_kernel_with_options(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    tl.store(Y_ptr + offsets, x * 2.0, mask=mask)
```

**`@triton.jit` 与普通 Python 函数的区别**：

```python
# 普通 Python 函数：在 CPU 上执行
def cpu_function(x):
    return x * 2

# Triton kernel：被编译为 GPU 代码，在 GPU 上执行
@triton.jit
def gpu_kernel(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    tl.store(Y_ptr + offsets, x * 2.0, mask=mask)

# 调用方式也不同：
# CPU 函数: result = cpu_function(data)
# GPU kernel: gpu_kernel[grid](X_ptr, Y_ptr, N, BLOCK=128)
#                        ↑
#                   需要指定 grid（启动多少个 program）
```

### 5.1.2 Kernel 函数签名的三种参数

Triton kernel 的函数参数分为三类，每类有不同的语义和用途：

```python
@triton.jit
def kernel_signature_demo(
    # ===== 第一类：指针参数（Pointer Parameters）=====
    # 指向 GPU 全局内存的指针，通常由 PyTorch 张量的 .data_ptr() 传入
    # 在 kernel 内通过 tl.load / tl.store 进行读写
    A_ptr,          # 输入矩阵指针
    B_ptr,          # 输出矩阵指针
    bias_ptr,       # 偏置向量指针（可选）

    # ===== 第二类：标量参数（Scalar Parameters）=====
    # 整数或浮点数标量，作为运行时值传入
    # 常用于：维度大小、stride、缩放因子等
    M,              # 矩阵行数（运行时值）
    N,              # 矩阵列数（运行时值）
    stride_am,      # A 的行 stride（运行时值）
    stride_an,      # A 的列 stride（运行时值）
    scale,          # 缩放因子（运行时值）

    # ===== 第三类：constexpr 参数（Compile-time Constants）=====
    # 编译期常量，标记为 tl.constexpr
    # 编译器为每个不同的值组合生成独立的编译产物
    BLOCK_M: tl.constexpr,   # 分块的行大小
    BLOCK_N: tl.constexpr,   # 分块的列大小
    HAS_BIAS: tl.constexpr,  # 是否使用偏置（编译时裁剪分支）
):
    pass
```

**三类参数的详细对比**：

| 特性 | 指针参数 | 标量参数 | constexpr 参数 |
|:---|:---|:---|:---|
| **类型标注** | 无特殊标注 | 无特殊标注 | `: tl.constexpr` |
| **值确定时间** | 运行时 | 运行时 | 编译时 |
| **传递方式** | GPU 内存地址 | 值拷贝到 GPU 常量内存 | 编译期嵌入代码 |
| **可否用于 shape** | 否 | 否 | 是 |
| **可否用于 dtype** | 否 | 否 | 是 |
| **可否用于条件分支** | 否 | 否 | 是（编译时裁剪） |
| **编译产物** | 共享同一份 | 共享同一份 | 每个值组合一份 |
| **典型用途** | 输入/输出数据 | 维度、stride、缩放 | BLOCK_SIZE、dtype、功能开关 |

### 5.1.3 指针参数详解

指针参数是 kernel 与外部数据交互的桥梁。在 PyTorch 中，张量的 `.data_ptr()` 返回其在 GPU 全局内存中的起始地址。

```python
import torch
import triton
import triton.language as tl

# ---- Kernel 定义 ----
@triton.jit
def vector_add_kernel(
    a_ptr,       # 指针参数：输入向量 A 的基地址
    b_ptr,       # 指针参数：输入向量 B 的基地址
    c_ptr,       # 指针参数：输出向量 C 的基地址
    n,           # 标量参数：向量长度
    BLOCK: tl.constexpr,  # constexpr 参数：分块大小
):
    # 获取当前 program 的 ID
    pid = tl.program_id(0)

    # 计算当前 program 负责的元素索引
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    # 创建边界 mask
    mask = offsets < n

    # 通过指针参数访问 GPU 全局内存
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)  # 读取 A
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)  # 读取 B

    # 计算
    c = a + b

    # 通过指针参数写回结果
    tl.store(c_ptr + offsets, c, mask=mask)


# ---- Python 主机端调用 ----
def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = a.shape[0]
    c = torch.empty_like(a)

    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)  # 启动多少个 program

    vector_add_kernel[grid](
        a.data_ptr(),   # 传递 A 的 GPU 地址 → a_ptr
        b.data_ptr(),   # 传递 B 的 GPU 地址 → b_ptr
        c.data_ptr(),   # 传递 C 的 GPU 地址 → c_ptr
        n,              # 传递向量长度 → n
        BLOCK=BLOCK,    # 传递 constexpr → BLOCK
    )
    return c

# 测试
a = torch.randn(10000, device='cuda', dtype=torch.float32)
b = torch.randn(10000, device='cuda', dtype=torch.float32)
c = vector_add(a, b)
print(f"结果正确: {torch.allclose(c, a + b)}")  # True
```

**指针参数的内部机制**：

```
Python 主机端                              GPU Kernel
─────────────                              ──────────
a = torch.randn((M, N), device='cuda')
a.data_ptr() → 0x7f1234000000    ──────►  a_ptr = 0x7f1234000000
                                           │
                                           ├─ a_ptr + 0      → 读取 A[0]
                                           ├─ a_ptr + 4      → 读取 A[1]  (float32, 4字节)
                                           ├─ a_ptr + 8      → 读取 A[2]
                                           └─ a_ptr + (i*4)  → 读取 A[i]

对于 2D 矩阵:
  A[i, j] 的地址 = A_ptr + i * stride_am + j * stride_an
                   ↑          ↑               ↑
                基地址    行偏移（字节）    列偏移（字节）
```

**多个指针参数的常见模式**：

```python
@triton.jit
def matmul_ptrs(
    A_ptr, B_ptr, C_ptr,         # 数据指针
    M, N, K,                     # 标量：维度大小
    stride_am, stride_ak,        # 标量：A 的 stride
    stride_bk, stride_bn,        # 标量：B 的 stride
    stride_cm, stride_cn,        # 标量：C 的 stride
    BLOCK_M: tl.constexpr,       # constexpr：分块大小
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pass
```

### 5.1.4 标量参数详解

标量参数用于传递运行时的数值信息，如维度大小、stride、缩放因子等。它们在 kernel 中被存储在 GPU 的常量内存（Constant Memory）中，访问速度极快。

```python
@triton.jit
def scalar_params_demo(
    A_ptr, B_ptr,
    M,                 # 标量：矩阵行数（int32 或 int64）
    N,                 # 标量：矩阵列数
    alpha,             # 标量：缩放因子（float32）
    stride_am,         # 标量：A 的行 stride
    stride_an,         # 标量：A 的列 stride
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 标量参数可以直接参与算术运算
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # 用 M 做边界检查
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # 用 N 做边界检查

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)  # M, N 用于 mask

    # 用 stride 计算指针偏移
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    a = tl.load(a_ptrs, mask=mask, other=0.0)

    # 用 alpha 做缩放
    b = a * alpha

    b_ptrs = B_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    tl.store(b_ptrs, b, mask=mask)
```

**标量参数的类型要求**：

| 参数类型 | Python 类型 | Triton 类型 | 说明 |
|:---|:---|:---|:---|
| 整数 | `int` | `tl.int32` 或 `tl.int64` | 自动推断 |
| 浮点数 | `float` | `tl.float32` | 自动推断 |
| 布尔值 | `bool` | `tl.int1` | 不推荐，用 constexpr 代替 |

**常见错误**：

```python
@triton.jit
def bad_scalar_kernel(X_ptr, N, BLOCK: tl.constexpr):
    # 错误：标量参数不能用于 shape
    # tile = tl.zeros((N,), dtype=tl.float32)  # N 是运行时值 → 编译错误！

    # 正确：使用 constexpr 作为 shape
    tile = tl.zeros((BLOCK,), dtype=tl.float32)

    # 错误：标量参数不能用于 tl.arange
    # offsets = tl.arange(0, N)  # N 是运行时值 → 编译错误！

    # 正确：使用 constexpr
    offsets = tl.arange(0, BLOCK)
```

### 5.1.5 constexpr 参数详解

`tl.constexpr` 参数是 Triton 编程中最具特色的概念之一。它们在编译时确定值，编译器为每个不同的 constexpr 值组合生成**独立的编译产物（编译特化）**。

```python
@triton.jit
def constexpr_demo(
    X_ptr, Y_ptr, N,
    BLOCK_SIZE: tl.constexpr,      # 分块大小
    DTYPE: tl.constexpr,           # 数据类型
    USE_MASK: tl.constexpr,        # 功能开关
    ACTIVATION: tl.constexpr,      # 激活函数选择
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # DTYPE 是 constexpr → 可以用作 dtype 参数
    acc = tl.zeros((BLOCK_SIZE,), dtype=DTYPE)

    # USE_MASK 是 constexpr → 编译时裁剪分支
    if USE_MASK:
        mask = offsets < N
        x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    else:
        x = tl.load(X_ptr + offsets)

    # ACTIVATION 是 constexpr → 编译时选择激活函数
    if ACTIVATION == "relu":
        y = tl.where(x > 0, x, 0.0)
    elif ACTIVATION == "sigmoid":
        y = tl.sigmoid(x)
    else:
        y = x  # 无激活

    tl.store(Y_ptr + offsets, y)
```

**constexpr 的编译特化行为**：

```python
# 调用 1: BLOCK_SIZE=128, DTYPE=float32, USE_MASK=True, ACTIVATION="relu"
#   → 编译版本 A：128 宽度、float32、有 mask、ReLU 激活
constexpr_demo[grid](X, Y, N,
    BLOCK_SIZE=128, DTYPE=tl.float32, USE_MASK=True, ACTIVATION="relu")

# 调用 2: BLOCK_SIZE=256, DTYPE=float16, USE_MASK=False, ACTIVATION="sigmoid"
#   → 编译版本 B：256 宽度、float16、无 mask、Sigmoid 激活
#   （与 A 完全不同的 PTX 代码）
constexpr_demo[grid](X, Y, N,
    BLOCK_SIZE=256, DTYPE=tl.float16, USE_MASK=False, ACTIVATION="sigmoid")

# 调用 3: BLOCK_SIZE=128, DTYPE=float16, USE_MASK=True, ACTIVATION="relu"
#   → 编译版本 C（与 A 和 B 都不同）
constexpr_demo[grid](X, Y, N,
    BLOCK_SIZE=128, DTYPE=tl.float16, USE_MASK=True, ACTIVATION="relu")
```

**constexpr 的使用场景总结**：

| 场景 | 示例 | 原因 |
|:---|:---|:---|
| **分块大小** | `BLOCK_M: tl.constexpr` | 影响寄存器分配、循环展开 |
| **数据类型** | `DTYPE: tl.constexpr` | 影响指令选择和内存布局 |
| **功能开关** | `HAS_BIAS: tl.constexpr` | 编译时裁剪不需要的代码路径 |
| **归约轴** | `REDUCE_AXIS: tl.constexpr` | 影响归约的实现策略 |
| **循环次数** | `NUM_STEPS: tl.constexpr` | 编译器可以完全展开循环 |

### 5.1.6 完整 Kernel 结构示例

以下是一个完整的 Triton kernel，展示了所有结构元素的典型用法：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def fused_gemm_bias_relu_kernel(
    # ---- 指针参数：输入/输出数据 ----
    A_ptr,              # 输入矩阵 A: (M, K)
    B_ptr,              # 权重矩阵 B: (K, N)
    C_ptr,              # 输出矩阵 C: (M, N)
    bias_ptr,           # 偏置向量: (N,)

    # ---- 标量参数：维度和 stride ----
    M,                  # A 的行数
    N,                  # B 的列数
    K,                  # A 的列数 = B 的行数
    stride_am, stride_ak,   # A 的 stride
    stride_bk, stride_bn,   # B 的 stride
    stride_cm, stride_cn,   # C 的 stride

    # ---- constexpr 参数：编译时常量 ----
    BLOCK_M: tl.constexpr,    # M 维度的分块大小
    BLOCK_N: tl.constexpr,    # N 维度的分块大小
    BLOCK_K: tl.constexpr,    # K 维度的分块大小
    HAS_BIAS: tl.constexpr,   # 是否使用偏置
    GROUP_SIZE_M: tl.constexpr,  # L2 cache 优化的 group 大小
):
    # ---- Step 1: 计算当前 program 负责的 tile 坐标 ----
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- Step 2: 创建索引 ----
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # ---- Step 3: 初始化累加器 ----
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ---- Step 4: 沿 K 维度迭代 ----
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # 加载 A tile
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        k_mask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)

        # 加载 B tile
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=k_mask, other=0.0)

        # 矩阵乘累加
        accumulator += tl.dot(a, b)

        # 推进 K 维度
        offs_k += BLOCK_K

    # ---- Step 5: 添加偏置（constexpr 分支裁剪）----
    if HAS_BIAS:
        bias_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        bias_mask = bias_offs < N
        bias = tl.load(bias_ptr + bias_offs, mask=bias_mask, other=0.0)
        accumulator += bias[None, :]  # 广播到每一行

    # ---- Step 6: 激活函数 ----
    result = tl.where(accumulator > 0, accumulator, 0.0)  # ReLU

    # ---- Step 7: 类型转换并存储 ----
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, result.to(tl.float16), mask=c_mask)
```

<div data-component="KernelStructureDiagram"></div>

[组件：KernelStructureDiagram - 交互式展示 Triton kernel 的代码结构与执行流程]

---

## 5.2 program_id 与 num_programs

### 5.2.1 tl.program_id(axis) 详解

`tl.program_id(axis)` 是 Triton 中最核心的内置函数之一。它返回当前 program 在指定轴上的 ID，用于确定当前 program 负责处理数据的哪一部分。

```python
@triton.jit
def program_id_demo(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    # 获取当前 program 在 axis=0（x 轴）上的 ID
    # 值范围: [0, grid_x - 1]
    pid = tl.program_id(0)

    # 使用 pid 确定当前 program 负责的数据范围
    # program 0: 处理 [0, BLOCK)
    # program 1: 处理 [BLOCK, 2*BLOCK)
    # program 2: 处理 [2*BLOCK, 3*BLOCK)
    # ...
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    tl.store(Y_ptr + offsets, x * 2.0, mask=mask)
```

**program_id 的三个轴**：

```python
@triton.jit
def three_axis_program_id(X_ptr, Y_ptr):
    # axis=0 (x 轴)：第一个维度的 program ID
    pid_x = tl.program_id(0)

    # axis=1 (y 轴)：第二个维度的 program ID
    pid_y = tl.program_id(1)

    # axis=2 (z 轴)：第三个维度的 program ID
    pid_z = tl.program_id(2)

    # 三个轴可以组合使用，形成三维的 program 网格
    # pid_x ∈ [0, grid_x), pid_y ∈ [0, grid_y), pid_z ∈ [0, grid_z)
    pass
```

**program_id 与 Grid 的关系**：

```
假设 Grid 定义为: grid = (4, 3, 2)  即 grid_x=4, grid_y=3, grid_z=2

总共启动 4 × 3 × 2 = 24 个 program

program_id 的值域:
  axis=0 (x): [0, 1, 2, 3]
  axis=1 (y): [0, 1, 2]
  axis=2 (z): [0, 1, 2]

三维 Grid 可视化:

     z=0                    z=1
   y=0  y=1  y=2         y=0  y=1  y=2
x=0 [P0] [P1] [P2]      [P12][P13][P14]
x=1 [P3] [P4] [P5]      [P15][P16][P17]
x=2 [P6] [P7] [P8]      [P18][P19][P20]
x=3 [P9] [P10][P11]     [P21][P22][P23]

每个 program 的坐标:
  P0:  (pid_x=0, pid_y=0, pid_z=0)
  P4:  (pid_x=1, pid_y=1, pid_z=0)
  P17: (pid_x=1, pid_y=2, pid_z=1)
  P23: (pid_x=3, pid_y=2, pid_z=1)
```

### 5.2.2 tl.num_programs(axis) 详解

`tl.num_programs(axis)` 返回指定轴上总共有多少个 program，即 Grid 在该轴的大小。

```python
@triton.jit
def num_programs_demo(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    # 获取每个轴的 program 数量
    num_pid_x = tl.num_programs(0)  # = grid_x
    num_pid_y = tl.num_programs(1)  # = grid_y（1D Grid 时为 1）
    num_pid_z = tl.num_programs(2)  # = grid_z（1D/2D Grid 时为 1）

    pid = tl.program_id(0)

    # 常见用法：计算总 program 数
    total_programs = num_pid_x  # 对于 1D Grid

    # 用于 Grid stride loop（后面会详细讲解）
    # 每个 program 处理多个 block
    for i in range(pid, tl.cdiv(N, BLOCK), total_programs):
        offsets = i * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
        tl.store(Y_ptr + offsets, x * 2.0, mask=mask)
```

**program_id 与 num_programs 的对比**：

| 函数 | 含义 | 返回值类型 | 值范围 | 典型用途 |
|:---|:---|:---|:---|:---|
| `tl.program_id(0)` | 当前 program 在 x 轴的 ID | `int32` | `[0, grid_x)` | 确定当前 program 负责的数据块 |
| `tl.program_id(1)` | 当前 program 在 y 轴的 ID | `int32` | `[0, grid_y)` | 2D 索引的第二维 |
| `tl.program_id(2)` | 当前 program 在 z 轴的 ID | `int32` | `[0, grid_z)` | 3D 索引的第三维 |
| `tl.num_programs(0)` | x 轴的 program 总数 | `int32` | `= grid_x` | Grid stride loop |
| `tl.num_programs(1)` | y 轴的 program 总数 | `int32` | `= grid_y` | Grid stride loop |
| `tl.num_programs(2)` | z 轴的 program 总数 | `int32` | `= grid_z` | Grid stride loop |

### 5.2.3 program_id 的典型使用模式

**模式 1：1D Grid — 一维数据分块**

```python
@triton.jit
def vector_scale_kernel(
    X_ptr, Y_ptr, N, scale,
    BLOCK: tl.constexpr,
):
    # 1D Grid: 每个 program 处理一个 BLOCK 大小的数据块
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    tl.store(Y_ptr + offsets, x * scale, mask=mask)

# 启动: grid = (triton.cdiv(N, BLOCK),)
# 总 program 数 = ceil(N / BLOCK)
```

**模式 2：2D Grid — 矩阵分块**

```python
@triton.jit
def matrix_transpose_kernel(
    A_ptr, B_ptr, M, N,
    stride_am, stride_an,
    stride_bm, stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 2D Grid: pid_m 处理行方向，pid_n 处理列方向
    pid_m = tl.program_id(0)  # 行方向的 program ID
    pid_n = tl.program_id(1)  # 列方向的 program ID

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # 读取 A 的一个 tile
    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    a = tl.load(a_ptrs, mask=mask, other=0.0)

    # 转置并写入 B
    b_ptrs = B_ptr + offs_n[:, None] * stride_bm + offs_m[None, :] * stride_bn
    tl.store(b_ptrs, tl.trans(a), mask=mask.T)

# 启动: grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
# 总 program 数 = ceil(M / BLOCK_M) × ceil(N / BLOCK_N)
```

**模式 3：3D Grid — 批量矩阵运算**

```python
@triton.jit
def batch_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    batch, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 3D Grid:
    #   axis=0: batch 维度
    #   axis=1: M 维度
    #   axis=2: N 维度
    pid_b = tl.program_id(0)  # batch index
    pid_m = tl.program_id(1)  # M 方向的 tile index
    pid_n = tl.program_id(2)  # N 方向的 tile index

    # 计算当前 batch 的起始地址偏移
    batch_offset_a = pid_b * stride_ab
    batch_offset_b = pid_b * stride_bb
    batch_offset_c = pid_b * stride_cb

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a_ptrs = A_ptr + batch_offset_a + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + batch_offset_b + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        k_mask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask, other=0.0)
        accumulator += tl.dot(a, b)
        offs_k += BLOCK_K

    c_ptrs = C_ptr + batch_offset_c + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)

# 启动: grid = (batch, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
# 总 program 数 = batch × ceil(M/BLOCK_M) × ceil(N/BLOCK_N)
```

<div data-component="ProgramIdVisualization"></div>

[组件：ProgramIdVisualization - 交互式可视化 tl.program_id 在 1D/2D/3D Grid 中的值分布]

---

## 5.3 Grid 的定义与启动

### 5.3.1 Grid 的基本概念

Grid 定义了 kernel 启动时创建多少个 program，以及这些 program 如何组织在三个维度上。在 Triton 中，Grid 通过 `kernel[grid](...)` 语法指定。

```python
# Grid 是一个描述 program 数量和组织方式的元组
# kernel[grid](arg1, arg2, ...)

# 1D Grid: 一维数组
grid_1d = (N,)
# → 启动 N 个 program，每个的 pid_x ∈ [0, N)
# → pid_y = 0, pid_z = 0

# 2D Grid: 二维数组
grid_2d = (M, N)
# → 启动 M × N 个 program
# → pid_x ∈ [0, M), pid_y ∈ [0, N)
# → pid_z = 0

# 3D Grid: 三维数组
grid_3d = (B, M, N)
# → 启动 B × M × N 个 program
# → pid_x ∈ [0, B), pid_y ∈ [0, M), pid_z ∈ [0, N)
```

### 5.3.2 Grid 的两种定义方式

**方式 1：Tuple 形式（静态 Grid）**

```python
import triton
import triton.language as tl

@triton.jit
def simple_kernel(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    tl.store(Y_ptr + offsets, x * 2.0, mask=mask)

# ---- Tuple 形式：直接指定每个维度的大小 ----
N = 10000
BLOCK = 1024

# 1D Grid: 启动 ceil(10000 / 1024) = 10 个 program
grid = (triton.cdiv(N, BLOCK),)  # (10,)
simple_kernel[grid](X, Y, N, BLOCK=BLOCK)
```

**方式 2：Callable Lambda 形式（动态 Grid）**

```python
# ---- Lambda 形式：Grid 大小可以依赖于 kernel 参数 ----
# Lambda 接收 kernel 的参数（除 constexpr 外），返回 grid 元组
def grid_fn(META):
    # META 是一个包含 constexpr 参数的字典
    return (
        triton.cdiv(N, META['BLOCK']),  # x 维度
    )

# 使用 lambda 形式启动
simple_kernel[grid_fn](X, Y, N, BLOCK=BLOCK)
```

**两种方式的对比**：

| 特性 | Tuple 形式 | Lambda 形式 |
|:---|:---|:---|
| **语法** | `kernel[(M, N)](...)` | `kernel[grid_fn](...)` |
| **Grid 大小确定时间** | 调用前 | 调用时（依赖 constexpr） |
| **可否访问 constexpr** | 不可以 | 可以通过 META 字典 |
| **适用场景** | Grid 大小固定 | Grid 大小依赖 constexpr 参数 |
| **常见用法** | 简单 kernel | 带多种 BLOCK_SIZE 的 kernel |

**Lambda 形式的详细示例**：

```python
@triton.jit
def flexible_kernel(
    A_ptr, B_ptr, M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    # ... kernel 实现 ...

# Lambda 形式：Grid 大小依赖 BLOCK_M, BLOCK_N, GROUP_SIZE_M
def matmul_grid(META):
    return (
        triton.cdiv(META['M'], META['BLOCK_M']) *
        triton.cdiv(META['N'], META['BLOCK_N']),
    )

# 调用时，META 自动包含所有 constexpr 参数
flexible_kernel[matmul_grid](
    A, B, M, N, K,
    A.stride(0), A.stride(1),
    B.stride(0), B.stride(1),
    BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
    GROUP_SIZE_M=8,
)
```

### 5.3.3 Grid 维度的选择策略

选择合适的 Grid 维度是 Triton 编程中的重要决策。不同维度的 Grid 适用于不同的数据结构和计算模式。

**1D Grid — 线性化处理**

```python
# 适用场景：一维数据、线性化处理多维数据
@triton.jit
def linear_kernel(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    tl.store(Y_ptr + offsets, x * 2.0, mask=mask)

# 1D Grid
grid = (triton.cdiv(N, BLOCK),)
```

**1D Grid 用于 2D 数据（线性化）**：

```python
# 将 2D 矩阵的 (M, N) 线性化为 1D
# program_id → pid → pid_m = pid // num_pid_n, pid_n = pid % num_pid_n
@triton.jit
def matmul_1d_grid(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # 从 1D pid 推导 2D 坐标
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        k_mask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask, other=0.0)
        accumulator += tl.dot(a, b)
        offs_k += BLOCK_K

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)

# 1D Grid: 所有 tile 线性排列
grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
```

**2D Grid — 自然的矩阵索引**：

```python
# 适用场景：矩阵运算，每个 program 天然对应一个 (row, col) 的 tile
@triton.jit
def matmul_2d_grid(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 直接使用 2D program_id，无需手动推导
    pid_m = tl.program_id(0)  # 行方向
    pid_n = tl.program_id(1)  # 列方向

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        k_mask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask, other=0.0)
        accumulator += tl.dot(a, b)
        offs_k += BLOCK_K

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)

# 2D Grid: 每个维度对应矩阵的一个维度
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
```

**Grid 维度选择指南**：

| Grid 维度 | 适用场景 | 优势 | 劣势 |
|:---|:---|:---|:---|
| **1D** | 向量操作、线性化矩阵 | 简单、支持 L2 cache 优化（swizzle） | 需要手动推导多维索引 |
| **2D** | 矩阵运算 | 索引自然、代码清晰 | 无法直接使用 swizzle 优化 |
| **3D** | 批量矩阵、3D 张量 | 直接对应数据维度 | Grid 维度数有限（最多 3） |

### 5.3.4 Grid 与 program 数量的关系

```python
# Grid 决定了启动的 program 总数
grid = (Gx, Gy, Gz)
total_programs = Gx * Gy * Gz

# 每个 program 都会被 GPU 的 SM（Streaming Multiprocessor）执行
# GPU 会自动将 program 调度到可用的 SM 上

# 示例：
# GPU 有 80 个 SM，每个 SM 可以同时运行多个 program
# 如果 total_programs = 1000，GPU 会将这 1000 个 program 分配到 80 个 SM 上
# 每个 SM 大约处理 1000/80 = 12.5 个 program
```

**Grid 大小对性能的影响**：

```
Grid 太小（program 数 < SM 数）:
  - GPU 利用率低：部分 SM 空闲
  - 例如: GPU 有 80 个 SM，但只有 10 个 program
  - 只有 10 个 SM 在工作，70 个 SM 空闲

Grid 太大（program 数 >> SM 数 × 每 SM 最大 program 数）:
  - 没有害处，只是调度开销微增
  - GPU 会自动排队执行
  - 但每个 program 的寄存器/共享内存使用会影响 occupancy

Grid 适中（program 数 ≈ SM 数 × 每 SM program 数）:
  - GPU 利用率最高
  - 通常 program 数 = SM 数 × (1 ~ 4) 是一个好的起点
```

<div data-component="GridDimensionSelector"></div>

[组件：GridDimensionSelector - 交互式选择 Grid 维度并查看 program 分配]

---

## 5.4 Triton Grid 与 CUDA Grid/Block 的映射

### 5.4.1 CUDA 的 Grid/Block 模型

在理解 Triton 的 Grid 之前，先回顾 CUDA 的 Grid/Block 模型：

```
CUDA 的层次结构:

Grid (gridDim.x × gridDim.y × gridDim.z)
  │
  ├── Block (0, 0, 0)              ← blockIdx = (0, 0, 0)
  │     ├── Thread (0, 0, 0)       ← threadIdx = (0, 0, 0)
  │     ├── Thread (1, 0, 0)       ← threadIdx = (1, 0, 0)
  │     ├── ...
  │     └── Thread (tx-1, ty-1, tz-1)
  │
  ├── Block (1, 0, 0)              ← blockIdx = (1, 0, 0)
  │     ├── Thread (0, 0, 0)
  │     ├── ...
  │     └── Thread (tx-1, ty-1, tz-1)
  │
  └── Block (gx-1, gy-1, gz-1)    ← blockIdx = (gx-1, gy-1, gz-1)
        └── ...

CUDA 启动: kernel<<<gridDim, blockDim>>>(...)
  gridDim:  Grid 的维度 (dim3)
  blockDim: 每个 Block 的线程数 (dim3)

总线程数 = gridDim.x × gridDim.y × gridDim.z × blockDim.x × blockDim.y × blockDim.z
```

**CUDA 的两级映射**：

```cuda
// CUDA kernel
__global__ void cuda_kernel(float* X, float* Y, int N) {
    // 第一级映射: Block → 数据块
    int block_start = blockIdx.x * blockDim.x;

    // 第二级映射: Thread → 数据元素
    int idx = block_start + threadIdx.x;

    if (idx < N) {
        Y[idx] = X[idx] * 2.0f;
    }
}

// 启动
int BLOCK_SIZE = 256;
int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
cuda_kernel<<<grid_size, BLOCK_SIZE>>>(X, Y, N);
```

### 5.4.2 Triton 的 Grid 模型

Triton 简化了 CUDA 的两级层次结构。在 Triton 中，没有 Block 和 Thread 的概念——只有 **Program**（也称为 program instance）。

```
Triton 的层次结构:

Grid (grid_x × grid_y × grid_z)
  │
  ├── Program (0, 0, 0)            ← program_id = (0, 0, 0)
  │     └── 一个 program 处理一块数据（BLOCK_SIZE 个元素）
  │           内部自动映射到多个 warp
  │
  ├── Program (1, 0, 0)            ← program_id = (1, 0, 0)
  │     └── ...
  │
  └── Program (gx-1, gy-1, gz-1)  ← program_id = (gx-1, gy-1, gz-1)
        └── ...

Triton 启动: kernel[grid](...)
  grid: tuple 或 lambda
  num_warps: 每个 program 的 warp 数（隐含了线程数）
```

### 5.4.3 详细映射关系

**Triton Program 与 CUDA Block/Thread 的对应**：

```
Triton Program                          CUDA 等价物
──────────────                          ───────────
program_id(0) = i                       blockIdx.x = i
一个 program 处理 BLOCK_SIZE 个元素     一个 Block 有 blockDim.x 个线程
num_warps = 4                           blockDim.x = 128 (= 4 × 32)
program 内的张量操作                     每个线程处理一个或多个元素

Triton 的 program → CUDA 的映射:
  program_id = (i, j, k)
  等价于:
  blockIdx = (i, j, k)
  blockDim = (num_warps * 32, 1, 1)  ← num_warps 决定 Block 大小

Triton 的 grid → CUDA 的映射:
  grid = (Gx, Gy, Gz)
  等价于:
  gridDim = (Gx, Gy, Gz)
```

**详细的对应表**：

| CUDA 概念 | Triton 对应 | 说明 |
|:---|:---|:---|
| `gridDim.x` | `grid[0]` = `tl.num_programs(0)` | x 轴的 Grid 大小 |
| `gridDim.y` | `grid[1]` = `tl.num_programs(1)` | y 轴的 Grid 大小 |
| `gridDim.z` | `grid[2]` = `tl.num_programs(2)` | z 轴的 Grid 大小 |
| `blockIdx.x` | `tl.program_id(0)` | 当前 Block 在 x 轴的 ID |
| `blockIdx.y` | `tl.program_id(1)` | 当前 Block 在 y 轴的 ID |
| `blockIdx.z` | `tl.program_id(2)` | 当前 Block 在 z 轴的 ID |
| `blockDim.x` | `num_warps * 32` | 每个 Block 的线程数（隐含） |
| `threadIdx.x` | （无直接对应） | Triton 自动管理线程映射 |
| `__syncthreads()` | （自动管理） | Triton 编译器自动插入同步 |

### 5.4.4 Triton 如何隐藏硬件复杂性

**CUDA 方式：用户必须手动管理两层映射**

```cuda
// CUDA: 用户需要同时处理 Block 级别和 Thread 级别
__global__ void matmul_cuda(float* A, float* B, float* C,
                            int M, int N, int K) {
    // 用户需要手动计算每个线程负责的行和列
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 用户需要选择 blockDim 和 gridDim
dim3 blockDim(16, 16);  // 每个 Block 256 个线程
dim3 gridDim((N + 15) / 16, (M + 15) / 16);
matmul_cuda<<<gridDim, blockDim>>>(A, B, C, M, N, K);
```

**Triton 方式：用户只需关心 Program 级别**

```python
# Triton: 用户只需要关心每个 program 做什么
@triton.jit
def matmul_triton(A_ptr, B_ptr, C_ptr,
                  M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    # 用户只需要关心 program 级别的索引
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        k_mask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask, other=0.0)
        accumulator += tl.dot(a, b)
        offs_k += BLOCK_K

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)

# 用户只需要指定 grid，不需要关心 blockDim
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
matmul_triton[grid](A, B, C, M, N, K, ...)
```

**Triton 隐藏的复杂性**：

```
Triton 编译器自动完成的工作:

1. 线程映射:
   CUDA: 用户手动 threadIdx.x → 元素索引
   Triton: tl.arange(0, BLOCK) 自动创建索引，编译器自动分配给线程

2. Warp 分配:
   CUDA: 用户需要理解 Warp (32 线程) 的概念
   Triton: num_warps 参数控制，编译器自动处理

3. 同步:
   CUDA: 用户手动 __syncthreads()
   Triton: 编译器在 tl.dot、tl.sum 等操作中自动插入

4. 寄存器分配:
   CUDA: 用户需要关心 register pressure
   Triton: 编译器根据 BLOCK_SIZE 和 num_warps 自动优化

5. 共享内存:
   CUDA: 用户手动分配和使用 __shared__ memory
   Triton: 编译器在需要时自动使用（如 tl.dot 的中间结果）
```

### 5.4.5 从 CUDA 到 Triton 的映射示例

**CUDA 实现**：

```cuda
// CUDA: 向量加法
__global__ void vec_add_cuda(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 启动
int threads = 256;
int blocks = (n + threads - 1) / threads;
vec_add_cuda<<<blocks, threads>>>(a, b, c, n);
```

**Triton 等价实现**：

```python
# Triton: 向量加法
@triton.jit
def vec_add_triton(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)  # ← 等价于 blockIdx.x
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    # tl.arange(0, BLOCK) 等价于 threadIdx.x (但更灵活)
    mask = offsets < n
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(c_ptr + offsets, a + b, mask=mask)

# 启动
BLOCK = 256
grid = (triton.cdiv(n, BLOCK),)  # ← 等价于 blocks
vec_add_triton[grid](a, b, c, n, BLOCK=BLOCK)
# 等价于: vec_add_cuda<<<blocks, 256>>>(a, b, c, n)
# 但用户不需要指定 blockDim（由 num_warps 决定）
```

<div data-component="CudaTritonMappingDiagram"></div>

[组件：CudaTritonMappingDiagram - 并排对比 CUDA 和 Triton 的 Grid/Block/Thread 映射]

---

## 5.5 多维索引计算

### 5.5.1 从 1D program_id 到多维索引

在 Triton 中，即使使用 1D Grid，也可以通过整数除法（`//`）和取模（`%`）运算将一维 program_id 映射到多维索引。这是最常用的模式之一。

**基本原理**：

```
1D program_id → 2D (pid_m, pid_n) 的映射:

假设总共有 M_tiles × N_tiles 个 tile:
  M_tiles = ceil(M / BLOCK_M)  — M 方向的 tile 数
  N_tiles = ceil(N / BLOCK_N)  — N 方向的 tile 数

映射方式 1: 行优先（Row-major）
  pid_m = pid // N_tiles      ← 行索引 = 整除
  pid_n = pid % N_tiles       ← 列索引 = 取模

  例如: N_tiles = 4, pid = 7
    pid_m = 7 // 4 = 1
    pid_n = 7 % 4 = 3
    → (行 1, 列 3)

映射方式 2: 列优先（Column-major）
  pid_m = pid % M_tiles       ← 行索引 = 取模
  pid_n = pid // M_tiles      ← 列索引 = 整除

  例如: M_tiles = 4, pid = 7
    pid_m = 7 % 4 = 3
    pid_n = 7 // 4 = 1
    → (行 3, 列 1)
```

**行优先映射的可视化**：

```
1D Grid (行优先): grid = (M_tiles * N_tiles,)
M_tiles = 3, N_tiles = 4

pid:  0   1   2   3   4   5   6   7   8   9  10  11

映射到 2D:
      N_tiles = 4
      col 0  col 1  col 2  col 3
row 0 [  0 ] [  1 ] [  2 ] [  3 ]    ← pid // 4 = 0
row 1 [  4 ] [  5 ] [  6 ] [  7 ]    ← pid // 4 = 1
row 2 [  8 ] [  9 ] [ 10 ] [ 11 ]    ← pid // 4 = 2
        ↑ pid % 4 = 0, 1, 2, 3

公式:
  pid_m = pid // N_tiles
  pid_n = pid % N_tiles
```

### 5.5.2 常见的多维索引映射模式

**模式 1：简单行优先映射**

```python
@triton.jit
def row_major_mapping(
    X_ptr, Y_ptr, M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # 行优先映射
    pid_m = pid // num_pid_n    # 行索引
    pid_n = pid % num_pid_n     # 列索引

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x_ptrs = X_ptr + offs_m[:, None] * N + offs_n[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], x, mask=mask)
```

**模式 2：Swizzle 映射（L2 Cache 优化）**

```python
@triton.jit
def swizzle_mapping(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Swizzle 映射：让相邻的 program 访问相同的 B tile
    # 这样 B tile 可以在 L2 cache 中被复用
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        k_mask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask, other=0.0)
        accumulator += tl.dot(a, b)
        offs_k += BLOCK_K

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)
```

**Swizzle 映射的可视化**：

```
行优先映射 (无 Swizzle):
  pid_m 的变化是: 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, ...
  pid_n 的变化是: 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, ...
  → 相邻 program (如 pid=0 和 pid=1) 访问不同的 B tile
  → B tile 在 L2 cache 中无法复用

Swizzle 映射 (GROUP_SIZE_M = 2):
  pid_m 的变化是: 0, 1, 0, 1, 2, 3, 2, 3, ...
  pid_n 的变化是: 0, 0, 1, 1, 0, 0, 1, 1, ...
  → 相邻 program (如 pid=0 和 pid=1) 访问相同的 B tile (pid_n=0)
  → B tile 在 L2 cache 中被复用！

分组可视化:
  Group 0: [(0,0), (1,0)] [(0,1), (1,1)]    ← pid_m = 0,1 共享 B tile
  Group 1: [(2,0), (3,0)] [(2,1), (3,1)]    ← pid_m = 2,3 共享 B tile
```

**模式 3：Batch + 2D 索引映射**

```python
@triton.jit
def batch_mapping(
    A_ptr, B_ptr,
    batch, M, N,
    stride_ab, stride_am, stride_an,
    stride_bb, stride_bm, stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 从 1D pid 推导 3D 索引 (batch, pid_m, pid_n)
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    tiles_per_batch = num_pid_m * num_pid_n

    pid_b = pid // tiles_per_batch       # batch 索引
    remainder = pid % tiles_per_batch
    pid_m = remainder // num_pid_n       # 行索引
    pid_n = remainder % num_pid_n        # 列索引

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    batch_offset = pid_b * stride_ab
    a_ptrs = A_ptr + batch_offset + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    a = tl.load(a_ptrs, mask=mask, other=0.0)

    b_ptrs = B_ptr + pid_b * stride_bb + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
    tl.store(b_ptrs, a * 2.0, mask=mask)
```

### 5.5.3 2D Grid 直接索引 vs 1D Grid 推导索引

```python
# ---- 方式 1: 2D Grid，直接使用 program_id ----
@triton.jit
def direct_2d_kernel(X_ptr, Y_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)  # 直接得到行索引
    pid_n = tl.program_id(1)  # 直接得到列索引
    # ... 简单直观

grid_2d = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

# ---- 方式 2: 1D Grid，通过 // 和 % 推导 ----
@triton.jit
def derived_2d_kernel(X_ptr, Y_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n    # 需要额外的除法运算
    pid_n = pid % num_pid_n     # 需要额外的取模运算
    # ... 额外的计算开销，但支持 L2 cache 优化

grid_1d = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
```

**两种方式的对比**：

| 维度 | 2D Grid 直接索引 | 1D Grid 推导索引 |
|:---|:---|:---|
| **代码简洁性** | 更简洁 | 需要额外的 // 和 % |
| **可读性** | 直观 | 需要理解映射逻辑 |
| **L2 Cache 优化** | 不支持 swizzle | 支持 swizzle 映射 |
| **灵活性** | 最多 3D | 可以推导任意维度 |
| **性能** | 无额外计算 | 有少量额外计算（//、%） |

### 5.5.4 多 program_id 的使用模式

当使用 2D 或 3D Grid 时，可以同时使用多个 `program_id`：

```python
@triton.jit
def multi_pid_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 同时使用两个 program_id
    pid_m = tl.program_id(0)  # M 方向的 tile 索引
    pid_n = tl.program_id(1)  # N 方向的 tile 索引

    # 创建 2D 索引
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        k_mask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask, other=0.0)
        accumulator += tl.dot(a, b)
        offs_k += BLOCK_K

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)

# 2D Grid
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
multi_pid_kernel[grid](...)
```

<div data-component="MultiDimIndexCalculator"></div>

[组件：MultiDimIndexCalculator - 交互式计算从 1D pid 到多维索引的映射]

---

## 5.6 分块策略

### 5.6.1 数据分块（Tiling）的基本原理

数据分块是 GPU 编程中最核心的优化策略之一。其基本思想是将大的数据结构分解为小的块（tile），每个 program 处理一个或多个块。

```
未分块 vs 分块:

未分块（一个 program 处理整个矩阵）:
+---+---+---+---+---+---+---+---+
|   |   |   |   |   |   |   |   |
|   |   |   |   |   |   |   |   |  ← 一个 program 处理所有元素
|   |   |   |   |   |   |   |   |  ← 需要加载整个矩阵到寄存器
|   |   |   |   |   |   |   |   |  ← 寄存器不够！
+---+---+---+---+---+---+---+---+

分块（每个 program 处理一个 tile）:
+---+---+---+---+---+---+---+---+
| P0| P0| P1| P1| P2| P2| P3| P3|
| P0| P0| P1| P1| P2| P2| P3| P3|
+---+---+---+---+---+---+---+---+
| P4| P4| P5| P5| P6| P6| P7| P7|  ← 每个 program 只处理 2×2 的 tile
| P4| P4| P5| P5| P6| P6| P7| P7|  ← 寄存器足够
+---+---+---+---+---+---+---+---+

BLOCK_M = 2, BLOCK_N = 2
Grid = (2, 4) → 8 个 program
```

**分块的优势**：

| 优势 | 说明 |
|:---|:---|
| **寄存器适配** | 小块可以放入 GPU 寄存器，避免溢出到全局内存 |
| **数据复用** | 加载到寄存器的数据可以被多次使用（如矩阵乘法中的行/列复用） |
| **并行度** | 多个 program 可以同时处理不同的块，充分利用 GPU 的 SM |
| **缓存友好** | 小块数据更可能留在 L1/L2 缓存中 |

### 5.6.2 BLOCK_SIZE 的选择

BLOCK_SIZE 是 Triton 编程中最重要的 constexpr 参数之一。它直接影响寄存器使用、缓存效率和并行度。

**BLOCK_SIZE 选择的基本原则**：

```python
# 原则 1: BLOCK_SIZE 应该是 32 的倍数（对齐 Warp 大小）
# 一个 Warp 有 32 个线程，BLOCK_SIZE 应该是 32 的整数倍
BLOCK_M = 32    # 1 个 Warp
BLOCK_M = 64    # 2 个 Warp
BLOCK_M = 128   # 4 个 Warp
BLOCK_M = 256   # 8 个 Warp

# 原则 2: 对于 tl.dot，BLOCK_K 应该是 16 的倍数
# Tensor Core 要求 K 维度至少为 16（对于 FP16）
BLOCK_K = 16    # 最小对齐
BLOCK_K = 32    # 推荐
BLOCK_K = 64    # 更大的块

# 原则 3: BLOCK_SIZE 不能太大（寄存器溢出）
# 太大的 BLOCK_SIZE 会导致寄存器溢出到局部内存（慢）
# 通常 BLOCK_M × BLOCK_N ≤ 128 × 128 是安全的

# 原则 4: BLOCK_SIZE 不能太小（并行度不足）
# 太小的 BLOCK_SIZE 会导致 program 数量过多，调度开销增大
# 通常 BLOCK_M ≥ 64 是一个好的起点
```

**常见 BLOCK_SIZE 组合**：

| 任务 | BLOCK_M | BLOCK_N | BLOCK_K | 说明 |
|:---|:---|:---|:---|:---|
| **向量运算** | 1024 | — | — | 一维分块 |
| **矩阵加法** | 64 | 64 | — | 二维分块 |
| **矩阵乘法 (小)** | 64 | 64 | 32 | 适合小矩阵 |
| **矩阵乘法 (中)** | 128 | 128 | 32 | 通用选择 |
| **矩阵乘法 (大)** | 256 | 128 | 64 | 适合大矩阵，需要更多寄存器 |
| **Attention** | 128 | 128 | — | Q/K/V 的分块 |

**BLOCK_SIZE 对性能的影响**：

```python
@triton.jit
def benchmark_kernel(X_ptr, Y_ptr, M, N,
                     BLOCK_M: tl.constexpr,
                     BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x_ptrs = X_ptr + offs_m[:, None] * N + offs_n[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], x * 2.0, mask=mask)

# 不同 BLOCK_SIZE 的效果:
# BLOCK_M=32, BLOCK_N=32: program 数多，每个 program 轻量
# BLOCK_M=128, BLOCK_N=128: program 数少，每个 program 重量
# 需要通过 benchmark 找到最优值
```

### 5.6.3 处理非对齐维度

当数据维度不是 BLOCK_SIZE 的整数倍时，需要处理边界情况。

**方法 1：Mask 处理（推荐）**

```python
@triton.jit
def mask_boundary_kernel(
    X_ptr, Y_ptr, M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 创建二维 mask：True 表示有效索引，False 表示越界
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # 加载时使用 mask 和 other
    x_ptrs = X_ptr + offs_m[:, None] * N + offs_n[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0)  # 越界位置填充 0.0

    # 存储时也使用 mask（越界位置不写入）
    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], x * 2.0, mask=mask)

# 当 M=100, BLOCK_M=64 时:
# pid_m=0: offs_m = [0, 1, ..., 63]   → 全部有效
# pid_m=1: offs_m = [64, 65, ..., 127] → 只有 [64, ..., 99] 有效
#            mask = [True, True, ..., True, False, ..., False]
#                          36 个 True       28 个 False
```

**方法 2：Block Pointer 的 boundary_check**

```python
@triton.jit
def block_ptr_boundary_kernel(
    X_ptr, Y_ptr, M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block Pointer 自动处理边界
    X_block_ptr = tl.make_block_ptr(
        base=X_ptr, shape=(M, N), strides=(stride_xm, stride_xn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )

    # boundary_check 自动处理越界
    x = tl.load(X_block_ptr, boundary_check=(0, 1), padding_option="zero")

    Y_block_ptr = tl.make_block_ptr(
        base=Y_ptr, shape=(M, N), strides=(stride_ym, stride_yn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )
    tl.store(Y_block_ptr, x * 2.0, boundary_check=(0, 1))
```

**方法 3：Padding（填充到对齐大小）**

```python
import torch

def padded_kernel_launch(X, M, N, BLOCK_M, BLOCK_N):
    # 计算填充后的大小
    M_padded = triton.cdiv(M, BLOCK_M) * BLOCK_M
    N_padded = triton.cdiv(N, BLOCK_N) * BLOCK_N

    # 填充输入矩阵
    if M_padded != M or N_padded != N:
        X_padded = torch.zeros((M_padded, N_padded), device=X.device, dtype=X.dtype)
        X_padded[:M, :N] = X
    else:
        X_padded = X

    Y_padded = torch.empty_like(X_padded)

    # 启动 kernel（不需要 mask）
    grid = (triton.cdiv(M_padded, BLOCK_M), triton.cdiv(N_padded, BLOCK_N))
    simple_copy_kernel[grid](X_padded, Y_padded, M_padded, N_padded,
                              BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

    # 裁剪回原始大小
    return Y_padded[:M, :N]
```

**三种方法的对比**：

| 方法 | 优点 | 缺点 | 适用场景 |
|:---|:---|:---|:---|
| **Mask** | 灵活、无额外内存 | 每次 load/store 都需要 mask | 通用场景 |
| **Block Pointer** | 代码简洁、编译器优化 | API 稍复杂 | 规则块访问 |
| **Padding** | kernel 内无边界检查 | 额外内存开销 | 批量处理、避免边界检查开销 |

### 5.6.4 分块策略的实际应用

**矩阵乘法的三层分块**：

```
矩阵乘法 C = A × B 的分块策略:

原始矩阵:
  A: (M, K)    B: (K, N)    C: (M, N)

第一层分块: Program 级别
  每个 program 负责 C 的一个 (BLOCK_M × BLOCK_N) 的 tile
  A_tile: (BLOCK_M, K)    B_tile: (K, BLOCK_N)

第二层分块: K 维度循环
  将 K 维度分成大小为 BLOCK_K 的块
  A_tile: (BLOCK_M, BLOCK_K)    B_tile: (BLOCK_K, BLOCK_N)

第三层分块: Tensor Core 硬件
  编译器自动将 BLOCK_M × BLOCK_N 分解为 Tensor Core 指令
  例如 16×8 的 MMA 指令

分块层次:
+------------------------------------------+
| Program (pid_m, pid_n)                    |
|  C_tile: BLOCK_M × BLOCK_N               |
|  +--------------------------------------+|
|  | K 循环 (k = 0, 1, ..., K/BLOCK_K-1) ||
|  |  A_tile: BLOCK_M × BLOCK_K           ||
|  |  B_tile: BLOCK_K × BLOCK_N           ||
|  |  +----------------------------------+||
|  |  | Tensor Core MMA                   ||
|  |  |  16×8 子块                        ||
|  |  +----------------------------------+||
|  +--------------------------------------+|
+------------------------------------------+
```

<div data-component="TilingStrategyDiagram"></div>

[组件：TilingStrategyDiagram - 动画展示三层分块策略的执行过程]

---

## 5.7 Grid Stride Loop

### 5.7.1 什么是 Grid Stride Loop

Grid Stride Loop 是一种处理"数据量大于 Grid 中 program 数量"情况的模式。当 `num_programs < 数据维度` 时，每个 program 需要处理多个数据块。

```
Grid Stride Loop 的基本思想:

假设 N = 10000, BLOCK = 1024, Grid 只启动 4 个 program

传统方式（一个 program 一个块）:
  需要 ceil(10000/1024) = 10 个 program
  → Grid = (10,)

Grid Stride Loop（一个 program 多个块）:
  只启动 4 个 program，每个 program 处理多个块
  Program 0: 处理块 0, 4, 8  (stride = 4 = grid_size)
  Program 1: 处理块 1, 5, 9
  Program 2: 处理块 2, 6
  Program 3: 处理块 3, 7
```

**Grid Stride Loop 的基本模式**：

```python
@triton.jit
def grid_stride_basic(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    # 获取当前 program 的 ID 和总 program 数
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # Grid Stride Loop: 每次迭代处理一个 BLOCK，步进 num_programs 个 BLOCK
    for i in range(pid, tl.cdiv(N, BLOCK), num_programs):
        # i 是当前处理的块索引
        offsets = i * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N

        x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
        tl.store(Y_ptr + offsets, x * 2.0, mask=mask)

# 启动: grid = (4,)  ← 只启动 4 个 program
# 每个 program 会循环处理多个块
```

### 5.7.2 Grid Stride Loop 的执行流程

```
N = 10000, BLOCK = 1024, Grid = (4,)

总块数 = ceil(10000 / 1024) = 10
每个 program 处理的块:

Program 0 (pid=0):
  迭代 i=0: offsets = [0, 1, ..., 1023]
  迭代 i=4: offsets = [4096, 4097, ..., 5119]
  迭代 i=8: offsets = [8192, 8193, ..., 9215]

Program 1 (pid=1):
  迭代 i=1: offsets = [1024, 1025, ..., 2047]
  迭代 i=5: offsets = [5120, 5121, ..., 6143]
  迭代 i=9: offsets = [9216, 9217, ..., 9999]  ← 最后一块，部分越界

Program 2 (pid=2):
  迭代 i=2: offsets = [2048, 2049, ..., 3071]
  迭代 i=6: offsets = [6144, 6145, ..., 7167]

Program 3 (pid=3):
  迭代 i=3: offsets = [3072, 3073, ..., 4095]
  迭代 i=7: offsets = [7168, 7169, ..., 8191]

时间线:
  SM 0: [块0] [块4] [块8]
  SM 1: [块1] [块5] [块9]
  SM 2: [块2] [块6]
  SM 3: [块3] [块7]
  → 所有 SM 均匀分担负载
```

### 5.7.3 Grid Stride Loop 的优势

**优势 1：控制 program 数量**

```python
# 当数据量很大时，直接启动所有块可能导致 program 数量过多
# Grid Stride Loop 允许限制 program 数量

N = 1_000_000_000  # 10 亿元素
BLOCK = 1024

# 不使用 Grid Stride Loop:
# 需要 ceil(10^9 / 1024) ≈ 976,563 个 program
# 可能导致调度开销过大

# 使用 Grid Stride Loop:
# 只启动 1024 个 program，每个处理约 954 个块
# 调度开销可控
```

**优势 2：更好的负载均衡**

```python
# Grid Stride Loop 天然具有负载均衡能力
# 因为每个 program 处理的块数量几乎相同
# 不会出现某些 program 提前完成、等待其他 program 的情况
```

**优势 3：支持 L2 Cache 优化**

```python
# 通过控制 program 数量和 stride 模式
# 可以优化 L2 Cache 的复用
```

### 5.7.4 Grid Stride Loop 的完整实现

**1D Grid Stride Loop**：

```python
@triton.jit
def grid_stride_1d(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # Grid Stride Loop
    for block_idx in range(pid, tl.cdiv(N, BLOCK), num_programs):
        offsets = block_idx * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N

        x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

        # 计算
        y = x * 2.0 + 1.0

        tl.store(Y_ptr + offsets, y, mask=mask)
```

**2D Grid Stride Loop（矩阵行方向）**：

```python
@triton.jit
def grid_stride_2d_rows(
    A_ptr, B_ptr, M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)  # 列方向：每个 program 负责一列 tile
    pid_m_start = tl.program_id(1)  # 行方向：起始行 tile
    num_programs_m = tl.num_programs(1)  # 行方向的 program 数

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    # 在行方向上使用 Grid Stride Loop
    for pid_m in range(pid_m_start, tl.cdiv(M, BLOCK_M), num_programs_m):
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = offs_m < M
        mask = m_mask[:, None] & n_mask[None, :]

        a_ptrs = A_ptr + offs_m[:, None] * N + offs_n[None, :]
        a = tl.load(a_ptrs, mask=mask, other=0.0)

        b = a * 2.0

        b_ptrs = B_ptr + offs_m[:, None] * N + offs_n[None, :]
        tl.store(b_ptrs, b, mask=mask)
```

**K 维度的 Grid Stride Loop（矩阵乘法）**：

```python
@triton.jit
def matmul_grid_stride(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_STEPS: tl.constexpr,  # K 维度的总步数（constexpr 以便循环展开）
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K 维度的循环（不是 Grid Stride，而是简单的 for 循环）
    for k in range(K_STEPS):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)

        accumulator += tl.dot(a, b)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)
```

<div data-component="GridStrideLoopAnimation"></div>

[组件：GridStrideLoopAnimation - 动画展示 Grid Stride Loop 的执行过程和负载均衡]

---

## 5.8 Launch 参数详解

### 5.8.1 Launch 参数总览

Triton kernel 的启动参数（Launch Parameters）控制 kernel 在 GPU 上的执行方式。这些参数可以在两个地方指定：`@triton.jit` 装饰器中（默认值）或 kernel 启动时（覆盖默认值）。

```python
# 方式 1: 在 @triton.jit 中指定默认值
@triton.jit(num_warps=4, num_stages=3)
def my_kernel(...):
    pass

# 方式 2: 在启动时覆盖
my_kernel[grid](..., num_warps=8, num_stages=2)

# 方式 3: 混合使用
@triton.jit(num_warps=4)  # 默认 4 个 warp
def my_kernel(...):
    pass

# 大多数时候使用默认的 4 个 warp
my_kernel[grid](...)

# 特殊情况下覆盖为 8 个 warp
my_kernel[grid](..., num_warps=8)
```

### 5.8.2 num_warps — 每个 Program 的 Warp 数

`num_warps` 指定每个 program 使用多少个 Warp（1 Warp = 32 个线程）。它直接影响 program 的并行度、寄存器使用和 occupancy。

```python
@triton.jit
def num_warps_demo(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    tl.store(Y_ptr + offsets, x * 2.0, mask=mask)

# num_warps 与 BLOCK_SIZE 的关系:
# BLOCK 决定了每个 program 处理的元素数
# num_warps 决定了处理这些元素的线程数

# 当 BLOCK=128, num_warps=4:
#   每个 warp 处理 128/4 = 32 个元素（刚好 1 个 warp 32 个线程）
#   → 每个线程处理 1 个元素

# 当 BLOCK=256, num_warps=4:
#   每个 warp 处理 256/4 = 64 个元素
#   → 每个线程处理 2 个元素

# 当 BLOCK=256, num_warps=8:
#   每个 warp 处理 256/8 = 32 个元素
#   → 每个线程处理 1 个元素
```

**num_warps 的选择指南**：

| num_warps | 线程数 | 寄存器/warp | Occupancy | 适用场景 |
|:---|:---|:---|:---|:---|
| **1** | 32 | 最多 | 最高 | 极小的 BLOCK，简单计算 |
| **2** | 64 | 较多 | 较高 | 小 BLOCK (64-128) |
| **4** | 128 | 中等 | 中等 | **默认值**，通用场景 |
| **8** | 256 | 较少 | 较低 | 大 BLOCK (256-512)，需要更多并行度 |

**num_warps 对性能的影响**：

```python
# num_warps 太小 (如 num_warps=1, BLOCK=256):
#   - 每个线程处理 8 个元素 (256/32)
#   - 需要循环 8 次，每个元素的计算串行化
#   - 但每个 warp 有更多寄存器可用
#   - 适合计算密集型 kernel

# num_warps 太大 (如 num_warps=8, BLOCK=64):
#   - 每个 warp 只处理 8 个元素 (64/8)
#   - 大部分线程空闲
#   - 浪费硬件资源
#   - 不推荐

# 推荐: num_warps = BLOCK_SIZE / 32
#   - 每个线程处理 1 个元素
#   - 最大化并行度
```

### 5.8.3 num_stages — Software Pipeline Stages 数

`num_stages` 控制 Triton 编译器生成的 software pipeline 的 stage 数。Software pipeline 是一种通过重叠计算和内存访问来提升性能的技术。

```
Software Pipeline 的原理:

num_stages=1 (无流水线):
  [Load k=0] [Compute k=0] [Load k=1] [Compute k=1] [Load k=2] [Compute k=2]
  → 加载和计算串行执行

num_stages=2 (双缓冲):
  [Load k=0] [Load k=1 + Compute k=0] [Load k=2 + Compute k=1] [Compute k=2]
  → 下一块的加载与当前块的计算重叠

num_stages=3 (三缓冲):
  [Load k=0] [Load k=1] [Load k=2 + Compute k=0] [Load k=3 + Compute k=1] ...
  → 更深的流水线，更多的加载与计算重叠
```

**num_stages 的选择指南**：

| num_stages | 效果 | 适用场景 |
|:---|:---|:---|
| **1** | 无流水线 | 计算密集型、内存访问少 |
| **2** | 双缓冲 | 通用场景，加载和计算可以重叠 |
| **3** | 三缓冲 | 内存延迟高、计算量大的场景 |
| **4+** | 更深流水线 | 特殊优化场景，通常收益递减 |

```python
@triton.jit(num_stages=3)  # 使用 3 级流水线
def pipelined_matmul(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 编译器会自动将这个循环转换为 software pipeline
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        k_mask = offs_k[None, :] < K - k * BLOCK_K

        a = tl.load(a_ptrs, mask=k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask, other=0.0)

        accumulator += tl.dot(a, b)
        offs_k += BLOCK_K

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)
```

### 5.8.4 maxnreg — 最大寄存器数

`maxnreg` 限制每个 thread 使用的最大寄存器数。它通过牺牲每个线程的寄存器数量来提高 occupancy（每个 SM 可以同时运行更多 program）。

```python
# maxnreg 的使用场景:
# 当 kernel 使用太多寄存器时，occupancy 会下降
# 通过限制 maxnreg，可以提高 occupancy

# 示例: 一个寄存器压力很大的 kernel
@triton.jit(maxnreg=128)  # 限制每个线程最多使用 128 个寄存器
def register_limited_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # ... 复杂的计算，使用大量寄存器 ...
    pass
```

**maxnreg 与 Occupancy 的关系**：

```
GPU 每个 SM 的寄存器总数是固定的（例如 65536 个）

当 maxnreg=256, num_warps=4 (128 线程):
  每个 program 使用 128 × 256 = 32768 个寄存器
  每 SM 可以运行 65536 / 32768 = 2 个 program

当 maxnreg=128, num_warps=4 (128 线程):
  每个 program 使用 128 × 128 = 16384 个寄存器
  每 SM 可以运行 65536 / 16384 = 4 个 program（occupancy 提高）

但 maxnreg 太小会导致寄存器溢出到局部内存（慢）
```

### 5.8.5 debug 参数

`debug=True` 让 Triton 编译器生成更易读的代码和更详细的错误信息，但会显著降低性能。

```python
@triton.jit(debug=True)  # 开启调试模式
def debug_kernel(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    tl.store(Y_ptr + offsets, x * 2.0, mask=mask)

# debug=True 的效果:
# 1. 生成更易读的 Triton IR 和 PTX
# 2. 添加边界检查和错误信息
# 3. 禁用某些优化（方便调试）
# 4. 性能下降显著，仅在调试时使用
```

### 5.8.6 Launch 参数的完整示例

```python
import torch
import triton
import triton.language as tl

@triton.jit
def optimized_matmul(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # L2 cache 优化的 swizzle 映射
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        k_mask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=k_mask, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask, other=0.0)
        accumulator += tl.dot(a, b)
        offs_k += BLOCK_K

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator.to(tl.float16), mask=c_mask)


def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_SIZE_M = 8

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    # 启动时指定 launch 参数
    optimized_matmul[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=4,     # 4 个 warp = 128 线程
        num_stages=3,    # 3 级流水线
        # maxnreg=128,   # 可选：限制寄存器数
        # debug=False,   # 可选：关闭调试模式
    )

    return C
```

### 5.8.7 Launch 参数调优指南

**调优流程**：

```
1. 确定 BLOCK_SIZE:
   ├── 根据数据维度选择合适的 BLOCK_M, BLOCK_N, BLOCK_K
   ├── BLOCK_M, BLOCK_N 通常是 32 的倍数
   ├── BLOCK_K 通常是 16 的倍数
   └── 初始值: BLOCK_M=128, BLOCK_N=128, BLOCK_K=32

2. 确定 num_warps:
   ├── 默认: num_warps = 4
   ├── 如果 BLOCK_SIZE 很大: num_warps = 8
   ├── 如果 BLOCK_SIZE 很小: num_warps = 2
   └── 经验法则: num_warps ≈ BLOCK_SIZE / 32

3. 确定 num_stages:
   ├── 默认: num_stages = 3
   ├── 如果计算密集: num_stages = 2
   ├── 如果内存密集: num_stages = 4
   └── 需要 benchmark 确定最优值

4. 确定 maxnreg（可选）:
   ├── 通常不需要手动设置
   ├── 如果 occupancy 太低: 尝试 maxnreg = 128 或 64
   └── 注意：太小会导致寄存器溢出

5. Benchmark:
   ├── 测试不同参数组合
   ├── 使用 torch.cuda.Event 计时
   └── 选择性能最优的参数
```

<div data-component="LaunchParameterTuner"></div>

[组件：LaunchParameterTuner - 交互式调整 Launch 参数并查看对性能的影响]

---

## 本章小结

本章系统介绍了 Triton kernel 的程序结构与 Grid/Block 映射机制。核心要点如下：

1. **Kernel 基本结构**：Triton kernel 由 `@triton.jit` 装饰器标记，参数分为三类：指针参数（指向 GPU 全局内存）、标量参数（运行时数值）和 constexpr 参数（编译期常量）。constexpr 参数会触发编译特化，每个值组合生成独立的编译产物。

2. **program_id 与 num_programs**：`tl.program_id(axis)` 返回当前 program 在指定轴上的 ID（axis=0/1/2 对应 x/y/z），`tl.num_programs(axis)` 返回该轴的总 program 数。这两个函数是确定"当前 program 处理哪块数据"的核心工具。

3. **Grid 的定义与启动**：Grid 可以用 tuple 形式（`(Gx, Gy, Gz)`）或 callable lambda 形式（`lambda META: (Gx, Gy, Gz)`）定义。1D Grid 适合线性化处理，2D Grid 适合矩阵运算，3D Grid 适合批量操作。

4. **Triton 与 CUDA 的映射**：Triton 的 program 对应 CUDA 的 Block，program_id 对应 blockIdx。Triton 隐藏了 Thread 级别的管理——用户不需要手动处理 threadIdx、`__syncthreads()`、共享内存分配等细节。

5. **多维索引计算**：从 1D program_id 推导多维索引使用 `//`（整除）和 `%`（取模）运算。Swizzle 映射通过改变 program 的调度顺序来优化 L2 Cache 命中率。

6. **分块策略**：BLOCK_SIZE 应该是 32 的倍数（对齐 Warp），BLOCK_K 应该是 16 的倍数（对齐 Tensor Core）。非对齐维度通过 mask、boundary_check 或 padding 处理。

7. **Grid Stride Loop**：当数据量大于 Grid 中的 program 数量时，每个 program 通过 Grid Stride Loop 处理多个数据块，实现负载均衡。

8. **Launch 参数**：`num_warps`（每个 program 的 Warp 数，默认 4）、`num_stages`（software pipeline 级数，默认 3）、`maxnreg`（最大寄存器数）、`debug`（调试模式）共同决定了 kernel 的执行特性。

---

## 思考题

### 概念理解题

1. **constexpr 的编译特化**：假设一个 kernel 有两个 constexpr 参数 `BLOCK_M` 和 `BLOCK_N`，分别可以取值 64、128、256。理论上最多会生成多少个不同的编译版本？在实际使用中，如何减少编译版本的数量？

2. **program_id 的本质**：`tl.program_id(0)` 在底层等价于 CUDA 的 `blockIdx.x`。为什么不直接暴露 `blockIdx.x` 和 `threadIdx.x`？Triton 的抽象层带来了什么好处和限制？

3. **Grid 维度选择**：在什么情况下应该使用 2D Grid 而不是 1D Grid + `//` `%` 推导？请从代码可读性、性能和灵活性三个角度分析。

### 实践题

4. **1D 到 3D 索引映射**：编写一个 kernel，使用 1D Grid 处理一个 3D 张量 `(B, M, N)`。要求使用 `//` 和 `%` 运算从 1D program_id 推导出 `(pid_b, pid_m, pid_n)` 三维索引。

5. **Grid Stride Loop 实现**：编写一个向量加法 kernel，只启动 8 个 program，但需要处理 100 万个元素。使用 Grid Stride Loop 确保所有元素都被处理。

6. **BLOCK_SIZE 调优**：对于一个 4096×4096 的矩阵乘法，分别测试 `BLOCK_M=64, BLOCK_N=64, BLOCK_K=16`、`BLOCK_M=128, BLOCK_N=128, BLOCK_K=32` 和 `BLOCK_M=256, BLOCK_N=128, BLOCK_K=64` 三种配置的性能。分析哪种配置最优，为什么。

### 设计思考题

7. **Swizzle 映射的设计**：Swizzle 映射通过改变 program 的调度顺序来优化 L2 Cache。请设计一个场景，说明为什么行优先映射会导致 L2 Cache miss，而 Swizzle 映射可以解决这个问题。

8. **num_warps 与 BLOCK_SIZE 的关系**：为什么推荐 `num_warps ≈ BLOCK_SIZE / 32`？如果 `num_warps` 远大于或远小于这个值，会发生什么？

9. **Triton 的抽象层次**：Triton 隐藏了 CUDA 的 Thread 级别抽象，只暴露 Program 级别。这种设计在什么场景下会成为限制？你认为是否有必要提供一种"半自动"的 Thread 级别控制？

### 进阶题

10. **Launch 参数自动调优**：设计一个自动调优脚本，遍历不同的 `BLOCK_M`、`BLOCK_N`、`BLOCK_K`、`num_warps`、`num_stages` 组合，找到最优的 Launch 参数。使用 `triton.testing.Benchmark` 或自定义 benchmark 框架。

11. **自定义 Swizzle 模式**：Triton 的 `GROUP_SIZE_M` 参数实现了一种简单的 Swizzle 模式。请设计一种更复杂的 Swizzle 策略，例如根据 L2 Cache 大小动态调整 group 大小。

12. **Grid Stride Loop 与 Occupancy**：分析 Grid Stride Loop 如何影响 GPU 的 occupancy。在什么情况下，使用 Grid Stride Loop（较少 program，每个处理多个块）比直接启动所有 program 更优？

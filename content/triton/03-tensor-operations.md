---
title: "Chapter 3: 张量操作与计算原语"
description: "掌握 Triton 的张量创建、加载与存储、算术运算、归约操作、矩阵乘法、类型转换、广播机制与高级操作原语"
date: "2026-06-11"
---

# Chapter 3: 张量操作与计算原语

> **学习目标**：
> - 掌握 Triton 中张量创建的多种方式：`tl.zeros`、`tl.full`、`tl.arange` 及 `tl.constexpr` 编译期常量的使用
> - 理解 `tl.load` 与 `tl.store` 的完整参数语义，包括 `mask`、`other`、`boundary_check`、`cache_modifier`、`eviction_policy`，以及内存合并访问模式
> - 熟练使用逐元素算术运算（加减乘除、比较、逻辑）与一元数学函数（`tl.exp`、`tl.log`、`tl.sqrt` 等），理解类型提升规则
> - 掌握归约操作（`tl.sum`、`tl.min`、`tl.max`、`tl.reduce`）的使用方法与底层 Warp Shuffle 并行实现原理
> - 深入理解 `tl.dot` 矩阵乘法的 Tensor Core 映射机制与支持的 dtype 组合（f16×f16、bf16×bf16、tf32×tf32）
> - 掌握类型转换（隐式提升、`.to()` 显式转换、FP8 类型）、广播机制与高级操作（`tl.trans`、`tl.cat`、`tl.join`）

---

## 3.1 张量创建

在 Triton kernel 中，所有计算都以**张量 (Tensor)** 为基本单位。与 PyTorch 不同，Triton 中的张量不是堆上的 Python 对象——它们是编译器 IR 中的**值类型 (value type)**，在编译时就被映射到 GPU 寄存器或共享内存上。理解张量的创建方式是编写 Triton kernel 的第一步。

### 3.1.1 tl.zeros — 零张量

`tl.zeros` 创建一个所有元素为零的张量，通常用于初始化累加器。

```python
@triton.jit
def accumulate_kernel(
    X_ptr, Y_ptr, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 创建一个 BLOCK_M × BLOCK_N 的二维零张量，数据类型为 float32
    # 这个张量在编译时确定 shape，运行时映射到寄存器
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 也可以创建一维零张量
    row_buffer = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # 标量零张量（0 维）
    scalar_zero = tl.zeros((1,), dtype=tl.float32)
```

**`tl.zeros` 参数详解**：

| 参数 | 类型 | 说明 |
|:---|:---|:---|
| `shape` | `tuple[int, ...]` | 张量的形状，必须是编译期常量（`tl.constexpr`） |
| `dtype` | `tl.dtype` | 数据类型，默认为 `tl.float32` |

**核心要点**：
- `shape` 中的每个维度大小必须在编译期可知，不能使用运行时变量
- 常见的 dtype 包括 `tl.float16`、`tl.bfloat16`、`tl.float32`、`tl.int32`、`tl.int64`
- 创建的零张量在底层被优化为寄存器清零指令，开销极小

**常见陷阱**：

```python
@triton.jit
def bad_kernel(X, BLOCK: tl.constexpr):
    N = tl.load(X)  # N 是运行时值
    # 错误！N 不是 constexpr，不能作为 shape
    # result = tl.zeros((N,), dtype=tl.float32)  # 编译错误！

    # 正确做法：使用 constexpr 作为 shape，配合 mask 处理边界
    result = tl.zeros((BLOCK,), dtype=tl.float32)
```

<div data-component="TensorCreationZeros"></div>

[组件：TensorCreationZeros - 可视化 tl.zeros 的寄存器分配过程]

### 3.1.2 tl.full — 常量填充张量

`tl.full` 创建一个所有元素为指定常量值的张量，比 `tl.zeros` + 逐元素赋值更高效。

```python
@triton.jit
def fill_kernel(
    Y_ptr, N,
    BLOCK: tl.constexpr,
):
    # 创建全 1 张量
    ones = tl.full((BLOCK,), value=1.0, dtype=tl.float32)

    # 创建全 -1 张量（常用于初始化 mask 的 "other" 值）
    neg_ones = tl.full((BLOCK,), value=-1.0, dtype=tl.float32)

    # 创建二维常量张量
    # 例如：初始化一个 128×64 的张量，所有值为 0.5
    bias = tl.full((128, 64), value=0.5, dtype=tl.float16)

    # 创建整数常量张量
    indices = tl.full((BLOCK,), value=42, dtype=tl.int32)
```

**`tl.full` 参数详解**：

| 参数 | 类型 | 说明 |
|:---|:---|:---|
| `shape` | `tuple[int, ...]` | 张量形状，编译期常量 |
| `value` | `int` 或 `float` | 填充值，编译期常量 |
| `dtype` | `tl.dtype` | 数据类型 |

**与 `tl.zeros` 的对比**：

| 特性 | `tl.zeros` | `tl.full` |
|:---|:---|:---|
| 填充值 | 固定为 0 | 任意常量 |
| 底层实现 | 寄存器清零 | 寄存器赋值 |
| 使用场景 | 累加器初始化 | 常量偏置、默认值 |
| 等价写法 | `tl.full(..., 0, ...)` | — |

### 3.1.3 tl.arange — 索引序列

`tl.arange` 创建一个连续整数序列，是 Triton 中构建索引的核心工具。几乎每个 Triton kernel 都会用到它。

```python
@triton.jit
def index_demo_kernel(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    # 创建一维索引: [0, 1, 2, ..., BLOCK-1]
    # start 和 end 都必须是编译期常量
    offsets = tl.arange(0, BLOCK)

    # 通过加上 program 的起始偏移，得到全局索引
    pid = tl.program_id(0)
    global_offsets = pid * BLOCK + offsets  # [pid*BLOCK, pid*BLOCK+1, ...]

    # 创建 mask 防止越界
    mask = global_offsets < N

    # 使用索引进行加载
    x = tl.load(X_ptr + global_offsets, mask=mask, other=0.0)
    tl.store(Y_ptr + global_offsets, x * 2.0, mask=mask)
```

**`tl.arange` 的关键约束**：

```python
# 正确：start 和 end 都是 constexpr
offsets = tl.arange(0, 128)           # [0, 1, ..., 127]
offsets = tl.arange(0, BLOCK_SIZE)    # BLOCK_SIZE 是 tl.constexpr

# 错误：不能使用运行时值
# offsets = tl.arange(0, N)           # N 是运行时变量 → 编译错误！

# 正确但需要注意：start 不一定从 0 开始
offsets = tl.arange(64, 128)          # [64, 65, ..., 127]
```

**构建二维索引的技巧**：

```python
@triton.jit
def matmul_index_kernel(A, B, C, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 创建一维索引
    offs_m = tl.arange(0, BLOCK_M)  # shape: (BLOCK_M,)
    offs_n = tl.arange(0, BLOCK_N)  # shape: (BLOCK_N,)

    # 通过 None 索引实现广播，创建二维索引网格
    # offs_m[:, None]  → shape: (BLOCK_M, 1) — 列向量
    # offs_n[None, :]  → shape: (1, BLOCK_N) — 行向量
    # 两者相加 → shape: (BLOCK_M, BLOCK_N) — 二维网格
    offs_m_2d = offs_m[:, None]  # 广播为列
    offs_n_2d = offs_n[None, :]  # 广播为行

    # 计算二维全局索引
    m_indices = pid_m * BLOCK_M + offs_m_2d  # shape: (BLOCK_M, BLOCK_N)
    n_indices = pid_n * BLOCK_N + offs_n_2d  # shape: (BLOCK_M, BLOCK_N)

    # 使用二维索引加载数据
    a_ptrs = A + m_indices * N + n_indices  # 计算线性地址
    data = tl.load(a_ptrs)
```

<div data-component="ArangeIndexVisualization"></div>

[组件：ArangeIndexVisualization - 可视化 tl.arange 与 None 广播构建二维索引的过程]

### 3.1.4 tl.constexpr — 编译期常量

`tl.constexpr` 是 Triton 类型系统中的一个重要概念。标记为 `tl.constexpr` 的参数在**编译时**确定值，编译器会为每个不同的 constexpr 值生成**独立的编译产物**。

```python
@triton.jit
def constexpr_demo(
    X, Y, N,
    BLOCK_SIZE: tl.constexpr,    # 编译期常量：block 大小
    DTYPE: tl.constexpr,         # 编译期常量：数据类型
    USE_MASK: tl.constexpr,      # 编译期常量：是否使用 mask
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # DTYPE 是 constexpr，可以用作 tl.zeros 的参数
    acc = tl.zeros((BLOCK_SIZE,), dtype=DTYPE)

    if USE_MASK:
        # USE_MASK 是 constexpr，这个分支在编译时决定
        # 不会生成条件跳转指令
        mask = offsets < N
        x = tl.load(X + offsets, mask=mask, other=0.0)
    else:
        # 假设 N 总是 BLOCK_SIZE 的倍数，不需要 mask
        x = tl.load(X + offsets)

    tl.store(Y + offsets, x)
```

**constexpr 的编译行为**：

```python
# 调用 1：BLOCK_SIZE=128 → 编译版本 A
constexpr_demo[grid](X, Y, N, BLOCK_SIZE=128, DTYPE=tl.float32, USE_MASK=True)

# 调用 2：BLOCK_SIZE=256 → 编译版本 B（与 A 完全不同的 PTX 代码）
constexpr_demo[grid](X, Y, N, BLOCK_SIZE=256, DTYPE=tl.float32, USE_MASK=True)

# 调用 3：BLOCK_SIZE=128, DTYPE=tl.float16 → 编译版本 C
constexpr_demo[grid](X, Y, N, BLOCK_SIZE=128, DTYPE=tl.float16, USE_MASK=False)
```

**constexpr vs 普通参数**：

| 特性 | constexpr 参数 | 普通参数 |
|:---|:---|:---|
| **值确定时间** | 编译时 | 运行时 |
| **可用于 shape** | 可以 | 不可以 |
| **可用于 dtype** | 可以 | 不可以 |
| **条件分支** | 编译时裁剪 | 生成条件跳转 |
| **编译产物** | 每个值组合一份 | 共享同一份 |
| **性能影响** | 无运行时开销 | 可能有分支开销 |

**最佳实践**：

```python
@triton.jit
def best_practice_kernel(
    # 指针和维度 → 普通参数（运行时传入）
    X_ptr, Y_ptr, M, N,

    # Tile 大小 → constexpr（编译时确定，影响循环展开和寄存器分配）
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,

    # 数据类型 → constexpr（编译时确定类型大小和指令选择）
    DTYPE: tl.constexpr,

    # 功能开关 → constexpr（编译时裁剪不需要的代码路径）
    HAS_BIAS: tl.constexpr,
):
    pass  # kernel 实现...
```

### 3.1.5 动态 Shape vs 静态 Shape

Triton 中的 shape 概念分为两个层次：

**静态 Shape（编译期已知）**：

```python
@triton.jit
def static_shape_kernel(X, Y, BLOCK: tl.constexpr):
    # BLOCK 是 constexpr → 这个张量的 shape 在编译时完全确定
    # 编译器知道确切的寄存器数量，可以做最优分配
    tile = tl.zeros((BLOCK,), dtype=tl.float32)  # 静态 shape
    offsets = tl.arange(0, BLOCK)                 # 静态 shape

    # tl.dot 的输出 shape 也由输入 shape 静态决定
    a = tl.zeros((64, 32), dtype=tl.float16)
    b = tl.zeros((32, 64), dtype=tl.float16)
    c = tl.dot(a, b)  # shape: (64, 64) — 静态已知
```

**动态 Shape（运行时部分已知）**：

```python
@triton.jit
def dynamic_shape_kernel(X, Y, N, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)  # 静态 shape: (BLOCK,)
    mask = offsets < N              # N 是运行时值

    # tl.load 的实际加载数量取决于 mask 中 True 的个数
    # 但张量的 shape 始终是 (BLOCK,)，只是部分元素是 "garbage"
    data = tl.load(X + offsets, mask=mask, other=0.0)

    # 归约时，mask 保证了只对有效元素求和
    total = tl.sum(data, axis=0)  # 结果是标量，但 mask 影响计算结果
```

**关键理解**：在 Triton 中，**张量的 shape 本身始终是静态的**（由 constexpr 决定）。所谓的 "动态" 指的是实际有效元素的数量是运行时确定的，通过 `mask` 机制来处理。

---

## 3.2 加载与存储

`tl.load` 和 `tl.store` 是 Triton 中最基本的内存操作原语。它们是从 GPU 全局内存（Global Memory / HBM）读写数据的唯一方式。理解它们的完整参数语义对编写高性能 kernel 至关重要。

### 3.2.1 tl.load — 从全局内存加载

`tl.load` 从 GPU 全局内存中读取数据到寄存器中。

```python
@triton.jit
def load_demo(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # ========== 最简用法 ==========
    # 只传入指针，假设数据连续且不越界
    data_simple = tl.load(X_ptr + offsets)

    # ========== 带 mask 的用法 ==========
    # mask=True 的位置正常加载，mask=False 的位置不加载（零开销）
    data_masked = tl.load(X_ptr + offsets, mask=mask)

    # ========== 带 mask 和 other 的用法 ==========
    # mask=False 的位置用 other 值填充
    data_with_default = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # other 也可以是特殊值
    data_neg_inf = tl.load(X_ptr + offsets, mask=mask, other=-float('inf'))  # Softmax 中常用

    tl.store(Y_ptr + offsets, data_with_default, mask=mask)
```

**`tl.load` 完整参数详解**：

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `pointer` | 张量指针 | — | 要加载的内存地址（可以是标量或张量） |
| `mask` | `bool` 张量 | `None` | 掩码张量，shape 与 pointer 一致 |
| `other` | 标量或张量 | `None` | mask=False 位置的填充值 |
| `boundary_check` | `tuple[int, ...]` | `()` | 需要边界检查的维度列表 |
| `padding_option` | `str` | `"zero"` | 边界外的填充方式: `"zero"` 或 `"nan"` |
| `cache_modifier` | `str` | `""` | 缓存修饰符 |
| `eviction_policy` | `str` | `""` | 驱逐策略 |
| `volatile` | `bool` | `False` | 是否标记为 volatile（禁用缓存优化） |

### 3.2.2 tl.load 的 mask 与 other 机制

`mask` 是 Triton 处理边界条件的核心机制。它避免了显式的 `if` 分支判断，让编译器能生成更高效的向量化加载指令。

```python
@triton.jit
def masked_load_demo(
    X_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    # 创建 mask：True 表示有效索引，False 表示越界
    mask = offsets < N

    # ---- 用法 1：mask=True 加载，mask=False 不加载（保持未初始化） ----
    # 危险！mask=False 位置的值是不确定的（garbage）
    # x_unsafe = tl.load(X_ptr + offsets, mask=mask)

    # ---- 用法 2：mask=False 填充 0.0 ----
    # 安全！越界位置被填充为 0.0
    x_safe = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # ---- 用法 3：mask=False 填充 -inf（Softmax 场景） ----
    # tl.exp(-inf) = 0，不会影响 softmax 的分母
    x_softmax = tl.load(X_ptr + offsets, mask=mask, other=-float('inf'))

    # ---- 用法 4：mask=False 填充 1.0（乘法场景） ----
    x_multiply = tl.load(X_ptr + offsets, mask=mask, other=1.0)
```

**mask 的底层行为**：

```
假设 BLOCK=8, N=5, offsets = [0, 1, 2, 3, 4, 5, 6, 7]

mask = [True, True, True, True, True, False, False, False]

tl.load(ptr + offsets, mask=mask, other=0.0):
  加载:    [X[0], X[1], X[2], X[3], X[4],  skip,  skip,  skip]
  结果:    [X[0], X[1], X[2], X[3], X[4],  0.0,   0.0,   0.0 ]
                              ↑ 有效加载      ↑ other 填充
```

**性能提示**：
- `other=0.0` 比 `other=-float('inf')` 更高效，因为 0 可以直接用清零指令
- 如果确信不会越界（如 N 总是 BLOCK 的倍数），可以省略 mask 来获得最佳性能
- `mask` 为 `None` 时等价于所有元素都有效，编译器不做边界检查

### 3.2.3 tl.load 的高级参数

**cache_modifier — 缓存修饰符**：

```python
@triton.jit
def cache_modifier_demo(X_ptr, Y_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)

    # 默认行为：经过 L1/L2 缓存
    data_default = tl.load(X_ptr + offsets)

    # "ca" (cache all): 所有层级都缓存（默认行为）
    data_ca = tl.load(X_ptr + offsets, cache_modifier="ca")

    # "cg" (cache global): 只缓存在 L2，跳过 L1
    # 适用于数据只使用一次的场景，避免污染 L1 缓存
    data_cg = tl.load(X_ptr + offsets, cache_modifier="cg")

    # "cs" (cache streaming): 流式访问模式提示
    data_cs = tl.load(X_ptr + offsets, cache_modifier="cs")

    # "wb" (write-back): 写回策略提示
    data_wb = tl.load(X_ptr + offsets, cache_modifier="wb")

    # "wt" (write-through): 写穿策略提示
    data_wt = tl.load(X_ptr + offsets, cache_modifier="wt")

    tl.store(Y_ptr + offsets, data_cg)
```

**cache_modifier 选择指南**：

| 修饰符 | 含义 | 适用场景 | PTX 映射 |
|:---|:---|:---|:---|
| `""` (默认) | 经过所有缓存 | 通用场景 | `ld.global` |
| `"ca"` | Cache All | 数据会被多次使用 | `ld.global.ca` |
| `"cg"` | Cache Global | 数据只用一次，避免 L1 污染 | `ld.global.cg` |
| `"cs"` | Cache Streaming | 大量顺序读取 | `ld.global.cs` |

**eviction_policy — 驱逐策略**：

```python
@triton.jit
def eviction_demo(X_ptr, Y_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)

    # "evict_normal": 正常替换策略（默认）
    data_normal = tl.load(X_ptr + offsets, eviction_policy="evict_normal")

    # "evict_first": 优先驱逐（数据不太可能再被访问）
    # 适用于一次性读取的数据
    data_first = tl.load(X_ptr + offsets, eviction_policy="evict_first")

    # "evict_last": 最后驱逐（数据很可能再被访问）
    # 适用于会被反复使用的数据（如权重矩阵）
    data_last = tl.load(X_ptr + offsets, eviction_policy="evict_last")

    tl.store(Y_ptr + offsets, data_first)
```

**eviction_policy 选择指南**：

| 策略 | 含义 | 适用场景 |
|:---|:---|:---|
| `"evict_normal"` | 默认 | 通用场景 |
| `"evict_first"` | 优先驱逐 | 流式数据、只读一次的中间结果 |
| `"evict_last"` | 最后驱逐 | 权重矩阵、需要复用的数据 |

<div data-component="LoadParameterExplorer"></div>

[组件：LoadParameterExplorer - 交互式探索 tl.load 各参数对生成 PTX 的影响]

### 3.2.4 tl.store — 存储到全局内存

`tl.store` 将寄存器中的数据写回 GPU 全局内存。

```python
@triton.jit
def store_demo(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # ========== 最简用法 ==========
    tl.store(Y_ptr + offsets, x)

    # ========== 带 mask 的用法 ==========
    # mask=False 的位置不写入（避免越界写入）
    tl.store(Y_ptr + offsets, x * 2.0, mask=mask)

    # ========== 带 boundary_check 的用法 ==========
    # boundary_check 指定需要边界检查的维度
    # 这是另一种边界保护方式，常与 block pointer 配合使用
    tl.store(Y_ptr + offsets, x * 2.0, boundary_check=(0,))
```

**`tl.store` 完整参数详解**：

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `pointer` | 张量指针 | — | 目标内存地址 |
| `value` | 张量 | — | 要存储的值 |
| `mask` | `bool` 张量 | `None` | 掩码，False 位置不写入 |
| `boundary_check` | `tuple[int, ...]` | `()` | 边界检查维度 |
| `cache_modifier` | `str` | `""` | 缓存修饰符 |
| `eviction_policy` | `str` | `""` | 驱逐策略 |

**boundary_check vs mask 的区别**：

```python
@triton.jit
def boundary_vs_mask(
    X_ptr, Y_ptr, M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 方式 1：使用 mask（推荐用于简单场景）
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]  # 二维 mask
    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], x, mask=mask)

    # 方式 2：使用 boundary_check（推荐与 block pointer 配合）
    # boundary_check 让编译器自动生成边界检查代码
    # padding_option 指定越界位置的填充值
    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :],
             x, boundary_check=(0, 1))
```

### 3.2.5 内存合并访问模式

**内存合并 (Memory Coalescing)** 是 GPU 性能的关键因素。当一个 warp 中的 32 个线程同时访问全局内存时，如果它们访问的地址是**连续的**，GPU 的内存控制器会将这些访问合并为一次或少数几次内存事务。

```
合并访问 (Coalesced Access) - 高效：

线程 0 访问地址 0x0000
线程 1 访问地址 0x0004    →  合并为一次 128 字节的内存事务
线程 2 访问地址 0x0008       （一个 cache line）
...
线程 31 访问地址 0x007C

带宽利用率: ~100%

非合并访问 (Uncoalesced Access) - 低效：

线程 0 访问地址 0x0000
线程 1 访问地址 0x1000    →  无法合并，需要 32 次独立的内存事务
线程 2 访问地址 0x2000
...
线程 31 访问地址 0x1F000

带宽利用率: ~3%（每次事务只用了 4 字节 / 32 字节 cache line）
```

**Triton 中的内存合并**：

```python
@triton.jit
def coalesced_access_demo(
    X_ptr, Y_ptr, M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # ---- 合并访问：同一 warp 内的线程访问连续地址 ----
    # 当 BLOCK_N 是 warp 大小的倍数时，
    # 同一 warp 内的线程会访问连续的列偏移
    # → 编译器生成合并的内存事务
    coalesced_offsets = pid_n * BLOCK_N + offs_n  # 连续的列偏移
    data_coalesced = tl.load(X_ptr + pid_m * N + coalesced_offsets)

    # ---- 跨步访问（不合并）----
    # 如果按行优先存储，但按列索引加载 → 跨步访问
    # stride = N，可能很大 → 不连续
    strided_offsets = pid_n * BLOCK_N + offs_n
    data_strided = tl.load(X_ptr + offs_m * N + strided_offsets)
    # ↑ 这里 offs_m 是线性增长的，但实际访问地址间隔 N 个 float
    # 当 N 很大时，这些访问在物理内存中不连续
```

**行优先 vs 列优先加载的性能差异**：

```python
@triton.jit
def row_vs_col_access(
    X_ptr, M, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)

    # 行访问：连续地址 → 合并（快）
    # 假设 X 按行优先存储
    row_data = tl.load(X_ptr + pid * N + offsets)  # 连续访问 X[pid*N .. pid*N+BLOCK-1]

    # 列访问：跨步地址 → 不合并（慢）
    col_data = tl.load(X_ptr + offsets * N + pid)  # 跨步访问 X[pid], X[N+pid], X[2*N+pid], ...
```

**性能提示**：

| 访问模式 | 带宽利用率 | 建议 |
|:---|:---|:---|
| 连续访问（stride=1） | ~100% | 最优，优先使用 |
| 小跨步访问（stride≤4） | 75-100% | 可接受 |
| 大跨步访问（stride>32） | <10% | 避免，考虑转置 |
| 随机访问 | 极低 | 使用共享内存重组 |

---

## 3.3 算术运算

Triton 支持丰富的算术运算，所有运算都是**逐元素 (element-wise)** 的，即对张量中的每个元素独立执行相同的操作。

### 3.3.1 逐元素基本算术运算

```python
@triton.jit
def arithmetic_demo(
    A_ptr, B_ptr, C_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)

    # ========== 基本四则运算 ==========
    add_result = a + b       # 逐元素加法
    sub_result = a - b       # 逐元素减法
    mul_result = a * b       # 逐元素乘法
    div_result = a / b       # 逐元素除法（浮点）
    mod_result = a % b       # 逐元素取模（整数）

    # ========== 复合赋值运算 ==========
    a += b    # 等价于 a = a + b
    a -= b    # 等价于 a = a - b
    a *= b    # 等价于 a = a * b
    a /= b    # 等价于 a = a / b

    # ========== 标量与张量的运算 ==========
    scaled = a * 2.0         # 标量广播到每个元素
    shifted = a + 1.0        # 标量加法
    reciprocal = 1.0 / a     # 标量除以张量

    # ========== 复合运算 ==========
    fused = a * b + c        # 乘加（FMA）
    polynomial = a * a + 2.0 * a + 1.0  # 多项式

    tl.store(C_ptr + offsets, fused, mask=mask)
```

**支持的算术运算符**：

| 运算符 | 含义 | 支持类型 | 注意事项 |
|:---|:---|:---|:---|
| `+` | 加法 | 整数/浮点 | 整数溢出是未定义行为 |
| `-` | 减法 | 整数/浮点 | — |
| `*` | 乘法 | 整数/浮点 | 整数乘法可能溢出 |
| `/` | 除法 | 整数/浮点 | 整数除法向零截断；浮点除以 0 为 inf |
| `%` | 取模 | 整数/浮点 | 浮点取模使用 `fmod` 语义 |
| `//` | 整除 | 整数/浮点 | 向负无穷截断 |
| `**` | 幂 | 浮点 | 底数和指数都支持张量 |
| `@` | 矩阵乘法 | 浮点 | 等价于 `tl.dot`，不推荐直接使用 |

### 3.3.2 比较运算

```python
@triton.jit
def comparison_demo(
    A_ptr, B_ptr, C_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)

    # 比较运算返回 bool 张量
    eq_result  = (a == b)    # 逐元素等于
    ne_result  = (a != b)    # 逐元素不等于
    lt_result  = (a <  b)    # 逐元素小于
    le_result  = (a <= b)    # 逐元素小于等于
    gt_result  = (a >  b)    # 逐元素大于
    ge_result  = (a >= b)    # 逐元素大于等于

    # 比较结果常用于 tl.where 条件选择
    # 选取 a 和 b 中的较大值
    max_val = tl.where(a > b, a, b)

    # ReLU 激活函数
    relu = tl.where(a > 0, a, 0.0)

    # 带符号的截断
    clipped = tl.where(a > 1.0, 1.0, tl.where(a < -1.0, -1.0, a))  # clip to [-1, 1]

    tl.store(C_ptr + offsets, max_val, mask=mask)
```

**`tl.where` 条件选择**：

```python
# tl.where(condition, true_value, false_value)
# condition: bool 张量
# true_value: condition=True 时的值
# false_value: condition=False 时的值
# 三个参数的 shape 需要满足广播规则

@triton.jit
def where_demo(X, Y, N, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x = tl.load(X + offsets)

    # ReLU
    y = tl.where(x > 0, x, 0.0)

    # Leaky ReLU
    y_leaky = tl.where(x > 0, x, 0.01 * x)

    # Hard sigmoid: clamp((x+3)/6, 0, 1)
    y_hard_sig = tl.where(x >= 3.0, 1.0, tl.where(x <= -3.0, 0.0, (x + 3.0) / 6.0))
```

### 3.3.3 逻辑运算

```python
@triton.jit
def logical_demo(
    A_ptr, B_ptr, C_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)

    # 逻辑运算符（对 bool 张量操作）
    a_pos = a > 0
    b_pos = b > 0

    and_result = a_pos & b_pos    # 逻辑与（按位与，因为 bool 用 0/1 表示）
    or_result  = a_pos | b_pos    # 逻辑或
    xor_result = a_pos ^ b_pos    # 逻辑异或
    not_result = ~a_pos           # 逻辑非（按位取反）

    # 常见用法：组合多个条件
    both_positive = (a > 0) & (b > 0)
    either_large = (a > 10) | (b > 10)
    in_range = (a >= 0) & (a <= 1)

    tl.store(C_ptr + offsets, tl.where(both_positive, a + b, 0.0), mask=mask)
```

### 3.3.4 一元数学函数

Triton 提供了丰富的内置数学函数，它们会被编译为高效的 GPU 指令。

```python
@triton.jit
def math_functions_demo(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # ========== 指数和对数 ==========
    exp_x = tl.exp(x)           # e^x，编译为 EX2 或 MUFU.EXP 指令
    exp2_x = tl.exp2(x)         # 2^x，比 tl.exp 更快（硬件原生支持）
    log_x = tl.log(x)           # ln(x)
    log2_x = tl.log2(x)         # log2(x)，比 tl.log 更快

    # ========== 幂和根 ==========
    sqrt_x = tl.sqrt(x)         # √x
    abs_x = tl.abs(x)           # |x|
    rsqrt_x = tl.math.rsqrt(x)  # 1/√x（比 1/tl.sqrt(x) 更精确更快）

    # ========== 取整 ==========
    floor_x = tl.math.floor(x)  # 向下取整
    ceil_x = tl.math.ceil(x)    # 向上取整

    # ========== 三角函数 ==========
    sin_x = tl.math.sin(x)
    cos_x = tl.math.cos(x)
    tanh_x = tl.math.tanh(x)

    # ========== 常用复合函数 ==========
    # Sigmoid: 1 / (1 + exp(-x))
    # 注意：直接计算 1/(1+exp(-x)) 在 x 很大时 exp(-x) 会下溢
    sigmoid_x = tl.sigmoid(x)   # 内置实现，数值稳定

    # GELU 近似: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    gelu_x = 0.5 * x * (1.0 + tl.math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

    # Swish / SiLU: x * sigmoid(x)
    swish_x = x * tl.sigmoid(x)

    # Softplus: log(1 + exp(x))
    # 注意：直接计算 exp(x) 可能溢出，使用 tl.where 保护
    softplus_x = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))

    tl.store(Y_ptr + offsets, sigmoid_x, mask=mask)
```

**数学函数性能对比**：

| 函数 | 硬件指令 | 相对速度 | 备注 |
|:---|:---|:---|:---|
| `tl.exp(x)` | EX2 + MUL | 快 | 底数为 e |
| `tl.exp2(x)` | EX2 | 最快 | 底数为 2，硬件原生 |
| `tl.log(x)` | LG2 + MUL | 快 | — |
| `tl.log2(x)` | LG2 | 最快 | 硬件原生 |
| `tl.sqrt(x)` | MUFU.SQRT | 快 | — |
| `tl.rsqrt(x)` | MUFU.RSQRT | 快 | 1/√x，比 sqrt 更推荐 |
| `tl.sigmoid(x)` | 组合指令 | 中等 | 数值稳定实现 |
| `tl.math.tanh(x)` | 组合指令 | 中等 | — |
| `tl.abs(x)` | AND | 极快 | 单指令 |

**性能技巧：用 `exp2` 代替 `exp`**：

```python
@triton.jit
def fast_exp(X, Y, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x = tl.load(X + offsets)

    # 方式 1：直接使用 tl.exp（内部会做 exp2 转换，有精度损失）
    y1 = tl.exp(x)

    # 方式 2：手动转换为 exp2（可控制精度，有时更快）
    # e^x = 2^(x * log2(e))
    LOG2_E = 1.4426950408889634  # log2(e)
    y2 = tl.exp2(x * LOG2_E)

    # 两者在大多数场景下等价
    tl.store(Y + offsets, y2)
```

### 3.3.5 类型提升规则

当两个不同数据类型的张量参与运算时，Triton 会按照**类型提升 (Type Promotion)** 规则自动将它们转换为相同的类型。

```python
@triton.jit
def type_promotion_demo(
    X_f16_ptr, X_f32_ptr, Y_ptr, BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)

    x_f16 = tl.load(X_f16_ptr + offsets)  # float16 张量
    x_f32 = tl.load(X_f32_ptr + offsets)  # float32 张量

    # 类型提升规则：低精度 → 高精度
    # float16 + float32 → float32（结果是 float32）
    result = x_f16 + x_f32

    # int32 + float32 → float32
    # 但 Triton 中 int + float 的行为取决于具体版本
    # 建议显式转换以避免歧义

    # 显式转换（推荐）
    x_f16_to_f32 = x_f16.to(tl.float32)
    result_explicit = x_f16_to_f32 + x_f32

    tl.store(Y_ptr + offsets, result)
```

**类型提升规则表**：

| 操作数类型 A | 操作数类型 B | 结果类型 | 说明 |
|:---|:---|:---|:---|
| `float16` | `float16` | `float16` | 同类型不提升 |
| `float16` | `float32` | `float32` | 低 → 高 |
| `bfloat16` | `float32` | `float32` | 低 → 高 |
| `float32` | `float32` | `float32` | 同类型不提升 |
| `int32` | `int32` | `int32` | 整数同类型 |
| `int16` | `int32` | `int32` | 低 → 高 |
| `int32` | `float32` | `float32` | 整数 → 浮点 |
| `float16` | `bfloat16` | `float32` | 两种 16-bit → 32-bit |

<div data-component="TypePromotionChart"></div>

[组件：TypePromotionChart - 交互式类型提升规则图]

---

## 3.4 归约操作

归约 (Reduction) 是将一个张量沿某个维度"压缩"为更小张量的操作。典型的归约包括求和、求最大值、求最小值。在深度学习中，归约操作无处不在：softmax 的分母、layer norm 的均值和方差、注意力机制的权重归一化。

### 3.4.1 tl.sum — 沿轴求和

```python
@triton.jit
def sum_demo(
    X_ptr, Y_ptr, M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # 加载一个 BLOCK_M × BLOCK_N 的 tile
    x_ptrs = X_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # ========== 沿 axis=1 归约（沿列方向求和） ==========
    # 输入 shape: (BLOCK_M, BLOCK_N) → 输出 shape: (BLOCK_M,)
    # 每一行被压缩为一个标量
    row_sum = tl.sum(x, axis=1)

    # ========== 沿 axis=0 归约（沿行方向求和） ==========
    # 输入 shape: (BLOCK_M, BLOCK_N) → 输出 shape: (BLOCK_N,)
    # 每一列被压缩为一个标量
    col_sum = tl.sum(x, axis=0)

    # ========== 全局归约 ==========
    # 先沿 axis=1 归约，再沿 axis=0 归约
    # 或者直接使用 keep_dims 保持维度
    total_sum = tl.sum(x, axis=1)  # (BLOCK_M,)
    total_sum = tl.sum(total_sum, axis=0)  # 标量

    # ========== keep_dims 选项 ==========
    # 保持被归约的维度（大小为 1）
    row_sum_keep = tl.sum(x, axis=1, keep_dims=True)
    # 输出 shape: (BLOCK_M, 1) 而不是 (BLOCK_M,)
    # 这在广播运算中很有用
```

**归约的维度语义**：

```
原始张量 x: shape = (4, 8)

x = [[ 1,  2,  3,  4,  5,  6,  7,  8],   ← 行 0
     [ 9, 10, 11, 12, 13, 14, 15, 16],   ← 行 1
     [17, 18, 19, 20, 21, 22, 23, 24],   ← 行 2
     [25, 26, 27, 28, 29, 30, 31, 32]]   ← 行 3

tl.sum(x, axis=1) → 沿列方向压缩（每行求和）
结果: [36, 100, 164, 228]  shape: (4,)

tl.sum(x, axis=0) → 沿行方向压缩（每列求和）
结果: [52, 56, 60, 64, 68, 72, 76, 80]  shape: (8,)

tl.sum(x, axis=1, keep_dims=True) → 保持维度
结果: [[36], [100], [164], [228]]  shape: (4, 1)
```

### 3.4.2 tl.min 与 tl.max

```python
@triton.jit
def min_max_demo(
    X_ptr, Min_ptr, Max_ptr, M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    x_ptrs = X_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask, other=float('inf'))  # 注意 other 的选择

    # 沿 axis=1 求每行的最大值
    # 对于 max：other 应设为 -float('inf')，这样越界元素不会影响最大值
    # 但我们上面用了 inf，所以这里需要重新加载
    x_for_max = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    row_max = tl.max(x_for_max, axis=1)  # shape: (BLOCK_M,)

    # 沿 axis=1 求每行的最小值
    # 对于 min：other 应设为 float('inf')，这样越界元素不会影响最小值
    row_min = tl.min(x, axis=1)  # shape: (BLOCK_M,)

    # keep_dims 保持维度
    row_max_keep = tl.max(x_for_max, axis=1, keep_dims=True)  # shape: (BLOCK_M, 1)

    tl.store(Min_ptr + offs_m, row_min)
    tl.store(Max_ptr + offs_m, row_max)
```

**归约操作的 `other` 值选择**：

| 归约操作 | `other` 推荐值 | 原因 |
|:---|:---|:---|
| `tl.sum` | `0.0` | 0 不影响求和 |
| `tl.max` | `-float('inf')` | -∞ 不会被选为最大值 |
| `tl.min` | `float('inf')` | +∞ 不会被选为最小值 |

### 3.4.3 tl.reduce — 自定义归约

`tl.reduce` 允许用户定义自定义的归约操作。它接受一个二元函数作为归约的核心操作。

```python
@triton.jit
def custom_reduce_demo(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # ========== 使用 tl.reduce 自定义归约 ==========

    # 自定义归约函数：两个元素取最大值
    # 参数 a, b: 当前归约的两个操作数
    # 返回值: 归约结果
    def max_fn(a, b):
        return tl.where(a > b, a, b)

    # 使用自定义归约函数
    row_max = tl.reduce(x, axis=0, combine_fn=max_fn)

    # 自定义归约函数：两个元素取最小值
    def min_fn(a, b):
        return tl.where(a < b, a, b)

    row_min = tl.reduce(x, axis=0, combine_fn=min_fn)

    # 自定义归约：L1 范数（绝对值之和）
    def l1_add(a, b):
        return tl.abs(a) + tl.abs(b)

    # 注意：这个例子需要更复杂的处理，这里只是展示语法
    # 实际上 L1 范数应该先取绝对值再求和

    # 自定义归约：逻辑与（检查所有元素是否都大于 0）
    def logical_and_fn(a, b):
        return (a > 0) & (b > 0)

    all_positive = tl.reduce(x > 0, axis=0, combine_fn=logical_and_fn)

    tl.store(Y_ptr + pid, row_max)
```

**`tl.reduce` 参数详解**：

| 参数 | 类型 | 说明 |
|:---|:---|:---|
| `input` | 张量 | 输入张量 |
| `axis` | `int` | 归约的轴 |
| `combine_fn` | callable | 二元归约函数，接受两个相同类型的参数，返回相同类型 |
| `keep_dims` | `bool` | 是否保持归约维度 |

**自定义归约函数的约束**：

```python
# combine_fn 必须满足以下条件：
# 1. 接受两个相同类型的参数
# 2. 返回与参数相同类型的值
# 3. 必须是结合律的 (associative)：f(a, f(b, c)) == f(f(a, b), c)
# 4. 不需要满足交换律，但满足交换律的函数更高效

# 合法的 combine_fn 示例：
def add_fn(a, b): return a + b            # 加法：结合律 ✓
def max_fn(a, b): return tl.where(a > b, a, b)  # 最大值：结合律 ✓
def mul_fn(a, b): return a * b            # 乘法：结合律 ✓

# 不合法的 combine_fn 示例（不满足结合律）：
# def bad_fn(a, b): return a - b          # 减法：不满足结合律 ✗
```

### 3.4.4 归约的并行实现原理（Warp Shuffle Reduction）

理解归约的底层实现有助于编写更高效的 kernel。在 GPU 上，归约操作通常分两个阶段完成：

**阶段 1：Warp 内归约（使用 Warp Shuffle）**

```
一个 Warp 有 32 个线程，每个线程持有一个值：

线程:    T0  T1  T2  T3  T4  T5  T6  T7  ... T31
值:      v0  v1  v2  v3  v4  v5  v6  v7  ... v31

Step 1: __shfl_xor_sync(offset=16)
  T0 += T16, T1 += T17, ..., T15 += T31
  前 16 个线程各持有 2 个元素的部分和

Step 2: __shfl_xor_sync(offset=8)
  T0 += T8, T1 += T9, ..., T7 += T15
  前 8 个线程各持有 4 个元素的部分和

Step 3: __shfl_xor_sync(offset=4)
  T0 += T4, T1 += T5, T2 += T6, T3 += T7
  前 4 个线程各持有 8 个元素的部分和

Step 4: __shfl_xor_sync(offset=2)
  T0 += T2, T1 += T3
  T0 和 T1 各持有 16 个元素的部分和

Step 5: __shfl_xor_sync(offset=1)
  T0 += T1
  T0 持有全部 32 个元素的总和

总步数: log2(32) = 5 步（非常高效！）
```

**阶段 2：Warp 间归约（使用共享内存）**

```
假设一个 Block 有 4 个 Warp：

Warp 0: T0 持有部分和 S0
Warp 1: T0 持有部分和 S1
Warp 2: T0 持有部分和 S2
Warp 3: T0 持有部分和 S3

Step 1: 各 Warp 的 T0 将部分和写入共享内存
  shared[0] = S0
  shared[1] = S1
  shared[2] = S2
  shared[3] = S3

Step 2: __syncthreads()

Step 3: 由一个 Warp 对共享内存中的值做最终归约
  T0 = shared[0] + shared[1] + shared[2] + shared[3]
```

**Triton 中的归约性能特点**：

```python
@triton.jit
def reduction_performance_demo(
    X_ptr, Y_ptr, M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    x_ptrs = X_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # tl.sum 内部自动实现 Warp Shuffle + 共享内存归约
    # 用户不需要关心底层细节
    # 编译器会根据 Tile 大小选择最优的归约策略
    row_sum = tl.sum(x, axis=1)

    # 性能提示：
    # 1. 归约的 axis 越小（元素越少），越可能只用 Warp Shuffle
    # 2. 归约的 axis 越大（元素越多），需要 Warp Shuffle + 共享内存
    # 3. BLOCK_N = 128 通常是一个好的选择（4 个 Warp）

    tl.store(Y_ptr + offs_m, row_sum)
```

<div data-component="WarpShuffleReduction"></div>

[组件：WarpShuffleReduction - 动画演示 Warp Shuffle 归约的执行过程]

---

## 3.5 矩阵乘法

`tl.dot` 是 Triton 中最重要的计算原语。它执行矩阵乘累加 (Matrix Multiply-Accumulate, MMA) 操作，并且在支持 Tensor Core 的 GPU 上会自动映射到硬件 Tensor Core 指令。

### 3.5.1 tl.dot 基本用法

```python
@triton.jit
def dot_basic_demo(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 创建索引
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 加载 A 和 B 的 tile
    # A: shape (BLOCK_M, BLOCK_K)
    a_ptrs = A_ptr + offs_m[:, None] * K + offs_k[None, :]
    a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)

    # B: shape (BLOCK_K, BLOCK_N)
    b_ptrs = B_ptr + offs_k[:, None] * N + offs_n[None, :]
    b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)

    # ========== 核心操作：矩阵乘法 ==========
    # tl.dot(a, b) 执行 a @ b，即矩阵乘法
    # a: shape (M, K), b: shape (K, N) → 结果: shape (M, N)
    c = tl.dot(a, b)  # shape: (BLOCK_M, BLOCK_N)

    # 存储结果
    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

**`tl.dot` 的核心特性**：

| 特性 | 说明 |
|:---|:---|
| **乘累加语义** | `tl.dot(a, b)` 执行 `a @ b`，不做累加 |
| **累加器模式** | `accumulator += tl.dot(a, b)` 实现乘累加 |
| **Tensor Core 映射** | 在 Ampere+ GPU 上自动映射到 HMMA/MMA 指令 |
| **精度要求** | 累加器通常使用 `float32` 以保持精度 |

### 3.5.2 tl.dot 的完整参数

```python
@triton.jit
def dot_full_params_demo(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B_ptr + offs_k[:, None] * N + offs_n[None, :]

    # 初始化累加器（float32 精度）
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)

        # ========== 带完整参数的 tl.dot ==========
        c = tl.dot(
            a,                          # 输入 A: shape (BLOCK_M, BLOCK_K)
            b,                          # 输入 B: shape (BLOCK_K, BLOCK_N)
            allow_tf32=True,            # 是否允许使用 TF32 精度（Ampere+ GPU）
            out_dtype=tl.float32,       # 输出数据类型（累加器精度）
        )

        # 等价的数学运算：
        # c[i, j] = sum_k(a[i, k] * b[k, j]) for all k
        # 在底层，编译器会使用 Tensor Core 指令（如 HMMA.1688.F32.F16）

        accumulator += c

        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    # 转换回目标精度并存储
    c = accumulator.to(tl.float16)
    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, c)
```

**`tl.dot` 参数详解**：

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `input` | 张量 | — | 左矩阵 A，shape: (M, K) |
| `other` | 张量 | — | 右矩阵 B，shape: (K, N) |
| `allow_tf32` | `bool` | `True` | 是否允许使用 TF32 精度（Ampere+） |
| `out_dtype` | `tl.dtype` | `None` | 输出 dtype，None 时与输入相同 |

### 3.5.3 Tensor Core 映射

Tensor Core 是 NVIDIA GPU（Volta 架构及以后）中的专用矩阵运算单元。一个 Tensor Core 每周期可以执行一个小型矩阵乘累加操作，吞吐量远超普通的 CUDA Core。

**Tensor Core 的硬件规格**：

| GPU 架构 | Tensor Core 指令 | 单步矩阵大小 | 支持精度 |
|:---|:---|:---|:---|
| Volta (SM 7.0) | HMMA.884 | 8×8×4 | FP16 |
| Turing (SM 7.5) | HMMA.884 | 8×8×4 | FP16, INT8 |
| Ampere (SM 8.0) | HMMA.1688 / HMMA.16816 | 16×8×8 / 16×8×16 | FP16, BF16, TF32, INT8 |
| Hopper (SM 9.0) | HMMA.16832 | 16×8×32 | FP16, BF16, FP8, TF32 |
| Blackwell (SM 10.0) | — | — | FP4, FP6, FP8 |

**Triton 如何映射到 Tensor Core**：

```
Triton 代码:
  accumulator += tl.dot(a_tile, b_tile)
  其中 a_tile: shape (128, 32), b_tile: shape (32, 128)

编译器内部:
  1. 将 128×32 × 32×128 的大矩阵乘法
     分解为多个 16×8×K 的 Tensor Core 指令
  2. 每个 Tensor Core 指令处理一个小子块
  3. 使用 HMMA.16816.F32.F16 指令（Ampere 架构）

生成的 PTX:
  mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {f32_0, f32_1, f32_2, f32_3},
    {f16_0, f16_1, f16_2, f16_3},
    {f16_4, f16_5, f16_6, f16_7},
    {f32_0, f32_1, f32_2, f32_3};
```

**`tl.dot` 的 Tile 大小约束**：

```python
# 正确：Tile 大小是 Tensor Core 指令要求的倍数
a = tl.zeros((128, 16), dtype=tl.float16)  # 16 是 k 维度的最小对齐要求
b = tl.zeros((16, 128), dtype=tl.float16)
c = tl.dot(a, b)  # OK

# 正确：其他合法大小
a = tl.zeros((64, 32), dtype=tl.float16)
b = tl.zeros((32, 64), dtype=tl.float16)
c = tl.dot(a, b)  # OK

# 错误：Tile 大小不满足对齐要求
# a = tl.zeros((7, 5), dtype=tl.float16)  # 不满足 16×8 的对齐约束
# b = tl.zeros((5, 7), dtype=tl.float16)
# c = tl.dot(a, b)  # 可能编译错误或性能极差
```

### 3.5.4 支持的 dtype 组合

`tl.dot` 对输入和输出的 dtype 组合有严格的要求，这些限制源于 Tensor Core 硬件支持的指令格式。

```python
@triton.jit
def dot_dtype_demo(A, B, C, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # ========== 组合 1: FP16 × FP16 → FP16/FP32 ==========
    a_f16 = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float16)
    b_f16 = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float16)

    # 输出 FP32（推荐：累加精度更高）
    c_f32 = tl.dot(a_f16, b_f16, out_dtype=tl.float32)

    # 输出 FP16（精度损失，但节省内存）
    c_f16 = tl.dot(a_f16, b_f16, out_dtype=tl.float16)

    # ========== 组合 2: BF16 × BF16 → FP32 ==========
    a_bf16 = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.bfloat16)
    b_bf16 = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.bfloat16)
    c_from_bf16 = tl.dot(a_bf16, b_bf16, out_dtype=tl.float32)

    # ========== 组合 3: TF32 × TF32 → FP32 ==========
    # TF32 是 Ampere+ 引入的格式：1 位符号 + 8 位指数 + 10 位尾数
    # 精度介于 FP16 和 FP32 之间
    a_tf32 = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)  # TF32 使用 float32 容器
    b_tf32 = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    c_from_tf32 = tl.dot(a_tf32, b_tf32, allow_tf32=True, out_dtype=tl.float32)

    # ========== 组合 4: INT8 × INT8 → INT32 ==========
    # 用于量化推理
    a_int8 = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.int8)
    b_int8 = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.int8)
    c_int32 = tl.dot(a_int8, b_int8, out_dtype=tl.int32)

    tl.store(C, c_f32)
```

**支持的 dtype 组合汇总**：

| 输入 A dtype | 输入 B dtype | 输出 dtype | GPU 要求 | Tensor Core 指令 |
|:---|:---|:---|:---|:---|
| `float16` | `float16` | `float16` | Volta+ | HMMA |
| `float16` | `float16` | `float32` | Volta+ | HMMA (推荐) |
| `bfloat16` | `bfloat16` | `float32` | Ampere+ | HMMA |
| `float32` (TF32) | `float32` (TF32) | `float32` | Ampere+ | HMMA (TF32 模式) |
| `int8` | `int8` | `int32` | Turing+ | IMMA |
| `float8e4m3fn` | `float8e4m3fn` | `float32` | Hopper+ | DMMA (FP8) |
| `float8e5m2` | `float8e5m2` | `float32` | Hopper+ | DMMA (FP8) |

**allow_tf32 参数详解**：

```python
@triton.jit
def tf32_demo(A, B, C, BLOCK: tl.constexpr):
    a = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    b = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)

    # allow_tf32=True（默认）：将 FP32 截断为 TF32 格式
    # 精度：尾数从 23 位截断到 10 位
    # 性能：可以使用 Tensor Core，速度快 4-8 倍
    c_tf32 = tl.dot(a, b, allow_tf32=True)

    # allow_tf32=False：使用完整的 FP32 精度
    # 不能使用 Tensor Core（除非使用软件模拟）
    # 性能：慢很多
    c_fp32 = tl.dot(a, b, allow_tf32=False)

    # TF32 的精度损失：
    # FP32: 1.123456789012345 → 精确
    # TF32: 1.123456789012345 → 1.12353515625（尾数截断到 10 位）
    # 相对误差: ~2^-10 ≈ 0.1%
```

### 3.5.5 矩阵乘法的完整实现

```python
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # ---- L2 Cache 优化：使用 swizzle 模式分配 program ----
    # 将相邻的 program 分配到访问相同 B tile 的位置
    # 这样 B tile 可以被多个 program 复用，提高 L2 缓存命中率
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- 创建索引 ----
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # ---- 初始化累加器（float32 精度） ----
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ---- 沿 K 维度迭代 ----
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # 加载 A tile: (BLOCK_M, BLOCK_K)
        a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        k_remaining = K - k * BLOCK_K
        a_mask = offs_k[None, :] < k_remaining
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # 加载 B tile: (BLOCK_K, BLOCK_N)
        b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = offs_k[:, None] < k_remaining
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # 矩阵乘累加（编译器自动映射到 Tensor Core 指令）
        accumulator += tl.dot(a, b)

        # 推进 K 维度指针
        offs_k += BLOCK_K

    # ---- 类型转换并存储结果 ----
    c = accumulator.to(C.dtype.element_ty)
    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
```

<div data-component="MatmulTilingVisualization"></div>

[组件：MatmulTilingVisualization - 动画展示矩阵乘法的分块计算过程与 Tensor Core 映射]

---

## 3.6 类型转换

### 3.6.1 .to(dtype) 显式类型转换

`.to()` 方法是 Triton 中进行显式类型转换的主要方式。它将张量的每个元素从当前类型转换为目标类型。

```python
@triton.jit
def type_cast_demo(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # 加载 FP32 数据
    x_f32 = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # ========== FP32 → FP16 ==========
    # 截断尾数精度（23 位 → 10 位），范围也缩小
    x_f16 = x_f32.to(tl.float16)

    # ========== FP32 → BF16 ==========
    # 截断尾数精度（23 位 → 7 位），但保持 FP32 的指数范围
    x_bf16 = x_f32.to(tl.bfloat16)

    # ========== FP16 → FP32 ==========
    # 扩展精度（无损转换）
    x_f32_again = x_f16.to(tl.float32)

    # ========== FP32 → INT32 ==========
    # 截断为整数（向零截断）
    x_int = x_f32.to(tl.int32)

    # ========== INT32 → FP32 ==========
    # 整数转浮点（精确，只要不超过 FP32 的尾数精度）
    x_float = x_int.to(tl.float32)

    # ========== 带舍入模式的转换（Hopper+ GPU） ==========
    # 某些 GPU 支持指定舍入模式
    # x_f16_rn = x_f32.to(tl.float16, fp_rounding="rtne")  # Round to Nearest Even
    # x_f16_rz = x_f32.to(tl.float16, fp_rounding="rtz")   # Round to Zero

    tl.store(Y_ptr + offsets, x_f16.to(tl.float32), mask=mask)
```

**类型转换的精度损失**：

| 转换方向 | 精度影响 | 典型误差 | 建议 |
|:---|:---|:---|:---|
| `float32 → float16` | 尾数 23→10 位 | ~10^-3 | 权重和激活值 |
| `float32 → bfloat16` | 尾数 23→7 位 | ~10^-2 | 训练中的梯度 |
| `float32 → tf32` | 尾数 23→10 位 | ~10^-3 | Tensor Core 输入 |
| `float16 → float32` | 无损 | 0 | 恢复精度 |
| `float32 → int32` | 截断小数 | 整数部分 | 量化场景 |
| `int32 → float32` | 无损（如果 |n| < 2^24） | 0 | 索引计算 |

### 3.6.2 隐式类型提升

在算术运算中，不同类型的张量会自动进行类型提升。

```python
@triton.jit
def implicit_promotion_demo(
    A_f16_ptr, B_f32_ptr, C_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    a_f16 = tl.load(A_f16_ptr + offsets, mask=mask, other=0.0)  # float16
    b_f32 = tl.load(B_f32_ptr + offsets, mask=mask, other=0.0)  # float32

    # 隐式提升：float16 + float32 → float32
    # a_f16 被自动转换为 float32 后再相加
    c = a_f16 + b_f32  # 结果是 float32

    # 但需要注意：tl.dot 不支持隐式提升！
    # 两个输入必须是相同的 dtype
    # tl.dot(a_f16, b_f32)  # 错误！

    # 正确做法：显式转换
    a_f32 = a_f16.to(tl.float32)
    # 或者
    b_f16 = b_f32.to(tl.float16)

    tl.store(C_ptr + offsets, c, mask=mask)
```

### 3.6.3 FP8 类型支持

FP8 是 Hopper 架构（H100）引入的新型低精度浮点格式，主要用于加速 LLM 推理中的矩阵乘法。

```python
@triton.jit
def fp8_demo(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # ========== FP8 E4M3 格式 ==========
    # 1 位符号 + 4 位指数 + 3 位尾数
    # 范围: ±448，最小正规数: 2^-9
    # 精度: ~1% 相对误差
    a_ptrs = A_ptr + offs_m[:, None] * K + offs_k[None, :]
    a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    a_fp8 = a.to(tl.float8e4m3fn)  # 转换为 FP8 E4M3

    # ========== FP8 E5M2 格式 ==========
    # 1 位符号 + 5 位指数 + 2 位尾数
    # 范围: ±57344，最小正规数: 2^-14
    # 精度: ~25% 相对误差（比 E4M3 粗糙）
    b_ptrs = B_ptr + offs_k[:, None] * N + offs_n[None, :]
    b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)
    b_fp8 = b.to(tl.float8e5m2)  # 转换为 FP8 E5M2

    # FP8 矩阵乘法（需要 Hopper+ GPU）
    # 注意：FP8 的 tl.dot 只在特定硬件上支持
    # accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # accumulator += tl.dot(a_fp8, b_fp8)  # 需要 SM 9.0+

    # 如果不支持 FP8，先转换回 FP16/BF16
    a_f16 = a_fp8.to(tl.float16)
    b_f16 = b_fp8.to(tl.float16)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    accumulator += tl.dot(a_f16, b_f16)

    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, accumulator)
```

**FP8 格式对比**：

| 格式 | 符号 | 指数 | 尾数 | 范围 | 精度 | 典型用途 |
|:---|:---|:---|:---|:---|:---|:---|
| `float8e4m3fn` | 1 | 4 | 3 | ±448 | ~1% | 权重 (Forward) |
| `float8e5m2` | 1 | 5 | 2 | ±57344 | ~25% | 梯度 (Backward) |
| `float16` | 1 | 5 | 10 | ±65504 | ~0.1% | 通用 |
| `bfloat16` | 1 | 8 | 7 | ±3.4e38 | ~1% | 训练 |
| `float32` | 1 | 8 | 23 | ±3.4e38 | ~10^-7 | 累加器 |

<div data-component="FPFormatComparison"></div>

[组件：FPFormatComparison - 可视化不同浮点格式的位域分布]

---

## 3.7 广播机制

广播 (Broadcasting) 允许不同 shape 的张量参与运算。Triton 的广播规则与 NumPy/PyTorch 一致。

### 3.7.1 广播规则

```python
@triton.jit
def broadcast_demo(
    X_ptr, Y_ptr, Z_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # ========== 广播规则：从右向左对齐 ==========
    # 规则 1：如果两个维度不相等且都不是 1，则报错
    # 规则 2：如果维度是 1，则扩展（复制）到与另一个匹配

    # 示例 1：(BLOCK_M, 1) + (1, BLOCK_N) → (BLOCK_M, BLOCK_N)
    col_vector = offs_m[:, None]  # shape: (BLOCK_M, 1) — 列向量
    row_vector = offs_n[None, :]  # shape: (1, BLOCK_N) — 行向量

    # 广播加法：列向量 + 行向量 → 矩阵
    # 每个 (i, j) 位置的值 = col_vector[i] + row_vector[j]
    grid_2d = col_vector + row_vector  # shape: (BLOCK_M, BLOCK_N)

    # 示例 2：(BLOCK_M, BLOCK_N) + (BLOCK_N,) → (BLOCK_M, BLOCK_N)
    a = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    b = tl.zeros((BLOCK_N,), dtype=tl.float32)
    # b 会被广播到 (1, BLOCK_N)，然后扩展到 (BLOCK_M, BLOCK_N)
    c = a + b  # shape: (BLOCK_M, BLOCK_N)

    # 示例 3：(BLOCK_M, BLOCK_N) + scalar → (BLOCK_M, BLOCK_N)
    d = a + 1.0  # 标量广播到所有元素

    # 示例 4：(BLOCK_M, 1) * (BLOCK_M, BLOCK_N) → (BLOCK_M, BLOCK_N)
    scale = tl.load(X_ptr + offs_m[:, None])  # shape: (BLOCK_M, 1)
    data = tl.load(Y_ptr + offs_m[:, None] * N + offs_n[None, :])  # shape: (BLOCK_M, BLOCK_N)
    scaled_data = scale * data  # 每行乘以对应的缩放因子

    tl.store(Z_ptr + offs_m[:, None] * N + offs_n[None, :], scaled_data)
```

**广播规则图示**：

```
广播规则（从右向左对齐维度）：

示例 1:
  A: shape (4, 1)     B: shape (1, 8)
  +---+               +---+---+---+---+---+---+---+---+
  | a0|               | b0| b1| b2| b3| b4| b5| b6| b7|
  | a1|               +---+---+---+---+---+---+---+---+
  | a2|
  | a3|
  +---+

  A + B → shape (4, 8):
  +----+----+----+----+----+----+----+----+
  |a0+b0|a0+b1|a0+b2|a0+b3|a0+b4|a0+b5|a0+b6|a0+b7|
  |a1+b0|a1+b1|a1+b2|a1+b3|a1+b4|a1+b5|a1+b6|a1+b7|
  |a2+b0|a2+b1|a2+b2|a2+b3|a2+b4|a2+b5|a2+b6|a2+b7|
  |a3+b0|a3+b1|a3+b2|a3+b3|a3+b4|a3+b5|a3+b6|a3+b7|
  +----+----+----+----+----+----+----+----+

示例 2:
  A: shape (4, 8)     B: shape (8,)
  → B 被广播为 (1, 8)，然后扩展为 (4, 8)
  → 逐元素运算

示例 3: 不能广播的情况
  A: shape (4, 8)     B: shape (3, 8)
  → 第一维 4 ≠ 3 且都不是 1 → 报错！
```

### 3.7.2 expand_dims

`tl.expand_dims` 用于在指定位置插入一个大小为 1 的新维度，常用于构造广播所需的 shape。

```python
@triton.jit
def expand_dims_demo(
    X_ptr, Y_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    x = tl.load(X_ptr + offsets)  # shape: (BLOCK,)

    # 在 axis=0 插入新维度 → shape: (1, BLOCK)
    x_row = tl.expand_dims(x, 0)

    # 在 axis=1 插入新维度 → shape: (BLOCK, 1)
    x_col = tl.expand_dims(x, 1)

    # 等价的 None 索引语法：
    x_row_eq = x[None, :]       # shape: (1, BLOCK)
    x_col_eq = x[:, None]       # shape: (BLOCK, 1)

    # 使用 expand_dims 进行广播运算
    # 创建一个行向量和一个列向量
    rows = tl.expand_dims(tl.arange(0, BLOCK), 1)  # (BLOCK, 1)
    cols = tl.expand_dims(tl.arange(0, BLOCK), 0)  # (1, BLOCK)

    # 广播加法：创建距离矩阵
    distance = tl.abs(rows - cols)  # shape: (BLOCK, BLOCK)

    tl.store(Y_ptr + offsets, tl.sum(distance, axis=1))
```

### 3.7.3 隐式广播

Triton 在某些操作中支持隐式广播，无需显式调用 `expand_dims`。

```python
@triton.jit
def implicit_broadcast_demo(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 加载矩阵: shape (BLOCK_M, BLOCK_N)
    a_ptrs = A_ptr + offs_m[:, None] * N + offs_n[None, :]
    a = tl.load(a_ptrs)

    # 加载行偏置: shape (BLOCK_M,) — 一维
    row_bias = tl.load(Bias_ptr + offs_m)

    # 隐式广播：一维 + 二维
    # row_bias (BLOCK_M,) 被隐式扩展为 (BLOCK_M, 1)，然后广播到 (BLOCK_M, BLOCK_N)
    # 等价于: a + row_bias[:, None]
    result = a + row_bias

    # 但为了代码清晰和避免歧义，建议显式广播
    result_explicit = a + row_bias[:, None]  # 推荐

    tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :], result_explicit)
```

---

## 3.8 高级操作

### 3.8.1 tl.trans — 转置

`tl.trans` 对二维张量进行转置，交换两个维度。

```python
@triton.jit
def trans_demo(
    X_ptr, Y_ptr, M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 加载一个 (BLOCK_M, BLOCK_N) 的 tile
    x_ptrs = X_ptr + offs_m[:, None] * N + offs_n[None, :]
    x = tl.load(x_ptrs)  # shape: (BLOCK_M, BLOCK_N)

    # 转置：shape (BLOCK_M, BLOCK_N) → (BLOCK_N, BLOCK_M)
    x_t = tl.trans(x)

    # 等价的写法（Python 语法糖）
    x_t_alt = x.T  # 与 tl.trans(x) 等价

    # 转置的典型应用：矩阵乘法中 B 矩阵的加载优化
    # 如果 B 按行存储但需要按列加载，可以先加载再转置
    # B: shape (K, N) → 加载 (BLOCK_K, BLOCK_N) → 不需要转置
    # B^T: shape (N, K) → 加载 (BLOCK_N, BLOCK_K) → 转置为 (BLOCK_K, BLOCK_N)

    # 存储转置后的结果（需要调整索引）
    y_ptrs = Y_ptr + offs_n[:, None] * M + offs_m[None, :]
    tl.store(y_ptrs, x_t)
```

**转置的底层实现**：

```
转置操作在 Triton 中主要是元数据操作（改变 shape 信息），
实际的物理数据移动在后续的 load/store 或 dot 操作中体现。

原始张量 (3×4):     转置后 (4×3):
+--+--+--+--+       +--+--+--+
|00|01|02|03|       |00|10|20|
+--+--+--+--+       +--+--+--+
|10|11|12|13|       |01|11|21|
+--+--+--+--+       +--+--+--+
|20|21|22|23|       |02|12|22|
+--+--+--+--+       +--+--+--+
                     |03|13|23|
                     +--+--+--+

在 Tensor Core 操作中，转置可能影响矩阵的 "row-major" vs "col-major" 布局，
进而影响 Tensor Core 指令的选择。
```

### 3.8.2 tl.dot_trans — 带转置的矩阵乘法

`tl.dot_trans` 允许在矩阵乘法中对输入矩阵进行转置，避免显式转置操作的开销。

```python
@triton.jit
def dot_trans_demo(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 加载 A: shape (BLOCK_M, BLOCK_K) — 正常
    a_ptrs = A_ptr + offs_m[:, None] * K + offs_k[None, :]
    a = tl.load(a_ptrs)

    # 加载 B^T: 形状是 (BLOCK_N, BLOCK_K) — 注意维度顺序
    # 如果 B 的原始形状是 (K, N) 且按行存储，
    # 那么 B^T 的加载方式是按列访问 B
    b_t_ptrs = B_ptr + offs_n[:, None] * K + offs_k[None, :]
    b_t = tl.load(b_t_ptrs)  # shape: (BLOCK_N, BLOCK_K)

    # 使用 tl.dot_trans 计算 A @ B = A @ (B^T)^T
    # 参数：trans_a=False (A 不转置), trans_b=True (B 需要转置)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    accumulator += tl.dot(a, b_t, trans_b=True)
    # 等价于: accumulator += tl.dot(a, tl.trans(b_t))
    # 但 tl.dot_trans 避免了显式转置的开销

    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, accumulator)
```

### 3.8.3 tl.cat — 拼接

`tl.cat` 沿指定维度拼接两个张量。

```python
@triton.jit
def cat_demo(
    A_ptr, B_ptr, C_ptr, M, N_A, N_B,
    BLOCK_M: tl.constexpr,
    BLOCK_NA: tl.constexpr,
    BLOCK_NB: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # 加载两个张量
    a = tl.load(A_ptr + offs_m[:, None] * N_A + tl.arange(0, BLOCK_NA)[None, :])
    b = tl.load(B_ptr + offs_m[:, None] * N_B + tl.arange(0, BLOCK_NB)[None, :])

    # 沿 axis=1（列方向）拼接
    # a: shape (BLOCK_M, BLOCK_NA) + b: shape (BLOCK_M, BLOCK_NB)
    # → result: shape (BLOCK_M, BLOCK_NA + BLOCK_NB)
    c = tl.cat(a, b, axis=1)

    # 沿 axis=0（行方向）拼接
    # a: shape (BLOCK_M, BLOCK_NA) + b: shape (BLOCK_M, BLOCK_NA)
    # → result: shape (2 * BLOCK_M, BLOCK_NA)
    # c_rows = tl.cat(a, b, axis=0)

    tl.store(C_ptr + offs_m[:, None] * (BLOCK_NA + BLOCK_NB) +
             tl.arange(0, BLOCK_NA + BLOCK_NB)[None, :], c)
```

### 3.8.4 tl.interleave — 交错

`tl.interleave` 将两个张量的元素交错排列。

```python
@triton.jit
def interleave_demo(
    A_ptr, B_ptr, C_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    half = BLOCK // 2

    # 加载两个张量
    a = tl.load(A_ptr + offsets[:half])  # shape: (half,)
    b = tl.load(B_ptr + offsets[:half])  # shape: (half,)

    # 交错排列：[a0, b0, a1, b1, a2, b2, ...]
    # 结果 shape: (BLOCK,)
    interleaved = tl.interleave(a, b)

    # 等价的逻辑：
    # result[0] = a[0]
    # result[1] = b[0]
    # result[2] = a[1]
    # result[3] = b[1]
    # ...

    tl.store(C_ptr + offsets, interleaved)
```

**交错的典型应用**：在某些硬件上，交错排列可以改善内存合并访问模式。

### 3.8.5 tl.join — 组合

`tl.join` 将两个一维张量组合成一个二维张量，第一维大小为 2。

```python
@triton.jit
def join_demo(
    A_ptr, B_ptr, C_ptr, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    a = tl.load(A_ptr + offsets, mask=mask, other=0.0)  # shape: (BLOCK,)
    b = tl.load(B_ptr + offsets, mask=mask, other=0.0)  # shape: (BLOCK,)

    # 组合为二维张量
    # a: shape (BLOCK,) + b: shape (BLOCK,)
    # → result: shape (2, BLOCK)
    combined = tl.join(a, b)

    # combined[0, :] = a
    # combined[1, :] = b

    # 组合的典型应用：准备数据用于需要 2D 输入的操作
    # 例如，将实部和虚部组合为复数表示

    tl.store(C_ptr + offsets, combined[0, :])  # 存储 a 的部分
```

**高级操作对比汇总**：

| 操作 | 输入 | 输出 | 典型用途 |
|:---|:---|:---|:---|
| `tl.trans(x)` | (M, N) | (N, M) | 矩阵转置 |
| `tl.dot(a, b)` | (M, K) × (K, N) | (M, N) | 矩阵乘法 |
| `tl.dot(a, b, trans_b=True)` | (M, K) × (N, K) | (M, N) | 带转置的矩阵乘法 |
| `tl.cat(a, b, axis)` | (M, N1) + (M, N2) | (M, N1+N2) | 沿轴拼接 |
| `tl.interleave(a, b)` | (N,) + (N,) | (2*N,) | 交错排列 |
| `tl.join(a, b)` | (N,) + (N,) | (2, N) | 组合为 2D |
| `tl.expand_dims(x, axis)` | (M, N) | (M, 1, N) | 插入新维度 |

<div data-component="AdvancedOpsPlayground"></div>

[组件：AdvancedOpsPlayground - 交互式演示高级张量操作的输入输出关系]

---

## 3.9 性能优化要点与常见陷阱

### 3.9.1 性能优化 Checklist

```python
@triton.jit
def performance_best_practices(
    A, B, C, M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # ✓ 优化 1：累加器使用 float32，保持数值精度
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K

        # ✓ 优化 2：合理使用 mask 和 other
        # other=0.0 保证越界元素不影响乘法结果
        a_ptrs = A + (offs_m[:, None] * K + offs_k[None, :])
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)

        b_ptrs = B + (offs_k[:, None] * N + offs_n[None, :])
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)

        # ✓ 优化 3：使用 tl.dot 而非手动循环
        accumulator += tl.dot(a, b)

        offs_k += BLOCK_K

    # ✓ 优化 4：延迟类型转换到最后
    c = accumulator.to(tl.float16)

    c_ptrs = C + (offs_m[:, None] * N + offs_n[None, :])
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

### 3.9.2 常见陷阱与解决方案

**陷阱 1：tl.dot 的输入 dtype 不匹配**

```python
@triton.jit
def pitfall_dot_dtype(A, B, C, BLOCK: tl.constexpr):
    a = tl.zeros((BLOCK, BLOCK), dtype=tl.float16)
    b = tl.zeros((BLOCK, BLOCK), dtype=tl.bfloat16)

    # 错误！tl.dot 的两个输入必须是相同 dtype
    # c = tl.dot(a, b)  # 编译错误！

    # 正确做法：统一 dtype
    c = tl.dot(a, b.to(tl.float16))  # 或 a.to(tl.bfloat16)
```

**陷阱 2：归约时 other 值选择不当**

```python
@triton.jit
def pitfall_reduce_other(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # 错误！max 的 other=0.0，如果所有有效值都是负数，结果错误
    # x_bad = tl.load(X + offsets, mask=mask, other=0.0)
    # row_max_bad = tl.max(x_bad, axis=0)  # 可能返回 0.0 而非实际最大值

    # 正确做法：max 的 other 应该是 -inf
    x_good = tl.load(X + offsets, mask=mask, other=-float('inf'))
    row_max_good = tl.max(x_good, axis=0)

    tl.store(Y + pid, row_max_good)
```

**陷阱 3：Tile 大小不满足对齐要求**

```python
@triton.jit
def pitfall_alignment(A, B, C, BLOCK: tl.constexpr):
    # 某些 GPU 对 tl.dot 的输入大小有对齐要求
    # 例如 Ampere 要求 k 维度至少为 16

    a = tl.zeros((64, 8), dtype=tl.float16)   # k=8，可能不满足要求
    b = tl.zeros((8, 64), dtype=tl.float16)
    # c = tl.dot(a, b)  # 可能性能极差或编译失败

    # 推荐：k 维度至少为 16（最好为 32 的倍数）
    a_safe = tl.zeros((64, 32), dtype=tl.float16)
    b_safe = tl.zeros((32, 64), dtype=tl.float16)
    c = tl.dot(a_safe, b_safe)  # OK
```

**陷阱 4：不必要的数据拷贝**

```python
@triton.jit
def pitfall_copy(X, Y, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    x = tl.load(X + offsets)

    # 错误！这个赋值创建了数据的拷贝
    # y = x + 0.0  # 不必要的运算

    # 正确做法：直接使用引用（Triton 中张量是值类型，赋值就是拷贝）
    # 但 Triton 编译器会优化掉不必要的拷贝，所以关键是减少冗余运算
    y = x  # Triton 中这就是拷贝，但编译器会优化
    tl.store(Y + offsets, y)
```

**陷阱 5：tl.arange 的范围不对齐**

```python
@triton.jit
def pitfall_arange(X, Y, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # BLOCK 应该是 2 的幂（至少是 16 的倍数）
    # 这样编译器才能生成高效的向量化加载指令

    # 推荐的 BLOCK 值：16, 32, 64, 128, 256, 512, 1024
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask, other=0.0)
    tl.store(Y + offsets, x, mask=mask)
```

---

## 本章小结

本章系统介绍了 Triton 的张量操作与计算原语，这些是编写任何 Triton kernel 的基础构件。核心要点如下：

1. **张量创建**：`tl.zeros`、`tl.full` 创建常量张量，`tl.arange` 创建索引序列。所有 shape 参数必须是 `tl.constexpr` 编译期常量。张量在底层映射到寄存器，是值类型而非堆对象。

2. **加载与存储**：`tl.load` 支持 `mask`（边界保护）、`other`（默认值）、`cache_modifier`（缓存策略）、`eviction_policy`（驱逐策略）等参数。`tl.store` 支持 `mask` 和 `boundary_check`。内存合并访问是性能关键——确保同一 warp 内的线程访问连续地址。

3. **算术运算**：所有运算都是逐元素的。比较运算返回 `bool` 张量，配合 `tl.where` 实现条件选择。一元数学函数（`tl.exp`、`tl.log`、`tl.sqrt`）被编译为高效 GPU 指令。类型提升遵循"低精度→高精度"的自动转换规则。

4. **归约操作**：`tl.sum`、`tl.min`、`tl.max` 沿指定轴压缩张量。`tl.reduce` 支持自定义归约函数。底层使用 Warp Shuffle（`__shfl_xor_sync`）+ 共享内存实现高效的并行归约。

5. **矩阵乘法**：`tl.dot` 是最重要的计算原语，在 Ampere+ GPU 上自动映射到 Tensor Core 指令（HMMA/MMA）。支持的 dtype 组合包括 f16×f16→f16/f32、bf16×bf16→f32、tf32×tf32→f32、int8×int8→int32。累加器建议使用 float32 以保持精度。

6. **类型转换**：`.to(dtype)` 进行显式转换，算术运算中自动进行隐式提升。FP8 类型（`tl.float8e4m3fn`、`tl.float8e5m2`）在 Hopper+ GPU 上支持，用于加速 LLM 推理。

7. **广播机制**：遵循 NumPy/PyTorch 的广播规则，从右向左对齐维度。`tl.expand_dims` 插入新维度，`None` 索引是更简洁的等价语法。

8. **高级操作**：`tl.trans` 转置、`tl.dot_trans` 带转置的矩阵乘法、`tl.cat` 拼接、`tl.interleave` 交错、`tl.join` 组合。这些操作避免了显式数据搬运的开销。

---

## 思考题

### 概念理解题

1. **constexpr 的编译影响**：如果一个 kernel 有两个 `tl.constexpr` 参数，`BLOCK_M` 可以取 {64, 128, 256}，`DTYPE` 可以取 {fp16, fp32}，那么最多会生成多少个不同的编译产物？这对编译时间有什么影响？

2. **mask 机制的本质**：为什么 Triton 使用 `mask` 参数而不是 `if/else` 分支来处理边界条件？从 SIMT 执行模型的角度分析两者的性能差异。

3. **归约的结合律要求**：`tl.reduce` 要求 `combine_fn` 满足结合律。如果使用浮点减法（不满足结合律）作为归约函数，结果会怎样？举例说明。

### 实践题

4. **实现 Row-major Softmax**：使用本章学到的 `tl.load`、`tl.max`、`tl.exp`、`tl.sum` 实现一个逐行 softmax kernel。要求处理非 BLOCK_SIZE 倍数的列数。

5. **实现带偏置的矩阵乘法**：在标准矩阵乘法基础上，加入行偏置（每个输出行加一个不同的标量）。利用广播机制实现。

6. **对比不同 cache_modifier**：写一个简单的向量加法 kernel，分别使用 `"ca"` 和 `"cg"` cache modifier，对比两者在只读场景和读写混合场景下的性能差异。

### 设计思考题

7. **Tile 大小的选择**：在矩阵乘法中，`BLOCK_M`、`BLOCK_N`、`BLOCK_K` 应该如何选择？考虑 Tensor Core 的指令约束、寄存器压力、占用率等因素。

8. **FP8 的精度权衡**：FP8 E4M3 和 FP8 E5M2 分别适合什么场景？为什么前向传播用 E4M3 而反向传播用 E5M2？

9. **广播的性能影响**：隐式广播在什么情况下会产生性能开销？什么情况下编译器能完全优化掉？

### 进阶题

10. **自定义归约实现 L2 范数**：使用 `tl.reduce` 实现向量的 L2 范数计算。需要处理数值溢出问题（先求最大值，再归一化后求和）。

11. **Tensor Core 指令分析**：使用 `TRITON_PRINT_AUTOTUNING=1` 或查看生成的 PTX 代码，分析一个简单的 `tl.dot` 操作被映射到了哪条 Tensor Core 指令。

12. **内存访问模式优化**：给定一个列优先存储的矩阵，设计一种加载模式使得内存访问尽可能合并。对比直接按列加载和先加载行再转置的性能差异。

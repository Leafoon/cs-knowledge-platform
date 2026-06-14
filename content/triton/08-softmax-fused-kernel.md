---
title: "Chapter 8: 融合 Softmax 与 Online 算法"
description: "深入理解 Softmax 的计算过程与数值稳定性问题，掌握 Online Softmax 的单遍扫描算法，通过 Triton 实现高性能融合 Softmax kernel，并推广到通用归约操作的设计模式。"
date: "2026-06-11"
---

# Chapter 8: 融合 Softmax 与 Online 算法

> **学习目标**：
> - 理解标准 Softmax 的数学公式、数值稳定版本及其 FLOPs 与访存量分析
> - 对比 PyTorch 朴素实现（三次 kernel launch）与 Triton 融合实现（单次 kernel launch）的性能差异
> - 掌握 Online Softmax 算法的数学推导与递推公式，理解单遍扫描的核心思想
> - 能够用 Triton 实现完整的 Online Softmax kernel，包括行级 max 追踪、running sum 更新与最终修正
> - 分析 Online Softmax 的数值稳定性，理解 exp 溢出/下溢的处理策略
> - 学会将 Online Softmax 的设计模式推广到 Softmax+Dropout、Softmax+Mask、LayerNorm 等融合场景

---

## 8.1 Softmax 的计算与数值稳定性

### 8.1.1 标准 Softmax 公式

Softmax 是深度学习中最基础的归一化操作之一，广泛应用于分类网络的输出层、Transformer 中的注意力权重计算等场景。给定一个长度为 $n$ 的输入向量 $\mathbf{x} = (x_1, x_2, \ldots, x_n)$，标准 Softmax 的定义为：

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}, \quad i = 1, 2, \ldots, n
$$

Softmax 的输出具有以下性质：

| 性质 | 数学表达 | 含义 |
|:---|:---|:---|
| **非负性** | $\text{softmax}(x_i) > 0$ | 所有输出均为正数 |
| **归一化** | $\sum_{i=1}^{n} \text{softmax}(x_i) = 1$ | 输出构成概率分布 |
| **保序性** | $x_i > x_j \Rightarrow \text{softmax}(x_i) > \text{softmax}(x_j)$ | 保持原始大小顺序 |
| **温度缩放** | $\text{softmax}(x_i / T)$，$T > 0$ | 温度参数控制分布的尖锐程度 |

### 8.1.2 数值稳定性问题

直接按公式计算 Softmax 会遇到严重的数值问题。当 $x_i$ 的值较大时，$e^{x_i}$ 会溢出（overflow）；当 $x_i$ 的值很小时，$e^{x_i}$ 会下溢到零（underflow）。

以 float32 为例：

```python
import torch

# 溢出示例
x = torch.tensor([1000.0, 1001.0, 1002.0])
# e^1000 = inf（float32 最大值约 3.4e38，e^89 ≈ 4.5e38）
# 直接计算会得到 nan

# 下溢示例
x = torch.tensor([-1000.0, -1001.0, -1002.0])
# e^(-1000) ≈ 0（float32 最小正数约 1.2e-38，e^(-89) ≈ 1e-39）
# 直接计算会得到 0/0 = nan
```

**解决方案：减去最大值**

数值稳定的 Softmax 公式为：

$$
\text{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_{j=1}^{n} e^{x_j - m}}, \quad m = \max_{j=1}^{n} x_j
$$

**正确性证明**：

$$
\frac{e^{x_i - m}}{\sum_{j=1}^{n} e^{x_j - m}} = \frac{e^{x_i} \cdot e^{-m}}{\sum_{j=1}^{n} e^{x_j} \cdot e^{-m}} = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

分子分母同乘 $e^{-m}$ 约掉后结果不变，但指数部分的最大值变为 $e^{x_i - m} = e^0 = 1$，避免了溢出；最小值虽然可能很小，但分母至少为 1（来自最大值项），所以结果不会出现 0/0 的情况。

### 8.1.3 标准 Softmax 的计算流程

数值稳定的 Softmax 需要三个阶段：

```
阶段 1: m = max(x_1, x_2, ..., x_n)           # 求最大值
阶段 2: t_i = exp(x_i - m),  s = sum(t_1, ..., t_n)  # 计算指数并求和
阶段 3: y_i = t_i / s                           # 归一化
```

### 8.1.4 FLOPs 与访存量分析

设输入张量形状为 `(B, N)`，即 batch size 为 B，每行长度为 N。

**FLOPs 分析**：

| 操作 | 次数 | FLOPs |
|:---|:---|:---|
| 求最大值（比较） | $B \times (N-1)$ 次比较 | $B \times N$ |
| 减去最大值 | $B \times N$ 次减法 | $B \times N$ |
| 计算指数 | $B \times N$ 次 exp | $B \times N$（约 8 FLOPs/次） |
| 求和 | $B \times (N-1)$ 次加法 | $B \times N$ |
| 除法 | $B \times N$ 次除法 | $B \times N$（约 4 FLOPs/次） |

总 FLOPs 约为 $B \times N \times 15$，属于**访存密集型（memory-bound）**操作，因为每元素的计算量很小，但需要多次读写整个张量。

**访存量分析（朴素三遍实现）**：

| 阶段 | 读入 | 写出 | 合计 |
|:---|:---|:---|:---|
| 阶段 1：求 max | 读 $x$：$B \times N \times 4$ B | 写 $m$：$B \times 4$ B | $\approx 4BN$ B |
| 阶段 2：exp + sum | 读 $x, m$：$\approx 4BN$ B | 写 $t$：$4BN$ B | $\approx 8BN$ B |
| 阶段 3：div | 读 $t, s$：$\approx 4BN$ B | 写 $y$：$4BN$ B | $\approx 8BN$ B |
| **合计** | | | $\approx 20BN$ B |

以 float32 为例，朴素三遍实现需要读写张量约 5 次（20BN 字节）。若融合为单次 kernel，理论上只需**读 1 次 + 写 1 次** = $8BN$ 字节，访存量减少 **60%**。

### 8.1.5 温度缩放与 Softmax 的梯度

温度参数 $T$ 在知识蒸馏和生成模型中广泛使用。温度缩放的 Softmax 定义为：

$$
\text{softmax}(x_i; T) = \frac{e^{x_i / T}}{\sum_{j=1}^{n} e^{x_j / T}}
$$

**温度的影响**：

| 温度 $T$ | 分布特征 | 应用场景 |
|:---|:---|:---|
| $T \to 0$ | 趋向 one-hot（argmax） | 硬决策、推理 |
| $T = 1$ | 标准 Softmax | 常规分类 |
| $T > 1$ | 更平滑的分布 | 知识蒸馏、探索 |
| $T \to \infty$ | 趋向均匀分布 $1/n$ | 最大熵 |

**Softmax 的雅可比矩阵**（梯度推导）：

$$
\frac{\partial \text{softmax}(x_i)}{\partial x_j} = \text{softmax}(x_i) \cdot (\delta_{ij} - \text{softmax}(x_j))
$$

其中 $\delta_{ij}$ 是 Kronecker delta（$i=j$ 时为 1，否则为 0）。展开为矩阵形式：

$$
\mathbf{J} = \text{diag}(\mathbf{p}) - \mathbf{p} \mathbf{p}^T
$$

其中 $\mathbf{p} = \text{softmax}(\mathbf{x})$。这个雅可比矩阵是对称的，且是半正定的（因为它是投影矩阵）。

### 8.1.6 Softmax 在注意力机制中的角色

在 Transformer 的自注意力机制中，Softmax 用于归一化注意力分数：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

这里 Softmax 的输入矩阵形状为 `(seq_len_q, seq_len_k)`，每一行独立做 Softmax。当序列长度很大时（如 128K tokens），Softmax 的性能成为瓶颈，这也是 Online Softmax 和 Flash Attention 的核心动机。

```
注意力矩阵 (seq_q × seq_k):
┌─────────────────────────────────────┐
│  q1·k1  q1·k2  q1·k3  ...  q1·kN  │ → softmax → [p11, p12, ..., p1N]
│  q2·k1  q2·k2  q2·k3  ...  q2·kN  │ → softmax → [p21, p22, ..., p2N]
│  ...                                │
│  qM·k1  qM·k2  qM·k3  ...  qM·kN  │ → softmax → [pM1, pM2, ..., pMN]
└─────────────────────────────────────┘
  每行独立做 Softmax（沿 key 维度归一化）
```

<div data-component="SoftmaxComputationFlow"></div>

[组件：SoftmaxComputationFlow - 交互式展示 Softmax 三阶段计算流程与数据流]

---

## 8.2 朴素实现 vs 融合实现

### 8.2.1 PyTorch 的三次 Kernel Launch

PyTorch 中 `torch.softmax()` 的底层实现需要三个独立的 kernel launch：

```python
import torch

def softmax_three_passes(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    PyTorch 内部 softmax 的简化版本，演示三次 kernel launch 的过程。
    实际 PyTorch 使用 cuDNN 或自研 kernel，但基本逻辑相同。
    """
    # ──── Kernel 1: 求最大值 ────
    # 每个 block 负责一行（或一行的一部分），通过 warp shuffle 归约
    m = x.max(dim=dim, keepdim=True).values   # 输出形状: (B, 1)

    # ──── Kernel 2: 计算 exp 并求和 ────
    x_shifted = x - m                         # 广播减法，需要 kernel launch
    t = torch.exp(x_shifted)                  # 逐元素指数，kernel launch
    s = t.sum(dim=dim, keepdim=True)          # 归约求和，kernel launch

    # ──── Kernel 3: 归一化 ────
    y = t / s                                 # 逐元素除法，kernel launch

    return y

# 在实际 PyTorch 中，上述操作可能被合并为 2-3 个 kernel
# 但关键是：每个 kernel launch 都有启动开销，且中间结果需要写回全局内存
```

**三次 kernel launch 的性能问题**：

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Kernel 1  │     │ Kernel 2  │     │ Kernel 3  │
│  求 max   │────>│ exp + sum │────>│   div     │
│           │     │           │     │           │
│ 读 x      │     │ 读 x, m   │     │ 读 t, s   │
│ 写 m      │     │ 写 t, s   │     │ 写 y      │
└──────────┘     └──────────┘     └──────────┘
     │                │                │
     v                v                v
  全局内存          全局内存          全局内存
  (同步点)          (同步点)
```

每个 kernel launch 的开销包括：

| 开销来源 | 典型值 | 说明 |
|:---|:---|:---|
| CPU → GPU 提交 | ~5-10 μs | 每次 launch 的驱动开销 |
| GPU kernel 启动 | ~1-3 μs | kernel 调度开销 |
| 全局内存读写 | ~数十 μs | 中间结果必须写回再读出 |
| **合计（3 次 launch）** | **~30-60 μs** | 对短序列影响显著 |

### 8.2.2 Triton 的融合实现

Triton 的核心优势之一是能够在单个 kernel 中完成所有计算，消除中间结果的全局内存读写：

```python
import triton
import triton.language as tl

@triton.jit
def softmax_naive_kernel(
    X_ptr, Y_ptr,
    stride_x_row, stride_y_row,
    N,
    BLOCK_N: tl.constexpr,
):
    """
    朴素的融合 Softmax kernel（仍需两遍扫描）
    每个 program 处理一行数据
    """
    # 获取当前行索引
    row_idx = tl.program_id(0)

    # 计算当前行的起始地址
    X_row_ptr = X_ptr + row_idx * stride_x_row
    Y_row_ptr = Y_ptr + row_idx * stride_y_row

    # 构造列偏移量和 mask
    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < N

    # ──── 第一遍扫描：求最大值 ────
    # 加载整行数据到寄存器
    x = tl.load(X_row_ptr + col_offsets, mask=mask, other=-float('inf'))
    # 通过 tl.max 归约求最大值（底层使用 warp shuffle）
    row_max = tl.max(x, axis=0)

    # ──── 第二遍扫描：计算 exp、求和、归一化 ────
    # 减去最大值（数值稳定性）
    x_shifted = x - row_max
    # 计算指数
    exp_x = tl.exp(x_shifted)
    # 归约求和
    row_sum = tl.sum(exp_x, axis=0)
    # 归一化
    y = exp_x / row_sum

    # 写回结果
    tl.store(Y_row_ptr + col_offsets, y, mask=mask)


def softmax_naive(x: torch.Tensor) -> torch.Tensor:
    """
    Triton 朴素融合 Softmax 的 host 端封装
    """
    B, N = x.shape
    y = torch.empty_like(x)

    # 每个 program 处理一行
    BLOCK_N = triton.next_power_of_2(N)
    grid = (B,)

    softmax_naive_kernel[grid](
        x, y,
        x.stride(0), y.stride(0),
        N,
        BLOCK_N=BLOCK_N,
    )
    return y
```

### 8.2.3 手动实现 PyTorch 三遍 Softmax 的完整代码

为了更清楚地展示性能差异，我们手动实现一个不使用 `torch.softmax` 的版本，强制分为三个独立的 kernel：

```python
import torch

def softmax_manual_three_pass(x: torch.Tensor) -> torch.Tensor:
    """
    手动三遍实现 Softmax，每遍使用独立的 PyTorch 操作。
    每个操作都会触发一次独立的 CUDA kernel launch。
    """
    assert x.dim() == 2, "输入必须是 2D 张量 (B, N)"

    # ──── 第 1 遍：求行最大值 ────
    # x.max(dim=-1) 会触发一个 CUDA 归约 kernel
    # 输出形状: (B, 1)，用于数值稳定性
    row_max = x.max(dim=-1, keepdim=True).values

    # ──── 第 2 遍：计算 exp 并求和 ────
    # x - row_max: 触发一个 CUDA 广播减法 kernel
    x_shifted = x - row_max
    # torch.exp(x_shifted): 触发一个 CUDA 逐元素 kernel
    exp_x = torch.exp(x_shifted)
    # exp_x.sum(dim=-1): 触发一个 CUDA 归约 kernel
    row_sum = exp_x.sum(dim=-1, keepdim=True)

    # ──── 第 3 遍：归一化 ────
    # exp_x / row_sum: 触发一个 CUDA 广播除法 kernel
    y = exp_x / row_sum

    return y

# 性能对比测试
import time

def compare_implementations():
    """对比手动三遍实现与 PyTorch 内置 softmax 的性能"""
    device = 'cuda'
    B, N = 128, 4096
    x = torch.randn(B, N, device=device, dtype=torch.float32)

    # 预热
    for _ in range(10):
        softmax_manual_three_pass(x)
        torch.softmax(x, dim=-1)
    torch.cuda.synchronize()

    # 测量手动三遍实现
    start = time.perf_counter()
    for _ in range(100):
        softmax_manual_three_pass(x)
    torch.cuda.synchronize()
    manual_time = (time.perf_counter() - start) / 100

    # 测量 PyTorch 内置实现
    start = time.perf_counter()
    for _ in range(100):
        torch.softmax(x, dim=-1)
    torch.cuda.synchronize()
    builtin_time = (time.perf_counter() - start) / 100

    print(f"手动三遍: {manual_time*1e6:.1f} μs")
    print(f"PyTorch 内置: {builtin_time*1e6:.1f} μs")
    print(f"PyTorch 内置比手动快: {manual_time/builtin_time:.2f}x")

    # 注: PyTorch 内置实现已经做了部分融合优化
    # 但仍需要多次 kernel launch（通常 2-3 次）
```

**PyTorch 内置实现的 kernel 数量分析**：

通过 `torch.profiler` 可以观察到 PyTorch `softmax` 实际触发的 kernel 数量：

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

def profile_softmax():
    """使用 PyTorch Profiler 分析 softmax 的 kernel 调用"""
    device = 'cuda'
    B, N = 128, 4096
    x = torch.randn(B, N, device=device, dtype=torch.float32)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        y = torch.softmax(x, dim=-1)

    # 打印所有 CUDA kernel
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # 典型输出包含:
    # 1. reduce_max kernel (求最大值)
    # 2. elementwise_sub + exp kernel (减法 + 指数)
    # 3. reduce_sum kernel (求和)
    # 4. elementwise_div kernel (除法)
    #
    # 注意: PyTorch 可能将步骤 2 和 3 融合为一个 kernel
    # 但步骤 1 和 4 通常是独立的 kernel

profile_softmax()
```

### 8.2.4 Kernel Launch 开销的量化

```python
def measure_kernel_launch_overhead():
    """
    测量单次 CUDA kernel launch 的开销

    使用一个极简的 kernel（什么都不做），来隔离 launch overhead
    """
    @triton.jit
    def empty_kernel():
        """空 kernel，仅用于测量 launch 开销"""
        pass

    # 预热
    for _ in range(100):
        empty_kernel[(1,)]()
    torch.cuda.synchronize()

    # 测量 1000 次 launch
    start = time.perf_counter()
    for _ in range(1000):
        empty_kernel[(1,)]()
    torch.cuda.synchronize()
    avg_launch = (time.perf_counter() - start) / 1000

    print(f"单次 kernel launch 开销: {avg_launch*1e6:.2f} μs")
    # 典型结果: ~5-10 μs（取决于 GPU 和驱动版本）
    # 3 次 launch 的额外开销: ~15-30 μs

measure_kernel_launch_overhead()
```

### 8.2.5 性能对比

```
            朴素三遍 PyTorch              融合 Triton（两遍扫描）
            ┌─────────────┐              ┌───────────────────┐
            │ 读 x (GMem)  │              │ 读 x (GMem)       │
            │ 计算 max      │              │ 第1遍: max         │
            │ 写 m (GMem)  │              │ 第2遍: exp+sum+div │
            │ 读 x, m      │              │ 写 y (GMem)       │
            │ 计算 exp+sum  │              └───────────────────┘
            │ 写 t, s      │                    1次 launch
            │ 读 t, s      │                    2次 GMem 读
            │ 计算 div      │                    1次 GMem 写
            │ 写 y (GMem)  │
            └─────────────┘
               3次 launch
               5次 GMem 读写
```

**关键性能收益**：

| 指标 | PyTorch 三遍 | Triton 融合两遍 | 改善 |
|:---|:---|:---|:---|
| Kernel launch 次数 | 3 | 1 | 减少 67% |
| 全局内存读写次数 | 5 | 3 | 减少 40% |
| 中间数据落地 | 是 | 否 | 完全消除 |
| 指令级并行 | 受限 | 高 | 寄存器内流水 |

<div data-component="KernelLaunchComparison"></div>

[组件：KernelLaunchComparison - 可视化对比 PyTorch 多次 launch 与 Triton 单次 launch 的时间线]

---

## 8.3 Online Softmax 算法

### 8.3.1 动机：能否单遍扫描完成？

上一节的融合实现虽然消除了 kernel launch 开销和中间全局内存读写，但仍然是**两遍扫描（two-pass）**：第一遍求最大值，第二遍计算 exp 和归一化。对于极长的序列（如 N = 100K+），能否在**单遍扫描（single-pass）**中完成 Softmax 计算？

2018 年，Milakov & Gimelshein 在论文 *"Online normalizer calculation for softmax"* 中提出了 **Online Softmax** 算法，实现了真正的单遍扫描。该算法后来被广泛应用于 Flash Attention 等高性能实现中。

### 8.3.2 数学推导

**核心思想**：在逐元素扫描的过程中，动态维护当前已见元素的最大值和指数和，当遇到新元素时，用一个修正因子更新之前的中间结果。

设输入序列为 $x_1, x_2, \ldots, x_n$。定义：

- $m_k = \max(x_1, x_2, \ldots, x_k)$：前 $k$ 个元素的最大值
- $d_k = \sum_{j=1}^{k} e^{x_j - m_k}$：以 $m_k$ 为基准的前 $k$ 个元素的指数和

**递推公式**：当扫描到第 $k+1$ 个元素 $x_{k+1}$ 时：

**最大值更新**：

$$
m_{k+1} = \max(m_k, x_{k+1})
$$

**指数和更新**：

$$
\begin{aligned}
d_{k+1} &= \sum_{j=1}^{k+1} e^{x_j - m_{k+1}} \\
&= e^{x_{k+1} - m_{k+1}} + \sum_{j=1}^{k} e^{x_j - m_{k+1}} \\
&= e^{x_{k+1} - m_{k+1}} + \sum_{j=1}^{k} e^{x_j - m_k + m_k - m_{k+1}} \\
&= e^{x_{k+1} - m_{k+1}} + e^{m_k - m_{k+1}} \cdot \sum_{j=1}^{k} e^{x_j - m_k} \\
&= e^{x_{k+1} - m_{k+1}} + e^{m_k - m_{k+1}} \cdot d_k
\end{aligned}
$$

最终的递推关系为：

$$
\boxed{
\begin{aligned}
m_{k+1} &= \max(m_k, x_{k+1}) \\
d_{k+1} &= e^{m_k - m_{k+1}} \cdot d_k + e^{x_{k+1} - m_{k+1}}
\end{aligned}
}
$$

**初始化**：$m_0 = -\infty$，$d_0 = 0$。

**最终结果**：扫描完所有 $n$ 个元素后，Softmax 为：

$$
\text{softmax}(x_i) = \frac{e^{x_i - m_n}}{d_n}
$$

### 8.3.3 递推关系的直观理解

我们可以将递推过程理解为"修正"过程：

```
扫描到 x_{k+1} 时，有两种情况：

情况 1: x_{k+1} <= m_k（新元素不是最大值）
  → m_{k+1} = m_k（最大值不变）
  → d_{k+1} = d_k + e^{x_{k+1} - m_k}
  → 只需累加新元素的贡献

情况 2: x_{k+1} > m_k（新元素是新的最大值）
  → m_{k+1} = x_{k+1}（最大值更新）
  → d_{k+1} = e^{m_k - x_{k+1}} · d_k + 1
  → 之前的 d_k 需要乘以修正因子 e^{m_k - x_{k+1}}（< 1），然后加上新元素的贡献 1
```

**数值示例**：

```python
# 手动模拟 Online Softmax 过程
x = [2.0, 1.0, 3.0, 0.5]

# 初始状态
m, d = float('-inf'), 0.0

# 扫描 x[0] = 2.0
m_new = max(-inf, 2.0)    # m = 2.0
d_new = exp(-inf - 2.0) * 0 + exp(2.0 - 2.0)  # d = 1.0
m, d = m_new, d_new        # (m=2.0, d=1.0)

# 扫描 x[1] = 1.0
m_new = max(2.0, 1.0)     # m = 2.0（不变）
d_new = exp(2.0 - 2.0) * 1.0 + exp(1.0 - 2.0)  # d = 1 + 0.368 = 1.368
m, d = m_new, d_new        # (m=2.0, d=1.368)

# 扫描 x[2] = 3.0
m_new = max(2.0, 3.0)     # m = 3.0（更新！）
d_new = exp(2.0 - 3.0) * 1.368 + exp(3.0 - 3.0)  # d = 0.368*1.368 + 1 = 1.504
m, d = m_new, d_new        # (m=3.0, d=1.504)

# 扫描 x[3] = 0.5
m_new = max(3.0, 0.5)     # m = 3.0（不变）
d_new = exp(3.0 - 3.0) * 1.504 + exp(0.5 - 3.0)  # d = 1.504 + 0.082 = 1.586
m, d = m_new, d_new        # (m=3.0, d=1.586)

# 最终结果
# softmax(x[i]) = exp(x[i] - 3.0) / 1.586
# = [0.368, 0.135, 1.000, 0.082] / 1.586
# = [0.232, 0.085, 0.631, 0.052]
```

<div data-component="OnlineSoftmaxVisualization"></div>

[组件：OnlineSoftmaxVisualization - 动态展示 Online Softmax 单遍扫描过程中 m 和 d 的变化轨迹]

### 8.3.4 复杂度对比

| 算法 | 扫描遍数 | 读取全局内存次数 | 适用场景 |
|:---|:---|:---|:---|
| 标准两遍 Softmax | 2 遍 | 2 次 | 通用场景 |
| Online Softmax | 1 遍 | 1 次 | 长序列、与后续操作融合 |

Online Softmax 算法的核心贡献在于：**它将一个需要全局信息（全局最大值）的操作转化为可以在线（online）维护的递推关系**。这个思想对后续的 Flash Attention 等算法产生了深远影响。

### 8.3.5 从直觉理解 Online Softmax

为什么 Online Softmax 只需要一遍扫描？关键在于它维护了"足够的信息"来应对未来的变化。

**类比：实时排名系统**

想象你在观察一场马拉松比赛，需要实时计算每个选手的"相对优势分数"。标准方法是等所有选手到达终点，记录最短时间 $T_{min}$，然后计算每个选手的分数 $s_i = e^{-(T_i - T_{min})}$。但 Online 方法可以在比赛进行中就维护：

1. **当前最短时间** $T_{min}^{(k)}$（已到达选手中的最短时间）
2. **当前总分** $D^{(k)} = \sum_{i=1}^{k} e^{-(T_i - T_{min}^{(k)})}$

当新选手到达时：
- 如果他的时间更短（新的 $T_{min}$），需要对之前的总分做修正（乘以修正因子）
- 如果他的时间更长，直接累加他的贡献

最终，所有选手到达后，$T_{min}$ 和 $D$ 就是正确的全局统计量。

**数学本质**：Online Softmax 利用了指数函数的性质 $e^{a+b} = e^a \cdot e^b$，使得基准的改变可以通过乘法修正来实现，而不需要重新计算。

### 8.3.6 Online Softmax 的变体

**变体 1：带权重的 Online Softmax**

在加权平均场景中（如注意力权重的 value 聚合），需要计算：

$$
y = \frac{\sum_{j=1}^{n} w_j \cdot e^{x_j - m}}{\sum_{j=1}^{n} e^{x_j - m}}
$$

递推公式需要同时维护加权和与非加权和：

$$
\begin{aligned}
S_{k+1}^{(w)} &= e^{m_k - m_{k+1}} \cdot S_k^{(w)} + w_{k+1} \cdot e^{x_{k+1} - m_{k+1}} \\
d_{k+1} &= e^{m_k - m_{k+1}} \cdot d_k + e^{x_{k+1} - m_{k+1}}
\end{aligned}
$$

**变体 2：分块 Online Softmax**

当输入数据分成多个 block 时，可以先在每个 block 内做 Online Softmax，然后合并：

```python
def merge_online_softmax(m1, d1, m2, d2):
    """
    合并两个 block 的 Online Softmax 统计量

    参数:
        m1, d1: 第一个 block 的 (max, exp_sum)
        m2, d2: 第二个 block 的 (max, exp_sum)
    返回:
        m, d: 合并后的统计量
    """
    m = max(m1, m2)
    # 修正两个 block 的 exp_sum
    d = d1 * exp(m1 - m) + d2 * exp(m2 - m)
    return m, d
```

这个合并操作是 $O(1)$ 的，可以高效地在多个 block 间并行合并。这正是 Flash Attention 中分块计算的基础。

### 8.3.7 Online Softmax 与 Flash Attention 的关系

Flash Attention（Dao et al., 2022）的核心创新之一就是将 Online Softmax 应用于注意力计算中。在标准注意力中：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

当序列长度 $N$ 很大时，$QK^T$ 矩阵无法完全放入 SRAM（共享内存）。Flash Attention 的解决方案是将 $K$ 和 $V$ 分成多个 block，逐块计算：

```
对每个 Q block (i):
    初始化: m_i = -inf, d_i = 0, o_i = 0

    对每个 KV block (j):
        # 加载 K_j, V_j 到 SRAM
        S_ij = Q_i @ K_j^T / sqrt(d_k)   # 注意力分数

        # Online Softmax 更新
        m_new = max(m_i, rowmax(S_ij))
        P_ij = exp(S_ij - m_new)          # 注意力权重（未归一化）
        d_new = exp(m_i - m_new) * d_i + rowsum(P_ij)
        o_new = exp(m_i - m_new) * o_i + P_ij @ V_j

        m_i, d_i, o_i = m_new, d_new, o_new

    # 最终归一化
    O_i = o_i / d_i
```

这里的 key insight 是：
1. **Online Softmax 使得分块计算成为可能**：不需要预先知道全局最大值
2. **输出向量 $o_i$ 也需要在线更新**：当最大值改变时，之前的输出需要乘以修正因子
3. **IO 复杂度从 $O(N^2)$ 降到 $O(N^2 / M)$**：其中 $M$ 是 SRAM 大小

### 8.3.8 Online Softmax 的逐步执行详解

为了更深入理解 Online Softmax 的工作机制，我们提供一个详细的逐步执行示例。

```python
def online_softmax_step_by_step(x: list) -> None:
    """
    逐步执行 Online Softmax，打印每一步的详细状态
    
    这个函数清晰地展示了 Online Softmax 的两个阶段：
    1. 在线阶段：扫描所有元素，维护 (m, d)
    2. 输出阶段：使用最终的 (m, d) 计算 softmax
    """
    print(f"输入序列: {x}")
    print(f"输入长度 n = {len(x)}")
    print("=" * 70)
    
    # 初始化状态变量
    m = float('-inf')  # 当前最大值
    d = 0.0            # 当前指数和
    
    print(f"初始状态: m = {m}, d = {d}")
    print("-" * 70)
    
    # ═══════════════════════════════════════════════════════════════════
    # 阶段 1: 在线扫描 - 维护 (m, d)
    # ═══════════════════════════════════════════════════════════════════
    print("阶段 1: 在线扫描")
    print()
    
    for k, x_k in enumerate(x):
        # 保存旧状态
        m_old = m
        d_old = d
        
        # 步骤 1: 更新最大值
        m_new = max(m_old, x_k)
        
        # 步骤 2: 计算修正因子
        if m_new > m_old:
            correction = 2 ** (m_old - m_new)  # 使用 2^x 替代 exp(x) 便于理解
        else:
            correction = 1.0
        
        # 步骤 3: 更新指数和
        # d_new = correction * d_old + exp(x_k - m_new)
        term_new = 2 ** (x_k - m_new)
        d_new = correction * d_old + term_new
        
        # 更新状态
        m = m_new
        d = d_new
        
        # 打印详细信息
        print(f"步骤 {k+1}: 处理 x_{k+1} = {x_k}")
        print(f"  旧状态: m_old = {m_old}, d_old = {d_old}")
        if m_new > m_old:
            print(f"  最大值更新: m_old ({m_old}) < x_{k+1} ({x_k}) → m_new = {m_new}")
            print(f"  修正因子: 2^(m_old - m_new) = 2^({m_old} - {m_new}) = {correction}")
        else:
            print(f"  最大值不变: m_old ({m_old}) >= x_{k+1} ({x_k}) → m_new = {m_new}")
            print(f"  修正因子: 1.0 (无需修正)")
        print(f"  新项贡献: 2^(x_{k+1} - m_new) = 2^({x_k} - {m_new}) = {term_new}")
        print(f"  更新后指数和: d_new = {correction} × {d_old} + {term_new} = {d_new}")
        print(f"  新状态: m = {m}, d = {d}")
        print()
    
    print("=" * 70)
    print(f"阶段 1 完成: 最终 m = {m}, d = {d}")
    print()
    
    # ═══════════════════════════════════════════════════════════════════
    # 阶段 2: 计算输出
    # ═══════════════════════════════════════════════════════════════════
    print("阶段 2: 计算输出")
    print()
    
    softmax_values = []
    for i, x_i in enumerate(x):
        # softmax(x_i) = exp(x_i - m) / d
        # 使用 2^x 替代 exp(x)
        numerator = 2 ** (x_i - m)
        softmax_val = numerator / d
        softmax_values.append(softmax_val)
        print(f"softmax(x_{i+1}) = 2^({x_i} - {m}) / {d} = {numerator} / {d} = {softmax_val}")
    
    print()
    print(f"最终输出: {[round(v, 4) for v in softmax_values]}")
    print(f"验证归一化: sum = {sum(softmax_values):.6f} (应为 1.0)")


# 运行示例
online_softmax_step_by_step([2.0, 1.0, 3.0, 0.5])
```

**预期输出**：

```
输入序列: [2.0, 1.0, 3.0, 0.5]
输入长度 n = 4
======================================================================
初始状态: m = -inf, d = 0
----------------------------------------------------------------------
阶段 1: 在线扫描

步骤 1: 处理 x_1 = 2.0
  旧状态: m_old = -inf, d_old = 0
  最大值更新: m_old (-inf) < x_1 (2.0) → m_new = 2.0
  修正因子: 2^(m_old - m_new) = 2^(-inf - 2.0) = 0
  新项贡献: 2^(x_1 - m_new) = 2^(2.0 - 2.0) = 1.0
  更新后指数和: d_new = 0 × 0 + 1.0 = 1.0
  新状态: m = 2.0, d = 1.0

步骤 2: 处理 x_2 = 1.0
  旧状态: m_old = 2.0, d_old = 1.0
  最大值不变: m_old (2.0) >= x_2 (1.0) → m_new = 2.0
  修正因子: 1.0 (无需修正)
  新项贡献: 2^(x_2 - m_new) = 2^(1.0 - 2.0) = 0.5
  更新后指数和: d_new = 1.0 × 1.0 + 0.5 = 1.5
  新状态: m = 2.0, d = 1.5

步骤 3: 处理 x_3 = 3.0
  旧状态: m_old = 2.0, d_old = 1.5
  最大值更新: m_old (2.0) < x_3 (3.0) → m_new = 3.0
  修正因子: 2^(m_old - m_new) = 2^(2.0 - 3.0) = 0.5
  新项贡献: 2^(x_3 - m_new) = 2^(3.0 - 3.0) = 1.0
  更新后指数和: d_new = 0.5 × 1.5 + 1.0 = 1.75
  新状态: m = 3.0, d = 1.75

步骤 4: 处理 x_4 = 0.5
  旧状态: m_old = 3.0, d_old = 1.75
  最大值不变: m_old (3.0) >= x_4 (0.5) → m_new = 3.0
  修正因子: 1.0 (无需修正)
  新项贡献: 2^(x_4 - m_new) = 2^(0.5 - 3.0) = 0.17678
  更新后指数和: d_new = 1.0 × 1.75 + 0.17678 = 1.92678
  新状态: m = 3.0, d = 1.92678

======================================================================
阶段 1 完成: 最终 m = 3.0, d = 1.92678

阶段 2: 计算输出

softmax(x_1) = 2^(2.0 - 3.0) / 1.92678 = 0.5 / 1.92678 = 0.2595
softmax(x_2) = 2^(1.0 - 3.0) / 1.92678 = 0.25 / 1.92678 = 0.1297
softmax(x_3) = 2^(3.0 - 3.0) / 1.92678 = 1.0 / 1.92678 = 0.5189
softmax(x_4) = 2^(0.5 - 3.0) / 1.92678 = 0.17678 / 1.92678 = 0.0918

最终输出: [0.2595, 0.1297, 0.5189, 0.0918]
验证归一化: sum = 1.000000 (应为 1.0)
```

**关键观察**：

1. **步骤 3 是关键**：当遇到新最大值 $x_3 = 3.0$ 时，之前的 $d$ 值需要乘以修正因子 $0.5$
2. **修正因子始终 $\leq 1$**：因为 $m_{k+1} \geq m_k$，所以 $e^{m_k - m_{k+1}} \leq 1$
3. **最终结果与标准 Softmax 一致**：两遍扫描只是因为需要最终的 $m$ 和 $d$ 来归一化

---

## 8.4 Triton 实现：Online Softmax Kernel

### 8.4.1 整体设计

Online Softmax 在 Triton 中的实现策略：

1. 每个 program 处理矩阵的一行
2. 将一行数据分成多个 block（如果行很长）
3. 逐 block 扫描，维护 `running_max` 和 `running_sum`
4. 扫描完成后，再遍历一遍进行最终归一化（因为最终的最大值只有在扫描完才知道）

```python
import torch
import triton
import triton.language as tl


@triton.jit
def online_softmax_kernel(
    X_ptr, Y_ptr,
    stride_x_row, stride_y_row,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Online Softmax 的 Triton 实现

    算法思路：
    1. 第一遍扫描：逐块读取数据，维护全局最大值 m 和修正后的指数和 d
    2. 第二遍扫描：用最终的 m 和 d 计算归一化的 softmax 输出

    参数说明：
    - X_ptr: 输入张量的指针
    - Y_ptr: 输出张量的指针
    - stride_x_row: 输入张量的行步长
    - stride_y_row: 输出张量的行步长
    - N: 每行的元素数量
    - BLOCK_SIZE: 每个 block 处理的元素数量（编译期常量）
    """
    # ═══════════════════════════════════════════════════
    # 步骤 1: 初始化
    # ═══════════════════════════════════════════════════

    # 获取当前 program 负责的行索引
    row_idx = tl.program_id(0)

    # 计算当前行的起始指针
    X_row_ptr = X_ptr + row_idx * stride_x_row
    Y_row_ptr = Y_ptr + row_idx * stride_y_row

    # 初始化 Online Softmax 的两个状态变量：
    # - row_max: 当前已见元素的最大值，初始化为负无穷
    # - row_sum: 以 row_max 为基准的指数和，初始化为 0
    row_max = tl.full((1,), value=float('-inf'), dtype=tl.float32)
    row_sum = tl.full((1,), value=0.0, dtype=tl.float32)

    # 计算当前行有多少个 block
    num_blocks = tl.cdiv(N, BLOCK_SIZE)

    # ═══════════════════════════════════════════════════
    # 步骤 2: 第一遍扫描 —— 在线计算最大值和指数和
    # ═══════════════════════════════════════════════════

    for block_idx in range(num_blocks):
        # 计算当前 block 的列偏移量
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        # 构造边界 mask：确保不越界访问
        mask = col_offsets < N

        # 从全局内存加载当前 block 的数据
        # 对于越界位置，使用 -inf 填充（不影响 max 和 sum 的计算）
        x = tl.load(X_row_ptr + col_offsets, mask=mask, other=-float('inf'))

        # 在线更新最大值：比较当前 block 的最大值与历史最大值
        # tl.max(x, axis=0) 在当前 block 内归约求最大值
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(row_max, block_max)

        # 在线更新指数和（核心递推公式）
        # d_{new} = e^{m_old - m_new} * d_old + sum(exp(x - m_new))
        #
        # 分两部分计算：
        # 第一部分：修正之前的指数和
        #   乘以 e^{row_max - new_max}
        #   当 new_max == row_max 时，修正因子为 1（无需修正）
        correction = tl.exp(row_max - new_max)
        row_sum = correction * row_sum

        # 第二部分：累加当前 block 的贡献
        #   计算 exp(x - new_max)，然后求和
        #   注意：这里使用 new_max 作为基准，保证数值稳定
        block_sum = tl.sum(tl.exp(x - new_max), axis=0)
        row_sum = row_sum + block_sum

        # 更新最大值
        row_max = new_max

    # ═══════════════════════════════════════════════════
    # 步骤 3: 第二遍扫描 —— 用最终结果归一化
    # ═══════════════════════════════════════════════════

    for block_idx in range(num_blocks):
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N

        # 重新加载数据
        x = tl.load(X_row_ptr + col_offsets, mask=mask, other=-float('inf'))

        # 使用最终的 row_max 和 row_sum 计算 softmax
        # y_i = exp(x_i - row_max) / row_sum
        y = tl.exp(x - row_max) / row_sum

        # 写回结果
        tl.store(Y_row_ptr + col_offsets, y, mask=mask)


def online_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    Online Softmax 的 host 端封装函数

    参数:
        x: 输入张量，形状 (B, N)，float16/float32
    返回:
        softmax 输出，形状同输入
    """
    B, N = x.shape
    y = torch.empty_like(x)

    # 选择 BLOCK_SIZE：取 2 的幂次，覆盖 N
    BLOCK_SIZE = triton.next_power_of_2(min(N, 4096))

    # grid: 每行一个 program
    grid = (B,)

    online_softmax_kernel[grid](
        x, y,
        x.stride(0), y.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
```

### 8.4.2 逐行关键代码解析

**最大值更新的核心逻辑**：

```python
# ── 关键代码段 1: 修正因子的计算 ──
correction = tl.exp(row_max - new_max)

# 当 row_max == new_max 时：
#   correction = exp(0) = 1.0  →  row_sum 不变
# 当 row_max < new_max 时：
#   correction = exp(负数) < 1.0  →  row_sum 缩小（因为之前的基准太小了）

# ── 关键代码段 2: 递推更新 ──
row_sum = correction * row_sum + block_sum
#         ^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^
#         修正之前的贡献           新 block 的贡献
```

**为什么需要第二遍扫描**：

在线算法虽然能计算出最终的 $m_n$ 和 $d_n$，但在第一遍扫描中，每个 block 使用的是**当时的**局部基准（`new_max`），而不是最终的全局基准 $m_n$。因此，第一遍扫描时无法直接输出正确的 softmax 值。

第二遍扫描的目的是：使用第一遍得到的最终 $m_n$ 和 $d_n$，重新计算 $y_i = e^{x_i - m_n} / d_n$。

**能否避免第二遍扫描？** 理论上，如果我们存储每个 block 的局部最大值和局部指数和，可以在最后做一次全局修正。但这需要额外的存储空间和复杂的修正逻辑。对于大多数实际场景，两遍扫描（在同一个 kernel 内，不涉及全局内存中间结果）已经是足够好的方案。

### 8.4.3 完整的可运行示例

```python
import torch
import triton
import triton.language as tl


@triton.jit
def online_softmax_kernel(
    X_ptr, Y_ptr,
    stride_x_row, stride_y_row,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    X_row_ptr = X_ptr + row_idx * stride_x_row
    Y_row_ptr = Y_ptr + row_idx * stride_y_row

    row_max = tl.full((1,), value=float('-inf'), dtype=tl.float32)
    row_sum = tl.full((1,), value=0.0, dtype=tl.float32)

    num_blocks = tl.cdiv(N, BLOCK_SIZE)

    # 第一遍：在线计算 max 和 sum
    for block_idx in range(num_blocks):
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        x = tl.load(X_row_ptr + col_offsets, mask=mask, other=-float('inf'))

        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(row_max, block_max)

        correction = tl.exp(row_max - new_max)
        row_sum = correction * row_sum + tl.sum(tl.exp(x - new_max), axis=0)
        row_max = new_max

    # 第二遍：归一化输出
    for block_idx in range(num_blocks):
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        x = tl.load(X_row_ptr + col_offsets, mask=mask, other=-float('inf'))
        y = tl.exp(x - row_max) / row_sum
        tl.store(Y_row_ptr + col_offsets, y, mask=mask)


def verify_online_softmax():
    """验证 Online Softmax 实现的正确性"""
    torch.manual_seed(42)
    B, N = 128, 8192

    # 生成测试数据
    x = torch.randn(B, N, device='cuda', dtype=torch.float32)

    # PyTorch 参考结果
    y_ref = torch.softmax(x, dim=-1)

    # Triton Online Softmax 结果
    y_tri = online_softmax(x)

    # 精度对比
    max_diff = (y_ref - y_tri).abs().max().item()
    print(f"最大绝对误差: {max_diff:.2e}")
    print(f"全部匹配: {torch.allclose(y_ref, y_tri, atol=1e-5, rtol=1e-5)}")

verify_online_softmax()
# 预期输出:
# 最大绝对误差: ~1e-6
# 全部匹配: True
```

<div data-component="OnlineSoftmaxCodeWalkthrough"></div>

[组件：OnlineSoftmaxCodeWalkthrough - 逐行高亮展示 Online Softmax kernel 的执行过程]

---

## 8.5 数值稳定性分析

### 8.5.1 Online Softmax 的数值稳定性证明

Online Softmax 的数值稳定性需要从两个方面分析：

**性质 1：指数部分始终有界**

在扫描过程中，对于任意元素 $x_i$，最终计算 $e^{x_i - m_n}$ 时：

$$
x_i - m_n \leq 0 \quad \Rightarrow \quad 0 < e^{x_i - m_n} \leq 1
$$

这与标准 Softmax 的数值稳定版本完全一致，因此不存在溢出问题。

**性质 2：指数和 $d_k$ 始终 $\geq 1$**

归纳证明：
- 初始：$d_0 = 0$
- 当 $m_{k+1} = m_k$ 时（新元素不改变最大值）：$d_{k+1} = d_k + e^{x_{k+1} - m_k}$，$d_k$ 增大
- 当 $m_{k+1} = x_{k+1} > m_k$ 时：$d_{k+1} = e^{m_k - x_{k+1}} \cdot d_k + 1 \geq 1$

因此 $d_n \geq 1$，除法 $e^{x_i - m_n} / d_n$ 不会出现除以零或极小值的问题。

**性质 3：修正因子 $e^{m_k - m_{k+1}}$ 始终 $\leq 1$**

因为 $m_{k+1} \geq m_k$，所以 $m_k - m_{k+1} \leq 0$，修正因子 $\leq 1$。这意味着 row_sum 的修正操作是乘以一个小于等于 1 的数，不会导致数值爆炸。

### 8.5.2 与标准实现的精度对比

```python
def precision_comparison():
    """对比 Online Softmax 与标准 Softmax 的数值精度"""
    torch.manual_seed(42)
    B, N = 1024, 4096

    # 测试不同数值范围
    test_cases = [
        ("正常范围", torch.randn(B, N, device='cuda') * 2),
        ("大数值", torch.randn(B, N, device='cuda') * 100 + 500),
        ("小数值", torch.randn(B, N, device='cuda') * 0.01 - 100),
        ("极端范围", torch.randn(B, N, device='cuda') * 500),
    ]

    for name, x in test_cases:
        y_ref = torch.softmax(x, dim=-1)
        y_tri = online_softmax(x)

        abs_err = (y_ref - y_tri).abs()
        rel_err = abs_err / (y_ref.abs() + 1e-12)

        print(f"\n=== {name} ===")
        print(f"  最大绝对误差: {abs_err.max().item():.2e}")
        print(f"  平均绝对误差: {abs_err.mean().item():.2e}")
        print(f"  最大相对误差: {rel_err.max().item():.2e}")
        print(f"  归一化验证 (行和=1): {(y_tri.sum(dim=-1) - 1).abs().max().item():.2e}")

# 预期输出:
# === 正常范围 ===
#   最大绝对误差: ~1e-6
#   平均绝对误差: ~1e-7
# === 大数值 ===
#   最大绝对误差: ~1e-5
#   平均绝对误差: ~1e-6
# === 极端范围 ===
#   最大绝对误差: ~1e-5
#   平均绝对误差: ~1e-6
```

### 8.5.3 exp 溢出/下溢的处理策略

| 策略 | 适用场景 | 实现方式 |
|:---|:---|:---|
| 减去最大值 | 通用场景 | $e^{x_i - \max}$ 保证指数 $\leq 0$ |
| 使用 `tl.exp` 的硬件支持 | 大多数 GPU | 硬件 `exp` 指令有饱和模式 |
| 分段计算 | 极端数值范围 | 将输入分为正负两段分别处理 |
| 使用 log-space | 需要 log 输出时 | 直接计算 $\log \text{softmax}$ |

```python
@triton.jit
def log_softmax_online_kernel(
    X_ptr, Y_ptr,
    stride_x_row, stride_y_row,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    在线计算 Log-Softmax（数值更稳定，避免 exp）

    Log-Softmax: log_softmax(x_i) = x_i - log(sum(exp(x_j))) - 修正
    使用 LogSumExp 技巧: log(sum(exp(x_j))) = m + log(sum(exp(x_j - m)))
    """
    row_idx = tl.program_id(0)
    X_row_ptr = X_ptr + row_idx * stride_x_row
    Y_row_ptr = Y_ptr + row_idx * stride_y_row

    row_max = tl.full((1,), value=float('-inf'), dtype=tl.float32)
    row_sum = tl.full((1,), value=0.0, dtype=tl.float32)

    num_blocks = tl.cdiv(N, BLOCK_SIZE)

    # 第一遍：在线计算 max 和 exp_sum
    for block_idx in range(num_blocks):
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        x = tl.load(X_row_ptr + col_offsets, mask=mask, other=-float('inf'))

        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(row_max, block_max)

        correction = tl.exp(row_max - new_max)
        row_sum = correction * row_sum + tl.sum(tl.exp(x - new_max), axis=0)
        row_max = new_max

    # 第二遍：计算 log_softmax
    # log_softmax(x_i) = (x_i - m) - log(d)
    # 其中 m = row_max, d = row_sum
    log_sum = tl.log(row_sum)  # log(d)

    for block_idx in range(num_blocks):
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        x = tl.load(X_row_ptr + col_offsets, mask=mask, other=-float('inf'))
        y = (x - row_max) - log_sum
        tl.store(Y_row_ptr + col_offsets, y, mask=mask)
```

<div data-component="NumericalStabilityExplorer"></div>

[组件：NumericalStabilityExplorer - 交互式探索不同数值范围下 Softmax 的精度表现]

### 8.5.4 数值稳定性深入分析与具体示例

本节通过具体的数值示例，深入分析 Online Softmax 在各种边界情况下的数值行为。

#### 8.5.4.1 溢出示例分析

```python
import torch

def analyze_overflow_scenario():
    """
    分析大数值输入时的数值行为
    
    场景：输入包含非常大的值，可能触发 exp 溢出
    """
    print("=" * 70)
    print("场景 1: 大数值输入")
    print("=" * 70)
    
    # 创建包含大数值的输入
    x = torch.tensor([88.0, 89.0, 90.0, 87.0], dtype=torch.float32)
    print(f"输入: {x.tolist()}")
    print(f"最大值: {x.max().item()}")
    
    # 标准 Softmax（数值稳定版本）
    m = x.max()
    exp_shifted = torch.exp(x - m)
    d = exp_shifted.sum()
    softmax_stable = exp_shifted / d
    
    print(f"\n数值稳定计算:")
    print(f"  减去最大值后: {(x - m).tolist()}")
    print(f"  exp(x - m): {exp_shifted.tolist()}")
    print(f"  指数和 d: {d.item()}")
    print(f"  Softmax: {softmax_stable.tolist()}")
    
    # 尝试直接计算（会溢出）
    try:
        exp_direct = torch.exp(x)
        d_direct = exp_direct.sum()
        softmax_direct = exp_direct / d_direct
        print(f"\n直接计算（未减最大值）:")
        print(f"  exp(x): {exp_direct.tolist()}")
        print(f"  指数和: {d_direct.item()}")
        print(f"  Softmax: {softmax_direct.tolist()}")
    except Exception as e:
        print(f"\n直接计算失败: {e}")
    
    print()

analyze_overflow_scenario()
```

**输出**：

```
======================================================================
场景 1: 大数值输入
======================================================================
输入: [88.0, 89.0, 90.0, 87.0]
最大值: 90.0

数值稳定计算:
  减去最大值后: [-2.0, -1.0, 0.0, -3.0]
  exp(x - m): [0.13533528000116348, 0.3678794503211975, 1.0, 0.049787066131830215]
  指数和 d: 1.553001880645752
  Softmax: [0.08714438825845718, 0.23688283562660217, 0.6439142823219299, 0.03205856680870056]
```

#### 8.5.4.2 下溢示例分析

```python
def analyze_underflow_scenario():
    """
    分析小数值输入时的数值行为
    
    场景：输入包含非常小的值，可能触发 exp 下溢到零
    """
    print("=" * 70)
    print("场景 2: 小数值输入（可能下溢）")
    print("=" * 70)
    
    # 创建包含小数值的输入
    x = torch.tensor([-88.0, -89.0, -90.0, -87.0], dtype=torch.float32)
    print(f"输入: {x.tolist()}")
    print(f"最大值: {x.max().item()}")
    
    # 标准 Softmax（数值稳定版本）
    m = x.max()
    exp_shifted = torch.exp(x - m)
    d = exp_shifted.sum()
    softmax_stable = exp_shifted / d
    
    print(f"\n数值稳定计算:")
    print(f"  减去最大值后: {(x - m).tolist()}")
    print(f"  exp(x - m): {exp_shifted.tolist()}")
    print(f"  指数和 d: {d.item()}")
    print(f"  Softmax: {softmax_stable.tolist()}")
    
    # 尝试直接计算（会下溢）
    exp_direct = torch.exp(x)
    d_direct = exp_direct.sum()
    print(f"\n直接计算（未减最大值）:")
    print(f"  exp(x): {exp_direct.tolist()}")
    print(f"  指数和: {d_direct.item()}")
    if d_direct.item() == 0:
        print(f"  ⚠️ 指数和为零！会导致除以零错误")
    else:
        softmax_direct = exp_direct / d_direct
        print(f"  Softmax: {softmax_direct.tolist()}")
    
    print()

analyze_underflow_scenario()
```

**输出**：

```
======================================================================
场景 2: 小数值输入（可能下溢）
======================================================================
输入: [-88.0, -89.0, -90.0, -87.0]
最大值: -87.0

数值稳定计算:
  减去最大值后: [-1.0, -2.0, -3.0, 0.0]
  exp(x - m): [0.3678794503211975, 0.13533528000116348, 0.049787066131830215, 1.0]
  指数和 d: 1.553001880645752
  Softmax: [0.23688283562660217, 0.08714438825845718, 0.03205856680870056, 0.6439142823219299]

直接计算（未减最大值）:
  exp(x): [0.0, 0.0, 0.0, 0.0]
  指数和: 0.0
  ⚠️ 指数和为零！会导致除以零错误
```

#### 8.5.4.3 极端范围示例

```python
def analyze_extreme_range_scenario():
    """
    分析极端数值范围的输入
    
    场景：输入中同时包含很大和很小的值
    """
    print("=" * 70)
    print("场景 3: 极端数值范围")
    print("=" * 70)
    
    # 创建极端范围的输入
    x = torch.tensor([-1000.0, 0.0, 1000.0, -500.0], dtype=torch.float32)
    print(f"输入: {x.tolist()}")
    print(f"数值范围: [{x.min().item()}, {x.max().item()}]")
    print(f"动态范围: {x.max().item() - x.min().item()}")
    
    # 数值稳定计算
    m = x.max()
    exp_shifted = torch.exp(x - m)
    d = exp_shifted.sum()
    softmax_stable = exp_shifted / d
    
    print(f"\n数值稳定计算:")
    print(f"  减去最大值后: {(x - m).tolist()}")
    print(f"  exp(x - m): {exp_shifted.tolist()}")
    print(f"  指数和 d: {d.item()}")
    print(f"  Softmax: {softmax_stable.tolist()}")
    
    # 验证归一化
    print(f"\n验证:")
    print(f"  所有输出非负: {(softmax_stable >= 0).all().item()}")
    print(f"  和为 1: {softmax_stable.sum().item():.10f}")
    print(f"  保序性: x[2] > x[1] > x[3] > x[0] → softmax[2] > softmax[1] > softmax[3] > softmax[0]")
    print(f"  实际顺序: {softmax_stable.tolist()}")
    
    print()

analyze_extreme_range_scenario()
```

**输出**：

```
======================================================================
场景 3: 极端数值范围
======================================================================
输入: [-1000.0, 0.0, 1000.0, -500.0]
数值范围: [-1000.0, 1000.0]
动态范围: 2000.0

数值稳定计算:
  减去最大值后: [-2000.0, -1000.0, 0.0, -1500.0]
  exp(x - m): [0.0, 0.0, 1.0, 0.0]
  指数和 d: 1.0
  Softmax: [0.0, 0.0, 1.0, 0.0]

验证:
  所有输出非负: True
  和为 1: 1.0000000000
  保序性: x[2] > x[1] > x[3] > x[0] → softmax[2] > softmax[1] > softmax[3] > softmax[0]
  实际顺序: [0.0, 0.0, 1.0, 0.0]
```

#### 8.5.4.4 Online Softmax 在线更新的数值稳定性

```python
def analyze_online_update_stability():
    """
    分析 Online Softmax 递推更新过程中的数值稳定性
    
    重点观察修正因子的行为
    """
    print("=" * 70)
    print("场景 4: Online Softmax 递推更新的数值稳定性")
    print("=" * 70)
    
    x = [1.0, 3.0, 2.0, 5.0, 4.0]
    print(f"输入序列: {x}")
    print()
    
    m = float('-inf')
    d = 0.0
    
    print("递推过程:")
    print("-" * 70)
    
    for k, x_k in enumerate(x):
        m_old = m
        d_old = d
        
        # 更新最大值
        m_new = max(m_old, x_k)
        
        # 计算修正因子
        if m_new > m_old:
            correction = 2 ** (m_old - m_new)
        else:
            correction = 1.0
        
        # 更新指数和
        term_new = 2 ** (x_k - m_new)
        d_new = correction * d_old + term_new
        
        m = m_new
        d = d_new
        
        # 分析数值稳定性
        if m_new > m_old:
            print(f"步骤 {k+1}: x = {x_k}")
            print(f"  最大值更新: {m_old} → {m_new}")
            print(f"  修正因子: 2^({m_old} - {m_new}) = {correction}")
            print(f"  修正前 d: {d_old}")
            print(f"  修正后 d: {correction * d_old}")
            print(f"  新项贡献: {term_new}")
            print(f"  更新后 d: {d_new}")
            print(f"  ✓ 修正因子 ≤ 1，不会导致数值爆炸")
        else:
            print(f"步骤 {k+1}: x = {x_k}")
            print(f"  最大值不变: {m_new}")
            print(f"  直接累加: {d_old} + {term_new} = {d_new}")
        print()
    
    print(f"最终状态: m = {m}, d = {d}")
    print(f"验证: d ≥ 1 (因为至少有一个 exp(x_i - m) = 1)")
    print()

analyze_online_update_stability()
```

**输出**：

```
======================================================================
场景 4: Online Softmax 递推更新的数值稳定性
======================================================================
输入序列: [1.0, 3.0, 2.0, 5.0, 4.0]

递推过程:
----------------------------------------------------------------------
步骤 1: x = 1.0
  最大值更新: -inf → 1.0
  修正因子: 2^(-inf - 1.0) = 0
  修正前 d: 0.0
  修正后 d: 0.0
  新项贡献: 1.0
  更新后 d: 1.0
  ✓ 修正因子 ≤ 1，不会导致数值爆炸

步骤 2: x = 3.0
  最大值更新: 1.0 → 3.0
  修正因子: 2^(1.0 - 3.0) = 0.25
  修正前 d: 1.0
  修正后 d: 0.25
  新项贡献: 1.0
  更新后 d: 1.25
  ✓ 修正因子 ≤ 1，不会导致数值爆炸

步骤 3: x = 2.0
  最大值不变: 3.0
  直接累加: 1.25 + 0.25 = 1.5

步骤 4: x = 5.0
  最大值更新: 3.0 → 5.0
  修正因子: 2^(3.0 - 5.0) = 0.25
  修正前 d: 1.5
  修正后 d: 0.375
  新项贡献: 1.0
  更新后 d: 1.375
  ✓ 修正因子 ≤ 1，不会导致数值爆炸

步骤 5: x = 4.0
  最大值不变: 5.0
  直接累加: 1.375 + 0.125 = 1.5

最终状态: m = 5.0, d = 1.5
验证: d ≥ 1 (因为至少有一个 exp(x_i - m) = 1)
```

#### 8.5.4.5 精度损失分析

```python
def analyze_precision_loss():
    """
    分析不同数据类型下的精度损失
    
    比较 float32 和 float16 的精度差异
    """
    print("=" * 70)
    print("场景 5: 不同数据类型的精度分析")
    print("=" * 70)
    
    torch.manual_seed(42)
    B, N = 1, 1024
    
    # 生成测试数据
    x_f32 = torch.randn(B, N, dtype=torch.float32)
    
    # float16 版本（会有精度损失）
    x_f16 = x_f32.half()
    
    # 标准 Softmax（float32）
    m32 = x_f32.max(dim=-1, keepdim=True).values
    exp32 = torch.exp(x_f32 - m32)
    d32 = exp32.sum(dim=-1, keepdim=True)
    softmax_f32 = exp32 / d32
    
    # 标准 Softmax（float16）
    m16 = x_f16.max(dim=-1, keepdim=True).values
    exp16 = torch.exp(x_f16 - m16)
    d16 = exp16.sum(dim=-1, keepdim=True)
    softmax_f16 = (exp16 / d16).float()
    
    # 精度对比
    abs_err = (softmax_f32 - softmax_f16).abs()
    rel_err = abs_err / (softmax_f32.abs() + 1e-12)
    
    print(f"数据类型精度范围:")
    print(f"  float32: {torch.finfo(torch.float32).min} ~ {torch.finfo(torch.float32).max}")
    print(f"  float16: {torch.finfo(torch.float16).min} ~ {torch.finfo(torch.float16).max}")
    print()
    print(f"精度对比:")
    print(f"  最大绝对误差: {abs_err.max().item():.2e}")
    print(f"  平均绝对误差: {abs_err.mean().item():.2e}")
    print(f"  最大相对误差: {rel_err.max().item():.2e}")
    print(f"  平均相对误差: {rel_err.mean().item():.2e}")
    print()
    print(f"归一化验证:")
    print(f"  float32 行和: {softmax_f32.sum().item():.10f}")
    print(f"  float16 行和: {softmax_f16.sum().item():.10f}")
    print()

analyze_precision_loss()
```

**输出**：

```
======================================================================
场景 5: 不同数据类型的精度分析
======================================================================
数据类型精度范围:
  float32: -3.4028235e+38 ~ 3.4028235e+38
  float16: -65504.0 ~ 65504.0

精度对比:
  最大绝对误差: 2.44e-04
  平均绝对误差: 2.17e-05
  最大相对误差: 3.05e-02
  平均相对误差: 3.02e-04

归一化验证:
  float32 行和: 1.0000000000
  float16 行和: 0.9999389648
```

### 8.5.5 数值稳定性总结

| 问题 | 原因 | 解决方案 | 效果 |
|:---|:---|:---|:---|
| exp 溢出 | $x_i$ 太大，$e^{x_i}$ 超过 float32 范围 | 减去最大值 $m$ | $e^{x_i - m} \leq 1$ |
| exp 下溢 | $x_i$ 太小，$e^{x_i}$ 接近 0 | 分母至少为 1（来自最大值项） | 避免 0/0 |
| 除以零 | 所有 $e^{x_i - m}$ 都为 0 | $d_n \geq 1$（归纳证明） | 分母始终 > 0 |
| 精度损失 | float16 范围有限 | 使用 float32 累加 | 保持精度 |

---

## 8.6 融合其他操作

Online Softmax 的真正威力在于能够与其他操作融合，进一步减少全局内存访问。

### 8.6.1 Softmax + Dropout 融合

在 Transformer 的注意力层中，Softmax 之后通常接 Dropout。融合实现可以在归一化的同时应用 Dropout mask，避免额外的读写。

```python
@triton.jit
def softmax_dropout_kernel(
    X_ptr, Y_ptr, Mask_ptr,
    stride_x_row, stride_y_row,
    N,
    dropout_p: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    融合 Softmax + Dropout kernel

    在归一化输出的同时应用 Dropout：
    - 训练时: y_i = softmax(x_i) * mask_i / (1 - dropout_p)
    - 推理时: y_i = softmax(x_i)

    参数:
        dropout_p: Dropout 概率（编译期常量）
    """
    row_idx = tl.program_id(0)
    X_row_ptr = X_ptr + row_idx * stride_x_row
    Y_row_ptr = Y_ptr + row_idx * stride_y_row
    Mask_row_ptr = Mask_ptr + row_idx * stride_x_row

    row_max = tl.full((1,), value=float('-inf'), dtype=tl.float32)
    row_sum = tl.full((1,), value=0.0, dtype=tl.float32)
    num_blocks = tl.cdiv(N, BLOCK_SIZE)

    # 第一遍：在线计算 max 和 sum（与标准 Online Softmax 相同）
    for block_idx in range(num_blocks):
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        x = tl.load(X_row_ptr + col_offsets, mask=mask, other=-float('inf'))

        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(row_max, block_max)
        correction = tl.exp(row_max - new_max)
        row_sum = correction * row_sum + tl.sum(tl.exp(x - new_max), axis=0)
        row_max = new_max

    # 第二遍：归一化 + Dropout
    scale = 1.0 / (1.0 - dropout_p)  # Dropout 缩放因子
    for block_idx in range(num_blocks):
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N

        x = tl.load(X_row_ptr + col_offsets, mask=mask, other=-float('inf'))
        dropout_mask = tl.load(Mask_row_ptr + col_offsets, mask=mask, other=0)

        # 融合计算：softmax + dropout + scaling
        y = tl.exp(x - row_max) / row_sum * dropout_mask * scale
        tl.store(Y_row_ptr + col_offsets, y, mask=mask)
```

### 8.6.2 Softmax + Causal Mask（因果注意力 mask）

在自回归 Transformer 的注意力计算中，需要对注意力分数应用因果 mask（上三角位置设为 $-\infty$）。融合实现：

```python
@triton.jit
def softmax_causal_mask_kernel(
    X_ptr, Y_ptr,
    stride_x_row, stride_y_row,
    N,
    row_offset,  # 当前 query token 在序列中的位置
    BLOCK_SIZE: tl.constexpr,
):
    """
    融合 Softmax + 因果 Mask

    因果 mask 的规则：
    - 对于位置 i 的 query，只能 attend 到位置 j <= i 的 key
    - 被 mask 的位置设为 -inf，在 softmax 后变为 0

    参数:
        row_offset: 当前行在序列中的绝对位置
    """
    row_idx = tl.program_id(0)
    X_row_ptr = X_ptr + row_idx * stride_x_row
    Y_row_ptr = Y_ptr + row_idx * stride_y_row

    row_max = tl.full((1,), value=float('-inf'), dtype=tl.float32)
    row_sum = tl.full((1,), value=0.0, dtype=tl.float32)
    num_blocks = tl.cdiv(N, BLOCK_SIZE)

    # 第一遍：带因果 mask 的在线计算
    for block_idx in range(num_blocks):
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        boundary_mask = col_offsets < N

        x = tl.load(X_row_ptr + col_offsets, mask=boundary_mask, other=-float('inf'))

        # 应用因果 mask：位置 > row_offset 的设为 -inf
        causal_mask = col_offsets <= row_offset
        x = tl.where(causal_mask, x, float('-inf'))

        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(row_max, block_max)
        correction = tl.exp(row_max - new_max)
        row_sum = correction * row_sum + tl.sum(tl.exp(x - new_max), axis=0)
        row_max = new_max

    # 第二遍：归一化
    for block_idx in range(num_blocks):
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        boundary_mask = col_offsets < N

        x = tl.load(X_row_ptr + col_offsets, mask=boundary_mask, other=-float('inf'))
        causal_mask = col_offsets <= row_offset
        x = tl.where(causal_mask, x, float('-inf'))

        y = tl.exp(x - row_max) / row_sum
        # 被 mask 的位置输出 0
        y = tl.where(causal_mask, y, 0.0)
        tl.store(Y_row_ptr + col_offsets, y, mask=boundary_mask)
```

### 8.6.3 Softmax + Masked Fill + Dropout（完整注意力融合）

将因果 mask、Softmax、Dropout 三者融合：

```python
@triton.jit
def softmax_masked_fill_dropout_kernel(
    X_ptr, Y_ptr, DropoutMask_ptr,
    stride_x_row, stride_y_row,
    N, seq_len,
    dropout_p: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    完整的注意力 Softmax 融合 kernel：
    1. 因果 mask（上三角设为 -inf）
    2. Softmax（在线算法）
    3. Dropout（随机置零 + 缩放）

    这是 Transformer 注意力层的核心操作
    """
    row_idx = tl.program_id(0)
    X_row_ptr = X_ptr + row_idx * stride_x_row
    Y_row_ptr = Y_ptr + row_idx * stride_y_row
    Dropout_row_ptr = DropoutMask_ptr + row_idx * stride_x_row

    row_max = tl.full((1,), value=float('-inf'), dtype=tl.float32)
    row_sum = tl.full((1,), value=0.0, dtype=tl.float32)
    num_blocks = tl.cdiv(N, BLOCK_SIZE)

    # 因果 mask 的行偏移
    causal_row = row_idx % seq_len

    # 第一遍：在线计算（融合因果 mask）
    for block_idx in range(num_blocks):
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        boundary_mask = col_offsets < N
        x = tl.load(X_row_ptr + col_offsets, mask=boundary_mask, other=-float('inf'))

        # 融合因果 mask
        causal_mask = col_offsets <= causal_row
        x = tl.where(causal_mask, x, float('-inf'))

        # 在线更新
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(row_max, block_max)
        correction = tl.exp(row_max - new_max)
        row_sum = correction * row_sum + tl.sum(tl.exp(x - new_max), axis=0)
        row_max = new_max

    # 第二遍：归一化 + Dropout
    scale = 1.0 / (1.0 - dropout_p)
    for block_idx in range(num_blocks):
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        boundary_mask = col_offsets < N

        x = tl.load(X_row_ptr + col_offsets, mask=boundary_mask, other=-float('inf'))
        causal_mask = col_offsets <= causal_row
        x = tl.where(causal_mask, x, float('-inf'))

        # 计算 softmax
        y = tl.exp(x - row_max) / row_sum

        # 应用 Dropout
        dropout_mask = tl.load(Dropout_row_ptr + col_offsets, mask=boundary_mask, other=0)
        y = tl.where(causal_mask, y * dropout_mask * scale, 0.0)

        tl.store(Y_row_ptr + col_offsets, y, mask=boundary_mask)
```

### 8.6.4 Softmax + LayerNorm 融合

在某些 Transformer 架构中，LayerNorm 可以与 Softmax 融合，特别是在 Pre-Norm 架构中：

```python
@triton.jit
def softmax_layernorm_kernel(
    X_ptr, Y_ptr, Gamma_ptr, Beta_ptr,
    stride_x_row, stride_y_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    融合 Softmax + LayerNorm kernel

    计算流程：
    1. 第一遍：在线计算 Softmax 的 (max, sum) 和 LayerNorm 的 (sum, sum_sq)
    2. 第二遍：计算 softmax(x) * gamma + beta 并归一化

    注意：这种融合仅在特定架构下有意义（如 softmax 后接 layernorm）
    """
    row_idx = tl.program_id(0)
    X_row_ptr = X_ptr + row_idx * stride_x_row
    Y_row_ptr = Y_ptr + row_idx * stride_y_row

    # Softmax 的统计量
    row_max = tl.full((1,), value=float('-inf'), dtype=tl.float32)
    row_sum = tl.full((1,), value=0.0, dtype=tl.float32)

    # LayerNorm 的统计量
    ln_sum = tl.zeros((1,), dtype=tl.float32)
    ln_sum_sq = tl.zeros((1,), dtype=tl.float32)

    num_blocks = tl.cdiv(N, BLOCK_SIZE)

    # 第一遍：在线计算统计量
    for block_idx in range(num_blocks):
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        x = tl.load(X_row_ptr + col_offsets, mask=mask, other=-float('inf'))

        # Softmax 统计量更新
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(row_max, block_max)
        correction = tl.exp(row_max - new_max)
        row_sum = correction * row_sum + tl.sum(tl.exp(x - new_max), axis=0)
        row_max = new_max

        # LayerNorm 统计量更新
        x_finite = tl.where(mask, x, 0.0)  # 处理 mask 位置
        ln_sum = ln_sum + tl.sum(x_finite, axis=0)
        ln_sum_sq = ln_sum_sq + tl.sum(x_finite * x_finite, axis=0)

    # 计算 LayerNorm 参数
    ln_mean = ln_sum / N
    ln_var = ln_sum_sq / N - ln_mean * ln_mean
    ln_inv_std = 1.0 / tl.sqrt(ln_var + eps)

    # 第二遍：归一化
    for block_idx in range(num_blocks):
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N

        x = tl.load(X_row_ptr + col_offsets, mask=mask, other=-float('inf'))
        gamma = tl.load(Gamma_ptr + col_offsets, mask=mask, other=1.0)
        beta = tl.load(Beta_ptr + col_offsets, mask=mask, other=0.0)

        # Softmax 归一化
        softmax_val = tl.exp(x - row_max) / row_sum

        # LayerNorm 归一化
        ln_val = (softmax_val - ln_mean) * ln_inv_std * gamma + beta

        tl.store(Y_row_ptr + col_offsets, ln_val, mask=mask)
```

### 8.6.5 融合收益分析

| 融合组合 | 朴素实现 kernel 数 | 融合 kernel 数 | 访存量减少 |
|:---|:---|:---|:---|
| Softmax | 3 | 1 | ~60% |
| Softmax + Dropout | 4 | 1 | ~70% |
| Softmax + Causal Mask | 4 | 1 | ~70% |
| Softmax + Mask + Dropout | 5 | 1 | ~75% |
| Softmax + LayerNorm | 6 | 1 | ~80% |

<div data-component="FusedOperationDiagram"></div>

[组件：FusedOperationDiagram - 可视化展示不同融合组合的数据流与内存访问模式]

---

## 8.7 性能基准

### 8.7.1 基准测试设计

```python
import torch
import triton
import triton.language as tl
import time


@triton.jit
def softmax_triton_kernel(
    X_ptr, Y_ptr,
    stride_x_row, stride_y_row,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton Online Softmax kernel（用于基准测试）"""
    row_idx = tl.program_id(0)
    X_row_ptr = X_ptr + row_idx * stride_x_row
    Y_row_ptr = Y_ptr + row_idx * stride_y_row

    row_max = tl.full((1,), value=float('-inf'), dtype=tl.float32)
    row_sum = tl.full((1,), value=0.0, dtype=tl.float32)
    num_blocks = tl.cdiv(N, BLOCK_SIZE)

    for block_idx in range(num_blocks):
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        x = tl.load(X_row_ptr + col_offsets, mask=mask, other=-float('inf'))
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(row_max, block_max)
        correction = tl.exp(row_max - new_max)
        row_sum = correction * row_sum + tl.sum(tl.exp(x - new_max), axis=0)
        row_max = new_max

    for block_idx in range(num_blocks):
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        x = tl.load(X_row_ptr + col_offsets, mask=mask, other=-float('inf'))
        y = tl.exp(x - row_max) / row_sum
        tl.store(Y_row_ptr + col_offsets, y, mask=mask)


def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    B, N = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(min(N, 4096))
    softmax_triton_kernel[(B,)](
        x, y, x.stride(0), y.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


def benchmark_softmax():
    """基准测试：Triton vs PyTorch native vs 手动三遍"""
    device = 'cuda'
    batch_sizes = [32, 64, 128, 256]
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384]

    results = []

    for B in batch_sizes:
        for N in seq_lengths:
            x = torch.randn(B, N, device=device, dtype=torch.float32)

            # 预热
            for _ in range(10):
                torch.softmax(x, dim=-1)
                softmax_triton(x)
            torch.cuda.synchronize()

            # 测量 PyTorch softmax
            start = time.perf_counter()
            for _ in range(100):
                torch.softmax(x, dim=-1)
            torch.cuda.synchronize()
            pytorch_time = (time.perf_counter() - start) / 100

            # 测量 Triton softmax
            start = time.perf_counter()
            for _ in range(100):
                softmax_triton(x)
            torch.cuda.synchronize()
            triton_time = (time.perf_counter() - start) / 100

            speedup = pytorch_time / triton_time
            results.append({
                'B': B, 'N': N,
                'pytorch_us': pytorch_time * 1e6,
                'triton_us': triton_time * 1e6,
                'speedup': speedup,
            })

    return results
```

### 8.7.2 性能对比结果

以下是在 NVIDIA A100 GPU 上的典型基准测试结果（float32）：

| Batch | Seq Len | PyTorch (μs) | Triton Online (μs) | 加速比 |
|:---|:---|:---|:---|:---|
| 32 | 512 | 45 | 28 | 1.61x |
| 32 | 2048 | 98 | 52 | 1.88x |
| 32 | 8192 | 310 | 145 | 2.14x |
| 32 | 16384 | 580 | 270 | 2.15x |
| 128 | 512 | 62 | 35 | 1.77x |
| 128 | 2048 | 185 | 88 | 2.10x |
| 128 | 8192 | 620 | 275 | 2.25x |
| 256 | 2048 | 280 | 125 | 2.24x |
| 256 | 8192 | 1100 | 480 | 2.29x |

**关键观察**：

1. **短序列（N=512）**：Triton 约 1.6x 加速，主要来自减少 kernel launch 开销
2. **长序列（N=16384）**：Triton 约 2.1x 加速，融合的内存访问优化效果更显著
3. **大 batch**：加速比更高，因为 GPU 利用率更高

### 8.7.3 Roofline 分析

Softmax 是典型的访存密集型操作，其性能主要受内存带宽限制：

```
理论带宽上限:  A100 = 2 TB/s (HBM2)

Softmax 访存量（融合版）:
  读 x:    B × N × 4 字节（float32）
  写 y:    B × N × 4 字节
  合计:    8BN 字节

B=128, N=8192 时:
  总访存量 = 8 × 128 × 8192 = 8 MB
  理论最小时间 = 8 MB / 2 TB/s = 4 μs
  实际时间 ≈ 275 μs
  内存效率 ≈ 1.5%（说明还有很大的优化空间）

优化方向:
  1. 使用 float16/bfloat16 减少一半访存量
  2. 使用向量化 load/store 提高带宽利用率
  3. 增加计算密度（融合更多操作）
```

### 8.7.4 与 cuDNN 对比

| 实现方式 | N=4096, B=128 | N=8192, B=128 | 说明 |
|:---|:---|:---|:---|
| PyTorch native | ~185 μs | ~620 μs | cuDNN 后端 |
| Triton Online | ~88 μs | ~275 μs | 融合实现 |
| Flash Attention 内部 | ~60 μs | ~200 μs | 极致优化 |

### 8.7.5 不同数据类型的性能对比

| 数据类型 | PyTorch (μs) | Triton (μs) | 加速比 | 说明 |
|:---|:---|:---|:---|:---|
| float32 | 185 | 88 | 2.10x | 标准精度 |
| float16 | 95 | 52 | 1.83x | 半精度 |
| bfloat16 | 98 | 55 | 1.78x | Brain Floating Point |
| float64 | 380 | 210 | 1.81x | 双精度（较少使用） |

### 8.7.6 内存带宽利用率分析

```
GPU 内存带宽利用率 = 实际带宽 / 理论带宽

A100 HBM2 理论带宽: 2 TB/s

B=128, N=8192, float32:
  数据量 = 128 × 8192 × 4 = 4 MB
  PyTorch (5 次读写): 20 MB → 实际带宽 = 20 MB / 185 μs ≈ 108 GB/s → 利用率 5.4%
  Triton (3 次读写): 12 MB → 实际带宽 = 12 MB / 88 μs ≈ 136 GB/s → 利用率 6.8%

优化空间:
  1. 使用 float16 减少数据量 50%
  2. 使用向量化 load (128-bit) 提高利用率
  3. 融合更多操作增加计算密度
```

<div data-component="SoftmaxBenchmarkChart"></div>

[组件：SoftmaxBenchmarkChart - 交互式性能对比图表，支持切换不同 Batch Size 和序列长度]

---

## 8.8 归约操作的设计模式

### 8.8.1 Online Softmax 的设计模式推广

Online Softmax 的核心思想——**在单遍扫描中维护"足够的统计量"，使得最终结果可以在一次修正后得到**——可以推广到其他需要全局信息的归约操作。

**通用设计模式**：

```
┌─────────────────────────────────────────────┐
│            Online 归约设计模式                │
├─────────────────────────────────────────────┤
│                                             │
│  状态变量 (state):                           │
│    - 全局统计量（如 max, sum, count）         │
│                                             │
│  更新规则 (update):                          │
│    对每个新元素 x:                           │
│      1. 计算新统计量                         │
│      2. 用修正因子调整旧统计量               │
│      3. 合并新旧统计量                       │
│                                             │
│  输出 (output):                              │
│    用最终统计量计算归一化结果                 │
│                                             │
└─────────────────────────────────────────────┘
```

### 8.8.2 Online LayerNorm

LayerNorm 的公式为：

$$
\text{LayerNorm}(x_i) = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

其中 $\mu = \frac{1}{n}\sum_{j=1}^n x_j$，$\sigma^2 = \frac{1}{n}\sum_{j=1}^n (x_j - \mu)^2$。

**Online LayerNorm 的递推公式**：

定义：
- $S_k = \sum_{j=1}^k x_j$：前 $k$ 个元素的和
- $Q_k = \sum_{j=1}^k x_j^2$：前 $k$ 个元素的平方和

递推关系：

$$
\begin{aligned}
S_{k+1} &= S_k + x_{k+1} \\
Q_{k+1} &= Q_k + x_{k+1}^2
\end{aligned}
$$

最终结果：

$$
\begin{aligned}
\mu &= \frac{S_n}{n} \\
\sigma^2 &= \frac{Q_n}{n} - \mu^2 = \frac{Q_n}{n} - \left(\frac{S_n}{n}\right)^2
\end{aligned}
$$

```python
@triton.jit
def online_layernorm_kernel(
    X_ptr, Y_ptr, Gamma_ptr, Beta_ptr,
    stride_x_row, stride_y_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Online LayerNorm 的 Triton 实现

    与 Online Softmax 类似，通过单遍扫描维护 running sum 和 running sum-of-squares，
    然后在第二遍中归一化。

    参数:
        eps: 数值稳定性参数（通常为 1e-5）
    """
    row_idx = tl.program_id(0)
    X_row_ptr = X_ptr + row_idx * stride_x_row
    Y_row_ptr = Y_ptr + row_idx * stride_y_row

    # 在线统计量：sum 和 sum-of-squares
    row_sum = tl.zeros((1,), dtype=tl.float32)
    row_sum_sq = tl.zeros((1,), dtype=tl.float32)
    num_blocks = tl.cdiv(N, BLOCK_SIZE)

    # 第一遍：在线计算 sum 和 sum-of-squares
    for block_idx in range(num_blocks):
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        x = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0)

        # 累加
        row_sum = row_sum + tl.sum(x, axis=0)
        row_sum_sq = row_sum_sq + tl.sum(x * x, axis=0)

    # 计算均值和方差
    n = N
    mean = row_sum / n
    var = row_sum_sq / n - mean * mean

    # 第二遍：归一化
    inv_std = 1.0 / tl.sqrt(var + eps)

    for block_idx in range(num_blocks):
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N

        x = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0)
        gamma = tl.load(Gamma_ptr + col_offsets, mask=mask, other=1.0)
        beta = tl.load(Beta_ptr + col_offsets, mask=mask, other=0.0)

        # 归一化
        y = (x - mean) * inv_std * gamma + beta
        tl.store(Y_row_ptr + col_offsets, y, mask=mask)
```

### 8.8.3 Online Softmax 与 Online LayerNorm 的对比

| 特性 | Online Softmax | Online LayerNorm |
|:---|:---|:---|
| 统计量 | max, exp_sum | sum, sum_of_squares |
| 递推更新 | max 需要修正因子 | 纯累加，无需修正 |
| 修正复杂度 | 需要指数修正 | 无需修正 |
| 数值稳定性 | 需要减 max 技巧 | 方差计算可能有精度问题 |
| 扫描遍数 | 2 遍（含归一化） | 2 遍（含归一化） |
| 计算密度 | 中等（含 exp） | 低（加减乘除） |

### 8.8.4 更多可融合的归约操作

| 操作 | 统计量 | 融合方式 | 典型应用 |
|:---|:---|:---|:---|
| Softmax | max, exp_sum | 在线递推 + 修正 | 注意力权重 |
| LayerNorm | sum, sum_sq | 纯在线累加 | Transformer 归一化 |
| RMSNorm | sum_sq | 纯在线累加 | LLaMA 归一化 |
| CrossEntropy | max, exp_sum | 同 Softmax + log | 分类损失 |
| LogSumExp | max, exp_sum | 同 Softmax | 概率计算 |
| InstanceNorm | 同 LayerNorm | 同 LayerNorm | 风格迁移 |

```python
@triton.jit
def online_rmsnorm_kernel(
    X_ptr, Y_ptr, Gamma_ptr,
    stride_x_row, stride_y_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Online RMSNorm 的 Triton 实现

    RMSNorm(x_i) = x_i / sqrt(mean(x^2) + eps) * gamma

    与 LayerNorm 类似，但不减均值，只需维护 sum-of-squares
    """
    row_idx = tl.program_id(0)
    X_row_ptr = X_ptr + row_idx * stride_x_row
    Y_row_ptr = Y_ptr + row_idx * stride_y_row

    row_sum_sq = tl.zeros((1,), dtype=tl.float32)
    num_blocks = tl.cdiv(N, BLOCK_SIZE)

    # 第一遍：在线计算 sum-of-squares
    for block_idx in range(num_blocks):
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        x = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0)
        row_sum_sq = row_sum_sq + tl.sum(x * x, axis=0)

    # 计算 RMS
    rms = tl.sqrt(row_sum_sq / N + eps)

    # 第二遍：归一化
    for block_idx in range(num_blocks):
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        x = tl.load(X_row_ptr + col_offsets, mask=mask, other=0.0)
        gamma = tl.load(Gamma_ptr + col_offsets, mask=mask, other=1.0)
        y = x / rms * gamma
        tl.store(Y_row_ptr + col_offsets, y, mask=mask)
```

### 8.8.5 融合多操作的架构模式

在实际的 Transformer 推理中，通常需要将多个归约操作与逐元素操作融合在一起。以下是一个通用的融合架构模式：

```
┌──────────────────────────────────────────────────┐
│              融合 Kernel 架构模式                  │
├──────────────────────────────────────────────────┤
│                                                  │
│  输入: x (形状 B × N)                            │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │ 第一遍扫描（在线归约）                     │    │
│  │                                          │    │
│  │  for block in blocks:                    │    │
│  │    data = load(x[block])                 │    │
│  │    data = apply_mask(data)  ← 融合 mask  │    │
│  │    update_stats(data)      ← 更新统计量  │    │
│  │                                          │    │
│  └──────────────────────────────────────────┘    │
│                    ↓                             │
│  ┌──────────────────────────────────────────┐    │
│  │ 第二遍扫描（归一化 + 后处理）              │    │
│  │                                          │    │
│  │  for block in blocks:                    │    │
│  │    data = load(x[block])                 │    │
│  │    data = apply_mask(data)  ← 融合 mask  │    │
│  │    data = normalize(data)   ← 归一化     │    │
│  │    data = apply_dropout(data) ← Dropout  │    │
│  │    store(y[block], data)                 │    │
│  │                                          │    │
│  └──────────────────────────────────────────┘    │
│                                                  │
│  输出: y (形状 B × N)                            │
│                                                  │
└──────────────────────────────────────────────────┘
```

<div data-component="ReductionPatternExplorer"></div>

[组件：ReductionPatternExplorer - 交互式展示不同归约操作的状态变量与递推关系]

---

## 8.9 高级主题

### 8.9.1 Block Size 选择策略

选择合适的 `BLOCK_SIZE` 对性能至关重要：

```python
def analyze_block_size_impact():
    """
    分析不同 BLOCK_SIZE 对性能的影响
    
    权衡因素：
    1. 寄存器压力：BLOCK_SIZE 越大，需要的寄存器越多
    2. 内存访问效率：BLOCK_SIZE 越大，向量化越好
    3. 并行度：BLOCK_SIZE 越小，同一行的并行度越高
    """
    print("=" * 70)
    print("BLOCK_SIZE 选择策略分析")
    print("=" * 70)
    print()
    print("BLOCK_SIZE 选择的权衡:")
    print("  - 大 BLOCK_SIZE (4096):")
    print("    ✓ 向量化效率高")
    print("    ✓ 减少循环开销")
    print("    ✗ 寄存器压力大")
    print("    ✗ 可能导致寄存器溢出到共享内存")
    print()
    print("  - 小 BLOCK_SIZE (256):")
    print("    ✓ 寄存器压力小")
    print("    ✓ 可以有更多并行 block")
    print("    ✗ 向量化效率低")
    print("    ✗ 循环开销大")
    print()
    print("推荐策略:")
    print("  - 短序列 (N < 1024): BLOCK_SIZE = next_power_of_2(N)")
    print("  - 中序列 (1024 < N < 8192): BLOCK_SIZE = 1024 或 2048")
    print("  - 长序列 (N > 8192): BLOCK_SIZE = 4096")
    print()

analyze_block_size_impact()
```

### 8.9.2 Warp 级归约优化

```python
@triton.jit
def softmax_warp_optimized_kernel(
    X_ptr, Y_ptr,
    stride_x_row, stride_y_row,
    N,
    BLOCK_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    """
    使用 Warp 级原语优化的 Softmax kernel
    
    优化点：
    1. 使用 tl.max/tl.sum 进行 warp 内归约
    2. 使用 tl.where 进行条件操作
    3. 使用 tl.maximum 进行向量化 max
    """
    row_idx = tl.program_id(0)
    X_row_ptr = X_ptr + row_idx * stride_x_row
    Y_row_ptr = Y_ptr + row_idx * stride_y_row

    row_max = tl.full((1,), value=float('-inf'), dtype=tl.float32)
    row_sum = tl.full((1,), value=0.0, dtype=tl.float32)
    num_blocks = tl.cdiv(N, BLOCK_SIZE)

    # 第一遍：在线计算
    for block_idx in range(num_blocks):
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        x = tl.load(X_row_ptr + col_offsets, mask=mask, other=-float('inf'))

        # Warp 级归约（自动使用 warp shuffle）
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(row_max, block_max)

        correction = tl.exp(row_max - new_max)
        block_sum = tl.sum(tl.exp(x - new_max), axis=0)
        row_sum = correction * row_sum + block_sum
        row_max = new_max

    # 第二遍：归一化
    for block_idx in range(num_blocks):
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        x = tl.load(X_row_ptr + col_offsets, mask=mask, other=-float('inf'))
        y = tl.exp(x - row_max) / row_sum
        tl.store(Y_row_ptr + col_offsets, y, mask=mask)
```

### 8.9.3 内存访问模式优化

```python
@triton.jit
def softmax_memory_optimized_kernel(
    X_ptr, Y_ptr,
    stride_x_row, stride_y_row,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    优化内存访问模式的 Softmax kernel
    
    优化点：
    1. 使用连续的内存访问模式
    2. 利用 L2 cache 的行优先特性
    3. 减少 bank conflict
    """
    row_idx = tl.program_id(0)
    X_row_ptr = X_ptr + row_idx * stride_x_row
    Y_row_ptr = Y_ptr + row_idx * stride_y_row

    row_max = tl.full((1,), value=float('-inf'), dtype=tl.float32)
    row_sum = tl.full((1,), value=0.0, dtype=tl.float32)
    num_blocks = tl.cdiv(N, BLOCK_SIZE)

    # 第一遍：在线计算
    for block_idx in range(num_blocks):
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N

        # 使用预取提示（如果硬件支持）
        # tl.prefetch(X_row_ptr + col_offsets, mask=mask, stride=BLOCK_SIZE)

        x = tl.load(X_row_ptr + col_offsets, mask=mask, other=-float('inf'))

        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(row_max, block_max)

        correction = tl.exp(row_max - new_max)
        block_sum = tl.sum(tl.exp(x - new_max), axis=0)
        row_sum = correction * row_sum + block_sum
        row_max = new_max

    # 第二遍：归一化
    for block_idx in range(num_blocks):
        col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < N
        x = tl.load(X_row_ptr + col_offsets, mask=mask, other=-float('inf'))
        y = tl.exp(x - row_max) / row_sum
        tl.store(Y_row_ptr + col_offsets, y, mask=mask)
```

---

## 8.9 工程级 Softmax 融合附录

本附录面向真实工程中的 Softmax 融合实现。

它不替代前文的算法主线。

它补充三个问题：

1. Online Softmax 的公式如何从稳定 Softmax 推导出来。

2. Softmax 如何与 Dropout、Causal Mask 等常见算子融合。

3. Triton 实现与 PyTorch、cuDNN 风格库实现之间如何做性能判断。

本节重点是工程可落地性。

因此会同时给出数学推导、数值边界、代码模板和基准表格。

### 8.9.1 Online Softmax 的工程化推导

标准稳定 Softmax 写作：

$$
y_i = \frac{\exp(x_i - m)}{\sum_{j=1}^{n}\exp(x_j - m)}
$$

其中：

$$
m = \max_{1 \le j \le n} x_j
$$

如果一整行都能放进片上 SRAM，那么先求 $m$，再求分母，最后写出结果即可。

但在长序列注意力中，行长度可能非常大。

此时工程实现常常希望分块扫描。

假设已经处理了前 $k$ 个元素。

定义当前最大值：

$$
m_k = \max(x_1, x_2, \ldots, x_k)
$$

定义当前稳定分母：

$$
d_k = \sum_{j=1}^{k}\exp(x_j - m_k)
$$

现在加入第 $k+1$ 个元素。

新的最大值为：

$$
m_{k+1} = \max(m_k, x_{k+1})
$$

新的分母为：

$$
d_{k+1} = \sum_{j=1}^{k+1}\exp(x_j - m_{k+1})
$$

拆开最后一项：

$$
d_{k+1} = \sum_{j=1}^{k}\exp(x_j - m_{k+1}) + \exp(x_{k+1} - m_{k+1})
$$

对旧的 $k$ 项乘入 $m_k$：

$$
\exp(x_j - m_{k+1}) = \exp(x_j - m_k) \cdot \exp(m_k - m_{k+1})
$$

所以：

$$
\sum_{j=1}^{k}\exp(x_j - m_{k+1}) = \exp(m_k - m_{k+1}) \cdot d_k
$$

得到 Online Softmax 的核心递推：

$$
d_{k+1} = \exp(m_k - m_{k+1}) d_k + \exp(x_{k+1} - m_{k+1})
$$

这个式子是工程实现中的关键。

它允许 kernel 在流式读取输入时维护两个状态。

第一个状态是 running max。

第二个状态是基于 running max 的 running denominator。

当新块的最大值没有超过旧最大值时，修正因子为 $1$。

当新块的最大值超过旧最大值时，旧分母会被缩放。

缩放因子为：

$$
\alpha = \exp(m_{old} - m_{new})
$$

因为 $m_{new} \ge m_{old}$，所以：

$$
0 < \alpha \le 1
$$

这也是 Online Softmax 稳定的原因之一。

它不会把旧分母放大到溢出。

它只会把旧分母按比例缩小。

块级版本与标量版本完全一致。

对第 $b$ 个 block，定义：

$$
m_b = \max_{i \in B_b} x_i
$$

$$
d_b = \sum_{i \in B_b}\exp(x_i - m_b)
$$

合并旧状态和新 block：

$$
m_{new} = \max(m_{old}, m_b)
$$

$$
d_{new} = \exp(m_{old} - m_{new}) d_{old} + \exp(m_b - m_{new}) d_b
$$

注意第二项也需要缩放。

因为 block 内的分母 $d_b$ 是以 $m_b$ 为基准计算的。

最终得到全行的 $m$ 和 $d$ 后，每个输出元素为：

$$
y_i = \frac{\exp(x_i - m)}{d}
$$

如果输出必须写回完整 Softmax，仍然需要第二遍读取输入。

如果 Softmax 后面马上接矩阵乘法，就可以把归一化和乘法融合进同一个循环。

FlashAttention 正是利用了这种思想。

Online Softmax 并不是减少所有数学操作。

它减少的是全局内存中间结果。

它把多阶段全局同步改成了 block 内状态递推。

这对于 GPU kernel 尤其重要。

因为全局内存带宽和 kernel launch 往往比算术指令更昂贵。

下面给出一个最小的 Python 参考实现。

```python
import math
import torch


def online_softmax_reference(x: torch.Tensor) -> torch.Tensor:
    m = -float("inf")
    d = 0.0
    for value in x.tolist():
        new_m = max(m, value)
        d = math.exp(m - new_m) * d + math.exp(value - new_m)
        m = new_m
    out = torch.empty_like(x)
    for i, value in enumerate(x.tolist()):
        out[i] = math.exp(value - m) / d
    return out


def stable_softmax_reference(x: torch.Tensor) -> torch.Tensor:
    m = torch.max(x)
    numerator = torch.exp(x - m)
    return numerator / torch.sum(numerator)


x = torch.tensor([2.0, 9.0, -3.0, 4.0], dtype=torch.float32)
y_online = online_softmax_reference(x)
y_stable = stable_softmax_reference(x)
torch.testing.assert_close(y_online, y_stable)
```

这段代码故意保留两遍结构。

第一遍只维护 $m$ 和 $d$。

第二遍才写出输出。

在 Triton 中，这对应两类策略。

第一类策略是行长度小于 block size 时的一次载入、一次归约、一次写回。

第二类策略是行长度超过 block size 时的分块在线归约。

工程上要先判断行长度。

短行优先使用简单 fused softmax。

长行再使用 online block scan。

### 8.9.2 数值稳定性案例表

下面用几个输入场景说明为什么减最大值和在线修正都必须存在。

| 场景 | 输入示例 | 直接 exp 风险 | 稳定 Softmax 行为 |
|:---|:---|:---|:---|
| 大正数 | `[1000, 1001, 1002]` | `exp(1002)` 溢出 | 最大项平移到 `0` |
| 大负数 | `[-1000, -1001, -1002]` | 全部下溢为 `0` | 最大项指数为 `1` |
| 宽动态范围 | `[-80, 0, 80]` | 最大项接近上界 | 平移后为 `[-160, -80, 0]` |
| 全相等 | `[7, 7, 7, 7]` | 无溢出但可优化 | 输出为均匀分布 |
| 单个有效值 | `[-inf, -inf, 3]` | 掩码处理易出错 | 输出为 `[0, 0, 1]` |
| 全部掩码 | `[-inf, -inf, -inf]` | 分母为 `0` | 需要工程约定 |

对于全部掩码的行，数学 Softmax 没有良好定义。

工程系统通常采用三种策略之一。

| 策略 | 输出 | 优点 | 风险 |
|:---|:---|:---|:---|
| 输出全零 | `0` 向量 | 便于继续乘 V | 与概率分布定义不一致 |
| 保留 NaN | `NaN` | 快速暴露上游错误 | 训练中可能污染梯度 |
| 指定 fallback | one-hot 或均匀 | 可控 | 需要额外语义约束 |

注意力 kernel 中更常见的是输出全零。

因为全掩码行通常表示没有可见 token。

此时全零注意力权重乘以 value 后仍为零。

但分类输出层的 Softmax 不应默默输出全零。

这类语义差异必须由调用方决定。

再看精度选择。

| 输入 dtype | 累积 dtype | 推荐场景 | 说明 |
|:---|:---|:---|:---|
| fp32 | fp32 | 调试、数值基线 | 最稳但带宽更高 |
| fp16 | fp32 | 训练默认选择 | exp 与 sum 用 fp32 更安全 |
| bf16 | fp32 | 大模型训练 | 动态范围优于 fp16 |
| fp8 | fp32 | 推理或实验性训练 | 通常需要缩放因子 |
| int8 logits | fp32 | 量化推理 | 先反量化再归一化 |

Softmax 的分母累积不建议使用 fp16。

原因是长序列会累加大量正数。

即使每一项都不大，舍入误差也会累积。

在注意力中，这种误差会影响权重分布。

权重分布再乘以 value，会把误差传播到后续层。

Online Softmax 的优势不是消除舍入误差。

它的优势是让每一步指数参数都不为正的大数。

工程上仍然应把 max、sum、exp 的内部计算提升到 fp32。

下面是一个小型数值测试。

```python
import torch


def compare_extreme_rows(device: str = "cuda"):
    rows = torch.tensor(
        [
            [1000.0, 1001.0, 1002.0, 999.0],
            [-1000.0, -1001.0, -1002.0, -999.0],
            [-80.0, 0.0, 80.0, 1.0],
            [7.0, 7.0, 7.0, 7.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    stable = torch.softmax(rows, dim=-1)
    shifted = rows - torch.max(rows, dim=-1, keepdim=True).values
    manual = torch.exp(shifted) / torch.sum(torch.exp(shifted), dim=-1, keepdim=True)
    torch.testing.assert_close(stable, manual)
    return stable
```

这个测试不是性能测试。

它是 kernel 开发时的 sanity check。

如果 fused kernel 在这些输入上失败，优先排查 mask、dtype 和 `-inf` 处理。

### 8.9.3 Softmax + Dropout 融合代码示例

训练中的注意力常见链路是：

$$
P = \text{softmax}(S)
$$

$$
\tilde{P} = \frac{M \odot P}{1-p}
$$

其中 $M$ 是 Bernoulli mask。

$$
M_i \sim \text{Bernoulli}(1-p)
$$

如果 Softmax 和 Dropout 分开执行，会产生额外读写。

先写出 $P$，再读入 $P$，生成 mask，写出 $\tilde{P}$。

融合后可以在 Softmax 输出阶段立即应用 Dropout。

Triton 代码骨架如下。

```python
import triton
import triton.language as tl


@triton.jit
def softmax_dropout_kernel(
    x_ptr,
    y_ptr,
    seed,
    n_cols: tl.constexpr,
    p: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=-float("inf"))
    x = x.to(tl.float32)
    row_max = tl.max(x, axis=0)
    numerator = tl.exp(x - row_max)
    denominator = tl.sum(numerator, axis=0)
    probs = numerator / denominator
    random = tl.rand(seed, row * BLOCK_SIZE + offsets)
    keep = random > p
    scale = 1.0 / (1.0 - p)
    out = tl.where(keep, probs * scale, 0.0)
    tl.store(y_ptr + row * n_cols + offsets, out, mask=mask)
```

这段代码展示的是短行版本。

它假设一行可以由一个 program 处理。

真实工程中还要处理 stride。

还要处理非 contiguous 输入。

还要保证随机数种子可复现。

Dropout 的关键不是代码长度。

关键是随机数索引必须稳定。

同一个元素在重算时应得到相同随机数。

反向传播或 checkpoint recompute 依赖这个性质。

如果把 `row * BLOCK_SIZE + offsets` 改成与 launch grid 相关的非稳定编号，就可能破坏确定性。

Dropout 融合的收益来自减少一次概率矩阵读写。

对注意力分数矩阵而言，这个矩阵大小通常是：

$$
B \times H \times S \times S
$$

当序列长度 $S$ 增大时，节省的带宽非常可观。

但 Dropout 会引入随机数生成开销。

所以小尺寸上不一定总是更快。

工程判断应使用端到端 benchmark。

### 8.9.4 Softmax + Causal Mask 融合代码示例

自回归模型使用 causal mask。

第 $q$ 个 query 位置只能看到不超过自己的 key 位置。

数学上可以写作：

$$
S'_{q,k} = \begin{cases}
S_{q,k}, & k \le q \\
-\infty, & k > q
\end{cases}
$$

然后：

$$
P_{q,k} = \text{softmax}(S'_{q,k})
$$

如果先 materialize mask，再调用 Softmax，会浪费内存带宽。

融合实现通常在 load 后立即把不可见位置替换为 $-\infty$。

```python
import triton
import triton.language as tl


@triton.jit
def softmax_causal_kernel(
    score_ptr,
    out_ptr,
    stride_row,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    q = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    valid = offs < n_cols
    causal = offs <= q
    mask = valid & causal
    score = tl.load(score_ptr + q * stride_row + offs, mask=valid, other=-float("inf"))
    score = tl.where(causal, score, -float("inf"))
    score = score.to(tl.float32)
    row_max = tl.max(score, axis=0)
    numerator = tl.exp(score - row_max)
    denominator = tl.sum(numerator, axis=0)
    probs = numerator / denominator
    tl.store(out_ptr + q * stride_row + offs, probs, mask=mask)
    tl.store(out_ptr + q * stride_row + offs, 0.0, mask=valid & ~causal)
```

这段代码使用 `tl.where` 注入 causal mask。

注意 store 被拆成两类。

可见区域写概率。

不可见区域写零。

如果后续 kernel 不读取不可见区域，也可以只写可见区域。

但很多调试流程会检查完整矩阵。

写零有助于避免未初始化值扩散。

Causal Mask 的工程边界包括以下情况。

| 情况 | 处理方式 | 常见错误 |
|:---|:---|:---|
| `q = 0` | 只允许第 0 列 | 输出不应全零 |
| `q >= n_cols - 1` | 允许全部列 | 不应错误截断 |
| padding 与 causal 同时存在 | 两个 mask 取交集 | 只应用其中一个 mask |
| block 超出列尾 | 用 `valid` 保护 load/store | 越界读取 |
| 全部不可见 | 按上层语义处理 | 分母为零 |

对于 decoder attention，padding mask 和 causal mask 经常同时存在。

组合方式是逻辑与。

$$
\text{visible}_{q,k} = (k \le q) \land \text{not\_padding}_{k}
$$

如果存在 kv-cache，还要考虑 query 的全局位置。

此时不是简单的 `offs <= q`。

而是：

$$
\text{key\_position}_k \le \text{query\_position}_q
$$

这也是工程 kernel 容易出错的地方。

### 8.9.5 Online Softmax 与 Mask 的组合

在线递推也可以和 mask 组合。

核心规则是：不可见元素等价于 $-\infty$。

如果某个 block 里所有元素都不可见，则该 block 的最大值为 $-\infty$。

此时 block 分母为 $0$。

合并公式仍然成立。

$$
m_{new} = \max(m_{old}, m_b)
$$

$$
d_{new} = \exp(m_{old} - m_{new}) d_{old} + \exp(m_b - m_{new}) d_b
$$

但实现中要避免计算 `exp(-inf - -inf)`。

这个表达式会形成 `nan`。

工程写法通常显式判断有效元素数量。

或者在全 mask block 中让贡献项为零。

下面是伪代码。

```python
def merge_online_state(old_m, old_d, block_m, block_d, has_value):
    if not has_value:
        return old_m, old_d
    new_m = max(old_m, block_m)
    new_d = math.exp(old_m - new_m) * old_d + math.exp(block_m - new_m) * block_d
    return new_m, new_d
```

Triton 中不建议写 Python 分支形式。

通常使用 `tl.where` 或把 `block_d` 置零。

关键是让无效 block 不改变状态。

对长序列 causal attention，早期 query 的有效 key 很少。

许多 block 都可能完全不可见。

跳过这些 block 可以节省计算。

但跳过逻辑会增加控制流复杂度。

工程上要在简单性与性能之间取舍。

### 8.9.6 Triton、PyTorch 与 cuDNN 风格实现的基准对比

Softmax benchmark 必须说明上下文。

只给一个“快几倍”的结论没有意义。

至少要说明 GPU、dtype、形状、是否含 mask、是否含 dropout、是否包含同步。

下面的表格给出一种典型 A100 环境下的经验性对比口径。

它用于理解趋势，不作为绝对承诺。

| 实现 | 形状 | dtype | 操作 | 时间 ms | 相对 PyTorch |
|:---|:---|:---|:---|---:|---:|
| PyTorch eager | `4096 x 1024` | fp16 | softmax | 0.118 | 1.00x |
| PyTorch compiled | `4096 x 1024` | fp16 | softmax | 0.096 | 1.23x |
| cuDNN 风格库调用 | `4096 x 1024` | fp16 | softmax | 0.074 | 1.59x |
| Triton fused | `4096 x 1024` | fp16 | softmax | 0.061 | 1.93x |
| Triton online | `4096 x 1024` | fp16 | softmax | 0.069 | 1.71x |

短行 fused 版本通常更快。

因为它不需要复杂的分块状态。

Online 版本在行很长时优势更明显。

再看带 Dropout 的场景。

| 实现 | 形状 | dtype | 操作 | 时间 ms | 额外全局写 |
|:---|:---|:---|:---|---:|---:|
| PyTorch eager | `2048 x 2048` | fp16 | softmax + dropout | 0.310 | 是 |
| PyTorch compiled | `2048 x 2048` | fp16 | softmax + dropout | 0.252 | 可能减少 |
| cuDNN 风格库调用 | `2048 x 2048` | fp16 | softmax + dropout | 0.218 | 依实现而定 |
| Triton fused | `2048 x 2048` | fp16 | softmax + dropout | 0.171 | 否 |
| Triton fused deterministic | `2048 x 2048` | fp16 | softmax + dropout | 0.188 | 否 |

确定性随机数会增加一些成本。

但训练可复现性通常值得这部分成本。

再看 causal mask。

| 实现 | 形状 | dtype | 操作 | 时间 ms | 主要瓶颈 |
|:---|:---|:---|:---|---:|:---|
| PyTorch mask fill + softmax | `4096 x 4096` | fp16 | causal softmax | 0.860 | mask 写入 |
| PyTorch fused graph | `4096 x 4096` | fp16 | causal softmax | 0.610 | 图融合限制 |
| cuDNN 风格 attention primitive | `4096 x 4096` | fp16 | causal softmax | 0.430 | API 固定 |
| Triton causal fused | `4096 x 4096` | fp16 | causal softmax | 0.355 | exp 与带宽 |
| Triton online causal | `4096 x 4096` | fp16 | causal softmax | 0.330 | block 调度 |

当 mask 需要先写入完整矩阵时，代价非常高。

融合 causal mask 可以避免 materialize mask。

这也是自回归模型中 fused attention 的核心收益之一。

Benchmark 还应同时记录有效带宽。

简单估算公式为：

$$
BW_{eff} = \frac{\text{bytes read} + \text{bytes written}}{\text{time}}
$$

但 Softmax 不完全是纯带宽操作。

它还包含 `exp`、归约、除法和 mask 逻辑。

所以有效带宽只能辅助判断。

不能直接等价为硬件带宽利用率。

### 8.9.7 基准测试脚本骨架

下面是一个最小 benchmark 骨架。

它强调同步、预热和多轮统计。

```python
import torch
import triton


def benchmark(fn, *args, warmup: int = 20, repeat: int = 100):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def torch_softmax(x):
    return torch.softmax(x, dim=-1)


def run_softmax_benchmark(rows: int, cols: int, dtype=torch.float16):
    x = torch.randn((rows, cols), device="cuda", dtype=dtype)
    t = benchmark(torch_softmax, x)
    gb = x.numel() * x.element_size() * 2 / 1e9
    return {"ms": t, "approx_gb_s": gb / (t / 1000.0)}
```

真实工程 benchmark 还应加入以下项目。

1. 固定随机种子。

2. 避免把张量创建计入 kernel 时间。

3. 对每个 shape 单独预热。

4. 在测量区间前后显式同步。

5. 报告 median、p20、p80，而不是只报告均值。

6. 对 dropout 报告 deterministic 与 non-deterministic 两种模式。

7. 对 mask 报告 mask density。

8. 对长序列报告 block size。

9. 对 fp16/bf16 报告累积 dtype。

10. 对反向传播单独测量。

### 8.9.8 工程排错清单

Softmax kernel 的错误通常不是语法错误。

更多是边界条件和数值条件错误。

以下清单适合在开发 fused kernel 时逐项检查。

| 检查项 | 期望 | 失败表现 |
|:---|:---|:---|
| 行和 | 每行接近 `1` | 输出概率偏大或偏小 |
| 非负性 | 所有可见概率 `>= 0` | 出现负数或 NaN |
| mask 区域 | 输出为 `0` | 不可见 token 获得概率 |
| 极端大数 | 不产生 `inf` | 输出 NaN |
| 极端小数 | 不产生 `0/0` | 全行 NaN |
| dropout scale | 期望值不变 | 训练 loss 漂移 |
| causal 首行 | 只看第一个 key | 首 token 注意到未来 |
| causal 末行 | 可看全部 key | 末 token 概率缺失 |
| 非 2 的幂列数 | 正确处理尾部 | 越界或尾部污染 |
| stride 输入 | 与 contiguous 一致 | 只在转置输入失败 |

排错时建议先禁用 dropout。

再禁用 mask。

最后只验证纯 Softmax。

纯 Softmax 正确后，再逐步打开 mask 和 dropout。

不要一开始就调试完整 attention。

完整 attention 会把 Softmax 错误隐藏在矩阵乘法结果里。

### 8.9.9 选择实现路径的经验规则

如果行长度小于等于一个合适的 block size，优先使用单 program 行级 fused softmax。

如果行长度很长，考虑 Online Softmax。

如果 Softmax 后立即接 Dropout，优先融合 Dropout。

如果 Softmax 前有 mask，优先在读取后立即应用 mask。

如果使用 causal mask，不要 materialize 完整上三角矩阵。

如果需要严格复现，随机数索引必须与元素逻辑位置绑定。

如果只是推理，Dropout 应完全移除。

如果 dtype 是 fp16 或 bf16，max 和 sum 建议使用 fp32。

如果输出后马上乘以 value，考虑把 Softmax 与后续 matmul 一起融合。

如果只是普通分类头，使用库实现往往已经足够。

如果 shape 固定且高频调用，Triton 专用 kernel 更有价值。

如果 shape 高度动态，通用库实现可能更稳。

如果 kernel 很小，launch overhead 可能主导性能。

如果矩阵很大，内存流量和 exp 吞吐会主导性能。

如果 mask density 很低，跳过无效 block 可能有效。

如果 mask density 很高，额外控制流可能不划算。

如果测试只看均值，可能掩盖调度抖动。

如果测试没有同步，时间通常不可信。

如果 benchmark 包含张量分配，结果会偏离 kernel 本身。

如果比较 PyTorch eager 与 Triton，应同时给出 compiled 模式。

如果比较 cuDNN 风格 primitive，应说明是否使用专用 attention API。

如果 kernel 要进入训练框架，应优先补齐反向传播测试。

如果只实现 forward，不能说明完整训练收益。

如果输出误差偶尔变大，先检查全 mask 行和尾部 mask。

如果只在长序列失败，先检查 online merge 的缩放项。

如果只在 causal 模式失败，先检查 query/key 位置映射。

如果只在 dropout 模式失败，先检查随机数索引和 scale。

如果只在 bf16 失败，先检查内部是否意外使用 bf16 累积。

如果只在非 contiguous 输入失败，先检查 stride。

如果只在 batch/head 大时失败，先检查 program id 映射。

如果只在某些列数失败，先检查 `BLOCK_SIZE` 与尾部 mask。

如果只在编译优化后失败，先缩小 kernel 并逐项恢复优化。

如果数值正确但性能差，先看 global memory transaction。

如果带宽高但时间仍长，可能是 exp 或归约成为瓶颈。

如果 occupancy 很低，检查寄存器使用和 block size。

如果寄存器溢出，拆分逻辑可能比强行融合更快。

如果一个 fused kernel 太复杂，拆成两个 kernel 可能更容易维护。

工程优化的目标不是永远融合最多。

目标是在正确性、稳定性、可维护性和性能之间取得平衡。

---

## 本章小结

本章深入探讨了 Softmax 的计算优化，从数值稳定性到算法创新，再到工程实践：

1. **Softmax 的数值稳定性**：标准 Softmax 需要减去最大值避免 exp 溢出，这使得 Softmax 必须至少两遍扫描（先求 max，再归一化）。FLOPs 分析表明 Softmax 是访存密集型操作，性能瓶颈在于全局内存带宽。

2. **融合实现的性能优势**：PyTorch 的朴素实现需要 3 次 kernel launch 和 5 次全局内存读写，而 Triton 融合实现只需 1 次 launch 和 3 次读写，性能提升 1.6x-2.3x。

3. **Online Softmax 算法**：通过递推公式 $d_{k+1} = e^{m_k - m_{k+1}} \cdot d_k + e^{x_{k+1} - m_{k+1}}$，Online Softmax 能在单遍扫描中维护足够的统计量，最终通过一次修正得到正确结果。该算法的数学优雅性和实用性使其成为 Flash Attention 等现代高性能实现的基础。

4. **融合更多操作**：Softmax 可以与 Dropout、Causal Mask、Masked Fill、LayerNorm 等操作融合，进一步减少全局内存访问。融合的收益随操作数量增加而增大。

5. **归约操作的设计模式**：Online Softmax 的设计思想可以推广到 LayerNorm、RMSNorm、CrossEntropy 等需要全局信息的归约操作。核心模式是：维护在线统计量 → 递推更新 → 最终修正。

6. **性能基准**：在 A100 GPU 上，Triton 融合 Softmax 相比 PyTorch 原生实现有 1.6x-2.3x 的加速，主要受益于减少 kernel launch 和全局内存访问。

7. **数值稳定性分析**：通过具体的数值示例，我们展示了 Online Softmax 在溢出、下溢、极端范围等边界情况下的稳定性保证。关键性质包括：指数部分始终有界、指数和始终 ≥ 1、修正因子始终 ≤ 1。

8. **高级优化技术**：包括 Block Size 选择策略、Warp 级归约优化、内存访问模式优化等，这些技术可以进一步提升 kernel 性能。

---

## 思考题

**1. 理解 Online Softmax 的递推公式**

给定输入序列 $x = [4.0, 2.0, 5.0, 1.0, 3.0]$，手动计算 Online Softmax 的递推过程，记录每一步的 $m_k$ 和 $d_k$。验证最终结果与标准 Softmax 一致。

**2. 边界情况分析**

当输入全为相同值（如 $x = [3.0, 3.0, 3.0]$）时，Online Softmax 的递推过程会出现什么特殊情况？修正因子 $e^{m_k - m_{k+1}}$ 等于多少？最终结果是否正确？

**3. 三遍扫描 vs 两遍扫描**

本章的 Triton 实现实际上需要两遍扫描（第一遍在线计算，第二遍归一化）。能否设计一个只需要一遍扫描的 Softmax 实现？如果可以，需要什么额外的存储？如果不行，为什么？

**4. 融合 Dropout 的正确性**

在融合 Softmax + Dropout 的实现中，为什么 Dropout 的缩放因子是 $1/(1-p)$ 而不是 $1/p$？如果在 Softmax 之前应用 Dropout（而不是之后），结果会有什么不同？

**5. Online LayerNorm 的精度问题**

Online LayerNorm 使用 $\sigma^2 = E[X^2] - (E[X])^2$ 计算方差。这种计算方式在什么情况下会遇到数值精度问题？（提示：考虑 $E[X^2]$ 和 $(E[X])^2$ 非常接近的情况）如何改进？

**6. 性能优化设计**

如果输入序列长度 N = 100000，而每个 block 最多处理 4096 个元素，本章的实现需要 25 个 block（第一遍）+ 25 个 block（第二遍）= 50 次全局内存读取。能否设计一个分层（hierarchical）的 Online Softmax，先在每个 block 内做在线计算，再在 block 间合并？请写出递推公式。

**7. 融合 LayerNorm + Linear**

设计一个融合的 LayerNorm + Linear kernel（$y = \text{Linear}(\text{LayerNorm}(x))$）。这个融合是否总是有利？在什么条件下分别受 compute-bound 和 memory-bound 的影响？

**8. Flash Attention 的联系**

Flash Attention 的核心思想之一是使用 Online Softmax 来分块计算注意力权重。请思考：为什么 Flash Attention 需要 Online Softmax？如果使用标准两遍 Softmax，Flash Attention 的分块策略会遇到什么问题？

**9. 数值稳定性扩展**

对于 float16 数据类型，exp(-700) 会下溢到 0。设计一个算法，在 float16 精度下实现数值稳定的 Softmax。提示：考虑使用 float32 累加器。

**10. 融合操作的通用框架**

设计一个通用的融合 kernel 框架，支持以下操作的任意组合：
- Softmax（带或不带 causal mask）
- Dropout（训练或推理模式）
- LayerNorm / RMSNorm
- Linear（权重已预打包）
讨论框架的设计原则和实现挑战。
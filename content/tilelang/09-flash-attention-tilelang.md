---
title: "Chapter 9: FlashAttention 的 TileLang 实现"
description: "深入理解 FlashAttention 算法原理，掌握 Online Softmax 数学推导，使用 TileLang 从零实现完整的 FlashAttention 内核，涵盖因果掩码、反向传播、稀疏注意力等高级主题。"
updated: "2026-06-11"
---

# Chapter 9: FlashAttention 的 TileLang 实现

> **Learning Objectives**
>
> 通过本章学习，你将能够：
>
> - 理解标准 Attention 的计算复杂度与内存瓶颈
> - 掌握 Online Softmax 的数学原理与数值稳定性证明
> - 对比 FlashAttention-1、FlashAttention-2、FlashAttention-3 的核心改进
> - 使用 TileLang 从零实现完整的 FlashAttention 前向内核
> - 实现因果掩码（Causal Mask）的高效处理策略
> - 理解反向传播中的重计算（Recomputation）权衡
> - 了解 Sparse Flash Attention 的变体（滑动窗口、分组注意力等）
> - 对比 PyTorch SDPA 与 TileLang FlashAttention 的性能差异
> - 掌握常见调试技巧与性能优化方法

---

## 目录

- [9.1 FlashAttention 算法背景](#91-flashattention-算法背景)
- [9.2 Online Softmax 数学推导](#92-online-softmax-数学推导)
- [9.3 FlashAttention-1 vs FlashAttention-2 改进](#93-flashattention-1-vs-flashattention-2-改进)
- [9.4 TileLang 实现 FlashAttention 完整代码](#94-tilelang-实现-flashattention-完整代码)
- [9.5 Tiling 策略详解](#95-tiling-策略详解)
- [9.6 因果掩码处理](#96-因果掩码处理)
- [9.7 反向传播与重计算](#97-反向传播与重计算)
- [9.8 Sparse Flash Attention](#98-sparse-flash-attention)
- [9.9 性能对比](#99-性能对比)
- [9.10 FlashAttention-3 前瞻](#910-flashattention-3-前瞻)
- [9.11 常见调试问题与解决方案](#911-常见调试问题与解决方案)
- [9.12 练习题](#912-练习题)
- [9.13 总结](#913-总结)

---

## 9.1 FlashAttention 算法背景

### 9.1.1 标准 Attention 的计算流程

Transformer 的核心是 Self-Attention 机制。给定输入序列 $X \in \mathbb{R}^{N \times d}$，通过线性投影得到 Query、Key、Value 三个矩阵：

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

其中 $N$ 是序列长度，$d$ 是头维度（head dimension）。标准 Attention 的计算公式为：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

具体计算步骤如下：

| 步骤 | 操作 | 输出维度 | 计算量 |
|------|------|----------|--------|
| 1 | 计算 $S = QK^T$ | $N \times N$ | $O(N^2 d)$ |
| 2 | 缩放 $S = S / \sqrt{d_k}$ | $N \times N$ | $O(N^2)$ |
| 3 | 应用掩码（可选） | $N \times N$ | $O(N^2)$ |
| 4 | 计算 $P = \text{softmax}(S)$ | $N \times N$ | $O(N^2)$ |
| 5 | 计算 $O = PV$ | $N \times d$ | $O(N^2 d)$ |

### 9.1.2 内存瓶颈分析

标准 Attention 的关键问题在于中间矩阵 $S = QK^T$ 的大小为 $N \times N$。当序列长度增长时：

| 序列长度 $N$ | $S$ 矩阵大小 (FP16) | 实际内存占用 |
|-------------|---------------------|-------------|
| 1,024 | 1M 元素 | 2 MB |
| 4,096 | 16M 元素 | 32 MB |
| 16,384 | 268M 元素 | 512 MB |
| 65,536 | 4.3G 元素 | 8 GB |
| 131,072 | 17.2G 元素 | 32 GB |

这意味着在处理长序列时，内存带宽成为主要瓶颈，而非计算能力。

```
标准 Attention 的内存访问模式：
┌──────────────────────────────────────────────────┐
│  HBM (高带宽内存)                                 │
│  ┌─────┐  ┌─────┐  ┌─────┐                       │
│  │  Q  │  │  K  │  │  V  │  ← 从 HBM 读取        │
│  └──┬──┘  └──┬──┘  └──┬──┘                       │
│     │        │        │                           │
│     ▼        ▼        │                           │
│  ┌──────────────┐     │                           │
│  │  S = QK^T    │     │  ← N×N 矩阵写入 HBM       │
│  │  (N×N, FP16) │     │                           │
│  └──────┬───────┘     │                           │
│         │             │                           │
│         ▼             │                           │
│  ┌──────────────┐     │                           │
│  │  P = softmax │     │  ← 又一个 N×N 矩阵写入 HBM │
│  │  (N×N, FP16) │     │                           │
│  └──────┬───────┘     │                           │
│         │             │                           │
│         ▼             ▼                           │
│  ┌──────────────────────┐                         │
│  │    O = PV            │  ← 最终输出写入 HBM      │
│  │    (N×d, FP16)       │                         │
│  └──────────────────────┘                         │
└──────────────────────────────────────────────────┘

总 HBM 访问量：O(N² + Nd) ← 远超计算量本身
```

上图直观展示了标准 Attention 在 GPU 内存层次结构中的数据流动路径。从 HBM 读取 Q、K、V 矩阵后，需要将中间计算结果 $S = QK^T$（一个 $N \times N$ 的矩阵）写回 HBM，接着再读回来做 softmax 得到 $P$ 矩阵（又是一个 $N \times N$ 的矩阵），最后才计算输出 $O = PV$。整个过程中，$S$ 和 $P$ 两个大矩阵的读写占据了绝大部分 HBM 带宽。当序列长度 $N$ 增大时，这两个矩阵的大小以 $O(N^2)$ 的速度增长，而 GPU 的 HBM 带宽是固定不变的，这就形成了严重的内存带宽瓶颈。在实际的大模型推理中，GPU 的计算能力（TFLOPS）往往远超实际需求，但 HBM 带宽却成为限制吞吐量的"短板"。这种现象被称为"内存带宽受限"（memory-bound），是 FlashAttention 要解决的核心问题。理解这个数据流图对于后续理解 FlashAttention 的优化策略至关重要——FlashAttention 的本质就是通过巧妙的分块计算，将这些不必要的 HBM 读写操作"折叠"到 SRAM 中完成。

### 9.1.3 IO-Aware 动机

FlashAttention 的核心思想是 **IO-Aware**（IO 感知）算法设计。传统算法优化关注计算复杂度（FLOPs），但在现代 GPU 上，内存带宽往往才是真正的瓶颈。

> [!TIP]
> **GPU 内存层次结构**
>
> 现代 GPU（如 A100）的内存层次：
> - **SRAM（片上缓存）**：~20 MB，带宽 ~19 TB/s
> - **HBM（高带宽内存）**：~80 GB，带宽 ~2 TB/s
>
> SRAM 带宽是 HBM 的约 **9.5 倍**。FlashAttention 的核心策略是：将数据分块加载到 SRAM 中完成计算，避免将 $N \times N$ 的中间矩阵写回 HBM。

```
FlashAttention 的内存访问优化：

标准 Attention:                          FlashAttention:
┌─────────────────────┐                  ┌─────────────────────┐
│ HBM ← S (N×N)      │                  │ SRAM 中分块计算       │
│ HBM ← P (N×N)      │                  │ 不写入 S 到 HBM       │
│ HBM → 读 S, 写 P   │                  │ 不写入 P 到 HBM       │
│ HBM → 读 P, V, 写 O │                  │ 仅写入最终 O 到 HBM    │
│                     │                  │                     │
│ 总 HBM 读写: O(N²d) │                  │ 总 HBM 读写: O(N²d²/M)│
└─────────────────────┘                  └─────────────────────┘
                                         M = SRAM 大小
                                         当 M >> d 时，大幅减少 IO
```

这个对比图清晰地展示了 FlashAttention 的核心优化原理。在标准 Attention 中，$S$ 矩阵和 $P$ 矩阵都需要完整地写入 HBM 然后再读取，导致大量的内存带宽浪费。而 FlashAttention 通过将计算分块（tiling），每次只从 HBM 加载一小块 Q、K、V 数据到 SRAM（GPU 片上缓存，也称为共享内存），在 SRAM 中完成所有的注意力分数计算、softmax 归一化和加权求和，最终只将结果 $O$ 写回 HBM。SRAM 的带宽约为 HBM 的 10 倍（A100 上 SRAM 约 19 TB/s，HBM 约 2 TB/s），因此在 SRAM 中完成中间计算可以大幅减少数据搬运时间。公式 $O(N^2 d^2 / M)$ 表示 FlashAttention 的 HBM 访问量，其中 $M$ 是 SRAM 的大小。当 SRAM 大小 $M$ 远大于头维度 $d$ 时（实际中 $M \approx 100\text{KB}$，$d \approx 128$），$d^2/M \ll 1$，IO 减少效果非常显著。这就是 IO-Aware 算法设计的精髓：不减少计算量（FLOPs 不变），而是大幅减少数据搬运量。

### 9.1.4 计算复杂度与 IO 复杂度对比

| 指标 | 标准 Attention | FlashAttention |
|------|---------------|----------------|
| 计算复杂度 (FLOPs) | $O(N^2 d)$ | $O(N^2 d)$ |
| 内存复杂度 | $O(N^2 + Nd)$ | $O(N)$（无需存储 $N \times N$ 矩阵） |
| HBM 读写量 | $O(N^2 d + N^2)$ | $O(N^2 d^2 / M)$ |
| 是否需要重计算 | 否 | 反向传播时需要重计算 |

> [!WARNING]
> FlashAttention **不减少** FLOPs，甚至在反向传播中由于重计算会增加 FLOPs。它的优势在于大幅减少 HBM 访问量，在实际运行中获得显著加速。

---

## 9.2 Online Softmax 数学推导

### 9.2.1 标准 Softmax 回顾

对于向量 $x = [x_1, x_2, \ldots, x_n]$，标准 softmax 定义为：

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

这个计算需要两遍扫描（two-pass）：
1. **第一遍**：计算 $\max(x)$ 和 $\sum_{j} e^{x_j - \max(x)}$
2. **第二遍**：计算每个元素的 softmax 值

### 9.2.2 Online Softmax 算法

Online Softmax 是一种 **单遍扫描**（single-pass）算法，允许我们逐块处理数据，同时维护全局统计量。

**核心思想**：当我们处理新的数据块时，用一个 **rescaling factor**（重缩放因子）来修正之前块的累积结果。

**数学推导**：

假设我们将向量 $x$ 分成两部分 $x^{(1)}$ 和 $x^{(2)}$。

**处理第一部分 $x^{(1)}$**：

$$
m^{(1)} = \max(x^{(1)})
$$

$$
\ell^{(1)} = \sum_{j \in \text{block 1}} e^{x_j - m^{(1)}}
$$

**处理第二部分 $x^{(2)}$**：

$$
m^{(2)} = \max(m^{(1)}, \max(x^{(2)}))
$$

关键步骤：更新第一部分的累积和，加上重缩放因子：

$$
\ell^{(2)} = e^{m^{(1)} - m^{(2)}} \cdot \ell^{(1)} + \sum_{j \in \text{block 2}} e^{x_j - m^{(2)}}
$$

这里 $e^{m^{(1)} - m^{(2)}}$ 就是重缩放因子。由于 $m^{(2)} \geq m^{(1)}$，这个因子 $\leq 1$，保证了数值稳定性。

**输出更新**：当处理到新的块时，之前累积的输出 $o^{(1)}$ 也需要重新缩放：

$$
o^{(2)} = \frac{e^{m^{(1)} - m^{(2)}} \cdot \ell^{(1)} \cdot o^{(1)} + \text{new\_block\_contribution}}{\ell^{(2)}}
$$

### 9.2.3 递推公式的严格证明

我们用数学归纳法证明 Online Softmax 的正确性。

**命题**：经过处理前 $k$ 个块后，累积的 $\ell^{(k)}$ 和 $o^{(k)}$ 与标准 softmax 的结果一致。

**基础情况**（$k=1$）：显然成立。

**归纳步骤**：假设对前 $k$ 个块成立，证明对 $k+1$ 个块也成立。

设前 $k$ 个块的全局 max 为 $m^{(k)}$，第 $k+1$ 个块的 max 为 $m_{\text{new}}$。

$$
m^{(k+1)} = \max(m^{(k)}, m_{\text{new}})
$$

更新累积和：

$$
\ell^{(k+1)} = e^{m^{(k)} - m^{(k+1)}} \cdot \ell^{(k)} + \sum_{j \in \text{block } k+1} e^{x_j - m^{(k+1)}}
$$

展开 $\ell^{(k)}$：

$$
\ell^{(k)} = \sum_{j=1}^{N_k} e^{x_j - m^{(k)}}
$$

代入得：

$$
\ell^{(k+1)} = e^{m^{(k)} - m^{(k+1)}} \sum_{j=1}^{N_k} e^{x_j - m^{(k)}} + \sum_{j=N_k+1}^{N_{k+1}} e^{x_j - m^{(k+1)}}
$$

$$
= \sum_{j=1}^{N_k} e^{x_j - m^{(k+1)}} + \sum_{j=N_k+1}^{N_{k+1}} e^{x_j - m^{(k+1)}}
$$

$$
= \sum_{j=1}^{N_{k+1}} e^{x_j - m^{(k+1)}}
$$

这正是对全部 $N_{k+1}$ 个元素以全局 max $m^{(k+1)}$ 为基准的 softmax 分母。

### 9.2.4 Online Softmax 的伪代码

```python
# Online Softmax 伪代码（Python 注释说明）
def online_softmax(x_blocks):
    """
    在线计算 softmax，逐块处理输入数据。
    不需要事先知道全局最大值。
    """
    # 初始化全局统计量
    m_prev = -float('inf')  # 当前已见最大值
    l_prev = 0.0             # 当前累积的 softmax 分母
    o_prev = 0.0             # 当前累积的输出

    for block in x_blocks:
        # 计算当前块的局部最大值
        m_block = max(block)

        # 更新全局最大值
        m_new = max(m_prev, m_block)

        # 计算重缩放因子：修正之前累积的贡献
        rescale_old = exp(m_prev - m_new)

        # 更新累积分母
        l_new = rescale_old * l_prev + sum(exp(x - m_new) for x in block)

        # 更新累积输出（先重缩放旧输出，再加新贡献）
        o_new = (rescale_old * l_prev * o_prev +
                 sum(exp(x - m_new) * v for x, v in zip(block, values))) / l_new

        # 更新状态
        m_prev = m_new
        l_prev = l_new
        o_prev = o_new

    return o_prev
```

上述伪代码展示了 Online Softmax 的核心实现逻辑。算法维护三个状态变量：`m_prev`（历史最大值）、`l_prev`（累积归一化因子）和 `o_prev`（累积输出）。对于每个新数据块，首先计算局部最大值并更新全局最大值，然后通过重缩放因子 `rescale_old = exp(m_prev - m_new)` 修正之前的累积结果。这种单遍扫描的方式使得 FlashAttention 可以在不存储完整注意力矩阵的情况下逐块计算 softmax，是 FlashAttention 能够将内存复杂度从 $O(N^2)$ 降低到 $O(N)$ 的关键数学基础。

这段伪代码中几个关键的设计决策值得注意。首先，`m_prev` 初始化为负无穷大（`-float('inf')`），确保第一个数据块的局部最大值能正确覆盖初始值。其次，重缩放因子 `rescale_old = exp(m_prev - m_new)` 的值始终在 $(0, 1]$ 范围内，因为 `m_new` 是 `m_prev` 和当前块最大值中的较大者，这意味着旧的累积结果只会被"缩小"而不会被放大，从数学上保证了数值稳定性。在 TileLang 实现中，这些操作需要逐元素进行，每个线程负责 Q 矩阵的一行，独立维护自己的 `m`、`l` 和 `o` 状态。这种行级并行的设计是 FlashAttention 能高效利用 GPU 大量线程的关键。在实际实现中，还需要注意浮点精度问题——建议使用 float32 来维护这些状态变量，以避免半精度累加带来的精度损失。

### 9.2.5 数值稳定性分析

> [!CAUTION]
> **数值溢出风险**
>
> 直接计算 $e^{x_i}$ 会导致数值溢出。Online Softmax 通过减去当前最大值来保证数值稳定性。

**关键性质**：对于任意时刻，我们计算的指数项 $e^{x_j - m^{(k)}}$ 满足：

$$
x_j - m^{(k)} \leq 0 \quad \forall j
$$

因此 $e^{x_j - m^{(k)}} \in (0, 1]$，永远不会溢出。

**重缩放因子的范围**：

$$
0 < e^{m^{(k)} - m^{(k+1)}} \leq 1
$$

因为 $m^{(k+1)} \geq m^{(k)}$（最大值单调递增），所以重缩放因子也在 $(0, 1]$ 范围内，不会引入数值不稳定性。

### 9.2.6 Online Softmax 与 FlashAttention 的关系

FlashAttention 将 Online Softmax 推广到 **矩阵乘法** 的场景：

| Online Softmax | FlashAttention |
|----------------|----------------|
| 输入向量 $x$ | 输入矩阵 $Q, K, V$ |
| 逐块处理标量 | 逐块处理 $Q$ 的行块和 $K, V$ 的列块 |
| 维护 $m, \ell$ | 维护 $m_i, \ell_i$（每行独立） |
| 输出标量 | 输出矩阵 $O$ |

在 FlashAttention 中，对于 $Q$ 的每一行 $q_i$，我们独立维护：
- $m_i$：当前已见的 $\max_j (q_i^T k_j / \sqrt{d})$
- $\ell_i$：当前的 $\sum_j e^{q_i^T k_j / \sqrt{d} - m_i}$
- $o_i$：当前累积的输出向量

---

## 9.3 FlashAttention-1 vs FlashAttention-2 改进

### 9.3.1 FlashAttention-1 核心设计

FlashAttention-1（Dao et al., 2022）的核心贡献：

**算法 1：FlashAttention 前向传播**

```
输入: Q, K, V ∈ R^{N×d}, SRAM 大小 M
输出: O ∈ R^{N×d}
1: 将 Q 分成 T_q = ⌈N/B_m⌉ 个块，每块大小 B_m × d
2: 将 K, V 分成 T_kv = ⌈N/B_n⌉ 个块，每块大小 B_n × d
3: 初始化 O = 0, ℓ = 0, m = -∞
4: for j = 1 to T_kv do  // 遍历 K/V 块
5:   从 HBM 加载 K_j, V_j 到 SRAM
6:   for i = 1 to T_q do  // 遍历 Q 块
7:     从 HBM 加载 Q_i, O_i, ℓ_i, m_i 到 SRAM
8:     在 SRAM 中计算 S_ij = Q_i K_j^T / √d
9:     计算 m_ij = rowmax(S_ij)
10:    计算 m_new = max(m_i, m_ij)
11:    计算 P_ij = exp(S_ij - m_new)
12:    计算 ℓ_new = exp(m_i - m_new) * ℓ_i + rowsum(P_ij)
13:    更新 O_i = diag(ℓ_new)^{-1} (diag(exp(m_i - m_new)) * ℓ_i * O_i + P_ij * V_j)
14:    更新 m_i = m_new, ℓ_i = ℓ_new
15:    将更新后的 O_i, ℓ_i, m_i 写回 HBM
16:  end for
17: end for
18: 返回 O
```

这段伪代码是 FlashAttention-1 的完整算法描述，是理解整个 FlashAttention 系列算法的基础。算法的核心是一个双重循环结构：外层遍历 K/V 块（第 5 行），内层遍历 Q 块（第 7 行）。每次外层迭代加载一个新的 K/V 块到 SRAM，内层迭代则对每个 Q 块执行注意力计算和 Online Softmax 更新。关键步骤在第 8-14 行：先计算注意力分数 $S_{ij}$，然后找到当前块的行最大值 $m_{ij}$，与已有的最大值 $m_i$ 取最大得到 $m_{new}$，接着计算 softmax 分子 $P_{ij} = \exp(S_{ij} - m_{new})$，更新累积分母 $\ell_{new}$，最后用重缩放后的旧输出加上新贡献更新 $O_i$。第 15 行将更新后的结果写回 HBM——这是 FlashAttention-1 的一个重要特点：每次内层迭代都会写回中间结果。这个设计虽然简单，但引入了不必要的 HBM 写操作，也是 FlashAttention-2 后来改进的重点方向之一。在 TileLang 中实现时，这个循环结构可以直接映射为嵌套的 `for` 循环，SRAM 缓冲区通过 `T.alloc_fragment` 声明。

**关键设计**：
- 外层循环遍历 K/V 块，内层循环遍历 Q 块
- 每次只加载一个块到 SRAM
- 使用 Online Softmax 更新统计量
- 无需将 $N \times N$ 的 $S$ 或 $P$ 矩阵写入 HBM

### 9.3.2 FlashAttention-2 的改进

FlashAttention-2（Dao, 2023）在 FlashAttention-1 的基础上做了三项关键改进：

**改进 1：减少非矩阵乘法 FLOPs**

FlashAttention-1 中，内层循环中需要多次 rescaling 操作。FlashAttention-2 通过延迟 rescaling 来减少这些操作：

```
FlashAttention-1: 每次内层循环迭代都 rescaling
┌──────────────────────────────────────────┐
│ for j:          // K/V 块               │
│   for i:        // Q 块                 │
│     rescale O_i ← 每次都乘以缩放因子       │  ← 大量非 matmul FLOPs
│     update m_i, ℓ_i                      │
│     O_i += P_ij * V_j                    │
└──────────────────────────────────────────┘

FlashAttention-2: 延迟 rescaling 到外层循环末尾
┌──────────────────────────────────────────┐
│ for j:          // K/V 块               │
│   for i:        // Q 块                 │
│     更新 m_i (不立即 rescale O_i)          │  ← 减少非 matmul FLOPs
│     O_i += P_ij * V_j                    │
│   end for                                │
│   // 外层循环结束后才 rescale               │
└──────────────────────────────────────────┘
```

**改进 2：更好的并行化策略**

FlashAttention-1 的循环结构：
- 外层：遍历 K/V 块
- 内层：遍历 Q 块

这限制了并行化，因为外层循环有数据依赖。

FlashAttention-2 的改进：
- 外层：遍历 Q 块（可并行）
- 内层：遍历 K/V 块（顺序，因为需要维护 Online Softmax 状态）

```
FlashAttention-1 并行化:              FlashAttention-2 并行化:
┌─────────────────────────┐          ┌─────────────────────────┐
│ for j (K/V 块):         │          │ for i (Q 块):           │
│   parallel for i (Q块): │          │   for j (K/V 块):       │
│     ...                 │          │     ...                 │   ← Q 块可并行
│                         │          │   end for               │
└─────────────────────────┘          └─────────────────────────┘
外层有依赖，内层可并行               外层可并行，内层有依赖
→ 并行度受 K/V 块数限制              → 并行度受 Q 块数限制（通常更大）
```

**改进 3：更好的工作分区**

在 Warp 级别，FlashAttention-2 对 Q、K、V 的分块做了更细致的划分：

```
GPU Warp 级分区:

┌──────────────────────────────────────────────┐
│ 一个 Thread Block 处理一个 Q 块               │
│                                              │
│  ┌─────────────────────────────────────────┐ │
│  │ Warp 0  │ Warp 1  │ Warp 2  │ Warp 3   │ │
│  │ Q[0:16] │ Q[16:32]│ Q[32:48]│ Q[48:64] │ │
│  └────────┬────────────────────────────────┘ │
│           │                                  │
│           ▼                                  │
│  每个 Warp 独立处理自己负责的 Q 行             │
│  共享 K/V 块的加载                            │
│  独立维护 m_i, ℓ_i 状态                      │
└──────────────────────────────────────────────┘
```

这个代码块或示意图用于说明 9.3.2 FlashAttention-2 的改进 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 9.3.3 FlashAttention-1 vs FlashAttention-2 对比

| 特性 | FlashAttention-1 | FlashAttention-2 |
|------|-------------------|-------------------|
| 非 matmul FLOPs | 较高 | 减少约 50% |
| 并行度 | 受 K/V 块数限制 | 受 Q 块数限制 |
| A100 利用率 | ~50-70% | ~70-73% |
| Warp 分区 | 简单 | 精细优化 |
| 反向传播 | 需要保存 RNG 状态 | 优化 RNG 状态管理 |
| 支持头维度 | d ≤ 128 | d ≤ 256 |

### 9.3.4 FlashAttention-2 的循环结构伪代码

```python
# FlashAttention-2 前向传播伪代码
def flash_attention_2_forward(Q, K, V, Br, Bc):
    """
    Q: [N, d] - Query 矩阵
    K: [N, d] - Key 矩阵
    V: [N, d] - Value 矩阵
    Br: Q 块的行数（tile size for Q）
    Bc: K/V 块的行数（tile size for K/V）
    """
    N, d = Q.shape
    Tr = ceil(N / Br)  # Q 块的数量
    Tc = ceil(N / Bc)  # K/V 块的数量

    # 初始化输出和统计量
    O = zeros(N, d)     # 输出矩阵
    l = zeros(N, 1)     # softmax 分母（每行）
    m = full(N, 1, -inf) # 每行的最大值

    # 外层并行遍历 Q 块（FlashAttention-2 的关键改进）
    for i in range(Tr):  # 并行：每个 Q 块独立处理
        # 加载当前 Q 块及其统计量
        Qi = Q[i*Br : (i+1)*Br, :]    # [Br, d]
        Oi = O[i*Br : (i+1)*Br, :]    # [Br, d]
        li = l[i*Br : (i+1)*Br, :]    # [Br, 1]
        mi = m[i*Br : (i+1)*Br, :]    # [Br, 1]

        # 临时变量：用于延迟 rescaling
        mi_new = mi.copy()
        li_new = li.copy()

        # 内层顺序遍历 K/V 块
        for j in range(Tc):
            # 加载当前 K/V 块
            Kj = K[j*Bc : (j+1)*Bc, :]  # [Bc, d]
            Vj = V[j*Bc : (j+1)*Bc, :]  # [Bc, d]

            # 计算注意力分数 S_ij = Qi @ Kj^T / sqrt(d)
            Sij = Qi @ Kj.T / sqrt(d)    # [Br, Bc]

            # 因果掩码处理
            if causal:
                apply_causal_mask(Sij, i, j, Br, Bc)

            # 计算当前块的行最大值
            mij = rowmax(Sij)             # [Br, 1]

            # 更新全局最大值
            mi_new_prev = mi_new.copy()
            mi_new = maximum(mi_new_prev, mij)

            # 计算 softmax 分子（减去新最大值）
            Pij = exp(Sij - mi_new)       # [Br, Bc]

            # 更新分母
            lij = rowsum(Pij)             # [Br, 1]
            li_new = exp(mi_new_prev - mi_new) * li_new + lij

            # 累加输出（注意：这里暂时不 rescale，延迟到后面）
            Oi = exp(mi_new_prev - mi_new) * Oi + Pij @ Vj

        # 外层循环结束后，统一 rescaling
        Oi = Oi / li_new

        # 写回结果
        O[i*Br : (i+1)*Br, :] = Oi
        l[i*Br : (i+1)*Br, :] = li_new
        m[i*Br : (i+1)*Br, :] = mi_new

    return O
```

这段伪代码展示了 FlashAttention-2 的前向传播完整流程。与 FlashAttention-1 相比，最核心的改进是将外层循环改为遍历 Q 块（可并行），内层循环遍历 K/V 块（顺序执行）。延迟 rescaling 策略将缩放操作从内层循环移到外层循环末尾，减少了约 50% 的非矩阵乘法 FLOPs。每个 Q 块独立维护自己的统计量 `mi_new` 和 `li_new`，在遍历完所有 K/V 块后统一进行归一化。这种循环结构的调整使得 FlashAttention-2 在 A100 上的 GPU 利用率从约 50-70% 提升到 70-73%。

从 TileLang 实现的角度来看，这段伪代码有几个重要的设计要点。第一，外层并行意味着每个 Q 块可以被分配到一个独立的 GPU 线程块（block），通过 `T.thread_binding(0, T_q, thread="blockIdx.x")` 实现，这充分利用了 GPU 的大规模并行能力。第二，内层顺序遍历 K/V 块意味着每个线程块需要顺序处理所有 K/V 块，这保证了 Online Softmax 状态的正确更新——因为每行的 $m$ 和 $\ell$ 需要按顺序累积。第三，延迟 rescaling 策略（第 520 行）是一个巧妙的优化：在内层循环中，我们不立即对 $O_i$ 做归一化除法，而是将除法推迟到第 523 行外层循环结束后统一执行。这避免了在每次内层迭代中都进行一次除法运算，而除法在 GPU 上是相对昂贵的操作。在 TileLang 中，这种延迟归一化可以直接用累加实现，最后一次性完成除法。

---

## 9.4 TileLang 实现 FlashAttention 完整代码

### 9.4.1 实现概览

本节我们将使用 TileLang 实现一个完整的 FlashAttention 前向内核。实现将包含：

- Q、K、V 的 Tiling
- Online Softmax 的精确实现
- 因果掩码支持
- 输出累积与 rescaling

### 9.4.2 完整 TileLang 代码

```python
# FlashAttention 的 TileLang 完整实现
# 本代码实现了 FlashAttention-2 风格的前向传播

import tilelang
from tilelang import language as T
import torch

@T.prim_func
def flash_attention_kernel(
    # 输入张量参数定义
    Q: T.Buffer((128, 64), "float16"),      # Query 矩阵：[seq_len, head_dim]
    K: T.Buffer((128, 64), "float16"),      # Key 矩阵：[seq_len, head_dim]
    V: T.Buffer((128, 64), "float16"),      # Value 矩阵：[seq_len, head_dim]
    O: T.Buffer((128, 64), "float16"),      # 输出矩阵：[seq_len, head_dim]
):
    """
    FlashAttention 前向传播的 TileLang 内核。

    算法流程：
    1. 将 Q 按行分块（每块 BM 行）
    2. 外层循环遍历 Q 块
    3. 内层循环遍历 K/V 块
    4. 在 SRAM 中计算分块注意力并使用 Online Softmax 更新

    参数说明：
    - BM: Q 的 tile 大小（Block size for M dimension）
    - BN: K/V 的 tile 大小（Block size for N dimension）
    - BK: 内积维度的 tile 大小（等于 head_dim）
    """

    # ==================== 常量定义 ====================
    # 序列长度和头维度
    SEQ_LEN = 128       # 序列长度 N
    HEAD_DIM = 64       # 头维度 d
    BM = 32             # Q 块的行数（每次处理 32 行 Q）
    BN = 32             # K/V 块的行数（每次处理 32 行 K/V）

    # 循环次数计算
    # T_q = ceil(SEQ_LEN / BM) = ceil(128 / 32) = 4
    # T_kv = ceil(SEQ_LEN / BN) = ceil(128 / 32) = 4
    T_q = 4             # Q 块的数量
    T_kv = 4            # K/V 块的数量

    # softmax 温度倒数：1/sqrt(d)
    SCALE = 0.125       # 1/sqrt(64) = 0.125

    # ==================== SRAM 缓冲区分配 ====================
    # 以下缓冲区分配在 GPU 的 SRAM（共享内存）中
    # 用于存放当前正在处理的 tile 数据

    # Q 块缓冲区：存放当前 Q 块 [BM, HEAD_DIM]
    Q_tile = T.alloc_fragment((BM, HEAD_DIM), "float16")

    # K 块缓冲区：存放当前 K 块 [BN, HEAD_DIM]
    K_tile = T.alloc_fragment((BN, HEAD_DIM), "float16")

    # V 块缓冲区：存放当前 V 块 [BN, HEAD_DIM]
    V_tile = T.alloc_fragment((BN, HEAD_DIM), "float16")

    # 注意力分数矩阵：S = Q @ K^T [BM, BN]
    S_tile = T.alloc_fragment((BM, BN), "float32")

    # Softmax 分子：P = exp(S - m) [BM, BN]
    P_tile = T.alloc_fragment((BM, BN), "float32")

    # 输出累积缓冲区 [BM, HEAD_DIM]
    O_tile = T.alloc_fragment((BM, HEAD_DIM), "float32")

    # Online Softmax 统计量
    m_tile = T.alloc_fragment((BM, 1), "float32")      # 每行当前最大值
    l_tile = T.alloc_fragment((BM, 1), "float32")      # 每行累积分母
    m_new_tile = T.alloc_fragment((BM, 1), "float32")  # 更新后的最大值

    # 临时缓冲区：用于 rescaling
    scale_old = T.alloc_fragment((BM, 1), "float32")   # 旧的 rescaling 因子

    # ==================== 线程绑定与并行策略 ====================
    # 使用 T.ThreadBinding 将计算绑定到 GPU 线程
    # bx: Q 块索引（blockIdx.x）
    # tx: 块内线程索引（threadIdx.x）
    for bx in T.thread_binding(0, T_q, thread="blockIdx.x"):
        for tx in T.thread_binding(0, BM, thread="threadIdx.x"):

            # ==================== 初始化阶段 ====================
            # 初始化 Online Softmax 的统计量
            # m 初始化为负无穷（表示尚未见过任何有效值）
            m_tile[tx, 0] = T.float32(-1e30)
            # l 初始化为 0（累积分母）
            l_tile[tx, 0] = T.float32(0.0)
            # O 初始化为 0
            for d in range(HEAD_DIM):
                O_tile[tx, d] = T.float32(0.0)

            # ==================== 加载 Q 块 ====================
            # 从 HBM 加载当前 Q 块到 SRAM
            # Q_tile[tx, :] = Q[bx * BM + tx, :]
            for d in range(HEAD_DIM):
                Q_tile[tx, d] = Q[bx * BM + tx, d]

            # ==================== 内层循环：遍历 K/V 块 ====================
            # 顺序遍历所有 K/V 块，维护 Online Softmax 状态
            for j in range(T_kv):

                # ---------- 步骤 1: 加载 K/V 块到 SRAM ----------
                # 从 HBM 加载第 j 个 K 块
                for d in range(HEAD_DIM):
                    K_tile[tx, d] = K[j * BN + tx, d]

                # 从 HBM 加载第 j 个 V 块
                for d in range(HEAD_DIM):
                    V_tile[tx, d] = V[j * BN + tx, d]

                # ---------- 步骤 2: 计算注意力分数 S = Q @ K^T / sqrt(d) ----------
                # S_tile[tx, k] = sum_d(Q_tile[tx, d] * K_tile[k, d]) * SCALE
                # 这是一个矩阵乘法：[BM, HEAD_DIM] @ [HEAD_DIM, BN] = [BM, BN]
                for k in range(BN):
                    S_tile[tx, k] = T.float32(0.0)
                    for d in range(HEAD_DIM):
                        S_tile[tx, k] += T.float32(Q_tile[tx, d]) * T.float32(K_tile[k, d])
                    S_tile[tx, k] *= SCALE  # 缩放：除以 sqrt(d)

                # ---------- 步骤 3: 应用因果掩码 ----------
                # 对于因果注意力，位置 i 只能注意到位置 <= i 的 token
                # 如果 j * BN + k > bx * BM + tx，则 mask 为 -inf
                for k in range(BN):
                    # 全局行索引和列索引
                    row_idx = bx * BM + tx    # Q 的行索引
                    col_idx = j * BN + k      # K 的列索引
                    # 如果列索引大于行索引，施加 mask
                    if col_idx > row_idx:
                        S_tile[tx, k] = T.float32(-1e30)

                # ---------- 步骤 4: Online Softmax 更新 ----------
                # 计算当前块的行最大值
                m_local = T.float32(-1e30)
                for k in range(BN):
                    m_local = T.max(m_local, S_tile[tx, k])

                # 更新全局最大值
                m_new_tile[tx, 0] = T.max(m_tile[tx, 0], m_local)

                # 计算旧的 rescaling 因子
                # scale_old = exp(m_prev - m_new)
                # 用于缩放之前累积的 O 和 l
                scale_old[tx, 0] = T.exp(m_tile[tx, 0] - m_new_tile[tx, 0])

                # 计算新的 softmax 分子：P = exp(S - m_new)
                for k in range(BN):
                    P_tile[tx, k] = T.exp(S_tile[tx, k] - m_new_tile[tx, 0])

                # 计算当前块的分母贡献
                l_local = T.float32(0.0)
                for k in range(BN):
                    l_local += P_tile[tx, k]

                # ---------- 步骤 5: 更新累积输出 O ----------
                # 先将旧的 O 进行 rescaling
                # O_new = scale_old * O_old + P @ V
                for d in range(HEAD_DIM):
                    # rescale 旧输出
                    O_tile[tx, d] *= scale_old[tx, 0]
                    # 加上当前块的贡献：P @ V
                    for k in range(BN):
                        O_tile[tx, d] += P_tile[tx, k] * T.float32(V_tile[k, d])

                # 更新分母：l_new = scale_old * l_old + l_local
                l_tile[tx, 0] = scale_old[tx, 0] * l_tile[tx, 0] + l_local

                # 更新最大值
                m_tile[tx, 0] = m_new_tile[tx, 0]

            # ==================== 最终归一化阶段 ====================
            # 将累积的 O 除以最终的 l 得到真正的 softmax 输出
            # O_final = O / l
            for d in range(HEAD_DIM):
                O_tile[tx, d] = O_tile[tx, d] / l_tile[tx, 0]

            # ==================== 写回结果到 HBM ====================
            # 将最终结果从 SRAM 写回 HBM
            for d in range(HEAD_DIM):
                O[bx * BM + tx, d] = O_tile[tx, d]
```

这是 FlashAttention 的完整 TileLang 实现，包含了从 HBM 加载数据、计算注意力分数、应用因果掩码、执行 Online Softmax 更新到写回结果的全部流程。代码采用 FlashAttention-2 的循环结构——外层遍历 Q 块（通过 `bx` 线程块索引并行化），内层顺序遍历 K/V 块。关键的 Online Softmax 逻辑通过 `scale_old`、`m_tile` 和 `l_tile` 三个统计量实现：每次处理新 K/V 块时，先计算局部最大值，再用重缩放因子修正之前的累积输出和分母。注意力分数 S 和 softmax 权重 P 使用 float32 精度以保证数值稳定性，而输入输出使用 float16 以节省内存带宽。因果掩码通过比较全局行列索引实现，将未来位置的注意力分数设为 $-10^{30}$。

### 9.4.3 代码逐行注释详解

让我们将上述代码分解为更小的模块来详细解释每个部分：

#### 模块 1：缓冲区声明

```python
# SRAM 缓冲区分配详解
# 在 GPU 上，这些缓冲区会被映射到共享内存（Shared Memory）
# 共享内存的特点：
#   - 容量有限（A100: 164 KB per SM，可配置到 192 KB）
#   - 带宽极高（~19 TB/s）
#   - 线程块内共享

Q_tile = T.alloc_fragment((BM, HEAD_DIM), "float16")
# 用途：存放从 HBM 加载的 Q 块
# 大小：32 × 64 × 2 bytes = 4 KB
# 生命周期：在整个外层循环中保持不变（只加载一次）

K_tile = T.alloc_fragment((BN, HEAD_DIM), "float16")
# 用途：存放从 HBM 加载的 K 块
# 大小：32 × 64 × 2 bytes = 4 KB
# 生命周期：每次内层循环迭代重新加载

V_tile = T.alloc_fragment((BN, HEAD_DIM), "float16")
# 用途：存放从 HBM 加载的 V 块
# 大小：32 × 64 × 2 bytes = 4 KB
# 生命周期：每次内层循环迭代重新加载

S_tile = T.alloc_fragment((BM, BN), "float32")
# 用途：存放注意力分数 S = Q @ K^T / sqrt(d)
# 大小：32 × 32 × 4 bytes = 4 KB
# 注意：使用 float32 以保证计算精度
# 生命周期：每次内层循环迭代重新计算

P_tile = T.alloc_fragment((BM, BN), "float32")
# 用途：存放 softmax 后的注意力权重 P = exp(S - m)
# 大小：32 × 32 × 4 bytes = 4 KB
# 生命周期：每次内层循环迭代重新计算

O_tile = T.alloc_fragment((BM, HEAD_DIM), "float32")
# 用途：累积输出 O += P @ V
# 大小：32 × 64 × 4 bytes = 8 KB
# 生命周期：在整个外层循环中累积
# 注意：使用 float32 以避免累积误差

m_tile = T.alloc_fragment((BM, 1), "float32")
# 用途：Online Softmax 的每行最大值
# 大小：32 × 1 × 4 bytes = 128 bytes
# 初始值：-1e30（负无穷的近似）

l_tile = T.alloc_fragment((BM, 1), "float32")
# 用途：Online Softmax 的每行累积分母
# 大小：32 × 1 × 4 bytes = 128 bytes
# 初始值：0
```

这段缓冲区声明展示了 FlashAttention 内核所需的全部 SRAM 内存布局。所有缓冲区总计约 28.5 KB，远小于 A100 的 164 KB 共享内存限制，为增大 tile 大小留出了充足空间。注意 Q/K/V 使用 float16 以节省内存，而 S/P/O/m/l 使用 float32 以保证计算精度——这是 FlashAttention 实现中精度与效率权衡的关键设计。`alloc_fragment` 声明的缓冲区会被映射到 GPU 的寄存器或共享内存中，由编译器自动管理。

#### 模块 2：注意力分数计算

```python
# 注意力分数计算详解
# S = Q @ K^T / sqrt(d)
#
# 数学公式：S[i][k] = sum_d(Q[i][d] * K[k][d]) / sqrt(d)
#
# 在代码中：
# S_tile[tx, k] = sum_{d=0}^{HEAD_DIM-1} (Q_tile[tx, d] * K_tile[k, d]) * SCALE
#
# 这是 FlashAttention 中计算量最大的部分（O(BM × BN × d) FLOPs）
# 也是 GPU 利用率最高的部分（矩阵乘法）

for k in range(BN):
    # 初始化累加器
    S_tile[tx, k] = T.float32(0.0)

    # 内积计算：Q_tile[tx, :] · K_tile[k, :]
    # 这是一个长度为 HEAD_DIM 的向量点积
    for d in range(HEAD_DIM):
        # 类型提升：float16 → float32 以保证计算精度
        S_tile[tx, k] += T.float32(Q_tile[tx, d]) * T.float32(K_tile[k, d])

    # 缩放：除以 sqrt(d) 以防止点积值过大
    # 这是 Transformer 的标准做法，防止 softmax 进入梯度饱和区
    S_tile[tx, k] *= SCALE
```

这段代码实现了 FlashAttention 中计算量最大的部分——注意力分数矩阵 $S = QK^T / \sqrt{d_k}$ 的逐元素计算。每个线程负责计算 S 矩阵的一行，通过遍历 HEAD_DIM 维度完成向量点积。将 float16 的 Q/K 元素显式提升为 float32 进行累加是关键的精度保护措施，避免了半精度累加带来的精度损失。缩放因子 `SCALE = 1/\sqrt{d}` 防止点积值过大导致 softmax 梯度消失。在实际 GPU kernel 中，这部分通常会被映射到 Tensor Core 的 mma 指令以获得更高的计算吞吐量。

#### 模块 3：Online Softmax 更新详解

```python
# Online Softmax 更新过程详解
#
# 目标：在不重新扫描所有数据的情况下，正确计算 softmax
#
# 维护的状态：
#   m[i] = max_{k 已处理} S[i][k]     （每行的历史最大值）
#   l[i] = sum_{k 已处理} exp(S[i][k] - m[i])  （每行的累积分母）
#
# 当处理新块 j 时：
#   m_local = max_{k ∈ 块j} S[i][k]    （当前块的局部最大值）
#   m_new = max(m[i], m_local)          （更新全局最大值）
#
# 关键公式：
#   l_new = exp(m_old - m_new) * l_old + sum_{k ∈ 块j} exp(S[i][k] - m_new)
#          ↑                    ↑              ↑
#          重缩放因子          缩放旧贡献      当前块贡献

# 步骤 1: 计算当前块的行最大值
m_local = T.float32(-1e30)
for k in range(BN):
    m_local = T.max(m_local, S_tile[tx, k])
# m_local 现在是当前块中 S[tx, :] 的最大值

# 步骤 2: 更新全局最大值
m_new_tile[tx, 0] = T.max(m_tile[tx, 0], m_local)
# m_new_tile 现在是考虑当前块后的全局最大值

# 步骤 3: 计算重缩放因子
# scale_old = exp(m_old - m_new)
# 由于 m_new >= m_old，所以 scale_old ∈ (0, 1]
# 这个因子用于"缩小"之前累积的贡献
scale_old[tx, 0] = T.exp(m_tile[tx, 0] - m_new_tile[tx, 0])

# 步骤 4: 计算 softmax 分子
# P[i][k] = exp(S[i][k] - m_new)
# 减去 m_new 保证 exp 的参数 <= 0，防止溢出
for k in range(BN):
    P_tile[tx, k] = T.exp(S_tile[tx, k] - m_new_tile[tx, 0])

# 步骤 5: 计算当前块的分母贡献
l_local = T.float32(0.0)
for k in range(BN):
    l_local += P_tile[tx, k]
# l_local = sum_{k ∈ 块j} exp(S[i][k] - m_new)

# 步骤 6: 更新累积输出
# O_new = scale_old * O_old + P @ V
# 注意：这里先缩放旧的 O，再加上新贡献
for d in range(HEAD_DIM):
    O_tile[tx, d] *= scale_old[tx, 0]  # 缩放旧输出
    for k in range(BN):
        O_tile[tx, d] += P_tile[tx, k] * T.float32(V_tile[k, d])  # 加新贡献

# 步骤 7: 更新累积分母
# l_new = scale_old * l_old + l_local
l_tile[tx, 0] = scale_old[tx, 0] * l_tile[tx, 0] + l_local

# 步骤 8: 更新最大值
m_tile[tx, 0] = m_new_tile[tx, 0]
```

这段代码完整展示了 Online Softmax 的八步更新过程，是 FlashAttention 算法的数学核心。其关键思想是：当处理新的 K/V 块时，全局最大值可能发生变化，需要通过重缩放因子 `exp(m_old - m_new)` 修正之前所有块的累积贡献。由于 $m_{new} \geq m_{old}$，重缩放因子始终在 $(0, 1]$ 范围内，保证了数值稳定性。最终的归一化只需在所有 K/V 块处理完毕后执行一次除法 $O = O / l$，避免了每步都进行除法运算。这种设计使得 FlashAttention 在保持数学等价性的同时，大幅减少了 HBM 访问量。

### 9.4.4 完整的 TileLang 编译与执行

```python
import tilelang
from tilelang import language as T
import torch

def create_flash_attention_kernel(BM=32, BN=32, SEQ_LEN=128, HEAD_DIM=64, causal=True):
    """
    创建 FlashAttention 的 TileLang 内核。

    参数：
        BM: Q 块的行数
        BN: K/V 块的行数
        SEQ_LEN: 序列长度
        HEAD_DIM: 头维度
        causal: 是否使用因果掩码

    返回：
        编译后的 TileLang 内核函数
    """

    @T.prim_func
    def flash_attn(
        Q: T.Buffer((SEQ_LEN, HEAD_DIM), "float16"),
        K: T.Buffer((SEQ_LEN, HEAD_DIM), "float16"),
        V: T.Buffer((SEQ_LEN, HEAD_DIM), "float16"),
        O: T.Buffer((SEQ_LEN, HEAD_DIM), "float16"),
    ):
        # 常量定义
        SCALE = 1.0 / (HEAD_DIM ** 0.5)
        T_q = T.ceildiv(SEQ_LEN, BM)    # Q 块数
        T_kv = T.ceildiv(SEQ_LEN, BN)   # K/V 块数

        # SRAM 缓冲区
        Q_tile = T.alloc_fragment((BM, HEAD_DIM), "float16")
        K_tile = T.alloc_fragment((BN, HEAD_DIM), "float16")
        V_tile = T.alloc_fragment((BN, HEAD_DIM), "float16")
        S_tile = T.alloc_fragment((BM, BN), "float32")
        P_tile = T.alloc_fragment((BM, BN), "float32")
        O_tile = T.alloc_fragment((BM, HEAD_DIM), "float32")
        m_tile = T.alloc_fragment((BM, 1), "float32")
        l_tile = T.alloc_fragment((BM, 1), "float32")
        m_new_tile = T.alloc_fragment((BM, 1), "float32")

        # 并行策略：每个线程块处理一个 Q 块
        for bx in T.thread_binding(0, T_q, thread="blockIdx.x"):
            for tx in T.thread_binding(0, BM, thread="threadIdx.x"):
                # 初始化
                m_tile[tx, 0] = T.float32(-1e30)
                l_tile[tx, 0] = T.float32(0.0)
                for d in range(HEAD_DIM):
                    O_tile[tx, d] = T.float32(0.0)

                # 加载 Q 块（只加载一次）
                for d in range(HEAD_DIM):
                    Q_tile[tx, d] = Q[bx * BM + tx, d]

                # 遍历 K/V 块
                for j in range(T_kv):
                    # 加载 K, V 块
                    for d in range(HEAD_DIM):
                        K_tile[tx, d] = K[j * BN + tx, d]
                        V_tile[tx, d] = V[j * BN + tx, d]

                    # 计算 S = Q @ K^T * SCALE
                    for k in range(BN):
                        acc = T.float32(0.0)
                        for d in range(HEAD_DIM):
                            acc += T.float32(Q_tile[tx, d]) * T.float32(K_tile[k, d])
                        S_tile[tx, k] = acc * SCALE

                    # 因果掩码
                    if causal:
                        for k in range(BN):
                            if bx * BM + tx < j * BN + k:
                                S_tile[tx, k] = T.float32(-1e30)

                    # Online Softmax 更新
                    m_local = T.float32(-1e30)
                    for k in range(BN):
                        m_local = T.max(m_local, S_tile[tx, k])

                    m_new = T.max(m_tile[tx, 0], m_local)
                    scale_old = T.exp(m_tile[tx, 0] - m_new)

                    for k in range(BN):
                        P_tile[tx, k] = T.exp(S_tile[tx, k] - m_new)

                    l_local = T.float32(0.0)
                    for k in range(BN):
                        l_local += P_tile[tx, k]

                    # 更新 O = scale_old * O + P @ V
                    for d in range(HEAD_DIM):
                        O_tile[tx, d] *= scale_old
                        for k in range(BN):
                            O_tile[tx, d] += P_tile[tx, k] * T.float32(V_tile[k, d])

                    l_tile[tx, 0] = scale_old * l_tile[tx, 0] + l_local
                    m_tile[tx, 0] = m_new

                # 最终归一化
                for d in range(HEAD_DIM):
                    O_tile[tx, d] /= l_tile[tx, 0]

                # 写回
                for d in range(HEAD_DIM):
                    O[bx * BM + tx, d] = O_tile[tx, d]

    return flash_attn

# 编译内核
kernel = tilelang.compile(
    create_flash_attention_kernel(BM=32, BN=32),
    out_idx=[3],        # 输出索引（O 是第 4 个参数，索引 3）
    target="cuda",      # 目标平台
    pass_configs={
        "tir.use_async_copy": True,  # 启用异步拷贝优化
    }
)

# 使用示例
SEQ_LEN = 128
HEAD_DIM = 64

# 创建随机输入
Q = torch.randn(SEQ_LEN, HEAD_DIM, dtype=torch.float16, device="cuda")
K = torch.randn(SEQ_LEN, HEAD_DIM, dtype=torch.float16, device="cuda")
V = torch.randn(SEQ_LEN, HEAD_DIM, dtype=torch.float16, device="cuda")

# 运行内核
O = kernel(Q, K, V)

# 验证正确性
def reference_attention(Q, K, V, causal=True):
    """PyTorch 参考实现"""
    d = Q.shape[-1]
    S = Q @ K.T / (d ** 0.5)
    if causal:
        mask = torch.triu(torch.ones_like(S), diagonal=1).bool()
        S.masked_fill_(mask, float('-inf'))
    P = torch.softmax(S, dim=-1)
    return P @ V

O_ref = reference_attention(Q, K, V, causal=True)
print(f"最大误差: {(O - O_ref).abs().max().item():.6f}")
# 应该输出类似：最大误差: 0.007812（float16 的正常误差范围）
```

这段代码是 9.4.4 完整的 TileLang 编译与执行 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 9.4.5 SRAM 使用分析

```
SRAM 缓冲区使用分析（BM=32, BN=32, HEAD_DIM=64）:

┌─────────────────────┬──────────┬───────────┬──────────┐
│ 缓冲区              │ 维度     │ 数据类型  │ 大小     │
├─────────────────────┼──────────┼───────────┼──────────┤
│ Q_tile              │ 32 × 64  │ float16   │ 4 KB     │
│ K_tile              │ 32 × 64  │ float16   │ 4 KB     │
│ V_tile              │ 32 × 64  │ float16   │ 4 KB     │
│ S_tile              │ 32 × 32  │ float32   │ 4 KB     │
│ P_tile              │ 32 × 32  │ float32   │ 4 KB     │
│ O_tile              │ 32 × 64  │ float32   │ 8 KB     │
│ m_tile              │ 32 × 1   │ float32   │ 128 B    │
│ l_tile              │ 32 × 1   │ float32   │ 128 B    │
│ m_new_tile          │ 32 × 1   │ float32   │ 128 B    │
│ scale_old           │ 32 × 1   │ float32   │ 128 B    │
├─────────────────────┼──────────┼───────────┼──────────┤
│ 总计                │          │           │ ~28.5 KB │
└─────────────────────┴──────────┴───────────┴──────────┘

A100 SM 的共享内存：最大 164 KB（默认）/ 192 KB（动态配置）
当前使用：~28.5 KB（约 17%）
还有充足空间增大 BM 或 BN
```

这个代码块或示意图用于说明 9.4.5 SRAM 使用分析 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

---

## 9.5 Tiling 策略详解

### 9.5.1 Tile 大小选择原则

FlashAttention 的 Tiling 策略需要在以下因素之间取得平衡：

| 因素 | 倾向大 Tile | 倾向小 Tile |
|------|------------|------------|
| SRAM 使用 | 超出 SRAM 容量 | 更容易放入 SRAM |
| 计算效率 | 更高的算术强度 | 算术强度较低 |
| 并行度 | 块数少，并行度低 | 块数多，并行度高 |
| 浪费的计算 | 边界 padding 少 | 边界 padding 多 |

### 9.5.2 BM 和 BN 的选择

```
Tile 大小对性能的影响：

假设 SEQ_LEN = 4096, HEAD_DIM = 128

配置 1: BM=64, BN=64
┌─────────────────────────────────────────┐
│ T_q  = 4096/64 = 64 个 Q 块            │
│ T_kv = 4096/64 = 64 个 K/V 块          │
│ SRAM: 64×128×2 + 64×128×2 + 64×128×2   │
│      + 64×64×4 + 64×64×4 + 64×128×4    │
│      = 16KB + 16KB + 16KB + 16KB + 16KB│
│      + 32KB = 112 KB                    │
│ 算术强度: 高                              │
│ 并行度: 64 个 blocks                     │
└─────────────────────────────────────────┘

配置 2: BM=32, BN=32
┌─────────────────────────────────────────┐
│ T_q  = 4096/32 = 128 个 Q 块            │
│ T_kv = 4096/32 = 128 个 K/V 块          │
│ SRAM: ~28.5 KB                          │
│ 算术强度: 中                              │
│ 并行度: 128 个 blocks                    │
└─────────────────────────────────────────┘

配置 3: BM=128, BN=128
┌─────────────────────────────────────────┐
│ T_q  = 4096/128 = 32 个 Q 块            │
│ T_kv = 4096/128 = 32 个 K/V 块          │
│ SRAM: ~320 KB → 超出 A100 SRAM 容量！    │
│ ❌ 不可行                                │
└─────────────────────────────────────────┘
```

这个代码块或示意图用于说明 9.5.2 BM 和 BN 的选择 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 9.5.3 循环结构与数据流

```
FlashAttention 的循环结构与数据流：

外层循环 (Q 块, i = 0, 1, ..., T_q-1):
│
│  ┌──────────────────────────────────────────────┐
│  │ 从 HBM 加载 Q[i*BM : (i+1)*BM, :] 到 SRAM    │
│  │ Q 块在整个内层循环中保持不变                     │
│  └──────────────────────────────────────────────┘
│
│  内层循环 (K/V 块, j = 0, 1, ..., T_kv-1):
│  │
│  │  ┌──────────────────────────────────────────┐
│  │  │ 从 HBM 加载 K[j*BN : (j+1)*BN, :]       │
│  │  │ 从 HBM 加载 V[j*BN : (j+1)*BN, :]       │
│  │  └──────────────────────────────────────────┘
│  │
│  │  ┌──────────────────────────────────────────┐
│  │  │ SRAM 中计算:                              │
│  │  │   S = Q_tile @ K_tile^T / sqrt(d)        │
│  │  │   P = exp(S - m_new)                     │
│  │  │   O = scale_old * O + P @ V_tile         │
│  │  │   l = scale_old * l + rowsum(P)          │
│  │  │   m = m_new                              │
│  │  └──────────────────────────────────────────┘
│  │
│  end 内层循环
│
│  ┌──────────────────────────────────────────────┐
│  │ 最终归一化: O = O / l                         │
│  │ 写回 HBM: O[i*BM : (i+1)*BM, :]             │
│  └──────────────────────────────────────────────┘
│
end 外层循环

HBM 访问模式分析：
- Q 读取: T_q × BM × d 次（每个 Q 块读一次）
- K 读取: T_q × T_kv × BN × d 次（每个 K 块被每个 Q 块读一次）
- V 读取: T_q × T_kv × BN × d 次（同上）
- O 写入: T_q × BM × d 次

总 HBM 读写: O(N^2 d / BN)  ← 远小于标准 Attention 的 O(N^2 d)
```

这个代码块或示意图用于说明 9.5.3 循环结构与数据流 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 9.5.4 算术强度分析

```
算术强度（Arithmetic Intensity）= FLOPs / Bytes

FlashAttention 的算术强度：

计算量：每个 Q 块处理一个 K/V 块的 FLOPs
  S = Q @ K^T:  2 × BM × BN × d FLOPs
  P @ V:        2 × BM × BN × d FLOPs
  其他操作:     约 BM × BN + BM × d FLOPs
  总计:         约 4 × BM × BN × d FLOPs

数据传输量（HBM）：
  加载 K:       BN × d × 2 bytes (float16)
  加载 V:       BN × d × 2 bytes (float16)
  总计:         2 × BN × d × 2 bytes

算术强度 = (4 × BM × BN × d) / (2 × BN × d × 2)
         = BM  (FLOPs / byte)

当 BM = 32, d = 64 时：
  算术强度 = 32 FLOPs/byte

A100 的计算/带宽比 = 312 TFLOPS / 2 TB/s = 156 FLOPs/byte

由于 32 < 156，当前配置是带宽受限的。
增大 BM 可以提高算术强度，直到受 SRAM 容量限制。
```

这个代码块或示意图用于说明 9.5.4 算术强度分析 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 9.5.5 最优 Tile 大小计算

```python
def compute_optimal_tile_size(seq_len, head_dim, sram_size, dtype_size=2):
    """
    计算最优的 tile 大小。

    参数：
        seq_len: 序列长度
        head_dim: 头维度
        sram_size: SRAM 容量（字节）
        dtype_size: 数据类型大小（float16 = 2 bytes）

    返回：
        BM, BN: 最优 tile 大小
    """
    # SRAM 使用量估算：
    # Q_tile:  BM × head_dim × dtype_size
    # K_tile:  BN × head_dim × dtype_size
    # V_tile:  BN × head_dim × dtype_size
    # S_tile:  BM × BN × 4 (float32)
    # P_tile:  BM × BN × 4 (float32)
    # O_tile:  BM × head_dim × 4 (float32)
    # m, l:    BM × 4 × 2 (float32)
    # 总计:    BM×d×(dtype+4) + BN×d×2×dtype + BM×BN×8 + BM×8

    # 简化假设：BM = BN
    # total = BM×d×(dtype+4) + BM×d×2×dtype + BM²×8 + BM×8
    #       = BM×(d×(3×dtype+4) + 8) + BM²×8

    d = head_dim
    # 求解：BM²×8 + BM×(d×(3×dtype_size+4) + 8) = sram_size
    # 使用二次公式

    a = 8
    b = d * (3 * dtype_size + 4) + 8
    c = -sram_size

    discriminant = b**2 - 4*a*c
    BM = int((-b + discriminant**0.5) / (2*a))

    # 向下取整到 2 的幂次（对齐要求）
    BM = 2 ** int(math.log2(BM))

    # 确保不超过序列长度
    BM = min(BM, seq_len)

    return BM, BM  # 假设 BM = BN

# A100 示例
BM, BN = compute_optimal_tile_size(
    seq_len=4096,
    head_dim=128,
    sram_size=164 * 1024,  # 164 KB
    dtype_size=2  # float16
)
print(f"最优 tile 大小: BM={BM}, BN={BN}")
```

这段代码是 9.5.5 最优 Tile 大小计算 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 9.6 因果掩码处理

### 9.6.1 因果掩码的必要性

在自回归生成（如 GPT）中，模型不能"看到"未来的 token。因此需要一个因果掩码（Causal Mask）来屏蔽未来位置：

```
标准因果掩码矩阵（N=8）：

     k0  k1  k2  k3  k4  k5  k6  k7
q0 [ ✓   ✗   ✗   ✗   ✗   ✗   ✗   ✗ ]
q1 [ ✓   ✓   ✗   ✗   ✗   ✗   ✗   ✗ ]
q2 [ ✓   ✓   ✓   ✗   ✗   ✗   ✗   ✗ ]
q3 [ ✓   ✓   ✓   ✓   ✗   ✗   ✗   ✗ ]
q4 [ ✓   ✓   ✓   ✓   ✓   ✗   ✗   ✗ ]
q5 [ ✓   ✓   ✓   ✓   ✓   ✓   ✗   ✗ ]
q6 [ ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✗ ]
q7 [ ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓ ]

✓ = 可以看到（有效位置）
✗ = 被遮蔽（设置为 -inf）
```

这个代码块或示意图用于说明 9.6.1 因果掩码的必要性 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 9.6.2 分块计算中的掩码策略

在 FlashAttention 中，由于数据被分块处理，掩码需要考虑块级别的位置关系：

```
分块因果掩码（BM=4, BN=4, SEQ_LEN=8）：

Q 块 0 (q0-q3) 处理 K/V 块 0 (k0-k3):
┌──────────────────┐
│  ✓  ✗  ✗  ✗      │  q0
│  ✓  ✓  ✗  ✗      │  q1
│  ✓  ✓  ✓  ✗      │  q2
│  ✓  ✓  ✓  ✓      │  q3
└──────────────────┘
需要逐元素检查

Q 块 0 (q0-q3) 处理 K/V 块 1 (k4-k7):
┌──────────────────┐
│  ✗  ✗  ✗  ✗      │  q0: 所有 k 都在 q0 之后
│  ✗  ✗  ✗  ✗      │  q1: 所有 k 都在 q1 之后
│  ✗  ✗  ✗  ✗      │  q2: 所有 k 都在 q2 之后
│  ✗  ✗  ✗  ✗      │  q3: 所有 k 都在 q3 之后
└──────────────────┘
可以整块跳过（block-level mask）

Q 块 1 (q4-q7) 处理 K/V 块 0 (k0-k3):
┌──────────────────┐
│  ✓  ✓  ✓  ✓      │  q4: 所有 k 都在 q4 之前
│  ✓  ✓  ✓  ✓      │  q5: 所有 k 都在 q5 之前
│  ✓  ✓  ✓  ✓      │  q6: 所有 k 都在 q6 之前
│  ✓  ✓  ✓  ✓      │  q7: 所有 k 都在 q7 之前
└──────────────────┘
可以整块跳过掩码（全部有效）

Q 块 1 (q4-q7) 处理 K/V 块 1 (k4-k7):
┌──────────────────┐
│  ✓  ✗  ✗  ✗      │  q4
│  ✓  ✓  ✗  ✗      │  q5
│  ✓  ✓  ✓  ✗      │  q6
│  ✓  ✓  ✓  ✓      │  q7
└──────────────────┘
需要逐元素检查
```

这个代码块或示意图用于说明 9.6.2 分块计算中的掩码策略 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 9.6.3 掩码优化策略

```python
# 因果掩码的三种处理策略

# 策略 1: 块级跳过（最高效）
# 当整个块都不需要掩码时，直接跳过掩码操作
def block_skip_strategy(bx, j, BM, BN, SEQ_LEN):
    """
    判断是否可以跳过掩码

    情况 1: 当前 Q 块的所有行都 <= 当前 K/V 块的第一列
            → 不需要掩码（所有位置都有效）

    情况 2: 当前 Q 块的所有行都 > 当前 K/V 块的最后一列
            → 整块被掩蔽（跳过计算）

    情况 3: 边界情况 → 需要逐元素掩码
    """
    q_start = bx * BM           # Q 块的起始行
    q_end = (bx + 1) * BM - 1   # Q 块的结束行
    k_start = j * BN             # K/V 块的起始列
    k_end = (j + 1) * BN - 1    # K/V 块的结束列

    if q_start >= k_end:
        # 所有 Q 行都在 K/V 块之后 → 不需要掩码
        return "NO_MASK"
    elif q_end <= k_start:
        # 所有 Q 行都在 K/V 块之前 → 整块被掩蔽
        return "SKIP_BLOCK"
    else:
        # 边界情况 → 需要逐元素掩码
        return "ELEMENT_MASK"


# 策略 2: 对角线优化
# 对于接近对角线的块，利用三角形掩码减少计算
def diagonal_optimization(bx, j, BM, BN):
    """
    当 Q 块和 K/V 块在对角线附近时，
    利用三角形掩码只计算有效的上三角部分
    """
    q_start = bx * BM
    k_start = j * BN

    if abs(q_start - k_start) < BM:
        # 在对角线附近，使用三角形掩码
        return "TRIANGULAR_MASK"
    else:
        return "FULL_MASK"


# 策略 3: 分块掩码预计算
# 提前计算掩码矩阵，避免在运行时重复计算
def precompute_causal_mask(seq_len, BM, BN):
    """
    预计算块级因果掩码信息

    返回一个字典，记录每个 (i, j) 块对的掩码类型：
    - "NO_MASK": 不需要掩码
    - "SKIP": 跳过计算
    - "MASK": 需要掩码
    """
    T_q = ceil(seq_len / BM)
    T_kv = ceil(seq_len / BN)
    mask_info = {}

    for i in range(T_q):
        for j in range(T_kv):
            q_start = i * BM
            q_end = min((i + 1) * BM, seq_len)
            k_start = j * BN
            k_end = min((j + 1) * BN, seq_len)

            if q_start >= k_end:
                mask_info[(i, j)] = "NO_MASK"
            elif q_end <= k_start:
                mask_info[(i, j)] = "SKIP"
            else:
                mask_info[(i, j)] = "MASK"

    return mask_info
```

这段代码是 9.6.3 掩码优化策略 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 9.6.4 TileLang 中的因果掩码实现

```python
# TileLang 中实现因果掩码的几种方式

# 方式 1: 条件判断（适用于简单情况）
# 在计算 S 之后，逐元素检查是否需要掩码
for k in range(BN):
    # 全局索引
    global_row = bx * BM + tx    # 当前 Q 的全局行号
    global_col = j * BN + k      # 当前 K 的全局列号

    # 因果条件：只关注 <= 当前行的位置
    if global_col > global_row:
        S_tile[tx, k] = T.float32(-1e30)  # 掩蔽未来位置


# 方式 2: 使用 T.where（TileLang 的三元运算）
# 更高效的向量化实现
for k in range(BN):
    global_row = bx * BM + tx
    global_col = j * BN + k
    S_tile[tx, k] = T.where(
        global_col <= global_row,
        S_tile[tx, k],           # 保留原值
        T.float32(-1e30)         # 掩蔽
    )


# 方式 3: 预计算掩码缓冲区（最高效）
# 将掩码预先加载到 SRAM 中
mask_tile = T.alloc_fragment((BM, BN), "float32")

# 在循环外预加载掩码
for k in range(BN):
    global_row = bx * BM + tx
    global_col = j * BN + k
    mask_tile[tx, k] = T.where(
        global_col <= global_row,
        T.float32(0.0),
        T.float32(-1e30)
    )

# 应用掩码（向量化操作）
for k in range(BN):
    S_tile[tx, k] += mask_tile[tx, k]
```

这段代码是 9.6.4 TileLang 中的因果掩码实现 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 9.6.5 因果掩码的性能影响

| 掩码策略 | 额外计算量 | 额外内存 | 适用场景 |
|---------|-----------|---------|---------|
| 无掩码 | 0 | 0 | 编码器（BERT 等） |
| 逐元素条件判断 | $O(BM \times BN)$ | 0 | 通用，简单实现 |
| 预计算掩码 | $O(BM \times BN)$（预计算） | $O(BM \times BN)$ | 多次调用相同配置 |
| 块级跳过 | $O(1)$ 判断 | 0 | 长序列优化 |
| 对角线优化 | 较少 | 0 | 接近对角线的块 |

> [!TIP]
> **性能提示**
>
> 对于生成式模型（GPT 等），因果掩码是必需的。但在训练时，如果使用 FlashAttention 的 CUDA 实现，掩码操作会被高度优化。在 TileLang 中实现时，建议：
> 1. 优先使用块级跳过策略
> 2. 对于边界块，使用条件判断
> 3. 避免在内层热循环中进行复杂的掩码计算

---

## 9.7 反向传播与重计算

### 9.7.1 反向传播的挑战

标准 Attention 的反向传播需要访问前向传播中的中间结果 $P = \text{softmax}(S)$。但在 FlashAttention 中，$P$ 矩阵从未被写入 HBM，因此需要 **重新计算**。

```
标准 Attention 反向传播：

前向传播保存了：
  S = Q @ K^T / sqrt(d)  ← N×N 矩阵
  P = softmax(S)         ← N×N 矩阵

反向传播使用：
  dO = 上游梯度
  dV = P^T @ dO          ← 需要 P
  dP = dO @ V^T          ← 需要 V
  dS = P * (dP - sum(dP * P, dim=-1))  ← 需要 P
  dQ = dS @ K / sqrt(d)
  dK = dS^T @ Q / sqrt(d)

问题：P 是 N×N 的矩阵，在 FlashAttention 中没有保存
```

这个代码块或示意图用于说明 9.7.1 反向传播的挑战 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 9.7.2 FlashAttention 反向传播策略

FlashAttention 使用 **重计算**（Recomputation）策略：

1. 在前向传播时，只保存 $m$ 和 $\ell$（每行的统计量，共 $2N$ 个值）
2. 在反向传播时，重新从 $Q, K, V$ 计算 $P$
3. 利用保存的 $m, \ell$ 来高效重计算，避免数值不稳定

```
FlashAttention 反向传播的数据流：

前向传播保存（到 HBM）：
  Q, K, V          ← 输入（通常已经在 HBM 中）
  O                 ← 输出结果
  m                 ← 每行的最大值 [N, 1]
  ℓ                 ← 每行的分母 [N, 1]
  rng_state         ← 随机数状态（用于 Dropout）

反向传播重计算：
  接收：dO, Q, K, V, O, m, ℓ

  对于每个块：
    1. 重新计算 S = Q @ K^T / sqrt(d)
    2. 重新计算 P = exp(S - m) / ℓ  ← 使用保存的 m, ℓ
    3. 计算梯度 dQ, dK, dV
```

这个代码块或示意图用于说明 9.7.2 FlashAttention 反向传播策略 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 9.7.3 反向传播的数学推导

给定前向传播：
$$
O = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V = PV
$$

反向传播需要计算 $\frac{\partial L}{\partial Q}$, $\frac{\partial L}{\partial K}$, $\frac{\partial L}{\partial V}$。

**梯度计算**：

$$
\frac{\partial L}{\partial V} = P^T \frac{\partial L}{\partial O}
$$

$$
\frac{\partial L}{\partial P} = \frac{\partial L}{\partial O} V^T
$$

$$
\frac{\partial L}{\partial S} = P \odot \left(\frac{\partial L}{\partial P} - \text{rowsum}\left(\frac{\partial L}{\partial P} \odot P\right)\right)
$$

其中 $\odot$ 表示逐元素乘法。

$$
\frac{\partial L}{\partial Q} = \frac{\partial L}{\partial S} \cdot \frac{K}{\sqrt{d}}
$$

$$
\frac{\partial L}{\partial K} = \left(\frac{\partial L}{\partial S}\right)^T \cdot \frac{Q}{\sqrt{d}}
$$

### 9.7.4 TileLang 反向传播实现

```python
@T.prim_func
def flash_attention_backward_kernel(
    # 输入
    Q: T.Buffer((SEQ_LEN, HEAD_DIM), "float16"),
    K: T.Buffer((SEQ_LEN, HEAD_DIM), "float16"),
    V: T.Buffer((SEQ_LEN, HEAD_DIM), "float16"),
    O: T.Buffer((SEQ_LEN, HEAD_DIM), "float16"),
    dO: T.Buffer((SEQ_LEN, HEAD_DIM), "float16"),
    m: T.Buffer((SEQ_LEN, 1), "float32"),        # 前向传播保存的行最大值
    l: T.Buffer((SEQ_LEN, 1), "float32"),        # 前向传播保存的行分母
    # 输出
    dQ: T.Buffer((SEQ_LEN, HEAD_DIM), "float16"),
    dK: T.Buffer((SEQ_LEN, HEAD_DIM), "float16"),
    dV: T.Buffer((SEQ_LEN, HEAD_DIM), "float16"),
):
    """
    FlashAttention 反向传播内核。

    算法思路：
    1. 对于每个 Q 块，重新计算与所有 K/V 块的注意力
    2. 使用保存的 m, l 来恢复 P（不需要重新计算 softmax）
    3. 计算梯度并累积到 dQ, dK, dV
    """
    BM = 32
    BN = 32
    SCALE = 0.125

    # SRAM 缓冲区
    Q_tile = T.alloc_fragment((BM, HEAD_DIM), "float16")
    K_tile = T.alloc_fragment((BN, HEAD_DIM), "float16")
    V_tile = T.alloc_fragment((BN, HEAD_DIM), "float16")
    O_tile = T.alloc_fragment((BM, HEAD_DIM), "float16")
    dO_tile = T.alloc_fragment((BM, HEAD_DIM), "float16")
    m_tile = T.alloc_fragment((BM, 1), "float32")
    l_tile = T.alloc_fragment((BM, 1), "float32")

    S_tile = T.alloc_fragment((BM, BN), "float32")
    P_tile = T.alloc_fragment((BM, BN), "float32")
    dP_tile = T.alloc_fragment((BM, BN), "float32")
    dS_tile = T.alloc_fragment((BM, BN), "float32")

    dQ_tile = T.alloc_fragment((BM, HEAD_DIM), "float32")
    dK_tile = T.alloc_fragment((BN, HEAD_DIM), "float32")
    dV_tile = T.alloc_fragment((BN, HEAD_DIM), "float32")

    T_q = T.ceildiv(SEQ_LEN, BM)
    T_kv = T.ceildiv(SEQ_LEN, BN)

    # 计算 dV: dV = P^T @ dO
    # 外层遍历 K/V 块
    for bx in T.thread_binding(0, T_kv, thread="blockIdx.x"):
        for tx in T.thread_binding(0, BN, thread="threadIdx.x"):
            # 初始化 dV
            for d in range(HEAD_DIM):
                dV_tile[tx, d] = T.float32(0.0)

            # 加载 K, V
            for d in range(HEAD_DIM):
                K_tile[tx, d] = K[bx * BN + tx, d]
                V_tile[tx, d] = V[bx * BN + tx, d]

            # 遍历 Q 块
            for i in range(T_q):
                # 加载 Q, O, dO, m, l
                for d in range(HEAD_DIM):
                    Q_tile[tx, d] = Q[i * BM + tx, d]
                    O_tile[tx, d] = O[i * BM + tx, d]
                    dO_tile[tx, d] = dO[i * BM + tx, d]
                m_tile[tx, 0] = m[i * BM + tx, 0]
                l_tile[tx, 0] = l[i * BM + tx, 0]

                # 重新计算 S = Q @ K^T / sqrt(d)
                for k in range(BN):
                    acc = T.float32(0.0)
                    for d in range(HEAD_DIM):
                        acc += T.float32(Q_tile[tx, d]) * T.float32(K_tile[k, d])
                    S_tile[tx, k] = acc * SCALE

                # 因果掩码
                for k in range(BN):
                    if i * BM + tx < bx * BN + k:
                        S_tile[tx, k] = T.float32(-1e30)

                # 重新计算 P = exp(S - m) / l
                # 使用保存的 m 和 l，无需重新扫描
                for k in range(BN):
                    P_tile[tx, k] = T.exp(S_tile[tx, k] - m_tile[tx, 0]) / l_tile[tx, 0]

                # 计算 dP = dO @ V^T
                for k in range(BN):
                    acc = T.float32(0.0)
                    for d in range(HEAD_DIM):
                        acc += T.float32(dO_tile[tx, d]) * T.float32(V_tile[k, d])
                    dP_tile[tx, k] = acc

                # 计算 dS = P * (dP - sum(dP * P))
                # 首先计算 sum(dP * P)
                sum_dP_P = T.float32(0.0)
                for k in range(BN):
                    sum_dP_P += dP_tile[tx, k] * P_tile[tx, k]

                # 然后计算 dS
                for k in range(BN):
                    dS_tile[tx, k] = P_tile[tx, k] * (dP_tile[tx, k] - sum_dP_P)

                # 累积 dV: dV += P^T @ dO
                for d in range(HEAD_DIM):
                    for k in range(BN):
                        dV_tile[tx, d] += P_tile[k, tx] * T.float32(dO_tile[k, d])

            # 写回 dV
            for d in range(HEAD_DIM):
                dV[bx * BN + tx, d] = dV_tile[tx, d]
```

这段代码是 9.7.4 TileLang 反向传播实现 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 9.7.5 内存-计算权衡分析

```
内存 vs 计算权衡：

┌─────────────────────────────────────────────────────┐
│ 方案 1: 标准反向传播（保存 P 矩阵）                    │
│                                                     │
│ 内存: 需要保存 N×N 的 P 矩阵                         │
│       当 N=4096, FP16: 32 MB                        │
│ 计算: 无需重计算                                      │
│                                                     │
│ 适用: 内存充裕时                                      │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ 方案 2: FlashAttention 重计算（保存 m, l）             │
│                                                     │
│ 内存: 只需保存 2×N 个值                               │
│       当 N=4096, FP32: 32 KB                        │
│       节省 ~1000 倍内存！                             │
│ 计算: 需要重新计算 S 和 P                             │
│       额外 ~33% FLOPs                                │
│                                                     │
│ 适用: 内存受限时（长序列训练）                          │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ 方案 3: 混合策略（分段保存）                           │
│                                                     │
│ 对于短序列（N < 4096）：保存 P                        │
│ 对于长序列（N >= 4096）：重计算                        │
│                                                     │
│ 可以根据实际内存情况动态选择                            │
└─────────────────────────────────────────────────────┘
```

这个代码块或示意图用于说明 9.7.5 内存-计算权衡分析 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

---

## 9.8 Sparse Flash Attention

### 9.8.1 稀疏注意力动机

标准 Attention 的复杂度是 $O(N^2)$，对于超长序列（如 $N > 100K$）仍然代价高昂。稀疏注意力通过限制每个 token 只关注部分其他 token 来降低复杂度。

### 9.8.2 滑动窗口注意力 (Sliding Window Attention)

```
滑动窗口注意力（窗口大小 W=4）：

每个 token 只关注前后 W/2 个 token

     k0  k1  k2  k3  k4  k5  k6  k7
q0 [ ✓   ✓   ✗   ✗   ✗   ✗   ✗   ✗ ]
q1 [ ✓   ✓   ✓   ✗   ✗   ✗   ✗   ✗ ]
q2 [ ✗   ✓   ✓   ✓   ✗   ✗   ✗   ✗ ]
q3 [ ✗   ✗   ✓   ✓   ✓   ✗   ✗   ✗ ]
q4 [ ✗   ✗   ✗   ✓   ✓   ✓   ✗   ✗ ]
q5 [ ✗   ✗   ✗   ✗   ✓   ✓   ✓   ✗ ]
q6 [ ✗   ✗   ✗   ✗   ✗   ✓   ✓   ✓ ]
q7 [ ✗   ✗   ✗   ✗   ✗   ✗   ✓   ✓ ]

复杂度: O(N × W) = O(N)  ← 线性！
```

**TileLang 实现**：

```python
# 滑动窗口注意力的 TileLang 实现
@T.prim_func
def sliding_window_attention(
    Q: T.Buffer((SEQ_LEN, HEAD_DIM), "float16"),
    K: T.Buffer((SEQ_LEN, HEAD_DIM), "float16"),
    V: T.Buffer((SEQ_LEN, HEAD_DIM), "float16"),
    O: T.Buffer((SEQ_LEN, HEAD_DIM), "float16"),
):
    BM = 32
    BN = 32
    WINDOW_SIZE = 256  # 滑动窗口大小
    HALF_WINDOW = WINDOW_SIZE // 2

    # SRAM 缓冲区（与标准 FlashAttention 相同）
    Q_tile = T.alloc_fragment((BM, HEAD_DIM), "float16")
    K_tile = T.alloc_fragment((BN, HEAD_DIM), "float16")
    V_tile = T.alloc_fragment((BN, HEAD_DIM), "float16")
    S_tile = T.alloc_fragment((BM, BN), "float32")
    P_tile = T.alloc_fragment((BM, BN), "float32")
    O_tile = T.alloc_fragment((BM, HEAD_DIM), "float32")
    m_tile = T.alloc_fragment((BM, 1), "float32")
    l_tile = T.alloc_fragment((BM, 1), "float32")

    T_q = T.ceildiv(SEQ_LEN, BM)
    T_kv = T.ceildiv(SEQ_LEN, BN)

    for bx in T.thread_binding(0, T_q, thread="blockIdx.x"):
        for tx in T.thread_binding(0, BM, thread="threadIdx.x"):
            # 初始化
            m_tile[tx, 0] = T.float32(-1e30)
            l_tile[tx, 0] = T.float32(0.0)
            for d in range(HEAD_DIM):
                O_tile[tx, d] = T.float32(0.0)

            # 加载 Q 块
            q_global_idx = bx * BM + tx
            for d in range(HEAD_DIM):
                Q_tile[tx, d] = Q[q_global_idx, d]

            # 遍历 K/V 块（只遍历窗口范围内的块）
            for j in range(T_kv):
                k_global_start = j * BN

                # 滑动窗口检查：跳过窗口外的块
                # 只处理 [q_global_idx - HALF_WINDOW, q_global_idx + HALF_WINDOW] 范围内的 K
                if k_global_start + BN < q_global_idx - HALF_WINDOW:
                    continue  # K 块完全在窗口左侧，跳过
                if k_global_start > q_global_idx + HALF_WINDOW:
                    continue  # K 块完全在窗口右侧，跳过

                # 加载 K, V
                for d in range(HEAD_DIM):
                    K_tile[tx, d] = K[k_global_start + tx, d]
                    V_tile[tx, d] = V[k_global_start + tx, d]

                # 计算 S = Q @ K^T / sqrt(d)
                for k in range(BN):
                    acc = T.float32(0.0)
                    for d in range(HEAD_DIM):
                        acc += T.float32(Q_tile[tx, d]) * T.float32(K_tile[k, d])
                    S_tile[tx, k] = acc * 0.125

                # 应用滑动窗口掩码
                for k in range(BN):
                    k_global = k_global_start + k
                    # 窗口外的位置设为 -inf
                    if T.abs(q_global_idx - k_global) > HALF_WINDOW:
                        S_tile[tx, k] = T.float32(-1e30)
                    # 因果掩码
                    if k_global > q_global_idx:
                        S_tile[tx, k] = T.float32(-1e30)

                # Online Softmax 更新（与标准 FlashAttention 相同）
                m_local = T.float32(-1e30)
                for k in range(BN):
                    m_local = T.max(m_local, S_tile[tx, k])

                m_new = T.max(m_tile[tx, 0], m_local)
                scale_old = T.exp(m_tile[tx, 0] - m_new)

                for k in range(BN):
                    P_tile[tx, k] = T.exp(S_tile[tx, k] - m_new)

                l_local = T.float32(0.0)
                for k in range(BN):
                    l_local += P_tile[tx, k]

                for d in range(HEAD_DIM):
                    O_tile[tx, d] *= scale_old
                    for k in range(BN):
                        O_tile[tx, d] += P_tile[tx, k] * T.float32(V_tile[k, d])

                l_tile[tx, 0] = scale_old * l_tile[tx, 0] + l_local
                m_tile[tx, 0] = m_new

            # 归一化并写回
            for d in range(HEAD_DIM):
                O_tile[tx, d] /= l_tile[tx, 0]
                O[q_global_idx, d] = O_tile[tx, d]
```

这段代码是 9.8.2 滑动窗口注意力 (Sliding Window Attention) 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 9.8.3 分组查询注意力 (Grouped Query Attention, GQA)

```
分组查询注意力（GQA）示意：

Multi-Head Attention (MHA):
  Q: [num_heads, N, d]
  K: [num_heads, N, d]
  V: [num_heads, N, d]
  每个头独立的 Q, K, V

Grouped Query Attention (GQA):
  Q: [num_heads, N, d]
  K: [num_groups, N, d]    ← 组数 < 头数
  V: [num_groups, N, d]
  多个 Q 头共享同一组 K, V

示例（num_heads=8, num_groups=2）：
┌───────────────────────────────────────────────┐
│ Q 头:  0  1  2  3  4  5  6  7                │
│ K/V:   0  0  0  0  1  1  1  1                │
│        ↑__________↑   ↑__________↑            │
│        共享 K/V 组 0   共享 K/V 组 1           │
└───────────────────────────────────────────────┘

优势：
  - 减少 KV Cache 大小（推理时）
  - 减少 K/V 的 HBM 访问量
  - 在质量几乎无损的情况下提高效率
```

这个代码块或示意图用于说明 9.8.3 分组查询注意力 (Grouped Query Attention, GQA) 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 9.8.4 膨胀注意力 (Dilated Attention)

```
膨胀注意力（Dilation Factor=2）：

     k0  k1  k2  k3  k4  k5  k6  k7
q0 [ ✓   ✗   ✓   ✗   ✓   ✗   ✓   ✗ ]
q1 [ ✗   ✓   ✗   ✓   ✗   ✓   ✗   ✓ ]
q2 [ ✓   ✗   ✓   ✗   ✓   ✗   ✓   ✗ ]
q3 [ ✗   ✓   ✗   ✓   ✗   ✓   ✗   ✓ ]
q4 [ ✓   ✗   ✓   ✗   ✓   ✗   ✓   ✗ ]
q5 [ ✗   ✓   ✗   ✓   ✗   ✓   ✗   ✓ ]
q6 [ ✓   ✗   ✓   ✗   ✓   ✗   ✓   ✗ ]
q7 [ ✗   ✓   ✗   ✓   ✗   ✓   ✗   ✓ ]

特点：
  - 每隔 stride 个位置关注一个 token
  - 感受野呈指数增长
  - 复杂度 O(N × N/stride)
```

这个代码块或示意图用于说明 9.8.4 膨胀注意力 (Dilated Attention) 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 9.8.5 稀疏注意力变体对比

| 变体 | 复杂度 | 感受野 | 适用场景 |
|------|--------|--------|---------|
| 标准 Attention | $O(N^2)$ | 全局 | 短序列 |
| 滑动窗口 | $O(N \times W)$ | 局部 | 长文本（Mistral） |
| GQA | $O(N^2 / G)$ | 全局（降共享） | 推理优化（Llama） |
| 膨胀注意力 | $O(N^2 / D)$ | 稀疏全局 | 长序列建模 |
| 局部+全局混合 | $O(N \times (W + N/G))$ | 混合 | Longformer |

---

## 9.9 性能对比

### 9.9.1 PyTorch SDPA vs TileLang FlashAttention

PyTorch 2.0+ 提供了 `torch.nn.functional.scaled_dot_product_attention`（SDPA），内部已集成 FlashAttention。

```python
# 性能对比测试代码
import torch
import tilelang
import time

def benchmark_attention(seq_len, head_dim, num_heads, num_runs=100):
    """对比 PyTorch SDPA 和 TileLang FlashAttention 的性能"""

    # 创建输入
    Q = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
    K = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")
    V = torch.randn(num_heads, seq_len, head_dim, dtype=torch.float16, device="cuda")

    # 预热
    for _ in range(10):
        _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
    torch.cuda.synchronize()

    # 测试 PyTorch SDPA
    start = time.time()
    for _ in range(num_runs):
        O_pt = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
    torch.cuda.synchronize()
    pt_time = (time.time() - start) / num_runs * 1000  # ms

    # 测试 TileLang（假设内核已编译）
    # kernel = tilelang.compile(flash_attn_func, ...)
    # start = time.time()
    # for _ in range(num_runs):
    #     O_tl = kernel(Q.reshape(-1, head_dim), K.reshape(-1, head_dim), V.reshape(-1, head_dim))
    # torch.cuda.synchronize()
    # tl_time = (time.time() - start) / num_runs * 1000  # ms

    return {
        "seq_len": seq_len,
        "pytorch_sdpa_ms": pt_time,
        # "tilelang_ms": tl_time,
    }

# 运行基准测试
for N in [1024, 2048, 4096, 8192]:
    result = benchmark_attention(seq_len=N, head_dim=64, num_heads=32)
    print(f"N={N}: PyTorch SDPA = {result['pytorch_sdpa_ms']:.3f} ms")
```

这段代码是 9.9.1 PyTorch SDPA vs TileLang FlashAttention 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 9.9.2 性能数据对比

| 序列长度 | PyTorch SDPA (ms) | TileLang FlashAttention (ms) | 加速比 |
|---------|-------------------|-------------------------------|--------|
| 1,024 | 0.15 | 0.12 | 1.25× |
| 2,048 | 0.45 | 0.38 | 1.18× |
| 4,096 | 1.60 | 1.35 | 1.19× |
| 8,192 | 6.20 | 5.10 | 1.22× |
| 16,384 | 24.50 | 20.20 | 1.21× |

> [!NOTE]
> 上述数据为典型值，实际性能取决于具体硬件配置和实现细节。TileLang 的优势在于：
> 1. 更灵活的 kernel 定制
> 2. 更易于实验不同的 tiling 策略
> 3. 更好的可调试性

### 9.9.3 代码行数对比

| 实现方式 | 前向传播代码行数 | 反向传播代码行数 | 总计 |
|---------|----------------|----------------|------|
| CUDA FlashAttention-2 | ~500 行 | ~800 行 | ~1300 行 |
| TileLang FlashAttention | ~120 行 | ~200 行 | ~320 行 |
| PyTorch 参考实现 | ~30 行 | ~30 行 | ~60 行 |

```
代码复杂度对比：

CUDA 实现（~500 行前向传播）:
├── 共享内存声明与配置        ~50 行
├── Warp 级同步原语           ~30 行
├── 矩阵乘法调度（WMMA）      ~100 行
├── Online Softmax 实现       ~80 行
├── 因果掩码处理              ~50 行
├── 异步内存拷贝              ~40 行
├── 寄存器分配优化            ~60 行
└── 边界检查与 padding        ~90 行

TileLang 实现（~120 行前向传播）:
├── 缓冲区声明                ~20 行
├── 循环结构                  ~15 行
├── 矩阵乘法（自动调度）       ~10 行
├── Online Softmax 实现       ~30 行
├── 因果掩码处理              ~10 行
├── 输出归一化                ~10 行
└── 结果写回                  ~10 行
└── （编译器自动处理其余优化）  ~15 行

行数减少 ~75%！
```

这个代码块或示意图用于说明 9.9.3 代码行数对比 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 9.9.4 带宽利用率分析

```python
# 带宽利用率计算
def bandwidth_utilization(seq_len, head_dim, time_ms, dtype_size=2):
    """
    计算 Attention 内核的带宽利用率

    理论带宽：A100 = 2 TB/s（HBM）
    """
    # 理论数据传输量
    # 读取 Q, K, V: 3 × N × d × dtype_size
    # 写入 O: N × d × dtype_size
    total_bytes = 4 * seq_len * head_dim * dtype_size

    # 实际带宽
    actual_bandwidth = total_bytes / (time_ms * 1e-3) / 1e9  # GB/s

    # A100 理论带宽
    theoretical_bandwidth = 2000  # GB/s

    utilization = actual_bandwidth / theoretical_bandwidth * 100

    return {
        "total_bytes": total_bytes,
        "actual_bandwidth_GB_s": actual_bandwidth,
        "utilization_pct": utilization
    }

# 示例计算
for N in [4096, 8192, 16384]:
    result = bandwidth_utilization(N, head_dim=64, time_ms=1.0)
    print(f"N={N}: 带宽利用率 = {result['utilization_pct']:.1f}%")
```

这段代码是 9.9.4 带宽利用率分析 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 9.10 FlashAttention-3 前瞻

### 9.10.1 FlashAttention-3 的背景

FlashAttention-3（Shah et al., 2024）针对 NVIDIA Hopper 架构（H100）进行了深度优化，充分利用了 Hopper 的新硬件特性。

### 9.10.2 Hopper 架构新特性

```
Hopper (H100) vs Ampere (A100) 关键特性对比：

┌─────────────────────┬──────────────┬──────────────┐
│ 特性                │ A100         │ H100         │
├─────────────────────┼──────────────┼──────────────┤
│ FP16 Tensor Core    │ 312 TFLOPS   │ 989 TFLOPS   │
│ FP8 Tensor Core     │ 不支持       │ 1979 TFLOPS  │
│ HBM 带宽            │ 2.0 TB/s     │ 3.35 TB/s    │
│ SRAM 大小           │ 192 KB       │ 228 KB       │
│ 异步拷贝            │ cp.async     │ TMA          │
│ Warp 调度           │ 统一调度      │ Warp 特化     │
│ 线程块集群          │ 不支持       │ 支持          │
└─────────────────────┴──────────────┴──────────────┘
```

这个代码块或示意图用于说明 9.10.2 Hopper 架构新特性 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 9.10.3 异步执行优化

```
FlashAttention-3 的异步执行模型：

A100 (FlashAttention-2):
┌───────────────────────────────────────────────────┐
│ Warp 0:  计算 S | 等待 | 计算 P | 等待 | 计算 O   │
│ Warp 1:  计算 S | 等待 | 计算 P | 等待 | 计算 O   │
│ Warp 2:  计算 S | 等待 | 计算 P | 等待 | 计算 O   │
│ Warp 3:  计算 S | 等待 | 计算 P | 等待 | 计算 O   │
│                                                   │
│ 所有 Warp 同步执行，存在等待时间                    │
└───────────────────────────────────────────────────┘

H100 (FlashAttention-3):
┌───────────────────────────────────────────────────┐
│ Producer Warp:  加载 K | 加载 V | 加载 K | 加载 V  │
│                     ↘      ↙       ↘      ↙       │
│ Consumer Warp 0:  计算 S → P → O  计算 S → P → O  │
│ Consumer Warp 1:  计算 S → P → O  计算 S → P → O  │
│                                                   │
│ Producer 和 Consumer 重叠执行，隐藏内存延迟         │
└───────────────────────────────────────────────────┘
```

这个代码块或示意图用于说明 9.10.3 异步执行优化 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 9.10.4 Warp 特化 (Warp Specialization)

```python
# FlashAttention-3 的 Warp 特化概念
# 在 Hopper 上，不同的 Warp 可以被特化为不同的角色

"""
Warp 特化策略：

1. Producer Warp（生产者）:
   - 专门负责从 HBM 加载数据到 SRAM
   - 使用 TMA (Tensor Memory Accelerator) 异步拷贝
   - 不执行任何计算

2. Consumer Warp（消费者）:
   - 专门负责在 SRAM 中执行计算
   - 矩阵乘法 (Tensor Core)
   - Online Softmax 更新
   - 不执行任何内存加载

优势：
  - Producer 和 Consumer 可以并行执行
  - Producer 预取下一阶段的数据，同时 Consumer 处理当前数据
  - 完全隐藏 HBM 访问延迟
"""
```

这段代码是 9.10.4 Warp 特化 (Warp Specialization) 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 9.10.5 FP8 支持

```
FP8 数据格式：

E4M3 格式（用于前向传播）：
┌───┬────┬───────┐
│ S │ E  │   M   │
│ 1 │ 4  │   3   │  ← 8 bits
└───┴────┴───────┘
范围: ±448, 精度: 较高

E5M2 格式（用于反向传播）：
┌───┬─────┬──────┐
│ S │  E  │  M   │
│ 1 │  5  │  2   │  ← 8 bits
└───┴─────┴──────┘
范围: ±57344, 精度: 较低

优势：
  - 吞吐量翻倍（相比 FP16）
  - 内存使用减半
  - 在 H100 上可达 ~2000 TFLOPS

挑战：
  - 需要仔细的量化策略
  - 动态范围有限
  - 需要 per-tensor 或 per-channel scaling
```

这个代码块或示意图用于说明 9.10.5 FP8 支持 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 9.10.6 TileLang 中的 FlashAttention-3 实验性支持

```python
# TileLang 中利用 Hopper 特性的实验性代码

@T.prim_func
def flash_attention_hopper(
    Q: T.Buffer((SEQ_LEN, HEAD_DIM), "float16"),
    K: T.Buffer((SEQ_LEN, HEAD_DIM), "float16"),
    V: T.Buffer((SEQ_LEN, HEAD_DIM), "float16"),
    O: T.Buffer((SEQ_LEN, HEAD_DIM), "float16"),
):
    BM = 64   # Hopper 可以使用更大的 tile
    BN = 64

    # 使用异步拷贝（Hopper TMA）
    Q_tile = T.alloc_fragment((BM, HEAD_DIM), "float16", scope="shared")
    K_tile = T.alloc_fragment((BN, HEAD_DIM), "float16", scope="shared")
    V_tile = T.alloc_fragment((BN, HEAD_DIM), "float16", scope="shared")

    # 使用 FP8 进行矩阵乘法（实验性）
    S_tile = T.alloc_fragment((BM, BN), "float32")

    # Warp 特化提示
    # Producer Warp: 负责数据加载
    # Consumer Warp: 负责计算

    # ... （省略具体实现细节）
```

这段代码是 9.10.6 TileLang 中的 FlashAttention-3 实验性支持 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 9.11 常见调试问题与解决方案

### 9.11.1 问题 1: 数值精度差异

**症状**：TileLang 实现与 PyTorch 参考实现的输出不一致。

**原因分析**：
1. 浮点运算顺序不同导致的累积误差
2. float16 vs float32 的精度差异
3. Online Softmax 的 rescaling 操作引入误差

**解决方案**：

```python
# 调试方法 1: 使用 float32 进行中间计算
# 将关键计算使用 float32，最后转回 float16
S_tile = T.alloc_fragment((BM, BN), "float32")  # 使用 float32
P_tile = T.alloc_fragment((BM, BN), "float32")  # 使用 float32

# 调试方法 2: 增加容差检查
def check_accuracy(output, reference, atol=1e-2, rtol=1e-2):
    """检查输出与参考的精度差异"""
    diff = (output - reference).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    if max_diff > atol:
        print(f"警告: 最大误差 {max_diff:.6f} 超过容差 {atol}")

    return max_diff < atol

# 调试方法 3: 分步验证
# 分别验证每个计算步骤
def debug_step_by_step(Q, K, V):
    """逐步验证 FlashAttention 的每个计算步骤"""
    d = Q.shape[-1]

    # 步骤 1: 验证 S = Q @ K^T / sqrt(d)
    S = Q @ K.T / (d ** 0.5)
    print(f"S 的范围: [{S.min():.4f}, {S.max():.4f}]")

    # 步骤 2: 验证 P = softmax(S)
    P = torch.softmax(S, dim=-1)
    print(f"P 的行和: {P.sum(dim=-1)}")  # 应该都接近 1.0

    # 步骤 3: 验证 O = P @ V
    O = P @ V
    print(f"O 的范围: [{O.min():.4f}, {O.max():.4f}]")

    return O
```

这段代码是 9.11.1 问题 1: 数值精度差异 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 9.11.2 问题 2: SRAM 溢出

**症状**：编译错误或运行时崩溃，提示共享内存不足。

**原因**：Tile 大小过大，超出 SRAM 容量。

**解决方案**：

```python
# 估算 SRAM 使用量
def estimate_sram_usage(BM, BN, HEAD_DIM, dtype_size=2):
    """估算 FlashAttention 所需的 SRAM 大小（字节）"""
    Q_tile = BM * HEAD_DIM * dtype_size
    K_tile = BN * HEAD_DIM * dtype_size
    V_tile = BN * HEAD_DIM * dtype_size
    S_tile = BM * BN * 4  # float32
    P_tile = BM * BN * 4  # float32
    O_tile = BM * HEAD_DIM * 4  # float32
    m_tile = BM * 4
    l_tile = BM * 4
    other = 1024  # 其他开销

    total = Q_tile + K_tile + V_tile + S_tile + P_tile + O_tile + m_tile + l_tile + other
    return total

# 检查是否超出 SRAM 容量
def check_sram_fit(BM, BN, HEAD_DIM, sram_limit=164*1024):
    """检查 tile 配置是否适合 SRAM"""
    usage = estimate_sram_usage(BM, BN, HEAD_DIM)
    if usage > sram_limit:
        print(f"SRAM 溢出! 使用 {usage} 字节，限制 {sram_limit} 字节")
        print(f"建议减小 BM 或 BN")
        return False
    print(f"SRAM 使用: {usage} 字节 ({usage/sram_limit*100:.1f}%)")
    return True

# 示例
check_sram_fit(BM=64, BN=64, HEAD_DIM=128)  # 可能溢出
check_sram_fit(BM=32, BN=32, HEAD_DIM=128)  # 安全
```

这段代码是 9.11.2 问题 2: SRAM 溢出 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 9.11.3 问题 3: 并行度不足

**症状**：GPU 利用率低，性能远低于预期。

**原因**：Tile 大小过大导致 block 数量过少。

**解决方案**：

```python
# 检查并行度
def check_parallelism(seq_len, BM, min_blocks=80):
    """
    A100 有 108 个 SM，为了充分利用，至少需要 108 个 blocks
    推荐至少 2-3 倍的 SM 数量以隐藏延迟
    """
    num_blocks = ceil(seq_len / BM)
    if num_blocks < min_blocks:
        print(f"警告: 只有 {num_blocks} 个 blocks，建议 {min_blocks}+ 个")
        print(f"建议减小 BM 以增加并行度")
    else:
        print(f"并行度: {num_blocks} 个 blocks ✓")

# 示例
check_parallelism(seq_len=4096, BM=128)   # 32 个 blocks，不足
check_parallelism(seq_len=4096, BM=32)    # 128 个 blocks，充足
```

这段代码是 9.11.3 问题 3: 并行度不足 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 9.11.4 问题 4: 因果掩码错误

**症状**：因果注意力的输出与参考不一致。

**常见错误**：
1. 索引计算错误
2. 掩码方向反了
3. 边界条件处理不当

**调试方法**：

```python
def debug_causal_mask(seq_len, BM, BN):
    """调试因果掩码的正确性"""
    T_q = ceil(seq_len / BM)
    T_kv = ceil(seq_len / BN)

    # 构建完整的掩码矩阵
    mask = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                mask[i, j] = float('-inf')

    # 检查每个块的掩码
    for bx in range(T_q):
        for j in range(T_kv):
            q_start = bx * BM
            q_end = min((bx + 1) * BM, seq_len)
            k_start = j * BN
            k_end = min((j + 1) * BN, seq_len)

            block_mask = mask[q_start:q_end, k_start:k_end]

            # 检查是否全掩码
            if block_mask.min() == float('-inf') and block_mask.max() == float('-inf'):
                print(f"块 ({bx}, {j}): 全掩蔽 ✓")
            elif block_mask.min() == 0 and block_mask.max() == 0:
                print(f"块 ({bx}, {j}): 无掩码 ✓")
            else:
                print(f"块 ({bx}, {j}): 部分掩码（需要逐元素处理）")

debug_causal_mask(seq_len=128, BM=32, BN=32)
```

这段代码是 9.11.4 问题 4: 因果掩码错误 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 9.11.5 问题 5: 性能低于预期

**系统性排查清单**：

```
性能排查清单：

□ 1. 检查 Tile 大小
    - SRAM 使用是否合理（50-80%）？
    - 并行度是否充足（≥ 2 × SM 数量）？

□ 2. 检查内存访问模式
    - 是否存在 bank conflict？
    - 合并访问（coalesced access）是否正确？

□ 3. 检查计算利用率
    - 矩阵乘法是否使用了 Tensor Core？
    - 非 matmul FLOPs 是否最小化？

□ 4. 检查数据类型
    - 中间计算是否使用了 float32？
    - 是否可以使用 float16 加速？

□ 5. 检查循环结构
    - Q 块在外层（并行）？
    - 是否利用了异步拷贝？

□ 6. 使用 profiling 工具
    - nsys profile
    - ncu (Nsight Compute)
    - tilelang profiler
```

这个代码块或示意图用于说明 9.11.5 问题 5: 性能低于预期 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 9.11.6 常用调试工具

```python
# TileLang 调试工具集

# 1. 编译后查看生成的 CUDA 代码
kernel = tilelang.compile(flash_attn_func, target="cuda")
print(kernel.get_kernel_source())  # 查看生成的 CUDA 代码

# 2. 性能分析
import tilelang.profiling as profiling

# 测量内核执行时间
latency = profiling.do_bench(
    kernel,
    args=[Q, K, V],
    warmup=100,
    rep=1000
)
print(f"延迟: {latency:.3f} ms")

# 3. 正确性验证
O_tilelang = kernel(Q, K, V)
O_reference = torch.nn.functional.scaled_dot_product_attention(
    Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0), is_causal=True
).squeeze(0)

# 逐元素比较
abs_diff = (O_tilelang - O_reference).abs()
rel_diff = abs_diff / (O_reference.abs() + 1e-6)
print(f"最大绝对误差: {abs_diff.max():.6f}")
print(f"最大相对误差: {rel_diff.max():.6f}")

# 4. 中间结果检查
# 可以在内核中添加调试输出（编译到 CPU 模式）
```

这段代码是 9.11.6 常用调试工具 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 9.12 练习题

### 练习 1: 实现基础 FlashAttention（无因果掩码）

**题目**：修改本章提供的 TileLang FlashAttention 代码，实现不带因果掩码的版本（用于编码器模型如 BERT）。

**要求**：
1. 移除因果掩码相关的代码
2. 验证输出与 `torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)` 一致
3. 测量移除掩码后的性能提升

**提示**：
- 只需删除或注释掉因果掩码相关的 `if` 语句
- 性能提升可能不明显，因为掩码操作本身开销不大

---

### 练习 2: 实现滑动窗口注意力

**题目**：使用 TileLang 实现滑动窗口注意力（Sliding Window Attention），窗口大小 $W$ 作为参数。

**要求**：
1. 实现 `W=256` 和 `W=512` 两个版本
2. 处理边界情况（序列开头和结尾的 token）
3. 验证正确性并测量性能

**核心思路**：

```python
# 滑动窗口掩码的核心逻辑
def is_in_window(q_idx, k_idx, window_size):
    """检查 k_idx 是否在 q_idx 的窗口范围内"""
    half_window = window_size // 2
    return abs(q_idx - k_idx) <= half_window

# 在内核中，对每个 S_tile 的元素应用窗口掩码
for k in range(BN):
    k_global = j * BN + k
    q_global = bx * BM + tx
    if not is_in_window(q_global, k_global, WINDOW_SIZE):
        S_tile[tx, k] = T.float32(-1e30)
```

这段代码是 练习 2: 实现滑动窗口注意力 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

### 练习 3: 优化 Tile 大小

**题目**：编写一个自动搜索最优 tile 大小的函数。

**要求**：
1. 给定 `seq_len`、`head_dim`、`sram_size`，搜索使性能最优的 `BM` 和 `BN`
2. 搜索空间：`BM, BN ∈ {16, 32, 64, 128}`
3. 输出每个配置的：
   - SRAM 使用量
   - 并行度（block 数量）
   - 算术强度
   - 预估性能评分

**参考框架**：

```python
def search_optimal_tile(seq_len, head_dim, sram_size):
    """搜索最优 tile 大小"""
    candidates = [16, 32, 64, 128]
    results = []

    for BM in candidates:
        for BN in candidates:
            sram_usage = estimate_sram_usage(BM, BN, head_dim)
            if sram_usage > sram_size:
                continue

            num_blocks = ceil(seq_len / BM)
            arith_intensity = BM  # 简化的算术强度计算

            # 综合评分（需要权衡多个因素）
            score = arith_intensity * min(num_blocks, 108) / 108

            results.append({
                "BM": BM, "BN": BN,
                "sram": sram_usage,
                "blocks": num_blocks,
                "score": score
            })

    return sorted(results, key=lambda x: -x["score"])
```

这段代码是 练习 3: 优化 Tile 大小 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

### 练习 4: 实现 FP8 FlashAttention（挑战题）

**题目**：修改 FlashAttention 内核，使用 FP8 (E4M3) 数据类型进行矩阵乘法计算。

**要求**：
1. 将 Q、K、V 转换为 FP8 格式
2. 在计算 $S = QK^T$ 时使用 FP8 Tensor Core
3. 保持 float32 精度进行 Online Softmax 更新
4. 验证精度损失在可接受范围内

**注意事项**：
- FP8 的动态范围有限，需要量化策略
- 建议使用 per-tensor scaling factor
- 反向传播可能需要使用 E5M2 格式

---

### 练习 5: 实现带 Dropout 的 FlashAttention

**题目**：在 FlashAttention 中添加 Dropout 支持，以一定概率随机屏蔽注意力权重。

**要求**：
1. 添加 `dropout_prob` 参数
2. 在计算 $P = \text{softmax}(S)$ 之后应用 Dropout
3. 使用正确的缩放：`P = P / (1 - dropout_prob)`（inverted dropout）
4. 确保反向传播时使用相同的 Dropout mask
5. 使用 RNG 状态保证可重现性

**核心思路**：

```python
# Dropout 的实现思路
import tilelang.language as T

# 在 SRAM 中生成随机数
# 注意：需要在每次迭代中生成相同的随机数序列
# 通常使用 Philox RNG 并保存状态

# 前向传播中的 Dropout
for k in range(BN):
    # 生成随机数
    rand = T.float32(T.philox_random(seed, bx * BM + tx, j * BN + k))
    # 应用 Dropout
    if rand < dropout_prob:
        P_tile[tx, k] = T.float32(0.0)
    else:
        P_tile[tx, k] = P_tile[tx, k] / (1.0 - dropout_prob)
```

这段代码是 练习 5: 实现带 Dropout 的 FlashAttention 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

### 练习 6: 对比不同精度的性能（扩展题）

**题目**：测量 FlashAttention 在不同精度（FP16、BF16、FP32）下的性能差异。

**要求**：
1. 实现支持三种精度的内核
2. 测量每种精度的：
   - 内核执行时间
   - 内存使用量
   - 输出精度（与 FP64 参考对比）
3. 绘制性能-精度权衡图

---

### 练习 7: 实现 GQA (Grouped Query Attention)

**题目**：修改 FlashAttention 以支持 GQA，其中多个 Q 头共享同一组 K、V 头。

**要求**：
1. 添加 `num_heads` 和 `num_groups` 参数
2. 实现 Q 头到 K/V 组的映射
3. 验证与标准 MHA 在 `num_groups == num_heads` 时的结果一致
4. 测量 GQA 的内存节省

---

## 9.13 总结

### 本章要点回顾

```
✅ FlashAttention 核心思想：
   - IO-Aware 算法设计
   - 避免将 N×N 中间矩阵写入 HBM
   - 在 SRAM 中分块计算

✅ Online Softmax 关键公式：
   - m_new = max(m_old, m_local)
   - l_new = exp(m_old - m_new) * l_old + l_local
   - O_new = exp(m_old - m_new) * O_old + P @ V

✅ FlashAttention-2 改进：
   - 减少非 matmul FLOPs（延迟 rescaling）
   - 更好的并行化（Q 块在外层）
   - 更好的 Warp 分区

✅ TileLang 实现优势：
   - 代码行数减少 ~75%（相比 CUDA）
   - 编译器自动处理底层优化
   - 更易于实验和调试

✅ 因果掩码处理：
   - 块级跳过优化
   - 逐元素条件判断

✅ 反向传播策略：
   - 重计算 P 矩阵
   - 保存 m, l 统计量
   - 内存-计算权衡
```

这个代码块或示意图用于说明 本章要点回顾 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 关键公式速查表

| 公式 | 表达式 | 说明 |
|------|--------|------|
| 注意力分数 | $S = QK^T / \sqrt{d}$ | Query 和 Key 的相似度 |
| Softmax | $P = \text{softmax}(S)$ | 注意力权重 |
| 输出 | $O = PV$ | Value 的加权和 |
| Online max | $m_{\text{new}} = \max(m_{\text{old}}, m_{\text{local}})$ | 更新行最大值 |
| Online sum | $\ell_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \ell_{\text{old}} + \ell_{\text{local}}$ | 更新累积分母 |
| 输出更新 | $O_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} O_{\text{old}} + PV$ | 累积输出 |

### 性能优化清单

```
FlashAttention 性能优化清单：

1. Tile 大小选择
   □ SRAM 使用率 50-80%
   □ 并行度 ≥ 2 × SM 数量
   □ 算术强度接近 roofline 拐点

2. 内存访问优化
   □ 合并访问（coalesced access）
   □ 避免 bank conflict
   □ 使用异步拷贝

3. 计算优化
   □ 使用 Tensor Core（矩阵乘法）
   □ 减少非 matmul FLOPs
   □ 使用合适的数据类型

4. 并行化策略
   □ Q 块在外层（可并行）
   □ 合理的 Warp 分区
   □ 利用 Warp 级原语
```

这个代码块或示意图用于说明 性能优化清单 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

---

## 🎯 扩展阅读

1. **FlashAttention 原始论文**: Dao, T. et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022.

2. **FlashAttention-2**: Dao, T. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." 2023.

3. **FlashAttention-3**: Shah, J. et al. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." 2024.

4. **Online Softmax**: Milakov, M. and Gimelshein, N. "Online normalizer calculation for softmax." 2018.

5. **TileLang 官方文档**: https://github.com/tile-ai/tilelang

6. **GPU 构 Roofline 模型**: Williams, S. et al. "Roofline: an insightful visual performance model for multicore architectures." Communications of the ACM, 2009.

---

## 🔮 下一章预告

**Chapter 10: FlashMLA 的 TileLang 实现**

在下一章中，我们将深入探讨 FlashMLA（Multi-head Latent Attention）的实现，这是 DeepSeek 模型中使用的高效注意力机制。我们将学习：

- MLA 的低秩压缩原理
- KV Cache 的压缩与解压
- 与标准 MHA 的性能对比
- TileLang 中的完整实现

MLA 通过将 Key 和 Value 投影到低维潜在空间，大幅减少了 KV Cache 的内存占用，是长序列推理的关键优化技术。

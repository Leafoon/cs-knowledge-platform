---
title: "Chapter 9: FlashAttention 原理与 Triton 实现"
description: "深入理解 FlashAttention 的 IO-aware 算法设计、Online Softmax 技巧、FlashAttention-1/2 的演进，并通过 Triton 完整实现走读掌握核心 kernel 编写技术"
date: "2026-06-11"
---

# Chapter 9: FlashAttention 原理与 Triton 实现

> **学习目标**：
> - 理解标准 Scaled Dot-Product Attention 的 O(N²) 内存瓶颈及其在 GPU 内存层次上的根本原因
> - 掌握 IO-aware 算法设计思想：以 FLOPs 换内存带宽，通过 Tiling 在 SRAM 中完成注意力计算
> - 理解 Online Softmax 的数学原理：如何在单遍扫描中维护 m/l 状态变量完成 softmax 归一化
> - 能够读懂并解释 FlashAttention-1 的完整算法伪代码，理解 Br/Bc 分块策略和 recomputation 策略
> - 了解 FlashAttention-2 的三大改进：减少非 matmul FLOPs、序列维度并行、warp 级优化
> - 通过完整的 Triton 实现走读，掌握 FlashAttention forward kernel 的逐行实现细节

---

## 9.1 标准 Attention 的问题

### 9.1.1 Scaled Dot-Product Attention 公式

Transformer 的核心计算是 Scaled Dot-Product Attention。给定查询矩阵 Q、键矩阵 K 和值矩阵 V：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

其中：
- Q ∈ ℝ^{N × d_k}：查询矩阵，N 为序列长度，d_k 为头维度
- K ∈ ℝ^{N × d_k}：键矩阵
- V ∈ ℝ^{N × d_v}：值矩阵
- 输出 ∈ ℝ^{N × d_v}

让我们先看一个朴素的 PyTorch 实现：

```python
import torch
import torch.nn.functional as F

def standard_attention(Q, K, V):
    """
    标准的 Scaled Dot-Product Attention 实现

    参数:
        Q: (batch, heads, N, d)  查询张量
        K: (batch, heads, N, d)  键张量
        V: (batch, heads, N, d)  值张量

    返回:
        output: (batch, heads, N, d)  注意力输出
    """
    d_k = Q.shape[-1]

    # 步骤 1: 计算注意力分数矩阵 S = Q @ K^T
    # 形状: (batch, heads, N, N)  ← 这是一个 N×N 的矩阵！
    S = torch.matmul(Q, K.transpose(-2, -1))

    # 步骤 2: 缩放
    S = S / (d_k ** 0.5)

    # 步骤 3: 应用 softmax（沿最后一个维度）
    # 注意: softmax 需要读取整行来计算归一化因子
    P = F.softmax(S, dim=-1)

    # 步骤 4: 计算输出 O = P @ V
    # 形状: (batch, heads, N, d)
    O = torch.matmul(P, V)

    return O
```

### 9.1.2 O(N²) 的内存与计算分析

朴素实现存在严重的内存问题。让我们仔细分析每一步的空间和时间复杂度：

```
标准 Attention 的复杂度分析:

┌─────────────────────────────────────────────────────────┐
│                    计算步骤                               │
├──────────────┬──────────────┬───────────────────────────┤
│    操作       │  时间复杂度   │      空间复杂度            │
├──────────────┼──────────────┼───────────────────────────┤
│ S = Q @ K^T  │  O(N² · d)   │  O(N²)  ← N×N 矩阵！     │
│ P = softmax  │  O(N²)       │  O(N²)                    │
│ O = P @ V    │  O(N² · d)   │  O(N · d)                 │
├──────────────┼──────────────┼───────────────────────────┤
│ 总计          │  O(N² · d)   │  O(N² + N·d) ≈ O(N²)     │
└──────────────┴──────────────┴───────────────────────────┘

当 N = 4096, d = 64, dtype = float16 时:
  S 矩阵大小 = 4096 × 4096 × 2 bytes = 32 MB（每个 head！）
  如果有 32 个 head: 32 × 32 MB = 1 GB（仅注意力分数矩阵）

当 N = 16384 时:
  S 矩阵大小 = 16384 × 16384 × 2 bytes = 512 MB（每个 head）
```

### 9.1.3 为什么朴素实现无法利用 GPU 内存层次

问题的核心在于**中间结果 S = QK^T 必须被物化（materialized）到 HBM 中**。

```
GPU 内存层次结构（以 A100 为例）:

┌─────────────────────────────────────────────┐
│              HBM (高带宽内存)                 │
│         容量: 80 GB                         │
│         带宽: 2 TB/s                        │
│         延迟: ~400 cycles                   │
│                                             │
│    Q, K, V, S, P, O 都存储在这里            │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │          L2 Cache                    │    │
│  │     宆量: 40 MB                      │    │
│  │     带宽: ~5 TB/s                    │    │
│  │     延迟: ~200 cycles                │    │
│  │                                     │    │
│  │  ┌───────────────────────────────┐  │    │
│  │  │       SRAM (共享内存)          │  │    │
│  │  │  容量: 192 KB / SM            │  │    │
│  │  │  带宽: ~19 TB/s               │  │    │
│  │  │  延迟: ~20 cycles             │  │    │
│  │  │                               │  │    │
│  │  │  ┌─────────────────────────┐  │  │    │
│  │  │  │    寄存器文件            │  │  │    │
│  │  │  │  容量: 256 KB / SM       │  │  │    │
│  │  │  │  带宽: ~80 TB/s          │  │  │    │
│  │  │  │  延迟: ~1 cycle          │  │  │    │
│  │  │  └─────────────────────────┘  │  │    │
│  │  └───────────────────────────────┘  │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘

关键瓶颈:
  HBM 带宽: 2 TB/s
  SRAM 带宽: 19 TB/s    ← 差 ~10 倍
  计算吞吐: ~312 TFLOPS (FP16 Tensor Core)

  算术强度 (Arithmetic Intensity) = FLOPs / Bytes
  如果 < 160 (A100 的 roofline 拐点), 则为 memory-bound
```

朴素 Attention 的问题是：

```
朴素实现的内存访问模式:

步骤 1: S = Q @ K^T
  读取: Q (N×d), K (N×d)          → 从 HBM 读取
  写入: S (N×N)                    → 写入 HBM    ← 瓶颈！

步骤 2: P = softmax(S)
  读取: S (N×N)                    → 从 HBM 读取  ← 再次读！
  写入: P (N×N)                    → 写入 HBM    ← 再次写！

步骤 3: O = P @ V
  读取: P (N×N), V (N×d)          → 从 HBM 读取  ← 第三次读！
  写入: O (N×d)                    → 写入 HBM

总计 HBM 访问量:
  读: O(N² + N·d) ≈ O(N²)    (N² 主导)
  写: O(N²)                  (S 和 P 各 N²)

当 N=4096, d=64 时:
  每个 head 的 HBM 访问 ≈ 2 × 4096² × 2 bytes + ... ≈ 64 MB

问题: S 矩阵被写了 1 次、读了 1 次，P 矩阵被写了 1 次、读了 1 次
      这些都是不必要的 HBM 往返！
```

---

## 9.2 IO-aware 算法设计

### 9.2.1 核心思想：以 FLOPs 换内存

FlashAttention 的核心洞察来自 Dao et al. 2022 的论文 *"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"*：

```
核心思想对比:

朴素实现:
  1. 计算完整的 S = QK^T    → 存入 HBM
  2. 计算完整的 P = softmax(S) → 存入 HBM
  3. 计算完整的 O = PV       → 存入 HBM

  HBM 访问: O(N²)  ← 瓶颈
  FLOPs:    O(N²d)

FlashAttention:
  1. 将 Q, K, V 分成小块
  2. 每个小块在 SRAM 中完成 QK^T + softmax + PV
  3. 用 online softmax 技巧在块间合并结果
  4. 永远不物化完整的 N×N 注意力矩阵

  HBM 访问: O(N²d² / M)  ← 大幅减少！
  FLOPs:    O(N²d)        ← 不变（或略增）
  其中 M 是 SRAM 大小
```

关键公式推导：

```
设:
  N = 序列长度
  d = 头维度
  M = SRAM 大小 (A100 上约 192 KB per SM)

分块策略: 将 Q 分成 T_r 个块（每块 Br 行），将 K,V 分成 T_c 个块（每块 Bc 行）
  T_r = ⌈N / Br⌉
  T_c = ⌈N / Bc⌉

约束: Br × d + Bc × d + Br × Bc ≤ M
  即: Q 块 + K/V 块 + 中间结果必须能放入 SRAM

HBM 访问次数分析:
  对于 Q 的每个块 (T_r 个):
    对于 K,V 的每个块 (T_c 个):
      读取 Q 块: Br × d
      读取 K 块: Bc × d
      读取 V 块: Bc × d
      (不写入 S 或 P 到 HBM!)

  总 HBM 读取 = T_r × T_c × (Br + 2×Bc) × d × sizeof(dtype)
             = (N/Br) × (N/Bc) × (Br + 2×Bc) × d × 2   (FP16)
             ≈ N² × d × 2 / Bc  (当 Br ≈ Bc 时)

  如果 Bc ≈ √M / (4d):
    HBM 访问 = N² × d² / M × 常数
```

### 9.2.2 Tiling 策略：将 Q/K/V 分块

```
Tiling 的直觉理解:

假设 N = 8, d = 2, Br = 2, Bc = 2

Q 矩阵 (8×2) 分成 4 个块:        K 矩阵 (8×2) 分成 4 个块:
┌──────────┐                      ┌──────────┐
│ Q₀ (2×2) │ ← 第 0 块            │ K₀ (2×2) │
├──────────┤                      ├──────────┤
│ Q₁ (2×2) │ ← 第 1 块            │ K₁ (2×2) │
├──────────┤                      ├──────────┤
│ Q₂ (2×2) │ ← 第 2 块            │ K₂ (2×2) │
├──────────┤                      ├──────────┤
│ Q₃ (2×2) │ ← 第 3 块            │ K₃ (2×2) │
└──────────┘                      └──────────┘

计算过程 (外层遍历 Q 块, 内层遍历 K/V 块):

for i in range(4):           # 遍历 Q 的块
    加载 Q_i 到 SRAM
    初始化 O_i = 0, l_i = 0, m_i = -inf
    for j in range(4):       # 遍历 K,V 的块
        加载 K_j, V_j 到 SRAM
        计算 S_ij = Q_i @ K_j^T          (在 SRAM 中)
        更新 m_i, l_i, O_i (online softmax)
    将 O_i 写回 HBM

关键: S_ij (2×2) 太小了，计算完就丢弃，永远不写入 HBM！
```

### 9.2.3 内存层次利用对比

| 特性 | 标准 Attention | FlashAttention |
|:---|:---|:---|
| **HBM 读写** | O(N²) | O(N²d²/M) |
| **SRAM 使用** | 几乎不用 | 充分利用 |
| **中间结果物化** | S 和 P 都写入 HBM | 从不物化 S, P |
| **FLOPs** | O(N²d) | O(N²d)（略增） |
| **精确性** | 精确 | 精确（不是近似！） |
| **recomputation** | 无 | backward 时重算 S, P |

```
Roofline 模型分析:

算术强度 = FLOPs / Bytes_transferred

标准 Attention:
  FLOPs ≈ 2N²d (两次 matmul)
  Bytes ≈ 2 × N² × 2 (读写 S 和 P)
  算术强度 ≈ d / 2 = 32 (当 d=64)
  → memory-bound! (A100 拐点 ≈ 160)

FlashAttention:
  FLOPs ≈ 2N²d (相同)
  Bytes ≈ 2N²d²/M × 2 (更少的 HBM 访问)
  算术强度 ≈ M / (2d) ≈ 1536 (当 M=192KB, d=64)
  → compute-bound! 利用了 GPU 的计算能力

            ┌─────────────────────────────────────┐
  TFLOPS    │                    ╱╱╱╱╱╱╱╱╱╱╱╱╱╱  │ ← 计算上限
            │               ╱╱╱╱╱╱                │
            │          ╱╱╱╱╱  FlashAttention       │
            │     ╱╱╱╱╱       在这里                │
            │ ╱╱╱╱                                  │
            │╱  Standard                            │
            │   Attention                           │
            │   在这里                               │
            └─────────────────────────────────────┘
              算术强度 (FLOPs/Byte)
```

---

## 9.3 Online Softmax 在 Attention 中的应用

### 9.3.1 经典 Softmax 的两遍扫描问题

经典 softmax 的数值稳定版本需要两遍扫描：

$$
\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}
$$

```python
def classic_softmax(x):
    """经典 softmax: 需要两遍扫描"""
    # 第一遍: 找最大值
    m = max(x)
    # 第二遍: 计算 exp 和归一化因子
    exp_x = [exp(xi - m) for xi in x]
    l = sum(exp_x)
    # 归一化
    return [ei / l for ei in exp_x]
```

在 Attention 中，如果我们按块处理 Q 和 K，每个 Q 块需要对所有 K 块计算 S = QK^T。如果我们天真地先遍历所有 K 块找到最大值，再遍历一次计算 softmax，就需要**两遍扫描** K/V 块——这会将 HBM 读取量翻倍。

### 9.3.2 Online Softmax 的递推公式

Online Softmax（Milakov & Gimelshein, 2018）的关键洞察是：**可以在单遍扫描中维护 softmax 的归一化因子**。

```
核心递推关系:

维护两个状态变量:
  m_i = 当前已见元素的最大值
  l_i = 当前已见元素的 exp 和（减去当前最大值）

当处理新元素 x_new 时:

  m_new = max(m_i, x_new)

  l_new = l_i × exp(m_i - m_new) + exp(x_new - m_new)
         ↑                    ↑
    旧的 exp 和需要修正    新元素的 exp 值
    (乘以修正因子)

最终:
  softmax(x_i) = exp(x_i - m_final) / l_final
```

数学推导：

```
假设我们已经处理了 x₁, ..., xₖ，维护了:
  m⁽ᵏ⁾ = max(x₁, ..., xₖ)
  l⁽ᵏ⁾ = Σᵢ₌₁ᵏ exp(xᵢ - m⁽ᵏ⁾)

现在处理 xₖ₊₁:

  m⁽ᵏ⁺¹⁾ = max(m⁽ᵏ⁾, xₖ₊₁)

  l⁽ᵏ⁺¹⁾ = Σᵢ₌₁ᵏ⁺¹ exp(xᵢ - m⁽ᵏ⁺¹⁾)
          = Σᵢ₌₁ᵏ exp(xᵢ - m⁽ᵏ⁺¹⁾) + exp(xₖ₊₁ - m⁽ᵏ⁺¹⁾)

  对于第一项:
    exp(xᵢ - m⁽ᵏ⁺¹⁾) = exp(xᵢ - m⁽ᵏ⁾ + m⁽ᵏ⁾ - m⁽ᵏ⁺¹⁾)
                       = exp(xᵢ - m⁽ᵏ⁾) × exp(m⁽ᵏ⁾ - m⁽ᵏ⁺¹⁾)

  因此:
    Σᵢ₌₁ᵏ exp(xᵢ - m⁽ᵏ⁺¹⁾) = l⁽ᵏ⁾ × exp(m⁽ᵏ⁾ - m⁽ᵏ⁺¹⁾)

  所以:
    l⁽ᵏ⁺¹⁾ = l⁽ᵏ⁾ × exp(m⁽ᵏ⁾ - m⁽ᵏ⁺¹⁾) + exp(xₖ₊₁ - m⁽ᵏ⁺¹⁾)
```

### 9.3.3 在 Attention 中应用 Online Softmax

在 FlashAttention 中，我们不是对单个向量做 softmax，而是对**注意力矩阵的每一行**做 softmax。每一行的计算涉及与所有 K 块的点积。

```
FlashAttention 的 Online Softmax 更新:

对于 Q 的第 i 块 (Q_i), 在遍历 K/V 块的过程中:

维护:
  m_i:  Q_i 与已见 K 块的注意力分数最大值 (形状: Br × 1)
  l_i:  Q_i 与已见 K 块的 exp 和      (形状: Br × 1)
  O_i:  Q_i 与已见 K/V 块的部分输出    (形状: Br × d)

当处理第 j 个 K/V 块时:

  # 1. 计算当前块的注意力分数
  S_ij = Q_i @ K_j^T / √d          (形状: Br × Bc)

  # 2. 更新最大值
  m_ij = max(S_ij, dim=-1)          (形状: Br × 1)
  m_new = max(m_i, m_ij)            逐元素取最大

  # 3. 计算修正因子
  alpha = exp(m_i - m_new)           旧的 exp 和需要修正
  beta  = exp(m_ij - m_new)          当前块的 softmax 分子

  # 4. 更新 exp 和
  l_new = alpha × l_i + sum(exp(S_ij - m_new), dim=-1)

  # 5. 更新输出
  P_ij = exp(S_ij - m_new)           当前块的 softmax 权重
  O_i  = (alpha × l_i × O_i + P_ij @ V_j) / l_new
         ↑                              ↑
    旧输出需要修正                   新的部分输出
         × 修正因子

  # 6. 更新状态
  m_i = m_new
  l_i = l_new
```

### 9.3.4 Online Softmax 的正确性证明

```
为什么 Online Softmax 能得到和经典 softmax 完全相同的结果？

证明（归纳法）:

命题: 遍历完所有 K 块后, O_i = softmax(Q_i @ K^T / √d) @ V

基础: 初始时 m_i = -∞, l_i = 0, O_i = 0
      这对应于"没有看到任何数据"的状态

归纳: 假设处理前 j 个 K 块后, 状态 (m_i, l_i, O_i) 正确地表示了
      前 j 个块的 partial softmax 结果

      处理第 j+1 个块时:
      m_new = max(m_i, m_ij) 正确维护了全局最大值
      l_new 正确维护了所有 exp 的和
      O_i 的更新等价于:
        O_new = (Σ_all exp(s - m_new) × v) / l_new
              = softmax(scores) @ V   (对前 j+1 个块)

结论: 遍历完所有 T_c 个 K/V 块后, (m_i, l_i, O_i) 给出了
      精确的 softmax 归一化结果 —— 不是近似！

关键等式:
  O_final = Σⱼ [ exp(S_ij - m_final) / l_final ] @ V_j
          = softmax(Q_i @ K^T / √d) @ V   ← 完全等价！
```

---

## 9.4 FlashAttention-1 实现

### 9.4.1 算法伪代码

FlashAttention-1 的完整算法（Dao et al., 2022）：

```
Algorithm 1: FlashAttention Forward

输入: Q, K, V ∈ ℝ^{N×d} (在 HBM 中)
参数: SRAM 大小 M, 块大小 Br, Bc
输出: O ∈ ℝ^{N×d}

1.  初始化:
2.    Q 块数 T_r = ⌈N / Br⌉
3.    K 块数 T_c = ⌈N / Bc⌉
4.    O ← zeros(N, d)           // 在 HBM 中
5.    l ← zeros(N)              // 在 HBM 中（每个行位置一个值）
6.    m ← -∞ × ones(N)         // 在 HBM 中

7.  将 Q 分成 T_r 块: Q_1, ..., Q_{T_r}，每块 Br × d
8.  将 K 分成 T_c 块: K_1, ..., K_{T_c}，每块 Bc × d
9.  将 V 分成 T_c 块: V_1, ..., V_{T_c}，每块 Bc × d
10. 将 O 分成 T_r 块: O_1, ..., O_{T_r}
11. 将 l 分成 T_r 块: l_1, ..., l_{T_r}
12. 将 m 分成 T_r 块: m_1, ..., m_{T_r}

13. for j = 1 to T_c:                              // 外层: 遍历 K/V 块
14.   从 HBM 加载 K_j, V_j 到 SRAM
15.   for i = 1 to T_r:                            // 内层: 遍历 Q 块
16.     从 HBM 加载 Q_i, O_i, l_i, m_i 到 SRAM
17.
18.     // 计算注意力分数
19.     S_ij = Q_i @ K_j^T                          // Br × Bc, 在 SRAM 中
20.
21.     // 计算当前块的行最大值
22.     m̃_ij = rowmax(S_ij)                          // Br × 1
23.
24.     // 计算 exp(注意力分数 - 新最大值)
25.     P̃_ij = exp(S_ij - m̃_ij)                      // Br × Bc, 在 SRAM 中
26.
27.     // 计算当前块的 exp 和
28.     l̃_ij = rowsum(P̃_ij)                          // Br × 1
29.
30.     // 更新全局最大值
31.     m_new = max(m_i, m̃_ij)                       // 逐元素
32.
33.     // 更新 exp 和（带修正因子）
34.     l_new = exp(m_i - m_new) × l_i
35.           + exp(m̃_ij - m_new) × l̃_ij
36.
37.     // 更新输出
38.     O_i ← diag(l_new)^{-1} × (
39.           diag(exp(m_i - m_new)) × l_i × O_i    // 修正旧输出
40.           + exp(m̃_ij - m_new) × P̃_ij @ V_j )    // 加上新贡献
41.
42.     // 更新状态
43.     m_i ← m_new
44.     l_i ← l_new
45.
46.     // 将更新后的 O_i, l_i, m_i 写回 HBM
47.     将 O_i, l_i, m_i 写回 HBM
48.
49. 返回 O
```

### 9.4.2 Block Size 选择（Br, Bc）

```
Block Size 选择的约束:

1. SRAM 容量约束:
   Br × d × sizeof(dtype)     // Q 块
 + Bc × d × sizeof(dtype)     // K 块
 + Bc × d × sizeof(dtype)     // V 块
 + Br × Bc × sizeof(dtype)    // S_ij 中间结果
 + Br × d × sizeof(dtype)     // O_i
 + Br × sizeof(dtype) × 2     // l_i, m_i
 ≤ M (SRAM 大小)

   化简 (假设 d=64, FP16):
   (3Br + 2Bc) × d × 2 + Br × Bc × 2 + Br × 4 ≤ M
   (3Br + 2Bc) × 128 + Br × Bc × 2 + Br × 4 ≤ 192 × 1024

2. 选择策略:
   - Br = Bc = √(M / (4d)) 是一个常用的选择
   - 当 d = 64, M = 192KB: Br = Bc ≈ 64
   - 当 d = 128, M = 192KB: Br = Bc ≈ 45 → 通常取 32 或 64

3. 实际常用配置 (A100):
   ┌────────────┬──────────┬──────────┐
   │ 头维度 d    │ Br       │ Bc       │
   ├────────────┼──────────┼──────────┤
   │ 64         │ 128      │ 64       │
   │ 80         │ 128      │ 32       │
   │ 128        │ 64       │ 64       │
   └────────────┴──────────┴──────────┘

   通常 Br > Bc，因为 Q 块被多次加载（每个 K/V 块都要用）
   而 K/V 块只加载一次（在当前 j 循环中被所有 Q 块共享）
```

### 9.4.3 Recomputation 策略

FlashAttention 在 forward 时**不保存** N×N 的注意力矩阵 S 和 P，这节省了大量内存。但在 backward 时，梯度计算需要用到 S 和 P。

```
Recomputation 策略:

标准 Attention:
  Forward: 保存 S (N×N), P (N×N) 到 HBM
  Backward: 从 HBM 读取 S, P 计算梯度
  内存: O(N²)

FlashAttention:
  Forward: 只保存 Q, K, V, O, l, m (共 O(N·d + N))
  Backward: 从保存的 Q, K, V, O, l, m 重新计算 S, P
            通过反向遍历 K/V 块, 利用 (l, m) 状态恢复 softmax

  内存: O(N·d + N) ≈ O(N)   ← 线性内存！

  代价: Forward 的 FLOPs 不变
        Backward 的 FLOPs 约增加 1 倍（需要重算 S, P）

  但: 由于减少了 HBM 访问, 实际 wall-clock 时间往往更快！

时间-空间权衡:
  ┌──────────────────┬────────────┬──────────────┬───────────┐
  │ 方法              │ Forward    │ 内存峰值      │ 总时间     │
  ├──────────────────┼────────────┼──────────────┼───────────┤
  │ 标准 Attention    │ 1× FLOPs   │ O(N²)        │ T_fwd + T_bwd │
  │ FlashAttention   │ 1× FLOPs   │ O(N)         │ ≈ 1.0×    │
  │ (带 recomputation)│ + 重算 S,P │              │ (更快!)   │
  └──────────────────┴────────────┴──────────────┴───────────┘
```

### 9.4.4 保存什么用于 backward

```
Forward 保存的信息 (用于 backward 时重算):

  Q, K, V:  原始输入           O(N·d)
  O:        注意力输出          O(N·d)
  l:        softmax 归一化因子  O(N)
  m:        行最大值            O(N)
  ────────────────────────────────────
  总计: O(N·d)

Backward 时:
  给定 Q_i, K_j, V_j, O_i, l_i, m_i
  可以完全恢复:
    S_ij = Q_i @ K_j^T / √d
    P_ij = exp(S_ij - m_final) / l_final

  这就是 recomputation 的核心！
  代价: 额外一次 QK^T 和 softmax 计算
  收益: 节省了 N² 的存储空间
```

---

## 9.5 FlashAttention-2 改进

### 9.5.1 减少非 matmul FLOPs

FlashAttention-2（Dao, 2023）的第一个改进是减少非矩阵乘法的浮点运算：

```
FlashAttention-1 的非 matmul 开销:

在内层循环中, FA-1 的更新公式 (简化版):
  m_new = max(m_i, rowmax(S_ij))
  P_ij = exp(S_ij - m_new)
  l_new = exp(m_i - m_new) × l_i + rowsum(P_ij)
  O_i = diag(exp(m_i - m_new)) × O_i + P_ij @ V_j
  ... (最后再除以 l_new)

问题: 每次内层循环都要做 rescaling (乘以修正因子)
      而且最后需要除以 l_new —— 这需要对 O_i 做矩阵-向量逐行乘法

FlashAttention-2 的改进:

  将 rescaling 延迟到循环结束后:
  O_i = O_i / l_i    (只在所有 K/V 块处理完后做一次!)

  内层循环中的更新简化为:
  m_new = max(m_i, rowmax(S_ij))
  P_ij = exp(S_ij - m_new)
  l_new = exp(m_i - m_new) × l_i + rowsum(P_ij)
  O_i = exp(m_i - m_new) × O_i + P_ij @ V_j    // 累加, 不除以 l

  最终:
  O_i = O_i / l_i    // 一次性归一化

  减少的非 matmul FLOPs:
    FA-1: 每次内层循环都做 rescaling → O(T_r × T_c × Br × d) 次乘法
    FA-2: 只在最后做一次 → O(T_r × Br × d) 次乘法
    减少约 T_c 倍的非 matmul FLOPs！
```

### 9.5.2 更好的并行策略

```
FlashAttention-1 的并行策略:

  外层循环: 遍历 K/V 块 (j = 1 to T_c)  ← 不可并行（依赖累积状态）
  内层循环: 遍历 Q 块 (i = 1 to T_r)    ← 可并行（不同 Q 块独立）

  并行维度: 内层 i 循环 (T_r 个并行任务)
  网格大小: grid = (batch × heads × T_r,)

  问题: 当序列很长时, T_r 很大, 但 batch × heads 可能很小
        导致 GPU 利用率不足

FlashAttention-2 的改进:

  方案 1: 在序列长度维度上并行
    grid = (batch × heads, T_r)
    每个 program 负责一个 Q 块
    不同 program 之间通过原子操作累加结果

  方案 2: 反转循环顺序
    外层: 遍历 Q 块 (i = 1 to T_r)     ← 可并行
    内层: 遍历 K/V 块 (j = 1 to T_c)   ← 顺序执行

  这样:
    每个 program 处理一个完整的 Q_i
    遍历所有 K/V 块, 维护自己的 (m_i, l_i, O_i)
    最后直接写回结果, 不需要原子操作

  grid = (batch × heads × T_r,)

  优势: 更好的负载均衡, 更高的 GPU 占用率
```

### 9.5.3 Warp 级优化

```
FlashAttention-2 的 Warp 级分工:

在每个 thread block (CTA) 内部, 将 warps 分成不同的组:

假设一个 block 有 4 个 warps (128 threads):

FlashAttention-1:
  所有 4 个 warps 协作加载 Q_i, K_j, V_j
  然后所有 warps 协作计算 S_ij = Q_i @ K_j^T
  然后所有 warps 协作计算 P_ij @ V_j

  问题: 需要在 warps 间同步 (barrier), 开销大

FlashAttention-2:
  方案: 将 warps 分成两组
    Warp 0-1: 负责计算 S_ij 的前半部分行
    Warp 2-3: 负责计算 S_ij 的后半部分行

  每个 warp 独立:
    1. 加载自己负责的 Q 行
    2. 与共享的 K_j 块计算 S_ij
    3. 计算 P_ij @ V_j
    4. 更新自己的 (m, l, O)

  优势: 减少了 warp 间同步, 更好的指令流水线
```

### 9.5.4 FlashAttention-1 vs FlashAttention-2 对比

| 特性 | FlashAttention-1 | FlashAttention-2 |
|:---|:---|:---|
| **非 matmul FLOPs** | 较多（每次内层循环 rescaling） | 较少（延迟到最终归一化） |
| **并行策略** | 内层 i 循环并行 | 外层 i 循环并行（更好的负载均衡） |
| **Warp 分工** | 协作式 | 分组独立式 |
| **A100 利用率** | ~50-70% | ~70-73% |
| **相对速度** | 1× (基准) | ~2× (在 A100 上) |
| **HBM 访问** | O(N²d²/M) | O(N²d²/M)（相同） |
| **精确性** | 精确 | 精确 |

---

## 9.6 Triton 实现走读

### 9.6.1 环境准备与完整代码

下面我们给出一个完整的、可运行的 FlashAttention forward kernel 的 Triton 实现。这个实现参考了 OpenAI Triton 官方 tutorial 的风格，并添加了详细的中文注释。

```python
"""
FlashAttention Forward Kernel — Triton 实现

参考:
  - Tri Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention
    with IO-Awareness", NeurIPS 2022
  - OpenAI Triton FlashAttention Tutorial
    https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html

本实现仅包含 forward pass, 用于教学目的。
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def _flash_attn_fwd_kernel(
    # ======== 指针参数 (指向 HBM 中的张量) ========
    Q,          # 查询张量: (batch, heads, N, d)
    K,          # 键张量:   (batch, heads, N, d)
    V,          # 值张量:   (batch, heads, N, d)
    O,          # 输出张量: (batch, heads, N, d)
    l_ptr,      # softmax 归一化因子: (batch, heads, N)  — 每行一个值
    m_ptr,      # 行最大值: (batch, heads, N)            — 每行一个值
    # ======== 标量参数 ========
    stride_qb, stride_qh, stride_qn, stride_qd,  # Q 的 stride
    stride_kb, stride_kh, stride_kn, stride_kd,  # K 的 stride
    stride_vb, stride_vh, stride_vn, stride_vd,  # V 的 stride
    stride_ob, stride_oh, stride_on, stride_od,  # O 的 stride
    N,          # 序列长度
    d,          # 头维度 (head dimension)
    scale,      # 缩放因子: 1 / sqrt(d)
    # ======== 编译期常量 ========
    BLOCK_M: tl.constexpr,    # Q 块的行数 (Br)
    BLOCK_N: tl.constexpr,    # K/V 块的行数 (Bc)
    BLOCK_D: tl.constexpr,    # 头维度的块大小 (通常等于 d)
):
    """
    FlashAttention Forward Kernel

    每个 program 处理一个 Q 块 (Br 行), 遍历所有 K/V 块 (T_c 次)。
    grid = (batch × heads × T_r,)

    与伪代码的对应:
      外层 for j (K/V 块): 由 grid 中不同 program 的内层循环实现
      内层 for i (Q 块):   每个 program 负责一个固定的 i
    """
    # ---- Step 1: 确定当前 program 负责哪个 batch、head、Q 块 ----
    # program_id(0) 是一维 grid 中的线性索引
    pid = tl.program_id(0)

    # 从线性索引反推 (batch_idx, head_idx, q_block_idx)
    # grid = (batch × heads × T_r,)
    num_q_blocks = tl.cdiv(N, BLOCK_M)    # T_r = ⌈N / Br⌉
    pid_batch = pid // (num_q_blocks)      # 等价于 pid // (heads * T_r) 如果三维展开
    # 注意: 这里简化了索引计算; 实际实现通常用三维 grid
    # 为教学目的, 我们假设 grid = (batch * heads * T_r,)
    # 并在 batch 维度上用 stride 来定位

    # 更清晰的实现: 使用二维或三维 grid
    # 这里我们用一维 grid + 除法来解码
    # (实际的 triton tutorial 通常用 batch_heads 维度合并)

    # 计算当前 program 在 Q 块维度上的索引
    q_block_idx = pid % num_q_blocks      # 第几个 Q 块
    batch_head_idx = pid // num_q_blocks   # 第几个 (batch, head) 组合

    # 解码 batch 和 head 索引
    # 假设 batch_head_idx = batch_idx * num_heads + head_idx
    # (这需要在启动 kernel 时正确设置 grid)
    # 为简化, 我们在后面的实现中直接使用二维 grid

    # ---- Step 2: 计算 Q 块在序列维度上的偏移 ----
    # q_start: 当前 Q 块的起始行号
    q_start = q_block_idx * BLOCK_M
    # q_offsets: 当前 Q 块中每一行的行号, 形状 (BLOCK_M,)
    q_offsets = q_start + tl.arange(0, BLOCK_M)
    # d_offsets: 头维度的偏移, 形状 (BLOCK_D,)
    d_offsets = tl.arange(0, BLOCK_D)

    # ---- Step 3: 加载 Q 块到 SRAM ----
    # Q 块的形状: (BLOCK_M, BLOCK_D)
    # 需要构造二维指针: Q_ptr + q_offsets[:, None] * stride_qn + d_offsets[None, :] * stride_qd
    q_ptrs = Q + (q_offsets[:, None] * stride_qn + d_offsets[None, :] * stride_qd)
    # 加载, 边界检查: 如果 q_offsets >= N, 填充 0
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < N, other=0.0)
    # q_block 形状: (BLOCK_M, BLOCK_D), 存储在 SRAM/寄存器中

    # ---- Step 4: 初始化累积状态 ----
    # m_i: 每行的当前最大值, 形状 (BLOCK_M,)
    # 初始化为负无穷 — 表示还没有看到任何 K 块
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    # l_i: 每行的 exp 和（归一化因子）, 形状 (BLOCK_M,)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # O_i: 累积输出, 形状 (BLOCK_M, BLOCK_D)
    o_i = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # ---- Step 5: 计算 K/V 块数 ----
    num_kv_blocks = tl.cdiv(N, BLOCK_N)  # T_c = ⌈N / Bc⌉

    # ---- Step 6: 遍历所有 K/V 块 (核心循环) ----
    # 这是 FlashAttention 的核心: 在 SRAM 中完成所有计算,
    # 不物化 N×N 的注意力矩阵
    for j in range(num_kv_blocks):
        # ---- 6.1: 计算 K/V 块的偏移 ----
        kv_start = j * BLOCK_N
        kv_offsets = kv_start + tl.arange(0, BLOCK_N)

        # ---- 6.2: 加载 K 块 ----
        # K 块形状: (BLOCK_N, BLOCK_D)
        k_ptrs = K + (kv_offsets[:, None] * stride_kn + d_offsets[None, :] * stride_kd)
        k_block = tl.load(k_ptrs, mask=kv_offsets[:, None] < N, other=0.0)

        # ---- 6.3: 计算注意力分数 S_ij = Q_i @ K_j^T ----
        # q_block: (BLOCK_M, BLOCK_D)
        # k_block: (BLOCK_N, BLOCK_D)
        # k_block.T: (BLOCK_D, BLOCK_N)
        # s_block: (BLOCK_M, BLOCK_N)
        s_block = tl.dot(q_block, k_block.trans(1, 0))
        # 应用缩放因子 1/√d
        s_block = s_block * scale

        # ---- 6.4: 加载 V 块 ----
        # V 块形状: (BLOCK_N, BLOCK_D)
        v_ptrs = V + (kv_offsets[:, None] * stride_vn + d_offsets[None, :] * stride_vd)
        v_block = tl.load(v_ptrs, mask=kv_offsets[:, None] < N, other=0.0)

        # ---- 6.5: Online Softmax 更新 ----
        # 当前块的行最大值: m̃_ij = rowmax(S_ij)
        # 形状: (BLOCK_M,)
        m_ij = tl.max(s_block, axis=1)

        # 更新全局最大值: m_new = max(m_i, m̃_ij)
        m_new = tl.maximum(m_i, m_ij)

        # 计算修正因子: exp(m_i - m_new)
        # 用于修正之前累积的 exp 和
        alpha = tl.exp(m_i - m_new)

        # 计算当前块的 softmax 分子: exp(S_ij - m_new)
        # 注意: 用 m_new 而不是 m_ij, 保证数值一致性
        p_block = tl.exp(s_block - m_new[:, None])

        # 计算当前块的 exp 和: l̃_ij = rowsum(P̃_ij)
        l_ij = tl.sum(p_block, axis=1)

        # 更新全局 exp 和: l_new = alpha × l_i + l̃_ij
        l_new = alpha * l_i + l_ij

        # ---- 6.6: 更新输出 ----
        # O_i = (alpha × l_i × O_i + P̃_ij @ V_j) / l_new
        # 注意: 这里我们累积到 o_i, 最后再除以 l_i
        # FA-2 风格: 延迟归一化
        # o_i = alpha × o_i + P̃_ij @ V_j
        # (l_i 的更新已经体现在 l_new 中)
        o_i = o_i * alpha[:, None]                    # 修正旧输出
        o_i = o_i + tl.dot(p_block.to(tl.float16), v_block)  # 加上新贡献

        # ---- 6.7: 更新状态变量 ----
        m_i = m_new
        l_i = l_new

    # ---- Step 7: 最终归一化 ----
    # O_i = O_i / l_i
    o_i = o_i / l_i[:, None]

    # ---- Step 8: 写回结果到 HBM ----
    o_ptrs = O + (q_offsets[:, None] * stride_on + d_offsets[None, :] * stride_od)
    # 边界检查: 只写有效的行
    tl.store(o_ptrs, o_i, mask=q_offsets[:, None] < N)

    # 可选: 保存 l 和 m 用于 backward (recomputation)
    l_ptrs = l_ptr + (batch_head_idx * N + q_offsets)
    m_ptrs = m_ptr + (batch_head_idx * N + q_offsets)
    tl.store(l_ptrs, l_i, mask=q_offsets < N)
    tl.store(m_ptrs, m_i, mask=q_offsets < N)
```

### 9.6.2 启动函数（Python 封装）

```python
def flash_attn_forward(Q, K, V):
    """
    FlashAttention Forward 的 Python 封装函数

    参数:
        Q: (batch, heads, N, d) float16/bfloat16
        K: (batch, heads, N, d) float16/bfloat16
        V: (batch, heads, N, d) float16/bfloat16

    返回:
        O: (batch, heads, N, d) float16/bfloat16

    用法:
        Q = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
        K = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
        V = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
        O = flash_attn_forward(Q, K, V)
    """
    batch, heads, N, d = Q.shape

    # 分配输出张量
    O = torch.empty_like(Q)

    # 分配 l 和 m (用于 recomputation / debugging)
    l = torch.empty((batch, heads, N), device=Q.device, dtype=torch.float32)
    m = torch.empty((batch, heads, N), device=Q.device, dtype=torch.float32)

    # 缩放因子
    scale = 1.0 / math.sqrt(d)

    # 选择块大小
    BLOCK_M = 64   # Br: Q 块的行数
    BLOCK_N = 64   # Bc: K/V 块的行数
    BLOCK_D = d    # 头维度（假设 d 是 2 的幂且 ≤ 128）

    # 计算 grid 大小
    num_q_blocks = triton.cdiv(N, BLOCK_M)
    grid = (batch * heads * num_q_blocks,)  # 一维 grid

    # 启动 kernel
    _flash_attn_fwd_kernel[grid](
        Q, K, V, O, l, m,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        N, d, scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    return O
```

### 9.6.3 Kernel 逐行详细注释版

为了更深入理解，我们把关键部分拆开，逐行解释每一步的计算逻辑和数据流：

```python
@triton.jit
def _flash_attn_fwd_kernel_detailed(
    Q, K, V, O,
    l_ptr, m_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    N, d, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    详细注释版 FlashAttention Forward Kernel

    内存访问模式图示:

    HBM (高带宽内存):
    ┌──────────────────────────────────────────┐
    │  Q [batch, heads, N, d]                  │
    │  K [batch, heads, N, d]                  │
    │  V [batch, heads, N, d]                  │
    │  O [batch, heads, N, d]  ← 写回         │
    │  l [batch, heads, N]     ← 写回         │
    │  m [batch, heads, N]     ← 写回         │
    └──────────────┬───────────────────────────┘
                   │ 加载
                   ▼
    SRAM (片上共享内存 / 寄存器):
    ┌──────────────────────────────────────────┐
    │  q_block [BLOCK_M, BLOCK_D]              │
    │  k_block [BLOCK_N, BLOCK_D]  ← 每次循环加载│
    │  v_block [BLOCK_N, BLOCK_D]  ← 每次循环加载│
    │  s_block [BLOCK_M, BLOCK_N]  ← 中间结果    │
    │  p_block [BLOCK_M, BLOCK_N]  ← 中间结果    │
    │  o_i [BLOCK_M, BLOCK_D]      ← 累积输出    │
    │  m_i [BLOCK_M]               ← 行最大值    │
    │  l_i [BLOCK_M]               ← exp 和     │
    └──────────────────────────────────────────┘
    """

    # =============================================
    # Phase 1: 初始化 — 确定当前 program 的身份
    # =============================================

    # 每个 program 处理一个 Q 块 (BLOCK_M 行)
    # grid = (batch * heads * num_q_blocks,)
    pid = tl.program_id(0)

    # 当前 Q 块的起始行号
    q_block_idx = pid  # 简化: 假设 grid 已经正确设置
    q_start = q_block_idx * BLOCK_M

    # 行偏移: [q_start, q_start+1, ..., q_start+BLOCK_M-1]
    q_offsets = q_start + tl.arange(0, BLOCK_M)
    # 维度偏移: [0, 1, ..., BLOCK_D-1]
    d_offsets = tl.arange(0, BLOCK_D)

    # =============================================
    # Phase 2: 加载 Q 块
    # =============================================

    # Q 在内存中的布局: Q[batch, head, seq, dim]
    # 通过 stride 计算指针偏移:
    #   addr = base + batch_idx * stride_b + head_idx * stride_h
    #        + row * stride_n + col * stride_d
    # 这里我们假设 Q 的指针已经偏移到了正确的 (batch, head) 位置
    # (在 Python 封装中处理)

    q_ptrs = Q + (q_offsets[:, None] * stride_qn + d_offsets[None, :] * stride_qd)
    # q_ptrs 形状: (BLOCK_M, BLOCK_D)
    # 每个元素是一个 HBM 地址

    # 加载 Q 块到 SRAM
    # mask 防止越界: 如果 q_offsets >= N, 填充 0
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < N, other=0.0)
    # q_block 形状: (BLOCK_M, BLOCK_D)
    # 此后 q_block 存在于 SRAM/寄存器中, 访问延迟 ~20 cycles

    # =============================================
    # Phase 3: 初始化累积状态
    # =============================================

    # m_i: 每行的当前最大值
    # 初始化为 -inf, 这样第一个 K 块的 max 会自动成为初始最大值
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # l_i: 每行的 exp(x - m) 的和
    # 初始化为 0, 表示还没有任何贡献
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # o_i: 累积输出, 形状 (BLOCK_M, BLOCK_D)
    # 初始化为 0
    o_i = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # =============================================
    # Phase 4: 核心循环 — 遍历 K/V 块
    # =============================================

    num_kv_blocks = tl.cdiv(N, BLOCK_N)

    for j in range(num_kv_blocks):
        # ----- 4.1: 计算 K/V 块的行偏移 -----
        kv_start = j * BLOCK_N
        kv_offsets = kv_start + tl.arange(0, BLOCK_N)

        # ----- 4.2: 加载 K 块到 SRAM -----
        k_ptrs = K + (kv_offsets[:, None] * stride_kn + d_offsets[None, :] * stride_kd)
        k_block = tl.load(k_ptrs, mask=kv_offsets[:, None] < N, other=0.0)
        # k_block: (BLOCK_N, BLOCK_D)

        # ----- 4.3: 计算注意力分数 S = Q @ K^T -----
        # 数学: S[i][j] = dot(Q[i, :], K[j, :]) / sqrt(d)
        # tl.dot 执行矩阵乘法: (BLOCK_M, BLOCK_D) @ (BLOCK_D, BLOCK_N)
        #                        = (BLOCK_M, BLOCK_N)
        s_block = tl.dot(q_block, tl.trans(k_block, 1, 0))
        s_block = s_block * scale
        # s_block: (BLOCK_M, BLOCK_N)
        # 这就是 Q_i @ K_j^T / sqrt(d)
        # 注意: 这个 N×N 的块只存在于 SRAM 中, 从不写入 HBM!

        # ----- 4.4: 加载 V 块到 SRAM -----
        v_ptrs = V + (kv_offsets[:, None] * stride_vn + d_offsets[None, :] * stride_vd)
        v_block = tl.load(v_ptrs, mask=kv_offsets[:, None] < N, other=0.0)
        # v_block: (BLOCK_N, BLOCK_D)

        # ----- 4.5: Online Softmax 更新 -----

        # (a) 当前块的行最大值
        m_ij = tl.max(s_block, axis=1)
        # m_ij: (BLOCK_M,) — 每行的最大值

        # (b) 更新全局最大值
        m_new = tl.maximum(m_i, m_ij)
        # m_new: (BLOCK_M,) — max(m_old, m_ij)

        # (c) 修正因子: exp(m_old - m_new)
        # 当 m_new > m_old 时, alpha < 1, 表示之前的 exp 和需要缩小
        # 当 m_new == m_old 时, alpha = 1, 无需修正
        alpha = tl.exp(m_i - m_new)
        # alpha: (BLOCK_M,)

        # (d) 计算当前块的 softmax 分子
        # 注意使用 m_new (全局最大值) 而不是 m_ij (当前块最大值)
        p_block = tl.exp(s_block - m_new[:, None])
        # p_block: (BLOCK_M, BLOCK_N)
        # 这就是 exp(S_ij - m_new), 不是最终的 softmax 权重!
        # 最终权重 = p_block / l_final

        # (e) 当前块的 exp 和
        l_ij = tl.sum(p_block, axis=1)
        # l_ij: (BLOCK_M,)

        # (f) 更新全局 exp 和
        l_new = alpha * l_i + l_ij
        # l_new: (BLOCK_M,)
        # 旧的 exp 和需要乘以修正因子 alpha, 加上当前块的贡献

        # ----- 4.6: 更新输出 -----
        # 公式: o_new = (alpha * l_old * o_old + p_block @ V_j) / l_new
        # FA-2 风格: 延迟除以 l, 只做累积
        # o_new = alpha * o_old + p_block @ V_j

        # 修正旧输出 (乘以 alpha)
        o_i = o_i * alpha[:, None]
        # 加上新贡献: P̃_ij @ V_j
        # tl.dot: (BLOCK_M, BLOCK_N) @ (BLOCK_N, BLOCK_D) = (BLOCK_M, BLOCK_D)
        o_i = o_i + tl.dot(p_block.to(Q.type.element_ty), v_block)

        # ----- 4.7: 更新状态变量 -----
        m_i = m_new
        l_i = l_new

    # =============================================
    # Phase 5: 最终归一化
    # =============================================

    # O_i = o_i / l_i
    # o_i 累积了 alpha * o_old + P @ V, 但没有除以 l
    # 这里一次性完成归一化
    o_i = o_i / l_i[:, None]
    # o_i: (BLOCK_M, BLOCK_D) — 最终的注意力输出

    # =============================================
    # Phase 6: 写回结果到 HBM
    # =============================================

    o_ptrs = O + (q_offsets[:, None] * stride_on + d_offsets[None, :] * stride_od)
    tl.store(o_ptrs, o_i.to(O.type.element_ty), mask=q_offsets[:, None] < N)

    # 保存 l 和 m (可选, 用于 backward recomputation 或 debugging)
    l_ptrs = l_ptr + q_offsets
    m_ptrs = m_ptr + q_offsets
    tl.store(l_ptrs, l_i, mask=q_offsets < N)
    tl.store(m_ptrs, m_i, mask=q_offsets < N)
```

### 9.6.4 二维 Grid 版本

上面的实现使用了一维 grid，索引解码比较麻烦。下面是使用二维 grid 的更清晰版本：

```python
@triton.jit
def _flash_attn_fwd_kernel_2d(
    Q, K, V, O,
    l_ptr, m_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    N, d, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    使用二维 grid 的 FlashAttention Forward Kernel

    grid = (batch * heads, num_q_blocks)

    program_id(0) → (batch, head) 组合的线性索引
    program_id(1) → Q 块索引
    """
    # ---- 解码 (batch, head) 索引 ----
    pid_batch_head = tl.program_id(0)  # batch_idx * heads + head_idx
    pid_q_block = tl.program_id(1)     # 第几个 Q 块

    # ---- Q 块的行偏移 ----
    q_start = pid_q_block * BLOCK_M
    q_offsets = q_start + tl.arange(0, BLOCK_M)
    d_offsets = tl.arange(0, BLOCK_D)

    # ---- 计算 Q, K, V, O 的基地址 ----
    # 通过 (batch, head) 索引定位到正确的 (N, d) 矩阵
    q_base = Q + pid_batch_head * stride_qh
    k_base = K + pid_batch_head * stride_kh
    v_base = V + pid_batch_head * stride_vh
    o_base = O + pid_batch_head * stride_oh

    # ---- 加载 Q 块 ----
    q_ptrs = q_base + (q_offsets[:, None] * stride_qn + d_offsets[None, :] * stride_qd)
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < N, other=0.0)

    # ---- 初始化状态 ----
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # ---- 遍历 K/V 块 ----
    for j in range(0, tl.cdiv(N, BLOCK_N)):
        kv_start = j * BLOCK_N
        kv_offsets = kv_start + tl.arange(0, BLOCK_N)

        # 加载 K_j, V_j
        k_ptrs = k_base + (kv_offsets[:, None] * stride_kn + d_offsets[None, :] * stride_kd)
        k_block = tl.load(k_ptrs, mask=kv_offsets[:, None] < N, other=0.0)

        v_ptrs = v_base + (kv_offsets[:, None] * stride_vn + d_offsets[None, :] * stride_vd)
        v_block = tl.load(v_ptrs, mask=kv_offsets[:, None] < N, other=0.0)

        # 计算 S_ij = Q_i @ K_j^T / sqrt(d)
        s_block = tl.dot(q_block, tl.trans(k_block, 1, 0)) * scale

        # Online Softmax 更新
        m_ij = tl.max(s_block, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p_block = tl.exp(s_block - m_new[:, None])
        l_ij = tl.sum(p_block, axis=1)
        l_new = alpha * l_i + l_ij

        # 更新输出
        o_i = o_i * alpha[:, None] + tl.dot(p_block.to(Q.type.element_ty), v_block)

        # 更新状态
        m_i = m_new
        l_i = l_new

    # 最终归一化
    o_i = o_i / l_i[:, None]

    # 写回
    o_ptrs = o_base + (q_offsets[:, None] * stride_on + d_offsets[None, :] * stride_od)
    tl.store(o_ptrs, o_i.to(O.type.element_ty), mask=q_offsets[:, None] < N)

    # 保存 l, m
    lm_base = l_ptr + pid_batch_head * N
    mm_base = m_ptr + pid_batch_head * N
    tl.store(lm_base + q_offsets, l_i, mask=q_offsets < N)
    tl.store(mm_base + q_offsets, m_i, mask=q_offsets < N)


def flash_attn_forward_v2(Q, K, V):
    """
    使用二维 grid 的 FlashAttention Forward 封装

    grid = (batch * heads, T_r)
    """
    batch, heads, N, d = Q.shape

    O = torch.empty_like(Q)
    l = torch.empty((batch, heads, N), device=Q.device, dtype=torch.float32)
    m = torch.empty((batch, heads, N), device=Q.device, dtype=torch.float32)

    scale = 1.0 / math.sqrt(d)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(d)

    num_q_blocks = triton.cdiv(N, BLOCK_M)
    grid = (batch * heads, num_q_blocks)

    _flash_attn_fwd_kernel_2d[grid](
        Q, K, V, O, l, m,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        N, d, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )

    return O
```

---

## 9.7 因果注意力（Causal Attention）

### 9.7.1 因果 Mask 的原理

在自回归模型（如 GPT）中，注意力必须是**因果的**：位置 i 只能关注位置 j ≤ i。这通过一个下三角 mask 实现：

```
标准因果 Mask (N=6):

         K₀  K₁  K₂  K₃  K₄  K₅
    Q₀ [ 1   0   0   0   0   0 ]
    Q₁ [ 1   1   0   0   0   0 ]
    Q₂ [ 1   1   1   0   0   0 ]
    Q₃ [ 1   1   1   1   0   0 ]
    Q₄ [ 1   1   1   1   1   0 ]
    Q₅ [ 1   1   1   1   1   1 ]

1 = 允许关注, 0 = 被 mask (设为 -∞)

朴素实现:
  S = Q @ K^T / sqrt(d)
  S = S.masked_fill(mask == 0, float('-inf'))
  P = softmax(S, dim=-1)    # mask 位置的 softmax 权重 = 0
  O = P @ V
```

### 9.7.2 FlashAttention 中的 Causal Mask 优化

在 FlashAttention 的分块计算中，我们不需要物化完整的 mask 矩阵。关键观察是：

```
分块计算中的因果 Mask:

将注意力分数矩阵分块 (BLOCK_M=2, BLOCK_N=2):

         K块₀  K块₁  K块₂  K块₃
    Q块₀ [  ?     -     -     -  ]    ? = 需要计算 (全在对角线之上)
    Q块₁ [  1     ?     -     -  ]    1 = 全部有效
    Q块₂ [  1     1     ?     -  ]    - = 全部被 mask
    Q块₃ [  1     1     1     ?  ]

三种情况:
  1. 整个块在对角线之上 (j > i): 整块被 mask → 跳过！不需要计算
  2. 整个块在对角线之下 (j < i): 整块有效 → 正常计算
  3. 块跨越对角线 (j == i): 部分有效 → 需要逐元素 mask

优化:
  情况 1: 直接跳过整个 K/V 块 (节省计算和内存访问!)
  情况 2: 不需要任何 mask 操作
  情况 3: 只在块内应用 mask
```

### 9.7.3 因果 Attention 的 Triton 实现

```python
@triton.jit
def _flash_attn_fwd_causal_kernel(
    Q, K, V, O,
    l_ptr, m_ptr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    N, d, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    FlashAttention Forward Kernel — 支持 Causal Mask

    关键改动:
      1. 计算当前 Q 块和 K 块的因果关系
      2. 跳过完全在对角线之上的 K 块
      3. 对跨越对角线的块应用 mask
    """
    # ---- 初始化 (与非因果版本相同) ----
    pid_batch_head = tl.program_id(0)
    pid_q_block = tl.program_id(1)

    q_start = pid_q_block * BLOCK_M
    q_offsets = q_start + tl.arange(0, BLOCK_M)
    d_offsets = tl.arange(0, BLOCK_D)

    q_base = Q + pid_batch_head * stride_qh
    k_base = K + pid_batch_head * stride_kh
    v_base = V + pid_batch_head * stride_vh
    o_base = O + pid_batch_head * stride_oh

    q_ptrs = q_base + (q_offsets[:, None] * stride_qn + d_offsets[None, :] * stride_qd)
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < N, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # ---- 因果 Mask 的关键: 计算有效的 K 块范围 ----
    # 对于 Q 块 [q_start, q_start + BLOCK_M):
    #   位置 i 只能关注 j <= i
    #   因此最小的有效 K 位置 = 0
    #   最大的有效 K 位置 = q_start + BLOCK_M - 1
    #   所以有效的 K 块数 = ceil((q_start + BLOCK_M) / BLOCK_N)
    num_kv_blocks = tl.cdiv(N, BLOCK_N)

    # 因果 mask 的优化: 最后一个需要处理的 K 块
    # 位置 q_start 的最后一个可见 K 位置是 q_start
    # 所以最后一个需要处理的 K 块是 q_start // BLOCK_N
    # 但要注意: Q 块的最后一行 q_start + BLOCK_M - 1 可以看到更多 K
    causal_num_kv_blocks = tl.minimum(
        tl.cdiv(q_start + BLOCK_M, BLOCK_N),
        num_kv_blocks
    )

    # ---- 遍历 K/V 块 ----
    for j in range(0, causal_num_kv_blocks):
        kv_start = j * BLOCK_N
        kv_offsets = kv_start + tl.arange(0, BLOCK_N)

        # 加载 K_j, V_j
        k_ptrs = k_base + (kv_offsets[:, None] * stride_kn + d_offsets[None, :] * stride_kd)
        k_block = tl.load(k_ptrs, mask=kv_offsets[:, None] < N, other=0.0)

        v_ptrs = v_base + (kv_offsets[:, None] * stride_vn + d_offsets[None, :] * stride_vd)
        v_block = tl.load(v_ptrs, mask=kv_offsets[:, None] < N, other=0.0)

        # 计算 S_ij
        s_block = tl.dot(q_block, tl.trans(k_block, 1, 0)) * scale

        # ---- 应用因果 Mask ----
        # 对于 Q 块中的每一行 q_row, K 块中的每一列 k_col:
        #   如果 k_col > q_row, 则被 mask (设为 -inf)
        # 构造二维 mask: q_offsets[:, None] >= kv_offsets[None, :]
        causal_mask = q_offsets[:, None] >= kv_offsets[None, :]
        # causal_mask: (BLOCK_M, BLOCK_N), bool 类型
        # True = 有效, False = 被 mask

        # 将被 mask 的位置设为 -inf
        s_block = tl.where(causal_mask, s_block, -float("inf"))
        # 被 mask 的位置在 softmax 后权重为 0

        # ---- Online Softmax 更新 ----
        # 注意: 被 mask 的位置值为 -inf, max 和 sum 会正确处理
        m_ij = tl.max(s_block, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p_block = tl.exp(s_block - m_new[:, None])
        l_ij = tl.sum(p_block, axis=1)
        l_new = alpha * l_i + l_ij

        o_i = o_i * alpha[:, None] + tl.dot(p_block.to(Q.type.element_ty), v_block)

        m_i = m_new
        l_i = l_new

    # 最终归一化
    o_i = o_i / l_i[:, None]

    # 写回
    o_ptrs = o_base + (q_offsets[:, None] * stride_on + d_offsets[None, :] * stride_od)
    tl.store(o_ptrs, o_i.to(O.type.element_ty), mask=q_offsets[:, None] < N)

    lm_base = l_ptr + pid_batch_head * N
    mm_base = m_ptr + pid_batch_head * N
    tl.store(lm_base + q_offsets, l_i, mask=q_offsets < N)
    tl.store(mm_base + q_offsets, m_i, mask=q_offsets < N)
```

### 9.7.4 Causal Mask 的性能优化技巧

```
Causal Mask 的额外优化:

优化 1: 提前终止循环
  对于 Q 块 [q_start, q_start + BLOCK_M):
  只需要遍历到 j = ceil((q_start + BLOCK_M) / BLOCK_N)
  后面的 K 块完全在对角线之上, 可以跳过

  节省的计算量:
    对于最后一个 Q 块, 几乎只计算一半的 K 块
    平均而言, 约节省 50% 的计算 (但实际取决于块大小)

优化 2: 整块 mask 判断
  如果整个 K 块都在 Q 块的对角线之上:
    即 kv_start > q_start + BLOCK_M - 1
  则跳过整个块, 不需要计算 S_ij

  在循环条件中已经通过 causal_num_kv_blocks 实现了

优化 3: 减少 mask 计算开销
  causal_mask 的构造: q_offsets[:, None] >= kv_offsets[None, :]
  这是一个简单的整数比较, 开销很小

  但如果 BLOCK_M 和 BLOCK_N 很大, 这个 mask 矩阵也很大
  可以考虑用范围判断替代逐元素比较:
    如果整个 Q 行都能看到整个 K 块 → 不需要 mask
    如果整个 Q 行都看不到 K 块 → 整块跳过
    否则 → 需要逐元素 mask

  实际中, 由于 BLOCK_M 和 BLOCK_N 通常为 64-128,
  逐元素 mask 的开销可以接受
```

### 9.7.5 封装函数（支持 causal）

```python
def flash_attn_forward(Q, K, V, causal=False):
    """
    FlashAttention Forward — 支持因果和非因果模式

    参数:
        Q, K, V: (batch, heads, N, d) float16/bfloat16
        causal:  是否使用因果 mask

    返回:
        O: (batch, heads, N, d)
    """
    batch, heads, N, d = Q.shape
    O = torch.empty_like(Q)
    l = torch.empty((batch, heads, N), device=Q.device, dtype=torch.float32)
    m = torch.empty((batch, heads, N), device=Q.device, dtype=torch.float32)

    scale = 1.0 / math.sqrt(d)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(d)

    num_q_blocks = triton.cdiv(N, BLOCK_M)
    grid = (batch * heads, num_q_blocks)

    if causal:
        _flash_attn_fwd_causal_kernel[grid](
            Q, K, V, O, l, m,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            N, d, scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        )
    else:
        _flash_attn_fwd_kernel_2d[grid](
            Q, K, V, O, l, m,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            N, d, scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        )

    return O
```

### 9.7.6 验证正确性

```python
def verify_flash_attention():
    """
    验证 Triton FlashAttention 实现与 PyTorch 标准实现的一致性
    """
    torch.manual_seed(42)

    batch, heads, N, d = 2, 4, 1024, 64
    Q = torch.randn(batch, heads, N, d, device='cuda', dtype=torch.float16)
    K = torch.randn(batch, heads, N, d, device='cuda', dtype=torch.float16)
    V = torch.randn(batch, heads, N, d, device='cuda', dtype=torch.float16)

    # PyTorch 标准实现
    scale = 1.0 / math.sqrt(d)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    P = torch.softmax(S.float(), dim=-1).half()
    O_ref = torch.matmul(P, V)

    # Triton FlashAttention (非因果)
    O_triton = flash_attn_forward(Q, K, V, causal=False)

    # 比较
    diff = (O_ref - O_triton).abs().max().item()
    print(f"非因果模式最大误差: {diff:.6f}")
    assert diff < 1e-2, f"误差过大: {diff}"

    # 因果模式
    causal_mask = torch.triu(torch.ones(N, N, device='cuda'), diagonal=1).bool()
    S_causal = S.masked_fill(causal_mask[None, None, :, :], float('-inf'))
    P_causal = torch.softmax(S_causal.float(), dim=-1).half()
    O_ref_causal = torch.matmul(P_causal, V)

    O_triton_causal = flash_attn_forward(Q, K, V, causal=True)

    diff_causal = (O_ref_causal - O_triton_causal).abs().max().item()
    print(f"因果模式最大误差: {diff_causal:.6f}")
    assert diff_causal < 1e-2, f"因果误差过大: {diff_causal}"

    print("✓ 验证通过!")


if __name__ == "__main__":
    verify_flash_attention()
```

---

## 9.8 性能分析与对比

### 9.8.1 FlashAttention vs 标准 PyTorch Attention

```python
import time

def benchmark_attention():
    """
    对比 FlashAttention 与 PyTorch 标准 Attention 的性能
    """
    torch.manual_seed(42)

    configs = [
        # (batch, heads, N, d)
        (2, 8, 512, 64),
        (2, 8, 1024, 64),
        (2, 8, 2048, 64),
        (2, 8, 4096, 64),
        (2, 8, 8192, 64),
        (1, 8, 16384, 64),
    ]

    print(f"{'N':>6} | {'PyTorch (ms)':>14} | {'FlashAttn (ms)':>14} | {'Speedup':>8} | {'Mem PyTorch (MB)':>16} | {'Mem Flash (MB)':>16}")
    print("-" * 90)

    for batch, heads, N, d in configs:
        Q = torch.randn(batch, heads, N, d, device='cuda', dtype=torch.float16)
        K = torch.randn(batch, heads, N, d, device='cuda', dtype=torch.float16)
        V = torch.randn(batch, heads, N, d, device='cuda', dtype=torch.float16)

        # PyTorch 标准实现
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        for _ in range(10):
            S = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
            P = torch.softmax(S, dim=-1)
            O_ref = torch.matmul(P, V)
        torch.cuda.synchronize()
        mem_pytorch = torch.cuda.max_memory_allocated() / 1024 / 1024

        # FlashAttention
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        for _ in range(10):
            O_flash = flash_attn_forward(Q, K, V, causal=False)
        torch.cuda.synchronize()
        mem_flash = torch.cuda.max_memory_allocated() / 1024 / 1024

        # 正确性验证
        diff = (O_ref.half() - O_flash).abs().max().item()

        print(f"{N:>6} | {'':>14} | {'':>14} | {'':>8} | {mem_pytorch:>14.1f} | {mem_flash:>14.1f} | diff={diff:.4f}")

    print("\n注意: 完整的 benchmark 需要使用 torch.cuda.Event 计时,")
    print("此处仅展示内存对比的框架代码。")
```

### 9.8.2 内存 Scaling 行为

```
不同序列长度下的内存使用对比:

序列长度 N │ 标准 Attention │ FlashAttention │ 内存节省
            │ (N² 矩阵)      │ (O(N·d))       │
────────────┼────────────────┼────────────────┼──────────
    512     │     2 MB       │    0.25 MB     │   8×
   1024     │     8 MB       │    0.5 MB      │  16×
   2048     │    32 MB       │    1.0 MB      │  32×
   4096     │   128 MB       │    2.0 MB      │  64×
   8192     │   512 MB       │    4.0 MB      │ 128×
  16384     │  2048 MB       │    8.0 MB      │ 256×
  32768     │  8192 MB       │   16.0 MB      │ 512×

计算公式:
  标准 Attention: 2 × N² × 2 bytes (S 和 P, FP16)
  FlashAttention: N × d × 2 bytes (Q, K, V, O, FP16) + N × 8 bytes (l, m, FP32)
                ≈ 4Nd + 8N bytes

注: 这是每个 head 的内存; 实际总内存需乘以 batch × heads
```

### 9.8.3 延迟与吞吐对比

```
FlashAttention vs 标准 Attention 性能对比 (A100-80GB):

┌──────────────────────────────────────────────────────────────────┐
│ 延迟 (ms) — batch=2, heads=8, d=64, FP16                        │
│                                                                  │
│  N      │ PyTorch │ FlashAttn │ xformers  │ FlashAttn-2          │
│─────────┼─────────┼───────────┼───────────┼─────────────────────│
│   512   │   0.5   │    0.3    │    0.3    │     0.2              │
│  1024   │   1.8   │    0.8    │    0.7    │     0.5              │
│  2048   │   7.0   │    2.5    │    2.2    │     1.5              │
│  4096   │  28.0   │    8.0    │    7.5    │     5.0              │
│  8192   │ 112.0   │   28.0    │   26.0    │    17.0              │
│ 16384   │ OOM     │   95.0    │   88.0    │    58.0              │
└──────────────────────────────────────────────────────────────────┘

观察:
  1. 标准 Attention 在 N=16384 时 OOM (内存不足)
  2. FlashAttention 的加速比随 N 增大而增大
  3. FlashAttention-2 比 FlashAttention-1 快约 1.5-2×

┌──────────────────────────────────────────────────────────────────┐
│ 吞吐 (tokens/sec) — batch=2, heads=8, d=64, FP16               │
│                                                                  │
│  N      │ PyTorch │ FlashAttn │ FlashAttn-2                     │
│─────────┼─────────┼───────────┼─────────────────────────────────│
│   512   │  2.0M   │   3.4M    │    5.1M                         │
│  1024   │  1.1M   │   2.6M    │    4.1M                         │
│  2048   │  0.6M   │   1.6M    │    2.7M                         │
│  4096   │  0.3M   │   1.0M    │    1.6M                         │
│  8192   │  0.15M  │   0.6M    │    1.0M                         │
└──────────────────────────────────────────────────────────────────┘
```

### 9.8.4 不同头维度的影响

```
头维度 d 对 FlashAttention 性能的影响 (N=4096, batch=2, heads=8):

┌─────────────────────────────────────────────────┐
│  d    │ Block Size │ HBM 访问 │ 相对速度        │
│───────┼────────────┼──────────┼────────────────│
│  32   │ Br=Bc=128  │   最少   │   1.0× (基准)  │
│  64   │ Br=Bc=64   │   2×     │   0.7×         │
│  128  │ Br=64,Bc=32│   4×     │   0.4×         │
│  256  │ Br=Bc=16   │   16×    │   0.15×        │
└─────────────────────────────────────────────────┘

分析:
  d 越大 → 每个块能容纳的行列数越少 → 循环次数越多
  → HBM 访问次数增加 → 性能下降

  这是因为 SRAM 大小固定 (192 KB), d 增大意味着:
    Q 块: Br × d × 2 bytes 占用更多 SRAM
    K 块: Bc × d × 2 bytes 占用更多 SRAM
    → Br 和 Bc 必须减小以适应 SRAM

  这就是为什么 FlashAttention 在小 d (如 64) 时效果最好
```

### 9.8.5 Roofline 模型分析

```
A100 Roofline 模型:

                    FP16 Tensor Core 峰值: 312 TFLOPS
                    HBM 带宽: 2 TB/s
                    拐点: 312T / 2T = 156 FLOPs/Byte

  TFLOPS
    │
312 │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
    │                     ╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱
    │                ╱╱╱╱╱
    │           ╱╱╱╱╱
    │      ╱╱╱╱╱   FlashAttention (d=64, N=4096)
    │  ╱╱╱╱        算术强度 ≈ M/(4d) ≈ 768
    │╱╱            → 接近计算上限!
    │╱ Standard Attention (d=64, N=4096)
    │  算术强度 ≈ d/2 = 32
    │  → 严重 memory-bound
    └──────────────────────────────────────────────────────
       1     10    100   1000  FLOPs/Byte

FlashAttention 将注意力计算从 memory-bound 区域
移动到了 compute-bound 区域, 这就是加速的根本原因。
```

### 9.8.6 使用 PyTorch 内置 FlashAttention

在实际项目中，推荐直接使用 PyTorch 2.0+ 内置的 `scaled_dot_product_attention`，它自动选择 FlashAttention 后端：

```python
import torch
import torch.nn.functional as F

# PyTorch 2.0+ 内置 FlashAttention
def pytorch_flash_attention(Q, K, V, causal=False):
    """
    使用 PyTorch 内置的 FlashAttention

    PyTorch 会自动选择最优后端:
      - FlashAttention (如果安装了 flash-attn 包)
      - Memory-efficient attention (xformers 风格)
      - 标准 attention (fallback)
    """
    return F.scaled_dot_product_attention(
        Q, K, V,
        is_causal=causal,
        # 可选: scale=1.0/math.sqrt(d)  (默认自动计算)
    )

# 验证
Q = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
K = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
V = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)

O = pytorch_flash_attention(Q, K, V, causal=True)
print(f"输出形状: {O.shape}")  # (2, 8, 1024, 64)

# 查看实际使用的后端
# 通过 torch.backends.cuda.sdp_kernel 上下文管理器可以选择后端
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False,
):
    O_flash = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    print("使用 FlashAttention 后端")
```

---

## 9.9 完整可运行示例

### 9.9.1 端到端测试代码

```python
"""
完整的 FlashAttention 测试脚本

使用方法:
    python flash_attention_demo.py

依赖:
    pip install torch triton
    (需要 CUDA GPU)
"""

import torch
import triton
import triton.language as tl
import math
import time


# ============================================================
# Triton Kernel: FlashAttention Forward (非因果 + 因果)
# ============================================================

@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, O,
    l_ptr, m_ptr,
    stride_qh, stride_qn, stride_qd,
    stride_kh, stride_kn, stride_kd,
    stride_vh, stride_vn, stride_vd,
    stride_oh, stride_on, stride_od,
    N, d, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """
    FlashAttention Forward Kernel

    grid = (heads * T_r,)
    每个 program 处理一个 (head, Q块) 对
    """
    pid = tl.program_id(0)
    num_q_blocks = tl.cdiv(N, BLOCK_M)

    head_idx = pid // num_q_blocks
    q_block_idx = pid % num_q_blocks

    q_start = q_block_idx * BLOCK_M
    q_offsets = q_start + tl.arange(0, BLOCK_M)
    d_offsets = tl.arange(0, BLOCK_D)

    # 定位到正确的 (head, N, d) 切片
    q_base = Q + head_idx * stride_qh
    k_base = K + head_idx * stride_kh
    v_base = V + head_idx * stride_vh
    o_base = O + head_idx * stride_oh

    # 加载 Q 块
    q_ptrs = q_base + (q_offsets[:, None] * stride_qn + d_offsets[None, :] * stride_qd)
    q_mask = q_offsets[:, None] < N
    q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # 初始化状态
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # 计算因果模式下的有效 K 块数
    num_kv_blocks = tl.cdiv(N, BLOCK_N)
    if IS_CAUSAL:
        # 因果模式: 只处理对角线以下的 K 块
        causal_kv_blocks = tl.minimum(tl.cdiv(q_start + BLOCK_M, BLOCK_N), num_kv_blocks)
    else:
        causal_kv_blocks = num_kv_blocks

    # 遍历 K/V 块
    for j in range(0, causal_kv_blocks):
        kv_start = j * BLOCK_N
        kv_offsets = kv_start + tl.arange(0, BLOCK_N)

        # 加载 K_j
        k_ptrs = k_base + (kv_offsets[:, None] * stride_kn + d_offsets[None, :] * stride_kd)
        k_mask = kv_offsets[:, None] < N
        k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # 加载 V_j
        v_ptrs = v_base + (kv_offsets[:, None] * stride_vn + d_offsets[None, :] * stride_vd)
        v_block = tl.load(v_ptrs, mask=k_mask, other=0.0)

        # S_ij = Q_i @ K_j^T / sqrt(d)
        s_block = tl.dot(q_block, tl.trans(k_block, 1, 0)) * scale

        # 应用因果 mask
        if IS_CAUSAL:
            causal_mask = q_offsets[:, None] >= kv_offsets[None, :]
            s_block = tl.where(causal_mask, s_block, -float("inf"))

        # Online Softmax 更新
        m_ij = tl.max(s_block, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p_block = tl.exp(s_block - m_new[:, None])
        l_ij = tl.sum(p_block, axis=1)
        l_new = alpha * l_i + l_ij

        # 更新输出
        o_i = o_i * alpha[:, None] + tl.dot(p_block.to(Q.type.element_ty), v_block)

        # 更新状态
        m_i = m_new
        l_i = l_new

    # 最终归一化
    o_i = o_i / l_i[:, None]

    # 写回
    o_ptrs = o_base + (q_offsets[:, None] * stride_on + d_offsets[None, :] * stride_od)
    tl.store(o_ptrs, o_i.to(O.type.element_ty), mask=q_offsets[:, None] < N)

    # 保存 l, m
    l_base = l_ptr + head_idx * N
    m_base = m_ptr + head_idx * N
    tl.store(l_base + q_offsets, l_i, mask=q_offsets < N)
    tl.store(m_base + q_offsets, m_i, mask=q_offsets < N)


def flash_attn(Q, K, V, causal=False):
    """FlashAttention Forward — 单 batch 封装"""
    heads, N, d = Q.shape

    O = torch.empty_like(Q)
    l = torch.empty((heads, N), device=Q.device, dtype=torch.float32)
    m = torch.empty((heads, N), device=Q.device, dtype=torch.float32)

    scale = 1.0 / math.sqrt(d)
    BLOCK_M = min(64, triton.next_power_of_2(N))
    BLOCK_N = min(64, triton.next_power_of_2(N))
    BLOCK_D = triton.next_power_of_2(d)

    num_q_blocks = triton.cdiv(N, BLOCK_M)
    grid = (heads * num_q_blocks,)

    _flash_attn_fwd_kernel[grid](
        Q, K, V, O, l, m,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        N, d, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        IS_CAUSAL=causal,
    )

    return O


# ============================================================
# 参考实现 (PyTorch)
# ============================================================

def reference_attention(Q, K, V, causal=False):
    """标准 PyTorch Attention 参考实现"""
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if causal:
        N = Q.shape[-2]
        mask = torch.triu(torch.ones(N, N, device=Q.device), diagonal=1).bool()
        S = S.masked_fill(mask, float('-inf'))
    P = torch.softmax(S.float(), dim=-1).half()
    O = torch.matmul(P, V)
    return O


# ============================================================
# 测试
# ============================================================

def main():
    print("=" * 60)
    print("FlashAttention Triton 实现 — 测试与验证")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("错误: 需要 CUDA GPU")
        return

    device = 'cuda'
    dtype = torch.float16

    test_cases = [
        (8, 256, 64),
        (8, 512, 64),
        (8, 1024, 64),
        (8, 2048, 64),
        (4, 4096, 64),
        (4, 8192, 64),
    ]

    for heads, N, d in test_cases:
        print(f"\n测试: heads={heads}, N={N}, d={d}")

        Q = torch.randn(heads, N, d, device=device, dtype=dtype)
        K = torch.randn(heads, N, d, device=device, dtype=dtype)
        V = torch.randn(heads, N, d, device=device, dtype=dtype)

        for causal in [False, True]:
            mode = "causal" if causal else "non-causal"

            # 参考实现
            O_ref = reference_attention(Q, K, V, causal=causal)

            # Triton 实现
            O_triton = flash_attn(Q, K, V, causal=causal)

            # 比较
            diff = (O_ref - O_triton).abs().max().item()
            status = "✓" if diff < 0.05 else "✗"
            print(f"  {mode:>12}: max_diff = {diff:.6f} {status}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

---

## 本章小结

本章深入介绍了 FlashAttention 的原理与 Triton 实现。核心要点如下：

1. **标准 Attention 的瓶颈**：朴素实现需要物化 N×N 的注意力矩阵 S 和 P，导致 O(N²) 的 HBM 读写。当序列长度增大时，内存访问成为主要瓶颈（memory-bound）。

2. **IO-aware 算法设计**：FlashAttention 通过 Tiling 策略将 Q/K/V 分成小块，在 SRAM 中完成所有中间计算，避免将 N×N 矩阵写入 HBM。核心思想是**以 FLOPs 换内存带宽**——虽然 FLOPs 略有增加，但由于 HBM 访问大幅减少，整体性能反而提升。

3. **Online Softmax**：通过维护 m（行最大值）和 l（exp 和）两个状态变量，可以在单遍扫描中完成 softmax 的归一化计算。递推公式 `l_new = exp(m_old - m_new) × l_old + exp(m_ij - m_new) × l_ij` 保证了数值稳定性。

4. **FlashAttention-1**：首次实现了 IO-aware 的精确注意力。关键参数 Br（Q 块行数）和 Bc（K/V 块行数）需要根据 SRAM 大小选择。通过 recomputation 策略，backward 时重算 S 和 P，将内存从 O(N²) 降到 O(N)。

5. **FlashAttention-2**：三大改进——减少非 matmul FLOPs（延迟归一化）、序列维度并行（更好的 GPU 利用率）、warp 级优化（减少同步开销）。在 A100 上实现了约 2× 的加速。

6. **Triton 实现**：完整的 forward kernel 约 80-100 行代码，核心是 `tl.dot` 计算 S = QK^T、Online Softmax 更新、`tl.dot` 计算 PV。二维 grid 结构清晰：`(heads, num_q_blocks)`。

7. **因果注意力**：通过 `causal_kv_blocks` 提前终止循环，以及 `q_offsets[:, None] >= kv_offsets[None, :]` 构造块内 mask，高效支持自回归模型。

8. **性能特征**：FlashAttention 在长序列（N≥1024）下优势明显，且内存使用随 N 线性增长而非二次增长。头维度 d 越小，分块效率越高。

---

## 思考题

### 概念理解题

1. **IO-aware 的含义**：FlashAttention 论文标题中的 "IO-Aware" 具体指什么？为什么传统 Attention 实现不是 IO-aware 的？

2. **Online Softmax 的正确性**：在线 softmax 递推公式中，为什么需要将旧的 exp 和乘以 `exp(m_old - m_new)` 这个修正因子？如果不修正会怎样？

3. **精确 vs 近似**：FlashAttention 是精确的还是近似的？它和 Linformer、Performer 等近似注意力方法有什么本质区别？

4. **Recomputation 的权衡**：FlashAttention 在 backward 时重算 S 和 P，而不是保存它们。这个策略在什么情况下可能不是最优的？

5. **Block Size 选择**：为什么 FlashAttention 通常选择 Br > Bc（例如 Br=128, Bc=64）而不是 Br = Bc？从 SRAM 利用率和循环结构的角度分析。

### 实践题

6. **实现反向传播**：基于本章的 forward kernel，设计 backward kernel 的伪代码。提示：需要重新计算 S 和 P，以及计算 dQ、dK、dV。

7. **GQA/MQA 支持**：Grouped Query Attention (GQA) 中，多个 Q head 共享同一组 K/V head。修改本章的 kernel 以支持 GQA，说明需要改动哪些部分。

8. **性能调优**：在你的 GPU 上运行本章的测试代码，尝试不同的 BLOCK_M 和 BLOCK_N 组合（32/64/128），绘制性能随块大小变化的曲线。

### 设计思考题

9. **FlashAttention-3**：FlashAttention-2 进一步优化了 warp 分工和指令调度。如果你要设计 FlashAttention-3，你会从哪些方面入手？（提示：考虑 Hopper 架构的新特性，如 TMA、WGMMA）

10. **Ring Attention**：对于超长序列（N > 1M），单个 GPU 的 SRAM 无法容纳。Ring Attention 通过在多个 GPU 间分布式计算来处理超长序列。设计一个基于 FlashAttention 分块策略的 Ring Attention 方案。

11. **与 KV Cache 的关系**：在 LLM 推理中，KV Cache 存储了历史 token 的 K 和 V。FlashAttention 的分块策略如何与 KV Cache 配合？当新 token 的 Q 块与历史 KV 交互时，有什么特殊考虑？

### 进阶题

12. **编译器分析**：使用 `TRITON_PRINT_AUTOTUNING=1` 运行本章的 kernel，分析 Triton 编译器生成的 PTX 代码。重点关注 `tl.dot` 被编译成了什么指令（Tensor Core 指令？），以及 SRAM 的使用情况。

13. **跨平台适配**：FlashAttention 的 Triton 实现在 AMD MI250 GPU 上可能需要调整哪些参数？AMD 的 SRAM 大小和 Tensor Core 等效单元（Matrix Core）有什么不同？

14. **Flash-Decoding**：在 LLM 推理的 decode 阶段，Q 只有 1 行（单个新 token），但 K/V 有 N 行（完整 KV Cache）。设计一个针对这个场景优化的 FlashAttention kernel，说明与本章实现的主要差异。

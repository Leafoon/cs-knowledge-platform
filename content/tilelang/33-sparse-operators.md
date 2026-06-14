---
title: "Chapter 33: 稀疏算子与结构化稀疏"
description: "深入理解 TileLang 在稀疏计算领域的应用：Structured Sparsity（2:4 稀疏）、Block Sparse、Sparse FlashAttention 实现，掌握 TileLang 的 Tile 抽象如何简化稀疏算子的编程复杂度"
updated: 2026-06-11
---

# Chapter 33: 稀疏算子与结构化稀疏

> **Learning Objectives**
>
> 1. 理解稀疏计算的基本概念与分类（结构化/非结构化稀疏）
> 2. 掌握 2:4 结构化稀疏的原理与 NVIDIA 硬件支持
> 3. 学会使用 TileLang 实现 Block Sparse 算子
> 4. 理解 Sparse FlashAttention 的设计与实现
> 5. 掌握 Sparse GEMM 的 TileLang 实现方法
> 6. 了解稀疏格式（CSR/CSC/BSR）的选择策略
> 7. 分析稀疏 vs 稠密的性能特征与适用场景

---

## 33.1 稀疏计算基础

### 33.1.1 什么是稀疏计算

稀疏计算是指在矩阵或张量中，大部分元素为零的情况下，利用这种稀疏性来减少计算量和内存占用的技术。在深度学习中，稀疏性可以来源于：

```
稀疏性的来源：

1. 权重剪枝 (Weight Pruning)
   ├── 非结构化剪枝: 随机移除权重
   └── 结构化剪枝: 按通道/块移除

2. 激活稀疏 (Activation Sparsity)
   ├── ReLU 后的零值激活
   └── Top-K 稀疏注意力

3. 专家稀疏 (MoE Sparsity)
   ├── 每个 token 只激活 K 个专家
   └── 其他专家的计算为零

4. 注意力稀疏 (Attention Sparsity)
   ├── 局部注意力窗口
   └── 稀疏注意力模式
```

<div data-component="SparsityTypeComparison"></div>

### 33.1.2 稀疏类型对比

| 稀疏类型 | 定义 | 硬件支持 | 加速比 | 编程难度 |
|----------|------|---------|--------|---------|
| 非结构化稀疏 | 任意位置为零 | 有限 | 1.5-2x | 高 |
| 2:4 结构化稀疏 | 每 4 个元素中 2 个零 | NVIDIA A100+ | 2x | 中 |
| Block Sparse | 按块为单位稀疏 | 通用 | 1.5-3x | 中 |
| N:M 稀疏 | 每 M 个中 N 个零 | 部分支持 | 1.3-2x | 中 |
| 通道稀疏 | 整行为零 | 通用 | 2-4x | 低 |

### 33.1.3 稀疏格式概述

```
常见稀疏存储格式：

1. CSR (Compressed Sparse Row)
   values:  [a, b, c, d, e, f]     # 非零值
   indices: [0, 2, 1, 3, 0, 2]     # 列索引
   indptr:  [0, 2, 4, 6]           # 行指针

   矩阵:
   ┌─────────────┐
   │ a  0  b  0  │  row 0: [0, 2)
   │ 0  c  0  d  │  row 1: [2, 4)
   │ e  0  f  0  │  row 2: [4, 6)
   └─────────────┘

2. CSC (Compressed Sparse Column)
   类似 CSR，但按列压缩

3. BSR (Block Sparse Row)
   按块存储，适合块稀疏模式
   block_values: [B0, B1, B2, ...]  # 块矩阵
   block_indices: [0, 2, 1, ...]    # 块列索引
   block_indptr: [0, 2, 3, ...]     # 块行指针

4. ELLPACK (ELL)
   固定每行非零元素数
   values:  [[a, b], [c, d], [e, f]]
   indices: [[0, 2], [1, 3], [0, 2]]
```

---

## 33.2 2:4 结构化稀疏

### 33.2.1 2:4 稀疏原理

2:4 结构化稀疏是 NVIDIA 在 Ampere 架构（A100）中引入的硬件加速稀疏模式。其规则是：**在每 4 个连续元素中，恰好有 2 个为零**。

```
2:4 结构化稀疏示例：

原始矩阵 (4×4):
┌──────────────────┐
│ 1.0  0.0  3.0  0.0 │  → 保留 [1.0, 3.0], 索引 [0, 2]
│ 0.0  2.0  0.0  4.0 │  → 保留 [2.0, 4.0], 索引 [1, 3]
│ 5.0  0.0  0.0  6.0 │  → 保留 [5.0, 6.0], 索引 [0, 3]
│ 0.0  7.0  8.0  0.0 │  → 保留 [7.0, 8.0], 索引 [1, 2]
└──────────────────┘

压缩存储:
  values:  [[1.0, 3.0], [2.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
  indices: [[0, 2],    [1, 3],    [0, 3],    [1, 2]]

压缩比: 50% (每 4 个元素压缩为 2 个值 + 2 个索引)
```

### 33.2.2 硬件加速原理

NVIDIA A100/H100 的 Tensor Core 支持 2:4 稀疏加速：

```
2:4 稀疏 Tensor Core 工作流程：

稠密矩阵 A (M×K)        稀疏矩阵 B (K×N)
┌──────────────┐        ┌──────────────┐
│              │        │  压缩值       │
│   M × K      │   ×    │  索引矩阵     │  → C (M×N)
│              │        │  (2:4 格式)   │
└──────────────┘        └──────────────┘

硬件指令:
  ldmatrix    → 加载稠密 A 块到寄存器
  ldmatrix.sp → 加载稀疏 B 块 (压缩值 + 索引)
  mma.sp      → 稀疏矩阵乘法 (2x 吞吐量)

性能提升:
  稠密 mma.sync: 256 FP16 FLOPS/cycle (per SM)
  稀疏 mma.sp:   512 FP16 FLOPS/cycle (per SM)
  理论加速比: 2x
```

### 33.2.3 TileLang 实现 2:4 稀疏 GEMM

```python
import tilelang
from tilelang import T
import torch

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64

@T.prim_func
def sparse_24_gemm(
    A: T.Tensor([BLOCK_M, BLOCK_K], "float16"),        # 稠密矩阵
    B_values: T.Tensor([BLOCK_K // 2, BLOCK_N], "float16"),  # 压缩的稀疏值
    B_indices: T.Tensor([BLOCK_K // 2, BLOCK_N], "int16"),   # 稀疏索引
    C: T.Tensor([BLOCK_M, BLOCK_N], "float16"),
):
    """2:4 结构化稀疏 GEMM"""
    # 分配存储
    A_smem = T.alloc_shared([BLOCK_M, BLOCK_K], "float16")
    B_val_smem = T.alloc_shared([BLOCK_K // 2, BLOCK_N], "float16")
    B_idx_smem = T.alloc_shared([BLOCK_K // 2, BLOCK_N], "int16")

    A_frag = T.alloc_fragment([BLOCK_M, BLOCK_K], "float16")
    C_frag = T.alloc_fragment([BLOCK_M, BLOCK_N], "float32")

    T.clear(C_frag)

    for k in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
        # 加载稠密 A
        for i, j in T.Parallel(BLOCK_M, BLOCK_K):
            A_smem[i, j] = A[bx * BLOCK_M + i, k * BLOCK_K + j]

        # 加载稀疏 B (压缩格式)
        for i, j in T.Parallel(BLOCK_K // 2, BLOCK_N):
            B_val_smem[i, j] = B_values[k * BLOCK_K // 2 + i, by * BLOCK_N + j]
            B_idx_smem[i, j] = B_indices[k * BLOCK_K // 2 + i, by * BLOCK_N + j]

        # 加载到寄存器
        T.copy(A_smem, A_frag)

        # 稀疏 GEMM (硬件加速)
        T.sparse_gemm(A_frag, B_val_smem, B_idx_smem, C_frag)

    # 写回
    for i, j in T.Parallel(BLOCK_M, BLOCK_N):
        C[bx * BLOCK_M + i, by * BLOCK_N + j] = C_frag[i, j]
```

<div data-component="StructuredSparsityDemo"></div>

### 33.2.4 2:4 稀疏的权重重排

在实际应用中，需要将预训练的稠密权重转换为 2:4 稀疏格式：

```python
def convert_to_24_sparse(weight):
    """
    将稠密权重转换为 2:4 稀疏格式
    策略: 保留每 4 个元素中绝对值最大的 2 个
    """
    rows, cols = weight.shape
    assert cols % 4 == 0, "列数必须是 4 的倍数"

    # 重塑为 (rows, cols//4, 4)
    weight_reshaped = weight.reshape(rows, cols // 4, 4)

    # 找到每组中绝对值最大的 2 个
    abs_weight = torch.abs(weight_reshaped)
    _, top2_indices = torch.topk(abs_weight, 2, dim=-1)

    # 创建稀疏掩码
    mask = torch.zeros_like(weight_reshaped, dtype=torch.bool)
    mask.scatter_(-1, top2_indices, True)

    # 应用掩码
    sparse_weight = weight_reshaped * mask

    # 提取压缩值和索引
    values = sparse_weight[mask].reshape(rows, cols // 2, 2)
    indices = top2_indices.sort(dim=-1).values

    return sparse_weight.reshape(rows, cols), values, indices

# 使用示例
weight = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
sparse_weight, values, indices = convert_to_24_sparse(weight)

# 计算稀疏率
sparsity = (sparse_weight == 0).float().mean()
print(f"稀疏率: {sparsity:.1%}")  # 应该接近 50%
```

---

## 33.3 Block Sparse 实现

### 33.3.1 Block Sparse 简介

Block Sparse 是按块为单位进行稀疏化的模式，更适合通用硬件：

```
Block Sparse 示例（块大小 2×2）:

原始矩阵 (4×4):
┌───────────────────┐
│ 1  2 │ 0  0 │ 3  4 │
│ 5  6 │ 0  0 │ 7  8 │
├──────┼──────┼──────┤
│ 0  0 │ 9 10 │ 0  0 │
│ 0  0 │11 12 │ 0  0 │
└───────────────────┘

块结构 (2×2 块):
┌─────────────┐
│  B0  │ 0  │ B1 │
│  0   │ B2 │ 0  │
└─────────────┘

BSR 存储:
  block_values: [B0, B2, B1]  # 3 个非零块
  block_indices: [0, 1, 2]     # 块列索引
  block_indptr: [0, 2, 3]      # 块行指针
```

<div data-component="BlockSparseVisualizer"></div>

### 33.3.2 TileLang Block Sparse GEMM

```python
@T.prim_func
def block_sparse_gemm(
    A: T.Tensor([M, K], "float16"),
    B_blocks: T.Tensor([num_blocks, block_size, block_size], "float16"),
    block_row_ptr: T.Tensor([num_block_rows + 1], "int32"),
    block_col_idx: T.Tensor([num_blocks], "int32"),
    C: T.Tensor([M, N], "float16"),
):
    """Block Sparse GEMM"""
    BLOCK_M = 128
    BLOCK_N = block_size
    BLOCK_K = block_size

    # 每个线程块处理输出的一行块
    row_block = T.thread_binding("blockIdx.x")

    if row_block < M // BLOCK_M:
        C_frag = T.alloc_fragment([BLOCK_M, N], "float32")
        T.clear(C_frag)

        # 遍历该行的所有非零块
        start = block_row_ptr[row_block]
        end = block_row_ptr[row_block + 1]

        for nz_idx in T.serial(start, end):
            col_block = block_col_idx[nz_idx]
            B_block = B_blocks[nz_idx]  # (block_size, block_size)

            # 加载对应的 A 块
            A_smem = T.alloc_shared([BLOCK_M, BLOCK_K], "float16")
            for i, j in T.Parallel(BLOCK_M, BLOCK_K):
                A_smem[i, j] = A[row_block * BLOCK_M + i,
                                  col_block * BLOCK_K + j]

            # 加载 B 块到 Shared Memory
            B_smem = T.alloc_shared([BLOCK_K, BLOCK_N], "float16")
            for i, j in T.Parallel(BLOCK_K, BLOCK_N):
                B_smem[i, j] = B_block[i, j]

            # GEMM 累积
            frag_offset = col_block * BLOCK_N
            T.gemm(A_smem, B_smem,
                   C_frag[:, frag_offset:frag_offset + BLOCK_N])

        # 写回结果
        for i, j in T.Parallel(BLOCK_M, N):
            C[row_block * BLOCK_M + i, j] = C_frag[i, j]
```

### 33.3.3 Block Sparse 的稀疏模式生成

```python
def generate_block_sparse_pattern(num_rows, num_cols, block_size, sparsity):
    """
    生成随机的 Block Sparse 模式
    """
    num_block_rows = num_rows // block_size
    num_block_cols = num_cols // block_size

    # 创建块级别的稀疏掩码
    block_mask = torch.rand(num_block_rows, num_block_cols) > sparsity

    # 提取非零块的位置
    nz_positions = torch.where(block_mask)

    # 构建 BSR 格式
    row_ptr = [0]
    col_idx = []
    num_blocks = 0

    for r in range(num_block_rows):
        nz_cols = nz_positions[1][nz_positions[0] == r]
        col_idx.extend(nz_cols.tolist())
        num_blocks += len(nz_cols)
        row_ptr.append(num_blocks)

    return {
        "block_mask": block_mask,
        "row_ptr": torch.tensor(row_ptr, dtype=torch.int32),
        "col_idx": torch.tensor(col_idx, dtype=torch.int32),
        "num_blocks": num_blocks,
        "sparsity": 1 - num_blocks / (num_block_rows * num_block_cols),
    }
```

### 33.3.4 动态 Block Sparse

在某些场景下，稀疏模式是动态的（如 Top-K 注意力）：

```python
@T.prim_func
def dynamic_block_sparse_attention(
    Q: T.Tensor([batch, seq_len, n_heads, d], "float16"),
    K: T.Tensor([batch, seq_len, n_heads, d], "float16"),
    V: T.Tensor([batch, seq_len, n_heads, d], "float16"),
    BlockMask: T.Tensor([batch, n_heads, num_blocks_q, num_blocks_k], "bool"),
    Output: T.Tensor([batch, seq_len, n_heads, d], "float16"),
):
    """动态 Block Sparse Attention"""
    # 根据 BlockMask 决定计算哪些块
    for bq in T.grid(num_blocks_q):
        O_frag = T.alloc_fragment([BLOCK_M, d], "float32")
        T.clear(O_frag)

        for bk in T.serial(num_blocks_k):
            # 检查该块是否需要计算
            if BlockMask[batch_idx, head_idx, bq, bk]:
                # 加载 K, V 块
                K_smem = T.alloc_shared([BLOCK_N, d], "float16")
                V_smem = T.alloc_shared([BLOCK_N, d], "float16")

                # 计算注意力分数
                S_frag = T.alloc_fragment([BLOCK_M, BLOCK_N], "float32")
                T.gemm(Q_smem, K_smem, S_frag, transpose_B=True)

                # Softmax + P @ V
                # ...
```

---

## 33.4 Sparse FlashAttention

### 33.4.1 稀疏注意力模式

```
常见的稀疏注意力模式：

1. 全注意力 (Full Attention)
   ┌─────────────┐
   │ ■ ■ ■ ■ ■ ■ │  每个 token 关注所有其他 token
   │ ■ ■ ■ ■ ■ ■ │  计算复杂度: O(n²)
   │ ■ ■ ■ ■ ■ ■ │
   │ ■ ■ ■ ■ ■ ■ │
   │ ■ ■ ■ ■ ■ ■ │
   │ ■ ■ ■ ■ ■ ■ │
   └─────────────┘

2. 因果注意力 (Causal Attention)
   ┌─────────────┐
   │ ■ □ □ □ □ □ │  每个 token 只关注之前的 token
   │ ■ ■ □ □ □ □ │  计算复杂度: O(n²/2)
   │ ■ ■ ■ □ □ □ │
   │ ■ ■ ■ ■ □ □ │
   │ ■ ■ ■ ■ ■ □ │
   │ ■ ■ ■ ■ ■ ■ │
   └─────────────┘

3. 局部窗口注意力 (Sliding Window)
   ┌─────────────┐
   │ ■ ■ ■ □ □ □ │  每个 token 只关注窗口内的 token
   │ □ ■ ■ ■ □ □ │  计算复杂度: O(n × w)
   │ □ □ ■ ■ ■ □ │
   │ □ □ □ ■ ■ ■ │
   │ □ □ □ □ ■ ■ │
   │ □ □ □ □ □ ■ │
   └─────────────┘

4. 块稀疏注意力 (Block Sparse)
   ┌─────────────┐
   │ ■ ■ □ ■ □ □ │  按块选择性计算
   │ ■ ■ □ ■ □ □ │  计算复杂度: O(n × b)
   │ □ □ ■ □ ■ ■ │
   │ ■ ■ □ ■ □ □ │
   │ □ □ ■ □ ■ ■ │
   │ □ □ ■ □ ■ ■ │
   └─────────────┘
```

### 33.4.2 TileLang Sparse FlashAttention 实现

```python
@T.prim_func
def sparse_flash_attention(
    Q: T.Tensor([batch, seq_len, n_heads, d], "float16"),
    K: T.Tensor([batch, seq_len, n_heads, d], "float16"),
    V: T.Tensor([batch, seq_len, n_heads, d], "float16"),
    BlockMask: T.Tensor([num_q_blocks, num_kv_blocks], "bool"),
    Output: T.Tensor([batch, seq_len, n_heads, d], "float16"),
):
    """Block Sparse FlashAttention"""
    BLOCK_M = 128
    BLOCK_N = 64
    NUM_STAGES = 2

    for bx in T.grid(T.ceildiv(seq_len, BLOCK_M)):
        # 加载 Q 块
        Q_smem = T.alloc_shared([BLOCK_M, d], "float16")
        T.copy(Q[bx * BLOCK_M:(bx + 1) * BLOCK_M], Q_smem)

        # 初始化 Online Softmax
        m_prev = T.alloc_fragment([BLOCK_M], "float32")
        l_prev = T.alloc_fragment([BLOCK_M], "float32")
        O_frag = T.alloc_fragment([BLOCK_M, d], "float32")
        T.clear(m_prev)
        T.clear(l_prev)
        T.clear(O_frag)

        # 遍历 KV 块（只处理非零块）
        for by in T.serial(T.ceildiv(seq_len, BLOCK_N)):
            # 检查该块是否需要计算
            if BlockMask[bx, by]:
                # 加载 K, V 块
                K_smem = T.alloc_shared([BLOCK_N, d], "float16")
                V_smem = T.alloc_shared([BLOCK_N, d], "float16")
                T.copy(K[by * BLOCK_N:(by + 1) * BLOCK_N], K_smem)
                T.copy(V[by * BLOCK_N:(by + 1) * BLOCK_N], V_smem)

                # 计算 Q @ K^T
                S_frag = T.alloc_fragment([BLOCK_M, BLOCK_N], "float32")
                T.gemm(Q_smem, K_smem, S_frag, transpose_B=True)
                S_frag /= T.sqrt(d)

                # Online Softmax 更新
                m_new = T.alloc_fragment([BLOCK_M], "float32")
                T.reduce_max(S_frag, m_new, dim=1)
                m_max = T.maximum(m_prev, m_new)

                # 计算 exp(S - m_max)
                P_frag = T.alloc_fragment([BLOCK_M, BLOCK_N], "float16")
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    P_frag[i, j] = T.exp(S_frag[i, j] - m_max[i])

                # 更新分母
                l_new = T.alloc_fragment([BLOCK_M], "float32")
                T.reduce_sum(P_frag, l_new, dim=1)
                correction = T.exp(m_prev - m_max)
                l_prev = l_prev * correction + l_new

                # 更新输出
                for i in T.Parallel(BLOCK_M):
                    O_frag[i, :] *= correction[i]
                T.gemm(P_frag, V_smem, O_frag)

                m_prev = m_max

        # 写回输出
        for i in T.Parallel(BLOCK_M):
            O_frag[i, :] /= l_prev[i]
        T.copy(O_frag, Output[bx * BLOCK_M:(bx + 1) * BLOCK_M])
```

### 33.4.3 稀疏注意力的性能分析

<div data-component="SparseVsDensePerformance"></div>

| 注意力模式 | 序列长度 2K | 序列长度 8K | 序列长度 32K |
|-----------|-----------|-----------|------------|
| Full Attention | 18,000 tok/s | 16,500 tok/s | 14,000 tok/s |
| Causal (50%) | 25,000 tok/s | 23,000 tok/s | 20,000 tok/s |
| Sliding Window (w=256) | 45,000 tok/s | 42,000 tok/s | 38,000 tok/s |
| Block Sparse (50%) | 30,000 tok/s | 28,000 tok/s | 25,000 tok/s |
| Block Sparse (25%) | 50,000 tok/s | 47,000 tok/s | 42,000 tok/s |

---

## 33.4b 稀疏格式详解

### 33.4b.1 CSR (Compressed Sparse Row) 格式详解

CSR 是最常用的稀疏存储格式，特别适合行优先访问：

```python
class CSRMatrix:
    """CSR 格式稀疏矩阵"""

    def __init__(self, dense_matrix=None):
        if dense_matrix is not None:
            self.from_dense(dense_matrix)

    def from_dense(self, dense):
        """从稠密矩阵转换为 CSR 格式"""
        rows, cols = dense.shape
        self.values = []
        self.col_indices = []
        self.row_ptr = [0]

        for i in range(rows):
            for j in range(cols):
                if dense[i, j] != 0:
                    self.values.append(dense[i, j])
                    self.col_indices.append(j)
            self.row_ptr.append(len(self.values))

        self.shape = (rows, cols)
        self.nnz = len(self.values)

    def to_dense(self):
        """转换回稠密矩阵"""
        import numpy as np
        dense = np.zeros(self.shape)
        for i in range(self.shape[0]):
            for idx in range(self.row_ptr[i], self.row_ptr[i + 1]):
                j = self.col_indices[idx]
                dense[i, j] = self.values[idx]
        return dense

    def get_row(self, i):
        """获取第 i 行的非零元素"""
        start = self.row_ptr[i]
        end = self.row_ptr[i + 1]
        return {
            "values": self.values[start:end],
            "columns": self.col_indices[start:end],
        }

# 示例
dense = np.array([
    [1, 0, 0, 2],
    [0, 3, 0, 0],
    [4, 0, 5, 0],
])
csr = CSRMatrix(dense)
print(f"values: {csr.values}")       # [1, 2, 3, 4, 5]
print(f"col_indices: {csr.col_indices}")  # [0, 3, 1, 0, 2]
print(f"row_ptr: {csr.row_ptr}")     # [0, 2, 3, 5]
print(f"nnz: {csr.nnz}")            # 5
```

### 33.4b.2 CSC (Compressed Sparse Column) 格式详解

CSC 按列压缩，适合列优先访问和矩阵转置：

```python
class CSCMatrix:
    """CSC 格式稀疏矩阵"""

    def __init__(self, dense_matrix=None):
        if dense_matrix is not None:
            self.from_dense(dense_matrix)

    def from_dense(self, dense):
        """从稠密矩阵转换为 CSC 格式"""
        rows, cols = dense.shape
        self.values = []
        self.row_indices = []
        self.col_ptr = [0]

        for j in range(cols):
            for i in range(rows):
                if dense[i, j] != 0:
                    self.values.append(dense[i, j])
                    self.row_indices.append(i)
            self.col_ptr.append(len(self.values))

        self.shape = (rows, cols)
        self.nnz = len(self.values)

    def transpose(self):
        """CSC 转置 = CSR"""
        csr = CSRMatrix()
        csr.values = self.values
        csr.col_indices = self.row_indices
        csr.row_ptr = self.col_ptr
        csr.shape = (self.shape[1], self.shape[0])
        csr.nnz = self.nnz
        return csr

# CSR 和 CSC 的对偶关系
# CSR 的转置是 CSC，CSC 的转置是 CSR
# 这使得矩阵转置操作非常高效
```

### 33.4b.3 BSR (Block Sparse Row) 格式详解

BSR 按块存储，特别适合块稀疏模式：

```python
class BSRMatrix:
    """BSR 格式稀疏矩阵"""

    def __init__(self, dense_matrix=None, block_size=4):
        self.block_size = block_size
        if dense_matrix is not None:
            self.from_dense(dense_matrix)

    def from_dense(self, dense):
        """从稠密矩阵转换为 BSR 格式"""
        rows, cols = dense.shape
        bs = self.block_size
        assert rows % bs == 0 and cols % bs == 0

        block_rows = rows // bs
        block_cols = cols // bs

        self.block_values = []  # 非零块
        self.block_col_indices = []  # 块列索引
        self.block_row_ptr = [0]  # 块行指针

        for br in range(block_rows):
            for bc in range(block_cols):
                block = dense[br*bs:(br+1)*bs, bc*bs:(bc+1)*bs]
                if np.any(block != 0):
                    self.block_values.append(block)
                    self.block_col_indices.append(bc)
            self.block_row_ptr.append(len(self.block_values))

        self.shape = (rows, cols)
        self.num_blocks = len(self.block_values)

    def get_block(self, block_row, block_col_idx):
        """获取指定块"""
        start = self.block_row_ptr[block_row]
        end = self.block_row_ptr[block_row + 1]
        for i in range(start, end):
            if self.block_col_indices[i] == block_col_idx:
                return self.block_values[i]
        return np.zeros((self.block_size, self.block_size))

# BSR vs CSR 性能对比
# BSR 优势：
# 1. 块内元素连续存储 → 缓存友好
# 2. 索引开销更小（每块只需一个索引）
# 3. 可以使用 Tensor Core 进行块内矩阵乘法
# BSR 劣势：
# 1. 需要块对齐
# 2. 块内有零元素时浪费存储
```

### 33.4b.4 ELLPACK (ELL) 格式详解

ELL 格式适合每行非零元素数量相近的矩阵：

```python
class ELLMatrix:
    """ELL 格式稀疏矩阵"""

    def __init__(self, dense_matrix=None):
        if dense_matrix is not None:
            self.from_dense(dense_matrix)

    def from_dense(self, dense):
        """从稠密矩阵转换为 ELL 格式"""
        rows, cols = dense.shape

        # 计算每行的最大非零元素数
        max_nnz_per_row = max(np.count_nonzero(dense[i]) for i in range(rows))

        # 固定宽度存储
        self.values = np.zeros((rows, max_nnz_per_row))
        self.col_indices = np.zeros((rows, max_nnz_per_row), dtype=int)

        for i in range(rows):
            nz_count = 0
            for j in range(cols):
                if dense[i, j] != 0:
                    self.values[i, nz_count] = dense[i, j]
                    self.col_indices[i, nz_count] = j
                    nz_count += 1

        self.shape = (rows, cols)
        self.max_nnz_per_row = max_nnz_per_row

# ELL 格式的优缺点
# 优点：
# 1. 无分支 → GPU 友好
# 2. 内存访问规则 → 合并访问
# 3. 适合 SIMD 执行
# 缺点：
# 1. 每行非零数差异大时浪费严重
# 2. 存储开销 = rows × max_nnz_per_row
```

### 33.4b.5 格式选择决策矩阵

| 矩阵特征 | 推荐格式 | 原因 |
|----------|---------|------|
| 随机稀疏 | CSR | 通用，存储效率高 |
| 列优先访问 | CSC | 列访问高效 |
| 块稀疏 | BSR | 块内连续，缓存友好 |
| 每行非零数相近 | ELL | 无分支，GPU 友好 |
| 2:4 结构化 | 专用格式 | 硬件加速 |
| 超稀疏（< 1%） | COO | 存储开销最小 |
| 动态稀疏 | CSR | 易于更新 |

---

## 33.4c 结构化稀疏硬件支持

### 33.4c.1 NVIDIA Ampere 2:4 稀疏支持

Ampere 架构（A100）引入了硬件级 2:4 稀疏加速：

```
Ampere 2:4 稀疏 Tensor Core 架构：

稠密模式:
  MMA 指令: mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
  吞吐量: 256 FP16 ops/cycle/SM

稀疏模式:
  MMA 指令: mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
  吞吐量: 512 FP16 ops/cycle/SM (2x)

硬件实现:
  1. 稀疏矩阵 B 使用压缩格式存储
  2. 索引矩阵记录非零元素位置
  3. Tensor Core 内部有专门的解压逻辑
  4. 解压和计算在硬件级别重叠
```

### 33.4c.2 NVIDIA Hopper 稀疏增强

Hopper 架构（H100）进一步增强了稀疏支持：

```
Hopper 稀疏增强：

1. 更大的稀疏 Tile
   - Ampere: 16×8×16 稀疏 Tile
   - Hopper: 16×8×32 稀疏 Tile
   - 更高的计算密度

2. 异步稀疏拷贝
   - TMA 支持稀疏数据的异步传输
   - 稀疏索引和值可以同时传输

3. 结构化稀疏灵活性
   - 支持更多 N:M 模式（如 2:4, 4:8）
   - 更灵活的稀疏模式约束

4. 性能提升
   - 相比 Ampere 稀疏：~1.5x
   - 相比 Ampere 稠密：~3x
```

### 33.4c.3 稀疏 Tensor Core 指令映射

```python
# TileLang 稀疏 GEMM 到硬件指令的映射

"""
TileLang 代码:
    T.sparse_gemm(A_frag, B_val_smem, B_idx_smem, C_frag)

编译器生成的 PTX 代码:
    // 加载稠密 A 矩阵
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {a0, a1, a2, a3}, [A_smem];

    // 加载稀疏 B 矩阵（值 + 索引）
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {b0, b1, b2, b3}, [B_val_smem];
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {i0, i1, i2, i3}, [B_idx_smem];

    // 稀疏矩阵乘法
    mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
        {c0, c1, c2, c3}, {a0, a1, a2, a3}, {b0, b1}, {i0, i1}, {c0, c1, c2, c3};
"""
```

---

## 33.4d Sparse FlashAttention 详细算法

### 33.4d.1 在线 Softmax 与稀疏性的交互

```
在线 Softmax 在稀疏场景下的挑战：

1. 未计算块的 Softmax 贡献
   - 稀疏模式跳过某些 KV 块
   - 这些块的 Softmax 值为 0
   - 但分母 l 需要正确计算

2. 数值稳定性
   - 跳过块可能导致 m 值不稳定
   - 需要确保 max 值的正确性

3. 实现策略
   - 策略 A: 先计算所有块的 max，再计算 Softmax
   - 策略 B: 使用在线更新，跳过块的贡献为 0
   - 策略 C: 使用稀疏感知的在线 Softmax
```

### 33.4d.2 稀疏感知在线 Softmax 实现

```python
@T.prim_func
def sparse_aware_online_softmax(
    Q: T.Tensor([batch, seq_len, n_heads, d], "float16"),
    K: T.Tensor([batch, seq_len, n_heads, d], "float16"),
    V: T.Tensor([batch, seq_len, n_heads, d], "float16"),
    BlockMask: T.Tensor([num_q_blocks, num_kv_blocks], "bool"),
    Output: T.Tensor([batch, seq_len, n_heads, d], "float16"),
):
    """稀疏感知的在线 Softmax FlashAttention"""
    BLOCK_M = 128
    BLOCK_N = 64

    for bx in T.grid(T.ceildiv(seq_len, BLOCK_M)):
        Q_smem = T.alloc_shared([BLOCK_M, d], "float16")
        T.copy(Q[bx * BLOCK_M:(bx + 1) * BLOCK_M], Q_smem)

        # 在线 Softmax 状态
        m_prev = T.alloc_fragment([BLOCK_M], "float32")
        l_prev = T.alloc_fragment([BLOCK_M], "float32")
        O_frag = T.alloc_fragment([BLOCK_M, d], "float32")

        # 初始化为负无穷和零
        for i in T.Parallel(BLOCK_M):
            m_prev[i] = -T.inf("float32")
            l_prev[i] = 0.0
        T.clear(O_frag)

        for by in T.serial(T.ceildiv(seq_len, BLOCK_N)):
            # 稀疏检查：只处理非零块
            if BlockMask[bx, by]:
                K_smem = T.alloc_shared([BLOCK_N, d], "float16")
                V_smem = T.alloc_shared([BLOCK_N, d], "float16")
                T.copy(K[by * BLOCK_N:(by + 1) * BLOCK_N], K_smem)
                T.copy(V[by * BLOCK_N:(by + 1) * BLOCK_N], V_smem)

                # 计算 Q @ K^T
                S_frag = T.alloc_fragment([BLOCK_M, BLOCK_N], "float32")
                T.gemm(Q_smem, K_smem, S_frag, transpose_B=True)
                S_frag /= T.sqrt(d)

                # 在线 Softmax 更新
                m_new = T.alloc_fragment([BLOCK_M], "float32")
                T.reduce_max(S_frag, m_new, dim=1)
                m_max = T.maximum(m_prev, m_new)

                # 计算 exp(S - m_max)
                P_frag = T.alloc_fragment([BLOCK_M, BLOCK_N], "float16")
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    P_frag[i, j] = T.exp(S_frag[i, j] - m_max[i])

                # 更新分母
                l_new = T.alloc_fragment([BLOCK_M], "float32")
                T.reduce_sum(P_frag, l_new, dim=1)
                correction = T.exp(m_prev - m_max)
                l_prev = l_prev * correction + l_new

                # 更新输出
                for i in T.Parallel(BLOCK_M):
                    O_frag[i, :] *= correction[i]
                T.gemm(P_frag, V_smem, O_frag)

                m_prev = m_max
            # 如果 BlockMask[bx, by] 为 False，跳过此块
            # m_prev 和 l_prev 保持不变

        # 写回输出
        for i in T.Parallel(BLOCK_M):
            if l_prev[i] > 0:
                O_frag[i, :] /= l_prev[i]
        T.copy(O_frag, Output[bx * BLOCK_M:(bx + 1) * BLOCK_M])
```

### 33.4d.3 稀疏 FlashAttention 的性能分析

```
稀疏 FlashAttention 性能分析：

理论计算量：
  全注意力: O(n² × d)
  稀疏注意力 (sparsity=s): O(n² × d × (1-s))

实际性能影响因素：
  1. 稀疏模式的规则性
     - 规则稀疏（如因果掩码）→ 高效
     - 随机稀疏 → 额外开销

  2. 块大小选择
     - 小块 → 更细粒度的稀疏 → 更好的加速
     - 大块 → 更好的计算效率 → 但稀疏性降低

  3. 内存访问模式
     - 稀疏访问可能导致缓存未命中
     - 需要预取优化

性能数据 (H100, seq_len=8192):
  全注意力: 16,500 tok/s
  因果掩码 (50%): 23,000 tok/s (+39%)
  块稀疏 (50%): 28,000 tok/s (+70%)
  块稀疏 (25%): 42,000 tok/s (+155%)
```

### 33.4d.4 稀疏掩码的生成与优化

```python
def generate_optimal_block_mask(seq_len, block_size, sparsity_target):
    """
    生成优化的块稀疏掩码

    优化目标：
    1. 满足稀疏率目标
    2. 保持因果性（如果需要）
    3. 最大化连续块的数量（减少分支）
    """
    num_blocks = seq_len // block_size
    mask = torch.ones(num_blocks, num_blocks, dtype=torch.bool)

    # 因果掩码：下三角
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            mask[i, j] = False

    # 随机稀疏化
    current_sparsity = 1 - mask.float().mean().item()
    if current_sparsity < sparsity_target:
        # 需要更多稀疏
        additional_sparsity = sparsity_target - current_sparsity
        # 随机选择一些上三角块保留
        upper_tri = torch.triu(torch.ones(num_blocks, num_blocks), diagonal=1)
        candidates = torch.where(upper_tri & ~mask)
        num_to_keep = int(len(candidates[0]) * (1 - additional_sparsity))
        keep_indices = torch.randperm(len(candidates[0]))[:num_to_keep]
        mask[candidates[0][keep_indices], candidates[1][keep_indices]] = True

    return mask
```

---

## 33.5 Sparse GEMM

### 33.5.1 非结构化稀疏 GEMM

```python
@T.prim_func
def unstructured_sparse_gemm(
    A: T.Tensor([M, K], "float16"),
    B_values: T.Tensor([nnz], "float16"),      # 非零值
    B_row_idx: T.Tensor([nnz], "int32"),        # 行索引
    B_col_ptr: T.Tensor([N + 1], "int32"),      # 列指针
    C: T.Tensor([M, N], "float16"),
):
    """非结构化稀疏 GEMM (CSC 格式)"""
    # 每个线程块处理一列
    col = T.thread_binding("blockIdx.x")

    if col < N:
        # 该列的非零元素范围
        start = B_col_ptr[col]
        end = B_col_ptr[col + 1]
        nnz_col = end - start

        # 累加器
        C_col = T.alloc_fragment([M], "float32")
        T.clear(C_col)

        # 遍历该列的非零元素
        for nz in T.serial(start, end):
            row = B_row_idx[nz]
            val = B_values[nz]

            # A[:, row] * val
            for i in T.Parallel(M):
                C_col[i] += A[i, row] * val

        # 写回
        for i in T.Parallel(M):
            C[i, col] = C_col[i]
```

### 33.5.2 混合稀疏 GEMM

在实际场景中，可能需要同时支持稠密和稀疏矩阵的混合运算：

```python
@T.prim_func
def hybrid_sparse_gemm(
    A: T.Tensor([M, K], "float16"),
    B_dense: T.Tensor([K, N_dense], "float16"),
    B_sparse: T.Tensor([K // 2, N_sparse], "float16"),
    B_indices: T.Tensor([K // 2, N_sparse], "int16"),
    C: T.Tensor([M, N_dense + N_sparse], "float16"),
):
    """混合稀疏 GEMM: 部分稠密 + 部分稀疏"""
    # 稠密部分
    C_dense = T.alloc_fragment([BLOCK_M, BLOCK_N], "float32")
    T.clear(C_dense)
    # ... 标准稠密 GEMM

    # 稀疏部分
    C_sparse = T.alloc_fragment([BLOCK_M, BLOCK_N], "float32")
    T.clear(C_sparse)
    # ... 2:4 稀疏 GEMM

    # 合并输出
    for i, j in T.Parallel(BLOCK_M, N_dense + N_sparse):
        if j < N_dense:
            C[i, j] = C_dense[i, j]
        else:
            C[i, j] = C_sparse[i, j - N_dense]
```

---

## 33.6 稀疏格式选择策略

### 33.6.1 格式选择决策树

```
稀疏格式选择决策树：

1. 稀疏模式是否规则？
   ├── 是 → 2:4 结构化稀疏
   │       优势: 硬件加速，2x 吞吐
   │       限制: 需要 NVIDIA A100+
   │
   └── 否 → 是否有块结构？
       ├── 是 → BSR (Block Sparse Row)
       │       优势: 高效块访问，缓存友好
       │       限制: 需要块对齐
       │
       └── 否 → 行/列是否有规律？
           ├── 按行有规律 → CSR
           │       优势: 行访问高效
           │
           └── 按列有规律 → CSC
                   优势: 列访问高效
```

### 33.6.2 格式性能对比

| 格式 | 存储开销 | GEMM 性能 | 适用场景 |
|------|---------|----------|---------|
| CSR | O(nnz + rows) | 中等 | 通用稀疏 |
| CSC | O(nnz + cols) | 中等 | 列优先访问 |
| BSR | O(nnz + block_rows) | 高 | 块稀疏 |
| 2:4 | O(nnz/2 + nnz/2) | 最高 | 结构化稀疏 |
| ELLPACK | O(rows × max_nnz) | 高 | 均匀稀疏 |

### 33.6.3 格式转换工具

```python
def csr_to_bsr(csr_matrix, block_size):
    """CSR 格式转 BSR 格式"""
    rows, cols = csr_matrix.shape
    br = rows // block_size
    bc = cols // block_size

    block_values = []
    block_indices = []
    block_indptr = [0]

    for i in range(br):
        row_start = i * block_size
        row_end = (i + 1) * block_size

        # 找到该行块范围内的所有非零块
        nz_blocks = set()
        for r in range(row_start, row_end):
            for idx in range(csr_matrix.indptr[r], csr_matrix.indptr[r + 1]):
                c = csr_matrix.indices[idx]
                nz_blocks.add(c // block_size)

        # 提取每个非零块
        for bc_idx in sorted(nz_blocks):
            block = np.zeros((block_size, block_size), dtype=csr_matrix.dtype)
            for r in range(row_start, row_end):
                for idx in range(csr_matrix.indptr[r], csr_matrix.indptr[r + 1]):
                    c = csr_matrix.indices[idx]
                    if c // block_size == bc_idx:
                        block[r - row_start, c % block_size] = csr_matrix.data[idx]

            block_values.append(block)
            block_indices.append(bc_idx)

        block_indptr.append(len(block_values))

    return {
        "block_values": np.array(block_values),
        "block_indices": np.array(block_indices, dtype=np.int32),
        "block_indptr": np.array(block_indptr, dtype=np.int32),
    }
```

---

## 33.7 稀疏训练与推理

### 33.7.1 稀疏训练流程

```
稀疏训练流程：

1. 预训练稠密模型
   └── 获得初始权重

2. 稀疏化
   ├── 非结构化剪枝
   │   ├── 幅值剪枝: 移除绝对值最小的权重
   │   └── 梯度剪枝: 移除梯度最小的权重
   │
   └── 结构化剪枝
       ├── 通道剪枝: 移除整个通道
       └── 块剪枝: 移除整个块

3. 微调
   └── 在稀疏约束下继续训练

4. 稀疏推理
   ├── 使用稀疏格式存储
   └── 使用稀疏算子计算
```

### 33.7.2 稀疏推理优化

```python
class SparseLinear(torch.nn.Module):
    """稀疏线性层"""

    def __init__(self, in_features, out_features, sparsity=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity

        # 初始化权重
        self.weight = torch.nn.Parameter(
            torch.randn(out_features, in_features)
        )

        # 稀疏掩码
        self.register_buffer('mask', None)

    def prune(self):
        """执行稀疏化"""
        # 2:4 稀疏化
        _, sparse_weight, indices = convert_to_24_sparse(self.weight.data)
        self.weight.data = sparse_weight
        self.sparse_values = sparse_weight
        self.sparse_indices = indices

    def forward(self, x):
        if self.training:
            # 训练时使用稠密计算
            return torch.nn.functional.linear(x, self.weight)
        else:
            # 推理时使用稀疏计算
            return sparse_linear_tilelang(x, self.sparse_values, self.sparse_indices)
```

---

## 33.8 性能分析与优化

### 33.8.1 稀疏加速比分析

```
稀疏加速比影响因素：

理论加速比 = 1 / (1 - sparsity)
  50% 稀疏 → 2x
  75% 稀疏 → 4x
  90% 稀疏 → 10x

实际加速比 = 理论加速比 × 效率因子
  效率因子:
    - 索引存储开销: 0.85-0.95
    - 非规则访问: 0.70-0.90
    - 硬件利用率: 0.80-0.95
    - 格式转换开销: 0.90-0.99

实际加速比 ≈ 理论加速比 × 0.6-0.8
```

### 33.8.2 稀疏性能优化策略

| 策略 | 描述 | 效果 |
|------|------|------|
| 块对齐 | 确保稀疏块与硬件对齐 | +10-20% |
| 预取优化 | 预取下一个非零块 | +5-10% |
| 压缩存储 | 使用更紧凑的索引格式 | +5-15% |
| 混合精度 | 稀疏部分使用更高精度 | 无性能损失 |
| 动态调度 | 根据稀疏度动态选择算法 | +5-10% |

### 33.8.3 稀疏 vs 稠密决策

```python
def should_use_sparse(weight_shape, sparsity, hardware_info):
    """判断是否应该使用稀疏计算"""
    M, N, K = weight_shape

    # 计算理论加速比
    theoretical_speedup = 1 / (1 - sparsity)

    # 估算实际加速比
    efficiency = estimate_efficiency(weight_shape, sparsity, hardware_info)
    actual_speedup = theoretical_speedup * efficiency

    # 考虑格式转换开销
    conversion_overhead = estimate_conversion_cost(weight_shape, sparsity)

    # 决策
    net_speedup = actual_speedup - conversion_overhead

    return net_speedup > 1.1, net_speedup
```

---

## 33.9 稀疏计算的高级话题

### 33.9.1 动态稀疏

在推理时，稀疏模式可能是动态变化的（如 Top-K 注意力）。TileLang 支持动态稀疏模式：

```python
@T.prim_func
def dynamic_sparse_attention(
    Q: T.Tensor([batch, seq, n_heads, d], "float16"),
    K: T.Tensor([batch, seq, n_heads, d], "float16"),
    V: T.Tensor([batch, seq, n_heads, d], "float16"),
    TopKIndices: T.Tensor([batch, n_heads, seq, top_k], "int32"),
    Output: T.Tensor([batch, seq, n_heads, d], "float16"),
):
    """动态稀疏注意力: 每个 Query 只关注 Top-K 个 Key"""
    for bx in T.grid(T.ceildiv(seq, BLOCK_M)):
        Q_smem = T.alloc_shared([BLOCK_M, d], "float16")
        T.copy(Q[bx * BLOCK_M:(bx + 1) * BLOCK_M], Q_smem)

        O_frag = T.alloc_fragment([BLOCK_M, d], "float32")
        T.clear(O_frag)

        # 每个 Query 只处理 Top-K 个 Key
        for k_idx in T.serial(top_k):
            # 读取 Top-K 索引
            kv_positions = T.alloc_fragment([BLOCK_M], "int32")
            for i in T.Parallel(BLOCK_M):
                kv_positions[i] = TopKIndices[batch_idx, head_idx,
                                              bx * BLOCK_M + i, k_idx]

            # 加载对应的 K, V
            K_tile = T.alloc_fragment([BLOCK_M, d], "float16")
            V_tile = T.alloc_fragment([BLOCK_M, d], "float16")
            for i, j in T.Parallel(BLOCK_M, d):
                K_tile[i, j] = K[batch_idx, kv_positions[i], head_idx, j]
                V_tile[i, j] = V[batch_idx, kv_positions[i], head_idx, j]

            # 计算注意力
            S_frag = T.alloc_fragment([BLOCK_M, BLOCK_M], "float32")
            T.gemm(Q_smem, K_tile, S_frag, transpose_B=True)
            S_frag /= T.sqrt(d)

            # Softmax + P × V
            P_frag = T.alloc_fragment([BLOCK_M, BLOCK_M], "float16")
            for i, j in T.Parallel(BLOCK_M, BLOCK_M):
                P_frag[i, j] = T.exp(S_frag[i, j])

            T.gemm(P_frag, V_tile, O_frag)
```

### 33.9.2 稀疏 MoE

在 MoE 模型中，每个 token 只激活少数专家，这本质上是一种稀疏计算：

```python
@T.prim_func
def sparse_moe_layer(
    Tokens: T.Tensor([batch, seq, d_model], "float16"),
    ExpertWeights: T.Tensor([num_experts, d_model, d_ff], "float16"),
    RouterLogits: T.Tensor([batch, seq, num_experts], "float32"),
    Output: T.Tensor([batch, seq, d_model], "float16"),
):
    """稀疏 MoE 层"""
    for token_idx in T.Pipelined(batch * seq):
        # 路由计算
        logits = T.alloc_fragment([num_experts], "float32")
        for e in T.Parallel(num_experts):
            logits[e] = RouterLogits[token_idx // seq, token_idx % seq, e]

        # Top-K 选择 (K=2)
        topk = T.alloc_fragment([2], "int32")
        gates = T.alloc_fragment([2], "float32")
        T.topk(logits, topk, gates, k=2)

        # 只计算选中的专家
        out_frag = T.alloc_fragment([d_model], "float32")
        T.clear(out_frag)

        for k in T.serial(2):
            expert_id = topk[k]
            gate = gates[k]

            # 加载 token
            x = T.alloc_fragment([d_model], "float16")
            for j in T.Parallel(d_model):
                x[j] = Tokens[token_idx // seq, token_idx % seq, j]

            # Expert GEMM
            h = T.alloc_fragment([d_ff], "float32")
            T.gemm(x, ExpertWeights[expert_id], h)

            # SiLU 激活
            for j in T.Parallel(d_ff):
                h[j] = h[j] * (1.0 / (1.0 + T.exp(-h[j])))

            # 累积
            for j in T.Parallel(d_model):
                out_frag[j] += gate * h[j]

        # 写回
        for j in T.Parallel(d_model):
            Output[token_idx // seq, token_idx % seq, j] = T.cast(out_frag[j], "float16")
```

### 33.9.3 稀疏卷积

稀疏卷积在 3D 点云处理和自动驾驶中广泛应用：

```python
@T.prim_func
def sparse_conv3d(
    Features: T.Tensor([num_points, in_channels], "float16"),
    Weights: T.Tensor([out_channels, kernel_size**3, in_channels], "float16"),
    Indices: T.Tensor([num_points, kernel_size**3], "int32"),
    Output: T.Tensor([num_points, out_channels], "float16"),
):
    """稀疏 3D 卷积"""
    for point_idx in T.Pipelined(num_points):
        # 加载特征
        feat = T.alloc_fragment([in_channels], "float16")
        for c in T.Parallel(in_channels):
            feat[c] = Features[point_idx, c]

        # 累积输出
        out = T.alloc_fragment([out_channels], "float32")
        T.clear(out)

        # 遍历卷积核
        for k in T.serial(kernel_size**3):
            neighbor_idx = Indices[point_idx, k]
            if neighbor_idx >= 0:  # 有效邻居
                # 加载邻居特征
                neighbor_feat = T.alloc_fragment([in_channels], "float16")
                for c in T.Parallel(in_channels):
                    neighbor_feat[c] = Features[neighbor_idx, c]

                # 累积卷积
                for oc in T.Parallel(out_channels):
                    for ic in T.serial(in_channels):
                        out[oc] += Weights[oc, k, ic] * neighbor_feat[ic]

        # 写回
        for c in T.Parallel(out_channels):
            Output[point_idx, c] = T.cast(out[c], "float16")
```

### 33.9.4 稀疏格式自动选择

```python
class SparseFormatSelector:
    """稀疏格式自动选择器"""

    def __init__(self):
        self.formats = ["csr", "csc", "bsr", "2:4", "ell"]

    def select(self, matrix_shape, sparsity_pattern, hardware):
        """根据矩阵特征选择最优稀疏格式"""
        rows, cols = matrix_shape

        # 检查是否满足 2:4 稀疏条件
        if self.can_use_24(sparsity_pattern, hardware):
            return "2:4"

        # 检查是否有块结构
        block_size = self.detect_block_structure(sparsity_pattern)
        if block_size > 0:
            return "bsr", block_size

        # 检查行/列稀疏度
        row_sparsity = self.compute_row_sparsity(sparsity_pattern)
        col_sparsity = self.compute_col_sparsity(sparsity_pattern)

        if row_sparsity < col_sparsity:
            return "csr"
        else:
            return "csc"

    def can_use_24(self, pattern, hardware):
        """检查是否可以使用 2:4 稀疏"""
        if hardware["arch"] < "ampere":
            return False
        # 检查是否每 4 个元素中恰好 2 个非零
        flat = pattern.reshape(-1, 4)
        nnz_per_group = flat.sum(dim=1)
        return (nnz_per_group == 2).all()

    def detect_block_structure(self, pattern, max_block=16):
        """检测块结构"""
        for bs in [4, 8, 16]:
            if pattern.shape[0] % bs == 0 and pattern.shape[1] % bs == 0:
                block_pattern = pattern.reshape(-1, bs, pattern.shape[1] // bs, bs)
                block_pattern = block_pattern.transpose(1, 2).reshape(-1, bs * bs)
                # 检查每个块是否全零或全非零
                block_sum = block_pattern.sum(dim=1)
                if ((block_sum == 0) | (block_sum == bs * bs)).all():
                    return bs
        return 0
```

### 33.9.5 稀疏计算的性能建模

```python
class SparsePerformanceModel:
    """稀疏计算性能模型"""

    def estimate_performance(self, matrix_shape, sparsity, format, hardware):
        """估算稀疏计算性能"""
        M, N, K = matrix_shape

        # 理论计算量
        dense_flops = 2 * M * N * K
        sparse_flops = dense_flops * (1 - sparsity)

        # 内存访问
        if format == "2:4":
            # 压缩存储
            values_bytes = sparse_flops / 2 * 2  # FP16
            indices_bytes = sparse_flops / 2 * 2  # INT16
            memory_bytes = values_bytes + indices_bytes
        elif format == "bsr":
            # 块稀疏存储
            num_blocks = M * N * (1 - sparsity) / (block_size ** 2)
            memory_bytes = num_blocks * block_size * block_size * 2
        else:
            # CSR/CSC
            memory_bytes = sparse_flops * 2 + (M + 1) * 4

        # 硬件限制
        compute_bound = sparse_flops / hardware["peak_tflops"] / 1e12
        memory_bound = memory_bytes / hardware["memory_bandwidth"] / 1e9

        # 实际性能
        latency = max(compute_bound, memory_bound)
        actual_tflops = sparse_flops / latency / 1e12

        return {
            "dense_flops": dense_flops,
            "sparse_flops": sparse_flops,
            "memory_bytes": memory_bytes,
            "compute_bound": compute_bound,
            "memory_bound": memory_bound,
            "latency": latency,
            "actual_tflops": actual_tflops,
            "speedup": dense_flops / sparse_flops * min(compute_bound, memory_bound) / latency,
        }
```

### 33.9.6 稀疏训练技术

```python
class SparseTraining:
    """稀疏训练框架"""

    def __init__(self, model, target_sparsity=0.5):
        self.model = model
        self.target_sparsity = target_sparsity

    def magnitude_prune(self, threshold=None):
        """幅值剪枝"""
        all_weights = []
        for name, param in self.model.named_parameters():
            if "weight" in name:
                all_weights.append(param.data.abs().flatten())

        all_weights = torch.cat(all_weights)

        if threshold is None:
            # 自动确定阈值
            threshold = torch.quantile(all_weights, self.target_sparsity)

        # 应用掩码
        masks = {}
        for name, param in self.model.named_parameters():
            if "weight" in name:
                mask = param.data.abs() > threshold
                masks[name] = mask
                param.data *= mask

        return masks

    def gradual_pruning(self, epoch, total_epochs):
        """渐进式剪枝"""
        # 稀疏度从 0 逐渐增加到目标值
        current_sparsity = self.target_sparsity * (epoch / total_epochs)
        return self.magnitude_prune(current_sparsity)

    def fine_tune(self, train_loader, epochs=5):
        """剪枝后微调"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        for epoch in range(epochs):
            for batch in train_loader:
                loss = self.model(batch)
                loss.backward()

                # 应用稀疏掩码梯度
                for name, param in self.model.named_parameters():
                    if name in self.masks:
                        param.grad *= self.masks[name]

                optimizer.step()
                optimizer.zero_grad()
```

`SparseTraining` 类封装了稀疏训练的完整工作流，核心包含三个阶段：幅值剪枝、渐进式剪枝和掩码感知微调。`magnitude_prune` 方法收集模型中所有命名为 `weight` 的参数，将其展平拼接后使用 `torch.quantile` 自动确定剪枝阈值。`torch.quantile` 的优势在于它不依赖参数的具体分布假设，直接根据目标稀疏率分位数来定位阈值，比固定阈值或标准差阈值更具鲁棒性。`gradual_pruning` 方法实现了“渐进式剪枝”——稀疏率从 0 线性增加到目标值，这类似于学习率预热（warmup）的思想，让模型有充分时间适应逐渐增加的稀疏约束。训练过程中的梯度掩码（`param.grad *= self.masks[name]`）是一个容易被忽略但至关重要的细节：如果不对已剪枝权重的梯度进行掩码，优化器更新可能会在稀疏位置引入非零值，从而破坏稀疏结构。在 PyTorch 中实现梯度掩码的另一种更高效方式是通过 `register_hook` 在反向传播时自动应用掩码，避免在每个训练步骤中重复遍历参数。

---

> **从高级算法到数值精度：** 前面九个小节系统地介绍了稀疏计算的核心技术——从 2:4 结构化稀疏、Block Sparse 到 FlashAttention 的稀疏化，从格式选择到动态稀疏和 MoE。这些技术回答了“如何实现稀疏计算”的问题。然而，稀疏计算在实际部署中还面临一个根本性挑战：稀疏化带来的精度损失如何量化和控制？这是决定稀疏方案能否满足业务精度要求的关键。接下来的 33.9b 节将聚焦于稀疏计算的数值精度问题，分析误差来源和控制策略。

## 33.9b 稀疏计算的数值精度

### 33.9b.1 稀疏对数值精度的影响

```
稀疏计算的数值精度分析：

1. 剪枝引入的误差
   - 权重剪枝会移除部分权重
   - 被移除的权重贡献了部分计算结果
   - 误差大小取决于被移除权重的绝对值

2. 累加顺序的影响
   - 稀疏计算的累加顺序与稠密不同
   - 浮点加法不满足结合律
   - 不同累加顺序可能导致不同结果

3. 量化误差
   - 稀疏存储可能需要额外的量化
   - 索引存储引入舍入误差
   - 压缩/解压缩过程可能引入误差

误差量化：
  稠密计算: C_dense = A @ B
  稀疏计算: C_sparse = A @ B_sparse
  误差: ||C_dense - C_sparse|| / ||C_dense||

  50% 稀疏率下，典型误差: 1e-3 ~ 1e-2
  75% 稀疏率下，典型误差: 1e-2 ~ 1e-1
```

### 33.9b.2 稀疏计算的误差控制

```python
class SparsePrecisionManager:
    """稀疏计算精度管理器"""

    def __init__(self, target_error=1e-3):
        self.target_error = target_error

    def prune_with_precision_control(self, weight, sparsity):
        """带精度控制的剪枝"""
        # 方法 1: 幅值剪枝 + 误差补偿
        mask = self._magnitude_prune(weight, sparsity)
        pruned_weight = weight * mask

        # 计算误差
        error = (weight - pruned_weight).abs().mean().item()
        if error > self.target_error:
            # 降低稀疏率
            actual_sparsity = sparsity * (self.target_error / error)
            mask = self._magnitude_prune(weight, actual_sparsity)
            pruned_weight = weight * mask

        return pruned_weight, mask

    def _magnitude_prune(self, weight, sparsity):
        """幅值剪枝"""
        threshold = torch.quantile(weight.abs().flatten(), sparsity)
        mask = weight.abs() > threshold
        return mask

    def verify_accuracy(self, dense_output, sparse_output):
        """验证精度"""
        max_error = (dense_output - sparse_output).abs().max().item()
        mean_error = (dense_output - sparse_output).abs().mean().item()
        relative_error = mean_error / dense_output.abs().mean().item()

        return {
            "max_error": max_error,
            "mean_error": mean_error,
            "relative_error": relative_error,
            "passed": relative_error < self.target_error,
        }
```

`SparsePrecisionManager` 的设计核心在于“精度优先的剪枝策略”。标准幅值剪枝（magnitude pruning）仅考虑权重的绝对值，而 `prune_with_precision_control` 方法在此基础上引入了误差反馈机制：当剪枝后的误差超过目标阈值 `target_error` 时，会自动降低实际稀疏率（`actual_sparsity = sparsity × (target_error / error)`）以换取更高的精度。这种自适应策略避免了“一刀切”剪枝带来的精度坍塌问题，特别适合对精度敏感的推理场景（如医疗影像、金融预测）。`verify_accuracy` 方法则从最大误差、平均误差和相对误差三个维度全面评估稀疏输出的质量。其中相对误差指标 `relative_error = mean_error / dense_mean` 是最常用的参考标准——它衡量了稀疏化引入的偏差相对于原始信号强度的比例。在实际部署中，建议将 `target_error` 设置为 1e-3 作为起点，然后根据下游任务对精度的容忍度进行微调。值得注意的是，该精度管理器并未处理激活稀疏带来的误差，在实际稀疏推理流水线中，激活量化误差和权重稀疏误差会产生叠加效应，需要联合建模和优化。

### 33.9b.3 混合精度稀疏计算

```python
@T.prim_func
def mixed_precision_sparse_gemm(
    A: T.Tensor([M, K], "float16"),
    B_values: T.Tensor([K // 2, N], "float16"),
    B_indices: T.Tensor([K // 2, N], "int16"),
    C: T.Tensor([M, N], "float32"),  # FP32 输出保证精度
):
    """混合精度稀疏 GEMM"""
    for bx, by in T.grid(T.ceildiv(M, BM), T.ceildiv(N, BN)):
        A_smem = T.alloc_shared([BM, BK], "float16")
        B_val_smem = T.alloc_shared([BK // 2, BN], "float16")
        C_frag = T.alloc_fragment([BM, BN], "float32")  # FP32 累加

        T.clear(C_frag)

        for k in T.Pipelined(T.ceildiv(K, BK)):
            # 加载 FP16 数据
            for i, j in T.Parallel(BM, BK):
                A_smem[i, j] = A[bx * BM + i, k * BK + j]
            for i, j in T.Parallel(BK // 2, BN):
                B_val_smem[i, j] = B_values[k * BK // 2 + i, by * BN + j]

            # 稀疏 GEMM: FP16 × FP16 → FP32 累加
            T.sparse_gemm(A_smem, B_val_smem, B_indices, C_frag)

        # 直接输出 FP32
        for i, j in T.Parallel(BM, BN):
            C[bx * BM + i, by * BN + j] = C_frag[i, j]
```

`mixed_precision_sparse_gemm` 展示了稀疏计算中混合精度的最佳实践：输入矩阵 A 和 B 使用 FP16 格式以减少内存带宽和存储开销，而累加器 `C_frag` 则使用 FP32 格式以保护累加过程中的数值精度。这种“FP16 乘法 + FP32 累加”的模式借鉴了 NVIDIA Tensor Core 的硬件设计理念——Tensor Core 在执行 `mma` 指令时，输入的 A 和 B 可以是 FP16，但内部累加始终以 FP32 进行。在稀疏场景下，由于索引压缩存储可能额外引入对齐误差，FP32 累加对保证最终输出精度尤为重要。代码中输出的 C 矩阵直接写为 FP32 而非转换回 FP16，这一设计选择反映了稀疏推理的一个常见权衡：如果下游算子支持 FP32 输入，保留高精度输出可以避免额外的精度损失；如果需要 FP16 输出，则应在写入前显式调用 `T.cast` 进行类型转换。`T.sparse_gemm` 的稀疏 GEMM 原语在底层会映射到 PTX `mma.sp` 指令，其内部的解压、索引匹配和乘加操作对开发者完全透明，这是 TileLang 抽象能力的重要体现。

---

> **从精度理论到工程落地：** 前面两个小节从理论角度分析了稀疏化引入的数值误差来源和控制策略。然而，稀疏计算的真正价值体现在实际工程的落地效果中——如何设计高效的格式转换流水线、如何优化内存布局以最大化硬件利用率、以及如何构建可靠的调试工具链，这些才是决定稀疏方案能否在生产环境中持续运行的关键。接下来的第 33.10 节将聚焦于稀疏计算的工程实践，涵盖从格式转换到性能剖析的完整工具链。

## 33.10 稀疏计算的工程实践

### 33.10.1 稀疏格式转换流水线

在实际工程中，稀疏格式转换是一个关键的预处理步骤：

```python
class SparseFormatPipeline:
    """稀疏格式转换流水线"""

    def __init__(self, source_format, target_format):
        self.source = source_format
        self.target = target_format
        self.converters = {
            ("dense", "csr"): self._dense_to_csr,
            ("dense", "csc"): self._dense_to_csc,
            ("dense", "bsr"): self._dense_to_bsr,
            ("dense", "2:4"): self._dense_to_24,
            ("csr", "bsr"): self._csr_to_bsr,
            ("bsr", "csr"): self._bsr_to_csr,
            ("csr", "csc"): self._csr_to_csc,
        }

    def convert(self, data):
        """执行格式转换"""
        key = (self.source, self.target)
        if key not in self.converters:
            raise ValueError(f"Unsupported conversion: {key}")
        return self.converters[key](data)

    def _dense_to_24(self, dense_matrix):
        """稠密到 2:4 稀疏转换"""
        rows, cols = dense_matrix.shape
        assert cols % 4 == 0

        # 重塑为 (rows, cols//4, 4)
        reshaped = dense_matrix.reshape(rows, cols // 4, 4)

        # 保留每 4 个元素中绝对值最大的 2 个
        abs_values = torch.abs(reshaped)
        _, top2_indices = torch.topk(abs_values, 2, dim=-1)

        # 创建稀疏矩阵
        mask = torch.zeros_like(reshaped, dtype=torch.bool)
        mask.scatter_(-1, top2_indices, True)
        sparse = reshaped * mask

        # 提取压缩格式
        values = sparse[mask].reshape(rows, cols // 2, 2)
        indices = top2_indices.sort(dim=-1).values

        return {
            "values": values,
            "indices": indices,
            "sparsity": 0.5,
        }

    def _dense_to_bsr(self, dense_matrix, block_size=4):
        """稠密到 BSR 转换"""
        rows, cols = dense_matrix.shape
        bs = block_size
        block_rows = rows // bs
        block_cols = cols // bs

        block_values = []
        block_col_indices = []
        block_row_ptr = [0]

        for br in range(block_rows):
            for bc in range(block_cols):
                block = dense_matrix[br*bs:(br+1)*bs, bc*bs:(bc+1)*bs]
                if torch.any(block != 0):
                    block_values.append(block)
                    block_col_indices.append(bc)
            block_row_ptr.append(len(block_values))

        return {
            "block_values": torch.stack(block_values) if block_values else torch.tensor([]),
            "block_col_indices": torch.tensor(block_col_indices, dtype=torch.int32),
            "block_row_ptr": torch.tensor(block_row_ptr, dtype=torch.int32),
            "num_blocks": len(block_values),
            "block_size": block_size,
        }
```

`SparseFormatPipeline` 类以字典派发（dictionary dispatch）的模式实现了一个可扩展的稀疏格式转换流水线。该设计将每个格式转换路径映射为 `(source, target)` 键值对，使得新增转换路径只需在 `converters` 字典中注册新方法即可，符合开闭原则。其中 `_dense_to_24` 方法的实现特别值得注意：它先通过 `reshape` 将矩阵分组为每 4 个元素一组，然后使用 `torch.topk` 在每组中选出绝对值最大的 2 个元素——这一步骤实现了 2:4 剪枝的核心语义，即“在局部窗口内保留最重要的权重”。最终通过 `mask.scatter_` 将选中的元素位置标记为 True，并提取压缩后的值和索引。`_dense_to_bsr` 方法则以嵌套循环遍历每个 `block_size × block_size` 的子块，使用 `torch.any(block != 0)` 作为块级稀疏判定条件。这种按块遍历的方式在块数较多时可能成为性能瓶颈，实际工程中可以考虑使用 `torch.nn.functional.unfold` 或直接利用矩阵操作进行向量化块检测来加速。整个流水线的输出统一为字典格式，包含数据和元信息（如 `num_blocks`、`sparsity`），这种设计便于下游稀疏算子直接消费，减少了中间表示的转换成本。

### 33.10.2 稀疏计算的内存优化

```python
def optimize_sparse_memory_layout(sparse_data, format_type):
    """
    优化稀疏数据的内存布局

    目标：
    1. 减少索引存储开销
    2. 提高缓存命中率
    3. 对齐内存访问
    """
    if format_type == "2:4":
        # 2:4 稀疏的内存布局优化
        values = sparse_data["values"]
        indices = sparse_data["indices"]

        # 将值和索引交错存储，提高缓存局部性
        interleaved = torch.stack([values, indices.float()], dim=-1)
        interleaved = interleaved.reshape(-1, 2)

        # 确保 16 字节对齐
        if interleaved.numel() * 2 % 16 != 0:
            padding_size = 16 - (interleaved.numel() * 2 % 16) // 2
            interleaved = torch.cat([interleaved, torch.zeros(padding_size, 2)])

        return interleaved

    elif format_type == "bsr":
        # BSR 的内存布局优化
        block_values = sparse_data["block_values"]
        block_size = sparse_data["block_size"]

        # 确保每个块的内存对齐
        aligned_size = ((block_size * block_size * 2 + 15) // 16) * 16
        padded_blocks = []
        for block in block_values:
            flat = block.flatten()
            if flat.numel() * 2 < aligned_size:
                flat = torch.cat([flat, torch.zeros(aligned_size // 2 - flat.numel())])
            padded_blocks.append(flat)

        return torch.stack(padded_blocks)

    elif format_type == "csr":
        # CSR 的内存布局优化
        values = sparse_data["values"]
        col_indices = sparse_data["col_indices"]
        row_ptr = sparse_data["row_ptr"]

        # 交错存储 values 和 col_indices
        # 有助于向量化加载
        nnz = len(values)
        interleaved = torch.zeros(nnz * 2)
        interleaved[0::2] = values
        interleaved[1::2] = col_indices.float()

        return {
            "interleaved": interleaved,
            "row_ptr": row_ptr,
            "nnz": nnz,
        }
```

`optimize_sparse_memory_layout` 函数针对三种主流稀疏格式分别设计了内存布局优化策略，核心目标是提升 GPU 内存访问的合并性（coalescing）和缓存局部性。对于 2:4 格式，采用值-索引交错存储（interleaved）的方式将非零值和对应索引紧密排列，使得一次内存事务可以同时获取计算所需的数据和元数据，避免了分离存储带来的额外访存延迟；此外，16 字节对齐的要求确保了 GPU L1/Shared Memory 的缓存行能被高效利用。对于 BSR 格式，逐个块进行对齐填充而非全局填充，这种细粒度的策略在不显著增加存储开销的前提下保证了每个块访问的首地址对齐。对于 CSR 格式，`interleaved[0::2]` 和 `interleaved[1::2]` 的交错写入方式利用了 PyTorch 的切片赋值，将值数组和列索引数组编织为单一连续缓冲区。这种交错布局在 GPU 向量化加载时尤其有利——warp 内的相邻线程可以连续访问 `[val0, idx0, val1, idx1, ...]` 序列，减少了全局内存事务的数量。理解这些内存优化技术，对于将稀疏算子的实际性能推向硬件理论极限至关重要。

### 33.10.3 稀疏计算的调试工具

```python
class SparseDebugger:
    """稀疏计算调试工具"""

    @staticmethod
    def verify_sparsity(matrix, expected_sparsity, tolerance=0.05):
        """验证稀疏率"""
        actual_sparsity = (matrix == 0).float().mean().item()
        if abs(actual_sparsity - expected_sparsity) > tolerance:
            print(f"WARNING: Expected sparsity {expected_sparsity:.1%}, "
                  f"got {actual_sparsity:.1%}")
            return False
        return True

    @staticmethod
    def verify_format(sparse_data, format_type):
        """验证稀疏格式的正确性"""
        if format_type == "2:4":
            values = sparse_data["values"]
            indices = sparse_data["indices"]

            # 检查每 4 个元素中恰好有 2 个非零
            for row in range(values.shape[0]):
                for group in range(values.shape[1]):
                    nz_count = (values[row, group] != 0).sum().item()
                    if nz_count != 2:
                        print(f"ERROR: Row {row}, Group {group} has {nz_count} non-zeros")
                        return False

            # 检查索引范围
            if (indices >= 4).any() or (indices < 0).any():
                print("ERROR: Index out of range")
                return False

        elif format_type == "csr":
            row_ptr = sparse_data["row_ptr"]
            values = sparse_data["values"]

            # 检查 row_ptr 单调递增
            if not (row_ptr[1:] >= row_ptr[:-1]).all():
                print("ERROR: row_ptr not monotonically increasing")
                return False

            # 检查 nnz 一致性
            if row_ptr[-1] != len(values):
                print("ERROR: nnz mismatch")
                return False

        return True

    @staticmethod
    def profile_sparse_gemm(A, B_sparse, format_type, num_runs=100):
        """Profile 稀疏 GEMM 性能"""
        import time

        # Warmup
        for _ in range(10):
            sparse_gemm(A, B_sparse, format_type)
        torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for _ in range(num_runs):
            sparse_gemm(A, B_sparse, format_type)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / num_runs

        # 计算性能指标
        M, K = A.shape
        N = B_sparse.shape[-1]
        dense_flops = 2 * M * N * K

        # 稀疏 FLOPS
        if format_type == "2:4":
            sparse_flops = dense_flops * 0.5
        elif format_type == "bsr":
            sparsity = 1 - B_sparse["num_blocks"] / (K // B_sparse["block_size"] * N // B_sparse["block_size"])
            sparse_flops = dense_flops * (1 - sparsity)

        tflops = sparse_flops / elapsed / 1e12

        return {
            "elapsed_ms": elapsed * 1000,
            "dense_tflops": dense_flops / elapsed / 1e12,
            "sparse_tflops": tflops,
            "speedup": dense_flops / sparse_flops,
        }
```

`SparseDebugger` 类以三个维度的验证函数构成了稀疏计算的完整调试工具链。第一个维度是稀疏率验证：`verify_sparsity` 通过统计零元素占比并与目标稀疏率对比，能在模型稀疏化后快速发现剪枝不到位或过度剪枝的问题。第二个维度是格式正确性验证：`verify_format` 针对不同稀疏格式（2:4、CSR 等）检查其结构约束是否满足——例如 2:4 格式要求每 4 个元素恰好 2 个非零、CSR 要求 `row_ptr` 单调递增，这些检查能有效预防因格式转换错误导致的静默精度损失。第三个维度是性能剖析：`profile_sparse_gemm` 通过运行多轮预热后的 Benchmark 循环来计算实际计算吞吐量和加速比。值得注意的是，`profile_sparse_gemm` 中的 `torch.cuda.synchronize()` 调用是保证计时准确性的关键——它确保所有 GPU 操作在计时停止前完成，避免异步执行带来的计时偏差。这三个维度的验证相互补充，形成了“正确性 + 性能”的双重保障机制，是稀疏算子上线前不可或缺的质量门禁。

### 33.10.4 稀疏计算最佳实践总结

```
稀疏计算最佳实践：

1. 格式选择
   ├── 2:4 稀疏 → 硬件支持时首选
   ├── BSR → 块稀疏模式
   ├── CSR → 通用稀疏
   └── ELL → 每行非零数相近

2. 性能优化
   ├── 使用硬件加速（如 NVIDIA 2:4）
   ├── 优化内存布局（对齐、局部性）
   ├── 使用 Software Pipelining
   └── 减少索引存储开销

3. 工程实践
   ├── 验证稀疏格式正确性
   ├── Profile 确认性能收益
   ├── 处理边界条件
   └── 与稠密计算混合使用

4. 调试技巧
   ├── 检查稀疏率是否符合预期
   ├── 验证索引范围
   ├── 对比稠密和稀疏结果
   └── 使用 NCU 分析硬件利用率
```

---

## Summary

| 稀疏类型 | 加速比 | 硬件要求 | 适用场景 |
|----------|--------|---------|---------|
| 2:4 结构化 | 1.5-2x | NVIDIA A100+ | 权重剪枝 |
| Block Sparse | 1.5-3x | 通用 | 块稀疏模式 |
| 非结构化 | 1.3-1.8x | 通用 | 通用稀疏 |
| 注意力稀疏 | 2-5x | 通用 | 长序列 |

以上总结表清晰地展示了不同稀疏类型在加速比、硬件门槛和应用场景上的差异。在实际工程中，选择稀疏方案时需要综合考虑硬件兼容性、稀疏模式规则性和性能回报率三个维度。2:4 结构化稀疏虽然硬件门槛最高，但凭借 NVIDIA 从 Ampere 到 Hopper 的持续硬件支持，已经成为推理加速的首选方案。Block Sparse 则以通用性和灵活性见长，特别适合那些稀疏模式具有天然块结构的场景（如 MoE 路由、分块注意力）。读者在后续章节中将看到这些稀疏技术在 TileLang 中的完整实现与性能调优实践，从而形成从理论到落地的完整知识体系。

---

## Exercises

### Exercise 1: 2:4 稀疏权重转换
实现一个函数，将任意 FP16 权重矩阵转换为 2:4 稀疏格式，并验证稀疏率。

### Exercise 2: Block Sparse GEMM
实现一个 TileLang Block Sparse GEMM，要求：
- 支持任意块大小
- 使用 BSR 格式存储
- 达到理论性能的 80%+

### Exercise 3: 稀疏 FlashAttention
实现一个支持块稀疏掩码的 FlashAttention，要求：
- 支持因果掩码
- 支持自定义块稀疏模式
- 性能优于全注意力

---

## Thinking Questions

1. **为什么 2:4 稀疏的实际加速比低于理论值 2x？** 提示：考虑索引存储、非规则访问和硬件限制。

2. **在什么稀疏率下，稀疏计算比稠密计算更高效？** 提示：考虑固定开销与可变收益的平衡。

3. **Block Sparse 的块大小如何选择？** 提示：考虑缓存行大小、内存对齐和计算效率。

4. **稀疏 FlashAttention 如何处理动态稀疏模式？** 提示：考虑在线稀疏度估计和动态调度。

---

## Extension Reading

1. **NVIDIA Structured Sparsity Whitepaper** - NVIDIA 2:4 稀疏技术白皮书
2. **Sparse GPU Kernels for Deep Learning** - 稀疏 GPU 内核论文
3. **FlashAttention with Block Sparse Pattern** - 稀疏注意力论文
4. **Megablocks: Efficient Sparse Training** - 稀疏训练框架
5. **TileLang Sparse Programming Guide** - TileLang 稀疏编程指南

---

## Next Chapter Preview

> **Chapter 34: TileLang 生态前沿与未来展望**
>
> 下一章将探讨 TileLang 的未来发展方向，包括更多硬件后端支持、AI Agent 辅助算子生成、Auto Schedule 智能调优进化、与 PyTorch 2.0/torch.compile 的深度集成等前沿话题。

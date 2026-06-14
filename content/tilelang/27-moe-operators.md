---
title: "Chapter 27: MoE 算子与专家并行"
description: "深入理解 Mixture-of-Experts 架构的核心算子实现，包括 Expert GEMM 调度、Token 分发汇聚、Grouped GEMM 等，掌握 TileLang 在 MoE 系统中的应用"
updated: "2025-01-01"
---

# Chapter 27: MoE 算子与专家并行

> **Learning Objectives**
>
> 1. 理解 MoE（Mixture-of-Experts）架构的基本原理
> 2. 掌握 Router/Expert/Combiner 三大组件的实现
> 3. 学会 Expert GEMM 的调度策略
> 4. 理解 Token 分发与汇聚的实现机制
> 5. 掌握 Gating 机制的实现方法
> 6. 学会 Grouped GEMM 的 TileLang 实现
> 7. 理解动态负载均衡的挑战与解决方案
> 8. 能够分析 MoE 系统的性能瓶颈与优化策略

---

## 1. MoE 架构原理

### 1.1 什么是 MoE

Mixture-of-Experts（MoE）是一种条件计算架构，通过动态选择部分"专家"网络来处理每个输入 token，从而在增加模型参数量的同时保持计算量不变。

```
MoE 前向传播流程：

输入 Token x
    │
    ▼
┌─────────┐
│  Router  │  ← 计算 Gate 分数
│ (Gating) │
└─────────┘
    │
    ▼ Gate Scores: [0.3, 0.0, 0.5, 0.2]
    │
    ▼ Top-K Selection (K=2)
    │
    ├──→ Expert 0 (score=0.3) ──→ E_0(x) × 0.3
    ├──→ Expert 2 (score=0.5) ──→ E_2(x) × 0.5
    │
    ▼
┌─────────┐
│ Combiner │  ← 加权求和
└─────────┘
    │
    ▼
输出 y = 0.3 × E_0(x) + 0.5 × E_2(x)
```

<div data-component="MoEArchitectureDiagram"></div>

### 1.2 MoE 的数学表达

对于输入 token $x$，MoE 层的输出为：

$$
y = \sum_{i=0}^{E-1} g_i(x) \cdot E_i(x)
$$

其中：
- $E$ 是专家总数
- $g_i(x)$ 是 Router 为 token $x$ 分配给专家 $i$ 的权重
- $E_i(x)$ 是专家 $i$ 对 token $x$ 的计算结果

Gate 函数通常使用 Top-K 选择：

$$
g_i(x) = \begin{cases}
\frac{\text{softmax}(W_g \cdot x)_i}{\sum_{j \in \text{TopK}} \text{softmax}(W_g \cdot x)_j} & \text{if } i \in \text{TopK} \\
0 & \text{otherwise}
\end{cases}
$$

### 1.3 MoE 的优势

| 特性 | Dense Model | MoE Model |
|------|------------|-----------|
| 参数量 | P | P × E（E 为专家数） |
| 计算量 | FLOPs | FLOPs × K/E（K 为 Top-K） |
| 内存占用 | 低 | 高（需存储所有专家） |
| 模型容量 | 有限 | 可扩展 |
| 训练效率 | 标准 | 需要负载均衡损失 |

> [!TIP]
> MoE 的核心优势在于解耦了模型参数量和计算量。通过增加专家数量，可以在不增加推理计算量的情况下大幅增加模型容量。

### 1.4 代表性 MoE 模型

| 模型 | 专家数 | Top-K | 特点 |
|------|--------|-------|------|
| Switch Transformer | 128-2048 | 1 | 简化路由，单专家选择 |
| GShard | 2048 | 2 | 双专家选择 |
| DeepSeek-V3 | 256 | 8 | 共享专家 + 路由专家 |
| Mixtral 8x7B | 8 | 2 | 少量大专家 |
| Grok-1 | 8 | 2 | 类似 Mixtral |

---

理解了 MoE 的整体架构后，接下来我们需要深入到具体实现层面。Router 是 MoE 系统的"大脑"，它决定了每个 token 的计算路径——这个看似简单的选择过程，实际上蕴含着丰富的设计考量和工程挑战。从门控函数的数值稳定性到 Top-K 选择的并行实现，Router 的设计质量直接影响整个 MoE 系统的模型容量利用率和训练稳定性。

## 2. Router 实现

### 2.1 Router 的计算过程

Router 负责为每个 token 计算专家选择概率：

```python
import tilelang
from tilelang import Profiler
import tilelang.language as T
import torch

def router_forward(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
):
    """Router forward pass."""

    @T.prim_func
    def router_kernel(
        X: T.Buffer((batch_size, seq_len, hidden_dim), "float32"),
        W_gate: T.Buffer((num_experts, hidden_dim), "float32"),
        GateScores: T.Buffer((batch_size, seq_len, num_experts), "float32"),
        TopKIndices: T.Buffer((batch_size, seq_len, top_k), "int32"),
        TopKWeights: T.Buffer((batch_size, seq_len, top_k), "float32"),
    ):
        with T.Kernel(batch_size, seq_len, threads=256) as (b, s):
            # Compute gate scores: scores = X @ W_gate^T
            scores = T.alloc_fragment((num_experts,), "float32")

            for e in T.serial(num_experts):
                acc = T.alloc_fragment((1,), "float32")
                acc[0] = T.float32(0)
                for d in T.serial(hidden_dim):
                    acc[0] += X[b, s, d] * W_gate[e, d]
                scores[e] = acc[0]

            # Apply softmax
            max_val = T.alloc_fragment((1,), "float32")
            max_val[0] = T.float32(-1e30)
            for e in T.serial(num_experts):
                max_val[0] = T.max(max_val[0], scores[e])

            sum_val = T.alloc_fragment((1,), "float32")
            sum_val[0] = T.float32(0)
            for e in T.serial(num_experts):
                scores[e] = T.exp(scores[e] - max_val[0])
                sum_val[0] += scores[e]

            for e in T.serial(num_experts):
                scores[e] /= sum_val[0]
                GateScores[b, s, e] = scores[e]

            # Top-K selection
            for k in T.serial(top_k):
                best_idx = T.alloc_fragment((1,), "int32")
                best_val = T.alloc_fragment((1,), "float32")
                best_val[0] = T.float32(-1e30)
                best_idx[0] = T.int32(0)

                for e in T.serial(num_experts):
                    if scores[e] > best_val[0]:
                        best_val[0] = scores[e]
                        best_idx[0] = e

                TopKIndices[b, s, k] = best_idx[0]
                TopKWeights[b, s, k] = best_val[0]
                scores[best_idx[0]] = T.float32(-1e30)  # Mark as selected

            # Renormalize weights
            weight_sum = T.alloc_fragment((1,), "float32")
            weight_sum[0] = T.float32(0)
            for k in T.serial(top_k):
                weight_sum[0] += TopKWeights[b, s, k]
            for k in T.serial(top_k):
                TopKWeights[b, s, k] /= weight_sum[0]

    return router_kernel
```

这段代码实现了 Router 的核心计算流程，负责为每个 token 计算专家选择概率。代码首先通过矩阵乘法计算每个 token 对所有专家的原始分数，然后使用 softmax 将分数归一化为概率分布。Top-K 选择采用贪心策略，每次选取最高分的专家并将其分数置为负无穷以避免重复选择。最后对选中的 K 个专家权重进行重新归一化，确保权重和为 1。这种设计的优势在于简单高效，但需要注意 softmax 计算的数值稳定性（使用 max_val 减法防止溢出）以及 Top-K 选择的时间复杂度为 O(K×E)。

进一步深入分析，这段 Router 实现采用了 register-level 的数据局部性优化策略。通过 `T.alloc_fragment` 将中间变量放置在寄存器文件中，避免了共享内存的访问延迟。在 softmax 计算中，先做 max 减法再做 exp 是数值计算中的标准范式，它确保了即使原始分数差异很大（例如过大或过小），exp 之后也不会出现上溢或下溢。而在 Top-K 选择阶段，采用"选择后置为负无穷"而非维护一个堆结构，是因为 K 通常很小（2-8），此时贪心线性扫描的常数因子远小于堆操作的开销。这种权衡在 GPU 架构上尤为关键——分支发散（branch divergence）是性能杀手，而线性扫描的 if-else 判断在 warp 内部可能导致线程间执行路径不一致，实际工程中通常会配合 warp-level 的归约操作来进一步优化 Top-K 的选择效率。

### 2.2 路由策略对比

| 策略 | 说明 | 优点 | 缺点 |
|------|------|------|------|
| Token Choice | 每个 token 选择 Top-K 专家 | 灵活，适合大多数场景 | 可能负载不均衡 |
| Expert Choice | 每个专家选择 Top-K tokens | 天然负载均衡 | 可能丢弃 token |
| Soft MoE | 使用软权重，不稀疏选择 | 无负载问题 | 计算量大 |
| Hash Routing | 使用哈希函数分配 | 简单高效 | 分配质量依赖哈希函数 |

<div data-component="ExpertGEMMScheduling"></div>

---

Router 为每个 token 决定了目标专家，下一步就是将这个决策转化为实际的计算。Expert GEMM 调度是 MoE 实现中最具挑战性的环节——不同于传统的 Dense GEMM 中所有 token 共享同一套权重，MoE 中每个专家有独立的权重矩阵，而不同专家接收到的 token 数量又极度不均。如何将这些"小且散"的矩阵乘法高效地组织起来，是决定 MoE 系统吞吐量的关键所在。

## 3. Expert GEMM 调度策略

### 3.1 朴素实现：循环遍历专家

```python
def moe_naive(
    num_tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
):
    """Naive MoE implementation: loop over experts."""

    @T.prim_func
    def moe_kernel(
        X: T.Buffer((num_tokens, hidden_dim), "float32"),
        W_up: T.Buffer((num_experts, intermediate_dim, hidden_dim), "float32"),
        W_down: T.Buffer((num_experts, hidden_dim, intermediate_dim), "float32"),
        TopKIndices: T.Buffer((num_tokens, 2), "int32"),
        TopKWeights: T.Buffer((num_tokens, 2), "float32"),
        Y: T.Buffer((num_tokens, hidden_dim), "float32"),
    ):
        with T.Kernel(num_tokens, threads=256) as (token_idx):
            for k in T.serial(2):
                expert_idx = TopKIndices[token_idx, k]
                weight = TopKWeights[token_idx, k]

                # Expert FFN: up projection
                hidden = T.alloc_fragment((intermediate_dim,), "float32")
                for i in T.serial(intermediate_dim):
                    acc = T.alloc_fragment((1,), "float32")
                    acc[0] = T.float32(0)
                    for d in T.serial(hidden_dim):
                        acc[0] += X[token_idx, d] * W_up[expert_idx, i, d]
                    hidden[i] = T.gelu(acc[0])

                # Down projection
                for d in T.serial(hidden_dim):
                    acc = T.alloc_fragment((1,), "float32")
                    acc[0] = T.float32(0)
                    for i in T.serial(intermediate_dim):
                        acc[0] += hidden[i] * W_down[expert_idx, d, i]
                    Y[token_idx, d] += weight * acc[0]

    return moe_kernel
```

这段朴素实现直接体现了 MoE 计算的最直观思路：对每个 token，依次查询其被分配到的 K 个专家，分别完成每个专家的 FFN 计算，最后将结果加权累加。这种实现的教学意义在于清晰地展示了 Router→Expert→Combiner 的完整数据流。但从性能角度，它暴露了 MoE 系统的本质困境。首先，最内层三重循环（token_dim × intermediate_dim × hidden_dim）导致每个 token 的计算量虽小但访问分散，GPU 的 SM 单元无法形成有效的 warp 级并行。其次，权重加载 `W_up[expert_idx, i, d]` 中的 expert_idx 是动态的，这意味着不同 token 访问的权重区域可能完全不同，L2 cache 命中率极低。更糟糕的是，当多个 token 恰好被路由到同一专家时，这些 token 会重复加载完全相同的权重矩阵，造成严重的内存带宽浪费。在实际的 8 专家 Mixtral 模型中，这种朴素实现的 GPU 利用率通常不足 30%，计算吞吐量远低于理论峰值的十分之一。

### 3.2 问题分析

朴素实现的问题：

1. **低并行度**：每个 token 串行处理 K 个专家
2. **低 GPU 利用率**：专家权重加载不连续
3. **无法利用 GEMM 优化**：每个专家的矩阵乘法太小

### 3.3 改进策略：Token 分组批量 GEMM

```python
def moe_grouped_gemm(
    num_tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    top_k: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 32,
):
    """MoE with grouped GEMM strategy."""

    @T.prim_func
    def grouped_gemm_kernel(
        X: T.Buffer((num_tokens, hidden_dim), "float32"),
        W: T.Buffer((num_experts, intermediate_dim, hidden_dim), "float32"),
        TokenExpertMap: T.Buffer((num_experts, num_tokens), "int32"),
        TokenCount: T.Buffer((num_experts,), "int32"),
        Y: T.Buffer((num_tokens, intermediate_dim), "float32"),
    ):
        # Process each expert's tokens in parallel
        with T.Kernel(num_experts, T.ceildiv(num_tokens, block_M), threads=256) as (
            expert_idx, tile_idx
        ):
            num_assigned = TokenCount[expert_idx]

            if tile_idx * block_M < num_assigned:
                Y_local = T.alloc_fragment((block_M, block_N), "float32")
                T.clear(Y_local)

                # Process tokens assigned to this expert
                for k_tile in T.serial(T.ceildiv(hidden_dim, block_K)):
                    X_local = T.alloc_fragment((block_M, block_K), "float32")
                    W_local = T.alloc_fragment((block_K, block_N), "float32")

                    # Load token features
                    for i in T.serial(block_M):
                        token_idx = tile_idx * block_M + i
                        if token_idx < num_assigned:
                            real_token = TokenExpertMap[expert_idx, token_idx]
                            for j in T.serial(block_K):
                                k_idx = k_tile * block_K + j
                                if k_idx < hidden_dim:
                                    X_local[i, j] = X[real_token, k_idx]
                                else:
                                    X_local[i, j] = T.float32(0)

                    # Load weight slice
                    for i, j in T.serial(block_K, block_N):
                        k_idx = k_tile * block_K + i
                        if k_idx < hidden_dim:
                            W_local[i, j] = W[expert_idx, j, k_idx]
                        else:
                            W_local[i, j] = T.float32(0)

                    T.gemm(X_local, W_local, Y_local)

                # Store results
                for i in T.serial(block_M):
                    token_idx = tile_idx * block_M + i
                    if token_idx < num_assigned:
                        real_token = TokenExpertMap[expert_idx, token_idx]
                        for j in T.serial(block_N):
                            Y[real_token, j] = Y_local[i, j]

    return grouped_gemm_kernel
```

Grouped GEMM 策略的核心思想是将"按 token 找专家"的访问模式彻底反转，变为"按专家找 token"。通过事先构建 TokenExpertMap（专家到 token 的映射表），每个专家的计算变成了一个独立的、连续的 GEMM 操作：该专家收到的所有 token 构成矩阵 M 维度，权重矩阵本身构成 N 和 K 维度。这种转换带来了三重性能收益：第一，TileLang 的分块 GEMM 原语 `T.gemm(X_local, W_local, Y_local)` 可以充分利用 Tensor Core 的硬件加速，这是朴素实现完全无法做到的；第二，每个专家的权重只加载一次，被该专家处理的所有 token 共享，消除了重复加载；第三，连续的内存访问模式大幅提升了 L2 cache 和全局内存带宽的利用率。在实际的 64 专家 MoE 部署中，Grouped GEMM 相比朴素实现的加速比通常可达 5-10 倍，是 MoE 高效实现的基石。

---

Token 分组需要依赖一个关键的前置步骤：将路由决策转化为"每个专家收到哪些 token"的映射关系。这个过程称为 Token 分发（Dispatch），它是 MoE 数据流中的"交通调度"环节——将来自不同序列位置的 token 按照路由结果重新排列，为后续的 Grouped GEMM 做好准备。

## 4. Token 分发与汇聚

### 4.1 Token 分发机制

<div data-component="TokenDispatchFlow"></div>

Token 分发是将输入 token 按照路由结果分配到对应专家的过程：

```python
def token_dispatch(
    num_tokens: int,
    num_experts: int,
    top_k: int,
):
    """Token dispatch: create expert-to-token mapping."""

    @T.prim_func
    def dispatch_kernel(
        TopKIndices: T.Buffer((num_tokens, top_k), "int32"),
        TopKWeights: T.Buffer((num_tokens, top_k), "float32"),
        TokenExpertMap: T.Buffer((num_experts, num_tokens), "int32"),
        TokenWeightMap: T.Buffer((num_experts, num_tokens), "float32"),
        TokenCount: T.Buffer((num_experts,), "int32"),
    ):
        with T.Kernel(num_experts, threads=1) as (expert_idx):
            count = T.alloc_fragment((1,), "int32")
            count[0] = 0

            for token_idx in T.serial(num_tokens):
                for k in T.serial(top_k):
                    if TopKIndices[token_idx, k] == expert_idx:
                        TokenExpertMap[expert_idx, count[0]] = token_idx
                        TokenWeightMap[expert_idx, count[0]] = TopKWeights[token_idx, k]
                        count[0] += 1

            TokenCount[expert_idx] = count[0]

    return dispatch_kernel
```

这段基础的 dispatch 实现采用每个专家独立扫描所有 token 的方式，逻辑清晰但并行度有限。每个专家线程（threads=1）需要串行遍历全部 num_tokens × top_k 个路由决策，当 token 数量达到数万级别时，这种串行扫描的延迟会显著影响整体 MoE 层的前向传播时间。一个重要的设计考量是 TokenExpertMap 的布局：它为每个专家预留了 num_tokens 大小的空间（上界），这保证了极端情况下（所有 token 都路由到同一专家）不会发生越界，但带来了 O(num_experts × num_tokens) 的存储开销。在实际部署中，当专家数量达到 256 且 token 数量为 8192 时，仅 TokenExpertMap 就需要 256 × 8192 × 4 bytes = 8MB 的存储空间，这在推理的 KV-cache 已经占用大量显存的场景下是一个不可忽视的额外开销。

### 4.2 高效的 Dispatch 实现

```python
def efficient_token_dispatch(
    num_tokens: int,
    num_experts: int,
    top_k: int,
):
    """Efficient token dispatch using parallel prefix sum."""

    @T.prim_func
    def dispatch_kernel(
        TopKIndices: T.Buffer((num_tokens, top_k), "int32"),
        TopKWeights: T.Buffer((num_tokens, top_k), "float32"),
        ExpertCounts: T.Buffer((num_experts,), "int32"),
        SortedIndices: T.Buffer((num_tokens * top_k,), "int32"),
        SortedWeights: T.Buffer((num_tokens * top_k,), "float32"),
        SortedExpertIds: T.Buffer((num_tokens * top_k,), "int32"),
    ):
        total_entries = num_tokens * top_k

        # Step 1: Count tokens per expert
        with T.Kernel(num_experts, threads=256) as (e):
            local_count = T.alloc_fragment((1,), "int32")
            local_count[0] = 0

            for i in T.serial(T.ceildiv(total_entries, 256)):
                idx = i * 256 + T.thread_id()
                if idx < total_entries:
                    token_idx = idx // top_k
                    k_idx = idx % top_k
                    if TopKIndices[token_idx, k_idx] == e:
                        local_count[0] += 1

            # Reduce count
            for offset in T.serial(5):
                local_count[0] += T.warp_shuffle_down(local_count[0], 1 << offset)

            shared = T.alloc_shared((8,), "int32")
            warp_id = T.thread_id() // 32
            lane_id = T.thread_id() % 32

            if lane_id == 0:
                shared[warp_id] = local_count[0]
            T.syncthreads()

            if T.thread_id() == 0:
                total = 0
                for i in T.serial(8):
                    total += shared[i]
                ExpertCounts[e] = total

        # Step 2: Prefix sum on counts (for scatter offsets)
        # ... (prefix sum implementation)

        # Step 3: Scatter tokens to sorted positions
        with T.Kernel(T.ceildiv(total_entries, 256), threads=256) as (block_idx):
            idx = block_idx * 256 + T.thread_id()
            if idx < total_entries:
                token_idx = idx // top_k
                k_idx = idx % top_k
                expert_id = TopKIndices[token_idx, k_idx]

                # Atomic increment for position
                pos = T.atomic_add(ExpertCounts, expert_id, 1)
                SortedIndices[pos] = token_idx
                SortedWeights[pos] = TopKWeights[token_idx, k_idx]
                SortedExpertIds[pos] = expert_id

    return dispatch_kernel
```

高效 dispatch 的设计采用了经典的并行计数-前缀和-散射（Count-PrefixSum-Scatter）三阶段模式，这是 GPU 基数排序（Radix Sort）在 MoE 分发场景的简化应用。第一阶段使用 256 线程并行统计每个专家分配到的 token 数量，通过 warp shuffle 在寄存器层面完成线程块内的归约，避免了全局内存的原子操作开销。第二阶段的前缀和（代码中以注释省略）是确定每个专家在输出数组中起始偏移的关键——它确保了多个专家可以同时向输出数组写入而不发生冲突。第三阶段的散射使用了 `T.atomic_add` 来解决并行写入的竞争条件，但原子操作本身是串行化的，当大量 token 被分配到少数专家时（极端情况下 80% 的 token 集中在 2-3 个专家），原子操作的竞争会成为新的瓶颈。一个更优的做法是在第一步就预先计算好每个线程块内各专家的局部计数和局部偏移，将原子的竞争范围从全局缩小到块内，这可以减少 10-20 倍的原子操作碰撞概率。

### 4.3 Token 汇聚机制

Token 汇聚是将专家计算结果按路由权重加权求和：

```python
def token_combine(
    num_tokens: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
):
    """Token combine: weighted sum of expert outputs."""

    @T.prim_func
    def combine_kernel(
        ExpertOutputs: T.Buffer((num_tokens, hidden_dim), "float32"),
        TopKIndices: T.Buffer((num_tokens, top_k), "int32"),
        TopKWeights: T.Buffer((num_tokens, top_k), "float32"),
        Y: T.Buffer((num_tokens, hidden_dim), "float32"),
    ):
        with T.Kernel(num_tokens, T.ceildiv(hidden_dim, 64), threads=256) as (
            token_idx, dim_tile
        ):
            local_out = T.alloc_fragment((64,), "float32")
            T.clear(local_out)

            # Accumulate weighted outputs from selected experts
            for k in T.serial(top_k):
                expert_idx = TopKIndices[token_idx, k]
                weight = TopKWeights[token_idx, k]

                for d in T.serial(64):
                    dim_idx = dim_tile * 64 + d
                    if dim_idx < hidden_dim:
                        local_out[d] += weight * ExpertOutputs[expert_idx * num_tokens + token_idx, dim_idx]

            # Store result
            for d in T.serial(64):
                dim_idx = dim_tile * 64 + d
                if dim_idx < hidden_dim:
                    Y[token_idx, dim_idx] = local_out[d]

    return combine_kernel
```

Token 汇聚（Combine）是 dispatch 的逆过程——它将分散在各专家输出中的计算结果重新收集并按路由权重加权求和。这段实现采用了二维线程网格：token 维度保证每个 token 独立并行，hidden_dim 维度分块（tile=64）处理以保证足够的计算密度。汇聚过程的内存访问模式是一个值得深入分析的点：`ExpertOutputs[expert_idx * num_tokens + token_idx, dim_idx]` 的访问天然是非连续的，因为 expert_idx 由路由结果动态决定。当 GPU warp 内的 32 个线程处理不同的 token 时，它们可能访问完全不同的 expert_idx 偏移，导致全局内存访问无法合并为一次连续的事务（coalesced access）。一个优化思路是将输出布局从 [num_tokens, hidden_dim] 改为按 expert 组织，使得同一 expert 的输出在内存中连续存储，从而在 combine 阶段可以利用局部性。但在实践中，这种布局转换（也称为 permute 操作）本身也有开销，因此需要根据具体的专家数量和 token 分布来权衡。

---

Token 的分发和汇聚解决了"谁来算"的问题，而 Gating 机制则决定了"算多少"——权重如何分配。一个精心设计的 Gating 函数不仅要准确评估每个专家对当前 token 的适配程度，还要在训练过程中自然地引导 token 均匀分布，避免"专家破产"（expert collapse）这一 MoE 训练中最臭名昭著的问题。

## 5. Gating 机制实现

### 5.1 常见 Gating 函数

| Gating 类型 | 公式 | 特点 |
|-------------|------|------|
| Softmax Gating | $g = \text{softmax}(Wx)$ | 标准选择，权重和为 1 |
| Sigmoid Gating | $g = \sigma(Wx)$ | 独立激活，权重和可大于 1 |
| Noisy Top-K | $g = \text{softmax}(Wx + \epsilon)$ | 添加噪声探索 |
| Expert Choice | $g = \text{topk}(Wx, K)$ | 专家选择 token |

### 5.2 Noisy Gating 实现

```python
def noisy_gating(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
):
    """Noisy Top-K gating for better load balancing."""

    @T.prim_func
    def noisy_gate_kernel(
        X: T.Buffer((batch_size, seq_len, hidden_dim), "float32"),
        W_gate: T.Buffer((num_experts, hidden_dim), "float32"),
        W_noise: T.Buffer((num_experts, hidden_dim), "float32"),
        TopKIndices: T.Buffer((batch_size, seq_len, top_k), "int32"),
        TopKWeights: T.Buffer((batch_size, seq_len, top_k), "float32"),
    ):
        with T.Kernel(batch_size, seq_len, threads=256) as (b, s):
            scores = T.alloc_fragment((num_experts,), "float32")

            # Compute clean scores
            for e in T.serial(num_experts):
                acc = T.alloc_fragment((1,), "float32")
                acc[0] = T.float32(0)
                for d in T.serial(hidden_dim):
                    acc[0] += X[b, s, d] * W_gate[e, d]
                scores[e] = acc[0]

            # Add noise (using Box-Muller transform for Gaussian)
            # For simplicity, use uniform noise here
            noise_scale = T.float32(0.1)
            for e in T.serial(num_experts):
                noise = T.float32(0)  # Would be random in practice
                for d in T.serial(hidden_dim):
                    noise += X[b, s, d] * W_noise[e, d]
                scores[e] += noise_scale * T.sigmoid(noise)

            # Top-K selection with softmax
            # ... (same as before)

    return noisy_gate_kernel
```

Noisy Gating 是 Switch Transformer 论文中提出的关键技术，它通过在 softmax 之前向 logits 添加可学习的噪声项来鼓励训练早期的探索行为。噪声的大小由 `W_noise[e, d]` 参数化，并通过 sigmoid 函数压缩到 (0,1) 区间再乘以 `noise_scale` 缩放。这种设计的精妙之处在于可微分性：噪声项通过 `sigmoid(noise)` 而非直接使用 Gaussian 噪声，保证了梯度可以同时流向 W_gate 和 W_noise，使得模型能够自适应地学习"何时需要探索、何时应该确定性路由"。在训练初期，W_noise 通常被初始化为较大的值以促进 token 在各专家间的均匀分布；随着训练进行，W_gate 逐渐学会有意义的专家选择模式，W_noise 的影响自然衰减。此外，这种噪声机制还有一个额外的副作用——它等价于对路由决策施加了 dropout 式的正则化，这有助于防止模型过早陷入"所有 token 都路由到同一个专家"的局部最优。

### 5.3 Load Balancing Loss

为了防止所有 token 都路由到少数专家，需要添加辅助损失：

$$
\mathcal{L}_{\text{balance}} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot P_i
$$

其中：
- $f_i$ 是分配给专家 $i$ 的 token 比例
- $P_i$ 是所有 token 对专家 $i$ 的平均路由概率
- $\alpha$ 是平衡系数（通常为 0.01）

```python
def load_balance_loss(
    num_tokens: int,
    num_experts: int,
):
    """Compute load balancing auxiliary loss."""

    @T.prim_func
    def balance_loss_kernel(
        GateScores: T.Buffer((num_tokens, num_experts), "float32"),
        TopKIndices: T.Buffer((num_tokens, 2), "int32"),
        Loss: T.Buffer((1,), "float32"),
    ):
        with T.Kernel(1, threads=256) as ():
            # Compute f_i: fraction of tokens routed to expert i
            expert_counts = T.alloc_fragment((num_experts,), "float32")
            T.clear(expert_counts)

            for t in T.serial(T.ceildiv(num_tokens, 256)):
                idx = t * 256 + T.thread_id()
                if idx < num_tokens:
                    for k in T.serial(2):
                        e = TopKIndices[idx, k]
                        expert_counts[e] += T.float32(1)

            # Reduce counts
            # ...

            # Compute P_i: average gate probability
            expert_probs = T.alloc_fragment((num_experts,), "float32")
            T.clear(expert_probs)

            for t in T.serial(T.ceildiv(num_tokens, 256)):
                idx = t * 256 + T.thread_id()
                if idx < num_tokens:
                    for e in T.serial(num_experts):
                        expert_probs[e] += GateScores[idx, e]

            # Reduce and compute loss
            # loss = alpha * num_experts * sum(f_i * P_i)
            # ...

    return balance_loss_kernel
```

负载均衡损失的设计本质上是对抗 MoE 训练中的"马太效应"：被选中的专家获得更多梯度更新，从而在下一次路由中更容易被选中，形成正反馈循环，最终导致只有少数专家活跃而其余专家退化。损失函数 $\mathcal{L}_{\text{balance}} = \alpha \cdot E \cdot \sum f_i \cdot P_i$ 通过惩罚 $f_i$ 和 $P_i$ 的乘积来鼓励均匀分布——当 $f_i$（实际分配比例）和 $P_i$（路由概率均值）高度不匹配时（例如某专家路由概率很高但实际未被选中），损失会增大。这里有一个关键的实现细节：$f_i$ 的计算基于 Top-K 选择后的硬分配（离散），而 $P_i$ 基于 softmax 之前的概率（连续），这种"硬软结合"的设计保证了辅助损失的梯度能够有效传导到路由器参数。$\alpha$ 的选择需要经验调优——过大会使路由退化为均匀随机（丧失专家专业化），过小则无法防止专家崩塌。

---

Router 和 Gating 决定了"谁来算"和"怎么分配"，现在我们将目光转向当前 MoE 领域最受瞩目的工程实践——DeepSeek-V3 的 MoE 架构。它在传统稀疏 MoE 的基础上引入了一个看似简单但效果显著的改进：共享专家。这个设计巧妙地解决了稀疏 MoE 中长期存在的知识冗余和训练不稳定性问题。

## 6. DeepSeek-V3 MoE 架构分析

### 6.1 DeepSeek-V3 的 MoE 设计

DeepSeek-V3 采用了一种创新的 MoE 设计：

```
DeepSeek-V3 MoE 架构：

输入 x
    │
    ├──→ Shared Expert (共享专家) ──→ S(x)
    │
    └──→ Router ──→ Top-K 路由专家 ──→ Σ g_i(x) * E_i(x)
    │
    ▼
输出 y = S(x) + Σ g_i(x) * E_i(x)
```

### 6.2 关键特性

| 特性 | 说明 |
|------|------|
| 共享专家 | 始终激活的专家，捕获通用知识 |
| 路由专家 | 按需激活的专家，捕获特定知识 |
| 细粒度专家 | 每个专家更小，数量更多 |
| 节点级路由 | 跨节点的专家分布与通信 |

<div data-component="MoEPerformanceAnalysis"></div>

### 6.3 DeepSeek-V3 MoE 实现

```python
def deepseek_moe(
    num_tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_shared_experts: int,
    num_routed_experts: int,
    top_k: int,
):
    """DeepSeek-V3 style MoE with shared experts."""

    @T.prim_func
    def deepseek_moe_kernel(
        X: T.Buffer((num_tokens, hidden_dim), "float32"),
        W_shared_up: T.Buffer((num_shared_experts, intermediate_dim, hidden_dim), "float32"),
        W_shared_down: T.Buffer((num_shared_experts, hidden_dim, intermediate_dim), "float32"),
        W_routed_up: T.Buffer((num_routed_experts, intermediate_dim, hidden_dim), "float32"),
        W_routed_down: T.Buffer((num_routed_experts, hidden_dim, intermediate_dim), "float32"),
        W_gate: T.Buffer((num_routed_experts, hidden_dim), "float32"),
        Y: T.Buffer((num_tokens, hidden_dim), "float32"),
    ):
        with T.Kernel(num_tokens, threads=256) as (token_idx):
            # Part 1: Shared experts (always active)
            shared_out = T.alloc_fragment((hidden_dim,), "float32")
            T.clear(shared_out)

            for se in T.serial(num_shared_experts):
                hidden = T.alloc_fragment((intermediate_dim,), "float32")
                # Up projection
                for i in T.serial(intermediate_dim):
                    acc = T.float32(0)
                    for d in T.serial(hidden_dim):
                        acc += X[token_idx, d] * W_shared_up[se, i, d]
                    hidden[i] = T.gelu(acc)

                # Down projection
                for d in T.serial(hidden_dim):
                    acc = T.float32(0)
                    for i in T.serial(intermediate_dim):
                        acc += hidden[i] * W_shared_down[se, d, i]
                    shared_out[d] += acc

            # Part 2: Router
            gate_scores = T.alloc_fragment((num_routed_experts,), "float32")
            for e in T.serial(num_routed_experts):
                acc = T.float32(0)
                for d in T.serial(hidden_dim):
                    acc += X[token_idx, d] * W_gate[e, d]
                gate_scores[e] = acc

            # Top-K selection
            top_indices = T.alloc_fragment((top_k,), "int32")
            top_weights = T.alloc_fragment((top_k,), "float32")

            for k in T.serial(top_k):
                best_idx = 0
                best_val = T.float32(-1e30)
                for e in T.serial(num_routed_experts):
                    if gate_scores[e] > best_val:
                        best_val = gate_scores[e]
                        best_idx = e
                top_indices[k] = best_idx
                top_weights[k] = best_val
                gate_scores[best_idx] = T.float32(-1e30)

            # Normalize weights
            weight_sum = T.float32(0)
            for k in T.serial(top_k):
                weight_sum += top_weights[k]
            for k in T.serial(top_k):
                top_weights[k] /= weight_sum

            # Part 3: Routed experts
            routed_out = T.alloc_fragment((hidden_dim,), "float32")
            T.clear(routed_out)

            for k in T.serial(top_k):
                e = top_indices[k]
                w = top_weights[k]

                hidden = T.alloc_fragment((intermediate_dim,), "float32")
                for i in T.serial(intermediate_dim):
                    acc = T.float32(0)
                    for d in T.serial(hidden_dim):
                        acc += X[token_idx, d] * W_routed_up[e, i, d]
                    hidden[i] = T.gelu(acc)

                for d in T.serial(hidden_dim):
                    acc = T.float32(0)
                    for i in T.serial(intermediate_dim):
                        acc += hidden[i] * W_routed_down[e, d, i]
                    routed_out[d] += w * acc

            # Combine shared and routed outputs
            for d in T.serial(hidden_dim):
                Y[token_idx, d] = shared_out[d] + routed_out[d]

    return deepseek_moe_kernel
```

DeepSeek-V3 的 MoE 实现将共享专家和路由专家以相加的方式融合（而非串行排列），这种设计的精妙之处在于：共享专家捕获跨所有 token 的通用语言知识和语法结构，提供了稳定的"基线信号"，使路由专家可以专注于学习领域特定的、有区分性的知识模式。从优化的角度分析，共享专家的存在起到了"梯度高速公路"的作用——即使在训练早期路由分配极不均衡时，共享专家仍然为所有 token 提供一致的梯度信号，防止了 MoE 层输出方差的剧烈波动。此外，DeepSeek-V3 采用的细粒度专家策略（大量小专家而非少量大专家）在计算上等价于增加了"专家选择的组合空间"——256 个专家中选择 8 个（每个较小）的组合多样性远高于 8 个专家中选择 2 个（每个较大），这允许模型学习到更精细的知识分工。从系统层面看，细粒度专家更有利于分布式部署：每个 GPU 节点可以容纳更多的专家副本，从而减少了跨节点的 All-to-All 通信量。

---

DeepSeek-V3 的成功实践证明了高效 MoE 算子实现的关键作用。当我们拥有大量细粒度专家时，如何高效地执行这些专家的矩阵乘法就成为了系统性能的决定性因素。这正是 Grouped GEMM 技术大放异彩的舞台。

## 7. Grouped GEMM 实现

### 7.1 Grouped GEMM 的概念

Grouped GEMM 是将多个小 GEMM 操作打包在一起执行的技术，特别适合 MoE 场景：

<div data-component="GroupedGEMMDemo"></div>

```
Grouped GEMM 示例：

Expert 0: tokens [0, 3, 7] × W_0 → Y[0, 3, 7]
Expert 1: tokens [1, 4, 8] × W_1 → Y[1, 4, 8]
Expert 2: tokens [2, 5, 6] × W_2 → Y[2, 5, 6]

传统方式：3 个独立的 GEMM
Grouped GEMM：1 个融合的 Grouped GEMM
```

### 7.2 TileLang Grouped GEMM 实现

```python
def grouped_gemm_moe(
    num_experts: int,
    max_tokens_per_expert: int,
    hidden_dim: int,
    intermediate_dim: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 32,
):
    """Grouped GEMM for MoE using TileLang."""

    @T.prim_func
    def grouped_gemm_kernel(
        # Token data (sorted by expert)
        X: T.Buffer((num_experts * max_tokens_per_expert, hidden_dim), "float32"),
        # Expert weights
        W: T.Buffer((num_experts, intermediate_dim, hidden_dim), "float32"),
        # Token counts per expert
        ExpertCounts: T.Buffer((num_experts,), "int32"),
        # Output
        Y: T.Buffer((num_experts * max_tokens_per_expert, intermediate_dim), "float32"),
    ):
        # Launch one kernel per expert, with tiles over tokens
        with T.Kernel(
            num_experts,
            T.ceildiv(max_tokens_per_expert, block_M),
            T.ceildiv(intermediate_dim, block_N),
            threads=256,
        ) as (expert_idx, token_tile, dim_tile):
            num_tokens = ExpertCounts[expert_idx]

            if token_tile * block_M < num_tokens:
                # Compute GEMM for this expert's tokens
                Y_local = T.alloc_fragment((block_M, block_N), "float32")
                T.clear(Y_local)

                for k_tile in T.serial(T.ceildiv(hidden_dim, block_K)):
                    X_local = T.alloc_fragment((block_M, block_K), "float32")
                    W_local = T.alloc_fragment((block_K, block_N), "float32")

                    # Load token features (offset by expert)
                    base = expert_idx * max_tokens_per_expert
                    for i, j in T.serial(block_M, block_K):
                        token_idx = base + token_tile * block_M + i
                        k_idx = k_tile * block_K + j
                        if token_idx < base + num_tokens and k_idx < hidden_dim:
                            X_local[i, j] = X[token_idx, k_idx]
                        else:
                            X_local[i, j] = T.float32(0)

                    # Load weight slice
                    for i, j in T.serial(block_K, block_N):
                        k_idx = k_tile * block_K + i
                        n_idx = dim_tile * block_N + j
                        if k_idx < hidden_dim and n_idx < intermediate_dim:
                            W_local[i, j] = W[expert_idx, n_idx, k_idx]
                        else:
                            W_local[i, j] = T.float32(0)

                    T.gemm(X_local, W_local, Y_local)

                # Store results
                base = expert_idx * max_tokens_per_expert
                for i, j in T.serial(block_M, block_N):
                    token_idx = base + token_tile * block_M + i
                    n_idx = dim_tile * block_N + j
                    if token_idx < base + num_tokens and n_idx < intermediate_dim:
                        Y[token_idx, n_idx] = Y_local[i, j]

    return grouped_gemm_kernel
```

TileLang 的 Grouped GEMM 实现展现了三层并行结构的优雅设计。最外层按专家索引并行（num_experts），确保不同专家的计算可以在 GPU 的 SM 之间自由调度；中间层按 token 分块并行，处理同一专家内的大量 token；最内层按输出维度分块，保证每个线程块有足够的工作量来隐藏内存延迟。分块大小 block_M=64, block_N=64, block_K=32 的选择遵循了 Tensor Core 的硬件约束：A100 的 Tensor Core 期望 M×N×K 的 tile 维度为 16×16×16 或 16×8×16（FP16），选择 64×64×32 可以保证 tile 恰好被 2×4×2 个 Tensor Core warp 覆盖，避免了 tile 边界的浪费。一个关键的工程细节是 pad-to-zero 的处理：当 token 数量不是 block_M 的整数倍时，超出的位置被零填充，这保证了 T.gemm 原始始终处理完整的 tile，简化了边界逻辑但引入了约 5%-15% 的无效计算（取决于 token 分布的不均匀程度）。

### 7.3 Grouped GEMM 的性能优势

| 方案 | 内核启动次数 | GPU 利用率 | 内存效率 |
|------|------------|-----------|---------|
| 循环单 GEMM | E 次 | 低 | 差 |
| 批量 GEMM | 1 次 | 中 | 中 |
| Grouped GEMM | 1 次 | 高 | 好 |

---

Grouped GEMM 解决了计算效率的问题，但它假设了一个理想前提——每个专家的 token 负载是相对均匀的。然而在真实的 MoE 训练和推理中，token 分布的不均匀性是普遍存在的。如果不对这种不均衡加以控制，即使 GEMM 算子本身再高效，过载专家的延迟也会拖慢整个 MoE 层，形成"木桶效应"。

## 8. 动态负载均衡

### 8.1 负载不均衡问题

MoE 系统中的负载不均衡是一个核心挑战：

```
理想情况：
Expert 0: 100 tokens
Expert 1: 100 tokens
Expert 2: 100 tokens
Expert 3: 100 tokens

实际情况（不均衡）：
Expert 0: 250 tokens  ← 过载
Expert 1: 50 tokens   ← 欠载
Expert 2: 180 tokens
Expert 3: 20 tokens   ← 严重欠载
```

### 8.2 负载均衡策略

| 策略 | 方法 | 优点 | 缺点 |
|------|------|------|------|
| 辅助损失 | 添加 load balance loss | 简单 | 影响模型质量 |
| 容量限制 | 限制每个专家处理的最大 token 数 | 可控 | 可能丢弃 token |
| Expert Choice | 专家选择 token | 天然均衡 | 实现复杂 |
| 随机路由 | 添加随机噪声 | 无损 | 需要噪声调节 |

### 8.3 容量限制实现

```python
def capacity_limited_dispatch(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    capacity_factor: float = 1.25,
):
    """Capacity-limited token dispatch."""

    @T.prim_func
    def dispatch_kernel(
        TopKIndices: T.Buffer((num_tokens, top_k), "int32"),
        TopKWeights: T.Buffer((num_tokens, top_k), "float32"),
        ExpertCapacity: T.Buffer((num_experts,), "int32"),
        TokenExpertMap: T.Buffer((num_experts, int(num_tokens * 1.25 / 1)), "int32"),
        TokenWeightMap: T.Buffer((num_experts, int(num_tokens * 1.25 / 1)), "float32"),
        TokenCount: T.Buffer((num_experts,), "int32"),
        DroppedCount: T.Buffer((1,), "int32"),
    ):
        with T.Kernel(num_experts, threads=1) as (expert_idx):
            capacity = ExpertCapacity[expert_idx]
            count = T.alloc_fragment((1,), "int32")
            count[0] = 0
            dropped = T.alloc_fragment((1,), "int32")
            dropped[0] = 0

            for token_idx in T.serial(num_tokens):
                for k in T.serial(top_k):
                    if TopKIndices[token_idx, k] == expert_idx:
                        if count[0] < capacity:
                            TokenExpertMap[expert_idx, count[0]] = token_idx
                            TokenWeightMap[expert_idx, count[0]] = TopKWeights[token_idx, k]
                            count[0] += 1
                        else:
                            dropped[0] += 1

            TokenCount[expert_idx] = count[0]
            if expert_idx == 0:
                DroppedCount[0] = dropped[0]

    return dispatch_kernel
```

容量限制（Capacity Factor）是 GShard 论文中引入的工程化负载均衡策略，它为每个专家设定处理 token 的上限：`capacity = capacity_factor × num_tokens / num_experts`。当 capacity_factor=1.25 时，每个专家最多可以处理平均负载 1.25 倍的 token，超出部分的 token 被直接丢弃（随机选择丢弃哪些）。这种"硬截断"策略的代价是可能丢失信息，但其优势在于提供了确定性的延迟上限——无论路由分布多么极端，每个专家的计算时间都不会超过 capacity×FFN_time，这对生产环境的延迟 SLO 保障至关重要。值得注意的是，token 丢弃在训练中实际上起到了额外的正则化作用：被丢弃的 token 相当于经历了"随机专家 dropout"，这迫使模型不依赖特定的专家组合，增强了路由的鲁棒性。在实践中，capacity_factor 的调优是一个经验活——过小（如 1.0）会导致大量丢弃影响收敛，过大（如 2.0）则失去负载均衡效果。

---

负载均衡为 MoE 系统提供了稳定性保障，但要将这些技术真正落地，我们还需要一个能够灵活表达上述所有算子细节的编程框架。TileLang 凭借其显式的内存管理、灵活的分块控制和原生的 GEMM 支持，为 MoE 算子的高效实现提供了独特优势。

## 9. TileLang 在 MoE 中的应用

### 9.1 TileLang 的优势

TileLang 在 MoE 实现中的独特优势：

1. **灵活的分块控制**：可以精细控制专家计算的并行策略
2. **显式内存管理**：避免 MoE 中频繁的内存分配
3. **自定义 GEMM**：针对专家特点定制 GEMM 实现
4. **跨专家并行**：支持多专家同时计算

### 9.2 完整 MoE 前向传播

```python
def moe_forward_complete(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    top_k: int,
):
    """Complete MoE forward pass using TileLang."""

    @T.prim_func
    def moe_forward_kernel(
        X: T.Buffer((batch_size, seq_len, hidden_dim), "float32"),
        W_gate: T.Buffer((num_experts, hidden_dim), "float32"),
        W_up: T.Buffer((num_experts, intermediate_dim, hidden_dim), "float32"),
        W_down: T.Buffer((num_experts, hidden_dim, intermediate_dim), "float32"),
        Y: T.Buffer((batch_size, seq_len, hidden_dim), "float32"),
    ):
        num_tokens = batch_size * seq_len

        # Step 1: Router - compute gate scores
        GateScores = T.alloc_buffer((num_tokens, num_experts), "float32")
        TopKIndices = T.alloc_buffer((num_tokens, top_k), "int32")
        TopKWeights = T.alloc_buffer((num_tokens, top_k), "float32")

        with T.Kernel(num_tokens, threads=128) as (token_idx):
            scores = T.alloc_fragment((num_experts,), "float32")
            for e in T.serial(num_experts):
                acc = T.float32(0)
                for d in T.serial(hidden_dim):
                    flat_idx = token_idx
                    b = flat_idx // seq_len
                    s = flat_idx % seq_len
                    acc += X[b, s, d] * W_gate[e, d]
                scores[e] = acc

            # Softmax
            max_val = T.float32(-1e30)
            for e in T.serial(num_experts):
                max_val = T.max(max_val, scores[e])
            sum_val = T.float32(0)
            for e in T.serial(num_experts):
                scores[e] = T.exp(scores[e] - max_val)
                sum_val += scores[e]
            for e in T.serial(num_experts):
                scores[e] /= sum_val
                GateScores[token_idx, e] = scores[e]

            # Top-K
            for k in T.serial(top_k):
                best_idx = 0
                best_val = T.float32(-1e30)
                for e in T.serial(num_experts):
                    if scores[e] > best_val:
                        best_val = scores[e]
                        best_idx = e
                TopKIndices[token_idx, k] = best_idx
                TopKWeights[token_idx, k] = best_val
                scores[best_idx] = T.float32(-1e30)

            # Normalize
            w_sum = T.float32(0)
            for k in T.serial(top_k):
                w_sum += TopKWeights[token_idx, k]
            for k in T.serial(top_k):
                TopKWeights[token_idx, k] /= w_sum

        # Step 2: Expert computation
        ExpertOut = T.alloc_buffer((num_tokens, hidden_dim), "float32")
        T.clear(ExpertOut)

        for e in T.serial(num_experts):
            with T.Kernel(T.ceildiv(num_tokens, 64), threads=256) as (tile_idx):
                for i in T.serial(64):
                    token_idx = tile_idx * 64 + i
                    if token_idx < num_tokens:
                        # Check if this token routes to expert e
                        weight = T.float32(0)
                        for k in T.serial(top_k):
                            if TopKIndices[token_idx, k] == e:
                                weight = TopKWeights[token_idx, k]

                        if weight > T.float32(0):
                            # FFN computation
                            hidden = T.alloc_fragment((intermediate_dim,), "float32")
                            for j in T.serial(intermediate_dim):
                                acc = T.float32(0)
                                for d in T.serial(hidden_dim):
                                    b = token_idx // seq_len
                                    s = token_idx % seq_len
                                    acc += X[b, s, d] * W_up[e, j, d]
                                hidden[j] = T.gelu(acc)

                            for d in T.serial(hidden_dim):
                                acc = T.float32(0)
                                for j in T.serial(intermediate_dim):
                                    acc += hidden[j] * W_down[e, d, j]
                                ExpertOut[token_idx, d] += weight * acc

        # Step 3: Write output
        with T.Kernel(num_tokens, threads=256) as (token_idx):
            for d in T.serial(hidden_dim):
                b = token_idx // seq_len
                s = token_idx % seq_len
                Y[b, s, d] = ExpertOut[token_idx, d]

    return moe_forward_kernel
```

这个完整的 MoE 前向传播实现将 Router、Expert Computation、Output Write 三个步骤编排在一个 TileLang 函数中，展示了端到端的 MoE 计算流程。在 Step 2 的专家计算部分，代码通过 `for e in T.serial(num_experts)` 循环遍历所有专家，每个专家内部使用 256 线程的 kernel 并行处理 token。虽然这种实现比起 Grouped GEMM 在计算效率上有所折衷，但其优势在于数据流的清晰性：GateScores、TopKIndices 等中间结果通过 `T.alloc_buffer` 分配在全局内存中，不同 kernel 之间通过这些 buffer 传递数据，形成了一个显式的生产者-消费者依赖链。这种设计使得 TileLang 编译器可以进行跨 kernel 的优化——例如将相邻 kernel 的 buffer 分配合并为一次内存分配以减少碎片，或者通过 buffer 的生命周期分析在不同 kernel 之间复用内存。此外，Step 2 中每个专家 kernel 仅处理实际被路由到该专家的 token（通过 `if weight > 0` 判断），这在负载均衡良好的情况下可以避免约 (1 - top_k/E) 比例的无效计算。

---

掌握了 MoE 前向传播的完整实现后，我们接下来要将视角从单 GPU 扩展到多 GPU 环境。在实际的大规模部署中，MoE 的性能瓶颈往往不在计算本身，而在跨设备的通信开销。理解通信模式和优化策略，是从"能跑"到"跑得快"的关键跨越。

## 10. 性能优化与通信开销分析

### 10.1 MoE 的性能瓶颈

| 瓶颈 | 原因 | 优化方向 |
|------|------|---------|
| 路由开销 | 逐 token 计算 gate 分数 | 批量化 gate 计算 |
| 负载不均衡 | 部分专家过载 | 负载均衡策略 |
| 内存带宽 | 专家权重加载 | 权重预取、缓存 |
| 跨节点通信 | Token 需要发送到远程专家 | 通信与计算重叠 |

### 10.2 通信优化策略

```
All-to-All 通信模式：

节点 0                    节点 1
┌─────────┐              ┌─────────┐
│ Expert 0 │ ←────────── │ Token A │
│ Expert 1 │ ←────────── │ Token B │
└─────────┘              └─────────┘
┌─────────┐              ┌─────────┐
│ Token C  │ ──────────→ │ Expert 2 │
│ Token D  │ ──────────→ │ Expert 3 │
└─────────┘              └─────────┘

优化：计算与通信重叠
┌────────────────────────────────────┐
│ Time                               │
│ ├── All-to-All Send ──────────┐    │
│ │   ├── Expert 0 Compute ─────┤    │
│ │   │   ├── Expert 1 Compute ─┤    │
│ │   │   │   ├── All-to-All Recv ── │
└────────────────────────────────────┘
```

### 10.3 性能基准对比

| 配置 | Dense FFN | MoE (8 experts) | MoE (64 experts) |
|------|-----------|-----------------|------------------|
| 参数量 | 1B | 8B | 64B |
| 计算量 | 1B FLOPs | 1B FLOPs | 1B FLOPs |
| 内存占用 | 4 GB | 32 GB | 256 GB |
| 延迟 (A100) | 0.5 ms | 0.8 ms | 1.2 ms |
| 吞吐量 | 2000 tok/s | 1250 tok/s | 833 tok/s |

> [!WARNING]
> MoE 虽然可以大幅增加模型容量，但会带来额外的内存占用和通信开销。在实际部署中，需要仔细权衡模型容量和系统效率。

---

性能分析明确了通信是 MoE 的核心瓶颈，这自然引出了一个更深层的问题：在多 GPU 的环境下，如何将不同的专家分布到不同的设备上，使得每个 GPU 既能执行本地专家的计算，又能高效地与其他 GPU 交换 token 数据？这就是专家并行（Expert Parallelism）所要解决的核心问题。

## 11. 高级话题：MoE 的分布式训练

### 11.1 Expert Parallelism

Expert Parallelism 是将不同专家分布在不同设备上的并行策略：

```python
# Expert Parallelism 示例
# 8 个专家分布在 2 个 GPU 上
# GPU 0: Expert 0, 1, 2, 3
# GPU 1: Expert 4, 5, 6, 7

def expert_parallel_forward():
    """Expert parallel forward pass."""
    # 1. Router 计算（所有 GPU 都执行）
    gate_scores = router(x)

    # 2. All-to-All 通信：将 token 发送到对应专家所在的 GPU
    dispatched_tokens = all_to_all_dispatch(x, gate_scores)

    # 3. 本地专家计算
    local_outputs = local_expert_compute(dispatched_tokens)

    # 4. All-to-All 通信：将结果发回原始 GPU
    combined_outputs = all_to_all_combine(local_outputs)

    return combined_outputs
```

这段伪代码精确地描述了 Expert Parallelism 的四个阶段。阶段 1 中 Router 计算是数据并行的——每个 GPU 持有不同的 batch 数据，但都独立运行相同的 Router 逻辑。阶段 2 的 All-to-All Dispatch 是通信密集型操作：每个 GPU 需要将本地 batch 中的每个 token 发送到持有目标专家的远程 GPU。这种通信模式的特点是全互联（all-to-all），即每个 GPU 既向所有其他 GPU 发送数据，又从所有其他 GPU 接收数据。在 NCCL 的实现中，这是通过环式 All-to-All（Ring All-to-All）算法完成的，通信量为 `(num_gpus - 1) / num_gpus × total_tokens × hidden_dim × top_k`。阶段 3 的本地计算是完全并行的，没有通信开销。阶段 4 是阶段 2 的逆过程，将专家计算结果发送回 token 原始所在的 GPU。整个流程中，通信占总时间的比例随专家数量的增加而上升——当专家数从 8 增加到 256 时，在典型的 8 GPU 部署中，通信占比可能从 15% 上升到 40% 以上。

### 11.2 Pipeline MoE

```python
# Pipeline MoE：将专家计算与通信重叠
def pipeline_moe():
    """Pipeline MoE with overlapped communication."""
    # Batch 0: 发送 tokens
    # Batch 0: 计算专家
    # Batch 1: 发送 tokens (与 Batch 0 的计算重叠)
    # Batch 0: 收集结果
    # Batch 1: 计算专家
    # ...
```

Pipeline MoE 的核心思想是借鉴 CPU 流水线的设计理念，将通信操作与计算操作放在不同的 CUDA Stream 上并行执行。当 Batch 1 正在进行 All-to-All 通信时，GPU 的计算单元可以同时处理 Batch 0 的专家 FFN 计算。这种重叠的关键前提是：专家计算（GEMM）主要消耗 Tensor Core 和 SM 的计算资源，而 All-to-All 通信主要消耗 NVLink/NVSwitch 的带宽，两者在不同的硬件单元上执行，天然适合流水线并行。但在实践中，实现真实的重叠并不简单——因为通信和计算共享 DRAM 带宽（用于读写 token 数据），流水线中的每个阶段必须使用不同的内存区域（例如 double-buffering），而且需要精确的 stream 同步来避免数据竞争。有效的流水线深度通常为 2-3 级，过深的流水线会因为同步开销和内存占用而收益递减。

---

## 12. 总结

### 关键要点

- **MoE 架构** 通过条件计算解耦模型参数量和计算量
- **Router/Expert/Combiner** 是 MoE 的三大核心组件
- **Grouped GEMM** 是 MoE 高效实现的关键技术
- **负载均衡** 是 MoE 系统的核心挑战
- **DeepSeek-V3** 的共享专家 + 路由专家设计是有效的解决方案
- **Expert Parallelism** 需要高效的 All-to-All 通信

### MoE 实现选择指南

```
专家数量？
├── 少 (2-8) → 简单循环实现
├── 中 (8-64) → Grouped GEMM
└── 多 (64+) → 分布式 Expert Parallelism
    └── 通信开销大？ → Pipeline MoE
```

---

前面章节覆盖了 MoE 的核心概念、算子实现和系统优化，下面的练习将帮助你将理论知识转化为实践能力。每个练习都围绕 MoE 实现中的一个关键组件展开，建议在动手编码之前先回顾对应章节的代码和分析。

## 13. 练习

### 练习 1：基础 Router 实现

实现一个支持 Top-1 和 Top-2 路由的 Router 模块。

### 练习 2：Grouped GEMM 优化

优化本章的 Grouped GEMM 实现，使其支持不均匀的 token 分布。

### 练习 3：负载均衡损失

实现 Load Balance Loss，并验证其对 token 分布的影响。

### 练习 4：Expert Choice 路由

实现 Expert Choice 路由策略，即每个专家主动选择 Top-K 个 token。

### 练习 5：MoE 反向传播

实现 MoE 层的反向传播，包括 Router 梯度和专家梯度的计算。

---

以下思考题旨在引导你超越代码实现，深入思考 MoE 系统的设计哲学和理论本质。这些问题没有标准答案，但它们对应着 MoE 研究前沿中最活跃的探索方向。

## 14. 思考题

1. **MoE 模型为什么在训练中可能出现"专家崩塌"问题？如何通过路由策略缓解？**

2. **Grouped GEMM 相比逐专家 GEMM 的性能优势在哪里？在什么情况下优势不明显？**

3. **DeepSeek-V3 的共享专家设计有什么理论依据？为什么它能提升模型质量？**

4. **在分布式 MoE 训练中，All-to-All 通信的瓶颈如何量化？有哪些优化手段？**

5. **如何设计一个自适应的 MoE 路由策略，根据输入动态调整专家选择？**

---

以下扩展阅读涵盖了 MoE 领域的重要学术工作和工业实践，建议按照从基础到前沿的顺序阅读。Switch Transformer 和 GShard 建立了 MoE 的理论基础，Mixtral 和 DeepSeek-V3 则展示了这些理论在商业级系统中的工程实现。

## 15. 扩展阅读

1. **Switch Transformer**：Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models" (JMLR 2022)
2. **Mixtral**：Jiang et al., "Mixtral of Experts" (2024)
3. **DeepSeek-V3**：DeepSeek-AI, "DeepSeek-V3 Technical Report" (2024)
4. **GShard**：Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation" (ICLR 2021)
5. **Expert Choice Routing**：Zhou et al., "Mixture-of-Experts with Expert Choice Routing" (NeurIPS 2022)

---

前向传播是 MoE 的"可见面"，而反向传播才是训练的核心。MoE 的反向传播面临独特的梯度计算挑战：Router 的梯度需要通过离散的 Top-K 选择传播（本质上不可微），专家的梯度需要处理稀疏的 token 分配，而辅助损失的梯度则需要与主任务损失进行平衡。理解这些梯度流的精确路径，是训练出稳定、均衡的 MoE 模型的前提。

## 16. 高级话题：MoE 的梯度计算

### 16.1 MoE 反向传播的挑战

MoE 的反向传播面临独特挑战：

1. **路由梯度**：Router 的梯度需要通过 Top-K 选择传播
2. **专家梯度**：每个专家只处理部分 token，梯度稀疏
3. **负载均衡梯度**：辅助损失的梯度需要特殊处理

### 16.2 Router 梯度计算

```python
def router_backward(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
):
    """Router backward pass."""

    @T.prim_func
    def router_backward_kernel(
        DY: T.Buffer((batch_size, seq_len, hidden_dim), "float32"),
        X: T.Buffer((batch_size, seq_len, hidden_dim), "float32"),
        W_gate: T.Buffer((num_experts, hidden_dim), "float32"),
        TopKIndices: T.Buffer((batch_size, seq_len, top_k), "int32"),
        TopKWeights: T.Buffer((batch_size, seq_len, top_k), "float32"),
        DX: T.Buffer((batch_size, seq_len, hidden_dim), "float32"),
        DW_gate: T.Buffer((num_experts, hidden_dim), "float32"),
    ):
        with T.Kernel(batch_size, seq_len, threads=256) as (b, s):
            # Compute gradient w.r.t. gate scores
            gate_scores = T.alloc_fragment((num_experts,), "float32")
            gate_grads = T.alloc_fragment((num_experts,), "float32")

            # Forward: compute gate scores
            for e in T.serial(num_experts):
                acc = T.float32(0)
                for d in T.serial(hidden_dim):
                    acc += X[b, s, d] * W_gate[e, d]
                gate_scores[e] = acc

            # Softmax
            max_val = T.float32(-1e30)
            for e in T.serial(num_experts):
                max_val = T.max(max_val, gate_scores[e])
            sum_val = T.float32(0)
            for e in T.serial(num_experts):
                gate_scores[e] = T.exp(gate_scores[e] - max_val)
                sum_val += gate_scores[e]
            for e in T.serial(num_experts):
                gate_scores[e] /= sum_val

            # Backward: compute gradient w.r.t. gate scores
            T.clear(gate_grads)
            for k in T.serial(top_k):
                e = TopKIndices[b, s, k]
                w = TopKWeights[b, s, k]
                # Gradient flows through weighted expert output
                for d in T.serial(hidden_dim):
                    gate_grads[e] += DY[b, s, d] * X[b, s, d]  # Simplified

            # Backward through softmax
            softmax_grad = T.alloc_fragment((num_experts,), "float32")
            for e in T.serial(num_experts):
                softmax_grad[e] = gate_scores[e] * (
                    gate_grads[e] - gate_scores[e] * gate_grads[e]
                )

            # Compute gradient w.r.t. W_gate
            for e in T.serial(num_experts):
                for d in T.serial(hidden_dim):
                    # Atomic add for gradient accumulation
                    T.atomic_add(DW_gate, (e, d), softmax_grad[e] * X[b, s, d])

            # Compute gradient w.r.t. X
            for d in T.serial(hidden_dim):
                acc = T.float32(0)
                for e in T.serial(num_experts):
                    acc += softmax_grad[e] * W_gate[e, d]
                DX[b, s, d] = acc

    return router_backward_kernel
```

Router 的反向传播需要处理一个核心矛盾：Top-K 选择本身是离散的、不可微的，但我们仍然需要梯度来更新路由器的参数。实际上，这段代码采用的是"直通估计器"（Straight-Through Estimator, STE）的思路：在反向传播中，我们假设计算图仅通过被选中的专家流动梯度（即 gate_grads 只累积选中专家的梯度），而 softmax 函数本身的梯度计算则和标准反向传播完全一致。这种做法的数学直觉是：虽然 Top-K 的 argmax 操作不可微，但我们将其视为恒等映射（identity）来传递梯度——"假装" gate 分数直接影响了最终输出。在实践中，这种近似是足够有效的，因为辅助负载均衡损失提供了额外的路由信号，使得 STE 的梯度偏差不会导致 Router 完全失效。此外，DW_gate 的更新使用了 `T.atomic_add` 而非直接写入，这是因为多个 token 可能同时更新同一个 weight 元素，原子操作保证了梯度累加的正确性。

### 16.3 专家梯度计算

```python
def expert_backward(
    num_tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
):
    """Expert backward pass."""

    @T.prim_func
    def expert_backward_kernel(
        DY: T.Buffer((num_tokens, hidden_dim), "float32"),
        X: T.Buffer((num_tokens, hidden_dim), "float32"),
        W_up: T.Buffer((num_experts, intermediate_dim, hidden_dim), "float32"),
        W_down: T.Buffer((num_experts, hidden_dim, intermediate_dim), "float32"),
        ExpertMap: T.Buffer((num_experts, num_tokens), "int32"),
        ExpertCount: T.Buffer((num_experts,), "int32"),
        DX: T.Buffer((num_tokens, hidden_dim), "float32"),
        DW_up: T.Buffer((num_experts, intermediate_dim, hidden_dim), "float32"),
        DW_down: T.Buffer((num_experts, hidden_dim, intermediate_dim), "float32"),
    ):
        with T.Kernel(num_experts, threads=256) as (e):
            num_assigned = ExpertCount[e]

            for t in T.serial(num_assigned):
                token_idx = ExpertMap[e, t]

                # Forward: hidden = W_up @ x
                hidden = T.alloc_fragment((intermediate_dim,), "float32")
                for i in T.serial(intermediate_dim):
                    acc = T.float32(0)
                    for d in T.serial(hidden_dim):
                        acc += X[token_idx, d] * W_up[e, i, d]
                    hidden[i] = T.gelu(acc)

                # Backward through down projection
                dhidden = T.alloc_fragment((intermediate_dim,), "float32")
                for i in T.serial(intermediate_dim):
                    acc = T.float32(0)
                    for d in T.serial(hidden_dim):
                        acc += DY[token_idx, d] * W_down[e, d, i]
                    dhidden[i] = acc * T.gelu_grad(hidden[i])

                # Backward through up projection
                for d in T.serial(hidden_dim):
                    acc = T.float32(0)
                    for i in T.serial(intermediate_dim):
                        acc += dhidden[i] * W_up[e, i, d]
                    DX[token_idx, d] += acc

                # Weight gradients
                for i, d in T.serial(intermediate_dim, hidden_dim):
                    T.atomic_add(DW_up, (e, i, d), dhidden[i] * X[token_idx, d])
                for d, i in T.serial(hidden_dim, intermediate_dim):
                    T.atomic_add(DW_down, (e, d, i), DY[token_idx, d] * hidden[i])

    return expert_backward_kernel
```

专家反向传播在 MoE 中具有独特的稀疏特性：每个专家只计算自己被分配到的 token 子集的梯度，而非全部 token。代码通过 `ExpertMap[e, t]` 获取专家 e 的第 t 个 token 索引，实现了精确的稀疏梯度计算。这种稀疏性带来了双重影响：积极的方面是，总反向传播计算量与 token 总数乘以 top_k 成正比（而非乘以总专家数），显著减少了计算开销；消极的方面是，每个专家的梯度更新基于的 token 样本量较小，梯度估计的方差更大，可能导致训练不稳定。在 GELU 激活函数的反向传播中，`T.gelu_grad(hidden[i])` 的使用是一个关键细节——它需要前向传播时计算出的 hidden 值作为输入，这就要求在实现中要么重新计算 hidden（recompute，节省内存但增加计算），要么在前向传播时就将 hidden 存储下来（节省计算但增加内存）。在内存受限的推理场景中，通常选择 recompute 策略以用计算换内存。

---

梯度计算是训练的核心，但当我们将 MoE 部署到多 GPU 环境时，通信开销会迅速成为比计算更紧迫的瓶颈。All-to-All 通信不仅在分布式训练中占据主导时间，还直接影响了推理的延迟和吞吐量。下面我们将深入探讨 All-to-All 通信的实现细节和优化策略。

## 17. 高级话题：MoE 的通信优化

### 17.1 All-to-All 通信的实现

```python
def all_to_all_dispatch(
    num_tokens: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
):
    """All-to-All communication for token dispatch."""

    @T.prim_func
    def a2a_dispatch_kernel(
        X: T.Buffer((num_tokens, hidden_dim), "float32"),
        TopKIndices: T.Buffer((num_tokens, top_k), "int32"),
        TopKWeights: T.Buffer((num_tokens, top_k), "float32"),
        SendBuffer: T.Buffer((num_tokens * top_k, hidden_dim + 2), "float32"),
        RecvBuffer: T.Buffer((num_tokens * top_k, hidden_dim + 2), "float32"),
    ):
        with T.Kernel(num_tokens, threads=128) as (token_idx):
            for k in T.serial(top_k):
                expert_idx = TopKIndices[token_idx, k]
                weight = TopKWeights[token_idx, k]
                send_offset = token_idx * top_k + k

                # Pack token data with metadata
                for d in T.serial(hidden_dim):
                    SendBuffer[send_offset, d] = X[token_idx, d]
                SendBuffer[send_offset, hidden_dim] = T.cast(token_idx, "float32")
                SendBuffer[send_offset, hidden_dim + 1] = weight

        # All-to-All communication happens here (implemented by NCCL)
        # ...

        with T.Kernel(num_tokens * top_k, threads=128) as (idx):
            # Unpack received data
            for d in T.serial(hidden_dim):
                # Store to appropriate expert input buffer
                pass

    return a2a_dispatch_kernel
```

All-to-All 通信是 MoE 分布式部署中最关键的系统级操作，其本质是将分散在各 GPU 上的 token 数据按照"token 所属专家所在 GPU"的规则进行重新排列。这段实现将 token 数据打包为连续的内存块（SendBuffer），在每个 token 数据后附加了原始索引（token_idx）和路由权重（weight）作为元数据，形成 `[hidden_dim + 2]` 的 payload。这种"数据+元数据"的打包策略是 MPI/NCCL 通信中的标准范式——通过将离散的、非连续的数据元素合并为一条连续的消息，可以避免多次小消息通信带来的延迟开销。在接收端，元数据用于将专家输出正确地还原回原始 token 位置，这在分布式环境中尤为重要，因为不同 GPU 接收到的 token 数量和顺序都是动态变化的。NCCL 底层的 All-to-All 实现通常采用两个阶段：第一阶段通过 scatter 式操作分发数据，第二阶段通过 gather 式操作汇聚结果，两个阶段的总通信量与 token 总数的平方根成正比。

### 17.2 通信与计算重叠

```python
def overlapped_moe():
    """MoE with overlapped communication and computation."""

    # Stream 1: Send tokens for batch i+1
    # Stream 2: Compute experts for batch i
    # Stream 3: Receive results for batch i-1

    # Implementation requires multi-stream CUDA
    # TileLang can express this with multiple kernels
```

通信与计算重叠是 MoE 性能优化的"最后一公里"。在理想的重叠流水线中，GPU 的计算单元永远不会因等待通信而空闲——当一个 batch 在进行 All-to-All 通信时，另一个 batch 的专家 GEMM 计算正在 Tensor Core 上执行。这种重叠的有效性取决于两个关键因素：通信量和计算量的比值（communication-to-computation ratio），以及 GPU 硬件上通信和计算资源之间的独立性。在实际的 A100/H100 GPU 上，NVLink 和 Tensor Core 之间确实具有较好的独立性，但共享内存（HBM）的带宽是所有操作共享的瓶颈——无论是通信（读取 token 数据进行打包）还是计算（读取专家权重和 token 特征），都需要访问 HBM。因此，重叠的实际收益通常在 10%-25% 之间，远低于理论上的 2 倍加速。

### 17.3 通信开销分析

| 配置 | Token 大小 | 通信量 | A100 带宽 | 通信时间 |
|------|-----------|--------|----------|---------|
| 8 experts, top-2 | 4KB | 256KB | 600 GB/s | 0.4 μs |
| 64 experts, top-8 | 4KB | 2MB | 600 GB/s | 3.3 μs |
| 256 experts, top-8 | 4KB | 8MB | 600 GB/s | 13.3 μs |
| 8 experts, top-2 | 16KB | 1MB | 600 GB/s | 1.7 μs |
| 64 experts, top-8 | 16KB | 8MB | 600 GB/s | 13.3 μs |

---

通信优化的前提是对负载的精确感知——如果我们不知道哪些专家过载、哪些专家饥饿，就无法做出有针对性的路由调整。实时负载监控为 MoE 系统提供了"自我感知"的能力，而自适应路由则将这种感知转化为自动的负载再均衡。

## 18. 高级话题：MoE 的负载均衡监控

### 18.1 实时负载监控

```python
def load_monitoring(
    num_experts: int,
    window_size: int = 1000,
):
    """Real-time load monitoring for MoE."""

    @T.prim_func
    def monitor_kernel(
        TokenCounts: T.Buffer((num_experts,), "int32"),
        LoadHistory: T.Buffer((window_size, num_experts), "int32"),
        Step: T.Buffer((1,), "int32"),
        LoadStats: T.Buffer((num_experts, 3), "float32"),  # mean, std, max
    ):
        step = Step[0] % window_size

        with T.Kernel(num_experts, threads=1) as (e):
            # Update history
            LoadHistory[step, e] = TokenCounts[e]

            # Compute statistics
            total = T.float32(0)
            max_val = T.int32(0)
            for i in T.serial(window_size):
                total += T.cast(LoadHistory[i, e], "float32")
                max_val = T.max(max_val, LoadHistory[i, e])

            mean = total / window_size

            variance = T.float32(0)
            for i in T.serial(window_size):
                diff = T.cast(LoadHistory[i, e], "float32") - mean
                variance += diff * diff
            variance /= window_size

            LoadStats[e, 0] = mean
            LoadStats[e, 1] = T.sqrt(variance)
            LoadStats[e, 2] = T.cast(max_val, "float32")

    return monitor_kernel
```

负载监控模块维护了一个滑动窗口（window_size=1000）的历史记录，为每个专家计算三个关键统计量：均值（mean）反映长期负载水平、标准差（std）反映负载波动性、最大值（max）反映负载峰值。这种统计方法借鉴了时间序列分析中的移动平均思想，能够平滑短期的随机波动，捕捉长期的负载趋势。窗口大小的选择是一个重要的超参数：过小的窗口（如 100）对瞬时的路由变化过于敏感，可能导致路由策略频繁振荡；过大的窗口（如 10000）则对负载变化的响应太迟钝。在实践中，窗口大小通常设置为一个训练 step 的 token 数的 10-50 倍，以保证统计量的稳定性。此外，LoadHistory 使用了环形缓冲区（通过 `Step[0] % window_size` 实现），避免了动态内存分配的开销，这在 GPU kernel 中是一个重要的性能优化。

### 18.2 自适应路由调整

```python
def adaptive_routing(
    num_experts: int,
    alpha: float = 0.1,
):
    """Adaptive routing based on load statistics."""

    @T.prim_func
    def adaptive_kernel(
        GateScores: T.Buffer((num_experts,), "float32"),
        LoadStats: T.Buffer((num_experts, 3), "float32"),
        AdjustedScores: T.Buffer((num_experts,), "float32"),
    ):
        with T.Kernel(1, threads=num_experts) as ():
            e = T.thread_id()
            if e < num_experts:
                mean_load = LoadStats[e, 0]
                std_load = LoadStats[e, 1]

                # Penalize overloaded experts
                penalty = alpha * (mean_load - 1.0 / num_experts)
                AdjustedScores[e] = GateScores[e] - penalty

    return adaptive_kernel
```

自适应路由的核心是对路由分数施加与负载成比例的惩罚项：`penalty = alpha * (mean_load - 1/num_experts)`。当某专家的历史负载均值超过均匀分布期望值（1/num_experts）时，其当前的路由分数会被下调，从而降低未来被选中的概率。这种负反馈机制使得负载分布具有自我调节的能力。alpha 参数控制反馈强度——过大会导致路由完全由负载历史决定，丧失基于内容的专家选择能力；过小则不足以抑制负载漂移。一个更完善的实现可以在惩罚项中引入标准差（std_load），对负载波动大的专家施加更大的不确定性惩罚，从而鼓励模型选择那些负载稳定且内容适配度高的专家。这实质上是将 MoE 的路由问题转化为一个带约束的在线分配问题（online assignment with capacity constraints），每次路由决策都需要在"内容匹配度"和"负载均衡度"之间寻找帕累托最优。

---

## 19. MoE 模型的训练策略

### 19.1 训练中的负载均衡损失

```python
def compute_balance_loss(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    alpha: float = 0.01,
):
    """Compute load balancing auxiliary loss."""

    @T.prim_func
    def balance_loss_kernel(
        GateScores: T.Buffer((num_tokens, num_experts), "float32"),
        TopKIndices: T.Buffer((num_tokens, top_k), "int32"),
        Loss: T.Buffer((1,), "float32"),
    ):
        with T.Kernel(1, threads=256) as ():
            # Compute f_i: fraction of tokens routed to expert i
            expert_counts = T.alloc_fragment((num_experts,), "float32")
            T.clear(expert_counts)

            for t in T.serial(T.ceildiv(num_tokens, 256)):
                idx = t * 256 + T.thread_id()
                if idx < num_tokens:
                    for k in T.serial(top_k):
                        e = TopKIndices[idx, k]
                        expert_counts[e] += T.float32(1)

            # Reduce counts
            for offset in T.serial(5):
                for e in T.serial(num_experts):
                    expert_counts[e] += T.warp_shuffle_down(expert_counts[e], 1 << offset)

            # Compute P_i: average gate probability
            expert_probs = T.alloc_fragment((num_experts,), "float32")
            T.clear(expert_probs)

            for t in T.serial(T.ceildiv(num_tokens, 256)):
                idx = t * 256 + T.thread_id()
                if idx < num_tokens:
                    for e in T.serial(num_experts):
                        expert_probs[e] += GateScores[idx, e]

            # Reduce probs
            for offset in T.serial(5):
                for e in T.serial(num_experts):
                    expert_probs[e] += T.warp_shuffle_down(expert_probs[e], 1 << offset)

            # Compute loss = alpha * E * sum(f_i * P_i)
            if T.thread_id() == 0:
                loss = T.float32(0)
                for e in T.serial(num_experts):
                    f_i = expert_counts[e] / (num_tokens * top_k)
                    P_i = expert_probs[e] / num_tokens
                    loss += f_i * P_i
                Loss[0] = alpha * num_experts * loss

    return balance_loss_kernel
```

这个负载均衡损失的实现使用了 warp shuffle 进行高效的线程块内归约。`T.warp_shuffle_down` 是一种寄存器级别的数据交换操作，它允许同一个 warp 内的 32 个线程在不经过共享内存的情况下直接交换数据——每次 shuffle 将数据向低线程号方向移动 `offset` 个位置，从而实现 log2(32)=5 次迭代的蝶形归约。这种实现比使用共享内存的归约快约 2-3 倍，因为避免了共享内存的写入-同步-读取的完整周期。损失函数的计算中，$f_i$ 使用了实际的 Top-K 分配（硬计数），$P_i$ 使用了 softmax 之前的门控概率（连续值），这种组合策略在前面已有讨论。值得注意的是，负载均衡损失以 $\alpha \cdot E$ 为系数乘入最终损失——当专家数量 E 很大时（例如 256），这个系数会自动放大，使得大规模 MoE 中的负载不均衡受到更强的惩罚，这是一个巧妙的自适应缩放机制。

---

训练策略决定了 MoE 模型的"先天条件"，但模型最终的价值体现在推理部署中的表现。MoE 模型的参数量通常比同等计算量的 Dense 模型大一个数量级，这意味着内存需求呈指数级增长。如何在有限的 GPU 显存中运行大规模的 MoE 模型，是推理部署中最现实的工程问题。

## 20. MoE 的实际部署考虑

### 20.1 内存需求分析

| 模型 | 专家数 | 参数量 | FP16 内存 | 推理内存 |
|------|--------|--------|----------|---------|
| Mixtral 8x7B | 8 | 46.7B | 93 GB | ~50 GB |
| DeepSeek-V3 | 256 | 671B | 1.3 TB | ~300 GB |
| Switch-Base | 128 | 7.4B | 15 GB | ~8 GB |
| GShard | 2048 | 600B | 1.2 TB | ~400 GB |

### 20.2 推理优化策略

| 策略 | 说明 | 效果 |
|------|------|------|
| 专家缓存 | 只加载活跃专家到 GPU | 减少 50-80% 内存 |
| 专家量化 | INT8/INT4 量化专家权重 | 减少 50-75% 内存 |
| 预测路由 | 预测未来 token 的路由 | 减少延迟 |
| 批量路由 | 批量处理多个请求 | 提高吞吐 |

### 20.3 专家缓存实现

```python
class ExpertCache:
    """Cache for expert weights."""

    def __init__(self, num_experts, cache_size, hidden_dim, intermediate_dim):
        self.num_experts = num_experts
        self.cache_size = cache_size
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim

        # GPU cache
        self.cached_experts = {}
        self.cache_queue = []

    def get_expert(self, expert_idx):
        """Get expert, loading to GPU if needed."""
        if expert_idx in self.cached_experts:
            return self.cached_experts[expert_idx]

        # Evict least recently used expert
        if len(self.cached_experts) >= self.cache_size:
            lru = self.cache_queue.pop(0)
            del self.cached_experts[lru]

        # Load expert to GPU
        expert_weights = self.load_expert_from_cpu(expert_idx)
        self.cached_experts[expert_idx] = expert_weights
        self.cache_queue.append(expert_idx)

        return expert_weights
```

专家缓存是 MoE 推理优化的核心策略，利用的是"时间局部性"原理：相邻的 token（尤其是在自回归生成中）往往选择相似甚至相同的专家集合。LRU（Least Recently Used）淘汰策略是最简单也最有效的缓存替换算法之一——它假设最近被访问的专家在近期最可能被再次访问。在 DeepSeek-V3 的 256 专家部署中，如果每个 GPU 仅缓存 16-32 个专家（而非全部 256 个），显存占用可以减少 87.5%-93.75%。缓存命中的关键在于路由的"粘性"——如果路由器的输出分布变化剧烈，缓存命中率会急剧下降。因此，有效的专家缓存通常与"路由正则化"配合使用，在训练阶段就鼓励相邻 token 路由到相似的专家集合。此外，缓存的加载延迟（从 CPU 内存或 NVMe 加载专家权重到 GPU 显存）通常在毫秒级别，对于延迟敏感的在线推理服务，可以结合"预测路由"提前预取未来可能需要的专家，将加载延迟隐藏在计算时间中。

---

掌握了训练和部署的核心技术后，让我们将视野扩展到 MoE 领域的未来发展方向。当前的 MoE 架构仍处于快速演进中，从更细粒度的专家设计到动态调整的专家数量，从多层次的路由结构到稀疏激活策略，每一项改进都有可能重新定义大模型的扩展范式。

## 21. MoE 的未来发展方向

### 21.1 技术趋势

| 趋势 | 说明 | 影响 |
|------|------|------|
| 更多更小的专家 | 细粒度专家设计 | 提高专业化程度 |
| 共享专家 | 所有 token 共享的专家 | 提高通用知识 |
| 动态专家数 | 根据输入复杂度调整 | 提高效率 |
| 层次化 MoE | 多级路由结构 | 提高扩展性 |
| 稀疏激活 | 只激活部分专家参数 | 减少计算 |

### 21.2 研究方向

1. **更好的路由策略**：如何更准确地分配 token 到专家
2. **负载均衡**：如何在不影响模型质量的情况下实现均衡
3. **专家并行**：如何高效地在多设备上分布专家
4. **内存优化**：如何减少 MoE 的内存占用
5. **训练稳定性**：如何稳定 MoE 模型的训练

---

## 22. 总结（扩展）

### MoE 实现方法全对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| 循环遍历 | 简单直观 | 低并行度 | 少量专家 |
| 批量 GEMM | 高并行度 | 内存开销大 | 中等专家数 |
| Grouped GEMM | 平衡性能 | 实现复杂 | 大量专家 |
| Expert Parallel | 可扩展 | 通信开销 | 分布式训练 |
| Pipeline MoE | 重叠通信 | 实现复杂 | 多节点 |

### 关键经验

1. **路由策略是核心**：好的路由可以提高模型质量和系统效率
2. **负载均衡不可忽视**：不均衡会导致性能瓶颈
3. **Grouped GEMM 是关键**：高效实现的基础
4. **通信优化很重要**：All-to-All 通信是瓶颈
5. **内存管理是挑战**：需要专家缓存等技术

---

前面章节建立了 MoE 的基础理论和核心实现，但路由算法（尤其是非 Token Choice 的策略）在学术文献中有丰富的变体。本节将深入分析 Hash Routing、Expert Choice Routing 和 Soft MoE 这三种重要的替代路由方案，它们的共同特点是：不依赖"token 选专家"的范式，而是从完全不同的角度解决路由问题。

## 24. MoE 路由算法深度分析

### 24.1 Hash Routing 实现

Hash Routing 使用哈希函数将 token 分配到专家，简单高效但不考虑 token 内容：

```python
def hash_routing(
    num_tokens: int,
    num_experts: int,
    top_k: int,
):
    """Hash-based routing for MoE."""

    @T.prim_func
    def hash_router_kernel(
        X: T.Buffer((num_tokens, hidden_dim), "float32"),
        TopKIndices: T.Buffer((num_tokens, top_k), "int32"),
        TopKWeights: T.Buffer((num_tokens, top_k), "float32"),
    ):
        with T.Kernel(num_tokens, threads=1) as (token_idx):
            # Simple hash based on token index
            for k in T.serial(top_k):
                expert_idx = (token_idx * 2654435761 + k * 40503) % num_experts
                TopKIndices[token_idx, k] = expert_idx
                TopKWeights[token_idx, k] = T.float32(1.0) / top_k

    return hash_router_kernel
```

Hash Routing 的核心公式 `expert_idx = (token_idx * 2654435761 + k * 40503) % num_experts` 使用了经典的 Knuth 乘法哈希：乘数 2654435761（即黄金比例的倒数 1/φ × 2^32）具有最优的雪崩效应，能够将连续的 token 索引均匀地分散到专家空间中。这种路由方式的突出优势在于零计算开销（不需要矩阵乘法、不需要 softmax），且天然实现了完美的负载均衡（每个专家恰好分配 num_tokens/num_experts 个 token）。但其致命缺陷是完全忽略了 token 的语义内容——无论 token 是"量子力学"还是"今天天气"，只要位置索引相同就被分配到同一专家。因此 Hash Routing 通常只用于两种场景：一是作为与其他路由策略的消融实验基准，验证"基于内容的路由"是否真的在提升模型质量；二是在极大规模分布式训练中，作为初始的均匀分配策略，后续再通过微调路由参数进行精细化调整。

### 24.2 Expert Choice Routing

Expert Choice 路由让每个专家主动选择 Top-K 个 token，天然实现负载均衡：

```python
def expert_choice_routing(
    num_tokens: int,
    num_experts: int,
    capacity_factor: float = 1.0,
):
    """Expert Choice routing: each expert selects top-k tokens."""

    @T.prim_func
    def expert_choice_kernel(
        GateScores: T.Buffer((num_tokens, num_experts), "float32"),
        ExpertSelectedTokens: T.Buffer((num_experts, int(num_tokens * 1.0)), "int32"),
        ExpertSelectedWeights: T.Buffer((num_experts, int(num_tokens * 1.0)), "float32"),
        ExpertSelectedCount: T.Buffer((num_experts,), "int32"),
    ):
        tokens_per_expert = T.cast(T.float32(num_tokens) / num_experts * capacity_factor, "int32")

        with T.Kernel(num_experts, threads=256) as (expert_idx):
            # Each expert selects top tokens_per_expert tokens
            selected = T.alloc_fragment((1,), "int32")
            selected[0] = 0

            # Find top tokens for this expert
            for t in T.serial(num_tokens):
                if selected[0] < tokens_per_expert:
                    # Check if this token is in top-k for this expert
                    score = GateScores[t, expert_idx]
                    # Simple threshold-based selection
                    if score > T.float32(0.1):  # Threshold
                        ExpertSelectedTokens[expert_idx, selected[0]] = t
                        ExpertSelectedWeights[expert_idx, selected[0]] = score
                        selected[0] += 1

            ExpertSelectedCount[expert_idx] = selected[0]

    return expert_choice_kernel
```

Expert Choice Routing 彻底翻转了传统 MoE 的"token 选专家"范式，改为"专家选 token"。这种转换在数学上等价于对 GateScores 矩阵进行转置后再做 Top-K 选择，但带来的系统特性截然不同。最重要的是负载均衡的天然性：每个专家处理的 token 数量被精确控制为 `tokens_per_expert = num_tokens / num_experts * capacity_factor`，因此所有专家的计算负载完美均匀。但这种路由方式也有代价：不能保证每个 token 都被至少一个专家选中——如果某个 token 对所有专家的 GateScores 都低于阈值（代码中的 0.1），它就会被所有专家"忽略"，等价于被丢弃。这种 token 遗漏问题在模型训练的早期尤为严重，因为此时 GateScores 尚未学习到有意义的分布。NeurIPS 2022 的 Expert Choice 论文提出的解决方案是：先让专家选择 token，再检查是否有遗漏的 token，如果有则强制分配到负载最低的专家，这种"专家选择 + token 兜底"的混合策略在实践中效果最好。

### 24.3 Soft MoE 实现

Soft MoE 使用软权重，所有专家都参与计算，避免了离散路由的问题：

```python
def soft_moe_forward(
    num_tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
):
    """Soft MoE: all experts contribute with soft weights."""

    @T.prim_func
    def soft_moe_kernel(
        X: T.Buffer((num_tokens, hidden_dim), "float32"),
        W_gate: T.Buffer((num_experts, hidden_dim), "float32"),
        W_up: T.Buffer((num_experts, intermediate_dim, hidden_dim), "float32"),
        W_down: T.Buffer((num_experts, hidden_dim, intermediate_dim), "float32"),
        Y: T.Buffer((num_tokens, hidden_dim), "float32"),
    ):
        with T.Kernel(num_tokens, threads=256) as (token_idx):
            # Compute soft gate scores (softmax over all experts)
            gate_scores = T.alloc_fragment((num_experts,), "float32")
            max_val = T.float32(-1e30)

            for e in T.serial(num_experts):
                acc = T.float32(0)
                for d in T.serial(hidden_dim):
                    acc += X[token_idx, d] * W_gate[e, d]
                gate_scores[e] = acc
                max_val = T.max(max_val, acc)

            sum_val = T.float32(0)
            for e in T.serial(num_experts):
                gate_scores[e] = T.exp(gate_scores[e] - max_val)
                sum_val += gate_scores[e]

            for e in T.serial(num_experts):
                gate_scores[e] /= sum_val

            # Weighted sum of all expert outputs
            Y_local = T.alloc_fragment((hidden_dim,), "float32")
            T.clear(Y_local)

            for e in T.serial(num_experts):
                hidden = T.alloc_fragment((intermediate_dim,), "float32")
                for i in T.serial(intermediate_dim):
                    acc = T.float32(0)
                    for d in T.serial(hidden_dim):
                        acc += X[token_idx, d] * W_up[e, i, d]
                    hidden[i] = T.gelu(acc)

                for d in T.serial(hidden_dim):
                    acc = T.float32(0)
                    for i in T.serial(intermediate_dim):
                        acc += hidden[i] * W_down[e, d, i]
                    Y_local[d] += gate_scores[e] * acc

            for d in T.serial(hidden_dim):
                Y[token_idx, d] = Y_local[d]

    return soft_moe_kernel
```

Soft MoE 回归到了 MoE 最原始的数学定义——所有专家都参与计算，输出的加权混合完全由 softmax 权重决定。这种方式取消了离散的 Top-K 选择，具有完美可微性和训练稳定性。输出公式 $y = \sum_{e} \text{softmax}_e(W_g \cdot x) \cdot \text{FFN}_e(x)$ 恰好是标准的 attention 形式，只不过将 attention 的 key/value 替换为了专家输出。但从计算量角度看，Soft MoE 需要每个 token 都经过所有 E 个专家的 FFN，计算复杂度为 O(E × FFN_cost)，这与稀疏 MoE 的 O(K × FFN_cost) 形成鲜明对比。因此 Soft MoE 的计算效率劣势随专家数量的增加呈线性放大——当 E=256、K=8 时，Soft MoE 的计算量是稀疏 MoE 的 32 倍。缓解这个劣势的路径正在被学术界探索，包括：使用低秩分解近似所有专家的输出（类似 LoRA 的思想），或将 Soft MoE 的计算转换为一个大型 GEMM（将 E 维度和 batch 维度合并），利用 Tensor Core 的高吞吐来补偿计算量的增加。

### 24.4 路由算法对比

| 算法 | 负载均衡 | 计算效率 | 模型质量 | 实现复杂度 |
|------|---------|---------|---------|-----------|
| Token Choice | 需要辅助损失 | 高 | 好 | 低 |
| Expert Choice | 天然均衡 | 高 | 中 | 中 |
| Hash Routing | 完美均衡 | 最高 | 差 | 最低 |
| Soft MoE | 完美均衡 | 低 | 最好 | 中 |

---

不同的路由算法在模型质量、计算效率和负载均衡之间形成了不同的权衡点，但它们最终都需要落到多 GPU 硬件上执行。Expert Parallelism 作为 MoE 分布式训练的核心策略，其本质是将"专家"视为并行化的基本单元——这既是 MoE 区别于传统数据并行的独特之处，也是其系统复杂性的主要来源。

## 25. Expert Parallelism 策略详解

### 25.1 Tensor Parallelism vs Expert Parallelism

```python
"""
MoE 模型的并行策略：

1. Tensor Parallelism (TP):
   - 每个专家的权重矩阵被分割到多个 GPU
   - 适合单个专家很大的场景
   - 通信：AllReduce

2. Expert Parallelism (EP):
   - 不同专家分布在不同 GPU
   - 适合专家数很多的场景
   - 通信：All-to-All

3. Expert + Tensor Parallelism (EP+TP):
   - 专家分布在多个 GPU（EP）
   - 每个专家的权重也被分割（TP）
   - 适合超大模型
   - 通信：All-to-All + AllReduce
"""
```

Tensor Parallelism 和 Expert Parallelism 代表了两种互补的并行维度：TP 在权重矩阵内部进行切分（例如将 intermediate_dim 维度分割到不同 GPU），适合单个专家权重过大无法放入单 GPU 显存的场景；EP 在专家之间进行切分，适合专家数量众多但单个专家适中的场景。两者的通信模式有本质差异：TP 的 AllReduce 是带宽受限型操作——每步都需要将各 GPU 的计算结果进行求和，通信量与输出维度成正比；而 EP 的 All-to-All 是延迟敏感型操作——每个 GPU 需要与所有其他 GPU 交换 token 数据。在实际的混合并行（EP+TP）部署中，通常的配置是：EP 在节点间（跨 NVSwitch 域），TP 在节点内（同一 NVSwitch 域内的 GPU 之间），这样可以最大化利用节点内的高带宽 NVLink 进行 TP 通信，而将带宽较低的跨节点通信留给 EP 的 All-to-All。这种"外 EP 内 TP"的拓扑结构已经成为当前 MoE 大型集群训练的标准范式。

### 25.2 All-to-All 通信实现

```python
def all_to_all_communication(
    num_tokens: int,
    hidden_dim: int,
    num_experts: int,
    num_gpus: int,
):
    """All-to-All communication for MoE."""

    @T.prim_func
    def a2a_kernel(
        LocalTokens: T.Buffer((num_tokens, hidden_dim), "float32"),
        ExpertAssignment: T.Buffer((num_tokens,), "int32"),
        SendBuffer: T.Buffer((num_tokens, hidden_dim), "float32"),
        RecvBuffer: T.Buffer((num_tokens, hidden_dim), "float32"),
    ):
        experts_per_gpu = num_experts // num_gpus

        # Step 1: Pack tokens by destination GPU
        with T.Kernel(num_tokens, threads=128) as (token_idx):
            expert_id = ExpertAssignment[token_idx]
            dest_gpu = expert_id // experts_per_gpu
            # Store token in send buffer with destination info
            for d in T.serial(hidden_dim):
                SendBuffer[token_idx, d] = LocalTokens[token_idx, d]

        # Step 2: All-to-All communication (NCCL)
        # This is handled by NCCL library, not shown in TileLang

        # Step 3: Unpack received tokens
        with T.Kernel(num_tokens, threads=128) as (token_idx):
            for d in T.serial(hidden_dim):
                LocalTokens[token_idx, d] = RecvBuffer[token_idx, d]

    return a2a_kernel
```

All-to-All 通信的实现将 token 打包阶段和 NCCL 通信阶段分离，这种分离设计使得 TileLang 可以专注于表达计算逻辑，将通信委托给成熟的 NCCL 库处理。在打包阶段，`dest_gpu = expert_id // experts_per_gpu` 将专家 ID 映射为目标 GPU ID——这是一个简单的整除映射，假设专家按顺序均匀分布在 GPU 上。更复杂的布局策略（例如根据通信拓扑优化专家placement）可以通过修改这个映射函数来实现，而无需改动通信逻辑。值得注意的是 SendBuffer 的布局：在这个简化实现中，SendBuffer 的大小为 `[num_tokens, hidden_dim]`，假设最坏情况下所有 token 都需要被发送（all-to-all 的极端情况）。在实际优化中，可以使用稀疏表示——只打包实际需要跨 GPU 传输的 token，这需要额外的 token 计数和偏移量信息来支持 NCCL 的变长 All-to-All（vAllToAll）接口。

### 25.3 通信量分析

| 配置 | Token 数 | Hidden Dim | 每 GPU 通信量 | 通信时间 (A100 NVLink) |
|------|---------|------------|-------------|----------------------|
| 8 experts, 2 GPUs | 1024 | 4096 | 8 MB | 0.01 ms |
| 64 experts, 8 GPUs | 4096 | 4096 | 32 MB | 0.05 ms |
| 256 experts, 32 GPUs | 8192 | 7168 | 56 MB | 0.08 ms |

---

All-to-All 通信为 Expert Parallelism 提供了数据传输基础设施，而 Grouped GEMM 则为专家计算提供了计算加速引擎。但标准的 Grouped GEMM 在处理不均匀 token 分布时仍有性能损失——某些专家的 token 数量远多于分组 tile 的整数倍时，大量计算槽被零填充占用。动态 Grouped GEMM 旨在解决这一"长尾问题"。

## 26. Grouped GEMM 的进阶优化

### 26.1 动态 Grouped GEMM

```python
def dynamic_grouped_gemm(
    num_experts: int,
    max_tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
):
    """Dynamic grouped GEMM with variable token counts."""

    @T.prim_func
    def dyn_grouped_gemm_kernel(
        X: T.Buffer((max_tokens, hidden_dim), "float32"),
        W: T.Buffer((num_experts, intermediate_dim, hidden_dim), "float32"),
        ExpertStart: T.Buffer((num_experts,), "int32"),
        ExpertCount: T.Buffer((num_experts,), "int32"),
        Y: T.Buffer((max_tokens, intermediate_dim), "float32"),
    ):
        # Process each expert independently
        with T.Kernel(num_experts, T.ceildiv(max_tokens, 64), threads=256) as (e, tile):
            start = ExpertStart[e]
            count = ExpertCount[e]
            tile_start = tile * 64

            if tile_start < count:
                Y_local = T.alloc_fragment((64, 64), "float32")
                T.clear(Y_local)

                for k_tile in T.serial(T.ceildiv(hidden_dim, 32)):
                    X_local = T.alloc_fragment((64, 32), "float32")
                    W_local = T.alloc_fragment((32, 64), "float32")

                    # Load tokens for this expert
                    for i in T.serial(64):
                        token_idx = start + tile_start + i
                        if tile_start + i < count and token_idx < max_tokens:
                            for j in T.serial(32):
                                k_idx = k_tile * 32 + j
                                if k_idx < hidden_dim:
                                    X_local[i, j] = X[token_idx, k_idx]
                                else:
                                    X_local[i, j] = T.float32(0)
                        else:
                            for j in T.serial(32):
                                X_local[i, j] = T.float32(0)

                    # Load weights
                    for i, j in T.serial(32, 64):
                        k_idx = k_tile * 32 + i
                        if k_idx < hidden_dim:
                            W_local[i, j] = W[e, j, k_idx]
                        else:
                            W_local[i, j] = T.float32(0)

                    T.gemm(X_local, W_local, Y_local)

                # Store results
                for i in T.serial(64):
                    token_idx = start + tile_start + i
                    if tile_start + i < count and token_idx < max_tokens:
                        for j in T.serial(64):
                            Y[token_idx, j] = Y_local[i, j]

    return dyn_grouped_gemm_kernel
```

动态 Grouped GEMM 通过 ExpertStart 和 ExpertCount 两个辅助数组来精确描述每个专家的 token 区间，避免了固定大小的专家数组中的空间浪费。在实现中，每个 tile 在处理 token 之前先检查 `tile_start < count`，这消除了对已经处理完所有 token 的专家继续执行无效 tile 的开销。X_local 的数据加载采用了双层条件判断：外层判断 token 是否在有效范围内，内层判断 K 维度是否在 hidden_dim 范围内——后者确保了边界 tile 的正确零填充。这种方式虽然增加了分支逻辑，但由于 GPU 分支发散主要发生在 warp 内部（32 个线程之间的分支），而这里的条件判断基于 tile 级别的参数（tile_start 和 count），同一 warp 内的线程通常沿着相同的分支路径执行，因此分支发散的实际开销很小。动态 Grouped GEMM 在不均匀 token 分布下的性能优势尤为明显——当 max_tokens/token_per_expert 的比值超过 2 时（即 token 分布极度不均），相比固定大小的 Grouped GEMM 可以减少约 40%-60% 的无效计算。

### 26.2 Grouped GEMM 的性能优化技巧

| 技巧 | 说明 | 效果 |
|------|------|------|
| Token Padding | 将不均匀的 token 数量 pad 到相同 | 简化实现，但浪费计算 |
| Bucket Sort | 按 token 数量分桶 | 减少 load imbalance |
| Stream-K | 流式处理多个专家 | 提高 GPU 利用率 |
| Fusion | 融合 GELU 等激活函数 | 减少内存访问 |

---

Grouped GEMM 解决了计算效率问题，但在多节点 MoE 部署中，通信开销往往比计算更值得关注。前面的 All-to-All 分析表明，通信可能是 MoE 分布式部署的主要瓶颈。下面我们将探讨如何通过流水线化和通信压缩等高级技术来进一步优化 MoE 的通信效率。

## 27. MoE 部署的通信优化

### 27.1 通信与计算重叠策略

```python
def overlapped_moe_pipeline(
    batch_size: int,
    num_experts: int,
    hidden_dim: int,
):
    """MoE pipeline with overlapped communication."""

    @T.prim_func
    def overlapped_kernel(
        X: T.Buffer((batch_size, hidden_dim), "float32"),
        Y: T.Buffer((batch_size, hidden_dim), "float32"),
    ):
        # Stage 1: Compute router scores (local)
        # Stage 2: All-to-All dispatch (communication)
        # Stage 3: Expert computation (compute)
        # Stage 4: All-to-All combine (communication)

        # Overlap: Stage 2 of batch i+1 with Stage 3 of batch i
        # This requires multi-stream execution

        with T.Kernel(1, threads=1) as ():
            # Placeholder for pipeline logic
            pass

    return overlapped_kernel
```

通信与计算重叠的实现关键在于多 CUDA Stream 的编排。在 MoE 流水线中，至少需要两个独立的 Stream：通信流（用于 All-to-All）和计算流（用于专家 GEMM）。在 NVIDIA GPU 上，NVLink 通信引擎和计算 SM 是独立的硬件单元，因此理论上可以完全并行。但在实践中，重叠的效率受限于三个因素：第一，通信和计算共享 DRAM 带宽控制器——如果两个流同时访问 HBM，会产生带宽竞争，降低各自的吞吐；第二，NCCL 的 All-to-All 实现通常会产生多个 kernel launch（打包、通信、解包），这些 kernel 本身也占用 SM 资源；第三，CUDA Stream 之间的同步需要精心设计——不正确的同步会导致 deadlock 或数据竞争。有效的重叠通常能带来 15%-25% 的端到端延迟改善，在通信时间占比较高的配置中（如 256 专家、32 GPU）收益最为显著。

### 27.2 通信量优化技术

| 技术 | 说明 | 效果 |
|------|------|------|
| Token Compression | 压缩传输的 token 数据 | 减少 50% 通信量 |
| Top-K Pruning | 只传输 top-k 个专家的结果 | 减少通信量 |
| Hierarchical All-to-All | 节点内 + 节点间分层通信 | 减少跨节点通信 |
| Async Communication | 异步通信与计算重叠 | 隐藏通信延迟 |

---

通信优化是 MoE 部署的"最后一公里"，而本章的练习则将引导你亲手实践这些优化技术。从简单的 Hash Routing 对比到复杂的 Grouped GEMM 优化，每个练习都对应着 MoE 系统工程中的一个关键决策点。

## 28. 练习（扩展）

### 练习 6：Hash Routing 实现

实现 Hash Routing 路由策略，并与 Token Choice 路由进行性能对比。

### 练习 7：Expert Choice 路由

实现 Expert Choice 路由策略，验证其天然负载均衡的特性。

### 练习 8：Soft MoE

实现 Soft MoE 前向传播，对比其与稀疏 MoE 的计算量和模型质量。

### 练习 9：All-to-All 通信分析

分析不同规模 MoE 模型的 All-to-All 通信量，绘制通信时间与专家数的关系图。

### 练习 10：Grouped GEMM 优化

优化 Grouped GEMM 实现，使其在不均匀 token 分布下仍能高效运行。

---

以下扩展思考题将你的思维从工程实现推向系统设计的更高层面。MoE 的研究不仅关于"如何实现"，更关于"为什么这样设计"和"还有其他可能吗"。这些问题中的一些目前仍是开放研究课题，你的思考可能指向有价值的研究方向。

## 29. 思考题（扩展）

6. **Expert Choice 路由如何保证每个 token 至少被一个专家选中？如果不保证会有什么问题？**

7. **Soft MoE 为什么在训练中比稀疏 MoE 更稳定？它在推理中的计算效率劣势如何缓解？**

8. **在多节点 MoE 部署中，如何设计分层 All-to-All 通信以减少跨节点流量？**

9. **Grouped GEMM 中的 Stream-K 调度策略是如何工作的？它如何提高 GPU 利用率？**

10. **如何设计一个自适应的 MoE 系统，根据输入的复杂度动态调整激活的专家数量？**

---

## 30. 下一章预告

> **Chapter 28: TileLang vs Triton 深度对比**
>
> 下一章将深入对比 TileLang 和 Triton 两种 GPU 编程框架的设计哲学、编程模型、性能表现和适用场景，帮助读者在实际项目中做出明智的技术选择。
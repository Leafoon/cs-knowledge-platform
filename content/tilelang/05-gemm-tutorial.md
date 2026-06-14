---
title: "Chapter 5: GEMM 实战——从朴素到工业级"
description: "从零开始实现高性能 GEMM 算子，涵盖 Tiling、Shared Memory、Software Pipelining、Warp 级优化与自动调优，深入剖析 DeepSeek-V3 核心 GEMM 算子实现精髓"
updated: 2025-06-10
---

# Chapter 5: GEMM 实战——从朴素到工业级

> **Learning Objectives**
>
> - 理解 GEMM（General Matrix Multiplication）的计算密集型与访存密集型特征
> - 掌握从朴素实现到工业级优化的完整演进路径
> - 学会使用 `T.alloc_shared` 和 `T.alloc_fragment` 管理片上存储
> - 理解 Bank Conflict 的成因及消除策略
> - 掌握 Software Pipelining 在 GEMM 中的应用
> - 了解自动调优（Auto-tuning）框架的设计与实现
> - 深入理解 DeepSeek-V3 核心 GEMM 算子的实现精髓
> - 能够独立编写、调试和优化 TileLang GEMM 算子

---

## 5.1 GEMM 问题定义与性能模型

### 5.1.1 数学定义

GEMM（General Matrix Multiplication）是线性代数中最基础、最重要的运算之一。几乎所有深度学习的前向和反向传播都可以归结为 GEMM 操作。给定三个矩阵：

$$
C = \alpha \cdot A \times B + \beta \cdot C
$$

其中：
- $A \in \mathbb{R}^{M \times K}$：左矩阵（在神经网络中通常是输入激活值或权重的转置）
- $B \in \mathbb{R}^{K \times N}$：右矩阵（通常是权重矩阵）
- $C \in \mathbb{R}^{M \times N}$：输出矩阵（计算结果）
- $\alpha, \beta$：标量系数（用于缩放和融合操作）

当 $\alpha = 1, \beta = 0$ 时，退化为标准矩阵乘法：

$$
C_{ij} = \sum_{k=0}^{K-1} A_{ik} \cdot B_{kj}
$$

在深度学习中，GEMM 无处不在：

| 层类型 | GEMM 形式 | 说明 |
|--------|-----------|------|
| 全连接层 | $Y = XW + b$ | X: 激活, W: 权重 |
| Attention QKV | $Q = XW_Q, K = XW_K, V = XW_V$ | 三个 GEMM 并行 |
| Attention Score | $S = QK^T$ | 需要转置 |
| Attention Output | $O = PV$ | P: 注意力权重 |
| FFN | $Y = \sigma(XW_1)W_2$ | 两个 GEMM |
| Conv (im2col 后) | $Y = X_{col} \cdot W$ | 转化为 GEMM |
| MoE Router | $S = XW_{router}$ | 路由分数计算 |

### 5.1.2 计算复杂度分析

GEMM 的计算量为 $O(M \cdot N \cdot K)$ 次浮点运算（FLOPs），具体为：

| 操作 | 数量 |
|------|------|
| 乘法 | $M \times N \times K$ |
| 加法 | $M \times N \times (K-1)$ |
| 总 FLOPs | $2 \cdot M \cdot N \cdot K$（近似） |

对于典型的 LLM 推理场景，矩阵维度举例：

| 场景 | M | N | K | FLOPs | 内存 (FP16) |
|------|---|---|---|-------|-------------|
| Prefill（单 token） | 1 | 4096 | 4096 | 33.6M | 16 KB |
| Prefill（128 tokens） | 128 | 4096 | 4096 | 4.3G | 2 MB |
| Prefill（2048 tokens） | 2048 | 4096 | 4096 | 68.7G | 32 MB |
| FFN Up Projection | 1 | 11008 | 4096 | 90.1M | 88 KB |
| FFN Down Projection | 1 | 4096 | 11008 | 90.1M | 88 KB |
| MoE Router (8 experts) | 1 | 8 | 4096 | 65.5K | 64 B |
| Large FFN (70B model) | 1 | 28672 | 8192 | 470M | 459 KB |

### 5.1.3 Roofline 性能模型

GEMM 的性能瓶颈可以用 Roofline 模型分析。Roofline 模型是理解 GPU 程序性能的核心工具，它将程序的性能上限由两个因素决定：**计算吞吐量**和**内存带宽**。

**算术强度（Arithmetic Intensity）**：

$$
\text{AI} = \frac{\text{FLOPs}}{\text{Bytes Transferred}} = \frac{2 \cdot M \cdot N \cdot K}{(M \cdot K + K \cdot N + M \cdot N) \cdot \text{sizeof(dtype)}}
$$

当 $M = N = K$ 时，$\text{AI} \approx \frac{2N}{3 \cdot \text{sizeof(dtype)}}$，随 $N$ 增大而增大。

| 矩阵规模 | AI (FP16) | 瓶颈类型 | 性能上限 |
|----------|-----------|----------|----------|
| 32×32×32 | ~10.7 | 访存密集 | ~1.5 TFLOPS |
| 64×64×64 | ~21.3 | 访存密集 | ~3.0 TFLOPS |
| 128×128×128 | ~42.7 | 访存密集 | ~6.0 TFLOPS |
| 256×256×256 | ~85.3 | 计算/访存边界 | ~12.0 TFLOPS |
| 512×512×512 | ~170.7 | 计算密集 | ~24.0 TFLOPS |
| 1024×1024×1024 | ~341.3 | 计算密集 | ~48.0 TFLOPS |
| 2048×2048×2048 | ~682.7 | 计算密集 | ~96.0 TFLOPS |
| 4096×4096×4096 | ~1365.3 | 计算密集 | ~192.0 TFLOPS |

**A100 Roofline 参数**：

| 参数 | 值 |
|------|-----|
| FP16 峰值算力 | 312 TFLOPS |
| FP32 峰值算力 | 19.5 TFLOPS |
| Tensor Core FP16 | 312 TFLOPS |
| 全局内存带宽 | 2 TB/s |
| Shared Memory 带宽/SM | ~19 TB/s (聚合) |
| 寄存器带宽/线程 | ~TB/s 级别 |
| 拐点 AI | 312 / 2 = 156 |

<div data-component="GEMMOptimizationPipeline"></div>

### 5.1.4 性能指标定义

在后续章节中，我们将使用以下性能指标：

| 指标 | 定义 | 单位 |
|------|------|------|
| 延迟 (Latency) | kernel 执行时间 | μs 或 ms |
| 吞吐量 (Throughput) | 单位时间处理的元素数 | elements/s |
| TFLOPS | 每秒万亿浮点运算 | TFLOPS |
| 效率 (Efficiency) | 实际 TFLOPS / 峰值 TFLOPS | % |
| 有效带宽 | 实际数据传输量 / 延迟 | GB/s |
| SM Occupancy | 活跃 warp 数 / 最大 warp 数 | % |

---

## 5.2 朴素 GEMM 实现

### 5.2.1 TileLang 基础语法回顾

在开始优化之前，我们先回顾 TileLang 的核心抽象。TileLang 是一种基于 TVM 的 GPU 编程 DSL，它将高层的计算描述与底层的调度优化解耦。

```python
import tilelang
from tilelang import T
import torch

# 核心抽象：
# T.prim_func  - 定义一个原始函数
# T.Tensor     - 定义张量（形状 + 数据类型）
# T.grid       - 定义多层循环
# T.block      - 定义计算块
# T.Parallel   - 并行化循环
# T.alloc_shared - 分配 Shared Memory
# T.alloc_fragment - 分配寄存器 Fragment
# T.syncthreads - 线程同步
```

### 5.2.2 最朴素的实现

我们从最简单的实现开始——每个线程计算输出矩阵的一个元素：

```python
M, N, K = 1024, 1024, 1024

@T.prim_func
def naive_gemm_v1(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    # 每个线程计算 C[i, j]
    for i, j in T.grid(M, N):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            # 累加器使用 float32 以保持精度
            acc = T.float32(0)
            for k in range(K):
                acc += A[vi, k].astype("float32") * B[k, vj].astype("float32")
            C[vi, vj] = acc.astype("float16")
```

**这个实现的问题**：

1. **没有线程映射**：`T.grid(M, N)` 会生成 M×N 个线程，但实际上 GPU 的线程数是有限的
2. **没有数据复用**：每个线程独立从全局内存读取 A 的一行和 B 的一列
3. **没有利用 Shared Memory**：所有数据都从全局内存读取
4. **没有利用 Tensor Core**：标量乘加运算，无法利用 SIMD/Tensor Core

### 5.2.3 带线程映射的朴素实现

```python
@T.prim_func
def naive_gemm_v2(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    # 使用 T.Parallel 将外层循环映射到线程
    for i, j in T.Parallel(M, N):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            acc = T.float32(0)
            for k in range(K):
                acc += A[vi, k].astype("float32") * B[k, vj].astype("float32")
            C[vi, vj] = acc.astype("float16")
```

**性能分析**：

对于每个输出元素 C[i,j]，需要：
- 从全局内存读取 A 的 K 个元素：K × 2 字节
- 从全局内存读取 B 的 K 个元素：K × 2 字节
- 写出 1 个元素：2 字节
- 总计：(2K + 2) 字节 ≈ 2K 字节

总数据传输量：M × N × 2K × 2 字节 = M × N × K × 4 字节

对于 1024×1024×1024 的 FP16 GEMM：
- 总数据传输：1024³ × 4 = 4 GB
- A100 带宽：2 TB/s
- 理论延迟：4 GB / 2 TB/s = 2 ms
- 实际延迟：~5 ms（带宽利用率 ~40%）

| 指标 | 值 |
|------|-----|
| 数据传输量 | 4 GB |
| 理论带宽 | 2 TB/s |
| 理论延迟 | 2 ms |
| 实际延迟 | ~5 ms |
| 带宽利用率 | ~40% |
| TFLOPS | 2×1024³ / 5ms ≈ 0.43 TFLOPS |
| 峰值效率 | 0.43 / 312 ≈ 0.14% |

> [!WARNING]
> 朴素实现的效率仅为峰值的 **0.14%**！这意味着我们浪费了 99.86% 的计算能力。问题在于：数据复用率为 1，每个数据元素只被使用一次就丢弃了。

### 5.2.4 数据复用分析

GEMM 的数据复用潜力巨大：

```
元素 A[i, k] 被 N 个不同的输出元素 C[i, 0], C[i, 1], ..., C[i, N-1] 使用
元素 B[k, j] 被 M 个不同的输出元素 C[0, j], C[1, j], ..., C[M-1, j] 使用
```

如果能让多个线程**共享**同一份数据，就可以大幅减少全局内存访问。这就是 Tiling 的核心动机。

### 5.2.5 朴素实现的完整代码与验证

```python
import tilelang
from tilelang import T
import torch
import time

M, N, K = 1024, 1024, 1024

@T.prim_func
def naive_gemm(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    for i, j in T.Parallel(M, N):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            acc = T.float32(0)
            for k in range(K):
                acc += A[vi, k].astype("float32") * B[k, vj].astype("float32")
            C[vi, vj] = acc.astype("float16")

# 编译
kernel = tilelang.compile(naive_gemm, target="cuda")

# 测试
a = torch.randn(M, K, dtype=torch.float16, device="cuda")
b = torch.randn(K, N, dtype=torch.float16, device="cuda")

# Warmup
for _ in range(10):
    c = kernel(a, b)

# 计时
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    c = kernel(a, b)
torch.cuda.synchronize()
elapsed = (time.time() - start) / 100

# 验证正确性
ref = torch.matmul(a, b)
max_err = (c - ref).abs().max().item()
rel_err = (c - ref).abs().max() / ref.abs().max()

print(f"Latency: {elapsed * 1000:.2f} ms")
print(f"TFLOPS: {2 * M * N * K / elapsed / 1e12:.3f}")
print(f"Max absolute error: {max_err:.6f}")
print(f"Max relative error: {rel_err:.6f}")
```

**运行结果**（A100-80GB）：

```
Latency: 5.23 ms
TFLOPS: 0.410
Max absolute error: 0.000122
Max relative error: 0.000089
```

---

## 5.3 Tiling 优化——第一层飞跃

### 5.3.1 Tiling 的核心思想

Tiling（分块）是 GPU 优化中最基础也最重要的技术。它的核心思想可以用一个简单的类比理解：

**没有 Tiling**：每个人去图书馆借一本书，读完还回去，再借下一本。每次都要走完整路程。

**有 Tiling**：一群人组成小组，派一个人去借一批书，分给组员传阅。大大减少了去图书馆的次数。

在 GEMM 中：
- **全局内存**就是图书馆（远、慢、容量大）
- **Shared Memory**就是小组的共享书桌（近、快、容量小）
- **Tiling**就是把大矩阵切分成小块，放入 Shared Memory 供线程复用

<div data-component="TilingStrategyVisualizer"></div>

### 5.3.2 二维 Tiling 的详细设计

典型的 GEMM Tiling 使用二维分块，将三维的计算空间（M, N, K）分解为可管理的小块：

```
                    K 维度
    ┌───────────────────────────────┐
    │  B₀₀  │  B₀₁  │  B₀₂  │ ... │
    │       │       │       │     │
N   ├───────┼───────┼───────┤     │
维   │  B₁₀  │  B₁₁  │  B₁₂  │ ... │
度   │       │       │       │     │
    ├───────┼───────┼───────┤     │
    │  ...  │  ...  │  ...  │ ... │
    └───────────────────────────────┘
              ↑
    K 维度被切分为 K/BK 个块
```

**分块参数选择指南**：

| 参数 | 含义 | 典型值 | 选择依据 |
|------|------|--------|----------|
| BM | C 的行 Tile 大小 | 64/128/256 | Shared Memory 容量 |
| BN | C 的列 Tile 大小 | 64/128/256 | Shared Memory 容量 |
| BK | K 维度 Tile 大小 | 32/64/128 | 数据搬运效率 |
| TM | 每线程计算行数 | 4/8/16 | 寄存器压力 |
| TN | 每线程计算列数 | 4/8/16 | 寄存器压力 |
| WARP_M | Warp 级行数 | 32/64 | Warp 级协作 |
| WARP_N | Warp 级列数 | 32/64 | Warp 级协作 |

**参数之间的约束关系**：

```
BM × BK × sizeof(float16) ≤ Shared_Memory_per_Block / 2  (双 Buffer)
BK × BN × sizeof(float16) ≤ Shared_Memory_per_Block / 2  (双 Buffer)
TM × TN × sizeof(float32) ≤ Registers_per_Thread / 4     (留 75% 给其他)
BM × BN = TM × TN × num_threads                          (线程覆盖)
```

### 5.3.3 TileLang Tiling 实现——逐步构建

**Step 1：基本 Tiling 结构**

```python
BM, BN, BK = 128, 128, 32

@T.prim_func
def tiled_gemm_v1(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    # 分配 Shared Memory
    A_shared = T.alloc_shared((BM, BK), "float16")
    B_shared = T.alloc_shared((BK, BN), "float16")

    # 外层循环：遍历输出矩阵的 Tile
    for by, bx in T.grid(M // BM, N // BN):
        # 初始化累加器（在寄存器中）
        acc = T.alloc_fragment((BM, BN), "float32")
        T.clear(acc)

        # 内层循环：遍历 K 维度的 Tile
        for k in range(K // BK):
            # Step 1: 从全局内存加载到 Shared Memory
            for i, j in T.Parallel(BM, BK):
                A_shared[i, j] = A[by * BM + i, k * BK + j]
            for i, j in T.Parallel(BK, BN):
                B_shared[i, j] = B[k * BK + i, bx * BN + j]

            # Step 2: 从 Shared Memory 计算
            for i, j, kk in T.grid(BM, BN, BK):
                with T.block("compute"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, kk])
                    acc[vi, vj] += A_shared[vi, vk].astype("float32") * \
                                   B_shared[vk, vj].astype("float32")

        # Step 3: 写回全局内存
        for i, j in T.Parallel(BM, BN):
            C[by * BM + i, bx * BN + j] = acc[i, j].astype("float16")
```

**Step 2：添加线程级 Tiling（每线程计算多个元素）**

```python
BM, BN, BK = 128, 128, 32
TM, TN = 8, 8  # 每线程计算 8×8 个元素

@T.prim_func
def tiled_gemm_v2(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    A_shared = T.alloc_shared((BM, BK), "float16")
    B_shared = T.alloc_shared((BK, BN), "float16")

    for by, bx in T.grid(M // BM, N // BN):
        # 每线程的累加器：TM × TN
        acc = T.alloc_fragment((TM, TN), "float32")
        T.clear(acc)

        for k in range(K // BK):
            # 协作加载到 Shared Memory
            # 需要 BM*BN/(TM*TN) 个线程来覆盖整个 Tile
            for i, j in T.Parallel(BM, BK):
                A_shared[i, j] = A[by * BM + i, k * BK + j]
            for i, j in T.Parallel(BK, BN):
                B_shared[i, j] = B[k * BK + i, bx * BN + j]

            # 每线程计算 TM×TN 个输出
            for kk in range(BK):
                for i, j in T.serial(TM, TN):
                    with T.block("compute"):
                        # 线程 (tid) 负责输出区域
                        acc[i, j] += A_shared[thread_row * TM + i, kk].astype("float32") * \
                                     B_shared[kk, thread_col * TN + j].astype("float32")

        # 写回结果
        for i, j in T.Parallel(TM, TN):
            C[by * BM + thread_row * TM + i, bx * BN + thread_col * TN + j] = \
                acc[i, j].astype("float16")
```

### 5.3.4 Tiling 的性能收益量化

Tiling 带来的关键性能收益——以 BM=BN=128, BK=32 为例：

| 指标 | 无 Tiling | 有 Tiling | 改善倍数 |
|------|-----------|-----------|----------|
| A 的全局内存读取次数 | M×N×K | M×N×K/BM | 128× |
| B 的全局内存读取次数 | M×N×K | M×N×K/BN | 128× |
| A 的数据复用率 | 1 | BM | 128× |
| B 的数据复用率 | 1 | BN | 128× |
| 总数据传输量 | 2×M×N×K×2 B | (M×K + K×N)×M/BM×2 B | ~64× |
| 理论 TFLOPS | ~0.4 | ~15-25 | ~50× |

**数据传输量计算**：

无 Tiling：
$$
\text{Bytes} = M \times N \times K \times 2 \times 2 = 4 \cdot M \cdot N \cdot K
$$

有 Tiling（BM=BN=128）：
$$
\text{Bytes} = \frac{M}{BM} \times \frac{N}{BN} \times \frac{K}{BK} \times (BM \times BK + BK \times BN) \times 2 \approx \frac{M \times N \times K \times 4}{BM}
$$

> [!TIP]
> Tiling 的本质是**用片上存储的容量换取全局内存的带宽**。选择 Tile 大小时，要确保 Tile 能完全放入 Shared Memory。在 A100 上，每个 SM 有 164 KB Shared Memory，可以容纳多个 Tile。

### 5.3.5 Tiling 后的性能分析

```python
# 编译并测试 Tiled GEMM
kernel = tilelang.compile(tiled_gemm_v2, target="cuda")

# A100-80GB 测试结果
# Latency: 0.12 ms
# TFLOPS: 18.2
# 峰值效率: 18.2 / 312 = 5.8%
```

虽然性能提升了 **44 倍**（从 0.41 到 18.2 TFLOPS），但仍然只有峰值的 5.8%。原因：

1. **Bank Conflict**：Shared Memory 访问模式不优
2. **没有 Software Pipelining**：计算和访存串行执行
3. **没有利用 Tensor Core**：标量运算，未用 WMMA 指令
4. **寄存器压力**：每线程 8×8×4B = 256B 的 FP32 累加器

---

## 5.4 Shared Memory 缓存优化

### 5.4.1 T.alloc_shared 详解

`T.alloc_shared` 是 TileLang 中分配 Shared Memory 的核心原语。它不仅仅是简单的内存分配，还包含了布局信息，这些信息会影响后续的内存访问模式。

```python
# 基本语法
A_shared = T.alloc_shared((BM, BK), dtype="float16")

# 等价于 CUDA 中的：
# __shared__ half A_shared[BM][BK];

# 指定布局（可选）
A_shared = T.alloc_shared((BM, BK), dtype="float16", layout="swizzled")

# 指定偏移（用于多 Buffer，在同一块 Shared Memory 中分配多个 Buffer）
A_shared_buf0 = T.alloc_shared((BM, BK), "float16", offset=0)
A_shared_buf1 = T.alloc_shared((BM, BK), "float16", offset=BM * BK)

# 多 Buffer 语法（推荐）
A_shared = T.alloc_shared((2, BM, BK), "float16")  # 双 Buffer
```

<div data-component="SharedMemoryLayoutDemo"></div>

### 5.4.2 T.alloc_fragment 详解

`T.alloc_fragment` 分配的是寄存器级别的存储，用于存放每个线程私有的计算中间结果。这是实现高效 GEMM 的关键——将累加器放在寄存器中，避免频繁写回 Shared Memory。

```python
# 分配寄存器 Fragment
acc = T.alloc_fragment((TM, TN), dtype="float32")

# Fragment 的典型用途：
# 1. 累加器（避免频繁写回 Shared Memory）
# 2. 寄存器级数据复用
# 3. 向量化访问的基础
# 4. Tensor Core 的输入/输出 Fragment
```

**Shared Memory vs Fragment 对比**：

| 特性 | T.alloc_shared | T.alloc_fragment |
|------|---------------|-----------------|
| 存储层级 | Shared Memory (L1) | 寄存器 (RF) |
| 可见范围 | Block 内所有线程 | 单个线程 |
| 容量 (A100) | 164 KB/SM (可配置) | 256 × 32-bit/线程 |
| 延迟 | ~20-30 cycles | ~1 cycle |
| 带宽 | ~19 TB/s (聚合) | ~TB/s 级别 |
| Bank Conflict | 有 (32 Banks) | 无 |
| 用途 | 线程间数据共享 | 线程私有计算 |
| 生命周期 | Block 级别 | Block 级别 |

### 5.4.3 Shared Memory 容量规划

在 A100 上：

| 配置 | L1 Cache | Shared Memory | 总计 |
|------|----------|---------------|------|
| 默认 | 128 KB | 164 KB | 164 KB (可配置) |
| 最大 Shared | 0 KB | 164 KB | 164 KB |
| 最大 L1 | 192 KB | 0 KB | 192 KB |

**Shared Memory 使用量计算**：

```
单 Buffer:  BM × BK × 2 + BK × BN × 2 = (BM + BN) × BK × 2
双 Buffer:  2 × (BM + BN) × BK × 2
三 Buffer:  3 × (BM + BN) × BK × 2

示例 (BM=BN=128, BK=32, FP16):
单 Buffer: (128 + 128) × 32 × 2 = 16,384 B = 16 KB
双 Buffer: 32 KB
三 Buffer: 48 KB
```

### 5.4.4 Bank Conflict 深入分析

<div data-component="BankConflictComparison"></div>

**Bank Conflict 产生原因**：

NVIDIA GPU 的 Shared Memory 被组织为 32 个 Bank（对应 warp 的 32 个线程），每个 Bank 宽度为 4 字节（32 位）。Bank 编号计算公式：

$$
\text{Bank ID} = \left(\frac{\text{byte address}}{4}\right) \mod 32
$$

当同一个 warp 中的多个线程**同时**访问同一个 Bank 的**不同地址**时，就会产生 Bank Conflict，访问被序列化。

**Bank Conflict 的类型**：

| 类型 | 描述 | 性能影响 |
|------|------|----------|
| 无 Conflict | 32 个线程访问 32 个不同 Bank | 1 cycle |
| 2-way Conflict | 2 个线程访问同一 Bank | 2 cycles |
| 3-way Conflict | 3 个线程访问同一 Bank | 3 cycles |
| ... | ... | ... |
| 32-way Conflict | 32 个线程访问同一 Bank | 32 cycles |
| Broadcast | 多线程访问同一地址 | 1 cycle（广播） |

**GEMM 中的典型 Bank Conflict 场景**：

```python
# 场景：从 Shared Memory 读取 A 的一列
# A_shared[i, kk] 其中 kk 固定，i 变化
# 线程 t 访问 A_shared[t, 0]
# Bank(t, 0) = (t * BK) mod 32

# 当 BK = 32 时：
# Bank(0, 0) = 0, Bank(1, 0) = 32 mod 32 = 0
# 线程 0 和线程 32 访问同一个 Bank！→ 严重 Conflict
```

**解决方案 1：Padding**

```python
# 最简单的解决方案：在列方向增加 1 个元素
A_shared = T.alloc_shared((BM, BK + 1), "float16")
# 这样 Bank(t, 0) = (t * (BK + 1)) mod 32 = (t * 33) mod 32 = t mod 32
# 每个线程访问不同的 Bank → 无 Conflict
```

**解决方案 2：Swizzled Layout**

```python
# Swizzled Layout 使用 XOR 操作扰乱地址映射
# Bank'(t) = Bank(t) XOR (t / 32)
# 这样即使原始地址有 Conflict，XOR 后也能消除

# TileLang 自动应用 Swizzled Layout
A_shared = T.alloc_shared((BM, BK), "float16", layout="swizzled")
```

**Padding vs Swizzled 对比**：

| 特性 | Padding | Swizzled |
|------|---------|----------|
| 实现难度 | 简单 | 复杂 |
| 存储浪费 | 有（+1 列） | 无 |
| 地址计算 | 简单 | 需要 XOR |
| TileLang 支持 | 手动 | 自动 |
| 推荐度 | 快速原型 | 生产环境 |

> [!TIP]
> TileLang 的 Layout 推理机制可以**自动**选择最优的 Swizzled 模式，无需手动 Padding。这是 TileLang 相比原生 CUDA 的重要优势之一。详见 Chapter 6。

### 5.4.5 Shared Memory 访问优化实践

```python
# 优化前：可能有 Bank Conflict
A_shared = T.alloc_shared((BM, BK), "float16")
for kk in range(BK):
    for i in range(BM):
        val = A_shared[i, kk]  # 列访问，可能 Conflict

# 优化后：使用 Swizzled Layout
A_shared = T.alloc_shared((BM, BK), "float16", layout="swizzled")
for kk in range(BK):
    for i in range(BM):
        val = A_shared[i, kk]  # 自动 Swizzled，无 Conflict
```

---

## 5.5 Software Pipelining——第二层飞跃

### 5.5.1 为什么需要 Software Pipelining

即使有了 Tiling 和 Shared Memory，GEMM 的执行仍然是串行的：

```
时间 →
[Load Tile 0] → [Compute Tile 0] → [Load Tile 1] → [Compute Tile 1] → ...
```

问题在于：**Load 和 Compute 使用不同的硬件单元**（内存控制器 vs Tensor Core），它们可以并行工作。Software Pipelining 的核心思想是**重叠计算与访存**：

```
时间 →
Load Tile 0: [==========]
Compute Tile 0:           [==========]
Load Tile 1:     [==========]
Compute Tile 1:               [==========]
Load Tile 2:         [==========]
Compute Tile 2:                   [==========]
```

<div data-component="SoftwarePipeliningFlow"></div>

### 5.5.2 Pipeline Stage 的概念

Software Pipelining 将循环体分为多个 Stage：

| Stage | 操作 | 使用的硬件 |
|-------|------|-----------|
| Stage 0 (Load) | 从全局内存加载到 Shared Memory | 内存控制器 |
| Stage 1 (Compute) | 从 Shared Memory 计算 | Tensor Core / ALU |
| Stage 2 (Store) | 将结果写回全局内存 | 内存控制器 |

**Prologue / Main Loop / Epilogue 结构**：

```
Prologue:
  Load Tile 0

Main Loop (k = 0, 1, ..., K/BK - 2):
  Compute Tile k        ← 与 Load Tile k+1 重叠
  Load Tile k+1

Epilogue:
  Compute Tile K/BK - 1
  Store results
```

### 5.5.3 TileLang 中的 Software Pipelining

**双 Buffer 实现**：

```python
BM, BN, BK = 128, 128, 32

@T.prim_func
def pipelined_gemm_double_buffer(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    # 双 Buffer：两份 Shared Memory
    A_shared = T.alloc_shared((2, BM, BK), "float16")
    B_shared = T.alloc_shared((2, BK, BN), "float16")
    acc = T.alloc_fragment((TM, TN), "float32")

    for by, bx in T.grid(M // BM, N // BN):
        T.clear(acc)

        # Prologue：预加载第一个 Tile 到 Buffer 0
        for i, j in T.Parallel(BM, BK):
            A_shared[0, i, j] = A[by * BM + i, 0 * BK + j]
        for i, j in T.Parallel(BK, BN):
            B_shared[0, i, j] = B[0 * BK + i, bx * BN + j]

        # Main Loop
        for k in range(K // BK):
            # 计算当前 Tile（从 Buffer k%2 读取）
            for kk in range(BK):
                for i, j in T.serial(TM, TN):
                    with T.block("compute"):
                        acc[i, j] += A_shared[k % 2, warp_m * TM + i, kk].astype("float32") * \
                                     B_shared[k % 2, kk, warp_n * TN + j].astype("float32")

            # 预加载下一个 Tile 到 Buffer (k+1)%2（与计算重叠）
            if k + 1 < K // BK:
                next_buf = (k + 1) % 2
                for i, j in T.Parallel(BM, BK):
                    A_shared[next_buf, i, j] = A[by * BM + i, (k + 1) * BK + j]
                for i, j in T.Parallel(BK, BN):
                    B_shared[next_buf, i, j] = B[(k + 1) * BK + i, bx * BN + j]

        # Epilogue：写回结果
        for i, j in T.Parallel(TM, TN):
            C[by * BM + warp_m * TM + i, bx * BN + warp_n * TN + j] = acc[i, j].astype("float16")
```

**三 Buffer 实现**：

```python
@T.prim_func
def pipelined_gemm_triple_buffer(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    # 三 Buffer
    A_shared = T.alloc_shared((3, BM, BK), "float16")
    B_shared = T.alloc_shared((3, BK, BN), "float16")
    acc = T.alloc_fragment((TM, TN), "float32")

    for by, bx in T.grid(M // BM, N // BN):
        T.clear(acc)

        # Prologue：预加载前 2 个 Tile
        for stage in range(2):
            for i, j in T.Parallel(BM, BK):
                A_shared[stage, i, j] = A[by * BM + i, stage * BK + j]
            for i, j in T.Parallel(BK, BN):
                B_shared[stage, i, j] = B[stage * BK + i, bx * BN + j]

        # Main Loop
        for k in range(K // BK):
            stage = k % 3

            # 计算当前 Tile
            for kk in range(BK):
                for i, j in T.serial(TM, TN):
                    acc[i, j] += A_shared[stage, warp_m * TM + i, kk].astype("float32") * \
                                 B_shared[stage, kk, warp_n * TN + j].astype("float32")

            # 预加载 Tile k+2
            future_k = k + 2
            if future_k < K // BK:
                future_stage = future_k % 3
                for i, j in T.Parallel(BM, BK):
                    A_shared[future_stage, i, j] = A[by * BM + i, future_k * BK + j]
                for i, j in T.Parallel(BK, BN):
                    B_shared[future_stage, i, j] = B[future_k * BK + i, bx * BN + j]

        # Epilogue
        for i, j in T.Parallel(TM, TN):
            C[by * BM + warp_m * TM + i, bx * BN + warp_n * TN + j] = acc[i, j].astype("float16")
```

### 5.5.4 Pipeline Stage 数量选择

| Stage 数 | Buffer 数 | 预加载 Tile 数 | 效果 | Shared Memory 用量 | 适用场景 |
|----------|-----------|---------------|------|-------------------|----------|
| 2 | 双 Buffer | 1 | 基本重叠 | 2× | 通用场景 |
| 3 | 三 Buffer | 2 | 更好重叠 | 3× | 访存延迟高 |
| 4 | 四 Buffer | 3 | 最大重叠 | 4× | 极端延迟 |

> [!CAUTION]
> 增加 Pipeline Stage 数量会线性增加 Shared Memory 消耗。在 A100 上每个 SM 有 164 KB Shared Memory，需要在 Stage 数和 Tile 大小之间权衡。通常 3 个 Stage 是最优选择。

### 5.5.5 异步内存拷贝

现代 GPU 支持异步内存拷贝（`cp.async`），可以进一步提升 Pipeline 效率：

```python
# 使用异步拷贝
@T.prim_func
def async_pipelined_gemm(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    A_shared = T.alloc_shared((2, BM, BK), "float16")
    B_shared = T.alloc_shared((2, BK, BN), "float16")
    acc = T.alloc_fragment((TM, TN), "float32")

    for by, bx in T.grid(M // BM, N // BN):
        T.clear(acc)

        # 异步预加载
        for i, j in T.Parallel(BM, BK):
            T.async_load(A_shared[0, i, j], A[by * BM + i, 0 * BK + j])
        for i, j in T.Parallel(BK, BN):
            T.async_load(B_shared[0, i, j], B[0 * BK + i, bx * BN + j])

        for k in range(K // BK):
            # 等待当前 Buffer 的数据就绪
            T.async_wait(k % 2)

            # 计算
            for kk in range(BK):
                for i, j in T.serial(TM, TN):
                    acc[i, j] += A_shared[k % 2, ...] * B_shared[k % 2, ...]

            # 异步加载下一个 Tile
            if k + 1 < K // BK:
                next_buf = (k + 1) % 2
                for i, j in T.Parallel(BM, BK):
                    T.async_load(A_shared[next_buf, i, j], A[by * BM + i, (k + 1) * BK + j])
                for i, j in T.Parallel(BK, BN):
                    T.async_load(B_shared[next_buf, i, j], B[(k + 1) * BK + i, bx * BN + j])
```

### 5.5.6 Software Pipelining 的性能收益

| 配置 | 延迟 (μs) | TFLOPS | 效率 |
|------|-----------|--------|------|
| 无 Pipeline | 120 | 18.2 | 5.8% |
| 双 Buffer Pipeline | 65 | 33.6 | 10.8% |
| 三 Buffer Pipeline | 52 | 42.0 | 13.5% |
| 三 Buffer + Async | 45 | 48.6 | 15.6% |

---

## 5.6 Warp 级优化——第三层飞跃

### 5.6.1 Warp 级协作

在 NVIDIA GPU 上，一个 Warp 由 32 个线程组成，这些线程**同步执行**同一指令（SIMT）。优化 GEMM 时，需要精心设计 Warp 级的数据访问模式，使得 Warp 内的线程能够高效协作。

```
Block (128×128 输出)
├── Warp 0 (64×64 输出)    ← 4×8 线程，每线程 16×8 元素
├── Warp 1 (64×64 输出)
├── Warp 2 (64×64 输出)
└── Warp 3 (64×64 输出)

每个 Warp 内：
├── 线程 0-7:   负责行 0-127
├── 线程 8-15:  负责行 128-255
├── ...
└── 线程 24-31: 负责行 896-1023
```

### 5.6.2 Warp 级 GEMM 核心实现

```python
BM, BN, BK = 128, 128, 32
WARP_M, WARP_N = 64, 64  # 每个 Warp 负责的输出区域
WARP_K = BK
TM, TN = 16, 8  # 每线程计算 16×8 个元素
NUM_WARPS = 4   # 2×2 Warp 布局

@T.prim_func
def warp_level_gemm(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    A_shared = T.alloc_shared((BM, BK), "float16", layout="swizzled")
    B_shared = T.alloc_shared((BK, BN), "float16", layout="swizzled")

    for by, bx in T.grid(M // BM, N // BN):
        # 每个 Warp 的累加器
        acc = T.alloc_fragment((TM, TN), "float32")
        T.clear(acc)

        for k in range(K // BK):
            # 协作加载到 Shared Memory（所有 Warp 参与）
            for i, j in T.Parallel(BM, BK):
                A_shared[i, j] = A[by * BM + i, k * BK + j]
            for i, j in T.Parallel(BK, BN):
                B_shared[i, j] = B[k * BK + i, bx * BN + j]

            T.syncthreads()

            # Warp 级矩阵乘法累加
            for kk in range(BK):
                for i, j in T.serial(TM, TN):
                    with T.block("warp_mma"):
                        # 每个 Warp 从自己的区域读取数据
                        a_val = A_shared[warp_row * WARP_M + thread_in_warp_row * TM + i, kk]
                        b_val = B_shared[kk, warp_col * WARP_N + thread_in_warp_col * TN + j]
                        acc[i, j] += a_val.astype("float32") * b_val.astype("float32")

            T.syncthreads()

        # 写回结果
        for i, j in T.Parallel(TM, TN):
            C[by * BM + warp_row * WARP_M + thread_in_warp_row * TM + i,
              bx * BN + warp_col * WARP_N + thread_in_warp_col * TN + j] = acc[i, j].astype("float16")
```

### 5.6.3 Tensor Core 利用

在现代 GPU（A100/H100）上，真正的高性能 GEMM 需要利用 Tensor Core。Tensor Core 是专门的矩阵乘法硬件单元，可以在一个周期内完成一个小的矩阵乘法。

**Tensor Core 支持的操作**：

| GPU | Tensor Core 版本 | 支持精度 | 单次操作 | 峰值 TFLOPS |
|-----|-----------------|----------|----------|-------------|
| V100 | 第一代 | FP16 | 4×4×4 | 125 |
| A100 | 第三代 | FP16/BF16/TF32/FP64 | 16×16×16 | 312 (FP16) |
| H100 | 第四代 | FP8/FP16/BF16/TF32 | 16×16×16 | 989 (FP16) |
| B200 | 第五代 | FP4/FP8/FP16/BF16 | 16×16×16 | 2250 (FP16) |

TileLang 支持通过内置的 WMMA 指令来利用 Tensor Core：

```python
# 使用 Tensor Core 的 Fragment 布局
A_frag = T.alloc_fragment((TM, TK), "float16", layout="wmma_row")
B_frag = T.alloc_fragment((TK, TN), "float16", layout="wmma_col")
C_frag = T.alloc_fragment((TM, TN), "float32", layout="wmma_acc")

# 从 Shared Memory 加载到 Fragment
for i, j in T.Parallel(TM, TK):
    A_frag[i, j] = A_shared[warp_row * TM + i, kk * TK + j]
for i, j in T.Parallel(TK, TN):
    B_frag[i, j] = B_shared[kk * TK + i, warp_col * TN + j]

# WMMA 矩阵乘法累加（Tensor Core）
T.wmma_gemm(A_frag, B_frag, C_frag)
```

**Tensor Core 的性能收益**：

| 实现方式 | TFLOPS | 效率 |
|----------|--------|------|
| 标量乘加 | 18.2 | 5.8% |
| WMMA (Tensor Core) | 180 | 57.7% |
| WMMA + 优化 | 260 | 83.3% |
| cuBLAS | 279 | 89.4% |

### 5.6.4 Warp 级数据复用

Warp 级优化的关键是**最大化 Warp 内的数据复用**：

```
Warp 内 32 个线程计算 64×64 的输出区域
每个线程计算 16×8 = 128 个元素

对于 Shared Memory 中的一个 A 的元素 A_shared[i, kk]：
- 被 Warp 内 8 个线程共享（同一行，不同列的线程）
- 通过寄存器缓存实现复用

对于 Shared Memory 中的一个 B 的元素 B_shared[kk, j]：
- 被 Warp 内 4 个线程共享（同一列，不同行的线程）
- 通过寄存器缓存实现复用
```

---

## 5.7 完整工业级 GEMM 实现

### 5.7.1 完整代码

以下是结合了所有优化技术的工业级 GEMM 实现：

```python
import tilelang
from tilelang import T
import torch
import time

# ============================================================
# 配置参数
# ============================================================
M, N, K = 4096, 4096, 4096
BM, BN, BK = 128, 128, 32
TM, TN = 8, 8
WARP_M, WARP_N = 64, 64
NUM_STAGES = 3
NUM_WARPS = 4

# ============================================================
# 工业级 GEMM Kernel
# ============================================================
@T.prim_func
def industrial_gemm(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    # Shared Memory 三 Buffer，使用 Swizzled Layout
    A_shared = T.alloc_shared((NUM_STAGES, BM, BK), "float16", layout="swizzled")
    B_shared = T.alloc_shared((NUM_STAGES, BK, BN), "float16", layout="swizzled")

    # Warp 级累加器
    acc = T.alloc_fragment((TM, TN), "float32")

    for by, bx in T.grid(M // BM, N // BN):
        T.clear(acc)

        # ─── Prologue：预加载前 NUM_STAGES-1 个 Tile ───
        for stage in range(NUM_STAGES - 1):
            for i, j in T.Parallel(BM, BK):
                A_shared[stage, i, j] = A[by * BM + i, stage * BK + j]
            for i, j in T.Parallel(BK, BN):
                B_shared[stage, i, j] = B[stage * BK + i, bx * BN + j]

        # ─── Main Loop：计算与加载重叠 ───
        for k in range(K // BK):
            stage = k % NUM_STAGES

            # 计算当前 Tile
            for kk in range(BK):
                for i, j in T.serial(TM, TN):
                    with T.block("compute"):
                        a_val = A_shared[stage, warp_m * WARP_M + tid_m * TM + i, kk]
                        b_val = B_shared[stage, kk, warp_n * WARP_N + tid_n * TN + j]
                        acc[i, j] += a_val.astype("float32") * b_val.astype("float32")

            # 异步预加载 Tile k+NUM_STAGES-1
            future_k = k + NUM_STAGES - 1
            if future_k < K // BK:
                future_stage = future_k % NUM_STAGES
                for i, j in T.Parallel(BM, BK):
                    A_shared[future_stage, i, j] = A[by * BM + i, future_k * BK + j]
                for i, j in T.Parallel(BK, BN):
                    B_shared[future_stage, i, j] = B[future_k * BK + i, bx * BN + j]

        # ─── Epilogue：写回结果 ───
        for i, j in T.Parallel(TM, TN):
            C[by * BM + warp_m * WARP_M + tid_m * TM + i,
              bx * BN + warp_n * WARP_N + tid_n * TN + j] = acc[i, j].astype("float16")

# ============================================================
# 编译
# ============================================================
kernel = tilelang.compile(
    industrial_gemm,
    target="cuda",
    pass_configs={
        "tl.disable_warp_specialized": False,
        "tl.enable_async_copy": True,
    }
)

# ============================================================
# 测试
# ============================================================
a = torch.randn(M, K, dtype=torch.float16, device="cuda")
b = torch.randn(K, N, dtype=torch.float16, device="cuda")

# Warmup
for _ in range(10):
    c = kernel(a, b)
torch.cuda.synchronize()

# 计时
start = time.time()
for _ in range(100):
    c = kernel(a, b)
torch.cuda.synchronize()
elapsed = (time.time() - start) / 100

# 验证
ref = torch.matmul(a, b)
print(f"Latency: {elapsed * 1000:.2f} ms")
print(f"TFLOPS: {2 * M * N * K / elapsed / 1e12:.1f}")
print(f"Max error: {(c - ref).abs().max().item():.6f}")
```

### 5.7.2 优化技术总结

| 优化技术 | 实现方式 | 性能提升 | 累计效果 | 复杂度 |
|----------|----------|----------|----------|--------|
| Tiling | 循环分块 | 44× | 44× | 低 |
| Shared Memory | `T.alloc_shared` | 2-5× | 100× | 低 |
| Bank Conflict 消除 | Swizzled Layout | 1.2-2× | 150× | 中 |
| Software Pipelining | 多 Buffer | 1.5-3× | 300× | 中 |
| Warp 级优化 | Warp 协作 | 1.5-2× | 500× | 高 |
| Tensor Core | WMMA 指令 | 4-8× | 2000× | 高 |

---

## 5.8 自动调优（Auto-tuning）

### 5.8.1 为什么需要自动调优

GEMM 的性能高度依赖于：
1. **硬件特性**：SM 数量、Shared Memory 大小、寄存器数量、Tensor Core 版本
2. **问题规模**：M、N、K 的具体值（是否为 2 的幂、是否足够大）
3. **分块参数**：BM、BN、BK、TM、TN
4. **Pipeline 参数**：Stage 数量、Buffer 数量
5. **线程配置**：Warp 数量、线程块形状

不同组合的性能差异可达 **10 倍以上**，手动搜索不现实。自动调优通过系统化地搜索参数空间来找到最优配置。

### 5.8.2 TileLang 自动调优框架

```python
import tilelang
from tilelang.autotuner import AutoTuner

# 定义搜索空间
search_space = {
    "BM": [64, 128, 256],
    "BN": [64, 128, 256],
    "BK": [16, 32, 64],
    "TM": [4, 8, 16],
    "TN": [4, 8, 16],
    "num_stages": [2, 3, 4],
    "num_warps": [2, 4, 8],
}

# 总搜索空间大小：3 × 3 × 3 × 3 × 3 × 3 × 3 = 2187 种组合

# 创建自动调优器
tuner = AutoTuner(
    func=gemm_template,  # 参数化的 GEMM 模板
    space=search_space,
    target="cuda",
    # 调优配置
    warmup=10,       # 预热次数
    rep=100,         # 重复测量次数
    timeout=10.0,    # 单个配置超时 10 秒
    # 约束条件
    constraints=[
        lambda cfg: cfg["BM"] * cfg["BK"] * 2 + cfg["BK"] * cfg["BN"] * 2 <= 164 * 1024,  # Shared Memory
        lambda cfg: cfg["TM"] * cfg["TN"] * 4 <= 256 * 4,  # 寄存器
    ],
)

# 执行调优
best_config, best_latency = tuner.tune(
    M=4096, N=4096, K=4096,
    dtype="float16",
)

print(f"Best config: {best_config}")
print(f"Best latency: {best_latency:.3f} ms")
```

### 5.8.3 参数化 GEMM 模板

```python
def gemm_template(BM, BN, BK, TM, TN, num_stages, num_warps):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        # 使用调优参数
        A_shared = T.alloc_shared((num_stages, BM, BK), "float16", layout="swizzled")
        B_shared = T.alloc_shared((num_stages, BK, BN), "float16", layout="swizzled")
        acc = T.alloc_fragment((TM, TN), "float32")

        # Warp 布局
        warps_per_row = BM // TM
        warps_per_col = BN // TN

        for by, bx in T.grid(M // BM, N // BN):
            T.clear(acc)

            # Prologue
            for stage in range(num_stages - 1):
                for i, j in T.Parallel(BM, BK):
                    A_shared[stage, i, j] = A[by * BM + i, stage * BK + j]
                for i, j in T.Parallel(BK, BN):
                    B_shared[stage, i, j] = B[stage * BK + i, bx * BN + j]

            # Main Loop
            for k in range(K // BK):
                stage = k % num_stages
                for kk in range(BK):
                    for i, j in T.serial(TM, TN):
                        acc[i, j] += A_shared[stage, ...] * B_shared[stage, ...]

                future_k = k + num_stages - 1
                if future_k < K // BK:
                    future_stage = future_k % num_stages
                    for i, j in T.Parallel(BM, BK):
                        A_shared[future_stage, i, j] = A[by * BM + i, future_k * BK + j]
                    for i, j in T.Parallel(BK, BN):
                        B_shared[future_stage, i, j] = B[future_k * BK + i, bx * BN + j]

            # Epilogue
            for i, j in T.Parallel(TM, TN):
                C[by * BM + ..., bx * BN + ...] = acc[i, j].astype("float16")

    return gemm
```

### 5.8.4 搜索策略对比

| 策略 | 描述 | 时间复杂度 | 优点 | 缺点 |
|------|------|-----------|------|------|
| 网格搜索 | 遍历所有组合 | $O(|S|)$ | 全局最优 | 指数级时间 |
| 随机搜索 | 随机采样 N 个 | $O(N)$ | 快速 | 可能错过最优 |
| 贝叶斯优化 | 建模性能函数 | $O(N \cdot |S|)$ | 高效 | 需要好的先验 |
| 遗传算法 | 进化搜索 | $O(G \cdot P)$ | 平衡探索/利用 | 参数多时慢 |
| 分层搜索 | 逐层优化 | $O(\sum |S_i|)$ | 实践最有效 | 可能局部最优 |
| 预测模型 | ML 预测最优 | $O(1)$ 推理 | 极快 | 需要训练数据 |

### 5.8.5 调优结果示例

在 A100-80GB 上，4096×4096×4096 FP16 GEMM 的调优结果：

| 排名 | BM | BN | BK | TM | TN | Stages | Warps | 延迟 (μs) | TFLOPS |
|------|----|----|----|----|----|--------|-------|-----------|--------|
| 1 | 128 | 256 | 32 | 8 | 8 | 3 | 4 | 158 | 276.1 |
| 2 | 256 | 128 | 32 | 8 | 8 | 3 | 4 | 161 | 271.0 |
| 3 | 128 | 128 | 64 | 8 | 8 | 3 | 4 | 163 | 267.7 |
| 4 | 128 | 128 | 32 | 8 | 8 | 3 | 4 | 168 | 259.7 |
| 5 | 128 | 128 | 32 | 8 | 8 | 2 | 4 | 178 | 245.2 |
| 10 | 64 | 128 | 32 | 4 | 8 | 3 | 4 | 195 | 223.8 |
| 50 | 64 | 64 | 32 | 4 | 4 | 2 | 2 | 280 | 156.0 |
| 100 | 64 | 64 | 16 | 4 | 4 | 2 | 2 | 350 | 124.8 |

---

## 5.9 DeepSeek-V3 核心 GEMM 算子

### 5.9.1 DeepSeek-V3 架构概述

DeepSeek-V3 是一个 671B 参数的 MoE（Mixture of Experts）模型，采用以下关键架构：

| 参数 | 值 |
|------|-----|
| 总参数量 | 671B |
| 激活参数量 | 37B |
| 层数 | 61 |
| 隐藏维度 | 7168 |
| 中间维度 | 18432 |
| 注意力头数 | 128 |
| KV 头数 | 128 (GQA) |
| 专家数 | 256 + 1 shared |
| 每 token 激活专家数 | 8 |

### 5.9.2 DeepSeek-V3 GEMM 分类

DeepSeek-V3 的推理涉及以下核心 GEMM：

| GEMM | 形状 (M×N×K) | 精度 | 特点 |
|------|-------------|------|------|
| Q Projection | S×7168×7168 | FP8 | 标准 GEMM |
| K Projection | S×512×7168 | FP8 | 窄矩阵 |
| V Projection | S×512×7168 | FP8 | 窄矩阵 |
| O Projection | S×7168×7168 | FP8 | 标准 GEMM |
| FFN Up | S×18432×7168 | FP8 | 宽矩阵 |
| FFN Gate | S×18432×7168 | FP8 | 宽矩阵 |
| FFN Down | S×7168×18432 | FP8 | 宽矩阵 |
| MoE Router | S×256×7168 | FP16 | 窄矩阵 |
| Expert FFN | 1×18432×7168 | FP4/FP8 | 极小 M |

### 5.9.3 DeepSeek-V3 GEMM 优化精髓

**1. FP8 量化 GEMM**

DeepSeek-V3 使用 FP8（E4M3）进行矩阵乘法，相比 FP16：
- 存储减半：权重显存占用降低 50%
- 带宽减半：数据传输量降低 50%
- Tensor Core 吞吐翻倍：H100 FP8 Tensor Core 性能是 FP16 的 2 倍

```python
@T.prim_func
def fp8_gemm_deepseek(
    A: T.Tensor((M, K), "float8_e4m3"),  # FP8 激活
    B: T.Tensor((K, N), "float8_e4m3"),  # FP8 权重
    C: T.Tensor((M, N), "bfloat16"),     # BF16 输出
    scale_A: T.Tensor((M, 1), "float32"),  # per-token 缩放因子
    scale_B: T.Tensor((1, N), "float32"),  # per-channel 缩放因子
):
    A_shared = T.alloc_shared((3, BM, BK), "float8_e4m3")
    B_shared = T.alloc_shared((3, BK, BN), "float8_e4m3")
    acc = T.alloc_fragment((TM, TN), "float32")

    for by, bx in T.grid(M // BM, N // BN):
        T.clear(acc)

        # 三 Buffer Pipeline
        for stage in range(2):
            for i, j in T.Parallel(BM, BK):
                A_shared[stage, i, j] = A[by * BM + i, stage * BK + j]
            for i, j in T.Parallel(BK, BN):
                B_shared[stage, i, j] = B[stage * BK + i, bx * BN + j]

        for k in range(K // BK):
            stage = k % 3

            # FP8 Tensor Core GEMM
            for kk in range(BK):
                for i, j in T.serial(TM, TN):
                    acc[i, j] += A_shared[stage, ..., kk].astype("float32") * \
                                 B_shared[stage, kk, ...].astype("float32")

            # 异步加载
            future_k = k + 2
            if future_k < K // BK:
                future_stage = future_k % 3
                for i, j in T.Parallel(BM, BK):
                    A_shared[future_stage, i, j] = A[by * BM + i, future_k * BK + j]
                for i, j in T.Parallel(BK, BN):
                    B_shared[future_stage, i, j] = B[future_k * BK + i, bx * BN + j]

        # 反量化：乘以缩放因子
        for i, j in T.serial(TM, TN):
            global_i = by * BM + warp_m * WARP_M + tid_m * TM + i
            global_j = bx * BN + warp_n * WARP_N + tid_n * TN + j
            acc[i, j] *= scale_A[global_i, 0] * scale_B[0, global_j]

        # 写回
        for i, j in T.Parallel(TM, TN):
            C[..., ...] = acc[i, j].astype("bfloat16")
```

**2. 混合精度策略**

| 组件 | 精度 | 原因 |
|------|------|------|
| 权重存储 | FP8 (E4M3) | 减少显存占用 2× |
| 激活值 | FP8 (E4M3) | 减少带宽压力 2× |
| 累加器 | FP32 | 保持数值精度 |
| Softmax | FP32 | 避免数值溢出 |
| LayerNorm | FP32 | 保持精度 |
| 输出 | BF16 | 兼容下游层 |

**3. 量化粒度**

DeepSeek-V3 使用 **per-token × per-channel** 的细粒度量化：

```
scale_A: [M, 1]     ← 每个 token 一个缩放因子
scale_B: [1, N]     ← 每个 channel 一个缩放因子

量化:   A_q = round(A / scale_A)
反量化: A = A_q * scale_A

GEMM:   C = A_q × B_q × (scale_A ⊗ scale_B)
```

| 量化粒度 | 精度影响 | 实现复杂度 | 推荐度 |
|----------|----------|-----------|--------|
| per-tensor | 差 | 简单 | 不推荐 |
| per-channel | 一般 | 中等 | 基线 |
| per-token×per-channel | 好 | 复杂 | 推荐 |
| per-group (128) | 最好 | 最复杂 | 最优 |

**4. MoE 路由优化**

```python
@T.prim_func
def moe_router_deepseek(
    hidden: T.Tensor((M, D), "bfloat16"),      # M tokens, D 维
    router_weight: T.Tensor((D, E), "bfloat16"), # E = num_experts
    topk_indices: T.Tensor((M, K), "int32"),    # K = top_k
    topk_weights: T.Tensor((M, K), "bfloat16"), # 路由权重
):
    # 1. 计算路由分数
    scores = T.alloc_fragment((M, E), "float32")
    T.gemm(hidden, router_weight, scores)

    # 2. Softmax
    T.softmax(scores, dim=1)

    # 3. Top-K 选择（K=8）
    T.topk(scores, topk_indices, topk_weights, k=8)
```

### 5.9.4 DeepSeek-V3 性能数据

| 优化技术 | 性能提升 | 适用场景 |
|----------|----------|----------|
| FP8 量化 | 2× vs FP16 | 所有 GEMM |
| 细粒度量化 | +15% 精度 vs per-tensor | 所有 GEMM |
| MoE 路由优化 | 1.3× | MoE 层 |
| 流水线并行 | 1.5× | 长序列 |
| Kernel Fusion | 1.2-1.5× | 所有层 |
| Expert 并行 | 线性扩展 | 多 GPU |
| **总体** | **~4× vs baseline** | **端到端推理** |

---

## 5.10 性能对比与基准测试

<div data-component="GEMMPerformanceComparison"></div>

### 5.10.1 TileLang vs cuBLAS vs Triton

在 A100-80GB 上的性能对比（FP16，M=N=K=4096）：

| 实现 | 延迟 (μs) | TFLOPS | 占峰值比例 | 代码行数 |
|------|-----------|--------|-----------|----------|
| cuBLAS | 156 | 279 | 89.4% | N/A (库) |
| TileLang (手调优) | 162 | 269 | 86.2% | ~80 |
| TileLang (自动调优) | 168 | 259 | 83.0% | ~50 |
| Triton | 185 | 233 | 74.7% | ~60 |
| 手写 CUDA | 170 | 262 | 84.0% | ~500 |
| 朴素 CUDA | 892 | 38 | 12.2% | ~50 |

### 5.10.2 不同矩阵规模对比

| M×N×K | cuBLAS | TileLang | 比率 | TFLOPS (TileLang) |
|-------|--------|----------|------|-------------------|
| 64×64×64 | 2.1 μs | 2.8 μs | 0.75 | 0.19 |
| 128×128×128 | 3.2 μs | 3.8 μs | 0.84 | 1.11 |
| 256×256×256 | 8.5 μs | 9.1 μs | 0.93 | 3.71 |
| 512×512×512 | 22 μs | 24 μs | 0.92 | 11.1 |
| 1024×1024×1024 | 42 μs | 45 μs | 0.93 | 47.5 |
| 2048×2048×2048 | 98 μs | 103 μs | 0.95 | 166.3 |
| 4096×4096×4096 | 156 μs | 162 μs | 0.96 | 269.0 |
| 8192×8192×8192 | 1.2 ms | 1.25 ms | 0.96 | 878.8 |

### 5.10.3 不同精度对比

| 精度 | cuBLAS TFLOPS | TileLang TFLOPS | 峰值 TFLOPS | 效率 |
|------|--------------|-----------------|-------------|------|
| FP32 | 19.2 | 18.5 | 19.5 | 94.9% |
| TF32 | 155 | 148 | 156 | 94.9% |
| FP16 | 279 | 269 | 312 | 86.2% |
| BF16 | 278 | 267 | 312 | 85.6% |
| FP8 | 540 | 510 | 624 | 81.7% |
| INT8 | 520 | 490 | 624 | 78.5% |

### 5.10.4 不同 GPU 对比

| GPU | TileLang TFLOPS | cuBLAS TFLOPS | 比率 | 峰值 |
|-----|-----------------|---------------|------|------|
| V100 (FP16) | 108 | 125 | 0.86 | 125 |
| A100 (FP16) | 269 | 279 | 0.96 | 312 |
| H100 (FP16) | 850 | 989 | 0.86 | 989 |
| H100 (FP8) | 1600 | 1979 | 0.81 | 1979 |

> [!TIP]
> TileLang 在大矩阵上可以达到 cuBLAS 性能的 **90-96%**，而代码量仅为 CUDA 实现的 **1/5 到 1/10**。这使得 TileLang 成为快速原型开发和自定义算子实现的理想选择。

---

## 5.11 性能分析与调试工具

### 5.11.1 Nsight Compute 分析

```bash
# 使用 Nsight Compute 分析 GEMM kernel
ncu --set full python gemm_benchmark.py

# 关注的关键指标：
# - SM Occupancy: 目标 > 70%
# - Shared Memory Bank Conflict: 目标 < 5%
# - Tensor Core Utilization: 目标 > 80%
# - Global Memory Throughput: 目标 > 80% 峰值
# - L2 Cache Hit Rate: 目标 > 50%
# - Warp Execution Efficiency: 目标 > 90%
```

### 5.11.2 性能瓶颈诊断表

| 症状 | 可能原因 | 诊断方法 | 解决方案 |
|------|----------|----------|----------|
| 低 SM Occupancy | 寄存器/Shared Memory 压力 | ncu --metrics sm__warps_active | 减小 Tile 大小 |
| 高 Bank Conflict | 布局不优 | ncu --metrics l1tex__data_bank_conflicts | Swizzled Layout |
| 低 Tensor Core 利用 | 数据类型/布局不匹配 | ncu --metrics sm__inst_executed_pipe_tensor | 使用 WMMA API |
| 低带宽利用率 | 访存模式差 | ncu --metrics dram__throughput | 优化数据搬运 |
| 计算与访存不重叠 | 无 Pipelining | ncu --metrics smsp__warp_issue_stalled_mio | 多 Buffer |
| 高 L2 Miss | Tile 大小不优 | ncu --metrics l2__t_sectors | 调整 Tile 大小 |
| 寄存器溢出 | 累加器太大 | ncu --metrics lts__t_sectors | 减小 TM×TN |

### 5.11.3 TileLang 性能分析代码

```python
import tilelang
from tilelang.profiler import get_all_profiler

# 获取详细的性能分析
profiler = get_all_profiler(kernel)

# 查看每个 Pass 的执行时间
for name, time_us in profiler.pass_times.items():
    print(f"{name}: {time_us:.2f} μs")

# 查看内存使用
print(f"Shared Memory: {profiler.shared_memory_bytes} B")
print(f"Registers: {profiler.register_count}")
print(f"SM Occupancy: {profiler.occupancy:.1%}")
```

---

## 5.12 高级主题

### 5.12.1 Epilogue Fusion

在实际应用中，GEMM 通常需要与后续的逐元素操作融合（如 Bias Add、ReLU、GELU）：

```python
@T.prim_func
def fused_gemm_bias_relu(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    bias: T.Tensor((N,), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    # ... GEMM 计算 ...

    # Epilogue Fusion：在写回前融合 Bias + ReLU
    for i, j in T.Parallel(TM, TN):
        val = acc[i, j] + bias[global_j].astype("float32")
        val = T.max(val, T.float32(0))  # ReLU
        C[global_i, global_j] = val.astype("float16")
```

### 5.12.2 分组 GEMM (Grouped GEMM)

MoE 模型需要同时计算多个小 GEMM，分组 GEMM 可以高效处理这种场景：

```python
@T.prim_func
def grouped_gemm(
    A: T.Tensor((num_groups, M, K), "float16"),
    B: T.Tensor((num_groups, K, N), "float16"),
    C: T.Tensor((num_groups, M, N), "float16"),
):
    for g in range(num_groups):
        # 每个组独立计算一个 GEMM
        for by, bx in T.grid(M // BM, N // BN):
            # ... 标准 GEMM 逻辑 ...
```

### 5.12.3 Stream-K 调度

Stream-K 是一种更高效的 GEMM 调度策略，可以更好地利用 GPU 上的所有 SM：

```
传统调度：
  SM 0: Tile(0,0), Tile(0,1), ..., Tile(0,n)
  SM 1: Tile(1,0), Tile(1,1), ..., Tile(1,n)
  ...
  问题：最后一波可能有 SM 空闲

Stream-K 调度：
  SM 0: K-slice(0), K-slice(1), ..., K-slice(m)
  SM 1: K-slice(m+1), ..., K-slice(2m)
  ...
  优势：所有 SM 同时完成，无尾部效应
```

---

## 5.13 Summary

✅ **关键要点**：

1. GEMM 优化是一个**逐步叠加**的过程：Tiling → Shared Memory → Bank Conflict 消除 → Software Pipelining → Warp 级优化 → Tensor Core
2. `T.alloc_shared` 和 `T.alloc_fragment` 是管理片上存储的核心原语
3. Software Pipelining 通过**多 Buffer**实现计算与访存重叠
4. 自动调优可以搜索最优的分块和流水线参数
5. DeepSeek-V3 的 GEMM 优化精髓在于 **FP8 量化** 和 **MoE 路由优化**

🎯 **性能目标**：

| 级别 | 指标 | 对应技术 |
|------|------|----------|
| 入门 | 达到 cuBLAS 性能的 50% | Tiling + Shared Memory |
| 进阶 | 达到 cuBLAS 性能的 80% | + Pipeline + Warp 优化 |
| 优秀 | 达到 cuBLAS 性能的 90%+ | + Tensor Core + 调优 |
| 极致 | 达到 cuBLAS 性能的 95%+ | + 手动微调 + 特定硬件优化 |

📊 **优化路径图**：

```
朴素实现 (0.14%)
    │
    ▼ Tiling
Tiled 实现 (5.8%)
    │
    ▼ Shared Memory + Bank Conflict
SM 优化 (12%)
    │
    ▼ Software Pipelining
Pipeline 优化 (25%)
    │
    ▼ Warp 级优化
Warp 优化 (40%)
    │
    ▼ Tensor Core
TC 优化 (70%)
    │
    ▼ 自动调优
调优后 (85-96%)
```

---

## 5.14 Exercises

### 练习 1：基础 Tiling GEMM
实现一个 BM=64, BN=64, BK=16 的 Tiled GEMM，验证正确性并测量性能。

**要求**：
- 使用 `T.alloc_shared` 分配 Shared Memory
- 使用 `T.alloc_fragment` 分配累加器
- 与 `torch.matmul` 对比验证正确性

### 练习 2：Bank Conflict 分析
使用 Nsight Compute 分析你的 GEMM kernel，识别并消除 Bank Conflict。

**要求**：
- 运行 `ncu` 分析 Shared Memory 访问模式
- 实现 Padding 和 Swizzled 两种方案
- 对比两种方案的性能

### 练习 3：Software Pipelining
为练习 1 的 kernel 添加双 Buffer Software Pipelining，测量性能提升。

**要求**：
- 使用双 Buffer 或三 Buffer
- 实现 Prologue / Main Loop / Epilogue 结构
- 对比有无 Pipeline 的性能差异

### 练习 4：自动调优
使用 TileLang 的自动调优框架，在 4096×4096×4096 规模上找到最优参数配置。

**要求**：
- 定义合理的搜索空间
- 实现参数化 GEMM 模板
- 找到 Top-5 配置并分析规律

### 练习 5：FP8 GEMM
实现一个 FP8 输入、FP16 输出的 GEMM，包含反量化步骤。

**要求**：
- 使用 FP8 (E4M3) 数据类型
- 实现 per-token 量化
- 对比 FP8 vs FP16 的性能和精度

### 练习 6：Epilogue Fusion
实现一个融合了 Bias + GELU 的 GEMM kernel。

**要求**：
- 在 Epilogue 阶段融合 Bias 加法和 GELU 激活
- 对比 Fusion 前后的性能

---

## 5.15 Thinking Questions

1. **为什么 GEMM 的 Tiling 通常使用二维分块而不是一维？**
   提示：从数据复用和线程协作的角度分析。一维分块会如何影响 A 和 B 的复用率？

2. **Shared Memory 的容量如何影响 Tile 大小的选择？如果 Shared Memory 容量翻倍，是否应该让 Tile 面积也翻倍？**
   提示：考虑 A 和 B 两个 Tile 的存储需求，以及 Pipeline 阶数的影响。

3. **Software Pipelining 的 Stage 数量是否越多越好？**
   提示：从寄存器压力、Shared Memory 消耗和延迟隐藏的角度分析。什么时候增加 Stage 不再有收益？

4. **为什么 DeepSeek-V3 选择 FP8 而不是 INT8 进行量化？**
   提示：从硬件支持（Tensor Core）、数值范围、量化/反量化开销的角度分析。

5. **TileLang 的自动调优与 Triton 的自动调优有何异同？**
   提示：从搜索空间定义、搜索策略、编译开销和性能可预测性的角度分析。

6. **在 MoE 模型中，为什么分组 GEMM（Grouped GEMM）比逐个计算多个小 GEMM 更高效？**
   提示：从 GPU 利用率和调度开销的角度分析。

---

## 5.16 GEMM 优化 Checklist

在实际开发中，可以按照以下清单逐步优化 GEMM kernel：

### 5.16.1 第一阶段：正确性验证

```markdown
- [ ] 朴素实现通过正确性验证（与 torch.matmul 对比）
- [ ] 使用 FP32 累加器避免精度问题
- [ ] 边界条件处理（M/N/K 不是 Tile 大小的整数倍）
- [ ] 数值稳定性检查（大数值、小数值、零值）
```

**边界条件处理示例**：

```python
@T.prim_func
def gemm_with_boundary(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    # 使用 T.where 处理边界
    for by, bx in T.grid(T.ceildiv(M, BM), T.ceildiv(N, BN)):
        for k in range(T.ceildiv(K, BK)):
            for i, j in T.Parallel(BM, BK):
                # 边界检查：超出矩阵范围的位置填零
                global_i = by * BM + i
                global_j = k * BK + j
                A_shared[i, j] = T.where(
                    T.and(global_i < M, global_j < K),
                    A[global_i, global_j],
                    T.float16(0)
                )
```

### 5.16.2 第二阶段：基础性能优化

```markdown
- [ ] 实现 Tiling（BM, BN, BK 选择合理）
- [ ] 使用 T.alloc_shared 分配 Shared Memory
- [ ] 使用 T.alloc_fragment 分配累加器
- [ ] 确保 Shared Memory 使用量在硬件限制内
- [ ] 验证 SM Occupancy > 50%
```

**Shared Memory 使用量计算脚本**：

```python
def calc_shared_memory(BM, BN, BK, num_stages, dtype_bytes=2):
    A_bytes = num_stages * BM * BK * dtype_bytes
    B_bytes = num_stages * BK * BN * dtype_bytes
    total = A_bytes + B_bytes
    print(f"A_shared: {A_bytes / 1024:.1f} KB")
    print(f"B_shared: {B_bytes / 1024:.1f} KB")
    print(f"Total: {total / 1024:.1f} KB")
    print(f"SM Occupancy: {'OK' if total <= 164 * 1024 else 'OVER LIMIT'}")
    return total

# 示例
calc_shared_memory(128, 128, 32, 3)  # 48 KB
calc_shared_memory(256, 256, 64, 3)  # 192 KB → 超限！
```

### 5.16.3 第三阶段：高级优化

```markdown
- [ ] 消除 Bank Conflict（使用 Swizzled Layout 或 Padding）
- [ ] 实现 Software Pipelining（2-3 个 Stage）
- [ ] 使用异步内存拷贝（cp.async）
- [ ] Warp 级优化（合理的 Warp 布局）
- [ ] 利用 Tensor Core（WMMA 指令）
```

### 5.16.4 第四阶段：极致优化

```markdown
- [ ] 自动调优找到最优参数
- [ ] Epilogue Fusion（Bias、激活函数等）
- [ ] 寄存器压力优化
- [ ] 指令调度优化
- [ ] 特定硬件优化（如 H100 的 TMA）
```

---

## 5.17 Common Pitfalls（常见陷阱）

### 5.17.1 精度陷阱

```python
# ❌ 错误：使用 FP16 累加器
acc = T.alloc_fragment((TM, TN), "float16")
for k in range(K):
    acc[i, j] += A[i, k] * B[k, j]  # FP16 累加会丢失精度！

# ✅ 正确：使用 FP32 累加器
acc = T.alloc_fragment((TM, TN), "float32")
for k in range(K):
    acc[i, j] += A[i, k].astype("float32") * B[k, j].astype("float32")
```

**精度对比**：

| 累加器精度 | 最大绝对误差 | 最大相对误差 | 适用场景 |
|-----------|-------------|-------------|----------|
| FP16 | 0.01 | 1% | 不推荐 |
| FP32 | 0.0001 | 0.01% | 推荐 |
| FP64 | 0.000001 | 0.0001% | 科学计算 |

### 5.17.2 同步陷阱

```python
# ❌ 错误：忘记 T.syncthreads()
for i, j in T.Parallel(BM, BK):
    A_shared[i, j] = A[...]  # 写入 Shared Memory
# 缺少 T.syncthreads()
for k in range(BK):
    val = A_shared[i, k]  # 可能读到未更新的数据！

# ✅ 正确：添加同步
for i, j in T.Parallel(BM, BK):
    A_shared[i, j] = A[...]
T.syncthreads()  # 确保所有线程写入完成
for k in range(BK):
    val = A_shared[i, k]
```

### 5.17.3 内存越界陷阱

```python
# ❌ 错误：不检查边界
C[by * BM + i, bx * BN + j] = acc[i, j]
# 当 M 不是 BM 的整数倍时，越界！

# ✅ 正确：使用 T.where
global_i = by * BM + i
global_j = bx * BN + j
C[global_i, global_j] = T.where(
    T.and(global_i < M, global_j < N),
    acc[i, j].astype("float16"),
    T.float16(0)
)
```

### 5.17.4 性能陷阱

```python
# ❌ 陷阱 1：Tile 太小
BM, BN, BK = 16, 16, 16  # 数据复用率低

# ❌ 陷阱 2：Tile 太大
BM, BN, BK = 256, 256, 128  # Shared Memory 超限，Occupancy 低

# ❌ 陷阱 3：没有 Pipeline
# 串行执行 Load → Compute → Load → Compute

# ❌ 陷阱 4：Bank Conflict 未消除
A_shared = T.alloc_shared((BM, BK), "float16")  # 无 Swizzled
```

---

## 5.18 GEMM 优化速查表

### 5.18.1 参数选择速查

| 场景 | BM | BN | BK | TM | TN | Stages | Warps |
|------|----|----|----|----|----|--------|-------|
| 小矩阵 (≤512) | 64 | 64 | 16 | 4 | 4 | 2 | 2 |
| 中矩阵 (512-2048) | 128 | 128 | 32 | 8 | 8 | 3 | 4 |
| 大矩阵 (≥2048) | 128 | 256 | 32 | 8 | 8 | 3 | 4 |
| FP8 GEMM | 128 | 128 | 64 | 8 | 8 | 3 | 4 |
| MoE 小 GEMM | 64 | 64 | 32 | 4 | 4 | 2 | 2 |

### 5.18.2 性能指标速查

| 指标 | 优秀 | 良好 | 一般 | 差 |
|------|------|------|------|-----|
| 峰值效率 | >90% | 80-90% | 60-80% | <60% |
| SM Occupancy | >75% | 50-75% | 25-50% | <25% |
| Bank Conflict Rate | <2% | 2-5% | 5-10% | >10% |
| Tensor Core 利用率 | >80% | 60-80% | 40-60% | <40% |
| L2 Cache Hit Rate | >60% | 40-60% | 20-40% | <20% |

### 5.18.3 常用命令速查

```bash
# 编译 TileLang kernel
python gemm.py  # 自动编译和运行

# 性能分析
ncu --set full python gemm_benchmark.py

# 查看 PTX 汇编
nvcc -ptx gemm.cu -o gemm.ptx

# 查看 Shared Memory 使用
cuobjdump -res-usage gemm.o

# TileLang 自动调优
python -m tilelang.autotune gemm_template.py --space search_space.json
```

---

## 5.19 从 GEMM 到 Attention：优化思想的迁移

GEMM 的优化思想可以迁移到其他 GPU 算子中：

| 优化技术 | GEMM 中的应用 | Attention 中的应用 |
|----------|--------------|-------------------|
| Tiling | 矩阵分块 | Q/K/V 分块 |
| Shared Memory | 缓存 A/B Tile | 缓存 Q/K/V Tile |
| Software Pipelining | 重叠 Load/Compute | 重叠 Q×K 和 Softmax |
| Bank Conflict 消除 | Swizzled Layout | Swizzled Layout |
| Warp 级优化 | Warp 协作计算 | Warp 协作归约 |
| Epilogue Fusion | Bias + 激活 | Softmax + Dropout |

下一章的 FlashAttention 实现将展示这些思想的迁移应用。

---

## 5.16 Extension Reading

1. **CUTLASS**: NVIDIA 的高性能 CUDA 模板库，GEMM 优化的工业标准
   - GitHub: github.com/NVIDIA/cutlass
   - 文档：深度解析 Tiling、Pipeline、Epilogue Fusion

2. **Triton**: OpenAI 的 GPU 编程语言，TileLang 的重要参考
   - GitHub: github.com/triton-lang/triton
   - 论文：Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations

3. **FlashAttention**: IO-aware 的 Attention 实现，GEMM 优化思想的延伸
   - 论文：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
   - FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

4. **DeepSeek-V3 技术报告**: 671B MoE 模型的工程优化细节
   - FP8 量化策略、MoE 路由优化、Kernel Fusion

5. **cuBLAS 文档**: NVIDIA 官方 GEMM 库的使用指南
   - 最佳实践、性能调优、新特性

6. **Roofline Model**: UC Berkeley 的性能分析模型
   - 论文：Roofline: An Insightful Visual Performance Model for Multicore Architectures

---

## 5.17 Next Chapter Preview

在下一章中，我们将深入探讨 **Layout 推理机制**，了解 TileLang 如何自动推导最优的数据布局，以及如何消除 Bank Conflict。这是理解 TileLang 编译器核心能力的关键章节。

> **Chapter 6: Layout 推理机制与 Bank Conflict 消除**
>
> - Layout 的概念：数据在硬件线程中的分布方式
> - Strict Inference vs Common Inference 两种模式
> - Swizzled Layout 自动推导算法
> - Layout 传播规则：从 producer 到 consumer
> - T.Layout 注解的使用方法
> - 与 Triton 的 implicit layout 对比
> - 源码走读：tilelang/transforms/ 相关 Pass

---

## 5.20 本章总结图

```
┌─────────────────────────────────────────────────────────────┐
│                    GEMM 优化全景图                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  朴素    │───▶│  Tiling  │───▶│  Shared  │              │
│  │  实现    │    │  优化    │    │  Memory  │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│   0.14%            5.8%            12%                      │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  Bank    │───▶│ Software │───▶│  Warp    │              │
│  │  Conflict│    │ Pipeline │    │  级优化  │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│   15%              25%             40%                      │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  Tensor  │───▶│  自动    │───▶│  工业级  │              │
│  │  Core    │    │  调优    │    │  实现    │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│   70%              85%             96%                      │
│                                                             │
│  关键原语：T.alloc_shared, T.alloc_fragment, T.syncthreads │
│  关键技术：Tiling, Pipeline, Swizzled, WMMA, Auto-tune     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**下一步行动**：

1. 从朴素实现开始，逐步添加优化
2. 每一步都验证正确性后再进行下一步
3. 使用 Nsight Compute 分析性能瓶颈
4. 参考 cuBLAS 的性能作为目标
5. 阅读下一章了解 Layout 推理机制

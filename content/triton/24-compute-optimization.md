# Chapter 24: 计算优化与指令级并行

> **学习目标**：
> - 掌握 Tensor Core 利用与 tl.dot 优化
> - 理解循环展开（unroll）与指令级并行（ILP）
> - 掌握异步执行与流水线重叠策略
> - 了解 Triton 的 compute-sanitizer 集成

---

## 24.1 计算优化概览

### 24.1.1 GPU 计算模型

GPU 的计算能力来源于其大规模并行架构。以 NVIDIA A100 为例，它拥有 6912 个 FP32 CUDA 核心和 432 个 Tensor Core，理论峰值算力可达 19.5 TFLOPS（FP32）和 312 TFLOPS（FP16 Tensor Core）。

```
GPU 计算资源层次
┌──────────────────────────────────────────────────┐
│                    SM (Streaming Multiprocessor)   │
│  ┌─────────────┐  ┌──────────────┐               │
│  │ CUDA Cores  │  │ Tensor Core  │               │
│  │ (FP32/INT32)│  │ (FP16/TF32/  │               │
│  │             │  │  BF16/FP8)   │               │
│  └─────────────┘  └──────────────┘               │
│  ┌─────────────┐  ┌──────────────┐               │
│  │ Load/Store  │  │ Special Func │               │
│  │  Units      │  │  Units (SFU) │               │
│  └─────────────┘  └──────────────┘               │
│  ┌──────────────────────────────────────┐        │
│  │           Register File              │        │
│  │         (256 KB per SM)              │        │
│  └──────────────────────────────────────┘        │
│  ┌──────────────────────────────────────┐        │
│  │         Shared Memory                │        │
│  │         (164 KB per SM)              │        │
│  └──────────────────────────────────────┘        │
└──────────────────────────────────────────────────┘
```

### 24.1.2 计算优化的三个维度

计算优化通常从三个维度入手：

| 维度 | 优化目标 | 关键技术 |
|------|---------|---------|
| **吞吐量** | 最大化每秒执行的操作数 | Tensor Core 利用、循环展开、ILP |
| **延迟** | 最小化单次操作的等待时间 | 异步执行、流水线重叠 |
| **效率** | 最大化实际算力/理论峰值 | 精度选择、Roofline 调优 |

```
优化层次金字塔

        ▲ Roofline 调优
       ▲▲▲ 计算强度最大化
      ▲▲▲▲▲ Tensor Core 利用
     ▲▲▲▲▲▲▲ 循环展开 & ILP
    ▲▲▲▲▲▲▲▲▲ 异步执行 & 流水线
   ▲▲▲▲▲▲▲▲▲▲▲ 精度-性能权衡
  ▲▲▲▲▲▲▲▲▲▲▲▲▲ 调试 & 正确性验证
 ────────────────────────────────
  从底层到高层，逐层优化
```

---

## 24.2 Tensor Core 利用与 tl.dot 优化

### 24.2.1 Tensor Core 基础

Tensor Core 是 NVIDIA GPU 中专门用于矩阵乘加运算的硬件单元。每个 Tensor Core 可以在一个时钟周期内完成一个 4×4×4 的矩阵乘加操作（D = A × B + C）。

```
Tensor Core 矩阵乘加操作

  A (4×4)          B (4×4)          D (4×4)
┌──────────┐    ┌──────────┐    ┌──────────┐
│ a00 a01  │    │ b00 b01  │    │ d00 d01  │
│ a10 a11  │  × │ b10 b11  │  + │ d10 d11  │
│ ...  ... │    │ ...  ... │    │ ...  ... │
│ a30 a31  │    │ b30 b31  │    │ d30 d31  │
└──────────┘    └──────────┘    └──────────┘

  操作：D = A × B + C
  精度：A, B 通常为 FP16/BF16/FP8，C 和 D 为 FP32
  吞吐量：比 CUDA Core 高 8 倍（FP16 精度下）
```

不同 GPU 架构的 Tensor Core 支持：

| GPU 架构 | Tensor Core 版本 | 支持精度 | 矩阵尺寸 |
|---------|-----------------|---------|---------|
| Volta (V100) | 1.0 | FP16 | 4×4×4 |
| Turing (T4) | 2.0 | INT8, FP16 | 4×4×4 |
| Ampere (A100) | 3.0 | TF32, BF16, FP16, FP8, INT8 | 16×8×16 |
| Hopper (H100) | 4.0 | FP8, BF16, FP16, FP32 | 16×8×16 |

### 24.2.2 tl.dot 的 Tensor Core 映射

在 Triton 中，`tl.dot` 是触发 Tensor Core 的核心操作。编译器会自动将 `tl.dot` 映射到对应的 MMA（Matrix Multiply-Accumulate）指令。

```python
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 获取 program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 计算 tile 起始偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 初始化累加器
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K 维度循环
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # 计算当前 K 偏移
        offs_k_cur = k * BLOCK_K + offs_k

        # 加载 A 和 B 的 tile
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k_cur[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k_cur[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0)

        # tl.dot 触发 Tensor Core 计算
        # A (FP16/BF16) × B (FP16/BF16) → FP32 累加
        accumulator += tl.dot(a, b)

    # 存储结果
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

`tl.dot` 在编译后会生成如下 PTX 指令（以 A100 为例）：

```ptx
// tl.dot → mma.sync.aligned.m16n8k16 指令
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
    {a0, a1, a2, a3, a4, a5, a6, a7},
    {b0, b1},
    {c0, c1, c2, c3};
```

### 24.2.3 allow_tf32 参数控制

TF32（TensorFloat-32）是 Ampere 架构引入的格式，它使用 10 位指数和 13 位尾数（共 19 位），在保持接近 FP32 精度的同时获得 FP16 的速度。

```python
import torch

# 默认行为：allow_tf32 = True（PyTorch 1.12+）
# 允许 cuBLAS/cuDNN 使用 TF32 加速
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 禁用 TF32，使用完整 FP32 精度
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

TF32 精度影响分析：

| 操作 | FP32 参考 | TF32 误差 | 能否接受 |
|------|----------|----------|---------|
| 矩阵乘法 (M=1024) | 1.0 | ~0.1% | 通常可接受 |
| 矩阵乘法 (M=8192) | 1.0 | ~1.0% | 需谨慎 |
| Softmax + 归约 | 1.0 | ~0.5% | 取决于场景 |
| 物理模拟 | 1.0 | ~5.0% | 通常不可接受 |

### 24.2.4 FP8/BF16/FP16 精度选择

不同精度格式在 Tensor Core 上的性能差异：

| 精度格式 | 位宽 | A100 TFLOPS | H100 TFLOPS | 相对精度 | 典型应用 |
|---------|------|-------------|-------------|---------|---------|
| FP32 | 32 | 19.5 | 67 | 1.0× | 参考/累加 |
| TF32 | 19 | 156 | 756 | ~0.9× | 训练默认 |
| BF16 | 16 | 312 | 1100 | ~0.9× | 训练/推理 |
| FP16 | 16 | 312 | 1100 | ~0.9× | 推理 |
| FP8 (E4M3) | 8 | — | 1800 | ~0.7× | 大规模推理 |
| FP8 (E5M2) | 8 | — | 1800 | ~0.5× | 梯度/低精度 |
| INT8 | 8 | 624 | 2200 | 量化 | 推理/量化 |

```python
@triton.jit
def gemm_fp8_kernel(
    a_ptr, b_ptr, c_ptr,
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

    # FP8 累加器
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k_cur = k * BLOCK_K + offs_k

        # 加载 FP8 数据
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k_cur[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k_cur[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0)

        # FP8 × FP8 → FP32 累加（Hopper 架构）
        accumulator += tl.dot(a, b)

    c = accumulator.to(tl.float16)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

### 24.2.5 Tensor Core 对齐要求

Tensor Core 对矩阵维度有严格的对齐要求。不满足对齐条件会导致性能下降或编译失败。

```
Tensor Core 维度对齐要求（A100）

  MMA 指令: mma.sync.aligned.m16n8k16

  ┌─────────────────────────────────────────┐
  │  A 矩阵 (M × K)                         │
  │  ├─ M 必须是 16 的倍数                    │
  │  ├─ K 必须是 16 的倍数                    │
  │  └─ 内存布局: row-major                  │
  ├─────────────────────────────────────────┤
  │  B 矩阵 (K × N)                         │
  │  ├─ K 必须是 16 的倍数                    │
  │  ├─ N 必须是 8 的倍数                     │
  │  └─ 内存布局: col-major                  │
  ├─────────────────────────────────────────┤
  │  C/D 矩阵 (M × N)                       │
  │  ├─ M 必须是 16 的倍数                    │
  │  └─ N 必须是 8 的倍数                     │
  └─────────────────────────────────────────┘
```

```python
@triton.jit
def gemm_with_padding(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    自动处理 Tensor Core 对齐的 GEMM 实现
    BLOCK_M 和 BLOCK_N 通常是 16 的倍数
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 确保 BLOCK_M 是 16 的倍数，BLOCK_N 是 8 的倍数
    # Triton 编译器会自动填充 padding
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k_cur = k * BLOCK_K + offs_k

        # 使用 mask 处理边界情况
        a_mask = (offs_m[:, None] < M) & (offs_k_cur[None, :] < K)
        b_mask = (offs_k_cur[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k_cur[None, :] * stride_ak,
                     mask=a_mask, other=0.0)
        b = tl.load(b_ptr + offs_k_cur[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                     mask=b_mask, other=0.0)

        accumulator += tl.dot(a, b)

    # 输出时也需要 mask
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             accumulator, mask=c_mask)
```

---

## 24.3 循环展开与编译器优化

### 24.3.1 循环展开的概念

循环展开（Loop Unrolling）是将循环体复制多次以减少循环开销的技术。在 GPU 上，循环展开可以：

1. 减少分支指令开销
2. 增加指令级并行（ILP）
3. 为编译器提供更大的优化空间

```
循环展开示例

  原始循环（展开因子=1）：
  for i in range(4):
      a[i] = b[i] + c[i]

  展开后（展开因子=2）：
  for i in range(0, 4, 2):
      a[i] = b[i] + c[i]
      a[i+1] = b[i+1] + c[i+1]

  展开后（展开因子=4）：
  a[0] = b[0] + c[0]
  a[1] = b[1] + c[1]
  a[2] = b[2] + c[2]
  a[3] = b[3] + c[3]
```

### 24.3.2 Triton 编译器自动展开

Triton 编译器会根据目标架构自动进行循环展开。展开因子由 `num_warps` 和循环体大小决定。

```python
@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton 会自动对简单循环进行展开
    这里的 arange 操作会被展开
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 加载
    x = tl.load(x_ptr + offs, mask=offs < n_elements, other=0.0)
    y = tl.load(y_ptr + offs, mask=offs < n_elements, other=0.0)

    # 计算 - 编译器会自动优化
    output = x + y

    # 存储
    tl.store(output_ptr + offs, output, mask=offs < n_elements)
```

### 24.3.3 手动展开指导

在某些情况下，手动展开可以带来更好的性能。

```python
@triton.jit
def reduce_with_unroll(
    x_ptr, output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    UNROLL_FACTOR: tl.constexpr,
):
    """
    手动展开的归约 kernel
    UNROLL_FACTOR 控制展开程度
    """
    pid = tl.program_id(0)

    # 每个 program 处理 UNROLL_FACTOR 个 BLOCK
    base_offset = pid * BLOCK_SIZE * UNROLL_FACTOR

    # 初始化
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # 手动展开循环
    for i in range(UNROLL_FACTOR):
        offset = base_offset + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < N
        x = tl.load(x_ptr + offset, mask=mask, other=0.0)
        acc += x

    # 最终归约
    result = tl.sum(acc)
    tl.store(output_ptr + pid, result)
```

```python
@triton.jit
def gemm_unrolled_k(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_UNROLL: tl.constexpr,
):
    """
    K 维度手动展开的 GEMM kernel
    通过展开增加 ILP
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 初始化累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K 维度循环，每次处理 K_UNROLL 个 BLOCK_K
    for k_base in range(0, tl.cdiv(K, BLOCK_K * K_UNROLL)):
        # 保存多个累加器以增加 ILP
        acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for sub_k in range(K_UNROLL):
            k = k_base * K_UNROLL + sub_k
            offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

            # 加载 A
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)

            # 加载 B
            b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

            # 交替累加到不同累加器，增加 ILP
            if sub_k % 2 == 0:
                acc0 += tl.dot(a, b)
            else:
                acc1 += tl.dot(a, b)

        acc += acc0 + acc1

    # 存储结果
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

### 24.3.4 展开因子的选择

展开因子的选择需要权衡多个因素：

| 展开因子 | 优点 | 缺点 | 适用场景 |
|---------|------|------|---------|
| 1 | 代码简单 | 循环开销大 | 基准实现 |
| 2 | 适中 | — | 通用场景 |
| 4 | ILP 高 | 寄存器压力大 | 计算密集型 |
| 8 | ILP 最高 | 可能导致寄存器溢出 | 小 kernel |

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_autotuned_unroll(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    通过 autotune 自动选择最优展开因子
    不同 num_warps 对应不同的 ILP 策略
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k_cur = k * BLOCK_K + offs_k

        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k_cur[None, :] * stride_ak,
                     mask=(offs_m[:, None] < M) & (offs_k_cur[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_k_cur[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                     mask=(offs_k_cur[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

---

## 24.4 指令级并行（ILP）

### 24.4.1 ILP 概念

指令级并行（Instruction-Level Parallelism, ILP）是指单个线程内多条指令同时执行的能力。GPU 虽然是 SIMT 架构，但每个 SM 内的多个执行单元可以同时处理不同类型的指令。

```
ILP 执行模型

  时钟周期 →
  ┌──────────┬──────────┬──────────┬──────────┬──────────┐
  │  Load    │  Load    │  Store   │  FP ADD  │  FP MUL  │  ← 周期 1
  │  操作 A  │  操作 B  │  操作 C  │  操作 D  │  操作 E  │
  ├──────────┼──────────┼──────────┼──────────┼──────────┤
  │  Load    │  Tensor  │  Tensor  │  Load    │  FP ADD  │  ← 周期 2
  │  操作 F  │  Core 1  │  Core 2  │  操作 G  │  操作 H  │
  ├──────────┼──────────┼──────────┼──────────┼──────────┤
  │  Store   │  Load    │  FP MUL  │  Tensor  │  Load    │  ← 周期 3
  │  操作 I  │  操作 J  │  操作 K  │  Core 3  │  操作 L  │
  └──────────┴──────────┴──────────┴──────────┴──────────┘

  不同执行单元可以同时工作，实现 ILP
```

### 24.4.2 Triton 中的 ILP 调度

Triton 编译器会自动进行 ILP 调度，但理解其原理有助于编写更高效的代码。

```python
@triton.jit
def ilp_friendly_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    ILP 友好的 kernel 设计
    尽量让独立操作并行执行
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 加载两个独立的数据源
    a = tl.load(a_ptr + offs, mask=offs < N, other=0.0)
    b = tl.load(b_ptr + offs, mask=offs < N, other=0.0)

    # 这些操作是独立的，可以并行执行（ILP）
    c = a * 2.0 + 1.0        # 操作 1
    d = b * 3.0 - 0.5        # 操作 2（与操作 1 独立）

    # 存储结果
    tl.store(c_ptr + offs, c, mask=offs < N)
    tl.store(d_ptr + offs, d, mask=offs < N)
```

```python
@triton.jit
def multi_accumulator_dot(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_ACCUMULATORS: tl.constexpr,
):
    """
    多累加器 ILP 优化
    使用多个独立累加器增加 Tensor Core 利用率
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 多个独立累加器
    accs = [tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32) for _ in range(NUM_ACCUMULATORS)]

    k_iterations = tl.cdiv(K, BLOCK_K)
    for k in range(0, k_iterations, NUM_ACCUMULATORS):
        offs_k_base = k * BLOCK_K

        # 加载多个 tile
        for i in range(NUM_ACCUMULATORS):
            if k + i < k_iterations:
                offs_k_cur = offs_k_base + i * BLOCK_K + offs_k

                a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k_cur[None, :] * stride_ak
                b_ptrs = b_ptr + offs_k_cur[:, None] * stride_bk + offs_n[None, :] * stride_bn

                a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k_cur[None, :] < K), other=0.0)
                b = tl.load(b_ptrs, mask=(offs_k_cur[:, None] < K) & (offs_n[None, :] < N), other=0.0)

                # 独立的 dot 操作，可以并行执行
                accs[i] += tl.dot(a, b)

    # 合并所有累加器
    acc = accs[0]
    for i in range(1, NUM_ACCUMULATORS):
        acc += accs[i]

    # 存储结果
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

### 24.4.3 操作重叠与依赖分析

```
操作依赖图示例

  ┌─────────┐     ┌─────────┐     ┌─────────┐
  │ Load A  │────→│  A × B  │────→│  累加   │
  └─────────┘     └─────────┘     └─────────┘
       │               ↑               │
       │               │               ↓
  ┌─────────┐     ┌─────────┐     ┌─────────┐
  │ Load B  │────→│  加载   │────→│  存储   │
  └─────────┘     └─────────┘     └─────────┘

  独立路径可以并行：
  - Load A 和 Load B 可以同时执行
  - Load B 和 A×B 可以重叠
  - 累加和存储可以重叠
```

```python
@triton.jit
def overlap_example(
    a_ptr, b_ptr, c_ptr, d_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    展示操作重叠的 kernel
    相邻迭代的操作可以重叠执行
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 第一次加载
    a = tl.load(a_ptr + offs, mask=offs < N, other=0.0)

    # 第一次计算 + 第二次加载（可以重叠）
    c1 = a * 2.0
    b = tl.load(b_ptr + offs, mask=offs < N, other=0.0)

    # 第二次计算 + 第一次存储（可以重叠）
    d1 = b + c1
    tl.store(d_ptr + offs, d1, mask=offs < N)

    # 第三次计算
    c2 = a * 3.0 + b

    tl.store(c_ptr + offs, c2, mask=offs < N)
```

### 24.4.4 ILP 测量与分析

```python
import triton
import triton.language as tl
import torch
import time

def measure_kernel(kernel_func, grid, args, warmup=10, rep=100):
    """测量 kernel 执行时间"""
    # Warmup
    for _ in range(warmup):
        kernel_func[grid](*args)

    # 同步
    torch.cuda.synchronize()

    # 计时
    start = time.time()
    for _ in range(rep):
        kernel_func[grid](*args)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / rep

    return elapsed * 1000  # 返回毫秒

# 对比不同 ILP 策略
@triton.jit
def kernel_low_ilp(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """低 ILP kernel：顺序执行"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < N, other=0.0)
    y = x * 2.0
    z = y + 1.0
    tl.store(out_ptr + offs, z, mask=offs < N)

@triton.jit
def kernel_high_ilp(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    """高 ILP kernel：独立操作并行"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < N, other=0.0)
    # 两个独立操作可以同时执行
    y = x * 2.0
    z = x * 3.0
    result = y + z
    tl.store(out_ptr + offs, result, mask=offs < N)
```

---

## 24.5 异步执行与流水线重叠

### 24.5.1 GPU 异步执行模型

GPU 支持多种异步操作，包括内存拷贝、kernel 执行和事件同步。

```
GPU 异步执行模型

  CPU 线程                GPU 线程
  ─────────             ─────────
  │ 启动 Kernel A │      │          │
  │───────────────│─────→│ 执行 A   │
  │ 启动 Kernel B │      │          │
  │───────────────│─────→│ 执行 B   │
  │ 启动 Kernel C │      │          │
  │───────────────│─────→│ 执行 C   │
  │ 同步等待      │      │          │
  │───────────────│      │ 完成 A,B,C│
  │ 继续执行      │←─────│          │

  异步操作允许 CPU 快速提交多个 kernel
```

### 24.5.2 cp.async 与内存异步

`cp.async` 是 NVIDIA GPU 支持的异步内存拷贝指令，可以将数据从 Global Memory 直接拷贝到 Shared Memory，无需寄存器中转。

```python
@triton.jit
def async_copy_example(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    使用异步拷贝的 GEMM kernel
    通过 num_stages 控制流水线深度
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 使用 make_block_ptr 进行异步加载
    a_ptr_block = tl.make_block_ptr(
        a_ptr + pid_m * BLOCK_M * K,
        shape=(BLOCK_M, K),
        strides=(K, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_ptr_block = tl.make_block_ptr(
        b_ptr + pid_n * BLOCK_N,
        shape=(K, N),
        strides=(1, N),
        offsets=(0, 0),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # 异步加载 A tile
        a = tl.load(a_ptr_block, boundary_check=(0, 1), padding_option=0)
        # 异步加载 B tile
        b = tl.load(b_ptr_block, boundary_check=(0, 1), padding_option=0)

        # 计算
        acc += tl.dot(a, b)

        # 前进 block pointer
        a_ptr_block = tl.advance(a_ptr_block, [0, BLOCK_K])
        b_ptr_block = tl.advance(b_ptr_block, [BLOCK_K, 0])

    # 存储结果
    c_ptr_block = tl.make_block_ptr(
        c_ptr + pid_m * BLOCK_M * N + pid_n * BLOCK_N,
        shape=(M, N),
        strides=(N, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_ptr_block, acc, boundary_check=(0, 1))
```

### 24.5.3 跨 kernel 异步（Stream）

在实际应用中，经常需要将多个 kernel 提交到不同的 stream 中并行执行。

```python
import torch

# 创建多个 CUDA stream
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()

# 在默认 stream 中启动 kernel A
with torch.cuda.stream(stream1):
    output_a = some_kernel(input_a)

# 在 stream1 中启动 kernel B（依赖 A 的结果）
with torch.cuda.stream(stream1):
    output_b = another_kernel(output_a)

# 在 stream2 中启动独立的 kernel C
with torch.cuda.stream(stream2):
    output_c = independent_kernel(input_c)

# 同步所有 stream
torch.cuda.synchronize()
```

### 24.5.4 事件同步

CUDA 事件提供了精确的同步机制，可以跨 stream 进行同步。

```python
import torch

# 创建 CUDA 事件
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# 记录开始事件
start_event.record()

# 执行 kernel
kernel[grid](*args)

# 记录结束事件
end_event.record()

# 等待事件完成
torch.cuda.synchronize()

# 获取执行时间（毫秒）
elapsed_time = start_event.elapsed_time(end_event)
print(f"Kernel 执行时间: {elapsed_time:.2f} ms")
```

```python
@triton.jit
def async_pipeline_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    异步流水线 kernel
    使用 num_stages 控制加载和计算的重叠程度
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 第一次加载（无重叠）
    offs_k_0 = offs_k
    a_0 = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k_0[None, :] * stride_ak,
                   mask=(offs_m[:, None] < M) & (offs_k_0[None, :] < K), other=0.0)
    b_0 = tl.load(b_ptr + offs_k_0[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                   mask=(offs_k_0[:, None] < K) & (offs_n[None, :] < N), other=0.0)

    for k in range(0, tl.cdiv(K, BLOCK_K) - 1):
        # 计算当前 tile
        acc += tl.dot(a_0, b_0)

        # 同时加载下一个 tile（与计算重叠）
        offs_k_next = (k + 1) * BLOCK_K + offs_k
        a_next = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k_next[None, :] * stride_ak,
                          mask=(offs_m[:, None] < M) & (offs_k_next[None, :] < K), other=0.0)
        b_next = tl.load(b_ptr + offs_k_next[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                          mask=(offs_k_next[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # 更新 tile
        a_0 = a_next
        b_0 = b_next

    # 处理最后一个 tile
    acc += tl.dot(a_0, b_0)

    # 存储结果
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

---

## 24.6 计算强度优化

### 24.6.1 计算强度（Arithmetic Intensity）

计算强度定义为计算量（FLOPs）与访存量（Bytes）的比值：

```
计算强度 = FLOPs / Bytes

示例：矩阵乘法 C = A × B，其中 A: M×K, B: K×N

  FLOPs = 2 × M × N × K（乘法 + 加法）
  Bytes = (M×K + K×N + M×N) × dtype_size

  对于 FP16：
  Bytes = (M×K + K×N + M×N) × 2

  计算强度 = 2×M×N×K / ((M×K + K×N + M×N) × 2)
           = M×N×K / (M×K + K×N + M×N)
```

### 24.6.2 Roofline 模型

Roofline 模型将计算性能与内存带宽联系起来：

```
性能
  ↑
  │      ┌──────────────────────────
  │     /
  │    /  计算受限区
  │   /
  │  /
  │ /
  │/───────────────────────────────→ 计算强度
  │        ↑
  │     Ridge Point
  │
  │  内存受限区

  Ridge Point = Peak Compute / Peak Bandwidth
  A100 FP16: 312 TFLOPS / 2 TB/s = 156 FLOPs/Byte
```

### 24.6.3 Tile Size 优化

选择合适的 tile size 可以最大化计算强度：

```python
@triton.autotune(
    configs=[
        # 小 tile：低计算强度，适合小矩阵
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        # 中等 tile：中等计算强度
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        # 大 tile：高计算强度，适合大矩阵
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        # 非方 tile：特殊形状优化
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_roofline_optimized(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    通过 autotune 自动选择最优 tile size
    不同矩阵形状对应不同的最优配置
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k_cur = k * BLOCK_K + offs_k

        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k_cur[None, :] * stride_ak,
                     mask=(offs_m[:, None] < M) & (offs_k_cur[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_k_cur[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                     mask=(offs_k_cur[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

### 24.6.4 Roofline 调优实践

```
A100 Roofline 调优示例

  目标：最大化 GEMM 性能

  步骤 1: 计算理论峰值
  ├─ FP16 Tensor Core: 312 TFLOPS
  ├─ HBM 带宽: 2 TB/s
  └─ Ridge Point: 156 FLOPs/Byte

  步骤 2: 分析当前 kernel
  ├─ 假设 M=1024, N=1024, K=1024
  ├─ FLOPs = 2 × 1024³ ≈ 2.1 TFLOPs
  ├─ Bytes = 3 × 1024² × 2 ≈ 6 MB
  └─ 计算强度 ≈ 350 FLOPs/Byte > 156 → 计算受限

  步骤 3: 选择 tile size
  ├─ 大 tile (128×128×32): 计算强度高，适合计算受限
  ├─ 中 tile (64×64×32): 适中
  └─ 小 tile (32×32×32): 计算强度低，适合内存受限

  步骤 4: 验证
  ├─ Profile kernel，观察 SM 利用率
  ├─ 调整 num_warps 和 num_stages
  └─ 达到 90%+ 理论峰值
```

---

## 24.7 数值精度权衡

### 24.7.1 精度-速度权衡概览

不同数值精度在性能和精度之间存在权衡：

```
精度-速度权衡图

  性能
  ↑
  │  INT8  ●
  │
  │        FP8 (E4M3)  ●
  │
  │              FP16  ●
  │                    BF16  ●
  │
  │                        TF32  ●
  │
  │                              FP32  ●
  └─────────────────────────────────────────→ 精度
```

### 24.7.2 FP32 vs TF32

```python
import torch

def compare_fp32_tf32():
    """对比 FP32 和 TF32 精度"""
    # 创建测试数据
    M, N, K = 1024, 1024, 1024
    a = torch.randn(M, K, dtype=torch.float32, device='cuda')
    b = torch.randn(K, N, dtype=torch.float32, device='cuda')

    # FP32 参考结果
    torch.backends.cuda.matmul.allow_tf32 = False
    c_fp32 = torch.mm(a, b)

    # TF32 结果
    torch.backends.cuda.matmul.allow_tf32 = True
    c_tf32 = torch.mm(a, b)

    # 计算误差
    error = torch.abs(c_fp32 - c_tf32).mean().item()
    relative_error = error / torch.abs(c_fp32).mean().item()

    print(f"TF32 平均绝对误差: {error:.6f}")
    print(f"TF32 相对误差: {relative_error:.6f}")
    print(f"TF32 最大误差: {torch.abs(c_fp32 - c_tf32).max().item():.6f}")

    # 性能对比
    torch.backends.cuda.matmul.allow_tf32 = False
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(100):
        _ = torch.mm(a, b)
    end.record()
    torch.cuda.synchronize()
    time_fp32 = start.elapsed_time(end) / 100

    torch.backends.cuda.matmul.allow_tf32 = True
    start.record()
    for _ in range(100):
        _ = torch.mm(a, b)
    end.record()
    torch.cuda.synchronize()
    time_tf32 = start.elapsed_time(end) / 100

    print(f"FP32 时间: {time_fp32:.2f} ms")
    print(f"TF32 时间: {time_tf32:.2f} ms")
    print(f"加速比: {time_fp32/time_tf32:.2f}x")

# compare_fp32_tf32()
```

### 24.7.3 FP16 vs BF16

```python
import torch

def compare_fp16_bf16():
    """对比 FP16 和 BF16 精度"""
    M, N, K = 1024, 1024, 1024

    # FP16
    a_fp16 = torch.randn(M, K, dtype=torch.float16, device='cuda')
    b_fp16 = torch.randn(K, N, dtype=torch.float16, device='cuda')
    c_fp16 = torch.mm(a_fp16, b_fp16)

    # BF16
    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    b_bf16 = torch.randn(K, N, dtype=torch.bfloat16, device='cuda')
    c_bf16 = torch.mm(a_bf16, b_bf16)

    # FP32 参考
    a_fp32 = a_bf16.float()
    b_fp32 = b_bf16.float()
    c_fp32 = torch.mm(a_fp32, b_fp32)

    # 误差分析
    error_fp16 = torch.abs(c_fp32 - c_fp16.float()).mean().item()
    error_bf16 = torch.abs(c_fp32 - c_bf16.float()).mean().item()

    print(f"FP16 误差: {error_fp16:.6f}")
    print(f"BF16 误差: {error_bf16:.6f}")

    # 性能对比
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(100):
        _ = torch.mm(a_fp16, b_fp16)
    end.record()
    torch.cuda.synchronize()
    time_fp16 = start.elapsed_time(end) / 100

    start.record()
    for _ in range(100):
        _ = torch.mm(a_bf16, b_bf16)
    end.record()
    torch.cuda.synchronize()
    time_bf16 = start.elapsed_time(end) / 100

    print(f"FP16 时间: {time_fp16:.2f} ms")
    print(f"BF16 时间: {time_bf16:.2f} ms")

# compare_fp16_bf16()
```

### 24.7.4 典型应用场景

| 场景 | 推荐精度 | 原因 |
|------|---------|------|
| 模型训练（前向） | BF16/TF32 | 平衡速度和精度 |
| 模型训练（反向） | BF16 | 梯度对精度要求较低 |
| 模型推理（精度敏感） | FP16 | 广泛支持，精度好 |
| 模型推理（延迟敏感） | INT8/FP8 | 最高性能 |
| 科学计算 | FP32/FP64 | 需要高精度 |
| 强化学习 | FP32 | 数值稳定性关键 |
| 推荐系统 | FP16/INT8 | 大规模嵌入表 |
| 图像处理 | FP16 | 天然适合低精度 |

```python
@triton.jit
def mixed_precision_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    混合精度 GEMM kernel
    使用 FP16 输入，FP32 累加，FP16 输出
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # FP32 累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k_cur = k * BLOCK_K + offs_k

        # 加载 FP16 数据
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k_cur[None, :] * stride_ak,
                     mask=(offs_m[:, None] < M) & (offs_k_cur[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_k_cur[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                     mask=(offs_k_cur[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # FP16 × FP16 → FP32 累加
        acc += tl.dot(a, b)

    # 转换为 FP16 输出
    c = acc.to(tl.float16)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

### 24.7.5 精度验证方法

```python
import torch

def verify_precision():
    """验证 kernel 精度的完整流程"""
    M, N, K = 1024, 1024, 1024

    # 创建测试数据
    a = torch.randn(M, K, dtype=torch.float16, device='cuda')
    b = torch.randn(K, N, dtype=torch.float16, device='cuda')

    # 参考结果（FP32）
    c_ref = torch.mm(a.float(), b.float()).half()

    # 测试 kernel 结果
    c_kernel = my_gemm_kernel(a, b)

    # 计算各种误差指标
    abs_error = torch.abs(c_ref - c_kernel)
    rel_error = abs_error / (torch.abs(c_ref) + 1e-6)

    print(f"最大绝对误差: {abs_error.max().item():.6f}")
    print(f"平均绝对误差: {abs_error.mean().item():.6f}")
    print(f"最大相对误差: {rel_error.max().item():.6f}")
    print(f"平均相对误差: {rel_error.mean().item():.6f}")
    print(f"Cosine Similarity: {torch.nn.functional.cosine_similarity(c_ref.flatten(), c_kernel.flatten(), dim=0).item():.6f}")

    # 验证通过标准
    assert abs_error.max().item() < 0.01, "最大绝对误差过大"
    assert rel_error.mean().item() < 0.001, "平均相对误差过大"
    print("精度验证通过！")
```

---

## 24.8 compute-sanitizer 集成

### 24.8.1 compute-sanitizer 概述

compute-sanitizer 是 NVIDIA 提供的 GPU 计算工具，类似于 CPU 上的 Valgrind。它包含多个工具：

| 工具 | 功能 | 使用场景 |
|------|------|---------|
| `memcheck` | 内存错误检测 | 越界访问、未初始化内存 |
| `racecheck` | 竞态条件检测 | 共享内存写冲突 |
| `synccheck` | 同步错误检测 | warp 同步问题 |
| `tool` | 性能分析 | 内存访问模式分析 |

### 24.8.2 内存错误检测

```bash
# 使用 compute-sanitizer 检测内存错误
compute-sanitizer --tool memcheck python my_kernel.py

# 输出示例
========= Invalid __global__ read of size 4 bytes
=========     at 0x00000148 in my_kernel
=========     by thread (32,0,0) in block (0,0,0)
=========     Address 0x7f1234567890 is out of bounds
=========     Saved host backtrace up to driver entry point
=========     at launch of kernel my_kernel
```

### 24.8.3 竞态条件检测

```python
import triton
import triton.language as tl

@triton.jit
def race_condition_example(
    x_ptr, output_ptr, N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    存在竞态条件的 kernel
    多个 program 同时写入同一个地址
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + offs, mask=offs < N, other=0.0)
    partial_sum = tl.sum(x)

    # 竞态条件：多个 program 同时原子加
    tl.atomic_add(output_ptr, partial_sum)

# 正确的实现
@triton.jit
def no_race_condition(
    x_ptr, output_ptr, N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    无竞态条件的 kernel
    每个 program 写入不同位置
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + offs, mask=offs < N, other=0.0)
    partial_sum = tl.sum(x)

    # 每个 program 写入独立位置
    tl.store(output_ptr + pid, partial_sum)
```

```bash
# 检测竞态条件
compute-sanitizer --tool racecheck python my_kernel.py

# 输出示例
========= Race check report
=========     at 0x00000148 in my_kernel
=========     at address 0x7f1234567890
=========     Thread (32,0,0) in block (0,0,0) : Write address 0x7f1234567890
=========     Thread (64,0,0) in block (0,0,0) : Write address 0x7f1234567890
=========     Data race detected between threads
```

### 24.8.4 Triton 内置调试

```python
import triton
import triton.language as tl

# Triton 提供内置的调试工具
@triton.jit
def debug_kernel(
    x_ptr, output_ptr, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + offs, mask=offs < N, other=0.0)

    # 打印调试信息（仅在调试模式下有效）
    tl.static_print(f"pid={pid}, offs={offs}")

    output = x * 2.0
    tl.store(output_ptr + offs, output, mask=offs < N)

# 启用调试模式
# TRITON_PRINT_AUTOTUNING=1 python my_kernel.py
```

### 24.8.5 性能分析集成

```python
import torch
import triton
import triton.language as tl
from triton.testing import do_bench

@triton.jit
def benchmark_kernel(
    x_ptr, output_ptr, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs, mask=offs < N, other=0.0)
    output = x * 2.0 + 1.0
    tl.store(output_ptr + offs, output, mask=offs < N)

def benchmark_performance():
    """性能基准测试"""
    N = 1024 * 1024
    BLOCK_SIZE = 1024
    x = torch.randn(N, dtype=torch.float32, device='cuda')
    output = torch.empty(N, dtype=torch.float32, device='cuda')

    # 使用 Triton 内置 benchmark
    ms = do_bench(lambda: benchmark_kernel[(N // BLOCK_SIZE,)](
        x, output, N, BLOCK_SIZE=BLOCK_SIZE
    ))

    # 计算带宽
    bytes_transferred = N * 4 * 2  # 读 + 写
    bandwidth_gb_s = bytes_transferred / (ms / 1000) / 1e9

    print(f"执行时间: {ms:.3f} ms")
    print(f"带宽: {bandwidth_gb_s:.1f} GB/s")

# benchmark_performance()
```

### 24.8.6 compute-sanitizer 与 Triton 集成

```python
import subprocess
import os

def run_with_sanitizer(script_path, tool='memcheck'):
    """
    使用 compute-sanitizer 运行 Triton kernel
    """
    cmd = [
        'compute-sanitizer',
        '--tool', tool,
        'python', script_path
    ]

    print(f"运行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    print("STDOUT:")
    print(result.stdout)

    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    return result.returncode == 0

# 使用示例
# run_with_sanitizer('my_kernel.py', 'memcheck')
# run_with_sanitizer('my_kernel.py', 'racecheck')
```

---

## 24.9 优化案例：GEMM Kernel 完整流程

### 24.9.1 Roofline 分析

```python
import torch

def roofline_analysis():
    """
    对 GEMM 进行 Roofline 分析
    """
    # A100 参数
    peak_tflops = 312  # FP16 Tensor Core
    hbm_bandwidth_gbs = 2000  # 2 TB/s

    # Ridge Point
    ridge_point = peak_tflops * 1e12 / (hbm_bandwidth_gbs * 1e9)
    print(f"Ridge Point: {ridge_point:.1f} FLOPs/Byte")

    # 分析不同矩阵形状
    shapes = [
        (64, 64, 64),       # 小矩阵
        (256, 256, 256),    # 中等矩阵
        (1024, 1024, 1024), # 大矩阵
        (4096, 4096, 4096), # 超大矩阵
    ]

    for M, N, K in shapes:
        flops = 2 * M * N * K
        bytes_accessed = (M * K + K * N + M * N) * 2  # FP16
        ai = flops / bytes_accessed

        # 理论性能
        if ai >= ridge_point:
            perf_tflops = peak_tflops
            bound = "计算受限"
        else:
            perf_tflops = ai * hbm_bandwidth_gbs / 1000
            bound = "内存受限"

        print(f"\n矩阵 {M}×{N}×{K}:")
        print(f"  FLOPs: {flops/1e9:.2f} GFLOPs")
        print(f"  Bytes: {bytes_accessed/1e6:.2f} MB")
        print(f"  计算强度: {ai:.1f} FLOPs/Byte")
        print(f"  理论性能: {perf_tflops:.1f} TFLOPS ({bound})")

# roofline_analysis()
```

### 249.2 朴素实现

```python
import triton
import triton.language as tl

@triton.jit
def gemm_naive(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    朴素 GEMM 实现
    性能约为 cuBLAS 的 10-20%
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k_cur = k * BLOCK_K + offs_k

        # 加载 A
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k_cur[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k_cur[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # 加载 B
        b_ptrs = b_ptr + offs_k_cur[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (offs_k_cur[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # 计算
        acc += tl.dot(a, b)

    # 存储
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)
```

### 24.9.3 优化版本 1：Tensor Core 利用

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_tensor_core(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    优化版本 1：确保 Tensor Core 对齐
    性能提升 2-3 倍
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 使用 block pointer 进行对齐加载
    a_block = tl.make_block_ptr(
        a_ptr + pid_m * BLOCK_M * K,
        shape=(BLOCK_M, K),
        strides=(K, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block = tl.make_block_ptr(
        b_ptr + pid_n * BLOCK_N,
        shape=(K, N),
        strides=(1, N),
        offsets=(0, 0),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_block, boundary_check=(0, 1), padding_option=0)
        b = tl.load(b_block, boundary_check=(0, 1), padding_option=0)

        acc += tl.dot(a, b)

        a_block = tl.advance(a_block, [0, BLOCK_K])
        b_block = tl.advance(b_block, [BLOCK_K, 0])

    c_block = tl.make_block_ptr(
        c_ptr + pid_m * BLOCK_M * N + pid_n * BLOCK_N,
        shape=(M, N),
        strides=(N, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block, acc, boundary_check=(0, 1))
```

### 24.9.4 优化版本 2：软件流水线

```python
@triton.autotune(
    configs=[
        # num_stages=2: 双缓冲
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        # num_stages=3: 三缓冲
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        # num_stages=4: 四缓冲
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_pipelined(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    优化版本 2：添加软件流水线
    性能提升 20-40%
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 第一次加载
    offs_k_0 = offs_k
    a_0 = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k_0[None, :] * stride_ak,
                   mask=(offs_m[:, None] < M) & (offs_k_0[None, :] < K), other=0.0)
    b_0 = tl.load(b_ptr + offs_k_0[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                   mask=(offs_k_0[:, None] < K) & (offs_n[None, :] < N), other=0.0)

    k_iterations = tl.cdiv(K, BLOCK_K)
    for k in range(k_iterations - 1):
        # 计算当前 tile
        acc += tl.dot(a_0, b_0)

        # 加载下一个 tile（与计算重叠）
        offs_k_next = (k + 1) * BLOCK_K + offs_k
        a_next = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k_next[None, :] * stride_ak,
                          mask=(offs_m[:, None] < M) & (offs_k_next[None, :] < K), other=0.0)
        b_next = tl.load(b_ptr + offs_k_next[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                          mask=(offs_k_next[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        a_0 = a_next
        b_0 = b_next

    # 处理最后一个 tile
    acc += tl.dot(a_0, b_0)

    # 存储结果
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

### 24.9.5 优化版本 3：ILP 与多累加器

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'NUM_ACC': 2}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'NUM_ACC': 4}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_ilp_optimized(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_ACC: tl.constexpr,
):
    """
    优化版本 3：ILP 与多累加器
    性能提升 10-30%
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # 多个独立累加器
    accs = [tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32) for _ in range(NUM_ACC)]

    k_iterations = tl.cdiv(K, BLOCK_K)
    for k in range(0, k_iterations, NUM_ACC):
        for i in range(NUM_ACC):
            if k + i < k_iterations:
                offs_k_cur = (k + i) * BLOCK_K + offs_k

                a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k_cur[None, :] * stride_ak
                b_ptrs = b_ptr + offs_k_cur[:, None] * stride_bk + offs_n[None, :] * stride_bn

                a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k_cur[None, :] < K), other=0.0)
                b = tl.load(b_ptrs, mask=(offs_k_cur[:, None] < K) & (offs_n[None, :] < N), other=0.0)

                accs[i] += tl.dot(a, b)

    # 合并累加器
    acc = accs[0]
    for i in range(1, NUM_ACC):
        acc += accs[i]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

### 24.9.6 完整性能对比

```python
import torch
import time

def benchmark_gemm():
    """
    完整的 GEMM 性能对比
    """
    M, N, K = 4096, 4096, 4096

    # 创建输入
    a = torch.randn(M, K, dtype=torch.float16, device='cuda')
    b = torch.randn(K, N, dtype=torch.float16, device='cuda')
    c = torch.empty(M, N, dtype=torch.float16, device='cuda')

    # cuBLAS 参考
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(10):
        torch.mm(a, b)
    end.record()
    torch.cuda.synchronize()
    cublas_time = start.elapsed_time(end) / 10

    # Triton kernel
    grid = (triton.cdiv(M, 128), triton.cdiv(N, 128))
    start.record()
    for _ in range(10):
        gemm_ilp_optimized[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, NUM_ACC=2,
        )
    end.record()
    torch.cuda.synchronize()
    triton_time = start.elapsed_time(end) / 10

    # 计算 FLOPS
    flops = 2 * M * N * K
    cublas_tflops = flops / (cublas_time / 1000) / 1e12
    triton_tflops = flops / (triton_time / 1000) / 1e12

    print(f"矩阵大小: {M}×{N}×{K}")
    print(f"cuBLAS: {cublas_time:.2f} ms ({cublas_tflops:.2f} TFLOPS)")
    print(f"Triton: {triton_time:.2f} ms ({triton_tflops:.2f} TFLOPS)")
    print(f"Triton/cuBLAS: {triton_tflops/cublas_tflops*100:.1f}%")

# benchmark_gemm()
```

---

## 24.10 高级优化技巧

### 24.10.1 Split-K 优化

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 2}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K', 'SPLIT_K'],
)
@triton.jit
def gemm_split_k(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """
    Split-K GEMM
    将 K 维度分片，增加并行度
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)  # 额外的 K 维度 grid

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 计算当前 K 分片的范围
    k_start = pid_k * tl.cdiv(K, SPLIT_K)
    k_end = min(k_start + tl.cdiv(K, SPLIT_K), K)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(k_start, k_end, BLOCK_K):
        offs_k_cur = k + offs_k
        valid_k = offs_k_cur < k_end

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k_cur[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k_cur[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & valid_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=valid_k[:, None] & (offs_n[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

    # 原子累加到全局输出
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.atomic_add(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

### 24.10.2 向量化加载优化

```python
@triton.jit
def vectorized_load_kernel(
    x_ptr, output_ptr, N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    向量化加载优化
    使用更大的数据类型加载更多数据
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 使用向量化加载（128 位）
    # 每次加载 4 个 FP32
    vector_size = 4
    offs_vec = offs // vector_size

    # 加载向量化数据
    x_vec = tl.load(x_ptr + offs_vec, mask=offs_vec < N // vector_size)

    # 提取单个元素
    x = tl.view(x_vec, [BLOCK_SIZE])

    output = x * 2.0 + 1.0
    tl.store(output_ptr + offs, output, mask=offs < N)
```

### 24.10.3 Warp 级优化

```python
@triton.jit
def warp_level_optimization(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Warp 级 GEMM 优化
    每个 warp 处理一个小 tile
    """
    # 每个 warp 处理的 tile 大小
    WARP_M = BLOCK_M // 4  # 假设 4 个 warp
    WARP_N = BLOCK_N // 4

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Warp ID
    warp_id = tl.arange(0, 4)
    warp_m = (warp_id // 2) * WARP_M
    warp_n = (warp_id % 2) * WARP_N

    offs_m = pid_m * BLOCK_M + warp_m + tl.arange(0, WARP_M)
    offs_n = pid_n * BLOCK_N + warp_n + tl.arange(0, WARP_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((WARP_M, WARP_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k_cur = k * BLOCK_K + offs_k

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k_cur[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k_cur[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k_cur[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k_cur[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

---

## 24.11 调试与性能分析

### 24.11.1 性能瓶颈诊断

```python
import torch
import triton
import triton.language as tl
from triton.testing import do_bench

def diagnose_kernel():
    """
    诊断 kernel 性能瓶颈
    """
    N = 1024 * 1024
    BLOCK_SIZE = 1024
    x = torch.randn(N, dtype=torch.float32, device='cuda')
    output = torch.empty(N, dtype=torch.float32, device='cuda')

    @triton.jit
    def kernel(x_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offs, mask=offs < N, other=0.0)
        output = x * 2.0 + 1.0
        tl.store(output_ptr + offs, output, mask=offs < N)

    # 测量性能
    ms = do_bench(lambda: kernel[(N // BLOCK_SIZE,)](x, output, N, BLOCK_SIZE=BLOCK_SIZE))

    # 计算带宽
    bytes_accessed = N * 4 * 2  # 读 + 写
    bandwidth_gb_s = bytes_accessed / (ms / 1000) / 1e9

    # A100 理论峰值带宽
    peak_bandwidth_gb_s = 2000

    utilization = bandwidth_gb_s / peak_bandwidth_gb_s * 100

    print(f"执行时间: {ms:.3f} ms")
    print(f"实际带宽: {bandwidth_gb_s:.1f} GB/s")
    print(f"峰值带宽: {peak_bandwidth_gb_s} GB/s")
    print(f"带宽利用率: {utilization:.1f}%")

    if utilization < 50:
        print("瓶颈: 内存带宽未充分利用")
        print("建议: 检查内存访问模式，使用向量化加载")
    elif utilization < 80:
        print("瓶颈: 部分优化空间")
        print("建议: 尝试不同的 tile size 或展开因子")
    else:
        print("性能良好: 接近理论峰值")
```

### 24.11.2 调试工具集

```python
import triton

# Triton 内置调试选项
import os

# 启用 IR dump
os.environ['TRITON_PRINT_AUTOTUNING'] = '1'

# 启用详细编译日志
os.environ['TRITON_FRONTEND_LOG'] = '1'

# 启用性能分析
os.environ['TRITON_PERF'] = '1'

@triton.jit
def debug_friendly_kernel(
    x_ptr, output_ptr, N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    调试友好的 kernel
    添加静态打印信息
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 静态打印（编译时）
    tl.static_print(f"BLOCK_SIZE={BLOCK_SIZE}")

    x = tl.load(x_ptr + offs, mask=offs < N, other=0.0)
    output = x * 2.0 + 1.0
    tl.store(output_ptr + offs, output, mask=offs < N)
```

### 24.11.3 性能回归检测

```python
import torch
import time
from typing import List, Tuple

def performance_regression_test(
    kernel_func,
    grid_func,
    args_func,
    input_sizes: List[Tuple[int, ...]],
    baseline_ms: List[float],
    tolerance: float = 0.1,
):
    """
    性能回归检测
    """
    results = []

    for size, baseline in zip(input_sizes, baseline_ms):
        args = args_func(*size)
        grid = grid_func(*size)

        # Warmup
        for _ in range(10):
            kernel_func[grid](*args)

        torch.cuda.synchronize()

        # 测量
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(100):
            kernel_func[grid](*args)
        end.record()

        torch.cuda.synchronize()
        current_ms = start.elapsed_time(end) / 100

        # 检测回归
        regression = current_ms > baseline * (1 + tolerance)
        results.append({
            'size': size,
            'baseline_ms': baseline,
            'current_ms': current_ms,
            'regression': regression,
            'change_pct': (current_ms - baseline) / baseline * 100,
        })

        status = "REGRESSION" if regression else "OK"
        print(f"Size {size}: {current_ms:.2f}ms (baseline: {baseline:.2f}ms) [{status}]")

    # 总结
    regressions = sum(1 for r in results if r['regression'])
    print(f"\n总结: {regressions}/{len(results)} 个用例出现回归")

    return results
```

---

## 24.12 优化策略总结

### 24.12.1 优化检查清单

```
Triton Kernel 优化检查清单

  1. Tensor Core 利用
     □ 使用 tl.dot 进行矩阵乘法
     □ 确保维度对齐（16/8 的倍数）
     □ 选择合适的精度（FP16/BF16/FP8）

  2. 内存访问优化
     □ 使用 block pointer 进行对齐加载
     □ 启用 boundary_check 处理边界
     □ 使用向量化加载（128 位）

  3. 循环优化
     □ 使用 autotune 自动选择 tile size
     □ 考虑循环展开（手动或编译器）
     □ 添加软件流水线（num_stages）

  4. ILP 优化
     □ 使用多累加器增加并行度
     □ 确保独立操作可以并行
     □ 避免不必要的依赖

  5. 异步执行
     □ 使用 cp.async 进行异步拷贝
     □ 考虑跨 kernel 异步（stream）
     □ 使用事件同步

  6. 精度权衡
     □ 根据场景选择精度
     □ 验证数值稳定性
     □ 考虑混合精度策略
```

### 24.12.2 性能优化路径

```
性能优化路径

  阶段 1: 正确性
  ┌─────────────────────────────────────┐
  │ 实现正确的 kernel                   │
  │ 验证结果与参考实现一致              │
  └─────────────────────────────────────┘
            ↓
  阶段 2: 基础优化
  ┌─────────────────────────────────────┐
  │ 启用 Tensor Core (tl.dot)           │
  │ 选择合适的 tile size                │
  │ 使用 autotune                       │
  └─────────────────────────────────────┘
            ↓
  阶段 3: 内存优化
  ┌─────────────────────────────────────┐
  │ 优化内存访问模式                    │
  │ 使用 block pointer                  │
  │ 启用软件流水线                      │
  └─────────────────────────────────────┘
            ↓
  阶段 4: 计算优化
  ┌─────────────────────────────────────┐
  │ 循环展开                            │
  │ ILP 优化                            │
  │ 多累加器                            │
  └─────────────────────────────────────┘
            ↓
  阶段 5: 高级优化
  ┌─────────────────────────────────────┐
  │ Split-K                             │
  │ Warp 级优化                         │
  │ 混合精度                            │
  └─────────────────────────────────────┘
            ↓
  阶段 6: 验证与调优
  ┌─────────────────────────────────────┐
  │ 性能回归检测                        │
  │ compute-sanitizer 验证              │
  │ Roofline 分析                       │
  └─────────────────────────────────────┘
```

---

## 本章小结

本章深入探讨了 Triton kernel 的计算优化技术：

1. **Tensor Core 利用**：通过 `tl.dot` 触发 Tensor Core，理解 MMA 指令映射和维度对齐要求
2. **循环展开**：编译器自动展开与手动展开的权衡，展开因子的选择
3. **指令级并行**：ILP 概念、多累加器策略、操作重叠与依赖分析
4. **异步执行**：`cp.async` 指令、跨 kernel 异步（Stream）、事件同步机制
5. **计算强度优化**：Roofline 模型、tile size 选择、Ridge Point 分析
6. **数值精度权衡**：FP32/TF32/FP16/BF16/FP8 的精度-速度权衡与应用场景
7. **compute-sanitizer**：内存错误检测、竞态条件检测、性能分析工具集成
8. **优化案例**：从 Roofline 分析到 Tensor Core 利用到完整性能提升的全流程

**核心收获**：

- 计算优化的本质是**最大化硬件利用率**和**隐藏延迟**
- Tensor Core 是现代 GPU 的核心计算单元，充分利用它可获得 8-16 倍加速
- ILP 和异步执行是隐藏延迟的关键技术
- 精度选择需要在性能和数值稳定性之间权衡
- 系统性的优化流程比零散的优化更有效

---

## 思考题

**题目 1**：解释 Tensor Core 的 MMA 指令 `mma.sync.aligned.m16n8k16` 中各参数的含义。为什么 M 必须是 16 的倍数？

**题目 2**：在 GEMM kernel 中，使用 `num_stages=4` 的软件流水线。如果每个 stage 需要额外 16KB 寄存器，A100 每 SM 有 256KB 寄存器，最多能同时运行多少个 warp？如果使用 `num_stages=2` 呢？

**题目 3**：比较 FP16 和 BF16 在训练和推理中的优缺点。为什么 BF16 在训练中更受欢迎？

**题目 4**：解释 Split-K 优化的原理。在什么矩阵形状下 Split-K 有效？如何选择最优的 `SPLIT_K` 值？

**题目 5**：设计一个实验来测量 ILP 对 kernel 性能的影响。你需要控制哪些变量？如何隔离其他因素？

**题目 6**：解释 `cp.async` 指令如何实现 Global → Shared Memory 的异步拷贝。与普通 `tl.load` 相比，它有什么优势？

**题目 7**：在多 GPU 系统中，如何实现跨 GPU 的异步执行？需要考虑哪些同步问题？

**题目 8**：解释 Roofline 模型中的 Ridge Point 概念。如果一个 kernel 的计算强度低于 Ridge Point，应该优先优化什么？

**题目 9**：比较手动循环展开和编译器自动展开的优缺点。在什么情况下手动展开更有优势？

**题目 10**：设计一个混合精度 GEMM kernel，使用 FP16 输入、FP32 累加、FP16 输出。如何验证其数值稳定性？

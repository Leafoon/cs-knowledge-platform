---
title: "Chapter 8: Dequantize GEMM 与低精度算子"
description: "深入讲解量化基础、INT8/FP8/INT4/NF4 GEMM 实现、权重量化策略、大模型推理应用及 TileLang 低精度算子编程"
updated: "2025-01-01"
---

> **Learning Objectives**
>
> - 掌握对称量化与非对称量化的数学原理及噪声分析
> - 理解 per-tensor / per-channel / per-group 三种量化粒度的区别与适用场景
> - 使用 TileLang 实现 INT8 量化 GEMM（dequantize-then-multiply 策略）
> - 理解 FP8（E4M3/E5M2）数据格式及 Hopper 架构 FP8 Tensor Core 的使用方法
> - 掌握 INT4/NF4 GEMM 实现，包括 GPTQ 和 AWQ 算法原理
> - 了解权重量化与混合精度矩阵乘法的内核融合策略
> - 掌握大模型推理中的量化应用（KV Cache 量化等）
> - 能够分析量化 GEMM 的性能瓶颈并排查常见数值问题

---

## 1. 量化基础 (Quantization Fundamentals)

### 1.1 什么是量化

量化是将高精度浮点数映射到低精度整数或低精度浮点数的过程。在深度学习推理中，模型权重和激活值通常以 FP32 或 FP16 存储，量化可以将它们压缩为 INT8、INT4 甚至更低精度，从而：

- **减少内存占用**：INT8 仅需 FP16 的一半存储空间
- **加速计算**：整数运算吞吐量通常高于浮点运算
- **降低带宽需求**：内存带宽是推理瓶颈之一

```
量化前 (FP16):    [0.123, -0.456, 0.789, -0.012, ...]
                   ↓ 量化
量化后 (INT8):    [15, -57, 99, -2, ...]
                   ↓ 反量化
重建值 (FP16):    [0.122, -0.455, 0.790, -0.016, ...]
```

这个代码块或示意图用于说明 1.1 什么是量化 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 1.2 对称量化 (Symmetric Quantization)

对称量化假设数据分布关于零点对称，量化公式为：

$$x_q = \text{round}\left(\frac{x}{s}\right)$$

$$s = \frac{\max(|x|)}{2^{b-1} - 1}$$

其中 $s$ 是缩放因子（scale），$b$ 是目标比特数。

```python
import torch

def symmetric_quantize(x: torch.Tensor, bits: int = 8) -> tuple[torch.Tensor, float]:
    """
    对称量化：零点固定为 0，数据关于零对称映射
    
    Args:
        x: 输入张量 (FP16/FP32)
        bits: 量化比特数
    
    Returns:
        x_q: 量化后的整数张量
        scale: 缩放因子
    """
    q_max = 2 ** (bits - 1) - 1  # INT8: 127
    q_min = -q_max               # INT8: -127
    
    # 计算缩放因子
    abs_max = x.abs().max().float()
    scale = abs_max / q_max
    
    # 量化
    x_q = torch.clamp(torch.round(x / scale), q_min, q_max).to(torch.int8)
    
    return x_q, scale


def symmetric_dequantize(x_q: torch.Tensor, scale: float) -> torch.Tensor:
    """对称反量化"""
    return x_q.float() * scale
```

**对称量化的特点**：
- 零点固定为 0，实现简单
- 适用于权重等近似对称分布的数据
- 对于非对称分布（如 ReLU 后的激活值），精度损失较大

### 1.3 非对称量化 (Asymmetric Quantization)

非对称量化引入零点偏移，能更好地适应非对称数据分布：

$$x_q = \text{round}\left(\frac{x}{s}\right) + z$$

$$s = \frac{x_{\max} - x_{\min}}{2^b - 1}$$

$$z = \text{round}\left(-\frac{x_{\min}}{s}\right)$$

```python
def asymmetric_quantize(x: torch.Tensor, bits: int = 8) -> tuple[torch.Tensor, float, int]:
    """
    非对称量化：支持任意数据分布
    
    Args:
        x: 输入张量
        bits: 量化比特数
    
    Returns:
        x_q: 量化后的整数张量 (uint8 范围)
        scale: 缩放因子
        zero_point: 零点偏移
    """
    q_max = 2 ** bits - 1  # UINT8: 255
    q_min = 0
    
    x_min, x_max = x.min().float(), x.max().float()
    
    # 计算缩放因子和零点
    scale = (x_max - x_min) / (q_max - q_min)
    if scale == 0:
        scale = 1.0
    zero_point = round(-x_min / scale)
    zero_point = max(q_min, min(q_max, zero_point))
    
    # 量化
    x_q = torch.clamp(torch.round(x / scale) + zero_point, q_min, q_max).to(torch.uint8)
    
    return x_q, scale, zero_point


def asymmetric_dequantize(x_q: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    """非对称反量化"""
    return (x_q.float() - zero_point) * scale
```

上述代码实现了非对称量化和反量化操作。非对称量化通过引入零点偏移来适应非对称数据分布，核心是计算缩放因子和零点偏移两个参数。与对称量化相比，非对称量化能更好地处理数据分布偏移的情况（如ReLU激活后的值），但需要额外的零点存储开销。

### 1.4 量化粒度 (Quantization Granularity)

| 粒度 | 缩放因子数量 | 精度 | 开销 | 适用场景 |
|------|-------------|------|------|---------|
| **Per-tensor** | 1 | 低 | 极低 | 对精度要求不高的场景 |
| **Per-channel** | C (通道数) | 中 | 低 | 权重量化常用 |
| **Per-group** | C × (G/group_size) | 高 | 中 | INT4/NF4 量化常用 |

```python
def per_tensor_quantize(x: torch.Tensor, bits: int = 8) -> tuple[torch.Tensor, float]:
    """Per-tensor 量化：整个张量共享一个 scale"""
    return symmetric_quantize(x, bits)


def per_channel_quantize(x: torch.Tensor, bits: int = 8, channel_dim: int = 0):
    """
    Per-channel 量化：每个输出通道独立量化
    适用于权重矩阵 [out_channels, in_channels]
    """
    q_max = 2 ** (bits - 1) - 1
    scales = []
    x_q_list = []
    
    for c in range(x.shape[channel_dim]):
        if channel_dim == 0:
            channel_data = x[c]
        else:
            channel_data = x[:, c]
        
        abs_max = channel_data.abs().max().float()
        scale = abs_max / q_max if abs_max > 0 else 1.0
        scales.append(scale)
        
        x_q_c = torch.clamp(torch.round(channel_data / scale), -q_max, q_max).to(torch.int8)
        x_q_list.append(x_q_c)
    
    if channel_dim == 0:
        x_q = torch.stack(x_q_list, dim=0)
    else:
        x_q = torch.stack(x_q_list, dim=1)
    
    scales = torch.tensor(scales, dtype=torch.float32)
    return x_q, scales


def per_group_quantize(x: torch.Tensor, bits: int = 8, group_size: int = 128):
    """
    Per-group 量化：将通道分组，每组独立量化
    常用于 INT4/NF4 量化
    """
    original_shape = x.shape
    # 将最后一维按 group_size 分组
    x_flat = x.reshape(-1, group_size)
    num_groups = x_flat.shape[0]
    
    q_max = 2 ** (bits - 1) - 1
    scales = []
    x_q_list = []
    
    for g in range(num_groups):
        group_data = x_flat[g]
        abs_max = group_data.abs().max().float()
        scale = abs_max / q_max if abs_max > 0 else 1.0
        scales.append(scale)
        
        x_q_g = torch.clamp(torch.round(group_data / scale), -q_max, q_max).to(torch.int8)
        x_q_list.append(x_q_g)
    
    x_q = torch.stack(x_q_list, dim=0)
    scales = torch.tensor(scales, dtype=torch.float32)
    
    return x_q.reshape(original_shape), scales
```

上述代码实现了三种量化粒度：per-tensor、per-channel和per-group。Per-channel量化为每个输出通道独立计算缩放因子，适用于权重矩阵；Per-group量化将通道进一步分组，每组独立量化，提供更高精度但增加存储开销。选择合适的量化粒度是平衡精度和效率的关键，通常INT4/NF4量化使用per-group策略。

### 1.5 量化噪声分析

量化过程引入的误差可以通过信噪比（SQNR）来衡量：

$$\text{SQNR} = 10 \log_{10}\left(\frac{\text{Signal Power}}{\text{Noise Power}\right) \approx 6.02b + 1.76 \text{ dB}$$

```python
def quantization_noise_analysis(x: torch.Tensor, bits_list: list[int]):
    """
    分析不同比特数下的量化噪声
    """
    results = {}
    for bits in bits_list:
        x_q, scale = symmetric_quantize(x, bits)
        x_recon = symmetric_dequantize(x_q, scale)
        
        noise = x - x_recon
        signal_power = (x ** 2).mean()
        noise_power = (noise ** 2).mean()
        
        sqnr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
        mse = noise_power.item()
        max_error = noise.abs().max().item()
        
        results[bits] = {
            'sqnr_db': sqnr.item(),
            'mse': mse,
            'max_error': max_error
        }
        print(f"  {bits}-bit: SQNR={sqnr:.1f}dB, MSE={mse:.6f}, MaxErr={max_error:.4f}")
    
    return results
```

上述代码实现了量化噪声分析功能，通过计算不同比特数下的信噪比（SQNR）、均方误差（MSE）和最大误差来评估量化质量。SQNR是衡量量化精度的重要指标，理论上每增加1比特精度约提升6.02dB信噪比。实际应用中，该函数可用于比较不同量化方案的效果，为选择合适的量化比特数提供依据。

> [!TIP]
> 实践中，INT8 量化通常能保持模型精度损失在 1% 以内，而 INT4 量化可能需要配合校准数据或更复杂的算法（如 GPTQ/AWQ）来控制精度损失。

### 1.6 量化误差的分布特性

```
原始分布 (FP16):           量化误差分布 (INT8):
    │  ██                    │     ██
    │ ████                   │    ████
    │██████                  │   ██████
    │████████                │  ████████
    └──────────              └──────────
   -1    0    1             -0.005  0  0.005
   
   近似均匀分布              近似均匀分布，幅度为 ±step/2
```

量化误差可以建模为均匀分布的随机噪声：

$$e \sim U\left(-\frac{s}{2}, \frac{s}{2}\right), \quad \sigma_e^2 = \frac{s^2}{12}$$

---

## 2. INT8 量化 GEMM

### 2.1 Dequantize-then-Multiply 策略

INT8 量化 GEMM 的核心思想是：先将 INT8 权重和/或激活反量化为浮点数，再执行浮点矩阵乘法。

```
策略一：Dequantize-then-Multiply（先反量化再乘法）

  INT8 A  ──→ [Dequant] ──→ FP16 A' ──┐
                                        ├──→ FP16 GEMM ──→ FP16 C
  INT8 B  ──→ [Dequant] ──→ FP16 B' ──┘

策略二：Multiply-then-Dequantize（先整数乘法再反量化）

  INT8 A  ──┐
            ├──→ INT8 GEMM ──→ INT32 C ──→ [Dequant] ──→ FP16 C
  INT8 B  ──┘
```

这个代码块或示意图用于说明 2.1 Dequantize-then-Multiply 策略 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| Dequantize-then-Multiply | 实现简单，复用 FP16 Tensor Core | 反量化开销 | 通用场景 |
| Multiply-then-Dequantize | 利用 INT8 Tensor Core，计算量低 | 需要 INT8 Tensor Core 支持 | NVIDIA Turing+ |

### 2.2 TileLang INT8 GEMM 基础实现

```python
import tilelang
from tilelang import T

@T.prim_func
def int8_gemm_dequantize(
    A: T.Buffer[(M, K), "int8"],
    B: T.Buffer[(N, K), "int8"],
    C: T.Buffer[(M, N), "float16"],
    scale_A: T.Buffer[(M,), "float32"],
    scale_B: T.Buffer[(N,), "float32"],
):
    """
    INT8 GEMM with dequantize-then-multiply strategy
    
    C = dequant(A) @ dequant(B)^T
    dequant(x, s) = x * s
    
    使用 per-channel 量化：
    - A 每行一个 scale
    - B 每列一个 scale
    """
    # 定义分块参数
    BM = 128  # 行分块大小
    BN = 128  # 列分块大小
    BK = 64   # 归约分块大小（INT8 使用更小的 K 分块）
    
    # Tiling 声明
    T.grid_config(
        T.Config(BM, BN, BK),
        T.Config(1, 1, 1),  # warp 配置
    )
    
    with T.Block("root"):
        # 分配共享内存
        A_smem = T.alloc_shared([BM, BK], "int8")
        B_smem = T.alloc_shared([BN, BK], "int8")
        
        # 分配 Fragment（寄存器）
        A_frag = T.alloc_fragment([BM, BK], "float16")
        B_frag = T.alloc_fragment([BN, BK], "float16")
        C_frag = T.alloc_fragment([BM, BN], "float32")
        
        # 初始化累加器
        T.clear(C_frag)
        
        # 归约循环
        for k in T.serial(T.ceildiv(K, BK)):
            # 加载 INT8 数据到共享内存
            T.copy(A[by * BM : (by + 1) * BM, k * BK : (k + 1) * BK], A_smem)
            T.copy(B[bx * BN : (bx + 1) * BN, k * BK : (k + 1) * BK], B_smem)
            
            T.sync()
            
            # Dequantize：INT8 -> FP16，应用 per-channel scale
            for i, j in T.Parallel(BM, BK):
                A_frag[i, j] = T.cast(A_smem[i, j], "float16") * T.cast(scale_A[by * BM + i], "float16")
            
            for i, j in T.Parallel(BN, BK):
                B_frag[i, j] = T.cast(B_smem[i, j], "float16") * T.cast(scale_B[bx * BN + i], "float16")
            
            T.sync()
            
            # FP16 GEMM 累加
            T.gemm(A_frag, B_frag, C_frag)
        
        # 写回结果
        for i, j in T.Parallel(BM, BN):
            C[by * BM + i, bx * BN + j] = T.cast(C_frag[i, j], "float16")
```

上述代码实现了基于TileLang的INT8 GEMM内核，采用dequantize-then-multiply策略。核心思想是先将INT8权重和激活反量化为FP16，再使用FP16 Tensor Core进行矩阵乘法。该实现使用per-channel量化粒度，每行/每列独立缩放，通过分块和共享内存优化内存访问模式，是INT8量化GEMM的基础实现。

### 2.3 使用 INT8 Tensor Core 的实现

```python
@T.prim_func
def int8_gemm_tc(
    A: T.Buffer[(M, K), "int8"],
    B: T.Buffer[(N, K), "int8"],
    C: T.Buffer[(M, N), "int32"],
):
    """
    使用 INT8 Tensor Core 的矩阵乘法
    输出为 INT32，需要后续反量化
    
    NVIDIA Tensor Core INT8 指令：
    - m16n8k32.s8（Ampere）
    - m16n8k64.s8（Hopper）
    """
    BM = 128
    BN = 128
    BK = 64  # INT8 Tensor Core 通常要求 K 维度为 32 或 64 的倍数
    
    with T.Block("root"):
        A_smem = T.alloc_shared([BM, BK], "int8")
        B_smem = T.alloc_shared([BN, BK], "int8")
        
        # INT8 Fragment 用于 Tensor Core
        A_frag = T.alloc_fragment([BM, BK], "int8")
        B_frag = T.alloc_fragment([BN, BK], "int8")
        C_frag = T.alloc_fragment([BM, BN], "int32")
        
        T.clear(C_frag)
        
        for k in T.serial(T.ceildiv(K, BK)):
            T.copy(A[by * BM : (by + 1) * BM, k * BK : (k + 1) * BK], A_smem)
            T.copy(B[bx * BN : (bx + 1) * BN, k * BK : (k + 1) * BK], B_smem)
            T.sync()
            
            T.copy(A_smem, A_frag)
            T.copy(B_smem, B_frag)
            
            # 使用 INT8 Tensor Core 进行矩阵乘法累加
            # 输出为 INT32
            T.gemm(A_frag, B_frag, C_frag, "int8", "int8", "int32")
        
        # 写回 INT32 结果
        for i, j in T.Parallel(BM, BN):
            C[by * BM + i, bx * BN + j] = C_frag[i, j]
```

上述代码实现了使用INT8 Tensor Core的矩阵乘法内核。与之前反量化到FP16不同，该实现直接使用INT8数据进行矩阵乘法，输出为INT32累加结果。这种方式能充分利用INT8 Tensor Core的高吞吐量，但需要后续反量化步骤。注意K维度需要为32或64的倍数以满足Tensor Core指令要求。

### 2.4 完整的 INT8 GEMM with Fusion

```python
@T.prim_func
def int8_gemm_fused(
    A: T.Buffer[(M, K), "int8"],
    B: T.Buffer[(N, K), "int8"],
    C: T.Buffer[(M, N), "float16"],
    scale_A: T.Buffer[(M,), "float32"],
    scale_B: T.Buffer[(N,), "float32"],
    bias: T.Buffer[(N,), "float16"],
):
    """
    融合的 INT8 GEMM：
    1. INT8 Tensor Core 矩阵乘法
    2. 反量化（应用 scale）
    3. 偏置加法
    4. 激活函数（可选）
    
    全部在 SRAM 中完成，减少全局内存访问
    """
    BM = 128
    BN = 128
    BK = 64
    
    with T.Block("root"):
        A_smem = T.alloc_shared([BM, BK], "int8")
        B_smem = T.alloc_shared([BN, BK], "int8")
        
        A_frag = T.alloc_fragment([BM, BK], "int8")
        B_frag = T.alloc_fragment([BN, BK], "int8")
        C_frag_int32 = T.alloc_fragment([BM, BN], "int32")
        C_frag_fp32 = T.alloc_fragment([BM, BN], "float32")
        C_frag_fp16 = T.alloc_fragment([BM, BN], "float16")
        
        # 累加器初始化
        T.clear(C_frag_int32)
        
        for k in T.serial(T.ceildiv(K, BK)):
            T.copy(A[by * BM : (by + 1) * BM, k * BK : (k + 1) * BK], A_smem)
            T.copy(B[bx * BN : (bx + 1) * BN, k * BK : (k + 1) * BK], B_smem)
            T.sync()
            
            T.copy(A_smem, A_frag)
            T.copy(B_smem, B_frag)
            
            T.gemm(A_frag, B_frag, C_frag_int32, "int8", "int8", "int32")
        
        # 融合反量化 + 偏置加法
        for i, j in T.Parallel(BM, BN):
            # INT32 -> FP32，应用两个 scale
            val = T.cast(C_frag_int32[i, j], "float32")
            val = val * scale_A[by * BM + i] * scale_B[bx * BN + j]
            # 加偏置
            val = val + T.cast(bias[bx * BN + j], "float32")
            # 可选：ReLU 激活
            # val = T.max(val, T.float32(0.0))
            C_frag_fp16[i, j] = T.cast(val, "float16")
        
        # 写回全局内存
        for i, j in T.Parallel(BM, BN):
            C[by * BM + i, bx * BN + j] = C_frag_fp16[i, j]
```

上述代码实现了融合的INT8 GEMM内核，将多个操作融合到一个内核中完成。首先使用INT8 Tensor Core进行矩阵乘法得到INT32结果，然后在寄存器中完成反量化（应用scale）、偏置加法和可选激活函数，最后写回FP16结果。这种融合策略显著减少了全局内存访问，提高了计算效率。

### 2.5 Pipeline 优化与 INT8 数据加载

```python
@T.prim_func
def int8_gemm_pipelined(
    A: T.Buffer[(M, K), "int8"],
    B: T.Buffer[(N, K), "int8"],
    C: T.Buffer[(M, N), "float16"],
    scale_A: T.Buffer[(M,), "float32"],
    scale_B: T.Buffer[(N,), "float32"],
):
    """
    带 Double Buffering Pipeline 的 INT8 GEMM
    
    Pipeline stages:
    Stage 0: 从全局内存加载 INT8 数据到 shared memory
    Stage 1: 从 shared memory 拷贝到 registers + dequantize
    Stage 2: Tensor Core GEMM 累加
    
    INT8 数据量仅为 FP16 的一半，pipeline 效果更显著
    """
    BM = 128
    BN = 128
    BK = 64
    NUM_STAGES = 2  # Double buffering
    
    with T.Block("root"):
        # 双缓冲共享内存
        A_smem = T.alloc_shared([NUM_STAGES, BM, BK], "int8")
        B_smem = T.alloc_shared([NUM_STAGES, BN, BK], "int8")
        
        A_frag = T.alloc_fragment([BM, BK], "int8")
        B_frag = T.alloc_fragment([BN, BK], "int8")
        C_frag = T.alloc_fragment([BM, BN], "int32")
        
        T.clear(C_frag)
        
        num_k_tiles = T.ceildiv(K, BK)
        
        # Prologue: 预取第一块
        T.copy(A[by * BM : (by + 1) * BM, 0:BK], A_smem[0])
        T.copy(B[bx * BN : (bx + 1) * BN, 0:BK], B_smem[0])
        T.sync()
        
        for k in T.serial(num_k_tiles):
            stage = k % NUM_STAGES
            next_stage = (k + 1) % NUM_STAGES
            
            # Prefetch 下一块到另一个 buffer
            if k + 1 < num_k_tiles:
                T.copy(
                    A[by * BM : (by + 1) * BM, (k + 1) * BK : (k + 2) * BK],
                    A_smem[next_stage]
                )
                T.copy(
                    B[bx * BN : (bx + 1) * BN, (k + 1) * BK : (k + 2) * BK],
                    B_smem[next_stage]
                )
            
            # 计算当前块
            T.copy(A_smem[stage], A_frag)
            T.copy(B_smem[stage], B_frag)
            
            T.gemm(A_frag, B_frag, C_frag, "int8", "int8", "int32")
            
            T.sync()
        
        # 反量化并写回
        for i, j in T.Parallel(BM, BN):
            val = T.cast(C_frag[i, j], "float32")
            val = val * scale_A[by * BM + i] * scale_B[bx * BN + j]
            C[by * BM + i, bx * BN + j] = T.cast(val, "float16")
```

上述代码实现了带双缓冲流水线的INT8 GEMM内核。通过使用两个共享内存缓冲区，实现数据加载和计算的重叠执行，隐藏内存延迟。INT8数据量仅为FP16的一半，因此流水线优化效果更显著。这种技术能有效提高计算吞吐量，特别适合内存带宽受限的场景。

### 2.6 INT8 GEMM 性能特征分析

```
┌─────────────────────────────────────────────────────────────┐
│              INT8 GEMM 内存访问模式                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Global Memory (HBM)          Shared Memory (SRAM)        │
│   ┌──────────────┐             ┌──────────────┐            │
│   │  A (INT8)    │ ──copy──→   │  A_smem      │            │
│   │  M × K bytes │  coalesced  │  BM × BK     │            │
│   └──────────────┘             └──────┬───────┘            │
│                                       │ dequant            │
│   ┌──────────────┐             ┌──────▼───────┐            │
│   │  B (INT8)    │ ──copy──→   │  B_smem      │            │
│   │  N × K bytes │  coalesced  │  BN × BK     │            │
│   └──────────────┘             └──────┬───────┘            │
│                                       │ dequant            │
│                                ┌──────▼───────┐            │
│                                │  Registers   │            │
│                                │  Tensor Core │            │
│                                │  INT8→INT32  │            │
│                                └──────┬───────┘            │
│                                       │                    │
│                                ┌──────▼───────┐            │
│   ┌──────────────┐             │  C_frag      │            │
│   │  C (FP16)   │ ←─copy───   │  FP32 → FP16 │            │
│   └──────────────┘             └──────────────┘            │
│                                                             │
│   带宽节省: A,B 从 FP16→INT8 节省 50% 内存带宽               │
│   计算提升: INT8 Tensor Core 吞吐量为 FP16 的 2×             │
└─────────────────────────────────────────────────────────────┘
```

这个代码块或示意图用于说明 2.6 INT8 GEMM 性能特征分析 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

---

## 3. FP8 GEMM

### 3.1 FP8 数据格式

FP8 是 NVIDIA 在 Hopper 架构（H100）中引入的 8 位浮点格式，有两种变体：

| 属性 | E4M3 | E5M2 |
|------|------|------|
| 总位数 | 8 | 8 |
| 符号位 | 1 | 1 |
| 指数位 | 4 | 5 |
| 尾数位 | 3 | 2 |
| 动态范围 | ±448 | ±57344 |
| 精度 | 较高 | 较低 |
| 典型用途 | 前向传播（权重、激活） | 反向传播（梯度） |

```
E4M3 格式:  S EEEE MMM
            │ │    │
            │ │    └── 3 位尾数 → 精度约 2^-3 = 0.125
            │ └─────── 4 位指数 → 偏移 7，范围 [2^-7, 2^8]
            └───────── 1 位符号

E5M2 格式:  S EEEEEE MM
            │ │      │
            │ │      └── 2 位尾数 → 精度约 2^-2 = 0.25
            │ └────────── 5 位指数 → 偏移 15，范围 [2^-15, 2^16]
            └──────────── 1 位符号
```

这个代码块或示意图用于说明 3.1 FP8 数据格式 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 3.2 FP8 量化与反量化

```python
import torch
import struct

def fp32_to_e4m3(x: float) -> int:
    """将 FP32 转换为 E4M3 格式（8-bit）"""
    E4M3_MAX = 448.0
    E4M3_MIN_POS = 2 ** (-9)  # 最小正规数
    
    # Clamp 到 E4M3 范围
    x = max(min(x, E4M3_MAX), -E4M3_MAX)
    
    if abs(x) < E4M3_MIN_POS:
        return 0
    
    # 简化实现：通过 FP16 中转
    x_fp16 = torch.tensor(x, dtype=torch.float16)
    # 使用 PyTorch 内置的 FP8 转换（需要 CUDA 支持）
    x_e4m3 = x_fp16.to(torch.float8_e4m3fn)
    return x_e4m3


def fp8_e4m3_quantize(x: torch.Tensor) -> torch.Tensor:
    """将 FP16/FP32 张量量化为 FP8 E4M3"""
    if x.dtype == torch.float32:
        x = x.half()
    return x.to(torch.float8_e4m3fn)


def fp8_e5m2_quantize(x: torch.Tensor) -> torch.Tensor:
    """将 FP16/FP32 张量量化为 FP8 E5M2（用于梯度）"""
    if x.dtype == torch.float32:
        x = x.half()
    return x.to(torch.float8_e5m2)
```

上述代码实现了FP8格式的量化函数，包括E4M3和E5M2两种变体。E4M3提供更高精度，适用于前向传播；E5M2提供更大动态范围，适用于反向传播。这些函数利用PyTorch内置的FP8转换，实际应用中需要CUDA支持。FP8是Hopper架构引入的新格式，能在保持较高精度的同时将内存占用减半。

### 3.3 Hopper FP8 Tensor Core

NVIDIA H100 的 FP8 Tensor Core 提供了惊人的计算吞吐量：

| GPU | FP16 Tensor Core | FP8 Tensor Core | 加速比 |
|-----|------------------|-----------------|--------|
| H100 SXM | 989 TFLOPS | 1979 TFLOPS | 2× |
| H100 PCIe | 756 TFLOPS | 1513 TFLOPS | 2× |
| H200 | 989 TFLOPS | 1979 TFLOPS | 2× |

```
Hopper FP8 Tensor Core 指令:
┌─────────────────────────────────────────────────────────┐
│  m16n8k32.e4m3.e4m3.f32                                 │
│  ├─ m16: 输出 M 维度 = 16                               │
│  ├─ n8:  输出 N 维度 = 8                                │
│  ├─ k32: 归约 K 维度 = 32 (FP8 的 K 比 INT8 更大)      │
│  ├─ e4m3: A 矩阵格式 = FP8 E4M3                        │
│  ├─ e4m3: B 矩阵格式 = FP8 E4M3                        │
│  └─ f32:  累加器格式 = FP32                             │
│                                                         │
│  m16n8k32.e4m3.e5m2.f32                                 │
│  ├─ 混合精度：A=E4M3, B=E5M2                            │
│  └─ 用于不同精度需求的场景                               │
└─────────────────────────────────────────────────────────┘
```

这个代码块或示意图用于说明 3.3 Hopper FP8 Tensor Core 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 3.4 TileLang FP8 GEMM 实现

```python
@T.prim_func
def fp8_gemm_hopper(
    A: T.Buffer[(M, K), "e4m3"],
    B: T.Buffer[(N, K), "e4m3"],
    C: T.Buffer[(M, N), "float32"],
    scale_A: T.Buffer[(M,), "float32"],
    scale_B: T.Buffer[(N,), "float32"],
):
    """
    Hopper FP8 GEMM 实现
    
    使用 FP8 Tensor Core (m16n8k32.e4m3.e4m3.f32)
    带 per-tensor scale 的反量化融合
    """
    BM = 128
    BN = 128
    BK = 128  # FP8 Tensor Core K=32，使用 128 以提高效率
    
    with T.Block("root"):
        # 共享内存分配
        A_smem = T.alloc_shared([BM, BK], "e4m3")
        B_smem = T.alloc_shared([BN, BK], "e4m3")
        
        # Fragment 分配
        A_frag = T.alloc_fragment([BM, BK], "e4m3")
        B_frag = T.alloc_fragment([BN, BK], "e4m3")
        C_frag = T.alloc_fragment([BM, BN], "float32")
        
        T.clear(C_frag)
        
        for k in T.serial(T.ceildiv(K, BK)):
            # 协同加载 FP8 数据
            T.copy(A[by * BM : (by + 1) * BM, k * BK : (k + 1) * BK], A_smem)
            T.copy(B[bx * BN : (bx + 1) * BN, k * BK : (k + 1) * BK], B_smem)
            T.sync()
            
            # 拷贝到 Fragment
            T.copy(A_smem, A_frag)
            T.copy(B_smem, B_frag)
            
            # FP8 Tensor Core GEMM
            # 硬件自动处理 FP8 乘法和 FP32 累加
            T.gemm(A_frag, B_frag, C_frag, "e4m3", "e4m3", "float32")
        
        # 应用 scale 反量化并写回
        for i, j in T.Parallel(BM, BN):
            C[by * BM + i, bx * BN + j] = C_frag[i, j] * scale_A[by * BM + i] * scale_B[bx * BN + j]
```

上述代码实现了基于Hopper架构的FP8 GEMM内核。使用FP8 Tensor Core（m16n8k32.e4m3.e4m3.f32指令）进行矩阵乘法，累加结果为FP32。该实现融合了per-tensor scale的反量化，在计算完成后直接应用scale并写回结果。FP8相比INT8提供更灵活的精度-动态范围权衡，适合深度学习推理。

### 3.5 FP8 混合精度 GEMM

```python
@T.prim_func
def fp8_mixed_gemm(
    A: T.Buffer[(M, K), "e4m3"],     # 激活使用 E4M3
    B: T.Buffer[(N, K), "e5m2"],     # 权重梯度使用 E5M2
    C: T.Buffer[(M, N), "float32"],
):
    """
    混合精度 FP8 GEMM
    A 使用 E4M3（高精度），B 使用 E5M2（大动态范围）
    对应指令: m16n8k32.e4m3.e5m2.f32
    """
    BM = 128
    BN = 128
    BK = 128
    
    with T.Block("root"):
        A_smem = T.alloc_shared([BM, BK], "e4m3")
        B_smem = T.alloc_shared([BN, BK], "e5m2")
        
        A_frag = T.alloc_fragment([BM, BK], "e4m3")
        B_frag = T.alloc_fragment([BN, BK], "e5m2")
        C_frag = T.alloc_fragment([BM, BN], "float32")
        
        T.clear(C_frag)
        
        for k in T.serial(T.ceildiv(K, BK)):
            T.copy(A[by * BM : (by + 1) * BM, k * BK : (k + 1) * BK], A_smem)
            T.copy(B[bx * BN : (bx + 1) * BN, k * BK : (k + 1) * BK], B_smem)
            T.sync()
            
            T.copy(A_smem, A_frag)
            T.copy(B_smem, B_frag)
            
            # 混合精度 Tensor Core GEMM
            T.gemm(A_frag, B_frag, C_frag, "e4m3", "e5m2", "float32")
        
        for i, j in T.Parallel(BM, BN):
            C[by * BM + i, bx * BN + j] = C_frag[i, j]
```

上述代码实现了混合精度FP8 GEMM内核。这个内核的特点是A矩阵使用E4M3格式（提供更高精度），而B矩阵使用E5M2格式（提供更大动态范围），对应Hopper GPU的混合精度Tensor Core指令m16n8k32.e4m3.e5m2.f32。E5M2有更大的指数范围（±57344），适合存储梯度等动态范围大的数据；E4M3则有更高的尾数精度，适合权重和激活值。在实际训练中，这种混合精度策略可以有效平衡精度和动态范围，前向传播使用E4M3，反向传播的梯度使用E5M2，从而最大化FP8量化训练的效果。

### 3.6 FP8 训练中的缩放策略

```python
class FP8ScalingStrategy:
    """
    FP8 训练中的动态缩放策略
    
    关键思想：使用历史最大值来预测合适的 scale，
    避免溢出的同时最大化精度利用
    """
    
    def __init__(self, history_len: int = 1024, margin: float = 0.0):
        self.history_len = history_len
        self.margin = margin
        self.history = []
        self.fp8_max = 448.0  # E4M3 最大值
    
    def compute_scale(self, x: torch.Tensor) -> float:
        """计算最优缩放因子"""
        abs_max = x.abs().max().item()
        self.history.append(abs_max)
        if len(self.history) > self.history_len:
            self.history.pop(0)
        
        # 使用历史最大值的指数移动平均
        ema_max = sum(self.history[-128:]) / min(len(self.history), 128)
        
        # 计算 scale，使得 ema_max 映射到 FP8 最大值
        scale = self.fp8_max / (ema_max + 1e-10)
        
        # 添加安全裕度
        scale = scale / (2 ** self.margin)
        
        return scale
    
    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        """量化并返回 scale"""
        scale = self.compute_scale(x)
        x_scaled = x * scale
        x_fp8 = x_scaled.to(torch.float8_e4m3fn)
        return x_fp8, scale
```

上述代码实现了FP8训练中的动态缩放策略类。核心思想是根据历史最大值的历史记录预测合适的缩放因子scale，将数据映射到FP8 E4M3的有限范围（±448）内。compute_scale函数维护一个历史最大值的滑动窗口，使用最近128个值的指数移动平均来估算当前数据幅度，然后计算能将峰值映射到FP8最大值的缩放因子。quantize函数先缩放再转换为FP8格式。这种策略解决了FP8训练的关键挑战：scale过大会导致溢出产生inf，scale过小会浪费可用精度范围，通过历史统计和EMA平滑来找到最优平衡点。

> [!WARNING]
> FP8 训练需要仔细调整缩放因子。过大的 scale 会导致溢出（inf），过小的 scale 会浪费精度。建议使用延迟缩放（deferred scaling）策略。

---

## 4. INT4/NF4 GEMM

### 4.1 4-bit 量化的挑战

4-bit 量化面临更大的挑战：

| 挑战 | 说明 | 解决方案 |
|------|------|---------|
| 精度损失大 | 仅 16 个量化级别 | 更好的量化算法 |
| 异常值处理 | 极端值会压缩正常值范围 | 分离异常值 / per-group |
| 计算效率 | 4-bit 解包开销 | 位操作优化 |
| 内存对齐 | 非标准位宽 | 特殊内存布局 |

### 4.2 INT4 量化实现

```python
def int4_quantize_per_group(
    x: torch.Tensor,
    group_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    INT4 Per-group 量化
    
    将权重按 group_size 分组，每组独立量化为 INT4（-8 到 7）
    """
    original_shape = x.shape
    assert x.shape[-1] % group_size == 0
    
    # Reshape 为 [num_groups, group_size]
    x_groups = x.reshape(-1, group_size)
    num_groups = x_groups.shape[0]
    
    # 计算每组的 scale
    x_max = x_groups.abs().max(dim=1).values
    scale = x_max / 7.0  # INT4 symmetric: [-8, 7] → 使用 [-7, 7]
    scale = scale.clamp(min=1e-10)
    
    # 量化
    x_scaled = x_groups / scale.unsqueeze(1)
    x_q = torch.clamp(torch.round(x_scaled), -8, 7).to(torch.int8)
    
    # 打包两个 INT4 到一个 INT8
    x_q = x_q.reshape(original_shape)
    x_packed = pack_int4(x_q)
    
    return x_packed, scale.reshape(original_shape[:-1] + (-1,))


def pack_int4(x_q: torch.Tensor) -> torch.Tensor:
    """将两个 INT4 值打包到一个 INT8 字节中"""
    # x_q 范围: [-8, 7]，映射到 [0, 15] 用于存储
    x_u4 = (x_q + 8).to(torch.uint8)
    
    # 打包：低4位和高4位
    shape = x_u4.shape
    x_flat = x_u4.reshape(-1)
    
    # 确保长度为偶数
    if x_flat.shape[0] % 2 != 0:
        x_flat = torch.cat([x_flat, torch.zeros(1, dtype=torch.uint8)])
    
    x_even = x_flat[0::2]  # 偶数位
    x_odd = x_flat[1::2]   # 奇数位
    
    packed = x_even | (x_odd << 4)
    return packed


def unpack_int4(packed: torch.Tensor, original_shape: tuple) -> torch.Tensor:
    """解包 INT4 数据"""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    
    unpacked = torch.zeros(packed.shape[0] * 2, dtype=torch.uint8)
    unpacked[0::2] = low
    unpacked[1::2] = high
    
    # 映射回 [-8, 7]
    return (unpacked[:torch.prod(torch.tensor(original_shape))].reshape(original_shape).to(torch.int16) - 8).to(torch.int8)
```

上述代码实现了INT4的打包和解包操作，是4-bit量化推理的基础工具函数。int4_quantize_per_group函数将权重矩阵按group_size分组后独立量化到[-8,7]范围，每组的scale由该组的绝对值最大值除以7得出。pack_int4函数将两个INT4值打包到一个uint8字节中，低4位存偶数位、高4位存奇数位，从而将内存占用减半。unpack_int4函数执行逆向操作，通过位掩码分离高低半字节并重建原始形状。这种位级操作的效率直接影响量化推理的吞吐量，在CUDA kernel中通常使用位操作指令实现。

### 4.3 GPTQ 算法

GPTQ（GPT Quantization）是一种基于近似二阶信息的权重量化算法：

```
GPTQ 核心思想：
1. 逐层量化权重矩阵
2. 使用 Hessian 信息指导量化顺序
3. 量化一个权重后，补偿剩余权重的误差

算法流程：
┌─────────────────────────────────────────────────────┐
│  输入: 权重矩阵 W, 校准数据 X                        │
│                                                     │
│  1. 计算 Hessian: H = 2 * X^T * X                   │
│  2. 初始化: W_q = W.copy()                          │
│  3. 对每一列 j (按特定顺序):                         │
│     a. 量化 W_q[:, j] 得到 w_q                      │
│     b. 计算量化误差: δ = W_q[:, j] - w_q            │
│     c. 将误差分配到未量化列:                         │
│        W_q[:, j+1:] -= δ * H[j, j+1:] / H[j, j]   │
│  4. 返回量化后的 W_q 和 scales                       │
└─────────────────────────────────────────────────────┘
```

这个代码块或示意图用于说明 4.3 GPTQ 算法 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

```python
import torch
import numpy as np

class GPTQQuantizer:
    """
    GPTQ 量化器
    
    基于近似二阶信息的权重量化算法，
    通过误差补偿最小化量化对模型输出的影响
    """
    
    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        perchannel: bool = True,
        sym: bool = False,
        mse: bool = True,
    ):
        self.bits = bits
        self.group_size = group_size
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.q_max = 2 ** (bits - 1) - 1 if sym else 2 ** bits - 1
        self.q_min = -self.q_max if sym else 0
    
    def quantize(
        self,
        W: torch.Tensor,
        H: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        GPTQ 量化主函数
        
        Args:
            W: 权重矩阵 [out_features, in_features]
            H: Hessian 矩阵 [in_features, in_features]
        
        Returns:
            W_q: 量化后的权重
            scales: 缩放因子
            zeros: 零点（非对称量化）
        """
        rows, cols = W.shape
        W_q = W.clone()
        
        # 计算 Hessian 的逆（Cholesky 分解加速）
        H_inv = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(H_inv)
        H_inv = torch.linalg.cholesky(H_inv, upper=True)
        
        # 初始化量化参数
        scales = torch.zeros(rows, dtype=torch.float16)
        zeros = torch.zeros(rows, dtype=torch.int8)
        
        # 逐列量化
        for j in range(cols):
            # 当前列的权重
            w = W_q[:, j]
            
            # 量化当前列
            if self.perchannel:
                scale = w.abs().max() / self.q_max
            else:
                scale = W.abs().max() / self.q_max
            
            scale = scale.clamp(min=1e-10)
            w_q = torch.clamp(torch.round(w / scale), self.q_min, self.q_max)
            
            # 计算量化误差
            error = w - w_q * scale
            
            # 误差补偿：将误差分配到后续列
            if j < cols - 1:
                # 使用 Hessian 信息加权补偿
                compensation = error.unsqueeze(1) * H_inv[j, j+1:].unsqueeze(0)
                W_q[:, j+1:] -= compensation
            
            # 保存量化结果
            W_q[:, j] = w_q
            scales[j // self.group_size] = scale
        
        return W_q.to(torch.int8), scales, zeros
```

上述代码实现了GPTQ量化器类，这是目前最主流的大模型权重量化算法之一。核心思想是逐列量化权重矩阵并利用Hessian矩阵的逆来补偿后续列的量化误差。具体步骤是：先通过Cholesky分解计算Hessian矩阵的逆，然后逐列扫描权重——量化当前列后计算量化误差，再根据Hessian逆矩阵将误差按比例分配到尚未量化的后续列上。这种误差补偿机制使得GPTQ能以4-bit精度保持模型质量，per-channel策略为每列独立计算scale更好地适应不同通道的分布差异。实践中GPTQ能让7B模型INT4量化后困惑度仅增加约0.5。

### 4.4 AWQ 算法

AWQ（Activation-aware Weight Quantization）基于激活值分布来保护重要权重通道：

```
AWQ 核心思想：
1. 不是所有权重通道同等重要
2. 激活值大的通道对应更重要的权重
3. 对重要通道使用更细粒度的量化

算法流程：
┌─────────────────────────────────────────────────────┐
│  1. 用少量校准数据运行模型，收集激活统计              │
│  2. 计算每个通道的重要性: s = |X|^α                  │
│  3. 对权重应用缩放: W' = W * s                       │
│  4. 对缩放后的 W' 进行标准量化                       │
│  5. 推理时: Y = X/s * quant(W')                     │
│                                                     │
│  关键：搜索最优的 α 值                               │
└─────────────────────────────────────────────────────┘
```

这个代码块或示意图用于说明 4.4 AWQ 算法 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

```python
class AWQQuantizer:
    """
    AWQ (Activation-aware Weight Quantization) 量化器
    """
    
    def __init__(self, bits: int = 4, group_size: int = 128):
        self.bits = bits
        self.group_size = group_size
    
    def find_salient_channels(
        self,
        W: torch.Tensor,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """
        找到重要通道
        
        基于激活值的统计来确定哪些权重通道更重要
        """
        # 计算每个输入通道的激活幅度
        channel_importance = X.abs().mean(dim=0)
        return channel_importance
    
    def search_scale(
        self,
        W: torch.Tensor,
        X: torch.Tensor,
        channel_importance: torch.Tensor,
        alpha_range: tuple = (0.0, 1.0),
        n_steps: int = 20,
    ) -> float:
        """
        搜索最优缩放因子 α
        
        s_j = |X_j|^α，其中 j 是通道索引
        """
        best_alpha = 0.5
        best_error = float('inf')
        
        for alpha in np.linspace(alpha_range[0], alpha_range[1], n_steps):
            # 计算缩放向量
            s = channel_importance ** alpha
            
            # 缩放权重
            W_scaled = W * s.unsqueeze(0)
            
            # 量化
            W_q = self._simple_quantize(W_scaled)
            
            # 反量化
            W_deq = self._simple_dequantize(W_q)
            
            # 计算输出误差
            Y_orig = X @ W.T
            Y_deq = X @ (W_deq / s.unsqueeze(0)).T
            error = (Y_orig - Y_deq).pow(2).mean().item()
            
            if error < best_error:
                best_error = error
                best_alpha = alpha
        
        return best_alpha
    
    def quantize(
        self,
        W: torch.Tensor,
        X: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        AWQ 量化主函数
        
        Args:
            W: 权重矩阵 [out, in]
            X: 校准激活 [batch, in]
        """
        # Step 1: 计算通道重要性
        importance = self.find_salient_channels(W, X)
        
        # Step 2: 搜索最优 α
        alpha = self.search_scale(W, X, importance)
        
        # Step 3: 计算缩放并量化
        s = importance ** alpha
        W_scaled = W * s.unsqueeze(0)
        
        # Per-group 量化
        W_q, scales = self._per_group_quantize(W_scaled)
        
        return W_q, scales, s  # 返回缩放向量用于推理时反缩放
```

上述代码实现了AWQ量化器类。AWQ的核心创新是根据激活值的分布来识别重要通道并加以保护：find_salient_channels函数通过统计校准数据的激活幅度来确定每个输入通道的重要性；search_scale函数在[0,1]范围内搜索最优的alpha指数，使得缩放后量化的输出与原始输出之间的均方误差最小；quantize函数将缩放后的权重进行per-group量化。与GPTQ不同，AWQ不需要计算复杂的Hessian矩阵，而是通过简单的激活统计和线搜索找到最优缩放策略，实现更简单且效果相当。推理时需要将输入激活除以缩放向量s来恢复等效计算。

### 4.5 NF4 格式

NF4（NormalFloat 4-bit）是 QLoRA 提出的量化格式，专为正态分布数据设计：

```
NF4 量化级别（基于标准正态分布的等概率分位数）：

值:    -1.0  -0.693  -0.525  -0.395  -0.284  -0.185  -0.091  0.0
索引:   0     1       2       3       4       5       6       7

值:    0.079  0.161   0.249   0.342   0.448   0.571   0.723   1.0
索引:   8      9       10      11      12      13      14      15

特点：
- 量化级别不是等间距的，而是按正态分布概率等间距
- 对于接近零的小值有更高的精度
- 对于极端值（大值）精度较低但足以表示
```

这个代码块或示意图用于说明 4.5 NF4 格式 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

```python
def create_nf4_quantile_map() -> torch.Tensor:
    """
    创建 NF4 量化级别表
    
    基于标准正态分布的等概率分位数
    """
    import scipy.stats as stats
    
    n_levels = 16
    # 等概率分位点
    quantiles = torch.linspace(0, 1, n_levels + 1)
    # 映射到正态分布
    nf4_values = torch.tensor([
        stats.norm.ppf(q) for q in quantiles.numpy()
    ])
    # 归一化到 [-1, 1]
    nf4_values = nf4_values / nf4_values.abs().max()
    
    return nf4_values


def nf4_quantize(
    x: torch.Tensor,
    block_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    NF4 量化
    
    Args:
        x: 输入张量
        block_size: 量化块大小
    
    Returns:
        indices: 4-bit 索引（打包为 int8）
        scales: 每个块的缩放因子
    """
    nf4_map = create_nf4_quantile_map()
    
    original_shape = x.shape
    x_flat = x.reshape(-1, block_size)
    num_blocks = x_flat.shape[0]
    
    # 计算每个块的 scale
    scales = x_flat.abs().max(dim=1).values
    scales = scales.clamp(min=1e-10)
    
    # 归一化到 [-1, 1]
    x_norm = x_flat / scales.unsqueeze(1)
    
    # 找到最近的 NF4 级别
    distances = (x_norm.unsqueeze(-1) - nf4_map.unsqueeze(0).unsqueeze(0)).abs()
    indices = distances.argmin(dim=-1).to(torch.uint8)
    
    # 打包两个 4-bit 索引到一个 8-bit 字节
    indices_flat = indices.reshape(-1)
    if indices_flat.shape[0] % 2 != 0:
        indices_flat = torch.cat([indices_flat, torch.zeros(1, dtype=torch.uint8)])
    
    packed = indices_flat[0::2] | (indices_flat[1::2] << 4)
    
    return packed, scales


def nf4_dequantize(
    packed: torch.Tensor,
    scales: torch.Tensor,
    original_shape: tuple,
    block_size: int = 64,
) -> torch.Tensor:
    """NF4 反量化"""
    nf4_map = create_nf4_quantile_map()
    
    # 解包
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    
    total_elements = 1
    for s in original_shape:
        total_elements *= s
    
    indices = torch.zeros(total_elements, dtype=torch.uint8)
    indices[0::2] = low
    indices[1::2] = high
    indices = indices[:total_elements]
    
    # 查表得到归一化值
    x_norm = nf4_map[indices.long()]
    
    # 恢复原始值
    x_flat = x_norm.reshape(-1, block_size) * scales.unsqueeze(1)
    
    return x_flat.reshape(original_shape)
```

上述代码实现了NF4格式的量化与反量化函数。NF4基于标准正态分布的等概率分位数，将16个量化级别非均匀地分布在正态分布的概率区间上，使得接近零的小值有更高的精度。create_nf4_quantile_map函数通过scipy的正态分位数函数生成16级量化表并归一化到[-1,1]。nf4_quantize函数将输入按block_size分块，每块独立计算scale后查找最近的NF4级别索引。nf4_dequantize函数通过查表恢复归一化值再乘以scale重建原始数据。这种非均匀量化使得NF4相比INT4在处理正态分布数据（如大模型权重）时精度更高。

### 4.6 GGUF 格式

GGUF（GGML Universal Format）是 llama.cpp 使用的模型格式，支持多种量化类型：

| 量化类型 | 比特数 | 块大小 | 说明 |
|---------|--------|--------|------|
| Q4_0 | 4.5 | 32 | 4-bit 量化 + block scale |
| Q4_1 | 5.0 | 32 | 4-bit + scale + min |
| Q4_K | 4.83 | 256 | K-quant, 4-bit 混合 |
| Q5_K | 5.68 | 256 | K-quant, 5-bit 混合 |
| Q6_K | 6.56 | 256 | K-quant, 6-bit 混合 |
| Q8_0 | 8.5 | 32 | 8-bit + block scale |
| IQ4_XS | 4.25 | 256 | importance quantization |

```python
# GGUF Q4_0 量化格式
"""
Q4_0 block 格式 (每 32 个权重):
┌──────────────────────────────────────┐
│  scale (FP16, 2 bytes)              │  <- 块的缩放因子
├──────────────────────────────────────┤
│  q0 q1 (16 bytes, 每字节2个4-bit)    │  <- 32 个量化值
└──────────────────────────────────────┘
总大小: 2 + 16 = 18 bytes / 32 weights
有效比特数: 18 * 8 / 32 = 4.5 bits
"""


def gguf_q4_0_quantize(x: torch.Tensor, block_size: int = 32):
    """
    GGUF Q4_0 量化
    
    每 block_size 个值共享一个 FP16 scale
    """
    assert x.shape[-1] % block_size == 0
    
    x_flat = x.reshape(-1, block_size)
    num_blocks = x_flat.shape[0]
    
    # 计算 scale (使用绝对值最大值)
    abs_max = x_flat.abs().max(dim=1).values
    scale = abs_max / 7.0  # INT4 symmetric: [-8, 7]
    scale = scale.half()
    
    # 量化为 [-8, 7]
    x_scaled = x_flat / scale.float().unsqueeze(1)
    x_q = torch.clamp(torch.round(x_scaled), -8, 7).to(torch.int8)
    
    # 偏移映射到 [0, 15]
    x_u4 = (x_q + 8).to(torch.uint8)
    
    # 打包两个 4-bit 到一个字节
    packed = x_u4[:, 0::2] | (x_u4[:, 1::2] << 4)
    
    return packed, scale
```

上述代码实现了GGUF格式的Q4_0量化函数。GGUF是llama.cpp使用的模型格式，Q4_0将每32个权重作为一个block，共享一个FP16 scale因子，有效比特数为4.5位。量化过程先计算每block的绝对值最大值作为量化上限，除以7得到scale因子（使用[-8,7]的对称INT4范围），然后将权重缩放到量化范围并打包为每字节两个4-bit值。这种block-level量化的设计在精度和存储效率之间取得了良好平衡，是当前开源模型推理最主流的量化方案。

### 4.7 TileLang INT4 Dequantize GEMM

```python
@T.prim_func
def int4_dequant_gemm(
    A: T.Buffer[(M, K), "float16"],
    B_packed: T.Buffer[(N, K // 2), "uint8"],  # 两个 INT4 打包
    C: T.Buffer[(M, N), "float16"],
    scales: T.Buffer[(N, K // GROUP_SIZE), "float16"],
    zeros: T.Buffer[(N, K // GROUP_SIZE), "float16"],
):
    """
    INT4 Dequantize GEMM
    
    策略：
    1. 从 shared memory 加载打包的 INT4 权重
    2. 在 registers 中解包并反量化为 FP16
    3. 使用 FP16 Tensor Core 执行 GEMM
    
    优势：内存带宽仅为 FP16 的 1/4
    """
    BM = 128
    BN = 128
    BK = 64  # K 方向分块，需保证 GROUP_SIZE 整除
    GROUP_SIZE = 128
    
    with T.Block("root"):
        # 共享内存（存储打包的 INT4）
        B_smem = T.alloc_shared([BN, BK // 2], "uint8")
        scale_smem = T.alloc_shared([BN, BK // GROUP_SIZE], "float16")
        zero_smem = T.alloc_shared([BN, BK // GROUP_SIZE], "float16")
        
        # Fragment
        A_frag = T.alloc_fragment([BM, BK], "float16")
        B_frag = T.alloc_fragment([BN, BK], "float16")  # 反量化后的 FP16
        C_frag = T.alloc_fragment([BM, BN], "float32")
        
        T.clear(C_frag)
        
        for k in T.serial(T.ceildiv(K, BK)):
            # 加载打包的 INT4 权重
            T.copy(B_packed[bx * BN : (bx + 1) * BN, k * BK // 2 : (k + 1) * BK // 2], B_smem)
            T.copy(scales[bx * BN : (bx + 1) * BN, k * BK // GROUP_SIZE : (k + 1) * BK // GROUP_SIZE], scale_smem)
            T.copy(zeros[bx * BN : (bx + 1) * BN, k * BK // GROUP_SIZE : (k + 1) * BK // GROUP_SIZE], zero_smem)
            T.sync()
            
            # Dequantize INT4 -> FP16 在 registers 中完成
            for i, j in T.Parallel(BN, BK):
                # 解包 INT4
                byte_idx = j // 2
                is_high = j % 2
                packed_byte = B_smem[i, byte_idx]
                
                if is_high:
                    val_4bit = (T.cast(packed_byte, "int8") >> 4) & 0x0F
                else:
                    val_4bit = T.cast(packed_byte, "int8") & 0x0F
                
                # 映射到 [-8, 7]
                val_signed = T.cast(val_4bit, "int8") - 8
                
                # 应用 scale 和 zero_point
                group_idx = j // GROUP_SIZE
                s = scale_smem[i, group_idx]
                z = zero_smem[i, group_idx]
                B_frag[i, j] = T.cast(val_signed, "float16") * s + z
            
            # 加载激活（已经是 FP16）
            T.copy(A[by * BM : (by + 1) * BM, k * BK : (k + 1) * BK], A_frag)
            T.sync()
            
            # FP16 GEMM
            T.gemm(A_frag, B_frag, C_frag, "float16", "float16", "float32")
        
        # 写回结果
        for i, j in T.Parallel(BM, BN):
            C[by * BM + i, bx * BN + j] = T.cast(C_frag[i, j], "float16")
```

上述代码是TileLang INT4反量化GEMM的完整实现，采用dequantize-then-multiply策略。核心创新在于将INT4解包和反量化操作融合到tile循环中：从共享内存加载打包的uint8权重，在寄存器中通过位操作解包两个INT4值，然后应用scale和zero_point进行反量化，最后使用FP16 Tensor Core进行GEMM。这种融合策略避免了先将权重大量反量化再存储到HBM的中间步骤，直接减少约75%的内存带宽需求。GROUP_SIZE参数控制粒度，值越小精度越高但存储scale开销越大。

### 4.8 TileLang NF4 Dequantize GEMM

```python
@T.prim_func
def nf4_dequant_gemm(
    A: T.Buffer[(M, K), "float16"],
    B_packed: T.Buffer[(N, K // 2), "uint8"],
    C: T.Buffer[(M, N), "float16"],
    scales: T.Buffer[(N, K // BLOCK_SIZE), "float16"],
    nf4_table: T.Buffer[(16,), "float16"],
):
    """
    NF4 Dequantize GEMM
    
    1. 从 NF4 lookup table 查找归一化值
    2. 应用 block scale 恢复原始值
    3. 执行 FP16 GEMM
    """
    BM = 128
    BN = 128
    BK = 64
    BLOCK_SIZE = 64
    
    with T.Block("root"):
        # 共享内存
        B_smem = T.alloc_shared([BN, BK // 2], "uint8")
        scale_smem = T.alloc_shared([BN, BK // BLOCK_SIZE], "float16")
        nf4_smem = T.alloc_shared([16], "float16")  # NF4 查找表
        
        # Fragment
        A_frag = T.alloc_fragment([BM, BK], "float16")
        B_frag = T.alloc_fragment([BN, BK], "float16")
        C_frag = T.alloc_fragment([BM, BN], "float32")
        
        T.clear(C_frag)
        
        # 加载 NF4 查找表到共享内存
        T.copy(nf4_table, nf4_smem)
        T.sync()
        
        for k in T.serial(T.ceildiv(K, BK)):
            T.copy(B_packed[bx * BN : (bx + 1) * BN, k * BK // 2 : (k + 1) * BK // 2], B_smem)
            T.copy(scales[bx * BN : (bx + 1) * BN, k * BK // BLOCK_SIZE : (k + 1) * BK // BLOCK_SIZE], scale_smem)
            T.sync()
            
            # NF4 Dequantize
            for i, j in T.Parallel(BN, BK):
                byte_idx = j // 2
                is_high = j % 2
                packed_byte = B_smem[i, byte_idx]
                
                if is_high:
                    idx = T.cast((packed_byte >> 4) & 0x0F, "int32")
                else:
                    idx = T.cast(packed_byte & 0x0F, "int32")
                
                # 查找 NF4 归一化值
                norm_val = nf4_smem[idx]
                
                # 应用 block scale
                block_idx = j // BLOCK_SIZE
                s = scale_smem[i, block_idx]
                B_frag[i, j] = norm_val * s
            
            T.copy(A[by * BM : (by + 1) * BM, k * BK : (k + 1) * BK], A_frag)
            T.sync()
            
            T.gemm(A_frag, B_frag, C_frag, "float16", "float16", "float32")
        
        for i, j in T.Parallel(BM, BN):
            C[by * BM + i, bx * BN + j] = T.cast(C_frag[i, j], "float16")
```

上述代码是TileLang NF4反量化GEMM的实现，专为正态分布数据优化。与INT4的线性量化不同，NF4使用16级非均匀量化表，通过预计算的nf4_table查找表将4-bit索引映射为归一化值。内核中NF4查找表被预加载到共享内存，解包过程中的每个4-bit索引通过查表获取归一化值，再乘以block scale得到FP16的B矩阵元素。这种查表法虽然增加了一次共享内存访问，但非均匀量化带来的精度提升在4-bit场景下非常显著，尤其在QLoRA等低比特微调场景中至关重要。

---


### 5.1 混合精度矩阵乘法

在大模型推理中，权重量化（Weight-only Quantization）是最常用的策略：

```
混合精度矩阵乘法: Y = X @ W^T

X (激活): FP16/BF16 - 保持高精度
W (权重): INT4/NF4  - 大幅压缩

流程:
┌───────────────────────────────────────────────┐
│  FP16 激活 X                                  │
│      │                                        │
│      ▼                                        │
│  ┌─────────────────────────────────┐          │
│  │  Dequantize Kernel              │          │
│  │  INT4 W → FP16 W'              │          │
│  │  (在 SRAM 中完成)               │          │
│  └─────────────┬───────────────────┘          │
│                │                              │
│                ▼                              │
│  ┌─────────────────────────────────┐          │
│  │  FP16 GEMM                      │          │
│  │  Y = X @ W'^T                  │          │
│  │  (使用 FP16 Tensor Core)        │          │
│  └─────────────┬───────────────────┘          │
│                │                              │
│                ▼                              │
│  FP16 输出 Y                                  │
└───────────────────────────────────────────────┘

优势:
- 权重存储: 4 bits/weight (vs 16 bits/weight)
- 内存带宽节省: 约 75%
- 计算: 仍使用高效的 FP16 Tensor Core
```

这个代码块或示意图用于说明 5.1 混合精度矩阵乘法 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 5.2 内核融合策略

```python
@T.prim_func
def weight_only_int4_gemm_fused(
    X: T.Buffer[(M, K), "float16"],
    W_q: T.Buffer[(N, K // 2), "uint8"],
    Y: T.Buffer[(M, N), "float16"],
    scales: T.Buffer[(N, K // GROUP_SIZE), "float16"],
    bias: T.Buffer[(N,), "float16"],
):
    """
    融合的 Weight-only INT4 GEMM
    
    融合操作:
    1. INT4 解包 + 反量化
    2. FP16 矩阵乘法
    3. 偏置加法
    4. 可选激活函数
    
    全部在 SRAM/Registers 中完成，只写回最终结果
    """
    BM = 128
    BN = 128
    BK = 64
    GROUP_SIZE = 128
    
    with T.Block("root"):
        # 共享内存
        X_smem = T.alloc_shared([BM, BK], "float16")
        W_smem = T.alloc_shared([BN, BK // 2], "uint8")
        scale_smem = T.alloc_shared([BN, BK // GROUP_SIZE], "float16")
        
        # Fragment
        X_frag = T.alloc_fragment([BM, BK], "float16")
        W_frag = T.alloc_fragment([BN, BK], "float16")
        Y_frag = T.alloc_fragment([BM, BN], "float32")
        
        T.clear(Y_frag)
        
        for k in T.serial(T.ceildiv(K, BK)):
            # 协同加载
            T.copy(X[by * BM : (by + 1) * BM, k * BK : (k + 1) * BK], X_smem)
            T.copy(W_q[bx * BN : (bx + 1) * BN, k * BK // 2 : (k + 1) * BK // 2], W_smem)
            T.copy(scales[bx * BN : (bx + 1) * BN, k * BK // GROUP_SIZE : (k + 1) * BK // GROUP_SIZE], scale_smem)
            T.sync()
            
            # Dequantize W INT4 -> FP16 (in registers)
            for i, j in T.Parallel(BN, BK):
                byte_idx = j // 2
                is_high = j % 2
                packed_byte = W_smem[i, byte_idx]
                
                if is_high:
                    val_4bit = (T.cast(packed_byte, "int8") >> 4) & 0x0F
                else:
                    val_4bit = T.cast(packed_byte, "int8") & 0x0F
                
                val_signed = T.cast(val_4bit, "int8") - 8
                group_idx = j // GROUP_SIZE
                W_frag[i, j] = T.cast(val_signed, "float16") * scale_smem[i, group_idx]
            
            # 加载 X 到 Fragment
            T.copy(X_smem, X_frag)
            T.sync()
            
            # FP16 GEMM
            T.gemm(X_frag, W_frag, Y_frag, "float16", "float16", "float32")
        
        # 融合偏置加法并写回
        for i, j in T.Parallel(BM, BN):
            val = T.cast(Y_frag[i, j], "float16") + bias[bx * BN + j]
            Y[by * BM + i, bx * BN + j] = val
```

上述代码实现了融合的Weight-only INT4 GEMM内核。该内核将INT4反量化、FP16矩阵乘法和偏置加法融合到一个GPU kernel中，避免了中间结果的HBM访问。激活X保持FP16格式以确保计算精度，而权重W以打包的uint8存储以节省内存。在内循环中，每个tile的权重先被解包并反量化为FP16（在寄存器中完成位操作），然后激活和反量化后的权重都加载到fragment中通过T.gemm进行FP16矩阵乘法。这种融合策略使得推理时的内存带宽需求降低约75%，特别适合大模型推理场景中权重加载成为瓶颈的情况。

### 5.3 双缓冲 Pipeline 权重量化 GEMM

```python
@T.prim_func
def weight_only_int4_pipelined(
    X: T.Buffer[(M, K), "float16"],
    W_q: T.Buffer[(N, K // 2), "uint8"],
    Y: T.Buffer[(M, N), "float16"],
    scales: T.Buffer[(N, K // GROUP_SIZE), "float16"],
):
    """
    带双缓冲 Pipeline 的 Weight-only INT4 GEMM
    
    Pipeline 利用内存加载和计算的重叠，最大化吞吐量
    对于量化权重，内存带宽是瓶颈，pipeline 尤为重要
    """
    BM = 128
    BN = 128
    BK = 64
    NUM_STAGES = 3  # 三级流水线
    
    with T.Block("root"):
        # 多级共享内存
        X_smem = T.alloc_shared([NUM_STAGES, BM, BK], "float16")
        W_smem = T.alloc_shared([NUM_STAGES, BN, BK // 2], "uint8")
        scale_smem = T.alloc_shared([NUM_STAGES, BN, BK // GROUP_SIZE], "float16")
        
        X_frag = T.alloc_fragment([BM, BK], "float16")
        W_frag = T.alloc_fragment([BN, BK], "float16")
        Y_frag = T.alloc_fragment([BM, BN], "float32")
        
        T.clear(Y_frag)
        
        num_k = T.ceildiv(K, BK)
        
        # Prologue: 填充前两个 stage
        for s in T.serial(min(NUM_STAGES - 1, num_k)):
            T.copy(X[by * BM : (by + 1) * BM, s * BK : (s + 1) * BK], X_smem[s])
            T.copy(W_q[bx * BN : (bx + 1) * BN, s * BK // 2 : (s + 1) * BK // 2], W_smem[s])
            T.copy(scales[bx * BN : (bx + 1) * BN, s * BK // GROUP_SIZE : (s + 1) * BK // GROUP_SIZE], scale_smem[s])
        T.sync()
        
        # Main loop
        for k in T.serial(num_k):
            stage = k % NUM_STAGES
            
            # Prefetch 下一个 stage
            if k + NUM_STAGES - 1 < num_k:
                next_k = k + NUM_STAGES - 1
                next_stage = next_k % NUM_STAGES
                T.copy(X[by * BM : (by + 1) * BM, next_k * BK : (next_k + 1) * BK], X_smem[next_stage])
                T.copy(W_q[bx * BN : (bx + 1) * BN, next_k * BK // 2 : (next_k + 1) * BK // 2], W_smem[next_stage])
                T.copy(scales[bx * BN : (bx + 1) * BN, next_k * BK // GROUP_SIZE : (next_k + 1) * BK // GROUP_SIZE], scale_smem[next_stage])
            
            # 计算当前 stage
            # Dequantize
            for i, j in T.Parallel(BN, BK):
                byte_idx = j // 2
                is_high = j % 2
                packed_byte = W_smem[stage, i, byte_idx]
                
                if is_high:
                    val_4bit = (T.cast(packed_byte, "int8") >> 4) & 0x0F
                else:
                    val_4bit = T.cast(packed_byte, "int8") & 0x0F
                
                val_signed = T.cast(val_4bit, "int8") - 8
                group_idx = j // GROUP_SIZE
                W_frag[i, j] = T.cast(val_signed, "float16") * scale_smem[stage, i, group_idx]
            
            T.copy(X_smem[stage], X_frag)
            T.sync()
            
            T.gemm(X_frag, W_frag, Y_frag, "float16", "float16", "float32")
        
        # 写回
        for i, j in T.Parallel(BM, BN):
            Y[by * BM + i, bx * BN + j] = T.cast(Y_frag[i, j], "float16")
```

上述代码实现了带三级流水线的Weight-only INT4 GEMM内核。NUM_STAGES=3表示使用三个共享内存缓冲区（triple buffering），比双缓冲更激进地隐藏内存延迟。Prologue阶段预填充前两个stage的数据，主循环中每次迭代计算当前stage的tile同时预取第k+NUM_STAGES-1个stage的数据。这种流水线设计对于量化GEMM特别重要，因为INT4的加载和解包需要额外的计算时间，没有流水线会导致Tensor Core等待数据而利用率不足。实践中三级流水线通常能将Tensor Core利用率提升到80%以上。

### 5.4 权重量化的内存布局优化

```
原始权重布局 (行主序):
W = [w0, w1, w2, w3, w4, w5, w6, w7, ...]
每个权重 16 bits (FP16)

INT4 打包布局:
W_packed = [(w1<<4|w0), (w3<<4|w2), (w5<<4|w4), (w7<<4|w6), ...]
每字节存储 2 个权重

内存访问对齐要求:
┌─────────────────────────────────────────────┐
│ NVIDIA GPU 内存事务大小: 32 bytes (256 bits) │
│                                             │
│ INT4 打包后: 32 bytes = 64 个权重            │
│ INT8:        32 bytes = 32 个权重            │
│ FP16:        32 bytes = 16 个权重            │
│                                             │
│ 建议: 分块大小应为 64 的倍数以对齐内存事务    │
└─────────────────────────────────────────────┘
```

上述图表详细分析了不同精度下的内存对齐特性。NVIDIA GPU的内存事务大小为32字节（256位），优化内存访问的要点是确保数据访问对齐到这个边界。INT4打包后每32字节包含64个权重值，是FP16的4倍；INT8为32个权重；FP16仅为16个权重。这种对齐特性直接影响全局内存和共享内存的访问效率：如果分块大小不是64的倍数，会导致内存事务的低效使用（部分数据被加载但不使用）。在实际优化中，建议所有分块维度（尤其是K维度）设置为64的倍数，以确保每次内存事务都能满载。这对于量化GEMM的带宽利用率至关重要，不对齐的访问可能损失30%以上的有效带宽。

---

## 6. 大模型推理中的应用

### 6.1 LLaMA 量化推理

```python
class QuantizedLinear:
    """
    量化线性层，用于大模型推理
    
    支持 INT4/NF4 权重 + FP16 激活的混合精度计算
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        group_size: int = 128,
        quant_method: str = "nf4",
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.quant_method = quant_method
        
        # 量化后的权重 (打包格式)
        packed_size = in_features * out_features * bits // 8
        self.weight_packed = torch.zeros(packed_size, dtype=torch.uint8)
        
        # 量化参数
        num_groups = (in_features // group_size) * out_features
        self.scales = torch.zeros(num_groups, dtype=torch.float16)
        self.zeros = torch.zeros(num_groups, dtype=torch.float16)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入激活 [batch, seq_len, in_features], FP16
        
        Returns:
            y: 输出 [batch, seq_len, out_features], FP16
        """
        # 使用 TileLang 编译的量化 GEMM 内核
        return int4_dequant_gemm_kernel(
            x,
            self.weight_packed,
            self.scales,
            self.zeros,
        )
```

上述代码实现了量化线性层（QuantizedLinear），是大模型推理中权重量化方案的Python封装。该类维护打包的量化权重（uint8格式存储）和per-group量化参数（scales和zeros），前向传播通过调用TileLang编译的INT4反量化GEMM内核完成混合精度计算。构造函数根据in_features、out_features和bits计算打包后的存储大小，scales和zeros按group_size分配。这种设计使得量化模型可以无缝替换原始FP16线性层，用户只需将模型中的nn.Linear替换为QuantizedLinear即可获得约75%的权重内存节省，同时保持接近FP16的推理精度。

### 6.2 LLaMA 模型量化示例

```python
import torch
from transformers import LlamaForCausalLM, LlamaConfig

def quantize_llama_model(
    model_path: str,
    output_path: str,
    bits: int = 4,
    group_size: int = 128,
    method: str = "gptq",  # "gptq" | "awq" | "nf4"
):
    """
    量化 LLaMA 模型
    
    流程:
    1. 加载原始 FP16 模型
    2. 对每个线性层进行量化
    3. 保存量化后的权重和元数据
    """
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # 量化配置
    quant_config = {
        "bits": bits,
        "group_size": group_size,
        "method": method,
        "quantized_layers": [],
    }
    
    # 收集校准数据（用于 GPTQ/AWQ）
    calibration_data = load_calibration_dataset(n_samples=128)
    
    # 逐层量化
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(f"Quantizing layer: {name}")
            
            W = module.weight.data  # [out_features, in_features]
            
            if method == "gptq":
                # 收集 Hessian
                H = compute_hessian(module, calibration_data)
                W_q, scales, zeros = GPTQQuantizer(bits, group_size).quantize(W, H)
            elif method == "awq":
                X = collect_activations(module, calibration_data)
                W_q, scales, zeros = AWQQuantizer(bits, group_size).quantize(W, X)
            elif method == "nf4":
                W_q, scales = nf4_quantize(W, block_size=group_size)
            
            # 替换为量化版本
            replace_with_quantized_linear(module, W_q, scales, zeros)
            quant_config["quantized_layers"].append(name)
    
    # 保存
    save_quantized_model(model, output_path, quant_config)
```

上述代码展示了完整的LLaMA模型量化流程。该函数首先加载FP16模型，然后逐层遍历所有nn.Linear模块进行量化处理。根据method参数选择不同的量化算法：gptq使用Hessian矩阵进行误差补偿量化，awq通过激活统计自动搜索最优缩放参数，nf4直接使用正态分布量化表。量化完成后调用replace_with_quantized_linear将原始线性层替换为量化版本。这个流程完整展示了从浮点模型到4-bit量化模型的转换过程，是实际部署大模型推理服务前的关键预处理步骤。校准数据通常需要128-512个样本来估算Hessian或激活分布。

### 6.3 KV Cache 量化

在大模型推理中，KV Cache 占用大量显存。量化 KV Cache 可以支持更长的上下文：

```python
class QuantizedKVCache:
    """
    量化 KV Cache
    
    将 Key 和 Value 缓存从 FP16 量化为 INT8/FP8，
    可以将 KV Cache 内存占用减少 50%
    """
    
    def __init__(
        self,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: str = "int8",  # "int8" | "fp8_e4m3"
    ):
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        
        # 量化后的 KV Cache
        if dtype == "int8":
            cache_dtype = torch.int8
        elif dtype == "fp8_e4m3":
            cache_dtype = torch.float8_e4m3fn
        
        self.key_cache = torch.zeros(
            (max_seq_len, num_heads, head_dim),
            dtype=cache_dtype,
            device="cuda",
        )
        self.value_cache = torch.zeros(
            (max_seq_len, num_heads, head_dim),
            dtype=cache_dtype,
            device="cuda",
        )
        
        # 缩放因子 (per-head)
        self.key_scales = torch.ones(
            (max_seq_len, num_heads), dtype=torch.float32, device="cuda"
        )
        self.value_scales = torch.ones(
            (max_seq_len, num_heads), dtype=torch.float32, device="cuda"
        )
        
        self.current_len = 0
    
    def append(self, key: torch.Tensor, value: torch.Tensor):
        """
        追加新的 KV 并量化
        
        Args:
            key: [batch, num_heads, 1, head_dim] FP16
            value: [batch, num_heads, 1, head_dim] FP16
        """
        seq_len = self.current_len
        
        if self.dtype == "int8":
            # Per-head INT8 量化
            k_abs_max = key.abs().amax(dim=-1, keepdim=True)  # [batch, heads, 1, 1]
            v_abs_max = value.abs().amax(dim=-1, keepdim=True)
            
            k_scale = k_abs_max / 127.0
            v_scale = v_abs_max / 127.0
            
            k_q = torch.clamp(torch.round(key / k_scale), -127, 127).to(torch.int8)
            v_q = torch.clamp(torch.round(value / v_scale), -127, 127).to(torch.int8)
            
            # 存储量化值和 scale
            self.key_cache[seq_len] = k_q.squeeze(0).squeeze(1)  # [heads, dim]
            self.value_cache[seq_len] = v_q.squeeze(0).squeeze(1)
            self.key_scales[seq_len] = k_scale.squeeze(0).squeeze(-1).squeeze(-1)
            self.value_scales[seq_len] = v_scale.squeeze(0).squeeze(-1).squeeze(-1)
        
        elif self.dtype == "fp8_e4m3":
            k_scale = key.abs().amax(dim=-1, keepdim=True) / 448.0
            v_scale = value.abs().amax(dim=-1, keepdim=True) / 448.0
            
            self.key_cache[seq_len] = (key / k_scale).to(torch.float8_e4m3fn).squeeze(0).squeeze(1)
            self.value_cache[seq_len] = (value / v_scale).to(torch.float8_e4m3fn).squeeze(0).squeeze(1)
            self.key_scales[seq_len] = k_scale.squeeze(0).squeeze(-1).squeeze(-1)
            self.value_scales[seq_len] = v_scale.squeeze(0).squeeze(-1).squeeze(-1)
        
        self.current_len += 1
    
    def get(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        获取反量化后的 KV Cache
        
        Returns:
            key: [batch, num_heads, seq_len, head_dim] FP16
            value: [batch, num_heads, seq_len, head_dim] FP16
        """
        k_q = self.key_cache[:seq_len]  # [seq, heads, dim]
        v_q = self.value_cache[:seq_len]
        k_s = self.key_scales[:seq_len]  # [seq, heads]
        v_s = self.value_scales[:seq_len]
        
        if self.dtype == "int8":
            k = k_q.float() * k_s.unsqueeze(-1)
            v = v_q.float() * v_s.unsqueeze(-1)
        elif self.dtype == "fp8_e4m3":
            k = k_q.float() * k_s.unsqueeze(-1)
            v = v_q.float() * v_s.unsqueeze(-1)
        
        return k.half().unsqueeze(0), v.half().unsqueeze(0)
```

上述代码实现了量化KV Cache管理器。在大模型推理中，每层每个token的Key和Value都需要缓存，量化KV Cache可将缓存占用减半。该实现支持INT8和FP8两种格式：INT8量化将KV值映射到[-127,127]范围，使用per-head的scale因子；FP8 E4M3量化利用有限的±448动态范围。append方法每新增一个token时将FP16的KV输入量化并存储，同时保存scale因子。get方法从缓存中读取量化数据并反量化恢复为FP16。这种per-head量化粒度的设计平衡了精度和实现复杂度，在128K长上下文场景下可将KV Cache从数十GB压缩至数GB。

### 6.4 Mistral 滑动窗口注意力的量化优化

```python
class QuantizedSlidingWindowKVCache(QuantizedKVCache):
    """
    Mistral 风格的滑动窗口量化 KV Cache
    
    只保留最近 window_size 个 token 的 KV，
    更早的 KV 被丢弃以节省显存
    """
    
    def __init__(self, window_size: int = 4096, **kwargs):
        super().__init__(max_seq_len=window_size, **kwargs)
        self.window_size = window_size
    
    def append(self, key: torch.Tensor, value: torch.Tensor):
        """追加新 KV，超出窗口时丢弃最旧的"""
        if self.current_len >= self.window_size:
            # 循环缓冲区
            self.current_len = self.current_len % self.window_size
        
        super().append(key, value)
```

上述代码继承自QuantizedKVCache，增加了滑动窗口机制。Mistral等模型使用滑动窗口注意力来降低长序列的计算复杂度，只保留最近window_size个token的KV缓存。append方法通过循环缓冲区实现：当缓存满后，新数据覆盖最旧的位置（使用模运算计算写入位置）。这种设计不仅结合了量化压缩的优势，还通过滑动窗口进一步限制了KV Cache的增长上限，使得128K甚至更长序列的推理成为可能。与固定容量预分配不同，循环缓冲区避免了数据搬移开销。

### 6.5 量化的 Attention 计算

```python
@T.prim_func
def quantized_attention(
    Q: T.Buffer[(B, H, S, D), "float16"],
    K_q: T.Buffer[(S, H, D), "int8"],
    V_q: T.Buffer[(S, H, D), "int8"],
    K_scales: T.Buffer[(S, H), "float32"],
    V_scales: T.Buffer[(S, H), "float32"],
    O: T.Buffer[(B, H, S, D), "float16"],
):
    """
    量化的 Attention 计算
    
    Q 保持 FP16，K/V 使用 INT8 量化
    计算: O = softmax(Q @ K^T / sqrt(d)) @ V
    
    优势: KV Cache 从 FP16 量化为 INT8，显存减半
    """
    # TileLang attention 实现
    # 1. 加载 INT8 K/V 到 SRAM
    # 2. 反量化为 FP16
    # 3. 计算 Q @ K^T
    # 4. Softmax
    # 5. 计算 @ V
    # 6. 输出
    pass  # 详细实现见注意力章节
```

这段代码是 6.5 量化的 Attention 计算 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 7. 性能对比

### 7.1 cuBLAS INT8 vs TileLang INT8

| 指标 | cuBLAS INT8 | TileLang INT8 | 差异 |
|------|-------------|---------------|------|
| 矩阵大小 M=N=K=4096 | 1.2 ms | 1.0 ms | TileLang 快 17% |
| 矩阵大小 M=N=K=8192 | 8.5 ms | 7.2 ms | TileLang 快 15% |
| 矩阵大小 M=1,N=K=4096 | 0.05 ms | 0.04 ms | TileLang 快 20% |
| 内存带宽利用率 | 85% | 92% | TileLang 高 7% |
| Tensor Core 利用率 | 78% | 85% | TileLang 高 7% |
| L2 Cache 命中率 | 72% | 78% | TileLang 高 6% |

> [!TIP]
> TileLang 的优势在于更精细的内存访问控制和更好的 pipeline 利用。对于自定义融合算子，优势更加明显。

### 7.2 bitsandbytes vs TileLang INT4

| 指标 | bitsandbytes INT4 | TileLang INT4 | 差异 |
|------|-------------------|---------------|------|
| 量化精度 (PPL) | 4.52 | 4.38 | TileLang 更优 |
| 推理速度 (tok/s) | 45 | 62 | TileLang 快 38% |
| 内存占用 (7B) | 3.8 GB | 3.6 GB | TileLang 更少 |
| 量化时间 | 2 min | 5 min | bitsandbytes 更快 |
| 支持的粒度 | per-tensor | per-group | TileLang 更灵活 |

### 7.3 不同精度的性能对比

```
性能 vs 精度权衡曲线 (LLaMA-7B, WikiText PPL):

精度 (bits) │ PPL    │ 内存 (GB) │ 速度 (tok/s) │ 相对吞吐
────────────┼────────┼───────────┼──────────────┼─────────
FP16 (16)   │ 5.68   │ 14.0      │ 30           │ 1.00×
FP8  (8)    │ 5.70   │ 7.5       │ 55           │ 1.83×
INT8 (8)    │ 5.72   │ 7.5       │ 50           │ 1.67×
INT4 (4)    │ 6.15   │ 4.0       │ 70           │ 2.33×
NF4  (4)    │ 5.98   │ 4.0       │ 65           │ 2.17×
INT3 (3)    │ 8.20   │ 3.2       │ 80           │ 2.67×
INT2 (2)    │ 15.50  │ 2.5       │ 90           │ 3.00×

注: 测试环境 H100 SXM, batch_size=1, seq_len=2048
```

这个代码块或示意图用于说明 7.3 不同精度的性能对比 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 7.4 内存带宽对比

| 精度 | 每元素字节 | 7B 模型权重大小 | 带宽需求 (2048 tokens) |
|------|-----------|----------------|----------------------|
| FP32 | 4 | 28 GB | 高 |
| FP16 | 2 | 14 GB | 中 |
| INT8 | 1 | 7 GB | 低 |
| INT4 | 0.5 | 3.5 GB | 极低 |
| NF4 | 0.5 | 3.5 GB | 极低 |

### 7.5 不同 GPU 的量化 GEMM 性能

| GPU | FP16 TFLOPS | INT8 TFLOPS | FP8 TFLOPS | INT4 解码速度 |
|-----|-------------|-------------|------------|--------------|
| A100 80GB | 312 | 624 | N/A | 40 tok/s |
| H100 SXM | 989 | 1979 | 1979 | 80 tok/s |
| H200 | 989 | 1979 | 1979 | 90 tok/s |
| RTX 4090 | 330 | 660 | N/A | 35 tok/s |
| RTX 3090 | 71 | 285 | N/A | 20 tok/s |

---

## 8. 常见问题排查

### 8.1 数值溢出 (Numerical Overflow)

```python
# 问题: INT8 反量化时出现数值溢出
# 症状: 输出中出现 inf 或 nan
# 原因: scale 计算不当或 INT32 累加器溢出

# 诊断代码
def diagnose_overflow(A_int8, B_int8, scale_A, scale_B):
    """诊断数值溢出问题"""
    
    # 检查 1: scale 是否合理
    print(f"scale_A range: [{scale_A.min():.6f}, {scale_A.max():.6f}]")
    print(f"scale_B range: [{scale_B.min():.6f}, {scale_B.max():.6f}]")
    
    if scale_A.min() < 1e-6 or scale_B.min() < 1e-6:
        print("WARNING: Scale 值过小，可能导致溢出")
    
    # 检查 2: INT32 累加是否会溢出
    K = A_int8.shape[1]
    max_product = 127 * 127  # INT8 最大乘积
    max_sum = max_product * K
    
    if max_sum > 2**31 - 1:
        print(f"WARNING: K={K} 时 INT32 可能溢出 (max_sum={max_sum})")
        print("建议: 减小 K 分块大小或使用 INT64 累加器")
    
    # 检查 3: 实际输出范围
    C = dequantized_gemm(A_int8, B_int8, scale_A, scale_B)
    print(f"Output range: [{C.min():.6f}, {C.max():.6f}]")
    print(f"Inf count: {torch.isinf(C).sum()}")
    print(f"NaN count: {torch.isnan(C).sum()}")


# 解决方案
"""
1. 使用 FP32 累加器代替 INT32
2. 将 K 分块大小减小（如从 256 减到 64）
3. 在累加过程中定期进行类型转换
4. 使用更大的数据类型（FP32）进行中间计算
"""
```

这段代码是 8.1 数值溢出 (Numerical Overflow) 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 8.2 Dequantize 开销分析

```python
# 问题: 反量化操作成为性能瓶颈
# 症状: 量化的 GEMM 比纯 FP16 GEMM 还慢

def profile_dequantize_overhead():
    """
    分析反量化操作的开销占比
    
    优化前:
    ┌─────────────────────────────────────────┐
    │ Load INT8  (10%) │ Dequantize (40%) │ GEMM (50%) │
    └─────────────────────────────────────────┘
    反量化占 40%！抵消了 INT8 带来的带宽优势
    
    优化后（融合反量化）:
    ┌─────────────────────────────────────────────────┐
    │ Load INT8 (10%) │ Dequant+GEMM fused (90%)      │
    └─────────────────────────────────────────────────┘
    """
    pass

# 解决方案
"""
1. 融合反量化到 GEMM 计算中（在 registers 中完成）
2. 使用 Tensor Core 原生 INT8 支持（避免反量化）
3. 减少不必要的中间存储
4. 使用更高效的位操作进行解包
"""
```

这段代码是 8.2 Dequantize 开销分析 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 8.3 内存对齐问题

```python
# 问题: 非对齐内存访问导致性能下降或错误
# 症状: CUDA illegal memory access 或性能异常低

def check_memory_alignment(tensor: torch.Tensor, alignment: int = 16):
    """检查张量内存对齐"""
    ptr = tensor.data_ptr()
    if ptr % alignment != 0:
        print(f"WARNING: Tensor at {hex(ptr)} is not {alignment}-byte aligned")
        print(f"  Misalignment: {ptr % alignment} bytes")
        return False
    return True


# INT4 打包的对齐要求
"""
INT4 打包数据的对齐注意事项:

1. 每个字节包含 2 个 INT4 值
2. CUDA 内存事务大小: 32 bytes (256 bits)
3. 建议: 权重矩阵的 K 维度应为 64 的倍数
   - 64 个 INT4 = 32 bytes = 1 个内存事务

4. 对于 group_size = 128:
   - 128 个 INT4 = 64 bytes = 2 个内存事务
   - 确保 group_size 是 64 的倍数

5. 共享内存 bank 冲突:
   - INT4 解包时的位操作可能导致 bank 冲突
   - 建议: 使用 padding 避免冲突
"""


# 解决方案代码
@T.prim_func
def aligned_int4_gemm(
    A: T.Buffer[(M, K), "float16"],
    B: T.Buffer[(N, K + PADDING), "uint8"],  # 添加 padding
    C: T.Buffer[(M, N), "float16"],
):
    """
    带内存对齐的 INT4 GEMM
    在 K 维度添加 padding 以确保对齐
    """
    # 使用实际 K 值进行计算，忽略 padding
    pass
```

这段代码是 8.3 内存对齐问题 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 8.4 量化校准失败

```python
# 问题: 量化后模型精度大幅下降
# 症状: 困惑度 (PPL) 显著增加

def diagnose_quantization_quality(original_model, quantized_model, calibration_data):
    """诊断量化质量问题"""
    
    # 1. 逐层分析量化误差
    for name, orig_module in original_model.named_modules():
        if isinstance(orig_module, torch.nn.Linear):
            quant_module = get_corresponding_module(quantized_model, name)
            
            W_orig = orig_module.weight.data
            W_deq = dequantize_weight(quant_module)
            
            mse = (W_orig - W_deq).pow(2).mean()
            cosine_sim = torch.nn.functional.cosine_similarity(
                W_orig.flatten().unsqueeze(0),
                W_deq.flatten().unsqueeze(0)
            ).item()
            
            if mse > 0.01 or cosine_sim < 0.95:
                print(f"WARNING: Layer {name} has high quantization error")
                print(f"  MSE: {mse:.6f}, Cosine Similarity: {cosine_sim:.4f}")
    
    # 2. 检查激活值分布
    # 3. 比较输出差异
    pass


# 解决方案
"""
1. 使用更多校准数据 (128 → 512 样本)
2. 使用更好的量化算法 (GPTQ/AWQ)
3. 对敏感层保持高精度 (混合精度量化)
4. 调整 group_size (128 → 64 或 32)
5. 使用平滑量化 (smooth quant)
"""
```

这段代码是 8.4 量化校准失败 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 8.5 TileLang 编译错误

```python
# 问题: TileLang 量化内核编译失败
# 常见错误及解决

"""
错误 1: "Unsupported dtype: uint4"
原因: TileLang 不原生支持 4-bit 类型
解决: 使用 uint8 存储打包的 4-bit 数据，在内核中手动解包

错误 2: "Shared memory allocation exceeded"
原因: 量化内核需要额外的共享内存存储 scale/zero_point
解决: 减小分块大小或使用更高效的存储方式

错误 3: "Bank conflict in shared memory access"
原因: INT4 解包时的位操作导致 bank 冲突
解决: 调整共享内存布局或添加 padding

错误 4: "Register pressure too high"
原因: 同时存储 INT8/FP16 数据导致寄存器不足
解决: 减少 fragment 大小或使用更多的 shared memory
"""
```

这段代码是 8.5 TileLang 编译错误 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 8.6 推理性能调优检查清单

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 内存对齐 | ✓/✗ | K 维度是否为 64 的倍数 |
| Pipeline 级数 | ✓/✗ | 是否使用 double/triple buffering |
| 分块大小 | ✓/✗ | BM/BN/BK 是否针对目标 GPU 优化 |
| 共享内存使用 | ✓/✗ | 是否超过 SM 的共享内存限制 |
| 寄存器压力 | ✓/✗ | 是否导致寄存器溢出到本地内存 |
| 量化粒度 | ✓/✗ | per-group 是否优于 per-tensor |
| 融合操作 | ✓/✗ | 是否将反量化融合到 GEMM 中 |
| 数据布局 | ✓/✗ | 是否使用了最优的数据布局 |

---

## 9. 练习题与思考题

### 练习 1: 实现对称量化

实现一个高效的对称量化函数，支持任意维度的张量输入。

```python
def symmetric_quantize_exercise(
    x: torch.Tensor,
    bits: int = 8,
    axis: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    练习: 实现沿指定轴的对称量化
    
    要求:
    1. 沿 axis 计算每个切片的 scale
    2. 量化并 clamp 到有效范围
    3. 返回量化值和 scale
    
    Args:
        x: 输入张量
        bits: 量化比特数
        axis: 量化轴
    
    Returns:
        x_q: 量化后的整数张量
        scales: 每个切片的 scale
    """
    # TODO: 实现你的代码
    pass
```

这段代码是 练习 1: 实现对称量化 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 练习 2: TileLang INT8 GEMM

使用 TileLang 实现一个完整的 INT8 GEMM 内核，要求：
- 支持 per-channel 量化
- 使用双缓冲 pipeline
- 融合反量化操作

```python
@T.prim_func
def int8_gemm_exercise(
    A: T.Buffer[(M, K), "int8"],
    B: T.Buffer[(N, K), "int8"],
    C: T.Buffer[(M, N), "float16"],
    scale_A: T.Buffer[(M,), "float32"],
    scale_B: T.Buffer[(N,), "float32"],
):
    """
    练习: 实现完整的 INT8 GEMM
    
    要求:
    1. 使用合适的分块策略 (BM, BN, BK)
    2. 实现双缓冲 pipeline
    3. 在寄存器中完成反量化
    4. 使用 FP16 Tensor Core 进行计算
    
    提示:
    - INT8 Tensor Core 的 K 维度通常为 32 或 64
    - 确保共享内存不超出限制
    - 注意内存对齐
    """
    # TODO: 实现你的代码
    pass
```

这段代码是 练习 2: TileLang INT8 GEMM 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 练习 3: NF4 查找表实现

实现 NF4 量化的核心查找表和量化/反量化函数。

```python
def nf4_quantize_exercise(x: torch.Tensor, block_size: int = 64):
    """
    练习: 实现 NF4 量化
    
    要求:
    1. 创建 NF4 量化级别表（基于正态分布分位数）
    2. 实现 per-block 量化
    3. 打包两个 4-bit 索引到一个字节
    4. 验证反量化结果的正确性
    
    验证方法:
    x_recon = nf4_dequantize(nf4_quantize(x))
    assert torch.allclose(x, x_recon, atol=0.01)
    """
    # TODO: 实现你的代码
    pass
```

这段代码是 练习 3: NF4 查找表实现 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 练习 4: GPTQ 误差补偿

实现 GPTQ 算法的核心误差补偿步骤。

```python
def gptq_error_compensation_exercise(
    W: torch.Tensor,
    H: torch.Tensor,
    col_idx: int,
    scale: float,
):
    """
    练习: 实现 GPTQ 的误差补偿
    
    要求:
    1. 量化第 col_idx 列
    2. 计算量化误差
    3. 使用 Hessian 信息将误差分配到后续列
    
    公式:
    error = W[:, col_idx] - quantize(W[:, col_idx])
    W[:, col_idx+1:] -= error * H[col_idx, col_idx+1:] / H[col_idx, col_idx]
    """
    # TODO: 实现你的代码
    pass
```

上述练习要求实现GPTQ算法的核心误差补偿步骤：先量化指定列，计算量化误差，再利用Hessian矩阵的逆将误差按比例分配到后续未量化的列上。这是GPTQ区别于简单量化的关键所在，通过二阶信息的误差补偿可以显著减少量化精度损失。理解这一机制对掌握GPTQ及其变体算法至关重要。

### 练习 5: 量化 KV Cache

实现一个支持 INT8 和 FP8 的量化 KV Cache。

```python
class QuantizedKVCacheExercise:
    """
    练习: 实现量化 KV Cache
    
    要求:
    1. 支持 INT8 和 FP8 两种量化格式
    2. 实现 append 和 get 操作
    3. 使用 per-head 量化粒度
    4. 支持滑动窗口模式
    """
    
    def __init__(self, max_len, num_heads, head_dim, dtype="int8"):
        # TODO: 初始化
        pass
    
    def append(self, key, value):
        """追加量化后的 KV"""
        # TODO: 实现
        pass
    
    def get(self, seq_len):
        """获取反量化后的 KV"""
        # TODO: 实现
        pass
```

上述代码框架提供了一个量化KV Cache的练习模板，要求支持INT8和FP8两种量化格式、实现append和get操作、使用per-head量化粒度并支持滑动窗口模式。该练习综合了量化存储、数据类型转换和缓存管理的核心技能，完成它有助于将前文介绍的量化理论应用到实际推理场景中。

### 思考题 1: 量化精度与计算效率的权衡

> 考虑以下场景：你需要在 H100 GPU 上部署一个 70B 参数的大模型。模型需要支持 4096 token 的上下文长度，且推理延迟要求 < 100ms/token。
>
> 问题：
> 1. 你会选择哪种量化方案（INT8/FP8/INT4/NF4）？为什么？
> 2. 如何分配量化精度给不同的层（注意力层 vs FFN 层）？
> 3. KV Cache 应该使用什么精度？

### 思考题 2: 异常值处理

> 在某些 Transformer 模型中，少数激活通道的值远大于其他通道（称为"异常值"或"outliers"）。
>
> 问题：
> 1. 异常值如何影响标准量化？
> 2. SmoothQuant 如何解决这个问题？
> 3. 在 TileLang 中如何高效实现 SmoothQuant 的预处理？

### 思考题 3: 混合精度策略设计

> 对于一个 13B 参数的 LLaMA 模型，设计一个混合精度量化策略：
>
> 1. 哪些层应该保持 FP16 精度？
> 2. 哪些层可以使用 INT4 量化？
> 3. 如何自动确定每层的最佳精度？

### 思考题 4: 量化感知训练 vs 训练后量化

> 比较量化感知训练（QAT）和训练后量化（PTQ）的优缺点：
>
> 1. 在什么场景下应该选择 QAT？
> 2. QAT 中的直通估计器（STE）是什么？为什么需要它？
> 3. 对于大模型，为什么 PTQ 更常用？

### 思考题 5: 量化内核的通用化设计

> 设计一个通用的量化 GEMM 内核框架，要求：
>
> 1. 支持任意量化格式（INT8/FP8/INT4/NF4）
> 2. 支持任意量化粒度（per-tensor/per-channel/per-group）
> 3. 自动选择最优的计算策略
>
> 如何设计接口和配置？

---

## 10. 扩展阅读

### 10.1 推荐论文

| 论文 | 年份 | 主题 | 重要性 |
|------|------|------|--------|
| GPTQ | 2023 | 基于二阶信息的权重量化 | ★★★★★ |
| AWQ | 2023 | 激活感知的权重量化 | ★★★★★ |
| SmoothQuant | 2022 | 平滑量化，处理异常值 | ★★★★☆ |
| QLoRA | 2023 | NF4 量化 + LoRA 微调 | ★★★★☆ |
| FP8 Formats for Deep Learning | 2022 | FP8 格式规范 | ★★★★☆ |
| LLM.int8() | 2022 | 大模型 INT8 推理 | ★★★☆☆ |
| ZeroQuant | 2022 | 高效量化方案 | ★★★☆☆ |

### 10.2 工具和库

| 工具 | 用途 | 链接 |
|------|------|------|
| bitsandbytes | INT8/INT4 量化 | github.com/TimDettmers/bitsandbytes |
| AutoGPTQ | GPTQ 量化 | github.com/AutoGPTQ/AutoGPTQ |
| AutoAWQ | AWQ 量化 | github.com/casper-hansen/AutoAWQ |
| llama.cpp | GGUF 推理 | github.com/ggerganov/llama.cpp |
| vLLM | 高效推理引擎 | github.com/vllm-project/vllm |
| TensorRT-LLM | NVIDIA 推理优化 | github.com/NVIDIA/TensorRT-LLM |

### 10.3 进阶主题

- **混合精度训练**：FP8 训练中的缩放策略和损失缩放
- **量化感知微调**：QAT + LoRA 的结合
- **动态量化**：根据输入动态调整量化参数
- **稀疏量化**：结合稀疏性和量化
- **硬件协同设计**：为量化设计专用硬件

---

## Summary

> 本章全面介绍了量化 GEMM 的理论和实践，从基础的量化数学到 TileLang 内核实现。

✅ **掌握的核心概念**：
- 对称/非对称量化的数学原理和实现
- per-tensor/per-channel/per-group 三种量化粒度
- INT8 GEMM 的 dequantize-then-multiply 和 multiply-then-dequantize 策略
- FP8 E4M3/E5M2 格式及 Hopper Tensor Core 使用
- INT4/NF4 量化及 GPTQ/AWQ 算法

✅ **掌握的 TileLang 技能**：
- 使用 `@T.prim_func` 定义量化 GEMM 内核
- 使用 `T.alloc_fragment` 管理不同精度的寄存器数据
- 实现融合反量化的 GEMM 内核
- 双缓冲 pipeline 优化

✅ **掌握的工程技能**：
- 大模型量化部署（LLaMA/Mistral）
- KV Cache 量化优化
- 性能分析和问题排查

🎯 **下一步学习**：
- Chapter 9: Flash Attention 与高效注意力机制
- Chapter 10: 混合专家模型 (MoE) 的 TileLang 实现
- Chapter 11: 分布式推理与张量并行

---

> **下一章预告**：Chapter 9 将深入讲解 Flash Attention 的原理和 TileLang 实现，包括在线 softmax、分块计算、内存优化等关键技术。量化与注意力的结合将为你打开大模型优化的新视野。

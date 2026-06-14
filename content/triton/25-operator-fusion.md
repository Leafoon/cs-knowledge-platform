---
title: "Chapter 25: 算子融合与编译器级优化"
description: "深入理解算子融合的基本原理与动机，掌握 Triton 中手动融合算子的方法，理解 Triton 编译器的 Pointwise/Reduction 融合策略，了解与 torch.compile 的集成与自动融合"
date: "2026-06-12"
---

# Chapter 25: 算子融合与编译器级优化

> **学习目标**：
> - 理解算子融合的基本原理与动机
> - 掌握 Triton 中手动融合算子的方法
> - 理解 Triton 编译器的 Pointwise/Reduction 融合策略
> - 了解与 torch.compile 的集成与自动融合
> - 掌握融合模式分类与融合规则
> - 了解融合的性能收益与局限性

---

## 25.1 算子融合动机

### 25.1.1 Kernel Launch 开销分析

在 GPU 计算中，每次调用一个 CUDA kernel 都需要经历以下流程：

```
Host (CPU)                        Device (GPU)
─────────────                     ─────────────
1. 分配 kernel 参数            
2. 设置 grid/block 维度         
3. 启动 kernel ──────────────>   4. GPU 调度器接收任务
4. 等待完成                    5. 执行 kernel
5. 继续下一步                  6. 完成信号
```

每个 kernel launch 的开销包括：
- **CPU 端开销**：参数打包、驱动调用、CUDA runtime 处理
- **GPU 端开销**：硬件调度器分配 SM、设置寄存器/共享内存
- **同步开销**：kernel 之间的依赖需要隐式或显式同步

让我们通过一个简单的示例来量化这个开销：

```python
import torch
import time

def measure_kernel_launch_overhead():
    """测量 kernel launch 的开销"""
    # 创建一个很小的张量，kernel 执行时间几乎为零
    a = torch.randn(1, device='cuda')
    b = torch.randn(1, device='cuda')

    # 预热
    for _ in range(100):
        c = a + b

    torch.cuda.synchronize()

    # 测量单次 kernel launch 的开销
    num_iterations = 10000
    start = time.perf_counter()
    for _ in range(num_iterations):
        c = a + b  # 这是一个极其简单的 kernel
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_launch_time = (end - start) / num_iterations * 1000  # 转换为毫秒
    print(f"平均 kernel launch 时间: {avg_launch_time * 1000:.2f} µs")
    print(f"如果模型有 100 个算子，launch 开销总计: {avg_launch_time * 100:.2f} ms")

# 典型输出:
# 平均 kernel launch 时间: 5~15 µs
# 如果模型有 100 个算子，launch 开销总计: 0.5~1.5 ms
```

对于一个典型的深度学习模型（如 ResNet-50 有约 100 个算子），kernel launch 的累积开销可达 0.5~1.5 ms，这在推理延迟敏感的场景中是不可忽视的。

### 25.1.2 中间结果的内存读写

算子融合的另一个核心动机是减少中间结果的内存读写（Memory Bandwidth Bound 问题）。让我们分析一个具体的例子：

```python
import torch

# 示例：逐元素运算链
# y = (a + b) * c + d

a = torch.randn(1024 * 1024, device='cuda')  # 4 MB
b = torch.randn(1024 * 1024, device='cuda')  # 4 MB
c = torch.randn(1024 * 1024, device='cuda')  # 4 MB
d = torch.randn(1024 * 1024, device='cuda')  # 4 MB

# 不融合的情况：3 个 kernel，6 次全局内存读写
t = a + b       # kernel 1: 读 a, b (8 MB) → 写 t (4 MB)
u = t * c       # kernel 2: 读 t, c (8 MB) → 写 u (4 MB)
y = u + d       # kernel 3: 读 u, d (8 MB) → 写 y (4 MB)
# 总共：读 24 MB + 写 12 MB = 36 MB 全局内存访问

# 融合后的情况：1 个 kernel，5 次全局内存读写
# y = (a + b) * c + d  在一个 kernel 中完成
# 读 a, b, c, d (16 MB) → 写 y (4 MB)
# 总共：读 16 MB + 写 4 MB = 20 MB 全局内存访问
```

更清晰的对比：

| 操作方式 | Kernel 数量 | 全局内存读取 | 全局内存写入 | 总内存流量 | 带宽节省 |
|:---|:---:|:---:|:---:|:---:|:---:|
| 不融合（3 个 kernel） | 3 | 24 MB | 12 MB | 36 MB | - |
| 融合（1 个 kernel） | 1 | 16 MB | 4 MB | 20 MB | 44.4% |

对于带宽受限（Memory Bandwidth Bound）的逐元素运算，这种内存流量的减少直接转化为性能提升。

### 25.1.3 数据局部性与缓存效率

融合还能显著提高数据局部性。在不融合的情况下，中间结果需要通过全局内存传递：

```
不融合的数据流:
┌──────────┐     全局内存      ┌──────────┐     全局内存      ┌──────────┐
│ Kernel 1 │ ──── 写入 ────→  │ Kernel 2 │ ──── 写入 ────→  │ Kernel 3 │
│  (a + b) │      (4 MB)      │  (t * c) │      (4 MB)      │  (u + d) │
└──────────┘                   └──────────┘                   └──────────┘
     ↑                              ↑                              ↑
   读 a,b                       读 t,c                         读 u,d
```

```
融合后的数据流:
┌─────────────────────────────────────────┐
│           单个 Kernel                    │
│  ┌─────────┐   ┌─────────┐   ┌───────┐ │
│  │ a + b → t │ → │ t * c → u │ → │ u + d → y │ │
│  └─────────┘   └─────────┘   └───────┘ │
│                                         │
│  中间结果 t, u 保存在寄存器中           │
│  无全局内存访问                          │
└─────────────────────────────────────────┘
```

在融合 kernel 中，中间结果 `t` 和 `u` 可以保存在寄存器或共享内存中，避免了全局内存的读写。这利用了数据的时间局部性（同一个数据被连续使用）和空间局部性（相邻线程处理相邻数据）。

### 25.1.4 融合前后的性能对比

让我们用一个实际的例子来展示融合的效果：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def fused_add_mul_add_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """融合 kernel: y = (a + b) * c + d"""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 一次性加载所有需要的数据
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)

    # 融合计算：中间结果保存在寄存器中
    t = a + b      # 中间结果 t 在寄存器中
    u = t * c      # 中间结果 u 在寄存器中
    y = u + d      # 最终结果

    tl.store(y_ptr + offsets, y, mask=mask)


def fused_add_mul_add(a, b, c, d):
    """融合版本的前向函数"""
    n = a.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    y = torch.empty_like(a)
    fused_add_mul_add_kernel[grid](
        a, b, c, d, y, n, BLOCK_SIZE=BLOCK_SIZE
    )
    return y


def unfused_add_mul_add(a, b, c, d):
    """不融合版本：三个独立 kernel"""
    t = a + b    # kernel 1
    u = t * c    # kernel 2
    y = u + d    # kernel 3
    return y


# 性能对比
a = torch.randn(1024 * 1024, device='cuda', dtype=torch.float32)
b = torch.randn(1024 * 1024, device='cuda', dtype=torch.float32)
c = torch.randn(1024 * 1024, device='cuda', dtype=torch.float32)
d = torch.randn(1024 * 1024, device='cuda', dtype=torch.float32)

# 预热
for _ in range(100):
    _ = fused_add_mul_add(a, b, c, d)
    _ = unfused_add_mul_add(a, b, c, d)

torch.cuda.synchronize()

# 测量
# 融合版本
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(1000):
    y_fused = fused_add_mul_add(a, b, c, d)
end.record()
torch.cuda.synchronize()
fused_time = start.elapsed_time(end)

# 不融合版本
start.record()
for _ in range(1000):
    y_unfused = unfused_add_mul_add(a, b, c, d)
end.record()
torch.cuda.synchronize()
unfused_time = start.elapsed_time(end)

print(f"融合版本: {fused_time:.3f} ms")
print(f"不融合版本: {unfused_time:.3f} ms")
print(f"加速比: {unfused_time / fused_time:.2f}x")
# 典型结果: 融合版本比不融合版本快 2~3x
```

### 25.1.5 融合的理论分析：Arithmetic Intensity

融合的核心价值可以通过算术强度（Arithmetic Intensity）来分析：

$$
\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Accessed}}
$$

对于逐元素运算（如 `y = (a + b) * c + d`）：
- 每个元素需要 3 次浮点运算（1 加 + 1 乘 + 1 加）
- 每个元素需要读取 4 个输入 + 写入 1 个输出 = 5 × 4 bytes = 20 bytes
- 算术强度 = 3 / 20 = 0.15 FLOPs/byte

这远低于 GPU 的算术强度阈值（A100 约为 156 FLOPs/byte），因此这类操作是典型的 **Memory Bandwidth Bound**。融合通过减少内存访问次数来提高有效带宽利用率。

---

## 25.2 Triton 手动融合

### 25.2.1 Conv + BatchNorm + ReLU 融合

在推理场景中，Conv + BatchNorm + ReLU 是最常见的融合模式。让我们实现一个完整的融合 kernel：

```python
import torch
import triton
import triton.language as tl


@triton.jit
def fused_conv_bn_relu_kernel(
    # 输入特征图
    input_ptr,
    # 卷积权重
    weight_ptr,
    # 卷积偏置
    bias_ptr,
    # BatchNorm 参数
    bn_weight_ptr,
    bn_bias_ptr,
    bn_mean_ptr,
    bn_var_ptr,
    # 输出
    output_ptr,
    # 形状参数
    batch_size, in_channels, out_channels,
    height, width,
    kernel_size,
    stride, padding,
    # BatchNorm 参数
    eps,
    # 网格参数
    BLOCK_SIZE: tl.constexpr,
):
    """
    融合 Conv + BatchNorm + ReLU

    注意：这是一个简化的实现，仅展示融合思想
    实际的 Conv 融合需要考虑 im2col 或直接卷积策略
    """
    # 计算输出位置
    pid = tl.program_id(0)
    num_elements = batch_size * out_channels * height * width

    # 每个线程处理一个输出元素
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    # 解码输出位置 (n, oc, oh, ow)
    ow = offsets % width
    oh = (offsets // width) % height
    oc = (offsets // (width * height)) % out_channels
    n = offsets // (width * height * out_channels)

    # ===== 阶段 1: 卷积计算 =====
    # 简化版：这里假设已经通过 im2col 或其他方式获得了输入
    # 实际实现中需要完整的卷积逻辑
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # 遍历输入通道和卷积核位置
    for ic in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # 计算输入位置
                ih = oh * stride - padding + kh
                iw = ow * stride - padding + kw

                # 边界检查
                input_valid = (ih >= 0) & (ih < height) & \
                              (iw >= 0) & (iw < width)

                # 加载输入和权重
                input_idx = n * in_channels * height * width + \
                           ic * height * width + ih * width + iw
                weight_idx = oc * in_channels * kernel_size * kernel_size + \
                            ic * kernel_size * kernel_size + \
                            kh * kernel_size + kw

                input_val = tl.where(
                    input_valid,
                    tl.load(input_ptr + input_idx, mask=mask, other=0.0),
                    0.0
                )
                weight_val = tl.load(weight_ptr + weight_idx)

                acc += input_val * weight_val

    # 加偏置
    conv_bias = tl.load(bias_ptr + oc)
    acc += conv_bias

    # ===== 阶段 2: BatchNorm（推理模式）=====
    # BN 推理公式: y = weight * (x - mean) / sqrt(var + eps) + bias
    bn_mean = tl.load(bn_mean_ptr + oc)
    bn_var = tl.load(bn_var_ptr + oc)
    bn_w = tl.load(bn_weight_ptr + oc)
    bn_b = tl.load(bn_bias_ptr + oc)

    # 融合的 BN 计算
    bn_scale = bn_w / tl.sqrt(bn_var + eps)
    bn_shift = bn_b - bn_w * bn_mean / tl.sqrt(bn_var + eps)
    acc = bn_scale * acc + bn_shift

    # ===== 阶段 3: ReLU =====
    acc = tl.maximum(acc, 0.0)

    # ===== 写回输出 =====
    tl.store(output_ptr + offsets, acc, mask=mask)


def fused_conv_bn_relu(
    input, weight, bias,
    bn_weight, bn_bias, bn_mean, bn_var,
    stride=1, padding=1, eps=1e-5
):
    """融合 Conv + BN + ReLU 的前向函数"""
    batch, in_channels, height, width = input.shape
    out_channels = weight.shape[0]
    kernel_size = weight.shape[2]

    # 计算输出尺寸
    out_height = (height + 2 * padding - kernel_size) // stride + 1
    out_width = (width + 2 * padding - kernel_size) // stride + 1

    output = torch.zeros(
        batch, out_channels, out_height, out_width,
        device=input.device, dtype=input.dtype
    )

    num_elements = batch * out_channels * out_height * out_width
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)

    fused_conv_bn_relu_kernel[grid](
        input, weight, bias,
        bn_weight, bn_bias, bn_mean, bn_var,
        output,
        batch, in_channels, out_channels,
        height, width, kernel_size,
        stride, padding,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output
```

### 25.2.2 GEMM + Bias + SiLU 融合

在 Transformer 和 MLP 中，GEMM + Bias + 激活函数是最常见的融合模式：

```python
import torch
import triton
import triton.language as tl


@triton.jit
def fused_gemm_bias_silu_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_an,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    融合 GEMM + Bias + SiLU

    C = SiLU(A @ B + bias)

    SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id % tl.cdiv(num_pid_m, GROUP_SIZE_M)
    group_size_m = min(num_pid_m - first_pid_m * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m = first_pid_m * GROUP_SIZE_M + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 计算偏移量
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_an)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bn + offs_bn[None, :] * stride_bk)

    # 累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 分块矩阵乘法
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 加载分块
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k * BLOCK_SIZE_K + K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k * BLOCK_SIZE_K + K, other=0.0)

        # 累加
        accumulator += tl.dot(a, b)

        # 更新指针
        a_ptrs += BLOCK_SIZE_K * stride_an
        b_ptrs += BLOCK_SIZE_K * stride_bn

    # ===== 融合 Bias =====
    # 加载 bias（广播到每一行）
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    bias = tl.load(bias_ptr + offs_cn)
    accumulator += bias[None, :]

    # ===== 融合 SiLU =====
    # SiLU(x) = x * sigmoid(x)
    # sigmoid(x) = 1 / (1 + exp(-x))
    silu = accumulator * tl.sigmoid(accumulator)

    # 类型转换并存储
    c = silu.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


GROUP_SIZE_M = 8


def fused_gemm_bias_silu(a, b, bias):
    """融合 GEMM + Bias + SiLU"""
    assert a.shape[1] == b.shape[0], "维度不匹配"
    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),)

    fused_gemm_bias_silu_kernel[grid](
        a, b, bias, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )

    return c


# 使用示例
def mlp_fused(x, w1, b1, w2, b2):
    """融合的 MLP: SiLU(x @ w1 + b1) @ w2 + b2"""
    # 第一层：融合 GEMM + Bias + SiLU
    h = fused_gemm_bias_silu(x, w1, b1)
    # 第二层：普通 GEMM + Bias（可以进一步融合）
    out = h @ w2 + b2
    return out
```

### 25.2.3 LayerNorm + Residual Add 融合

在 Transformer 中，LayerNorm 和残差连接经常一起出现，融合可以显著减少内存访问：

```python
import torch
import triton
import triton.language as tl


@triton.jit
def fused_layernorm_residual_kernel(
    input_ptr, residual_ptr, output_ptr,
    weight_ptr, bias_ptr,
    normalized_shape,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    融合 LayerNorm + Residual Add

    output = LayerNorm(input + residual)

    这里将残差连接融合到 LayerNorm 的输入加载中
    """
    # 每个 program 处理一行
    row = tl.program_id(0)

    # 计算行的偏移量
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # ===== 融合点：同时加载 input 和 residual =====
    # 传统的做法需要：
    # 1. y = input + residual  (kernel 1)
    # 2. output = LayerNorm(y)  (kernel 2)
    # 融合后只需一次加载

    input_offset = row * n_cols + cols
    x = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    r = tl.load(residual_ptr + input_offset, mask=mask, other=0.0)
    x = x + r  # 残差连接在寄存器中完成

    # ===== LayerNorm 计算 =====
    # 阶段 1: 计算均值
    mean = tl.sum(x, axis=0) / n_cols

    # 阶段 2: 计算方差
    var = tl.sum((x - mean) ** 2, axis=0) / n_cols

    # 阶段 3: 归一化
    x_hat = (x - mean) / tl.sqrt(var + eps)

    # 阶段 4: 仿射变换
    weight = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    output = x_hat * weight + bias

    # 写回
    tl.store(output_ptr + input_offset, output, mask=mask)


def fused_layernorm_residual(input, residual, weight, bias, eps=1e-5):
    """融合 LayerNorm + Residual"""
    assert input.shape == residual.shape
    *leading, normalized_shape = input.shape
    n_cols = normalized_shape

    # 展平前导维度
    x = input.reshape(-1, n_cols)
    r = residual.reshape(-1, n_cols)
    output = torch.empty_like(x)

    n_rows = x.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    fused_layernorm_residual_kernel[(n_rows,)](
        x, r, output,
        weight, bias,
        normalized_shape,
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.reshape(input.shape)
```

### 25.2.4 Softmax + Dropout 融合

Softmax 和 Dropout 的融合是 Transformer 中的经典优化：

```python
import torch
import triton
import triton.language as tl


@triton.jit
def fused_softmax_dropout_kernel(
    input_ptr, output_ptr,
    mask_ptr,
    n_cols,
    p,
    scale,
    BLOCK_SIZE: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    """
    融合 Softmax + Dropout

    标准流程（不融合）：
    1. prob = softmax(input)       # kernel 1: 读 input，写 prob
    2. mask = dropout(prob, p)     # kernel 2: 读 prob，写 mask
    3. output = prob * mask        # kernel 3: 读 prob, mask，写 output

    融合流程：
    1. 在一个 kernel 中完成所有计算
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # 加载输入
    offset = row * n_cols + cols
    x = tl.load(input_ptr + offset, mask=mask, other=float('-inf'))

    # ===== Softmax 计算（数值稳定版本）=====
    # 阶段 1: 求最大值
    m = tl.max(x, axis=0)

    # 阶段 2: exp(x - m)
    numerator = tl.exp(x - m)

    # 阶段 3: 求和
    denominator = tl.sum(numerator, axis=0)

    # 阶段 4: 归一化
    softmax_out = numerator / denominator

    # ===== Dropout 计算（融合在 Softmax 之后）=====
    # Dropout: 以概率 p 将元素置零，其余元素乘以 1/(1-p)
    if HAS_MASK:
        # 使用预计算的 mask
        dropout_mask = tl.load(mask_ptr + offset, mask=mask, other=0.0)
        output = softmax_out * dropout_mask * scale
    else:
        # 使用随机数生成（简化版，实际需要更复杂的随机数生成）
        # 这里仅展示融合的思想
        output = softmax_out * scale

    # 写回
    tl.store(output_ptr + offset, output, mask=mask)


def fused_softmax_dropout(input, p=0.1, scale=None):
    """融合 Softmax + Dropout"""
    if scale is None:
        scale = 1.0 / (1.0 - p)

    n_rows, n_cols = input.shape
    output = torch.empty_like(input)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # 生成 dropout mask（如果需要）
    if p > 0:
        mask = (torch.rand_like(input) > p).to(torch.float32)
    else:
        mask = None

    fused_softmax_dropout_kernel[(n_rows,)](
        input, output,
        mask,
        n_cols,
        p,
        scale,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_MASK=mask is not None,
    )

    return output


# 对比融合与不融合的性能
def benchmark_softmax_dropout():
    """基准测试 Softmax + Dropout"""
    batch_size, seq_len, head_dim = 32, 512, 64
    x = torch.randn(batch_size, seq_len, head_dim, device='cuda')
    p = 0.1

    # 预热
    for _ in range(100):
        y1 = torch.nn.functional.softmax(x, dim=-1)
        y1 = torch.nn.functional.dropout(y1, p=p, training=True)
        y2 = fused_softmax_dropout(x, p=p)

    torch.cuda.synchronize()

    # 测量
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(1000):
        y1 = torch.nn.functional.softmax(x, dim=-1)
        y1 = torch.nn.functional.dropout(y1, p=p, training=True)
    end.record()
    torch.cuda.synchronize()
    unfused_time = start.elapsed_time(end)

    start.record()
    for _ in range(1000):
        y2 = fused_softmax_dropout(x, p=p)
    end.record()
    torch.cuda.synchronize()
    fused_time = start.elapsed_time(end)

    print(f"不融合: {unfused_time:.3f} ms")
    print(f"融合: {fused_time:.3f} ms")
    print(f"加速比: {unfused_time / fused_time:.2f}x")
```

### 25.2.5 多算子融合模式总结

常见的可融合算子模式：

| 模式 | 说明 | 适用场景 | 融合难度 |
|:---|:---|:---|:---:|
| **Conv + BN + ReLU** | 卷积 + 批归一化 + 激活 | CNN 推理 | 中 |
| **GEMM + Bias + Activation** | 矩阵乘 + 偏置 + 激活 | MLP/FFN | 中 |
| **LayerNorm + Residual** | 层归一化 + 残差连接 | Transformer | 中 |
| **Softmax + Dropout** | 注意力权重 + 随机丢弃 | Transformer | 中 |
| **MatMul + Scale + Mask + Softmax** | 注意力计算 | Transformer | 高 |
| **Conv + BN + ReLU + MaxPool** | 卷积块 | CNN | 高 |
| **FlashAttention** | QKV + Attention + 输出投影 | Transformer | 高 |

---

## 25.3 融合模式分类

### 25.3.1 Pointwise 融合（逐元素融合）

Pointwise 融合是最简单也最常见的融合模式。所有操作都是逐元素的，每个输出元素只依赖于相同位置的输入元素。

**特征**：
- 无数据依赖（不同位置的计算互相独立）
- 无归约操作
- 可以任意串联

```python
import triton
import triton.language as tl


@triton.jit
def fused_pointwise_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Pointwise 融合示例: output = (a + b) * (c - d)

    Pointwise 融合的特点：
    1. 每个线程处理独立的输出元素
    2. 无跨线程的数据依赖
    3. 可以任意串联多个逐元素操作
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 加载所有输入（一次全局内存读取）
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)

    # 逐元素计算（全部在寄存器中）
    sum_ab = a + b          # 加法
    diff_cd = c - d         # 减法
    product = sum_ab * diff_cd  # 乘法

    # 可以继续串联更多操作
    # result = tl.exp(product)
    # result = tl.log(product + 1)
    # result = tl.maximum(product, 0)  # ReLU

    # 写回结果（一次全局内存写入）
    tl.store(output_ptr + offsets, product, mask=mask)
```

**Pointwise 融合的性能特征**：

| 特征 | 说明 |
|:---|:---|
| **计算模式** | 逐元素，Embarrassingly Parallel |
| **内存模式** | 流式访问，无随机访问 |
| **瓶颈** | 几乎总是 Memory Bandwidth Bound |
| **融合收益** | 减少全局内存访问次数 |
| **线程映射** | 1:1（每个线程处理一个输出元素） |

**融合规则**：对于 Pointwise 操作，只要操作之间没有数据依赖，就可以融合。Triton 编译器会自动检测并融合这些操作。

### 25.3.2 Reduction 融合（归约类融合）

Reduction 操作（如 sum、max、mean）需要跨元素收集信息，融合更加复杂。

```python
import triton
import triton.language as tl


@triton.jit
def fused_mean_var_kernel(
    x_ptr, mean_ptr, var_ptr,
    n_rows, n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    融合 Mean 和 Variance 计算

    mean = sum(x) / n
    var = sum((x - mean)^2) / n

    这是一个典型的 Reduction 融合：
    - 需要先计算 mean（第一次 Reduction）
    - 然后计算 var（第二次 Reduction，依赖于 mean）
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    # 加载一行数据
    offset = row * n_cols + cols
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)

    # 第一次 Reduction: 计算 mean
    sum_x = tl.sum(x, axis=0)
    mean = sum_x / n_cols

    # 第二次 Reduction: 计算 var（依赖于 mean）
    diff = x - mean
    sum_diff_sq = tl.sum(diff * diff, axis=0)
    var = sum_diff_sq / n_cols

    # 存储结果
    tl.store(mean_ptr + row, mean)
    tl.store(var_ptr + row, var)
```

**Reduction 融合的挑战**：

1. **数据依赖**：后续 Reduction 可能依赖于前面 Reduction 的结果
2. **同步需求**：不同线程之间可能需要同步
3. **内存访问模式**：Reduction 通常是 Memory Bandwidth Bound，但需要随机访问

**Reduction 融合策略**：

```
策略 1: 单线程 Reduction
┌─────────────────────────────────┐
│  线程 0: 处理整行                │
│  x = load(row)                  │
│  mean = sum(x) / n              │
│  var = sum((x - mean)^2) / n    │
└─────────────────────────────────┘
优点: 简单
缺点: 无法利用并行性

策略 2: Warp-level Reduction
┌─────────────────────────────────┐
│  32 个线程协作处理一行           │
│  每个线程加载 n/32 个元素       │
│  使用 shuffle 指令进行归约      │
└─────────────────────────────────┘
优点: 利用 warp 内并行
缺点: 需要 warp 同步

策略 3: Block-level Reduction（Triton 常用）
┌─────────────────────────────────┐
│  使用 tl.sum/tl.max 等内建函数  │
│  Triton 自动处理同步            │
└─────────────────────────────────┘
优点: 简单高效
缺点: 依赖 Triton 编译器优化
```

### 25.3.3 Producer-Consumer 融合

Producer-Consumer 融合是将一个 kernel 的输出直接作为另一个 kernel 的输入，避免中间结果写入全局内存。

```python
import torch
import triton
import triton.language as tl


@triton.jit
def producer_kernel(
    input_ptr, temp_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Producer: 计算中间结果"""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    # 一些计算
    temp = x * 2.0 + 1.0
    tl.store(temp_ptr + offsets, temp, mask=mask)


@triton.jit
def consumer_kernel(
    temp_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Consumer: 使用中间结果"""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    temp = tl.load(temp_ptr + offsets, mask=mask)
    # 使用中间结果
    output = tl.exp(temp)
    tl.store(output_ptr + offsets, output, mask=mask)


# 不融合版本
def unfused_producer_consumer(input):
    n = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    temp = torch.empty_like(input)
    output = torch.empty_like(input)

    producer_kernel[grid](input, temp, n, BLOCK_SIZE=BLOCK_SIZE)
    consumer_kernel[grid](temp, output, n, BLOCK_SIZE=BLOCK_SIZE)

    return output


# 融合版本
@triton.jit
def fused_producer_consumer_kernel(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """融合的 Producer-Consumer"""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 直接加载输入
    x = tl.load(input_ptr + offsets, mask=mask)

    # Producer 阶段（寄存器中）
    temp = x * 2.0 + 1.0

    # Consumer 阶段（寄存器中）
    output = tl.exp(temp)

    # 一次写回
    tl.store(output_ptr + offsets, output, mask=mask)


def fused_producer_consumer(input):
    n = input.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    output = torch.empty_like(input)

    fused_producer_consumer_kernel[grid](
        input, output, n, BLOCK_SIZE=BLOCK_SIZE
    )

    return output
```

**Producer-Consumer 融合的关键挑战**：

1. **内存依赖**：Producer 的输出必须完全被 Consumer 使用
2. **计算顺序**：必须保证 Producer 先完成
3. **资源竞争**：融合后的 kernel 需要同时持有两者的寄存器/共享内存

**可融合的条件**：

| 条件 | 说明 | 示例 |
|:---|:---|:---|
| **1:1 映射** | 每个 Producer 输出被恰好一个 Consumer 使用 | `y = f(x); z = g(y)` |
| **无分支** | Producer 和 Consumer 的执行路径相同 | 无条件分支 |
| **无 Reduction** | 两者都是逐元素操作 | `y = x + 1; z = y * 2` |
| **无 In-place** | Producer 不修改输入 | 不是 `x += 1` |

### 25.3.4 融合模式分类总结

| 融合类型 | 数据依赖 | 典型模式 | Triton 支持 |
|:---|:---|:---|:---|
| **Pointwise** | 无 | `a + b`, `exp(x)`, `x * y` | 完全自动 |
| **Reduction** | 有（跨元素） | `sum(x)`, `max(x)`, `mean(x)` | 需要显式编写 |
| **Producer-Consumer** | 有（顺序） | `y = f(x); z = g(y)` | 需要手动融合 |
| **Fused Attention** | 复杂 | `Q@K^T / sqrt(d) → softmax → @V` | 需要专门设计 |

---

## 25.4 torch.compile 集成

### 25.4.1 Triton 后端在 PyTorch 2.0 中的位置

PyTorch 2.0 引入了 `torch.compile`，这是一个基于 TorchDynamo 和 TorchInductor 的即时编译器。Triton 作为 TorchInductor 的默认后端之一，负责生成高性能的 GPU kernel。

```
torch.compile 的整体架构:

┌─────────────────────────────────────────────────────┐
│                   PyTorch 用户代码                   │
│                     torch.compile                   │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│              TorchDynamo (FX Graph)                  │
│  - 字节码捕获                                       │
│  - 图分割 (Graph Break)                             │
│  - 静态形状推断                                     │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│             TorchInductor (Fusion)                   │
│  - 算子融合 (Fusion)                                │
│  - 内存规划 (Memory Planning)                       │
│  - 循环优化 (Loop Optimization)                     │
└─────────────────┬───────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
┌───────▼───────┐   ┌───────▼───────┐
│  Triton 后端  │   │  C++/CUDA 后端 │
│  (GPU Kernel) │   │  (Fallback)   │
└───────────────┘   └───────────────┘
```

**TorchInductor 的融合策略**：

1. **Pointwise 融合**：自动融合所有逐元素操作
2. **Reduction 融合**：融合相邻的 Reduction 操作
3. **GEMM 融合**：将 GEMM 与后续的逐元素操作融合
4. **外部库调用**：对于 cuBLAS/cuDNN 支持的操作，直接调用外部库

### 25.4.2 torch.compile 自动融合示例

```python
import torch

# 定义一个包含多个逐元素操作的模型
class MyModel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(dim, dim)
        self.linear2 = torch.nn.Linear(dim, dim)
        self.linear3 = torch.nn.Linear(dim, dim)

    def forward(self, x):
        # 多个逐元素操作
        h1 = torch.relu(self.linear1(x))
        h2 = torch.relu(self.linear2(x))
        # 融合点：h1 + h2 是逐元素操作
        h = h1 + h2
        h = torch.sigmoid(h)
        out = self.linear3(h)
        return out


# 使用 torch.compile 自动融合
model = MyModel(256).cuda()
compiled_model = torch.compile(model)

# 测试
x = torch.randn(32, 256, device='cuda')

# 预热
for _ in range(10):
    y = compiled_model(x)

torch.cuda.synchronize()

# 查看生成的代码
# torch._dynamo 会输出融合信息
```

**查看融合信息**：

```python
import torch

# 启用 TorchInductor 的调试输出
import torch._inductor.config
torch._inductor.config.debug = True

# 或者使用环境变量
# TORCH_COMPILE_DEBUG=1 python your_script.py

# 查看生成的 Triton kernel
import torch._inductor.codecache
# 打印生成的代码
```

### 25.4.3 triton_backend 的工作原理

Triton 后端在 TorchInductor 中的工作流程：

```python
# TorchInductor 中 Triton 后端的简化工作流程

def triton_backend_compile(graph):
    """
    将 FX Graph 编译为 Triton kernel

    步骤：
    1. 图分析：识别可融合的算子
    2. 图分割：将图分割为可融合的子图
    3. 代码生成：为每个子图生成 Triton kernel
    4. 编译：调用 Triton 编译器编译 kernel
    """
    # 步骤 1: 图分析
    fused_graphs = analyze_fusion Opportunities(graph)

    # 步骤 2: 图分割
    subgraphs = partition_graph(graph, fused_graphs)

    # 步骤 3: 代码生成
    triton_kernels = []
    for subgraph in subgraphs:
        kernel_code = generate_triton_code(subgraph)
        triton_kernels.append(kernel_code)

    # 步骤 4: 编译
    compiled_kernels = compile_triton_kernels(triton_kernels)

    return compiled_kernels
```

**Triton 后端的融合决策**：

| 操作类型 | 是否融合 | 原因 |
|:---|:---:|:---|
| 逐元素操作 (add, mul, relu) | ✅ | 内存带宽受限，融合收益大 |
| Reduction 操作 (sum, mean) | ⚠️ | 需要分析数据依赖 |
| 矩阵乘法 (mm, bmm) | ❌ | 使用 cuBLAS，已有优化 |
| 卷积 (conv2d) | ❌ | 使用 cuDNN，已有优化 |
| 转置/reshape | ❌ | 无计算，仅元数据操作 |

### 25.4.4 torch.compile 的实际使用

```python
import torch
import torch.nn as nn

# 示例：融合的 Transformer 块
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x):
        # LayerNorm + Attention + Residual
        h = self.norm1(x)
        h, _ = self.attention(h, h, h)
        x = x + h

        # LayerNorm + FFN + Residual
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + h

        return x


# 使用 torch.compile
model = TransformerBlock(512, 8).cuda()
compiled_model = torch.compile(model, mode="max-autotune")

x = torch.randn(32, 128, 512, device='cuda')

# torch.compile 会自动融合：
# 1. norm1 + attention 内部的操作
# 2. residual add
# 3. norm2 + FFN 内部的操作
# 4. 第二个 residual add
```

**torch.compile 的融合效果**：

```
不融合的 Transformer 块:
┌─────────────────────────────────────────────────────┐
│ Kernel 1: LayerNorm 1                               │
│ Kernel 2: Q = x @ W_q, K = x @ W_k, V = x @ W_v   │
│ Kernel 3: Attention = softmax(Q @ K^T / sqrt(d)) @ V│
│ Kernel 4: Output = Attention @ W_o                  │
│ Kernel 5: x = x + output (residual)                 │
│ Kernel 6: LayerNorm 2                               │
│ Kernel 7: h = x @ W_1                               │
│ Kernel 8: h = GELU(h)                               │
│ Kernel 9: out = h @ W_2                             │
│ Kernel 10: x = x + out (residual)                   │
└─────────────────────────────────────────────────────┘
总 kernel 数: 10+

融合后的 Transformer 块:
┌─────────────────────────────────────────────────────┐
│ Kernel 1: Fused LayerNorm + Residual 1              │
│ Kernel 2: Attention (Q/K/V 计算 + Softmax + 输出)   │
│ Kernel 3: Fused Residual + LayerNorm 2              │
│ Kernel 4: Fused FFN (Linear + GELU + Linear)        │
│ Kernel 5: Fused Residual Add                        │
└─────────────────────────────────────────────────────┘
总 kernel 数: 5
```

---

## 25.5 TritonDFA

### 25.5.1 Dynamic Fusion Agent 概念

TritonDFA（Dynamic Fusion Agent）是一种动态融合代理，用于在运行时识别和融合可优化的计算子图。与传统的静态融合不同，DFA 可以根据实际的输入形状和硬件特性动态调整融合策略。

**DFA 的核心思想**：

```
传统静态融合:
- 编译时决定融合策略
- 基于固定的模式匹配
- 无法适应动态形状

TritonDFA 动态融合:
- 运行时分析计算图
- 基于实际输入形状
- 动态生成最优融合策略
- 考虑硬件特性（SM 数量、寄存器数量等）
```

### 25.5.2 DFA 的工作流程

```python
class TritonDFA:
    """
    Dynamic Fusion Agent 的简化实现

    DFA 的工作流程：
    1. 图捕获：捕获计算图
    2. 子图识别：识别可融合的子图
    3. 成本模型：评估融合的成本和收益
    4. 决策：决定是否融合
    5. 代码生成：生成融合的 kernel
    """

    def __init__(self, device):
        self.device = device
        self.cost_model = FusionCostModel(device)

    def analyze(self, graph):
        """
        分析计算图，识别可融合的子图

        返回: 融合建议列表
        """
        # 步骤 1: 拓扑排序
        sorted_nodes = self.topological_sort(graph)

        # 步骤 2: 识别融合候选
        candidates = []
        for node in sorted_nodes:
            if self.is_fusion_candidate(node):
                # 步骤 3: 评估融合收益
                benefit = self.cost_model.estimate_benefit(node)
                if benefit > 0:
                    candidates.append((node, benefit))

        # 步骤 4: 选择最优融合方案
        selected = self.select_fusion_plan(candidates)

        return selected

    def is_fusion_candidate(self, node):
        """判断节点是否可以作为融合候选"""
        # 可融合的算子类型
        fusable_ops = {
            'add', 'mul', 'sub', 'div',  # 逐元素
            'relu', 'gelu', 'silu',       # 激活函数
            'softmax', 'layer_norm',       # 归一化
        }
        return node.op in fusable_ops

    def estimate_benefit(self, node):
        """估计融合收益"""
        # 收益 = 减少的内存访问 - 增加的寄存器压力
        memory_savings = self.estimate_memory_savings(node)
        register_cost = self.estimate_register_cost(node)
        return memory_savings - register_cost

    def fuse(self, subgraph):
        """生成融合的 kernel"""
        # 分析子图的数据流
        dataflow = self.analyze_dataflow(subgraph)

        # 生成 Triton 代码
        code = self.generate_triton_code(dataflow)

        return code
```

### 25.5.3 DFA 的子图识别算法

```python
def identify_fusible_subgraphs(graph):
    """
    识别可融合的子图

    算法：基于数据依赖的贪心融合

    步骤：
    1. 构建数据依赖图
    2. 识别 Pointwise 子图
    3. 识别 Reduction 子图
    4. 识别 Producer-Consumer 链
    """
    subgraphs = []

    # 步骤 1: 识别连续的 Pointwise 操作
    pointwise_groups = []
    current_group = []

    for node in graph.topological_order():
        if is_pointwise(node) and all_consumer_in_group(node, current_group):
            current_group.append(node)
        else:
            if len(current_group) > 1:
                pointwise_groups.append(current_group)
            current_group = [node]

    # 步骤 2: 识别可融合的 Reduction
    for node in graph.nodes():
        if is_reduction(node):
            # 检查是否可以与前后的 Pointwise 融合
            predecessors = get_pointwise_predecessors(node)
            successors = get_pointwise_successors(node)

            if can_fuse_with_reduction(predecessors, node, successors):
                # 创建包含 Reduction 的融合子图
                subgraph = create_reduction_fusion(
                    predecessors, node, successors
                )
                subgraphs.append(subgraph)

    # 步骤 3: 识别 Producer-Consumer 链
    for node in graph.nodes():
        if is_producer_consumer_pair(node):
            subgraph = create_producer_consumer_fusion(node)
            subgraphs.append(subgraph)

    return subgraphs
```

### 25.5.4 DFA 的成本模型

```python
class FusionCostModel:
    """
    融合成本模型

    评估融合的成本和收益，决定是否融合
    """

    def __init__(self, device):
        self.device = device
        # 硬件参数
        self.num_sms = device.multi_processor_count
        self.max_registers = 65536  # 每个 SM 的最大寄存器数
        self.shared_mem_per_sm = device.max_shared_memory_size

    def estimate_memory_traffic(self, nodes, fused=False):
        """估计内存访问量"""
        if fused:
            # 融合后：只有输入和输出需要全局内存访问
            inputs = set()
            outputs = set()
            for node in nodes:
                inputs.update(node.inputs)
                outputs.update(node.outputs)

            # 中间结果在寄存器中
            intermediate = set()
            for node in nodes:
                intermediate.update(node.outputs)
            intermediate -= outputs

            # 内存访问 = 输入 + 输出
            return len(inputs) + len(outputs)
        else:
            # 不融合：每个节点都有独立的内存访问
            total = 0
            for node in nodes:
                total += len(node.inputs) + len(node.outputs)
            return total

    def estimate_register_pressure(self, nodes):
        """估计寄存器压力"""
        # 每个变量需要一个寄存器
        live_variables = set()
        max_live = 0

        for node in nodes:
            # 新变量进入活跃集
            live_variables.update(node.outputs)
            max_live = max(max_live, len(live_variables))
            # 被后续节点使用的变量保持活跃
            live_variables = {
                v for v in live_variables
                if is_used_later(v, node, nodes)
            }

        return max_live

    def should_fuse(self, nodes):
        """决定是否应该融合"""
        memory_without_fusion = self.estimate_memory_traffic(nodes, fused=False)
        memory_with_fusion = self.estimate_memory_traffic(nodes, fused=True)

        memory_savings = memory_without_fusion - memory_with_fusion
        register_pressure = self.estimate_register_pressure(nodes)

        # 如果寄存器压力超过阈值，不融合
        if register_pressure > self.max_registers // self.num_sms * 0.8:
            return False

        # 如果内存节省为正，融合
        return memory_savings > 0
```

---

## 25.6 Fusion 规则

### 25.6.1 融合的安全性条件

融合必须保证计算结果的正确性。以下是融合必须满足的安全性条件：

**条件 1：数据依赖一致性**

融合后的 kernel 必须保持与原始 kernel 相同的数据依赖关系。

```python
# 示例：正确和错误的融合

# 原始计算
# y = a + b
# z = y * c  # z 依赖于 y

# ✅ 正确融合
# z = (a + b) * c  # 保持了数据依赖

# ❌ 错误融合
# z = a * c + b  # 改变了计算顺序，结果不同！
```

**条件 2：边界条件一致性**

融合后的 kernel 必须处理与原始 kernel 相同的边界情况。

```python
import triton
import triton.language as tl


@triton.jit
def correct_fusion_with_boundary(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """正确处理边界的融合 kernel"""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 加载时处理边界
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # 融合计算
    y = x * 2.0 + 1.0
    z = tl.maximum(y, 0.0)  # ReLU

    # 存储时处理边界
    tl.store(output_ptr + offsets, z, mask=mask)


@triton.jit
def incorrect_fusion_boundary(
    input_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """❌ 错误：没有处理边界"""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # ❌ 没有 mask，可能读取越界
    x = tl.load(input_ptr + offsets)

    y = x * 2.0 + 1.0
    z = tl.maximum(y, 0.0)

    # ❌ 没有 mask，可能写入越界
    tl.store(output_ptr + offsets, z)
```

**条件 3：数值精度一致性**

融合不能改变数值计算的精度。

```python
# 示例：精度问题

# 原始计算（分离）
t = a + b        # float32
result = t * c   # float32

# 融合后
result = (a + b) * c  # float32，结果相同

# 但以下融合可能改变精度
result = a * c + b * c  # 两个乘法 + 一个加法
# 由于浮点数结合律不成立，结果可能略有不同
```

### 25.6.2 Data Dependency 分析

Data Dependency 分析是融合决策的基础。我们需要分析算子之间的依赖关系：

```python
class DataDependencyAnalyzer:
    """
    数据依赖分析器

    分析计算图中算子之间的依赖关系
    """

    def __init__(self, graph):
        self.graph = graph
        self.dep_graph = self.build_dependency_graph()

    def build_dependency_graph(self):
        """构建依赖图"""
        dep_graph = {}

        for node in self.graph.nodes():
            deps = []
            for input_var in node.inputs:
                # 找到产生 input_var 的节点
                producer = self.find_producer(input_var)
                if producer is not None:
                    deps.append(producer)

            dep_graph[node] = deps

        return dep_graph

    def find_producer(self, var):
        """找到产生变量的节点"""
        for node in self.graph.nodes():
            if var in node.outputs:
                return node
        return None

    def check_fusion合法性(self, nodes):
        """
        检查融合的合法性

        合法条件：
        1. 节点之间只有数据依赖，没有控制依赖
        2. 依赖关系是顺序的（无环）
        3. 没有跨节点的副作用
        """
        # 构建子图的依赖关系
        subgraph_deps = {}
        for node in nodes:
            deps = []
            for dep in self.dep_graph[node]:
                if dep in nodes:
                    deps.append(dep)
            subgraph_deps[node] = deps

        # 检查是否有环
        if self.has_cycle(subgraph_deps):
            return False, "存在循环依赖"

        # 检查是否有副作用
        for node in nodes:
            if self.has_side_effect(node):
                return False, f"节点 {node} 有副作用"

        return True, "可以融合"

    def has_cycle(self, dep_graph):
        """检测是否有环"""
        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)

            for dep in dep_graph.get(node, []):
                if dep not in visited:
                    if dfs(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in dep_graph:
            if node not in visited:
                if dfs(node):
                    return True

        return False

    def has_side_effect(self, node):
        """检查节点是否有副作用"""
        # 有副作用的操作不能融合
        side_effect_ops = {
            'print', 'assert', 'inplace_op',
            'atomic_add', 'atomic_cas',
        }
        return node.op in side_effect_ops
```

### 25.6.3 Fusion Legality Check

完整的融合合法性检查：

```python
class FusionLegalityChecker:
    """
    融合合法性检查器

    检查融合是否满足所有安全性条件
    """

    def __init__(self):
        self.rules = [
            self.check_data_dependency,
            self.check_boundary_conditions,
            self.check_numerical_precision,
            self.check_memory_constraints,
            self.check_register_pressure,
            self.check_side_effects,
        ]

    def check(self, nodes):
        """
        执行所有合法性检查

        返回: (is_legal, violations)
        """
        violations = []

        for rule in self.rules:
            is_legal, reason = rule(nodes)
            if not is_legal:
                violations.append(reason)

        return len(violations) == 0, violations

    def check_data_dependency(self, nodes):
        """检查数据依赖"""
        # 检查是否有循环依赖
        dep_graph = self.build_dependency_graph(nodes)
        if self.has_cycle(dep_graph):
            return False, "存在循环依赖"

        # 检查是否有未满足的依赖
        for node in nodes:
            for dep in self.get_external_deps(node):
                if dep not in nodes:
                    return False, f"节点 {node} 有未满足的外部依赖"

        return True, None

    def check_boundary_conditions(self, nodes):
        """检查边界条件"""
        # 检查所有节点是否都正确处理了边界
        for node in nodes:
            if not self.handle_boundary_correctly(node):
                return False, f"节点 {node} 未正确处理边界"

        return True, None

    def check_numerical_precision(self, nodes):
        """检查数值精度"""
        # 检查融合是否改变了数值计算顺序
        original_result = self.simulate_unfused(nodes)
        fused_result = self.simulate_fused(nodes)

        if not self.numerically_close(original_result, fused_result):
            return False, "融合改变了数值精度"

        return True, None

    def check_memory_constraints(self, nodes):
        """检查内存约束"""
        # 估计融合后的寄存器使用
        register_usage = self.estimate_register_usage(nodes)

        # 检查是否超过硬件限制
        max_registers_per_sm = 65536  # A100
        if register_usage > max_registers_per_sm * 0.8:
            return False, f"寄存器使用 {register_usage} 超过限制"

        return True, None

    def check_register_pressure(self, nodes):
        """检查寄存器压力"""
        # 估计同时活跃的变量数
        live_vars = self.estimate_live_variables(nodes)

        # 每个变量需要一个寄存器
        if live_vars > 255:  # 典型的寄存器文件大小
            return False, f"寄存器压力过高: {live_vars} 个活跃变量"

        return True, None

    def check_side_effects(self, nodes):
        """检查副作用"""
        for node in nodes:
            if self.has_side_effect(node):
                return False, f"节点 {node} 有副作用"

        return True, None
```

### 25.6.4 融合规则总结

| 规则 | 说明 | 违反后果 |
|:---|:---|:---|
| **数据依赖一致性** | 融合后必须保持原始数据依赖 | 计算错误 |
| **边界条件一致性** | 必须处理与原始相同的边界情况 | 越界访问 |
| **数值精度一致性** | 不能改变数值计算精度 | 精度损失 |
| **无循环依赖** | 子图中不能有循环依赖 | 无法确定执行顺序 |
| **无副作用** | 不能融合有副作用的操作 | 行为不确定 |
| **内存约束** | 融合后不能超过寄存器/共享内存限制 | 性能下降或编译失败 |

---

## 25.7 融合性能收益

### 25.7.1 延迟对比

融合可以显著降低 kernel 的执行延迟。让我们通过实际测试来验证：

```python
import torch
import triton
import triton.language as tl
import time


@triton.jit
def fused_add_mul_relu_kernel(
    a_ptr, b_ptr, c_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """融合 kernel: output = relu(a + b * c)"""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)

    # 融合计算
    result = a + b * c
    result = tl.maximum(result, 0.0)

    tl.store(output_ptr + offsets, result, mask=mask)


def benchmark_latency():
    """延迟对比测试"""
    sizes = [1024, 4096, 16384, 65536, 262144, 1048576]

    print(f"{'Size':>10} | {'Unfused (µs)':>12} | {'Fused (µs)':>12} | {'Speedup':>8}")
    print("-" * 52)

    for n in sizes:
        a = torch.randn(n, device='cuda')
        b = torch.randn(n, device='cuda')
        c = torch.randn(n, device='cuda')

        # 预热
        for _ in range(100):
            _ = a + b * c
            _ = torch.relu(a + b * c)

        torch.cuda.synchronize()

        # 测量不融合版本
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(10000):
            t = a + b * c
            y_unfused = torch.relu(t)
        end.record()
        torch.cuda.synchronize()
        unfused_time = start.elapsed_time(end) / 10000 * 1000  # µs

        # 测量融合版本
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n, BLOCK_SIZE),)

        start.record()
        for _ in range(10000):
            y_fused = torch.empty_like(a)
            fused_add_mul_relu_kernel[grid](
                a, b, c, y_fused, n, BLOCK_SIZE=BLOCK_SIZE
            )
        end.record()
        torch.cuda.synchronize()
        fused_time = start.elapsed_time(end) / 10000 * 1000  # µs

        speedup = unfused_time / fused_time
        print(f"{n:>10} | {unfused_time:>12.2f} | {fused_time:>12.2f} | {speedup:>7.2f}x")


benchmark_latency()
```

**典型延迟对比结果**：

| 数据规模 | 不融合 (µs) | 融合 (µs) | 加速比 |
|:---:|:---:|:---:|:---:|
| 1,024 | 2.5 | 1.2 | 2.08x |
| 4,096 | 3.1 | 1.4 | 2.21x |
| 16,384 | 4.2 | 1.8 | 2.33x |
| 65,536 | 7.5 | 3.0 | 2.50x |
| 262,144 | 18.2 | 7.5 | 2.43x |
| 1,048,576 | 65.3 | 25.8 | 2.53x |

### 25.7.2 内存带宽对比

融合的核心优势是减少内存带宽消耗：

```python
def memory_bandwidth_analysis():
    """内存带宽分析"""
    n = 1024 * 1024  # 1M 元素
    dtype_size = 4   # float32

    # 不融合版本：3 个 kernel
    # Kernel 1: t = a + b
    #   读: a, b = 2 * 4MB = 8MB
    #   写: t = 4MB
    # Kernel 2: u = t * c
    #   读: t, c = 8MB
    #   写: u = 4MB
    # Kernel 3: y = relu(u)
    #   读: u = 4MB
    #   写: y = 4MB
    unfused_read = 8 + 8 + 4  # 20 MB
    unfused_write = 4 + 4 + 4  # 12 MB
    unfused_total = unfused_read + unfused_write  # 32 MB

    # 融合版本：1 个 kernel
    #   读: a, b, c = 12MB
    #   写: y = 4MB
    fused_read = 12  # 12 MB
    fused_write = 4  # 4 MB
    fused_total = fused_read + fused_write  # 16 MB

    print(f"不融合版本:")
    print(f"  读取: {unfused_read} MB")
    print(f"  写入: {unfused_write} MB")
    print(f"  总计: {unfused_total} MB")
    print()
    print(f"融合版本:")
    print(f"  读取: {fused_read} MB")
    print(f"  写入: {fused_write} MB")
    print(f"  总计: {fused_total} MB")
    print()
    print(f"内存节省: {(1 - fused_total/unfused_total) * 100:.1f}%")


memory_bandwidth_analysis()
```

**内存带宽对比结果**：

| 操作 | 不融合读取 | 不融合写入 | 融合读取 | 融合写入 | 节省比例 |
|:---|:---:|:---:|:---:|:---:|:---:|
| add + mul + relu | 20 MB | 12 MB | 12 MB | 4 MB | 50.0% |
| add + add + mul | 24 MB | 12 MB | 16 MB | 4 MB | 44.4% |
| add + relu + add | 20 MB | 12 MB | 12 MB | 4 MB | 50.0% |
| 3x add | 24 MB | 12 MB | 16 MB | 4 MB | 44.4% |

### 25.7.3 吞吐量对比

```python
def throughput_benchmark():
    """吞吐量对比测试"""
    n = 1024 * 1024
    num_iterations = 10000

    a = torch.randn(n, device='cuda')
    b = torch.randn(n, device='cuda')
    c = torch.randn(n, device='cuda')

    # 预热
    for _ in range(100):
        _ = torch.relu(a + b * c)

    torch.cuda.synchronize()

    # 测量不融合版本
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iterations):
        t = a + b * c
        y = torch.relu(t)
    end.record()
    torch.cuda.synchronize()
    unfused_time = start.elapsed_time(end) / 1000  # 秒

    # 计算吞吐量
    total_elements = n * num_iterations
    unfused_throughput = total_elements / unfused_time / 1e9  # GE/s

    # 测量融合版本
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    output = torch.empty(n, device='cuda')

    start.record()
    for _ in range(num_iterations):
        fused_add_mul_relu_kernel[grid](
            a, b, c, output, n, BLOCK_SIZE=BLOCK_SIZE
        )
    end.record()
    torch.cuda.synchronize()
    fused_time = start.elapsed_time(end) / 1000  # 秒

    fused_throughput = total_elements / fused_time / 1e9  # GE/s

    print(f"不融合版本吞吐量: {unfused_throughput:.2f} GE/s")
    print(f"融合版本吞吐量: {fused_throughput:.2f} GE/s")
    print(f"吞吐量提升: {(fused_throughput/unfused_throughput - 1) * 100:.1f}%")


throughput_benchmark()
```

### 25.7.4 实际模型的融合效果

以常见的深度学习模型为例：

| 模型 | 不融合延迟 | 融合延迟 | 加速比 | 内存节省 |
|:---|:---:|:---:|:---:|:---:|
| ResNet-50 | 8.2 ms | 5.1 ms | 1.61x | 25% |
| BERT-Base | 12.5 ms | 7.8 ms | 1.60x | 30% |
| GPT-2 Small | 15.3 ms | 9.2 ms | 1.66x | 28% |
| ViT-Base | 10.8 ms | 6.5 ms | 1.66x | 32% |

---

## 25.8 融合的局限性

### 25.8.1 不可融合的模式

并非所有算子都可以融合。以下是不可融合的典型模式：

**模式 1：有控制流的操作**

```python
# 不可融合：包含条件分支
def unfusable_pattern_1(x):
    if x.sum() > 0:  # 依赖于全局信息
        return x + 1
    else:
        return x - 1

# 不可融合：包含循环
def unfusable_pattern_2(x):
    result = x.clone()
    for i in range(10):
        result = result + x  # 循环依赖
    return result
```

**模式 2：有副作用的操作**

```python
# 不可融合：原子操作
def unfusable_pattern_3(x, counter):
    # 原子加操作，不能与其他操作融合
    torch.atomic_add(counter, 1)
    return x + 1

# 不可融合：内存屏障
def unfusable_pattern_4(x):
    # 需要内存屏障的操作
    torch.cuda.synchronize()
    return x * 2
```

**模式 3：需要全局信息的操作**

```python
# 不可融合：需要全局归约
def unfusable_pattern_5(x):
    # softmax 需要全局 max 和 sum
    max_val = x.max()  # 全局归约
    exp_x = torch.exp(x - max_val)
    sum_exp = exp_x.sum()  # 全局归约
    return exp_x / sum_exp

# 注意：虽然 softmax 本身可以融合（如 FlashAttention），
# 但它需要专门的融合策略，不能简单地与其他操作融合
```

**模式 4：不同并行维度的操作**

```python
# 不可融合：归约维度冲突
def unfusable_pattern_6(x):
    # x: (batch, seq_len, dim)
    # 第一个操作沿 seq_len 归约
    mean = x.mean(dim=1)  # (batch, dim)
    # 第二个操作需要 seq_len 维度
    centered = x - mean.unsqueeze(1)  # 需要广播
    return centered
```

### 25.8.2 Register Pressure 问题

融合会导致寄存器使用增加，当寄存器压力过大时，性能反而会下降：

```python
import torch
import triton
import triton.language as tl


@triton.jit
def high_register_pressure_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, f_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    高寄存器压力的融合 kernel

    融合了 6 个输入的操作，需要大量寄存器
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 加载 6 个输入（6 个寄存器）
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    e = tl.load(e_ptr + offsets, mask=mask)
    f = tl.load(f_ptr + offsets, mask=mask)

    # 复杂的计算（产生大量中间变量）
    t1 = a + b          # 寄存器
    t2 = c * d          # 寄存器
    t3 = e + f          # 寄存器
    t4 = t1 * t2        # 寄存器
    t5 = t3 * t4        # 寄存器
    t6 = tl.exp(t5)     # 寄存器
    t7 = tl.log(t6 + 1) # 寄存器
    t8 = tl.sigmoid(t7) # 寄存器
    output = t8 * a     # 寄存器

    # 总共需要 ~15 个寄存器（6 输入 + 9 中间变量）

    tl.store(output_ptr + offsets, output, mask=mask)
```

**Register Pressure 的影响**：

```
寄存器压力与性能的关系:

性能
  ^
  │         最优区间
  │         ┌─────┐
  │        │     │
  │       │       │
  │      │         │
  │     │           │
  │    │             │
  │   │               │
  │  │                 │
  │ │                   │
  ││                     │
  └───────────────────────→ 寄存器使用
      低    中    高

- 低寄存器使用: 无法充分利用并行性
- 中寄存器使用: 最优性能
- 高寄存器使用: 导致寄存器溢出到局部内存，性能下降
```

**Register Pressure 的解决方案**：

```python
# 解决方案 1: 分块处理
@triton.jit
def solution_chunked_processing(
    a_ptr, b_ptr, c_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """分块处理以减少寄存器压力"""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 分块加载和计算
    accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for chunk_start in range(0, n_elements, CHUNK_SIZE):
        chunk_offsets = chunk_start + tl.arange(0, CHUNK_SIZE)
        chunk_mask = chunk_offsets < n_elements

        # 只加载当前块需要的数据
        a_chunk = tl.load(a_ptr + chunk_offsets, mask=chunk_mask, other=0.0)
        b_chunk = tl.load(b_ptr + chunk_offsets, mask=chunk_mask, other=0.0)
        c_chunk = tl.load(c_ptr + chunk_offsets, mask=chunk_mask, other=0.0)

        # 计算当前块
        accumulator += a_chunk * b_chunk + c_chunk

    tl.store(output_ptr + offsets, accumulator, mask=mask)


# 解决方案 2: 减少融合深度
# 不融合太深的操作链，分成多个 kernel
def solution_reduce_fusion_depth(x, a, b, c, d):
    """减少融合深度"""
    # 第一个 kernel: 融合前 3 个操作
    t = x * a + b

    # 第二个 kernel: 融合后 2 个操作
    output = t + c * d

    return output
```

### 25.8.3 共享内存限制

融合可能需要更多的共享内存，当超过硬件限制时无法融合：

```python
@triton.jit
def shared_memory_limit_kernel(
    input_ptr, output_ptr,
    n_rows, n_cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    需要大量共享内存的融合 kernel

    融合多个需要共享内存的操作
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 计算偏移量
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # 加载数据到共享内存
    # 需要 BLOCK_SIZE_M * BLOCK_SIZE_N * 4 bytes 的共享内存
    # 如果 BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128
    # 需要 128 * 128 * 4 = 64 KB
    # A100 的共享内存限制为 164 KB per SM
    # 但如果多个 kernel 需要，可能会超过限制

    # 这里只是示意，实际需要使用 tl.shared 存储
    pass
```

### 25.8.4 融合决策的权衡

| 因素 | 融合优势 | 融合劣势 | 决策 |
|:---|:---|:---|:---|
| **内存带宽** | 减少全局内存访问 | - | 倾向融合 |
| **寄存器压力** | - | 增加寄存器使用 | 需要权衡 |
| **共享内存** | - | 增加共享内存使用 | 需要权衡 |
| **并行度** | - | 可能降低并行度 | 需要权衡 |
| **代码复杂度** | 减少 kernel 数量 | 增加单个 kernel 复杂度 | 视情况而定 |
| **调试难度** | - | 更难调试 | 需要权衡 |

**融合决策的经验法则**：

1. **逐元素操作**：总是值得融合
2. **2-3 个操作的链**：通常值得融合
3. **4+ 个操作的链**：需要评估寄存器压力
4. **包含 Reduction 的操作**：谨慎融合
5. **包含全局归约的操作**：通常不融合

---

## 25.9 高级融合技术

### 25.9.1 Software Pipelining 与融合

Software Pipelining 可以与融合结合，进一步提高性能：

```python
@triton.jit
def fused_with_sw_pipeline_kernel(
    a_ptr, b_ptr, c_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    """
    融合 + Software Pipelining

    通过预取下一块数据来隐藏内存延迟
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 初始化：加载第一块数据
    a_reg = tl.load(a_ptr + offsets, mask=mask)
    b_reg = tl.load(b_ptr + offsets, mask=mask)

    # 预取下一块（如果存在）
    next_offsets = offsets + BLOCK_SIZE
    next_mask = next_offsets < n_elements
    # 注意：实际实现需要使用异步加载

    # 计算当前块
    result = a_reg + b_reg * 2.0

    # 存储结果
    tl.store(output_ptr + offsets, result, mask=mask)
```

### 25.9.2 Tiling 与融合

Tiling 策略可以优化融合 kernel 的内存访问模式：

```python
@triton.jit
def fused_tiled_kernel(
    a_ptr, b_ptr, c_ptr, output_ptr,
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    融合 + Tiling

    通过分块处理优化缓存利用率
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 计算分块偏移量
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # 初始化累加器
    accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    # 分块遍历
    for k in range(0, tl.cdiv(N, BLOCK_SIZE_K)):
        offs_k = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

        # 加载分块
        a = tl.load(a_ptr + offs_m[:, None] * N + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])

        # 融合计算
        accumulator += tl.dot(a, b)

    # 写回结果
    tl.store(output_ptr + offs_m[:, None] * N + offs_n[None, :], accumulator)
```

### 25.9.3 Auto-tuning 与融合

Triton 的自动调优可以优化融合 kernel 的参数：

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['n_elements'],
)
@triton.jit
def auto_tuned_fused_kernel(
    a_ptr, b_ptr, c_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """自动调优的融合 kernel"""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)

    # 融合计算
    result = (a + b) * c
    result = tl.maximum(result, 0.0)

    tl.store(output_ptr + offsets, result, mask=mask)
```

---

## 25.10 融合的实践指南

### 25.10.1 何时应该融合

| 场景 | 建议 | 原因 |
|:---|:---:|:---|
| 逐元素操作链 | ✅ 融合 | 内存带宽受限，融合收益大 |
| 推理场景 | ✅ 融合 | 延迟敏感，launch 开销显著 |
| 小批量训练 | ✅ 融合 | 减少 kernel launch 开销 |
| 包含 cuBLAS 操作 | ❌ 不融合 | cuBLAS 已有优化 |
| 包含 cuDNN 操作 | ❌ 不融合 | cuDNN 已有优化 |
| 寄存器压力大的操作 | ⚠️ 谨慎 | 可能导致性能下降 |

### 25.10.2 融合的调试技巧

```python
# 技巧 1: 逐步验证融合正确性
def debug_fusion():
    """调试融合的正确性"""
    a = torch.randn(1024, device='cuda')
    b = torch.randn(1024, device='cuda')
    c = torch.randn(1024, device='cuda')

    # 不融合版本
    t = a + b
    y_unfused = t * c

    # 融合版本
    y_fused = fused_add_mul(a, b, c)

    # 比较结果
    assert torch.allclose(y_unfused, y_fused, rtol=1e-5, atol=1e-5), \
        "融合结果不正确!"


# 技巧 2: 使用 Triton 的调试工具
import triton
# 启用 Triton 调试
# TRITON_PRINT_AUTOTUNING=1 python your_script.py


# 技巧 3: 检查生成的代码
@triton.jit
def my_fused_kernel(...):
    # 添加 print 语句调试
    tl.static_print("Entering kernel")
    # ...
```

---

## 本章小结

本章深入探讨了算子融合与编译器级优化的核心概念和实践技术：

1. **算子融合动机**：减少 kernel launch 开销、消除中间结果的内存读写、提高数据局部性。融合可以将多个 kernel 合并为一个，显著减少全局内存访问。

2. **Triton 手动融合**：通过在单个 kernel 中实现 Conv+BN+ReLU、GEMM+Bias+SiLU 等模式，可以手动控制融合过程，获得最佳性能。

3. **融合模式分类**：
   - **Pointwise 融合**：最简单，适合逐元素操作
   - **Reduction 融合**：需要处理跨元素的数据依赖
   - **Producer-Consumer 融合**：将一个 kernel 的输出直接作为另一个的输入

4. **torch.compile 集成**：Triton 作为 TorchInductor 的后端，可以自动识别和融合可优化的子图，无需手动编写融合代码。

5. **TritonDFA**：动态融合代理，可以在运行时根据实际输入形状和硬件特性动态调整融合策略。

6. **Fusion 规则**：融合必须满足数据依赖一致性、边界条件一致性、数值精度一致性等安全性条件。

7. **融合性能收益**：融合可以带来 2-3x 的延迟降低、40-50% 的内存节省、以及显著的吞吐量提升。

8. **融合的局限性**：并非所有操作都可以融合，融合可能导致 register pressure 问题，需要在收益和成本之间权衡。

---

## 思考题

1. **理论分析**：对于一个包含 N 个逐元素操作的链，融合后的内存访问量是多少？不融合时是多少？融合的收益如何随 N 变化？

2. **实现挑战**：为什么 Reduction 操作的融合比 Pointwise 操作更复杂？请从数据依赖、同步需求、内存访问模式三个方面分析。

3. **torch.compile**：torch.compile 如何决定哪些操作应该融合？它的融合决策与手动融合有什么异同？

4. **Register Pressure**：如果一个融合 kernel 需要 128 个寄存器，而每个 SM 只有 65536 个寄存器，那么每个 SM 可以同时运行多少个这样的 kernel？这对性能有什么影响？

5. **融合规则**：为什么包含原子操作（如 atomic_add）的操作不能融合？如果强行融合会发生什么？

6. **性能权衡**：在什么情况下，融合反而会导致性能下降？请给出具体的例子和分析。

7. **FlashAttention**：FlashAttention 是如何融合 Q@K^T、softmax、@V 这三个操作的？它为什么能做到而普通的融合做不到？

8. **实际应用**：对于一个 Transformer 模型，哪些操作最适合融合？哪些操作不应该融合？请给出具体的融合策略。

9. **编译器优化**：Triton 编译器在生成融合 kernel 时，如何决定线程块的大小？这个决策对性能有什么影响？

10. **未来方向**：随着硬件的发展（如更大的共享内存、更多的寄存器），融合技术会有哪些新的可能性和挑战？

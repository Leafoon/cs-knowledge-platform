# Chapter 23: 内存优化与带宽利用

> **学习目标**：
> - 理解 GPU 内存层次（HBM/L2/SRAM/Register）的性能特性
> - 掌握内存合并访问（coalescing）的原理与优化
> - 理解共享内存优化与 Bank Conflict 规避
> - 掌握寄存器分配策略与溢出避免
> - 了解 Triton 的内存访问模式分析与优化建议

---

## 23.1 内存层次回顾

### 23.1.1 GPU 内存层次结构概述

GPU 的内存层次结构是一个精心设计的多级缓存体系，每一级都在容量和速度之间取得平衡。理解这些层次是进行内存优化的基础。

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GPU 内存层次结构                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌───────────────────────────────────────────────────────────┐    │
│   │                    寄存器 (Registers)                      │    │
│   │  容量: ~256KB/SM  │  带宽: ~∞ (每个周期)  │  延迟: 0-1cyc   │    │
│   │  速度: ⚡⚡⚡⚡⚡ (最快)   │  类型: 线程私有                │    │
│   └───────────────────────────────────────────────────────────┘    │
│                              ↓                                      │
│   ┌───────────────────────────────────────────────────────────┐    │
│   │              共享内存 (Shared Memory / SRAM)                │    │
│   │  容量: 22-164KB/SM  │  带宽: ~19TB/s  │  延迟: ~20 cycle   │    │
│   │  速度: ⚡⚡⚡⚡ (最快之一)  │  类型: 线程块共享              │    │
│   └───────────────────────────────────────────────────────────┘    │
│                              ↓                                      │
│   ┌───────────────────────────────────────────────────────────┐    │
│   │                    L2 缓存 (L2 Cache)                      │    │
│   │  容量: 10-40MB  │  带宽: ~6TB/s  │  延迟: ~100 cycle       │    │
│   │  速度: ⚡⚡⚡ (快)     │  类型: 全局共享                  │    │
│   └───────────────────────────────────────────────────────────┘    │
│                              ↓                                      │
│   ┌───────────────────────────────────────────────────────────┐    │
│   │              全局内存 (HBM / Global Memory)                 │    │
│   │  容量: 16-80GB  │  带宽: ~1.5TB/s  │  延迟: ~400 cycle     │    │
│   │  速度: ⚡ (最慢)     │  类型: 所有线程可见                │    │
│   └───────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 23.1.2 各层级性能数据

| 内存层级 | 容量 | 带宽 | 延迟 (周期) | 可见范围 | 典型用途 |
|---------|------|------|------------|---------|---------|
| 寄存器 | 256KB/SM | ~∞ | 0-1 | 单个线程 | 局部变量、累加器 |
| 共享内存 | 22-164KB/SM | 19 TB/s | ~20 | 线程块内 | 数据复用、临时缓冲 |
| L2 缓存 | 10-40MB | 6 TB/s | ~100 | 全设备 | 全局数据缓存 |
| HBM (全局内存) | 16-80GB | 1.5 TB/s | ~400 | 所有线程 | 大规模数据存储 |

### 23.1.3 带宽计算示例

```python
import triton
import triton.language as tl
import torch

@triton.jit
def bandwidth_test_kernel(
    input_ptr,      # 输入数据指针
    output_ptr,     # 输出数据指针
    n_elements,     # 元素总数
    BLOCK_SIZE: tl.constexpr,  # 块大小
):
    """带宽测试内核 —— 测量实际内存带宽"""
    # 计算当前线程处理的偏移量
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 掩码：防止越界访问
    mask = offsets < n_elements
    
    # 从全局内存加载数据 —— 这是带宽瓶颈所在
    data = tl.load(input_ptr + offsets, mask=mask)
    
    # 简单操作：加 1
    result = data + 1.0
    
    # 写回全局内存
    tl.store(output_ptr + offsets, result, mask=mask)


def measure_bandwidth():
    """测量实际带宽"""
    device = torch.device('cuda')
    n = 1 << 28  # 256M 元素
    x = torch.randn(n, device=device, dtype=torch.float32)
    y = torch.empty(n, device=device, dtype=torch.float32)
    
    BLOCK_SIZE = 1024
    grid = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # 预热
    bandwidth_test_kernel[grid](x, y, n, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()
    
    # 计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    bandwidth_test_kernel[grid](x, y, n, BLOCK_SIZE=BLOCK_SIZE)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end)
    # 读写各一次：2 * 256M * 4 bytes
    data_bytes = 2 * n * 4
    bandwidth_gb_s = (data_bytes / (1e9)) / (elapsed_ms / 1e3)
    
    print(f"耗时: {elapsed_ms:.2f} ms")
    print(f"数据量: {data_bytes / 1e9:.2f} GB")
    print(f"带宽: {bandwidth_gb_s:.2f} GB/s")
    print(f"HBM 理论峰值: 1.5 TB/s")
    print(f"带宽利用率: {bandwidth_gb_s / 1500 * 100:.1f}%")

# 运行测试
# measure_bandwidth()
```

### 23.1.4 延迟层次的直观理解

```
┌─────────────────────────────────────────────────────────────────┐
│                    延迟层次对比 (单位: 纳秒)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  寄存器 (1 ns)        ██                                         │
│  L1/SRAM (20 ns)      ████████████████████████                   │
│  L2 缓存 (100 ns)     ████████████████████████████████████████   │
│  HBM (400 ns)         ████████████████████████████████████████   │
│                       ════════════════════════════════════════   │
│  如果 HBM 延迟 = 1 天:                                          │
│    L2 缓存 = 6 小时                                             │
│    SRAM = 1.2 小时                                              │
│    寄存器 = 6 分钟                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**关键洞察**：
- HBM 延迟是寄存器的 **400 倍**
- L2 缓存比 HBM 快 **4 倍**
- SRAM 比 HBM 快 **20 倍**
- **每一次 HBM 访问都是一次"长途旅行"**

### 23.1.5 带宽 vs 延迟的区别

```python
# 带宽 (Bandwidth): 单位时间传输的数据量
# 延迟 (Latency): 单次访问的等待时间

# 带宽示例：
# 100 个线程同时读取 100 个 float → 带宽 = 100 * 4B / 周期
# 这是"吞吐量"问题

# 延迟示例：
# 1 个线程读取 1 个 float → 等待 ~400 个周期
# 这是"等待时间"问题

# Triton 的优化策略：
# 1. 批量加载 (Vectorized Load): 用向量加载隐藏延迟
# 2. 预取 (Prefetch): 提前加载下一块数据
# 3. 流水线 (Pipelining): 计算与加载重叠
```

---

## 23.2 内存合并（Coalescing）

### 23.2.1 什么是内存合并

内存合并（Coalescing）是 GPU 内存访问优化的核心概念。当一个 warp（32 个线程）的内存访问满足特定模式时，硬件可以将这些访问合并为一次或少数几次内存事务。

```
┌─────────────────────────────────────────────────────────────────┐
│                    内存合并示意图                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  情况 1: 合并访问 (Coalesced Access)                             │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐             │
│  │ T0  │ T1  │ T2  │ T3  │ ... │T29  │T30  │T31  │  线程        │
│  └──┬──┴──┬──┴──┬──┴──┬──┴─────┴──┬──┴──┬──┴──┬──┘             │
│     ↓     ↓     ↓     ↓          ↓     ↓     ↓                  │
│  ┌──┴──┬──┴──┬──┴──┬──┴──┬─────┬──┴──┬──┴──┬──┴──┐             │
│  │ 0B  │ 4B  │ 8B  │12B  │...  │120B │124B │128B │  地址        │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘             │
│                                                                 │
│  → 32 个线程访问连续地址                                          │
│  → 1 次 128B 事务完成                                            │
│  → 效率: 100%                                                    │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  情况 2: 未合并访问 (Uncoalesced Access)                         │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐             │
│  │ T0  │ T1  │ T2  │ T3  │ ... │T29  │T30  │T31  │  线程        │
│  └──┬──┴──┬──┴──┬──┴──┬──┴─────┴──┬──┴──┬──┴──┬──┘             │
│     ↓     ↓     ↓     ↓          ↓     ↓     ↓                  │
│  ┌──┴──┬──┴──┬──┴──┬──┴──┬─────┬──┴──┬──┴──┬──┴──┐             │
│  │ 0B  │ 32B │ 64B │ 96B │...  │960B │992B │1024B│  地址        │
│  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘             │
│                                                                 │
│  → 32 个线程访问间隔 32B 的地址                                   │
│  → 每个线程触发一次事务 (128B cache line)                         │
│  → 效率: 32/128 = 25%                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 23.2.2 128B Cache Line 机制

```python
# GPU 内存系统以 128 字节的 cache line 为单位传输数据
# 一个 warp 有 32 个线程
# 每个线程访问 4 字节 (float32)
# 32 * 4 = 128 字节 = 一个 cache line

# 合并条件：
# - 一个 warp 的所有线程访问的地址必须落在同一个 128B 对齐的区域内
# - 访问模式必须是连续的 (stride-1)

# 验证示例：
import torch

# 合并访问
x_coalesced = torch.arange(32, device='cuda')  # 连续地址
# T0 → addr 0, T1 → addr 4, ..., T31 → addr 124
# 全部落在 [0, 128) 范围内 ✓

# 未合并访问
x_uncoalesced = torch.arange(32, device='cuda') * 32  # stride-32
# T0 → addr 0, T1 → addr 128, T2 → addr 256, ...
# 每个线程触发独立的 cache line 访问 ✗
```

### 23.2.3 Triton 如何保证合并访问

```python
import triton
import triton.language as tl
import torch

@triton.jit
def vectorized_load_kernel(
    input_ptr,       # 输入指针
    output_ptr,      # 输出指针
    n_elements,      # 元素总数
    BLOCK_SIZE: tl.constexpr,  # 块大小
):
    """Triton 自动保证合并访问的内核"""
    pid = tl.program_id(axis=0)
    
    # 关键：offsets 是连续的 [0, 1, 2, ..., BLOCK_SIZE-1]
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # tl.load 会自动合并相邻线程的访问
    # 32 个线程 (一个 warp) 会访问连续的 128B 区域
    data = tl.load(input_ptr + offsets, mask=mask)
    
    # 计算
    result = data * 2.0 + 1.0
    
    # tl.store 也会自动合并
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def strided_load_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """分块加载：Triton 自动处理合并"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 行偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # 列偏移
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # 创建 2D 偏移量
    # offsets[i, j] = (offs_m[i], offs_n[j])
    offsets = offs_m[:, None] * n_cols + offs_n[None, :]
    
    # mask
    mask = (offs_m[:, None] < n_rows) & (offs_n[None, :] < n_cols)
    
    # 加载整个块
    # 在内存中，同一行的元素是连续的
    # 同一列的元素间隔 n_cols
    # Triton 会尽量合并行方向的访问
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # 处理
    result = data + 1.0
    
    tl.store(output_ptr + offsets, result, mask=mask)
```

### 23.2.4 未合并访问的识别与修复

```python
import triton
import triton.language as tl

# ❌ 错误示例：未合并访问
@triton.jit
def uncoalesced_kernel_bad(
    input_ptr,
    output_ptr,
    stride_input,  # 输入步长
    stride_output, # 输出步长
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """未合并访问示例"""
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    
    # 问题：stride_input 不是 1
    # 例如 stride_input = 32，每个线程访问的地址间隔 128B
    # 这会导致 32 次独立的内存事务
    data = tl.load(input_ptr + offsets * stride_input)
    
    tl.store(output_ptr + offsets * stride_output, data)


# ✅ 正确示例：合并访问
@triton.jit
def coalesced_kernel_good(
    input_ptr,
    output_ptr,
    stride_input,
    stride_output,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """合并访问示例"""
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    
    # 方案 1：如果可能，使用 stride-1 访问
    data = tl.load(input_ptr + offsets)
    
    # 方案 2：如果必须 strided，确保步长是 128B 的倍数
    # 128B / 4B(float32) = 32 个元素
    # 如果 stride 是 32 的倍数，可以合并
    
    tl.store(output_ptr + offsets, data)


# 📊 性能对比
def compare_coalesced_performance():
    """对比合并与未合并访问的性能"""
    import torch
    
    n = 1 << 26  # 64M 元素
    BLOCK_SIZE = 1024
    
    # 创建测试数据
    input_tensor = torch.randn(n, device='cuda', dtype=torch.float32)
    output_tensor = torch.empty(n, device='cuda', dtype=torch.float32)
    
    grid = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # 合并访问
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    coalesced_kernel_good[grid](
        input_tensor, output_tensor,
        1, 1, n, BLOCK_SIZE
    )
    end.record()
    torch.cuda.synchronize()
    coalesced_time = start.elapsed_time(end)
    
    print(f"合并访问: {coalesced_time:.3f} ms")
    print(f"带宽: {2 * n * 4 / 1e9 / (coalesced_time / 1e3):.2f} GB/s")

# compare_coalesced_performance()
```

### 23.2.5 合并访问的硬件视角

```
┌─────────────────────────────────────────────────────────────────┐
│                 L2 Cache Line 分配策略                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SM0 的 Warp 访问: [0, 128) bytes                               │
│  SM1 的 Warp 访问: [128, 256) bytes                             │
│  SM2 的 Warp 访问: [256, 384) bytes                             │
│  ...                                                            │
│                                                                 │
│  每个 warp 访问一个 128B cache line                              │
│  多个 SM 的 warp 可以同时访问不同的 cache line                    │
│  L2 缓存会缓存最近访问的 cache line                              │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  L2 Cache (共享)                                         │    │
│  │  ┌───────┬───────┬───────┬───────┬───────┬───────┐      │    │
│  │  │ 0-127 │128-255│256-383│ ...   │...    │...    │      │    │
│  │  │ [命中]│ [命中]│ [等待]│       │       │       │      │    │
│  │  └───────┴───────┴───────┴───────┴───────┴───────┘      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 23.3 共享内存优化与 Bank Conflict

### 23.3.1 共享内存架构

共享内存（Shared Memory）是 GPU 上最快的片上存储，位于每个 Streaming Multiprocessor (SM) 内部。

```
┌─────────────────────────────────────────────────────────────────┐
│                    共享内存架构                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  一个 SM 的共享内存组织:                                         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Shared Memory (22-164KB)                    │    │
│  ├─────────┬─────────┬─────────┬─────────┬─────────┬───────┤    │
│  │ Bank 0  │ Bank 1  │ Bank 2  │ Bank 3  │  ...    │Bank 31│    │
│  ├─────────┼─────────┼─────────┼─────────┼─────────┼───────┤    │
│  │ Byte 0  │ Byte 4  │ Byte 8  │ Byte 12 │  ...    │Byte 124│   │
│  │ Byte 128│ Byte 132│ Byte 136│ Byte 140│  ...    │Byte 252│   │
│  │ Byte 256│ Byte 260│ Byte 264│ Byte 268│  ...    │Byte 380│   │
│  │  ...     │  ...    │  ...    │  ...    │  ...    │  ...   │   │
│  └─────────┴─────────┴─────────┴─────────┴─────────┴───────┘    │
│                                                                 │
│  每个 Bank 宽度: 4 字节 (32 bits)                                │
│  总 Bank 数: 32 个                                               │
│  一个周期可以服务 32 个独立的 4B 访问                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 23.3.2 Bank Conflict 的成因

当多个线程在同一周期访问同一个 Bank 的不同地址时，会发生 Bank Conflict。硬件必须串行化这些访问，导致性能下降。

```python
import triton
import triton.language as tl

@triton.jit
def bank_conflict_example(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Bank Conflict 示例"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # 在共享内存中创建 2D 块
    # offsets[i, j] = offs_m[i] * n_cols + offs_n[j]
    
    # 情况 1: 没有 bank conflict (理想情况)
    # 如果 offs_n 是连续的，同一 warp 的线程访问不同的 bank
    data = tl.load(input_ptr + offs_m[:, None] * n_cols + offs_n[None, :])
    
    # 情况 2: 可能的 bank conflict
    # 如果 offs_m 不是连续的，或者 offs_m 的步长是 32 的倍数
    # 可能导致多个线程访问同一个 bank
    data_with_conflict = tl.load(
        input_ptr + offs_m[:, None] * 32 + offs_n[None, :]
    )
    
    # 输出
    tl.store(output_ptr + offs_m[:, None] * n_cols + offs_n[None, :], data)


@triton.jit
def no_bank_conflict_example(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """避免 Bank Conflict 的示例"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # 使用 swizzle 模式避免 bank conflict
    # 在计算地址时添加偏移，使访问分散到不同的 bank
    
    # 计算带 swizzle 的偏移
    # 对于 32 个 bank，每个 bank 4 字节
    # swizzle 后，不同行的访问会映射到不同的 bank
    offsets = offs_m[:, None] * n_cols + offs_n[None, :]
    
    # 添加列方向的 swizzle
    # 这会改变 bank 的映射，避免冲突
    swizzle_offsets = offsets + (offs_m[:, None] % 32)
    
    data = tl.load(input_ptr + swizzle_offsets)
    tl.store(output_ptr + offsets, data)
```

### 23.3.3 Bank Conflict 的类型

```
┌─────────────────────────────────────────────────────────────────┐
│                    Bank Conflict 类型                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  类型 1: 没有冲突 (No Conflict)                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  线程 0  → Bank 0                                       │    │
│  │  线程 1  → Bank 1                                       │    │
│  │  线程 2  → Bank 2                                       │    │
│  │  ...                                                    │    │
│  │  线程 31 → Bank 31                                      │    │
│  │  → 32 个独立访问，1 个周期完成                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  类型 2: 2-路冲突 (2-way Conflict)                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  线程 0  → Bank 0                                       │    │
│  │  线程 1  → Bank 0  ← 冲突!                              │    │
│  │  线程 2  → Bank 1                                       │    │
│  │  线程 3  → Bank 1  ← 冲突!                              │    │
│  │  ...                                                    │    │
│  │  → 2 个访问冲突，需要 2 个周期                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  类型 3: N-路冲突 (N-way Conflict)                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  所有 32 个线程都访问 Bank 0                             │    │
│  │  → 32 个访问冲突，需要 32 个周期                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 23.3.4 Swizzle 模式消除 Bank Conflict

```python
import triton
import triton.language as tl

@triton.jit
def swizzle_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """使用 Swizzle 模式消除 Bank Conflict"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Swizzle 模式的核心思想：
    # 对于 2D 数组 A[row][col]，bank = (row + col) % 32
    # 通过添加 col 的偏移，使得同一行的不同列访问不同的 bank
    
    # 计算原始偏移
    offsets = offs_m[:, None] * n_cols + offs_n[None, :]
    
    # Swizzle: 添加列索引的偏移
    # 这会改变 bank 的映射，避免同行访问同一 bank
    swizzled_offsets = offsets + offs_n[None, :] % 32
    
    data = tl.load(input_ptr + swizzled_offsets)
    
    # 输出时不使用 swizzle（保持数据完整性）
    tl.store(output_ptr + offsets, data)


@triton.jit
def padding_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """使用 Padding 消除 Bank Conflict"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Padding 的思想：
    # 在每行末尾添加 padding 元素，使每行长度变为 32 的倍数
    # 这样不同行的相同列会映射到不同的 bank
    
    # 计算 padding 后的偏移
    # 假设每行有 n_cols 个有效元素 + 1 个 padding 元素
    padded_n_cols = n_cols + 1  # 添加 1 个 padding
    
    # 计算偏移（带 padding）
    offsets = offs_m[:, None] * padded_n_cols + offs_n[None, :]
    
    # 加载数据
    data = tl.load(input_ptr + offsets)
    
    # 输出（不含 padding）
    output_offsets = offs_m[:, None] * n_cols + offs_n[None, :]
    tl.store(output_ptr + output_offsets, data)


@triton.jit
def advanced_swizzle_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """高级 Swizzle 模式：分组 Swizzle"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # 分组 Swizzle: 
    # 将 32 个 bank 分成 4 组，每组 8 个 bank
    # 每组内的访问使用不同的 swizzle 策略
    
    # 计算 bank 索引
    # bank = (row * n_cols + col) % 32
    # 为了消除冲突，我们重新映射 bank
    
    # 原始偏移
    offsets = offs_m[:, None] * n_cols + offs_n[None, :]
    
    # 分组 swizzle
    # 对于每个线程，计算其在 warp 中的线程 ID
    # 然后根据线程 ID 应用不同的 swizzle
    thread_id = tl.arange(0, BLOCK_M)[:, None]  # 简化：假设 BLOCK_M = warp size
    
    # 应用 swizzle
    # swizzle = (row + (col >> 3)) & 3  # 分组 swizzle
    swizzle = (offs_m[:, None] + (offs_n[None, :] >> 3)) & 3
    
    swizzled_offsets = offsets + swizzle * 8  # 每组 8 个 bank
    
    data = tl.load(input_ptr + swizzled_offsets)
    tl.store(output_ptr + offsets, data)
```

### 23.3.5 Bank Conflict 检测与分析

```python
# 检测 Bank Conflict 的方法

import triton
import triton.language as tl
import torch

@triton.jit
def detect_bank_conflict(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """检测是否有 bank conflict 的内核"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # 计算每个线程访问的 bank
    # bank = (offset / 4) % 32
    # 对于 2D 数组：bank = (row * n_cols + col) % 32
    
    # 检查是否有多个线程访问同一个 bank
    # 如果有，说明存在 bank conflict
    
    # 简单的检测方法：
    # 统计每个 bank 被访问的次数
    # 如果最大访问次数 > 1，则存在 bank conflict
    
    offsets = offs_m[:, None] * n_cols + offs_n[None, :]
    banks = (offsets // 4) % 32
    
    # 在 Triton 中，我们无法直接统计 bank 访问次数
    # 但可以通过性能分析工具检测
    
    # 加载数据
    data = tl.load(input_ptr + offsets)
    tl.store(output_ptr + offsets, data)


def analyze_bank_conflict():
    """分析 bank conflict"""
    n_rows = 1024
    n_cols = 1024
    
    input_tensor = torch.randn(n_rows, n_cols, device='cuda', dtype=torch.float32)
    output_tensor = torch.empty(n_rows, n_cols, device='cuda', dtype=torch.float32)
    
    BLOCK_M = 32  # warp size
    BLOCK_N = 32
    
    grid = (n_rows // BLOCK_M, n_cols // BLOCK_N)
    
    # 运行内核
    detect_bank_conflict[grid](
        input_tensor, output_tensor,
        n_rows, n_cols,
        BLOCK_M, BLOCK_N
    )
    
    # 使用 NVIDIA Nsight Compute 分析 bank conflict
    # ncu --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_conflict \
    #     python -c "import torch; ..."

# analyze_bank_conflict()
```

### 23.3.6 共享内存优化策略总结

```
┌─────────────────────────────────────────────────────────────────┐
│                共享内存优化策略总结                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  策略 1: Padding                                               │
│  ─────────────────                                              │
│  原理: 在每行末尾添加 padding 元素                               │
│  效果: 使每行长度变为 32 的倍数，避免 bank conflict               │
│  代价: 浪费少量共享内存 (每个线程块浪费 1 个元素/行)               │
│  适用: 2D 数组访问，行方向 stride-1                              │
│                                                                 │
│  策略 2: Swizzle                                               │
│  ─────────────────                                              │
│  原理: 在计算地址时添加偏移，改变 bank 映射                       │
│  效果: 不浪费内存，但增加计算复杂度                               │
│  代价: 额外的地址计算开销                                        │
│  适用: 对内存利用率要求高的场景                                  │
│                                                                 │
│  策略 3: 分组 Swizzle                                           │
│  ─────────────────                                              │
│  原理: 将 bank 分组，每组使用不同的 swizzle 策略                  │
│  效果: 更精细的 bank 分配，减少冲突                              │
│  代价: 复杂度最高                                               │
│  适用: 高度优化的场景，如 FlashAttention                        │
│                                                                 │
│  策略 4: 转置访问                                              │
│  ─────────────────                                              │
│  原理: 改变访问顺序，使访问模式变为 stride-1                     │
│  效果: 自然避免 bank conflict                                    │
│  代价: 可能增加全局内存访问                                      │
│  适用: 可以改变访问顺序的场景                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 23.4 寄存器分配策略

### 23.4.1 寄存器的重要性

寄存器是 GPU 上最快的存储，但容量有限。Triton 编译器会自动管理寄存器分配，但理解其工作原理有助于写出更高效的代码。

```
┌─────────────────────────────────────────────────────────────────┐
│                    寄存器分配层次                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  一个 SM 的寄存器:                                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              寄存器文件 (256KB)                           │    │
│  │  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐      │    │
│  │  │ T0  │ T1  │ T2  │ T3  │ ... │T30  │T31  │ ... │      │    │
│  │  │ 64R │ 64R │ 64R │ 64R │     │ 64R │ 64R │     │      │    │
│  │  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘      │    │
│  │                                                          │    │
│  │  每个线程最多: 255 个寄存器                                │    │
│  │  每个 SM: 65536 个寄存器                                  │    │
│  │  可同时运行的线程数 = 65536 / 每线程寄存器数               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  寄存器分配决策:                                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  如果每个线程用 32 个寄存器:                              │    │
│  │    每个 SM 可运行 65536 / 32 = 2048 个线程               │    │
│  │    = 2048 / 32 = 64 个 warp                              │    │
│  │                                                          │    │
│  │  如果每个线程用 64 个寄存器:                              │    │
│  │    每个 SM 可运行 65536 / 64 = 1024 个线程               │    │
│  │    = 1024 / 32 = 32 个 warp                              │    │
│  │                                                          │    │
│  │  寄存器使用越多 → 并行度越低 → 延迟隐藏能力越差           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 23.4.2 Triton 编译器的寄存器分配

```python
import triton
import triton.language as tl

@triton.jit
def register_heavy_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """寄存器使用较多的内核"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 加载多个数据块
    # 每个块需要寄存器存储
    data1 = tl.load(input_ptr + offsets, mask=mask)
    data2 = tl.load(input_ptr + offsets + n_elements, mask=mask)
    data3 = tl.load(input_ptr + offsets + 2 * n_elements, mask=mask)
    data4 = tl.load(input_ptr + offsets + 3 * n_elements, mask=mask)
    
    # 复杂计算
    # 每个中间结果都需要寄存器
    temp1 = data1 * 2.0 + data2
    temp2 = data3 * 3.0 + data4
    temp3 = temp1 + temp2
    temp4 = temp1 * temp2
    temp5 = temp3 + temp4
    temp6 = temp5 * temp3
    
    # 输出
    tl.store(output_ptr + offsets, temp6, mask=mask)


@triton.jit
def register_light_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """寄存器使用较少的内核"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 分步计算，减少同时使用的寄存器
    # 第一步：加载和初步处理
    data = tl.load(input_ptr + offsets, mask=mask)
    result = data * 2.0
    
    # 第二步：继续处理
    result = result + 1.0
    
    # 第三步：最终处理
    result = result * result
    
    # 输出
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def register_optimized_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """寄存器优化的内核"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 使用循环减少寄存器使用
    # Triton 编译器会优化循环，减少寄存器压力
    
    # 初始化累加器
    accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # 循环处理
    for i in range(4):
        data = tl.load(input_ptr + offsets + i * n_elements, mask=mask)
        accumulator += data
    
    # 输出
    tl.store(output_ptr + offsets, accumulator, mask=mask)
```

### 23.4.3 寄存器溢出（Spilling）

```python
import triton
import triton.language as tl

@triton.jit
def spilling_example(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """寄存器溢出示例"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 当寄存器不足时，编译器会将部分变量"溢出"到 L1/共享内存
    # 这会显著降低性能
    
    # 高寄存器压力的代码
    # 大量的临时变量
    data1 = tl.load(input_ptr + offsets, mask=mask)
    data2 = tl.load(input_ptr + offsets + 1, mask=mask)
    data3 = tl.load(input_ptr + offsets + 2, mask=mask)
    data4 = tl.load(input_ptr + offsets + 3, mask=mask)
    data5 = tl.load(input_ptr + offsets + 4, mask=mask)
    data6 = tl.load(input_ptr + offsets + 5, mask=mask)
    data7 = tl.load(input_ptr + offsets + 6, mask=mask)
    data8 = tl.load(input_ptr + offsets + 7, mask=mask)
    
    # 复杂的计算链
    temp1 = data1 * data2 + data3
    temp2 = data4 * data5 + data6
    temp3 = data7 * data8 + data1
    temp4 = temp1 * temp2 + temp3
    temp5 = temp1 + temp2 + temp3 + temp4
    temp6 = temp4 * temp5 + temp1
    temp7 = temp5 * temp6 + temp2
    temp8 = temp6 * temp7 + temp3
    
    # 最终结果
    result = temp4 + temp5 + temp6 + temp7 + temp8
    
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def spilling_avoided_example(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """避免寄存器溢出的示例"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 策略：分块处理，减少同时使用的寄存器
    
    # 第一块
    data1 = tl.load(input_ptr + offsets, mask=mask)
    data2 = tl.load(input_ptr + offsets + 1, mask=mask)
    temp1 = data1 * data2
    
    # 第二块（复用 data1, data2 的寄存器）
    data3 = tl.load(input_ptr + offsets + 2, mask=mask)
    data4 = tl.load(input_ptr + offsets + 3, mask=mask)
    temp2 = data3 * data4
    
    # 继续处理
    temp3 = temp1 + temp2
    
    # 第三块（复用之前的寄存器）
    data5 = tl.load(input_ptr + offsets + 4, mask=mask)
    data6 = tl.load(input_ptr + offsets + 5, mask=mask)
    temp4 = data5 * data6
    
    # 最终结果
    result = temp3 + temp4
    
    tl.store(output_ptr + offsets, result, mask=mask)


# 使用 maxnreg 控制寄存器使用
@triton.autotune(
    configs=[
        triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4, num_stages=3),
        triton.Config(kwargs={'BLOCK_SIZE': 256}, num_warps=8, num_stages=3),
    ],
    key=['n_elements'],
)
@triton.jit
def maxnreg_controlled_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """使用 maxnreg 控制寄存器使用"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 加载数据
    data = tl.load(input_ptr + offsets, mask=mask)
    
    # 计算
    result = data * 2.0 + 1.0
    
    # 输出
    tl.store(output_ptr + offsets, result, mask=mask)


# 编译时检查寄存器使用
def check_register_usage():
    """检查寄存器使用情况"""
    n_elements = 1 << 20
    BLOCK_SIZE = 1024
    
    # 编译内核
    kernel = register_heavy_kernel
    
    # 查看编译后的 PTX
    # 可以使用 triton.tools 工具分析寄存器使用
    
    print("寄存器使用分析:")
    print("1. 使用 NVIDIA Nsight Compute 查看寄存器分配")
    print("2. 使用 triton.tools 查看编译后的 PTX")
    print("3. 关注 'regs' 指标")

# check_register_usage()
```

### 23.4.4 寄存器优化技巧

```python
import triton
import triton.language as tl

@triton.jit
def register_optimization_tips(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """寄存器优化技巧示例"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 技巧 1: 重用变量，减少同时存活的变量数
    data = tl.load(input_ptr + offsets, mask=mask)
    data = data * 2.0  # 重用 data 变量
    data = data + 1.0  # 继续重用
    
    # 技巧 2: 使用 tl.where 避免条件分支
    # 条件分支可能导致寄存器泄漏
    result = tl.where(data > 0, data, -data)
    
    # 技巧 3: 避免不必要的中间变量
    # 不好的写法:
    # temp1 = data * 2.0
    # temp2 = temp1 + 1.0
    # temp3 = temp2 * temp2
    # result = temp3
    
    # 好的写法:
    result = (data * 2.0 + 1.0) ** 2
    
    # 技巧 4: 使用向量化操作
    # 一次加载多个元素，减少循环次数
    # data = tl.load(input_ptr + offsets)  # 加载 BLOCK_SIZE 个元素
    
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def loop_register_optimization(
    input_ptr,
    output_ptr,
    n_elements,
    n_iterations: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """循环中的寄存器优化"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 使用累加器模式，减少寄存器压力
    accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for i in range(n_iterations):
        data = tl.load(input_ptr + offsets + i * n_elements, mask=mask)
        accumulator += data
    
    # 输出
    tl.store(output_ptr + offsets, accumulator, mask=mask)


@triton.jit
def function_inlining_optimization(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """函数内联优化"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Triton 会自动内联小函数
    # 这有助于减少寄存器使用和函数调用开销
    
    # 定义辅助函数
    @tl.static_range(4)  # 静态循环，编译器可以优化
    def process_chunk(chunk_id):
        chunk_offsets = offsets + chunk_id * BLOCK_SIZE
        chunk_mask = chunk_offsets < n_elements
        return tl.load(input_ptr + chunk_offsets, mask=chunk_mask)
    
    # 使用辅助函数
    results = []
    for i in range(4):
        results.append(process_chunk(i))
    
    # 合并结果
    final_result = results[0] + results[1] + results[2] + results[3]
    
    tl.store(output_ptr + offsets, final_result, mask=mask)
```

---

## 23.5 数据复用与 Tiling

### 23.5.1 Tiling 的基本概念

Tiling（分块）是将大矩阵分解成小块进行处理的技术，可以显著提高数据复用率和内存效率。

```
┌─────────────────────────────────────────────────────────────────┐
│                    Tiling 分块示意图                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  原始矩阵 (1024 x 1024):                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐      │    │
│  │  │ 0,0 │ 0,1 │ 0,2 │ 0,3 │ 0,4 │ 0,5 │ 0,6 │ 0,7 │      │    │
│  │  ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤      │    │
│  │  │ 1,0 │ 1,1 │ 1,2 │ 1,3 │ 1,4 │ 1,5 │ 1,6 │ 1,7 │      │    │
│  │  ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤      │    │
│  │  │ 2,0 │ 2,1 │ 2,2 │ 2,3 │ 2,4 │ 2,5 │ 2,6 │ 2,7 │      │    │
│  │  ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤      │    │
│  │  │ ... │ ... │ ... │ ... │ ... │ ... │ ... │ ... │      │    │
│  │  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  Tiling 后 (块大小 256 x 256):                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  ┌─────────┬─────────┬─────────┬─────────┐              │    │
│  │  │ Block 0 │ Block 1 │ Block 2 │ Block 3 │              │    │
│  │  │ 256x256 │ 256x256 │ 256x256 │ 256x256 │              │    │
│  │  ├─────────┼─────────┼─────────┼─────────┤              │    │
│  │  │ Block 4 │ Block 5 │ Block 6 │ Block 7 │              │    │
│  │  │ 256x256 │ 256x256 │ 256x256 │ 256x256 │              │    │
│  │  ├─────────┼─────────┼─────────┼─────────┤              │    │
│  │  │ Block 8 │ Block 9 │ Block10 │ Block11 │              │    │
│  │  │ 256x256 │ 256x256 │ 256x256 │ 256x256 │              │    │
│  │  ├─────────┼─────────┼─────────┼─────────┤              │    │
│  │  │ Block12 │ Block13 │ Block14 │ Block15 │              │    │
│  │  │ 256x256 │ 256x256 │ 256x256 │ 256x256 │              │    │
│  │  └─────────┴─────────┴─────────┴─────────┘              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 23.5.2 Triton 中的 Tiling 实现

```python
import triton
import triton.language as tl
import torch

@triton.jit
def tiled_vector_add(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """简单的 Tiling 向量加法"""
    pid = tl.program_id(axis=0)
    
    # 计算当前块的偏移量
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 掩码：防止越界
    mask = offsets < n_elements
    
    # 加载数据块
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    
    # 计算
    c = a + b
    
    # 存储结果
    tl.store(c_ptr + offsets, c, mask=mask)


@triton.jit
def tiled_matrix_multiply(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_an,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Tiling 矩阵乘法：C = A × B"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 计算当前块的偏移量
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # 初始化累加器
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # 沿 K 维度分块循环
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # 计算当前 K 块的偏移
        offs_k_cur = k * BLOCK_K + offs_k
        
        # 掩码
        a_mask = (offs_am[:, None] < M) & (offs_k_cur[None, :] < K)
        b_mask = (offs_k_cur[:, None] < K) & (offs_bn[None, :] < N)
        
        # 加载 A 和 B 的块
        a = tl.load(
            a_ptr + offs_am[:, None] * stride_am + offs_k_cur[None, :] * stride_an,
            mask=a_mask, other=0.0
        )
        b = tl.load(
            b_ptr + offs_k_cur[:, None] * stride_bn + offs_bn[None, :] * stride_bk,
            mask=b_mask, other=0.0
        )
        
        # 累加
        accumulator += tl.dot(a, b)
    
    # 写回结果
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(
        c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn,
        accumulator, mask=c_mask
    )


def tiled_matmul_example():
    """Tiling 矩阵乘法示例"""
    M, N, K = 1024, 1024, 1024
    a = torch.randn(M, K, device='cuda', dtype=torch.float32)
    b = torch.randn(K, N, device='cuda', dtype=torch.float32)
    c = torch.empty(M, N, device='cuda', dtype=torch.float32)
    
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    grid = (tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N))
    
    tiled_matrix_multiply[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    # 验证结果
    expected = a @ b
    print(f"最大误差: {torch.max(torch.abs(c - expected)).item():.6f}")

# tiled_matmul_example()
```

### 23.5.3 Shared Memory 作为数据复用缓冲

```python
import triton
import triton.language as tl

@triton.jit
def shared_memory_reuse_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_an,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """使用共享内存进行数据复用的矩阵乘法"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 计算偏移量
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # 初始化累加器
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # 使用共享内存进行数据复用
    # 在 Triton 中，编译器会自动优化数据复用
    # 但我们可以显式地控制分块大小
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k_cur = k * BLOCK_K + offs_k
        
        # 加载 A 的块
        a_mask = (offs_am[:, None] < M) & (offs_k_cur[None, :] < K)
        a_block = tl.load(
            a_ptr + offs_am[:, None] * stride_am + offs_k_cur[None, :] * stride_an,
            mask=a_mask, other=0.0
        )
        
        # 加载 B 的块
        b_mask = (offs_k_cur[:, None] < K) & (offs_bn[None, :] < N)
        b_block = tl.load(
            b_ptr + offs_k_cur[:, None] * stride_bn + offs_bn[None, :] * stride_bk,
            mask=b_mask, other=0.0
        )
        
        # 数据复用：a_block 和 b_block 在当前迭代中被多次使用
        # 通过分块，减少了全局内存访问次数
        
        # 累加
        accumulator += tl.dot(a_block, b_block)
    
    # 写回结果
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(
        c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn,
        accumulator, mask=c_mask
    )


@triton.jit
def optimized_reuse_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """优化的数据复用示例"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 使用更大的块来提高数据复用率
    # 但要注意寄存器和共享内存限制
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # 初始化累加器
    accumulator = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # 沿 K 维度循环
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k_cur = k * BLOCK_K + offs_k
        
        # 加载 A 的块
        a_mask = (offs_m[:, None] < M) & (offs_k_cur[None, :] < K)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k_cur[None, :],
                     mask=a_mask, other=0.0)
        
        # 加载 B 的块
        b_mask = (offs_k_cur[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptr + offs_k_cur[:, None] * N + offs_n[None, :],
                     mask=b_mask, other=0.0)
        
        # 使用 tl.dot 进行矩阵乘法
        # Triton 会自动使用共享内存进行数据复用
        accumulator += tl.dot(a, b)
    
    # 写回结果
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :],
             accumulator, mask=c_mask)
```

### 23.5.4 Tiling 策略选择

```python
# Tiling 策略选择指南

# 策略 1: 固定块大小
# 优点：简单，易于实现
# 缺点：可能不是最优的
BLOCK_SIZE = 1024

# 策略 2: 自适应块大小
# 优点：根据数据大小自动调整
# 缺点：需要额外的计算
def adaptive_block_size(n_elements):
    if n_elements < 1024:
        return 128
    elif n_elements < 10240:
        return 256
    elif n_elements < 102400:
        return 512
    else:
        return 1024

# 策略 3: 2D Tiling
# 适用于矩阵操作
# 优点：更好地利用数据局部性
# 缺点：实现复杂
BLOCK_M = 128
BLOCK_N = 128

# 策略 4: 3D Tiling
# 适用于 3D 数据或需要额外维度的操作
# 优点：最大化数据复用
# 缺点：寄存器压力大
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32


@triton.jit
def adaptive_tiling_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """自适应 Tiling 内核"""
    pid = tl.program_id(0)
    
    # 根据 BLOCK_SIZE 自动调整偏移量
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 加载数据
    data = tl.load(input_ptr + offsets, mask=mask)
    
    # 处理
    result = data * 2.0
    
    # 输出
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def two_d_tiling_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """2D Tiling 矩阵乘法"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    accumulator = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k_cur = k * BLOCK_K + offs_k
        
        # 加载 A 和 B 的块
        a_mask = (offs_m[:, None] < M) & (offs_k_cur[None, :] < K)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k_cur[None, :],
                     mask=a_mask, other=0.0)
        
        b_mask = (offs_k_cur[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptr + offs_k_cur[:, None] * N + offs_n[None, :],
                     mask=b_mask, other=0.0)
        
        accumulator += tl.dot(a, b)
    
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :],
             accumulator, mask=c_mask)
```

---

## 23.6 Prefetch 预取优化

### 23.6.1 L2 Cache 预取

L2 缓存预取是通过提前将数据从 HBM 加载到 L2 缓存来隐藏内存延迟的技术。

```python
import triton
import triton.language as tl

@triton.jit
def l2_prefetch_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """L2 Cache 预取内核"""
    pid = tl.program_id(0)
    
    # 计算当前块和预取块的偏移量
    current_offset = pid * BLOCK_SIZE
    prefetch_offset = (pid + 1) * BLOCK_SIZE
    
    # 加载当前块
    offsets = current_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask)
    
    # 预取下一个块到 L2 缓存
    # 这不会立即使用数据，但会将其加载到 L2
    prefetch_mask = prefetch_offset + tl.arange(0, BLOCK_SIZE) < n_elements
    tl.load(input_ptr + prefetch_offset + tl.arange(0, BLOCK_SIZE),
            mask=prefetch_mask, cache='l2')
    
    # 处理当前块
    result = data * 2.0
    
    # 输出
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def multi_stage_prefetch_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """多阶段预取内核"""
    pid = tl.program_id(0)
    
    # 预取多个块
    for stage in range(NUM_STAGES):
        prefetch_offset = (pid + stage) * BLOCK_SIZE
        prefetch_mask = prefetch_offset + tl.arange(0, BLOCK_SIZE) < n_elements
        
        # 预取到 L2
        tl.load(input_ptr + prefetch_offset + tl.arange(0, BLOCK_SIZE),
                mask=prefetch_mask, cache='l2')
    
    # 加载当前块
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask)
    
    # 处理
    result = data * 2.0
    
    # 输出
    tl.store(output_ptr + offsets, result, mask=mask)
```

### 23.6.2 cp.async 预取

cp.async 是 NVIDIA GPU 提供的异步复制指令，可以将数据从全局内存直接复制到共享内存，同时允许计算继续进行。

```python
import triton
import triton.language as tl

@triton.jit
def cp_async_example(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """cp.async 预取示例"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 使用 tl.load 的异步模式
    # 在 Triton 中，可以通过设置属性来启用异步加载
    
    # 方法 1: 使用 cache='l2' 预取到 L2
    data = tl.load(input_ptr + offsets, mask=mask, cache='l2')
    
    # 方法 2: 使用 prefetch 属性
    # data = tl.load(input_ptr + offsets, mask=mask, prefetch=True)
    
    # 处理
    result = data * 2.0
    
    # 输出
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def async_pipeline_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """异步流水线内核"""
    pid = tl.program_id(0)
    
    # 计算所有需要的偏移量
    all_offsets = []
    for stage in range(NUM_STAGES + 1):
        offset = (pid + stage) * BLOCK_SIZE
        if offset < n_elements:
            all_offsets.append(offset)
    
    # 预取后续块
    for i, offset in enumerate(all_offsets[1:]):
        prefetch_offsets = offset + tl.arange(0, BLOCK_SIZE)
        prefetch_mask = prefetch_offsets < n_elements
        tl.load(input_ptr + prefetch_offsets, mask=prefetch_mask, cache='l2')
    
    # 加载当前块
    current_offsets = all_offsets[0] + tl.arange(0, BLOCK_SIZE)
    current_mask = current_offsets < n_elements
    data = tl.load(input_ptr + current_offsets, mask=current_mask)
    
    # 处理
    result = data * 2.0
    
    # 输出
    tl.store(output_ptr + current_offsets, result, mask=current_mask)
```

### 23.6.3 num_stages 与预取的关系

```python
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # 不同的 num_stages 配置
        triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4, num_stages=1),
        triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4, num_stages=3),
        triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4, num_stages=4),
        triton.Config(kwargs={'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
        triton.Config(kwargs={'BLOCK_SIZE': 256}, num_warps=8, num_stages=3),
    ],
    key=['n_elements'],
)
@triton.jit
def num_stages_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """num_stages 配置示例"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 加载数据
    data = tl.load(input_ptr + offsets, mask=mask)
    
    # 处理
    result = data * 2.0
    
    # 输出
    tl.store(output_ptr + offsets, result, mask=mask)


def analyze_num_stages():
    """分析 num_stages 的影响"""
    n_elements = 1 << 24  # 16M 元素
    
    # 不同的 num_stages 配置
    configs = [
        {'num_stages': 1, 'name': '无预取'},
        {'num_stages': 2, 'name': '2 阶段预取'},
        {'num_stages': 3, 'name': '3 阶段预取'},
        {'num_stages': 4, 'name': '4 阶段预取'},
    ]
    
    for config in configs:
        print(f"\n配置: {config['name']}")
        print(f"  num_stages: {config['num_stages']}")
        print(f"  预取距离: {config['num_stages']} 个块")
        print(f"  隐藏延迟: {config['num_stages']} * BLOCK_SIZE * 4 bytes")
        
        # 性能分析
        # 实际性能取决于：
        # 1. 数据访问模式
        # 2. 计算与内存的比例
        # 3. 共享内存使用量
        # 4. 寄存器使用量

# analyze_num_stages()
```

### 23.6.4 预取策略总结

```
┌─────────────────────────────────────────────────────────────────┐
│                    预取策略总结                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  策略 1: L2 预取                                               │
│  ─────────────────                                              │
│  原理: 提前将数据加载到 L2 缓存                                 │
│  优点: 减少 HBM 访问延迟                                        │
│  缺点: 增加 L2 缓存压力                                         │
│  适用: 数据访问模式可预测的场景                                  │
│                                                                 │
│  策略 2: cp.async 预取                                          │
│  ─────────────────                                              │
│  原理: 使用异步复制指令，计算与加载重叠                          │
│  优点: 最大化计算/内存重叠                                       │
│  缺点: 需要硬件支持                                             │
│  适用: 高带宽需求的场景                                         │
│                                                                 │
│  策略 3: 多阶段预取                                             │
│  ─────────────────                                              │
│  原理: 预取多个块，增加预取距离                                 │
│  优点: 更好地隐藏延迟                                           │
│  缺点: 增加缓存压力和寄存器使用                                 │
│  适用: 高延迟场景                                               │
│                                                                 │
│  策略 4: 自适应预取                                             │
│  ─────────────────                                              │
│  原理: 根据数据访问模式动态调整预取策略                          │
│  优点: 适应不同场景                                             │
│  缺点: 实现复杂                                                 │
│  适用: 数据访问模式变化大的场景                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 23.7 Eviction Policy（淘汰策略）

### 23.7.1 L2 Cache 的淘汰策略

L2 缓存使用淘汰策略来管理有限的缓存空间。理解这些策略有助于设计对 L2 友好的访问模式。

```python
import triton
import triton.language as tl

# evict_first: 优先淘汰最近最少使用的数据
# evict_last: 优先淘汰最近最多使用的数据

@triton.jit
def l2_friendly_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """L2 友好的访问模式"""
    pid = tl.program_id(0)
    
    # 顺序访问：对 L2 友好
    # 每个块访问连续的内存区域
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 顺序加载
    data = tl.load(input_ptr + offsets, mask=mask)
    
    # 处理
    result = data * 2.0
    
    # 顺序存储
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def l2_unfriendly_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """L2 不友好的访问模式"""
    pid = tl.program_id(0)
    
    # 随机访问：对 L2 不友好
    # 使用随机偏移量
    import random
    random_offsets = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
    
    # 伪随机打乱偏移量
    # 注意：这只是一个示意，实际的随机访问会导致 L2 命中率下降
    shuffled_offsets = (random_offsets * 7 + 13) % n_elements
    
    mask = shuffled_offsets < n_elements
    
    # 随机加载
    data = tl.load(input_ptr + shuffled_offsets, mask=mask)
    
    # 处理
    result = data * 2.0
    
    # 随机存储
    tl.store(output_ptr + shuffled_offsets, result, mask=mask)
```

### 23.7.2 设计 L2 友好的访问模式

```python
import triton
import triton.language as tl
import torch

@triton.jit
def l2_optimized_kernel(
    input_ptr,
    output_ptr,
    n_rows,
    n_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """L2 优化的访问模式"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 计算偏移量
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # L2 友好的访问模式：
    # 1. 顺序访问：先访问行，再访问列
    # 2. 空间局部性：访问相邻的内存位置
    # 3. 时间局部性：重复访问最近使用的数据
    
    # 顺序加载（对 L2 友好）
    offsets = offs_m[:, None] * n_cols + offs_n[None, :]
    mask = (offs_m[:, None] < n_rows) & (offs_n[None, :] < n_cols)
    
    # 加载数据
    data = tl.load(input_ptr + offsets, mask=mask)
    
    # 处理
    result = data * 2.0
    
    # 顺序存储
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def blocked_access_pattern(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """分块访问模式：对 L2 友好"""
    pid = tl.program_id(0)
    
    # 计算当前组的起始位置
    group_start = pid * GROUP_SIZE
    
    # 在组内顺序访问
    for i in range(GROUP_SIZE):
        offset = group_start + i * BLOCK_SIZE
        if offset < n_elements:
            # 顺序访问
            offsets = offset + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            
            # 加载
            data = tl.load(input_ptr + offsets, mask=mask)
            
            # 处理
            result = data * 2.0
            
            # 存储
            tl.store(output_ptr + offsets, result, mask=mask)


def analyze_l2_efficiency():
    """分析 L2 效率"""
    n_elements = 1 << 24  # 16M 元素
    
    # 测试不同的访问模式
    patterns = [
        {'name': '顺序访问', 'stride': 1},
        {'name': '步长-2 访问', 'stride': 2},
        {'name': '步长-4 访问', 'stride': 4},
        {'name': '步长-32 访问', 'stride': 32},
        {'name': '随机访问', 'stride': 'random'},
    ]
    
    for pattern in patterns:
        print(f"\n访问模式: {pattern['name']}")
        print(f"  步长: {pattern['stride']}")
        
        if pattern['stride'] == 1:
            print(f"  L2 命中率: 高 (顺序访问)")
            print(f"  带宽利用率: 高")
        elif pattern['stride'] == 'random':
            print(f"  L2 命中率: 低 (随机访问)")
            print(f"  带宽利用率: 低")
        else:
            print(f"  L2 命中率: 中等")
            print(f"  带宽利用率: 中等")

# analyze_l2_efficiency()
```

### 23.7.3 淘汰策略对性能的影响

```
┌─────────────────────────────────────────────────────────────────┐
│                 淘汰策略对性能的影响                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  场景 1: 顺序访问 (evict_first 策略)                            │
│  ─────────────────────────────────                              │
│  访问模式: 0, 1, 2, 3, ..., N                                   │
│  淘汰策略: evict_first (淘汰最早访问的数据)                      │
│  L2 命中率: 高 (数据在被淘汰前被多次访问)                        │
│  性能: ⚡⚡⚡⚡⚡ (优秀)                                          │
│                                                                 │
│  场景 2: 随机访问 (evict_first 策略)                            │
│  ─────────────────────────────────                              │
│  访问模式: random(0, N)                                         │
│  淘汰策略: evict_first                                          │
│  L2 命中率: 低 (数据可能在被淘汰后才被再次访问)                   │
│  性能: ⚡ (差)                                                   │
│                                                                 │
│  场景 3: 重复访问 (evict_first 策略)                            │
│  ─────────────────────────────────                              │
│  访问模式: 0, 1, 2, 0, 1, 2, ...                                │
│  淘汰策略: evict_first                                          │
│  L2 命中率: 中等 (取决于重复频率)                                │
│  性能: ⚡⚡⚡ (中等)                                              │
│                                                                 │
│  场景 4: 大工作集 (evict_last 策略)                             │
│  ─────────────────────────────────                              │
│  访问模式: 顺序访问超过 L2 容量                                  │
│  淘汰策略: evict_last (淘汰最近最多使用的数据)                   │
│  L2 命中率: 低 (所有数据都被淘汰)                                │
│  性能: ⚡ (差)                                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 23.8 优化 Checklist

### 23.8.1 内存优化检查清单

```
┌─────────────────────────────────────────────────────────────────┐
│                    内存优化检查清单                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ☐ 1. 内存合并检查                                              │
│     ─────────────────                                           │
│     ☐ 所有线程是否访问连续地址？                                 │
│     ☐ 步长是否为 1（stride-1）？                                │
│     ☐ 是否避免了跨步访问（strided access）？                     │
│     ☐ 128B cache line 是否被充分利用？                          │
│                                                                 │
│  ☐ 2. 共享内存检查                                              │
│     ─────────────────                                           │
│     ☐ 是否有 bank conflict？                                    │
│     ☐ 是否使用了 padding 或 swizzle？                           │
│     ☐ 共享内存使用量是否在限制内？                               │
│     ☐ 是否最大化了数据复用？                                    │
│                                                                 │
│  ☐ 3. 寄存器检查                                                │
│     ─────────────────                                           │
│     ☐ 是否避免了寄存器溢出（spilling）？                         │
│     ☐ 是否重用了变量以减少寄存器压力？                           │
│     ☐ 是否使用了循环展开？                                      │
│     ☐ 是否使用了 tl.where 避免条件分支？                        │
│                                                                 │
│  ☐ 4. Tiling 检查                                               │
│     ─────────────────                                           │
│     ☐ 块大小是否合适？                                          │
│     ☐ 是否最大化了数据局部性？                                  │
│     ☐ 是否考虑了硬件限制？                                      │
│     ☐ 是否使用了自适应块大小？                                  │
│                                                                 │
│  ☐ 5. 预取检查                                                  │
│     ─────────────────                                           │
│     ☐ 是否启用了 L2 预取？                                      │
│     ☐ num_stages 是否合适？                                     │
│     ☐ 是否计算与内存重叠？                                      │
│     ☐ 预取距离是否合适？                                        │
│                                                                 │
│  ☐ 6. 淘汰策略检查                                              │
│     ─────────────────                                           │
│     ☐ 访问模式是否对 L2 友好？                                  │
│     ☐ 是否避免了随机访问？                                      │
│     ☐ 是否考虑了数据局部性？                                    │
│     ☐ 是否设计了合适的访问顺序？                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 23.8.2 优化检查代码示例

```python
import triton
import triton.language as tl
import torch

# 完整的优化检查示例
@triton.jit
def optimized_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """完全优化的内核"""
    pid = tl.program_id(0)
    
    # 1. 内存合并：使用连续偏移量
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 2. 向量化加载：一次加载多个元素
    data = tl.load(input_ptr + offsets, mask=mask)
    
    # 3. 寄存器优化：重用变量
    data = data * 2.0  # 重用 data 变量
    data = data + 1.0  # 继续重用
    
    # 4. 避免条件分支：使用 tl.where
    result = tl.where(data > 0, data, -data)
    
    # 5. 顺序存储：对 L2 友好
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def check_memory_access_pattern(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """检查内存访问模式的内核"""
    pid = tl.program_id(0)
    
    # 检查 1: 内存合并
    # 所有线程访问连续地址
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 检查 2: 向量化
    # 一次加载 BLOCK_SIZE 个元素
    data = tl.load(input_ptr + offsets, mask=mask)
    
    # 检查 3: 数据复用
    # 同一块数据被多次使用
    result = data * 2.0 + 1.0
    result = result * result
    
    # 检查 4: 顺序存储
    tl.store(output_ptr + offsets, result, mask=mask)


def validate_optimization():
    """验证优化效果"""
    n_elements = 1 << 20  # 1M 元素
    BLOCK_SIZE = 1024
    
    # 创建测试数据
    input_tensor = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    output_tensor = torch.empty(n_elements, device='cuda', dtype=torch.float32)
    
    # 计算网格
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # 运行优化内核
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    optimized_kernel[grid](input_tensor, output_tensor, n_elements, BLOCK_SIZE)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end)
    
    # 计算带宽
    data_bytes = 2 * n_elements * 4  # 读写各一次
    bandwidth_gb_s = (data_bytes / 1e9) / (elapsed_ms / 1e3)
    
    print(f"性能分析:")
    print(f"  耗时: {elapsed_ms:.3f} ms")
    print(f"  带宽: {bandwidth_gb_s:.2f} GB/s")
    print(f"  HBM 峰值: 1.5 TB/s")
    print(f"  带宽利用率: {bandwidth_gb_s / 1500 * 100:.1f}%")
    
    # 检查优化点
    print(f"\n优化检查:")
    print(f"  ✓ 内存合并: 使用连续偏移量")
    print(f"  ✓ 向量化加载: 一次加载 {BLOCK_SIZE} 个元素")
    print(f"  ✓ 寄存器优化: 重用变量")
    print(f"  ✓ 避免条件分支: 使用 tl.where")
    print(f"  ✓ 顺序存储: 对 L2 友好")

# validate_optimization()
```

### 23.8.3 性能分析工具

```python
# 性能分析工具使用指南

def performance_analysis_guide():
    """性能分析工具指南"""
    
    print("1. NVIDIA Nsight Compute")
    print("   - 分析寄存器使用")
    print("   - 分析共享内存 bank conflict")
    print("   - 分析 L2 缓存命中率")
    print("   - 命令: ncu --metrics ... python your_script.py")
    
    print("\n2. NVIDIA Nsight Systems")
    print("   - 分析内核执行时间")
    print("   - 分析内存传输时间")
    print("   - 分析计算与内存重叠")
    print("   - 命令: nsys profile python your_script.py")
    
    print("\n3. Triton 内置工具")
    print("   - 查看编译后的 PTX")
    print("   - 分析寄存器分配")
    print("   - 命令: triton.tools ...")
    
    print("\n4. PyTorch Profiler")
    print("   - 分析整体性能")
    print("   - 分析内存使用")
    print("   - 命令: with torch.profiler.profile(...) as prof: ...")


def optimization_checklist_summary():
    """优化检查清单总结"""
    
    print("内存优化检查清单:")
    print("=" * 50)
    
    checklist = [
        ("内存合并", [
            "使用连续偏移量",
            "避免步长大于 1 的访问",
            "充分利用 128B cache line",
        ]),
        ("共享内存", [
            "检测并避免 bank conflict",
            "使用 padding 或 swizzle",
            "最大化数据复用",
        ]),
        ("寄存器", [
            "避免寄存器溢出",
            "重用变量减少压力",
            "使用 tl.where 避免分支",
        ]),
        ("Tiling", [
            "选择合适的块大小",
            "最大化数据局部性",
            "考虑硬件限制",
        ]),
        ("预取", [
            "启用 L2 预取",
            "配置合适的 num_stages",
            "计算与内存重叠",
        ]),
        ("淘汰策略", [
            "设计 L2 友好的访问模式",
            "避免随机访问",
            "考虑数据局部性",
        ]),
    ]
    
    for category, items in checklist:
        print(f"\n{category}:")
        for item in items:
            print(f"  ✓ {item}")

# optimization_checklist_summary()
```

---

## 本章小结

本章深入探讨了 GPU 内存优化的核心概念和技术，包括：

1. **内存层次结构**：理解了 HBM、L2 缓存、共享内存和寄存器的性能特性
2. **内存合并**：掌握了 coalesced vs uncoalesced 访问的区别，以及如何通过连续偏移量实现合并
3. **共享内存优化**：了解了 bank conflict 的成因，以及 padding 和 swizzle 策略
4. **寄存器分配**：学习了 Triton 编译器的寄存器管理策略，以及如何避免溢出
5. **数据复用**：掌握了 tiling 技术，以及如何通过共享内存提高数据复用率
6. **预取技术**：了解了 L2 预取和 cp.async 的工作原理，以及 num_stages 的配置
7. **淘汰策略**：理解了 L2 缓存的淘汰机制，以及如何设计 L2 友好的访问模式
8. **优化清单**：提供了完整的内存优化检查清单，帮助系统性地进行优化

**关键要点**：
- 内存优化的核心是**减少全局内存访问**，**提高缓存命中率**
- 合并访问是 GPU 内存优化的基础，必须保证 warp 内线程访问连续地址
- 共享内存是提高数据复用的关键，但要注意 bank conflict
- 寄存器优化需要在并行度和寄存器使用之间取得平衡
- 预取和淘汰策略可以显著影响性能，需要根据具体场景选择

**下一步**：
- 在实际项目中应用这些优化技术
- 使用性能分析工具验证优化效果
- 针对特定场景（如矩阵乘法、卷积）进行深度优化

---

## 思考题

1. **内存合并问题**：
   在什么情况下，即使使用连续偏移量，内存访问也可能不是合并的？如何解决？

2. **Bank Conflict 分析**：
   假设有一个 32x32 的共享内存数组，每个元素是 float32（4 字节）。分析以下访问模式是否会导致 bank conflict：
   - 访问 A[i][i]（对角线访问）
   - 访问 A[i][0]（列访问）
   - 访问 A[0][i]（行访问）

3. **寄存器优化**：
   如何在保持算法正确性的同时，减少内核的寄存器使用量？请给出具体的优化策略。

4. **Tiling 策略**：
   对于一个大型矩阵乘法（4096x4096），如何选择合适的块大小（BLOCK_M, BLOCK_N, BLOCK_K）？考虑寄存器限制、共享内存限制和性能因素。

5. **预取配置**：
   在什么场景下，增加 num_stages 可能不会提高性能？如何确定最优的 num_stages 值？

6. **淘汰策略设计**：
   设计一个数据访问模式，使其对 L2 缓存友好，同时满足特定的计算需求（例如，需要同时访问矩阵的行和列）。

7. **综合优化**：
   给定一个实际的计算任务（例如，实现一个高效的卷积操作），如何综合应用本章学到的所有内存优化技术？请详细说明优化思路和具体实现。

---

## 参考资源

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Triton Documentation](https://triton-lang.org/)
- [GPU Memory Optimization](https://developer.nvidia.com/blog/optimizing-gpu-memory-access-patterns/)
- [Shared Memory Bank Conflicts](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#shared-memory-bank-conflicts)
- [Memory Coalescing](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#memory-coalescing)

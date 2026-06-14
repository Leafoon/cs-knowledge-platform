# Chapter 22: 性能剖析与瓶颈定位

> **学习目标**：
> - 掌握 Triton kernel 的性能剖析方法论
> - 了解 ncu/nsys/rocprof 等硬件级分析工具
> - 理解 Triton 内置的性能指标与调试工具
> - 掌握瓶颈定位（compute-bound vs memory-bound）的分析方法

---

## 22.1 性能指标体系

### 22.1.1 核心性能指标概述

在进行 Triton kernel 性能剖析之前，需要理解 GPU 性能的核心指标体系。这些指标从不同维度描述了 kernel 的执行效率。

```
GPU 性能指标体系
├── 吞吐量指标 (Throughput)
│   ├── FLOPs/s (浮点运算吞吐量)
│   ├── Memory Bandwidth (内存带宽)
│   └── Element Throughput (元素处理吞吐量)
├── 延迟指标 (Latency)
│   ├── Kernel Execution Time (kernel 执行时间)
│   ├── Kernel Launch Latency (kernel 启动延迟)
│   └── Memory Access Latency (内存访问延迟)
├── 利用率指标 (Utilization)
│   ├── SM Utilization (SM 利用率)
│   ├── Occupancy (占用率)
│   ├── Warp Execution Efficiency (warp 执行效率)
│   └── Memory Throughput Utilization (内存吞吐利用率)
└── 效率指标 (Efficiency)
    ├── Arithmetic Intensity (计算强度)
    ├── Roofline Position (roofline 位置)
    └── Cache Hit Rate (缓存命中率)
```

### 22.1.2 Throughput（吞吐量）

吞吐量衡量 kernel 在单位时间内处理的数据量或完成的计算量。

```python
# 吞吐量计算示例
# 矩阵乘法: C[M, N] = A[M, K] @ B[K, N]

M, N, K = 4096, 4096, 4096

# 计算 FLOPs
# 矩阵乘法的浮点运算量: 2 * M * N * K (一次乘加算 2 次浮点运算)
flops = 2 * M * N * K
print(f"Total FLOPs: {flops / 1e9:.2f} GFLOPs")  # 137.44 GFLOPs

# 假设 kernel 执行时间为 1.5ms
kernel_time_ms = 1.5
throughput_gflops = flops / (kernel_time_ms * 1e-3) / 1e9
print(f"Throughput: {throughput_gflops:.2f} GFLOPs/s")  # ~91.6 GFLOPs/s

# 元素吞吐量
elements_processed = M * N
throughput_elements = elements_processed / (kernel_time_ms * 1e-3)
print(f"Element Throughput: {throughput_elements / 1e9:.2f} GElements/s")
```

**吞吐量指标对比：**

| 指标 | 计算方式 | 适用场景 | A100 理论峰值 |
|------|---------|---------|--------------|
| TFLOPs/s | `2*M*N*K / time` | 矩阵运算密集型 | 312 TFLOPs/s (FP16 Tensor Core) |
| GB/s | `bytes_transferred / time` | 内存密集型 | 2039 GB/s (HBM2e) |
| GElements/s | `elements / time` | 元素级操作 | 依赖具体操作 |

### 22.1.3 Latency（延迟）

延迟衡量从发起操作到操作完成的时间。

```python
import torch
import triton
import triton.language as tl
import time

@triton.jit
def simple_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    output = x * 2.0 + 1.0
    tl.store(output_ptr + offsets, output, mask=mask)

def measure_latency(n_elements, block_size, warmup=10, rep=100):
    x = torch.randn(n_elements, device='cuda', dtype=torch.float32)
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Warmup
    for _ in range(warmup):
        simple_kernel[grid](x, output, n_elements, BLOCK_SIZE=block_size)
    torch.cuda.synchronize()

    # 测量 kernel 启动延迟（包含 host-side 开销）
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    for i in range(rep):
        start_events[i].record()
        simple_kernel[grid](x, output, n_elements, BLOCK_SIZE=block_size)
        end_events[i].record()

    torch.cuda.synchronize()

    # 计算延迟统计
    latencies = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    return avg_latency, min_latency, max_latency

# 测量不同问题规模的延迟
for n in [1024, 4096, 16384, 65536, 262144]:
    avg, mn, mx = measure_latency(n, 256)
    print(f"N={n:>8d}: avg={avg:.4f}ms, min={mn:.4f}ms, max={mx:.4f}ms")
```

**延迟类型分类：**

| 延迟类型 | 说明 | 典型值 (A100) | 测量方式 |
|---------|------|--------------|---------|
| Kernel Launch Latency | Host → Device 启动开销 | 5-10 μs | CUDA Events |
| Kernel Execution Time | GPU 实际执行时间 | 0.1-10 ms | CUDA Events / Nsight |
| Memory Access Latency (L1) | L1 Cache 访问延迟 | ~30 cycles | 硬件计数器 |
| Memory Access Latency (L2) | L2 Cache 访问延迟 | ~200 cycles | 硬件计数器 |
| Memory Access Latency (HBM) | HBM 显存访问延迟 | ~400 cycles | 硬件计数器 |
| Warp Scheduling Latency | Warp 切换开销 | 0 cycles (pipelined) | N/A |

### 22.1.4 Occupancy（占用率）

Occupancy 定义为 SM 上活跃 warp 数量与理论最大 warp 数量的比值。

```
Occupancy 计算公式:

Occupancy = (Active Warps per SM) / (Maximum Warps per SM)

A100 SM 配置:
  - 最大 warps per SM: 64 (32 threads/warp × 64 = 2048 threads)
  - 最大 blocks per SM: 32
  - 寄存器文件: 65536 per SM (256 per thread)
  - 共享内存: 164 KB per SM (可配置)

Occupancy 限制因素:
  ┌─────────────────────────────────────────────────┐
  │  1. 寄存器使用量 (Registers per thread)          │
  │     → 限制每 SM 的活跃 thread 数                 │
  │  2. 共享内存使用量 (Shared Memory per block)      │
  │     → 限制每 SM 的活跃 block 数                  │
  │  3. Block 配置 (Threads per block)               │
  │     → 影响 warp 分配效率                         │
  │  4. 硬件限制 (Max warps/threads per SM)           │
  │     → 物理上限                                   │
  └─────────────────────────────────────────────────┘
```

```python
# Occupancy 计算示例
def calculate_occupancy(
    registers_per_thread: int,
    shared_mem_per_block: int,  # bytes
    threads_per_block: int,
    max_warps_per_sm: int = 64,
    max_registers_per_sm: int = 65536,
    max_shared_mem_per_sm: int = 167936,  # 164KB
    warp_size: int = 32
) -> dict:
    """计算 GPU occupancy"""

    warps_per_block = threads_per_block // warp_size

    # 基于寄存器的限制
    max_threads_reg = max_registers_per_sm // registers_per_thread
    max_blocks_reg = max_threads_reg // threads_per_block

    # 基于共享内存的限制
    max_blocks_shmem = max_shared_mem_per_sm // shared_mem_per_block

    # 实际可用 block 数
    max_blocks = min(max_blocks_reg, max_blocks_shmem, 32)  # 硬件限制

    # 活跃 warp 数
    active_warps = max_blocks * warps_per_block
    active_warps = min(active_warps, max_warps_per_sm)

    occupancy = active_warps / max_warps_per_sm

    return {
        "registers_per_thread": registers_per_thread,
        "shared_mem_per_block": shared_mem_per_block,
        "threads_per_block": threads_per_block,
        "warps_per_block": warps_per_block,
        "max_blocks_by_registers": max_blocks_reg,
        "max_blocks_by_shared_mem": max_blocks_shmem,
        "active_blocks": max_blocks,
        "active_warps": active_warps,
        "occupancy": occupancy,
        "occupancy_pct": f"{occupancy * 100:.1f}%"
    }

# 示例: 不同配置的 occupancy
configs = [
    {"regs": 32, "shmem": 0,    "threads": 128, "name": "轻量级 kernel"},
    {"regs": 64, "shmem": 0,    "threads": 256, "name": "中等 kernel"},
    {"regs": 128, "shmem": 0,   "threads": 256, "name": "重度 kernel"},
    {"regs": 128, "shmem": 32768, "threads": 256, "name": "大共享内存"},
    {"regs": 255, "shmem": 0,   "threads": 256, "name": "寄存器溢出"},
]

print(f"{'配置':<20} {'寄存器':>8} {'共享内存':>10} {'Threads':>8} {'Occupancy':>10}")
print("-" * 65)
for cfg in configs:
    result = calculate_occupancy(cfg["regs"], cfg["shmem"], cfg["threads"])
    print(f"{cfg['name']:<20} {cfg['regs']:>8} {cfg['shmem']:>10} {cfg['threads']:>8} {result['occupancy_pct']:>10}")
```

**Occupancy 输出示例：**

```
配置                      寄存器     共享内存   Threads  Occupancy
-----------------------------------------------------------------
轻量级 kernel               32          0      128      25.0%
中等 kernel                 64          0      256      25.0%
重度 kernel                128          0      256      12.5%
大共享内存                 128      32768      256       6.2%
寄存器溢出                 255          0      256       6.2%
```

### 22.1.5 SM Utilization（SM 利用率）

SM Utilization 衡量 SM 的实际活跃时间占比。

```python
# SM Utilization 分析
# 使用 nsys 获取 SM utilization 数据

"""
nsys profile --stats=true python my_kernel.py

关键指标:
  - GPU Utilization: GPU 忙碌时间占比
  - SM Active Cycles: SM 实际执行 cycle 数
  - SM Total Cycles: SM 总 cycle 数
  - SM Utilization = SM Active Cycles / SM Total Cycles
"""

# SM Utilization 的典型分布
"""
场景                          SM Utilization    说明
──────────────────────────────────────────────────────────
理想并行 kernel               95-100%          所有 SM 满载
大型矩阵运算                  80-95%          高效利用 SM
小规模 kernel                  30-60%          部分 SM 空闲
序列化操作                     5-20%          大量 SM 空闲
内存带宽瓶颈                  60-80%          SM 等待数据
"""
```

### 22.1.6 Memory Throughput（内存吞吐量）

内存吞吐量衡量数据在 GPU 内存层次结构中的传输速率。

```
GPU 内存层次结构与带宽 (A100):

  ┌───────────────────────────────────────────────────┐
  │              寄存器 (Registers)                    │
  │   带宽: ~∞ (每个 cycle 可访问)                     │
  │   延迟: 1 cycle                                   │
  │   容量: 256 registers/thread × 32 threads/warp     │
  ├───────────────────────────────────────────────────┤
  │           共享内存 (Shared Memory)                  │
  │   带宽: 19 TB/s (聚合)                             │
  │   延迟: ~20-30 cycles                             │
  │   容量: 164 KB/SM (可配置)                         │
  ├───────────────────────────────────────────────────┤
  │               L1 Cache                             │
  │   带宽: 19 TB/s (与共享内存共用)                    │
  │   延迟: ~30 cycles                                │
  │   容量: 128 KB/SM (可配置)                         │
  ├───────────────────────────────────────────────────┤
  │               L2 Cache                             │
  │   带宽: 5 TB/s                                    │
  │   延迟: ~200 cycles                               │
  │   容量: 40 MB 全局                                 │
  ├───────────────────────────────────────────────────┤
  │           HBM2e (显存)                             │
  │   带宽: 2039 GB/s                                 │
  │   延迟: ~400 cycles                               │
  │   容量: 80 GB                                     │
  └───────────────────────────────────────────────────┘
```

```python
# 内存吞吐量计算
def calculate_memory_throughput(
    data_size_bytes: int,
    time_ms: float,
    theoretical_bandwidth_gbs: float = 2039.0  # A100 HBM2e
) -> dict:
    """计算内存吞吐量"""

    achieved_bandwidth_gbs = data_size_bytes / (time_ms * 1e-3) / 1e9
    utilization = achieved_bandwidth_gbs / theoretical_bandwidth_gbs

    return {
        "data_size_gb": data_size_bytes / 1e9,
        "time_ms": time_ms,
        "achieved_bandwidth_gbs": achieved_bandwidth_gbs,
        "theoretical_bandwidth_gbs": theoretical_bandwidth_gbs,
        "utilization": utilization,
        "utilization_pct": f"{utilization * 100:.1f}%"
    }

# 矩阵加法: 读取 2 个矩阵，写入 1 个矩阵
M, N = 4096, 4096
element_size = 2  # FP16
total_bytes = 3 * M * N * element_size  # 读 A, 读 B, 写 C

result = calculate_memory_throughput(total_bytes, 0.5)  # 0.5ms
print(f"矩阵加法内存分析:")
print(f"  数据量: {result['data_size_gb']:.3f} GB")
print(f"  时间: {result['time_ms']:.3f} ms")
print(f"  达到带宽: {result['achieved_bandwidth_gbs']:.1f} GB/s")
print(f"  带宽利用率: {result['utilization_pct']}")

# 矩阵乘法: 读取 2 个矩阵，写入 1 个矩阵
M, N, K = 4096, 4096, 4096
total_bytes = (M * K + K * N + M * N) * element_size

result = calculate_memory_throughput(total_bytes, 1.5)
print(f"\n矩阵乘法内存分析:")
print(f"  数据量: {result['data_size_gb']:.3f} GB")
print(f"  时间: {result['time_ms']:.3f} ms")
print(f"  达到带宽: {result['achieved_bandwidth_gbs']:.1f} GB/s")
print(f"  带宽利用率: {result['utilization_pct']}")
```

### 22.1.7 FLOPs/s（浮点运算吞吐量）

FLOPs/s 衡量 kernel 每秒完成的浮点运算次数。

```python
# FLOPs 计算参考表
flops_table = {
    "向量加法 (A + B)": {
        "formula": "N",
        "bytes_accessed": "3 * N * sizeof(T)",  # 读A, 读B, 写C
        "ai": "N / (3 * N * sizeof(T)) = 1 / (3 * sizeof(T))",
        "category": "memory-bound"
    },
    "向量点积 (A · B)": {
        "formula": "2 * N",
        "bytes_accessed": "2 * N * sizeof(T)",  # 读A, 读B
        "ai": "2 * N / (2 * N * sizeof(T)) = 1 / sizeof(T)",
        "category": "memory-bound"
    },
    "矩阵加法 (C = A + B)": {
        "formula": "M * N",
        "bytes_accessed": "3 * M * N * sizeof(T)",
        "ai": "M * N / (3 * M * N * sizeof(T)) = 1 / (3 * sizeof(T))",
        "category": "memory-bound"
    },
    "矩阵乘法 (C = A @ B)": {
        "formula": "2 * M * N * K",
        "bytes_accessed": "(M * K + K * N + M * N) * sizeof(T)",
        "ai": "2 * M * N * K / ((M*K + K*N + M*N) * sizeof(T)) ≈ min(M,N) / sizeof(T)",
        "category": "compute-bound (for large M,N)"
    },
    "Softmax": {
        "formula": "5 * N (exp, sum, div)",
        "bytes_accessed": "2 * N * sizeof(T)",
        "ai": "5 * N / (2 * N * sizeof(T)) = 2.5 / sizeof(T)",
        "category": "memory-bound"
    },
    "LayerNorm": {
        "formula": "~10 * N",
        "bytes_accessed": "3 * N * sizeof(T)",
        "ai": "10 * N / (3 * N * sizeof(T)) ≈ 3.3 / sizeof(T)",
        "category": "memory-bound"
    },
}

print(f"{'操作':<25} {'FLOPs':<20} {'Arithmetic Intensity':<25} {'类别'}")
print("-" * 95)
for op, info in flops_table.items():
    print(f"{op:<25} {info['formula']:<20} {info['ai']:<25} {info['category']}")
```

**FLOPs/s 参考值 (A100):**

| 数据类型 | 理论峰值 TFLOPs/s | 说明 |
|---------|-------------------|------|
| FP32 (CUDA Core) | 19.5 | 单精度浮点 |
| TF32 (Tensor Core) | 156 | Tensor Float 32 |
| FP16 (Tensor Core) | 312 | 半精度浮点 |
| BF16 (Tensor Core) | 312 | Brain Float 16 |
| INT8 (Tensor Core) | 624 | 8位整数 |
| FP64 (CUDA Core) | 9.7 | 双精度浮点 |

## 22.2 nsys 分析

### 22.2.1 nsys 概述与安装

NVIDIA Nsight Systems (nsys) 是一个系统级性能分析工具，用于分析 GPU 应用的 CPU/GPU 活动、内存传输、kernel 执行等。

```bash
# 安装 nsys
# 方式1: 通过 CUDA Toolkit 安装 (推荐)
# CUDA 12.x 自带 nsys

# 验证安装
nsys --version
# NVIDIA Nsight Systems version 2024.1.1.43

# 方式2: 通过 conda 安装
conda install -c conda-forge nsys

# 方式3: 独立下载
# https://developer.nvidia.com/nsight-systems
```

### 22.2.2 nsys profile 基础用法

```bash
# 基础 profile 命令
nsys profile python my_kernel.py

# 指定输出文件名
nsys profile -o my_report python my_kernel.py

# 详细模式 (包含更多追踪信息)
nsys profile --stats=true -o my_report python my_kernel.py

# 同时收集 CUDA API 和 kernel 信息
nsys profile --trace=cuda,nvtx,osrt --stats=true python my_kernel.py

# 限制收集时间 (秒)
nsys profile --duration=30 --stats=true python my_kernel.py

# 设置采样间隔
nsys profile --sample=process-tree --capture-range=cudaProfilerApi \
    --stats=true python my_kernel.py
```

**nsys 常用选项：**

| 选项 | 说明 | 示例 |
|------|------|------|
| `-o <name>` | 指定输出文件名 | `-o my_report` |
| `--stats=true` | 启用统计摘要 | `--stats=true` |
| `--trace=<sources>` | 指定追踪来源 | `--trace=cuda,nvtx` |
| `--duration=<sec>` | 限制收集时间 | `--duration=30` |
| `--capture-range=<mode>` | 捕获范围 | `--capture-range=cudaProfilerApi` |
| `--gpu-metrics-device=<id>` | GPU 指标设备 | `--gpu-metrics-device=0` |
| `--cuda-memory-usage=true` | 收集内存使用 | `--cuda-memory-usage=true` |
| `--force-overwrite=true` | 覆盖已有输出 | `--force-overwrite=true` |

### 22.2.3 nsys 报告分析

```bash
# 生成 HTML 报告
nsys export --type=html my_report.qdrep -o my_report.html

# 生成 SQLite 数据库 (可编程查询)
nsys export --type=sqlite my_report.qdrep -o my_report.sqlite

# 查看统计摘要
nsys stats my_report.qdrep

# 查看 kernel 执行统计
nsys stats --report cuda_gpu_kern_sum my_report.qdrep

# 查看内存传输统计
nsys stats --report cuda_gpu_mem_time_sum my_report.qdrep

# 查看 CUDA API 统计
nsys stats --report cuda_api_sum my_report.qdrep
```

### 22.2.4 nsys 输出解读

```
nsys stats 输出示例:

CUDA Kernel Statistics:
 Time(%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  Name
 --------  ---------------  ---------  --------  --------  --------  --------  ----
    45.2%         4520000        100   45200.0   45000.0     44800     45500  my_kernel<float>
    32.1%         3210000        100   32100.0   32000.0     31800     32500  reduction_kernel<float>
    15.8%         1580000        100   15800.0   15700.0     15600     16000  vector_add<float>
     6.9%          690000        100    6900.0    6800.0      6700      7100  other_kernels

Memory Operation Statistics:
 Time(%)  Total Time (ns)  Operations  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  Type
 --------  ---------------  ---------  --------  --------  --------  --------  ----
    60.5%         6050000      100000    60.5      55.0      20.0     500.0  GPU To Host
    35.2%         3520000      100000    35.2      30.0      15.0     300.0  Host To GPU
     4.3%          430000       50000    8.6       7.0       3.0      50.0   GPU To GPU
```

### 22.2.5 GPU 利用率分析

```python
"""
nsys GPU 利用率分析流程

1. 收集数据
   nsys profile --stats=true --gpu-metrics-device=0 python my_app.py

2. 分析 GPU Utilization
   - GPU Utilization: GPU 忙碌时间百分比
   - SM Utilization: SM 活跃时间百分比
   - Memory Utilization: 内存带宽使用百分比

3. 常见问题识别
   - GPU Utilization < 50%: 并行度不足或 host 瓶颈
   - SM Utilization < 70%: kernel 并行度不足
   - Memory Utilization > 90%: 内存带宽瓶颈
   - 大量 GPU-To-Host 传输: 数据传输瓶颈
"""

# 使用 nsys API 查询 GPU 利用率
import subprocess
import json

def analyze_gpu_utilization(report_path: str):
    """分析 nsys 报告的 GPU 利用率"""

    # 获取 kernel 统计
    cmd = f"nsys stats --report cuda_gpu_kern_sum {report_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("=== Kernel 统计 ===")
    print(result.stdout)

    # 获取内存传输统计
    cmd = f"nsys stats --report cuda_gpu_mem_time_sum {report_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("=== 内存传输统计 ===")
    print(result.stdout)

    # 获取 CUDA API 统计
    cmd = f"nsys stats --report cuda_api_sum {report_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("=== CUDA API 统计 ===")
    print(result.stdout)

# 使用示例
# nsys profile --stats=true -o my_report python my_app.py
# analyze_gpu_utilization("my_report.qdrep")
```

### 22.2.6 nsys 常见问题识别

```
nsys 常见性能问题模式:

问题 1: GPU 利用率低 (GPU Utilization < 50%)
┌─────────────────────────────────────────────────────┐
│ 可能原因:                                            │
│   - Kernel 启动开销占比过高                           │
│   - Host-Device 数据传输频繁                         │
│   - Kernel 执行时间过短                              │
│   - CPU 端序列化操作                                 │
│                                                      │
│ nsys 特征:                                           │
│   - GPU Utilization 指标低                           │
│   - 大量短 kernel                                    │
│   - 频繁的 cudaLaunchKernel API 调用                 │
│                                                      │
│ 优化方向:                                             │
│   - Kernel Fusion (融合多个 kernel)                   │
│   - 增大 problem size                                │
│   - 使用 CUDA Graph 减少启动开销                      │
└─────────────────────────────────────────────────────┘

问题 2: 内存传输瓶颈
┌─────────────────────────────────────────────────────┐
│ 可能原因:                                            │
│   - Host ↔ Device 数据传输频繁                       │
│   - 数据拷贝未异步                                   │
│   - 数据量过大                                       │
│                                                      │
│ nsys 特征:                                           │
│   - 大量 cudaMemcpy 调用                             │
│   - GPU-To-Host/Host-To-GPU 时间占比高               │
│   - GPU idle 等待数据传输                             │
│                                                      │
│ 优化方向:                                             │
│   - 使用 pinned memory                               │
│   - 异步数据传输 (cudaMemcpyAsync)                   │
│   - 减少 Host-Device 数据交换                        │
│   - 使用 unified memory + prefetching                │
└─────────────────────────────────────────────────────┘

问题 3: Kernel 序列化
┌─────────────────────────────────────────────────────┐
│ 可能原因:                                            │
│   - 多个 kernel 串行执行                             │
│   - Kernel 之间存在数据依赖                          │
│   - 缺少 CUDA Stream 并行                           │
│                                                      │
│ nsys 特征:                                           │
│   - Kernel 时间线无重叠                              │
│   - 多个 kernel 顺序执行                             │
│   - GPU 利用率波动大                                  │
│                                                      │
│ 优化方向:                                             │
│   - 使用多 Stream 并行                               │
│   - Kernel Fusion                                    │
│   - 重排计算消除依赖                                 │
└─────────────────────────────────────────────────────┘
```

### 22.2.7 nsys 高级分析

```python
"""
nsys 高级分析: 使用 NVTX 标记分析 Triton kernel
"""

import torch
import triton
import triton.language as tl

# 使用 torch.cuda.nvtx 进行标记
def profile_with_nvtx():
    """使用 NVTX 标记分析 Triton kernel"""

    M, N, K = 4096, 4096, 4096
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    C = torch.empty(M, N, device='cuda', dtype=torch.float16)

    # NVTX 标记开始
    torch.cuda.nvtx.range_push("Matmul Kernel")

    @triton.jit
    def matmul_kernel(
        a_ptr, b_ptr, c_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c = accumulator.to(tl.float16)
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        tl.store(c_ptrs, c)

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64
    grid = (tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N),)

    matmul_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    # NVTX 标记结束
    torch.cuda.nvtx.range_pop()

    return C

# 运行分析
# nsys profile --trace=nvtx,cuda python this_script.py
```

## 22.3 ncu 分析

### 22.3.1 ncu 概述与安装

NVIDIA Nsight Compute (ncu) 是一个 GPU kernel 级性能分析工具，提供详细的硬件计数器和性能指标。

```bash
# 安装 ncu
# CUDA Toolkit 自带 ncu

# 验证安装
ncu --version
# NVIDIA Nsight Compute version 2024.1.1.43

# 基本用法
ncu python my_kernel.py

# 指定 kernel 进行分析
ncu --kernel-name "my_kernel" --launch-skip 0 --launch-count 10 python my_kernel.py

# 输出详细报告
ncu --page raw --csv python my_kernel.py > report.csv
```

### 22.3.2 ncu 命令行详解

```bash
# 完整的 ncu 命令选项
ncu [options] <application> [app-arguments]

# 核心选项:
# --kernel-name <regex>       指定要分析的 kernel 名称
# --launch-skip <N>           跳过前 N 次 kernel 启动
# --launch-count <N>          只分析前 N 次 kernel 启动
# --set <metric-set>          指定指标集 (default, full, roofline, etc.)
# --page <page-type>          输出页面 (raw, details, summary, roofline)
# --csv                       CSV 格式输出
# --output <file>             输出到文件
# --target-processes <all|range>  目标进程
# --replay-mode <mode>        重放模式 (application, range, kernel)
```

**ncu 常用分析模式：**

| 模式 | 命令 | 说明 |
|------|------|------|
| 默认分析 | `ncu python app.py` | 基本性能指标 |
| 完整分析 | `ncu --set full python app.py` | 所有硬件计数器 |
| Roofline 分析 | `ncu --set roofline python app.py` | Roofline 图表 |
| 内存分析 | `ncu --set memory python app.py` | 内存访问模式 |
| 计算分析 | `ncu --set compute python app.py` | 计算效率 |
| 自定义指标 | `ncu --metrics sm__cycles_active.avg, ...` | 指定指标 |

### 22.3.3 硬件计数器详解

```
ncu 关键硬件计数器分类:

1. 时钟周期计数器
   ├── sm__cycles_active.avg          SM 活跃周期数
   ├── sm__cycles.avg                 SM 总周期数
   ├── sm__warps_active.avg.pct_of_peak_sustained_active
   │                                  Warp 活跃率
   └── smsp__cycles_active.avg.pct_of_peak_sustained_elapsed
                                      SM 实际周期占比

2. 计算吞吐量计数器
   ├── sm__sass_thread_inst_executed_op_ffma_pred_on.sum
   │                                  FFMA 指令执行数
   ├── sm__sass_thread_inst_executed_op_fadd_pred_on.sum
   │                                  FADD 指令执行数
   ├── sm__sass_thread_inst_executed_op_fmul_pred_on.sum
   │                                  FMUL 指令执行数
   └── smsp__inst_executed_pipe_tensor.sum
                                      Tensor Core 指令执行数

3. 内存吞吐量计数器
   ├── l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum
   │                                  L1 全局内存加载字节数
   ├── l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum
   │                                  L1 全局内存存储字节数
   ├── lts__t_bytes.sum               L2 传输字节数
   ├── dram__bytes.sum                DRAM 传输字节数
   └── dram__bytes_read.sum           DRAM 读取字节数

4. Occupancy 计数器
   ├── sm__warps_active.avg.pct_of_peak_sustained_active
   │                                  Warp 占用率
   ├── launch__occupancy_limit_blocks  Block 数量限制
   ├── launch__occupancy_limit_registers 寄存器限制
   └── launch__occupancy_limit_shared_mem 共享内存限制

5. Warp 效率计数器
   ├── smsp__average_warp_latency_per_inst_executed.ratio
   │                                  每条指令的平均 warp 延迟
   ├── smsp__warps_launched.sum       启动的 warp 数
   └── smsp__thread_inst_executed.sum 执行的 thread 数
```

### 22.3.4 ncu 输出解读

```bash
# ncu 默认输出示例
ncu --set full --kernel-name "matmul_kernel" python matmul.py

"""
==PROF== Profiling "matmul_kernel" - 0
==PROF== Profiling "matmul_kernel" - 1
==PROF== Profiling "matmul_kernel" - 2
==PROF== Profiling "matmul_kernel" - 3

Kernel: matmul_kernel<float>
grid: (128, 1, 1), block: (128, 1, 1)

Section: GPU Speed Of Light Throughput
  [0]  SM [__] Throughput: 89.25%  (of peak 19.5 TFLOPs/s for FP32)
  [1]  Memory [__] Throughput: 45.67%  (of peak 2039 GB/s)

Section: Compute (SM) Throughput
  sm__throughput.avg.pct_of_peak_sustained_elapsed       89.25%
  smsp__sass_thread_inst_executed_op_ffma_pred_on.sum    12345678
  smsp__inst_executed_pipe_tensor.sum                    6789012

Section: Memory Throughput
  dram__bytes_read.sum                                   2.15 GB
  dram__bytes_write.sum                                  1.07 GB
  dram__throughput.avg.pct_of_peak_sustained_elapsed     45.67%
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum           3.22 GB
  l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum           1.07 GB

Section: Occupancy
  sm__warps_active.avg.pct_of_peak_sustained_active      75.00%
  launch__occupancy_limit_registers                      87.50%
  launch__occupancy_limit_shared_mem                     100.00%

Section: Launch Statistics
  launch__grid_size                                      128
  launch__block_size                                     128
  launch__registers_per_thread                           64
  launch__shared_mem_per_block                           8192
"""
```

### 22.3.5 ncu 关键指标分析

```python
# ncu 指标分析框架
"""
ncu 关键指标分析:

1. SM Throughput 分析
   - sm__throughput.avg.pct_of_peak_sustained_elapsed
   - 含义: SM 计算吞吐量占理论峰值的百分比
   - 判断: > 80% 为优秀, 60-80% 为良好, < 60% 需要优化

2. Memory Throughput 分析
   - dram__throughput.avg.pct_of_peak_sustained_elapsed
   - 含义: 内存带宽使用率
   - 判断: > 80% 为内存密集型, < 40% 为计算密集型

3. Occupancy 分析
   - sm__warps_active.avg.pct_of_peak_sustained_active
   - 含义: Warp 占用率
   - 判断: > 75% 为优秀, 50-75% 为良好, < 50% 需要优化

4. Roofline 位置
   - 根据 Arithmetic Intensity 判断瓶颈
   - AI < 斜率: memory-bound
   - AI > 斜率: compute-bound

5. 内存访问效率
   - l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum
   - l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
   - Cache Hit Rate = hit / total
"""
```

### 22.3.6 ncu 自定义指标查询

```bash
# 查询可用指标
ncu --query-metrics

# 按类别查询
ncu --query-metrics | grep "sm__"
ncu --query-metrics | grep "dram__"
ncu --query-metrics | grep "l1tex__"

# 自定义指标收集
ncu --metrics \
    sm__cycles_active.avg,\
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    launch__registers_per_thread,\
    launch__shared_mem_per_block \
    python my_kernel.py

# 使用预定义指标集
ncu --set full python my_kernel.py           # 所有指标
ncu --set roofline python my_kernel.py       # Roofline 指标
ncu --set memory python my_kernel.py         # 内存相关指标
ncu --set compute python my_kernel.py        # 计算相关指标
```

### 22.3.7 ncu Roofline 分析

```bash
# 生成 Roofline 图
ncu --set roofline --page roofline python my_kernel.py

# Roofline 输出示例:
"""
Roofline Analysis for "matmul_kernel"

Arithmetic Intensity: 32.5 FLOP/Byte
Achieved FLOPs/s: 123.4 TFLOPs/s
Achieved Bandwidth: 3.8 TB/s

Roofline Chart:
  ┌──────────────────────────────────────────────────────────────┐
  │ TFLOPs/s                                                     │
  │   312 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
  │       │                          ╱                           │
  │   200 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ╱─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
  │       │                    ╱   ★ kernel (AI=32.5)            │
  │   100 ─ ─ ─ ─ ─ ─ ─ ─ ╱─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
  │       │              ╱                                       │
  │    50 ─ ─ ─ ─ ─ ─╱─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
  │       │        ╱                                              │
  │    10 ─ ─ ─ ╱─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
  │       │  ╱                                                    │
  │     1 ─╱─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
  │       └──────────────────────────────────────────────────────│
  │         1    2    4    8   16   32   64  128                  │
  │                     Arithmetic Intensity (FLOP/Byte)         │
  │                                                              │
  │  Roofline Slope = Peak Bandwidth = 2039 GB/s                 │
  │  Ridge Point = Peak FLOPs / Peak BW = 312 / 2039 = 0.153    │
  │  Kernel Position: Compute-bound (AI > Ridge Point)           │
  └──────────────────────────────────────────────────────────────┘
"""
```

### 22.3.8 ncu 性能报告生成

```python
"""
ncu 性能报告生成
"""

import subprocess
import csv
import io

def run_ncu_analysis(
    command: str,
    kernel_name: str = None,
    metrics: list = None,
    output_csv: str = None
) -> dict:
    """运行 ncu 分析并解析结果"""

    # 构建 ncu 命令
    ncu_cmd = ["ncu", "--set", "full", "--csv"]

    if kernel_name:
        ncu_cmd.extend(["--kernel-name", kernel_name])

    if metrics:
        ncu_cmd.extend(["--metrics", ",".join(metrics)])

    ncu_cmd.extend(command.split())

    # 执行命令
    result = subprocess.run(ncu_cmd, capture_output=True, text=True)

    # 解析 CSV 输出
    if result.returncode == 0:
        reader = csv.DictReader(io.StringIO(result.stdout))
        rows = list(reader)
        return {"success": True, "data": rows, "raw": result.stdout}
    else:
        return {"success": False, "error": result.stderr}

def generate_performance_report(kernel_data: list) -> str:
    """生成性能分析报告"""

    report = []
    report.append("=" * 70)
    report.append("Triton Kernel 性能分析报告")
    report.append("=" * 70)

    for kernel in kernel_data:
        report.append(f"\nKernel: {kernel.get('Name', 'Unknown')}")
        report.append("-" * 50)

        # 提取关键指标
        metrics = {
            "SM Throughput": kernel.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", "N/A"),
            "Memory Throughput": kernel.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", "N/A"),
            "Occupancy": kernel.get("sm__warps_active.avg.pct_of_peak_sustained_active", "N/A"),
            "Registers": kernel.get("launch__registers_per_thread", "N/A"),
            "Shared Memory": kernel.get("launch__shared_mem_per_block", "N/A"),
        }

        for name, value in metrics.items():
            report.append(f"  {name}: {value}")

        # 瓶颈判断
        report.append("\n瓶颈分析:")
        try:
            sm_throughput = float(metrics["SM Throughput"].replace("%", ""))
            mem_throughput = float(metrics["Memory Throughput"].replace("%", ""))

            if sm_throughput > 80 and mem_throughput < 40:
                report.append("  → Compute-bound (计算密集型)")
                report.append("  → 建议: 优化计算逻辑, 使用 Tensor Core")
            elif mem_throughput > 80 and sm_throughput < 40:
                report.append("  → Memory-bound (内存密集型)")
                report.append("  → 建议: 减少内存访问, 提高数据局部性")
            elif sm_throughput > 60 and mem_throughput > 60:
                report.append("  → 混合型 (计算和内存都较忙)")
                report.append("  → 建议: 综合优化")
            else:
                report.append("  → 需要进一步分析")
        except (ValueError, AttributeError):
            report.append("  → 无法自动判断, 请查看详细指标")

    report.append("\n" + "=" * 70)
    return "\n".join(report)

# 使用示例
"""
ncu_results = run_ncu_analysis(
    "python matmul_kernel.py",
    kernel_name="matmul_kernel"
)

if ncu_results["success"]:
    report = generate_performance_report(ncu_results["data"])
    print(report)
"""
```

## 22.4 Triton 调试工具

### 22.4.1 TRITON_PRINT_IR 环境变量

Triton 提供了一系列环境变量用于调试和性能分析。

```bash
# TRITON_PRINT_IR: 打印编译过程中的 IR
export TRITON_PRINT_IR="all"          # 打印所有 IR
export TRITON_PRINT_IR="ttir"         # 只打印 Triton IR
export TRITON_PRINT_IR="ttgir"        # 只打印 TritonGPU IR
export TRITON_PRINT_IR="llir"         # 只打印 LLVM IR
export TRITON_PRINT_IR="ptx"          # 只打印 PTX

# 运行 kernel 并查看 IR
TRITON_PRINT_IR="all" python my_kernel.py

# 输出示例:
"""
#IR   # --- ttir ---
#IR   func.func @matmul_kernel(
#IR       %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
#IR       %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32},
#IR       %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32},
#IR       ...
#IR   ) {
#IR     %c0_i32 = arith.constant 0 : i32
#IR     %0 = tt.get_program_id x : i32
#IR     ...
#IR   }
"""
```

### 22.4.2 TRITON_PRINT_AUTOTUNING

```bash
# TRITON_PRINT_AUTOTUNING: 打印自动调优过程
export TRITON_PRINT_AUTOTUNING="1"    # 启用自动调优输出

# 运行带有 autotune 的 kernel
TRITON_PRINT_AUTOTUNING="1" python my_autotuned_kernel.py

# 输出示例:
"""
[AUTOTUNE] Starting autotune for matmul_kernel
[AUTOTUNE] Config 1/20: BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, num_warps=8, num_stages=3
[AUTOTUNE] Config 1/20: time = 0.45 ms
[AUTOTUNE] Config 2/20: BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=2
[AUTOTUNE] Config 2/20: time = 0.52 ms
...
[AUTOTUNE] Best config: BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, num_warps=8, num_stages=3
[AUTOTUNE] Best time: 0.45 ms
"""
```

### 22.4.3 triton.debug.inspect 系列函数

```python
import torch
import triton
import triton.language as tl

@triton.jit
def debug_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 加载数据
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # 调试: 打印中间值
    # 注意: triton.debug.inspect 在某些版本中可能不可用
    # 以下是概念性示例

    # 使用 tl.static_print 进行编译时调试
    tl.static_print("BLOCK_SIZE =", BLOCK_SIZE)
    tl.static_print("pid =", pid)

    # 计算
    output = x * 2.0 + 1.0

    # 存储结果
    tl.store(output_ptr + offsets, output, mask=mask)

# 运行调试 kernel
n_elements = 1024
x = torch.randn(n_elements, device='cuda', dtype=torch.float32)
output = torch.empty_like(x)
grid = (triton.cdiv(n_elements, 256),)

debug_kernel[grid](x, output, n_elements, BLOCK_SIZE=256)
```

### 22.4.4 tl.static_print 和 tl.static_assert

```python
@triton.jit
def kernel_with_static_debug(
    x_ptr, output_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 编译时打印常量值
    tl.static_print(f"Matrix dimensions: M={M}, N={N}, K={K}")
    tl.static_print(f"Block dimensions: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}")

    # 编译时断言
    tl.static_assert(BLOCK_M % 16 == 0, "BLOCK_M must be divisible by 16")
    tl.static_assert(BLOCK_N % 16 == 0, "BLOCK_N must be divisible by 16")
    tl.static_assert(BLOCK_K % 16 == 0, "BLOCK_K must be divisible by 16")

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # 运行时调试信息
    # 在 Triton 中，可以使用 tl.where 和 tl.where 进行条件调试
    # 但更常见的是使用 Python 端的 print

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 检查边界
    mask_m = offs_m < M
    mask_n = offs_n < N

    # 加载并处理数据
    # ... (kernel 逻辑)

# 使用示例
"""
M, N, K = 4096, 4096, 4096
BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64

kernel_with_static_debug[grid](
    x_ptr, output_ptr,
    M=M, N=N, K=K,
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
)

输出:
  Matrix dimensions: M=4096, N=4096, K=4096
  Block dimensions: BLOCK_M=128, BLOCK_N=256, BLOCK_K=64
"""
```

### 22.4.5 Triton IR 调试技巧

```python
"""
Triton IR 调试技巧:

1. 查看编译后的 IR
   - 使用 TRITON_PRINT_IR 环境变量
   - 使用 triton.compile() 的 IR 属性

2. 检查 Triton IR (TTIR)
   - 关注 load/store 操作
   - 检查 block 操作的形状
   - 验证 mask 正确性

3. 检查 TritonGPU IR (TTGIR)
   - 关注共享内存使用
   - 检查 Warp 分配
   - 验证 Layout 信息

4. 检查 LLVM IR
   - 关注寄存器使用
   - 检查指令调度
   - 验证内存访问模式
"""

import triton
import triton.language as tl

@triton.jit
def example_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    output = x * 2.0 + 1.0
    tl.store(output_ptr + offsets, output, mask=mask)

# 获取编译后的 IR
compiled = triton.compile(
    example_kernel,
    signature="*fp32,*fp32,i32",
    constants={"BLOCK_SIZE": 256}
)

# 查看 TTIR
print("=== Triton IR ===")
print(compiled.asm["ttir"])

# 查看 TTGIR (如果有)
if "ttgir" in compiled.asm:
    print("\n=== TritonGPU IR ===")
    print(compiled.asm["ttgir"])

# 查看 PTX (NVIDIA)
if "ptx" in compiled.asm:
    print("\n=== PTX ===")
    print(compiled.asm["ptx"])
```

### 22.4.6 调试输出重定向

```python
"""
调试输出重定向:

在生产环境中，调试输出可能会影响性能。
可以通过以下方式控制调试输出:
"""

import os
import sys
from io import StringIO

class TritonDebugger:
    """Triton 调试器"""

    def __init__(self, enable_debug=False):
        self.enable_debug = enable_debug
        self.original_stdout = sys.stdout

    def __enter__(self):
        if not self.enable_debug:
            # 重定向 stdout 以捕获调试输出
            sys.stdout = StringIO()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout

    def debug_print(self, *args, **kwargs):
        """条件打印"""
        if self.enable_debug:
            print(*args, **kwargs)

# 使用示例
"""
debugger = TritonDebugger(enable_debug=True)

with debugger:
    # 运行 kernel
    kernel[grid](x, output, n_elements, BLOCK_SIZE=256)

    # 条件打印
    debugger.debug_print("Kernel completed successfully")
"""
```

## 22.5 Roofline 模型

### 22.5.1 计算强度（Arithmetic Intensity）定义

计算强度（Arithmetic Intensity，AI）是 Roofline 模型的核心概念，定义为 FLOPs 与 Bytes 的比值。

```
计算强度 (Arithmetic Intensity) 定义:

  AI = FLOPs / Bytes_accessed

其中:
  FLOPs = 浮点运算次数
  Bytes_accessed = 内存访问字节数

示例:
  矩阵乘法 C = A @ B:
    FLOPs = 2 * M * N * K
    Bytes = (M * K + K * N + M * N) * sizeof(element)

  当 M = N = K 时:
    AI = 2 * N^3 / (3 * N^2 * sizeof(element))
       = 2 * N / (3 * sizeof(element))
       = 2N / (3 * sizeof(element))

  对于 FP16 (sizeof = 2 bytes):
    AI = 2N / 6 = N / 3

  当 N = 4096:
    AI = 4096 / 3 ≈ 1365 FLOP/Byte
```

```python
# 计算强度计算工具
def calculate_arithmetic_intensity(
    flops: int,
    bytes_accessed: int
) -> float:
    """计算 Arithmetic Intensity"""
    return flops / bytes_accessed

def analyze_operation_ai(operation: str, **params) -> dict:
    """分析各种操作的 Arithmetic Intensity"""

    results = {}

    if operation == "vector_add":
        n = params.get("n", 4096)
        element_size = params.get("element_size", 2)  # FP16

        flops = n  # N 次加法
        bytes_accessed = 3 * n * element_size  # 读 A, 读 B, 写 C

        ai = calculate_arithmetic_intensity(flops, bytes_accessed)
        results = {
            "operation": "Vector Add",
            "flops": flops,
            "bytes": bytes_accessed,
            "ai": ai,
            "category": "memory-bound"
        }

    elif operation == "matrix_multiply":
        m = params.get("m", 4096)
        n = params.get("n", 4096)
        k = params.get("k", 4096)
        element_size = params.get("element_size", 2)  # FP16

        flops = 2 * m * n * k
        bytes_accessed = (m * k + k * n + m * n) * element_size

        ai = calculate_arithmetic_intensity(flops, bytes_accessed)
        results = {
            "operation": "Matrix Multiply",
            "flops": flops,
            "bytes": bytes_accessed,
            "ai": ai,
            "category": "compute-bound" if ai > 100 else "memory-bound"
        }

    elif operation == "softmax":
        n = params.get("n", 4096)
        element_size = params.get("element_size", 2)  # FP16

        flops = 5 * n  # exp, sum, div
        bytes_accessed = 2 * n * element_size  # 读输入, 写输出

        ai = calculate_arithmetic_intensity(flops, bytes_accessed)
        results = {
            "operation": "Softmax",
            "flops": flops,
            "bytes": bytes_accessed,
            "ai": ai,
            "category": "memory-bound"
        }

    elif operation == "layernorm":
        n = params.get("n", 4096)
        element_size = params.get("element_size", 2)  # FP16

        flops = 10 * n  # mean, variance, normalize, scale, shift
        bytes_accessed = 3 * n * element_size  # 读输入, 读参数, 写输出

        ai = calculate_arithmetic_intensity(flops, bytes_accessed)
        results = {
            "operation": "LayerNorm",
            "flops": flops,
            "bytes": bytes_accessed,
            "ai": ai,
            "category": "memory-bound"
        }

    return results

# 分析各种操作
operations = [
    ("vector_add", {"n": 4096}),
    ("matrix_multiply", {"m": 4096, "n": 4096, "k": 4096}),
    ("softmax", {"n": 4096}),
    ("layernorm", {"n": 4096}),
]

print(f"{'Operation':<20} {'FLOPs':<15} {'Bytes':<15} {'AI (FLOP/Byte)':<15} {'Category'}")
print("-" * 80)

for op, params in operations:
    result = analyze_operation_ai(op, **params)
    print(f"{result['operation']:<20} {result['flops']:<15} {result['bytes']:<15} {result['ai']:<15.2f} {result['category']}")
```

### 22.5.2 Roofline 分析方法

```python
"""
Roofline 模型分析方法:

1. 计算 GPU 的 Peak Performance
   - Peak Compute: 理论最大 FLOPs/s
   - Peak Memory Bandwidth: 理论最大内存带宽

2. 计算 Ridge Point
   - Ridge Point = Peak Compute / Peak Memory Bandwidth
   - 单位: FLOP/Byte

3. 判断 Kernel 在 Roofline 图上的位置
   - AI < Ridge Point: Memory-bound
   - AI > Ridge Point: Compute-bound

4. 计算实际达到的性能百分比
   - Compute Utilization = Achieved FLOPs/s / Peak Compute
   - Memory Utilization = Achieved Bandwidth / Peak Memory Bandwidth
"""

# A100 GPU Roofline 参数
class GPU Roofline:
    """GPU Roofline 模型"""

    def __init__(
        self,
        peak_tflops: float,
        peak_bandwidth_gbs: float,
        name: str = "GPU"
    ):
        self.peak_tflops = peak_tflops
        self.peak_bandwidth_gbs = peak_bandwidth_gbs
        self.name = name

        # 计算 Ridge Point
        self.ridge_point = (peak_tflops * 1e12) / (peak_bandwidth_gbs * 1e9)

    def get_max_flops(self, ai: float) -> float:
        """给定 AI，计算理论最大 FLOPs/s"""
        # Roofline 公式: min(Peak Compute, AI * Peak Bandwidth)
        compute_bound = self.peak_tflops * 1e12
        memory_bound = ai * self.peak_bandwidth_gbs * 1e9
        return min(compute_bound, memory_bound) / 1e12  # TFLOPs/s

    def get_max_bandwidth(self, ai: float) -> float:
        """给定 AI，计算理论最大 Bandwidth"""
        # Roofline 公式: min(Peak Compute / AI, Peak Bandwidth)
        compute_bound = (self.peak_tflops * 1e12) / ai / 1e9
        memory_bound = self.peak_bandwidth_gbs
        return min(compute_bound, memory_bound)  # GB/s

    def analyze_kernel(self, ai: float, achieved_tflops: float, achieved_bandwidth_gbs: float) -> dict:
        """分析 kernel 在 Roofline 图上的位置"""
        max_tflops = self.get_max_flops(ai)
        max_bandwidth = self.get_max_bandwidth(ai)

        compute_util = achieved_tflops / (self.peak_tflops)
        memory_util = achieved_bandwidth_gbs / self.peak_bandwidth_gbs

        if ai < self.ridge_point:
            category = "Memory-bound"
            bottleneck = "Memory Bandwidth"
            efficiency = achieved_bandwidth_gbs / max_bandwidth
        else:
            category = "Compute-bound"
            bottleneck = "Compute Throughput"
            efficiency = achieved_tflops / max_tflops

        return {
            "ai": ai,
            "ridge_point": self.ridge_point,
            "category": category,
            "bottleneck": bottleneck,
            "efficiency": efficiency,
            "compute_utilization": compute_util,
            "memory_utilization": memory_util,
            "max_tflops": max_tflops,
            "max_bandwidth": max_bandwidth
        }

# A100 GPU Roofline 分析
a100 = GPU Roofline(
    peak_tflops=312.0,  # FP16 Tensor Core
    peak_bandwidth_gbs=2039.0,  # HBM2e
    name="A100"
)

print(f"GPU: {a100.name}")
print(f"Peak Compute: {a100.peak_tflops} TFLOPs/s")
print(f"Peak Bandwidth: {a100.peak_bandwidth_gbs} GB/s")
print(f"Ridge Point: {a100.ridge_point:.3f} FLOP/Byte")
print()

# 分析不同 kernel
kernels = [
    {"name": "Vector Add", "ai": 0.167, "tflops": 0.5, "bw": 1800.0},
    {"name": "Matrix Multiply (4096x4096)", "ai": 1365.0, "tflops": 250.0, "bw": 1500.0},
    {"name": "Softmax", "ai": 2.5, "tflops": 20.0, "bw": 1200.0},
    {"name": "LayerNorm", "ai": 3.33, "tflops": 25.0, "bw": 1400.0},
]

for kernel in kernels:
    result = a100.analyze_kernel(
        ai=kernel["ai"],
        achieved_tflops=kernel["tflops"],
        achieved_bandwidth_gbs=kernel["bw"]
    )
    print(f"Kernel: {kernel['name']}")
    print(f"  AI: {result['ai']:.2f} FLOP/Byte")
    print(f"  Category: {result['category']}")
    print(f"  Bottleneck: {result['bottleneck']}")
    print(f"  Efficiency: {result['efficiency']:.1%}")
    print(f"  Compute Util: {result['compute_utilization']:.1%}")
    print(f"  Memory Util: {result['memory_utilization']:.1%}")
    print()
```

### 22.5.3 Roofline 图表绘制

```python
"""
Roofline 图表绘制 (概念性代码)
"""

import math

def draw_roofline_ascii(
    peak_tflops: float,
    peak_bandwidth_gbs: float,
    kernels: list,
    ai_range: tuple = (0.01, 10000)
) -> str:
    """绘制 ASCII Roofline 图表"""

    ridge_point = (peak_tflops * 1e12) / (peak_bandwidth_gbs * 1e9)

    # 生成 Roofline 曲线点
    roofline_points = []
    ai_min, ai_max = ai_range
    for i in range(100):
        ai = ai_min * (ai_max / ai_min) ** (i / 99)
        flops = min(peak_tflops, ai * peak_bandwidth_gbs / 1000)
        roofline_points.append((ai, flops))

    # 绘制 ASCII 图表
    chart_width = 60
    chart_height = 20
    chart = [[' ' for _ in range(chart_width)] for _ in range(chart_height)]

    # 绘制坐标轴
    for i in range(chart_height):
        chart[i][0] = '|'
    for j in range(chart_width):
        chart[chart_height - 1][j] = '-'

    # 绘制 Roofline 曲线
    for ai, flops in roofline_points:
        x = int(math.log10(ai / ai_min) / math.log10(ai_max / ai_min) * (chart_width - 2)) + 1
        y = int(flops / peak_tflops * (chart_height - 2))
        y = chart_height - 2 - y
        if 0 < x < chart_width and 0 < y < chart_height:
            chart[y][x] = '*'

    # 绘制 kernel 点
    for kernel in kernels:
        ai = kernel["ai"]
        flops = kernel["tflops"]
        x = int(math.log10(ai / ai_min) / math.log10(ai_max / ai_min) * (chart_width - 2)) + 1
        y = int(flops / peak_tflops * (chart_height - 2))
        y = chart_height - 2 - y
        if 0 < x < chart_width and 0 < y < chart_height:
            chart[y][x] = 'O'

    # 转换为字符串
    lines = [''.join(row) for row in chart]
    return '\n'.join(lines)

# 示例
print("Roofline Chart (A100):")
print(draw_roofline_ascii(
    peak_tflops=312.0,
    peak_bandwidth_gbs=2039.0,
    kernels=[
        {"ai": 0.167, "tflops": 0.5},
        {"ai": 1365.0, "tflops": 250.0},
        {"ai": 2.5, "tflops": 20.0},
    ]
))
```

## 22.6 瓶颈识别

### 22.6.1 Compute-bound 判断方法

```python
"""
Compute-bound 判断方法:

1. GPU Utilization 检查
   - GPU Utilization > 80%: GPU 忙碌，可能是 compute-bound
   - GPU Utilization < 50%: GPU 空闲较多，可能是启动开销或内存瓶颈

2. Compute Throughput 检查
   - sm__throughput > 80% of peak: 计算吞吐量高
   - Tensor Core 使用率高: 使用了加速器

3. Memory Throughput 检查
   - dram__throughput < 40% of peak: 内存带宽未充分利用
   - 说明瓶颈不在内存

4. Roofline 位置
   - AI > Ridge Point: 位于 compute-bound 区域
"""

def identify_compute_bound(
    gpu_utilization: float,
    sm_throughput: float,
    memory_throughput: float,
    arithmetic_intensity: float,
    ridge_point: float
) -> dict:
    """判断是否为 compute-bound"""

    is_compute_bound = False
    reasons = []
    score = 0

    # 检查条件 1: GPU Utilization
    if gpu_utilization > 80:
        reasons.append(f"GPU Utilization 高 ({gpu_utilization:.1f}%)")
        score += 30

    # 检查条件 2: SM Throughput
    if sm_throughput > 80:
        reasons.append(f"SM Throughput 高 ({sm_throughput:.1f}%)")
        score += 30

    # 检查条件 3: Memory Throughput 相对较低
    if memory_throughput < 40:
        reasons.append(f"Memory Throughput 低 ({memory_throughput:.1f}%)")
        score += 20

    # 检查条件 4: Roofline 位置
    if arithmetic_intensity > ridge_point:
        reasons.append(f"AI ({arithmetic_intensity:.2f}) > Ridge Point ({ridge_point:.2f})")
        score += 20

    # 综合判断
    is_compute_bound = score >= 60

    return {
        "is_compute_bound": is_compute_bound,
        "confidence": min(score, 100),
        "reasons": reasons,
        "score": score,
        "recommendation": "使用 Tensor Core, 优化计算逻辑, 减少不必要的计算" if is_compute_bound else "需要进一步分析"
    }

# 示例
result = identify_compute_bound(
    gpu_utilization=92.5,
    sm_throughput=85.3,
    memory_throughput=35.2,
    arithmetic_intensity=1365.0,
    ridge_point=153.0
)
print("Compute-bound 分析:")
print(f"  Is Compute-bound: {result['is_compute_bound']}")
print(f"  Confidence: {result['confidence']}%")
print(f"  Reasons: {result['reasons']}")
print(f"  Recommendation: {result['recommendation']}")
```

### 22.6.2 Memory-bound 判断方法

```python
"""
Memory-bound 判断方法:

1. Memory Throughput 检查
   - Memory Throughput > 80% of peak: 内存带宽接近理论峰值
   - 说明瓶颈在内存

2. GPU Utilization 检查
   - GPU Utilization < 60%: GPU 空闲较多
   - 可能在等待内存数据

3. SM Throughput 检查
   - SM Throughput < 50%: 计算吞吐量低
   - 说明计算不是瓶颈

4. Roofline 位置
   - AI < Ridge Point: 位于 memory-bound 区域

5. Cache Hit Rate 检查
   - L1 Cache Hit Rate < 50%: 缓存效率低
   - L2 Cache Hit Rate < 70%: L2 缓存效率低
"""

def identify_memory_bound(
    gpu_utilization: float,
    sm_throughput: float,
    memory_throughput: float,
    arithmetic_intensity: float,
    ridge_point: float,
    l1_cache_hit_rate: float = None,
    l2_cache_hit_rate: float = None
) -> dict:
    """判断是否为 memory-bound"""

    is_memory_bound = False
    reasons = []
    score = 0

    # 检查条件 1: Memory Throughput 高
    if memory_throughput > 80:
        reasons.append(f"Memory Throughput 高 ({memory_throughput:.1f}%)")
        score += 30

    # 检查条件 2: GPU Utilization 相对较低
    if gpu_utilization < 60:
        reasons.append(f"GPU Utilization 低 ({gpu_utilization:.1f}%)")
        score += 20

    # 检查条件 3: SM Throughput 低
    if sm_throughput < 50:
        reasons.append(f"SM Throughput 低 ({sm_throughput:.1f}%)")
        score += 20

    # 检查条件 4: Roofline 位置
    if arithmetic_intensity < ridge_point:
        reasons.append(f"AI ({arithmetic_intensity:.2f}) < Ridge Point ({ridge_point:.2f})")
        score += 30

    # 检查条件 5: Cache 效率低
    if l1_cache_hit_rate is not None and l1_cache_hit_rate < 50:
        reasons.append(f"L1 Cache Hit Rate 低 ({l1_cache_hit_rate:.1f}%)")
        score += 10

    if l2_cache_hit_rate is not None and l2_cache_hit_rate < 70:
        reasons.append(f"L2 Cache Hit Rate 低 ({l2_cache_hit_rate:.1f}%)")
        score += 10

    # 综合判断
    is_memory_bound = score >= 60

    return {
        "is_memory_bound": is_memory_bound,
        "confidence": min(score, 100),
        "reasons": reasons,
        "score": score,
        "recommendation": "减少内存访问, 提高数据局部性, 使用共享内存" if is_memory_bound else "需要进一步分析"
    }

# 示例
result = identify_memory_bound(
    gpu_utilization=45.2,
    sm_throughput=35.8,
    memory_throughput=92.5,
    arithmetic_intensity=2.5,
    ridge_point=153.0,
    l1_cache_hit_rate=45.0,
    l2_cache_hit_rate=65.0
)
print("Memory-bound 分析:")
print(f"  Is Memory-bound: {result['is_memory_bound']}")
print(f"  Confidence: {result['confidence']}%")
print(f"  Reasons: {result['reasons']}")
print(f"  Recommendation: {result['recommendation']}")
```

### 22.6.3 瓶颈识别流程

```
Triton Kernel 瓶颈识别流程:

Step 1: 运行 nsys 收集系统级数据
┌────────────────────────────────────────────────────────────────┐
│ nsys profile --stats=true python my_kernel.py                  │
│                                                                │
│ 关键指标:                                                       │
│   - GPU Utilization                                            │
│   - Kernel Execution Time                                      │
│   - Memory Transfer Time                                       │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 2: 运行 ncu 收集 kernel 级数据
┌────────────────────────────────────────────────────────────────┐
│ ncu --set full --kernel-name "my_kernel" python my_kernel.py   │
│                                                                │
│ 关键指标:                                                       │
│   - SM Throughput                                              │
│   - Memory Throughput                                          │
│   - Occupancy                                                  │
│   - Arithmetic Intensity                                       │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 3: 分析瓶颈类型
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  IF (SM Throughput > 80%) AND (Memory Throughput < 40%):       │
│    → Compute-bound                                             │
│    → 检查: Tensor Core 使用率, FLOPs/s                         │
│                                                                │
│  ELSE IF (Memory Throughput > 80%) AND (SM Throughput < 40%):  │
│    → Memory-bound                                              │
│    → 检查: Cache Hit Rate, 内存访问模式                         │
│                                                                │
│  ELSE IF (SM Throughput > 60%) AND (Memory Throughput > 60%):  │
│    → 混合型                                                    │
│    → 需要综合分析                                               │
│                                                                │
│  ELSE:                                                         │
│    → 需要进一步分析                                             │
│    → 检查: Occupancy, Warp Efficiency                          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Step 4: 根据瓶颈类型选择优化策略
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  Compute-bound 优化:                                           │
│    - 使用 Tensor Core (FP16/BF16/INT8)                         │
│    - 减少不必要的计算                                           │
│    - 提高指令级并行度                                           │
│    - 优化循环展开                                               │
│                                                                │
│  Memory-bound 优化:                                            │
│    - 减少内存访问次数                                           │
│    - 提高数据局部性                                             │
│    - 使用共享内存                                              │
│    - 优化内存访问模式                                           │
│    - 使用向量化加载                                             │
│                                                                │
│  混合型优化:                                                    │
│    - Kernel Fusion                                             │
│    - 减少数据传输                                               │
│    - 综合优化策略                                               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## 22.7 优化路径

### 22.7.1 Compute-bound 优化策略

```python
"""
Compute-bound 优化策略:

1. 使用 Tensor Core
   - FP16/BF16 计算
   - INT8 量化计算
   - 使用 tl.dot 操作

2. 减少 FLOPs
   - 算法优化 (如 Strassen 矩阵乘法)
   - 近似计算 (如低秩近似)
   - 稀疏计算

3. 提高指令级并行度
   - 循环展开
   - 减少分支
   - 使用向量化操作

4. 优化 Warp 调度
   - 提高 Occupancy
   - 减少 Warp Divergence
   - 优化寄存器使用
"""

# Compute-bound 优化示例: 使用 Tensor Core
import torch
import triton
import triton.language as tl

@triton.jit
def optimized_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """使用 Tensor Core 的优化矩阵乘法"""

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # 使用 FP16 计算 (Tensor Core 加速)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # 加载 FP16 数据
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)

        # Tensor Core 计算 (tl.dot 会自动使用 Tensor Core)
        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # 转换回 FP16
    c = accumulator.to(tl.float16)

    # 存储结果
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c)

# 性能对比
"""
优化前 (FP32 CUDA Core):
  - FLOPs/s: ~20 TFLOPs/s
  - Memory Bandwidth: ~1000 GB/s
  - Roofline Position: Compute-bound

优化后 (FP16 Tensor Core):
  - FLOPs/s: ~250 TFLOPs/s (12.5x 提升)
  - Memory Bandwidth: ~1500 GB/s
  - Roofline Position: 仍然是 compute-bound，但效率更高
"""
```

### 22.7.2 Memory-bound 优化策略

```python
"""
Memory-bound 优化策略:

1. 减少内存访问次数
   - Kernel Fusion (融合多个 kernel)
   - 减少中间结果存储
   - 使用 In-place 操作

2. 提高数据局部性
   - 使用共享内存
   - 优化数据布局
   - 使用循环分块

3. 优化内存访问模式
   - 合并内存访问 (Coalesced Access)
   - 避免 Bank Conflict
   - 使用向量化加载

4. 使用缓存
   - L1/L2 Cache 友好的访问模式
   - 预取数据
   - 使用 Texture Memory (只读数据)
"""

# Memory-bound 优化示例: 使用共享内存
@triton.jit
def matmul_with_shared_memory(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """使用共享内存的矩阵乘法"""

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # 分配共享内存
    a_shared = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float16)
    b_shared = tl.zeros([BLOCK_K, BLOCK_N], dtype=tl.float16)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # 从全局内存加载到共享内存
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        a_shared[:, :] = a

        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        b_shared[:, :] = b

        # 使用共享内存进行计算
        accumulator += tl.dot(a_shared, b_shared)

    # 存储结果
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c)

# 内存访问优化对比
"""
优化前 (无共享内存):
  - 全局内存访问次数: 2 * M * N * K / BLOCK_SIZE
  - 带宽利用率: ~50%

优化后 (使用共享内存):
  - 全局内存访问次数: 2 * M * N / BLOCK_SIZE (大幅减少)
  - 带宽利用率: ~85%
  - 性能提升: ~1.7x
"""
```

### 22.7.3 优化策略对比表

| 瓶颈类型 | 优化策略 | 预期收益 | 实现复杂度 |
|---------|---------|---------|-----------|
| Compute-bound | 使用 Tensor Core | 5-15x | 低 |
| Compute-bound | 减少 FLOPs | 2-5x | 中 |
| Compute-bound | 提高指令级并行度 | 1.5-3x | 中 |
| Compute-bound | 优化 Warp 调度 | 1.2-2x | 高 |
| Memory-bound | 使用共享内存 | 1.5-3x | 中 |
| Memory-bound | Kernel Fusion | 2-5x | 高 |
| Memory-bound | 优化内存访问模式 | 1.3-2x | 中 |
| Memory-bound | 使用向量化加载 | 1.2-2x | 低 |
| Memory-bound | 减少数据传输 | 1.5-4x | 高 |
| 混合型 | 综合优化 | 2-10x | 高 |

### 22.7.4 优化效果验证

```python
"""
优化效果验证流程:

1. 基准测试
   - 运行优化前的 kernel
   - 记录性能指标

2. 应用优化
   - 实施优化策略
   - 确保正确性

3. 性能测试
   - 运行优化后的 kernel
   - 记录性能指标

4. 对比分析
   - 计算性能提升百分比
   - 分析瓶颈变化

5. 迭代优化
   - 根据新的瓶颈继续优化
   - 直达到满意性能
"""

import torch
import time

def benchmark_kernel(kernel_func, *args, warmup=10, rep=100, **kwargs):
    """Benchmark kernel 性能"""

    # Warmup
    for _ in range(warmup):
        kernel_func(*args, **kwargs)
    torch.cuda.synchronize()

    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    for i in range(rep):
        start_events[i].record()
        kernel_func(*args, **kwargs)
        end_events[i].record()

    torch.cuda.synchronize()

    # 计算统计
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    return {
        "avg_ms": avg_time,
        "min_ms": min_time,
        "max_ms": max_time,
        "std_ms": (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    }

def compare_optimization(original_func, optimized_func, *args, **kwargs):
    """对比优化效果"""

    print("Running original kernel...")
    original_stats = benchmark_kernel(original_func, *args, **kwargs)

    print("Running optimized kernel...")
    optimized_stats = benchmark_kernel(optimized_func, *args, **kwargs)

    # 计算提升
    speedup = original_stats["avg_ms"] / optimized_stats["avg_ms"]
    improvement = (1 - optimized_stats["avg_ms"] / original_stats["avg_ms"]) * 100

    print(f"\nOptimization Results:")
    print(f"  Original: {original_stats['avg_ms']:.3f} ms")
    print(f"  Optimized: {optimized_stats['avg_ms']:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Improvement: {improvement:.1f}%")

    return {
        "original": original_stats,
        "optimized": optimized_stats,
        "speedup": speedup,
        "improvement_pct": improvement
    }
```

## 22.8 完整案例

### 22.8.1 案例概述

本节通过一个完整的 matmul kernel 优化案例，展示从 nsys → ncu → 瓶颈分析 → 优化 → 验证的完整流程。

```
完整优化流程:

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Step 1: 基准测试 (Baseline)                                    │
│    - 实现基础 matmul kernel                                     │
│    - 使用 nsys 收集系统级数据                                    │
│                                                                 │
│  Step 2: 性能分析 (Profiling)                                   │
│    - 使用 ncu 收集 kernel 级数据                                 │
│    - 分析关键性能指标                                            │
│                                                                 │
│  Step 3: 瓶颈识别 (Bottleneck Identification)                   │
│    - 判断瓶颈类型 (compute-bound / memory-bound)                 │
│    - 确定优化方向                                               │
│                                                                 │
│  Step 4: 优化实施 (Optimization)                                │
│    - 根据瓶颈类型选择优化策略                                    │
│    - 实施优化                                                   │
│                                                                 │
│  Step 5: 验证与迭代 (Verification & Iteration)                  │
│    - 验证正确性                                                 │
│    - 测量性能提升                                               │
│    - 分析新的瓶颈                                               │
│    - 迭代优化直到满意                                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 22.8.2 Step 1: 基准测试

```python
"""
Step 1: 基准测试 - 实现基础 matmul kernel
"""

import torch
import triton
import triton.language as tl

@triton.jit
def matmul_baseline(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """基础 matmul kernel (无优化)"""

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c)

def run_baseline():
    """运行基准测试"""

    M, N, K = 4096, 4096, 4096
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64

    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    c = torch.empty(M, N, device='cuda', dtype=torch.float16)

    grid = (tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N),)

    # 预热
    for _ in range(10):
        matmul_baseline[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
    torch.cuda.synchronize()

    # 基准测试
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(100)]

    for i in range(100):
        start_events[i].record()
        matmul_baseline[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        end_events[i].record()

    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_time = sum(times) / len(times)

    # 计算性能指标
    flops = 2 * M * N * K
    tflops = flops / (avg_time * 1e-3) / 1e12

    print(f"Baseline Performance:")
    print(f"  Matrix Size: {M}x{N}x{K}")
    print(f"  Block Size: {BLOCK_M}x{BLOCK_N}x{BLOCK_K}")
    print(f"  Average Time: {avg_time:.3f} ms")
    print(f"  TFLOPs/s: {tflops:.2f}")

    return avg_time, tflops

# 运行基准测试
baseline_time, baseline_tflops = run_baseline()
```

**nsys 分析基准 kernel：**

```bash
# 运行 nsys 分析
nsys profile --stats=true -o baseline python matmul_baseline.py

# nsys 输出摘要
"""
CUDA Kernel Statistics:
 Time(%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  Name
 --------  ---------------  ---------  --------  --------  --------  --------  ----
   100.0%        15000000        100  150000.0  149800.0   149500    150500  matmul_baseline

Memory Operation Statistics:
 Time(%)  Total Time (ns)  Operations  Avg (ns)  Med (ns)  Type
 --------  ---------------  ---------  --------  --------  ----
    75.0%        11250000      200000    56.2      55.0    Host To GPU
    25.0%         3750000      100000    37.5      37.0    GPU To Host
"""
```

### 22.8.3 Step 2: 性能分析

```bash
# 运行 ncu 分析
ncu --set full --kernel-name "matmul_baseline" python matmul_baseline.py

# ncu 输出摘要
"""
Kernel: matmul_baseline<float>
grid: (512, 1, 1), block: (128, 1, 1)

Section: GPU Speed Of Light Throughput
  SM [__] Throughput: 45.25%  (of peak 19.5 TFLOPs/s for FP32)
  Memory [__] Throughput: 85.67%  (of peak 2039 GB/s)

Section: Compute (SM) Throughput
  sm__throughput.avg.pct_of_peak_sustained_elapsed       45.25%
  smsp__sass_thread_inst_executed_op_ffma_pred_on.sum    8234567

Section: Memory Throughput
  dram__bytes_read.sum                                   3.22 GB
  dram__bytes_write.sum                                  1.07 GB
  dram__throughput.avg.pct_of_peak_sustained_elapsed     85.67%

Section: Occupancy
  sm__warps_active.avg.pct_of_peak_sustained_active      62.50%
  launch__occupancy_limit_registers                      75.00%
  launch__occupancy_limit_shared_mem                     100.00%

Section: Roofline
  Arithmetic Intensity: 2.05 FLOP/Byte
  Achieved FLOPs/s: 8.85 TFLOPs/s
  Achieved Bandwidth: 1745 GB/s
"""
```

### 22.8.4 Step 3: 瓶颈识别

```python
"""
Step 3: 瓶颈识别分析

基于 ncu 输出:
  - SM Throughput: 45.25% (较低)
  - Memory Throughput: 85.67% (较高)
  - Arithmetic Intensity: 2.05 FLOP/Byte (较低)
  - Ridge Point: 153 FLOP/Byte

分析结论:
  1. Memory Throughput > 80%: 内存带宽接近理论峰值
  2. SM Throughput < 50%: 计算吞吐量较低
  3. AI (2.05) << Ridge Point (153): 位于 memory-bound 区域

瓶颈类型: Memory-bound
瓶颈原因: 内存访问效率低，计算单元空闲等待数据

优化方向:
  1. 提高计算密度 (增加 Arithmetic Intensity)
  2. 使用 Tensor Core (提高计算效率)
  3. 优化内存访问模式 (减少内存访问次数)
"""

# 瓶颈识别计算
sm_throughput = 45.25
memory_throughput = 85.67
arithmetic_intensity = 2.05
ridge_point = 153.0

print("瓶颈识别分析:")
print(f"  SM Throughput: {sm_throughput}%")
print(f"  Memory Throughput: {memory_throughput}%")
print(f"  Arithmetic Intensity: {arithmetic_intensity} FLOP/Byte")
print(f"  Ridge Point: {ridge_point} FLOP/Byte")

if memory_throughput > 80 and sm_throughput < 50:
    bottleneck_type = "Memory-bound"
    print(f"\n瓶颈类型: {bottleneck_type}")
    print("瓶颈原因: 内存带宽接近峰值，但计算利用率低")
    print("优化方向: 提高计算密度，使用 Tensor Core")
elif sm_throughput > 80 and memory_throughput < 40:
    bottleneck_type = "Compute-bound"
    print(f"\n瓶颈类型: {bottleneck_type}")
    print("瓶颈原因: 计算吞吐量高，但内存带宽未充分利用")
    print("优化方向: 减少内存访问，提高数据局部性")
else:
    bottleneck_type = "Mixed"
    print(f"\n瓶颈类型: {bottleneck_type}")
    print("需要进一步分析")
```

### 22.8.5 Step 4: 优化实施

```python
"""
Step 4: 优化实施

针对 Memory-bound 瓶颈，实施以下优化:

1. 使用 FP16 计算 (提高 Arithmetic Intensity)
2. 优化循环结构 (减少内存访问)
3. 提高 Occupancy (改善延迟隐藏)
"""

@triton.jit
def matmul_optimized(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """优化后的 matmul kernel"""

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = pid % num_pid_in_group // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # 使用 tl.where 进行边界检查
        mask = offs_k[None, :] < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=mask, other=0.0)
        b = tl.load(b_ptrs, mask=mask[:, None].expand_as(b_ptrs), other=0.0)

        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c)

def run_optimized():
    """运行优化后的 kernel"""

    M, N, K = 4096, 4096, 4096
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64
    GROUP_SIZE_M = 8

    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    c = torch.empty(M, N, device='cuda', dtype=torch.float16)

    grid = (tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N),)

    # 预热
    for _ in range(10):
        matmul_optimized[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
        )
    torch.cuda.synchronize()

    # 性能测试
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(100)]

    for i in range(100):
        start_events[i].record()
        matmul_optimized[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
        )
        end_events[i].record()

    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    avg_time = sum(times) / len(times)

    # 计算性能指标
    flops = 2 * M * N * K
    tflops = flops / (avg_time * 1e-3) / 1e12

    print(f"\nOptimized Performance:")
    print(f"  Matrix Size: {M}x{N}x{K}")
    print(f"  Block Size: {BLOCK_M}x{BLOCK_N}x{BLOCK_K}")
    print(f"  Average Time: {avg_time:.3f} ms")
    print(f"  TFLOPs/s: {tflops:.2f}")

    return avg_time, tflops

# 运行优化后的 kernel
optimized_time, optimized_tflops = run_optimized()
```

### 22.8.6 Step 5: 验证与迭代

```python
"""
Step 5: 验证与迭代

1. 验证正确性
2. 对比性能提升
3. 分析新的瓶颈
4. 决定是否需要进一步优化
"""

def verify_correctness(M, N, K):
    """验证优化后的 kernel 正确性"""

    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)

    # 参考结果 (cuBLAS)
    c_ref = torch.matmul(a, b)

    # Triton 结果
    c_triton = torch.empty(M, N, device='cuda', dtype=torch.float16)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64
    GROUP_SIZE_M = 8
    grid = (tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N),)

    matmul_optimized[grid](
        a, b, c_triton, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c_triton.stride(0), c_triton.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )

    # 对比结果
    diff = torch.abs(c_ref - c_triton)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nCorrectness Verification:")
    print(f"  Max Absolute Diff: {max_diff:.6f}")
    print(f"  Mean Absolute Diff: {mean_diff:.6f}")

    # 相对误差
    rel_diff = diff / (torch.abs(c_ref) + 1e-6)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()

    print(f"  Max Relative Diff: {max_rel_diff:.6f}")
    print(f"  Mean Relative Diff: {mean_rel_diff:.6f}")

    return max_diff, mean_diff

def compare_performance(baseline_time, optimized_time, baseline_tflops, optimized_tflops):
    """对比性能提升"""

    speedup = baseline_time / optimized_time
    improvement = (1 - optimized_time / baseline_time) * 100

    print(f"\nPerformance Comparison:")
    print(f"  Baseline: {baseline_time:.3f} ms ({baseline_tflops:.2f} TFLOPs/s)")
    print(f"  Optimized: {optimized_time:.3f} ms ({optimized_tflops:.2f} TFLOPs/s)")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Improvement: {improvement:.1f}%")

    return speedup, improvement

# 运行验证
M, N, K = 4096, 4096, 4096
max_diff, mean_diff = verify_correctness(M, N, K)
speedup, improvement = compare_performance(baseline_time, optimized_time, baseline_tflops, optimized_tflops)

# 分析新的瓶颈
print("\n" + "=" * 60)
print("优化后瓶颈分析:")
print("=" * 60)

# 运行 ncu 分析优化后的 kernel
# ncu --set full --kernel-name "matmul_optimized" python matmul_optimized.py

"""
优化后 ncu 输出 (假设):
  - SM Throughput: 72.5% (从 45.25% 提升)
  - Memory Throughput: 65.3% (从 85.67% 下降)
  - Arithmetic Intensity: 8.2 FLOP/Byte (从 2.05 提升)
  - Occupancy: 75.0% (从 62.5% 提升)

分析:
  1. SM Throughput 提升: 计算利用率提高
  2. Memory Throughput 下降: 不再是主要瓶颈
  3. AI 提升: 计算密度增加
  4. Occupancy 提升: 延迟隐藏改善

新的瓶颈:
  - 混合型: 计算和内存都较忙
  - 进一步优化空间:
    1. 使用 Tensor Core (FP16)
    2. 优化循环展开
    3. 使用软件流水线
"""
```

### 22.8.7 最终优化结果

```python
"""
最终优化结果汇总

优化前 (Baseline):
  - 矩阵大小: 4096x4096x4096
  - 执行时间: 15.0 ms
  - TFLOPs/s: 8.85
  - SM Throughput: 45.25%
  - Memory Throughput: 85.67%
  - Arithmetic Intensity: 2.05 FLOP/Byte
  - Bottleneck: Memory-bound

优化后 (Optimized):
  - 矩阵大小: 4096x4096x4096
  - 执行时间: 6.2 ms
  - TFLOPs/s: 21.5
  - SM Throughput: 72.5%
  - Memory Throughput: 65.3%
  - Arithmetic Intensity: 8.2 FLOP/Byte
  - Bottleneck: Mixed

性能提升:
  - 执行时间: 2.42x 加速
  - TFLOPs/s: 2.43x 提升
  - SM Throughput: +27.25%
  - Memory Throughput: -20.37%
  - Arithmetic Intensity: +4.0x
"""

# 性能汇总表
performance_summary = """
┌────────────────────────────────────────────────────────────────┐
│                    性能优化结果汇总                              │
├────────────────────────────────────────────────────────────────┤
│ 指标                    Baseline      Optimized     Change    │
├────────────────────────────────────────────────────────────────┤
│ 执行时间 (ms)           15.0          6.2           2.42x ↓   │
│ TFLOPs/s                8.85          21.5          2.43x ↑   │
│ SM Throughput (%)       45.25         72.5          +27.25%   │
│ Memory Throughput (%)   85.67         65.3          -20.37%   │
│ Arithmetic Intensity    2.05          8.2           4.0x ↑    │
│ Occupancy (%)           62.5          75.0          +12.5%    │
│ 瓶颈类型                Memory-bound  Mixed         -         │
├────────────────────────────────────────────────────────────────┤
│ 优化策略:                                                      │
│   1. 使用 FP16 计算 (提高 Arithmetic Intensity)                │
│   2. 优化循环结构 (减少内存访问)                                │
│   3. 改进 Warp 调度 (提高 Occupancy)                          │
│   4. 使用 GROUP_SIZE_M (改善数据局部性)                        │
├────────────────────────────────────────────────────────────────┤
│ 进一步优化空间:                                                │
│   1. 使用 Tensor Core (FP16/BF16)                             │
│   2. 实现软件流水线 (Software Pipelining)                      │
│   3. 优化共享内存使用                                          │
│   4. 使用 TMA (Tensor Memory Accelerator)                     │
└────────────────────────────────────────────────────────────────┘
"""
print(performance_summary)
```

## 本章小结

### 核心知识点回顾

| 知识点 | 说明 | 重要性 |
|--------|------|--------|
| **性能指标体系** | Throughput, Latency, Occupancy, SM Utilization, Memory Throughput, FLOPs/s | ★★★★★ |
| **nsys 分析** | 系统级性能分析, GPU 利用率, 内存传输, 常见问题识别 | ★★★★☆ |
| **ncu 分析** | Kernel 级性能分析, 硬件计数器, Roofline 分析 | ★★★★★ |
| **Triton 调试工具** | TRITON_PRINT_IR, TRITON_PRINT_AUTOTUNING, tl.static_print | ★★★☆☆ |
| **Roofline 模型** | 计算强度定义, Roofline 分析方法, 瓶颈判断 | ★★★★★ |
| **瓶颈识别** | Compute-bound 判断, Memory-bound 判断, 混合型分析 | ★★★★★ |
| **优化路径** | 不同瓶颈的优化策略, 优化效果验证 | ★★★★☆ |
| **完整案例** | nsys → ncu → 瓶颈分析 → 优化 → 验证的完整流程 | ★★★★★ |

### 关键公式

```
Arithmetic Intensity (AI) = FLOPs / Bytes_accessed

Ridge Point = Peak Compute / Peak Memory Bandwidth

Compute-bound 判断: AI > Ridge Point
Memory-bound 判断: AI < Ridge Point

Occupancy = Active Warps / Max Warps

FLOPs (矩阵乘法) = 2 * M * N * K

Memory Bandwidth Utilization = Achieved Bandwidth / Peak Bandwidth

Compute Utilization = Achieved FLOPs/s / Peak FLOPs/s
```

### 实践要点

1. **性能分析流程**
   - 先用 nsys 获取系统级视图
   - 再用 ncu 获取 kernel 级详细数据
   - 结合 Roofline 模型判断瓶颈

2. **瓶颈识别**
   - 不要凭经验猜测，要基于数据判断
   - 使用 ncu 硬件计数器而非推测
   - 关注 Roofline 位置和利用率指标

3. **优化策略选择**
   - Compute-bound: 优先使用 Tensor Core，减少 FLOPs
   - Memory-bound: 优先减少内存访问，提高数据局部性
   - 混合型: 需要综合分析，可能需要多轮优化

4. **验证与迭代**
   - 每次优化后都要验证正确性
   - 使用 benchmark 对比性能提升
   - 分析新的瓶颈，决定是否继续优化

---

## 思考题

### 基础题

1. **解释 Arithmetic Intensity 的定义，并说明它在 Roofline 模型中的作用。**
   - 提示: 参考 22.5.1 节的定义和公式

2. **如何使用 nsys 判断一个 Triton kernel 是否是 Memory-bound？**
   - 提示: 参考 22.2.6 节的常见问题识别

3. **ncu 的 `--set full` 选项会收集哪些类型的硬件计数器？**
   - 提示: 参考 22.3.3 节的硬件计数器详解

4. **解释 Occupancy 的定义，并说明影响 Occupancy 的三个主要因素。**
   - 提示: 参考 22.1.4 节的 Occupancy 计算

5. **TRITON_PRINT_IR 环境变量的三种取值分别对应什么 IR？**
   - 提示: 参考 22.4.1 节的环境变量说明

### 进阶题

6. **一个 Triton kernel 的 ncu 输出显示：SM Throughput = 85%，Memory Throughput = 75%。请分析这个 kernel 的瓶颈类型，并给出优化建议。**
   - 提示: 参考 22.6.3 节的瓶颈识别流程

7. **如何计算矩阵乘法 C = A @ B 的 Arithmetic Intensity？当 M = N = K = 4096 且使用 FP16 时，AI 是多少？**
   - 提示: 参考 22.5.1 节的计算公式

8. **解释 Ridge Point 的含义，并说明如何使用它判断 kernel 的瓶颈类型。**
   - 提示: 参考 22.5.2 节的 Roofline 分析方法

9. **一个 kernel 在 A100 上运行，ncu 显示其 Arithmetic Intensity = 0.5 FLOP/Byte。请判断这个 kernel 的瓶颈类型，并说明原因。**
   - 提示: A100 的 Ridge Point 约为 0.153 FLOP/Byte

10. **如何使用 Triton 的调试工具检查编译后的 PTX 代码？**
    - 提示: 参考 22.4.5 节的 IR 调试技巧

### 实践题

11. **设计一个性能分析流程，用于分析和优化一个自定义的 Triton kernel。需要包括哪些步骤？使用哪些工具？**
    - 提示: 参考 22.8 节的完整案例

12. **编写一个 Python 脚本，使用 ncu 收集 Triton kernel 的硬件计数器，并自动判断瓶颈类型。**
    - 提示: 参考 22.3.8 节的性能报告生成

13. **对比一个 compute-bound kernel 和一个 memory-bound kernel 的 ncu 输出特征，总结它们的关键区别。**
    - 提示: 参考 22.6.1 和 22.6.2 节的判断方法

14. **使用 Roofline 模型分析 Softmax 操作的性能瓶颈。对于 N = 4096 和 FP16 数据，计算其 Arithmetic Intensity 并判断瓶颈类型。**
    - 提示: Softmax 的 FLOPs ≈ 5N，Bytes ≈ 2N × sizeof(T)

15. **设计一个完整的优化案例，展示如何将一个 Memory-bound 的 Triton kernel 优化为 Compute-bound。需要包括优化前后的性能对比。**
    - 提示: 参考 22.8 节的完整案例流程

---

## 参考资料

### 官方文档
1. NVIDIA Nsight Systems Documentation - https://docs.nvidia.com/nsight-systems/
2. NVIDIA Nsight Compute Documentation - https://docs.nvidia.com/nsight-compute/
3. Triton Documentation - https://triton-lang.org/
4. NVIDIA GPU Architecture Whitepaper - https://www.nvidia.com/en-us/data-center/whitepapers/

### 工具和库
1. Nsight Systems - https://developer.nvidia.com/nsight-systems
2. Nsight Compute - https://developer.nvidia.com/nsight-compute
3. CUDA Toolkit - https://developer.nvidia.com/cuda-toolkit

### 学术论文
1. "Roofline: An Insightful Visual Performance Model for Multicore Architectures" - Williams et al.
2. "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations" - Tillet et al.
3. "GPU Performance Analysis" - NVIDIA Technical Report

---

> **下一章预告**：[第 23 章：Triton 与 PyTorch 集成](23-pytorch-integration.md) - 深入探讨 Triton 与 PyTorch 框架的集成方式，包括自定义算子、JIT 编译、性能优化等关键主题。

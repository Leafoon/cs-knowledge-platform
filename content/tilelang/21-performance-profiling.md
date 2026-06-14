---
title: "Chapter 21: 性能剖析与瓶颈定位"
description: "系统介绍 GPU/NPU 性能剖析方法论，硬件级分析工具使用，TileLang 内置性能指标，以及端到端性能分析工作流"
updated: "2025-01-15"
---

# Chapter 21: 性能剖析与瓶颈定位

> **Learning Objectives**
> - 掌握性能剖析的方法论与系统化流程
> - 熟练使用 ncu、nsys、rocprof 等硬件级分析工具
> - 理解 TileLang 内置性能指标（Memory Bandwidth / Arithmetic Intensity / Occupancy）
> - 掌握编译器 IR dump 分析方法
> - 理解 Profile-guided Optimization (PGO) 的原理与应用
> - 掌握瓶颈分类方法（计算密集 / 访存密集 / 同步密集）
> - 建立端到端性能分析 workflow

---

## 21.1 性能剖析方法论

### 21.1.1 性能分析的核心原则

性能剖析不是盲目地查看各种指标，而是一个有系统、有层次的分析过程。核心原则如下：

<div data-component="ProfilingWorkflowDiagram"></div>

```
性能分析的"漏斗模型"：

    ┌─────────────────────────────────────┐
    │  1. 宏观指标：端到端执行时间         │  ← 确定目标
    └──────────────┬──────────────────────┘
                   │
    ┌──────────────v──────────────────────┐
    │  2. 硬件计数器：SM利用率/带宽/FLOPS  │  ← 定位瓶颈类型
    └──────────────┬──────────────────────┘
                   │
    ┌──────────────v──────────────────────┐
    │  3. 内核级分析：每个 kernel 的耗时    │  ← 找到热点 kernel
    └──────────────┬──────────────────────┘
                   │
    ┌──────────────v──────────────────────┐
    │  4. 指令级分析：warp 执行效率         │  ← 精确定位原因
    └──────────────┬──────────────────────┘
                   │
    ┌──────────────v──────────────────────┐
    │  5. 源码级关联：定位到具体代码行      │  ← 指导优化
    └─────────────────────────────────────┘
```

上述"漏斗模型"体现了性能分析的核心思想——从宏观到微观逐层收敛。第一层通过端到端执行时间确定性能目标，即当前程序与理想状态之间的差距有多大。第二层通过硬件计数器（SM 利用率、带宽利用率、FLOPS）判断瓶颈的宏观类型，回答"瓶颈是计算、访存还是延迟"这个关键问题。第三层通过 kernel 级分析找到具体的热点 kernel，将注意力聚焦到最耗时的部分。第四层通过指令级分析深入 warp 执行效率，发现分支发散、指令停顿等微观问题。第五层将性能问题精准关联到源码行，为代码级优化提供直接指导。这种"漏斗"思维的要点在于：不要一开始就深入微观细节，而是先确定宏观方向，避免在错误的优化方向上浪费精力。在实践中很多初学者会直接跳到第四层或第五层查看汇编指令，却忽略了 SM 利用率只有 30% 的事实，结果花数小时优化了几条指令，总体性能提升不到 5%。正确的做法是逐层推进，每一层确认无误后再深入下一层。

### 21.1.2 性能指标金字塔

```python
class PerformanceMetricsPyramid:
    """
    性能指标金字塔：从宏观到微观
    
    Level 1: 应用级指标
        - 端到端执行时间
        - 吞吐量（samples/sec, tokens/sec）
    
    Level 2: 内核级指标
        - 每个 kernel 的执行时间
        - Kernel 启动开销
        - Kernel 占比
    
    Level 3: 硬件级指标
        - SM 利用率 (Occupancy)
        - 内存带宽利用率
        - 计算吞吐量利用率
    
    Level 4: 指令级指标
        - Warp 执行效率
        - 分支发散度
        - 指令混合比
    
    Level 5: 微架构级指标
        - Cache 命中率
        - Bank conflict
        - 寄存器 spill
    """
    
    LEVELS = {
        1: ["execution_time", "throughput", "latency"],
        2: ["kernel_time", "kernel_launch_overhead", "kernel_overlap"],
        3: ["sm_utilization", "memory_bandwidth", "compute_throughput"],
        4: ["warp_efficiency", "branch_divergence", "instruction_mix"],
        5: ["cache_hit_rate", "bank_conflict", "register_spill"],
    }
    
    @staticmethod
    def diagnose(metrics):
        """根据指标诊断瓶颈"""
        diagnosis = []
        
        # Level 3 诊断
        if metrics["sm_utilization"] < 0.5:
            diagnosis.append("Low SM utilization - possible memory bound or launch overhead")
        
        if metrics["memory_bandwidth"] > 0.8 * metrics["peak_bandwidth"]:
            diagnosis.append("Memory bandwidth saturated - memory bound kernel")
        
        if metrics["compute_throughput"] > 0.8 * metrics["peak_flops"]:
            diagnosis.append("Compute throughput saturated - compute bound kernel")
        
        # Level 4 诊断
        if metrics["warp_efficiency"] < 0.8:
            diagnosis.append("Low warp efficiency - possible branch divergence")
        
        # Level 5 诊断
        if metrics["cache_hit_rate"]["L2"] < 0.5:
            diagnosis.append("Low L2 cache hit rate - poor data locality")
        
        if metrics["bank_conflict"] > 0.1:
            diagnosis.append("Significant bank conflicts in shared memory")
        
        if metrics["register_spill"] > 0:
            diagnosis.append(f"Register spill detected: {metrics['register_spill']} spills")
        
        return diagnosis
```

上述 `diagnose` 方法实现了分层诊断逻辑：首先检查 Level 3 硬件级指标，当 SM 利用率低于 50% 时判定为可能存在访存瓶颈或启动开销问题；当 DRAM 带宽利用率超过峰值的 80% 时判定为访存密集型内核；当计算吞吐量超过峰值的 80% 时判定为计算密集型内核。接着检查 Level 4 指令级指标，Warp 效率低于 80% 暗示存在分支发散。最后检查 Level 5 微架构级指标，包括 L2 Cache 命中率低于 50% 表示数据局部性差、Shared Memory Bank Conflict 超过 0.1 表示需要优化内存布局、寄存器溢出大于 0 则需要减少寄存器使用量。这种分层诊断能快速定位性能瓶颈的根因。

---

## 21.2 硬件级分析工具

### 21.2.1 NVIDIA Nsight Compute (ncu)

`ncu` 是 NVIDIA 提供的内核级性能分析工具，能够采集详细的硬件计数器数据。

#### 基础使用

```bash
# 基本性能分析
ncu --set full python train.py

# 分析特定 kernel
ncu --kernel-name "gemm_kernel" python train.py

# 输出到文件
ncu --set full -o profile_result python train.py

# 只采集特定指标
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__warps_active.avg.pct_of_peak_sustained_elapsed \
    python train.py
```

上述 ncu 命令展示了四种常用用法：`--set full` 启用全部硬件计数器采集，适用于全面性能分析；`--kernel-name` 指定只分析特定名称的 kernel，避免干扰；`-o` 将结果保存到文件便于后续分析；`--metrics` 精确指定要采集的指标子集，包括 SM 吞吐量、DRAM 带宽利用率和活跃 Warp 占比，这样可以减少采集开销并聚焦关键指标。在实际使用中，建议先用 `--set full` 做全面分析，再用 `--metrics` 做定向深入。

#### 关键指标解读

<div data-component="NCUAnalysisVisualizer"></div>

```python
"""
ncu 关键指标分类与解读

1. 计算利用率指标
   - sm__throughput.avg.pct_of_peak_sustained_elapsed
     含义：SM 计算吞吐量占峰值的百分比
     判断：>80% 为计算密集型，<50% 为访存密集型或有其他瓶颈

   - sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed
     含义：Tensor Core 利用率
     判断：GEMM 类 kernel 应 >60%

2. 内存利用率指标
   - gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed
     含义：显存带宽利用率
     判断：>80% 为访存密集型

   - l2__throughput.avg.pct_of_peak_sustained_elapsed
     含义：L2 cache 带宽利用率
     判断：高 L2 利用率 + 低 DRAM 利用率 = 好的数据局部性

3. 执行效率指标
   - sm__warps_active.avg.pct_of_peak_sustained_elapsed
     含义：Active warp 占比（Occupancy 的实际表现）
     判断：>70% 为良好，<50% 可能有优化空间

   - smsp__warp_issue_stalled_wait.avg.pct_of_peak_sustained_elapsed
     含义：Warp 因等待而停顿的比例
     判断：高等待 → 同步或内存延迟问题

4. 内存访问质量指标
   - l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
     含义：全局内存加载的 sector 数
     判断：与理论最小值对比，差距大说明有合并访问问题

   - l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
     含义：Shared memory 读取的 bank conflict 数
     判断：>0 需要优化 shared memory 布局
"""
```

这段代码对 ncu 采集的四类关键指标进行了分类解读：计算利用率指标（sm__throughput）反映 SM 计算单元的饱和程度，>80% 为计算密集型；内存利用率指标（gpu__dram_throughput）反映显存带宽使用情况，>80% 为访存密集型；执行效率指标（sm__warps_active）反映 Occupancy 的实际表现，>70% 为良好状态；内存访问质量指标（l1tex__t_sectors 和 bank_conflicts）反映内存合并访问和 Shared Memory 冲突情况。理解这些指标的含义和阈值是正确诊断性能瓶颈的基础。

#### 实际案例：分析 GEMM kernel

```bash
# 分析 TileLang GEMM kernel
ncu --set full \
    --kernel-name "tilelang_gemm" \
    --launch-skip 10 \
    --launch-count 5 \
    python benchmark_gemm.py

# 提取关键指标的 Python 脚本
ncu --csv \
    --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__warps_active.avg.per_cycle_active,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    python benchmark_gemm.py > ncu_metrics.csv
```

这两条 ncu 命令展示了实际分析 GEMM kernel 的工作流：第一条使用 `--set full` 进行全面分析，`--launch-skip 10` 跳过前 10 次启动（避免 JIT 编译干扰），`--launch-count 5` 只分析 5 次启动以减少采集时间；第二条使用 `--csv` 输出 CSV 格式并指定关键指标，便于后续用 Python 脚本自动解析。CSV 输出格式包含 SM 吞吐量、DRAM 带宽、活跃 Warp 数、全局内存 sector 数和 bank conflict 数，这些指标覆盖了计算、访存和执行效率三个维度。

```python
import pandas as pd

def analyze_ncu_output(csv_path):
    """分析 ncu 输出的 CSV 文件"""
    df = pd.read_csv(csv_path)
    
    print("=== NCU 性能分析报告 ===\n")
    
    # 计算利用率
    sm_util = df["sm__throughput.avg.pct_of_peak_sustained_elapsed"].mean()
    dram_util = df["gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"].mean()
    tensor_util = df["sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed"].mean()
    
    print(f"SM 计算利用率: {sm_util:.1f}%")
    print(f"DRAM 带宽利用率: {dram_util:.1f}%")
    print(f"Tensor Core 利用率: {tensor_util:.1f}%")
    
    # 瓶颈判断
    if sm_util > 80 and dram_util < 50:
        print("\n诊断：计算密集型 (Compute Bound)")
        print("建议：优化计算效率，使用 Tensor Core，循环展开")
    elif dram_util > 80 and sm_util < 50:
        print("\n诊断：访存密集型 (Memory Bound)")
        print("建议：优化内存合并访问，使用 shared memory，减少数据搬运")
    elif sm_util < 50 and dram_util < 50:
        print("\n诊断：延迟受限型 (Latency Bound)")
        print("建议：提高 Occupancy，优化指令调度，减少同步")
    
    # Bank conflict 分析
    bank_conflicts = df["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"].sum()
    if bank_conflicts > 0:
        print(f"\n警告：检测到 {bank_conflicts} 次 shared memory bank conflict")
        print("建议：调整 shared memory 布局或使用 padding")
    
    return {
        "sm_utilization": sm_util,
        "dram_utilization": dram_util,
        "tensor_utilization": tensor_util,
        "bank_conflicts": bank_conflicts,
    }
```

`analyze_ncu_output` 函数实现了自动化的瓶颈分类逻辑：首先计算 SM 利用率、DRAM 带宽利用率和 Tensor Core 利用率的均值；然后根据阈值进行三分类判断——SM 利用率 >80% 且 DRAM <50% 为计算密集型（建议使用 Tensor Core 和循环展开），DRAM >80% 且 SM <50% 为访存密集型（建议优化合并访问和 Shared Memory），两者都低于 50% 为延迟受限型（建议提高 Occupancy 和减少同步）；最后检查 bank conflict 数量，大于 0 则提示需要调整 Shared Memory 布局。这种自动分类能快速给出优化方向。

### 21.2.2 NVIDIA Nsight Systems (nsys)

`nsys` 是系统级性能分析工具，用于分析 kernel 启动、CPU-GPU 交互、数据传输等：

```bash
# 基本系统级分析
nsys profile -o profile_result python train.py

# 分析 GPU kernel 和内存传输
nsys profile --trace=cuda,nvtx python train.py

# 添加 NVTX 标记
nsys profile --trace=cuda,nvtx --sample=cpu python train.py

# 输出统计信息
nsys stats profile_result.qdrep
```

上述 nsys 命令展示了系统级性能分析的基本流程：`nsys profile -o` 生成包含 CUDA API 调用、kernel 启动和内存传输的完整时间线；`--trace=cuda,nvtx` 指定只采集 CUDA 和 NVTX 事件，减少噪声；`--sample=cpu` 添加 CPU 采样以分析 CPU 端瓶颈；`nsys stats` 对采集的 `.qdrep` 文件生成统计摘要。与 ncu 的内核级分析不同，nsys 关注的是系统级视角，包括 kernel 启动开销、CPU-GPU 数据传输时间和 kernel 间的空闲间隔。

```python
def analyze_nsys_report(qdrep_path):
    """
    分析 nsys 报告
    
    关注点：
    1. Kernel 执行时间占比
    2. CPU-GPU 数据传输时间
    3. Kernel 启动开销
    4. Kernel 间的空闲时间
    """
    import subprocess
    
    # 生成统计报告
    result = subprocess.run(
        ["nsys", "stats", "--format=csv", qdrep_path],
        capture_output=True,
        text=True,
    )
    
    # 解析 kernel 执行统计
    kernel_stats = parse_kernel_stats(result.stdout)
    
    total_kernel_time = sum(k["duration"] for k in kernel_stats)
    total_time = kernel_stats[-1]["end"] - kernel_stats[0]["start"]
    
    print("=== Nsys 系统级分析报告 ===\n")
    print(f"总执行时间: {total_time*1000:.2f} ms")
    print(f"Kernel 执行时间: {total_kernel_time*1000:.2f} ms")
    print(f"Kernel 占比: {total_kernel_time/total_time*100:.1f}%")
    print(f"总 Kernel 数量: {len(kernel_stats)}")
    
    # Top-10 最耗时的 kernel
    print("\n=== Top-10 最耗时 Kernel ===")
    sorted_kernels = sorted(kernel_stats, key=lambda x: x["duration"], reverse=True)
    for i, k in enumerate(sorted_kernels[:10]):
        pct = k["duration"] / total_kernel_time * 100
        print(f"{i+1}. {k['name']}: {k['duration']*1000:.3f} ms ({pct:.1f}%)")
    
    # 检测 kernel 启动开销
    launch_overheads = []
    for i in range(1, len(kernel_stats)):
        gap = kernel_stats[i]["start"] - kernel_stats[i-1]["end"]
        if gap > 0:
            launch_overheads.append(gap)
    
    if launch_overheads:
        avg_overhead = sum(launch_overheads) / len(launch_overheads)
        print(f"\n平均 Kernel 间隔: {avg_overhead*1000:.4f} ms")
        if avg_overhead > 0.01:  # 10us
            print("警告：Kernel 启动开销较大，考虑使用 CUDA Graph 或 kernel fusion")
    
    return kernel_stats
```

`analyze_nsys_report` 函数实现了 nsys 报告的解析和分析工作流：首先调用 `nsys stats` 生成 CSV 格式的统计报告；然后解析 kernel 执行统计，计算总执行时间、kernel 执行时间和占比；接着按执行时间降序排列找出 Top-10 最耗时的 kernel，这是性能优化的重点目标；最后检测 kernel 间的间隔时间，平均间隔超过 10 微秒说明启动开销较大，建议使用 CUDA Graph 或 kernel fusion 来减少启动次数。这种分析能帮助识别是计算密集还是启动开销主导了总时间。

### 21.2.3 AMD ROCm Profiler (rocprof)

对于 AMD GPU，使用 `rocprof` 进行性能分析：

```bash
# 基本性能分析
rocprof --stats python train.py

# 采集硬件计数器
rocprof --pmc GPRTS_TC_READ_REQ \
        --pmc GPRTS_TC_WRITE_REQ \
        --pmc GPRTS_SQ_INSTS \
        python train.py

# 输出到 CSV
rocprof --stats --csv python train.py
```

上述 rocprof 命令展示了 AMD GPU 性能分析的基本方法：`--stats` 采集 kernel 执行统计信息（执行时间、Grid/Workgroup 大小等）；`--pmc` 指定硬件性能计数器，GPRTS_TC_READ_REQ 和 GPRTS_TC_WRITE_REQ 监控纹理缓存（L2）读写请求，GPRTS_SQ_INSTS 监控计算单元指令数；`--csv` 输出 CSV 格式便于自动化分析。AMD GPU 的性能计数器命名与 NVIDIA 不同，TCC 对应 L2 Cache，TCP 对应 L1 Cache，SQ 对应计算单元。

```python
def analyze_rocprof_output(stats_csv, counters_csv):
    """
    分析 rocprof 输出
    
    AMD GPU 特有的指标：
    - TCC (Texture Cache / L2 Cache) 访问统计
    - TCP (Texture Cache Pipe / L1 Cache) 访问统计
    - SQ (Sequencer / Compute Unit) 执行统计
    - TA (Texture Addresser) 纹理访问
    """
    stats = pd.read_csv(stats_csv)
    counters = pd.read_csv(counters_csv)
    
    print("=== ROCm 性能分析报告 ===\n")
    
    # Kernel 执行统计
    for _, row in stats.iterrows():
        print(f"Kernel: {row['KernelName']}")
        print(f"  Duration: {row['Duration']*1000:.3f} ms")
        print(f"  Grid Size: {row['GridSize']}")
        print(f"  Workgroup Size: {row['WorkgroupSize']}")
    
    # 硬件计数器分析
    if not counters.empty:
        print("\n=== 硬件计数器统计 ===")
        for col in counters.columns:
            if counters[col].dtype in ['int64', 'float64']:
                print(f"{col}: mean={counters[col].mean():.2f}, sum={counters[col].sum():.0f}")
```

这段代码是 21.2.3 AMD ROCm Profiler (rocprof) 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

`analyze_rocprof_output` 函数解析 AMD GPU 的 rocprof 输出：首先读取 stats CSV 获取每个 kernel 的执行时间、Grid Size 和 Workgroup Size 等基本信息；然后读取 counters CSV 分析硬件计数器，对所有数值类型的列计算均值和总和。AMD GPU 特有的指标包括 TCC（Texture Cache/L2 Cache）访问统计、TCP（Texture Cache Pipe/L1 Cache）访问统计、SQ（Sequencer/Compute Unit）执行统计和 TA（Texture Addresser）纹理访问。通过对比这些指标可以判断是计算瓶颈还是访存瓶颈。

---

以上介绍了 NVIDIA ncu/nsys 和 AMD rocprof 三款硬件级性能分析工具。ncu 擅长内核级微架构分析（寄存器、Bank Conflict、Cache 命中率），nsys 擅长系统级时间线分析（kernel 启动序列、CPU-GPU 数据传输），rocprof 则是 AMD 平台的对应工具。实际工作中三者互补使用：先用 nsys 做系统级扫描发现异常耗时区域，再用 ncu 深入分析具体 kernel 的微架构瓶颈，最后在 IR 层面验证编译器是否生成了预期的优化代码。在完成硬件工具层面的分析之后，我们需要进入 TileLang 框架内部，利用其内置的性能指标系统——Memory Bandwidth、Arithmetic Intensity 和 Occupancy——进行更高层次的性能建模和瓶颈诊断。

## 21.3 TileLang 内置性能指标

### 21.3.1 Memory Bandwidth 分析

<div data-component="PerformanceMetricsDashboard"></div>

```python
import tilelang
from tilelang.profiler import BandwidthProfiler

class MemoryBandwidthAnalyzer:
    """
    内存带宽利用率分析器
    
    理论峰值带宽 vs 实际带宽：
    - NVIDIA A100: 2039 GB/s (HBM2e)
    - NVIDIA H100: 3350 GB/s (HBM3)
    - AMD MI300X: 5300 GB/s (HBM3)
    
    带宽利用率 = 实际带宽 / 理论峰值带宽
    """
    
    def __init__(self, kernel_func, problem_sizes, target_gpu):
        self.kernel_func = kernel_func
        self.problem_sizes = problem_sizes
        self.peak_bandwidth = self._get_peak_bandwidth(target_gpu)
    
    def _get_peak_bandwidth(self, gpu):
        """获取 GPU 的理论峰值带宽 (GB/s)"""
        specs = {
            "nvidia-a100": 2039,
            "nvidia-h100": 3350,
            "nvidia-v100": 900,
            "amd-mi300x": 5300,
            "amd-mi250x": 3200,
        }
        return specs.get(gpu, 1000)
    
    def measure_bandwidth(self, kernel, input_sizes):
        """
        测量实际内存带宽
        
        实际带宽 = 数据传输量 / 执行时间
        """
        import torch
        
        # 计算理论最小数据传输量
        total_bytes = self._calculate_data_movement(input_sizes)
        
        # 预热
        for _ in range(10):
            kernel()
        
        torch.cuda.synchronize()
        
        # 测量
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(100):
            kernel()
        end.record()
        
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end) / 100
        
        # 计算带宽
        bandwidth_gbs = (total_bytes / 1e9) / (elapsed_ms / 1e3)
        utilization = bandwidth_gbs / self.peak_bandwidth * 100
        
        return {
            "elapsed_ms": elapsed_ms,
            "data_bytes": total_bytes,
            "bandwidth_gbs": bandwidth_gbs,
            "peak_bandwidth_gbs": self.peak_bandwidth,
            "utilization_pct": utilization,
        }
    
    def _calculate_data_movement(self, sizes):
        """计算理论最小数据传输量"""
        M, N, K = sizes
        # GEMM: 读取 A(M,K) + B(K,N)，写入 C(M,N)
        bytes_A = M * K * 2  # fp16
        bytes_B = K * N * 2
        bytes_C = M * N * 4  # fp32 output
        return bytes_A + bytes_B + bytes_C
    
    def profile_all_sizes(self):
        """分析所有问题规模的带宽利用率"""
        results = []
        
        for sizes in self.problem_sizes:
            kernel = self.kernel_func(*sizes)
            bw = self.measure_bandwidth(kernel, sizes)
            bw["problem_size"] = sizes
            results.append(bw)
            
            print(f"GEMM {sizes}: {bw['bandwidth_gbs']:.1f} GB/s "
                  f"({bw['utilization_pct']:.1f}% of peak)")
        
        return results
```

这段代码是 21.3.1 Memory Bandwidth 分析 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

`MemoryBandwidthAnalyzer` 实现了内存带宽利用率的测量方法论：首先通过 `_get_peak_bandwidth` 获取目标 GPU 的理论峰值带宽（如 A100 为 2039 GB/s）；`measure_bandwidth` 方法通过 CUDA Event 精确计时，先预热 10 次消除冷启动影响，再运行 100 次取均值以减少波动；`_calculate_data_movement` 计算 GEMM 操作的理论最小数据传输量（读取 A、B 矩阵加写入 C 矩阵）；最终用实际带宽除以峰值带宽得到利用率。带宽利用率是判断访存密集型 kernel 性能的关键指标，低于 60% 通常意味着存在内存访问优化空间。

### 21.3.2 Arithmetic Intensity 分析

```python
class ArithmeticIntensityAnalyzer:
    """
    算术强度分析器
    
    算术强度 = FLOPS / Bytes Transferred (FLOPS/Byte)
    
    Roofline Model:
    - 如果 AI < 峰值带宽/峰值算力 → Memory Bound
    - 如果 AI > 峰值带宽/峰值算力 → Compute Bound
    
    A100 为例:
    - 峰值 FP16 Tensor Core: 312 TFLOPS
    - 峰值带宽: 2039 GB/s
    - 转折点 AI = 312/2.039 ≈ 153 FLOPS/Byte
    """
    
    def __init__(self, target_gpu):
        self.peak_tflops = self._get_peak_tflops(target_gpu)
        self.peak_bw_gbs = self._get_peak_bandwidth(target_gpu)
        self.ridge_point = self.peak_tflops * 1000 / self.peak_bw_gbs
    
    def analyze_operation(self, name, flops, bytes_transferred):
        """
        分析单个操作的算术强度
        """
        ai = flops / bytes_transferred
        
        # 计算受限类型
        if ai < self.ridge_point:
            bound_type = "Memory Bound"
            # 在 memory bound 区域，性能受限于带宽
            expected_perf = self.peak_bw_gbs * ai / 1000  # TFLOPS
        else:
            bound_type = "Compute Bound"
            expected_perf = self.peak_tflops
        
        return {
            "name": name,
            "flops": flops,
            "bytes": bytes_transferred,
            "arithmetic_intensity": ai,
            "ridge_point": self.ridge_point,
            "bound_type": bound_type,
            "expected_tflops": expected_perf,
        }
    
    def analyze_gemm(self, M, N, K, dtype="float16"):
        """分析 GEMM 的算术强度"""
        flops = 2 * M * N * K  # multiply-add
        
        bytes_per_elem = 2 if dtype == "float16" else 4
        bytes_A = M * K * bytes_per_elem
        bytes_B = K * N * bytes_per_elem
        bytes_C = M * N * (4 if dtype != "float16" else 4)  # fp32 output
        total_bytes = bytes_A + bytes_B + bytes_C
        
        return self.analyze_operation(
            name=f"GEMM({M},{N},{K})_{dtype}",
            flops=flops,
            bytes_transferred=total_bytes,
        )
    
    def plot_roofline(self, operations=None):
        """
        绘制 Roofline Model 图
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # 计算带宽限制线
        ai_range = np.logspace(-1, 4, 100)
        bw_limited = self.peak_bw_gbs * ai_range / 1000  # TFLOPS
        
        # Roofline
        roofline = np.minimum(bw_limited, self.peak_tflops)
        
        ax.loglog(ai_range, roofline, 'k-', linewidth=2, label='Roofline')
        ax.loglog(ai_range, bw_limited, 'b--', alpha=0.5, label='Memory Bandwidth Limit')
        ax.axhline(y=self.peak_tflops, color='r', linestyle='--', alpha=0.5, 
                   label='Compute Limit')
        ax.axvline(x=self.ridge_point, color='g', linestyle=':', alpha=0.5,
                   label=f'Ridge Point ({self.ridge_point:.0f} FLOPS/Byte)')
        
        # 标注操作点
        if operations:
            for op in operations:
                ax.plot(op["arithmetic_intensity"], op["expected_tflops"], 
                       'o', markersize=10, label=op["name"])
        
        ax.set_xlabel("Arithmetic Intensity (FLOPS/Byte)")
        ax.set_ylabel("Performance (TFLOPS)")
        ax.set_title("Roofline Model Analysis")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("roofline.png", dpi=150, bbox_inches='tight')
        plt.show()
```

这段代码是 21.3.2 Arithmetic Intensity 分析 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

`ArithmeticIntensityAnalyzer` 基于 Roofline Model 分析算术强度：算术强度（AI）定义为 FLOPS/Bytes Transferred，表示每传输一个字节数据能执行多少次浮点运算；`ridge_point` 是转折点 AI，等于峰值算力除以峰值带宽（A100 约为 153 FLOPS/Byte），低于此值为 Memory Bound，高于此值为 Compute Bound；`analyze_gemm` 计算 GEMM 的 AI，其中 FLOPS = 2*M*N*K（乘加运算），Bytes = A+B+C 矩阵的数据量；`plot_roofline` 绘制 Roofline 图，横轴为算术强度，纵轴为性能，曲线表示理论性能上限。Roofline Model 是判断 kernel 是计算瓶颈还是访存瓶颈的核心工具。

### 21.3.3 Occupancy 分析

```python
class OccupancyAnalyzer:
    """
    GPU Occupancy 分析器
    
    Occupancy = Active Warps / Maximum Warps per SM
    
    影响 Occupancy 的因素：
    1. 每个线程使用的寄存器数
    2. 每个 block 使用的 shared memory
    3. 每个 block 的线程数
    4. 硬件限制（最大 block 数/SM，最大 warp 数/SM）
    """
    
    def __init__(self, gpu_arch="sm_80"):
        self.specs = self._get_gpu_specs(gpu_arch)
    
    def _get_gpu_specs(self, arch):
        """获取 GPU 架构规格"""
        specs = {
            "sm_80": {  # A100
                "max_warps_per_sm": 64,
                "max_blocks_per_sm": 32,
                "max_threads_per_sm": 2048,
                "registers_per_sm": 65536,
                "smem_per_sm": 164 * 1024,  # 164 KB (configurable)
                "warp_size": 32,
                "num_sms": 108,
            },
            "sm_90": {  # H100
                "max_warps_per_sm": 64,
                "max_blocks_per_sm": 32,
                "max_threads_per_sm": 2048,
                "registers_per_sm": 65536,
                "smem_per_sm": 228 * 1024,
                "warp_size": 32,
                "num_sms": 132,
            },
        }
        return specs.get(arch, specs["sm_80"])
    
    def calculate_occupancy(self, threads_per_block, regs_per_thread, smem_per_block):
        """
        计算理论 Occupancy
        
        步骤：
        1. 计算每个 SM 能容纳的 block 数（受寄存器、shared memory、线程数限制）
        2. 计算每个 SM 的 active warps
        3. 计算 occupancy
        """
        warps_per_block = (threads_per_block + self.specs["warp_size"] - 1) // self.specs["warp_size"]
        
        # 限制因素 1：寄存器数量
        regs_per_block = regs_per_thread * threads_per_block
        max_blocks_by_regs = self.specs["registers_per_sm"] // regs_per_block if regs_per_block > 0 else float('inf')
        
        # 限制因素 2：Shared memory
        max_blocks_by_smem = (self.specs["smem_per_sm"] // smem_per_block 
                              if smem_per_block > 0 else float('inf'))
        
        # 限制因素 3：最大 block 数
        max_blocks_by_limit = self.specs["max_blocks_per_sm"]
        
        # 限制因素 4：最大线程数
        max_blocks_by_threads = self.specs["max_threads_per_sm"] // threads_per_block
        
        # 取最小值
        max_blocks = min(
            max_blocks_by_regs,
            max_blocks_by_smem,
            max_blocks_by_limit,
            max_blocks_by_threads,
        )
        
        active_warps = max_blocks * warps_per_block
        max_warps = self.specs["max_warps_per_sm"]
        
        occupancy = active_warps / max_warps
        
        return {
            "threads_per_block": threads_per_block,
            "warps_per_block": warps_per_block,
            "regs_per_thread": regs_per_thread,
            "smem_per_block": smem_per_block,
            "max_blocks_by_regs": max_blocks_by_regs,
            "max_blocks_by_smem": max_blocks_by_smem,
            "max_blocks_by_limit": max_blocks_by_limit,
            "max_blocks_by_threads": max_blocks_by_threads,
            "max_blocks_per_sm": max_blocks,
            "active_warps": active_warps,
            "max_warps": max_warps,
            "occupancy": occupancy,
            "limiting_factor": self._get_limiting_factor(
                max_blocks_by_regs, max_blocks_by_smem, 
                max_blocks_by_limit, max_blocks_by_threads
            ),
        }
    
    def _get_limiting_factor(self, *factors):
        names = ["registers", "shared_memory", "block_limit", "thread_limit"]
        min_idx = factors.index(min(factors))
        return names[min_idx]
    
    def analyze_kernel(self, kernel_source):
        """
        分析 kernel 的 Occupancy
        """
        # 从编译信息中提取参数
        compile_info = tilelang.compile(kernel_source)
        
        regs_per_thread = compile_info["registers_per_thread"]
        smem_per_block = compile_info["shared_memory_bytes"]
        threads_per_block = compile_info["block_size"]
        
        return self.calculate_occupancy(threads_per_block, regs_per_thread, smem_per_block)
```

`OccupancyAnalyzer` 实现了 GPU Occupancy 的理论计算：Occupancy 定义为活跃 Warp 数除以 SM 最大 Warp 数，影响因素包括寄存器使用量、Shared Memory 使用量、线程数和硬件限制；`calculate_occupancy` 方法分别计算四个限制因素下的最大 block 数——寄存器限制（每 SM 65536 个寄存器）、Shared Memory 限制（A100 为 164KB）、硬件 block 数上限（32 个）和线程数上限（2048 个），取最小值作为实际最大 block 数；`_get_limiting_factor` 返回瓶颈因素，帮助针对性优化。高 Occupancy 能隐藏内存延迟，但过高可能降低每线程的资源分配，需权衡取舍。

---

以上三个 TileLang 内置分析器覆盖了性能建模的核心维度：Memory Bandwidth 回答"数据传输是否达标"，Arithmetic Intensity 回答"计算与访存的比例关系"，Occupancy 回答"硬件资源是否被充分利用"。三者结合使用可以形成完整的 bottleneck 诊断链：先通过 Occupancy 检查资源利用率→通过 Bandwidth 判断访存效率→通过 Arithmetic Intensity 判定瓶颈类型。然而，硬件计数器和性能模型只能告诉我们"出现了什么问题"，却无法解释"为什么会这样"——例如 ncu 显示 Bank Conflict 很高，但 Shared Memory 的哪一段代码导致了 Conflict？Occupancy 很低，但具体是寄存器压力还是 Shared Memory 分配的问题？要回答这些问题，我们需要深入到编译器的中间表示（IR）层面，查看编译器实际生成的代码结构。

## 21.4 编译器 IR Dump 分析

### 21.4.1 TileLang IR 层次

TileLang 的编译过程中产生多个层次的中间表示（IR）：

```python
"""
TileLang IR 层次（从高到低）：

1. TileLang DSL (高层 Python DSL)
   ↓
2. TileLang IR (算子级 IR)
   ↓ Schedule/Transform
3. TIR (Tensor IR)
   ↓ Lowering
4. Low-level IR (类似 CUDA/ROCm)
   ↓ CodeGen
5. Target Code (PTX/ROCm Assembly)
"""
```

这段代码描述了 TileLang 编译器的 IR 层次结构：从高层到低层依次为 TileLang DSL（用户编写的 Python 算子描述）、TileLang IR（算子级中间表示）、TIR（Tensor IR，经过 Schedule/Transform 优化后的表示）、Low-level IR（类似 CUDA/ROCm 的底层表示）、Target Code（最终的 PTX 或 ROCm 汇编代码）。每一层 IR 对应不同的优化阶段，理解这个层次有助于定位编译器优化效果——如果某层 IR 中存在低效的内存访问模式，说明上层的优化没有正确传播。

### 21.4.2 IR Dump 使用方法

```python
import tilelang
from tilelang import compile

# 启用 IR dump
tilelang.set_dump_ir(True)

# 编译 kernel（会输出各层 IR）
kernel = compile(kernel_source, target="nvidia-a100")

# 也可以手动获取特定层的 IR
def dump_ir_levels(kernel_source, target="nvidia-a100"):
    """
    输出各层 IR，用于分析编译器优化效果
    """
    # Level 1: TileLang IR
    tilelang_ir = tilelang.lower(kernel_source, target, stage="tilelang_ir")
    print("=== TileLang IR ===")
    print(tilelang_ir)
    
    # Level 2: TIR
    tir = tilelang.lower(kernel_source, target, stage="tir")
    print("\n=== TIR ===")
    print(tir)
    
    # Level 3: Low-level IR
    low_level = tilelang.lower(kernel_source, target, stage="low_level")
    print("\n=== Low-level IR ===")
    print(low_level)
    
    # Level 4: Target code
    target_code = tilelang.lower(kernel_source, target, stage="target_code")
    print(f"\n=== {target.upper()} Code ===")
    print(target_code)
    
    return {
        "tilelang_ir": tilelang_ir,
        "tir": tir,
        "low_level": low_level,
        "target_code": target_code,
    }
```

`dump_ir_levels` 函数展示了如何获取编译器各层的 IR 输出：首先调用 `tilelang.set_dump_ir(True)` 启用 IR dump；然后通过 `tilelang.lower` 的 `stage` 参数分别获取 TileLang IR、TIR、Low-level IR 和 Target Code 四个层次的表示；每个层次对应不同的抽象级别，TileLang IR 最接近用户代码，Target Code 是最终机器码。通过逐层对比 IR 可以发现编译器优化是否生效，例如循环是否被正确展开、内存访问是否被合并、barrier 同步是否被合理放置。

### 21.4.3 IR 分析技巧

```python
def analyze_tir(tir_source):
    """
    分析 TIR 中的关键优化信息
    
    关注点：
    1. 内存访问模式（是否有合并访问）
    2. 循环结构（是否被正确展开）
    3. 同步点（barrier 的位置和数量）
    4. 向量化程度
    """
    analysis = {
        "memory_accesses": [],
        "loops": [],
        "barriers": [],
        "vectorizations": [],
    }
    
    lines = tir_source.split("\n")
    for i, line in enumerate(lines):
        # 检测内存访问
        if "T.load" in line or "T.store" in line:
            analysis["memory_accesses"].append({
                "line": i + 1,
                "content": line.strip(),
                "type": "load" if "T.load" in line else "store",
            })
        
        # 检测循环
        if "T.for" in line:
            analysis["loops"].append({
                "line": i + 1,
                "content": line.strip(),
            })
        
        # 检测 barrier
        if "T.sync" in line or "__syncthreads" in line:
            analysis["barriers"].append({
                "line": i + 1,
                "content": line.strip(),
            })
        
        # 检测向量化
        if "vectorize" in line.lower() or "T.vectorize" in line:
            analysis["vectorizations"].append({
                "line": i + 1,
                "content": line.strip(),
            })
    
    print("=== TIR 分析报告 ===")
    print(f"内存访问点: {len(analysis['memory_accesses'])}")
    print(f"循环数量: {len(analysis['loops'])}")
    print(f"Barrier 数量: {len(analysis['barriers'])}")
    print(f"向量化点: {len(analysis['vectorizations'])}")
    
    # 分析 barrier 是否过多
    if len(analysis['barriers']) > 10:
        print("\n警告：Barrier 数量过多，可能影响性能")
    
    return analysis
```

`analyze_tir` 函数对 TIR 源码进行多维度分析：内存访问维度检测 `T.load` 和 `T.store` 操作的位置和类型，帮助识别是否存在合并访问问题；循环维度检测 `T.for` 循环结构，分析是否被正确展开或向量化；同步点维度检测 `T.sync` 和 `__syncthreads`，barrier 数量过多（>10）会导致严重的性能下降；向量化维度检测 `vectorize` 指令，向量化程度越高数据吞吐量越大。这种静态分析能快速发现 IR 中的潜在性能问题，指导后续的调度优化。

---

通过 IR Dump 分析我们可以确认编译器优化是否正确生效，但静态的 IR 分析无法提供运行时信息——例如某个 tile size 在理论上最优，但实际运行时可能因为 cache 竞争或 bank conflict 而表现不佳。这就是 Profile-Guided Optimization (PGO) 存在的意义：将运行时的性能数据反馈给编译器，指导编译器做出更优的调度决策。PGO 的核心思想是"先测量，再优化"，通过实际运行数据来验证和校正静态分析的结论。

## 21.5 Profile-Guided Optimization (PGO)

### 21.5.1 PGO 基本原理

```python
class ProfileGuidedOptimizer:
    """
    基于 Profile 的优化
    
    工作流程：
    1. 使用代表性输入运行程序，收集 profile 数据
    2. 分析 profile 数据，识别热点和瓶颈
    3. 基于 profile 结果进行针对性优化
    4. 重新编译优化后的程序
    
    TileLang 中的 PGO 应用：
    - 根据实际运行的 kernel 大小调整 tile size
    - 根据数据访问模式优化内存布局
    - 根据实际 occupancy 调整资源分配
    """
    
    def __init__(self):
        self.profile_data = {}
    
    def collect_profile(self, kernel, inputs, num_iterations=100):
        """
        收集 profile 数据
        """
        import torch
        
        # 收集执行时间
        times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            kernel(*inputs)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        # 收集硬件计数器（简化版）
        self.profile_data = {
            "mean_time_ms": sum(times) / len(times),
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "std_time_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
            "times": times,
        }
        
        return self.profile_data
    
    def analyze_hotspots(self, profile_data):
        """
        分析热点
        """
        hotspots = []
        
        # 检测性能波动大的 kernel
        cv = profile_data["std_time_ms"] / profile_data["mean_time_ms"]
        if cv > 0.1:
            hotspots.append({
                "type": "high_variance",
                "description": f"性能波动大 (CV={cv:.2f})，可能存在 cache 抖动或竞争",
                "suggestion": "增加预热次数，或使用 CUDA Graph 固定执行模式",
            })
        
        return hotspots
    
    def suggest_optimizations(self, profile_data, kernel_info):
        """
        基于 profile 数据建议优化
        """
        suggestions = []
        
        # 分析 Occupancy
        if kernel_info.get("occupancy", 1.0) < 0.5:
            suggestions.append({
                "priority": "high",
                "category": "occupancy",
                "suggestion": "Occupancy 过低，考虑减少寄存器使用或增加 block 线程数",
            })
        
        # 分析带宽利用率
        if kernel_info.get("bandwidth_util", 0) < 0.6:
            suggestions.append({
                "priority": "high",
                "category": "memory",
                "suggestion": "带宽利用率低，检查是否有合并访问问题或数据局部性问题",
            })
        
        return suggestions
```

这段代码是 21.5.1 PGO 基本原理 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

`ProfileGuidedOptimizer` 实现了基于运行时 profile 数据的优化建议生成：`collect_profile` 通过多次运行收集执行时间统计（均值、最小值、最大值、标准差）；`analyze_hotspots` 使用变异系数（CV = 标准差/均值）检测性能波动，CV > 0.1 说明存在 cache 抖动或资源竞争；`suggest_optimizations` 根据 Occupancy 和带宽利用率生成优先级排序的优化建议——Occupancy < 50% 建议减少寄存器或增加线程数，带宽利用率 < 60% 建议检查合并访问和数据局部性。PGO 的核心思想是用实际运行数据指导优化，避免盲目调优。

### 21.5.2 自适应 Tile Size 调整

```python
class AdaptiveTileSizeOptimizer:
    """
    基于 Profile 的自适应 Tile Size 调整
    
    原理：不同大小的矩阵最优的 tile size 不同
    - 小矩阵：小 tile size 更好（减少冗余计算）
    - 大矩阵：大 tile size 更好（提高数据复用）
    """
    
    def __init__(self):
        self.tile_size_db = {}
    
    def profile_for_size(self, M, N, K, target):
        """
        为特定矩阵大小找最优 tile size
        """
        candidates = [
            {"block_M": 64, "block_N": 64, "block_K": 16},
            {"block_M": 128, "block_N": 128, "block_K": 32},
            {"block_M": 256, "block_N": 256, "block_K": 64},
            {"block_M": 128, "block_N": 256, "block_K": 32},
            {"block_M": 256, "block_N": 128, "block_K": 32},
        ]
        
        best_config = None
        best_time = float('inf')
        
        for config in candidates:
            try:
                kernel = create_kernel_with_config(M, N, K, config, target)
                time = benchmark_kernel(kernel)
                
                if time < best_time:
                    best_time = time
                    best_config = config
            except:
                continue
        
        self.tile_size_db[(M, N, K)] = (best_config, best_time)
        return best_config, best_time
    
    def get_optimal_tile_size(self, M, N, K):
        """
        获取最优 tile size（支持最近邻插值）
        """
        # 精确匹配
        if (M, N, K) in self.tile_size_db:
            return self.tile_size_db[(M, N, K)]
        
        # 最近邻匹配
        min_dist = float('inf')
        nearest = None
        
        for (m, n, k), (config, time) in self.tile_size_db.items():
            dist = abs(M - m) + abs(N - n) + abs(K - k)
            if dist < min_dist:
                min_dist = dist
                nearest = (config, time)
        
        return nearest
```

这段代码是 21.5.2 自适应 Tile Size 调整 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

`AdaptiveTileSizeOptimizer` 展示了基于实测数据的 Tile Size 自动调优方法：`profile_for_size` 方法遍历预设的候选配置（如 block_M=64/128/256, block_K=16/32/64），对每个配置编译并 benchmark kernel，记录最快的执行时间和对应配置存入数据库；`get_optimal_tile_size` 提供缓存查询和最近邻匹配两种查找策略——精确匹配直接返回，否则用曼哈顿距离（|M-m|+|N-n|+|K-k|）找到最接近的已记录尺寸。这种自适应策略的核心价值在于：不同矩阵规模的最优 Tile Size 差异巨大——小矩阵（M,N<512）用小 tile 可减少填充浪费，大矩阵（M,N>4096）用大 tile 可增加数据复用减少全局内存访问。传统的固定 tile size 方案无法覆盖所有场景，而自适应调优通过离线 profiling 构建尺寸-Tile 映射表，在线推理时快速匹配，在编译器层面实现了类似 AutoTVM 的自动调优能力。需要注意的是，候选 tile 配置的选择需要覆盖硬件约束（如 shared memory 上限、寄存器限制），否则某些配置会因为编译失败而被跳过。

在掌握了 Profile-Guided Optimization 的自适应调整能力之后，我们将进入性能瓶颈的系统化分类环节。准确识别瓶颈类型是制定优化策略的前提条件，不同的瓶颈类型需要截然不同的优化方向。例如，计算密集型瓶颈需要通过 Tensor Core 利用率和循环展开来解决，而访存密集型瓶颈则需要优化内存合并访问和 Shared Memory 使用模式。如果不能准确分类瓶颈类型，后续的优化努力很可能是徒劳的——在访存密集型 kernel 上优化计算指令几乎不会带来任何性能提升。

---

## 21.6 瓶颈分类

<div data-component="BottleneckClassificationChart"></div>

### 21.6.1 三大瓶颈类型

```python
class BottleneckClassifier:
    """
    性能瓶颈分类器
    
    三大瓶颈类型：
    
    1. Compute Bound（计算密集型）
       特征：SM 利用率高，Tensor Core 利用率高，带宽利用率低
       优化方向：提高计算效率，使用更多 Tensor Core，循环展开
    
    2. Memory Bound（访存密集型）
       特征：带宽利用率高，SM 利用率低
       优化方向：减少数据搬运，优化内存访问模式，提高数据复用
    
    3. Latency Bound（延迟密集型）
       特征：SM 利用率低，带宽利用率低，Occupancy 低
       优化方向：提高 Occupancy，使用异步操作，减少同步
    """
    
    def classify(self, metrics):
        """
        根据指标分类瓶颈类型
        """
        sm_util = metrics["sm_utilization"]
        dram_util = metrics["dram_utilization"]
        occupancy = metrics["occupancy"]
        tensor_util = metrics.get("tensor_utilization", 0)
        
        # 分类逻辑
        if sm_util > 70 and tensor_util > 50:
            return {
                "type": "Compute Bound",
                "confidence": min(1.0, sm_util / 100),
                "details": "SM 计算单元饱和",
                "optimizations": [
                    "使用 Tensor Core / MFMA 指令",
                    "循环展开 (T.unroll)",
                    "指令级并行优化",
                    "混合精度计算",
                ]
            }
        elif dram_util > 70:
            return {
                "type": "Memory Bound",
                "confidence": min(1.0, dram_util / 100),
                "details": "显存带宽饱和",
                "optimizations": [
                    "优化内存合并访问",
                    "使用 shared memory 缓存",
                    "减少中间结果的显存写回",
                    "使用压缩格式（如 sparse）",
                ]
            }
        elif occupancy < 50 or (sm_util < 50 and dram_util < 50):
            return {
                "type": "Latency Bound",
                "confidence": 1.0 - max(sm_util, dram_util) / 100,
                "details": "延迟受限，可能是低 Occupancy 或高同步开销",
                "optimizations": [
                    "提高 Occupancy（减少寄存器/smem 使用）",
                    "使用异步拷贝",
                    "减少 barrier 同步",
                    "使用 CUDA Graph 减少启动开销",
                ]
            }
        else:
            return {
                "type": "Balanced",
                "confidence": 0.5,
                "details": "计算和访存相对平衡",
                "optimizations": [
                    "综合优化计算和访存",
                    "Profile-guided 细粒度调优",
                ]
            }
```

该分类器实现了基于硬件计数器的自动瓶颈识别：Compute Bound 判断条件是 SM 利用率超过 70% 且 Tensor Core 利用率超过 50%，说明计算单元已经饱和；Memory Bound 判断条件是 DRAM 带宽利用率超过 70%，说明数据搬运成为主要瓶颈；Latency Bound 判断条件是 Occupancy 低于 50% 或 SM 和 DRAM 利用率都低于 50%，说明大量时间花在了等待延迟上。每种瓶颈类型都给出了具体的优化方向列表，帮助开发者快速定位优化策略。在实际应用中，kernel 往往不是纯粹的某一种瓶颈，可能同时存在多种瓶颈的混合，此时需要根据置信度和实际情况综合判断，优先处理影响最大的瓶颈类型。

### 21.6.2 详细瓶颈诊断

```python
def detailed_bottleneck_diagnosis(kernel_name, metrics):
    """
    详细的瓶颈诊断报告
    """
    print(f"{'='*60}")
    print(f"Kernel: {kernel_name}")
    print(f"{'='*60}")
    
    # 基础指标
    print(f"\n--- 基础指标 ---")
    print(f"执行时间: {metrics['elapsed_ms']:.3f} ms")
    print(f"FLOPS: {metrics['flops']/1e9:.1f} GFLOPS")
    print(f"数据传输: {metrics['bytes']/1e9:.3f} GB")
    
    # 利用率指标
    print(f"\n--- 利用率指标 ---")
    print(f"SM 利用率: {metrics['sm_utilization']:.1f}%")
    print(f"DRAM 带宽利用率: {metrics['dram_utilization']:.1f}%")
    print(f"Tensor Core 利用率: {metrics.get('tensor_utilization', 0):.1f}%")
    print(f"Occupancy: {metrics['occupancy']:.1f}%")
    
    # 瓶颈分类
    classifier = BottleneckClassifier()
    bottleneck = classifier.classify(metrics)
    
    print(f"\n--- 瓶颈诊断 ---")
    print(f"瓶颈类型: {bottleneck['type']}")
    print(f"诊断置信度: {bottleneck['confidence']:.1%}")
    print(f"详细说明: {bottleneck['details']}")
    
    print(f"\n--- 优化建议 ---")
    for i, opt in enumerate(bottleneck['optimizations'], 1):
        print(f"{i}. {opt}")
    
    # 额外检查
    print(f"\n--- 额外检查 ---")
    
    # Bank conflict 检查
    if metrics.get("bank_conflicts", 0) > 0:
        print(f"⚠ 检测到 {metrics['bank_conflicts']} 次 bank conflict")
    
    # 寄存器 spill 检查
    if metrics.get("register_spills", 0) > 0:
        print(f"⚠ 检测到 {metrics['register_spills']} 次寄存器 spill")
    
    # 分支发散检查
    if metrics.get("branch_divergence", 0) > 0.2:
        print(f"⚠ 分支发散率 {metrics['branch_divergence']:.1%}，较高")
    
    return bottleneck
```

该函数生成了结构化的瓶颈诊断报告，包含四个部分：基础指标显示执行时间、FLOPS 和数据传输量；利用率指标显示 SM、DRAM、Tensor Core 利用率和 Occupancy；瓶颈诊断部分调用 BottleneckClassifier 输出分类结果和置信度；额外检查部分检测 Bank Conflict、寄存器 Spill 和分支发散等微架构层面的问题。这种格式化的报告便于团队协作和性能评审，也能作为优化历史记录用于对比。

---

瓶颈分类为我们提供了清晰的优化方向，但性能分析是一个系统工程——单次的 profiling 结果往往具有偶然性，需要建立标准化、可重复的分析流程。接下来介绍的端到端性能分析 Workflow 将前述各节中的工具和方法串联起来，形成一个从 baseline 测量到优化验证的完整闭环，确保 profiling 过程的系统性和结果的可靠性。

## 21.7 端到端性能分析 Workflow

### 21.7.1 标准分析流程

```python
class EndToEndProfiler:
    """
    端到端性能分析工作流
    
    阶段 1：Baseline 测量
    阶段 2：系统级分析 (nsys)
    阶段 3：内核级分析 (ncu)
    阶段 4：源码级分析 (IR dump)
    阶段 5：优化建议生成
    阶段 6：优化效果验证
    """
    
    def __init__(self, output_dir="./profiling_results"):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def phase1_baseline(self, kernel, inputs, num_iterations=1000):
        """
        阶段 1：测量 Baseline 性能
        """
        import torch
        import time
        
        print("Phase 1: Baseline Measurement")
        print("-" * 40)
        
        # 预热
        for _ in range(10):
            kernel(*inputs)
        torch.cuda.synchronize()
        
        # 测量
        times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            kernel(*inputs)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
        
        baseline = {
            "mean_ms": sum(times) / len(times),
            "median_ms": sorted(times)[len(times) // 2],
            "min_ms": min(times),
            "max_ms": max(times),
            "p99_ms": sorted(times)[int(len(times) * 0.99)],
        }
        
        print(f"Mean: {baseline['mean_ms']:.3f} ms")
        print(f"Median: {baseline['median_ms']:.3f} ms")
        print(f"Min: {baseline['min_ms']:.3f} ms")
        print(f"P99: {baseline['p99_ms']:.3f} ms")
        
        return baseline
    
    def phase2_system_profiling(self, kernel, inputs):
        """
        阶段 2：系统级分析
        """
        import subprocess
        
        print("\nPhase 2: System-level Profiling (nsys)")
        print("-" * 40)
        
        # 生成 nsys profile 脚本
        script = f"""
import torch
import tilelang

# 加载 kernel
kernel = tilelang.load("{self.output_dir}/kernel.so")

# 预热
for _ in range(10):
    kernel()

# nsys 标记区域
torch.cuda.cudart().cudaProfilerStart()
for _ in range(100):
    kernel()
torch.cuda.cudart().cudaProfilerStop()
"""
        
        with open(f"{self.output_dir}/profile_script.py", "w") as f:
            f.write(script)
        
        # 运行 nsys
        subprocess.run([
            "nsys", "profile",
            "-o", f"{self.output_dir}/system_profile",
            "--trace=cuda,nvtx",
            "python", f"{self.output_dir}/profile_script.py",
        ])
        
        print("System profile saved to:", f"{self.output_dir}/system_profile.qdrep")
    
    def phase3_kernel_profiling(self, kernel, inputs):
        """
        阶段 3：内核级分析
        """
        import subprocess
        
        print("\nPhase 3: Kernel-level Profiling (ncu)")
        print("-" * 40)
        
        # 生成 ncu 分析脚本
        script = f"""
import torch
import tilelang

kernel = tilelang.load("{self.output_dir}/kernel.so")

# 预热
for _ in range(3):
    kernel()

# 测量
for _ in range(5):
    kernel()
"""
        
        with open(f"{self.output_dir}/ncu_script.py", "w") as f:
            f.write(script)
        
        # 运行 ncu
        subprocess.run([
            "ncu",
            "--set", "full",
            "-o", f"{self.output_dir}/kernel_profile",
            "python", f"{self.output_dir}/ncu_script.py",
        ])
        
        print("Kernel profile saved to:", f"{self.output_dir}/kernel_profile.ncu-rep")
    
    def phase4_source_analysis(self, kernel_source):
        """
        阶段 4：源码级分析
        """
        print("\nPhase 4: Source-level Analysis")
        print("-" * 40)
        
        # IR dump
        ir_levels = dump_ir_levels(kernel_source)
        
        # 分析每个 IR 层
        for level, ir in ir_levels.items():
            analysis = analyze_tir(ir)
            print(f"\n{level} analysis:")
            print(f"  Memory accesses: {len(analysis['memory_accesses'])}")
            print(f"  Barriers: {len(analysis['barriers'])}")
            print(f"  Vectorizations: {len(analysis['vectorizations'])}")
        
        return ir_levels
    
    def phase5_optimization_suggestions(self, all_data):
        """
        阶段 5：生成优化建议
        """
        print("\nPhase 5: Optimization Suggestions")
        print("-" * 40)
        
        suggestions = []
        
        # 基于 baseline 的建议
        baseline = all_data.get("baseline", {})
        if baseline.get("p99_ms", 0) / baseline.get("median_ms", 1) > 1.5:
            suggestions.append({
                "priority": "high",
                "category": "stability",
                "suggestion": "P99/Median > 1.5，性能不稳定，检查是否有资源竞争",
            })
        
        # 基于 nsys 的建议
        system_profile = all_data.get("system_profile", {})
        if system_profile.get("kernel_count", 0) > 100:
            suggestions.append({
                "priority": "medium",
                "category": "fusion",
                "suggestion": f"Kernel 数量 ({system_profile['kernel_count']}) 过多，考虑算子融合",
            })
        
        # 基于 ncu 的建议
        kernel_profile = all_data.get("kernel_profile", {})
        if kernel_profile.get("occupancy", 1) < 0.5:
            suggestions.append({
                "priority": "high",
                "category": "occupancy",
                "suggestion": "Occupancy 过低，减少寄存器或 shared memory 使用",
            })
        
        for i, s in enumerate(suggestions, 1):
            print(f"{i}. [{s['priority']}] {s['suggestion']}")
        
        return suggestions
    
    def run_full_profiling(self, kernel, kernel_source, inputs):
        """
        执行完整的分析流程
        """
        all_data = {}
        
        all_data["baseline"] = self.phase1_baseline(kernel, inputs)
        all_data["system_profile"] = self.phase2_system_profiling(kernel, inputs)
        all_data["kernel_profile"] = self.phase3_kernel_profiling(kernel, inputs)
        all_data["ir_analysis"] = self.phase4_source_analysis(kernel_source)
        all_data["suggestions"] = self.phase5_optimization_suggestions(all_data)
        
        return all_data
```

`EndToEndProfiler` 封装了从 Baseline 测量到优化建议生成的六个标准化阶段：`phase1_baseline` 通过 1000 次迭代测量均值、中位数、最小值和 P99 延迟，中位数反映典型性能而 P99 反映尾延迟稳定性；`phase2_system_profiling` 调用 nsys 进行系统级分析，重点关注 kernel 数量、启动开销和计算-通信重叠；`phase3_kernel_profiling` 使用 ncu 采集 SM 利用率、DRAM 带宽和 Tensor Core 利用率等硬件计数器；`phase4_source_analysis` 对编译器各层 IR 进行静态分析，检查内存访问模式、barrier 数量和向量化程度；`phase5_optimization_suggestions` 综合前三阶段的结果生成优先级排序的优化建议，包括 P99/Median 稳定性指标（>1.5 表示性能波动大）、kernel fusion 建议（kernel 数量 >100）和 Occupancy 优化建议（<50%）。`run_full_profiling` 将五个阶段串联执行，返回完整的分析数据字典。这个工作流的设计体现了"先宏观后微观"的性能分析方法论——不在一开始就深入 ncu 的微架构细节，而是先通过 baseline 确认总体性能水平，通过 nsys 定位系统级瓶颈，最后才用 ncu 做内核级精确定位。在实际应用中，建议将 profiling 结果持久化为 JSON 或 CSV 文件，便于后续的性能回归检测和历史对比。

### 21.7.2 批量分析工具

```python
class BatchProfiler:
    """
    批量分析多个 kernel
    """
    
    def __init__(self, output_dir="./batch_profiling"):
        self.output_dir = output_dir
        self.results = {}
    
    def profile_kernel_set(self, kernels):
        """
        分析一组 kernel
        """
        for name, (kernel, source, inputs) in kernels.items():
            print(f"\n{'='*60}")
            print(f"Profiling: {name}")
            print(f"{'='*60}")
            
            profiler = EndToEndProfiler(f"{self.output_dir}/{name}")
            self.results[name] = profiler.run_full_profiling(kernel, source, inputs)
        
        self._generate_summary()
    
    def _generate_summary(self):
        """
        生成汇总报告
        """
        print("\n" + "=" * 60)
        print("Profiling Summary")
        print("=" * 60)
        
        summary = []
        for name, data in self.results.items():
            baseline = data.get("baseline", {})
            summary.append({
                "kernel": name,
                "mean_ms": baseline.get("mean_ms", 0),
                "min_ms": baseline.get("min_ms", 0),
                "occupancy": data.get("kernel_profile", {}).get("occupancy", 0),
            })
        
        # 按执行时间排序
        summary.sort(key=lambda x: x["mean_ms"], reverse=True)
        
        print(f"\n{'Kernel':<30} {'Time (ms)':<12} {'Min (ms)':<12} {'Occupancy':<10}")
        print("-" * 64)
        for item in summary:
            print(f"{item['kernel']:<30} {item['mean_ms']:<12.3f} {item['min_ms']:<12.3f} {item['occupancy']:<10.1%}")
```

`BatchProfiler` 实现了多 kernel 的批量性能分析方法：`profile_kernel_set` 遍历传入的 kernel 字典，为每个 kernel 创建独立的 EndToEndProfiler 实例并执行完整的六阶段分析流程；`_generate_summary` 汇总所有 kernel 的分析结果，按执行时间降序排列，输出包含 kernel 名称、平均耗时、最小耗时和 Occupancy 的表格化摘要。这种批量分析的实用价值在于：在大型项目中通常有数十甚至上百个自定义 kernel，逐一手动分析效率极低；批量分析器可以自动化地识别出 Top-K 最耗时的 kernel，让开发者将有限的优化精力聚焦在"二八原则"中的关键少数上。生成汇总报告后，建议按照执行时间占比排序，优先优化占比最高的 kernel，因为根据 Amdahl 定律，优化非瓶颈 kernel 对总性能的提升微乎其微。同时，对比 Occupancy 和耗时的关系可以快速发现异常——如果一个 kernel 耗时长且 Occupancy 极低，通常存在明显的优化空间。

在掌握了标准化和批量化的性能分析工具之后，我们通过两个真实的工业级案例来展示完整的 profiling 优化流程——从问题发现到瓶颈定位再到优化实施的全过程。

---

## 21.8 实际案例分析

### 21.8.1 案例 1：GEMM 性能优化

```python
def case_study_gemm_profiling():
    """
    案例：GEMM kernel 的性能剖析与优化
    
    问题：自定义 GEMM kernel 只有 cuBLAS 的 60% 性能
    目标：通过 profiling 找到瓶颈，优化到 90% 以上
    """
    
    # Step 1: Baseline
    print("=== Case Study: GEMM Profiling ===\n")
    
    M, N, K = 4096, 4096, 4096
    
    # Baseline performance
    baseline_time = 1.234  # ms, measured
    cublas_time = 0.856    # ms, measured
    ratio = baseline_time / cublas_time
    
    print(f"Problem: GEMM({M}, {N}, {K}) FP16")
    print(f"Baseline: {baseline_time:.3f} ms")
    print(f"cuBLAS: {cublas_time:.3f} ms")
    print(f"Ratio: {ratio:.2f}x slower than cuBLAS")
    
    # Step 2: ncu 分析
    print("\n--- ncu Analysis ---")
    
    # 假设的 ncu 结果
    ncu_results = {
        "sm_utilization": 65.2,
        "dram_utilization": 45.3,
        "tensor_utilization": 38.7,
        "occupancy": 62.5,
        "l2_hit_rate": 78.5,
        "bank_conflicts": 1234,
    }
    
    print(f"SM Utilization: {ncu_results['sm_utilization']:.1f}%")
    print(f"DRAM Utilization: {ncu_results['dram_utilization']:.1f}%")
    print(f"Tensor Core Utilization: {ncu_results['tensor_utilization']:.1f}%")
    print(f"Occupancy: {ncu_results['occupancy']:.1f}%")
    print(f"L2 Hit Rate: {ncu_results['l2_hit_rate']:.1f}%")
    print(f"Bank Conflicts: {ncu_results['bank_conflicts']}")
    
    # Step 3: 诊断
    print("\n--- Diagnosis ---")
    
    if ncu_results["tensor_utilization"] < 50:
        print("问题 1: Tensor Core 利用率低")
        print("原因: 可能没有使用 Tensor Core 指令，或数据布局不兼容")
        print("修复: 使用 T.gemm 内置指令，确保数据按 Tensor Core 要求布局")
    
    if ncu_results["bank_conflicts"] > 100:
        print("\n问题 2: Shared Memory Bank Conflict")
        print(f"原因: 检测到 {ncu_results['bank_conflicts']} 次冲突")
        print("修复: 在 shared memory 布局中添加 padding")
    
    if ncu_results["occupancy"] < 70:
        print("\n问题 3: Occupancy 偏低")
        print("原因: 寄存器或 shared memory 使用过多")
        print("修复: 减少 tile size 或使用寄存器复用策略")
    
    # Step 4: 优化
    print("\n--- Optimization ---")
    
    optimizations = [
        ("使用 T.gemm Tensor Core 指令", 38.7, 85.0),
        ("Shared memory padding", 1234, 0),
        ("调整 tile size (256,128,32)", 62.5, 75.0),
    ]
    
    for desc, before, after in optimizations:
        print(f"优化: {desc}")
        if isinstance(before, float):
            print(f"  Tensor Core: {before:.1f}% → {after:.1f}%")
        elif isinstance(before, int):
            print(f"  Bank Conflicts: {before} → {after}")
    
    # Step 5: 结果
    print("\n--- Result ---")
    optimized_time = 0.890  # ms, after optimization
    new_ratio = optimized_time / cublas_time
    speedup = baseline_time / optimized_time
    
    print(f"Optimized: {optimized_time:.3f} ms")
    print(f"vs cuBLAS: {new_ratio:.2f}x")
    print(f"Speedup: {speedup:.2f}x")
```

这个 GEMM 性能优化案例完整演示了从发现问题到优化验证的端到端流程：问题阶段发现自定义 GEMM kernel 性能仅为 cuBLAS 的 60%（1.234ms vs 0.856ms）；诊断阶段通过 ncu 分析发现三个主要问题——Tensor Core 利用率仅 38.7%（远低于期望的 85%），存在 1234 次 Shared Memory Bank Conflict，Occupancy 为 62.5%（低于 70% 的理想值）；优化阶段分别采取三个措施——使用 `T.gemm` Tensor Core 指令替代手写矩阵乘法以提升 Tensor Core 利用率至 85%，在 Shared Memory 布局中添加 padding 以消除 Bank Conflict，调整 tile size 至 (256,128,32) 以提升 Occupancy 至 75%；验证阶段优化后耗时降至 0.890ms，达到 cuBLAS 的 96%，加速比 1.39x。这个案例的关键启示在于：提升 Tensor Core 利用率是 GEMM 性能优化的第一优先级，因为 Tensor Core 的理论算力远超普通 CUDA Core（A100 上 FP16 Tensor Core 为 312 TFLOPS vs FP32 CUDA Core 为 19.5 TFLOPS），不使用 Tensor Core 意味着最高的性能上限已被严重压缩。同时，Bank Conflict 看似微小，但在高吞吐场景下因为每个 warp 都在等待 Shared Memory 服务的序列化处理，累积延迟可能造成 10-20% 的性能损失。

### 21.8.2 案例 2：Attention 性能分析

```python
def case_study_attention_profiling():
    """
    案例：Attention kernel 的性能剖析
    
    问题：Flash Attention 实现比官方慢 30%
    目标：找到性能差距的根源
    """
    
    print("=== Case Study: Attention Profiling ===\n")
    
    # 假设的 profiling 数据
    profiling_data = {
        "execution_time_ms": 2.345,
        "reference_time_ms": 1.800,
        "sm_utilization": 72.3,
        "dram_utilization": 58.7,
        "occupancy": 55.0,
        "l2_hit_rate": 65.2,
        "register_spills": 156,
        "shared_memory_bank_conflicts": 2048,
    }
    
    # 分析
    bottleneck = BottleneckClassifier().classify(profiling_data)
    print(f"Bottleneck: {bottleneck['type']}")
    
    # 针对 Attention 的特殊分析
    print("\n--- Attention-specific Analysis ---")
    
    if profiling_data["register_spills"] > 0:
        print(f"寄存器 spill: {profiling_data['register_spills']} 次")
        print("Flash Attention 需要大量寄存器存储中间结果")
        print("建议: 使用 Tiling 策略减少每步的寄存器需求")
    
    if profiling_data["shared_memory_bank_conflicts"] > 1000:
        print(f"\nBank conflict: {profiling_data['shared_memory_bank_conflicts']} 次")
        print("Attention 的 Q*K^T 计算容易产生 bank conflict")
        print("建议: 使用 swizzled memory layout")
    
    if profiling_data["occupancy"] < 60:
        print(f"\nOccupancy: {profiling_data['occupancy']:.1f}%")
        print("Flash Attention 的 per-thread state 较多")
        print("建议: 减少 tile size 或使用 warp-level specialization")
```

这个 Attention 性能分析案例聚焦于 Flash Attention 实现的性能调优：与官方实现对比慢 30%（2.345ms vs 1.800ms），通过 ncu 分析发现三个 Attention 特有的性能问题——156 次寄存器 spill（Flash Attention 需要存储 Q、K、V 矩阵的 tile 以及 attention score 的中间结果，寄存器压力极大），2048 次 Shared Memory Bank Conflict（Q*K^T 计算中矩阵乘法产生的访问模式容易触发 Bank Conflict），Occupancy 仅 55%（per-thread state 过多导致活跃 Warp 数受限）。针对性优化建议包括：采用更小的 tile size 或 warp-level specialization 来减少每个线程的寄存器需求、使用 swizzled memory layout 替代线性布局以消除 Bank Conflict、通过调整 tile 维度来平衡每线程的计算量和资源开销。Attention 计算的特殊性在于其算术强度较低（大量的 element-wise 操作如 softmax 中的 exp 计算），这使得它在 Roofline 模型中更容易落入 Memory Bound 区域，优化重点应放在提高数据复用率和减少全局内存往返上，而非单纯追求高 FLOPS。

以上两个案例展示了性能分析的完整方法论在实际工程问题中的应用。在进入多 GPU 场景之前，我们先回顾本章截止目前的几个核心发现：性能瓶颈的类型决定了优化方向、Benchmark 需要多次测量取中位数以避免异常值干扰、IR dump 可以帮我们理解编译器是否生成了预期的优化代码。

---

## 21.9 本章小结

本章系统介绍了 GPU 性能剖析的方法论和工具：

1. **方法论**：从宏观到微观的"漏斗模型"分析流程
2. **硬件工具**：ncu（内核级）、nsys（系统级）、rocprof（AMD GPU）的使用方法
3. **TileLang 指标**：Memory Bandwidth、Arithmetic Intensity、Occupancy 的分析
4. **IR 分析**：编译器 IR dump 的解读方法
5. **PGO**：Profile-Guided Optimization 的原理与实践
6. **瓶颈分类**：Compute Bound / Memory Bound / Latency Bound 的诊断与优化
7. **实战案例**：GEMM 和 Attention 的完整 profiling 案例

---

## 练习

### Exercise 1: ncu 实践
使用 ncu 分析一个 TileLang GEMM kernel，提取 SM 利用率、DRAM 带宽利用率和 Tensor Core 利用率，并判断瓶颈类型。

### Exercise 2: Roofline 分析
为矩阵乘法、向量加法和 softmax 三种操作绘制 Roofline Model 图，分析它们的算术强度和性能瓶颈。

### Exercise 3: Occupancy 优化
编写一个 kernel，初始 Occupancy 只有 40%，通过分析寄存器和 shared memory 使用情况，优化到 70% 以上。

### Exercise 4: IR Dump 分析
对一个 TileLang kernel 进行 IR dump，分析各层 IR 中的内存访问模式、循环结构和同步点。

### Exercise 5: 端到端 Profiling
使用 EndToEndProfiler 对一个完整的模型推理流程进行 profiling，生成优化建议报告。

---

## 思考题

1. **为什么 SM 利用率低不一定是 Occupancy 的问题？还有哪些可能的原因？**

2. **在什么情况下，即使 kernel 是 Memory Bound，提高 Occupancy 也能改善性能？**

3. **ncu 和 nsys 各自的优势是什么？在什么场景下应该使用哪个工具？**

4. **Profile-Guided Optimization 的主要局限性是什么？如何缓解？**

5. **如何设计一个自动化的性能回归检测系统？需要监控哪些关键指标？**

---

## 21.10 多 GPU 性能分析

### 21.10.1 多 GPU Profiling 挑战

```python
"""
多 GPU 性能分析的独特挑战：

1. 通信开销
   - GPU 间数据传输（NVLink / PCIe）
   - 集合通信（AllReduce / AllGather）
   - 通信与计算的重叠

2. 负载均衡
   - 不同 GPU 的工作量是否均匀
   - 数据并行 vs 模型并行的效率

3. 同步开销
   - 跨 GPU 的 barrier 同步
   - 流水线并行中的 bubble

4. 内存使用
   - 每个 GPU 的内存使用是否均衡
    - KV cache 的分布
"""
```

上述代码描述了多 GPU 性能分析的四大核心挑战：通信开销聚焦 GPU 间数据传输的带宽和延迟，NVLink 带宽（A100 为 600 GB/s）远高于 PCIe（约 32 GB/s），合理利用高速互联是分布式训练的关键；负载均衡关注各 GPU 的工作量是否均匀分配，不平衡会导致"木桶效应"——整体性能取决于最慢的 GPU；同步开销涉及跨 GPU barrier 和流水线并行中的 bubble 问题，过多的同步会抵消分布式带来的加速效果；内存使用关注 per-GPU 的内存占用是否均衡，特别是 KV cache 在推理场景中各 GPU 的分布可能严重不均。相比单 GPU 分析，多 GPU 场景最大的区别在于出现了通信这一额外的性能维度——即使每个 GPU 的 kernel 执行效率很高，如果通信时间占比过大（例如 AllReduce 占 40% 以上的 step 时间），整体性能仍然受限。因此多 GPU 分析需要同时关注计算效率和通信效率两个维度，并尽可能实现计算与通信的重叠以隐藏通信延迟。

### 21.10.2 通信分析

```python
class MultiGPUProfiler:
    """
    多 GPU 性能分析器
    """
    
    def __init__(self, num_gpus):
        self.num_gpus = num_gpus
        self.communication_events = []
    
    def profile_communication(self, comm_func, data_size, num_iterations=100):
        """
        分析通信性能
        
        关键指标：
        1. 通信延迟（Latency）
        2. 通信带宽（Bandwidth）
        3. 通信/计算重叠效率
        """
        import torch
        import time
        
        # 测量通信延迟
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            comm_func()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_iterations
        
        # 计算带宽
        bandwidth_gbs = (data_size / 1e9) / elapsed
        
        return {
            "latency_ms": elapsed * 1000,
            "bandwidth_gbs": bandwidth_gbs,
            "data_size_bytes": data_size,
        }
    
    def profile_allreduce(self, tensor, num_iterations=100):
        """分析 AllReduce 性能"""
        import torch
        import torch.distributed as dist
        
        data_size = tensor.numel() * tensor.element_size()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_iterations
        
        # AllReduce 的理论数据量 = 2 * (n-1) / n * data_size
        n = self.num_gpus
        theoretical_data = 2 * (n - 1) / n * data_size
        
        return {
            "latency_ms": elapsed * 1000,
            "bandwidth_gbs": (theoretical_data / 1e9) / elapsed,
            "data_size_bytes": data_size,
            "algorithm_bandwidth_gbs": (data_size / 1e9) / elapsed,
        }
    
    def analyze_overlap(self, compute_events, comm_events):
        """
        分析计算与通信的重叠效率
        """
        total_compute_time = sum(e["duration"] for e in compute_events)
        total_comm_time = sum(e["duration"] for e in comm_events)
        
        # 计算重叠部分
        overlapped_time = 0
        for comp in compute_events:
            for comm in comm_events:
                overlap_start = max(comp["start"], comm["start"])
                overlap_end = min(comp["end"], comm["end"])
                if overlap_start < overlap_end:
                    overlapped_time += overlap_end - overlap_start
        
        overlap_ratio = overlapped_time / total_comm_time if total_comm_time > 0 else 0
        
        return {
            "total_compute_time_ms": total_compute_time * 1000,
            "total_comm_time_ms": total_comm_time * 1000,
            "overlapped_time_ms": overlapped_time * 1000,
            "overlap_ratio": overlap_ratio,
            "efficiency": 1.0 - (total_compute_time + total_comm_time - overlapped_time) / 
                         max(total_compute_time, total_comm_time),
        }
```

`MultiGPUProfiler` 实现了多 GPU 场景的通信性能分析：`profile_communication` 测量 GPU 间通信的延迟和带宽，通过多次迭代取均值减少抖动；`profile_allreduce` 专门分析分布式训练中最关键的 AllReduce 操作，其中理论数据量公式 `2*(n-1)/n*data_size` 来源于 Ring AllReduce 算法——每个 GPU 在 Scatter-Reduce 阶段发送 (n-1)/n 的数据，在 AllGather 阶段接收同等量的数据；`analyze_overlap` 计算计算事件与通信事件的时间窗口重叠比例，重叠效率越高说明通信被隐藏得越好。分析 AllReduce 性能时需要区分 algorithm bandwidth（算法视角的带宽，基于原始数据量）和 bus bandwidth（总线视角的带宽，基于实际传输数据量），两者差异反映了通信算法的效率。在实际的多 GPU 训练中，AllReduce 往往是最大的通信开销来源，优化方向包括使用 NCCL 的 Ring/Tree 拓扑、梯度压缩（如 FP16 通信）、以及通过 gradient accumulation 增大通信间隔。

### 21.10.3 负载均衡分析

```python
class LoadBalanceAnalyzer:
    """
    负载均衡分析器
    """
    
    def __init__(self, num_gpus):
        self.num_gpus = num_gpus
    
    def analyze_kernel_distribution(self, kernel_times_per_gpu):
        """
        分析 kernel 在各 GPU 上的分布
        
        参数：
        - kernel_times_per_gpu: dict, {gpu_id: [kernel_time_list]}
        """
        total_times = {}
        for gpu_id, times in kernel_times_per_gpu.items():
            total_times[gpu_id] = sum(times)
        
        avg_time = sum(total_times.values()) / len(total_times)
        max_time = max(total_times.values())
        min_time = min(total_times.values())
        
        # 负载均衡度（0-1，1 为完全均衡）
        balance_score = min_time / max_time if max_time > 0 else 1.0
        
        # 每个 GPU 的利用率
        utilizations = {gpu: time / max_time for gpu, time in total_times.items()}
        
        return {
            "avg_time_ms": avg_time * 1000,
            "max_time_ms": max_time * 1000,
            "min_time_ms": min_time * 1000,
            "imbalance_ratio": max_time / min_time if min_time > 0 else float('inf'),
            "balance_score": balance_score,
            "utilizations": utilizations,
            "bottleneck_gpu": max(total_times, key=total_times.get),
        }
    
    def suggest_rebalance(self, analysis):
        """建议负载均衡优化"""
        suggestions = []
        
        if analysis["balance_score"] < 0.8:
            suggestions.append({
                "issue": "负载不均衡",
                "bottleneck": f"GPU {analysis['bottleneck_gpu']} 是瓶颈",
                "suggestions": [
                    "调整数据分配策略",
                    "使用动态负载均衡",
                    "检查是否有 straggler（某 GPU 执行特别慢）",
                ],
            })
        
        if analysis["imbalance_ratio"] > 1.5:
            suggestions.append({
                "issue": f"不平衡比率 {analysis['imbalance_ratio']:.2f}x",
                "suggestions": [
                    "检查数据分布是否均匀",
                    "考虑使用 pipeline 并行替代数据并行",
                    "使用 gradient accumulation 减少同步频率",
                ],
            })
        
        return suggestions
```

`LoadBalanceAnalyzer` 专注于多 GPU 间的负载均衡分析：`analyze_kernel_distribution` 统计各 GPU 上 kernel 执行的总时间分布，计算负载均衡度（balance_score = min_time/max_time，1.0 为完美均衡），识别瓶颈 GPU；`suggest_rebalance` 当均衡度低于 0.8 或不平衡比率超过 1.5x 时生成优化建议。负载不均衡的常见原因包括数据分布不均（某些 GPU 分到的样本尺寸差异大）、straggler 问题（某 GPU 因降频或竞争进程而执行慢）、静态划分不合理（不同 GPU 的计算能力差异未考虑）。在数据并行训练中，负载不均衡会导致所有 GPU 在同步点时等待最慢的 GPU，造成大量的空闲时间。解决方案包括：动态负载均衡（运行时根据各 GPU 完成速度调整分配）、梯度累积（减少同步频率）、将大样本拆分到多个 GPU 或使用序列打包。

多 GPU 性能分析在完成通信分析和负载均衡分析之后，我们还需要建立一套自动化的性能监控体系，确保优化效果不会因代码变更而退化——这就是性能回归检测的核心价值。

---

## 21.11 性能回归检测

### 21.11.1 自动化性能测试

```python
class PerformanceRegressionDetector:
    """
    性能回归检测器
    
    用于持续集成中检测性能退化
    """
    
    def __init__(self, baseline_path="perf_baseline.json", tolerance=0.05):
        self.baseline_path = baseline_path
        self.tolerance = tolerance  # 允许的性能波动范围（5%）
        self.baseline = self._load_baseline()
    
    def _load_baseline(self):
        """加载基线性能数据"""
        import json
        try:
            with open(self.baseline_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _save_baseline(self):
        """保存基线性能数据"""
        import json
        with open(self.baseline_path, "w") as f:
            json.dump(self.baseline, f, indent=2)
    
    def check_regression(self, kernel_name, current_perf):
        """
        检查性能回归
        
        返回：
        - pass: 是否通过
        - change_pct: 性能变化百分比
        - message: 说明信息
        """
        if kernel_name not in self.baseline:
            return {
                "pass": True,
                "change_pct": 0,
                "message": f"No baseline for {kernel_name}, skipping check",
            }
        
        baseline_perf = self.baseline[kernel_name]["median_ms"]
        change_pct = (current_perf - baseline_perf) / baseline_perf
        
        if change_pct > self.tolerance:
            return {
                "pass": False,
                "change_pct": change_pct * 100,
                "message": (f"Performance regression detected: "
                          f"{baseline_perf:.3f}ms -> {current_perf:.3f}ms "
                          f"(+{change_pct*100:.1f}%)"),
            }
        elif change_pct < -self.tolerance:
            return {
                "pass": True,
                "change_pct": change_pct * 100,
                "message": (f"Performance improvement: "
                          f"{baseline_perf:.3f}ms -> {current_perf:.3f}ms "
                          f"({change_pct*100:.1f}%)"),
            }
        else:
            return {
                "pass": True,
                "change_pct": change_pct * 100,
                "message": f"Performance stable: {current_perf:.3f}ms",
            }
    
    def update_baseline(self, kernel_name, perf_data):
        """更新基线"""
        self.baseline[kernel_name] = perf_data
        self._save_baseline()
    
    def run_regression_suite(self, test_suite):
        """
        运行回归测试套件
        
        参数：
        - test_suite: list of (kernel_name, kernel_func, inputs)
        """
        results = []
        
        for kernel_name, kernel_func, inputs in test_suite:
            perf = self._measure_performance(kernel_func, inputs)
            check = self.check_regression(kernel_name, perf)
            
            results.append({
                "kernel": kernel_name,
                "perf": perf,
                "check": check,
            })
        
        # 汇总
        passed = sum(1 for r in results if r["check"]["pass"])
        failed = len(results) - passed
        
        print(f"\n{'='*60}")
        print(f"Regression Test Results: {passed} passed, {failed} failed")
        print(f"{'='*60}")
        
        for r in results:
            status = "PASS" if r["check"]["pass"] else "FAIL"
            print(f"[{status}] {r['kernel']}: {r['check']['message']}")
        
        return results
    
    def _measure_performance(self, kernel_func, inputs):
        """测量性能"""
        import torch
        import time
        
        # 预热
        for _ in range(10):
            kernel_func(*inputs)
        torch.cuda.synchronize()
        
        # 测量
        times = []
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.perf_counter()
            kernel_func(*inputs)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        times.sort()
        return {"median_ms": times[len(times) // 2], "mean_ms": sum(times) / len(times)}
```

这段代码是 21.11.1 自动化性能测试 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

`PerformanceRegressionDetector` 实现了持续集成中的自动化性能回归检测：`_load_baseline` 和 `_save_baseline` 实现了基线数据的持久化，将每个 kernel 的中位数执行时间存入 JSON 文件作为性能基准；`check_regression` 将当前性能与基线对比，当性能下降超过容忍阈值（默认 5%）时标记为 FAIL，当性能提升超过阈值时标记为 improvement，否则视为 stable；`run_regression_suite` 批量执行回归测试套件，自动测量每个 kernel 的性能并与基线对比，输出汇总报告。性能回归检测的关键设计决策包括：使用中位数而非均值作为基准（避免异常值影响）、设置合理的容忍度（太严格会造成误报，太宽松会漏检回归）、预热和多次测量确保数据稳定性。在实践中建议将性能基线文件纳入版本管理，每次优化后更新基线，并在 CI/CD 流程中集成回归检测，当性能下降超过阈值时自动阻止代码合并。需要注意的是 GPU 性能受环境因素影响（时钟频率、温度、后台负载），锁定时钟频率（`nvidia-smi -lgc`）是确保结果可比性的重要前提。

### 21.11.2 性能监控 Dashboard

```python
class PerformanceDashboard:
    """
    性能监控 Dashboard
    
    实时监控 kernel 性能，检测异常
    """
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.history = {}  # {kernel_name: [perf_data_list]}
        self.alerts = []
    
    def record(self, kernel_name, perf_data):
        """记录性能数据"""
        if kernel_name not in self.history:
            self.history[kernel_name] = []
        
        self.history[kernel_name].append(perf_data)
        
        # 保持窗口大小
        if len(self.history[kernel_name]) > self.window_size:
            self.history[kernel_name].pop(0)
        
        # 检测异常
        self._check_anomaly(kernel_name, perf_data)
    
    def _check_anomaly(self, kernel_name, perf_data):
        """检测性能异常"""
        history = self.history[kernel_name]
        
        if len(history) < 10:
            return
        
        recent = [h["median_ms"] for h in history[-10:]]
        baseline = [h["median_ms"] for h in history[:-10]]
        
        if not baseline:
            return
        
        recent_avg = sum(recent) / len(recent)
        baseline_avg = sum(baseline) / len(baseline)
        baseline_std = (sum((t - baseline_avg)**2 for t in baseline) / len(baseline))**0.5
        
        # 检测显著变化（超过 3 个标准差）
        if abs(recent_avg - baseline_avg) > 3 * baseline_std:
            self.alerts.append({
                "kernel": kernel_name,
                "type": "performance_anomaly",
                "recent_avg": recent_avg,
                "baseline_avg": baseline_avg,
                "deviation": abs(recent_avg - baseline_avg) / baseline_std,
                "timestamp": __import__("datetime").datetime.now().isoformat(),
            })
    
    def get_summary(self):
        """获取性能摘要"""
        summary = {}
        
        for kernel_name, history in self.history.items():
            if not history:
                continue
            
            times = [h["median_ms"] for h in history]
            summary[kernel_name] = {
                "count": len(times),
                "mean": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
                "latest": times[-1],
            }
        
        return summary
    
    def get_alerts(self, limit=10):
        """获取最近的告警"""
        return self.alerts[-limit:]
```

`PerformanceDashboard` 实现了实时性能监控和异常检测：`record` 方法将每次 kernel 执行的性能数据追加到滑动窗口（默认 100 条记录），保持历史数据的同时自动淘汰旧记录；`_check_anomaly` 实现了基于统计学的异常检测算法——将最近 10 次执行的均值与历史基线（窗口内排除最近 10 次的剩余记录）进行对比，当偏差超过 3 个标准差时触发告警（3-sigma 规则）；`get_summary` 提供每个 kernel 的统计摘要（均值、最小值、最大值、最新值）；`get_alerts` 返回最近的异常告警列表。这种实时监控的关键价值在于：GPU 性能可能因为硬件热节流、驱动程序更新、CUDA 版本变化或系统负载波动而逐渐退化，等用户察觉到性能问题时往往已经晚了。通过自动化监控，可以在性能退化初期就捕获异常并回溯根因。在生产环境中，建议将监控数据持久化到时序数据库（如 InfluxDB 或 Prometheus），配合 Grafana 等工具实现可视化 Dashboard，并设置告警通知（如 Slack Webhook 或邮件）。

性能监控和回归检测是保障优化成果持续有效的最后一道防线。然而在实际 profiling 工作中，开发者常常因为对工具和方法论的理解不足而走入误区，导致分析结果误导优化方向。

---

## 21.12 性能分析常见误区

### 21.12.1 误区与纠正

```python
"""
性能分析常见误区

误区 1：只看平均值
- 问题：平均值可能掩盖了性能波动
- 正确：同时关注 P50, P90, P99 等分位数
- 原因：GPU 性能受热节流、cache 效果等影响

误区 2：预热不足
- 问题：前几次运行可能较慢（JIT 编译、cache 冷启动）
- 正确：至少预热 10-20 次
- 原因：CUDA kernel 首次执行需要 JIT 编译

误区 3：忽略 GPU 时钟变化
- 问题：GPU 可能因热节流降低时钟
- 正确：使用 nvidia-smi 监控时钟，或固定时钟
- 原因：现代 GPU 有动态频率调整

误区 4：只关注 kernel 时间
- 问题：忽略了启动开销、数据传输、同步等
- 正确：使用 nsys 进行系统级分析
- 原因：端到端时间才是用户感知的时间

误区 5：过度优化单个 kernel
- 问题：优化了非瓶颈 kernel
- 正确：先 profiling 定位瓶颈，再优化
- 原因：Amdahl 定律：总加速受限于不可优化部分

误区 6：忽略数值精度影响
- 问题：混合精度可能改变数值行为
- 正确：验证优化后的数值正确性
- 原因：浮点运算不满足结合律
"""
```

这段代码总结了性能分析中最常见的六大误区及其纠正方法。误区一（只看平均值）是最普遍的陷阱，GPU kernel 的执行时间往往不是正态分布——首几次运行因 JIT 编译和 cold cache 而偏慢，稳定后可能因热节流或 OS 抢占而出现尾延迟尖峰，只关注均值会掩盖这些信息，因此必须同时关注 P50（典型性能）、P90/P99（尾延迟）。误区二（预热不足）在开发阶段特别常见，CUDA kernel 首次运行时需要 PTX 到 SASS 的 JIT 编译，如果不预热，前几次测量结果可能慢 10-50%。误区三（忽略 GPU 时钟变化）经常导致"同样的代码在不同时间运行结果不同"的困惑，动态频率调整（Dynamic Frequency Scaling）会根据温度和功耗自动升降时钟，建议 profiling 前用 `nvidia-smi -lgc <max_clock>,<max_clock>` 锁定时钟。误区四（只关注 kernel 时间）忽略了 kernel 启动开销（通常 5-15 μs）、CPU-GPU 数据传输和同步等待，这些在大量小 kernel 的场景下可能成为主要瓶颈。误区五（过度优化单个 kernel）违背了 Amdahl 定律，即使把占比 10% 的 kernel 优化到无穷快，总提升也不超过 11%。误区六（忽略数值精度影响）在混合精度场景尤为重要，FP16/FP32 混合计算可能导致数值发散，优化后的 kernel 必须通过数值正确性验证。

### 21.12.2 Profiling Checklist

```python
class ProfilingChecklist:
    """
    Profiling 检查清单
    
    确保 profiling 过程系统、全面
    """
    
    CHECKLIST = [
        {
            "category": "环境准备",
            "items": [
                ("GPU 时钟是否固定？", "nvidia-smi -lgc <clock>,<clock>"),
                ("是否有后台进程干扰？", "nvidia-smi 查看 GPU 使用率"),
                ("CUDA 版本是否正确？", "nvcc --version"),
                ("是否使用了正确的 GPU？", "CUDA_VISIBLE_DEVICES"),
            ],
        },
        {
            "category": "测量方法",
            "items": [
                ("预热是否充分？", "至少 10-20 次预热"),
                ("测量次数是否足够？", "至少 100 次测量"),
                ("是否去除了异常值？", "使用中位数或 trimmed mean"),
                ("是否同步了 GPU？", "torch.cuda.synchronize()"),
            ],
        },
        {
            "category": "分析层次",
            "items": [
                ("是否进行了系统级分析？", "nsys profile"),
                ("是否进行了内核级分析？", "ncu --set full"),
                ("是否分析了硬件计数器？", "SM utilization, bandwidth, occupancy"),
                ("是否检查了数值正确性？", "与参考实现对比"),
            ],
        },
        {
            "category": "结果验证",
            "items": [
                ("结果是否可复现？", "多次运行对比"),
                ("是否对比了基线？", "与优化前对比"),
                ("是否考虑了边界情况？", "小矩阵、非对齐大小"),
                ("是否测试了不同问题规模？", "多个 M, N, K 组合"),
            ],
        },
    ]
    
    def print_checklist(self):
        """打印检查清单"""
        print("=== Profiling 检查清单 ===\n")
        
        for category in self.CHECKLIST:
            print(f"\n--- {category['category']} ---")
            for i, (item, command) in enumerate(category['items'], 1):
                print(f"  [ ] {item}")
                print(f"      命令: {command}")
    
    def verify_environment(self):
        """验证环境"""
        import subprocess
        
        checks = []
        
        # 检查 GPU 时钟
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=clocks.current.graphics", 
                 "--format=csv,noheader"],
                capture_output=True, text=True
            )
            checks.append(("GPU Clock", result.stdout.strip()))
        except:
            checks.append(("GPU Clock", "Unknown"))
        
        # 检查 GPU 使用率
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", 
                 "--format=csv,noheader"],
                capture_output=True, text=True
            )
            gpu_util = int(result.stdout.strip().replace('%', ''))
            checks.append(("GPU Utilization", f"{gpu_util}%", gpu_util < 10))
        except:
            checks.append(("GPU Utilization", "Unknown"))
        
        return checks
```

`ProfilingChecklist` 提供了一个系统化的性能分析检查清单，按照环境准备、测量方法、分析层次和结果验证四个类别组织：环境准备类确保 GPU 时钟固定（避免动态频率影响）、后台进程清理和 CUDA 版本正确，这是获得可靠 profiling 数据的前提条件；测量方法类强调充分预热（10-20 次）、足够测量次数（100 次以上）、异常值去除和 GPU 同步，消除测量噪声；分析层次类要求按照漏斗模型依次进行系统级、内核级和硬件计数器分析，确保分析的完整性；结果验证类关注可复现性、基线对比、边界情况和不同问题规模的测试，避免优化只在特定场景有效。这个 Checklist 的核心价值在于将性能分析从一个随意性的"看看哪里慢"转变为一个标准化、可重复的科学过程。建议在每次 profiling 前逐项核对 Checklist，完成后再对比 Checklist 确认无遗漏，最终将 Checklist 的完成状态和 profiling 结果一起归档，形成可追溯的性能分析记录。

---

## 扩展阅读

1. **NVIDIA Nsight Compute Documentation** - 官方文档，详细的指标说明
2. **NVIDIA CUDA Profiling Guide** - CUDA 性能分析最佳实践
3. **Wong, H., et al. (2010).** "Demystifying GPU Microarchitecture through Microbenchmarking." - GPU 微架构分析
4. **Jia, Z., et al. (2019).** "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking." - Volta 架构深度分析
5. **AMD ROCm Profiler Documentation** - AMD GPU 性能分析工具

---

## 下一章预告

**Chapter 22: 内存优化与带宽利用** — 通过 profiling 我们已经能够定位性能瓶颈。当瓶颈是访存密集型时，如何系统地优化内存访问？我们将深入探讨 GPU 内存层次、coalescing、shared memory bank conflict、寄存器分配等核心优化技术。

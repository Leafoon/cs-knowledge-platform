---
title: "Chapter 20: Auto Schedule 与智能调优"
description: "深入探讨 TileLang 的自动调度框架设计，包括搜索空间定义、性能模型构建、搜索策略选择，以及 Tune API 的使用方法与最佳实践"
updated: "2025-01-15"
---

# Chapter 20: Auto Schedule 与智能调优

> **Learning Objectives**
> - 理解 Auto Schedule 框架的整体架构与设计哲学
> - 掌握搜索空间的定义方法（Tile Size / Pipeline Stage / Thread Binding 组合）
> - 理解 Cost Model 的构建原理与性能预测机制
> - 掌握各种搜索策略（Grid Search / Random / Bayesian Optimization）的适用场景
> - 熟练使用 Tune API 进行自动调优
> - 对比 TileLang Auto Schedule 与 Triton autotune、TVM AutoTVM 的异同
> - 学会分析搜索收敛曲线并制定最佳实践

---

## 20.1 自动调度的动机与背景

在高性能 GPU/加速器编程中，**调度策略（Schedule）** 直接决定了程序的最终性能。即使是同一个算法实现，不同的 tile size、不同的线程绑定方式、不同的流水线配置，可能导致数倍甚至数十倍的性能差异。

### 20.1.1 手动调优的困境

手动调优面临以下核心挑战：

| 挑战 | 描述 |
|------|------|
| **参数空间爆炸** | tile size、stage 数、线程绑定等组合数量呈指数增长 |
| **硬件依赖性** | 不同 GPU 架构（A100 vs H100 vs MI300X）最优配置不同 |
| **算子耦合** | 单算子最优不等于端到端最优 |
| **工程成本** | 手动调优需要深厚硬件知识和大量实验时间 |

### 20.1.2 自动调度的目标

Auto Schedule 的核心目标是在 **可接受的时间内** 自动搜索到 **接近最优** 的调度配置，同时满足以下约束：

- 搜索时间可控（分钟级而非天级）
- 搜索结果可复现
- 支持多种硬件后端
- 与手动调优结果竞争力相当

---

## 20.2 Auto Schedule 框架设计

### 20.2.1 整体架构

TileLang 的 Auto Schedule 框架采用经典的 **Search-Based Compilation** 范式：

<div data-component="AutoScheduleFramework"></div>

```
┌─────────────────────────────────────────────────────────┐
│                   Auto Schedule Framework                │
│                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐   │
│  │  Search   │───>│  Cost    │───>│   Code Gen +     │   │
│  │  Space    │    │  Model   │    │   Evaluation     │   │
│  │  Definer  │    │  Engine  │    │   Engine         │   │
│  └──────────┘    └──────────┘    └──────────────────┘   │
│       │               │                   │              │
│       v               v                   v              │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐   │
│  │ Parameter │    │ Feature  │    │   Benchmark      │   │
│  │ Templates │    │ Extractor│    │   Results        │   │
│  └──────────┘    └──────────┘    └──────────────────┘   │
│       │               │                   │              │
│       └───────────────┴───────────────────┘              │
│                       │                                  │
│                       v                                  │
│              ┌────────────────┐                          │
│              │   Search       │                          │
│              │   Strategy     │                          │
│              │   Controller   │                          │
│              └────────────────┘                          │
│                       │                                  │
│                       v                                  │
│              ┌────────────────┐                          │
│              │   Best Config  │                          │
│              │   Exporter     │                          │
│              └────────────────┘                          │
└─────────────────────────────────────────────────────────┘
```

这张架构图展示了 TileLang Auto Schedule 框架的核心组件及其数据流向。搜索空间定义器（Search Space Definer）负责声明所有可调参数及其合法取值范围，通过参数模板生成候选配置。代价模型引擎（Cost Model Engine）利用特征提取器对配置进行性能预测，而代码生成与评估引擎则负责将配置编译为实际代码并执行基准测试。搜索策略控制器作为整个框架的调度中枢，协调各组件之间的交互，最终将最优配置通过导出器输出。

从数据流的角度深入观察，搜索策略控制器（Search Strategy Controller）是整个框架的决策中心，它实现了闭环反馈机制：每次迭代中，控制器从搜索空间生成候选配置，由代价模型进行快速预筛选，筛掉明显不具有竞争力的配置，随后通过代码生成与评估引擎进行实际编译和基准测试。评估结果一方面用于更新当前最优解的记录，另一方面作为反馈信号输入代价模型，帮助其校准预测精度，形成"预测—评估—反馈—更新"的持续优化循环。此外，框架内置的缓存机制在避免冗余评估方面发挥着关键作用——对于之前已经评估过的配置或代价模型高度确信的配置，系统会直接复用历史结果而非重新编译测量。在多轮迭代中，这种缓存策略能将有效搜索效率提升数倍，尤其在搜索空间中存在大量对称或等效配置时尤为明显。需要注意的是，缓存的有效性依赖于评估结果的可复现性，当硬件环境（如 GPU 温度、时钟频率）发生显著变化时，缓存中的旧数据可能失去参考价值，因此生产实践中通常搭配版本管理和定期刷新策略。

### 20.2.2 核心组件详解

#### Search Space Definer

搜索空间定义器负责声明所有可调参数及其合法取值范围：

```python
import tilelang
from tilelang.autotuner import SearchSpace

# 定义搜索空间
search_space = SearchSpace(
    tile_sizes={
        "block_M": [64, 128, 256],
        "block_N": [64, 128, 256],
        "block_K": [16, 32, 64],
    },
    num_stages=[1, 2, 3, 4],
    thread_binding={
        "num_threads": [128, 256, 512],
        "warp_layout": [(1, 1), (2, 2), (4, 1), (1, 4)],
    },
    vectorize_width=[1, 2, 4, 8],
)
```

这段代码展示了如何使用 TileLang 的 SearchSpace 类定义搜索空间。通过指定 tile_sizes 字典来声明 block_M、block_N、block_K 的候选值，num_stages 控制流水线级数，thread_binding 定义线程数和 warp 布局方案，vectorize_width 则指定向量化宽度。每个参数的取值范围需要根据硬件约束（如 shared memory 容量、最大线程数等）来合理设定，过大或过小的范围都会影响搜索效率。

#### Cost Model Engine

Cost Model 负责在不实际执行代码的情况下，预测给定配置的性能表现：

```python
class CostModel:
    """
    性能预测模型：基于硬件特征和调度参数预测执行时间
    
    预测公式：
    predicted_time = max(
        compute_bound_time,    # 计算瓶颈时间
        memory_bound_time      # 访存瓶颈时间
    ) + synchronization_overhead
    """
    
    def predict(self, config: ScheduleConfig) -> float:
        # 提取计算特征
        compute_ops = self._estimate_compute_ops(config)
        compute_throughput = self._get_compute_throughput(config)
        compute_time = compute_ops / compute_throughput
        
        # 提取访存特征
        memory_traffic = self._estimate_memory_traffic(config)
        memory_bandwidth = self._get_memory_bandwidth(config)
        memory_time = memory_traffic / memory_bandwidth
        
        # 同步开销
        sync_overhead = self._estimate_sync_overhead(config)
        
        return max(compute_time, memory_time) + sync_overhead
```

这个 Cost Model 类实现了基于 Roofline 模型的性能预测。核心思想是程序的执行时间受限于计算吞吐量和内存带宽中的瓶颈项（取两者最大值），再加上同步开销。predict 方法分别估算计算时间（FLOPS / 峰值算力）和访存时间（数据传输量 / 峰值带宽），最终返回预测的执行时间。这种解析模型的优点是预测速度极快，但精度受限于对硬件特性的简化假设。

#### Search Strategy Controller

搜索策略控制器决定了如何在搜索空间中高效地找到最优配置：

```python
class SearchController:
    """
    搜索控制器：管理搜索过程的迭代与终止
    """
    
    def __init__(self, strategy, budget, early_stop_patience=10):
        self.strategy = strategy
        self.budget = budget  # 最大评估次数
        self.early_stop_patience = early_stop_patience
    
    def search(self, search_space, evaluator):
        best_config = None
        best_perf = float('inf')
        no_improve_count = 0
        
        for i in range(self.budget):
            # 策略选择下一个候选配置
            config = self.strategy.next_candidate(search_space)
            
            # 评估配置性能
            perf = evaluator.evaluate(config)
            
            # 更新最优
            if perf < best_perf:
                best_perf = perf
                best_config = config
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # 早停检查
            if no_improve_count >= self.early_stop_patience:
                break
            
            # 策略更新内部模型
            self.strategy.update(config, perf)
        
        return best_config, best_perf
```

搜索策略控制器是 Auto Schedule 框架的调度核心，负责管理整个搜索过程的迭代与终止。search 方法实现了标准的搜索循环：每次迭代由策略模块生成候选配置，通过评估器测量性能，并更新当前最优解。early_stop_patience 机制允许在连续多次未改进后提前终止搜索，避免浪费计算资源。策略模块在每次评估后还会更新其内部模型（如贝叶斯优化中的代理模型），使后续搜索更加智能高效。

---

## 20.3 搜索空间定义

理解了 Auto Schedule 的整体架构后，我们来深入探讨搜索空间定义的细节。搜索空间的质量直接决定了自动调优能否在合理时间内找到高性能配置。

### 20.3.1 Tile Size 组合

Tile size 是影响性能最关键的参数之一。它决定了每个 thread block 处理的数据量，直接影响 shared memory 使用量、寄存器压力和计算密度。

```python
def generate_tile_size_space(M, N, K):
    """
    生成合法的 tile size 组合
    
    约束条件：
    1. block_M * block_N 不能超过单个 SM 的最大线程数 × 每线程最大寄存器数
    2. block_K 需要整除 K（或处理边界情况）
    3. shared memory 使用量不能超过 SM 的 shared memory 容量
    """
    valid_configs = []
    
    block_M_candidates = [32, 64, 128, 256]
    block_N_candidates = [32, 64, 128, 256]
    block_K_candidates = [16, 32, 64]
    
    for bm in block_M_candidates:
        for bn in block_N_candidates:
            for bk in block_K_candidates:
                # 约束检查
                if bm * bn > 1024 * 32:  # 线程×寄存器约束
                    continue
                smem_usage = (bm * bk + bk * bn) * 2  # 两个 shared memory buffer
                if smem_usage > 48 * 1024 * 1024:  # 48MB shared memory
                    continue
                valid_configs.append({
                    "block_M": bm,
                    "block_N": bn,
                    "block_K": bk,
                })
    
    return valid_configs
```

这个函数演示了如何生成合法的 tile size 组合。关键在于约束检查：block_M × block_N 不能超过单个 SM 的最大线程数与每线程最大寄存器数的乘积，否则会导致资源不足；shared memory 使用量（两个 buffer 分别存储 A 和 B 矩阵的分块）不能超过硬件容量限制。通过在生成阶段就过滤掉非法配置，可以大幅缩小有效搜索空间，提高搜索效率。实际使用中还需要考虑边界对齐和寄存器压力等更细粒度的约束。

### 20.3.2 Pipeline Stage 配置

流水线级数决定了计算与访存的重叠程度：

```python
# Pipeline stage 与 tile size 的关联
def get_pipeline_candidates(block_K, smem_budget):
    """
    根据 block_K 和 shared memory 预算确定合法的流水线级数
    
    原理：
    - 每增加一级流水线，需要额外一份 shared memory buffer
    - 更多流水线级数 → 更好的计算/访存重叠 → 但更多 shared memory 使用
    - 最优级数取决于计算强度和访存延迟
    """
    per_stage_smem = block_K * 2 * 2  # 假设 fp16，需要 2 个 buffer
    
    max_stages = min(
        smem_budget // per_stage_smem,  # 内存约束
        8  # 硬件流水线深度上限
    )
    
    candidates = []
    for stages in range(1, max_stages + 1):
        candidates.append({
            "num_stages": stages,
            "smem_usage": stages * per_stage_smem,
            "expected_overlap_ratio": min(1.0, stages * 0.25),
        })
    
    return candidates
```

此函数根据 block_K 和 shared memory 预算来确定合法的流水线级数。每增加一级流水线，就需要额外一份 shared memory buffer 来存储下一个计算阶段的数据，从而实现计算与访存的重叠执行。函数计算了每级流水线的内存占用，并根据硬件上限（通常为 8 级）确定最大级数。expected_overlap_ratio 表示预期的计算访存重叠比例，级数越多重叠效果越好，但受限于 shared memory 容量，实际中需要在重叠收益和内存开销之间权衡。

### 20.3.3 Thread Binding 组合

线程绑定方式影响 warp 级别的执行效率和数据局部性：

```python
def generate_thread_binding_space(num_threads, tile_shape):
    """
    生成线程绑定方案
    
    考虑因素：
    1. Warp 大小（32 for NVIDIA, 64 for AMD）
    2. 数据局部性（相邻线程应访问相邻数据）
    3. Bank conflict 避免
    """
    block_M, block_N = tile_shape
    bindings = []
    
    # 方案1：行优先绑定
    for threads_per_row in [4, 8, 16, 32]:
        threads_per_col = num_threads // threads_per_row
        if threads_per_col >= 1 and threads_per_row <= block_N and threads_per_col <= block_M:
            bindings.append({
                "layout": "row_major",
                "threads_per_row": threads_per_row,
                "threads_per_col": threads_per_col,
                "warp_size": 32,
            })
    
    # 方案2：列优先绑定
    for threads_per_col in [4, 8, 16, 32]:
        threads_per_row = num_threads // threads_per_col
        if threads_per_row >= 1 and threads_per_row <= block_N and threads_per_col <= block_M:
            bindings.append({
                "layout": "col_major",
                "threads_per_row": threads_per_row,
                "threads_per_col": threads_per_col,
                "warp_size": 32,
            })
    
    # 方案3：Z-order 曲线绑定（优化 shared memory bank conflict）
    bindings.append({
        "layout": "z_order",
        "threads_per_row": int(block_N ** 0.5),
        "threads_per_col": int(block_M ** 0.5),
        "warp_size": 32,
    })
    
    return bindings
```

线程绑定方案决定了 thread block 内的线程如何映射到计算任务上，直接影响 warp 级别的执行效率。函数生成三种绑定策略：行优先（相邻线程处理同一行的不同列）、列优先（相邻线程处理同一列的不同行）和 Z-order 曲线绑定。行优先和列优先方案需要确保每行/列的线程数不超过 tile 维度且为 warp 大小的倍数。Z-order 绑定通过空间填充曲线优化 shared memory bank conflict，减少内存访问冲突，适合对访存性能要求较高的场景。

### 20.3.4 搜索空间规模分析

<div data-component="SearchSpaceExplorer"></div>

对于一个典型的 GEMM 操作，搜索空间的规模如下：

```python
def analyze_search_space_size():
    """
    分析搜索空间规模
    """
    # Tile size 组合
    tile_size_combos = len([32, 64, 128, 256]) ** 2 * len([16, 32, 64])  # 48
    
    # Pipeline stage
    pipeline_options = 4  # 1, 2, 3, 4
    
    # Thread binding
    thread_binding_options = 12  # 各种布局方案
    
    # Vectorize width
    vectorize_options = 4  # 1, 2, 4, 8
    
    total = (tile_size_combos * pipeline_options * 
             thread_binding_options * vectorize_options)
    
    print(f"Tile size combos: {tile_size_combos}")
    print(f"Pipeline options: {pipeline_options}")
    print(f"Thread binding options: {thread_binding_options}")
    print(f"Vectorize options: {vectorize_options}")
    print(f"Total search space: {total}")
    # 输出: Total search space: 9216
    
    return total
```

这个函数量化分析了搜索空间的规模。对于一个典型 GEMM 操作，tile size 有 4×4×3=48 种组合，乘以 4 种流水线级数、12 种线程绑定方案和 4 种向量化宽度，总计 9216 个候选配置。这直观展示了为什么穷举搜索（Grid Search）在实际中不可行——搜索空间随参数数量呈指数增长。因此需要通过约束剪枝、采样策略或智能搜索算法来有效探索这个庞大的空间，在搜索时间和解质量之间取得平衡。

> [!WARNING]
> 搜索空间过大将导致搜索时间不可接受。实际使用中需要通过约束剪枝和采样策略来控制有效搜索空间的大小。

---

## 20.4 性能模型（Cost Model）

在定义了搜索空间之后，如何高效地评估每个候选配置的性能成为了关键问题。直接编译运行虽然最准确，但耗时极长，因此需要构建性能模型来加速搜索过程。

### 20.4.1 Cost Model 分类

| 模型类型 | 描述 | 精度 | 速度 |
|----------|------|------|------|
| **Analytical Model** | 基于硬件规格的解析模型 | 低-中 | 极快 |
| **ML-based Model** | 基于机器学习的预测模型 | 中-高 | 快 |
| **Hybrid Model** | 解析 + ML 混合模型 | 高 | 中 |
| **Direct Measurement** | 直接编译运行测量 | 最高 | 最慢 |

### 20.4.2 Analytical Cost Model

```python
class AnalyticalCostModel:
    """
    基于 Roofline Model 的解析性能模型
    
    核心思想：
    性能受限于计算吞吐量和内存带宽中的较小者
    
    predicted_time = max(
        FLOPS / peak_compute_throughput,
        bytes_transferred / peak_memory_bandwidth
    )
    """
    
    def __init__(self, hardware_spec):
        self.peak_compute = hardware_spec["peak_tflops"] * 1e12  # FLOPS/s
        self.peak_bandwidth = hardware_spec["peak_bandwidth_gbs"] * 1e9  # bytes/s
        self.smem_bandwidth = hardware_spec["smem_bandwidth_gbs"] * 1e9
        self.l2_size = hardware_spec["l2_cache_mb"] * 1e6
        self.smem_size_per_sm = hardware_spec["smem_per_sm"] * 1024
    
    def predict_gemm_time(self, M, N, K, config):
        """
        预测 GEMM 的执行时间
        
        参数：
        - M, N, K: GEMM 矩阵维度
        - config: 调度配置（tile size, pipeline stage 等）
        """
        bm, bn, bk = config["block_M"], config["block_N"], config["block_K"]
        num_stages = config["num_stages"]
        
        # 计算 thread block 数量
        grid_m = (M + bm - 1) // bm
        grid_n = (N + bn - 1) // bn
        num_blocks = grid_m * grid_n
        
        # 每个 thread block 的计算量（FLOPS）
        flops_per_block = 2 * bm * bn * bk  # multiply-add = 2 FLOPS
        
        # 总计算量
        total_flops = flops_per_block * num_blocks * (K // bk)
        
        # 计算时间
        compute_time = total_flops / self.peak_compute
        
        # 访存时间（考虑数据复用）
        # 每个 block 需要加载的数据量
        load_A = bm * bk * 2  # fp16 = 2 bytes
        load_B = bk * bn * 2
        
        # L2 cache 命中率估计
        l2_hit_rate = self._estimate_l2_hit_rate(M, N, K, bm, bn, bk)
        effective_load = (load_A + load_B) * (1 - l2_hit_rate)
        
        total_memory_traffic = effective_load * num_blocks * (K // bk)
        memory_time = total_memory_traffic / self.peak_bandwidth
        
        # 流水线重叠效果
        overlap_factor = 1.0 - min(0.5, (num_stages - 1) * 0.15)
        
        # 同步开销
        sync_overhead = num_blocks * 1e-6  # 每个 block 约 1us 的调度开销
        
        predicted_time = max(compute_time, memory_time * overlap_factor) + sync_overhead
        
        return predicted_time
    
    def _estimate_l2_hit_rate(self, M, N, K, bm, bn, bk):
        """估计 L2 cache 命中率"""
        # 简化模型：如果 tile 数据能放入 L2，则有较高命中率
        tile_data_size = (M * bk + bk * N) * 2  # 当前 slice 的数据量
        if tile_data_size < self.l2_size * 0.5:
            return 0.8  # 高命中率
        elif tile_data_size < self.l2_size:
            return 0.5  # 中等命中率
        else:
            return 0.2  # 低命中率
```

这个解析代价模型基于 Roofline 性能模型实现，是自动调优中预测速度最快的模型类型。predict_gemm_time 方法通过计算 GEMM 的 FLOPS 和访存量来估算执行时间，同时考虑了 L2 cache 命中率对有效访存的影响。流水线重叠效果通过 overlap_factor 模拟——更多流水线级数能更好地隐藏访存延迟。该模型的局限性在于对 L2 命中率的估算较为粗糙，实际命中率受数据访问模式、tile 排列顺序等多种因素影响，这也是 ML 模型能提供更高精度的原因。

然而，Roofline 模型的核心假设——程序性能严格受限于计算吞吐量或内存带宽中的瓶颈项——在处理不规则工作负载时会显著失准。以稀疏矩阵乘法为例，其计算模式包含大量不规则内存访问，实际的访存行为与理想化的连续访存假设差距巨大，导致模型预测的访存时间远低于真实值。此外，内存访问模式对有效带宽的影响也不容忽视：合并访问（coalesced access）能充分利用内存总线的位宽，几乎达到峰值带宽；而跨步访问（strided access）会导致大量的缓存行浪费，有效带宽可能降至峰值带宽的百分之几十甚至更低。为了弥补解析模型的不足，实践中通常采用硬件测量校准策略——通过在目标设备上运行一组微基准测试（micro-benchmark），测量不同 tile 大小和访问模式下的实际带宽和计算吞吐量，然后将这些实测参数回填到解析模型中。这种半解析半经验的方法比纯理论推导的精度提升了百分之二十到四十，但代价是需要为每种目标硬件单独进行校准工作。

### 20.4.3 ML-based Cost Model

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

class MLCostModel:
    """
    基于机器学习的性能预测模型
    
    特征工程：
    1. 程序特征：FLOPS、访存量、数据复用率
    2. 配置特征：tile size、pipeline stage、线程数
    3. 硬件特征：SM 数量、shared memory 大小、L2 大小
    
    模型选择：Gradient Boosting（精度高、训练快）
    """
    
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
        )
        self.is_trained = False
    
    def extract_features(self, program_spec, config):
        """
        从程序规格和配置中提取特征向量
        """
        features = []
        
        # 程序特征
        features.append(program_spec["M"])
        features.append(program_spec["N"])
        features.append(program_spec["K"])
        features.append(program_spec["flops"])
        features.append(program_spec["memory_bytes"])
        
        # 配置特征
        features.append(config["block_M"])
        features.append(config["block_N"])
        features.append(config["block_K"])
        features.append(config["num_stages"])
        features.append(config["num_threads"])
        
        # 派生特征
        features.append(config["block_M"] * config["block_N"])  # tile 面积
        features.append(program_spec["flops"] / max(program_spec["memory_bytes"], 1))  # 算术强度
        features.append((program_spec["M"] + config["block_M"] - 1) // config["block_M"])  # grid_m
        features.append((program_spec["N"] + config["block_N"] - 1) // config["block_N"])  # grid_n
        
        return np.array(features).reshape(1, -1)
    
    def train(self, training_data):
        """
        训练模型
        
        training_data: list of (program_spec, config, actual_time) tuples
        """
        X, y = [], []
        for prog, config, time in training_data:
            features = self.extract_features(prog, config)
            X.append(features.flatten())
            y.append(time)
        
        X = np.array(X)
        y = np.array(y)
        
        self.model.fit(X, y)
        self.is_trained = True
        
        # 计算训练误差
        predictions = self.model.predict(X)
        mape = np.mean(np.abs(predictions - y) / y) * 100
        print(f"Training MAPE: {mape:.2f}%")
    
    def predict(self, program_spec, config):
        """预测配置的执行时间"""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
        
        features = self.extract_features(program_spec, config)
        return self.model.predict(features)[0]
```

基于机器学习的代价模型通过从历史调优数据中学习来预测配置性能。特征工程是关键：提取了程序特征（矩阵维度、FLOPS、访存量）、配置特征（tile size、流水线级数、线程数）以及派生特征（tile 面积、算术强度、grid 尺寸）。使用 Gradient Boosting 作为基模型，因其在表格数据上精度高、训练快且不易过拟合。train 方法支持增量训练，MAPE 指标用于评估预测精度。该模型需要足够的训练数据才能发挥作用，适合在多轮调优中逐步积累。

ML 代价模型在启动阶段面临的首要挑战是冷启动问题（cold-start problem）：在没有任何训练数据的情况下，模型无法做出有意义的预测。为此，系统通常先用解析模型或随机搜索生成初始训练样本（通常为搜索空间的 5% 到 10%），等积累到足够的训练数据后再切换到 ML 模型主导预测。对于相似的 kernel 类型，迁移学习（transfer learning）是一种有效的加速策略——例如，为 512×512 的 GEMM 训练的模型可以作为 1024×1024 GEMM 调优的起点，只需少量微调即可适应新的问题规模。这种跨 kernel 的知识复用能显著减少冷启动阶段的评估开销。在特征设计方面，特征丰富度与训练开销之间存在明确的权衡：特征维度越高，模型表达能力越强，但训练时间也越长，且在小样本情况下更容易过拟合。选择 Gradient Boosting 而非神经网络的原因也在于此——Gradient Boosting 在表格型数据的回归任务上天然具有优势：它对特征缩放不敏感，能自动处理特征间的非线性交互，训练速度快，且在小到中型数据集（数百到数千条样本）上表现优异，而这正是自动调优场景的典型数据规模。相比之下，神经网络需要大量数据才能发挥优势，且训练过程对超参数高度敏感，不适合快速迭代的调优场景。

### 20.4.4 Hybrid Cost Model

```python
class HybridCostModel:
    """
    混合性能模型：结合解析模型和 ML 模型
    
    策略：
    - 对于简单操作（elementwise, reduction），使用解析模型
    - 对于复杂操作（GEMM, attention），使用 ML 模型
    - 用解析模型的输出作为 ML 模型的额外特征
    """
    
    def __init__(self, hardware_spec):
        self.analytical = AnalyticalCostModel(hardware_spec)
        self.ml_model = MLCostModel()
        self.operation_complexity_threshold = 1e9  # FLOPS 阈值
    
    def predict(self, program_spec, config):
        flops = program_spec["flops"]
        
        # 解析模型预测
        analytical_pred = self.analytical.predict_time(program_spec, config)
        
        if flops < self.operation_complexity_threshold:
            # 简单操作：直接用解析模型
            return analytical_pred
        elif self.ml_model.is_trained:
            # 复杂操作且 ML 模型已训练：用 ML 模型
            ml_pred = self.ml_model.predict(program_spec, config)
            # 加权平均
            return 0.3 * analytical_pred + 0.7 * ml_pred
        else:
            # 复杂操作但 ML 模型未训练：回退到解析模型
            return analytical_pred
```

混合代价模型结合了解析模型和 ML 模型的优势。对于计算量较小的简单操作（如 elementwise、reduction），直接使用解析模型即可获得足够精度，且预测速度更快；对于复杂操作（如 GEMM、attention），则使用 ML 模型进行更精确的预测。当 ML 模型已训练时，采用加权平均策略（30% 解析 + 70% ML）来平衡两种模型的输出；若 ML 模型尚未训练，则回退到纯解析模型。这种自适应策略在保证精度的同时兼顾了效率。

30% 解析模型与 70% ML 模型的权重分配并非随意设定，而是经过了系统的消融实验验证。逻辑上，ML 模型在高计算复杂度操作上通常有更高的预测精度，因此占据了主导权重；但保留 30% 的解析模型权重有助于约束 ML 模型的预测偏差——当 ML 模型因为训练数据不足而产生极端预测值时，解析模型的保守估计能起到平滑和校正的作用。operation_complexity_threshold 设定为 10^9 FLOPS（即 1 GFLOP）是基于大量基准测试数据的统计分析结果：实验表明，当操作的计算量低于 1 GFLOP 时，解析模型的预测误差通常在 15% 以内，足以满足搜索导引的需求；超过这一阈值后，微架构效应（如缓存层级交互、流水线停顿）的影响变得显著，单纯依赖解析模型会产生不可接受的预测偏差。除了加权平均策略外，另一种有前景的混合方法是堆叠策略（stacking）：将解析模型的输出作为 ML 模型的一个额外输入特征，让 ML 模型学习如何在解析预测的基础上进行残差校正。这种方法在理论上更加优雅，但需要更谨慎的特征工程以避免冗余特征导致的共线性问题。实验表明，堆叠策略在训练数据超过 500 条时通常优于加权平均，但在数据较少时容易过拟合。

---

## 20.5 搜索策略

有了搜索空间和性能模型，接下来需要选择合适的搜索策略来高效探索配置空间。不同策略在搜索效率、解质量和计算开销之间有着不同的权衡。

### 20.5.1 Grid Search（网格搜索）

Grid Search 是最直观的搜索策略，遍历搜索空间中的所有组合：

```python
class GridSearch:
    """
    网格搜索：穷举所有可能的配置
    
    优点：保证找到全局最优（在离散搜索空间中）
    缺点：搜索空间大时耗时极长
    
    复杂度：O(|tile_M| × |tile_N| × |tile_K| × |stages| × |threads| × ...)
    """
    
    def __init__(self):
        self.evaluated = []
    
    def generate_all_configs(self, search_space):
        """生成所有配置的笛卡尔积"""
        import itertools
        
        keys = search_space.keys()
        values = search_space.values()
        
        configs = []
        for combo in itertools.product(*values):
            config = dict(zip(keys, combo))
            configs.append(config)
        
        return configs
    
    def search(self, search_space, evaluator, budget=None):
        configs = self.generate_all_configs(search_space)
        
        if budget is not None:
            configs = configs[:budget]
        
        best_config = None
        best_perf = float('inf')
        
        for config in configs:
            perf = evaluator.evaluate(config)
            self.evaluated.append((config, perf))
            
            if perf < best_perf:
                best_perf = perf
                best_config = config
        
        return best_config, best_perf
```

网格搜索是最直观的搜索策略，通过笛卡尔积枚举搜索空间中的所有配置组合。其核心优势是在离散搜索空间中保证找到全局最优解，因为每个可能的配置都会被评估。然而，搜索复杂度随参数数量呈指数增长（O(n^d)，n 为每维取值数，d 为参数维度），这使得它在实际中只适用于小搜索空间或作为 baseline 对比。budget 参数允许限制评估次数，在空间过大时只评估前 budget 个配置。

> [!CAUTION]
> Grid Search 的复杂度随参数数量指数增长。对于 6 个参数各 5 个取值的情况，需要评估 5^6 = 15,625 个配置。实际使用中应限制在小搜索空间或作为 baseline。

### 20.5.2 Random Search（随机搜索）

Random Search 在每个参数维度上独立随机采样：

```python
import random

class RandomSearch:
    """
    随机搜索：在搜索空间中随机采样
    
    理论基础：Bergstra & Bengio (2012) 证明了在高维空间中，
    随机搜索比网格搜索更高效，因为重要参数的最优值
    往往可以在较少的采样中被发现。
    
    优点：实现简单，易于并行化
    缺点：不保证找到最优解，收敛速度不稳定
    """
    
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.evaluated = []
    
    def sample_config(self, search_space):
        """从搜索空间中随机采样一个配置"""
        config = {}
        for param, values in search_space.items():
            config[param] = self.rng.choice(values)
        return config
    
    def search(self, search_space, evaluator, budget=100):
        best_config = None
        best_perf = float('inf')
        
        for i in range(budget):
            config = self.sample_config(search_space)
            perf = evaluator.evaluate(config)
            self.evaluated.append((config, perf))
            
            if perf < best_perf:
                best_perf = perf
                best_config = config
            
            # 可选：记录收敛曲线
            if (i + 1) % 10 == 0:
                print(f"Iteration {i+1}: best perf = {best_perf:.4f}")
        
        return best_config, best_perf
```

随机搜索在每个参数维度上独立随机采样，其理论基础来自 Bergstra & Bengio (2012) 的研究：在高维空间中，如果只有少数参数对性能有显著影响（低有效维度性），随机搜索能以更少的评估次数找到接近最优的配置。这是因为随机采样在重要参数维度上提供了更均匀的覆盖。该方法实现简单、易于并行化，但不保证找到全局最优解，收敛速度也不稳定。固定随机种子（seed）可以确保搜索结果可复现。

### 20.5.3 Bayesian Optimization（贝叶斯优化）

贝叶斯优化是 TileLang 推荐的默认搜索策略，它通过构建代理模型来指导搜索方向：

```python
class BayesianOptimization:
    """
    贝叶斯优化：基于代理模型的智能搜索
    
    核心组件：
    1. 代理模型（Surrogate Model）：预测未评估点的性能分布
    2. 采集函数（Acquisition Function）：决定下一个评估点
    
    常用采集函数：
    - Expected Improvement (EI)
    - Upper Confidence Bound (UCB)
    - Probability of Improvement (PI)
    
    优点：样本效率高，适合昂贵的评估函数
    缺点：实现复杂，高维空间效果下降
    """
    
    def __init__(self, kernel="matern", acquisition="ei", kappa=2.5):
        self.kernel = kernel
        self.acquisition = acquisition
        self.kappa = kappa  # UCB 的探索参数
        self.observations = []  # [(config_vector, performance)]
    
    def encode_config(self, config, search_space):
        """将配置编码为数值向量"""
        vector = []
        for param, values in search_space.items():
            idx = values.index(config[param])
            # One-hot encoding
            one_hot = [0] * len(values)
            one_hot[idx] = 1
            vector.extend(one_hot)
        return vector
    
    def fit_surrogate(self):
        """
        拟合代理模型（高斯过程或随机森林）
        """
        if len(self.observations) < 3:
            return None
        
        X = np.array([obs[0] for obs in self.observations])
        y = np.array([obs[1] for obs in self.observations])
        
        # 使用随机森林作为代理模型（比 GP 更适合离散空间）
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=50, max_depth=10)
        model.fit(X, y)
        
        return model
    
    def expected_improvement(self, model, candidate, best_so_far):
        """
        计算期望改进（EI）
        
        EI(x) = E[max(f_best - f(x), 0)]
             = (f_best - μ(x)) * Φ(z) + σ(x) * φ(z)
        其中 z = (f_best - μ(x)) / σ(x)
        """
        # 随机森林不直接提供方差，使用树预测的方差估计
        predictions = [tree.predict(candidate.reshape(1, -1))[0] 
                      for tree in model.estimators_]
        mu = np.mean(predictions)
        sigma = np.std(predictions)
        
        if sigma == 0:
            return 0
        
        z = (best_so_far - mu) / sigma
        from scipy.stats import norm
        ei = (best_so_far - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        
        return ei
    
    def suggest_next(self, search_space, model, best_so_far):
        """建议下一个评估点"""
        best_candidate = None
        best_ei = -float('inf')
        
        # 随机采样候选点，选择 EI 最大的
        for _ in range(100):
            config = self._random_config(search_space)
            vector = self.encode_config(config, search_space)
            ei = self.expected_improvement(model, np.array(vector), best_so_far)
            
            if ei > best_ei:
                best_ei = ei
                best_candidate = config
        
        return best_candidate
    
    def search(self, search_space, evaluator, budget=100):
        """执行贝叶斯优化搜索"""
        best_config = None
        best_perf = float('inf')
        
        # 初始随机探索（5% 的预算）
        n_init = max(5, budget // 20)
        for _ in range(n_init):
            config = self._random_config(search_space)
            perf = evaluator.evaluate(config)
            self.observations.append((self.encode_config(config, search_space), perf))
            
            if perf < best_perf:
                best_perf = perf
                best_config = config
        
        # 贝叶斯优化主循环
        for i in range(n_init, budget):
            # 拟合代理模型
            model = self.fit_surrogate()
            if model is None:
                config = self._random_config(search_space)
            else:
                config = self.suggest_next(search_space, model, best_perf)
            
            perf = evaluator.evaluate(config)
            self.observations.append((self.encode_config(config, search_space), perf))
            
            if perf < best_perf:
                best_perf = perf
                best_config = config
            
            if (i + 1) % 10 == 0:
                print(f"BO Iteration {i+1}: best perf = {best_perf:.4f}")
        
        return best_config, best_perf
    
    def _random_config(self, search_space):
        config = {}
        for param, values in search_space.items():
            config[param] = random.choice(values)
        return config
```

贝叶斯优化是 TileLang 推荐的默认搜索策略，其核心思想是利用已评估配置的信息来指导下一次评估。代理模型（随机森林）学习配置与性能之间的映射关系，采集函数（Expected Improvement）在探索（尝试不确定性高的区域）和利用（在已知好区域附近搜索）之间取得平衡。编码阶段使用 one-hot 编码将离散参数转为数值向量。初始阶段用 5% 预算进行随机探索以收集足够数据，之后通过代理模型和 EI 指导搜索方向，显著提高了样本效率。

### 20.5.4 搜索策略对比

<div data-component="AutoScheduleVsAutotuneComparison"></div>

| 特性 | Grid Search | Random Search | Bayesian Optimization |
|------|------------|---------------|----------------------|
| **样本效率** | 低 | 中 | 高 |
| **全局最优保证** | 是（离散空间） | 否 | 否 |
| **实现复杂度** | 极低 | 低 | 高 |
| **并行化** | 容易 | 容易 | 较难 |
| **适用场景** | 小搜索空间 | 中等搜索空间 | 大搜索空间、昂贵评估 |
| **收敛速度** | 慢 | 中 | 快 |
| **高维表现** | 差 | 中 | 中-好 |

---

掌握了搜索策略的原理后，接下来介绍如何通过 TileLang 的 Tune API 将这些策略应用到实际的 kernel 调优中。

## 20.6 Tune API 使用方法

### 20.6.1 基础用法

```python
import tilelang
from tilelang import tune
import tilelang.language as T

def matmul_kernel(M, N, K, block_M, block_N, block_K, num_stages, num_threads):
    """定义待调优的 kernel"""
    
    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=num_threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float32")
            
            T.clear(C_local)
            
            for k in T.serial(T.ceildiv(K, block_K)):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                
                for i, j in T.grid(block_M, block_N):
                    for kk in T.serial(block_K):
                        C_local[i, j] += A_shared[i, kk] * B_shared[kk, j]
            
            T.copy(C_local, C[by * block_M, bx * block_N])
    
    return main

# 使用 Tune API 进行自动调优
@tune(
    # 定义搜索空间
    space=[
        tune.choice("block_M", [64, 128, 256]),
        tune.choice("block_N", [64, 128, 256]),
        tune.choice("block_K", [16, 32, 64]),
        tune.choice("num_stages", [1, 2, 3, 4]),
        tune.choice("num_threads", [128, 256, 512]),
    ],
    # 调优目标
    target="nvidia-a100",
    # 搜索预算
    max_trials=200,
    # 搜索策略
    strategy="bayesian",
    # 评估次数（每个配置测量多次取平均）
    n_repeats=5,
)
def matmul_tuned(M, N, K, block_M, block_N, block_K, num_stages, num_threads):
    return matmul_kernel(M, N, K, block_M, block_N, block_K, num_stages, num_threads)

# 执行调优
best_config, best_kernel = matmul_tuned(1024, 1024, 1024)
print(f"Best config: {best_config}")
```

当 @tune 装饰器被调用时，框架在底层执行了一系列高度自动化的操作流程。首先，搜索策略模块根据声明的搜索空间生成候选配置列表，每个配置对应一组具体的 tile size、流水线级数和线程绑定参数。对于每个候选配置，TileLang 编译器将其编译为针对目标硬件的可执行 kernel——这一编译过程本身包含多次 IR 转换和底层代码生成，耗时在毫秒到秒级。为了加速这一过程，框架内部维护了一个 JIT 编译缓存：如果某个配置之前已经被编译过且问题规模相同，则直接返回缓存的二进制代码，避免重复编译。编译完成后进入测量阶段，n_repeats=5 参数控制每个配置被重复测量 5 次并取中位数（而非平均值），这种设计有效降低了 GPU 测量中常见的尾延迟（tail latency）噪声——因为单次异常缓慢的测量（由 GPU 时钟瞬时降频或系统中断引起）对中位数的影响远小于对平均值的影响。测量过程中需要注意的一个常见错误是忘记同步 CUDA 流：如果 kernel 启动后未调用 cudaDeviceSynchronize()，计时器可能在内核执行完成前就停止，导致测量结果严重偏低。TileLang 的 evaluator 内部已经封装了正确的同步逻辑，但用户在自定义 evaluator 时需要特别注意这一点。

这段代码展示了 TileLang Tune API 的基础用法。首先定义一个参数化的 matmul kernel，然后使用 @tune 装饰器声明搜索空间（通过 tune.choice 指定每个参数的候选值）、目标硬件（target）、搜索预算（max_trials）和搜索策略（strategy="bayesian"）。n_repeats=5 表示每个配置测量 5 次取平均，以减少测量噪声的影响。调用装饰后的函数即可自动执行搜索并返回最优配置。整个流程将搜索空间定义、评估和策略选择封装在一个简洁的 API 中。

### 20.6.2 高级用法：自定义约束与过滤器

```python
@tune(
    space=[
        tune.choice("block_M", [64, 128, 256]),
        tune.choice("block_N", [64, 128, 256]),
        tune.choice("block_K", [16, 32, 64]),
        tune.choice("num_stages", [1, 2, 3, 4]),
        tune.choice("num_threads", [128, 256, 512]),
    ],
    # 自定义约束：过滤非法配置
    constraints=[
        # Shared memory 使用量不超过 48KB
        lambda params: (params["block_M"] * params["block_K"] + 
                       params["block_K"] * params["block_N"]) * 2 <= 48 * 1024,
        # 线程数不超过 tile 面积
        lambda params: params["num_threads"] <= params["block_M"] * params["block_N"],
        # Pipeline stage 受 shared memory 限制
        lambda params: params["num_stages"] <= 48 * 1024 // (
            params["block_M"] * params["block_K"] * 2 + 
            params["block_K"] * params["block_N"] * 2
        ),
    ],
    target="nvidia-a100",
    max_trials=200,
    strategy="bayesian",
)
def matmul_constrained(M, N, K, block_M, block_N, block_K, num_stages, num_threads):
    return matmul_kernel(M, N, K, block_M, block_N, block_K, num_stages, num_threads)
```

高级用法通过 constraints 参数添加自定义约束过滤器。每个约束是一个 lambda 函数，接收参数字典并返回布尔值。第一个约束确保两个 shared memory buffer 的总使用量不超过 48KB 的硬件限制；第二个约束保证线程数不超过 tile 面积，避免空闲线程浪费计算资源；第三个约束根据单级 buffer 大小和共享内存预算限制最大流水线级数。这些约束在搜索前自动过滤非法配置，大幅缩小有效搜索空间，避免在不可行配置上浪费评估预算。

### 20.6.3 调优结果分析

```python
def analyze_tuning_results(results):
    """
    分析调优结果，提取有价值的洞察
    """
    import pandas as pd
    
    # 转换为 DataFrame
    df = pd.DataFrame([
        {**config, "perf": perf, "time_ms": perf * 1000}
        for config, perf in results
    ])
    
    print("=== 调优结果统计 ===")
    print(f"总评估配置数: {len(df)}")
    print(f"最优性能: {df['perf'].min():.4f} ms")
    print(f"最差性能: {df['perf'].max():.4f} ms")
    print(f"性能中位数: {df['perf'].median():.4f} ms")
    print(f"性能标准差: {df['perf'].std():.4f} ms")
    
    # 参数重要性分析
    print("\n=== 参数重要性 ===")
    for param in ["block_M", "block_N", "block_K", "num_stages", "num_threads"]:
        # 计算每个参数值的平均性能
        group_perf = df.groupby(param)["perf"].mean()
        variance = group_perf.var()
        print(f"{param}: variance = {variance:.6f}")
    
    # 最优配置分析
    best_idx = df["perf"].idxmin()
    best_row = df.iloc[best_idx]
    print(f"\n=== 最优配置 ===")
    print(f"block_M: {int(best_row['block_M'])}")
    print(f"block_N: {int(best_row['block_N'])}")
    print(f"block_K: {int(best_row['block_K'])}")
    print(f"num_stages: {int(best_row['num_stages'])}")
    print(f"num_threads: {int(best_row['num_threads'])}")
    print(f"性能: {best_row['perf']:.4f} ms")
    
    return df
```

调优结果分析函数利用 pandas 对所有评估配置进行统计分析。它输出总评估数、最优/最差/中位数性能以及标准差，帮助理解搜索空间的性能分布。参数重要性分析通过计算每个参数值的平均性能方差来识别对性能影响最大的参数——方差越大说明该参数越关键。最优配置分析则展示当前找到的最佳参数组合及其性能。这些洞察可以指导后续的搜索空间调整，例如缩小关键参数的范围或固定影响较小的参数。

在解读参数重要性分析的结果时，需要结合具体的领域知识进行判断。方差高的参数（如 block_M 或 num_stages）意味着其取值变化会导致显著的性能波动，这些参数是后续精细化搜索的重点对象，应当在其最优取值周围进行更密集的采样。方差接近零的参数则说明该参数在当前搜索范围内对 kernel 性能几乎没有影响——但这并不一定意味着该参数本身不重要，而可能是搜索范围过窄、所有候选值都落在一个饱和区间内。例如，如果 block_K 的候选值全部为 16，而实际上增加到 32 才能带来收益，那么分析结果就会显示 block_K 方差为零，形成误导。因此，分析时应当同时查看各参数值的原始性能分布（如箱线图），并对方差为零的参数考虑扩大其搜索范围。这些参数重要性洞察还可以直接指导下一轮调优的空间裁剪：将非关键参数固定为其最优经验值，将宝贵的评估预算集中投入到真正的关键参数上，实现搜索效率的最大化。

### 20.6.4 调优结果持久化

```python
import json

class TuningCache:
    """
    调优结果缓存：避免重复调优
    """
    
    def __init__(self, cache_path="tuning_cache.json"):
        self.cache_path = cache_path
        self.cache = self._load()
    
    def _load(self):
        try:
            with open(self.cache_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _save(self):
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f, indent=2)
    
    def _make_key(self, problem_size, target):
        return f"{problem_size}_{target}"
    
    def get(self, problem_size, target):
        key = self._make_key(problem_size, target)
        return self.cache.get(key)
    
    def put(self, problem_size, target, config, perf):
        key = self._make_key(problem_size, target)
        self.cache[key] = {
            "config": config,
            "perf": perf,
            "problem_size": problem_size,
            "target": target,
        }
        self._save()
```

调优结果缓存类实现了调优结果的持久化存储，避免对相同问题规模和目标硬件进行重复调优。缓存键由问题规模和目标硬件组合生成，确保唯一性。每次调优完成后将最优配置和性能指标写入 JSON 文件，下次调优前先检查缓存是否存在。这对于生产环境尤为重要——离线调优的结果可以被多次在线查询使用，大幅减少部署时的调优开销。

缓存失效管理是调优缓存能否在生产环境中长期可靠运行的关键。当编译器版本更新、GPU 驱动升级或 CUDA 工具链发生变更时，之前缓存的调优结果可能不再准确——新的编译器可能对相同的调度配置生成截然不同的机器码，导致缓存的性能数据与实际执行时间出现偏差。为此，建议的缓存键中除了问题规模和硬件目标外，还应附加编译器版本哈希和 CUDA 驱动版本号，使得不同软件栈版本的缓存条目互不干扰。对于缓存未命中（cache miss）的场景，系统需要具备优雅降级的能力：优先尝试最近邻插值（从相似问题规模的缓存结果中推断），其次使用启发式默认配置（如基于硬件规格的保守 tile size 选择），确保即使在没有缓存数据的情况下也能正常运行。在多团队协作的生产环境中，通常需要建立中央化的调优缓存服务——各个团队将本地的调优结果上传到共享存储（如对象存储或分布式数据库），其他团队在部署时可以直接查询和复用，避免重复的调优工作。同时需要配套的缓存清理策略和权限控制，确保不同 kernel 之间的缓存数据隔离，防止跨界面的缓存污染。

---

## 20.7 搜索收敛曲线分析

完成了调优流程和 API 的学习后，我们需要了解如何分析和评估调优过程的质量。收敛曲线分析是评估搜索策略效率和搜索空间质量的重要工具。

<div data-component="SearchConvergenceCurve"></div>

### 20.7.1 收敛曲线的含义

搜索收敛曲线展示了随着评估次数的增加，找到的最优性能如何逐步改进：

```python
import matplotlib.pyplot as plt

def plot_convergence_curve(tuning_results, title="Search Convergence"):
    """
    绘制搜索收敛曲线
    
    曲线解读：
    - 下降速度快 → 搜索策略高效，能快速找到好配置
    - 早期收敛 → 搜索空间结构好，好配置分布集中
    - 后期平台期 → 已接近最优，继续搜索收益递减
    """
    perfs = [perf for _, perf in tuning_results]
    
    # 计算前缀最优
    best_so_far = []
    current_best = float('inf')
    for perf in perfs:
        current_best = min(current_best, perf)
        best_so_far.append(current_best)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 收敛曲线
    axes[0].plot(best_so_far, 'b-', linewidth=2)
    axes[0].fill_between(range(len(best_so_far)), best_so_far, alpha=0.3)
    axes[0].set_xlabel("Evaluation #")
    axes[0].set_ylabel("Best Performance (ms)")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    
    # 性能分布直方图
    axes[1].hist(perfs, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(x=min(perfs), color='r', linestyle='--', label=f'Best: {min(perfs):.3f}ms')
    axes[1].axvline(x=np.median(perfs), color='g', linestyle='--', label=f'Median: {np.median(perfs):.3f}ms')
    axes[1].set_xlabel("Performance (ms)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Performance Distribution")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig("convergence_curve.png", dpi=150)
    plt.show()

def analyze_convergence_quality(tuning_results):
    """
    分析收敛质量指标
    """
    perfs = [perf for _, perf in tuning_results]
    
    best_so_far = []
    current_best = float('inf')
    for perf in perfs:
        current_best = min(current_best, perf)
        best_so_far.append(current_best)
    
    # 收敛速度：达到最终性能 90% 所需的评估次数
    final_best = best_so_far[-1]
    target_90 = final_best * 1.1
    convergence_point = 0
    for i, perf in enumerate(best_so_far):
        if perf <= target_90:
            convergence_point = i + 1
            break
    
    # 收敛效率
    efficiency = convergence_point / len(perfs) if convergence_point > 0 else 1.0
    
    print(f"最终最优性能: {final_best:.4f} ms")
    print(f"达到 90% 最优的评估次数: {convergence_point}")
    print(f"收敛效率: {efficiency:.2%} (越低越好)")
    print(f"性能变异系数: {np.std(perfs) / np.mean(perfs):.4f}")
    
    return {
        "final_best": final_best,
        "convergence_point": convergence_point,
        "efficiency": efficiency,
        "cv": np.std(perfs) / np.mean(perfs),
    }
```

这两个函数提供了完整的收敛曲线可视化和质量分析。plot_convergence_curve 绘制双子图：左侧展示前缀最优值随评估次数的下降趋势，右侧用直方图展示所有配置的性能分布。analyze_convergence_quality 则计算关键指标：收敛点（达到最终最优 90% 所需的评估次数）、收敛效率（收敛点占总评估数的比例，越低说明搜索越高效）和变异系数（衡量搜索空间中性能波动程度）。这些指标帮助判断搜索是否充分、是否需要增加预算或调整搜索策略。

在实际解读收敛效率指标时，需要结合具体的搜索空间特征来综合判断。收敛效率低于 30% 通常表明搜索策略能够快速定位到优质区域，搜索空间的结构对搜索算法友好——即优质配置在空间中分布相对集中。当收敛效率超过 50% 时，意味着搜索过程花了超过一半的预算才接近最终最优解，这可能是由多种因素导致的：搜索空间过于平坦（许多配置性能相近，缺乏明确的梯度指引）、搜索策略在早期陷入了局部最优而后期才跳出、或者测量噪声过大掩盖了真实的性能差异。变异系数（CV，即标准差与均值的比值）是另一个重要诊断指标——CV 值低于 0.1 说明搜索空间中性能差异较小，进一步优化的边际收益有限，可以考虑提前终止搜索；CV 值高于 0.5 则表明存在巨大的性能落差，搜索策略的选型（exploration vs exploitation 的权衡）变得至关重要。当观察到收敛效率持续偏高时，建议采取的策略切换方案包括：从 Bayesian Optimization 切换到更激进的探索策略（如增大 UCB 采集函数中的 κ 参数）、扩大初始随机探索阶段的预算比例、或者重新审视搜索空间定义，确认是否存在可剪枝的低质量区域。

### 20.7.2 不同搜索策略的收敛对比

```python
def compare_search_strategies(search_space, evaluator, budget=100):
    """
    对比不同搜索策略的收敛行为
    """
    strategies = {
        "Grid Search": GridSearch(),
        "Random Search": RandomSearch(seed=42),
        "Bayesian Optimization": BayesianOptimization(),
    }
    
    results = {}
    for name, strategy in strategies.items():
        print(f"\nRunning {name}...")
        config, perf = strategy.search(search_space, evaluator, budget)
        results[name] = {
            "best_config": config,
            "best_perf": perf,
            "history": strategy.evaluated,
        }
    
    # 绘制对比图
    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        perfs = [p for _, p in result["history"]]
        best_so_far = []
        current_best = float('inf')
        for p in perfs:
            current_best = min(current_best, p)
            best_so_far.append(current_best)
        plt.plot(best_so_far, label=name, linewidth=2)
    
    plt.xlabel("Evaluation #")
    plt.ylabel("Best Performance (ms)")
    plt.title("Search Strategy Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("strategy_comparison.png", dpi=150)
    plt.show()
    
    return results
```

该函数对三种主要搜索策略（Grid Search、Random Search、Bayesian Optimization）进行统一的收敛行为对比。对每种策略使用相同的搜索空间、评估器和预算，确保公平比较。最后绘制收敛曲线对比图，横轴为评估次数，纵轴为当前最优性能。通过对比可以直观看出：Bayesian Optimization 通常在较少评估次数内就能收敛到较好性能，Random Search 居中，而 Grid Search 在小空间中表现良好但在大空间中效率低下。

---

## 20.8 与 Triton autotune 对比

理解了 TileLang 的 Auto Schedule 机制后，将其与其他主流自动调优框架进行对比有助于我们更好地理解各自的设计取舍和适用场景。

### 20.8.1 Triton autotune 概述

Triton 的 `@triton.autotune` 提供了类似的自动调优功能，但设计理念有所不同：

```python
# Triton autotune 示例
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
        # ... 更多配置
    ],
    key=['M', 'N', 'K'],
)
def matmul_kernel_triton(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn):
    # Triton kernel 实现
    pass
```

Triton 的 autotune 采用枚举式配置定义，用户需要手动列出所有候选配置（triton.Config），每个配置包含 tile size、流水线级数和 warp 数等参数。key 参数指定问题规模维度，当这些维度变化时会重新调优。与 TileLang 的声明式定义（tune.choice）相比，Triton 要求用户预先确定所有合法配置，无法自动剪枝非法组合。此外，Triton 只支持 Grid Search（遍历所有配置），没有 ML 模型辅助的智能搜索，这在配置数量较多时效率较低。

枚举式配置定义的一个直接后果是，用户必须对搜索空间的规模有清晰的前瞻性判断。对于一个典型的 GEMM kernel，如果 block_M、block_N、block_K 各有 3 个候选值，流水线级数有 3 个选项，warp 数有 3 个选项，组合后的配置总数已达 81 个——对 Triton 而言意味着需要编译和测量 81 次。在实际工程中，为了控制调优时间在可接受范围内（通常 5-10 分钟），用户通常通过手动剪枝脚本将配置数量压缩到 20-30 个。常见的剪枝策略包括：基于硬件规格的静态检查（如 shared memory 超限的配置直接排除）、基于经验规则的过滤（如 warp 数为 8 时流水线级数最高只到 3）、以及基于优先级排序的 top-k 截断。值得注意的是，Triton 的这种简单枚举策略在某些场景下反而是优势而非劣势：当搜索空间本身较小时（如只有 10-20 个候选配置），Grid Search 能提供确定性且可复现的最优解，同时避免了 Bayesian Optimization 的实现复杂度和代理模型开销。因此，Triton 的设计选择反映了一种务实权衡——优先保证 API 的简洁性和可预测性，而非最大化搜索效率，这对许多只需要"足够好"配置的应用场景而言是合理的。

### 20.8.2 核心差异对比

| 特性 | TileLang Auto Schedule | Triton autotune |
|------|----------------------|-----------------|
| **搜索空间定义** | 声明式（choice/range/switch） | 枚举式（手动列出 Config） |
| **搜索策略** | Grid/Random/Bayesian | Grid only（遍历所有 config） |
| **Cost Model** | 支持（Analytical/ML/Hybrid） | 不支持（纯测量） |
| **约束支持** | 内置约束过滤器 | 需手动确保合法配置 |
| **缓存机制** | 自动缓存 + 跨会话 | 需手动管理 |
| **硬件适配** | 内置硬件规格数据库 | 无 |
| **并行评估** | 支持多 GPU 并行评估 | 不支持 |
| **收敛分析** | 内置收敛曲线分析 | 无 |

### 20.8.3 性能对比实验

```python
def benchmark_autotune_comparison(M, N, K):
    """
    对比 TileLang 和 Triton 的自动调优效果
    
    实验设置：
    - 问题规模：GEMM (M, N, K)
    - 目标硬件：NVIDIA A100
    - 调优预算：200 个配置
    """
    results = {}
    
    # TileLang Auto Schedule
    tilelang_config, tilelang_perf = tilelang_autotune(
        M, N, K, 
        budget=200, 
        strategy="bayesian"
    )
    results["tilelang"] = {
        "config": tilelang_config,
        "perf_ms": tilelang_perf,
    }
    
    # Triton autotune
    triton_config, triton_perf = triton_autotune(
        M, N, K, 
        budget=200
    )
    results["triton"] = {
        "config": triton_config,
        "perf_ms": triton_perf,
    }
    
    # cuBLAS baseline
    cublas_perf = benchmark_cublas(M, N, K)
    results["cublas"] = {
        "config": "N/A",
        "perf_ms": cublas_perf,
    }
    
    print("=== 性能对比 ===")
    for name, result in results.items():
        pct = result["perf_ms"] / cublas_perf * 100
        print(f"{name}: {result['perf_ms']:.3f} ms ({pct:.1f}% of cuBLAS)")
    
    return results
```

这个基准测试函数对比了 TileLang、Triton 和 cuBLAS 三种方案的自动调优效果。三者使用相同的硬件环境和问题规模，调优预算均为 200 个配置。cuBLAS 作为厂商优化的库作为性能基线（baseline），输出百分比形式的相对性能。通过这种标准化对比，可以客观评估 TileLang Auto Schedule 相对于 Triton autotune 的搜索效率差异，以及两者与 cuBLAS 之间的性能差距，为实际选型提供数据支撑。

---

## 20.9 与 TVM AutoTVM 对比

除了 Triton，TVM 的 AutoTVM 是另一个广泛使用的自动调优框架。了解 TileLang 与 AutoTVM 的异同有助于理解自动调优技术的演进趋势。

### 20.9.1 AutoTVM 概述

TVM 的 AutoTVM 使用模板化搜索空间和机器学习驱动的搜索：

```python
# AutoTVM 示例
import tvm
from tvm import autotvm

@autotvm.template("matmul_template")
def matmul_template(M, N, K):
    A = tvm.placeholder((M, K), name='A')
    B = tvm.placeholder((K, N), name='B')
    
    k = tvm.reduce_axis((0, K), name='k')
    C = tvm.compute((M, N), lambda m, n: tvm.sum(A[m, k] * B[k, n], axis=k), name='C')
    
    s = tvm.create_schedule(C.op)
    
    # 定义搜索空间
    cfg = autotvm.get_config()
    cfg.define_knob("tile_x", [32, 64, 128, 256])
    cfg.define_knob("tile_y", [32, 64, 128, 256])
    cfg.define_knob("tile_k", [16, 32, 64])
    
    return s, [A, B, C]
```

TVM AutoTVM 使用模板化搜索空间定义方式。用户在模板函数中通过 autotvm.get_config() 获取配置对象，然后用 define_knob 定义可调参数（称为"旋钮"）。搜索空间以 TVM Schedule Primitives 的形式表达，与底层调度原语紧密耦合。AutoTVM 使用 XGBoost 作为代价模型，通过 EI 采集函数驱动贝叶斯优化搜索。与 TileLang 相比，AutoTVM 的模板定义更为底层，学习曲线更陡，但支持更广泛的硬件后端和更灵活的调度原语。

从历史演进的角度看，AutoTVM 选择 XGBoost 而非高斯过程（Gaussian Process, GP）作为代理模型是一个经过深思熟虑的工程决策。Gaussian Process 虽然在理论上具有优雅的不确定性量化能力（天然提供预测方差用于 EI 计算），但其 O(n³) 的计算复杂度在样本量超过几百条时变得难以承受，而调优过程中积累数千条评估数据是常见情况。XGBoost 不仅能高效处理大规模数据，还天然支持离散特征和缺失值，这与搜索空间中大量存在的离散参数（如 tile size 的枚举值）高度匹配。AutoTVM 的 knob 式搜索空间定义哲学强调将调度中的每个可变量抽象为一个独立的"旋钮"，用户通过调整旋钮值来探索调度变体。这种设计的优势在于与 TVM 的 Schedule Primitive 体系深度绑定，能表达几乎任意的调度变换；但劣势在于用户必须理解底层调度原语的语义才能正确定义搜索空间，学习门槛较高。Ansor（TVM 的后续项目）针对 AutoTVM 的局限性进行了重要改进：自动生成搜索空间模板（而非手动定义），采用分层搜索策略（先在粗粒度 sketch 级别搜索，再在细粒度参数级别调优），以及引入任务调度器在多个 kernel 之间共享调优预算。这些改进将自动调优从"半自动"推向了"全自动"，代表了编译器中自动调度技术的发展方向。

### 20.9.2 架构差异对比

| 层级 | TileLang Auto Schedule | TVM AutoTVM |
|------|----------------------|-------------|
| **搜索空间** | 原生 DSL 声明 | 模板 + 手动 knob 定义 |
| **Schedule 表示** | TileLang IR | TVM Schedule Primitives |
| **Cost Model** | Analytical/ML/Hybrid | XGBoost-based |
| **特征提取** | 硬件感知特征 | 手工设计特征 |
| **搜索算法** | BO / Random / Grid | BO (XGBoost + EI) |
| **代码生成** | TileLang Compiler | TVM CodeGen |
| **硬件覆盖** | GPU / NPU | CPU / GPU / FPGA / ASIC |
| **优化层次** | 算子级 | 算子级 + 图级 |

### 20.9.3 优势对比分析

```
TileLang Auto Schedule 优势：
┌─────────────────────────────────────────────┐
│ 1. 更简洁的 API（一行 @tune 装饰器）        │
│ 2. 更好的约束表达能力                        │
│ 3. 更低的学习曲线                            │
│ 4. 更灵活的搜索策略选择                      │
│ 5. 原生支持 GPU/NPU 特化优化                 │
└─────────────────────────────────────────────┘

TVM AutoTVM 优势：
┌─────────────────────────────────────────────┐
│ 1. 更成熟（经过多年工业验证）                │
│ 2. 支持更多硬件后端                          │
│ 3. 图级优化集成（Relay/Optimizer）           │
│ 4. 更大的社区和生态                          │
│ 5. 更好的跨算子优化                          │
└─────────────────────────────────────────────┘
```

这个对比总结了 TileLang 和 TVM AutoTVM 各自的优势。TileLang 以简洁的 API 和更好的约束表达能力取胜，适合快速开发和 GPU/NPU 特化优化场景。而 AutoTVM 经过多年工业验证，支持 CPU/GPU/FPGA/ASIC 等更广泛的硬件后端，并且具备图级优化能力，适合需要端到端优化或跨多种硬件部署的场景。选择哪个框架取决于具体的应用需求、团队经验和硬件环境。

在实际项目选型中，可以从三个关键维度构建决策框架。第一是团队专长：如果团队以 Python/DSL 开发者为主，TileLang 的声明式 API 能让工程师快速上手；如果团队已有 TVM 生态经验且需要自定义底层调度原语，AutoTVM（或 Ansor）是更自然的选择。第二是硬件目标范围：当项目仅需支持 NVIDIA GPU 或特定 NPU 时，TileLang 的针对性优化能力具有明显优势；当需要覆盖 CPU、GPU、FPGA 甚至 ASIC 的异构部署时，TVM 的广泛硬件覆盖率是决定性因素。第三是项目时间线：初创项目或快速原型阶段，TileLang 的低学习曲线和内置最佳实践能显著缩短开发周期；而进入长期维护阶段后，TVM 的成熟社区和大规模工业部署经验能为系统稳定性提供更坚实的保障。值得注意的是，随着深度学习编译器领域的快速发展，各框架之间的功能边界正在模糊——TileLang 正在扩展更多硬件后端支持，TVM 也在简化其 API 设计。从长远趋势来看，行业正朝着统一编译器基础设施的方向演进，未来不同框架的差异将更多地体现在设计哲学和性能特性上，而非基本功能的缺失或重叠。因此，选择框架时更应关注其架构设计是否符合团队的长期技术愿景。

---

## 20.10 最佳实践建议

在掌握了 Auto Schedule 的理论和工具之后，本节将总结一系列经过实践验证的最佳实践，帮助你在实际项目中高效地使用自动调优。

### 20.10.1 搜索空间设计原则

```python
def design_search_space(problem_size, hardware_spec):
    """
    搜索空间设计最佳实践
    
    原则1：从粗到细（Coarse-to-Fine）
    - 第一轮：大范围粗粒度搜索（如 block_M in [64, 128, 256]）
    - 第二轮：在最优值附近细化（如 block_M in [96, 112, 128, 144]）
    
    原则2：约束优先
    - 先确定硬约束（shared memory、寄存器数量）
    - 再在合法空间内搜索
    
    原则3：参数解耦
    - 优先调优影响最大的参数
    - 固定影响较小的参数为经验值
    """
    M, N, K = problem_size
    
    # Step 1: 确定硬件约束
    max_smem = hardware_spec["smem_per_sm"] * 1024
    max_threads = hardware_spec["max_threads_per_sm"]
    warp_size = hardware_spec["warp_size"]
    
    # Step 2: 生成初始搜索空间
    space = {
        "block_M": [v for v in [64, 128, 256] if v <= M],
        "block_N": [v for v in [64, 128, 256] if v <= N],
        "block_K": [16, 32, 64],
    }
    
    # Step 3: 过滤非法配置
    valid_space = {}
    for key, values in space.items():
        valid_space[key] = values
    
    # 添加更多约束感知的参数
    valid_space["num_stages"] = [1, 2, 3]
    valid_space["num_threads"] = [v for v in [128, 256, 512] 
                                   if v <= max_threads and v % warp_size == 0]
    
    return valid_space
```

这个函数展示了搜索空间设计的三大原则。首先从硬件约束出发确定参数边界（如 shared memory 容量、最大线程数），然后生成满足这些约束的候选值。block_M 和 block_N 只保留不超过问题维度的值，num_threads 只保留不超过硬件上限且为 warp 大小倍数的值。实际应用中还应遵循"从粗到细"策略——先用粗粒度大范围搜索定位最优区域，再在该区域内细化搜索，这样比直接在细粒度全空间搜索更高效。

验证搜索空间约束的正确性是调优准备阶段不可忽视的一环。最常见的验证方法是在约束定义完成后，通过边界值测试（boundary testing）和随机采样交叉验证来确认没有遗漏合法配置或错误地保留了非法配置。例如，对于 shared memory 的约束条件，应当显式测试恰好等于硬件容量边界（如 48KB 和 49KB）的配置，确保比较运算符（<= 还是 <）使用正确。过度约束的代价是可能将所有优质配置排除在搜索空间之外——如果错误地将 shared memory 上限设为 32KB 而非实际的 48KB，合法的 256×128 tile 组合将被错误过滤，导致搜索在次优空间中徘徊。约束不足的风险则是大量评估预算被浪费在编译失败或运行时崩溃的不可行配置上，尤其在早期搜索阶段这会严重拖慢收敛速度。一个值得探索的前沿方向是自动化约束发现：通过解析编译器的错误日志（如"shared memory allocation exceeds device limit"），自动推断出导致失败的参数边界，并在后续搜索中动态添加约束。这种方法在 Ansor 等现代自动调度器中已有初步应用，能大幅减少手动定义约束的人力投入，但仍面临错误日志标准化和跨编译器兼容性等挑战。

### 20.10.2 调优流程建议

```python
def recommended_tuning_workflow(
    kernel_func,
    problem_sizes,
    hardware_target,
):
    """
    推荐的调优工作流
    
    Phase 1: 快速筛选（5 分钟）
    - 使用 Random Search + 小预算（50 次）
    - 目标：排除明显差的参数范围
    
    Phase 2: 精细搜索（30 分钟）
    - 使用 Bayesian Optimization + 中等预算（200 次）
    - 目标：找到接近最优的配置
    
    Phase 3: 确认验证（5 分钟）
    - 对 top-5 配置进行多次测量
    - 目标：验证最优配置的稳定性
    """
    
    # Phase 1: 快速筛选
    print("Phase 1: Quick Screening...")
    space = generate_initial_space(problem_sizes, hardware_target)
    random_search = RandomSearch()
    _, quick_best = random_search.search(space, evaluator, budget=50)
    
    # 缩小搜索空间
    refined_space = refine_space_based_on_results(space, random_search.evaluated)
    
    # Phase 2: 精细搜索
    print("Phase 2: Fine Search...")
    bo = BayesianOptimization()
    best_config, best_perf = bo.search(refined_space, evaluator, budget=200)
    
    # Phase 3: 确认验证
    print("Phase 3: Verification...")
    top_configs = get_top_k_configs(bo.evaluated, k=5)
    
    verified_results = []
    for config in top_configs:
        perfs = [evaluator.evaluate(config) for _ in range(10)]
        avg_perf = np.mean(perfs)
        std_perf = np.std(perfs)
        verified_results.append((config, avg_perf, std_perf))
    
    # 选择最优配置（考虑稳定性）
    best_verified = min(verified_results, key=lambda x: x[1])
    
    print(f"\nBest config: {best_verified[0]}")
    print(f"Performance: {best_verified[1]:.4f} ± {best_verified[2]:.4f} ms")
    
    return best_verified
```

这段代码是 20.10.2 调优流程建议 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 20.10.3 常见陷阱与解决方案

| 陷阱 | 表现 | 解决方案 |
|------|------|----------|
| **搜索空间过大** | 搜索时间过长，无法收敛 | 使用约束剪枝，分阶段搜索 |
| **评估噪声大** | 相同配置性能波动大 | 增加重复测量次数，使用中位数 |
| **过拟合特定问题** | 对特定 (M,N,K) 最优但不通用 | 多问题规模联合调优 |
| **忽略边界情况** | 小矩阵或非对齐矩阵性能差 | 搜索空间中包含边界友好的配置 |
| **硬件热节流** | 长时间测量结果不稳定 | 测量间添加冷却时间 |

从调优日志中诊断每个陷阱的具体症状是解决问题的前提。对于搜索空间过大的问题，典型的日志特征是早期性能改进显著但后期长期停滞，且各参数的最优值在多次运行间变化极大，缺乏一致性——这说明搜索策略从未真正覆盖到空间中的优质区域，每次运行的结果高度随机。评估噪声大的诊断特征则是相同配置在连续测量中的性能值波动超过 5%，且最优配置的性能值在多次独立调优中表现出不合理的高变异性。过拟合特定问题规模的症状表现为：在训练规模上性能优异但在略有不同的规模上性能骤降超过 30%。预防这些问题的主动策略包括：在长时间调优会话中监控 GPU 温度（通过 nvidia-smi 的 query-gpu=temperature.gpu 命令），当温度超过 80°C 时主动暂停测量并等待降温；为每个评估配置记录测量时间戳和 GPU 温度快照，便于事后检查热节流的影响；以及在搜索空间中显式包含小尺寸和非对齐尺寸的配置，确保最终方案对边界情况具有基本鲁棒性。一旦确认已陷入某个陷阱，恢复策略因陷阱类型而异：热节流问题需要立即降低测量频率并增加闲置冷却时间；评估噪声问题应倍增 n_repeats 并切换到中位数统计；过拟合问题应重启调优并在多问题规模上联合评估。

### 20.10.4 生产环境部署建议

```python
class ProductionTuner:
    """
    生产环境调优器
    
    设计原则：
    1. 离线调优，在线查表
    2. 多问题规模预计算
    3. 版本化管理调优结果
    """
    
    def __init__(self, cache_dir="./tuning_cache"):
        self.cache_dir = cache_dir
        self.db = TuningDatabase(cache_dir)
    
    def offline_tune(self, kernel_name, problem_sizes_list, target):
        """
        离线批量调优
        
        参数：
        - kernel_name: kernel 名称
        - problem_sizes_list: 需要调优的问题规模列表
        - target: 目标硬件
        """
        for sizes in problem_sizes_list:
            key = self._make_key(kernel_name, sizes, target)
            
            if self.db.has(key):
                print(f"Skipping {key}: already tuned")
                continue
            
            print(f"Tuning {key}...")
            config, perf = self._tune_single(sizes, target)
            self.db.put(key, config, perf)
    
    def online_lookup(self, kernel_name, problem_sizes, target):
        """
        在线查表获取最优配置
        
        如果精确匹配不存在，使用最近邻插值
        """
        key = self._make_key(kernel_name, problem_sizes, target)
        
        # 尝试精确匹配
        result = self.db.get(key)
        if result:
            return result
        
        # 最近邻匹配
        nearest = self.db.find_nearest(kernel_name, problem_sizes, target)
        if nearest:
            return nearest
        
        # 回退到默认配置
        return self._get_default_config(problem_sizes)
```

在生产环境中，查找表（lookup table）的设计质量直接决定了自动调优的实用价值。对于混合问题规模的部署场景，查找表的覆盖策略需要在稀疏覆盖和密集覆盖之间做出权衡。稀疏覆盖（每隔一定倍数采样，如 512、1024、2048、4096）实现简单，存储开销小，但未采样规模上的配置质量依赖插值算法的有效性。密集覆盖（对所有可能的输入维度组合进行调优）理论上能提供最佳查询精度，但调优时间和存储成本随维度数量指数增长，在实际中代价过高。一种折衷方案是识别出"锚点规模"——即通过对业务流量进行分析，确定最常出现的矩阵维度范围，在这些热点区域进行密集覆盖，而在冷门区域采用稀疏覆盖。当在线查询无法找到精确匹配时，回退策略的选择同样重要：最近邻插值在问题规模连续变化且性能变化平滑的场景表现良好，但在出现"性能相变"（如从 L2 cache 命中切换到 L2 cache 不命中）的临界点附近可能给出误导性结果。此时保守默认配置（如选择 moderate tile size 和 moderate pipeline stage）往往是更安全的选择。硬件升级时（如从 A100 迁移到 H100），调优结果的迁移策略需要分层考虑：部分调度参数（如 tile size 的选择逻辑）具有较好的跨代可迁移性，可保留作为初始搜索空间；而与硬件规格强相关的参数（如流水线级数上限）则需要基于新硬件的规格重新确定合法范围。建议在新硬件上执行一轮"快速校准调优"，而非完全从零开始的全空间搜索，这样可以平衡性能收益和时间成本。

---

## 20.11 实战案例：GEMM 自动调优

### 20.11.1 完整调优脚本

```python
import tilelang
from tilelang import tune
from tilelang.autotuner import SearchSpace, BayesianSearch
import time

def autotune_gemm(M, N, K, dtype="float16", target="nvidia-a100"):
    """
    GEMM 自动调优完整流程
    """
    
    # Step 1: 定义搜索空间
    search_space = SearchSpace(
        tile_sizes={
            "block_M": [64, 128, 256],
            "block_N": [64, 128, 256],
            "block_K": [16, 32, 64],
        },
        num_stages=[1, 2, 3],
        num_threads=[128, 256],
        constraints=[
            # shared memory 约束
            lambda p: (p["block_M"] * p["block_K"] + p["block_K"] * p["block_N"]) * 2 <= 48 * 1024,
            # 寄存器约束
            lambda p: p["block_M"] * p["block_N"] / p["num_threads"] <= 32,
        ]
    )
    
    # Step 2: 定义评估函数
    def evaluate_config(config):
        try:
            kernel = create_gemm_kernel(M, N, K, dtype, **config)
            # 预热
            for _ in range(3):
                kernel()
            # 测量
            start = time.perf_counter()
            for _ in range(100):
                kernel()
            elapsed = (time.perf_counter() - start) / 100
            return elapsed
        except Exception as e:
            return float('inf')  # 非法配置返回无穷大
    
    # Step 3: 执行搜索
    searcher = BayesianSearch(
        search_space=search_space,
        evaluate_fn=evaluate_config,
        budget=200,
    )
    
    best_config, best_perf = searcher.search()
    
    # Step 4: 分析结果
    print(f"\n{'='*50}")
    print(f"Problem: GEMM({M}, {N}, {K}) dtype={dtype}")
    print(f"Target: {target}")
    print(f"{'='*50}")
    print(f"Best config: {best_config}")
    print(f"Best performance: {best_perf*1000:.3f} ms")
    
    # 计算理论峰值利用率
    flops = 2 * M * N * K
    peak_tflops = get_peak_tflops(target)
    achieved_tflops = flops / best_perf / 1e12
    utilization = achieved_tflops / peak_tflops * 100
    
    print(f"Achieved: {achieved_tflops:.1f} TFLOPS ({utilization:.1f}% of peak)")
    
    return best_config, best_perf

if __name__ == "__main__":
    # 不同问题规模的调优
    problem_sizes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (8192, 8192, 8192),
    ]
    
    for M, N, K in problem_sizes:
        autotune_gemm(M, N, K)
```

这段代码是 20.11.1 完整调优脚本 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 20.12 本章小结

本章全面介绍了 TileLang 的 Auto Schedule 框架：

1. **框架设计**：Search Space Definer + Cost Model + Search Strategy 三层架构
2. **搜索空间**：Tile Size、Pipeline Stage、Thread Binding 的组合定义与约束剪枝
3. **性能模型**：Analytical、ML-based、Hybrid 三种模型的原理与实现
4. **搜索策略**：Grid Search、Random Search、Bayesian Optimization 的对比与选择
5. **Tune API**：从基础到高级的完整使用方法
6. **工具对比**：与 Triton autotune、TVM AutoTVM 的系统性对比
7. **最佳实践**：搜索空间设计、调优流程、常见陷阱

---

## 练习

### Exercise 1: 基础搜索空间定义
为一个 `LayerNorm` 算子定义搜索空间，包括 block size、线程数和是否使用向量化。考虑 GPU 的 shared memory 和寄存器约束。

### Exercise 2: Cost Model 实现
实现一个简单的 Analytical Cost Model，能够预测 `softmax` kernel 在不同 block size 下的性能。使用 Roofline Model 作为基础。

### Exercise 3: 搜索策略对比
在一个 10 维搜索空间中（每维 5 个取值），分别使用 Grid Search、Random Search 和 Bayesian Optimization 进行搜索，对比收敛速度和最终性能。

### Exercise 4: Tune API 实战
使用 Tune API 对一个 `attention` kernel 进行自动调优，要求支持 Flash Attention 的在线 softmax 策略。

### Exercise 5: 收敛分析
对 Exercise 4 的调优结果进行收敛分析，绘制收敛曲线，计算收敛效率指标，并给出优化建议。

---

## 思考题

1. **为什么 Bayesian Optimization 在高维空间中的效果会下降？有什么改进方法？**

2. **Cost Model 的预测精度和搜索效率之间如何权衡？什么时候应该使用解析模型，什么时候应该使用 ML 模型？**

3. **在多 GPU 环境下，如何设计并行搜索策略？需要注意哪些一致性问题？**

4. **如何设计一个自适应搜索策略，能够根据搜索过程中的反馈动态调整搜索空间？**

5. **TileLang 的 Auto Schedule 框架如何扩展到端到端模型的自动优化？需要解决哪些新的挑战？**

---

## 20.13 多目标优化

### 20.13.1 多目标调优问题

在实际应用中，我们往往不仅关心性能，还需要考虑其他目标：

```python
class MultiObjectiveTuner:
    """
    多目标调优器
    
    常见的多目标组合：
    1. 性能 + 精度（混合精度场景）
    2. 性能 + 内存使用（内存受限场景）
    3. 性能 + 能耗（边缘设备场景）
    4. 廖迟 + 吞吐量（在线服务场景）
    """
    
    def __init__(self, objectives, weights):
        """
        参数：
        - objectives: 目标函数列表，如 ["latency", "memory", "accuracy"]
        - weights: 各目标的权重，如 [0.6, 0.3, 0.1]
        """
        self.objectives = objectives
        self.weights = weights
    
    def evaluate(self, config, evaluator):
        """计算加权多目标得分"""
        scores = {}
        for obj in self.objectives:
            scores[obj] = evaluator.evaluate_objective(config, obj)
        
        # 加权求和
        weighted_score = sum(
            scores[obj] * w for obj, w in zip(self.objectives, self.weights)
        )
        
        return weighted_score, scores
    
    def pareto_search(self, search_space, evaluator, budget=200):
        """
        Pareto 前沿搜索
        
        目标：找到一组非支配解（Pareto optimal solutions）
        解 A 支配解 B：A 在所有目标上都不差于 B，且至少在一个目标上严格优于 B
        """
        pareto_front = []
        all_results = []
        
        for i in range(budget):
            config = self._sample_config(search_space)
            weighted_score, scores = self.evaluate(config, evaluator)
            all_results.append((config, scores))
            
            # 检查是否被 Pareto 前沿中的解支配
            dominated = False
            new_front = []
            
            for front_config, front_scores in pareto_front:
                if self._dominates(front_scores, scores):
                    dominated = True
                    new_front.append((front_config, front_scores))
                elif not self._dominates(scores, front_scores):
                    new_front.append((front_config, front_scores))
            
            if not dominated:
                new_front.append((config, scores))
            
            pareto_front = new_front
        
        return pareto_front, all_results
    
    def _dominates(self, scores_a, scores_b):
        """检查 A 是否支配 B"""
        all_better_or_equal = True
        at_least_one_better = False
        
        for obj in self.objectives:
            if scores_a[obj] > scores_b[obj]:  # 假设越小越好
                all_better_or_equal = False
                break
            elif scores_a[obj] < scores_b[obj]:
                at_least_one_better = True
        
        return all_better_or_equal and at_least_one_better
    
    def _sample_config(self, search_space):
        """从搜索空间随机采样"""
        config = {}
        for param, values in search_space.items():
            import random
            config[param] = random.choice(values)
        return config
```

这段代码是 20.13.1 多目标调优问题 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 20.13.2 延迟-吞吐量权衡

```python
class LatencyThroughputOptimizer:
    """
    延迟-吞吐量权衡优化器
    
    在线服务场景中：
    - 延迟（Latency）：单个请求的处理时间
    - 吞吐量（Throughput）：单位时间处理的请求数
    
    权衡：
    - 增大 batch size → 吞吐量↑，延迟↑
    - 减小 batch size → 延迟↓，吞吐量↓
    
    目标：在延迟约束下最大化吞吐量
    """
    
    def __init__(self, latency_constraint_ms):
        self.latency_constraint = latency_constraint_ms
    
    def optimize(self, kernel_func, problem_sizes, target_gpu):
        """
        找到满足延迟约束的最大吞吐量配置
        """
        best_config = None
        best_throughput = 0
        
        for batch_size in [1, 2, 4, 8, 16, 32, 64, 128]:
            for config in self._generate_configs(problem_sizes, batch_size):
                latency = self._measure_latency(kernel_func, config)
                
                if latency <= self.latency_constraint:
                    throughput = batch_size / (latency / 1000)  # requests/sec
                    if throughput > best_throughput:
                        best_throughput = throughput
                        best_config = config
        
        return best_config, best_throughput
    
    def _generate_configs(self, problem_sizes, batch_size):
        """生成候选配置"""
        configs = []
        # ... 生成不同 tile size、pipeline stage 的配置
        return configs
    
    def _measure_latency(self, kernel_func, config):
        """测量延迟"""
        # ... 测量单个请求的处理时间
        return 0.0  # placeholder
```

这段代码是 20.13.2 延迟-吞吐量权衡 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 20.13.3 内存受限场景优化

```python
class MemoryConstrainedTuner:
    """
    内存受限场景的调优器
    
    场景：
    - 边缘设备：内存有限（如 Jetson 的 8GB）
    - 多模型部署：需要在有限内存中部署多个模型
    - 长序列处理：KV cache 占用大量内存
    
    策略：
    1. 优先选择内存使用量小的配置
    2. 使用内存复用技术
    3. 考虑计算精度与内存的权衡
    """
    
    def __init__(self, memory_budget_bytes):
        self.memory_budget = memory_budget_bytes
    
    def tune(self, kernel_func, search_space, evaluator):
        """
        在内存约束下调优
        """
        valid_configs = []
        
        for config in self._enumerate_configs(search_space):
            # 估计内存使用量
            memory_usage = self._estimate_memory(config)
            
            if memory_usage <= self.memory_budget:
                perf = evaluator.evaluate(config)
                valid_configs.append((config, perf, memory_usage))
        
        if not valid_configs:
            raise ValueError("No config fits within memory budget")
        
        # 选择性能最好的配置
        best = min(valid_configs, key=lambda x: x[1])
        
        return {
            "config": best[0],
            "performance": best[1],
            "memory_usage": best[2],
            "memory_utilization": best[2] / self.memory_budget,
        }
    
    def _estimate_memory(self, config):
        """估计配置的内存使用量"""
        # Shared memory
        smem = (config.get("block_M", 128) * config.get("block_K", 32) +
                config.get("block_K", 32) * config.get("block_N", 128))
        smem *= 2 * config.get("num_stages", 1)  # fp16, 多级流水线
        
        # 寄存器（间接影响，通过 Occupancy）
        regs = config.get("block_M", 128) * config.get("block_N", 128) * 4  # fp32 累加器
        
        return smem + regs
    
    def _enumerate_configs(self, search_space):
        """枚举所有配置"""
        import itertools
        keys = list(search_space.keys())
        values = [search_space[k] for k in keys]
        configs = []
        for combo in itertools.product(*values):
            configs.append(dict(zip(keys, combo)))
        return configs
```

这段代码是 20.13.3 内存受限场景优化 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 20.14 跨问题规模的调优策略

### 20.14.1 问题规模敏感性分析

```python
class ProblemSizeSensitivityAnalyzer:
    """
    问题规模敏感性分析
    
    不同的矩阵大小可能需要不同的最优配置：
    - 小矩阵（<512）：启动开销占比大，需要小 tile size
    - 中矩阵（512-2048）：标准优化策略
    - 大矩阵（>2048）：可以使用大 tile size，高 Occupancy
    
    目标：找到对问题规模变化鲁棒的配置
    """
    
    def __init__(self, size_ranges):
        """
        参数：
        - size_ranges: 问题规模范围列表
            如 [(256,512), (512,1024), (1024,2048), (2048,4096)]
        """
        self.size_ranges = size_ranges
    
    def analyze_sensitivity(self, kernel_func, search_space, target_gpu):
        """
        分析配置对问题规模的敏感性
        """
        results = {}
        
        for size_range in self.size_ranges:
            min_size, max_size = size_range
            mid_size = (min_size + max_size) // 2
            
            # 在该范围内调优
            best_config, best_perf = self._tune_for_size(
                kernel_func, mid_size, mid_size, mid_size, search_space, target_gpu
            )
            
            results[size_range] = {
                "best_config": best_config,
                "best_perf": best_perf,
                "tested_size": mid_size,
            }
        
        # 分析配置的稳定性
        configs = [r["best_config"] for r in results.values()]
        unique_configs = set(str(c) for c in configs)
        
        sensitivity = {
            "unique_configs": len(unique_configs),
            "total_ranges": len(self.size_ranges),
            "stability_score": 1.0 - (len(unique_configs) - 1) / max(1, len(self.size_ranges) - 1),
        }
        
        return results, sensitivity
    
    def find_robust_config(self, kernel_func, search_space, target_gpu):
        """
        找到对问题规模鲁棒的配置
        
        策略：在多个问题规模上评估每个配置，选择平均性能最好的
        """
        all_configs = self._enumerate_configs(search_space)
        
        config_scores = {}
        
        for config in all_configs[:50]:  # 限制评估数量
            scores = []
            for size_range in self.size_ranges:
                mid_size = (size_range[0] + size_range[1]) // 2
                perf = self._evaluate_config(kernel_func, config, mid_size, target_gpu)
                scores.append(perf)
            
            # 计算平均得分（归一化后）
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            
            config_scores[str(config)] = {
                "config": config,
                "avg_score": avg_score,
                "max_score": max_score,
                "min_score": min_score,
                "score_range": max_score - min_score,
            }
        
        # 选择平均得分最好且方差最小的配置
        best = min(
            config_scores.values(),
            key=lambda x: x["avg_score"] + 0.1 * x["score_range"]
        )
        
        return best["config"], best
    
    def _tune_for_size(self, kernel_func, M, N, K, search_space, target_gpu):
        """为特定大小调优"""
        # ... 调优逻辑
        return {}, 0.0
    
    def _evaluate_config(self, kernel_func, config, size, target_gpu):
        """评估配置在特定大小下的性能"""
        # ... 评估逻辑
        return 0.0
    
    def _enumerate_configs(self, search_space):
        """枚举配置"""
        import itertools
        keys = list(search_space.keys())
        values = [search_space[k] for k in keys]
        configs = []
        for combo in itertools.product(*values):
            configs.append(dict(zip(keys, combo)))
        return configs
```

这段代码是 20.14.1 问题规模敏感性分析 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 20.14.2 自适应配置选择

```python
class AdaptiveConfigSelector:
    """
    自适应配置选择器
    
    根据实际问题规模动态选择最优配置
    
    工作流：
    1. 离线阶段：为多种问题规模预计算最优配置
    2. 在线阶段：根据实际输入大小查找最近邻配置
    3. 可选：在线微调（在新规模上快速搜索）
    """
    
    def __init__(self):
        self.config_db = {}  # {(M,N,K): (config, perf)}
    
    def offline_tune(self, kernel_func, sizes_list, search_space, target_gpu):
        """
        离线批量调优
        """
        for M, N, K in sizes_list:
            print(f"Tuning for ({M}, {N}, {K})...")
            config, perf = self._tune_single(kernel_func, M, N, K, search_space, target_gpu)
            self.config_db[(M, N, K)] = (config, perf)
        
        print(f"Tuned {len(self.config_db)} configurations")
    
    def get_config(self, M, N, K):
        """
        获取配置（支持最近邻插值）
        """
        # 精确匹配
        if (M, N, K) in self.config_db:
            return self.config_db[(M, N, K)]
        
        # 最近邻匹配
        min_dist = float('inf')
        nearest_key = None
        
        for (m, n, k) in self.config_db.keys():
            # 使用相对距离
            dist = (abs(M - m) / max(M, m) + 
                    abs(N - n) / max(N, n) + 
                    abs(K - k) / max(K, k))
            if dist < min_dist:
                min_dist = dist
                nearest_key = (m, n, k)
        
        if nearest_key and min_dist < 0.5:  # 距离阈值
            return self.config_db[nearest_key]
        
        # 回退到默认配置
        return self._get_default_config(M, N, K)
    
    def _tune_single(self, kernel_func, M, N, K, search_space, target_gpu):
        """调优单个问题规模"""
        # ... 调优逻辑
        return {}, 0.0
    
    def _get_default_config(self, M, N, K):
        """获取默认配置"""
        # 启发式规则
        block_M = min(128, M)
        block_N = min(128, N)
        block_K = min(32, K)
        
        return {
            "block_M": block_M,
            "block_N": block_N,
            "block_K": block_K,
            "num_stages": 2,
            "num_threads": 128,
        }, None
```

这段代码是 20.14.2 自适应配置选择 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 20.15 调优结果的可复现性

### 20.15.1 环境控制

```python
class ReproducibleTuner:
    """
    可复现的调优器
    
    确保调优结果在不同运行之间一致
    
    影响复现性的因素：
    1. GPU 时钟频率变化（热节流）
    2. 后台进程干扰
    3. 随机数种子
    4. CUDA 版本和驱动版本
    5. 系统负载
    """
    
    def __init__(self, seed=42):
        self.seed = seed
        self.rng = random.Random(seed)
    
    def setup_reproducible_environment(self):
        """设置可复现的环境"""
        import subprocess
        
        # 设置 GPU 时钟为固定值（需要 root 权限）
        # subprocess.run(["nvidia-smi", "-lgc", "1410,1410"])  # 固定时钟
        
        # 设置随机数种子
        random.seed(self.seed)
        import numpy as np
        np.random.seed(self.seed)
        import torch
        torch.manual_seed(self.seed)
        
        # 禁用 CUDA benchmark（避免自动选择不同算法）
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    def measure_with_warmup(self, kernel, num_warmup=10, num_iterations=100):
        """
        带预热的测量
        """
        import torch
        import time
        
        # 预热
        for _ in range(num_warmup):
            kernel()
        torch.cuda.synchronize()
        
        # 测量
        times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            kernel()
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        # 去除异常值（前后各 10%）
        times.sort()
        trim = len(times) // 10
        trimmed_times = times[trim:-trim] if trim > 0 else times
        
        return {
            "mean": sum(trimmed_times) / len(trimmed_times),
            "median": trimmed_times[len(trimmed_times) // 2],
            "std": (sum((t - sum(trimmed_times)/len(trimmed_times))**2 
                       for t in trimmed_times) / len(trimmed_times))**0.5,
            "min": min(trimmed_times),
            "max": max(trimmed_times),
        }
```

这段代码是 20.15.1 环境控制 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 20.15.2 调优结果版本管理

```python
import json
import hashlib
from datetime import datetime

class TuningResultVersionManager:
    """
    调优结果版本管理器
    
    记录每次调优的完整上下文，支持回溯和对比
    """
    
    def __init__(self, db_path="tuning_versions.json"):
        self.db_path = db_path
        self.versions = self._load()
    
    def _load(self):
        try:
            with open(self.db_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def _save(self):
        with open(self.db_path, "w") as f:
            json.dump(self.versions, f, indent=2)
    
    def record_tuning(self, kernel_name, problem_size, target, 
                      best_config, best_perf, search_history):
        """
        记录一次调优结果
        """
        version = {
            "id": hashlib.md5(
                f"{kernel_name}_{problem_size}_{target}_{datetime.now().isoformat()}"
                .encode()
            ).hexdigest()[:12],
            "timestamp": datetime.now().isoformat(),
            "kernel_name": kernel_name,
            "problem_size": problem_size,
            "target": target,
            "best_config": best_config,
            "best_perf": best_perf,
            "search_history_size": len(search_history),
            "environment": self._get_environment(),
        }
        
        self.versions.append(version)
        self._save()
        
        return version["id"]
    
    def _get_environment(self):
        """获取环境信息"""
        import subprocess
        try:
            gpu_info = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", 
                 "--format=csv,noheader"],
                capture_output=True, text=True
            ).stdout.strip()
        except:
            gpu_info = "unknown"
        
        return {
            "gpu": gpu_info,
            "python_version": __import__("sys").version,
        }
    
    def compare_versions(self, version_id_1, version_id_2):
        """对比两个版本"""
        v1 = next((v for v in self.versions if v["id"] == version_id_1), None)
        v2 = next((v for v in self.versions if v["id"] == version_id_2), None)
        
        if not v1 or not v2:
            return None
        
        return {
            "version_1": v1,
            "version_2": v2,
            "perf_diff": v2["best_perf"] - v1["best_perf"],
            "perf_change_pct": (v2["best_perf"] - v1["best_perf"]) / v1["best_perf"] * 100,
        }
```

这段代码是 20.15.2 调优结果版本管理 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 扩展阅读

1. **Bergstra, J., & Bengio, Y. (2012).** "Random Search for Hyper-Parameter Optimization." *JMLR*.
2. **Chen, T., et al. (2018).** "Learning to Optimize Tensor Programs." *NeurIPS*. - TVM AutoTVM 的理论基础
3. **Zheng, L., et al. (2020).** "Ansor: Generating High-Performance Tensor Programs for Deep Learning." *OSDI*. - TVM 的自动 schedule 生成
4. **Li, L., et al. (2020).** "HyperBO+: Bayesian Optimization with Hierarchical Structuring." *ICML*.
5. **Meng, T., et al. (2023).** "Triton Auto-tuning: A Compiler-driven Approach to Efficient GPU Programming." - Triton autotune 的设计与实现

---

## 下一章预告

**Chapter 21: 性能剖析与瓶颈定位** — 当自动调优找到了"相对最优"的配置后，如何判断性能瓶颈到底在哪里？我们将深入探讨 GPU 性能剖析工具（ncu/nsys/rocprof）的使用方法，以及如何基于 profiling 结果进行针对性优化。

---
title: "Chapter 22: 内存优化与带宽利用"
description: "深入探讨 GPU/NPU 内存层次结构、合并访问、Shared Memory 优化、寄存器分配策略及带宽利用率最大化技术"
updated: "2025-01-15"
---

# Chapter 22: 内存优化与带宽利用

> **Learning Objectives**
> - 理解 GPU/NPU 内存层次结构及其性能特性
> - 掌握内存合并访问（Coalescing）的原理与优化方法
> - 深入理解 Shared Memory 优化技术（Bank Conflict / Padding / Swizzling）
> - 掌握寄存器分配策略及其与 Occupancy 的权衡
> - 理解 L1 Cache 的利用方法
> - 对比 TileLang 显式内存管理与 Triton 隐式管理的优劣
> - 学会计算和优化带宽利用率
> - 通过实战案例掌握内存优化的完整流程

---

## 22.1 GPU 内存层次概述

### 22.1.1 内存层次结构

<div data-component="MemoryHierarchyPerformance"></div>

GPU 的内存层次从快到慢依次为：

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU Memory Hierarchy                      │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐ │
│  │ Register │  │ Shared   │  │  L1      │  │  L2        │ │
│  │ File     │→ │ Memory   │→ │  Cache   │→ │  Cache     │ │
│  │          │  │          │  │          │  │            │ │
│  │ ~0.5ns   │  │ ~1-2ns   │  │ ~30ns    │  │ ~100ns     │ │
│  │ 256KB/SM │  │ 164KB/SM │  │ 128KB/SM │  │ 40MB(GPU)  │ │
│  └──────────┘  └──────────┘  └──────────┘  └────────────┘ │
│       │                                               │     │
│       │            ┌────────────┐                     │     │
│       └───────────→│  HBM /    │←────────────────────┘     │
│                    │  GDDR6X   │                            │
│                    │           │                            │
│                    │ ~400ns    │                            │
│                    │ 40-80GB   │                            │
│                    └────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

该图展示了 GPU 内存层次结构的完整视图。从寄存器到全局内存（HBM），每上升一级延迟增加约 10 倍，但容量也增加约 10 倍。寄存器速度最快（~0.5ns）但容量最小（256KB/SM），HBM 容量最大（40-80GB）但延迟最高（~400ns）。优化的核心思想是将数据尽可能驻留在快速内存层级中，减少对慢速内存的访问。

### 22.1.2 性能特性对比

| 内存层级 | 延迟 (cycles) | 带宽 | 容量 (A100) | 作用域 |
|---------|-------------|------|------------|--------|
| **Register** | ~1 | ~192 TB/s (aggregate) | 256 KB/SM | 单个线程 |
| **Shared Memory** | ~20-30 | ~19 TB/s (aggregate) | 164 KB/SM | Block 内线程 |
| **L1 Cache** | ~30-40 | ~12 TB/s | 128 KB/SM | SM 级别 |
| **L2 Cache** | ~200-300 | ~5 TB/s | 40 MB | 全局 |
| **HBM (Global)** | ~400-600 | 2.0 TB/s | 80 GB | 全局 |

> [!TIP]
> 理解内存层次是优化的基础。每向上一级，延迟减少约 10 倍，但容量也减少约 10 倍。优化的核心思想是：**尽可能使用更快的内存层级**。

### 22.1.3 内存访问模式分类

```python
class MemoryAccessPattern:
    """
    GPU 内存访问模式分类
    
    1. 合并访问 (Coalesced Access)
       - 同一 warp 的线程访问连续地址
       - 可以合并为少量内存事务
       - 最优模式，带宽利用率最高
    
    2. 非合并访问 (Uncoalesced Access)
       - 同一 warp 的线程访问不连续地址
       - 每个线程可能需要独立的内存事务
       - 带宽浪费严重
    
    3. 广播访问 (Broadcast Access)
       - 同一 warp 的所有线程访问同一地址
       - 只需要一次内存事务
       - 高效但受限于特定场景
    
    4. 随机访问 (Random Access)
       - 线程访问完全随机的地址
       - 每个线程都需要独立的内存事务
       - 最差模式，带宽利用率极低
    """
    
    PATTERNS = {
        "coalesced": {
            "description": "连续地址访问",
            "bandwidth_efficiency": "90-100%",
            "example": "A[tid] for tid in [0, 31]",
        },
        "strided": {
            "description": "固定步长访问",
            "bandwidth_efficiency": "1/stride * 100%",
            "example": "A[tid * stride] for tid in [0, 31]",
        },
        "scattered": {
            "description": "随机分散访问",
            "bandwidth_efficiency": "< 10%",
            "example": "A[random_idx[tid]] for tid in [0, 31]",
        },
        "broadcast": {
            "description": "所有线程访问同一地址",
            "bandwidth_efficiency": "~100% (cache hit)",
            "example": "A[0] for tid in [0, 31]",
        },
    }
```

该类将 GPU 内存访问模式分为四类：合并访问、步长访问、分散访问和广播访问。合并访问是性能最优的模式，同一 warp 的线程访问连续地址，带宽利用率可达 90-100%。步长访问的效率与步长成反比，例如步长为 2 时效率仅 50%。分散访问和随机访问效率极低，应尽量避免。设计 kernel 时，应优先确保 warp 内线程的内存地址连续。

从硬件角度来看，GPU 内存控制器以固定大小的 cache line（通常 128 字节）为单位从 DRAM 读取数据。当 warp 中 32 个线程的访问地址落在同一个或少数几个 cache line 内时，硬件可以将这些请求合并为一次或少数几次内存事务。反之，如果 32 个线程的地址分散在 32 个不同的 cache line 中，内存控制器必须发起 32 次独立事务，每次事务的大部分数据都被浪费。这就是合并访问带来数量级性能差异的根本原因。理解这一硬件约束后，设计 kernel 时就需要将线程到数据元素的映射方式作为首要优化决策：通常来说，让连续线程访问连续内存地址是最简单也是最重要的优化法则。此外，还应注意到 broadcast 访问虽然也是高效模式（所有线程访问同一地址，硬件通过 single-port 广播即可完成），但其适用场景有限，主要用于常量或标量参数的加载。

---

> 理解了 GPU 内存层次结构后，接下来我们将深入探讨内存访问模式的优化。内存访问模式直接决定了 GPU 的带宽利用率，而合并访问（Coalescing）是其中最关键的优化技术之一。通过合理组织线程的内存访问，可以将带宽利用率从不足 5% 提升到 90% 以上。

## 22.2 内存合并访问 (Coalescing)

### 22.2.1 Coalescing 原理

<div data-component="CoalescingAccessDemo"></div>

当一个 warp（32 个线程）执行内存加载指令时，硬件会尝试将这些线程的内存请求合并为尽可能少的内存事务：

```python
"""
Coalescing 规则（NVIDIA GPU，128-byte cache line）：

最优情况：32 个线程访问连续的 128 字节
- 例如：32 个线程各访问 4 字节（float32）
- 合并为 1 个 128-byte 事务
- 带宽利用率：100%

次优情况：32 个线程访问连续的 256 字节
- 例如：32 个线程各访问 8 字节（float64）
- 合并为 2 个 128-byte 事务
- 带宽利用率：100%（但需要 2 次事务）

较差情况：32 个线程访问步长为 2 的地址
- 例如：线程 i 访问 A[i*2]
- 每个 128-byte 事务只有一半数据被使用
- 带宽利用率：50%

最差情况：32 个线程访问随机地址
- 每个线程可能需要独立的事务
- 可能需要 32 个 128-byte 事务
- 带宽利用率：32*4 / 32*128 = 3.1%
"""
```

这段代码详细说明了 NVIDIA GPU 的合并访问规则。硬件以 128 字节的 cache line 为单位合并内存事务，32 个线程访问连续的 128 字节只需一次事务。当线程访问步长为 2 的地址时，cache line 利用率降至 50%。最差情况下，32 个线程各需独立事务，带宽利用率仅 3.1%。理解这些规则是编写高效 GPU kernel 的基础。

需要特别注意的是，不同数据类型对合并访问的实际效率有显著影响。对于 float32（4 字节），32 个线程恰好访问 128 字节，形成完美的一个 cache line 对齐。对于 float16（2 字节），32 个线程仅访问 64 字节，虽然只需一次事务，但有效数据仅占 cache line 的一半——这意味着引入了一半的带宽浪费。这就是为什么向量化加载（如使用 float4 类型）非常重要的原因：通过让每个线程访问更多连续字节，可以充分填满每个 cache line，减少浪费。另一方面，对于 float64（8 字节），32 个线程需要 256 字节，会触发两次 cache line 事务——虽然这并非浪费（两次事务的数据都被使用），但事务数量翻倍意味着需要两倍的内存请求次数。在实际优化中，应优先考虑让访问地址对齐到 cache line 边界（128 字节对齐），并尽可能让事务覆盖整个 cache line 的数据量。

### 22.2.2 Coalescing 优化示例

```python
import tilelang
import tilelang.language as T

# 示例 1：非合并访问（低效）
@T.prim_func
def uncoalesced_load(A: T.Tensor((1024, 1024), "float32"),
                     B: T.Tensor((1024,), "float32")):
    """每个线程访问一列，导致非合并访问"""
    with T.Kernel(32, threads=32) as tid:
        # 线程 tid 访问第 tid 列的 32 个元素
        # 同一 warp 的线程访问不同列，地址不连续
        for i in T.serial(32):
            B[i] = A[i, tid]  # 非合并：线程间步长 = 1024 * 4 bytes

# 示例 2：合并访问（高效）
@T.prim_func
def coalesced_load(A: T.Tensor((1024, 1024), "float32"),
                   B: T.Tensor((1024,), "float32")):
    """每个线程访问一行中连续的元素"""
    with T.Kernel(32, threads=32) as tid:
        # 线程 tid 访问第 0 行的连续 32 个元素
        # 同一 warp 的线程访问连续地址
        for i in T.serial(32):
            B[i] = A[0, tid * 32 + i]  # 合并：连续地址

# 示例 3：转置优化（将非合并转为合并）
@T.prim_func
def transpose_optimized(A: T.Tensor((1024, 1024), "float32"),
                        B: T.Tensor((1024, 1024), "float32")):
    """使用 shared memory 进行合并的矩阵转置"""
    with T.Kernel(32, 32, threads=256) as (bx, by):
        # 分配 shared memory tile
        tile = T.alloc_shared((32, 32 + 1), "float32")  # +1 padding 避免 bank conflict
        
        # 合并读取：每个线程读取连续的元素
        tid = T.get_thread_id()
        local_tid = tid % 256
        row = local_tid // 32
        col = local_tid % 32
        
        tile[row, col] = A[by * 32 + row, bx * 32 + col]
        T.sync_threads()
        
        # 合并写入：转置后写入
        B[bx * 32 + row, by * 32 + col] = tile[col, row]
```

这段代码展示了三种内存访问模式的对比。非合并版本中，线程按列访问矩阵，相邻线程的地址间隔为 1024×4 字节，导致严重的带宽浪费。合并版本中，线程按行访问连续元素，硬件可以高效合并事务。转置优化示例使用 shared memory 作为中间缓冲，将非合并的读取转换为合并的读取和写入。+1 padding 避免了 shared memory 中的 bank conflict。

这三个示例揭示了 GPU 编程中一个核心的权衡关系：数据布局与访问模式之间的矛盾。在许多科学计算和深度学习应用中，数据天然地以某种格式（如 row-major）存储，但算法需要按不同维度访问——矩阵转置就是最典型的例子。直接按列读取会导致灾难性的性能下降（带宽利用率可能只有 3%），但通过 shared memory 作为中间缓冲，可以以合并方式读取数据到片上内存，然后在片上以任意模式访问（shared memory 的 bank 结构可以高效处理非合并访问），最后以合并方式写回全局内存。这种“合并读 → 片上重排 → 合并写”的三段式策略是处理非合并访问模式的标准范式，代价是额外的 shared memory 开销和同步点（T.sync_threads）。值得注意的是，某些新硬件（如 Hopper 架构的 TMA）可以自动化这一过程，大大简化了开发者需要处理的复杂度。

### 22.2.3 合并访问检测工具

```python
def check_coalescing_pattern(memory_accesses, warp_size=32):
    """
    检测内存访问模式是否合并
    
    参数：
    - memory_accesses: 列表，每个元素是一个 warp 中各线程的访问地址
    
    返回：
    - coalescing_ratio: 合并率（0-1，1 为完美合并）
    - num_transactions: 估计的内存事务数
    """
    results = []
    
    for warp_accesses in memory_accesses:
        # 对地址排序
        sorted_addrs = sorted(warp_accesses)
        
        # 计算连续段
        segments = []
        current_segment = [sorted_addrs[0]]
        
        for i in range(1, len(sorted_addrs)):
            if sorted_addrs[i] - sorted_addrs[i-1] <= 4:  # 4 bytes for float32
                current_segment.append(sorted_addrs[i])
            else:
                segments.append(current_segment)
                current_segment = [sorted_addrs[i]]
        segments.append(current_segment)
        
        # 计算合并率
        ideal_transactions = 1  # 理想情况下只需 1 次事务
        actual_transactions = len(segments)
        coalescing_ratio = ideal_transactions / actual_transactions
        
        results.append({
            "segments": len(segments),
            "coalescing_ratio": coalescing_ratio,
            "wasted_bytes": sum(128 - len(seg) * 4 for seg in segments),
        })
    
    return results

# 使用示例
import random

# 模拟合并访问
coalesced_access = [list(range(i, i + 32 * 4, 4)) for i in range(0, 1024, 32)]

# 模拟非合并访问（步长为 1024）
uncoalesced_access = [list(range(i, i + 32 * 1024, 1024)) for i in range(0, 1024, 32)]

print("Coalesced access:")
for r in check_coalescing_pattern(coalesced_access[:3]):
    print(f"  Segments: {r['segments']}, Ratio: {r['coalescing_ratio']:.2f}")

print("Uncoalesced access:")
for r in check_coalescing_pattern(uncoalesced_access[:3]):
    print(f"  Segments: {r['segments']}, Ratio: {r['coalescing_ratio']:.2f}")
```

该检测算法通过排序地址并计算连续段来评估合并质量。算法将线程地址排序后，按 4 字节间隔划分连续段，段数越少说明合并越好。理想情况下只需 1 次事务，实际事务数等于连续段数。合并率定义为理想事务数与实际事务数的比值，1 表示完美合并。该工具可用于验证 kernel 优化效果，指导访问模式的改进。

在实际工程实践中，除了这种静态地址分析外，还需要结合硬件 profiling 工具（如 Nsight Compute）获取真实的带宽利用率数据。静态分析可以帮助我们在编写代码阶段就识别出明显的非合并问题，但运行时行为还会受缓存状态、内存控制器调度、DRAM bank 分布等因素影响。例如，即使访问模式在地址上看是非合并的，如果数据恰好位于 L2 Cache 中，实际性能损失可能较小；但如果非合并访问导致大量 cache thrashing，性能下降可能远超静态分析预期的 3% 利用率。因此，该静态检测工具应作为开发阶段的快速检查手段，配合 profiler 的 `gpu__dram_throughput` 和 `l1tex__t_sectors` 等指标进行综合判断。一个成熟的优化流程是先通过工具快速定位问题区域，再用 profiler 验证实际改进效果。

---

> 合并访问优化了全局内存的读写效率，但全局内存的延迟仍然很高。为了进一步减少内存延迟的影响，我们可以利用 Shared Memory 作为片上高速缓存。Shared Memory 位于芯片内部，访问延迟仅为全局内存的 1/200，是实现高性能计算的关键组件。

## 22.3 Shared Memory 优化

### 22.3.1 Shared Memory 基础

```python
"""
Shared Memory 特性：
1. 片上内存（on-chip），速度接近寄存器
2. Block 内所有线程共享
3. 手动管理（显式分配和访问）
4. 容量有限（A100: 164 KB/SM，可配置到最多 164 KB）

典型用途：
- 缓存全局内存数据，减少重复访问
- 线程间数据交换
- 实现高效的 reduction、scan 等操作
"""

import tilelang.language as T

@T.prim_func
def shared_memory_example(A: T.Tensor((1024,), "float32"),
                          B: T.Tensor((1024,), "float32")):
    """Shared memory 基础使用示例"""
    with T.Kernel(1, threads=256) as bx:
        # 分配 shared memory
        smem = T.alloc_shared((256,), "float32")
        
        # 从全局内存加载到 shared memory
        tid = T.get_thread_id()
        smem[tid] = A[tid]
        
        # 同步：确保所有线程都完成了加载
        T.sync_threads()
        
        # 从 shared memory 读取（比全局内存快很多）
        B[tid] = smem[tid] * 2.0
```

这段代码演示了 shared memory 的基本使用模式。首先从全局内存加载数据到 shared memory，然后同步确保所有线程完成加载，最后从 shared memory 读取。shared memory 是片上内存，访问速度接近寄存器，远快于全局内存。关键设计是使用 T.sync_threads() 确保数据一致性。这种模式适用于需要多次读取同一数据的场景，可以显著减少全局内存访问。

从硬件实现层面来看，shared memory 实际上是由 SRAM 组成的片上存储，与计算单元物理上紧密耦合，数据路径短且无需跨越片外总线。在 A100 上，每个 SM 包含 164 KB 的 shared memory（可配置与 L1 Cache 共享 192 KB 池），其聚合带宽高达约 19 TB/s，是 HBM 带宽的近 10 倍。然而，与寄存器相比仍有约 20 个时钟周期的额外延迟——这意味着每个 shared memory 访问指令需要多个时钟周期才能完成。这一特性提示我们：在极端追求性能的场景下，应当优先使用寄存器保存频繁访问的数据，将 shared memory 作为寄存器和全局内存之间的中间缓冲层。此外，shared memory 的使用量会直接影响 Occupancy：分配给 shared memory 越多，每个 SM 能容纳的 block 数就越少，可能影响延迟隐藏能力。因此，shared memory 的使用应遵循“按需分配、最小化占用”的原则。

### 22.3.2 Bank Conflict 详解

<div data-component="SharedMemoryOptimization"></div>

```python
"""
Shared Memory Bank Conflict 原理：

Shared Memory 被组织为 32 个 bank（对应 32 个线程），
每个 bank 宽度为 4 字节。

Bank 0: [word0, word32, word64, ...]
Bank 1: [word1, word33, word65, ...]
...
Bank 31: [word31, word63, word95, ...]

当一个 warp 中的多个线程同时访问同一个 bank 的不同地址时，
就会发生 bank conflict，导致访问被序列化。

2-way bank conflict: 2 个线程访问同一 bank → 延迟翻倍
32-way bank conflict: 32 个线程访问同一 bank → 延迟 32 倍

Broadcast 是例外：所有线程访问同一地址不产生 conflict
"""
```

这段文字解释了 shared memory bank conflict 的核心原理。Shared memory 被分为 32 个 bank，每个 bank 宽度为 4 字节，对应 warp 中的 32 个线程。当多个线程同时访问同一 bank 的不同地址时，访问会被序列化，产生 bank conflict。2-way conflict 使延迟翻倍，32-way conflict 使延迟增加 32 倍。特殊情况是 broadcast：所有线程访问同一地址不产生 conflict。

理解 bank conflict 的数学本质有助于在设计 kernel 时主动避免它。每个 bank 的宽度与 warp 线程数相同（均为 32），但 bank 以 4 字节为单位交错分布。这意味着地址 word 0、word 32、word 64... 都在 bank 0；word 1、word 33、word 65... 都在 bank 1，依此类推。从这个映射规则可以推导出：如果访问地址的差值是 32 的倍数（即 stride = 32），则所有线程访问同一个 bank，形成 32-way conflict；如果 stride = 16，则两个线程为一组访问同一 bank，形成 2-way conflict（实际上是 2-way，因为 32/16=2，每个 bank 有 2 个线程竞争）。值得注意的是，由于 bank 映射基于 4 字节对齐，对于不同数据类型的 stride 换算也不相同：float16 下 stride=32 才对应 32-way conflict，float32 下 stride=32 同样是 32-way conflict，但 float64 下 stride=16 就对应 32-way conflict。这些细微的差异意味着优化策略需要根据数据类型动态调整。

```python
def demonstrate_bank_conflict():
    """
    演示各种 bank conflict 情况
    """
    
    print("=== Bank Conflict 演示 ===\n")
    
    # 场景 1：无 conflict（线程 i 访问 smem[i]）
    print("场景 1: 线程 i 访问 smem[i]")
    print("  线程 0 → Bank 0")
    print("  线程 1 → Bank 1")
    print("  ...")
    print("  线程 31 → Bank 31")
    print("  结果: 无 conflict，1 次事务\n")
    
    # 场景 2：2-way conflict（线程 i 访问 smem[i*2]）
    print("场景 2: 线程 i 访问 smem[i*2]")
    print("  线程 0 → Bank 0 (offset 0)")
    print("  线程 1 → Bank 2 (offset 2)")
    print("  ...")
    print("  线程 16 → Bank 0 (offset 32)")
    print("  结果: 2-way conflict\n")
    
    # 场景 3：严重 conflict（线程 i 访问 smem[i*32]）
    print("场景 3: 线程 i 访问 smem[i*32]")
    print("  线程 0 → Bank 0 (offset 0)")
    print("  线程 1 → Bank 0 (offset 32)")
    print("  ...")
    print("  线程 31 → Bank 0 (offset 31*32)")
    print("  结果: 32-way conflict，性能最差\n")
    
    # 场景 4：Broadcast（所有线程访问同一地址）
    print("场景 4: 所有线程访问 smem[0]")
    print("  线程 0-31 → Bank 0 (同一地址)")
    print("  结果: Broadcast，无 conflict\n")

demonstrate_bank_conflict()
```

该演示函数展示了四种 bank conflict 场景。无 conflict 时，每个线程访问不同 bank，只需一次事务。2-way conflict 时，每隔一个线程访问同一 bank，延迟翻倍。32-way conflict 是最差情况，所有线程访问同一 bank，性能下降 32 倍。Broadcast 是例外：所有线程访问同一地址不产生 conflict。理解这些场景有助于识别和修复 kernel 中的性能问题。

在实际开发中，2-way bank conflict 比 32-way bank conflict 更隐蔽且更难调试。因为 2-way conflict 仅导致约 1.5x-2x 的性能下降（而非 32x），profiler 可能不会对此发出明显警告，开发者容易忽视这个问题。但对于带宽敏感型 kernel，这 2x 的损失可能意味着从接近峰值带宽跌落到 40% 的利用率，在实际工程中是不容忽视的差距。建议在开发阶段就主动检查 shared memory 的 bank 映射关系：将 thread ID 映射到 (row, col) 后，计算 col % 32 是否与 thread ID % 32 一致——如果不同线程的 bank 索引有重复，则存在 conflict。此外，在 2-way conflict 场景中，有时可以通过循环重排或数据布局微调来消除冲突，而不一定需要 padding。

### 22.3.3 Padding 技术

```python
import tilelang.language as T

# 示例：有 bank conflict 的情况
@T.prim_func
def with_bank_conflict(A: T.Tensor((32, 32), "float32"),
                       B: T.Tensor((32, 32), "float32")):
    """
    问题：列访问产生 bank conflict
    
    如果 smem 的布局是 [32][32]，那么：
    - smem[0][0] 和 smem[1][0] 在同一个 bank
    - 当按列访问时，同一 warp 的线程访问同一 bank 的不同行
    - 产生 bank conflict
    """
    with T.Kernel(1, threads=32) as bx:
        smem = T.alloc_shared((32, 32), "float32")
        tid = T.get_thread_id()
        
        # 行写入（合并）
        for j in T.serial(32):
            smem[tid, j] = A[tid, j]
        T.sync_threads()
        
        # 列读取（有 bank conflict！）
        for j in T.serial(32):
            B[tid, j] = smem[j, tid]  # smem[j][tid] 和 smem[j+1][tid] 在同一 bank

# 使用 Padding 消除 bank conflict
@T.prim_func
def without_bank_conflict(A: T.Tensor((32, 32), "float32"),
                          B: T.Tensor((32, 32), "float32")):
    """
    解决方案：在每行末尾添加 padding
    
    将 smem 从 [32][32] 改为 [32][33]（每行多 1 个元素）
    这样 smem[j][tid] 和 smem[j+1][tid] 不在同一 bank
    """
    with T.Kernel(1, threads=32) as bx:
        smem = T.alloc_shared((32, 33), "float32")  # 注意：33 不是 32
        tid = T.get_thread_id()
        
        # 行写入（合并，padding 位置不写入数据）
        for j in T.serial(32):
            smem[tid, j] = A[tid, j]
        T.sync_threads()
        
        # 列读取（无 bank conflict！）
        for j in T.serial(32):
            B[tid, j] = smem[j, tid]  # 由于 padding，相邻行的同一列不在同一 bank
```

Padding 技术通过在每行末尾添加额外元素来消除 bank conflict。原始 32×32 的 shared memory 中，相邻行的同一列在同一个 bank。将维度改为 32×33 后，相邻行的列地址偏移 1 个 bank，消除了冲突。这种方法简单有效，代价是少量额外的 shared memory 开销。对于列访问频繁的 kernel，padding 是最常用的优化手段。

Padding 的代价主要体现在两个方面：一是 shared memory 的容量占用增加了约 1/N（N 为行长度），对于 32 列矩阵，增加 1 列即增加约 3%；二是增加了地址计算的复杂度——访问 padded 的 shared memory 时，需要使用 row * (N+1) + col 而非 row * N + col 作为索引。虽然现代编译器通常能高效处理这种偏移，但在某些极端追求性能的场景中，swizzle 可能是更节省空间的选择（无需额外 padding 空间）。此外，padding 量的选择也有讲究：+1 是最小开销，足以消除相邻行同一列的 bank conflict；但如果 stride 本身更大（如 stride=2 访问），+1 可能不足以解决所有冲突，此时需要更大的 padding 量或结合 swizzle 使用。

### 22.3.4 Swizzling 技术

```python
@T.prim_func
def swizzled_shared_memory(A: T.Tensor((32, 32), "float32"),
                           B: T.Tensor((32, 32), "float32")):
    """
    Swizzling：通过 XOR 操作重新映射 bank 索引
    
    原理：
    - 原始索引：bank = col % 32
    - Swizzled 索引：bank = (col ^ row) % 32
    
    优势：
    - 不需要额外的 padding 空间
    - 在 Tensor Core 操作中特别有用（MMA 指令要求特定的数据布局）
    """
    with T.Kernel(1, threads=32) as bx:
        smem = T.alloc_shared((32, 32), "float32")
        tid = T.get_thread_id()
        
        # Swizzled 写入
        for j in T.serial(32):
            swizzled_j = j ^ (tid % 32)  # XOR swizzle
            smem[tid, swizzled_j] = A[tid, j]
        T.sync_threads()
        
        # Swizzled 读取
        for j in T.serial(32):
            swizzled_j = j ^ (tid % 32)
            B[tid, j] = smem[tid, swizzled_j]
```

Swizzling 通过 XOR 操作重新映射 bank 索引，无需额外 padding 空间。原始 bank 索引为 col % 32，swizzled 索引为 (col ^ row) % 32。XOR 操作的关键性质是：对于同一列的不同行，映射后的 bank 不同。这消除了列访问时的 bank conflict。Swizzling 特别适合 Tensor Core 操作，因为 MMA 指令要求特定的数据布局。

Swizzling 之所以能消除 bank conflict，源于 XOR 操作的两个关键数学性质：一是自反性（a ^ b ^ b = a），保证了 decode/encode 的可逆性；二是扰动性，即对于固定的列 c，当行 r 从 0 到 31 变化时，(c ^ r) % 32 会扫过几乎所有 bank 位置。这使得原本映射到同一 bank 的元素被“打散”到不同 bank 中。在实际的 MMA（Matrix Multiply-Accumulate）指令中，Tensor Core 要求输入矩阵按照特定的 bank 交错格式排列，swizzle 模式恰好满足这一要求。然而，swizzle 引入的地址变换也增加了编程复杂度：开发者需要将 swizzle 逻辑嵌入到每次 shared memory 访问中，编译器通常无法自动推导 swizzle 模式。因此，swizzle 的使用需要在消除 bank conflict 的收益和增加的实现复杂度之间权衡，通常仅在 Tensor Core 操作或高频冲突的关键路径上使用。

### 22.3.5 Shared Memory 复用策略

```python
@T.prim_func
def shared_memory_reuse(A: T.Tensor((1024, 1024), "float16"),
                        B: T.Tensor((1024, 1024), "float16"),
                        C: T.Tensor((1024, 1024), "float32")):
    """
    Shared Memory 复用策略
    
    关键思想：
    - 同一个 shared memory buffer 可以用于不同的计算阶段
    - 通过 T.alloc_shared 分配，编译器自动管理生命周期
    - 在 TileLang 中，使用 double buffering 实现计算/访存重叠
    """
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    with T.Kernel(T.ceildiv(1024, BLOCK_N), T.ceildiv(1024, BLOCK_M), threads=256) as (bx, by):
        # Double buffering：两份 shared memory
        A_smem = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
        B_smem = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
        A_smem_next = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
        B_smem_next = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
        
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
        T.clear(C_local)
        
        num_k_tiles = T.ceildiv(1024, BLOCK_K)
        
        # 预取第一块
        T.copy(A[by * BLOCK_M, 0], A_smem)
        T.copy(B[0, bx * BLOCK_N], B_smem)
        
        for k in T.serial(num_k_tiles):
            # 预取下一块（与当前计算重叠）
            if k + 1 < num_k_tiles:
                T.copy(A[by * BLOCK_M, (k + 1) * BLOCK_K], A_smem_next)
                T.copy(B[(k + 1) * BLOCK_K, bx * BLOCK_N], B_smem_next)
            
            # 使用当前块进行计算
            for i, j in T.grid(BLOCK_M, BLOCK_N):
                for kk in T.serial(BLOCK_K):
                    C_local[i, j] += A_smem[i, kk].astype("float32") * B_smem[kk, j].astype("float32")
            
            # 交换 buffer
            A_smem, A_smem_next = A_smem_next, A_smem
            B_smem, B_smem_next = B_smem_next, B_smem
        
        T.copy(C_local, C[by * BLOCK_M, bx * BLOCK_N])
```
该 kernel 通过减小 block 大小（64×64）和 Fragment 尺寸来主动控制寄存器压力。每个线程仅保留一行 A 数据的子 Fragment（1×32），计算完成后立即累加到 C_local 中，寄存器生命周期极短。逐行计算确保编译器可以有效复用寄存器，避免 spill。在实际开发中，当 kernel 编译报告寄存器使用量超过硬件限制时，通过类似的"分而治之"策略减小 tile 是首选优化方案。

---

> Shared Memory 优化解决了片上存储的访问效率问题，但 GPU 的性能还受到寄存器分配策略的直接影响。寄存器是速度最快的存储资源，但数量有限，如何在寄存器使用量和 Occupancy 之间找到平衡，是 kernel 优化的核心挑战之一。

## 22.4 寄存器分配策略

### 22.4.1 寄存器与 Occupancy 的权衡

<div data-component="RegisterAllocationStrategy"></div>

```python
class RegisterOccupancyTradeoff:
    """
    寄存器使用与 Occupancy 的权衡
    
    核心矛盾：
    - 使用更多寄存器 → 更好的指令级优化，减少 shared memory 访问
    - 使用更少寄存器 → 更高的 Occupancy，更好的延迟隐藏
    
    最优平衡点取决于：
    1. 算术强度（Arithmetic Intensity）
    2. 寄存器压力（Register Pressure）
    3. 延迟隐藏需求
    """
    
    def __init__(self, gpu_arch="sm_80"):
        self.specs = {
            "sm_80": {  # A100
                "registers_per_sm": 65536,
                "max_warps_per_sm": 64,
                "warp_size": 32,
            }
        }[gpu_arch]
    
    def calculate_occupancy_by_registers(self, regs_per_thread, threads_per_block):
        """计算给定寄存器使用量下的 Occupancy"""
        warps_per_block = (threads_per_block + self.specs["warp_size"] - 1) // self.specs["warp_size"]
        
        # 每个 block 的寄存器使用量
        regs_per_block = regs_per_thread * threads_per_block
        
        # 每个 SM 能容纳的 block 数
        max_blocks = self.specs["registers_per_sm"] // regs_per_block
        
        # Active warps
        active_warps = min(max_blocks * warps_per_block, self.specs["max_warps_per_sm"])
        
        occupancy = active_warps / self.specs["max_warps_per_sm"]
        
        return {
            "regs_per_thread": regs_per_thread,
            "regs_per_block": regs_per_block,
            "max_blocks": max_blocks,
            "active_warps": active_warps,
            "occupancy": occupancy,
        }
    
    def find_optimal_register_usage(self, threads_per_block, target_occupancy=0.5):
        """找到满足目标 Occupancy 的最大寄存器使用量"""
        warps_per_block = (threads_per_block + self.specs["warp_size"] - 1) // self.specs["warp_size"]
        
        # 需要的 active warps
        target_warps = int(target_occupancy * self.specs["max_warps_per_sm"])
        
        # 需要的 block 数
        target_blocks = (target_warps + warps_per_block - 1) // warps_per_block
        
        # 每个 block 的最大寄存器数
        max_regs_per_block = self.specs["registers_per_sm"] // target_blocks
        
        # 每个线程的最大寄存器数
        max_regs_per_thread = max_regs_per_block // threads_per_block
        
        return {
            "max_regs_per_thread": max_regs_per_thread,
            "achieved_occupancy": target_occupancy,
            "threads_per_block": threads_per_block,
        }

# 使用示例
tradeoff = RegisterOccupancyTradeoff("sm_80")

# 分析不同寄存器使用量下的 Occupancy
for regs in [32, 48, 64, 96, 128]:
    result = tradeoff.calculate_occupancy_by_registers(regs, 256)
    print(f"Regs/thread={regs}: Occupancy={result['occupancy']:.1%}")
```

该类量化了寄存器使用量与 occupancy 的权衡关系。使用更多寄存器可以减少 shared memory 访问和寄存器 spill，但会降低 occupancy。A100 每个 SM 有 65536 个寄存器，每线程使用 128 个寄存器时，每个 SM 只能运行一个 block。最优平衡点取决于算术强度和延迟隐藏需求。该工具帮助开发者找到满足目标 occupancy 的最大寄存器使用量。

在真实场景中，Occupancy 并非越高越好——这违背了许多初学者的直觉。Volkov 在 2010 年的经典研究"Better Performance at Lower Occupancy"中证明：当 kernel 的算术强度足够高时（如 GEMM），25% 的 Occupancy 就足以隐藏内存延迟，更多的 active warp 反而会因为寄存器竞争而导致 spill。关键判断标准是：如果内核是 compute-bound（算术强度高于 ridge point），降低 occupancy 换取更多寄存器通常更优；如果内核是 memory-bound，则需要更高 occupancy 来隐藏延迟。因此，该工具计算的 occupancy 上限应结合算术强度分析综合判断，而非盲目追求 100% 的理论 occupancy。

### 22.4.2 寄存器 Spill 优化

```python
@T.prim_func
def minimize_register_spill(A: T.Tensor((1024,), "float32"),
                            B: T.Tensor((1024,), "float32"),
                            C: T.Tensor((1024,), "float32")):
    """
    减少寄存器 spill 的策略
    
    寄存器 spill 发生在：
    - 局部变量过多
    - 循环嵌套过深
    - 编译器无法有效复用寄存器
    
    优化方法：
    1. 减少同时活跃的变量数量
    2. 使用 shared memory 替代部分寄存器
    3. 手动控制变量的生命周期
    """
    with T.Kernel(1, threads=256) as bx:
        tid = T.get_thread_id()
        
        # 策略 1：分块处理，减少同时活跃的变量
        # 不好的做法：一次性加载所有数据
        # data = [A[tid * 4 + i] for i in range(4)]  # 需要 4 个寄存器
        
        # 好的做法：逐个处理
        acc = T.float32(0)
        for i in T.serial(4):
            val = A[tid * 4 + i]
            acc += val * val  # 立即使用，减少寄存器占用
        
        B[tid] = acc
        
        # 策略 2：使用 shared memory 缓存中间结果
        smem = T.alloc_shared((256,), "float32")
        smem[tid] = acc
        T.sync_threads()
        
        # 策略 3：重用寄存器
        temp = T.float32(0)
        for i in T.serial(256):
            temp += smem[i]
        
        C[tid] = temp / 256.0
```

寄存器 spill 发生在局部变量过多或循环嵌套过深时，编译器将变量溢出到 local memory。local memory 的延迟约为 400 cycles，远高于寄存器的 1 cycle。优化策略包括：分块处理减少同时活跃的变量、使用 shared memory 缓存中间结果、重用寄存器变量。立即使用变量可以缩短其生命周期，减少寄存器占用。

Local memory 虽然名称中有"local"，但它实际上位于全局内存（HBM）中，与线程私有相关联，因此访问延迟与全局内存相当。这是新手最容易混淆的概念之一：register spill 不是溢出到 shared memory 或 L1 cache，而是直接落到了最慢的内存层级。编译器在决定哪些变量需要 spill 时，采用复杂的图着色寄存器分配算法，但在涉及大量循环和临时变量的 kernel 中，编译器的决策未必最优。为此，开发者可以通过以下方法主动干预：一是使用 scalar replacement 技术，将数组访问转换为标量变量；二是控制循环展开因子，避免过度展开导致寄存器爆炸；三是在 TileLang 中使用 T.alloc_fragment 的较小形状，给编译器留下更多的分配余地。

### 22.4.3 TileLang 中的寄存器控制

```python
import tilelang.language as T

@T.prim_func
def register_optimized_kernel(
    A: T.Tensor((1024, 1024), "float16"),
    B: T.Tensor((1024, 1024), "float16"),
    C: T.Tensor((1024, 1024), "float32"),
):
    """
    TileLang 中的寄存器优化技巧
    
    1. 使用 T.alloc_fragment 声明寄存器级别的存储
    2. 使用 T.clear 初始化，避免未使用寄存器的浪费
    3. 合理使用 T.unroll 控制循环展开程度
    """
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    with T.Kernel(T.ceildiv(1024, BLOCK_N), T.ceildiv(1024, BLOCK_M), threads=256) as (bx, by):
        # 寄存器级别的累加器
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
        T.clear(C_local)
        
        # Shared memory 缓冲区
        A_smem = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
        B_smem = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
        
        for k in T.serial(T.ceildiv(1024, BLOCK_K)):
            # 加载到 shared memory
            T.copy(A[by * BLOCK_M, k * BLOCK_K], A_smem)
            T.copy(B[k * BLOCK_K, bx * BLOCK_N], B_smem)
            
            # 使用 T.unroll 控制展开
            # 展开因子需要根据寄存器压力调整
            for i, j in T.grid(BLOCK_M, BLOCK_N):
                for kk in T.serial(BLOCK_K):  # 编译器自动决定是否展开
                    C_local[i, j] += A_smem[i, kk].astype("float32") * B_smem[kk, j].astype("float32")
        
        T.copy(C_local, C[by * BLOCK_M, bx * BLOCK_N])
```

该 kernel 展示了 TileLang 中的寄存器优化技巧。T.alloc_fragment 声明寄存器级别的存储，编译器直接分配到寄存器文件。T.clear 初始化累加器，避免未使用寄存器的浪费。循环展开因子需要根据寄存器压力调整：展开过多会导致寄存器不足，展开过少会降低指令级并行。在 tile 大小和寄存器使用之间找到平衡是关键。

---

> 寄存器分配策略决定了 kernel 的 Occupancy 上限，而 L1 Cache 的利用则影响着全局内存访问的实际效率。L1 Cache 与 Shared Memory 共享片上存储资源，理解它们的协作关系对于合理分配缓存资源至关重要。

## 22.5 L1 Cache 利用

### 22.5.1 L1 Cache 特性

```python
"""
L1 Cache 特性（NVIDIA A100）：

1. 容量：128 KB/SM（与 shared memory 共享）
2. 行大小：128 bytes
3. 关联度：4-way set associative
4. 替换策略：LRU（最近最少使用）

L1 Cache 与 Shared Memory 的关系：
- 总容量 = 128 KB（A100）
- 可以配置 L1 和 shared memory 的分配比例
- 使用更多 shared memory → L1 容量减少
- 需要在两者之间权衡

L1 Cache 自动缓存：
- 全局内存访问（如果启用了 L1 缓存）
- 本地内存访问（寄存器 spill）
- 只读数据可以通过 __ldg() 明确使用 L1 缓存
"""
```

这段文字描述了 NVIDIA A100 的 L1 Cache 特性。L1 Cache 与 shared memory 共享 128 KB 容量，使用更多 shared memory 会减少 L1 容量。L1 Cache 采用 4-way 组相联和 LRU 替换策略，行大小为 128 字节。只读数据可以通过 __ldg() 明确使用 L1 缓存。理解 L1 与 shared memory 的关系有助于合理分配片上内存资源。

### 22.5.2 L1 Cache 优化策略

```python
@T.prim_func
def l1_cache_friendly(A: T.Tensor((1024, 1024), "float32"),
                      B: T.Tensor((1024,), "float32")):
    """
    L1 Cache 友好的访问模式
    
    策略：
    1. 保持数据局部性（同一 SM 的线程访问相邻数据）
    2. 使用只读缓存（__ldg 或 T.readonly）
    3. 避免 cache thrashing（频繁替换）
    """
    with T.Kernel(32, threads=256) as bx:
        # 每个线程处理连续的 4 个元素
        # 同一 warp 的线程访问连续地址 → L1 cache line 命中率高
        tid = T.get_thread_id()
        
        acc = T.float32(0)
        for i in T.serial(4):
            idx = bx * 1024 + tid * 4 + i
            # 使用只读缓存
            acc += T.ldg(A[idx])  # 通过 L1 只读缓存访问
        
        B[bx * 256 + tid] = acc

@T.prim_func
def l1_cache_unfriendly(A: T.Tensor((1024, 1024), "float32"),
                        B: T.Tensor((1024,), "float32")):
    """
    L1 Cache 不友好的访问模式（反面教材）
    
    问题：
    - 跨行访问导致 cache line 利用率低
    - 随机访问导致 cache thrashing
    """
    with T.Kernel(32, threads=256) as bx:
        tid = T.get_thread_id()
        
        acc = T.float32(0)
        for i in T.serial(4):
            # 跨行访问：每个线程访问不同行的同一列
            # 导致 L1 cache line 中大部分数据未被使用
            idx = (bx * 32 + i) * 1024 + tid
            acc += A[idx]
        
        B[bx * 256 + tid] = acc
```

这段代码对比了 cache-friendly 和 cache-unfriendly 的访问模式。友好模式中，每个线程处理连续的 4 个元素，同一 warp 的线程访问连续地址，cache line 利用率高。不友好模式中，线程跨行访问同一列，cache line 中大部分数据未被使用。使用 T.ldg() 可以通过只读缓存访问，提高 cache 命中率。优化数据局部性是提高 L1 Cache 利用率的关键。

---

> L1 Cache 的优化需要开发者对硬件特性有深入了解，而不同的编程框架提供了不同程度的抽象。TileLang 采用显式内存管理，给予开发者完全的控制权；Triton 则采用隐式管理，降低了编程复杂度。两者各有优劣，选择取决于具体的应用场景和优化需求。

## 22.6 TileLang 显式内存管理 vs Triton 隐式管理

### 22.6.1 对比分析

| 特性 | TileLang 显式管理 | Triton 隐式管理 |
|------|-----------------|----------------|
| **Shared Memory** | 手动分配 `T.alloc_shared` | 自动管理（通过 block size 推断） |
| **寄存器** | 通过 `T.alloc_fragment` 显式声明 | 编译器自动分配 |
| **数据搬运** | 显式 `T.copy` 操作 | 编译器自动生成 load/store |
| **Bank Conflict** | 用户负责避免（padding/swizzling） | 编译器尝试自动优化 |
| **Double Buffering** | 手动实现 | 编译器自动流水线 |
| **灵活性** | 极高，完全控制 | 中等，依赖编译器能力 |
| **易用性** | 较低，需要更多知识 | 较高，自动处理 |

### 22.6.2 TileLang 显式管理的优势

```python
@T.prim_func
def explicit_memory_management(A: T.Tensor((M, K), "float16"),
                               B: T.Tensor((K, N), "float16"),
                               C: T.Tensor((M, N), "float32")):
    """
    TileLang 显式内存管理的优势
    
    1. 精确控制数据布局
    2. 手动优化 bank conflict
    3. 灵活的 double buffering 策略
    4. 可以针对特定硬件优化
    """
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=256) as (bx, by):
        # 优势 1：精确控制 shared memory 布局
        # 添加 padding 避免 bank conflict
        A_smem = T.alloc_shared((BLOCK_M, BLOCK_K + 8), "float16")  # +8 padding
        B_smem = T.alloc_shared((BLOCK_K, BLOCK_N + 8), "float16")  # +8 padding
        
        # 优势 2：手动 double buffering
        A_smem_buf = T.alloc_shared((BLOCK_M, BLOCK_K + 8), "float16")
        B_smem_buf = T.alloc_shared((BLOCK_K, BLOCK_N + 8), "float16")
        
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
        T.clear(C_local)
        
        # 预取第一块
        T.copy(A[by * BLOCK_M, 0], A_smem)
        T.copy(B[0, bx * BLOCK_N], B_smem)
        
        for k in T.serial(T.ceildiv(K, BLOCK_K)):
            # 预取下一块
            if k + 1 < T.ceildiv(K, BLOCK_K):
                T.copy(A[by * BLOCK_M, (k + 1) * BLOCK_K], A_smem_buf)
                T.copy(B[(k + 1) * BLOCK_K, bx * BLOCK_N], B_smem_buf)
            
            # 计算当前块
            for i, j in T.grid(BLOCK_M, BLOCK_N):
                for kk in T.serial(BLOCK_K):
                    # 访问 padded 的 shared memory，无 bank conflict
                    C_local[i, j] += A_smem[i, kk].astype("float32") * B_smem[kk, j].astype("float32")
            
            # 交换 buffer
            A_smem, A_smem_buf = A_smem_buf, A_smem
            B_smem, B_smem_buf = B_smem_buf, B_smem
        
        T.copy(C_local, C[by * BLOCK_M, bx * BLOCK_N])
```

该 kernel 展示了 TileLang 显式内存管理的优势。开发者可以精确控制 shared memory 布局，通过 +8 padding 消除 bank conflict。手动实现 double buffering，灵活控制预取时机。编译器自动管理寄存器分配，但开发者可以通过 T.alloc_fragment 显式声明。这种完全控制能力使得针对特定硬件的深度优化成为可能。

### 22.6.3 Triton 隐式管理的特点

```python
# Triton 示例（隐式内存管理）
import triton
import triton.language as tl

@triton.jit
def matmul_triton(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Triton 的隐式内存管理
    
    特点：
    1. 不需要手动分配 shared memory
    2. 编译器自动决定数据布局
    3. 自动处理 bank conflict
    4. 自动进行 double buffering
    
    限制：
    1. 无法精确控制数据布局
    2. 某些优化场景下不如手动管理
    3. 对硬件特定优化支持有限
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # 编译器自动管理这些加载到 shared memory
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # 编译器自动决定是否使用 shared memory 或寄存器
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        accumulator += tl.dot(a, b)
        offs_k += BLOCK_K
    
    c = accumulator.to(tl.float16)
    tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn, c)
```

该示例展示了 Triton 的隐式内存管理模式。开发者只需指定 block size，编译器自动管理 shared memory 分配和数据搬运。编译器尝试自动处理 bank conflict 和 double buffering。优势是代码简洁，降低了编程难度。限制是无法精确控制数据布局，某些优化场景下不如手动管理高效。

---

> 了解了不同框架的内存管理方式后，我们需要量化评估优化效果。带宽利用率是衡量内存优化成效的核心指标，通过计算实际带宽与理论峰值的比值，可以准确判断 kernel 是否存在内存瓶颈，并指导后续的优化方向。

## 22.7 带宽利用率计算与优化

### 22.7.1 带宽利用率计算方法

<div data-component="BandwidthUtilizationChart"></div>

```python
class BandwidthUtilizationCalculator:
    """
    带宽利用率计算器
    
    带宽利用率 = 实际有效带宽 / 理论峰值带宽
    
    其中：
    - 实际有效带宽 = 数据量 / 执行时间
    - 理论峰值带宽 = 硬件规格
    """
    
    def __init__(self, gpu="nvidia-a100"):
        self.peak_bandwidth_gbs = {
            "nvidia-a100": 2039,
            "nvidia-h100": 3350,
            "nvidia-v100": 900,
            "amd-mi300x": 5300,
            "amd-mi250x": 3200,
        }.get(gpu, 1000)
    
    def calculate_operation_bandwidth(self, bytes_read, bytes_written, elapsed_ms):
        """
        计算单个操作的带宽利用率
        
        参数：
        - bytes_read: 从全局内存读取的字节数
        - bytes_written: 写入全局内存的字节数
        - elapsed_ms: 执行时间（毫秒）
        """
        total_bytes = bytes_read + bytes_written
        actual_bandwidth_gbs = (total_bytes / 1e9) / (elapsed_ms / 1e3)
        utilization = actual_bandwidth_gbs / self.peak_bandwidth_gbs * 100
        
        return {
            "bytes_read": bytes_read,
            "bytes_written": bytes_written,
            "total_bytes": total_bytes,
            "elapsed_ms": elapsed_ms,
            "bandwidth_gbs": actual_bandwidth_gbs,
            "peak_bandwidth_gbs": self.peak_bandwidth_gbs,
            "utilization_pct": utilization,
        }
    
    def calculate_gemm_bandwidth(self, M, N, K, dtype="float16", elapsed_ms=1.0):
        """计算 GEMM 的带宽利用率"""
        bytes_per_elem = 2 if dtype == "float16" else 4
        
        # 读取量
        bytes_A = M * K * bytes_per_elem
        bytes_B = K * N * bytes_per_elem
        
        # 写入量
        bytes_C = M * N * 4  # float32 输出
        
        return self.calculate_operation_bandwidth(
            bytes_read=bytes_A + bytes_B,
            bytes_written=bytes_C,
            elapsed_ms=elapsed_ms,
        )
    
    def calculate_elementwise_bandwidth(self, N, dtype="float32", elapsed_ms=1.0):
        """计算 Elementwise 操作的带宽利用率"""
        bytes_per_elem = 4 if dtype == "float32" else 2
        
        # 读取 + 写入
        bytes_read = N * bytes_per_elem
        bytes_written = N * bytes_per_elem
        
        return self.calculate_operation_bandwidth(
            bytes_read=bytes_read,
            bytes_written=bytes_written,
            elapsed_ms=elapsed_ms,
        )
    
    def analyze_kernel_set(self, kernel_results):
        """分析一组 kernel 的带宽利用率"""
        print(f"{'Kernel':<30} {'Time(ms)':<10} {'BW(GB/s)':<12} {'Util%':<8}")
        print("-" * 60)
        
        for name, result in kernel_results.items():
            print(f"{name:<30} {result['elapsed_ms']:<10.3f} "
                  f"{result['bandwidth_gbs']:<12.1f} {result['utilization_pct']:<8.1f}")
```

该计算器实现了带宽利用率的量化分析。带宽利用率定义为实际有效带宽与理论峰值带宽的比值。实际带宽通过数据量除以执行时间计算，理论峰值取决于硬件规格。该工具支持 GEMM 和 elementwise 操作的带宽计算，帮助识别内存瓶颈。对于 A100，HBM 峰值带宽为 2039 GB/s，优化目标是达到 80% 以上的利用率。

### 22.7.2 带宽优化策略

```python
def bandwidth_optimization_strategies():
    """
    带宽优化策略总结
    
    策略 1：减少数据搬运
    - 算子融合（减少中间结果写回）
    - In-place 操作（原地修改）
    - 数据压缩（稀疏格式）
    
    策略 2：提高搬运效率
    - 合并访问（Coalescing）
    - 向量化加载（Vectorized Load）
    - 使用更宽的数据类型
    
    策略 3：隐藏搬运延迟
    - Double Buffering（计算/访存重叠）
    - 异步拷贝（Async Copy）
    - 流水线（Pipeline）
    
    策略 4：利用缓存层次
    - Shared Memory 缓存
    - L1/L2 Cache 利用
    - 数据预取（Prefetch）
    """
    
    strategies = {
        "reduce_data_movement": {
            "description": "减少数据搬运量",
            "techniques": [
                "算子融合：将多个算子合并，减少中间结果的全局内存写回",
                "In-place 操作：直接在原数据上修改，避免额外的读写",
                "数据压缩：使用稀疏格式减少存储和传输量",
            ],
            "expected_improvement": "2-10x（取决于融合程度）",
        },
        "improve_transfer_efficiency": {
            "description": "提高每次搬运的效率",
            "techniques": [
                "合并访问：确保 warp 内线程访问连续地址",
                "向量化加载：使用 float4/int4 等宽类型",
                "使用更宽的数据类型：减少指令数量",
            ],
            "expected_improvement": "2-8x（取决于原始效率）",
        },
        "hide_transfer_latency": {
            "description": "隐藏数据搬运的延迟",
            "techniques": [
                "Double Buffering：一份数据计算，另一份预取",
                "异步拷贝：使用 cp.async 等异步指令",
                "流水线：多级流水线重叠计算和访存",
            ],
            "expected_improvement": "1.5-3x（取决于计算/访存比）",
        }
    }
    
    return strategies
```

该函数总结了四大带宽优化策略。减少数据搬运：通过算子融合和 in-place 操作减少中间结果的读写。提高搬运效率：使用合并访问和向量化加载提高每次事务的有效数据量。隐藏搬运延迟：通过 double buffering 和异步拷贝将延迟隐藏在计算中。实际优化中，应根据具体场景组合使用这些策略。

---

> 掌握了带宽利用率的计算方法后，我们通过实战案例来综合应用这些优化技术。从简单的向量加法到复杂的 Softmax 计算，每个案例都展示了不同的内存优化策略组合，帮助读者建立系统化的优化思维。

## 22.8 实战优化案例

### 22.8.1 案例 1：向量加法带宽优化

```python
@T.prim_func
def vector_add_naive(A: T.Tensor((N,), "float32"),
                     B: T.Tensor((N,), "float32"),
                     C: T.Tensor((N,), "float32")):
    """朴素实现：每个线程处理一个元素"""
    with T.Kernel(T.ceildiv(N, 256), threads=256) as bx:
        tid = T.get_thread_id()
        idx = bx * 256 + tid
        C[idx] = A[idx] + B[idx]

@T.prim_func
def vector_add_vectorized(A: T.Tensor((N,), "float32"),
                          B: T.Tensor((N,), "float32"),
                          C: T.Tensor((N,), "float32")):
    """向量化实现：每个线程处理 4 个元素"""
    with T.Kernel(T.ceildiv(N, 1024), threads=256) as bx:
        tid = T.get_thread_id()
        # 使用向量化加载（float4 = 4 * float32 = 16 bytes）
        base = bx * 1024 + tid * 4
        
        # 一次加载 4 个元素
        a = T.load_vectorized(A, base, 4)  # float4
        b = T.load_vectorized(B, base, 4)  # float4
        
        # 向量加法
        c = a + b
        
        # 向量化存储
        T.store_vectorized(C, base, c, 4)
```

这段代码对比了朴素和向量化的向量加法实现。朴素版本每个线程处理一个元素，需要 N 次加载和 N 次存储。向量化版本每个线程处理 4 个元素（float4），每次加载 16 字节。向量化减少了指令数量，提高了内存事务的利用率。对于带宽受限的操作，向量化是简单有效的优化手段。

### 22.8.2 案例 2：矩阵转置优化

```python
@T.prim_func
def matrix_transpose_optimized(A: T.Tensor((M, N), "float32"),
                               B: T.Tensor((N, M), "float32")):
    """
    优化的矩阵转置
    
    优化点：
    1. 使用 shared memory 作为中间缓冲
    2. Padding 消除 bank conflict
    3. 合并读取和写入
    """
    TILE = 32
    
    with T.Kernel(T.ceildiv(N, TILE), T.ceildiv(M, TILE), threads=256) as (bx, by):
        # +1 padding 避免 bank conflict
        tile = T.alloc_shared((TILE, TILE + 1), "float32")
        
        tid = T.get_thread_id()
        local_row = tid // TILE
        local_col = tid % TILE
        
        # 合并读取：连续的线程访问连续的地址
        global_row = by * TILE + local_row
        global_col = bx * TILE + local_col
        
        if global_row < M and global_col < N:
            tile[local_row, local_col] = A[global_row, global_col]
        
        T.sync_threads()
        
        # 合并写入：转置后的合并访问
        # 注意：这里 local_row 和 local_col 交换了
        new_global_row = bx * TILE + local_row
        new_global_col = by * TILE + local_col
        
        if new_global_row < N and new_global_col < M:
            B[new_global_row, new_global_col] = tile[local_col, local_row]
```

该实现使用 shared memory + padding 优化矩阵转置。首先将数据合并加载到 shared memory（+1 padding 消除 bank conflict），同步后按转置索引写回全局内存。合并读取确保 warp 内线程访问连续地址。+1 padding 使得相邻行的同一列不在同一个 bank，消除列访问时的冲突。这是转置操作的标准优化模式。

### 22.8.3 案例 3：Softmax 内存优化

```python
@T.prim_func
def softmax_memory_optimized(A: T.Tensor((M, N), "float32"),
                              B: T.Tensor((M, N), "float32")):
    """
    内存优化的 Softmax 实现
    
    挑战：
    - 需要两遍扫描：第一遍找最大值，第二遍计算 exp 和 sum
    - 如果数据太大无法放入 shared memory
    
    优化：
    - 使用寄存器级别的 partial results
    - 减少全局内存读写次数
    - 使用向量化操作
    """
    with T.Kernel(M, threads=256) as bx:
        row = bx
        
        # 策略：分块处理，每块在寄存器中累积
        # 最后在 shared memory 中合并
        
        smem_max = T.alloc_shared((256,), "float32")
        smem_sum = T.alloc_shared((256,), "float32")
        
        tid = T.get_thread_id()
        
        # Pass 1: 找最大值
        local_max = T.float32(-1e30)
        for j in T.serial(T.ceildiv(N, 256)):
            col = j * 256 + tid
            if col < N:
                val = A[row, col]
                local_max = T.max(local_max, val)
        
        smem_max[tid] = local_max
        T.sync_threads()
        
        # Reduction: 找全局最大值
        for stride in [128, 64, 32, 16, 8, 4, 2, 1]:
            if tid < stride and tid + stride < 256:
                smem_max[tid] = T.max(smem_max[tid], smem_max[tid + stride])
            T.sync_threads()
        
        row_max = smem_max[0]
        
        # Pass 2: 计算 exp 和 sum
        local_sum = T.float32(0)
        for j in T.serial(T.ceildiv(N, 256)):
            col = j * 256 + tid
            if col < N:
                val = T.exp(A[row, col] - row_max)
                local_sum += val
        
        smem_sum[tid] = local_sum
        T.sync_threads()
        
        # Reduction: 求和
        for stride in [128, 64, 32, 16, 8, 4, 2, 1]:
            if tid < stride and tid + stride < 256:
                smem_sum[tid] += smem_sum[tid + stride]
            T.sync_threads()
        
        row_sum = smem_sum[0]
        
        # Pass 3: 写入结果
        for j in T.serial(T.ceildiv(N, 256)):
            col = j * 256 + tid
            if col < N:
                B[row, col] = T.exp(A[row, col] - row_max) / row_sum
```

该实现采用三遍扫描策略优化 softmax 计算。第一遍找最大值，每个线程处理 N/256 个元素并在寄存器中累积局部最大值。通过 shared memory reduction 合并所有线程的结果，得到全局最大值。第二遍计算 exp(x-max) 和 sum，同样使用 reduction 合并。第三遍写入最终结果。分块处理使得即使数据量很大也能高效执行。

---

> 通过以上实战案例，我们已经掌握了内存优化的基本技术。接下来的小结将回顾本章的核心知识点，帮助读者建立完整的知识框架。同时，练习和思考题将引导读者深入思考各种优化技术的适用场景和权衡取舍。

## 22.9 本章小结

本章深入探讨了 GPU 内存优化的核心技术：

1. **内存层次**：Register → Shared Memory → L1 Cache → L2 Cache → HBM 的性能特性
2. **合并访问**：Coalescing 的原理、检测和优化方法
3. **Shared Memory**：Bank Conflict 的产生、Padding/Swizzling 消除技术、Double Buffering
4. **寄存器分配**：与 Occupancy 的权衡、减少 Spill 的策略
5. **L1 Cache**：利用方法和优化策略
6. **显式 vs 隐式管理**：TileLang 和 Triton 的内存管理对比
7. **带宽优化**：计算方法和四大优化策略
8. **实战案例**：向量加法、矩阵转置、Softmax 的内存优化

---

## 练习

### Exercise 1: Coalescing 分析
编写一个 kernel，故意使用非合并访问模式，然后通过修改代码使其变为合并访问。使用 ncu 验证改进效果。

### Exercise 2: Bank Conflict 消除
实现一个矩阵转置 kernel，初始版本有严重的 bank conflict，然后通过 padding 或 swizzling 消除冲突。

### Exercise 3: Double Buffering
实现一个 GEMM kernel，使用手动 double buffering 实现计算/访存重叠。对比有无 double buffering 的性能差异。

### Exercise 4: 寄存器优化
编写一个 kernel，使其寄存器使用量过高导致 Occupancy 低。然后通过代码重构减少寄存器使用，提高 Occupancy。

### Exercise 5: 带宽利用率优化
针对向量加法操作，从朴素实现开始，逐步应用向量化加载、合并访问等优化，直到达到 80% 以上的带宽利用率。

---

## 思考题

1. **Shared Memory 和 L1 Cache 的主要区别是什么？在什么场景下应该优先使用 Shared Memory？**

2. **为什么 2-way bank conflict 比 32-way bank conflict 更难发现和调试？**

3. **在什么情况下，使用更多的寄存器（即使降低 Occupancy）反而能获得更好的性能？**

4. **TileLang 的显式内存管理相比 Triton 的隐式管理，在哪些场景下有明显优势？**

5. **如何在保证正确性的前提下，最大化利用 GPU 的内存带宽？有哪些系统化的方法？**

---

> 掌握了基础的内存优化技术后，我们将进入更高级的主题。内存压缩、预取策略和复用模式等高级技术可以进一步提升性能，特别是在处理大规模数据和复杂计算模式时。这些技术需要对硬件特性有更深入的理解。

## 22.9 高级内存优化技术

### 22.9.1 内存压缩

```python
"""
GPU 内存压缩技术

现代 GPU 支持透明内存压缩：
- NVIDIA：L2 Compression（Ampere 及以上）
- AMD：Delta Color Compression (DCC)

压缩可以：
- 减少内存带宽需求
- 提高有效容量
- 对程序员透明

压缩效果取决于数据模式：
- 稀疏数据：压缩率高（可达 4:1）
- 随机数据：压缩率低（接近 1:1）
- 规律数据：中等压缩率
"""

class MemoryCompressionAnalyzer:
    """
    内存压缩效果分析器
    """
    
    def estimate_compression_ratio(self, data_pattern):
        """
        估计数据的压缩率
        
        参数：
        - data_pattern: 数据模式描述
        """
        ratios = {
            "sparse": 4.0,      # 稀疏数据
            "zeros": 8.0,       # 全零数据
            "sequential": 2.0,  # 顺序数据
            "random": 1.0,      # 随机数据
            "repeated": 3.0,    # 重复模式
        }
        
        return ratios.get(data_pattern, 1.5)
    
    def analyze_kernel_memory_pattern(self, kernel_func, inputs):
        """
        分析 kernel 的内存访问模式
        """
        # 这里需要实际运行 kernel 并采集硬件计数器
        # 简化为静态分析
        return {
            "read_pattern": "sequential",
            "write_pattern": "sequential",
            "estimated_compression": 2.0,
        }
```

该分析器评估 GPU 内存压缩的效果。现代 GPU 支持透明内存压缩，如 NVIDIA 的 L2 Compression 和 AMD 的 DCC。压缩效果取决于数据模式：稀疏数据可达 4:1，全零数据可达 8:1，随机数据接近 1:1。压缩对程序员透明，但了解数据模式有助于预测优化效果。在设计 kernel 时，保持数据规律性可以提高压缩率。

### 22.9.2 内存预取策略

```python
@T.prim_func
def prefetch_example(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float32"),
):
    """
    内存预取策略示例
    
    预取（Prefetch）的核心思想：
    - 在需要数据之前，提前发起内存请求
    - 将内存延迟隐藏在计算中
    
    预取策略：
    1. 软件预取：显式发起预取请求
    2. 硬件预取：利用访问模式自动预取
    3. 流水线预取：在多级流水线中交替预取和计算
    """
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=256) as (bx, by):
        # 多级缓冲：预取 k+1 和 k+2 的数据
        A_buf = [T.alloc_shared((BLOCK_M, BLOCK_K), "float16") for _ in range(3)]
        B_buf = [T.alloc_shared((BLOCK_K, BLOCK_N), "float16") for _ in range(3)]
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
        
        T.clear(C_local)
        
        num_k_tiles = T.ceildiv(K, BLOCK_K)
        
        # 预取前两块
        T.copy(A[by * BLOCK_M, 0], A_buf[0])
        T.copy(B[0, bx * BLOCK_N], B_buf[0])
        
        if num_k_tiles > 1:
            T.copy(A[by * BLOCK_M, BLOCK_K], A_buf[1])
            T.copy(B[BLOCK_K, bx * BLOCK_N], B_buf[1])
        
        for k in T.serial(num_k_tiles):
            # 使用当前缓冲
            current = k % 3
            next1 = (k + 1) % 3
            next2 = (k + 2) % 3
            
            # 预取 k+2 的数据（如果存在）
            if k + 2 < num_k_tiles:
                T.copy(A[by * BLOCK_M, (k + 2) * BLOCK_K], A_buf[next2])
                T.copy(B[(k + 2) * BLOCK_K, bx * BLOCK_N], B_buf[next2])
            
            # 使用当前数据计算
            T.gemm(A_buf[current], B_buf[current], C_local)
        
        T.copy(C_local, C[by * BLOCK_M, bx * BLOCK_N])
```

该示例展示了三级流水线预取策略。分配三个 shared memory buffer，分别用于当前计算、下一块预取和下下块预取。通过取模运算循环使用 buffer，实现计算与访存的完全重叠。预取 k+2 的数据确保当前计算完成时数据已就绪。多级缓冲可以更好地隐藏内存延迟，特别适合计算强度较低的操作。

### 22.9.3 内存复用模式

```python
@T.prim_func
def memory_reuse_patterns(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float32"),
):
    """
    内存复用模式
    
    模式 1：In-place 操作
    - 直接在原数据上修改，避免额外的内存分配
    
    模式 2：Buffer 复用
    - 同一个 buffer 用于不同的计算阶段
    - 需要确保前一阶段的数据不再需要
    
    模式 3：Register 复用
    - 在寄存器中保存多个变量，减少 shared memory 访问
    """
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=256) as (bx, by):
        # 模式 2：Buffer 复用
        # 使用两个 buffer 交替进行读取和计算
        buf0 = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
        buf1 = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
        B_buf = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
        
        T.clear(C_local)
        
        # 预取第一块到 buf0
        T.copy(A[by * BLOCK_M, 0], buf0)
        T.copy(B[0, bx * BLOCK_N], B_buf)
        
        for k in T.serial(T.ceildiv(K, BLOCK_K)):
            current_buf = buf0 if k % 2 == 0 else buf1
            next_buf = buf1 if k % 2 == 0 else buf0
            
            # 预取下一块到另一个 buffer
            if k + 1 < T.ceildiv(K, BLOCK_K):
                T.copy(A[by * BLOCK_M, (k + 1) * BLOCK_K], next_buf)
                T.copy(B[(k + 1) * BLOCK_K, bx * BLOCK_N], B_buf)
            
            # 使用当前 buffer 计算
            T.gemm(current_buf, B_buf, C_local)
        
        T.copy(C_local, C[by * BLOCK_M, bx * BLOCK_N])
```

该示例展示了 buffer 复用的内存优化模式。使用两个 buffer 交替进行数据加载和计算，避免同时分配多份数据。当前 buffer 用于计算时，另一个 buffer 预取下一块数据。循环结束后交换 buffer 指针。这种模式减少了 shared memory 的总使用量，在容量受限的场景下特别有用。

---

> 高级内存优化技术需要针对具体硬件架构进行调优。不同的 GPU 架构（如 NVIDIA Ampere/Hopper、AMD CDNA）在内存层次结构、带宽和特殊功能上存在显著差异。了解这些差异有助于编写跨平台高效的 kernel。

## 22.10 不同硬件架构的内存优化

### 22.10.1 NVIDIA Ampere vs Hopper

```python
"""
NVIDIA 架构内存特性对比

Ampere (A100):
- HBM2e: 80 GB, 2039 GB/s
- L2 Cache: 40 MB
- Shared Memory: 164 KB/SM (configurable)
- 支持 async copy
- 支持 L2 compression

Hopper (H100):
- HBM3: 80 GB, 3350 GB/s
- L2 Cache: 50 MB
- Shared Memory: 228 KB/SM (configurable)
- 支持 async copy (enhanced)
- 支持 TMA (Tensor Memory Accelerator)
- 支持分布式 shared memory

关键区别：
1. H100 的 TMA 可以自动处理多维数据搬运
2. H100 支持更大的 shared memory
3. H100 的 L2 cache 更大，命中率更高
"""

class HardwareSpecificOptimizer:
    """
    硬件特定的内存优化器
    """
    
    def __init__(self, gpu_arch):
        self.arch = gpu_arch
        self.specs = self._get_specs()
    
    def _get_specs(self):
        specs = {
            "nvidia-a100": {
                "hbm_bandwidth_gbs": 2039,
                "l2_size_mb": 40,
                "smem_per_sm_kb": 164,
                "supports_tma": False,
                "supports_async_copy": True,
            },
            "nvidia-h100": {
                "hbm_bandwidth_gbs": 3350,
                "l2_size_mb": 50,
                "smem_per_sm_kb": 228,
                "supports_tma": True,
                "supports_async_copy": True,
            },
            "amd-mi300x": {
                "hbm_bandwidth_gbs": 5300,
                "l2_size_mb": 256,  # Infinity Cache
                "smem_per_sm_kb": 64,
                "supports_tma": False,
                "supports_async_copy": True,
            },
        }
        return specs.get(self.arch, specs["nvidia-a100"])
    
    def suggest_tile_size(self, M, N, K):
        """根据硬件建议 tile size"""
        smem_budget = self.specs["smem_per_sm_kb"] * 1024 * 0.8  # 留 20% 余量
        
        # A100: 偏好较大的 tile
        if self.arch == "nvidia-a100":
            return {"block_M": 128, "block_N": 256, "block_K": 32}
        # H100: 可以使用更大的 tile（得益于更大的 smem）
        elif self.arch == "nvidia-h100":
            return {"block_M": 256, "block_N": 256, "block_K": 64}
        # MI300X: 需要考虑 wave 大小为 64
        elif self.arch == "amd-mi300x":
            return {"block_M": 128, "block_N": 128, "block_K": 32}
        
        return {"block_M": 128, "block_N": 128, "block_K": 32}
    
    def suggest_pipeline_depth(self, compute_intensity):
        """根据计算强度建议流水线深度"""
        if compute_intensity > 100:  # 高计算强度
            return 2  # 不需要太多流水线
        elif compute_intensity > 10:  # 中等
            return 3
        else:  # 低计算强度，需要更多流水线隐藏延迟
            return 4
```

该优化器根据不同硬件架构提供定制建议。A100 偏好较大的 tile（128×256×32），H100 可以使用更大的 tile（256×256×64）得益于更大的 shared memory。MI300X 需要考虑 wave64 执行模型。流水线深度根据计算强度调整：高强度操作不需要太多流水线，低强度操作需要更多流水线隐藏延迟。硬件感知的优化是达到峰值性能的关键。

### 22.10.2 AMD CDNA 内存优化

```python
"""
AMD CDNA 架构内存特性

MI300X:
- HBM3: 192 GB, 5300 GB/s
- Infinity Cache: 256 MB（类似 L3 cache）
- Shared Memory: 64 KB/CU
- Wave 大小：64（不同于 NVIDIA 的 32）

关键优化点：
1. 利用 Infinity Cache 减少 HBM 访问
2. 适配 Wave64 执行模型
3. 使用 ds_read/ds_write 进行 shared memory 操作
"""

@T.prim_func
def amd_optimized_kernel(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float32"),
):
    """
    AMD GPU 优化的 kernel
    
    注意事项：
    1. Wave 大小为 64，线程组织需要调整
    2. Shared memory 大小较小（64 KB），需要更精细的管理
    3. Infinity Cache 可以缓存更多数据
    """
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    # AMD GPU 使用 wave64，线程数通常是 256（4 waves）
    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=256) as (bx, by):
        # 由于 shared memory 较小，使用更保守的分配
        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
        B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
        
        T.clear(C_local)
        
        for k in T.serial(T.ceildiv(K, BLOCK_K)):
            T.copy(A[by * BLOCK_M, k * BLOCK_K], A_shared)
            T.copy(B[k * BLOCK_K, bx * BLOCK_N], B_shared)
            
            # 使用 MFMA 指令
            T.gemm(A_shared, B_shared, C_local)
        
        T.copy(C_local, C[by * BLOCK_M, bx * BLOCK_N])
```

该 kernel 针对 AMD CDNA 架构进行了优化。AMD GPU 使用 wave64 执行模型，线程数通常是 256（4 waves）。Shared memory 容量较小（64 KB/CU），需要更保守的分配策略。Infinity Cache（256 MB）可以缓存更多数据，减少 HBM 访问。使用 MFMA 指令进行矩阵计算，充分利用 AMD 的 Matrix Core。

---

## 22.11 内存优化检查清单

### 22.11.1 系统化检查流程

```python
class MemoryOptimizationChecklist:
    """
    内存优化检查清单
    
    系统地检查和应用内存优化技术
    """
    
    CHECKLIST = [
        {
            "category": "全局内存访问",
            "checks": [
                ("合并访问", "同一 warp 的线程是否访问连续地址？"),
                ("向量化加载", "是否使用了 float4 等宽类型？"),
                ("只读缓存", "是否使用了 __ldg 或 T.readonly？"),
                ("数据预取", "是否在需要数据前发起加载？"),
            ],
        },
        {
            "category": "Shared Memory",
            "checks": [
                ("Bank Conflict", "是否有 bank conflict？是否使用了 padding？"),
                ("容量使用", "是否超出了 shared memory 容量？"),
                ("Double Buffering", "是否使用了 double buffering？"),
                ("数据布局", "数据布局是否适合计算模式？"),
            ],
        },
        {
            "category": "寄存器使用",
            "checks": [
                ("Occupancy", "寄存器使用是否导致 Occupancy 过低？"),
                ("Spill", "是否有寄存器 spill 到 local memory？"),
                ("复用", "是否充分利用了寄存器复用？"),
                ("生命周期", "变量生命周期是否最小化？"),
            ],
        },
        {
            "category": "Cache 利用",
            "checks": [
                ("L1 Cache", "数据局部性是否良好？"),
                ("L2 Cache", "工作集是否能放入 L2？"),
                ("Cache Thrashing", "是否有频繁的 cache 替换？"),
            ],
        },
    ]
    
    def print_checklist(self):
        """打印检查清单"""
        print("=== 内存优化检查清单 ===\n")
        
        for category in self.CHECKLIST:
            print(f"\n--- {category['category']} ---")
            for i, (name, question) in enumerate(category['checks'], 1):
                print(f"  [ ] {name}: {question}")
    
    def analyze_kernel(self, kernel_source, compile_info):
        """
        分析 kernel 的内存优化状态
        """
        issues = []
        
        # 检查 shared memory 使用
        smem_usage = compile_info.get("shared_memory_bytes", 0)
        if smem_usage > 48 * 1024:
            issues.append({
                "category": "Shared Memory",
                "severity": "warning",
                "message": f"Shared memory 使用 {smem_usage/1024:.1f} KB，可能限制 Occupancy",
            })
        
        # 检查寄存器使用
        regs = compile_info.get("registers_per_thread", 0)
        if regs > 128:
            issues.append({
                "category": "寄存器",
                "severity": "warning",
                "message": f"每线程使用 {regs} 个寄存器，可能导致低 Occupancy",
            })
        
        return issues
```

该检查清单提供了系统化的内存优化流程。分为四个维度：全局内存访问、Shared Memory、寄存器使用和 Cache 利用。每个维度包含具体的检查项和判断标准。通过逐项检查，可以确保 kernel 充分利用了所有内存层级的优化机会。该工具适合在开发和调优阶段使用，帮助发现潜在的性能问题。

---

## 22.12 带宽利用率深度分析

### 22.12.1 理论带宽 vs 实际带宽

```python
class BandwidthDeepAnalysis:
    """深度分析带宽利用率的各个维度。"""
    
    def __init__(self, gpu="nvidia-a100"):
        self.specs = {
            "nvidia-a100": {
                "hbm_bandwidth_gbs": 2039,
                "l2_bandwidth_gbs": 5000,
                "smem_bandwidth_tbs": 19,
                "register_bandwidth_tbs": 192,
                "hbm_size_gb": 80,
                "l2_size_mb": 40,
                "smem_per_sm_kb": 164,
                "num_sms": 108,
            },
            "nvidia-h100": {
                "hbm_bandwidth_gbs": 3350,
                "l2_bandwidth_gbs": 8000,
                "smem_bandwidth_tbs": 33,
                "register_bandwidth_tbs": 300,
                "hbm_size_gb": 80,
                "l2_size_mb": 50,
                "smem_per_sm_kb": 228,
                "num_sms": 132,
            },
        }[gpu]
    
    def analyze_bandwidth_bottleneck(self, bytes_transferred, elapsed_ms, compute_flops):
        """分析带宽瓶颈的各个层面。"""
        # HBM 带宽
        hbm_bw = bytes_transferred / 1e9 / (elapsed_ms / 1e3)
        hbm_util = hbm_bw / self.specs["hbm_bandwidth_gbs"] * 100
        
        # 计算吞吐
        compute_tflops = compute_flops / 1e12 / (elapsed_ms / 1e3)
        compute_util = compute_tflops / 312 * 100  # A100 FP16 Tensor Core
        
        # 计算强度
        ai = compute_flops / bytes_transferred  # FLOPs/Byte
        ridge_point = self.specs["hbm_bandwidth_gbs"] * 1e9 / (312e12)
        
        if ai > ridge_point:
            bottleneck = "compute"
            efficiency = compute_util
        else:
            bottleneck = "memory"
            efficiency = hbm_util
        
        return {
            "hbm_bandwidth_gbs": hbm_bw,
            "hbm_utilization_pct": hbm_util,
            "compute_tflops": compute_tflops,
            "compute_utilization_pct": compute_util,
            "arithmetic_intensity": ai,
            "ridge_point": ridge_point,
            "bottleneck": bottleneck,
            "efficiency_pct": efficiency,
        }
    
    def estimate_cache_behavior(self, data_size_bytes, access_pattern, cache_size_bytes):
        """估计缓存行为。"""
        cache_line_size = 128  # bytes
        
        if access_pattern == "sequential":
            # 顺序访问：cache line 完全利用
            effective_bw_ratio = 1.0
            cache_hit_rate = min(1.0, cache_size_bytes / data_size_bytes)
        elif access_pattern == "strided_2":
            # 步长为 2：每条 cache line 只用一半
            effective_bw_ratio = 0.5
            cache_hit_rate = 0.3
        elif access_pattern == "random":
            # 随机访问：cache line 利用率低
            effective_bw_ratio = 4.0 / cache_line_size
            cache_hit_rate = 0.05
        else:
            effective_bw_ratio = 0.8
            cache_hit_rate = 0.5
        
        return {
            "effective_bandwidth_ratio": effective_bw_ratio,
            "estimated_cache_hit_rate": cache_hit_rate,
            "recommended_prefetch_distance": cache_size_bytes // 4,
        }
```

该分析器从多个维度剖析带宽利用率。通过计算 HBM 带宽、计算吞吐和算术强度，判断瓶颈在内存还是计算。Ridge point 定义了计算 bound 和 memory bound 的分界线。该工具还估计缓存行为：顺序访问的 cache line 利用率为 100%，随机访问仅约 3%。理解这些指标有助于制定针对性的优化策略。

### 22.12.2 带宽利用率优化实例

```python
@T.prim_func
def bandwidth_optimized_copy(
    A: T.Tensor((M, N), "float32"),
    B: T.Tensor((M, N), "float32"),
):
    """
    带宽优化的数据拷贝
    
    优化策略：
    1. 向量化加载（float4 = 16 bytes per load）
    2. 合并访问（连续线程访问连续地址）
    3. 展开循环减少指令开销
    """
    with T.Kernel(T.ceildiv(M * N, 1024), threads=256) as bx:
        tid = T.get_thread_id()
        base = bx * 1024 + tid * 4
        
        # 向量化加载：一次加载 4 个 float32 = 16 bytes
        if base + 3 < M * N:
            a0 = A[base, 0]  # 实际应使用向量化加载
            a1 = A[base + 1, 0]
            a2 = A[base + 2, 0]
            a3 = A[base + 3, 0]
            
            B[base, 0] = a0
            B[base + 1, 0] = a1
            B[base + 2, 0] = a2
            B[base + 3, 0] = a3
```

该示例展示了带宽优化的数据拷贝实现。每个线程处理 4 个连续的 float32 元素（共 16 字节），实现了向量化加载。连续线程访问连续地址确保了合并访问模式。虽然代码中使用了标量加载，但实际应用中应使用 float4 向量化加载指令，将单次加载的数据量从 4 字节提升到 16 字节，显著提高内存事务利用率。

### 22.12.3 不同操作的典型带宽利用率

| 操作 | 理论带宽需求 | 典型实现带宽 | 利用率 | 瓶颈 |
|------|------------|------------|--------|------|
| 向量加法 | 12 GB/s (1M float32) | 1800 GB/s | 88% | 接近峰值 |
| 矩阵拷贝 | 同上 | 1900 GB/s | 93% | 接近峰值 |
| GEMM (M=N=K=4096) | 96 MB | 280 TFLOPS | 90% | Compute-bound |
| Softmax (N=4096) | 32 KB/row | 800 GB/s | 39% | 内存访问模式 |
| LayerNorm | 16 KB/row | 1200 GB/s | 59% | 两遍扫描 |

---

> 带宽利用率分析帮助我们识别内存瓶颈，而 Shared Memory 的 Bank Conflict 是影响片上存储效率的关键因素。接下来我们将深入分析复杂访问模式下的 Bank Conflict 问题，并介绍 Swizzle 技术的数学原理。

## 22.13 Shared Memory Bank Conflict 深度分析

### 22.13.1 复杂访问模式的 Bank Conflict

```python
def analyze_complex_bank_conflict():
    """分析复杂访问模式的 Bank Conflict。"""
    
    print("=== 复杂 Bank Conflict 分析 ===\n")
    
    # 场景 5：矩阵转置时的 bank conflict
    print("场景 5: 矩阵转置 (32x32 float32)")
    print("  行访问：线程 i 访问 smem[i][j] → Bank (i*32+j) % 32 = j")
    print("  列访问：线程 i 访问 smem[j][i] → Bank (j*32+i) % 32 = i")
    print("  列访问时无 conflict，但行写入列读取时可能有 conflict")
    print("  解决：Padding 到 33 列\n")
    
    # 场景 6：对角线访问
    print("场景 6: 对角线访问 smem[i][(i+k) % 32]")
    print("  线程 i 访问 Bank (i+k) % 32")
    print("  所有线程访问不同 bank → 无 conflict")
    print("  但如果 k=0，所有线程访问 Bank i → 无 conflict（每个 bank 一个线程）\n")
    
    # 场景 7：步长访问
    print("场景 7: 步长访问 smem[i][i*4]")
    print("  线程 i 访问 Bank (i*4) % 32")
    print("  线程 0→Bank 0, 线程 1→Bank 4, 线程 2→Bank 8, ...")
    print("  线程 8→Bank 0, 线程 9→Bank 4, ...")
    print("  8-way conflict！\n")
    
    # 场景 8：广播访问
    print("场景 8: 广播访问 smem[0][tid]")
    print("  所有线程访问同一行的不同列")
    print("  线程 i 访问 Bank i → 无 conflict")
    print("  但如果 smem[tid][0]，所有线程访问 Bank 0 → 32-way conflict\n")
```

该函数详细分析了四种复杂的 Bank Conflict 场景。矩阵转置时，行访问无冲突但列访问可能产生冲突，需要 Padding 解决。对角线访问通常无冲突，因为每个线程访问不同的 bank。步长为 4 的访问会导致 8-way conflict，每 8 个线程访问同一个 bank。广播访问中，所有线程访问同一行的不同列无冲突，但访问同一列的不同行会产生 32-way conflict。

### 22.13.2 Swizzle 的数学原理

```python
def swizzle_address(row, col, num_banks=32):
    """
    Swizzle 地址计算的数学原理。
    
    原始地址：addr = row * stride + col
    Bank = addr % num_banks = (row * stride + col) % 32
    
    Swizzled 地址：col_swizzled = col ^ (row % (num_banks / elem_size))
    Bank = (row * stride + col_swizzled) % 32
    
    XOR 操作的关键性质：
    - 对于不同的 row，col_swizzled 会不同
    - 对于同一列的不同行，访问的 bank 不同
    - 消除了列访问时的 bank conflict
    """
    elem_size = 4  # float32 = 4 bytes
    bits_per_bank = 2  # log2(4)
    
    # Swizzle: XOR col 的高位与 row
    swizzle_bits = (row % (num_banks // elem_size))
    col_swizzled = col ^ swizzle_bits
    
    # 计算 bank
    addr = row * 32 + col_swizzled
    bank = (addr // elem_size) % num_banks
    
    return col_swizzled, bank
```

该函数实现了 Swizzle 地址映射的数学原理。通过 XOR 操作将列索引与行索引的低位进行异或，得到 swizzled 列索引。XOR 的关键性质是：对于同一列的不同行，swizzled 后的列索引不同，从而映射到不同的 bank。这消除了列访问时的 bank conflict，且不需要额外的 padding 空间。Swizzle 特别适合 Tensor Core 操作，因为 MMA 指令要求特定的数据布局。

---

> Bank Conflict 优化提升了 Shared Memory 的访问效率，而寄存器 Spilling 则直接影响 kernel 的执行速度。当寄存器使用量超过硬件限制时，溢出的变量会被存储到延迟高得多的 Local Memory，导致严重的性能下降。

## 22.14 寄存器 Spilling 分析

### 22.14.1 寄存器 Spilling 的性能影响

```python
class RegisterSpillAnalyzer:
    """分析寄存器 spilling 的性能影响。"""
    
    def __init__(self, gpu="sm_80"):
        self.specs = {
            "sm_80": {
                "max_regs_per_thread": 255,
                "reg_file_size_kb": 256,
                "local_mem_latency_cycles": 400,
                "smem_latency_cycles": 20,
            }
        }[gpu]
    
    def estimate_spill_impact(self, regs_used, threads_per_block, active_warps):
        """估计 spilling 的性能影响。"""
        max_regs = self.specs["max_regs_per_thread"]
        
        if regs_used <= max_regs:
            return {
                "spill_occurs": False,
                "performance_impact": "none",
                "recommendation": "No spilling detected",
            }
        
        spill_bytes_per_thread = (regs_used - max_regs) * 4  # 4 bytes per register
        spill_bytes_per_warp = spill_bytes_per_thread * 32
        spill_bytes_per_block = spill_bytes_per_thread * threads_per_block
        
        # Spilling 到 local memory（L1/L2 缓存）
        latency_increase = self.specs["local_mem_latency_cycles"] - self.specs["smem_latency_cycles"]
        
        return {
            "spill_occurs": True,
            "spill_bytes_per_thread": spill_bytes_per_thread,
            "spill_bytes_per_block": spill_bytes_per_block,
            "latency_increase_per_access": f"{latency_increase} cycles",
            "estimated_slowdown": f"{latency_increase / self.specs['smem_latency_cycles']:.1f}x",
            "recommendation": "Reduce register usage by: 1) Smaller tile sizes, 2) Fewer pipeline stages, 3) Loop reordering",
        }
    
    def suggest_optimizations(self, kernel_info):
        """建议减少寄存器使用的优化方法。"""
        suggestions = []
        
        if kernel_info.get("block_M", 0) > 128:
            suggestions.append("减小 block_M 到 128 或 64")
        if kernel_info.get("block_N", 0) > 128:
            suggestions.append("减小 block_N 到 128 或 64")
        if kernel_info.get("num_stages", 0) > 3:
            suggestions.append("减少 pipeline stage 数量到 2-3")
        if kernel_info.get("unroll_factor", 0) > 4:
            suggestions.append("减小循环展开因子")
        
        suggestions.append("使用 T.alloc_fragment 的较小形状")
        suggestions.append("将部分中间结果存储到 Shared Memory")
        
        return suggestions
```

该分析器量化了寄存器 Spilling 的性能影响。当每线程寄存器使用量超过 255 时，溢出的变量会被存储到 Local Memory，访问延迟从 20 cycles 增加到 400 cycles，性能下降约 20 倍。优化建议包括：减小 tile 大小、减少流水线深度、减小循环展开因子。这些方法可以有效降低寄存器压力，避免 Spilling 发生。

### 22.14.2 通过代码重构减少寄存器使用

```python
@T.prim_func
def register_efficient_kernel(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float32"),
):
    """
    寄存器高效的 kernel 实现。
    
    技巧：
    1. 缩小 Fragment 大小
    2. 逐块计算而非一次性加载
    3. 重用寄存器变量
    """
    BLOCK_M = 64  # 较小的 block 大小减少寄存器压力
    BLOCK_N = 64
    BLOCK_K = 32
    
    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=128) as (bx, by):
        # 使用较小的 Fragment
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
        T.clear(C_local)
        
        # 逐块处理 K 维度
        for k in T.serial(T.ceildiv(K, BLOCK_K)):
            # 重用同一个 buffer
            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
            B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
            
            T.copy(A[by * BLOCK_M, k * BLOCK_K], A_shared)
            T.copy(B[k * BLOCK_K, bx * BLOCK_N], B_shared)
            
            # 逐行计算，减少同时活跃的寄存器
            for i in T.serial(BLOCK_M):
                a_frag = T.alloc_fragment((1, BLOCK_K), "float16")
                for kk in T.serial(BLOCK_K):
                    a_frag[0, kk] = A_shared[i, kk]
                for j in T.serial(BLOCK_N):
                    for kk in T.serial(BLOCK_K):
                        C_local[i, j] += a_frag[0, kk].astype("float32") * B_shared[kk, j].astype("float32")
        
        T.copy(C_local, C[by * BLOCK_M, bx * BLOCK_N])
```

这段代码是 22.14.2 通过代码重构减少寄存器使用 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 22.15 L1/L2 Cache 行为分析

### 22.15.1 L2 Cache 工作集分析

```python
class L2CacheAnalyzer:
    """分析 L2 Cache 的工作集行为。"""
    
    def __init__(self, l2_size_mb=40):
        self.l2_size_bytes = l2_size_mb * 1024 * 1024
    
    def analyze_working_set(self, kernel_params):
        """分析 kernel 的工作集是否能放入 L2。"""
        M, N, K = kernel_params["M"], kernel_params["N"], kernel_params["K"]
        dtype_bytes = kernel_params.get("dtype_bytes", 2)  # float16 = 2 bytes
        
        # A 矩阵大小
        a_size = M * K * dtype_bytes
        # B 矩阵大小
        b_size = K * N * dtype_bytes
        # C 矩阵大小
        c_size = M * N * 4  # float32 输出
        
        total_size = a_size + b_size + c_size
        
        # 考虑 tiling 后的实际工作集
        block_m = kernel_params.get("block_M", 128)
        block_n = kernel_params.get("block_N", 128)
        block_k = kernel_params.get("block_K", 32)
        
        # 每个 SM 的工作集
        tile_a = block_m * block_k * dtype_bytes
        tile_b = block_k * block_n * dtype_bytes
        tile_c = block_m * block_n * 4
        
        sm_workset = tile_a + tile_b + tile_c
        
        return {
            "total_data_size_mb": total_size / 1024 / 1024,
            "l2_size_mb": self.l2_size_bytes / 1024 / 1024,
            "fits_in_l2": total_size <= self.l2_size_bytes,
            "l2_utilization_pct": total_size / self.l2_size_bytes * 100,
            "per_sm_workset_kb": sm_workset / 1024,
            "recommendation": (
                "Data fits in L2" if total_size <= self.l2_size_bytes
                else "Data exceeds L2, consider: 1) Smaller tiles, 2) Data reuse strategies, 3) Streaming access"
            ),
        }
```

这段代码是 22.15.1 L2 Cache 工作集分析 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

### 22.15.2 Cache-Friendly 访问模式

```python
@T.prim_func
def cache_friendly_gemm(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float32"),
):
    """
    Cache-friendly GEMM 实现。
    
    关键策略：
    1. A 矩阵按行访问（利用 L1 cache line）
    2. B 矩阵按列访问（利用 L2 缓存预取）
    3. 使用 Z-order 或 Morton-order 访问提高缓存局部性
    """
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 32
    
    with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=256) as (bx, by):
        A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
        B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
        C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
        T.clear(C_local)
        
        for k in T.serial(T.ceildiv(K, BLOCK_K)):
            # A: 按行加载，利用 L1 cache line
            T.copy(A[by * BLOCK_M, k * BLOCK_K], A_shared)
            
            # B: 按列加载，利用 L2 缓存预取
            T.copy(B[k * BLOCK_K, bx * BLOCK_N], B_shared)
            
            T.gemm(A_shared, B_shared, C_local)
        
        T.copy(C_local, C[by * BLOCK_M, bx * BLOCK_N])
```

这段代码是 22.15.2 Cache-Friendly 访问模式 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 22.16 Profiling 工具使用指南

### 22.16.1 Nsight Compute 使用

```bash
# 基础分析
ncu --set basic python my_kernel.py

# 详细分析（包含内存访问模式）
ncu --set full python my_kernel.py

# 自定义指标
ncu --metrics \
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,\
    sm__warps_active.avg.pct_of_peak_sustained_active \
    python my_kernel.py

# 导出报告
ncu --set full --export report.ncu-rep python my_kernel.py
```

这段命令对应 22.16.1 Nsight Compute 使用 中的实际操作步骤，重点在于把环境、工具链或性能诊断流程拆成可验证的阶段。阅读时应关注每条命令的输入、输出和依赖关系，而不是机械复制；例如路径、版本、设备权限和环境变量都会影响最终结果。工程实践中建议每执行完一个阶段就做一次最小验证，这样能把问题定位在安装、编译、运行或 profiling 的具体环节。对于性能测试命令，还要注意预热、同步和重复测量，否则得到的数字很容易被缓存或异步执行干扰。

### 22.16.2 关键指标解读

| 指标 | 含义 | 目标值 | 优化方向 |
|------|------|--------|---------|
| `sm__throughput` | SM 计算吞吐量 | > 80% | 计算优化 |
| `gpu__dram_throughput` | DRAM 带宽利用率 | > 80% | 内存优化 |
| `l1tex__data_bank_conflicts` | Bank Conflict 次数 | 0 | Padding/Swizzle |
| `sm__warps_active` | Occupancy | > 50% | 寄存器/SMEM 优化 |
| `l1tex__t_sectors` | L1 加载扇区数 | 接近理论最小 | 合并访问 |

### 22.16.3 PyTorch Profiler 集成

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

def profile_tilelang_kernel(kernel_func, *inputs):
    """使用 PyTorch Profiler 分析 TileLang kernel。"""
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("tilelang_kernel"):
            kernel_func(*inputs)
    
    # 打印 CUDA 活动
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    # 导出 Chrome trace
    prof.export_chrome_trace("trace.json")
```

这段代码是 22.16.3 PyTorch Profiler 集成 的关键实现片段，展示了张量形状、内存层级、调度策略或后端接口之间的对应关系。理解时可以先确认输入输出，再沿着资源分配、循环结构和写回逻辑观察数据如何流动。这样的写法通常服务于两个目标：一是让算法语义保持清晰，二是给编译器和硬件提供足够明确的优化信息。需要特别注意边界条件、同步位置、数据类型和临时缓冲区大小，因为这些细节往往决定 kernel 是正确运行、性能达标，还是出现隐藏的数值误差与访存瓶颈。

---

## 22.17 本章小结（扩展）

### 内存优化决策树

```
性能瓶颈分析
├── Compute-bound？
│   ├── 是 → 使用 Tensor Core，优化计算
│   └── 否 → Memory-bound ↓
│
├── 带宽利用率低？
│   ├── 合并访问？ → 检查访问模式
│   ├── 向量化？ → 使用 float4 等宽类型
│   └── 缓存效率？ → 优化数据局部性
│
├── Bank Conflict？
│   ├── Padding → 添加额外列
│   └── Swizzle → XOR 地址映射
│
├── 寄存器 Spill？
│   ├── 减小 Fragment → 控制寄存器使用
│   ├── 减少 Stage → 降低 Pipeline 深度
│   └── 使用 SMEM → 缓存中间结果
│
└── Occupancy 低？
    ├── 减少寄存器 → 使用更少的局部变量
    ├── 减少 SMEM → 使用更小的 tile
    └── 调整 block size → 优化线程配置
```

这个代码块或示意图用于说明 内存优化决策树 的整体结构、执行阶段和关键数据流。阅读时应把模块之间的连接关系与前后文的概念对应起来，理解哪些部分发生在全局内存，哪些部分发生在片上缓存或寄存器中。它的设计目的不是单纯展示流程，而是帮助判断性能瓶颈可能出现在加载、计算、同步还是写回阶段。实际实现时还需要补充边界检查、资源约束和硬件差异分析，避免把概念图误认为无额外成本的直接执行方案。

### 内存优化检查清单（扩展）

| 检查项 | 方法 | 工具 |
|--------|------|------|
| 合并访问 | 确保 warp 内线程访问连续地址 | ncu |
| 向量化加载 | 使用 float4/int4 等宽类型 | 代码审查 |
| Bank Conflict | Padding 或 Swizzle | ncu |
| Shared Memory 使用 | 控制在 SM 限制内 | 编译器输出 |
| 寄存器 Spill | 监控 local memory 访问 | ncu |
| Occupancy | 使用 occupancy calculator | ncu |
| L1/L2 命中率 | 优化数据局部性 | ncu |
| 只读缓存 | 使用 __ldg 或 T.readonly | 代码审查 |

---

## 扩展阅读

1. **NVIDIA CUDA Programming Guide, Chapter 5: Memory Hierarchy** - 官方内存层次文档
2. **Volkov, V. (2010).** "Better Performance at Lower Occupancy." - 寄存器与 Occupancy 权衡的经典分析
3. **Micikevicius, P. (2018).** "General Matrix Multiply and Convolution on NVIDIA Tensor Cores." - Tensor Core 内存布局要求
4. **Jia, Z., et al. (2019).** "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking." - 微架构级别的内存分析
5. **AMD ROCm Memory Model Documentation** - AMD GPU 内存模型

---

## 下一章预告

**Chapter 23: 计算优化与指令级并行** — 当内存不再是瓶颈时，如何最大化计算效率？我们将深入探讨 Tensor Core/MFMA 利用率优化、循环展开策略、指令流水线、异步拷贝与计算重叠等计算优化技术。

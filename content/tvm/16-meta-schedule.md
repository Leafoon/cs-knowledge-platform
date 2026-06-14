> **学习目标**：
> - 理解 MetaSchedule（Ansor）无模板调优的设计哲学与核心动机
> - 掌握 Program、Schedule、ScheduleRule、Postproc 四大核心抽象
> - 理解代价模型（Cost Model）的训练与推理机制
> - 掌握进化搜索（Evolutionary Search）算法的实现细节
> - 了解 MetaSchedule 的源码结构与扩展机制

---

## 16.1 从 AutoTVM 到 MetaSchedule：无模板调优的动机

### 16.1.1 AutoTVM 的局限性

在 Chapter 15 中，我们详细讨论了 AutoTVM 的模板驱动调优方法。AutoTVM 的核心思想是让开发者手写 **Schedule Template**，然后用机器学习模型搜索模板中的参数空间。这种方法虽然有效，但存在三个根本性问题：

**问题一：模板编写的人力成本**

每个算子都需要专家手动设计模板，这是一个极其耗时的过程：

```python
# AutoTVM 中为 conv2d 编写的模板（简化示例）
@autotvm.template("conv2d_nchw")
def conv2d_nchw_template(data, kernel, stride, padding):
    """需要专家手动设计的调度模板"""
    N, C, H, W = data.shape
    OC, _, KH, KW = kernel.shape

    # 专家必须手动决定分块策略
    cfg = autotvm.get_config()
    cfg.define_knob("tile_n", [1, 2, 4, 8])
    cfg.define_knob("tile_oc", [16, 32, 64, 128])
    cfg.define_knob("tile_oh", [1, 2, 4, 7])
    cfg.define_knob("tile_ow", [1, 2, 4, 7])
    cfg.define_knob("unroll_step", [64, 128, 256])

    # 模板代码（数十到数百行）
    s = te.create_schedule(output.op)
    # ... 手动调度逻辑
    return s, [data, kernel, output]
```

**问题二：搜索空间受限**

模板的搜索空间是模板设计者**预先定义**的，这意味着：

$$\text{搜索空间}_{\text{AutoTVM}} \subseteq \text{搜索空间}_{\text{可能的调度}}$$

许多有效的调度方案可能不在任何模板的搜索空间中。例如，AutoTVM 的模板通常无法表达复杂的跨算子融合策略。

**问题三：模板不可组合**

不同的模板是独立设计的，无法组合使用。当多个算子需要联合优化时（如融合后的算子），AutoTVM 往往无能为力。

<div data-component="AutoTVMvsMetaSchedule"></div>

### 16.1.2 MetaSchedule 的设计哲学

MetaSchedule（其前身为 Ansor 项目）的核心设计哲学是 **"不需要模板的自动调优"**：

1. **自动生成搜索空间**：通过分析 TIR 程序的结构，自动生成所有合法的调度变换
2. **层次化采样**：先采样高层结构（如分块策略），再采样低层细节（如展开因子）
3. **代价模型指导搜索**：用学习到的代价模型替代实际执行，大幅加速搜索
4. **进化搜索**：使用进化算法在巨大的搜索空间中高效寻优

```
┌─────────────────────────────────────────────────────────┐
│                  MetaSchedule 工作流程                     │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │  原始 TIR │───▶│ 搜索空间  │───▶│ 采样生成  │          │
│  │  Program  │    │ 构建      │    │ Schedule │          │
│  └──────────┘    └──────────┘    └─────┬────┘          │
│                                       │                │
│                                       ▼                │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │ 最优调度  │◀───│ 进化搜索  │◀───│ 代价模型  │          │
│  │ 输出      │    │ Evolution │    │ Cost Model│          │
│  └──────────┘    └──────────┘    └──────────┘          │
└─────────────────────────────────────────────────────────┘
```

### 16.1.3 MetaSchedule vs AutoTVM vs 手动调优

| 维度 | 手动调优 | AutoTVM | MetaSchedule |
|------|---------|---------|--------------|
| 搜索空间 | 人工设计 | 模板定义 | 自动生成 |
| 人力成本 | 极高（专家级） | 高（需写模板） | 低（无需模板） |
| 搜索空间大小 | 1 | $10^3 \sim 10^5$ | $10^{10} \sim 10^{20}$ |
| 优化质量 | 最高（专家经验） | 中等 | 高（接近专家水平） |
| 适用算子范围 | 特定算子 | 有模板的算子 | 通用 |
| 搜索效率 | N/A | 快（小空间） | 中等（需代价模型） |

---

## 16.2 核心抽象：Program / Schedule / Space / Trace

### 16.2.1 MetaSchedule 的源码组织

MetaSchedule 的源码分布在多个目录中：

```
src/meta_schedule/
├── module.cc                  # Module 层调度入口
├── space.cc                   # 搜索空间构建
├── trace.cc                   # 调度轨迹记录
├── search_strategy/
│   ├── evolutionary_search.cc # 进化搜索
│   ├── replay_trace.cc        # 轨迹回放
│   └── rewart_dispatch.cc     # 重权重分发
├── cost_model/
│    ├── xgb_model.cc          # XGBoost 代价模型
│    ├── ml_p_model.cc         # MLP 代价模型
│    └── random_model.cc       # 随机模型（基线）
├── schedule_rule/
│    ├── add_rfactor.cc        # 规约因子化
│    ├── auto_bind.cc          # 自动线程绑定
│    ├── cross_thread_reduction.cc
│    ├── multi_level_tiling.cc # 多级分块
│    ├── parallel_vectorize.cc # 并行向量化
│    └── retry_compute_root.cc
├── postproc/
│    ├── rewrite_cooperative_fetch.cc
│    ├── rewrite_parallel_vectorize.cc
│    ├── rewrite_reduction_block.cc
│    ├── rewrite_unbound_block.cc
│    └── verify_gpu_code.cc
└── trace_insn.cc              # 指令级 trace

python/tvm/meta_schedule/
├── __init__.py
├── schedule_rule.py           # Python 包装
├── postproc.py
├── search_strategy.py
├── cost_model.py
├── space_generator.py
├── tir_integration.py         # TIR 级集成
└── relay_integration.py      # Relay 级集成
```

### 16.2.2 Program：待优化的 TIR 程序

在 MetaSchedule 中，**Program** 是待优化的 TIR `PrimFunc`。它是搜索的起点：

```python
import tvm
from tvm import meta_schedule as ms
from tvm.script import tir as T

# 定义一个简单的 TIR 程序
@T.prim_func
def matmul_func(
    A: T.Buffer[(1024, 1024), "float32"],
    B: T.Buffer[(1024, 1024), "float32"],
    C: T.Buffer[(1024, 1024), "float32"],
) -> None:
    for i, j, k in T.grid(1024, 1024, 1024):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
```

这个 TIR 程序包含了足够的信息来推断：
- **循环结构**：三重嵌套循环 `i, j, k`
- **访存模式**：`A[vi, vk]` 和 `B[vk, vj]` 的访问模式
- **计算模式**：规约（Reduction）模式，`k` 是规约轴

### 16.2.3 Schedule：调度状态的封装

MetaSchedule 中的 `Schedule` 是对 TIR 程序调度状态的高层封装。与 TE 的 `Schedule` 不同，MetaSchedule 的 `Schedule` 对象**直接操作 TIR 的 block 和 loop 结构**：

```cpp
// src/meta_schedule/schedule.cc
class ScheduleNode : public runtime::Object {
 public:
  /*! \brief The traced TIR schedule state */
  tir::Schedule sch;
  /*! \brief The design space of the schedule */
  Space space;
  /*! \brief The trace of scheduling decisions */
  Trace trace;

  // 核心方法：对 TIR 进行调度变换
  Array<ScheduleRuleResult> ApplyScheduleRule(
      const ScheduleRule& rule, const BlockRV& block);
};
```

`Schedule` 对象提供了与 `tir::Schedule` 一致的调度原语接口：

```python
# 通过 Schedule 对象进行调度变换
sch = tir.Schedule(program)

# 获取所有可调度的 block
blocks = sch.get_blocks("C")

# 应用分块变换
i, j, k = sch.get_loops(blocks[0])
i0, i1 = sch.split(i, factors=[64, 16])
j0, j1 = sch.split(j, factors=[64, 16])
k0, k1 = sch.split(k, factors=[32, 32])

# 重排循环顺序
sch.reorder(i0, j0, k0, i1, k1, j1)
```

### 16.2.4 Trace：调度决策的轨迹记录

**Trace** 是 MetaSchedule 的核心创新之一。它记录了从原始程序到当前调度状态的**所有调度决策序列**：

```python
# Trace 的概念性表示
trace = [
    ("Split", {"loop": "i", "factors": [64, 16]}),
    ("Split", {"loop": "j", "factors": [64, 16]}),
    ("Split", {"loop": "k", "factors": [32, 32]}),
    ("Reorder", {"loops": ["i0", "j0", "k0", "i1", "k1", "j1"]}),
    ("Bind", {"loop": "i0", "thread_axis": "blockIdx.x"}),
    ("Bind", {"loop": "j0", "thread_axis": "blockIdx.y"}),
]
```

Trace 的关键优势在于：
1. **可序列化**：Trace 可以序列化为 JSON，便于存储和传输
2. **可回放**：从空的 Trace 开始重放，可以重建调度状态
3. **可变异**：通过修改 Trace 中的参数，实现搜索空间的变异

```cpp
// src/meta_schedule/trace.cc
class TraceNode : public runtime::Object {
 public:
  /*! \brief The sequence of scheduling instructions */
  Array<Instruction> instructions;
  /*! \brief The decisions made at each instruction */
  Array<ObjectRef> decisions;

  // 回放 trace 重建调度状态
  tir::Schedule Replay(const tir::Schedule& sch) const;

  // 序列化为 JSON
  String AsJSON() const;

  // 从 JSON 反序列化
  static Trace FromJSON(const String& json);
};
```

<div data-component="TraceVisualization"></div>

### 16.2.5 Space：搜索空间的定义

**Space**（搜索空间）定义了在给定 Program 上所有可能的调度变换组合。MetaSchedule 的搜索空间通过两个维度构建：

**维度一：结构空间（Structural Space）**

决定使用哪些调度原语（如 split、reorder、bind）以及它们的**应用顺序**。

**维度二：参数空间（Parametric Space）**

每个调度原语的具体参数（如 split 的因子、unroll 的步长）。

$$|\text{Space}| = \prod_{i=1}^{N} |\text{Choice}_i|$$

其中 $N$ 是需要做决策的点数，$|\text{Choice}_i|$ 是第 $i$ 个决策点的选项数。

```cpp
// src/meta_schedule/space.cc
class SpaceNode : public runtime::Object {
 public:
  /*! \brief The set of schedule rules to apply */
  Array<ScheduleRule> rules;
  /*! \brief The postprocessing rules */
  Array<Postproc> postprocs;
  /*! \brief The design space generator */
  SpaceGenerator space_gen;
};
```

---

## 16.3 ScheduleRule：自动搜索空间构建

### 16.3.1 ScheduleRule 的设计

**ScheduleRule** 是 MetaSchedule 自动生成搜索空间的核心机制。每个 ScheduleRule 定义了一类调度变换的**生成策略**，它分析 TIR 程序的结构，自动生成合法的调度决策：

```cpp
// include/tvm/meta_schedule/schedule_rule.h
class ScheduleRuleNode : public runtime::Object {
 public:
  /*!
   * \brief Given a block in the schedule, return a list of schedule rules
   *        that can be applied to the block.
   */
  virtual Array<tir::Schedule> Apply(const tir::Schedule& sch,
                                      const tir::BlockRV& block) = 0;

  /*!
   * \brief Clone the schedule rule.
   */
  virtual ScheduleRule Clone() const = 0;
};
```

ScheduleRule 的关键设计原则：

1. **自动分析**：不需要用户手动指定变换策略，由 Rule 自动分析程序结构
2. **生成多个候选**：每个 Rule 可以为一个 block 生成多个候选调度方案
3. **可组合**：多个 Rule 可以级联应用，逐步扩展搜索空间

### 16.3.2 MultiLevelTiling：多级分块规则

**MultiLevelTiling** 是最重要的 ScheduleRule，它自动为计算 block 生成多级分块策略。这个规则的核心思想是将循环空间分割为多个层次的 tile，以匹配硬件的多级存储层次：

```cpp
// src/meta_schedule/schedule_rule/multi_level_tiling.cc
class MultiLevelTilingNode : public ScheduleRuleNode {
 public:
  /*! \brief The tiling structure, e.g., "SSRSRS" for:
   *   Space-Space-Reduce-Space-Reduce-Space
   *   对应 GPU 的: BlockTile-ThreadTile-WarpTile-VectorTile */
  String structure;

  /*! \brief The tile sizes to explore at each level */
  Array<Array<Integer>> tile_sizes;

  /*! \brief Whether to apply cache read/write */
  bool cache_read;
  bool cache_write;

  Array<tir::Schedule> Apply(const tir::Schedule& sch,
                              const tir::BlockRV& block) override;
};
```

**分块结构字符串** `structure` 的含义：

| 字符 | 含义 | 示例 |
|------|------|------|
| `S` | Space loop（空间循环） | 输出维度（如矩阵的行、列） |
| `R` | Reduce loop（规约循环） | 累加维度（如矩阵乘的 K 维度） |

常见的分块结构：

```
"SSRSRS"  → 适用于 GPU（Block → Thread → Warp → Vector）
"SRS"     → 适用于 CPU（外层空间 → 规约 → 内层空间）
"SSSRRSRS" → 适用于更复杂的存储层次
```

**分块因子的自动推导**：

对于一个长度为 $L$ 的循环，MultiLevelTiling 需要选择分块因子 $t_1, t_2, \ldots, t_k$ 使得：

$$L = t_1 \times t_2 \times \cdots \times t_k \times r$$

其中 $r$ 是剩余因子。MetaSchedule 使用**整数分解**来自动生成合法的因子组合：

```python
def generate_tile_factors(length, num_levels):
    """为给定长度的循环生成多级分块因子"""
    factors = []
    for combo in integer_partitions(length, num_levels):
        if is_valid_tiling(combo):  # 检查是否满足硬件约束
            factors.append(combo)
    return factors

# 例如，对于长度 1024 的循环，3 级分块：
# [64, 8, 2] → 64 * 8 * 2 = 1024
# [32, 16, 2] → 32 * 16 * 2 = 1024
# [128, 4, 2] → 128 * 4 * 2 = 1024
```

### 16.3.3 CrossThreadReduction：跨线程规约

对于 GPU 上的规约操作（如求和、求最大值），需要特殊处理以确保线程间的正确同步：

```cpp
// src/meta_schedule/schedule_rule/cross_thread_reduction.cc
class CrossThreadReductionNode : public ScheduleRuleNode {
 public:
  /*! \brief Whether to use warp-level reduction */
  bool use_warp_reduction;

  Array<tir::Schedule> Apply(const tir::Schedule& sch,
                              const tir::BlockRV& block) override {
    // 1. 识别规约循环
    Array<tir::LoopRV> reduce_loops = GetReductionLoops(sch, block);

    // 2. 将规约循环绑定到 threadIdx.x
    tir::Schedule result = sch.clone();
    result.bind(reduce_loops[0], "threadIdx.x");

    // 3. 插入同步屏障
    result.annotate(block, "thread_extent", /*...*/);

    // 4. 如果启用 warp 级规约，进一步优化
    if (use_warp_reduction) {
      ApplyWarpReduction(result, block);
    }

    return {result};
  }
};
```

跨线程规约的两种策略：

**策略一：Block 级规约（简单但低效）**

```
每个线程计算部分和 → __syncthreads() → 线程 0 汇总
```

**策略二：Warp 级规约（高效）**

```
每个 warp 内 shuffle 规约 → 跨 warp 汇总 → 最终结果
```

```python
# Warp 级规约的 TIR 表示
@T.prim_func
def warp_reduce(A: T.Buffer[(1024,), "float32"],
                B: T.Buffer[(1,), "float32"]):
    for i in T.thread_binding(1024, "threadIdx.x"):
        with T.block("reduce"):
            vi = T.axis.spatial(1024, i)
            T.reads(A[vi])
            T.writes(B[0])
            # Warp shuffle reduce
            T.tvm_warp_shuffle_down(T.float32(0), A[vi], 1)
```

### 16.3.4 AddRFactor：规约因子化

对于大规模规约操作，**AddRFactor** 规则通过引入中间 buffer 来减少规约的并行度需求：

```cpp
// src/meta_schedule/schedule_rule/add_rfactor.cc
class AddRFactorNode : public ScheduleRuleNode {
 public:
  /*! \brief The maximum number of threads for reduction */
  int max_threadblocks;

  Array<tir::Schedule> Apply(const tir::Schedule& sch,
                              const tir::BlockRV& block) override {
    // 1. 找到最外层的规约循环
    tir::LoopRV outer_reduce = GetOutermostReduceLoop(sch, block);

    // 2. 对规约循环进行因子分解
    // 原始: for k in range(1024): C[i,j] += A[i,k]*B[k,j]
    // 分解: for ko in range(32):
    //          for ki in range(32):
    //              C_rf[ko, i, j] += A[i, ko*32+ki] * B[ko*32+ki, j]
    //        for ko in range(32):
    //            C[i,j] += C_rf[ko, i, j]

    tir::Schedule result = sch.clone();
    result.rfactor(outer_reduce, /*axis=*/0);

    return {result};
  }
};
```

<div data-component="RFactorVisualization"></div>

### 16.3.5 AutoBind：自动线程绑定

**AutoBind** 规则自动将空间循环绑定到 GPU 线程层次结构：

```cpp
// src/meta_schedule/schedule_rule/auto_bind.cc
class AutoBindNode : public ScheduleRuleNode {
 public:
  /*! \brief The maximum number of threads per block */
  int max_threads_per_block;

  Array<tir::Schedule> Apply(const tir::Schedule& sch,
                              const tir::BlockRV& block) override {
    // 1. 获取所有空间循环
    Array<tir::LoopRV> space_loops = GetSpaceLoops(sch, block);

    // 2. 自动决定如何映射到 threadIdx/blockIdx
    //    策略: 将最内层的循环绑定到 threadIdx.x
    //          外层循环依次绑定到 threadIdx.y, blockIdx.x, ...
    tir::Schedule result = sch.clone();

    int64_t total_extent = 1;
    for (auto loop : space_loops) {
      total_extent *= sch.get(loop)->extent;
    }

    if (total_extent <= max_threads_per_block) {
      // 全部绑到线程
      for (int i = 0; i < space_loops.size(); i++) {
        result.bind(space_loops[i], ThreadAxis(i));
      }
    } else {
      // 需要分层: 部分绑 block，部分绑 thread
      SplitForBind(result, space_loops, max_threads_per_block);
    }

    return {result};
  }
};
```

### 16.3.6 ParallelVectorize：并行向量化

**ParallelVectorize** 规则尝试将最内层循环向量化以利用 SIMD 指令：

```cpp
// src/meta_schedule/schedule_rule/parallel_vectorize.cc
class ParallelVectorizeNode : public ScheduleRuleNode {
 public:
  /*! \brief The vector lengths to try */
  Array<Integer> vector_lens;

  Array<tir::Schedule> Apply(const tir::Schedule& sch,
                              const tir::BlockRV& block) override {
    Array<tir::Schedule> results;

    for (int vlen : vector_lens) {
      tir::Schedule candidate = sch.clone();
      tir::LoopRV innermost = GetInnermostLoop(sch, block);

      // 检查循环长度是否可以被向量长度整除
      int64_t extent = sch.get(innermost)->extent;
      if (extent % vlen != 0) continue;

      // 分割并绑定
      auto [outer, inner] = candidate.split(innermost, factors={extent / vlen, vlen});
      candidate.vectorize(inner);

      results.push_back(candidate);
    }

    return results;
  }
};
```

---

## 16.4 Postproc：后处理规则

### 16.4.1 Postproc 的设计

**Postproc**（Post-processing）规则在调度生成后对结果进行修正和验证。它们确保生成的调度代码满足硬件约束和正确性要求：

```cpp
// include/tvm/meta_schedule/postproc.h
class PostprocNode : public runtime::Object {
 public:
  /*!
   * \brief Apply postprocessing to the schedule.
   * \return true if the schedule is valid after postprocessing.
   */
  virtual bool Apply(const tir::Schedule& sch) = 0;

  virtual Postproc Clone() const = 0;
};
```

### 16.4.2 常见的 Postproc 规则

**VerifyGPUCode：GPU 代码验证**

验证生成的 GPU 代码满足硬件限制（如线程块大小、共享内存大小）：

```cpp
// src/meta_schedule/postproc/verify_gpu_code.cc
class VerifyGPUCodeNode : public PostprocNode {
 public:
  int max_shared_memory_per_block;
  int max_threads_per_block;
  int max_vector_bytes;

  bool Apply(const tir::Schedule& sch) override {
    // 遍历所有 block，检查 GPU 相关约束
    for (const auto& block : sch->GetBlocks()) {
      auto bounds = GetBlockBounds(sch, block);

      // 检查线程数限制
      if (bounds.num_threads > max_threads_per_block) return false;

      // 检查共享内存限制
      if (bounds.shared_memory > max_shared_memory_per_block) return false;

      // 检查向量化字节数
      if (bounds.vector_bytes > max_vector_bytes) return false;
    }
    return true;
  }
};
```

**RewriteCooperativeFetch：协作预取重写**

将共享内存的读取模式重写为协作预取（cooperative fetch），确保一个 warp 的所有线程协作加载数据：

```cpp
// src/meta_schedule/postproc/rewrite_cooperative_fetch.cc
class RewriteCooperativeFetchNode : public PostprocNode {
 public:
  bool Apply(const tir::Schedule& sch) override {
    // 查找所有从 global memory 到 shared memory 的读取
    for (const auto& block : sch->GetBlocks()) {
      if (IsSharedMemoryWrite(sch, block)) {
        // 重写为协作预取模式
        RewriteAsCooperativeFetch(sch, block);
      }
    }
    return true;
  }
};
```

**RewriteUnboundBlock：无界 block 重写**

当一个 block 的循环没有被绑定到任何线程时，自动添加默认的线程绑定：

```cpp
// src/meta_schedule/postproc/rewrite_unbound_block.cc
class RewriteUnboundBlockNode : public PostprocNode {
 public:
  bool Apply(const tir::Schedule& sch) override {
    for (const auto& block : sch->GetBlocks()) {
      if (IsGPUBlock(sch, block) && !HasThreadBinding(sch, block)) {
        // 自动绑定到 threadIdx.x
        AutoBindToThreadX(sch, block);
      }
    }
    return true;
  }
};
```

### 16.4.3 Postproc 的执行流程

Postproc 规则按照优先级顺序依次执行。如果任何规则失败，该调度方案将被丢弃：

```
生成调度 → Postproc 1 (VerifyGPUCode)
         → Postproc 2 (RewriteCooperativeFetch)
         → Postproc 3 (RewriteUnboundBlock)
         → 验证通过 → 加入候选池
         → 验证失败 → 丢弃
```

---

## 16.5 代价模型（Cost Model）

### 16.5.1 代价模型的作用

在 MetaSchedule 的搜索过程中，直接在硬件上执行每个候选调度方案来评估其性能是不现实的（一个典型的调优过程可能需要评估数万到数十万个候选方案）。代价模型的作用是**预测**每个候选方案的执行代价，从而避免实际执行：

$$\text{Cost}(\text{schedule}) \approx \text{Predicted by model}$$

代价模型的训练数据来源于：
1. **实际测量**：在搜索过程中定期在硬件上测量部分候选方案的真实性能
2. **特征提取**：从 TIR 程序和调度 trace 中提取特征

### 16.5.2 特征提取

MetaSchedule 从调度 trace 中提取特征向量，用于训练代价模型：

```cpp
// src/meta_schedule/cost_model/feature_extractor.cc
class PerNodeFeature {
 public:
  // 每个 TIR 节点的特征
  int64_t arith_intensity;      // 算术强度（FLOPs / Bytes）
  int64_t buffer_region_size;   // 访问的 buffer 区域大小
  int64_t unroll_factor;        // 展开因子
  int64_t vectorize_factor;     // 向量化因子
  int64_t thread_extent;        // 线程范围
  int64_t block_extent;         // Block 范围
  // ... 更多特征
};

class FeatureExtractor {
 public:
  /*! \brief Extract features from a TIR schedule */
  Feature Extract(const tir::Schedule& sch) {
    Feature feat;
    for (const auto& block : sch->GetBlocks()) {
      feat.per_node_features.push_back(ExtractBlockFeatures(sch, block));
    }
    feat.structural_features = ExtractStructuralFeatures(sch);
    return feat;
  }
};
```

关键特征包括：

| 特征类别 | 特征名 | 说明 |
|---------|--------|------|
| 计算特征 | arith_intensity | 算术强度（FLOPs / 内存访问字节数） |
| 访存特征 | working_set_size | 工作集大小（字节） |
| 访存特征 | reuse_distance | 数据复用距离 |
| 并行特征 | num_threads | 并行线程数 |
| 调度特征 | tile_size | 分块大小 |
| 调度特征 | unroll_factor | 循环展开因子 |
| 调度特征 | vectorize_factor | 向量化宽度 |
| 结构特征 | num_loops | 循环嵌套深度 |
| 结构特征 | num_blocks | Block 数量 |

### 16.5.3 XGBoost 代价模型

MetaSchedule 默认使用 **XGBoost** 作为代价模型，它是一个梯度提升树（Gradient Boosted Tree）模型：

```cpp
// src/meta_schedule/cost_model/xgb_model.cc
class XGBModelNode : public CostModelNode {
 public:
  /*! \brief The XGBoost model parameters */
  xgboost::Learner* learner;
  /*! \brief The training data buffer */
  std::vector<xgboost::DMatrix*> train_data;
  /*! \brief The number of training rounds per update */
  int num_training_rounds;

  void Update(const Array<MeasureInput>& inputs,
              const Array<MeasureResult>& results) override {
    // 1. 将测量结果转换为训练样本
    for (int i = 0; i < inputs.size(); i++) {
      Feature feat = extractor_->Extract(inputs[i]->sch);
      float label = results[i]->costs[0];  // 实际执行时间
      train_data.push_back(feat, label);
    }

    // 2. 增量训练 XGBoost 模型
    learner->UpdateOneIter(/*iter=*/num_updates++, train_data);
  }

  Array<CostResult> Predict(const Array<MeasureInput>& inputs) override {
    Array<CostResult> predictions;
    for (const auto& input : inputs) {
      Feature feat = extractor_->Extract(input->sch);
      float pred = learner->Predict(feat);
      predictions.push_back(CostResult(pred));
    }
    return predictions;
  }
};
```

**XGBoost 代价模型的优势**：

1. **非线性建模**：树模型可以捕捉特征之间的非线性交互
2. **特征重要性**：可以自动识别哪些特征对性能影响最大
3. **增量学习**：支持在线更新，随着搜索的进行模型越来越准
4. **快速推理**：树模型的推理速度非常快，适合大规模搜索

### 16.5.4 MLP 代价模型

除了 XGBoost，MetaSchedule 还支持基于**多层感知器（MLP）**的代价模型：

```cpp
// src/meta_schedule/cost_model/ml_p_model.cc
class MLPModelNode : public CostModelNode {
 public:
  /*! \brief The neural network model (via TVM's own NN library or external) */
  void* model;
  /*! \brief The hidden layer dimensions */
  Array<Integer> hidden_dims;
  /*! \brief The learning rate */
  float learning_rate;

  void Update(const Array<MeasureInput>& inputs,
              const Array<MeasureResult>& results) override {
    // 使用 PyTorch 或自定义的 NN 训练
    // 输入: 特征向量
    // 输出: 预测的执行时间
    for (int i = 0; i < inputs.size(); i++) {
      Feature feat = extractor_->Extract(inputs[i]->sch);
      float label = results[i]->costs[0];
      TrainStep(feat, label);
    }
  }
};
```

### 16.5.5 代价模型的训练策略

MetaSchedule 使用 **"先探索后利用"（Explore then Exploit）** 的策略：

```
阶段 1: 随机采样 + 实际测量（收集初始训练数据）
阶段 2: 训练代价模型
阶段 3: 用代价模型指导搜索（预测 → 选择最优候选 → 测量 → 更新模型）
重复阶段 2-3 直到搜索预算用完
```

$$\text{采样策略} = \begin{cases} \text{随机采样} & \text{with probability } \epsilon \\ \text{模型指导采样} & \text{with probability } 1 - \epsilon \end{cases}$$

---

## 16.6 进化搜索（Evolutionary Search）

### 16.6.1 进化搜索算法概述

MetaSchedule 使用**进化搜索**在巨大的搜索空间中高效寻优。进化搜索是一种受生物进化启发的优化算法，它维护一个**种群**（population），通过**变异**（mutation）和**选择**（selection）逐步改进种群的质量：

```
初始化种群（随机采样 N 个候选方案）
重复以下步骤直到预算用完：
    1. 评估种群中每个个体的适应度（用代价模型预测）
    2. 选择适应度最高的 K 个个体作为父代
    3. 对父代进行变异，生成新的子代
    4. 用子代替换种群中最差的个体
返回种群中适应度最高的个体
```

### 16.6.2 MetaSchedule 进化搜索的实现

```cpp
// src/meta_schedule/search_strategy/evolutionary_search.cc
class EvolutionarySearchNode : public SearchStrategyNode {
 public:
  /*! \brief The population size */
  int population_size;
  /*! \brief The number of individuals to select as parents */
  int num_parents;
  /*! \brief The mutation probability for each decision */
  double mutation_prob;
  /*! \brief The maximum number of evolution rounds */
  int max_rounds;

  Array<MeasureCandidate> GenerateCandidates() override {
    // 1. 初始化种群
    std::vector<Trace> population = InitializePopulation();

    for (int round = 0; round < max_rounds; round++) {
      // 2. 用代价模型评估种群
      std::vector<double> scores = cost_model_->Predict(population);

      // 3. 选择父代（锦标赛选择）
      std::vector<Trace> parents = TournamentSelect(
          population, scores, num_parents);

      // 4. 变异生成子代
      std::vector<Trace> children;
      for (auto& parent : parents) {
        Trace child = Mutate(parent);
        if (IsValid(child)) {
          children.push_back(child);
        }
      }

      // 5. 替换种群中最差的个体
      ReplaceWeakest(population, children, scores);
    }

    // 6. 返回种群中最优的候选方案
    return TopKFromPopulation(population, batch_size_);
  }

  Trace Mutate(const Trace& parent) {
    // 随机选择一个调度决策进行变异
    int idx = RandomInt(0, parent->instructions.size());

    // 获取该决策的所有可能选项
    Array<ObjectRef> options = GetDecisionOptions(parent, idx);

    // 随机选择一个不同的选项
    ObjectRef new_decision = RandomChoice(options);

    // 创建变异后的 trace
    return parent->WithDecision(idx, new_decision);
  }
};
```

### 16.6.3 变异操作的类型

进化搜索中的变异操作对应于对调度决策的修改：

**变异类型一：参数变异**

修改某个调度原语的参数（如分块因子）：

```
原始: Split(loop="i", factors=[64, 16])
变异: Split(loop="i", factors=[32, 32])
```

**变异类型二：结构变异**

添加或删除某个调度原语：

```
原始: [Split, Reorder, Bind]
变异: [Split, Reorder, Bind, Unroll]  # 添加 Unroll
```

**变异类型三：顺序变异**

改变调度原语的应用顺序：

```
原始: [Split(i), Split(j), Reorder(i0, j0, i1, j1)]
变异: [Split(j), Split(i), Reorder(j0, i0, j1, i1)]
```

### 16.6.4 种群初始化策略

种群初始化的质量直接影响搜索效率。MetaSchedule 使用两种初始化策略：

**策略一：随机采样**

从搜索空间中随机生成合法的调度方案：

```python
def random_sample(program, space_rules):
    """从搜索空间中随机采样一个调度方案"""
    sch = tir.Schedule(program)
    trace = Trace()

    for rule in space_rules:
        # 获取所有可应用的 block
        blocks = rule.applicable_blocks(sch)
        for block in blocks:
            # 随机选择一个合法的变换
            options = rule.get_options(sch, block)
            choice = random.choice(options)
            rule.apply(sch, block, choice)
            trace.record(rule, block, choice)

    return sch, trace
```

**策略二：种子注入**

从已知的高效调度方案（如 AutoTVM 的模板调优结果）中注入：

```python
def inject_seeds(population, seeds):
    """将已知的高效方案注入种群"""
    for seed in seeds:
        population.append(seed)
    # 用随机方案填充剩余位置
    while len(population) < population_size:
        population.append(random_sample())
```

### 16.6.5 锦标赛选择（Tournament Selection）

MetaSchedule 使用锦标赛选择来挑选父代个体：

```python
def tournament_select(population, scores, num_parents, tournament_size=3):
    """锦标赛选择"""
    parents = []
    for _ in range(num_parents):
        # 随机选择 tournament_size 个个体
        candidates = random.sample(range(len(population)), tournament_size)
        # 选择其中适应度最高的
        best = max(candidates, key=lambda i: scores[i])
        parents.append(population[best])
    return parents
```

锦标赛选择的优势：
1. **多样性保持**：不会像轮盘赌选择那样过度偏向高适应度个体
2. **参数可调**：通过调整 `tournament_size` 控制选择压力
3. **实现简单**：不需要对适应度进行归一化

<div data-component="EvolutionarySearchVisualization"></div>

---

## 16.7 端到端调优流程

### 16.7.1 MetaSchedule 的 Python API

MetaSchedule 提供了简洁的 Python API 来执行端到端的自动调优：

```python
import tvm
from tvm import meta_schedule as ms
from tvm import relay

# 1. 加载模型
mod, params = relay.frontend.from_onnx(onnx_model)

# 2. 配置调优参数
config = ms.TuneConfig(
    strategy="evolutionary",        # 搜索策略
    num_trials_per_iter=64,         # 每轮采样数
    max_trials_per_task=10000,      # 每个任务的最大测量次数
    max_trials_global=20000,        # 全局最大测量次数
)

# 3. 创建搜索空间生成器
space = ms.space_generator.SpaceGenerator(
    schedule_rules=[
        ms.schedule_rule.MultiLevelTiling(
            structure="SSRSRS",
            tile_sizes=[[1, 2, 4, 8], [1, 2, 4]],
        ),
        ms.schedule_rule.AutoBind(max_threads=1024),
        ms.schedule_rule.ParallelVectorize(vector_lens=[4, 8]),
    ],
    postprocs=[
        ms.postproc.VerifyGPUCode(
            max_shared_memory_per_block=48 * 1024,
            max_threads_per_block=1024,
        ),
    ],
)

# 4. 执行调优
database = ms.tune_relay(
    mod=mod,
    params=params,
    target="cuda",
    config=config,
    space=space,
    work_dir="./tuning_logs",
)

# 5. 编译最优结果
lib = ms.compile_relay(
    mod=mod,
    params=params,
    target="cuda",
    database=database,
)
```

### 16.7.2 调优数据库（Tuning Database）

MetaSchedule 将所有测量过的调度方案及其性能存储在**调优数据库**中：

```python
# 调优数据库的结构
class Database:
    """存储所有调优记录的数据库"""
    records: List[Record]

class Record:
    """单条调优记录"""
    workload: Workload          # 工作负载（TIR 程序的哈希）
    trace: Trace                # 调度轨迹
    run_secs: List[float]       # 实际运行时间
    target: Target              # 目标硬件
    arg_info: List[TensorType]  # 参数类型信息
```

数据库支持以下查询操作：

```python
# 查询最优的调度方案
best_record = database.query_best(workload_hash, target)

# 查询相似工作负载的调度方案
similar_records = database.query_topk(workload_hash, target, k=10)

# 导出所有记录为 JSON
database.save("tuning_records.json")

# 从 JSON 导入记录
database = Database.load("tuning_records.json")
```

### 16.7.3 TIR 级调优（不经过 Relay）

MetaSchedule 也支持直接在 TIR 级别进行调优，跳过 Relay 层：

```python
from tvm import meta_schedule as ms
from tvm.script import tir as T

# 定义 TIR 程序
@T.prim_func
def matmul(
    A: T.Buffer[(1024, 1024), "float32"],
    B: T.Buffer[(1024, 1024), "float32"],
    C: T.Buffer[(1024, 1024), "float32"],
) -> None:
    for i, j, k in T.grid(1024, 1024, 1024):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

# 在 TIR 级别调优
sch = ms.tune_tir(
    func=matmul,
    target="cuda",
    config=ms.TuneConfig(max_trials_per_task=2000),
    work_dir="./tir_tuning_logs",
)

# 获取最优调度
print(sch.mod())
```

### 16.7.4 分布式调优

MetaSchedule 支持通过 RPC 进行分布式调优，将测量任务分发到多个设备上：

```python
# 配置远程设备
runner = ms.runner.RPCRunner(
    rpc_config=ms.runner.RPCConfig(
        tracker_host="192.168.1.100",
        tracker_port=9190,
        target_devices=[
            "cuda -keys=cuda,gpu -max_num_threads=1024 -arch=sm_80",
        ],
    ),
    max_workers=4,  # 并行测量的 worker 数量
)

# 使用分布式 runner 执行调优
database = ms.tune_relay(
    mod=mod,
    params=params,
    target="cuda",
    config=config,
    runner=runner,
)
```

---

## 16.8 MetaSchedule 的扩展机制

### 16.8.1 自定义 ScheduleRule

用户可以实现自定义的 ScheduleRule：

```python
from tvm.meta_schedule import ScheduleRule

@tvm._ffi.register_object("meta_schedule.CustomRule")
class MyCustomRule(ScheduleRule):
    """自定义调度规则"""

    def __init__(self, my_param):
        self.__init_handle_by_constructor__(
            _ffi_api.ScheduleRuleCustomRule, my_param)

    def apply(self, sch, block):
        """应用自定义调度规则"""
        # 获取循环结构
        loops = sch.get_loops(block)

        # 自定义分块策略
        if len(loops) >= 3:
            outer, inner = sch.split(loops[-1], factors=[None, 8])
            sch.vectorize(inner)

        return [sch]
```

### 16.8.2 自定义代价模型

用户可以实现自定义的代价模型：

```python
from tvm.meta_schedule import CostModel

class MyCostModel(CostModel):
    """自定义代价模型"""

    def __init__(self):
        self.model = train_my_model()

    def update(self, inputs, results):
        """用新的测量结果更新模型"""
        for inp, res in zip(inputs, results):
            features = extract_features(inp.sch)
            self.model.train_step(features, res.costs[0])

    def predict(self, inputs):
        """预测候选方案的代价"""
        predictions = []
        for inp in inputs:
            features = extract_features(inp.sch)
            pred = self.model.predict(features)
            predictions.append(pred)
        return predictions
```

### 16.8.3 自定义搜索策略

用户可以实现自定义的搜索策略：

```python
from tvm.meta_schedule import SearchStrategy

class MySearchStrategy(SearchStrategy):
    """自定义搜索策略"""

    def generate_candidates(self, num_candidates):
        """生成候选调度方案"""
        candidates = []
        for _ in range(num_candidates):
            # 使用自定义的搜索逻辑
            sch = self.my_search_logic()
            candidates.append(MeasureCandidate(sch))
        return candidates
```

---

## 16.9 性能分析与调优技巧

### 16.9.1 搜索效率分析

MetaSchedule 的搜索效率可以通过以下指标衡量：

| 指标 | 说明 | 典型值 |
|------|------|--------|
| 收敛速度 | 达到最优性能 95% 所需的测量次数 | $10^2 \sim 10^3$ |
| 搜索覆盖度 | 搜索空间中被探索的比例 | $10^{-8} \sim 10^{-5}$ |
| 模型准确度 | 代价模型预测的相关性 | $\rho > 0.8$ |
| 最终性能 | 相对于手动调优的性能比 | 0.9 ~ 1.0 |

### 16.9.2 常见调优技巧

**技巧一：预热阶段的采样数量**

初始预热阶段需要足够的随机采样来训练代价模型：

```python
config = ms.TuneConfig(
    # 预热采样数应为搜索空间复杂度的函数
    num_trials_per_iter=64,
    # 建议: 预热采样数 ≈ 100 ~ 500
)
```

**技巧二：搜索空间的裁剪**

通过约束条件减小搜索空间，提高搜索效率：

```python
# 约束分块因子必须是 2 的幂
ms.schedule_rule.MultiLevelTiling(
    structure="SSRSRS",
    tile_sizes=[
        [1, 2, 4, 8, 16, 32, 64],  # 只考虑 2 的幂
    ],
)
```

**技巧三：多任务共享调优记录**

相似的算子可以共享调优记录：

```python
# 创建共享数据库
database = ms.database.create("json", work_dir="./shared_db")

# 多个调优任务共享同一个数据库
for task in tasks:
    ms.tune_tir(
        func=task.func,
        target=task.target,
        database=database,  # 共享数据库
    )
```

---

## 16.10 搜索空间的数学分析

### 16.10.1 搜索空间大小的理论估算

对于一个具有 $L$ 层循环嵌套、每层循环长度为 $N_i$ 的 TIR 程序，MetaSchedule 的搜索空间大小可以通过以下公式估算：

$$|\text{Space}| = \prod_{i=1}^{L} \left( \sum_{k=1}^{K} \binom{d(N_i, k)}{1} \right) \times |\text{Reorder}(L)| \times |\text{Annotate}|$$

其中：
- $d(N_i, k)$ 表示将长度 $N_i$ 分成 $k$ 段的合法分方案数
- $|\text{Reorder}(L)| = L!$ 表示循环重排的可能数
- $|\text{Annotate}|$ 表示注解（如 unroll、vectorize）的可能选择数

**具体示例**：一个矩阵乘法 $C = A \times B$（$M=N=K=1024$）的搜索空间：

```
三重循环: for i, j, k in grid(1024, 1024, 1024)

分块选项 (每个维度):
  i: 可分为 [1,2,4,8,16,32,64,128,256,512,1024] 的因子组合
  j: 同上
  k: 同上

对于 2 级分块: 每个维度约 10 × 10 = 100 种组合
三个维度: 100^3 = 10^6 种分块方案

重排: 3! = 6 种

注解: 2^3 = 8 种 (每个循环可选 unroll/vectorize/parallel)

总计: 10^6 × 6 × 8 ≈ 5 × 10^7 种
```

对于更复杂的算子（如多头注意力），搜索空间可达 $10^{15}$ 以上。

### 16.10.2 搜索空间的结构特性

MetaSchedule 的搜索空间具有以下数学特性：

**层次性（Hierarchical）**：搜索空间可以分解为多个层次，每个层次对应不同的调度粒度：

```
层次 1: 选择应用哪些 ScheduleRule（如 MultiLevelTiling, AutoBind）
层次 2: 选择每个 Rule 的参数（如分块因子、线程绑定方式）
层次 3: 选择 Postproc 规则（如是否重写协作预取）
```

**约束性（Constrained）**：许多参数组合是非法的，需要满足硬件约束：

$$\text{合法空间} \subset \text{理论空间}$$

约束条件包括：
- 线程数不超过硬件限制：$\prod_{\text{bound axes}} t_i \leq T_{\max}$
- 共享内存不超过限制：$\text{shared\_mem}(\text{tiles}) \leq S_{\max}$
- 分块因子整除循环长度：$N_i \mod t_i = 0$

**连续性（Continuous）**：相邻的调度方案通常具有相似的性能，这为局部搜索提供了基础：

$$\text{如果 } s_1 \text{ 和 } s_2 \text{ 仅在一个决策点不同，则 } |\text{Perf}(s_1) - \text{Perf}(s_2)| \text{ 通常较小}$$

### 16.10.3 搜索效率的理论分析

进化搜索的效率可以用以下模型分析：

设搜索空间大小为 $N$，种群大小为 $P$，每轮变异生成 $C$ 个子代，则经过 $G$ 轮进化后，找到全局最优的概率为：

$$P(\text{找到最优}) \geq 1 - \left(1 - \frac{P}{N}\right)^G$$

对于 $N = 10^{10}$，$P = 100$，$G = 1000$：

$$P(\text{找到最优}) \geq 1 - \left(1 - 10^{-8}\right)^{100000} \approx 0.001$$

这意味着进化搜索不太可能找到绝对最优解，但代价模型的指导可以大幅提高搜索效率：

$$P(\text{找到近似最优}) \approx 1 - e^{-\alpha \cdot G / N^{1/d}}$$

其中 $d$ 是搜索空间的有效维度，$\alpha$ 是代价模型的指导效率系数。

---

## 16.11 MetaSchedule 的高级特性

### 16.11.1 多任务调优（Multi-Task Tuning）

MetaSchedule 支持多个相关任务共享调优知识：

```python
# 多任务调优：多个相似的算子共享代价模型
from tvm.meta_schedule import TuneContext

# 创建共享的代价模型
shared_cost_model = ms.cost_model.XGBModel(
    num_warmup_samples=100,
    num_samples_per_iter=64,
)

# 为每个任务创建调优上下文
tasks = []
for workload in workloads:
    ctx = TuneContext(
        target=target,
        space_generator=ms.space_generator.PostOrderApply(),
        search_strategy=ms.search_strategy.ReplayTrace(),
        cost_model=shared_cost_model,  # 共享代价模型
    )
    tasks.append(ctx)

# 联合调优
database = ms.tune_tasks(
    tasks=tasks,
    config=config,
    work_dir="./multi_task_logs",
)
```

多任务调优的优势在于：
1. **知识迁移**：在一个任务上学到的代价模型可以指导其他任务的搜索
2. **样本效率**：共享训练数据，减少总测量次数
3. **一致性**：相似算子获得一致的优化策略

### 16.11.2 自动调度规则选择

MetaSchedule 可以自动选择最合适的调度规则组合：

```python
# 自动选择调度规则
space_gen = ms.space_generator.PostOrderApply(
    sch_rules="auto",  # 自动选择
    postprocs="auto",  # 自动选择
)
```

自动选择的逻辑：

```cpp
// src/meta_schedule/space_generator/post_order_apply.cc
Array<ScheduleRule> AutoSelectScheduleRules(const tir::PrimFunc& func,
                                             const Target& target) {
  Array<ScheduleRule> rules;

  // 分析程序特征
  bool has_reduce = HasReductionLoop(func);
  bool is_gpu = target->GetTargetDeviceType() == kDLGPU;
  int num_loops = CountLoops(func);

  if (is_gpu) {
    // GPU 目标：添加多级分块和线程绑定规则
    rules.push_back(MultiLevelTiling("SSRSRS"));
    rules.push_back(AutoBind(1024));
    rules.push_back(CrossThreadReduction());
  } else {
    // CPU 目标：添加并行和向量化规则
    rules.push_back(MultiLevelTiling("SRS"));
    rules.push_back(ParallelVectorize({4, 8}));
  }

  if (has_reduce) {
    rules.push_back(AddRFactor());
  }

  return rules;
}
```

### 16.11.3 Trace 的高级操作

Trace 支持多种高级操作，用于搜索空间的变异和探索：

```python
# 1. Trace 的组合（Composition）
trace_a = sch.trace()  # 初始调度的 trace
trace_b = ApplyMutation(trace_a, mutation_params)  # 变异后的 trace

# 2. Trace 的剪裁（Pruning）
# 移除效果不大的调度决策
pruned_trace = PruneTrace(trace, impact_threshold=0.01)

# 3. Trace 的拼接（Concatenation）
# 将两个不同阶段的调度决策拼接
combined = ConcatenateTraces(trace_stage1, trace_stage2)

# 4. Trace 的分析（Analysis）
# 分析每个调度决策对性能的影响
for i, instruction in enumerate(trace):
    impact = EstimateImpact(trace, i)
    print(f"Step {i}: {instruction}, Impact: {impact:.3f}")
```

### 16.11.4 调优记录的重用与迁移

```python
# 从之前调优中加载记录
database = ms.database.JSONDatabase(
    path_workload="workloads.json",
    path_tuning_record="tuning_records.json",
)

# 在新硬件上调优（迁移学习）
# 旧代价模型的参数作为初始化
new_cost_model = ms.cost_model.XGBModel(
    num_warmup_samples=50,  # 少量预热样本
    init_model=old_model,   # 从旧模型初始化
)

# 使用迁移学习进行调优
database_new = ms.tune_tir(
    func=matmul,
    target=new_target,
    cost_model=new_cost_model,
    database=database,  # 导入旧记录
)
```

---

## 16.12 实战案例：ResNet-50 端到端调优

### 16.12.1 完整的调优脚本

以下是一个完整的 ResNet-50 模型在 GPU 上使用 MetaSchedule 调优的示例：

```python
import tvm
from tvm import relay, meta_schedule as ms
import onnx

# 1. 加载 ONNX 模型
onnx_model = onnx.load("resnet50.onnx")
mod, params = relay.frontend.from_onnx(
    onnx_model,
    shape={"input": (1, 3, 224, 224)},
    dtype="float32",
)

# 2. 定义优化 Pass 序列
seq = tvm.transform.Sequential([
    relay.transform.InferType(),
    relay.transform.FuseOps(fuse_opt_level=2),
    relay.transform.ToBasicBlockNormalForm(),
])
mod = seq(mod)

# 3. 配置 MetaSchedule
config = ms.TuneConfig(
    strategy="evolutionary",
    num_trials_per_iter=64,
    max_trials_per_task=2000,
    max_trials_global=20000,
)

# 4. 定义搜索空间
space = ms.space_generator.SpaceGenerator(
    schedule_rules=[
        ms.schedule_rule.MultiLevelTiling(
            structure="SSRSRS",
            tile_sizes=[
                [1, 2, 4, 8, 16, 32],
                [1, 2, 4, 8],
            ],
            cache_read=True,
            cache_write=True,
        ),
        ms.schedule_rule.AutoBind(max_threads=1024),
        ms.schedule_rule.CrossThreadReduction(
            use_warp_reduction=True,
        ),
        ms.schedule_rule.ParallelVectorize(
            vector_lens=[4, 8],
        ),
    ],
    postprocs=[
        ms.postproc.VerifyGPUCode(
            max_shared_memory_per_block=48 * 1024,
            max_threads_per_block=1024,
            max_vector_bytes=16,
        ),
        ms.postproc.RewriteCooperativeFetch(),
        ms.postproc.RewriteUnboundBlock(),
    ],
)

# 5. 执行调优
database = ms.tune_relay(
    mod=mod,
    params=params,
    target="cuda",
    config=config,
    space=space,
    work_dir="./resnet50_tuning",
)

# 6. 编译最优结果
lib = ms.compile_relay(
    mod=mod,
    params=params,
    target="cuda",
    database=database,
)

# 7. 导出编译库
lib.export_library("resnet50_cuda.tar")
```

### 16.12.2 调优过程分析

在调优过程中，MetaSchedule 会经历以下阶段：

```
阶段 1: 预热 (Warmup) - 前 100 次测量
  ├── 随机采样各种调度方案
  ├── 在 GPU 上实际测量性能
  └── 收集初始训练数据

阶段 2: 模型训练 (Training)
  ├── 用收集的数据训练 XGBoost 代价模型
  └── 模型开始能够区分好的和差的调度方案

阶段 3: 指导搜索 (Guided Search) - 100~2000 次
  ├── 代价模型预测每个候选方案的性能
  ├── 选择预测最优的方案进行实际测量
  ├── 进化搜索在有希望的区域深入探索
  └── 模型持续更新，预测越来越准

阶段 4: 收敛 (Convergence) - 最后 500 次
  ├── 性能改进趋于平稳
  ├── 搜索集中在最优区域附近
  └── 最终找到接近最优的调度方案
```

### 16.12.3 调优结果分析

```python
# 分析调优结果
import json

# 加载调优记录
with open("resnet50_tuning/tuning_records.json") as f:
    records = json.load(f)

# 分析性能分布
costs = [r["run_secs"] for r in records]
print(f"总测量次数: {len(costs)}")
print(f"最优性能: {min(costs):.4f} ms")
print(f"最差性能: {max(costs):.4f} ms")
print(f"平均性能: {sum(costs)/len(costs):.4f} ms")
print(f"性能方差: {np.std(costs):.4f} ms")

# 找到最优的调度方案
best_idx = costs.index(min(costs))
best_record = records[best_idx]
print(f"\n最优调度方案:")
print(f"  Workload: {best_record['workload']}")
print(f"  Trace: {best_record['trace']}")
```

### 16.12.4 常见问题排查

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| 调优不收敛 | 搜索空间太大 | 减少 tile_sizes 的选项范围 |
| 性能不理想 | 代价模型不准确 | 增加预热采样数 |
| 编译错误 | Postproc 未正确过滤 | 检查 VerifyGPUCode 的约束 |
| 内存不足 | 种群太大 | 减小 population_size |
| 测量超时 | RPC 连接问题 | 检查网络和设备状态 |

---

## 16.13 MetaSchedule 与 AutoTVM 的详细对比

### 16.13.1 搜索空间构建方式对比

| 维度 | AutoTVM | MetaSchedule |
|------|---------|--------------|
| 空间定义 | 手动模板 `cfg.define_knob()` | 自动生成 `ScheduleRule` |
| 空间大小 | $10^3 \sim 10^5$ | $10^{10} \sim 10^{20}$ |
| 空间质量 | 依赖专家经验 | 自动覆盖所有合法变换 |
| 可扩展性 | 需要为每个算子写模板 | 通用规则自动适配 |
| 组合能力 | 无法组合 | 多个 Rule 级联组合 |

### 16.13.2 搜索算法对比

| 维度 | AutoTVM | MetaSchedule |
|------|---------|--------------|
| 搜索算法 | XGBoost + 随机采样 | 进化搜索 + 代价模型 |
| 代价模型 | XGBoost (特征工程) | XGBoost/MLP (自动特征) |
| 特征提取 | 手动设计的特征 | 自动从 Trace 提取 |
| 搜索效率 | 低（随机探索） | 高（模型指导） |
| 收敛速度 | 较快（小空间） | 较慢但质量更高 |

### 16.13.3 运行时对比

```python
# AutoTVM 调优脚本
import tvm
from tvm import autotvm

@autotvm.template("matmul")
def matmul_template(N):
    cfg = autotvm.get_config()
    cfg.define_knob("tile_x", [32, 64, 128])
    cfg.define_knob("tile_y", [32, 64, 128])
    cfg.define_knob("tile_k", [32, 64])
    # ... 手动模板代码

# MetaSchedule 调优脚本（无需模板）
sch = ms.tune_tir(
    func=matmul_func,
    target="cuda",
    config=ms.TuneConfig(max_trials_per_task=2000),
)
```

### 16.13.4 性能对比实验

在 ResNet-50 上的典型性能对比：

| 方法 | 调优时间 | 测量次数 | 最终性能 (ms) | 相对性能 |
|------|---------|---------|--------------|---------|
| 手动调优 | 数周 | N/A | 1.20 | 1.00x |
| AutoTVM | 4 小时 | 10000 | 1.45 | 0.83x |
| MetaSchedule | 2 小时 | 5000 | 1.25 | 0.96x |
| MetaSchedule (大规模) | 8 小时 | 20000 | 1.18 | 1.02x |

<div data-component="AutoTVMvsMetaSchedulePerformance"></div>

---

## 16.14 MetaSchedule 的内部实现细节

### 16.14.1 MeasureInput 与 MeasureResult

MetaSchedule 的测量系统使用 `MeasureInput` 和 `MeasureResult` 来记录输入输出：

```cpp
// include/tvm/meta_schedule/measure_candidate.h
class MeasureInputNode : public Object {
 public:
  /*! \brief The task to be measured */
  Task task;
  /*! \brief The schedule to be measured */
  tir::Schedule sch;
  /*! \brief The trace of scheduling decisions */
  Trace trace;
  /*! \brief The build result (compiled module) */
  runtime::Module build_result;
};

class MeasureResultNode : public Object {
 public:
  /*! \brief The run time costs (in seconds) */
  Array<FloatImm> costs;
  /*! \brief The error message if failed */
  String error_msg;
  /*! \brief The error from compilation */
  String error_from_compile;
  /*! \brief All no error */
  bool no_error;
};
```

### 16.14.2 Builder 与 Runner

MetaSchedule 将测量过程分为两个阶段：**Build**（编译）和 **Run**（执行）：

```cpp
// src/meta_schedule/builder/local_builder.cc
class LocalBuilderNode : public BuilderNode {
 public:
  /*! \brief The number of builder processes */
  int max_build_workers;

  Array<BuildResult> Build(const Array<MeasureInput>& inputs) override {
    // 1. 将 TIR 编译为可执行模块
    Array<BuildResult> results;
    for (const auto& input : inputs) {
      try {
        // 调用 TVM 的 build 系统
        runtime::Module mod = BuildModule(input->sch->mod(), target_);
        results.push_back(BuildResult(mod, ""));
      } catch (const std::exception& e) {
        results.push_back(BuildResult(nullptr, e.what()));
      }
    }
    return results;
  }
};

// src/meta_schedule/runner/local_runner.cc
class LocalRunnerNode : public RunnerNode {
 public:
  /*! \brief The number of repeat measurements */
  int repeat;
  /*! \brief The number of warmup iterations */
  int number;
  /*! \brief The minimum run time */
  double min_repeat_ms;

  Array<MeasureResult> Run(
      const Array<MeasureInput>& inputs,
      const Array<BuildResult>& build_results) override {
    Array<MeasureResult> results;

    for (size_t i = 0; i < inputs.size(); i++) {
      if (!build_results[i]->success) {
        results.push_back(MeasureResult({-1.0}, build_results[i]->error_msg));
        continue;
      }

      // 在本地设备上执行
      try {
        runtime::Module mod = build_results[i]->module;
        PackedFunc func = mod.GetFunction("main");

        // 预热
        for (int w = 0; w < number; w++) {
          func.CallPacked(args);
        }

        // 正式测量
        auto start = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < repeat; r++) {
          func.CallPacked(args);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double time = std::chrono::duration<double>(end - start).count() / repeat;
        results.push_back(MeasureResult({time}, ""));
      } catch (const std::exception& e) {
        results.push_back(MeasureResult({-1.0}, e.what()));
      }
    }

    return results;
  }
};
```

### 16.14.3 RPC Runner（远程测量）

RPC Runner 支持在远程设备上进行测量，这对于嵌入式设备和远程 GPU 非常有用：

```python
# 配置 RPC Runner
runner = ms.runner.RPCRunner(
    rpc_config=ms.runner.RPCConfig(
        tracker_host="0.0.0.0",
        tracker_port=9190,
        key="android_gpu",
        session_timeout_sec=60,
    ),
    max_workers=4,
    repeat=3,
    number=10,
    min_repeat_ms=500,
)

# 启动 RPC Tracker（在另一台机器上）
# python -m tvm.exec.rpc_tracker --host 0.0.0.0 --port 9190

# 启动 RPC Server（在目标设备上）
# python -m tvm.exec.rpc_server --tracker 0.0.0.0:9190 --key android_gpu
```

### 16.14.4 MeasureCallback 回调机制

```python
# 自定义测量回调
class MyMeasureCallback(ms.measure_callback.MeasureCallback):
    def callback(self, measure_inputs, measure_results):
        """每次测量完成后的回调"""
        for inp, res in zip(measure_inputs, measure_results):
            if res.no_error:
                print(f"Measurement: {res.costs[0]:.4f} ms")
            else:
                print(f"Error: {res.error_msg}")

# 注册回调
config = ms.TuneConfig(
    max_trials_per_task=1000,
    measure_callbacks=[MyMeasureCallback()],
)
```

### 16.14.5 并行测量机制

```cpp
// src/meta_schedule/runner/pooled_runnable.cc
class PooledRunner {
 public:
  void RunParallel(const Array<MeasureInput>& inputs) {
    // 创建线程池
    std::vector<std::thread> workers;
    std::atomic<int> next_task(0);

    for (int i = 0; i < max_workers_; i++) {
      workers.emplace_back([&, this]() {
        while (true) {
          int task_idx = next_task.fetch_add(1);
          if (task_idx >= inputs.size()) break;

          // 执行单个测量
          MeasureResult result = RunSingle(inputs[task_idx]);
          results_[task_idx] = result;
        }
      });
    }

    // 等待所有任务完成
    for (auto& worker : workers) {
      worker.join();
    }
  }
};
```

---

## 16.15 搜索空间裁剪与约束

### 16.15.1 约束传播

MetaSchedule 使用约束传播来减少搜索空间：

```python
def propagate_constraints(space, hardware_limits):
    """根据硬件约束裁剪搜索空间"""

    # 约束 1: 线程数不超过最大值
    max_threads = hardware_limits.get("max_threads_per_block", 1024)
    space = filter_by_constraint(
        space,
        lambda config: config.num_threads <= max_threads
    )

    # 约束 2: 共享内存不超过限制
    max_shared_mem = hardware_limits.get("max_shared_memory", 48 * 1024)
    space = filter_by_constraint(
        space,
        lambda config: config.shared_memory <= max_shared_mem
    )

    # 约束 3: 寄存器使用不超过限制
    max_registers = hardware_limits.get("max_registers", 255)
    space = filter_by_constraint(
        space,
        lambda config: config.registers_per_thread <= max_registers
    )

    return space
```

### 16.15.2 对称性消除

许多调度方案在功能上是等价的（对称的），消除这些对称性可以大幅减小搜索空间：

```python
def eliminate_symmetry(candidates):
    """消除对称的调度方案"""
    unique = []
    seen = set()

    for candidate in candidates:
        # 计算调度的规范化表示
        canonical = canonicalize_schedule(candidate)

        if canonical not in seen:
            seen.add(canonical)
            unique.append(candidate)

    return unique

def canonicalize_schedule(schedule):
    """规范化调度表示"""
    # 重命名循环变量为标准形式
    # 消除顺序无关的操作
    # ...
    return normalized_form
```

### 16.15.3 启发式搜索空间引导

```python
def heuristic_space_guidance(program, hardware_info):
    """基于启发式规则引导搜索空间"""

    rules = []

    # 规则 1: 矩阵乘法应该使用分块
    if is_matmul(program):
        rules.append({
            "type": "tiling",
            "preferred_sizes": [16, 32, 64, 128],
            "max_levels": 3,
        })

    # 规则 2: 卷积应该使用 im2col 或 Winograd
    if is_conv2d(program):
        rules.append({
            "type": "algorithm",
            "options": ["im2col", "winograd", "direct"],
        })

    # 规则 3: 规约操作应该使用 warp 级规约
    if has_reduction(program) and is_gpu(hardware_info):
        rules.append({
            "type": "reduction",
            "strategy": "warp_shuffle",
            "require_sync": True,
        })

    return rules
```

---

## 16.16 MetaSchedule 的错误处理与恢复

### 16.16.1 编译错误处理

```python
class CompilationErrorHandler:
    """编译错误处理"""

    def handle_error(self, error, context):
        """处理编译错误"""
        error_type = type(error).__name__

        if error_type == "ScheduleError":
            # 调度错误：无效的调度操作
            return self._handle_schedule_error(error, context)
        elif error_type == "VerifyError":
            # 验证错误：GPU 代码不满足约束
            return self._handle_verify_error(error, context)
        elif error_type == "BuildError":
            # 构建错误：代码生成失败
            return self._handle_build_error(error, context)
        else:
            # 未知错误
            return self._handle_unknown_error(error, context)

    def _handle_schedule_error(self, error, context):
        """处理调度错误"""
        # 记录错误并跳过该候选方案
        logging.warning(f"Schedule error: {error}")
        return None

    def _handle_verify_error(self, error, context):
        """处理验证错误"""
        # 调整约束并重试
        adjusted = self._adjust_constraints(error, context)
        return adjusted
```

### 16.16.2 运行时错误恢复

```cpp
// 运行时错误恢复机制
class RuntimeErrorHandler {
 public:
  MeasureResult RunWithRecovery(
      const MeasureInput& input,
      int max_retries = 3) {
    for (int retry = 0; retry < max_retries; retry++) {
      try {
        return Run(input);
      } catch (const TimeoutError& e) {
        // 超时：减少测量次数重试
        LOG(WARNING) << "Timeout, retrying with fewer repeats";
        repeat_ = std::max(1, repeat_ / 2);
      } catch (const DeviceError& e) {
        // 设备错误：等待并重试
        LOG(WARNING) << "Device error, waiting...";
        std::this_thread::sleep_for(std::chrono::seconds(5));
      } catch (const std::exception& e) {
        // 其他错误：记录并返回错误结果
        return MeasureResult({-1.0}, e.what());
      }
    }
    return MeasureResult({-1.0}, "Max retries exceeded");
  }
};
```

---

## 16.17 MetaSchedule 与其他调优框架的对比

### 16.17.1 与 TVM AutoScheduler (Ansor) 的关系

MetaSchedule 实际上是 Ansor 的演进版本：

| 特性 | Ansor (旧) | MetaSchedule (新) |
|------|-----------|------------------|
| 代码位置 | `src/auto_schedule/` | `src/meta_schedule/` |
| 搜索策略 | 仅进化搜索 | 可插拔（进化/随机/重放） |
| 代价模型 | XGBoost only | 可插拔（XGBoost/MLP/Random） |
| 调度规则 | 硬编码 | 可插拔（ScheduleRule） |
| 后处理 | 硬编码 | 可插拔（Postproc） |
| 扩展性 | 有限 | 高度可扩展 |

### 16.17.2 与 Halide AutoSchedule 的对比

| 特性 | Halide AutoSchedule | MetaSchedule |
|------|-------------------|--------------|
| 目标领域 | 图像处理 | 通用深度学习 |
| 搜索空间 | 基于规则 | 自动生成 |
| 代价模型 | 线性模型 | 非线性（XGBoost） |
| GPU 支持 | 有限 | 完善 |
| 多后端 | CPU/GPU | 全平台 |

### 16.17.3 与 TorchDynamo/Inductor 的对比

| 特性 | TorchInductor | MetaSchedule |
|------|--------------|--------------|
| 集成方式 | PyTorch 原生 | 独立编译器 |
| 代码生成 | Triton/C++ | TIR→多后端 |
| 调优方式 | 规则驱动 | 搜索驱动 |
| 扩展性 | 受限于 PyTorch | 高度可扩展 |
| 性能 | 接近最优 | 最优（经过搜索） |

---

## 16.10 本章小结

本章深入探讨了 MetaSchedule（Ansor）无模板调优系统的核心设计与实现：

| 概念 | 作用 | 关键源码 |
|------|------|---------|
| Program | 待优化的 TIR 程序 | `src/meta_schedule/module.cc` |
| Schedule | 调度状态封装 | `src/meta_schedule/schedule.cc` |
| Trace | 调度决策轨迹 | `src/meta_schedule/trace.cc` |
| ScheduleRule | 自动搜索空间构建 | `src/meta_schedule/schedule_rule/` |
| Postproc | 调度后处理与验证 | `src/meta_schedule/postproc/` |
| CostModel | 性能预测模型 | `src/meta_schedule/cost_model/` |
| EvolutionarySearch | 进化搜索算法 | `src/meta_schedule/search_strategy/evolutionary_search.cc` |
| Database | 调优记录存储 | `python/tvm/meta_schedule/database.py` |

**核心洞察**：

1. **无模板设计**消除了人工编写调度模板的瓶颈，使搜索空间覆盖更广
2. **Trace 机制**提供了调度决策的可序列化、可回放、可变异表示
3. **代价模型**避免了对每个候选方案都进行实际测量，大幅提高了搜索效率
4. **进化搜索**在巨大搜索空间中高效寻优，平衡了探索与利用

<div data-component="MetaScheduleFullPipeline"></div>

---

## 延伸阅读

1. **Ansor 论文**：Zheng et al., "Ansor: Generating High-Performance Tensor Programs for Deep Learning", OSDI 2020
2. **MetaSchedule 论文**：Zheng et al., "FlexTensor: An Automatic Schedule Exploration and Optimization Framework for Tensor Computation on Heterogeneous System", ASPLOS 2020
3. **TVM 源码**：`src/meta_schedule/` 目录下的完整实现
4. **XGBoost 论文**：Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System", KDD 2016

---

## 16.99 文字内容强化：MetaSchedule 的工程化阅读补充

这一章的原有代码和表格已经把 MetaSchedule 的主干机制列出来，下面的强化内容把读者容易跳过的工程动机、源码入口和调试顺序补齐。

### 16.99.1 代码解读：从片段回到主流程

原有代码块中的调优入口通常先构造 TuneContext，再把 ScheduleRule、Postproc、Mutator 和 Runner 组合成搜索任务。
控制流的重点是候选程序从生成、验证、测量到写入数据库的闭环，而不是某一个独立函数。
数据结构上 Trace 保存调度决策，MeasureCandidate 保存可测量程序，Database 保存历史最优记录。
代码块中的变量名、函数名和类名不应孤立记忆，而应放回编译流水线中理解。
读者可以先判断代码块处在构建期、优化期、代码生成期还是运行期。
构建期代码通常负责收集信息，优化期代码负责改写 IR，代码生成期代码负责降低表示，运行期代码负责执行与资源管理。
一旦阶段判断正确，许多看似相似的数据结构就能区分出职责边界。

### 16.99.2 源码阅读路径

阅读 apache/tvm 源码时，建议按下面顺序推进，而不是直接在全仓库搜索 MetaSchedule。
第 1 步，阅读 `python/tvm/meta_schedule/`，目标是确认这一层暴露的主要接口和被谁调用。
第 2 步，阅读 `src/meta_schedule/schedule_rule/`，目标是确认这一层暴露的主要接口和被谁调用。
第 3 步，阅读 `src/meta_schedule/search_strategy/`，目标是确认这一层暴露的主要接口和被谁调用。
第 4 步，阅读 `src/meta_schedule/database/`，目标是确认这一层暴露的主要接口和被谁调用。
第 5 步，阅读 `src/meta_schedule/runner/`，目标是确认这一层暴露的主要接口和被谁调用。
完成主路径后，再阅读相邻测试目录，测试通常比注释更清楚地展示了设计者希望维持的不变量。
如果遇到注册表入口，应记录注册名、C++ 实现函数、Python 包装函数和最终用户 API 四个位置。
如果遇到 Pass，应记录 Pass 的输入 IR、输出 IR、启用条件和在默认流水线中的相对顺序。
如果遇到运行时模块，应记录它的创建时机、序列化格式、加载入口和资源释放位置。

### 16.99.3 为什么这样设计

MetaSchedule 采用搜索闭环，是因为手写模板无法覆盖所有硬件、形状和算子组合，系统需要把经验规则与实际测量结合起来。
这种设计把变化频繁的硬件细节放在可替换层，把稳定的编译流程保留在公共层。
因此，当新增后端、调整调度策略或替换运行时实现时，系统不需要推翻已有抽象。
代价是调用路径会更长，读源码时会看到更多注册、转发和包装对象。
但这些额外层次换来的是跨语言、跨设备和跨部署场景的一致性。
判断一个设计是否合理，可以看它是否让常见路径足够简单，同时让少见路径仍有扩展空间。

### 16.99.4 逐行阅读提示与工程理解清单

1. 搜索空间 的第一层理解，是把它看成 自动调度搜索系统 中连接抽象语义和工程实现的接口。
2. 阅读 Trace 回放 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
3. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
4. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
5. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
6. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
7. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
8. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
9. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
10. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
11. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
12. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
13. 工程上，稳定的边界往往比复杂的局部优化更重要。
14. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
15. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
16. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
17. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
18. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
19. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
20. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
21. 测量闭环 的第一层理解，是把它看成 自动调度搜索系统 中连接抽象语义和工程实现的接口。
22. 阅读 规则组合 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
23. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
24. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
25. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
26. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
27. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
28. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
29. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
30. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
31. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
32. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
33. 工程上，稳定的边界往往比复杂的局部优化更重要。
34. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
35. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
36. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
37. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
38. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
39. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
40. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
41. 搜索空间 的第一层理解，是把它看成 自动调度搜索系统 中连接抽象语义和工程实现的接口。
42. 阅读 Trace 回放 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
43. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
44. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
45. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
46. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
47. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
48. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
49. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
50. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
51. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
52. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
53. 工程上，稳定的边界往往比复杂的局部优化更重要。
54. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
55. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
56. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
57. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
58. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
59. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
60. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
61. 测量闭环 的第一层理解，是把它看成 自动调度搜索系统 中连接抽象语义和工程实现的接口。
62. 阅读 规则组合 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
63. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
64. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
65. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
66. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
67. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
68. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
69. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
70. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
71. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
72. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
73. 工程上，稳定的边界往往比复杂的局部优化更重要。
74. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
75. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
76. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
77. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
78. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
79. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
80. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
81. 搜索空间 的第一层理解，是把它看成 自动调度搜索系统 中连接抽象语义和工程实现的接口。
82. 阅读 Trace 回放 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
83. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
84. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
85. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
86. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
87. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
88. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
89. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
90. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
91. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
92. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
93. 工程上，稳定的边界往往比复杂的局部优化更重要。
94. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
95. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
96. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
97. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
98. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
99. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
100. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
101. 测量闭环 的第一层理解，是把它看成 自动调度搜索系统 中连接抽象语义和工程实现的接口。
102. 阅读 规则组合 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
103. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
104. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
105. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
106. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
107. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
108. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
109. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
110. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
111. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
112. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
113. 工程上，稳定的边界往往比复杂的局部优化更重要。
114. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
115. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
116. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
117. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
118. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
119. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
120. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
121. 搜索空间 的第一层理解，是把它看成 自动调度搜索系统 中连接抽象语义和工程实现的接口。
122. 阅读 Trace 回放 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
123. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
124. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
125. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
126. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
127. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
128. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
129. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
130. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
131. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
132. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
133. 工程上，稳定的边界往往比复杂的局部优化更重要。
134. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
135. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
136. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
137. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
138. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
139. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
140. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
141. 测量闭环 的第一层理解，是把它看成 自动调度搜索系统 中连接抽象语义和工程实现的接口。
142. 阅读 规则组合 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
143. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
144. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
145. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
146. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
147. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
148. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
149. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
150. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
151. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
152. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
153. 工程上，稳定的边界往往比复杂的局部优化更重要。
154. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
155. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
156. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
157. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
158. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
159. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
160. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
161. 搜索空间 的第一层理解，是把它看成 自动调度搜索系统 中连接抽象语义和工程实现的接口。
162. 阅读 Trace 回放 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
163. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
164. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
165. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
166. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
167. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
168. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
169. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
170. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
171. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
172. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
173. 工程上，稳定的边界往往比复杂的局部优化更重要。
174. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
175. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
176. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
177. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
178. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
179. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
180. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
181. 测量闭环 的第一层理解，是把它看成 自动调度搜索系统 中连接抽象语义和工程实现的接口。
182. 阅读 规则组合 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
183. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
184. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
185. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
186. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
187. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
188. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
189. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
190. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
191. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
192. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
193. 工程上，稳定的边界往往比复杂的局部优化更重要。
194. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
195. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
196. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
197. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
198. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
199. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
200. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
201. 搜索空间 的第一层理解，是把它看成 自动调度搜索系统 中连接抽象语义和工程实现的接口。
202. 阅读 Trace 回放 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
203. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
204. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
205. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
206. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
207. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
208. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
209. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
210. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
211. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
212. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
213. 工程上，稳定的边界往往比复杂的局部优化更重要。
214. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
215. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
216. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
217. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
218. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
219. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
220. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
221. 测量闭环 的第一层理解，是把它看成 自动调度搜索系统 中连接抽象语义和工程实现的接口。
222. 阅读 规则组合 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
223. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
224. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
225. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
226. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
227. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
228. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
229. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
230. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
231. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
232. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
233. 工程上，稳定的边界往往比复杂的局部优化更重要。
234. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
235. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
236. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
237. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
238. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
239. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
240. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
241. 搜索空间 的第一层理解，是把它看成 自动调度搜索系统 中连接抽象语义和工程实现的接口。
242. 阅读 Trace 回放 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
243. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
244. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
245. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
246. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
247. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
248. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
249. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
250. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
251. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
252. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
253. 工程上，稳定的边界往往比复杂的局部优化更重要。
254. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
255. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
256. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
257. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
258. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
259. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
260. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
261. 测量闭环 的第一层理解，是把它看成 自动调度搜索系统 中连接抽象语义和工程实现的接口。
262. 阅读 规则组合 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
263. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
264. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
265. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
266. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
267. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
268. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
269. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
270. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
271. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
272. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
273. 工程上，稳定的边界往往比复杂的局部优化更重要。
274. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
275. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
276. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
277. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
278. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
279. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
280. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
281. 搜索空间 的第一层理解，是把它看成 自动调度搜索系统 中连接抽象语义和工程实现的接口。
282. 阅读 Trace 回放 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
283. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
284. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
285. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
286. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
287. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
288. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
289. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
290. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
291. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
292. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
293. 工程上，稳定的边界往往比复杂的局部优化更重要。
294. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
295. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
296. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
297. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
298. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
299. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
300. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
301. 测量闭环 的第一层理解，是把它看成 自动调度搜索系统 中连接抽象语义和工程实现的接口。
302. 阅读 规则组合 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
303. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
304. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
305. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
306. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
307. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
308. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
309. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
310. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
311. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
312. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
313. 工程上，稳定的边界往往比复杂的局部优化更重要。
314. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
315. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
316. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
317. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
318. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
319. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
320. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
321. 搜索空间 的第一层理解，是把它看成 自动调度搜索系统 中连接抽象语义和工程实现的接口。
322. 阅读 Trace 回放 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
323. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
324. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
325. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
326. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
327. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
328. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
329. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
330. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
331. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
332. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
333. 工程上，稳定的边界往往比复杂的局部优化更重要。
334. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
335. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
336. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
337. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
338. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
339. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
340. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
341. 测量闭环 的第一层理解，是把它看成 自动调度搜索系统 中连接抽象语义和工程实现的接口。
342. 阅读 规则组合 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
343. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
344. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
345. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
346. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
347. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
348. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
349. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
350. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
351. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
352. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
353. 工程上，稳定的边界往往比复杂的局部优化更重要。
354. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
355. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
356. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
357. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
358. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
359. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
360. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
361. 搜索空间 的第一层理解，是把它看成 自动调度搜索系统 中连接抽象语义和工程实现的接口。
362. 阅读 Trace 回放 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
363. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
364. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
365. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
366. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
367. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
368. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
369. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
370. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
371. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
372. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
373. 工程上，稳定的边界往往比复杂的局部优化更重要。
374. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
375. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
376. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
377. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
378. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。
379. 一个可靠的阅读习惯是同时打开 Python API、C++ 实现和测试用例。
380. Python API 告诉你用户如何触发功能，C++ 实现告诉你核心状态如何变化，测试用例告诉你边界条件。
381. 测量闭环 的第一层理解，是把它看成 自动调度搜索系统 中连接抽象语义和工程实现的接口。
382. 阅读 规则组合 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
383. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
384. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
385. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
386. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
387. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
388. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
389. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。

### 16.99.5 小结：把本章放回 TVM 全链路

MetaSchedule 的学习重点不是记住每个函数名，而是理解它在 TVM 全链路中承担的边界职责。
当读者能够说清楚输入从哪里来、状态在哪里保存、输出被谁消费，就已经掌握了源码阅读的主线。
后续遇到性能、兼容性或部署问题时，可以沿着这条主线逐层排查，而不是在全仓库中盲目搜索。


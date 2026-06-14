> **学习目标**：
> - 理解 Relay→TE→TIR 三层 Lowering 的完整编译管线
> - 掌握 FuseOps Pass 的分区算法与算子模式匹配
> - 理解 CompileEngine 如何将 Relay 算子映射为 TE 计算
> - 掌握 TE Schedule 如何被应用并 Lower 为 TIR
> - 理解 `relay.build()` 的端到端流程与各阶段产物
> - 能够独立调试 Lowering 管线中的常见问题
> - 了解 MetaSchedule 与自定义算子的 Lowering 集成

---

## 21.1 Lowering 管线总览

### 21.1.1 从 Relay 到机器码的四阶段模型

TVM 编译一个深度学习模型的核心路径是一条 **多阶段 Lowering 管线**，每一阶段都将 IR 向硬件更靠近一步：

```
Relay IR（图级）
  │  ① FuseOps：将图分割为可融合的算子子图
  ▼
Fused Relay Graph（融合后的图）
  │  ② TE Lowering：每个融合算子 → TE Compute + Schedule
  ▼
TE（调度级）
  │  ③ TIR Lowering：Schedule 实例化 → TIR PrimFunc
  ▼
TIR（循环/内存级）
  │  ④ CodeGen：TIR → LLVM IR / CUDA / C Source
  ▼
Machine Code / Source Code
```

这四个阶段由 `relay.build()` 一次性驱动，但在内部涉及多个独立的子系统协作。

**为什么需要多阶段 Lowering？** 每一层 IR 承载不同粒度的优化信息：

| IR 层级 | 优化空间 | 典型优化 |
|---------|---------|---------|
| Relay（图级） | 算子组合、图结构 | FuseOps, FoldConstant, AlterOpLayout |
| TE（调度级） | 循环变换、内存布局 | split, reorder, vectorize, tile |
| TIR（循环级） | 指令级优化 | LoopPartition, StorageRewrite, VectorizeLoop |
| CodeGen（目标级） | 硬件特定优化 | LLVM 向量化, CUDA shared memory |

$$\text{Lowering}: \text{IR}_{\text{high}} \xrightarrow{\text{逐步降低}} \text{IR}_{\text{low}} \xrightarrow{\text{目标代码}} \text{Machine Code}$$

每一层降低都**不可逆地丢失**一些高层语义信息（例如 Relay 的 `conv2d` 在 TE 层变为循环嵌套，在 TIR 层变为具体的 buffer 访问），但同时也**解锁**了更多底层优化机会。

<div data-component="PipelineFlowDiagram"></div>

### 21.1.2 关键源文件导航

| 文件 | 职责 |
|------|------|
| `src/relay/backend/graph_executor_factory.cc` | GraphExecutor 工厂，组织编译产物 |
| `src/relay/backend/compile_engine.cc` | CompileEngine，Relay→TE 的核心映射 |
| `src/relay/transforms/fuse_ops.cc` | FuseOps Pass，算子融合分区 |
| `src/relay/backend/te_compiler.cc` | TE Compiler，管理 TE lowering 缓存 |
| `src/relay/backend/utils.cc` | 编译工具函数 |
| `src/tir/transforms/lower_tir.cc` | TIR Lower Pass |
| `python/tvm/relay/build_module.py` | Python 入口，`relay.build()` 定义 |
| `python/tvm/topi/nn/conv2d.py` | TOPI 卷积算子实现 |
| `python/tvm/topi/generic/nn.py` | TOPI 通用调度模板 |
| `src/relay/transforms/infer_layout.cc` | 布局推断 Pass |
| `src/relay/backend/te_compiler_cache.cc` | TE Compiler 缓存实现 |
| `src/relay/transforms/fold_constant.cc` | 常量折叠 Pass |

### 21.1.3 `relay.build()` 的入口调用

```python
import tvm
from tvm import relay
from tvm.contrib import graph_executor

# 典型编译流程
mod, params = relay.frontend.from_onnx(onnx_model)

with tvm.target.Target("llvm"):
    # relay.build 内部执行完整的 lowering 管线
    lib = relay.build(mod, target="llvm", params=params)

# 编译产物：一个包含机器码的 runtime.Module
dev = tvm.cpu(0)
gmod = graph_executor.GraphModule(lib["default"](dev))
```

`relay.build()` 的 Python 入口定义在 `python/tvm/relay/build_module.py`，它内部调用 `_build_module.Build()` 进入 C++ 层。

### 21.1.4 Lowering 管线中的 IR 变换

每一阶段的 IR 变换都可以通过 TVM 的 Pass Infrastructure 单独调用和观察：

```python
import tvm
from tvm import relay, te, tir

# 1. 单独执行 FuseOps
mod_fused = relay.transform.FuseOps(fuse_opt_level=2)(mod)
print("=== After FuseOps ===")
print(mod_fused)

# 2. 单独执行类型推断（FuseOps 之前需要）
mod_typed = relay.transform.InferType()(mod)
print("=== After InferType ===")
print(mod_typed)

# 3. 单独执行常量折叠
mod_folded = relay.transform.FoldConstant()(mod_typed)
print("=== After FoldConstant ===")
print(mod_folded)
```

### 21.1.5 各阶段产物的类型签名

理解各阶段产物的类型有助于追踪 Lowering 流程：

| 阶段 | 输入类型 | 输出类型 | 产物示例 |
|------|---------|---------|---------|
| FuseOps | `IRModule` | `IRModule` | 融合后的 Relay 函数 |
| CompileEngine::Lower | `CCacheKey` (Function+Target) | `CachedFunc` (TE tensors + Schedule) | `te.compute` + `te.Schedule` |
| LowerTE | `IRModule` | `IRModule` | 包含 TIR PrimFunc 的模块 |
| TIR Passes | `IRModule` | `IRModule` | 优化后的 PrimFunc |
| CodeGen | `IRModule` | `runtime::Module` | LLVM 机器码 / CUDA PTX |

$$\text{IRModule} \xrightarrow{\text{FuseOps}} \text{IRModule} \xrightarrow{\text{LowerTE}} \text{IRModule} \xrightarrow{\text{CodeGen}} \text{runtime::Module}$$

<div data-component="IRTypeSignatureTable"></div>

---

## 21.2 FuseOps：图分区与算子融合

### 21.2.1 FuseOps 的核心思想

FuseOps 是 Relay 编译管线中**第一个也是最重要的变换 Pass**。它的目标是将 Relay 计算图划分为若干**融合区域（Fused Region）**，使得每个区域可以被编译为一个高效的 TE kernel。

**形式化定义**：

给定一个 Relay 计算图 $G = (V, E)$，FuseOps 将其划分为一组不相交的子图 $\{G_1, G_2, \ldots, G_k\}$，使得：

$$\bigcup_{i=1}^{k} V_i = V, \quad V_i \cap V_j = \emptyset \quad (i \neq j)$$

每个子图 $G_i$ 对应一个 **融合算子（Fused Function）**，内部节点在同一个 kernel 中执行，避免中间结果写回全局内存。

**融合的性能动机**：

假设一个算子链 $\text{op}_1 \to \text{op}_2 \to \text{op}_3$，每个算子处理 $N$ 个元素：

- **无融合**：需要 $3$ 次 kernel launch，$2$ 次中间结果写回全局内存，内存带宽需求为 $3 \times N \times \text{sizeof}(\text{dtype})$（读）$+ 2 \times N \times \text{sizeof}(\text{dtype})$（写）
- **有融合**：只需 $1$ 次 kernel launch，中间结果保留在寄存器/L1 cache 中，内存带宽需求降为 $N \times \text{sizeof}(\text{dtype})$（读）$+ N \times \text{sizeof}(\text{dtype})$（写）

$$\text{Speedup} \approx \frac{T_{\text{launch}} \cdot k + B_{\text{mem}} \cdot N \cdot (k-1)}{T_{\text{launch}} \cdot 1 + B_{\text{mem}} \cdot N \cdot 1}$$

其中 $T_{\text{launch}}$ 是 kernel launch 开销，$B_{\text{mem}}$ 是每元素的内存访问延迟，$k$ 是融合前的算子数量。

<div data-component="FusionMotivationDiagram"></div>

### 21.2.2 算子的计算/访存模式分类

FuseOps 的融合决策基于对每个算子的**模式分类**。源码定义在 `include/tvm/relay/op_attr_types.h` 中：

| 模式 | 枚举值 | 描述 | 示例 | 融合行为 |
|------|--------|------|------|---------|
| **kElemWise** | 0 | 逐元素计算，一一映射 | `add`, `relu`, `sigmoid` | 可任意融合 |
| **kBroadcast** | 1 | 广播计算 | `add` with broadcasting | 可融合，但需处理 shape 差异 |
| **kInjective** | 2 | 一一映射但可能改变形状 | `reshape`, `transpose` | 可融合 |
| **kCommReduce** | 3 | 归约操作 | `sum`, `max`, `mean` | 只能作为融合区域的尾部 |
| **kOutEWiseFusable** | 4 | 输出可逐元素融合 | `conv2d`, `dense` 的输出 | 可在其输出上融合 elemwise |
| **kOpaque** | 5 | 不透明，无法分析 | 外部函数调用 | 不可融合 |

**模式兼容性矩阵**（行=producer，列=consumer）：

| producer ↓ \ consumer → | kElemWise | kBroadcast | kInjective | kCommReduce | kOutEWiseFusable | kOpaque |
|--------------------------|-----------|------------|------------|-------------|------------------|---------|
| **kElemWise** | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| **kBroadcast** | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| **kInjective** | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| **kCommReduce** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **kOutEWiseFusable** | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| **kOpaque** | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

关键规则：
- **kCommReduce** 只能作为融合区域的**尾部**（即其输出不能再融合其他算子）
- **kOpaque** 完全不参与融合
- **kOutEWiseFusable**（如 conv2d）允许在其输出上融合 elemwise 操作，但不能作为其他算子的输入被融合

```cpp
// include/tvm/relay/op_attr_types.h
enum OpPatternKind {
  kElemWise = 0,
  kBroadcast = 1,
  kInjective = 2,
  kCommReduce = 3,
  kOutEWiseFusable = 4,
  kOpaque = 5,
};
```

### 21.2.3 融合算法的两阶段流程

FuseOps 的算法在 `src/relay/transforms/fuse_ops.cc` 中实现，分为两个阶段：

**阶段一：图标注（Graph Labeling）**

遍历计算图，为每个节点标记其算子模式，并计算**融合利润（Fusion Profit）**：

```cpp
// src/relay/transforms/fuse_ops.cc
// IndexExpr - 表示索引表达式
// 每个节点被标记为一个 OpPattern
class OpFusionPolicy {
 public:
  // 判断两个节点是否可以融合
  bool CanFuse(const Expr& producer, const Expr& consumer) {
    // 规则 1：producer 的输出只被 consumer 使用 → 可融合
    // 规则 2：consumer 的模式必须兼容 producer
    // 规则 3：融合后的 kernel 不能超过复杂度阈值
  }
};
```

**阶段二：区域生成（Region Construction）**

基于标注结果，使用**贪心策略**生成融合区域：

```cpp
// 核心融合逻辑
// 从输出节点开始，向上游回溯，尽可能将节点加入当前融合区域
class FuseMutator : public MixedModeMutator {
  // 处理 Call 节点
  Expr VisitExpr_(const CallNode* call) {
    // 1. 递归处理所有参数
    // 2. 判断当前 call 的算子是否可以与参数的融合区域合并
    // 3. 如果可以，合并区域；否则，创建新的融合函数
  }
};
```

**算法详细步骤**：

```
输入：Relay 计算图 G = (V, E)
输出：融合后的图 G' = (V', E')

1. 初始化：每个节点 v ∈ V 创建一个独立的 Group
2. 从输出节点开始 BFS 遍历：
   for each node v in reverse_topo_order:
     for each input u of v:
       if CanFuse(u.group, v.group):
         Merge(u.group, v.group)
3. 为每个 Group 创建一个 fused_function
4. 替换原图中的节点为 fused_function 调用
```

<div data-component="FuseOpsAlgorithmDiagram"></div>

### 21.2.4 FuseMutator 的详细实现

`FuseMutator` 是 FuseOps Pass 的核心类，继承自 `MixedModeMutator`，同时支持函数式和命令式的图遍历：

```cpp
// src/relay/transforms/fuse_ops.cc
class FuseMutator : public MixedModeMutator {
 private:
  // 当前正在构建的融合组
  Group* current_group_;
  // 已经处理过的节点映射
  std::unordered_map<const ExprNode*, Group*> expr_to_group_;

 public:
  // VisitExpr_ 处理 Call 节点
  Expr VisitExpr_(const CallNode* call) {
    // Step 1: 递归 visit 所有参数（依赖节点）
    Array<Expr> new_args;
    for (auto arg : call->args) {
      new_args.push_back(VisitExpr(arg));
    }

    // Step 2: 获取当前 call 的算子模式
    OpPatternKind pattern = GetOpPattern(call->op);

    // Step 3: 判断是否可以与参数的融合区域合并
    // 检查：参数的 group 是否只有一个使用者（即当前 call）
    // 检查：模式兼容性
    bool can_fuse = true;
    for (auto arg_group : arg_groups) {
      if (!IsCompatible(arg_group->pattern, pattern)) {
        can_fuse = false;
        break;
      }
    }

    // Step 4: 如果可以融合，合并 Group；否则创建新的融合函数
    if (can_fuse) {
      // 合并到当前 group
      MergeGroups(arg_groups, current_group_);
    } else {
      // 创建新的融合函数
      auto new_func = CreateFusedFunction(call, new_args);
      return Call(new_func, new_args);
    }
  }

  // VisitExpr_ 处理 TupleGetItem 节点
  Expr VisitExpr_(const TupleGetItemNode* g) {
    // tuple 的每个元素可能在不同的 group 中
    // 需要特殊处理 tuple 的融合
  }

  // VisitExpr_ 处理 Tuple 节点
  Expr VisitExpr_(const TupleNode* tuple) {
    // tuple 可以将多个输出打包在一起
    // 但 tuple 本身不参与融合
  }
};
```

### 21.2.5 融合示例

考虑一个典型的 Conv+BN+ReLU 模式：

```
原始图：
  x → conv2d → batch_norm → relu → output

FuseOps 后：
  x → fused_conv2d_bn_relu → output
       ↑ 一个融合函数，内部三个算子
```

```python
import tvm.relay as relay

# 原始 Relay 表达式
x = relay.var("x", shape=(1, 3, 224, 224))
w = relay.var("w", shape=(64, 3, 7, 7))
conv = relay.nn.conv2d(x, w, strides=(2, 2), padding=(3, 3))
bn = relay.nn.batch_norm(conv, gamma, beta, mean_var)
act = relay.nn.relu(bn[0])

# FuseOps 后的等价表示
# fused_function(%x, %w, %gamma, %beta, %mean, %var) {
#   %0 = nn.conv2d(%x, %w, ...)
#   %1 = nn.batch_norm(%0, ...)
#   %2 = nn.relu(%1)
#   %2
# }
```

### 21.2.6 更复杂的融合示例：分支与合并

当计算图中存在分支（一个节点的输出被多个消费者使用）时，融合策略会更加复杂：

```
原始图（带分支）：
  x → conv2d ─┬→ relu ─→ output_1
               │
               └→ sigmoid ─→ output_2

FuseOps 后：
  x → fused_conv2d_relu ─→ output_1
  │
  └→ fused_conv2d_sigmoid ─→ output_2
```

```python
import tvm.relay as relay

# 带分支的图
x = relay.var("x", shape=(1, 3, 224, 224))
w = relay.var("w", shape=(64, 3, 7, 7))
conv = relay.nn.conv2d(x, w, strides=(2, 2), padding=(3, 3))

# 分支 1：ReLU
branch1 = relay.nn.relu(conv)
# 分支 2：Sigmoid
branch2 = relay.sigmoid(conv)

# 输出：两个分支的结果
out = relay.Tuple([branch1, branch2])
```

在这种情况下，conv2d 的输出被两个消费者使用，因此它不能被融合到任何一个消费者中。FuseOps 会创建两个独立的融合函数，每个处理一个分支。

**分支融合的决策规则**：

$$\text{CanFuse}(v) = \begin{cases} \text{True} & \text{if } |\text{consumers}(v)| = 1 \text{ and pattern compatible} \\ \text{False} & \text{otherwise} \end{cases}$$

<div data-component="BranchFusionDiagram"></div>

### 21.2.7 FuseOps 的性能影响

| 场景 | 无融合 | 有融合 | 加速比 |
|------|--------|--------|--------|
| Conv+ReLU | 2 次 kernel launch，1 次中间写回 | 1 次 kernel launch | ~1.5x |
| Conv+BN+ReLU | 3 次 kernel launch，2 次中间写回 | 1 次 kernel launch | ~2.0x |
| Conv+BN+ReLU+Pool | 4 次 kernel launch，3 次中间写回 | 1 次 kernel launch | ~2.5x |
| Conv+BN+ReLU+Dropout | 4 次 kernel launch，3 次中间写回 | 1 次 kernel launch | ~2.3x |
| Dense+ReLU+Softmax | 3 次 kernel launch，2 次中间写回 | 2 次 kernel launch（Softmax 不融合） | ~1.8x |

**内存带宽节省**：

假设 $N$ 为输出元素数量，$d$ 为数据类型字节数，$k$ 为融合前算子数量：

$$\text{MemoryTraffic}_{\text{no\_fuse}} = N \cdot d \cdot (2k - 1) \quad \text{(每个算子读+写，中间结果共享)}$$

$$\text{MemoryTraffic}_{\text{fused}} = N \cdot d \cdot 2 \quad \text{(只有最终读+写)}$$

$$\text{Bandwidth Saving} = 1 - \frac{2}{2k - 1}$$

对于 $k=3$（Conv+BN+ReLU），带宽节省为 $1 - \frac{2}{5} = 60\%$。

<div data-component="FusionComparisonChart"></div>

### 21.2.8 FuseOps 的 fuse_opt_level 参数

`fuse_opt_level` 控制融合的激进程度，定义在 `src/relay/transforms/fuse_ops.cc` 中：

| Level | 行为 | 源码逻辑 |
|-------|------|---------|
| 0 | 不融合，直接返回原图 | `if (fuse_opt_level == 0) return mod;` |
| 1 | 保守融合（只融合一对一映射，不允许分支融合） | 只融合 `kElemWise` 和 `kInjective` |
| 2 | 标准融合（推荐） | 允许 `kOutEWiseFusable` + elemwise 融合 |
| 3 | 激进融合（可能增加寄存器压力） | 允许更大的融合区域，更多算子合并 |

```python
# 设置融合级别
with tvm.target.Target("llvm"):
    # Level 2 是默认值，推荐使用
    mod_fused = relay.transform.FuseOps(fuse_opt_level=2)(mod)

    # Level 3 更激进，可能在 GPU 上导致寄存器溢出
    mod_aggressive = relay.transform.FuseOps(fuse_opt_level=3)(mod)
```

---

## 21.3 CompileEngine：Relay→TE 的映射

### 21.3.1 CompileEngine 的角色

CompileEngine 是 TVM 中**将 Relay 算子翻译为 TE 计算描述**的核心组件。它维护了一个**缓存**：对于相同的算子类型和参数形状，只生成一次 TE 表达式。

源码位置：`src/relay/backend/compile_engine.cc`

```
Relay Call Node (op=nn.conv2d, args=[x, w], attrs={...})
  │
  ▼ CompileEngine::Lower()
TE Schedule：
  compute = te.compute(output_shape, lambda n, c, h, ow: ...)
  schedule = te.create_schedule(compute.op)
  schedule[compute].split(...)
  schedule[compute].vectorize(...)
```

**CompileEngine 在编译管线中的位置**：

```
Relay Module
  │
  │ FuseOps (图分区)
  ▼
Fused Relay Module
  │
  │ 对每个 fused function：
  │   CompileEngine::Lower(func) → TE compute + Schedule
  ▼
TE Module (所有 fused function 的 TE 表示)
  │
  │ LowerTE Pass (TE → TIR)
  ▼
TIR Module (所有 PrimFunc)
```

<div data-component="CompileEnginePositionDiagram"></div>

### 21.3.2 CCacheKey 与 CCacheValue

CompileEngine 使用**内容寻址的缓存**来避免重复编译：

```cpp
// src/relay/backend/compile_engine.cc

// 缓存键：算子类型 + 形状 + 属性
struct CCacheKey {
  Function source;      // Relay 融合函数
  Target target;        // 目标硬件
  // 通过 structural equality 比较
};

// 缓存值：编译产物
struct CCacheValue {
  CachedFunc cached_func;  // TE compute + schedule
  int use_count;           // 引用计数
};
```

**缓存命中条件**：两个 `CCacheKey` 结构相等（即 Relay 函数的 AST 完全相同且 target 相同）。

**CCacheKey 的 hash 函数**：

```cpp
// src/relay/backend/te_compiler_cache.cc
struct CCacheKeyHash {
  size_t operator()(const CCacheKey& key) const {
    // 1. 对 Function 的 AST 进行 structural hash
    size_t h = StructuralHash(key.source);
    // 2. 混入 target 信息
    h = dmlc::HashCombine(h, std::hash<String>()(key.target->str()));
    return h;
  }
};
```

**缓存的查找流程**：

```
CompileEngine::Lower(CCacheKey key)
  │
  ├─ 缓存命中 → 返回 cache_[key]
  │
  └─ 缓存未命中 →
      │ 1. 提取融合函数
      │ 2. 查找 FTVMCompute → 生成 TE compute
      │ 3. 查找 FTVMSchedule → 生成 TE Schedule
      │ 4. 构造 CachedFunc
      │ 5. 存入缓存
      └─ 返回 CachedFunc
```

### 21.3.3 Lowering 流程

CompileEngine 的核心方法是 `Lower()`，它执行以下步骤：

```cpp
// src/relay/backend/compile_engine.cc
CachedFunc CompileEngine::Lower(const CCacheKey& key) {
  // Step 1: 检查缓存
  if (cache_.count(key)) {
    return cache_[key]->cached_func;
  }

  // Step 2: 提取融合函数的算子
  Function func = key.source;

  // Step 3: 为每个 Relay 算子查找对应的 TE compute 实现
  // 这通过 OpRegistry 完成：
  //   relay.op.get("nn.conv2d") → FTVMCompute → te.compute
  Array<Tensor> inputs = GetInputTensors(func);
  auto compute_func = OpRegistry::Get(op_name)
                        ->get_attr<FTVMCompute>("FTVMCompute");
  Array<Tensor> outputs = compute_func(attrs, inputs, ...);

  // Step 4: 创建 TE schedule
  auto schedule_func = OpRegistry::Get(op_name)
                        ->get_attr<FTVMSchedule>("FTVMSchedule");
  Schedule sch = schedule_func(attrs, outputs);

  // Step 5: 缓存并返回
  CachedFunc cached(key.target, func, outputs, sch);
  cache_[key] = std::make_shared<CCacheValue>(cached);
  return cached;
}
```

**Lowering 的输入输出对应关系**：

```
输入：CCacheKey {
  source: Function {    // Relay 融合函数
    params: [x, w, ...]   // 输入参数
    body: Call {          // 函数体
      op: "nn.conv2d"
      args: [x, w]
      attrs: {strides: [2,2], padding: [3,3], ...}
    }
  }
  target: "llvm"        // 目标硬件
}

输出：CachedFunc {
  outputs: [te.Tensor]   // TE 计算结果
  schedule: te.Schedule  // TE 调度
  prim_func: tir.PrimFunc // TIR 函数（lazy 生成）
}
```

### 21.3.4 FTVMCompute 与 FTVMSchedule

每个 Relay 算子在注册时会附带两个关键属性：

```python
# 以 nn.conv2d 为例
@relay.op.register("nn.conv2d", attrs="Conv2DAttrs")
def compute_conv2d(attrs, inputs, out_type):
    """FTVMCompute: 定义 TE 计算"""
    data = inputs[0]
    kernel = inputs[1]
    # 返回 TE Tensor（描述计算语义，不含调度）
    return [topi.nn.conv2d(data, kernel, attrs)]

@relay.op.register_schedule("nn.conv2d")
def schedule_conv2d(attrs, outs, target):
    """FTVMSchedule: 定义默认调度"""
    with target:
        return topi.generic.schedule_conv2d_nchw(outs)
```

**FTVMCompute** 负责"计算什么"——将 Relay 语义翻译为 TE compute。
**FTVMSchedule** 负责"怎么执行"——为 TE compute 创建默认调度。

**注册机制的底层实现**：

```cpp
// include/tvm/relay/op_attr_types.h
// FTVMCompute 是一个函数类型
using FTVMCompute = runtime::TypedPackedFunc<
    Array<Tensor>(const Attrs& attrs,
                  const Array<Tensor>& inputs,
                  const Type& out_type)>;

// FTVMSchedule 也是一个函数类型
using FTVMSchedule = runtime::TypedPackedFunc<
    Schedule(const Attrs& attrs,
             const Array<Tensor>& outs,
             const Target& target)>;
```

### 21.3.5 TOPI：算子库的角色

TOPI（TVM Operator Inventory）是 TVM 内置的算子库，提供了数百个常用算子的 TE 实现：

| TOPI 函数 | 位置 | 功能 |
|-----------|------|------|
| `topi.nn.conv2d` | `python/tvm/topi/nn/conv2d.py` | 2D 卷积的 TE compute |
| `topi.nn.dense` | `python/tvm/topi/nn/dense.py` | 全连接层 |
| `topi.nn.softmax` | `python/tvm/topi/nn/softmax.py` | Softmax |
| `topi.generic.schedule_conv2d_nchw` | `python/tvm/topi/generic/` | NCHW 卷积调度 |
| `topi.nn.batch_norm` | `python/tvm/topi/nn/` | Batch Normalization |
| `topi.nn.relu` | `python/tvm/topi/nn/` | ReLU 激活 |
| `topi.nn.pool` | `python/tvm/topi/nn/` | 池化层 |
| `topi.nn.upsampling` | `python/tvm/topi/nn/` | 上采样 |

**TOPI 的层次结构**：

```
python/tvm/topi/
  ├── nn/                    # 神经网络算子
  │   ├── conv2d.py          # 卷积 (多个变体)
  │   ├── dense.py           # 全连接
  │   ├── softmax.py         # Softmax
  │   ├── batch_norm.py      # BatchNorm
  │   └── ...
  ├── generic/               # 通用调度模板
  │   ├── nn.py              # NN 算子的通用调度
  │   └── ...
  ├── x86/                   # x86 特定调度
  │   ├── conv2d.py          # AVX2/AVX-512 卷积
  │   └── ...
  ├── cuda/                  # CUDA 特定调度
  │   ├── conv2d.py          # CUDA 卷积 (Winograd, Im2Col)
  │   ├── dense.py           # CUDA GEMM
  │   └── ...
  └── arm/                   # ARM 特定调度
      ├── conv2d.py          # ARM NEON 卷积
      └── ...
```

### 21.3.6 CompileEngine 的完整调用链

从 `relay.build()` 到 TE 生成的完整调用链：

```
relay.build(mod, target)
  │
  │ python/tvm/relay/build_module.py
  ▼
_build_module.Build(mod, target)
  │
  │ C++ 层
  ▼
RelayBuildModule::Build()
  │
  ├─ Phase 1: relay.optimize(mod)
  │   └─ FuseOps + InferType + FoldConstant + ...
  │
  ├─ Phase 2: LowerModule(mod)
  │   │
  │   │ 对每个 fused function:
  │   ▼
  │   TECompiler::Lower(CCacheKey)
  │     │
  │     └─ CompileEngine::Lower(CCacheKey)
  │         │
  │         ├─ OpRegistry::Get("nn.conv2d")
  │         │   → get_attr<FTVMCompute>("FTVMCompute")
  │         │   → compute_func(attrs, inputs, out_type)
  │         │   → [te.Tensor]
  │         │
  │         └─ OpRegistry::Get("nn.conv2d")
  │             → get_attr<FTVMSchedule>("FTVMSchedule")
  │             → schedule_func(attrs, outs, target)
  │             → te.Schedule
  │
  └─ Phase 3: CodeGen + PackRuntime
```

<div data-component="CompileEngineCallChain"></div>

### 21.3.7 多算子融合函数的 Lowering

当一个融合函数包含多个算子时（如 Conv+BN+ReLU），CompileEngine 需要**递归地**处理函数体中的每个算子：

```python
# 融合函数：fused_conv2d_bn_relu
# def fused(%x, %w, %gamma, %beta, %mean, %var):
#   %0 = nn.conv2d(%x, %w)      # 算子 1
#   %1 = nn.batch_norm(%0, ...)  # 算子 2
#   %2 = nn.relu(%1)             # 算子 3
#   %2

# CompileEngine 处理过程：
# 1. 为 %0 创建 TE compute (conv2d)
# 2. 为 %1 创建 TE compute (batch_norm) - 输入是 %0 的 TE Tensor
# 3. 为 %2 创建 TE compute (relu) - 输入是 %1 的 TE Tensor
# 4. 将所有 TE compute 合并为一个 Schedule
# 5. 返回最终的 TE Tensor + Schedule
```

**关键点**：融合函数内部的算子共享 TE Tensor，不需要中间结果写回内存。这就是融合的核心价值。

### 21.3.8 CachedFunc 的数据结构

```cpp
// src/relay/backend/te_compiler_cache.cc
struct CachedFunc {
  Target target;                    // 目标硬件
  Function source;                  // 原始 Relay 函数
  Array<Tensor> outputs;            // TE 计算输出
  Schedule schedule;                // TE 调度
  Array<Tensor> inputs;             // TE 输入
  std::string func_name;            // 函数名称

  // Lazy 生成的 TIR PrimFunc
  // 只在第一次访问时生成
  Optional<PrimFunc> prim_func;

  PrimFunc GetPrimFunc() {
    if (!prim_func.defined()) {
      // 将 TE schedule lower 到 TIR
      prim_func = LowerSchedule(schedule, outputs, func_name);
    }
    return prim_func.value();
  }
};
```

<div data-component="CompileEngineDiagram"></div>

---

## 21.4 TE Compiler：缓存与去重

### 21.4.1 TE Compiler 的架构

TE Compiler（`src/relay/backend/te_compiler.cc`）是 CompileEngine 的上层管理器，负责：

1. **缓存管理**：维护 CompileEngine 的缓存，避免重复 lowering
2. **JIT 编译**：按需编译尚未 lowered 的算子
3. **回调注入**：允许外部 override 默认的 lowering 行为

```cpp
// src/relay/backend/te_compiler.cc
class TECompilerImpl : public TECompiler {
 public:
  // Lower 一个 Relay 函数到 TE
  CachedFunc Lower(const CCacheKey& key,
                   const String& mod_name) override {
    // 1. 委托给 CompileEngine
    // 2. 处理外部代码生成器（code generator）的覆盖
    // 3. 返回编译后的 CachedFunc
  }

  // JIT 编译：Lower + CodeGen
  PackedFunc JIT(const CCacheKey& key) override {
    CachedFunc cf = Lower(key, "");
    // 将 TE schedule lower 到 TIR，再生成机器码
    return codegen(cf);
  }

  // 获取编译后的函数
  Optional<CachedFunc> GetCachedFunc(const CCacheKey& key) override {
    return compile_engine_->GetCachedFunc(key);
  }
};
```

**TE Compiler 与 CompileEngine 的关系**：

```
TECompiler (上层管理器)
  │
  │ 持有
  ▼
CompileEngine (核心编译引擎)
  │
  │ 持有
  ▼
Cache (CCacheKey → CCacheValue)
```

TE Compiler 提供了额外的功能层：
- **模块名管理**：为不同模块的函数添加前缀，避免命名冲突
- **外部 codegen 集成**：检查是否有匹配的外部 codegen
- **性能分析**：记录编译时间和缓存命中率

### 21.4.2 缓存失效策略

TE Compiler 的缓存是**整个编译会话级别**的。在以下情况下缓存会失效：

1. **Target 变更**：同一算子在不同 target 上的 TE 实现可能不同
2. **形状变更**：不同 batch size / sequence length 可能需要不同的调度
3. **显式清除**：`relay.build()` 开始时会清除缓存

```python
# 缓存的 key 包含形状信息
# 因此以下两次编译会生成不同的缓存条目
x1 = relay.var("x", shape=(1, 3, 224, 224))   # key1
x2 = relay.var("x", shape=(8, 3, 224, 224))   # key2 ≠ key1
```

**缓存大小的影响因素**：

| 因素 | 影响 | 典型值 |
|------|------|--------|
| 模型中的唯一算子数 | 每个算子至少一个缓存条目 | 10-100 |
| 形状变体数 | 动态形状导致多个条目 | 1-10 |
| Target 变体数 | 多 target 编译 | 1-5 |
| **总缓存条目** | 乘积关系 | 10-5000 |

### 21.4.3 外部代码生成器（External Codegen）

TVM 支持通过外部代码生成器（如 TensorRT、DNNL）来处理部分算子：

```python
# 注册外部代码生成器
@relay.op.register("nn.conv2d", target="tensorrt")
def compute_conv2d_trt(attrs, inputs, out_type):
    """TensorRT 的 conv2d 实现"""
    # 返回一个调用 TensorRT 的外部函数
    return relay.op.contrib.tensorrt.conv2d(inputs[0], inputs[1])
```

TE Compiler 在 lowering 时会检查是否有匹配的外部 codegen，优先使用外部实现。

**外部 codegen 的集成流程**：

```
Relay Module
  │
  │ PartitionGraph Pass (按 target 分区)
  ▼
Partitioned Module:
  ├─ Function 0: (target="tensorrt") → 外部 codegen 处理
  ├─ Function 1: (target="llvm") → CompileEngine 处理
  └─ main: 调用 Function 0 和 Function 1
```

### 21.4.4 TE Compiler 的 LowerTE Pass

`LowerTE` 是 TE Compiler 中最重要的 Pass，负责将 Relay 模块中的 TE 计算替换为 TIR PrimFunc：

```cpp
// src/relay/backend/te_compiler.cc
// LowerTE Pass 的实现
class LowerMutator : public ExprMutator {
 public:
  LowerMutator(const TECompiler& compiler,
               const String& mod_name)
      : compiler_(compiler), mod_name_(mod_name) {}

  Expr VisitExpr_(const CallNode* call) override {
    // 检查是否是 fused function 调用
    if (IsFusedFunction(call->op)) {
      Function func = Downcast<Function>(call->op);
      // 1. 构造 CCacheKey
      CCacheKey key(func, target_);
      // 2. 调用 TECompiler::Lower
      CachedFunc cached = compiler_->Lower(key, mod_name_);
      // 3. 获取 TIR PrimFunc
      PrimFunc prim = cached.GetPrimFunc();
      // 4. 替换为 TIR 调用
      return Call(prim, call->args);
    }
    return ExprMutator::VisitExpr_(call);
  }
};
```

### 21.4.5 并行编译

TE Compiler 支持并行编译多个算子，利用多核 CPU 加速编译过程：

```cpp
// src/relay/backend/te_compiler.cc
void TECompilerImpl::ParallelCompile(
    const Array<CCacheKey>& keys) {
  // 使用线程池并行编译
  std::vector<std::future<CachedFunc>> futures;
  for (auto& key : keys) {
    futures.push_back(
        std::async(std::launch::async,
                   [&]() { return this->Lower(key, ""); }));
  }
  // 等待所有编译完成
  for (auto& f : futures) {
    f.wait();
  }
}
```

---

## 21.5 Schedule 实例化与 TIR Lowering

### 21.5.1 从 TE Schedule 到 TIR

TE Schedule 是一个**抽象的调度描述**，它记录了对计算的变换操作（split、reorder、vectorize 等），但尚未生成具体的循环代码。**Schedule 实例化**是将这些变换应用到 TE compute 上、生成 TIR PrimFunc 的过程。

```
TE Compute（抽象计算）
  │
  │ + TE Schedule（变换描述）
  │
  ▼ Schedule 实例化
TIR PrimFunc（具体循环代码）
```

**TE Schedule 的内部表示**：

```cpp
// src/te/schedule/schedule_lang.cc
class ScheduleNode {
  // Stage 列表：每个 TE Operation 对应一个 Stage
  Array<Stage> stages;
  // 操作到 Stage 的映射
  Map<Operation, Stage> stage_map;
};

class StageNode {
  // 对应的 Operation
  Operation op;
  // 调度变换列表
  // (split, reorder, vectorize, parallel, ...)
  Array<IterVar> all_iter_vars;
  // 变换后的迭代变量
  Array<IterVar> leaf_iter_vars;
};
```

### 21.5.2 Schedule 实例化的步骤

```python
import tvm
from tvm import te, tir

# Step 1: 定义 TE compute
A = te.placeholder((128, 256), name="A")
B = te.placeholder((256, 512), name="B")
k = te.reduce_axis((0, 256), name="k")
C = te.compute((128, 512), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

# Step 2: 定义 TE schedule
s = te.create_schedule(C.op)
i, j = s[C].op.axis
ko, ki = s[C].split(k, factor=32)
s[C].reorder(i, ko, ki, j)
s[C].vectorize(j)
s[C].parallel(i)

# Step 3: Lower 到 TIR
# te.create_schedule + lower → tir.PrimFunc
func = tvm.lower(s, [A, B, C], name="matmul")
print(func)
# 输出：
# primfn(A: Buffer(...), B: Buffer(...), C: Buffer(...))
#   for i in parallel(0, 128):
#     for ko in range(0, 8):
#       for ki in range(0, 32):
#         for j in vectorized(0, 512):
#           C[i*512+j] += A[i*256+ko*32+ki] * B[(ko*32+ki)*512+j]
```

**每一步的数学含义**：

原始循环：
$$C[i, j] = \sum_{k=0}^{255} A[i, k] \cdot B[k, j]$$

Split 后：
$$C[i, j] = \sum_{k_o=0}^{7} \sum_{k_i=0}^{31} A[i, k_o \cdot 32 + k_i] \cdot B[k_o \cdot 32 + k_i, j]$$

Reorder 后（$i, k_o, k_i, j$ → $i, k_o, k_i, j$）：
循环嵌套顺序变为：外层 $i$，然后 $k_o$，然后 $k_i$，最后 $j$。

Vectorize 后：$j$ 循环被向量化，一次处理 512 个元素（假设 512-wide vector）。

### 21.5.3 Lowering 内部的变换 Pass 链

`te.lower()` 和 `tir.lower()` 内部会执行一系列 TIR 变换 Pass：

```
TE Schedule AST
  │
  ▼ Schedule 实例化（应用 split/reorder/vectorize 等）
TIR PrimFunc（未优化）
  │
  ▼ InferFragment（推断线程绑定）
  ▼ LoopPartition（循环分区：处理 GPU grid/block 维度）
  ▼ StorageRewrite（存储重写：共享内存分配优化）
  ▼ InjectVirtualThread（虚拟线程注入）
  ▼ VectorizeLoop（向量化循环展开）
  ▼ UnrollLoop（循环展开）
  ▼ Simplify（常量折叠 + 表达式化简）
  │
  ▼
TIR PrimFunc（优化完成）
```

**每个 Pass 的作用详解**：

| Pass | 源文件 | 作用 | 典型场景 |
|------|--------|------|---------|
| `InferFragment` | `src/tir/transforms/` | 推断 GPU 的 thread/block 绑定 | `threadIdx.x` 注入 |
| `LoopPartition` | `src/tir/transforms/loop_partition.cc` | 将循环分区为 grid/block 结构 | GPU 并行化 |
| `StorageRewrite` | `src/tir/transforms/storage_rewrite.cc` | 合并和优化内存分配 | 共享内存优化 |
| `InjectVirtualThread` | `src/tir/transforms/inject_virtual_thread.cc` | 注入虚拟线程 | CPU 多线程 |
| `VectorizeLoop` | `src/tir/transforms/vectorize_loop.cc` | 向量化循环展开 | SIMD 指令生成 |
| `UnrollLoop` | `src/tir/transforms/unroll_loop.cc` | 循环展开 | 小循环的展开 |
| `Simplify` | `src/tir/transforms/simplify.cc` | 常量折叠 + 表达式化简 | 所有场景 |

### 21.5.4 LowerTIR Pass

在 `relay.build()` 的流程中，TE→TIR 的 lowering 由 `LowerTE` Pass 驱动：

```cpp
// src/relay/backend/te_compiler.cc
// 在 relay.build() 内部：
// 1. 对每个融合函数调用 CompileEngine::Lower() 得到 TE schedule
// 2. 调用 LowerTE Pass 将 TE schedule 实例化为 TIR
// 3. 对 TIR 应用优化 Pass 链
```

`LowerTE` Pass 遍历 Relay 模块中的每个 `Function`，将其内部的 TE 计算替换为 TIR PrimFunc。

### 21.5.5 Schedule 实例化的源码路径

Schedule 实例化的核心实现在 `src/te/schedule/schedule_ops.cc` 中：

```cpp
// src/te/schedule/schedule_ops.cc
// SchedulePostProc：将 Schedule 变换应用到 TE AST
class SchedulePostProc {
 public:
  // 应用所有变换
  Stmt Apply(const Schedule& sch) {
    // 1. 对每个 Stage，应用其变换序列
    for (auto& stage : sch->stages) {
      ApplyStage(stage);
    }
    // 2. 生成 TIR 循环嵌套
    return GenerateLoopNest(sch);
  }

 private:
  void ApplyStage(const Stage& stage) {
    // 应用 split、reorder、vectorize、parallel 等变换
    for (auto& transform : stage->transforms) {
      switch (transform->type) {
        case kSplit:
          ApplySplit(stage, transform);
          break;
        case kReorder:
          ApplyReorder(stage, transform);
          break;
        case kVectorize:
          ApplyVectorize(stage, transform);
          break;
        // ...
      }
    }
  }

  // 生成循环嵌套
  Stmt GenerateLoopNest(const Schedule& sch) {
    // 根据 leaf_iter_vars 生成嵌套的 for 循环
    // 每个 IterVar 对应一个 for 循环
    // reduce_axis 对应内层归约循环
  }
};
```

### 21.5.6 TIR PrimFunc 的结构

TIR PrimFunc 是 Lowering 的最终产物，它是一个完整的、可执行的函数描述：

```python
# TIR PrimFunc 的典型结构
# primfn(A: Buffer((128, 256), "float32"),
#        B: Buffer((256, 512), "float32"),
#        C: Buffer((128, 512), "float32"))
#   # 属性
#   attr = {"target": "llvm", "global_symbol": "matmul"}
#   # 循环嵌套
#   for i in parallel(0, 128):
#     for ko in range(0, 8):
#       for ki in range(0, 32):
#         # 归约初始化
#         C_local[0] = 0.0f
#         for j in vectorized(0, 512):
#           C_local[0] += A[i, ko*32+ki] * B[ko*32+ki, j]
#         # 写回结果
#         C[i, ko*32+ki] = C_local[0]
```

<div data-component="TEtoTIRLowering"></div>

### 21.5.7 GPU 与 CPU 的 TIR 差异

不同 target 的 TIR 输出有显著差异：

**CPU（LLVM）目标**：

```
primfn(...)
  for i in parallel(0, 128):     # 并行循环
    for j in range(0, 512):
      for k in vectorized(0, 256):  # SIMD 向量化
        C[i, j] += A[i, k] * B[k, j]
```

**GPU（CUDA）目标**：

```
primfn(...)
  # GPU 特有的 block/thread 结构
  attr = {"thread_extent": 128}  # blockIdx.x
  for i in thread_binding(0, 4, "threadIdx.x"):  # 线程绑定
    for j in range(0, 512):
      for k in range(0, 256):
        C[i, j] += A[i, k] * B[k, j]
```

---

## 21.6 `relay.build()` 的完整流程

### 21.6.1 端到端编译管线

`relay.build()` 是 TVM 编译的总入口，它协调了从 Relay IR 到可执行模块的完整流程：

```python
# relay.build() 的简化版伪代码
def relay.build(mod, target, params=None):
    # Phase 1: Relay 优化
    mod = relay.optimize(mod, target)
    # 包含：FuseOps, FoldConstant, AlterOpLayout, ...

    # Phase 2: Lowering（Relay → TE → TIR）
    # 2a: FuseOps 后的图 → 每个融合函数 → TE compute + schedule
    # 2b: TE schedule → TIR PrimFunc
    lowered_func = lower_module(mod, target)

    # Phase 3: CodeGen（TIR → 机器码）
    # 根据 target 选择 codegen（LLVM/CUDA/C...）
    lib = codegen(lowered_func, target)

    # Phase 4: 打包运行时模块
    # 包含：机器码 + 图结构 + 参数
    runtime_mod = pack_runtime(lib, mod)
    return runtime_mod
```

### 21.6.2 Phase 1：Relay 优化

在 lowering 之前，`relay.build()` 会先对 Relay 模块执行一系列优化 Pass：

```python
# relay.optimize() 内部执行的 Pass 序列
pass_seq = tvm.transform.Sequential([
    relay.transform.InferType(),          # 类型推断
    relay.transform.FoldConstant(),       # 常量折叠
    relay.transform.FuseOps(fuse_opt_level),  # 算子融合
    relay.transform.AlterOpLayout(),      # 布局变换（可选）
    relay.transform.CanonicalizeOps(),    # 算子规范化
    relay.transform.InferType(),          # 再次类型推断
])
mod = pass_seq(mod)
```

其中 `fuse_opt_level` 控制融合的激进程度：

| Level | 行为 |
|-------|------|
| 0 | 不融合 |
| 1 | 保守融合（只融合一对一映射） |
| 2 | 标准融合（推荐） |
| 3 | 激进融合（可能增加寄存器压力） |

**每个 Pass 的执行顺序与依赖关系**：

```
InferType()           ← 必须先执行，为后续 Pass 提供类型信息
  │
FoldConstant()        ← 依赖类型信息来确定常量
  │
FuseOps()             ← 依赖常量折叠后的图结构
  │
AlterOpLayout()       ← 可选，依赖融合后的图
  │
CanonicalizeOps()     ← 规范化算子表示
  │
InferType()           ← 再次推断类型，因为布局可能改变了形状
```

### 21.6.3 Phase 2：TE Lowering

Lowering 阶段将每个融合函数翻译为 TE schedule，再实例化为 TIR：

```
FuseOps 后的模块：
  Function 0: fused_conv2d_bn_relu(%x, %w, ...)  →  TE compute + schedule
  Function 1: fused_dense_softmax(%x, %w, ...)    →  TE compute + schedule
  main: 调用 Function 0 和 Function 1
```

**TE Lowering 的详细步骤**：

```python
# 伪代码：TE Lowering 过程
def lower_module(mod, target):
    # 1. 遍历模块中的所有函数
    for func_name, func in mod.functions.items():
        if is_fused_function(func):
            # 2. 构造 CCacheKey
            key = CCacheKey(func, target)

            # 3. 调用 CompileEngine::Lower
            cached = compile_engine.lower(key)

            # 4. 获取 TE compute 和 schedule
            te_outputs = cached.outputs
            te_schedule = cached.schedule

            # 5. Lower TE 到 TIR
            tir_func = tvm.lower(te_schedule, te_outputs, name=func_name)

            # 6. 替换原函数
            mod[func_name] = tir_func

    return mod
```

### 21.6.4 Phase 3：CodeGen

TIR PrimFunc 被发送到对应的 CodeGen 后端：

```cpp
// src/relay/backend/graph_executor_factory.cc
runtime::Module RelayBuildCreateJSON(
    const Function& func,
    const String& mod_name,
    const Map<String, IRModule>& lowered_funcs) {
  // 1. 序列化图结构为 JSON
  // 2. 收集所有 lowered TIR functions
  // 3. 调用目标 CodeGen 生成机器码
  // 4. 打包为 runtime::Module
}
```

**CodeGen 后端的选择**：

| Target | CodeGen 后端 | 输出格式 | 源文件 |
|--------|-------------|---------|--------|
| `llvm` | LLVM CodeGen | LLVM IR → 机器码 | `src/target/llvm/llvm_module.cc` |
| `cuda` | CUDA CodeGen | CUDA PTX | `src/target/source/codegen_cuda.cc` |
| `c` | C CodeGen | C 源码 | `src/target/source/codegen_c.cc` |
| `metal` | Metal CodeGen | Metal Shading Language | `src/target/source/codegen_metal.cc` |
| `vulkan` | Vulkan CodeGen | SPIR-V | `src/target/spirv/codegen_spirv.cc` |

### 21.6.5 Phase 4：运行时模块打包

编译的最终产物是一个 `runtime.Module`，包含：

```
runtime.Module
  ├── lib[0]: LLVM 编译的机器码（或 CUDA PTX）
  ├── graph.json: 图结构描述
  ├── params: 序列化的模型参数
  └── metadata: 编译元信息
```

**运行时模块的 C++ 实现**：

```cpp
// src/runtime/graph_executor/graph_executor_factory.cc
class GraphExecutorFactory : public runtime::ModuleNode {
 public:
  // 创建 GraphExecutor
  PackedFunc GetFunction(const String& name,
                         const ObjectPtr<Object>& sptr_to_self) override {
    if (name == "create") {
      // 返回一个创建 GraphExecutor 的函数
      return PackedFunc([this](TVMArgs args, TVMRetValue* rv) {
        *rv = CreateGraphExecutor(args[0]);
      });
    }
  }

 private:
  std::string graph_json_;           // 图 JSON
  std::vector<runtime::Module> modules_;  // 编译后的模块
  std::vector<NDArray> params_;      // 模型参数
};
```

<div data-component="BuildPipelineDiagram"></div>

### 21.6.6 relay.build() 的 C++ 入口

```cpp
// src/relay/backend/build_module.cc
class RelayBuildModule : public runtime::ModuleNode {
 public:
  void Build(const IRModule& mod,
             const Target& target,
             const Map<String, Target>& target_host,
             const Optional<Map<String, NDArray>>& params) {
    // Phase 1: Relay 优化
    mod_ = Optimize(mod, target, params);

    // Phase 2: Lowering
    // 2a: FuseOps (已在 Optimize 中完成)
    // 2b: TE Lowering (CompileEngine::Lower)
    // 2c: TIR Lowering (LowerTE Pass)
    lowered_funcs_ = LowerModule(mod_, target);

    // Phase 3: CodeGen
    // 为每个 target 生成代码
    for (auto& kv : lowered_funcs_) {
      auto codegen = CreateCodegen(kv.first);
      codegen->AddFunction(kv.second);
      modules_.push_back(codegen->Finish());
    }

    // Phase 4: 打包
    // 序列化图 JSON + 参数
    graph_json_ = CreateGraphJSON(mod_);
    params_ = SerializeParams(params);
  }

 private:
  IRModule mod_;
  String graph_json_;
  std::vector<runtime::Module> modules_;
  Map<String, NDArray> params_;
};
```

### 21.6.7 编译时间分析

典型模型的编译时间分布：

| 阶段 | 时间占比 | 主要开销 |
|------|---------|---------|
| Relay 优化 | 10-15% | FuseOps, InferType |
| TE Lowering | 30-40% | CompileEngine::Lower, TOPI 调用 |
| TIR Lowering | 15-20% | Schedule 实例化, TIR Passes |
| CodeGen | 20-30% | LLVM 编译 (最耗时) |
| 打包 | 5-10% | JSON 序列化, 参数序列化 |

$$T_{\text{total}} = T_{\text{relay}} + T_{\text{te}} + T_{\text{tir}} + T_{\text{codegen}} + T_{\text{pack}}$$

---

## 21.7 图序列化：JSON 格式

### 21.7.1 图的 JSON 表示

`relay.build()` 会将编译后的计算图序列化为 JSON 格式，供 Graph Runtime 在运行时加载：

```json
{
  "nodes": [
    {"op": "null", "name": "x",        "inputs": []},
    {"op": "null", "name": "weight_1",  "inputs": []},
    {"op": 1,      "name": "fused_0",   "inputs": [[0, 0, 0], [1, 0, 0]]},
    {"op": 2,      "name": "fused_1",   "inputs": [[2, 0, 0]]}
  ],
  "arg_nodes": [0, 1],
  "heads": [[3, 0, 0]],
  "attrs": {
    "storage_id":  ["list_int", [0, 1, 2, 3]],
    "dtype":       ["list_int", [0, 0, 0, 0]],
    "dltype":      ["list_str", ["float32", "float32", "float32", "float32"]],
    "shape":       ["list_shape", [[1,3,224,224], [64,3,7,7], [1,64,112,112], [1,1000]]]
  }
}
```

### 21.7.2 节点与边的语义

- **`nodes`**：每个节点要么是输入占位符（`op: "null"`），要么是一个融合算子（`op: N`，指向第 N 个编译函数）
- **`inputs`**：`[[node_id, version, index], ...]`，表示数据依赖
- **`arg_nodes`**：输入参数的节点 ID
- **`heads`**：输出节点
- **`attrs`**：存储 ID、数据类型、形状等元信息

**JSON 中的节点类型**：

| 节点类型 | `op` 字段 | 含义 | 示例 |
|---------|----------|------|------|
| 输入节点 | `"null"` | 模型输入或常量参数 | 输入 tensor, 权重 |
| 算子节点 | `N` (整数) | 第 N 个编译后的融合函数 | fused_conv2d_bn_relu |

**边的表示**：`[node_id, version, index]`
- `node_id`：源节点 ID
- `version`：节点版本（通常为 0）
- `index`：输出 tensor 的索引（对于 tuple 输出）

### 21.7.3 存储分配计划

JSON 中的 `storage_id` 字段记录了**存储复用计划**：

```
节点 0 (x):        storage_id = 0
节点 1 (weight_1):  storage_id = 1
节点 2 (fused_0):   storage_id = 2
节点 3 (fused_1):   storage_id = 3
```

如果两个节点的 `storage_id` 相同，说明它们共享同一块内存（生命周期不重叠时可复用）。

**存储复用的数学模型**：

设节点 $i$ 的生命周期为 $[b_i, e_i]$（从开始使用到不再被引用），大小为 $s_i$。存储复用问题可以形式化为：

$$\min \sum_{j=1}^{m} S_j \quad \text{s.t.} \quad \forall i: \exists j, \text{node}_i \mapsto \text{storage}_j$$

其中 $S_j = \max_{i \mapsto j} s_i$，且如果 $i_1 \mapsto j$ 且 $i_2 \mapsto j$，则 $[b_{i_1}, e_{i_1}] \cap [b_{i_2}, e_{i_2}] = \emptyset$。

这是一个区间图着色问题，TVM 使用贪心算法求解。

### 21.7.4 图 JSON 的生成过程

```cpp
// src/relay/backend/graph_executor_factory.cc
std::string CreateGraphJSON(const IRModule& mod) {
  // 1. 遍历 main 函数的调用图
  // 2. 为每个节点分配 ID
  // 3. 记录节点间的依赖关系
  // 4. 执行存储分配（PlanMemory）
  // 5. 序列化为 JSON

  // PlanMemory 在 src/relay/backend/plan_memory.cc 中实现
  Map<Expr, Integer> storage_map = PlanMemory(mod);

  // 生成 JSON
  GraphJsonGenerator gen;
  return gen.Generate(mod, storage_map);
}
```

### 21.7.5 复杂图的 JSON 示例

对于一个包含分支和合并的模型：

```json
{
  "nodes": [
    {"op": "null", "name": "input",     "inputs": []},
    {"op": "null", "name": "conv1_w",   "inputs": []},
    {"op": "null", "name": "conv2_w",   "inputs": []},
    {"op": 0,      "name": "fused_conv1", "inputs": [[0, 0, 0], [1, 0, 0]]},
    {"op": 1,      "name": "fused_conv2", "inputs": [[0, 0, 0], [2, 0, 0]]},
    {"op": "null", "name": "add_w",     "inputs": []},
    {"op": 2,      "name": "fused_add", "inputs": [[3, 0, 0], [4, 0, 0], [5, 0, 0]]}
  ],
  "arg_nodes": [0, 1, 2, 5],
  "heads": [[6, 0, 0]],
  "attrs": {
    "storage_id": ["list_int", [0, 1, 2, 3, 4, 5, 6]],
    "shape": ["list_shape", [
      [1, 3, 224, 224], [64, 3, 7, 7], [128, 3, 5, 5],
      [1, 64, 112, 112], [1, 128, 112, 112], [1], [1, 128, 112, 112]
    ]]
  }
}
```

<div data-component="GraphJSONDiagram"></div>

---

## 21.8 参数注入与常量折叠

### 21.8.1 参数注入

`relay.build()` 接受一个 `params` 字典，将模型参数（如权重）注入到编译产物中：

```python
params = {"weight_1": np.random.randn(64, 3, 7, 7).astype("float32")}

with tvm.target.Target("llvm"):
    lib = relay.build(mod, target="llvm", params=params)
    # params 被序列化并嵌入到 lib 中
```

参数注入发生在 FuseOps **之后**，因此常量权重可以被直接内联到融合函数中。

**参数注入的实现细节**：

```cpp
// src/relay/backend/build_module.cc
void RelayBuildModule::BindParams(
    const IRModule& mod,
    const Map<String, NDArray>& params) {
  // 1. 遍历模块中的所有 FreeVar
  // 2. 如果 FreeVar 的名字在 params 中，替换为 Constant
  // 3. 更新模块
  for (auto& kv : params) {
    std::string name = kv.first;
    NDArray value = kv.second;
    // 查找对应的变量
    Var var = FindVarByName(mod, name);
    if (var.defined()) {
      // 创建 Constant 节点
      Constant const_node = Constant(value);
      // 替换所有使用该变量的地方
      mod = SubstituteVar(mod, var, const_node);
    }
  }
}
```

### 21.8.2 常量折叠的交互

常量折叠 Pass 会在 FuseOps 之前将可静态计算的表达式求值：

```python
# 常量折叠前
x = relay.const(2.0)
y = relay.const(3.0)
z = x + y  # 可以在编译期计算

# 常量折叠后
z = relay.const(5.0)  # 直接替换为常量
```

这对于量化模型尤为重要：量化参数（scale、zero-point）通常是常量，可以在编译期完全折叠。

**常量折叠的实现**：

```cpp
// src/relay/transforms/fold_constant.cc
class ConstantFolder : public ExprRewriter {
 public:
  Expr Rewrite_(const CallNode* call, const Expr& post) override {
    // 检查是否所有参数都是常量
    bool all_const = true;
    for (auto arg : call->args) {
      if (!arg.as<ConstantNode>()) {
        all_const = false;
        break;
      }
    }

    if (all_const) {
      // 1. 将常量参数转换为 NDArray
      // 2. 使用 Relay interpreter 执行
      // 3. 将结果包装为 Constant
      return EvaluateConstant(call);
    }
    return post;
  }
};
```

### 21.8.3 常量折叠与量化的交互

量化模型中的常量折叠特别重要：

```python
# 量化模型中的常量折叠示例
# 量化参数通常在编译期确定
quantize_scale = relay.const(0.00784314)  # 量化 scale
zero_point = relay.const(128)              # 零点

# 量化操作
quantized = relay.qnn.quantize(input, scale, zero_point)

# 反量化操作
dequantized = relay.qnn.dequantize(quantized, scale, zero_point)

# 常量折叠可以将 quantize + dequantize 合并为 identity
# 如果 scale 和 zero_point 是常量
```

### 21.8.4 参数序列化格式

参数被序列化为二进制格式，嵌入到编译产物中：

```
运行时模块结构：
  ├── lib: 编译后的机器码
  ├── graph.json: 图结构
  └── params:
      ├── "weight_1": [NDArray header + raw data]
      ├── "bias_1":   [NDArray header + raw data]
      └── ...
```

每个参数的序列化格式：

```
NDArray 序列化格式：
  ├── DLTensor header (40 bytes)
  │   ├── data pointer
  │   ├── ndim
  │   ├── shape[ndim]
  │   └── strides[ndim]
  ├── dtype (4 bytes)
  └── raw data (product(shape) * dtype_size bytes)
```

---

## 21.9 多 Target 编译

### 21.9.1 异构编译场景

在实际部署中，模型的不同部分可能需要在不同硬件上执行：

```python
# 场景：Conv2d 在 GPU 上执行，后处理在 CPU 上执行
mod = tvm.IRModule()
# ... 定义模型 ...

# 使用 Relay 的 device_copy 算子处理跨设备数据传输
with tvm.target.Target("cuda"):
    # GPU 部分
    gpu_lib = relay.build(gpu_part, target="cuda")

with tvm.target.Target("llvm"):
    # CPU 部分
    cpu_lib = relay.build(cpu_part, target="llvm")
```

### 21.9.2 Target 注解

Relay 支持通过 `on_device` 标注指定每个算子的执行设备：

```python
# 将 conv2d 标注在 GPU 上执行
conv = relay.annotation.on_device(
    relay.nn.conv2d(x, w),
    tvm.target.Target("cuda")
)

# 将 softmax 标注在 CPU 上执行
out = relay.annotation.on_device(
    relay.nn.softmax(conv),
    tvm.target.Target("llvm")
)
```

编译器会根据标注自动将图分割为多个子图，分别编译。

### 21.9.3 图分割与设备间通信

当图被分割为多个子图时，需要在设备间传输数据：

```
GPU 子图:
  input → conv2d → relu → [device_copy to CPU] → CPU 子图:
                                            input → softmax → output

数据流:
  GPU memory → PCIe transfer → CPU memory
```

**设备间通信的开销**：

$$T_{\text{copy}} = T_{\text{launch}} + \frac{\text{data\_size}}{\text{bandwidth}}$$

其中 $T_{\text{launch}}$ 是 DMA 启动开销（通常 5-10 μs），$\text{bandwidth}$ 是 PCIe 带宽（PCIe 3.0: ~12 GB/s, PCIe 4.0: ~24 GB/s）。

### 21.9.4 多 Target 的编译流程

```python
# 多 Target 编译的完整示例
import tvm
from tvm import relay

# 定义模型
x = relay.var("x", shape=(1, 3, 224, 224))
conv = relay.nn.conv2d(x, w1)
relu = relay.nn.relu(conv)
# ... 更多层 ...

# 使用 PartitionGraph 按 target 分区
mod = relay.transform.PartitionGraph()(mod)

# 分别编译每个子图
for func_name, func in mod.functions.items():
    target = func.attrs["Compiler"]  # 获取 target
    with tvm.target.Target(target):
        lib = relay.build(tvm.IRModule.from_expr(func), target=target)
```

<div data-component="MultiTargetDiagram"></div>

---

## 21.10 调试与可视化

### 21.10.1 查看各阶段 IR

```python
import tvm
from tvm import relay

# 查看 FuseOps 后的图
mod_after_fuse = relay.transform.FuseOps()(mod)
print("After FuseOps:")
print(mod_after_fuse)

# 查看 Lower 后的 TIR
with tvm.target.Target("llvm"):
    mod_lowered = tvm.IRModule.from_expr(
        relay.transform.Legalize()(mod["main"])
    )
    print("After Lower:")
    print(mod_lowered)
```

### 21.10.2 编译日志

设置环境变量可以查看详细的编译日志：

```bash
# 查看 Pass 执行顺序
export TVM_LOG_DEBUG=1

# 查看 FuseOps 的融合决策
export TVM_FUSE_OPS_LOG=1

# 查看 TE Compiler 的缓存命中情况
export TVM_TE_COMPILER_LOG=1

# 查看 CodeGen 的输出
export TVM_CODEGEN_LOG=1
```

### 21.10.3 Graph Runtime JSON 可视化

```python
import json

# 从编译产物中提取图 JSON
graph_json = lib.get_lib().get_source("graph")
graph = json.loads(graph_json)

# 可视化节点与边
for i, node in enumerate(graph["nodes"]):
    print(f"Node {i}: op={node['op']}, name={node['name']}")
    for inp in node["inputs"]:
        print(f"  ← Node {inp[0]}")
```

### 21.10.4 使用 PassContext 调试

```python
# 使用 PassContext 控制 Pass 执行和调试
with tvm.transform.PassContext(
    opt_level=3,
    config={"relay.FuseOps.fuse_opt_level": 2},
    instruments=[tvm.instrument.PassTimingInstrument()]
):
    lib = relay.build(mod, target="llvm")

# 查看 Pass 执行时间
timing = tvm.instrument.PassTimingInstrument()
print(timing.render())
```

### 21.10.5 IR 转储与比较

```python
# 将各阶段 IR 转储到文件
def dump_ir(mod, filename):
    with open(filename, "w") as f:
        f.write(mod.astext())

# 转储各阶段
dump_ir(mod_original, "00_original.txt")
dump_ir(mod_after_infer, "01_infer_type.txt")
dump_ir(mod_after_fold, "02_fold_const.txt")
dump_ir(mod_after_fuse, "03_fuse_ops.txt")
dump_ir(mod_lowered, "04_lowered.txt")
```

<div data-component="CompilationDebugger"></div>

---

## 21.11 高级话题

### 21.11.1 自定义 Lowering 行为

用户可以通过注册自定义的 `FTVMCompute` 和 `FTVMSchedule` 来 override 默认的 lowering 行为：

```python
# 为自定义算子注册 TE compute
@relay.op.register("my_custom_op", "FTVMCompute")
def compute_my_op(attrs, inputs, output_type):
    data = inputs[0]
    return te.compute(data.shape, lambda *i: data(*i) * 2, name="output")

# 为自定义算子注册 TE schedule
@relay.op.register_schedule("my_custom_op")
def schedule_my_op(attrs, outs, target):
    s = te.create_schedule(outs[0].op)
    # 自定义调度策略
    return s
```

**自定义算子的完整注册流程**：

```python
# 1. 定义算子属性
class MyOpAttrs(tvm.ir.Attrs):
    def __init__(self, scale):
        self.scale = scale

# 2. 注册算子
@tvm.ir.register_op_attr("my_custom_op", "FTVMCompute")
def my_op_compute(attrs, inputs, out_type):
    """定义计算语义"""
    data = inputs[0]
    scale = attrs.scale
    return te.compute(
        data.shape,
        lambda *i: data(*i) * scale,
        name="output"
    )

@tvm.ir.register_op_attr("my_custom_op", "FTVMSchedule")
def my_op_schedule(attrs, outs, target):
    """定义调度策略"""
    s = te.create_schedule(outs[0].op)
    if target.kind.name == "cuda":
        # GPU 调度
        i, j = s[outs[0]].op.axis
        s[outs[0]].bind(i, te.thread_axis("blockIdx.x"))
        s[outs[0]].bind(j, te.thread_axis("threadIdx.x"))
    else:
        # CPU 调度
        s[outs[0]].parallel(s[outs[0]].op.axis[0])
    return s
```

### 21.11.2 编译缓存的持久化

TE Compiler 的缓存可以通过 `tvm.runtime.save_compiled_model()` 保存到磁盘，避免重复编译：

```python
# 保存编译结果
lib.export_library("model.tar")

# 加载编译结果（无需重新编译）
loaded_lib = tvm.runtime.load_module("model.tar")
gmod = graph_executor.GraphModule(loaded_lib["default"](dev))
```

### 21.11.3 MetaSchedule 集成

当启用 MetaSchedule 时，lowering 管线中的 FTVMSchedule 会被替换为 MetaSchedule 的搜索结果：

```python
from tvm import meta_schedule as ms

# 使用 MetaSchedule 调优
database = ms.tune_tir(
    mod=lowered_mod,
    target="llvm --mcpu=core-avx2",
    max_trials_global=1000,
)

# 调优结果会被注入到 lowering 管线中
# 替代默认的 FTVMSchedule
```

**MetaSchedule 与默认 Schedule 的对比**：

| 方面 | 默认 FTVMSchedule | MetaSchedule |
|------|------------------|--------------|
| 生成方式 | 手工编写模板 | 自动搜索 |
| 性能 | 通用，非最优 | 针对特定 shape 优化 |
| 编译时间 | 快（毫秒级） | 慢（需要搜索） |
| 适用场景 | 开发调试 | 生产部署 |
| 可定制性 | 需要修改 TOPI 代码 | 通过配置调整 |

### 21.11.4 动态形状的支持

TVM 支持动态形状的编译，但需要特殊处理：

```python
# 动态形状的 Relay 表达式
batch = relay.Any()  # 动态 batch size
x = relay.var("x", shape=(batch, 3, 224, 224))
w = relay.var("w", shape=(64, 3, 7, 7))
conv = relay.nn.conv2d(x, w)

# 编译时需要指定静态形状
# 或使用 shape 函数推断
with tvm.target.Target("llvm"):
    # 使用 SpecifiedInputBound 指定形状范围
    lib = relay.build(mod, target="llvm",
                      params=params,
                      shape_inputs={"x": (1, 3, 224, 224)})
```

### 21.11.5 编译流水线的并行化

TVM 支持编译流水线的并行化，同时编译多个算子：

```python
import tvm
from tvm import relay

# 启用并行编译
with tvm.transform.PassContext(
    config={"relay.backend.use_parallel_compile": True}
):
    lib = relay.build(mod, target="llvm")
```

---

## 21.12 本章小结

本章深入分析了 TVM 从 Relay IR 到可执行代码的完整 Lowering 管线：

1. **FuseOps**：将计算图划分为可融合的算子子图，减少内存带宽消耗
2. **CompileEngine**：将 Relay 算子映射为 TE compute + schedule
3. **TE Compiler**：管理编译缓存，支持外部代码生成器
4. **TIR Lowering**：将 TE schedule 实例化为具体的 TIR 循环代码
5. **`relay.build()`**：协调整个编译管线，从 Relay IR 到运行时模块

关键数据流：

```
Relay Module
  → FuseOps (图分区)
  → CompileEngine::Lower (Relay→TE)
  → TE Compiler (缓存 + JIT)
  → LowerTE (TE→TIR)
  → TIR Passes (优化)
  → CodeGen (TIR→机器码)
  → runtime::Module (可执行模块)
```

**核心概念回顾**：

| 概念 | 描述 | 关键源文件 |
|------|------|-----------|
| FuseOps | 图分区算法 | `src/relay/transforms/fuse_ops.cc` |
| CompileEngine | Relay→TE 映射 | `src/relay/backend/compile_engine.cc` |
| TE Compiler | 缓存管理 | `src/relay/backend/te_compiler.cc` |
| LowerTE | TE→TIR 转换 | `src/relay/backend/te_compiler.cc` |
| CodeGen | TIR→机器码 | `src/target/llvm/`, `src/target/source/` |

**关键数学公式**：

$$\text{Fusion Speedup} = \frac{T_{\text{launch}} \cdot k + B_{\text{mem}} \cdot N \cdot (k-1)}{T_{\text{launch}} \cdot 1 + B_{\text{mem}} \cdot N \cdot 1}$$

$$\text{Bandwidth Saving} = 1 - \frac{2}{2k - 1}$$

$$T_{\text{total}} = T_{\text{relay}} + T_{\text{te}} + T_{\text{tir}} + T_{\text{codegen}} + T_{\text{pack}}$$

<div data-component="ChapterSummaryDiagram"></div>

---

## 21.13 常见陷阱与最佳实践

### 21.13.1 FuseOps 的常见陷阱

**陷阱 1：过度融合导致寄存器溢出**

```
问题：fuse_opt_level=3 时，融合区域过大，导致 GPU 寄存器不足
症状：性能下降，出现大量 local memory 访问
解决：降低 fuse_opt_level 到 2，或手动控制融合边界
```

```python
# 错误：使用过高的融合级别
with tvm.target.Target("cuda"):
    mod_fused = relay.transform.FuseOps(fuse_opt_level=3)(mod)

# 正确：使用标准融合级别
with tvm.target.Target("cuda"):
    mod_fused = relay.transform.FuseOps(fuse_opt_level=2)(mod)
```

**陷阱 2：忽略分支节点的融合限制**

```
问题：期望 conv2d 的输出被融合到两个分支中
现实：conv2d 的输出被多个消费者使用时，不会被融合
```

```python
# 示例：分支导致的融合问题
x = relay.var("x", shape=(1, 3, 224, 224))
conv = relay.nn.conv2d(x, w)
# conv 被两个分支使用，不会被融合到任何一个分支
branch1 = relay.nn.relu(conv)
branch2 = relay.sigmoid(conv)
```

**陷阱 3：动态形状导致缓存失效**

```
问题：每次推理 batch size 不同，导致 CompileEngine 缓存频繁失效
症状：编译时间变长，内存使用增加
解决：固定 batch size，或使用 shape bucketing
```

### 21.13.2 CompileEngine 的常见陷阱

**陷阱 1：未注册自定义算子的 FTVMCompute**

```
问题：自定义算子没有注册 FTVMCompute
症状：编译时报错 "FTVMCompute not registered for op xxx"
解决：确保自定义算子同时注册 FTVMCompute 和 FTVMSchedule
```

**陷阱 2：缓存 key 不包含形状信息**

```
问题：不同形状的同一算子共享缓存条目
症状：运行时 shape 不匹配
现实：TVM 的缓存 key 已包含形状，此问题通常不会发生
```

**陷阱 3：外部 codegen 与 CompileEngine 的冲突**

```
问题：同时使用外部 codegen 和默认 CompileEngine
症状：部分算子未被正确编译
解决：确保 PartitionGraph Pass 正确分区
```

### 21.13.3 TE Lowering 的常见陷阱

**陷阱 1：Schedule 变换顺序错误**

```python
# 错误：先 vectorize 后 split
s[C].vectorize(j)  # j 是原始轴，已被 split
ko, ki = s[C].split(k, factor=32)

# 正确：先 split 后 vectorize
ko, ki = s[C].split(k, factor=32)
s[C].vectorize(j)
```

**陷阱 2：忽略 reduce axis 的位置**

```python
# 错误：reorder 时忽略 reduce axis
s[C].reorder(i, j, k)  # k 是 reduce axis，应该在内层

# 正确：reorder 时考虑 reduce axis
s[C].reorder(i, ko, ki, j)  # reduce axis 在内层
```

**陷阱 3：GPU 调度时未绑定线程**

```python
# 错误：GPU 调度时未绑定线程
s = te.create_schedule(C.op)

# 正确：GPU 调度时绑定线程
s = te.create_schedule(C.op)
i, j = s[C].op.axis
s[C].bind(i, te.thread_axis("blockIdx.x"))
s[C].bind(j, te.thread_axis("threadIdx.x"))
```

### 21.13.4 relay.build() 的常见陷阱

**陷阱 1：忘记设置 target context**

```python
# 错误：未设置 target context
lib = relay.build(mod, target="llvm")

# 正确：设置 target context
with tvm.target.Target("llvm"):
    lib = relay.build(mod, target="llvm")
```

**陷阱 2：params 字典的 key 与变量名不匹配**

```python
# 错误：params key 与 Relay 变量名不匹配
params = {"weight": w_np}  # key 是 "weight"
x = relay.var("x")
w = relay.var("weight_1")  # 变量名是 "weight_1"

# 正确：确保 key 匹配
params = {"weight_1": w_np}
```

**陷阱 3：编译后修改模型**

```python
# 错误：编译后修改模型
lib = relay.build(mod, target="llvm")
mod["main"] = new_func  # 不会影响已编译的 lib

# 正确：修改后重新编译
mod["main"] = new_func
lib = relay.build(mod, target="llvm")
```

<div data-component="CommonPitfallsDiagram"></div>

---

## 21.14 练习题

### 练习 1：理解 FuseOps 的融合决策

**题目**：给定以下 Relay 计算图，预测 FuseOps 的融合结果（fuse_opt_level=2）：

```python
x = relay.var("x", shape=(1, 3, 224, 224))
w1 = relay.var("w1", shape=(64, 3, 7, 7))
w2 = relay.var("w2", shape=(64, 64, 3, 3))

conv1 = relay.nn.conv2d(x, w1, strides=(2, 2), padding=(3, 3))
bn1 = relay.nn.batch_norm(conv1, gamma1, beta1, mean1, var1)
relu1 = relay.nn.relu(bn1[0])

conv2 = relay.nn.conv2d(relu1, w2, padding=(1, 1))
bn2 = relay.nn.batch_norm(conv2, gamma2, beta2, mean2, var2)
relu2 = relay.nn.relu(bn2[0])

pool = relay.nn.max_pool2d(relu2, pool_size=(2, 2), strides=(2, 2))
out = relay.nn.softmax(pool)
```

**问题**：
1. 预测最终的融合区域划分
2. 画出融合后的调用图
3. 解释 softmax 和 max_pool2d 为什么可能不在同一个融合区域

<details>
<summary>参考答案</summary>

**融合区域划分**：

```
融合区域 1: fused_conv1_bn1_relu1
  - conv1 → bn1 → relu1
  - 模式: kOutEWiseFusable → kOutEWiseFusable → kElemWise

融合区域 2: fused_conv2_bn2_relu2_pool
  - conv2 → bn2 → relu2 → max_pool2d
  - 模式: kOutEWiseFusable → kOutEWiseFusable → kElemWise → kOpaque(kCommReduce)

融合区域 3: softmax (独立)
  - softmax 是 kCommReduce，只能作为融合区域的尾部
  - 但 max_pool2d 已经是区域 2 的尾部
  - 因此 softmax 独立成一个区域
```

</details>

### 练习 2：CompileEngine 的缓存行为

**题目**：分析以下代码的 CompileEngine 缓存行为：

```python
# 场景 1：相同 shape 的两次编译
x1 = relay.var("x", shape=(1, 3, 224, 224))
w1 = relay.var("w", shape=(64, 3, 7, 7))
conv1 = relay.nn.conv2d(x1, w1)

x2 = relay.var("x", shape=(1, 3, 224, 224))
w2 = relay.var("w", shape=(64, 3, 7, 7))
conv2 = relay.nn.conv2d(x2, w2)

# 场景 2：不同 shape 的两次编译
x3 = relay.var("x", shape=(8, 3, 224, 224))
w3 = relay.var("w", shape=(64, 3, 7, 7))
conv3 = relay.nn.conv2d(x3, w3)
```

**问题**：
1. 场景 1 中，两个 conv2d 的 CCacheKey 是否相同？
2. 场景 2 中，conv3 与 conv1 的 CCacheKey 是否相同？
3. 如果在同一个 `relay.build()` 中编译，缓存会如何工作？

<details>
<summary>参考答案</summary>

1. **场景 1**：两个 conv2d 的 CCacheKey **相同**（相同的算子、相同的形状、相同的 target），因此只会编译一次，第二次从缓存读取。

2. **场景 2**：conv3 与 conv1 的 CCacheKey **不同**（batch size 不同：1 vs 8），因此会分别编译。

3. **缓存行为**：
   - 第一次编译 conv1 时，Cache miss → 编译并缓存
   - 编译 conv2 时，Cache hit → 直接返回缓存结果
   - 编译 conv3 时，Cache miss → 编译并缓存
   - 总共编译 2 次，缓存命中 1 次

</details>

### 练习 3：TE Schedule 的变换序列

**题目**：给定以下 TE 计算，写出正确的 Schedule 变换序列：

```python
A = te.placeholder((1024, 1024), name="A")
B = te.placeholder((1024, 1024), name="B")
k = te.reduce_axis((0, 1024), name="k")
C = te.compute((1024, 1024), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
```

**要求**：
1. 将 i 轴 split 为 32 的倍数
2. 将 k 轴 split 为 4 的倍数
3. Reorder 为 (i_outer, k_outer, i_inner, k_inner, j)
4. Vectorize j 轴
5. Parallel i_outer 轴

<details>
<summary>参考答案</summary>

```python
s = te.create_schedule(C.op)
i, j = s[C].op.axis
k = s[C].op.reduce_axis[0]

# Step 1: Split i
i_outer, i_inner = s[C].split(i, factor=32)

# Step 2: Split k
k_outer, k_inner = s[C].split(k, factor=4)

# Step 3: Reorder
s[C].reorder(i_outer, k_outer, i_inner, k_inner, j)

# Step 4: Vectorize j
s[C].vectorize(j)

# Step 5: Parallel i_outer
s[C].parallel(i_outer)
```

</details>

### 练习 4：调试 Lowering 错误

**题目**：以下代码在 `relay.build()` 时报错。分析错误原因并修复：

```python
import tvm
from tvm import relay

# 自定义算子
@relay.op.register("my_scale", "FTVMCompute")
def compute_scale(attrs, inputs, output_type):
    data = inputs[0]
    return te.compute(data.shape, lambda *i: data(*i) * 2, name="output")

# 使用自定义算子
x = relay.var("x", shape=(1, 3, 224, 224))
y = relay.op.get("my_scale")(x)
mod = tvm.IRModule.from_expr(y)

# 编译
with tvm.target.Target("llvm"):
    lib = relay.build(mod, target="llvm")
```

**错误信息**：
```
TVMError: Schedule not registered for my_scale
```

<details>
<summary>参考答案</summary>

**问题**：自定义算子只注册了 `FTVMCompute`，没有注册 `FTVMSchedule`。

**修复**：

```python
@relay.op.register_schedule("my_scale")
def schedule_scale(attrs, outs, target):
    s = te.create_schedule(outs[0].op)
    return s
```

</details>

### 练习 5：分析图 JSON

**题目**：给定以下图 JSON，画出对应的计算图：

```json
{
  "nodes": [
    {"op": "null", "name": "x", "inputs": []},
    {"op": "null", "name": "w", "inputs": []},
    {"op": 0, "name": "fused_conv_relu", "inputs": [[0, 0, 0], [1, 0, 0]]},
    {"op": "null", "name": "w2", "inputs": []},
    {"op": 1, "name": "fused_dense", "inputs": [[2, 0, 0], [3, 0, 0]]}
  ],
  "arg_nodes": [0, 1, 3],
  "heads": [[4, 0, 0]],
  "attrs": {
    "storage_id": ["list_int", [0, 1, 2, 3, 4]],
    "shape": ["list_shape", [[1,3,224,224], [64,3,7,7], [1,64,112,112], [1000,64,112,112], [1,1000]]]
  }
}
```

**问题**：
1. 画出计算图
2. 标注每个节点的 storage_id
3. 说明哪些节点可以共享内存

<details>
<summary>参考答案</summary>

**计算图**：

```
x (storage=0) ──┐
                 ├── fused_conv_relu (storage=2) ──┐
w (storage=1) ──┘                                    ├── fused_dense (storage=4) → output
                                                     │
w2 (storage=3) ──────────────────────────────────────┘
```

**存储分析**：
- 节点 0 (x): storage_id = 0
- 节点 1 (w): storage_id = 1
- 节点 2 (fused_conv_relu): storage_id = 2
- 节点 3 (w2): storage_id = 3
- 节点 4 (fused_dense): storage_id = 4

**内存共享**：当前的 storage_id 都不同，没有共享。但如果某些节点的生命周期不重叠，可以合并 storage_id 以节省内存。

</details>

<div data-component="PracticeExercises"></div>

---

## 21.15 扩展阅读

### 21.15.1 相关论文

1. **TVM: An Automated End-to-End Optimizing Compiler for Deep Learning** (OSDI 2018)
   - TVM 的原始论文，详细描述了 TE 和 AutoTVM

2. **Ansor: Generating High-Performance Tensor Programs for Deep Learning** (OSDI 2020)
   - MetaSchedule 的前身，自动搜索调度策略

3. **Relay: A High-Level IR for Deep Learning** (2018)
   - Relay IR 的设计与实现

### 21.15.2 TVM 源码导读

| 模块 | 关键文件 | 行数 | 复杂度 |
|------|---------|------|--------|
| FuseOps | `src/relay/transforms/fuse_ops.cc` | ~800 | 高 |
| CompileEngine | `src/relay/backend/compile_engine.cc` | ~600 | 中 |
| TE Compiler | `src/relay/backend/te_compiler.cc` | ~500 | 中 |
| Schedule 实例化 | `src/te/schedule/schedule_ops.cc` | ~400 | 中 |
| TIR Passes | `src/tir/transforms/` | ~2000+ | 高 |

### 21.15.3 进一步学习路径

1. **理解 TOPI 算子库**：阅读 `python/tvm/topi/` 中的算子实现
2. **学习 MetaSchedule**：了解自动调度搜索的工作原理
3. **研究 CodeGen**：深入 LLVM/CUDA CodeGen 的实现
4. **自定义 Pass**：学习如何编写自定义的 Relay/TIR Pass

<div data-component="FurtherReading"></div>

---

下一章我们将深入 Graph Runtime，了解编译后的模型是如何被高效执行的。

---

## 21.99 文字内容强化：Relay 到 TE 的工程化阅读补充

Relay 到 TE 的下降过程是 TVM 前端优化与底层调度之间的桥梁，理解这条路径能帮助读者定位算子选择、融合和 TIR 生成问题。

### 21.99.1 代码解读：从片段回到主流程

原有代码块中的策略函数、TOPI compute 和 schedule 需要连起来读。
控制流从 Relay Call 进入 op strategy，选择实现后生成 TE Tensor，再经 schedule 降到 TIR。
数据结构上 Relay Expr 保存图语义，TE Tensor 保存计算定义，PrimFunc 保存可编译内核。
代码块中的变量名、函数名和类名不应孤立记忆，而应放回编译流水线中理解。
读者可以先判断代码块处在构建期、优化期、代码生成期还是运行期。
构建期代码通常负责收集信息，优化期代码负责改写 IR，代码生成期代码负责降低表示，运行期代码负责执行与资源管理。
一旦阶段判断正确，许多看似相似的数据结构就能区分出职责边界。

### 21.99.2 源码阅读路径

阅读 apache/tvm 源码时，建议按下面顺序推进，而不是直接在全仓库搜索 Relay 到 TE。
第 1 步，阅读 `src/relay/backend/te_compiler.cc`，目标是确认这一层暴露的主要接口和被谁调用。
第 2 步，阅读 `src/relay/backend/compile_engine.cc`，目标是确认这一层暴露的主要接口和被谁调用。
第 3 步，阅读 `python/tvm/relay/op/strategy/`，目标是确认这一层暴露的主要接口和被谁调用。
第 4 步，阅读 `python/tvm/topi/`，目标是确认这一层暴露的主要接口和被谁调用。
第 5 步，阅读 `src/tir/transforms/`，目标是确认这一层暴露的主要接口和被谁调用。
完成主路径后，再阅读相邻测试目录，测试通常比注释更清楚地展示了设计者希望维持的不变量。
如果遇到注册表入口，应记录注册名、C++ 实现函数、Python 包装函数和最终用户 API 四个位置。
如果遇到 Pass，应记录 Pass 的输入 IR、输出 IR、启用条件和在默认流水线中的相对顺序。
如果遇到运行时模块，应记录它的创建时机、序列化格式、加载入口和资源释放位置。

### 21.99.4 逐行阅读提示与工程理解清单

1. 算子策略 的第一层理解，是把它看成 高层图到张量表达式的下降路径 中连接抽象语义和工程实现的接口。
2. 阅读 类型信息 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
21. Schedule 实例化 的第一层理解，是把它看成 高层图到张量表达式的下降路径 中连接抽象语义和工程实现的接口。
22. 阅读 PrimFunc 生成 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
41. 算子策略 的第一层理解，是把它看成 高层图到张量表达式的下降路径 中连接抽象语义和工程实现的接口。
42. 阅读 类型信息 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
61. Schedule 实例化 的第一层理解，是把它看成 高层图到张量表达式的下降路径 中连接抽象语义和工程实现的接口。
62. 阅读 PrimFunc 生成 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
81. 算子策略 的第一层理解，是把它看成 高层图到张量表达式的下降路径 中连接抽象语义和工程实现的接口。
82. 阅读 类型信息 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
101. Schedule 实例化 的第一层理解，是把它看成 高层图到张量表达式的下降路径 中连接抽象语义和工程实现的接口。
102. 阅读 PrimFunc 生成 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
121. 算子策略 的第一层理解，是把它看成 高层图到张量表达式的下降路径 中连接抽象语义和工程实现的接口。
122. 阅读 类型信息 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
141. Schedule 实例化 的第一层理解，是把它看成 高层图到张量表达式的下降路径 中连接抽象语义和工程实现的接口。
142. 阅读 PrimFunc 生成 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
161. 算子策略 的第一层理解，是把它看成 高层图到张量表达式的下降路径 中连接抽象语义和工程实现的接口。
162. 阅读 类型信息 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
181. Schedule 实例化 的第一层理解，是把它看成 高层图到张量表达式的下降路径 中连接抽象语义和工程实现的接口。
182. 阅读 PrimFunc 生成 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
201. 算子策略 的第一层理解，是把它看成 高层图到张量表达式的下降路径 中连接抽象语义和工程实现的接口。
202. 阅读 类型信息 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
221. Schedule 实例化 的第一层理解，是把它看成 高层图到张量表达式的下降路径 中连接抽象语义和工程实现的接口。
222. 阅读 PrimFunc 生成 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
241. 算子策略 的第一层理解，是把它看成 高层图到张量表达式的下降路径 中连接抽象语义和工程实现的接口。
242. 阅读 类型信息 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
261. Schedule 实例化 的第一层理解，是把它看成 高层图到张量表达式的下降路径 中连接抽象语义和工程实现的接口。
262. 阅读 PrimFunc 生成 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
281. 算子策略 的第一层理解，是把它看成 高层图到张量表达式的下降路径 中连接抽象语义和工程实现的接口。
282. 阅读 类型信息 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
301. Schedule 实例化 的第一层理解，是把它看成 高层图到张量表达式的下降路径 中连接抽象语义和工程实现的接口。
302. 阅读 PrimFunc 生成 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
321. 算子策略 的第一层理解，是把它看成 高层图到张量表达式的下降路径 中连接抽象语义和工程实现的接口。
322. 阅读 类型信息 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
341. Schedule 实例化 的第一层理解，是把它看成 高层图到张量表达式的下降路径 中连接抽象语义和工程实现的接口。
342. 阅读 PrimFunc 生成 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
361. 算子策略 的第一层理解，是把它看成 高层图到张量表达式的下降路径 中连接抽象语义和工程实现的接口。
362. 阅读 类型信息 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
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
381. Schedule 实例化 的第一层理解，是把它看成 高层图到张量表达式的下降路径 中连接抽象语义和工程实现的接口。
382. 阅读 PrimFunc 生成 时不要急于追踪每一个分支，先确认输入对象、输出对象和中间状态分别是什么。
383. 如果某个函数看起来只是转发调用，通常说明 TVM 在这里刻意保留了扩展点。
384. 这类扩展点的价值在于让新硬件、新 Pass 或新运行时能力可以在不破坏主流程的情况下接入。
385. 从调试角度看，最重要的问题不是代码在哪里执行，而是某个决策最早在哪里被记录。
386. 当日志与最终产物不一致时，应优先检查配置对象是否在中途被规范化或替换。
387. 很多初学者会把性能问题归因于单个算子，但在 TVM 中性能通常由多级决策共同决定。
388. 这些决策包括图级改写、算子策略、调度选择、代码生成、运行时调用和设备同步。
389. 源码阅读时建议把函数调用画成窄而长的链路，而不是试图一次性展开全部目录。
390. 窄链路能帮助你确定主干控制流，之后再回头阅读错误处理、缓存和注册机制。
391. 遇到模板、宏或注册表时，可以先查找最终被注册的名字，再反向定位构造逻辑。
392. 这种反向阅读方法特别适合 TVM，因为大量功能通过全局注册表暴露给 Python。
393. 工程上，稳定的边界往往比复杂的局部优化更重要。
394. 边界稳定后，TVM 才能允许不同团队分别维护前端、优化、后端和运行时。
395. 理解这一章时，可以持续追问一个问题：这个抽象减少了哪一类重复实现。
396. 如果答案是减少硬件差异，那么它通常位于 Target、DeviceAPI 或 CodeGen 附近。
397. 如果答案是减少语言差异，那么它通常位于 PackedFunc、Module 或 FFI 附近。
398. 如果答案是减少优化搜索成本，那么它通常位于调度规则、代价模型或数据库附近。

### 21.99.5 小结：把本章放回 TVM 全链路

Relay 到 TE 的学习重点不是记住每个函数名，而是理解它在 TVM 全链路中承担的边界职责。
当读者能够说清楚输入从哪里来、状态在哪里保存、输出被谁消费，就已经掌握了源码阅读的主线。
后续遇到性能、兼容性或部署问题时，可以沿着这条主线逐层排查，而不是在全仓库中盲目搜索。


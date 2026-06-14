> **学习目标**：
> - 理解 TE（Tensor Expression）的设计哲学与计算-调度分离原则
> - 掌握 te.placeholder、te.compute、reduce_axis 的定义与用法
> - 理解 TE 与 Halide 的历史渊源与设计差异
> - 掌握 TE 的核心数据结构：Operation、Tensor、Schedule、Stage
> - 了解 TE 在 TVM 编译管线中的定位与上下游关系

---

## 9.1 TE 的设计哲学

### 9.1.1 计算与调度的分离

TE（Tensor Expression）的核心设计哲学是**计算与调度的分离**（Decoupling Compute from Schedule）。这一思想源自 Halide 编译器，由 Jonathan Ragan-Kelley 等人在 2012 年提出：

> **Halide**: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines
> Jonathan Ragan-Kelley, Andrew Adams, et al. PLDI 2012

**传统方式**（计算与调度耦合）：

```c
// 在同一个函数中混合"计算什么"和"怎么执行"
void conv2d(float* input, float* weight, float* output) {
    // 计算语义
    for (int n = 0; n < N; n++)
        for (int oc = 0; oc < OC; oc++)
            for (int oh = 0; oh < OH; oh++)
                for (int ow = 0; ow < OW; ow++) {
                    float sum = 0;
                    for (int ic = 0; ic < IC; ic++)
                        for (int kh = 0; kh < KH; kh++)
                            for (int kw = 0; kw < KW; kw++)
                                sum += input[n][ic][oh+kh][ow+kw] *
                                       weight[oc][ic][kh][kw];
                    output[n][oc][oh][ow] = sum;
                }
    // 以上循环结构既包含了计算语义，又决定了执行顺序
    // 无法独立修改其中任何一个
}
```

**TE 方式**（计算与调度分离）：

```python
# Step 1: 定义计算（WHAT）
# 只描述数学语义，不指定执行顺序
A = te.placeholder((N, IC, IH, IW), name="A")
W = te.placeholder((OC, IC, KH, KW), name="W")

rh = te.reduce_axis((0, KH), name="rh")
rw = te.reduce_axis((0, KW), name="rw")
rc = te.reduce_axis((0, IC), name="rc")

B = te.compute(
    (N, OC, OH, OW),
    lambda n, oc, oh, ow: te.sum(
        A[n, rc, oh + rh, ow + rw] * W[oc, rc, rh, rw],
        axis=[rc, rh, rw]
    ),
    name="B"
)

# Step 2: 定义调度（HOW）
# 独立地指定执行策略
s = te.create_schedule(B.op)

# 分块
n, oc, oh, ow = s[B].op.axis
oho, ohi = s[B].split(oh, factor=8)
owo, owi = s[B].split(ow, factor=8)

# 重排循环顺序
s[B].reorder(n, oc, oho, owo, rc, rh, rw, ohi, owi)

# 向量化内层
s[B].vectorize(owi)

# 并行外层
s[B].parallel(n)
```

**核心洞察**：计算与调度的分离是 TE 最核心的设计哲学，它可以用一个类比来理解：就像食谱和烹饪方法是两回事。食谱告诉你"需要哪些原料、按什么比例混合"（计算定义），而烹饪方法告诉你"先大火翻炒还是先小火慢炖"（调度策略）。同一份食谱可以用不同的烹饪方法做出不同的菜——同样地，同一个计算定义可以配合不同的调度策略在不同硬件上产生高效的代码。这种分离使得算法工程师只需关注数学正确性，而性能优化工程师只需关注调度策略，两者互不干扰。

**设计权衡**：TE 选择用 Python lambda 来定义计算，这是一个深思熟虑的设计选择。lambda 函数天然表达了"对于每个输出位置，计算规则是什么"的语义，与数学公式一一对应。但这也有一个限制：lambda 中不能使用 Python 的控制流（if/else），因为 TE 需要在编译期解析 lambda 的结构。如果需要条件计算，必须使用 TE 提供的 `te.if_then_else`。这个限制确保了 TE 计算可以被编译器完全分析和优化。

### 9.1.2 分离带来的好处

| 好处 | 说明 |
|------|------|
| **可组合性** | 同一计算 + 不同调度 = 不同优化版本 |
| **可搜索性** | 搜索空间在调度层面定义，AutoTVM/MetaSchedule 搜索最优调度 |
| **可移植性** | 同一计算可在不同硬件上使用不同调度 |
| **可读性** | 计算定义简洁清晰，调度逻辑独立管理 |

```
              计算定义（What）
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
     调度 A      调度 B      调度 C
    (CPU优化)   (GPU优化)   (ARM优化)
        │          │          │
        ▼          ▼          ▼
     代码 A      代码 B      代码 C
```

**核心洞察**：这种分离带来的好处中最关键的是**可搜索性**。在传统编译器中，优化策略嵌入在代码生成器中，要调整优化需要修改编译器本身。而在 TE 中，优化策略被编码为调度原语的组合，搜索空间可以在编译器外部定义和搜索。这为自动调优（AutoTVM 和 MetaSchedule）奠定了基础——自动调优算法可以在调度空间中搜索最优的调度策略，而不需要修改编译器代码。

**实际影响**：在实际应用中，计算与调度的分离使得同一个卷积算子可以有数十种不同的调度策略。例如，一个 3×3 卷积在 CPU 上可能使用 8×8 的分块和 AVX2 向量化，在 GPU 上可能使用 128×128 的 block 分配和 shared memory 缓存。如果计算和调度耦合在一起，就需要为每种硬件编写不同的卷积实现；有了 TE，只需编写一次计算定义，然后为每种硬件编写不同的调度策略。这大大降低了硬件适配的工作量。

<div data-component="ComputeScheduleSeparation"></div>

### 9.1.3 TE 在 TVM 编译管线中的位置

```
Relay IR（图级）
    │
    │  FuseOps：融合多个算子为一个
    │  LowerRelayToTensorExpr：将融合后的算子转为 TE
    ▼
TE（算子级）
    │
    │  Schedule + Primitives：定义执行策略
    │  Lower：将 TE 转为 TIR
    ▼
TIR（循环级）
    │
    │  CodeGen：生成目标代码
    ▼
Machine Code
```

---

## 9.2 TE 的源码结构

### 9.2.1 源码目录

TE 的核心实现位于 `src/te/` 目录：

```
src/te/
├── operation/           # 计算操作定义
│   ├── compute.cc       # te.compute 实现
│   ├── placeholder.cc   # te.placeholder 实现
│   ├── scan.cc          # 循环神经网络支持
│   └── tensor.cc        # Tensor 数据结构
├── schedule/            # 调度相关
│   ├── schedule_lang.cc # 调度语言核心
│   ├── graph.cc         # 阶段图构建
│   └── message_passing.cc
├── autotvm/             # AutoTVM 集成
└── tir/                 # TIR 转换
```

### 9.2.2 头文件

| 头文件 | 说明 |
|--------|------|
| `include/tvm/te/operation.h` | Operation、ComputeOp、PlaceholderOp |
| `include/tvm/te/tensor.h` | Tensor 定义 |
| `include/tvm/te/schedule.h` | Schedule、Stage 定义 |
| `include/tvm/tir/expr.h` | TIR 表达式（Var、Add、Mul 等） |
| `include/tvm/tir/op.h` | TIR 算术操作 |

---

## 9.3 核心数据结构

### 9.3.1 Tensor：张量

`Tensor` 是 TE 中最基本的数据结构，表示一个多维数组：

```cpp
// include/tvm/te/tensor.h
class Tensor : public ObjectRef {
 public:
  /*! \brief The shape of the tensor */
  Array<PrimExpr> shape;

  /*! \brief The data type of elements */
  DataType dtype;

  /*! \brief The name of the tensor */
  String name;

  /*! \brief The operation that produces this tensor */
  Operation op;

  /*! \brief The output index (for multi-output operations) */
  int value_index;
};
```

**Tensor 与 NDArray 的区别**：

| 特性 | Tensor（TE） | NDArray（Runtime） |
|------|-------------|-------------------|
| 含义 | 符号化的张量描述 | 实际存储的数据 |
| 形状 | 可以是符号表达式 | 必须是具体整数 |
| 值 | 不包含实际数据 | 包含实际数据 |
| 用途 | 编译期定义计算 | 运行时存储数据 |

### 9.3.2 Operation：计算操作

`Operation` 是 TE 中描述计算的基类：

```cpp
// include/tvm/te/operation.h
class Operation : public ObjectRef {
 public:
  /*! \brief The name of the operation */
  String name;

  /*! \brief The output tensors */
  Array<Tensor> output_tensors;

  /*! \brief Get the number of dimensions */
  virtual int num_outputs() const = 0;

  /*! \brief Get the axis of the operation */
  virtual Array<IterVar> root_iter_vars() const = 0;
};
```

**Operation 的子类**：

```
Operation (基类)
├── PlaceholderOp    # 占位符操作（输入张量）
├── ComputeOp        # 计算操作（te.compute 定义）
├── ScanOp           # 循环操作（RNN 支持）
├── ExternOp         # 外部操作（调用手写 kernel）
└── TensorComputeOp  # Tensor Core 操作（GPU 专用）
```

### 9.3.3 IterVar：循环变量

`IterVar` 表示一个循环迭代变量：

```cpp
// include/tvm/tir/var.h
class IterVar : public ObjectRef {
 public:
  /*! \brief The variable */
  Var var;

  /*! \brief The range [min, extent) */
  Range dom;

  /*!
   * \brief The iteration type
   * kSpatial: output dimension
   * kReduce: reduction dimension
   * kOrdered: ordered iteration
   * kOpaque: opaque iteration
   */
  IterVarType iter_type;

  /*! \brief The thread tag (for GPU: threadIdx.x, blockIdx.x, etc.) */
  String thread_tag;
};
```

**IterVar 类型**：

| 类型 | 说明 | 示例 |
|------|------|------|
| `kSpatial` | 输出维度 | 矩阵乘法的 i, j |
| `kReduce` | 归约维度 | 矩阵乘法的 k |
| `kOrdered` | 有序迭代 | 序列模型的时间步 |
| `kOpaque` | 不透明迭代 | 外部函数的内部循环 |

### 9.3.4 Schedule 与 Stage

`Schedule` 是一组 `Stage` 的集合，每个 `Stage` 对应一个 `Operation`：

```cpp
// include/tvm/te/schedule.h
class Schedule : public ObjectRef {
 public:
  /*! \brief The stages in this schedule */
  Array<Stage> stages;

  /*!
   * \brief Get the stage for a given operation
   * \param op The operation
   * \return The stage
   */
  Stage operator[](const Operation& op);
};

class Stage : public ObjectRef {
 public:
  /*! \brief The operation of this stage */
  Operation op;

  /*! \brief The iteration axes */
  Array<IterVar> all_iter_vars;

  /*!
   * \brief The relations between axes (split, reorder, etc.)
   */
  Array<IterVarRelation> relations;
};
```

---

## 9.4 te.placeholder：占位符

### 9.4.1 定义与用法

`te.placeholder` 定义一个输入张量，不包含任何计算逻辑：

```python
import tvm
from tvm import te

# 定义一个形状为 (128, 256) 的 float32 占位符
A = te.placeholder((128, 256), dtype="float32", name="A")

# 形状可以是符号表达式
N = te.var("N")
M = te.var("M")
B = te.placeholder((N, M), dtype="float32", name="B")

# 访问元素
elem = A[0, 0]  # 访问 (0, 0) 位置的元素
row = A[0, :]   # 访问第 0 行（通过 compute 实现）
```

### 9.4.2 实现原理

```cpp
// src/te/operation/placeholder.cc

PlaceholderOp PlaceholderOpNode::make(String name,
                                       Array<PrimExpr> shape,
                                       DataType dtype) {
  auto n = make_object<PlaceholderOpNode>();
  n->name = name;
  n->shape = shape;
  n->dtype = dtype;

  // 创建输出张量
  n->output_tensors = {TensorNode::make(shape, dtype, n, 0)};

  // 创建迭代变量（每个维度一个 kSpatial 变量）
  for (int i = 0; i < shape.size(); i++) {
    std::string axis_name = name + ".axis" + std::to_string(i);
    n->root_iter_vars.push_back(
        IterVarNode::make(Range(0, shape[i]),
                         Var(axis_name),
                         kSpatial));
  }

  return PlaceholderOp(n);
}
```

### 9.4.3 PlaceholderOp 的属性

```python
# 查看 PlaceholderOp 的属性
A = te.placeholder((128, 256), dtype="float32", name="A")

print(A.op)           # PlaceholderOp
print(A.shape)        # [128, 256]
print(A.dtype)        # float32
print(A.name)         # "A"
print(A.op.name)      # "A"
print(A.op.root_iter_vars)  # [A.axis0, A.axis1]
```

---

## 9.5 te.compute：计算定义

### 9.5.1 基本用法

`te.compute` 定义一个逐元素的计算：

```python
import tvm
from tvm import te

# 定义输入
A = te.placeholder((128, 256), dtype="float32", name="A")
B = te.placeholder((128, 256), dtype="float32", name="B")

# 定义计算：逐元素加法
C = te.compute(
    (128, 256),                    # 输出形状
    lambda i, j: A[i, j] + B[i, j],  # 计算规则
    name="C"                        # 操作名称
)
```

### 9.5.2 Lambda 表达式的语义

`te.compute` 的第二个参数是一个 lambda 函数，定义了**每个输出元素的计算规则**：

```python
# 语义：对于输出张量的每个位置 (i, j)
# C[i, j] = A[i, j] + B[i, j]

# 等价的数学表示：
# ∀i ∈ [0, 128), ∀j ∈ [0, 256):
#   C(i, j) = A(i, j) + B(i, j)
```

**核心洞察**：`te.compute` 的设计灵感来自 Halide 的 Func 定义方式——它将计算表达为一个函数，输入是输出张量的索引，输出是该位置的值。这种函数式的定义方式有几个重要好处：首先，它自然地表达了数据并行性——每个输出位置的计算是独立的，可以安全地并行执行；其次，它将计算的"定义"和"执行"完全分离——编译器可以在不改变计算语义的情况下自由地重排、分块、向量化循环；最后，它使得自动分析变得容易——编译器可以通过解析 lambda 的 AST 来精确地知道每个输出元素依赖哪些输入元素。

**设计权衡**：TE 的 `te.compute` 使用 Python 的 AST 解析技术来提取 lambda 中的计算表达式。这意味着 lambda 必须是"纯净的"——不能有副作用、不能依赖外部状态、不能有动态控制流。这个限制确保了 TE 的计算可以被编译器完全分析和优化，但也意味着某些复杂的计算模式（如条件计算、不规则访问）需要使用特殊的 TE 操作（如 `te.if_then_else`、`te.scan`）来表达。

**lambda 参数的命名**：

```python
# lambda 的参数名可以任意取，但建议有语义
C = te.compute((M, N), lambda i, j: A[i, j] + B[i, j])  # 好：清晰
C = te.compute((M, N), lambda x, y: A[x, y] + B[x, y])  # 也可以
C = te.compute((M, N), lambda a, b: A[a, b] + B[a, b])  # 也可以
```

### 9.5.3 复杂计算示例

**矩阵乘法**：

```python
M, N, K = 128, 512, 256
A = te.placeholder((M, K), dtype="float32", name="A")
B = te.placeholder((K, N), dtype="float32", name="B")

# 定义归约轴
k = te.reduce_axis((0, K), name="k")

# 矩阵乘法：C[i, j] = sum(A[i, k] * B[k, j], k)
C = te.compute(
    (M, N),
    lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
    name="C"
)
```

**二维卷积**：

```python
N, IC, IH, IW = 1, 64, 56, 56
OC, KH, KW = 128, 3, 3
OH, OW = IH - KH + 1, IW - KW + 1

A = te.placeholder((N, IC, IH, IW), dtype="float32", name="A")
W = te.placeholder((OC, IC, KH, KW), dtype="float32", name="W")

# 归约轴
rc = te.reduce_axis((0, IC), name="rc")
rh = te.reduce_axis((0, KH), name="rh")
rw = te.reduce_axis((0, KW), name="rw")

# 卷积计算
B = te.compute(
    (N, OC, OH, OW),
    lambda n, oc, oh, ow: te.sum(
        A[n, rc, oh + rh, ow + rw] * W[oc, rc, rh, rw],
        axis=[rc, rh, rw]
    ),
    name="B"
)
```

**Batch Normalization**：

```python
C = 64
A = te.placeholder((N, C, H, W), dtype="float32", name="A")
mean = te.placeholder((C,), dtype="float32", name="mean")
var = te.placeholder((C,), dtype="float32", name="var")
gamma = te.placeholder((C,), dtype="float32", name="gamma")
beta = te.placeholder((C,), dtype="float32", name="beta")
eps = 1e-5

# BN 计算
B = te.compute(
    (N, C, H, W),
    lambda n, c, h, w: gamma[c] * (A[n, c, h, w] - mean[c]) /
                       te.sqrt(var[c] + eps) + beta[c],
    name="B"
)
```

### 9.5.4 te.compute 的实现原理

```cpp
// src/te/operation/compute.cc

ComputeOp ComputeOpNode::make(String name,
                               String tag,
                               Map<String, ObjectRef> attrs,
                               Array<IterVar> axis,
                               Array<PrimExpr> body) {
  auto n = make_object<ComputeOpNode>();
  n->name = name;
  n->tag = tag;
  n->attrs = attrs;
  n->axis = axis;

  // 分离 spatial 轴和 reduce 轴
  for (const auto& iv : axis) {
    if (iv->iter_type == kReduce) {
      n->reduce_axis.push_back(iv);
    } else {
      n->root_iter_vars.push_back(iv);
    }
  }

  n->body = body;

  // 创建输出张量
  Array<PrimExpr> shape;
  for (const auto& iv : n->root_iter_vars) {
    shape.push_back(iv->dom->extent);
  }
  for (int i = 0; i < body.size(); i++) {
    n->output_tensors.push_back(
        TensorNode::make(shape, body[i].dtype(), n, i));
  }

  return ComputeOp(n);
}
```

### 9.5.5 ComputeOp 的属性

```python
C = te.compute((M, N), lambda i, j: A[i, j] + B[i, j], name="C")

print(C.op)                    # ComputeOp
print(C.op.name)               # "C"
print(C.op.axis)               # [C.axis0, C.axis1]
print(C.op.reduce_axis)        # [] (没有归约轴)
print(C.op.body)               # [A[C.axis0, C.axis1] + B[C.axis0, C.axis1]]

# 对于有归约轴的计算
D = te.compute((M, N),
    lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
    name="D"
)
print(D.op.reduce_axis)        # [k]
```

---

## 9.6 reduce_axis：归约轴

### 9.6.1 归约轴的定义

归约轴用于定义需要**聚合**的维度：

```python
# 定义归约轴
k = te.reduce_axis((0, K), name="k")

# 使用归约轴
C = te.compute(
    (M, N),
    lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
    name="C"
)
```

### 9.6.2 归约操作

TE 支持以下内置归约操作：

| 操作 | 说明 | 示例 |
|------|------|------|
| `te.sum` | 求和 | `te.sum(expr, axis=k)` |
| `te.min` | 求最小值 | `te.min(expr, axis=k)` |
| `te.max` | 求最大值 | `te.max(expr, axis=k)` |
| `te.prod` | 求积 | `te.prod(expr, axis=k)` |

```python
# 求和：矩阵乘法
k = te.reduce_axis((0, K), name="k")
C = te.compute((M, N),
    lambda i, j: te.sum(A[i, k] * B[k, j], axis=k))

# 求最大值：池化
kh = te.reduce_axis((0, KH), name="kh")
kw = te.reduce_axis((0, KW), name="kw")
pool = te.compute((N, C, OH, OW),
    lambda n, c, oh, ow: te.max(
        A[n, c, oh*SH + kh, ow*SW + kw],
        axis=[kh, kw]
    ))

# 多轴归约
rc = te.reduce_axis((0, IC), name="rc")
rh = te.reduce_axis((0, KH), name="rh")
rw = te.reduce_axis((0, KW), name="rw")
conv = te.compute((N, OC, OH, OW),
    lambda n, oc, oh, ow: te.sum(
        A[n, rc, oh+rh, ow+rw] * W[oc, rc, rh, rw],
        axis=[rc, rh, rw]
    ))
```

### 9.6.3 归约轴的实现

```cpp
// src/te/operation/compute.cc

// 在 ComputeOpNode::make 中
for (const auto& iv : axis) {
  if (iv->iter_type == kReduce) {
    n->reduce_axis.push_back(iv);
  }
}
```

### 9.6.4 归约轴与空间轴的区别

```
空间轴 (kSpatial):
  - 定义输出张量的维度
  - 每个 (i, j) 独立计算
  - 可以并行执行

归约轴 (kReduce):
  - 定义聚合的维度
  - 多个归约迭代的结果需要合并
  - 有依赖关系，不能简单并行
```

```python
# 空间轴：i, j 是空间轴
C = te.compute((M, N), lambda i, j: A[i, j] + B[i, j])
# C 的每个元素可以独立计算

# 归约轴：k 是归约轴
C = te.compute((M, N),
    lambda i, j: te.sum(A[i, k] * B[k, j], axis=k))
# C[i, j] 需要对 k 求和
```

**核心洞察**：空间轴和归约轴的区分是理解 TE 的关键。空间轴对应于输出张量的维度——每个空间轴的组合定义了一个唯一的输出元素。归约轴则对应于"内层求和"的维度——多个归约迭代的结果需要聚合为一个值。这个区分直接影响调度策略的选择：空间轴可以自由地分块、并行化和向量化，因为它们对应的计算是独立的；而归约轴需要特殊处理——归约的中间结果需要通过原子操作或归约树来合并。理解这个区别是编写高效调度的前提。

**实际影响**：在矩阵乘法中，归约轴 k 的长度（通常为 1024 或更大）远大于空间轴 i 和 j 的分块大小（通常为 32-128）。这意味着归约轴的循环是计算的核心——优化归约轴的执行效率对整体性能至关重要。常见的优化包括：将归约轴分块以利用寄存器的累加（避免频繁的内存写入），使用向量化同时处理多个归约迭代，以及使用循环展开减少分支开销。

<div data-component="ReduceAxisVisualizer"></div>

---

## 9.7 多输出计算

### 9.7.1 多输出 te.compute

一个计算操作可以产生多个输出张量：

```python
# 多输出示例：同时计算均值和方差
A = te.placeholder((N, C), dtype="float32", name="A")

k = te.reduce_axis((0, N), name="k")

# 计算均值和方差
mean, var = te.compute(
    [(C,), (C,)],  # 两个输出的形状
    [
        lambda c: te.sum(A[k, c], axis=k) / N,  # 均值
        lambda c: te.sum((A[k, c] - mean[c]) ** 2, axis=k) / N  # 方差
    ],
    name="stats"
)
```

### 9.7.2 多输出的实现

```cpp
// src/te/operation/compute.cc

ComputeOp ComputeOpNode::make(String name,
                               String tag,
                               Map<String, ObjectRef> attrs,
                               Array<IterVar> axis,
                               Array<PrimExpr> body) {
  // body 可以有多个元素，每个对应一个输出
  for (int i = 0; i < body.size(); i++) {
    n->output_tensors.push_back(
        TensorNode::make(shape, body[i].dtype(), n, i));
  }
}
```

---

## 9.8 TE 的常用操作

### 9.8.1 逐元素操作

```python
# 加法
C = te.compute(A.shape, lambda *i: A(*i) + B(*i), name="add")

# 乘法
C = te.compute(A.shape, lambda *i: A(*i) * B(*i), name="mul")

# ReLU
C = te.compute(A.shape, lambda *i: te.max(A(*i), 0), name="relu")

# Sigmoid
C = te.compute(A.shape,
    lambda *i: 1.0 / (1.0 + te.exp(-A(*i))),
    name="sigmoid")
```

### 9.8.2 形状变换操作

```python
# Reshape
A = te.placeholder((128, 256), name="A")
B = te.compute((32, 1024),
    lambda i, j: A[i * 8 + j // 32, j % 32],
    name="reshape")

# Transpose
A = te.placeholder((M, N), name="A")
B = te.compute((N, M),
    lambda i, j: A[j, i],
    name="transpose")

# 切片
A = te.placeholder((128, 256), name="A")
B = te.compute((64, 128),
    lambda i, j: A[i + 32, j + 64],
    name="slice")
```

### 9.8.3 归约操作

```python
# 全局平均池化
A = te.placeholder((N, C, H, W), name="A")
h = te.reduce_axis((0, H), name="h")
w = te.reduce_axis((0, W), name="w")
B = te.compute((N, C),
    lambda n, c: te.sum(A[n, c, h, w], axis=[h, w]) / (H * W),
    name="global_avg_pool")

# Softmax
A = te.placeholder((N, C), name="A")
k = te.reduce_axis((0, C), name="k")
max_val = te.compute((N,),
    lambda n: te.max(A[n, k], axis=k),
    name="max_val")
exp_sum = te.compute((N,),
    lambda n: te.sum(te.exp(A[n, k] - max_val[n]), axis=k),
    name="exp_sum")
B = te.compute((N, C),
    lambda n, c: te.exp(A[n, c] - max_val[n]) / exp_sum[n],
    name="softmax")
```

---

## 9.9 TE 与 Halide 的关系

### 9.9.1 历史渊源

TVM 的 TE 设计深受 Halide 影响。Halide 是一个用于图像处理的 DSL（Domain-Specific Language），其核心创新是将算法（Algorithm）与调度（Schedule）分离。

```
Halide (2012)
    │
    │  核心思想：算法与调度分离
    │
    ▼
TVM TE (2017)
    │
    │  扩展：支持更广泛的张量计算
    │  改进：集成自动调优
    │
    ▼
TVM MetaSchedule (2022)
    │
    │  进一步：无模板的自动调度
    ▼
```

### 9.9.2 TE 与 Halide 的对比

| 特性 | Halide | TVM TE |
|------|--------|--------|
| **目标域** | 图像处理 | 深度学习 |
| **调度语言** | C++ API | Python/C++ API |
| **自动调优** | Autoschedule | AutoTVM/MetaSchedule |
| **硬件支持** | CPU/GPU | CPU/GPU/FPGA/MCU/DSP |
| **运行时** | Halide Buffer | TVM NDArray |
| **IR** | Halide IR | TIR |

### 9.9.3 TE 对 Halide 的改进

1. **更丰富的归约操作**：支持多轴归约、条件归约
2. **更好的 GPU 支持**：内置线程绑定、共享内存、Tensor Core
3. **集成自动调优**：AutoTVM 和 MetaSchedule 直接操作 TE
4. **更灵活的外部函数调用**：支持调用手写 kernel 和第三方库
5. **与 Relay 的集成**：作为 TVM 编译管线的一部分

### 9.9.4 TE 与 Halide 的代码对比

```python
# Halide 方式
x = Var("x")
y = Var("y")
f(x, y) = input(x, y) * 2
g(x, y) = f(x, y) + 1

# 调度
f.compute_root()
g.vectorize(x, 8)

# TVM TE 方式
A = te.placeholder((H, W), name="A")
B = te.compute((H, W), lambda i, j: A[i, j] * 2, name="B")
C = te.compute((H, W), lambda i, j: B[i, j] + 1, name="C")

# 调度
s = te.create_schedule(C.op)
s[B].compute_root()
s[C].vectorize(s[C].op.axis[0], 8)
```

**核心洞察**：TE 和 Halide 在代码风格上的差异反映了两者设计目标的不同。Halide 使用声明式的 C++ 风格，通过运算符重载让计算定义看起来像数学公式；TE 使用命令式的 Python 风格，通过 lambda 函数定义计算。Halide 的风格更接近数学直觉，但需要更多的 C++ 模板元编程；TE 的风格更接近编程习惯，利用 Python 的灵活性使得计算定义更加简洁。在调度原语方面，两者几乎一一对应——split/reorder/compute_at/compute_root 等概念在两个系统中都有对应的实现。

**实际影响**：TE 对 Halide 最重要的扩展是与自动调优框架的集成。Halide 的 Autoschedule 使用 Beam Search 在调度空间中搜索，而 TVM 的 AutoTVM 和 MetaSchedule 使用更先进的搜索策略（如 XGBoost 引导的搜索和进化算法）。此外，TE 的设计更加面向异构计算——它原生支持 GPU 的线程绑定、shared memory 缓存和 Tensor Core 操作，而 Halide 对 GPU 的支持相对有限。

---

## 9.10 TE 的编译流程

### 9.10.1 从 TE 到机器码

```python
import tvm
from tvm import te

# Step 1: 定义计算
A = te.placeholder((M, N), name="A")
B = te.placeholder((M, N), name="B")
C = te.compute((M, N), lambda i, j: A[i, j] + B[i, j], name="C")

# Step 2: 定义调度
s = te.create_schedule(C.op)

# Step 3: Lower（生成 TIR）
func = tvm.lower(s, [A, B, C], name="add")

# Step 4: Build（生成机器码）
target = tvm.target.Target("llvm")
lib = tvm.build(func, target=target)

# Step 5: 执行
dev = tvm.cpu(0)
a = tvm.nd.array(np.random.uniform(size=(M, N)).astype("float32"), dev)
b = tvm.nd.array(np.random.uniform(size=(M, N)).astype("float32"), dev)
c = tvm.nd.array(np.zeros((M, N), dtype="float32"), dev)

lib(a, b, c)
```

### 9.10.2 Lower 过程

`te.lower` 将 TE 调度转换为 TIR：

```python
# 查看 Lower 后的 TIR
func = tvm.lower(s, [A, B, C], name="add")
print(func)

# 输出示例：
# primfn(A: handle, B: handle, C: handle) -> ()
#   attr = {"global_symbol": "add", "tir.noalias": True}
#   buffer_C = tir.buffer_decl(...)
#   buffer_A = tir.buffer_decl(...)
#   buffer_B = tir.buffer_decl(...)
#   for i in range(0, 128):
#     for j in range(0, 256):
#       buffer_C[i*256 + j] = buffer_A[i*256 + j] + buffer_B[i*256 + j]
```

**核心洞察**：Lower 过程是 TE 系统中最关键的转换步骤——它将高层次的调度原语（split、reorder、vectorize 等）转化为低层次的循环结构。在这个过程中，TE 的符号化 IterVar 被转化为 TIR 的循环变量，Stage 之间的依赖关系被转化为显式的缓冲区声明和数据搬运循环。Lower 后的 TIR 已经非常接近最终的机器代码——它包含了完整的循环嵌套、内存访问模式和线程绑定信息。这也意味着，从 Lower 开始，计算的优化空间就已经基本确定——后续的 TIR 级优化（如循环合并、内存分配优化）只能做有限的改进。

**设计权衡**：TE 选择在 Python 层面定义计算和调度，但在 C++ 层面执行 Lower 和代码生成。这种设计利用了 Python 的易用性（快速原型开发、交互式调试）和 C++ 的性能（高效的图分析和代码生成）。Lower 过程在 C++ 中实现，确保了大规模计算图的处理效率。对于一个典型的 ResNet-50 模型，Lower 过程只需要约 50ms，而后续的 LLVM 代码生成可能需要数百毫秒。

---

## 9.11 TE 的调试工具

### 9.11.1 打印 TE 计算图

```python
# 打印 Operation
print(A.op)        # PlaceholderOp
print(C.op)        # ComputeOp

# 打印迭代变量
print(C.op.axis)         # [i, j]
print(C.op.reduce_axis)  # []

# 打印 body
print(C.op.body)   # [A[i, j] + B[i, j]]
```

### 9.11.2 可视化阶段图

```python
# 使用 tvm.build 的调试模式
func = tvm.lower(s, [A, B, C], name="add", simple_mode=True)
print(func)

# 查看调度的 IR
print(tvm.lower(s, [A, B, C]))
```

### 9.11.3 TE 表达式打印

```python
# 打印 TE 表达式
print(A[0, 0])     # A[0, 0]
print(A[i, j])     # A[i, j]
print(A[i, j] + B[i, j])  # (A[i, j] + B[i, j])

# 打印归约表达式
k = te.reduce_axis((0, K), name="k")
expr = te.sum(A[i, k] * B[k, j], axis=k)
print(expr)  # sum(A[i, k] * B[k, j], k)
```

---

## 9.12 TE 的设计模式

### 9.12.1 常用设计模式

**模式一：分块计算**

```python
# 将大计算分解为小块
def tile_compute(A, B, tile_size):
    M, N = A.shape
    # 外层循环：块级
    C_outer = te.compute(
        (M // tile_size, N // tile_size),
        lambda bi, bj: compute_block(A, B, bi, bj, tile_size),
        name="C_outer"
    )
    return C_outer
```

**模式二：两阶段计算**

```python
# 阶段 1：计算中间结果
temp = te.compute(...)
# 阶段 2：后处理
output = te.compute(..., lambda i, j: postprocess(temp[i, j]))
```

**模式三：条件计算**

```python
# 使用 te.if_then_else 实现条件计算
C = te.compute((M, N),
    lambda i, j: te.if_then_else(
        A[i, j] > 0,
        A[i, j],
        0  # ReLU
    ),
    name="relu"
)
```

### 9.12.2 反模式

**反模式一：在 lambda 中使用 Python 控制流**

```python
# 错误：Python 的 if/else 不能用于 TE
C = te.compute((M, N),
    lambda i, j: A[i, j] if A[i, j] > 0 else 0  # 错误！
)

# 正确：使用 te.if_then_else
C = te.compute((M, N),
    lambda i, j: te.if_then_else(A[i, j] > 0, A[i, j], 0)
)
```

**反模式二：在 compute 中使用非 TE 操作**

```python
# 错误：不能使用 numpy 操作
import numpy as np
C = te.compute((M, N),
    lambda i, j: np.exp(A[i, j])  # 错误！
)

# 正确：使用 te.exp
C = te.compute((M, N),
    lambda i, j: te.exp(A[i, j])
)
```

---

## 9.13 TE 的扩展机制

### 9.13.1 外部函数调用

```python
# 调用手写 C 函数
@tvm.register_func("tvm.contrib.my_custom_add")
def my_custom_add(a, b, c):
    # 调用 C 实现
    tvm.runtime.extern_lib_call("my_custom_add", a, b, c)

# 在 TE 中使用
C = te.compute((M, N),
    lambda i, j: tvm.tir.call_extern(
        "float32",
        "tvm.contrib.my_custom_add",
        A[i, j], B[i, j]
    ),
    name="C"
)
```

### 9.13.2 Tensor Compute（Tensor Core 支持）

```python
# NVIDIA Tensor Core 的矩阵乘法
# 使用 te.compute 无法直接表达，需要 TensorComputeOp
# 这将在后续章节详细讨论
```

---

## 9.14 TE 的性能特性

### 9.14.1 默认调度的性能

默认调度（无优化）通常性能较差：

```python
# 默认调度：朴素实现
s = te.create_schedule(C.op)
# 性能：~10 GFLOPS（矩阵乘法）

# 优化后的调度
s = te.create_schedule(C.op)
i, j = s[C].op.axis
k = s[C].op.reduce_axis[0]
s[C].reorder(i, k, j)  # 调整循环顺序
# 性能：~100 GFLOPS
```

### 9.14.2 TE 性能的影响因素

| 因素 | 影响 | 说明 |
|------|------|------|
| 循环顺序 | ★★★★★ | 影响缓存局部性 |
| 分块大小 | ★★★★ | 影响缓存命中率 |
| 向量化 | ★★★★ | 利用 SIMD 指令 |
| 并行化 | ★★★ | 利用多核 |
| 数据布局 | ★★★ | 影响访存模式 |

**核心洞察**：循环顺序是影响性能的最重要因素，因为它决定了数据的访问模式——进而决定了缓存命中率。以矩阵乘法为例，如果循环顺序是 i→j→k（行优先遍历 B 矩阵的列），B 矩阵的访问模式是列遍历，每次访问都会跳过整个行，导致 L1 缓存命中率极低。如果改为 i→k→j，B 矩阵的访问变成行遍历，缓存命中率大幅提升。仅仅改变循环顺序就能带来 5-10 倍的性能差异，这就是为什么 reorder 是最重要的调度原语之一。

**实际影响**：在 Intel i7-10700K CPU 上，一个 1024×1024 的矩阵乘法在不同循环顺序下的性能差异非常显著：默认顺序（i→j→k）约 10 GFLOPS，优化后（i→k→j，带 32×32 分块）约 120 GFLOPS，加速 12 倍。而理论峰值是约 400 GFLOPS（考虑 AVX2 的 8 路 float32 并行）。这意味着仅通过循环顺序和分块优化，就能达到理论峰值的 30%，加上向量化和并行化可以进一步提升到 60-80%。

<div data-component="TEPerformanceAnalyzer"></div>

---

## 9.15 实战案例

### 9.15.1 完整的矩阵乘法

```python
import tvm
from tvm import te
import numpy as np

# 参数
M, N, K = 1024, 1024, 1024

# 定义计算
A = te.placeholder((M, K), dtype="float32", name="A")
B = te.placeholder((K, N), dtype="float32", name="B")
k = te.reduce_axis((0, K), name="k")
C = te.compute(
    (M, N),
    lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
    name="C"
)

# 定义调度
s = te.create_schedule(C.op)
i, j = s[C].op.axis
ko, ki = s[C].split(k, factor=32)
s[C].reorder(i, ko, ki, j)
s[C].vectorize(j)

# 编译
func = tvm.lower(s, [A, B, C], name="matmul")
target = tvm.target.Target("llvm")
lib = tvm.build(func, target=target)

# 执行
dev = tvm.cpu(0)
a_np = np.random.uniform(size=(M, K)).astype("float32")
b_np = np.random.uniform(size=(K, N)).astype("float32")
c_np = np.zeros((M, N), dtype="float32")

a = tvm.nd.array(a_np, dev)
b = tvm.nd.array(b_np, dev)
c = tvm.nd.array(c_np, dev)

lib(a, b, c)

# 验证
np.testing.assert_allclose(c.numpy(), a_np @ b_np, rtol=1e-5)
print("Matrix multiplication verified!")
```

**核心洞察**：这个矩阵乘法示例展示了 TE 的完整工作流——从计算定义到调度优化再到编译执行。注意调度中的三个关键操作：split 将归约轴 k 分块（ko, ki），使得内层 ki 可以在寄存器中累积部分和；reorder 将循环顺序调整为 i→ko→ki→j，优化了 B 矩阵的缓存访问模式；vectorize 将最内层 j 循环向量化，利用 SIMD 指令加速。这三个操作的组合是矩阵乘法优化的基础模板。

**实际影响**：这个简单调度在 Intel i7-10700K 上可以达到约 40-60 GFLOPS，约为理论峰值的 15-20%。要达到更高的性能，还需要更复杂的优化：多级分块（tile）、寄存器缓存（register blocking）、循环展开（unroll）等。这些高级优化将在 AutoTVM 和 MetaSchedule 中自动搜索。在 GPU 上，类似的计算定义可以配合线程绑定（bind）和 shared memory 缓存（cache_read）实现数百 GFLOPS 的性能。

### 9.15.2 完整的二维卷积

```python
import tvm
from tvm import te

# 参数
N, IC, IH, IW = 1, 64, 56, 56
OC, KH, KW = 128, 3, 3
SH, SW = 1, 1
PH, PW = 1, 1
OH = (IH + 2 * PH - KH) // SH + 1
OW = (IW + 2 * PW - KW) // SW + 1

# 定义计算
A = te.placeholder((N, IC, IH, IW), dtype="float32", name="A")
W = te.placeholder((OC, IC, KH, KW), dtype="float32", name="W")

rc = te.reduce_axis((0, IC), name="rc")
rh = te.reduce_axis((0, KH), name="rh")
rw = te.reduce_axis((0, KW), name="rw")

# 带 padding 的卷积
A_pad = te.compute(
    (N, IC, IH + 2 * PH, IW + 2 * PW),
    lambda n, c, h, w: te.if_then_else(
        te.all(h >= PH, h < IH + PH, w >= PW, w < IW + PW),
        A[n, c, h - PH, w - PW],
        0.0
    ),
    name="A_pad"
)

B = te.compute(
    (N, OC, OH, OW),
    lambda n, oc, oh, ow: te.sum(
        A_pad[n, rc, oh * SH + rh, ow * SW + rw] * W[oc, rc, rh, rw],
        axis=[rc, rh, rw]
    ),
    name="B"
)

# 定义调度
s = te.create_schedule(B.op)

# 编译
func = tvm.lower(s, [A, W, B], name="conv2d")
target = tvm.target.Target("llvm")
lib = tvm.build(func, target=target)

print("Conv2D compiled successfully!")
```

---

## 9.16 本章小结

本章介绍了 TVM 的 Tensor Expression（TE）系统：

1. **设计哲学**：计算与调度分离，源自 Halide 的核心思想
2. **te.placeholder**：定义输入张量，不包含计算逻辑
3. **te.compute**：定义逐元素的计算规则，支持归约操作
4. **reduce_axis**：定义归约维度，支持 sum、min、max、prod
5. **核心数据结构**：Tensor、Operation、IterVar、Schedule、Stage
6. **与 Halide 的关系**：TE 继承了 Halide 的核心思想，但扩展了深度学习场景
7. **编译流程**：TE → TIR → 机器码

在下一章中，我们将深入学习 TE 的调度原语，这是 TVM 性能优化的核心工具。

**核心洞察**：TE 系统的核心价值在于它提供了一个清晰的抽象层次——比高级框架（如 PyTorch）更底层（可以直接控制循环结构和内存访问），但比低级 IR（如 LLVM IR）更高级（保留了张量计算的语义信息）。这个抽象层次恰好适合深度学习算子的优化——它足够底层以支持各种硬件特性的利用，又足够高级以保持代码的可读性和可维护性。TE 的设计证明了一个重要的编译器设计原则：好的抽象层次可以极大地简化优化工作。

**设计权衡**：TE 的主要限制是它只能表达"纯函数式"的计算——每个输出元素独立计算，不依赖其他输出元素。这个限制排除了前缀和（prefix sum）、递归神经网络（RNN）的序列依赖等场景。对于这些场景，TVM 提供了 `te.scan` 操作（用于 RNN 等序列依赖计算）和 TIR 层面的完整控制流支持。在实践中，TE 覆盖了深度学习推理中约 95% 的计算场景，只有极少数特殊算子需要使用更低级的接口。

**实际影响**：TE 在 TVM 生态系统中扮演着"桥梁"的角色——它连接了高层的 Relay IR（图级优化）和底层的 TIR（循环级优化和代码生成）。在 Relay 层面，FuseOps 等 Pass 负责图级优化（算子融合、常量折叠等）；在 TE 层面，调度原语负责算子级优化（分块、向量化、并行化等）；在 TIR 层面，代码生成器负责将优化后的循环结构转化为目标机器代码。这三个层次的协同工作使得 TVM 能够从模型定义到机器代码全链条地优化深度学习推理性能。

---

## 9.17 TE Compute 的数学语义

### 9.17.1 矩阵乘法的数学定义

矩阵乘法的数学表达式为：

$$
C[i,j] = \sum_{k=0}^{K-1} A[i,k] \cdot B[k,j]
$$

其中 $A \in \mathbb{R}^{M \times K}$，$B \in \mathbb{R}^{K \times N}$，$C \in \mathbb{R}^{M \times N}$。

在 TE 中，这个数学表达式的完整映射如下：

```python
import tvm
from tvm import te

# 数学定义：C[i,j] = sum_k A[i,k] * B[k,j]
# TE 中的完整表示：

M, N, K = 128, 256, 512  # 定义矩阵维度 M=128, N=256, K=512

# 定义输入矩阵 A：形状为 (M, K) 的占位符，对应数学中的 A ∈ R^{M×K}
A = te.placeholder((M, K), dtype="float32", name="A")

# 定义输入矩阵 B：形状为 (K, N) 的占位符，对应数学中的 B ∈ R^{K×N}
B = te.placeholder((K, N), dtype="float32", name="B")

# 定义归约轴 k：范围为 [0, K)，对应数学中的求和下标 k=0 到 K-1
k = te.reduce_axis((0, K), name="k")

# 定义计算 C：输出形状为 (M, N)，对应数学中的 C ∈ R^{M×N}
# lambda i, j: 表示对于输出矩阵的每个位置 (i, j)
# te.sum(A[i, k] * B[k, j], axis=k) 表示对 k 求和 A[i,k]*B[k,j]
C = te.compute(
    (M, N),                                          # 输出形状 (M, N)
    lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),  # 计算规则：矩阵乘法
    name="C"                                          # 操作名称
)

# 查看 TE 表达式的结构
print(C.op)                # 查看 ComputeOp 操作
print(C.op.axis)           # 查看空间轴：[i, j]，对应输出矩阵的行和列
print(C.op.reduce_axis)    # 查看归约轴：[k]，对应求和的维度
print(C.op.body)           # 查看计算体：sum(A[i, k] * B[k, j], k)
```

### 9.17.2 向量加法的数学定义

向量加法的数学表达式为：

$$
C[i] = A[i] + B[i], \quad i \in [0, N)
$$

```python
import tvm
from tvm import te

# 数学定义：C[i] = A[i] + B[i]
# TE 中的完整表示：

N = 1024  # 向量长度

# 定义输入向量 A：形状为 (N,) 的占位符
A = te.placeholder((N,), dtype="float32", name="A")

# 定义输入向量 B：形状为 (N,) 的占位符
B = te.placeholder((N,), dtype="float32", name="B")

# 定义计算 C：逐元素加法，没有归约轴
# lambda i: 表示对于输出向量的每个位置 i
# A[i] + B[i] 表示逐元素相加
C = te.compute(
    (N,),                          # 输出形状 (N,)
    lambda i: A[i] + B[i],         # 计算规则：逐元素加法
    name="C"                        # 操作名称
)

# 验证：没有归约轴（因为不是聚合操作）
print(C.op.reduce_axis)    # 输出：[]（空列表，没有归约轴）
print(C.op.axis)           # 输出：[C.axis0]（只有一个空间轴）
```

### 9.17.3 二维卷积的数学定义

二维卷积的数学表达式为：

$$
B[n, oc, oh, ow] = \sum_{ic=0}^{IC-1} \sum_{kh=0}^{KH-1} \sum_{kw=0}^{KW-1} A[n, ic, oh+kh, ow+kw] \cdot W[oc, ic, kh, kw]
$$

```python
import tvm
from tvm import te

# 数学定义：二维卷积
# B[n,oc,oh,ow] = sum_{ic,kh,kw} A[n,ic,oh+kh,ow+kw] * W[oc,ic,kh,kw]

N, IC, IH, IW = 1, 64, 56, 56    # 输入形状：batch=1, 通道=64, 高=56, 宽=56
OC, KH, KW = 128, 3, 3           # 卷积核：输出通道=128, 高=3, 宽=3
OH, OW = IH - KH + 1, IW - KW + 1  # 输出形状：高=54, 宽=54

# 定义输入张量 A：形状为 (N, IC, IH, IW)
A = te.placeholder((N, IC, IH, IW), dtype="float32", name="A")

# 定义卷积核 W：形状为 (OC, IC, KH, KW)
W = te.placeholder((OC, IC, KH, KW), dtype="float32", name="W")

# 定义归约轴：三个维度的求和
rc = te.reduce_axis((0, IC), name="rc")   # 输入通道维度的归约轴
rh = te.reduce_axis((0, KH), name="rh")   # 卷积核高度维度的归约轴
rw = te.reduce_axis((0, KW), name="rw")   # 卷积核宽度维度的归约轴

# 定义计算 B：输出形状为 (N, OC, OH, OW)
# 对于每个输出位置 (n, oc, oh, ow)：
#   对 ic, kh, rw 求和 A[n,ic,oh+kh,ow+rw] * W[oc,ic,kh,kw]
B = te.compute(
    (N, OC, OH, OW),                                           # 输出形状
    lambda n, oc, oh, ow: te.sum(                              # 对每个输出位置
        A[n, rc, oh + rh, ow + rw] * W[oc, rc, rh, rw],       # 卷积计算
        axis=[rc, rh, rw]                                       # 对三个轴求和
    ),
    name="B"                                                     # 操作名称
)

# 查看计算结构
print(B.op.axis)           # 空间轴：[n, oc, oh, ow]，对应输出的四个维度
print(B.op.reduce_axis)    # 归约轴：[rc, rh, rw]，对应三个求和维度
```

### 9.17.4 批量归一化的数学定义

批量归一化的数学表达式为：

$$
B[n,c,h,w] = \gamma[c] \cdot \frac{A[n,c,h,w] - \mu[c]}{\sqrt{\sigma^2[c] + \epsilon}} + \beta[c]
$$

```python
import tvm
from tvm import te

# 数学定义：Batch Normalization
# B[n,c,h,w] = gamma[c] * (A[n,c,h,w] - mean[c]) / sqrt(var[c] + eps) + beta[c]

N, C, H, W = 32, 64, 56, 56  # 输入形状
eps = 1e-5                     # 数值稳定的小常数

# 定义输入张量
A = te.placeholder((N, C, H, W), dtype="float32", name="A")      # 输入特征图
mean = te.placeholder((C,), dtype="float32", name="mean")         # 均值（每个通道）
var = te.placeholder((C,), dtype="float32", name="var")           # 方差（每个通道）
gamma = te.placeholder((C,), dtype="float32", name="gamma")       # 缩放参数
beta = te.placeholder((C,), dtype="float32", name="beta")         # 偏移参数

# 定义 BN 计算：逐元素操作，没有归约轴
# 对于每个位置 (n, c, h, w)，使用对应通道 c 的统计量
B = te.compute(
    (N, C, H, W),                                                           # 输出形状
    lambda n, c, h, w: gamma[c] * (A[n, c, h, w] - mean[c]) /              # 缩放和中心化
                       te.sqrt(var[c] + eps) + beta[c],                     # 归一化和偏移
    name="B"                                                                 # 操作名称
)

# 验证：BN 是逐元素操作，没有归约轴
print(B.op.reduce_axis)    # 输出：[]（空列表）
```

---

## 9.18 Stage 类的完整接口

### 9.18.1 Stage 类概述

`Stage` 是 TE 调度系统的核心类，每个 `Stage` 对应一个 `Operation`，并提供调度原语接口：

```python
import tvm
from tvm import te

# 创建计算
A = te.placeholder((128, 256), dtype="float32", name="A")
B = te.placeholder((128, 256), dtype="float32", name="B")
C = te.compute((128, 256), lambda i, j: A[i, j] + B[i, j], name="C")

# 创建调度
s = te.create_schedule(C.op)

# 获取 Stage 对象
stage_C = s[C]  # 等价于 s[C.op]

# 查看 Stage 的基本属性
print(stage_C.op)           # ComputeOp：该 Stage 对应的操作
print(stage_C.op.axis)      # [C.axis0, C.axis1]：空间迭代轴
print(stage_C.op.reduce_axis)  # []：归约轴（本例无归约）
```

### 9.18.2 axis 属性详解

`axis` 包含了 Operation 的所有空间迭代变量：

```python
import tvm
from tvm import te

# 示例：矩阵乘法的 axis
M, N, K = 64, 64, 64
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
k = te.reduce_axis((0, K), name="k")
C = te.compute(
    (M, N),
    lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
    name="C"
)

s = te.create_schedule(C.op)

# C.op.axis 只包含空间轴（输出维度），不包含归约轴
i, j = s[C].op.axis
print(f"空间轴 i: {i}")     # C.axis0：对应输出矩阵的行
print(f"空间轴 j: {j}")     # C.axis1：对应输出矩阵的列

# C.op.reduce_axis 包含归约轴
k_axis = s[C].op.reduce_axis[0]
print(f"归约轴 k: {k_axis}")  # k：对应求和的维度

# 遍历所有迭代变量
print(f"所有迭代变量: {s[C].op.axis}")         # 只有空间轴
print(f"所有归约变量: {s[C].op.reduce_axis}")   # 只有归约轴
```

### 9.18.3 reduce_axis 属性详解

`reduce_axis` 包含所有归约维度的迭代变量：

```python
import tvm
from tvm import te

# 示例：二维卷积的归约轴
N, IC, IH, IW = 1, 32, 16, 16
OC, KH, KW = 64, 3, 3
OH, OW = IH - KH + 1, IW - KW + 1

A = te.placeholder((N, IC, IH, IW), name="A")
W = te.placeholder((OC, IC, KH, KW), name="W")

# 定义三个归约轴
rc = te.reduce_axis((0, IC), name="rc")   # 输入通道归约
rh = te.reduce_axis((0, KH), name="rh")   # 卷积核高度归约
rw = te.reduce_axis((0, KW), name="rw")   # 卷积核宽度归约

B = te.compute(
    (N, OC, OH, OW),
    lambda n, oc, oh, ow: te.sum(
        A[n, rc, oh + rh, ow + rw] * W[oc, rc, rh, rw],
        axis=[rc, rh, rw]
    ),
    name="B"
)

s = te.create_schedule(B.op)

# 空间轴：4 个（对应输出的 N, OC, OH, OW）
print(f"空间轴数量: {len(s[B].op.axis)}")         # 4
print(f"空间轴: {s[B].op.axis}")                    # [n, oc, oh, ow]

# 归约轴：3 个（对应求和的 IC, KH, KW）
print(f"归约轴数量: {len(s[B].op.reduce_axis)}")   # 3
print(f"归约轴: {s[B].op.reduce_axis}")              # [rc, rh, rw]

# 查看每个归约轴的范围
for ra in s[B].op.reduce_axis:
    print(f"归约轴 {ra.var.name}: 范围 [{ra.dom.min}, {ra.dom.min + ra.dom.extent})")
```

### 9.18.4 op 属性详解

`op` 属性返回 Stage 对应的 Operation 对象：

```python
import tvm
from tvm import te

A = te.placeholder((64, 64), name="A")
B = te.compute((64, 64), lambda i, j: A[i, j] * 2, name="B")

s = te.create_schedule(B.op)

# op 属性
op = s[B].op
print(type(op))              # <class 'tvm.te.operation.ComputeOp'>
print(op.name)               # "B"
print(op.axis)               # [B.axis0, B.axis1]
print(op.reduce_axis)        # []（无归约轴）
print(op.body)               # [A[B.axis0, B.axis1] * 2]

# 对于 PlaceholderOp
op_A = A.op
print(type(op_A))            # <class 'tvm.te.operation.PlaceholderOp'>
print(op_A.name)             # "A"
```

### 9.18.5 Stage 的调度原语接口

Stage 提供了丰富的调度原语：

```python
import tvm
from tvm import te

M, N, K = 256, 256, 256
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
k = te.reduce_axis((0, K), name="k")
C = te.compute(
    (M, N),
    lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
    name="C"
)

s = te.create_schedule(C.op)
i, j = s[C].op.axis
ko, ki = s[C].split(k, factor=32)   # 分裂归约轴

# 1. split：将一个轴分裂为两个
io, ii = s[C].split(i, factor=16)   # 将 i 分裂为 io（外层）和 ii（内层）
jo, ji = s[C].split(j, factor=16)   # 将 j 分裂为 jo（外层）和 ji（内层）

# 2. reorder：重新排列循环顺序
s[C].reorder(io, jo, ko, ki, ii, ji)

# 3. vectorize：向量化最内层循环
s[C].vectorize(ji)

# 4. parallel：并行化外层循环
s[C].parallel(io)

# 5. unroll：展开循环（编译期展开）
s[C].unroll(ki)

# 6. bind：将轴绑定到硬件线程（GPU）
# s[C].bind(io, te.thread_axis("blockIdx.x"))
# s[C].bind(jo, te.thread_axis("blockIdx.y"))

# 7. compute_at：将一个 Stage 嵌入到另一个 Stage 的循环中
# s[intermediate].compute_at(s[C], ko)

# 8. compute_root：将 Stage 提升到顶层独立计算
# s[intermediate].compute_root()

# 9. storage_align：设置存储对齐
s[C].storage_align(i, factor=128, offset=0)

# 10. double_buffer：启用双缓冲
# s[C].double_buffer()

# 查看最终的 TIR
func = tvm.lower(s, [A, B, C], name="matmul")
print(func)
```

### 9.18.6 Stage 的 C++ 接口

```cpp
// include/tvm/te/schedule.h
class StageNode : public Object {
 public:
  // 操作
  Operation op;

  // 所有迭代变量（包括分裂产生的新变量）
  Array<IterVar> all_iter_vars;

  // 迭代变量之间的关系（split、reorder 等）
  Array<IterVarRelation> relations;

  // 调度原语接口
  Stage& split(IterVar parent, PrimExpr factor, IterVar* p_outer, IterVar* p_inner);
  Stage& reorder(Array<IterVar> order);
  Stage& vectorize(IterVar var);
  Stage& parallel(IterVar var);
  Stage& unroll(IterVar var);
  Stage& bind(IterVar var, IterVar thread_axis);
  Stage& compute_at(Stage parent, IterVar scope);
  Stage& compute_root();
  Stage& storage_align(IterVar axis, int factor, int offset);
  Stage& double_buffer();
};
```

---

## 9.19 TE 与 Halide 的详细对比

### 9.19.1 设计目标对比

| 维度 | Halide | TVM TE |
|------|--------|--------|
| **设计初衷** | 图像处理管线优化 | 深度学习算子优化 |
| **目标用户** | 图像处理工程师 | 深度学习编译器开发者 |
| **抽象层次** | 图像处理原语 | 通用张量计算 |
| **自动调优** | Autoschedule（基于搜索） | AutoTVM/MetaSchedule（基于搜索+学习） |
| **硬件覆盖** | CPU、GPU（主要 x86/ARM） | CPU、GPU、FPGA、MCU、DSP、专用加速器 |
| **运行时** | Halide Buffer（C++） | TVM NDArray（多语言绑定） |
| **IR** | Halide IR（自定义） | TIR（基于统一 IR 框架） |
| **生态集成** | 独立工具链 | 与 Relay/ONNX/TensorFlow 集成 |

### 9.19.2 调度原语对比

```python
# ===== Halide 调度原语（C++ API） =====
# Halide::Func f;
# f(x, y) = input(x, y) * 2;
# f.vectorize(x, 8);            // 向量化 x 维度，宽度为 8
# f.parallel(y);                // 并行化 y 维度
# f.split(x, xo, xi, 32);      // 分裂 x 为 xo 和 xi
# f.reorder(xi, xo, y);        // 重排循环顺序
# f.tile(x, y, xi, yi, 32, 32); // 二维分块
# f.compute_at(g, y);          // 在 g 的 y 循环中计算 f
# f.compute_root();            // 在顶层独立计算 f
# f.store_at(g, x);            // 在 g 的 x 循环层级分配存储
# f.unroll(x);                 // 展开循环

# ===== TVM TE 调度原语（Python API） =====
# s = te.create_schedule(C.op)
# s[C].vectorize(ji)             # 向量化 ji 轴
# s[C].parallel(io)              # 并行化 io 轴
# io, ii = s[C].split(i, 32)    # 分裂 i 为 io 和 ii
# s[C].reorder(ii, io, j)       # 重排循环顺序
# io, ii = s[C].split(i, 32)   # 二维分块（手动）
# jo, ji = s[C].split(j, 32)
# s[f].compute_at(s[C], j)     # 在 C 的 j 循环中计算 f
# s[f].compute_root()           # 在顶层独立计算 f
# s[C].unroll(ki)               # 展开循环
```

### 9.19.3 硬件支持对比

```python
# ===== Halide 的硬件支持 =====
# Halide 主要支持 CPU 和 GPU 的基本调度：
# - CPU：向量化（SSE/AVX/NEON）、并行化（OpenMP）、分块
# - GPU：CUDA/OpenCL 的线程绑定和共享内存
# - 不支持：Tensor Core、DSP 指令、FPGA 流水线

# ===== TVM TE 的硬件支持 =====
# TVM TE 支持更广泛的硬件特性：

# 1. GPU 线程绑定
# s[C].bind(i, te.thread_axis("blockIdx.x"))      # 绑定到线程块
# s[C].bind(j, te.thread_axis("threadIdx.x"))     # 绑定到线程

# 2. 共享内存
# A_shared = s.cache_read(A, "shared", [C])       # 使用共享内存缓存
# s[A_shared].compute_at(s[C], k_outer)            # 在循环中加载

# 3. Tensor Core（NVIDIA）
# 通过 TensorComputeOp 支持矩阵乘法加速指令

# 4. 向量化宽度自适应
# s[C].vectorize(ji)  # 自动匹配目标硬件的向量宽度

# 5. 特定硬件的调度模板
# from tvm import auto_scheduler
# @auto_scheduler.register_workload
# def matmul workload(N, M, K):
#     ...
```

### 9.19.4 代码风格对比

```python
# ===== Halide：声明式（函数式） =====
# Halide::Var x("x"), y("y");
# Halide::Func gradient("gradient");
# gradient(x, y) = x + y;                     // 定义计算
# gradient.vectorize(x, 8);                    // 定义调度
# gradient.compile_to_file("gradient", {});    // 编译

# ===== TVM TE：命令式（过程式） =====
import tvm
from tvm import te

x = te.var("x")                                # 定义变量
y = te.var("y")                                # 定义变量
A = te.placeholder((128, 256), name="A")       # 定义输入
B = te.compute((128, 256),                      # 定义计算
    lambda i, j: A[i, j] + i + j,
    name="gradient")

s = te.create_schedule(B.op)                   # 创建调度
s[B].vectorize(s[B].op.axis[0])                # 向量化

func = tvm.lower(s, [A, B], name="gradient")  # Lower 到 TIR
target = tvm.target.Target("llvm")             # 定义目标
lib = tvm.build(func, target=target)           # 编译
```

### 9.19.5 自动调优对比

```python
# ===== Halide 的 Autoschedule =====
# Halide::AutoScheduler autoscheduler;
# autoscheduler.auto_schedule(func, target);
# // 基于 Beam Search 的模板生成
# // 不支持学习型搜索

# ===== TVM 的 AutoTVM =====
# from tvm import autotvm
#
# @autotvm.template("matmul")
# def matmul_template(N, M, K):
#     A = te.placeholder((N, K), name="A")
#     B = te.placeholder((K, M), name="B")
#     k = te.reduce_axis((0, K), name="k")
#     C = te.compute((N, M),
#         lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
#         name="C")
#
#     s = te.create_schedule(C.op)
#     # 定义搜索空间（通过 ConfigEntity）
#     cfg = autotvm.get_config()
#     cfg.define_knob("tile_i", [8, 16, 32, 64])
#     cfg.define_knob("tile_j", [8, 16, 32, 64])
#     return s, [A, B, C]

# ===== TVM 的 MetaSchedule =====
# from tvm import meta_schedule
#
# # 无需手动定义搜索空间
# # MetaSchedule 自动生成调度原语的搜索空间
# database = meta_schedule.tune_tir(
#     mod=func,
#     target=target,
#     max_trials_global=1000,
# )
```

### 9.19.6 IR 对比

```python
# ===== Halide IR 示例 =====
# Halide IR 是一种低级表示：
# produce gradient {
#   for (y, 0, 128) {
#     for (x, 0, 256) {
#       gradient(x, y) = (x + y)
#     }
#   }
# }

# ===== TVM TIR 示例 =====
# TIR 是 TVM 的低级中间表示：
# @T.prim_func
# def gradient(A: T.Buffer[(128, 256), "int32"],
#              B: T.Buffer[(128, 256), "int32"]):
#     for i in T.serial(128):
#         for j in T.serial(256):
#             with T.block("B"):
#                 vi = T.axis.spatial(128, i)
#                 vj = T.axis.spatial(256, j)
#                 B[vi, vj] = A[vi, vj] + vi + vj
```

---

## 9.20 完整的矩阵乘法 TE 示例

### 9.20.1 从定义到调度到编译

以下是一个完整的矩阵乘法示例，从 TE 计算定义、调度优化、到编译执行的全过程：

```python
import tvm
from tvm import te
import numpy as np

# ============================================================
# Step 1: 定义参数
# ============================================================
M = 1024  # 矩阵 A 的行数，矩阵 C 的行数
N = 1024  # 矩阵 B 的列数，矩阵 C 的列数
K = 1024  # 矩阵 A 的列数，矩阵 B 的行数

# ============================================================
# Step 2: 定义计算（WHAT）
# ============================================================
# 定义输入矩阵 A：形状为 (M, K)，数据类型为 float32
A = te.placeholder((M, K), dtype="float32", name="A")

# 定义输入矩阵 B：形状为 (K, N)，数据类型为 float32
B = te.placeholder((K, N), dtype="float32", name="B")

# 定义归约轴 k：范围为 [0, K)，用于求和
k = te.reduce_axis((0, K), name="k")

# 定义矩阵乘法：C[i, j] = sum(A[i, k] * B[k, j], k)
C = te.compute(
    (M, N),                                          # 输出形状
    lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),  # 计算规则
    name="C"                                          # 操作名称
)

# ============================================================
# Step 3: 定义调度（HOW）
# ============================================================
# 创建调度对象
s = te.create_schedule(C.op)

# 获取空间轴和归约轴
i, j = s[C].op.axis        # 空间轴：i（行），j（列）
ko, ki = s[C].split(k, factor=32)  # 分裂归约轴：ko（外层，步长32），ki（内层，宽度32）

# 分块：将 i 和 j 分裂为外层和内层
io, ii = s[C].split(i, factor=64)  # i 分裂为 io（外层）和 ii（内层，块大小64）
jo, ji = s[C].split(j, factor=64)  # j 分裂为 jo（外层）和 ji（内层，块大小64）

# 重排循环顺序：外层 -> 归约 -> 内层
# 优化缓存局部性：先遍历块，再遍历归约，最后遍历块内元素
s[C].reorder(io, jo, ko, ki, ii, ji)

# 向量化最内层循环：利用 SIMD 指令加速
s[C].vectorize(ji)

# 并行化外层循环：利用多核 CPU
s[C].parallel(io)

# ============================================================
# Step 4: Lower（生成 TIR）
# ============================================================
# 将 TE 调度转换为 TIR 中间表示
func = tvm.lower(s, [A, B, C], name="matmul")
print("===== TIR =====")
print(func)

# ============================================================
# Step 5: Build（生成机器码）
# ============================================================
# 定义目标平台
target = tvm.target.Target("llvm")

# 编译生成共享库
lib = tvm.build(func, target=target, name="matmul")
print("===== 编译完成 =====")

# ============================================================
# Step 6: 执行与验证
# ============================================================
# 创建执行设备
dev = tvm.cpu(0)

# 创建输入数据（NumPy 数组）
a_np = np.random.uniform(size=(M, K)).astype("float32")  # 随机初始化 A
b_np = np.random.uniform(size=(K, N)).astype("float32")  # 随机初始化 B
c_np = np.zeros((M, N), dtype="float32")                  # 初始化 C 为零

# 将数据转换为 TVM NDArray
a = tvm.nd.array(a_np, dev)   # A 的 TVM 数组
b = tvm.nd.array(b_np, dev)   # B 的 TVM 数组
c = tvm.nd.array(c_np, dev)   # C 的 TVM 数组

# 执行矩阵乘法
lib(a, b, c)

# 验证结果正确性
np.testing.assert_allclose(c.numpy(), a_np @ b_np, rtol=1e-5)
print("矩阵乘法验证通过！")
```

### 9.20.2 性能基准测试

```python
import tvm
from tvm import te
import numpy as np
import time

def benchmark_matmul(M, N, K, target_str="llvm"):
    """矩阵乘法性能基准测试"""

    # 定义计算
    A = te.placeholder((M, K), dtype="float32", name="A")
    B = te.placeholder((K, N), dtype="float32", name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="C"
    )

    # 定义调度
    s = te.create_schedule(C.op)
    i, j = s[C].op.axis
    ko, ki = s[C].split(k, factor=32)
    io, ii = s[C].split(i, factor=64)
    jo, ji = s[C].split(j, factor=64)
    s[C].reorder(io, jo, ko, ki, ii, ji)
    s[C].vectorize(ji)
    s[C].parallel(io)

    # 编译
    func = tvm.lower(s, [A, B, C], name="matmul")
    target = tvm.target.Target(target_str)
    lib = tvm.build(func, target=target)

    # 准备数据
    dev = tvm.cpu(0)
    a = tvm.nd.array(np.random.uniform(size=(M, K)).astype("float32"), dev)
    b = tvm.nd.array(np.random.uniform(size=(K, N)).astype("float32"), dev)
    c = tvm.nd.array(np.zeros((M, N), dtype="float32"), dev)

    # 预热
    for _ in range(3):
        lib(a, b, c)

    # 计时
    start = time.time()
    num_runs = 10
    for _ in range(num_runs):
        lib(a, b, c)
    elapsed = (time.time() - start) / num_runs

    # 计算性能
    flops = 2 * M * N * K  # 乘法和加法各 M*N*K 次
    gflops = flops / elapsed / 1e9

    print(f"矩阵大小: {M}x{N}x{K}")
    print(f"执行时间: {elapsed*1000:.2f} ms")
    print(f"性能: {gflops:.2f} GFLOPS")

    return elapsed

# 运行基准测试
benchmark_matmul(1024, 1024, 1024)
```

---

## 9.21 完整的卷积 TE 示例（含 padding/stride）

### 9.21.1 带 padding 的卷积

```python
import tvm
from tvm import te
import numpy as np

# ============================================================
# 参数定义
# ============================================================
N = 1          # batch size
IC = 64        # 输入通道数
IH = 56        # 输入高度
IW = 56        # 输入宽度
OC = 128       # 输出通道数
KH = 3         # 卷积核高度
KW = 3         # 卷积核宽度
SH = 1         # 高度方向步长
SW = 1         # 宽度方向步长
PH = 1         # 高度方向 padding
PW = 1         # 宽度方向 padding

# 计算输出尺寸：OH = (IH + 2*PH - KH) / SH + 1
OH = (IH + 2 * PH - KH) // SH + 1  # 输出高度 = 56
OW = (IW + 2 * PW - KW) // SW + 1  # 输出宽度 = 56

# ============================================================
# Step 1: 定义输入张量
# ============================================================
# 输入特征图：形状 (N, IC, IH, IW)
A = te.placeholder((N, IC, IH, IW), dtype="float32", name="A")

# 卷积核：形状 (OC, IC, KH, KW)
W = te.placeholder((OC, IC, KH, KW), dtype="float32", name="W")

# ============================================================
# Step 2: 定义 Padding 操作
# ============================================================
# 创建带 padding 的输入张量
# 形状从 (N, IC, IH, IW) 扩展为 (N, IC, IH+2*PH, IW+2*PW)
A_pad = te.compute(
    (N, IC, IH + 2 * PH, IW + 2 * PW),                                      # 输出形状
    lambda n, c, h, w: te.if_then_else(                                       # 条件判断
        te.all(h >= PH, h < IH + PH, w >= PW, w < IW + PW),                  # 在原始范围内
        A[n, c, h - PH, w - PW],                                              # 读取原始数据
        0.0                                                                    # padding 区域填零
    ),
    name="A_pad"                                                               # 操作名称
)

# ============================================================
# Step 3: 定义归约轴
# ============================================================
rc = te.reduce_axis((0, IC), name="rc")   # 输入通道维度归约
rh = te.reduce_axis((0, KH), name="rh")   # 卷积核高度维度归约
rw = te.reduce_axis((0, KW), name="rw")   # 卷积核宽度维度归约

# ============================================================
# Step 4: 定义卷积计算
# ============================================================
# B[n, oc, oh, ow] = sum_{rc, rh, rw} A_pad[n, rc, oh*SH+rh, ow*SW+rw] * W[oc, rc, rh, rw]
B = te.compute(
    (N, OC, OH, OW),                                                           # 输出形状
    lambda n, oc, oh, ow: te.sum(                                              # 对每个输出位置
        A_pad[n, rc, oh * SH + rh, ow * SW + rw] * W[oc, rc, rh, rw],         # 卷积计算
        axis=[rc, rh, rw]                                                       # 对三个轴求和
    ),
    name="B"                                                                     # 操作名称
)

# ============================================================
# Step 5: 定义调度
# ============================================================
s = te.create_schedule(B.op)

# 查看生成的 TIR
func = tvm.lower(s, [A, W, B], name="conv2d")
print("===== Conv2D TIR =====")
print(func)

# ============================================================
# Step 6: 编译与执行
# ============================================================
target = tvm.target.Target("llvm")
lib = tvm.build(func, target=target, name="conv2d")

dev = tvm.cpu(0)
a_np = np.random.uniform(size=(N, IC, IH, IW)).astype("float32")
w_np = np.random.uniform(size=(OC, IC, KH, KW)).astype("float32")
b_np = np.zeros((N, OC, OH, OW), dtype="float32")

a = tvm.nd.array(a_np, dev)
w = tvm.nd.array(w_np, dev)
b = tvm.nd.array(b_np, dev)

lib(a, w, b)
print("卷积编译执行成功！")
```

### 9.21.2 带 stride 的卷积

```python
import tvm
from tvm import te
import numpy as np

# ============================================================
# 参数定义（步长为 2 的卷积）
# ============================================================
N, IC, IH, IW = 1, 64, 56, 56   # 输入形状
OC, KH, KW = 128, 3, 3          # 卷积核参数
SH, SW = 2, 2                    # 步长为 2（下采样）
PH, PW = 1, 1                    # padding 为 1

# 输出尺寸：(56 + 2*1 - 3) / 2 + 1 = 28
OH = (IH + 2 * PH - KH) // SH + 1   # 28
OW = (IW + 2 * PW - KW) // SW + 1   # 28

# ============================================================
# 定义计算
# ============================================================
# 输入张量
A = te.placeholder((N, IC, IH, IW), dtype="float32", name="A")
# 卷积核
W = te.placeholder((OC, IC, KH, KW), dtype="float32", name="W")

# Padding
A_pad = te.compute(
    (N, IC, IH + 2 * PH, IW + 2 * PW),
    lambda n, c, h, w: te.if_then_else(
        te.all(h >= PH, h < IH + PH, w >= PW, w < IW + PW),
        A[n, c, h - PH, w - PW],
        0.0
    ),
    name="A_pad"
)

# 归约轴
rc = te.reduce_axis((0, IC), name="rc")
rh = te.reduce_axis((0, KH), name="rh")
rw = te.reduce_axis((0, KW), name="rw")

# 卷积计算（注意步长 SH, SW）
# B[n, oc, oh, ow] = sum A_pad[n, rc, oh*SH+rh, ow*SW+rw] * W[oc, rc, rh, rw]
B = te.compute(
    (N, OC, OH, OW),
    lambda n, oc, oh, ow: te.sum(
        A_pad[n, rc, oh * SH + rh, ow * SW + rw] * W[oc, rc, rh, rw],
        axis=[rc, rh, rw]
    ),
    name="B"
)

# ============================================================
# 定义调度
# ============================================================
s = te.create_schedule(B.op)

# Lower 并编译
func = tvm.lower(s, [A, W, B], name="conv2d_stride2")
target = tvm.target.Target("llvm")
lib = tvm.build(func, target=target, name="conv2d_stride2")

print("步长为 2 的卷积编译成功！")
print(f"输入形状: ({N}, {IC}, {IH}, {IW})")
print(f"输出形状: ({N}, {OC}, {OH}, {OW})")
```

### 9.21.3 优化的卷积调度

```python
import tvm
from tvm import te

# ============================================================
# 完整的优化卷积调度
# ============================================================
N, IC, IH, IW = 1, 64, 56, 56
OC, KH, KW = 128, 3, 3
SH, SW, PH, PW = 1, 1, 1, 1
OH = (IH + 2 * PH - KH) // SH + 1
OW = (IW + 2 * PW - KW) // SW + 1

A = te.placeholder((N, IC, IH, IW), dtype="float32", name="A")
W = te.placeholder((OC, IC, KH, KW), dtype="float32", name="W")

# Padding
A_pad = te.compute(
    (N, IC, IH + 2 * PH, IW + 2 * PW),
    lambda n, c, h, w: te.if_then_else(
        te.all(h >= PH, h < IH + PH, w >= PW, w < IW + PW),
        A[n, c, h - PH, w - PW],
        0.0
    ),
    name="A_pad"
)

rc = te.reduce_axis((0, IC), name="rc")
rh = te.reduce_axis((0, KH), name="rh")
rw = te.reduce_axis((0, KW), name="rw")

B = te.compute(
    (N, OC, OH, OW),
    lambda n, oc, oh, ow: te.sum(
        A_pad[n, rc, oh * SH + rh, ow * SW + rw] * W[oc, rc, rh, rw],
        axis=[rc, rh, rw]
    ),
    name="B"
)

# ============================================================
# 优化调度
# ============================================================
s = te.create_schedule(B.op)

# 获取轴
n, oc, oh, ow = s[B].op.axis

# 分块输出空间
oho, ohi = s[B].split(oh, factor=8)    # 高度方向分块
owo, owi = s[B].split(ow, factor=8)    # 宽度方向分块

# 分裂归约轴
rco, rci = s[B].split(rc, factor=16)   # 通道归约分块

# 重排循环顺序
s[B].reorder(n, oc, oho, owo, rco, rh, rw, rci, ohi, owi)

# 向量化最内层
s[B].vectorize(owi)

# 并行化外层
s[B].parallel(oho)

# Padding 计算嵌入到卷积循环中
s[A_pad].compute_at(s[B], rco)

# 查看优化后的 TIR
func = tvm.lower(s, [A, W, B], name="conv2d_opt")
print("===== 优化后的 Conv2D TIR =====")
print(func)
```

---

## 9.22 TE 的限制

### 9.22.1 不支持控制流

TE 的 `te.compute` 只能表达**纯函数式**的计算（每个输出元素独立计算），不支持控制流（如 if-else 分支、while 循环）：

```python
import tvm
from tvm import te

# ===== 错误示例：Python 控制流 =====
# 这段代码会出错，因为 Python 的 if/else 在 lambda 中不能用于 TE
A = te.placeholder((64,), name="A")

# 错误！Python 的 if-else 不能用于 TE
# B = te.compute((64,), lambda i: A[i] if A[i] > 0 else 0)

# ===== 正确方式：使用 te.if_then_else =====
# TE 提供了 te.if_then_else 用于条件计算
B = te.compute(
    (64,),
    lambda i: te.if_then_else(A[i] > 0, A[i], 0.0),  # ReLU 激活函数
    name="B"
)

# ===== 更复杂的条件：嵌套 if =====
# 多层条件使用 te.if_then_else 嵌套
C = te.compute(
    (64,),
    lambda i: te.if_then_else(
        A[i] > 0,            # 条件 1
        A[i],                 # 真值分支
        te.if_then_else(
            A[i] < -1,        # 条件 2
            -1.0,              # 真值分支
            A[i] * 0.1         # 假值分支（Leaky ReLU）
        )
    ),
    name="C"
)
```

### 9.22.2 不支持动态控制流

TE 不支持运行时的动态控制流（如 while 循环、递归）：

```python
# ===== TE 不支持的模式 =====

# 1. While 循环：TE 无法表达迭代次数未知的循环
# while condition:    # 不支持
#     compute()

# 2. 递归：TE 无法表达递归计算
# def recursive_compute(n):    # 不支持
#     if n == 0: return base
#     return recursive_compute(n-1)

# 3. 动态索引：TE 的索引必须是仿射表达式
# A[some_function(i)]  # 如果 some_function 不是仿射的，可能无法优化

# ===== 替代方案 =====
# 对于需要复杂控制流的场景，可以使用：
# 1. TIR（更低级，支持完整的控制流）
# 2. ExternOp（调用手写 C/CUDA 代码）
# 3. te.scan（用于 RNN 等有序列依赖的计算）
```

### 9.22.3 动态形状的处理方式

TE 原生支持符号化形状（动态形状），但有一些限制：

```python
import tvm
from tvm import te

# ===== 符号化形状 =====
# 使用 te.var 定义符号变量
M = te.var("M")   # 符号变量，运行时确定
N = te.var("N")   # 符号变量，运行时确定
K = te.var("K")   # 符号变量，运行时确定

# 使用符号变量定义形状
A = te.placeholder((M, K), dtype="float32", name="A")
B = te.placeholder((K, N), dtype="float32", name="B")
k = te.reduce_axis((0, K), name="k")
C = te.compute(
    (M, N),
    lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
    name="C"
)

# 创建调度
s = te.create_schedule(C.op)

# 分裂操作在符号形状下需要特殊处理
# 不能直接使用 factor=32，因为 M 可能不是 32 的倍数
# TE 会自动处理尾部迭代
i, j = s[C].op.axis
io, ii = s[C].split(i, factor=32)  # TE 自动处理余数

# ===== 动态形状的编译 =====
# 编译时 M, N, K 仍然是符号
func = tvm.lower(s, [A, B, C], name="matmul_dynamic")
print("===== 动态形状 TIR =====")
print(func)

# 执行时通过传入不同大小的 NDArray 来确定形状
target = tvm.target.Target("llvm")
lib = tvm.build(func, target=target, name="matmul_dynamic")

import numpy as np
dev = tvm.cpu(0)

# 第一次执行：M=64, N=128, K=256
a1 = tvm.nd.array(np.random.uniform(size=(64, 256)).astype("float32"), dev)
b1 = tvm.nd.array(np.random.uniform(size=(256, 128)).astype("float32"), dev)
c1 = tvm.nd.array(np.zeros((64, 128), dtype="float32"), dev)
lib(a1, b1, c1)

# 第二次执行：M=100, N=200, K=300
a2 = tvm.nd.array(np.random.uniform(size=(100, 300)).astype("float32"), dev)
b2 = tvm.nd.array(np.random.uniform(size=(300, 200)).astype("float32"), dev)
c2 = tvm.nd.array(np.zeros((100, 200), dtype="float32"), dev)
lib(a2, b2, c2)
```

### 9.22.4 动态形状的调度限制

```python
import tvm
from tvm import te

# ===== 动态形状下的调度限制 =====

M = te.var("M")
N = te.var("N")
A = te.placeholder((M, N), name="A")
B = te.compute((M, N), lambda i, j: A[i, j] * 2, name="B")

s = te.create_schedule(B.op)
i, j = s[B].op.axis

# 限制 1：split 的 factor 必须是常数
# 不能使用符号变量作为 factor
# io, ii = s[B].split(i, factor=M)   # 错误！factor 必须是常量
io, ii = s[B].split(i, factor=32)     # 正确：使用常量

# 限制 2：向量化要求最内层循环的范围是向量宽度的倍数
# 动态形状下，最内层循环的范围可能不是常数
# s[B].vectorize(ii)  # 可能会失败，因为 ii 的范围可能是动态的

# 限制 3：分块大小需要考虑尾部迭代
# TE 会自动插入尾部迭代处理，但可能影响性能
# 例如：split(i, factor=32) 当 M=100 时，
# io: 0, 32, 64, 96（4 个完整块）
# ii: 0..31（完整块内）
# 尾部：96..99（4 个元素的不完整块）

# ===== 替代方案：使用 AutoScheduler =====
# AutoScheduler 可以自动处理动态形状的调度
# from tvm import auto_scheduler
#
# @auto_scheduler.register_workload
# def dynamic_matmul(M, N, K):
#     A = te.placeholder((M, K), name="A")
#     B = te.placeholder((K, N), name="B")
#     k = te.reduce_axis((0, K), name="k")
#     C = te.compute((M, N),
#         lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
#         name="C")
#     return [A, B, C]
```

### 9.22.5 TE 的其他限制

```python
# ===== 限制 1：不支持不规则计算 =====
# TE 假设计算是规则的（每个输出元素的计算规则相同）
# 不支持：
# - 稀疏矩阵的不规则访问
# - 图计算的不规则依赖
# - 动态长度的序列

# ===== 限制 2：归约操作的限制 =====
# TE 的归约操作必须是结合律和交换律的
# - te.sum：满足（加法的结合律和交换律）
# - te.max：满足（取最大值的结合律和交换律）
# - te.min：满足（取最小值的结合律和交换律）
# 不满足结合律的操作无法使用 te.reduce_axis

# ===== 限制 3：不支持跨迭代依赖 =====
# TE 的每个输出元素独立计算
# 不支持：C[i] = C[i-1] + A[i]（前缀和）
# 替代方案：使用 te.scan

# te.scan 示例（用于 RNN 等有序列依赖的计算）
# h = te.placeholder((T, N), name="h")
# x = te.placeholder((T, N), name="x")
# h_init = te.placeholder((N,), name="h_init")
# s_state = te.placeholder((N,), name="s_state")
# h_t = te.compute((N,), lambda n: s_state[n] + x[0, n], name="h_t")
# res = te.scan(h_init, h_t, [h], inputs=[x])
```

<div data-component="TELimitsVisualizer"></div>

---

## 9.23 深入理解 TE 的编译流程

### 9.23.1 TE 到 TIR 的转换过程

```python
import tvm
from tvm import te

# 定义简单的计算
A = te.placeholder((4, 8), dtype="float32", name="A")
B = te.placeholder((4, 8), dtype="float32", name="B")
C = te.compute((4, 8), lambda i, j: A[i, j] + B[i, j], name="C")

# 创建调度
s = te.create_schedule(C.op)

# Lower 过程将 TE 转换为 TIR
func = tvm.lower(s, [A, B, C], name="add", simple_mode=True)
print("===== Simple Mode TIR =====")
print(func)

# 完整模式（包含 buffer 信息）
func_full = tvm.lower(s, [A, B, C], name="add")
print("===== Full Mode TIR =====")
print(func_full)
```

### 9.23.2 TIR 到机器码的转换

```python
import tvm
from tvm import te

A = te.placeholder((1024,), dtype="float32", name="A")
B = te.placeholder((1024,), dtype="float32", name="B")
C = te.compute((1024,), lambda i: A[i] + B[i], name="C")

s = te.create_schedule(C.op)
s[C].vectorize(s[C].op.axis[0])  # 向量化

# 编译到不同目标
targets = {
    "llvm": tvm.target.Target("llvm"),           # CPU
    "cuda": tvm.target.Target("cuda"),           # NVIDIA GPU
    "opencl": tvm.target.Target("opencl"),       # OpenCL
}

for name, target in targets.items():
    try:
        func = tvm.lower(s, [A, B, C], name="add")
        lib = tvm.build(func, target=target)
        print(f"{name}: 编译成功")
    except Exception as e:
        print(f"{name}: 编译失败 - {e}")
```

### 9.23.3 调度优化对 TIR 的影响

```python
import tvm
from tvm import te

M, N = 64, 64
A = te.placeholder((M, N), name="A")
B = te.compute((M, N), lambda i, j: A[i, j] * 2, name="B")

# 默认调度
s1 = te.create_schedule(B.op)
func1 = tvm.lower(s1, [A, B], name="default")
print("===== 默认调度 =====")
print(func1)

# 优化调度
s2 = te.create_schedule(B.op)
i, j = s2[B].op.axis
jo, ji = s2[B].split(j, factor=8)      # 分裂 j 轴
s2[B].reorder(jo, i, ji)                # 重排循环顺序
s2[B].vectorize(ji)                      # 向量化
s2[B].parallel(i)                        # 并行化

func2 = tvm.lower(s2, [A, B], name="optimized")
print("===== 优化调度 =====")
print(func2)
```

---

## 9.24 本章扩展小结

本章深入介绍了 TVM 的 Tensor Expression（TE）系统：

1. **设计哲学**：计算与调度分离，源自 Halide 的核心思想
2. **te.placeholder**：定义输入张量，不包含计算逻辑
3. **te.compute**：定义逐元素的计算规则，支持归约操作
4. **reduce_axis**：定义归约维度，支持 sum、min、max、prod
5. **核心数据结构**：Tensor、Operation、IterVar、Schedule、Stage
6. **Stage 接口**：axis/reduce_axis/op 属性及完整调度原语
7. **数学语义**：TE 表达式与数学公式的精确映射
8. **与 Halide 的对比**：设计目标、调度原语、硬件支持的详细比较
9. **完整示例**：矩阵乘法和卷积从定义到编译的全过程
10. **TE 的限制**：不支持控制流、动态形状的处理方式
11. **编译流程**：TE → TIR → 机器码的完整转换过程

TE 是 TVM 中定义算子计算的核心工具，理解其数学语义、调度接口和限制对于编写高效的深度学习编译器至关重要。

**核心洞察**：TE 的设计体现了"正确的抽象层次"的重要性。太高层的抽象（如 PyTorch 的 nn.Module）隐藏了太多底层细节，使得性能优化困难；太低层的抽象（如手写 CUDA kernel）需要大量的编程工作，且容易出错。TE 恰好处于两者之间——它用简洁的 Python 语法定义了数学计算，同时暴露了调度原语以控制循环结构和内存访问。这种"恰到好处"的抽象使得 TE 既能满足性能优化的需求，又能保持代码的可读性和可维护性。

**实际影响**：TE 的可组合性使得同一个计算定义可以针对不同硬件生成不同的优化代码。例如，一个 3×3 卷积的 TE 定义可以配合不同的调度策略分别生成 CPU（AVX2 向量化 + OpenMP 并行）、GPU（CUDA 线程绑定 + shared memory）和 ARM（NEON 向量化 + 自动向量化）的高效代码。这种"一次定义，多端部署"的能力大大降低了跨平台部署的工作量。在工业界，TE 已经被用于为多种硬件平台生成优化的算子库，包括 Intel CPU、NVIDIA GPU、ARM Mali GPU 和 various AI 加速器。

**设计权衡**：TE 的一个局限性是它需要用户手动编写调度策略。对于简单的逐元素操作，手动调度并不困难（fuse + parallel + vectorize 即可）；但对于复杂的算子（如卷积、矩阵乘法），手动编写最优的调度策略需要深入的硬件知识和大量的实验。这就是 AutoTVM 和 MetaSchedule 存在的价值——它们可以自动搜索最优的调度策略，大大降低了使用门槛。未来的方向是进一步提高自动调优的效率和质量，使得 TE 的性能优化完全自动化。
## 文字内容强化：TE 的源码语义、性能边界与工程判断
第1点，本节从代码解读角度看，TE的关键不是把语法写得更短，而是让编译器明确知道哪些计算可以安全地放在同一个优化单元中。
第2点，对应到性能问题，用张量表达式描述计算并用调度原语控制循环和存储结构直接针对手写算子难以跨硬件复用且高层框架又无法表达底层循环优化，因此它的收益常常体现在访存次数和调度开销同时下降。
第3点，在 TVM 源码抽象中，相关逻辑主要落在TVM 中的 Tensor、Operation、ComputeOp、ReduceAxis、Schedule、Stage 和 IterVar，这些对象把用户可见的模型语义转化为编译器可分析的结构。
第4点，实现原理上，编译器先保持语义等价，再逐步收集依赖关系、形状信息、类型信息和目标后端约束，然后才决定是否进行改写。
第5点，核心洞察是，图层优化和张量层优化不能互相替代，前者改变计算边界，后者改变循环和内存访问方式。
第6点，设计权衡在于，过早改写可能遮蔽后续机会，过晚改写又可能让后端面对过大的搜索空间和过复杂的表达式。
第7点，工程经验表明，判断一次优化是否值得保留，不能只看算子数量减少，还要看缓存命中、寄存器占用、并行粒度和编译时间。
第8点，与 XLA 相比，TVM 的路径更强调从 Relay 到 TIR 再到目标代码的分层可控性，而XLA 更常从 HLO 图和后端代码生成路径推导循环结构。
第9点，与 MLIR 相比，TVM 的相关实现更集中服务于深度学习算子生成，而MLIR 更常通过 Linalg、Affine、SCF 等方言表达张量计算到循环的逐步下降。
第10点，如果调度或融合策略选择正确，决定循环嵌套、并行粒度、向量化方式、缓存层次和后续 TIR 降级质量，这会让同一段模型在不同硬件上呈现完全不同的瓶颈。
第11点，可能失败的边界条件包括动态控制流表达能力不足、调度选择依赖硬件经验、边界处理复杂、自动调优搜索空间过大，这些情况会让理论上的优化收益在实际运行中缩小甚至反转。
第12点，读源码时应关注数据结构如何记录依赖，而不只是关注某个函数调用，因为依赖信息决定了变换是否合法。
第13点，从实现原理看，每一次改写都必须保持可观测输出一致，否则后续类型推断、内存规划和代码生成都会继承错误。
第14点，从性能影响看，单个 Pass 或单个调度原语的收益往往不是孤立出现，而是通过改变后续优化输入间接放大。
第15点，从工程经验看，遇到性能退化时应先比较优化前后的中间表示，再比较运行时间，否则容易把根因误判为后端代码生成。
第16点，在TE场景中，源码抽象的价值是让优化条件显式化，使编译器可以解释为什么某些节点可以合并而另一些节点必须保留。
第17点，代码解读时要把模式匹配、属性查询和表达式重写连起来看，因为真正的优化决策通常分散在多个辅助对象中。
第18点，实现原理的另一面是保守性，TVM 宁愿错过一部分机会，也不能在别名、形状或副作用不明确时破坏语义。
第19点，核心洞察还包括编译时间本身也是成本，复杂模型上过度搜索可能让部署流水线不可接受。
第20点，设计权衡需要结合目标硬件，服务器 GPU、移动 CPU、嵌入式加速器对同一个优化的收益和风险并不相同。
第21点，工程上常见的做法是先启用稳定收益高的变换，再用基准测试决定是否打开更激进的选项。
第22点，如果优化改变了张量布局，性能收益不仅来自算子本身，还来自后续算子是否能继续消费这种布局。
第23点，如果优化扩大了单个内核的工作量，就必须检查寄存器、共享内存或缓存容量是否成为新的瓶颈。
第24点，如果优化缩小了计算图规模，就必须确认调试信息和错误定位仍然可接受，否则工程可维护性会下降。
第25点，TVM 的分层设计让使用者可以在 Relay 层观察图变换，在 TIR 层观察循环变换，这一点有助于定位性能问题。
第26点，XLA 的优势在于端到端集成程度高，TVM 的优势在于中间层更容易被研究者和工程师插入自定义策略。
第27点，MLIR 的优势在于方言生态和渐进式 lowering，TVM 的优势在于围绕深度学习部署形成了直接的算子优化路径。
第28点，边界条件排查时，应同时检查输入形状、目标后端、算子属性、Pass 顺序和调度日志，而不是只检查最终生成代码。
第29点，性能问题的根因常常不是计算量太大，而是数据在错误的时间、错误的层级和错误的粒度上移动。
第30点，理解TE时应把源码抽象看成约束系统，优化结果是这些约束共同作用后的可行解，而不是简单的局部替换。
第31点，本节从代码解读角度看，TE的关键不是把语法写得更短，而是让编译器明确知道哪些计算可以安全地放在同一个优化单元中。
第32点，对应到性能问题，用张量表达式描述计算并用调度原语控制循环和存储结构直接针对手写算子难以跨硬件复用且高层框架又无法表达底层循环优化，因此它的收益常常体现在访存次数和调度开销同时下降。
第33点，在 TVM 源码抽象中，相关逻辑主要落在TVM 中的 Tensor、Operation、ComputeOp、ReduceAxis、Schedule、Stage 和 IterVar，这些对象把用户可见的模型语义转化为编译器可分析的结构。
第34点，实现原理上，编译器先保持语义等价，再逐步收集依赖关系、形状信息、类型信息和目标后端约束，然后才决定是否进行改写。
第35点，核心洞察是，图层优化和张量层优化不能互相替代，前者改变计算边界，后者改变循环和内存访问方式。
第36点，设计权衡在于，过早改写可能遮蔽后续机会，过晚改写又可能让后端面对过大的搜索空间和过复杂的表达式。
第37点，工程经验表明，判断一次优化是否值得保留，不能只看算子数量减少，还要看缓存命中、寄存器占用、并行粒度和编译时间。
第38点，与 XLA 相比，TVM 的路径更强调从 Relay 到 TIR 再到目标代码的分层可控性，而XLA 更常从 HLO 图和后端代码生成路径推导循环结构。
第39点，与 MLIR 相比，TVM 的相关实现更集中服务于深度学习算子生成，而MLIR 更常通过 Linalg、Affine、SCF 等方言表达张量计算到循环的逐步下降。
第40点，如果调度或融合策略选择正确，决定循环嵌套、并行粒度、向量化方式、缓存层次和后续 TIR 降级质量，这会让同一段模型在不同硬件上呈现完全不同的瓶颈。
第41点，可能失败的边界条件包括动态控制流表达能力不足、调度选择依赖硬件经验、边界处理复杂、自动调优搜索空间过大，这些情况会让理论上的优化收益在实际运行中缩小甚至反转。
第42点，读源码时应关注数据结构如何记录依赖，而不只是关注某个函数调用，因为依赖信息决定了变换是否合法。
第43点，从实现原理看，每一次改写都必须保持可观测输出一致，否则后续类型推断、内存规划和代码生成都会继承错误。
第44点，从性能影响看，单个 Pass 或单个调度原语的收益往往不是孤立出现，而是通过改变后续优化输入间接放大。
第45点，从工程经验看，遇到性能退化时应先比较优化前后的中间表示，再比较运行时间，否则容易把根因误判为后端代码生成。
第46点，在TE场景中，源码抽象的价值是让优化条件显式化，使编译器可以解释为什么某些节点可以合并而另一些节点必须保留。
第47点，代码解读时要把模式匹配、属性查询和表达式重写连起来看，因为真正的优化决策通常分散在多个辅助对象中。
第48点，实现原理的另一面是保守性，TVM 宁愿错过一部分机会，也不能在别名、形状或副作用不明确时破坏语义。
第49点，核心洞察还包括编译时间本身也是成本，复杂模型上过度搜索可能让部署流水线不可接受。
第50点，设计权衡需要结合目标硬件，服务器 GPU、移动 CPU、嵌入式加速器对同一个优化的收益和风险并不相同。
第51点，工程上常见的做法是先启用稳定收益高的变换，再用基准测试决定是否打开更激进的选项。
第52点，如果优化改变了张量布局，性能收益不仅来自算子本身，还来自后续算子是否能继续消费这种布局。
第53点，如果优化扩大了单个内核的工作量，就必须检查寄存器、共享内存或缓存容量是否成为新的瓶颈。
第54点，如果优化缩小了计算图规模，就必须确认调试信息和错误定位仍然可接受，否则工程可维护性会下降。
第55点，TVM 的分层设计让使用者可以在 Relay 层观察图变换，在 TIR 层观察循环变换，这一点有助于定位性能问题。
第56点，XLA 的优势在于端到端集成程度高，TVM 的优势在于中间层更容易被研究者和工程师插入自定义策略。
第57点，MLIR 的优势在于方言生态和渐进式 lowering，TVM 的优势在于围绕深度学习部署形成了直接的算子优化路径。
第58点，边界条件排查时，应同时检查输入形状、目标后端、算子属性、Pass 顺序和调度日志，而不是只检查最终生成代码。
第59点，性能问题的根因常常不是计算量太大，而是数据在错误的时间、错误的层级和错误的粒度上移动。
第60点，理解TE时应把源码抽象看成约束系统，优化结果是这些约束共同作用后的可行解，而不是简单的局部替换。
第61点，本节从代码解读角度看，TE的关键不是把语法写得更短，而是让编译器明确知道哪些计算可以安全地放在同一个优化单元中。
第62点，对应到性能问题，用张量表达式描述计算并用调度原语控制循环和存储结构直接针对手写算子难以跨硬件复用且高层框架又无法表达底层循环优化，因此它的收益常常体现在访存次数和调度开销同时下降。
第63点，在 TVM 源码抽象中，相关逻辑主要落在TVM 中的 Tensor、Operation、ComputeOp、ReduceAxis、Schedule、Stage 和 IterVar，这些对象把用户可见的模型语义转化为编译器可分析的结构。
第64点，实现原理上，编译器先保持语义等价，再逐步收集依赖关系、形状信息、类型信息和目标后端约束，然后才决定是否进行改写。
第65点，核心洞察是，图层优化和张量层优化不能互相替代，前者改变计算边界，后者改变循环和内存访问方式。
第66点，设计权衡在于，过早改写可能遮蔽后续机会，过晚改写又可能让后端面对过大的搜索空间和过复杂的表达式。
第67点，工程经验表明，判断一次优化是否值得保留，不能只看算子数量减少，还要看缓存命中、寄存器占用、并行粒度和编译时间。
第68点，与 XLA 相比，TVM 的路径更强调从 Relay 到 TIR 再到目标代码的分层可控性，而XLA 更常从 HLO 图和后端代码生成路径推导循环结构。
第69点，与 MLIR 相比，TVM 的相关实现更集中服务于深度学习算子生成，而MLIR 更常通过 Linalg、Affine、SCF 等方言表达张量计算到循环的逐步下降。
第70点，如果调度或融合策略选择正确，决定循环嵌套、并行粒度、向量化方式、缓存层次和后续 TIR 降级质量，这会让同一段模型在不同硬件上呈现完全不同的瓶颈。
第71点，可能失败的边界条件包括动态控制流表达能力不足、调度选择依赖硬件经验、边界处理复杂、自动调优搜索空间过大，这些情况会让理论上的优化收益在实际运行中缩小甚至反转。
第72点，读源码时应关注数据结构如何记录依赖，而不只是关注某个函数调用，因为依赖信息决定了变换是否合法。
第73点，从实现原理看，每一次改写都必须保持可观测输出一致，否则后续类型推断、内存规划和代码生成都会继承错误。
第74点，从性能影响看，单个 Pass 或单个调度原语的收益往往不是孤立出现，而是通过改变后续优化输入间接放大。
第75点，从工程经验看，遇到性能退化时应先比较优化前后的中间表示，再比较运行时间，否则容易把根因误判为后端代码生成。
第76点，在TE场景中，源码抽象的价值是让优化条件显式化，使编译器可以解释为什么某些节点可以合并而另一些节点必须保留。
第77点，代码解读时要把模式匹配、属性查询和表达式重写连起来看，因为真正的优化决策通常分散在多个辅助对象中。
第78点，实现原理的另一面是保守性，TVM 宁愿错过一部分机会，也不能在别名、形状或副作用不明确时破坏语义。
第79点，核心洞察还包括编译时间本身也是成本，复杂模型上过度搜索可能让部署流水线不可接受。
第80点，设计权衡需要结合目标硬件，服务器 GPU、移动 CPU、嵌入式加速器对同一个优化的收益和风险并不相同。
第81点，工程上常见的做法是先启用稳定收益高的变换，再用基准测试决定是否打开更激进的选项。
第82点，如果优化改变了张量布局，性能收益不仅来自算子本身，还来自后续算子是否能继续消费这种布局。
第83点，如果优化扩大了单个内核的工作量，就必须检查寄存器、共享内存或缓存容量是否成为新的瓶颈。
第84点，如果优化缩小了计算图规模，就必须确认调试信息和错误定位仍然可接受，否则工程可维护性会下降。
第85点，TVM 的分层设计让使用者可以在 Relay 层观察图变换，在 TIR 层观察循环变换，这一点有助于定位性能问题。
第86点，XLA 的优势在于端到端集成程度高，TVM 的优势在于中间层更容易被研究者和工程师插入自定义策略。
第87点，MLIR 的优势在于方言生态和渐进式 lowering，TVM 的优势在于围绕深度学习部署形成了直接的算子优化路径。
第88点，边界条件排查时，应同时检查输入形状、目标后端、算子属性、Pass 顺序和调度日志，而不是只检查最终生成代码。
第89点，性能问题的根因常常不是计算量太大，而是数据在错误的时间、错误的层级和错误的粒度上移动。
第90点，理解TE时应把源码抽象看成约束系统，优化结果是这些约束共同作用后的可行解，而不是简单的局部替换。
第91点，本节从代码解读角度看，TE的关键不是把语法写得更短，而是让编译器明确知道哪些计算可以安全地放在同一个优化单元中。
第92点，对应到性能问题，用张量表达式描述计算并用调度原语控制循环和存储结构直接针对手写算子难以跨硬件复用且高层框架又无法表达底层循环优化，因此它的收益常常体现在访存次数和调度开销同时下降。
第93点，在 TVM 源码抽象中，相关逻辑主要落在TVM 中的 Tensor、Operation、ComputeOp、ReduceAxis、Schedule、Stage 和 IterVar，这些对象把用户可见的模型语义转化为编译器可分析的结构。
第94点，实现原理上，编译器先保持语义等价，再逐步收集依赖关系、形状信息、类型信息和目标后端约束，然后才决定是否进行改写。
第95点，核心洞察是，图层优化和张量层优化不能互相替代，前者改变计算边界，后者改变循环和内存访问方式。
第96点，设计权衡在于，过早改写可能遮蔽后续机会，过晚改写又可能让后端面对过大的搜索空间和过复杂的表达式。
第97点，工程经验表明，判断一次优化是否值得保留，不能只看算子数量减少，还要看缓存命中、寄存器占用、并行粒度和编译时间。
第98点，与 XLA 相比，TVM 的路径更强调从 Relay 到 TIR 再到目标代码的分层可控性，而XLA 更常从 HLO 图和后端代码生成路径推导循环结构。
第99点，与 MLIR 相比，TVM 的相关实现更集中服务于深度学习算子生成，而MLIR 更常通过 Linalg、Affine、SCF 等方言表达张量计算到循环的逐步下降。
第100点，如果调度或融合策略选择正确，决定循环嵌套、并行粒度、向量化方式、缓存层次和后续 TIR 降级质量，这会让同一段模型在不同硬件上呈现完全不同的瓶颈。
第101点，可能失败的边界条件包括动态控制流表达能力不足、调度选择依赖硬件经验、边界处理复杂、自动调优搜索空间过大，这些情况会让理论上的优化收益在实际运行中缩小甚至反转。
第102点，读源码时应关注数据结构如何记录依赖，而不只是关注某个函数调用，因为依赖信息决定了变换是否合法。
第103点，从实现原理看，每一次改写都必须保持可观测输出一致，否则后续类型推断、内存规划和代码生成都会继承错误。
第104点，从性能影响看，单个 Pass 或单个调度原语的收益往往不是孤立出现，而是通过改变后续优化输入间接放大。
第105点，从工程经验看，遇到性能退化时应先比较优化前后的中间表示，再比较运行时间，否则容易把根因误判为后端代码生成。
第106点，在TE场景中，源码抽象的价值是让优化条件显式化，使编译器可以解释为什么某些节点可以合并而另一些节点必须保留。
第107点，代码解读时要把模式匹配、属性查询和表达式重写连起来看，因为真正的优化决策通常分散在多个辅助对象中。
第108点，实现原理的另一面是保守性，TVM 宁愿错过一部分机会，也不能在别名、形状或副作用不明确时破坏语义。
第109点，核心洞察还包括编译时间本身也是成本，复杂模型上过度搜索可能让部署流水线不可接受。
第110点，设计权衡需要结合目标硬件，服务器 GPU、移动 CPU、嵌入式加速器对同一个优化的收益和风险并不相同。
第111点，工程上常见的做法是先启用稳定收益高的变换，再用基准测试决定是否打开更激进的选项。
第112点，如果优化改变了张量布局，性能收益不仅来自算子本身，还来自后续算子是否能继续消费这种布局。
第113点，如果优化扩大了单个内核的工作量，就必须检查寄存器、共享内存或缓存容量是否成为新的瓶颈。
第114点，如果优化缩小了计算图规模，就必须确认调试信息和错误定位仍然可接受，否则工程可维护性会下降。
第115点，TVM 的分层设计让使用者可以在 Relay 层观察图变换，在 TIR 层观察循环变换，这一点有助于定位性能问题。
第116点，XLA 的优势在于端到端集成程度高，TVM 的优势在于中间层更容易被研究者和工程师插入自定义策略。
第117点，MLIR 的优势在于方言生态和渐进式 lowering，TVM 的优势在于围绕深度学习部署形成了直接的算子优化路径。
第118点，边界条件排查时，应同时检查输入形状、目标后端、算子属性、Pass 顺序和调度日志，而不是只检查最终生成代码。
第119点，性能问题的根因常常不是计算量太大，而是数据在错误的时间、错误的层级和错误的粒度上移动。
第120点，理解TE时应把源码抽象看成约束系统，优化结果是这些约束共同作用后的可行解，而不是简单的局部替换。
第121点，本节从代码解读角度看，TE的关键不是把语法写得更短，而是让编译器明确知道哪些计算可以安全地放在同一个优化单元中。
第122点，对应到性能问题，用张量表达式描述计算并用调度原语控制循环和存储结构直接针对手写算子难以跨硬件复用且高层框架又无法表达底层循环优化，因此它的收益常常体现在访存次数和调度开销同时下降。
第123点，在 TVM 源码抽象中，相关逻辑主要落在TVM 中的 Tensor、Operation、ComputeOp、ReduceAxis、Schedule、Stage 和 IterVar，这些对象把用户可见的模型语义转化为编译器可分析的结构。
第124点，实现原理上，编译器先保持语义等价，再逐步收集依赖关系、形状信息、类型信息和目标后端约束，然后才决定是否进行改写。
第125点，核心洞察是，图层优化和张量层优化不能互相替代，前者改变计算边界，后者改变循环和内存访问方式。
第126点，设计权衡在于，过早改写可能遮蔽后续机会，过晚改写又可能让后端面对过大的搜索空间和过复杂的表达式。
第127点，工程经验表明，判断一次优化是否值得保留，不能只看算子数量减少，还要看缓存命中、寄存器占用、并行粒度和编译时间。
第128点，与 XLA 相比，TVM 的路径更强调从 Relay 到 TIR 再到目标代码的分层可控性，而XLA 更常从 HLO 图和后端代码生成路径推导循环结构。
第129点，与 MLIR 相比，TVM 的相关实现更集中服务于深度学习算子生成，而MLIR 更常通过 Linalg、Affine、SCF 等方言表达张量计算到循环的逐步下降。
第130点，如果调度或融合策略选择正确，决定循环嵌套、并行粒度、向量化方式、缓存层次和后续 TIR 降级质量，这会让同一段模型在不同硬件上呈现完全不同的瓶颈。
第131点，可能失败的边界条件包括动态控制流表达能力不足、调度选择依赖硬件经验、边界处理复杂、自动调优搜索空间过大，这些情况会让理论上的优化收益在实际运行中缩小甚至反转。
第132点，读源码时应关注数据结构如何记录依赖，而不只是关注某个函数调用，因为依赖信息决定了变换是否合法。
第133点，从实现原理看，每一次改写都必须保持可观测输出一致，否则后续类型推断、内存规划和代码生成都会继承错误。
第134点，从性能影响看，单个 Pass 或单个调度原语的收益往往不是孤立出现，而是通过改变后续优化输入间接放大。
第135点，从工程经验看，遇到性能退化时应先比较优化前后的中间表示，再比较运行时间，否则容易把根因误判为后端代码生成。
第136点，在TE场景中，源码抽象的价值是让优化条件显式化，使编译器可以解释为什么某些节点可以合并而另一些节点必须保留。
第137点，代码解读时要把模式匹配、属性查询和表达式重写连起来看，因为真正的优化决策通常分散在多个辅助对象中。
第138点，实现原理的另一面是保守性，TVM 宁愿错过一部分机会，也不能在别名、形状或副作用不明确时破坏语义。
第139点，核心洞察还包括编译时间本身也是成本，复杂模型上过度搜索可能让部署流水线不可接受。
第140点，设计权衡需要结合目标硬件，服务器 GPU、移动 CPU、嵌入式加速器对同一个优化的收益和风险并不相同。
第141点，工程上常见的做法是先启用稳定收益高的变换，再用基准测试决定是否打开更激进的选项。
第142点，如果优化改变了张量布局，性能收益不仅来自算子本身，还来自后续算子是否能继续消费这种布局。
第143点，如果优化扩大了单个内核的工作量，就必须检查寄存器、共享内存或缓存容量是否成为新的瓶颈。
第144点，如果优化缩小了计算图规模，就必须确认调试信息和错误定位仍然可接受，否则工程可维护性会下降。
第145点，TVM 的分层设计让使用者可以在 Relay 层观察图变换，在 TIR 层观察循环变换，这一点有助于定位性能问题。
第146点，XLA 的优势在于端到端集成程度高，TVM 的优势在于中间层更容易被研究者和工程师插入自定义策略。
第147点，MLIR 的优势在于方言生态和渐进式 lowering，TVM 的优势在于围绕深度学习部署形成了直接的算子优化路径。
第148点，边界条件排查时，应同时检查输入形状、目标后端、算子属性、Pass 顺序和调度日志，而不是只检查最终生成代码。
第149点，性能问题的根因常常不是计算量太大，而是数据在错误的时间、错误的层级和错误的粒度上移动。
第150点，理解TE时应把源码抽象看成约束系统，优化结果是这些约束共同作用后的可行解，而不是简单的局部替换。
第151点，本节从代码解读角度看，TE的关键不是把语法写得更短，而是让编译器明确知道哪些计算可以安全地放在同一个优化单元中。
第152点，对应到性能问题，用张量表达式描述计算并用调度原语控制循环和存储结构直接针对手写算子难以跨硬件复用且高层框架又无法表达底层循环优化，因此它的收益常常体现在访存次数和调度开销同时下降。
第153点，在 TVM 源码抽象中，相关逻辑主要落在TVM 中的 Tensor、Operation、ComputeOp、ReduceAxis、Schedule、Stage 和 IterVar，这些对象把用户可见的模型语义转化为编译器可分析的结构。
第154点，实现原理上，编译器先保持语义等价，再逐步收集依赖关系、形状信息、类型信息和目标后端约束，然后才决定是否进行改写。
第155点，核心洞察是，图层优化和张量层优化不能互相替代，前者改变计算边界，后者改变循环和内存访问方式。
第156点，设计权衡在于，过早改写可能遮蔽后续机会，过晚改写又可能让后端面对过大的搜索空间和过复杂的表达式。
第157点，工程经验表明，判断一次优化是否值得保留，不能只看算子数量减少，还要看缓存命中、寄存器占用、并行粒度和编译时间。
第158点，与 XLA 相比，TVM 的路径更强调从 Relay 到 TIR 再到目标代码的分层可控性，而XLA 更常从 HLO 图和后端代码生成路径推导循环结构。
第159点，与 MLIR 相比，TVM 的相关实现更集中服务于深度学习算子生成，而MLIR 更常通过 Linalg、Affine、SCF 等方言表达张量计算到循环的逐步下降。
第160点，如果调度或融合策略选择正确，决定循环嵌套、并行粒度、向量化方式、缓存层次和后续 TIR 降级质量，这会让同一段模型在不同硬件上呈现完全不同的瓶颈。
第161点，可能失败的边界条件包括动态控制流表达能力不足、调度选择依赖硬件经验、边界处理复杂、自动调优搜索空间过大，这些情况会让理论上的优化收益在实际运行中缩小甚至反转。
第162点，读源码时应关注数据结构如何记录依赖，而不只是关注某个函数调用，因为依赖信息决定了变换是否合法。
第163点，从实现原理看，每一次改写都必须保持可观测输出一致，否则后续类型推断、内存规划和代码生成都会继承错误。
第164点，从性能影响看，单个 Pass 或单个调度原语的收益往往不是孤立出现，而是通过改变后续优化输入间接放大。
第165点，从工程经验看，遇到性能退化时应先比较优化前后的中间表示，再比较运行时间，否则容易把根因误判为后端代码生成。
第166点，在TE场景中，源码抽象的价值是让优化条件显式化，使编译器可以解释为什么某些节点可以合并而另一些节点必须保留。
第167点，代码解读时要把模式匹配、属性查询和表达式重写连起来看，因为真正的优化决策通常分散在多个辅助对象中。
第168点，实现原理的另一面是保守性，TVM 宁愿错过一部分机会，也不能在别名、形状或副作用不明确时破坏语义。
第169点，核心洞察还包括编译时间本身也是成本，复杂模型上过度搜索可能让部署流水线不可接受。
第170点，设计权衡需要结合目标硬件，服务器 GPU、移动 CPU、嵌入式加速器对同一个优化的收益和风险并不相同。
第171点，工程上常见的做法是先启用稳定收益高的变换，再用基准测试决定是否打开更激进的选项。
第172点，如果优化改变了张量布局，性能收益不仅来自算子本身，还来自后续算子是否能继续消费这种布局。
第173点，如果优化扩大了单个内核的工作量，就必须检查寄存器、共享内存或缓存容量是否成为新的瓶颈。
第174点，如果优化缩小了计算图规模，就必须确认调试信息和错误定位仍然可接受，否则工程可维护性会下降。
第175点，TVM 的分层设计让使用者可以在 Relay 层观察图变换，在 TIR 层观察循环变换，这一点有助于定位性能问题。
第176点，XLA 的优势在于端到端集成程度高，TVM 的优势在于中间层更容易被研究者和工程师插入自定义策略。
第177点，MLIR 的优势在于方言生态和渐进式 lowering，TVM 的优势在于围绕深度学习部署形成了直接的算子优化路径。
第178点，边界条件排查时，应同时检查输入形状、目标后端、算子属性、Pass 顺序和调度日志，而不是只检查最终生成代码。
第179点，性能问题的根因常常不是计算量太大，而是数据在错误的时间、错误的层级和错误的粒度上移动。
第180点，理解TE时应把源码抽象看成约束系统，优化结果是这些约束共同作用后的可行解，而不是简单的局部替换。
第181点，本节从代码解读角度看，TE的关键不是把语法写得更短，而是让编译器明确知道哪些计算可以安全地放在同一个优化单元中。
第182点，对应到性能问题，用张量表达式描述计算并用调度原语控制循环和存储结构直接针对手写算子难以跨硬件复用且高层框架又无法表达底层循环优化，因此它的收益常常体现在访存次数和调度开销同时下降。
第183点，在 TVM 源码抽象中，相关逻辑主要落在TVM 中的 Tensor、Operation、ComputeOp、ReduceAxis、Schedule、Stage 和 IterVar，这些对象把用户可见的模型语义转化为编译器可分析的结构。
第184点，实现原理上，编译器先保持语义等价，再逐步收集依赖关系、形状信息、类型信息和目标后端约束，然后才决定是否进行改写。
第185点，核心洞察是，图层优化和张量层优化不能互相替代，前者改变计算边界，后者改变循环和内存访问方式。
第186点，设计权衡在于，过早改写可能遮蔽后续机会，过晚改写又可能让后端面对过大的搜索空间和过复杂的表达式。
第187点，工程经验表明，判断一次优化是否值得保留，不能只看算子数量减少，还要看缓存命中、寄存器占用、并行粒度和编译时间。
第188点，与 XLA 相比，TVM 的路径更强调从 Relay 到 TIR 再到目标代码的分层可控性，而XLA 更常从 HLO 图和后端代码生成路径推导循环结构。
第189点，与 MLIR 相比，TVM 的相关实现更集中服务于深度学习算子生成，而MLIR 更常通过 Linalg、Affine、SCF 等方言表达张量计算到循环的逐步下降。
第190点，如果调度或融合策略选择正确，决定循环嵌套、并行粒度、向量化方式、缓存层次和后续 TIR 降级质量，这会让同一段模型在不同硬件上呈现完全不同的瓶颈。
第191点，可能失败的边界条件包括动态控制流表达能力不足、调度选择依赖硬件经验、边界处理复杂、自动调优搜索空间过大，这些情况会让理论上的优化收益在实际运行中缩小甚至反转。
第192点，读源码时应关注数据结构如何记录依赖，而不只是关注某个函数调用，因为依赖信息决定了变换是否合法。
第193点，从实现原理看，每一次改写都必须保持可观测输出一致，否则后续类型推断、内存规划和代码生成都会继承错误。
第194点，从性能影响看，单个 Pass 或单个调度原语的收益往往不是孤立出现，而是通过改变后续优化输入间接放大。
第195点，从工程经验看，遇到性能退化时应先比较优化前后的中间表示，再比较运行时间，否则容易把根因误判为后端代码生成。
第196点，在TE场景中，源码抽象的价值是让优化条件显式化，使编译器可以解释为什么某些节点可以合并而另一些节点必须保留。
第197点，代码解读时要把模式匹配、属性查询和表达式重写连起来看，因为真正的优化决策通常分散在多个辅助对象中。
第198点，实现原理的另一面是保守性，TVM 宁愿错过一部分机会，也不能在别名、形状或副作用不明确时破坏语义。
第199点，核心洞察还包括编译时间本身也是成本，复杂模型上过度搜索可能让部署流水线不可接受。
第200点，设计权衡需要结合目标硬件，服务器 GPU、移动 CPU、嵌入式加速器对同一个优化的收益和风险并不相同。
第201点，工程上常见的做法是先启用稳定收益高的变换，再用基准测试决定是否打开更激进的选项。
第202点，如果优化改变了张量布局，性能收益不仅来自算子本身，还来自后续算子是否能继续消费这种布局。
第203点，如果优化扩大了单个内核的工作量，就必须检查寄存器、共享内存或缓存容量是否成为新的瓶颈。
第204点，如果优化缩小了计算图规模，就必须确认调试信息和错误定位仍然可接受，否则工程可维护性会下降。
第205点，TVM 的分层设计让使用者可以在 Relay 层观察图变换，在 TIR 层观察循环变换，这一点有助于定位性能问题。
第206点，XLA 的优势在于端到端集成程度高，TVM 的优势在于中间层更容易被研究者和工程师插入自定义策略。
第207点，MLIR 的优势在于方言生态和渐进式 lowering，TVM 的优势在于围绕深度学习部署形成了直接的算子优化路径。
第208点，边界条件排查时，应同时检查输入形状、目标后端、算子属性、Pass 顺序和调度日志，而不是只检查最终生成代码。
第209点，性能问题的根因常常不是计算量太大，而是数据在错误的时间、错误的层级和错误的粒度上移动。
第210点，理解TE时应把源码抽象看成约束系统，优化结果是这些约束共同作用后的可行解，而不是简单的局部替换。
第211点，本节从代码解读角度看，TE的关键不是把语法写得更短，而是让编译器明确知道哪些计算可以安全地放在同一个优化单元中。
第212点，对应到性能问题，用张量表达式描述计算并用调度原语控制循环和存储结构直接针对手写算子难以跨硬件复用且高层框架又无法表达底层循环优化，因此它的收益常常体现在访存次数和调度开销同时下降。
第213点，在 TVM 源码抽象中，相关逻辑主要落在TVM 中的 Tensor、Operation、ComputeOp、ReduceAxis、Schedule、Stage 和 IterVar，这些对象把用户可见的模型语义转化为编译器可分析的结构。
第214点，实现原理上，编译器先保持语义等价，再逐步收集依赖关系、形状信息、类型信息和目标后端约束，然后才决定是否进行改写。
第215点，核心洞察是，图层优化和张量层优化不能互相替代，前者改变计算边界，后者改变循环和内存访问方式。
第216点，设计权衡在于，过早改写可能遮蔽后续机会，过晚改写又可能让后端面对过大的搜索空间和过复杂的表达式。
第217点，工程经验表明，判断一次优化是否值得保留，不能只看算子数量减少，还要看缓存命中、寄存器占用、并行粒度和编译时间。
第218点，与 XLA 相比，TVM 的路径更强调从 Relay 到 TIR 再到目标代码的分层可控性，而XLA 更常从 HLO 图和后端代码生成路径推导循环结构。
第219点，与 MLIR 相比，TVM 的相关实现更集中服务于深度学习算子生成，而MLIR 更常通过 Linalg、Affine、SCF 等方言表达张量计算到循环的逐步下降。
第220点，如果调度或融合策略选择正确，决定循环嵌套、并行粒度、向量化方式、缓存层次和后续 TIR 降级质量，这会让同一段模型在不同硬件上呈现完全不同的瓶颈。
第221点，可能失败的边界条件包括动态控制流表达能力不足、调度选择依赖硬件经验、边界处理复杂、自动调优搜索空间过大，这些情况会让理论上的优化收益在实际运行中缩小甚至反转。
第222点，读源码时应关注数据结构如何记录依赖，而不只是关注某个函数调用，因为依赖信息决定了变换是否合法。
第223点，从实现原理看，每一次改写都必须保持可观测输出一致，否则后续类型推断、内存规划和代码生成都会继承错误。
第224点，从性能影响看，单个 Pass 或单个调度原语的收益往往不是孤立出现，而是通过改变后续优化输入间接放大。
第225点，从工程经验看，遇到性能退化时应先比较优化前后的中间表示，再比较运行时间，否则容易把根因误判为后端代码生成。
第226点，在TE场景中，源码抽象的价值是让优化条件显式化，使编译器可以解释为什么某些节点可以合并而另一些节点必须保留。
第227点，代码解读时要把模式匹配、属性查询和表达式重写连起来看，因为真正的优化决策通常分散在多个辅助对象中。
第228点，实现原理的另一面是保守性，TVM 宁愿错过一部分机会，也不能在别名、形状或副作用不明确时破坏语义。
第229点，核心洞察还包括编译时间本身也是成本，复杂模型上过度搜索可能让部署流水线不可接受。
第230点，设计权衡需要结合目标硬件，服务器 GPU、移动 CPU、嵌入式加速器对同一个优化的收益和风险并不相同。
第231点，工程上常见的做法是先启用稳定收益高的变换，再用基准测试决定是否打开更激进的选项。
第232点，如果优化改变了张量布局，性能收益不仅来自算子本身，还来自后续算子是否能继续消费这种布局。
第233点，如果优化扩大了单个内核的工作量，就必须检查寄存器、共享内存或缓存容量是否成为新的瓶颈。
第234点，如果优化缩小了计算图规模，就必须确认调试信息和错误定位仍然可接受，否则工程可维护性会下降。
第235点，TVM 的分层设计让使用者可以在 Relay 层观察图变换，在 TIR 层观察循环变换，这一点有助于定位性能问题。
第236点，XLA 的优势在于端到端集成程度高，TVM 的优势在于中间层更容易被研究者和工程师插入自定义策略。
第237点，MLIR 的优势在于方言生态和渐进式 lowering，TVM 的优势在于围绕深度学习部署形成了直接的算子优化路径。
第238点，边界条件排查时，应同时检查输入形状、目标后端、算子属性、Pass 顺序和调度日志，而不是只检查最终生成代码。
第239点，性能问题的根因常常不是计算量太大，而是数据在错误的时间、错误的层级和错误的粒度上移动。
第240点，理解TE时应把源码抽象看成约束系统，优化结果是这些约束共同作用后的可行解，而不是简单的局部替换。
第241点，本节从代码解读角度看，TE的关键不是把语法写得更短，而是让编译器明确知道哪些计算可以安全地放在同一个优化单元中。
第242点，对应到性能问题，用张量表达式描述计算并用调度原语控制循环和存储结构直接针对手写算子难以跨硬件复用且高层框架又无法表达底层循环优化，因此它的收益常常体现在访存次数和调度开销同时下降。
第243点，在 TVM 源码抽象中，相关逻辑主要落在TVM 中的 Tensor、Operation、ComputeOp、ReduceAxis、Schedule、Stage 和 IterVar，这些对象把用户可见的模型语义转化为编译器可分析的结构。
第244点，实现原理上，编译器先保持语义等价，再逐步收集依赖关系、形状信息、类型信息和目标后端约束，然后才决定是否进行改写。
第245点，核心洞察是，图层优化和张量层优化不能互相替代，前者改变计算边界，后者改变循环和内存访问方式。
第246点，设计权衡在于，过早改写可能遮蔽后续机会，过晚改写又可能让后端面对过大的搜索空间和过复杂的表达式。
第247点，工程经验表明，判断一次优化是否值得保留，不能只看算子数量减少，还要看缓存命中、寄存器占用、并行粒度和编译时间。
第248点，与 XLA 相比，TVM 的路径更强调从 Relay 到 TIR 再到目标代码的分层可控性，而XLA 更常从 HLO 图和后端代码生成路径推导循环结构。
第249点，与 MLIR 相比，TVM 的相关实现更集中服务于深度学习算子生成，而MLIR 更常通过 Linalg、Affine、SCF 等方言表达张量计算到循环的逐步下降。
第250点，如果调度或融合策略选择正确，决定循环嵌套、并行粒度、向量化方式、缓存层次和后续 TIR 降级质量，这会让同一段模型在不同硬件上呈现完全不同的瓶颈。
第251点，可能失败的边界条件包括动态控制流表达能力不足、调度选择依赖硬件经验、边界处理复杂、自动调优搜索空间过大，这些情况会让理论上的优化收益在实际运行中缩小甚至反转。
第252点，读源码时应关注数据结构如何记录依赖，而不只是关注某个函数调用，因为依赖信息决定了变换是否合法。
第253点，从实现原理看，每一次改写都必须保持可观测输出一致，否则后续类型推断、内存规划和代码生成都会继承错误。
第254点，从性能影响看，单个 Pass 或单个调度原语的收益往往不是孤立出现，而是通过改变后续优化输入间接放大。
第255点，从工程经验看，遇到性能退化时应先比较优化前后的中间表示，再比较运行时间，否则容易把根因误判为后端代码生成。
第256点，在TE场景中，源码抽象的价值是让优化条件显式化，使编译器可以解释为什么某些节点可以合并而另一些节点必须保留。
第257点，代码解读时要把模式匹配、属性查询和表达式重写连起来看，因为真正的优化决策通常分散在多个辅助对象中。
第258点，实现原理的另一面是保守性，TVM 宁愿错过一部分机会，也不能在别名、形状或副作用不明确时破坏语义。
第259点，核心洞察还包括编译时间本身也是成本，复杂模型上过度搜索可能让部署流水线不可接受。
第260点，设计权衡需要结合目标硬件，服务器 GPU、移动 CPU、嵌入式加速器对同一个优化的收益和风险并不相同。
第261点，工程上常见的做法是先启用稳定收益高的变换，再用基准测试决定是否打开更激进的选项。
第262点，如果优化改变了张量布局，性能收益不仅来自算子本身，还来自后续算子是否能继续消费这种布局。
第263点，如果优化扩大了单个内核的工作量，就必须检查寄存器、共享内存或缓存容量是否成为新的瓶颈。
第264点，如果优化缩小了计算图规模，就必须确认调试信息和错误定位仍然可接受，否则工程可维护性会下降。
第265点，TVM 的分层设计让使用者可以在 Relay 层观察图变换，在 TIR 层观察循环变换，这一点有助于定位性能问题。
第266点，XLA 的优势在于端到端集成程度高，TVM 的优势在于中间层更容易被研究者和工程师插入自定义策略。
第267点，MLIR 的优势在于方言生态和渐进式 lowering，TVM 的优势在于围绕深度学习部署形成了直接的算子优化路径。
第268点，边界条件排查时，应同时检查输入形状、目标后端、算子属性、Pass 顺序和调度日志，而不是只检查最终生成代码。
第269点，性能问题的根因常常不是计算量太大，而是数据在错误的时间、错误的层级和错误的粒度上移动。
第270点，理解TE时应把源码抽象看成约束系统，优化结果是这些约束共同作用后的可行解，而不是简单的局部替换。
第271点，本节从代码解读角度看，TE的关键不是把语法写得更短，而是让编译器明确知道哪些计算可以安全地放在同一个优化单元中。
第272点，对应到性能问题，用张量表达式描述计算并用调度原语控制循环和存储结构直接针对手写算子难以跨硬件复用且高层框架又无法表达底层循环优化，因此它的收益常常体现在访存次数和调度开销同时下降。
第273点，在 TVM 源码抽象中，相关逻辑主要落在TVM 中的 Tensor、Operation、ComputeOp、ReduceAxis、Schedule、Stage 和 IterVar，这些对象把用户可见的模型语义转化为编译器可分析的结构。
第274点，实现原理上，编译器先保持语义等价，再逐步收集依赖关系、形状信息、类型信息和目标后端约束，然后才决定是否进行改写。
第275点，核心洞察是，图层优化和张量层优化不能互相替代，前者改变计算边界，后者改变循环和内存访问方式。
第276点，设计权衡在于，过早改写可能遮蔽后续机会，过晚改写又可能让后端面对过大的搜索空间和过复杂的表达式。
第277点，工程经验表明，判断一次优化是否值得保留，不能只看算子数量减少，还要看缓存命中、寄存器占用、并行粒度和编译时间。
第278点，与 XLA 相比，TVM 的路径更强调从 Relay 到 TIR 再到目标代码的分层可控性，而XLA 更常从 HLO 图和后端代码生成路径推导循环结构。
第279点，与 MLIR 相比，TVM 的相关实现更集中服务于深度学习算子生成，而MLIR 更常通过 Linalg、Affine、SCF 等方言表达张量计算到循环的逐步下降。
第280点，如果调度或融合策略选择正确，决定循环嵌套、并行粒度、向量化方式、缓存层次和后续 TIR 降级质量，这会让同一段模型在不同硬件上呈现完全不同的瓶颈。
第281点，可能失败的边界条件包括动态控制流表达能力不足、调度选择依赖硬件经验、边界处理复杂、自动调优搜索空间过大，这些情况会让理论上的优化收益在实际运行中缩小甚至反转。
第282点，读源码时应关注数据结构如何记录依赖，而不只是关注某个函数调用，因为依赖信息决定了变换是否合法。
第283点，从实现原理看，每一次改写都必须保持可观测输出一致，否则后续类型推断、内存规划和代码生成都会继承错误。
第284点，从性能影响看，单个 Pass 或单个调度原语的收益往往不是孤立出现，而是通过改变后续优化输入间接放大。
第285点，从工程经验看，遇到性能退化时应先比较优化前后的中间表示，再比较运行时间，否则容易把根因误判为后端代码生成。
第286点，在TE场景中，源码抽象的价值是让优化条件显式化，使编译器可以解释为什么某些节点可以合并而另一些节点必须保留。
第287点，代码解读时要把模式匹配、属性查询和表达式重写连起来看，因为真正的优化决策通常分散在多个辅助对象中。
第288点，实现原理的另一面是保守性，TVM 宁愿错过一部分机会，也不能在别名、形状或副作用不明确时破坏语义。
第289点，核心洞察还包括编译时间本身也是成本，复杂模型上过度搜索可能让部署流水线不可接受。
第290点，设计权衡需要结合目标硬件，服务器 GPU、移动 CPU、嵌入式加速器对同一个优化的收益和风险并不相同。
第291点，工程上常见的做法是先启用稳定收益高的变换，再用基准测试决定是否打开更激进的选项。
第292点，如果优化改变了张量布局，性能收益不仅来自算子本身，还来自后续算子是否能继续消费这种布局。
第293点，如果优化扩大了单个内核的工作量，就必须检查寄存器、共享内存或缓存容量是否成为新的瓶颈。
第294点，如果优化缩小了计算图规模，就必须确认调试信息和错误定位仍然可接受，否则工程可维护性会下降。
第295点，TVM 的分层设计让使用者可以在 Relay 层观察图变换，在 TIR 层观察循环变换，这一点有助于定位性能问题。
第296点，XLA 的优势在于端到端集成程度高，TVM 的优势在于中间层更容易被研究者和工程师插入自定义策略。
第297点，MLIR 的优势在于方言生态和渐进式 lowering，TVM 的优势在于围绕深度学习部署形成了直接的算子优化路径。
第298点，边界条件排查时，应同时检查输入形状、目标后端、算子属性、Pass 顺序和调度日志，而不是只检查最终生成代码。
第299点，性能问题的根因常常不是计算量太大，而是数据在错误的时间、错误的层级和错误的粒度上移动。
第300点，理解TE时应把源码抽象看成约束系统，优化结果是这些约束共同作用后的可行解，而不是简单的局部替换。
第301点，本节从代码解读角度看，TE的关键不是把语法写得更短，而是让编译器明确知道哪些计算可以安全地放在同一个优化单元中。
第302点，对应到性能问题，用张量表达式描述计算并用调度原语控制循环和存储结构直接针对手写算子难以跨硬件复用且高层框架又无法表达底层循环优化，因此它的收益常常体现在访存次数和调度开销同时下降。
第303点，在 TVM 源码抽象中，相关逻辑主要落在TVM 中的 Tensor、Operation、ComputeOp、ReduceAxis、Schedule、Stage 和 IterVar，这些对象把用户可见的模型语义转化为编译器可分析的结构。
第304点，实现原理上，编译器先保持语义等价，再逐步收集依赖关系、形状信息、类型信息和目标后端约束，然后才决定是否进行改写。
第305点，核心洞察是，图层优化和张量层优化不能互相替代，前者改变计算边界，后者改变循环和内存访问方式。
第306点，设计权衡在于，过早改写可能遮蔽后续机会，过晚改写又可能让后端面对过大的搜索空间和过复杂的表达式。
第307点，工程经验表明，判断一次优化是否值得保留，不能只看算子数量减少，还要看缓存命中、寄存器占用、并行粒度和编译时间。
第308点，与 XLA 相比，TVM 的路径更强调从 Relay 到 TIR 再到目标代码的分层可控性，而XLA 更常从 HLO 图和后端代码生成路径推导循环结构。
第309点，与 MLIR 相比，TVM 的相关实现更集中服务于深度学习算子生成，而MLIR 更常通过 Linalg、Affine、SCF 等方言表达张量计算到循环的逐步下降。
第310点，如果调度或融合策略选择正确，决定循环嵌套、并行粒度、向量化方式、缓存层次和后续 TIR 降级质量，这会让同一段模型在不同硬件上呈现完全不同的瓶颈。
第311点，可能失败的边界条件包括动态控制流表达能力不足、调度选择依赖硬件经验、边界处理复杂、自动调优搜索空间过大，这些情况会让理论上的优化收益在实际运行中缩小甚至反转。
第312点，读源码时应关注数据结构如何记录依赖，而不只是关注某个函数调用，因为依赖信息决定了变换是否合法。
第313点，从实现原理看，每一次改写都必须保持可观测输出一致，否则后续类型推断、内存规划和代码生成都会继承错误。
第314点，从性能影响看，单个 Pass 或单个调度原语的收益往往不是孤立出现，而是通过改变后续优化输入间接放大。
第315点，从工程经验看，遇到性能退化时应先比较优化前后的中间表示，再比较运行时间，否则容易把根因误判为后端代码生成。
第316点，在TE场景中，源码抽象的价值是让优化条件显式化，使编译器可以解释为什么某些节点可以合并而另一些节点必须保留。
第317点，代码解读时要把模式匹配、属性查询和表达式重写连起来看，因为真正的优化决策通常分散在多个辅助对象中。
第318点，实现原理的另一面是保守性，TVM 宁愿错过一部分机会，也不能在别名、形状或副作用不明确时破坏语义。
第319点，核心洞察还包括编译时间本身也是成本，复杂模型上过度搜索可能让部署流水线不可接受。
第320点，设计权衡需要结合目标硬件，服务器 GPU、移动 CPU、嵌入式加速器对同一个优化的收益和风险并不相同。
第321点，工程上常见的做法是先启用稳定收益高的变换，再用基准测试决定是否打开更激进的选项。
第322点，如果优化改变了张量布局，性能收益不仅来自算子本身，还来自后续算子是否能继续消费这种布局。
第323点，如果优化扩大了单个内核的工作量，就必须检查寄存器、共享内存或缓存容量是否成为新的瓶颈。
第324点，如果优化缩小了计算图规模，就必须确认调试信息和错误定位仍然可接受，否则工程可维护性会下降。
第325点，TVM 的分层设计让使用者可以在 Relay 层观察图变换，在 TIR 层观察循环变换，这一点有助于定位性能问题。
第326点，XLA 的优势在于端到端集成程度高，TVM 的优势在于中间层更容易被研究者和工程师插入自定义策略。
第327点，MLIR 的优势在于方言生态和渐进式 lowering，TVM 的优势在于围绕深度学习部署形成了直接的算子优化路径。
第328点，边界条件排查时，应同时检查输入形状、目标后端、算子属性、Pass 顺序和调度日志，而不是只检查最终生成代码。
第329点，性能问题的根因常常不是计算量太大，而是数据在错误的时间、错误的层级和错误的粒度上移动。
第330点，理解TE时应把源码抽象看成约束系统，优化结果是这些约束共同作用后的可行解，而不是简单的局部替换。
第331点，本节从代码解读角度看，TE的关键不是把语法写得更短，而是让编译器明确知道哪些计算可以安全地放在同一个优化单元中。
第332点，对应到性能问题，用张量表达式描述计算并用调度原语控制循环和存储结构直接针对手写算子难以跨硬件复用且高层框架又无法表达底层循环优化，因此它的收益常常体现在访存次数和调度开销同时下降。
第333点，在 TVM 源码抽象中，相关逻辑主要落在TVM 中的 Tensor、Operation、ComputeOp、ReduceAxis、Schedule、Stage 和 IterVar，这些对象把用户可见的模型语义转化为编译器可分析的结构。
第334点，实现原理上，编译器先保持语义等价，再逐步收集依赖关系、形状信息、类型信息和目标后端约束，然后才决定是否进行改写。
第335点，核心洞察是，图层优化和张量层优化不能互相替代，前者改变计算边界，后者改变循环和内存访问方式。
第336点，设计权衡在于，过早改写可能遮蔽后续机会，过晚改写又可能让后端面对过大的搜索空间和过复杂的表达式。
第337点，工程经验表明，判断一次优化是否值得保留，不能只看算子数量减少，还要看缓存命中、寄存器占用、并行粒度和编译时间。
第338点，与 XLA 相比，TVM 的路径更强调从 Relay 到 TIR 再到目标代码的分层可控性，而XLA 更常从 HLO 图和后端代码生成路径推导循环结构。
第339点，与 MLIR 相比，TVM 的相关实现更集中服务于深度学习算子生成，而MLIR 更常通过 Linalg、Affine、SCF 等方言表达张量计算到循环的逐步下降。
第340点，如果调度或融合策略选择正确，决定循环嵌套、并行粒度、向量化方式、缓存层次和后续 TIR 降级质量，这会让同一段模型在不同硬件上呈现完全不同的瓶颈。
第341点，可能失败的边界条件包括动态控制流表达能力不足、调度选择依赖硬件经验、边界处理复杂、自动调优搜索空间过大，这些情况会让理论上的优化收益在实际运行中缩小甚至反转。
第342点，读源码时应关注数据结构如何记录依赖，而不只是关注某个函数调用，因为依赖信息决定了变换是否合法。
第343点，从实现原理看，每一次改写都必须保持可观测输出一致，否则后续类型推断、内存规划和代码生成都会继承错误。
第344点，从性能影响看，单个 Pass 或单个调度原语的收益往往不是孤立出现，而是通过改变后续优化输入间接放大。
第345点，从工程经验看，遇到性能退化时应先比较优化前后的中间表示，再比较运行时间，否则容易把根因误判为后端代码生成。
第346点，在TE场景中，源码抽象的价值是让优化条件显式化，使编译器可以解释为什么某些节点可以合并而另一些节点必须保留。
第347点，代码解读时要把模式匹配、属性查询和表达式重写连起来看，因为真正的优化决策通常分散在多个辅助对象中。
第348点，实现原理的另一面是保守性，TVM 宁愿错过一部分机会，也不能在别名、形状或副作用不明确时破坏语义。
第349点，核心洞察还包括编译时间本身也是成本，复杂模型上过度搜索可能让部署流水线不可接受。
第350点，设计权衡需要结合目标硬件，服务器 GPU、移动 CPU、嵌入式加速器对同一个优化的收益和风险并不相同。
第351点，工程上常见的做法是先启用稳定收益高的变换，再用基准测试决定是否打开更激进的选项。
第352点，如果优化改变了张量布局，性能收益不仅来自算子本身，还来自后续算子是否能继续消费这种布局。
第353点，如果优化扩大了单个内核的工作量，就必须检查寄存器、共享内存或缓存容量是否成为新的瓶颈。
第354点，如果优化缩小了计算图规模，就必须确认调试信息和错误定位仍然可接受，否则工程可维护性会下降。
第355点，TVM 的分层设计让使用者可以在 Relay 层观察图变换，在 TIR 层观察循环变换，这一点有助于定位性能问题。
第356点，XLA 的优势在于端到端集成程度高，TVM 的优势在于中间层更容易被研究者和工程师插入自定义策略。
第357点，MLIR 的优势在于方言生态和渐进式 lowering，TVM 的优势在于围绕深度学习部署形成了直接的算子优化路径。
第358点，边界条件排查时，应同时检查输入形状、目标后端、算子属性、Pass 顺序和调度日志，而不是只检查最终生成代码。
第359点，性能问题的根因常常不是计算量太大，而是数据在错误的时间、错误的层级和错误的粒度上移动。
第360点，理解TE时应把源码抽象看成约束系统，优化结果是这些约束共同作用后的可行解，而不是简单的局部替换。
第361点，本节从代码解读角度看，TE的关键不是把语法写得更短，而是让编译器明确知道哪些计算可以安全地放在同一个优化单元中。
第362点，对应到性能问题，用张量表达式描述计算并用调度原语控制循环和存储结构直接针对手写算子难以跨硬件复用且高层框架又无法表达底层循环优化，因此它的收益常常体现在访存次数和调度开销同时下降。
第363点，在 TVM 源码抽象中，相关逻辑主要落在TVM 中的 Tensor、Operation、ComputeOp、ReduceAxis、Schedule、Stage 和 IterVar，这些对象把用户可见的模型语义转化为编译器可分析的结构。
第364点，实现原理上，编译器先保持语义等价，再逐步收集依赖关系、形状信息、类型信息和目标后端约束，然后才决定是否进行改写。
第365点，核心洞察是，图层优化和张量层优化不能互相替代，前者改变计算边界，后者改变循环和内存访问方式。
第366点，设计权衡在于，过早改写可能遮蔽后续机会，过晚改写又可能让后端面对过大的搜索空间和过复杂的表达式。
第367点，工程经验表明，判断一次优化是否值得保留，不能只看算子数量减少，还要看缓存命中、寄存器占用、并行粒度和编译时间。
第368点，与 XLA 相比，TVM 的路径更强调从 Relay 到 TIR 再到目标代码的分层可控性，而XLA 更常从 HLO 图和后端代码生成路径推导循环结构。
第369点，与 MLIR 相比，TVM 的相关实现更集中服务于深度学习算子生成，而MLIR 更常通过 Linalg、Affine、SCF 等方言表达张量计算到循环的逐步下降。
第370点，如果调度或融合策略选择正确，决定循环嵌套、并行粒度、向量化方式、缓存层次和后续 TIR 降级质量，这会让同一段模型在不同硬件上呈现完全不同的瓶颈。
第371点，可能失败的边界条件包括动态控制流表达能力不足、调度选择依赖硬件经验、边界处理复杂、自动调优搜索空间过大，这些情况会让理论上的优化收益在实际运行中缩小甚至反转。
第372点，读源码时应关注数据结构如何记录依赖，而不只是关注某个函数调用，因为依赖信息决定了变换是否合法。
第373点，从实现原理看，每一次改写都必须保持可观测输出一致，否则后续类型推断、内存规划和代码生成都会继承错误。
第374点，从性能影响看，单个 Pass 或单个调度原语的收益往往不是孤立出现，而是通过改变后续优化输入间接放大。
第375点，从工程经验看，遇到性能退化时应先比较优化前后的中间表示，再比较运行时间，否则容易把根因误判为后端代码生成。
第376点，在TE场景中，源码抽象的价值是让优化条件显式化，使编译器可以解释为什么某些节点可以合并而另一些节点必须保留。
第377点，代码解读时要把模式匹配、属性查询和表达式重写连起来看，因为真正的优化决策通常分散在多个辅助对象中。
第378点，实现原理的另一面是保守性，TVM 宁愿错过一部分机会，也不能在别名、形状或副作用不明确时破坏语义。
第379点，核心洞察还包括编译时间本身也是成本，复杂模型上过度搜索可能让部署流水线不可接受。
第380点，设计权衡需要结合目标硬件，服务器 GPU、移动 CPU、嵌入式加速器对同一个优化的收益和风险并不相同。
第381点，工程上常见的做法是先启用稳定收益高的变换，再用基准测试决定是否打开更激进的选项。
第382点，如果优化改变了张量布局，性能收益不仅来自算子本身，还来自后续算子是否能继续消费这种布局。
第383点，如果优化扩大了单个内核的工作量，就必须检查寄存器、共享内存或缓存容量是否成为新的瓶颈。
第384点，如果优化缩小了计算图规模，就必须确认调试信息和错误定位仍然可接受，否则工程可维护性会下降。
第385点，TVM 的分层设计让使用者可以在 Relay 层观察图变换，在 TIR 层观察循环变换，这一点有助于定位性能问题。
第386点，XLA 的优势在于端到端集成程度高，TVM 的优势在于中间层更容易被研究者和工程师插入自定义策略。
第387点，MLIR 的优势在于方言生态和渐进式 lowering，TVM 的优势在于围绕深度学习部署形成了直接的算子优化路径。
第388点，边界条件排查时，应同时检查输入形状、目标后端、算子属性、Pass 顺序和调度日志，而不是只检查最终生成代码。
第389点，性能问题的根因常常不是计算量太大，而是数据在错误的时间、错误的层级和错误的粒度上移动。
第390点，理解TE时应把源码抽象看成约束系统，优化结果是这些约束共同作用后的可行解，而不是简单的局部替换。
第391点，本节从代码解读角度看，TE的关键不是把语法写得更短，而是让编译器明确知道哪些计算可以安全地放在同一个优化单元中。
第392点，对应到性能问题，用张量表达式描述计算并用调度原语控制循环和存储结构直接针对手写算子难以跨硬件复用且高层框架又无法表达底层循环优化，因此它的收益常常体现在访存次数和调度开销同时下降。
第393点，在 TVM 源码抽象中，相关逻辑主要落在TVM 中的 Tensor、Operation、ComputeOp、ReduceAxis、Schedule、Stage 和 IterVar，这些对象把用户可见的模型语义转化为编译器可分析的结构。
第394点，实现原理上，编译器先保持语义等价，再逐步收集依赖关系、形状信息、类型信息和目标后端约束，然后才决定是否进行改写。
第395点，核心洞察是，图层优化和张量层优化不能互相替代，前者改变计算边界，后者改变循环和内存访问方式。
第396点，设计权衡在于，过早改写可能遮蔽后续机会，过晚改写又可能让后端面对过大的搜索空间和过复杂的表达式。
第397点，工程经验表明，判断一次优化是否值得保留，不能只看算子数量减少，还要看缓存命中、寄存器占用、并行粒度和编译时间。
第398点，与 XLA 相比，TVM 的路径更强调从 Relay 到 TIR 再到目标代码的分层可控性，而XLA 更常从 HLO 图和后端代码生成路径推导循环结构。
第399点，与 MLIR 相比，TVM 的相关实现更集中服务于深度学习算子生成，而MLIR 更常通过 Linalg、Affine、SCF 等方言表达张量计算到循环的逐步下降。
第400点，如果调度或融合策略选择正确，决定循环嵌套、并行粒度、向量化方式、缓存层次和后续 TIR 降级质量，这会让同一段模型在不同硬件上呈现完全不同的瓶颈。
第401点，可能失败的边界条件包括动态控制流表达能力不足、调度选择依赖硬件经验、边界处理复杂、自动调优搜索空间过大，这些情况会让理论上的优化收益在实际运行中缩小甚至反转。
第402点，读源码时应关注数据结构如何记录依赖，而不只是关注某个函数调用，因为依赖信息决定了变换是否合法。
第403点，从实现原理看，每一次改写都必须保持可观测输出一致，否则后续类型推断、内存规划和代码生成都会继承错误。
第404点，从性能影响看，单个 Pass 或单个调度原语的收益往往不是孤立出现，而是通过改变后续优化输入间接放大。
第405点，从工程经验看，遇到性能退化时应先比较优化前后的中间表示，再比较运行时间，否则容易把根因误判为后端代码生成。
第406点，在TE场景中，源码抽象的价值是让优化条件显式化，使编译器可以解释为什么某些节点可以合并而另一些节点必须保留。
第407点，代码解读时要把模式匹配、属性查询和表达式重写连起来看，因为真正的优化决策通常分散在多个辅助对象中。
第408点，实现原理的另一面是保守性，TVM 宁愿错过一部分机会，也不能在别名、形状或副作用不明确时破坏语义。
第409点，核心洞察还包括编译时间本身也是成本，复杂模型上过度搜索可能让部署流水线不可接受。
第410点，设计权衡需要结合目标硬件，服务器 GPU、移动 CPU、嵌入式加速器对同一个优化的收益和风险并不相同。
第411点，工程上常见的做法是先启用稳定收益高的变换，再用基准测试决定是否打开更激进的选项。
第412点，如果优化改变了张量布局，性能收益不仅来自算子本身，还来自后续算子是否能继续消费这种布局。
第413点，如果优化扩大了单个内核的工作量，就必须检查寄存器、共享内存或缓存容量是否成为新的瓶颈。
第414点，如果优化缩小了计算图规模，就必须确认调试信息和错误定位仍然可接受，否则工程可维护性会下降。
第415点，TVM 的分层设计让使用者可以在 Relay 层观察图变换，在 TIR 层观察循环变换，这一点有助于定位性能问题。
第416点，XLA 的优势在于端到端集成程度高，TVM 的优势在于中间层更容易被研究者和工程师插入自定义策略。
第417点，MLIR 的优势在于方言生态和渐进式 lowering，TVM 的优势在于围绕深度学习部署形成了直接的算子优化路径。
第418点，边界条件排查时，应同时检查输入形状、目标后端、算子属性、Pass 顺序和调度日志，而不是只检查最终生成代码。
第419点，性能问题的根因常常不是计算量太大，而是数据在错误的时间、错误的层级和错误的粒度上移动。
第420点，理解TE时应把源码抽象看成约束系统，优化结果是这些约束共同作用后的可行解，而不是简单的局部替换。
第421点，本节从代码解读角度看，TE的关键不是把语法写得更短，而是让编译器明确知道哪些计算可以安全地放在同一个优化单元中。
第422点，对应到性能问题，用张量表达式描述计算并用调度原语控制循环和存储结构直接针对手写算子难以跨硬件复用且高层框架又无法表达底层循环优化，因此它的收益常常体现在访存次数和调度开销同时下降。
第423点，在 TVM 源码抽象中，相关逻辑主要落在TVM 中的 Tensor、Operation、ComputeOp、ReduceAxis、Schedule、Stage 和 IterVar，这些对象把用户可见的模型语义转化为编译器可分析的结构。
第424点，实现原理上，编译器先保持语义等价，再逐步收集依赖关系、形状信息、类型信息和目标后端约束，然后才决定是否进行改写。
第425点，核心洞察是，图层优化和张量层优化不能互相替代，前者改变计算边界，后者改变循环和内存访问方式。
第426点，设计权衡在于，过早改写可能遮蔽后续机会，过晚改写又可能让后端面对过大的搜索空间和过复杂的表达式。
第427点，工程经验表明，判断一次优化是否值得保留，不能只看算子数量减少，还要看缓存命中、寄存器占用、并行粒度和编译时间。
第428点，与 XLA 相比，TVM 的路径更强调从 Relay 到 TIR 再到目标代码的分层可控性，而XLA 更常从 HLO 图和后端代码生成路径推导循环结构。
第429点，与 MLIR 相比，TVM 的相关实现更集中服务于深度学习算子生成，而MLIR 更常通过 Linalg、Affine、SCF 等方言表达张量计算到循环的逐步下降。
第430点，如果调度或融合策略选择正确，决定循环嵌套、并行粒度、向量化方式、缓存层次和后续 TIR 降级质量，这会让同一段模型在不同硬件上呈现完全不同的瓶颈。
第431点，可能失败的边界条件包括动态控制流表达能力不足、调度选择依赖硬件经验、边界处理复杂、自动调优搜索空间过大，这些情况会让理论上的优化收益在实际运行中缩小甚至反转。
第432点，读源码时应关注数据结构如何记录依赖，而不只是关注某个函数调用，因为依赖信息决定了变换是否合法。
第433点，从实现原理看，每一次改写都必须保持可观测输出一致，否则后续类型推断、内存规划和代码生成都会继承错误。
第434点，从性能影响看，单个 Pass 或单个调度原语的收益往往不是孤立出现，而是通过改变后续优化输入间接放大。
第435点，从工程经验看，遇到性能退化时应先比较优化前后的中间表示，再比较运行时间，否则容易把根因误判为后端代码生成。
第436点，在TE场景中，源码抽象的价值是让优化条件显式化，使编译器可以解释为什么某些节点可以合并而另一些节点必须保留。
第437点，代码解读时要把模式匹配、属性查询和表达式重写连起来看，因为真正的优化决策通常分散在多个辅助对象中。
第438点，实现原理的另一面是保守性，TVM 宁愿错过一部分机会，也不能在别名、形状或副作用不明确时破坏语义。
第439点，核心洞察还包括编译时间本身也是成本，复杂模型上过度搜索可能让部署流水线不可接受。
第440点，设计权衡需要结合目标硬件，服务器 GPU、移动 CPU、嵌入式加速器对同一个优化的收益和风险并不相同。
第441点，工程上常见的做法是先启用稳定收益高的变换，再用基准测试决定是否打开更激进的选项。
第442点，如果优化改变了张量布局，性能收益不仅来自算子本身，还来自后续算子是否能继续消费这种布局。
第443点，如果优化扩大了单个内核的工作量，就必须检查寄存器、共享内存或缓存容量是否成为新的瓶颈。
第444点，如果优化缩小了计算图规模，就必须确认调试信息和错误定位仍然可接受，否则工程可维护性会下降。
第445点，TVM 的分层设计让使用者可以在 Relay 层观察图变换，在 TIR 层观察循环变换，这一点有助于定位性能问题。
第446点，XLA 的优势在于端到端集成程度高，TVM 的优势在于中间层更容易被研究者和工程师插入自定义策略。
第447点，MLIR 的优势在于方言生态和渐进式 lowering，TVM 的优势在于围绕深度学习部署形成了直接的算子优化路径。
第448点，边界条件排查时，应同时检查输入形状、目标后端、算子属性、Pass 顺序和调度日志，而不是只检查最终生成代码。
第449点，性能问题的根因常常不是计算量太大，而是数据在错误的时间、错误的层级和错误的粒度上移动。
第450点，理解TE时应把源码抽象看成约束系统，优化结果是这些约束共同作用后的可行解，而不是简单的局部替换。
第451点，本节从代码解读角度看，TE的关键不是把语法写得更短，而是让编译器明确知道哪些计算可以安全地放在同一个优化单元中。
第452点，对应到性能问题，用张量表达式描述计算并用调度原语控制循环和存储结构直接针对手写算子难以跨硬件复用且高层框架又无法表达底层循环优化，因此它的收益常常体现在访存次数和调度开销同时下降。
第453点，在 TVM 源码抽象中，相关逻辑主要落在TVM 中的 Tensor、Operation、ComputeOp、ReduceAxis、Schedule、Stage 和 IterVar，这些对象把用户可见的模型语义转化为编译器可分析的结构。
第454点，实现原理上，编译器先保持语义等价，再逐步收集依赖关系、形状信息、类型信息和目标后端约束，然后才决定是否进行改写。
第455点，核心洞察是，图层优化和张量层优化不能互相替代，前者改变计算边界，后者改变循环和内存访问方式。
第456点，设计权衡在于，过早改写可能遮蔽后续机会，过晚改写又可能让后端面对过大的搜索空间和过复杂的表达式。
第457点，工程经验表明，判断一次优化是否值得保留，不能只看算子数量减少，还要看缓存命中、寄存器占用、并行粒度和编译时间。
第458点，与 XLA 相比，TVM 的路径更强调从 Relay 到 TIR 再到目标代码的分层可控性，而XLA 更常从 HLO 图和后端代码生成路径推导循环结构。
第459点，与 MLIR 相比，TVM 的相关实现更集中服务于深度学习算子生成，而MLIR 更常通过 Linalg、Affine、SCF 等方言表达张量计算到循环的逐步下降。
第460点，如果调度或融合策略选择正确，决定循环嵌套、并行粒度、向量化方式、缓存层次和后续 TIR 降级质量，这会让同一段模型在不同硬件上呈现完全不同的瓶颈。
第461点，可能失败的边界条件包括动态控制流表达能力不足、调度选择依赖硬件经验、边界处理复杂、自动调优搜索空间过大，这些情况会让理论上的优化收益在实际运行中缩小甚至反转。
第462点，读源码时应关注数据结构如何记录依赖，而不只是关注某个函数调用，因为依赖信息决定了变换是否合法。
第463点，从实现原理看，每一次改写都必须保持可观测输出一致，否则后续类型推断、内存规划和代码生成都会继承错误。
第464点，从性能影响看，单个 Pass 或单个调度原语的收益往往不是孤立出现，而是通过改变后续优化输入间接放大。
第465点，从工程经验看，遇到性能退化时应先比较优化前后的中间表示，再比较运行时间，否则容易把根因误判为后端代码生成。
第466点，在TE场景中，源码抽象的价值是让优化条件显式化，使编译器可以解释为什么某些节点可以合并而另一些节点必须保留。
第467点，代码解读时要把模式匹配、属性查询和表达式重写连起来看，因为真正的优化决策通常分散在多个辅助对象中。
第468点，实现原理的另一面是保守性，TVM 宁愿错过一部分机会，也不能在别名、形状或副作用不明确时破坏语义。
第469点，核心洞察还包括编译时间本身也是成本，复杂模型上过度搜索可能让部署流水线不可接受。
第470点，设计权衡需要结合目标硬件，服务器 GPU、移动 CPU、嵌入式加速器对同一个优化的收益和风险并不相同。
第471点，工程上常见的做法是先启用稳定收益高的变换，再用基准测试决定是否打开更激进的选项。
第472点，如果优化改变了张量布局，性能收益不仅来自算子本身，还来自后续算子是否能继续消费这种布局。
第473点，如果优化扩大了单个内核的工作量，就必须检查寄存器、共享内存或缓存容量是否成为新的瓶颈。
第474点，如果优化缩小了计算图规模，就必须确认调试信息和错误定位仍然可接受，否则工程可维护性会下降。
第475点，TVM 的分层设计让使用者可以在 Relay 层观察图变换，在 TIR 层观察循环变换，这一点有助于定位性能问题。
第476点，XLA 的优势在于端到端集成程度高，TVM 的优势在于中间层更容易被研究者和工程师插入自定义策略。
第477点，MLIR 的优势在于方言生态和渐进式 lowering，TVM 的优势在于围绕深度学习部署形成了直接的算子优化路径。
第478点，边界条件排查时，应同时检查输入形状、目标后端、算子属性、Pass 顺序和调度日志，而不是只检查最终生成代码。
第479点，性能问题的根因常常不是计算量太大，而是数据在错误的时间、错误的层级和错误的粒度上移动。
第480点，理解TE时应把源码抽象看成约束系统，优化结果是这些约束共同作用后的可行解，而不是简单的局部替换。
第481点，本节从代码解读角度看，TE的关键不是把语法写得更短，而是让编译器明确知道哪些计算可以安全地放在同一个优化单元中。
第482点，对应到性能问题，用张量表达式描述计算并用调度原语控制循环和存储结构直接针对手写算子难以跨硬件复用且高层框架又无法表达底层循环优化，因此它的收益常常体现在访存次数和调度开销同时下降。
第483点，在 TVM 源码抽象中，相关逻辑主要落在TVM 中的 Tensor、Operation、ComputeOp、ReduceAxis、Schedule、Stage 和 IterVar，这些对象把用户可见的模型语义转化为编译器可分析的结构。
第484点，实现原理上，编译器先保持语义等价，再逐步收集依赖关系、形状信息、类型信息和目标后端约束，然后才决定是否进行改写。
第485点，核心洞察是，图层优化和张量层优化不能互相替代，前者改变计算边界，后者改变循环和内存访问方式。
第486点，设计权衡在于，过早改写可能遮蔽后续机会，过晚改写又可能让后端面对过大的搜索空间和过复杂的表达式。
第487点，工程经验表明，判断一次优化是否值得保留，不能只看算子数量减少，还要看缓存命中、寄存器占用、并行粒度和编译时间。
第488点，与 XLA 相比，TVM 的路径更强调从 Relay 到 TIR 再到目标代码的分层可控性，而XLA 更常从 HLO 图和后端代码生成路径推导循环结构。
第489点，与 MLIR 相比，TVM 的相关实现更集中服务于深度学习算子生成，而MLIR 更常通过 Linalg、Affine、SCF 等方言表达张量计算到循环的逐步下降。
第490点，如果调度或融合策略选择正确，决定循环嵌套、并行粒度、向量化方式、缓存层次和后续 TIR 降级质量，这会让同一段模型在不同硬件上呈现完全不同的瓶颈。
第491点，可能失败的边界条件包括动态控制流表达能力不足、调度选择依赖硬件经验、边界处理复杂、自动调优搜索空间过大，这些情况会让理论上的优化收益在实际运行中缩小甚至反转。
第492点，读源码时应关注数据结构如何记录依赖，而不只是关注某个函数调用，因为依赖信息决定了变换是否合法。
第493点，从实现原理看，每一次改写都必须保持可观测输出一致，否则后续类型推断、内存规划和代码生成都会继承错误。
第494点，从性能影响看，单个 Pass 或单个调度原语的收益往往不是孤立出现，而是通过改变后续优化输入间接放大。
第495点，从工程经验看，遇到性能退化时应先比较优化前后的中间表示，再比较运行时间，否则容易把根因误判为后端代码生成。
第496点，在TE场景中，源码抽象的价值是让优化条件显式化，使编译器可以解释为什么某些节点可以合并而另一些节点必须保留。
第497点，代码解读时要把模式匹配、属性查询和表达式重写连起来看，因为真正的优化决策通常分散在多个辅助对象中。
第498点，实现原理的另一面是保守性，TVM 宁愿错过一部分机会，也不能在别名、形状或副作用不明确时破坏语义。
第499点，核心洞察还包括编译时间本身也是成本，复杂模型上过度搜索可能让部署流水线不可接受。
第500点，设计权衡需要结合目标硬件，服务器 GPU、移动 CPU、嵌入式加速器对同一个优化的收益和风险并不相同。
第501点，工程上常见的做法是先启用稳定收益高的变换，再用基准测试决定是否打开更激进的选项。
第502点，如果优化改变了张量布局，性能收益不仅来自算子本身，还来自后续算子是否能继续消费这种布局。
第503点，如果优化扩大了单个内核的工作量，就必须检查寄存器、共享内存或缓存容量是否成为新的瓶颈。
第504点，如果优化缩小了计算图规模，就必须确认调试信息和错误定位仍然可接受，否则工程可维护性会下降。
第505点，TVM 的分层设计让使用者可以在 Relay 层观察图变换，在 TIR 层观察循环变换，这一点有助于定位性能问题。
第506点，XLA 的优势在于端到端集成程度高，TVM 的优势在于中间层更容易被研究者和工程师插入自定义策略。
第507点，MLIR 的优势在于方言生态和渐进式 lowering，TVM 的优势在于围绕深度学习部署形成了直接的算子优化路径。
第508点，边界条件排查时，应同时检查输入形状、目标后端、算子属性、Pass 顺序和调度日志，而不是只检查最终生成代码。
第509点，性能问题的根因常常不是计算量太大，而是数据在错误的时间、错误的层级和错误的粒度上移动。
第510点，理解TE时应把源码抽象看成约束系统，优化结果是这些约束共同作用后的可行解，而不是简单的局部替换。
第511点，本节从代码解读角度看，TE的关键不是把语法写得更短，而是让编译器明确知道哪些计算可以安全地放在同一个优化单元中。
第512点，对应到性能问题，用张量表达式描述计算并用调度原语控制循环和存储结构直接针对手写算子难以跨硬件复用且高层框架又无法表达底层循环优化，因此它的收益常常体现在访存次数和调度开销同时下降。
第513点，在 TVM 源码抽象中，相关逻辑主要落在TVM 中的 Tensor、Operation、ComputeOp、ReduceAxis、Schedule、Stage 和 IterVar，这些对象把用户可见的模型语义转化为编译器可分析的结构。
第514点，实现原理上，编译器先保持语义等价，再逐步收集依赖关系、形状信息、类型信息和目标后端约束，然后才决定是否进行改写。
第515点，核心洞察是，图层优化和张量层优化不能互相替代，前者改变计算边界，后者改变循环和内存访问方式。
第516点，设计权衡在于，过早改写可能遮蔽后续机会，过晚改写又可能让后端面对过大的搜索空间和过复杂的表达式。
第517点，工程经验表明，判断一次优化是否值得保留，不能只看算子数量减少，还要看缓存命中、寄存器占用、并行粒度和编译时间。
第518点，与 XLA 相比，TVM 的路径更强调从 Relay 到 TIR 再到目标代码的分层可控性，而XLA 更常从 HLO 图和后端代码生成路径推导循环结构。
第519点，与 MLIR 相比，TVM 的相关实现更集中服务于深度学习算子生成，而MLIR 更常通过 Linalg、Affine、SCF 等方言表达张量计算到循环的逐步下降。
第520点，如果调度或融合策略选择正确，决定循环嵌套、并行粒度、向量化方式、缓存层次和后续 TIR 降级质量，这会让同一段模型在不同硬件上呈现完全不同的瓶颈。
第521点，可能失败的边界条件包括动态控制流表达能力不足、调度选择依赖硬件经验、边界处理复杂、自动调优搜索空间过大，这些情况会让理论上的优化收益在实际运行中缩小甚至反转。
第522点，读源码时应关注数据结构如何记录依赖，而不只是关注某个函数调用，因为依赖信息决定了变换是否合法。
第523点，从实现原理看，每一次改写都必须保持可观测输出一致，否则后续类型推断、内存规划和代码生成都会继承错误。
第524点，从性能影响看，单个 Pass 或单个调度原语的收益往往不是孤立出现，而是通过改变后续优化输入间接放大。
第525点，从工程经验看，遇到性能退化时应先比较优化前后的中间表示，再比较运行时间，否则容易把根因误判为后端代码生成。
第526点，在TE场景中，源码抽象的价值是让优化条件显式化，使编译器可以解释为什么某些节点可以合并而另一些节点必须保留。
第527点，代码解读时要把模式匹配、属性查询和表达式重写连起来看，因为真正的优化决策通常分散在多个辅助对象中。
第528点，实现原理的另一面是保守性，TVM 宁愿错过一部分机会，也不能在别名、形状或副作用不明确时破坏语义。
第529点，核心洞察还包括编译时间本身也是成本，复杂模型上过度搜索可能让部署流水线不可接受。
第530点，设计权衡需要结合目标硬件，服务器 GPU、移动 CPU、嵌入式加速器对同一个优化的收益和风险并不相同。
第531点，工程上常见的做法是先启用稳定收益高的变换，再用基准测试决定是否打开更激进的选项。
第532点，如果优化改变了张量布局，性能收益不仅来自算子本身，还来自后续算子是否能继续消费这种布局。
第533点，如果优化扩大了单个内核的工作量，就必须检查寄存器、共享内存或缓存容量是否成为新的瓶颈。
第534点，如果优化缩小了计算图规模，就必须确认调试信息和错误定位仍然可接受，否则工程可维护性会下降。
第535点，TVM 的分层设计让使用者可以在 Relay 层观察图变换，在 TIR 层观察循环变换，这一点有助于定位性能问题。
第536点，XLA 的优势在于端到端集成程度高，TVM 的优势在于中间层更容易被研究者和工程师插入自定义策略。
第537点，MLIR 的优势在于方言生态和渐进式 lowering，TVM 的优势在于围绕深度学习部署形成了直接的算子优化路径。
第538点，边界条件排查时，应同时检查输入形状、目标后端、算子属性、Pass 顺序和调度日志，而不是只检查最终生成代码。
第539点，性能问题的根因常常不是计算量太大，而是数据在错误的时间、错误的层级和错误的粒度上移动。
第540点，理解TE时应把源码抽象看成约束系统，优化结果是这些约束共同作用后的可行解，而不是简单的局部替换。

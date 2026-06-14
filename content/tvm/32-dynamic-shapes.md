> **学习目标**：
> - 理解深度学习中动态形状问题的本质与挑战
> - 掌握 TVM 中处理动态形状的多种技术方案
> - 理解 Relax 中符号形状（Symbolic Shape）的原生支持
> - 对比 Relay 与 Relax 在动态形状处理上的差异
> - 掌握 Shape Func 推断机制与动态 Kernel 生成策略

---

## 32.1 动态形状问题的本质

### 32.1.1 为什么需要动态形状？

在实际的深度学习部署场景中，输入张量的形状经常是变化的：

| 场景 | 动态维度 | 说明 |
|------|---------|------|
| **NLP 推理** | 序列长度 | 不同输入的 token 数量不同 |
| **目标检测** | 检测框数量 | NMS 后的输出数量不确定 |
| **图神经网络** | 节点/边数 | 图的拓扑结构因输入而异 |
| **批推理** | 批大小 | 在线服务中批大小动态变化 |
| **语音识别** | 音频长度 | 不同语音样本的时长不同 |

动态形状给编译器带来了根本性的挑战：**编译时未知的维度意味着无法确定循环边界、无法进行静态内存规划、无法预计算最优的调度策略**。

### 32.1.2 静态形状 vs 动态形状的编译差异

```
静态形状编译流程：
  模型定义 (batch=1, seq=128)
    ↓ 形状完全已知
  编译优化（循环展开、向量化、内存预分配）
    ↓ 生成固定形状的 kernel
  执行（零形状推理开销）

动态形状编译流程：
  模型定义 (batch=?, seq=?)
    ↓ 形状未知
  需要：符号推理 / 动态 kernel / 多版本编译
    ↓
  执行（运行时形状推理 + kernel 参数化）
```

### 32.1.3 动态形状处理的三个层次



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
┌─────────────────────────────────────────────────────┐
│  层次三：运行时代码生成（最灵活，最高开销）            │
│    在运行时根据实际形状 JIT 编译最优 kernel           │
├─────────────────────────────────────────────────────┤
│  层次二：参数化 kernel（平衡方案）                    │
│    编译时生成接受形状参数的 kernel                    │
├─────────────────────────────────────────────────────┤
│  层次一：多版本编译（最简单，空间换时间）              │
│    预编译一组固定形状的 kernel，运行时选择             │
└─────────────────────────────────────────────────────┘
```

---

## 32.2 Relay 中的动态形状支持

### 32.2.1 Relay 的 Any 类型

Relay 最初通过 `Any` 类型来处理动态形状：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import relay

# 使用 Any 表示动态维度
x = relay.var("x", relay.TensorType([relay.Any(), 784], "float32"))
w = relay.var("w", relay.TensorType([784, 256], "float32"))
y = relay.nn.dense(x, w)
```

`Any` 类型的定义在 `include/tvm/relay/type.h`：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
class AnyType : public Type {
 public:
  // AnyType 是一个单例，表示任意维度
  static AnyType make();
};
```

### 32.2.2 Any 类型的局限

`Any` 方案存在根本性缺陷：

| 问题 | 描述 |
|------|------|
| **形状信息丢失** | `Any` 只表示"不知道"，无法推理两个 `Any` 维度是否相同 |
| **约束无法表达** | 无法表达 `dim_a == dim_b` 或 `dim_a = 2 * dim_b` 这样的约束 |
| **优化受限** | 编译器无法对未知维度做任何优化 |
| **运行时开销** | 每次执行都需要重新推断形状 |

### 32.2.3 Relay 的 PartialEval 与形状推断

Relay 通过 `PartialEval` Pass 尝试在编译时推断出尽可能多的形状信息：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm.relay.transform import PartialEval

# PartialEval 尝试将已知信息传播到下游
mod = PartialEval()(mod)
```

但这种方法对真正动态的情况效果有限。

---

## 32.3 Relax 中的符号形状（Symbolic Shape）

### 32.3.1 符号变量的设计

Relax 使用 `tir.Var` 作为符号维度变量，这是对 `Any` 的根本性改进：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import tir, relax

# 创建符号变量
batch = tir.Var("batch", "int64")
seq_len = tir.Var("seq_len", "int64")

# 使用符号变量定义形状
x = relax.Var("x", relax.TensorStructInfo((batch, seq_len, 768), "float32"))
w = relax.Var("w", relax.TensorStructInfo((768, 3072), "float32"))
```

符号变量的关键优势：

| 特性 | Any | Symbolic Var |
|------|-----|-------------|
| 约束推理 | 不支持 | 支持（`batch > 0`） |
| 相同性推理 | 不支持 | 同一变量 = 同一维度 |
| 表达式计算 | 不支持 | 支持（`2 * seq_len`） |
| 优化空间 | 无 | 可基于符号关系优化 |

### 32.3.2 形状表达式（ShapeExpr）

Relax 的 `ShapeExpr` 可以包含符号变量的算术表达式：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 创建形状表达式
m = tir.Var("m", "int64")
n = tir.Var("n", "int64")

# 形状可以是符号表达式
shape = relax.ShapeExpr([m, n])
shape_doubled = relax.ShapeExpr([2 * m, n])

# 在 StructInfo 中使用
sinfo = relax.TensorStructInfo(shape, "float32")
```

`ShapeExpr` 定义在 `include/tvm/relax/expr.h`：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
class ShapeExpr : public RelaxExpr {
 public:
  /*! \brief The values of the shape expression */
  Array<PrimExpr> values;
};
```

### 32.3.3 符号形状的约束传播

当两个符号变量参与同一运算时，Relax 的 StructInfo 推断会自动传播约束：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 如果 x 的形状是 (batch, seq_len, 768)
# 且 w 的形状是 (768, 3072)
# 则 matmul(x, w) 的输出形状自动推断为 (batch, seq_len, 3072)

@R.function
def transformer_ffn(x: R.Tensor((batch, seq_len, 768), "float32"),
                    w1: R.Tensor((768, 3072), "float32"),
                    b1: R.Tensor((3072,), "float32")):
    with R.dataflow():
        # 输出形状自动推断为 (batch, seq_len, 3072)
        lv0 = R.matmul(x, w1)
        lv1 = R.add(lv0, b1)
        R.output(lv1)
    return lv1
```

### 32.3.4 符号形状的算术推理

Relax 可以推理涉及符号变量的算术关系：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 示例：自注意力中的 head 维度推理
num_heads = tir.Var("num_heads", "int64")
head_dim = tir.Var("head_dim", "int64")
hidden_size = num_heads * head_dim  # 隐含约束

@R.function
def attention(x: R.Tensor((batch, seq_len, hidden_size), "float32")):
    # reshape 到 (batch, seq_len, num_heads, head_dim)
    # Relax 知道 hidden_size == num_heads * head_dim，所以 reshape 合法
    reshaped = R.reshape(x, (batch, seq_len, num_heads, head_dim))
    ...
```

---

## 32.4 Dynamic TIR

### 32.4.1 什么是 Dynamic TIR？

当 TIR 函数的循环边界或 Buffer 形状包含符号变量时，称为 Dynamic TIR。这是处理动态形状的核心技术：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import tir, script

@T.prim_func
def dynamic_matmul(
    A: T.Buffer((T.Var("m"), T.Var("k")), "float32"),
    B: T.Buffer((T.Var("k"), T.Var("n")), "float32"),
    C: T.Buffer((T.Var("m"), T.Var("n")), "float32"),
    m: T.int64, k: T.int64, n: T.int64
):
    # 循环边界是符号变量
    for i, j, kk in T.grid(m, n, k):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, kk])
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
```

### 32.4.2 Dynamic TIR 的实现挑战

Dynamic TIR 面临的核心问题是：**如何生成高效的代码来处理未知的循环边界？**

| 挑战 | 说明 | 解决方案 |
|------|------|---------|
| **向量化** | 未知边界无法确定 SIMD 宽度 | 尾部处理循环 + 条件向量化 |
| **内存分配** | Buffer 大小在编译时未知 | 运行时分配 + 形状参数传递 |
| **循环优化** | 分块大小需要适配运行时形状 | 参数化分块 + 自适应策略 |
| **并行度** | 未知形状影响线程划分 | 运行时动态调度 |

### 32.4.3 Loop Partitioning for Dynamic Shapes

TVM 的 `LoopPartition` Pass（`src/tir/transforms/loop_partition.cc`）会将包含符号边界的循环分割为可优化的静态部分和尾部处理部分：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 原始的动态循环
for i in range(n):  # n 是符号变量
    C[i] = A[i] + B[i]

# LoopPartition 后（假设向量化宽度为 4）
for i in range(0, n // 4 * 4, 4):
    C[i:i+4] = A[i:i+4] + B[i:i+4]  # 向量化部分
for i in range(n // 4 * 4, n):  # 尾部处理
    C[i] = A[i] + B[i]
```

### 32.4.4 Dynamic TIR 的 Buffer 管理

在 Dynamic TIR 中，Buffer 的形状在编译时未知，需要运行时传入：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
@T.prim_func
def dynamic_relu(
    data: T.handle,
    output: T.handle,
    n: T.int64  # 形状参数
):
    # 通过 T.handle 接受运行时 Buffer
    A = T.match_buffer(data, (n,), "float32")
    B = T.match_buffer(output, (n,), "float32")

    for i in T.serial(n):
        with T.block("B"):
            vi = T.axis.spatial(n, i)
            B[vi] = T.max(A[vi], T.float32(0))
```

---

## 32.5 Shape Func：形状推断函数

### 32.5.1 Shape Func 的设计

在动态形状场景下，算子的输出形状需要在运行时根据输入形状计算。TVM 使用 **Shape Func** 来描述这种推断关系：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# Shape Func 描述了输出形状如何依赖输入形状
@T.prim_func
def conv2d_shape_func(
    data_shape: T.Buffer((4,), "int64"),    # 输入形状 [N, C, H, W]
    weight_shape: T.Buffer((4,), "int64"),   # 权重形状 [O, C, KH, KW]
    output_shape: T.Buffer((4,), "int64"),   # 输出形状 [N, O, OH, OW]
):
    # 计算输出形状
    N = data_shape[0]
    O = weight_shape[0]
    H = data_shape[2]
    W = data_shape[3]
    KH = weight_shape[2]
    KW = weight_shape[3]

    output_shape[0] = N
    output_shape[1] = O
    output_shape[2] = (H - KH) // 1 + 1  # stride=1, padding=0
    output_shape[3] = (W - KW) // 1 + 1
```

### 32.5.2 Shape Func 的注册

每个 Relax 算子可以注册一个对应的 Shape Func：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// src/relax/op/tensor/binary.cc 中的形状推断
StructInfo InferStructInfoBinaryBroadcast(const Call& call,
                                          const BlockBuilder& ctx) {
  TensorStructInfo lhs = GetStructInfoAs<TensorStructInfo>(call->args[0]);
  TensorStructInfo rhs = GetStructInfoAs<TensorStructInfo>(call->args[1]);

  // 使用 broadcast 规则推断输出形状
  ShapeExpr out_shape = BroadcastShape(
      lhs->shape.value(), rhs->shape.value());

  return TensorStructInfo(out_shape, lhs->dtype);
}
```

### 32.5.3 运行时形状推断

当编译时无法确定形状时，Shape Func 在运行时执行：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# Relax VM 的执行流程：
# 1. 执行 Shape Func 计算输出形状
# 2. 使用输出形状分配内存
# 3. 执行实际的计算 kernel

# 伪代码
def execute_call_tir_dynamic(tir_func, inputs, shape_func):
    # 第一步：计算输出形状
    output_shape = shape_func(*[inp.shape for inp in inputs])

    # 第二步：分配输出内存
    output = alloc_tensor(output_shape, dtype)

    # 第三步：执行 kernel
    tir_func(*inputs, output)

    return output
```

### 32.5.4 Shape Expr 与 Symbolic Constraint

Relax 可以在 StructInfo 中嵌入符号约束：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 定义一个带约束的函数
@R.function
def constrained_matmul(
    x: R.Tensor((m, k), "float32"),
    w: R.Tensor((k, n), "float32"),
):
    # 约束：k 维度必须匹配
    # Relax 的 StructInfo 推断会自动验证
    with R.dataflow():
        lv0 = R.matmul(x, w)  # 输出：(m, n)
        R.output(lv0)
    return lv0
```

---

## 32.6 动态形状的调度策略

### 32.6.1 挑战：调度空间的变化

对于静态形状，MetaSchedule 可以离线搜索最优调度。但动态形状下，不同的运行时形状可能需要不同的最优调度：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
形状 (1, 128) → 最优分块: tile=1x128
形状 (32, 128) → 最优分块: tile=32x128
形状 (1024, 128) → 最优分块: tile=64x128

问题：应该为哪种形状优化？
```

### 32.6.2 符号形状的调度

TVM 支持使用符号变量进行调度，生成参数化的 kernel：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import meta_schedule

# 使用符号形状进行调优
m = tir.Var("m", "int64")
n = tir.Var("n", "int64")

# MetaSchedule 可以为符号形状搜索调度
# 生成的 kernel 对所有形状都有效
sch = meta_schedule.tune(
    func=matmul_func,
    target="llvm",
    space=meta_schedule.SpaceGenerator(
        # 调度空间可以包含符号变量的约束
    ),
)
```

### 32.6.3 分段策略（Piecewise Strategy）

对于形状范围差异很大的情况，可以使用分段策略：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 定义形状范围
shape_ranges = {
    "small": (1, 32),       # batch 1-32
    "medium": (32, 256),    # batch 32-256
    "large": (256, 1024),   # batch 256-1024
}

# 为每个范围编译一个优化的 kernel
kernels = {}
for name, (lo, hi) in shape_ranges.items():
    # 使用范围内的代表形状进行调优
    representative_shape = (lo + hi) // 2
    kernels[name] = compile_kernel(representative_shape)

# 运行时根据实际形状选择 kernel
def select_kernel(batch_size):
    if batch_size <= 32:
        return kernels["small"]
    elif batch_size <= 256:
        return kernels["medium"]
    else:
        return kernels["large"]
```

### 32.6.4 Auto-tuning with Dynamic Shapes

MetaSchedule 提供了对动态形状的支持：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm.meta_schedule import TuneContext, SpaceGenerator
from tvm.meta_schedule.search_strategy import EvolutionarySearch

# 为动态形状定义调优上下文
ctx = TuneContext(
    mod=mod,
    target=target,
    space_generator=SpaceGenerator.PostOrderApply(),
    search_strategy=EvolutionarySearch(
        # 可以指定形状采样范围
        population_size=128,
        num_trials_per_iter=64,
    ),
)

# 调优时采样多种形状
for trial in range(1000):
    # 从形状范围中随机采样
    batch_size = np.random.randint(1, 128)
    seq_len = np.random.randint(16, 512)

    # 评估当前调度在采样形状上的性能
    sch = ctx.current_schedule
    latency = evaluate(sch, batch_size, seq_len)
    ctx.record(latency)
```

---

## 32.7 动态形状的运行时处理

### 32.7.1 Relax VM 的动态形状支持

Relax VM 原生支持动态形状的执行：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import tvm
from tvm import relax

# 编译支持动态形状的模型
@tvm.script.ir_module
class DynamicModel:
    @R.function
    def main(x: R.Tensor((batch, 784), "float32"),
             w: R.Tensor((784, 256), "float32")):
        with R.dataflow():
            lv0 = R.matmul(x, w)
            R.output(lv0)
        return lv0

# 编译（不需要指定 batch 大小）
ex = relax.vm.build(DynamicModel, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

# 运行时传入不同的 batch 大小
x1 = tvm.nd.array(np.random.randn(1, 784).astype("float32"))
x32 = tvm.nd.array(np.random.randn(32, 784).astype("float32"))
x128 = tvm.nd.array(np.random.randn(128, 784).astype("float32"))

out1 = vm["main"](x1)      # batch=1
out32 = vm["main"](x32)    # batch=32
out128 = vm["main"](x128)  # batch=128
```

### 32.7.2 Shape Memoization

为了避免重复的形状推断，Relax VM 使用 Shape Memoization 缓存：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// src/relax/backend/vm/vm.cc
class VirtualMachine {
 private:
  // 缓存已计算的形状
  std::unordered_map<ShapeKey, ShapeValue> shape_cache_;

  // 对于相同的输入形状组合，直接返回缓存的结果
  ShapeValue ComputeShape(const ShapeKey& key) {
    auto it = shape_cache_.find(key);
    if (it != shape_cache_.end()) {
      return it->second;
    }
    // 计算新形状并缓存
    ShapeValue value = shape_func_(key);
    shape_cache_[key] = value;
    return value;
  }
};
```

### 32.7.3 内存管理与动态形状

动态形状下的内存管理更加复杂：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# Relax VM 的内存分配策略：

# 策略一：按需分配（简单但可能碎片化）
output = alloc_tensor(shape, dtype)

# 策略二：内存池（减少分配开销）
# VM 维护一个内存池，对相同 dtype 和相似大小的请求复用内存
pool = MemoryPool()
output = pool.alloc(shape, dtype)

# 策略三：预分配（如果有形状范围信息）
max_shape = (max_batch, max_seq_len, hidden_size)
pre_allocated = alloc_tensor(max_shape, dtype)
# 实际使用时只访问有效区域
output = pre_allocated[:batch, :seq_len, :]
```

---

## 32.8 动态形状的优化技术

### 32.8.1 Symbolic Loop Analysis

TVM 的 TIR 变换 Pass 需要能够处理符号变量的算术推理：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# src/tir/transforms/simplify.cc 中的符号推理

# 示例：符号常量折叠
# 如果已知 k > 0，则 (k + 3) // 4 * 4 >= k 成立
# 可以安全地将循环上界从 k 提升到 (k + 3) // 4 * 4

@T.prim_func
def simplified_loop(
    A: T.Buffer((T.Var("n"),), "float32"),
    n: T.int64
):
    # 原始循环
    # for i in range(n):
    #     A[i] = A[i] + 1

    # 简化后（向量化）
    n_aligned = (n + 3) // 4 * 4
    for i in range(0, n_aligned, 4):
        if i + 3 < n:
            # 完整的向量操作
            A[i:i+4] = A[i:i+4] + T.broadcast(T.float32(1), 4)
        else:
            # 尾部处理
            for j in range(i, n):
                A[j] = A[j] + 1
```

### 32.8.2 条件执行优化

对于动态形状，某些代码路径可能不会被使用：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 条件编译：根据形状范围选择不同的实现
@T.prim_func
def adaptive_softmax(
    logits: T.handle,
    output: T.handle,
    batch: T.int64,
    vocab: T.int64
):
    if batch <= 32:
        # 小 batch：使用朴素实现
        naive_softmax(logits, output, batch, vocab)
    else:
        # 大 batch：使用优化实现
        optimized_softmax(logits, output, batch, vocab)
```

### 32.8.3 算子融合与动态形状

动态形状对算子融合提出了特殊挑战：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 静态形状下的融合
# relu(add(matmul(x, w), b)) → fused_kernel(x, w, b)

# 动态形状下的融合需要考虑：
# 1. 融合后的 kernel 必须接受形状参数
# 2. 中间 Buffer 的形状需要动态计算
# 3. 融合可能降低（因为形状参数的传递开销）

# Relax 的 FuseOpsByPattern 会自动处理这些情况
```

---

## 32.9 Relay vs Relax 的动态形状支持对比

### 32.9.1 架构层面的差异

| 特性 | Relay | Relax |
|------|-------|-------|
| **类型系统** | TensorType + Any | TensorStructInfo + Symbolic Var |
| **形状推理** | 独立的 ShapeFunc Pass | 集成在 StructInfo 推断中 |
| **符号约束** | 不支持 | 原生支持符号变量的算术推理 |
| **动态 TIR** | 需要特殊处理路径 | 一等公民支持 |
| **VM 支持** | 需要 ShapeFunc 编译 | 原生支持符号形状执行 |

### 32.9.2 同一模型的对比



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# ========== Relay 版本 ==========
import tvm.relay as relay

x = relay.var("x", relay.TensorType([relay.Any(), 784], "float32"))
w = relay.var("w", relay.TensorType([784, 256], "float32"))
y = relay.nn.dense(x, w)
# 问题：Any 无法推理 batch 的具体关系

# ========== Relax 版本 ==========
import tvm.relax as relax

batch = tir.Var("batch", "int64")

@R.function
def main(x: R.Tensor((batch, 784), "float32"),
         w: R.Tensor((784, 256), "float32")):
    with R.dataflow():
        # batch 变量可以传递到下游
        lv0 = R.matmul(x, w)  # 输出：(batch, 256)
        R.output(lv0)
    return lv0
# 优势：batch 变量在整个计算图中保持一致
```

### 32.9.3 编译流程的差异



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
Relay 的动态形状编译：
  模型（Any 类型）
    ↓ 需要额外的 ShapeFunc 推断 Pass
  ShapeFunc + 静态 kernel
    ↓ ShapeFunc 需要单独编译
  执行：先执行 ShapeFunc，再执行 kernel

Relax 的动态形状编译：
  模型（Symbolic Var）
    ↓ StructInfo 推断自动处理符号形状
  包含符号参数的 kernel
    ↓ 直接编译
  执行：kernel 接受形状参数，一次调用完成
```

---

## 32.10 实战：Transformer 的动态形状处理

### 32.10.1 问题定义

Transformer 模型中，序列长度是动态的。以 BERT 为例：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# BERT 的输入形状：
# input_ids: (batch, seq_len) - seq_len 是动态的
# attention_mask: (batch, seq_len) - 同上
# token_type_ids: (batch, seq_len) - 同上
```

### 32.10.2 Relax 中的 Transformer 实现



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import tir, relax
import tvm.script

batch = tir.Var("batch", "int64")
seq_len = tir.Var("seq_len", "int64")
hidden_size = 768
num_heads = 12
head_dim = hidden_size // num_heads

@tvm.script.ir_module
class BERT:
    @R.function
    def attention(
        x: R.Tensor((batch, seq_len, hidden_size), "float32"),
        w_q: R.Tensor((hidden_size, hidden_size), "float32"),
        w_k: R.Tensor((hidden_size, hidden_size), "float32"),
        w_v: R.Tensor((hidden_size, hidden_size), "float32"),
    ):
        with R.dataflow():
            # Q, K, V 投影
            Q = R.matmul(x, w_q)  # (batch, seq_len, hidden_size)
            K = R.matmul(x, w_k)
            V = R.matmul(x, w_v)

            # reshape 到多头
            Q = R.reshape(Q, (batch, seq_len, num_heads, head_dim))
            K = R.reshape(K, (batch, seq_len, num_heads, head_dim))
            V = R.reshape(V, (batch, seq_len, num_heads, head_dim))

            # transpose 到 (batch, num_heads, seq_len, head_dim)
            Q = R.permute_dims(Q, (0, 2, 1, 3))
            K = R.permute_dims(K, (0, 2, 1, 3))
            V = R.permute_dims(V, (0, 2, 1, 3))

            # 注意力分数
            scores = R.matmul(Q, R.permute_dims(K, (0, 1, 3, 2)))
            scores = R.divide(scores, R.const(float(head_dim ** 0.5)))

            attn = R.nn.softmax(scores, axis=-1)

            # 加权求和
            output = R.matmul(attn, V)

            # reshape 回原始形状
            output = R.permute_dims(output, (0, 2, 1, 3))
            output = R.reshape(output, (batch, seq_len, hidden_size))

            R.output(output)
        return output
```

### 32.10.3 编译与执行



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 编译（batch 和 seq_len 都是符号的）
ex = relax.vm.build(BERT, target="cuda")
vm = relax.VirtualMachine(ex, tvm.device("cuda", 0))

# 运行时传入不同的形状
for seq in [16, 32, 64, 128, 256]:
    x = tvm.nd.array(np.random.randn(1, seq, 768).astype("float32"))
    w_q = tvm.nd.array(np.random.randn(768, 768).astype("float32"))
    w_k = tvm.nd.array(np.random.randn(768, 768).astype("float32"))
    w_v = tvm.nd.array(np.random.randn(768, 768).astype("float32"))

    output = vm["attention"](x, w_q, w_k, w_v)
    print(f"seq_len={seq}, output shape={output.shape}")
```

---

## 32.11 高级话题：动态形状的 Profiling

### 32.11.1 形状敏感的性能分析

不同形状的执行时间可能差异很大，需要形状敏感的 profiling：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import runtime

# 使用 TVM Profiler 进行形状敏感的分析
with tvm.runtime.Profiler() as prof:
    for batch_size in [1, 4, 16, 64, 256]:
        for seq_len in [32, 64, 128, 256, 512]:
            x = create_input(batch_size, seq_len)
            output = vm["main"](x)
            prof.record(f"b{batch_size}_s{seq_len}")

# 分析不同形状的性能特征
print(prof.table())
```

### 32.11.2 形状对 Kernel 性能的影响



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 性能分析示例
# 不同形状下，最优的分块策略不同

# 小形状：更小的 tile 以减少开销
#   tile_m=32, tile_n=32
# 中等形状：标准 tile
#   tile_m=64, tile_n=64
# 大形状：更大的 tile 以提高利用率
#   tile_m=128, tile_n=128

# 符号调度需要选择对所有形状都表现良好的 tile
# 这通常意味着选择一个"安全"的中间值
```

---

## 32.12 形状约束求解

### 32.12.1 符号约束系统

Relax 的 StructInfo 推断过程中，需要求解涉及符号变量的约束方程。约束求解器定义在 `src/relax/analysis/struct_info_analysis.cc`：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// 约束示例：
// 约束1: out_shape[0] = batch
// 约束2: out_shape[1] = seq_len
// 约束3: out_shape[2] = hidden_size
// 约束4: hidden_size == num_heads * head_dim

class ConstraintSolver {
 public:
  // 添加相等约束
  void AddEqualConstraint(PrimExpr lhs, PrimExpr rhs);
  // 添加不等式约束
  void AddInequalityConstraint(PrimExpr expr, PrimExpr bound, bool is_upper);
  // 检查约束是否可满足
  bool IsSatisfiable();
  // 获取变量的取值范围
  Range GetVariableRange(tir::Var var);
};
```

### 32.12.2 形状相等推理

当两个张量在同一维度上使用同一符号变量时，Relax 自动推断它们的形状约束：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 示例：自注意力中的形状推理
batch = tir.Var("batch", "int64")
seq_len = tir.Var("seq_len", "int64")
num_heads = tir.Var("num_heads", "int64")
head_dim = tir.Var("head_dim", "int64")
hidden = num_heads * head_dim

@R.function
def self_attention(
    Q: R.Tensor((batch, seq_len, hidden), "float32"),
    K: R.Tensor((batch, seq_len, hidden), "float32"),
    V: R.Tensor((batch, seq_len, hidden), "float32"),
):
    with R.dataflow():
        # reshape: hidden -> num_heads * head_dim
        # Relax 知道 hidden == num_heads * head_dim，所以 reshape 合法
        Qh = R.reshape(Q, (batch, seq_len, num_heads, head_dim))
        Kh = R.reshape(K, (batch, seq_len, num_heads, head_dim))
        Vh = R.reshape(V, (batch, seq_len, num_heads, head_dim))

        # 转置: (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        Qt = R.permute_dims(Qh, (0, 2, 1, 3))
        Kt = R.permute_dims(Kh, (0, 2, 1, 3))
        Vt = R.permute_dims(Vh, (0, 2, 1, 3))

        # 注意力分数: (batch, num_heads, seq_len, seq_len)
        # Relax 推断 Kt 的转置维度为 (batch, num_heads, head_dim, seq_len)
        scores = R.matmul(Qt, R.permute_dims(Kt, (0, 1, 3, 2)))

        # Softmax
        attn = R.nn.softmax(scores, axis=-1)

        # 加权求和
        output = R.matmul(attn, Vt)  # (batch, num_heads, seq_len, head_dim)

        # 转置回来
        output = R.permute_dims(output, (0, 2, 1, 3))  # (batch, seq_len, num_heads, head_dim)
        output = R.reshape(output, (batch, seq_len, hidden))  # (batch, seq_len, hidden)

        R.output(output)
    return output
```

### 32.12.3 隐式约束传播

某些约束不需要显式声明，Relax 可以自动推断：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 约束传播示例
m = tir.Var("m", "int64")
n = tir.Var("n", "int64")
k = tir.Var("k", "int64")

# 函数 1
@R.function
def func1(x: R.Tensor((m, k), "float32"),
          w: R.Tensor((k, n), "float32")):
    # 输出形状自动推断为 (m, n)
    lv0 = R.matmul(x, w)
    return lv0

# 函数 2
@R.function
def func2(x: R.Tensor((m, k), "float32"),
          w: R.Tensor((k, n), "float32")):
    lv0 = func1(x, w)  # 调用 func1，输出形状为 (m, n)
    # lv0 的形状已知为 (m, n)
    lv1 = R.add(lv0, R.const(np.ones((1, n)).astype("float32")))
    # broadcast 语义：(m, n) + (1, n) -> (m, n)
    return lv1
```

### 32.12.4 约束冲突检测

当约束不可满足时，Relax 会报告清晰的错误信息：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 约束冲突示例
@R.function
def conflicting_shapes(
    x: R.Tensor((m, 64), "float32"),
    w: R.Tensor((128, n), "float32"),
):
    # 错误：matmul 要求 x 的最后一维等于 w 的第一维
    # 即 64 == 128，这是不可满足的
    lv0 = R.matmul(x, w)
    return lv0

# Relax 报错：
# "StructInfoError: Cannot unify shape dimension 64 and 128.
#  In function main, at matmul operation."
```

---

## 32.13 动态形状下的 TIR 变换

### 32.13.1 LoopPartition 与符号变量

TVM 的 `LoopPartition` Pass 在处理符号边界时需要特殊策略：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# src/tir/transforms/loop_partition.cc

# 原始循环（符号边界）
for i in range(n):  # n 是符号变量
    A[i] = B[i] + 1

# LoopPartition 的策略：
# 1. 检测 n 是否可以被向量化宽度整除
# 2. 如果不能，分割为可优化部分和尾部

# 结果（假设向量化宽度 = 4）：
# 主循环：n // 4 次迭代，每次处理 4 个元素
for i in range(0, n // 4 * 4, 4):
    A[i:i+4] = B[i:i+4] + 1  # 向量化

# 尾部循环：处理剩余的 n % 4 个元素
for i in range(n // 4 * 4, n):
    A[i] = B[i] + 1
```

### 32.13.2 StorageRewrite 与动态形状

`StorageRewrite` Pass 在动态形状下需要保守处理：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 静态形状下的存储重用
# 如果两个张量的生命周期不重叠，可以共享内存
A = alloc_tensor((1024,))  # 生命周期: [0, 100]
B = alloc_tensor((1024,))  # 生命周期: [200, 300]
# A 和 B 可以共享同一块内存

# 动态形状下的存储重用
# 由于形状未知，无法确定内存大小
A = alloc_tensor((n,))  # n 是符号变量
B = alloc_tensor((n,))
# 如果生命周期不重叠，仍然可以共享
# 但需要运行时确认 n 的值相同
```

### 32.13.3 VectorizePass 与符号变量

向量化在动态形状下面临挑战：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 静态形状的向量化
for i in range(1024):
    A[i] = B[i] + 1
# 可以直接向量化为：
for i in range(0, 1024, 4):
    A[i:i+4] = B[i:i+4] + 1

# 动态形状的向量化
for i in range(n):
    A[i] = B[i] + 1
# 需要生成条件判断：
for i in range(0, (n + 3) // 4 * 4, 4):
    if i + 3 < n:
        # 完整的向量操作
        A[i:i+4] = B[i:i+4] + 1
    else:
        # 标量处理尾部
        for j in range(i, n):
            A[j] = B[j] + 1
```

### 32.13.4 ThreadSync 与动态形状

GPU 线程同步在动态形状下需要注意：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 动态形状下的共享内存同步
@T.prim_func
def dynamic_kernel(
    A: T.handle, B: T.handle, n: T.int64
):
    A_buf = T.match_buffer(A, (n,), "float32")
    B_buf = T.match_buffer(B, (n,), "float32")

    # 共享内存分配（大小未知）
    shared = T.alloc_shared((T.min(n, 256),), "float32")

    # 需要确保所有线程都完成写入后再读取
    for i in T.thread_binding(0, T.min(n, 256), "threadIdx.x"):
        shared[i] = A_buf[i]
    T.tvm_storage_sync("shared")  # 同步屏障

    for i in T.thread_binding(0, T.min(n, 256), "threadIdx.x"):
        B_buf[i] = shared[i] * 2
```

---

## 32.14 动态形状的实际部署案例

### 32.14.1 NLP 模型的动态序列长度

以 BERT 为例，展示完整的动态形状处理流程：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import tvm
from tvm import relax, tir
from tvm.relax.frontend import from_onnx
import onnx

# 1. 加载 BERT 模型（ONNX 格式）
model = onnx.load("bert-base-uncased.onnx")

# 2. 定义动态形状
shape_dict = {
    "input_ids": ["batch", "seq_len"],
    "attention_mask": ["batch", "seq_len"],
    "token_type_ids": ["batch", "seq_len"],
}

# 3. 导入到 Relax
mod = from_onnx(model, shape_dict=shape_dict)

# 4. 查看生成的符号形状
print(mod["main"])
# 输出中会包含 batch 和 seq_len 作为符号变量

# 5. 编译
target = tvm.target.Target("cuda")
ex = relax.vm.build(mod, target)
vm = relax.VirtualMachine(ex, tvm.device("cuda", 0))

# 6. 运行不同序列长度的推理
for seq_len in [32, 64, 128, 256, 512]:
    input_ids = tvm.nd.array(
        np.random.randint(0, 30000, (1, seq_len)).astype("int64"),
        tvm.device("cuda", 0)
    )
    attention_mask = tvm.nd.array(
        np.ones((1, seq_len)).astype("int64"),
        tvm.device("cuda", 0)
    )
    token_type_ids = tvm.nd.array(
        np.zeros((1, seq_len)).astype("int64"),
        tvm.device("cuda", 0)
    )

    output = vm["main"](input_ids, attention_mask, token_type_ids)
    print(f"seq_len={seq_len}, output shape={output.shape}")
```

### 32.14.2 目标检测中的动态框数量

NMS（非极大值抑制）后的检测框数量是动态的：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 目标检测模型的动态输出
# 输入：固定形状 (1, 3, 640, 640)
# 输出：动态形状 (num_detections, 6)
#   其中 num_detections 在每次推理时不同

@tvm.script.ir_module
class DetectionModel:
    @R.function
    def main(
        image: R.Tensor((1, 3, 640, 640), "float32"),
        boxes: R.Tensor((num_proposals, 4), "float32"),
        scores: R.Tensor((num_proposals,), "float32"),
    ):
        with R.dataflow():
            # Backbone 特征提取（静态形状）
            features = R.call_packed("backbone", image,
                                     sinfo_args=R.TensorStructInfo((1, 256, 20, 20), "float32"))

            # 检测头（静态形状）
            det_boxes = R.call_packed("detection_head", features,
                                       sinfo_args=R.TensorStructInfo((1, 8400, 4), "float32"))
            det_scores = R.call_packed("score_head", features,
                                        sinfo_args=R.TensorStructInfo((1, 8400, 80), "float32"))

            # NMS（动态输出）
            # num_detections 依赖于输入数据
            final_boxes, final_scores, num_detections = R.call_packed(
                "nms", det_boxes, det_scores, 0.5, 100,
                sinfo_args=[
                    R.TensorStructInfo(("max_det", 4), "float32"),
                    R.TensorStructInfo(("max_det",), "float32"),
                    R.TensorStructInfo((), "int64"),
                ]
            )

            R.output(final_boxes, final_scores, num_detections)
        return final_boxes, final_scores, num_detections
```

### 32.14.3 图神经网络的动态拓扑

图神经网络中，图的拓扑结构因输入而异：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 图卷积的动态形状处理
# 节点特征：(num_nodes, feature_dim) - num_nodes 动态
# 邻接矩阵：(num_nodes, num_nodes) - 依赖 num_nodes

@R.function
def gcn_layer(
    node_features: R.Tensor(("N", "F"), "float32"),
    adj_matrix: R.Tensor(("N", "N"), "float32"),
    weight: R.Tensor(("F", "F_out"), "float32"),
):
    with R.dataflow():
        # 特征变换
        h = R.matmul(node_features, weight)  # (N, F_out)

        # 邻居聚合：A * H
        # adj_matrix: (N, N), h: (N, F_out)
        # 输出: (N, F_out)
        aggregated = R.matmul(adj_matrix, h)

        # 激活
        output = R.nn.relu(aggregated)

        R.output(output)
    return output
```

---

## 32.15 动态形状的调试技巧

### 32.15.1 打印符号形状信息



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm.relax.analysis import infer_struct_info

# 在编译过程中打印每个变量的 StructInfo
@tvm.transform.module_pass(opt_level=0)
def PrintStructInfo(mod, ctx):
    """打印所有变量的 StructInfo"""
    for gv, func in mod.functions.items():
        print(f"\n函数: {gv}")
        for param in func.params:
            sinfo = param.struct_info
            if sinfo is not None:
                print(f"  参数 {param.name_hint}: {sinfo}")
    return mod

# 在编译管线中使用
pipeline = tvm.transform.Sequential([
    PrintStructInfo("推断前"),
    relax.transform.InferStructInfo(),
    PrintStructInfo("推断后"),
])
```

### 32.15.2 形状约束追踪



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def trace_shape_constraints(mod):
    """追踪模块中的所有形状约束"""
    from tvm.relax.analysis import collect_shape_constraints

    constraints = collect_shape_constraints(mod)
    print("形状约束：")
    for i, constraint in enumerate(constraints):
        print(f"  [{i}] {constraint}")

    return constraints
```

### 32.15.3 运行时形状检查



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def debug_dynamic_shapes(vm, inputs, expected_shapes):
    """调试动态形状的运行时行为"""
    # 执行推理
    output = vm["main"](*inputs)

    # 检查输出形状
    for name, expected in expected_shapes.items():
        actual = getattr(output, name).shape if isinstance(output, dict) else output.shape

        # 解析期望形状（可能包含符号变量）
        for i, (a, e) in enumerate(zip(actual, expected)):
            if isinstance(e, str):
                print(f"  {name}[{i}]: {a} (符号: {e})")
            elif a != e:
                print(f"  {name}[{i}]: 实际 {a} != 期望 {e} ⚠️")
            else:
                print(f"  {name}[{i}]: {a} ✓")
```

### 32.15.4 常见错误及解决方案

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| "Cannot unify 64 and 128" | 形状约束冲突 | 检查维度匹配 |
| "Unknown rank" | 未指定 ndim | 使用 `TensorStructInfo(dtype, ndim=4)` |
| "Symbolic shape not supported" | 后端不支持 | 使用 Relax VM 而非 GraphRuntime |
| "Shape function mismatch" | Shape Func 推断错误 | 检查 Shape Func 实现 |
| "Runtime shape exceeds static bound" | 运行时形状超出编译时范围 | 调整形状范围参数 |

---

## 32.16 动态形状的高级优化

### 32.16.1 形状特化编译

对于常见的形状值，可以预编译特化的 kernel：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import meta_schedule

# 定义常见形状
common_shapes = [
    (1, 128),     # 单条推理
    (8, 128),     # 小批量
    (32, 128),    # 中等批量
    (128, 128),   # 大批量
]

# 为每个形状编译一个特化的版本
specialized_kernels = {}
for shape in common_shapes:
    # 固定形状进行调优
    sch = meta_schedule.tune_fixed_shape(
        func=matmul_func,
        shape=shape,
        target=target,
    )
    specialized_kernels[shape] = sch

# 运行时根据实际形状选择最优 kernel
def select_best_kernel(actual_batch, actual_seq):
    best_shape = min(common_shapes,
                     key=lambda s: abs(s[0] - actual_batch) + abs(s[1] - actual_seq))
    return specialized_kernels[best_shape]
```

### 32.16.2 动态形状的算子融合优化

动态形状对算子融合有特殊影响：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 静态形状下可以激进融合
# conv2d + batch_norm + relu → 一个融合 kernel
# 因为所有形状在编译时已知

# 动态形状下的融合考虑
# 1. 融合后的 kernel 需要传递更多形状参数
# 2. 中间张量的内存需要动态分配
# 3. 融合可能降低并行度（如果形状太小）

# Relax 的 FuseOpsByPattern 会考虑这些因素
# 自动选择是否融合
```

### 32.16.3 缓存友好的动态形状处理



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 对于频繁变化的形状，可以使用缓存策略
class ShapeCache:
    """形状到 kernel 的缓存"""

    def __init__(self, max_cache_size=100):
        self.cache = {}
        self.max_size = max_cache_size

    def get_kernel(self, shape, compile_func):
        """获取或编译 kernel"""
        shape_key = tuple(shape)

        if shape_key in self.cache:
            return self.cache[shape_key]

        # 缓存满时，使用 LRU 策略淘汰
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]

        # 编译新 kernel
        kernel = compile_func(shape)
        self.cache[shape_key] = kernel
        return kernel
```

---

## 32.99 文字内容强化：动态形状 的工程化理解

这一节用于把前文的 API、IR、Pass、Runtime 和部署片段串联为更完整的工程叙事。
很多学习者第一次阅读 TVM 文档时会觉得示例代码很多，但真正上线时仍然不知道如何判断方案是否可靠。
原因在于 TVM 不是单个推理库，而是一条从模型语义到硬件代码的编译链路。
链路越长，越需要把每一步的业务目标、内部机制、适用边界和失败模式说清楚。

### 32.99.1 代码解读的阅读方法

1. 阅读本章代码时，首先要判断它是在构建模型、转换 IR、配置 target、选择 executor、执行 tuning，还是加载 runtime artifact。
2. 如果片段位于前端导入阶段，重点不是性能，而是语义是否完整保留下来。
3. 如果片段位于 Relay、Relax 或 TIR 优化阶段，重点是 pass 是否改变了张量形状、数据类型、布局和算子边界。
4. 如果片段位于运行时阶段，重点是参数、输入、设备上下文和编译产物是否一一对应。
5. 不要把示例中的单个函数调用理解成黑盒魔法，它通常只是封装了多层 IR 变换。
6. 调试时应把模型切成前端、IR、调度、代码生成和运行时五个观察面。
7. 性能分析时应把端到端耗时、单算子耗时、数据搬运耗时和编译耗时分开记录。
8. 数值验证时应同时比较最大误差、平均误差、相对误差和业务指标。
9. 部署验证时应记录 TVM 版本、LLVM 版本、Python 包版本、目标硬件型号和驱动版本。
10. 当代码可以运行但结果不稳定时，优先怀疑输入预处理、随机数、线程调度和未固定的编译参数。

- 围绕“符号形状、Shape Function 与运行时 shape 计算”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“bucket、specialization 和 fully dynamic kernel 的权衡”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“NLP、检测、推荐和图模型中的动态维度模式”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 XLA dynamic shape、ONNX Runtime 和 TensorRT profile 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 32.99.2 业务意义

1. 动态形状 的业务价值不只是让模型跑得更快，而是让同一个模型可以在不同成本、功耗和延迟约束下交付。
2. 在服务器场景中，核心指标通常是吞吐、P95/P99 延迟、资源利用率和多租户隔离。
3. 在移动端场景中，核心指标通常是首帧时间、持续发热、内存峰值和包体大小。
4. 在嵌入式场景中，核心指标通常是 Flash 占用、静态内存、实时性和掉电恢复能力。
5. 在云端批处理场景中，编译时间可以接受更长，但调优记录和缓存复用变得非常重要。
6. 在在线服务场景中，编译产物需要可回滚、可审计、可灰度，而不能只在开发机上验证。
7. 业务方关心的是 SLA、成本和稳定性，编译器工程师关心的是 IR 正确性、优化空间和后端能力。
8. 优秀的 TVM 项目需要把这两类语言翻译成共同的指标体系。
9. 当优化收益只有少量百分点时，应评估它是否值得引入新的维护复杂度。
10. 当优化收益很大但只在少数输入上成立时，应评估输入分布变化后的风险。

- 围绕“符号形状、Shape Function 与运行时 shape 计算”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“bucket、specialization 和 fully dynamic kernel 的权衡”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“NLP、检测、推荐和图模型中的动态维度模式”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 XLA dynamic shape、ONNX Runtime 和 TensorRT profile 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 32.99.3 TVM 内部机制

1. TVM 的核心机制是分层表示和逐层降低，高层 IR 保留模型语义，低层 IR 暴露循环、内存和并行结构。
2. Relay 更适合表达静态计算图和传统深度学习算子。
3. Relax 更强调动态形状、函数式组合和跨层优化。
4. TIR 是最终性能调优的关键，因为它决定循环嵌套、内存作用域、向量化和线程映射。
5. PassContext 会影响优化级别、禁用或启用的 pass，以及某些后端特定配置。
6. Target 不只是字符串，它包含硬件特征、指令集、运行时约定和 codegen 选择。
7. Executor 决定模型入口如何组织，Runtime 决定编译产物如何加载和调用。
8. 参数绑定会影响常量折叠、内存规划和代码体积。
9. Layout rewrite 会影响算子融合和后端库调用。
10. AutoTVM 与 MetaSchedule 的调优记录会把搜索结果沉淀为可复用资产。

- 围绕“符号形状、Shape Function 与运行时 shape 计算”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“bucket、specialization 和 fully dynamic kernel 的权衡”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“NLP、检测、推荐和图模型中的动态维度模式”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 XLA dynamic shape、ONNX Runtime 和 TensorRT profile 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 32.99.4 适用场景

1. 当模型结构相对稳定、目标硬件明确、性能收益可以通过基准测试确认时，动态形状 相关技术最容易发挥价值。
2. 当团队需要支持多种硬件后端时，TVM 的统一 IR 和 Target 抽象可以降低重复适配成本。
3. 当模型中存在框架运行时开销、算子融合机会或布局转换冗余时，编译优化通常能带来明显收益。
4. 当部署环境不能依赖完整 Python 栈时，AOT、CRT 或导出后的 runtime artifact 更有意义。
5. 当硬件厂商提供高性能库但模型图需要复杂切分时，BYOC 和外部 codegen 是常见选择。
6. 当输入形状变化频繁时，应提前设计 shape 策略，而不是在上线前才补动态形状支持。
7. 当模型版本迭代频繁时，应把编译、调优、验证和发布纳入 CI/CD。
8. 当业务对精度非常敏感时，应把优化收益和数值回归一起评估。
9. 当系统存在多模型串联时，应评估端到端 pipeline，而不是只优化单个模型。
10. 当部署设备数量很大时，编译产物的一致性和可追踪性比单次实验性能更重要。

- 围绕“符号形状、Shape Function 与运行时 shape 计算”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“bucket、specialization 和 fully dynamic kernel 的权衡”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“NLP、检测、推荐和图模型中的动态维度模式”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 XLA dynamic shape、ONNX Runtime 和 TensorRT profile 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 32.99.5 限制条件

1. TVM 并不能自动解决所有性能问题，尤其不能替代对模型结构、硬件层次和数据分布的理解。
2. 如果前端导入阶段已经丢失语义，后续 pass 很难恢复原始意图。
3. 如果目标硬件缺少成熟 codegen 或 runtime 支持，理论上的 IR 优化收益可能无法兑现。
4. 如果模型包含大量小算子，调度优化可能被调用开销和内存同步抵消。
5. 如果输入分布与校准集或 benchmark 数据差异较大，性能和精度结论都可能失真。
6. 如果编译参数没有固定，调优结果难以复现。
7. 如果只看平均延迟，可能忽略 P99 抖动和内存峰值。
8. 如果只在开发机验证，可能忽略生产驱动、内核版本和设备温控策略。
9. 如果项目缺少 IR dump 和 artifact 存档，出现线上问题时很难追溯。
10. 如果团队没有维护编译工具链的能力，过度定制会形成长期负担。

- 围绕“符号形状、Shape Function 与运行时 shape 计算”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“bucket、specialization 和 fully dynamic kernel 的权衡”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“NLP、检测、推荐和图模型中的动态维度模式”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 XLA dynamic shape、ONNX Runtime 和 TensorRT profile 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 32.99.6 工程经验

1. 第一条经验是先建立可靠 baseline，再做任何复杂优化。
2. baseline 应包含原框架结果、TVM 未调优结果、TVM 调优结果和目标平台参考结果。
3. 第二条经验是每次只改变一个变量，例如 target、layout、executor、batch 或调优记录。
4. 第三条经验是把编译日志、IR dump、调优数据库和运行时产物一起归档。
5. 第四条经验是为每个模型维护最小可复现输入，便于定位前端导入和数值误差。
6. 第五条经验是把 shape、dtype、layout 写入模型契约，而不是散落在脚本里。
7. 第六条经验是使用真实业务样本做最终验证，因为随机输入无法覆盖预处理和分布偏移。
8. 第七条经验是把冷启动和热启动分开统计。
9. 第八条经验是为不同硬件维护独立调优记录，不要假设一个 schedule 可以跨平台复用。
10. 第九条经验是对编译产物做哈希和版本标记，方便灰度和回滚。
11. 第十条经验是让模型工程师、系统工程师和硬件工程师共享同一份性能报告。

- 围绕“符号形状、Shape Function 与运行时 shape 计算”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“bucket、specialization 和 fully dynamic kernel 的权衡”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“NLP、检测、推荐和图模型中的动态维度模式”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 XLA dynamic shape、ONNX Runtime 和 TensorRT profile 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 32.99.7 常见误区

1. 误区一是认为 TVM build 成功就等于模型可以生产上线。
2. 误区二是认为单次 benchmark 的最小值代表真实线上延迟。
3. 误区三是把调优时间算进推理收益，或者完全忽略调优成本。
4. 误区四是只关注算子融合数量，却不检查内存带宽和 cache 行为。
5. 误区五是看到 INT8 就默认更快，却忽略硬件是否有高效低精度指令。
6. 误区六是看到动态形状支持就默认所有输入尺寸都高效。
7. 误区七是把外部库调用当成万能方案，却忽略数据布局转换成本。
8. 误区八是把示例代码复制到生产环境，却没有补齐错误处理和 artifact 管理。
9. 误区九是只比较平均精度，不检查关键类别或长尾输入。
10. 误区十是把编译器问题、模型问题和硬件问题混在一起排查。

- 围绕“符号形状、Shape Function 与运行时 shape 计算”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“bucket、specialization 和 fully dynamic kernel 的权衡”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“NLP、检测、推荐和图模型中的动态维度模式”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 XLA dynamic shape、ONNX Runtime 和 TensorRT profile 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 32.99.8 生产部署注意事项

1. 生产部署前应冻结模型文件、输入预处理、编译配置和目标硬件描述。
2. 编译产物应包含可读元数据，例如模型版本、TVM commit、target、executor、runtime 和调优数据库版本。
3. 发布流程应支持灰度、回滚和双跑验证。
4. 线上监控应覆盖延迟、错误率、内存峰值、设备温度和业务指标。
5. 对于多线程 CPU 部署，应固定线程数并观察 NUMA、亲和性和其他服务的干扰。
6. 对于 GPU 部署，应区分 host 计时和 device 计时，并处理异步执行带来的误判。
7. 对于移动端部署，应关注长时间运行后的降频，而不是只看前几次推理。
8. 对于嵌入式部署，应把静态内存和栈空间作为硬约束。
9. 对于远程 RPC 测试，应把网络传输时间从设备执行时间中剥离。
10. 对于安全敏感业务，应限制模型产物和日志中的数据泄露风险。

- 围绕“符号形状、Shape Function 与运行时 shape 计算”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“bucket、specialization 和 fully dynamic kernel 的权衡”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“NLP、检测、推荐和图模型中的动态维度模式”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 XLA dynamic shape、ONNX Runtime 和 TensorRT profile 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 32.99.9 与同类系统对比

1. 与 TensorRT 相比，TVM 的硬件覆盖更开放，TensorRT 在 NVIDIA GPU 上的工程成熟度更高。
2. 与 XLA 相比，TVM 更强调可调度性和多后端扩展，XLA 更强调与框架图执行的深度集成。
3. 与 MLIR 相比，TVM 更像面向深度学习部署的完整编译器，MLIR 更像可构建编译器的基础设施。
4. 与 ONNX Runtime 相比，TVM 更关注提前编译和内核生成，ONNX Runtime 更强调运行时图优化和 execution provider 生态。
5. 与 Triton 相比，TVM 覆盖端到端模型编译，Triton 更适合手写或自动生成 GPU kernel。
6. 与 TFLite 相比，TVM 的后端扩展更灵活，TFLite 在移动端生态和模型格式上更标准化。
7. 与厂商 NPU SDK 相比，TVM 更中立，厂商 SDK 往往能访问更底层的私有能力。
8. 选择系统时不应只看峰值性能，还应看调试成本、团队经验、社区活跃度和长期维护风险。
9. 如果项目只部署到单一成熟硬件，专用推理引擎可能更省事。
10. 如果项目需要跨硬件、跨模型长期演进，TVM 的编译器化路线更有战略价值。

- 围绕“符号形状、Shape Function 与运行时 shape 计算”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“bucket、specialization 和 fully dynamic kernel 的权衡”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“NLP、检测、推荐和图模型中的动态维度模式”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 XLA dynamic shape、ONNX Runtime 和 TensorRT profile 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 32.99.10 章节复盘

1. 回到本章，动态形状 的关键不是记住所有 API，而是理解为什么这些 API 会出现在编译链路的这个位置。
2. 当你看到一段代码时，应能说出它改变了模型语义、调度空间、内存布局、运行时入口还是部署产物。
3. 当你看到一个性能数字时，应能说出它的测试输入、硬件状态、计时方法和误差范围。
4. 当你看到一个优化 pass 时，应能说出它依赖的前置假设和可能破坏的边界条件。
5. 当你准备上线时，应能说出失败后如何回滚、如何复现、如何定位和如何与业务方沟通影响。
6. 这套思维比单个示例更重要，因为 TVM 的 API 会演进，但编译部署的工程约束长期稳定。
7. 后续学习中，可以把每一章都转化为一张决策表：何时使用、收益来自哪里、风险是什么、如何验证。
8. 只有把代码、机制和工程策略放在一起，TVM 才不只是工具箱，而是可运行的生产系统。
9. 因此，本章新增的文字说明应作为阅读代码段的上下文，而不是替代对原始代码的逐行理解。
10. 如果遇到与示例不一致的实际项目，应优先回到模型约束和目标硬件，而不是机械套用章节流程。

- 围绕“符号形状、Shape Function 与运行时 shape 计算”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“bucket、specialization 和 fully dynamic kernel 的权衡”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“NLP、检测、推荐和图模型中的动态维度模式”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 XLA dynamic shape、ONNX Runtime 和 TensorRT profile 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。


## 32.17 本章小结

本章深入探讨了 TVM 中动态形状处理的各个方面：

1. **问题本质**：动态形状是 NLP、目标检测等场景的普遍需求
2. **Relay 的局限**：`Any` 类型无法表达符号约束，优化空间有限
3. **Relax 的优势**：符号变量（Symbolic Var）原生支持动态形状推理
4. **Dynamic TIR**：处理符号循环边界的低层技术
5. **Shape Func**：运行时形状推断机制
6. **调度策略**：分段编译、符号调度等应对动态形状的方法
7. **性能考量**：形状对 kernel 性能的影响及 profiling 方法
8. **约束求解**：形状约束的自动推断与冲突检测
9. **TIR 变换**：LoopPartition、Vectorize、StorageRewrite 对动态形状的处理
10. **部署案例**：NLP、目标检测、图神经网络的动态形状实战
11. **调试技巧**：形状追踪、约束检查、运行时验证

动态形状处理是 TVM 从 Relay 演进到 Relax 的重要驱动力之一，Relax 的符号形状设计从根本上解决了 `Any` 类型的局限性。

<div data-component="DynamicShapesComparisonTable"></div>

> **下一章预告**：第 33 章将介绍 TVM 的性能剖析与基准测试工具，帮助开发者系统性地分析和优化模型性能。

---

## 附录 A：动态形状的符号算术推理规则

### A.1 加法推理

```python
# 符号变量加法
m = tir.Var("m", "int64")
n = tir.Var("n", "int64")

# 规则：m + n 产生新的符号表达式
# 不同变量相加不能简化
expr1 = m + n  # 表达式：Add(m, n)

# 相同变量相加可以简化
expr2 = m + m  # 简化为：Mul(2, m)

# 常量加法可以折叠
expr3 = m + 3  # 表达式：Add(m, 3)
expr4 = 3 + 5  # 简化为：8
```

### A.2 乘法推理



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 符号变量乘法
m = tir.Var("m", "int64")

# 规则：m * k 产生乘法表达式
expr1 = m * 2  # 表达式：Mul(2, m)

# 乘法分配律
expr2 = m * 2 + m * 3  # 简化为：Mul(5, m)

# 除法逆运算
expr3 = (m * 8) // 8  # 简化为：m

# 但需要注意整数除法的截断
expr4 = (m * 8 + 3) // 8  # 不能简化为 m，因为有截断
```

### A.3 除法与取模推理



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 整数除法
m = tir.Var("m", "int64")

# 规则：m // k 的取值范围
# 如果 m >= 0，则 0 <= m // k <= m / k

# 对齐计算
aligned = (m + 3) // 4 * 4  # 向上对齐到 4 的倍数
# 属性：aligned >= m
# 属性：aligned % 4 == 0

# 常见的对齐模式
tile_count = (m + 31) // 32  # 分块数量
remainder = m % 32           # 余数
```

### A.4 形状广播推理



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 广播规则
m = tir.Var("m", "int64")
n = tir.Var("n", "int64")

# 标量广播
# (m, n) + scalar → (m, n)

# 向量广播
# (m, n) + (1, n) → (m, n)  # 沿第一维广播
# (m, n) + (m, 1) → (m, n)  # 沿第二维广播

# 不确定的广播
# (m, n) + (m, n) → (m, n)  # 相同形状
# (m, n) + (k, n) → 错误！    # m 和 k 可能不同
# 除非有约束 m == k
```

---

## 附录 B：动态形状的常见模式与解决方案

### B.1 变长序列处理



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 模式：处理变长序列（如 NLP 中的 token 序列）
# 挑战：不同样本的序列长度不同

# 解决方案 1：Padding + Masking
# 将所有序列 pad 到最大长度
max_len = 512
padded_input = pad_to_length(input, max_len)  # (batch, max_len)
attention_mask = create_mask(input_lengths)     # (batch, max_len)

# 解决方案 2：动态形状（Relax 原生支持）
@R.function
def process_variable_length(
    x: R.Tensor(("batch", "seq_len"), "int64"),
    mask: R.Tensor(("batch", "seq_len"), "float32"),
):
    # 直接处理变长序列，无需 padding
    ...
```

### B.2 动态批处理



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 模式：在线服务中批大小动态变化
# 挑战：不同请求到达时批大小不同

# 解决方案：符号批大小
batch = tir.Var("batch", "int64")

@R.function
def serve_request(
    x: R.Tensor((batch, feature_dim), "float32"),
    w: R.Tensor((feature_dim, output_dim), "float32"),
):
    with R.dataflow():
        # 自动适应任意批大小
        lv0 = R.matmul(x, w)  # (batch, output_dim)
        R.output(lv0)
    return lv0
```

### B.3 多尺度特征处理



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 模式：图像处理中输入分辨率可变
# 挑战：特征图大小依赖于输入分辨率

# 解决方案：符号空间维度
H = tir.Var("H", "int64")
W = tir.Var("W", "int64")

@R.function
def process_multiscale(
    image: R.Tensor((1, 3, H, W), "float32"),
    weight: R.Tensor((64, 3, 7, 7), "float32"),
):
    with R.dataflow():
        # 卷积输出形状：(1, 64, H-6, W-6)（假设 stride=1, padding=0）
        feature = R.nn.conv2d(image, weight)
        R.output(feature)
    return feature
```

### B.4 条件形状依赖



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 模式：输出形状依赖于输入数据的值（如 NMS）
# 挑战：形状不仅是动态的，还依赖于运行时值

# 解决方案：使用符号变量 + ShapeFunc
@R.function
def nms(
    boxes: R.Tensor((N, 4), "float32"),
    scores: R.Tensor((N,), "float32"),
    threshold: R.Tensor((), "float32"),
):
    # 输出形状是动态的，依赖于 scores 的值和 threshold
    # 使用 ShapeFunc 在运行时推断
    result = R.call_packed(
        "nms_impl", boxes, scores, threshold,
        sinfo_args=R.TensorStructInfo(("M", 4), "float32")
    )
    return result
```

---

## 附录 C：动态形状相关的源码索引

| 功能 | 源码文件 | 关键函数/类 |
|------|---------|------------|
| 符号形状定义 | `include/tvm/relax/struct_info.h` | `TensorStructInfo` |
| StructInfo 推断 | `src/relax/transform/infer_struct_info.cc` | `InferStructInfo` |
| 形状约束求解 | `src/relax/analysis/struct_info_analysis.cc` | `ConstraintSolver` |
| Dynamic TIR | `src/tir/transforms/loop_partition.cc` | `LoopPartition` |
| 向量化 | `src/tir/transforms/vectorize_loop.cc` | `VectorizeLoop` |
| 存储重写 | `src/tir/transform/storage_rewrite.cc` | `StorageRewrite` |
| 符号简化 | `src/tir/op.cc` | `Simplify` |
| ShapeFunc 注册 | `src/relay/op/` | 各算子的 `FInferStructInfo` |
| Relax VM 动态执行 | `src/relax/backend/vm/vm.cc` | `VirtualMachine` |
| PyTorch 前端动态导入 | `python/tvm/relax/frontend/pytorch_translator.py` | `PyTorchImporter` |
| ONNX 前端动态导入 | `python/tvm/relax/frontend/onnx_translator.py` | `OnnxImporter` |

---

## 附录 D：动态形状的数学基础

### D.1 符号算术的形式化定义

设 $\mathcal{S}$ 为符号变量集合，$\mathcal{E}$ 为符号表达式集合：

$$
\mathcal{E} ::= n \mid s \mid \mathcal{E} + \mathcal{E} \mid \mathcal{E} - \mathcal{E} \mid \mathcal{E} \times \mathcal{E} \mid \mathcal{E} \div \mathcal{E}
$$

其中 $n \in \mathbb{Z}$ 为常量，$s \in \mathcal{S}$ 为符号变量。

### D.2 等价类与约束求解

对于形状约束系统，TVM 使用等价类方法进行求解：

$$
\text{Equal}(a, b) \iff [a] = [b]
$$

其中 $[a]$ 表示 $a$ 所在的等价类。

### D.3 广播规则的形式化

广播规则的形式化定义：

$$
\text{broadcast}(d_1, d_2) = \begin{cases}
d_1 & \text{if } d_2 = 1 \\
d_2 & \text{if } d_1 = 1 \\
d_1 & \text{if } d_1 = d_2 \\
\text{error} & \text{otherwise}
\end{cases}
$$

对于符号维度，需要考虑：

$$
\text{broadcast}(d_1, d_2) = \begin{cases}
d_1 & \text{if } d_2 = 1 \\
d_2 & \text{if } d_1 = 1 \\
d_1 & \text{if } \text{Equal}(d_1, d_2) \\
d_1 & \text{if } d_1 \text{ 和 } d_2 \text{ 无法确定不等} \\
\text{error} & \text{if } d_1 \neq d_2 \text{ 且均不为 1}
\end{cases}
$$

### D.4 形状推断的复杂度

形状推断的计算复杂度取决于约束系统的结构：

- **线性约束**：$O(n \log n)$，使用并查集
- **多项式约束**：$O(n^2)$，需要符号推理
- **非线性约束**：NP-hard，需要近似求解

TVM 主要处理线性约束（加法和乘法），对于更复杂的情况使用保守近似。

---

## 附录 E：动态形状的测试策略

### E.1 测试用例设计



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import pytest
import numpy as np
import tvm
from tvm import relax, tir

class TestDynamicShapes:
    """动态形状测试套件"""

    @pytest.fixture
    def symbols(self):
        """创建符号变量"""
        return {
            "batch": tir.Var("batch", "int64"),
            "seq_len": tir.Var("seq_len", "int64"),
            "hidden": tir.Var("hidden", "int64"),
        }

    def test_matmul_dynamic_batch(self, symbols):
        """测试动态批大小的矩阵乘法"""
        m = symbols["batch"]

        @R.function
        def matmul_func(
            x: R.Tensor((m, 256), "float32"),
            w: R.Tensor((256, 512), "float32"),
        ):
            with R.dataflow():
                lv0 = R.matmul(x, w)
                R.output(lv0)
            return lv0

        # 编译
        ex = relax.vm.build(matmul_func, target="llvm")
        vm = relax.VirtualMachine(ex, tvm.cpu())

        # 测试不同的批大小
        for batch_size in [1, 4, 16, 64, 128]:
            x_np = np.random.randn(batch_size, 256).astype("float32")
            w_np = np.random.randn(256, 512).astype("float32")

            result = vm["matmul_func"](
                tvm.nd.array(x_np),
                tvm.nd.array(w_np),
            )

            # 验证形状
            assert result.shape == (batch_size, 512)

            # 验证正确性
            ref = np.matmul(x_np, w_np)
            np.testing.assert_allclose(result.numpy(), ref, rtol=1e-3)

    def test_attention_dynamic_seq(self, symbols):
        """测试动态序列长度的注意力"""
        batch = symbols["batch"]
        seq = symbols["seq_len"]

        @R.function
        def attention_func(
            Q: R.Tensor((batch, seq, 64), "float32"),
            K: R.Tensor((batch, seq, 64), "float32"),
        ):
            with R.dataflow():
                K_T = R.permute_dims(K, (0, 2, 1))
                scores = R.matmul(Q, K_T)
                attn = R.nn.softmax(scores, axis=-1)
                R.output(attn)
            return attn

        # 编译
        ex = relax.vm.build(attention_func, target="llvm")
        vm = relax.VirtualMachine(ex, tvm.cpu())

        # 测试不同的序列长度
        for seq_len in [16, 32, 64, 128, 256]:
            Q_np = np.random.randn(1, seq_len, 64).astype("float32")
            K_np = np.random.randn(1, seq_len, 64).astype("float32")

            result = vm["attention_func"](
                tvm.nd.array(Q_np),
                tvm.nd.array(K_np),
            )

            # 验证形状
            assert result.shape == (1, seq_len, seq_len)

    def test_reshape_symbolic(self, symbols):
        """测试符号形状的 reshape"""
        m = symbols["batch"]
        n = symbols["hidden"]

        @R.function
        def reshape_func(
            x: R.Tensor((m, n), "float32"),
        ):
            with R.dataflow():
                # n = 12 * 64 = 768
                lv0 = R.reshape(x, (m, 12, 64))
                R.output(lv0)
            return lv0

        # 需要约束 n == 768
        # 这里假设 n 是 768 的符号表示
```

### E.2 边界条件测试



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def test_edge_cases():
    """测试动态形状的边界条件"""
    # 边界 1：最小形状
    # batch = 1, seq_len = 1
    x = np.random.randn(1, 1, 64).astype("float32")
    # ...

    # 边界 2：最大形状（受内存限制）
    # batch = 1024, seq_len = 4096
    # ...

    # 边界 3：形状为 0（空张量）
    # 某些情况下可能遇到空张量
    x = np.random.randn(0, 64).astype("float32")
    # 需要确保不会崩溃

    # 边界 4：非常大的形状
    # 需要确保整数不溢出
```

### E.3 性能回归测试



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def test_performance_no_regression():
    """测试动态形状下没有性能回归"""
    # 基线性能
    baseline_latency = {
        (1, 128): 0.001,    # 1ms
        (32, 128): 0.005,   # 5ms
        (128, 128): 0.015,  # 15ms
    }

    # 当前性能
    for shape, expected in baseline_latency.items():
        actual = measure_latency(shape)
        ratio = actual / expected
        assert ratio < 1.1, f"形状 {shape} 性能回归: {ratio:.2f}x"
```

---

## 附录 F：动态形状的最佳实践

### F.1 设计原则

1. **尽早确定动态维度**：在模型设计阶段就考虑哪些维度需要动态
2. **使用有意义的符号名称**：`batch`, `seq_len` 比 `m`, `n` 更清晰
3. **添加形状约束**：在需要时显式声明约束
4. **测试多种形状**：覆盖边界条件和常见形状

### F.2 性能建议

1. **避免过度动态化**：只将真正需要动态的维度设为符号
2. **使用形状特化**：对常见形状预编译优化版本
3. **缓存编译结果**：避免重复编译相同形状的 kernel
4. **监控形状分布**：记录运行时的形状分布，指导优化

### F.3 调试建议

1. **打印 StructInfo**：在每个 Pass 后检查形状信息
2. **验证约束一致性**：确保所有约束都可满足
3. **使用参考实现**：对比动态和静态版本的正确性
4. **检查边界情况**：特别关注形状为 0、1、最大值的情况

---

## 附录 G：术语表

| 术语 | 英文 | 说明 |
|------|------|------|
| 符号变量 | Symbolic Variable | 编译时未知的形状维度变量 |
| 符号形状 | Symbolic Shape | 包含符号变量的形状表达式 |
| 形状推断 | Shape Inference | 根据输入形状计算输出形状的过程 |
| ShapeFunc | Shape Function | 运行时形状推断函数 |
| StructInfo | Structured Info | Relax 的结构化类型信息 |
| 动态 TIR | Dynamic TIR | 包含符号变量的 TIR 代码 |
| 形状约束 | Shape Constraint | 符号变量之间的关系约束 |
| 广播 | Broadcasting | 不同形状张量的运算规则 |
| LoopPartition | Loop Partitioning | 将动态循环分割为静态和尾部 |
| 符号调度 | Symbolic Schedule | 使用符号变量的调度策略 |
| 形状特化 | Shape Specialization | 为特定形状预编译优化版本 |
| Any 类型 | Any Type | Relay 中的动态维度类型 |

---

## 附录 H：常见问题解答

### H.1 Q: 动态形状会导致性能下降吗？

**A:** 不一定。性能影响取决于：
- **符号调度**：如果能找到对所有形状都有效的调度，性能可能接近静态版本
- **形状特化**：对常见形状预编译优化版本可以达到与静态版本相同的性能
- **运行时开销**：形状推断和内存分配会引入少量开销，通常可以忽略

### H.2 Q: 如何选择使用 Any 类型还是符号变量？

**A:** 优先使用符号变量（Relax）。只有在使用 Relay 且无法避免时才使用 Any 类型。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
选择决策：
├── 使用 Relax？
│   ├── 是 → 使用 tir.Var（符号变量）
│   └── 否 → 使用 relay.Any（Any 类型）
└── 需要约束推理？
    ├── 是 → 必须使用符号变量
    └── 否 → 两者都可
```

### H.3 Q: 动态形状下如何进行算子融合？

**A:** Relax 的 FuseOpsByPattern 会自动处理动态形状。融合时需要注意：
- 融合后的 kernel 需要传递更多形状参数
- 某些融合模式可能不适用于所有形状
- 可以通过配置控制融合策略

### H.4 Q: 如何调试动态形状的编译错误？

**A:** 推荐以下调试步骤：
1. 使用 `InferStructInfo` Pass 打印推断结果
2. 检查符号约束是否可满足
3. 验证 reshape 的维度是否匹配
4. 使用 `relax.analysis.well_formed()` 检查 IR 合法性

### H.5 Q: 动态形状下的内存如何管理？

**A:** Relax VM 提供了多种内存管理策略：
- **按需分配**：每次执行时根据实际形状分配内存
- **内存池**：维护内存池，复用相同大小的内存
- **预分配**：如果有形状范围信息，可以预分配最大形状的内存

### H.6 Q: 是否可以混合使用静态和动态形状？

**A:** 可以。推荐的做法是：
- 对确定不变的维度使用静态值
- 对可能变化的维度使用符号变量
- 这样可以在保持灵活性的同时最大化优化空间



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 示例：批大小动态，特征维度静态
batch = tir.Var("batch", "int64")
@R.function
def main(x: R.Tensor((batch, 768), "float32"),  # batch 动态，768 静态
         w: R.Tensor((768, 256), "float32")):     # 完全静态
    ...
```

### H.7 Q: 动态形状如何影响自动调优？

**A:** 自动调优工具（MetaSchedule）支持符号形状：
- 可以为符号形状搜索调度
- 搜索空间比静态形状更大
- 可能需要更多的调优试验来找到最优调度
- 建议对形状范围进行分段调优

---

## 32.18 符号形状传播算法详解

### 32.18.1 符号形状传播的核心问题

在 Relax 中，每个张量的形状由 `TensorStructInfo` 描述。当张量经过算子变换时，编译器需要推断输出张量的形状。对于包含符号变量的形状，这个过程涉及符号算术推理和约束求解：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
输入形状推断问题：
  给定：x 的形状为 (batch, seq_len, 768)
        w 的形状为 (768, 3072)
  求：  R.matmul(x, w) 的输出形状

推断过程：
  1. 检查收缩维度：x 的最后一维 768 == w 的第一维 768 ✓
  2. 保留 batch 维度：batch, seq_len
  3. 取输出维度：3072
  4. 结果：(batch, seq_len, 3072)
```

### 32.18.2 等价类与统一算法

TVM 使用基于等价类（Union-Find）的统一算法来处理符号变量的相等约束。该算法在 `src/relax/analysis/struct_info_analysis.cc` 中实现：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// src/relax/analysis/struct_info_analysis.cc — 符号统一算法
class SymbolUnifier {
 public:
  // 初始化：每个符号变量是自己的等价类代表
  void Init(tir::Var var) {
    parent_[var.get()] = var.get();
    rank_[var.get()] = 0;
  }

  // 查找：路径压缩，找到等价类的代表
  tir::Var Find(tir::Var var) {
    if (parent_[var.get()] != var.get()) {
      parent_[var.get()] = Find(parent_[var.get()]).get();  // 路径压缩
    }
    return tir::Var::GetRepr(parent_[var.get()]);
  }

  // 合并：将两个符号变量的等价类合并
  // 返回 true 表示合并成功，false 表示冲突
  bool Unify(tir::Var a, tir::Var b) {
    tir::Var root_a = Find(a);
    tir::Var root_b = Find(b);

    // 已经在同一等价类中
    if (root_a == root_b) return true;

    // 按秩合并：将秩小的树挂到秩大的树下
    if (rank_[root_a.get()] < rank_[root_b.get()]) {
      parent_[root_a.get()] = root_b.get();
    } else if (rank_[root_a.get()] > rank_[root_b.get()]) {
      parent_[root_b.get()] = root_a.get();
    } else {
      parent_[root_b.get()] = root_a.get();
      rank_[root_a.get()]++;
    }
    return true;
  }

  // 检查两个符号是否等价
  bool IsEqual(tir::Var a, tir::Var b) {
    return Find(a) == Find(b);
  }

 private:
  std::unordered_map<const Object*, const Object*> parent_;
  std::unordered_map<const Object*, int> rank_;
};
```

### 32.18.3 形状传播的约束类型

形状推断过程中会产生多种约束：

| 约束类型 | 含义 | 示例 | 处理方式 |
|---------|------|------|---------|
| **相等约束** | 两个维度必须相等 | matmul 收缩维度 | 统一（Union） |
| **广播约束** | 一个维度可以是 1 或等于另一个 | add 的逐元素广播 | 条件统一 |
| **算术约束** | 维度之间存在算术关系 | `hidden = num_heads * head_dim` | 符号表达式记录 |
| **范围约束** | 维度的取值范围 | `batch > 0`, `seq_len >= 1` | 符号不等式 |
| **常量约束** | 维度是编译时常量 | `hidden_size = 768` | 直接替换 |

### 32.18.4 广播规则的符号传播

广播规则在符号维度下的实现需要特别处理：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def broadcast_shape_dim(dim_a, dim_b):
    """符号维度的广播规则

    参数：
        dim_a: PrimExpr - 第一个张量在该维度的大小
        dim_b: PrimExpr - 第二个张量在该维度的大小

    返回：
        PrimExpr - 广播后的维度大小

    规则：
        1. 如果 dim_a 是常量 1 → 返回 dim_b（广播 dim_a）
        2. 如果 dim_b 是常量 1 → 返回 dim_a（广播 dim_b）
        3. 如果 dim_a 和 dim_b 是相同的符号变量 → 返回 dim_a
        4. 如果 dim_a 和 dim_b 是等价的符号表达式 → 返回 dim_a
        5. 否则 → 报错（无法确定是否可以广播）
    """
    # 规则 1 和 2：标量广播
    if is_const_one(dim_a):
        return dim_b
    if is_const_one(dim_b):
        return dim_a

    # 规则 3：相同的符号变量
    if isinstance(dim_a, tir.Var) and isinstance(dim_b, tir.Var):
        if symbol_unifier.IsEqual(dim_a, dim_b):
            return dim_a  # 已证明相等
        # 尝试统一
        if symbol_unifier.Unify(dim_a, dim_b):
            return dim_a  # 统一成功
        # 无法确定是否相等 → 保守处理
        raise StructInfoError(
            f"Cannot broadcast dimensions {dim_a} and {dim_b}: "
            f"they may not be equal")

    # 规则 4：涉及算术表达式的比较
    # 使用符号简化器检查是否相等
    diff = tir.Simplify(dim_a - dim_b)
    if isinstance(diff, tir.IntImm) and diff.value == 0:
        return dim_a  # 简化后证明相等

    raise StructInfoError(
        f"Cannot determine broadcast result for {dim_a} and {dim_b}")
```

### 32.18.5 形状传播的完整流程

以一个 Transformer 模型为例，展示形状如何逐层传播：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import tir, relax

batch = tir.Var("batch", "int64")
seq_len = tir.Var("seq_len", "int64")
hidden = 768
num_heads = 12
head_dim = hidden // num_heads  # 64

# ====== 形状传播追踪 ======

# 第 1 层：输入 x
# x: (batch, seq_len, 768)
# 等价类: {batch}, {seq_len}

# 第 2 层：Q = R.matmul(x, w_q)
# x: (batch, seq_len, 768), w_q: (768, 768)
# 收缩维度：768 == 768 ✓
# Q 输出形状：(batch, seq_len, 768)
# 等价类不变

# 第 3 层：Q_h = R.reshape(Q, (batch, seq_len, num_heads, head_dim))
# 输入：(batch, seq_len, 768)
# 目标：(batch, seq_len, 12, 64)
# 验证：batch * seq_len * 768 == batch * seq_len * 12 * 64
#       即 768 == 12 * 64 = 768 ✓
# Q_h 输出形状：(batch, seq_len, 12, 64)

# 第 4 层：Q_t = R.permute_dims(Q_h, [0, 2, 1, 3])
# 输入：(batch, seq_len, 12, 64)
# 按照维度重排 [0, 2, 1, 3]
# Q_t 输出形状：(batch, 12, seq_len, 64)

# 第 5 层：scores = R.matmul(Q_t, K_t_T)
# Q_t: (batch, 12, seq_len, 64), K_t_T: (batch, 12, 64, seq_len)
# 收缩维度：64 == 64 ✓
# batch 维度广播：batch == batch ✓
# head 维度广播：12 == 12 ✓
# scores 输出形状：(batch, 12, seq_len, seq_len)
# 注意：seq_len 出现了两次，这是合法的！
```

---

## 32.19 Dynamic TIR 循环生成机制

### 32.19.1 从 Relax 到 TIR 的代码生成路径

当 Relax 包含符号形状时，编译器需要生成能处理动态边界的 TIR 代码。完整的代码生成路径如下：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
Relax IR（包含符号变量）
    ↓ LegalizeOps Pass
    将 R.matmul(x, w) 转换为 R.call_tir(matmul_tir_func, (x, w), out_sinfo)
    ↓ CallTIRRewrite Pass
    将 call_tir 重写为低层形式，添加形状参数
    ↓ TIR Lower
    生成包含符号变量的 TIR PrimFunc
    ↓ CodeGen
    针对目标平台生成代码（LLVM IR / CUDA / WASM 等）
```

### 32.19.2 符号边界的循环代码生成

对于包含符号边界的循环，TVM 的 CodeGen 需要生成运行时可变的循环边界：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# Relax 中的动态 matmul
@R.function
def dynamic_matmul(
    x: R.Tensor((m, k), "float32"),  # m, k 是符号变量
    w: R.Tensor((k, n), "float32"),  # n 是符号变量
):
    with R.dataflow():
        lv0 = R.matmul(x, w)
        R.output(lv0)
    return lv0

# LegalizeOps 后生成的 TIR 函数
@T.prim_func
def matmul_tir(
    A: T.Buffer((T.Var("m"), T.Var("k")), "float32"),
    B: T.Buffer((T.Var("k"), T.Var("n")), "float32"),
    C: T.Buffer((T.Var("m"), T.Var("n")), "float32"),
    m: T.int64, k: T.int64, n: T.int64  # 形状参数
):
    # 循环边界是符号变量，运行时传入实际值
    for i in T.serial(m):          # 外层循环：m 次迭代
        for j in T.serial(n):      # 中层循环：n 次迭代
            for kk in T.serial(k): # 内层循环：k 次迭代
                with T.block("C"):
                    vi = T.axis.spatial(m, i)
                    vj = T.axis.spatial(n, j)
                    vk = T.axis.reduce(k, kk)
                    # 初始化（只在第一次迭代时）
                    if vk == 0:
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

# LLVM CodeGen 生成的伪代码：
# define void @matmul_tir(float* %A, float* %B, float* %C,
#                          i64 %m, i64 %k, i64 %n) {
# entry:
#   br label %outer_loop
# outer_loop:                      ; i = 0, 1, ..., m-1
#   %i = phi i64 [0, %entry], [%i_next, %outer_latch]
#   br label %middle_loop
# middle_loop:                     ; j = 0, 1, ..., n-1
#   %j = phi i64 [0, %outer_loop], [%j_next, %middle_latch]
#   store float 0.0, ...          ; C[i,j] = 0
#   br label %inner_loop
# inner_loop:                      ; kk = 0, 1, ..., k-1
#   %kk = phi i64 [0, %middle_loop], [%kk_next, %inner_loop]
#   ; 加载 A[i, kk] 和 B[kk, j]
#   ; 乘加到 C[i, j]
#   %kk_next = add i64 %kk, 1
#   %kk_cond = icmp slt i64 %kk_next, %k  ; 运行时比较
#   br i1 %kk_cond, label %inner_loop, label %middle_latch
#   ...
# }
```

### 32.19.3 LoopPartition 对符号循环的处理

`LoopPartition` Pass 是处理符号循环的关键变换，定义在 `src/tir/transforms/loop_partition.cc`。它的核心任务是将一个包含符号边界的循环分割为可优化的静态部分和尾部处理部分：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# ====== LoopPartition 的完整处理流程 ======

# 原始循环（符号边界 n）
for i in range(n):
    A[i] = B[i] + 1.0

# 第一步：分析向量化需求
# 目标：将循环向量化为 4-wide SIMD 操作
# 问题：n 不是 4 的倍数，无法直接向量化

# 第二步：计算分割点
# 静态部分的上界：(n // 4) * 4   （n 的最大 4 的倍数倍数下取整）
# 尾部部分的下界：(n // 4) * 4
# 尾部部分的上界：n

# 第三步：生成分割后的代码
# 主循环（可向量化）
n_aligned = (n // 4) * 4
for i in range(0, n_aligned, 4):
    # 向量化加载 4 个 float32
    vec_A = T.load("float32x4", A.data, i)   # 一次加载 4 个元素
    vec_B = T.load("float32x4", B.data, i)
    vec_C = vec_A + vec_B                     # SIMD 加法
    T.store("float32x4", A.data, i, vec_C)   # 一次存储 4 个元素

# 尾部循环（标量处理剩余元素）
for i in range(n_aligned, n):
    A[i] = B[i] + 1.0

# LLVM 生成的代码：
# ; 主循环（向量化）
# main_loop:
#   %i = phi i64 [0, %entry], [%i_next, %main_loop]
#   %vec_a = load <4 x float>, ptr %a_ptr
#   %vec_b = load <4 x float>, ptr %b_ptr
#   %vec_c = fadd <4 x float> %vec_a, %vec_b
#   store <4 x float> %vec_c, ptr %a_ptr
#   %i_next = add i64 %i, 4
#   %cond = icmp slt i64 %i_next, %n_aligned
#   br i1 %cond, label %main_loop, label %tail
#
# ; 尾部循环（标量）
# tail:
#   %j = phi i64 [%n_aligned, %main_loop], [%j_next, %tail]
#   %a_val = load float, ptr %a_j_ptr
#   %b_val = load float, ptr %b_j_ptr
#   %c_val = fadd float %a_val, %b_val
#   store float %c_val, ptr %a_j_ptr
#   %j_next = add i64 %j, 1
#   %j_cond = icmp slt i64 %j_next, %n
#   br i1 %j_cond, label %tail, label %exit
```

### 32.19.4 GPU 上的动态循环处理

GPU 上处理符号循环需要额外考虑线程绑定和共享内存：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# GPU 上的动态形状 matmul kernel
@T.prim_func
def dynamic_matmul_gpu(
    A: T.handle, B: T.handle, C: T.handle,
    m: T.int64, k: T.int64, n: T.int64
):
    A_buf = T.match_buffer(A, (m, k), "float32")
    B_buf = T.match_buffer(B, (k, n), "float32")
    C_buf = T.match_buffer(C, (m, n), "float32")

    # 块大小常量
    BLOCK_M = T.int32(32)
    BLOCK_N = T.int32(32)
    BLOCK_K = T.int32(32)

    # 计算 grid 维度（需要向上取整）
    grid_m = T.ceildiv(m, BLOCK_M)  # (m + 31) // 32
    grid_n = T.ceildiv(n, BLOCK_N)  # (n + 31) // 32

    # 共享内存声明
    shared_A = T.alloc_shared((BLOCK_M, BLOCK_K), "float32")
    shared_B = T.alloc_shared((BLOCK_K, BLOCK_N), "float32")

    # Grid + Block 绑定
    for bx in T.thread_binding(0, grid_n, "blockIdx.x"):
        for by in T.thread_binding(0, grid_m, "blockIdx.y"):
            for tx in T.thread_binding(0, 32, "threadIdx.x"):
                for ty in T.thread_binding(0, 32, "threadIdx.y"):
                    # 局部累加器
                    acc = T.alloc_buffer((), "float32", scope="local")
                    acc[()] = T.float32(0)

                    # K 维度的分块循环
                    for ko in T.serial(T.ceildiv(k, BLOCK_K)):
                        # 从全局内存加载到共享内存
                        # 需要边界检查（因为 m, k, n 可能不是 BLOCK 的倍数）
                        i = by * BLOCK_M + ty
                        j = bx * BLOCK_N + tx

                        # 带边界检查的加载
                        if i < m and ko * BLOCK_K + tx < k:
                            shared_A[ty, tx] = A_buf[i, ko * BLOCK_K + tx]
                        else:
                            shared_A[ty, tx] = T.float32(0)

                        if ko * BLOCK_K + ty < k and j < n:
                            shared_B[ty, tx] = B_buf[ko * BLOCK_K + ty, j]
                        else:
                            shared_B[ty, tx] = T.float32(0)

                        T.tvm_storage_sync("shared")  # 同步屏障

                        # 局部矩阵乘法
                        for kk in T.serial(BLOCK_K):
                            acc[()] = acc[()] + shared_A[ty, kk] * shared_B[kk, tx]

                        T.tvm_storage_sync("shared")  # 再次同步

                    # 写回全局内存（带边界检查）
                    if by * BLOCK_M + ty < m and bx * BLOCK_N + tx < n:
                        C_buf[by * BLOCK_M + ty, bx * BLOCK_N + tx] = acc[()]

# GPU 动态循环的关键挑战总结：
#
# 1. 线程块大小固定，但 grid 大小需要运行时计算
#    → 使用 T.ceildiv 动态计算 grid 维度
#
# 2. 共享内存大小在编译时固定
#    → 使用常量 BLOCK_M, BLOCK_N 而非符号变量
#
# 3. 边界检查引入额外开销
#    → 编译器尽量将检查提升到外层循环
#
# 4. warp divergence 可能增加
#    → 当 n 不是 warp 大小的倍数时，最后一个 warp 有空闲线程
#
# 5. 内存合并效率可能下降
#    → 当 m 不是 BLOCK_M 的倍数时，边界区域的访问模式不理想

# ====== 动态循环与静态循环的代码生成对比 ======

| 特性 | 静态形状循环 | 动态形状循环 |
|------|------------|------------|
| **循环边界** | 编译时常量 | 运行时参数 |
| **向量化** | 直接向量化 | 需要 LoopPartition + 尾部处理 |
| **循环展开** | 可以完全展开 | 只能部分展开（带运行时检查） |
| **常量传播** | 全面传播 | 受限（符号变量阻断传播） |
| **寄存器分配** | 精确分配 | 可能需要额外寄存器存储形状参数 |
| **代码大小** | 较小 | 可能较大（多个代码路径） |
| **缓存效率** | 可精确优化 | 需要参数化分块 |
| **调试难度** | 较低 | 较高（运行时行为不确定） |
```

---

## 32.20 动态形状的内存分配算法

### 32.20.1 静态与动态内存分配的对比

| 特性 | 静态形状分配 | 动态形状分配 |
|------|------------|------------|
| **分配时机** | 编译时确定 | 运行时确定 |
| **内存大小** | 编译时常量 | 形状参数的函数 |
| **分配策略** | 静态规划（StaticPlanBlockMemory） | 按需分配 / 内存池 |
| **复用分析** | 编译时生命周期分析 | 运行时生命周期追踪 |
| **碎片化** | 无（一次性规划） | 可能有碎片 |
| **性能** | 零分配开销 | 有分配开销 |
| **灵活性** | 只支持固定形状 | 支持任意形状 |

### 32.20.2 运行时 alloc_tensor 机制

在动态形状场景下，Relax VM 使用 `AllocTensor` 指令在运行时分配内存：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# Relax VM 中 AllocTensor 指令的执行流程
#
# 指令格式：AllocTensor reg_dst, reg_shape, dtype, storage_scope
#
# 执行过程：
#   1. 从寄存器读取形状值（运行时才知道的符号变量值）
#   2. 计算所需内存大小：product(shape) * dtype_size
#   3. 从内存池请求分配
#   4. 创建 NDArray 包装分配的内存
#   5. 将 NDArray 写入目标寄存器

# C++ 伪代码（src/relax/backend/vm/vm.cc）
def execute_alloc_tensor(inst):
    # 步骤 1：读取形状
    shape_values = []
    for i in range(inst.num_shape_values):
        shape_values.append(read_register(inst.shape_regs[i]))

    # 步骤 2：计算字节数
    num_elements = 1
    for dim in shape_values:
        num_elements *= dim
    num_bytes = num_elements * dtype_size(inst.dtype)

    # 步骤 3：从内存池分配
    if memory_pool.has_available(num_bytes, inst.dtype):
        storage = memory_pool.reuse(num_bytes, inst.dtype)
    else:
        storage = device_api.alloc(num_bytes, inst.device)

    # 步骤 4：创建 NDArray
    ndarray = NDArray(storage, shape_values, inst.dtype, inst.device)

    # 步骤 5：写入寄存器
    write_register(inst.dst_reg, ndarray)
```

### 32.20.3 内存池设计

Relax VM 的内存池使用 size-class 分桶策略来减少碎片化：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
class VMMemoryPool:
    """Relax VM 的内存池实现

    设计原则：
    1. 按大小类别分桶：256B, 1KB, 4KB, 16KB, 64KB, 256KB, 1MB, ...
    2. 每个桶维护一个空闲列表（free list）
    3. 分配时从最合适的桶中取一个空闲块
    4. 释放时将块放回对应的桶
    5. 如果桶中没有空闲块，向设备申请新的大块内存
    """

    def __init__(self, device):
        self.device = device
        # 每个 size_class 对应一个空闲列表
        # size_class 的大小是 2 的幂
        self.free_lists = {}  # size_class -> List[NDArray]
        self.allocated = []   # 所有已分配的大块内存
        self.stats = {
            "num_allocs": 0,
            "num_reuses": 0,
            "peak_memory": 0,
            "current_memory": 0,
        }

    def _get_size_class(self, num_bytes):
        """找到大于等于 num_bytes 的最小 2 的幂"""
        size_class = 256  # 最小 256 字节
        while size_class < num_bytes:
            size_class *= 2
        return size_class

    def alloc(self, shape, dtype):
        """分配指定形状和类型的张量"""
        from tvm import nd

        # 计算所需字节数
        num_elements = 1
        for dim in shape:
            num_elements *= dim
        dtype_size = {"float32": 4, "float16": 2, "int64": 8}[dtype]
        num_bytes = num_elements * dtype_size

        # 找到合适的 size class
        size_class = self._get_size_class(num_bytes)

        # 尝试从空闲列表中复用
        if size_class in self.free_lists and self.free_lists[size_class]:
            storage = self.free_lists[size_class].pop()
            self.stats["num_reuses"] += 1
        else:
            # 需要新的分配
            storage = nd.empty((size_class // dtype_size,), dtype, self.device)
            self.allocated.append(storage)

        # 创建视图（只使用前 num_elements 个元素）
        ndarray = nd.array(storage.numpy()[:num_elements].reshape(shape), self.device)

        self.stats["num_allocs"] += 1
        self.stats["current_memory"] += num_bytes
        self.stats["peak_memory"] = max(
            self.stats["peak_memory"], self.stats["current_memory"])

        return ndarray

    def free(self, ndarray):
        """释放张量，将其内存归还到空闲列表"""
        num_bytes = ndarray.numpy().nbytes
        size_class = self._get_size_class(num_bytes)

        if size_class not in self.free_lists:
            self.free_lists[size_class] = []

        self.free_lists[size_class].append(ndarray)
        self.stats["current_memory"] -= num_bytes
```

### 32.20.4 带形状范围的静态内存规划

如果已知动态形状的范围，可以在编译时进行更优化的内存规划：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm.relax.transform import StaticPlanBlockMemory

# 配置形状范围信息
shape_ranges = {
    "batch": (1, 64),          # batch 在 1-64 之间
    "seq_len": (16, 512),      # seq_len 在 16-512 之间
    "hidden_size": (768, 768), # hidden_size 固定为 768
}

# 使用最大形状进行内存规划
# 编译时分配最大形状的内存
# 运行时只使用有效区域
max_batch = shape_ranges["batch"][1]       # 64
max_seq_len = shape_ranges["seq_len"][1]   # 512
hidden_size = shape_ranges["hidden_size"][1]  # 768

# 编译时规划的内存布局：
# 张量 A: 最大形状 (64, 512, 768)，float32
#   → 分配 64 * 512 * 768 * 4 = 100,663,296 字节 ≈ 96 MB
# 张量 B: 最大形状 (64, 512, 768)，float32
#   → 与 A 共享内存（生命周期不重叠）
# 实际使用 batch=1, seq_len=128 时：
#   只访问前 1 * 128 * 768 = 98,304 个元素

@R.function
def max_shape_model(
    x: R.Tensor(("batch", "seq_len", 768), "float32"),
    w: R.Tensor((768, 3072), "float32"),
):
    with R.dataflow():
        # 中间张量使用最大形状分配
        lv0 = R.matmul(x, w)     # (batch, seq_len, 3072)
        lv1 = R.nn.gelu(lv0)     # (batch, seq_len, 3072)
        lv2 = R.matmul(lv1,      # (batch, seq_len, 768)
                       R.permute_dims(w, (1, 0)))
        R.output(lv2)
    return lv2
```

### 32.20.5 内存分配策略对比表

| 策略 | 分配时机 | 碎片化 | 分配开销 | 内存效率 | 适用场景 |
|------|---------|--------|---------|---------|---------|
| **按需分配** | 运行时每算子 | 高 | 高 | 低 | 原型验证 |
| **内存池** | 运行时复用 | 中 | 低 | 中 | 通用部署 |
| **最大形状预分配** | 编译时 | 无 | 零 | 低（可能浪费） | 形状范围已知 |
| **静态规划** | 编译时 | 无 | 零 | 高 | 静态形状 |
| **混合策略** | 编译+运行时 | 低 | 低 | 高 | 生产部署 |

### 32.20.6 内存复用的生命周期分析

在 DataflowBlock 中，编译器可以分析每个变量的使用范围，实现内存复用：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 内存复用示例
with R.dataflow():
    a = R.matmul(x, w1)       # a 的生命周期：第 1-3 行
    b = R.nn.relu(a)          # b 的生命周期：第 2-4 行
    c = R.matmul(b, w2)       # c 的生命周期：第 3-5 行
    d = R.nn.gelu(c)          # d 的生命周期：第 4-6 行
    R.output(d)

# 生命周期分析结果：
# 行  | 活跃变量    | 内存使用
#  1  | a           | sizeof(a)
#  2  | a, b        | sizeof(a) + sizeof(b)
#  3  | b, c        | sizeof(b) + sizeof(c)     ← a 可以释放
#  4  | c, d        | sizeof(c) + sizeof(d)     ← b 可以释放
#  5  | d           | sizeof(d)                 ← c 可以释放

# 内存复用优化：
# a 和 c 不重叠 → 可以共享同一块内存（M1）
# b 和 d 不重叠 → 可以共享同一块内存（M2）
# 最终：只需要 2 块内存，而不是 4 块
# 峰值内存从 4*sizeof(tensor) 降到 2*sizeof(tensor)
```


**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。

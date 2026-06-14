> **学习目标**：
> - 理解 TVM Unity 的设计愿景：统一 Relay、TE、TIR 三层 IR 为一个连贯的编译栈
> - 掌握 Relax IR 的核心数据结构（Expr、StructInfo、Function、BlockBuilder）
> - 理解 Relax VM 的字节码设计与执行模型
> - 对比 Relay 与 Relax 在算子表示、动态形状、端到端编译上的差异
> - 能够使用 Relax API 构建并编译一个完整的模型

---

## 31.1 TVM Unity 的设计愿景

### 31.1.1 为什么需要 Unity？

TVM 在发展过程中形成了三层 IR 体系：**Relay**（计算图级）、**TE**（算子级调度描述）、**TIR**（低层循环 IR）。这三层 IR 各自独立演化，带来了显著的工程痛点：

| 痛点 | 描述 |
|------|------|
| **IR 间语义鸿沟** | Relay 的 `nn.conv2d` → TE 的 `te.compute` → TIR 的 `tir.For` 是三次独立的 lowering，中间信息大量丢失 |
| **重复基础设施** | Relay 和 TIR 各自维护一套 Pass 框架、类型系统、序列化机制 |
| **动态形状处理困难** | Relay 的类型系统假设静态形状，动态维度需要特殊处理路径 |
| **优化难以跨层** | 图级优化（如算子融合）无法直接感知算子内部的调度细节 |

TVM Unity（RFC）的核心思想是：**将所有 IR 统一到一个连贯的框架下，消除层次间的语义断层**。

```
┌──────────────────────────────────────────────────────────┐
│                    TVM Unity 统一视图                      │
│                                                          │
│   ┌─────────┐    ┌──────────┐    ┌──────────┐           │
│   │  Relax   │───→│   TIR    │───→│ Codegen  │           │
│   │ (图+算子) │    │ (循环IR)  │    │ (目标代码) │           │
│   └─────────┘    └──────────┘    └──────────┘           │
│        ↑                                              │
│   TE 作为调度原语嵌入 Relax，不再作为独立 IR 层          │
└──────────────────────────────────────────────────────────┘
```

<div data-component="TVMUnityArchitectureDiagram"></div>

### 31.1.2 Unity 的三大设计原则

**原则一：全程序 IR（Whole-program IR）**

Relax 将计算图表示和算子调度统一到同一个 IR 中。一个 Relax Function 可以同时包含图级的算子调用和低层的 TIR 计算定义：

```python
# Relax 中一个完整的函数可以包含图级逻辑和算子定义
@tvm.script.ir_module
class MyModule:
    @R.function
    def main(x: R.Tensor((128, 256), "float32"),
             w: R.Tensor((512, 256), "float32")):
        # 图级算子调用
        lv0 = R.matmul(x, w)
        lv1 = R.nn.relu(lv0)
        return lv1
```

**原则二：渐进式 Lowering（Progressive Lowering）**

不再一次性从 Relay 跳到 TE 再到 TIR，而是在 Relax 层面逐步 lower：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
高层 Relax（算子调用）
    ↓ LegalizeOps + FuseOps
中层 Relax（融合后的算子 + TIR 定义）
    ↓ TIR 调度优化
低层 TIR（循环级代码）
    ↓ CodeGen
目标代码
```

**原则三：First-class 动态形状支持**

Symbolic Shape 在 Relax 中是原生支持，不再需要 Relay 中的 `Any` 类型 hack。

### 31.1.3 Unity 的源码组织

TVM Unity 的代码主要分布在以下目录：

| 目录 | 内容 |
|------|------|
| `src/relax/ir/` | Relax IR 核心数据结构（Expr、StructInfo、Module） |
| `src/relax/transform/` | Relax 变换 Pass |
| `src/relax/backend/` | Relax VM 后端与代码生成 |
| `python/tvm/relax/` | Python API 封装 |
| `include/tvm/relax/` | C++ 头文件定义 |
| `src/relax/op/` | Relax 算子定义（tensor、nn、memory 等） |

---

## 31.2 Relax IR 核心数据结构

### 31.2.1 表达式层次体系（Expr Hierarchy）

Relax 的表达式体系定义在 `include/tvm/relax/expr.h` 中，继承自 `RelaxExpr`：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
RelaxExpr
├── Expr
│   ├── Var                    # 变量
│   ├── GlobalVar              # 全局函数引用
│   ├── Function               # 函数定义
│   ├── Call                   # 算子/函数调用
│   ├── SeqExpr                # 序列表达式（Let 链）
│   ├── If                     # 条件分支
│   ├── Tuple                  # 元组构造
│   ├── TupleGetItem           # 元组索引
│   ├── ShapeExpr              # 形状表达式
│   ├── ExternFunc             # 外部函数引用
│   └── Constant               # 常量张量
└── StructInfo                 # 结构化类型信息
    ├── TensorStructInfo       # 张量结构信息
    ├── TupleStructInfo        # 元组结构信息
    ├── ShapeStructInfo        # 形状结构信息
    ├── FuncStructInfo         # 函数结构信息
    └── ObjectStructInfo       # 通用对象
```

### 31.2.2 Var 与 GlobalVar

`Var` 是 Relax 中最基本的表达式，代表一个局部变量绑定：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# Python API
from tvm import relax

# 创建一个类型为 (128, 256) float32 张量的变量
x = relax.Var("x", relax.TensorStructInfo((128, 256), "float32"))
```

对应的 C++ 定义在 `include/tvm/relax/expr.h`：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
class Var : public RelaxExpr {
 public:
  /*! \brief The hint name of the variable */
  String name_hint();
  /*! \brief The struct info of the variable */
  Optional<StructInfo> struct_info_;
};
```

`GlobalVar` 用于引用 IRModule 中定义的全局函数：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# GlobalVar 通常由 IRModule 管理
mod = tvm.IRModule()
gv = relax.GlobalVar("my_func")
# 向模块添加函数
mod[gv] = relax.Function(params=[x, w], body=body, ret_struct_info=ret_sinfo)
```

### 31.2.3 Call 表达式

`Call` 是 Relax 中最常用的表达式，表示对算子或函数的调用：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 使用 op 调用内置算子
x = relax.Var("x", relax.TensorStructInfo((1, 3, 224, 224), "float32"))
w = relax.Var("w", relax.TensorStructInfo((64, 3, 7, 7), "float32"))

# 调用 conv2d 算子
conv_out = relax.op.nn.conv2d(x, w, strides=(1, 1), padding=(3, 3))
```

Call 的 C++ 结构：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
class Call : public RelaxExpr {
 public:
  /*! \brief The operator or function being called */
  Expr op;
  /*! \brief The arguments */
  Array<Expr> args;
  /*! \brief Optional struct info for the return value */
  Optional<StructInfo> struct_info_;
};
```

### 31.2.4 Function 定义

Relax 的 `Function` 是整个 IR 的核心抽象，它统一了图级函数和算子级函数：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
@tvm.script.ir_module
class MLP:
    @R.function
    def main(x: R.Tensor((1, 784), "float32"),
             w1: R.Tensor((784, 256), "float32"),
             b1: R.Tensor((256,), "float32"),
             w2: R.Tensor((256, 10), "float32"),
             b2: R.Tensor((10,), "float32")):
        with R.dataflow():
            lv0 = R.matmul(x, w1)
            lv1 = R.add(lv0, b1)
            lv2 = R.nn.relu(lv1)
            lv3 = R.matmul(lv2, w2)
            lv4 = R.add(lv3, b2)
            R.output(lv4)
        return lv4
```

Function 的关键属性：

| 属性 | 类型 | 说明 |
|------|------|------|
| `params` | `Array<Var>` | 函数参数列表 |
| `body` | `Expr` | 函数体 |
| `ret_struct_info` | `StructInfo` | 返回值的结构信息 |
| `is_pure` | `bool` | 是否为纯函数 |
| `attrs` | `DictAttrs` | 函数属性 |

### 31.2.5 SeqExpr 与 DataflowBlock

Relax 使用 `SeqExpr` 来表示一系列按顺序执行的表达式，类似于 Relay 的 `Let` 绑定链。但在 Dataflow 模式下，使用 `DataflowBlock` 来标记纯计算区域：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# DataflowBlock 内的绑定必须是纯计算
# 这允许编译器进行更激进的优化
with R.dataflow():
    lv0 = R.matmul(x, w)        # 纯计算
    lv1 = R.nn.gelu(lv0)        # 纯计算
    R.output(lv1)               # 标记输出变量
```

`DataflowBlock` 的语义：
- 块内的所有变量绑定都是纯的（无副作用）
- 编译器可以自由地重排、融合块内的计算
- `R.output()` 标记哪些变量会被块外引用

---

## 31.3 StructInfo：结构化类型系统

### 31.3.1 StructInfo vs Relay Type

Relay 使用传统的类型系统（`TensorType`、`FuncType`），而 Relax 引入了 `StructInfo` 作为类型/形状信息的载体。关键区别：

| 特性 | Relay Type | Relax StructInfo |
|------|-----------|-----------------|
| 形状表示 | 静态 `IndexExpr` 或 `Any` | 支持符号变量 `Var` 和运行时求值 |
| 层次 | 类型系统的一部分 | 独立于类型系统，更灵活 |
| VM 集成 | 需要独立的形状推断 | 直接驱动 VM 的内存分配 |
| 信息量 | 只有 dtype + shape | 可包含 device、layout 等额外信息 |

### 31.3.2 TensorStructInfo

`TensorStructInfo` 是最常用的 StructInfo，描述张量的形状和数据类型：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 静态形状
sinfo = relax.TensorStructInfo((128, 256), "float32")

# 动态形状：使用符号变量
m = tir.Var("m", "int64")
n = tir.Var("n", "int64")
sinfo_dyn = relax.TensorStructInfo((m, n), "float32")

# 未知秩（rank-agnostic）
sinfo_any = relax.TensorStructInfo(dtype="float32", ndim=-1)
```

C++ 定义在 `include/tvm/relax/struct_info.h`：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
class TensorStructInfo : public StructInfo {
 public:
  /*! \brief The shape expression, can be symbolic */
  Optional<Expr> shape;
  /*! \brief The data type */
  runtime::DataType dtype;
  /*! \brief The number of dimensions, -1 means unknown */
  int ndim;
};
```

### 31.3.3 ShapeStructInfo

`ShapeStructInfo` 用于描述形状值本身（而不是张量）：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 描述一个表示形状的值
shape_sinfo = relax.ShapeStructInfo([m, n])
```

### 31.3.4 FuncStructInfo

`FuncStructInfo` 描述函数的参数和返回值结构信息：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 描述一个函数的结构信息
param_sinfo = [relax.TensorStructInfo((128, 256), "float32")]
ret_sinfo = relax.TensorStructInfo((128, 512), "float32")
func_sinfo = relax.FuncStructInfo(param_sinfo, ret_sinfo)
```

### 31.3.5 StructInfo 推断

Relax 提供了 `StructInfoInfer` Pass，自动从表达式推断 StructInfo：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm.relax.transform import InferStructInfo

# 对整个模块进行 StructInfo 推断
mod = InferStructInfo()(mod)
```

推断规则定义在 `src/relax/op/op_common.h` 中的 `FInferStructInfo` 注册函数：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// 每个 Relax 算子注册一个 FInferStructInfo 函数
TVM_REGISTER_OP("relax.matmul")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoMatmul);
```

---

## 31.4 BlockBuilder：程序构建 API

### 31.4.1 BlockBuilder 的设计动机

手动构造 Relax IR 表达式树非常繁琐。`BlockBuilder` 提供了一个命令式的 API 来构建 Relax 函数和模块：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import relax

# 创建 BlockBuilder
bb = relax.BlockBuilder()

# 开始定义一个函数
x = relax.Var("x", relax.TensorStructInfo((1, 784), "float32"))
w = relax.Var("w", relax.TensorStructInfo((784, 256), "float32"))
bb.emit_func_output(relax.op.matmul(x, w))

# 获取构建好的模块
mod = bb.get()
```

### 31.4.2 BlockBuilder 核心 API

BlockBuilder 定义在 `python/tvm/relax/block_builder.py`，核心方法：

| 方法 | 说明 |
|------|------|
| `emit(expr)` | 在当前 DataflowBlock 中发射一个表达式，返回绑定变量 |
| `emit_output(expr)` | 标记 DataflowBlock 的输出 |
| `emit_expr(expr, name_hint)` | 发射表达式并返回带名称的变量 |
| `normalize(expr)` | 规范化表达式（确保变量绑定正确） |
| `lookup_binding(var)` | 查找变量的绑定定义 |
| `begin_dataflow()` | 开始一个新的 DataflowBlock |
| `end_dataflow()` | 结束当前 DataflowBlock |
| `begin_function(name)` | 开始一个新的 Function |
| `emit_func_output(output)` | 完成函数定义并输出 |
| `get()` | 获取构建好的 IRModule |

### 31.4.3 使用 BlockBuilder 构建复杂模型

下面展示使用 BlockBuilder 构建一个简单的 MLP：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import tvm
from tvm import relax, tir
import numpy as np

def build_mlp():
    bb = relax.BlockBuilder()

    # 定义参数形状
    x_shape = (1, 784)
    w1_shape = (784, 256)
    b1_shape = (256,)
    w2_shape = (256, 10)
    b2_shape = (10,)

    with bb.function("mlp", params=[]):
        # 声明输入
        x = relax.Var("x", relax.TensorStructInfo(x_shape, "float32"))
        w1 = relax.Var("w1", relax.TensorStructInfo(w1_shape, "float32"))
        b1 = relax.Var("b1", relax.TensorStructInfo(b1_shape, "float32"))
        w2 = relax.Var("w2", relax.TensorStructInfo(w2_shape, "float32"))
        b2 = relax.Var("b2", relax.TensorStructInfo(b2_shape, "float32"))

        with bb.dataflow():
            # 第一层：matmul + bias + relu
            lv0 = bb.emit(relax.op.matmul(x, w1))
            lv1 = bb.emit(relax.op.add(lv0, b1))
            lv2 = bb.emit(relax.op.nn.relu(lv1))

            # 第二层：matmul + bias
            lv3 = bb.emit(relax.op.matmul(lv2, w2))
            lv4 = bb.emit(relax.op.add(lv3, b2))

            # 标记输出
            bb.emit_output(lv4)

        # 完成函数定义
        bb.emit_func_output(lv4)

    return bb.get()

mod = build_mlp()
print(mod)
```

### 31.4.4 BlockBuilder 的内部状态机

BlockBuilder 维护一个内部状态来跟踪当前的构建上下文：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
BlockBuilder 状态：
├── current_builder_state_
│   ├── current_block_stmts_    # 当前 DataflowBlock 中的语句
│   ├── current_function_       # 正在构建的函数
│   └── binding_table_          # 变量绑定表
├── functions_                  # 已完成的函数集合
└── mod_                        # 最终的 IRModule
```

`normalize()` 方法是 BlockBuilder 的核心内部方法，确保所有子表达式都被正确绑定为变量：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# normalize 会将嵌套的 Call 展平为一系列 Let 绑定
# 例如：
#   R.add(R.matmul(x, w), b)
# 会被 normalize 为：
#   lv0 = R.matmul(x, w)
#   lv1 = R.add(lv0, b)
```

---

## 31.5 Relax 算子体系

### 31.5.1 算子注册

Relax 算子定义在 `src/relax/op/` 目录下，按功能分类：

| 子目录 | 内容 | 示例算子 |
|--------|------|---------|
| `tensor/` | 张量操作 | `matmul`, `add`, `multiply`, `reshape` |
| `nn/` | 神经网络 | `conv2d`, `linear`, `softmax`, `layer_norm` |
| `memory/` | 内存操作 | `alloc_tensor`, `reshape` |
| `image/` | 图像操作 | `resize2d` |
| `statistical/` | 统计操作 | `mean`, `variance`, `sum` |

算子通过 `TVM_REGISTER_OP` 宏注册：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// src/relax/op/tensor/binary.cc
TVM_REGISTER_OP("relax.add")
    .set_num_inputs(2)
    .add_argument("lhs", "Tensor", "The left tensor.")
    .add_argument("rhs", "Tensor", "The right tensor.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoBinaryBroadcast)
    .set_attr<Bool>("FPurity", Bool(true));
```

### 31.5.2 StructInfo 推断规则

每个 Relax 算子都必须注册一个 `FInferStructInfo` 函数，用于推断输出的 StructInfo：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// 以 matmul 为例
StructInfo InferStructInfoMatmul(const Call& call, const BlockBuilder& ctx) {
  TensorStructInfo lhs_sinfo = GetStructInfoAs<TensorStructInfo>(call->args[0]);
  TensorStructInfo rhs_sinfo = GetStructInfoAs<TensorStructInfo>(call->args[1]);

  // 验证维度匹配
  // M, K x K, N -> M, N
  DataType out_dtype = lhs_sinfo->dtype;
  ShapeExpr out_shape = ...;  // 计算输出形状
  return TensorStructInfo(out_shape, out_dtype);
}
```

### 31.5.3 自定义 Relax 算子

用户可以通过 Python API 注册自定义的 Relax 算子：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import relax
from tvm.relax.op import register_custom_op

@register_custom_op("relax.my_gelu", level=15)
def my_gelu(x: relax.Expr) -> relax.Expr:
    """自定义 GELU 激活函数"""
    return relax.call_pure_packed("my_gelu_impl", x, sinfo_args=x.struct_info)
```

---

## 31.6 Relax 变换 Pass

### 31.6.1 Pass 框架

Relax 的 Pass 框架与 Relay 类似，但有一些关键改进。Pass 基类定义在 `include/tvm/relax/transform.h`：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
class Pass : public PassInfo {
 public:
  virtual IRModule operator()(IRModule mod, const PassContext& ctx) const = 0;
};
```

### 31.6.2 核心变换 Pass

Relax 提供了一系列核心变换 Pass，定义在 `src/relax/transform/`：

| Pass | 源码位置 | 功能 |
|------|---------|------|
| `FuseOpsByPattern` | `fuse_ops_by_pattern.cc` | 基于模式匹配的算子融合 |
| `LegalizeOps` | `legalize_ops.cc` | 将 Relax 算子 lower 为 TIR |
| `CallTIRRewrite` | `call_tir_rewrite.cc` | 处理 call_tir 调用 |
| `DeadCodeElimination` | `dead_code_elimination.cc` | 死代码消除 |
| `Normalize` | `normalize.cc` | 规范化 IR 表示 |
| `FoldConstant` | `fold_constant.cc` | 常量折叠 |
| `LowerAllocTensor` | `lower_alloc_tensor.cc` | 内存分配 lowering |
| `BindParams` | `bind_params.cc` | 绑定模型参数 |
| `RemoveUnusedOutputs` | `remove_unused_outputs.cc` | 移除未使用的输出 |
| `RunCodegen` | `run_codegen.cc` | 调用外部 codegen |

### 31.6.3 LegalizeOps：从 Relax 到 TIR

`LegalizeOps` 是最关键的 Pass 之一，它将高层的 Relax 算子调用转换为 TIR 函数调用：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm.relax.transform import LegalizeOps

# LegalizeOps 会将：
#   lv0 = R.matmul(x, w)
# 转换为：
#   lv0 = R.call_tir(matmul_tir_func, (x, w), (128, 512), "float32")
mod = LegalizeOps()(mod)
```

LegalizeOps 为每个 Relax 算子查找对应的 TE 实现，然后将 TE 调度为 TIR：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
R.nn.conv2d(x, w)
    ↓ LegalizeOps
R.call_tir(conv2d_tir_func, (x, w), out_shape, out_dtype)
    ↓ TIR 内部
TIR PrimFunc（包含循环、内存访问等低层细节）
```

### 31.6.4 FuseOpsByPattern：基于模式的融合

`FuseOpsByPattern` 使用预定义的模式来匹配和融合常见的算子组合：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm.relax.transform import FuseOpsByPattern
from tvm.relax import fusion_pattern

# 定义融合模式
patterns = [
    fusion_pattern("relax.nn.conv2d", "relax.nn.relu"),
    fusion_pattern("relax.matmul", "relax.add", "relax.nn.gelu"),
]

mod = FuseOpsByPattern(patterns)(mod)
```

### 31.6.5 组合 Pass Pipeline

一个典型的 Relax 编译管线如下：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm.relax import transform

pipeline = tvm.transform.Sequential([
    transform.InferStructInfo(),          # 推断结构信息
    transform.FuseOpsByPattern(patterns), # 算子融合
    transform.LegalizeOps(),             # Lower 到 TIR
    transform.CallTIRRewrite(),          # 处理 call_tir
    transform.LowerAllocTensor(),        # Lower 内存分配
    transform.Normalize(),               # 规范化
    transform.DeadCodeElimination(),     # 死代码消除
])

mod = pipeline(mod)
```

---

## 31.7 Relax VM：虚拟机执行

### 31.7.1 Relax VM 的设计

Relax VM 是 TVM Unity 中的执行引擎，类似于 Relay VM 但有一些关键改进。VM 定义在 `src/relax/backend/vm/`：

| 文件 | 内容 |
|------|------|
| `vm.cc` | VM 主循环与执行引擎 |
| `vm_compiler.cc` | Relax→VM 字节码编译器 |
| `exec_builder.cc` | 可执行文件构建器 |
| `bytecode.h` | 字节码指令定义 |

### 31.7.2 VM 指令集

Relax VM 使用一套基于寄存器的字节码指令集，定义在 `include/tvm/relax/backend/vm/bytecode.h`：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
enum class Opcode : int32_t {
  kMove = 0,
  kCall = 1,
  kCallPacked = 2,
  kCallTIR = 3,
  kAllocTensor = 4,
  kAllocTensorReg = 5,
  kShapeOf = 6,
  kTuple = 7,
  kTupleGetItem = 8,
  kIf = 9,
  kGoto = 10,
  kInvokePacked = 11,
  kLoadConst = 12,
  kAllocStorage = 13,
  kAllocClosure = 14,
  kAssertOp = 15,
  kFatal = 16,
  kRetValue = 17,
};
```

每条指令的格式：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
┌─────────┬─────────┬──────────┬──────────┐
│  Opcode │  Reg A  │  Reg B   │  Reg C   │
│  (32b)  │  (32b)  │  (32b)   │  (32b)   │
└─────────┴─────────┴──────────┴──────────┘
```

### 31.7.3 VM 编译器

`VMCompiler` 将 Relax IR 编译为 VM 字节码：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import relax
from tvm.relax.backend.vm import VMCompiler

# 编译为 VM 可执行文件
compiler = VMCompiler()
exec = compiler.compile(mod, target="llvm")
```

编译过程分为几个阶段：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
Relax IR Module
    ↓ Normalize + FuseOps
融合后的 Relax Module
    ↓ LegalizeOps
包含 TIR 定义的 Relax Module
    ↓ LowerToVM
VM Executable（字节码 + TIR 函数）
```

### 31.7.4 VM 执行流程

VM 的执行入口在 `src/relax/backend/vm/vm.cc`：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
class VirtualMachine {
 public:
  void Init(const VMExec& exec, Device device);
  ObjectRef Invoke(const String& func_name, ...args);

 private:
  // 指令分发循环
  void RunLoop();
  // 各指令的执行函数
  void ExecuteCall(Instruction& inst);
  void ExecuteCallTIR(Instruction& inst);
  void ExecuteAllocTensor(Instruction& inst);
  // ...
};
```

执行一个 Relax 函数的完整流程：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import tvm
from tvm import relax
import numpy as np

# 1. 准备模型和数据
mod = build_mlp()  # 假设已构建好
x_np = np.random.randn(1, 784).astype("float32")

# 2. 编译
ex = relax.vm.build(mod, target="llvm")

# 3. 创建 VM
vm = relax.VirtualMachine(ex, tvm.cpu())

# 4. 执行
result = vm["main"](tvm.nd.array(x_np))
print(result)
```

### 31.7.5 VM 内存管理

Relax VM 使用 `DeviceAPI` 进行内存管理，支持以下内存分配策略：

| 策略 | 说明 |
|------|------|
| **按需分配** | 每次 `AllocTensor` 指令执行时分配新内存 |
| **内存池复用** | VM 维护一个内存池，重用已释放的内存 |
| **静态规划** | 编译时分析生命周期，预先规划内存布局 |

---

## 31.8 Relax 与 TIR 的交互

### 31.8.1 call_tir 机制

Relax 通过 `call_tir` 原语调用 TIR 函数：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 在 Relax 中调用一个 TIR 函数
@T.prim_func
def matmul_tir(A: T.Buffer((128, 256), "float32"),
               B: T.Buffer((256, 512), "float32"),
               C: T.Buffer((128, 512), "float32")):
    for i, j, k in T.grid(128, 512, 256):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

# 在 Relax 中使用
@R.function
def main(x: R.Tensor((128, 256), "float32"),
         w: R.Tensor((256, 512), "float32")):
    with R.dataflow():
        lv0 = R.call_tir(matmul_tir, (x, w),
                         out_sinfo=R.TensorStructInfo((128, 512), "float32"))
        R.output(lv0)
    return lv0
```

### 31.8.2 TIR 函数的封装

`call_tir` 在 VM 编译时会被编译为 `CallTIR` 指令，VM 执行时：

1. 从寄存器读取输入张量
2. 设置 TIR 函数的 Buffer 参数
3. 调用编译好的 TIR kernel
4. 将输出写回寄存器

---

## 31.9 Relax 的端到端编译示例

### 31.9.1 从 PyTorch 导入到 Relax

TVM 提供了从 PyTorch 模型导入到 Relax 的路径：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import tvm
from tvm import relax
from tvm.relax.frontend import from_pytorch

import torch
import torchvision

# 加载 PyTorch 模型
model = torchvision.models.resnet18()
model.eval()

# 转换为 Relax IR
input_info = [(("input",), (1, 3, 224, 224), "float32")]
mod = from_pytorch(model, input_info)
print(mod)
```

### 31.9.2 端到端编译流程



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm.relax.transform import FuseOpsByPattern, LegalizeOps
import tvm.transform

# 定义编译管线
target = tvm.target.Target("llvm -mcpu=core-avx2")

with target:
    pipeline = tvm.transform.Sequential([
        relax.transform.FuseOpsByPattern(),    # 算子融合
        relax.transform.LegalizeOps(),          # Lower 到 TIR
        relax.transform.CallTIRRewrite(),       # 处理 call_tir
        relax.transform.LowerAllocTensor(),     # 内存分配
        relax.transform.RunCodegen(target),     # 代码生成
    ])

    mod = pipeline(mod)

# 编译并执行
ex = relax.vm.build(mod, target)
vm = relax.VirtualMachine(ex, tvm.device("cpu", 0))

# 运行推理
input_data = tvm.nd.array(np.random.randn(1, 3, 224, 224).astype("float32"))
output = vm["main"](input_data)
```

### 31.9.3 Relax 的常量折叠

Relax 的常量折叠在 `src/relax/transform/fold_constant.cc` 中实现，比 Relay 版本更强大：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# Relax 可以折叠包含 TIR 函数调用的常量表达式
@R.function
def main():
    with R.dataflow():
        # 常量张量的 matmul 可以在编译时计算
        a = R.const(np.ones((4, 4)).astype("float32"))
        b = R.const(np.ones((4, 4)).astype("float32"))
        c = R.matmul(a, b)  # 编译时计算为 4.0
        R.output(c)
    return c
```

---

## 31.10 Relax 的调试与可视化

### 31.10.1 IR 打印

Relax IR 支持以 TVMScript 格式打印：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 打印整个模块
print(mod)

# 打印单个函数
print(mod["main"])

# 以结构化格式打印
print(relax.analysis.struct_info(mod["main"]))
```

### 31.10.2 IR 结构检查



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm.relax.analysis import verify_struct_info, check Wellformedness

# 检查模块的结构信息一致性
verify_struct_info(mod)

# 检查模块是否 well-formed
assert relax.analysis.well_formed(mod)
```

### 31.10.3 调试 Pass

在编译管线中插入调试步骤：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import transform

@transform.module_pass(opt_level=0)
def PrintIR(mod, ctx):
    """打印当前 IR 的 Pass"""
    print("=" * 60)
    print(mod)
    print("=" * 60)
    return mod

pipeline = tvm.transform.Sequential([
    relax.transform.InferStructInfo(),
    PrintIR(),  # 查看 InferStructInfo 之后的 IR
    relax.transform.FuseOpsByPattern(),
    PrintIR(),  # 查看融合之后的 IR
    relax.transform.LegalizeOps(),
    PrintIR(),  # 查看 LegalizeOps 之后的 IR
])
```

---

## 31.11 Relax 的设计模式与最佳实践

### 31.11.1 模块化设计

推荐将模型拆分为多个 Relax 函数，每个函数对应一个逻辑模块：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
@tvm.script.ir_module
class VisionModel:
    @R.function
    def backbone(x: R.Tensor((1, 3, 224, 224), "float32")):
        # ResNet backbone 逻辑
        ...

    @R.function
    def head(features: R.Tensor((1, 512), "float32")):
        # 分类头逻辑
        ...

    @R.function
    def main(x: R.Tensor((1, 3, 224, 224), "float32")):
        features = VisionModel.backbone(x)
        logits = VisionModel.head(features)
        return logits
```

### 31.11.2 动态形状的正确处理

在 Relax 中处理动态形状时，需要使用符号变量：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
m = tir.Var("m", "int64")  # 批大小为符号变量

@R.function
def main(x: R.Tensor((m, 784), "float32"),
         w: R.Tensor((784, 256), "float32")):
    with R.dataflow():
        lv0 = R.matmul(x, w)  # 输出形状自动推断为 (m, 256)
        R.output(lv0)
    return lv0
```

### 31.11.3 性能优化建议

| 建议 | 说明 |
|------|------|
| 使用 DataflowBlock | 尽可能将计算放入 `with R.dataflow()` 中 |
| 避免过度细分 | 不要为每个小操作创建单独的函数 |
| 合理使用 call_tir | 对性能关键的算子，直接提供 TIR 实现 |
| 利用常量折叠 | 在输入模型时预计算不变的张量 |

---

## 31.12 Relax VM 可执行文件格式

### 31.12.1 VM Executable 结构

Relax VM 的可执行文件（Executable）包含编译后的所有信息，定义在 `include/tvm/relax/backend/vm/executable.h`：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
class VMExecutable : public Object {
 public:
  /*! \brief The module containing all functions */
  IRModule mod;
  /*! \brief The bytecode for each function */
  Map<String, Bytecode> bytecodes;
  /*! \brief The constant pool (pre-computed tensors) */
  Array<ObjectRef> constants;
  /*! \brief The primitive functions (compiled TIR kernels) */
  Map<String, PackedFunc> primitive_funcs;
  /*! \brief Global section for metadata */
  Map<String, ObjectRef> metadata;
};
```

Executable 的序列化格式：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
┌──────────────────────────────────────────────┐
│ Header                                        │
│  - magic number                               │
│  - version                                    │
│  - target info                                │
├──────────────────────────────────────────────┤
│ Constants Section                             │
│  - 序列化的 NDArray（模型权重等）              │
├──────────────────────────────────────────────┤
│ Functions Section                             │
│  - 每个函数的字节码                            │
│  - 函数元数据（参数数量、寄存器数等）           │
├──────────────────────────────────────────────┤
│ Primitive Functions Section                   │
│  - 编译好的 TIR kernel（SO/PTX 等）           │
├──────────────────────────────────────────────┤
│ Metadata Section                              │
│  - 输入输出形状信息                            │
│  - 设备信息                                   │
└──────────────────────────────────────────────┘
```

### 31.12.2 字节码序列化

VM 字节码可以序列化为二进制格式，用于部署和分发：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import relax

# 编译模型
ex = relax.vm.build(mod, target="llvm")

# 导出为共享库
ex.export_library("model.so")

# 导出为 Relax VM 格式
ex.mod.export_library("model.ro")

# 在另一个进程中加载
loaded_ex = relax.vm.Executable.load("model.ro")
vm = relax.VirtualMachine(loaded_ex, tvm.cpu())
```

### 31.12.3 VM 的函数调用约定

Relax VM 使用寄存器传递参数，函数调用约定定义在 `src/relax/backend/vm/vm.cc`：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
函数调用流程：
1. 将参数从调用者的寄存器复制到被调用者的寄存器
2. 跳转到被调用函数的字节码起始位置
3. 执行函数体
4. 将返回值写入调用者的返回寄存器
5. 跳回调用点的下一条指令

寄存器分配：
- 每个函数有独立的寄存器空间
- 寄存器数量在编译时确定
- 临时变量使用虚拟寄存器，编译时映射到物理寄存器
```

### 31.12.4 常量池优化

Relax VM 使用常量池来高效管理模型权重等常量数据：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 常量池的工作方式：

# 1. 编译时：提取所有常量
#    R.const(np.ones((64, 3, 7, 7)).astype("float32"))
#    → 存入常量池，生成 LoadConst 指令

# 2. 执行时：从常量池加载
#    LoadConst reg_0, const_pool_idx_0
#    → 直接引用常量池中的 NDArray，无需复制

# 3. 多次引用同一常量：
#    LoadConst reg_1, const_pool_idx_0  # 同一常量
#    → 共享同一 NDArray 对象
```

---

## 31.13 Relax 的内存规划

### 31.13.1 Relax 的内存分配策略

Relax VM 在编译时进行内存规划，决定哪些张量可以共享内存：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm.relax.transform import MemoryPlanning

# MemoryPlanning Pass 分析张量的生命周期，优化内存分配
mod = MemoryPlanning()(mod)
```

内存规划的核心算法在 `src/relax/transform/memory_planning.cc` 中实现：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
输入：一系列张量操作及其生命周期
输出：每个张量的内存分配方案

算法：
1. 构建生命周期图（Lifetime Graph）
2. 使用图着色算法找到最小内存使用
3. 将不冲突的张量分配到同一内存区域
4. 插入显式的 AllocTensor 和 KillTensor 指令
```

### 31.13.2 Inplace 算子优化

对于 inplace 操作（如 relu、add），Relax 可以复用输入的内存：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 原始代码
with R.dataflow():
    lv0 = R.matmul(x, w)      # 输出：新分配的张量 A
    lv1 = R.nn.relu(lv0)       # 输出：新分配的张量 B
    R.output(lv1)

# Inplace 优化后
with R.dataflow():
    lv0 = R.matmul(x, w)      # 输出：张量 A
    lv1 = R.nn.relu(lv0)       # 输出：复用张量 A 的内存（inplace）
    R.output(lv1)              # 返回的是同一块内存
```

### 31.13.3 内存规划与 DataflowBlock

`DataflowBlock` 的纯计算语义使得内存规划更加高效：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# DataflowBlock 中的变量绑定都是纯的
# 这意味着：
# 1. 编译器可以自由重排计算顺序
# 2. 可以激进地进行内存复用
# 3. 可以消除未使用的中间变量

with R.dataflow():
    a = R.matmul(x, w1)       # a 的生命周期
    b = R.nn.relu(a)          # b 的生命周期
    c = R.matmul(b, w2)       # c 的生命周期
    # 如果只有 c 被输出，a 和 b 可以共享内存
    R.output(c)
```

---

## 31.14 Relax 与 Relay 的全面对比

### 31.14.1 设计哲学对比

| 维度 | Relay | Relax |
|------|-------|-------|
| **设计理念** | 函数式编程风格 | 混合命令式/函数式 |
| **IR 层次** | 图级 IR，与 TE/TIR 分离 | 统一 IR，内嵌 TIR |
| **类型系统** | 传统类型系统 | StructInfo 结构化信息 |
| **动态形状** | Any 类型 hack | 原生符号变量 |
| **优化方式** | 图级 Pass + 算子级 TE | 图级 Pass + TIR Pass 统一 |
| **执行模式** | GraphRuntime / VM | VM 为主 |
| **社区状态** | 维护模式 | 活跃开发 |

### 31.14.2 语法对比



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# ========== Relay 语法 ==========
import tvm.relay as relay

# 定义变量
x = relay.var("x", relay.TensorType([1, 64], "float32"))
w = relay.var("w", relay.TensorType([64, 128], "float32"))

# 定义计算
y = relay.nn.dense(x, w)
z = relay.nn.relu(y)

# 定义函数
func = relay.Function([x, w], z)
mod = tvm.IRModule.from_expr(func)

# ========== Relax 语法 ==========
import tvm.relax as relax

# 使用 TVMScript 定义
@tvm.script.ir_module
class MyModule:
    @R.function
    def main(x: R.Tensor((1, 64), "float32"),
             w: R.Tensor((64, 128), "float32")):
        with R.dataflow():
            lv0 = R.matmul(x, w)
            lv1 = R.nn.relu(lv0)
            R.output(lv1)
        return lv1

# 或使用 BlockBuilder
bb = relax.BlockBuilder()
with bb.function("main"):
    x = relax.Var("x", relax.TensorStructInfo((1, 64), "float32"))
    w = relax.Var("w", relax.TensorStructInfo((64, 128), "float32"))
    with bb.dataflow():
        lv0 = bb.emit(relax.op.matmul(x, w))
        lv1 = bb.emit(relax.op.nn.relu(lv0))
        bb.emit_output(lv1)
    bb.emit_func_output(lv1)
mod = bb.get()
```

### 31.14.3 编译管线对比



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
Relay 编译管线：
  Relay IR
    ↓ InferType（类型推断）
    ↓ FuseOps（算子融合）
    ↓ ToANormalForm（规范化）
    ↓ LowerTE（Lower 到 TE）
    ↓ AutoTVM/MetaSchedule（调度搜索）
    ↓ TIR Lower（Lower 到 TIR）
    ↓ CodeGen（代码生成）
    ↓ GraphRuntime/VM（执行）

Relax 编译管线：
  Relax IR
    ↓ InferStructInfo（结构信息推断）
    ↓ FuseOpsByPattern（模式匹配融合）
    ↓ LegalizeOps（Lower 到 TIR）
    ↓ TIR 优化 Pass
    ↓ CodeGen（代码生成）
    ↓ VM（执行）

关键区别：
- Relax 不需要 TE 作为中间层
- Relax 的融合使用模式匹配而非静态分析
- Relax 的 VM 更加统一和高效
```

### 31.14.4 算子表示对比



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# ========== Relay 中的卷积 ==========
# 算子调用
y = relay.nn.conv2d(x, w,
                    strides=(1, 1),
                    padding=(1, 1),
                    dilation=(1, 1),
                    groups=1,
                    channels=64,
                    kernel_size=(3, 3))

# 属性通过 Conv2DAttrs 传递
# 属性在编译时固定

# ========== Relax 中的卷积 ==========
# 算子调用
y = R.nn.conv2d(x, w,
                strides=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                groups=1,
                data_layout="NCHW",
                kernel_layout="OIHW")

# 属性同样通过 attrs 传递
# 但可以更好地与 TIR 集成
```

### 31.14.5 错误处理对比



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# ========== Relay 的错误处理 ==========
# 错误通常在运行时才发现
# 例如：类型不匹配在 InferType Pass 中报告

# ========== Relax 的错误处理 ==========
# StructInfo 提供了更早的错误检测

# 示例：形状不匹配
@R.function
def main(x: R.Tensor((1, 64), "float32"),
         w: R.Tensor((128, 256), "float32")):
    # 错误：matmul 的维度不匹配 (64 != 128)
    # Relax 会在 InferStructInfo 时就报告错误
    lv0 = R.matmul(x, w)
    return lv0

# Relax 的错误信息更友好：
# "StructInfoError: matmul requires matching contraction dimensions,
#  got 64 and 128"
```

---

## 31.15 Relax 的前端导入

### 31.15.1 从 PyTorch 导入到 Relax

TVM 提供了从 PyTorch 模型导入到 Relax 的完整路径：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import torch
import torchvision
from tvm.relax.frontend import from_pytorch

# 准备 PyTorch 模型
model = torchvision.models.resnet18(pretrained=False)
model.eval()

# 定义输入信息
input_info = [
    (("input",), (1, 3, 224, 224), "float32"),
]

# 转换为 Relax IR
mod = from_pytorch(model, input_info)

# 查看生成的 Relax IR
print(mod)
```

PyTorch 到 Relax 的映射关系：

| PyTorch Op | Relax Op | 说明 |
|-----------|----------|------|
| `torch.matmul` | `R.matmul` | 矩阵乘法 |
| `torch.nn.functional.conv2d` | `R.nn.conv2d` | 2D 卷积 |
| `torch.relu` | `R.nn.relu` | ReLU 激活 |
| `torch.add` | `R.add` | 逐元素加法 |
| `torch.reshape` | `R.reshape` | 形状变换 |
| `torch.cat` | `R.concat` | 张量拼接 |
| `torch.softmax` | `R.nn.softmax` | Softmax |
| `torch.nn.functional.linear` | `R.nn.linear` | 线性变换 |
| `torch.permute` | `R.permute_dims` | 维度重排 |
| `torch.nn.functional.gelu` | `R.nn.gelu` | GELU 激活 |

### 31.15.2 从 ONNX 导入到 Relax



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm.relax.frontend import from_onnx
import onnx

# 加载 ONNX 模型
onnx_model = onnx.load("model.onnx")

# 转换为 Relax IR
mod = from_onnx(onnx_model, shape_dict={"input": (1, 3, 224, 224)})
```

### 31.15.3 从 FX Graph 导入



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm.relax.frontend import from_fx
import torch

# 使用 FX tracer
model = MyModel()
fx_graph = torch.fx.symbolic_trace(model)

# 转换为 Relax
mod = from_fx(fx_graph, input_info)
```

### 31.15.4 自定义算子映射

当 PyTorch 中使用了自定义算子时，需要注册映射规则：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm.relax.frontend import register_pytorch_op_converter

@register_pytorch_op_converter("my_custom_op")
def convert_my_custom_op(node, graph_converter):
    """自定义 PyTorch 算子到 Relax 的映射"""
    # 获取输入
    args = [graph_converter.get_value(arg) for arg in node.args]

    # 调用对应的 Relax 算子
    return graph_converter.builder.emit(
        relax.op.my_custom_op(*args)
    )
```

---

## 31.16 Relax 的调试工具

### 31.16.1 IR 可视化

Relax 支持将 IR 导出为 DOT 格式进行可视化：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm.relax.analysis import visualize

# 生成 DOT 文件
visualize(mod, "model.dot")

# 使用 graphviz 渲染
# dot -Tpng model.dot -o model.png
```

### 31.16.2 StructInfo 调试



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm.relax.analysis import analyze_struct_info

# 分析整个模块的 StructInfo
info = analyze_struct_info(mod)
for func_name, func_info in info.items():
    print(f"\n函数: {func_name}")
    for i, param_info in enumerate(func_info["params"]):
        print(f"  参数 {i}: {param_info}")
    print(f"  返回值: {func_info['ret']}")
```

### 31.16.3 Pass 调试



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import transform

@transform.module_pass(opt_level=0)
class DebugPass:
    """在编译管线中插入调试信息"""
    def __init__(self, stage_name):
        self.stage_name = stage_name

    def transform_module(self, mod, ctx):
        print(f"\n{'='*60}")
        print(f"阶段: {self.stage_name}")
        print(f"{'='*60}")
        for gv, func in mod.functions.items():
            print(f"\n函数: {gv}")
            print(func)
        return mod

# 使用
pipeline = tvm.transform.Sequential([
    DebugPass("原始 IR"),
    relax.transform.InferStructInfo(),
    DebugPass("StructInfo 推断后"),
    relax.transform.FuseOpsByPattern(),
    DebugPass("算子融合后"),
    relax.transform.LegalizeOps(),
    DebugPass("LegalizeOps 后"),
])
```

### 31.16.4 单元测试辅助工具



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import tvm
from tvm import relax, testing

def assert_relax_equal(actual_mod, expected_mod):
    """断言两个 Relax 模块结构相等"""
    for gv in expected_mod.functions:
        assert gv in actual_mod.functions, f"缺少函数 {gv}"
        actual_func = actual_mod[gv]
        expected_func = expected_mod[gv]
        # 比较函数结构
        testing.assert_structural_equal(actual_func, expected_func)

def create_test_module():
    """创建测试用的 Relax 模块"""
    bb = relax.BlockBuilder()
    with bb.function("test_func"):
        x = relax.Var("x", relax.TensorStructInfo((1, 64), "float32"))
        with bb.dataflow():
            lv0 = bb.emit(relax.op.nn.relu(x))
            bb.emit_output(lv0)
        bb.emit_func_output(lv0)
    return bb.get()
```

---

## 31.17 Relax 的社区与未来发展

### 31.17.1 Relax 的发展路线

TVM Unity / Relax 的发展路线图：

| 阶段 | 状态 | 内容 |
|------|------|------|
| **Phase 1** | ✅ 完成 | Relax IR 基础架构 |
| **Phase 2** | ✅ 完成 | Relax VM 与编译管线 |
| **Phase 3** | ✅ 完成 | PyTorch/ONNX 前端导入 |
| **Phase 4** | 🔄 进行中 | 全面替换 Relay 为默认前端 |
| **Phase 5** | 🔄 进行中 | MetaSchedule 与 Relax 深度集成 |
| **Phase 6** | 📋 计划中 | Relax 的分布式推理支持 |

### 31.17.2 MLC-LLM 与 Relax

MLC-LLM 是基于 TVM Unity/Relax 的 LLM 推理引擎：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# MLC-LLM 使用 Relax 编译 LLM 模型
from mlc_chat import ChatModule

# 加载预编译的 LLM
cm = ChatModule(
    model="Llama-2-7b-chat-hf-q4f16_1",
    device="cuda"
)

# 运行推理
response = cm.generate("What is TVM?")
print(response)
```

MLC-LLM 的 Relax 编译流程：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
HuggingFace 模型
    ↓ Relax 前端导入
Relax IR（包含注意力、FFN 等模块）
    ↓ 算子融合（FlashAttention、FFN 融合）
融合后的 Relax IR
    ↓ LegalizeOps + TIR 优化
TIR（带调度信息）
    ↓ CodeGen（CUDA/ROCm/Metal/Vulkan）
目标代码
    ↓ Relax VM 执行
推理结果
```

### 31.17.3 Relax 的扩展性

Relax 设计了良好的扩展机制：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 1. 注册新的 Relax 算子
@relax.op.register("my_new_op")
def my_new_op(x, y):
    return relax.call_pure_packed("my_new_op_impl", x, y,
                                   sinfo_args=x.struct_info)

# 2. 注册新的变换 Pass
@tvm.transform.module_pass(opt_level=1)
def MyCustomPass(mod, ctx):
    # 自定义的变换逻辑
    return mod

# 3. 注册新的后端 Codegen
TVM_REGISTER_GLOBAL("relax.ext.my_backend")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    // 自定义的代码生成逻辑
});
```

---

## 31.99 文字内容强化：TVM Unity 的工程化理解

这一节用于把前文的 API、IR、Pass、Runtime 和部署片段串联为更完整的工程叙事。
很多学习者第一次阅读 TVM 文档时会觉得示例代码很多，但真正上线时仍然不知道如何判断方案是否可靠。
原因在于 TVM 不是单个推理库，而是一条从模型语义到硬件代码的编译链路。
链路越长，越需要把每一步的业务目标、内部机制、适用边界和失败模式说清楚。

### 31.99.1 代码解读的阅读方法

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

- 围绕“Relax、TIR 与运行时统一后的语义收益”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“动态形状、张量表达和函数调用的端到端表达”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“从 Relay 迁移到 Relax 的工程路径”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 MLIR、XLA StableHLO 和 PyTorch 2.x 编译栈的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 31.99.2 业务意义

1. TVM Unity 的业务价值不只是让模型跑得更快，而是让同一个模型可以在不同成本、功耗和延迟约束下交付。
2. 在服务器场景中，核心指标通常是吞吐、P95/P99 延迟、资源利用率和多租户隔离。
3. 在移动端场景中，核心指标通常是首帧时间、持续发热、内存峰值和包体大小。
4. 在嵌入式场景中，核心指标通常是 Flash 占用、静态内存、实时性和掉电恢复能力。
5. 在云端批处理场景中，编译时间可以接受更长，但调优记录和缓存复用变得非常重要。
6. 在在线服务场景中，编译产物需要可回滚、可审计、可灰度，而不能只在开发机上验证。
7. 业务方关心的是 SLA、成本和稳定性，编译器工程师关心的是 IR 正确性、优化空间和后端能力。
8. 优秀的 TVM 项目需要把这两类语言翻译成共同的指标体系。
9. 当优化收益只有少量百分点时，应评估它是否值得引入新的维护复杂度。
10. 当优化收益很大但只在少数输入上成立时，应评估输入分布变化后的风险。

- 围绕“Relax、TIR 与运行时统一后的语义收益”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“动态形状、张量表达和函数调用的端到端表达”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“从 Relay 迁移到 Relax 的工程路径”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 MLIR、XLA StableHLO 和 PyTorch 2.x 编译栈的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 31.99.3 TVM 内部机制

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

- 围绕“Relax、TIR 与运行时统一后的语义收益”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“动态形状、张量表达和函数调用的端到端表达”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“从 Relay 迁移到 Relax 的工程路径”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 MLIR、XLA StableHLO 和 PyTorch 2.x 编译栈的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 31.99.4 适用场景

1. 当模型结构相对稳定、目标硬件明确、性能收益可以通过基准测试确认时，TVM Unity 相关技术最容易发挥价值。
2. 当团队需要支持多种硬件后端时，TVM 的统一 IR 和 Target 抽象可以降低重复适配成本。
3. 当模型中存在框架运行时开销、算子融合机会或布局转换冗余时，编译优化通常能带来明显收益。
4. 当部署环境不能依赖完整 Python 栈时，AOT、CRT 或导出后的 runtime artifact 更有意义。
5. 当硬件厂商提供高性能库但模型图需要复杂切分时，BYOC 和外部 codegen 是常见选择。
6. 当输入形状变化频繁时，应提前设计 shape 策略，而不是在上线前才补动态形状支持。
7. 当模型版本迭代频繁时，应把编译、调优、验证和发布纳入 CI/CD。
8. 当业务对精度非常敏感时，应把优化收益和数值回归一起评估。
9. 当系统存在多模型串联时，应评估端到端 pipeline，而不是只优化单个模型。
10. 当部署设备数量很大时，编译产物的一致性和可追踪性比单次实验性能更重要。

- 围绕“Relax、TIR 与运行时统一后的语义收益”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“动态形状、张量表达和函数调用的端到端表达”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“从 Relay 迁移到 Relax 的工程路径”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 MLIR、XLA StableHLO 和 PyTorch 2.x 编译栈的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 31.99.5 限制条件

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

- 围绕“Relax、TIR 与运行时统一后的语义收益”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“动态形状、张量表达和函数调用的端到端表达”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“从 Relay 迁移到 Relax 的工程路径”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 MLIR、XLA StableHLO 和 PyTorch 2.x 编译栈的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 31.99.6 工程经验

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

- 围绕“Relax、TIR 与运行时统一后的语义收益”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“动态形状、张量表达和函数调用的端到端表达”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“从 Relay 迁移到 Relax 的工程路径”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 MLIR、XLA StableHLO 和 PyTorch 2.x 编译栈的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 31.99.7 常见误区

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

- 围绕“Relax、TIR 与运行时统一后的语义收益”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“动态形状、张量表达和函数调用的端到端表达”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“从 Relay 迁移到 Relax 的工程路径”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 MLIR、XLA StableHLO 和 PyTorch 2.x 编译栈的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 31.99.8 生产部署注意事项

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

- 围绕“Relax、TIR 与运行时统一后的语义收益”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“动态形状、张量表达和函数调用的端到端表达”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“从 Relay 迁移到 Relax 的工程路径”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 MLIR、XLA StableHLO 和 PyTorch 2.x 编译栈的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 31.99.9 与同类系统对比

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

- 围绕“Relax、TIR 与运行时统一后的语义收益”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“动态形状、张量表达和函数调用的端到端表达”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“从 Relay 迁移到 Relax 的工程路径”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 MLIR、XLA StableHLO 和 PyTorch 2.x 编译栈的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 31.99.10 章节复盘

1. 回到本章，TVM Unity 的关键不是记住所有 API，而是理解为什么这些 API 会出现在编译链路的这个位置。
2. 当你看到一段代码时，应能说出它改变了模型语义、调度空间、内存布局、运行时入口还是部署产物。
3. 当你看到一个性能数字时，应能说出它的测试输入、硬件状态、计时方法和误差范围。
4. 当你看到一个优化 pass 时，应能说出它依赖的前置假设和可能破坏的边界条件。
5. 当你准备上线时，应能说出失败后如何回滚、如何复现、如何定位和如何与业务方沟通影响。
6. 这套思维比单个示例更重要，因为 TVM 的 API 会演进，但编译部署的工程约束长期稳定。
7. 后续学习中，可以把每一章都转化为一张决策表：何时使用、收益来自哪里、风险是什么、如何验证。
8. 只有把代码、机制和工程策略放在一起，TVM 才不只是工具箱，而是可运行的生产系统。
9. 因此，本章新增的文字说明应作为阅读代码段的上下文，而不是替代对原始代码的逐行理解。
10. 如果遇到与示例不一致的实际项目，应优先回到模型约束和目标硬件，而不是机械套用章节流程。

- 围绕“Relax、TIR 与运行时统一后的语义收益”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“动态形状、张量表达和函数调用的端到端表达”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“从 Relay 迁移到 Relax 的工程路径”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 MLIR、XLA StableHLO 和 PyTorch 2.x 编译栈的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。


## 31.18 本章小结

本章深入介绍了 TVM Unity 的设计愿景与 Relax IR 的核心技术：

1. **Unity 愿景**：统一 Relay/TE/TIR 为一个连贯的编译栈，消除语义鸿沟
2. **Relax IR**：新的表达式体系（Expr、Call、Function）和结构化类型系统（StructInfo）
3. **BlockBuilder**：命令式的程序构建 API
4. **变换 Pass**：LegalizeOps、FuseOpsByPattern 等核心变换
5. **Relax VM**：基于字节码的虚拟机执行引擎
6. **端到端流程**：从 PyTorch 导入到 VM 执行的完整管线
7. **内存规划**：Relax 的内存分配与 inplace 优化
8. **前端导入**：PyTorch/ONNX/FX 到 Relax 的完整路径
9. **调试工具**：IR 可视化、StructInfo 分析、Pass 调试

Relax 代表了 TVM 的未来方向，它的设计解决了 Relay 时代的许多根本性问题，特别是在动态形状和端到端优化方面。

<div data-component="TVMUnitySummaryTable"></div>

> **下一章预告**：第 32 章将深入讨论动态形状与符号推理，这是 Relax 设计中的一个重要改进点。

---

## 附录 A：Relax IR 完整 API 参考

### A.1 核心表达式类型

| 类型 | 文件位置 | 说明 |
|------|---------|------|
| `relax.Expr` | `include/tvm/relax/expr.h` | 所有表达式的基类 |
| `relax.Var` | `include/tvm/relax/expr.h` | 局部变量 |
| `relax.GlobalVar` | `include/tvm/relax/expr.h` | 全局函数引用 |
| `relax.Function` | `include/tvm/relax/expr.h` | 函数定义 |
| `relax.Call` | `include/tvm/relax/expr.h` | 算子/函数调用 |
| `relax.SeqExpr` | `include/tvm/relax/expr.h` | 序列表达式 |
| `relax.If` | `include/tvm/relax/expr.h` | 条件分支 |
| `relax.Tuple` | `include/tvm/relax/expr.h` | 元组构造 |
| `relax.TupleGetItem` | `include/tvm/relax/expr.h` | 元组索引 |
| `relax.ShapeExpr` | `include/tvm/relax/expr.h` | 形状表达式 |
| `relax.ExternFunc` | `include/tvm/relax/expr.h` | 外部函数引用 |
| `relax.Constant` | `include/tvm/relax/expr.h` | 常量张量 |

### A.2 StructInfo 类型

| 类型 | 文件位置 | 说明 |
|------|---------|------|
| `relax.StructInfo` | `include/tvm/relax/struct_info.h` | 基类 |
| `relax.TensorStructInfo` | `include/tvm/relax/struct_info.h` | 张量结构信息 |
| `relax.TupleStructInfo` | `include/tvm/relax/struct_info.h` | 元组结构信息 |
| `relax.ShapeStructInfo` | `include/tvm/relax/struct_info.h` | 形状结构信息 |
| `relax.FuncStructInfo` | `include/tvm/relax/struct_info.h` | 函数结构信息 |
| `relax.ObjectStructInfo` | `include/tvm/relax/struct_info.h` | 通用对象 |

### A.3 BlockBuilder API

| 方法 | 文件位置 | 说明 |
|------|---------|------|
| `BlockBuilder.emit()` | `python/tvm/relax/block_builder.py` | 发射表达式 |
| `BlockBuilder.emit_output()` | `python/tvm/relax/block_builder.py` | 标记输出 |
| `BlockBuilder.normalize()` | `python/tvm/relax/block_builder.py` | 规范化 |
| `BlockBuilder.dataflow()` | `python/tvm/relax/block_builder.py` | 开始 DataflowBlock |
| `BlockBuilder.function()` | `python/tvm/relax/block_builder.py` | 开始函数定义 |
| `BlockBuilder.get()` | `python/tvm/relax/block_builder.py` | 获取 IRModule |

### A.4 变换 Pass 列表

| Pass | 源码位置 | 说明 |
|------|---------|------|
| `InferStructInfo` | `src/relax/transform/infer_struct_info.cc` | 推断结构信息 |
| `FuseOpsByPattern` | `src/relax/transform/fuse_ops_by_pattern.cc` | 模式融合 |
| `LegalizeOps` | `src/relax/transform/legalize_ops.cc` | Lower 到 TIR |
| `CallTIRRewrite` | `src/relax/transform/call_tir_rewrite.cc` | call_tir 重写 |
| `Normalize` | `src/relax/transform/normalize.cc` | 规范化 |
| `FoldConstant` | `src/relax/transform/fold_constant.cc` | 常量折叠 |
| `DeadCodeElimination` | `src/relax/transform/dead_code_elimination.cc` | 死代码消除 |
| `LowerAllocTensor` | `src/relax/transform/lower_alloc_tensor.cc` | 内存分配 |
| `BindParams` | `src/relax/transform/bind_params.cc` | 参数绑定 |
| `RemoveUnusedOutputs` | `src/relax/transform/remove_unused_outputs.cc` | 移除未用输出 |
| `RunCodegen` | `src/relax/transform/run_codegen.cc` | 外部 codegen |
| `MemoryPlanning` | `src/relax/transform/memory_planning.cc` | 内存规划 |
| `ToNonDataflow` | `src/relax/transform/to_non_dataflow.cc` | 移除 Dataflow |
| `CallRewrite` | `src/relax/transform/call_rewrite.cc` | 调用重写 |
| `StaticPlanBlockMemory` | `src/relax/transform/static_plan_block_memory.cc` | 静态内存规划 |

### A.5 算子分类索引

| 类别 | 源码位置 | 典型算子 |
|------|---------|---------|
| 二元算子 | `src/relax/op/tensor/binary.cc` | `add`, `multiply`, `divide`, `mod` |
| 一元算子 | `src/relax/op/tensor/unary.cc` | `abs`, `ceil`, `floor`, `neg`, `relu` |
| 归约算子 | `src/relax/op/tensor/reduce.cc` | `sum`, `max`, `min`, `mean`, `variance` |
| 形状算子 | `src/relax/op/tensor/manipulate.cc` | `reshape`, `permute_dims`, `concat`, `split` |
| 线性代数 | `src/relax/op/tensor/linear_algebra.cc` | `matmul`, `einsum` |
| NN 算子 | `src/relax/op/nn/` | `conv2d`, `linear`, `softmax`, `layer_norm` |
| 统计算子 | `src/relax/op/statistical.cc` | `mean`, `variance` |
| 内存算子 | `src/relax/op/memory.cc` | `alloc_tensor`, `kill_object` |

---

## 附录 B：Relax 编译管线配置模板

### B.1 标准编译管线

```python
import tvm
from tvm import relax
from tvm import transform

def create_standard_pipeline(target, opt_level=3):
    """创建标准的 Relax 编译管线"""
    with target:
        pipeline = transform.Sequential([
            # 第一阶段：分析与规范化
            relax.transform.InferStructInfo(),
            relax.transform.Normalize(),
            relax.transform.FoldConstant(),

            # 第二阶段：优化
            relax.transform.FuseOpsByPattern(),
            relax.transform.DeadCodeElimination(),
            relax.transform.RemoveUnusedOutputs(),

            # 第三阶段：Lowering
            relax.transform.LegalizeOps(),
            relax.transform.CallTIRRewrite(),
            relax.transform.LowerAllocTensor(),

            # 第四阶段：内存规划
            relax.transform.StaticPlanBlockMemory(),

            # 第五阶段：代码生成
            relax.transform.RunCodegen(target),
        ], opt_level=opt_level)

    return pipeline
```

### B.2 GPU 专用管线



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def create_gpu_pipeline(target):
    """创建 GPU 专用的编译管线"""
    with target:
        pipeline = transform.Sequential([
            # 基础优化
            relax.transform.InferStructInfo(),
            relax.transform.Normalize(),
            relax.transform.FoldConstant(),

            # GPU 特定的融合模式
            relax.transform.FuseOpsByPattern([
                # Conv2d + BN + ReLU 融合
                ("pattern_conv2d_bn_relu", "relax.nn.conv2d + relax.nn.batch_norm + relax.nn.relu"),
                # MatMul + GELU 融合
                ("pattern_matmul_gelu", "relax.matmul + relax.nn.gelu"),
            ]),

            # Lowering
            relax.transform.LegalizeOps(),
            relax.transform.CallTIRRewrite(),

            # GPU 内存优化
            relax.transform.LowerAllocTensor(),
            relax.transform.StaticPlanBlockMemory(),

            # 代码生成
            relax.transform.RunCodegen(target),
        ])

    return pipeline
```

### B.3 嵌入式设备管线



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def create_embedded_pipeline(target, memory_budget_mb=256):
    """创建嵌入式设备的编译管线（内存受限）"""
    with target:
        pipeline = transform.Sequential([
            # 激进的常量折叠
            relax.transform.InferStructInfo(),
            relax.transform.Normalize(),
            relax.transform.FoldConstant(),

            # 最小化内存使用
            relax.transform.DeadCodeElimination(),
            relax.transform.RemoveUnusedOutputs(),

            # 保守的融合（避免过多中间张量）
            relax.transform.FuseOpsByPattern(max_fused_ops=2),

            # Lowering
            relax.transform.LegalizeOps(),
            relax.transform.CallTIRRewrite(),

            # 内存规划（带预算约束）
            relax.transform.StaticPlanBlockMemory(
                memory_budget=memory_budget_mb * 1024 * 1024
            ),

            # 代码生成
            relax.transform.RunCodegen(target),
        ])

    return pipeline
```

---

## 附录 C：Relax 常见模式示例

### C.1 残差连接模式



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
@tvm.script.ir_module
class ResNetBlock:
    @R.function
    def residual_block(
        x: R.Tensor((1, 64, 56, 56), "float32"),
        w1: R.Tensor((64, 64, 3, 3), "float32"),
        b1: R.Tensor((64,), "float32"),
        w2: R.Tensor((64, 64, 3, 3), "float32"),
        b2: R.Tensor((64,), "float32"),
    ):
        with R.dataflow():
            # 第一个卷积
            lv0 = R.nn.conv2d(x, w1, padding=(1, 1))
            lv1 = R.add(lv0, b1)
            lv2 = R.nn.relu(lv1)

            # 第二个卷积
            lv3 = R.nn.conv2d(lv2, w2, padding=(1, 1))
            lv4 = R.add(lv3, b2)

            # 残差连接
            lv5 = R.add(lv4, x)
            lv6 = R.nn.relu(lv5)

            R.output(lv6)
        return lv6
```

### C.2 多头注意力模式



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
batch = tir.Var("batch", "int64")
seq_len = tir.Var("seq_len", "int64")

@tvm.script.ir_module
class MultiHeadAttention:
    @R.function
    def multi_head_attention(
        Q: R.Tensor((batch, seq_len, 768), "float32"),
        K: R.Tensor((batch, seq_len, 768), "float32"),
        V: R.Tensor((batch, seq_len, 768), "float32"),
        W_q: R.Tensor((768, 768), "float32"),
        W_k: R.Tensor((768, 768), "float32"),
        W_v: R.Tensor((768, 768), "float32"),
        W_o: R.Tensor((768, 768), "float32"),
    ):
        with R.dataflow():
            # 线性投影
            Q_proj = R.matmul(Q, W_q)  # (batch, seq_len, 768)
            K_proj = R.matmul(K, W_k)
            V_proj = R.matmul(V, W_v)

            # reshape 到多头
            Q_heads = R.reshape(Q_proj, (batch, seq_len, 12, 64))
            K_heads = R.reshape(K_proj, (batch, seq_len, 12, 64))
            V_heads = R.reshape(V_proj, (batch, seq_len, 12, 64))

            # 转置
            Q_heads = R.permute_dims(Q_heads, (0, 2, 1, 3))
            K_heads = R.permute_dims(K_heads, (0, 2, 1, 3))
            V_heads = R.permute_dims(V_heads, (0, 2, 1, 3))

            # 注意力计算
            K_T = R.permute_dims(K_heads, (0, 1, 3, 2))
            scores = R.matmul(Q_heads, K_T)
            scores = R.divide(scores, R.const(8.0))  # sqrt(64)
            attn = R.nn.softmax(scores, axis=-1)

            # 加权求和
            context = R.matmul(attn, V_heads)

            # 转置回并 reshape
            context = R.permute_dims(context, (0, 2, 1, 3))
            context = R.reshape(context, (batch, seq_len, 768))

            # 输出投影
            output = R.matmul(context, W_o)

            R.output(output)
        return output
```

### C.3 Layer Normalization 模式



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
@tvm.script.ir_module
class LayerNorm:
    @R.function
    def layer_norm(
        x: R.Tensor((batch, seq_len, 768), "float32"),
        gamma: R.Tensor((768,), "float32"),
        beta: R.Tensor((768,), "float32"),
    ):
        with R.dataflow():
            # 计算均值
            mean = R.mean(x, axis=[-1], keepdims=True)

            # 计算方差
            diff = R.subtract(x, mean)
            var = R.mean(R.multiply(diff, diff), axis=[-1], keepdims=True)

            # 归一化
            std = R.sqrt(R.add(var, R.const(1e-5)))
            normed = R.divide(diff, std)

            # 仿射变换
            output = R.add(R.multiply(normed, gamma), beta)

            R.output(output)
        return output
```

---

## 附录 D：Relax 与 Relay 的迁移指南

### D.1 从 Relay 迁移到 Relax

| Relay 代码 | Relax 代码 | 说明 |
|-----------|-----------|------|
| `relay.var("x", TensorType(...))` | `relax.Var("x", TensorStructInfo(...))` | 变量定义 |
| `relay.Function([x, w], body)` | `R.function` + `with R.dataflow()` | 函数定义 |
| `relay.Call(op, args)` | `relax.op.xxx(*args)` 或 `R.xxx(*args)` | 算子调用 |
| `relay.Tuple([a, b])` | `R.tuple(a, b)` | 元组构造 |
| `relay.TupleGetItem(t, 0)` | `R.tuple_get_item(t, 0)` | 元组索引 |
| `relay.Let(x, val, body)` | `with R.dataflow(): lv = R.xxx(...)` | 变量绑定 |
| `relay.If(cond, true, false)` | `R.If(cond, true_branch, false_branch)` | 条件 |
| `relay.transform.InferType()` | `relax.transform.InferStructInfo()` | 类型推断 |
| `relay.transform.FuseOps()` | `relax.transform.FuseOpsByPattern()` | 算子融合 |

### D.2 迁移注意事项

1. **DataflowBlock**：Relax 要求将纯计算放入 `with R.dataflow()` 中
2. **StructInfo**：替代 Relay 的 Type 系统，提供更丰富的形状信息
3. **动态形状**：使用 `tir.Var` 替代 `relay.Any`
4. **TE 集成**：Relax 不再需要单独的 TE lowering，通过 `call_tir` 直接调用 TIR
5. **常量处理**：使用 `R.const()` 替代 `relay.const()`

### D.3 迁移脚本模板



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def migrate_relay_to_relax(relay_mod):
    """将 Relay 模块迁移到 Relax（简化的迁移工具）"""
    from tvm import relay, relax

    bb = relax.BlockBuilder()

    for gv, func in relay_mod.functions.items():
        if not isinstance(func, relay.Function):
            continue

        # 转换参数
        params = []
        for param in func.params:
            sinfo = convert_type_to_struct_info(param.checked_type)
            params.append(relax.Var(param.name_hint, sinfo))

        # 转换函数体（简化版）
        body = convert_relay_expr_to_relax(func.body, bb)

        # 构建 Relax 函数
        with bb.function(gv.name_hint):
            for p in params:
                bb.add_param(p)
            with bb.dataflow():
                output = bb.emit(body)
                bb.emit_output(output)
            bb.emit_func_output(output)

    return bb.get()

def convert_type_to_struct_info(relay_type):
    """将 Relay 类型转换为 StructInfo"""
    if isinstance(relay_type, tvm.relay.TensorType):
        return relax.TensorStructInfo(relay_type.shape, relay_type.dtype)
    elif isinstance(relay_type, tvm.relay.TupleType):
        fields = [convert_type_to_struct_info(f) for f in relay_type.fields]
        return relax.TupleStructInfo(fields)
    else:
        raise ValueError(f"不支持的类型: {type(relay_type)}")
```

---

## 附录 E：Relax 相关源码索引

| 功能 | 源码文件 | 关键函数/类 |
|------|---------|------------|
| Relax Expr 定义 | `include/tvm/relax/expr.h` | `Expr`, `Var`, `Call`, `Function` |
| StructInfo 定义 | `include/tvm/relax/struct_info.h` | `TensorStructInfo`, `FuncStructInfo` |
| IRModule | `include/tvm/relax/ir/module.h` | `IRModule` |
| BlockBuilder | `python/tvm/relax/block_builder.py` | `BlockBuilder` |
| InferStructInfo | `src/relax/transform/infer_struct_info.cc` | `InferStructInfo` |
| FuseOpsByPattern | `src/relax/transform/fuse_ops_by_pattern.cc` | `FuseOpsByPattern` |
| LegalizeOps | `src/relax/transform/legalize_ops.cc` | `LegalizeOps` |
| Normalize | `src/relax/transform/normalize.cc` | `Normalize` |
| FoldConstant | `src/relax/transform/fold_constant.cc` | `FoldConstant` |
| DCE | `src/relax/transform/dead_code_elimination.cc` | `DeadCodeElimination` |
| MemoryPlanning | `src/relax/transform/memory_planning.cc` | `MemoryPlanning` |
| VM Compiler | `src/relax/backend/vm/vm_compiler.cc` | `VMCompiler` |
| VM Executable | `src/relax/backend/vm/executable.h` | `VMExecutable` |
| VM Runtime | `src/relax/backend/vm/vm.cc` | `VirtualMachine` |
| Bytecode | `include/tvm/relax/backend/vm/bytecode.h` | `Opcode`, `Instruction` |
| PyTorch 前端 | `python/tvm/relax/frontend/pytorch_translator.py` | `from_pytorch` |
| ONNX 前端 | `python/tvm/relax/frontend/onnx_translator.py` | `from_onnx` |
| Relax 算子定义 | `src/relax/op/tensor/` | `matmul`, `add` 等 |
| NN 算子定义 | `src/relax/op/nn/` | `conv2d`, `softmax` 等 |

---

## 31.19 Relax Expr 类型层次深度解析

### 31.19.1 Expr 节点的 C++ 内存布局

Relax 的每个表达式节点都继承自 `RelaxExpr`，定义在 `include/tvm/relax/expr.h`。节点使用引用计数（`ObjectRef`）管理生命周期：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// include/tvm/relax/expr.h — 所有 Relax 表达式的基类
class RelaxExpr : public ObjectRef {
 public:
  // 获取此表达式的 StructInfo（编译时推断的类型/形状信息）
  Optional<StructInfo> struct_info() const;
  // 设置 StructInfo（通常由 InferStructInfo Pass 完成）
  void set_struct_info(StructInfo sinfo);
 protected:
  // 子类必须实现：将自身发送给 ExprVisitor/ExprFunctor
  void VisitExpr(ExprVisitor* visitor) const;
};
```

每个具体节点类型定义了不同的字段集合。以下是各节点的详细字段说明：

| 节点类型 | 源码位置 | 核心字段 | 语义 |
|---------|---------|---------|------|
| `Var` | `expr.h:85` | `name_hint: String`, `struct_info_: Optional<StructInfo>` | 局部变量，SSA 风格绑定 |
| `GlobalVar` | `expr.h:120` | `name_hint: String` | 引用 IRModule 中定义的全局函数 |
| `Constant` | `expr.h:140` | `data: NDArray` | 编译时常量张量，存储实际数据 |
| `ShapeExpr` | `expr.h:165` | `values: Array<PrimExpr>` | 形状值，可包含符号变量 |
| `ExternFunc` | `expr.h:190` | `global_symbol: String` | 引用外部 PackedFunc |
| `Call` | `expr.h:210` | `op: Expr`, `args: Array<Expr>`, `attrs: Attrs` | 算子/函数调用 |
| `Tuple` | `expr.h:260` | `fields: Array<Expr>` | 元组构造 |
| `TupleGetItem` | `expr.h:290` | `tuple: Expr`, `index: int` | 元组索引访问 |
| `Function` | `expr.h:320` | `params`, `body`, `ret_struct_info`, `is_pure`, `attrs` | 函数定义 |
| `If` | `expr.h:400` | `condition: Expr`, `true_branch: SeqExpr`, `false_branch: SeqExpr` | 条件分支 |
| `SeqExpr` | `expr.h:430` | `blocks: Array<BindingBlock>`, `body: Expr` | 顺序执行的表达式序列 |

### 31.19.2 If 表达式的执行语义

`If` 表达式实现了 Relax 中的控制流。与 Relay 的 `If` 不同，Relax 的 `If` 的两个分支都是 `SeqExpr`（而不是直接的 `Expr`），这允许分支内部包含多个绑定：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 构造一个 If 表达式的完整示例
import tvm
from tvm import relax, tir

@tvm.script.ir_module
class ConditionalModel:
    @R.function
    def main(x: R.Tensor((128, 256), "float32"),
             w1: R.Tensor((256, 512), "float32"),
             w2: R.Tensor((256, 512), "float32"),
             flag: R.Tensor((), "bool")):
        # If 表达式：根据 flag 选择不同的计算路径
        if R.expr_from_tensor(flag):  # 将 bool tensor 转为条件表达式
            # true 分支：使用 w1
            with R.dataflow():
                lv0 = R.matmul(x, w1)    # 矩阵乘法
                lv1 = R.nn.relu(lv0)     # ReLU 激活
                R.output(lv1)
            result = lv1
        else:
            # false 分支：使用 w2
            with R.dataflow():
                lv0 = R.matmul(x, w2)    # 矩阵乘法（不同的权重）
                lv1 = R.nn.gelu(lv0)     # GELU 激活
                R.output(lv1)
            result = lv1
        return result
```

If 表达式的 C++ 结构展示了其内部组织方式：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// include/tvm/relax/expr.h — If 节点定义
class If : public RelaxExpr {
 public:
  /*! \brief 条件表达式，必须推断为标量 bool 类型 */
  Expr condition;
  /*! \brief 条件为真时执行的序列表达式 */
  SeqExpr true_branch;
  /*! \brief 条件为假时执行的序列表达式 */
  SeqExpr false_branch;
};
// 关键约束：两个分支的输出必须具有相同的 StructInfo
// 例如：如果 true_branch 返回 Tensor((128, 512), "float32")
//       那么 false_branch 也必须返回相同形状和类型的张量
```

### 31.19.3 SeqExpr 与 BindingBlock 的内部结构

`SeqExpr` 是 Relax 中表示顺序执行的核心抽象。它由一系列 `BindingBlock` 组成，最后一个表达式是整个序列的返回值：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# SeqExpr 的等价 Python 表示（伪代码）
# SeqExpr {
#   blocks: [
#     DataflowBlock {  # 纯计算块
#       bindings: [
#         Binding(lv0, Call(matmul, [x, w])),
#         Binding(lv1, Call(relu, [lv0])),
#       ]
#     },
#     BindingBlock {   # 非数据流块（可以有副作用）
#       bindings: [
#         Binding(lv2, Call(print, [lv1])),  # 有副作用的操作
#       ]
#     }
#   ],
#   body: lv2  # 最终返回值
# }
```

`BindingBlock` 有两种变体：

| 类型 | 源码位置 | 约束 | 用途 |
|------|---------|------|------|
| `BindingBlock` | `expr.h:460` | 无特殊约束 | 通用绑定块，可包含副作用 |
| `DataflowBlock` | `expr.h:490` | 所有绑定必须是纯计算 | 编译器可自由优化的纯计算区域 |



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// include/tvm/relax/expr.h — BindingBlock 与 DataflowBlock
class BindingBlock : public RelaxExpr {
 public:
  Array<Binding> bindings;  // 块内的所有变量绑定
};

class DataflowBlock : public BindingBlock {
 public:
  // DataflowBlock 继承自 BindingBlock
  // 额外约束：所有 binding 的 RHS 必须是纯表达式
  // 编译器可以安全地进行以下优化：
  //   1. 重排计算顺序
  //   2. 消除未使用的变量
  //   3. 激进的内存复用
};
```

### 31.19.4 Tuple 与 TupleGetItem 的使用模式

元组是 Relax 中返回多个值的唯一方式。`Tuple` 构造一个元组，`TupleGetItem` 访问元组中的元素：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 元组在多输出算子中的典型使用
@tvm.script.ir_module
class BatchNormModel:
    @R.function
    def batch_norm(
        x: R.Tensor((1, 64, 56, 56), "float32"),
        gamma: R.Tensor((64,), "float32"),
        beta: R.Tensor((64,), "float32"),
        running_mean: R.Tensor((64,), "float32"),
        running_var: R.Tensor((64,), "float32"),
    ):
        with R.dataflow():
            # batch_norm 返回一个元组：(normalized, mean, var)
            # 第一个元素是归一化后的张量
            # 第二个元素是计算的均值
            # 第三个元素是计算的方差
            bn_result = R.nn.batch_norm(
                x, gamma, beta, running_mean, running_var
            )
            # 使用 TupleGetItem 访问元组的第一个元素
            normalized = bn_result[0]  # 等价于 R.tuple_get_item(bn_result, 0)
            R.output(normalized)
        return normalized
```

C++ 层面的 Tuple 和 TupleGetItem 结构：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// include/tvm/relax/expr.h
class Tuple : public RelaxExpr {
 public:
  /*! \brief 元组中的各字段表达式 */
  Array<Expr> fields;
  // StructInfo 自动推断为 TupleStructInfo([field.sinfo for field in fields])
};

class TupleGetItem : public RelaxExpr {
 public:
  /*! \brief 被索引的元组表达式 */
  Expr tuple;
  /*! \brief 索引位置（0-based） */
  int index;
  // StructInfo 自动推断为 tuple.sinfo[index]
};
```

### 31.19.5 Relax Expr 与 Relay Expr 的对比

| 特性 | Relay Expr | Relax Expr |
|------|-----------|-----------|
| **变量** | `Var`（带类型注解） | `Var`（带 StructInfo） |
| **全局引用** | `GlobalVar`（同名） | `GlobalVar`（同名） |
| **函数调用** | `Call(op, args, attrs)` | `Call(op, args, attrs)` |
| **控制流** | `If`（分支为 Expr） | `If`（分支为 SeqExpr） |
| **绑定** | `Let(var, value, body)` | `SeqExpr` + `BindingBlock` |
| **元组** | `Tuple` + `TupleGetItem` | `Tuple` + `TupleGetItem` |
| **常量** | `Constant(NDArray)` | `Constant(NDArray)` |
| **形状** | 无专用节点 | `ShapeExpr(Array<PrimExpr>)` |
| **外部函数** | `Function`（带外部属性） | `ExternFunc(global_symbol)` |
| **数据流标记** | 无 | `DataflowBlock`（纯计算区域） |
| **类型信息** | `checked_type: Type` | `struct_info: StructInfo` |

---

## 31.20 StructInfo 推断规则详解

### 31.20.1 FInferStructInfo 注册机制

每个 Relax 算子必须注册一个 `FInferStructInfo` 函数，用于在编译时推断输出的结构信息。注册过程通过 `TVM_REGISTER_OP` 宏完成：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// src/relax/op/tensor/binary.cc — 二元广播算子的注册
// 第一步：定义推断函数
StructInfo InferStructInfoBinaryBroadcast(const Call& call,
                                          const BlockBuilder& ctx) {
  // 从参数中获取输入的 StructInfo
  TensorStructInfo lhs_sinfo = GetStructInfoAs<TensorStructInfo>(call->args[0]);
  TensorStructInfo rhs_sinfo = GetStructInfoAs<TensorStructInfo>(call->args[1]);

  // 如果任一输入的形状未知（ndim == -1），返回保守估计
  if (lhs_sinfo->ndim == -1 || rhs_sinfo->ndim == -1) {
    return TensorStructInfo(lhs_sinfo->dtype, -1);  // 秩未知
  }

  // 推断输出形状：使用广播规则
  // 逐维度比较：1 可以广播到任意维度，相同维度保持不变
  const ShapeExpr* lhs_shape = lhs_sinfo->shape.as<ShapeExprNode>();
  const ShapeExpr* rhs_shape = rhs_sinfo->shape.as<ShapeExprNode>();

  Array<PrimExpr> out_values;
  for (int i = 0; i < lhs_sinfo->ndim; i++) {
    PrimExpr l = lhs_shape->values[i];
    PrimExpr r = rhs_shape->values[i];
    // 广播规则：如果 l==1 则取 r，如果 r==1 刌取 l，否则必须相等
    out_values.push_back(BroadcastShapeDim(l, r, ctx));
  }

  return TensorStructInfo(ShapeExpr(out_values), lhs_sinfo->dtype);
}

// 第二步：注册到算子
TVM_REGISTER_OP("relax.add")
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                 InferStructInfoBinaryBroadcast);
```

### 31.20.2 各类算子的推断规则

不同类别的算子使用不同的推断策略：

| 算子类别 | 推断策略 | 示例算子 | 源码位置 |
|---------|---------|---------|---------|
| **逐元素算子** | 广播规则推断输出形状 | `add`, `multiply`, `relu` | `src/relax/op/tensor/binary.cc` |
| **矩阵乘法** | 保留 batch 维度，收缩维度消去 | `matmul`, `linear` | `src/relax/op/tensor/linear_algebra.cc` |
| **卷积** | 基于 kernel/stride/padding 计算 | `conv2d`, `conv1d` | `src/relax/op/nn/convolution.cc` |
| **归约** | 消去归约轴（或保持 keepdims） | `sum`, `mean`, `max` | `src/relax/op/tensor/reduce.cc` |
| **形状变换** | 直接使用目标形状 | `reshape`, `permute_dims` | `src/relax/op/tensor/manipulate.cc` |
| **拼接** | 沿指定轴求和 | `concat` | `src/relax/op/tensor/manipulate.cc` |
| **激活函数** | 恒等推断（输出形状=输入形状） | `softmax`, `sigmoid` | `src/relax/op/nn/` |

### 31.20.3 matmul 算子的推断规则详解

矩阵乘法的 StructInfo 推断是最复杂的之一，因为它需要处理批处理维度和收缩维度：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// src/relax/op/tensor/linear_algebra.cc — matmul 推断规则
StructInfo InferStructInfoMatmul(const Call& call, const BlockBuilder& ctx) {
  // 获取两个输入的 StructInfo
  TensorStructInfo lhs_sinfo = GetStructInfoAs<TensorStructInfo>(call->args[0]);
  TensorStructInfo rhs_sinfo = GetStructInfoAs<TensorStructInfo>(call->args[1]);

  int lhs_ndim = lhs_sinfo->ndim;
  int rhs_ndim = rhs_sinfo->ndim;

  // 情况 1：2D × 2D → 标准矩阵乘法
  // (M, K) × (K, N) → (M, N)
  if (lhs_ndim == 2 && rhs_ndim == 2) {
    // 验证收缩维度匹配
    CheckContractionDimMatch(lhs_sinfo, rhs_sinfo, ctx);
    // 构造输出形状
    Array<PrimExpr> out_shape = {
        lhs_shape->values[0],  // M
        rhs_shape->values[1]   // N
    };
    return TensorStructInfo(ShapeExpr(out_shape), lhs_sinfo->dtype);
  }

  // 情况 2：ND × 2D → 批处理矩阵乘法
  // (B1, B2, ..., M, K) × (K, N) → (B1, B2, ..., M, N)
  if (lhs_ndim >= 2 && rhs_ndim == 2) {
    CheckContractionDimMatch(lhs_sinfo, rhs_sinfo, ctx);
    Array<PrimExpr> out_shape;
    // 保留 lhs 的 batch 维度
    for (int i = 0; i < lhs_ndim - 2; i++) {
      out_shape.push_back(lhs_shape->values[i]);
    }
    out_shape.push_back(lhs_shape->values[lhs_ndim - 2]);  // M
    out_shape.push_back(rhs_shape->values[1]);               // N
    return TensorStructInfo(ShapeExpr(out_shape), lhs_sinfo->dtype);
  }

  // 情况 3：ND × ND → 批处理矩阵乘法（两个输入都有 batch 维度）
  if (lhs_ndim >= 2 && rhs_ndim >= 2) {
    // 验证 batch 维度可以广播
    int batch_ndim = std::max(lhs_ndim, rhs_ndim) - 2;
    Array<PrimExpr> out_shape = BroadcastBatchDims(
        lhs_sinfo, rhs_sinfo, batch_ndim, ctx);
    CheckContractionDimMatch(lhs_sinfo, rhs_sinfo, ctx);
    out_shape.push_back(lhs_shape->values[lhs_ndim - 2]);  // M
    out_shape.push_back(rhs_shape->values[rhs_ndim - 1]);   // N
    return TensorStructInfo(ShapeExpr(out_shape), lhs_sinfo->dtype);
  }

  // 秩不足
  ctx->ReportFatal(Diagnostic::Error(call)
      << "matmul requires at least 2 dimensions per input");
  return TensorStructInfo();  // 不可达
}
```

### 31.20.4 自定义 StructInfo 推断规则

用户可以为自定义 Relax 算子编写推断规则：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 使用 Python API 注册自定义算子的 StructInfo 推断规则
from tvm import relax
from tvm.relax import StructInfo
from tvm.relax.expr import Call
from tvm.relax.block_builder import BlockBuilder

def my_custom_op_struct_info(call: Call, ctx: BlockBuilder) -> StructInfo:
    """自定义算子的 StructInfo 推断规则

    该算子接收两个输入：
    - x: 形状为 (batch, seq_len, hidden) 的张量
    - scale: 形状为 (hidden,) 的张量
    输出形状与 x 相同（逐元素缩放）
    """
    # 获取输入的 StructInfo
    x_sinfo = call.args[0].struct_info
    scale_sinfo = call.args[1].struct_info

    # 验证输入类型
    assert isinstance(x_sinfo, relax.TensorStructInfo), \
        f"第一个参数必须是 Tensor，得到 {type(x_sinfo)}"
    assert isinstance(scale_sinfo, relax.TensorStructInfo), \
        f"第二个参数必须是 Tensor，得到 {type(scale_sinfo)}"

    # 验证 scale 是 1D 张量
    assert scale_sinfo.ndim == 1, \
        f"scale 必须是 1D 张量，得到 {scale_sinfo.ndim}D"

    # 输出形状与第一个输入相同
    return relax.TensorStructInfo(
        shape=x_sinfo.shape,      # 保持输入 x 的形状
        dtype=x_sinfo.dtype       # 保持输入 x 的数据类型
    )

# 注册算子和推断规则
@relax.op.register("my_rms_scale", level=15)
def my_rms_scale(x: relax.Expr, scale: relax.Expr) -> relax.Expr:
    """自定义 RMS 缩放算子"""
    return relax.Call(
        relax.op.get("my_rms_scale"),
        [x, scale],
    )

# 注册 StructInfo 推断函数
from tvm._ffi import register_func
register_func("relax.op.my_rms_scale.FInferStructInfo",
              my_custom_op_struct_info)
```

---

## 31.21 BlockBuilder API 深度剖析

### 31.21.1 BlockBuilder 的内部状态机

`BlockBuilder` 是一个有状态的构建器，维护当前构建上下文。其内部状态定义在 `python/tvm/relax/block_builder.py`：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
BlockBuilder 内部状态：
├── _builder_stack: List[Frame]        # 构建帧栈
│   ├── FunctionFrame                  # 函数构建帧
│   │   ├── name: str                  # 函数名
│   │   ├── params: List[Var]          # 参数列表
│   │   └── blocks: List[BindingBlock] # 已完成的块
│   └── DataflowFrame                  # 数据流构建帧
│       ├── bindings: List[Binding]    # 当前块内的绑定
│       └── output_vars: List[Var]     # 输出变量
├── _binding_table: Dict[Var, Expr]    # 变量 → 表达式的映射
├── _functions: Dict[GlobalVar, Function]  # 已完成的函数
├── _mod: IRModule                     # 最终的模块
└── _emit_var_counter: int             # 自动生成变量名的计数器
```

### 31.21.2 normalize() 方法的深层机制

`normalize()` 是 BlockBuilder 最核心的内部方法。它将嵌套的表达式树展平为一系列原子绑定：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# normalize 的工作原理：
# 输入：嵌套的 Call 表达式
#   R.add(R.matmul(x, w), b)
#
# normalize 的处理过程：
#   1. 检查 R.add 的第一个参数 R.matmul(x, w)
#   2. R.matmul(x, w) 不是一个 Var，需要展平
#   3. 为 R.matmul(x, w) 创建新的变量 lv0
#   4. 递归 normalize R.matmul 的参数（x, w 已经是 Var，无需处理）
#   5. 发射绑定 lv0 = R.matmul(x, w)
#   6. 替换原表达式中的子表达式为 lv0
#   7. 检查 R.add 的第二个参数 b（已经是 Var）
#   8. 发射绑定 lv1 = R.add(lv0, b)
#   9. 返回 lv1

# normalize 的递归规则：
# - 如果表达式是 Var → 直接返回（已经是最简形式）
# - 如果表达式是 Constant → 如果引用次数 > 1，创建绑定
# - 如果表达式是 Call → 递归 normalize 每个参数，然后创建绑定
# - 如果表达式是 Tuple → 递归 normalize 每个字段，然后创建绑定
# - 如果表达式是 If → 递归 normalize 两个分支
```

normalize 的 C++ 实现核心逻辑：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// python/tvm/relax/block_builder.py（逻辑等价的 C++ 代码）
Expr BlockBuilder::Normalize(const Expr& expr) {
  // 情况 1：已经是变量，无需处理
  if (expr.as<VarNode>()) {
    return expr;
  }

  // 情况 2：如果是 Call，先 normalize 所有参数
  if (auto* call = expr.as<CallNode>()) {
    Array<Expr> new_args;
    for (const auto& arg : call->args) {
      new_args.push_back(Normalize(arg));  // 递归 normalize
    }
    Call new_call(call->op, new_args, call->attrs);

    // 情况 2a：如果在 DataflowBlock 内部，使用 emit
    if (InDataflowBlock()) {
      return Emit(new_call);  // 创建绑定并返回变量
    }
    // 情况 2b：如果在非数据流块，使用 emit_binding
    return EmitBinding(new_call);
  }

  // 情况 3：Tuple
  if (auto* tuple = expr.as<TupleNode>()) {
    Array<Expr> new_fields;
    for (const auto& field : tuple->fields) {
      new_fields.push_back(Normalize(field));
    }
    return Emit(Tuple(new_fields));
  }

  // 其他情况...
  return Emit(expr);
}
```

### 31.21.3 Dataflow 块的构建流程

`begin_dataflow()` 和 `end_dataflow()` 管理 DataflowBlock 的创建：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# Dataflow 块的构建状态机转换：
#
# begin_dataflow():
#   1. 创建新的 DataflowFrame
#   2. 压入构建帧栈
#   3. 初始化空的 bindings 列表
#
# 在 Dataflow 内部：
#   emit(expr):
#     1. normalize(expr) → 确保所有子表达式已绑定
#     2. 创建新的 Var，生成唯一的名称（lv0, lv1, ...）
#     3. 创建 Binding(Var, expr)
#     4. 添加到当前 DataflowFrame 的 bindings
#     5. 更新 binding_table
#     6. 返回 Var
#
#   emit_output(expr):
#     1. normalize(expr)
#     2. 将结果 Var 标记为 DataflowBlock 的输出
#     3. 输出变量可以被块外的代码引用
#
# end_dataflow():
#   1. 弹出 DataflowFrame
#   2. 从帧中的 bindings 构造 DataflowBlock 节点
#   3. 将 DataflowBlock 添加到父 FunctionFrame 的 blocks 列表
```

### 31.21.4 使用 BlockBuilder 构建 Transformer Block

下面展示使用 BlockBuilder 构建一个完整的 Transformer encoder block，展示其全部 API 的使用方式：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import tvm
from tvm import relax, tir
import numpy as np

def build_transformer_block(hidden_size=768, num_heads=12, ffn_size=3072):
    """使用 BlockBuilder 构建 Transformer encoder block"""
    head_dim = hidden_size // num_heads  # 64

    bb = relax.BlockBuilder()

    # 符号变量
    batch = tir.Var("batch", "int64")
    seq_len = tir.Var("seq_len", "int64")

    with bb.function("transformer_block",
                     params=[],
                     ret_struct_info=relax.TensorStructInfo(
                         (batch, seq_len, hidden_size), "float32")):

        # 声明输入和参数
        x = relax.Var("x", relax.TensorStructInfo(
            (batch, seq_len, hidden_size), "float32"))
        ln_weight = relax.Var("ln_weight",
            relax.TensorStructInfo((hidden_size,), "float32"))
        ln_bias = relax.Var("ln_bias",
            relax.TensorStructInfo((hidden_size,), "float32"))
        w_q = relax.Var("w_q",
            relax.TensorStructInfo((hidden_size, hidden_size), "float32"))
        w_k = relax.Var("w_k",
            relax.TensorStructInfo((hidden_size, hidden_size), "float32"))
        w_v = relax.Var("w_v",
            relax.TensorStructInfo((hidden_size, hidden_size), "float32"))
        w_o = relax.Var("w_o",
            relax.TensorStructInfo((hidden_size, hidden_size), "float32"))
        w_ff1 = relax.Var("w_ff1",
            relax.TensorStructInfo((hidden_size, ffn_size), "float32"))
        w_ff2 = relax.Var("w_ff2",
            relax.TensorStructInfo((ffn_size, hidden_size), "float32"))

        # ====== 第一部分：Layer Norm + Multi-Head Attention ======
        with bb.dataflow():
            # Layer Norm（简化版）
            ln_mean = bb.emit(relax.op.mean(x, axis=[-1], keepdims=True))
            ln_diff = bb.emit(relax.op.subtract(x, ln_mean))
            ln_var = bb.emit(relax.op.mean(
                relax.op.multiply(ln_diff, ln_diff), axis=[-1], keepdims=True))
            ln_std = bb.emit(relax.op.sqrt(
                relax.op.add(ln_var, relax.const(1e-5))))
            ln_normed = bb.emit(relax.op.divide(ln_diff, ln_std))
            ln_out = bb.emit(relax.op.add(
                relax.op.multiply(ln_normed, ln_weight), ln_bias))

            # Q, K, V 投影
            Q = bb.emit(relax.op.matmul(ln_out, w_q))
            K = bb.emit(relax.op.matmul(ln_out, w_k))
            V = bb.emit(relax.op.matmul(ln_out, w_v))

            # Reshape 到多头
            Q_h = bb.emit(relax.op.reshape(Q, (batch, seq_len, num_heads, head_dim)))
            K_h = bb.emit(relax.op.reshape(K, (batch, seq_len, num_heads, head_dim)))
            V_h = bb.emit(relax.op.reshape(V, (batch, seq_len, num_heads, head_dim)))

            # 转置到 (batch, num_heads, seq_len, head_dim)
            Q_t = bb.emit(relax.op.permute_dims(Q_h, [0, 2, 1, 3]))
            K_t = bb.emit(relax.op.permute_dims(K_h, [0, 2, 1, 3]))
            V_t = bb.emit(relax.op.permute_dims(V_h, [0, 2, 1, 3]))

            # 注意力计算
            K_T = bb.emit(relax.op.permute_dims(K_t, [0, 1, 3, 2]))
            scores = bb.emit(relax.op.matmul(Q_t, K_T))
            scores_scaled = bb.emit(relax.op.divide(scores,
                relax.const(float(head_dim ** 0.5))))
            attn = bb.emit(relax.op.nn.softmax(scores_scaled, axis=-1))
            context = bb.emit(relax.op.matmul(attn, V_t))

            # 转置回来
            context_t = bb.emit(relax.op.permute_dims(context, [0, 2, 1, 3]))
            context_flat = bb.emit(relax.op.reshape(
                context_t, (batch, seq_len, hidden_size)))

            # 输出投影
            attn_out = bb.emit(relax.op.matmul(context_flat, w_o))

            # 残差连接
            residual1 = bb.emit(relax.op.add(x, attn_out))

            bb.emit_output(residual1)

        bb.emit_func_output(residual1)

    return bb.get()

# 构建并查看
mod = build_transformer_block()
print(mod)
```

### 31.21.5 BlockBuilder 方法速查表

| 方法 | 返回类型 | 副作用 | 说明 |
|------|---------|--------|------|
| `emit(expr)` | `Var` | 在当前块中添加绑定 | 发射表达式并返回绑定变量 |
| `emit_output(expr)` | `None` | 标记变量为 DataflowBlock 输出 | 必须在 `with dataflow()` 内调用 |
| `emit_expr(expr, name_hint)` | `Var` | 在当前块中添加绑定 | 带自定义名称的 emit |
| `emit_binding(binding)` | `Var` | 添加显式绑定 | 用于 LetBinding 等 |
| `normalize(expr)` | `Expr` | 可能添加中间绑定 | 展平嵌套表达式 |
| `lookup_binding(var)` | `Optional[Expr]` | 无 | 查找变量的定义 |
| `begin_dataflow()` | `None` | 压入 DataflowFrame | 开始纯计算区域 |
| `end_dataflow()` | `DataflowBlock` | 弹出帧，构造块 | 结束纯计算区域 |
| `begin_function(name)` | `None` | 压入 FunctionFrame | 开始函数定义 |
| `end_function(ret_sinfo)` | `Function` | 弹出帧，构造函数 | 结束函数定义 |
| `emit_func_output(output)` | `None` | 完成函数定义 | 设置返回值 |
| `get()` | `IRModule` | 无 | 获取构建好的模块 |
| `add_param(var)` | `None` | 添加函数参数 | 在 `begin_function` 后调用 |
| `get_current_block()` | `BindingBlock` | 无 | 获取当前正在构建的块 |


**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。

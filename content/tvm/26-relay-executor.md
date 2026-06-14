> **学习目标**：
> - 理解 Relay Graph Executor、Relay VM 与 AOT Executor 三种执行模式的设计差异
> - 掌握 Relay VM 字节码编译流程与指令集架构
> - 深入理解 AOT Executor 的代码生成与内存规划策略
> - 了解执行器选择的工程权衡与适用场景
> - 掌握 VM 字节码的序列化、反序列化与跨平台部署

---

## 26.1 执行器体系概览

### 26.1.1 为什么需要多种执行器？

在 TVM 的运行时体系中，执行器（Executor）决定了编译后的模型**如何被实际执行**。不同的执行模式在以下维度上存在根本性权衡：

| 维度 | Graph Executor | Relay VM | AOT Executor |
|------|---------------|----------|--------------|
| **执行方式** | 算子逐个调用 | 字节码解释执行 | 编译为原生代码 |
| **动态控制流** | 不支持 | 支持 | 支持（有限） |
| **启动延迟** | 低 | 中 | 最低 |
| **执行效率** | 高 | 中高 | 最高 |
| **内存占用** | 中 | 中 | 低 |
| **可移植性** | 需运行时库 | 需 VM 运行时 | 仅需 C 运行时 |
| **调试能力** | 有限 | 字节码可追溯 | 源码级调试 |
| **源码位置** | `src/runtime/graph_executor/` | `src/runtime/vm/` | `src/relay/backend/` |

```
                    ┌──────────────────┐
                    │   Relay Module   │
                    │  (优化后的计算图)  │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │  Graph     │  │  Relay VM  │  │    AOT     │
     │  Executor  │  │  (虚拟机)   │  │  Executor  │
     └────────────┘  └────────────┘  └────────────┘
     算子级调度       字节码解释       编译为 C 代码
     无动态控制流     支持动态控制流   最高性能
```

<div data-component="ExecutorComparisonChart"></div>

### 26.1.2 执行器的编译流程

三种执行器共享 TVM 的前半段编译管线（Relay 优化 → 算子 lowering），在最后一步产生分化：

```python
import tvm
from tvm import relay
import numpy as np

# 定义 Relay 模块
x = relay.var("x", shape=(1, 3, 224, 224))
w = relay.var("w", shape=(64, 3, 7, 7))
conv = relay.nn.conv2d(x, w, strides=(2, 2), padding=(3, 3))
bn = relay.nn.batch_norm(conv, relay.var("gamma"), relay.var("beta"),
                         relay.var("mean"), relay.var("var"))
act = relay.nn.relu(bn[0])
func = relay.Function(relay.analysis.free_vars(act), act)
mod = tvm.IRModule.from_expr(func)

# 方式一：Graph Executor（传统方式）
with tvm.target.Target("llvm"):
    lib_graph = relay.build(mod, target="llvm", params=params)
# 输出：编译后的算子库 + 图 JSON 描述

# 方式二：Relay VM
with tvm.target.Target("llvm"):
    lib_vm = relay.vm.compile(mod, target="llvm", params=params)
# 输出：VM 字节码 + 编译后的算子库

# 方式三：AOT Executor
with tvm.target.Target("llvm"):
    lib_aot = relay.build(mod, target="llvm", params=params,
                          executor=relay.backend.Executor("aot"))
# 输出：编译后的 C 源码 / 目标代码
```

### 26.1.3 源码目录结构



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
src/
├── relay/
│   └── backend/
│       ├── graph_executor.cc      # Graph Executor 编译
│       ├── graph_executor_memory.cc # 图执行器内存规划
│       ├── vm/                     # VM 编译器
│       │   ├── compiler.cc         # Relay → VM 字节码
│       │   ├── vm.cc               # VM 执行引擎
│       │   └── transform/          # VM 特有的变换 Pass
│       └── aot/                    # AOT 编译器
│           ├── aot_lower_main.cc   # AOT 主函数生成
│           └── executor_codegen.cc # AOT 代码生成
├── runtime/
│   ├── graph_executor/
│   │   ├── graph_executor.cc       # 运行时图执行
│   │   └── graph_executor_factory.cc
│   ├── vm/
│   │   ├── vm.cc                   # VM 运行时
│   │   └── vm_module.cc            # VM Module 封装
│   └── crt/                        # CRT 微运行时（AOT 依赖）
```

---

## 26.2 Graph Executor 回顾

### 26.2.1 Graph Executor 的核心设计

Graph Executor 是 TVM 最早也最成熟的执行模式。其核心思想是将 Relay 计算图编译为一个**节点执行序列**，运行时按拓扑序依次调用每个算子：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
图 JSON 描述：
{
  "nodes": [
    {"op": "null", "name": "x"},           // node 0: 输入
    {"op": "null", "name": "w"},           // node 1: 权重
    {"op": 0, "name": "conv2d", "inputs": [[0,0,0], [1,0,0]]},  // node 2
    {"op": 1, "name": "relu", "inputs": [[2,0,0]]}              // node 3
  ],
  "arg_nodes": [0, 1],
  "node_row_ptr": [0, 0, 0, 2, 3],
  "heads": [[3, 0, 0]]
}
```

### 26.2.2 Graph Executor 的局限

Graph Executor 的设计存在三个根本性限制：

**限制一：不支持动态控制流**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 这段 Relay 代码无法用 Graph Executor 执行
def dynamic_func(x):
    # Relay 的 if 表达式需要运行时条件判断
    return relay.If(
        relay.greater(relay.op.sum(x), relay.const(0.0)),
        relay.op.nn.relu(x),
        relay.op.negative(x)
    )
# Graph Executor 假设执行顺序在编译时完全确定
```

**限制二：不支持递归与高阶函数**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 递归函数在 Graph Executor 中无法表达
def factorial(n):
    return relay.If(
        relay.equal(n, relay.const(1)),
        relay.const(1),
        n * factorial(n - 1)  # 递归调用
    )
```

**限制三：内存规划是静态的**

Graph Executor 在编译时确定所有中间张量的生命周期，无法适应运行时的动态内存需求。

---

## 26.3 Relay VM 字节码编译

### 26.3.1 VM 指令集设计

Relay VM 采用**寄存器式字节码**，每条指令操作寄存器（Register）中的值。VM 指令集定义在 `src/runtime/vm/vm.cc` 和 `include/tvm/runtime/vm.h` 中：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// include/tvm/runtime/vm/vm.h
enum class Opcode : int32_t {
  MOVE = 0,         // 寄存器间移动
  LOAD_CONST,       // 加载常量
  LOAD_CONSTI,      // 加载整数常量
  ADD,              // 加法（特殊化）
  SUB,              // 减法
  MUL,              // 乘法
  DIV,              // 除法
  CALL,             // 调用算子或函数
  TUPLE,            // 构造元组
  TUPLE_GET_ITEM,   // 获取元组元素
  IF,               // 条件分支
  GOTO,             // 无条件跳转
  INVOKE,           // 调用 VM 函数
  RET,              // 返回
  ALLOC_TENSOR,     // 分配张量
  ALLOC_TUPLE,      // 分配元组
  GET_FIELD,        // 获取字段
  THROW,            // 抛出异常
  LOAD_EXCEPTION,   // 加载异常
};
```

**VM 指令的内存布局**：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
┌─────────────────────────────────────┐
│  Opcode (4 bytes)                   │
├─────────────────────────────────────┤
│  操作数 1: 寄存器号 / 立即数          │
│  操作数 2: 寄存器号 / 立即数          │
│  操作数 3: 寄存器号 / 立即数          │
│  ...                                │
└─────────────────────────────────────┘

每条指令的长度由 Opcode 决定（变长编码）
```

### 26.3.2 Relay → VM 字节码编译

将 Relay IR 编译为 VM 字节码的过程由 `src/relay/backend/vm/compiler.cc` 中的 `VMCompiler` 实现：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// src/relay/backend/vm/compiler.cc（简化）
class VMCompiler : public ExprFunctor<void(const Expr&)> {
 public:
  CompilationOutput Compile(const IRModule& mod) {
    // Step 1: Relay 优化 Pass
    IRModule optimized = OptimizeModule(mod);

    // Step 2: 算子 lowering（Relay → TE → TIR → 机器码）
    LoweredFunctions lowered = LowerOperations(optimized);

    // Step 3: 生成 VM 函数
    for (auto& func : optimized->functions) {
      VisitExpr(func.second);  // 遍历 Relay 表达式生成字节码
    }

    // Step 4: 打包为 VM Bytecode
    return {bytecode_, lowered, constant_names_};
  }
};
```

**编译示例**：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import tvm
from tvm import relay
import numpy as np

# 定义模型
x = relay.var("x", shape=(1, 3))
w = relay.var("w", shape=(3, 5))
b = relay.var("b", shape=(5,))
y = relay.nn.dense(x, w)
z = relay.nn.relu(y + b)
func = relay.Function([x, w, b], z)
mod = tvm.IRModule.from_expr(func)

# 编译为 VM
exec = relay.vm.compile(mod, target="llvm", params={})

# 查看字节码
# exec.bytecode 会显示 VM 指令序列
```

### 26.3.3 字节码编译的详细流程

Relay VM 编译器将每种 Relay 表达式映射为对应的 VM 指令序列：

**函数调用编译**：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# Relay 表达式：
#   y = relay.nn.conv2d(x, w)
#
# 编译为 VM 字节码：
#   LOAD_CONST  r1, <weight_const_idx>   ; 加载权重常量
#   CALL        r2, "conv2d", [r0, r1]   ; 调用 conv2d 算子
#   MOVE        r0, r2                   ; 结果移入 r0
```

**条件分支编译**：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# Relay 表达式：
#   relay.If(cond, true_branch, false_branch)
#
# 编译为 VM 字节码：
#   IF          r0, <true_offset>, <false_offset>
#   ; true 分支的字节码...
#   GOTO        <end_offset>
#   ; false 分支的字节码...
#   ; end:
```

**Let 表达式编译**：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# Relay 表达式：
#   let a = f(x)
#   let b = g(a)
#   b
#
# 编译为 VM 字节码：
#   CALL        r1, "f", [r0]     ; a = f(x)
#   CALL        r2, "g", [r1]     ; b = g(a)
#   RET         r2                 ; return b
```

### 26.3.4 VM 函数与闭包

Relay VM 为每个函数生成一个 `VMFunction`，包含字节码和元数据：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// include/tvm/runtime/vm.h
struct VMFunction {
  std::string name;              // 函数名
  Index num_packed_args;         // PackedFunc 参数数
  Index num_args;                // 总参数数
  std::vector<Instruction> instructions;  // 字节码指令序列
  std::vector<Index> register_file_size;  // 寄存器文件大小
};
```

闭包（Closure）在 VM 中表示为捕获环境 + 函数指针：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# Relay 中的闭包
def make_adder(n):
    # 返回一个闭包，捕获 n
    return relay.Function([x], x + n)

adder_5 = make_adder(relay.const(5))
result = adder_5(relay.const(3))  # 结果为 8
```

### 26.3.5 VM 运行时执行

VM 运行时在 `src/runtime/vm/vm.cc` 中实现，核心是一个**指令分发循环**：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// src/runtime/vm/vm.cc（简化）
void VirtualMachine::Run() {
  while (true) {
    const auto& instr = code_[pc_];  // 取指令
    switch (instr.opcode) {
      case Opcode::MOVE:
        WriteRegister(instr.dst, ReadRegister(instr.src));
        pc_++;
        break;

      case Opcode::LOAD_CONST:
        WriteRegister(instr.dst, constants_[instr.const_index]);
        pc_++;
        break;

      case Opcode::CALL: {
        // 获取 PackedFunc
        auto func = GetPackedFunc(instr.func_name);
        // 收集参数
        std::vector<TVMValue> args;
        for (auto reg : instr.args) {
          args.push_back(ReadRegister(reg));
        }
        // 调用算子
        func->CallPacked(args, &ret_val);
        WriteRegister(instr.dst, ret_val);
        pc_++;
        break;
      }

      case Opcode::IF: {
        auto cond = ReadRegister(instr.cond).operator bool();
        pc_ = cond ? instr.true_offset : instr.false_offset;
        break;
      }

      case Opcode::GOTO:
        pc_ = instr.offset;
        break;

      case Opcode::RET:
        return;  // 函数返回
    }
  }
}
```

**VM 执行的栈帧管理**：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
调用栈：
┌─────────────────────────┐
│  VMFunction: main       │
│  PC: 15                 │
│  Registers: [x, w, y]   │
├─────────────────────────┤
│  VMFunction: block_0    │
│  PC: 3                  │
│  Registers: [a, b, c]   │
├─────────────────────────┤
│  VMFunction: block_1    │
│  PC: 0                  │
│  Registers: [d, e]      │
└─────────────────────────┘
```

---

## 26.4 VM 特有的优化 Pass

### 26.4.1 Lambda Lift Pass

VM 编译器需要将嵌套的局部函数提升为顶层函数，以便为每个函数生成独立的字节码：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 优化前：嵌套函数
def outer(x):
    def inner(y):
        return x + y
    return inner(x)

# Lambda Lift 后：所有函数提升为顶层
def inner(x, y):  # 捕获的变量 x 变为显式参数
    return x + y

def outer(x):
    return inner(x, x)
```

**源码位置**：`src/relay/backend/vm/lambda_lift.cc`



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// src/relay/backend/vm/lambda_lift.cc（简化）
class LambdaLifter : public ExprMutator {
  Expr VisitExpr_(const FunctionNode* func_node) override {
    // 将自由变量提升为函数参数
    auto free_vars = FreeVars(GetRef<Function>(func_node));
    // 创建新的参数列表
    std::vector<Var> new_params;
    for (auto var : free_vars) {
      new_params.push_back(var);
    }
    for (auto param : func_node->params) {
      new_params.push_back(param);
    }
    // 注册为顶层函数
    auto lifted_name = GenerateName(func_node);
    module_->Add(lifted_name, new_func);
    // 替换为函数引用
    return FunctionCall(lifted_name, new_params);
  }
};
```

### 26.4.2 算子内联与特化

VM 编译器对特定模式的算子调用进行内联和特化优化：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 特化前：通用的算子调用
y = relay.op.add(x, relay.const(1.0))

# 特化后：使用特化的 ADD 指令而非通用 CALL
# ADD r2, r0, r1  （直接使用 ADD 指令，避免 PackedFunc 调用开销）
```

**支持特化的算子**：`ADD`、`SUB`、`MUL`、`DIV`、`TUPLE`、`TUPLE_GET_ITEM`

### 26.4.3 内存规划 Pass

VM 编译器实现了基于**寄存器生命周期分析**的内存规划：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 寄存器生命周期分析示例
# 指令序列：
# 0: ALLOC_TENSOR r0, shape=(1,64,112,112)  ; r0 生命开始
# 1: CALL r1, "conv2d", [r_input, r_weight] ; r1 生命开始
# 2: CALL r2, "relu", [r1]                  ; r2 生命开始，r1 可回收
# 3: MOVE r0, r2                             ; r0 生命结束（旧值），r2 可回收
# 4: RET r0

# 内存规划结果：r1 和 r2 可以复用同一块内存
```

**源码位置**：`src/relay/backend/vm/transform/rewrite_packed_call.cc`

---

## 26.5 AOT Executor

### 26.5.1 AOT 设计理念

AOT（Ahead-of-Time）Executor 的核心思想是将 Relay 计算图**直接编译为 C 源码**，完全消除运行时的解释开销：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
传统 VM 执行流程：
  Relay IR → 字节码 → [VM 解释执行] → 结果
  （需要 VM 运行时库）

AOT 执行流程：
  Relay IR → C 源码 → [编译器编译] → 原生代码 → [直接执行] → 结果
  （仅需标准 C 库）
```

**AOT 的核心优势**：

| 优势 | 说明 |
|------|------|
| **零解释开销** | 无字节码分发循环，直接执行机器码 |
| **最小依赖** | 仅需 CRT（C Runtime）库，无 VM 运行时 |
| **嵌入式友好** | 可部署到无操作系统的裸机环境 |
| **可调试性** | 生成的 C 代码可直接用 gdb 调试 |
| **确定性** | 执行路径完全确定，无运行时分支 |

### 26.5.2 AOT 代码生成

AOT 代码生成由 `src/relay/backend/aot/` 中的代码实现：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import tvm
from tvm import relay

# 模型定义
x = relay.var("x", shape=(1, 3, 224, 224))
w = relay.var("w", shape=(64, 3, 7, 7))
conv = relay.nn.conv2d(x, w, strides=(2, 2), padding=(3, 3))
act = relay.nn.relu(conv)
func = relay.Function([x, w], act)
mod = tvm.IRModule.from_expr(func)

# AOT 编译
with tvm.target.Target("c"):
    lib = relay.build(
        mod,
        target="c",
        params={},
        executor=relay.backend.Executor("aot", {"interface-api": "packed"})
    )

# 查看生成的 C 代码
# lib.get_lib().get_source() 返回生成的 C 源码
```

**生成的 C 代码结构**（示意）：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```c
// tvmgen_default.c（自动生成）
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/crt/ndarray.h"

// 全局工作空间
static uint8_t workspace[WORKSPACE_SIZE];

// 子图函数
static void tvmgen_default_conv2d(
    DLTensor* x,
    DLTensor* w,
    DLTensor* conv_output
) {
    // 调用编译后的 conv2d kernel
    tvmgen_default_conv2d_kernel(
        (float*)x->data,
        (float*)w->data,
        (float*)conv_output->data
    );
}

static void tvmgen_default_relu(
    DLTensor* conv_output,
    DLTensor* output
) {
    tvmgen_default_relu_kernel(
        (float*)conv_output->data,
        (float*)output->data
    );
}

// 主入口函数
int tvmgen_default_run(
    DLTensor* x,
    DLTensor* w,
    DLTensor* output
) {
    // 分配中间张量（使用工作空间）
    DLTensor conv_output;
    conv_output.data = workspace;
    conv_output.shape = ...;

    // 按拓扑序执行
    tvmgen_default_conv2d(x, w, &conv_output);
    tvmgen_default_relu(&conv_output, output);

    return 0;
}
```

### 26.5.3 AOT 内存规划

AOT Executor 实现了基于**图着色**的内存规划算法，最小化中间张量的内存占用：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 内存规划示例
# 计算图：
#   x → conv2d → relu → output1
#   x → conv2d → sigmoid → output2

# 内存规划结果：
#   conv2d 的输出被 relu 和 sigmoid 共享
#   relu 和 sigmoid 不会同时执行，可复用同一输出缓冲区

# 内存分配表：
# ┌─────────────────┬──────────┬────────┬──────────┐
# │ 张量             │ 偏移量    │ 大小    │ 生命周期   │
# ├─────────────────┼──────────┼────────┼──────────┤
# │ x (输入)         │ 0        │ 602112 │ 全程      │
# │ w (权重)         │ 602112   │ 18816  │ 全程      │
# │ conv_output     │ 620928   │ 3211264│ [0, 2]   │
# │ relu_output     │ 620928   │ 3211264│ [1, 2]   │ ← 复用 conv_output
# └─────────────────┴──────────┴────────┴──────────┘
```

**源码位置**：`src/relay/backend/graph_executor_memory.cc`



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// 基于图着色的内存规划（简化）
class StorageAllocator : public StorageAllocaBaseVisitor {
 public:
  void Plan(const IndexedForwardGraph& graph) {
    // 1. 计算每个节点的活跃范围
    // 2. 使用图着色算法分配存储
    // 3. 输出偏移量表
    for (auto node : graph.nodes) {
      auto storage = GetStorage(node);
      if (storage == nullptr) {
        // 分配新存储
        storage = AllocStorage(node->size);
      }
      node->storage = storage;
      // 当最后一个使用者完成时释放
      if (IsLastUse(node)) {
        FreeStorage(storage);
      }
    }
  }
};
```

### 26.5.4 AOT 接口模式

AOT Executor 支持两种接口模式：

**Packed 接口**（通用，与 PackedFunc 兼容）：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```c
// 使用 TVMValue 数组传递参数
int tvmgen_default_run(
    TVMValue* args,
    int* type_codes,
    int num_args
) {
    DLTensor* x = (DLTensor*)args[0].v_handle;
    DLTensor* w = (DLTensor*)args[1].v_handle;
    DLTensor* output = (DLTensor*)args[2].v_handle;
    // ...
}
```

**C 接口**（嵌入式友好，类型安全）：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```c
// 直接使用 DLTensor 指针
int tvmgen_default_run(
    DLTensor* x,
    DLTensor* w,
    DLTensor* output
) {
    // 直接使用，无需类型转换
    // ...
}
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 选择接口模式
lib = relay.build(
    mod, target="c",
    executor=relay.backend.Executor("aot", {
        "interface-api": "c",      # 或 "packed"
        "unpacked-api": True,
        "workspace-byte-alignment": 8,
    })
)
```

---

## 26.6 VM 字节码序列化与部署

### 26.6.1 字节码序列化

VM 字节码可以序列化为二进制文件，用于跨设备部署：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 编译并保存 VM
exec = relay.vm.compile(mod, target="llvm", params=params)

# 序列化为文件
bytecode = exec.bytecode
with open("model.vm", "wb") as f:
    f.write(bytecode)

# 在目标设备上加载
loaded_exec = tvm.runtime.load_module("model.vm")
```

**字节码文件格式**：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
┌──────────────────────────────────┐
│  Magic Number (4 bytes)          │
│  Version (4 bytes)               │
├──────────────────────────────────┤
│  函数表 (Function Table)          │
│  ├── 函数数量 (4 bytes)           │
│  ├── 函数 1:                     │
│  │   ├── 名称长度 + 名称          │
│  │   ├── 指令数量                 │
│  │   ├── 指令序列                 │
│  │   └── 寄存器文件大小           │
│  ├── 函数 2: ...                 │
│  └── 函数 N: ...                 │
├──────────────────────────────────┤
│  常量表 (Constants Table)         │
│  ├── 常量数量                    │
│  └── 常量数据 (NDArray)           │
├──────────────────────────────────┤
│  全局符号表                       │
│  └── PackedFunc 名称映射          │
└──────────────────────────────────┘
```

### 26.6.2 Module 系统集成

VM 通过 `vm_module.cc` 将 VM 功能封装为 TVM Module 系统的一部分：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// src/runtime/vm/vm_module.cc（简化）
class VMModule : public ModuleNode {
 public:
  // 获取 VM 中注册的函数
  PackedFunc GetFunction(const std::string& name,
                         const ObjectPtr<Object>& sptr) override {
    if (name == "invoke") {
      // 返回主执行函数
      return PackedFunc([this](TVMArgs args, TVMValue* rv) {
        this->Run(args, rv);
      });
    }
    // 查找 VM 函数表
    auto it = func_map_.find(name);
    if (it != func_map_.end()) {
      return MakePackedFunc(it->second);
    }
    return nullptr;
  }

  const char* type_key() const override { return "vm"; }

 private:
  VirtualMachine vm_;
  std::unordered_map<std::string, Index> func_map_;
};
```

---

## 26.7 执行器选择指南

### 26.7.1 决策矩阵

选择执行器时需要考虑以下因素：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def choose_executor(model, target_device, constraints):
    """根据场景选择最合适的执行器"""

    # 检查是否需要动态控制流
    has_dynamic_control_flow = check_dynamic_control_flow(model)

    # 检查目标设备资源
    is_embedded = target_device.memory < 1024 * 1024  # < 1MB

    if is_embedded:
        # 嵌入式设备：AOT 是最佳选择
        return "aot"

    if has_dynamic_control_flow:
        # 需要动态控制流：必须使用 VM
        return "vm"

    # 静态模型 + 资源充足：Graph Executor 或 AOT
    if constraints.get("minimize_latency"):
        return "aot"      # AOT 启动最快
    elif constraints.get("ease_of_debugging"):
        return "vm"       # VM 支持逐步调试
    else:
        return "graph"    # Graph Executor 最成熟
```

### 26.7.2 性能对比

在不同场景下三种执行器的相对性能：

| 场景 | Graph Executor | Relay VM | AOT |
|------|---------------|----------|-----|
| **静态 CNN 推理** | ★★★★ | ★★★ | ★★★★★ |
| **动态 NLP 模型** | ✗ | ★★★★ | ★★★ |
| **嵌入式 MCU** | ✗ | ★★ | ★★★★★ |
| **调试与分析** | ★★ | ★★★★★ | ★★★ |
| **批量推理** | ★★★★ | ★★★ | ★★★★★ |
| **交互式推理** | ★★★★ | ★★★★ | ★★★ |

### 26.7.3 未来趋势：Relax VM

随着 TVM Unity 的推进，Relax IR 正在取代 Relay，其配套的 Relax VM 提供了更现代的执行模型：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# Relax VM（TVM Unity 方向）
from tvm import relax

# Relax 计算图定义
with relax.Builder() as builder:
    with relax.dataflow():
        lv0 = relax.op.nn.conv2d(x, w)
        lv1 = relax.op.nn.relu(lv0)
        builder.emit_output(lv1)
    func = builder.get_func()

# Relax VM 编译
ex = relax.vm.build(mod, target="llvm")
```

---

## 26.8 高级主题

### 26.8.1 VM Profiling 支持

Relay VM 内置了 profiling 支持，可以统计每条指令的执行时间：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import tvm
from tvm import relay, runtime

# 启用 VM profiling
exec = relay.vm.compile(mod, target="llvm", params=params)

# 创建 VM 实例并启用 profiling
vm = runtime.vm.VirtualMachine(exec, tvm.cpu())
vm.set_profiler(True)

# 执行推理
result = vm.invoke("main", input_data)

# 获取 profiling 结果
profile_data = vm.get_exec_stats()
# profile_data 包含每个函数、每条指令的执行时间
```

### 26.8.2 VM 与 RPC 集成

VM 可以通过 RPC 远程执行，用于开发板调试：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 连接远程设备
remote = tvm.rpc.connect("raspberry-pi", 9190)

# 在远程设备上创建 VM
vm = runtime.vm.VirtualMachine(exec, remote.device("cpu"))

# 远程执行
result = vm.invoke("main", remote_input)
```

### 26.8.3 AOT 与 CMSIS-NN 集成

AOT 生成的 C 代码可以直接调用 CMSIS-NN 等嵌入式算子库：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 使用 CMSIS-NN 后端的 AOT 编译
with tvm.target.Target("cmsis-nn"):
    lib = relay.build(
        mod,
        target="cmsis-nn",
        params=params,
        executor=relay.backend.Executor("aot", {
            "interface-api": "c",
            "unpacked-api": True,
        })
    )

# 生成的 C 代码会调用 CMSIS-NN 的卷积函数
# arm_convolve_s8()、arm_relu_q7() 等
```

---

## 26.9 实战示例：端到端执行器对比

### 26.9.1 完整的三种执行器对比实验



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import tvm
from tvm import relay
import numpy as np
import time

def build_and_benchmark(mod, params, target_str, executor_type):
    """编译并基准测试指定执行器"""
    target = tvm.target.Target(target_str)

    if executor_type == "graph":
        with target:
            lib = relay.build(mod, target=target, params=params)
    elif executor_type == "vm":
        with target:
            lib = relay.vm.compile(mod, target=target, params=params)
    elif executor_type == "aot":
        with target:
            lib = relay.build(
                mod, target=target, params=params,
                executor=relay.backend.Executor("aot")
            )

    # 部署
    dev = tvm.device(target_str, 0)
    if executor_type == "vm":
        from tvm.runtime.vm import VirtualMachine
        vm = VirtualMachine(lib, dev)
        # Warmup
        for _ in range(10):
            vm.invoke("main", input_data)
        # Benchmark
        start = time.time()
        for _ in range(100):
            result = vm.invoke("main", input_data)
        elapsed = time.time() - start
    else:
        from tvm.contrib import graph_executor
        module = graph_executor.GraphModule(lib["default"](dev))
        module.set_input("x", input_data)
        # Warmup
        for _ in range(10):
            module.run()
        # Benchmark
        start = time.time()
        for _ in range(100):
            module.run()
        elapsed = time.time() - start
        result = module.get_output(0)

    return elapsed / 100, result

# 测试模型：ResNet-18 的简化版本
def create_test_model():
    data = relay.var("data", shape=(1, 3, 224, 224))
    weight1 = relay.var("weight1", shape=(64, 3, 7, 7))
    conv1 = relay.nn.conv2d(data, weight1, strides=(2, 2), padding=(3, 3))
    bn1 = relay.nn.batch_norm(conv1, *[relay.var(f"bn_{i}") for i in range(4)])
    relu1 = relay.nn.relu(bn1[0])
    pool = relay.nn.global_avg_pool2d(relu1)
    flat = relay.op.nn.batch_flatten(pool)
    weight2 = relay.var("weight2", shape=(1000, 64))
    dense = relay.nn.dense(flat, weight2)
    func = relay.Function(relay.analysis.free_vars(dense), dense)
    return tvm.IRModule.from_expr(func)

# 运行对比
mod = create_test_model()
params = {}  # 实际使用时需要加载预训练权重

for exe in ["graph", "vm", "aot"]:
    try:
        avg_time, _ = build_and_benchmark(mod, params, "llvm", exe)
        print(f"{exe:8s}: {avg_time*1000:.2f} ms")
    except Exception as e:
        print(f"{exe:8s}: 编译失败 - {e}")
```

### 26.9.2 AOT 嵌入式部署完整流程



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import tvm
from tvm import relay
from tvm.micro import export_model_library_format
import tarfile

# 1. 定义模型
x = relay.var("x", shape=(1, 1, 28, 28))  # MNIST
w1 = relay.var("w1", shape=(32, 1, 3, 3))
conv = relay.nn.conv2d(x, w1, padding=(1, 1))
relu = relay.nn.relu(conv)
pool = relay.nn.max_pool2d(relu, pool_size=(2, 2))
flat = relay.nn.batch_flatten(pool)
w2 = relay.var("w2", shape=(10, 32*14*14))
out = relay.nn.dense(flat, w2)
func = relay.Function(relay.analysis.free_vars(out), out)
mod = tvm.IRModule.from_expr(func)

# 2. AOT 编译（目标：C 代码）
with tvm.target.Target("c"):
    lib = relay.build(
        mod, target="c",
        params={},
        executor=relay.backend.Executor("aot", {
            "interface-api": "c",
            "unpacked-api": True,
        }),
        runtime=relay.backend.Runtime("crt", {"system-lib": True})
    )

# 3. 导出为 Model Library Format
mlf = export_model_library_format(lib, "mnist_model.tar")

# 4. 该 tar 文件包含：
#    - src/: 生成的 C 源码
#    - lib/: 编译后的目标文件
#    - graph/: 图描述信息
#    - metadata/: 模型元数据
```

---

## 26.99 文字内容强化：Relay 执行器 的工程化理解

这一节用于把前文的 API、IR、Pass、Runtime 和部署片段串联为更完整的工程叙事。
很多学习者第一次阅读 TVM 文档时会觉得示例代码很多，但真正上线时仍然不知道如何判断方案是否可靠。
原因在于 TVM 不是单个推理库，而是一条从模型语义到硬件代码的编译链路。
链路越长，越需要把每一步的业务目标、内部机制、适用边界和失败模式说清楚。

### 26.99.1 代码解读的阅读方法

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

- 围绕“Graph Executor、Relay VM 与 AOT Executor 的边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“运行时模块、参数绑定与内存规划的关系”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“动态控制流、静态图执行与嵌入式 AOT 的取舍”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“调试、序列化和跨平台部署的工程约束”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 26.99.2 业务意义

1. Relay 执行器 的业务价值不只是让模型跑得更快，而是让同一个模型可以在不同成本、功耗和延迟约束下交付。
2. 在服务器场景中，核心指标通常是吞吐、P95/P99 延迟、资源利用率和多租户隔离。
3. 在移动端场景中，核心指标通常是首帧时间、持续发热、内存峰值和包体大小。
4. 在嵌入式场景中，核心指标通常是 Flash 占用、静态内存、实时性和掉电恢复能力。
5. 在云端批处理场景中，编译时间可以接受更长，但调优记录和缓存复用变得非常重要。
6. 在在线服务场景中，编译产物需要可回滚、可审计、可灰度，而不能只在开发机上验证。
7. 业务方关心的是 SLA、成本和稳定性，编译器工程师关心的是 IR 正确性、优化空间和后端能力。
8. 优秀的 TVM 项目需要把这两类语言翻译成共同的指标体系。
9. 当优化收益只有少量百分点时，应评估它是否值得引入新的维护复杂度。
10. 当优化收益很大但只在少数输入上成立时，应评估输入分布变化后的风险。

- 围绕“Graph Executor、Relay VM 与 AOT Executor 的边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“运行时模块、参数绑定与内存规划的关系”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“动态控制流、静态图执行与嵌入式 AOT 的取舍”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“调试、序列化和跨平台部署的工程约束”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 26.99.3 TVM 内部机制

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

- 围绕“Graph Executor、Relay VM 与 AOT Executor 的边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“运行时模块、参数绑定与内存规划的关系”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“动态控制流、静态图执行与嵌入式 AOT 的取舍”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“调试、序列化和跨平台部署的工程约束”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 26.99.4 适用场景

1. 当模型结构相对稳定、目标硬件明确、性能收益可以通过基准测试确认时，Relay 执行器 相关技术最容易发挥价值。
2. 当团队需要支持多种硬件后端时，TVM 的统一 IR 和 Target 抽象可以降低重复适配成本。
3. 当模型中存在框架运行时开销、算子融合机会或布局转换冗余时，编译优化通常能带来明显收益。
4. 当部署环境不能依赖完整 Python 栈时，AOT、CRT 或导出后的 runtime artifact 更有意义。
5. 当硬件厂商提供高性能库但模型图需要复杂切分时，BYOC 和外部 codegen 是常见选择。
6. 当输入形状变化频繁时，应提前设计 shape 策略，而不是在上线前才补动态形状支持。
7. 当模型版本迭代频繁时，应把编译、调优、验证和发布纳入 CI/CD。
8. 当业务对精度非常敏感时，应把优化收益和数值回归一起评估。
9. 当系统存在多模型串联时，应评估端到端 pipeline，而不是只优化单个模型。
10. 当部署设备数量很大时，编译产物的一致性和可追踪性比单次实验性能更重要。

- 围绕“Graph Executor、Relay VM 与 AOT Executor 的边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“运行时模块、参数绑定与内存规划的关系”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“动态控制流、静态图执行与嵌入式 AOT 的取舍”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“调试、序列化和跨平台部署的工程约束”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 26.99.5 限制条件

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

- 围绕“Graph Executor、Relay VM 与 AOT Executor 的边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“运行时模块、参数绑定与内存规划的关系”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“动态控制流、静态图执行与嵌入式 AOT 的取舍”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“调试、序列化和跨平台部署的工程约束”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 26.99.6 工程经验

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

- 围绕“Graph Executor、Relay VM 与 AOT Executor 的边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“运行时模块、参数绑定与内存规划的关系”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“动态控制流、静态图执行与嵌入式 AOT 的取舍”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“调试、序列化和跨平台部署的工程约束”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 26.99.7 常见误区

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

- 围绕“Graph Executor、Relay VM 与 AOT Executor 的边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“运行时模块、参数绑定与内存规划的关系”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“动态控制流、静态图执行与嵌入式 AOT 的取舍”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“调试、序列化和跨平台部署的工程约束”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 26.99.8 生产部署注意事项

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

- 围绕“Graph Executor、Relay VM 与 AOT Executor 的边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“运行时模块、参数绑定与内存规划的关系”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“动态控制流、静态图执行与嵌入式 AOT 的取舍”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“调试、序列化和跨平台部署的工程约束”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 26.99.9 与同类系统对比

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

- 围绕“Graph Executor、Relay VM 与 AOT Executor 的边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“运行时模块、参数绑定与内存规划的关系”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“动态控制流、静态图执行与嵌入式 AOT 的取舍”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“调试、序列化和跨平台部署的工程约束”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 26.99.10 章节复盘

1. 回到本章，Relay 执行器 的关键不是记住所有 API，而是理解为什么这些 API 会出现在编译链路的这个位置。
2. 当你看到一段代码时，应能说出它改变了模型语义、调度空间、内存布局、运行时入口还是部署产物。
3. 当你看到一个性能数字时，应能说出它的测试输入、硬件状态、计时方法和误差范围。
4. 当你看到一个优化 pass 时，应能说出它依赖的前置假设和可能破坏的边界条件。
5. 当你准备上线时，应能说出失败后如何回滚、如何复现、如何定位和如何与业务方沟通影响。
6. 这套思维比单个示例更重要，因为 TVM 的 API 会演进，但编译部署的工程约束长期稳定。
7. 后续学习中，可以把每一章都转化为一张决策表：何时使用、收益来自哪里、风险是什么、如何验证。
8. 只有把代码、机制和工程策略放在一起，TVM 才不只是工具箱，而是可运行的生产系统。
9. 因此，本章新增的文字说明应作为阅读代码段的上下文，而不是替代对原始代码的逐行理解。
10. 如果遇到与示例不一致的实际项目，应优先回到模型约束和目标硬件，而不是机械套用章节流程。

- 围绕“Graph Executor、Relay VM 与 AOT Executor 的边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“运行时模块、参数绑定与内存规划的关系”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“动态控制流、静态图执行与嵌入式 AOT 的取舍”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“调试、序列化和跨平台部署的工程约束”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。


## 26.10 本章小结

本章深入解析了 TVM 的三种执行器体系：

1. **Graph Executor**：成熟的算子级调度，适合静态模型推理
2. **Relay VM**：字节码解释执行，支持动态控制流和递归
3. **AOT Executor**：编译为 C 源码，零解释开销，适合嵌入式部署

**关键源码索引**：

| 模块 | 源码路径 |
|------|---------|
| Graph Executor 运行时 | `src/runtime/graph_executor/graph_executor.cc` |
| VM 编译器 | `src/relay/backend/vm/compiler.cc` |
| VM 运行时 | `src/runtime/vm/vm.cc` |
| VM Module | `src/runtime/vm/vm_module.cc` |
| AOT 代码生成 | `src/relay/backend/aot/` |
| Lambda Lift | `src/relay/backend/vm/lambda_lift.cc` |
| 内存规划 | `src/relay/backend/graph_executor_memory.cc` |

<div data-component="ExecutorArchitectureDiagram"></div>

---

## 26.11 VM 指令编码详解

### 26.11.1 指令编码格式

每条 VM 指令使用变长编码，操作码决定后续操作数的数量和类型：

```
指令编码布局：
┌─────────────────────────────────────────────────────────┐
│ Opcode (8 bits) │ 操作数 1 │ 操作数 2 │ ... │ 操作数 N │
└─────────────────────────────────────────────────────────┘

CALL 指令编码示例：
┌────────┬──────────┬────────────┬────────────┬────────────┐
│ 0x09   │ dst_reg  │ func_name  │ num_args   │ arg_regs[] │
│ (8bit) │ (16bit)  │ (32bit)    │ (16bit)    │ (16bit×N)  │
└────────┴──────────┴────────────┴────────────┴────────────┘

IF 指令编码示例：
┌────────┬──────────┬──────────────┬───────────────┐
│ 0x0A   │ cond_reg │ true_offset  │ false_offset  │
│ (8bit) │ (16bit)  │ (32bit)      │ (32bit)       │
└────────┴──────────┴──────────────┴───────────────┘
```

### 26.11.2 寄存器分配

VM 编译器在编译阶段为每个中间值分配寄存器号：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 寄存器分配示例
# Relay 表达式：
#   let a = conv2d(x, w)     → r0
#   let b = batch_norm(a)    → r1
#   let c = relu(b)          → r2
#   let d = pool2d(c)        → r3
#   let e = dense(d, w2)     → r4
#   e

# 寄存器分配结果：
# r0: conv2d 输出（生命周期 [0, 1]）
# r1: batch_norm 输出（生命周期 [1, 2]）
# r2: relu 输出（生命周期 [2, 3]）
# r3: pool2d 输出（生命周期 [3, 4]）
# r4: dense 输出（生命周期 [4, 5]）

# 寄存器复用：
# r0 和 r1 可以复用（生命周期不重叠）
# 但为了简单，VM 编译器通常不进行寄存器复用
```

### 26.11.3 常量池管理

VM 将常量（模型参数、字面量）存储在常量池中：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// src/runtime/vm/vm.cc
struct VMState {
  // 常量池
  std::vector<ObjectRef> constants;
  // 寄存器文件（按函数帧组织）
  std::vector<TVMValue> registers;
  // 调用栈
  std::vector<StackFrame> call_stack;
  // 当前函数索引
  Index func_index;
  // 程序计数器
  Index pc;
};
```

**常量池的序列化**：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
常量池格式：
┌───────────────────────────────┐
│ 常量数量 (4 bytes)             │
├───────────────────────────────┤
│ 常量 0:                       │
│   ├── 数据类型 (4 bytes)       │
│   ├── 形状维度数 (4 bytes)     │
│   ├── 形状 (8 × ndim bytes)   │
│   └── 数据 (size × elem_size) │
├───────────────────────────────┤
│ 常量 1: ...                   │
└───────────────────────────────┘
```

---

## 26.12 Graph Executor 深度解析

### 26.12.1 图执行器的数据结构

Graph Executor 的核心数据结构定义在 `src/runtime/graph_executor/graph_executor.cc` 中：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// src/runtime/graph_executor/graph_executor.cc
class GraphExecutor : public ModuleNode {
 public:
  // 图结构
  struct Node {
    uint32_t op_type;           // 算子类型
    std::string name;           // 节点名称
    std::vector<NodeEntry> inputs;  // 输入边
  };

  struct NodeEntry {
    uint32_t node_id;    // 源节点 ID
    uint32_t index;      // 源节点输出索引
    uint32_t version;    // 版本号
  };

  // 存储管理
  struct StorageEntry {
    std::vector<int64_t> shape;
    DLDataType dtype;
    DLDevice device;
    void* data;           // 数据指针
    size_t size;          // 大小（字节）
  };

  // 执行计划
  std::vector<Node> nodes_;
  std::vector<StorageEntry> storage_pool_;
  std::vector<uint32_t> node_row_ptr_;
  std::vector<uint32_t> input_nodes_;
  std::vector<uint32_t> output_nodes_;
  std::vector<OpState> op_states_;
};
```

### 26.12.2 图执行器的执行流程



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// src/runtime/graph_executor/graph_executor.cc（简化）
void GraphExecutor::Run() {
  // 按拓扑序执行所有节点
  for (size_t i = 0; i < nodes_.size(); i++) {
    const auto& node = nodes_[i];

    // 收集输入
    std::vector<DLTensor> inputs;
    for (const auto& entry : node.inputs) {
      auto& storage = storage_pool_[entry.node_id];
      DLTensor tensor;
      tensor.data = storage.data;
      tensor.shape = storage.shape.data();
      tensor.ndim = storage.shape.size();
      tensor.dtype = storage.dtype;
      inputs.push_back(tensor);
    }

    // 获取输出存储
    auto& output = storage_pool_[i];
    DLTensor output_tensor;
    output_tensor.data = output.data;
    output_tensor.shape = output.shape.data();
    output_tensor.ndim = output.shape.size();
    output_tensor.dtype = output.dtype;

    // 调用算子
    if (node.op_type == kTvmOp) {
      auto func = GetPackedFunc(node.name);
      func->CallPacked(inputs, &output_tensor);
    }
  }
}
```

### 26.12.3 图执行器的内存规划算法

Graph Executor 使用**图着色算法**进行内存规划：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 内存规划算法（概念性描述）
def plan_memory(graph):
    """基于活跃度分析的内存规划"""

    # Step 1: 计算每个节点的活跃范围
    # 节点 i 在 [first_use(i), last_use(i)] 期间活跃
    live_ranges = {}
    for i, node in enumerate(graph.nodes):
        first_use = i
        last_use = i
        # 查找所有使用此节点输出的节点
        for j in range(i + 1, len(graph.nodes)):
            if any(inp.node_id == i for inp in graph.nodes[j].inputs):
                last_use = j
        live_ranges[i] = (first_use, last_use)

    # Step 2: 构建干涉图
    # 如果两个节点的活跃范围重叠，则它们不能共享内存
    interference = {}
    for i in range(len(graph.nodes)):
        for j in range(i + 1, len(graph.nodes)):
            if ranges_overlap(live_ranges[i], live_ranges[j]):
                interference[(i, j)] = True

    # Step 3: 图着色分配
    # 使用贪心着色算法
    allocation = greedy_coloring(graph, interference)

    # Step 4: 计算偏移量
    memory_plan = compute_offsets(allocation, graph)

    return memory_plan
```

### 26.12.4 算子调度策略

Graph Executor 支持两种算子调度模式：

**同步调度**：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 同步执行：逐个算子执行，等待完成后再执行下一个
def run_sync(graph_executor):
    for node in graph_executor.nodes:
        execute_operator(node)
        wait_for_completion()  # 阻塞等待
```

**异步调度**（GPU 场景）：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 异步执行：利用 GPU 的流机制重叠计算和数据传输
def run_async(graph_executor):
    stream = create_gpu_stream()
    for node in graph_executor.nodes:
        execute_operator_async(node, stream)
    synchronize(stream)  # 最后统一等待
```

---

## 26.13 AOT 代码生成详解

### 26.13.1 AOT 主函数生成

AOT 编译器为整个计算图生成一个主入口函数：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// src/relay/backend/aot/aot_lower_main.cc（简化）
class AOTCodeGenerator {
 public:
  void GenerateMainFunction(const IRModule& mod) {
    // 1. 分析计算图拓扑
    auto topo_order = TopologicalSort(mod);

    // 2. 生成内存分配代码
    EmitWorkspaceAllocation();

    // 3. 生成子图调用序列
    for (auto& node : topo_order) {
      if (IsRelayOp(node)) {
        EmitOpCall(node);
      } else if (IsControlFlow(node)) {
        EmitControlFlow(node);
      }
    }

    // 4. 生成清理代码
    EmitCleanup();
  }

 private:
  void EmitOpCall(const Node& node) {
    // 生成算子调用代码
    code_ << "  tvmgen_default_" << SanitizeName(node.name)
          << "(";
    // 输出参数
    for (auto& out : node.outputs) {
      code_ << "&" << GetBufferName(out) << ", ";
    }
    // 输入参数
    for (auto& inp : node.inputs) {
      code_ << "&" << GetBufferName(inp) << ", ";
    }
    // 工作空间
    code_ << "workspace);\n";
  }
};
```

### 26.13.2 中间张量的生命周期管理

AOT 使用**引用计数**管理中间张量的生命周期：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```c
// 生成的 C 代码示例（带生命周期注释）
int tvmgen_default_run(
    DLTensor* arg0,   // 输入 x
    DLTensor* arg1,   // 权重 w
    DLTensor* output  // 输出
) {
    // 中间张量 0：conv2d 输出（生命周期 [0, 2]）
    DLTensor intermediate_0;
    intermediate_0.data = workspace + 0;
    intermediate_0.shape = (int64_t[]){1, 64, 112, 112};
    intermediate_0.ndim = 4;
    intermediate_0.dtype = (DLDataType){kDLFloat, 32, 1};

    // 中间张量 1：relu 输出（生命周期 [1, 2]）
    // 可以复用 intermediate_0 的内存
    DLTensor intermediate_1;
    intermediate_1.data = workspace + 0;  // 复用！
    intermediate_1.shape = (int64_t[]){1, 64, 112, 112};
    intermediate_1.ndim = 4;
    intermediate_1.dtype = (DLDataType){kDLFloat, 32, 1};

    // 执行
    tvmgen_default_conv2d(arg0, arg1, &intermediate_0, workspace);
    tvmgen_default_relu(&intermediate_0, &intermediate_1, workspace);
    tvmgen_default_pool2d(&intermediate_1, output, workspace);

    return 0;
}
```

### 26.13.3 AOT 与 PackedFunc 的互操作

AOT 生成的代码可以与 TVM 的 PackedFunc 系统互操作：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```c
// AOT Packed 接口模式
int tvmgen_default_run(
    TVMValue* args,    // PackedFunc 参数数组
    int* type_codes,   // 类型码数组
    int num_args       // 参数数量
) {
    // 解包参数
    DLTensor* x = (DLTensor*)args[0].v_handle;
    DLTensor* w = (DLTensor*)args[1].v_handle;
    DLTensor* output = (DLTensor*)args[2].v_handle;

    // 执行计算
    // ...

    return 0;
}

// 注册为 PackedFunc
TVM_REGISTER_GLOBAL("tvmgen_default___tvm_main__")
.set_body(tvmgen_default_run);
```

---

## 26.14 执行器性能分析

### 26.14.1 开销分析

三种执行器的开销来源不同：

| 开销类型 | Graph Executor | Relay VM | AOT |
|---------|---------------|----------|-----|
| **指令分发** | 每节点一次函数调用 | 每指令一次 switch | 零（直接执行） |
| **内存分配** | 运行时分配 | 运行时分配 | 编译时确定 |
| **类型检查** | 运行时检查 | 运行时检查 | 编译时检查 |
| **函数调用** | PackedFunc 开销 | PackedFunc + VM 开销 | 内联/直接调用 |
| **缓存效率** | 中 | 中 | 高 |

### 26.14.2 延迟分解



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
推理延迟分解（典型 CNN 模型）：

Graph Executor:
  指令分发: 5% | 算子计算: 85% | 内存分配: 5% | 其他: 5%

Relay VM:
  指令分发: 8% | 算子计算: 82% | 内存分配: 5% | 其他: 5%

AOT Executor:
  指令分发: 0% | 算子计算: 92% | 内存分配: 0% | 其他: 8%
```

### 26.14.3 内存占用对比



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
内存占用分解（ResNet-50 模型）：

Graph Executor:
  模型参数: 100MB | 中间张量: 50MB | 运行时开销: 2MB

Relay VM:
  模型参数: 100MB | 中间张量: 50MB | 字节码: 100KB | 运行时开销: 1MB

AOT Executor:
  模型参数: 100MB | 中间张量: 30MB (优化后) | 代码: 500KB | 运行时开销: 100KB
```

---

## 26.15 执行器调试与诊断

### 26.15.1 Graph Executor 调试



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 使用 DebugExecutor 进行调试
from tvm.contrib import graph_executor

# 创建调试模式的执行器
module = graph_executor.GraphModule(lib["default"](dev))

# 设置输入
module.set_input("x", input_data)

# 单步执行
module.run()

# 检查中间结果
for i in range(module.get_num_outputs()):
    output = module.get_output(i)
    print(f"Output {i}: shape={output.shape}, "
          f"min={output.numpy().min():.4f}, "
          f"max={output.numpy().max():.4f}")
```

### 26.15.2 VM 调试



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# VM 逐步调试
from tvm.runtime.vm import VirtualMachine

vm = VirtualMachine(exec, dev)

# 启用详细日志
vm.set_debug(True)

# 执行并查看每条指令
result = vm.invoke("main", input_data)

# 查看字节码执行统计
stats = vm.get_exec_stats()
print(f"总指令数: {stats['total_instructions']}")
print(f"总执行时间: {stats['total_time_ms']:.2f} ms")
```

### 26.15.3 AOT 调试



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```c
// AOT 生成的 C 代码可以直接用 gdb 调试
// 在编译时添加调试信息：
//   gcc -g -O0 -DTVM_CRT_DEBUG tvmgen_default.c

// 在代码中添加断点
int tvmgen_default_run(...) {
    // 调试：打印输入统计
    #ifdef TVM_CRT_DEBUG
    printf("Input: shape=[%ld, %ld, %ld, %ld], "
           "min=%f, max=%f\n",
           x->shape[0], x->shape[1], x->shape[2], x->shape[3],
           min_value(x), max_value(x));
    #endif

    tvmgen_default_conv2d(...);

    // 调试：检查中间结果
    #ifdef TVM_CRT_DEBUG
    printf("After conv2d: min=%f, max=%f\n",
           min_value(&intermediate), max_value(&intermediate));
    #endif

    return 0;
}
```

---

## 26.16 未来演进：从 Relay 到 Relax

### 26.16.1 Relax VM 的设计改进

TVM Unity 引入的 Relax IR 对执行器设计进行了多项改进：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# Relax VM 的核心改进

# 1. 统一的函数调用语义
# Relay: 函数调用通过 PackedFunc，存在类型擦除开销
# Relax: 函数调用直接映射到编译后的函数

# 2. 更好的动态形状支持
# Relay: 动态形状需要运行时推断
# Relax: 编译时确定形状约束，运行时仅传递值

# 3. 结构化控制流
# Relay: If/While 等控制流需要特殊处理
# Relax: 原生支持结构化控制流

# 4. 更高效的内存管理
# Relay: 依赖运行时内存规划
# Relax: 编译时确定内存布局，支持就地更新
```

### 26.16.2 Relax VM 编译示例



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import relax

# Relax 模型定义
with relax.Builder() as builder:
    with relax.dataflow():
        # 数据流区域
        lv0 = relax.op.nn.conv2d(x, w)
        lv1 = relax.op.nn.relu(lv0)
        builder.emit_output(lv1)

    # 构建函数
    func = builder.get_func()

# Relax VM 编译
ex = relax.vm.build(mod, target="llvm")

# 执行
vm = relax.VirtualMachine(ex, dev)
result = vm["main"](input_data)
```

### 26.16.3 Relax 与 Relay 的执行器对比

| 特性 | Relay VM | Relax VM |
|------|----------|----------|
| **IR 格式** | Relay Expr | Relax Expr + StructInfo |
| **类型检查** | 运行时 | 编译时 |
| **形状推断** | 运行时 | 编译时 |
| **内存规划** | 运行时 | 编译时 |
| **动态形状** | 有限支持 | 原生支持 |
| **控制流** | If/While | 结构化 |
| **性能** | 中 | 高 |

---

## 26.17 总结与最佳实践

### 26.17.1 执行器选择决策树



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
是否需要动态控制流？
├── 是 → Relay VM / Relax VM
│        └── 是否需要最小延迟？
│            ├── 是 → Relax VM
│            └── 否 → Relay VM
└── 否 → 是否是嵌入式设备？
    ├── 是 → AOT Executor
    └── 否 → 是否需要调试？
        ├── 是 → Relay VM（调试模式）
        └── 否 → Graph Executor / AOT
```

### 26.17.2 性能优化建议



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 最佳实践：根据场景选择最优配置

# 场景 1：云服务推理（高吞吐）
config = {
    "executor": "graph",       # 或 "aot"
    "target": "cuda",
    "opt_level": 3,
    "auto_scheduler": True,
}

# 场景 2：边缘设备推理（低延迟）
config = {
    "executor": "aot",
    "target": "llvm -mcpu=cortex-a72",
    "opt_level": 3,
    "quantization": "int8",
}

# 场景 3：嵌入式 MCU 推理（最小资源）
config = {
    "executor": "aot",
    "target": "c",
    "runtime": "crt",
    "quantization": "int8",
    "workspace_alignment": 8,
}

# 场景 4：研究与调试
config = {
    "executor": "vm",
    "target": "llvm",
    "debug": True,
    "profiling": True,
}
```

### 26.17.3 常见问题排查

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| VM 执行速度慢 | 字节码解释开销 | 切换到 AOT 或 Graph Executor |
| AOT 编译失败 | 不支持的算子 | 使用 Graph Executor 或添加自定义算子 |
| 内存不足 | 中间张量过大 | 启用内存规划优化，减小 batch size |
| 动态形状错误 | Graph Executor 不支持 | 使用 VM 或 Relax |
| 调试信息不足 | 编译时优化移除了信息 | 使用 debug 模式编译 |

---

## 26.18 执行器扩展机制

### 26.18.1 自定义执行器

TVM 支持通过扩展机制添加自定义执行器：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 自定义执行器的注册
from tvm.relay.backend import Executor

# 注册自定义执行器
@Executor.register("custom_executor")
class CustomExecutor:
    """自定义执行器示例"""

    def __init__(self, mod, target, params):
        self.mod = mod
        self.target = target
        self.params = params

    def build(self):
        """编译模型"""
        # 自定义编译逻辑
        pass

    def execute(self, inputs):
        """执行推理"""
        # 自定义执行逻辑
        pass

# 使用自定义执行器
lib = relay.build(
    mod, target="llvm", params=params,
    executor=Executor("custom_executor", {"option": "value"})
)
```

### 26.18.2 执行器插件系统



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 执行器插件接口
class ExecutorPlugin:
    """执行器插件基类"""

    def pre_compile(self, mod):
        """编译前处理"""
        return mod

    def post_compile(self, lib):
        """编译后处理"""
        return lib

    def pre_execute(self, inputs):
        """执行前处理"""
        return inputs

    def post_execute(self, outputs):
        """执行后处理"""
        return outputs

# 注册插件
class ProfilingPlugin(ExecutorPlugin):
    """性能分析插件"""

    def __init__(self):
        self.timings = []

    def pre_execute(self, inputs):
        self.start_time = time.time()
        return inputs

    def post_execute(self, outputs):
        elapsed = time.time() - self.start_time
        self.timings.append(elapsed)
        return outputs
```

### 26.18.3 执行器回调机制



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 执行器回调
class ExecutorCallback:
    """执行器回调接口"""

    def on_node_start(self, node_id, node_name):
        """节点开始执行时调用"""
        pass

    def on_node_end(self, node_id, node_name, output):
        """节点执行完成时调用"""
        pass

    def on_execution_start(self):
        """整个执行开始时调用"""
        pass

    def on_execution_end(self, result):
        """整个执行完成时调用"""
        pass

# 使用回调进行调试
class DebugCallback(ExecutorCallback):
    """调试回调"""

    def on_node_start(self, node_id, node_name):
        print(f"开始执行节点 {node_id}: {node_name}")

    def on_node_end(self, node_id, node_name, output):
        print(f"完成节点 {node_id}: {node_name}, "
              f"输出范围 [{output.min():.4f}, {output.max():.4f}]")
```

---

## 26.19 执行器性能调优

### 26.19.1 算子级调优



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def optimize_operator_execution(mod, target):
    """算子级别的执行优化"""

    # 1. 算子融合
    mod = relay.transform.FuseOps(fuse_opt_level=3)(mod)

    # 2. 常量折叠
    mod = relay.transform.FoldConstant()(mod)

    # 3. 布局变换
    mod = relay.transform.AlterOpLayout()(mod)

    # 4. 自动调度
    if target.kind.name == "llvm":
        # 使用 MetaSchedule 自动调优
        database = meta_schedule.tune_tasks(
            tasks=extract_tasks(mod),
            target=target,
        )
        mod = meta_schedule.compile_database(mod, database)

    return mod
```

### 26.19.2 内存级调优



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def optimize_memory_execution(mod, target):
    """内存级别的执行优化"""

    # 1. 内存规划优化
    mod = relay.transform.PlanAndUpdateMemory()(mod)

    # 2. 存储重写
    mod = relay.transform.StorageRewrite()(mod)

    # 3. 内存对齐
    mod = relay.transform.MemoryAlignment(alignment=8)(mod)

    return mod
```

### 26.19.3 图级调优



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def optimize_graph_execution(mod, target):
    """图级别的执行优化"""

    # 1. 死代码消除
    mod = relay.transform.DeadCodeElimination()(mod)

    # 2. 公共子表达式消除
    mod = relay.transform.CommonSubexprElimTir()(mod)

    # 3. 部分求值
    mod = relay.transform.PartialEvaluate()(mod)

    # 4. 算子调度优化
    mod = relay.transform.DefuseOps()(mod)
    mod = relay.transform.FuseOps(fuse_opt_level=3)(mod)

    return mod
```

---

## 26.20 执行器测试与验证

### 26.20.1 执行器正确性测试



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def test_executor_correctness(mod, params, test_data):
    """测试执行器的正确性"""

    # 使用三种执行器执行相同的输入
    results = {}

    for executor_type in ["graph", "vm", "aot"]:
        try:
            lib = compile_with_executor(mod, params, executor_type)
            result = execute_model(lib, test_data)
            results[executor_type] = result
        except Exception as e:
            print(f"{executor_type} 执行失败: {e}")

    # 比较结果
    if "graph" in results and "vm" in results:
        diff = np.abs(results["graph"] - results["vm"])
        print(f"Graph vs VM 最大差异: {diff.max():.6f}")

    if "graph" in results and "aot" in results:
        diff = np.abs(results["graph"] - results["aot"])
        print(f"Graph vs AOT 最大差异: {diff.max():.6f}")
```

### 26.20.2 执行器性能测试



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def benchmark_executor_performance(mod, params, target, num_runs=100):
    """基准测试执行器性能"""

    results = {}

    for executor_type in ["graph", "vm", "aot"]:
        # 编译
        start = time.time()
        lib = compile_with_executor(mod, params, executor_type)
        compile_time = time.time() - start

        # 预热
        for _ in range(10):
            execute_model(lib, test_data)

        # 基准测试
        start = time.time()
        for _ in range(num_runs):
            execute_model(lib, test_data)
        avg_latency = (time.time() - start) / num_runs

        results[executor_type] = {
            "compile_time": compile_time,
            "avg_latency": avg_latency,
            "throughput": 1.0 / avg_latency,
        }

    # 打印对比
    print("=== 执行器性能对比 ===")
    for exe, perf in results.items():
        print(f"{exe}: 编译={perf['compile_time']:.2f}s, "
              f"延迟={perf['avg_latency']*1000:.2f}ms, "
              f"吞吐={perf['throughput']:.1f} samples/s")
```

### 26.20.3 执行器压力测试



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def stress_test_executor(mod, params, target, duration=60):
    """执行器压力测试"""

    start_time = time.time()
    iteration = 0
    errors = 0

    while time.time() - start_time < duration:
        try:
            # 随机输入
            input_data = np.random.randn(1, 3, 224, 224).astype("float32")

            # 执行推理
            result = execute_model(lib, input_data)

            # 检查结果有效性
            if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                errors += 1
                print(f"迭代 {iteration}: 检测到 NaN/Inf")

            iteration += 1

        except Exception as e:
            errors += 1
            print(f"迭代 {iteration}: 错误 - {e}")

    print(f"压力测试完成: {iteration} 次迭代, {errors} 次错误")
    print(f"错误率: {errors/iteration*100:.2f}%")
```

---

## 26.21 VM 指令集详解

### 26.21.1 Relay VM 指令概述

Relay 虚拟机基于寄存器式指令集，每条指令包含操作码、目标寄存器和操作数寄存器。指令集的设计目标是：

1. **表达力**：覆盖 Relay IR 中所有算子
2. **效率**：最小化指令数量，减少解释开销
3. **可扩展性**：支持自定义算子指令



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
┌─────────────────────────────────────────────────────┐
│              Relay VM 指令格式                        │
├──────────┬──────────┬────────────────────────────────┤
│  Opcode  │  Dest    │  Operands (变长)               │
│  (8 bit) │  (16 bit)│  (寄存器索引列表)               │
└──────────┴──────────┴────────────────────────────────┘

指令内存布局:
  Byte 0:     操作码 (Opcode)
  Byte 1-2:   目标寄存器索引 (16-bit, little-endian)
  Byte 3-4:   操作数数量 N (16-bit)
  Byte 5-6:   操作数 0 的寄存器索引
  Byte 7-8:   操作数 1 的寄存器索引
  ...
  Byte 2N+3..2N+4: 操作数 N-1 的寄存器索引
```

### 26.21.2 核心指令列表

| 指令名 | 操作码 | 操作数 | 说明 |
|--------|--------|--------|------|
| `Move` | 0x01 | src_reg | 寄存器间数据移动 |
| `LoadConst` | 0x02 | const_idx | 从常量池加载 |
| `Invoke` | 0x03 | func_idx, args[] | 调用函数 |
| `InvokePacked` | 0x04 | packed_idx, arity, args[] | 调用 packed 函数 |
| `If` | 0x05 | cond_reg, true_offset, false_offset | 条件分支 |
| `Goto` | 0x06 | offset | 无条件跳转 |
| `AllocTensor` | 0x07 | shape[], dtype, storage | 分配张量 |
| `AllocTensorReg` | 0x08 | shape_reg, dtype, storage | 动态形状分配 |
| `AllocStorage` | 0x09 | size, alignment, dtype_hint | 分配存储 |
| `ShapeOf` | 0x0A | tensor_reg | 获取张量形状 |
| `ReshapeTensor` | 0x0B | tensor_reg, new_shape | 重塑张量 |
| `DeviceCopy` | 0x0C | src_reg, src_dev, dst_dev | 设备间拷贝 |
| `Fatal` | 0x0D | (none) | 终止执行 |
| `Ret` | 0x0E | result_reg | 返回结果 |

### 26.21.3 指令执行引擎源码

指令执行引擎位于 `src/runtime/vm/vm.cc`，核心是 `VirtualMachine::Execute()` 函数中的 `switch` 分发：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// 文件: src/runtime/vm/vm.cc
// 核心执行循环（简化版）

void VirtualMachine::Execute(const VMFunc& func,
                              const std::vector<ObjectRef>& args) {
  // 初始化 PC（程序计数器）
  PC = 0;

  while (true) {
    // 取指：从当前 PC 位置读取指令
    const Instruction& insn = func.instructions[PC];

    switch (insn.op) {
      case Opcode::Move: {
        // Move 指令：将源寄存器的值复制到目标寄存器
        // 开销极低，仅是指针赋值 + 引用计数
        WriteRegister(insn.dst, ReadRegister(insn.src));
        PC++;  // 推进程序计数器
        break;
      }

      case Opcode::LoadConst: {
        // LoadConst：从常量池中加载预编译的常量张量
        // 常量池在编译阶段构建，包含所有模型权重和常量
        const auto& const_val = const_pool_[insn.const_index];
        WriteRegister(insn.dst, const_val);
        PC++;
        break;
      }

      case Opcode::InvokePacked: {
        // InvokePacked：调用已编译的 packed 函数（算子）
        // packed_func_idx 指向注册在 PackedFunc 注册表中的函数
        // arity 是函数参数个数
        // 参数从寄存器文件中读取
        PackedFunc func = packed_funcs_[insn.packed_index];

        // 收集参数：将寄存器中的值转为 TVM Args
        TVMRetValue rv;
        std::vector<TVMValue> values;
        std::vector<int> type_codes;
        for (int i = 0; i < insn.arity; i++) {
          auto arg = ReadRegister(insn.args[i]);
          // 将 ObjectRef 转换为 TVMValue
          TVMArgValue val;
          PackTo(&val, arg);
          values.push_back(val.value);
          type_codes.push_back(val.type_code);
        }

        // 调用 packed 函数
        func.CallPacked(TVMArgs(values.data(), type_codes.data(), arity),
                        &rv);
        // 存储返回值
        if (rv.type_code() != kTVMNullptr) {
          WriteRegister(insn.dst, rv);
        }
        PC++;
        break;
      }

      case Opcode::If: {
        // If：条件分支指令
        // 读取条件寄存器（通常是标量布尔值）
        bool cond = ReadRegister(insn.cond).operator bool();
        if (cond) {
          PC += insn.true_offset;   // 跳转到 true 分支
        } else {
          PC += insn.false_offset;  // 跳转到 false 分支
        }
        break;
      }

      case Opcode::Goto: {
        // Goto：无条件跳转
        PC += insn.offset;
        break;
      }

      case Opcode::AllocTensor: {
        // AllocTensor：分配新的张量
        // 形状在编译时已确定（静态形状场景）
        auto storage = ReadRegister(insn.storage);
        DLTensor* tensor = static_cast<DLTensor*>(storage.operator->());
        // 设置形状
        std::vector<int64_t> shape(insn.shape, insn.shape + insn.ndim);
        WriteRegister(insn.dst, TVMNDArray::FromNDArray(tensor, shape));
        PC++;
        break;
      }

      case Opcode::Ret: {
        // Ret：返回指令，将结果写入返回寄存器
        return_register_ = ReadRegister(insn.result);
        return;  // 退出执行循环
      }

      case Opcode::Fatal: {
        // Fatal：致命错误，终止执行
        LOG(FATAL) << "VM encountered fatal instruction";
        break;
      }

      default:
        LOG(FATAL) << "Unknown opcode: " << static_cast<int>(insn.op);
    }
  }
}
```

### 26.21.4 指令编码效率分析



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
指令编码空间分析：

┌──────────────────────────────────────────────────────────┐
│                  指令编码效率                              │
├──────────────────┬───────────┬───────────┬───────────────┤
│ 指令类型         │ 字节数    │ 操作数范围 │ 频率           │
├──────────────────┼───────────┼───────────┼───────────────┤
│ Move             │ 5         │ 65536     │ 高（寄存器分配）│
│ LoadConst        │ 5         │ 65536     │ 中（模型加载） │
│ InvokePacked     │ 5+2N      │ 无限      │ 最高（算子调用）│
│ If               │ 9         │ ±32768    │ 低（控制流）   │
│ Goto             │ 5         │ ±32768    │ 低（跳转）     │
│ AllocTensor      │ 5+4*ndim  │ 65536     │ 中（内存分配） │
│ Ret              │ 5         │ 65536     │ 每函数一次     │
└──────────────────┴───────────┴───────────┴───────────────┘

编译时优化:
├── 常量折叠：LoadConst + InvokePacked → 合并为单个 LoadConst
├── 死代码消除：未使用的 dst 寄存器 → 删除指令
├── 寄存器分配：最小化 Move 指令数量
└── 指令重排：提高 cache 命中率
```

---

## 26.22 字节码格式

### 26.22.1 字节码文件结构

Relay VM 编译输出的字节码采用自定义二进制格式，文件扩展名为 `.ro`（Relay Object）。整体结构如下：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
┌─────────────────────────────────────────┐
│           Relay Bytecode File           │
│          (.ro 文件格式)                  │
├─────────────────────────────────────────┤
│  Magic Number (4 bytes): 0x524C5942     │  ← "RLYB"
│  Version (4 bytes): 0x00000001          │
├─────────────────────────────────────────┤
│  Header Section                         │
│  ├── 全局变量数量 (4 bytes)             │
│  ├── 函数表偏移 (8 bytes)               │
│  ├── 常量池偏移 (8 bytes)               │
│  └── 入口函数索引 (4 bytes)             │
├─────────────────────────────────────────┤
│  Function Table                         │
│  ├── 函数 0: RegisterFile 大小          │
│  │   ├── 参数数量                       │
│  │   ├── 指令数量                       │
│  │   └── 指令序列                       │
│  ├── 函数 1: ...                        │
│  └── 函数 N: ...                        │
├─────────────────────────────────────────┤
│  Constant Pool                          │
│  ├── 常量 0: NDArray (DLTensor)         │
│  │   ├── 数据类型 (DLDataType)          │
│  │   ├── 形状 (int64_t[])              │
│  │   ├── 数据对齐                       │
│  │   └── 原始数据 (raw bytes)           │
│  ├── 常量 1: ...                        │
│  └── 常量 M: ...                        │
├─────────────────────────────────────────┤
│  Global Section                         │
│  ├── 全局变量初始值                      │
│  └── 外部函数绑定表                      │
├─────────────────────────────────────────┤
│  Symbol Table (可选)                    │
│  └── 函数名 → 索引映射                  │
└─────────────────────────────────────────┘
```

### 26.22.2 字节码序列化源码



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// 文件: src/runtime/vm/serialize.cc
// 字节码序列化核心逻辑

class BytecodeSerializer {
public:
  // 序列化整个 VM 模块到二进制流
  void Serialize(const VMModule& mod, dmlc::Stream* strm) {
    // 1. 写入 magic number
    uint32_t magic = kTVMVMBytecodeMagic;
    strm->Write(magic);

    // 2. 写入版本号
    uint32_t version = kTVMVMBytecodeVersion;
    strm->Write(version);

    // 3. 写入函数数量
    uint64_t num_funcs = mod.functions_.size();
    strm->Write(num_funcs);

    // 4. 逐个序列化函数
    for (const auto& func : mod.functions_) {
      SerializeFunction(func, strm);
    }

    // 5. 写入常量池
    SerializeConstantPool(mod.const_pool_, strm);

    // 6. 写入全局变量
    SerializeGlobals(mod.globals_, strm);
  }

private:
  // 序列化单个函数
  void SerializeFunction(const VMFunc& func, dmlc::Stream* strm) {
    // 函数名
    strm->Write(func.name);

    // 寄存器文件大小
    strm->Write(func.register_file_size);

    // 参数数量
    uint64_t num_params = func.params.size();
    strm->Write(num_params);

    // 指令数量
    uint64_t num_insns = func.instructions.size();
    strm->Write(num_insns);

    // 序列化每条指令
    for (const auto& insn : func.instructions) {
      SerializeInstruction(insn, strm);
    }
  }

  // 序列化单条指令
  void SerializeInstruction(const Instruction& insn, dmlc::Stream* strm) {
    // 操作码 (1 byte)
    uint8_t op = static_cast<uint8_t>(insn.op);
    strm->Write(op);

    // 目标寄存器 (2 bytes)
    uint16_t dst = static_cast<uint16_t>(insn.dst);
    strm->Write(dst);

    // 根据操作码写入不同字段
    switch (insn.op) {
      case Opcode::Move:
        strm->Write(static_cast<uint16_t>(insn.src));
        break;

      case Opcode::LoadConst:
        strm->Write(static_cast<uint32_t>(insn.const_index));
        break;

      case Opcode::InvokePacked:
        strm->Write(static_cast<uint32_t>(insn.packed_index));
        strm->Write(static_cast<uint32_t>(insn.arity));
        for (int i = 0; i < insn.arity; i++) {
          strm->Write(static_cast<uint16_t>(insn.args[i]));
        }
        break;

      case Opcode::If:
        strm->Write(static_cast<uint16_t>(insn.cond));
        strm->Write(static_cast<int32_t>(insn.true_offset));
        strm->Write(static_cast<int32_t>(insn.false_offset));
        break;

      case Opcode::Goto:
        strm->Write(static_cast<int32_t>(insn.offset));
        break;

      // ... 其他指令的序列化
    }
  }

  // 序列化常量池
  void SerializeConstantPool(const std::vector<ObjectRef>& pool,
                              dmlc::Stream* strm) {
    uint64_t num_consts = pool.size();
    strm->Write(num_consts);

    for (const auto& const_val : pool) {
      if (const_val->IsInstance<NDArray::Container>()) {
        // NDArray 常量
        auto ndarray = Downcast<NDArray>(const_val);
        strm->Write(static_cast<uint8_t>(0));  // 类型标记

        // 写入 DLTensor 元数据
        DLTensor* dl = ndarray.operator->();
        strm->Write(dl->ndim);
        for (int i = 0; i < dl->ndim; i++) {
          strm->Write(dl->shape[i]);
        }
        strm->Write(dl->dtype.bits);
        strm->Write(dl->dtype.lanes);

        // 写入原始数据
        int64_t num_elems = 1;
        for (int i = 0; i < dl->ndim; i++) num_elems *= dl->shape[i];
        int64_t data_bytes = num_elems * (dl->dtype.bits / 8);
        strm->Write(static_cast<const char*>(dl->data), data_bytes);
      }
    }
  }
};
```

### 26.22.3 字节码反序列化与加载



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// 文件: src/runtime/vm/serialize.cc
// 字节码反序列化

class BytecodeDeserializer {
public:
  VMModule Deserialize(dmlc::Stream* strm) {
    VMModule mod;

    // 1. 验证 magic number
    uint32_t magic;
    strm->Read(&magic);
    CHECK_EQ(magic, kTVMVMBytecodeMagic) << "Invalid bytecode file";

    // 2. 读取版本号并检查兼容性
    uint32_t version;
    strm->Read(&version);
    CHECK_LE(version, kTVMVMBytecodeVersion) << "Unsupported bytecode version";

    // 3. 读取函数表
    uint64_t num_funcs;
    strm->Read(&num_funcs);
    for (uint64_t i = 0; i < num_funcs; i++) {
      mod.functions_.push_back(DeserializeFunction(strm));
    }

    // 4. 读取常量池
    mod.const_pool_ = DeserializeConstantPool(strm);

    // 5. 读取全局变量
    mod.globals_ = DeserializeGlobals(strm);

    return mod;
  }
};

// 加载字节码并创建可执行模块
// 文件: src/runtime/vm/vm.cc
VMModule VMModule::LoadFromBinary(void* strm) {
  auto* stream = static_cast<dmlc::Stream*>(strm);
  BytecodeDeserializer deserializer;
  VMModule mod = deserializer.Deserialize(stream);
  // 绑定 packed functions
  mod.BindPackedFunctions();
  return mod;
}
```

### 26.22.4 字节码调试工具



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def disassemble_bytecode(ro_file_path):
    """反汇编 Relay VM 字节码文件，用于调试"""

    import struct

    with open(ro_file_path, 'rb') as f:
        # 读取 header
        magic = struct.unpack('<I', f.read(4))[0]
        assert magic == 0x524C5942, f"Invalid magic: {hex(magic)}"
        version = struct.unpack('<I', f.read(4))[0]
        print(f"Bytecode Version: {version}")

        # 读取函数数量
        num_funcs = struct.unpack('<Q', f.read(8))[0]
        print(f"Functions: {num_funcs}")

        # 操作码名称映射
        opcode_names = {
            0x01: "Move", 0x02: "LoadConst", 0x03: "Invoke",
            0x04: "InvokePacked", 0x05: "If", 0x06: "Goto",
            0x07: "AllocTensor", 0x08: "AllocTensorReg",
            0x09: "AllocStorage", 0x0A: "ShapeOf",
            0x0B: "ReshapeTensor", 0x0C: "DeviceCopy",
            0x0D: "Fatal", 0x0E: "Ret",
        }

        # 逐函数反汇编
        for func_idx in range(num_funcs):
            # 函数名（长度前缀字符串）
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')
            print(f"\n=== Function {func_idx}: {name} ===")

            # 寄存器文件大小
            reg_size = struct.unpack('<I', f.read(4))[0]
            print(f"  Registers: {reg_size}")

            # 参数数量
            num_params = struct.unpack('<Q', f.read(8))[0]
            print(f"  Parameters: {num_params}")

            # 指令数量
            num_insns = struct.unpack('<Q', f.read(8))[0]
            print(f"  Instructions: {num_insns}")

            # 反汇编每条指令
            for insn_idx in range(num_insns):
                op = struct.unpack('<B', f.read(1))[0]
                dst = struct.unpack('<H', f.read(2))[0]
                op_name = opcode_names.get(op, f"Unknown(0x{op:02x})")

                # 根据操作码读取操作数
                operands = []
                if op == 0x01:  # Move
                    src = struct.unpack('<H', f.read(2))[0]
                    operands.append(f"r{src}")
                elif op == 0x02:  # LoadConst
                    idx = struct.unpack('<I', f.read(4))[0]
                    operands.append(f"const[{idx}]")
                elif op == 0x04:  # InvokePacked
                    packed_idx = struct.unpack('<I', f.read(4))[0]
                    arity = struct.unpack('<I', f.read(4))[0]
                    args = []
                    for _ in range(arity):
                        args.append(f"r{struct.unpack('<H', f.read(2))[0]}")
                    operands.append(f"packed[{packed_idx}]({', '.join(args)})")
                elif op == 0x0E:  # Ret
                    result = struct.unpack('<H', f.read(2))[0]
                    operands.append(f"r{result}")

                # 格式化输出
                args_str = ", ".join(operands)
                print(f"  [{insn_idx:4d}] {op_name:16s} r{dst}, {args_str}")
```

---

## 26.23 AOT 代码生成流程

### 26.23.1 AOT 执行器概述

AOT（Ahead-of-Time）执行器将 Relay 模型直接编译为目标平台的 C 代码，消除运行时解释开销。与 Graph Executor 和 VM Executor 相比：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
┌─────────────────────────────────────────────────────────────────┐
│                    执行器对比                                     │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│ 特性          │ Graph Exec   │ VM Exec      │ AOT Exec           │
├──────────────┼──────────────┼──────────────┼────────────────────┤
│ 执行方式      │ 预编译算子    │ 字节码解释   │ 原生 C 代码         │
│ 运行时开销    │ 中           │ 高           │ 极低                │
│ 控制流支持    │ 有限         │ 完整         │ 完整                │
│ 动态形状      │ 不支持       │ 支持         │ 有限支持            │
│ 代码大小      │ 中           │ 大           │ 小                  │
│ 适用场景      │ 服务端推理    │ 研究/调试    │ 嵌入式/边缘设备     │
│ 编译时间      │ 短           │ 中           │ 长                  │
│ 内存占用      │ 运行时动态    │ 运行时动态   │ 编译时静态          │
└──────────────┴──────────────┴──────────────┴────────────────────┘
```

### 26.23.2 AOT 代码生成器架构



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
AOT 代码生成流程:

Relay IR Module
    │
    ▼
┌─────────────────┐
│  Pass Pipeline  │  ← 优化 pass: 算子融合、常量折叠等
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LowerToVM()    │  ← 先降低到 VM IR（可选路径）
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│          AOT Code Generator             │
│  ┌─────────────────────────────────┐    │
│  │  1. 遍历 Relay 函数              │    │
│  │     ├── 收集输入/输出张量信息     │    │
│  │     ├── 分配内存计划              │    │
│  │     └── 确定算子调用顺序          │    │
│  ├─────────────────────────────────┤    │
│  │  2. 生成 C 头文件                │    │
│  │     ├── 结构体定义               │    │
│  │     ├── 函数声明                 │    │
│  │     └── 常量数组                 │    │
│  ├─────────────────────────────────┤    │
│  │  3. 生成 C 实现文件              │    │
│  │     ├── 初始化函数               │    │
│  │     ├── 推理主函数               │    │
│  │     ├── 内存管理函数             │    │
│  │     └── 算子调用代码             │    │
│  ├─────────────────────────────────┤    │
│  │  4. 生成 CMakeLists.txt          │    │
│  │     ├── 源文件列表               │    │
│  │     ├── 编译选项                 │    │
│  │     └── 链接库                   │    │
│  └─────────────────────────────────┘    │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  输出文件:       │
│  ├── model.h     │  ← 接口头文件
│  ├── model.c     │  ← 推理实现
│  ├── model_lib.c │  ← 算子库实现
│  └── CMakeLists  │  ← 构建脚本
└─────────────────┘
```

### 26.23.3 AOT 代码生成源码



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// 文件: src/relay/backend/aot_executor_codegen.cc
// AOT 代码生成器核心实现

class AOTCodeGenerator : public ExprFunctor<void(const Expr&)> {
public:
  // 主入口：将 Relay 模块编译为 C 代码
  std::string Generate(const IRModule& mod, const String& target) {
    // 1. 运行优化 pass
    auto pass_ctx = relay::transform::PassContext::Create();
    pass_ctx->config.Set(
        "relay.backend.use_aot",
        tvm::Bool(true));
    With<PassContext> ctx_scope(pass_ctx);

    // 2. 遍历模块中的所有函数
    for (const auto& kv : mod->functions) {
      if (auto* func_node = kv.second.as<FunctionNode>()) {
        auto func = GetRef<Function>(func_node);
        VisitExpr(func->body);
      }
    }

    // 3. 生成 C 代码
    std::ostringstream header, impl;
    EmitHeader(header);
    EmitMemoryPlan(impl);
    EmitOperators(impl);
    EmitMainFunction(impl);

    return header.str() + "\n" + impl.str();
  }

protected:
  // 访问函数调用节点
  void VisitExpr_(const CallNode* call) final {
    // 获取算子名称
    auto op_node = call->op.as<OpNode>();
    std::string op_name = op_node->name;

    // 收集输入张量
    std::vector<std::string> input_names;
    for (const auto& arg : call->args) {
      input_names.push_back(GetTensorName(arg));
    }

    // 生成输出张量名
    std::string output_name = "tensor_" + std::to_string(tensor_counter_++);

    // 记录算子调用
    operators_.push_back({op_name, input_names, output_name});
  }

  // 访问常量节点
  void VisitExpr_(const ConstantNode* const_node) final {
    // 将常量数据嵌入到生成的代码中
    std::string name = "const_" + std::to_string(const_counter_++);
    constants_[name] = SerializeConstant(const_node->data);
  }

private:
  // 发射内存规划代码
  void EmitMemoryPlan(std::ostringstream& os) {
    os << "// 内存池定义\n";
    os << "static uint8_t workspace[" << workspace_size_ << "];\n\n";

    os << "// 张量存储分配\n";
    for (const auto& alloc : memory_plan_) {
      os << "static DLTensor " << alloc.name << " = {\n";
      os << "  .data = workspace + " << alloc.offset << ",\n";
      os << "  .ndim = " << alloc.ndim << ",\n";
      os << "  .dtype = {" << alloc.dtype_code << ", "
         << alloc.dtype_bits << ", 1},\n";
      os << "  .shape = " << alloc.shape_expr << ",\n";
      os << "  .strides = NULL,\n";
      os << "  .byte_offset = 0\n";
      os << "};\n";
    }
  }

  // 发射算子调用代码
  void EmitOperators(std::ostringstream& os) {
    os << "\n// 推理主函数\n";
    os << "int TVMFunc(DLTensor* inputs[], DLTensor* outputs[]) {\n";

    // 将输入拷贝到内部存储
    for (int i = 0; i < num_inputs_; i++) {
      os << "  memcpy(input_" << i << ".data, inputs[" << i
         << "]->data, input_" << i << "_size);\n";
    }

    // 按顺序调用每个算子
    for (const auto& op : operators_) {
      os << "\n  // " << op.name << "\n";
      os << "  " << op.name << "(";
      for (size_t i = 0; i < op.inputs.size(); i++) {
        os << "&" << op.inputs[i];
        if (i < op.inputs.size() - 1) os << ", ";
      }
      os << ", &" << op.output << ");\n";
    }

    // 将输出拷贝到输出张量
    for (int i = 0; i < num_outputs_; i++) {
      os << "\n  memcpy(outputs[" << i << "]->data, output_" << i
         << ".data, output_" << i << "_size);\n";
    }

    os << "  return 0;\n";
    os << "}\n";
  }

  // 数据结构
  struct OperatorCall {
    std::string name;
    std::vector<std::string> inputs;
    std::string output;
  };

  struct MemoryAllocation {
    std::string name;
    size_t offset;
    int ndim;
    int dtype_code;
    int dtype_bits;
    std::string shape_expr;
  };

  int tensor_counter_ = 0;
  int const_counter_ = 0;
  size_t workspace_size_ = 0;
  int num_inputs_ = 0;
  int num_outputs_ = 0;
  std::vector<OperatorCall> operators_;
  std::vector<MemoryAllocation> memory_plan_;
  std::unordered_map<std::string, std::string> constants_;
};
```

### 26.23.4 AOT 生成的 C 代码示例



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```c
// 生成的 model.h 文件示例
#ifndef MODEL_H_
#define MODEL_H_

#include <stdint.h>
#include <dlpack/dlpack.h>

#ifdef __cplusplus
extern "C" {
#endif

// 模型输入输出数量
#define MODEL_NUM_INPUTS  1
#define MODEL_NUM_OUTPUTS 1

// 输入张量形状
static const int64_t input_0_shape[] = {1, 3, 224, 224};

// 输出张量形状
static const int64_t output_0_shape[] = {1, 1000};

// 工作空间大小（字节）
#define MODEL_WORKSPACE_SIZE (12582912)  // 12MB

// 初始化函数：加载权重、分配内存
int Model_Init(void);

// 推理函数：执行前向计算
int Model_Run(DLTensor* inputs[], DLTensor* outputs[]);

// 释放函数：释放资源
int Model_Release(void);

#ifdef __cplusplus
}
#endif
#endif  // MODEL_H_
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```c
// 生成的 model.c 文件示例（简化）
#include "model.h"
#include <string.h>

// 内存池（静态分配，避免 malloc 开销）
static uint8_t workspace[MODEL_WORKSPACE_SIZE];

// 张量定义（指向内存池中的特定偏移）
static DLTensor conv1_output = {
  .data = workspace + 0,
  .ndim = 4,
  .dtype = {kDLFloat, 32, 1},
  .shape = (int64_t[]){1, 64, 112, 112},
  .strides = NULL,
  .byte_offset = 0
};

static DLTensor bn1_output = {
  .data = workspace + 3211264,
  .ndim = 4,
  .dtype = {kDLFloat, 32, 1},
  .shape = (int64_t[]){1, 64, 112, 112},
  .strides = NULL,
  .byte_offset = 0
};

// ... 更多张量定义 ...

// 常量权重（编译时嵌入）
static const float conv1_weight[] = { /* ... */ };
static const float bn1_gamma[] = { /* ... */ };

int Model_Init(void) {
  // 绑定权重到张量
  // 此处为简化的初始化逻辑
  return 0;
}

int Model_Run(DLTensor* inputs[], DLTensor* outputs[]) {
  // 拷贝输入数据
  memcpy(workspace + 1505280,  // input_0 偏移
         inputs[0]->data,
         1505280);  // 1*3*224*224*sizeof(float)

  // 逐层推理
  // Conv2d: conv1
  tvmgen_default_conv1(
    (float*)(workspace + 1505280),   // input
    (float*)(conv1_weight),           // weight
    (float*)(workspace + 0)           // output
  );

  // BatchNorm: bn1
  tvmgen_default_bn1(
    (float*)(workspace + 0),          // input
    (float*)(bn1_gamma),              // gamma
    (float*)(bn1_beta),               // beta
    (float*)(bn1_mean),               // mean
    (float*)(bn1_var),                // variance
    (float*)(workspace + 3211264)     // output
  );

  // ReLU: relu1
  tvmgen_default_relu1(
    (float*)(workspace + 3211264),    // input
    (float*)(workspace + 3211264)     // output (in-place)
  );

  // ... 更多层 ...

  // 拷贝输出
  memcpy(outputs[0]->data,
         workspace + 12578816,  // output_0 偏移
         4000);  // 1*1000*sizeof(float)

  return 0;
}

int Model_Release(void) {
  // 静态分配，无需释放
  return 0;
}
```

### 26.23.5 AOT 内存规划算法



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def plan_memory_aot(relay_mod):
    """AOT 内存规划：计算每个张量的存储偏移

    算法：基于生命周期分析的内存复用
    - 分析每个张量的定义点和最后使用点
    - 如果两个张量的生命周期不重叠，可以复用同一块内存
    - 目标：最小化总内存使用量
    """

    # 1. 收集所有张量及其生命周期
    tensor_lifetimes = {}
    for i, op_call in enumerate(get_operator_sequence(relay_mod)):
        # 输出张量的生命周期：从定义点到最后使用点
        output = op_call.output
        define_point = i
        last_use = find_last_use(relay_mod, output, start=i)
        tensor_lifetimes[output] = (define_point, last_use)

    # 2. 贪心内存分配
    # 按定义点排序
    sorted_tensors = sorted(tensor_lifetimes.items(),
                            key=lambda x: x[1][0])

    memory_blocks = []  # (start_offset, end_offset, tensor_name)
    allocations = {}

    for tensor_name, (define, last_use) in sorted_tensors:
        tensor_size = compute_tensor_size(relay_mod, tensor_name)

        # 查找可复用的内存块
        allocated = False
        for block in memory_blocks:
            block_start, block_end, block_tensor = block
            block_last_use = tensor_lifetimes[block_tensor][1]
            # 如果旧张量已不再使用，可以复用
            if block_last_use < define:
                allocations[tensor_name] = block_start
                block[2] = tensor_name  # 更新占用者
                allocated = True
                break

        if not allocated:
            # 分配新内存块
            offset = sum(b[1] - b[0] for b in memory_blocks)
            allocations[tensor_name] = offset
            memory_blocks.append([offset, offset + tensor_size, tensor_name])

    total_size = sum(b[1] - b[0] for b in memory_blocks)
    print(f"总内存需求: {total_size / 1024 / 1024:.2f} MB")
    print(f"张量数量: {len(allocations)}")

    return allocations, total_size
```

### 26.23.6 AOT 编译流程完整示例



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import tvm
from tvm import relay
import numpy as np

def compile_aot_model():
    """完整的 AOT 编译流程示例"""

    # 1. 定义模型（以简单的两层网络为例）
    data = relay.var("data", shape=(1, 3, 224, 224))
    weight1 = relay.var("weight1", shape=(64, 3, 7, 7))

    # Conv2d + ReLU
    conv1 = relay.nn.conv2d(data, weight1, strides=(2, 2),
                             padding=(3, 3))
    relu1 = relay.nn.relu(conv1)

    # Global Average Pooling
    gap = relay.nn.global_avg_pool2d(relu1)
    flat = relay.nn.batch_flatten(gap)

    # Fully Connected
    weight2 = relay.var("weight2", shape=(1000, 64))
    dense = relay.nn.dense(flat, weight2)

    # Softmax
    output = relay.nn.softmax(dense)

    # 创建函数
    func = relay.Function(
        [data, weight1, weight2], output
    )
    mod = tvm.IRModule.from_expr(func)

    # 2. 设置目标平台
    target = tvm.target.Target("c")  # 纯 C 代码生成

    # 3. 配置 AOT 编译
    executor = relay.backend.Executor(
        "aot",
        {
            "interface-api": "packed",    # 使用 packed 接口
            "unpacked-api": "c",          # C 风格 unpacked 接口
            "workspace-byte-alignment": 8, # 内存对齐
            "constant-byte-alignment": 8,
        }
    )

    # 4. 编译
    with tvm.transform.PassContext(
        opt_level=3,
        config={"relay.backend.executor": executor}
    ):
        lib = relay.build(mod, target=target, params={})

    # 5. 导出 C 代码
    lib.export_library(
        "model_aot.tar",
        fcompile=tvm.contrib.cc.create_csource_compiler_lib
    )

    print("AOT 编译完成，生成文件: model_aot.tar")

    # 6. 查看生成的代码
    # 解压后可以看到 model.h, model.c 等文件
    import tarfile
    with tarfile.open("model_aot.tar", "r") as tar:
        for member in tar.getmembers():
            print(f"  {member.name} ({member.size} bytes)")

    return lib
```

---

## 26.24 执行器内存管理机制

### 26.24.1 内存池设计



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
┌─────────────────────────────────────────────────────────────┐
│                   执行器内存池架构                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 Storage Pool Layer                   │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐             │   │
│  │  │ CPU Pool │ │ GPU Pool │ │ DSP Pool │ ...          │   │
│  │  │ (Host)   │ │ (Device) │ │ (Accel)  │             │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘             │   │
│  │       │             │             │                   │   │
│  │       ▼             ▼             ▼                   │   │
│  │  ┌──────────────────────────────────────────────┐   │   │
│  │  │           Memory Allocator                    │   │   │
│  │  │  ┌─────────┐ ┌──────────┐ ┌──────────────┐  │   │   │
│  │  │  │Best Fit │ │Pool Alloc│ │Arena Alloc   │  │   │   │
│  │  │  └─────────┘ └──────────┘ └──────────────┘  │   │   │
│  │  └──────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 NDArray Cache                        │   │
│  │  常量张量缓存（权重、偏置等）                          │   │
│  │  跨推理共享，避免重复加载                              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Workspace Buffer                        │   │
│  │  临时工作空间（中间结果、转置缓冲区等）                 │   │
│  │  推理结束后释放                                       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 26.24.2 内存分配策略对比



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
class MemoryAllocatorBenchmark:
    """内存分配策略性能对比"""

    def benchmark_strategies(self, allocations):
        """对比不同分配策略的效果"""

        strategies = {
            "naive": self.naive_allocate,
            "best_fit": self.best_fit_allocate,
            "pool": self.pool_allocate,
            "arena": self.arena_allocate,
        }

        results = {}
        for name, allocator in strategies.items():
            peak_mem, alloc_time = allocator(allocations.copy())
            results[name] = {
                "peak_memory_mb": peak_mem / 1024 / 1024,
                "alloc_time_ms": alloc_time * 1000,
            }

        # 打印对比表
        print("策略          | 峰值内存(MB) | 分配时间(ms)")
        print("-" * 50)
        for name, r in results.items():
            print(f"{name:14s}| {r['peak_memory_mb']:12.2f} | "
                  f"{r['alloc_time_ms']:12.3f}")

        return results

    def naive_allocate(self, allocs):
        """朴素分配：每个张量独立分配"""
        total = 0
        start = time.time()
        for name, size in allocs:
            total += size  # 不复用，直接累加
        return total, time.time() - start

    def best_fit_allocate(self, allocs):
        """最佳适配：找到最小的可用空闲块"""
        free_blocks = []  # (offset, size)
        allocated = {}
        peak = 0
        start = time.time()

        for name, size in allocs:
            # 查找最小可用块
            best = None
            for i, (off, sz) in enumerate(free_blocks):
                if sz >= size:
                    if best is None or sz < free_blocks[best][1]:
                        best = i

            if best is not None:
                off, sz = free_blocks.pop(best)
                allocated[name] = off
                if sz > size:
                    free_blocks.append((off + size, sz - size))
            else:
                offset = peak
                allocated[name] = offset
                peak += size

            peak = max(peak, allocated[name] + size)

        return peak, time.time() - start

    def pool_allocate(self, allocs):
        """池分配：预分配固定大小的块"""
        block_size = 1024 * 1024  # 1MB blocks
        pool = []
        start = time.time()
        peak = 0

        for name, size in allocs:
            blocks_needed = (size + block_size - 1) // block_size
            peak = max(peak, len(pool) * block_size + blocks_needed * block_size)
            for _ in range(blocks_needed):
                pool.append(bytearray(block_size))

        return peak, time.time() - start

    def arena_allocate(self, allocs):
        """竞技场分配：线性分配，批量释放"""
        arena_size = 0
        start = time.time()

        for name, size in allocs:
            # 简单线性推进（不考虑复用）
            arena_size += size

        return arena_size, time.time() - start
```


**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。

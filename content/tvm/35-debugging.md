> **学习目标**：
> - 掌握 TVM 编译流程中各阶段的常见错误类型及其诊断方法
> - 学会使用 Relay 可视化工具、TIR dump、Pass 调试等手段定位问题
> - 了解 TVM_LOG_DEBUG 等环境变量在调试中的作用
> - 建立性能问题排查的系统化 checklist
> - 掌握 RPC 远程调试的方法与常见问题
> - 能够独立诊断和解决常见的 RuntimeError

---

## 35.1 调试概述

### 35.1.1 TVM 编译管线回顾

TVM 的编译流程涉及多个阶段，每个阶段都可能引入错误。理解错误发生的阶段是快速定位问题的关键：

```
前端导入 → Relay IR → Pass 优化 → TE Lowering → TIR → CodeGen → Runtime
  (1)        (2)        (3)         (4)        (5)     (6)       (7)
```

| 阶段 | 常见错误类型 |
|------|-------------|
| (1) 前端导入 | 算子不支持、shape 推断失败、dtype 不匹配 |
| (2) Relay IR | 类型错误、shape mismatch、未注册算子 |
| (3) Pass 优化 | 优化 pass 引入的 bug、无限循环、内存爆炸 |
| (4) TE Lowering | 调度错误、非法循环变换、bound 推断失败 |
| (5) TIR | Buffer 访问越界、类型不匹配、未定义变量 |
| (6) CodeGen | 目标不支持的指令、LLVM 版本不兼容 |
| (7) Runtime | 段错误、内存溢出、数值错误 |

### 35.1.2 调试策略总览



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 通用调试流程
import tvm
from tvm import relay

# Step 1: 缩小问题范围
# 逐阶段检查中间结果

# Step 2: 启用详细日志
import os
os.environ["TVM_LOG_DEBUG"] = "default"

# Step 3: 降低优化级别
with tvm.transform.PassContext(opt_level=0):
    # 测试是否是某个 pass 引入的问题
    pass

# Step 4: 检查中间 IR
print(relay.print(mod))  # 打印 Relay IR
print(tvm.lower(s, inputs, simple_mode=True))  # 打印 TIR
```

### 35.1.3 错误分类

TVM 中的错误可以分为三大类：

1. **编译时错误**（Compile-time Errors）：在 `tvm.build()` 或 `relay.build()` 过程中抛出
2. **运行时错误**（Runtime Errors）：在执行编译好的函数时发生
3. **逻辑错误**（Logic Errors）：程序能运行但结果不正确

<div data-component="TVMErrorTaxonomy"></div>

---

## 35.2 常见编译错误类型

### 35.2.1 Shape Mismatch 错误

Shape mismatch 是最常见的错误之一，通常发生在算子的输入张量形状不兼容时。

**典型错误信息**：

```
TVMError: Check failed: lhs->shape.size() == rhs->shape.size() (4 vs 2)
: Incompatible broadcast type [1, 64, 56, 56] vs [64, 64]
```

**常见原因与修复**：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 原因 1：张量维度不匹配
A = relay.var("A", shape=(1, 64, 56, 56))
B = relay.var("B", shape=(64, 64))
# 错误：A 是 4D，B 是 2D，无法直接相加
C = relay.add(A, B)  # RuntimeError!

# 修复：使用 reshape 或 broadcast_to
B_reshaped = relay.reshape(B, (1, 64, 1, 1))
C = relay.add(A, B_reshaped)  # 正确：广播到 (1, 64, 56, 56)
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 原因 2：卷积 kernel 尺寸与输入不匹配
# 输入通道数必须匹配
data = relay.var("data", shape=(1, 3, 224, 224))
weight = relay.var("weight", shape=(64, 16, 7, 7))  # 错误：16 ≠ 3
conv = relay.nn.conv2d(data, weight)  # RuntimeError!

# 修复
weight = relay.var("weight", shape=(64, 3, 7, 7))  # 正确：输入通道 = 3
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 原因 3：matmul 的内维度不匹配
A = relay.var("A", shape=(128, 256))
B = relay.var("B", shape=(512, 64))
# matmul 要求 A 的列数 = B 的行数
C = relay.nn.matmul(A, B)  # RuntimeError: 256 ≠ 512

# 修复
B = relay.var("B", shape=(256, 64))
C = relay.nn.matmul(A, B)  # 正确
```

### 35.2.2 Dtype 错误

数据类型不匹配也是常见错误，特别是在混合精度计算时。

**典型错误信息**：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
TVMError: Check failed: lhs.dtype() == rhs.dtype() (float32 vs int32)
: Cannot add expressions of type float32 and int32
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 原因 1：混合 dtype 运算
A = relay.var("A", shape=(10,), dtype="float32")
B = relay.var("B", shape=(10,), dtype="int32")
C = relay.add(A, B)  # RuntimeError!

# 修复：显式类型转换
B_float = relay.cast(B, "float32")
C = relay.add(A, B_float)
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 原因 2：量化模型中的 dtype 问题
# INT8 量化需要仔细处理 scale 和 zero_point
A = relay.var("A", shape=(1, 64, 56, 56), dtype="int8")
B = relay.var("B", shape=(1, 64, 56, 56), dtype="uint8")
C = relay.add(A, B)  # 错误：int8 和 uint8 不能直接相加

# 修复：统一为同一 dtype
B_int8 = relay.cast(B, "int8")
C = relay.add(A, B_int8)
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 原因 3：TE 中的 dtype 不匹配
A = te.placeholder((128,), name="A", dtype="float32")
B = te.placeholder((128,), name="B", dtype="float64")
C = te.compute((128,), lambda i: A[i] + B[i], name="C")
# 在某些情况下可能报错

# 修复：统一 dtype
B = te.placeholder((128,), name="B", dtype="float32")
```

### 35.2.3 Unsupported Op 错误

当导入外部框架模型时，可能遇到 TVM 不支持的算子。

**典型错误信息**：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
TVMError: Operator my_custom_op is not registered in Relay
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 前端导入时的不支持算子
import onnx
model = onnx.load("model.onnx")
# 假设模型包含一个 TVM 不支持的自定义算子

mod, params = relay.frontend.from_onnx(model, shape_dict)
# 可能报错：Unsupported operator

# 修复方案 1：注册自定义算子（见第 34 章）
# 修复方案 2：更新 TVM 版本（新版本可能已支持）
# 修复方案 3：在导出 ONNX 前替换为支持的算子组合
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 识别不支持的算子
try:
    mod, params = relay.frontend.from_onnx(model, shape_dict)
except Exception as e:
    print(f"导入失败: {e}")
    # 检查模型中的算子列表
    for node in model.graph.node:
        print(f"  {node.op_type}: {node.name}")
```

### 35.2.4 Type Inference 失败

Relay 的类型推断系统可能在遇到类型不一致的表达式时报错。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 类型推断失败示例
A = relay.var("A", relay.TensorType((1, 3, 224, 224), "float32"))
# 尝试对 scalar 和 tensor 运算
scalar = relay.const(2.0)
B = relay.multiply(A, scalar)  # 通常可以自动广播

# 但某些情况下推断可能失败
# 当 Relay 无法推断出完整的类型时：
# TVMError: Unable to unify TensorType([1, 3, 224, 224], float32) 
#           and TensorType([], float32)

# 修复：显式指定类型或使用 relay.expand_dims
scalar_tensor = relay.reshape(scalar, (1,))
scalar_broadcast = relay.broadcast_to(scalar_tensor, A.checked_type.shape)
B = relay.multiply(A, scalar_broadcast)
```

### 35.2.5 Operator Not Registered 错误



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 当使用 Relay 的低级 API 时
# 如果算子未注册，会报错

# 检查算子是否已注册
from tvm import relay
print(relay.op.op.get("nn.conv2d"))     # 正常返回
# print(relay.op.op.get("my_op"))       # 报错：not registered

# 查看所有已注册的算子
# 可以通过 tvm.relay.op 注册表查询
```

<div data-component="ErrorTypeExplorer"></div>

---

## 35.3 Relay 可视化工具

### 35.3.1 使用 `tvm.relay.visualization`

TVM 提供了 Relay IR 的可视化工具，可以将计算图渲染为图形表示。

```python
import tvm
from tvm import relay
from tvm.relay import visualization

# 定义一个简单的模型
data = relay.var("data", shape=(1, 3, 224, 224))
weight = relay.var("weight", shape=(64, 3, 7, 7))
conv = relay.nn.conv2d(data, weight, strides=(2, 2), padding=(3, 3))
bn = relay.nn.batch_norm(conv, relay.var("gamma"), relay.var("beta"),
                          relay.var("mean"), relay.var("var"))
relu = relay.nn.relu(bn[0])
func = relay.Function(relay.analysis.free_vars(relu), relu)
mod = tvm.IRModule.from_expr(func)

# 可视化
dot = visualization.plot(mod)

# 渲染为不同的格式
dot.render("model_graph", format="svg")   # SVG
dot.render("model_graph", format="pdf")   # PDF
dot.render("model_graph", format="png")   # PNG

# 在 Jupyter Notebook 中直接显示
# dot  # 直接在 cell 中输出
```

### 35.3.2 可视化输出解读

可视化的计算图中：
- **矩形节点**：表示张量操作（conv2d、relu 等）
- **椭圆节点**：表示变量（输入、权重）
- **边**：表示数据流向
- **标签**：显示 shape 和 dtype 信息



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 查看优化前后的图对比
mod_opt = tvm.transform.Sequential([
    relay.transform.FuseOps(fuse_opt_level=2),
    relay.transform.InferType(),
])(mod)

# 优化前
dot_before = visualization.plot(mod)
dot_before.render("before_fusion", format="svg")

# 优化后（算子融合后节点减少）
dot_after = visualization.plot(mod_opt)
dot_after.render("after_fusion", format="svg")
```

### 35.3.3 自定义可视化



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 可以自定义节点样式
from tvm.relay.visualization import DOTGraph

# 获取底层的 DOT 图对象进行自定义
dot = visualization.plot(mod, show_metadata=True)

# 显示每个节点的详细元数据（shape、dtype、attributes）
# show_metadata=True 会在节点标签中显示额外信息
```

### 35.3.4 常见可视化调试场景

| 场景 | 可视化目的 | 关注点 |
|------|-----------|--------|
| 算子融合调试 | 检查哪些算子被融合 | 节点是否合并 |
| Shape 传播 | 验证 shape 是否正确 | 边上的 shape 标签 |
| 数据流分析 | 追踪数据依赖 | 连接关系 |
| 内存分析 | 识别大的中间张量 | 张量大小标签 |
| 量化调试 | 检查量化/反量化节点 | dtype 标签 |



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 调试算子融合
mod_fused = relay.transform.FuseOps(fuse_opt_level=3)(mod)
print("融合前算子数:", len(relay.analysis.post_order_visit(mod["main"], lambda x: [x])))
print("融合后算子数:", len(relay.analysis.post_order_visit(mod_fused["main"], lambda x: [x])))

# 可视化融合后的图，检查融合边界
dot_fused = visualization.plot(mod_fused)
dot_fused.render("fused_graph", format="svg")
```

<div data-component="RelayVisualizationDemo"></div>

---

## 35.4 TIR dump：tvm.lower() 输出

### 35.4.1 使用 tvm.lower()

`tvm.lower()` 是查看调度变换结果的核心工具。它将 Schedule 和输入参数转换为 TIR（TVM Intermediate Representation），这是代码生成的直接输入。

```python
import tvm
from tvm import te

# 定义计算
A = te.placeholder((128, 128), name="A")
B = te.placeholder((128, 128), name="B")
C = te.compute((128, 128), lambda i, j: A[i, j] + B[i, j], name="C")

# 创建调度
s = te.create_schedule(C.op)
i, j = s[C].op.axis
i0, i1 = s[C].split(i, factor=32)
s[C].reorder(i0, j, i1)
s[C].vectorize(i1)
s[C].parallel(i0)

# 查看 TIR
print("=== simple_mode=True ===")
print(tvm.lower(s, [A, B, C], simple_mode=True))
```

输出示例：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# from tvm.script import tir as T
@T.prim_func
def main(A: T.Buffer[(128, 128), "float32"],
         B: T.Buffer[(128, 128), "float32"],
         C: T.Buffer[(128, 128), "float32"]):
    for i0 in T.parallel(4):
        for j in T.serial(128):
            for i1 in T.vectorized(32):
                C[i0 * 32 + i1, j] = A[i0 * 32 + i1, j] + B[i0 * 32 + i1, j]
```

### 35.4.2 simple_mode 与详细模式



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# simple_mode=True：简洁输出，适合快速检查调度变换
print(tvm.lower(s, [A, B, C], simple_mode=True))

# simple_mode=False：详细输出，包含 buffer 信息、bound 等
print(tvm.lower(s, [A, B, C], simple_mode=False))
```

详细模式输出：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
@T.prim_func
def main(A: T.Buffer[(128, 128), "float32"],
         B: T.Buffer[(128, 128), "float32"],
         C: T.Buffer[(128, 128), "float32"]):
    # buffer 定义
    A_data = T.var("float32")
    B_data = T.var("float32")
    C_data = T.var("float32")
    
    # buffer alias
    A_1 = T.buffer_decl([16384], dtype="float32", data=A_data)
    B_1 = T.buffer_decl([16384], dtype="float32", data=B_data)
    C_1 = T.buffer_decl([16384], dtype="float32", data=C_data)
    
    # 循环结构
    for i0 in T.serial(4, annotations={"pragma_parallel": T.int32(4)}):
        for j in T.serial(128):
            for i1 in T.serial(32, annotations={"pragma_vectorize": T.int32(32)}):
                with T.block("C"):
                    vi = T.axis.spatial(128, i0 * 32 + i1)
                    vj = T.axis.spatial(128, j)
                    C_1[vi * 128 + vj] = A_1[vi * 128 + vj] + B_1[vi * 128 + vj]
```

### 35.4.3 阅读 TIR 输出的技巧

**1. 识别循环类型**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# T.serial → 普通串行循环
# T.parallel → 已并行化的循环
# T.vectorized → 已向量化的循环
# T.unrolled → 已展开的循环
for i in T.serial(100):       # 普通循环
    for j in T.parallel(4):   # 并行循环
        for k in T.vectorized(8):  # 向量循环
            ...
```

**2. 识别 Block 结构**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# T.block 定义了一个计算块，是 TIR 的核心概念
with T.block("C"):
    # 块的迭代域声明
    vi = T.axis.spatial(128, ...)  # 空间轴：vi ∈ [0, 128)
    vj = T.axis.spatial(128, ...)
    # 块的计算体
    C[vi, vj] = A[vi, vj] + B[vi, vj]
```

**3. 检查 Buffer 访问**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 检查 buffer 访问是否正确
# 例如：是否有越界、对齐是否正确
with T.block("compute"):
    vi = T.axis.spatial(M, ...)
    vj = T.axis.spatial(N, ...)
    # 检查 A 的访问模式
    T.reads(A[vi * 2:vi * 2 + 2, vj])  # 带范围的读
    T.writes(C[vi, vj])
    C[vi, vj] = A[vi * 2, vj] + A[vi * 2 + 1, vj]
```

### 35.4.4 通过 TIR 调试调度错误



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 常见调度错误 1：向量化不兼容
s = te.create_schedule(C.op)
i, j = s[C].op.axis
s[C].vectorize(i)  # 如果 i 的范围不是常量，会报错

# 通过 TIR 检查
try:
    tir = tvm.lower(s, [A, B, C], simple_mode=True)
    print(tir)
except tvm.TVMError as e:
    print(f"向量化失败: {e}")
    # 回退：先 split 再 vectorize
    i0, i1 = s[C].split(i, factor=8)
    s[C].vectorize(i1)  # i1 的范围是常量 8
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 常见调度错误 2：parallel 与 reduce 冲突
k = te.reduce_axis((0, K), name="k")
D = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k))
s = te.create_schedule(D.op)
i, j = s[D].op.axis

# 错误地对归约轴外层并行化（如果有依赖的话）
# 通过 TIR 检查归约结构
print(tvm.lower(s, [A, B, D], simple_mode=True))
```

### 35.4.5 TIR 与 Python 调度的对应关系



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# Python 调度代码          →  TIR 输出
s[C].split(i, factor=32)  →  for i0 in T.serial(...): for i1 in T.serial(32):
s[C].reorder(j, i)        →  for j in T.serial(...): for i in T.serial(...):
s[C].vectorize(j)         →  for j in T.vectorized(...):
s[C].parallel(i)          →  for i in T.parallel(...):
s[C].unroll(k)            →  for k in T.unrolled(...):
s[C].bind(i, blockIdx.x)  →  for i in T.thread_binding(..., "blockIdx.x")
```

<div data-component="TIRInspectionTool"></div>

---

## 35.5 Pass 调试

### 35.5.1 PassContext 配置

`tvm.transform.PassContext` 控制 Relay/TIR pass 的执行行为。通过配置 PassContext 可以调试 pass 引入的问题。

```python
import tvm
from tvm import relay
from tvm.transform import PassContext

# 基本配置
with PassContext(opt_level=3) as ctx:
    # 默认优化级别，执行所有优化
    lib = relay.build(mod, target="llvm", params=params)
```

### 35.5.2 降低优化级别



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# opt_level=0：不执行任何优化 pass
# 用于判断问题是否由某个优化 pass 引入
with PassContext(opt_level=0):
    lib = relay.build(mod, target="llvm", params=params)
    # 如果 opt_level=0 能成功但 opt_level=3 失败
    # 说明是某个优化 pass 引入的问题
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 逐级排查
for level in range(4):
    try:
        with PassContext(opt_level=level):
            lib = relay.build(mod, target="llvm", params=params)
        print(f"opt_level={level}: 成功")
    except Exception as e:
        print(f"opt_level={level}: 失败 - {e}")
        break
```

### 35.5.3 启用/禁用特定 Pass



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# required_pass：必须执行的 pass
# disabled_pass：禁用的 pass

with PassContext(
    opt_level=3,
    required_pass=["InferType"],        # 必须执行类型推断
    disabled_pass=["FuseOps"]           # 禁用算子融合
):
    lib = relay.build(mod, target="llvm", params=params)
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 通过禁用特定 pass 来定位问题
# 逐个禁用 pass，找到引入问题的 pass
from tvm.relay import transform

all_passes = [
    "InferType",
    "FoldConstant",
    "FuseOps",
    "CombineParallelConv2D",
    "FoldScaleAxis",
    "CanonicalizeOps",
    "AlterOpLayout",
]

for pass_name in all_passes:
    try:
        with PassContext(opt_level=3, disabled_pass=[pass_name]):
            lib = relay.build(mod, target="llvm", params=params)
        print(f"禁用 {pass_name}: 成功")
    except Exception as e:
        print(f"禁用 {pass_name}: 仍然失败")
```

### 35.5.4 Pass 追踪（Trace）



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# PassContext 的 trace 参数用于追踪 pass 的执行
# 可以记录每个 pass 的输入输出

with PassContext(opt_level=3) as ctx:
    # 执行构建
    lib = relay.build(mod, target="llvm", params=params)
    # 查看执行了哪些 pass
    print(ctx._get_trace())
```

### 35.5.5 PrintIR Pass

`PrintIR` 是一个特殊的 pass，用于在 pass 管线中打印 IR 状态：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm.relay import transform

# 在特定位置插入 PrintIR 来查看中间状态
seq = tvm.transform.Sequential([
    transform.InferType(),
    transform.PrintIR(),  # 打印类型推断后的 IR
    transform.FuseOps(fuse_opt_level=2),
    transform.PrintIR(),  # 打印融合后的 IR
    transform.DefuseOps(),
    transform.PrintIR(),  # 打印反融合后的 IR
])

with PassContext(opt_level=0):
    mod_opt = seq(mod)
```

### 35.5.6 自定义 Pass 用于调试



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 创建一个自定义 pass 来检查 IR 状态
@tvm.transform.function_pass(opt_level=0)
def MyDebugPass(func, mod, ctx):
    """打印函数的详细信息"""
    print(f"Function name: {func.attrs.global_var.name_hint}")
    print(f"Parameters: {[p.name_hint for p in func.params]}")
    print(f"Body type: {func.body.checked_type}")
    
    # 遍历所有调用节点
    class CallCounter(tvm.relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.call_count = 0
            
        def visit_call(self, call):
            self.call_count += 1
            print(f"  Call: {call.op}")
            return super().visit_call(call)
    
    counter = CallCounter()
    counter.visit(func.body)
    print(f"Total calls: {counter.call_count}")
    
    return func

# 使用自定义 pass
with PassContext(opt_level=0, required_pass=["MyDebugPass"]):
    mod_opt = tvm.transform.Sequential([MyDebugPass()])(mod)
```

<div data-component="PassDebugger"></div>

---

## 35.6 TVM_LOG_DEBUG 与其他环境变量

### 35.6.1 TVM_LOG_DEBUG

`TVM_LOG_DEBUG` 是 TVM 最重要的调试环境变量，启用后会输出详细的内部日志。

```bash
# 启用调试日志
export TVM_LOG_DEBUG="default"

# 运行你的 TVM 程序
python my_tvm_script.py
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 也可以在 Python 中设置
import os
os.environ["TVM_LOG_DEBUG"] = "default"

import tvm  # 必须在 import tvm 之前设置
```

**日志输出示例**：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
[14:23:45] src/te/schedule/schedule_lang.cc:123: Split axis i into i.outer and i.inner with factor=32
[14:23:45] src/te/schedule/schedule_lang.cc:200: Reorder axes: [i.outer, j, i.inner]
[14:23:46] src/tir/ir/buffer.cc:145: Buffer A allocated with shape [128, 128]
[14:23:46] src/target/codegen/codegen.cc:89: Generating LLVM IR for function main
```

### 35.6.2 TVM_LOG_LEVEL

控制日志的详细程度：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```bash
# 日志级别：DEBUG, INFO, WARNING, ERROR
export TVM_LOG_LEVEL="DEBUG"     # 最详细
export TVM_LOG_LEVEL="INFO"      # 信息级别
export TVM_LOG_LEVEL="WARNING"   # 只显示警告和错误
export TVM_LOG_LEVEL="ERROR"     # 只显示错误
```

### 35.6.3 CODEGEN_DEBUG

用于调试代码生成阶段：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```bash
# 启用代码生成的调试输出
export CODEGEN_DEBUG=1

# 对于 CUDA 代码生成
export TVM_CUDA_DEBUG=1
# 会输出生成的 CUDA 代码和编译信息
```

### 35.6.4 TVM_TRACE_EXECUTION

启用执行跟踪，记录运行时的每个算子执行：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import tvm
from tvm import runtime

# 启用执行跟踪
runtime.enable_trace_execution()

# 运行推理
out = graph_mod.run()

# 禁用跟踪
runtime.disable_trace_execution()
```

跟踪输出会显示每个算子的执行时间和参数。

### 35.6.5 其他有用的环境变量

| 环境变量 | 作用 | 默认值 |
|----------|------|--------|
| `TVM_LOG_DEBUG` | 启用调试日志 | 未设置 |
| `TVM_LOG_LEVEL` | 日志级别 | INFO |
| `TVM_NUM_THREADS` | TVM 使用的线程数 | CPU 核心数 |
| `TVM_BIND_MAIN_THREAD` | 绑定主线程到大核 | 0 |
| `CUDA_LAUNCH_BLOCKING` | CUDA 同步执行（调试用） | 0 |
| `TVM_USE_MSAN` | 启用内存消毒器 | 0 |
| `TVM_FFI_DEBUG` | 启用 FFI 调试检查 | 0 |
| `TVM_ADDITIONAL_LIBRARY_PATH` | 额外的库搜索路径 | "" |



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```bash
# 典型的调试环境变量组合
export TVM_LOG_DEBUG="default"
export TVM_LOG_LEVEL="DEBUG"
export TVM_NUM_THREADS=1          # 单线程，便于复现
export CUDA_LAUNCH_BLOCKING=1     # CUDA 同步执行
```

### 35.6.6 编译器 Tracing



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 使用 Python 的 trace 机制来追踪 TVM 内部调用
import sys

class TVMTracer:
    def __init__(self):
        self.call_stack = []
        
    def trace_calls(self, frame, event, arg):
        if event == 'call':
            module = frame.f_globals.get('__name__', '')
            if 'tvm' in module:
                func_name = frame.f_code.co_name
                self.call_stack.append(func_name)
                print(f"{'  ' * len(self.call_stack)}→ {module}.{func_name}")
        elif event == 'return':
            module = frame.f_globals.get('__name__', '')
            if 'tvm' in module:
                if self.call_stack:
                    self.call_stack.pop()
        return self.trace_calls

# 使用
tracer = TVMTracer()
sys.settrace(tracer.trace_calls)
# ... 执行 TVM 操作 ...
sys.settrace(None)
```

<div data-component="DebugEnvironmentVariables"></div>

---

## 35.7 性能问题排查 Checklist

### 35.7.1 性能问题排查流程

当模型推理速度不如预期时，按以下流程系统排查：

```
性能不达标
├── 1. 检查计算定义是否正确
│   └── 是否有冗余计算、不必要的内存分配
├── 2. 检查调度是否合理
│   ├── 分块大小是否合适
│   ├── 是否向量化
│   └── 是否并行化
├── 3. 检查内存访问模式
│   ├── 缓存命中率
│   └── 内存带宽利用率
├── 4. 检查代码生成质量
│   └── 查看生成的目标代码
└── 5. 检查运行时配置
    ├── 线程数
    └── 内存分配器
```

### 35.7.2 Checklist 1：计算定义



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 检查 1：是否有不必要的中间张量
# 不好的写法
temp = te.compute((M, N), lambda i, j: A[i, j] * 2, name="temp")
C = te.compute((M, N), lambda i, j: temp[i, j] + B[i, j], name="C")
# temp 会分配额外的内存

# 好的写法：使用 compute_inline 消除中间张量
temp = te.compute((M, N), lambda i, j: A[i, j] * 2, name="temp")
C = te.compute((M, N), lambda i, j: temp[i, j] + B[i, j], name="C")
s = te.create_schedule(C.op)
s[temp].compute_inline()  # 内联 temp，不分配内存
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 检查 2：归约是否有冗余
# 不好的写法：多次遍历同一个数据
sum_val = te.compute((M,), lambda i: te.sum(A[i, k], axis=k), ...)
sq_sum = te.compute((M,), lambda i: te.sum(A[i, k] * A[i, k], axis=k), ...)
# A 被遍历了两次

# 好的写法：合并归约
# 使用 tuple 计算同时输出 sum 和 sq_sum
```

### 35.7.3 Checklist 2：调度质量



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 使用 Profiling 检查每个算子的耗时
import tvm
from tvm import relay

# 构建模型
lib = relay.build(mod, target="llvm", params=params)

# 使用 graph runtime 的 profiling 功能
from tvm.contrib import graph_executor

dev = tvm.cpu(0)
graph_mod = graph_executor.GraphModule(lib["default"](dev))
graph_mod.set_input("data", data)

# 执行 profiling
profile_results = graph_mod.profile()
print(profile_results)
```

**调度 Checklist**：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# ✓ 检查项 1：是否使用了并行化
print(tvm.lower(s, inputs, simple_mode=True))
# 查看是否有 T.parallel

# ✓ 检查项 2：是否使用了向量化
# 查看是否有 T.vectorized

# ✓ 检查项 3：分块大小是否合理
# L1 cache ≈ 32KB → tile_m * tile_n * 4bytes ≈ 32KB
# 对于 float32: tile_m * tile_n ≈ 8192 → tile ≈ 90x90

# ✓ 检查项 4：是否展开了小循环
# 归约循环如果次数少（如 3x3 卷积），应该展开

# ✓ 检查项 5：是否使用了缓存（cache_read/cache_write）
# 检查是否有 shared/local scope 的 buffer
```

### 35.7.4 Checklist 3：内存访问模式



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 检查访存模式的工具
def analyze_memory_access(schedule, inputs):
    """分析调度后的内存访问模式"""
    stmt = tvm.lower(schedule, inputs, simple_mode=False)
    
    # 打印 TIR，手动检查：
    # 1. 内层循环是否连续访问内存
    # 2. 是否有 stride 过大的访问
    # 3. 是否有不必要的随机访问
    print(stmt)

# 矩阵乘法中 B[k, j] 的访问模式
# 如果 j 在最内层：B[k, j] 是行访问（好）
# 如果 k 在最内层：B[k, j] 是列访问（差）
```

### 35.7.5 Checklist 4：代码生成质量



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 查看生成的 LLVM IR
func = tvm.build(s, inputs, target="llvm")
print(func.get_source())

# 查看生成的 CUDA 代码
func_gpu = tvm.build(s, inputs, target="cuda")
print(func_gpu.imported_modules[0].get_source())
```

检查要点：
- 是否有不必要的 load/store
- 循环是否被正确展开
- SIMD 指令是否被正确使用
- GPU：shared memory 是否被正确使用

### 35.7.6 Checklist 5：运行时配置



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 检查线程数配置
import tvm
tvm.cpu(0).max_num_threads  # 最大线程数

# 检查内存分配
# TVM 使用自定义的内存分配器
# 可以通过环境变量调整
# TVM_POOL_ALLOCATOR_MAX_FREE_BYTES: 池分配器最大空闲字节
```

<div data-component="PerformanceChecklist"></div>

---

## 35.8 RPC 远程调试

### 35.8.1 RPC 调试概述

TVM 的 RPC（Remote Procedure Call）机制允许在远程设备（如 ARM 开发板、GPU 服务器）上部署和执行模型。调试远程设备上的问题需要特殊的工具和方法。

```python
# RPC 基本使用
import tvm
from tvm import rpc

# 连接到远程设备
remote = rpc.connect("远程设备IP", 端口号)

# 在远程设备上执行
remote.upload("my_lib.tar")
lib = remote.load_module("my_lib.tar")
func = lib["main"]

# 创建远程上下文
ctx = remote.cpu(0)  # 或 remote.gpu(0)
a_remote = tvm.nd.array(a_np, ctx)
b_remote = tvm.nd.array(b_np, ctx)
c_remote = tvm.nd.array(c_np, ctx)

func(a_remote, b_remote, c_remote)
```

### 35.8.2 设置 RPC Tracker



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```bash
# 在服务器上启动 RPC Tracker
python -m tvm.exec.rpc_server --tracker=0.0.0.0:9190 --key=android

# 在远程设备上启动 RPC Server
python -m tvm.exec.rpc_server --tracker=服务器IP:9190 --key=android
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 通过 Tracker 连接
from tvm import rpc

# 使用 tracker 自动发现设备
remote = rpc.connect_tracker("服务器IP:9190")
device = remote.request("android", timeout=60)
```

### 35.8.3 常见 RPC 错误与修复

**错误 1：连接超时**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
RuntimeError: RPCConnectionError: Failed to connect to remote device
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 可能原因：
# 1. 网络不通
# 2. RPC Server 未启动
# 3. 防火墙阻止

# 诊断步骤：
# Step 1: 检查网络连通性
# ping 远程设备

# Step 2: 检查 RPC Server 是否在运行
# 在远程设备上查看进程

# Step 3: 检查端口是否开放
# telnet 远程设备IP 端口号

# Step 4: 使用更长的超时
remote = rpc.connect("IP", port, timeout=120)
```

**错误 2：设备未找到**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
RuntimeError: Cannot find available device with key: android
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 检查 key 是否匹配
# Tracker 端和 Server 端的 key 必须一致

# 查看可用设备
tracker = rpc.connect_tracker("IP:9190")
print(tracker.text_summary())  # 列出所有已注册的设备
```

**错误 3：远程执行内存错误**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
RuntimeError: Check failed: (offset + elem_bytes * num_elems) <= buf_size
: Buffer access out of bounds
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 可能原因：
# 1. 远程设备内存不足
# 2. 模型太大无法放入设备内存
# 3. Buffer 分配错误

# 诊断：
# Step 1: 检查设备内存
dev = remote.gpu(0)
print(f"GPU 内存: {dev.total_memory / 1024**2:.0f} MB")

# Step 2: 减小模型或 batch size
# Step 3: 使用更节省内存的量化格式
```

### 35.8.4 远程 Profiling



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 在远程设备上进行性能分析
remote = rpc.connect("IP", port)

# 交叉编译到目标架构
target = tvm.target.arm_cpu("rasp3b")
lib = tvm.build(s, inputs, target=target)

# 上传到远程设备
remote.upload(lib.export_library())
func = remote.load_module("lib.tar")

# 在远程设备上测量性能
ctx = remote.cpu(0)
evaluator = func.time_evaluator(func.entry_name, ctx, repeat=10)
result = evaluator(a_remote, b_remote, c_remote)
print(f"远程执行时间: {result.mean * 1000:.3f} ms")
```

### 35.8.5 远程调试技巧



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 技巧 1：在远程设备上打印中间结果
# 使用 RPC 调用在设备上执行自定义代码

# 技巧 2：使用 device_api 查询设备信息
from tvm._ffi import runtime_ctypes

# 查询远程设备信息
dev = remote.gpu(0)
print(f"设备名: {dev.device_name}")
print(f"计算能力: {dev.compute_version}")
print(f"多处理器数: {dev.multi_processor_count}")

# 技巧 3：远程获取日志
# 启用远程设备的调试日志
remote.upload("set_debug.sh")  # 包含 export TVM_LOG_DEBUG=1 的脚本
```

<div data-component="RPCDebuggingDemo"></div>

---

## 35.9 常见 RuntimeError 及解决方案

### 35.9.1 Check Failed 错误

TVM 使用 `ICHECK` 宏进行内部断言。当断言失败时，会抛出包含 `Check failed:` 的错误。

**错误 1：Shape 推断失败**

```
TVMError: Check failed: lhs->shape.size() == rhs->shape.size() (3 vs 4)
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 原因：两个张量的维度数不同
# 修复：检查输入 shape 是否正确
data = relay.var("data", shape=(1, 3, 224))
weight = relay.var("weight", shape=(64, 3, 7, 7))  # 4D vs 3D

# 使用 relay.transform.InferType() 定位错误
from tvm.relay.transform import InferType
mod = InferType()(mod)  # 会在类型推断失败时抛出详细错误
```

**错误 2：Buffer 访问越界**



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
TVMError: Check failed: (offset + elem_bytes * num_elems) <= buf_size
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 原因：生成的代码访问了 buffer 的边界之外
# 可能是调度错误导致的索引计算错误

# 修复：检查 split 和 reorder 后的索引公式
# 使用 tvm.lower() 检查 TIR
tir = tvm.lower(s, inputs, simple_mode=True)
print(tir)
# 手动验证索引是否在合法范围内
```

### 35.9.2 TVMError: 全局函数未找到



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
TVMError: Cannot find the global function "my_module.my_function"
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 原因 1：函数名拼写错误
# 修复：检查函数注册名

# 原因 2：自定义算子未正确注册
# 确保使用了 @tvm.register_func 装饰器
@tvm.register_func("my_module.my_function")
def my_function(x):
    return x * 2

# 原因 3：动态库未加载
# 如果函数定义在 .so 文件中，需要先加载
tvm.runtime.load_module("my_lib.so")
```

### 35.9.3 Shape Inference 错误



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
TVMError: In operator ..., shape inference failed
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 使用 InferType pass 定位问题
from tvm.relay.transform import InferType

try:
    mod_typed = InferType()(mod)
except tvm.TVMError as e:
    print(f"类型推断失败: {e}")
    # 打印当前 IR 查看问题
    print(relay.print(mod))

# 常见原因：
# 1. 动态 shape 未正确处理
# 2. 自定义算子缺少 type relation
# 3. 部分类型信息缺失
```

### 35.9.4 Out of Memory 错误



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
RuntimeError: [target] Out of memory: failed to allocate X bytes
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 原因 1：模型太大
# 修复：减小 batch size 或使用量化

# 原因 2：中间张量太大
# 修复：使用 compute_inline 减少中间张量
s[temp].compute_inline()

# 原因 3：内存碎片
# 修复：使用内存池分配器
# TVM 支持 arena-based 分配器

# 诊断：使用内存分析
import tvm
from tvm.contrib import graph_executor

# 设置内存限制
dev = tvm.gpu(0)
# 检查设备可用内存
```

### 35.9.5 Stack Trace 解读

TVM 的错误堆栈跟踪包含 C++ 和 Python 的混合信息：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
Traceback (most recent call last):
  File "my_script.py", line 42, in <module>
    lib = relay.build(mod, target="llvm")
  File "/path/to/tvm/python/tvm/relay/build_module.py", line 248, in build
    return _build_module.build(mod, target, target_host, params)
  File "tvm/_ffi/_cython/./packed_func.h", line 78, in tvm::runtime::PackedFunc::CallRVP
tvm._ffi.base.TVMError: Traceback (most recent call last):
  [bt] 5 /path/to/tvm/build/libtvm.so(tvm::relay::TypeInferencer::VisitExpr_+0x45) [0x7f...]
  [bt] 4 /path/to/tvm/build/libtvm.so(tvm::relay::InferType+0x123) [0x7f...]
  [bt] 3 /path/to/tvm/build/libtvm.so(tvm::relay::CheckType+0x45) [0x7f...]
  ...
```

**解读技巧**：
1. 最上面的 Python traceback 显示调用入口
2. C++ 的 `[bt]` 行显示 TVM 内部的调用链
3. 函数名（如 `TypeInferencer::VisitExpr_`）指示错误发生的具体阶段
4. 使用 `addr2line` 或 `gdb` 可以获取更精确的位置



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```bash
# 使用 addr2line 解析地址
addr2line -e /path/to/tvm/build/libtvm.so 0x7f... -f

# 使用 gdb 调试
gdb python
(gdb) run my_script.py
# 当崩溃时
(gdb) bt  # 查看完整调用栈
```

<div data-component="RuntimeErrorGuide"></div>

---

## 35.10 调试 AutoTVM 和 Meta-Schedule

### 35.10.1 AutoTVM 调试

AutoTVM 使用搜索算法自动寻找最优调度参数。调试 AutoTVM 问题需要关注搜索空间和代价模型。

```python
import tvm
from tvm import autotvm

# 定义搜索空间
@autotvm.template("my_matmul")
def my_matmul(N, M, K):
    A = te.placeholder((N, K), name="A")
    B = te.placeholder((K, M), name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k))
    
    s = te.create_schedule(C.op)
    
    # 定义搜索空间
    cfg = autotvm.get_config()
    cfg.define_knob("tile_x", [16, 32, 64, 128])
    cfg.define_knob("tile_y", [16, 32, 64, 128])
    
    # 应用配置
    i, j = s[C].op.axis
    k_axis = s[C].op.reduce_axis[0]
    tile_x = cfg["tile_x"].val
    tile_y = cfg["tile_y"].val
    
    i0, i1 = s[C].split(i, factor=tile_x)
    j0, j1 = s[C].split(j, factor=tile_y)
    s[C].reorder(i0, j0, i1, j1)
    s[C].vectorize(j1)
    s[C].parallel(i0)
    
    return s, [A, B, C]
```

**常见 AutoTVM 错误**：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 错误 1：搜索空间过大导致超时
# 修复：减小搜索空间
cfg.define_knob("tile_x", [32, 64])  # 只保留 2 个选项

# 错误 2：某些配置导致编译失败
# 修复：在 template 中添加合法性检查
if tile_x > N:
    tile_x = N  # 确保 tile 大小不超过循环范围

# 错误 3：Tune 过程中的 RuntimeError
# 修复：使用 error handling
@autotvm.template("my_matmul")
def my_matmul(...):
    try:
        # 调度逻辑
        ...
    except Exception as e:
        # 返回一个安全的 fallback 调度
        s = te.create_schedule(C.op)
        return s, [A, B, C]
```

### 35.10.2 Meta-Schedule 调试

Meta-Schedule 是 TVM 的下一代自动调度框架。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import meta_schedule as ms

# 调试搜索空间
space = ms.space.ScheduleSpace("llvm")
# 检查搜索空间的大小
print(f"搜索空间大小: {space.size()}")

# 调试搜索过程
database = ms.tune_tir(
    mod=mod,
    target=target,
    max_trials=100,
    # 启用详细日志
    task_scheduler=ms.task_scheduler.RoundRobin(),
)

# 检查搜索结果
for record in database:
    print(f"运行时间: {record.run_secs:.6f}s")
    print(f"调度:\n{record.trace}")
```

### 35.10.3 Tuning 日志分析



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# AutoTVM 日志文件分析
import autotvm

# 加载日志
with autotvm.apply_history_best("tuning_log.json"):
    with tvm.target.Target("llvm"):
        lib = relay.build(mod, target="llvm", params=params)

# 分析日志内容
with open("tuning_log.json", "r") as f:
    for line in f:
        entry = autotvm.decode(line.encode())
        config = entry.config
        cost = entry.cost
        print(f"配置: {config}, 耗时: {cost:.6f}s")
```

---

## 35.11 内存错误调试

### 35.11.1 内存溢出（Out of Bounds）



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# TVM 使用 bound checking 来检测 buffer 越界
# 在 debug 模式下会启用更严格的检查

# 检查 TIR 中的 buffer 访问
tir = tvm.lower(s, inputs, simple_mode=False)
print(tir)

# 关注 T.block 中的 reads/writes 声明
# 如果声明的范围与实际访问不一致，会导致越界
```

### 35.11.2 Use-After-Free

在 Graph Runtime 中，中间张量的生命周期由 runtime 管理。如果手动管理内存，可能出现 use-after-free。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 错误示例
ctx = tvm.gpu(0)
a = tvm.nd.array(np.zeros((10,), dtype="float32"), ctx)
b = a  # b 和 a 共享同一块内存
del a   # 如果底层内存被释放，b 也会失效
print(b)  # 可能 crash

# 正确做法：确保在使用期间保持引用
a = tvm.nd.array(np.zeros((10,), dtype="float32"), ctx)
b = a
# 使用 b
print(b)
# 用完后再释放
del a, b
```

### 35.11.3 内存泄漏检测



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 使用 Python 的 tracemalloc 检测内存增长
import tracemalloc

tracemalloc.start()

# 执行多次推理
for _ in range(100):
    graph_mod.run()

snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("内存分配 Top 10:")
for stat in top_stats[:10]:
    print(stat)
```

### 35.11.4 TVM_USE_MSAN



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```bash
# 启用 Memory Sanitizer（需要编译时启用 MSAN）
export TVM_USE_MSAN=1

# MSAN 会检测未初始化内存的读取
# 需要 TVM 以 MSAN 支持编译
cmake -DUSE_MSAN=ON ..
```

### 35.11.5 Address Sanitizer



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```bash
# 使用 Address Sanitizer 编译 TVM
cmake -DUSE_ASAN=ON ..

# 或者使用 Python 级别的检测
# 对于 TVM 的 Python binding
python -X faulthandler my_script.py
```

---

## 35.12 交叉编译问题排查

### 35.12.1 Target Triple 不匹配



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 错误：target 与实际设备不匹配
target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu")
# 但实际设备是 armv7l

# 修复：确认设备的 target triple
# 在远程设备上运行：
# uname -m    → aarch64 / armv7l / x86_64
# cat /proc/cpuinfo  → 查看 CPU 信息

# 正确的 target
target = tvm.target.Target("llvm -mtriple=armv7l-linux-gnueabihf")  # ARM 32-bit
target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu")       # ARM 64-bit
target = tvm.target.Target("llvm -mtriple=x86_64-linux-gnu")        # x86 64-bit
```

### 35.12.2 缺失的 Intrinsic



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
LLVM ERROR: Cannot select: 0x...: f32 = fma ...
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 原因：目标 CPU 不支持某些指令
# 修复：指定正确的 CPU 特性

target = tvm.target.Target("llvm -mcpu=cortex-a53 -mattr=+neon")
target = tvm.target.Target("llvm -mcpu=skylake -mattr=+avx2,+fma")
```

### 35.12.3 LLVM 版本问题



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 不同 LLVM 版本的 IR 格式可能不同
# TVM 需要与 LLVM 版本匹配

# 检查 TVM 链接的 LLVM 版本
import tvm
print(tvm.target.codegen.llvm_version_major())

# 常见问题：
# - LLVM 14+ 的 opaque pointer 不兼容 LLVM 13
# - LLVM 15+ 的某些 pass 被移除
# - TVM 的 LLVM backend 需要与编译时使用的 LLVM 版本一致
```

### 35.12.4 CodeGen 后端错误



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# CUDA CodeGen 错误
# "ptxas fatal : Value 'sm_52' is not defined for option 'gpu-name'"

# 修复：指定正确的 GPU 架构
target = tvm.target.Target("cuda -arch=sm_70")   # V100
target = tvm.target.Target("cuda -arch=sm_80")   # A100
target = tvm.target.Target("cuda -arch=sm_86")   # RTX 3090
```



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# OpenCL CodeGen 错误
# "Build program failure"

# 可能原因：
# 1. OpenCL 版本不匹配
# 2. 内核代码语法错误
# 3. 设备不支持的特性

# 诊断：
os.environ["TVM_OPENCL_DUMP_SOURCE"] = "1"
# 会输出生成的 OpenCL 代码，可以手动检查
```

---

## 35.13 实战调试案例

### 35.13.1 案例 1：ONNX 模型导入失败

**问题**：导入一个 ONNX 模型时报错。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import onnx
from tvm import relay

model = onnx.load("model.onnx")
mod, params = relay.frontend.from_onnx(model)
# 报错：Unsupported operator: MyCustomOp
```

**排查步骤**：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# Step 1: 查看模型中的所有算子
for node in model.graph.node:
    print(f"{node.op_type}: {node.name}")

# Step 2: 确认不支持的算子
# 找到 MyCustomOp 的位置

# Step 3: 检查 TVM 是否在新版本中支持
# 查看 TVM 文档或 changelog

# Step 4: 方案选择
# 方案 A：使用 relay.op.Register 注册自定义算子
# 方案 B：在 PyTorch/ONNX 中替换为支持的算子组合
# 方案 C：使用 relay.frontend.from_onnx 的自定义转换器

# Step 5: 如果是 opset 版本问题
mod, params = relay.frontend.from_onnx(model, opset=13)
```

### 35.13.2 案例 2：TVM Lowering 阶段崩溃

**问题**：`tvm.lower()` 报错。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 报错信息
# TVMError: Check failed: (p->value >= min && p->value < max) is false:
# IterVar's value out of range

# 原因：split 的 factor 导致索引越界
s = te.create_schedule(C.op)
i, j = s[C].op.axis
i0, i1 = s[C].split(i, factor=256)  # 但 i 的范围只有 128！

# 修复：确保 factor 不超过轴的范围
i0, i1 = s[C].split(i, factor=min(256, 128))  # 使用 min 确保安全
```

### 35.13.3 案例 3：GPU 调度性能异常

**问题**：GPU 上的矩阵乘法比 CPU 还慢。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 排查步骤

# Step 1: 检查生成的 CUDA 代码
func = tvm.build(s, [A, B, C], target="cuda")
cuda_source = func.imported_modules[0].get_source()
print(cuda_source)

# Step 2: 检查是否正确使用了 shared memory
# 如果没有 shared memory → 数据每次都从全局内存加载 → 慢

# Step 3: 检查 thread block 大小
# 如果 block 太小 → GPU 利用率低
# 如果 block 太大 → 可能超出硬件限制

# Step 4: 检查 grid 大小
# 如果 grid 太小 → SM 利用率低

# Step 5: 使用 NSight 分析
# nsys profile python my_script.py
# 查看 GPU 时间分布
```

### 35.13.4 案例 4：量化精度下降

**问题**：INT8 量化后精度大幅下降。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 排查步骤

# Step 1: 检查量化参数
# 查看 scale 和 zero_point 是否合理
for expr, param in quant_params.items():
    print(f"{expr}: scale={param.scale}, zero_point={param.zero_point}")

# Step 2: 检查量化范围
# 如果 calibration 数据分布不具有代表性
# 会导致量化范围不准确

# Step 3: 检查溢出
# INT8 范围: [-128, 127]
# 如果中间结果超出范围 → 溢出 → 精度损失

# Step 4: 使用混合精度
# 对敏感层使用 INT16 或 float16
# 只对不敏感层使用 INT8
```

<div data-component="DebugCaseStudies"></div>

---

## 35.14 调试 Relay Pass 管线

### 35.13b.1 理解 Pass 管线

Relay 的优化通过一系列 Pass 来完成。每个 Pass 接收一个 IRModule 并返回变换后的 IRModule。理解 Pass 的执行顺序和相互依赖对于调试至关重要。

```python
import tvm
from tvm import relay
from tvm.relay import transform

# 查看默认的优化管线
# relay.build() 内部会创建一个 Pass 序列
# 可以手动构建相同的管线来逐步调试

# 默认优化管线（简化版）
pass_list = [
    transform.InferType(),                    # 类型推断
    transform.FoldConstant(),                 # 常量折叠
    transform.FuseOps(fuse_opt_level=2),      # 算子融合
    transform.CombineParallelConv2D(),        # 合并并行卷积
    transform.FoldScaleAxis(),                # 折叠 scale axis
    transform.CanonicalizeOps(),              # 规范化算子
    transform.AlterOpLayout(),                # 修改算子布局
]

# 逐步执行并检查
current_mod = mod
for i, p in enumerate(pass_list):
    try:
        current_mod = p(current_mod)
        print(f"Pass {i}: {p} - 成功")
    except Exception as e:
        print(f"Pass {i}: {p} - 失败: {e}")
        break
```

### 35.13b.2 Pass 依赖分析

某些 Pass 依赖其他 Pass 的输出。如果前置 Pass 被跳过，后续 Pass 可能失败：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 依赖链示例
# FuseOps 依赖 InferType（需要类型信息来决定融合策略）
# FoldScaleAxis 依赖 InferType
# AlterOpLayout 依赖 InferType

# 如果跳过 InferType，后续 pass 可能失败
try:
    with PassContext(opt_level=3, disabled_pass=["InferType"]):
        lib = relay.build(mod, target="llvm")
except tvm.TVMError as e:
    print(f"禁用 InferType 导致失败: {e}")
    # 修复：确保 InferType 不被禁用
```

### 35.13b.3 自定义调试 Pass



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm.relay import transform
from tvm.relay.base import Node

@transform.function_pass(opt_level=0)
class ShapeChecker:
    """检查所有张量的 shape 是否在预期范围内"""
    
    def __init__(self, max_dims=6, max_size=2**30):
        self.max_dims = max_dims
        self.max_size = max_size
    
    def transform_function(self, func, mod, ctx):
        class Visitor(relay.ExprVisitor):
            def __init__(self, checker):
                super().__init__()
                self.checker = checker
                self.violations = []
            
            def visit_call(self, call):
                # 检查输出 shape
                if hasattr(call, 'checked_type') and call.checked_type:
                    shape = call.checked_type.shape
                    if len(shape) > self.checker.max_dims:
                        self.violations.append(
                            f"维度数 {len(shape)} 超过限制 {self.checker.max_dims}: {call.op}"
                        )
                    size = 1
                    for s in shape:
                        if isinstance(s, tvm.tir.IntImm):
                            size *= int(s)
                    if size > self.checker.max_size:
                        self.violations.append(
                            f"张量大小 {size} 超过限制 {self.checker.max_size}: {call.op}"
                        )
                return super().visit_call(call)
        
        visitor = Visitor(self)
        visitor.visit(func)
        for v in visitor.violations:
            print(f"[ShapeChecker] WARNING: {v}")
        return func

# 使用
with PassContext(opt_level=0, required_pass=["ShapeChecker"]):
    checker = ShapeChecker(max_dims=4, max_size=2**28)
    mod_checked = tvm.transform.Sequential([checker, transform.InferType()])(mod)
```

### 35.13b.4 Pass 执行时间分析



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import time

def profile_passes(mod, pass_list):
    """分析每个 pass 的执行时间"""
    results = []
    current_mod = mod
    
    for p in pass_list:
        start = time.time()
        try:
            current_mod = p(current_mod)
            elapsed = time.time() - start
            results.append((str(p), elapsed, "success"))
        except Exception as e:
            elapsed = time.time() - start
            results.append((str(p), elapsed, f"failed: {e}"))
    
    # 打印结果
    print(f"{'Pass':<40} {'Time (ms)':>10} {'Status':>10}")
    print("-" * 65)
    for name, elapsed, status in results:
        print(f"{name:<40} {elapsed*1000:>10.2f} {status:>10}")
    
    return results

# 使用
pass_list = [
    transform.InferType(),
    transform.FoldConstant(),
    transform.FuseOps(fuse_opt_level=2),
    transform.CombineParallelConv2D(),
]
profile_passes(mod, pass_list)
```

<div data-component="PassPipelineDebugger"></div>

---

## 35.15 TVM 内部断言与错误处理

### 35.13c.1 ICHECK 宏

TVM 使用 `ICHECK` 宏（Internal Check）进行内部断言。与 `DCHECK` 不同，`ICHECK` 在 release 模式下也会生效。

```cpp
// TVM 源码中的 ICHECK 使用示例
// include/tvm/runtime/logging.h

// 基本用法
ICHECK(condition) << "Error message";

// 带条件的检查
ICHECK_EQ(a, b) << "Expected a == b, got " << a << " vs " << b;
ICHECK_NE(a, b) << "Expected a != b";
ICHECK_LT(a, b) << "Expected a < b";
ICHECK_LE(a, b) << "Expected a <= b";
ICHECK_GT(a, b) << "Expected a > b";
ICHECK_GE(a, b) << "Expected a >= b";
```

### 35.13c.2 TVMError 异常类



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# Python 中的 TVMError
import tvm._ffi.base

# TVMError 是所有 TVM 错误的基类
try:
    # TVM 操作
    pass
except tvm.TVMError as e:
    print(f"TVM 错误: {e}")
    # 错误信息通常包含：
    # 1. C++ 源文件和行号
    # 2. 断言条件
    # 3. 附加的错误描述
```

### 35.13c.3 错误恢复策略



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def safe_build(mod, target, params=None, fallback_target=None):
    """带错误恢复的构建函数"""
    try:
        lib = relay.build(mod, target=target, params=params)
        return lib, "primary"
    except tvm.TVMError as e:
        print(f"主目标构建失败: {e}")
        
        if fallback_target:
            try:
                print(f"尝试回退目标: {fallback_target}")
                lib = relay.build(mod, target=fallback_target, params=params)
                return lib, "fallback"
            except tvm.TVMError as e2:
                print(f"回退目标也失败: {e2}")
        
        # 最终回退：使用最低优化级别
        try:
            print("尝试 opt_level=0")
            with PassContext(opt_level=0):
                lib = relay.build(mod, target=target, params=params)
            return lib, "low_opt"
        except tvm.TVMError as e3:
            raise RuntimeError(f"所有构建尝试均失败: {e3}") from e3
```

---

## 35.16 数值调试

### 35.13d.1 数值不一致问题

有时模型在不同后端（CPU vs GPU）或不同优化级别下产生不同的结果。这可能是正常的浮点精度差异，也可能是调度错误。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import numpy as np

def compare_results(result_cpu, result_gpu, atol=1e-5, rtol=1e-5):
    """比较两个结果的数值差异"""
    cpu_np = result_cpu.numpy()
    gpu_np = result_gpu.numpy()
    
    # 绝对误差
    abs_diff = np.abs(cpu_np - gpu_np)
    max_abs_diff = np.max(abs_diff)
    mean_abs_diff = np.mean(abs_diff)
    
    # 相对误差
    rel_diff = abs_diff / (np.abs(cpu_np) + 1e-8)
    max_rel_diff = np.max(rel_diff)
    
    # 检查是否在容忍范围内
    is_close = np.allclose(cpu_np, gpu_np, atol=atol, rtol=rtol)
    
    print(f"最大绝对误差: {max_abs_diff:.2e}")
    print(f"平均绝对误差: {mean_abs_diff:.2e}")
    print(f"最大相对误差: {max_rel_diff:.2e}")
    print(f"是否一致 (atol={atol}, rtol={rtol}): {is_close}")
    
    if not is_close:
        # 找到差异最大的位置
        idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        print(f"差异最大位置: {idx}")
        print(f"  CPU 值: {cpu_np[idx]}")
        print(f"  GPU 值: {gpu_np[idx]}")
        print(f"  差异: {abs_diff[idx]}")
    
    return is_close
```

### 35.13d.2 数值溢出检测



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def check_tensor_health(tensor, name="tensor"):
    """检查张量的数值健康状态"""
    arr = tensor.numpy()
    
    print(f"\n=== {name} 数值检查 ===")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  范围: [{np.min(arr):.6e}, {np.max(arr):.6e}]")
    print(f"  均值: {np.mean(arr):.6e}")
    print(f"  标准差: {np.std(arr):.6e}")
    
    # 检查异常值
    nan_count = np.sum(np.isnan(arr))
    inf_count = np.sum(np.isinf(arr))
    zero_count = np.sum(arr == 0)
    
    if nan_count > 0:
        print(f"  ⚠ 包含 {nan_count} 个 NaN!")
    if inf_count > 0:
        print(f"  ⚠ 包含 {inf_count} 个 Inf!")
    if zero_count > arr.size * 0.9:
        print(f"  ⚠ 90% 以上元素为零 ({zero_count}/{arr.size})")
    
    # 检查数值分布
    percentiles = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])
    print(f"  分位数: 1%={percentiles[0]:.4e}, 50%={percentiles[3]:.4e}, 99%={percentiles[6]:.4e}")
```

### 35.13d.3 中间结果打印



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 在 Relay 中插入调试打印节点
# 使用 relay.annotation 注入调试代码

def add_debug_prints(mod, print_every=10):
    """在 Relay 模型中添加调试打印（用于开发调试）"""
    class DebugInjector(relay.ExprMutator):
        def __init__(self):
            super().__init__()
            self.count = 0
        
        def visit_call(self, call):
            new_call = super().visit_call(call)
            self.count += 1
            
            if self.count % print_every == 0:
                # 注入一个 debug 标注
                # 注意：这需要在 runtime 中实现对应的处理
                return relay.annotation.checkpoint(new_call)
            
            return new_call
    
    injector = DebugInjector()
    return injector(mod)
```

### 35.13d.4 参考实现对比



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def verify_against_reference(s, inputs, ref_func, target="llvm"):
    """将 TVM 调度的结果与参考实现对比"""
    # TVM 实现
    func = tvm.build(s, inputs, target=target)
    ctx = tvm.device(target, 0)
    
    # 创建输入数据
    input_arrays = []
    for inp in inputs:
        if isinstance(inp, te.Tensor):
            shape = [int(d) for d in inp.shape]
            dtype = inp.dtype
            arr_np = np.random.uniform(-1, 1, shape).astype(dtype)
            input_arrays.append(tvm.nd.array(arr_np, ctx))
    
    # TVM 计算
    output = tvm.nd.array(np.zeros_like(input_arrays[-1].numpy()), ctx)
    func(*input_arrays, output)
    
    # 参考计算
    ref_output = ref_func(*[x.numpy() for x in input_arrays])
    
    # 对比
    return compare_results(output, tvm.nd.array(ref_output, ctx))
```

<div data-component="NumericalDebugger"></div>

---

## 35.17 多线程调试

### 35.13e.1 线程安全问题

TVM 的并行调度可能引入线程安全问题，特别是在共享数据的场景下。

```python
# 常见的线程安全问题

# 问题 1：False sharing
# 当不同线程写入同一 cache line 的不同位置时
# 会导致性能严重下降

# 检测方法：
# 使用 perf 工具（Linux）
# perf stat -e cache-misses,cache-references python my_script.py

# 问题 2：数据竞争
# 当多个线程同时读写同一内存位置时

# 检测方法：
# TVM_NUM_THREADS=1 单线程运行，对比结果
# 如果单线程正确但多线程不正确，可能存在数据竞争
```

### 35.13e.2 单线程调试模式



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 强制单线程运行以便调试
import os
os.environ["TVM_NUM_THREADS"] = "1"

# 在代码中设置
import tvm
tvm.cpu(0).max_num_threads  # 查看最大线程数

# 对比单线程和多线程的结果
def test_thread_safety(func, inputs, ctx):
    """测试多线程安全性"""
    # 单线程运行
    os.environ["TVM_NUM_THREADS"] = "1"
    result_single = func(*inputs)
    
    # 多线程运行
    os.environ["TVM_NUM_THREADS"] = str(os.cpu_count())
    result_multi = func(*inputs)
    
    # 对比
    is_same = compare_results(result_single, result_multi, atol=0, rtol=0)
    if not is_same:
        print("⚠ 检测到线程安全问题！")
    else:
        print("✓ 多线程结果一致")
    
    return is_same
```

---

## 35.18 调试工具链总结

### 35.18.1 调试工具速查表

| 工具 | 用途 | 使用场景 |
|------|------|---------|
| `tvm.lower()` | 查看 TIR | 验证调度变换 |
| `relay.print()` | 打印 Relay IR | 查看计算图 |
| `visualization.plot()` | 可视化计算图 | 理解模型结构 |
| `TVM_LOG_DEBUG` | 启用详细日志 | 定位内部错误 |
| `PassContext(opt_level=0)` | 禁用优化 | 排查 pass 问题 |
| `disabled_pass` | 禁用特定 pass | 隔离问题 pass |
| `PrintIR` pass | 打印中间 IR | 查看 pass 效果 |
| `time_evaluator` | 性能测量 | 量化性能 |
| `graph_executor.profile()` | 算子 profiling | 定位性能瓶颈 |
| `addr2line` / `gdb` | 解析 C++ 堆栈 | C++ 崩溃调试 |
| `tracemalloc` | 内存追踪 | 检测内存泄漏 |
| `rpc` | 远程设备调试 | 交叉编译调试 |
| `NSight` / `nvprof` | GPU profiling | CUDA 性能分析 |

### 35.18.2 错误类型 → 工具映射



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
编译错误
├── Shape mismatch → relay.transform.InferType()
├── Dtype 错误 → relay.print() 检查类型
├── 不支持算子 → 查看 frontend 文档
├── Pass 崩溃 → PassContext(0) + 二分禁用
├── Lowering 错误 → tvm.lower() + 检查 TIR
└── CodeGen 错误 → 检查目标代码 + LLVM 版本

运行时错误
├── 数值错误 → 检查调度 + 中间结果
├── 内存错误 → ASAN/MSAN + bound check
├── 性能问题 → profiling + checklist
└── RPC 错误 → 网络诊断 + 设备检查
```

### 35.18.3 调试命令速查



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```bash
# 环境变量
export TVM_LOG_DEBUG="default"
export TVM_LOG_LEVEL="DEBUG"
export TVM_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1
export TVM_OPENCL_DUMP_SOURCE=1
export CODEGEN_DEBUG=1

# Python 调试
python -X faulthandler my_script.py  # 启用 C-level traceback

# GPU 调试
CUDA_LAUNCH_BLOCKING=1 python my_script.py  # 同步 CUDA 执行
nsys profile python my_script.py             # NSight profiling
compute-sanitizer python my_script.py         # CUDA memory check
```

<div data-component="DebugToolchainSummary"></div>

---

## 35.99 文字内容强化：调试 的工程化理解

这一节用于把前文的 API、IR、Pass、Runtime 和部署片段串联为更完整的工程叙事。
很多学习者第一次阅读 TVM 文档时会觉得示例代码很多，但真正上线时仍然不知道如何判断方案是否可靠。
原因在于 TVM 不是单个推理库，而是一条从模型语义到硬件代码的编译链路。
链路越长，越需要把每一步的业务目标、内部机制、适用边界和失败模式说清楚。

### 35.99.1 代码解读的阅读方法

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

- 围绕“前端导入、Pass、TIR、CodeGen 和 Runtime 的分层定位”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“IR dump、日志、最小复现和数值对齐的组合方法”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“远程 RPC、设备差异和性能问题的排查顺序”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与框架调试器、编译器 sanitizer 和硬件 profiler 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 35.99.2 业务意义

1. 调试 的业务价值不只是让模型跑得更快，而是让同一个模型可以在不同成本、功耗和延迟约束下交付。
2. 在服务器场景中，核心指标通常是吞吐、P95/P99 延迟、资源利用率和多租户隔离。
3. 在移动端场景中，核心指标通常是首帧时间、持续发热、内存峰值和包体大小。
4. 在嵌入式场景中，核心指标通常是 Flash 占用、静态内存、实时性和掉电恢复能力。
5. 在云端批处理场景中，编译时间可以接受更长，但调优记录和缓存复用变得非常重要。
6. 在在线服务场景中，编译产物需要可回滚、可审计、可灰度，而不能只在开发机上验证。
7. 业务方关心的是 SLA、成本和稳定性，编译器工程师关心的是 IR 正确性、优化空间和后端能力。
8. 优秀的 TVM 项目需要把这两类语言翻译成共同的指标体系。
9. 当优化收益只有少量百分点时，应评估它是否值得引入新的维护复杂度。
10. 当优化收益很大但只在少数输入上成立时，应评估输入分布变化后的风险。

- 围绕“前端导入、Pass、TIR、CodeGen 和 Runtime 的分层定位”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“IR dump、日志、最小复现和数值对齐的组合方法”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“远程 RPC、设备差异和性能问题的排查顺序”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与框架调试器、编译器 sanitizer 和硬件 profiler 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 35.99.3 TVM 内部机制

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

- 围绕“前端导入、Pass、TIR、CodeGen 和 Runtime 的分层定位”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“IR dump、日志、最小复现和数值对齐的组合方法”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“远程 RPC、设备差异和性能问题的排查顺序”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与框架调试器、编译器 sanitizer 和硬件 profiler 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 35.99.4 适用场景

1. 当模型结构相对稳定、目标硬件明确、性能收益可以通过基准测试确认时，调试 相关技术最容易发挥价值。
2. 当团队需要支持多种硬件后端时，TVM 的统一 IR 和 Target 抽象可以降低重复适配成本。
3. 当模型中存在框架运行时开销、算子融合机会或布局转换冗余时，编译优化通常能带来明显收益。
4. 当部署环境不能依赖完整 Python 栈时，AOT、CRT 或导出后的 runtime artifact 更有意义。
5. 当硬件厂商提供高性能库但模型图需要复杂切分时，BYOC 和外部 codegen 是常见选择。
6. 当输入形状变化频繁时，应提前设计 shape 策略，而不是在上线前才补动态形状支持。
7. 当模型版本迭代频繁时，应把编译、调优、验证和发布纳入 CI/CD。
8. 当业务对精度非常敏感时，应把优化收益和数值回归一起评估。
9. 当系统存在多模型串联时，应评估端到端 pipeline，而不是只优化单个模型。
10. 当部署设备数量很大时，编译产物的一致性和可追踪性比单次实验性能更重要。

- 围绕“前端导入、Pass、TIR、CodeGen 和 Runtime 的分层定位”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“IR dump、日志、最小复现和数值对齐的组合方法”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“远程 RPC、设备差异和性能问题的排查顺序”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与框架调试器、编译器 sanitizer 和硬件 profiler 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 35.99.5 限制条件

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

- 围绕“前端导入、Pass、TIR、CodeGen 和 Runtime 的分层定位”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“IR dump、日志、最小复现和数值对齐的组合方法”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“远程 RPC、设备差异和性能问题的排查顺序”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与框架调试器、编译器 sanitizer 和硬件 profiler 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 35.99.6 工程经验

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

- 围绕“前端导入、Pass、TIR、CodeGen 和 Runtime 的分层定位”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“IR dump、日志、最小复现和数值对齐的组合方法”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“远程 RPC、设备差异和性能问题的排查顺序”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与框架调试器、编译器 sanitizer 和硬件 profiler 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 35.99.7 常见误区

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

- 围绕“前端导入、Pass、TIR、CodeGen 和 Runtime 的分层定位”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“IR dump、日志、最小复现和数值对齐的组合方法”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“远程 RPC、设备差异和性能问题的排查顺序”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与框架调试器、编译器 sanitizer 和硬件 profiler 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 35.99.8 生产部署注意事项

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

- 围绕“前端导入、Pass、TIR、CodeGen 和 Runtime 的分层定位”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“IR dump、日志、最小复现和数值对齐的组合方法”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“远程 RPC、设备差异和性能问题的排查顺序”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与框架调试器、编译器 sanitizer 和硬件 profiler 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 35.99.9 与同类系统对比

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

- 围绕“前端导入、Pass、TIR、CodeGen 和 Runtime 的分层定位”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“IR dump、日志、最小复现和数值对齐的组合方法”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“远程 RPC、设备差异和性能问题的排查顺序”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与框架调试器、编译器 sanitizer 和硬件 profiler 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 35.99.10 章节复盘

1. 回到本章，调试 的关键不是记住所有 API，而是理解为什么这些 API 会出现在编译链路的这个位置。
2. 当你看到一段代码时，应能说出它改变了模型语义、调度空间、内存布局、运行时入口还是部署产物。
3. 当你看到一个性能数字时，应能说出它的测试输入、硬件状态、计时方法和误差范围。
4. 当你看到一个优化 pass 时，应能说出它依赖的前置假设和可能破坏的边界条件。
5. 当你准备上线时，应能说出失败后如何回滚、如何复现、如何定位和如何与业务方沟通影响。
6. 这套思维比单个示例更重要，因为 TVM 的 API 会演进，但编译部署的工程约束长期稳定。
7. 后续学习中，可以把每一章都转化为一张决策表：何时使用、收益来自哪里、风险是什么、如何验证。
8. 只有把代码、机制和工程策略放在一起，TVM 才不只是工具箱，而是可运行的生产系统。
9. 因此，本章新增的文字说明应作为阅读代码段的上下文，而不是替代对原始代码的逐行理解。
10. 如果遇到与示例不一致的实际项目，应优先回到模型约束和目标硬件，而不是机械套用章节流程。

- 围绕“前端导入、Pass、TIR、CodeGen 和 Runtime 的分层定位”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“IR dump、日志、最小复现和数值对齐的组合方法”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“远程 RPC、设备差异和性能问题的排查顺序”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与框架调试器、编译器 sanitizer 和硬件 profiler 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。


## 35.19 本章小结

### 35.19.1 核心要点回顾

1. **错误定位**：理解 TVM 编译管线的各个阶段，快速定位错误发生的阶段
2. **编译错误**：Shape mismatch、dtype 不匹配、不支持算子是最常见的三类编译错误
3. **可视化工具**：`visualization.plot()` 和 `tvm.lower()` 是最常用的调试工具
4. **Pass 调试**：通过 `PassContext(opt_level=0)` 和 `disabled_pass` 隔离问题
5. **日志系统**：`TVM_LOG_DEBUG` 和 `TVM_LOG_LEVEL` 控制调试日志输出
6. **性能排查**：建立系统化的 checklist，从计算定义到运行时配置逐项检查
7. **RPC 调试**：远程设备调试需要额外的网络和设备诊断
8. **RuntimeError**：学会阅读 C++/Python 混合堆栈跟踪

### 35.19.2 调试心态

> "调试是理解程序的最有效方式。" —— Gerald Weinberg

- **不要猜测**：用工具观察，而不是凭直觉修改
- **二分法**：逐个禁用 pass / 降低优化级别来隔离问题
- **最小复现**：将问题代码缩减到最小可复现示例
- **阅读源码**：TVM 是开源的，最终手段是阅读 C++ 源码

### 35.19.3 练习

**练习 1**：故意制造一个 shape mismatch 错误，使用 `InferType()` pass 定位错误发生的节点。

```python
# 创建一个包含 shape mismatch 的 Relay 模型
# 使用 relay.transform.InferType() 捕获错误
# 分析错误信息，定位问题节点
```

**练习 2**：使用 `tvm.lower()` 对比一个正确调度和一个错误调度的 TIR 输出。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 定义计算
A = te.placeholder((64,), name="A")
B = te.placeholder((64,), name="B")
C = te.compute((64,), lambda i: A[i] + B[i], name="C")

# 调度 1：正确
s1 = te.create_schedule(C.op)
i, = s1[C].op.axis
i0, i1 = s1[C].split(i, factor=8)
s1[C].vectorize(i1)

# 调度 2：错误（对整个轴向量化，但范围不是常量）
s2 = te.create_schedule(C.op)
i, = s2[C].op.axis
# s2[C].vectorize(i)  # 这会失败吗？

# 对比两者的 TIR 输出
```

**练习 3**：设置 `TVM_LOG_DEBUG` 环境变量，运行一个简单的 TVM 程序，分析输出的日志信息。

**练习 4**：使用 `PassContext(disabled_pass=[...])` 逐个禁用 Relay pass，找到导致模型精度下降的 pass。

---

> **下一章预告**：第 36 章将介绍 TVM 的性能基准测试与自动化测试框架。


**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。

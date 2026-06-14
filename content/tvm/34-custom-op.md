> **学习目标**：
> - 理解 TVM 中自定义算子的必要性与应用场景
> - 掌握 Relay 算子注册（Op Registration）的完整流程
> - 学会使用 TE Compute 定义自定义算子的计算逻辑
> - 理解外部函数调用（Extern Function Call）机制
> - 掌握 BYOC（Bring Your Own Codegen）框架的使用方法

---

## 34.1 自定义算子概述

### 34.1.1 为什么需要自定义算子？

虽然 TVM 内置了大量常用算子，但在实际部署中经常需要自定义算子：

| 场景 | 示例 | 说明 |
|------|------|------|
| **新算子** | FlashAttention、Mamba SSM | 最新论文中的新计算模式 |
| **第三方库** | cuDNN、oneDNN、TensorRT | 利用已高度优化的库 |
| **硬件特化** | NPU 指令、DSP 原语 | 特定硬件的专用指令 |
| **业务逻辑** | 自定义 NMS、特殊激活函数 | 业务特有的计算逻辑 |
| **遗留代码** | 已有的 C/CUDA 实现 | 复用已验证的代码 |

### 34.1.2 自定义算子的三种方式

TVM 提供了三种自定义算子的方式：

```
┌─────────────────────────────────────────────────────────────┐
│                自定义算子的方式                               │
│                                                             │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │ TE Compute    │  │ Extern Func   │  │ BYOC          │   │
│  │ (纯 TVM 实现)  │  │ (外部函数调用) │  │ (代码生成集成) │   │
│  └───────────────┘  └───────────────┘  └───────────────┘   │
│         ↓                  ↓                   ↓            │
│  使用 TVM 的调度    调用外部 C/CUDA    利用第三方编译器       │
│  原语定义计算       函数               (TensorRT, CUTLASS)   │
│                                                             │
│  适用：全新算子      适用：已有实现      适用：已有代码生成器  │
└─────────────────────────────────────────────────────────────┘
```

### 34.1.3 算子注册的层次体系

TVM 的算子注册分为多个层次：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
算子注册层次：

1. Relay Op 定义（算子接口层）
   - 定义算子的名称、参数、属性
   - 注册类型推断规则
   - 注册计算逻辑的引用

2. Compute 定义（计算逻辑层）
   - 使用 TE 或 TIR 定义具体计算
   - 一个 Relay Op 可以有多个 Compute 实现

3. Schedule 定义（调度层）
   - 为 Compute 定义调度策略
   - 不同硬件后端可以有不同的 Schedule

4. CodeGen 集成（代码生成层）
   - 为特定后端生成优化代码
   - BYOC 框架提供统一接口
```

---

## 34.2 Relay 算子注册

### 34.2.1 算子注册表（Op Registry）

TVM 使用全局注册表管理所有算子，定义在 `include/tvm/relay/op.h`：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
class OpRegistry {
 public:
  // 获取或创建算子
  static OpRegistry& Register(const String& name, int level);

  // 设置属性
  OpRegistry& set_num_inputs(int n);
  OpRegistry& add_argument(const String& name, const String& type_info);
  OpRegistry& set_attrs_type(const String& type_key);
  OpRegistry& set_support_level(int level);

  // 注册各种函数
  template <typename F>
  OpRegistry& set_attr(const String& attr_name, F f, int plevel);

  OpRegistry& add_type_rel(const String& name, FTVMRelayTypeRel func);
};
```

### 34.2.2 定义一个新的 Relay 算子

下面以定义一个 `my_gelu` 算子为例，展示完整的注册流程：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// src/relay/op/tensor/unary.cc 或自定义文件

#include <tvm/relay/op.h>
#include <tvm/relay/type.h>

namespace tvm {
namespace relay {

// 第一步：注册算子的基本信息
TVM_REGISTER_OP("my_gelu")
    .set_num_inputs(1)                           // 输入数量
    .add_argument("data", "Tensor", "Input tensor")  // 参数描述
    .set_support_level(10)                        // 支持级别
    .set_attr<FTVMRelayTypeRel>("FTVMRelayTypeRel", "IdentityRel")
    .set_attr<TOpPattern>("TOpPattern", kElemWise);

// 第二步：注册类型推断函数
bool MyGeluRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
  // types[0] 是输入类型，types[1] 是返回类型（待填充）
  ICHECK_EQ(types.size(), 2);

  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  // 输出类型与输入类型相同
  reporter->Assign(types[1], TensorType(data->shape, data->dtype));
  return true;
}

TVM_REGISTER_OP("my_gelu")
    .add_type_rel("MyGelu", MyGeluRel);

}  // namespace relay
}  // namespace tvm
```

### 34.2.3 注册属性

算子可以注册各种属性（Attrs），用于传递编译时参数：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// 定义属性结构
struct MyConvAttrs : public tvm::AttrsNode<MyConvAttrs> {
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> dilation;
  int groups;
  std::string data_layout;
  std::string kernel_layout;

  TVM_DECLARE_ATTRS(MyConvAttrs, "relay.attrs.MyConvAttrs") {
    TVM_ATTR_FIELD(strides).set_default({1, 1});
    TVM_ATTR_FIELD(padding).set_default({0, 0});
    TVM_ATTR_FIELD(dilation).set_default({1, 1});
    TVM_ATTR_FIELD(groups).set_default(1);
    TVM_ATTR_FIELD(data_layout).set_default("NCHW");
    TVM_ATTR_FIELD(kernel_layout).set_default("OIHW");
  }
};

// 注册属性
TVM_REGISTER_NODE_TYPE(MyConvAttrs);

TVM_REGISTER_OP("my_conv2d")
    .set_attrs_type<MyConvAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "Input data")
    .add_argument("weight", "Tensor", "Convolution weight");
```

### 34.2.4 注册 Python 前端

在 Python 中为自定义算子创建便捷的调用接口：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# python/tvm/relay/op/nn/nn.py

def my_gelu(data):
    """自定义 GELU 激活函数

    Parameters
    ----------
    data : relay.Expr
        输入张量

    Returns
    -------
    result : relay.Expr
        应用 GELU 后的张量
    """
    return _make.my_gelu(data)

# 或者使用 FFI 直接调用
from tvm._ffi import register_func

@register_func("relay.op.my_gelu")
def my_gelu_ffi(data):
    """FFI 注册的算子实现"""
    return _ffi_api.my_gelu(data)
```

### 34.2.5 在 Relay 中使用自定义算子



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import relay

# 方式一：直接调用
x = relay.var("x", relay.TensorType([1, 64, 56, 56], "float32"))
y = relay.op.get("my_gelu")(x)

# 方式二：通过 Python 封装
from tvm.relay.op.nn import my_gelu
y = my_gelu(x)

# 方式三：使用 Call 节点
y = relay.Call(relay.op.get("my_gelu"), [x])
```

---

## 34.3 TE Compute 定义算子

### 34.3.1 TE 的设计哲学

Tensor Expression (TE) 是 TVM 定义算子计算逻辑的核心方式。TE 的核心思想是 **计算与调度分离**：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
算子 = Compute（计算是什么）+ Schedule（如何执行）

Compute：定义输入到输出的数学映射
Schedule：定义循环结构、内存布局、并行策略等
```

### 34.3.2 基本的 TE Compute



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import te

# 定义输入占位符
M, N, K = 1024, 1024, 1024
A = te.placeholder((M, K), dtype="float32", name="A")
B = te.placeholder((K, N), dtype="float32", name="B")

# 定义规约轴
k = te.reduce_axis((0, K), name="k")

# 定义计算：矩阵乘法
C = te.compute(
    (M, N),
    lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
    name="C"
)

# 创建默认调度
s = te.create_schedule(C.op)
```

### 34.3.3 自定义 GELU 算子的 TE 实现



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import tvm
from tvm import te
import numpy as np

def te_gelu(A):
    """使用 TE 定义 GELU 激活函数

    GELU(x) = x * Φ(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    # 获取输入形状
    shape = A.shape

    # 定义常量
    sqrt_2_over_pi = te.const(np.sqrt(2.0 / np.pi), dtype="float32")
    coeff = te.const(0.044715, dtype="float32")
    half = te.const(0.5, dtype="float32")
    one = te.const(1.0, dtype="float32")

    # 定义计算
    return te.compute(
        shape,
        lambda *indices: half * A(*indices) * (
            one + te.tanh(
                sqrt_2_over_pi * (A(*indices) + coeff * A(*indices) * A(*indices) * A(*indices))
            )
        ),
        name="gelu"
    )

# 使用
A = te.placeholder((128, 256), dtype="float32", name="A")
B = te_gelu(A)
s = te.create_schedule(B.op)
```

### 34.3.4 注册 TE Compute 到算子

将 TE 定义的计算逻辑注册为 Relay 算子的实现：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import relay, te

# 第一步：定义 TE Compute
def my_gelu_compute(attrs, inputs, out_type):
    """Relay 算子的 TE Compute 实现"""
    return [te_gelu(inputs[0])]

# 第二步：注册 Compute 实现
from tvm.relay.op import op as _op

@_op.register("my_gelu")
def my_gelu_func():
    """注册 my_gelu 算子的完整定义"""
    return {
        "FTVMCompute": my_gelu_compute,
        "TOpPattern": "elemewise",
    }

# 第三步：在 Relay 中使用
x = relay.var("x", relay.TensorType([128, 256], "float32"))
y = relay.op.get("my_gelu")(x)
```

### 34.3.5 带属性的 TE Compute

对于需要属性参数的算子：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def my_pool2d_compute(attrs, inputs, out_type):
    """2D 池化算子的 TE Compute"""
    # 解析属性
    pool_size = attrs.pool_size
    strides = attrs.strides
    padding = attrs.padding
    pool_type = attrs.pool_type  # "avg" 或 "max"

    data = inputs[0]
    batch, channels, in_h, in_w = data.shape
    out_h = (in_h + 2 * padding[0] - pool_size[0]) // strides[0] + 1
    out_w = (in_w + 2 * padding[1] - pool_size[1]) // strides[1] + 1

    # 定义池化轴
    kh = te.reduce_axis((0, pool_size[0]), name="kh")
    kw = te.reduce_axis((0, pool_size[1]), name="kw")

    if pool_type == "avg":
        return [te.compute(
            (batch, channels, out_h, out_w),
            lambda n, c, oh, ow: te.sum(
                data[n, c, oh * strides[0] + kh, ow * strides[1] + kw],
                axis=[kh, kw]
            ) / (pool_size[0] * pool_size[1]),
            name="pool2d"
        )]
    else:  # max
        return [te.compute(
            (batch, channels, out_h, out_w),
            lambda n, c, oh, ow: te.max(
                data[n, c, oh * strides[0] + kh, ow * strides[1] + kw],
                axis=[kh, kw]
            ),
            name="pool2d"
        )]
```

---

## 34.4 外部函数调用

### 34.4.1 外部函数调用的动机

有时候我们需要调用已有的 C/C++ 或 CUDA 实现，而不是用 TE 重新实现。TVM 提供了 `call_extern` 和 `call_packed` 机制：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import te

# 方式一：call_extern - 调用 C 函数
def my_custom_kernel(data):
    """使用外部 C 函数实现的自定义 kernel"""
    return te.compute(
        data.shape,
        lambda *i: tvm.tir.call_extern(
            "float32",                    # 返回类型
            "my_custom_c_kernel",         # C 函数名
            data(*i),                     # 传入参数
        ),
        name="custom_output"
    )

# 方式二：call_packed - 调用 PackedFunc
def my_packed_kernel(data):
    """使用 PackedFunc 实现的自定义 kernel"""
    return te.compute(
        data.shape,
        lambda *i: tvm.tir.call_packed(
            "my_packed_func",
            data,
        ),
        name="packed_output"
    )
```

### 34.4.2 注册外部 C 函数

首先需要实现 C 函数，然后在 TVM 中注册：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// my_custom_ops.cc

#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

// 实现自定义 kernel
void my_custom_c_kernel(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        // 自定义计算逻辑
        output[i] = input[i] * input[i] + 1.0f;
    }
}

// 注册为 TVM PackedFunc
TVM_REGISTER_GLOBAL("my_custom_c_kernel")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    DLTensor* input = args[0];
    DLTensor* output = args[1];
    int n = args[2];

    float* in_data = static_cast<float*>(input->data);
    float* out_data = static_cast<float*>(output->data);

    my_custom_c_kernel(in_data, out_data, n);
});
```

### 34.4.3 CUDA 外部函数

对于 GPU 计算，可以注册 CUDA kernel：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// my_custom_cuda_ops.cc

#include <tvm/runtime/registry.h>
#include <cuda_runtime.h>

// CUDA kernel 实现（在 .cu 文件中）
// __global__ void my_cuda_kernel(float* input, float* output, int n) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < n) {
//         output[idx] = input[idx] * input[idx] + 1.0f;
//     }
// }

// 注册 CUDA 函数
TVM_REGISTER_GLOBAL("my_custom_cuda_kernel")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    DLTensor* input = args[0];
    DLTensor* output = args[1];
    int n = args[2];

    float* in_data = static_cast<float*>(input->data);
    float* out_data = static_cast<float*>(output->data);

    // 配置 kernel 参数
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // 启动 CUDA kernel
    my_cuda_kernel<<<blocks, threads>>>(in_data, out_data, n);
    cudaDeviceSynchronize();
});
```

### 34.4.4 使用 External Library

TVM 支持直接链接外部库（如 cuDNN、MKL）：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import te, topi
import tvm.contrib.cudnn as cudnn

# 使用 cuDNN 实现卷积
def conv2d_cudnn(data, weight, strides, padding, dilation, groups):
    """使用 cuDNN 的卷积实现"""
    return topi.nn.conv2d_cudnn(data, weight, strides, padding, dilation, groups)

# 使用 MKL 实现矩阵乘法
import tvm.contrib.mkl as mkl

def matmul_mkl(A, B):
    """使用 MKL 的矩阵乘法实现"""
    return topi.nn.matmul_mkl(A, B)
```

### 34.4.5 混合调度

可以将 TE 计算与外部函数调用混合使用：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import te

def my_fused_op(data, weight):
    """混合 TE 和外部调用的自定义算子"""
    # 步骤1：使用 TE 定义的部分计算
    M, K = data.shape
    N, _ = weight.shape

    # 矩阵乘法部分使用 TE
    k = te.reduce_axis((0, K), name="k")
    matmul_out = te.compute(
        (M, N),
        lambda i, j: te.sum(data[i, k] * weight[j, k], axis=k),
        name="matmul"
    )

    # 步骤2：激活函数部分使用外部调用
    result = te.compute(
        (M, N),
        lambda i, j: tvm.tir.call_extern(
            "float32",
            "my_custom_gelu",
            matmul_out[i, j],
        ),
        name="gelu"
    )

    return result
```

---

## 34.5 BYOC（Bring Your Own Codegen）

### 34.5.1 BYOC 的设计动机

BYOC 框架允许将部分计算图委托给第三方编译器（如 TensorRT、CUTLASS、ONNX Runtime），这是集成外部算子最灵活的方式：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
完整的计算图
    ↓ BYOC Pattern Matching
匹配的子图 → 外部编译器（如 TensorRT）
未匹配的部分 → TVM 原生编译
    ↓ 合并执行
```

### 34.5.2 BYOC 架构

BYOC 框架定义在 `src/relay/backend/contrib/`：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
src/relay/backend/contrib/
├── codegen_c/          # C 代码生成示例
│   ├── codegen_c.h
│   └── codegen_c.cc
├── codegen_json/       # JSON 序列化
│   └── codegen_json.cc
├── tensorrt/           # TensorRT 集成
│   ├── tensorrt_codegen.cc
│   └── tensorrt_runtime.cc
├── cutlass/            # CUTLASS 集成
│   ├── codegen.cc
│   └── gemm_profiler.cc
├── cublas/             # cuBLAS 集成
│   └── codegen.cc
└── vitis_ai/           # Xilinx Vitis AI
    └── codegen.cc
```

### 34.5.3 定义 BYOC Codegen

要实现自定义的 BYOC codegen，需要实现以下接口：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// include/tvm/relay/backend/contrib/codegen_json.h

class JSONSerializer : public MemoizedExprTranslator<JSONGraphNodeEntry> {
 public:
  // 序列化算子
  virtual std::string JIT(const JSONGraphNode& node) = 0;
};

// 自定义 Codegen
class MyCustomCodegen : public JSONSerializer {
 public:
  // 序列化一个 Call 节点
  JSONGraphNodeEntry VisitCall_(const CallNode* call) override {
    // 获取算子名称
    std::string op_name = GetRef<Call>(call)->op.as<OpNode>()->name;

    // 创建 JSON 节点
    auto node = std::make_shared<JSONGraphNode>(op_name, op_name);

    // 添加输入
    for (auto arg : call->args) {
      auto entry = VisitExpr(arg);
      node->AddInput(entry);
    }

    // 添加属性
    SerializeNodeAttrs(call->attrs, node);

    // 生成代码
    std::string code = JIT(*node);

    return AddNode(node, GetRef<Call>(call));
  }

  // 生成最终代码
  std::string JIT(const JSONGraphNode& node) override {
    // 为外部编译器生成代码
    std::stringstream code;
    code << "my_custom_compile(" << node.name << ")";
    return code.str();
  }
};
```

### 34.5.4 注册 BYOC Codegen



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// 注册自定义 Codegen
TVM_REGISTER_GLOBAL("relay.ext.my_custom")
.set_body_typed([](const IRModule& mod, const String& code) {
  MyCustomCodegen codegen;
  return codegen.Run(mod, code);
});
```

### 34.5.5 使用 BYOC Codegen



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import relay

# 标记需要委托给外部 codegen 的算子
@relay.op.register("my_custom_op", level=15)
def my_custom_op(x, w):
    """使用外部 codegen 的自定义算子"""
    return relay.Call(
        relay.op.get("my_custom_op"),
        [x, w],
        attrs=relay.attrs.Conv2DAttrs(strides=(1, 1), padding=(0, 0))
    )

# 在编译管线中启用 BYOC
from tvm.relay.op.contrib import MyCustomCodegen

mod = ...  # 原始模块

# 使用 pattern matching 标记可以外部化的算子
patterns = [("my_custom", relay.op.get("my_custom_op"))]
mod = relay.transform.MergeComposite(patterns)(mod)

# 调用外部 codegen
mod = relay.transform.RunCodegen(["my_custom"])(mod)

# 编译
lib = relay.build(mod, target="llvm")
```

### 34.5.6 CUTLASS BYOC 示例

TVM 内置了 CUTLASS 的 BYOC 集成，展示了一个完整的集成案例：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm.relay.op.contrib import cutlass

# 配置 CUTLASS codegen
cutlass_target = tvm.target.Target(
    "cuda -model=a100 -max_num_threads=1024",
    host="llvm"
)

# 使用 CUTLASS 编译矩阵乘法
mod = ...  # 包含 matmul 的 Relay 模块

with cutlass_target:
    # 自动选择最优的 CUTLASS kernel
    mod = cutlass.partition_for_cutlass(mod)
    mod = cutlass.compile_for_cutlass(mod, cutlass_target, tmp_dir="/tmp/cutlass")

lib = relay.build(mod, cutlass_target)
```

---

## 34.6 完整的自定义算子开发流程

### 34.6.1 从 TE 到 Relay 的完整流程

下面展示一个完整的自定义算子开发流程，以 `flash_attention` 为例：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import tvm
from tvm import te, relay, tir
import numpy as np

# ====== 第一步：定义 TE Compute ======
def flash_attention_compute(Q, K, V, scale):
    """FlashAttention 的 TE 实现（简化版）"""
    batch, seq_len, head_dim = Q.shape
    _, kv_len, _ = K.shape

    # 定义规约轴
    k = te.reduce_axis((0, kv_len), name="k")

    # 计算 Q*K^T
    attn_scores = te.compute(
        (batch, seq_len, kv_len),
        lambda b, i, j: te.sum(Q[b, i, k] * K[b, j, k], axis=k) * scale,
        name="attn_scores"
    )

    # Softmax（简化为 max-sub-exp-sum 模式）
    max_score = te.compute(
        (batch, seq_len),
        lambda b, i: te.max(attn_scores[b, i, j], axis=j),
        name="max_score"
    )

    # ... 省略完整的 softmax 和加权求和实现 ...

    return output

# ====== 第二步：注册到 Relay ======
def flash_attention_func():
    """注册 FlashAttention 算子"""
    @tvm.ir.register_op_attr("flash_attention", "FTVMCompute")
    def flash_attention_compute_op(attrs, inputs, out_type):
        Q, K, V = inputs
        scale = attrs.scale
        return [flash_attention_compute(Q, K, V, scale)]

    return relay.op.get("flash_attention")

# ====== 第三步：定义 Python API ======
def flash_attention(Q, K, V, scale=None):
    """FlashAttention 的 Python API"""
    if scale is None:
        head_dim = Q.checked_type.shape[-1]
        scale = 1.0 / (head_dim ** 0.5)

    return relay.op.flash_attention(Q, K, V, scale=scale)

# ====== 第四步：使用 ======
batch, seq_len, head_dim = 1, 128, 64
Q = relay.var("Q", relay.TensorType([batch, seq_len, head_dim], "float32"))
K = relay.var("K", relay.TensorType([batch, seq_len, head_dim], "float32"))
V = relay.var("V", relay.TensorType([batch, seq_len, head_dim], "float32"))

output = flash_attention(Q, K, V)
```

### 34.6.2 调度优化

自定义算子通常需要特定的调度优化：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def schedule_flash_attention(s, output):
    """为 FlashAttention 定义优化调度"""
    # 获取所有计算阶段
    attn_scores = s[output].op.input_tensors[0]
    max_score = s[attn_scores].op.output(0)

    # 分块策略
    batch, seq, kv = s[attn_scores].op.axis
    tile_seq = s[attn_scores].split(seq, factor=64)
    tile_kv = s[attn_scores].split(kv, factor=64)

    # 重排循环
    s[attn_scores].reorder(batch, tile_seq, tile_kv, seq, kv)

    # 绑定到 GPU 线程
    bx = s[attn_scores].axis[0]
    tx = s[attn_scores].axis[1]
    s[attn_scores].bind(bx, te.thread_axis("blockIdx.x"))
    s[attn_scores].bind(tx, te.thread_axis("threadIdx.x"))

    # 使用共享内存
    shared_Q = s.cache_read(Q, "shared", [attn_scores])
    shared_K = s.cache_read(K, "shared", [attn_scores])

    return s

# 在编译时使用自定义调度
s = te.create_schedule(output.op)
s = schedule_flash_attention(s, output)
```

### 34.6.3 测试与验证



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def test_custom_op():
    """测试自定义算子的正确性"""
    # 准备测试数据
    batch, seq_len, head_dim = 1, 128, 64
    Q_np = np.random.randn(batch, seq_len, head_dim).astype("float32")
    K_np = np.random.randn(batch, seq_len, head_dim).astype("float32")
    V_np = np.random.randn(batch, seq_len, head_dim).astype("float32")
    scale = 1.0 / (head_dim ** 0.5)

    # 参考实现（PyTorch）
    import torch
    Q_torch = torch.from_numpy(Q_np)
    K_torch = torch.from_numpy(K_np)
    V_torch = torch.from_numpy(V_np)

    attn = torch.matmul(Q_torch, K_torch.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    ref_output = torch.matmul(attn, V_torch).numpy()

    # TVM 实现
    tvm_output = run_tvm_model(Q_np, K_np, V_np, scale)

    # 比较结果
    np.testing.assert_allclose(tvm_output, ref_output, rtol=1e-3, atol=1e-3)
    print("测试通过！")

test_custom_op()
```

---

## 34.7 高级话题：算子性能调优

### 34.7.1 使用 AutoTVM 调优自定义算子



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import auto_scheduler
import tvm.te as te

# 定义自定义算子的计算（不包含调度）
@auto_scheduler.register_workload
def my_custom_op_compute(M, N, K, dtype):
    """注册为 AutoScheduler 工作负载"""
    A = te.placeholder((M, K), dtype=dtype, name="A")
    B = te.placeholder((K, N), dtype=dtype, name="B")

    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="C"
    )
    return [A, B, C]

# AutoScheduler 会自动搜索最优调度
target = tvm.target.Target("cuda")
task = auto_scheduler.SearchTask(
    func=my_custom_op_compute,
    args=(1024, 1024, 1024, "float32"),
    target=target,
)

# 搜索最优调度
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=200,
    measure_callbacks=[auto_scheduler.RecordToFile("custom_op.json")],
    verbose=2,
)
task.tune(tune_option)

# 应用最优调度
sch, args = task.apply_best("custom_op.json")
```

### 34.7.2 使用 MetaSchedule 调优



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import meta_schedule

# 定义 TIR 函数
@T.prim_func
def my_custom_tir(
    A: T.Buffer((1024, 1024), "float32"),
    B: T.Buffer((1024, 1024), "float32"),
    C: T.Buffer((1024, 1024), "float32"),
):
    for i, j, k in T.grid(1024, 1024, 1024):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

# 使用 MetaSchedule 搜索
database = meta_schedule.tune_tir(
    mod=tvm.IRModule({"main": my_custom_tir}),
    target=tvm.target.Target("cuda"),
    config=meta_schedule.TuneConfig(
        max_trials_global=1000,
        max_trials_per_task=200,
    ),
    work_dir="./tune_results",
)

# 应用最优调度
best = database.query_best(my_custom_tir, target="cuda")
print(f"最优耗时: {best.run_secs[0] * 1000:.3f} ms")
```

---

## 34.8 实战案例

### 34.8.1 案例一：自定义 RoPE 算子

Rotary Position Embedding (RoPE) 是 LLM 中常用的算子：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import tvm
from tvm import te
import numpy as np

def te_rope(q, cos_cached, sin_cached):
    """RoPE 的 TE 实现"""
    batch, seq_len, num_heads, head_dim = q.shape

    # 将 head_dim 分成两半
    half_dim = head_dim // 2

    # 定义计算：旋转位置编码
    def rope_compute(b, s, h, d):
        # 获取 cos 和 sin 值
        cos_val = cos_cached[s, d % half_dim]
        sin_val = sin_cached[s, d % half_dim]

        # 获取原始值
        q_val = q[b, s, h, d]

        # 配对值（前半部分与后半部分配对）
        q_pair = tvm.tir.Select(
            d < half_dim,
            q[b, s, h, d + half_dim],
            q[b, s, h, d - half_dim]
        )

        # 应用旋转
        sign = tvm.tir.Select(d < half_dim, -1.0, 1.0)
        return q_val * cos_val + sign * q_pair * sin_val

    return te.compute(
        (batch, seq_len, num_heads, head_dim),
        rope_compute,
        name="rope"
    )

# 注册为 Relay 算子
@tvm.ir.register_op_attr("rope", "FTVMCompute")
def rope_compute_op(attrs, inputs, out_type):
    q, cos_cached, sin_cached = inputs
    return [te_rope(q, cos_cached, sin_cached)]
```

### 34.8.2 案例二：集成第三方 NMS

非极大值抑制（NMS）是目标检测中的关键算子，通常需要集成外部实现：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// nms_extern.cc

#include <tvm/runtime/registry.h>
#include <vector>
#include <algorithm>

struct Box {
    float x1, y1, x2, y2;
    float score;
    int class_id;
};

float iou(const Box& a, const Box& b) {
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);

    float inter_area = std::max(0.0f, inter_x2 - inter_x1) *
                       std::max(0.0f, inter_y2 - inter_y1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);

    return inter_area / (area_a + area_b - inter_area);
}

void nms_kernel(const float* boxes, const float* scores,
                int* indices, int num_boxes, float iou_threshold,
                int max_output) {
    std::vector<int> order(num_boxes);
    for (int i = 0; i < num_boxes; i++) order[i] = i;

    // 按分数排序
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return scores[a] > scores[b];
    });

    std::vector<bool> suppressed(num_boxes, false);
    int count = 0;

    for (int i = 0; i < num_boxes && count < max_output; i++) {
        int idx = order[i];
        if (suppressed[idx]) continue;

        indices[count++] = idx;

        for (int j = i + 1; j < num_boxes; j++) {
            int jdx = order[j];
            if (suppressed[jdx]) continue;

            Box a = {boxes[idx*4], boxes[idx*4+1], boxes[idx*4+2], boxes[idx*4+3]};
            Box b = {boxes[jdx*4], boxes[jdx*4+1], boxes[jdx*4+2], boxes[jdx*4+3]};

            if (iou(a, b) > iou_threshold) {
                suppressed[jdx] = true;
            }
        }
    }
}

// 注册到 TVM
TVM_REGISTER_GLOBAL("tvm.contrib.nms")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    DLTensor* boxes = args[0];
    DLTensor* scores = args[1];
    DLTensor* indices = args[2];
    float iou_threshold = args[3];
    int max_output = args[4];

    int num_boxes = boxes->shape[0];

    nms_kernel(
        static_cast<float*>(boxes->data),
        static_cast<float*>(scores->data),
        static_cast<int*>(indices->data),
        num_boxes, iou_threshold, max_output
    );
});
```

### 34.8.3 案例三：使用 BYOC 集成 TensorRT



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import relay
from tvm.relay.op.contrib import tensorrt

# 1. 准备包含 TensorRT 可处理算子的模块
mod = ...  # Relay 模块

# 2. 配置 TensorRT
tensorrt_config = tensorrt.TRTConfig(
    workspace_size=1 << 30,  # 1GB workspace
    max_batch_size=8,
    precision="fp16",        # 使用 FP16
)

# 3. 分区和编译
mod = tensorrt.partition_for_tensorrt(mod, tensorrt_config)
mod = tensorrt.compile_tensorrt(mod, tensorrt_config)

# 4. 执行
lib = relay.build(mod, target="cuda")
ex = tvm.runtime.vm(lib, tvm.device("cuda", 0))
output = ex["main"](input_data)
```

---

## 34.9 自定义算子的最佳实践

### 34.9.1 选择合适的方式

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| 全新算子，无现有实现 | TE Compute | TVM 可以自动调度优化 |
| 已有 C/CUDA 实现 | Extern Func | 复用已验证的代码 |
| 需要集成第三方库 | BYOC | 统一的集成框架 |
| 需要极致性能 | Extern + 手动调度 | 对关键路径手动优化 |
| 快速原型验证 | TE + AutoSchedule | 自动搜索最优调度 |

### 34.9.2 测试策略



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
class CustomOpTester:
    """自定义算子测试框架"""

    def __init__(self):
        self.test_cases = []

    def add_test_case(self, input_shapes, expected_output=None):
        self.test_cases.append({
            "input_shapes": input_shapes,
            "expected": expected_output,
        })

    def test_correctness(self, op_func, ref_func):
        """测试正确性"""
        for case in self.test_cases:
            # 生成随机输入
            inputs = [
                np.random.randn(*shape).astype("float32")
                for shape in case["input_shapes"]
            ]

            # TVM 输出
            tvm_output = op_func(*inputs)

            # 参考输出
            ref_output = ref_func(*inputs)

            # 比较
            np.testing.assert_allclose(
                tvm_output, ref_output,
                rtol=1e-3, atol=1e-3,
                err_msg=f"测试失败: {case['input_shapes']}"
            )
        print("所有测试通过！")

    def test_performance(self, op_func, dev, num_trials=100):
        """测试性能"""
        times = []
        for case in self.test_cases:
            inputs = [
                tvm.nd.array(np.random.randn(*shape).astype("float32"), dev)
                for shape in case["input_shapes"]
            ]

            # 预热
            for _ in range(10):
                op_func(*inputs)

            # 计时
            start = time.time()
            for _ in range(num_trials):
                op_func(*inputs)
            dev.sync()
            elapsed = (time.time() - start) / num_trials

            times.append(elapsed)
            print(f"输入形状 {case['input_shapes']}: {elapsed*1000:.3f} ms")

        return times
```

### 34.9.3 常见陷阱

| 陷阱 | 说明 | 解决方案 |
|------|------|---------|
| **形状不匹配** | TE Compute 中的形状推断错误 | 使用 `te.placeholder` 明确指定形状 |
| **数据类型不匹配** | 不同类型之间的隐式转换 | 使用 `tvm.tir.Cast` 显式转换 |
| **内存对齐** | 外部函数要求特定的内存对齐 | 使用 `T.aligned` 声明 |
| **异步执行** | GPU kernel 是异步的 | 确保同步后再读取结果 |
| **线程安全** | PackedFunc 不是线程安全的 | 使用锁或为每个线程创建独立实例 |

---

## 34.10 TIR 级别的自定义算子

### 34.10.1 使用 TIR 定义算子

除了 TE，还可以直接使用 TIR 定义算子：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import tir, script

@T.prim_func
def custom_softmax(
    data: T.handle,
    output: T.handle,
    n: T.int64,
    m: T.int64
):
    # 声明 Buffer
    A = T.match_buffer(data, (n, m), "float32")
    B = T.match_buffer(output, (n, m), "float32")

    # 计算 softmax
    for i in T.serial(n):
        # 第一步：找最大值
        max_val = T.float32(-3.4e38)
        for j in T.serial(m):
            max_val = T.max(max_val, A[i, j])

        # 第二步：计算 exp(x - max) 的和
        sum_val = T.float32(0.0)
        for j in T.serial(m):
            sum_val = sum_val + T.exp(A[i, j] - max_val)

        # 第三步：计算 softmax
        for j in T.serial(m):
            B[i, j] = T.exp(A[i, j] - max_val) / sum_val

# 注册为全局函数
@tvm.register_func("tvm.contrib.custom_softmax")
def custom_softmax_func(data, output):
    n, m = data.shape
    custom_softmax(data, output, n, m)
```

### 34.10.2 TIR 算子的调度优化

TIR 算子可以直接应用 TIR 级别的调度：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import tir

@T.prim_func
def optimized_softmax(
    data: T.handle,
    output: T.handle,
    n: T.int64,
    m: T.int64
):
    A = T.match_buffer(data, (n, m), "float32")
    B = T.match_buffer(output, (n, m), "float32")

    # 使用 T.block 定义计算块
    for i in T.serial(n):
        with T.block("row"):
            vi = T.axis.spatial(n, i)

            # 最大值计算
            max_val = T.alloc_buffer((), "float32", scope="local")
            max_val[()] = T.float32(-3.4e38)
            for j in T.serial(m):
                with T.block("max"):
                    vj = T.axis.reduce(m, j)
                    max_val[()] = T.max(max_val[()], A[vi, vj])

            # 求和计算
            sum_val = T.alloc_buffer((), "float32", scope="local")
            sum_val[()] = T.float32(0.0)
            for j in T.serial(m):
                with T.block("sum"):
                    vj = T.axis.reduce(m, j)
                    sum_val[()] = sum_val[()] + T.exp(A[vi, vj] - max_val[()])

            # Softmax 输出
            for j in T.serial(m):
                with T.block("softmax"):
                    vj = T.axis.spatial(m, j)
                    B[vi, vj] = T.exp(A[vi, vj] - max_val[()]) / sum_val[()]
```

### 34.10.3 TIR 自定义原语

TVM 允许注册自定义的 TIR 原语：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// 注册自定义 TIR 原语
TVM_REGISTER_GLOBAL("tir.my_custom_intrin")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    // 定义自定义原语的行为
    PrimExpr expr = args[0];
    *ret = tir::Call(expr.dtype, "my_custom_intrin", {expr});
});

// 在 TIR 中使用
// A[i] = T.my_custom_intrin(B[i])
```

---

## 34.11 算子的序列化与反序列化

### 34.11.1 Relay 算子的序列化

自定义算子需要支持序列化以便模型保存和加载：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm import relay

# 注册自定义算子的序列化函数
@relay.op.register("my_custom_op", "FTVMRelayPrinter")
def my_custom_op_printer(call):
    """自定义算子的打印/序列化函数"""
    # 返回可序列化的表示
    return {
        "op": "my_custom_op",
        "args": [arg for arg in call.args],
        "attrs": call.attrs.asdict() if call.attrs else {},
    }

# 注册反序列化函数
@relay.op.register("my_custom_op", "FTVMRelayParser")
def my_custom_op_parser(data):
    """自定义算子的反序列化函数"""
    args = [relay.from_json(arg) for arg in data["args"]]
    attrs = relay.attrs.MyCustomAttrs(**data["attrs"])
    return relay.op.my_custom_op(*args, **attrs)
```

### 34.11.2 模型导出与加载



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import json
import tvm

def export_model_with_custom_op(mod, params, file_path):
    """导包含自定义算子的模型"""
    # 序列化 Relay 模型
    relay_json = tvm.ir.save_json(mod)

    # 保存参数
    param_bytes = tvm.runtime.save_param_dict(params)

    # 写入文件
    with open(file_path, "wb") as f:
        # 写入 JSON 长度
        json_bytes = relay_json.encode("utf-8")
        f.write(len(json_bytes).to_bytes(8, "little"))
        # 写入 JSON
        f.write(json_bytes)
        # 写入参数
        f.write(param_bytes)

def load_model_with_custom_op(file_path):
    """加载包含自定义算子的模型"""
    with open(file_path, "rb") as f:
        # 读取 JSON 长度
        json_len = int.from_bytes(f.read(8), "little")
        # 读取 JSON
        relay_json = f.read(json_len).decode("utf-8")
        # 读取参数
        param_bytes = f.read()

    # 反序列化
    mod = tvm.ir.load_json(relay_json)
    params = tvm.runtime.load_param_dict(param_bytes)

    return mod, params
```

---

## 34.12 算子的版本管理

### 34.12.1 算子版本号

自定义算子应该有版本管理：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 定义版本化的算子
@relay.op.register("my_custom_op_v2", level=15)
def my_custom_op_v2(x, y, alpha=1.0):
    """版本 2：增加了 alpha 参数"""
    return relay.Call(
        relay.op.get("my_custom_op_v2"),
        [x, y],
        attrs=relay.attrs.MyCustomOpAttrsV2(alpha=alpha)
    )

# 兼容性处理
@relay.op.register("my_custom_op", level=15)
def my_custom_op(x, y):
    """旧版本，重定向到新版本"""
    return my_custom_op_v2(x, y, alpha=1.0)
```

### 34.12.2 算子兼容性检查



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def check_op_compatibility(mod, target_version):
    """检查模块中的算子版本兼容性"""
    incompatible_ops = []

    for gv, func in mod.functions.items():
        if isinstance(func, relay.Function):
            # 遍历函数体中的所有 Call 节点
            for call in relay.analysis.collect_nodes_from_expr(relay.Call, func.body):
                op_name = call.op.name if hasattr(call.op, 'name') else str(call.op)

                # 检查版本兼容性
                if op_name in OP_VERSION_TABLE:
                    op_version = OP_VERSION_TABLE[op_name]
                    if op_version > target_version:
                        incompatible_ops.append({
                            "op": op_name,
                            "version": op_version,
                            "required": target_version,
                            "function": gv.name_hint,
                        })

    return incompatible_ops
```

---

## 34.13 算子的文档生成

### 34.13.1 自动生成 API 文档



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def generate_op_documentation(op_name):
    """生成算子的 API 文档"""
    op = relay.op.get(op_name)

    doc = f"""
# {op_name}

## 描述
{op.get_attr('description') or '无描述'}

## 参数
"""

    # 获取参数信息
    arg_descriptions = op.get_attr('argument_descriptions') or []
    for i, desc in enumerate(arg_descriptions):
        doc += f"- **arg{i}**: {desc}\n"

    # 获取属性信息
    attrs_type = op.get_attr('attrs_type')
    if attrs_type:
        doc += f"\n## 属性 ({attrs_type.__name__})\n"
        for field_name, field_desc in attrs_type.__dict__.items():
            if not field_name.startswith('_'):
                doc += f"- **{field_name}**: {field_desc}\n"

    # 示例
    examples = op.get_attr('examples') or []
    if examples:
        doc += "\n## 示例\n"
        for example in examples:
            doc += f"```python\n{example}\n

**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```\n"

    return doc
```

### 34.13.2 算子测试文档



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def generate_op_test_doc(op_name, test_cases):
    """生成算子的测试文档"""
    doc = f"# {op_name} 测试用例\n\n"

    for i, case in enumerate(test_cases):
        doc += f"## 测试 {i+1}: {case['name']}\n\n"
        doc += f"**输入形状**: {case['input_shapes']}\n"
        doc += f"**期望输出**: {case['expected']}\n"
        doc += f"**测试代码**:\n```python\n{case['code']}\n

**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```\n\n"

    return doc
```

---

## 34.14 算子的性能对比分析

### 34.14.1 多实现性能对比



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def compare_op_implementations(impls, input_shapes, dev, num_trials=100):
    """对比同一算子的不同实现的性能"""
    import time

    results = {}
    for impl_name, impl_func in impls.items():
        # 创建测试数据
        inputs = [tvm.nd.array(np.random.randn(*shape).astype("float32"), dev)
                  for shape in input_shapes]

        # 预热
        for _ in range(10):
            impl_func(*inputs)
        dev.sync()

        # 测量
        start = time.time()
        for _ in range(num_trials):
            impl_func(*inputs)
        dev.sync()
        elapsed = (time.time() - start) / num_trials

        results[impl_name] = {
            "elapsed_ms": elapsed * 1000,
            "throughput": 1.0 / elapsed,
        }

    # 找到最快的实现
    fastest = min(results.items(), key=lambda x: x[1]["elapsed_ms"])

    print("\n性能对比结果：")
    print("-" * 60)
    print(f"{'实现':20s} {'耗时 (ms)':15s} {'相对性能':15s}")
    print("-" * 60)

    for impl_name, result in results.items():
        relative = fastest[1]["elapsed_ms"] / result["elapsed_ms"]
        marker = " ← 最快" if impl_name == fastest[0] else ""
        print(f"{impl_name:20s} {result['elapsed_ms']:15.3f} {relative:15.2f}x{marker}")

    print("-" * 60)
    return results
```

### 34.14.2 不同形状下的性能分析



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def analyze_op_scalability(op_func, shape_ranges, dev):
    """分析算子在不同输入形状下的性能扩展性"""
    results = {}

    for shape_desc, shape_gen in shape_ranges.items():
        shape_results = []
        for shape in shape_gen:
            # 创建输入
            inputs = [tvm.nd.array(np.random.randn(*shape).astype("float32"), dev)]

            # 测量
            time_f = tvm.runtime.module.time_evaluator(
                "main", dev, number=100, repeat=5
            )
            result = time_f(*inputs)

            shape_results.append({
                "shape": shape,
                "elapsed_ms": result.mean * 1000,
                "flops": compute_flops(shape),
            })

        results[shape_desc] = shape_results

    # 分析扩展性
    print("\n性能扩展性分析：")
    for shape_desc, shape_results in results.items():
        print(f"\n{shape_desc}:")
        for r in shape_results:
            efficiency = r["flops"] / (r["elapsed_ms"] / 1000) / 1e9
            print(f"  形状 {r['shape']}: {r['elapsed_ms']:.3f} ms, "
                  f"效率 {efficiency:.1f} GFLOPS")

    return results
```

---

## 34.15 自定义算子的工程实践

### 34.15.1 项目结构

推荐的自定义算子项目结构：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
my_custom_ops/
├── CMakeLists.txt           # 构建配置
├── include/
│   └── my_ops/
│       ├── op_defs.h        # 算子定义头文件
│       └── kernels.h        # kernel 声明
├── src/
│   ├── relay_ops.cc         # Relay 算子注册
│   ├── te_compute.py        # TE Compute 定义
│   ├── tir_kernels.cc       # TIR kernel 实现
│   └── cuda_kernels.cu      # CUDA kernel 实现
├── python/
│   └── my_ops/
│       ├── __init__.py
│       ├── relay_api.py     # Python API
│       └── schedule.py      # 调度定义
├── tests/
│   ├── test_relay_ops.py    # Relay 算子测试
│   ├── test_te_compute.py   # TE Compute 测试
│   └── test_performance.py  # 性能测试
└── docs/
    └── api.md               # API 文档
```

### 34.15.2 CMakeLists.txt 示例



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cmake
cmake_minimum_required(VERSION 3.18)
project(my_custom_ops)

# 查找 TVM
find_package(TVM REQUIRED)

# 编译 Relay 算子库
add_library(my_relay_ops SHARED
    src/relay_ops.cc
)
target_link_libraries(my_relay_ops TVM::tvm)

# 编译 CUDA kernel
find_package(CUDA REQUIRED)
cuda_add_library(my_cuda_kernels SHARED
    src/cuda_kernels.cu
)

# 安装
install(TARGETS my_relay_ops my_cuda_kernels
        LIBRARY DESTINATION lib)
install(DIRECTORY python/my_ops
        DESTINATION lib/python)
```

### 34.15.3 测试框架



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# tests/test_custom_ops.py
import pytest
import tvm
from tvm import relay
import numpy as np

class TestCustomOps:
    """自定义算子测试套件"""

    @pytest.fixture
    def dev(self):
        return tvm.cpu(0)

    @pytest.fixture
    def sample_inputs(self):
        return {
            "x": np.random.randn(128, 256).astype("float32"),
            "w": np.random.randn(256, 512).astype("float32"),
        }

    def test_my_gelu_shape(self, dev, sample_inputs):
        """测试 GELU 算子的输出形状"""
        x = relay.var("x", relay.TensorType([128, 256], "float32"))
        y = relay.op.my_gelu(x)
        func = relay.Function([x], y)

        # 编译并执行
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(tvm.IRModule.from_expr(func), "llvm")
        module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
        module.set_input("x", tvm.nd.array(sample_inputs["x"], dev))
        module.run()
        output = module.get_output(0)

        # 验证形状
        assert output.shape == (128, 256)

    def test_my_gelu_correctness(self, dev, sample_inputs):
        """测试 GELU 算子的正确性"""
        x_np = sample_inputs["x"]

        # 参考实现
        def gelu_ref(x):
            return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

        ref_output = gelu_ref(x_np)

        # TVM 实现
        x = relay.var("x", relay.TensorType([128, 256], "float32"))
        y = relay.op.my_gelu(x)
        func = relay.Function([x], y)

        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(tvm.IRModule.from_expr(func), "llvm")
        module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
        module.set_input("x", tvm.nd.array(x_np, dev))
        module.run()
        tvm_output = module.get_output(0).numpy()

        # 比较
        np.testing.assert_allclose(tvm_output, ref_output, rtol=1e-3, atol=1e-3)

    def test_my_gelu_performance(self, dev, sample_inputs):
        """测试 GELU 算子的性能"""
        x = relay.var("x", relay.TensorType([128, 256], "float32"))
        y = relay.op.my_gelu(x)
        func = relay.Function([x], y)

        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(tvm.IRModule.from_expr(func), "llvm")
        module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
        module.set_input("x", tvm.nd.array(sample_inputs["x"], dev))

        # 预热
        for _ in range(10):
            module.run()

        # 测量
        time_f = module.module.time_evaluator("run", dev, number=100, repeat=5)
        result = time_f()

        # 验证性能（示例阈值）
        assert result.mean < 0.001, f"性能过低: {result.mean*1000:.3f} ms"
```

---

## 34.99 文字内容强化：自定义算子 的工程化理解

这一节用于把前文的 API、IR、Pass、Runtime 和部署片段串联为更完整的工程叙事。
很多学习者第一次阅读 TVM 文档时会觉得示例代码很多，但真正上线时仍然不知道如何判断方案是否可靠。
原因在于 TVM 不是单个推理库，而是一条从模型语义到硬件代码的编译链路。
链路越长，越需要把每一步的业务目标、内部机制、适用边界和失败模式说清楚。

### 34.99.1 代码解读的阅读方法

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

- 围绕“Relay/Relax 算子注册、类型关系和属性推断”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“TE/TIR 实现、外部库调用和 BYOC 的边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“新算子验证、调度优化和多后端维护成本”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 PyTorch Extension、TensorRT Plugin、MLIR Dialect 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 34.99.2 业务意义

1. 自定义算子 的业务价值不只是让模型跑得更快，而是让同一个模型可以在不同成本、功耗和延迟约束下交付。
2. 在服务器场景中，核心指标通常是吞吐、P95/P99 延迟、资源利用率和多租户隔离。
3. 在移动端场景中，核心指标通常是首帧时间、持续发热、内存峰值和包体大小。
4. 在嵌入式场景中，核心指标通常是 Flash 占用、静态内存、实时性和掉电恢复能力。
5. 在云端批处理场景中，编译时间可以接受更长，但调优记录和缓存复用变得非常重要。
6. 在在线服务场景中，编译产物需要可回滚、可审计、可灰度，而不能只在开发机上验证。
7. 业务方关心的是 SLA、成本和稳定性，编译器工程师关心的是 IR 正确性、优化空间和后端能力。
8. 优秀的 TVM 项目需要把这两类语言翻译成共同的指标体系。
9. 当优化收益只有少量百分点时，应评估它是否值得引入新的维护复杂度。
10. 当优化收益很大但只在少数输入上成立时，应评估输入分布变化后的风险。

- 围绕“Relay/Relax 算子注册、类型关系和属性推断”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“TE/TIR 实现、外部库调用和 BYOC 的边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“新算子验证、调度优化和多后端维护成本”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 PyTorch Extension、TensorRT Plugin、MLIR Dialect 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 34.99.3 TVM 内部机制

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

- 围绕“Relay/Relax 算子注册、类型关系和属性推断”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“TE/TIR 实现、外部库调用和 BYOC 的边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“新算子验证、调度优化和多后端维护成本”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 PyTorch Extension、TensorRT Plugin、MLIR Dialect 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 34.99.4 适用场景

1. 当模型结构相对稳定、目标硬件明确、性能收益可以通过基准测试确认时，自定义算子 相关技术最容易发挥价值。
2. 当团队需要支持多种硬件后端时，TVM 的统一 IR 和 Target 抽象可以降低重复适配成本。
3. 当模型中存在框架运行时开销、算子融合机会或布局转换冗余时，编译优化通常能带来明显收益。
4. 当部署环境不能依赖完整 Python 栈时，AOT、CRT 或导出后的 runtime artifact 更有意义。
5. 当硬件厂商提供高性能库但模型图需要复杂切分时，BYOC 和外部 codegen 是常见选择。
6. 当输入形状变化频繁时，应提前设计 shape 策略，而不是在上线前才补动态形状支持。
7. 当模型版本迭代频繁时，应把编译、调优、验证和发布纳入 CI/CD。
8. 当业务对精度非常敏感时，应把优化收益和数值回归一起评估。
9. 当系统存在多模型串联时，应评估端到端 pipeline，而不是只优化单个模型。
10. 当部署设备数量很大时，编译产物的一致性和可追踪性比单次实验性能更重要。

- 围绕“Relay/Relax 算子注册、类型关系和属性推断”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“TE/TIR 实现、外部库调用和 BYOC 的边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“新算子验证、调度优化和多后端维护成本”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 PyTorch Extension、TensorRT Plugin、MLIR Dialect 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 34.99.5 限制条件

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

- 围绕“Relay/Relax 算子注册、类型关系和属性推断”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“TE/TIR 实现、外部库调用和 BYOC 的边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“新算子验证、调度优化和多后端维护成本”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 PyTorch Extension、TensorRT Plugin、MLIR Dialect 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 34.99.6 工程经验

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

- 围绕“Relay/Relax 算子注册、类型关系和属性推断”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“TE/TIR 实现、外部库调用和 BYOC 的边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“新算子验证、调度优化和多后端维护成本”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 PyTorch Extension、TensorRT Plugin、MLIR Dialect 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 34.99.7 常见误区

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

- 围绕“Relay/Relax 算子注册、类型关系和属性推断”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“TE/TIR 实现、外部库调用和 BYOC 的边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“新算子验证、调度优化和多后端维护成本”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 PyTorch Extension、TensorRT Plugin、MLIR Dialect 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 34.99.8 生产部署注意事项

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

- 围绕“Relay/Relax 算子注册、类型关系和属性推断”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“TE/TIR 实现、外部库调用和 BYOC 的边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“新算子验证、调度优化和多后端维护成本”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 PyTorch Extension、TensorRT Plugin、MLIR Dialect 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 34.99.9 与同类系统对比

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

- 围绕“Relay/Relax 算子注册、类型关系和属性推断”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“TE/TIR 实现、外部库调用和 BYOC 的边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“新算子验证、调度优化和多后端维护成本”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 PyTorch Extension、TensorRT Plugin、MLIR Dialect 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 34.99.10 章节复盘

1. 回到本章，自定义算子 的关键不是记住所有 API，而是理解为什么这些 API 会出现在编译链路的这个位置。
2. 当你看到一段代码时，应能说出它改变了模型语义、调度空间、内存布局、运行时入口还是部署产物。
3. 当你看到一个性能数字时，应能说出它的测试输入、硬件状态、计时方法和误差范围。
4. 当你看到一个优化 pass 时，应能说出它依赖的前置假设和可能破坏的边界条件。
5. 当你准备上线时，应能说出失败后如何回滚、如何复现、如何定位和如何与业务方沟通影响。
6. 这套思维比单个示例更重要，因为 TVM 的 API 会演进，但编译部署的工程约束长期稳定。
7. 后续学习中，可以把每一章都转化为一张决策表：何时使用、收益来自哪里、风险是什么、如何验证。
8. 只有把代码、机制和工程策略放在一起，TVM 才不只是工具箱，而是可运行的生产系统。
9. 因此，本章新增的文字说明应作为阅读代码段的上下文，而不是替代对原始代码的逐行理解。
10. 如果遇到与示例不一致的实际项目，应优先回到模型约束和目标硬件，而不是机械套用章节流程。

- 围绕“Relay/Relax 算子注册、类型关系和属性推断”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“TE/TIR 实现、外部库调用和 BYOC 的边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“新算子验证、调度优化和多后端维护成本”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“与 PyTorch Extension、TensorRT Plugin、MLIR Dialect 的对比”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。


## 34.16 本章小结

本章全面介绍了 TVM 中自定义算子开发的各个方面：

1. **Relay 算子注册**：定义算子接口、属性、类型推断规则
2. **TE Compute**：使用张量表达式定义计算逻辑
3. **外部函数调用**：集成已有的 C/CUDA 实现
4. **BYOC 框架**：集成第三方编译器和代码生成器
5. **性能调优**：使用 AutoTVM/MetaSchedule 优化自定义算子
6. **最佳实践**：选择合适的方式、测试策略、常见陷阱
7. **TIR 级别算子**：直接使用 TIR 定义和优化算子
8. **序列化与版本管理**：模型导出/加载和算子版本控制
9. **文档与测试**：API 文档生成和完整的测试框架
10. **工程实践**：项目结构、构建系统、测试框架

自定义算子是 TVM 灵活性的重要体现，掌握这些技术可以让 TVM 适应几乎任何计算场景。

<div data-component="CustomOpDevelopmentFlowchart"></div>

> **下一章预告**：第 35 章将介绍 TVM 开发中的调试与错误排查技巧，帮助开发者快速定位和解决问题。

---

## 附录 A：自定义算子开发速查表

### A.1 方式选择决策树

```
需要自定义算子？
├── 已有 C/CUDA 实现？
│   ├── 是 → 使用 Extern Function
│   └── 否 → 继续判断
├── 需要集成第三方库？
│   ├── 是 → 使用 BYOC
│   └── 否 → 继续判断
├── 需要自动调度优化？
│   ├── 是 → 使用 TE Compute
│   └── 否 → 使用 TIR 直接定义
└── 需要极致性能？
    ├── 是 → 手动 TIR + 自定义调度
    └── 否 → TE Compute + AutoSchedule
```

### A.2 常用 API 速查

| 任务 | API | 示例 |
|------|-----|------|
| 注册 Relay 算子 | `TVM_REGISTER_OP` | `TVM_REGISTER_OP("my_op")` |
| 注册 Python 算子 | `@register_custom_op` | `@register_custom_op("my_op")` |
| 定义 TE Compute | `te.compute` | `te.compute(shape, lambda_func)` |
| 调用外部函数 | `tir.call_extern` | `tir.call_extern("float32", "func", args)` |
| 注册 PackedFunc | `@register_func` | `@register_func("func_name")` |
| 注册类型推断 | `.set_attr("FInferStructInfo", func)` | 见算子注册示例 |
| 注册 Compute | `@tvm.ir.register_op_attr` | `@tvm.ir.register_op_attr("op", "FTVMCompute")` |
| 注册调度 | `@tvm.ir.register_op_attr` | `@tvm.ir.register_op_attr("op", "FTVMSchedule")` |

### A.3 文件组织模板



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
my_project/
├── src/
│   ├── ops/
│   │   ├── my_op_relay.cc      # Relay 算子注册
│   │   ├── my_op_te.py          # TE Compute
│   │   └── my_op_tir.cc         # TIR kernel
│   ├── kernels/
│   │   ├── my_kernel_cpu.cc     # CPU kernel
│   │   └── my_kernel_cuda.cu    # CUDA kernel
│   └── byoc/
│       └── my_codegen.cc        # BYOC Codegen
├── tests/
│   ├── test_correctness.py      # 正确性测试
│   ├── test_performance.py      # 性能测试
│   └── test_compatibility.py    # 兼容性测试
└── docs/
    └── api.md                   # API 文档
```

---

## 附录 B：Relay 算子注册完整模板

### B.1 算子定义模板



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// my_op.cc

#include <tvm/relay/op.h>
#include <tvm/relay/type.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace relay {

// 属性定义
struct MyOpAttrs : public tvm::AttrsNode<MyOpAttrs> {
  float alpha;
  bool use_bias;

  TVM_DECLARE_ATTRS(MyOpAttrs, "relay.attrs.MyOpAttrs") {
    TVM_ATTR_FIELD(alpha).set_default(1.0f)
      .describe("Scaling factor");
    TVM_ATTR_FIELD(use_bias).set_default(false)
      .describe("Whether to use bias");
  }
};

TVM_REGISTER_NODE_TYPE(MyOpAttrs);

// 类型推断
bool MyOpRel(const Array<Type>& types, int num_inputs,
             const Attrs& attrs, const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 3);  // input + output

  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  const auto* weight = types[1].as<TensorTypeNode>();
  if (weight == nullptr) return false;

  // 验证维度
  ICHECK_EQ(data->shape.size(), 2);
  ICHECK_EQ(weight->shape.size(), 2);

  // 输出形状
  Array<IndexExpr> oshape = {data->shape[0], weight->shape[1]};
  reporter->Assign(types[2], TensorType(oshape, data->dtype));
  return true;
}

// 注册算子
TVM_REGISTER_OP("my_op")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "Input data")
    .add_argument("weight", "Tensor", "Weight matrix")
    .set_attrs_type<MyOpAttrs>()
    .set_support_level(10)
    .add_type_rel("MyOp", MyOpRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

}  // namespace relay
}  // namespace tvm
```

### B.2 Python API 模板



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# python/tvm/relay/op/my_op.py

from tvm.relay import Expr
from tvm._ffi import register_func

def my_op(data: Expr, weight: Expr, alpha: float = 1.0,
          use_bias: bool = False) -> Expr:
    """自定义算子

    Parameters
    ----------
    data : Expr
        输入张量，形状 (M, K)
    weight : Expr
        权重矩阵，形状 (K, N)
    alpha : float
        缩放因子
    use_bias : bool
        是否使用偏置

    Returns
    -------
    result : Expr
        输出张量，形状 (M, N)
    """
    from tvm.relay.op import _make
    return _make.my_op(data, weight, alpha, use_bias)
```

---

## 附录 C：BYOC 开发模板

### C.1 自定义 Codegen 模板



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// my_codegen.h

#include "src/relay/backend/contrib/codegen_json/codegen_json.h"

namespace tvm {
namespace relay {
namespace contrib {

class MyCodegen : public JSONSerializer {
 public:
  MyCodegen() : JSONSerializer("my_codegen") {}

  std::string JIT(const JSONGraphNode& node) override {
    std::stringstream ss;
    ss << "my_backend::" << node.GetOp();
    ss << "(";
    for (size_t i = 0; i < node.GetInputs().size(); i++) {
      if (i > 0) ss << ", ";
      ss << "%" << i;
    }
    ss << ")";
    return ss.str();
  }

 protected:
  std::string GetUniqueName(const std::string& prefix) {
    auto it = name_map_.find(prefix);
    if (it != name_map_.end()) {
      return it->second;
    }
    std::string unique = prefix + "_" + std::to_string(name_map_.size());
    name_map_[prefix] = unique;
    return unique;
  }

 private:
  std::unordered_map<std::string, std::string> name_map_;
};

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
```

### C.2 注册 Codegen



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// my_codegen.cc

#include "my_codegen.h"

namespace tvm {
namespace relay {
namespace contrib {

// 注册 pattern 函数
Array<Pattern> MyCodegenPatterns() {
  Array<Pattern> patterns;
  // 注册可处理的算子模式
  patterns.push_back(Pattern("my_pattern",
    OpPattern::kOpaque,
    {Op::Get("nn.conv2d"), Op::Get("nn.relu")}));
  return patterns;
}

// 注册 codegen
TVM_REGISTER_GLOBAL("relay.ext.my_codegen")
.set_body_typed([](const IRModule& mod, const String& code) {
  MyCodegen codegen;
  return codegen.Run(mod, code);
});

// 注册 pattern
TVM_REGISTER_GLOBAL("relay.ext.my_codegen.patterns")
.set_body_typed(MyCodegenPatterns);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
```

---

## 附录 D：自定义算子相关源码索引

| 功能 | 源码文件 | 关键函数/类 |
|------|---------|------------|
| Relay Op 注册 | `include/tvm/relay/op.h` | `OpRegistry` |
| 属性定义 | `include/tvm/relay/attrs/` | 各种 `Attrs` |
| 类型推断 | `src/relay/ir/type_functor.cc` | `TypeReporter` |
| TE Compute | `python/tvm/te/` | `te.compute` |
| TE 调度原语 | `python/tvm/te/operation.py` | `split`, `reorder` 等 |
| TIR 定义 | `include/tvm/tir/` | `PrimFunc`, `Buffer` |
| 外部函数注册 | `src/runtime/registry.cc` | `TVM_REGISTER_GLOBAL` |
| PackedFunc | `include/tvm/runtime/packed_func.h` | `PackedFunc` |
| BYOC 框架 | `src/relay/backend/contrib/` | `JSONSerializer` |
| AutoTVM 模板 | `python/tvm/auto_scheduler/` | `register_workload` |
| MetaSchedule | `python/tvm/meta_schedule/` | `tune_tir` |
| PyTorch 算子映射 | `python/tvm/relax/frontend/pytorch_translator.py` | `PyTorchImporter` |
| ONNX 算子映射 | `python/tvm/relax/frontend/onnx_translator.py` | `OnnxImporter` |
| CUTLASS BYOC | `src/relay/backend/contrib/cutlass/` | `CutlassCodegen` |
| TensorRT BYOC | `src/relay/backend/contrib/tensorrt/` | `TensorRTCodegen` |
| cuBLAS BYOC | `src/relay/backend/contrib/cublas/` | `CuBLASCodegen` |

---

## 34.17 Relay 算子注册完整流程深度解析

### 34.17.1 算子注册的完整生命周期

一个 Relay 算子从 C++ 定义到 Python 可用需要经历以下步骤：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
算子注册完整生命周期：

第一步：定义属性结构体（Attrs）
    ↓ TVM_DECLARE_ATTRS + TVM_REGISTER_NODE_TYPE
第二步：定义类型推断函数（TypeRel）
    ↓ 定义 bool XxxRel(types, num_inputs, attrs, reporter)
第三步：定义计算逻辑引用（FTVMCompute）
    ↓ 定义 Array<te::Tensor> XxxCompute(attrs, inputs, out_type)
第四步：定义调度引用（FTVMSchedule）
    ↓ 定义 Schedule XxxSchedule(attrs, outs, target)
第五步：注册算子到 OpRegistry
    ↓ TVM_REGISTER_OP("xxx")
第六步：创建 Python API
    ↓ def xxx(...) -> relay.Expr
第七步：使用算子
    ↓ y = relay.op.xxx(x, ...)
```

### 34.17.2 OpRegistry 单例机制

所有 Relay 算子都注册在一个全局单例注册表中，定义在 `include/tvm/relay/op.h`：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// include/tvm/relay/op.h — OpRegistry 核心定义
class OpRegistry {
 public:
  // 获取或创建指定名称的算子注册表项
  // 如果已存在则返回现有项，否则创建新项
  static OpRegistry& Register(const String& op_name, int level) {
    // 从全局注册表中查找
    auto& reg = GlobalOpRegistry()->Register(op_name, level);
    return reg;
  }

  // 设置算子的输入数量
  OpRegistry& set_num_inputs(int n) {
    get_mutable()->num_inputs = n;
    return *this;
  }

  // 添加参数描述（用于文档生成）
  OpRegistry& add_argument(const String& name, const String& type_info,
                           const String& description) {
    get_mutable()->arguments.push_back({name, type_info, description});
    return *this;
  }

  // 设置属性类型
  template<typename AttrType>
  OpRegistry& set_attrs_type() {
    get_mutable()->attrs_type_key = AttrType::_type_key;
    return *this;
  }

  // 注册类型推断函数
  OpRegistry& add_type_rel(const String& rel_name, FTVMRelayTypeRel func) {
    get_mutable()->type_rel = func;
    get_mutable()->type_rel_name = rel_name;
    return *this;
  }

  // 注册任意属性（如 FInferStructInfo, TOpPattern 等）
  template<typename F>
  OpRegistry& set_attr(const String& attr_name, F func, int plevel) {
    get_mutable()->attrs.Set(attr_name, TypedPackedFunc<F>(func), plevel);
    return *this;
  }

  // 完成注册（不可逆操作）
  void finalize();

 private:
  // 内部存储
  std::shared_ptr<OpNode> data_;
};

// 全局注册表单例
class GlobalOpRegistry {
 public:
  static GlobalOpRegistry* Global() {
    static GlobalOpRegistry* inst = new GlobalOpRegistry();
    return inst;
  }

  // 按名称查找算子
  static const Op& Get(const String& name) {
    return Global()->GetOp(name);
  }

 private:
  std::unordered_map<String, Op> op_map_;
};
```

### 34.17.3 FInferStructInfo vs FTVMRelayTypeRel

Relay 和 Relax 使用不同的类型推断机制：

| 特性 | FTVMRelayTypeRel（Relay） | FInferStructInfo（Relax） |
|------|--------------------------|--------------------------|
| **输入** | `Array<Type>`, `Attrs` | `Call`, `BlockBuilder` |
| **输出** | 填充 `TypeReporter` | 返回 `StructInfo` |
| **形状表示** | `IndexExpr` 或 `Any` | `PrimExpr`（可含符号变量） |
| **约束传播** | 有限 | 通过 `BlockBuilder` 报告 |
| **源码位置** | `src/relay/ir/type_functor.cc` | `src/relax/op/op_common.h` |
| **错误报告** | 返回 `false` | 调用 `ctx->ReportFatal()` |



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// ====== Relay 类型推断（FTVMRelayTypeRel） ======
// src/relay/op/type_relations.cc
bool MatmulRel(const Array<Type>& types, int num_inputs,
               const Attrs& attrs, const TypeReporter& reporter) {
  // types[0] = 输入 A 的类型
  // types[1] = 输入 B 的类型
  // types[2] = 输出类型（待填充）
  ICHECK_EQ(types.size(), 3);

  const auto* A = types[0].as<TensorTypeNode>();
  const auto* B = types[1].as<TensorTypeNode>();
  if (!A || !B) return false;

  // 验证收缩维度
  // A 的最后一维必须等于 B 的倒数第二维
  if (!reporter->Equal(A->shape.back(), B->shape[B->shape.size() - 2])) {
    return false;
  }

  // 构造输出形状
  Array<IndexExpr> oshape;
  for (size_t i = 0; i < A->shape.size() - 1; i++) {
    oshape.push_back(A->shape[i]);
  }
  oshape.push_back(B->shape.back());

  // 填充输出类型
  reporter->Assign(types[2], TensorType(oshape, A->dtype));
  return true;
}

// ====== Relax StructInfo 推断（FInferStructInfo） ======
// src/relax/op/tensor/linear_algebra.cc
StructInfo InferStructInfoMatmul(const Call& call, const BlockBuilder& ctx) {
  // 获取输入的 StructInfo
  TensorStructInfo lhs = GetStructInfoAs<TensorStructInfo>(call->args[0]);
  TensorStructInfo rhs = GetStructInfoAs<TensorStructInfo>(call->args[1]);

  // 验证收缩维度（使用符号统一）
  CheckContractionDimMatch(lhs, rhs, ctx);

  // 构造输出形状（支持符号变量）
  Array<PrimExpr> out_shape;
  for (int i = 0; i < lhs->ndim - 1; i++) {
    out_shape.push_back(lhs->shape.as<ShapeExprNode>()->values[i]);
  }
  out_shape.push_back(rhs->shape.as<ShapeExprNode>()->values.back());

  return TensorStructInfo(ShapeExpr(out_shape), lhs->dtype);
}
```

### 34.17.4 完整示例：注册 rms_norm 算子



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// ====== 第一步：定义属性 ======
// src/relay/attrs/nn.h

struct RMSNormAttrs : public tvm::AttrsNode<RMSNormAttrs> {
  float epsilon;      // 数值稳定性常数
  int axis;           // 归一化的轴

  TVM_DECLARE_ATTRS(RMSNormAttrs, "relay.attrs.RMSNormAttrs") {
    TVM_ATTR_FIELD(epsilon)
        .set_default(1e-6f)
        .describe("Small constant for numerical stability");
    TVM_ATTR_FIELD(axis)
        .set_default(-1)
        .describe("Axis along which to normalize");
  }
};

TVM_REGISTER_NODE_TYPE(RMSNormAttrs);

// ====== 第二步：类型推断函数 ======
// src/relay/op/nn/nn.cc

bool RMSNormRel(const Array<Type>& types, int num_inputs,
                const Attrs& attrs, const TypeReporter& reporter) {
  // types[0] = 输入 x, types[1] = weight, types[2] = 输出
  ICHECK_EQ(types.size(), 3);

  const auto* x = types[0].as<TensorTypeNode>();
  const auto* w = types[1].as<TensorTypeNode>();
  if (!x || !w) return false;

  // weight 必须是 1D，且长度等于归一化维度
  ICHECK_EQ(w->shape.size(), 1);
  reporter->Assign(types[2], TensorType(x->shape, x->dtype));
  return true;
}

// ====== 第三步：注册算子 ======
TVM_REGISTER_OP("nn.rms_norm")
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "Input tensor")
    .add_argument("weight", "Tensor", "Scale parameter")
    .set_attrs_type<RMSNormAttrs>()
    .set_support_level(1)
    .add_type_rel("RMSNorm", RMSNormRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque)
    .set_attr<FInferStructInfo>("FInferStructInfo",
                                 InferStructInfoRMSNorm);

// ====== 第四步：TE Compute 实现 ======
// python/tvm/topi/nn/rms_norm.py

def rms_norm_compute(x, weight, epsilon, axis):
    """RMS Norm 的 TE 实现"""
    # 计算 RMS: sqrt(mean(x^2) + epsilon)
    x_squared = te.compute(x.shape, lambda *i: x(*i) * x(*i), name="x_sq")
    rms = te.compute(
        (x.shape[0], x.shape[1], 1),  # 保持归一化维度为 1
        lambda b, s, d: te.sum(x_squared[b, s, k], axis=k) / x.shape[-1],
        name="rms_mean"
    )
    rms_safe = te.compute(
        rms.shape,
        lambda *i: te.sqrt(rms(*i) + te.const(epsilon, "float32")),
        name="rms_safe"
    )
    # 归一化并缩放
    normalized = te.compute(
        x.shape,
        lambda b, s, d: x[b, s, d] / rms_safe[b, s, 0] * weight[d],
        name="rms_normed"
    )
    return normalized
```

---

## 34.18 BYOC 外部代码生成接口详解

### 34.18.1 BYOC 完整流程

BYOC（Bring Your Own Codegen）的编译流程分为四个阶段：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
BYOC 编译流程：

原始 Relay 模块
    ↓ 第一阶段：MergeComposite（模式匹配）
    将连续的算子序列匹配为预定义的复合模式
    例如：conv2d + bias + relu → "conv2d_bias_relu" 模式
    ↓ 第二阶段：AnnotateTarget（标记目标）
    为匹配的子图标记目标 codegen
    例如：标记为 "tensorrt" 或 "cutlass"
    ↓ 第三阶段：PartitionGraph（图分区）
    将标记的子图分区为独立的函数
    未标记的节点保留在 TVM 原生编译路径中
    ↓ 第四阶段：RunCodegen（代码生成）
    调用目标 codegen 为每个分区生成代码
    生成运行时可调用的外部函数
    ↓
    混合执行模块（部分 TVM 原生 + 部分外部）
```

### 34.18.2 JSONSerializer 接口

BYOC codegen 通常继承 `JSONSerializer`，它提供了通用的图序列化功能：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// src/relay/backend/contrib/codegen_json/codegen_json.h

class JSONSerializer : public MemoizedExprTranslator<JSONGraphNodeEntry> {
 public:
  explicit JSONSerializer(const String& symbol) : symbol_(symbol) {}

  // 主入口：序列化整个函数
  String Run(const IRModule& mod, const String& code) {
    // 遍历模块中的所有函数
    for (const auto& [gv, func] : mod->functions) {
      if (auto* f = func.as<FunctionNode>()) {
        // 序列化函数体
        VisitExpr(GetRef<Function>(f));
      }
    }
    // 调用 JIT 生成最终代码
    return JIT();
  }

 protected:
  // 序列化 Call 节点（子类通常重写此方法）
  JSONGraphNodeEntry VisitExpr_(const CallNode* call) override {
    // 1. 获取算子名称
    auto op_node = call->op.as<OpNode>();
    std::string op_name = op_node->name;

    // 2. 创建 JSON 图节点
    auto node = std::make_shared<JSONGraphNode>(
        /* name_ = */ op_name,
        /* op_type_ = */ op_name);

    // 3. 添加输入边
    for (const auto& arg : call->args) {
      auto entry = VisitExpr(arg);
      node->AddInput(entry.id, entry.index);
    }

    // 4. 序列化属性
    if (call->attrs.defined()) {
      SerializeNodeAttrs(call->attrs, node);
    }

    // 5. 记录输出形状
    auto struct_info = GetStructInfo(GetRef<Call>(call));
    node->SetShape(GetShape(struct_info));
    node->SetDType(GetDType(struct_info));

    return AddNode(node, GetRef<Call>(call));
  }

  // 子类必须实现：为 JSON 节点生成目标代码
  virtual String JIT(const JSONGraphNode& node) = 0;

 private:
  String symbol_;
  std::vector<std::shared_ptr<JSONGraphNode>> nodes_;
};
```

### 34.18.3 内置 BYOC 后端对比

| 后端 | 源码位置 | 目标硬件 | 支持的算子 | 精度 | 特点 |
|------|---------|---------|-----------|------|------|
| **TensorRT** | `contrib/tensorrt/` | NVIDIA GPU | conv, matmul, bn, pool, act | FP16/INT8 | 自动 kernel 选择 |
| **CUTLASS** | `contrib/cutlass/` | NVIDIA GPU | matmul, conv2d | FP16/FP32 | 模板化 GEMM |
| **cuBLAS** | `contrib/cublas/` | NVIDIA GPU | matmul, batch_matmul | FP16/FP32 | 成熟的 BLAS 库 |
| **DNNL** | `contrib/dnnl/` | x86 CPU | conv, matmul, bn, pool | FP32/INT8 | Intel 优化库 |
| **Vitis AI** | `contrib/vitis_ai/` | Xilinx FPGA/DPUCZD | 量化算子 | INT8 | FPGA 加速 |
| **CMSIS-NN** | `contrib/cmsis_nn/` | ARM Cortex-M | conv, dw_conv, pool | INT8 | 嵌入式推理 |

### 34.18.4 自定义 BYOC 后端开发指南



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# ====== 第一步：定义模式匹配规则 ======
# python/tvm/relay/op/contrib/my_backend.py

from tvm import relay
from tvm.relay import transform

def partition_for_my_backend(mod):
    """将可处理的子图分区为 MyBackend 函数"""

    # 定义可处理的算子模式
    conv2d_pattern = relay.op.get("nn.conv2d")
    relu_pattern = relay.op.get("nn.relu")
    add_pattern = relay.op.get("add")

    # 合并连续的算子为复合模式
    # 例如：conv2d + add + relu → "my_backend.conv2d_bias_relu"
    patterns = [
        ("my_backend.conv2d_bias_relu",
         relay.pattern.is_op("nn.relu")(
             relay.pattern.is_op("add")(
                 relay.pattern.is_op("nn.conv2d")(
                     relay.pattern.wildcard(),
                     relay.pattern.wildcard()
                 ),
                 relay.pattern.wildcard()
             )
         )),
    ]

    # 应用模式匹配
    seq = tvm.transform.Sequential([
        transform.MergeComposite(patterns),
        transform.AnnotateTarget("my_backend"),
        transform.PartitionGraph(),
    ])

    return seq(mod)

# ====== 第二步：实现 Codegen ======
# src/relay/backend/contrib/my_backend/codegen.cc

# ====== 第三步：实现 Runtime ======
# src/relay/backend/contrib/my_backend/runtime.cc

# ====== 第四步：注册 ======
TVM_REGISTER_GLOBAL("relay.ext.my_backend")
.set_body_typed([](const IRModule& mod, const String& code) {
    MyBackendCodegen codegen;
    return codegen.Run(mod, code);
});
```

### 34.18.5 运行时外部函数链接

BYOC 生成的外部函数在运行时通过 PackedFunc 注册表链接：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 运行时加载流程：
#
# 1. 编译阶段：
#    - BYOC Codegen 将子图编译为目标代码（如 TensorRT engine）
#    - 代码打包为 shared library 或序列化为文件
#
# 2. 加载阶段：
#    - 加载 shared library
#    - 注册 PackedFunc 到全局注册表
#    - 函数名格式：symbol_name + "_" + subgraph_id
#
# 3. 执行阶段：
#    - VM 遇到外部函数调用
#    - 通过 PackedFunc 注册表查找并调用
#    - 数据通过 DLTensor 传递（零拷贝）

# Python 端的加载示例
import tvm

# 加载包含 BYOC 代码的模块
lib = tvm.runtime.load_module("model_with_byoc.so")

# 外部函数会自动注册
# 可以通过名称查找
my_func = lib["my_backend_subgraph_0"]

# 调用
output = my_func(input_data)
```

---

## 34.19 自定义算子的调试与错误排查

### 34.19.1 常见错误类型与解决方案

| 错误类型 | 错误信息 | 原因 | 解决方案 |
|---------|---------|------|---------|
| **未注册算子** | `Op not registered: my_op` | 算子未注册到 OpRegistry | 检查 `TVM_REGISTER_OP` 是否执行 |
| **类型推断失败** | `Type inference failed` | TypeRel 函数返回 false | 检查类型推断函数的约束验证 |
| **形状不匹配** | `Shape mismatch` | TE Compute 中的形状推断错误 | 使用 `te.placeholder` 明确指定形状 |
| **数据类型错误** | `dtype mismatch` | 输入输出类型不一致 | 使用 `tvm.tir.Cast` 显式转换 |
| **空指针** | `Segmentation fault` | 外部函数中的内存访问错误 | 检查 Buffer 边界和对齐 |
| **属性缺失** | `Attr not found` | 使用了未定义的属性 | 检查 `TVM_DECLARE_ATTRS` |
| **调度错误** | `Schedule not found` | 缺少对应的调度实现 | 注册 `FTVMSchedule` |
| **CUDA 错误** | `CUDA error: invalid argument` | CUDA kernel 参数错误 | 检查 kernel 启动配置 |

### 34.19.2 TE Compute 调试技巧



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 调试 TE Compute 的常用方法

# 方法 1：打印中间张量
def debug_te_compute(A):
    """在 TE 计算中插入调试打印"""
    # 使用 te.compute 创建中间结果
    intermediate = te.compute(
        A.shape,
        lambda *i: A(*i) * 2,
        name="debug_intermediate"  # 有意义的名称有助于调试
    )

    result = te.compute(
        A.shape,
        lambda *i: intermediate(*i) + 1,
        name="result"
    )

    return result

# 方法 2：使用 Python 参考实现对比
def verify_te_compute(te_func, ref_func, input_shapes):
    """验证 TE 计算的正确性"""
    import numpy as np
    import tvm

    # 创建输入数据
    inputs_np = [np.random.randn(*s).astype("float32") for s in input_shapes]

    # TE 实现
    te_inputs = [te.placeholder(s, dtype="float32") for s in input_shapes]
    te_output = te_func(*te_inputs)
    s = te.create_schedule(te_output.op)
    func = tvm.build(s, te_inputs + [te_output], target="llvm")

    tvm_inputs = [tvm.nd.array(x) for x in inputs_np]
    tvm_output = tvm.nd.array(np.zeros(te_output.shape, dtype="float32"))
    func(*tvm_inputs, tvm_output)

    # 参考实现
    ref_output = ref_func(*inputs_np)

    # 比较
    np.testing.assert_allclose(
        tvm_output.numpy(), ref_output,
        rtol=1e-4, atol=1e-4,
        err_msg="TE 计算结果与参考实现不匹配"
    )
    print("✓ TE 计算验证通过")

# 方法 3：使用 tvm.te 编译时打印
def print_te_schedule(s, output):
    """打印 TE 调度的循环结构"""
    print("调度结构：")
    print(tvm.lower(s, [output.op.input_tensors[0], output], simple_mode=True))
```

### 34.19.3 TIR 级别调试



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# TIR 调试工具

# 方法 1：检查 TIR 的 SSA 属性
from tvm import tir

def verify_tir_ssa(func: tir.PrimFunc):
    """验证 TIR 函数是否满足 SSA（静态单赋值）形式"""
    from tvm.tir.analysis import verify_ssa
    try:
        verify_ssa(func)
        print("✓ TIR 函数满足 SSA 形式")
    except Exception as e:
        print(f"✗ SSA 验证失败: {e}")

# 方法 2：检查 Buffer 访问边界
def check_buffer_bounds(func: tir.PrimFunc):
    """检查 TIR 函数中的 Buffer 访问是否越界"""
    from tvm.tir import analysis

    # 收集所有 Buffer Store 和 Load 操作
    stmt = func.body
    errors = []

    class BufferAccessChecker(tir.PrimExprVisitor):
        def visit_buffer_load(self, op):
            # 检查索引是否在 Buffer 范围内
            buf = op.buffer
            for i, (idx, shape) in enumerate(zip(op.indices, buf.shape)):
                # 简化检查：常量索引是否越界
                if isinstance(idx, tir.IntImm) and isinstance(shape, tir.IntImm):
                    if idx.value < 0 or idx.value >= shape.value:
                        errors.append(
                            f"Buffer '{buf.name}' 索引 [{i}]={idx.value} "
                            f"越界（形状 {shape.value}）"
                        )

    checker = BufferAccessChecker()
    checker.visit_stmt(stmt)

    if errors:
        print(f"发现 {len(errors)} 个边界错误：")
        for err in errors:
            print(f"  - {err}")
    else:
        print("✓ 未发现明显的越界访问")

# 方法 3：打印 TIR 的 TVMScript 表示
def print_tir(func: tir.PrimFunc):
    """以 TVMScript 格式打印 TIR 函数"""
    from tvm import script as tvm_script
    print(tvm_script.asscript(func))
```

### 34.19.4 使用 GDB/LLDB 调试 TVM C++ 代码



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```bash
# ====== GDB 调试 TVM 的步骤 ======

# 第一步：编译 Debug 版本的 TVM
cmake .. -DCMAKE_BUILD_TYPE=Debug \
         -DCMAKE_CXX_FLAGS="-g -O0" \
         -DUSE_LLVM=/path/to/llvm-config
make -j$(nproc)

# 第二步：在 GDB 中运行 Python 脚本
gdb --args python my_custom_op_test.py

# 第三步：设置断点
(gdb) break tvm::relay::MyCustomOp    # 在自定义算子处断点
(gdb) break tir::Evaluate             # 在 TIR 求值处断点

# 第四步：运行
(gdb) run

# 第五步：当断点命中时，检查变量
(gdb) print call->args[0]            # 查看 Call 的第一个参数
(gdb) print struct_info               # 查看 StructInfo
(gdb) bt                              # 查看调用栈

# ====== LLDB 调试（macOS） ======
lldb -- python my_custom_op_test.py
(lldb) breakpoint set --name MyCustomOp
(lldb) run
(lldb) frame variable                  # 查看当前帧的变量
(lldb) thread backtrace                # 查看调用栈
```

### 34.19.5 调试策略决策表

| 问题现象 | 首选调试方法 | 工具 | 说明 |
|---------|------------|------|------|
| 算子输出形状错误 | 检查 StructInfo | `print(sinfo)` | 验证推断规则 |
| 算子输出值错误 | TE 参考实现对比 | `np.testing.assert_allclose` | 逐步定位错误计算 |
| 编译失败 | 检查 IR 合法性 | `relax.analysis.well_formed()` | 验证 IR 结构 |
| 运行时崩溃 | GDB/LLDB 断点 | `gdb --args python ...` | 检查内存访问 |
| 性能异常 | Profiler 分析 | `time_evaluator` | 逐层计时 |
| CUDA kernel 错误 | CUDA 调试 | `cuda-gdb`, `compute-sanitizer` | 检查 GPU 内存 |
| 调度错误 | 打印 TIR | `tvm.lower(s, ...)` | 查看循环结构 |
| 属性解析错误 | 检查 Attrs 定义 | `print(call.attrs)` | 验证属性字段 |

### 34.19.6 完整调试工作流



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 完整的自定义算子调试工作流
def debug_custom_op_workflow():
    """自定义算子的系统化调试流程"""

    # ====== 第 1 步：验证算子注册 ======
    print("第 1 步：验证算子注册")
    try:
        op = relay.op.get("my_custom_op")
        print(f"  ✓ 算子已注册: {op.name}")
        print(f"    输入数量: {op.get_attr('num_inputs')}")
    except Exception as e:
        print(f"  ✗ 算子未注册: {e}")
        return

    # ====== 第 2 步：验证类型推断 ======
    print("\n第 2 步：验证类型推断")
    x = relay.var("x", relay.TensorType([128, 256], "float32"))
    try:
        y = relay.op.my_custom_op(x)
        print(f"  ✓ 类型推断成功")
        print(f"    输出类型: {y.checked_type}")
    except Exception as e:
        print(f"  ✗ 类型推断失败: {e}")
        return

    # ====== 第 3 步：验证 TE Compute ======
    print("\n第 3 步：验证 TE Compute")
    try:
        verify_te_compute(
            te_func=my_custom_op_compute,
            ref_func=my_custom_op_reference,
            input_shapes=[(128, 256)]
        )
    except AssertionError as e:
        print(f"  ✗ TE Compute 验证失败: {e}")
        return

    # ====== 第 4 步：验证编译 ======
    print("\n第 4 步：验证编译")
    try:
        func = relay.Function([x], y)
        mod = tvm.IRModule.from_expr(func)
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, "llvm")
        print("  ✓ 编译成功")
    except Exception as e:
        print(f"  ✗ 编译失败: {e}")
        return

    # ====== 第 5 步：验证执行 ======
    print("\n第 5 步：验证执行")
    try:
        dev = tvm.cpu(0)
        module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
        x_np = np.random.randn(128, 256).astype("float32")
        module.set_input("x", tvm.nd.array(x_np))
        module.run()
        output = module.get_output(0).numpy()
        print(f"  ✓ 执行成功")
        print(f"    输出形状: {output.shape}")
        print(f"    输出范围: [{output.min():.4f}, {output.max():.4f}]")
    except Exception as e:
        print(f"  ✗ 执行失败: {e}")
        return

    # ====== 第 6 步：性能基准 ======
    print("\n第 6 步：性能基准")
    time_f = module.module.time_evaluator("run", dev, number=100, repeat=5)
    result = time_f()
    print(f"  平均延迟: {result.mean * 1000:.3f} ms")
    print(f"  标准差: {result.results.std() * 1000:.3f} ms")

    print("\n✓ 所有验证通过！")
```


**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。

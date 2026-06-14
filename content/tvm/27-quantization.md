> **学习目标**：
> - 理解量化（Quantization）的数学原理与精度-性能权衡
> - 掌握 TVM 的 QNN Dialect 设计与量化算子注册机制
> - 理解校准（Calibration）流程与量化参数计算方法
> - 掌握 INT8/FP16 低精度推理的端到端编译流程
> - 了解量化感知训练（QAT）在 TVM 中的支持方案

---

## 27.1 量化基础

### 27.1.1 为什么需要量化？

深度学习模型通常以 FP32（32 位浮点）训练，但推理阶段存在三个关键约束：

| 约束 | FP32 | INT8 | 加速比 |
|------|------|------|--------|
| **内存占用** | 4 bytes/elem | 1 byte/elem | 4× |
| **计算吞吐** | 1× | 2-8× | 2-8× |
| **内存带宽** | 1× | 4× | 4× |
| **功耗** | 1× | ~0.1× | ~10× |

对于 ResNet-50 模型：
- FP32 参数量：~100MB
- INT8 参数量：~25MB
- 推理速度：INT8 通常比 FP32 快 2-4×

### 27.1.2 量化数学原理

**线性量化（Affine Quantization）** 将浮点值映射到整数：

$$q = \text{round}\left(\frac{x}{s}\right) + z$$

其中：
- $x$ 是浮点值
- $q$ 是量化后的整数值
- $s$ 是缩放因子（scale）
- $z$ 是零点（zero point）

**反量化**（从整数恢复浮点）：

$$\hat{x} = s \cdot (q - z)$$

**量化参数的确定**：

$$s = \frac{x_{\max} - x_{\min}}{q_{\max} - q_{\min}}$$

$$z = \text{round}\left(\frac{-x_{\min}}{s}\right) + q_{\min}$$

对于 INT8 量化：
- $q_{\min} = -128$, $q_{\max} = 127$（有符号）或 $q_{\min} = 0$, $q_{\max} = 255$（无符号）

```
浮点域                    整数域
-1.0 ─────────────────── -128
     │                    │
     │    scale = 0.01    │
     │                    │
 0.0 ───────────────────  +z
     │                    │
     │                    │
+1.0 ─────────────────── +127
```

<div data-component="QuantizationMappingChart"></div>

### 27.1.3 对称量化与非对称量化

**对称量化**（Symmetric）假设 $z = 0$：

$$q = \text{round}\left(\frac{x}{s}\right), \quad s = \frac{\max(|x_{\min}|, |x_{\max}|)}{127}$$

**非对称量化**（Asymmetric）使用完整的 $z$ 参数：

$$q = \text{round}\left(\frac{x}{s}\right) + z, \quad s = \frac{x_{\max} - x_{\min}}{255}$$

| 特性 | 对称量化 | 非对称量化 |
|------|---------|-----------|
| **零点** | $z = 0$ | $z \neq 0$ |
| **范围利用** | 对称分布更优 | 任意分布均可 |
| **计算开销** | 较低（无零点偏移） | 较高 |
| **典型应用** | 权重、激活（ReLU 后） | 激活（含负值） |

### 27.1.4 量化粒度

| 粒度 | 说明 | 参数量 | 精度 |
|------|------|--------|------|
| **Per-tensor** | 整个张量共享一组 $s, z$ | 1 scale + 1 zero_point | 低 |
| **Per-channel** | 每个输出通道一组参数 | C scales + C zero_points | 中高 |
| **Per-group** | 每组元素一组参数 | 更多参数 | 最高 |

```python
# 量化粒度示例
import numpy as np

# Per-tensor 量化
def per_tensor_quantize(x, num_bits=8):
    q_min, q_max = -128, 127
    scale = (x.max() - x.min()) / (q_max - q_min)
    zero_point = np.round(-x.min() / scale + q_min).astype(int)
    q = np.clip(np.round(x / scale + zero_point), q_min, q_max).astype(np.int8)
    return q, scale, zero_point

# Per-channel 量化（沿 axis 0，即输出通道维度）
def per_channel_quantize(x, num_bits=8, axis=0):
    q_min, q_max = -128, 127
    # 对每个通道计算独立的 scale
    channel_max = np.max(np.abs(x), axis=tuple(range(1, x.ndim)), keepdims=True)
    scales = channel_max / 127.0
    q = np.clip(np.round(x / scales), q_min, q_max).astype(np.int8)
    return q, scales
```

---

## 27.2 TVM 量化框架概览

### 27.2.1 量化框架架构

TVM 的量化框架由三个核心组件组成：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
┌───────────────────────────────────────────────┐
│                量化框架架构                      │
├───────────────────────────────────────────────┤
│  QNN Dialect                                  │
│  ├── qnn.conv2d, qnn.dense, qnn.add          │
│  ├── qnn.quantize, qnn.dequantize            │
│  └── 量化参数传播与融合                         │
├───────────────────────────────────────────────┤
│  校准（Calibration）                           │
│  ├── 统计收集（MinMax / KL / Percentile）      │
│  ├── 量化参数计算                              │
│  └── 校准数据集管理                             │
├───────────────────────────────────────────────┤
│  量化 Pass                                     │
│  ├── QnnCanonicalize: QNN → 标准算子           │
│  ├── Realize: 插入量化/反量化节点               │
│  └── 常量折叠: 预计算量化权重                    │
└───────────────────────────────────────────────┘
```

**源码目录**：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
src/relay/qnn/
├── op/                      # QNN 算子实现
│   ├── conv2d.cc            # qnn.conv2d
│   ├── dense.cc             # qnn.dense
│   ├── add.cc               # qnn.add
│   ├── mul.cc               # qnn.mul
│   ├── concatenate.cc       # qnn.concatenate
│   ├── requantize.cc        # 重量化
│   ├── dequantize.cc        # 反量化
│   └── quantize.cc          # 量化
└── transform/               # QNN 变换 Pass
    ├── canonicalize.cc      # QNN → 标准算子转换
    ├── combine_parallel.cc  # 并行 QNN 融合
    └── flatten.cc           # QNN 表达式扁平化

python/tvm/relay/quantize/
├── __init__.py              # 量化入口
├── quantize.py              # 量化配置与流程
├── calibration.py           # 校准算法
├── _annotate.py             # 算子注解
├── _realize.py              # 量化实现
└── _partition.py            # 量化区域划分
```

### 27.2.2 QNN Dialect 算子列表

QNN Dialect 为每个需要量化的 Relay 算子提供对应的量化版本：

| QNN 算子 | 对应 Relay 算子 | 量化参数 |
|----------|----------------|---------|
| `qnn.conv2d` | `nn.conv2d` | input_scale, input_zero_point, kernel_scale, kernel_zero_point |
| `qnn.dense` | `nn.dense` | 同上 |
| `qnn.add` | `add` | lhs_scale, lhs_zp, rhs_scale, rhs_zp |
| `qnn.mul` | `mul` | 同上 |
| `qnn.concatenate` | `concatenate` | 每个输入的 scale 和 zero_point |
| `qnn.requantize` | — | input_scale/zp → output_scale/zp |
| `qnn.dequantize` | — | input_scale, input_zero_point |
| `qnn.quantize` | — | output_scale, output_zero_point |

---

## 27.3 QNN 算子详解

### 27.3.1 qnn.conv2d 量化卷积

量化卷积是最关键的量化算子，其计算过程为：

$$\text{output} = \text{requantize}\left(\sum_{k}(q_x - z_x) \cdot (q_w - z_w) + b\right)$$

其中减去零点的操作在整数域完成，累加在 INT32 中进行，最后重量化到目标精度。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import tvm
from tvm import relay

# 定义量化卷积
x = relay.var("x", shape=(1, 3, 224, 224), dtype="int8")
w = relay.var("w", shape=(64, 3, 7, 7), dtype="int8")

# 量化参数
input_scale = relay.const(0.05)
input_zero_point = relay.const(0)
kernel_scale = relay.const(0.01, dtype="float32")  # per-channel
kernel_zero_point = relay.const(0)

# 量化卷积
conv = relay.qnn.conv2d(
    x, w,
    input_scale=input_scale,
    input_zero_point=input_zero_point,
    kernel_scale=kernel_scale,
    kernel_zero_point=kernel_zero_point,
    channels=64,
    kernel_size=(7, 7),
    strides=(2, 2),
    padding=(3, 3),
    out_dtype="int32"  # 累加结果为 INT32
)
```

**qnn.conv2d 的内部实现**（`src/relay/qnn/op/conv2d.cc`）：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// src/relay/qnn/op/conv2d.cc（简化）
Expr QnnConv2d(const Expr& data, const Expr& weight,
               const Expr& input_scale, const Expr& input_zp,
               const Expr& kernel_scale, const Expr& kernel_zp,
               const Conv2DAttrs* attrs) {
  // Step 1: 将输入从 int8 反量化为 int32
  auto data_int32 = Cast(data, "int32");
  data_int32 = Subtract(data_int32, input_zp);

  auto weight_int32 = Cast(weight, "int32");
  weight_int32 = Subtract(weight_int32, kernel_zp);

  // Step 2: 执行整数卷积（INT32 累加）
  auto conv_result = Conv2D(data_int32, weight_int32, attrs);

  // Step 3: 乘以 scale（在 float 或 int 中完成）
  auto output_scale = Multiply(input_scale, kernel_scale);
  auto result_float = Multiply(Cast(conv_result, "float32"), output_scale);

  return result_float;
}
```

### 27.3.2 qnn.dense 量化全连接



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 量化全连接层
x = relay.var("x", shape=(1, 256), dtype="int8")
w = relay.var("w", shape=(1000, 256), dtype="int8")

output = relay.qnn.dense(
    x, w,
    input_scale=relay.const(0.1),
    input_zero_point=relay.const(0),
    kernel_scale=relay.const(0.05),
    kernel_zero_point=relay.const(0),
    units=1000,
    out_dtype="int32"
)
```

### 27.3.3 qnn.add 量化加法

量化加法需要处理**两个不同 scale 和 zero_point 的输入**：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 量化加法：融合 residual connection
a = relay.var("a", dtype="int8")
b = relay.var("b", dtype="int8")

output = relay.qnn.add(
    a, b,
    lhs_scale=relay.const(0.1),
    lhs_zero_point=relay.const(0),
    rhs_scale=relay.const(0.2),
    rhs_zero_point=relay.const(0),
    output_scale=relay.const(0.15),
    output_zero_point=relay.const(0),
)
```

**qnn.add 的计算过程**：

$$q_{\text{out}} = \text{round}\left(\frac{s_a \cdot (q_a - z_a) + s_b \cdot (q_b - z_b)}{s_{\text{out}}}\right) + z_{\text{out}}$$

### 27.3.4 requantize 重量化

当一个量化算子的输出需要作为另一个量化算子的输入时，如果 scale/zero_point 不同，需要进行重量化：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 重量化：调整量化参数
input_q = relay.var("input", dtype="int32")  # 卷积的 INT32 输出

# 从 INT32 重量化到 INT8
requantized = relay.qnn.requantize(
    input_q,
    input_scale=relay.const(0.001),    # INT32 的 scale
    input_zero_point=relay.const(0),
    output_scale=relay.const(0.05),    # INT8 的 scale
    output_zero_point=relay.const(0),
    out_dtype="int8"
)
```

---

## 27.4 校准（Calibration）

### 27.4.1 校准流程概览

校准是量化过程中最关键的步骤，决定了量化参数的质量：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
校准流程：
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  FP32 模型   │───▶│  统计收集     │───▶│  参数计算     │
│             │    │  (Calibration)│    │  (s, z)      │
└─────────────┘    └─────────────┘    └─────────────┘
                        │
                   ┌────┴────┐
                   │ 校准数据 │
                   │ (小批量) │
                   └─────────┘
```

### 27.4.2 校准算法

TVM 支持多种校准算法，定义在 `python/tvm/relay/quantize/calibration.py` 中：

**算法一：MinMax 校准**

最简单的方法，直接使用观测到的最大最小值：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# MinMax 校准
def minmax_calibration(activation_stats):
    """使用最小最大值计算量化参数"""
    x_min = activation_stats["min"]
    x_max = activation_stats["max"]
    scale = (x_max - x_min) / 255.0
    zero_point = round(-x_min / scale)
    return scale, zero_point
```

**算法二：KL 散度校准**

通过最小化量化前后的分布差异来选择最优阈值：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import numpy as np

def kl_calibration(activations, num_bits=8):
    """使用 KL 散度选择最优量化阈值"""
    # 1. 计算激活值的直方图
    hist, bin_edges = np.histogram(activations, bins=2048, density=True)

    # 2. 对于不同的截断阈值，计算 KL 散度
    best_threshold = None
    best_kl = float('inf')

    for threshold_idx in range(128, 2048):
        threshold = bin_edges[threshold_idx]

        # 原始分布（截断后归一化）
        p = hist[:threshold_idx]
        p = p / p.sum()

        # 量化后的分布
        q = quantize_distribution(p, num_bits)

        # KL 散度
        kl = np.sum(p * np.log(p / (q + 1e-10)))

        if kl < best_kl:
            best_kl = kl
            best_threshold = threshold

    return best_threshold
```

**算法三：百分位数校准**

使用激活值分布的百分位数作为范围：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def percentile_calibration(activations, percentile=99.99):
    """使用百分位数确定量化范围"""
    x_min = np.percentile(activations, 100 - percentile)
    x_max = np.percentile(activations, percentile)
    return x_min, x_max
```

### 27.4.3 TVM 校准 API

TVM 提供了完整的校准 API：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import tvm
from tvm import relay
from tvm.relay.quantize import calibrate

# 1. 准备校准数据集
calib_dataset = load_calibration_data("calib_images/", num_samples=100)

# 2. 定义校准函数
def calibration_fn(mod, calib_data):
    """执行校准并收集统计信息"""
    # 创建执行器
    with tvm.target.Target("llvm"):
        lib = relay.build(mod, target="llvm")
    dev = tvm.cpu()
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    # 收集每层的激活值统计
    stats = {}
    for batch in calib_data:
        module.set_input("data", batch)
        module.run()
        # 获取中间层的输出统计
        for i, node_name in enumerate(module.get_num_outputs()):
            output = module.get_output(i).numpy()
            if node_name not in stats:
                stats[node_name] = {"min": float('inf'), "max": float('-inf')}
            stats[node_name]["min"] = min(stats[node_name]["min"], output.min())
            stats[node_name]["max"] = max(stats[node_name]["max"], output.max())

    return stats

# 3. 执行量化 + 校准
quantized_mod = relay.quantize.quantize(
    mod,
    params=params,
    dataset=calib_dataset,
    calibrate_mode="kl",        # "kl" / "minmax" / "percentile"
)
```

---

## 27.5 端到端量化流程

### 27.5.1 Post-Training Quantization (PTQ)

训练后量化是最常用的量化方式，无需重新训练：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import tvm
from tvm import relay
import numpy as np

# Step 1: 加载 FP32 模型
import onnx
model = onnx.load("resnet50.onnx")
mod, params = relay.frontend.from_onnx(
    model,
    shape_dict={"data": (1, 3, 224, 224)}
)

# Step 2: 定义量化配置
quantize_config = relay.quantize.current_qconfig(
    skip_conv_layers=[],       # 不跳过任何卷积层
    nbit_input=8,              # 输入量化位宽
    nbit_weight=8,             # 权重量化位宽
    nbit_activation=8,         # 激活量化位宽
    calibrate_mode="kl",       # 校准算法
    weight_scale="channel",    # 权重量化粒度：per-channel
    dtype_input="int8",
    dtype_weight="int8",
)

# Step 3: 执行量化
with relay.quantize.qconfig(quantize_config):
    quantized_mod = relay.quantize.quantize(mod, params=params)

# Step 4: 量化感知的编译优化
with tvm.target.Target("llvm -mattr=+v8.2a,+dotprod"):
    lib = relay.build(quantized_mod, target="llvm", params=params)

# Step 5: 部署
dev = tvm.cpu()
module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
module.set_input("data", input_data)
module.run()
output = module.get_output(0)
```

### 27.5.2 量化效果验证



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def compare_accuracy(fp32_mod, int8_mod, test_data, test_labels):
    """对比 FP32 和 INT8 模型的精度"""
    dev = tvm.cpu()

    # FP32 模型推理
    with tvm.target.Target("llvm"):
        fp32_lib = relay.build(fp32_mod, target="llvm")
    fp32_exec = tvm.contrib.graph_executor.GraphModule(fp32_lib["default"](dev))

    # INT8 模型推理
    with tvm.target.Target("llvm"):
        int8_lib = relay.build(int8_mod, target="llvm")
    int8_exec = tvm.contrib.graph_executor.GraphModule(int8_lib["default"](dev))

    fp32_correct = 0
    int8_correct = 0
    total = len(test_data)

    for i in range(total):
        # FP32 推理
        fp32_exec.set_input("data", test_data[i])
        fp32_exec.run()
        fp32_pred = np.argmax(fp32_exec.get_output(0).numpy())

        # INT8 推理
        int8_exec.set_input("data", test_data[i])
        int8_exec.run()
        int8_pred = np.argmax(int8_exec.get_output(0).numpy())

        if fp32_pred == test_labels[i]:
            fp32_correct += 1
        if int8_pred == test_labels[i]:
            int8_correct += 1

    print(f"FP32 准确率: {fp32_correct/total*100:.2f}%")
    print(f"INT8 准确率: {int8_correct/total*100:.2f}%")
    print(f"精度损失: {(fp32_correct - int8_correct)/total*100:.2f}%")
```

### 27.5.3 量化模型大小对比



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import os

def compare_model_size(fp32_lib, int8_lib):
    """对比量化前后的模型大小"""
    fp32_path = "/tmp/fp32_model.so"
    int8_path = "/tmp/int8_model.so"

    fp32_lib.export_library(fp32_path)
    int8_lib.export_library(int8_path)

    fp32_size = os.path.getsize(fp32_path) / (1024 * 1024)
    int8_size = os.path.getsize(int8_path) / (1024 * 1024)

    print(f"FP32 模型大小: {fp32_size:.2f} MB")
    print(f"INT8 模型大小: {int8_size:.2f} MB")
    print(f"压缩比: {fp32_size/int8_size:.2f}x")
```

---

## 27.6 FP16 半精度支持

### 27.6.1 FP16 数据类型

FP16（IEEE 754 半精度浮点）使用 16 位表示：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
FP16 位布局：
┌─────┬──────────┬─────────────────┐
│ 符号 │ 指数 (5b) │  尾数 (10b)      │
└─────┴──────────┴─────────────────┘
  15    14...10     9...0

范围：±65504，精度：~3.3 位十进制
```

**FP16 vs BF16**：

| 类型 | 位数 | 指数 | 尾数 | 范围 | 精度 |
|------|------|------|------|------|------|
| FP32 | 32 | 8 | 23 | ±3.4e38 | 7.2 位 |
| FP16 | 16 | 5 | 10 | ±65504 | 3.3 位 |
| BF16 | 16 | 8 | 7 | ±3.4e38 | 2.4 位 |

### 27.6.2 TVM 中的 FP16 量化



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import tvm
from tvm import relay

# FP16 量化：自动将 FP32 算子转换为 FP16
mod, params = relay.frontend.from_onnx(model)

# 方式一：使用 ToMixedPrecision Pass
from tvm.relay.transform import ToMixedPrecision

# 配置 FP16 转换
mixed_precision_config = {
    "float16_ops": ["nn.conv2d", "nn.dense", "nn.conv2d_transpose"],
    "skip_ops": ["nn.softmax"],  # softmax 保持 FP32
}

mod_fp16 = ToMixedPrecision("float16")(mod)
```

**ToMixedPrecision Pass 的工作原理**：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// src/relay/transforms/to_mixed_precision.cc（简化）
class MixedPrecisionRewriter : public MixedModeMutator {
  Expr RewriteConv2D(const CallNode* call) {
    // 1. 将权重转换为 FP16
    auto weight_fp16 = Cast(call->args[1], DataType::Float(16));
    // 2. 将输入转换为 FP16
    auto input_fp16 = Cast(call->args[0], DataType::Float(16));
    // 3. 执行 FP16 卷积
    auto result = Conv2D(input_fp16, weight_fp16, attrs);
    // 4. 转回 FP32（如果后续算子需要）
    return Cast(result, DataType::Float(32));
  }
};
```

### 27.6.3 FP16 + INT8 混合精度

实际部署中，混合使用 FP16 和 INT8 可以获得最佳精度-性能平衡：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 混合精度策略
# 卷积层：INT8（计算密集，量化收益大）
# 注意力层：FP16（精度敏感）
# 分类头：FP32（精度最关键）

def mixed_precision_compile(mod, params):
    # 1. 标记需要保持高精度的层
    skip_layers = ["layer4.2.fc", "classifier"]

    # 2. 对非跳过层执行 INT8 量化
    with relay.quantize.qconfig(
        nbit_input=8,
        nbit_weight=8,
        skip_conv_layers=skip_layers
    ):
        mod_int8 = relay.quantize.quantize(mod, params=params)

    # 3. 对剩余层使用 FP16
    mod_mixed = relay.transform.ToMixedPrecision("float16")(mod_int8)

    return mod_mixed
```

---

## 27.7 量化感知训练支持

### 27.7.1 QAT 原理

量化感知训练（Quantization-Aware Training）在训练过程中模拟量化效果，使模型适应量化带来的精度损失：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
标准训练：                QAT 训练：
  FP32 前向传播            FP32 前向传播
       │                       │
       │                  量化模拟（Fake Quant）
       │                  q = round(x/s)*s
       │                       │
  FP32 反向传播            FP32 反向传播
  （直通估计器 STE）        （直通估计器 STE）
```

**直通估计器（Straight-Through Estimator, STE）**：

$$\frac{\partial q}{\partial x} \approx \begin{cases} 1 & \text{if } q_{\min} \leq x \leq q_{\max} \\ 0 & \text{otherwise} \end{cases}$$

### 27.7.2 TVM 中的 QAT 支持



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import tvm
from tvm import relay

# QAT 模型的导入（通常从 PyTorch/TensorFlow 导入已训练的 QAT 模型）

# 方式一：导入 PyTorch QAT 模型
import torch
from torch.quantization import get_default_qat_qconfig, prepare_qat

# PyTorch QAT 训练完成后的模型
qat_model = load_qat_pytorch_model()
# 转换为 TorchScript
scripted = torch.jit.trace(qat_model, example_input)
# 导入到 TVM
mod, params = relay.frontend.from_pytorch(scripted, input_shapes)

# 方式二：在 Relay 中插入 FakeQuant 节点
def insert_fake_quant(mod):
    """在关键位置插入伪量化节点"""
    class FakeQuantInserter(relay.ExprMutator):
        def visit_call(self, call):
            new_args = []
            for arg in call.args:
                if arg.checked_type.dtype == "float32":
                    # 插入 FakeQuant
                    fq = relay.qnn.quantize(
                        arg,
                        output_scale=relay.const(0.1),
                        output_zero_point=relay.const(0),
                        out_dtype="float32"  # 伪量化：输出仍是 float
                    )
                    new_args.append(fq)
                else:
                    new_args.append(arg)
            return relay.Call(call.op, new_args, call.attrs)
    return FakeQuantInserter().visit(mod)
```

---

## 27.8 量化代码生成优化

### 27.8.1 INT8 向量化指令

TVM 可以在代码生成阶段将量化操作映射到硬件特定的 INT8 向量化指令：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# x86 AVX-512 VNNI 指令
with tvm.target.Target("llvm -mcpu=cascadelake -mattr=+vnni"):
    lib = relay.build(quantized_mod, target="llvm")
# 生成的代码使用 vpdpbusd 指令进行 INT8 点积

# ARM NEON + SDOT 指令
with tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+dotprod"):
    lib = relay.build(quantized_mod, target="llvm")
# 生成的代码使用 SDOT 指令
```

### 27.8.2 VNNI/DotProd 指令映射



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// LLVM CodeGen 中的 INT8 点积指令生成
// src/target/llvm/codegen_llvm.cc

// x86 VNNI: vpdpbusd
// 指令语义：
//   for i in 0..15:
//     accum[i] += int32(a[4*i:4*i+3]) · int32(b[4*i:4*i+3])

// ARM DotProd: sdot
// 指令语义：
//   for i in 0..3:
//     accum[i] += int32(a[4*i]) * int32(b[4*i]) + ...
```

### 27.8.3 量化卷积的循环优化



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 量化卷积的调度优化
# INT8 卷积可以使用更激进的分块策略，因为数据量更小

# FP32 卷积分块
# TILE_N=32, TILE_K=32 → 需要 32*32*4 = 4096B 缓存

# INT8 卷积分块
# TILE_N=64, TILE_K=64 → 需要 64*64*1 = 4096B 缓存（相同空间，4x 计算量）
```

---

## 27.9 实战：完整量化部署示例

### 27.9.1 ResNet-50 INT8 量化部署



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import tvm
from tvm import relay
import numpy as np
import onnx

def quantize_resnet50():
    # 1. 加载模型
    model = onnx.load("resnet50-v1-7.onnx")
    mod, params = relay.frontend.from_onnx(
        model,
        shape_dict={"data": (1, 3, 224, 224)}
    )

    # 2. 预处理：融合 BN 到 Conv
    seq = tvm.transform.Sequential([
        relay.transform.InferType(),
        relay.transform.FoldConstant(),
        relay.transform.FuseOps(fuse_opt_level=2),
    ])
    mod = seq(mod)

    # 3. 准备校准数据
    calib_data = []
    for i in range(100):
        img = np.random.randn(1, 3, 224, 224).astype("float32")
        calib_data.append(img)

    # 4. 量化配置
    with relay.quantize.qconfig(
        nbit_input=8,
        nbit_weight=8,
        calibrate_mode="kl",
        weight_scale="channel",
        skip_conv_layers=[],
    ):
        quantized_mod = relay.quantize.quantize(
            mod, params=params, dataset=calib_data
        )

    # 5. 编译
    target = tvm.target.Target("llvm -mcpu=skylake-avx512")
    with target:
        lib = relay.build(quantized_mod, target=target, params=params)

    # 6. 部署
    dev = tvm.cpu()
    module = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    # 7. 推理
    input_data = np.random.randn(1, 3, 224, 224).astype("float32")
    module.set_input("data", input_data)
    module.run()
    output = module.get_output(0)

    return output

result = quantize_resnet50()
```

---

## 27.99 文字内容强化：量化 的工程化理解

这一节用于把前文的 API、IR、Pass、Runtime 和部署片段串联为更完整的工程叙事。
很多学习者第一次阅读 TVM 文档时会觉得示例代码很多，但真正上线时仍然不知道如何判断方案是否可靠。
原因在于 TVM 不是单个推理库，而是一条从模型语义到硬件代码的编译链路。
链路越长，越需要把每一步的业务目标、内部机制、适用边界和失败模式说清楚。

### 27.99.1 代码解读的阅读方法

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

- 围绕“量化参数、校准数据与 QNN 算子的语义”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“INT8、FP16、混合精度在不同硬件上的收益边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“精度回归、溢出和尺度传播的排查方法”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“量化部署与 TensorRT、oneDNN、TFLite 的差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 27.99.2 业务意义

1. 量化 的业务价值不只是让模型跑得更快，而是让同一个模型可以在不同成本、功耗和延迟约束下交付。
2. 在服务器场景中，核心指标通常是吞吐、P95/P99 延迟、资源利用率和多租户隔离。
3. 在移动端场景中，核心指标通常是首帧时间、持续发热、内存峰值和包体大小。
4. 在嵌入式场景中，核心指标通常是 Flash 占用、静态内存、实时性和掉电恢复能力。
5. 在云端批处理场景中，编译时间可以接受更长，但调优记录和缓存复用变得非常重要。
6. 在在线服务场景中，编译产物需要可回滚、可审计、可灰度，而不能只在开发机上验证。
7. 业务方关心的是 SLA、成本和稳定性，编译器工程师关心的是 IR 正确性、优化空间和后端能力。
8. 优秀的 TVM 项目需要把这两类语言翻译成共同的指标体系。
9. 当优化收益只有少量百分点时，应评估它是否值得引入新的维护复杂度。
10. 当优化收益很大但只在少数输入上成立时，应评估输入分布变化后的风险。

- 围绕“量化参数、校准数据与 QNN 算子的语义”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“INT8、FP16、混合精度在不同硬件上的收益边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“精度回归、溢出和尺度传播的排查方法”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“量化部署与 TensorRT、oneDNN、TFLite 的差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 27.99.3 TVM 内部机制

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

- 围绕“量化参数、校准数据与 QNN 算子的语义”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“INT8、FP16、混合精度在不同硬件上的收益边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“精度回归、溢出和尺度传播的排查方法”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“量化部署与 TensorRT、oneDNN、TFLite 的差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 27.99.4 适用场景

1. 当模型结构相对稳定、目标硬件明确、性能收益可以通过基准测试确认时，量化 相关技术最容易发挥价值。
2. 当团队需要支持多种硬件后端时，TVM 的统一 IR 和 Target 抽象可以降低重复适配成本。
3. 当模型中存在框架运行时开销、算子融合机会或布局转换冗余时，编译优化通常能带来明显收益。
4. 当部署环境不能依赖完整 Python 栈时，AOT、CRT 或导出后的 runtime artifact 更有意义。
5. 当硬件厂商提供高性能库但模型图需要复杂切分时，BYOC 和外部 codegen 是常见选择。
6. 当输入形状变化频繁时，应提前设计 shape 策略，而不是在上线前才补动态形状支持。
7. 当模型版本迭代频繁时，应把编译、调优、验证和发布纳入 CI/CD。
8. 当业务对精度非常敏感时，应把优化收益和数值回归一起评估。
9. 当系统存在多模型串联时，应评估端到端 pipeline，而不是只优化单个模型。
10. 当部署设备数量很大时，编译产物的一致性和可追踪性比单次实验性能更重要。

- 围绕“量化参数、校准数据与 QNN 算子的语义”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“INT8、FP16、混合精度在不同硬件上的收益边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“精度回归、溢出和尺度传播的排查方法”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“量化部署与 TensorRT、oneDNN、TFLite 的差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 27.99.5 限制条件

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

- 围绕“量化参数、校准数据与 QNN 算子的语义”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“INT8、FP16、混合精度在不同硬件上的收益边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“精度回归、溢出和尺度传播的排查方法”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“量化部署与 TensorRT、oneDNN、TFLite 的差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 27.99.6 工程经验

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

- 围绕“量化参数、校准数据与 QNN 算子的语义”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“INT8、FP16、混合精度在不同硬件上的收益边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“精度回归、溢出和尺度传播的排查方法”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“量化部署与 TensorRT、oneDNN、TFLite 的差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 27.99.7 常见误区

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

- 围绕“量化参数、校准数据与 QNN 算子的语义”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“INT8、FP16、混合精度在不同硬件上的收益边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“精度回归、溢出和尺度传播的排查方法”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“量化部署与 TensorRT、oneDNN、TFLite 的差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 27.99.8 生产部署注意事项

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

- 围绕“量化参数、校准数据与 QNN 算子的语义”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“INT8、FP16、混合精度在不同硬件上的收益边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“精度回归、溢出和尺度传播的排查方法”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“量化部署与 TensorRT、oneDNN、TFLite 的差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 27.99.9 与同类系统对比

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

- 围绕“量化参数、校准数据与 QNN 算子的语义”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“INT8、FP16、混合精度在不同硬件上的收益边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“精度回归、溢出和尺度传播的排查方法”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“量化部署与 TensorRT、oneDNN、TFLite 的差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。

### 27.99.10 章节复盘

1. 回到本章，量化 的关键不是记住所有 API，而是理解为什么这些 API 会出现在编译链路的这个位置。
2. 当你看到一段代码时，应能说出它改变了模型语义、调度空间、内存布局、运行时入口还是部署产物。
3. 当你看到一个性能数字时，应能说出它的测试输入、硬件状态、计时方法和误差范围。
4. 当你看到一个优化 pass 时，应能说出它依赖的前置假设和可能破坏的边界条件。
5. 当你准备上线时，应能说出失败后如何回滚、如何复现、如何定位和如何与业务方沟通影响。
6. 这套思维比单个示例更重要，因为 TVM 的 API 会演进，但编译部署的工程约束长期稳定。
7. 后续学习中，可以把每一章都转化为一张决策表：何时使用、收益来自哪里、风险是什么、如何验证。
8. 只有把代码、机制和工程策略放在一起，TVM 才不只是工具箱，而是可运行的生产系统。
9. 因此，本章新增的文字说明应作为阅读代码段的上下文，而不是替代对原始代码的逐行理解。
10. 如果遇到与示例不一致的实际项目，应优先回到模型约束和目标硬件，而不是机械套用章节流程。

- 围绕“量化参数、校准数据与 QNN 算子的语义”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“INT8、FP16、混合精度在不同硬件上的收益边界”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“精度回归、溢出和尺度传播的排查方法”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。
- 围绕“量化部署与 TensorRT、oneDNN、TFLite 的差异”，工程评审时应同时追问收益来源、验证方法、失败影响和维护责任。


## 27.10 本章小结

本章全面介绍了 TVM 的量化支持体系：

1. **量化基础**：线性量化、对称/非对称、量化粒度
2. **QNN Dialect**：量化算子定义与实现
3. **校准流程**：MinMax、KL 散度、百分位数校准
4. **INT8/FP16 推理**：端到端量化编译与部署
5. **QAT 支持**：量化感知训练与 FakeQuant
6. **硬件优化**：VNNI/DotProd 指令利用

**关键源码索引**：

| 模块 | 源码路径 |
|------|---------|
| QNN 算子实现 | `src/relay/qnn/op/` |
| QNN 变换 Pass | `src/relay/qnn/transform/` |
| 量化配置 | `python/tvm/relay/quantize/quantize.py` |
| 校准算法 | `python/tvm/relay/quantize/calibration.py` |
| FP16 转换 | `src/relay/transforms/to_mixed_precision.cc` |

<div data-component="QuantizationPipelineDiagram"></div>

---

## 27.11 量化算术细节

### 27.11.1 量化乘法的精度保持

两个量化值相乘时，需要仔细处理精度：

$$q_a \cdot q_b = s_a \cdot s_b \cdot (q_a - z_a) \cdot (q_b - z_b)$$

在 INT8 × INT8 → INT32 的场景下：

```c
// 量化乘法的实现
int32_t quantized_multiply(int8_t a, int8_t b,
                           int8_t za, int8_t zb) {
    // 反量化零点
    int32_t a_unbiased = (int32_t)a - (int32_t)za;
    int32_t b_unbiased = (int32_t)b - (int32_t)zb;

    // INT8 × INT8 → INT32（无精度损失）
    int32_t product = a_unbiased * b_unbiased;

    return product;
}
```

### 27.11.2 量化加法的 scale 对齐

当两个量化值的 scale 不同时，需要先对齐：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```c
// 量化加法：scale 对齐
int8_t quantized_add(int8_t a, int8_t b,
                     float sa, float sb, float s_out,
                     int8_t za, int8_t zb, int8_t z_out) {
    // 方法 1：转换到浮点域（简单但慢）
    float fa = sa * ((float)a - (float)za);
    float fb = sb * ((float)b - (float)zb);
    float result = fa + fb;
    int8_t q = (int8_t)round(result / s_out) + z_out;
    return q;

    // 方法 2：定点数运算（高效）
    // 使用 LUT（查找表）预计算乘法因子
}
```

### 27.11.3 量化 Batch Normalization

BN 层在推理时可以完全融合到前一层：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# BN 融合的数学推导
# 原始：y = gamma * (x - mean) / sqrt(var + eps) + beta
# 变形：y = (gamma / sqrt(var + eps)) * x + (beta - gamma * mean / sqrt(var + eps))
# 简化：y = scale_bn * x + shift_bn

# 量化版本：
# q_y = round((scale_bn * s_x * (q_x - z_x) + shift_bn) / s_y) + z_y

def fuse_bn_to_conv(conv_weight, conv_bias, bn_gamma, bn_beta,
                     bn_mean, bn_var, bn_eps=1e-5):
    """将 BN 融合到卷积权重和偏置中"""
    scale = bn_gamma / np.sqrt(bn_var + bn_eps)
    # 融合后的权重
    fused_weight = conv_weight * scale.reshape(-1, 1, 1, 1)
    # 融合后的偏置
    fused_bias = scale * (conv_bias - bn_mean) + bn_beta
    return fused_weight, fused_bias
```

### 27.11.4 量化 ReLU6 和 Hardswish



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# ReLU6 的量化实现
def quantized_relu6(input_q, input_scale, input_zp, output_scale, output_zp):
    """量化 ReLU6: min(max(x, 0), 6)"""
    # 计算 6 的量化值
    six_q = round(6.0 / output_scale) + output_zp
    zero_q = round(0.0 / output_scale) + output_zp

    # 在整数域实现
    result = np.clip(input_q, zero_q, six_q)
    return result

# Hardswish 的量化实现
# hardswish(x) = x * relu6(x + 3) / 6
def quantized_hardswish(input_q, input_scale, input_zp):
    """量化 Hardswish"""
    # 需要多次乘法和移位操作
    # 通常使用查找表（LUT）实现
    pass
```

---

## 27.12 量化策略进阶

### 27.12.1 混合精度量化

不同层对量化的敏感度不同，混合精度量化为每层选择最优位宽：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def mixed_precision_quantization(mod, sensitivity_analysis):
    """基于敏感度分析的混合精度量化"""

    # 敏感度分析结果（每层的精度损失）
    # layer_0: 0.01%  → INT4
    # layer_1: 0.5%   → INT8
    # layer_2: 2.0%   → FP16
    # layer_3: 0.02%  → INT4

    bitwidth_assignment = {}
    for layer, sensitivity in sensitivity_analysis.items():
        if sensitivity < 0.05:
            bitwidth_assignment[layer] = 4    # INT4
        elif sensitivity < 0.5:
            bitwidth_assignment[layer] = 8    # INT8
        elif sensitivity < 1.0:
            bitwidth_assignment[layer] = 16   # FP16
        else:
            bitwidth_assignment[layer] = 32   # FP32

    # 应用混合精度配置
    return apply_bitwidth(mod, bitwidth_assignment)
```

### 27.12.2 量化感知训练的 STE 改进

标准 STE 的梯度估计存在偏差，改进方法包括：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 标准 STE
class StraightThroughEstimator:
    def forward(self, x, scale, zp, qmin, qmax):
        # 前向：量化
        q = torch.clamp(torch.round(x / scale) + zp, qmin, qmax)
        return q

    def backward(self, grad_output):
        # 反向：直通（梯度直接传递）
        return grad_output

# 改进：LSQ（Learned Step Size Quantization）
class LSQEstimator:
    def forward(self, x, scale, zp, qmin, qmax):
        self.save_for_backward(x, scale)
        q = torch.clamp(torch.round(x / scale) + zp, qmin, qmax)
        return q

    def backward(self, grad_output):
        x, scale = self.saved_tensors
        # 梯度缩减因子
        grad_scale = 1.0 / torch.sqrt(qmax * x.numel())
        # 只在量化范围内传递梯度
        mask = (x >= qmin * scale) & (x <= qmax * scale)
        return grad_output * mask.float() * grad_scale
```

### 27.12.3 量化友好的模型设计



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 量化友好的设计原则

# 1. 使用 ReLU6 而非 ReLU
#    ReLU6 有界，量化范围更小
def relu6(x):
    return min(max(x, 0), 6)

# 2. 避免过深的残差连接
#    深层残差会导致 scale 传播误差累积
#    建议：每 4-6 层添加一个残差连接

# 3. 使用 Group Normalization 替代 Batch Normalization
#    GN 对量化更友好，不依赖 batch 统计
def group_norm(x, gamma, beta, num_groups=32):
    # 对每个 group 独立归一化
    pass

# 4. 使用 1×1 卷积进行通道变换
#    1×1 卷积的量化误差更小
```

---

## 27.13 硬件特定量化优化

### 27.13.1 ARM NEON INT8 优化



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```c
// ARM NEON INT8 矩阵乘法
// 使用 vmlal_s8 指令（有符号 8 位乘加到 16 位）
void neon_int8_matmul(const int8_t* A, const int8_t* B, int32_t* C,
                      int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j += 4) {
            int32x4_t acc = vdupq_n_s32(0);
            for (int k = 0; k < K; k += 8) {
                int8x8_t a = vld1_s8(&A[i * K + k]);
                int8x8_t b = vld1_s8(&B[k * N + j]);
                // 8 个 INT8 乘加到 4 个 INT32
                acc = vmlal_s8(acc, vget_low_s8(a), vget_low_s8(b));
            }
            vst1q_s32(&C[i * N + j], acc);
        }
    }
}
```

### 27.13.2 x86 AVX-512 VNNI 优化



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```c
// x86 AVX-512 VNNI INT8 点积
// 使用 _mm512_dpbusd_epi32 指令
void vnni_int8_conv(const int8_t* input, const int8_t* kernel,
                    int32_t* output, int channels) {
    __m512i acc = _mm512_setzero_si512();

    for (int c = 0; c < channels; c += 64) {
        // 加载 64 个 INT8 值
        __m512i inp = _mm512_loadu_si512(&input[c]);
        __m512i ker = _mm512_loadu_si512(&kernel[c]);

        // INT8 点积：每 4 个元素为一组，累加到 INT32
        acc = _mm512_dpbusd_epi32(acc, inp, ker);
    }

    // 存储结果
    _mm512_storeu_si512(output, acc);
}
```

### 27.13.3 NVIDIA Tensor Core INT8



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cuda
// NVIDIA Tensor Core INT8 矩阵乘法
// 使用 wmma（Warp Matrix Multiply Accumulate）API
__global__ void tensorcore_int8_matmul(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K
) {
    // 声明 warp 级别的矩阵片段
    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t> c_frag;

    wmma::fill_fragment(c_frag, 0);

    // 加载矩阵片段
    wmma::load_matrix_sync(a_frag, A + ..., K);
    wmma::load_matrix_sync(b_frag, B + ..., N);

    // Tensor Core 矩阵乘法
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // 存储结果
    wmma::store_matrix_sync(C + ..., c_frag, N, wmma::mem_row_major);
}
```

---

## 27.14 量化调试与诊断

### 27.14.1 量化误差分析



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def analyze_quantization_error(fp32_model, int8_model, test_data):
    """分析量化误差的来源和分布"""

    errors = {}
    for layer_name in fp32_model.layer_names:
        # 获取 FP32 和 INT8 的中间输出
        fp32_output = fp32_model.get_layer_output(layer_name, test_data)
        int8_output = int8_model.get_layer_output(layer_name, test_data)

        # 计算误差统计
        abs_error = np.abs(fp32_output - int8_output)
        rel_error = abs_error / (np.abs(fp32_output) + 1e-10)

        errors[layer_name] = {
            "mean_abs_error": np.mean(abs_error),
            "max_abs_error": np.max(abs_error),
            "mean_rel_error": np.mean(rel_error),
            "max_rel_error": np.max(rel_error),
            "cosine_similarity": cosine_sim(fp32_output, int8_output),
        }

    return errors
```

### 27.14.2 量化友好的调试工具



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# TVM 量化调试工具
def debug_quantization(mod, params, calib_data):
    """量化过程的详细调试"""

    # 1. 检查每层的数值范围
    print("=== 数值范围分析 ===")
    for name, param in params.items():
        data = param.numpy()
        print(f"{name}: min={data.min():.6f}, max={data.max():.6f}, "
              f"mean={data.mean():.6f}, std={data.std():.6f}")

    # 2. 检查量化参数
    print("\n=== 量化参数 ===")
    for name, param in params.items():
        scale, zp = compute_quantization_params(param.numpy())
        print(f"{name}: scale={scale:.6f}, zero_point={zp}")

    # 3. 模拟量化后的数值分布
    print("\n=== 量化后数值分布 ===")
    for name, param in params.items():
        data = param.numpy()
        scale, zp = compute_quantization_params(data)
        quantized = np.clip(np.round(data / scale) + zp, -128, 127)
        dequantized = (quantized - zp) * scale
        error = np.abs(data - dequantized)
        print(f"{name}: max_error={error.max():.6f}, "
              f"mean_error={error.mean():.6f}")
```

---

## 27.15 量化部署最佳实践

### 27.15.1 完整的量化部署流程



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def end_to_end_quantization_pipeline(model_path, calib_data_path,
                                      target_device):
    """完整的量化部署流程"""

    # 1. 加载模型
    mod, params = load_model(model_path)

    # 2. 图优化（量化前）
    mod = optimize_graph(mod)

    # 3. 敏感度分析
    sensitivity = analyze_layer_sensitivity(mod, calib_data_path)

    # 4. 确定量化配置
    quant_config = determine_quantization_config(
        sensitivity, target_device)

    # 5. 校准
    with relay.quantize.qconfig(**quant_config):
        quantized_mod = relay.quantize.quantize(
            mod, params=params, dataset=calib_data_path)

    # 6. 量化后优化
    quantized_mod = post_quantization_optimize(quantized_mod)

    # 7. 编译
    lib = compile_model(quantized_mod, target_device)

    # 8. 验证精度
    accuracy = validate_accuracy(lib, val_data_path)

    # 9. 性能测试
    latency = benchmark_latency(lib, target_device)

    print(f"精度: {accuracy:.2f}%")
    print(f"延迟: {latency:.2f} ms")

    return lib
```

### 27.15.2 常见量化问题排查

| 问题 | 症状 | 解决方案 |
|------|------|---------|
| 精度大幅下降 | >5% 精度损失 | 使用 KL 校准，增加校准数据量 |
| 某层误差大 | 层级精度异常 | 跳过该层量化，使用更高位宽 |
| 数值溢出 | 输出 NaN/Inf | 检查 scale 计算，使用对称量化 |
| 性能提升不明显 | <1.5× 加速 | 检查目标平台是否支持 INT8 指令 |
| 编译失败 | QNN 算子不支持 | 更新 TVM 版本或添加自定义算子 |

### 27.15.3 量化配置模板



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 模板 1：高精度优先（精度损失 <0.5%）
config_high_accuracy = {
    "nbit_input": 8,
    "nbit_weight": 8,
    "calibrate_mode": "kl",
    "weight_scale": "channel",
    "skip_conv_layers": ["first_conv", "last_fc"],
}

# 模板 2：性能优先（精度损失 <2%）
config_high_performance = {
    "nbit_input": 8,
    "nbit_weight": 8,
    "calibrate_mode": "minmax",
    "weight_scale": "tensor",
    "skip_conv_layers": [],
}

# 模板 3：极致压缩（精度损失 <5%）
config_extreme_compression = {
    "nbit_input": 4,
    "nbit_weight": 4,
    "calibrate_mode": "percentile",
    "weight_scale": "channel",
    "skip_conv_layers": ["first_conv"],
}
```

---

## 27.16 量化与自动调优集成

### 27.16.1 量化感知的调度搜索

量化模型的最优调度与 FP32 模型不同，需要专门的搜索策略：



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
from tvm.meta_schedule import TuneConfig

# 量化感知的自动调优
def quantization_aware_tuning(quantized_mod, target):
    """在量化模型上执行自动调优"""

    # 量化模型的搜索空间特点：
    # 1. INT8 算子可以使用更激进的分块
    # 2. 向量化宽度可以更大（INT8 vs FP32）
    # 3. 内存占用更小，可以使用更大的 tile

    config = TuneConfig(
        max_trials=200,
        num_trials_per_iter=10,
        strategy="evolutionary",
        # 量化特定的搜索空间约束
        space_config={
            "tile_sizes": [16, 32, 64, 128],  # INT8 可以用更大 tile
            "vectorize_width": [16, 32],        # INT8 向量化更宽
        }
    )

    with target:
        database = meta_schedule.tune_tasks(
            tasks=extract_tasks(quantized_mod),
            config=config,
            work_dir="./quant_tuning_results",
        )

    return database
```

### 27.16.2 量化模型的调度模板



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# 针对量化模型的调度模板
def quantized_conv2d_schedule(s, C, input_dtype="int8"):
    """INT8 卷积的调度模板"""

    if input_dtype == "int8":
        # INT8 可以使用更大的分块
        tile_h, tile_w = 64, 64
        tile_c = 32
        vectorize_factor = 16  # 16 个 INT8 = 128 位
    else:
        # FP32 使用较小的分块
        tile_h, tile_w = 32, 32
        tile_c = 16
        vectorize_factor = 4   # 4 个 FP32 = 128 位

    # 分块
    h, w = s[C].op.axis[0], s[C].op.axis[1]
    c = s[C].op.reduce_axis[0]

    ho, hi = s[C].split(h, factor=tile_h)
    wo, wi = s[C].split(w, factor=tile_w)
    co, ci = s[C].split(c, factor=tile_c)

    s[C].reorder(ho, wo, co, hi, ci, wi)

    # 向量化
    s[C].vectorize(wi)

    # 并行
    s[C].parallel(ho)

    return s
```

### 27.16.3 量化与算子融合



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def quantized_fusion_pass(mod):
    """量化模型的算子融合"""

    # 量化模型的融合模式
    # 1. QConv2d + ReLU → QConv2dReLU
    # 2. QConv2d + BN + ReLU → QConv2dBNReLU
    # 3. QDense + ReLU → QDenseReLU
    # 4. QAdd + ReLU → QAddReLU

    class QuantizedFuser(relay.ExprMutator):
        def visit_call(self, call):
            # 模式 1: QConv2d + ReLU
            if (call.op == relay.op.get("nn.relu") and
                isinstance(call.args[0], relay.Call) and
                call.args[0].op == relay.op.get("qnn.conv2d")):

                conv_call = call.args[0]
                # 融合为带 ReLU 的量化卷积
                return relay.qnn.conv2d(
                    conv_call.args[0],  # 输入
                    conv_call.args[1],  # 权重
                    **conv_call.attrs,
                    activation="relu"   # 添加激活函数
                )

            return super().visit_call(call)

    return QuantizedFuser().visit(mod)
```

---

## 27.17 量化数值稳定性

### 27.17.1 数值溢出检测



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def detect_quantization_overflow(model, test_data):
    """检测量化过程中的数值溢出"""

    overflow_report = {}

    for layer_name in model.layer_names:
        # 获取该层的输出范围
        output = model.get_layer_output(layer_name, test_data)

        # 检查 INT32 累加器是否溢出
        if output.dtype == np.int32:
            max_val = np.max(np.abs(output))
            if max_val > 2**31 - 1:
                overflow_report[layer_name] = {
                    "type": "int32_overflow",
                    "max_value": max_val,
                    "suggestion": "使用更大的累加器或调整量化参数"
                }

        # 检查 INT8 输出是否饱和
        if output.dtype == np.int8:
            saturation_rate = np.mean(np.abs(output) == 127)
            if saturation_rate > 0.01:  # 超过 1% 饱和
                overflow_report[layer_name] = {
                    "type": "int8_saturation",
                    "saturation_rate": saturation_rate,
                    "suggestion": "调整 scale 或使用非对称量化"
                }

    return overflow_report
```

### 27.17.2 Scale 稳定性优化



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def stabilize_quantization_scales(mod, params):
    """优化量化 scale 的稳定性"""

    # 问题：某些层的 scale 可能非常小或非常大
    # 导致量化精度损失或数值不稳定

    class ScaleStabilizer(relay.ExprMutator):
        def visit_call(self, call):
            if call.op == relay.op.get("qnn.conv2d"):
                # 检查输入 scale
                input_scale = call.args[2]
                kernel_scale = call.args[4]

                # 如果 scale 太小，可能导致下溢
                if is_scale_too_small(input_scale):
                    input_scale = adjust_scale(input_scale, min_val=1e-6)

                # 如果 scale 太大，可能导致上溢
                if is_scale_too_large(input_scale):
                    input_scale = adjust_scale(input_scale, max_val=1.0)

                # 重新构建调用
                return relay.qnn.conv2d(
                    call.args[0], call.args[1],
                    input_scale, call.args[3],
                    kernel_scale, call.args[5],
                    **call.attrs
                )

            return super().visit_call(call)

    return ScaleStabilizer().visit(mod)
```

### 27.17.3 量化误差传播分析



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def analyze_error_propagation(mod, params, calib_data):
    """分析量化误差在模型中的传播"""

    # 逐层分析量化误差
    layer_errors = {}
    accumulated_error = 0.0

    for layer_name in get_layer_order(mod):
        # 获取 FP32 和 INT8 的输出
        fp32_out = get_fp32_output(mod, layer_name, calib_data)
        int8_out = get_int8_output(mod, layer_name, calib_data)

        # 计算单层误差
        layer_error = np.mean(np.abs(fp32_out - int8_out))

        # 计算累积误差
        accumulated_error += layer_error

        layer_errors[layer_name] = {
            "layer_error": layer_error,
            "accumulated_error": accumulated_error,
            "relative_error": layer_error / (np.mean(np.abs(fp32_out)) + 1e-10),
        }

    return layer_errors
```

---

## 27.18 量化模型压缩

### 27.18.1 量化 + 剪枝联合优化



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def quantize_and_prune(mod, params, sparsity=0.5):
    """量化与剪枝的联合优化"""

    # 1. 先进行幅度剪枝
    pruned_params = {}
    for name, param in params.items():
        data = param.numpy()
        if len(data.shape) >= 2:
            # 计算阈值
            threshold = np.percentile(np.abs(data), sparsity * 100)
            # 应用掩码
            mask = np.abs(data) >= threshold
            pruned_data = data * mask
            pruned_params[name] = tvm.nd.array(pruned_data)
        else:
            pruned_params[name] = param

    # 2. 对剪枝后的模型进行量化
    with relay.quantize.qconfig(
        nbit_input=8,
        nbit_weight=8,
        calibrate_mode="kl",
    ):
        quantized_mod = relay.quantize.quantize(mod, params=pruned_params)

    # 3. 统计压缩效果
    original_size = sum(p.numpy().nbytes for p in params.values())
    compressed_size = count_nonzero_bytes(pruned_params)  # 仅非零元素
    compression_ratio = original_size / max(compressed_size, 1)

    print(f"原始大小: {original_size / 1024:.1f} KB")
    print(f"压缩后大小: {compressed_size / 1024:.1f} KB")
    print(f"压缩比: {compression_ratio:.1f}x")

    return quantized_mod, pruned_params
```

### 27.18.2 混合精度量化压缩



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def mixed_precision_compress(mod, params, target_size_kb):
    """通过混合精度量化达到目标大小"""

    # 计算当前模型大小
    current_size = sum(p.numpy().nbytes for p in params.values())

    # 逐层分析敏感度
    sensitivity = analyze_layer_sensitivity(mod, params)

    # 按敏感度排序，低敏感度层使用更低精度
    sorted_layers = sorted(sensitivity.items(), key=lambda x: x[1])

    bitwidth_config = {}
    for layer_name, sens in sorted_layers:
        if current_size <= target_size_kb * 1024:
            break

        if sens < 0.1:
            bitwidth_config[layer_name] = 4   # INT4
            current_size *= 0.5
        elif sens < 0.5:
            bitwidth_config[layer_name] = 8   # INT8
        else:
            bitwidth_config[layer_name] = 16  # FP16

    # 应用混合精度配置
    return apply_mixed_precision(mod, bitwidth_config)
```

---

## 27.19 量化部署平台适配

### 27.19.1 x86 平台优化



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# x86 平台的量化优化配置
x86_quant_config = {
    "target": "llvm -mcpu=skylake-avx512 -mattr=+vnni",
    "nbit_input": 8,
    "nbit_weight": 8,
    "vectorize_width": 64,  # AVX-512 = 512 bits = 64 INT8
    "use_vnni": True,       # 使用 VNNI 指令
    "parallel_threads": 4,   # 多线程并行
}
```

### 27.19.2 ARM 平台优化



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# ARM 平台的量化优化配置
arm_quant_config = {
    "target": "llvm -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+dotprod",
    "nbit_input": 8,
    "nbit_weight": 8,
    "vectorize_width": 16,  # NEON = 128 bits = 16 INT8
    "use_dotprod": True,    # 使用 SDOT 指令
    "num_threads": 4,        # 大核数量
}
```

### 27.19.3 GPU 平台优化



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
# GPU 平台的量化优化配置
gpu_quant_config = {
    "target": "cuda",
    "nbit_input": 8,
    "nbit_weight": 8,
    "use_tensor_cores": True,  # 使用 Tensor Core
    "block_size": 16,          # Tensor Core 块大小
    "num_warps": 4,            # Warp 数量
}
```

---

## 27.20 总结与进阶

### 27.20.1 量化知识图谱



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
量化知识体系：

基础层
├── 量化数学原理
├── 数据类型（INT8/INT4/FP16/BF16）
└── 量化粒度（per-tensor/per-channel/per-group）

实现层
├── QNN Dialect
├── 校准算法
├── 量化 Pass
└── 代码生成优化

应用层
├── PTQ（训练后量化）
├── QAT（量化感知训练）
├── 混合精度量化
└── 量化 + 剪枝

平台层
├── x86 + VNNI
├── ARM + DotProd
├── GPU + Tensor Core
└── MCU + CMSIS-NN
```

### 27.20.2 进阶学习资源

| 资源 | 类型 | 说明 |
|------|------|------|
| TVM 量化教程 | 官方教程 | `tvm.apache.org/docs/topic/vta/quantize` |
| QNN 论文 | 论文 | "Quantized Neural Networks" (Courbariaux et al.) |
| GEMMLOWP | 库 | Google 的低精度 GEMM 库 |
| NVIDIA TensorRT | 工具 | GPU 量化推理引擎 |
| ARM CMSIS-NN | 库 | ARM MCU 量化算子库 |

---

## 27.21 量化与模型压缩联合优化

### 27.21.1 量化 + 知识蒸馏



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def quantization_with_distillation(teacher_model, student_model,
                                    train_data, quant_config):
    """量化感知的知识蒸馏"""

    # 1. 教师模型（FP32）提供软标签
    teacher_model.eval()

    # 2. 学生模型（量化）学习教师的输出分布
    for epoch in range(num_epochs):
        for batch in train_data:
            # 教师推理
            with torch.no_grad():
                teacher_output = teacher_model(batch)
                teacher_soft = F.softmax(teacher_output / temperature, dim=1)

            # 学生推理（带量化模拟）
            student_output = student_model(batch)
            student_soft = F.softmax(student_output / temperature, dim=1)

            # 损失函数
            # 硬标签损失
            hard_loss = F.cross_entropy(student_output, batch.labels)
            # 软标签损失（KL 散度）
            soft_loss = F.kl_div(
                student_soft.log(), teacher_soft, reduction='batchmean'
            )
            # 总损失
            loss = alpha * hard_loss + (1 - alpha) * soft_loss

            # 反向传播
            loss.backward()
            optimizer.step()
```

### 27.21.2 量化 + 结构化剪枝



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def quantize_with_structured_pruning(mod, params, prune_ratio=0.3):
    """量化与结构化剪枝的联合优化"""

    # 1. 分析通道重要性
    channel_importance = {}
    for name, param in params.items():
        if len(param.shape) == 4:  # Conv 权重
            # 使用 L1 范数作为通道重要性
            importance = np.sum(np.abs(param.numpy()), axis=(1, 2, 3))
            channel_importance[name] = importance

    # 2. 移除不重要的通道
    pruned_params = {}
    prune_masks = {}
    for name, param in params.items():
        if name in channel_importance:
            importance = channel_importance[name]
            threshold = np.percentile(importance, prune_ratio * 100)
            mask = importance >= threshold
            pruned_params[name] = param.numpy()[mask]
            prune_masks[name] = mask
        else:
            pruned_params[name] = param.numpy()

    # 3. 重建模型结构
    mod_pruned = rebuild_model_with_pruned_channels(mod, prune_masks)

    # 4. 量化
    with relay.quantize.qconfig(**quant_config):
        mod_quantized = relay.quantize.quantize(mod_pruned, params=pruned_params)

    return mod_quantized
```

### 27.21.3 量化 + 低秩分解



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def quantize_with_low_rank(mod, params, rank_ratio=0.5):
    """量化与低秩分解的联合优化"""

    decomposed_params = {}
    for name, param in params.items():
        if len(param.shape) == 4:  # Conv 权重 [OC, IC, KH, KW]
            # SVD 分解
            OC, IC, KH, KW = param.shape
            W_2d = param.numpy().reshape(OC, -1)

            U, S, Vt = np.linalg.svd(W_2d, full_matrices=False)

            # 保留前 rank 个奇异值
            rank = int(min(OC, IC * KH * KW) * rank_ratio)

            # 低秩近似
            U_low = U[:, :rank]
            S_low = np.diag(S[:rank])
            Vt_low = Vt[:rank, :]

            # 存储分解后的权重
            decomposed_params[f"{name}_U"] = tvm.nd.array(U_low)
            decomposed_params[f"{name}_S"] = tvm.nd.array(S_low)
            decomposed_params[f"{name}_Vt"] = tvm.nd.array(Vt_low)
        else:
            decomposed_params[name] = param

    # 重建模型
    mod_decomposed = rebuild_with_low_rank(mod, decomposed_params)

    # 量化
    with relay.quantize.qconfig(**quant_config):
        mod_quantized = relay.quantize.quantize(mod_decomposed)

    return mod_quantized
```

---

## 27.22 量化模型的评估指标

### 27.22.1 精度指标



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def evaluate_quantization_quality(fp32_outputs, int8_outputs, labels):
    """评估量化质量"""

    metrics = {}

    # 1. 分类精度
    fp32_preds = np.argmax(fp32_outputs, axis=1)
    int8_preds = np.argmax(int8_outputs, axis=1)

    metrics["fp32_accuracy"] = np.mean(fp32_preds == labels)
    metrics["int8_accuracy"] = np.mean(int8_preds == labels)
    metrics["accuracy_drop"] = metrics["fp32_accuracy"] - metrics["int8_accuracy"]

    # 2. 输出相似度
    metrics["cosine_similarity"] = np.mean([
        cosine_similarity(fp32_outputs[i], int8_outputs[i])
        for i in range(len(fp32_outputs))
    ])

    # 3. 相对误差
    relative_error = np.abs(fp32_outputs - int8_outputs) / (np.abs(fp32_outputs) + 1e-10)
    metrics["mean_relative_error"] = np.mean(relative_error)
    metrics["max_relative_error"] = np.max(relative_error)

    # 4. 信噪比（SNR）
    signal_power = np.mean(fp32_outputs ** 2)
    noise_power = np.mean((fp32_outputs - int8_outputs) ** 2)
    metrics["snr_db"] = 10 * np.log10(signal_power / (noise_power + 1e-10))

    return metrics
```

### 27.22.2 性能指标



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def benchmark_quantized_model(lib, device, num_runs=100):
    """基准测试量化模型性能"""

    module = tvm.contrib.graph_executor.GraphModule(lib["default"](device))

    # 预热
    for _ in range(10):
        module.run()

    # 延迟测试
    latencies = []
    for _ in range(num_runs):
        start = time.time()
        module.run()
        latencies.append((time.time() - start) * 1000)  # ms

    # 吞吐量测试
    batch_sizes = [1, 4, 8, 16, 32]
    throughputs = {}
    for bs in batch_sizes:
        module.set_input("x", np.random.randn(bs, 3, 224, 224).astype("float32"))
        start = time.time()
        for _ in range(50):
            module.run()
        elapsed = time.time() - start
        throughputs[bs] = 50 * bs / elapsed  # samples/s

    metrics = {
        "mean_latency_ms": np.mean(latencies),
        "p50_latency_ms": np.percentile(latencies, 50),
        "p99_latency_ms": np.percentile(latencies, 99),
        "throughput": throughputs,
    }

    return metrics
```

### 27.22.3 模型大小指标



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def measure_model_size(lib, export_path):
    """测量量化模型的大小"""

    # 导出模型
    lib.export_library(export_path)

    # 文件大小
    file_size = os.path.getsize(export_path)

    # 参数大小
    param_size = sum(
        p.numpy().nbytes for p in lib.get_params().values()
    )

    # 代码大小
    code_size = file_size - param_size

    metrics = {
        "total_size_bytes": file_size,
        "total_size_mb": file_size / (1024 * 1024),
        "param_size_bytes": param_size,
        "code_size_bytes": code_size,
        "compression_ratio": "N/A",  # 与 FP32 对比
    }

    return metrics
```

---

## 27.23 量化部署案例研究

### 27.23.1 MobileNet INT8 量化



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def quantize_mobilenet():
    """MobileNet V2 的 INT8 量化"""

    # 1. 加载模型
    model = onnx.load("mobilenetv2-7.onnx")
    mod, params = relay.frontend.from_onnx(
        model, shape_dict={"input": (1, 3, 224, 224)}
    )

    # 2. MobileNet 特定的量化配置
    # 深度可分离卷积对量化更敏感
    quant_config = {
        "nbit_input": 8,
        "nbit_weight": 8,
        "calibrate_mode": "kl",
        "weight_scale": "channel",  # per-channel 量化
        "skip_conv_layers": [],  # 不跳过任何层
    }

    # 3. 量化
    with relay.quantize.qconfig(**quant_config):
        quantized_mod = relay.quantize.quantize(mod, params=params)

    # 4. 编译
    with tvm.target.Target("llvm -mcpu=cortex-a73"):
        lib = relay.build(quantized_mod, target="llvm", params=params)

    return lib
```

### 27.23.2 BERT INT8 量化



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def quantize_bert():
    """BERT 模型的 INT8 量化"""

    # 1. 加载模型
    model = onnx.load("bert-base-uncased.onnx")
    mod, params = relay.frontend.from_onnx(
        model, shape_dict={
            "input_ids": (1, 128),
            "attention_mask": (1, 128),
        }
    )

    # 2. BERT 特定的量化配置
    # 注意力层对量化更敏感
    quant_config = {
        "nbit_input": 8,
        "nbit_weight": 8,
        "calibrate_mode": "percentile",
        "percentile": 99.99,
        "skip_conv_layers": [],  # BERT 没有卷积层
    }

    # 3. 量化
    with relay.quantize.qconfig(**quant_config):
        quantized_mod = relay.quantize.quantize(mod, params=params)

    return quantized_mod
```

### 27.23.3 YOLOv5 INT8 量化



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def quantize_yolov5():
    """YOLOv5 目标检测模型的 INT8 量化"""

    # 1. 加载模型
    model = onnx.load("yolov5s.onnx")
    mod, params = relay.frontend.from_onnx(
        model, shape_dict={"images": (1, 3, 640, 640)}
    )

    # 2. YOLOv5 特定的量化配置
    # 检测头对精度更敏感
    quant_config = {
        "nbit_input": 8,
        "nbit_weight": 8,
        "calibrate_mode": "kl",
        "weight_scale": "channel",
        "skip_conv_layers": ["model.24"],  # 跳过检测头
    }

    # 3. 量化
    with relay.quantize.qconfig(**quant_config):
        quantized_mod = relay.quantize.quantize(mod, params=params)

    return quantized_mod
```

---

## 27.24 量化数学公式推导

### 27.24.1 线性量化数学基础

线性量化（Uniform Quantization）将连续浮点数映射到离散整数，核心是建立浮点值与整数值之间的线性关系。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
基本量化公式:

给定浮点数 x ∈ [α, β]（α = min, β = max）

量化步长（scale）:
  s = (β - α) / (2^n - 1)

  其中 n 是量化位数（如 n=8 表示 INT8）

零点（zero-point）:
  z = round(-α / s)

  对于对称量化: z = 0（当 α = -β 时）
  对于非对称量化: z ≠ 0

量化（float → int）:
  x_q = clamp(round(x / s) + z, 0, 2^n - 1)

  其中 clamp(x, lo, hi) = max(lo, min(hi, x))

反量化（int → float）:
  x = s * (x_q - z)

┌──────────────────────────────────────────────────────────┐
│              量化映射关系图                                │
│                                                          │
│  浮点空间                    整数空间                     │
│                                                          │
│   β ───────────────→ 2^n - 1 (如 255)                    │
│   │                    │                                 │
│   │    x              │   x_q = round(x/s) + z          │
│   │    ↑              │       ↑                          │
│   │    │              │       │                          │
│   α ───────────────→ 0                                   │
│                                                          │
│  对称量化: α = -β, z = 0                                 │
│  ─────────────────────────────                           │
│   -β ─────────────→ 2^(n-1) - 1 (如 127)                │
│    0 ─────────────→ 0                                    │
│    β ─────────────→ -2^(n-1) (如 -128)                   │
│                                                          │
│  注意: 对称量化中负数范围比正数多一个（-128）               │
└──────────────────────────────────────────────────────────┘
```

### 27.24.2 量化算术运算



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
量化乘法推导:

设 x_q, w_q 分别是输入和权重的量化整数值
设 s_x, s_w 分别是对应的 scale
设 z_x, z_w 分别是对应的 zero-point

浮点乘法: y = x * w
  = s_x * (x_q - z_x) * s_w * (w_q - z_w)

令 s_y = s_x * s_w（输出的 scale）

量化乘法:
  y_q = round(y / s_y) + z_y
      = round((x_q - z_x) * (w_q - z_z)) + z_y

  当 z_y = 0（输出也是对称量化）时:
  y_q = (x_q - z_x) * (w_q - z_w)

  这就是 "re-quantization" 的数学基础

量化卷积（Conv2d）:

  浮点: Y[n,c,h,w] = Σ_{k,r,s} X[n,k,h+r,w+s] * W[c,k,r,s]

  量化后:
  Y_q[n,c,h,w] = Σ_{k,r,s} (X_q[n,k,h+r,w+s] - z_x) *
                  (W_q[c,k,r,s] - z_w)

  实现优化: 将减法分离
  Y_q = Σ X_q * W_q
       - z_x * Σ W_q          ← 可预计算
       - z_w * Σ X_q          ← 可预计算
       + z_x * z_w * K*C*R*S  ← 常量

  其中 Σ W_q 和 Σ X_q 可以提前计算，大幅减少运行时开销

量化矩阵乘法（Dense/Linear）:

  浮点: Y[n,m] = Σ_k X[n,k] * W[m,k] + bias[m]

  量化后:
  Y_q[n,m] = Σ_k (X_q[n,k] - z_x) * (W_q[m,k] - z_w) + bias_q[m]

  展开:
  Y_q[n,m] = Σ_k X_q[n,k] * W_q[m,k]    ← 主要计算（INT8 → INT32）
            - z_x * Σ_k W_q[m,k]          ← 权重统计（预计算）
            - z_w * Σ_k X_q[n,k]          ← 输入统计（在线计算）
            + z_x * z_w * K                ← 常量
            + bias_q[m]                     ← 偏置（预量化）
```

### 27.24.3 量化精度损失分析



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
量化误差来源分析:

1. 截断误差（Rounding Error）
   e_round = x_q * s + α - x
   |e_round| ≤ s/2

   当 s 越小（位数越多），截断误差越小

2. 饱和误差（Clipping Error）
   当 x < α 或 x > β 时:
   e_clip = x - clamp(x, α, β)

   解决方案:
   - 使用 percentile 统计（如 99.99%）代替 min/max
   - 使用 KL 散度选择最优截断阈值

3. 累积误差（Accumulated Error）
   多层网络中，误差逐层累积:
   E_total ≈ √(Σ_i e_i²)  （假设误差独立同分布）

   实践中: 越深的网络，量化误差越大
   解决方案: 混合精度量化（敏感层用高精度）

误差敏感度分析:

  层类型          │ 对量化的敏感度 │ 建议精度
  ───────────────┼───────────────┼──────────
  Conv2d (前几层) │ 高            │ INT8 或 FP16
  Conv2d (深层)   │ 中            │ INT8
  Dense/Linear    │ 高            │ INT8 或 FP16
  BatchNorm       │ 低            │ INT8
  ReLU            │ 无影响        │ INT8
  Softmax         │ 高            │ FP16 或 FP32
  Attention       │ 高            │ FP16 或 INT8
  LayerNorm       │ 中            │ FP16
  Embedding       │ 中            │ INT8
```

### 27.24.4 量化训练（QAT）数学



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
直通估计器（Straight-Through Estimator, STE）:

前向传播: 使用量化值
  x_q = Quantize(x)
  y = f(x_q)

反向传播: 假设量化操作的梯度为 1
  ∂L/∂x ≈ ∂L/∂y * ∂y/∂x_q * 1  （STE 近似）

  更精确地说:
  ∂L/∂x = ∂L/∂y * ∂y/∂x_q,  if α ≤ x ≤ β
           0,                  otherwise

  这是因为:
  - 在量化范围内，梯度通过（假设 round 操作梯度为 1）
  - 超出范围的值被 clamp，梯度为 0（死区）

量化参数学习:

  scale 和 zero-point 也可以作为可学习参数:
  s_learnable = exp(log_s)  ← 确保 scale 为正
  z_learnable = sigmoid(z_raw) * (2^n - 1)  ← 确保在有效范围内

  梯度更新:
  ∂L/∂log_s = ∂L/∂x_q * ∂x_q/∂log_s
            = ∂L/∂x_q * (-x/s)  ← 链式法则

混合精度位宽搜索:

  目标: min L(θ, b)  subject to  Memory(b) ≤ B_target

  其中:
  - θ: 模型参数
  - b = [b_1, b_2, ..., b_L]: 每层的位宽
  - B_target: 目标内存大小

  搜索方法:
  1. 固定每层为 2/4/8 bit，穷举组合
  2. 使用强化学习（NAS）搜索最优位宽分配
  3. 基于 Hessian 的敏感度分析（HAWQ 系列）
```

---

## 27.25 QNN 算子注册表

### 27.25.1 QNN 算子概述

QNN（Quantized Neural Network）算子注册表是 TVM 中管理量化算子的核心组件。每个 QNN 算子封装了量化特定的逻辑：requantize、反量化输入、量化输出。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
QNN 算子注册表结构:

┌─────────────────────────────────────────────────────────────┐
│                    QNN Operator Registry                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ qnn.conv2d        │  │ qnn.dense        │                │
│  │ 输入: int8        │  │ 输入: int8        │                │
│  │ 权重: int8        │  │ 权重: int8        │                │
│  │ 输出: int32       │  │ 输出: int32       │                │
│  │ 属性: scale, zp   │  │ 属性: scale, zp   │                │
│  └──────────────────┘  └──────────────────┘                │
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ qnn.add           │  │ qnn.multiply     │                │
│  │ 输入: int8, int8  │  │ 输入: int8, int8  │                │
│  │ 输出: int8        │  │ 输出: int8        │                │
│  │ 属性: scale, zp   │  │ 属性: scale, zp   │                │
│  └──────────────────┘  └──────────────────┘                │
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ qnn.requantize    │  │ qnn.dequantize   │                │
│  │ 输入: int32       │  │ 输入: int8        │                │
│  │ 输出: int8        │  │ 输出: float32     │                │
│  │ 属性: in_s, out_s │  │ 属性: scale, zp   │                │
│  └──────────────────┘  └──────────────────┘                │
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ qnn.concatenate   │  │ qnn.avg_pool2d   │                │
│  │ 输入: int8[]      │  │ 输入: int8        │                │
│  │ 输出: int8        │  │ 输出: int8        │                │
│  │ 属性: scale, zp   │  │ 属性: scale, zp   │                │
│  └──────────────────┘  └──────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### 27.25.2 QNN 算子注册源码



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```cpp
// 文件: src/relay/qnn/op/convolution.cc
// QNN Conv2d 算子注册

// QNN Conv2d 的属性
struct QNNConv2DAttrs : public tvm::AttrsNode<QNNConv2DAttrs> {
  // 输入量化参数
  Array<FloatExpr> input_scale;
  Array<IntExpr> input_zero_point;

  // 权重量化参数
  Array<FloatExpr> kernel_scale;
  Array<IntExpr> kernel_zero_point;

  // 输出量化参数
  Array<FloatExpr> output_scale;
  Array<IntExpr> output_zero_point;

  // 卷积参数
  Array<IndexExpr> strides;
  Array<IndexExpr> padding;
  Array<IndexExpr> dilation;
  int groups;
  String data_layout;
  String kernel_layout;
  String out_layout;

  TVM_DECLARE_ATTRS(QNNConv2DAttrs, "relay.attrs.QNNConv2DAttrs") {
    TVM_ATTR_FIELD(input_scale).describe("Input tensor scale");
    TVM_ATTR_FIELD(input_zero_point).describe("Input tensor zero point");
    TVM_ATTR_FIELD(kernel_scale).describe("Kernel tensor scale");
    TVM_ATTR_FIELD(kernel_zero_point).describe("Kernel tensor zero point");
    TVM_ATTR_FIELD(output_scale).describe("Output tensor scale");
    TVM_ATTR_FIELD(output_zero_point).describe("Output tensor zero point");
    TVM_ATTR_FIELD(strides).set_default(Array<IndexExpr>({1, 1}));
    TVM_ATTR_FIELD(padding).set_default(Array<IndexExpr>({0, 0}));
    TVM_ATTR_FIELD(dilation).set_default(Array<IndexExpr>({1, 1}));
    TVM_ATTR_FIELD(groups).set_default(1);
  }
};

// QNN Conv2d 的类型关系
bool QNNConv2DRel(const Array<Type>& types,
                   int num_inputs,
                   const Attrs& attrs,
                   const TypeReporter& reporter) {
  // types: [data, weight, bias, result]
  CHECK_EQ(types.size(), 4);

  auto data = types[0].as<TensorTypeNode>();
  auto weight = types[1].as<TensorTypeNode>();

  // 验证输入数据类型为 int8
  CHECK(data->dtype == DataType::Int(8))
    << "QNN Conv2d expects int8 input, got " << data->dtype;

  // 验证权重数据类型为 int8
  CHECK(weight->dtype == DataType::Int(8))
    << "QNN Conv2d expects int8 weight, got " << weight->dtype;

  // 计算输出形状（与普通 Conv2d 相同）
  auto param = attrs.as<QNNConv2DAttrs>();
  auto out_shape = ComputeConv2DOutputShape(
      data->shape, weight->shape,
      param->strides, param->padding, param->dilation);

  // 输出类型为 int32（累加结果）
  reporter->Assign(types[3],
                   TensorType(out_shape, DataType::Int(32)));
  return true;
}

// 注册 QNN Conv2d 算子
RELAY_REGISTER_OP("qnn.conv2d")
    .describe("Quantized 2D convolution")
    .set_num_inputs(3)  // data, weight, bias
    .add_argument("data", "Tensor", "The input tensor")
    .add_argument("weight", "Tensor", "The weight tensor")
    .add_argument("bias", "Tensor", "The bias tensor")
    .set_attrs_type_key("relay.attrs.QNNConv2DAttrs")
    .set_support_level(11)
    .add_type_rel("QNNConv2D", QNNConv2DRel);

// QNN Conv2d 的 compute 定义
// 文件: src/relay/qnn/op/convolution.cc
Expr MakeQNNConv2D(Expr data, Expr weight, Expr bias,
                    Array<IndexExpr> strides,
                    Array<IndexExpr> padding,
                    Array<IndexExpr> dilation,
                    int groups,
                    Array<FloatExpr> input_scale,
                    Array<IntExpr> input_zero_point,
                    Array<FloatExpr> kernel_scale,
                    Array<IntExpr> kernel_zero_point,
                    Array<FloatExpr> output_scale,
                    Array<IntExpr> output_zero_point) {
  auto attrs = make_object<QNNConv2DAttrs>();
  attrs->strides = strides;
  attrs->padding = padding;
  attrs->dilation = dilation;
  attrs->groups = groups;
  attrs->input_scale = input_scale;
  attrs->input_zero_point = input_zero_point;
  attrs->kernel_scale = kernel_scale;
  attrs->kernel_zero_point = kernel_zero_point;
  attrs->output_scale = output_scale;
  attrs->output_zero_point = output_zero_point;
  static const Op& op = Op::Get("qnn.conv2d");
  return Call(op, {data, weight, bias}, Attrs(attrs), {});
}
```

### 27.25.3 QNN 到低级算子的降低



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
def lower_qnn_conv2d_to_tir(qnn_call):
    """将 qnn.conv2d 降低为 TIR 算子

    这是 QNN 算子到实际计算的关键转换步骤:
    qnn.conv2d → 低级 INT8 卷积 + requantize
    """

    # 提取参数
    data = qnn_call.args[0]       # int8 输入
    weight = qnn_call.args[1]     # int8 权重
    bias = qnn_call.args[2]       # int32 偏置

    input_scale = qnn_call.attrs.input_scale
    input_zp = qnn_call.attrs.input_zero_point
    kernel_scale = qnn_call.attrs.kernel_scale
    kernel_zp = qnn_call.attrs.kernel_zero_point
    output_scale = qnn_call.attrs.output_scale
    output_zp = qnn_call.attrs.output_zero_point

    # 降低后的计算（伪代码）:
    #
    # for n, c, h, w in output_shape:
    #   acc = 0  # int32 累加器
    #   for k, r, s in kernel:
    #     x_val = int32(data[n,k,h*stride+r, w*stride+s]) - int32(input_zp)
    #     w_val = int32(weight[c,k,r,s]) - int32(kernel_zp)
    #     acc += x_val * w_val  # INT8 * INT8 → INT32
    #
    #   # 加偏置
    #   acc += bias[c]
    #
    #   # Requantize: int32 → int8
    #   # combined_scale = input_scale * kernel_scale / output_scale
    #   combined_scale = input_scale * kernel_scale / output_scale
    #   acc = acc * combined_scale + output_zp
    #   output[n,c,h,w] = clamp(acc, -128, 127)  # int8 范围

    return tvm.tir.PrimFunc(...)
```

### 27.25.4 QNN 算子完整性列表

| QNN 算子 | 输入类型 | 输出类型 | 说明 |
|----------|---------|---------|------|
| `qnn.conv2d` | int8, int8 | int32 | 量化卷积 |
| `qnn.dense` | int8, int8 | int32 | 量化全连接 |
| `qnn.add` | int8, int8 | int8 | 量化加法 |
| `qnn.mul` | int8, int8 | int8 | 量化乘法 |
| `qnn.avg_pool2d` | int8 | int8 | 量化平均池化 |
| `qnn.max_pool2d` | int8 | int8 | 量化最大池化 |
| `qnn.concatenate` | int8[] | int8 | 量化拼接 |
| `qnn.requantize` | int32 | int8 | 重量化 |
| `qnn.dequantize` | int8 | float32 | 反量化 |
| `qnn.quantize` | float32 | int8 | 量化 |
| `qnn.relu` | int8 | int8 | 量化 ReLU |
| `qnn.batch_matmul` | int8, int8 | int32 | 量化批量矩阵乘 |
| `qnn.conv2d_transpose` | int8, int8 | int32 | 量化转置卷积 |
| `qnn.avg_pool3d` | int8 | int8 | 量化 3D 平均池化 |
| `qnn.adaptive_avg_pool2d` | int8 | int8 | 量化自适应池化 |

---

## 27.26 校准算法 KL 散度

### 27.26.1 校准问题定义

量化校准的核心问题是：**如何选择最优的截断阈值 T**，使得量化后的分布与原始分布之间的信息损失最小。



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
校准问题形式化:

给定: 激活值的直方图 H = {h_1, h_2, ..., h_n}
      其中 h_i 是落在第 i 个 bin 中的样本数
      bin 的范围为 [b_i, b_{i+1})

目标: 找到最优阈值 T*，使得:
      T* = argmin_T KL(P || Q_T)

      其中:
      - P 是原始浮点分布
      - Q_T 是使用阈值 T 量化后的分布
      - KL 是 KL 散度（Kullback-Leibler Divergence）

量化范围映射:
  当 T 确定后:
  - 对称量化: [-T, T] → [-127, 127]
  - 非对称量化: [min, T] → [0, 255]

截断策略:
  1. Min-Max: T = max(|x|)  ← 最保守，但可能受离群值影响
  2. Percentile: T = percentile(|x|, p)  ← 如 p=99.99
  3. KL 散度: T = argmin KL(P || Q_T)  ← 最优信息保留
  4. MSE: T = argmin MSE(x, quantize(x))  ← 最小均方误差
```

### 27.26.2 KL 散度计算公式



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
KL 散度定义:

对于离散分布 P 和 Q:
  KL(P || Q) = Σ_i P(i) * log(P(i) / Q(i))

  其中:
  - P(i) = h_i / Σ h_i  （归一化直方图）
  - Q(i) 是量化后的重建分布

KL 散度的直觉:
  - 衡量用 Q 近似 P 时的信息损失
  - KL(P||Q) = 0 当且仅当 P = Q
  - KL(P||Q) ≥ 0（非负性，Gibbs 不等式）

量化后的重建分布 Q_T:

  给定阈值 T，假设量化到 N 个级别（如 N=128 对于对称 INT8）

  1. 将 [-T, T] 均匀分为 N 个区间
  2. 每个区间内的所有值映射到区间中心
  3. 超出 [-T, T] 的值被截断到边界

  具体地:
  - 量化区间宽度: δ = 2T / N
  - 第 k 个区间的中心: c_k = -T + (k + 0.5) * δ
  - 量化后 bin i 的值:
    如果 bin i 完全在 [-T, T] 内:
      Q_T(i) = 区间中心值 * 该区间内的样本数
    如果 bin i 部分超出:
      Q_T(i) = 截断到边界的值 * 截断后的样本数

KL 散度计算步骤:

  1. 收集直方图: 从校准数据集获取激活值的直方图
  2. 候选阈值: 选择一系列候选 T 值
  3. 对每个 T:
     a. 将直方图分为 "在范围内" 和 "截断" 两部分
     b. 在范围内的部分: 均匀量化到 N 个级别
     c. 截断的部分: 合并到边界级别
     d. 计量 KL(P || Q_T)
  4. 选择使 KL 散度最小的 T
```

### 27.26.3 KL 散度校准实现



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
import numpy as np
from scipy import stats

def compute_kl_divergence(p, q):
    """计算两个离散分布之间的 KL 散度

    参数:
        p: 原始分布 (归一化直方图)
        q: 量化重建分布 (归一化直方图)

    返回:
        KL(P || Q) 散度值
    """
    # 避免 log(0)
    epsilon = 1e-10
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)

    # 重新归一化
    p = p / p.sum()
    q = q / q.sum()

    # KL 散度: Σ p_i * log(p_i / q_i)
    kl = np.sum(p * np.log(p / q))
    return kl


def calibrate_kl(activations, num_bins=2048, num_quantiles=128):
    """KL 散度校准算法

    算法流程:
    1. 收集校准数据集上所有激活值
    2. 构建直方图
    3. 搜索最优截断阈值

    参数:
        activations: 校准数据集上的激活值列表
        num_bins: 直方图 bin 数量
        num_quantiles: 量化级别数（对称 INT8: 128）

    返回:
        最优阈值 T
    """

    # 1. 收集所有激活值
    all_values = []
    for act in activations:
        all_values.extend(act.flatten())
    all_values = np.array(all_values)

    # 2. 构建直方图
    # 只考虑绝对值（对称量化）
    abs_values = np.abs(all_values)
    max_val = abs_values.max()

    # 创建直方图: [0, max_val] 分为 num_bins 个 bin
    hist, bin_edges = np.histogram(abs_values, bins=num_bins,
                                    range=(0, max_val))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 3. 搜索最优阈值
    # 候选阈值: 每个 bin 的右边界
    best_kl = float('inf')
    best_threshold = max_val

    # 从 128 个 bin 开始搜索（至少需要 num_quantiles 个 bin）
    for i in range(num_quantiles, num_bins):
        threshold = bin_edges[i + 1]

        # 构建原始分布 P
        # 范围内的直方图 + 范围外的截断
        p_hist = hist.copy().astype(float)

        # 构建量化重建分布 Q
        # 将 [0, threshold] 分为 num_quantiles 个区间
        q_hist = np.zeros(num_bins)

        # 范围内的部分: 均匀量化
        bin_width = threshold / num_quantiles
        for k in range(num_quantiles):
            # 第 k 个量化区间的范围
            lo = k * bin_width
            hi = (k + 1) * bin_width
            center = (lo + hi) / 2

            # 统计落在这个区间内的原始样本
            mask = (bin_edges[:-1] >= lo) & (bin_edges[1:] <= hi)
            count = hist[mask].sum()

            # 找到 center 所在的 bin
            center_bin = int(center / max_val * num_bins)
            center_bin = min(center_bin, num_bins - 1)
            q_hist[center_bin] += count

        # 范围外的部分: 截断到边界
        overflow_mask = bin_edges[:-1] >= threshold
        overflow_count = hist[overflow_mask].sum()
        # 截断到最大量化值
        boundary_bin = num_bins - 1
        q_hist[boundary_bin] += overflow_count

        # 计算 KL 散度
        kl = compute_kl_divergence(p_hist, q_hist)

        if kl < best_kl:
            best_kl = kl
            best_threshold = threshold

    print(f"KL 校准完成: 最优阈值 = {best_threshold:.4f}, "
          f"KL 散度 = {best_kl:.6f}")

    return best_threshold


def calibrate_percentile(activations, percentile=99.99):
    """Percentile 校准（更简单但次优）

    参数:
        activations: 激活值列表
        percentile: 百分位数

    返回:
        阈值 T
    """
    all_values = np.concatenate([act.flatten() for act in activations])
    threshold = np.percentile(np.abs(all_values), percentile)
    return threshold


def calibrate_mse(activations, num_bins=2048, num_quantiles=128):
    """MSE 校准: 最小化均方误差

    目标: T* = argmin_T E[(x - Q(x))^2]
    """
    all_values = np.concatenate([act.flatten() for act in activations])
    abs_values = np.abs(all_values)
    max_val = abs_values.max()

    best_mse = float('inf')
    best_threshold = max_val

    for threshold in np.linspace(max_val / num_bins, max_val, num_bins):
        # 量化
        scale = threshold / num_quantiles
        quantized = np.round(abs_values / scale)
        quantized = np.clip(quantized, 0, num_quantiles)
        dequantized = quantized * scale

        # MSE
        mse = np.mean((abs_values - dequantized) ** 2)

        if mse < best_mse:
            best_mse = mse
            best_threshold = threshold

    return best_threshold
```

### 27.26.4 校准算法对比



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```
┌──────────────────────────────────────────────────────────────────┐
│                    校准算法对比                                    │
├────────────┬──────────────┬────────────┬──────────────────────────┤
│ 算法        │ 速度         │ 精度       │ 适用场景                  │
├────────────┼──────────────┼────────────┼──────────────────────────┤
│ Min-Max    │ O(n) 最快    │ 最差       │ 快速原型验证              │
│            │              │            │ 离群值少的场景            │
├────────────┼──────────────┼────────────┼──────────────────────────┤
│ Percentile │ O(n log n)   │ 中等       │ 大多数 CNN 模型           │
│ (99.99%)   │              │            │ 离群值较多的场景          │
├────────────┼──────────────┼────────────┼──────────────────────────┤
│ KL 散度    │ O(n * B)     │ 最优       │ 对精度要求高的场景        │
│            │ B=bin 数量   │            │ NLP/Transformer 模型     │
├────────────┼──────────────┼────────────┼──────────────────────────┤
│ MSE        │ O(n * T)     │ 接近最优   │ 计算资源充足的场景        │
│            │ T=候选阈值数 │            │ 需要客观指标的场景        │
├────────────┼──────────────┼────────────┼──────────────────────────┤
│ ACIQ       │ O(n)         │ 好         │ 快速校准 + 高精度         │
│ (分析法)    │              │            │ 假设激活服从高斯分布      │
└────────────┴──────────────┴────────────┴──────────────────────────┘

KL 散度算法的直觉:
  - 直方图 P 代表 "真实世界" 的激活值分布
  - 量化后的 Q 代表 "用有限级别表示" 的近似分布
  - KL(P||Q) 衡量 "用 Q 代替 P 会损失多少信息"
  - 最小化 KL = 最大化信息保留 = 最优量化
```

### 27.26.5 校准流程集成



**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。
```python
class TVMCalibrator:
    """TVM 量化校准器集成类"""

    def __init__(self, mod, params, calibration_data,
                 method="kl", num_samples=100):
        self.mod = mod
        self.params = params
        self.calibration_data = calibration_data[:num_samples]
        self.method = method

        # 收集每层的激活值
        self.activation_histograms = {}

    def collect_activations(self):
        """通过 Hook 收集每层的激活值"""

        # 创建带 Hook 的模块
        # 使用 TVM 的 debug executor 收集中间结果
        from tvm.contrib import graph_executor

        # 编译 debug 版本
        target = "llvm"
        with tvm.transform.PassContext(opt_level=0):
            lib = relay.build(self.mod, target=target, params=self.params)

        # 创建执行器
        dev = tvm.device(target, 0)
        runtime = graph_executor.GraphModule(lib["default"](dev))

        # 执行校准数据，收集统计信息
        for i, data in enumerate(self.calibration_data):
            runtime.set_input("data", tvm.nd.array(data))
            runtime.run()

            # 获取中间层输出
            for j in range(runtime.get_num_outputs()):
                output = runtime.get_output(j).numpy()
                layer_name = f"layer_{j}"
                if layer_name not in self.activation_histograms:
                    self.activation_histograms[layer_name] = []
                self.activation_histograms[layer_name].append(output)

            if i % 10 == 0:
                print(f"收集校准数据: {i}/{len(self.calibration_data)}")

    def calibrate(self):
        """执行校准，返回每层的量化参数"""
        self.collect_activations()

        calibration_params = {}
        for layer_name, activations in self.activation_histograms.items():
            if self.method == "kl":
                threshold = calibrate_kl(activations)
            elif self.method == "percentile":
                threshold = calibrate_percentile(activations)
            elif self.method == "mse":
                threshold = calibrate_mse(activations)
            else:
                threshold = calibrate_percentile(activations, percentile=99.9)

            scale = threshold / 127.0  # INT8 对称量化
            calibration_params[layer_name] = {
                "scale": scale,
                "zero_point": 0,
                "threshold": threshold,
            }
            print(f"  {layer_name}: scale={scale:.6f}, "
                  f"threshold={threshold:.4f}")

        return calibration_params
```


**代码解读（正文补充）**：

- **业务意义**：上面的片段不是孤立 API 演示，而是在说明本章主题如何落到真实模型部署流程中；阅读时应关注输入、输出、目标硬件和运行时边界，而不是只记住函数名。
- **TVM 内部机制**：该片段通常会触发前端导入、IR 构造、Pass 优化、Lowering、CodeGen 或 Runtime 调用中的一个或多个阶段；如果结果异常，应优先判断问题发生在哪一层 IR，而不是直接修改最终生成代码。
- **适用场景**：它适合用于教学验证、最小复现、性能基线建立或单一优化点确认；在生产环境中还需要补充版本锁定、输入分布校验、目标设备基准测试和回滚策略。
- **限制条件**：示例默认省略了异常处理、数据集规模、硬件差异、编译缓存和长期维护成本；当模型包含动态形状、自定义算子或厂商私有后端时，需要结合本章后续工程经验进行扩展。

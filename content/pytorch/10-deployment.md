---
title: "Chapter 10. 生产级部署"
description: "使用 TorchScript 和 ONNX 跨越 Python 与 C++ 的鸿沟，以及模型量化压缩技术"
updated: "2026-01-22"
---

# Chapter 10. 生产级部署 (Deployment)

> **Learning Objectives**
> *   理解 **Tracing** (Trace execution) 与 **Scripting** (Parse AST) 的本质区别。
> *   使用 `torch.onnx.export` 导出标准格式模型。
> *   通过 **Quantization** (量化) 将模型体积压缩 75% 并加速推理。

---

## 10.1 为什么要离开 Python？

PyTorch 的 **Eager Mode** 依赖 Python 解释器。但在生产环境（如嵌入式设备、高并发 C++ 服务）中，Python 可能太慢或不可用。
我们需要一种**中间表示 (IR)**，它独立于 Python，能在 C++ 运行时 (LibTorch) 或其他硬件 (TensorRT, FPGA) 上跑。

---

## 10.2 TorchScript: PyTorch 的亲儿子

TorchScript 有两种创建方式。这是一个面试常考题，也是初学者的噩梦。

### 交互演示：Tracing vs Scripting

*   **Tracing (jit.trace)**: 像**录像机**。它运行一遍代码，记下经过的路径。
    *   **缺点**: 如果代码里有 `if x > 0`，录像时只走了 `True` 分支，那么生成的图就**永远**只包含 `True` 分支。
*   **Scripting (jit.script)**: 像**翻译机**。它直接解析 Python 源代码 (AST)，能理解 `if-else`。

可以尝试在下方的 Interactive Component 中切换模式，观察对于包含 `if-else` 的代码，Tracing 会如何**丢失逻辑**。

<div data-component="TorchScriptVisualizer"></div>

### 10.2.1 代码实战

```python
import torch

class MyModule(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x * 2
        else:
            return x + 1

model = MyModule()
example_input = torch.rand(3, 3)

# === 方式 1: Tracing (推荐优先尝试) ===
# 优点: 兼容性好，支持大部分 Python 语法
# 缺点: 无法处理动态控制流 (Control Flow)
traced_model = torch.jit.trace(model, example_input)
# ⚠️ 警告: 如果输入改变导致 x.sum() <= 0，traced_model 依然会执行 x * 2！

# === 方式 2: Scripting ===
# 优点: 完美支持 Control Flow
# 缺点: 对 Python 语法限制极多 (它是 Python 的子集)
scripted_model = torch.jit.script(model)

# === 保存 ===
scripted_model.save("model.pt")

# === C++ 载入 ===
# auto module = torch::jit::load("model.pt");
```

---

## 10.3 ONNX: AI 界的 PDF

ONNX (Open Neural Network Exchange) 旨在打破框架壁垒。

```python
torch.onnx.export(
    model,               # 模型
    example_input,       # 用于推导形状的伪输入
    "model.onnx",        # 输出路径
    opset_version=13,    # 算子集版本 (越新支持的算子越多)
    do_constant_folding=True, # 常量折叠优化
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},  # 声明第 0 维是动态的
        "output": {0: "batch_size"}
    }
)
```

**最佳实践**: 部署到 NVIDIA GPU 时，通常路径是 **PyTorch -> ONNX -> TensorRT**。

---

## 10.4 模型量化 (Quantization)

你真的需要 32 位浮点数吗？
**INT8 量化**将数值范围从 `[-3.4e38, 3.4e38]` 压缩到 `[-128, 127]`。

### 交互演示：FP32 vs INT8

*   **FP32 (蓝色)**: 连续分布，占用 4MB。
*   **INT8 (绿色)**: 离散分布，占用 1MB。
可以看到 INT8 是对 FP32 的近似采样。对于深度学习网络，这种近似通常只会带来极其微小的精度损失(<1%)。

<div data-component="QuantizationVisualizer"></div>

### 10.4.1 动态量化 (Dynamic Quantization) 代码

最简单，无需重新训练，特别适合 LSTM/Transformer (权重很大)。

```python
import torch.quantization

quantized_model = torch.quantization.quantize_dynamic(
    model, 
    {torch.nn.Linear, torch.nn.LSTM}, # 指定要量化的层类型
    dtype=torch.qint8                 # 目标类型
)

# 看看效果
print(f"Size before: {os.path.getsize('model.pt')/1e6} MB")
print(f"Size after:  {os.path.getsize('q_model.pt')/1e6} MB")
```

---

## 10.5 本章小结

*   需要 C++ 部署 -> **TorchScript**。
*   需要 TensorRT/OpenVINO 加速 -> **ONNX**。
*   模型包含复杂 if-else -> 用 **jit.script**。
*   模型只是简单堆叠 -> 用 **jit.trace**。
*   移动端对体积敏感 -> 上 **Quantization**。

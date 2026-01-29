---
title: "Chapter 3. 神经网络核心组件"
description: "详解 nn.Module 生命周期、常用层、Sequential 容器以及模型参数管理"
updated: "2026-01-22"
---

> **Learning Objectives**
> *   掌握 `nn.Module` 的生命周期与骨架代码。
> *   直观理解激活函数 (`ReLU`, `Sigmoid`, `GELU`) 的特性。
> *   学会使用 `Linear`, `Conv2d`, `BatchNorm` 搭建模型。
> *   使用 `nn.Sequential` 快速构建流水线。

---

## 3.1 万物皆 Module

在 PyTorch 中，所有的层（Layer）、模型（Model）、甚至部分损失函数（Loss），都继承自 `torch.nn.Module`。
这赋予了它们嵌套的能力：一个模型可以包含多个子模型，子模型又包含多个层。

### 3.1.1 核心骨架 (Template)

```python
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 定义部件 (LEGO Bricks)
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.fc = nn.Linear(32 * 26 * 26, 128)
        
    def forward(self, x):
        # 2. 定义组装逻辑 (Assembly)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.fc(torch.flatten(x, 1))
        return x
```

> [!IMPORTANT]
> **Magic Method**:
> `model(x)` 实际上调用了 `__call__`，它在执行 `forward` 前后会运行 registered hooks。**永远不要显式调用 .forward(x)**。

---

## 3.2 激活函数 (Activation Functions)

激活函数为神经网络引入了**非线性**。如果没有它们，深层网络就会退化为单层线性变换。

### 交互演示：激活函数图鉴

拖动滑块，对比不同激活函数的输出与特性。
*   **ReLU**: 最常用，计算快，但在负区间梯度为 0 (Dead ReLU)。
*   **Sigmoid**: 经典的 S 曲线，但两端梯度几乎为 0 (梯度消失)，目前仅用于输出层。
*   **GELU**: BERT/GPT 的标配，比 ReLU 更平滑，允许微小的负值。

<div data-component="ActivationVisualizer"></div>

---

## 3.3 常用层详解

### 3.3.1 全连接层 (Linear)

$$ y = xA^T + b $$
最基础的层。输入输出特征数固定。
**坑点**: 输入 Tensor 的最后一维必须等于 `in_features`。

### 3.3.2 卷积层 (Conv2d)

提取图像的空间特征。重点参数：`kernel_size` (卷积核大小), `stride` (步长), `padding` (填充)。
计算公式：$Output = \frac{Input - Kernel + 2 \times Padding}{Stride} + 1$

### 3.3.3 归一化层 (BatchNorm / LayerNorm)

*   **BatchNorm**: 沿着 Batch 维度归一化。对 CV 任务有效。
*   **LayerNorm**: 沿着 Feature 维度归一化。对 NLP (Transformer) 任务有效。
*   **重要**: `BatchNorm` 在训练和推理时行为不同，**务必切换 `model.train()` / `model.eval()`**。

---

## 3.4 容器：nn.Sequential

对于简单的“一条路走到黑”的模型，`nn.Sequential` 是神器。

```python
model = nn.Sequential(
    nn.Conv2d(1, 32, 3),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(32 * 26 * 26, 10)
)
```

### 交互演示：数据流向可视化

很多新手在写 `Sequential` 时，最头疼的就是**形状匹配**（上一层的 Output 必须等于下一层的 Input）。
下图展示了一个 Tensor 如何在不同层之间“变形”。

<div data-component="SequentialFlowVisualizer"></div>

---

## 3.5 管理参数

`nn.Module` 会自动扫描所有成员变量，将类型为 `nn.Parameter` 或子 `nn.Module` 的属性注册为参数。

```python
# 遍历参数
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.shape)

# 获取 State Dict (用于保存模型)
sd = model.state_dict()
```

---

## 3.6 本章小结

*   `nn.Module` 是核心基类，必须实现 `__init__` 和 `forward`。
*   **激活函数**的选择至关重要，默认首选 **ReLU** 或 **GELU**。
*   **BatchNorm** 需要区分 train/eval 模式。
*   **Sequential** 适合快速搭建简单模型，复杂拓扑 (ResNet) 需自定义 Class。

> [!TIP]
> **思考题**:
> 为什么 `nn.functional.relu` (函数) 和 `nn.ReLU` (类) 同时存在？应该用哪个？
> (答案：Layer 类适合放在 `Sequential` 中；函数 API 适合在 `forward` 中灵活调用，无需在 `__init__` 实例化状态。)

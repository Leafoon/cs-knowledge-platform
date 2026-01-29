---
title: "Chapter 9. 高阶函数式 API"
description: "掌握 torch.func, vmap 并行加速以及使用 Hooks 黑魔法修改模型行为"
updated: "2026-01-22"
---

> **Learning Objectives**
> *   了解 PyTorch 的两种面孔：OO (`nn.Module`) vs Functional (`torch.func`).
> *   **Deep Dive**: 为什么 `vmap` 能消除 Python 循环并加速。
> *   使用 **Hooks** (`register_forward_hook`) 进行特征提取和梯度修改。
> *   理解 Eager Mode 的优势与代价。

---

## 9.1 面向对象 (OO) 只是表象

PyTorch 表面上是面向对象的（定义 `class Net(nn.Module)`），但其底层内核（Aten）是函数式的。
在做 **Meta-Learning** (MAML) 或 **Ensembling** 时，我们需要更纯粹的函数式操作。

**PyTorch 2.0 更新**: `torch.func` (受到 Google JAX/Functorch 启发) 现已成为官方标准库。

```python
# 传统的 Statelful 方式
model = Net()
out = model(x)

# 新的 Functional (Stateless) 方式
# 把参数提取出来，把模型变成一个纯函数 f(params, x) -> y
from torch.func import functional_call
params = dict(model.named_parameters())
out = functional_call(model, params, x)
```

为什么要这么麻烦？为了下面的黑魔法 —— `vmap`。

---

## 9.2 vmap: 自动向量化 (Vectorizing Map)

假设你想计算 Batch 中**每一个样本**的梯度（Per-sample Gradients）。
*   **方法 A (Loop)**: 写一个 `for` 循环，每个样本 Backward 一次。-> **巨慢**，无法利用 GPU 并行。
*   **方法 B (vmap)**: 使用 `vmap` 自动将“处理单样本的函数”转换为“处理 Batch 的函数”。-> **极快**。

### 交互演示：Loop vs Vectorized

点击运行，直观感受 `vmap` 是如何将串行任务并行化的。

<div data-component="ParallelVisualizer"></div>

**代码详解**:

```python
import torch
from torch.func import vmap, grad

# 1. 定义针对“单个样本”的 Loss 计算逻辑
def compute_loss_single(params, sample, target):
    pred = functional_call(model, params, sample)
    return torch.nn.functional.mse_loss(pred, target)

# 2. 我们想计算梯度：grad(loss)
compute_grad_single = grad(compute_loss_single)

# 3. 魔法时刻：使用 vmap 自动批量化
# in_dims=(None, 0, 0): 
#   - params: None (即不在 batch 维度上切片，所有样本共享同一套参数)
#   - sample: 0 (输入数据的第0维是 batch 维度)
#   - target: 0 (标签的第0维是 batch 维度)
compute_sample_grads = vmap(compute_grad_single, in_dims=(None, 0, 0))

# 4. 执行
# 现在的 per_sample_grads 包含了一个 Batch 的梯度，形状是 [Batch, Param_Dim]
per_sample_grads = compute_sample_grads(params, batch_samples, batch_targets)
```

---

## 9.3 Hooks: 给模型装个“后门”

有时候我们想拿到中间层（比如 `conv1`）的输出做可视化（Feature Map Visualization），或者想修改中间的梯度（Gradient Clipping）。
直接改 `forward` 代码太侵入式了。**Hooks** 允许我们在不碰源码的情况下“监听”数据流。

### 交互演示：Hooks 拦截流

观察数据流经 Layer 时，Hook 是如何触发并将数据副本（Activation）保存到外部字典的。
在 Layer 的输出端（红点），可以清晰地看到数据被 Hook 截获。

<div data-component="HookVisualizer"></div>

**实战代码：提取中间层特征 (Feature Extraction)**

```python
activation_dict = {}

# 定义 Hook 函数：签名必须是 (module, input, output)
def get_activation(name):
    def hook(model, input, output):
        # .detach() 很重要！否则会保留计算图，导致显存爆炸 (Memory Leak)
        activation_dict[name] = output.detach() 
    return hook

# 注册：告诉 PyTorch，“我要监听 model.conv1 这一层”
# handle 是一个句柄，用来以后移除这个 hook
handle = model.conv1.register_forward_hook(get_activation('conv1'))

# 正常前向传播
output = model(input_img)

# 此时字典里已经有了我们要的数据
print(activation_dict['conv1'].shape) # e.g., [1, 64, 112, 112]

# 用完记得移除，否则 hook 会一直存在
handle.remove()
```

---

## 9.4 总结

*   **vmap** 让你能够用写“单样本代码”的逻辑，自动获得“Batch 并行”的性能。
*   **Hooks** 是 PyTorch 这类动态图框架的神技，用于 Debug、可视化或魔改梯度。
*   **Eager Mode** (默认模式) 虽然方便调试，但因为过度依赖 Python 解释器调度，性能有天花板。这也是为什么在生产环境我们要用 `torch.compile`（后文详述）。

> [!TIP]
> **思考题**:
> 如果我在 `hook` 函数里修改了 `output`（例如 `return output * 2`），后续层的输入会变吗？
> (答案：会！Hook 不仅能读，还能写。这就是为什么它能用来做“中间层干预”。)

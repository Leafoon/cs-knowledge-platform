---
title: "Chapter 2. 自动微分：PyTorch 的灵魂"
description: "深度解构 Autograd 引擎、计算图构建过程、反向传播与梯度钩子"
updated: "2026-01-22"
---

> **Learning Objectives**
> *   理解“计算图 (Computational Graph)”是如何动态构建的。
> *   掌握 `requires_grad` 和 `.backward()` 的核心机制。
> *   学会控制梯度流：停止梯度 (`no_grad`)、梯度累积与清零。
> *   (进阶) 理解叶子节点 (Leaf Node) 与非叶子节点的区别。

---

## 2.1 什么是自动微分？

训练神经网络的核心是**优化**，而优化的核心是**梯度**。手动推导复杂网络的梯度（链式法则）几乎是不可能的。
PyTorch 的 `torch.autograd` 是一个“录像机”。它会“记录”你的所有操作，构建一张**有向无环图 (DAG)**，然后通过链式法则自动反向传播梯度。

### 交互演示：动态计算图

下图展示了一个简单的计算过程：$y = (x + w) \times 2$。
这是一个非常直观的动态过程：
1.  **Forward**: 数据从输入流向输出，沿途构建图。
2.  **Backward**: 梯度从输出流回输入 (Backprop)，利用链式法则逐层相乘。

<div data-component="ComputationalGraph"></div>

---

## 2.2 核心机制：requires_grad

如何告诉 PyTorch 哪些变量需要计算梯度？

```python
x = torch.tensor([3.0], requires_grad=True)
```

*   **`requires_grad=True`**: 开启“录像模式”。PyTorch 会追踪在此 Tensor 上发生的所有操作。
*   **`is_leaf`**: 用户创建的 Tensor 通常是**叶子节点**（图中蓝色的 x, w）。由运算产生的 Tensor（如 `a = x+w`）是非叶子节点。

> [!IMPORTANT]
> **为什么要区分叶子节点？**
> 为了节省显存，反向传播结束后，**只有叶子节点的梯度会被保留** (`x.grad`)。中间节点的梯度 (`a.grad`) 会被自动释放。

---

## 2.3 反向传播：.backward()

调用 `.backward()` 后，PyTorch 会利用链式法则计算梯度，并将结果**累加**到各个叶子节点的 `.grad` 属性中。

```python
# 1. 前向计算
y = (x + w) * 2

# 2. 反向传播
y.backward()

# 3. 访问梯度
print(f"x.grad: {x.grad}")    # dy/dx = 2
```

### 梯度累积陷阱 (Acumulation Trap)

**必须记住**：`grad` 属性是累加的，而不是覆盖。
这意味着在每个训练 Batch 开始前，你必须手动清零梯度。

```python
# ❌ 错误做法：没有清零
for i in range(3):
    y.backward()
    print(x.grad) 
    # 输出: 2, 4, 6 (梯度一直在变大！)

# ✅ 正确做法
for i in range(3):
    x.grad.zero_() # 或者 optimizer.zero_grad()
    y.backward()
    print(x.grad)
    # 输出: 2, 2, 2
```

---

## 2.4 控制梯度流

在推理（Inference）或验证（Validation）阶段，我们不需要计算梯度。关闭 Autograd 可以显著减少显存占用并加速计算。

### 2.4.1 torch.no_grad()

```python
# 上下文管理器，作用域内产生的 Tensor 均无梯度
with torch.no_grad():
    y_pred = model(x_test)
```

### 2.4.2 .detach()

创建一个新的 Tensor，与原 Tensor 共享数据，但**切断**与计算图的联系。常用于将 Loss 传入 Metric 计算函数，或传入 Matplotlib 绘图。

```python
plt.plot(loss.detach().numpy())
```

---

## 2.5 进阶：Vector-Jacobian Product

通常我们对标量 Loss 调用 `.backward()`。
如果输出是一个向量（比如 `y = [y1, y2]`），PyTorch 不知道该对哪个元素求导。此时你需要传入一个与 `y` 同形状的权重向量 `gradient`。

这在多任务学习 (Multi-task Learning) 中非常有用：可以给不同任务的 Loss 分配不同的权重。

```python
y.backward(torch.tensor([1.0, 0.5])) # 任务1权重1.0，任务2权重0.5
```

---

## 2.6 本章小结

*   `requires_grad=True` 是开启 Autograd 的开关。
*   计算图在 Forward 时动态构建，Backward 时销毁。
*   梯度是**累加**的，务必使用 `zero_grad()`。
*   叶子节点保留梯度，中间节点释放梯度。

> [!TIP]
> **思考题**：
> 已知 ReLU 函数在 $x=0$ 处不可导。PyTorch 是如何处理 `F.relu(0)` 的梯度的？
> (答案：PyTorch 通常人为规定 $x=0$ 处的梯度为 0 或 0.5，实际上在 sub-gradient 环境下取值通常为 0。)

---
title: "Chapter 1. 张量：PyTorch 的基石"
description: "深入理解 Tensor 内存模型、形状变换与广播机制"
updated: "2026-01-22"
---

# Chapter 1. 张量：PyTorch 的基石 (Tensors)

> **Learning Objectives**
> *   彻底理解 Tensor 的 **Storage** 与 **View** 的区别。
> *   熟练通过 `stride` 分析形状变换的零拷贝特性。
> *   掌握 **Broadcasting** (广播) 机制，避免隐式 Bug。
> *   熟悉常用的 Indexing 与 Slicing 操作。

---

## 1.1 什么是 Tensor？

在数学上，Tensor 是标量、向量、矩阵的高维推广。
在计算机科学（PyTorch）中，Tensor 是一个多维数组，支持 GPU 加速和自动微分。

```python
import torch

# 标量 (0-d tensor)
scalar = torch.tensor(7)

# 向量 (1-d tensor)
vector = torch.tensor([7, 7])

# 矩阵 (2-d tensor)
matrix = torch.tensor([[7, 8], [9, 10]])
```

---

## 1.2 核心概念：Storage vs View

这是 PyTorch 高效的核心秘密。
*   **Storage**: 实际存储数据的连续一维内存块（通常由于 C 语言实现）。
*   **View**: 定义了我们如何“看”这块内存（Shape, Stride, Offset）。

当我们执行 `transpose`、`view`、`slice` 等操作时，**PyTorch 并没有复制数据**，只是修改了 View 的元数据 (Stride)。

### 交互演示：Storage 与 Stride

随意切换 View 模式，观察下方的物理内存存储（Physical Memory）是否发生变化。
你会发现，**数据从未移动**，变的只是索引方式。

<div data-component="TensorStorageVisualizer"></div>

### 代码验证

```python
x = torch.arange(9).reshape(3, 3)
y = x.t() # 转置

# 它们共享内存地址！
print(x.storage().data_ptr() == y.storage().data_ptr()) # True

# 修改 y，x 也会变！
y[0, 1] = 999
print(x[1, 0]) # 999
```

> [!CAUTION]
> **View 必须连续**:
> 某些操作（如 `transpose`）产生的 Tensor 在内存中是不连续的 (non-contiguous)。
> 此时直接调用 `.view()` 会报错。需要先调用 `.contiguous()`（这会触发数据复制）。

---

## 1.3 广播机制 (Broadcasting)

广播允许你在不同形状的 Tensor 之间进行数学运算。这是深度学习代码简洁的关键，也是 Bug 的温床。

**规则**:
1.  从后向前对齐维度。
2.  如果不匹配，但其中一个维度是 1，则将该维度“复制”（Expand）以匹配另一个。
3.  如果都不匹配且都不是 1，报错。

### 交互演示：矩阵加向量

演示 `(3x1)` 的矩阵加上 `(1x3)` 的向量。
注意观察**虚线部分**：这是 PyTorch 自动补全（Broadcast）出来的虚拟数据。

<div data-component="TensorBroadcastingVisualizer"></div>

### 实战：如何避免广播 Bug？

隐式广播有时非常危险。

```python
# 预期：Element-wise 相加
a = torch.randn(5)    # shape: [5]
b = torch.randn(5, 1) # shape: [5, 1]

# 结果：Broadcast 成了 [5, 5] 矩阵！这通常不是你想要的。
c = a + b 
print(c.shape) # torch.Size([5, 5])
```

> [!TIP]
> **Best Practice**:
> 在处理复杂的维度变换时，养成手动 check shape 的习惯。
> 使用 `assert x.shape == (B, C, H, W)` 进行防御性编程。

---

## 1.4 常用操作速查

### 创建
*   `torch.zeros(3, 3)`
*   `torch.randn(3, 3)` (标准正态分布)
*   `torch.tensor([1, 2, 3])` (从 List)
*   `torch.from_numpy(np_array)` (共享内存)

### 形状变换
*   `x.view(B, -1)`: 展平 (Zero-copy)
*   `x.reshape(B, -1)`: 万能 (可能 Copy)
*   `x.permute(0, 3, 1, 2)`: 维度重排 (如 NCHW -> NHWC)
*   `x.unsqueeze(0)`: 增加维度 [3, 3] -> [1, 3, 3]
*   `x.squeeze()`: 压缩维度 [1, 3, 3] -> [3, 3]

### 索引
*   `x[:, 0]`: 取第 0 列
*   `x[x > 0]`: Mask Indexing (布尔索引)

---

## 1.5 本章小结

*   Tensor 是 PyTorch 的数据载体。
*   理解 **Stride** 才能理解为什么有些操作是 Zero-copy 的。
*   **Broadcasting** 是一把双刃剑，使用前需确保心中有数。

> [!TIP]
> **下一步**:
> 有了 Tensor，我们如何让机器自动计算梯度？
> 进入 [Chapter 2. 自动微分引擎](02-autograd.md)，我们将探索深度学习的核心魔法——Autograd。

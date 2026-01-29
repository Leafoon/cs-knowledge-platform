---
title: "Chapter 5. 优化器与训练循环"
description: "直观对比优化器算法，构建健壮的标准训练循环"
updated: "2026-01-22"
---

> **Learning Objectives**
> *   理解 **Loss Surface** 的概念。
> *   直观对比 **SGD** vs **Momentum** vs **Adam** 的寻优路径。
> *   掌握标准训练循环的 Code Snippet。
> *   学习如何使用 `lr_scheduler` 进行学习率衰减。

---

## 5.1 损失函数：导航的指南针

Loss Function 定义了我们离目标还有多远。
*   **Regression**: MSE (L2), MAE (L1), Huber.
*   **Classification**: CrossEntropy (Softmax + NLL), BCEWithLogits (Sigmoid + BCE).

---

## 5.2 优化器：如何下山？

如果把 Loss Surface 想象成连绵起伏的山脉，优化器的任务就是这就从山顶（高 Loss）走到山谷（低 Loss）。

### 交互演示：优化器寻路对比

下方的 2D 椭圆地形模拟了一个狭长的山谷（深度学习中非常常见的 Loss 地形）。
观察三种优化器的表现：
1.  **SGD (红色)**: 沿着梯度垂直方向走，在狭长山谷中剧烈震荡 ("Zig-Zag")，收敛极慢。
2.  **Momentum (橙色)**: 引入动量，积累速度，能冲过震荡，收敛更快。
3.  **Adam (绿色)**: 自适应调整每个维度的步长。在平坦的 X 轴方向迈大步，在陡峭的 Y 轴方向迈小步，走出完美的直线。

<div data-component="OptimizerPathVisualizer"></div>

### 5.2.1 为什么 Adam 是首选？

SGD 对所有参数使用相同的学习率。但在深度网络中，不同层的参数梯度甚至可能相差几个数量级。
Adam (Adaptive Moment Estimation) 这种**自适应算法**，相当于给每个参数都配了一个专属的学习率，这使得它对初始 LR 不那么敏感，且收敛极快。

**但是**，SGD + Momentum 在精调（Fine-tuning）阶段往往能获得略高的最终精度（泛化性更好）。目前的最佳实践是：
*   **Default**: AdamW (lr=3e-4 or 1e-3)
*   **CV SOTA**: SGD + Momentum (lr=0.1) + Cosine Decay

---

## 5.3 训练循环 (Training Loop)

```python
def train_step(model, inputs, targets, optimizer, criterion):
    # 1. Forward
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # 2. Backward
    optimizer.zero_grad() # ⚡️ 别忘清零
    loss.backward()
    
    # 3. Update
    optimizer.step()
    
    return loss.item()
```

---

## 5.4 学习率调度 (Scheduler)

"前期步子大（快速下山），后期步子小（精细搜索）"。

```python
# Cosine Annealing: 像余弦波一样平滑下降，非常流行
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    train(...)
    scheduler.step() # 每个 Epoch 结束调用
```

---

## 5.5 本章小结

*   **SGD** 简单但容易在峡谷震荡。
*   **Adam** 通过二阶动量自适应调整步长，是默认首选。
*   **Zero_grad** 是最容易遗忘的步骤。
*   配合 **LR Scheduler** 才能训练出 SOTA 模型。

> [!TIP]
> **思考题**:
> 为什么 AdamW 里的 "W" (Weight Decay fix) 很重要？
> (答案：标准的 L2 正则化在 Adam 这种自适应算法中效果不佳，AdamW 将权重衰减与梯度更新解耦，才使得它在 Transformer 上大放异彩。)

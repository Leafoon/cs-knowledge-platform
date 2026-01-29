---
title: "Chapter 7. 可视化与监控"
description: "使用 TensorBoard 监控训练指标，利用 Profiler 分析性能瓶颈"
updated: "2026-01-22"
---

> **Learning Objectives**
> *   学会编写 TensorBoard 监控代码 (`add_scalar`, `add_image`)。
> *   **Deep Dive**: 如何根据 Loss 曲线识别过拟合。
> *   通过 Profiler 定位训练慢的元凶 (CPU vs GPU)。
> *   常见性能瓶颈分析。

---

## 7.1 TensorBoard：训练过程的仪表盘

盯着控制台滚动的 Loss 数字看是不人道的，而且容易漏掉趋势信息（比如 Overfitting 时 Loss 还在降但 Val Accuracy 已经崩了）。

### 7.1.1 核心代码详解

```python
from torch.utils.tensorboard import SummaryWriter

# 初始化，指定日志目录
writer = SummaryWriter('runs/experiment_1')

for epoch in range(100):
    # record scalar
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    
    # record image
    writer.add_image('Input Images', img_grid, epoch)
    
writer.close()
```

---

## 7.2 训练动态分析 (Training Dynamics)

有了 TensorBoard，我们该看什么？
最重要的是 **Train Loss** 与 **Val Loss** 的关系。

### 交互演示：过拟合 vs 欠拟合

拖动滑块模拟训练 Epoch 的增加。
*   **Underfitting**: 两条线都很高。
*   **Optimal**: Val Loss 达到最低点。
*   **Overfitting**: Train Loss 继续低走，但 Val Loss 抬头。**注意观察红色区域**。

<div data-component="TrainingDynamicsVisualizer"></div>

这意味着什么？
*   如果在 **Underfitting** 阶段停止：你需要更复杂的模型，或者训练更久。
*   如果在 **Overfitting** 阶段停止：你训练过头了，模型在死记硬背。应该早停 (Early Stopping) 或增加正则化 (Dropout/Weight Decay)。

---

## 7.3 性能分析 (Profiling)

当模型训练太慢时，盲目猜测是没用的。你需要数据支持。

### 交互演示：Profiler 视图

寻找时间轴上的“气泡”（空闲时间）。
*   如果你看到 GPU 都是大段的空白，说明它在等 CPU (DataLoader)。
*   如果你看到密密麻麻的小色块，说明 Kernel Launch 开销太大。

<div data-component="ProfilerVisualizer"></div>

### 7.3.1 使用 torch.profiler

```python
with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    model(inputs)
    
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

---

## 7.4 常见性能杀手

### 杀手 1: Implicit Synchronization (隐式同步)

```python
# ❌ 每一步都把 tensor 拷回 CPU 打印
# 这会强制 GPU 等待，破坏并行流水线
print(loss.item()) 
```

### 杀手 2: DataLoader Sickness

*   `num_workers=0`: 单进程，在大规模图像训练中必慢。
*   没有 `pin_memory=True`: 导致 CPU->GPU 传输变慢。

---

## 7.5 本章小结

*   **TensorBoard** 是你的眼睛，**Profiler** 是你的听诊器。
*   时刻关注 **Train/Val Gap**，防止过拟合。
*   **GPU 利用率 (Volatile Utility)** 是衡量代码效率的第一指标。低利用率通常意味着 CPU 瓶颈。

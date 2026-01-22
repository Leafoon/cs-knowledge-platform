---
title: "Chapter 12. 性能极致优化"
description: "使用 PyTorch 2.0 torch.compile, CUDA Streams 和 Profiler 榨干显卡性能"
updated: "2026-01-22"
---

# Chapter 12. 性能极致优化 (Performance Tuning)

> **Learning Objectives**
> *   理解 **Memory Bound** (内存受限) vs **Compute Bound** (计算受限)。
> *   使用 `torch.compile` 进行算子融合 (Kernel Fusion)。
> *   使用 **CUDA Streams** 实现计算与通信的重叠 (Overlap)。
> *   使用 **CUDA Graphs** 消除 CPU 启动开销。

---

## 12.1 性能杀手：内存墙 (Memory Wall)

在深度学习中，90% 的算子（如 ReLU, Add, Norm）都是 **Memory Bound** 的。
这意味着 GPU 的计算核心（ALU）大部分时间都在**等待**数据从显存（HBM）搬运过来。

### 交互演示：Kernel Fusion (算子融合)

**Eager Mode** (左): 为每个操作启动一个 Kernel。`Read -> Calc -> Write`, `Read -> Calc -> Write`... 反复读写显存。
**Fused Kernel** (右): `torch.compile` 将它们合并。`Read -> Calc(Sin) -> Calc(Cos) -> Calc(Add) -> Write`。数据只走一次总线，速度飞快。

<div data-component="KernelFusionVisualizer"></div>

---

## 12.2 PyTorch 2.0: torch.compile

PyTorch 2.0 引入了 `torch.compile`，这是自 PyTorch 诞生以来最大的变革。
它默认使用 **Triton** 后端生成高度优化的融合算子。

```python
import torch

model = MyModel().cuda()

# === 一行代码加速 ===
# mode='default': 平衡编译时间和运行效率
# mode='reduce-overhead': 使用 CUDA Graphs，适合小 Batch
# mode='max-autotune': 最激进优化，编译很慢但跑得最快
opt_model = torch.compile(model, mode="reduce-overhead")

# 像往常一样使用
output = opt_model(input)
```

**Debug 技巧**:
如果编译报错，可以使用 `TORCH_LOGS="+inductor" python train.py` 查看 Inductor 生成的代码。

---

## 12.3 并行加速：CUDA Streams

默认情况下，PyTorch 在 Default Stream 上顺序执行所有 CUDA 操作。
但有时候，我们可以让两个独立的任务并行跑（例如，一边算卷积，一边把下个 Batch 的数据拷贝到 GPU）。

### 交互演示：Serial vs Multi-Stream

*   **Serial (Default)**: 任务 A 做完，任务 B 才能开始。
*   **Multi-Stream**: 任务 A 和 B 可以在不同的“车道”上同时跑。这叫 **Async Execution**。

<div data-component="CUDAStreamVisualizer"></div>

### 12.3.1 代码实战

```python
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

# 任务 A 在 Stream 1
with torch.cuda.stream(s1):
    A = torch.matmul(X, W1)

# 任务 B 在 Stream 2 (完全不依赖 A)
with torch.cuda.stream(s2):
    B = torch.matmul(Y, W2)

# 等待所有流完成
torch.cuda.synchronize()
```

---

## 12.4 CPU Overhead 与 CUDA Graphs

对于小模型（如 MLP）或小 Batch，GPU 跑一次可能只要 5ms，但 CPU 发射指令（Kernel Launch）却要 10ms。CPU 成了瓶颈。
**CUDA Graphs** 能够把一连串 Kernel 当作一张静态图，“录制”下来。
下次运行，只需通知 GPU “跑这张图”，GPU 就会自动按序执行，CPU 可以完全休息。

**启用方式**: 
1.  `torch.compile(mode="reduce-overhead")` (推荐)
2.  手动录制 `g = torch.cuda.CUDAGraph()` (比较麻烦，需保证 shape 固定)

---

## 12.5 本章小结

*   **Memory Bandwidth** 也是核心瓶颈，**Kernel Fusion** 是救星。
*   **torch.compile** 是现代 PyTorch 的必选项。
*   **CUDA Streams** 允许任务并行，掩盖 Latency。
*   **CUDA Graphs** 消除 CPU Launch Overhead。

> [!TIP]
> **Channels Last**:
> 在 NVIDIA Tensor Core 上，`NHWC` (Channels Last) 内存格式通常比 `NCHW` 快 20%。
> 记得调用 `model = model.to(memory_format=torch.channels_last)`。

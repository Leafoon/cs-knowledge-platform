---
title: "Chapter 8. 经典架构实战"
description: "从 ResNet 到 Transformer：现代深度学习基石的实现精讲"
updated: "2026-01-22"
---

# Chapter 8. 经典架构实战 (Architectures Guide)

> **Learning Objectives**
> *   彻底搞懂卷积的 **Stride** (步长) 和 **Padding** (填充)。
> *   从零实现 **ResNet** 的核心组件 `BasicBlock`。
> *   深入解构 Transformer 的 **Self-Attention** 机制。
> *   使用 `nn.MultiheadAttention` 构建序列模型。

---

## 8.1 卷积神经网络 (CNN): 基础与 ResNet

在写 ResNet 之前，我们先复习一下卷积操作的“黑话”。

### 交互演示：卷积算术 (Convolution Arithmetic)

调节右侧参数，观察输出尺寸的变化。
*   **Stride > 1**: 导致输出变小（下采样）。
*   **Padding**: 用于保持尺寸不变（Same Padding）。
*   **Receptive Field (感受野)**: 鼠标悬停在右侧输出 Grid 上，蓝色的高亮区域就是这个像素“看到”的原始图像区域。

<div data-component="ConvolutionVisualizer"></div>

### 8.1.1 ResNet 核心：Residual Block

深度网络难以训练的主要原因是**梯度消失/爆炸**和**退化问题**。
何恺明提出的 ResNet 通过引入**残差连接 (Residual Connection)**，即 $y = F(x) + x$，让梯度能走“高速公路”无损回传。

**Wait, `x` 怎么能和 `F(x)` 相加？**
这就要求 `F(x)` 的输出形状（Channel, H, W）必须和 `x` 完全一致。如果不一致（比如 stride=2 导致尺寸减半），我们需要对 `x` 也做一次下采样（通常用 1x1 卷积）。

```python
class BasicBlock(nn.Module):
    # expansion = 1 用于 ResNet-18/34。在 ResNet-50+ 中，Botteneck 结构的 expansion=4
    expansion = 1 

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 1. 第一层卷积：可能会改变尺寸 (stride)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 2. 第二层卷积：总是 stride=1, padding=1，保持尺寸不变
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 3. Shortcut (捷径)
        # 如果 stride > 1 (尺寸减半) 或者 in != out (通道数改变)
        # 那么 x 也需要经过一个 1x1 卷积来对齐形状
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        identity = x # 备份 x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # === 核心魔法：Element-wise Add ===
        out += self.shortcut(identity)
        
        out = self.relu(out) # 激活在相加之后做
        return out
```

---

## 8.2 Transformer 与 Self-Attention

RNN 需要一步步串行计算，无法并行。Transformer 抛弃了循环，完全依赖 Attention。

### 交互演示：Self-Attention 显微镜

*   **Head 0 (Semantic)**: 观察单词 **"it"** (代词)。你会发现它的注意力高度集中在 **"animal"** 上。
*   **Head 1 (Local)**: 很多 Head 实际上只学到了类似 "Look at previous token" 这种简单的语法规则。
*   **Masking**: 在 Decoder 中，我们还会看到下三角掩码（无法看到未来的词）。

<div data-component="AttentionMatrixVisualizer"></div>

### 8.2.1 代码：MultiheadAttention

PyTorch 官方 API 已经封装得很好了。

```python
# batch_first=True 非常重要，否则默认是 (Seq, Batch, Feat)
self_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)

# 假设输入一个 Batch 的句子
# Batch=32, 句长=10, 词向量维度=256
x = torch.randn(32, 10, 256)

# Self-Attention: Q, K, V 都是 x 自己
# output: [32, 10, 256], weights: [32, 10, 10] (注意力矩阵)
attn_output, attn_weights = self_attn(query=x, key=x, value=x)
```

---

## 8.3 循环神经网络 (RNN): LSTM

虽然 Transformer 是当红炸子鸡，但在**流式数据**（Streaming Data，如实时语音降噪）或**极小资源**场景下，LSTM 依然是王者。

```python
# input_size=100 (特征维), hidden_size=256, num_layers=2
lstm = nn.LSTM(100, 256, num_layers=2, batch_first=True)

# 输入: [Batch, Seq, Feat]
x = torch.randn(32, 10, 100)

# RNN 除了返回所有时间步的 output，还会返回最后时刻的隐状态 (h_n, c_n)
# output: [32, 10, 256] -> 每一秒的理解
# h_n: [2, 32, 256] -> 读完整个句子后的总结 (2 是层数)
output, (h_n, c_n) = lstm(x)
```

---

## 8.4 本章小结

*   **CNN**: 通过 `Conv2d` 获取局部特征，随着层数增加，感受野变大，学到全局特征。**ResNet** 是骨架。
*   **Transformer**: 通过 `Self-Attention` 直接捕捉全局依赖，并行度极高。
*   **API 建议**: 无论是 CNN (`Conv2d`) 还是 RNN/Transformer (`LSTM`, `MultiheadAttention`)，尽量使用封装好的层，而不是自己写矩阵乘法，因为官方实现经过了极致的 C++ 优化 (cuDNN/MKL)。

> [!TIP]
> **Thinking in Shape**:
> 学习架构时，不要纠结具体的公式推导。**时刻关注 Tensor 的 Shape 变化**。
> 如果你能闭着眼睛说出数据经过这层之后的 `(B, C, H, W)` 变成了多少，你就真的懂了。

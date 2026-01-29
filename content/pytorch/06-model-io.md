---
title: "Chapter 6. 模型 IO 与 Checkpoint"
description: "学会保存和加载模型状态，实现断点续训 (Resume Training)"
updated: "2026-01-22"
---

> **Learning Objectives**
> *   理解 `state_dict` 的字典结构。
> *   **Deep Dive**: 为什么不推荐 `torch.save(model)`。
> *   实现生产级的 Checkpoint 保存与加载机制（包含优化器状态、Epoch、Scheduler）。
> *   解决加载时的常见问题（设备不匹配、Key 不匹配）。

---

## 6.1 State Dict 是什么？

在 PyTorch 中，“模型”本质上就是代码逻辑（Class）加上参数（Dictionary）。
这个参数字典就是 `state_dict`。

```python
# 假设我们有一个简单的模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

model = MyModel()

# 打印 state_dict
print(model.state_dict())
```

**输出解析**:
```python
OrderedDict([
    ('fc.weight', tensor([[-0.12, ...], [0.23, ...]])), # fc 层的权重矩阵
    ('fc.bias', tensor([0.01, -0.05]))                  # fc 层的偏置向量
])
```
*   **Key**: 参数的名称（层级路径）。
*   **Value**: 实际的 Tensor 数据。

---

## 6.2 保存与加载：最佳实践

### 6.2.1 推荐：仅保存参数 (state_dict)

这种方式最稳健。即使你的代码目录结构变了，只要模型类的定义（`__init__` 里的层名字）没变，参数就能加载进去。

```python
# === 保存 (Save) ===
# torch.save 本质上是用 Python pickle 序列化对象
# 我们只保存字典，文件体积最小，兼容性最好
torch.save(model.state_dict(), 'model_weights.pth')


# === 加载 (Load) ===
# 1. 必须先实例化模型对象。
# 此时 model 里的参数是随机初始化的。
model = MyModel() 

# 2. 加载权重文件到内存
state_dict = torch.load('model_weights.pth')

# 3. 将参数字典“灌入”模型
# strict=True (默认): 要求文件里的 key 和模型里的 key 必须完全一致，多一个少一个都会报错。
model.load_state_dict(state_dict, strict=True) 

# 4. 关键：切换到评估模式
# 这会固定 Dropout 和 BatchNorm，否则推理结果不一致。
model.eval() 
```

### 6.2.2 不推荐：保存整个模型

`torch.save(model, 'model.pth')` 会序列化整个对象。如果代码变动，Pickle 解析会失败。**尽量避免**。

---

## 6.3 进阶：断点续训 (Checkpointing)

在训练大型模型时，Crash 是常态。如果训练了 3 天突然断电，你绝对不想从 Epoch 0 开始重跑。
因此，我们需要保存**Checkpoint（检查点）**。它不仅仅是模型权重，是**整个训练现场**的快照。

### 交互演示：Crash 与 Resume

模拟训练中断，并观察从 Checkpoint 恢复时的状态变化。

<div data-component="CheckpointSimulator"></div>

### 6.3.1 完整的保存代码详解

```python
def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), # ⚡️ 必须保存，否则动量丢失
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
```

---

## 6.4 迁移学习与局部加载

**场景**：你使用 ResNet50 做预训练，但想修改分层类 (Final Layer) 从 1000 类改为 10 类。
此时加载预训练权重时，FC 层的形状匹配不上。

### 交互演示：Transfer Learning 权重注入

观察权重是如何从 Source Model 流向 Target Model 的。
注意红色标记的层（FC Layer）：由于形状不匹配（Key 相同但 Shape 不同），它们被过滤掉了，保持随机初始化。这正是我们想要的——复用 Backbone，重训 Head。

<div data-component="TransferLearningVisualizer"></div>

**代码技巧**:

```python
pretrained_dict = torch.load('resnet50.pth')
model = ResNet50(num_classes=10) # 新模型
model_dict = model.state_dict()

# 字典推导式：只保留形状匹配的参数
pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                   if k in model_dict and v.shape == model_dict[k].shape}

# 更新并加载
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict) # strict=False 也可以，但手动过滤更可控
```

---

## 6.5 本章小结

1.  **state_dict** 是灵魂，永远只保存它。
2.  **断点续训** 三要素：Model Weights, Optimizer State, Epoch/Meta。
3.  **map_location** 是解决跨设备加载报错的钥匙。
4.  **Transfer Learning** 时需要过滤掉不匹配的层（通常是最后一层）。

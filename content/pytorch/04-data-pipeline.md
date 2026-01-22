---
title: "Chapter 4. 数据流水线"
description: "深入理解 Dataset 抽象、DataLoader 高效加载机制、采样策略以及自定义数据管道"
updated: "2026-01-22"
---

# Chapter 4. 数据流水线 (Dataset & DataLoader)

> **Learning Objectives**
> *   理解 Map-style vs Iterable-style Dataset 的区别。
> *   掌握 `Dataset` 的三板斧实现法。
> *   熟练配置 `DataLoader`：Workders, Pin Memory, Collate Function。
> *   (进阶) 理解 `Sampler` 的作用，通过可视化对比不同的采样策略。

---

## 4.1 PyTorch 数据加载哲学

训练深度学习模型时，GPU 的计算速度往往极快，导致 CPU 数据读取成为瓶颈 。
PyTorch 使用 **Dataset** (存储) 和 **DataLoader** (加载) 分离的设计模式。

### 交互演示：DataLoader 流水线

观察数据是如何从原始文件被读取、变换（Transform），最后被组装成 Tensor Batch 的。

<div data-component="BatchProcessor"></div>

---

## 4.2 自定义 Dataset (Map-style)

继承 `torch.utils.data.Dataset` 并实现三个魔法方法。

```python
class CustomImageDataset(Dataset):
    def __init__(self, img_labels_df, img_dir, transform=None):
        self.img_labels = img_labels_df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # 1. 拼路径
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # 2. 读文件 (IO)
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        
        # 3. 变换 (CPU Compute)
        if self.transform:
            image = self.transform(image)
            
        return image, label
```

---

## 4.3 Sampler：决定“怎么拿”数据

`DataLoader` 默认做的是随机 Shuffle (RandomSampler) 或顺序读取 (SequentialSampler)。但在某些场景下，我们需要干预这个过程。

### 交互演示：采样策略对比

*   **Sequential**: 顺序读取。验证集标配。
*   **Random**: 随机打乱。训练集标配。
*   **Weighted**: 如果你的正负样本比例是 1:100，模型根本学不到正样本。使用 WeightedSampler 可以强行让模型看到更多的正样本。

<div data-component="SamplerVisualizer"></div>

**代码实现 WeightedSampler**:

```python
from torch.utils.data import WeightedRandomSampler

# 假设样本权重 (与样本频率成反比)
weights = [0.1, 0.9, 0.1, ...] 
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# 传入 sampler 后，shuffle 参数必须为 False（互斥）
loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

---

## 4.4 DataLoader 性能调优

这是面试高频考点，也是生产环境中的大坑。

```python
loader = DataLoader(
    dataset,
    batch_size=128,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True, # PyTorch 2.0+ 推荐
    prefetch_factor=2
)
```

1.  **`num_workers`**: 子进程数。
    *   设为 `0`：主进程加载，方便 Debug，但慢。
    *   设为 `CPU 核数`：可能导致 CPU 争抢。
    *   **推荐**: `min(4 * GPU数, CPU核数)`。
2.  **`pin_memory=True`**:
    *   在 CPU 内存中分配“锁页内存 (Pinned Memory)”，这块内存可以直接通过 DMA (Direct Memory Access) 拷贝到 GPU，由于不经过 CPU 缓存，速度极快。
    *   **一句话：只要用 GPU 训练，就设为 True。**
3.  **`persistent_workers=True`**:
    *   每个 Epoch 结束后不销毁子进程，避免反复创建进程的开销。

---

## 4.5 数据增强 (Data Augmentation)

现代深度学习已经离不开增强。

```python
# torchvision.transforms.v2 (推荐)
from torchvision.transforms import v2

transforms = v2.Compose([
    v2.RandomResizedCrop(224),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.2),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, ...], std=[0.229, ...])
])
```

---

## 4.6 本章小结

*   **Dataset** 负责单个样本的 IO 和 Transform。
*   **DataLoader** 负责批量化、并行化。
*   遇到 **Class Imbalance** (样本不均衡) 问题，优先考虑使用 `WeightedRandomSampler`。
*   性能调优三板斧：`num_workers=4`, `pin_memory=True`, `persistent_workers=True`。

> [!TIP]
> **思考题**:
> 如果你的 Dataset `__getitem__` 只有简单的 Tensor 索引操作（数据全在内存里），把 `num_workers` 设大反而会变慢，为什么？
> (提示：进程间通信 IPC 的开销可能超过了简单的内存拷贝开销。)

---
title: "Chapter 11. 分布式训练"
description: "突破单卡显存限制：DDP 原理与 FSDP 大模型训练实战"
updated: "2026-01-22"
---

> **Learning Objectives**
> *   **Deep Dive**: DDP 的 `All-Reduce` 通信机制。
> *   掌握 `torchrun` 启动多卡训练的标准流程。
> *   理解 **FSDP** 如何通过模型切片 (Sharding) 来训练超大模型。

---

## 11.1 分布式并行策略

当我们有一堆 GPU 时，怎么用？
1.  **Data DataParallel (DDP)**: 每张卡存全套模型，分头处理不同数据。 -> **最常用**
2.  **Model Parallel (MP)**: 模型太大，单卡放不下，把模型切开（Layer 0-10 在 GPU0, Layer 11-20 在 GPU1）。 -> **通信开销大，Pipeline复杂**
3.  **Fully Sharded Data Parallel (FSDP)**: DDP 的进化版。每张卡只存 **1/N** 的模型参数。 -> **大模型标配**

### 交互演示：DDP vs FSDP

切换下方的模式，观察显存占用（Memory）的区别。

*   **DDP (蓝色)**: 每张卡都占满了显存（完整模型）。
*   **FSDP (绿色)**: 每张卡只占用了一小块（Shard）。这使得我们可以在有限的显存中塞入更大的模型。

<div data-component="DistributedVisualizer"></div>

---

## 11.2 DDP: 初始化与启动

别再用 `nn.DataParallel` 了！它不仅慢，还是单进程多线程（受 Python GIL 限制）。
DDP 是**多进程** (Multi-Process) 的，每张卡一个 Python 进程，真正的并行。

### 11.2.1 改造代码为 DDP

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    # 1. 初始化进程组 (Process Group)
    # 这一步会建立进程间的 TCP/IP 或 RDMA 连接
    # 一般不需要手动传 rank/world_size，torchrun 会通过环境变量传入
    dist.init_process_group("nccl") # NVIDIA GPU 推荐 nccl
    
    # 2. 获取当前进程的 ID
    rank = dist.get_rank()         # 全局第几号进程 (0, 1, 2, 3...)
    local_rank = int(os.environ["LOCAL_RANK"]) # 当前机器第几号 GPU
    
    # 3. 设定当前设备
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # 4. 构建模型并搬到 GPU
    model = MyModel().to(device)
    
    # 5. DDP 包装
    # 这会在后台创建一个 Hook，当 Backward 计算完梯度后，自动触发 All-Reduce
    model = DDP(model, device_ids=[local_rank])
    
    # 6. 数据采样 (关键！)
    # DistributedSampler 保证每张卡读到的数据是不重叠的
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    # 7. 训练循环
    for epoch in range(10):
        # 必须调用 set_epoch，否则导致 shuffle 种子一样，每轮数据顺序相同
        sampler.set_epoch(epoch) 
        
        for imgs, targets in dataloader:
            imgs, targets = imgs.to(device), targets.to(device)
            # ... 常规训练代码 (Forward, Backward, Optimizer) ...
            # 只有在 Backward 结束时，DDP 才会介入通信
```

### 11.2.2 使用 torchrun 启动

假设你有 4 张卡：
```bash
torchrun --nproc_per_node=4 train.py
```

---

## 11.3 FSDP: 给显存减负

当模型大到单卡放不下时（比如 参数量 > 1B），DDP 就OOM了。
FSDP (Fully Sharded Data Parallel) 将参数 $(W)$、梯度 $(G)$ 和 优化器状态 $(OS)$ 全部切分。

**代价是什么？**
通信换显存。
在 Forward 计算某一层的瞬间，FSDP 需要从其他卡把这一层的参数 **Gather** 过来（拼成完整层），算完后立刻**丢弃**（释放显存）。
在 Backward 时同理。

这就像大家凑钱买了一本书（大模型），平时撕开每人存几页。要读书时，临时把大家手里的页拼起来读，读完马上还回去。

---

## 11.4 本章小结

*   单机单卡 -> 常规代码。
*   单机多卡 / 多机多卡 -> **DDP** (DistributedDataParallel)。
*   超大模型 -> **FSDP**。
*   启动命令 -> **torchrun**。

> [!TIP]
> **思考题**:
> 在 DDP 训练中，打印 Loss 时，为什么我们通常只在 `rank == 0` 的进程打印？
> (答案：因为有 N 个进程在跑同样的代码。如果不加判断，你的控制台会刷出 N 行 Loss，既乱又没必要。)

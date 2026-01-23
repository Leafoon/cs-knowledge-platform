# Chapter 8: 分布式训练与加速

## 8.1 分布式训练概述

### 8.1.1 为什么需要分布式训练？

随着模型规模的增长，单卡训练面临严峻挑战：

| 模型 | 参数量 | FP16 显存 | 单卡 A100 (80GB) |
|------|--------|-----------|------------------|
| BERT-base | 110M | 0.2 GB | ✓ 可训练 |
| GPT-2 | 1.5B | 3 GB | ✓ 可训练 |
| **LLaMA-7B** | **7B** | **14 GB** | ✓ 勉强（需优化） |
| **LLaMA-13B** | **13B** | **26 GB** | ✗ 单卡不够 |
| **LLaMA-70B** | **70B** | **140 GB** | ✗ 需要多卡 |

**单卡训练的瓶颈**：
```python
# 7B 模型的显存需求
model_params = 7_000_000_000 * 2  # FP16: 14 GB
optimizer_state = 7_000_000_000 * 8  # Adam: 56 GB (momentum + variance)
gradients = 7_000_000_000 * 2  # 14 GB
activations = batch_size * seq_len * hidden_size * num_layers * 4  # ~10 GB

total = 14 + 56 + 14 + 10 = 94 GB  # 远超单卡容量！
```

<div data-component="DistributedTrainingNeedVisualizer"></div>

### 8.1.2 分布式训练范式

**三种主要策略**：

1. **数据并行（Data Parallelism, DP）**：
   - 每个 GPU 持有完整模型副本
   - 数据切分到不同 GPU
   - 梯度同步后更新

2. **模型并行（Model Parallelism, MP）**：
   - 模型切分到不同 GPU
   - 每个 GPU 只持有部分层
   - 激活值在 GPU 间传递

3. **混合并行（Hybrid Parallelism）**：
   - 数据并行 + 模型并行 + Pipeline 并行
   - 适用于超大模型（100B+）

<div data-component="ParallelismStrategyComparison"></div>

---

## 8.2 Hugging Face Accelerate

### 8.2.1 Accelerate 简介

**Accelerate** 是 Hugging Face 提供的分布式训练框架，核心优势：

> 一行代码切换：单GPU → 多GPU → TPU → 混合精度 → DeepSpeed

**安装**：

```bash
pip install accelerate
```

### 8.2.2 基础用法

**最简单的分布式训练**：

```python
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. 初始化 Accelerator
accelerator = Accelerator()

# 2. 加载模型和数据
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
train_dataloader = DataLoader(dataset, batch_size=8)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# 3. 准备分布式训练（关键！）
model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

# 4. 训练循环（与单GPU完全相同！）
for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        
        # 自动处理梯度同步
        accelerator.backward(loss)
        
        optimizer.step()
        optimizer.zero_grad()

# 5. 保存模型（自动处理分布式保存）
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained("./output")
```

**启动命令**：

```bash
# 单GPU
python train.py

# 多GPU（自动检测）
accelerate launch train.py

# 指定GPU数量
accelerate launch --num_processes 4 train.py

# 混合精度
accelerate launch --mixed_precision fp16 train.py
```

### 8.2.3 配置文件

**生成配置**：

```bash
accelerate config
```

交互式问答后生成 `default_config.yaml`：

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU  # 多GPU训练
num_processes: 4             # 4张卡
gpu_ids: all                 # 使用所有GPU
mixed_precision: bf16        # BF16混合精度
downcast_bf16: no
machine_rank: 0
main_process_ip: null
main_process_port: null
num_machines: 1
rdzv_backend: static
same_network: true
use_cpu: false
```

**使用配置启动**：

```bash
accelerate launch --config_file default_config.yaml train.py
```

### 8.2.4 高级特性

**梯度累积**：

```python
accelerator = Accelerator(gradient_accumulation_steps=4)

for batch in train_dataloader:
    # 自动处理累积逻辑
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

**混合精度 + 梯度裁剪**：

```python
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision="bf16",  # 或 "fp16"
    gradient_accumulation_steps=4,
)

# 训练循环
for batch in train_dataloader:
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        
        # 梯度裁剪
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        optimizer.zero_grad()
```

**分布式日志与评估**：

```python
# 只在主进程打印日志
if accelerator.is_main_process:
    print(f"Epoch {epoch}, Loss: {loss.item()}")

# 收集所有进程的指标
all_losses = accelerator.gather(loss)
avg_loss = all_losses.mean()

# 等待所有进程
accelerator.wait_for_everyone()
```

<div data-component="AccelerateWorkflowVisualizer"></div>

---

## 8.3 数据并行（Data Parallelism）

### 8.3.1 DistributedDataParallel (DDP)

**PyTorch 原生 DDP**：

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 1. 初始化进程组
def setup(rank, world_size):
    dist.init_process_group(
        backend="nccl",  # NVIDIA GPU
        init_method="env://",
        rank=rank,
        world_size=world_size
    )

# 2. 创建模型
def train(rank, world_size):
    setup(rank, world_size)
    
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(rank)
    model = DDP(model, device_ids=[rank])
    
    # 3. 分布式采样器
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=8)
    
    # 4. 训练
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # 每个epoch重新洗牌
        
        for batch in dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

# 5. 启动多进程
import torch.multiprocessing as mp

if __name__ == "__main__":
    world_size = 4
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
```

**DDP 工作原理**：

<div data-component="DDPCommunicationFlow"></div>

### 8.3.2 梯度同步机制

**Ring-AllReduce 算法**：

```python
# 伪代码展示 AllReduce 过程
def ring_allreduce(gradients, rank, world_size):
    """
    环形全归约：每个GPU与邻居交换梯度
    
    优势：通信量 = O(N)，不依赖GPU数量
    """
    chunk_size = len(gradients) // world_size
    
    # Reduce-Scatter 阶段
    for step in range(world_size - 1):
        send_to = (rank + 1) % world_size
        recv_from = (rank - 1 + world_size) % world_size
        
        # 发送和接收梯度块
        send_chunk = gradients[chunk_size * rank : chunk_size * (rank + 1)]
        recv_chunk = receive_from(recv_from)
        
        # 累加接收到的梯度
        gradients[...] += recv_chunk
    
    # AllGather 阶段
    for step in range(world_size - 1):
        # 类似过程，收集所有块
        pass
    
    # 最终每个GPU都有完整的平均梯度
    return gradients / world_size
```

**通信开销分析**：

| 算法 | 通信量 | 延迟 | 适用场景 |
|------|--------|------|----------|
| **Parameter Server** | O(2N × P) | 高 | 异构集群 |
| **AllReduce** | O(2N) | 中 | 同构GPU |
| **Ring-AllReduce** | O(2N) | 低 | 大规模训练 |

---

## 8.4 FSDP（Fully Sharded Data Parallel）

### 8.4.1 FSDP 原理

**核心思想**：将模型参数、梯度、优化器状态**分片**到不同GPU。

**与 DDP 的对比**：

| 特性 | DDP | FSDP |
|------|-----|------|
| 模型副本 | 每个GPU完整副本 | 参数分片 |
| 显存占用 | O(M) per GPU | O(M/N) per GPU |
| 通信 | 梯度AllReduce | 参数收集+分片 |
| 适用模型 | <10B | 10B - 100B+ |

<div data-component="FSDPShardingVisualizer"></div>

**FSDP 三阶段**：

1. **AllGather**：收集分片参数 → 重建完整层
2. **Compute**：前向/反向传播
3. **ReduceScatter**：梯度归约并重新分片

### 8.4.2 使用 FSDP

**PyTorch 原生 FSDP**：

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# 1. 定义包装策略（每个Transformer层独立分片）
auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer}
)

# 2. 包装模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # 完全分片
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    device_id=torch.cuda.current_device(),
)

# 3. 训练（与普通训练相同）
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**FSDP 分片策略**：

```python
from torch.distributed.fsdp import ShardingStrategy

# 1. FULL_SHARD（完全分片，最省显存）
ShardingStrategy.FULL_SHARD

# 2. SHARD_GRAD_OP（仅分片梯度和优化器状态）
ShardingStrategy.SHARD_GRAD_OP

# 3. NO_SHARD（不分片，等同于DDP）
ShardingStrategy.NO_SHARD

# 4. HYBRID_SHARD（节点内完全分片，节点间复制）
ShardingStrategy.HYBRID_SHARD
```

### 8.4.3 FSDP + Accelerate

```python
from accelerate import Accelerator

accelerator = Accelerator(
    fsdp_plugin={
        "sharding_strategy": "FULL_SHARD",
        "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        "backward_prefetch": "BACKWARD_PRE",
        "forward_prefetch": True,
    }
)

model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)

# 训练循环
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

**启动命令**：

```bash
accelerate launch \
    --config_file fsdp_config.yaml \
    --num_processes 4 \
    train.py
```

**配置文件示例**：

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_forward_prefetch: true
  fsdp_sharding_strategy: 1  # FULL_SHARD
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
mixed_precision: bf16
num_processes: 4
```

### 8.4.4 FSDP 性能优化

**CPU Offload**（显存极限优化）：

```python
from torch.distributed.fsdp import CPUOffload

model = FSDP(
    model,
    cpu_offload=CPUOffload(offload_params=True),  # 参数卸载到CPU
    # ...
)

# 显存节省: 额外 -30%
# 速度损失: -20-40%（CPU-GPU传输开销）
```

**激活检查点（Activation Checkpointing）**：

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing
)

# 自动应用到所有Transformer层
def check_fn(submodule):
    return isinstance(submodule, LlamaDecoderLayer)

apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT
    ),
    check_fn=check_fn
)
```

---

## 8.5 DeepSpeed

### 8.5.1 DeepSpeed ZeRO 概述

**ZeRO（Zero Redundancy Optimizer）** 三阶段：

| 阶段 | 分片内容 | 显存节省 | 通信开销 |
|------|---------|---------|----------|
| **ZeRO-1** | 优化器状态 | 4× | 与DDP相同 |
| **ZeRO-2** | + 梯度 | 8× | 1.5× DDP |
| **ZeRO-3** | + 模型参数 | 64× | 2× DDP |

<div data-component="DeepSpeedZeROStages"></div>

**显存占用计算**：

```python
# 假设模型参数: M = 7B
# FP16 训练显存需求

# DDP（每个GPU完整副本）
memory_ddp = (
    2 * M +      # 模型参数 (FP16)
    2 * M +      # 梯度
    12 * M       # 优化器状态 (Adam FP32: 4+4+4)
) = 16 * M = 16 * 7B = 112 GB

# ZeRO-1（分片优化器状态，N=4 GPUs）
memory_zero1 = 2*M + 2*M + 12*M/N = (16 - 12 + 12/4) * 7B = 28 GB

# ZeRO-2（+ 分片梯度）
memory_zero2 = 2*M + 2*M/N + 12*M/N = (16 - 14 + 14/4) * 7B = 14.5 GB

# ZeRO-3（+ 分片参数）
memory_zero3 = 2*M/N + 2*M/N + 12*M/N = 16*M/N = 16 * 7B / 4 = 28 GB / 4 = 7 GB
```

### 8.5.2 使用 DeepSpeed

**安装**：

```bash
pip install deepspeed
```

**配置文件** (`ds_config.json`)：

```json
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 3e-5,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 3e-5,
      "warmup_num_steps": 500
    }
  }
}
```

**Trainer 集成**：

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    deepspeed="./ds_config.json",  # ← 指定配置文件
    per_device_train_batch_size=8,
    num_train_epochs=3,
    bf16=True,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

**启动命令**：

```bash
# 使用 Trainer（自动启动DeepSpeed）
deepspeed --num_gpus=4 train.py

# 或使用 Accelerate
accelerate launch --config_file deepspeed_config.yaml train.py
```

### 8.5.3 ZeRO-3 Offload

**完整优化配置**：

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true,
      "buffer_count": 4,
      "fast_init": false
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

**性能对比**（LLaMA-70B）：

| 配置 | GPUs | 显存/GPU | 训练速度 |
|------|------|----------|----------|
| DDP | 32× A100 | 80 GB (不够) | - |
| ZeRO-2 | 8× A100 | 78 GB | 100% |
| **ZeRO-3** | **4× A100** | **68 GB** | **85%** |
| **ZeRO-3 + Offload** | **2× A100** | **42 GB** | **45%** |

<div data-component="DeepSpeedOffloadFlow"></div>

---

## 8.6 Pipeline 并行

### 8.6.1 Pipeline 并行原理

将模型**纵向切分**为多个阶段（stages），每个阶段在不同GPU上。

**朴素 Pipeline（气泡问题）**：

```
GPU 0: [F1]     [F2]     [F3]     [F4]
GPU 1:     [F1]     [F2]     [F3]     [F4]
GPU 2:         [F1]     [F2]     [F3]     [F4]
GPU 3:             [F1]     [F2]     [F3]     [F4]

气泡率 = (N-1)/N × 100% = 75%（4卡）
```

**GPipe（微批次Pipeline）**：

```
GPU 0: [F1][F2][F3][F4]     [B1][B2][B3][B4]
GPU 1:     [F1][F2][F3][F4] [B1][B2][B3][B4]
GPU 2:         [F1][F2][F3][F4] [B1][B2][B3][B4]
GPU 3:             [F1][F2][F3][F4] [B1][B2][B3][B4]

气泡率 降低到 ~50%
```

<div data-component="PipelineParallelismVisualizer"></div>

### 8.6.2 实现 Pipeline 并行

**使用 Megatron-LM**：

```python
from megatron import get_args, initialize_megatron
from megatron.core import parallel_state
from megatron.model import GPTModel

# 初始化
initialize_megatron(
    args_defaults={
        'pipeline_model_parallel_size': 4,  # 4个pipeline阶段
        'tensor_model_parallel_size': 1,
    }
)

# 模型自动切分
model = GPTModel(...)

# 训练
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

**DeepSpeed Pipeline**：

```json
{
  "pipeline": {
    "stages": "auto",
    "partition_method": "type:transformer",
    "activation_checkpoint_interval": 1
  }
}
```

---

## 8.7 张量并行（Tensor Parallelism）

### 8.7.1 张量并行原理

**列并行**（Column Parallel）：

```python
# 原始: Y = XW  (X: [B, H], W: [H, 4H], Y: [B, 4H])

# 切分 W 为 [W1 | W2]（沿列切分）
W1 = W[:, :2H]  # GPU 0
W2 = W[:, 2H:]  # GPU 1

# 并行计算
Y1 = X @ W1  # GPU 0: [B, 2H]
Y2 = X @ W2  # GPU 1: [B, 2H]

# 拼接
Y = concat([Y1, Y2], dim=1)  # [B, 4H]
```

**行并行**（Row Parallel）：

```python
# 原始: Y = XW  (X: [B, 4H], W: [4H, H], Y: [B, H])

# 切分 W 为 [W1; W2]（沿行切分）
W1 = W[:2H, :]  # GPU 0
W2 = W[2H:, :]  # GPU 1

# X 也需要对应切分
X1 = X[:, :2H]  # GPU 0
X2 = X[:, 2H:]  # GPU 1

# 并行计算
Y1 = X1 @ W1  # GPU 0: [B, H]
Y2 = X2 @ W2  # GPU 1: [B, H]

# AllReduce 求和
Y = Y1 + Y2  # [B, H]
```

<div data-component="TensorParallelismVisualizer"></div>

### 8.7.2 Megatron 张量并行

**Transformer 层的切分**：

```python
# Attention: Q, K, V 列并行
Q = X @ W_Q  # W_Q 列切分
K = X @ W_K
V = X @ W_V

# Output 行并行
O = Attention(Q, K, V) @ W_O  # W_O 行切分

# FFN: 第一层列并行，第二层行并行
H = gelu(X @ W_1)  # W_1 列切分
Y = H @ W_2        # W_2 行切分
```

**使用示例**：

```python
from megatron.core import parallel_state

# 初始化张量并行
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=2  # 2卡张量并行
)

# 使用并行层
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear

# 列并行层
layer1 = ColumnParallelLinear(
    input_size=768,
    output_size=3072,
    gather_output=False  # 不收集输出（保持切分状态）
)

# 行并行层
layer2 = RowParallelLinear(
    input_size=3072,
    output_size=768,
    input_is_parallel=True  # 输入已切分
)
```

---

## 8.8 3D 并行（数据+Pipeline+张量）

### 8.8.1 3D 并行组合

训练超大模型（175B+）需要组合所有并行策略：

```
总GPU数 = 数据并行度 × Pipeline并行度 × 张量并行度
N_total = DP × PP × TP

示例: 64 GPUs = 8 × 4 × 2
  - 数据并行: 8路（8份数据副本）
  - Pipeline并行: 4阶段（模型纵向切4段）
  - 张量并行: 2路（每层横向切2份）
```

<div data-component="ThreeDParallelismVisualizer"></div>

**显存占用分析**（GPT-175B）：

```python
M = 175B  # 参数量

# 单GPU（理论，不可行）
memory_single = 16 * M = 2800 GB

# 3D并行 (DP=8, PP=4, TP=2)
memory_3d = 16 * M / (PP * TP) / DP
          = 16 * 175B / (4 * 2) / 8
          = 43.75 GB  # 可在A100 80GB上运行！
```

### 8.8.2 Megatron-DeepSpeed

**配置示例**：

```json
{
  "train_batch_size": 512,
  "gradient_accumulation_steps": 1,
  "steps_per_print": 10,
  "zero_optimization": {
    "stage": 1,
    "reduce_bucket_size": 5e8
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 12
  },
  "pipeline": {
    "stages": 4,
    "partition_method": "parameters"
  },
  "wall_clock_breakdown": false
}
```

**启动脚本**：

```bash
deepspeed --num_gpus 64 \
    --num_nodes 8 \
    --hostfile hostfile \
    pretrain_gpt.py \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 4 \
    --num-layers 96 \
    --hidden-size 12288 \
    --num-attention-heads 96 \
    --micro-batch-size 2 \
    --global-batch-size 512 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --train-iters 500000 \
    --lr 6.0e-5 \
    --deepspeed_config ds_config.json
```

---

## 8.9 通信优化

### 8.9.1 通信原语

**常用集合通信操作**：

| 操作 | 描述 | 通信量 | 用途 |
|------|------|--------|------|
| **Broadcast** | 一对多广播 | O(N) | 发送初始参数 |
| **Reduce** | 多对一归约 | O(N) | 收集梯度 |
| **AllReduce** | 全归约 | O(2N) | DDP梯度同步 |
| **AllGather** | 全收集 | O(N×P) | FSDP参数收集 |
| **ReduceScatter** | 归约后分发 | O(N) | FSDP梯度分片 |

<div data-component="CollectiveCommunicationPrimitives"></div>

**PyTorch 实现**：

```python
import torch.distributed as dist

# AllReduce（求和）
tensor = torch.randn(1000).cuda()
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

# AllGather
tensor = torch.randn(1000).cuda()
tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
dist.all_gather(tensor_list, tensor)

# ReduceScatter
input_list = [torch.randn(1000).cuda() for _ in range(world_size)]
output = torch.zeros(1000).cuda()
dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM)
```

### 8.9.2 通信-计算重叠

**梯度累积 + 通信重叠**：

```python
# 错误：等待梯度计算完成后再通信
for batch in dataloader:
    loss.backward()  # 计算所有梯度
    all_reduce_gradients()  # 然后一次性通信
    optimizer.step()

# 正确：边计算边通信
for batch in dataloader:
    with model.no_sync():  # 禁用自动同步
        for micro_batch in split_batch(batch):
            loss = model(micro_batch)
            loss.backward()  # 计算部分梯度
            
            # 异步通信刚计算好的梯度
            if grad_ready():
                async_all_reduce(grad)
    
    # 等待所有通信完成
    wait_all_reduce()
    optimizer.step()
```

**NCCL 优化**：

```python
import os

# 启用 NCCL 优化
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_IB_DISABLE'] = '0'  # 启用 InfiniBand
os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'  # 网络接口
os.environ['NCCL_P2P_LEVEL'] = 'NVL'  # NVLink
```

---

## 8.10 实战案例：LLaMA-13B 多卡训练

**完整配置**（4× A100 40GB）：

```python
# train.py
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

# 1. 加载模型和数据
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    torch_dtype=torch.bfloat16,  # BF16节省显存
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# 2. 训练配置（FSDP + 梯度检查点）
training_args = TrainingArguments(
    output_dir="./llama-13b-fsdp",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # 小batch
    gradient_accumulation_steps=16,  # 累积 → 有效batch=32
    learning_rate=2e-5,
    bf16=True,
    
    # FSDP配置
    fsdp="full_shard auto_wrap",
    fsdp_config={
        "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
        "fsdp_backward_prefetch": "backward_pre",
        "fsdp_forward_prefetch": True,
    },
    
    # 优化
    gradient_checkpointing=True,
    optim="adamw_torch_fused",  # 融合Adam
    
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
)

# 3. 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```

**启动命令**：

```bash
# 使用 torchrun
torchrun --nproc_per_node=4 train.py

# 或使用 Accelerate
accelerate launch --num_processes 4 train.py
```

**性能监控**：

```python
# 在训练脚本中添加
import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    trainer.train()

# 导出性能分析
prof.export_chrome_trace("trace.json")

# 查看显存占用
print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
```

---

## 8.11 常见问题与调试

### 8.11.1 OOM 排查

**诊断步骤**：

```python
import torch

# 1. 检查当前显存占用
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

# 2. 查看显存分配历史
print(torch.cuda.memory_summary())

# 3. 启用显存分析
torch.cuda.memory._record_memory_history(enabled=True)
try:
    trainer.train()
except RuntimeError as e:
    if "out of memory" in str(e):
        torch.cuda.memory._dump_snapshot("oom.pickle")

# 4. 分析快照
import pickle
with open("oom.pickle", "rb") as f:
    snapshot = pickle.load(f)
    # 找到显存占用最大的张量
```

**解决方案优先级**：

```python
# 1. 减小batch size
per_device_train_batch_size = 1

# 2. 增加梯度累积
gradient_accumulation_steps = 32

# 3. 启用梯度检查点
gradient_checkpointing = True

# 4. 使用FSDP
fsdp = "full_shard"

# 5. CPU Offload
fsdp_config = {"fsdp_cpu_ram_efficient_loading": True}

# 6. 降低精度
bf16 = True

# 7. 减小序列长度
max_length = 512  # 从1024降到512
```

### 8.11.2 通信卡死

**常见原因**：

1. **不同步的集合通信**：

```python
# 错误：并非所有进程都参与
if rank == 0:
    dist.barrier()  # 其他进程不会执行，导致卡死

# 正确：所有进程都参与
dist.barrier()
```

2. **死锁**：

```python
# 错误：循环依赖
if rank == 0:
    dist.send(data, dst=1)
    dist.recv(data, src=1)
if rank == 1:
    dist.send(data, dst=0)
    dist.recv(data, src=0)

# 正确：使用非阻塞通信
req1 = dist.isend(data, dst=other_rank)
req2 = dist.irecv(buffer, src=other_rank)
req1.wait()
req2.wait()
```

3. **NCCL超时**：

```python
# 增加超时时间
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['NCCL_TIMEOUT'] = '7200'  # 2小时

dist.init_process_group(
    backend='nccl',
    timeout=datetime.timedelta(seconds=7200)
)
```

---

## 8.12 本章小结

### 核心要点

1. **Accelerate**：一行代码支持多种分布式策略
2. **DDP**：数据并行，适用于小模型（<10B）
3. **FSDP**：完全分片，适用于中大模型（10B-100B）
4. **DeepSpeed ZeRO**：三阶段优化，ZeRO-3可训练超大模型
5. **Pipeline 并行**：模型纵向切分，减少气泡
6. **张量并行**：模型横向切分，减少通信
7. **3D 并行**：组合所有策略，训练175B+模型

### 最佳实践

```python
# 小模型 (<7B)
# 使用 DDP + 混合精度
accelerate launch --mixed_precision bf16 train.py

# 中等模型 (7B-30B)
# 使用 FSDP
accelerate launch --config_file fsdp_config.yaml train.py

# 大模型 (30B-100B)
# 使用 DeepSpeed ZeRO-3
deepspeed --num_gpus 8 train.py --deepspeed ds_zero3_config.json

# 超大模型 (100B+)
# 使用 Megatron 3D并行
deepspeed pretrain_gpt.py \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 8 \
    --deepspeed_config ds_config.json
```

### 进一步阅读

- [Accelerate 文档](https://huggingface.co/docs/accelerate)
- [FSDP 教程](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [DeepSpeed 官方文档](https://www.deepspeed.ai/)
- [Megatron-LM 论文](https://arxiv.org/abs/1909.08053)

---

**下一章预告**：Chapter 9 将深入探讨**推理优化与部署**，包括 FlashAttention、KV cache、vLLM、TensorRT-LLM、ONNX等高效推理技术。

# Chapter 14: Accelerate 库完全指南

> **官方文档**: https://huggingface.co/docs/accelerate  
> **GitHub**: https://github.com/huggingface/accelerate  
> **发布版本**: Accelerate v0.27+（2026年1月）

## 14.1 Accelerate 设计哲学

### 14.1.1 统一的分布式训练接口

Hugging Face Accelerate 是一个旨在**将分布式训练的复杂性降到最低**的库。它的核心理念是：

> **"写一次代码，在任何配置下运行"**

#### 传统多 GPU 训练的痛点

在 Accelerate 出现之前，研究人员需要为不同硬件配置编写不同的代码：

```python
# 单 GPU 代码
model = Model()
model.to("cuda")
optimizer = torch.optim.AdamW(model.parameters())

for batch in dataloader:
    outputs = model(batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

```python
# 多 GPU DDP 代码（需要大量修改）
import torch.distributed as dist
dist.init_process_group(backend='nccl')
local_rank = int(os.environ["LOCAL_RANK"])

model = Model()
model = model.to(local_rank)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler)
optimizer = torch.optim.AdamW(model.parameters())

for batch in dataloader:
    batch = {k: v.to(local_rank) for k, v in batch.items()}
    outputs = model(batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

#### Accelerate 的解决方案

使用 Accelerate 后，**只需修改 3-5 行代码**，即可在单卡、多卡、混合精度、FSDP、DeepSpeed 等配置间自由切换：

```python
from accelerate import Accelerator

# 1. 创建 Accelerator 实例
accelerator = Accelerator()

model = Model()
optimizer = torch.optim.AdamW(model.parameters())
dataloader = DataLoader(dataset, batch_size=32)

# 2. 使用 prepare() 包装
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    outputs = model(batch)
    loss = outputs.loss
    # 3. 使用 backward() 替代 loss.backward()
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

**关键优势**：
- ✅ **代码统一**：同一份代码可在 CPU、单 GPU、多 GPU、TPU 上运行
- ✅ **自动设备管理**：`prepare()` 自动处理模型、数据的设备分配
- ✅ **混合精度支持**：无需手动 `autocast()` 和 `GradScaler`
- ✅ **梯度累积**：自动处理跨设备的梯度同步
- ✅ **Checkpoint 统一**：主进程保存，其他进程跳过

<div data-component="AccelerateWorkflow"></div>

---

### 14.1.2 与 Trainer 的关系

Accelerate 与 Hugging Face `Trainer` 的关系：

| 特性 | Trainer | Accelerate |
|------|---------|------------|
| **抽象层级** | 高层 API（隐藏训练循环） | 中层 API（保留训练循环控制） |
| **灵活性** | 通过 callback 和 `TrainingArguments` 定制 | 完全自定义训练逻辑 |
| **分布式支持** | 内部调用 Accelerate | 直接暴露分布式 API |
| **适用场景** | 标准监督学习、微调 | 强化学习、对抗训练、自定义损失 |
| **学习曲线** | 低（几乎零配置） | 中（需理解训练循环） |

**Trainer 底层使用 Accelerate**：

```python
# Trainer 内部实现（简化版）
class Trainer:
    def __init__(self, args):
        self.accelerator = Accelerator(
            mixed_precision=args.fp16 or args.bf16,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
    
    def training_step(self, model, inputs):
        outputs = model(**inputs)
        loss = outputs.loss
        self.accelerator.backward(loss)  # 内部调用 Accelerate
        return loss
```

**何时使用 Accelerate 而非 Trainer**：
1. 需要**自定义训练循环**（如 GAN 的生成器-判别器交替训练）
2. 非标准优化器更新策略（如梯度惩罚、梯度裁剪的特殊顺序）
3. 多模型联合训练（如多任务学习需要多个模型）
4. 希望**显式控制**每个训练步骤的细节

---

### 14.1.3 支持的后端

Accelerate 支持多种分布式后端，并通过**统一配置文件**管理：

| 后端 | 描述 | 适用场景 | 配置关键字 |
|------|------|----------|------------|
| **DDP** | PyTorch DistributedDataParallel | 单机多卡、小模型 | `use_ddp: true` |
| **FSDP** | Fully Sharded Data Parallel | 大模型（7B-70B） | `use_fsdp: true` |
| **DeepSpeed** | 微软 ZeRO 优化器 | 超大模型（70B+）、需要 Offload | `use_deepspeed: true` |
| **TPU** | Google Cloud TPU | TPU v2/v3/v4 | `tpu_config_file` |
| **MEGATRON-LM** | NVIDIA 3D 并行 | 千亿级模型 | `use_megatron_lm: true` |

**自动后端选择示例**：

```bash
# 配置向导会自动检测硬件并推荐后端
$ accelerate config

In which compute environment are you running?
  [0] This machine
  [1] AWS (Amazon SageMaker)
Please select: 0

Which type of machine are you using?
  [0] No distributed training
  [1] multi-CPU
  [2] multi-GPU
  [3] TPU
Please select: 2

How many machines are you using? 1
How many processes in total? 4  # 检测到 4 个 GPU

Do you want to use FSDP? [yes/NO]: yes
# 自动生成 default_config.yaml 文件
```

生成的配置文件 `~/.cache/huggingface/accelerate/default_config.yaml`：

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_sharding_strategy: 1  # FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
use_cpu: false
```

---

## 14.2 Accelerate 基础工作流

### 14.2.1 accelerate config 配置向导

`accelerate config` 是交互式配置工具，会生成适合当前硬件的配置文件。

#### 配置流程详解

```bash
$ accelerate config

# Step 1: 计算环境
In which compute environment are you running?
  [0] This machine
  [1] AWS (Amazon SageMaker)
Please select: 0

# Step 2: 分布式类型
What type of machine are you using?
  [0] No distributed training
  [1] multi-CPU
  [2] multi-GPU
  [3] TPU
Please select: 2

# Step 3: GPU 数量
How many different machines will you use? 1
How many processes in total will you use? 4

# Step 4: 混合精度
Do you wish to use FP16 or BF16 (mixed precision)?
  [0] no
  [1] fp16
  [2] bf16
  [3] fp8
Please select: 2

# Step 5: DeepSpeed（可选）
Do you want to use DeepSpeed? [yes/NO]: no

# Step 6: FSDP（可选）
Do you want to use FullyShardedDataParallel? [yes/NO]: yes

# FSDP 详细配置
What should be your sharding strategy?
  [0] FULL_SHARD (ZeRO-3)
  [1] SHARD_GRAD_OP (ZeRO-2)
  [2] NO_SHARD (DDP)
Please select: 0

Do you want to offload parameters to CPU? [yes/NO]: no

# 最终生成配置文件
accelerate configuration saved at ~/.cache/huggingface/accelerate/default_config.yaml
```

#### 手动编写配置文件

也可以跳过向导，直接创建 `accelerate_config.yaml`：

```yaml
# accelerate_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4  # 使用 4 个 GPU
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

**使用自定义配置文件启动**：

```bash
accelerate launch --config_file accelerate_config.yaml train.py
```

---

### 14.2.2 Accelerator 类核心 API

`Accelerator` 是 Accelerate 的核心类，提供以下关键方法：

#### 初始化参数

```python
from accelerate import Accelerator

accelerator = Accelerator(
    # 混合精度
    mixed_precision='bf16',  # 'no' | 'fp16' | 'bf16' | 'fp8'
    
    # 梯度累积
    gradient_accumulation_steps=4,  # 每 4 步更新一次参数
    
    # Logging
    log_with='tensorboard',  # 'tensorboard' | 'wandb' | 'comet_ml'
    project_dir='./outputs',  # checkpoint 保存路径
    
    # CPU Offload
    cpu=False,  # 强制使用 CPU
    
    # 设备放置策略
    device_placement=True,  # 自动将模型/数据移到设备
    
    # 分布式后端（通常从配置文件读取）
    # dispatch_batches、split_batches 等高级选项
)
```

#### prepare() 方法

**最重要的方法**，用于包装模型、优化器、数据加载器：

```python
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
```

**内部机制**：
1. **模型包装**：
   - 单 GPU → `model.to(device)`
   - 多 GPU DDP → `torch.nn.parallel.DistributedDataParallel(model)`
   - FSDP → `FullyShardedDataParallel(model)`
   
2. **优化器包装**：
   - 添加混合精度的 `GradScaler`（若启用 FP16）
   - 集成梯度累积逻辑
   
3. **数据加载器包装**：
   - 自动添加 `DistributedSampler`（多 GPU）
   - 处理 batch size 与 GPU 数量的关系

#### backward() 方法

```python
accelerator.backward(loss)
```

等价于：
```python
# 单 GPU
loss.backward()

# FP16 混合精度
scaler.scale(loss).backward()

# 梯度累积
(loss / gradient_accumulation_steps).backward()
```

#### 其他核心方法

```python
# 收集分布式结果
all_losses = accelerator.gather(loss)  # 从所有进程收集

# 等待所有进程
accelerator.wait_for_everyone()

# 打印（仅主进程）
accelerator.print(f"Epoch {epoch} completed")

# 主进程检查
if accelerator.is_main_process:
    model.save_pretrained("./outputs")

# 保存/加载 checkpoint
accelerator.save_state("checkpoint_dir")  # 保存优化器、调度器、RNG 状态
accelerator.load_state("checkpoint_dir")

# Logging
accelerator.log({"train_loss": loss.item(), "lr": lr}, step=global_step)

# 上下文管理器
with accelerator.accumulate(model):  # 梯度累积上下文
    outputs = model(batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

<div data-component="AcceleratorAPIDemo"></div>

---

### 14.2.3 代码修改最小化（3 行改动）

从单 GPU 代码迁移到 Accelerate 的**最小修改示例**：

#### 原始单 GPU 代码

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

# 模型和数据
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.to("cuda")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

train_dataset = ...
train_dataloader = DataLoader(train_dataset, batch_size=8)

# 训练循环
for epoch in range(3):
    for batch in train_dataloader:
        batch = {k: v.to("cuda") for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"Loss: {loss.item()}")
```

#### Accelerate 版本（仅 3 处修改）

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from accelerate import Accelerator  # 导入

# ✅ 修改 1: 创建 Accelerator
accelerator = Accelerator()

# 模型和数据
model = AutoModelForCausalLM.from_pretrained("gpt2")
# model.to("cuda")  # ❌ 删除手动设备分配
tokenizer = AutoTokenizer.from_pretrained("gpt2")
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

train_dataset = ...
train_dataloader = DataLoader(train_dataset, batch_size=8)

# ✅ 修改 2: 使用 prepare() 包装
model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

# 训练循环
for epoch in range(3):
    for batch in train_dataloader:
        # batch = {k: v.to("cuda") for k, v in batch.items()}  # ❌ 删除
        outputs = model(**batch)
        loss = outputs.loss
        
        # ✅ 修改 3: 使用 accelerator.backward()
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        
        accelerator.print(f"Loss: {loss.item()}")  # 仅主进程打印
```

**3 行修改总结**：
1. `accelerator = Accelerator()`
2. `model, optimizer, dataloader = accelerator.prepare(...)`
3. `accelerator.backward(loss)` 替换 `loss.backward()`

**额外收益**：
- 自动支持多 GPU（无需修改代码，仅需 `accelerate launch --num_processes=4 train.py`）
- 自动混合精度（添加 `--mixed_precision bf16`）
- 自动梯度累积（`Accelerator(gradient_accumulation_steps=4)`）

---

### 14.2.4 accelerate launch 启动脚本

#### 基础启动

```bash
# 单 GPU
accelerate launch train.py

# 多 GPU（4 卡）
accelerate launch --num_processes=4 train.py

# 指定 GPU
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes=3 train.py

# 使用配置文件
accelerate launch --config_file fsdp_config.yaml train.py

# 混合精度
accelerate launch --mixed_precision bf16 --num_processes=4 train.py
```

#### 多机训练

**主节点（机器 0）**：
```bash
accelerate launch \
    --num_processes=8 \  # 总进程数（2 机器 × 4 GPU）
    --num_machines=2 \
    --machine_rank=0 \
    --main_process_ip=192.168.1.100 \
    --main_process_port=29500 \
    train.py
```

**从节点（机器 1）**：
```bash
accelerate launch \
    --num_processes=8 \
    --num_machines=2 \
    --machine_rank=1 \
    --main_process_ip=192.168.1.100 \
    --main_process_port=29500 \
    train.py
```

#### 与 torchrun 的对比

```bash
# Accelerate 方式
accelerate launch --num_processes=4 train.py

# 等价的 torchrun 方式
torchrun --nproc_per_node=4 train.py
```

**Accelerate 的优势**：
- 统一的配置文件管理
- 自动处理环境变量（`RANK`、`LOCAL_RANK`、`WORLD_SIZE`）
- 更友好的错误提示
- 支持 FSDP/DeepSpeed 的高级配置

---

## 14.3 从单卡到多卡

### 14.3.1 单 GPU 训练

最简单的场景，Accelerate 会自动检测并使用单个 GPU：

```python
from accelerate import Accelerator
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

accelerator = Accelerator()

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 准备数据
dataset = load_dataset("glue", "sst2", split="train[:1000]")
def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
train_dataloader = DataLoader(tokenized_dataset, batch_size=16, shuffle=True)

# 准备训练
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

# 训练
model.train()
for epoch in range(3):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
    
    accelerator.print(f"Epoch {epoch} completed, Loss: {loss.item():.4f}")
```

**预期输出**：

```
Epoch 0 completed, Loss: 0.6234
Epoch 1 completed, Loss: 0.3421
Epoch 2 completed, Loss: 0.1876
```

**单 GPU 下的 Accelerate 行为**：
- `prepare()` 将模型移至 `cuda:0`
- `backward()` 等价于 `loss.backward()`
- 不会创建分布式进程

---

### 14.3.2 多 GPU 单机（DDP）

**无需修改代码**，仅需更改启动命令：

```bash
# 使用 4 个 GPU
accelerate launch --multi_gpu --num_processes=4 train.py
```

**或使用配置文件**：

```yaml
# ddp_config.yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 4
mixed_precision: bf16
use_cpu: false
```

```bash
accelerate launch --config_file ddp_config.yaml train.py
```

#### DDP 内部机制

当检测到多 GPU 时，Accelerate 会：

1. **初始化进程组**：
   ```python
   torch.distributed.init_process_group(backend='nccl')
   ```

2. **为每个进程分配 GPU**：
   - 进程 0 → `cuda:0`
   - 进程 1 → `cuda:1`
   - 进程 2 → `cuda:2`
   - 进程 3 → `cuda:3`

3. **包装模型为 DDP**：
   ```python
   model = torch.nn.parallel.DistributedDataParallel(
       model,
       device_ids=[local_rank],
       output_device=local_rank
   )
   ```

4. **数据加载器添加采样器**：
   ```python
   sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
   dataloader = DataLoader(dataset, sampler=sampler, batch_size=16)
   ```

5. **梯度同步**：
   - 每个 GPU 独立前向传播
   - `backward()` 时自动 all-reduce 梯度
   - 所有 GPU 使用相同的梯度更新参数

<div data-component="DistributedCommunicationVisualizer"></div>

#### 有效 Batch Size 计算

```python
# DDP 配置
num_gpus = 4
per_device_batch_size = 8
gradient_accumulation_steps = 2

# 有效 batch size 计算
effective_batch_size = (
    per_device_batch_size 
    * num_gpus 
    * gradient_accumulation_steps
)
# = 8 × 4 × 2 = 64
```

**代码示例**：

```python
accelerator = Accelerator(gradient_accumulation_steps=2)
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    # 使用累积上下文管理器
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

启动时 batch size 为 8，但每 2 步才更新一次参数，因此每个 GPU 的有效 batch size = 8 × 2 = 16，总有效 batch size = 16 × 4 = 64。

---

### 14.3.3 多机多卡集群

#### 环境要求

1. **网络互通**：所有节点可通过 IP 互相访问
2. **共享文件系统**（可选但推荐）：NFS、Lustre 等
3. **相同 CUDA/PyTorch 版本**
4. **SSH 免密登录**（若使用自动启动脚本）

#### 手动启动方式

**机器 0（192.168.1.10，4 个 GPU）**：

```bash
export MASTER_ADDR=192.168.1.10
export MASTER_PORT=29500
export WORLD_SIZE=8  # 2 机器 × 4 GPU
export RANK=0

accelerate launch \
    --num_processes=8 \
    --num_machines=2 \
    --machine_rank=0 \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    train.py
```

**机器 1（192.168.1.11，4 个 GPU）**：

```bash
export MASTER_ADDR=192.168.1.10
export MASTER_PORT=29500
export WORLD_SIZE=8
export RANK=1

accelerate launch \
    --num_processes=8 \
    --num_machines=2 \
    --machine_rank=1 \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    train.py
```

#### 使用 SLURM 集群

```bash
#!/bin/bash
#SBATCH --job-name=accelerate_train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

# 加载环境
module load cuda/11.8
source ~/miniconda3/bin/activate transformers_env

# 获取主节点信息
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# 启动训练
srun accelerate launch \
    --num_processes=$SLURM_NTASKS \
    --num_machines=$SLURM_NNODES \
    --machine_rank=$SLURM_NODEID \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    train.py
```

提交任务：
```bash
sbatch train_slurm.sh
```

---

### 14.3.4 混合精度集成

#### FP16 vs BF16 选择

| 特性 | FP16 | BF16 |
|------|------|------|
| **动态范围** | 小（5.96e-08 ~ 65504） | 大（1.18e-38 ~ 3.39e+38） |
| **精度** | 高（10 位尾数） | 低（7 位尾数） |
| **溢出风险** | 高（需要 Loss Scaling） | 低（无需 Loss Scaling） |
| **硬件支持** | V100+、A100、H100 | A100+、H100（需 Ampere 架构） |
| **推荐场景** | 小模型、推理 | 大模型训练（LLM） |

#### 启用混合精度

**方式 1：启动参数**

```bash
accelerate launch --mixed_precision bf16 --num_processes=4 train.py
```

**方式 2：代码中指定**

```python
accelerator = Accelerator(mixed_precision='bf16')
```

**方式 3：配置文件**

```yaml
# config.yaml
mixed_precision: bf16
```

#### FP16 的 Loss Scaling

FP16 训练时，Accelerate 会自动应用动态 Loss Scaling：

```python
# 内部实现（简化版）
from torch.cuda.amp import GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast(dtype=torch.float16):
        outputs = model(**batch)
        loss = outputs.loss
    
    # 放大 loss 防止梯度下溢
    scaler.scale(loss).backward()
    
    # Unscale 梯度并检查 inf/nan
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 更新参数
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

**用户代码**（无需手动处理）：

```python
accelerator = Accelerator(mixed_precision='fp16')
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)  # 自动处理 scaling
    optimizer.step()
    optimizer.zero_grad()
```

#### BF16 优势示例

```python
import torch

# FP16 会溢出
fp16_large = torch.tensor([65000.0], dtype=torch.float16)
fp16_result = fp16_large * 2  # 结果：inf（溢出）

# BF16 不会溢出
bf16_large = torch.tensor([65000.0], dtype=torch.bfloat16)
bf16_result = bf16_large * 2  # 结果：130000.0（正常）

print(f"FP16: {fp16_result.item()}")  # inf
print(f"BF16: {bf16_result.item()}")  # 130000.0
```

---

## 14.4 Accelerator 高级功能

### 14.4.1 梯度累积

梯度累积允许用**更小的 batch size** 模拟**更大的 batch size**，节省显存。

#### 基础用法

```python
accelerator = Accelerator(gradient_accumulation_steps=4)
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

**工作原理**：
- 前 3 次迭代：仅计算梯度，**不更新参数**
- 第 4 次迭代：累积的梯度平均后更新参数

#### 手动实现对比

```python
# 手动实现梯度累积（不推荐）
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    outputs = model(**batch)
    loss = outputs.loss / accumulation_steps  # ❌ 需要手动除以步数
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Accelerate 自动实现（推荐）
with accelerator.accumulate(model):
    outputs = model(**batch)
    loss = outputs.loss  # ✅ 无需手动除法
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

#### 分布式环境下的梯度累积

在多 GPU 环境中，梯度累积需要特殊处理：

```python
# 4 个 GPU，每个 batch_size=8，累积 4 步
# 有效 batch size = 8 × 4 GPU × 4 步 = 128

accelerator = Accelerator(gradient_accumulation_steps=4)

for batch in dataloader:
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs.loss
        
        # Accelerate 会在累积步数结束时自动同步梯度
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

**内部机制**：
- 累积期间：`model.require_backward_grad_sync = False`（跳过梯度同步）
- 最后一步：`model.require_backward_grad_sync = True`（执行 all-reduce）

---

### 14.4.2 Checkpoint 保存与恢复

#### 保存完整训练状态

```python
# 保存 checkpoint（包含模型、优化器、调度器、RNG 状态）
output_dir = "checkpoint-1000"
accelerator.save_state(output_dir)
```

生成的目录结构：

```
checkpoint-1000/
├── pytorch_model.bin       # 模型权重
├── optimizer.bin           # 优化器状态
├── scheduler.bin           # 学习率调度器
├── random_states_0.pkl     # RNG 状态（进程 0）
├── random_states_1.pkl     # RNG 状态（进程 1）
├── ...
└── scaler.pt               # GradScaler 状态（FP16 时）
```

#### 恢复训练

```python
# 恢复所有状态
accelerator.load_state("checkpoint-1000")

# 继续训练
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

#### 仅保存模型权重

```python
# 仅保存模型（用于推理或后续微调）
unwrapped_model = accelerator.unwrap_model(model)
accelerator.save(unwrapped_model.state_dict(), "model_weights.bin")
```

**为什么需要 `unwrap_model`**？

在分布式环境下，`prepare()` 会包装模型为 `DDP` 或 `FSDP`，导致 state_dict 的 key 前缀发生变化：

```python
# 原始模型
model.transformer.h.0.attn.c_attn.weight

# DDP 包装后
module.transformer.h.0.attn.c_attn.weight  # 多了 "module." 前缀
```

`unwrap_model()` 可以去除包装，恢复原始结构：

```python
wrapped_model = accelerator.prepare(model)
print(list(wrapped_model.state_dict().keys())[:3])
# ['module.transformer.wte.weight', 'module.transformer.wpe.weight', ...]

unwrapped_model = accelerator.unwrap_model(wrapped_model)
print(list(unwrapped_model.state_dict().keys())[:3])
# ['transformer.wte.weight', 'transformer.wpe.weight', ...]
```

#### 与 Hugging Face Hub 集成

```python
from huggingface_hub import HfApi

# 保存到本地
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained("./my_finetuned_model")
tokenizer.save_pretrained("./my_finetuned_model")

# 上传到 Hub
if accelerator.is_main_process:
    api = HfApi()
    api.upload_folder(
        folder_path="./my_finetuned_model",
        repo_id="username/my-model",
        repo_type="model"
    )
```

---

### 14.4.3 Logging 与同步

#### 集成 TensorBoard

```python
from accelerate import Accelerator

accelerator = Accelerator(log_with="tensorboard", project_dir="./logs")

# 初始化 tracker
accelerator.init_trackers(project_name="my_experiment")

# 训练循环中记录指标
for step, batch in enumerate(dataloader):
    outputs = model(**batch)
    loss = outputs.loss
    
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    
    # 记录 loss
    accelerator.log({"train_loss": loss.item()}, step=step)

# 结束记录
accelerator.end_training()
```

启动 TensorBoard：
```bash
tensorboard --logdir ./logs
```

#### 集成 Weights & Biases

```python
accelerator = Accelerator(log_with="wandb")

# 初始化（仅主进程登录）
accelerator.init_trackers(
    project_name="transformers-training",
    config={
        "learning_rate": 2e-5,
        "batch_size": 16,
        "epochs": 3
    },
    init_kwargs={"wandb": {"entity": "my-team"}}
)

# 记录指标
accelerator.log({
    "train_loss": loss.item(),
    "learning_rate": optimizer.param_groups[0]['lr'],
    "epoch": epoch
}, step=global_step)

# 记录模型
if accelerator.is_main_process:
    wandb.save("model_checkpoint.bin")
```

#### 多进程日志同步

```python
# gather() 收集所有进程的值
losses = []
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    losses.append(loss)
    
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()

# 收集所有 GPU 的 loss
all_losses = accelerator.gather(torch.stack(losses))

# 仅主进程打印
if accelerator.is_main_process:
    avg_loss = all_losses.mean().item()
    print(f"Average loss across all GPUs: {avg_loss:.4f}")
```

**gather() 详解**：

```python
# 假设 4 个 GPU，每个计算了一个 loss
# GPU 0: loss = 0.5
# GPU 1: loss = 0.6
# GPU 2: loss = 0.55
# GPU 3: loss = 0.58

loss_tensor = torch.tensor([loss])  # 当前进程的 loss
all_losses = accelerator.gather(loss_tensor)

# 结果（仅在主进程有效，其他进程为 None）
# all_losses = tensor([0.5, 0.6, 0.55, 0.58])
```

---

### 14.4.4 主进程控制（main_process_first）

某些操作（如数据预处理、模型下载）只需在主进程执行，其他进程等待：

#### 数据集预处理

```python
from datasets import load_dataset

with accelerator.main_process_first():
    # 仅主进程下载和处理数据集
    dataset = load_dataset("glue", "sst2")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 所有进程在此同步，确保数据集已准备好
dataloader = DataLoader(tokenized_dataset, batch_size=16)
```

**为什么需要这个**？

在多 GPU 环境下，如果所有进程同时下载数据集，会导致：
- 网络带宽浪费
- 文件系统竞争（多个进程写入同一缓存文件）
- 可能的数据损坏

`main_process_first()` 确保：
1. 主进程（Rank 0）先执行
2. 其他进程在 barrier 处等待
3. 主进程完成后释放 barrier
4. 所有进程继续执行

#### 模型下载

```python
with accelerator.main_process_first():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    # 模型会缓存到 ~/.cache/huggingface/hub/

# 其他进程直接从缓存加载，无需重复下载
```

#### 自定义操作

```python
if accelerator.is_main_process:
    # 仅主进程执行
    print("Preparing data...")
    prepare_custom_data()

# 同步所有进程
accelerator.wait_for_everyone()

# 所有进程继续
dataloader = load_prepared_data()
```

---

## 14.5 与 Trainer 集成

### 14.5.1 Trainer 自动检测 Accelerate 配置

Hugging Face `Trainer` **内部使用 Accelerate**，因此 `accelerate config` 生成的配置会自动生效：

```python
from transformers import Trainer, TrainingArguments

# 配置训练参数
training_args = TrainingArguments(
    output_dir="./outputs",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    # Trainer 会自动读取 ~/.cache/huggingface/accelerate/default_config.yaml
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# 启动训练
trainer.train()
```

启动命令：

```bash
# 方式 1: 使用 accelerate launch（推荐）
accelerate launch train_with_trainer.py

# 方式 2: 直接运行（Trainer 会自动检测配置）
python train_with_trainer.py
```

#### Trainer 内部 Accelerate 集成

```python
# Trainer 内部实现（简化版）
class Trainer:
    def __init__(self, args):
        # 自动创建 Accelerator
        self.accelerator = Accelerator(
            mixed_precision=self._get_mixed_precision(args),
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            log_with=args.report_to,
            project_dir=args.logging_dir
        )
        
        # 使用 prepare() 包装
        self.model = self.accelerator.prepare(self.model)
        self.optimizer = self.accelerator.prepare(self.optimizer)
    
    def training_step(self, model, inputs):
        outputs = model(**inputs)
        loss = outputs.loss
        self.accelerator.backward(loss)  # 内部调用
        return loss
```

---

### 14.5.2 自定义训练循环 vs Trainer

#### 何时使用自定义训练循环（Accelerate）

```python
from accelerate import Accelerator

accelerator = Accelerator()

# 完全自定义的训练逻辑
generator = Generator()
discriminator = Discriminator()

gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

generator, discriminator, gen_optimizer, disc_optimizer = accelerator.prepare(
    generator, discriminator, gen_optimizer, disc_optimizer
)

for epoch in range(100):
    for real_images in dataloader:
        # 训练判别器
        fake_images = generator(noise)
        disc_real = discriminator(real_images)
        disc_fake = discriminator(fake_images.detach())
        
        disc_loss = -torch.mean(disc_real) + torch.mean(disc_fake)
        accelerator.backward(disc_loss)
        disc_optimizer.step()
        disc_optimizer.zero_grad()
        
        # 训练生成器
        disc_fake = discriminator(fake_images)
        gen_loss = -torch.mean(disc_fake)
        accelerator.backward(gen_loss)
        gen_optimizer.step()
        gen_optimizer.zero_grad()
```

**适用场景**：
- GAN（生成器-判别器交替训练）
- 强化学习（策略网络 + 价值网络）
- 多任务学习（多个模型、多个损失函数）
- 需要自定义优化器更新策略

#### 何时使用 Trainer

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    logging_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_accuracy
)

trainer.train()
```

**适用场景**：
- 标准监督学习（分类、回归、序列标注）
- 预训练模型微调
- 不需要自定义训练循环
- 希望使用内置的 logging、evaluation、checkpoint

---

## 14.6 调试技巧

### 14.6.1 ACCELERATE_DEBUG_MODE

启用调试模式会打印详细的分布式信息：

```bash
ACCELERATE_DEBUG_MODE=1 accelerate launch --num_processes=4 train.py
```

**输出示例**：

```
[DEBUG] Initialized process group: rank=0, world_size=4
[DEBUG] Device assignment: cuda:0
[DEBUG] Model wrapped with DistributedDataParallel
[DEBUG] DataLoader using DistributedSampler
[DEBUG] Gradient accumulation steps: 1
[DEBUG] Mixed precision: bf16
...
```

#### 自定义调试信息

```python
import os

if os.environ.get("ACCELERATE_DEBUG_MODE"):
    accelerator.print(f"[DEBUG] Rank {accelerator.process_index}: Starting training")
    accelerator.print(f"[DEBUG] Model device: {next(model.parameters()).device}")
    accelerator.print(f"[DEBUG] Dataloader length: {len(dataloader)}")
```

---

### 14.6.2 gather() 与 reduce() 操作

#### gather() - 收集所有进程的张量

```python
# 每个进程计算一个值
local_accuracy = compute_accuracy(predictions, labels)

# 收集所有进程的准确率
all_accuracies = accelerator.gather(local_accuracy)

if accelerator.is_main_process:
    global_accuracy = all_accuracies.mean()
    print(f"Global Accuracy: {global_accuracy:.4f}")
```

**示例**：

```python
# GPU 0: local_accuracy = 0.85
# GPU 1: local_accuracy = 0.88
# GPU 2: local_accuracy = 0.82
# GPU 3: local_accuracy = 0.90

all_accuracies = accelerator.gather(torch.tensor([local_accuracy]))
# 结果: tensor([0.85, 0.88, 0.82, 0.90])

global_accuracy = all_accuracies.mean()
# 结果: 0.8625
```

#### reduce() - 聚合操作

```python
# 计算所有进程的 loss 总和
total_loss = accelerator.reduce(loss, reduction="sum")

# 计算平均值
avg_loss = accelerator.reduce(loss, reduction="mean")

# 最大值
max_loss = accelerator.reduce(loss, reduction="max")
```

**内部实现**：

```python
# reduce() 使用 all_reduce 通信原语
if reduction == "sum":
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
elif reduction == "mean":
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor /= world_size
```

---

### 14.6.3 死锁排查

#### 常见死锁原因

**原因 1：不同进程执行不同的代码路径**

```python
# ❌ 错误：某些进程跳过了集体通信操作
if accelerator.is_main_process:
    loss = model(batch).loss
    accelerator.backward(loss)  # 仅主进程调用，其他进程在 barrier 处等待 → 死锁
```

**修复**：

```python
# ✅ 正确：所有进程都执行相同的通信操作
loss = model(batch).loss
accelerator.backward(loss)  # 所有进程都调用
```

**原因 2：gather() 在非主进程访问结果**

```python
# ❌ 错误
all_losses = accelerator.gather(loss)
avg_loss = all_losses.mean()  # 非主进程中 all_losses 是 None → 崩溃
```

**修复**：

```python
# ✅ 正确
all_losses = accelerator.gather(loss)
if accelerator.is_main_process:
    avg_loss = all_losses.mean()  # 仅主进程处理
```

**原因 3：数据加载器长度不一致**

```python
# ❌ 错误：某些进程的 dataloader 提前结束
for batch in dataloader:  # 不同进程的迭代次数不同
    loss = model(batch).loss
    accelerator.backward(loss)  # 某些进程已退出循环 → 死锁
```

**修复**：

```python
# ✅ 方式 1: 确保所有进程的 dataloader 长度相同
sampler = DistributedSampler(dataset, drop_last=True)

# ✅ 方式 2: 使用 accelerator.prepare() 自动处理
dataloader = accelerator.prepare(dataloader)
```

#### 调试工具

```bash
# 设置超时（默认 30 分钟）
export NCCL_TIMEOUT=600  # 10 分钟

# 启用 NCCL 调试信息
export NCCL_DEBUG=INFO

# 启动训练
accelerate launch --num_processes=4 train.py
```

**检查进程状态**：

```bash
# 监控 GPU 进程
watch -n 1 nvidia-smi

# 检查僵死进程
ps aux | grep python | grep train.py

# 强制终止
pkill -9 -f train.py
```

---

## 总结与最佳实践

### ✅ Accelerate 使用检查清单

**代码修改**：
- [ ] 导入 `Accelerator`
- [ ] 创建 `accelerator = Accelerator(...)`
- [ ] 使用 `prepare()` 包装模型、优化器、数据加载器
- [ ] 将 `loss.backward()` 替换为 `accelerator.backward(loss)`
- [ ] 使用 `accelerator.print()` 替代 `print()`（避免重复输出）

**配置文件**：
- [ ] 运行 `accelerate config` 生成配置
- [ ] 检查 `~/.cache/huggingface/accelerate/default_config.yaml`
- [ ] 或创建自定义 `accelerate_config.yaml`

**启动命令**：
- [ ] 单 GPU：`accelerate launch train.py`
- [ ] 多 GPU：`accelerate launch --num_processes=N train.py`
- [ ] 自定义配置：`accelerate launch --config_file config.yaml train.py`

**进阶功能**：
- [ ] 混合精度：`mixed_precision='bf16'`
- [ ] 梯度累积：`gradient_accumulation_steps=N`
- [ ] Logging：`log_with='tensorboard'` 或 `'wandb'`
- [ ] Checkpoint：`accelerator.save_state()` / `load_state()`

### ⚠️ 常见陷阱

1. **忘记 `prepare()`**：直接使用未包装的模型/优化器
2. **设备不一致**：手动 `.to(device)` 与 `prepare()` 冲突
3. **打印重复**：使用 `print()` 而非 `accelerator.print()`
4. **Checkpoint 保存**：忘记 `unwrap_model()` 导致加载失败
5. **集体通信不一致**：某些进程跳过 `backward()` 或 `gather()`

### 📊 性能优化建议

| 优化项 | 建议 | 预期提升 |
|--------|------|----------|
| **混合精度** | 使用 BF16（A100+） | 1.5-2× 速度 |
| **梯度累积** | 增大有效 batch size | 提高 GPU 利用率 |
| **FSDP** | 7B+ 模型使用 FSDP | 节省 60-80% 显存 |
| **Flash Attention** | `use_flash_attention_2=True` | 节省 30-50% 显存 |
| **Gradient Checkpointing** | 大模型启用 | 节省 40-60% 显存（牺牲 20% 速度） |

### 🔗 扩展阅读

- **官方文档**: https://huggingface.co/docs/accelerate
- **GitHub 示例**: https://github.com/huggingface/accelerate/tree/main/examples
- **FSDP 教程**: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- **DeepSpeed 集成**: https://huggingface.co/docs/accelerate/usage_guides/deepspeed

---

**下一章预告**：Chapter 15 将深入探讨 **FSDP（Fully Sharded Data Parallel）**，包括 ZeRO 优化器的三个阶段、分片策略、与 DeepSpeed 的对比，以及如何在单机 4 卡上训练 70B 参数的大模型。

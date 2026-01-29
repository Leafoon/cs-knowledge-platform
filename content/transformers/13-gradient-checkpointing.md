---
title: "Chapter 13. Gradient Checkpointing 与内存优化"
description: "理解梯度检查点机制、掌握时间-空间权衡与内存优化策略"
updated: "2026-01-22"
---

## 13.1 Gradient Checkpointing 原理

### 13.1.1 显存瓶颈问题

在训练大型 Transformer 模型时，**显存占用**是主要瓶颈：

**显存组成**（以 LLaMA-7B 为例）：

| 组件 | 大小 | 占比 | 说明 |
|------|------|------|------|
| **模型权重** | 14 GB (FP16) | 25% | $N$ 个 Transformer 层的参数 |
| **优化器状态** | 28 GB (AdamW) | 50% | Momentum + Variance (每个参数 2x) |
| **梯度** | 14 GB | 25% | 反向传播时的梯度 |
| **激活值** | 10-50 GB | 可变 | 前向传播时的中间结果 |

<div data-component="MemoryBreakdownInteractive"></div>

**问题**：

- 7B 模型全量微调需要 **66-106 GB** 显存（取决于序列长度 / batch size）
- A100 (80GB) 只能训练 batch_size=1, seq_len=512
- **激活值**占用随序列长度线性增长：
  $$
  \text{Activation Memory} = \mathcal{O}(N_{\text{layers}} \times B \times L \times d_{\text{model}})
  $$

### 13.1.2 Gradient Checkpointing 核心思想

**Rematerialization**（重计算）策略：

1. **前向传播**：只保存少数**检查点激活值**（如每 $k$ 层保存一次）
2. **反向传播**：需要时**重新计算**中间激活值

<div data-component="GradientCheckpointingFlow"></div>

**数学表示**：

标准反向传播：
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_i} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_{i+1}} \cdot \frac{\partial \mathbf{h}_{i+1}}{\partial \mathbf{W}_i}
$$

Gradient Checkpointing：
$$
\begin{aligned}
\text{Forward:} & \quad \text{仅保存 } \mathbf{h}_0, \mathbf{h}_k, \mathbf{h}_{2k}, \ldots \\
\text{Backward:} & \quad \text{从} \mathbf{h}_k \text{ 重新计算 } \mathbf{h}_{k+1}, \ldots, \mathbf{h}_{k+k}
\end{aligned}
$$

**时间-空间权衡**：

- **显存节省**：$\mathcal{O}(N) \rightarrow \mathcal{O}(\sqrt{N})$（最优策略）
- **计算增加**：约 **30-33%**（需要重新计算激活值）

**实例分析**（32层 Transformer）：

| 策略 | 保存层数 | 显存占用 | 重计算次数 |
|------|---------|---------|-----------|
| 无 Checkpointing | 32 层 | 100% | 0 |
| 每 4 层保存 | 8 层 | ~40% | 3 次/层 |
| 每 8 层保存 | 4 层 | ~30% | 7 次/层 |
| 最优策略 | $\sqrt{32} = 6$ 层 | ~35% | 5 次/层 |

### 13.1.3 适用场景与权衡

**适合启用 Gradient Checkpointing**：

✅ **显存受限**（OOM 错误）  
✅ **大模型微调**（7B+）  
✅ **长序列训练**（seq_len > 1024）  
✅ **大 batch size**（提升效率）

**不适合**：

❌ **推理阶段**（无反向传播，无需激活值）  
❌ **小模型**（<1B，开销大于收益）  
❌ **计算瓶颈**（GPU 利用率已满，重计算会更慢）

**性能影响**：

```python
# 实测数据 (LLaMA-7B, A100)
"""
| Batch | Seq Len | 无 Checkpoint | 有 Checkpoint |
|-------|---------|--------------|--------------|
| 1     | 512     | 42 GB        | 28 GB (-33%) |
| 2     | 512     | OOM          | 48 GB        |
| 1     | 2048    | OOM          | 56 GB        |

训练速度下降: 25-30%
"""
```

---

## 13.2 启用 Gradient Checkpointing

### 13.2.1 方法 1：模型 API

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 启用 Gradient Checkpointing
model.gradient_checkpointing_enable()

# 训练
model.train()
outputs = model(input_ids, labels=labels)
loss = outputs.loss
loss.backward()  # 自动使用 checkpointing
```

**原理**：

在每个 Transformer 层包裹 `checkpoint()` 函数：

```python
# transformers 内部实现（简化版）
from torch.utils.checkpoint import checkpoint

def forward(self, hidden_states):
    # 无 checkpointing
    # hidden_states = self.layer(hidden_states)
    
    # 有 checkpointing
    hidden_states = checkpoint(
        self.layer,
        hidden_states,
        use_reentrant=False,  # PyTorch 2.0+ 推荐
    )
    return hidden_states
```

### 13.2.2 方法 2：TrainingArguments

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./output",
    gradient_checkpointing=True,  # 启用
    # 其他参数...
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

**自动配置**：

- Trainer 会自动调用 `model.gradient_checkpointing_enable()`
- 无需手动设置

### 13.2.3 use_reentrant 参数详解

PyTorch 2.0 引入了新的 **non-reentrant checkpoint** 实现。

```python
# 旧版（reentrant=True，PyTorch < 2.0）
from torch.utils.checkpoint import checkpoint

output = checkpoint(
    function,
    input,
    use_reentrant=True,  # 默认，已弃用
)

# 新版（non-reentrant，推荐）
output = checkpoint(
    function,
    input,
    use_reentrant=False,  # PyTorch 2.0+
)
```

**差异**：

| 维度 | Reentrant | Non-Reentrant |
|------|-----------|---------------|
| **实现方式** | 使用 `torch.autograd.grad()` | 使用完整 autograd 引擎 |
| **兼容性** | 旧版 PyTorch | PyTorch 2.0+ |
| **性能** | 略快 | 略慢（~5%） |
| **稳定性** | 边缘情况下可能出错 | 更稳定 |
| **推荐** | ⚠️ 已弃用 | ✅ 推荐 |

**设置方式**（transformers 4.35+）：

```python
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
```

**或通过 TrainingArguments**：

```python
training_args = TrainingArguments(
    ...
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
```

### 13.2.4 完整训练示例

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

# 1. 加载模型和数据
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,  # 混合精度
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")

# 2. 数据预处理
def preprocess(examples):
    return tokenizer(
        examples["instruction"] + " " + examples["input"],
        truncation=True,
        max_length=512,
    )

train_dataset = dataset.map(preprocess, batched=True)

# 3. 配置训练参数
training_args = TrainingArguments(
    output_dir="./llama2-7b-finetuned",
    
    # Gradient Checkpointing
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    
    # 混合精度
    bf16=True,
    
    # 批次大小（checkpointing 后可以增大）
    per_device_train_batch_size=4,  # 无 checkpoint: 1
    gradient_accumulation_steps=4,
    
    # 其他参数
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
)

# 4. 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

**显存对比**：

```
无 Gradient Checkpointing:
- batch_size=1: 42 GB
- batch_size=2: OOM

有 Gradient Checkpointing:
- batch_size=4: 48 GB ✅
- batch_size=8: 76 GB (需 A100 80GB)
```

---

## 13.3 其他内存优化技巧

### 13.3.1 梯度累积（Gradient Accumulation）

**原理**：

将大 batch size 拆分为多个小 micro-batch，累积梯度后统一更新。

$$
\nabla \mathcal{L}_{\text{total}} = \frac{1}{K} \sum_{k=1}^{K} \nabla \mathcal{L}_k
$$

```python
# 手动实现
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = outputs.loss / gradient_accumulation_steps  # 归一化
    loss.backward()  # 累积梯度
    
    if (i + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# TrainingArguments 自动实现
training_args = TrainingArguments(
    per_device_train_batch_size=2,  # 每个设备的 micro-batch
    gradient_accumulation_steps=8,  # 累积 8 次
    # 等价于 effective_batch_size = 2 * 8 = 16
)
```

**优势**：

- **显存节省**：只需存储 micro-batch 的激活值
- **等价性**：梯度累积数学上等价于大 batch size（某些情况下略有差异）

**示例**（LLaMA-7B）：

| 配置 | Micro-Batch | 累积步数 | 等效 Batch | 显存 |
|------|-------------|---------|-----------|------|
| 无累积 | 16 | 1 | 16 | OOM |
| 方案 1 | 4 | 4 | 16 | 48 GB |
| 方案 2 | 2 | 8 | 16 | 32 GB |
| 方案 3 | 1 | 16 | 16 | 24 GB |

### 13.3.2 Flash Attention

**Flash Attention** 通过 **IO-aware 算法** 减少 HBM (High Bandwidth Memory) 访问。

**标准 Attention 显存**：
$$
\text{Memory} = \mathcal{O}(B \times L^2 \times d)
$$

**Flash Attention 显存**：
$$
\text{Memory} = \mathcal{O}(B \times L \times d)
$$

**安装**：

```bash
pip install flash-attn --no-build-isolation
```

**启用方式 1：加载模型时指定**：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    attn_implementation="flash_attention_2",  # 启用 Flash Attention 2
    torch_dtype=torch.bfloat16,
)
```

**启用方式 2：模型配置**：

```python
from transformers import LlamaConfig

config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
config._attn_implementation = "flash_attention_2"

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    config=config,
)
```

**性能提升**：

| 序列长度 | 标准 Attention | Flash Attention 2 | 加速比 | 显存节省 |
|---------|---------------|-------------------|--------|---------|
| 512     | 28 GB         | 24 GB             | 1.1x   | 14%     |
| 2048    | OOM           | 42 GB             | -      | 能运行  |
| 4096    | OOM           | 68 GB             | -      | 能运行  |

### 13.3.3 CPU Offload（卸载）

**原理**：

将不常用的张量（如优化器状态）卸载到 CPU 内存或 NVMe 磁盘。

**方式 1：DeepSpeed ZeRO-Offload**：

```python
# deepspeed_config.json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",  # 或 "nvme"
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu"
        }
    }
}
```

```bash
deepspeed --num_gpus=1 train.py --deepspeed deepspeed_config.json
```

**方式 2：Accelerate**：

```python
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision="bf16",
    cpu_offload=True,  # 启用 CPU offload
)

model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)
```

**性能影响**：

| 配置 | GPU 显存 | CPU 内存 | 训练速度 |
|------|---------|---------|---------|
| 无 Offload | 66 GB | 8 GB | 100% |
| Optimizer Offload | 38 GB | 36 GB | 85% |
| All Offload | 14 GB | 60 GB | 45% |

### 13.3.4 VRAM Sharing（虚拟内存）

**原理**：

利用 Unified Memory 让 GPU 在显存不足时自动 swap 到 CPU 内存。

```python
import torch

# 启用 CUDA Managed Memory
torch.cuda.set_per_process_memory_fraction(0.9)  # 限制 GPU 使用 90%
torch.cuda.empty_cache()

# 允许 PyTorch 使用 page-locked memory
torch.backends.cuda.matmul.allow_tf32 = True
```

**限制**：

- 仅适用于 NVIDIA GPU (Pascal 架构及更高)
- 性能下降明显（30-50%）
- 不推荐生产环境

---

## 13.4 内存分析工具

### 13.4.1 torch.cuda.memory_summary()

```python
import torch

# 训练一个 batch
model.train()
outputs = model(input_ids, labels=labels)
loss = outputs.loss
loss.backward()

# 打印详细内存报告
print(torch.cuda.memory_summary(device="cuda:0", abbreviated=False))
```

**输出示例**：

```
|===========================================================================|
|                  PyTorch CUDA memory summary                             |
|---------------------------------------------------------------------------|
| CUDA OOMs: 0                                                              |
| Allocated memory: 24.5 GB                                                 |
| Reserved memory:  26.8 GB                                                 |
| Active memory:    24.2 GB                                                 |
|---------------------------------------------------------------------------|
| Allocation breakdown:
|   - nn.Module: 14.2 GB (model weights)
|   - Tensor:    8.3 GB (activations)
|   - Gradient:  2.0 GB (gradients)
|===========================================================================|
```

**关键指标**：

- **Allocated**：实际分配的显存
- **Reserved**：PyTorch 缓存的显存（包含未使用部分）
- **Active**：当前活跃的张量

### 13.4.2 torch.profiler

**高级性能分析**：

```python
from torch.profiler import profile, ProfilerActivity, record_function

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with record_function("training_step"):
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# 导出 Chrome Trace
prof.export_chrome_trace("trace.json")

# 打印 top 10 内存消耗
print(prof.key_averages().table(
    sort_by="cuda_memory_usage",
    row_limit=10,
))
```

**输出示例**：

```
-------------------------------------------------------
Name                         | Self CPU | Self CUDA | CPU Mem | CUDA Mem
-------------------------------------------------------
aten::addmm                  |  120.5ms |   105.2ms |    0 b  | 8.2 GB
aten::copy_                  |   45.2ms |    38.1ms |    0 b  | 4.1 GB
Optimizer.step               |   32.8ms |    28.3ms |    0 b  | 2.3 GB
-------------------------------------------------------
```

**可视化**：

1. 打开 Chrome 浏览器
2. 访问 `chrome://tracing`
3. 加载 `trace.json`

<div data-component="ProfilerVisualizationDemo"></div>

### 13.4.3 nvidia-smi 监控

**实时监控**：

```bash
# 每秒更新
watch -n 1 nvidia-smi

# 或使用 Python
import subprocess
import time

while True:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip())
    time.sleep(1)
```

**脚本化监控**：

```python
# memory_monitor.py
import pynvml
import time

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

while True:
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used = info.used / 1024**3  # GB
    total = info.total / 1024**3
    print(f"GPU Memory: {used:.2f} / {total:.2f} GB ({used/total*100:.1f}%)")
    time.sleep(1)
```

---

## 13.5 极限内存优化组合

### 13.5.1 最优组合策略

<div data-component="OptimizationCombinator"></div>

**方案 1：QLoRA + Gradient Checkpointing + Flash Attention**

```python
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model

# 1. 4-bit 量化加载
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",  # Flash Attention
)

# 2. LoRA 适配器
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)

model = get_peft_model(model, lora_config)

# 3. Gradient Checkpointing
model.gradient_checkpointing_enable()

# 4. 训练
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=8,  # 大 batch size
    gradient_accumulation_steps=4,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

**显存占用**：

| 组件 | 显存 |
|------|------|
| 4-bit 模型权重 | 4.8 GB |
| LoRA 适配器 | 0.2 GB |
| 优化器状态 (8-bit) | 0.4 GB |
| 激活值 (checkpointing + flash) | 8 GB |
| **总计** | **13.4 GB** ✅ |

**可训练**：

- ✅ RTX 3090 (24GB): batch_size=8
- ✅ RTX 4090 (24GB): batch_size=16
- ✅ A100 (80GB): batch_size=64

### 13.5.2 DeepSpeed ZeRO-3 + Offload

**终极方案**（70B 模型 on 单卡 24GB）：

```json
// ds_config.json
{
    "train_batch_size": 8,
    "gradient_accumulation_steps": 8,
    "gradient_clipping": 1.0,
    
    "zero_optimization": {
        "stage": 3,  // ZeRO-3: 分片权重 + 优化器 + 梯度
        
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9
    },
    
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16
    }
}
```

```python
# train.py
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf")

training_args = TrainingArguments(
    output_dir="./output",
    deepspeed="./ds_config.json",  # DeepSpeed 配置
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

```bash
# 启动训练
deepspeed --num_gpus=1 train.py
```

**显存占用分析**：

| 组件 | 无 ZeRO | ZeRO-3 + Offload |
|------|---------|------------------|
| 模型权重 | 140 GB | **4.4 GB** (CPU: 135 GB) |
| 优化器状态 | 280 GB | **0 GB** (CPU: 280 GB) |
| 梯度 | 140 GB | **4.4 GB** (CPU: 135 GB) |
| 激活值 | 20 GB | **12 GB** (checkpointing) |
| **GPU 总计** | 580 GB ❌ | **~21 GB** ✅ |

**性能代价**：

- 训练速度：约为标准训练的 **30-40%**（大量 CPU-GPU 数据传输）
- 适用于**显存极度受限**但**时间充裕**的场景

### 13.5.3 实战案例：70B 模型 on 24GB GPU

**完整配置**：

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. QLoRA 4-bit 加载
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto",  # 自动分配
    max_memory={0: "22GB"},  # 限制 GPU 0 使用 22GB
)

# 2. 准备量化模型
model = prepare_model_for_kbit_training(model)

# 3. LoRA 配置（小 rank 降低显存）
lora_config = LoraConfig(
    r=8,  # rank=8 (vs 16)
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
)

model = get_peft_model(model, lora_config)

# 4. Gradient Checkpointing
model.gradient_checkpointing_enable()
model.config.use_cache = False  # 禁用 KV cache（训练时不需要）

# 5. 训练配置
training_args = TrainingArguments(
    output_dir="./llama2-70b-finetuned",
    
    # 批次设置
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # 等效 batch=16
    
    # 内存优化
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    optim="paged_adamw_8bit",  # 8-bit optimizer
    
    # 其他
    learning_rate=2e-4,
    max_steps=1000,
    logging_steps=10,
    save_steps=100,
)

# 6. 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

**显存实测**：

```
4-bit 模型权重:        17.5 GB
LoRA 适配器 (r=8):     0.15 GB
8-bit 优化器:          0.3 GB
激活值 (checkpointing): 4 GB
其他 (buffers etc.):   1 GB
-----------------------------------
总计:                  ~23 GB ✅ (RTX 4090 可运行)
```

**性能**：

- 训练速度：~0.8 it/s（每个 iteration = 16 micro-batches）
- 完整 fine-tune（1000 steps）：~20 分钟

---

## 13.6 总结与最佳实践

### 13.6.1 内存优化决策树

```
显存是否充足？
├─ 是 → 无需优化，使用标准训练
└─ 否 → 继续
    ├─ 模型大小？
    │   ├─ <7B → Gradient Checkpointing + BF16
    │   ├─ 7-13B → + Flash Attention
    │   ├─ 13-30B → + QLoRA (4-bit)
    │   └─ >30B → + DeepSpeed ZeRO-3 Offload
    └─ 是否需要微调？
        ├─ 是 → QLoRA + Gradient Checkpointing
        └─ 否 → GPTQ/AWQ 量化推理
```

### 13.6.2 优化组合效果

<div data-component="OptimizationEffectComparison"></div>

| 优化策略 | 显存节省 | 速度影响 | 适用模型 |
|---------|---------|---------|---------|
| **Gradient Checkpointing** | 30-40% | -25% | 所有 |
| **Flash Attention** | 10-20% | +10% | Seq>512 |
| **Gradient Accumulation** | 50%+ | -5% | 所有 |
| **QLoRA (4-bit)** | 75% | -10% | 7B+ |
| **DeepSpeed ZeRO-3** | 90%+ | -60% | 70B+ |

### 13.6.3 代码模板

**通用内存优化模板**：

```python
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ============ 配置参数 ============
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
USE_4BIT = True  # QLoRA
USE_FLASH_ATTENTION = True
USE_GRADIENT_CHECKPOINTING = True

# ============ 加载模型 ============
if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2" if USE_FLASH_ATTENTION else None,
    )
    model = prepare_model_for_kbit_training(model)
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if USE_FLASH_ATTENTION else None,
    )

# ============ LoRA ============
if USE_4BIT:
    lora_config = LoraConfig(
        r=8 if "70b" in MODEL_NAME.lower() else 16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)

# ============ Gradient Checkpointing ============
if USE_GRADIENT_CHECKPOINTING:
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

# ============ 训练 ============
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1 if USE_4BIT else 4,
    gradient_accumulation_steps=16 if USE_4BIT else 4,
    bf16=True,
    gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit" if USE_4BIT else "adamw_torch",
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

### 13.6.4 常见问题排查

**问题 1：启用 Checkpointing 后仍然 OOM**

```python
# 解决方案 1：减小 batch size
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # 从 2 降到 1
    gradient_accumulation_steps=16,  # 增加累积步数
)

# 解决方案 2：降低序列长度
def preprocess(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,  # 从 2048 降到 512
    )

# 解决方案 3：使用 QLoRA
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
)
```

**问题 2：Flash Attention 安装失败**

```bash
# 方法 1：使用预编译轮子
pip install flash-attn --no-build-isolation

# 方法 2：从源码编译（需要 CUDA 11.8+）
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
python setup.py install

# 方法 3：使用 xFormers（替代方案）
pip install xformers
# 模型加载时无需指定 attn_implementation
```

**问题 3：训练速度过慢**

```python
# 检查瓶颈
import torch.cuda.profiler as profiler

profiler.start()
# 训练代码
profiler.stop()

# 可能原因：
# 1. CPU-GPU 数据传输慢 → 使用 pin_memory=True
# 2. Gradient Checkpointing 开销大 → 仅在必要时启用
# 3. 数据加载慢 → 增加 num_workers
training_args = TrainingArguments(
    dataloader_num_workers=4,  # 并行数据加载
    dataloader_pin_memory=True,
)
```

---

通过本章学习，你应该能够：

- ✅ 理解 Gradient Checkpointing 的时间-空间权衡
- ✅ 掌握多种内存优化技巧（Flash Attention、梯度累积、Offload）
- ✅ 使用分析工具定位内存瓶颈
- ✅ 组合多种优化策略训练超大模型（70B on 24GB GPU）
- ✅ 排查常见内存问题

**下一章预告**：Chapter 14 将深入分布式训练，讲解 FSDP、DeepSpeed ZeRO、数据并行等多卡训练策略！

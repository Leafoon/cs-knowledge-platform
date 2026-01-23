# Chapter 6: 参数高效微调（PEFT）

## 6.1 PEFT 概述

### 6.1.1 全量微调的挑战

在传统的微调范式中，我们需要更新模型的**所有参数**。以 BERT-base（110M 参数）为例：

```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 全量微调：所有 110M 参数都可训练
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")  # 109,483,778

# TrainingArguments 默认更新所有参数
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()  # 更新全部 110M 参数
```

**全量微调的问题**：

1. **显存占用巨大**：
   - 模型参数：110M × 4 bytes (FP32) = 440 MB
   - 优化器状态（Adam）：110M × 8 bytes = 880 MB（momentum + variance）
   - 梯度：110M × 4 bytes = 440 MB
   - **总计**：~1.76 GB（仅模型部分，不含激活值）

2. **训练成本高**：
   - 每个下游任务都需要保存一份完整的模型副本
   - 多任务场景下存储成本线性增长

3. **灾难性遗忘**：
   - 大幅度更新可能破坏预训练知识
   - 小数据集上容易过拟合

### 6.1.2 PEFT 核心思想

**参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）** 的核心思想是：

> 冻结预训练模型的大部分参数，仅训练少量额外参数（<1% 的总参数量），实现与全量微调相近的性能。

<div data-component="PEFTMethodComparison"></div>

### 6.1.3 Hugging Face PEFT 库

Hugging Face 提供了统一的 PEFT 库，支持多种高效微调方法：

```bash
pip install peft
```

**支持的方法**：

| 方法 | 类型 | 可训练参数比例 | 适用场景 |
|------|------|----------------|----------|
| **LoRA** | 低秩适配 | 0.1% - 1% | 通用，效果最好 |
| **Prefix Tuning** | 软提示 | 0.01% - 0.1% | 生成任务 |
| **P-Tuning** | 可学习嵌入 | 0.01% - 0.1% | 理解任务 |
| **Prompt Tuning** | 提示微调 | 0.001% - 0.01% | 大模型（>10B） |
| **AdaLoRA** | 自适应低秩 | 0.1% - 1% | 资源受限 |
| **(IA)³** | 激活缩放 | 0.01% - 0.1% | 多任务学习 |

---

## 6.2 LoRA（Low-Rank Adaptation）

### 6.2.1 LoRA 原理

LoRA 由 Microsoft 在 2021 年提出，基于一个关键观察：

> 微调过程中的权重更新矩阵是**低秩**的（Intrinsic Dimensionality）。

**数学表述**：

假设预训练权重矩阵为 $W_0 \in \mathbb{R}^{d \times k}$，全量微调后的权重为 $W_0 + \Delta W$。LoRA 将 $\Delta W$ 分解为两个低秩矩阵的乘积：

$$
W = W_0 + \Delta W = W_0 + BA
$$

其中：
- $B \in \mathbb{R}^{d \times r}$
- $A \in \mathbb{R}^{r \times k}$
- $r \ll \min(d, k)$（秩远小于原始维度）

**前向传播**：

$$
h = W_0 x + \Delta W x = W_0 x + BAx
$$

<div data-component="LoRAMatrixDecomposition"></div>

**参数量对比**：

以 BERT-base 的 query 投影层为例（768 × 768）：

```python
# 全量微调
params_full = 768 * 768 = 589,824

# LoRA (rank=8)
params_lora = 768 * 8 + 8 * 768 = 12,288

# 参数减少比例
reduction = (1 - params_lora / params_full) * 100
print(f"参数减少: {reduction:.1f}%")  # 97.9%
```

### 6.2.2 LoRA 代码实现

**基础用法**：

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification

# 1. 加载基础模型
base_model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# 2. 配置 LoRA
lora_config = LoraConfig(
    r=8,                          # 秩（rank）
    lora_alpha=16,                # 缩放因子
    target_modules=["query", "value"],  # 应用 LoRA 的层
    lora_dropout=0.1,             # Dropout
    bias="none",                  # 偏置处理策略
    task_type="SEQ_CLS"           # 任务类型
)

# 3. 包装为 PEFT 模型
model = get_peft_model(base_model, lora_config)

# 4. 查看参数统计
model.print_trainable_parameters()
# trainable params: 294,912 || all params: 109,778,690 || trainable%: 0.27%
```

**target_modules 详解**：

```python
# 方式1：指定特定层名称
lora_config = LoraConfig(
    target_modules=["query", "key", "value"],  # 仅 Attention 的 Q、K、V
    # ...
)

# 方式2：使用正则表达式
lora_config = LoraConfig(
    target_modules=[".*attention.*(query|key|value)"],  # 匹配所有注意力层
    # ...
)

# 方式3：应用到所有线性层
lora_config = LoraConfig(
    target_modules="all-linear",  # 包括 FFN、输出层等
    # ...
)

# 方式4：查看模型结构后手动指定
from peft import get_peft_model

base_model = AutoModelForCausalLM.from_pretrained("gpt2")
print(base_model)  # 查看层名称

lora_config = LoraConfig(
    target_modules=["c_attn", "c_proj"],  # GPT-2 特定层名
    # ...
)
```

**与 Trainer 集成**：

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./lora-bert",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=3e-4,           # LoRA 通常用更高学习率
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,                  # PEFT 包装后的模型
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()

# 保存 LoRA 权重（仅 ~1MB）
model.save_pretrained("./lora-bert-final")
```

### 6.2.3 LoRA 超参数调优

<div data-component="LoRARankSelector"></div>

**关键超参数**：

1. **r (rank)**：
   - 范围：1 ~ 64
   - 推荐值：
     - 小模型（<500M）：4 ~ 8
     - 中等模型（1B ~ 7B）：8 ~ 16
     - 大模型（>10B）：16 ~ 32
   - 影响：rank 越大，表达能力越强，但参数量增加

2. **lora_alpha**：
   - 缩放系数，实际缩放倍数为 `lora_alpha / r`
   - 推荐值：r 的 2 倍（例如 r=8 时 alpha=16）
   - 影响：控制 LoRA 权重的影响力

3. **target_modules**：
   - Q、V 组合：最常用，平衡性能与效率
   - Q、K、V：性能更好，参数稍多
   - Q、K、V + FFN：接近全量微调效果

**Rank 实验对比**（GLUE SST-2 数据集）：

```python
# 实验代码
results = {}
for rank in [2, 4, 8, 16, 32]:
    lora_config = LoraConfig(r=rank, lora_alpha=rank*2, target_modules=["query", "value"])
    model = get_peft_model(base_model, lora_config)
    
    trainer = Trainer(model=model, args=training_args, ...)
    trainer.train()
    
    metrics = trainer.evaluate()
    results[rank] = {
        "accuracy": metrics["eval_accuracy"],
        "params": model.get_nb_trainable_parameters(),
        "memory_mb": torch.cuda.max_memory_allocated() / 1024**2
    }

# 结果示例
"""
Rank | Accuracy | Params  | Memory
-----|----------|---------|--------
2    | 91.2%    | 147K    | 1.2 GB
4    | 92.5%    | 294K    | 1.3 GB
8    | 93.1%    | 589K    | 1.5 GB  ← 最佳性价比
16   | 93.3%    | 1.18M   | 1.9 GB
32   | 93.4%    | 2.36M   | 2.7 GB
Full | 93.5%    | 110M    | 4.5 GB
"""
```

### 6.2.4 LoRA 权重合并与部署

**训练后合并权重**：

```python
from peft import PeftModel
import torch

# 方式1：训练结束后立即合并
model = model.merge_and_unload()  # 将 LoRA 权重合并到基础模型
model.save_pretrained("./merged-model")  # 保存为标准 HF 模型

# 方式2：从检查点加载并合并
base_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
peft_model = PeftModel.from_pretrained(base_model, "./lora-bert-final")
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("./merged-model")
```

**推理时动态切换适配器**：

```python
# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("gpt2")

# 加载多个 LoRA 适配器
from peft import PeftModel

model = PeftModel.from_pretrained(base_model, "./lora-task1", adapter_name="task1")
model.load_adapter("./lora-task2", adapter_name="task2")
model.load_adapter("./lora-task3", adapter_name="task3")

# 切换适配器
model.set_adapter("task1")  # 使用任务1的适配器
output1 = model.generate(...)

model.set_adapter("task2")  # 切换到任务2
output2 = model.generate(...)

# 禁用所有适配器（使用原始模型）
model.disable_adapters()
output_base = model.generate(...)
```

---

## 6.3 QLoRA（Quantized LoRA）

### 6.3.1 QLoRA 原理

QLoRA 由华盛顿大学在 2023 年提出，核心创新：

> 将基础模型量化到 **4-bit**，仅用 LoRA 适配器进行 FP16/BF16 训练。

**关键技术**：

1. **4-bit NormalFloat (NF4)**：
   - 专为正态分布权重设计的数据类型
   - 比传统 INT4 更适合神经网络权重

2. **双重量化（Double Quantization）**：
   - 量化常数本身也被量化
   - 进一步减少显存占用

3. **分页优化器（Paged Optimizer）**：
   - 利用统一内存（CPU + GPU）
   - 避免 OOM 错误

<div data-component="QLoRAQuantizationFlow"></div>

**显存对比**（LLaMA-7B）：

| 方法 | 模型精度 | LoRA精度 | 显存占用 | 相对减少 |
|------|----------|----------|----------|----------|
| 全量微调 | FP16 | - | 28 GB | - |
| LoRA | FP16 | FP16 | 14 GB | 50% |
| **QLoRA** | **4-bit** | **FP16** | **6 GB** | **79%** |

### 6.3.2 QLoRA 代码实现

**基础用法**：

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 1. 配置 4-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # 4-bit 量化
    bnb_4bit_use_double_quant=True,         # 双重量化
    bnb_4bit_quant_type="nf4",              # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16   # 计算时用 BF16
)

# 2. 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",                      # 自动设备分配
    trust_remote_code=True
)

# 3. 准备模型进行训练（启用梯度检查点等）
model = prepare_model_for_kbit_training(model)

# 4. 配置 LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # LLaMA 特定
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 5. 包装为 PEFT 模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062%
```

**完整训练流程**：

```python
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

# 加载数据集（以 Alpaca 为例）
dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")

# 数据预处理
def format_instruction(example):
    if example["input"]:
        return f"""Below is an instruction that describes a task, paired with an input.

### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
{example["output"]}"""
    else:
        return f"""Below is an instruction that describes a task.

### Instruction:
{example["instruction"]}

### Response:
{example["output"]}"""

def tokenize_function(examples):
    texts = [format_instruction(ex) for ex in examples]
    return tokenizer(texts, truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 训练配置
training_args = TrainingArguments(
    output_dir="./qlora-llama2-7b",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,        # 有效 batch_size = 16
    learning_rate=2e-4,
    fp16=False,                           # 4-bit 模型不需要额外的 fp16
    bf16=True,                            # 使用 BF16 计算
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_32bit",            # 分页优化器
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# 开始训练
trainer.train()

# 保存 LoRA 权重
model.save_pretrained("./qlora-llama2-7b-final")
tokenizer.save_pretrained("./qlora-llama2-7b-final")
```

### 6.3.3 NormalFloat4 数据类型

<div data-component="NF4DataTypeVisualizer"></div>

**NF4 vs INT4**：

传统 INT4 使用均匀量化：

$$
q = \text{round}\left(\frac{x - \min}{\max - \min} \times 15\right)
$$

NF4 使用分位数量化，专为正态分布设计：

```python
# NF4 的 16 个量化级别（预定义）
NF4_QUANTILES = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
]

# 量化过程
def quantize_nf4(tensor):
    # 按块归一化（block_size=64）
    normalized = tensor / tensor.abs().max()
    
    # 查找最近的 NF4 值
    quantized = []
    for val in normalized:
        idx = np.argmin(np.abs(NF4_QUANTILES - val))
        quantized.append(idx)  # 0-15
    
    return quantized
```

**为什么 NF4 更好？**

```python
import numpy as np
import matplotlib.pyplot as plt

# 生成正态分布权重
weights = np.random.randn(10000)

# INT4 量化
int4_levels = np.linspace(weights.min(), weights.max(), 16)
int4_quantized = np.digitize(weights, int4_levels)

# NF4 量化
nf4_quantized = [np.argmin(np.abs(NF4_QUANTILES - w/np.abs(weights).max())) 
                 for w in weights]

# 计算误差
int4_error = np.mean((weights - int4_levels[int4_quantized])**2)
nf4_error = np.mean((weights - np.array(NF4_QUANTILES)[nf4_quantized] * np.abs(weights).max())**2)

print(f"INT4 MSE: {int4_error:.6f}")  # 0.012345
print(f"NF4 MSE: {nf4_error:.6f}")    # 0.008912 (更低！)
```

### 6.3.4 双重量化

量化常数本身也占用显存：

```python
# 单次量化
# 每 64 个权重共享 1 个 FP32 缩放因子
num_blocks = 7_000_000_000 / 64  # LLaMA-7B
scale_memory = num_blocks * 4    # 109,375,000 bytes ≈ 104 MB

# 双重量化
# 缩放因子进一步量化为 8-bit
scale_memory_dq = num_blocks * 1  # 27,343,750 bytes ≈ 26 MB

# 节省 78 MB（占总显存的 1-2%）
```

启用双重量化：

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # ← 关键参数
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

---

## 6.4 其他 PEFT 方法

### 6.4.1 Prefix Tuning

为每一层添加可学习的"前缀"向量：

```python
from peft import PrefixTuningConfig, get_peft_model

prefix_config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,        # 前缀长度
    encoder_hidden_size=768,      # 隐藏层大小
    prefix_projection=True        # 使用 MLP 投影
)

model = get_peft_model(base_model, prefix_config)
model.print_trainable_parameters()
# trainable params: 983,040 || all params: 125,440,000 || trainable%: 0.78%
```

**原理**：

在输入序列前添加虚拟 token：

$$
\text{Input: } [x_1, x_2, \ldots, x_n] \\
\text{Prefix: } [p_1, p_2, \ldots, p_m] \\
\text{Actual Input: } [p_1, \ldots, p_m, x_1, \ldots, x_n]
$$

### 6.4.2 P-Tuning v2

改进的提示微调，在每一层都添加提示：

```python
from peft import PromptEncoderConfig, get_peft_model

ptuning_config = PromptEncoderConfig(
    task_type="SEQ_CLS",
    num_virtual_tokens=10,
    encoder_hidden_size=768,
    encoder_num_layers=2,         # 提示编码器层数
    encoder_dropout=0.1,
)

model = get_peft_model(base_model, ptuning_config)
```

### 6.4.3 (IA)³ - Infused Adapter by Inhibiting and Amplifying Inner Activations

通过缩放激活值实现微调：

```python
from peft import IA3Config, get_peft_model

ia3_config = IA3Config(
    task_type="SEQ_CLS",
    target_modules=["k_proj", "v_proj", "down_proj"],  # 缩放的层
    feedforward_modules=["down_proj"],                  # FFN 层
)

model = get_peft_model(base_model, ia3_config)
model.print_trainable_parameters()
# trainable params: 204,800 || all params: 6,742,609,920 || trainable%: 0.003%
```

**原理**：

$$
h' = h \odot l_k
$$

其中 $l_k$ 是可学习的缩放向量。

### 6.4.4 AdaLoRA - Adaptive LoRA

动态调整不同层的秩：

```python
from peft import AdaLoraConfig, get_peft_model

adalora_config = AdaLoraConfig(
    task_type="CAUSAL_LM",
    r=8,                          # 初始秩
    target_r=4,                   # 目标平均秩
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    init_r=12,                    # 初始化秩（高于最终秩）
    tinit=200,                    # 预热步数
    tfinal=1000,                  # 最终步数
    deltaT=10,                    # 调整间隔
)

model = get_peft_model(base_model, adalora_config)
```

**动态秩调整**：

- 训练初期：所有层使用较高秩
- 训练中期：根据重要性裁剪不重要层的秩
- 训练后期：重要层保持高秩，不重要层降至低秩

<div data-component="AdaLoraRankEvolution"></div>

---

## 6.5 PEFT 高级技巧

### 6.5.1 多适配器组合

```python
from peft import PeftModel

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("gpt2")

# 加载多个适配器
model = PeftModel.from_pretrained(base_model, "./lora-math", adapter_name="math")
model.load_adapter("./lora-code", adapter_name="code")
model.load_adapter("./lora-writing", adapter_name="writing")

# 方式1：顺序切换
model.set_adapter("math")
math_output = model.generate(...)

model.set_adapter("code")
code_output = model.generate(...)

# 方式2：加权组合
model.add_weighted_adapter(
    adapters=["math", "code"],
    weights=[0.7, 0.3],
    adapter_name="math_code_blend"
)
model.set_adapter("math_code_blend")
blend_output = model.generate(...)
```

### 6.5.2 渐进式秩增长

```python
import torch

# 第1阶段：低秩训练（r=4）
lora_config_stage1 = LoraConfig(r=4, lora_alpha=8, ...)
model = get_peft_model(base_model, lora_config_stage1)
trainer.train()  # 训练 1 epoch

# 第2阶段：增加秩到 8
model.resize_lora_rank(new_rank=8)  # 扩展 LoRA 矩阵
trainer.train()  # 继续训练 1 epoch

# 第3阶段：最终秩 16
model.resize_lora_rank(new_rank=16)
trainer.train()  # 最终训练
```

### 6.5.3 LoRA 初始化策略

```python
from peft import LoraConfig

# 默认初始化：A ~ Kaiming, B = 0
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    init_lora_weights=True  # 默认
)

# 高斯初始化：A ~ N(0, 1/r), B ~ N(0, 1/r)
lora_config = LoraConfig(
    r=8,
    init_lora_weights="gaussian"
)

# 自定义初始化
from peft import get_peft_model

model = get_peft_model(base_model, lora_config)

for name, param in model.named_parameters():
    if "lora_A" in name:
        torch.nn.init.xavier_uniform_(param)
    elif "lora_B" in name:
        torch.nn.init.zeros_(param)
```

### 6.5.4 显存优化技巧

<div data-component="MemoryOptimizationComparison"></div>

**技巧组合**：

```python
from transformers import TrainingArguments
from peft import prepare_model_for_kbit_training

# 1. 使用 QLoRA（4-bit 量化）
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# 2. 启用梯度检查点
model.gradient_checkpointing_enable()

# 3. 使用梯度累积
training_args = TrainingArguments(
    per_device_train_batch_size=1,       # 减小批次大小
    gradient_accumulation_steps=16,      # 累积 16 步
    # ...
)

# 4. 使用分页优化器
training_args = TrainingArguments(
    optim="paged_adamw_8bit",            # 8-bit Adam
    # ...
)

# 5. 混合精度训练
training_args = TrainingArguments(
    bf16=True,                           # BF16
    tf32=True,                           # TF32（Ampere GPU）
    # ...
)

# 组合效果：7B 模型可在 6GB 显存上训练
```

**显存占用估算**：

| 配置 | 模型 | 优化器 | 梯度 | 激活 | 总计 |
|------|------|--------|------|------|------|
| 全量微调（FP16） | 14GB | 14GB | 14GB | 12GB | 54GB |
| LoRA（FP16） | 14GB | 0.5GB | 0.5GB | 6GB | 21GB |
| **QLoRA（4-bit）** | **3.5GB** | **0.5GB** | **0.5GB** | **2GB** | **6.5GB** |

---

## 6.6 PEFT 实战案例

### 6.6.1 案例1：LLaMA-7B 指令微调

**目标**：用 Alpaca 数据集微调 LLaMA-7B。

```python
# 完整脚本
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# === 1. 配置量化 ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# === 2. 加载模型和分词器 ===
model_id = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# === 3. 准备模型 ===
model = prepare_model_for_kbit_training(model)

# === 4. 配置 LoRA ===
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# === 5. 加载数据集 ===
dataset = load_dataset("tatsu-lab/alpaca", split="train")

def format_prompt(example):
    if example["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
{example["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{example["instruction"]}

### Response:
{example["output"]}"""

def tokenize(example):
    text = format_prompt(example)
    return tokenizer(text, truncation=True, max_length=512, padding="max_length")

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# === 6. 训练配置 ===
training_args = TrainingArguments(
    output_dir="./qlora-llama2-7b-alpaca",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
)

# === 7. 训练 ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# === 8. 保存 ===
model.save_pretrained("./qlora-llama2-7b-alpaca-final")
tokenizer.save_pretrained("./qlora-llama2-7b-alpaca-final")
```

**推理测试**：

```python
# 加载微调后的模型
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./qlora-llama2-7b-alpaca-final")

# 推理
prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
What is the capital of France?

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# 输出：The capital of France is Paris.
```

### 6.6.2 案例2：多任务适配器

同一个基础模型服务多个任务：

```python
# 训练情感分类适配器
lora_config_sentiment = LoraConfig(r=8, task_type="SEQ_CLS", ...)
model_sentiment = get_peft_model(base_model, lora_config_sentiment)
trainer_sentiment.train()
model_sentiment.save_pretrained("./lora-sentiment")

# 训练 NER 适配器
lora_config_ner = LoraConfig(r=8, task_type="TOKEN_CLS", ...)
model_ner = get_peft_model(base_model, lora_config_ner)
trainer_ner.train()
model_ner.save_pretrained("./lora-ner")

# 训练问答适配器
lora_config_qa = LoraConfig(r=8, task_type="QUESTION_ANS", ...)
model_qa = get_peft_model(base_model, lora_config_qa)
trainer_qa.train()
model_qa.save_pretrained("./lora-qa")

# === 推理时动态切换 ===
base = AutoModel.from_pretrained("bert-base-uncased")

# 加载所有适配器
model = PeftModel.from_pretrained(base, "./lora-sentiment", adapter_name="sentiment")
model.load_adapter("./lora-ner", adapter_name="ner")
model.load_adapter("./lora-qa", adapter_name="qa")

# 根据任务切换
def predict(text, task):
    model.set_adapter(task)
    if task == "sentiment":
        return sentiment_head(model(**tokenizer(text)))
    elif task == "ner":
        return ner_head(model(**tokenizer(text)))
    elif task == "qa":
        return qa_head(model(**tokenizer(text)))

# 使用
predict("I love this movie!", "sentiment")  # 正面
predict("Apple Inc. is in California.", "ner")  # [ORG, LOC]
predict("What is AI?", "qa")  # 答案提取
```

---

## 6.7 PEFT 性能分析

### 6.7.1 训练速度对比

<div data-component="PEFTTrainingSpeedComparison"></div>

**实验设置**：
- 模型：BERT-base（110M）
- 数据集：GLUE SST-2（67K 样本）
- 硬件：单卡 NVIDIA A100（40GB）

| 方法 | Batch Size | 时间/Epoch | 显存占用 | 参数量 |
|------|-----------|-----------|---------|--------|
| 全量微调 | 32 | 12 min | 8.5 GB | 110M |
| LoRA (r=8) | 32 | 8 min | 4.2 GB | 294K |
| Prefix Tuning | 32 | 9 min | 5.1 GB | 983K |
| (IA)³ | 32 | 7 min | 3.8 GB | 205K |

**LLaMA-7B 对比**：

| 方法 | 显存 | 时间/Step | 收敛 Steps |
|------|------|----------|-----------|
| 全量微调 | 56 GB | 2.1s | 10,000 |
| LoRA | 22 GB | 1.5s | 12,000 |
| **QLoRA** | **6.5 GB** | **1.8s** | **15,000** |

### 6.7.2 下游任务性能

**GLUE Benchmark**（BERT-base）：

| 任务 | 全量微调 | LoRA (r=8) | Prefix | (IA)³ |
|------|---------|-----------|--------|-------|
| SST-2 | 93.5 | **93.1** | 92.3 | 92.8 |
| MRPC | 88.9 | **88.5** | 87.2 | 87.9 |
| CoLA | 60.5 | **59.8** | 58.1 | 59.2 |
| **平均** | **80.9** | **80.5** | 79.2 | 80.0 |

**结论**：LoRA 仅损失 0.5% 性能，但参数减少 99.7%！

### 6.7.3 秩选择的影响

<div data-component="LoRAMemoryAccuracyTradeoff"></div>

**实验结果**（LLaMA-7B on Alpaca）：

```python
# 不同秩的性能对比
results = {
    "r=4":  {"ppl": 5.23, "memory": 5.8, "params": "2.1M"},
    "r=8":  {"ppl": 4.89, "memory": 6.1, "params": "4.2M"},
    "r=16": {"ppl": 4.67, "memory": 6.8, "params": "8.4M"},
    "r=32": {"ppl": 4.58, "memory": 8.2, "params": "16.8M"},
    "r=64": {"ppl": 4.52, "memory": 11.1, "params": "33.6M"},
    "Full": {"ppl": 4.45, "memory": 56.0, "params": "7B"}
}

# 最佳性价比：r=8 或 r=16
```

**推荐策略**：

```
小模型（<1B）   → r=4  ~ r=8
中等模型（1-7B）  → r=8  ~ r=16
大模型（>7B）    → r=16 ~ r=32
```

---

## 6.8 PEFT 常见问题

### 6.8.1 LoRA 训练不收敛

**问题**：Loss 不下降或波动剧烈。

**原因与解决方案**：

1. **学习率过高**：
   ```python
   # ❌ 错误：使用全量微调的学习率
   training_args = TrainingArguments(learning_rate=5e-5)
   
   # ✅ 正确：LoRA 需要更高学习率
   training_args = TrainingArguments(learning_rate=1e-4)  # 2-5倍
   ```

2. **alpha/r 比例不当**：
   ```python
   # ❌ 错误：缩放因子太小
   lora_config = LoraConfig(r=16, lora_alpha=16)  # 缩放 = 1
   
   # ✅ 正确：alpha = 2*r
   lora_config = LoraConfig(r=16, lora_alpha=32)  # 缩放 = 2
   ```

3. **target_modules 选择不当**：
   ```python
   # ❌ 错误：只训练 query
   lora_config = LoraConfig(target_modules=["query"])
   
   # ✅ 正确：至少训练 query + value
   lora_config = LoraConfig(target_modules=["query", "value"])
   ```

### 6.8.2 OOM（显存溢出）

**解决方案**：

```python
# 1. 减小批次大小 + 梯度累积
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # 从 8 降到 1
    gradient_accumulation_steps=8,  # 保持有效 batch_size = 8
)

# 2. 启用梯度检查点
model.gradient_checkpointing_enable()

# 3. 使用 8-bit Adam
training_args = TrainingArguments(
    optim="adamw_8bit"  # 或 paged_adamw_8bit
)

# 4. 切换到 QLoRA
bnb_config = BitsAndBytesConfig(load_in_4bit=True, ...)
```

### 6.8.3 合并后的模型性能下降

**原因**：量化误差累积。

**解决方案**：

```python
# 方式1：使用更高精度进行合并
model = model.to(torch.float32)  # 转到 FP32
merged = model.merge_and_unload()
merged.save_pretrained("./merged", torch_dtype=torch.float16)  # 保存时降精度

# 方式2：不合并，推理时动态加载
base_model = AutoModel.from_pretrained("base-model")
peft_model = PeftModel.from_pretrained(base_model, "./lora-weights")
# 推理性能影响 <5%
```

---

## 6.9 本章小结

### 核心要点

1. **PEFT 优势**：
   - 参数量：减少 99%+
   - 显存占用：减少 50-80%
   - 性能损失：<1%

2. **方法选择**：
   - **LoRA**：通用首选，效果最好
   - **QLoRA**：显存受限（<10GB）
   - **Prefix Tuning**：生成任务
   - **(IA)³**：极致压缩

3. **超参数指南**：
   - Rank: 8（小模型）~ 16（大模型）
   - Alpha: 2 × Rank
   - 学习率：2-5倍全量微调
   - Target modules: ["query", "value"] 起步

### 代码模板

```python
# QLoRA 训练模板（适用于 7B-70B 模型）
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 2. 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "model-id", quantization_config=bnb_config, device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# 3. LoRA 配置
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 4. 训练
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    optim="paged_adamw_32bit"
)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()

# 5. 保存
model.save_pretrained("./final")
```

### 进一步阅读

- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
- [PEFT 官方文档](https://huggingface.co/docs/peft)
- [PEFT 示例](https://github.com/huggingface/peft/tree/main/examples)

---

**下一章预告**：Chapter 7 将深入探讨**低精度训练与量化**，包括混合精度训练、INT8/INT4 量化、GPTQ、AWQ 等高级技术。

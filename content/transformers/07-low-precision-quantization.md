---
title: "Chapter 7. 低精度训练与量化"
description: "掌握混合精度训练（FP16/BF16）、理解量化技术与梯度累积"
updated: "2026-01-22"
---

## 7.1 混合精度训练概述

### 7.1.1 为什么需要低精度训练？

传统深度学习使用 **FP32（32-bit 浮点数）** 进行训练：

```python
import torch

# FP32 张量
tensor_fp32 = torch.randn(1000, 1000, dtype=torch.float32)
print(f"FP32 size: {tensor_fp32.element_size() * tensor_fp32.nelement() / 1024**2:.2f} MB")
# 输出: FP32 size: 3.81 MB
```

**FP32 的问题**：
- **显存占用大**：每个参数 4 字节
- **计算速度慢**：现代 GPU（Tensor Core）优化了低精度运算
- **带宽浪费**：数据传输成为瓶颈

**混合精度训练**的核心思想：

> 使用**低精度**（FP16/BF16）进行前向和反向传播，用 **FP32** 存储主权重和优化器状态。

<div data-component="PrecisionFormatComparison"></div>

### 7.1.2 精度格式对比

| 格式 | 位数 | 指数位 | 尾数位 | 数值范围 | 精度 | Tensor Core 支持 |
|------|------|--------|--------|----------|------|------------------|
| **FP32** | 32 | 8 | 23 | ±3.4×10³⁸ | ~7 位小数 | ✗ |
| **FP16** | 16 | 5 | 10 | ±6.5×10⁴ | ~3 位小数 | ✓ (V100+) |
| **BF16** | 16 | 8 | 7 | ±3.4×10³⁸ | ~2 位小数 | ✓ (A100+) |
| **TF32** | 19 | 8 | 10 | ±3.4×10³⁸ | ~3 位小数 | ✓ (A100+) |

**FP16 vs BF16**：

```
FP16 (Half Precision):
  1 bit (符号) | 5 bits (指数) | 10 bits (尾数)
  范围小，精度高，容易溢出

BF16 (Brain Float):
  1 bit (符号) | 8 bits (指数) | 7 bits (尾数)
  范围大，精度略低，不易溢出
```

<div data-component="FloatFormatBitLayout"></div>

### 7.1.3 Transformer 中的混合精度训练

**基础用法（PyTorch AMP）**：

```python
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2")

# 方式1：TrainingArguments 自动启用
training_args = TrainingArguments(
    output_dir="./output",
    fp16=True,  # 使用 FP16（V100、T4 等）
    # bf16=True,  # 或使用 BF16（A100、H100 等）
)

trainer = Trainer(model=model, args=training_args, ...)
trainer.train()

# 方式2：手动使用 autocast
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for batch in dataloader:
    with autocast():  # 自动混合精度上下文
        outputs = model(**batch)
        loss = outputs.loss
    
    # 梯度缩放（防止下溢）
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

**关键技术**：

1. **自动混合精度（AMP, Automatic Mixed Precision）**：
   - 前向传播：FP16/BF16
   - 权重更新：FP32
   - 损失缩放：防止梯度下溢

2. **动态损失缩放（Dynamic Loss Scaling）**：
   ```python
   # 为什么需要？
   # FP16 最小正数: 6e-8
   # 梯度可能小于此值 → 变为 0 → 训练失败
   
   # 解决方案：将损失乘以大数（如 65536）
   scaled_loss = loss * 65536
   scaled_loss.backward()  # 梯度也被放大
   
   # 更新时缩小回来
   for param in model.parameters():
       param.grad /= 65536
   ```

---

## 7.2 混合精度训练实战

### 7.2.1 FP16 训练

**完整示例**：

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

# 1. 加载模型和数据
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = load_dataset("glue", "sst2")

def tokenize(examples):
    return tokenizer(examples["sentence"], truncation=True, padding="max_length")

tokenized_dataset = dataset.map(tokenize, batched=True)

# 2. 配置 FP16 训练
training_args = TrainingArguments(
    output_dir="./fp16-bert",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    learning_rate=5e-5,
    fp16=True,  # ← 启用 FP16
    fp16_opt_level="O1",  # Apex 优化级别（可选）
    logging_steps=100,
    save_strategy="epoch",
)

# 3. 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

trainer.train()

# 性能提升：
# - 显存: 8.5 GB → 4.8 GB (↓43%)
# - 速度: 100% → 180% (↑80%)
# - 精度: 93.5% → 93.4% (↓0.1%)
```

**FP16 优化级别（Apex）**：

```python
# O0: FP32 训练（基准）
fp16_opt_level="O0"

# O1: 混合精度（推荐）
fp16_opt_level="O1"  # 自动决定哪些操作用 FP16

# O2: 几乎全 FP16
fp16_opt_level="O2"  # 主权重 FP16，小心溢出

# O3: 纯 FP16
fp16_opt_level="O3"  # 极致性能，易出问题
```

### 7.2.2 BF16 训练

**BF16 的优势**：

```python
import torch

# FP16 溢出示例
fp16_val = torch.tensor(65504.0, dtype=torch.float16)
print(fp16_val * 2)  # inf（溢出！）

# BF16 不会溢出
bf16_val = torch.tensor(65504.0, dtype=torch.bfloat16)
print(bf16_val * 2)  # 131008.0（正常）
```

**使用 BF16**：

```python
# 方式1：TrainingArguments
training_args = TrainingArguments(
    output_dir="./bf16-bert",
    bf16=True,  # ← 使用 BF16
    bf16_full_eval=True,  # 评估时也用 BF16
)

# 方式2：模型转换
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    torch_dtype=torch.bfloat16  # 直接加载为 BF16
).to("cuda")

# 方式3：手动转换
model = model.to(dtype=torch.bfloat16)
```

**硬件要求**：

- **FP16**: V100、T4、RTX 20/30 系列
- **BF16**: A100、H100、RTX 40 系列
- **TF32**: A100+（自动启用，无需配置）

### 7.2.3 TF32 自动加速

**TF32（TensorFloat-32）**：Ampere 架构（A100）自动启用。

```python
# 自动启用（默认）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 性能：与 FP32 相同的代码，自动获得 ~3x 加速

# 示例
model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")

# 在 A100 上，以下代码自动使用 TF32
outputs = model(**inputs)  # 无需修改代码
```

<div data-component="MixedPrecisionTrainingFlow"></div>

---

## 7.3 量化基础

### 7.3.1 什么是量化？

**量化（Quantization）** 是将浮点数映射到低位整数的过程：

$$
q = \text{round}\left(\frac{x}{\text{scale}}\right) + \text{zero\_point}
$$

**反量化（Dequantization）**：

$$
x \approx \text{scale} \times (q - \text{zero\_point})
$$

<div data-component="QuantizationProcessVisualizer"></div>

**量化类型**：

1. **对称量化（Symmetric）**：zero_point = 0
   $$q = \text{round}\left(\frac{x}{\text{scale}}\right)$$

2. **非对称量化（Asymmetric）**：zero_point ≠ 0
   $$q = \text{round}\left(\frac{x}{\text{scale}}\right) + \text{zero\_point}$$

**示例**：

```python
import torch

# 原始 FP32 权重
weights_fp32 = torch.randn(1000, 1000)

# INT8 对称量化
scale = weights_fp32.abs().max() / 127  # 127 = 2^7 - 1
weights_int8 = torch.round(weights_fp32 / scale).to(torch.int8)

# 反量化
weights_dequant = weights_int8.to(torch.float32) * scale

# 量化误差
error = (weights_fp32 - weights_dequant).abs().mean()
print(f"量化误差: {error:.6f}")  # ~0.001

# 显存节省
print(f"FP32: {weights_fp32.nbytes / 1024**2:.2f} MB")  # 3.81 MB
print(f"INT8: {weights_int8.nbytes / 1024**2:.2f} MB")  # 0.95 MB (↓75%)
```

### 7.3.2 逐张量 vs 逐通道量化

<div data-component="PerTensorVsPerChannelQuant"></div>

**逐张量量化（Per-Tensor）**：

```python
# 整个权重矩阵共享一个 scale
weight = torch.randn(128, 512)
scale = weight.abs().max() / 127
quant_weight = torch.round(weight / scale).to(torch.int8)

# 简单但精度较低
```

**逐通道量化（Per-Channel）**：

```python
# 每个输出通道独立的 scale
weight = torch.randn(128, 512)  # (out_channels, in_channels)
scales = weight.abs().max(dim=1, keepdim=True)[0] / 127  # (128, 1)
quant_weight = torch.round(weight / scales).to(torch.int8)

# 精度更高，广泛用于推理优化
```

**性能对比**：

| 方法 | 精度损失 | 计算开销 | 显存占用 |
|------|---------|----------|----------|
| 逐张量 | 中等 | 低 | 低 |
| 逐通道 | 低 | 略高 | 略高 |

---

## 7.4 INT8 量化

### 7.4.1 LLM.int8() - 8-bit Matrix Multiplication

**核心创新**（出自论文 *LLM.int8()*）：

> 混合精度矩阵乘法：大部分用 INT8，离群值（outliers）用 FP16。

**离群值问题**：

```python
# 大模型（>6.7B）中，0.1% 的激活值占据了巨大的数值范围
activations = model(inputs)
print(activations.abs().max())  # 可能 > 100
print(activations.abs().mean()) # 通常 < 1

# 如果用 INT8 量化，离群值会导致巨大误差
```

**LLM.int8() 解决方案**：

```python
from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig

# 配置 8-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,  # 离群值阈值
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# 推理（自动使用 INT8 + FP16 混合计算）
outputs = model.generate(**inputs)

# 显存占用: 14 GB → 7 GB (↓50%)
# 性能损失: < 1%
```

**工作原理**：

1. **检测离群值**：找出绝对值 > threshold 的元素
2. **分离计算**：
   - 99.9% 的元素：INT8 矩阵乘法
   - 0.1% 离群值：FP16 矩阵乘法
3. **合并结果**：相加得到最终输出

### 7.4.2 训练时 8-bit Adam

**8-bit AdamW** 可以大幅减少优化器状态占用：

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    optim="adamw_8bit",  # ← 使用 8-bit Adam
    # 或者
    # optim="paged_adamw_8bit",  # 分页优化器（QLoRA 推荐）
)

# 优化器状态占用:
# FP32 Adam: 模型参数 × 8 bytes (momentum + variance)
# 8-bit Adam: 模型参数 × 2 bytes (↓75%)

# 示例: 7B 模型
# FP32 Adam: 7B × 8 = 56 GB
# 8-bit Adam: 7B × 2 = 14 GB
```

---

## 7.5 INT4 量化

### 7.5.1 4-bit量化的挑战

将权重压缩到 **4-bit**（16 个离散值）极具挑战：

```python
# 4-bit 只有 16 个级别
int4_levels = [-8, -7, -6, ..., 0, ..., 6, 7]

# 对于标准正态分布的权重，误差很大
weights = torch.randn(1000000)
scale = weights.abs().max() / 7
quant_weights = torch.round(weights / scale).clamp(-8, 7)
dequant_weights = quant_weights * scale

mse = ((weights - dequant_weights) ** 2).mean()
print(f"INT4 MSE: {mse:.6f}")  # ~0.02（很大！）
```

### 7.5.2 NF4（NormalFloat4）量化

**QLoRA 的关键创新**：使用专为正态分布设计的数据类型。

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # ← 使用 NF4
    bnb_4bit_use_double_quant=True,  # 双重量化
    bnb_4bit_compute_dtype=torch.bfloat16  # 计算时用 BF16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# 显存占用: 14 GB → 3.5 GB (↓75%)
```

<div data-component="NF4vsINT4Comparison"></div>

**NF4 量化级别**（已在 Chapter 6 详细说明）：

```python
NF4_LEVELS = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
]
```

### 7.5.3 双重量化（Double Quantization）

**为什么需要？**

```python
# 7B 模型有 ~109M 个量化常数（scale）
num_blocks = 7_000_000_000 / 64  # 每 64 个参数共享 1 个 scale
num_scales = 109_375_000

# FP32 存储 scales: 109M × 4 bytes = 437 MB
# 这仍然占用大量显存！
```

**双重量化**：将 scales 本身也量化为 8-bit。

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # ← 启用双重量化
)

# 显存节省:
# - 原始 scales (FP32): 437 MB
# - 量化后 scales (INT8): 109 MB
# - 额外节省: 328 MB
```

<div data-component="DoubleQuantizationFlow"></div>

---

## 7.6 后训练量化（Post-Training Quantization, PTQ）

### 7.6.1 GPTQ (Generative Pre-trained Transformer Quantization)

**核心思想**：基于 Hessian 矩阵的最优量化。

**安装**：

```bash
pip install auto-gptq
```

**使用示例**：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

# 1. 配置 GPTQ
gptq_config = GPTQConfig(
    bits=4,  # 量化位数
    dataset="c4",  # 校准数据集
    tokenizer=None,  # 自动加载
    group_size=128,  # 分组大小
    desc_act=False,  # 激活重排序
)

# 2. 量化模型（需要在有 GPU 的环境下）
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=gptq_config,
    device_map="auto"
)

# 3. 保存量化模型
model.save_pretrained("./llama-2-7b-gptq-4bit")

# 4. 加载使用
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_quantized(
    "./llama-2-7b-gptq-4bit",
    device="cuda:0"
)

outputs = model.generate(**inputs)

# 性能: 几乎无损失（<1% perplexity 下降）
# 速度: 推理提速 2-3x
```

**GPTQ vs QLoRA**：

| 特性 | GPTQ | QLoRA |
|------|------|-------|
| 用途 | **推理** | **训练** |
| 量化时机 | 训练后 | 训练时 |
| 精度损失 | 极小（<1%） | 中等（<2%） |
| 推理速度 | 快 | 中等 |
| 显存占用 | 3.5 GB | 6.5 GB |
| 是否可训练 | ✗ | ✓ |

### 7.6.2 AWQ (Activation-aware Weight Quantization)

**核心创新**：根据激活值重要性保护关键权重。

```bash
pip install autoawq
```

**使用示例**：

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 1. 量化模型
model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. 配置量化参数
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# 3. 执行量化（需要少量校准数据）
model.quantize(tokenizer, quant_config=quant_config)

# 4. 保存
model.save_quantized("./llama-2-7b-awq-4bit")

# 5. 加载推理
model = AutoAWQForCausalLM.from_quantized("./llama-2-7b-awq-4bit")
outputs = model.generate(**inputs)

# 特点:
# - 推理速度比 GPTQ 更快（~20%）
# - 精度略优于 GPTQ
# - 支持更长上下文（FlashAttention-2 集成）
```

**AWQ vs GPTQ**：

<div data-component="PTQMethodComparison"></div>

---

## 7.7 梯度累积与 Checkpoint

### 7.7.1 梯度累积（Gradient Accumulation）

**问题**：显存不足，无法使用大 batch size。

**解决方案**：累积多个小 batch 的梯度，再更新参数。

```python
# 传统训练（batch_size=32）
for batch in dataloader:  # batch_size=32
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# 梯度累积（等效 batch_size=32）
accumulation_steps = 4
for i, batch in enumerate(dataloader):  # batch_size=8
    outputs = model(**batch)
    loss = outputs.loss / accumulation_steps  # ← 缩放损失
    loss.backward()  # 累积梯度
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()  # 每 4 步更新一次
        optimizer.zero_grad()
```

**在 Trainer 中使用**：

```python
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=8,  # 单卡 batch size
    gradient_accumulation_steps=4,   # 累积 4 步
    # 有效 batch_size = 8 × 4 = 32
)
```

<div data-component="GradientAccumulationVisualizer"></div>

### 7.7.2 梯度检查点（Gradient Checkpointing）

**问题**：大模型训练时，激活值占用大量显存。

**原理**：

```
正常训练:
  Forward: 存储所有中间激活值（用于 backward）
  显存占用: O(L × B × H)  (L=层数, B=batch, H=hidden_size)

梯度检查点:
  Forward: 只存储部分检查点
  Backward: 重新计算丢弃的激活值
  显存占用: O(√L × B × H)  (减少 √L 倍)
  时间开销: +20-30%
```

**使用方法**：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

# 启用梯度检查点
model.gradient_checkpointing_enable()

# 或在 TrainingArguments 中
training_args = TrainingArguments(
    output_dir="./output",
    gradient_checkpointing=True,  # ← 启用
)

# 显存节省: ~30-40%
# 训练速度: -20-30%
```

**完整组合（QLoRA + 梯度累积 + 检查点）**：

```python
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
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
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# 3. 准备训练（启用梯度检查点）
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True  # ← 启用
)

# 4. LoRA 配置
lora_config = LoraConfig(r=16, lora_alpha=32, ...)
model = get_peft_model(model, lora_config)

# 5. 训练配置
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=1,  # 最小 batch
    gradient_accumulation_steps=16,  # 累积 16 步
    bf16=True,
    optim="paged_adamw_8bit",  # 8-bit optimizer
    gradient_checkpointing=True,
)

# 总显存占用: ~6 GB
# 可在 RTX 3090 (24GB) 上训练 13B 模型！
```

---

## 7.8 量化感知训练（Quantization-Aware Training, QAT）

### 7.8.1 QAT 概述

**后训练量化（PTQ）** 的局限：
- 量化误差累积
- 大幅量化（如 4-bit）时性能下降明显

**量化感知训练（QAT）**：在训练时模拟量化，让模型适应量化误差。

```python
# PTQ: 训练完 → 量化
train(model)  # FP32
quantize(model)  # → INT8/INT4

# QAT: 训练时就模拟量化
for epoch in epochs:
    for batch in dataloader:
        # 前向传播时模拟量化
        outputs = quantized_forward(model, batch)
        loss.backward()
        optimizer.step()
```

**PyTorch 原生 QAT**：

```python
import torch
import torch.quantization as quant

# 1. 定义模型
model = MyModel()

# 2. 指定量化配置
model.qconfig = quant.get_default_qat_qconfig('fbgemm')

# 3. 准备 QAT
quant.prepare_qat(model, inplace=True)

# 4. 训练（自动模拟量化）
for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = model(batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 5. 转换为真正的量化模型
model.eval()
quant.convert(model, inplace=True)

# 6. 推理
outputs = model(input)  # INT8 推理
```

### 7.8.2 Transformer 的 QAT

**Hugging Face + Intel Neural Compressor**：

```bash
pip install neural-compressor
```

```python
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion
from neural_compressor import quantization
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# 配置 QAT
conf = PostTrainingQuantConfig(
    approach="static",  # 静态量化
    calibration_sampling_size=100,  # 校准样本数
)

# 量化
quantized_model = quantization.fit(
    model,
    conf,
    calib_dataloader=calib_dataloader,
    eval_func=eval_func
)

# 保存
quantized_model.save("./bert-qat-int8")

# 性能提升: 推理速度 +2-3x
# 精度损失: <0.5%
```

---

## 7.9 量化方法全景对比

<div data-component="QuantizationMethodsComprehensiveComparison"></div>

| 方法 | 位数 | 用途 | 精度损失 | 推理加速 | 训练成本 |
|------|------|------|----------|----------|----------|
| **FP16** | 16 | 训练 | <0.1% | 1.5x | 低 |
| **BF16** | 16 | 训练 | <0.1% | 1.5x | 低 |
| **INT8 (LLM.int8())** | 8 | 推理 | <1% | 2x | 无 |
| **GPTQ** | 4 | 推理 | <1% | 3x | 中（需校准） |
| **AWQ** | 4 | 推理 | <1% | 3.5x | 中（需校准） |
| **NF4 (QLoRA)** | 4 | 训练 | <2% | 1.2x | 高（需训练） |
| **QAT** | 8/4 | 训练+推理 | <0.5% | 2-3x | 很高 |

**选择建议**：

```python
# 训练场景
if 显存充足:
    use FP16/BF16  # 最佳性能
elif 显存受限:
    if 微调:
        use QLoRA  # 4-bit + LoRA
    else:
        use 梯度累积 + 检查点

# 推理场景
if 追求极致速度:
    use AWQ  # 最快
elif 追求精度:
    use GPTQ  # 最准
elif 内存极限:
    use NF4  # 最小
```

---

## 7.10 实战案例：在 6GB 显卡上训练 7B 模型

**完整脚本**：

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# === 1. 量化配置（4-bit NF4 + 双重量化） ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# === 2. 加载模型 ===
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
    max_memory={0: "5GB", "cpu": "20GB"}  # 限制显存使用
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# === 3. 准备训练（梯度检查点） ===
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True
)

# === 4. LoRA 配置 ===
lora_config = LoraConfig(
    r=8,  # 使用较小的秩
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # 只训练 Q 和 V
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# === 5. 加载数据集 ===
dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")

def tokenize(example):
    text = f"### Instruction:\n{example['instruction']}\n### Response:\n{example['output']}"
    return tokenizer(text, truncation=True, max_length=256)  # 限制长度

tokenized_dataset = dataset.map(tokenize)

# === 6. 训练配置（极致优化） ===
training_args = TrainingArguments(
    output_dir="./llama-2-7b-alpaca-6gb",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # 最小 batch
    gradient_accumulation_steps=32,  # 累积 32 步 → 有效 batch=32
    learning_rate=2e-4,
    bf16=True,  # BF16 计算
    optim="paged_adamw_8bit",  # 8-bit 分页优化器
    gradient_checkpointing=True,  # 梯度检查点
    logging_steps=10,
    save_strategy="epoch",
    max_grad_norm=0.3,  # 梯度裁剪
)

# === 7. 训练 ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# 监控显存
print(f"初始显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
trainer.train()
print(f"峰值显存: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

# === 8. 保存 ===
model.save_pretrained("./llama-2-7b-alpaca-final")

# 预期结果:
# - 峰值显存: ~5.8 GB
# - 训练速度: ~0.8 it/s (单卡 RTX 3060)
# - 性能: perplexity ~4.5（接近全量微调的 4.2）
```

---

## 7.11 常见问题与调试

### 7.11.1 FP16 训练不稳定

**症状**：Loss 变成 NaN 或 inf。

**原因与解决**：

```python
# 原因1: 梯度溢出
# 解决: 调整损失缩放
training_args = TrainingArguments(
    fp16=True,
    fp16_opt_level="O1",  # 降低优化级别
    max_grad_norm=1.0,  # 梯度裁剪
)

# 原因2: 学习率过高
# 解决: 降低学习率
training_args = TrainingArguments(
    learning_rate=1e-5,  # 从 5e-5 降到 1e-5
)

# 原因3: 切换到 BF16
training_args = TrainingArguments(
    bf16=True,  # BF16 更稳定
)
```

### 7.11.2 量化后性能严重下降

**诊断步骤**：

```python
# 1. 检查量化配置
print(model.config.quantization_config)

# 2. 测试不同量化方法
# INT8 vs INT4
bnb_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
bnb_config_4bit = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")

# 3. 评估量化误差
from transformers import pipeline

pipe_fp16 = pipeline("text-generation", model=model_fp16)
pipe_int4 = pipeline("text-generation", model=model_int4)

# 对比输出质量
prompt = "Once upon a time"
print(pipe_fp16(prompt, max_length=50))
print(pipe_int4(prompt, max_length=50))

# 4. 使用 QAT 或 GPTQ
# 如果 PTQ 性能不佳，考虑量化感知训练
```

### 7.11.3 OOM（显存溢出）

**解决方案（按优先级）**：

```python
# 1. 启用梯度累积
training_args.gradient_accumulation_steps = 8  # 从 1 → 8

# 2. 减小 batch size
training_args.per_device_train_batch_size = 1  # 降到 1

# 3. 启用梯度检查点
model.gradient_checkpointing_enable()

# 4. 使用 4-bit 量化
bnb_config = BitsAndBytesConfig(load_in_4bit=True, ...)

# 5. 缩短序列长度
tokenizer(..., max_length=256)  # 从 512 → 256

# 6. 使用 8-bit Adam
training_args.optim = "paged_adamw_8bit"

# 7. CPU offload
model = AutoModelForCausalLM.from_pretrained(
    ...,
    device_map="auto",
    max_memory={0: "10GB", "cpu": "30GB"}
)
```

---

## 7.12 本章小结

### 核心要点

1. **混合精度训练**：
   - FP16: V100+，速度 +80%
   - BF16: A100+，更稳定
   - TF32: A100 自动启用，无需配置

2. **量化方法**：
   - INT8（推理）: LLM.int8()，显存 -50%
   - INT4（推理）: GPTQ/AWQ，显存 -75%
   - NF4（训练）: QLoRA，显存 -77%

3. **显存优化组合**：
   - QLoRA + 梯度累积 + 梯度检查点 + 8-bit Adam
   - 可在 6GB 显卡上训练 7B 模型

### 最佳实践

```python
# 推理优化
model = AutoGPTQForCausalLM.from_quantized("gptq-4bit-model")  # GPTQ
# 或
model = AutoAWQForCausalLM.from_quantized("awq-4bit-model")  # AWQ（更快）

# 训练优化
bnb_config = BitsAndBytesConfig(load_in_4bit=True, ...)  # QLoRA
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    bf16=True,
    optim="paged_adamw_8bit",
)
```

### 进一步阅读

- [Mixed Precision Training 论文](https://arxiv.org/abs/1710.03740)
- [LLM.int8() 论文](https://arxiv.org/abs/2208.07339)
- [GPTQ 论文](https://arxiv.org/abs/2210.17323)
- [AWQ 论文](https://arxiv.org/abs/2306.00978)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)

---

**下一章预告**：Chapter 8 将探讨**分布式训练与加速**，包括 Accelerate、FSDP、DeepSpeed、多 GPU 训练策略等。

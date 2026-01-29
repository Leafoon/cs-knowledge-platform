---
title: "Chapter 5. Trainer API 完整指南"
description: "掌握 Trainer API 全流程、深入理解训练循环、回调机制与自定义扩展"
updated: "2026-01-22"
---

---

## 5.1 Trainer 核心概念

### 5.1.1 为什么使用 Trainer？

**传统 PyTorch 训练循环 vs Trainer 对比**：

```python
# ❌ 传统 PyTorch：需要手动实现大量细节
import torch
from torch.utils.data import DataLoader

model = MyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
dataloader = DataLoader(dataset, batch_size=16)

model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        # 手动处理设备迁移
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 前向传播
        outputs = model(**batch)
        loss = outputs.loss
        
        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # 手动实现梯度裁剪、学习率调度、日志记录、保存检查点...
        # 数百行代码

# ✅ Trainer：一键训练
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=5e-5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()  # 完成所有训练逻辑
```

**Trainer 自动处理的功能**：

1. ✅ **训练循环**：前向/反向传播、梯度更新
2. ✅ **分布式训练**：多 GPU、多节点自动支持
3. ✅ **混合精度**：FP16/BF16 自动混合精度训练
4. ✅ **梯度累积**：模拟大批次训练
5. ✅ **梯度裁剪**：防止梯度爆炸
6. ✅ **学习率调度**：warmup、线性/余弦衰减
7. ✅ **检查点管理**：自动保存最佳模型
8. ✅ **日志记录**：TensorBoard、WandB 集成
9. ✅ **评估循环**：验证集性能监控
10. ✅ **早停**：防止过拟合

<div data-component="TrainingLoopVisualizer"></div>

### 5.1.2 Trainer 架构概览

**核心组件关系图**：

```
┌─────────────────────────────────────────────────────────────┐
│                        Trainer                               │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ TrainingArgs │  │    Model     │  │   Dataset    │      │
│  │   参数配置    │  │   模型实例    │  │   训练数据    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │DataCollator  │  │  Optimizer   │  │   Scheduler  │      │
│  │  批处理逻辑   │  │   优化器     │  │  学习率调度   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Callbacks   │  │compute_metrics│ │   Logging    │      │
│  │  回调函数     │  │  评估指标    │  │   日志系统    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 5.1.3 最简训练示例

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset

# 1. 加载模型和 tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 2. 准备数据集
dataset = load_dataset("glue", "sst2")

def tokenize_function(examples):
    return tokenizer(examples["sentence"], truncation=True, padding=False)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3. 配置训练参数
training_args = TrainingArguments(
    output_dir="./results",          # 输出目录
    num_train_epochs=3,              # 训练轮数
    per_device_train_batch_size=16,  # 批次大小
    evaluation_strategy="epoch",     # 每个 epoch 评估一次
    save_strategy="epoch",           # 每个 epoch 保存一次
    logging_dir="./logs"             # 日志目录
)

# 4. 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"]
)

# 5. 开始训练
trainer.train()

# 6. 评估
metrics = trainer.evaluate()
print(metrics)

# 7. 保存模型
trainer.save_model("./final_model")
```

**训练输出示例**：

```
Epoch | Training Loss | Validation Loss | Runtime | Samples/s
------|---------------|-----------------|---------|----------
  1   |    0.4523     |     0.3891      | 120.5s  |  558.2
  2   |    0.3012     |     0.3654      | 118.3s  |  568.9
  3   |    0.2145     |     0.3721      | 119.1s  |  564.8

***** eval metrics *****
  eval_loss         = 0.3721
  eval_runtime      = 12.34s
  eval_samples_per_second = 687.5
```

---

## 5.2 TrainingArguments 参数详解

`TrainingArguments` 包含 **100+ 个参数**，控制训练的方方面面。

<div data-component="TrainingArgumentsExplorer"></div>

### 5.2.1 基础参数

**必需参数**：

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./results",  # 必需：模型和检查点保存目录
)
```

**核心训练参数**：

```python
args = TrainingArguments(
    output_dir="./results",
    
    # === 训练轮数 ===
    num_train_epochs=3,              # 总 epoch 数
    # 或使用
    max_steps=10000,                 # 总训练步数（与 num_train_epochs 二选一）
    
    # === 批次大小 ===
    per_device_train_batch_size=16,  # 每个 GPU 的训练批次大小
    per_device_eval_batch_size=32,   # 每个 GPU 的评估批次大小（可以更大）
    
    # === 学习率 ===
    learning_rate=5e-5,              # 初始学习率
    weight_decay=0.01,               # L2 正则化系数
    
    # === 优化器 ===
    optim="adamw_torch",             # 优化器类型
    # 可选: adamw_hf, adamw_torch, adamw_torch_fused, adamw_apex_fused,
    #      adafactor, adamw_anyprecision, sgd, adagrad
    
    # === 学习率调度 ===
    lr_scheduler_type="linear",      # 学习率调度器
    # 可选: linear, cosine, cosine_with_restarts, polynomial, constant,
    #      constant_with_warmup
    warmup_ratio=0.1,                # warmup 比例（总步数的 10%）
    # 或使用
    warmup_steps=500,                # warmup 步数（与 warmup_ratio 二选一）
)
```

### 5.2.2 评估与保存

```python
args = TrainingArguments(
    output_dir="./results",
    
    # === 评估策略 ===
    evaluation_strategy="steps",     # 评估时机
    # 可选: "no"（不评估）, "steps"（每 N 步）, "epoch"（每个 epoch）
    eval_steps=500,                  # 每 500 步评估一次
    eval_delay=0,                    # 延迟评估（从第 N 步开始）
    
    # === 保存策略 ===
    save_strategy="steps",           # 保存时机（同 evaluation_strategy）
    save_steps=500,                  # 每 500 步保存一次
    save_total_limit=3,              # 最多保留 3 个检查点（删除旧的）
    
    # === 最佳模型 ===
    load_best_model_at_end=True,     # 训练结束后加载最佳模型
    metric_for_best_model="eval_loss",  # 用于判断最佳模型的指标
    greater_is_better=False,         # eval_loss 越小越好
    
    # === 日志记录 ===
    logging_dir="./logs",            # TensorBoard 日志目录
    logging_strategy="steps",        # 日志记录时机
    logging_steps=100,               # 每 100 步记录一次
    logging_first_step=True,         # 记录第一步
    
    # === 报告集成 ===
    report_to=["tensorboard", "wandb"],  # 上报到哪些平台
    # 可选: tensorboard, wandb, mlflow, comet_ml, clearml, all, none
)
```

**保存目录结构**：

```
./results/
├── checkpoint-500/
│   ├── config.json
│   ├── model.safetensors
│   ├── optimizer.pt
│   ├── scheduler.pt
│   ├── trainer_state.json
│   └── training_args.bin
├── checkpoint-1000/
├── checkpoint-1500/
└── runs/  # TensorBoard 日志
    └── ...
```

### 5.2.3 混合精度与优化

```python
args = TrainingArguments(
    output_dir="./results",
    
    # === 混合精度训练 ===
    fp16=True,                       # 启用 FP16 混合精度（Nvidia GPU）
    # 或
    bf16=True,                       # 启用 BF16 混合精度（Ampere+ GPU，更稳定）
    fp16_opt_level="O1",             # FP16 优化级别（apex）
    
    # === 梯度累积 ===
    gradient_accumulation_steps=4,   # 累积 4 步再更新（相当于 batch_size × 4）
    
    # === 梯度裁剪 ===
    max_grad_norm=1.0,               # 梯度范数裁剪阈值
    
    # === 优化技巧 ===
    gradient_checkpointing=True,     # 启用梯度检查点（节省显存，但速度慢 20%）
    
    # === DataLoader 优化 ===
    dataloader_num_workers=4,        # 数据加载进程数
    dataloader_pin_memory=True,      # 固定内存（加速 GPU 传输）
    
    # === 编译优化（PyTorch 2.0+）===
    torch_compile=True,              # 启用 torch.compile 编译
    torch_compile_backend="inductor", # 编译后端
    torch_compile_mode="default",    # 编译模式（default, reduce-overhead, max-autotune）
)
```

**混合精度性能对比**：

```python
# FP32（默认）
# - 显存占用: 100%
# - 训练速度: 1.0x
# - 数值稳定性: 最佳

# FP16
# - 显存占用: ~50%
# - 训练速度: 2-3x
# - 数值稳定性: 可能溢出（需要 loss scaling）

# BF16（推荐，Ampere+ GPU）
# - 显存占用: ~50%
# - 训练速度: 2-3x
# - 数值稳定性: 优于 FP16（更大动态范围）
```

<div data-component="MixedPrecisionComparison"></div>

### 5.2.4 分布式训练

```python
args = TrainingArguments(
    output_dir="./results",
    
    # === 多 GPU ===
    # 自动检测所有可用 GPU（使用 DataParallel 或 DistributedDataParallel）
    # 无需额外配置
    
    # === DeepSpeed ===
    deepspeed="ds_config.json",      # DeepSpeed 配置文件
    
    # === FSDP（Fully Sharded Data Parallel）===
    fsdp="full_shard auto_wrap",     # FSDP 策略
    fsdp_config={
        "min_num_params": 1e6,       # 最小分片参数量
        "backward_prefetch": "backward_pre",
        "forward_prefetch": False,
        "cpu_offload": False
    },
    
    # === 分布式采样 ===
    ddp_find_unused_parameters=False,  # 是否查找未使用的参数（慢）
    ddp_bucket_cap_mb=25,            # DDP 通信桶大小
    
    # === 本地进程数 ===
    local_rank=-1,                   # 本地进程排名（自动设置）
)
```

**启动分布式训练**：

```bash
# 方式1：使用 accelerate（推荐）
accelerate launch train.py

# 方式2：使用 torchrun（PyTorch 原生）
torchrun --nproc_per_node=4 train.py

# 方式3：使用 deepspeed
deepspeed train.py --deepspeed ds_config.json
```

### 5.2.5 其他重要参数

```python
args = TrainingArguments(
    output_dir="./results",
    
    # === 种子 ===
    seed=42,                         # 随机种子（保证可复现）
    data_seed=42,                    # 数据采样种子
    
    # === 早停 ===
    # 需要配合 EarlyStoppingCallback 使用
    # metric_for_best_model 和 load_best_model_at_end 已在前面设置
    
    # === 推送到 Hub ===
    push_to_hub=True,                # 训练结束后推送到 Hugging Face Hub
    hub_model_id="my-model",         # Hub 上的模型名称
    hub_strategy="every_save",       # 推送策略
    # 可选: end, every_save, checkpoint, all_checkpoints
    
    # === 其他 ===
    remove_unused_columns=True,      # 自动移除模型不需要的列
    label_names=["labels"],          # 标签列名称
    include_inputs_for_metrics=False, # 是否在 compute_metrics 中包含输入
    
    # === 调试 ===
    debug="underflow_overflow",      # 调试模式
    # 可选: underflow_overflow（检测数值问题）
)
```

---

## 5.3 训练循环详解

<div data-component="TrainingStepBreakdown"></div>

### 5.3.1 训练流程剖析

**完整训练循环伪代码**：

```python
# Trainer.train() 内部流程（简化版）

for epoch in range(num_epochs):
    model.train()
    
    for step, batch in enumerate(train_dataloader):
        # 1. 前向传播
        outputs = model(**batch)
        loss = outputs.loss
        
        # 2. 损失缩放（混合精度）
        if fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        # 3. 梯度累积
        if (step + 1) % gradient_accumulation_steps == 0:
            # 4. 梯度裁剪
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # 5. 优化器更新
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # 6. 日志记录
            if (step + 1) % logging_steps == 0:
                log_metrics({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})
            
            # 7. 评估
            if (step + 1) % eval_steps == 0:
                eval_metrics = evaluate()
            
            # 8. 保存检查点
            if (step + 1) % save_steps == 0:
                save_checkpoint()
```

### 5.3.2 自定义训练步骤

**覆盖 `training_step()` 方法**：

```python
from transformers import Trainer

class CustomTrainer(Trainer):
    def training_step(self, model, inputs):
        """自定义单步训练逻辑"""
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # 前向传播
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        # 自定义损失调整（例如：添加正则项）
        l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
        loss = loss + 0.001 * l2_reg
        
        # 反向传播
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        self.accelerator.backward(loss)
        
        return loss.detach()
```

**覆盖 `compute_loss()` 方法**：

```python
class CustomLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """自定义损失计算"""
        labels = inputs.pop("labels")
        
        # 前向传播
        outputs = model(**inputs)
        logits = outputs.logits
        
        # 自定义损失（例如：Focal Loss）
        ce_loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** 2 * ce_loss).mean()
        
        return (focal_loss, outputs) if return_outputs else focal_loss
```

### 5.3.3 评估循环

**覆盖 `evaluation_loop()`**：

```python
class CustomEvalTrainer(Trainer):
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ...):
        """自定义评估循环"""
        model = self.model
        model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0
        
        for step, inputs in enumerate(dataloader):
            with torch.no_grad():
                outputs = model(**inputs)
                loss = outputs.loss
                logits = outputs.logits
            
            total_loss += loss.item()
            all_preds.append(logits.cpu())
            all_labels.append(inputs["labels"].cpu())
        
        # 计算指标
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        metrics = self.compute_metrics(
            EvalPrediction(predictions=all_preds, label_ids=all_labels)
        )
        metrics["eval_loss"] = total_loss / len(dataloader)
        
        return metrics
```

---

## 5.4 评估指标（compute_metrics）

### 5.4.1 基础评估函数

```python
from datasets import load_metric
import numpy as np

# 方式1：使用 datasets 库的 metric（推荐）
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    """计算准确率"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 使用
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)
```

**多指标评估**：

```python
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics(eval_pred):
    """计算多个分类指标"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
```

### 5.4.2 NLP 任务评估

**序列标注（NER）**：

```python
from datasets import load_metric

seqeval = load_metric("seqeval")

def compute_metrics(eval_pred):
    """NER 评估"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # 移除 padding（-100）
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_preds = [
        [label_names[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    
    results = seqeval.compute(predictions=true_preds, references=true_labels)
    
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]
    }
```

**问答（SQuAD）**：

```python
from datasets import load_metric

squad_metric = load_metric("squad")

def compute_metrics(eval_pred):
    """SQuAD 评估"""
    predictions, labels = eval_pred
    
    # predictions: (start_logits, end_logits)
    # labels: (start_positions, end_positions)
    
    # 解码预测答案（需要实现 postprocess 逻辑）
    predictions = postprocess_qa_predictions(
        examples=eval_examples,
        features=eval_features,
        predictions=predictions
    )
    
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples]
    
    return squad_metric.compute(predictions=predictions, references=references)
```

**文本生成（ROUGE）**：

```python
from datasets import load_metric

rouge = load_metric("rouge")

def compute_metrics(eval_pred):
    """摘要生成评估"""
    predictions, labels = eval_pred
    
    # 解码 token IDs 为文本
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # 替换 -100（用于 label padding）
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # ROUGE 需要换行分隔
    decoded_preds = ["\n".join(pred.split()) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.split()) for label in decoded_labels]
    
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    
    return {
        "rouge1": result["rouge1"].mid.fmeasure,
        "rouge2": result["rouge2"].mid.fmeasure,
        "rougeL": result["rougeL"].mid.fmeasure
    }
```

### 5.4.3 自定义复杂指标

```python
def compute_metrics(eval_pred):
    """自定义复杂指标"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # 1. 整体准确率
    acc = accuracy_score(labels, predictions)
    
    # 2. 每个类别的准确率
    per_class_acc = {}
    for i, label_name in enumerate(label_names):
        mask = labels == i
        if mask.sum() > 0:
            per_class_acc[f"acc_{label_name}"] = accuracy_score(
                labels[mask], predictions[mask]
            )
    
    # 3. 混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, predictions)
    
    # 4. 置信度分析
    probs = softmax(logits, axis=-1)
    max_probs = np.max(probs, axis=-1)
    avg_confidence = max_probs.mean()
    correct_confidence = max_probs[predictions == labels].mean()
    wrong_confidence = max_probs[predictions != labels].mean()
    
    return {
        "accuracy": acc,
        **per_class_acc,
        "avg_confidence": avg_confidence,
        "correct_confidence": correct_confidence,
        "wrong_confidence": wrong_confidence
    }
```

---

## 5.5 回调函数（Callbacks）

回调函数允许在训练的特定时刻插入自定义逻辑（如早停、学习率调整、自定义日志等）。

<div data-component="CallbackFlow"></div>

### 5.5.1 内置回调

**1. EarlyStoppingCallback - 早停**

```python
from transformers import EarlyStoppingCallback, Trainer

# 创建早停回调
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,  # 连续 3 次评估没有改善则停止
    early_stopping_threshold=0.0  # 改善阈值
)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[early_stopping]  # 添加回调
)

trainer.train()
```

**2. TensorBoardCallback - TensorBoard 集成**

```python
from transformers import TrainerCallback

# 默认已包含，无需手动添加
# 查看日志：
# tensorboard --logdir ./logs
```

**3. ProgressCallback - 进度条**

```python
# 默认包含，显示训练进度
# 可以禁用：
training_args = TrainingArguments(
    output_dir="./results",
    disable_tqdm=True  # 禁用进度条
)
```

### 5.5.2 自定义回调

**基础自定义回调**：

```python
from transformers import TrainerCallback

class CustomCallback(TrainerCallback):
    """自定义回调示例"""
    
    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时"""
        print("🚀 训练开始！")
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """每个 epoch 开始时"""
        print(f"📊 Epoch {state.epoch} 开始")
    
    def on_step_end(self, args, state, control, **kwargs):
        """每步结束时"""
        if state.global_step % 100 == 0:
            print(f"✅ 完成 {state.global_step} 步")
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """评估后"""
        print(f"📈 评估结果: {metrics}")
    
    def on_save(self, args, state, control, **kwargs):
        """保存检查点后"""
        print(f"💾 保存检查点: {state.global_step} 步")
    
    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时"""
        print("🎉 训练完成！")

# 使用
trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[CustomCallback()]
)
```

**高级回调：学习率重启**

```python
class CosineRestartCallback(TrainerCallback):
    """余弦退火学习率重启"""
    
    def __init__(self, restart_every=1000):
        self.restart_every = restart_every
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.restart_every == 0:
            # 重置学习率调度器
            optimizer = kwargs["optimizer"]
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.learning_rate
            print(f"🔄 学习率重启: {args.learning_rate}")
```

**梯度监控回调**：

```python
class GradientMonitorCallback(TrainerCallback):
    """监控梯度范数"""
    
    def on_step_end(self, args, state, control, model, **kwargs):
        if state.global_step % args.logging_steps == 0:
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            print(f"📏 梯度范数: {total_norm:.4f}")
            
            # 记录到 TensorBoard
            if hasattr(state, "log_history"):
                state.log_history[-1]["grad_norm"] = total_norm
```

**WandB 集成回调**：

```python
import wandb

class WandbCallback(TrainerCallback):
    """Weights & Biases 集成"""
    
    def on_train_begin(self, args, state, control, model, **kwargs):
        wandb.init(
            project="my-project",
            name=args.run_name,
            config=args.to_dict()
        )
        wandb.watch(model)
    
    def on_log(self, args, state, control, logs, **kwargs):
        wandb.log(logs)
    
    def on_train_end(self, args, state, control, **kwargs):
        wandb.finish()
```

### 5.5.3 回调执行顺序

**所有回调事件**：

```python
# 训练生命周期事件（按调用顺序）
on_init_end              # Trainer 初始化完成
on_train_begin           # 训练开始前
on_epoch_begin           # 每个 epoch 开始前
  on_step_begin          # 每步开始前
  on_substep_end         # 梯度累积子步骤结束（如果启用）
  on_step_end            # 每步结束后
  on_log                 # 日志记录时
  on_evaluate            # 评估时
  on_save                # 保存检查点时
  on_prediction_step     # 预测步骤时
on_epoch_end             # 每个 epoch 结束后
on_train_end             # 训练结束后
```

---

## 5.6 优化器与学习率调度

### 5.6.1 自定义优化器

**使用不同优化器**：

```python
from transformers import Trainer, TrainingArguments
from torch.optim import SGD, Adam

class CustomOptimizerTrainer(Trainer):
    def create_optimizer(self):
        """自定义优化器"""
        # 方式1：使用 PyTorch 原生优化器
        self.optimizer = SGD(
            self.model.parameters(),
            lr=self.args.learning_rate,
            momentum=0.9,
            weight_decay=self.args.weight_decay
        )
        
        # 方式2：不同层不同学习率
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if "bias" not in n],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if "bias" in n],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = Adam(optimizer_grouped_parameters, lr=self.args.learning_rate)
```

**层级学习率（Layer-wise Learning Rate Decay）**：

```python
def get_optimizer_grouped_parameters(model, learning_rate, weight_decay, layerwise_lr_decay=0.95):
    """为不同层设置不同学习率"""
    no_decay = ["bias", "LayerNorm.weight"]
    
    # BERT 有 12 层（layer.0 到 layer.11）
    num_layers = model.config.num_hidden_layers
    layers = [model.bert.embeddings] + list(model.bert.encoder.layer)
    layers.reverse()  # 从顶层到底层
    
    optimizer_grouped_parameters = []
    
    for i, layer in enumerate(layers):
        # 计算当前层的学习率（顶层最大，底层逐渐衰减）
        lr = learning_rate * (layerwise_lr_decay ** i)
        
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
    
    return optimizer_grouped_parameters

# 使用
class LayerwiseLRTrainer(Trainer):
    def create_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            get_optimizer_grouped_parameters(
                self.model,
                learning_rate=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                layerwise_lr_decay=0.95
            )
        )
```

### 5.6.2 自定义学习率调度器

```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class CustomSchedulerTrainer(Trainer):
    def create_scheduler(self, num_training_steps, optimizer=None):
        """自定义学习率调度器"""
        if optimizer is None:
            optimizer = self.optimizer
        
        # 余弦退火 + 重启
        self.lr_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=1000,  # 第一次重启周期
            T_mult=2,  # 每次重启周期翻倍
            eta_min=1e-6  # 最小学习率
        )
```

**Warmup + Linear Decay（手动实现）**：

```python
from torch.optim.lr_scheduler import LambdaLR

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Warmup + 线性衰减"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup 阶段：线性增长
            return float(current_step) / float(max(1, num_warmup_steps))
        # 衰减阶段：线性衰减到 0
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda)

# 使用
class WarmupLinearTrainer(Trainer):
    def create_scheduler(self, num_training_steps, optimizer=None):
        if optimizer is None:
            optimizer = self.optimizer
        
        num_warmup_steps = int(num_training_steps * 0.1)  # 10% warmup
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
```

<div data-component="LearningRateScheduler"></div>

---

## 5.7 高级训练技巧

### 5.7.1 梯度累积

**原理**：模拟更大的批次大小，避免显存不足。

```python
# 假设显存只能容纳 batch_size=8，但想要 batch_size=32 的效果
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,     # 实际批次大小
    gradient_accumulation_steps=4,     # 累积 4 步
    # 等效批次大小 = 8 × 4 = 32
)

# Trainer 自动处理：
# - 每 4 步才更新一次参数
# - 损失自动除以 4
# - 学习率调度器在累积完成后才步进
```

**实战示例**：

```python
# 目标：在 1 张 GPU（16GB）上训练 GPT-2 Large
# 问题：完整批次需要 32GB 显存

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,     # 单张 GPU 只能放 1 个样本
    gradient_accumulation_steps=32,    # 累积 32 步
    # 等效批次大小 = 32（与多 GPU 训练相同）
    
    fp16=True,  # 混合精度进一步节省显存
    gradient_checkpointing=True  # 牺牲 20% 速度换取 50% 显存
)
```

### 5.7.2 梯度检查点（Gradient Checkpointing）

**原理**：不保存所有中间激活值，反向传播时重新计算。

```python
# 启用梯度检查点
model.gradient_checkpointing_enable()

# 或在 TrainingArguments 中设置
training_args = TrainingArguments(
    output_dir="./results",
    gradient_checkpointing=True
)

# 效果：
# - 显存占用减少 ~50%
# - 训练速度降低 ~20%
# - 适合显存不足时使用
```

**自定义检查点策略**：

```python
# 只对部分层启用检查点
from torch.utils.checkpoint import checkpoint

class SelectiveCheckpointModel(torch.nn.Module):
    def forward(self, x):
        # 前几层正常计算
        x = self.layer1(x)
        x = self.layer2(x)
        
        # 中间层使用检查点（最耗显存）
        x = checkpoint(self.layer3, x)
        x = checkpoint(self.layer4, x)
        
        # 最后几层正常计算
        x = self.layer5(x)
        return x
```

### 5.7.3 模型并行（Model Parallelism）

**设备映射（Device Map）**：

```python
from transformers import AutoModelForCausalLM

# 自动跨 GPU 分配层
model = AutoModelForCausalLM.from_pretrained(
    "gpt-j-6B",
    device_map="auto",  # 自动分配到多张 GPU
    torch_dtype=torch.float16
)

# 手动指定设备映射
device_map = {
    "transformer.wte": 0,      # Embedding 层在 GPU 0
    "transformer.h.0": 0,      # 第 0 层在 GPU 0
    "transformer.h.1": 0,
    "transformer.h.2": 1,      # 第 2 层在 GPU 1
    # ...
    "lm_head": 3               # 输出层在 GPU 3
}

model = AutoModelForCausalLM.from_pretrained(
    "gpt-j-6B",
    device_map=device_map,
    torch_dtype=torch.float16
)
```

### 5.7.4 8-bit / 4-bit 训练

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 8-bit 量化加载
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=quantization_config,
    device_map="auto"
)

# 4-bit 量化（QLoRA）
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=quantization_config,
    device_map="auto"
)

# 显存占用对比：
# - FP32: ~28GB
# - FP16: ~14GB
# - 8-bit: ~7GB
# - 4-bit: ~3.5GB
```

---

## 5.8 实战案例

<div data-component="TrainingMetricsPlot"></div>

### 5.8.1 情感分类完整流程

```python
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# === 1. 数据准备 ===
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets = tokenized_datasets.remove_columns(["text"])

# 训练集采样（快速实验）
small_train = tokenized_datasets["train"].shuffle(seed=42).select(range(5000))
small_eval = tokenized_datasets["test"].shuffle(seed=42).select(range(500))

# === 2. 模型初始化 ===
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# === 3. 评估指标 ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

# === 4. 训练配置 ===
training_args = TrainingArguments(
    output_dir="./imdb-bert",
    
    # 基础参数
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    
    # 优化器
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    
    # 混合精度
    fp16=True,
    
    # 评估与保存
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    
    # 日志
    logging_dir="./logs",
    logging_steps=50,
    report_to=["tensorboard"],
    
    # 其他
    seed=42,
    dataloader_num_workers=4,
)

# === 5. DataCollator ===
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# === 6. Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train,
    eval_dataset=small_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# === 7. 训练 ===
trainer.train()

# === 8. 评估 ===
metrics = trainer.evaluate()
print(f"Final Metrics: {metrics}")

# === 9. 保存 ===
trainer.save_model("./imdb-bert-final")

# === 10. 推理测试 ===
test_texts = [
    "This movie was absolutely fantastic!",
    "Worst film I've ever seen."
]

inputs = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)

print(f"Predictions: {predictions}")  # tensor([1, 0]) - positive, negative
```

### 5.8.2 多 GPU 训练

```python
# 方式1：使用 Accelerate（推荐）
# accelerate config  # 首次运行，配置分布式策略
# accelerate launch train.py

# train.py
from accelerate import Accelerator

accelerator = Accelerator()

# 正常定义模型、数据、优化器
model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)

# 训练循环（Accelerator 自动处理分布式）
for batch in train_dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()

# 方式2：Trainer 自动检测
# 无需额外代码，Trainer 自动使用所有可见 GPU
trainer = Trainer(...)
trainer.train()

# 启动：
# python -m torch.distributed.launch --nproc_per_node=4 train.py
# 或
# torchrun --nproc_per_node=4 train.py
```

---

## 本章总结

**核心要点**：

1. ✅ **Trainer 优势**：自动化训练流程，支持分布式、混合精度、检查点管理
2. ✅ **TrainingArguments**：100+ 参数控制训练细节
3. ✅ **评估指标**：compute_metrics 自定义任务评估
4. ✅ **回调函数**：早停、自定义日志、WandB 集成
5. ✅ **优化技巧**：梯度累积、梯度检查点、层级学习率、量化训练
6. ✅ **分布式训练**：多 GPU、FSDP、DeepSpeed 无缝集成

**最佳实践**：

- 使用 `fp16=True` 或 `bf16=True` 加速训练
- 显存不足时：梯度累积 + 梯度检查点
- 大模型训练：device_map="auto" + 8-bit/4-bit 量化
- 实验阶段：小数据集 + 快速迭代
- 生产训练：早停 + 多指标监控 + 检查点管理

**下一章预告**：  
Chapter 6 将学习 **PEFT（参数高效微调）**，包括 LoRA、QLoRA、Adapter、Prefix Tuning 等技术，用 1% 的参数量实现接近全量微调的效果。

---

## 练习题

1. **基础题**：使用 Trainer 在 GLUE 的 SST-2 任务上微调 BERT，记录最佳 F1 分数和训练时长。

2. **进阶题**：实现一个自定义回调函数，监控每个 epoch 的梯度范数、学习率、验证集 loss，并绘制曲线图。

3. **挑战题**：在单张 GPU（12GB）上使用梯度累积 + 梯度检查点训练 RoBERTa-Large（355M 参数）。对比不同 `gradient_accumulation_steps`（2/4/8）和 `gradient_checkpointing`（开/关）组合的显存占用和训练速度。

4. **思考题**：为什么 BF16 比 FP16 更稳定？在什么硬件上可以使用 BF16？梯度累积会影响 BatchNorm 的行为吗？如何解决？

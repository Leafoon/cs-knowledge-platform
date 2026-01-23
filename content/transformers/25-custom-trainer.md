# Chapter 25: 自定义 Trainer 与训练循环 (Custom Trainer & Training Loop)

在前面的章节中，我们使用 `Trainer` API 进行了大量训练实验。本章将深入探讨 **Trainer 的内部机制**，学习如何通过继承 `Trainer` 类来自定义训练逻辑、实现自定义 Callback、完全自定义训练循环（使用 Accelerate）以及实现高级损失函数（Focal Loss、Contrastive Loss、KL Divergence 等）。这些技能对于研究前沿方法、适配特殊任务至关重要。

---

## 25.1 Trainer 内部机制

### 25.1.1 训练循环源码走读

`Trainer` 的核心训练循环位于 `train()` 方法中，简化后的逻辑如下：

```python
# transformers/trainer.py (简化版)
class Trainer:
    def train(self, resume_from_checkpoint=None):
        # 1. 准备阶段
        train_dataloader = self.get_train_dataloader()
        optimizer = self.create_optimizer()
        lr_scheduler = self.create_scheduler(optimizer)
        
        # 2. 分布式准备
        model, optimizer, train_dataloader = self.accelerator.prepare(
            self.model, optimizer, train_dataloader
        )
        
        # 3. 开始训练
        for epoch in range(num_epochs):
            model.train()
            for step, inputs in enumerate(train_dataloader):
                # 前向传播
                outputs = model(**inputs)
                loss = outputs.loss
                
                # 反向传播
                loss.backward()
                
                # 梯度累积
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                # 日志记录
                if step % logging_steps == 0:
                    self.log({"loss": loss.item()})
                
                # 保存检查点
                if step % save_steps == 0:
                    self.save_model(output_dir)
            
            # 每个 epoch 结束后评估
            if self.args.evaluation_strategy == "epoch":
                self.evaluate()
        
        return TrainOutput(...)
```

**关键组件**：
1. **DataLoader**：通过 `get_train_dataloader()` 创建
2. **Optimizer**：通过 `create_optimizer()` 创建（默认 AdamW）
3. **Scheduler**：通过 `create_scheduler()` 创建（线性衰减或余弦退火）
4. **Accelerator**：处理混合精度、分布式、梯度累积
5. **Checkpointing**：自动保存最佳模型和 optimizer 状态

### 25.1.2 钩子函数（Hooks）位置

`Trainer` 提供了大量钩子函数供子类重写：

<div data-component="TrainerHookFlow"></div>

| 钩子函数 | 调用时机 | 用途示例 |
|----------|----------|----------|
| **`compute_loss()`** | 每个 batch 前向传播后 | 自定义损失函数 |
| **`training_step()`** | 每个训练步骤 | 自定义梯度计算 |
| **`prediction_step()`** | 每个评估步骤 | 自定义评估逻辑 |
| **`evaluation_loop()`** | 整个评估循环 | 自定义评估流程 |
| **`create_optimizer()`** | 训练开始前 | 使用自定义优化器 |
| **`create_scheduler()`** | 训练开始前 | 自定义学习率调度 |
| **`save_model()`** | 保存检查点时 | 保存额外状态 |
| **`log()`** | 记录日志时 | 自定义日志格式 |

### 25.1.3 自定义评估指标

默认情况下，`Trainer` 只记录 loss。要添加自定义指标：

```python
from transformers import Trainer, TrainingArguments
from datasets import load_metric
import numpy as np

# 加载指标
accuracy_metric = load_metric("accuracy")
f1_metric = load_metric("f1")

def compute_metrics(eval_pred):
    """
    自定义评估指标函数
    
    Args:
        eval_pred: EvalPrediction 对象，包含 predictions 和 label_ids
    
    Returns:
        dict: 指标字典
    """
    predictions, labels = eval_pred
    
    # predictions 是 logits，需要转换为类别
    preds = np.argmax(predictions, axis=1)
    
    # 计算多个指标
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")
    
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"]
    }

# 使用
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics  # 传入自定义函数
)

# 训练时会自动输出 accuracy 和 f1
trainer.train()
```

**高级示例：多任务评估**
```python
def compute_multitask_metrics(eval_pred):
    """
    多任务学习的评估指标
    """
    predictions, labels = eval_pred
    
    # 假设 predictions 是 (batch, num_tasks, num_classes)
    task1_preds = np.argmax(predictions[:, 0, :], axis=1)
    task2_preds = np.argmax(predictions[:, 1, :], axis=1)
    
    task1_labels = labels[:, 0]
    task2_labels = labels[:, 1]
    
    return {
        "task1_accuracy": (task1_preds == task1_labels).mean(),
        "task2_accuracy": (task2_preds == task2_labels).mean(),
        "combined_accuracy": ((task1_preds == task1_labels) & (task2_preds == task2_labels)).mean()
    }
```

---

## 25.2 继承 Trainer 类

### 25.2.1 重写 compute_loss()

最常见的自定义需求是使用非标准损失函数。

**示例：Label Smoothing**
```python
import torch.nn.functional as F

class LabelSmoothingTrainer(Trainer):
    def __init__(self, label_smoothing=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = label_smoothing
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        使用 Label Smoothing 的交叉熵损失
        """
        labels = inputs.pop("labels")
        
        # 前向传播
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Label Smoothing 损失
        # 公式：(1 - ε) * NLL(y_true) + ε * NLL(uniform)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # One-hot encoding
        num_classes = logits.size(-1)
        one_hot = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1)
        
        # Smooth labels
        smooth_labels = one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        
        # Compute loss
        loss = -(smooth_labels * log_probs).sum(dim=-1).mean()
        
        return (loss, outputs) if return_outputs else loss
```

**使用**：
```python
trainer = LabelSmoothingTrainer(
    label_smoothing=0.1,
    model=model,
    args=training_args,
    train_dataset=train_dataset
)
trainer.train()
```

### 25.2.2 重写 training_step()

控制整个训练步骤（包括梯度裁剪、对抗训练等）。

**示例：对抗训练（FGM）**
```python
class AdversarialTrainer(Trainer):
    def __init__(self, adv_epsilon=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adv_epsilon = adv_epsilon
    
    def training_step(self, model, inputs):
        """
        Fast Gradient Method (FGM) 对抗训练
        
        步骤：
        1. 正常前向 + 反向，计算梯度
        2. 在 embedding 上添加对抗扰动
        3. 再次前向，计算对抗损失
        4. 反向传播对抗损失
        5. 恢复原始 embedding
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # === 第一步：正常训练 ===
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        # 反向传播（但不更新参数）
        self.accelerator.backward(loss)
        
        # === 第二步：对抗训练 ===
        # 保存原始 embedding
        embedding_layer = model.get_input_embeddings()
        original_embedding = embedding_layer.weight.data.clone()
        
        # 计算对抗扰动
        # r_adv = epsilon * g / ||g||_2
        grad = embedding_layer.weight.grad
        if grad is not None:
            norm = torch.norm(grad)
            if norm != 0:
                r_adv = self.adv_epsilon * grad / norm
                embedding_layer.weight.data = original_embedding + r_adv
        
        # 对抗样本前向传播
        with self.compute_loss_context_manager():
            adv_loss = self.compute_loss(model, inputs)
        
        if self.args.gradient_accumulation_steps > 1:
            adv_loss = adv_loss / self.args.gradient_accumulation_steps
        
        # 反向传播对抗损失
        self.accelerator.backward(adv_loss)
        
        # 恢复原始 embedding
        embedding_layer.weight.data = original_embedding
        
        # 总损失（用于日志）
        return (loss + adv_loss).detach()
```

### 25.2.3 重写 evaluation_loop()

完全自定义评估流程。

**示例：Top-K 准确率**
```python
class TopKTrainer(Trainer):
    def __init__(self, top_k=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_k = top_k
    
    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval"
    ):
        """
        自定义评估循环，计算 Top-K 准确率
        """
        model = self.model
        model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0.0
        
        for step, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            labels = inputs.pop("labels")
            
            with torch.no_grad():
                outputs = model(**inputs)
                loss = outputs.loss
                logits = outputs.logits
            
            # 收集预测和标签
            all_preds.append(logits.cpu())
            all_labels.append(labels.cpu())
            total_loss += loss.item()
        
        # 合并所有 batch
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # 计算 Top-K 准确率
        _, top_k_preds = torch.topk(all_preds, self.top_k, dim=1)
        correct = (top_k_preds == all_labels.unsqueeze(1)).any(dim=1).float()
        top_k_accuracy = correct.mean().item()
        
        # 计算 Top-1 准确率（标准准确率）
        top_1_preds = all_preds.argmax(dim=1)
        top_1_accuracy = (top_1_preds == all_labels).float().mean().item()
        
        metrics = {
            f"{metric_key_prefix}_loss": total_loss / len(dataloader),
            f"{metric_key_prefix}_top1_accuracy": top_1_accuracy,
            f"{metric_key_prefix}_top{self.top_k}_accuracy": top_k_accuracy
        }
        
        return EvalLoopOutput(
            predictions=all_preds.numpy(),
            label_ids=all_labels.numpy(),
            metrics=metrics,
            num_samples=len(all_labels)
        )
```

### 25.2.4 示例：对比学习 Trainer

实现 Contrastive Learning（如 SimCLR）的 Trainer：

```python
import torch
import torch.nn.functional as F

class ContrastiveLearningTrainer(Trainer):
    def __init__(self, temperature=0.07, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        对比学习损失（NT-Xent Loss）
        
        输入：
        - inputs: {"input_ids_1": ..., "input_ids_2": ...}（同一样本的两种增强）
        
        损失：
        L = -log( exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ) )
        """
        # 获取两个增强视图
        input_ids_1 = inputs.pop("input_ids_1")
        input_ids_2 = inputs.pop("input_ids_2")
        attention_mask_1 = inputs.pop("attention_mask_1", None)
        attention_mask_2 = inputs.pop("attention_mask_2", None)
        
        # 前向传播获取 embeddings
        outputs_1 = model(
            input_ids=input_ids_1,
            attention_mask=attention_mask_1
        )
        outputs_2 = model(
            input_ids=input_ids_2,
            attention_mask=attention_mask_2
        )
        
        # 提取 [CLS] 表示并归一化
        z1 = F.normalize(outputs_1.pooler_output, dim=1)  # (batch, hidden)
        z2 = F.normalize(outputs_2.pooler_output, dim=1)
        
        batch_size = z1.size(0)
        
        # 计算相似度矩阵
        # 拼接正负样本：[z1, z2] → (2*batch, hidden)
        embeddings = torch.cat([z1, z2], dim=0)
        
        # 计算 cosine similarity：(2*batch, 2*batch)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # 创建 mask（排除自身）
        mask = torch.eye(2 * batch_size, device=z1.device).bool()
        similarity_matrix.masked_fill_(mask, -1e9)
        
        # 正样本对的索引
        # z1[i] 的正样本是 z2[i]，索引为 i + batch_size
        positive_indices = torch.arange(batch_size, device=z1.device)
        
        # 计算损失（分两部分：z1→z2 和 z2→z1）
        # Part 1: z1 作为 anchor
        logits_1 = similarity_matrix[:batch_size]  # (batch, 2*batch)
        labels_1 = positive_indices + batch_size
        loss_1 = F.cross_entropy(logits_1, labels_1)
        
        # Part 2: z2 作为 anchor
        logits_2 = similarity_matrix[batch_size:]
        labels_2 = positive_indices
        loss_2 = F.cross_entropy(logits_2, labels_2)
        
        # 总损失
        loss = (loss_1 + loss_2) / 2
        
        return (loss, outputs_1) if return_outputs else loss
```

**使用**：
```python
# 数据集需要返回两个增强视图
class ContrastiveDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # 两种增强策略（示例：随机 dropout）
        encoding_1 = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        encoding_2 = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids_1": encoding_1["input_ids"].squeeze(0),
            "attention_mask_1": encoding_1["attention_mask"].squeeze(0),
            "input_ids_2": encoding_2["input_ids"].squeeze(0),
            "attention_mask_2": encoding_2["attention_mask"].squeeze(0)
        }
    
    def __len__(self):
        return len(self.texts)

# 训练
trainer = ContrastiveLearningTrainer(
    temperature=0.07,
    model=model,
    args=training_args,
    train_dataset=contrastive_dataset
)
trainer.train()
```

---

## 25.3 自定义 Callback

`TrainerCallback` 允许在训练的关键节点插入自定义逻辑。

### 25.3.1 TrainerCallback 基类

```python
from transformers import TrainerCallback, TrainerState, TrainerControl

class MyCallback(TrainerCallback):
    """
    自定义回调基类
    """
    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时调用"""
        pass
    
    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时调用"""
        pass
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """每个 epoch 开始时调用"""
        pass
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """每个 epoch 结束时调用"""
        pass
    
    def on_step_begin(self, args, state, control, **kwargs):
        """每个训练步骤开始时调用"""
        pass
    
    def on_step_end(self, args, state, control, **kwargs):
        """每个训练步骤结束时调用"""
        pass
    
    def on_evaluate(self, args, state, control, **kwargs):
        """评估时调用"""
        pass
    
    def on_save(self, args, state, control, **kwargs):
        """保存检查点时调用"""
        pass
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """记录日志时调用"""
        pass
```

**关键参数**：
- **`args`**：`TrainingArguments` 对象
- **`state`**：`TrainerState` 对象（包含 global_step, epoch, best_metric 等）
- **`control`**：`TrainerControl` 对象（可以控制训练流程）
- **`kwargs`**：额外参数（如 model, optimizer, logs）

### 25.3.2 事件触发点

完整的回调执行顺序：

```
on_train_begin
├─ on_epoch_begin (epoch 1)
│  ├─ on_step_begin (step 1)
│  ├─ on_step_end
│  ├─ on_log (if logging_steps)
│  ├─ on_evaluate (if evaluation_strategy)
│  ├─ on_save (if save_steps)
│  ├─ ... (more steps)
│  └─ on_epoch_end
├─ on_epoch_begin (epoch 2)
│  └─ ...
└─ on_train_end
```

### 25.3.3 示例：自定义学习率预热

实现 Warmup + Linear Decay（虽然内置，但作为示例）：

```python
class WarmupCallback(TrainerCallback):
    def __init__(self, warmup_steps=1000, total_steps=10000):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        # 保存初始学习率
        optimizer = kwargs.get("optimizer")
        self.base_lr = optimizer.param_groups[0]["lr"]
        print(f"✅ Warmup Callback initialized: {self.warmup_steps} steps warmup")
    
    def on_step_end(self, args, state, control, **kwargs):
        optimizer = kwargs.get("optimizer")
        current_step = state.global_step
        
        if current_step < self.warmup_steps:
            # Warmup 阶段：线性增长
            lr = self.base_lr * (current_step / self.warmup_steps)
        else:
            # Decay 阶段：线性衰减
            progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.base_lr * (1 - progress)
        
        # 更新学习率
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

# 使用
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[WarmupCallback(warmup_steps=500, total_steps=5000)]
)
```

**更多示例**：

**1. 早停（Early Stopping）**
```python
class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=3, threshold=0.001):
        self.patience = patience
        self.threshold = threshold
        self.best_metric = None
        self.wait = 0
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_metric = metrics.get("eval_loss")
        
        if self.best_metric is None or current_metric < self.best_metric - self.threshold:
            self.best_metric = current_metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"🛑 Early stopping triggered! Best metric: {self.best_metric}")
                control.should_training_stop = True  # 停止训练
        
        return control
```

**2. 梯度监控**
```python
class GradientMonitorCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        if state.global_step % args.logging_steps == 0:
            print(f"Step {state.global_step}: Gradient norm = {total_norm:.4f}")
            
            if total_norm > 10.0:
                print("⚠️  Warning: Gradient explosion detected!")
```

**3. 模型检查点版本管理**
```python
import shutil

class VersionedCheckpointCallback(TrainerCallback):
    def __init__(self, keep_last_n=3):
        self.keep_last_n = keep_last_n
        self.checkpoints = []
    
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = f"{args.output_dir}/checkpoint-{state.global_step}"
        self.checkpoints.append(checkpoint_dir)
        
        # 只保留最后 N 个检查点
        if len(self.checkpoints) > self.keep_last_n:
            old_checkpoint = self.checkpoints.pop(0)
            shutil.rmtree(old_checkpoint)
            print(f"🗑️  Removed old checkpoint: {old_checkpoint}")
```

---

## 25.4 完全自定义训练循环

有时 `Trainer` 的灵活性不够，需要完全自定义训练循环。使用 **Accelerate** 可以轻松实现。

### 25.4.1 使用 Accelerate 替代 Trainer

基础训练循环：

```python
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import torch

# 1. 初始化 Accelerator
accelerator = Accelerator()

# 2. 准备模型、优化器、数据
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=64)

# 3. 使用 Accelerator 包装
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

# 4. 学习率调度器
num_training_steps = len(train_dataloader) * num_epochs
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=num_training_steps
)

# 5. 训练循环
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        # 前向传播
        outputs = model(**batch)
        loss = outputs.loss
        
        # 反向传播（Accelerator 自动处理混合精度）
        accelerator.backward(loss)
        
        # 更新参数
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        # 打印日志
        if accelerator.is_main_process:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # 评估
    model.eval()
    total_correct = 0
    total_samples = 0
    
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            
            # 收集所有 GPU 的结果
            predictions, labels = accelerator.gather_for_metrics((predictions, batch["labels"]))
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = total_correct / total_samples
    if accelerator.is_main_process:
        print(f"Epoch {epoch} - Accuracy: {accuracy:.4f}")

# 6. 保存模型
accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained("./my_model", save_function=accelerator.save)
```

**关键 API**：
- **`accelerator.prepare()`**：自动处理设备放置、分布式包装
- **`accelerator.backward()`**：自动处理混合精度的梯度缩放
- **`accelerator.gather_for_metrics()`**：从所有设备收集结果
- **`accelerator.is_main_process`**：判断是否为主进程（用于日志）

### 25.4.2 手动实现梯度累积

```python
gradient_accumulation_steps = 4

for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(train_dataloader):
        # 前向传播
        outputs = model(**batch)
        loss = outputs.loss
        
        # 梯度累积：loss 需要除以累积步数
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        
        # 每 N 步更新一次参数
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
```

### 25.4.3 混合精度集成

Accelerate 自动处理混合精度，只需在初始化时指定：

```python
# 方式 1：命令行启动
# accelerate launch --mixed_precision fp16 train.py

# 方式 2：代码中指定
accelerator = Accelerator(mixed_precision="fp16")

# 其他代码不变，Accelerator 会自动：
# 1. 模型转换为 fp16
# 2. 梯度缩放（gradient scaling）
# 3. 动态损失缩放
```

### 25.4.4 分布式训练适配

使用 Accelerate 的分布式训练：

```bash
# 单机多卡
accelerate launch --multi_gpu --num_processes 4 train.py

# 多机多卡
accelerate launch \
    --multi_gpu \
    --num_machines 2 \
    --machine_rank 0 \
    --main_process_ip xxx.xxx.xxx.xxx \
    --num_processes 8 \
    train.py
```

**代码无需修改**，Accelerate 自动处理：
- 进程初始化
- 梯度同步
- 数据分片

---

## 25.5 高级损失函数

<div data-component="LossFunctionExplorer"></div>

### 25.5.1 Focal Loss

用于解决类别不平衡问题（提出于 RetinaNet）。

**公式**：
$$
\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

其中：
- $p_t$ 是真实类别的预测概率
- $\gamma$ 是聚焦参数（通常为 2）
- $\alpha_t$ 是类别权重

**实现**：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss for imbalanced classification
        
        Args:
            alpha (float): 类别权重
            gamma (float): 聚焦参数，越大越关注难分样本
            reduction (str): 'mean' or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (batch, num_classes) logits
            targets: (batch,) class indices
        """
        # 计算概率
        p = F.softmax(inputs, dim=1)
        
        # 获取真实类别的概率
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Focal Loss
        focal_loss = self.alpha * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# 在 Trainer 中使用
class FocalLossTrainer(Trainer):
    def __init__(self, alpha=0.25, gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss = self.focal_loss(logits, labels)
        
        return (loss, outputs) if return_outputs else loss
```

**效果**：
- **困难样本**（$p_t$ 低）：损失权重大，模型更关注
- **简单样本**（$p_t$ 高）：损失权重小（$(1-p_t)^\gamma$ 接近 0）
- **类别不平衡**：$\alpha$ 调整正负样本权重

### 25.5.2 Contrastive Loss

用于对比学习（SimCLR、MoCo、CLIP）。

**InfoNCE Loss**（Noise Contrastive Estimation）：
$$
\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)}
$$

**实现**（见 25.2.4 对比学习 Trainer）。

**关键点**：
- **正样本对**：同一样本的不同增强
- **负样本对**：batch 内其他样本
- **温度参数** $\tau$：控制分布平滑度（通常 0.07）

### 25.5.3 KL Divergence（知识蒸馏）

用于模型蒸馏（DistilBERT、TinyBERT）。

**蒸馏损失**：
$$
\mathcal{L}_{\text{distill}} = \text{KL}(\text{softmax}(z_s / T) \| \text{softmax}(z_t / T))
$$

其中：
- $z_s$ 是学生模型的 logits
- $z_t$ 是教师模型的 logits
- $T$ 是温度参数（软化分布）

**实现**：
```python
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature=2.0, alpha=0.5, *args, **kwargs):
        """
        知识蒸馏 Trainer
        
        Args:
            teacher_model: 教师模型（已训练好）
            temperature: 温度参数（软化分布）
            alpha: 蒸馏损失权重（总损失 = alpha * distill_loss + (1-alpha) * ce_loss）
        """
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()  # 教师模型始终在评估模式
        self.temperature = temperature
        self.alpha = alpha
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        
        # 学生模型前向传播
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        
        # 教师模型前向传播（不计算梯度）
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        
        # 1. Hard Loss（标准交叉熵）
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # 2. Soft Loss（KL 散度）
        # 使用温度软化分布
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL Divergence: D_KL(P || Q) = Σ P(x) log(P(x) / Q(x))
        soft_loss = F.kl_div(
            student_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)  # 缩放因子（温度平方）
        
        # 3. 总损失
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return (loss, student_outputs) if return_outputs else loss


# 使用
teacher_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
student_model = BertForSequenceClassification.from_pretrained("bert-tiny-uncased")

trainer = DistillationTrainer(
    teacher_model=teacher_model,
    temperature=2.0,
    alpha=0.7,
    model=student_model,
    args=training_args,
    train_dataset=train_dataset
)
trainer.train()
```

**为什么使用温度**？
- **低温（T=1）**：分布尖锐，接近 one-hot
- **高温（T>1）**：分布平滑，包含更多类间关系信息
- **$T^2$ 缩放**：抵消温度对梯度幅度的影响

### 25.5.4 多任务学习损失组合

同时训练多个任务（如情感分类 + NER）。

```python
class MultiTaskTrainer(Trainer):
    def __init__(self, task_weights=None, *args, **kwargs):
        """
        多任务学习 Trainer
        
        Args:
            task_weights: dict，例如 {"classification": 1.0, "ner": 0.5}
        """
        super().__init__(*args, **kwargs)
        self.task_weights = task_weights or {"classification": 1.0, "ner": 1.0}
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        假设模型输出多个 loss：
        outputs = {
            "classification_loss": ...,
            "ner_loss": ...,
            "logits": ...
        }
        """
        outputs = model(**inputs)
        
        # 加权组合多个损失
        total_loss = 0.0
        for task, weight in self.task_weights.items():
            task_loss = outputs.get(f"{task}_loss")
            if task_loss is not None:
                total_loss += weight * task_loss
        
        # 返回总损失和原始输出
        if return_outputs:
            outputs["loss"] = total_loss
            return total_loss, outputs
        else:
            return total_loss
```

**动态任务权重**（Uncertainty Weighting）：
```python
class UncertaintyWeightedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 可学习的任务权重（log(σ²)）
        self.log_vars = nn.Parameter(torch.zeros(2))  # 2 个任务
    
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        
        loss1 = outputs["classification_loss"]
        loss2 = outputs["ner_loss"]
        
        # Uncertainty weighting:
        # L_total = (1 / 2σ₁²) L₁ + (1 / 2σ₂²) L₂ + log(σ₁σ₂)
        precision1 = torch.exp(-self.log_vars[0])
        precision2 = torch.exp(-self.log_vars[1])
        
        total_loss = (
            precision1 * loss1 +
            precision2 * loss2 +
            self.log_vars[0] + self.log_vars[1]  # 正则化项
        )
        
        return (total_loss, outputs) if return_outputs else total_loss
```

---

## 25.6 实战案例：情感分析自定义训练

结合所有技术，实现一个完整的自定义训练流程。

```python
import torch
import torch.nn as nn
from transformers import (
    Trainer,
    TrainingArguments,
    BertForSequenceClassification,
    BertTokenizer,
    TrainerCallback
)
from datasets import load_dataset
import numpy as np

# 1. 自定义 Focal Loss Trainer
class SentimentTrainer(Trainer):
    def __init__(self, focal_gamma=2.0, label_smoothing=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Focal Loss + Label Smoothing
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        # 获取真实类别概率
        true_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        
        # Focal weight
        focal_weight = (1 - true_probs) ** self.focal_gamma
        
        # Label smoothing
        num_classes = logits.size(-1)
        smooth_labels = torch.zeros_like(probs).scatter_(
            1, labels.unsqueeze(1), 1 - self.label_smoothing
        ) + self.label_smoothing / num_classes
        
        # 损失
        loss = -(focal_weight.unsqueeze(1) * smooth_labels * log_probs).sum(dim=-1).mean()
        
        return (loss, outputs) if return_outputs else loss

# 2. 自定义 Callback（梯度监控 + 早停）
class MonitorCallback(TrainerCallback):
    def __init__(self, patience=3):
        self.patience = patience
        self.best_loss = float('inf')
        self.wait = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        # 梯度监控
        if state.global_step % 100 == 0:
            model = kwargs["model"]
            total_norm = sum(
                p.grad.norm(2).item() ** 2 
                for p in model.parameters() if p.grad is not None
            ) ** 0.5
            print(f"📊 Step {state.global_step}: Gradient norm = {total_norm:.4f}")
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # 早停
        current_loss = metrics.get("eval_loss")
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"🛑 Early stopping! Best loss: {self.best_loss:.4f}")
                control.should_training_stop = True

# 3. 准备数据
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = load_dataset("imdb")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 4. 训练参数
training_args = TrainingArguments(
    output_dir="./sentiment_model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True  # 混合精度
)

# 5. 模型和 Trainer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

trainer = SentimentTrainer(
    focal_gamma=2.0,
    label_smoothing=0.1,
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    callbacks=[MonitorCallback(patience=3)]
)

# 6. 训练
trainer.train()

# 7. 评估
results = trainer.evaluate()
print(f"✅ Final results: {results}")
```

---

## 25.7 章节总结

本章我们深入学习了 Trainer 的高级定制技术：

✅ **核心技能**：
- 理解 `Trainer` 内部训练循环（DataLoader → Forward → Backward → Optimizer Step）
- 重写 `compute_loss()`、`training_step()`、`evaluation_loop()`
- 实现自定义 Callback（早停、梯度监控、学习率调度）
- 使用 Accelerate 完全自定义训练循环
- 实现高级损失函数（Focal Loss、Contrastive Loss、KL Divergence）

✅ **实战能力**：
- 对抗训练（FGM）
- 对比学习（SimCLR）
- 知识蒸馏（KL Divergence）
- 多任务学习（动态任务权重）

✅ **最佳实践**：
- 梯度累积：`loss = loss / gradient_accumulation_steps`
- 混合精度：使用 `Accelerator(mixed_precision="fp16")`
- 分布式训练：`accelerate launch --multi_gpu`
- 早停：`control.should_training_stop = True`

**下一章预告**：Chapter 26 将进入**多模态模型**领域，学习 Vision-Language 模型（CLIP、BLIP、LLaVA）、图像编码器（ViT）、视觉问答微调、图像生成（Stable Diffusion）以及音频模型（Whisper、Wav2Vec2）。

# Chapter 4: Datasets 库与数据预处理

> **本章目标**：全面掌握 Hugging Face Datasets 库的核心功能，学习高效的数据加载、预处理和管理技巧，为模型训练做好数据准备。

---

## 4.1 Datasets 库基础

### 4.1.1 为什么需要 Datasets？

在深度学习训练中，数据处理往往占据大量时间和内存。Hugging Face Datasets 库通过以下创新解决了传统痛点：

**核心优势**：

1. **内存映射（Memory Mapping）**
   - 数据存储在磁盘上，按需加载到内存
   - 处理 TB 级数据集时内存占用极小
   - 基于 Apache Arrow 高性能列式存储

2. **零拷贝读取（Zero-Copy Reads）**
   - 避免数据在内存中重复复制
   - 显著提升数据加载速度

3. **智能缓存**
   - 自动缓存预处理结果（如 tokenization）
   - 避免重复计算

4. **互操作性**
   - 无缝转换为 PyTorch/TensorFlow 张量
   - 与 Pandas、NumPy 互通

**传统方式 vs Datasets 库对比**：

```python
# ❌ 传统方式：全部加载到内存
import pandas as pd

# 占用大量内存（假设 10GB 数据）
df = pd.read_csv("large_dataset.csv")  # 内存占用 ~10GB
texts = df["text"].tolist()

# 🟢 Datasets 库：内存映射
from datasets import load_dataset

# 内存占用 ~几百 MB（数据存在磁盘）
dataset = load_dataset("csv", data_files="large_dataset.csv")
print(f"数据集大小: {dataset.num_rows} 行")
print(f"内存占用: ~0.5GB")  # 仅索引和元数据
```

**性能对比实验**：

```python
import time
import psutil
import os

def measure_memory():
    """测量当前进程内存占用"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# 测试数据集：IMDB 电影评论（~500MB）
dataset_name = "imdb"

# 方式1: Datasets 库
start_mem = measure_memory()
start_time = time.time()

dataset = load_dataset(dataset_name, split="train")
print(f"Datasets 库:")
print(f"  加载时间: {time.time() - start_time:.2f}s")
print(f"  内存增量: {measure_memory() - start_mem:.2f}MB")
print(f"  数据集大小: {len(dataset)} 条\n")

# 输出示例:
# Datasets 库:
#   加载时间: 2.3s
#   内存增量: 45MB
#   数据集大小: 25000 条
```

### 4.1.2 加载数据集（load_dataset）

**基本用法**：

```python
from datasets import load_dataset

# 方式1: 从 Hub 加载（最常用）
dataset = load_dataset("glue", "mrpc")  # GLUE 基准的 MRPC 任务

# 方式2: 指定数据集配置
dataset = load_dataset(
    "glue",
    "mrpc",
    split="train",           # 只加载训练集
    cache_dir="./my_cache"   # 自定义缓存目录
)

# 方式3: 从本地文件加载
dataset = load_dataset("csv", data_files="my_data.csv")

# 方式4: 从多个文件加载
dataset = load_dataset(
    "json",
    data_files={
        "train": ["train1.json", "train2.json"],
        "test": "test.json"
    }
)

# 方式5: 流式加载（大数据集）
dataset = load_dataset("c4", "en", split="train", streaming=True)
```

**查看数据集结构**：

```python
dataset = load_dataset("imdb", split="train")

print("=== 数据集信息 ===")
print(dataset)
# 输出:
# Dataset({
#     features: ['text', 'label'],
#     num_rows: 25000
# })

print("\n=== 数据集特征 ===")
print(dataset.features)
# 输出:
# {'text': Value(dtype='string', id=None),
#  'label': ClassLabel(names=['neg', 'pos'], id=None)}

print("\n=== 第一条样本 ===")
print(dataset[0])
# 输出:
# {'text': 'Bromwell High is a cartoon comedy...', 'label': 1}

print("\n=== 前3条样本 ===")
print(dataset[:3])
```

### 4.1.3 Hub 数据集浏览

**在 Hub 上搜索数据集**：

```python
from datasets import list_datasets

# 列出所有可用数据集（10000+ 个）
all_datasets = list_datasets()
print(f"总数据集数量: {len(all_datasets)}")

# 搜索特定任务的数据集
sentiment_datasets = [d for d in all_datasets if 'sentiment' in d.lower()]
print(f"情感分析数据集: {sentiment_datasets[:5]}")

# 输出示例:
# ['amazon_polarity', 'imdb', 'yelp_review_full', 'sst2', 'tweet_eval']
```

**查看数据集元数据**：

```python
from datasets import load_dataset_builder

# 获取数据集信息（不下载数据）
builder = load_dataset_builder("squad")

print(f"描述: {builder.info.description[:100]}...")
print(f"引用: {builder.info.citation[:100]}...")
print(f"主页: {builder.info.homepage}")
print(f"许可证: {builder.info.license}")
print(f"数据集大小: {builder.info.dataset_size / 1e6:.2f}MB")
print(f"下载大小: {builder.info.download_size / 1e6:.2f}MB")
```

---

## 4.2 数据集操作

<div data-component="DatasetPipeline"></div>

### 4.2.1 map()：批量转换

`map()` 是 Datasets 库最强大的方法，用于对数据集的每个样本应用函数。

**基础用法**：

```python
from datasets import load_dataset

dataset = load_dataset("imdb", split="train[:1000]")  # 只加载1000条

# 定义转换函数
def add_length(example):
    """添加文本长度字段"""
    example["length"] = len(example["text"])
    return example

# 应用转换
dataset = dataset.map(add_length)

print(dataset[0])
# 输出: {'text': '...', 'label': 1, 'length': 1234}
```

**批量处理（推荐）**：

```python
# 批量处理比逐条处理快 10-100 倍
def add_length_batch(examples):
    """批量处理版本"""
    examples["length"] = [len(text) for text in examples["text"]]
    return examples

dataset = dataset.map(
    add_length_batch,
    batched=True,        # 启用批量处理
    batch_size=1000      # 每批 1000 个样本
)

# 性能对比
import time

# 逐条处理
start = time.time()
dataset.map(add_length, batched=False)
time_single = time.time() - start

# 批量处理
start = time.time()
dataset.map(add_length_batch, batched=True, batch_size=1000)
time_batch = time.time() - start

print(f"逐条处理: {time_single:.2f}s")
print(f"批量处理: {time_batch:.2f}s")
print(f"加速比: {time_single / time_batch:.2f}x")

# 典型输出:
# 逐条处理: 12.5s
# 批量处理: 0.8s
# 加速比: 15.6x
```

**多进程加速**：

```python
# 使用多核 CPU 并行处理
dataset = dataset.map(
    add_length_batch,
    batched=True,
    num_proc=4  # 使用 4 个进程
)

# 自动检测 CPU 核心数
import os
num_cores = os.cpu_count()
dataset = dataset.map(
    add_length_batch,
    batched=True,
    num_proc=num_cores
)
```

**Tokenization 集成**：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    """批量 tokenization"""
    return tokenizer(
        examples["text"],
        padding=False,       # 不在这里 padding（留给 DataCollator）
        truncation=True,
        max_length=512
    )

# 应用 tokenization
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=["text"]  # 移除原始文本（节省内存）
)

print(tokenized_dataset[0])
# 输出: {'input_ids': [101, 2023, ...], 'attention_mask': [1, 1, ...], 'label': 1}
```

**进度条与缓存**：

```python
dataset = dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing",           # 进度条描述
    load_from_cache_file=True,   # 使用缓存（默认）
    cache_file_name="tokenized_cache.arrow"  # 自定义缓存文件名
)

# 第二次运行会直接从缓存加载，速度极快
```

### 4.2.2 filter()：条件筛选

**基础过滤**：

```python
# 只保留长文本（>100 字符）
filtered_dataset = dataset.filter(lambda x: len(x["text"]) > 100)

print(f"原始大小: {len(dataset)}")
print(f"过滤后: {len(filtered_dataset)}")

# 输出:
# 原始大小: 1000
# 过滤后: 856
```

**批量过滤（更快）**：

```python
def filter_long_texts(examples):
    """批量过滤"""
    return [len(text) > 100 for text in examples["text"]]

filtered_dataset = dataset.filter(
    filter_long_texts,
    batched=True,
    batch_size=1000
)
```

**复杂条件**：

```python
# 过滤：正面评论 且 长度在 100-500 之间
def complex_filter(example):
    return (
        example["label"] == 1 and           # 正面评论
        100 < len(example["text"]) < 500    # 长度限制
    )

dataset = dataset.filter(complex_filter)
```

### 4.2.3 select()、shuffle()、train_test_split()

**select() - 选择特定索引**：

```python
# 选择前100条
subset = dataset.select(range(100))

# 选择特定索引
indices = [0, 10, 20, 30, 40]
subset = dataset.select(indices)

# 随机采样（使用 shuffle + select）
import random
indices = random.sample(range(len(dataset)), k=100)
subset = dataset.select(indices)
```

**shuffle() - 随机打乱**：

```python
# 完全打乱
shuffled_dataset = dataset.shuffle(seed=42)

# 部分打乱（只打乱前1000条）
shuffled_dataset = dataset.shuffle(seed=42).select(range(1000))
```

**train_test_split() - 划分数据集**：

```python
# 按比例划分
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

print(split_dataset)
# 输出:
# DatasetDict({
#     train: Dataset({features: [...], num_rows: 800})
#     test: Dataset({features: [...], num_rows: 200})
# })

train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

# 三分法（训练/验证/测试）
train_test = dataset.train_test_split(test_size=0.3, seed=42)
test_valid = train_test["test"].train_test_split(test_size=0.5, seed=42)

final_dataset = {
    "train": train_test["train"],       # 70%
    "validation": test_valid["train"],  # 15%
    "test": test_valid["test"]          # 15%
}
```

### 4.2.4 数据集拼接与交织

**concatenate_datasets - 垂直拼接**：

```python
from datasets import concatenate_datasets

dataset1 = load_dataset("imdb", split="train[:1000]")
dataset2 = load_dataset("imdb", split="test[:1000]")

# 拼接
combined = concatenate_datasets([dataset1, dataset2])
print(f"合并后大小: {len(combined)}")  # 2000
```

**interleave_datasets - 交织混合**：

```python
from datasets import interleave_datasets

# 从多个数据集交替采样
dataset1 = load_dataset("imdb", split="train", streaming=True)
dataset2 = load_dataset("yelp_review_full", split="train", streaming=True)

# 交织（1:1 比例）
interleaved = interleave_datasets([dataset1, dataset2])

# 自定义采样概率（70% 来自 dataset1，30% 来自 dataset2）
interleaved = interleave_datasets(
    [dataset1, dataset2],
    probabilities=[0.7, 0.3],
    seed=42
)
```

---

## 4.3 Tokenization 集成

### 4.3.1 使用 map() 批量 tokenize

**标准流程**：

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# 加载数据集和 tokenizer
dataset = load_dataset("glue", "sst2", split="train")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 定义 tokenization 函数
def tokenize_function(examples):
    return tokenizer(
        examples["sentence"],
        padding="max_length",  # 固定长度 padding
        truncation=True,
        max_length=128
    )

# 批量处理
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    num_proc=4
)

print(tokenized_dataset.column_names)
# 输出: ['sentence', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask']
```

### 4.3.2 remove_columns() 清理原始字段

**移除不需要的列**：

```python
# 只保留模型需要的字段
tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "idx"])

print(tokenized_dataset.column_names)
# 输出: ['label', 'input_ids', 'token_type_ids', 'attention_mask']

# 或者在 map() 时直接移除
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names  # 移除所有原始列
)
```

**重命名列**：

```python
# 将 'label' 重命名为 'labels'（Trainer 期望的名称）
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
```

### 4.3.3 set_format()：PyTorch/TensorFlow 格式

**转换为张量格式**：

```python
# 方式1: set_format() - 临时转换
tokenized_dataset.set_format(
    type="torch",  # 'torch', 'tensorflow', 'numpy', 'pandas'
    columns=["input_ids", "attention_mask", "labels"]
)

# 现在访问数据时自动返回 torch.Tensor
print(type(tokenized_dataset[0]["input_ids"]))
# 输出: <class 'torch.Tensor'>

# 方式2: with_format() - 返回新对象
torch_dataset = tokenized_dataset.with_format("torch")

# 重置格式
tokenized_dataset.reset_format()
```

**实战示例：准备 DataLoader**：

```python
from torch.utils.data import DataLoader

# 设置格式
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 创建 DataLoader
dataloader = DataLoader(tokenized_dataset, batch_size=16, shuffle=True)

# 使用
for batch in dataloader:
    print(batch.keys())  # dict_keys(['input_ids', 'attention_mask', 'labels'])
    print(batch["input_ids"].shape)  # torch.Size([16, 128])
    break
```

---

## 4.4 DataCollator 家族

DataCollator 是数据批处理的核心组件，负责将多个样本整理成模型可接受的批次格式。

<div data-component="DataCollatorDemo"></div>

### 4.4.1 DataCollatorWithPadding：动态 padding

**为什么需要动态 padding？**

```python
# 问题：序列长度不一致
samples = [
    {"input_ids": [101, 2023, 2003, 102]},           # 长度 4
    {"input_ids": [101, 7592, 2088, 999, 102]},      # 长度 5
    {"input_ids": [101, 1045, 2293, 2023, 3185, 102]} # 长度 6
]

# ❌ 无法直接堆叠成张量
import torch
try:
    torch.tensor([s["input_ids"] for s in samples])
except ValueError as e:
    print(f"错误: {e}")
# 输出: 错误: expected sequence of equal length tensors

# ✅ 使用 DataCollator 自动 padding
from transformers import DataCollatorWithPadding

collator = DataCollatorWithPadding(tokenizer=tokenizer)
batch = collator(samples)

print(batch["input_ids"])
# 输出: tensor([
#     [101, 2023, 2003,  102,    0,    0],  # padding 到最长
#     [101, 7592, 2088,  999,  102,    0],
#     [101, 1045, 2293, 2023, 3185,  102]
# ])

print(batch["attention_mask"])
# 输出: tensor([
#     [1, 1, 1, 1, 0, 0],  # 0 表示 padding 位置
#     [1, 1, 1, 1, 1, 0],
#     [1, 1, 1, 1, 1, 1]
# ])
```

**完整训练流程**：

```python
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
from torch.utils.data import DataLoader

# 1. Tokenization（不做 padding）
def tokenize_function(examples):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        padding=False  # 重要：不在这里 padding
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 2. 创建 DataCollator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 3. 手动使用（PyTorch DataLoader）
dataloader = DataLoader(
    tokenized_dataset,
    batch_size=8,
    collate_fn=data_collator  # 关键参数
)

for batch in dataloader:
    print(f"Batch shape: {batch['input_ids'].shape}")
    # 每个批次的长度不同（动态调整）
    break

# 4. 或直接传给 Trainer（推荐）
training_args = TrainingArguments(output_dir="./results")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator  # Trainer 自动使用
)
```

**性能对比**：

```python
# 固定 padding 到 512（浪费计算）
def tokenize_fixed(examples):
    return tokenizer(examples["sentence"], padding="max_length", max_length=512, truncation=True)

# 动态 padding（仅 padding 到批次内最长）
def tokenize_dynamic(examples):
    return tokenizer(examples["sentence"], truncation=True, padding=False)

# 假设平均长度 50，批次大小 32
# 固定 padding: 32 * 512 = 16,384 tokens/batch
# 动态 padding: 32 * ~60 = ~1,920 tokens/batch
# 计算节省: ~88%
```

### 4.4.2 DataCollatorForLanguageModeling：MLM 掩码

用于训练 BERT 类掩码语言模型（Masked Language Modeling）。

**工作原理**：

```python
from transformers import DataCollatorForLanguageModeling

# 创建 MLM collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,           # 启用 MLM
    mlm_probability=0.15  # 掩码 15% 的 token
)

# 示例数据
texts = ["Hello world", "Machine learning is amazing"]
tokenized = tokenizer(texts, return_tensors="pt", padding=True)

# 应用掩码
batch = data_collator([
    {k: v[i] for k, v in tokenized.items()} for i in range(len(texts))
])

print("原始 input_ids:")
print(tokenized["input_ids"])

print("\n掩码后 input_ids (部分被替换为 [MASK]):")
print(batch["input_ids"])

print("\nlabels (用于计算损失):")
print(batch["labels"])
# -100 表示不计算损失的位置（未被掩码的 token）
```

**掩码策略详解**：

```python
# BERT 的掩码策略：
# 选中的 15% token 中：
#   - 80% 替换为 [MASK]
#   - 10% 替换为随机 token
#   - 10% 保持不变

# 示例
original_text = "The quick brown fox jumps"
# 假设 'quick' 被选中掩码
# 可能结果：
# - "The [MASK] brown fox jumps"  (80% 概率)
# - "The dog brown fox jumps"     (10% 概率，随机词)
# - "The quick brown fox jumps"   (10% 概率，保持不变)
```

**从头训练 BERT 示例**：

```python
from transformers import BertForMaskedLM, Trainer, TrainingArguments

# 加载未训练的模型
model = BertForMaskedLM(config=config)

# 准备数据
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

def tokenize_function(examples):
    return tokenizer(examples["text"], return_special_tokens_mask=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# MLM DataCollator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm_probability=0.15
)

# 训练
training_args = TrainingArguments(
    output_dir="./bert-mlm",
    per_device_train_batch_size=8,
    num_train_epochs=3
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()
```

### 4.4.3 DataCollatorForSeq2Seq：Encoder-Decoder 专用

用于序列到序列任务（如翻译、摘要）。

**核心功能**：

1. **Decoder input 自动构建**：将 labels 右移一位作为 decoder_input_ids
2. **Label padding 处理**：用 -100 填充（CrossEntropyLoss 会忽略）
3. **同时处理 encoder 和 decoder 序列**

```python
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# 创建 Seq2Seq collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,  # 用于获取 pad_token_id
    label_pad_token_id=-100  # labels 的 padding 值
)

# 示例数据（翻译任务）
samples = [
    {
        "input_ids": [1, 2, 3, 4, 5],      # 源语言
        "labels": [10, 11, 12]              # 目标语言
    },
    {
        "input_ids": [1, 2, 3],
        "labels": [10, 11, 12, 13, 14]
    }
]

batch = data_collator(samples)

print("Input IDs (encoder):")
print(batch["input_ids"])
# tensor([[1, 2, 3, 4, 5],
#         [1, 2, 3, 0, 0]])  # padding 到相同长度

print("\nLabels (decoder output):")
print(batch["labels"])
# tensor([[ 10,  11,  12, -100, -100],
#         [ 10,  11,  12,   13,   14]])  # -100 表示 padding
```

**完整翻译微调示例**：

```python
from datasets import load_dataset

# 加载翻译数据集
dataset = load_dataset("wmt16", "de-en", split="train[:1000]")

def preprocess_function(examples):
    inputs = [f"translate German to English: {ex}" for ex in examples["de"]]
    targets = examples["en"]
    
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    
    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Seq2Seq DataCollator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)
```

### 4.4.4 自定义 DataCollator

**场景：添加自定义数据增强**

```python
from dataclasses import dataclass
from transformers.data.data_collator import DataCollatorMixin
import torch

@dataclass
class CustomDataCollator(DataCollatorMixin):
    tokenizer: AutoTokenizer
    
    def __call__(self, features):
        # 1. 标准 padding
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt"
        )
        
        # 2. 自定义增强：随机掩码 10% 的 token（数据增强）
        if self.training:
            mask_prob = 0.1
            input_ids = batch["input_ids"]
            probability_matrix = torch.full(input_ids.shape, mask_prob)
            
            # 不掩码特殊 token
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                for val in input_ids.tolist()
            ]
            probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
            
            masked_indices = torch.bernoulli(probability_matrix).bool()
            batch["input_ids"][masked_indices] = self.tokenizer.mask_token_id
        
        # 3. 添加自定义字段
        batch["custom_weight"] = torch.tensor([len(f["input_ids"]) for f in features])
        
        return batch

# 使用
collator = CustomDataCollator(tokenizer=tokenizer)
```

---

## 4.5 流式数据集（Streaming）

### 4.5.1 何时使用流式模式

**适用场景**：

1. **超大数据集**：数百 GB 或 TB 级别
2. **快速实验**：无需下载完整数据集即可开始训练
3. **动态数据**：实时更新的数据流

**传统 vs 流式对比**：

```python
# ❌ 传统模式：需要下载整个数据集（~800GB）
dataset = load_dataset("c4", "en", split="train")  # 等待数小时下载

# ✅ 流式模式：立即开始，按需加载
dataset = load_dataset("c4", "en", split="train", streaming=True)  # 秒级启动
```

### 4.5.2 IterableDataset vs Dataset

**核心差异**：

| 特性 | Dataset（标准） | IterableDataset（流式） |
|------|----------------|------------------------|
| 数据存储 | 完整下载到磁盘 | 按需下载 |
| 随机访问 | 支持 `dataset[i]` | 不支持 |
| 长度查询 | `len(dataset)` | 不支持（未知长度） |
| Shuffle | 支持全局 shuffle | 仅支持缓冲区 shuffle |
| 内存占用 | 固定（索引） | 极小 |

**使用方式**：

```python
from datasets import load_dataset

# 加载流式数据集
dataset = load_dataset("oscar", "unshuffled_deduplicated_en", split="train", streaming=True)

# 迭代访问（类似生成器）
for i, example in enumerate(dataset):
    print(example)
    if i >= 5:  # 只查看前5条
        break

# ❌ 不支持的操作
# print(len(dataset))  # TypeError
# print(dataset[0])    # TypeError
```

**与 Trainer 集成**：

```python
from transformers import Trainer

# 流式数据集可以直接传给 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=streaming_dataset,  # IterableDataset
    data_collator=data_collator
)

# Trainer 会自动处理流式迭代
trainer.train()
```

### 4.5.3 流式数据的 shuffle 与缓冲

**缓冲区 shuffle**：

```python
# 流式数据无法全局 shuffle，只能在缓冲区内 shuffle
shuffled_dataset = dataset.shuffle(
    seed=42,
    buffer_size=10000  # 缓存 10000 个样本进行 shuffle
)

# 工作原理：
# 1. 从数据流中读取 10000 个样本到缓冲区
# 2. 在缓冲区内随机 shuffle
# 3. 逐个返回样本
# 4. 每返回一个，从数据流中补充一个到缓冲区
# 5. 重复步骤 2-4
```

**take() 和 skip()**：

```python
# 只取前 1000 条
subset = dataset.take(1000)

# 跳过前 5000 条
dataset_after_5k = dataset.skip(5000)

# 组合：跳过前 5000，取接下来的 1000
subset = dataset.skip(5000).take(1000)
```

**map() 在流式数据上的应用**：

```python
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

# 流式 map（惰性执行）
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 迭代时才实际执行 tokenization
for example in tokenized_dataset:
    print(example.keys())
    break
```

---

## 4.6 自定义数据集

### 4.6.1 从 CSV/JSON 加载

**CSV 文件**：

```python
# 单个文件
dataset = load_dataset("csv", data_files="my_data.csv")

# 多个文件
dataset = load_dataset("csv", data_files=["file1.csv", "file2.csv"])

# 指定分割
dataset = load_dataset(
    "csv",
    data_files={
        "train": "train.csv",
        "test": "test.csv"
    }
)

# 自定义参数（传递给 pandas.read_csv）
dataset = load_dataset(
    "csv",
    data_files="data.csv",
    delimiter=";",           # 分隔符
    quotechar='"',
    column_names=["text", "label"]  # 自定义列名
)
```

**JSON 文件**：

```python
# JSON Lines 格式（每行一个 JSON 对象）
dataset = load_dataset("json", data_files="data.jsonl")

# 标准 JSON 数组
dataset = load_dataset("json", data_files="data.json", field="data")

# 示例 data.jsonl 内容：
# {"text": "Example 1", "label": 0}
# {"text": "Example 2", "label": 1}
```

### 4.6.2 从 Python 字典创建

**基础创建**：

```python
from datasets import Dataset

# 方式1：从字典创建
data_dict = {
    "text": ["Hello", "World", "Test"],
    "label": [0, 1, 0]
}

dataset = Dataset.from_dict(data_dict)
print(dataset)
# Dataset({
#     features: ['text', 'label'],
#     num_rows: 3
# })

# 方式2：从列表创建
data_list = [
    {"text": "Hello", "label": 0},
    {"text": "World", "label": 1},
    {"text": "Test", "label": 0}
]

dataset = Dataset.from_list(data_list)

# 方式3：从 Pandas DataFrame
import pandas as pd

df = pd.DataFrame({
    "text": ["Hello", "World", "Test"],
    "label": [0, 1, 0]
})

dataset = Dataset.from_pandas(df)
```

**指定特征类型**：

```python
from datasets import Dataset, Features, Value, ClassLabel

# 定义特征 schema
features = Features({
    "text": Value("string"),
    "label": ClassLabel(names=["negative", "positive"]),
    "score": Value("float32")
})

data = {
    "text": ["Good", "Bad"],
    "label": [1, 0],
    "score": [0.9, 0.1]
}

dataset = Dataset.from_dict(data, features=features)
print(dataset.features)
```

### 4.6.3 上传自定义数据集到 Hub

**准备数据集**：

```python
from datasets import Dataset, DatasetDict

# 创建训练集和测试集
train_data = {"text": [...], "label": [...]}
test_data = {"text": [...], "label": [...]}

train_dataset = Dataset.from_dict(train_data)
test_dataset = Dataset.from_dict(test_data)

# 组合成 DatasetDict
dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})
```

**上传到 Hub**：

```python
# 需要先登录
from huggingface_hub import login
login()  # 会提示输入 token

# 上传数据集
dataset_dict.push_to_hub("my_username/my_dataset_name")

# 添加数据集卡片（README.md）
dataset_dict.push_to_hub(
    "my_username/my_dataset_name",
    config_name="default",
    private=False  # 公开数据集
)
```

**加载自己的数据集**：

```python
# 其他人可以这样加载你的数据集
dataset = load_dataset("my_username/my_dataset_name")
```

---

## 本章总结

**核心要点**：

1. ✅ **Datasets 库优势**：内存映射、零拷贝、智能缓存，处理大数据集高效
2. ✅ **核心操作**：map()、filter()、select()、shuffle()、train_test_split()
3. ✅ **Tokenization 集成**：批量处理、多进程加速、移除原始列
4. ✅ **DataCollator 家族**：
   - `DataCollatorWithPadding` - 动态 padding（推荐）
   - `DataCollatorForLanguageModeling` - MLM 任务
   - `DataCollatorForSeq2Seq` - 翻译/摘要
5. ✅ **流式数据集**：适合超大数据集，IterableDataset 按需加载
6. ✅ **自定义数据集**：CSV/JSON 加载、Python 字典创建、上传到 Hub

**最佳实践**：

- 使用 `batched=True` + `num_proc` 加速数据处理
- Tokenization 时不做 padding，留给 DataCollator
- 移除不需要的列节省内存
- 大数据集使用流式模式
- 善用缓存避免重复计算

**下一章预告**：  
Chapter 5 将深入 **Trainer API**，学习完整的训练流程、TrainingArguments 参数详解、回调函数、多 GPU 训练等高级特性。

---

## 练习题

1. **基础题**：使用 Datasets 库加载 GLUE 的 MRPC 任务，tokenize 后查看第一个样本的 input_ids、attention_mask 和 labels。

2. **进阶题**：实现一个自定义 DataCollator，对文本进行随机大小写转换（数据增强），并与标准 DataCollator 对比训练效果。

3. **挑战题**：从 CSV 文件加载自定义数据集，划分训练/验证/测试集（70%/15%/15%），使用流式模式处理，计算每个分割的平均文本长度，最后上传到 Hugging Face Hub。

4. **思考题**：为什么动态 padding 比固定 padding 更高效？在什么情况下固定 padding 可能更合适？计算两种方式在 IMDB 数据集上的理论计算量差异（假设平均长度 200，最大长度 512）。

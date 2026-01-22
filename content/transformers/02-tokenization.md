# Chapter 2: Tokenization 深度剖析

> **本章目标**：深入理解 Transformers 库的 Tokenization 机制，掌握从原始文本到模型输入张量的完整转换流程，理解不同 tokenization 算法的设计动机与实现细节。

---

## 2.1 Tokenizer 核心概念

### 2.1.1 从文本到 ID 的映射过程

Tokenization 是自然语言处理中的核心步骤，负责将人类可读的文本转换为模型可处理的数字序列。在 Transformers 库中，这一过程通常包含以下几个阶段：

**完整流程**：
```
原始文本 → 预处理 → 分词 → 子词拆分 → ID映射 → 张量构建
```

让我们用一个具体示例来演示：

```python
from transformers import AutoTokenizer

# 加载 BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Hugging Face is amazing!"

# 完整的编码过程
encoded = tokenizer(text, return_tensors="pt")

print("原始文本:", text)
print("\nTokens:", tokenizer.tokenize(text))
print("\nToken IDs:", encoded["input_ids"])
print("\nAttention Mask:", encoded["attention_mask"])

# 解码回文本
decoded = tokenizer.decode(encoded["input_ids"][0])
print("\n解码后:", decoded)
```

**预期输出**：
```
原始文本: Hugging Face is amazing!

Tokens: ['hugging', 'face', 'is', 'amazing', '!']

Token IDs: tensor([[ 101, 17662,  2227,  2003,  6429,  999,  102]])

Attention Mask: tensor([[1, 1, 1, 1, 1, 1, 1]])

解码后: [CLS] hugging face is amazing ! [SEP]
```

<div data-component="TokenizationVisualizer"></div>

**关键观察**：
1. **特殊标记自动添加**：`[CLS]` (ID=101) 在开头，`[SEP]` (ID=102) 在结尾
2. **大小写归一化**：BERT-uncased 自动转为小写
3. **标点符号保留**：感叹号被单独编码为 ID 999
4. **Attention Mask**：全为 1 表示所有 token 都需要被关注

### 2.1.2 词汇表（Vocabulary）与特殊标记

每个 tokenizer 都有一个固定的词汇表（vocabulary），将 token 字符串映射到唯一的整数 ID。

**查看词汇表大小**：
```python
print(f"词汇表大小: {tokenizer.vocab_size}")
# 输出: 词汇表大小: 30522

# 查看特殊标记
print(f"[CLS] token ID: {tokenizer.cls_token_id}")  # 101
print(f"[SEP] token ID: {tokenizer.sep_token_id}")  # 102
print(f"[PAD] token ID: {tokenizer.pad_token_id}")  # 0
print(f"[MASK] token ID: {tokenizer.mask_token_id}")  # 103
print(f"[UNK] token ID: {tokenizer.unk_token_id}")  # 100
```

**特殊标记的作用**：

| 标记 | ID (BERT) | 用途 |
|------|-----------|------|
| `[PAD]` | 0 | 填充短序列，使批次中所有序列等长 |
| `[UNK]` | 100 | 未知词（不在词汇表中的词） |
| `[CLS]` | 101 | 分类任务的句子开头标记，其输出用于分类 |
| `[SEP]` | 102 | 句子分隔符（单句末尾或双句之间） |
| `[MASK]` | 103 | 掩码语言模型中的掩码标记 |

**访问词汇表**：
```python
# 获取前10个 token
print(list(tokenizer.get_vocab().items())[:10])

# 输出示例:
# [('[PAD]', 0), ('[unused0]', 1), ('[unused1]', 2), ...]

# 反向查询：从 ID 到 token
print(tokenizer.convert_ids_to_tokens([101, 2003, 102]))
# 输出: ['[CLS]', 'is', '[SEP]']
```

### 2.1.3 编码（encode）与解码（decode）

Transformers 提供了多种编码方法，适用于不同场景：

**方法对比**：

```python
text = "Hello, world!"

# 方法1: 直接调用 tokenizer（推荐，最灵活）
encoded1 = tokenizer(text, return_tensors="pt")
print("方法1 - tokenizer():")
print(f"  input_ids shape: {encoded1['input_ids'].shape}")
print(f"  keys: {encoded1.keys()}")

# 方法2: encode（只返回 ID 列表）
encoded2 = tokenizer.encode(text)
print("\n方法2 - encode():")
print(f"  type: {type(encoded2)}")
print(f"  IDs: {encoded2}")

# 方法3: encode_plus（返回字典，包含更多信息）
encoded3 = tokenizer.encode_plus(text, return_tensors="pt")
print("\n方法3 - encode_plus():")
print(f"  keys: {encoded3.keys()}")

# 方法4: batch_encode_plus（批量编码）
texts = ["Hello", "World"]
encoded4 = tokenizer.batch_encode_plus(texts, padding=True, return_tensors="pt")
print("\n方法4 - batch_encode_plus():")
print(f"  input_ids shape: {encoded4['input_ids'].shape}")
```

**解码方法**：

```python
ids = [101, 7592, 1010, 2088, 999, 102]

# 完整解码（包含特殊标记）
print(tokenizer.decode(ids))
# 输出: [CLS] hello, world! [SEP]

# 跳过特殊标记
print(tokenizer.decode(ids, skip_special_tokens=True))
# 输出: hello, world!

# 批量解码
batch_ids = [[101, 7592, 102], [101, 2088, 102]]
print(tokenizer.batch_decode(batch_ids, skip_special_tokens=True))
# 输出: ['hello', 'world']
```

---

## 2.2 Tokenization 算法家族

不同的 tokenization 算法有不同的设计哲学，解决了词汇表大小、未登录词（OOV）、多语言支持等问题。

<div data-component="TokenAlgorithmComparison"></div>

### 2.2.1 WordPiece（BERT、DistilBERT）

**核心思想**：贪婪最长匹配 + 子词拆分

WordPiece 由 Google 提出，用于 BERT 模型。它使用贪婪算法从左到右匹配最长的词汇表子串，如果无法匹配则拆分为更小的子词。

**算法流程**：
```
1. 尝试匹配整个词
2. 如果失败，从词的开头开始匹配最长前缀
3. 剩余部分添加 '##' 前缀继续匹配
4. 重复直到整个词被分解
```

**示例**：
```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 常见词：直接匹配
print(tokenizer.tokenize("playing"))
# 输出: ['playing']

# 稀有词：拆分为子词
print(tokenizer.tokenize("unhappiness"))
# 输出: ['un', '##hap', '##piness']

# 未登录词：逐字符拆分
print(tokenizer.tokenize("Huggingface"))
# 输出: ['hugging', '##face']

# 完全未知的词
print(tokenizer.tokenize("asdfghjkl"))
# 输出: ['as', '##d', '##f', '##g', '##h', '##j', '##k', '##l']
```

**关键特点**：
- **子词标记**：`##` 前缀表示该 token 是单词的延续部分
- **OOV 鲁棒性**：即使词不在词汇表中，也能分解为字符级 token
- **多语言支持**：BERT-multilingual 使用相同算法

### 2.2.2 Byte-Pair Encoding (BPE) - GPT-2、RoBERTa

**核心思想**：自底向上的统计合并

BPE 最初用于数据压缩，后被 GPT-2 引入 NLP。它从字符级开始，迭代合并最频繁的字符对。

**训练过程**：
```
1. 初始词汇表 = 所有字符 + </w> (词结束符)
2. 统计所有相邻字符对的频率
3. 合并最频繁的字符对，加入词汇表
4. 重复步骤2-3，直到达到目标词汇表大小
```

**示例**：
```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# GPT-2 使用 byte-level BPE
print(tokenizer.tokenize("Hello, world!"))
# 输出: ['Hello', ',', 'Ġworld', '!']
# 注意: 'Ġ' 表示空格（使用 Unicode U+0120）

print(tokenizer.tokenize("lower"))
# 输出: ['lower']

print(tokenizer.tokenize("lowercase"))
# 输出: ['lower', 'case']

# 编码时可以看到 byte-level 的优势
print(tokenizer.tokenize("café"))
# 输出: ['c', 'af', 'Ã', '©']（UTF-8 字节序列）
```

**与 WordPiece 的区别**：

| 特性 | BPE | WordPiece |
|------|-----|-----------|
| 合并策略 | 频率最高的字符对 | 最大化语言模型似然 |
| 子词标记 | 空格用 `Ġ` 或 `</w>` | 延续部分用 `##` |
| 字节级处理 | GPT-2 使用 byte-level（支持任意 Unicode） | 字符级（有 [UNK]） |
| 代表模型 | GPT-2, RoBERTa, BART | BERT, DistilBERT |

### 2.2.3 Unigram（XLNet、ALBERT）

**核心思想**：概率模型 + 动态规划

Unigram 将 tokenization 视为概率问题，每个 token 有一个出现概率，使用 Viterbi 算法找到最优分词路径。

**算法特点**：
- 从大词汇表开始，逐步移除低概率 token
- 使用 EM 算法优化 token 概率
- 允许多种分词方案，选择概率最高的

**示例**：
```python
from transformers import XLNetTokenizer

tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")

print(tokenizer.tokenize("unhappiness"))
# 输出: ['▁un', 'happiness']（'▁' 表示词开头）

print(tokenizer.tokenize("hello world"))
# 输出: ['▁hello', '▁world']

# Unigram 保留大小写
print(tokenizer.tokenize("BERT vs GPT"))
# 输出: ['▁BERT', '▁vs', '▁G', 'PT']
```

### 2.2.4 SentencePiece（T5、ALBERT、XLM-RoBERTa）

**核心思想**：语言无关的子词分词

SentencePiece 直接在原始文本上操作，不依赖预分词（pre-tokenization），特别适合没有明显词边界的语言（如中文、日文）。

**独特特性**：
1. **无需预分词**：将空格编码为特殊字符（`▁`）
2. **可逆性**：`decode(encode(text)) == text`（包括空格）
3. **语言无关**：同一算法适用于所有语言

**示例**：
```python
from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-small")

# 英文
print(tokenizer.tokenize("Hello, world!"))
# 输出: ['▁Hello', ',', '▁world', '!']

# 中文（无需分词）
print(tokenizer.tokenize("你好，世界！"))
# 输出: ['▁', '你好', '，', '世界', '！']

# 空格被保留
text = "  multiple   spaces  "
print(tokenizer.decode(tokenizer.encode(text)))
# 输出完全一致，包括所有空格
```

### 2.2.5 算法选择指南

**选择建议**：

| 场景 | 推荐算法 | 理由 |
|------|----------|------|
| 英文文本分类 | WordPiece (BERT) | 性能优异，预训练模型丰富 |
| 文本生成 | BPE (GPT-2) | 更好的生成连贯性 |
| 多语言任务 | SentencePiece (XLM-R, mT5) | 语言无关，统一处理 |
| 代码生成 | BPE (CodeGen, StarCoder) | 处理特殊字符能力强 |
| 资源受限 | Unigram (ALBERT) | 更小的词汇表 |

---

## 2.3 AutoTokenizer 使用详解

### 2.3.1 from_pretrained() 参数详解

`AutoTokenizer.from_pretrained()` 是加载 tokenizer 的统一入口，支持丰富的自定义选项。

**完整参数列表**：
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",           # 模型名称或本地路径
    
    # 词汇表修改
    additional_special_tokens=["<custom>"],  # 添加自定义特殊标记
    
    # 缓存控制
    cache_dir="./my_cache",        # 自定义缓存目录
    force_download=False,          # 强制重新下载
    resume_download=False,         # 断点续传
    
    # 本地加载
    local_files_only=False,        # 仅使用本地文件
    
    # 版本控制
    revision="main",               # 指定 Hub 分支/tag
    
    # 其他
    use_fast=True,                 # 使用 Fast Tokenizer（Rust 实现）
    trust_remote_code=False,       # 是否信任远程代码（自定义 tokenizer）
)
```

**关键参数详解**：

**1. additional_special_tokens - 添加自定义标记**
```python
tokenizer = AutoTokenizer.from_pretrained(
    "gpt2",
    additional_special_tokens=["<|endoftext|>", "<|startofchat|>"]
)

# 自定义标记会被添加到词汇表
print(f"原始词汇表大小: {tokenizer.vocab_size}")
# 输出: 50257

print(f"总标记数（含特殊标记）: {len(tokenizer)}")
# 输出: 50259

# 使用自定义标记
text = "<|startofchat|> Hello! <|endoftext|>"
print(tokenizer.tokenize(text))
```

**2. use_fast - Fast Tokenizer 的优势**
```python
# Fast Tokenizer（Rust 实现）
fast_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

# Slow Tokenizer（Python 实现）
slow_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)

import time

text = "This is a test sentence. " * 1000  # 重复1000次

# 性能对比
start = time.time()
fast_tokenizer(text)
fast_time = time.time() - start

start = time.time()
slow_tokenizer(text)
slow_time = time.time() - start

print(f"Fast Tokenizer: {fast_time:.4f}s")
print(f"Slow Tokenizer: {slow_time:.4f}s")
print(f"Speedup: {slow_time / fast_time:.2f}x")

# 典型输出: Fast 比 Slow 快 10-100 倍

# Fast Tokenizer 独有功能：offset mapping
encoded = fast_tokenizer(text, return_offsets_mapping=True)
print(encoded["offset_mapping"][:5])
# 输出: [(0, 0), (0, 4), (5, 7), (8, 9), (10, 14)]
# 每个 tuple 表示 token 在原始文本中的字符位置
```

### 2.3.2 批量编码（batch_encode_plus）

**动态 padding vs 固定 padding**：
```python
texts = [
    "Short text",
    "A much longer text that needs more tokens to encode properly"
]

# 方式1: 固定长度 padding
encoded1 = tokenizer(
    texts,
    padding="max_length",        # 填充到 max_length
    max_length=20,
    truncation=True,
    return_tensors="pt"
)

print("固定 padding:")
print(f"Shape: {encoded1['input_ids'].shape}")
print(f"Attention mask:\n{encoded1['attention_mask']}")

# 方式2: 批次内最长序列（推荐）
encoded2 = tokenizer(
    texts,
    padding="longest",           # 填充到批次内最长
    truncation=True,
    return_tensors="pt"
)

print("\n动态 padding:")
print(f"Shape: {encoded2['input_ids'].shape}")
print(f"Attention mask:\n{encoded2['attention_mask']}")

# 方式3: 不填充（返回列表）
encoded3 = tokenizer(
    texts,
    padding=False,
    truncation=True
)

print("\n无 padding:")
print(f"Type: {type(encoded3['input_ids'])}")
print(f"Lengths: {[len(ids) for ids in encoded3['input_ids']]}")
```

**预期输出**：
```
固定 padding:
Shape: torch.Size([2, 20])
Attention mask:
tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])

动态 padding:
Shape: torch.Size([2, 17])
Attention mask:
tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

无 padding:
Type: <class 'list'>
Lengths: [4, 17]
```

<div data-component="AttentionMaskBuilder"></div>

### 2.3.3 截断（truncation）与填充（padding）策略

**截断策略详解**：
```python
long_text = "This is a very long sentence " * 50  # 生成超长文本

# 策略1: 仅截断（最常用）
encoded = tokenizer(
    long_text,
    truncation=True,           # 启用截断
    max_length=128,            # 最大长度
    return_tensors="pt"
)
print(f"截断后长度: {encoded['input_ids'].shape[1]}")  # 128

# 策略2: 双句截断（用于句子对任务）
text_a = "First sentence " * 30
text_b = "Second sentence " * 30

encoded_pair = tokenizer(
    text_a,
    text_b,
    truncation="only_first",   # 仅截断第一句
    # truncation="only_second" # 仅截断第二句
    # truncation="longest_first" # 截断较长的句子（默认）
    max_length=100,
    return_tensors="pt"
)

print(f"句子对长度: {encoded_pair['input_ids'].shape[1]}")

# 策略3: 不截断（会抛出错误如果超长）
try:
    tokenizer(long_text, truncation=False, max_length=128)
except ValueError as e:
    print(f"错误: {e}")
```

**处理超长文档**（使用 stride）：
```python
# 滑动窗口切分长文档
long_document = "Paragraph one. " * 100

encoded_chunks = tokenizer(
    long_document,
    max_length=128,
    stride=20,                 # 重叠20个token
    truncation=True,
    return_overflowing_tokens=True,  # 返回所有切片
    return_tensors="pt"
)

print(f"切片数量: {len(encoded_chunks['input_ids'])}")
print(f"每个切片形状: {encoded_chunks['input_ids'][0].shape}")

# 查看重叠部分
print("\n第一个切片的最后5个token:")
print(tokenizer.decode(encoded_chunks['input_ids'][0][-5:]))
print("\n第二个切片的前5个token:")
print(tokenizer.decode(encoded_chunks['input_ids'][1][:5]))
# 可以看到重叠区域
```

### 2.3.4 返回张量格式（return_tensors）

```python
text = "Hello, world!"

# PyTorch 张量
pt_encoded = tokenizer(text, return_tensors="pt")
print(f"PyTorch: {type(pt_encoded['input_ids'])}")
print(f"Shape: {pt_encoded['input_ids'].shape}")

# TensorFlow 张量
tf_encoded = tokenizer(text, return_tensors="tf")
print(f"TensorFlow: {type(tf_encoded['input_ids'])}")

# NumPy 数组
np_encoded = tokenizer(text, return_tensors="np")
print(f"NumPy: {type(np_encoded['input_ids'])}")

# Python 列表（默认）
list_encoded = tokenizer(text)
print(f"List: {type(list_encoded['input_ids'])}")
```

---

## 2.4 高级 Tokenization 技巧

### 2.4.1 动态 padding（DataCollator）

在训练时，推荐使用 `DataCollator` 进行动态 padding，而不是在预处理时固定 padding，可显著减少计算浪费。

```python
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("glue", "sst2", split="train[:100]")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 预处理（不进行 padding）
def preprocess(examples):
    return tokenizer(examples["sentence"], truncation=True)

tokenized_dataset = dataset.map(preprocess, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "idx"])
tokenized_dataset.set_format("torch")

# 使用 DataCollator 动态 padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 创建 DataLoader
dataloader = DataLoader(
    tokenized_dataset,
    batch_size=8,
    collate_fn=data_collator
)

# 查看批次形状（每个批次可能不同）
for batch in dataloader:
    print(f"Batch shape: {batch['input_ids'].shape}")
    break

# 输出: Batch shape: torch.Size([8, 37])
# 注意：37 是该批次中最长序列的长度，不是固定值
```

**性能对比**：
```python
import time

# 方案1: 固定 padding 到 512
def preprocess_fixed(examples):
    return tokenizer(examples["sentence"], padding="max_length", max_length=512, truncation=True)

# 方案2: 动态 padding
def preprocess_dynamic(examples):
    return tokenizer(examples["sentence"], truncation=True)

# 测量实际计算量（以平均序列长度为指标）
dataset_fixed = dataset.map(preprocess_fixed, batched=True)
dataset_dynamic = dataset.map(preprocess_dynamic, batched=True)

avg_len_fixed = sum(len(ids) for ids in dataset_fixed["input_ids"]) / len(dataset_fixed)
avg_len_dynamic = sum(len(ids) for ids in dataset_dynamic["input_ids"]) / len(dataset_dynamic)

print(f"固定 padding 平均长度: {avg_len_fixed:.1f}")  # 512.0
print(f"动态 padding 平均长度: {avg_len_dynamic:.1f}")  # ~20-30
print(f"计算节省: {(1 - avg_len_dynamic/avg_len_fixed) * 100:.1f}%")  # ~95%
```

### 2.4.2 处理长文本（stride、max_length、overflow）

对于超过模型最大长度的文档（如 BERT 的 512），需要特殊处理策略。

**策略1：滑动窗口分片**
```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 长文档（>512 tokens）
long_text = """
[此处假设有一篇5000词的文章...]
""" * 10

# 使用 stride 切分
encoded = tokenizer(
    long_text,
    max_length=512,
    stride=128,                      # 重叠128个token
    truncation=True,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,     # 记录每个token在原文的位置
    return_tensors="pt"
)

print(f"分片数量: {len(encoded['input_ids'])}")
print(f"每片形状: {encoded['input_ids'].shape}")

# 示例：在问答任务中使用
# 对每个分片进行推理，找到 logits 最高的答案
for i, input_ids in enumerate(encoded['input_ids']):
    print(f"\n分片 {i}:")
    print(f"  Token范围: {encoded['offset_mapping'][i][0]} - {encoded['offset_mapping'][i][-1]}")
    # 进行推理...
```

**策略2：Longformer / BigBird（支持更长上下文）**
```python
from transformers import LongformerTokenizer

# Longformer 支持最长 4096 tokens
long_tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

encoded = long_tokenizer(long_text, return_tensors="pt")
print(f"Longformer 编码长度: {encoded['input_ids'].shape[1]}")  # 可达 4096
```

### 2.4.3 Fast Tokenizer 的高级功能

Fast Tokenizer（Rust 实现）提供了许多 Python 版本没有的功能。

**1. Offset Mapping（字符级对齐）**
```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

text = "Hello, world!"
encoded = tokenizer(text, return_offsets_mapping=True)

print("Token | Offsets | 原文片段")
print("-" * 40)
for token_id, (start, end) in zip(encoded["input_ids"], encoded["offset_mapping"]):
    token = tokenizer.decode([token_id])
    original = text[start:end] if start != end else "[SPECIAL]"
    print(f"{token:10} | ({start:2}, {end:2}) | {original}")

# 输出:
# Token      | Offsets | 原文片段
# ----------------------------------------
# [CLS]      | ( 0,  0) | [SPECIAL]
# hello      | ( 0,  5) | Hello
# ,          | ( 5,  6) | ,
# world      | ( 7, 12) | world
# !          | (12, 13) | !
# [SEP]      | ( 0,  0) | [SPECIAL]
```

**应用：命名实体识别（NER）的标签对齐**
```python
# 原始 NER 标签（字符级）
text = "John lives in New York"
char_labels = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 2, 2, 2, 2]
# 1=人名开头, 2=地名开头, 0=其他

# Tokenization
encoded = tokenizer(text, return_offsets_mapping=True)

# 将字符级标签对齐到 token 级
token_labels = []
for start, end in encoded["offset_mapping"]:
    if start == end:  # 特殊标记
        token_labels.append(-100)  # 忽略标签
    else:
        token_labels.append(char_labels[start])  # 使用首字符的标签

print("Token labels:", token_labels)
# 输出: [-100, 1, 0, 0, 2, 2, -100]
```

**2. Word IDs（单词级分组）**
```python
text = "Hello, world!"
encoded = tokenizer(text, return_offsets_mapping=True)

# 获取每个 token 对应的单词索引
word_ids = encoded.word_ids()
print(f"Word IDs: {word_ids}")
# 输出: [None, 0, 1, 2, 3, None]
# None 表示特殊标记，0-3 表示原文的单词索引

# 应用：聚合单词级别的预测
import torch

# 假设每个 token 有一个分数
token_scores = torch.tensor([0.0, 0.9, 0.1, 0.8, 0.7, 0.0])

# 聚合到单词级别（取最大值）
word_scores = []
for word_id in range(max(w for w in word_ids if w is not None) + 1):
    indices = [i for i, w in enumerate(word_ids) if w == word_id]
    word_scores.append(token_scores[indices].max().item())

print(f"Word scores: {word_scores}")
# 输出: [0.9, 0.1, 0.8, 0.7]（对应 "Hello", ",", "world", "!"）
```

### 2.4.4 自定义词汇表与训练 tokenizer

**从头训练 tokenizer（BPE）**：
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 初始化 tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# 训练配置
trainer = BpeTrainer(
    vocab_size=10000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

# 准备训练语料（文本文件列表）
files = ["corpus1.txt", "corpus2.txt"]

# 训练
tokenizer.train(files, trainer)

# 保存
tokenizer.save("my_tokenizer.json")

# 转换为 Transformers 兼容格式
from transformers import PreTrainedTokenizerFast

fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="my_tokenizer.json")
fast_tokenizer.save_pretrained("./my_custom_tokenizer")
```

**微调现有 tokenizer（添加领域词汇）**：
```python
# 场景：医疗领域微调
base_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 添加医疗术语
new_tokens = ["covid19", "mRNA", "vaccine", "antibody"]
num_added = base_tokenizer.add_tokens(new_tokens)

print(f"添加了 {num_added} 个新 token")
print(f"新词汇表大小: {len(base_tokenizer)}")

# 重要：调整模型的 embedding 层大小
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-uncased")
model.resize_token_embeddings(len(base_tokenizer))

# 新 token 的 embedding 会被随机初始化，需要继续训练
```

---

## 2.5 特殊场景处理

### 2.5.1 多语言 tokenization（XLM-RoBERTa）

```python
from transformers import XLMRobertaTokenizer

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

# 同时处理多种语言
texts = {
    "en": "Hello, world!",
    "zh": "你好，世界！",
    "ar": "مرحبا بالعالم",
    "ja": "こんにちは世界",
}

for lang, text in texts.items():
    tokens = tokenizer.tokenize(text)
    print(f"{lang}: {tokens}")

# 输出:
# en: ['▁Hello', ',', '▁world', '!']
# zh: ['▁', '你好', '，', '世界', '！']
# ar: ['▁مرحبا', '▁بال', 'عالم']
# ja: ['▁こんにち', 'は', '世界']
```

### 2.5.2 对话历史编码（chat templates）

**Chat 模板（Llama-2 / Mistral 风格）**：
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# 对话历史
conversation = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you!"},
    {"role": "user", "content": "What's the weather like?"},
]

# 使用 chat template 自动格式化
formatted = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,  # 先查看格式化后的文本
    add_generation_prompt=True  # 添加生成提示符
)

print(formatted)
# 输出（Llama-2 格式）:
# <s>[INST] Hello, how are you? [/INST] I'm doing well, thank you!</s>
# <s>[INST] What's the weather like? [/INST]

# 直接编码
encoded = tokenizer.apply_chat_template(
    conversation,
    tokenize=True,
    return_tensors="pt"
)
print(f"Encoded shape: {encoded.shape}")
```

### 2.5.3 结构化输入（JSON、表格）

**表格问答（TAPAS）**：
```python
from transformers import TapasTokenizer
import pandas as pd

tokenizer = TapasTokenizer.from_pretrained("google/tapas-base")

# 表格数据
table = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["NYC", "LA", "Chicago"]
})

# 问题
query = "What is Bob's age?"

# Tokenization（表格 + 问题）
encoded = tokenizer(
    table=table,
    queries=query,
    padding="max_length",
    max_length=512,
    return_tensors="pt"
)

print(f"Input IDs shape: {encoded['input_ids'].shape}")
print(f"Token type IDs: {encoded['token_type_ids'].shape}")  # 区分问题/表头/单元格
```

### 2.5.4 代码 tokenization（CodeBERT、CodeGen）

```python
from transformers import AutoTokenizer

code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# Python 代码
code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

# Tokenization（保留缩进和空格）
tokens = code_tokenizer.tokenize(code)
print(f"Tokens: {tokens}")

# 输出会保留 'Ġ' (空格标记)
# ['Ċ', 'def', 'Ġfibonacci', '(', 'n', '):', 'Ċ', 'Ġ', 'Ġ', 'Ġ', 'Ġif', ...]
```

---

## 2.6 常见陷阱与调试

### 2.6.1 Token ID 与 Position ID 的区别

```python
text = "Hello, world!"
encoded = tokenizer(text, return_tensors="pt")

print("Input IDs (token 映射):", encoded["input_ids"])
# 输出: tensor([[  101,  7592,  1010,  2088,   999,   102]])

# Position IDs（位置编码，通常自动生成）
position_ids = torch.arange(encoded["input_ids"].shape[1]).unsqueeze(0)
print("Position IDs:", position_ids)
# 输出: tensor([[0, 1, 2, 3, 4, 5]])

# 大多数模型会自动生成 position_ids，无需手动传入
# 但对于特殊任务（如相对位置编码）可能需要自定义
```

### 2.6.2 Attention Mask 的作用

```python
# Attention Mask 告诉模型哪些 token 需要关注（1），哪些应该忽略（0，通常是 padding）

texts = ["Short", "This is a much longer sentence"]
encoded = tokenizer(texts, padding=True, return_tensors="pt")

print("Input IDs:")
print(encoded["input_ids"])
print("\nAttention Mask:")
print(encoded["attention_mask"])

# 输出:
# Input IDs:
# tensor([[  101,  2460,   102,     0,     0,     0,     0,     0,     0],
#         [  101,  2023,  2003,  1037,  2172,  3092,  6251,   102,     0]])
# Attention Mask:
# tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 1, 1, 0]])

# 在自注意力计算中，mask=0 的位置会被设为 -inf，softmax 后变为 0
```

### 2.6.3 为什么有时需要 token_type_ids？

```python
# token_type_ids 用于区分多个句子（如 BERT 的句子对任务）

sentence_a = "What is AI?"
sentence_b = "Artificial Intelligence is the simulation of human intelligence."

# 句子对编码
encoded = tokenizer(sentence_a, sentence_b, return_tensors="pt")

print("Input IDs:")
print(encoded["input_ids"])
print("\nToken Type IDs:")
print(encoded["token_type_ids"])

# 输出:
# Input IDs:
# tensor([[  101,  2054,  2003,  9932,  1029,   102,  6342, ...,   102]])
# Token Type IDs:
# tensor([[    0,     0,     0,     0,     0,     0,     1, ...,     1]])
#          ↑------ 句子A (0) ------↑     ↑------- 句子B (1) -------↑

# 注意：GPT 系列模型不使用 token_type_ids
```

### 2.6.4 Tokenizer 版本不匹配问题

```python
# 常见错误：使用了不匹配的 tokenizer 和模型

# 错误示例
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("gpt2")  # ❌ 不匹配！

# 问题：
# 1. 词汇表不同（BERT: 30522, GPT-2: 50257）
# 2. 特殊标记不同（BERT有[CLS]/[SEP]，GPT-2没有）
# 3. Token ID 映射完全不同

# 正确做法：始终使用配套的 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModel.from_pretrained("gpt2")  # ✅ 匹配

# 或者统一使用模型名称
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
```

**检测不匹配的方法**：
```python
# 方法1: 检查词汇表大小
assert tokenizer.vocab_size == model.config.vocab_size, "Vocab size mismatch!"

# 方法2: 检查特殊标记
print(f"Tokenizer pad token: {tokenizer.pad_token}")
print(f"Model pad token ID: {model.config.pad_token_id}")

# 方法3: 测试编码/解码一致性
test_text = "Test"
ids = tokenizer.encode(test_text)
decoded = tokenizer.decode(ids, skip_special_tokens=True)
assert test_text.lower() == decoded.lower(), "Encode-decode mismatch!"
```

---

## 本章总结

**核心要点**：
1. ✅ **Tokenization 流程**：文本 → 预处理 → 分词 → 子词拆分 → ID 映射
2. ✅ **算法选择**：WordPiece (BERT)、BPE (GPT-2)、Unigram (XLNet)、SentencePiece (T5)
3. ✅ **AutoTokenizer**：统一接口，支持所有模型
4. ✅ **高级技巧**：动态 padding、滑动窗口、Fast Tokenizer、自定义词汇表
5. ✅ **特殊场景**：多语言、对话、表格、代码
6. ✅ **常见陷阱**：版本匹配、attention mask、token_type_ids

**下一章预告**：  
Chapter 3 将深入模型架构，学习 **Auto 类体系**、**模型加载机制**、**配置管理** 和 **预训练权重迁移**，完成从 tokenization 到模型推理的完整链路。

---

## 练习题

1. **基础题**：使用 GPT-2 tokenizer 编码句子 "I don't know"，观察输出的 token，解释为什么会有 `Ġ` 字符。

2. **进阶题**：实现一个函数，将字符级 NER 标签对齐到 BERT tokenizer 的 subword 级别（使用 `offset_mapping`）。

3. **挑战题**：在医疗数据集上从头训练一个 BPE tokenizer（词汇表大小 5000），并与 BERT tokenizer 在医疗术语覆盖率上进行对比。

4. **思考题**：为什么在训练时推荐使用 DataCollator 动态 padding，而不是预先 padding 到固定长度？计算并比较两种方式在 GLUE SST-2 数据集上的理论计算量差异。

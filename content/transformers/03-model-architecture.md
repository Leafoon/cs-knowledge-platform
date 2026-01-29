---
title: "Chapter 3. 模型架构与 Auto 类"
description: "全面掌握 Transformer 模型家族架构差异、深入理解 Auto 类自动匹配机制"
updated: "2026-01-22"
---

---

## 3.1 Transformer 模型家族概览

Transformer 架构自 2017 年 "Attention is All You Need" 论文发表以来，已演化出三大主要分支，每个分支针对不同的 NLP 任务进行了优化。

<div data-component="ArchitectureExplorer"></div>

### 3.1.1 Encoder-only 架构（BERT 系列）

**设计理念**：双向上下文编码，适合理解类任务

**核心特点**：
- **双向注意力**：每个 token 可以看到整个序列的所有 token
- **掩码语言模型（MLM）**预训练：随机掩盖 15% 的 token，预测被掩盖的词
- **句子对任务**：通过 `[SEP]` 分隔符和 segment embeddings 处理两个句子

**代表模型**：

| 模型 | 参数量 | 特点 | 适用场景 |
|------|--------|------|----------|
| **BERT-base** | 110M | 12层，768维，原始 BERT | 文本分类、NER、问答 |
| **BERT-large** | 340M | 24层，1024维，更强性能 | 需要更高精度的任务 |
| **RoBERTa** | 125M | 移除 NSP、更大 batch、更多数据 | BERT 的改进版，性能更优 |
| **ELECTRA** | 110M | 判别式预训练（更高效） | 小数据集、资源受限 |
| **DeBERTa** | 134M | 解耦注意力、增强位置编码 | SOTA 性能（GLUE 排行榜） |

**架构细节**：
```python
from transformers import BertModel, BertConfig

# 查看 BERT 配置
config = BertConfig.from_pretrained("bert-base-uncased")

print("=== BERT-base 架构参数 ===")
print(f"层数 (num_hidden_layers): {config.num_hidden_layers}")          # 12
print(f"隐藏维度 (hidden_size): {config.hidden_size}")                   # 768
print(f"注意力头数 (num_attention_heads): {config.num_attention_heads}") # 12
print(f"中间层维度 (intermediate_size): {config.intermediate_size}")     # 3072 (4*768)
print(f"最大序列长度 (max_position_embeddings): {config.max_position_embeddings}") # 512
print(f"词汇表大小 (vocab_size): {config.vocab_size}")                   # 30522

# 加载模型并查看结构
model = BertModel.from_pretrained("bert-base-uncased")
print(f"\n总参数量: {model.num_parameters():,}")

# 打印模型结构（简化）
print("\n=== 模型结构 ===")
for name, module in model.named_children():
    print(f"{name}: {module.__class__.__name__}")
```

**预期输出**：
```
=== BERT-base 架构参数 ===
层数 (num_hidden_layers): 12
隐藏维度 (hidden_size): 768
注意力头数 (num_attention_heads): 12
中间层维度 (intermediate_size): 3072
最大序列长度 (max_position_embeddings): 512
词汇表大小 (vocab_size): 30522

总参数量: 109,482,240

=== 模型结构 ===
embeddings: BertEmbeddings
encoder: BertEncoder
pooler: BertPooler
```

**BERT 的自注意力机制**：
```python
import torch

# 模拟 BERT 的双向注意力
batch_size, seq_len, hidden_size = 2, 5, 768

# 输入序列（假设已经过 embedding）
hidden_states = torch.randn(batch_size, seq_len, hidden_size)

# Attention mask（1 表示需要注意，0 表示忽略）
attention_mask = torch.ones(batch_size, seq_len)
attention_mask[0, 3:] = 0  # 第一个序列只有前3个token有效

# 扩展 mask 用于注意力计算
# BERT 使用 additive mask（加法形式）
extended_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0
print(f"Extended attention mask shape: {extended_mask.shape}")
# 输出: torch.Size([2, 1, 1, 5])

# 在注意力矩阵上应用 mask
# attention_scores: [batch, num_heads, seq_len, seq_len]
# 假设的注意力分数
attention_scores = torch.randn(batch_size, 12, seq_len, seq_len)
masked_scores = attention_scores + extended_mask

print("\n原始注意力分数（第一个序列，第一个头）:")
print(attention_scores[0, 0])
print("\nMasked 后（padding token 的分数变为 -10000）:")
print(masked_scores[0, 0])
```

### 3.1.2 Decoder-only 架构（GPT 系列）

**设计理念**：单向因果建模，适合生成类任务

**核心特点**：
- **因果注意力（Causal Attention）**：每个 token 只能看到之前的 token（自回归）
- **自回归预训练**：预测下一个 token（Language Modeling）
- **无需特殊标记**：通常只有 `<|endoftext|>` 作为分隔符

**代表模型**：

| 模型 | 参数量 | 特点 | 适用场景 |
|------|--------|------|----------|
| **GPT-2** | 124M-1.5B | OpenAI 经典生成模型 | 文本续写、创意写作 |
| **GPT-3** | 175B | 超大规模、上下文学习 | Few-shot 任务、对话 |
| **LLaMA-2** | 7B-70B | Meta 开源、高效架构 | 指令跟随、对话 |
| **Mistral-7B** | 7B | 滑动窗口注意力、高效推理 | 长文本生成、RAG |
| **Qwen** | 1.8B-72B | 中文优化、多模态 | 中文任务、代码生成 |

**架构细节**：
```python
from transformers import GPT2Model, GPT2Config

config = GPT2Config.from_pretrained("gpt2")

print("=== GPT-2 架构参数 ===")
print(f"层数: {config.n_layer}")                      # 12
print(f"隐藏维度: {config.n_embd}")                   # 768
print(f"注意力头数: {config.n_head}")                 # 12
print(f"最大序列长度: {config.n_positions}")          # 1024（比 BERT 更长）
print(f"词汇表大小: {config.vocab_size}")             # 50257

# 关键差异：GPT-2 使用 Causal Attention
model = GPT2Model.from_pretrained("gpt2")
print(f"\n总参数量: {model.num_parameters():,}")

# GPT-2 的因果掩码是固定的下三角矩阵
seq_len = 5
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
print("\n因果注意力掩码（下三角矩阵）:")
print(causal_mask)
# 输出:
# tensor([[1., 0., 0., 0., 0.],
#         [1., 1., 0., 0., 0.],
#         [1., 1., 1., 0., 0.],
#         [1., 1., 1., 1., 0.],
#         [1., 1., 1., 1., 1.]])
# 每一行表示该位置可以看到的 token（1=可见，0=不可见）
```

**LLaMA-2 的架构改进**：
```python
from transformers import LlamaConfig

config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")

print("=== LLaMA-2 改进点 ===")
print(f"RMSNorm 代替 LayerNorm: 更高效的归一化")
print(f"SwiGLU 激活函数: 性能优于 GELU")
print(f"旋转位置编码 (RoPE): 更好的长度外推")
print(f"分组查询注意力 (GQA): {config.num_key_value_heads} KV heads")
# LLaMA-2-7B: 32 个 Q heads，但只有 32 个 KV heads（1:1）
# 更大模型如 LLaMA-2-70B 使用 GQA（8个KV heads）以减少显存
```

### 3.1.3 Encoder-Decoder 架构（T5、BART）

**设计理念**：结合编码器和解码器，适合序列到序列任务

**核心特点**：
- **Encoder**：双向注意力，编码输入序列
- **Decoder**：因果注意力 + 交叉注意力（关注 encoder 输出）
- **Seq2Seq 预训练**：如 T5 的 Span Corruption、BART 的去噪自编码

**代表模型**：

| 模型 | 参数量 | 预训练任务 | 适用场景 |
|------|--------|------------|----------|
| **T5** | 60M-11B | Span Corruption（填空） | 文本到文本（翻译、摘要、QA） |
| **BART** | 140M | 去噪自编码（删除、替换） | 摘要、对话生成 |
| **mT5** | 300M-13B | 多语言 T5 | 跨语言任务 |
| **Flan-T5** | 80M-11B | 指令微调的 T5 | Zero-shot 任务 |

**架构细节**：
```python
from transformers import T5Model, T5Config

config = T5Config.from_pretrained("t5-small")

print("=== T5-small 架构参数 ===")
print(f"Encoder 层数: {config.num_layers}")          # 6
print(f"Decoder 层数: {config.num_decoder_layers}")  # 6
print(f"隐藏维度: {config.d_model}")                 # 512
print(f"注意力头数: {config.num_heads}")             # 8
print(f"FFN 维度: {config.d_ff}")                    # 2048
print(f"相对位置编码桶数: {config.relative_attention_num_buckets}")  # 32

# T5 的独特之处：所有任务统一为 "text-to-text"
# 输入前缀示例:
examples = {
    "翻译": "translate English to French: Hello, how are you?",
    "摘要": "summarize: [长文本...]",
    "分类": "cola sentence: This is a good sentence.",
    "问答": "question: What is NLP? context: Natural language processing...",
}

for task, prompt in examples.items():
    print(f"\n{task} 示例:")
    print(f"  输入: {prompt[:50]}...")
```

**Encoder-Decoder 的交叉注意力**：
```python
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# 输入和输出
input_text = "translate English to French: Hello"
target_text = "Bonjour"

# 编码
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
labels = tokenizer(target_text, return_tensors="pt").input_ids

# 前向传播
outputs = model(
    input_ids=input_ids,
    labels=labels,
    output_attentions=True  # 输出注意力权重
)

print(f"Loss: {outputs.loss.item():.4f}")
print(f"Encoder 注意力层数: {len(outputs.encoder_attentions)}")
print(f"Decoder self-attention 层数: {len(outputs.decoder_attentions)}")
print(f"Cross-attention 层数: {len(outputs.cross_attentions)}")

# 查看交叉注意力（decoder 关注 encoder 的程度）
cross_attn = outputs.cross_attentions[0]  # 第一层的交叉注意力
print(f"\nCross-attention shape: {cross_attn.shape}")
# 输出: [batch_size, num_heads, decoder_seq_len, encoder_seq_len]
```

### 3.1.4 架构选择指南

**任务 → 架构映射表**：

| 任务类型 | 推荐架构 | 推荐模型 | 理由 |
|----------|----------|----------|------|
| **文本分类** | Encoder-only | BERT, RoBERTa, DeBERTa | 需要理解整个句子的语义 |
| **命名实体识别（NER）** | Encoder-only | BERT, RoBERTa | Token 级别分类，需要双向上下文 |
| **抽取式问答** | Encoder-only | BERT, ELECTRA | 需要定位答案 span |
| **文本生成** | Decoder-only | GPT-2, LLaMA, Mistral | 自回归生成，单向依赖 |
| **对话系统** | Decoder-only | LLaMA-2-Chat, Qwen-Chat | 指令跟随能力强 |
| **摘要生成** | Encoder-Decoder | BART, Pegasus | 需要理解全文 + 生成摘要 |
| **机器翻译** | Encoder-Decoder | T5, mBART, mT5 | 经典 Seq2Seq 任务 |
| **代码生成** | Decoder-only | CodeGen, StarCoder | 代码具有自回归性质 |
| **嵌入 / 检索** | Encoder-only | BERT, Sentence-BERT | 需要固定长度向量表示 |

**关键决策因素**：

```python
def recommend_architecture(task_description):
    """根据任务描述推荐架构"""
    
    # 判断是否需要生成
    if any(word in task_description.lower() for word in ["生成", "续写", "创作", "对话"]):
        if "摘要" in task_description or "翻译" in task_description:
            return "Encoder-Decoder", ["T5", "BART", "mT5"]
        else:
            return "Decoder-only", ["GPT-2", "LLaMA-2", "Mistral"]
    
    # 判断是否需要理解
    elif any(word in task_description.lower() for word in ["分类", "情感", "NER", "问答"]):
        return "Encoder-only", ["BERT", "RoBERTa", "DeBERTa"]
    
    else:
        return "Unknown", ["请提供更多任务细节"]

# 测试
tasks = [
    "对电影评论进行情感分类",
    "根据上下文生成故事续写",
    "将英文文章翻译成中文",
    "从文本中识别人名和地名",
]

for task in tasks:
    arch, models = recommend_architecture(task)
    print(f"任务: {task}")
    print(f"  推荐架构: {arch}")
    print(f"  推荐模型: {', '.join(models)}\n")
```

---

## 3.2 Auto 类体系

Transformers 库提供了 `Auto*` 类，能够根据模型名称或配置自动选择正确的模型类，极大简化了代码编写。

### 3.2.1 AutoConfig：配置自动加载

**核心功能**：根据模型名称自动加载对应的配置类

```python
from transformers import AutoConfig

# 示例：加载不同模型的配置
model_names = ["bert-base-uncased", "gpt2", "t5-small", "facebook/bart-base"]

for name in model_names:
    config = AutoConfig.from_pretrained(name)
    print(f"\n=== {name} ===")
    print(f"配置类型: {config.__class__.__name__}")
    print(f"架构类型: {config.model_type}")
    print(f"隐藏维度: {config.hidden_size if hasattr(config, 'hidden_size') else config.d_model}")

# 输出:
# === bert-base-uncased ===
# 配置类型: BertConfig
# 架构类型: bert
# 隐藏维度: 768
#
# === gpt2 ===
# 配置类型: GPT2Config
# 架构类型: gpt2
# 隐藏维度: 768
#
# === t5-small ===
# 配置类型: T5Config
# 架构类型: t5
# 隐藏维度: 512
```

**修改配置**：
```python
# 加载默认配置并修改
config = AutoConfig.from_pretrained("bert-base-uncased")

# 修改层数（用于实验小模型）
config.num_hidden_layers = 6  # 从 12 层减少到 6 层
config.num_attention_heads = 6  # 对应减少注意力头

print(f"修改后的层数: {config.num_hidden_layers}")
print(f"修改后的注意力头数: {config.num_attention_heads}")

# 从修改后的配置初始化模型（随机权重）
from transformers import BertModel
model = BertModel(config)
print(f"新模型参数量: {model.num_parameters():,}")
# 输出会显著小于原始 BERT（因为层数减半）
```

### 3.2.2 AutoTokenizer：tokenizer 自动匹配

**核心功能**：根据模型名称自动加载对应的 tokenizer

```python
from transformers import AutoTokenizer

# 自动匹配 tokenizer
model_names = ["bert-base-uncased", "gpt2", "t5-small", "xlm-roberta-base"]

for name in model_names:
    tokenizer = AutoTokenizer.from_pretrained(name)
    print(f"\n=== {name} ===")
    print(f"Tokenizer 类型: {tokenizer.__class__.__name__}")
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"特殊标记: {tokenizer.all_special_tokens[:5]}")

# 输出示例:
# === bert-base-uncased ===
# Tokenizer 类型: BertTokenizerFast
# 词汇表大小: 30522
# 特殊标记: ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
#
# === gpt2 ===
# Tokenizer 类型: GPT2TokenizerFast
# 词汇表大小: 50257
# 特殊标记: ['<|endoftext|>']
```

### 3.2.3 AutoModel：通用模型加载

**核心功能**：加载基础模型（不含任务头）

```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 推理
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

print(f"输出类型: {type(outputs)}")
print(f"Last hidden state shape: {outputs.last_hidden_state.shape}")
# 输出: torch.Size([1, 7, 768])
#       [batch_size, sequence_length, hidden_size]

print(f"Pooler output shape: {outputs.pooler_output.shape}")
# 输出: torch.Size([1, 768])
#       [batch_size, hidden_size] - [CLS] token 的表示
```

**AutoModel 的局限性**：
```python
# AutoModel 只返回隐藏状态，无法直接用于特定任务

# ❌ 错误：期望获得分类 logits，但没有
outputs = model(**inputs)
# outputs.logits  # AttributeError: 'BaseModelOutputWithPoolingAndCrossAttentions' object has no attribute 'logits'

# ✅ 正确：使用任务专用模型
from transformers import AutoModelForSequenceClassification

classifier = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2  # 二分类
)
outputs = classifier(**inputs)
print(f"Logits shape: {outputs.logits.shape}")  # torch.Size([1, 2])
```

### 3.2.4 AutoModelForXXX：任务专用模型头

**常用任务类**：

| 类名 | 任务 | 输出 | 示例模型 |
|------|------|------|----------|
| `AutoModelForSequenceClassification` | 文本分类 | logits: [batch, num_labels] | BERT, RoBERTa |
| `AutoModelForTokenClassification` | Token 分类（NER） | logits: [batch, seq_len, num_labels] | BERT, DistilBERT |
| `AutoModelForQuestionAnswering` | 抽取式问答 | start_logits, end_logits | BERT, ELECTRA |
| `AutoModelForCausalLM` | 因果语言模型 | logits: [batch, seq_len, vocab_size] | GPT-2, LLaMA |
| `AutoModelForMaskedLM` | 掩码语言模型 | logits: [batch, seq_len, vocab_size] | BERT, RoBERTa |
| `AutoModelForSeq2SeqLM` | 序列到序列 | logits: [batch, tgt_len, vocab_size] | T5, BART |

**示例：文本分类**
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 加载预训练的情感分类模型
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 推理
texts = ["I love this movie!", "This is terrible."]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# 解析输出
logits = outputs.logits
predictions = torch.argmax(logits, dim=-1)
probabilities = torch.softmax(logits, dim=-1)

print("=== 分类结果 ===")
for i, text in enumerate(texts):
    print(f"\n文本: {text}")
    print(f"预测类别: {predictions[i].item()} ({'正面' if predictions[i] == 1 else '负面'})")
    print(f"概率分布: 负面={probabilities[i][0]:.3f}, 正面={probabilities[i][1]:.3f}")

# 输出:
# 文本: I love this movie!
# 预测类别: 1 (正面)
# 概率分布: 负面=0.001, 正面=0.999
#
# 文本: This is terrible.
# 预测类别: 0 (负面)
# 概率分布: 负面=0.998, 正面=0.002
```

**示例：命名实体识别（NER）**
```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

model_name = "dslim/bert-base-NER"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

text = "Elon Musk founded SpaceX in California."
inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)

with torch.no_grad():
    outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])

# 解析预测
predictions = torch.argmax(outputs.logits, dim=-1)[0]
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# 标签映射
id2label = model.config.id2label

print("=== NER 结果 ===")
for token, pred_id in zip(tokens, predictions):
    label = id2label[pred_id.item()]
    if label != "O":  # 跳过非实体
        print(f"{token:15} -> {label}")

# 输出示例:
# Elon            -> B-PER
# Musk            -> I-PER
# Space           -> B-ORG
# ##X             -> I-ORG
# California      -> B-LOC
```

**示例：文本生成（Causal LM）**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")

# 生成文本
outputs = model.generate(
    inputs["input_ids"],
    max_length=50,
    num_return_sequences=1,
    temperature=0.8,
    do_sample=True
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"生成文本: {generated_text}")

# 输出示例:
# 生成文本: Once upon a time, in a small village, there lived a young girl named Emma...
```

---

## 3.3 模型加载详解

### 3.3.1 from_pretrained() 参数全解析

`from_pretrained()` 是加载模型的核心方法，支持丰富的自定义选项。

**完整参数列表**：
```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "bert-base-uncased",              # 模型名称或本地路径
    
    # === 缓存与下载 ===
    cache_dir="./my_model_cache",     # 自定义缓存目录
    force_download=False,             # 强制重新下载
    resume_download=False,            # 断点续传
    proxies=None,                     # 代理设置
    local_files_only=False,           # 仅使用本地文件
    
    # === 模型配置 ===
    config=None,                      # 自定义配置对象
    
    # === 设备与精度 ===
    torch_dtype=torch.float16,        # 模型权重精度（fp32/fp16/bf16）
    device_map="auto",                # 自动设备分配（多 GPU / CPU offload）
    low_cpu_mem_usage=True,           # 低内存加载模式
    
    # === 权重处理 ===
    ignore_mismatched_sizes=False,    # 忽略形状不匹配的权重
    
    # === 安全与信任 ===
    trust_remote_code=False,          # 是否信任远程自定义代码
    
    # === 版本控制 ===
    revision="main",                  # Hub 分支/tag/commit
    
    # === 其他 ===
    output_loading_info=False,        # 返回加载详细信息
)
```

**关键参数详解**：

**1. torch_dtype - 控制模型精度**
```python
import torch
from transformers import AutoModel

# FP32（默认，最高精度）
model_fp32 = AutoModel.from_pretrained("bert-base-uncased")
print(f"FP32 模型显存: ~440MB")

# FP16（半精度，显存减半）
model_fp16 = AutoModel.from_pretrained(
    "bert-base-uncased",
    torch_dtype=torch.float16
)
print(f"FP16 模型显存: ~220MB")

# BF16（Brain Float16，推荐在 A100/H100 上使用）
model_bf16 = AutoModel.from_pretrained(
    "bert-base-uncased",
    torch_dtype=torch.bfloat16
)

# 检查实际精度
print(f"FP32 权重类型: {next(model_fp32.parameters()).dtype}")  # torch.float32
print(f"FP16 权重类型: {next(model_fp16.parameters()).dtype}")  # torch.float16
```

**2. device_map - 自动设备分配**
```python
# 场景1: 单 GPU
model = AutoModel.from_pretrained("bert-base-uncased", device_map="cuda:0")

# 场景2: 多 GPU 自动分配（模型并行）
model = AutoModel.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto"  # 自动在多个 GPU 间分配层
)

# 查看每层的设备分配
print(model.hf_device_map)
# 输出示例:
# {'embeddings': 0, 'layer.0': 0, 'layer.1': 0, ..., 'layer.30': 1, 'layer.31': 1}

# 场景3: CPU + GPU 混合（大模型无法全部放入 GPU）
model = AutoModel.from_pretrained(
    "facebook/opt-66b",
    device_map="auto",
    offload_folder="offload"  # CPU offload 的临时文件夹
)
```

**3. low_cpu_mem_usage - 低内存加载**
```python
# 传统加载方式（峰值内存 = 2x 模型大小）
# 1. 在内存中创建模型（随机权重）
# 2. 从磁盘加载权重到内存
# 3. 复制到模型对象
# 峰值: 模型 + 权重文件 = 2x

# 低内存加载（推荐）
model = AutoModel.from_pretrained(
    "bert-base-uncased",
    low_cpu_mem_usage=True  # 逐层加载，峰值内存 ~1x
)

# 对于大模型（>10GB）尤其重要
```

### 3.3.2 本地加载 vs Hub 加载

**从 Hub 加载（默认）**：
```python
# 方式1: 使用模型名称
model = AutoModel.from_pretrained("bert-base-uncased")

# 方式2: 使用完整 Hub 路径
model = AutoModel.from_pretrained("huggingface/bert-base-uncased")

# 方式3: 指定版本
model = AutoModel.from_pretrained("bert-base-uncased", revision="v1.0.0")
```

**保存到本地**：
```python
# 下载并保存
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 保存到本地目录
save_path = "./my_bert_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# 查看保存的文件
import os
print("保存的文件:")
for file in os.listdir(save_path):
    print(f"  {file}")

# 输出:
#   config.json
#   pytorch_model.bin  (或 model.safetensors)
#   tokenizer_config.json
#   vocab.txt
#   special_tokens_map.json
```

**从本地加载**：
```python
# 从本地目录加载
model = AutoModel.from_pretrained("./my_bert_model")
tokenizer = AutoTokenizer.from_pretrained("./my_bert_model")

# 禁止网络访问（仅使用本地文件）
model = AutoModel.from_pretrained(
    "./my_bert_model",
    local_files_only=True  # 如果文件不存在会报错
)
```

### 3.3.3 权重文件格式（safetensors vs bin vs h5）

**格式对比**：

| 格式 | 文件扩展名 | 优点 | 缺点 | 安全性 |
|------|-----------|------|------|--------|
| **SafeTensors** | `.safetensors` | 加载快、安全、跨框架 | 较新，部分旧模型不支持 | ✅ 高（无代码执行风险） |
| **PyTorch** | `.bin` | 兼容性好、广泛使用 | 加载慢、有安全风险 | ⚠️ 低（可能执行恶意代码） |
| **TensorFlow** | `.h5` | TensorFlow 原生格式 | 仅 TensorFlow 使用 | ⚠️ 中 |

**SafeTensors 的优势**：
```python
from transformers import AutoModel
import time

model_name = "bert-base-uncased"

# 测试 SafeTensors 加载速度
start = time.time()
model_safe = AutoModel.from_pretrained(model_name, use_safetensors=True)
safe_time = time.time() - start

# 测试 PyTorch bin 加载速度
start = time.time()
model_bin = AutoModel.from_pretrained(model_name, use_safetensors=False)
bin_time = time.time() - start

print(f"SafeTensors 加载时间: {safe_time:.2f}s")
print(f"PyTorch bin 加载时间: {bin_time:.2f}s")
print(f"提速: {bin_time / safe_time:.2f}x")

# 典型输出: SafeTensors 比 bin 快 2-3 倍
```

**转换格式**：
```python
# 将 PyTorch bin 转换为 SafeTensors
from safetensors.torch import save_file
import torch

model = AutoModel.from_pretrained("bert-base-uncased")

# 提取 state_dict
state_dict = model.state_dict()

# 保存为 SafeTensors
save_file(state_dict, "model.safetensors")

print("转换完成: model.safetensors")
```

### 3.3.4 分片加载大模型（sharded checkpoints）

**为什么需要分片？**
- 单个文件 >5GB 时，GitHub / HF Hub 上传受限
- 内存受限设备无法一次加载整个模型

**分片文件结构**：
```
llama-2-7b/
├── config.json
├── pytorch_model-00001-of-00002.bin  (5GB)
├── pytorch_model-00002-of-00002.bin  (2.8GB)
└── pytorch_model.bin.index.json      (索引文件)
```

**自动加载分片模型**：
```python
# Transformers 会自动处理分片文件
model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf")

# 内部流程:
# 1. 读取 pytorch_model.bin.index.json
# 2. 确定每层的权重在哪个分片文件中
# 3. 逐层加载（配合 low_cpu_mem_usage=True）
```

**手动分片保存**：
```python
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-large-uncased")

# 分片保存（每个分片 <5GB）
model.save_pretrained(
    "./my_large_model",
    max_shard_size="5GB"  # 每个分片最大 5GB
)

# 查看生成的文件
import os
for file in os.listdir("./my_large_model"):
    size_mb = os.path.getsize(f"./my_large_model/{file}") / 1024 / 1024
    print(f"{file:40} {size_mb:8.2f} MB")
```

---

## 3.4 模型配置（Config）

<div data-component="ConfigEditor"></div>

### 3.4.1 config.json 结构详解

**BERT 的 config.json 示例**：
```json
{
  "architectures": ["BertForMaskedLM"],
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "type_vocab_size": 2,
  "vocab_size": 30522
}
```

**字段详解**：
```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("bert-base-uncased")

print("=== 核心架构参数 ===")
print(f"模型类型: {config.model_type}")                   # bert
print(f"层数: {config.num_hidden_layers}")                # 12
print(f"隐藏维度: {config.hidden_size}")                  # 768
print(f"注意力头数: {config.num_attention_heads}")        # 12
print(f"每个头的维度: {config.hidden_size // config.num_attention_heads}")  # 64

print("\n=== FFN 参数 ===")
print(f"中间层维度: {config.intermediate_size}")          # 3072 (4 * 768)
print(f"激活函数: {config.hidden_act}")                   # gelu

print("\n=== 正则化参数 ===")
print(f"Dropout 概率: {config.hidden_dropout_prob}")      # 0.1
print(f"注意力 Dropout: {config.attention_probs_dropout_prob}")  # 0.1
print(f"LayerNorm epsilon: {config.layer_norm_eps}")      # 1e-12

print("\n=== 位置编码 ===")
print(f"最大序列长度: {config.max_position_embeddings}")  # 512
print(f"位置编码类型: {config.position_embedding_type}")  # absolute

print("\n=== 词汇表 ===")
print(f"词汇表大小: {config.vocab_size}")                 # 30522
print(f"Segment 类型数: {config.type_vocab_size}")        # 2（句子A/B）
```

### 3.4.2 修改模型配置

**场景1：调整模型大小（用于实验）**
```python
from transformers import BertConfig, BertModel

# 创建一个更小的 BERT
config = BertConfig(
    vocab_size=30522,
    hidden_size=384,          # 原始 768 的一半
    num_hidden_layers=6,      # 原始 12 的一半
    num_attention_heads=6,    # 保持 head_dim=64
    intermediate_size=1536,   # 4 * hidden_size
    max_position_embeddings=512
)

# 从配置初始化模型（随机权重）
tiny_bert = BertModel(config)
print(f"Tiny BERT 参数量: {tiny_bert.num_parameters():,}")  # ~28M（原始 110M）

# 对比
standard_bert = BertModel.from_pretrained("bert-base-uncased")
print(f"Standard BERT 参数量: {standard_bert.num_parameters():,}")
```

**场景2：修改任务头参数**
```python
from transformers import AutoModelForSequenceClassification

# 修改分类标签数
config = AutoConfig.from_pretrained("bert-base-uncased")
config.num_labels = 5  # 5分类任务

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    config=config,
    ignore_mismatched_sizes=True  # 忽略分类头的形状不匹配
)

print(f"分类头输出维度: {model.classifier.out_features}")  # 5
```

**场景3：启用额外输出**
```python
# 启用隐藏状态和注意力权重输出
config = AutoConfig.from_pretrained("bert-base-uncased")
config.output_hidden_states = True
config.output_attentions = True

model = AutoModel.from_pretrained("bert-base-uncased", config=config)

# 推理
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)

print(f"隐藏状态层数: {len(outputs.hidden_states)}")  # 13（embedding + 12层）
print(f"注意力权重层数: {len(outputs.attentions)}")  # 12
```

### 3.4.3 自定义配置类

**创建自定义配置**：
```python
from transformers import PretrainedConfig

class MyCustomConfig(PretrainedConfig):
    model_type = "my_custom_model"
    
    def __init__(
        self,
        vocab_size=50000,
        hidden_size=768,
        num_layers=12,
        custom_param="default",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.custom_param = custom_param

# 使用自定义配置
config = MyCustomConfig(
    vocab_size=10000,
    hidden_size=512,
    num_layers=6,
    custom_param="my_value"
)

# 保存配置
config.save_pretrained("./my_custom_config")

# 加载配置
loaded_config = MyCustomConfig.from_pretrained("./my_custom_config")
print(f"Custom param: {loaded_config.custom_param}")
```

---

## 3.5 模型输出结构

<div data-component="ModelOutputInspector"></div>

### 3.5.1 ModelOutput 基类

所有 Transformers 模型的输出都继承自 `ModelOutput`，提供了统一的接口。

**特点**：
1. **字典式访问**：`outputs["last_hidden_state"]`
2. **属性式访问**：`outputs.last_hidden_state`
3. **元组解包**：`loss, logits = outputs`（按定义顺序）

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model(**inputs)

print(f"输出类型: {type(outputs)}")
# 输出: <class 'transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions'>

# 访问方式1: 属性
print(f"Last hidden state shape: {outputs.last_hidden_state.shape}")

# 访问方式2: 字典
print(f"Pooler output shape: {outputs['pooler_output'].shape}")

# 访问方式3: 元组解包
last_hidden, pooler = outputs.to_tuple()[:2]
print(f"通过元组解包: {last_hidden.shape}, {pooler.shape}")
```

### 3.5.2 logits、hidden_states、attentions 的含义

**1. logits - 未归一化的分数**
```python
from transformers import AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

inputs = tokenizer("I love this!", return_tensors="pt")
outputs = model(**inputs)

logits = outputs.logits
print(f"Logits: {logits}")
# 输出: tensor([[-4.1589,  4.4513]], grad_fn=<AddmmBackward>)
#               ↑ 负面    ↑ 正面

# 转换为概率
probs = torch.softmax(logits, dim=-1)
print(f"Probabilities: {probs}")
# 输出: tensor([[0.0002, 0.9998]])
```

**2. hidden_states - 所有层的隐藏状态**
```python
model = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)

hidden_states = outputs.hidden_states
print(f"隐藏状态层数: {len(hidden_states)}")  # 13

# 每层的形状
for i, hs in enumerate(hidden_states):
    print(f"Layer {i}: {hs.shape}")

# 输出:
# Layer 0: torch.Size([1, 4, 768])  # Embedding 层
# Layer 1: torch.Size([1, 4, 768])  # Transformer 第1层
# ...
# Layer 12: torch.Size([1, 4, 768]) # Transformer 第12层

# 应用：提取中间层特征（如第9层）
layer_9_output = hidden_states[9]
```

**3. attentions - 注意力权重**
```python
model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)
outputs = model(**inputs)

attentions = outputs.attentions
print(f"注意力层数: {len(attentions)}")  # 12

# 每层的形状: [batch_size, num_heads, seq_len, seq_len]
print(f"第1层注意力形状: {attentions[0].shape}")
# 输出: torch.Size([1, 12, 4, 4])

# 可视化第1层第1个头的注意力
import matplotlib.pyplot as plt

attn_head_0 = attentions[0][0, 0].detach().numpy()  # [seq_len, seq_len]
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

plt.figure(figsize=(6, 6))
plt.imshow(attn_head_0, cmap='viridis')
plt.xticks(range(len(tokens)), tokens, rotation=90)
plt.yticks(range(len(tokens)), tokens)
plt.colorbar()
plt.title("Attention Weights (Layer 1, Head 1)")
plt.show()
```

### 3.5.3 output_hidden_states 与 output_attentions 参数

**全局设置 vs 推理时设置**：
```python
# 方式1: 在配置中全局设置
config = AutoConfig.from_pretrained("bert-base-uncased")
config.output_hidden_states = True
config.output_attentions = True
model = AutoModel.from_pretrained("bert-base-uncased", config=config)

# 方式2: 推理时临时设置（推荐）
model = AutoModel.from_pretrained("bert-base-uncased")
outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

# 方式2 的优势：不增加不必要的计算开销
```

**性能影响**：
```python
import time

model = AutoModel.from_pretrained("bert-base-uncased").cuda()
inputs = {k: v.cuda() for k, v in tokenizer("Hello " * 100, return_tensors="pt").items()}

# 测试1: 仅输出 last_hidden_state
start = time.time()
for _ in range(100):
    outputs = model(**inputs)
time_basic = time.time() - start

# 测试2: 输出所有 hidden_states 和 attentions
start = time.time()
for _ in range(100):
    outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
time_full = time.time() - start

print(f"基础输出耗时: {time_basic:.3f}s")
print(f"完整输出耗时: {time_full:.3f}s")
print(f"性能下降: {(time_full / time_basic - 1) * 100:.1f}%")

# 典型输出: 性能下降 ~10-20%（因为需要存储中间结果）
```

---

## 3.6 预训练权重的迁移学习

### 3.6.1 头部替换（忽略权重警告）

**场景：分类任务的标签数变化**
```python
from transformers import AutoModelForSequenceClassification

# 原模型: 2分类（SST-2）
original_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
print(f"原分类头: {original_model.classifier.out_features} 个类别")  # 2

# 新任务: 5分类
new_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english",
    num_labels=5,
    ignore_mismatched_sizes=True  # 关键参数
)
print(f"新分类头: {new_model.classifier.out_features} 个类别")  # 5

# 警告信息（正常）:
# Some weights of the model checkpoint at ... were not used when initializing ...
# - classifier.weight (torch.Size([2, 768]) != torch.Size([5, 768]))
# - classifier.bias (torch.Size([2]) != torch.Size([5]))
```

**检查哪些权重被重新初始化**：
```python
# 加载时获取详细信息
model, loading_info = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=10,
    ignore_mismatched_sizes=True,
    output_loading_info=True  # 返回加载信息
)

print("未使用的权重:")
for key in loading_info["missing_keys"]:
    print(f"  {key}")

print("\n意外的权重:")
for key in loading_info["unexpected_keys"]:
    print(f"  {key}")

# 输出:
# 未使用的权重:
#   classifier.weight
#   classifier.bias
```

### 3.6.2 部分权重初始化

**场景：使用预训练的 encoder，自定义 decoder**
```python
from transformers import BertModel
import torch.nn as nn

# 加载预训练的 BERT encoder
encoder = BertModel.from_pretrained("bert-base-uncased")

# 自定义分类头
class CustomClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size, 512)
        self.dropout = nn.Dropout(0.3)
        self.out_proj = nn.Linear(512, num_labels)
    
    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

# 组合模型
class MyModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = encoder
        self.classifier = CustomClassifier(768, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        logits = self.classifier(pooled)
        return logits

model = MyModel(num_labels=3)

# BERT 部分使用预训练权重，分类头随机初始化
print("BERT 权重已加载（预训练）")
print("分类头权重随机初始化")
```

### 3.6.3 跨模型权重迁移

**场景：从 BERT 迁移到 RoBERTa（架构相似）**
```python
from transformers import BertModel, RobertaModel
import torch

# 加载 BERT
bert = BertModel.from_pretrained("bert-base-uncased")
bert_state = bert.state_dict()

# 初始化 RoBERTa（随机权重）
roberta = RobertaModel(RobertaConfig())

# 尝试迁移权重（名称可能不完全匹配）
roberta_state = roberta.state_dict()

transferred = 0
for name, param in bert_state.items():
    # 尝试直接匹配
    if name in roberta_state and param.shape == roberta_state[name].shape:
        roberta_state[name] = param
        transferred += 1
    else:
        # 尝试重命名匹配（例如 'bert' -> 'roberta'）
        new_name = name.replace('bert', 'roberta')
        if new_name in roberta_state and param.shape == roberta_state[new_name].shape:
            roberta_state[new_name] = param
            transferred += 1

roberta.load_state_dict(roberta_state, strict=False)
print(f"成功迁移 {transferred} / {len(bert_state)} 个权重")

# 注意：这只是示例，实际中 BERT 和 RoBERTa 的架构差异较大
# 更推荐直接使用对应的预训练模型
```

---

## 本章总结

**核心要点**：
1. ✅ **架构分类**：Encoder-only（BERT）、Decoder-only（GPT）、Encoder-Decoder（T5）
2. ✅ **Auto 类体系**：AutoConfig、AutoTokenizer、AutoModel、AutoModelForXXX
3. ✅ **模型加载**：from_pretrained() 的丰富参数（精度、设备、缓存）
4. ✅ **配置管理**：修改 config 实现自定义架构
5. ✅ **输出结构**：logits、hidden_states、attentions 的含义与应用
6. ✅ **权重迁移**：头部替换、部分初始化、跨模型迁移

**下一章预告**：  
Chapter 4 将介绍 **Datasets 库**，学习如何高效加载、预处理数据集，并与 Transformers 模型无缝集成。

---

## 练习题

1. **基础题**：加载 GPT-2 模型，查看其 config 中的 `n_layer`、`n_embd`、`n_head`，并计算每个注意力头的维度。

2. **进阶题**：使用 `output_attentions=True`，提取 BERT 最后一层第一个头的注意力权重，并用 matplotlib 可视化注意力矩阵热力图。

3. **挑战题**：实现一个函数，自动将任意 BERT 变体（如 BERT-base、RoBERTa、ELECTRA）的 encoder 权重迁移到一个自定义的分类模型中，处理可能的命名差异。

4. **思考题**：为什么 Decoder-only 模型（GPT）使用因果掩码（causal mask），而 Encoder-only 模型（BERT）使用双向注意力？这对各自的预训练任务和下游应用有何影响？

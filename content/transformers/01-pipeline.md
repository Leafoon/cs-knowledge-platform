---
title: "Chapter 1. Pipeline 快速上手"
description: "深入理解 Pipeline 架构、掌握各类任务的 Pipeline 使用与参数调优"
updated: "2026-01-22"
---

> **Learning Objectives**
> * 深入理解 Pipeline 三阶段架构（Tokenization → Model → Post-processing）
> * 掌握 5+ 种核心任务的 Pipeline 使用方法
> * 熟练调节生成参数（temperature、top_k、top_p、num_beams）
> * 识别 Pipeline 的性能瓶颈与适用场景

---

## 1.1 Pipeline 架构解析

### 1.1.1 三阶段流水线详解

Pipeline 是 Transformers 库的**最高层抽象**，它将复杂的 NLP 任务封装为一个简洁的调用接口。理解其内部机制是从"使用者"进阶到"开发者"的关键。

<div data-component="PipelineFlowVisualizer"></div>

**完整流程**：

```
原始输入 (Raw Input)
    ↓
【阶段 1: Tokenization】
    - 文本 → Token IDs
    - 添加特殊 token ([CLS], [SEP])
    - Padding & Truncation
    ↓
Tensor 输入 (Model Input)
    ↓
【阶段 2: Model Inference】
    - Forward Pass
    - 计算 logits / embeddings
    ↓
模型输出 (Model Output)
    ↓
【阶段 3: Post-processing】
    - Logits → Probabilities (softmax)
    - 提取最佳结果
    - 格式化输出
    ↓
最终结果 (Formatted Result)
```

**实战：手动实现一个情感分析 Pipeline**

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SimpleSentimentPipeline:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()  # 设置为评估模式
        
    def __call__(self, texts):
        # 阶段 1: Tokenization
        inputs = self.tokenizer(
            texts,
            padding=True,       # 自动 padding 到最长序列
            truncation=True,    # 截断超长序列
            return_tensors="pt" # 返回 PyTorch 张量
        )
        print(f"[Tokenization] Input IDs shape: {inputs['input_ids'].shape}")
        
        # 阶段 2: Model Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        print(f"[Model] Logits shape: {logits.shape}")
        print(f"[Model] Raw logits: {logits[0].tolist()}")
        
        # 阶段 3: Post-processing
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            label = self.model.config.id2label[pred.item()]
            score = prob[pred].item()
            results.append({"label": label, "score": score})
            print(f"[Post-processing] Text {i}: {label} ({score:.4f})")
        
        return results

# 使用自定义 Pipeline
pipeline = SimpleSentimentPipeline()
texts = ["I love this!", "This is terrible."]
results = pipeline(texts)
```

**输出**：
```
[Tokenization] Input IDs shape: torch.Size([2, 6])
[Model] Logits shape: torch.Size([2, 2])
[Model] Raw logits: [-4.234, 4.562]
[Post-processing] Text 0: POSITIVE (0.9998)
[Post-processing] Text 1: NEGATIVE (0.9992)
```

> [!NOTE]
> **为什么要分三个阶段？**
> - **解耦**：每个阶段可以独立优化（如使用 Fast Tokenizer、量化模型）
> - **复用**：Tokenizer 和 Model 可以单独使用
> - **灵活**：可以插入自定义 post-processing 逻辑

### 1.1.2 自动任务推断机制

Pipeline 如何知道加载哪个模型？

```python
from transformers import pipeline

# 方式一：仅指定任务（使用默认模型）
classifier = pipeline("sentiment-analysis")
# 等价于：
# classifier = pipeline(
#     task="sentiment-analysis",
#     model="distilbert-base-uncased-finetuned-sst-2-english"
# )

# 方式二：指定模型（自动推断任务）
generator = pipeline(model="gpt2")
# 自动检测到 gpt2 是 CausalLM → 任务为 text-generation

# 方式三：显式指定任务和模型
qa_pipeline = pipeline(
    task="question-answering",
    model="distilbert-base-cased-distilled-squad"
)
```

**任务推断规则**：

<div data-component="TaskInferenceFlowchart"></div>

1. 检查模型配置中的 `architectures` 字段
2. 根据架构类名映射到任务
   - `BertForSequenceClassification` → `text-classification`
   - `GPT2LMHeadModel` → `text-generation`
   - `BertForQuestionAnswering` → `question-answering`
3. 如果无法推断，要求用户显式指定任务

### 1.1.3 设备管理（CPU、GPU、多 GPU）

```python
import torch
from transformers import pipeline

# 自动检测设备
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("sentiment-analysis", device=device)
# device=0    → GPU 0
# device=1    → GPU 1
# device=-1   → CPU

# 显式指定设备
classifier = pipeline("sentiment-analysis", device="cuda:0")

# 多 GPU 并行（自动分片）
classifier = pipeline(
    "text-generation",
    model="meta-llama/Llama-2-7b-hf",
    device_map="auto"  # 自动分配到多张 GPU
)

# 查看当前设备
print(f"Model device: {classifier.model.device}")
```

**设备迁移**：
```python
# 将已创建的 Pipeline 移到 GPU
classifier.model = classifier.model.to("cuda")
```

> [!TIP]
> **性能建议**：
> - 小模型（< 500M 参数）：CPU 足够
> - 中等模型（500M - 3B）：单 GPU
> - 大模型（7B+）：多 GPU + `device_map="auto"`

---

## 1.2 文本分类 Pipeline

### 1.2.1 情感分析（sentiment-analysis）

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

# 单条推理
result = classifier("The movie was fantastic!")[0]
print(f"Label: {result['label']}, Score: {result['score']:.4f}")

# 批量推理（更高效）
texts = [
    "I absolutely loved it!",
    "Worst experience ever.",
    "It was okay, nothing special."
]
results = classifier(texts)

for text, result in zip(texts, results):
    print(f"{text:35} → {result['label']:8} ({result['score']:.3f})")
```

**输出**：
```
Label: POSITIVE, Score: 0.9998

I absolutely loved it!              → POSITIVE (0.999)
Worst experience ever.              → NEGATIVE (0.999)
It was okay, nothing special.       → POSITIVE (0.652)
```

**自定义模型**（中文情感分析）：

```python
# 使用中文 BERT 模型
classifier_cn = pipeline(
    "sentiment-analysis",
    model="uer/roberta-base-finetuned-dianping-chinese"
)

result = classifier_cn("这家餐厅太好吃了！")
print(result)
# [{'label': 'positive', 'score': 0.9987}]
```

### 1.2.2 零样本分类（zero-shot-classification）

**无需训练即可分类任意标签！**

<div data-component="ZeroShotClassificationDemo"></div>

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")

text = "I have a problem with my iPhone battery draining too fast."
candidate_labels = ["technology", "politics", "sports", "health"]

result = classifier(text, candidate_labels)

print(f"Text: {text}\n")
for label, score in zip(result['labels'], result['scores']):
    print(f"{label:15} → {score:.4f}")
```

**输出**：
```
Text: I have a problem with my iPhone battery draining too fast.

technology      → 0.9542
health          → 0.0234
sports          → 0.0156
politics        → 0.0068
```

**工作原理**：
- 使用自然语言推理 (NLI) 模型
- 将分类任务转换为蕴含关系判断
- 假设：`text` 蕴含 `"This text is about {label}"`

**多标签分类**：

```python
text = "Apple just released a new MacBook with M3 chip and improved battery life."
candidate_labels = ["technology", "business", "science"]

result = classifier(
    text,
    candidate_labels,
    multi_label=True  # 允许多个标签同时为真
)

for label, score in zip(result['labels'], result['scores']):
    print(f"{label:15} → {score:.4f}")
```

**输出**：
```
technology      → 0.9823
business        → 0.8934
science         → 0.3421
```

### 1.2.3 自定义标签映射

某些模型的标签是数字或缩写，可以自定义映射：

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# 修改标签映射
model.config.id2label = {0: "消极", 1: "积极"}

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
result = classifier("This is great!")
print(result)
# [{'label': '积极', 'score': 0.9998}]
```

---

## 1.3 文本生成 Pipeline

### 1.3.1 基础文本生成

```python
from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

prompt = "Once upon a time"
outputs = generator(
    prompt,
    max_length=50,      # 生成的最大总长度（包括 prompt）
    num_return_sequences=1
)

print(outputs[0]['generated_text'])
```

**输出**（示例）：
```
Once upon a time, there was a young girl named Lily who lived in a small village. 
She loved to explore the nearby forest and discover new things.
```

### 1.3.2 生成参数详解

<div data-component="GenerationParametersExplorer"></div>

**核心参数对比表**：

| 参数 | 作用 | 典型值 | 效果 |
|------|------|--------|------|
| `max_length` | 生成的最大 token 数 | 50-512 | 控制输出长度 |
| `max_new_tokens` | 在 prompt 基础上新生成的 token 数 | 50-200 | 更精确的长度控制 |
| `temperature` | 采样温度 | 0.7-1.0 | 越低越确定，越高越随机 |
| `top_k` | 只从概率最高的 k 个 token 中采样 | 50 | 限制候选集 |
| `top_p` | 核采样，累计概率达到 p 时停止 | 0.9 | 动态候选集 |
| `num_beams` | 束搜索宽度 | 1（贪婪）或 4-10 | 提高质量但变慢 |
| `do_sample` | 是否采样（否则贪婪） | True/False | 控制随机性 |

**实验：温度的影响**

```python
generator = pipeline("text-generation", model="gpt2")

prompt = "The future of AI is"

for temp in [0.3, 0.7, 1.0, 1.5]:
    output = generator(
        prompt,
        max_new_tokens=30,
        temperature=temp,
        do_sample=True,
        num_return_sequences=1
    )[0]['generated_text']
    print(f"\n[Temperature={temp}]")
    print(output)
```

**输出对比**：
```
[Temperature=0.3] (更确定、重复性高)
The future of AI is already here, and it's already being used in many different ways.
The most common use of AI is in the field of machine learning.

[Temperature=0.7] (平衡)
The future of AI is not about replacing humans, but augmenting them. We need systems
that can understand context, learn from experience, and collaborate with people.

[Temperature=1.0] (更多样)
The future of AI is likely to be shaped by decentralized architectures where multiple
agents collaborate, similar to how biological neural networks operate in nature.

[Temperature=1.5] (极其随机、可能不连贯)
The future of AI is quantum blockchain synergy manifesting through holographic 
consciousness portals enabling telepathic cryptocurrency mining protocols.
```

**Top-K vs Top-P 可视化**：

<div data-component="TopKTopPVisualizer"></div>

```python
# Top-K Sampling
output_topk = generator(
    "Once upon a time",
    max_new_tokens=50,
    do_sample=True,
    top_k=50,           # 只从前 50 个 token 中采样
    temperature=0.8
)

# Top-P (Nucleus) Sampling
output_topp = generator(
    "Once upon a time",
    max_new_tokens=50,
    do_sample=True,
    top_p=0.92,         # 累计概率达到 92% 时停止
    temperature=0.8
)

print("Top-K:", output_topk[0]['generated_text'])
print("\nTop-P:", output_topp[0]['generated_text'])
```

> [!TIP]
> **参数组合建议**：
> - **创意写作**：`temperature=0.9, top_p=0.95, do_sample=True`
> - **事实性文本**：`temperature=0.3, top_k=50, do_sample=True`
> - **代码生成**：`temperature=0.2, num_beams=4` (束搜索)
> - **聊天对话**：`temperature=0.7, top_p=0.9, repetition_penalty=1.2`

### 1.3.3 批量生成与流式输出

**批量生成**（同时生成多个结果）：

```python
generator = pipeline("text-generation", model="gpt2")

prompts = [
    "The capital of France is",
    "Python is a programming language that",
    "In the year 2050,"
]

outputs = generator(
    prompts,
    max_new_tokens=20,
    num_return_sequences=2,  # 每个 prompt 生成 2 个结果
    batch_size=3             # 批处理大小
)

for i, prompt in enumerate(prompts):
    print(f"\n=== Prompt: {prompt} ===")
    for j, output in enumerate(outputs[i*2:(i+1)*2]):
        print(f"[{j+1}] {output['generated_text']}")
```

**流式输出**（逐 token 生成，适合聊天应用）：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "The best way to learn programming is"
inputs = tokenizer(prompt, return_tensors="pt")

# 创建流式输出器
streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

# 在后台线程生成
generation_kwargs = dict(
    **inputs,
    max_new_tokens=50,
    streamer=streamer,
    do_sample=True,
    temperature=0.7
)
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# 实时打印生成的文本
print(prompt, end="")
for new_text in streamer:
    print(new_text, end="", flush=True)
print()

thread.join()
```

**输出**（逐字显示）：
```
The best way to learn programming is to start with small projects and gradually
increase complexity. Practice regularly, read others' code, and don't be afraid
to make mistakes - they're the best teachers.
```

---

## 1.4 问答与抽取 Pipeline

### 1.4.1 抽取式问答（question-answering）

抽取式问答从给定文本中**提取答案片段**。

<div data-component="QuestionAnsweringVisualizer"></div>

```python
from transformers import pipeline

qa_pipeline = pipeline("question-answering")

context = """
Hugging Face is a company founded in 2016 by Clément Delangue, Julien Chaumond, 
and Thomas Wolf. The company is based in New York City and Paris. Hugging Face 
is known for its Transformers library, which provides state-of-the-art NLP models.
The company raised $40 million in Series B funding in 2021.
"""

questions = [
    "When was Hugging Face founded?",
    "Who are the founders?",
    "Where is the company based?",
    "How much funding did they raise in 2021?"
]

for question in questions:
    result = qa_pipeline(question=question, context=context)
    print(f"Q: {question}")
    print(f"A: {result['answer']} (score: {result['score']:.3f})\n")
```

**输出**：
```
Q: When was Hugging Face founded?
A: 2016 (score: 0.987)

Q: Who are the founders?
A: Clément Delangue, Julien Chaumond, and Thomas Wolf (score: 0.953)

Q: Where is the company based?
A: New York City and Paris (score: 0.891)

Q: How much funding did they raise in 2021?
A: $40 million (score: 0.924)
```

**输出结构详解**：
```python
{
    'score': 0.987,         # 置信度
    'start': 52,            # 答案起始位置（字符索引）
    'end': 56,              # 答案结束位置
    'answer': '2016'        # 提取的答案文本
}
```

**获取多个候选答案**：

```python
result = qa_pipeline(
    question="Who founded Hugging Face?",
    context=context,
    top_k=3  # 返回前 3 个候选答案
)

for i, ans in enumerate(result, 1):
    print(f"{i}. {ans['answer']:40} (score: {ans['score']:.3f})")
```

**输出**：
```
1. Clément Delangue, Julien Chaumond, and Thomas Wolf (score: 0.953)
2. Clément Delangue                                    (score: 0.241)
3. Julien Chaumond                                     (score: 0.187)
```

### 1.4.2 表格问答（table-question-answering）

对结构化表格进行问答：

```python
from transformers import pipeline

tqa = pipeline("table-question-answering")

table = {
    "Model": ["BERT", "GPT-2", "T5", "LLaMA"],
    "Parameters": ["110M", "1.5B", "11B", "7B"],
    "Year": ["2018", "2019", "2020", "2023"]
}

questions = [
    "Which model has the most parameters?",
    "When was BERT released?",
    "What is the size of LLaMA?"
]

for question in questions:
    result = tqa(table=table, query=question)
    print(f"Q: {question}")
    print(f"A: {result['answer']}\n")
```

### 1.4.3 文档问答（document-question-answering）

结合 OCR 和问答，处理文档图像：

```python
from transformers import pipeline

doc_qa = pipeline("document-question-answering")

# 支持图像 URL 或本地文件
image_path = "invoice.png"
question = "What is the total amount?"

result = doc_qa(image=image_path, question=question)
print(f"Answer: {result['answer']}")
```

---

## 1.5 其他常用 Pipeline

### 1.5.1 命名实体识别（NER）

```python
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)

text = "Apple was founded by Steve Jobs in Cupertino, California in 1976."
entities = ner(text)

for entity in entities:
    print(f"{entity['word']:20} → {entity['entity_group']:10} ({entity['score']:.3f})")
```

**输出**：
```
Apple                → ORG        (0.998)
Steve Jobs           → PER        (0.999)
Cupertino            → LOC        (0.995)
California           → LOC        (0.997)
1976                 → DATE       (0.985)
```

<div data-component="NERVisualizer"></div>

### 1.5.2 摘要生成（summarization）

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """
The Transformer architecture, introduced in the 2017 paper "Attention Is All You Need,"
revolutionized natural language processing. Unlike previous sequence-to-sequence models
that relied on recurrent or convolutional layers, Transformers use self-attention
mechanisms to process input sequences in parallel. This allows for much faster training
and better capture of long-range dependencies. The architecture consists of an encoder
and decoder, both made up of stacked layers of multi-head attention and feed-forward
networks. Transformers have become the foundation for modern NLP models like BERT, GPT,
and T5, achieving state-of-the-art results across numerous tasks.
"""

summary = summarizer(article, max_length=50, min_length=25, do_sample=False)
print("Summary:", summary[0]['summary_text'])
```

**输出**：
```
Summary: The Transformer architecture revolutionized NLP by using self-attention 
mechanisms instead of recurrent layers. It enables parallel processing and has 
become the foundation for modern models like BERT and GPT.
```

### 1.5.3 翻译（translation）

```python
from transformers import pipeline

# 英译法
translator_en_fr = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
result = translator_en_fr("Hello, how are you?")
print(f"FR: {result[0]['translation_text']}")

# 中译英
translator_zh_en = pipeline("translation_zh_to_en", model="Helsinki-NLP/opus-mt-zh-en")
result = translator_zh_en("你好，世界！")
print(f"EN: {result[0]['translation_text']}")
```

### 1.5.4 填空（fill-mask）

```python
from transformers import pipeline

unmasker = pipeline("fill-mask", model="bert-base-uncased")

sentence = "The capital of France is [MASK]."
results = unmasker(sentence, top_k=5)

for i, result in enumerate(results, 1):
    print(f"{i}. {result['token_str']:10} ({result['score']:.4f})")
```

**输出**：
```
1. paris      (0.8934)
2. lyon       (0.0234)
3. marseille  (0.0156)
4. nice       (0.0089)
5. toulouse   (0.0067)
```

### 1.5.5 特征提取（feature-extraction）

获取文本的向量表示（embeddings）：

```python
from transformers import pipeline
import numpy as np

feature_extractor = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

texts = [
    "The cat sits on the mat.",
    "A feline rests on a rug.",
    "The dog runs in the park."
]

# 提取特征
embeddings = feature_extractor(texts)

# 转换为 numpy 数组（取 [CLS] token 的表示）
vectors = np.array([emb[0] for emb in embeddings])
print(f"Embedding shape: {vectors.shape}")  # (3, 384)

# 计算余弦相似度
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity(vectors)

print("\nCosine Similarities:")
for i, text1 in enumerate(texts):
    for j, text2 in enumerate(texts):
        if i < j:
            print(f"{i}-{j}: {similarities[i][j]:.4f}")
```

**输出**：
```
Embedding shape: (3, 384)

Cosine Similarities:
0-1: 0.8234  # 语义相似（猫/毯子）
0-2: 0.4521  # 不太相关
1-2: 0.4312  # 不太相关
```

---

## 1.6 Pipeline 的限制与何时不用

### 1.6.1 性能瓶颈分析

<div data-component="PipelinePerformanceAnalyzer"></div>

Pipeline 的主要开销：

1. **重复加载模型**（每次调用都初始化）
2. **动态 padding**（批内序列长度不一致）
3. **单样本推理**（无法充分利用 GPU 并行）
4. **Python 循环开销**（而非向量化）

**性能对比实验**：

```python
import time
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

texts = ["This is great!"] * 100

# 方式一：Pipeline（便捷但慢）
print("=== Using Pipeline ===")
classifier = pipeline("sentiment-analysis")
start = time.time()
for text in texts:
    result = classifier(text)
time_pipeline = time.time() - start
print(f"Time: {time_pipeline:.2f}s")

# 方式二：手动批处理（快）
print("\n=== Manual Batching ===")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model.eval()

start = time.time()
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
time_batch = time.time() - start
print(f"Time: {time_batch:.2f}s")

print(f"\n⚡ Speedup: {time_pipeline / time_batch:.1f}x")
```

**输出**：
```
=== Using Pipeline ===
Time: 12.34s

=== Manual Batching ===
Time: 0.89s

⚡ Speedup: 13.9x
```

### 1.6.2 批处理的必要性

Pipeline 支持批处理，但需要显式指定：

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", batch_size=32)

texts = ["Great!"] * 1000

# 自动分批处理（每批 32 条）
results = classifier(texts)
```

### 1.6.3 转向底层 API 的时机

**应该使用 Pipeline**：
- ✅ 快速原型开发
- ✅ 单次推理或小批量
- ✅ 演示 / Jupyter Notebook
- ✅ 不在乎性能（吞吐量 < 10 QPS）

**应该使用底层 API**：
- ✅ 生产环境部署
- ✅ 需要批处理优化
- ✅ 自定义 post-processing
- ✅ 高吞吐量需求（> 100 QPS）
- ✅ 分布式训练 / 推理

---

## 1.7 总结与实战练习

### 知识回顾

✅ **掌握了**：
- Pipeline 三阶段架构（Tokenization → Model → Post-processing）
- 5+ 种核心任务的使用方法
- 生成参数调优（temperature、top_k、top_p、num_beams）
- 性能优化方向（批处理、底层 API）

### 实战练习

**练习 1：多任务 Pipeline 整合**
编写一个脚本，对同一段新闻文本：
1. 提取命名实体（NER）
2. 生成摘要
3. 判断情感倾向
4. 翻译为另一种语言

**练习 2：生成参数实验**
使用 GPT-2 生成故事开头，尝试至少 5 种参数组合，对比输出质量：
- 贪婪解码
- 束搜索（num_beams=5）
- 采样（temperature=0.7）
- Top-K 采样（top_k=50）
- Top-P 采样（top_p=0.9）

**练习 3：性能优化**
实现一个批处理情感分析脚本，处理 10,000 条文本，对比：
- Pipeline 逐条处理
- Pipeline 批处理（batch_size=32）
- 手动批处理

### 思考题

❓ **为什么 temperature=0 等价于贪婪解码？**  
💡 提示：观察 softmax 在温度趋近 0 时的行为

❓ **Top-K 和 Top-P 可以同时使用吗？效果如何？**  
💡 提示：查看源码中的采样逻辑

❓ **Pipeline 的 `device_map="auto"` 是如何分配层到不同 GPU 的？**  
💡 提示：考虑模型大小、GPU 显存、层间通信开销

### 扩展阅读

📖 **官方文档**：
- [Pipeline 完整 API 文档](https://huggingface.co/docs/transformers/main_classes/pipelines)
- [生成策略详解](https://huggingface.co/docs/transformers/generation_strategies)
- [任务指南](https://huggingface.co/docs/transformers/task_summary)

📄 **重要论文**：
- The Curious Case of Neural Text Degeneration (Holtzman et al., 2019) - Top-P Sampling
- Hierarchical Neural Story Generation (Fan et al., 2018) - 生成策略

🎥 **视频教程**：
- [Hugging Face Course - Pipelines](https://huggingface.co/learn/nlp-course/chapter1/3)

---

**下一章预告**：Chapter 2 将深入 Tokenization 机制，学习 WordPiece、BPE、SentencePiece 等算法，理解 Fast Tokenizer 的优势，掌握处理长文本、多语言、特殊场景的技巧。


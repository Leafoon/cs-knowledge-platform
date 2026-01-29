---
title: "Chapter 26. 多模态模型（Vision-Language Models）"
description: "学习 CLIP、BLIP、LLaVA 等视觉-语言模型、ViT 图像编码器、Whisper 语音识别"
updated: "2026-01-22"
---

前面的章节主要聚焦于纯文本模型。本章将探索**多模态（Multimodal）**领域，学习如何使用 Hugging Face Transformers 处理**图像+文本、音频+文本**等跨模态任务。我们将深入研究 CLIP、BLIP、LLaVA 等视觉-语言模型、ViT（Vision Transformer）图像编码器、Stable Diffusion 文本生成图像、Whisper 语音识别等前沿技术。

---

## 26.1 多模态架构概览

多模态模型的核心思想是将**不同模态**（vision、text、audio）的数据映射到**共享的表示空间**，从而实现跨模态理解和生成。

### 26.1.1 CLIP（对比学习）

**CLIP**（Contrastive Language-Image Pre-training，OpenAI 2021）通过对比学习在大规模图像-文本对上训练，学习统一的视觉-语言表示。

<div data-component="MultimodalArchitecture"></div>

**核心设计**：
- **双塔架构**（Two-Tower）：
  - **Image Encoder**：Vision Transformer（ViT）或 ResNet
  - **Text Encoder**：Transformer（类似 BERT）
- **对比损失**（InfoNCE）：
  $$
  \mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j) / \tau)}
  $$
  - 正样本对：匹配的图像-文本 $(I_i, T_i)$
  - 负样本对：batch 内其他图像-文本
  - 温度参数 $\tau$ 控制分布平滑度

**Hugging Face 使用**：
```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests

# 加载模型
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 准备数据
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

# 预处理
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# 前向传播
outputs = model(**inputs)

# 计算相似度
logits_per_image = outputs.logits_per_image  # (1, 3)
probs = logits_per_image.softmax(dim=1)  # (1, 3)

print("Label probabilities:")
for text, prob in zip(texts, probs[0]):
    print(f"{text}: {prob.item():.4f}")
```

**输出示例**：
```
Label probabilities:
a photo of a cat: 0.9921
a photo of a dog: 0.0065
a photo of a car: 0.0014
```

**应用场景**：
- **Zero-shot 图像分类**：无需训练即可分类
- **图像检索**：根据文本描述搜索图像
- **文本检索**：根据图像查找相关文本

### 26.1.2 BLIP / BLIP-2（视觉问答）

**BLIP**（Bootstrapping Language-Image Pre-training，Salesforce 2022）引入了**多任务统一框架**，支持图像描述、VQA、检索等任务。

**架构创新**：
1. **编码器-解码器架构**（Encoder-Decoder）：
   - **Image Encoder**：ViT
   - **Text Encoder**：BERT-like
   - **Text Decoder**：GPT-like（用于生成）
2. **三种训练目标**：
   - **ITC**（Image-Text Contrastive）：对比学习
   - **ITM**（Image-Text Matching）：二分类（匹配/不匹配）
   - **LM**（Language Modeling）：图像条件下的文本生成

**BLIP-2 改进**：
- **Q-Former**（Query Transformer）：轻量级模块桥接冻结的图像编码器和 LLM
- **两阶段训练**：
  1. 表示学习（从冻结的图像编码器学习）
  2. 生成学习（与冻结的 LLM 对齐）

**使用示例（图像描述）**：
```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# 加载模型
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 加载图像
image = Image.open("beach.jpg")

# 无条件生成（自动描述）
inputs = processor(image, return_tensors="pt")
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0], skip_special_tokens=True)
print(f"Caption: {caption}")

# 有条件生成（问答）
question = "What is on the beach?"
inputs = processor(image, question, return_tensors="pt")
outputs = model.generate(**inputs)
answer = processor.decode(outputs[0], skip_special_tokens=True)
print(f"Answer: {answer}")
```

### 26.1.3 LLaVA（大语言模型 + 视觉）

**LLaVA**（Large Language and Vision Assistant，2023）将预训练的 **ViT** 和 **LLaMA/Vicuna** 通过简单的**线性投影层**连接。

**架构**：
```
Image → ViT → Linear Projection → LLM (LLaMA/Vicuna) → Text
```

**训练流程**：
1. **预训练阶段**：只训练投影层（冻结 ViT 和 LLM）
   - 数据：图像-描述对（CC3M 等）
   - 目标：对齐视觉和语言特征
2. **指令微调阶段**：训练投影层 + LLM
   - 数据：多模态指令数据（GPT-4 生成）
   - 目标：提升对话能力

**使用示例**：
```python
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image

# 加载模型
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# 准备对话
image = Image.open("example.jpg")
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in this image?"}
        ]
    }
]

# 应用聊天模板
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(images=image, text=prompt, return_tensors="pt")

# 生成回复
outputs = model.generate(**inputs, max_new_tokens=200)
response = processor.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 26.1.4 Flamingo / IDEFICS

**Flamingo**（DeepMind 2022）支持**交错的图像-文本输入**（interleaved），适合多轮对话。

**IDEFICS**（HuggingFace 开源复现版）：
- 基于 Flamingo 架构
- 支持多图像输入
- 开放权重（80B 参数版本）

**使用示例**：
```python
from transformers import IdeficsForVisionText2Text, AutoProcessor

model = IdeficsForVisionText2Text.from_pretrained("HuggingFaceM4/idefics-9b")
processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics-9b")

# 多图像输入
prompts = [
    "User: What is in this image?",
    "<image>",
    "Assistant:",
    "User: And what about this one?",
    "<image>",
    "Assistant:"
]

images = [Image.open("img1.jpg"), Image.open("img2.jpg")]
inputs = processor(prompts, images=images, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

---

## 26.2 图像编码器

### 26.2.1 Vision Transformer (ViT)

**ViT**（Google 2020）将 Transformer 架构应用于图像，完全抛弃卷积操作。

**核心思想**：
1. **Patch Embedding**：将图像切分为固定大小的 patch（如 16×16）
2. **线性投影**：每个 patch 展平并通过线性层映射到嵌入维度
3. **位置编码**：添加可学习的位置嵌入
4. **Transformer Encoder**：标准 Self-Attention + FFN
5. **分类头**：[CLS] token 输出用于分类

<div data-component="VisionEncoderVisualizer"></div>

**数学表示**：
1. **Patch Embedding**：
   - 输入图像：$\mathbf{x} \in \mathbb{R}^{H \times W \times C}$
   - Patch 大小：$P \times P$
   - Patch 数量：$N = \frac{HW}{P^2}$
   - 展平：$\mathbf{x}_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$
   - 投影：$\mathbf{z}_0 = [\mathbf{x}_{\text{class}}; \mathbf{x}_p^1 \mathbf{E}; \dots; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{\text{pos}}$
   - 其中 $\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$

2. **Transformer Encoder**：
   $$
   \begin{aligned}
   \mathbf{z}'_\ell &= \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1} \\
   \mathbf{z}_\ell &= \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell
   \end{aligned}
   $$

3. **分类头**：
   $$
   \mathbf{y} = \text{LN}(\mathbf{z}_L^0)
   $$

**代码实现**：
```python
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

# 加载模型
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

# 预处理
image = Image.open("cat.jpg")
inputs = processor(images=image, return_tensors="pt")

# 前向传播
outputs = model(**inputs)
logits = outputs.logits

# 预测
predicted_class_idx = logits.argmax(-1).item()
print(f"Predicted class: {model.config.id2label[predicted_class_idx]}")
```

**架构变种**：
- **ViT-B/16**：Base 模型，16×16 patch（86M 参数）
- **ViT-L/16**：Large 模型（307M 参数）
- **ViT-H/14**：Huge 模型，14×14 patch（632M 参数）
- **DeiT**（Data-efficient ViT）：蒸馏训练，适合小数据集

### 26.2.2 CLIP Vision Encoder

CLIP 使用 ViT 作为视觉编码器，但有以下改进：
- **全局平均池化**：不使用 [CLS] token，而是对所有 patch 取平均
- **对比学习目标**：与文本编码器联合训练
- **更大的训练数据**：4亿图像-文本对

**提取特征**：
```python
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("example.jpg")
inputs = processor(images=image, return_tensors="pt")

# 提取图像特征
image_features = model.get_image_features(**inputs)  # (1, 512)
print(f"Image feature shape: {image_features.shape}")

# L2 归一化
image_features = image_features / image_features.norm(dim=-1, keepdim=True)
```

### 26.2.3 特征提取与对齐

**特征对齐目标**：
- 将图像特征 $\mathbf{v} \in \mathbb{R}^{d_v}$ 映射到语言空间 $\mathbb{R}^{d_t}$
- 常用方法：
  1. **线性投影**：$\mathbf{v}' = \mathbf{W}_v \mathbf{v}$
  2. **MLP**：$\mathbf{v}' = \text{MLP}(\mathbf{v})$
  3. **Q-Former**（BLIP-2）：Transformer 模块

**自定义特征提取器**：
```python
import torch
import torch.nn as nn

class VisionLanguageProjector(nn.Module):
    def __init__(self, vision_dim=768, text_dim=768, hidden_dim=1024):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, text_dim)
        )
    
    def forward(self, vision_features):
        """
        Args:
            vision_features: (batch, vision_dim)
        Returns:
            aligned_features: (batch, text_dim)
        """
        return self.projection(vision_features)

# 使用
projector = VisionLanguageProjector(vision_dim=512, text_dim=768)
vision_feat = torch.randn(4, 512)
aligned_feat = projector(vision_feat)  # (4, 768)
```

---

## 26.3 视觉问答微调

### 26.3.1 数据集（VQAv2、GQA）

**VQAv2**（Visual Question Answering v2）：
- 图像：COCO 数据集
- 问题：每张图像 3 个问题
- 答案：每个问题 10 个人工标注答案
- 总计：~1M 问答对

**GQA**（Visual Reasoning）：
- 强调推理能力（spatial、logical）
- 结构化场景图（Scene Graph）
- 22M 问答对

**数据加载**：
```python
from datasets import load_dataset

# 加载 VQAv2
dataset = load_dataset("HuggingFaceM4/VQAv2", split="train")

# 查看样例
sample = dataset[0]
print(f"Image: {sample['image']}")
print(f"Question: {sample['question']}")
print(f"Answers: {sample['answers']}")  # 多个答案
```

### 26.3.2 Processor（图像 + 文本预处理）

**Processor** 统一处理图像和文本输入：

```python
from transformers import BlipProcessor

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

# 处理单个样本
image = Image.open("image.jpg")
question = "What is in the image?"
inputs = processor(images=image, text=question, return_tensors="pt")

# inputs 包含：
# - pixel_values: (1, 3, 384, 384)
# - input_ids: (1, seq_len)
# - attention_mask: (1, seq_len)
```

**批量处理**：
```python
def preprocess_function(examples):
    """
    批量预处理函数
    """
    images = [img.convert("RGB") for img in examples["image"]]
    questions = examples["question"]
    
    # 处理输入
    inputs = processor(
        images=images,
        text=questions,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # 处理答案（取最常见答案）
    answers = [ans[0] for ans in examples["answers"]]
    targets = processor.tokenizer(
        answers,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )
    
    inputs["labels"] = targets["input_ids"]
    return inputs

# 应用到数据集
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names
)
```

### 26.3.3 训练与评估

**完整训练流程**：
```python
from transformers import BlipForQuestionAnswering, Trainer, TrainingArguments

# 1. 加载模型
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# 2. 训练参数
training_args = TrainingArguments(
    output_dir="./blip-vqa-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    warmup_steps=500,
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    load_best_model_at_end=True,
    fp16=True  # 混合精度
)

# 3. 自定义评估指标
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # 解码预测和标签
    pred_tokens = predictions.argmax(-1)
    pred_str = processor.batch_decode(pred_tokens, skip_special_tokens=True)
    label_str = processor.batch_decode(labels, skip_special_tokens=True)
    
    # 计算准确率（精确匹配）
    exact_match = sum(p.strip().lower() == l.strip().lower() 
                     for p, l in zip(pred_str, label_str)) / len(pred_str)
    
    return {"exact_match": exact_match}

# 4. 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics
)

# 5. 训练
trainer.train()

# 6. 推理
def answer_question(image_path, question):
    image = Image.open(image_path)
    inputs = processor(image, question, return_tensors="pt")
    
    outputs = model.generate(**inputs, max_length=50)
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    
    return answer

# 测试
answer = answer_question("test.jpg", "How many people are in the image?")
print(f"Answer: {answer}")
```

---

## 26.4 图像生成（Diffusion）

### 26.4.1 Stable Diffusion 与 Transformers

**Stable Diffusion** 使用扩散模型（Diffusion Model）从文本生成图像，Hugging Face 提供了完整的 Pipeline。

**架构组件**：
1. **Text Encoder**：CLIP Text Encoder（将文本转换为条件）
2. **UNet**：去噪网络（核心）
3. **VAE**（Variational AutoEncoder）：将像素空间压缩到潜在空间
4. **Scheduler**：控制去噪步数

**基础使用**：
```python
from diffusers import StableDiffusionPipeline
import torch

# 加载模型
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# 生成图像
prompt = "a beautiful sunset over mountains, highly detailed, 4k"
image = pipe(
    prompt,
    num_inference_steps=50,  # 去噪步数
    guidance_scale=7.5,      # CFG（Classifier-Free Guidance）强度
    height=512,
    width=512
).images[0]

image.save("generated_image.png")
```

### 26.4.2 Text-to-Image Pipeline

**完整流程**：
1. **文本编码**：
   ```python
   text_embeddings = pipe.text_encoder(text_input_ids)
   ```
2. **初始化噪声**：
   ```python
   latents = torch.randn((batch_size, 4, 64, 64))  # 潜在空间
   ```
3. **去噪循环**：
   ```python
   for t in pipe.scheduler.timesteps:
       # 预测噪声
       noise_pred = pipe.unet(latents, t, text_embeddings).sample
       
       # 更新 latents
       latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
   ```
4. **解码到像素空间**：
   ```python
   image = pipe.vae.decode(latents / 0.18215).sample
   ```

**高级参数**：
```python
image = pipe(
    prompt="a cyberpunk city at night",
    negative_prompt="blurry, low quality, ugly",  # 负提示词
    num_inference_steps=100,  # 更多步数 → 更高质量
    guidance_scale=9.0,       # 更高 CFG → 更符合提示词
    generator=torch.Generator("cuda").manual_seed(42)  # 固定随机种子
).images[0]
```

### 26.4.3 ControlNet 集成

**ControlNet** 允许使用额外的条件（如边缘图、深度图）控制生成过程。

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import cv2
import numpy as np

# 1. 加载 ControlNet
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

# 2. 创建 Pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# 3. 准备控制图像（Canny 边缘检测）
original_image = load_image("input.jpg")
image_array = np.array(original_image)
edges = cv2.Canny(image_array, 100, 200)
edges = Image.fromarray(edges)

# 4. 生成
output = pipe(
    prompt="a beautiful painting of a house",
    image=edges,  # 控制条件
    num_inference_steps=50
).images[0]

output.save("controlled_output.png")
```

**常用 ControlNet 类型**：
- **Canny**：边缘检测
- **Depth**：深度图
- **Pose**：人体姿态（OpenPose）
- **Scribble**：涂鸦
- **Seg**：语义分割

---

## 26.5 音频模型

### 26.5.1 Whisper（语音识别）

**Whisper**（OpenAI 2022）是一个强大的多语言语音识别模型，支持**转录**和**翻译**。

**架构**：
- **Encoder-Decoder Transformer**
- 训练数据：680k 小时多语言音频

**基础使用**：
```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# 加载模型
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# 加载音频
audio, sr = librosa.load("audio.mp3", sr=16000)  # 重采样到 16kHz

# 预处理
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

# 生成转录
generated_ids = model.generate(inputs["input_features"])
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"Transcription: {transcription}")
```

**多语言支持**：
```python
# 指定语言（中文）
forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")
generated_ids = model.generate(
    inputs["input_features"],
    forced_decoder_ids=forced_decoder_ids
)
```

**翻译到英文**：
```python
forced_decoder_ids = processor.get_decoder_prompt_ids(language="zh", task="translate")
generated_ids = model.generate(
    inputs["input_features"],
    forced_decoder_ids=forced_decoder_ids
)
translation = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Translation: {translation}")
```

**Pipeline 使用**：
```python
from transformers import pipeline

# 自动语音识别 Pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium",
    chunk_length_s=30,  # 长音频分块
    device=0  # GPU
)

# 转录
result = pipe("long_audio.wav")
print(result["text"])

# 带时间戳
result = pipe("audio.wav", return_timestamps=True)
for chunk in result["chunks"]:
    print(f"[{chunk['timestamp'][0]:.2f}s - {chunk['timestamp'][1]:.2f}s]: {chunk['text']}")
```

### 26.5.2 Wav2Vec2（自监督学习）

**Wav2Vec2**（Meta 2020）使用自监督学习从未标注音频中学习表示。

**预训练目标**：
- **Masked Prediction**：类似 BERT，遮蔽部分音频片段并预测

**微调用于 ASR**：
```python
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

# 加载模型
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# 加载音频
audio, sr = librosa.load("audio.wav", sr=16000)

# 预处理
inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

# 前向传播
with torch.no_grad():
    logits = model(inputs.input_values).logits

# CTC 解码
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]

print(f"Transcription: {transcription}")
```

**微调自定义数据**：
```python
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("common_voice", "zh-CN", split="train")

# 预处理函数
def prepare_dataset(batch):
    audio = batch["audio"]
    
    # 处理音频
    batch["input_values"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_values[0]
    
    # 处理文本标签
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    
    return batch

# 应用预处理
dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

# 训练
training_args = TrainingArguments(
    output_dir="./wav2vec2-zh-CN",
    per_device_train_batch_size=8,
    learning_rate=1e-4,
    num_train_epochs=5,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
```

### 26.5.3 音频分类与转录

**音频分类示例（情感识别）**：
```python
from transformers import pipeline

# 音频分类 Pipeline
classifier = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er"
)

# 预测情感
result = classifier("happy_speech.wav")
print(result)
# [{'label': 'hap', 'score': 0.85}, {'label': 'neu', 'score': 0.10}, ...]
```

**实时转录（流式处理）**：
```python
import pyaudio
import numpy as np

# 初始化音频流
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=1024
)

print("🎤 Start speaking...")

while True:
    # 读取音频块
    audio_chunk = stream.read(1024)
    audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
    
    # 处理
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
    
    # 推理
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    
    # 解码
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    if transcription.strip():
        print(f"Transcription: {transcription}")
```

---

## 26.6 实战案例：构建图像问答系统

结合所学知识，构建一个完整的图像问答 Web 应用。

```python
import gradio as gr
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image

# 加载模型
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

def answer_image_question(image, question):
    """
    图像问答函数
    
    Args:
        image: PIL Image
        question: str
    
    Returns:
        answer: str
    """
    # 预处理
    inputs = processor(images=image, text=question, return_tensors="pt")
    
    # 生成答案
    outputs = model.generate(**inputs, max_length=50)
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    
    return answer

# 创建 Gradio 界面
iface = gr.Interface(
    fn=answer_image_question,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Ask a Question", placeholder="What is in the image?")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="🖼️ Image Question Answering with BLIP",
    description="Upload an image and ask questions about it!",
    examples=[
        ["example1.jpg", "What is the main object?"],
        ["example2.jpg", "How many people are there?"],
        ["example3.jpg", "What color is the car?"]
    ]
)

# 启动
iface.launch(share=True)
```

---

## 26.7 性能优化与部署

### 1. **量化加速**

```python
from transformers import BitsAndBytesConfig

# 4-bit 量化
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = BlipForQuestionAnswering.from_pretrained(
    "Salesforce/blip-vqa-base",
    quantization_config=quantization_config,
    device_map="auto"
)
```

### 2. **批量推理**

```python
def batch_inference(images, questions, batch_size=8):
    """批量处理图像问答"""
    results = []
    
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        batch_questions = questions[i:i+batch_size]
        
        inputs = processor(
            images=batch_images,
            text=batch_questions,
            return_tensors="pt",
            padding=True
        )
        
        outputs = model.generate(**inputs, max_length=50)
        answers = processor.batch_decode(outputs, skip_special_tokens=True)
        
        results.extend(answers)
    
    return results
```

### 3. **TorchScript 导出**

```python
# 导出为 TorchScript（仅支持部分模型）
traced_model = torch.jit.trace(model, example_inputs)
traced_model.save("blip_vqa.pt")

# 加载
loaded_model = torch.jit.load("blip_vqa.pt")
```

---

## 26.8 章节总结

本章我们深入学习了多模态模型的核心技术：

✅ **核心技能**：
- 理解 CLIP 对比学习架构（图像-文本对齐）
- 使用 BLIP/LLaVA 进行视觉问答和图像描述
- 掌握 ViT（Vision Transformer）图像编码原理
- 使用 Stable Diffusion 生成图像（Text-to-Image）
- 使用 Whisper 进行多语言语音识别
- 微调 Wav2Vec2 进行 ASR 任务

✅ **实战能力**：
- 构建图像问答系统（BLIP VQA）
- ControlNet 条件图像生成
- 实时语音转录
- 多模态特征对齐

✅ **最佳实践**：
- Processor 统一处理多模态输入
- 使用 Pipeline 简化推理流程
- 量化加速（4-bit）降低显存
- 批量推理提升吞吐量

**下一章预告**：Chapter 27 将学习**强化学习与 RLHF**，包括 InstructGPT 的三阶段训练流程（SFT → RM → PPO）、TRL 库使用、DPO（Direct Preference Optimization）、以及实战指令微调 LLaMA。

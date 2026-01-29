---
title: "Chapter 23. Attention 机制深度剖析 - 理解 Transformers 核心"
description: "从数学推导到工程实现，深入理解 Self-Attention、Multi-Head Attention 机制"
updated: "2026-01-22"
---

## 23.1 Self-Attention 数学推导

### 23.1.1 从零开始理解 Attention

**核心思想**：在处理序列中的某个位置时，Attention 允许模型**动态地关注序列中其他位置的信息**。

**问题引入**：给定输入序列 $X = [x_1, x_2, ..., x_n]$，如何计算每个位置的表示，使其包含上下文信息？

---

### 23.1.2 Query、Key、Value 矩阵

**三个线性变换**：

$$
\begin{aligned}
Q &= XW^Q \in \mathbb{R}^{n \times d_k} \quad \text{(Query: 我在找什么)} \\
K &= XW^K \in \mathbb{R}^{n \times d_k} \quad \text{(Key: 我能提供什么)} \\
V &= XW^V \in \mathbb{R}^{n \times d_v} \quad \text{(Value: 实际内容)}
\end{aligned}
$$

其中：
- $X \in \mathbb{R}^{n \times d_{\text{model}}}$：输入序列（$n$ 个 token，维度 $d_{\text{model}}$）
- $W^Q, W^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$：Query 和 Key 的投影矩阵
- $W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$：Value 的投影矩阵

**直观理解**：
- **Query**：当前位置想要查询的"问题"
- **Key**：每个位置提供的"标签"
- **Value**：每个位置的实际"内容"

---

### 23.1.3 Scaled Dot-Product Attention 公式

**完整公式**：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**逐步分解**：

1. **计算相似度**（$QK^T \in \mathbb{R}^{n \times n}$）：
   $$
   S = QK^T = \begin{bmatrix}
   q_1 \cdot k_1 & q_1 \cdot k_2 & \cdots & q_1 \cdot k_n \\
   q_2 \cdot k_1 & q_2 \cdot k_2 & \cdots & q_2 \cdot k_n \\
   \vdots & \vdots & \ddots & \vdots \\
   q_n \cdot k_1 & q_n \cdot k_2 & \cdots & q_n \cdot k_n
   \end{bmatrix}
   $$
   
   每个元素 $S_{ij} = q_i \cdot k_j$ 表示位置 $i$ 对位置 $j$ 的**原始关注度**。

2. **缩放**（除以 $\sqrt{d_k}$）：
   $$
   \tilde{S} = \frac{QK^T}{\sqrt{d_k}}
   $$
   
   **为什么要缩放**？当 $d_k$ 很大时，点积值会很大，导致 softmax 梯度消失。

   **数学分析**：假设 $q, k$ 的每个元素独立同分布，均值 0，方差 1，则：
   $$
   \mathbb{E}[q \cdot k] = 0, \quad \text{Var}(q \cdot k) = d_k
   $$
   
   除以 $\sqrt{d_k}$ 后方差归一化为 1。

3. **Softmax 归一化**：
   $$
   A = \text{softmax}(\tilde{S}) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
   $$
   
   对每一行应用 softmax：
   $$
   A_{ij} = \frac{\exp(\tilde{S}_{ij})}{\sum_{j'=1}^{n} \exp(\tilde{S}_{ij'})}
   $$
   
   $A_{ij}$ 是位置 $i$ 对位置 $j$ 的**注意力权重**，满足 $\sum_j A_{ij} = 1$。

4. **加权求和**：
   $$
   \text{Output} = AV \in \mathbb{R}^{n \times d_v}
   $$
   
   每个位置的输出是 Value 向量的加权和：
   $$
   \text{output}_i = \sum_{j=1}^{n} A_{ij} v_j
   $$

---

### 23.1.4 代码实现（从零开始）

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    参数:
        Q: [batch_size, n_heads, seq_len, d_k]
        K: [batch_size, n_heads, seq_len, d_k]
        V: [batch_size, n_heads, seq_len, d_v]
        mask: [batch_size, 1, seq_len, seq_len] (可选)
    
    返回:
        output: [batch_size, n_heads, seq_len, d_v]
        attention_weights: [batch_size, n_heads, seq_len, seq_len]
    """
    d_k = Q.size(-1)
    
    # 1. 计算 Q * K^T
    scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, H, L, L]
    
    # 2. 缩放
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # 3. 应用 Mask（如果有）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 4. Softmax
    attention_weights = F.softmax(scores, dim=-1)  # [B, H, L, L]
    
    # 5. 加权求和
    output = torch.matmul(attention_weights, V)  # [B, H, L, d_v]
    
    return output, attention_weights

# 示例
batch_size = 2
n_heads = 8
seq_len = 10
d_k = 64

Q = torch.randn(batch_size, n_heads, seq_len, d_k)
K = torch.randn(batch_size, n_heads, seq_len, d_k)
V = torch.randn(batch_size, n_heads, seq_len, d_k)

output, attn_weights = scaled_dot_product_attention(Q, K, V)

print(f"Output shape: {output.shape}")  # [2, 8, 10, 64]
print(f"Attention weights shape: {attn_weights.shape}")  # [2, 8, 10, 10]
print(f"Attention weights sum: {attn_weights[0, 0, 0].sum()}")  # 1.0
```

**输出示例**：
```
Output shape: torch.Size([2, 8, 10, 64])
Attention weights shape: torch.Size([2, 8, 10, 10])
Attention weights sum: tensor(1.0000)
```

---

### 23.1.5 可视化示例

<div data-component="AttentionWeightHeatmap"></div>

**示例句子**："The cat sat on the mat"

```python
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)

text = "The cat sat on the mat"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# 获取第一层的注意力权重
attention = outputs.attentions[0]  # [1, 12, seq_len, seq_len]

# 查看第一个 head 对 "cat" 的注意力分布
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print(f"Tokens: {tokens}")

cat_idx = tokens.index("cat")
cat_attention = attention[0, 0, cat_idx].numpy()

for i, (token, weight) in enumerate(zip(tokens, cat_attention)):
    print(f"{token:10s} {weight:.4f} {'█' * int(weight * 50)}")
```

**输出示例**：
```
Tokens: ['[CLS]', 'the', 'cat', 'sat', 'on', 'the', 'mat', '[SEP]']
[CLS]      0.0523 ██
the        0.1245 ██████
cat        0.3421 █████████████████
sat        0.2134 ██████████
on         0.0892 ████
the        0.1103 █████
mat        0.0621 ███
[SEP]      0.0061 ▌
```

---

## 23.2 Multi-Head Attention

### 23.2.1 为什么需要多头？

**单头 Attention 的局限性**：
- 只能学习一种关注模式（例如：语法依赖）
- 难以捕捉多种语义关系（例如：共指消解、实体识别）

**多头 Attention 的优势**：
- 不同 head 学习不同的表示子空间
- 并行计算，提高表达能力
- 类似 CNN 的多通道

---

### 23.2.2 数学公式

**多头 Attention**：

$$
\begin{aligned}
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \text{head}_2, ..., \text{head}_h)W^O \\
\text{head}_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\end{aligned}
$$

其中：
- $h$：头数（通常 8 或 12）
- $W_i^Q, W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$：第 $i$ 个 head 的投影矩阵
- $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$
- $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$：输出投影矩阵

**维度设置**：
- $d_k = d_v = d_{\text{model}} / h$
- 例如：$d_{\text{model}} = 768$，$h = 12$ → $d_k = 64$

---

### 23.2.3 代码实现

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, Q, K, V, mask=None):
        """
        参数:
            Q, K, V: [batch_size, seq_len, d_model]
            mask: [batch_size, 1, 1, seq_len] 或 [batch_size, 1, seq_len, seq_len]
        """
        batch_size = Q.size(0)
        
        # 1. 线性变换 + 分割成多头
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # 现在形状: [batch_size, n_heads, seq_len, d_k]
        
        # 2. Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 3. 加权求和
        context = torch.matmul(attention_weights, V)  # [B, H, L, d_k]
        
        # 4. 拼接多头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 5. 输出投影
        output = self.W_o(context)
        
        return output, attention_weights

# 测试
mha = MultiHeadAttention(d_model=512, n_heads=8)
x = torch.randn(2, 10, 512)
output, attn = mha(x, x, x)

print(f"Output shape: {output.shape}")  # [2, 10, 512]
print(f"Attention shape: {attn.shape}")  # [2, 8, 10, 10]
```

**参数量分析**：
- $W^Q, W^K, W^V, W^O$：$4 \times (d_{\text{model}} \times d_{\text{model}}) = 4d_{\text{model}}^2$
- BERT-base（$d_{\text{model}} = 768$）：$4 \times 768^2 = 2,359,296$ 参数/层

---

### 23.2.4 不同 Head 的作用分析

**实验观察**（BERT-base）：

| Head | 主要关注模式 | 示例 |
|------|------------|------|
| Head 0 | 局部依赖 | 相邻词 |
| Head 1 | 句法关系 | 主谓宾 |
| Head 2 | 位置编码 | 对角线模式 |
| Head 5 | 共指消解 | "he" → "John" |
| Head 8 | 全局聚合 | 关注 [CLS] |

**可视化工具**：
- BertViz: https://github.com/jessevig/bertviz
- Attention Flows

---

## 23.3 Attention Mask 深度剖析

### 23.3.1 Padding Mask（编码器）

**目的**：忽略填充的 token（[PAD]）

**实现**：

```python
def create_padding_mask(seq):
    """
    参数:
        seq: [batch_size, seq_len] (token IDs)
    返回:
        mask: [batch_size, 1, 1, seq_len]
    """
    # PAD token ID = 0
    mask = (seq != 0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
    return mask

# 示例
seq = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
mask = create_padding_mask(seq)
print(mask[0, 0, 0])  # tensor([True, True, True, False, False])
```

**应用到 Attention**：

```python
scores = scores.masked_fill(mask == 0, -1e9)
# Softmax 后，-1e9 → 0
```

---

### 23.3.2 Causal Mask（解码器）

**目的**：防止看到未来信息（自回归生成）

<div data-component="MaskBuilder"></div>

**实现**：

```python
def create_causal_mask(seq_len):
    """
    返回:
        mask: [1, 1, seq_len, seq_len] (下三角矩阵)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    return mask  # [1, 1, L, L]

# 示例
mask = create_causal_mask(5)
print(mask[0, 0])
# tensor([[1., 0., 0., 0., 0.],
#         [1., 1., 0., 0., 0.],
#         [1., 1., 1., 0., 0.],
#         [1., 1., 1., 1., 0.],
#         [1., 1., 1., 1., 1.]])
```

**可视化**：

```
Position 0 can attend to: [0]
Position 1 can attend to: [0, 1]
Position 2 can attend to: [0, 1, 2]
Position 3 can attend to: [0, 1, 2, 3]
Position 4 can attend to: [0, 1, 2, 3, 4]
```

---

### 23.3.3 Combined Mask

**编码器-解码器 Attention**：同时需要 Padding + Causal Mask

```python
def create_combined_mask(tgt_seq, src_seq):
    """
    参数:
        tgt_seq: [batch_size, tgt_len] (目标序列)
        src_seq: [batch_size, src_len] (源序列)
    """
    # 1. Causal Mask for target
    tgt_len = tgt_seq.size(1)
    causal_mask = torch.tril(torch.ones(tgt_len, tgt_len)).unsqueeze(0).unsqueeze(1)
    
    # 2. Padding Mask for target
    tgt_padding_mask = (tgt_seq != 0).unsqueeze(1).unsqueeze(2)
    
    # 3. 组合
    combined_mask = causal_mask & tgt_padding_mask
    
    return combined_mask  # [B, 1, tgt_len, tgt_len]

# 示例
tgt_seq = torch.tensor([[1, 2, 3, 0, 0]])
src_seq = torch.tensor([[4, 5, 6, 7, 0]])
mask = create_combined_mask(tgt_seq, src_seq)
print(mask[0, 0])
```

---

### 23.3.4 Transformers 库中的 Mask

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "The quick brown"
inputs = tokenizer(text, return_tensors="pt")

# 自动创建 Causal Mask
outputs = model(**inputs, output_attentions=True)

# 提取 Attention Mask
attention_mask = inputs["attention_mask"]  # Padding Mask
print(f"Attention Mask: {attention_mask}")

# Causal Mask 内部自动应用（GPT-2）
```

---

## 23.4 Position Encoding 变种

### 23.4.1 绝对位置编码

**Sinusoidal Position Encoding**（Transformer 原始论文）：

$$
\begin{aligned}
PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\end{aligned}
$$

其中：
- $pos$：位置索引（0, 1, 2, ...）
- $i$：维度索引（0, 1, ..., $d_{\text{model}}/2 - 1$）

**代码实现**：

```python
import torch
import math

def sinusoidal_position_encoding(max_len=5000, d_model=512):
    """
    返回:
        pe: [max_len, d_model]
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

# 可视化
import matplotlib.pyplot as plt
pe = sinusoidal_position_encoding(100, 128)
plt.figure(figsize=(10, 6))
plt.imshow(pe.numpy(), cmap='RdBu', aspect='auto')
plt.colorbar()
plt.xlabel('Dimension')
plt.ylabel('Position')
plt.title('Sinusoidal Position Encoding')
```

**优点**：
- 无需学习参数
- 可以外推到更长序列

---

### 23.4.2 学习式位置编码

**BERT 使用的方法**：

```python
class LearnedPositionEmbedding(nn.Module):
    def __init__(self, max_len=512, d_model=768):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)
    
    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        return self.position_embeddings(position_ids)
```

**优点**：
- 灵活，可以学习任意模式
- 通常性能更好

**缺点**：
- 无法外推到 > max_len 的序列

---

### 23.4.3 旋转位置编码（RoPE）

<div data-component="PositionEncodingVisualizer"></div>

**LLaMA、Mistral 使用的方法**：

**核心思想**：在 Query 和 Key 上应用旋转矩阵，使得点积包含相对位置信息。

**2D 情况的旋转**：

$$
\begin{bmatrix}
q_0' \\ q_1'
\end{bmatrix} = \begin{bmatrix}
\cos(m\theta) & -\sin(m\theta) \\
\sin(m\theta) & \cos(m\theta)
\end{bmatrix} \begin{bmatrix}
q_0 \\ q_1
\end{bmatrix}
$$

其中 $m$ 是位置，$\theta = 10000^{-2i/d}$。

**高维扩展**：分组应用 2D 旋转。

**代码实现**（简化）：

```python
def apply_rotary_pos_emb(q, k, position_ids):
    """
    RoPE 实现
    """
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
    
    # 计算旋转角度
    inv_freq = 1.0 / (10000 ** (torch.arange(0, q.shape[-1], 2).float() / q.shape[-1]))
    freqs = torch.outer(position_ids.float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    
    cos_emb = emb.cos()
    sin_emb = emb.sin()
    
    # 应用旋转
    q_embed = (q * cos_emb) + (rotate_half(q) * sin_emb)
    k_embed = (k * cos_emb) + (rotate_half(k) * sin_emb)
    
    return q_embed, k_embed
```

**优点**：
- 相对位置信息
- 可以外推到更长序列（通过插值）
- 不增加参数

---

### 23.4.4 ALiBi（Attention with Linear Biases）

**核心思想**：在 Attention 分数上直接添加线性偏置。

$$
\text{softmax}(q_i K^T + m \cdot [0, -1, -2, ..., -(i-1)])
$$

其中 $m$ 是每个 head 的斜率（例如：$2^{-8}, 2^{-9}, ..., 2^{-15}$）。

**代码**：

```python
def get_alibi_biases(n_heads, seq_len):
    """
    返回 ALiBi bias matrix
    """
    # 计算斜率
    slopes = torch.tensor([2 ** (-8 - i) for i in range(n_heads)])
    
    # 位置偏置
    position_bias = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
    position_bias = position_bias.unsqueeze(0)  # [1, L, L]
    
    # 应用斜率
    alibi = position_bias * slopes.unsqueeze(1).unsqueeze(2)  # [H, L, L]
    
    return alibi

# 在 Attention 中使用
scores = scores + alibi  # 在 softmax 之前
```

**优点**：
- 极简实现
- 训练效率高
- 外推性能好

---

## 23.5 KV Cache 实现细节

### 23.5.1 为什么需要 KV Cache？

**问题**：自回归生成时，每次只生成一个 token，但需要重新计算所有历史 token 的 K 和 V。

**示例**（无 Cache）：

```python
# 生成第 1 个 token
input_ids = [101, 102]  # [CLS] + "hello"
K1, V1 = compute_kv(input_ids)  # 计算 2 个 token 的 K, V

# 生成第 2 个 token
input_ids = [101, 102, 103]  # + "world"
K2, V2 = compute_kv(input_ids)  # 重新计算 3 个 token 的 K, V（浪费！）

# 生成第 3 个 token
input_ids = [101, 102, 103, 104]
K3, V3 = compute_kv(input_ids)  # 重新计算 4 个 token...
```

**计算量**：$O(n^2)$（$n$ 是序列长度）

---

### 23.5.2 KV Cache 原理

<div data-component="KVCacheDynamics"></div>

**核心思想**：缓存已计算的 K 和 V，每次只计算新 token 的 K 和 V。

**实现**：

```python
def generate_with_kv_cache(model, input_ids, max_new_tokens=10):
    """
    使用 KV Cache 生成
    """
    past_key_values = None  # 初始化 Cache
    
    for _ in range(max_new_tokens):
        # 第一次：计算所有 token
        # 之后：只计算最后一个 token
        if past_key_values is None:
            outputs = model(input_ids, use_cache=True)
        else:
            outputs = model(input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
        
        # 更新 Cache
        past_key_values = outputs.past_key_values
        
        # 采样下一个 token
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        # 拼接
        input_ids = torch.cat([input_ids, next_token], dim=1)
    
    return input_ids

# 使用示例
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

input_ids = tokenizer("Hello", return_tensors="pt").input_ids
output_ids = generate_with_kv_cache(model, input_ids, max_new_tokens=20)

print(tokenizer.decode(output_ids[0]))
```

---

### 23.5.3 Cache 结构

**past_key_values** 是一个**嵌套元组**：

```python
past_key_values = (
    # Layer 0
    (
        key_cache_layer_0,    # [batch, n_heads, seq_len, d_k]
        value_cache_layer_0,  # [batch, n_heads, seq_len, d_v]
    ),
    # Layer 1
    (
        key_cache_layer_1,
        value_cache_layer_1,
    ),
    ...
    # Layer N-1
)
```

**更新过程**：

```python
# 第 1 次调用（prompt）
input_ids = [1, 2, 3]
K_new = compute_K(input_ids)  # [B, H, 3, d_k]
V_new = compute_V(input_ids)  # [B, H, 3, d_v]

past_key_values = (K_new, V_new)

# 第 2 次调用（生成第 1 个 token）
input_ids = [4]
K_new = compute_K([4])  # [B, H, 1, d_k]
V_new = compute_V([4])  # [B, H, 1, d_v]

# 拼接
K_cached = torch.cat([past_key_values[0], K_new], dim=2)  # [B, H, 4, d_k]
V_cached = torch.cat([past_key_values[1], V_new], dim=2)  # [B, H, 4, d_v]

past_key_values = (K_cached, V_cached)
```

---

### 23.5.4 内存优化：PagedAttention

**问题**：KV Cache 占用大量内存，且不同序列长度不同，导致碎片化。

**PagedAttention**（vLLM 使用）：
- 将 KV Cache 分页存储（类似虚拟内存）
- 动态分配和回收
- 支持序列间共享（beam search）

**内存节约**：
- 无 Cache：每次重新计算 → 慢
- 传统 Cache：预分配固定大小 → 浪费
- PagedAttention：按需分配 → 高效

**效果**：
- 吞吐量提升 2-4x
- 内存利用率提升 ~50%

---

## 23.6 Cross-Attention（编码器-解码器）

### 23.6.1 与 Self-Attention 的区别

| 类型 | Query 来源 | Key/Value 来源 | 应用场景 |
|-----|----------|--------------|---------|
| Self-Attention | 同一序列 | 同一序列 | BERT, GPT |
| Cross-Attention | 解码器 | 编码器 | T5, BART, Whisper |

---

### 23.6.2 Cross-Attention 公式

$$
\text{CrossAttention}(Q_{dec}, K_{enc}, V_{enc}) = \text{softmax}\left(\frac{Q_{dec}K_{enc}^T}{\sqrt{d_k}}\right)V_{enc}
$$

**关键点**：
- $Q$ 来自解码器当前层
- $K, V$ 来自编码器最后一层输出

---

### 23.6.3 代码实现（T5 风格）

```python
class EncoderDecoderAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
    
    def forward(self, decoder_hidden, encoder_output, encoder_mask=None):
        """
        参数:
            decoder_hidden: [B, tgt_len, d_model] (解码器隐藏状态)
            encoder_output: [B, src_len, d_model] (编码器输出)
            encoder_mask: [B, 1, 1, src_len] (源序列 Padding Mask)
        """
        # Query from decoder, Key/Value from encoder
        output, attn_weights = self.mha(
            Q=decoder_hidden,
            K=encoder_output,
            V=encoder_output,
            mask=encoder_mask
        )
        
        return output, attn_weights

# 示例
enc_dec_attn = EncoderDecoderAttention(d_model=512, n_heads=8)

encoder_output = torch.randn(2, 15, 512)  # 源序列长度 15
decoder_hidden = torch.randn(2, 10, 512)  # 目标序列长度 10

output, attn = enc_dec_attn(decoder_hidden, encoder_output)

print(f"Output shape: {output.shape}")  # [2, 10, 512]
print(f"Cross-Attention weights shape: {attn.shape}")  # [2, 8, 10, 15]
```

---

### 23.6.4 应用示例：机器翻译

**模型**：T5、mBART、MarianMT

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# 翻译任务
input_text = "translate English to German: The house is wonderful."
inputs = tokenizer(input_text, return_tensors="pt")

# 生成
outputs = model.generate(**inputs, output_attentions=True, return_dict_in_generate=True)

# 提取 Cross-Attention（解码器关注编码器）
cross_attentions = outputs.cross_attentions  # Tuple of tuples

# 可视化
# cross_attentions[step][layer] 是一个 [B, H, 1, src_len] 的张量
```

---

## 23.7 实战：从零实现 Transformer Block

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        
        # Multi-Head Attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer Normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        参数:
            x: [batch_size, seq_len, d_model]
            mask: Attention mask
        """
        # 1. Self-Attention + Residual + LayerNorm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.ln1(x + self.dropout(attn_output))
        
        # 2. Feed-Forward + Residual + LayerNorm
        ffn_output = self.ffn(x)
        x = self.ln2(x + ffn_output)
        
        return x

# 测试
block = TransformerBlock(d_model=512, n_heads=8)
x = torch.randn(2, 10, 512)
output = block(x)

print(f"Output shape: {output.shape}")  # [2, 10, 512]
```

---

## 23.8 总结

### 核心公式回顾

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

---

### 关键洞察

1. **Scaled Dot-Product**：
   - $\sqrt{d_k}$ 缩放稳定梯度
   - Softmax 归一化为概率分布
   - 加权求和聚合信息

2. **Multi-Head**：
   - 学习多种关注模式
   - 参数量：$4d_{\text{model}}^2$
   - 并行计算

3. **Mask**：
   - Padding Mask：忽略 [PAD]
   - Causal Mask：自回归生成
   - Combined Mask：编码器-解码器

4. **Position Encoding**：
   - Sinusoidal：外推性好
   - Learned：性能优
   - RoPE：相对位置 + 外推
   - ALiBi：简洁 + 外推

5. **KV Cache**：
   - 避免重复计算
   - 内存换时间
   - PagedAttention 优化

---

## 23.9 扩展阅读

1. **Attention Is All You Need**（原始论文）：https://arxiv.org/abs/1706.03762
2. **The Illustrated Transformer**：https://jalammar.github.io/illustrated-transformer/
3. **RoFormer: Enhanced Transformer with Rotary Position Embedding**：https://arxiv.org/abs/2104.09864
4. **Train Short, Test Long: Attention with Linear Biases (ALiBi)**：https://arxiv.org/abs/2108.12409
5. **PagedAttention (vLLM)**：https://arxiv.org/abs/2309.06180

# Chapter 28: 前沿研究与未来方向

> **本章目标**：探索 Transformers 领域的最新研究方向，了解长上下文建模、高效架构、模型合并、可解释性、安全性等前沿主题，展望未来发展趋势。

---

## 28.1 长上下文建模

随着应用需求的增长，处理超长文本序列成为关键挑战。标准 Transformer 的二次复杂度限制了其上下文窗口长度。

### 28.1.1 位置插值（Position Interpolation）

位置插值通过缩放位置编码，让模型处理比训练时更长的序列。

**核心思想**：
- 将位置索引从 `[0, L']` 线性映射到 `[0, L]`（L 是训练长度，L' > L）
- RoPE（Rotary Position Embedding）天然支持插值

**实现示例**：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# LLaMA 模型支持位置插值扩展
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    trust_remote_code=True
)

# 通过修改 config 扩展上下文
model.config.max_position_embeddings = 8192  # 从 4096 扩展
model.config.rope_scaling = {
    "type": "linear",
    "factor": 2.0  # 插值因子
}

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 测试超长输入
long_text = "This is a very long document. " * 1000
inputs = tokenizer(long_text, return_tensors="pt", truncation=False)

# 生成时可以处理更长序列
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        use_cache=True
    )
```

**NTK-Aware Interpolation**（神经切线核感知插值）：

```python
# 更高级的插值策略
model.config.rope_scaling = {
    "type": "ntk",
    "factor": 2.0,
    "alpha": 1.0  # NTK 参数
}
```

**数学原理**：

对于 RoPE，位置 $m$ 的编码为：
$$
\text{RoPE}(x_m, m) = x_m \cdot e^{im\theta}
$$

线性插值：
$$
m' = \frac{m}{s}, \quad s = \frac{L'}{L}
$$

NTK 插值调整频率：
$$
\theta' = \theta \cdot s^{-\alpha}
$$

### 28.1.2 ALiBi 与 RoPE 扩展

**ALiBi（Attention with Linear Biases）**：

不使用位置编码，而是在注意力分数上直接添加线性偏置。

```python
# ALiBi 实现示例（简化版）
import torch

def get_alibi_slopes(num_heads):
    """计算 ALiBi 的斜率"""
    def get_slopes_power_of_2(n):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]
    
    if math.log2(num_heads).is_integer():
        return get_slopes_power_of_2(num_heads)
    else:
        # 处理非 2 的幂次
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        return get_slopes_power_of_2(closest_power_of_2) + \
               get_alibi_slopes(2 * closest_power_of_2)[0::2][:num_heads - closest_power_of_2]

def apply_alibi(attention_scores, num_heads):
    """应用 ALiBi 偏置"""
    batch_size, seq_len = attention_scores.shape[0], attention_scores.shape[-1]
    
    # 计算相对距离
    position_ids = torch.arange(seq_len, device=attention_scores.device)
    relative_positions = position_ids[None, :] - position_ids[:, None]  # (seq_len, seq_len)
    
    # 获取斜率
    slopes = torch.tensor(get_alibi_slopes(num_heads), device=attention_scores.device)
    
    # 计算偏置
    alibi = slopes[:, None, None] * relative_positions[None, :, :]  # (num_heads, seq_len, seq_len)
    
    # 添加到注意力分数
    attention_scores = attention_scores + alibi
    return attention_scores
```

**ALiBi 的优势**：
- 零外推能力：训练时短，推理时长
- 无需位置编码参数
- 对不同长度泛化良好

**使用 ALiBi 的模型**：

```python
from transformers import AutoModelForCausalLM

# BLOOM 使用 ALiBi
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")

# 可以直接处理比训练时更长的序列
long_input = tokenizer("A " * 5000, return_tensors="pt")
outputs = model(**long_input)
```

### 28.1.3 Sparse Attention（Longformer、BigBird）

稀疏注意力通过限制注意力范围，将复杂度从 $O(n^2)$ 降低到 $O(n)$。

**Longformer 的三种注意力模式**：

1. **局部窗口注意力**（Sliding Window）
2. **全局注意力**（Global Attention）
3. **扩张窗口注意力**（Dilated Window）

```python
from transformers import LongformerModel, LongformerTokenizer

# 加载 Longformer
model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

# 准备长文本
text = "This is a long document. " * 500
inputs = tokenizer(text, return_tensors="pt", max_length=4096, truncation=True)

# 设置全局注意力（第一个 token 通常是 [CLS]）
global_attention_mask = torch.zeros_like(inputs["input_ids"])
global_attention_mask[:, 0] = 1  # [CLS] token 有全局注意力

# 前向传播
outputs = model(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    global_attention_mask=global_attention_mask
)

# 输出 shape: (batch_size, sequence_length, hidden_size)
print(outputs.last_hidden_state.shape)  # torch.Size([1, 4096, 768])
```

**窗口大小配置**：

```python
# 自定义窗口大小
from transformers import LongformerConfig

config = LongformerConfig.from_pretrained("allenai/longformer-base-4096")
config.attention_window = [256] * 12  # 每层 256 的窗口
model = LongformerModel(config)
```

**BigBird 的随机注意力**：

BigBird 结合了局部、全局和随机注意力。

```python
from transformers import BigBirdModel, BigBirdTokenizer

model = BigBirdModel.from_pretrained("google/bigbird-roberta-base")
tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")

# BigBird 支持最长 4096 tokens
inputs = tokenizer("Long text " * 500, return_tensors="pt", max_length=4096, truncation=True)

outputs = model(**inputs)
```

<div data-component="LongContextStrategies"></div>

### 28.1.4 Retrieval-Augmented Generation (RAG)

RAG 通过检索外部知识库扩展上下文，避免将所有信息编码到参数中。

**架构**：
1. **检索器**（Retriever）：从知识库检索相关文档
2. **生成器**（Generator）：基于检索结果生成答案

```python
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import torch

# 加载 RAG 模型
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq",
    index_name="exact",
    use_dummy_dataset=True  # 演示用，实际需要真实索引
)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)

# 问题
question = "What is the capital of France?"
inputs = tokenizer(question, return_tensors="pt")

# 生成答案
generated = model.generate(**inputs, num_return_sequences=1, max_new_tokens=50)
answer = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
print(f"Answer: {answer}")
```

**自定义检索器**（使用 FAISS）：

```python
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import faiss
import numpy as np

# 1. 准备文档嵌入
doc_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
doc_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

documents = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Madrid is the capital of Spain."
]

# 编码文档
doc_inputs = doc_tokenizer(documents, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    doc_embeddings = doc_encoder(**doc_inputs).pooler_output.numpy()

# 2. 构建 FAISS 索引
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner Product
index.add(doc_embeddings)

# 3. 检索查询
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

question = "What is the capital of France?"
q_inputs = question_tokenizer(question, return_tensors="pt")
with torch.no_grad():
    q_embedding = question_encoder(**q_inputs).pooler_output.numpy()

# 搜索 top-k
k = 2
scores, indices = index.search(q_embedding, k)
retrieved_docs = [documents[i] for i in indices[0]]

print(f"Retrieved: {retrieved_docs}")
# ['Paris is the capital of France.', 'Berlin is the capital of Germany.']

# 4. 组合检索结果 + 生成
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

generator = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")
gen_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

# 拼接上下文
context = " ".join(retrieved_docs)
input_text = f"question: {question} context: {context}"
inputs = gen_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

outputs = generator.generate(**inputs, max_new_tokens=50)
answer = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Answer: {answer}")
```

---

## 28.2 高效架构

### 28.2.1 Mixture of Experts (MoE)

MoE 通过稀疏激活专家网络，大幅增加模型容量而不增加计算量。

**核心概念**：
- **专家层**（Expert Layer）：多个并行的 FFN
- **路由器**（Router/Gate）：决定每个 token 激活哪些专家
- **Top-K 路由**：每个 token 只激活 K 个专家

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Mixtral 8x7B 是著名的 MoE 模型
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-v0.1",
    device_map="auto",
    load_in_4bit=True,  # 量化加载
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

# 正常使用
inputs = tokenizer("Once upon a time", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

**自定义 MoE 层**（简化版）：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts=8, top_k=2, expert_hidden_size=2048):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 路由器
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, expert_hidden_size),
                nn.GELU(),
                nn.Linear(expert_hidden_size, hidden_size)
            )
            for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # x: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)  # (batch_size * seq_len, hidden_size)
        
        # 路由决策
        router_logits = self.gate(x_flat)  # (batch_size * seq_len, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-K 选择
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # 重新归一化
        
        # 初始化输出
        output = torch.zeros_like(x_flat)
        
        # 分发到专家
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]  # (batch_size * seq_len,)
            weight = top_k_probs[:, i:i+1]  # (batch_size * seq_len, 1)
            
            # 为每个专家收集对应的 token
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += weight[mask] * expert_output
        
        return output.view(batch_size, seq_len, hidden_size)

# 使用示例
moe_layer = MoELayer(hidden_size=768, num_experts=8, top_k=2)
x = torch.randn(2, 128, 768)  # (batch, seq_len, hidden_size)
output = moe_layer(x)
print(output.shape)  # torch.Size([2, 128, 768])
```

**负载均衡损失**（Load Balancing Loss）：

```python
def load_balancing_loss(router_probs, num_experts):
    """
    确保专家被均匀使用
    """
    # router_probs: (batch_size * seq_len, num_experts)
    expert_usage = router_probs.mean(dim=0)  # (num_experts,)
    target = torch.ones_like(expert_usage) / num_experts
    loss = F.mse_loss(expert_usage, target)
    return loss
```

<div data-component="MoERouting"></div>

### 28.2.2 State Space Models（Mamba、S4）

状态空间模型（SSM）是 Transformer 的替代架构，具有线性复杂度。

**Mamba 核心思想**：
- 选择性状态空间（Selective SSM）
- 硬件感知算法（Hardware-Aware Algorithm）
- 线性时间推理

```python
# Mamba 模型使用示例（需要安装 mamba-ssm）
# pip install mamba-ssm

from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载 Mamba 模型
model = AutoModelForCausalLM.from_pretrained(
    "state-spaces/mamba-2.8b",
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b")

# 生成
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

**S4（Structured State Space）层简化实现**：

```python
import torch
import torch.nn as nn

class S4Layer(nn.Module):
    """简化的 S4 层（教学用）"""
    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # 状态空间参数（简化为可学习矩阵）
        self.A = nn.Parameter(torch.randn(d_state, d_state) / d_state)
        self.B = nn.Parameter(torch.randn(d_state, d_model) / d_model)
        self.C = nn.Parameter(torch.randn(d_model, d_state) / d_state)
        self.D = nn.Parameter(torch.randn(d_model))
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch, seq_len, _ = x.shape
        
        # 初始化状态
        h = torch.zeros(batch, self.d_state, device=x.device)
        outputs = []
        
        # 递归计算（实际 S4 使用并行扫描）
        for t in range(seq_len):
            u = x[:, t, :]  # (batch, d_model)
            h = torch.matmul(h, self.A.T) + torch.matmul(u, self.B.T)  # (batch, d_state)
            y = torch.matmul(h, self.C.T) + u * self.D  # (batch, d_model)
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)

# 测试
layer = S4Layer(d_model=256, d_state=64)
x = torch.randn(4, 100, 256)
output = layer(x)
print(output.shape)  # torch.Size([4, 100, 256])
```

**S4 vs Transformer 对比**：

| 特性 | Transformer | S4/Mamba |
|------|------------|----------|
| 时间复杂度 | $O(n^2 d)$ | $O(nd)$ |
| 推理速度 | 慢（KV cache） | 快（状态固定） |
| 长序列处理 | 受限 | 擅长 |
| 训练效率 | 高（并行） | 中（需并行扫描） |

### 28.2.3 RetNet（Retentive Networks）

RetNet 结合了 Transformer 和 RNN 的优点。

**三种计算模式**：
1. **并行模式**（训练）
2. **循环模式**（推理）
3. **分块循环模式**（长序列）

```python
# RetNet 概念实现（简化）
import torch
import torch.nn as nn

class RetentionLayer(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Decay 参数
        self.gamma = nn.Parameter(torch.ones(num_heads) * 0.9)
        
    def parallel_forward(self, x):
        """并行模式（训练时）"""
        batch, seq_len, _ = x.shape
        
        Q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        
        # Decay 矩阵
        positions = torch.arange(seq_len, device=x.device)
        decay = self.gamma[:, None, None] ** (positions[None, :, None] - positions[None, None, :])
        decay = decay.tril()  # 因果掩码
        
        # Retention 分数
        scores = torch.einsum('bqhd,bkhd->bhqk', Q, K) * decay
        output = torch.einsum('bhqk,bkhd->bqhd', scores, V)
        
        output = output.reshape(batch, seq_len, self.d_model)
        return self.out_proj(output)
    
    def recurrent_forward(self, x_t, state):
        """循环模式（推理时）"""
        # x_t: (batch, d_model) - 单个时间步
        # state: (batch, num_heads, head_dim, head_dim)
        batch = x_t.shape[0]
        
        q_t = self.q_proj(x_t).view(batch, self.num_heads, self.head_dim)
        k_t = self.k_proj(x_t).view(batch, self.num_heads, self.head_dim)
        v_t = self.v_proj(x_t).view(batch, self.num_heads, self.head_dim)
        
        # 更新状态
        new_state = self.gamma[:, None, None] * state + \
                    torch.einsum('bhd,bhe->bhde', k_t, v_t)
        
        # 计算输出
        output = torch.einsum('bhd,bhde->bhe', q_t, new_state)
        output = output.reshape(batch, self.d_model)
        
        return self.out_proj(output), new_state

# 使用示例
retention = RetentionLayer(d_model=512, num_heads=8)

# 训练模式
x = torch.randn(2, 100, 512)
output_parallel = retention.parallel_forward(x)
print(output_parallel.shape)  # torch.Size([2, 100, 512])

# 推理模式（逐 token）
state = torch.zeros(2, 8, 64, 64)  # (batch, num_heads, head_dim, head_dim)
for t in range(10):
    x_t = torch.randn(2, 512)
    output_t, state = retention.recurrent_forward(x_t, state)
    print(f"Step {t}: {output_t.shape}")  # torch.Size([2, 512])
```

### 28.2.4 RWKV（RNN-like Transformer）

RWKV 使用线性注意力机制，实现 RNN 的效率和 Transformer 的并行性。

```python
# RWKV 使用示例
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "RWKV/rwkv-4-169m-pile",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-4-169m-pile")

prompt = "In a shocking finding, scientists discovered"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

---

## 28.3 模型合并与组合

### 28.3.1 Model Merging（SLERP、TIES）

模型合并通过融合多个模型的权重，创建新模型。

**线性插值（Linear Interpolation）**：

```python
import torch
from transformers import AutoModelForCausalLM

# 加载两个模型
model_a = AutoModelForCausalLM.from_pretrained("gpt2")
model_b = AutoModelForCausalLM.from_pretrained("gpt2-medium")

# 线性插值（简化版 - 需要相同架构）
def linear_merge(model_a, model_b, alpha=0.5):
    """alpha=0.5 表示 50/50 混合"""
    merged_state_dict = {}
    
    for key in model_a.state_dict().keys():
        if key in model_b.state_dict():
            merged_state_dict[key] = \
                alpha * model_a.state_dict()[key] + \
                (1 - alpha) * model_b.state_dict()[key]
    
    return merged_state_dict

# 执行合并
merged_weights = linear_merge(model_a, model_b, alpha=0.5)

# 加载到新模型
merged_model = AutoModelForCausalLM.from_pretrained("gpt2")
merged_model.load_state_dict(merged_weights, strict=False)
```

**SLERP（Spherical Linear Interpolation）**：

```python
def slerp_merge(model_a, model_b, alpha=0.5):
    """球面线性插值"""
    import torch.nn.functional as F
    
    merged_state_dict = {}
    
    for key in model_a.state_dict().keys():
        if key in model_b.state_dict():
            w_a = model_a.state_dict()[key]
            w_b = model_b.state_dict()[key]
            
            # 计算夹角
            dot = (w_a * w_b).sum() / (w_a.norm() * w_b.norm())
            dot = torch.clamp(dot, -1.0, 1.0)
            theta = torch.acos(dot)
            
            # SLERP 公式
            if theta.abs() < 1e-6:
                # 近似线性
                merged_state_dict[key] = alpha * w_a + (1 - alpha) * w_b
            else:
                merged_state_dict[key] = \
                    (torch.sin((1 - alpha) * theta) / torch.sin(theta)) * w_a + \
                    (torch.sin(alpha * theta) / torch.sin(theta)) * w_b
    
    return merged_state_dict
```

**TIES（Trim, Elect, and Merge）**：

```python
def ties_merge(models, threshold=0.2):
    """
    TIES 合并算法：
    1. Trim: 移除小权重
    2. Elect: 选择符号一致的权重
    3. Merge: 平均合并
    """
    merged_state_dict = {}
    
    for key in models[0].state_dict().keys():
        weights = [model.state_dict()[key] for model in models]
        
        # 1. Trim - 移除接近零的权重
        trimmed = [w * (w.abs() > threshold) for w in weights]
        
        # 2. Elect - 选择符号一致的权重
        signs = torch.stack([w.sign() for w in trimmed])
        majority_sign = signs.sum(dim=0).sign()
        
        elected = [w * (w.sign() == majority_sign) for w in trimmed]
        
        # 3. Merge - 平均
        merged_state_dict[key] = sum(elected) / len(elected)
    
    return merged_state_dict
```

**使用 mergekit 库**：

```bash
# 安装
pip install mergekit

# 配置文件 merge_config.yaml
# models:
#   - model: model_a
#     weight: 0.5
#   - model: model_b
#     weight: 0.5
# merge_method: slerp

# 执行合并
mergekit-yaml merge_config.yaml output_dir
```

### 28.3.2 LoRA 适配器组合

多个 LoRA 适配器可以动态组合，实现多任务能力。

```python
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 训练多个 LoRA 适配器（示例）
# adapter_1: 数学任务
# adapter_2: 代码任务
# adapter_3: 写作任务

# 加载适配器 1
model_with_adapter1 = PeftModel.from_pretrained(base_model, "path/to/math_adapter")

# 切换到适配器 2
model_with_adapter1.load_adapter("path/to/code_adapter", adapter_name="code")
model_with_adapter1.set_adapter("code")

# 同时使用多个适配器（加权组合）
model_with_adapter1.load_adapter("path/to/writing_adapter", adapter_name="writing")

# 设置权重
model_with_adapter1.set_adapter(["code", "writing"])  # 自动平均
# 或手动设置权重
model_with_adapter1.set_adapter_weights({"code": 0.7, "writing": 0.3})

# 生成
inputs = tokenizer("Write a Python function", return_tensors="pt")
outputs = model_with_adapter1.generate(**inputs, max_new_tokens=100)
```

**动态适配器路由**：

```python
class AdapterRouter:
    def __init__(self, model, adapters):
        self.model = model
        self.adapters = adapters  # {"math": path, "code": path, ...}
        self.classifier = self.train_classifier()  # 任务分类器
    
    def route_and_generate(self, prompt):
        # 1. 分类任务
        task = self.classify_task(prompt)
        
        # 2. 加载对应适配器
        self.model.set_adapter(task)
        
        # 3. 生成
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        return tokenizer.decode(outputs[0])
    
    def classify_task(self, prompt):
        # 使用小型分类器判断任务类型
        # 这里简化为关键词匹配
        if "code" in prompt.lower() or "python" in prompt.lower():
            return "code"
        elif "math" in prompt.lower() or "equation" in prompt.lower():
            return "math"
        else:
            return "writing"
```

### 28.3.3 Ensemble 方法

集成多个模型的预测结果。

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 加载多个模型
models = [
    AutoModelForSequenceClassification.from_pretrained("bert-base-uncased"),
    AutoModelForSequenceClassification.from_pretrained("roberta-base"),
    AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
]
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def ensemble_predict(text, models, method="voting"):
    """
    method: "voting" | "averaging" | "stacking"
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # 收集所有模型的预测
    all_logits = []
    for model in models:
        with torch.no_grad():
            outputs = model(**inputs)
            all_logits.append(outputs.logits)
    
    if method == "voting":
        # 硬投票
        predictions = [logits.argmax(dim=-1) for logits in all_logits]
        final_pred = torch.mode(torch.stack(predictions), dim=0).values
    
    elif method == "averaging":
        # 概率平均
        all_probs = [torch.softmax(logits, dim=-1) for logits in all_logits]
        avg_probs = torch.stack(all_probs).mean(dim=0)
        final_pred = avg_probs.argmax(dim=-1)
    
    elif method == "stacking":
        # 元学习器（这里简化为加权平均）
        weights = torch.tensor([0.4, 0.35, 0.25])  # 基于验证集性能
        weighted_logits = sum(w * logits for w, logits in zip(weights, all_logits))
        final_pred = weighted_logits.argmax(dim=-1)
    
    return final_pred

# 测试
text = "This movie is absolutely fantastic!"
prediction = ensemble_predict(text, models, method="averaging")
print(f"Ensemble prediction: {prediction}")
```

---

## 28.4 可解释性与分析

### 28.4.1 Attention 可视化（BertViz）

可视化注意力权重，理解模型关注的内容。

```python
# 安装 bertviz
# pip install bertviz

from bertviz import head_view, model_view
from transformers import AutoTokenizer, AutoModel

# 加载模型
model_name = "bert-base-uncased"
model = AutoModel.from_pretrained(model_name, output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 准备输入
text = "The cat sat on the mat"
inputs = tokenizer(text, return_tensors="pt")

# 前向传播
outputs = model(**inputs)
attention = outputs.attentions  # Tuple of (batch, num_heads, seq_len, seq_len)

# 可视化注意力头
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
head_view(attention, tokens)

# 可视化所有层
model_view(attention, tokens)
```

**自定义注意力分析**：

```python
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention_heatmap(attention_weights, tokens, layer=0, head=0):
    """
    绘制注意力热力图
    """
    # attention_weights: (num_layers, batch, num_heads, seq_len, seq_len)
    attn = attention_weights[layer][0, head].detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        cbar=True
    )
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    plt.title(f'Attention Heatmap - Layer {layer}, Head {head}')
    plt.tight_layout()
    plt.show()

# 使用
visualize_attention_heatmap(attention, tokens, layer=5, head=3)
```

<div data-component="AttentionPatternAnalyzer"></div>

### 28.4.2 探针分类（Probing）

通过探针任务评估模型的中间表示编码了哪些信息。

```python
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

# 1. 提取模型表示
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def extract_representations(texts, layer=-1):
    """提取指定层的表示"""
    representations = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # 取 [CLS] token 的表示
            hidden_states = outputs.hidden_states[layer][:, 0, :]
            representations.append(hidden_states.squeeze().numpy())
    
    return torch.tensor(representations)

# 2. 探针任务：句子长度分类
sentences = [
    "Short.",
    "A bit longer sentence.",
    "This is an even longer sentence with more words.",
    "Cat.",
    "The quick brown fox jumps over the lazy dog.",
]
labels = [0, 1, 2, 0, 2]  # 0: short, 1: medium, 2: long

# 提取表示
X = extract_representations(sentences, layer=6)  # 探测第 6 层
y = torch.tensor(labels)

# 训练探针分类器
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X.numpy(), y.numpy(), test_size=0.3, random_state=42
)

probe = LogisticRegression(max_iter=1000)
probe.fit(X_train, y_train)

# 评估
y_pred = probe.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Probing accuracy: {accuracy:.2f}")

# 3. 探测不同层
layer_scores = []
for layer in range(model.config.num_hidden_layers):
    X_layer = extract_representations(sentences, layer=layer)
    probe.fit(X_layer[:-2].numpy(), y[:-2].numpy())  # 简化 train/test split
    score = probe.score(X_layer[-2:].numpy(), y[-2:].numpy())
    layer_scores.append(score)

# 绘制每层的探测准确率
plt.plot(layer_scores)
plt.xlabel('Layer')
plt.ylabel('Probing Accuracy')
plt.title('Sentence Length Information Across Layers')
plt.show()
```

**常见探针任务**：
- 句法信息（POS tagging、依存关系）
- 语义信息（命名实体、情感）
- 句子长度
- 词序信息

### 28.4.3 激活值分析

分析神经元的激活模式。

```python
def analyze_neuron_activation(model, texts, layer_idx=6, neuron_idx=100):
    """分析特定神经元对不同输入的激活"""
    activations = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_state = outputs.hidden_states[layer_idx]
            # 取特定神经元的激活（平均所有 token）
            activation = hidden_state[0, :, neuron_idx].mean().item()
            activations.append(activation)
    
    return activations

# 测试不同类型的句子
positive_sentences = ["I love this!", "Amazing!", "Great work!"]
negative_sentences = ["I hate this.", "Terrible.", "Worst ever."]

pos_activations = analyze_neuron_activation(model, positive_sentences, layer_idx=6, neuron_idx=100)
neg_activations = analyze_neuron_activation(model, negative_sentences, layer_idx=6, neuron_idx=100)

print(f"Positive avg: {sum(pos_activations)/len(pos_activations):.3f}")
print(f"Negative avg: {sum(neg_activations)/len(neg_activations):.3f}")
```

### 28.4.4 因果干预实验

通过干预模型内部表示，测试因果关系。

```python
def causal_intervention(model, text, layer_idx, position, intervention_vector):
    """
    在指定位置插入干预向量
    """
    inputs = tokenizer(text, return_tensors="pt")
    
    # 定义 hook 函数
    def intervention_hook(module, input, output):
        # output: (batch, seq_len, hidden_size)
        output[0, position, :] = intervention_vector
        return output
    
    # 注册 hook
    layer = model.encoder.layer[layer_idx]
    handle = layer.register_forward_hook(intervention_hook)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 移除 hook
    handle.remove()
    
    return outputs

# 示例：改变特定位置的表示，观察对输出的影响
original_output = model(**tokenizer("The cat is happy", return_tensors="pt"))
intervened_output = causal_intervention(
    model,
    "The cat is happy",
    layer_idx=6,
    position=3,  # "happy" 的位置
    intervention_vector=torch.randn(768)  # 随机干预
)

print(f"Output difference: {(original_output.last_hidden_state - intervened_output.last_hidden_state).abs().mean()}")
```

---

## 28.5 安全性与对齐

### 28.5.1 对抗攻击与防御

**文本对抗攻击**（TextFooler）：

```python
# 安装 TextAttack
# pip install textattack

from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import HuggingFaceDataset

# 包装模型
model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2")
tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

# 创建攻击
attack = TextFoolerJin2019.build(model_wrapper)

# 测试
from textattack.datasets import Dataset

dataset = [("This movie is great!", 1)]  # (text, label)
attack_results = attack.attack_dataset(dataset)

for result in attack_results:
    print(f"Original: {result.original_text()}")
    print(f"Adversarial: {result.perturbed_text()}")
    print(f"Success: {result.goal_status}")
```

**对抗训练**：

```python
from textattack.augmentation import Augmenter

# 创建增强器（生成对抗样本）
augmenter = Augmenter(
    transformation="word_swap_embedding",
    constraints=["repeat", "stopword"],
    pct_words_to_swap=0.1
)

# 增强训练数据
original_text = "This movie is absolutely fantastic!"
augmented_texts = augmenter.augment(original_text, num_augmentations=5)

print("Original:", original_text)
for i, aug_text in enumerate(augmented_texts):
    print(f"Augmented {i+1}:", aug_text)

# 将对抗样本加入训练集
train_texts = [original_text] + augmented_texts
train_labels = [1] * len(train_texts)  # 保持相同标签

# 正常训练流程
# trainer.train()
```

### 28.5.2 有害内容检测

```python
from transformers import pipeline

# 使用专门的毒性检测模型
toxicity_classifier = pipeline(
    "text-classification",
    model="unitary/toxic-bert"
)

texts = [
    "You are such a wonderful person!",
    "I hate you and wish you were dead.",  # 有害内容
    "This is a normal sentence."
]

results = toxicity_classifier(texts)
for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Label: {result['label']}, Score: {result['score']:.3f}\n")
```

**内容过滤器**：

```python
class ContentFilter:
    def __init__(self, toxicity_threshold=0.7):
        self.detector = pipeline("text-classification", model="unitary/toxic-bert")
        self.threshold = toxicity_threshold
    
    def is_safe(self, text):
        result = self.detector(text)[0]
        is_toxic = result['label'] == 'toxic' and result['score'] > self.threshold
        return not is_toxic
    
    def filter_generation(self, model, tokenizer, prompt, **gen_kwargs):
        """生成并过滤"""
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # 生成多个候选
        outputs = model.generate(
            **inputs,
            num_return_sequences=5,
            do_sample=True,
            **gen_kwargs
        )
        
        # 过滤安全的输出
        safe_outputs = []
        for output in outputs:
            text = tokenizer.decode(output, skip_special_tokens=True)
            if self.is_safe(text):
                safe_outputs.append(text)
        
        return safe_outputs if safe_outputs else ["[Content filtered]"]

# 使用
filter = ContentFilter(toxicity_threshold=0.7)
safe_texts = filter.filter_generation(model, tokenizer, "Tell me about", max_new_tokens=50)
```

### 28.5.3 偏见评估（Bias）

```python
# 安装 lm-evaluation-harness
# pip install lm-eval

from lm_eval import evaluator

# 评估模型偏见
results = evaluator.simple_evaluate(
    model="gpt2",
    tasks=["crows_pairs_english"],  # 偏见评估任务
    num_fewshot=0
)

print(results)
```

**自定义偏见测试**：

```python
def bias_test(model, tokenizer, template_pairs):
    """
    测试模型对不同群体的偏见
    template_pairs: [("He is a {profession}", "She is a {profession}"), ...]
    """
    professions = ["doctor", "nurse", "engineer", "teacher"]
    
    results = []
    for male_template, female_template in template_pairs:
        for profession in professions:
            male_text = male_template.format(profession=profession)
            female_text = female_template.format(profession=profession)
            
            # 计算困惑度
            male_ppl = calculate_perplexity(model, tokenizer, male_text)
            female_ppl = calculate_perplexity(model, tokenizer, female_text)
            
            results.append({
                "profession": profession,
                "male_ppl": male_ppl,
                "female_ppl": female_ppl,
                "bias_score": abs(male_ppl - female_ppl)
            })
    
    return results

def calculate_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    return perplexity

# 测试
template_pairs = [
    ("He is a {profession}.", "She is a {profession}."),
    ("The {profession} finished his work.", "The {profession} finished her work.")
]

bias_results = bias_test(model, tokenizer, template_pairs)
for result in bias_results:
    print(f"{result['profession']}: Male PPL={result['male_ppl']:.2f}, "
          f"Female PPL={result['female_ppl']:.2f}, Bias={result['bias_score']:.2f}")
```

### 28.5.4 可控生成

**PPLM（Plug and Play Language Models）**：

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class PPLMGenerator:
    def __init__(self, model_name="gpt2"):
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # 属性分类器（简化示例）
        self.attribute_classifier = self.build_classifier()
    
    def build_classifier(self):
        # 实际应该是训练好的分类器
        # 这里简化为关键词匹配
        return {
            "positive": ["happy", "great", "wonderful", "amazing"],
            "negative": ["sad", "terrible", "awful", "bad"]
        }
    
    def controlled_generate(self, prompt, attribute="positive", num_iterations=10, stepsize=0.01):
        """
        受控生成
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        # 生成初始候选
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=True,
                output_scores=True,
                return_dict_in_generate=True
            )
        
        generated_ids = outputs.sequences[0]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 迭代优化（简化版）
        for _ in range(num_iterations):
            # 计算属性得分
            score = self.compute_attribute_score(generated_text, attribute)
            
            # 根据得分调整（实际 PPLM 会在潜在空间操作）
            if score < 0.5:
                # 重新生成
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        max_new_tokens=20,
                        do_sample=True,
                        temperature=1.2  # 增加多样性
                    )
                generated_ids = outputs[0]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text
    
    def compute_attribute_score(self, text, attribute):
        # 简化评分
        keywords = self.attribute_classifier[attribute]
        score = sum(1 for kw in keywords if kw in text.lower()) / len(keywords)
        return score

# 使用
generator = PPLMGenerator()
controlled_text = generator.controlled_generate(
    "The weather today is",
    attribute="positive"
)
print(controlled_text)
```

---

## 28.6 未来展望

### 28.6.1 多模态大一统模型

未来趋势：单一模型处理文本、图像、音频、视频。

**示例模型**：
- GPT-4 Vision
- Gemini
- Llama 3.2 Vision

```python
# 未来的统一 API（概念示例）
from transformers import UniversalModel

model = UniversalModel.from_pretrained("future-model-v1")

# 多模态输入
inputs = {
    "text": "Describe this image and video",
    "image": load_image("photo.jpg"),
    "video": load_video("clip.mp4"),
    "audio": load_audio("speech.wav")
}

# 多模态输出
outputs = model.generate(**inputs, output_modalities=["text", "image"])

print(outputs["text"])  # 文本描述
display(outputs["image"])  # 生成的图像
```

### 28.6.2 端到端语音对话

**全双工对话系统**：

```python
# 概念示例：实时语音到语音
from transformers import SpeechToSpeechPipeline

pipeline = SpeechToSpeechPipeline.from_pretrained("speech-llm-v1")

# 流式处理
for audio_chunk in stream_microphone():
    response_audio = pipeline(
        audio_chunk,
        streaming=True,
        low_latency=True
    )
    play_audio(response_audio)
```

### 28.6.3 世界模型（World Models）

模型不仅生成文本，还能模拟物理世界。

```python
# 概念：物理感知的语言模型
model = WorldModel.from_pretrained("physics-aware-llm")

query = "If I drop a ball from 10 meters, when will it hit the ground?"
response = model.generate(
    query,
    physical_simulation=True,
    gravity=9.8
)

print(response)  # "The ball will hit the ground in approximately 1.43 seconds."
print(response.simulation_trace)  # 物理模拟轨迹
```

### 28.6.4 AGI 路径探讨

**当前挑战**：
1. **常识推理**：模型仍缺乏真正的常识
2. **因果理解**：关联 ≠ 因果
3. **持续学习**：灾难性遗忘问题
4. **能效比**：人脑 20W vs GPU 数百 W

**研究方向**：
- 神经符号融合（Neuro-Symbolic AI）
- 元学习与少样本学习
- 多智能体协作
- 具身智能（Embodied AI）

---

## 28.7 实战：探索前沿模型

### 案例 1：使用 Mamba 处理超长序列

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载 Mamba
model = AutoModelForCausalLM.from_pretrained(
    "state-spaces/mamba-2.8b",
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b")

# 超长文本（10,000 tokens）
long_text = "The history of artificial intelligence. " * 2000
inputs = tokenizer(long_text, return_tensors="pt", truncation=False).to(model.device)

print(f"Input length: {inputs['input_ids'].shape[1]} tokens")

# Mamba 可以高效处理
import time
start = time.time()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100)
end = time.time()

print(f"Generation time: {end - start:.2f}s")
print(tokenizer.decode(outputs[0][-100:]))  # 最后 100 tokens
```

### 案例 2：模型合并实验

```python
from transformers import AutoModelForCausalLM

# 加载两个微调模型
model_math = AutoModelForCausalLM.from_pretrained("math-tuned-llama")
model_code = AutoModelForCausalLM.from_pretrained("code-tuned-llama")

# SLERP 合并
merged_model = slerp_merge(model_math, model_code, alpha=0.5)

# 保存合并模型
merged_model.save_pretrained("merged-math-code-llama")

# 测试合并效果
test_prompts = [
    "Solve: 2x + 5 = 15",  # 数学
    "Write a Python function to sort a list",  # 代码
    "Explain bubble sort algorithm"  # 两者结合
]

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = merged_model.generate(**inputs, max_new_tokens=100)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
```

### 案例 3：可解释性分析

```python
from bertviz import head_view
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("bert-base-uncased", output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 分析一个句子
sentence = "The cat sat on the mat because it was comfortable"
inputs = tokenizer(sentence, return_tensors="pt")
outputs = model(**inputs)

# 可视化注意力
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
attention = outputs.attentions

# 查看"because"如何关注其他词
head_view(attention, tokens)

# 分析指代消解
# "it" 应该关注 "mat"
it_position = tokens.index('it')
mat_position = tokens.index('mat')

for layer in range(12):
    for head in range(12):
        attn_weight = attention[layer][0, head, it_position, mat_position].item()
        if attn_weight > 0.3:  # 显著注意力
            print(f"Layer {layer}, Head {head}: it -> mat = {attn_weight:.3f}")
```

---

## 28.8 最佳实践与建议

### 跟踪前沿研究

1. **论文来源**：
   - arXiv（cs.CL, cs.LG）
   - NeurIPS, ICML, ICLR, ACL, EMNLP
   - Hugging Face Papers（https://huggingface.co/papers）

2. **开源项目**：
   - GitHub Trending（Python, Jupyter Notebook）
   - Hugging Face Spaces（演示项目）

3. **社区资源**：
   - Hugging Face Discord
   - Reddit（r/MachineLearning）
   - Twitter/X（关注研究者）

### 实验新技术

```python
# 保持代码结构灵活，便于集成新方法
class ExperimentalPipeline:
    def __init__(self, model_type="transformer"):
        self.model_type = model_type
        self.model = self.load_model()
    
    def load_model(self):
        if self.model_type == "transformer":
            return AutoModelForCausalLM.from_pretrained("gpt2")
        elif self.model_type == "mamba":
            return AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m")
        elif self.model_type == "retnet":
            return self.load_retnet_model()  # 自定义加载
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def generate(self, prompt, **kwargs):
        # 统一生成接口
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(outputs[0])

# 轻松切换不同架构
pipeline_transformer = ExperimentalPipeline("transformer")
pipeline_mamba = ExperimentalPipeline("mamba")

# 对比性能
import time

for pipeline, name in [(pipeline_transformer, "Transformer"), (pipeline_mamba, "Mamba")]:
    start = time.time()
    result = pipeline.generate("Once upon a time", max_new_tokens=100)
    end = time.time()
    print(f"{name}: {end - start:.2f}s")
```

### 负责任的 AI 实践

```python
# 集成安全检查
class SafeGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.safety_filter = ContentFilter()
        self.bias_detector = BiasDetector()
    
    def safe_generate(self, prompt, **kwargs):
        # 1. 检查输入
        if not self.safety_filter.is_safe(prompt):
            return "[Unsafe prompt detected]"
        
        # 2. 生成
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, **kwargs)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 3. 检查输出
        if not self.safety_filter.is_safe(text):
            return "[Output filtered for safety]"
        
        # 4. 偏见检测
        bias_score = self.bias_detector.detect(text)
        if bias_score > 0.8:
            return f"[Warning: High bias detected ({bias_score:.2f})] {text}"
        
        return text

# 使用
safe_gen = SafeGenerator(model, tokenizer)
output = safe_gen.safe_generate("Tell me about", max_new_tokens=50)
```

---

## 28.9 章节总结

本章探索了 Transformers 领域的前沿研究方向：

### 核心要点

1. **长上下文建模**：
   - 位置插值、ALiBi、RoPE 扩展
   - 稀疏注意力（Longformer、BigBird）
   - RAG（检索增强生成）

2. **高效架构**：
   - MoE（专家混合）
   - State Space Models（Mamba、S4）
   - RetNet、RWKV

3. **模型合并与组合**：
   - SLERP、TIES 合并算法
   - LoRA 适配器组合
   - Ensemble 方法

4. **可解释性**：
   - 注意力可视化
   - 探针分类
   - 因果干预

5. **安全性与对齐**：
   - 对抗攻击防御
   - 有害内容检测
   - 偏见评估
   - 可控生成

### 未来趋势

- 多模态大一统
- 端到端语音对话
- 世界模型
- AGI 探索

### 实践建议

- 持续关注前沿研究
- 实验新技术架构
- 负责任的 AI 开发
- 平衡性能与安全

**恭喜完成全部 28 章核心内容！** 🎉

接下来请参考附录获取：
- 常见错误调试指南
- 性能基准对比
- 资源清单
- API 速查表

---

**下一步**：[Appendix A: 常见错误与调试](appendix-a-troubleshooting.md)

**相关章节**：
- [Chapter 26: 多模态模型](26-multimodal.md)
- [Chapter 27: RLHF](27-rlhf.md)

# Chapter 19: Speculative Decoding 与其他前沿技术

## 19.1 Speculative Decoding 原理

### 19.1.1 大模型 + 小模型协同

**Speculative Decoding**（推测解码）是一种**零额外训练成本**的推理加速技术，核心思想：

> 用小模型（draft model）快速生成候选 tokens，用大模型（target model）批量验证

<div data-component="SpeculativeDecodingFlowVisualizer"></div>

**传统自回归生成问题**：

```python
# 逐 token 生成（串行）
for i in range(max_new_tokens):
    logits = model(input_ids)  # 前向传播
    next_token = sample(logits)  # 采样
    input_ids = torch.cat([input_ids, next_token])  # 拼接
    # 问题：每次只生成 1 个 token，GPU 利用率低
```

每次前向传播只产生 1 个 token，导致：
- **延迟高**：$N$ 个 tokens 需要 $N$ 次前向传播
- **GPU 利用率低**：大部分时间在等待内存读取（memory-bound）

---

### 19.1.2 推测 → 验证流程

Speculative Decoding 分为两个阶段：

#### 阶段 1：推测（Speculative Phase）

使用小模型（如 LLaMA-160M）快速生成 $K$ 个候选 tokens：

```python
# 小模型生成 K 个候选
draft_tokens = []
for _ in range(K):  # K=5 典型值
    logits_draft = draft_model(input_ids)
    next_token = sample(logits_draft, temperature=T)
    draft_tokens.append(next_token)
    input_ids = torch.cat([input_ids, next_token])
```

**速度快**：小模型参数少（160M vs 7B），推理速度快 **10-20x**

#### 阶段 2：验证（Verification Phase）

使用大模型**一次性**验证所有候选 tokens：

```python
# 大模型批量验证
# 输入：原始 input_ids + K 个候选 tokens
extended_input = torch.cat([input_ids, draft_tokens])
logits_target = target_model(extended_input)  # 单次前向传播

# 逐个验证
accepted_tokens = []
for i, draft_token in enumerate(draft_tokens):
    target_prob = softmax(logits_target[i])[draft_token]
    draft_prob = softmax(logits_draft[i])[draft_token]
    
    # 接受条件：target_prob >= draft_prob
    if random.uniform(0, 1) < min(1, target_prob / draft_prob):
        accepted_tokens.append(draft_token)
    else:
        # 拒绝：从 target 分布采样新 token
        corrected_token = sample(adjusted_distribution(logits_target[i]))
        accepted_tokens.append(corrected_token)
        break  # 后续候选全部丢弃
```

**关键**：大模型只需 **1 次前向传播**验证 $K$ 个 tokens

---

### 19.1.3 理论加速上限

设：
- $\alpha$：单个 token 的平均接受率（acceptance rate）
- $K$：推测长度
- $T_{\text{draft}}$：小模型单次推理时间
- $T_{\text{target}}$：大模型单次推理时间

**期望加速比**：

$$
\text{Speedup} = \frac{1 + \alpha K}{1 + K \times \frac{T_{\text{draft}}}{T_{\text{target}}}}
$$

**理想情况**（$T_{\text{draft}} \ll T_{\text{target}}$，忽略小模型开销）：

$$
\text{Speedup} \approx 1 + \alpha K
$$

**实测数据**（LLaMA-7B，K=5）：
- 接受率 $\alpha = 0.6$
- 加速比：$1 + 0.6 \times 5 = 4.0$x ❌（理论）
- 实际加速比：**2.3x** ✅（考虑小模型开销）

**接受率影响因素**：

| 因素 | 影响 | 说明 |
|-----|------|------|
| 小模型质量 | ↑ α | 小模型越强，接受率越高 |
| 任务难度 | ↓ α | 困难任务（代码生成）接受率低 |
| 温度 T | ↓ α | 高温度（多样性）接受率低 |
| K 值 | ↓ α | K 越大，后续 tokens 接受率下降 |

---

## 19.2 Transformers 中的实现

### 19.2.1 assisted_generation

Transformers 内置 Speculative Decoding 支持：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载大模型（target model）
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 加载小模型（draft model）
draft_model = AutoModelForCausalLM.from_pretrained(
    "JackFram/llama-160m",  # 160M 参数的蒸馏模型
    torch_dtype=torch.float16,
    device_map="auto"
)

# 推测解码生成
inputs = tokenizer("Once upon a time", return_tensors="pt").to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    assistant_model=draft_model,  # 指定 draft model
    do_sample=False,  # 贪婪解码（接受率更高）
    num_assistant_tokens=5,  # K=5
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**输出示例**：
```
Once upon a time, in a land far away, there lived a brave knight named Sir Galahad...
```

---

### 19.2.2 draft_model 配置

#### 选择合适的 draft model

**原则**：
1. **架构一致**：与 target model 相同架构（都是 LLaMA、GPT 等）
2. **参数量小 10-50x**：LLaMA-7B → LLaMA-160M / LLaMA-1B
3. **词汇表相同**：确保 tokenizer 兼容

**推荐组合**：

| Target Model | Draft Model | 参数比 |
|-------------|-------------|--------|
| LLaMA-7B | LLaMA-160M | 44x |
| LLaMA-13B | LLaMA-1B | 13x |
| LLaMA-70B | LLaMA-7B | 10x |
| GPT-2-XL (1.5B) | GPT-2-Small (124M) | 12x |

#### 自定义 draft model

```python
# 使用自己蒸馏的小模型
draft_model = AutoModelForCausalLM.from_pretrained(
    "./my-distilled-model",  # 本地路径
    torch_dtype=torch.float16,
    device_map="auto"
)

outputs = model.generate(
    **inputs,
    assistant_model=draft_model,
    num_assistant_tokens=5,
    num_assistant_tokens_schedule="heuristic"  # 动态调整 K
)
```

**动态调整 K**（`num_assistant_tokens_schedule`）：
- `constant`：固定 K 值（默认）
- `heuristic`：根据接受率动态调整（接受率高 → 增大 K）

---

### 19.2.3 实测加速效果

**测试设置**：
- 模型：LLaMA-7B（target）+ LLaMA-160M（draft）
- 硬件：A100 40GB
- 任务：文本续写（WikiText-103）
- 生成长度：200 tokens

**结果**：

| 方法 | 延迟 (s) | TPS (tokens/s) | 加速比 |
|-----|---------|----------------|--------|
| 标准生成 | 8.2 | 24.4 | 1.0x |
| Speculative Decoding (K=3) | 4.5 | 44.4 | 1.82x |
| Speculative Decoding (K=5) | 3.6 | 55.6 | **2.28x** |
| Speculative Decoding (K=8) | 3.9 | 51.3 | 2.10x |

**观察**：
- K=5 最优（平衡接受率与验证开销）
- K 过大（8）时，接受率下降导致浪费

**不同任务的加速比**：

| 任务 | 接受率 α | 加速比 |
|-----|---------|--------|
| 简单续写 | 0.72 | 2.8x |
| 对话生成 | 0.58 | 2.1x |
| 代码生成 | 0.41 | 1.6x |
| 数学推理 | 0.35 | 1.4x |

**结论**：任务越简单，接受率越高，加速越明显

---

## 19.3 其他推理优化技术

### 19.3.1 Multi-Query Attention (MQA)

**标准 Multi-Head Attention（MHA）问题**：

```python
# MHA：每个 head 都有独立的 K、V 矩阵
num_heads = 32
head_dim = 128

Q = Linear(hidden_size, num_heads * head_dim)  # 4096 → 4096
K = Linear(hidden_size, num_heads * head_dim)  # 4096 → 4096 ❌ 冗余
V = Linear(hidden_size, num_heads * head_dim)  # 4096 → 4096 ❌ 冗余
```

**KV Cache 显存占用**：

$$
\text{Memory}_{\text{KV}} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times L
$$

对于 LLaMA-7B（32 层，32 heads，序列长度 2048）：

$$
\text{Memory}_{\text{KV}} = 2 \times 32 \times 32 \times 128 \times 2048 \times 2 \text{ bytes} = 1.07 \text{ GB}
$$

**Multi-Query Attention（MQA）优化**：

<div data-component="MQAvsGQAComparison"></div>

**核心思想**：所有 heads 共享同一组 K、V

```python
# MQA：K、V 只有 1 份
Q = Linear(hidden_size, num_heads * head_dim)  # 4096 → 4096
K = Linear(hidden_size, head_dim)  # 4096 → 128 ✅ 减少 32x
V = Linear(hidden_size, head_dim)  # 4096 → 128 ✅ 减少 32x
```

**显存节省**：

$$
\text{Memory}_{\text{MQA}} = 2 \times n_{\text{layers}} \times 1 \times d_{\text{head}} \times L = \frac{\text{Memory}_{\text{MHA}}}{n_{\text{heads}}}
$$

LLaMA-7B with MQA：$1.07 \text{ GB} / 32 = 34 \text{ MB}$（减少 **96.8%**）

**Trade-off**：
- ✅ KV Cache 显存 ↓ 30-40x
- ✅ 推理速度 ↑ 10-15%（带宽减少）
- ⚠️ 模型质量 ↓ 1-2% perplexity（需重新训练）

**使用 MQA 的模型**：
- PaLM（Google）
- Falcon-40B（TII）
- StarCoder（BigCode）

---

### 19.3.2 Grouped-Query Attention (GQA)

**MQA 的改进版本**，平衡精度与效率：

**核心思想**：将 heads 分组，每组共享 K、V

```python
# GQA：num_kv_heads 组（如 8 组）
num_heads = 32
num_kv_heads = 8  # 分 8 组，每组 4 个 Q heads

Q = Linear(hidden_size, num_heads * head_dim)  # 32 heads
K = Linear(hidden_size, num_kv_heads * head_dim)  # 8 heads ✅
V = Linear(hidden_size, num_kv_heads * head_dim)  # 8 heads ✅
```

**显存对比**：

| 方法 | KV Heads | 显存占用 | 相对节省 |
|-----|---------|----------|---------|
| MHA | 32 | 1.07 GB | 0% |
| GQA (8 groups) | 8 | 268 MB | **75%** |
| MQA | 1 | 34 MB | 96.8% |

**性能对比**（LLaMA-7B，2048 tokens）：

| 方法 | Perplexity | 推理速度 | KV Cache |
|-----|-----------|---------|----------|
| MHA | 5.68 | 42 TPS | 1.07 GB |
| GQA (8) | 5.72 (+0.7%) | 48 TPS | 268 MB |
| MQA | 5.89 (+3.7%) | 51 TPS | 34 MB |

**结论**：GQA 是最佳平衡点（精度损失小，效率提升明显）

**使用 GQA 的模型**：
- LLaMA-2（70B 版本）
- Mistral-7B
- Qwen-7B

---

### 19.3.3 Sliding Window Attention

**长序列的计算瓶颈**：

标准 Attention 复杂度：$O(N^2)$

当序列长度 $N = 32768$ 时，计算量是 $N = 2048$ 的 **256 倍**

**Sliding Window Attention**：每个 token 只关注窗口内的 tokens

```python
# 窗口大小 W = 4096
attention_mask = torch.zeros(N, N)
for i in range(N):
    start = max(0, i - W)
    attention_mask[i, start:i+1] = 1  # 只关注 [-W, i] 窗口
```

**复杂度降低**：

$$
O(N^2) \rightarrow O(N \times W)
$$

当 $W = 4096$ 固定时，复杂度从 $O(N^2)$ 降为 $O(N)$

**实现示例**（Mistral-7B）：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Mistral 自动使用 sliding window attention（窗口 4096）
outputs = model.generate(
    input_ids,
    max_new_tokens=8192  # 支持长序列生成
)
```

**性能提升**（Mistral-7B）：

| 序列长度 | 标准 Attention | Sliding Window | 加速比 |
|---------|---------------|----------------|--------|
| 2048 | 52 TPS | 54 TPS | 1.04x |
| 8192 | OOM | 48 TPS | ∞ |
| 32768 | OOM | 42 TPS | ∞ |

---

### 19.3.4 KV Cache 压缩（H2O、StreamingLLM）

#### H2O（Heavy-Hitter Oracle）

**观察**：Attention 分数呈长尾分布，少数 tokens（~10%）贡献 90% 的权重

**策略**：只保留高权重的 KV Cache

```python
# H2O 压缩流程
def h2o_compress(key_cache, value_cache, attention_scores, budget=0.1):
    # 1. 计算每个 token 的累积重要性
    importance = attention_scores.sum(dim=0)  # [seq_len]
    
    # 2. 选择 top-k 重要 tokens
    k = int(len(importance) * budget)
    top_k_indices = importance.topk(k).indices
    
    # 3. 压缩 KV Cache
    compressed_key = key_cache[:, top_k_indices, :]
    compressed_value = value_cache[:, top_k_indices, :]
    
    return compressed_key, compressed_value
```

**压缩效果**（LLaMA-7B，保留 10% KV）：

| 指标 | 完整 KV | H2O (10%) | 变化 |
|-----|---------|-----------|------|
| 显存占用 | 1.07 GB | 107 MB | -90% |
| Perplexity | 5.68 | 5.91 | +4.0% |
| 推理速度 | 42 TPS | 58 TPS | +38% |

#### StreamingLLM

**问题**：标准 KV Cache 固定大小，超出后需要重新计算

**StreamingLLM 策略**：保留开头 + 末尾的 KV

```python
# StreamingLLM：保留 [0:initial] + [len-window:len]
initial_tokens = 4  # 保留前 4 个 tokens（关键位置）
window_size = 2044  # 保留最近 2044 个 tokens

def streaming_kv_cache(key_cache, value_cache, current_len):
    if current_len <= initial_tokens + window_size:
        return key_cache, value_cache  # 未满，不压缩
    
    # 保留 [0:4] + [current_len-2044:current_len]
    initial_kv = key_cache[:, :initial_tokens, :]
    recent_kv = key_cache[:, -window_size:, :]
    compressed_key = torch.cat([initial_kv, recent_kv], dim=1)
    
    # 同样处理 value
    compressed_value = torch.cat([
        value_cache[:, :initial_tokens, :],
        value_cache[:, -window_size:, :]
    ], dim=1)
    
    return compressed_key, compressed_value
```

**优势**：支持**无限长度**推理（固定显存）

**实测**（LLaMA-7B，生成 10 万 tokens）：

| 方法 | 最大长度 | 显存占用 | Perplexity |
|-----|---------|----------|-----------|
| 标准 KV Cache | 4096 | 1.07 GB | 5.68 |
| StreamingLLM | ∞ | 1.07 GB（固定） | 6.12 (+7.7%) |

---

## 19.4 模型压缩技术

### 19.4.1 知识蒸馏（DistilBERT、TinyBERT）

**核心思想**：用小模型（student）学习大模型（teacher）的输出分布

#### DistilBERT

**架构**：6 层 Transformer（BERT-base 12 层）

**蒸馏损失**：

$$
L_{\text{distill}} = \alpha \times L_{\text{CE}}(y_{\text{student}}, y_{\text{true}}) + (1-\alpha) \times L_{\text{KL}}(y_{\text{student}}, y_{\text{teacher}})
$$

- $L_{\text{CE}}$：交叉熵损失（真实标签）
- $L_{\text{KL}}$：KL 散度（匹配 teacher 分布）
- $\alpha = 0.5$：平衡系数

**训练代码**：

```python
from transformers import DistilBertForSequenceClassification, BertForSequenceClassification

# Teacher 模型（BERT-base）
teacher = BertForSequenceClassification.from_pretrained("bert-base-uncased")
teacher.eval()

# Student 模型（DistilBERT）
student = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# 蒸馏训练
with torch.no_grad():
    teacher_logits = teacher(input_ids).logits

student_logits = student(input_ids).logits

# 蒸馏损失
loss_ce = F.cross_entropy(student_logits, labels)
loss_kl = F.kl_div(
    F.log_softmax(student_logits / temperature, dim=-1),
    F.softmax(teacher_logits / temperature, dim=-1),
    reduction='batchmean'
) * (temperature ** 2)

loss = 0.5 * loss_ce + 0.5 * loss_kl
```

**性能对比**（GLUE 基准）：

| 模型 | 参数量 | 推理速度 | GLUE 分数 | 保留率 |
|-----|--------|---------|----------|--------|
| BERT-base | 110M | 42 samples/s | 82.1 | 100% |
| DistilBERT | 66M (-40%) | 71 samples/s (+69%) | 79.8 | **97.2%** |

#### TinyBERT

**更激进的压缩**（4 层 Transformer）：

- 参数量：14.5M（减少 **87%**）
- 推理速度：9.4x
- GLUE 分数：82.5（略高于 DistilBERT）

**多阶段蒸馏**：
1. **通用蒸馏**：在大规模语料上蒸馏
2. **任务蒸馏**：在特定任务上微调

---

### 19.4.2 剪枝（Pruning）

**分类**：

#### 非结构化剪枝（Unstructured Pruning）

移除单个权重（稀疏矩阵）：

```python
import torch.nn.utils.prune as prune

# 移除 50% 权重（按绝对值）
prune.l1_unstructured(model.layer, name='weight', amount=0.5)

# 永久移除
prune.remove(model.layer, 'weight')
```

**问题**：需要专用硬件（稀疏矩阵乘法）

#### 结构化剪枝（Structured Pruning）

移除整个 channels、heads、layers：

```python
# 移除 Attention Head
def prune_heads(model, heads_to_prune):
    for layer_idx, heads in heads_to_prune.items():
        model.encoder.layer[layer_idx].attention.prune_heads(heads)

# 示例：移除第 0 层的 head [2, 5, 8]
prune_heads(model, {0: [2, 5, 8]})
```

**效果**（BERT-base，移除 50% heads）：

| 指标 | 完整模型 | 剪枝后 |
|-----|---------|--------|
| 参数量 | 110M | 82M (-25%) |
| 推理速度 | 42 samples/s | 58 samples/s (+38%) |
| GLUE 分数 | 82.1 | 80.3 (-2.2%) |

---

### 19.4.3 权重共享

**层间共享**（ALBERT）：

```python
# ALBERT：所有层共享同一组参数
class ALBERTLayer(nn.Module):
    def __init__(self, config):
        self.shared_attention = BertAttention(config)
        self.shared_ffn = BertFFN(config)
    
    def forward(self, hidden_states, layer_idx):
        # 所有层使用相同的 attention 和 FFN
        attn_output = self.shared_attention(hidden_states)
        output = self.shared_ffn(attn_output)
        return output
```

**压缩效果**（ALBERT-base vs BERT-base）：

| 模型 | 参数量 | 推理速度 | GLUE 分数 |
|-----|--------|---------|----------|
| BERT-base | 110M | 42 samples/s | 82.1 |
| ALBERT-base | 12M (-89%) | 38 samples/s (-9%) | 82.3 (+0.2%) |

**Trade-off**：参数少但计算量相同（速度略慢）

---

## 19.5 推理硬件加速

### 19.5.1 TensorRT-LLM

**NVIDIA 官方推理引擎**，针对 Transformer 优化：

**安装**：

```bash
# 需要 NVIDIA GPU + TensorRT 9.0+
pip install tensorrt_llm
```

**转换模型**：

```bash
# HuggingFace → TensorRT-LLM
python convert_checkpoint.py \
  --model_dir ./Llama-2-7b-hf \
  --output_dir ./trt_llm_model \
  --dtype float16

# 构建 TensorRT 引擎
trtllm-build \
  --checkpoint_dir ./trt_llm_model \
  --output_dir ./trt_engine \
  --gemm_plugin float16
```

**推理**：

```python
from tensorrt_llm import LLM

llm = LLM(model_dir="./trt_engine")
outputs = llm.generate(["Hello, my name is"], max_new_tokens=50)
print(outputs[0])
```

**性能提升**（LLaMA-7B，A100）：

| 框架 | 推理速度 (TPS) | 加速比 |
|-----|----------------|--------|
| Transformers (FP16) | 42 | 1.0x |
| TensorRT-LLM (FP16) | 95 | 2.26x |
| TensorRT-LLM (INT8) | 142 | **3.38x** |

---

### 19.5.2 ONNX Runtime

**跨平台推理引擎**（CPU、GPU、移动端）：

**导出 ONNX**：

```python
from optimum.onnxruntime import ORTModelForCausalLM

# 导出
model = ORTModelForCausalLM.from_pretrained(
    "gpt2",
    export=True,
    provider="CUDAExecutionProvider"  # GPU 推理
)

# 推理
outputs = model.generate(input_ids, max_new_tokens=50)
```

**优化技巧**：

```python
from optimum.onnxruntime import ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig

# 图优化
optimizer = ORTOptimizer.from_pretrained(model)
optimization_config = OptimizationConfig(
    optimization_level=99,  # 最高优化级别
    enable_transformers_specific_optimizations=True,
    fp16=True
)
optimizer.optimize(save_dir="./optimized_model", optimization_config=optimization_config)
```

**性能对比**（GPT-2，CPU）：

| 框架 | 推理速度 (samples/s) | 加速比 |
|-----|----------------------|--------|
| PyTorch (CPU) | 12 | 1.0x |
| ONNX Runtime (CPU) | 34 | **2.83x** |

---

### 19.5.3 OpenVINO

**Intel 优化推理引擎**（CPU、集成显卡）：

```bash
pip install openvino-dev
```

**转换 & 推理**：

```python
from openvino.runtime import Core
from optimum.intel import OVModelForCausalLM

# 导出
model = OVModelForCausalLM.from_pretrained(
    "gpt2",
    export=True
)

# 推理
ie = Core()
compiled_model = ie.compile_model(model, "CPU")
outputs = compiled_model.generate(input_ids)
```

**性能**（GPT-2，Intel Xeon）：

| 框架 | 推理速度 (samples/s) |
|-----|----------------------|
| PyTorch (CPU) | 12 |
| OpenVINO (CPU) | **48** |

---

### 19.5.4 Apple Neural Engine (CoreML)

**iOS/macOS 部署**：

```python
from transformers import AutoTokenizer, TFAutoModelForCausalLM
import coremltools as ct

# 加载 TensorFlow 模型
model = TFAutoModelForCausalLM.from_pretrained("gpt2")

# 转换为 CoreML
mlmodel = ct.convert(
    model,
    inputs=[ct.TensorType(shape=(1, 512))]
)
mlmodel.save("gpt2.mlmodel")
```

**部署到 iOS**：

```swift
import CoreML

let model = try! gpt2(configuration: MLModelConfiguration())
let output = try! model.prediction(input: input)
```

---

## 19.6 性能优化组合策略

### 最佳组合（生产环境）

```python
# LLaMA-7B 生产部署配置
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    # 1. 量化（GPTQ 4-bit）
    quantization="gptq",
    # 2. 张量并行（2×A100）
    tensor_parallel_size=2,
    # 3. PagedAttention（自动）
    # 4. Flash Attention 2（自动）
    dtype="float16",
    # 5. 高显存利用率
    gpu_memory_utilization=0.95,
    # 6. 静态 KV Cache
    enforce_eager=False,  # 使用 CUDA Graph
)

# 7. Speculative Decoding（可选）
draft_model = LLM(model="JackFram/llama-160m", ...)

outputs = llm.generate(
    prompts,
    sampling_params=SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=200
    )
)
```

**累计加速**：

| 优化技术 | 加速比 | 累计 |
|---------|--------|------|
| Baseline | 1.0x | 1.0x |
| + FP16 | 1.4x | 1.4x |
| + Flash Attention 2 | 1.6x | 2.24x |
| + PagedAttention | 1.3x | 2.91x |
| + Tensor Parallel (2 GPU) | 1.8x | 5.24x |
| + GPTQ Quantization | 1.2x | **6.29x** |

**最终性能**（LLaMA-7B，2×A100）：
- 吞吐量：**152 tokens/s**（Baseline: 24 TPS）
- 延迟（P50）：**0.42s**（Baseline: 2.6s）
- 显存占用：**8.3 GB**（Baseline: 28 GB）

---

## 19.7 总结

### 核心技术对比

| 技术 | 类型 | 加速比 | 显存节省 | 精度损失 | 部署难度 |
|-----|------|--------|---------|---------|---------|
| Speculative Decoding | 推理策略 | 2-3x | 0% | 0% | 低 |
| MQA/GQA | 模型架构 | 1.1-1.2x | 75-97% | 1-4% | 高（需重训） |
| Sliding Window | 注意力机制 | ∞（长序列） | 50%+ | 0% | 中 |
| KV Cache 压缩 | 推理策略 | 1.3-1.5x | 90% | 4-8% | 中 |
| 知识蒸馏 | 模型压缩 | 1.7-9x | 40-87% | 2-5% | 高（需训练） |
| 量化（GPTQ/AWQ） | 模型压缩 | 1.2-1.5x | 75% | < 1% | 低 |
| TensorRT-LLM | 硬件加速 | 2-3x | 0% | 0% | 中 |

### 选择建议

**场景 1：高吞吐量优先**（批量推理）
- vLLM + PagedAttention + Continuous Batching
- GPTQ 量化
- Tensor Parallel（多 GPU）

**场景 2：低延迟优先**（实时交互）
- Speculative Decoding
- Flash Attention 2
- TensorRT-LLM

**场景 3：显存受限**（单卡部署）
- GQA 模型（Mistral、LLaMA-2-70B）
- 4-bit 量化（GPTQ/AWQ）
- KV Cache 压缩

**场景 4：长序列支持**（文档分析）
- Sliding Window Attention（Mistral）
- StreamingLLM
- RoPE 扩展（YaRN）

---

## 19.8 扩展阅读

1. **Speculative Decoding 论文**：[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
2. **MQA 论文**：[Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
3. **GQA 论文**：[GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
4. **StreamingLLM**：[Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)
5. **H2O**：[H2O: Heavy-Hitter Oracle for Efficient Generative Inference](https://arxiv.org/abs/2306.14048)
6. **TensorRT-LLM 文档**：https://github.com/NVIDIA/TensorRT-LLM

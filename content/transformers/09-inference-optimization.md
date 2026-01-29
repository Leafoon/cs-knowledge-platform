---
title: "Chapter 9. 推理优化与部署"
description: "掌握推理加速技术、KV Cache、FlashAttention、BetterTransformer 等优化方法"
updated: "2026-01-22"
---

## 9.1 推理性能优化概述

### 9.1.1 推理与训练的差异

**关键区别**：

| 维度 | 训练 | 推理 |
|------|------|------|
| 目标 | 优化损失函数 | 最小化延迟/最大化吞吐 |
| 梯度 | 需要计算和存储 | 不需要 |
| 内存 | 激活值需保留 | 仅保留前向结果 |
| 批处理 | 固定batch size | 动态batching |
| 确定性 | 可重现性较弱 | 要求高度一致 |

**推理性能指标**：

```python
# 核心指标定义
import time

def measure_inference_metrics(model, tokenizer, prompts, num_runs=100):
    """测量推理性能的关键指标"""
    
    # 1. Time to First Token (TTFT) - 首token延迟
    start = time.time()
    inputs = tokenizer(prompts[0], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1)
    ttft = time.time() - start
    
    # 2. Tokens Per Second (TPS) - 吞吐量
    total_tokens = 0
    start = time.time()
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        total_tokens += outputs.shape[1]
    tps = total_tokens / (time.time() - start)
    
    # 3. Latency - 端到端延迟
    latencies = []
    for _ in range(num_runs):
        start = time.time()
        inputs = tokenizer(prompts[0], return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        latencies.append(time.time() - start)
    
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
    
    return {
        "TTFT": ttft,
        "TPS": tps,
        "avg_latency": avg_latency,
        "p95_latency": p95_latency
    }

# 实际测量
metrics = measure_inference_metrics(model, tokenizer, test_prompts)
print(f"Time to First Token: {metrics['TTFT']*1000:.2f} ms")
print(f"Tokens Per Second: {metrics['TPS']:.2f}")
print(f"Average Latency: {metrics['avg_latency']*1000:.2f} ms")
print(f"P95 Latency: {metrics['p95_latency']*1000:.2f} ms")
```

<div data-component="InferenceMetricsVisualizer"></div>

### 9.1.2 推理优化技术栈

**层次化优化策略**：

```
┌─────────────────────────────────────────┐
│  应用层优化                              │
│  - Prompt Engineering                   │
│  - Caching 策略                         │
│  - Request Batching                     │
├─────────────────────────────────────────┤
│  算法层优化                              │
│  - KV Cache                             │
│  - Flash Attention                      │
│  - Speculative Decoding                 │
├─────────────────────────────────────────┤
│  模型层优化                              │
│  - 量化 (INT8/INT4/FP8)                 │
│  - 蒸馏 (Distillation)                  │
│  - 剪枝 (Pruning)                       │
├─────────────────────────────────────────┤
│  系统层优化                              │
│  - Kernel Fusion                        │
│  - torch.compile                        │
│  - CUDA Graph                           │
├─────────────────────────────────────────┤
│  硬件层优化                              │
│  - TensorRT                             │
│  - ONNX Runtime                         │
│  - 专用硬件 (TPU/Inferentia)            │
└─────────────────────────────────────────┘
```

---

## 9.2 KV Cache 深度解析

### 9.2.1 KV Cache 原理

**自回归生成的问题**：

```python
# 朴素生成（每步重新计算所有token的K和V）
def naive_generation(model, input_ids, max_new_tokens=20):
    for _ in range(max_new_tokens):
        # 问题：每次都要计算整个序列的attention！
        outputs = model(input_ids)  # O(seq_len²)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
    return input_ids

# 使用KV Cache（复用历史K、V）
def cached_generation(model, input_ids, max_new_tokens=20):
    past_key_values = None  # 初始为空
    
    for _ in range(max_new_tokens):
        # 第一步：计算所有token
        # 后续步：只计算新token，复用past_key_values
        outputs = model(
            input_ids if past_key_values is None else input_ids[:, -1:],
            past_key_values=past_key_values,
            use_cache=True  # 启用KV Cache
        )
        
        past_key_values = outputs.past_key_values  # 保存K、V
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
    
    return input_ids
```

**内存占用计算**：

```python
# LLaMA-7B KV Cache 显存分析
num_layers = 32
num_heads = 32
head_dim = 128  # hidden_size / num_heads = 4096 / 32
batch_size = 1
seq_len = 2048
precision = 2  # FP16

# 每层的KV Cache大小
kv_cache_per_layer = 2 * batch_size * num_heads * seq_len * head_dim * precision
# 2 (K和V) × 1 × 32 × 2048 × 128 × 2 bytes = 32 MB

# 所有层的总KV Cache
total_kv_cache = kv_cache_per_layer * num_layers
# 32 MB × 32 = 1024 MB = 1 GB

print(f"KV Cache显存占用: {total_kv_cache / 1024**3:.2f} GB")
# 对于 batch_size=8, seq_len=2048: ~8 GB
```

<div data-component="KVCacheMechanismVisualizer"></div>

### 9.2.2 静态 vs 动态 KV Cache

**动态 KV Cache（默认）**：

```python
# 默认行为：动态增长
model = AutoModelForCausalLM.from_pretrained("gpt2")
inputs = tokenizer("Hello", return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    use_cache=True,  # 默认开启
    # KV Cache 逐token增长
)
```

**静态 KV Cache（PyTorch 2.2+）**：

```python
from transformers import StaticCache

# 预分配固定大小的KV Cache
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16
)

# 创建静态cache
cache = StaticCache(
    config=model.config,
    max_batch_size=4,
    max_cache_len=2048,
    device="cuda",
    dtype=torch.float16
)

# 使用静态cache生成
outputs = model.generate(
    **inputs,
    past_key_values=cache,
    max_new_tokens=50
)

# 优势：
# 1. 避免动态内存分配
# 2. 内存布局连续，cache友好
# 3. 可与torch.compile配合
# 4. 推理速度提升10-30%
```

**性能对比**：

| Cache类型 | 内存分配 | 速度 | 显存占用 |
|----------|---------|------|----------|
| 动态Cache | 每步分配 | 基线 | 最小 |
| 静态Cache | 预分配 | +15% | 固定（可能浪费） |

---

## 9.3 Flash Attention 与注意力优化

### 9.3.1 Flash Attention 2 原理

**标准Attention的IO瓶颈**：

```
标准实现（慢）：
1. 计算 QK^T → 写入HBM (慢)
2. 读取 QK^T, 计算 Softmax → 写入HBM
3. 读取 Softmax, 计算 Softmax @ V → 输出

Flash Attention（快）：
1. 分块计算，所有操作在SRAM完成
2. 避免中间结果写入HBM
3. IO复杂度从 O(N²) 降到 O(N)
```

<div data-component="FlashAttentionIOComparison"></div>

**使用 Flash Attention 2**：

```python
# 方法1：模型加载时启用
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",  # ← 关键参数
    device_map="auto"
)

# 方法2：安装flash-attn库（需要）
# pip install flash-attn --no-build-isolation

# 检查是否启用
print(model.config._attn_implementation)
# 输出: "flash_attention_2"

# 方法3：运行时替换
from transformers.models.llama.modeling_llama import LlamaFlashAttention2

# 自动替换所有attention层
model = model.to("cuda")
```

**性能提升**：

```python
import time

# 基线：标准attention
model_std = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
start = time.time()
outputs_std = model_std.generate(**inputs, max_new_tokens=100)
time_std = time.time() - start

# Flash Attention 2
model_fa2 = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    attn_implementation="flash_attention_2"
)
start = time.time()
outputs_fa2 = model_fa2.generate(**inputs, max_new_tokens=100)
time_fa2 = time.time() - start

print(f"标准Attention: {time_std:.2f}s")
print(f"Flash Attention 2: {time_fa2:.2f}s")
print(f"加速比: {time_std/time_fa2:.2f}x")
# 典型结果: 1.5x - 2.5x 加速
```

### 9.3.2 其他注意力优化

**Scaled Dot-Product Attention (SDPA, PyTorch 2.0+)**：

```python
# PyTorch内置的优化版attention
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    attn_implementation="sdpa"  # 自动选择最优kernel
)

# 支持的后端：
# - Flash Attention (if installed)
# - Memory-Efficient Attention (xFormers)
# - Fallback to standard implementation
```

**BetterTransformer（已弃用，推荐SDPA）**：

```python
# 旧方法（已不推荐）
# model = model.to_bettertransformer()

# 新方法：使用SDPA
model = AutoModelForCausalLM.from_pretrained(
    "bert-base-uncased",
    attn_implementation="sdpa"
)
```

---

## 9.4 torch.compile 编译优化

### 9.4.1 PyTorch 2.0 编译机制

**TorchDynamo + TorchInductor 架构**：

```python
import torch

model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda")

# 编译模型（首次运行会较慢）
compiled_model = torch.compile(
    model,
    mode="reduce-overhead",  # 或 "default", "max-autotune"
    fullgraph=True,          # 尝试编译整个图
    dynamic=False            # 静态形状（更快）
)

# 首次运行：触发编译（慢）
inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
with torch.no_grad():
    _ = compiled_model.generate(**inputs, max_new_tokens=10)

# 后续运行：使用编译后的kernel（快）
outputs = compiled_model.generate(**inputs, max_new_tokens=50)
```

**编译模式对比**：

| 模式 | 编译时间 | 运行速度 | 适用场景 |
|------|---------|---------|----------|
| `default` | 中 | +20-30% | 通用 |
| `reduce-overhead` | 快 | +10-20% | 低延迟推理 |
| `max-autotune` | 慢 | +30-50% | 批量推理 |

**最佳实践**：

```python
# 推荐配置
model = torch.compile(
    model,
    mode="reduce-overhead",
    backend="inductor"  # 默认后端
)

# 注意事项：
# 1. 首次运行会触发JIT编译（10-60秒）
# 2. 输入形状改变会重新编译
# 3. 动态控制流（if/for）可能导致回退
# 4. 与KV Cache配合使用效果更佳

# 禁用编译（调试时）
torch._dynamo.reset()
```

<div data-component="TorchCompileSpeedupChart"></div>

---

## 9.5 模型量化推理

### 9.5.1 动态量化（bitsandbytes）

**8-bit 推理**：

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 配置8-bit量化
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,  # 异常值处理阈值
    llm_int8_skip_modules=["lm_head"]  # 跳过某些层
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)

# 推理（与FP16相比，显存减半，速度略慢5-10%）
outputs = model.generate(**inputs, max_new_tokens=100)
```

**4-bit 推理（NF4）**：

```python
# NF4量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时上采样到BF16
    bnb_4bit_use_double_quant=True,         # 双重量化
    bnb_4bit_quant_type="nf4"               # NormalFloat4
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",  # 70B模型
    quantization_config=quantization_config,
    device_map="auto"
)

# 显存占用：70B模型从140GB降到~35GB（4卡A100）
```

### 9.5.2 静态量化（GPTQ）

**GPTQ 量化推理**：

```python
from transformers import GPTQConfig

# 加载GPTQ量化模型（需要预先量化）
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-GPTQ",  # Hub上的GPTQ模型
    device_map="auto",
    revision="gptq-4bit-128g-actorder_True"
)

# 或者自己量化（需要校准数据集）
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True  # activation order
)

model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantize_config=quantize_config
)

# 量化（需要校准数据）
model.quantize(calibration_dataset)
model.save_quantized("./llama-2-7b-gptq")
```

**性能对比**：

| 量化方法 | 精度损失 | 推理速度 | 显存占用 | 适用场景 |
|---------|---------|---------|---------|----------|
| FP16 | 0% | 基线 | 100% | 训练/高质量推理 |
| INT8 (bitsandbytes) | <1% | -5% | 50% | 通用推理 |
| INT4 NF4 | 1-2% | -10% | 25% | 显存受限 |
| GPTQ INT4 | <1% | +20% | 25% | 高吞吐推理 |
| AWQ INT4 | <0.5% | +30% | 25% | 最佳性能 |

<div data-component="QuantizationMethodComparison"></div>

---

## 9.6 批处理与动态Batching

### 9.6.1 静态批处理

**朴素批处理**：

```python
# 问题：不同长度的请求需要padding到相同长度
prompts = [
    "Hello",                    # 1 token
    "What is the capital",      # 4 tokens
    "Explain quantum physics"   # 3 tokens
]

# 强制padding到最长序列
inputs = tokenizer(
    prompts,
    return_tensors="pt",
    padding=True  # padding到4 tokens
).to("cuda")

# 浪费计算：短序列也要处理padding token
outputs = model.generate(**inputs, max_new_tokens=50)
```

**动态Batching（理想）**：

```python
# 按长度分组，减少padding浪费
from collections import defaultdict

def dynamic_batching(prompts, max_batch_size=8):
    """根据长度动态分组"""
    length_groups = defaultdict(list)
    
    for prompt in prompts:
        length = len(tokenizer.encode(prompt))
        # 按长度范围分组（0-10, 10-20, 20-30...）
        bucket = (length // 10) * 10
        length_groups[bucket].append(prompt)
    
    batches = []
    for bucket, group_prompts in length_groups.items():
        # 每组内部再分batch
        for i in range(0, len(group_prompts), max_batch_size):
            batch = group_prompts[i:i+max_batch_size]
            batches.append(batch)
    
    return batches

# 使用
batches = dynamic_batching(prompts)
for batch in batches:
    inputs = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=50)
```

### 9.6.2 Continuous Batching（vLLM引入）

**原理**：请求在完成时立即移出batch，新请求立即加入。

```python
# 传统静态batching
# Batch 1: [req1(50 tokens), req2(100 tokens), req3(80 tokens)]
# 等待最长的req2完成，整个batch才结束

# Continuous batching
# Step 1-50:  [req1, req2, req3]
# Step 51:    req1完成 → [req2, req3, req4(新加入)]
# Step 81:    req3完成 → [req2, req4, req5(新加入)]
# Step 101:   req2完成 → [req4, req5]

# 优势：GPU利用率更高，吞吐量提升2-3x
```

---

## 9.7 vLLM 高性能推理

### 9.7.1 PagedAttention 核心创新

**问题**：传统KV Cache分配方式效率低。

```python
# 传统方式：为每个请求预分配连续内存
kv_cache = torch.zeros(
    (batch_size, num_layers, 2, num_heads, max_seq_len, head_dim)
)
# 问题：
# 1. 必须预分配max_seq_len大小（浪费）
# 2. 请求提前结束，内存无法释放
# 3. 内存碎片化严重
```

**PagedAttention 解决方案**：

```
类比操作系统的虚拟内存分页：
- KV Cache 分割成固定大小的 Block（如16 tokens）
- 每个请求的 KV Cache 由多个 Block 组成（非连续）
- Block 按需分配，用完立即回收
- 支持多个请求共享相同 Block（Prefix共享）

示例：
Request A: [Block 1] → [Block 5] → [Block 9]
Request B: [Block 1] → [Block 3] → [Block 7]
            ↑ 共享的Prefix
```

<div data-component="PagedAttentionVisualizer"></div>

### 9.7.2 vLLM 使用指南

**离线推理**：

```python
from vllm import LLM, SamplingParams

# 初始化vLLM引擎
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=1,      # 张量并行度
    gpu_memory_utilization=0.9,  # GPU显存利用率
    max_num_seqs=256,            # 最大并发序列数
    max_model_len=4096           # 最大序列长度
)

# 采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# 批量推理（自动continuous batching）
prompts = [
    "Explain AI in simple terms:",
    "What is the meaning of life?",
    "Write a Python function to sort a list:"
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
    print("---")
```

**在线服务（OpenAI兼容API）**：

```bash
# 启动vLLM服务器
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --tensor-parallel-size 2 \
    --dtype float16 \
    --max-model-len 4096

# 客户端调用（完全兼容OpenAI API）
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123"
)

response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    temperature=0.7,
    max_tokens=100
)

print(response.choices[0].message.content)
```

**性能优势**：

```python
# 基准测试：LLaMA-7B，100个请求，平均输出100 tokens

# Transformers (基线)
# - Throughput: 20 req/s
# - Latency (P95): 5000 ms
# - GPU Utilization: 40%

# vLLM (PagedAttention + Continuous Batching)
# - Throughput: 60 req/s  (↑ 3x)
# - Latency (P95): 1800 ms (↓ 64%)
# - GPU Utilization: 85%  (↑ 2x)
```

---

## 9.8 Text Generation Inference (TGI)

### 9.8.1 TGI 架构与特性

**Hugging Face 官方推理解决方案**：

```bash
# Docker部署（推荐）
docker run --gpus all --shm-size 1g -p 8080:80 \
    -v $PWD/data:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-7b-chat-hf \
    --num-shard 1 \
    --max-batch-prefill-tokens 4096 \
    --max-total-tokens 8192

# 关键参数：
# --num-shard: 张量并行度（多GPU分片）
# --quantize: 量化方式（bitsandbytes, gptq, awq）
# --max-batch-prefill-tokens: prefill阶段最大token数
# --max-total-tokens: 总token数上限
```

**核心特性**：

1. **Flash Attention v2**：自动启用
2. **Paged Attention**：类似vLLM
3. **Continuous Batching**：动态调度
4. **张量并行**：多GPU支持
5. **Safetensors**：快速模型加载
6. **Streaming**：SSE流式输出

### 9.8.2 TGI 客户端调用

**HTTP API**：

```python
import requests

# 标准生成
response = requests.post(
    "http://localhost:8080/generate",
    json={
        "inputs": "What is deep learning?",
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.95,
            "repetition_penalty": 1.2
        }
    }
)

print(response.json()["generated_text"])

# 流式生成
response = requests.post(
    "http://localhost:8080/generate_stream",
    json={"inputs": "Explain quantum computing:", "parameters": {"max_new_tokens": 200}},
    stream=True
)

for chunk in response.iter_lines():
    if chunk:
        print(chunk.decode("utf-8"))
```

**Chat API（Messages格式）**：

```python
# Chat模板自动处理
response = requests.post(
    "http://localhost:8080/v1/chat/completions",
    json={
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is AI?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

### 9.8.3 TGI 高级配置

**量化推理**：

```bash
# GPTQ 4-bit量化
docker run --gpus all -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id TheBloke/Llama-2-7B-GPTQ \
    --quantize gptq

# AWQ量化（更快）
docker run --gpus all -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id TheBloke/Llama-2-7B-AWQ \
    --quantize awq
```

**张量并行（多GPU）**：

```bash
# 在4张GPU上分片加载70B模型
docker run --gpus all -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-70b-chat-hf \
    --num-shard 4 \
    --max-batch-prefill-tokens 2048
```

<div data-component="TGIArchitectureDiagram"></div>

---

## 9.9 Speculative Decoding（推测解码）

### 9.9.1 原理与动机

**问题**：大模型生成慢（每个token都需要完整前向传播）。

**核心思想**：
1. 用小模型（draft model）快速生成多个候选token
2. 用大模型（target model）一次性验证所有候选
3. 接受正确的token，拒绝错误的

```python
# 算法流程
def speculative_decoding(draft_model, target_model, prompt, k=5):
    """
    k: 推测的token数量
    """
    current_tokens = tokenizer.encode(prompt)
    
    while len(current_tokens) < max_length:
        # 1. Draft model快速生成k个token
        draft_tokens = draft_model.generate(
            current_tokens,
            max_new_tokens=k
        )[-k:]  # 取最后k个
        
        # 2. Target model验证（并行）
        # 将所有候选token拼接成一个序列，一次前向传播
        candidate_sequence = current_tokens + draft_tokens
        target_logits = target_model(candidate_sequence).logits[-k-1:]
        
        # 3. 逐个验证并采样
        accepted = 0
        for i in range(k):
            target_prob = softmax(target_logits[i])
            draft_token = draft_tokens[i]
            
            # 验证draft token是否被target model接受
            if random.random() < target_prob[draft_token] / draft_prob[draft_token]:
                current_tokens.append(draft_token)
                accepted += 1
            else:
                # 拒绝，从target model重新采样
                new_token = sample(target_prob)
                current_tokens.append(new_token)
                break  # 停止后续验证
        
        if accepted == 0:
            # 全部拒绝，降级到普通生成
            pass
    
    return current_tokens
```

**理论加速比**：

```
加速比 = (1 + k × α) / (1 + k)

其中：
- k: 推测token数
- α: 接受率（draft model与target model的一致性）

示例：k=5, α=0.6
加速比 = (1 + 5×0.6) / (1 + 5) = 4 / 6 ≈ 2.33x
```

<div data-component="SpeculativeDecodingFlow"></div>

### 9.9.2 Transformers 实现

**Assisted Generation**：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 大模型（目标模型）
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 小模型（draft model）
draft_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",  # 10x smaller
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")

# 启用assisted generation
inputs = tokenizer("Explain quantum computing:", return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    assistant_model=draft_model,  # ← 指定draft model
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7
)

# 实测加速：1.5x - 3x（取决于模型相似度）
```

**选择合适的 Draft Model**：

| Target Model | Draft Model | 加速比 |
|--------------|-------------|--------|
| LLaMA-70B | LLaMA-7B | 2.0x |
| LLaMA-70B | LLaMA-13B | 2.5x |
| GPT-3.5 | GPT-2 | 1.8x |
| Mistral-7B | TinyLlama-1.1B | 1.6x |

**最佳实践**：
- Draft model应与target model架构相同或相似
- Draft model大小约为target model的1/5到1/10
- 任务相关性越高，接受率越高

---

## 9.10 模型导出与部署

### 9.10.1 ONNX 导出

**使用 Optimum 导出**：

```python
from optimum.onnxruntime import ORTModelForCausalLM, ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig

# 导出到ONNX
model = ORTModelForCausalLM.from_pretrained(
    "gpt2",
    export=True,  # 自动导出
    provider="CUDAExecutionProvider"  # 或 CPUExecutionProvider
)

# 优化ONNX图
optimizer = ORTOptimizer.from_pretrained(model)
optimization_config = OptimizationConfig(
    optimization_level=99,  # 最高优化级别
    optimize_for_gpu=True,
    fp16=True  # 启用FP16
)

optimizer.optimize(
    optimization_config=optimization_config,
    save_dir="./gpt2-optimized-onnx"
)

# 推理
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer("Hello, I'm a language model", return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

**量化ONNX模型**：

```python
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# 动态量化（INT8）
quantizer = ORTQuantizer.from_pretrained("./gpt2-onnx")
dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)

quantizer.quantize(
    save_dir="./gpt2-onnx-quantized",
    quantization_config=dqconfig
)

# 加载量化模型
model = ORTModelForCausalLM.from_pretrained("./gpt2-onnx-quantized")

# 性能：推理速度 +30-50%，模型大小减半
```

### 9.10.2 TensorRT 优化

**使用 TensorRT-LLM**：

```bash
# 安装TensorRT-LLM
pip install tensorrt_llm -U --extra-index-url https://pypi.nvidia.com

# 构建TensorRT引擎
python build.py \
    --model_dir ./Llama-2-7b-hf \
    --dtype float16 \
    --use_gpt_attention_plugin float16 \
    --use_gemm_plugin float16 \
    --max_batch_size 8 \
    --max_input_len 1024 \
    --max_output_len 512 \
    --output_dir ./trt_engines/llama-7b

# 运行推理
python run.py \
    --engine_dir ./trt_engines/llama-7b \
    --tokenizer_dir ./Llama-2-7b-hf \
    --max_output_len 100 \
    --input_text "Explain deep learning:"

# 性能提升：2-4x相比PyTorch
```

**Optimum + TensorRT**：

```python
# 尚在开发中（Optimum-Nvidia）
from optimum.nvidia import TensorRTModel

model = TensorRTModel.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    max_batch_size=8,
    max_seq_len=2048
)

# 编译为TensorRT引擎
model.compile()

# 推理
outputs = model.generate(**inputs, max_new_tokens=100)
```

<div data-component="DeploymentStackComparison"></div>

---

## 9.11 生产部署最佳实践

### 9.11.1 服务化封装

**FastAPI 异步服务**：

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from transformers import pipeline

app = FastAPI()

# 全局加载模型
generator = pipeline(
    "text-generation",
    model="gpt2",
    device=0
)

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7

@app.post("/generate")
async def generate(request: GenerationRequest):
    try:
        # 异步执行生成
        result = await asyncio.to_thread(
            generator,
            request.prompt,
            max_length=request.max_length,
            temperature=request.temperature
        )
        return {"generated_text": result[0]["generated_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 健康检查
@app.get("/health")
async def health():
    return {"status": "healthy"}

# 启动：uvicorn main:app --host 0.0.0.0 --port 8000
```

**请求队列与批处理**：

```python
from queue import Queue
from threading import Thread
import time

class BatchInferenceQueue:
    def __init__(self, model, batch_size=8, timeout=0.1):
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue = Queue()
        self.thread = Thread(target=self._process_queue, daemon=True)
        self.thread.start()
    
    def _process_queue(self):
        while True:
            batch = []
            start_time = time.time()
            
            # 收集batch
            while len(batch) < self.batch_size:
                if time.time() - start_time > self.timeout:
                    break  # 超时，处理当前batch
                
                try:
                    item = self.queue.get(timeout=0.01)
                    batch.append(item)
                except:
                    continue
            
            if batch:
                # 批量推理
                prompts = [item["prompt"] for item in batch]
                outputs = self.model(prompts)
                
                # 返回结果
                for item, output in zip(batch, outputs):
                    item["future"].set_result(output)
    
    async def generate(self, prompt):
        future = asyncio.Future()
        self.queue.put({"prompt": prompt, "future": future})
        return await future

# 使用
queue = BatchInferenceQueue(generator)

@app.post("/generate")
async def generate(request: GenerationRequest):
    result = await queue.generate(request.prompt)
    return {"generated_text": result}
```

### 9.11.2 监控与可观测性

**Prometheus 指标暴露**：

```python
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Response

# 定义指标
request_count = Counter("inference_requests_total", "Total inference requests")
request_latency = Histogram("inference_latency_seconds", "Inference latency")
token_count = Counter("tokens_generated_total", "Total tokens generated")

@app.post("/generate")
async def generate(request: GenerationRequest):
    request_count.inc()  # 计数
    
    with request_latency.time():  # 计时
        result = await asyncio.to_thread(generator, request.prompt)
    
    # 统计生成的token数
    num_tokens = len(result[0]["generated_text"].split())
    token_count.inc(num_tokens)
    
    return {"generated_text": result[0]["generated_text"]}

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")
```

**Grafana 可视化面板**：

```yaml
# 关键指标
1. Requests Per Second (RPS)
   - Query: rate(inference_requests_total[1m])

2. P95 Latency
   - Query: histogram_quantile(0.95, inference_latency_seconds_bucket)

3. Tokens Per Second
   - Query: rate(tokens_generated_total[1m])

4. GPU Utilization
   - Query: nvidia_gpu_utilization_percent

5. GPU Memory Usage
   - Query: nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes
```

---

## 9.12 本章小结

### 核心要点

1. **KV Cache**：推理加速的基础，静态cache配合torch.compile效果更佳
2. **Flash Attention 2**：IO优化，1.5-2.5x加速，必备优化
3. **torch.compile**：PyTorch 2.0编译优化，首次运行慢但后续快
4. **量化**：INT8/INT4显存减半/减75%，速度略降或持平
5. **vLLM**：PagedAttention + Continuous Batching，吞吐量提升3x
6. **TGI**：Hugging Face官方方案，Flash Attention + 张量并行
7. **Speculative Decoding**：小模型辅助大模型，2-3x加速

### 推理优化决策树

```python
# 选择最优推理方案

if 延迟敏感（<100ms）:
    if 模型<1B:
        → torch.compile + Flash Attention + 量化
    else:
        → TensorRT-LLM + 量化
elif 吞吐量优先:
    if 并发请求多:
        → vLLM (Continuous Batching)
    else:
        → TGI (Flash Attention + 张量并行)
elif 显存受限:
    if 需要训练:
        → QLoRA (4-bit)
    else:
        → GPTQ/AWQ量化 + vLLM
else:
    → Transformers + Flash Attention 2 + Static KV Cache
```

### 性能基准（LLaMA-7B）

| 配置 | Latency (P95) | Throughput | 显存 |
|------|--------------|------------|------|
| 基线（FP16） | 450ms | 22 req/s | 14 GB |
| + Flash Attention | 280ms | 35 req/s | 14 GB |
| + torch.compile | 220ms | 45 req/s | 14 GB |
| + INT8量化 | 230ms | 43 req/s | 7 GB |
| **vLLM** | **180ms** | **68 req/s** | **9 GB** |
| **TGI** | **190ms** | **65 req/s** | **8 GB** |

### 进一步阅读

- [vLLM Paper](https://arxiv.org/abs/2309.06180)
- [Flash Attention 2](https://arxiv.org/abs/2307.08691)
- [Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [TGI Documentation](https://huggingface.co/docs/text-generation-inference)

---

**下一章预告**：我们已经完成了 Transformers 核心内容的学习。后续章节可以根据需要深入探讨特定主题，如多模态模型、RLHF、长上下文等高级话题。

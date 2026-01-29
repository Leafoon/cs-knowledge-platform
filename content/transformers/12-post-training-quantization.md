---
title: "Chapter 12. 训练后量化（PTQ）"
description: "掌握 GPTQ、AWQ、bitsandbytes 等训练后量化技术与实现方法"
updated: "2026-01-22"
---

## 12.1 PTQ 基础概念

### 12.1.1 训练后量化 vs 量化感知训练

**两大量化范式**：

| 维度 | 训练后量化（PTQ） | 量化感知训练（QAT） |
|------|-------------------|---------------------|
| **定义** | 在已训练模型上直接量化 | 训练时模拟量化行为 |
| **时间成本** | 分钟级（无需训练） | 小时级（需完整训练） |
| **精度损失** | 2-5% | <1%（接近全精度） |
| **适用场景** | 快速部署、资源受限 | 精度敏感任务 |
| **典型方法** | GPTQ、AWQ、bitsandbytes | PyTorch QAT、TensorRT |

**核心差异**：

PTQ 的数学本质是**后处理优化**：
$$
\mathbf{W}_{\text{quant}} = \arg\min_{\mathbf{W}_q \in \mathcal{Q}} \| \mathbf{W} - \mathbf{W}_q \|_F
$$

而 QAT 是**训练时约束**：
$$
\min_{\mathbf{W}} \mathcal{L}(\mathbf{W}) + \lambda \cdot \text{Quant}(\mathbf{W})
$$

<div data-component="PTQvsQATComparison"></div>

### 12.1.2 静态量化 vs 动态量化

**静态量化（Static Quantization）**：

- 需要**校准数据集**（calibration dataset）来统计激活值范围
- 量化参数在推理前确定（权重 + 激活值的 scale/zero_point）
- 推理速度最快（无运行时开销）

```python
# 静态量化示例（PyTorch 原生）
import torch
from torch.quantization import quantize_dynamic

# 准备校准数据
calibration_data = [...]

# 计算激活值统计
model.eval()
with torch.no_grad():
    for batch in calibration_data:
        model(batch)  # 收集统计信息

# 执行量化
quantized_model = torch.quantization.convert(model)
```

**动态量化（Dynamic Quantization）**：

- **无需**校准数据
- 权重预先量化，激活值在推理时动态量化
- 速度略慢（运行时计算 scale），但无需数据

```python
# 动态量化（bitsandbytes 默认方式）
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    quantization_config=quant_config,  # 动态量化
)
```

### 12.1.3 量化粒度

<div data-component="QuantizationGranularityVisualizer"></div>

**Per-Tensor Quantization**（张量级）：

整个张量共享一个 scale 和 zero_point：
$$
\mathbf{W}_q = \text{round}\left( \frac{\mathbf{W}}{s} \right), \quad s = \frac{\max(\mathbf{W}) - \min(\mathbf{W})}{2^b - 1}
$$

优点：内存高效（1个scale）  
缺点：精度较低（无法适应异质分布）

**Per-Channel Quantization**（通道级）：

每个输出通道独立量化（卷积层/线性层常用）：
$$
\mathbf{W}_{q,i} = \text{round}\left( \frac{\mathbf{W}_i}{s_i} \right), \quad s_i = \frac{\max(\mathbf{W}_i) - \min(\mathbf{W}_i)}{2^b - 1}
$$

优点：精度更高（适应每通道分布）  
缺点：额外存储 $C$ 个 scale（$C$ = 输出通道数）

**Per-Group Quantization**（分组级，LLM常用）：

例如 GPTQ 的 group_size=128：将权重矩阵分成 128 列一组

```python
# GPTQ group_size 示例
gptq_config = GPTQConfig(
    bits=4,
    group_size=128,  # 每128列共享一个scale
    desc_act=True,   # 激活值排序优化
)
```

---

## 12.2 GPTQ 量化

### 12.2.1 GPTQ 算法原理

**GPTQ（Optimal Brain Quantization for GPT）** 是一种基于 **Hessian 矩阵二阶信息** 的最优量化方法。

**核心思想**：

给定权重矩阵 $\mathbf{W} \in \mathbb{R}^{d_{out} \times d_{in}}$，量化误差为：
$$
\mathcal{E} = \| \mathbf{WX} - \mathbf{W}_q \mathbf{X} \|_F^2
$$

其中 $\mathbf{X}$ 是校准数据的激活值。

**GPTQ 优化目标**：最小化量化后的输出误差（而非权重误差）

**算法步骤**（逐列量化）：

1. **计算 Hessian 矩阵**：
   $$
   \mathbf{H} = 2\mathbf{X}\mathbf{X}^T
   $$

2. **Cholesky 分解**：
   $$
   \mathbf{H} = \mathbf{L}\mathbf{L}^T
   $$

3. **逐列量化权重**（贪心优化）：
   ```
   for i in range(d_in):
       w_q[i] = quantize(w[i])
       error = w[i] - w_q[i]
       # 将误差传播到未量化的权重
       w[i+1:] -= error * (H[i, i+1:] / H[i, i])
   ```

4. **分组量化**（group_size=128）：
   - 每 128 列为一组，共享 scale/zero_point
   - 减少量化参数存储（从 4096 个 scale → 32 个）

<div data-component="GPTQAlgorithmFlow"></div>

**数学直觉**：

GPTQ 利用 Hessian 矩阵的**逆**来分配量化误差：
- Hessian 对角线大的权重更重要 → 量化时更谨慎
- 量化误差会**补偿**到相关权重上（通过 $\mathbf{H}^{-1}$）

### 12.2.2 安装与环境准备

```bash
# 安装 auto-gptq（CUDA 编译版本，推荐）
pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/

# 或使用预编译轮子（快速）
pip install auto-gptq

# 验证安装
python -c "from auto_gptq import AutoGPTQForCausalLM; print('GPTQ installed!')"
```

**依赖项**：
- CUDA >= 11.8（推荐）
- PyTorch >= 2.0
- transformers >= 4.32
- accelerate >= 0.20

### 12.2.3 量化模型加载

**方式 1：加载 Hugging Face Hub 上的预量化模型**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载 GPTQ 4-bit 量化模型（TheBloke 社区发布）
model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-GPTQ",
    device_map="auto",  # 自动分配到 GPU
    revision="gptq-4bit-128g-actorder_True",  # 选择量化版本
)

tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-GPTQ")

# 推理
inputs = tokenizer("Once upon a time", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

**输出示例**：
```
Once upon a time, there was a young girl named Sophia who lived in a small village...
```

**显存占用**：
- FP16 模型：~14GB
- GPTQ 4-bit：~4.5GB（**节省 69%**）

**方式 2：自定义量化配置**

```python
from transformers import GPTQConfig

gptq_config = GPTQConfig(
    bits=4,  # 量化位数（4/3/2）
    group_size=128,  # 分组大小（-1 表示 per-channel）
    desc_act=True,  # 激活值排序（提升精度）
    sym=True,  # 对称量化（vs 非对称）
    damp_percent=0.01,  # Hessian 阻尼系数
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=gptq_config,
    device_map="auto",
)

# 保存量化模型
model.save_pretrained("./llama2-7b-gptq-4bit")
tokenizer.save_pretrained("./llama2-7b-gptq-4bit")
```

### 12.2.4 手动量化流程

```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset
import torch

# 1. 准备校准数据集
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

def preprocess(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        return_tensors="pt",
    )

calibration_data = dataset.map(preprocess, batched=True, remove_columns=["text"])
calibration_data = calibration_data.select(range(128))  # 仅需少量数据

# 2. 配置量化参数
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,  # 激活值降序排序
)

# 3. 加载模型
model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantize_config=quantize_config,
)

# 4. 执行量化（需要 5-10 分钟）
model.quantize(calibration_data, batch_size=1)

# 5. 保存量化模型
model.save_quantized("./llama2-7b-gptq-4bit")
tokenizer.save_pretrained("./llama2-7b-gptq-4bit")

print("Quantization complete!")
```

### 12.2.5 GPTQ vs bitsandbytes 对比

<div data-component="GPTQvsBitsAndBytesComparison"></div>

| 维度 | GPTQ | bitsandbytes (NF4) |
|------|------|-------------------|
| **量化算法** | Optimal Brain Quantization | NormalFloat + 双重量化 |
| **需要校准数据** | ✓（128-256 samples） | ✗（零校准） |
| **量化时间** | 5-10 分钟 | 秒级（加载时量化） |
| **推理速度** | **更快**（kernel 优化） | 快 |
| **显存占用** | 4.5GB（4-bit） | 4.8GB（4-bit + paged） |
| **精度** | **更高**（PPL ~6.1） | 略低（PPL ~6.3） |
| **微调支持** | 困难（需解量化） | **原生支持**（QLoRA） |
| **适用场景** | 纯推理部署 | 推理 + 微调 |

**性能基准（LLaMA-7B on A100）**：

| 配置 | 困惑度 | 推理速度 | 显存 |
|------|--------|---------|------|
| FP16 | 5.68 | 18 tokens/s | 14GB |
| GPTQ 4-bit | 6.12 (+0.44) | **35 tokens/s** | 4.5GB |
| bitsandbytes 4-bit | 6.28 (+0.60) | 28 tokens/s | 4.8GB |

**选择建议**：
- **纯推理 + 追求速度** → GPTQ
- **需要微调** → bitsandbytes (QLoRA)
- **无校准数据** → bitsandbytes

---

## 12.3 AWQ 量化

### 12.3.1 Activation-aware Weight Quantization 原理

**AWQ 核心创新**：**保护重要权重通道**，而非均匀量化所有权重。

**观察**：

在 LLM 中，1% 的权重通道贡献了 **80%** 的输出幅度（幂律分布）。

**AWQ 策略**：

1. **识别重要通道**（salient channels）：
   $$
   s_i = \frac{1}{N} \sum_{j=1}^{N} | \mathbf{X}_{ij} \cdot \mathbf{W}_i |
   $$
   其中 $\mathbf{X}$ 是激活值，$s_i$ 是第 $i$ 个通道的重要性分数。

2. **Per-Channel Scaling**：
   $$
   \mathbf{W}_i' = \alpha_i \cdot \mathbf{W}_i, \quad \mathbf{X}_i' = \frac{\mathbf{X}_i}{\alpha_i}
   $$
   重要通道的 $\alpha_i > 1$（放大权重，减少量化误差）

3. **量化缩放后的权重**：
   $$
   \mathbf{W}_{q,i} = \text{Quant}(\mathbf{W}_i')
   $$

<div data-component="AWQChannelProtection"></div>

**数学直觉**：

通过 **等价变换**（$\mathbf{W} \mathbf{X} = (\alpha \mathbf{W}) \cdot (\mathbf{X}/\alpha)$），将量化误差转移到**不重要的激活值**上。

**与 GPTQ 对比**：

- **GPTQ**：基于 Hessian 全局优化（更慢，精度更高）
- **AWQ**：基于激活值统计启发式（更快，精度略低）

### 12.3.2 安装 AutoAWQ

```bash
# 方式 1：从 PyPI 安装
pip install autoawq

# 方式 2：从源码编译（最新特性）
git clone https://github.com/casper-hansen/AutoAWQ
cd AutoAWQ
pip install -e .

# 验证
python -c "from awq import AutoAWQForCausalLM; print('AWQ ready!')"
```

### 12.3.3 AWQ 量化流程

**加载预量化模型**：

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 加载 AWQ 4-bit 模型
model = AutoAWQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-AWQ",
    fuse_layers=True,  # 融合层加速
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7B-AWQ")

# 推理
prompt = "Explain quantum computing in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

**手动量化模型**：

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset

# 1. 加载模型
model_path = "meta-llama/Llama-2-7b-hf"
quant_path = "llama2-7b-awq-4bit"

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. 准备校准数据（AWQ 只需 ~128 samples）
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:512]")

def preprocess(example):
    return tokenizer(
        example["text"],
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )

calib_data = [preprocess(d)["input_ids"] for d in dataset if len(d["text"]) > 100]
calib_data = calib_data[:128]  # AWQ 推荐 128 samples

# 3. 量化配置
quant_config = {
    "zero_point": True,  # 使用 zero_point（非对称量化）
    "q_group_size": 128,  # 分组大小
    "w_bit": 4,  # 权重位数
    "version": "GEMM",  # kernel 版本
}

# 4. 执行量化（约 3-5 分钟）
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calib_data,
)

# 5. 保存
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f"AWQ quantization saved to {quant_path}")
```

### 12.3.4 推理加速效果

**性能基准（LLaMA-7B on RTX 4090）**：

| 配置 | Batch=1 延迟 | Batch=32 吞吐 | 显存 |
|------|-------------|--------------|------|
| FP16 | 55 ms/token | 420 tokens/s | 14GB |
| AWQ 4-bit | **28 ms/token** | **780 tokens/s** | 4.2GB |
| GPTQ 4-bit | 31 ms/token | 720 tokens/s | 4.5GB |

**AWQ 优势**：

- **kernel 融合**：`fuse_layers=True` 将 QKV projection 融合
- **GEMM 优化**：专门优化的 4-bit GEMM kernel
- **更低延迟**：batch=1 场景下比 GPTQ 快 ~10%

**实测代码**：

```python
import time
import torch

# FP16 基准
model_fp16 = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
)

# AWQ 4-bit
model_awq = AutoAWQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-AWQ",
    fuse_layers=True,
)

# 测试
prompt = "Once upon a time" * 50  # 长序列
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# FP16 推理
start = time.time()
with torch.no_grad():
    outputs = model_fp16.generate(**inputs, max_new_tokens=100)
fp16_time = time.time() - start

# AWQ 推理
start = time.time()
with torch.no_grad():
    outputs = model_awq.generate(**inputs, max_new_tokens=100)
awq_time = time.time() - start

print(f"FP16: {fp16_time:.2f}s")
print(f"AWQ:  {awq_time:.2f}s ({fp16_time/awq_time:.1f}x faster)")

# 输出示例：
# FP16: 5.48s
# AWQ:  2.91s (1.9x faster)
```

---

## 12.4 其他量化方法

### 12.4.1 GGUF/GGML（llama.cpp 生态）

**GGUF（GPT-Generated Unified Format）** 是 llama.cpp 使用的量化格式。

**特点**：

- **CPU 推理优化**（AVX2/AVX512/NEON）
- 支持 **2-8 bit 量化**
- **mmap 加载**（快速启动）
- 跨平台（Mac M1/M2、Windows、Linux）

**量化类型**：

| 类型 | 位数 | 大小（7B） | 质量 |
|------|------|-----------|------|
| Q4_0 | 4.5 bit | 3.8GB | 中等 |
| Q4_K_M | 4.8 bit | 4.1GB | 较好 |
| Q5_K_M | 5.6 bit | 4.8GB | 好 |
| Q8_0 | 8.5 bit | 7.2GB | 很好 |

**使用示例（llama.cpp）**：

```bash
# 转换 HF 模型到 GGUF
python convert-hf-to-gguf.py \
    --model meta-llama/Llama-2-7b-hf \
    --outfile llama2-7b-f16.gguf

# 量化
./quantize llama2-7b-f16.gguf llama2-7b-q4_k_m.gguf q4_k_m

# CPU 推理
./main -m llama2-7b-q4_k_m.gguf \
    -p "Once upon a time" \
    -n 100 \
    -t 8  # 8 线程
```

**Python 绑定（llama-cpp-python）**：

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./llama2-7b-q4_k_m.gguf",
    n_ctx=2048,  # 上下文长度
    n_threads=8,  # CPU 线程
)

output = llm(
    "Q: What is the capital of France?\nA:",
    max_tokens=50,
    stop=["Q:", "\n"],
)

print(output["choices"][0]["text"])
# Output: " Paris."
```

### 12.4.2 HQQ（Half-Quadratic Quantization）

**HQQ** 是一种无需校准数据的快速量化方法。

**优势**：
- **零校准**（vs GPTQ/AWQ 需要数据）
- **快速量化**（秒级）
- 支持 **2/3/4 bit**

```python
from transformers import AutoModelForCausalLM
from hqq.engine.hf import HQQModelForCausalLM

# 加载 + 自动 HQQ 量化
model = HQQModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",
    # HQQ 配置
    quant_config={
        "weight_quant_params": {
            "nbits": 4,
            "group_size": 64,
        },
    },
)

# 推理（与标准 HF 模型一致）
inputs = tokenizer("Hello", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=50)
```

### 12.4.3 EETQ（Efficient Exact Token Quantization）

**EETQ** 针对 **INT8 权重量化** 优化（而非 4-bit）。

**特点**：
- **INT8 精度**（vs 4-bit 的 GPTQ/AWQ）
- **kernel 优化**（比 PyTorch native INT8 快 2x）
- 适合对精度要求高的场景

```bash
pip install eetq
```

```python
from transformers import AutoModelForCausalLM, EetqConfig

eetq_config = EetqConfig("int8")

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=eetq_config,
    device_map="auto",
)

# INT8 推理（显存 ~7GB，精度接近 FP16）
```

### 12.4.4 SmoothQuant

**SmoothQuant** 通过 **平滑激活值异常值** 来提升量化精度。

**问题**：

LLM 激活值存在极端异常值（outliers），导致 INT8 量化后精度大幅下降。

**解决方案**：

1. **迁移难度**：将量化难度从激活值转移到权重
   $$
   \mathbf{Y} = \mathbf{W} \mathbf{X} = (\mathbf{W} \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \mathbf{X})
   $$

2. **平滑因子**：
   $$
   s_i = \max(|\mathbf{X}_i|)^\alpha / \max(|\mathbf{W}_i|)^{1-\alpha}
   $$
   $\alpha$ 控制平滑程度（通常 0.5）

```python
# SmoothQuant 示例（需要 SmoothQuant 库）
from smoothquant import smooth_lm

# 平滑模型
smoothed_model = smooth_lm(
    model,
    calibration_data,
    alpha=0.5,  # 平滑因子
)

# INT8 量化
quantized_model = torch.quantization.quantize_dynamic(
    smoothed_model,
    {torch.nn.Linear},
    dtype=torch.qint8,
)
```

---

## 12.5 量化评估

### 12.5.1 困惑度（Perplexity）对比

**困惑度（PPL）** 是语言模型的核心指标：
$$
\text{PPL} = \exp\left( -\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_{<i}) \right)
$$

越低越好（表示模型预测能力越强）。

<div data-component="PerplexityComparisonChart"></div>

**实测代码**：

```python
from datasets import load_dataset
import torch
from torch.nn import CrossEntropyLoss

def calculate_perplexity(model, tokenizer, dataset_name="wikitext", split="test"):
    dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split=split)
    
    # 编码数据
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    
    max_length = 2048
    stride = 512
    seq_len = encodings.input_ids.size(1)
    
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to("cuda")
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # 忽略前缀
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        
        if end_loc == seq_len:
            break
    
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

# 测试各量化方法
models = {
    "FP16": model_fp16,
    "GPTQ 4-bit": model_gptq,
    "AWQ 4-bit": model_awq,
    "bitsandbytes 4-bit": model_bnb,
}

for name, model in models.items():
    ppl = calculate_perplexity(model, tokenizer)
    print(f"{name:20s} PPL: {ppl:.2f}")

# 输出示例：
# FP16                 PPL: 5.68
# GPTQ 4-bit           PPL: 6.12 (+7.7%)
# AWQ 4-bit            PPL: 6.18 (+8.8%)
# bitsandbytes 4-bit   PPL: 6.28 (+10.6%)
```

**结论**：

- GPTQ 精度最高（+7.7% PPL）
- AWQ 速度最快（1.9x）
- bitsandbytes 最灵活（支持微调）

### 12.5.2 下游任务准确率

**MMLU（Massive Multitask Language Understanding）基准**：

```python
from lm_eval import evaluator

# 评估量化模型
results = evaluator.simple_evaluate(
    model="hf-causal",
    model_args=f"pretrained={model_path},dtype=float16",
    tasks=["mmlu"],
    num_fewshot=5,
    batch_size=8,
)

print(f"MMLU Accuracy: {results['results']['mmlu']['acc']:.2%}")
```

**性能对比（LLaMA-7B）**：

| 模型 | MMLU | HellaSwag | ARC-C |
|------|------|-----------|-------|
| FP16 | 46.8% | 76.1% | 48.3% |
| GPTQ 4-bit | 46.2% (-0.6%) | 75.8% | 47.9% |
| AWQ 4-bit | 46.0% (-0.8%) | 75.6% | 47.7% |
| BNB 4-bit | 45.5% (-1.3%) | 75.2% | 47.3% |

### 12.5.3 推理吞吐量

<div data-component="QuantizationThroughputComparison"></div>

**测试代码**：

```python
import time
import torch

def benchmark_throughput(model, tokenizer, batch_size=8, seq_length=512):
    # 生成测试数据
    input_ids = torch.randint(0, 50000, (batch_size, seq_length)).to("cuda")
    
    # 预热
    for _ in range(5):
        with torch.no_grad():
            model(input_ids)
    
    # 测试
    torch.cuda.synchronize()
    start = time.time()
    
    num_iterations = 20
    for _ in range(num_iterations):
        with torch.no_grad():
            model(input_ids)
    
    torch.cuda.synchronize()
    end = time.time()
    
    total_tokens = batch_size * seq_length * num_iterations
    throughput = total_tokens / (end - start)
    
    return throughput

# 测试
models = {...}  # 同上

for name, model in models.items():
    throughput = benchmark_throughput(model, tokenizer)
    print(f"{name:20s} Throughput: {throughput:.0f} tokens/s")

# 输出示例（RTX 4090）：
# FP16                 Throughput: 4200 tokens/s
# GPTQ 4-bit           Throughput: 7800 tokens/s (1.86x)
# AWQ 4-bit            Throughput: 8500 tokens/s (2.02x)
# bitsandbytes 4-bit   Throughput: 6900 tokens/s (1.64x)
```

### 12.5.4 模型大小压缩比

**存储对比（LLaMA-7B）**：

| 格式 | 模型大小 | 压缩比 | 加载时间 |
|------|---------|-------|---------|
| FP32 | 26.0 GB | 1.0x | 15s |
| FP16 | 13.0 GB | 2.0x | 8s |
| GPTQ 4-bit | 3.8 GB | 6.8x | 12s（解压） |
| AWQ 4-bit | 3.6 GB | 7.2x | 10s |
| GGUF Q4_K_M | 4.1 GB | 6.3x | 2s（mmap） |

**实际部署建议**：

- **云部署（GPU）**：AWQ 或 GPTQ（速度优先）
- **边缘设备（CPU）**：GGUF（跨平台 + mmap）
- **需要微调**：bitsandbytes（QLoRA 支持）
- **研究/评估**：FP16（基准）

---

## 12.6 量化最佳实践

### 12.6.1 选择量化方法决策树

```
是否需要微调模型？
├─ 是 → bitsandbytes (QLoRA)
└─ 否 → 继续
    ├─ 是否有校准数据？
    │   ├─ 有 → GPTQ 或 AWQ
    │   │   ├─ 追求精度 → GPTQ
    │   │   └─ 追求速度 → AWQ
    │   └─ 无 → bitsandbytes 或 HQQ
    └─ 部署平台？
        ├─ GPU → AWQ/GPTQ
        └─ CPU → GGUF (llama.cpp)
```

### 12.6.2 量化前后性能验证清单

- [ ] **困惑度测试**（WikiText-2）：增幅 <10%
- [ ] **下游任务准确率**（MMLU）：下降 <2%
- [ ] **推理速度**：至少 1.5x 加速
- [ ] **显存占用**：压缩至 1/3 以下
- [ ] **生成质量**（人工评估）：无明显退化

### 12.6.3 常见问题排查

**问题 1：量化后模型输出乱码**

```python
# 原因：tokenizer 未正确加载
# 解决方案：
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=True,  # 使用 fast tokenizer
    trust_remote_code=True,  # 自定义模型需要
)
```

**问题 2：GPTQ 量化时 CUDA OOM**

```python
# 解决方案：减小 batch_size
model.quantize(
    calibration_data,
    batch_size=1,  # 从 4 降到 1
)
```

**问题 3：AWQ 加载失败（kernel 错误）**

```bash
# 重新编译 kernel
pip uninstall autoawq -y
pip install autoawq --no-cache-dir
```

---

## 12.7 总结与展望

### 12.7.1 量化方法对比总结

<div data-component="QuantizationMethodSummaryTable"></div>

| 方法 | 校准数据 | 量化时间 | 推理速度 | 精度 | 微调 | 推荐场景 |
|------|---------|---------|---------|------|------|---------|
| **GPTQ** | ✓ | 5-10 min | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✗ | 生产推理 |
| **AWQ** | ✓ | 3-5 min | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✗ | 低延迟推理 |
| **bitsandbytes** | ✗ | <1 min | ⭐⭐⭐ | ⭐⭐⭐ | ✓ | QLoRA微调 |
| **HQQ** | ✗ | <1 min | ⭐⭐⭐ | ⭐⭐ | ✗ | 快速实验 |
| **GGUF** | ✗ | 2 min | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✗ | CPU部署 |

### 12.7.2 未来趋势

1. **更低比特量化**：2-bit、1.58-bit（BitNet）
2. **混合精度量化**：关键层 FP16 + 其他层 4-bit
3. **硬件协同**：NPU 原生支持 INT4/INT8
4. **动态量化**：根据输入难度自适应调整精度

### 12.7.3 代码模板

**快速量化推理**：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 选择量化方法
METHOD = "AWQ"  # "GPTQ" | "AWQ" | "bitsandbytes"

# 2. 加载模型
if METHOD == "GPTQ":
    model = AutoModelForCausalLM.from_pretrained(
        "TheBloke/Llama-2-7B-GPTQ",
        device_map="auto",
    )
elif METHOD == "AWQ":
    from awq import AutoAWQForCausalLM
    model = AutoAWQForCausalLM.from_quantized(
        "TheBloke/Llama-2-7B-AWQ",
        fuse_layers=True,
    )
else:  # bitsandbytes
    from transformers import BitsAndBytesConfig
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    )

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 3. 推理
prompt = "Explain quantum computing:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

---

通过本章学习，你应该能够：
- ✅ 理解 PTQ 与 QAT 的本质差异
- ✅ 掌握 GPTQ、AWQ、bitsandbytes 的原理与用法
- ✅ 根据场景选择最优量化方法
- ✅ 评估量化模型的质量（PPL、MMLU、吞吐量）
- ✅ 排查常见量化问题

**下一章预告**：Chapter 13 将深入 Gradient Checkpointing 与内存优化，讲解如何在单卡 24GB GPU 上训练 70B 模型！

# Chapter 21: Optimum 库详解 - 硬件加速统一接口

## 21.1 Optimum 生态概览

### 21.1.1 什么是 Optimum？

**Optimum** 是 Hugging Face 开发的硬件加速器适配层，提供统一接口支持多种推理后端：

```
┌─────────────────────────────────────────────┐
│  Hugging Face Transformers (统一 API)       │
└───────────────┬─────────────────────────────┘
                │
┌───────────────▼─────────────────────────────┐
│  Optimum (硬件抽象层)                        │
│  ┌──────────┬──────────┬──────────┬────────┐
│  │ ONNX RT  │ Intel    │ Habana   │ AWS    │
│  │ (通用)   │ (CPU)    │ (Gaudi)  │ (Inf2) │
│  └──────────┴──────────┴──────────┴────────┘
└───────────────┬─────────────────────────────┘
                │
┌───────────────▼─────────────────────────────┐
│  硬件层                                     │
│  CPU / NVIDIA GPU / Intel / Habana / AWS   │
└─────────────────────────────────────────────┘
```

---

### 21.1.2 核心理念

**"Write once, run anywhere"** - 一套代码，多个硬件平台：

```python
# 统一的接口
from optimum.onnxruntime import ORTModelForSequenceClassification  # CPU/GPU
from optimum.intel import OVModelForSequenceClassification        # Intel OpenVINO
from optimum.habana import GaudiConfig, GaudiTrainer              # Habana Gaudi

# 相同的 API
model = ORTModelForSequenceClassification.from_pretrained("bert-base-uncased")
outputs = model(**inputs)  # 自动选择最优后端
```

---

### 21.1.3 支持的后端

<div data-component="OptimumBackendEcosystem"></div>

| 后端 | 硬件 | 适用场景 | Optimum 包 |
|-----|------|---------|-----------|
| **ONNX Runtime** | CPU / NVIDIA GPU / AMD | 通用推理 | `optimum[onnxruntime]` |
| **Intel** | Intel CPU / GPU | Intel 硬件优化 | `optimum[intel]` |
| **Habana** | Gaudi / Gaudi2 | 训练 + 推理 | `optimum[habana]` |
| **AWS** | Inferentia / Trainium | AWS 云部署 | `optimum[neuron]` |
| **AMD** | ROCm GPU | AMD GPU | `optimum[amd]` |
| **BetterTransformer** | PyTorch 原生 | FastPath 优化 | 内置 |

---

## 21.2 ONNX Runtime 加速

### 21.2.1 快速上手

**安装**：

```bash
pip install optimum[onnxruntime]

# GPU 版本
pip install optimum[onnxruntime-gpu]
```

**基本用法**：

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

# 自动导出 + 加载 ONNX
model = ORTModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english",
    export=True  # 首次运行导出 ONNX
)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# 推理（与 Transformers API 一致）
inputs = tokenizer("This movie is amazing!", return_tensors="pt")
outputs = model(**inputs)

print(outputs.logits.argmax().item())  # 1 (positive)
```

---

### 21.2.2 支持的任务

**文本任务**：

```python
from optimum.onnxruntime import (
    ORTModelForSequenceClassification,  # 分类
    ORTModelForTokenClassification,     # NER
    ORTModelForQuestionAnswering,       # QA
    ORTModelForCausalLM,                # 生成
    ORTModelForSeq2SeqLM,               # Seq2Seq
    ORTModelForFeatureExtraction,       # 特征提取
    ORTModelForMaskedLM,                # MLM
)
```

**多模态**：

```python
from optimum.onnxruntime import (
    ORTModelForImageClassification,     # 图像分类
    ORTModelForVision2Seq,              # 图像描述
    ORTModelForAudioClassification,     # 音频分类
)
```

---

### 21.2.3 动态量化

**CPU 推理加速**：

```python
from optimum.onnxruntime import ORTQuantizer, ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# 1. 导出 ONNX
model = ORTModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    export=True
)

# 2. 动态量化配置
quantization_config = AutoQuantizationConfig.avx512_vnni(
    is_static=False,  # 动态量化
    per_channel=True  # 逐通道量化
)

# 3. 量化
quantizer = ORTQuantizer.from_pretrained(model)
quantizer.quantize(
    save_dir="./bert_onnx_int8",
    quantization_config=quantization_config
)

# 4. 加载量化模型
quantized_model = ORTModelForSequenceClassification.from_pretrained("./bert_onnx_int8")
```

**性能对比**（BERT-base，Intel CPU）：

| 方法 | 推理速度 (samples/s) | 模型大小 | 精度损失 |
|-----|---------------------|---------|---------|
| PyTorch FP32 | 42 | 438 MB | 0% |
| ONNX FP32 | 118 (+181%) | 438 MB | 0% |
| ONNX INT8 (动态) | 178 (+324%) | 110 MB | < 1% |

---

### 21.2.4 静态量化

**需要校准数据集**（精度更高）：

<div data-component="QuantizationWorkflowVisualizer"></div>

```python
from datasets import load_dataset
from optimum.onnxruntime import ORTQuantizer, ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# 准备校准数据
def preprocess_fn(examples, tokenizer):
    return tokenizer(
        examples["sentence"],
        padding=True,
        truncation=True,
        max_length=128
    )

dataset = load_dataset("glue", "sst2", split="train[:1000]")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
calibration_dataset = dataset.map(
    lambda x: preprocess_fn(x, tokenizer),
    batched=True,
    remove_columns=dataset.column_names
)

# 静态量化配置
quantization_config = AutoQuantizationConfig.avx512_vnni(
    is_static=True,  # 静态量化
    per_channel=False
)

# 量化（包含校准）
model = ORTModelForSequenceClassification.from_pretrained("bert-base-uncased", export=True)
quantizer = ORTQuantizer.from_pretrained(model)
quantizer.quantize(
    save_dir="./bert_static_int8",
    quantization_config=quantization_config,
    calibration_dataset=calibration_dataset
)
```

**静态 vs 动态量化**：

| 类型 | 激活值量化 | 精度 | 速度 | 校准数据 |
|-----|-----------|------|------|---------|
| 动态 | 运行时量化 | 中等 | 较快 | ❌ 不需要 |
| 静态 | 预先量化 | 高 | 最快 | ✅ 需要 |

---

### 21.2.5 图优化级别

**控制 ONNX Runtime 优化**：

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import OptimizationConfig

model = ORTModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    export=True
)

# 优化配置
optimization_config = OptimizationConfig(
    optimization_level=99,  # 0 (禁用) | 1 (基础) | 2 (扩展) | 99 (全部)
    optimize_for_gpu=True,  # GPU 优化
    fp16=True,  # FP16 混合精度
    
    # Transformer 特定优化
    enable_gelu_approximation=True,  # GELU 近似
    enable_layer_norm_fusion=True,   # LayerNorm 融合
    enable_attention_fusion=True,    # Attention 融合
    enable_skip_layer_norm_fusion=True,  # SkipLayerNorm 融合
    enable_embed_layer_norm_fusion=True,  # EmbedLayerNorm 融合
    
    # 通用优化
    enable_gemm_fusion=True,  # Gemm 融合
    enable_bias_gelu_fusion=True,  # BiasGELU 融合
)

# 应用优化
model.optimize(optimization_config, save_dir="./bert_optimized")
```

**优化级别效果**：

| Level | 优化内容 | 加速比 (BERT) |
|-------|---------|---------------|
| 0 | 无优化 | 1.0x |
| 1 | 常量折叠、死代码消除 | 1.2x |
| 2 | + 算子融合 | 1.8x |
| 99 | + Transformer 优化、FP16 | **2.5x** |

---

## 21.3 Intel 优化

### 21.3.1 Intel Neural Compressor（量化）

**安装**：

```bash
pip install optimum[neural-compressor]
```

**训练后量化**（PTQ）：

```python
from optimum.intel import INCQuantizer
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# 配置量化
quantizer = INCQuantizer.from_pretrained(model)

# 量化（自动选择最优策略）
quantizer.quantize(
    quantization_config={
        "approach": "dynamic",  # 动态量化
    },
    save_directory="./bert_inc_int8"
)
```

**量化感知训练**（QAT）：

```python
from optimum.intel import INCTrainer, INCConfig

# 配置 QAT
inc_config = INCConfig(
    quantization_approach="qat",  # 量化感知训练
    metrics=["accuracy"],
    accuracy_criterion={"relative": 0.01}  # 精度损失 < 1%
)

# 训练器
trainer = INCTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    quantization_config=inc_config
)

# 训练（自动插入量化节点）
trainer.train()
trainer.save_model("./bert_qat")
```

---

### 21.3.2 OpenVINO（推理加速）

**安装**：

```bash
pip install optimum[openvino]
```

**导出 OpenVINO IR**：

```python
from optimum.intel import OVModelForSequenceClassification

# 导出 + 加载
model = OVModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    export=True  # 导出为 OpenVINO IR
)

# 保存
model.save_pretrained("./bert_openvino")

# 推理
inputs = tokenizer("OpenVINO is fast!", return_tensors="pt")
outputs = model(**inputs)
```

**性能优化**：

```python
from optimum.intel import OVConfig

# 优化配置
ov_config = OVConfig(
    compression="INT8",  # INT8 量化
    quantization_config={
        "algorithm": "DefaultQuantization",
        "preset": "performance",  # 性能优先 (或 "mixed" 精度优先)
    }
)

# 导出优化模型
model = OVModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    export=True,
    ov_config=ov_config
)
```

**CPU 推理对比**（Intel Xeon，BERT-base）：

| 框架 | 推理速度 (samples/s) | 加速比 |
|-----|---------------------|--------|
| PyTorch (CPU) | 42 | 1.0x |
| ONNX Runtime (CPU) | 118 | 2.8x |
| OpenVINO (FP32) | 156 | 3.7x |
| OpenVINO (INT8) | 168 | **4.0x** |

---

## 21.4 其他后端

### 21.4.1 Habana Gaudi（训练加速）

**Habana Gaudi** 是专为 AI 训练设计的加速器：

**安装**：

```bash
pip install optimum[habana]
```

**训练配置**：

```python
from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments

# Gaudi 配置
gaudi_config = GaudiConfig()
gaudi_config.use_habana_mixed_precision = True  # 混合精度

# 训练参数
training_args = GaudiTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=32,
    use_habana=True,  # 启用 Gaudi
    use_lazy_mode=True,  # Lazy 模式（性能更好）
    gaudi_config_name=gaudi_config
)

# 训练器
trainer = GaudiTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()
```

**性能对比**（BERT-large，批大小 32）：

| 硬件 | 训练速度 (samples/s) | 成本效率 |
|-----|---------------------|---------|
| NVIDIA A100 (40GB) | 142 | 1.0x |
| Habana Gaudi2 | 168 | **1.6x** |

---

### 21.4.2 AWS Inferentia（云端推理）

**AWS Inferentia** 是 AWS 的推理加速芯片：

**安装**（需在 AWS EC2 Inf1 实例）：

```bash
pip install optimum[neuron]
```

**导出 Neuron 模型**：

```python
from optimum.neuron import NeuronModelForSequenceClassification

# 导出
model = NeuronModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    export=True,
    batch_size=1,  # 固定批大小
    sequence_length=128  # 固定序列长度
)

# 保存
model.save_pretrained("./bert_neuron")
```

**推理**：

```python
# 加载 Neuron 模型
model = NeuronModelForSequenceClassification.from_pretrained("./bert_neuron")

# 推理（自动使用 Inferentia）
inputs = tokenizer("AWS Inferentia is efficient!", return_tensors="pt")
outputs = model(**inputs)
```

**成本对比**（推理吞吐量 / 美元）：

| 实例类型 | 吞吐量 (samples/s) | 每小时成本 | 成本效率 |
|---------|-------------------|-----------|---------|
| p3.2xlarge (V100) | 450 | $3.06 | 147 |
| inf1.xlarge (Inferentia) | 580 | $0.368 | **1576** (10.7x) |

---

### 21.4.3 AMD ROCm（AMD GPU）

**支持 AMD GPU 推理**：

```bash
pip install optimum[amd]
```

```python
from optimum.amd import ROCmModelForCausalLM

model = ROCmModelForCausalLM.from_pretrained(
    "gpt2",
    device_map="auto"  # 自动分配到 AMD GPU
)
```

---

## 21.5 Optimum + PEFT 联合优化

### 21.5.1 导出 LoRA 适配器

**训练 LoRA**：

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification

base_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1
)

# 添加 LoRA
model = get_peft_model(base_model, lora_config)
trainer.train()

# 保存 LoRA 适配器
model.save_pretrained("./bert_lora")
```

**导出 ONNX（合并 LoRA）**：

```python
from optimum.onnxruntime import ORTModelForSequenceClassification

# 加载基础模型 + LoRA
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
model = PeftModel.from_pretrained(model, "./bert_lora")

# 合并 LoRA 权重
merged_model = model.merge_and_unload()

# 导出 ONNX
from optimum.exporters.onnx import main_export
main_export(
    model=merged_model,
    output="./bert_lora_onnx",
    task="text-classification"
)
```

---

### 21.5.2 量化 + PEFT 联合优化

**QLoRA + ONNX Runtime**：

```python
from peft import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 1. 加载 4-bit 基础模型
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# 2. 准备 LoRA 训练
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)

# 3. 训练
trainer.train()

# 4. 合并 + 导出（需先反量化）
model = model.merge_and_unload()
model.save_pretrained("./llama_lora_merged")

# 5. 导出 ONNX 并重新量化
from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
ort_model = ORTModelForCausalLM.from_pretrained("./llama_lora_merged", export=True)
quantizer = ORTQuantizer.from_pretrained(ort_model)
quantizer.quantize(save_dir="./llama_lora_onnx_int8")
```

**效果**（LLaMA-7B）：

| 方法 | 显存占用 | 推理速度 | 模型大小 |
|-----|---------|---------|---------|
| 原始 FP16 | 14 GB | 1.0x | 13 GB |
| QLoRA (4-bit) | 4.5 GB | 0.8x | 3.5 GB |
| ONNX INT8 (合并 LoRA) | 2.1 GB | 2.3x | 1.8 GB |

---

## 21.6 实战案例：多后端自动选择

**目标**：根据硬件自动选择最优后端

<div data-component="BackendAutoSelector"></div>

```python
import platform
import torch
from typing import Literal

def get_optimal_backend() -> Literal["onnxruntime", "openvino", "neuron"]:
    """自动检测硬件并选择最优后端"""
    
    # 检查是否在 AWS Inferentia
    try:
        import torch_neuron
        return "neuron"
    except ImportError:
        pass
    
    # 检查 CPU 类型
    if platform.processor().startswith("Intel"):
        return "openvino"  # Intel CPU 优先 OpenVINO
    
    # 默认 ONNX Runtime
    return "onnxruntime"

def load_optimized_model(model_name: str, task: str = "sequence-classification"):
    """根据后端加载优化模型"""
    backend = get_optimal_backend()
    
    if backend == "onnxruntime":
        from optimum.onnxruntime import ORTModelForSequenceClassification
        model = ORTModelForSequenceClassification.from_pretrained(
            model_name,
            export=True,
            provider="CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
        )
    
    elif backend == "openvino":
        from optimum.intel import OVModelForSequenceClassification
        model = OVModelForSequenceClassification.from_pretrained(
            model_name,
            export=True
        )
    
    elif backend == "neuron":
        from optimum.neuron import NeuronModelForSequenceClassification
        model = NeuronModelForSequenceClassification.from_pretrained(
            model_name,
            export=True,
            batch_size=1,
            sequence_length=128
        )
    
    print(f"✅ Loaded model with {backend} backend")
    return model

# 使用
model = load_optimized_model("distilbert-base-uncased-finetuned-sst-2-english")
```

---

## 21.7 性能基准测试

### 21.7.1 跨后端性能对比

**测试配置**：
- 模型：BERT-base-uncased
- 任务：序列分类（SST-2）
- 批大小：8
- 序列长度：128

**CPU 推理**（Intel Xeon Gold 6230）：

| 后端 | 推理速度 (samples/s) | 延迟 P50 (ms) | 加速比 |
|-----|---------------------|--------------|--------|
| PyTorch Eager | 42 | 190 | 1.0x |
| ONNX Runtime (FP32) | 118 | 68 | 2.8x |
| ONNX Runtime (INT8) | 178 | 45 | 4.2x |
| OpenVINO (FP32) | 156 | 51 | 3.7x |
| OpenVINO (INT8) | 168 | 48 | **4.0x** |

**GPU 推理**（NVIDIA A100）：

| 后端 | 推理速度 (samples/s) | 延迟 P50 (ms) | 加速比 |
|-----|---------------------|--------------|--------|
| PyTorch Eager (FP32) | 520 | 15.4 | 1.0x |
| PyTorch (FP16) | 1240 | 6.5 | 2.4x |
| ONNX Runtime (FP32) | 680 | 11.8 | 1.3x |
| ONNX Runtime (FP16) | 1420 | 5.6 | **2.7x** |
| TensorRT (FP16) | 1680 | 4.8 | **3.2x** |

---

### 21.7.2 量化精度对比

**GLUE 基准测试**（BERT-base）：

| 量化方法 | SST-2 准确率 | MNLI (m/mm) | QQP F1 | 平均精度损失 |
|---------|-------------|-------------|--------|-------------|
| FP32（基线） | 92.3% | 84.5% / 84.8% | 91.2% | 0% |
| 动态 INT8 | 91.8% | 84.1% / 84.3% | 90.9% | **-0.5%** |
| 静态 INT8 | 92.1% | 84.3% / 84.6% | 91.0% | **-0.3%** |
| QAT INT8 | 92.2% | 84.4% / 84.7% | 91.1% | **-0.1%** |

**结论**：量化精度损失 < 1%，可接受

---

## 21.8 常见问题

### Q1: 如何选择量化方法（动态 vs 静态 vs QAT）？

**决策树**：

```
是否有校准数据集？
├─ 否 → 动态量化（最简单）
└─ 是
    └─ 精度要求是否严格？
        ├─ 否 → 静态量化（更快）
        └─ 是 → QAT（精度最高）
```

**推荐**：
- 开发阶段：动态量化
- 生产部署：静态量化
- 精度敏感任务（医疗、金融）：QAT

### Q2: Optimum 与原生 Transformers 性能差异？

**推理速度对比**（BERT-base，CPU）：

```python
# Transformers（无优化）
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
# 速度: 42 samples/s

# Optimum（ONNX Runtime INT8）
from optimum.onnxruntime import ORTModelForSequenceClassification
model = ORTModelForSequenceClassification.from_pretrained("bert-base-uncased", export=True)
# 速度: 178 samples/s (4.2x 加速)
```

### Q3: 如何调试 ONNX 导出错误？

**常见错误 1**：`Unsupported operator`

```python
# 解决方案：升级 ONNX opset
from optimum.exporters.onnx import main_export
main_export(
    model_name_or_path="model",
    output="./onnx",
    opset=14  # 指定更高版本
)
```

**常见错误 2**：`Dynamic axes not supported`

```python
# 解决方案：固定输入形状
from optimum.exporters.onnx import OnnxConfig

class CustomOnnxConfig(OnnxConfig):
    @property
    def inputs(self):
        return {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"}
        }
```

---

## 21.9 总结

### 核心要点

1. **Optimum 优势**：
   - 统一 API，多后端支持
   - 自动导出 + 优化
   - 与 Transformers 生态无缝集成

2. **后端选择**：
   - CPU：OpenVINO (Intel) / ONNX Runtime
   - NVIDIA GPU：ONNX Runtime / TensorRT
   - Habana：Gaudi（训练）
   - AWS：Inferentia（低成本推理）

3. **量化策略**：
   - 动态 INT8：快速部署，精度损失 < 1%
   - 静态 INT8：更快推理，需校准数据
   - QAT：最高精度，需重新训练

4. **性能提升**：
   - CPU：3-4x 加速（INT8 量化）
   - GPU：1.3-2.7x 加速（ONNX Runtime）
   - 成本：AWS Inferentia 成本效率 10x+

### 最佳实践

```python
# 1. 开发阶段：PyTorch Eager
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# 2. 优化阶段：ONNX Runtime + 动态量化
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
model = ORTModelForSequenceClassification.from_pretrained("bert-base-uncased", export=True)
quantizer = ORTQuantizer.from_pretrained(model)
quantizer.quantize(save_dir="./model_int8")

# 3. 生产部署：根据硬件选择最优后端
# - Intel CPU → OpenVINO
# - NVIDIA GPU → TensorRT
# - AWS → Inferentia
```

---

## 21.10 扩展阅读

1. **Optimum 官方文档**：https://huggingface.co/docs/optimum
2. **ONNX Runtime 性能调优**：https://onnxruntime.ai/docs/performance/
3. **Intel OpenVINO 指南**：https://docs.openvino.ai/
4. **Habana Gaudi 文档**：https://docs.habana.ai/
5. **AWS Neuron SDK**：https://awsdocs-neuron.readthedocs-hosted.com/

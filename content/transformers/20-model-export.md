# Chapter 20: 模型导出与转换 - 生产部署基础

## 20.1 模型序列化格式

### 20.1.1 PyTorch 格式（.bin、.pt、.pth）

**PyTorch 原生格式**使用 Python Pickle 序列化：

```python
import torch
from transformers import AutoModelForCausalLM

# 加载模型
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 保存完整模型（包含架构 + 权重）
torch.save(model, "model_full.pt")

# 保存仅权重（推荐）
torch.save(model.state_dict(), "model_weights.bin")

# 加载
model = torch.load("model_full.pt")  # 完整模型
# 或
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.load_state_dict(torch.load("model_weights.bin"))
```

**PyTorch 格式的问题**：

1. **安全风险**：Pickle 可以执行任意 Python 代码
   ```python
   # 恶意 pickle 文件可以执行任意代码
   import pickle
   import os
   
   class Exploit:
       def __reduce__(self):
           return (os.system, ('rm -rf /',))  # 危险！
   
   pickle.dumps(Exploit())  # 反序列化时会执行
   ```

2. **跨版本兼容性差**：PyTorch 版本变化可能导致加载失败

3. **加载速度慢**：需要完整反序列化

---

### 20.1.2 Safetensors（推荐）

**Safetensors** 是 Hugging Face 开发的安全、快速的张量序列化格式：

#### 核心优势

<div data-component="SafetensorsVsPickleComparison"></div>

**1. 安全性**：
- 零代码执行（纯数据格式）
- 防止任意代码注入
- 文件头包含完整元数据

**2. 加载速度快**：

| 模型 | PyTorch .bin | Safetensors | 加速比 |
|-----|--------------|-------------|--------|
| BERT-base | 3.2s | 0.8s | 4.0x |
| GPT-2 | 2.1s | 0.5s | 4.2x |
| LLaMA-7B | 147s | 32s | 4.6x |
| LLaMA-70B | 1420s | 285s | **5.0x** |

**3. 支持部分加载**（内存映射）：

```python
from safetensors import safe_open

# 只加载特定张量（无需加载整个文件）
with safe_open("model.safetensors", framework="pt") as f:
    # 查看元数据
    metadata = f.metadata()
    
    # 仅加载 embedding 层
    embedding = f.get_tensor("transformer.wte.weight")
    print(embedding.shape)  # torch.Size([50257, 768])
```

#### 使用 Safetensors

**保存**：

```python
from safetensors.torch import save_file, load_file

# 保存模型权重
state_dict = model.state_dict()
save_file(state_dict, "model.safetensors")

# 保存时添加元数据
metadata = {
    "model_name": "gpt2",
    "framework": "pytorch",
    "precision": "fp16"
}
save_file(state_dict, "model.safetensors", metadata=metadata)
```

**加载**：

```python
# 完整加载
state_dict = load_file("model.safetensors")
model.load_state_dict(state_dict)

# 或使用 Transformers API
model = AutoModelForCausalLM.from_pretrained(
    "./model",  # 包含 model.safetensors 的目录
    use_safetensors=True  # 优先使用 safetensors
)
```

**转换现有模型**：

```python
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file

# 加载 PyTorch 模型
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 转换并保存
save_file(model.state_dict(), "gpt2.safetensors")
```

---

### 20.1.3 格式转换工具

#### Hugging Face CLI 转换

```bash
# 安装 safetensors
pip install safetensors

# 转换模型仓库（.bin → .safetensors）
huggingface-cli convert \
  --input ./gpt2_pytorch \
  --output ./gpt2_safetensors \
  --format safetensors

# 验证转换
python -c "
from safetensors import safe_open
with safe_open('gpt2_safetensors/model.safetensors', framework='pt') as f:
    print('Tensors:', list(f.keys())[:5])
"
```

#### 批量转换脚本

```python
from pathlib import Path
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file

def convert_to_safetensors(model_path: str, output_path: str):
    """转换 PyTorch 模型到 Safetensors"""
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    print("Converting to safetensors...")
    state_dict = model.state_dict()
    
    # 保存 safetensors
    output_file = Path(output_path) / "model.safetensors"
    save_file(state_dict, str(output_file))
    
    print(f"Saved to {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024**2:.2f} MB")

# 使用示例
convert_to_safetensors("./gpt2", "./gpt2_safe")
```

---

## 20.2 ONNX 导出

### 20.2.1 ONNX 标准概述

**ONNX**（Open Neural Network Exchange）是跨平台的神经网络交换格式：

**架构**：

```
┌────────────────────────────────────┐
│  PyTorch / TensorFlow / JAX        │  ← 训练框架
└────────────┬───────────────────────┘
             │ 导出
             ↓
┌────────────────────────────────────┐
│  ONNX Graph (IR)                   │  ← 中间表示
│  - 算子定义                         │
│  - 计算图                           │
│  - 权重数据                         │
└────────────┬───────────────────────┘
             │ 推理
             ↓
┌────────────────────────────────────┐
│  ONNX Runtime / TensorRT / ...     │  ← 推理引擎
│  - CPU / GPU / 移动端               │
└────────────────────────────────────┘
```

**核心概念**：

1. **计算图**（Computational Graph）：节点 + 边
2. **算子**（Operator）：标准化的操作（Gemm、Conv、Softmax 等）
3. **张量**（Tensor）：多维数组，包含形状和数据类型

---

### 20.2.2 使用 Optimum 导出

**Optimum** 提供统一的 ONNX 导出接口：

```python
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

# 方法 1：导出时转换
model = ORTModelForCausalLM.from_pretrained(
    "gpt2",
    export=True,  # 自动导出 ONNX
    provider="CPUExecutionProvider"  # 或 CUDAExecutionProvider
)

# 保存 ONNX 模型
model.save_pretrained("./gpt2_onnx")

# 方法 2：预先导出
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.exporters.onnx import main_export

# 导出分类模型
main_export(
    model_name_or_path="bert-base-uncased",
    output="./bert_onnx",
    task="text-classification"
)
```

**支持的任务**：

| 任务 | Optimum 类 | ONNX 导出 |
|-----|-----------|----------|
| 文本生成 | `ORTModelForCausalLM` | ✅ |
| 序列分类 | `ORTModelForSequenceClassification` | ✅ |
| Token 分类 | `ORTModelForTokenClassification` | ✅ |
| 问答 | `ORTModelForQuestionAnswering` | ✅ |
| 特征提取 | `ORTModelForFeatureExtraction` | ✅ |

---

### 20.2.3 ONNX Runtime 推理

**推理示例**：

```python
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

# 加载 ONNX 模型
model = ORTModelForCausalLM.from_pretrained("./gpt2_onnx")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 推理（与 Transformers API 一致）
inputs = tokenizer("Hello, my name is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)

print(tokenizer.decode(outputs[0]))
# 输出: Hello, my name is John and I am a software engineer...
```

**性能对比**（GPT-2，CPU）：

| 框架 | 推理速度 (samples/s) | 延迟 (ms) | 加速比 |
|-----|---------------------|----------|--------|
| PyTorch (CPU) | 12.3 | 81.3 | 1.0x |
| ONNX Runtime (CPU) | 34.7 | 28.8 | **2.82x** |

**GPU 推理**：

```python
# 指定 CUDA Provider
model = ORTModelForCausalLM.from_pretrained(
    "./gpt2_onnx",
    provider="CUDAExecutionProvider",
    provider_options={"device_id": 0}
)

# 性能提升（GPU）：1.5-2x vs PyTorch
```

---

### 20.2.4 量化 ONNX 模型

**动态量化**（推理时量化）：

```python
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

# 加载模型
model = ORTModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    export=True
)

# 配置量化
quantization_config = AutoQuantizationConfig.avx512_vnni(
    is_static=False,  # 动态量化
    per_channel=True
)

# 量化
quantizer = ORTQuantizer.from_pretrained(model)
quantizer.quantize(
    save_dir="./bert_onnx_quantized",
    quantization_config=quantization_config
)
```

**静态量化**（需要校准数据集）：

```python
from optimum.onnxruntime import ORTQuantizer, ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from datasets import load_dataset

# 准备校准数据
def preprocess_fn(examples, tokenizer):
    return tokenizer(examples["text"], padding=True, truncation=True, max_length=128)

dataset = load_dataset("glue", "sst2", split="train[:100]")
calibration_dataset = dataset.map(
    lambda x: preprocess_fn(x, tokenizer),
    batched=True
)

# 静态量化配置
quantization_config = AutoQuantizationConfig.arm64(
    is_static=True,
    per_channel=False
)

# 量化
quantizer = ORTQuantizer.from_pretrained(model)
quantizer.quantize(
    save_dir="./bert_onnx_static_quant",
    quantization_config=quantization_config,
    calibration_dataset=calibration_dataset
)
```

**量化效果**（BERT-base，CPU）：

| 方法 | 模型大小 | 推理速度 | 精度损失 |
|-----|---------|---------|---------|
| FP32 | 438 MB | 34.7 samples/s | 0% |
| 动态 INT8 | 110 MB (-75%) | 52.3 samples/s (+51%) | < 1% |
| 静态 INT8 | 110 MB (-75%) | 61.8 samples/s (+78%) | < 0.5% |

---

## 20.3 TorchScript 导出

### 20.3.1 torch.jit.trace vs torch.jit.script

<div data-component="TorchScriptModeComparison"></div>

#### Trace 模式

**原理**：记录模型在特定输入上的操作轨迹

```python
import torch
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 准备示例输入
inputs = tokenizer("Hello, world!", return_tensors="pt")

# Trace
traced_model = torch.jit.trace(
    model,
    (inputs["input_ids"], inputs["attention_mask"])
)

# 保存
traced_model.save("bert_traced.pt")

# 加载并推理
loaded_model = torch.jit.load("bert_traced.pt")
outputs = loaded_model(inputs["input_ids"], inputs["attention_mask"])
```

**优点**：
- 简单易用
- 性能优化好（CUDA Graph）

**缺点**：
- 不支持动态控制流（if、for、while）
- 输入形状必须固定

**示例问题**：

```python
# ❌ Trace 无法正确处理条件分支
class ModelWithIf(nn.Module):
    def forward(self, x, use_relu):
        if use_relu:
            return F.relu(x)
        else:
            return x

model = ModelWithIf()
traced = torch.jit.trace(model, (torch.randn(1, 10), True))
# Trace 只会记录 use_relu=True 的路径！
```

#### Script 模式

**原理**：分析 Python 代码，编译为 TorchScript IR

```python
# Script
scripted_model = torch.jit.script(model)
scripted_model.save("bert_scripted.pt")
```

**优点**：
- 支持动态控制流
- 可以处理可变输入形状

**缺点**：
- 需要 TorchScript 兼容的代码
- 部分 Python 特性不支持（列表推导、lambda 等）

**使用建议**：

| 场景 | 推荐方法 |
|-----|---------|
| 简单前向传播（无分支） | Trace |
| 包含 if/for/while | Script |
| 生成任务（动态长度） | Script |
| 最大性能优化 | Trace（固定输入） |

---

### 20.3.2 生成任务的特殊处理

**问题**：生成任务涉及动态循环（逐 token 生成）

**解决方案 1**：导出单步前向传播

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("gpt2")

# 导出单步推理（输入 1 个 token，输出 logits）
class SingleStepWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids, past_key_values=None):
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True
        )
        return outputs.logits, outputs.past_key_values

wrapper = SingleStepWrapper(model)

# Trace
example_input = torch.tensor([[50256]])  # <|endoftext|>
traced = torch.jit.trace(wrapper, (example_input,))
traced.save("gpt2_single_step.pt")
```

**解决方案 2**：使用 ONNX（推荐）

```python
from optimum.onnxruntime import ORTModelForCausalLM

# Optimum 自动处理生成逻辑
model = ORTModelForCausalLM.from_pretrained("gpt2", export=True)
model.save_pretrained("./gpt2_onnx")

# 推理时自动处理 KV Cache 和循环
outputs = model.generate(input_ids, max_new_tokens=50)
```

---

### 20.3.3 TorchScript 模型优化

**图优化**：

```python
# 冻结模型（移除 dropout、batchnorm 等）
model.eval()
for param in model.parameters():
    param.requires_grad = False

# Trace with optimization
traced_model = torch.jit.trace(model, example_inputs)

# 优化计算图
optimized_model = torch.jit.optimize_for_inference(traced_model)

# 保存
optimized_model.save("model_optimized.pt")
```

**性能提升**（BERT-base，CPU）：

| 优化级别 | 推理速度 (samples/s) | 加速比 |
|---------|---------------------|--------|
| PyTorch Eager | 42 | 1.0x |
| TorchScript (trace) | 58 | 1.38x |
| TorchScript (optimized) | 67 | **1.60x** |

---

## 20.4 其他导出格式

### 20.4.1 CoreML（iOS 部署）

**转换到 CoreML**：

```python
import coremltools as ct
from transformers import TFAutoModel

# 1. 加载 TensorFlow 模型
tf_model = TFAutoModel.from_pretrained("bert-base-uncased", from_pt=True)

# 2. 转换为 CoreML
mlmodel = ct.convert(
    tf_model,
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, 128)),
        ct.TensorType(name="attention_mask", shape=(1, 128))
    ],
    minimum_deployment_target=ct.target.iOS15
)

# 3. 保存
mlmodel.save("bert.mlpackage")
```

**iOS 推理**（Swift）：

```swift
import CoreML

let model = try! bert(configuration: MLModelConfiguration())
let input = bertInput(input_ids: ids, attention_mask: mask)
let output = try! model.prediction(input: input)
```

---

### 20.4.2 TensorFlow Lite（移动端）

**转换流程**：

```python
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification

# 1. 加载 TensorFlow 模型
model = TFAutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english",
    from_pt=True
)

# 2. 转换为 TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 量化
tflite_model = converter.convert()

# 3. 保存
with open("distilbert.tflite", "wb") as f:
    f.write(tflite_model)
```

**Android 推理**（Kotlin）：

```kotlin
import org.tensorflow.lite.Interpreter

val interpreter = Interpreter(loadModelFile("distilbert.tflite"))
val output = Array(1) { FloatArray(2) }
interpreter.run(inputArray, output)
```

---

### 20.4.3 TensorRT（NVIDIA GPU）

**使用 TensorRT-LLM**（最新方法）：

```bash
# 安装
pip install tensorrt_llm

# 转换 HuggingFace 模型
python convert_checkpoint.py \
  --model_dir ./Llama-2-7b-hf \
  --output_dir ./trt_ckpt \
  --dtype float16

# 构建 TensorRT 引擎
trtllm-build \
  --checkpoint_dir ./trt_ckpt \
  --output_dir ./trt_engine \
  --max_batch_size 8 \
  --max_input_len 512 \
  --max_output_len 200
```

**推理**：

```python
from tensorrt_llm import LLM

llm = LLM(model_dir="./trt_engine")
outputs = llm.generate(["Hello, my name is"], max_new_tokens=50)
```

**性能**（LLaMA-7B，A100）：

| 框架 | 推理速度 (TPS) | 加速比 |
|-----|----------------|--------|
| Transformers (FP16) | 42 | 1.0x |
| TensorRT-LLM (FP16) | 95 | 2.26x |
| TensorRT-LLM (INT8) | 142 | **3.38x** |

---

### 20.4.4 ExecuTorch（边缘设备）

**Meta 新推出的边缘推理框架**：

```python
import torch
from executorch.exir import to_edge

# 1. 导出为 Edge IR
edge_program = to_edge(torch.export.export(model, example_inputs))

# 2. 转换为 ExecuTorch
executorch_program = edge_program.to_executorch()

# 3. 保存
with open("model.pte", "wb") as f:
    executorch_program.write_to_file(f)
```

**目标设备**：
- ARM Cortex-M（嵌入式）
- RISC-V
- 低功耗 IoT 设备

---

## 20.5 模型优化

### 20.5.1 ONNX Simplifier

**简化 ONNX 计算图**：

```bash
pip install onnx-simplifier

# 简化模型
python -m onnxsim model.onnx model_simplified.onnx
```

**效果示例**（BERT-base）：

| 指标 | 原始 ONNX | 简化后 | 改善 |
|-----|----------|--------|------|
| 节点数 | 1247 | 892 | -28% |
| 参数数 | 438 MB | 438 MB | 0% |
| 推理速度 | 34.7 samples/s | 38.2 samples/s | +10% |

**简化操作**：
- 常量折叠（Constant Folding）
- 算子融合（Operator Fusion）
- 移除无用节点

---

### 20.5.2 图优化（Operator Fusion）

<div data-component="OperatorFusionVisualizer"></div>

**融合示例**：

```
# 融合前
LayerNorm → Add → GELU

# 融合后
FusedLayerNormGELU  # 单个算子
```

**ONNX Runtime 自动优化**：

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import OptimizationConfig

model = ORTModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    export=True
)

# 配置优化
optimization_config = OptimizationConfig(
    optimization_level=99,  # 最高优化级别
    optimize_for_gpu=True,  # GPU 优化
    fp16=True,  # FP16 混合精度
    enable_gelu_approximation=True,  # GELU 近似
    enable_transformers_specific_optimizations=True  # Transformer 优化
)

# 应用优化
model.optimize(optimization_config, save_dir="./bert_optimized")
```

**常见融合模式**：

| 融合前 | 融合后 | 加速 |
|-------|--------|------|
| MatMul + Add | Gemm | 1.2x |
| LayerNorm + Add | FusedLayerNormAdd | 1.5x |
| GELU 分解 | FastGELU | 2.1x |
| Multi-Head Attention | FusedAttention | 3.2x |

---

### 20.5.3 常量折叠

**原理**：在编译时计算常量表达式

```python
# 优化前
x = a * 2 + 3  # 每次推理都计算 2 + 3

# 优化后（常量折叠）
x = a * 5  # 编译时已计算 2 + 3 = 5
```

**ONNX 中的应用**：

```python
import onnx
from onnx import optimizer

# 加载模型
model = onnx.load("model.onnx")

# 应用常量折叠
passes = ['eliminate_unused_initializer', 'fuse_consecutive_transposes']
optimized_model = optimizer.optimize(model, passes)

# 保存
onnx.save(optimized_model, "model_folded.onnx")
```

---

## 20.6 格式选择决策树

<div data-component="ModelExportDecisionTree"></div>

**推荐策略**：

### 生产环境

| 场景 | 推荐格式 | 理由 |
|-----|---------|------|
| 云端 GPU 推理 | ONNX + ONNX Runtime | 跨平台、优化好 |
| NVIDIA GPU（大模型） | TensorRT-LLM | 极致性能 |
| CPU 推理 | ONNX (quantized) | CPU 优化好 |
| iOS | CoreML | 原生支持、功耗低 |
| Android | TFLite | 轻量、集成简单 |
| 边缘设备 | ExecuTorch / TFLite | 超低功耗 |

### 开发阶段

- **PyTorch Eager**：快速迭代、调试
- **Safetensors**：模型分享、版本管理

---

## 20.7 实战案例：多格式导出流水线

**目标**：将 BERT 模型导出为多种格式

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.exporters.onnx import main_export
from safetensors.torch import save_file
import torch

model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 1. Safetensors（推荐存储格式）
save_file(model.state_dict(), "bert.safetensors")
print("✅ Safetensors: bert.safetensors")

# 2. ONNX（CPU/GPU 推理）
main_export(
    model_name_or_path=model_name,
    output="./bert_onnx",
    task="text-classification"
)
print("✅ ONNX: ./bert_onnx/model.onnx")

# 3. ONNX + 量化（CPU 优化）
from optimum.onnxruntime import ORTQuantizer, ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig

ort_model = ORTModelForSequenceClassification.from_pretrained("./bert_onnx")
quantizer = ORTQuantizer.from_pretrained(ort_model)
quantization_config = AutoQuantizationConfig.avx512_vnni(is_static=False)
quantizer.quantize(
    save_dir="./bert_onnx_int8",
    quantization_config=quantization_config
)
print("✅ ONNX INT8: ./bert_onnx_int8/model_quantized.onnx")

# 4. TorchScript（PyTorch 生态）
inputs = tokenizer("Hello, world!", return_tensors="pt")
traced = torch.jit.trace(model, (inputs["input_ids"], inputs["attention_mask"]))
traced.save("bert_traced.pt")
print("✅ TorchScript: bert_traced.pt")

# 5. 性能对比
import time

def benchmark(model, inputs, name):
    start = time.time()
    for _ in range(100):
        _ = model(**inputs)
    elapsed = time.time() - start
    print(f"{name}: {elapsed/100*1000:.2f} ms/sample")

# PyTorch
benchmark(model, inputs, "PyTorch Eager")

# ONNX
ort_model = ORTModelForSequenceClassification.from_pretrained("./bert_onnx")
benchmark(ort_model, inputs, "ONNX Runtime")

# ONNX INT8
ort_quant = ORTModelForSequenceClassification.from_pretrained("./bert_onnx_int8")
benchmark(ort_quant, inputs, "ONNX INT8")
```

**输出示例**：

```
✅ Safetensors: bert.safetensors
✅ ONNX: ./bert_onnx/model.onnx
✅ ONNX INT8: ./bert_onnx_int8/model_quantized.onnx
✅ TorchScript: bert_traced.pt

PyTorch Eager: 24.3 ms/sample
ONNX Runtime: 8.6 ms/sample (2.8x faster)
ONNX INT8: 5.2 ms/sample (4.7x faster)
```

---

## 20.8 常见问题

### Q1: Safetensors vs PyTorch .bin 如何选择？

**答案**：始终优先 Safetensors

| 需求 | Safetensors | PyTorch .bin |
|-----|-------------|--------------|
| 安全性 | ✅ 无代码执行 | ❌ Pickle 风险 |
| 加载速度 | ✅ 快 3-5x | ❌ 慢 |
| 跨版本兼容 | ✅ 稳定 | ⚠️ 可能失败 |
| 部分加载 | ✅ 支持 mmap | ❌ 需全量加载 |

### Q2: ONNX 导出失败 "Unsupported operator"

**原因**：模型包含 ONNX 不支持的算子

**解决方案**：

```python
# 1. 检查 ONNX 版本
import onnx
print(onnx.__version__)  # 需要 >= 1.12

# 2. 使用 Optimum（自动处理兼容性）
from optimum.exporters.onnx import main_export
main_export(
    model_name_or_path="model",
    output="./onnx",
    opset=14  # 指定 opset 版本
)

# 3. 自定义算子（高级）
from torch.onnx import register_custom_op_symbolic
```

### Q3: TorchScript trace 丢失分支逻辑？

**问题**：模型包含 if/for 等控制流

**解决方案**：使用 `torch.jit.script`

```python
# ❌ Trace 无法处理
traced = torch.jit.trace(model, inputs)

# ✅ Script 保留控制流
scripted = torch.jit.script(model)
```

或改用 ONNX（Optimum 自动处理）

---

## 20.9 总结

### 核心要点

1. **存储格式**：
   - 开发：Safetensors（安全、快速）
   - 生产：根据部署平台选择

2. **推理格式**：
   - CPU：ONNX + INT8 量化
   - NVIDIA GPU：TensorRT-LLM
   - 移动端：TFLite / CoreML
   - 边缘：ExecuTorch

3. **优化策略**：
   - 图优化：算子融合、常量折叠
   - 量化：动态 INT8（简单）、静态 INT8（精度高）
   - 硬件特定：利用专用加速器

4. **工具链**：
   - Optimum：统一导出接口
   - ONNX Runtime：跨平台推理
   - TensorRT：NVIDIA GPU 极致性能

### 性能提升路径

```
PyTorch Eager
  ↓ +2-3x (ONNX Runtime)
ONNX (FP32)
  ↓ +1.5-2x (INT8 量化)
ONNX INT8
  ↓ +1.5x (图优化)
ONNX INT8 Optimized
  → 总加速：5-8x（CPU）
```

---

## 20.10 扩展阅读

1. **Safetensors 仓库**：https://github.com/huggingface/safetensors
2. **ONNX 规范**：https://onnx.ai/
3. **Optimum 文档**：https://huggingface.co/docs/optimum
4. **TensorRT-LLM**：https://github.com/NVIDIA/TensorRT-LLM
5. **ExecuTorch**：https://pytorch.org/executorch

# Chapter 11: 混合精度训练与数值优化

## 11.1 浮点数格式基础

### 11.1.1 浮点数的本质

在深度学习中，模型参数、梯度、激活值都以浮点数（floating-point）形式存储。理解不同浮点格式的数学本质是优化训练效率的关键。

**IEEE 754 标准格式**：

浮点数由三部分组成：
$$
\text{Value} = (-1)^{\text{sign}} \times 2^{\text{exponent} - \text{bias}} \times (1 + \text{fraction})
$$

其中：
- **sign**（符号位，1 bit）：0 表示正数，1 表示负数
- **exponent**（指数位）：决定数值的**动态范围**（可表示的最大/最小值）
- **fraction**（尾数位/mantissa）：决定数值的**精度**（有效数字位数）

<div data-component="FloatFormatComparison"></div>

### 11.1.2 三大浮点格式对比

| 格式 | 总位数 | 符号位 | 指数位 | 尾数位 | 动态范围 | 精度 | 典型值 |
|------|--------|--------|--------|--------|----------|------|--------|
| **FP32** | 32 bit | 1 | 8 | 23 | $10^{-38} \sim 10^{38}$ | ~7位有效数字 | $1.175e-38$ |
| **FP16** | 16 bit | 1 | 5 | 10 | $6.1e^{-5} \sim 6.55e^{4}$ | ~3位有效数字 | $6.1e-5$ |
| **BF16** | 16 bit | 1 | 8 | 7 | $10^{-38} \sim 10^{38}$ | ~2位有效数字 | $1.175e-38$ |

**关键差异**：

1. **FP32 vs FP16**：
   - FP16 指数位只有5位（vs FP32的8位），导致动态范围骤降
   - FP16 最小正数 $6.1 \times 10^{-5}$，而 FP32 为 $1.2 \times 10^{-38}$
   - 梯度下溢风险：当梯度 $< 6.1e^{-5}$ 时，FP16 会变成 0

2. **BF16（Brain Floating Point）的优势**：
   - Google TPU 设计，**保留 FP32 的指数位（8位）**
   - 动态范围与 FP32 相同，避免溢出/下溢
   - 尾数位减少到7位（vs FP32的23位），精度下降但通常够用
   - **最适合深度学习**：梯度范围大（$10^{-10} \sim 10^{3}$），但对精度容忍度高

```python
import torch

# 三种格式的范围测试
x_fp32 = torch.tensor([1e-40, 1e40], dtype=torch.float32)
x_fp16 = torch.tensor([1e-40, 1e40], dtype=torch.float16)  # 会溢出/下溢
x_bf16 = torch.tensor([1e-40, 1e40], dtype=torch.bfloat16)

print(f"FP32: {x_fp32}")  # [1e-40, 1e+40]
print(f"FP16: {x_fp16}")  # [0., inf]  <- 超出范围！
print(f"BF16: {x_bf16}")  # [1e-40, 1e+40]  <- 与 FP32 一致
```

### 11.1.3 精度与范围的权衡

<div data-component="FloatPrecisionRangeTradeoff"></div>

**核心原则**：
- **动态范围**决定能否训练（避免梯度下溢/上溢）
- **精度**决定训练质量（累积误差控制）

在实际训练中：
- **前向传播**：激活值范围大 → 需要大动态范围（FP16风险高）
- **梯度计算**：梯度常 $< 10^{-3}$ → 易下溢（FP16致命，BF16安全）
- **权重更新**：权重 $\in [-1, 1]$ → 对精度要求高（FP32最佳）

因此混合精度的策略是：
- 大部分计算用 **FP16/BF16**（速度快）
- 关键步骤用 **FP32**（如权重累积）

---

## 11.2 自动混合精度（AMP）

### 11.2.1 AMP 的设计动机

**问题**：全 FP32 训练显存占用大、速度慢（现代 GPU Tensor Core 对 FP16/BF16 有 2-8倍加速）

**挑战**：
1. 全 FP16 训练导致梯度下溢（gradient underflow）
2. 手动管理不同精度的张量（forward FP16, backward FP32）非常繁琐

**解决方案**：PyTorch 的 `torch.cuda.amp`（Automatic Mixed Precision）
- **自动选择精度**：GEMM/Convolution 用 FP16，Softmax/Loss 用 FP32
- **动态 Loss Scaling**：梯度乘大数防下溢，更新时再除回去
- **几乎零代码改动**：只需封装 `autocast()` 和 `GradScaler()`

<div data-component="AMPWorkflow"></div>

### 11.2.2 GradScaler：梯度缩放核心机制

**原理**：

1. **Loss Scaling**（损失缩放）：
   $$
   \text{scaled\_loss} = \text{loss} \times \text{scale\_factor}
   $$
   默认 `scale = 65536 = 2^16`

2. **梯度反向传播**：
   $$
   \text{scaled\_grad} = \frac{\partial (\text{loss} \times \text{scale})}{\partial \theta} = \text{grad} \times \text{scale}
   $$
   梯度被放大 $2^{16}$ 倍，避免 FP16 下溢（$1e^{-7} \rightarrow 0.0066$）

3. **梯度更新前 Unscale**：
   $$
   \text{grad} = \frac{\text{scaled\_grad}}{\text{scale}}
   $$

4. **动态调整 Scale**：
   - 如果出现 **inf/NaN**（梯度爆炸）→ `scale /= 2`
   - 如果连续 N 步正常 → `scale *= 2`

<div data-component="GradScalerVisualizer"></div>

### 11.2.3 完整 AMP 训练代码

```python
import torch
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# 1. 加载模型（FP32，但会在 forward 时自动转 FP16）
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 2. 创建 GradScaler（管理 loss scaling）
scaler = GradScaler()

# 3. 手动训练循环（Trainer 会自动处理）
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for batch in dataloader:
    optimizer.zero_grad()
    
    # 混合精度前向传播
    with autocast():  # <- 自动选择 FP16/FP32
        outputs = model(**batch)
        loss = outputs.loss
    
    # 反向传播（梯度缩放）
    scaler.scale(loss).backward()  # loss * scale → grad * scale
    
    # 梯度更新（自动 unscale + clip + step + 动态调整 scale）
    scaler.step(optimizer)  # grad / scale → optimizer.step()
    scaler.update()  # 调整 scale（inf/NaN检测）

print(f"Final scale: {scaler.get_scale()}")  # 例如 65536
```

**关键细节**：

- `autocast()` 内的操作会**自动选择精度**：
  - **FP16**：`matmul`, `linear`, `conv`, `bmm`（计算密集型）
  - **FP32**：`log`, `exp`, `softmax`, `cross_entropy`（数值敏感型）
  
- `scaler.step(optimizer)` 内部逻辑：
  ```python
  def step(self, optimizer):
      grads = [p.grad for p in model.parameters()]
      
      # 1. Unscale 梯度
      for g in grads:
          g.div_(self.scale)
      
      # 2. 检查 inf/NaN
      if torch.any(torch.isnan(grads)) or torch.any(torch.isinf(grads)):
          self.scale /= 2  # 缩小 scale
          return  # 跳过本次更新
      
      # 3. 正常更新
      optimizer.step()
      
      # 4. 连续成功 → 增大 scale
      self.consecutive_steps += 1
      if self.consecutive_steps >= 2000:
          self.scale *= 2
          self.consecutive_steps = 0
  ```

### 11.2.4 使用 Trainer 的 AMP 训练

```python
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# 数据准备
dataset = load_dataset("glue", "mrpc")

def tokenize(examples):
    return tokenizer(
        examples["sentence1"], examples["sentence2"],
        truncation=True, padding="max_length", max_length=128
    )

train_dataset = dataset["train"].map(tokenize, batched=True)

# 训练参数（开启混合精度）
args = TrainingArguments(
    output_dir="./amp_model",
    fp16=True,  # <- 使用 FP16 混合精度（需 CUDA 能力 >= 7.0）
    # bf16=True,  # <- 或使用 BF16（Ampere/Hopper GPU，如 A100/H100）
    per_device_train_batch_size=32,  # 混合精度可增大 batch size
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
)

trainer.train()

# 训练结束后查看性能提升
# FP32: 12 it/s, 16GB VRAM
# FP16: 25 it/s,  9GB VRAM  <- 速度翻倍，显存减半！
```

**TrainingArguments 关键参数**：

| 参数 | 说明 | 适用场景 |
|------|------|----------|
| `fp16=True` | 启用 FP16 混合精度 | Volta/Turing GPU（V100, 2080Ti） |
| `bf16=True` | 启用 BF16 混合精度 | Ampere+ GPU（A100, 4090） |
| `fp16_opt_level="O1"` | Apex 混合精度级别（O0-O3） | 需要安装 Apex |
| `fp16_full_eval=True` | 评估时也用 FP16 | 加速验证 |

---

## 11.3 Transformers 混合精度最佳实践

### 11.3.1 选择 FP16 还是 BF16？

<div data-component="FP16vsBF16Comparison"></div>

**决策树**：

```
是否为 Ampere/Hopper GPU（A100/H100/4090）？
├─ 是 → 优先使用 bf16=True（动态范围大，更稳定）
└─ 否 → 使用 fp16=True
    ├─ 训练不稳定（loss=NaN）→ 降低学习率或增大 fp16_scale
    └─ 模型有 LayerNorm/BatchNorm → 确保其在 FP32 执行
```

**实际对比（GPT-2 在 GLUE）**：

| 配置 | 速度 | 显存 | MRPC准确率 |
|------|------|------|-----------|
| FP32 | 1.0x | 16GB | 88.2% |
| FP16 | 2.1x | 9GB  | 88.0% (稍有波动) |
| BF16 | 2.3x | 9GB  | 88.2% (与FP32一致) |

**结论**：BF16 是现代 GPU 的最佳选择（性能与稳定性兼得）

### 11.3.2 混合精度 + 梯度累积

当 batch size 受限时，结合梯度累积：

```python
args = TrainingArguments(
    bf16=True,
    per_device_train_batch_size=8,  # 单卡实际 batch
    gradient_accumulation_steps=16,  # 累积 16 步
    # 等效 batch size = 8 * 16 = 128
)

# Trainer 自动处理：每 16 步才调用 optimizer.step()
trainer.train()
```

**显存优化效果**：
- FP32 + bs=16：OOM（需 20GB）
- BF16 + bs=8 + grad_accum=16：10GB（等效 bs=128！）

### 11.3.3 混合精度 + Gradient Checkpointing

终极显存优化组合：

```python
model.gradient_checkpointing_enable()  # 重计算激活值

args = TrainingArguments(
    bf16=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=32,
    gradient_checkpointing=True,  # 或在 model 上开启
)

trainer.train()

# LLaMA-7B 微调显存对比：
# FP32: 32GB（OOM on 24GB GPU）
# BF16: 18GB
# BF16 + GradCheckpoint: 12GB
# BF16 + GradCheckpoint + bs=1 + grad_accum=64: 6.5GB  <- RTX 4090 可微调！
```

### 11.3.4 处理数值不稳定

**常见问题**：

1. **Loss 出现 NaN**：
   ```python
   # 解决方案 1：降低学习率
   args = TrainingArguments(
       fp16=True,
       learning_rate=1e-5,  # 原为 5e-5
   )
   
   # 解决方案 2：增大 loss scale
   args = TrainingArguments(
       fp16=True,
       fp16_backend="amp",  # 使用 PyTorch 原生 AMP
       # Trainer 会自动创建 GradScaler(init_scale=65536)
   )
   
   # 解决方案 3：切换到 BF16
   args = TrainingArguments(
       bf16=True,  # BF16 动态范围大，不需要 loss scaling
   )
   ```

2. **梯度爆炸（inf）**：
   ```python
   args = TrainingArguments(
       fp16=True,
       max_grad_norm=1.0,  # 梯度裁剪（在 unscale 后执行）
   )
   ```

3. **特定层需要 FP32**：
   ```python
   # 手动控制（不推荐，Trainer 会自动处理）
   model.lm_head = model.lm_head.float()  # 语言模型头用 FP32
   
   # Trainer 默认行为：
   # - LayerNorm/BatchNorm 自动用 FP32
   # - 损失计算自动用 FP32
   ```

---

## 11.4 混合精度的数学原理

### 11.4.1 为什么 FP16 能加速训练？

**硬件层面**：

现代 GPU 有专门的 **Tensor Core**（张量核心）：
- **V100**（Volta）：FP16 Tensor Core = 125 TFLOPS（FP32 为 15 TFLOPS，提速 **8倍**）
- **A100**（Ampere）：BF16/FP16 Tensor Core = 312 TFLOPS（FP32 为 19.5 TFLOPS，提速 **16倍**）
- **H100**（Hopper）：FP8 Tensor Core = 1000 TFLOPS（BF16 为 500 TFLOPS）

<div data-component="TensorCorePerformance"></div>

**矩阵乘法示例**：

对于 $C = A \times B$（$A, B \in \mathbb{R}^{n \times n}$）：
- **FP32**：$2n^3$ 次浮点运算（FLOPS），V100 需 $\frac{2n^3}{15 \times 10^{12}}$ 秒
- **FP16**：同样 $2n^3$ FLOPS，但 Tensor Core 需 $\frac{2n^3}{125 \times 10^{12}}$ 秒

计算 LLaMA-7B 一次前向传播（主要是 `linear` 层矩阵乘）：
- FP32：约 1.8 秒
- FP16 (Tensor Core)：约 0.25 秒（**加速 7倍**）

**带宽优势**：
- FP16 数据量是 FP32 的一半 → 显存带宽需求减半
- 例如：读取 100MB 权重，FP16 比 FP32 快一倍

### 11.4.2 混合精度的精度损失分析

**理论分析**：

假设真实梯度为 $g$，FP16 表示为 $\tilde{g}$，量化误差为 $\epsilon = g - \tilde{g}$。

**误差累积**（$T$ 步训练后）：
$$
\theta_T = \theta_0 - \eta \sum_{t=1}^{T} (\tilde{g}_t)
= \theta_0 - \eta \sum_{t=1}^{T} (g_t - \epsilon_t)
$$

如果 $\epsilon_t$ 是**独立同分布**的，期望为 0，则：
$$
\mathbb{E}[\theta_T] = \theta_0 - \eta \sum_{t=1}^{T} g_t  \quad \text{（无偏）}
$$

但方差会增大：
$$
\text{Var}[\theta_T] = \eta^2 \sum_{t=1}^{T} \text{Var}[\epsilon_t] \approx T \cdot \sigma_\epsilon^2
$$

**实践中的观察**：

大规模实验（BERT/GPT）表明：
- FP16/BF16 训练后模型与 FP32 几乎相同（**<0.1% 性能差异**）
- 原因：
  1. 深度学习对噪声有一定**鲁棒性**（dropout/正则化效果类似）
  2. **GradScaler 动态调整**避免了严重下溢
  3. 关键操作（Loss/Softmax）仍用 FP32

<div data-component="PrecisionLossComparison"></div>

---

## 11.5 高级混合精度技术

### 11.5.1 FP8 训练（实验性）

**Hopper GPU（H100）引入 FP8**：
- **E4M3**（4位指数，3位尾数）：范围 $[-448, 448]$，精度 ~1位有效数字
- **E5M2**（5位指数，2位尾数）：范围 $[-57344, 57344]$，精度 ~0.5位有效数字

**使用场景**：
- **E4M3** 用于前向传播（需要精度）
- **E5M2** 用于梯度（需要范围）

```python
# TransformerEngine（NVIDIA 库）
import transformer_engine.pytorch as te

# 自动 FP8 训练（需 H100）
with te.fp8_autocast():
    output = model(input)
    loss = criterion(output, target)
    loss.backward()

# 性能提升：H100 上比 BF16 再快 2 倍
```

**挑战**：
- FP8 精度极低，需要更复杂的缩放策略（per-tensor scaling）
- 目前仅在超大模型（>100B）训练中有明显优势

### 11.5.2 Stochastic Rounding（随机舍入）

**问题**：FP16/FP32 转换时，标准舍入（round-to-nearest）会导致**偏差累积**

**解决方案**：以概率进行舍入

对于真实值 $x = 3.7$，FP16 只能表示 $3.0$ 或 $4.0$：
- **标准舍入**：总是取 $4.0$（最近邻）
- **随机舍入**：
  $$
  \tilde{x} = \begin{cases}
  3.0, & \text{概率} = 1 - 0.7 = 0.3 \\
  4.0, & \text{概率} = 0.7
  \end{cases}
  $$

**优点**：期望无偏 $\mathbb{E}[\tilde{x}] = 3.7$

```python
# PyTorch 原生不支持，需要自定义（或使用 NVIDIA Apex）
def stochastic_round(x_fp32):
    x_fp16 = x_fp32.half()
    error = x_fp32 - x_fp16.float()
    mask = torch.rand_like(error) < error.abs()
    return torch.where(mask, x_fp16 + error.sign(), x_fp16)
```

**实验结果**（BERT-Large）：
- 标准 FP16：87.1% MNLI 准确率
- 随机舍入 FP16：87.4%（接近 FP32 的 87.5%）

### 11.5.3 混合精度分布式训练

**挑战**：AllReduce 梯度时，如何处理不同精度？

**策略 1：FP16 通信 + FP32 累积**（NVIDIA Apex 默认）

```python
from apex import amp

model, optimizer = amp.initialize(
    model, optimizer,
    opt_level="O2",  # FP16 训练 + FP32 Master Weights
)

# AllReduce 时：
# 1. 梯度在 FP16 下通信（节省带宽）
# 2. 接收后转 FP32 累积（避免精度损失）
# 3. 更新 FP32 主权重
# 4. 下次前向时转回 FP16
```

**策略 2：BF16 端到端**（PyTorch DDP 默认）

```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(model)

# BF16 不需要主权重：
# 1. 梯度 BF16 AllReduce
# 2. 直接更新 BF16 权重
# 优点：显存节省（无需双份权重）
# 缺点：更新精度略低（但实验证明影响<0.1%）
```

<div data-component="DistributedMixedPrecision"></div>

**性能对比（8xA100 训练 GPT-2）**：

| 配置 | 通信量 | 显存/卡 | 速度 |
|------|--------|---------|------|
| FP32 | 4GB/step | 20GB | 100 samples/s |
| FP16+FP32 Master | 2GB/step | 24GB | 180 samples/s |
| BF16 纯模式 | 2GB/step | 12GB | 190 samples/s |

**结论**：BF16 纯模式最优（通信快+显存省）

---

## 11.6 混合精度性能基准

### 11.6.1 不同模型的加速效果

<div data-component="MixedPrecisionBenchmark"></div>

**实验设置**：
- GPU：NVIDIA A100 (80GB)
- 框架：Transformers 4.36 + PyTorch 2.1
- 任务：语言模型预训练（批大小调至不 OOM 的最大值）

| 模型 | FP32速度 | BF16速度 | 加速比 | FP32显存 | BF16显存 | 显存节省 |
|------|----------|----------|--------|----------|----------|----------|
| BERT-Base | 120 s/s | 280 s/s | 2.33x | 8GB | 4.5GB | 44% |
| BERT-Large | 45 s/s | 110 s/s | 2.44x | 18GB | 9GB | 50% |
| GPT-2 (1.5B) | 18 s/s | 42 s/s | 2.33x | 32GB | 16GB | 50% |
| LLaMA-7B | 8 s/s | 22 s/s | 2.75x | OOM | 45GB | - |
| LLaMA-13B | - | 12 s/s | - | OOM | 78GB | - |

**观察**：
- 加速比稳定在 **2.3-2.8倍**（接近理论上限）
- 显存节省约 **50%**（权重+激活值都减半）
- 超大模型（>7B）在 FP32 下无法训练（OOM）

### 11.6.2 精度影响实验

**GLUE 基准测试**（BERT-Large 微调）：

| 任务 | FP32 | FP16 | BF16 | 差异 |
|------|------|------|------|------|
| MNLI | 87.5% | 87.2% (-0.3%) | 87.4% (-0.1%) | 可忽略 |
| QQP | 91.8% | 91.6% (-0.2%) | 91.8% (0%) | BF16=FP32 |
| QNLI | 93.1% | 92.9% (-0.2%) | 93.0% (-0.1%) | 可忽略 |
| SST-2 | 94.2% | 93.8% (-0.4%) | 94.1% (-0.1%) | FP16稍差 |

**结论**：
- **BF16** 几乎无损（差异 $\leq 0.1\%$）
- **FP16** 有轻微损失（0.2-0.4%），但仍在可接受范围
- 对于超大模型（>10B），差异进一步缩小（<0.05%）

### 11.6.3 训练稳定性对比

**GPT-2 预训练 Loss 曲线**：

<div data-component="TrainingStabilityComparison"></div>

```python
# 三种精度的 Loss 对比（10k steps）
# FP32:  Loss = 3.21 ± 0.05（平滑）
# BF16:  Loss = 3.23 ± 0.06（轻微波动）
# FP16:  Loss = 3.25 ± 0.12（波动较大，偶现 NaN）

# 训练成功率（100次独立运行）：
# FP32: 100%
# BF16: 100%
# FP16:  95%（5次出现 NaN 需重启）
```

**稳定性排序**：FP32 > BF16 >> FP16

**最佳实践**：
- 新模型实验：先用 FP32 验证可行性
- 生产训练：用 BF16（速度与稳定性平衡）
- 资源极度受限：FP16 + 降低学习率 + 监控 NaN

---

## 11.7 混合精度故障排查

### 11.7.1 常见问题诊断

**问题 1：Loss 突然变成 NaN**

```python
# 诊断步骤
import torch

# 1. 检查是否为梯度爆炸
trainer.add_callback(lambda trainer, state, control: 
    print(f"Step {state.global_step}, Loss: {state.log_history[-1]['loss']}")
)

# 2. 添加梯度监控
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm > 100:  # 梯度异常大
            print(f"Large grad in {name}: {grad_norm}")

# 3. 解决方案
args = TrainingArguments(
    bf16=True,  # 或从 fp16 切换到 bf16
    max_grad_norm=1.0,  # 梯度裁剪
    learning_rate=1e-5,  # 降低学习率
)
```

**问题 2：训练速度没有提升**

可能原因：
1. **未使用 Tensor Core**：模型维度不是 8 的倍数
   ```python
   # 检查模型配置
   print(model.config.hidden_size)  # 应为 768, 1024, 4096 等（8的倍数）
   
   # 如果是 765：
   config = AutoConfig.from_pretrained("model_name")
   config.hidden_size = 768  # 调整为 8 的倍数
   model = AutoModel.from_config(config)
   ```

2. **Batch Size 太小**：Tensor Core 需要大矩阵才高效
   ```python
   # 增大 batch size（或使用梯度累积）
   args = TrainingArguments(
       per_device_train_batch_size=32,  # 至少 16+
       gradient_accumulation_steps=4,
   )
   ```

**问题 3：显存没有减少**

```python
# 可能原因：模型已经在 FP32 加载
model = AutoModel.from_pretrained("gpt2")  # 默认 FP32
print(model.dtype)  # torch.float32

# 解决方案 1：训练时自动转换（Trainer 默认行为）
args = TrainingArguments(bf16=True)  # Trainer 会自动处理

# 解决方案 2：加载时直接用 BF16
model = AutoModel.from_pretrained("gpt2", torch_dtype=torch.bfloat16)
print(model.dtype)  # torch.bfloat16

# 解决方案 3：清理缓存
import gc
gc.collect()
torch.cuda.empty_cache()
```

### 11.7.2 高级调试技巧

**使用 PyTorch Profiler 定位瓶颈**：

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
) as prof:
    for batch in dataloader:
        with autocast():
            outputs = model(**batch)
            loss = outputs.loss
        loss.backward()
        optimizer.step()
        if step == 10:  # 只分析 10 步
            break

# 查看时间分布
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# 示例输出：
# Name                          | CUDA Time | CPU Time
# ----------------------------- | --------- | --------
# aten::matmul (FP16)           | 450ms     | 10ms     <- Tensor Core 加速
# aten::softmax (FP32)          | 120ms     | 5ms      <- 自动用 FP32
# GradScaler.scale              | 15ms      | 8ms
```

**检测 Tensor Core 使用率**：

```bash
# 训练时监控（另一个终端）
nvidia-smi dmon -s u

# 期望输出（Tensor Core 活跃）：
# GPU  SM  MEM  ENC  DEC  TENSOR
#   0  95   80   0    0    92     <- TENSOR 利用率 92%（很好！）

# 如果 TENSOR=0：未使用 Tensor Core（检查精度/矩阵大小）
```

---

## 11.8 混合精度与其他优化的组合

### 11.8.1 混合精度 + DeepSpeed ZeRO

```python
from transformers import TrainingArguments, Trainer

args = TrainingArguments(
    output_dir="./output",
    bf16=True,  # 混合精度
    deepspeed="ds_config_zero3.json",  # ZeRO-3
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
)

# ds_config_zero3.json 内容：
{
  "bf16": {"enabled": true},  # 与 TrainingArguments 一致
  "zero_optimization": {
    "stage": 3,
    "offload_param": {"device": "cpu"},  # CPU offload
    "offload_optimizer": {"device": "cpu"}
  }
}

trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()

# 效果（LLaMA-65B 微调）：
# 单 A100 (80GB)：
# - BF16 alone: OOM
# - DeepSpeed ZeRO-3 alone (FP32): OOM
# - BF16 + ZeRO-3 + CPU offload: 成功！峰值 78GB
```

### 11.8.2 混合精度 + Flash Attention

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    torch_dtype=torch.bfloat16,  # BF16
    attn_implementation="flash_attention_2",  # Flash Attention v2
)

args = TrainingArguments(
    bf16=True,
    per_device_train_batch_size=8,
    max_seq_length=4096,  # Flash Attention 支持超长序列
)

# 性能对比（LLaMA-7B, seq_len=4096）：
# BF16 标准 Attention: 12 samples/s, 45GB
# BF16 + Flash Attention: 28 samples/s, 32GB  <- 2.3x faster, 30% 显存节省
```

### 11.8.3 混合精度 + Quantization（QLoRA）

```python
from transformers import BitsAndBytesConfig

# 4-bit 量化 + BF16 计算
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算用 BF16
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b",
    quantization_config=quant_config,
)

# 显存对比：
# FP32 (70B): 280GB（需 4xA100）
# BF16 (70B): 140GB（需 2xA100）
# 4-bit + BF16 compute (70B): 35GB（单 A100 可微调！）
```

---

## 11.9 混合精度未来展望

### 11.9.1 FP8 的大规模应用

NVIDIA H100 的 FP8 Tensor Core 提供 **1000 TFLOPS**（BF16 的 2 倍），但需要框架支持。

**当前进展**：
- **TransformerEngine**（NVIDIA 开源）：自动 FP8 转换
- **Megatron-LM**（NVIDIA）：支持 FP8 预训练（GPT-3 规模）
- **Hugging Face Transformers**：实验性支持（v4.35+）

```python
# 使用 TransformerEngine（需 H100）
import transformer_engine.pytorch as te

# 自动 FP8 训练
with te.fp8_autocast(enabled=True):
    output = model(input)
    loss.backward()

# 性能：GPT-3 (175B) 在 256xH100 上
# BF16: 2.1 天/epoch
# FP8:  1.1 天/epoch  <- 接近 2 倍加速
```

### 11.9.2 混合精度 + 神经架构搜索

自动为每层选择最优精度：

```python
# 示例（研究阶段）
layer_precision = {
    "embeddings": "FP32",  # 嵌入层精度敏感
    "attention": "FP16",  # 注意力计算密集
    "ffn": "FP8",  # FFN 对精度容忍度高
    "lm_head": "FP32",  # 输出层需要高精度
}

# 效果（理论）：
# 全 FP32: 100% 性能, 1x 速度
# 全 BF16: 99.8% 性能, 2.5x 速度
# 混合策略: 99.9% 性能, 3.2x 速度  <- 最优！
```

### 11.9.3 硬件感知的自适应精度

根据 GPU 型号自动选择精度：

```python
def get_optimal_precision():
    if torch.cuda.get_device_capability()[0] >= 9:  # Hopper (H100)
        return "fp8"
    elif torch.cuda.get_device_capability()[0] >= 8:  # Ampere (A100)
        return "bf16"
    else:  # Volta/Turing
        return "fp16"

# Transformers 未来可能的 API：
args = TrainingArguments(
    auto_precision=True,  # 自动选择最优精度
)
```

---

## 11.10 总结与建议

### 11.10.1 快速决策指南

```
开始微调/预训练 Transformer 模型
├─ 使用 Ampere/Hopper GPU（A100/H100/4090）？
│  └─ 是 → bf16=True（首选）
│     └─ 否 → 继续判断
├─ 使用 Volta/Turing GPU（V100/2080Ti）？
│  └─ 是 → fp16=True + 监控 NaN
│     └─ 否（CPU/AMD GPU）→ 不使用混合精度
├─ 模型 > 10B 且显存不足？
│  └─ 是 → bf16=True + gradient_checkpointing=True + DeepSpeed ZeRO-3
└─ 追求极致速度？
   └─ 是 → bf16=True + Flash Attention + torch.compile
```

### 11.10.2 性能优化检查清单

- [ ] **启用混合精度**（`bf16=True` 或 `fp16=True`）
- [ ] **增大 Batch Size**（混合精度显存减半 → batch 可加倍）
- [ ] **启用 Gradient Checkpointing**（大模型必备）
- [ ] **确保矩阵维度是 8 的倍数**（Tensor Core 友好）
- [ ] **使用 Flash Attention**（长序列训练）
- [ ] **监控 Loss 稳定性**（FP16 需注意 NaN）
- [ ] **Profiling 验证 Tensor Core 使用**（`nvidia-smi dmon -s u`）

### 11.10.3 关键要点回顾

1. **浮点格式选择**：
   - **BF16**：现代 GPU 首选（动态范围大、训练稳定）
   - **FP16**：旧 GPU 备选（需 loss scaling 防下溢）
   - **FP32**：调试/基准测试

2. **性能提升**：
   - 速度：**2-3倍**（Tensor Core 加速）
   - 显存：节省 **~50%**（可训练更大模型/batch）
   - 精度损失：**<0.1%**（几乎可忽略）

3. **最佳组合**：
   - 小模型（<1B）：BF16 + batch size 增大
   - 中等模型（1-10B）：BF16 + Gradient Checkpointing
   - 大模型（10-100B）：BF16 + Gradient Checkpointing + DeepSpeed ZeRO
   - 超大模型（>100B）：FP8 + ZeRO-3 + CPU offload（需 H100）

4. **故障排查**：
   - Loss=NaN → 切换 BF16 或降低学习率
   - 速度无提升 → 检查矩阵大小（需 8 的倍数）+ 增大 batch
   - 显存无变化 → 确认模型加载精度（`model.dtype`）

### 11.10.4 代码模板

**标准混合精度训练**：

```python
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

# 1. 加载模型（可选：直接 BF16 加载节省显存）
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    torch_dtype=torch.bfloat16,  # 可选
)

# 2. 配置训练参数
args = TrainingArguments(
    output_dir="./output",
    bf16=True,  # 核心：启用混合精度
    per_device_train_batch_size=16,  # 混合精度可增大
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,  # 大模型必备
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
)

# 3. 训练
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
)

trainer.train()
```

**手动 AMP 控制**（高级用法）：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in dataloader:
        optimizer.zero_grad()
        
        # 混合精度前向
        with autocast():
            outputs = model(**batch)
            loss = outputs.loss
        
        # 梯度缩放反向传播
        scaler.scale(loss).backward()
        
        # 梯度裁剪（在 unscale 后）
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 更新（自动处理 inf/NaN）
        scaler.step(optimizer)
        scaler.update()
```

---

通过本章的学习，你应该能够：
- ✅ 理解 FP32/FP16/BF16 的数学本质与权衡
- ✅ 使用 Transformers Trainer 启用混合精度训练
- ✅ 掌握 GradScaler 的梯度缩放机制
- ✅ 诊断并解决混合精度训练中的常见问题
- ✅ 组合混合精度与其他优化技术（DeepSpeed/Flash Attention）
- ✅ 根据硬件与模型规模选择最优精度策略

**下一章预告**：Chapter 12 将深入分布式训练，讲解 DDP、FSDP、DeepSpeed ZeRO 的底层机制，以及如何在多卡/多节点上高效训练超大模型。

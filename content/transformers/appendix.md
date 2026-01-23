# 附录 (Appendices)

> **本附录提供**：常见错误调试、性能基准对比、资源清单、API 速查表等实用参考资料。

---

## Appendix A: 常见错误与调试

### A.1 CUDA Out of Memory (OOM)

**症状**：
```
RuntimeError: CUDA out of memory. Tried to allocate XX MiB 
(GPU 0; XX GiB total capacity; XX GiB already allocated; ...)
```

**原因**：
- 批次大小（batch size）过大
- 模型参数量过大
- 序列长度过长
- 梯度累积未清除
- 缓存未释放

**解决方案**：

1. **减小批次大小**：
```python
# 从大到小逐步尝试
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # 最小化批次
    gradient_accumulation_steps=16,  # 累积梯度模拟大批次
)
```

2. **使用梯度检查点**：
```python
model.gradient_checkpointing_enable()

# 或在 config 中
model.config.use_cache = False  # 禁用 KV cache
model.config.gradient_checkpointing = True
```

3. **混合精度训练**：
```python
training_args = TrainingArguments(
    fp16=True,  # 或 bf16=True
)
```

4. **量化加载模型**：
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
```

5. **清除缓存**：
```python
import torch
import gc

# 清除未使用的缓存
torch.cuda.empty_cache()
gc.collect()

# 在训练循环中
for batch in dataloader:
    optimizer.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    
    # 定期清理
    if step % 100 == 0:
        torch.cuda.empty_cache()
```

6. **使用 CPU offload**：
```python
from accelerate import infer_auto_device_map

device_map = infer_auto_device_map(
    model,
    max_memory={0: "10GiB", "cpu": "30GiB"}
)
model = AutoModelForCausalLM.from_pretrained(
    "large-model",
    device_map=device_map
)
```

---

### A.2 Tokenizer 不匹配

**症状**：
```
Warning: Some weights of the model checkpoint were not used: ['lm_head.weight']
```

**原因**：
- 使用了不同模型的 tokenizer
- 词汇表大小不一致
- 特殊 token 配置不同

**解决方案**：

1. **确保 tokenizer 和模型匹配**：
```python
# ✗ 错误
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # 不匹配！

# ✓ 正确
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

2. **检查词汇表大小**：
```python
print(f"Model vocab size: {model.config.vocab_size}")
print(f"Tokenizer vocab size: {len(tokenizer)}")

# 如果不一致，调整模型
if model.config.vocab_size != len(tokenizer):
    model.resize_token_embeddings(len(tokenizer))
```

3. **添加特殊 token**：
```python
# 添加新 token
special_tokens_dict = {'additional_special_tokens': ['<NEW_TOKEN>']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

# 调整模型嵌入层
model.resize_token_embeddings(len(tokenizer))
```

---

### A.3 权重加载警告

**症状**：
```
Some weights of XxxForSequenceClassification were not initialized from the model checkpoint:
['classifier.weight', 'classifier.bias']
```

**原因**：
- 微调任务头（classification head）未预训练
- 模型架构变化
- 正常现象（多数情况）

**解决方案**：

1. **正常情况（可忽略）**：
```python
# 从预训练模型加载用于分类任务
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3  # 分类头会随机初始化
)
# 警告是正常的，因为 BERT 预训练时没有分类头
```

2. **从微调检查点加载**：
```python
# 如果要加载之前微调过的模型
model = AutoModelForSequenceClassification.from_pretrained(
    "./fine-tuned-model"  # 本地路径，包含完整权重
)
```

3. **忽略特定层**：
```python
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    ignore_mismatched_sizes=True  # 忽略大小不匹配
)
```

---

### A.4 分布式训练卡死

**症状**：
- 程序启动后卡住不动
- 多进程无法通信
- `torch.distributed.init_process_group()` 超时

**原因**：
- 环境变量未设置
- 端口被占用
- 网络配置问题
- 代码有死锁

**解决方案**：

1. **检查环境变量**：
```bash
# 单机多卡
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=4
export RANK=0

# 多机
export MASTER_ADDR=主节点IP
export MASTER_PORT=29500
```

2. **使用 torchrun**（推荐）：
```bash
# 单机 4 卡
torchrun --nproc_per_node=4 train.py

# 多机（节点 0）
torchrun --nnodes=2 --nproc_per_node=4 \
         --node_rank=0 \
         --master_addr=192.168.1.1 \
         --master_port=29500 \
         train.py

# 多机（节点 1）
torchrun --nnodes=2 --nproc_per_node=4 \
         --node_rank=1 \
         --master_addr=192.168.1.1 \
         --master_port=29500 \
         train.py
```

3. **使用 Accelerate**：
```bash
accelerate config  # 配置
accelerate launch train.py  # 启动
```

4. **调试技巧**：
```python
import torch.distributed as dist

# 添加超时检测
dist.init_process_group(
    backend='nccl',
    timeout=datetime.timedelta(seconds=30)  # 30秒超时
)

# 打印调试信息
import os
print(f"RANK: {os.environ.get('RANK', 'Not set')}")
print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")
print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'Not set')}")
```

5. **常见死锁场景**：
```python
# ✗ 错误：不同进程执行不同代码路径
if rank == 0:
    dist.barrier()  # 只有 rank 0 等待，其他进程不等待 → 死锁

# ✓ 正确：所有进程执行相同操作
dist.barrier()  # 所有进程都等待

# ✗ 错误：条件不一致
if condition_that_varies_by_rank:
    dist.all_reduce(tensor)  # 只有部分进程参与 → 死锁

# ✓ 正确：所有进程都参与集合通信
dist.all_reduce(tensor)
```

---

### A.5 生成质量差

**症状**：
- 生成重复内容
- 输出不连贯
- 生成停不下来
- 生成结果单调

**原因**：
- 采样策略不当
- 温度设置不合理
- 没有设置停止条件
- 模型未充分训练

**解决方案**：

1. **调整采样参数**：
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    
    # 方法 1: Greedy (确定性)
    do_sample=False,
    
    # 方法 2: Top-K 采样
    do_sample=True,
    top_k=50,
    temperature=0.7,
    
    # 方法 3: Top-P (Nucleus)
    do_sample=True,
    top_p=0.9,
    temperature=0.8,
    
    # 方法 4: Beam Search
    num_beams=5,
    early_stopping=True,
    
    # 防止重复
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
)
```

2. **设置停止条件**：
```python
# 方法 1: EOS token
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id
)

# 方法 2: 自定义停止序列
from transformers import StoppingCriteria, StoppingCriteriaList

class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_sequences, tokenizer):
        self.stop_sequences = stop_sequences
        self.tokenizer = tokenizer
    
    def __call__(self, input_ids, scores, **kwargs):
        decoded = self.tokenizer.decode(input_ids[0])
        return any(seq in decoded for seq in self.stop_sequences)

stopping_criteria = StoppingCriteriaList([
    CustomStoppingCriteria(["\n\n", "END"], tokenizer)
])

outputs = model.generate(
    **inputs,
    stopping_criteria=stopping_criteria
)
```

3. **温度调优指南**：
```python
# temperature < 1.0: 更保守、确定
# temperature = 1.0: 标准采样
# temperature > 1.0: 更随机、创造性

# 事实性任务（QA、摘要）
temperature=0.3

# 创造性任务（故事、诗歌）
temperature=1.0

# 极度随机（头脑风暴）
temperature=1.5
```

4. **对比生成配置**：
```python
# 配置 1: 事实性生成
generation_config_factual = GenerationConfig(
    do_sample=False,  # Greedy
    max_new_tokens=100,
    repetition_penalty=1.1
)

# 配置 2: 平衡生成
generation_config_balanced = GenerationConfig(
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
    max_new_tokens=150,
    repetition_penalty=1.2
)

# 配置 3: 创造性生成
generation_config_creative = GenerationConfig(
    do_sample=True,
    top_k=50,
    temperature=1.0,
    max_new_tokens=200,
    no_repeat_ngram_size=2
)

# 使用
outputs = model.generate(**inputs, generation_config=generation_config_balanced)
```

---

## Appendix B: 性能基准测试

### B.1 常见模型推理速度对比

| 模型 | 参数量 | 序列长度 | 吞吐量 (tokens/s) | 延迟 (ms/token) | 显存 (GB) |
|------|--------|----------|-------------------|-----------------|-----------|
| BERT-base | 110M | 512 | 1200 | 0.8 | 0.4 |
| RoBERTa-large | 355M | 512 | 450 | 2.2 | 1.4 |
| GPT-2 (small) | 124M | 1024 | 800 | 1.25 | 0.5 |
| GPT-2 (medium) | 355M | 1024 | 320 | 3.1 | 1.4 |
| GPT-2 (large) | 774M | 1024 | 150 | 6.7 | 3.1 |
| GPT-2 (xl) | 1.5B | 1024 | 75 | 13.3 | 6.0 |
| LLaMA-7B | 7B | 2048 | 30 | 33 | 28 |
| LLaMA-13B | 13B | 2048 | 16 | 62 | 52 |
| LLaMA-70B | 70B | 2048 | 3 | 333 | 280 |

**测试环境**: NVIDIA A100 40GB, batch_size=1, FP16

### B.2 训练吞吐量对比

| 模型 | 批次大小 | 梯度累积 | 吞吐量 (samples/s) | GPU 利用率 |
|------|----------|----------|-------------------|------------|
| BERT-base | 32 | 1 | 120 | 85% |
| BERT-base | 8 | 4 | 115 | 82% |
| GPT-2 (medium) | 16 | 1 | 45 | 90% |
| GPT-2 (medium) | 4 | 4 | 42 | 88% |
| LLaMA-7B | 4 | 8 | 8 | 95% |
| LLaMA-7B (QLoRA) | 8 | 4 | 12 | 92% |

**测试环境**: NVIDIA A100 40GB, FP16/BF16

### B.3 显存占用对比表

| 操作 | BERT-base | GPT-2 | LLaMA-7B |
|------|-----------|-------|----------|
| 推理 (FP32) | 1.2 GB | 2.4 GB | 28 GB |
| 推理 (FP16) | 0.6 GB | 1.2 GB | 14 GB |
| 推理 (INT8) | 0.3 GB | 0.6 GB | 7 GB |
| 推理 (INT4) | 0.2 GB | 0.3 GB | 3.5 GB |
| 训练 (FP32) | 4.8 GB | 9.6 GB | 112 GB |
| 训练 (FP16 + AMP) | 2.8 GB | 5.2 GB | 56 GB |
| 训练 (LoRA) | 1.0 GB | 1.8 GB | 16 GB |
| 训练 (QLoRA 4-bit) | 0.6 GB | 1.0 GB | 9 GB |

### B.4 量化方法对比矩阵

| 量化方法 | 精度 | 速度 | 显存节省 | 准确度 | 易用性 |
|----------|------|------|----------|--------|--------|
| FP16 | 16-bit | ⭐⭐⭐⭐ | 50% | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| BF16 | 16-bit | ⭐⭐⭐⭐ | 50% | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| INT8 (动态) | 8-bit | ⭐⭐⭐⭐⭐ | 75% | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| INT8 (静态) | 8-bit | ⭐⭐⭐⭐⭐ | 75% | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| GPTQ | 4-bit | ⭐⭐⭐⭐ | 87.5% | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| AWQ | 4-bit | ⭐⭐⭐⭐⭐ | 87.5% | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| NF4 (QLoRA) | 4-bit | ⭐⭐⭐ | 87.5% | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## Appendix C: 资源清单

### C.1 官方文档与教程

**Hugging Face 官方资源**：
- 📖 [Transformers 文档](https://huggingface.co/docs/transformers)
- 📖 [Datasets 文档](https://huggingface.co/docs/datasets)
- 📖 [PEFT 文档](https://huggingface.co/docs/peft)
- 📖 [Accelerate 文档](https://huggingface.co/docs/accelerate)
- 📖 [Optimum 文档](https://huggingface.co/docs/optimum)
- 📖 [TRL 文档](https://huggingface.co/docs/trl)
- 🎓 [Hugging Face Course](https://huggingface.co/learn/nlp-course)
- 🎥 [YouTube 官方频道](https://www.youtube.com/@HuggingFace)

**PyTorch 相关**：
- 📖 [PyTorch 官方文档](https://pytorch.org/docs)
- 📖 [PyTorch Tutorials](https://pytorch.org/tutorials)
- 📖 [DeepSpeed 文档](https://www.deepspeed.ai/)
- 📖 [FSDP 指南](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

### C.2 重要论文列表

**基础架构**：
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) - Transformer 原论文
- [BERT](https://arxiv.org/abs/1810.04805) (2018) - 预训练语言模型
- [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019)
- [GPT-3](https://arxiv.org/abs/2005.14165) (2020) - 大规模语言模型
- [T5](https://arxiv.org/abs/1910.10683) (2020) - Text-to-Text Transfer

**高效训练**：
- [LoRA](https://arxiv.org/abs/2106.09685) (2021) - 低秩适配
- [QLoRA](https://arxiv.org/abs/2305.14314) (2023) - 量化 LoRA
- [FlashAttention](https://arxiv.org/abs/2205.14135) (2022) - IO优化注意力
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) (2023)

**量化与压缩**：
- [GPTQ](https://arxiv.org/abs/2210.17323) (2022) - 后训练量化
- [AWQ](https://arxiv.org/abs/2306.00978) (2023) - 激活感知量化
- [SmoothQuant](https://arxiv.org/abs/2211.10438) (2022)

**长上下文**：
- [Longformer](https://arxiv.org/abs/2004.05150) (2020) - 稀疏注意力
- [BigBird](https://arxiv.org/abs/2007.14062) (2020)
- [ALiBi](https://arxiv.org/abs/2108.12409) (2021) - 线性偏置
- [RoPE](https://arxiv.org/abs/2104.09864) (2021) - 旋转位置编码

**RLHF 与对齐**：
- [InstructGPT](https://arxiv.org/abs/2203.02155) (2022) - RLHF
- [DPO](https://arxiv.org/abs/2305.18290) (2023) - 直接偏好优化
- [Constitutional AI](https://arxiv.org/abs/2212.08073) (2022)

**多模态**：
- [CLIP](https://arxiv.org/abs/2103.00020) (2021)
- [Flamingo](https://arxiv.org/abs/2204.14198) (2022)
- [LLaVA](https://arxiv.org/abs/2304.08485) (2023)

### C.3 推荐开源项目

**训练框架**：
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) - 微调工具
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - 一站式微调
- [Unsloth](https://github.com/unslothai/unsloth) - 极速微调

**推理优化**：
- [vLLM](https://github.com/vllm-project/vllm) - 高性能推理
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference) - HF 官方
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - C++ 推理

**量化工具**：
- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

**评估框架**：
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [OpenAI Evals](https://github.com/openai/evals)

### C.4 社区资源

**论坛与社区**：
- 💬 [Hugging Face Discord](https://hf.co/join/discord)
- 💬 [Hugging Face Forums](https://discuss.huggingface.co/)
- 🐦 Twitter: [@huggingface](https://twitter.com/huggingface)
- 📧 [Newsletter](https://huggingface.co/subscribe)

**学习资源**：
- [Papers with Code](https://paperswithcode.com/)
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [Hugging Face Spaces](https://huggingface.co/spaces) - 在线演示

---

## Appendix D: API 速查表

### D.1 AutoModelForXXX 类列表

```python
from transformers import (
    # 因果语言模型（文本生成）
    AutoModelForCausalLM,
    
    # 序列到序列（翻译、摘要）
    AutoModelForSeq2SeqLM,
    
    # 掩码语言模型（填空）
    AutoModelForMaskedLM,
    
    # 序列分类（情感分析、文本分类）
    AutoModelForSequenceClassification,
    
    # Token 分类（NER、POS）
    AutoModelForTokenClassification,
    
    # 问答
    AutoModelForQuestionAnswering,
    
    # 多选题
    AutoModelForMultipleChoice,
    
    # 图像分类
    AutoModelForImageClassification,
    
    # 语音识别
    AutoModelForSpeechSeq2Seq,
    
    # 视觉问答
    AutoModelForVisualQuestionAnswering,
)
```

### D.2 TrainingArguments 参数速查

```python
from transformers import TrainingArguments

args = TrainingArguments(
    # 基础参数
    output_dir="./results",
    overwrite_output_dir=True,
    
    # 训练参数
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    max_grad_norm=1.0,
    
    # 评估与保存
    evaluation_strategy="steps",  # "no", "steps", "epoch"
    eval_steps=500,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    
    # 日志
    logging_dir="./logs",
    logging_steps=100,
    report_to=["tensorboard", "wandb"],
    
    # 混合精度
    fp16=True,  # NVIDIA GPU
    bf16=False,  # TPU / Ampere GPU
    fp16_opt_level="O1",
    
    # 分布式训练
    ddp_backend="nccl",
    ddp_find_unused_parameters=False,
    
    # DeepSpeed
    deepspeed="ds_config.json",
    
    # 其他
    seed=42,
    dataloader_num_workers=4,
    remove_unused_columns=True,
    push_to_hub=False,
)
```

### D.3 Generation Config 参数

```python
from transformers import GenerationConfig

config = GenerationConfig(
    # 长度控制
    max_length=100,
    max_new_tokens=50,
    min_length=0,
    min_new_tokens=0,
    
    # 采样策略
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    
    # Beam Search
    num_beams=5,
    num_beam_groups=1,
    diversity_penalty=0.0,
    early_stopping=True,
    
    # 重复控制
    repetition_penalty=1.2,
    no_repeat_ngram_size=3,
    encoder_no_repeat_ngram_size=0,
    
    # 停止条件
    eos_token_id=2,
    pad_token_id=0,
    forced_eos_token_id=None,
    
    # 多样性
    num_return_sequences=1,
    output_scores=False,
    output_attentions=False,
    output_hidden_states=False,
    return_dict_in_generate=False,
)

# 使用
outputs = model.generate(**inputs, generation_config=config)
```

### D.4 PEFT 配置参数

**LoRA 配置**：
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,  # 秩
    lora_alpha=32,  # 缩放因子
    target_modules=["q_proj", "v_proj"],  # 目标模块
    lora_dropout=0.1,
    bias="none",  # "none", "all", "lora_only"
    task_type="CAUSAL_LM",  # "SEQ_CLS", "SEQ_2_SEQ_LM", etc.
)

model = get_peft_model(model, lora_config)
```

**QLoRA 配置**：
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # "fp4", "nf4"
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # 嵌套量化
)

model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    quantization_config=bnb_config
)
```

**Prefix Tuning 配置**：
```python
from peft import PrefixTuningConfig

prefix_config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,
    prefix_projection=False,
)
```

**Prompt Tuning 配置**：
```python
from peft import PromptTuningConfig

prompt_config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,
    prompt_tuning_init="TEXT",  # "RANDOM", "TEXT"
    prompt_tuning_init_text="Classify if the tweet is positive, negative or neutral:",
    tokenizer_name_or_path="model_name",
)
```

---

## 附录总结

本附录提供了实用的参考资料：

- **Appendix A**: 5 个常见错误的诊断与解决方案
- **Appendix B**: 4 张性能基准对比表
- **Appendix C**: 官方文档、论文、项目、社区资源
- **Appendix D**: 4 类 API 速查表

**建议使用方式**：
1. 遇到问题时，先查 Appendix A 常见错误
2. 性能优化时，参考 Appendix B 基准数据
3. 深入学习时，浏览 Appendix C 论文和项目
4. 编码时，使用 Appendix D 作为速查手册

---

**🎉 恭喜！您已完成 Hugging Face Transformers 完整教程！**

从零基础的 Pipeline 到前沿的 MoE、Mamba、RLHF，您现在具备了：
- ✅ 系统的理论知识（28 章 + 附录）
- ✅ 丰富的实战经验（500+ 代码示例）
- ✅ 深度的底层理解（70+ 交互式组件）
- ✅ 生产级的工程能力（分布式、量化、部署）

**下一步行动建议**：
1. 选择一个感兴趣的项目动手实践
2. 加入 Hugging Face 社区交流
3. 关注最新论文和模型发布
4. 贡献开源项目，回馈社区

**继续学习的方向**：
- 多模态大模型（GPT-4V、Gemini）
- Agent 与工具调用
- 长上下文处理（100K+ tokens）
- 模型压缩极限优化

祝您在 AI 之路上越走越远！🚀

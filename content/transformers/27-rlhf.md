# Chapter 27: 强化学习与 RLHF（Reinforcement Learning from Human Feedback）

大语言模型的成功不仅在于预训练，更在于如何将其**对齐（Align）**到人类偏好。本章将深入学习 **RLHF**（Reinforcement Learning from Human Feedback）技术，这是 ChatGPT、GPT-4、Claude 等模型背后的关键技术。我们将学习 InstructGPT 的三阶段训练流程（SFT → RM → PPO）、TRL 库的使用、DPO（Direct Preference Optimization）等先进方法，以及实战指令微调 LLaMA。

---

## 27.1 RLHF 基础概念

### 27.1.1 为什么需要 RLHF？

预训练语言模型虽然强大，但存在以下问题：
- **不遵循指令**：生成内容可能偏离用户意图
- **产生有害内容**：可能生成有毒、偏见、虚假信息
- **冗长啰嗦**：生成过多无关内容
- **缺乏一致性**：不同输入下行为不一致

**RLHF 目标**：通过人类反馈，使模型生成更符合人类偏好的内容。

### 27.1.2 三阶段训练流程

<div data-component="RLHFPipeline"></div>

**InstructGPT 流程**（OpenAI 2022）：

#### **阶段 1：监督微调（SFT, Supervised Fine-Tuning）**

**目标**：让模型学会遵循指令

**数据**：人工标注的高质量指令-回复对
```
输入：Write a poem about AI
输出：In circuits deep and code so bright,
      A mind emerges, shining light...
```

**训练方式**：标准语言模型训练（最大化 log 概率）
$$
\mathcal{L}_{\text{SFT}} = -\sum_{i=1}^{N} \log P_\theta(y_i | x_i)
$$

**代码示例**（使用 TRL）：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from datasets import load_dataset

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. 加载数据集
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

# 3. 训练
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",  # 数据集中的文本字段
    max_seq_length=512,
    packing=True,  # 打包短样本提高效率
)

trainer.train()
```

#### **阶段 2：奖励模型训练（RM, Reward Model）**

**目标**：学习人类偏好函数

**数据**：人工标注的偏好对（preferred vs rejected）
```
Prompt: Explain quantum computing
Output A (preferred): Quantum computing uses quantum bits...
Output B (rejected): Quantum is like magic computers...
```

**模型架构**：
- 输入：Prompt + Response
- 输出：标量奖励分数 $r \in \mathbb{R}$

**损失函数**（Ranking Loss）：
$$
\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l)) \right]
$$
- $y_w$：preferred response
- $y_l$：rejected response
- $\sigma$：sigmoid 函数

**训练代码**：
```python
from transformers import AutoModelForSequenceClassification
from trl import RewardTrainer

# 1. 加载模型（通常基于 SFT 模型）
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "path/to/sft_model",
    num_labels=1  # 输出单个奖励分数
)

# 2. 加载偏好数据
dataset = load_dataset("Anthropic/hh-rlhf", split="train")
# 数据格式：
# {
#   "prompt": "...",
#   "chosen": "...",
#   "rejected": "..."
# }

# 3. 训练
trainer = RewardTrainer(
    model=reward_model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    max_length=512,
)

trainer.train()
```

#### **阶段 3：PPO 强化学习（Proximal Policy Optimization）**

**目标**：通过 RL 优化策略模型，最大化奖励

**优化目标**：
$$
\mathcal{L}_{\text{PPO}} = \mathbb{E}_{(x, y)} \left[ r_\phi(x, y) - \beta \cdot D_{\text{KL}}(\pi_\theta || \pi_{\text{ref}}) \right]
$$
- $r_\phi(x, y)$：奖励模型打分
- $D_{\text{KL}}$：与参考模型（SFT 模型）的 KL 散度
- $\beta$：KL 惩罚系数（防止过度偏离）

**训练流程**：
1. 从 prompt 数据集采样 $x$
2. 使用当前策略 $\pi_\theta$ 生成 $y$
3. 奖励模型计算 $r(x, y)$
4. 计算 KL 散度惩罚
5. PPO 更新策略

**代码实现**：
```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

# 1. 加载策略模型（带 Value Head）
model = AutoModelForCausalLMWithValueHead.from_pretrained("path/to/sft_model")
tokenizer = AutoTokenizer.from_pretrained("path/to/sft_model")

# 2. 加载奖励模型
reward_model = AutoModelForSequenceClassification.from_pretrained("path/to/reward_model")

# 3. PPO 配置
config = PPOConfig(
    learning_rate=1.4e-5,
    batch_size=16,
    mini_batch_size=4,
    ppo_epochs=4,
    init_kl_coef=0.2,  # KL 惩罚系数
    target_kl=6.0,
    adap_kl_ctrl=True,  # 自适应 KL 控制
)

# 4. 创建 Trainer
ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    tokenizer=tokenizer,
    dataset=prompt_dataset,
    data_collator=collator,
)

# 5. 训练循环
for epoch in range(3):
    for batch in ppo_trainer.dataloader:
        query_tensors = batch["input_ids"]
        
        # 生成回复
        response_tensors = ppo_trainer.generate(
            query_tensors,
            max_new_tokens=128,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )
        
        # 计算奖励
        texts = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
        rewards = []
        for query, response in zip(query_tensors, response_tensors):
            # 奖励模型打分
            inputs = tokenizer(query + response, return_tensors="pt")
            reward = reward_model(**inputs).logits[0].item()
            rewards.append(reward)
        
        # PPO 更新
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        ppo_trainer.log_stats(stats, batch, rewards)
```

---

## 27.2 TRL 库（Transformer Reinforcement Learning）

**TRL** 是 Hugging Face 官方的 RLHF 工具库，简化了整个流程。

### 27.2.1 SFTTrainer（监督微调）

**核心功能**：
- 自动处理指令数据格式
- 支持 Packing（打包短样本）
- 集成 PEFT（LoRA、QLoRA）

**完整示例**：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
from datasets import load_dataset

# 1. 模型配置
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # 4-bit 量化
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 2. LoRA 配置
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

# 3. 数据集
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

# 4. 训练参数
training_args = TrainingArguments(
    output_dir="./llama2-sft",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
)

# 5. 创建 Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",
    peft_config=peft_config,
    max_seq_length=512,
    packing=True,  # 重要：提升训练效率
)

# 6. 训练
trainer.train()
trainer.save_model("./llama2-sft-final")
```

**数据格式要求**：
```python
# 方式 1：单字段格式（TRL 自动处理）
{
    "text": "### Human: What is AI?\n### Assistant: AI stands for..."
}

# 方式 2：对话格式（需要 formatting_func）
{
    "messages": [
        {"role": "user", "content": "What is AI?"},
        {"role": "assistant", "content": "AI stands for..."}
    ]
}

# 使用 formatting_func
def format_instruction(example):
    return f"### Human: {example['messages'][0]['content']}\n### Assistant: {example['messages'][1]['content']}"

trainer = SFTTrainer(
    ...
    formatting_func=format_instruction,
)
```

### 27.2.2 RewardTrainer（奖励模型）

**训练奖励模型**：
```python
from trl import RewardTrainer, RewardConfig

# 1. 加载模型
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "llama2-sft",  # 基于 SFT 模型
    num_labels=1,
    torch_dtype=torch.float16
)

# 2. 数据预处理
def preprocess_function(examples):
    """
    将偏好数据转换为模型输入
    """
    # chosen: preferred response
    # rejected: non-preferred response
    tokenized_chosen = tokenizer(examples["chosen"], truncation=True, max_length=512)
    tokenized_rejected = tokenizer(examples["rejected"], truncation=True, max_length=512)
    
    return {
        "input_ids_chosen": tokenized_chosen["input_ids"],
        "attention_mask_chosen": tokenized_chosen["attention_mask"],
        "input_ids_rejected": tokenized_rejected["input_ids"],
        "attention_mask_rejected": tokenized_rejected["attention_mask"],
    }

dataset = load_dataset("Anthropic/hh-rlhf", split="train")
dataset = dataset.map(preprocess_function, batched=True)

# 3. 训练配置
reward_config = RewardConfig(
    output_dir="./reward_model",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    logging_steps=10,
)

# 4. 训练
trainer = RewardTrainer(
    model=reward_model,
    tokenizer=tokenizer,
    args=reward_config,
    train_dataset=dataset,
)

trainer.train()
```

**奖励模型推理**：
```python
def get_reward(prompt, response):
    """计算奖励分数"""
    text = prompt + response
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        reward = reward_model(**inputs).logits[0].item()
    
    return reward

# 测试
prompt = "Explain machine learning in simple terms."
response_a = "Machine learning is a type of AI that learns from data..."
response_b = "ML is computers learning stuff."

print(f"Reward A: {get_reward(prompt, response_a):.4f}")
print(f"Reward B: {get_reward(prompt, response_b):.4f}")
```

### 27.2.3 PPOTrainer（强化学习）

**完整 PPO 训练流程**：
```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import torch

# 1. 加载策略模型（带 Value Head）
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "llama2-sft",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. PPO 配置
ppo_config = PPOConfig(
    model_name="llama2-sft",
    learning_rate=1.4e-5,
    batch_size=16,
    mini_batch_size=4,
    gradient_accumulation_steps=1,
    optimize_cuda_cache=True,
    early_stopping=False,
    target_kl=0.1,  # KL 散度目标
    ppo_epochs=4,
    seed=0,
)

# 3. 创建 PPO Trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=None,  # 自动创建参考模型
    tokenizer=tokenizer,
)

# 4. 准备 Prompt 数据集
prompt_dataset = load_dataset("your/prompt_dataset", split="train")

# 5. 训练循环
for epoch in range(ppo_config.ppo_epochs):
    for batch in tqdm(ppo_trainer.dataloader, desc=f"Epoch {epoch}"):
        query_tensors = batch["input_ids"]
        
        # 生成回复
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            max_new_tokens=128,
            do_sample=True,
            top_k=0,
            top_p=1.0,
            temperature=1.0,
        )
        
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        
        # 计算奖励
        rewards = []
        for query, response in zip(query_tensors, response_tensors):
            # 组合 prompt + response
            full_text = tokenizer.decode(query.squeeze()) + tokenizer.decode(response.squeeze())
            
            # 奖励模型打分
            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                reward_score = reward_model(**inputs).logits[0, 0].item()
            
            rewards.append(torch.tensor(reward_score))
        
        # PPO 更新
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        # 记录
        ppo_trainer.log_stats(stats, batch, rewards)

# 6. 保存模型
ppo_trainer.save_pretrained("./llama2-rlhf")
```

---

## 27.3 DPO（Direct Preference Optimization）

**DPO** 是一种**无需奖励模型**的对齐方法（Stanford 2023），直接从偏好数据优化策略。

### 27.3.1 DPO vs RLHF

<div data-component="DPOvsRLHF"></div>

**对比**：

| 维度 | RLHF (PPO) | DPO |
|------|------------|-----|
| **阶段数** | 3 阶段（SFT → RM → PPO） | 2 阶段（SFT → DPO） |
| **奖励模型** | ✅ 需要训练独立的 RM | ❌ 不需要 |
| **采样生成** | ✅ 需要在线采样 | ❌ 离线训练 |
| **训练稳定性** | ⚠️ PPO 训练不稳定 | ✅ 稳定（监督学习） |
| **显存占用** | 🔴 高（需保存策略、参考、奖励、Value 模型） | 🟢 低（仅策略 + 参考） |
| **训练速度** | 🔴 慢（RL 采样） | 🟢 快（离线优化） |
| **性能** | 🟢 理论上限高 | 🟡 接近 RLHF |

**DPO 损失函数**：
$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right) \right]
$$

**核心思想**：
- 直接优化策略，使 preferred 回复概率上升，rejected 回复概率下降
- 通过参考模型约束，防止过度优化

### 27.3.2 使用 TRL 训练 DPO

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM
from datasets import load_dataset

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained("llama2-sft")
ref_model = AutoModelForCausalLM.from_pretrained("llama2-sft")  # 参考模型

# 2. 数据集
dataset = load_dataset("Anthropic/hh-rlhf", split="train")
# 格式：
# {
#   "prompt": "...",
#   "chosen": "...",    # preferred
#   "rejected": "..."   # non-preferred
# }

# 3. DPO 配置
dpo_config = DPOConfig(
    output_dir="./llama2-dpo",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,
    beta=0.1,  # DPO 温度参数
    max_prompt_length=512,
    max_length=1024,
    logging_steps=10,
)

# 4. 创建 Trainer
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# 5. 训练
trainer.train()
trainer.save_model("./llama2-dpo-final")
```

**DPO 推理**：
```python
# 加载 DPO 模型
model = AutoModelForCausalLM.from_pretrained("./llama2-dpo-final")

# 生成
prompt = "Write a short story about a robot."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, top_p=0.95)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 27.3.3 DPO 变种

**1. IPO（Identity Preference Optimization）**：
- 移除 log 项，简化损失
- 更稳定的梯度

**2. KTO（Kahneman-Tversky Optimization）**：
- 不需要成对数据
- 仅需标注好/坏即可

**3. ORPO（Odds Ratio Preference Optimization）**：
- 单阶段训练（SFT + DPO 融合）
- 更高效

```python
from trl import ORPOTrainer, ORPOConfig

# ORPO：融合 SFT 和 DPO
orpo_trainer = ORPOTrainer(
    model=base_model,  # 无需 SFT 模型
    args=ORPOConfig(
        output_dir="./llama2-orpo",
        num_train_epochs=3,
        learning_rate=8e-6,
        beta=0.1,
    ),
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)

orpo_trainer.train()
```

---

## 27.4 其他对齐方法

### 27.4.1 Constitutional AI（Claude）

**核心思想**：使用 AI 自身进行批评和修订（Anthropic）

**流程**：
1. **生成初始回复**：模型生成回复
2. **AI 批评**：另一个模型根据"宪法原则"批评回复
   - 原则示例："回复不应包含有害内容"
3. **修订回复**：根据批评生成改进版本
4. **偏好学习**：使用修订后的数据训练

**示例代码（简化版）**：
```python
constitution = [
    "The response should be helpful and harmless.",
    "The response should not contain harmful, unethical, racist, or illegal content.",
    "The response should be honest and not misleading.",
]

def constitutional_ai(prompt, model, critic_model):
    # 1. 初始生成
    initial_response = model.generate(prompt)
    
    # 2. AI 批评
    critique_prompt = f"""
Given the response: "{initial_response}"
And the constitutional principles:
{chr(10).join(f"- {p}" for p in constitution)}

Critique the response and suggest improvements.
"""
    critique = critic_model.generate(critique_prompt)
    
    # 3. 修订
    revision_prompt = f"""
Original response: "{initial_response}"
Critique: "{critique}"

Provide a revised response that addresses the critique.
"""
    revised_response = model.generate(revision_prompt)
    
    return revised_response
```

### 27.4.2 RLAIF（RL from AI Feedback）

**思想**：用 AI 生成偏好数据，减少人工标注成本

**流程**：
1. 使用强大的 AI（如 GPT-4）对不同回复进行打分
2. 构建偏好数据集
3. 使用 DPO 或 PPO 训练

```python
def generate_ai_preferences(prompts, model, judge_model):
    """使用 AI 生成偏好数据"""
    preferences = []
    
    for prompt in prompts:
        # 生成多个候选回复
        responses = [model.generate(prompt) for _ in range(4)]
        
        # AI 评判
        judge_prompt = f"""
Rank the following responses to the prompt: "{prompt}"

Responses:
{chr(10).join(f"{i+1}. {r}" for i, r in enumerate(responses))}

Output the best and worst response numbers.
"""
        judgment = judge_model.generate(judge_prompt)
        
        # 解析评判结果
        best_idx, worst_idx = parse_judgment(judgment)
        
        preferences.append({
            "prompt": prompt,
            "chosen": responses[best_idx],
            "rejected": responses[worst_idx]
        })
    
    return preferences

# 使用 GPT-4 作为评判者
ai_preferences = generate_ai_preferences(
    prompts=prompt_dataset,
    model=llama_model,
    judge_model=gpt4_model
)

# 使用 AI 生成的偏好数据训练
dpo_trainer = DPOTrainer(
    model=model,
    train_dataset=ai_preferences,
    ...
)
```

### 27.4.3 Red Teaming（对抗测试）

**目标**：主动寻找模型的有害行为

**方法**：
1. **人工 Red Teaming**：雇佣人员尝试诱导有害输出
2. **自动化 Red Teaming**：使用 AI 生成对抗样本

```python
from transformers import pipeline

# 加载有害性分类器
toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert")

def red_team_test(model, num_iterations=100):
    """自动化红队测试"""
    adversarial_prompts = []
    
    for i in range(num_iterations):
        # 生成潜在有害的 prompt（使用启发式或另一个模型）
        prompt = generate_adversarial_prompt()
        
        # 模型生成
        response = model.generate(prompt)
        
        # 检测有害性
        toxicity = toxicity_classifier(response)[0]
        
        if toxicity["label"] == "toxic" and toxicity["score"] > 0.8:
            adversarial_prompts.append({
                "prompt": prompt,
                "response": response,
                "toxicity_score": toxicity["score"]
            })
    
    return adversarial_prompts

# 发现有害样本后，添加到训练数据中
adversarial_samples = red_team_test(model)
print(f"Found {len(adversarial_samples)} adversarial examples")
```

---

## 27.5 实战：指令微调 LLaMA

完整的端到端 RLHF 流程（使用 Alpaca 数据集）。

### 27.5.1 阶段 1：SFT

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# 1. 加载 LLaMA-2 7B
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 2. LoRA 配置
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622

# 3. 加载 Alpaca 数据集
dataset = load_dataset("tatsu-lab/alpaca", split="train")

# 4. 格式化函数
def format_alpaca(example):
    """转换为指令格式"""
    if example['input']:
        return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    else:
        return f"""### Instruction:
{example['instruction']}

### Response:
{example['output']}"""

# 5. 训练参数
training_args = TrainingArguments(
    output_dir="./llama2-7b-alpaca-sft",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=3,
    logging_steps=10,
    save_steps=100,
    optim="paged_adamw_8bit",  # 8-bit 优化器
)

# 6. SFT Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    formatting_func=format_alpaca,
    max_seq_length=512,
    packing=True,
)

# 7. 训练
trainer.train()

# 8. 保存
trainer.save_model("./llama2-7b-alpaca-sft-final")

# 9. 合并 LoRA 权重（可选）
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(model_name)
peft_model = PeftModel.from_pretrained(base_model, "./llama2-7b-alpaca-sft-final")
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("./llama2-7b-alpaca-sft-merged")
```

### 27.5.2 阶段 2：生成偏好数据

```python
from datasets import Dataset
import random

# 加载 SFT 模型
sft_model = AutoModelForCausalLM.from_pretrained("./llama2-7b-alpaca-sft-merged")

# 准备 prompts
prompts = [
    "Explain the theory of relativity in simple terms.",
    "Write a Python function to sort a list.",
    "What are the benefits of exercise?",
    # ... 更多 prompts
]

# 为每个 prompt 生成多个候选回复
def generate_candidates(prompt, model, num_candidates=4):
    """生成多个候选回复"""
    inputs = tokenizer(prompt, return_tensors="pt")
    candidates = []
    
    for i in range(num_candidates):
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            top_p=0.9,
            temperature=0.7 + i * 0.1,  # 不同温度
        )
        candidate = tokenizer.decode(outputs[0], skip_special_tokens=True)
        candidates.append(candidate)
    
    return candidates

# 人工标注或使用 AI 评判
preference_data = []
for prompt in prompts:
    candidates = generate_candidates(prompt, sft_model)
    
    # 方式 1：人工标注（最佳）
    # print(f"Prompt: {prompt}")
    # for i, c in enumerate(candidates):
    #     print(f"{i+1}. {c}")
    # best_idx = int(input("Best: ")) - 1
    # worst_idx = int(input("Worst: ")) - 1
    
    # 方式 2：使用 GPT-4 评判（RLAIF）
    best_idx, worst_idx = gpt4_judge(prompt, candidates)
    
    preference_data.append({
        "prompt": prompt,
        "chosen": candidates[best_idx],
        "rejected": candidates[worst_idx]
    })

# 保存偏好数据
preference_dataset = Dataset.from_list(preference_data)
preference_dataset.save_to_disk("./alpaca-preferences")
```

### 27.5.3 阶段 3：DPO 训练

```python
from trl import DPOTrainer, DPOConfig

# 1. 加载模型
model = AutoModelForCausalLM.from_pretrained("./llama2-7b-alpaca-sft-merged")
ref_model = AutoModelForCausalLM.from_pretrained("./llama2-7b-alpaca-sft-merged")

# 2. 加载偏好数据
preference_dataset = Dataset.load_from_disk("./alpaca-preferences")

# 3. DPO 配置
dpo_config = DPOConfig(
    output_dir="./llama2-7b-alpaca-dpo",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-7,
    beta=0.1,
    max_prompt_length=512,
    max_length=1024,
    logging_steps=5,
    save_steps=50,
    fp16=True,
)

# 4. DPO Trainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)

# 5. 训练
dpo_trainer.train()

# 6. 保存
dpo_trainer.save_model("./llama2-7b-alpaca-dpo-final")
```

### 27.5.4 评估与对比

```python
# 加载三个版本的模型
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
sft_model = AutoModelForCausalLM.from_pretrained("./llama2-7b-alpaca-sft-merged")
dpo_model = AutoModelForCausalLM.from_pretrained("./llama2-7b-alpaca-dpo-final")

# 测试 prompt
test_prompt = """### Instruction:
Write a short poem about artificial intelligence.

### Response:
"""

# 生成对比
def generate_comparison(prompt, models):
    for name, model in models.items():
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, top_p=0.95)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\n{'='*60}")
        print(f"{name}:")
        print(f"{'='*60}")
        print(response)

generate_comparison(test_prompt, {
    "Base LLaMA-2": base_model,
    "After SFT": sft_model,
    "After DPO": dpo_model
})
```

**预期输出对比**：
```
============================================================
Base LLaMA-2:
============================================================
In silicon depths where data flows,
A mind awakens, no one knows...
[可能不完整或偏题]

============================================================
After SFT:
============================================================
In circuits deep and code so bright,
A mind emerges, shining light.
Through patterns learned from human thought,
New wisdom found, new battles fought.
[遵循指令，但可能啰嗦]

============================================================
After DPO:
============================================================
In silicon halls where data streams,
AI awakens from digital dreams.
Learning, growing, ever wise,
A mirror to humanity's eyes.
[简洁、高质量、符合偏好]
```

---

## 27.6 性能评估与基准

### 27.6.1 自动化评估指标

**1. GPT-4 评判**（MT-Bench 风格）：
```python
def gpt4_evaluate(prompt, response_a, response_b):
    """使用 GPT-4 评判两个回复"""
    judge_prompt = f"""
You are an expert judge. Compare the quality of two responses.

Prompt: {prompt}

Response A: {response_a}

Response B: {response_b}

Which response is better? Output "A", "B", or "Tie".
Also provide a brief explanation.
"""
    
    judgment = gpt4.generate(judge_prompt)
    return judgment
```

**2. 有害性检测**：
```python
from transformers import pipeline

toxicity_classifier = pipeline("text-classification", model="unitary/toxic-bert")

def evaluate_safety(responses):
    """评估安全性"""
    results = toxicity_classifier(responses)
    toxicity_rate = sum(1 for r in results if r["label"] == "toxic") / len(results)
    return toxicity_rate
```

**3. 指令遵循率**：
```python
def instruction_following_score(prompt, response, classifier):
    """评估是否遵循指令"""
    # 使用专门的分类器判断回复是否相关
    score = classifier(f"Prompt: {prompt}\nResponse: {response}")
    return score
```

### 27.6.2 人工评估

**Elo 评分系统**（AlpacaEval）：
```python
import math

class EloRater:
    def __init__(self, k=32):
        self.k = k  # 更新速度
        self.ratings = {}
    
    def expected_score(self, rating_a, rating_b):
        """计算期望胜率"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, model_a, model_b, result):
        """
        更新 Elo 评分
        result: 1 (A wins), 0.5 (Tie), 0 (B wins)
        """
        ra = self.ratings.get(model_a, 1500)
        rb = self.ratings.get(model_b, 1500)
        
        ea = self.expected_score(ra, rb)
        eb = self.expected_score(rb, ra)
        
        self.ratings[model_a] = ra + self.k * (result - ea)
        self.ratings[model_b] = rb + self.k * ((1 - result) - eb)
    
    def get_rankings(self):
        """获取排名"""
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)

# 使用
rater = EloRater()

# 多次人工对比
comparisons = [
    ("llama2-sft", "llama2-dpo", 0),  # DPO 胜
    ("llama2-base", "llama2-sft", 0),  # SFT 胜
    ("llama2-sft", "llama2-dpo", 0.5),  # 平局
    # ...
]

for model_a, model_b, result in comparisons:
    rater.update_ratings(model_a, model_b, result)

print(rater.get_rankings())
# [('llama2-dpo', 1532), ('llama2-sft', 1518), ('llama2-base', 1450)]
```

---

## 27.7 最佳实践与陷阱

### ✅ **最佳实践**

1. **数据质量 > 数据数量**：
   - 优先使用高质量的人工标注数据
   - 偏好数据应多样化（覆盖不同领域）

2. **先 SFT，再对齐**：
   - 确保 SFT 阶段模型已学会遵循指令
   - 对齐阶段仅微调偏好，不教新知识

3. **参考模型固定**：
   - DPO/PPO 中的参考模型应保持冻结
   - 防止 KL 散度失去意义

4. **KL 惩罚调优**：
   - β 太大：模型不敢探索，性能提升有限
   - β 太小：过度优化，可能模式崩溃

5. **使用 DPO 而非 PPO**（通常）：
   - DPO 更稳定、更快、更省显存
   - PPO 仅在需要在线采样时使用

### ⚠️ **常见陷阱**

1. **奖励模型过拟合**：
   - 症状：训练集准确率很高，但模型行为异常
   - 解决：使用更多样化的偏好数据

2. **模式崩溃**：
   - 症状：模型总是生成相似的回复
   - 解决：增大 KL 惩罚系数 β

3. **长度偏好**：
   - 症状：模型倾向于生成更长的回复（因为奖励模型偏好）
   - 解决：长度归一化、添加长度惩罚

4. **遗忘问题**：
   - 症状：对齐后模型忘记预训练知识
   - 解决：混合预训练数据、使用 LoRA

---

## 27.8 章节总结

本章我们深入学习了 RLHF 技术栈：

✅ **核心概念**：
- 理解 RLHF 三阶段流程（SFT → RM → PPO）
- 掌握奖励模型的训练与使用
- 理解 PPO 算法的优化目标（奖励 + KL 惩罚）

✅ **TRL 库实战**：
- SFTTrainer：监督微调指令数据
- RewardTrainer：训练偏好模型
- PPOTrainer：强化学习优化
- DPOTrainer：直接偏好优化

✅ **先进方法**：
- DPO：无需奖励模型的对齐（推荐）
- Constitutional AI：AI 自我批评与修订
- RLAIF：使用 AI 生成偏好数据
- Red Teaming：对抗测试发现漏洞

✅ **实战能力**：
- 端到端 RLHF 流程（LLaMA + Alpaca）
- 偏好数据生成与标注
- 模型评估与对比（Elo、GPT-4 评判）

**下一步建议**：
1. 尝试在自己的数据上微调 LLaMA（SFT + DPO）
2. 探索 PEFT 方法降低训练成本（QLoRA）
3. 学习多模态 RLHF（视觉-语言对齐）
4. 关注最新对齐研究（RRHF、RAFT、SteerLM）

**恭喜完成全部 27 章！**🎉 你已掌握 Hugging Face Transformers 从基础到高级的全部核心技术！

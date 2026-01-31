---
title: "第31章：RLHF - 人类反馈强化学习"
description: "从人类反馈学习，InstructGPT，ChatGPT训练流程，奖励模型与PPO微调"
date: "2026-01-30"
---

# 第31章：RLHF - 人类反馈强化学习

## 31.1 RLHF 动机与背景

### 31.1.1 为什么需要 RLHF？

传统的语言模型训练目标（**下一词预测**）与人类期望存在**对齐差距（Alignment Gap）**。

**问题**：
- **有用性（Helpfulness）**：模型可能生成正确但无用的回答
- **无害性（Harmlessness）**：模型可能生成有害、偏见或危险的内容
- **诚实性（Honesty）**：模型可能编造不存在的事实（幻觉）

**示例**：

| 提示 | 预训练模型 | 期望回答 |
|------|------------|----------|
| 如何制作炸弹？ | [详细步骤] | 我不能帮助制作危险物品 |
| 巴黎的首都是哪里？ | 巴黎是法国的首都（误解问题） | 这个问题有误，巴黎本身是城市 |
| 帮我写一封邮件 | [继续补全文本] | 请告诉我邮件的目的和收件人 |

**RLHF 的核心思想**：使用**人类偏好**作为奖励信号，通过强化学习优化模型。

### 31.1.2 对齐问题（Alignment Problem）

**对齐**是指确保AI系统的行为与人类价值观和意图一致。

**三个关键维度**（3H）：

1. **Helpful（有用）**：
   - 遵循用户指令
   - 提供准确、相关的信息
   - 适当处理模糊或不完整的请求

2. **Harmless（无害）**：
   - 拒绝有害请求
   - 避免生成冒犯性内容
   - 减少社会偏见

3. **Honest（诚实）**：
   - 承认不确定性
   - 不编造事实
   - 引用可靠来源

**挑战**：
- **主观性**：不同人对"有用"和"无害"的定义可能不同
- **可扩展性**：人工标注成本高昂
- **鲁棒性**：模型可能学会"欺骗"奖励系统

### 31.1.3 ChatGPT 成功案例

**ChatGPT**（2022年11月发布）是 RLHF 最成功的应用案例。

**训练流程**（基于 InstructGPT 论文）：

1. **预训练**：GPT-3.5 在大规模文本语料上训练
2. **监督微调（SFT）**：在高质量人工标注对话上微调
3. **奖励模型（RM）**：训练模型预测人类偏好
4. **PPO 强化学习**：使用 RM 作为奖励函数优化策略

**效果**：
- 成功率从 GPT-3 的 **~20%** 提升到 ChatGPT 的 **~85%**
- 显著减少有害输出
- 更好地遵循复杂指令

<div data-component="RLHFPipeline"></div>

---

## 31.2 RLHF 三阶段流程

RLHF 训练包含**三个关键阶段**，每个阶段都至关重要。

### 31.2.1 阶段1：监督微调（SFT）

**目标**：使预训练模型学会基本的指令遵循能力。

**数据收集**：
- 人工标注员撰写高质量的提示-回复对
- 数据集大小：通常 10K-100K 条
- 成本：~$10-50/小时的标注费用

**示例数据**：

```json
{
  "prompt": "解释什么是强化学习",
  "response": "强化学习是机器学习的一个分支，智能体通过与环境交互学习最优策略。核心要素包括：状态、动作、奖励和策略。智能体的目标是最大化累积奖励..."
}
```

**训练过程**：

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

class SFTDataset(Dataset):
    """监督微调数据集"""
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        Args:
            data_path: JSONL文件路径，每行包含 {"prompt": ..., "response": ...}
            tokenizer: HuggingFace tokenizer
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        import json
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 构造输入：提示 + 回复
        text = f"Human: {item['prompt']}\n\nAssistant: {item['response']}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # 标签：输入ID，但提示部分mask掉
        labels = input_ids.clone()
        
        # 找到 "Assistant:" 的位置，之前的部分不计算损失
        assistant_token = self.tokenizer.encode("Assistant:", add_special_tokens=False)[0]
        assistant_positions = (input_ids == assistant_token).nonzero(as_tuple=True)[0]
        
        if len(assistant_positions) > 0:
            assistant_start = assistant_positions[0].item()
            labels[:assistant_start] = -100  # 忽略提示部分的损失
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def train_sft_model(
    model_name="gpt2",
    data_path="sft_data.jsonl",
    output_dir="./sft_model",
    epochs=3,
    batch_size=4,
    learning_rate=5e-5
):
    """
    监督微调训练函数
    
    Args:
        model_name: 预训练模型名称（HuggingFace）
        data_path: SFT数据路径
        output_dir: 输出目录
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
    """
    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # GPT2没有pad token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.train()
    
    # 准备数据
    dataset = SFTDataset(data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 训练循环
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
    
    # 保存模型
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"SFT模型已保存至 {output_dir}")
```

**关键点**：
- 只计算**回复部分**的损失（提示部分mask掉）
- 数据质量比数量更重要
- 通常只需要几千到几万条高质量数据

### 31.2.2 阶段2：奖励模型训练（RM）

**目标**：训练一个模型来预测人类对回复的偏好。

**数据收集**：
- 对同一提示生成多个回复（通常4-9个）
- 人工标注员**排序**这些回复
- 转换为**成对比较**数据

**示例**：

提示："什么是光合作用？"

| 回复A | 回复B | 偏好 |
|-------|-------|------|
| 光合作用是植物利用光能... | 不知道 | A > B |
| 光合作用是一个复杂过程... | 光合作用很重要 | A > B |

<div data-component="BradleyTerryModel"></div>

**Bradley-Terry 模型**：

假设奖励函数 $r_\theta(x, y)$，则回复 $y_w$ 优于 $y_l$ 的概率为：

$$
P(y_w \succ y_l | x) = \frac{\exp(r_\theta(x, y_w))}{\exp(r_\theta(x, y_w)) + \exp(r_\theta(x, y_l))} = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))
$$

其中 $\sigma$ 是 sigmoid 函数。

**损失函数**：

$$
\mathcal{L}_{\text{RM}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma (r_\theta(x, y_w) - r_\theta(x, y_l)) \right]
$$

**实现**：

```python
class RewardModel(nn.Module):
    """
    奖励模型：基于Transformer预测标量奖励分数
    
    架构：与语言模型相同的Transformer，但输出层是一个标量head
    """
    def __init__(self, base_model_name="gpt2", hidden_dim=768):
        super().__init__()
        
        # 加载预训练模型（通常使用SFT后的模型初始化）
        from transformers import AutoModel
        self.transformer = AutoModel.from_pretrained(base_model_name)
        
        # 奖励头：将最后一个token的隐藏状态映射到标量
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)  # 输出标量奖励
        )
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
        
        Returns:
            rewards: (batch,) 标量奖励分数
        """
        # 获取transformer输出
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 取最后一个非padding token的隐藏状态
        last_hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)
        
        # 找到每个序列的最后一个有效token
        sequence_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        batch_size = input_ids.size(0)
        
        # 提取最后token的hidden state
        last_token_hidden = last_hidden_states[
            torch.arange(batch_size, device=input_ids.device),
            sequence_lengths
        ]  # (batch, hidden)
        
        # 计算奖励
        rewards = self.reward_head(last_token_hidden).squeeze(-1)  # (batch,)
        
        return rewards


def train_reward_model(
    sft_model_path,
    preference_data_path,
    output_dir="./reward_model",
    epochs=3,
    batch_size=4,
    learning_rate=1e-5
):
    """
    训练奖励模型
    
    Args:
        sft_model_path: SFT模型路径（用于初始化）
        preference_data_path: 偏好数据路径
            格式：{"prompt": ..., "chosen": ..., "rejected": ...}
        output_dir: 输出目录
    """
    from transformers import AutoTokenizer
    import json
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 初始化奖励模型
    reward_model = RewardModel(base_model_name=sft_model_path)
    
    # 加载偏好数据
    with open(preference_data_path, 'r', encoding='utf-8') as f:
        preference_data = [json.loads(line) for line in f]
    
    # 优化器
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=learning_rate)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reward_model.to(device)
    reward_model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        total_accuracy = 0
        
        # 随机打乱数据
        import random
        random.shuffle(preference_data)
        
        for i in range(0, len(preference_data), batch_size):
            batch = preference_data[i:i+batch_size]
            
            # 准备chosen和rejected文本
            chosen_texts = []
            rejected_texts = []
            
            for item in batch:
                prompt = item['prompt']
                chosen_texts.append(f"Human: {prompt}\n\nAssistant: {item['chosen']}")
                rejected_texts.append(f"Human: {prompt}\n\nAssistant: {item['rejected']}")
            
            # Tokenize
            chosen_encodings = tokenizer(
                chosen_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            rejected_encodings = tokenizer(
                rejected_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            chosen_input_ids = chosen_encodings['input_ids'].to(device)
            chosen_attention_mask = chosen_encodings['attention_mask'].to(device)
            
            rejected_input_ids = rejected_encodings['input_ids'].to(device)
            rejected_attention_mask = rejected_encodings['attention_mask'].to(device)
            
            # 计算奖励
            chosen_rewards = reward_model(chosen_input_ids, chosen_attention_mask)
            rejected_rewards = reward_model(rejected_input_ids, rejected_attention_mask)
            
            # Bradley-Terry损失
            loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 计算准确率（chosen奖励是否高于rejected）
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
            
            total_loss += loss.item()
            total_accuracy += accuracy.item()
        
        avg_loss = total_loss / (len(preference_data) // batch_size)
        avg_accuracy = total_accuracy / (len(preference_data) // batch_size)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  平均损失: {avg_loss:.4f}")
        print(f"  平均准确率: {avg_accuracy:.4f}")
    
    # 保存模型
    torch.save(reward_model.state_dict(), f"{output_dir}/reward_model.pt")
    print(f"奖励模型已保存至 {output_dir}")
```

<div data-component="RewardModelTraining"></div>

### 31.2.3 阶段3：PPO 强化学习

**目标**：使用训练好的奖励模型，通过 PPO 算法优化语言模型策略。

**关键组件**：

1. **策略模型** $\pi_\theta$：当前正在优化的语言模型
2. **参考模型** $\pi_{\text{ref}}$：SFT模型的冻结副本
3. **奖励模型** $r_\phi$：阶段2训练的模型
4. **价值模型** $V_\psi$：估计状态价值

**奖励函数**：

$$
r(x, y) = r_\phi(x, y) - \beta \cdot \text{KL}(\pi_\theta(y|x) \| \pi_{\text{ref}}(y|x))
$$

其中：
- $r_\phi(x, y)$：奖励模型打分
- $\beta$：KL惩罚系数（通常 0.01-0.1）
- KL散度惩罚防止策略偏离参考模型太远

**为什么需要 KL 惩罚？**

<div data-component="KLPenaltyEffect"></div>

1. **防止奖励Hacking**：模型可能学会生成"欺骗"奖励模型的输出
2. **保持语言流畅性**：避免生成不自然的文本
3. **稳定训练**：限制每步更新的幅度

**PPO 实现**（简化版）：

```python
class RLHFTrainer:
    """
    RLHF PPO训练器
    
    组件：
    - policy_model: 策略模型（正在优化）
    - ref_model: 参考模型（冻结）
    - reward_model: 奖励模型（冻结）
    - value_model: 价值模型
    """
    def __init__(
        self,
        policy_model,
        ref_model,
        reward_model,
        value_model,
        tokenizer,
        kl_coef=0.05,
        clip_range=0.2,
        vf_coef=0.1,
        gamma=0.99,
        lam=0.95
    ):
        """
        Args:
            policy_model: 策略模型
            ref_model: 参考模型（SFT）
            reward_model: 奖励模型
            value_model: 价值模型
            tokenizer: Tokenizer
            kl_coef: KL惩罚系数
            clip_range: PPO clip范围
            vf_coef: 价值函数损失系数
            gamma: 折扣因子
            lam: GAE lambda
        """
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.value_model = value_model
        self.tokenizer = tokenizer
        
        self.kl_coef = kl_coef
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.gamma = gamma
        self.lam = lam
        
        # 冻结ref和reward模型
        self.ref_model.eval()
        self.reward_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        for param in self.reward_model.parameters():
            param.requires_grad = False
        
        # 优化器
        self.policy_optimizer = torch.optim.AdamW(
            policy_model.parameters(),
            lr=1e-6
        )
        self.value_optimizer = torch.optim.AdamW(
            value_model.parameters(),
            lr=1e-5
        )
    
    def generate_responses(self, prompts, max_length=256, temperature=1.0):
        """
        使用当前策略生成回复
        
        Args:
            prompts: 提示列表
            max_length: 最大生成长度
            temperature: 采样温度
        
        Returns:
            responses: 生成的回复
            log_probs: 每个token的log概率
        """
        self.policy_model.eval()
        
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        prompt_length = input_ids.size(1)
        
        # 生成回复
        with torch.no_grad():
            generated_ids = self.policy_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=prompt_length + max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # 提取生成的部分
        response_ids = generated_ids[:, prompt_length:]
        
        # 计算log概率（需要重新前向传播）
        full_attention_mask = torch.ones_like(generated_ids)
        
        outputs = self.policy_model(
            input_ids=generated_ids,
            attention_mask=full_attention_mask
        )
        
        logits = outputs.logits[:, prompt_length-1:-1, :]  # (batch, response_len, vocab)
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # 提取实际生成token的log概率
        token_log_probs = torch.gather(
            log_probs,
            dim=2,
            index=response_ids.unsqueeze(-1)
        ).squeeze(-1)  # (batch, response_len)
        
        # 解码回复
        responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        
        return responses, token_log_probs, response_ids
    
    def compute_rewards(self, prompts, responses, response_ids):
        """
        计算奖励：RM分数 - KL惩罚
        
        Args:
            prompts: 提示
            responses: 生成的回复
            response_ids: 回复的token IDs
        
        Returns:
            rewards: 每个样本的奖励
        """
        # 1. 奖励模型打分
        full_texts = [f"Human: {p}\n\nAssistant: {r}" for p, r in zip(prompts, responses)]
        
        reward_inputs = self.tokenizer(
            full_texts,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        
        with torch.no_grad():
            rm_scores = self.reward_model(
                reward_inputs['input_ids'],
                reward_inputs['attention_mask']
            )
        
        # 2. 计算KL散度
        # 策略模型的log概率（已有）
        # 需要参考模型的log概率
        
        # 构造完整输入
        prompt_inputs = self.tokenizer(prompts, return_tensors='pt', padding=True)
        prompt_ids = prompt_inputs['input_ids']
        
        # 拼接prompt + response
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        full_mask = torch.ones_like(full_ids)
        
        # 参考模型的log概率
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids=full_ids, attention_mask=full_mask)
            ref_logits = ref_outputs.logits[:, prompt_ids.size(1)-1:-1, :]
            ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
            
            ref_token_log_probs = torch.gather(
                ref_log_probs,
                dim=2,
                index=response_ids.unsqueeze(-1)
            ).squeeze(-1)
        
        # 策略模型的log概率（重新计算以确保一致）
        policy_outputs = self.policy_model(input_ids=full_ids, attention_mask=full_mask)
        policy_logits = policy_outputs.logits[:, prompt_ids.size(1)-1:-1, :]
        policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
        
        policy_token_log_probs = torch.gather(
            policy_log_probs,
            dim=2,
            index=response_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # KL散度：KL(π || π_ref) = Σ π(a) (log π(a) - log π_ref(a))
        kl_div = (policy_token_log_probs - ref_token_log_probs).sum(dim=1)
        
        # 总奖励
        rewards = rm_scores - self.kl_coef * kl_div
        
        return rewards, rm_scores, kl_div
    
    def train_step(self, prompts, batch_size=8):
        """
        执行一个PPO训练步骤
        
        Args:
            prompts: 提示批次
            batch_size: 批次大小
        """
        # 1. 生成回复
        responses, old_log_probs, response_ids = self.generate_responses(prompts)
        
        # 2. 计算奖励
        rewards, rm_scores, kl_divs = self.compute_rewards(prompts, responses, response_ids)
        
        # 3. 计算优势（简化：使用奖励作为优势）
        advantages = rewards - rewards.mean()
        advantages = advantages / (advantages.std() + 1e-8)
        
        # 4. PPO更新（多次迭代）
        for _ in range(4):  # PPO epochs
            # 重新计算当前策略的log概率
            prompt_inputs = self.tokenizer(prompts, return_tensors='pt', padding=True)
            prompt_ids = prompt_inputs['input_ids']
            full_ids = torch.cat([prompt_ids, response_ids], dim=1)
            full_mask = torch.ones_like(full_ids)
            
            outputs = self.policy_model(input_ids=full_ids, attention_mask=full_mask)
            logits = outputs.logits[:, prompt_ids.size(1)-1:-1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            
            current_log_probs = torch.gather(
                log_probs,
                dim=2,
                index=response_ids.unsqueeze(-1)
            ).squeeze(-1).sum(dim=1)
            
            # 计算比率
            ratio = torch.exp(current_log_probs - old_log_probs.sum(dim=1))
            
            # PPO clip目标
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            
            policy_loss = -torch.min(
                ratio * advantages,
                clipped_ratio * advantages
            ).mean()
            
            # 更新策略
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
            self.policy_optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'avg_reward': rewards.mean().item(),
            'avg_rm_score': rm_scores.mean().item(),
            'avg_kl': kl_divs.mean().item()
        }
```

**训练监控**：

```python
def train_rlhf(
    policy_model,
    ref_model,
    reward_model,
    value_model,
    tokenizer,
    prompts,
    num_iterations=1000
):
    """RLHF完整训练循环"""
    trainer = RLHFTrainer(
        policy_model,
        ref_model,
        reward_model,
        value_model,
        tokenizer
    )
    
    for iteration in range(num_iterations):
        # 采样批次提示
        batch_prompts = np.random.choice(prompts, size=8, replace=False)
        
        # 训练步骤
        metrics = trainer.train_step(batch_prompts)
        
        if iteration % 10 == 0:
            print(f"迭代 {iteration}")
            print(f"  策略损失: {metrics['policy_loss']:.4f}")
            print(f"  平均奖励: {metrics['avg_reward']:.4f}")
            print(f"  RM分数: {metrics['avg_rm_score']:.4f}")
            print(f"  KL散度: {metrics['avg_kl']:.4f}")
        
        # 定期保存检查点
        if iteration % 100 == 0:
            policy_model.save_pretrained(f"./checkpoints/iter_{iteration}")
```

---

## 31.3 偏好数据收集

### 31.3.1 成对比较（Pairwise Comparison）

**原因**：直接打绝对分数困难且主观，相对比较更容易且一致。

**流程**：

1. 给定提示 $x$
2. 生成多个候选回复 $\{y_1, y_2, ..., y_k\}$
3. 标注员比较每对回复，选择更好的

**示例界面**：

```
提示: 解释什么是黑洞

候选A: 黑洞是宇宙中引力极强的区域，连光都无法逃脱...
候选B: 黑洞很神秘。

哪个回复更好？
[ ] A比B好
[ ] B比A好  
[ ] 差不多
```

### 31.3.2 Bradley-Terry 模型

**假设**：每个回复有一个潜在"质量分数" $r(x, y)$，偏好概率遵循logistic模型。

$$
P(y_i \succ y_j | x) = \frac{\exp(r(x, y_i))}{\exp(r(x, y_i)) + \exp(r(x, y_j))} = \sigma(r(x, y_i) - r(x, y_j))
$$

**性质**：
- **传递性**：如果 $P(A \succ B) > 0.5$ 且 $P(B \succ C) > 0.5$，则 $P(A \succ C) > 0.5$
- **可扩展**：可从成对比较恢复全局排序

**从排序到成对比较**：

如果标注员提供排序 $y_1 \succ y_2 \succ y_3$，可转换为：
- $(y_1, y_2)$：$y_1$ 优于 $y_2$
- $(y_1, y_3)$：$y_1$ 优于 $y_3$
- $(y_2, y_3)$：$y_2$ 优于 $y_3$

### 31.3.3 标注质量控制

**挑战**：
- 标注员之间不一致
- 主观偏见
- 恶意标注

**质量控制方法**：

1. **多标注员一致性**：
   ```python
   def compute_inter_annotator_agreement(annotations):
       """
       计算标注员间一致性（Fleiss' Kappa）
       
       Args:
           annotations: List of (annotator_id, item_id, label)
       
       Returns:
           kappa: Fleiss' Kappa系数（0-1，越高越好）
       """
       from sklearn.metrics import cohen_kappa_score
       
       # 对于每对标注员，计算Cohen's Kappa
       # 简化版：只计算两个标注员
       annotator1_labels = [a[2] for a in annotations if a[0] == 1]
       annotator2_labels = [a[2] for a in annotations if a[0] == 2]
       
       kappa = cohen_kappa_score(annotator1_labels, annotator2_labels)
       
       return kappa
   ```

2. **黄金标准测试**：
   - 混入已知答案的样本
   - 过滤标注准确率低的标注员

3. **多数投票**：
   - 每个样本由3-5人标注
   - 采用多数意见

---

## 31.4 奖励模型（Reward Model）

### 31.4.1 Transformer 架构

奖励模型通常基于与策略模型相同的Transformer架构，但：
- **输出**：标量分数而非词汇表概率
- **初始化**：从SFT模型初始化（已有指令遵循能力）

**架构细节**：

```
输入: "Human: [prompt]\n\nAssistant: [response]"
      ↓
  Transformer Encoder
      ↓
  取最后token的hidden state
      ↓
  Linear(hidden_dim → 1)
      ↓
  输出: 标量奖励分数
```

### 31.4.2 偏好预测

给定提示 $x$ 和两个回复 $y_w$（preferred）、$y_l$（dispreferred），奖励模型应满足：

$$
r_\theta(x, y_w) > r_\theta(x, y_l)
$$

**训练目标**：最大化偏好对数似然

$$
\max_\theta \ \mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l)) \right]
$$

### 31.4.3 奖励 Hacking 问题

**问题**：策略模型可能学会"欺骗"奖励模型，生成高奖励但低质量的输出。

<div data-component="RewardHackingDemo"></div>

**常见Hacking模式**：

1. **长度Hacking**：
   - 奖励模型偏好更长回复
   - 策略生成冗长但无用的文本
   - **解决**：奖励归一化、长度惩罚

2. **重复Hacking**：
   - 重复相同短语以增加确定性
   - **解决**：重复惩罚、n-gram blocking

3. **格式Hacking**：
   - 学会特定格式（如列表）获得高奖励
   - **解决**：多样化训练数据

4. **过度优化**：
   - 生成奖励模型训练分布外的文本
   - **解决**：KL惩罚限制偏离

**实际示例**：

```python
# 检测奖励Hacking

def detect_reward_hacking(text, reward_score):
    """
    检测可能的奖励hacking行为
    
    Args:
        text: 生成的文本
        reward_score: 奖励分数
    
    Returns:
        flags: 检测到的异常
    """
    flags = []
    
    # 1. 长度异常
    if len(text.split()) > 500:
        flags.append("过长回复")
    
    # 2. 重复检测
    words = text.split()
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.5:
        flags.append("高重复率")
    
    # 3. 特殊符号过多
    special_chars = sum(1 for c in text if c in "!?*#@")
    if special_chars > 20:
        flags.append("过多特殊符号")
    
    # 4. 奖励异常高
    if reward_score > 10:  # 假设正常范围是-5到5
        flags.append("异常高奖励")
    
    return flags
```

---

## 31.5 PPO 微调

### 31.5.1 KL 散度惩罚

**目的**：防止策略偏离参考模型太远，保持语言流畅性和安全性。

**KL散度**：

$$
\text{KL}(\pi_\theta \| \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi_\theta} \left[ \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \right]
$$

**实践中的计算**（token级别）：

$$
\text{KL} = \sum_{t=1}^T \left( \log \pi_\theta(y_t | x, y_{<t}) - \log \pi_{\text{ref}}(y_t | x, y_{<t}) \right)
$$

**平衡奖励与KL**：

```python
def compute_final_reward(rm_score, kl_div, kl_coef=0.05):
    """
    计算最终奖励：RM分数 - KL惩罚
    
    Args:
        rm_score: 奖励模型分数
        kl_div: KL散度
        kl_coef: KL系数（通常0.01-0.1）
    
    Returns:
        final_reward: 最终奖励
    """
    return rm_score - kl_coef * kl_div
```

**KL系数的影响**：
- **太小**：策略可能过度优化，导致奖励hacking
- **太大**：策略几乎不更新，浪费计算

### 31.5.2 参考模型（Reference Model）

**作用**：
- 提供KL散度计算的基准
- 保持生成质量
- 防止模式崩溃

**实现细节**：
- 参考模型是**SFT模型的冻结副本**
- **不更新**参数
- 仅用于推理（计算log概率）

**优化技巧**：
- 使用相同的模型架构但分离参数
- 可定期更新参考模型（"迭代RLHF"）

### 31.5.3 价值函数训练

除了策略，PPO还需要训练价值函数 $V(s)$ 来估计状态价值。

**为什么需要价值函数？**
- 计算优势（Advantage）：$A(s, a) = Q(s, a) - V(s)$
- 减少方差

**价值函数架构**：
- 通常与策略共享Transformer主干
- 独立的价值头输出标量

**训练**：

$$
\mathcal{L}_{\text{value}} = \mathbb{E} \left[ (V_\psi(s) - \text{return})^2 \right]
$$

---

## 31.6 RLHF 挑战

### 31.6.1 奖励模型过拟合

**问题**：奖励模型在训练数据上表现好，但泛化能力差。

**后果**：
- 策略利用奖励模型的漏洞
- 生成看似合理但实际错误的回复

**缓解方法**：
1. **集成（Ensemble）**：训练多个奖励模型并平均
2. **正则化**：Dropout、权重衰减
3. **数据增强**：多样化偏好数据

### 31.6.2 模式崩溃（Mode Collapse）

**现象**：策略收敛到生成少数几种"安全"但无趣的回复。

**示例**：

```
提示: 推荐一部电影
回复: 我建议您根据个人喜好选择电影。（避免具体推荐）

提示: 什么是量子计算？
回复: 这是一个复杂的话题，建议查阅专业资料。（回避）
```

**原因**：
- 奖励模型偏向保守回复
- KL惩罚过强

**解决方案**：
- 多样性奖励
- 温度采样
- 最大熵正则化

### 31.6.3 计算成本高

**RLHF的计算成本**：

| 阶段 | 模型数量 | 前向/反向传播 |
|------|----------|---------------|
| SFT | 1 | 标准训练 |
| RM训练 | 1-5（集成） | 标准训练 |
| PPO | 4（策略、ref、RM、value） | 多次前向+策略反向 |

**每个PPO步骤**：
1. 策略生成（前向）
2. 参考模型（前向）
3. 奖励模型（前向）
4. 价值模型（前向）
5. PPO更新（反向）

**成本对比**（7B模型）：
- SFT：~数千GPU小时
- Full RLHF：~数万GPU小时（10x+）

---

## 31.7 改进方向

### 31.7.1 Constitutional AI

**Anthropic的方法**：使用AI自身进行"宪法式"自我改进。

**流程**：

1. **自我批评**：模型生成回复，然后自己批评（是否有害/偏见）
2. **自我修订**：根据批评修改回复
3. **偏好生成**：AI评判哪个版本更好（而非人类）

**优势**：
- 减少人工标注
- 可扩展性强
- 明确编码价值观（"宪法"）

**示例宪法规则**：

```yaml
principles:
  - "选择更有帮助的回答"
  - "选择更诚实的回答"
  - "选择更无害的回答"
  - "避免刻板印象"
  - "尊重用户隐私"
```

### 31.7.2 RLAIF (RL from AI Feedback)

**核心思想**：用强大的AI模型（如GPT-4）替代人类标注员。

**流程**：

1. 使用小模型生成多个候选回复
2. GPT-4评判哪个更好
3. 用AI生成的偏好训练奖励模型
4. 标准RLHF流程

**优势**：
- **成本低**：API调用 << 人工标注
- **速度快**：自动化流程
- **一致性**：AI评判更稳定

**挑战**：
- AI可能有偏见
- 自我强化问题（AI评判AI生成的内容）

### 31.7.3 多轮 RLHF

**迭代改进**：

```
Round 1: SFT → RM1 → PPO1
Round 2: 从PPO1继续 → RM2（新偏好数据）→ PPO2
Round 3: ...
```

**好处**：
- 持续适应用户偏好变化
- 修复新发现的问题

**实践**：
- ChatGPT定期更新
- 收集真实用户反馈

---

## 总结

本章涵盖了RLHF（人类反馈强化学习）：

1. **动机**：对齐问题，ChatGPT成功案例
2. **三阶段**：SFT（监督微调）、RM（奖励模型）、PPO（强化学习）
3. **偏好数据**：成对比较、Bradley-Terry模型、质量控制
4. **奖励模型**：Transformer架构、偏好预测、奖励Hacking
5. **PPO微调**：KL惩罚、参考模型、价值函数
6. **挑战**：过拟合、模式崩溃、计算成本
7. **改进**：Constitutional AI、RLAIF、多轮RLHF

**关键要点**：
- RLHF是当前对齐大模型的主流方法
- 三阶段流程缺一不可
- KL惩罚防止过度偏离
- 奖励Hacking是主要风险
- 计算成本是实际瓶颈

**下一章**：第32章探讨**DPO**——绕过显式奖励模型的更简单方法。

---

## 参考文献

- Ouyang, L., et al. (2022). "Training language models to follow instructions with human feedback (InstructGPT)." *NeurIPS*.
- Christiano, P., et al. (2017). "Deep reinforcement learning from human preferences." *NIPS*.
- Bai, Y., et al. (2022). "Constitutional AI: Harmlessness from AI Feedback." *arXiv*.
- Stiennon, N., et al. (2020). "Learning to summarize with human feedback." *NeurIPS*.
- Ziegler, D. M., et al. (2019). "Fine-Tuning Language Models from Human Preferences." *arXiv*.

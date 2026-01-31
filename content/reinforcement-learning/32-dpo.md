---
title: "第32章：DPO与隐式奖励方法"
description: "直接偏好优化，绕过显式奖励模型，DPO变体与迭代改进"
date: "2026-01-30"
---

# 第32章：DPO与隐式奖励方法

## 32.1 DPO (Direct Preference Optimization)

### 32.1.1 绕过显式奖励模型

**DPO的核心创新**：直接从偏好数据优化策略，无需训练独立的奖励模型。

**RLHF的问题**：
1. **训练复杂**：需要3个阶段（SFT → RM → PPO）
2. **不稳定**：RM过拟合、PPO超参敏感
3. **计算昂贵**：每步需要4个模型前向传播
4. **奖励Hacking**：策略利用RM漏洞

**DPO的承诺**：
- ✅ 单阶段训练（类似SFT）
- ✅ 稳定收敛
- ✅ 计算高效（只需2个模 型）
- ✅ 理论保证（等价于RLHF最优解）

### 32.1.2 隐式奖励推导

**关键洞察**：奖励函数可以从最优策略**隐式恢复**。

**从Bradley-Terry模型开始**：

$$
P(y_w \succ y_l | x) = \frac{\exp(r^*(x, y_w))}{\exp(r^*(x, y_w)) + \exp(r^*(x, y_l))} = \sigma(r^*(x, y_w) - r^*(x, y_l))
$$

**RLHF的最优策略**（根据RL理论）：

$$
\pi^*(y | x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y | x) \exp\left( \frac{r^*(x, y)}{\beta} \right)
$$

其中：
- $Z(x)$ 是配分函数（partition function）
- $\beta$ 是KL惩罚系数
- $\pi_{\text{ref}}$ 是参考模型

**反解奖励**：

$$
r^*(x, y) = \beta \log \frac{\pi^*(y | x)}{\pi_{\text{ref}}(y | x)} + \beta \log Z(x)
$$

**代入Bradley-Terry**：

$$
\begin{aligned}
P(y_w \succ y_l | x) &= \sigma(r^*(x, y_w) - r^*(x, y_l)) \\
&= \sigma\left( \beta \log \frac{\pi^*(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi^*(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right)
\end{aligned}
$$

**配分函数抵消！** $Z(x)$ 在差值中消失。

### 32.1.3 Bradley-Terry 重参数化

**最终DPO目标**：

$$
P(y_w \succ y_l | x) = \sigma\left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right)
$$

**直观理解**：
- 当 $\pi_\theta(y_w) > \pi_{\text{ref}}(y_w)$ 且 $\pi_\theta(y_l) < \pi_{\text{ref}}(y_l)$，概率接近1✅
- 策略学会**增加**preferred回复的概率，**降低**rejected回复的概率

<div data-component="DPOvsRLHF"></div>

### 32.1.4 DPO 损失函数

**负对数似然**：

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} \right) \right]
$$

**完整实现**：

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def dpo_loss(
    policy_model,
    ref_model,
    prompts,
    chosen_responses,
    rejected_responses,
    tokenizer,
    beta=0.1
):
    """
    DPO损失函数
    
    Args:
        policy_model: 正在训练的策略模型
        ref_model: 冻结的参考模型（SFT模型）
        prompts: 提示列表
        chosen_responses: 优选回复列表
        rejected_responses: 拒绝回复列表
        tokenizer: Tokenizer
        beta: 温度参数（默认0.1）
    
    Returns:
        loss: DPO损失
        metrics: 训练指标
    """
    # 1. 构造完整文本
    chosen_texts = [f"{prompt}{response}" for prompt, response in zip(prompts, chosen_responses)]
    rejected_texts = [f"{prompt}{response}" for prompt, response in zip(prompts, rejected_responses)]
    
    # 2. Tokenize
    chosen_tokens = tokenizer(chosen_texts, return_tensors='pt', padding=True, truncation=True)
    rejected_tokens = tokenizer(rejected_texts, return_tensors='pt', padding=True, truncation=True)
    
    # 计算prompt长度（用于mask）
    prompt_tokens = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
    prompt_lengths = (prompt_tokens['attention_mask'].sum(dim=1) - 1).tolist()
    
    # 3. 策略模型的log概率
    policy_chosen_logps = compute_log_probs(
        policy_model,
        chosen_tokens['input_ids'],
        chosen_tokens['attention_mask'],
        prompt_lengths
    )
    
    policy_rejected_logps = compute_log_probs(
        policy_model,
        rejected_tokens['input_ids'],
        rejected_tokens['attention_mask'],
        prompt_lengths
    )
    
    # 4. 参考模型的log概率（冻结）
    with torch.no_grad():
        ref_chosen_logps = compute_log_probs(
            ref_model,
            chosen_tokens['input_ids'],
            chosen_tokens['attention_mask'],
            prompt_lengths
        )
        
        ref_rejected_logps = compute_log_probs(
            ref_model,
            rejected_tokens['input_ids'],
            rejected_tokens['attention_mask'],
            prompt_lengths
        )
    
    # 5. 计算隐式奖励差异
    policy_chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    policy_rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    
    reward_diff = policy_chosen_rewards - policy_rejected_rewards
    
    # 6. DPO损失
    loss = -F.logsigmoid(reward_diff).mean()
    
    # 7. 指标
    with torch.no_grad():
        # 准确率：chosen奖励 > rejected奖励
        accuracy = (reward_diff > 0).float().mean()
        
        # 隐式奖励
        chosen_rewards = policy_chosen_rewards.mean()
        rejected_rewards = policy_rejected_rewards.mean()
        
        # 奖励margin
        reward_margin = reward_diff.mean()
    
    metrics = {
        'loss': loss.item(),
        'accuracy': accuracy.item(),
        'chosen_reward': chosen_rewards.item(),
        'rejected_reward': rejected_rewards.item(),
        'reward_margin': reward_margin.item()
    }
    
    return loss, metrics


def compute_log_probs(model, input_ids, attention_mask, prompt_lengths):
    """
    计算序列的log概率（只计算回复部分）
    
    Args:
        model: 语言模型
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len)
        prompt_lengths: 每个样本的prompt长度列表
    
    Returns:
        log_probs: (batch,) 每个回复的总log概率
    """
    # 前向传播
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (batch, seq_len, vocab_size)
    
    # 获取每个token的log概率
    log_probs_all = F.log_softmax(logits, dim=-1)
    
    # 提取实际生成token的log概率
    batch_size = input_ids.size(0)
    seq_len = input_ids.size(1)
    
    # Shift：logits[t] 预测 input_ids[t+1]
    shifted_log_probs = log_probs_all[:, :-1, :]  # (batch, seq_len-1, vocab)
    shifted_input_ids = input_ids[:, 1:]  # (batch, seq_len-1)
    
    # 收集每个token的log概率
    token_log_probs = torch.gather(
        shifted_log_probs,
        dim=2,
        index=shifted_input_ids.unsqueeze(-1)
    ).squeeze(-1)  # (batch, seq_len-1)
    
    # Mask：只计算回复部分
    response_log_probs = []
    for i in range(batch_size):
        prompt_len = prompt_lengths[i]
        # 从prompt结束开始计算
        response_mask = torch.zeros(seq_len - 1, device=input_ids.device)
        response_mask[prompt_len:] = 1.0
        
        # 应用attention mask（去除padding）
        valid_mask = attention_mask[i, 1:].float()
        final_mask = response_mask * valid_mask
        
        # 计算回复的总log概率
        response_logp = (token_log_probs[i] * final_mask).sum()
        response_log_probs.append(response_logp)
    
    return torch.stack(response_log_probs)


# DPO训练循环
def train_dpo(
    model_name,
    preference_data,
    output_dir="./dpo_model",
    epochs=3,
    batch_size=4,
    learning_rate=5e-7,
    beta=0.1
):
    """
    DPO完整训练
    
    Args:
        model_name: 基础模型名称（SFT后的模型）
        preference_data: 偏好数据 [{"prompt": ..., "chosen": ..., "rejected": ...}, ...]
        output_dir: 输出目录
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率（通常比SFT小10-100倍）
        beta: DPO温度参数
    """
    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    policy_model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 冻结参考模型
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    # 优化器
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_model.to(device)
    ref_model.to(device)
    
    # 训练循环
    for epoch in range(epochs):
        policy_model.train()
        
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        # 批次迭代
        for i in range(0, len(preference_data), batch_size):
            batch = preference_data[i:i+batch_size]
            
            prompts = [item['prompt'] for item in batch]
            chosen = [item['chosen'] for item in batch]
            rejected = [item['rejected'] for item in batch]
            
            # 计算DPO损失
            loss, metrics = dpo_loss(
                policy_model,
                ref_model,
                prompts,
                chosen,
                rejected,
                tokenizer,
                beta=beta
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += metrics['loss']
            total_accuracy += metrics['accuracy']
            num_batches += 1
            
            if num_batches % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {num_batches}")
                print(f"  Loss: {metrics['loss']:.4f}")
                print(f"  Accuracy: {metrics['accuracy']:.2%}")
                print(f"  Reward Margin: {metrics['reward_margin']:.4f}")
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        print(f"\nEpoch {epoch+1}/{epochs} 完成")
        print(f"  平均损失: {avg_loss:.4f}")
        print(f"  平均准确率: {avg_accuracy:.2%}")
    
    # 保存模型
    policy_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nDPO模型已保存至 {output_dir}")
```

<div data-component="ImplicitRewardVisualization"></div>

---

## 32.2 DPO 优势

### 32.2.1 简化流程（无需 RM 和 PPO）

**对比**：

| 阶段 | RLHF | DPO |
|------|------|-----|
| 1 | SFT | SFT |
| 2 | 训练RM（~1天） | - |
| 3 | PPO（~3-7天） | **DPO训练（~1天）** |
| **总时间** | ~5-10天 | **~2天** |
| **模型数量** | 4（policy, ref, RM, value） | **2（policy, ref）** |
| **GPU需求** | 高（PPO需要同时加载4个模型） | **中（只需2个模型）** |

**代码对比**：

```python
# RLHF：复杂的3阶段流程
trainer_sft = SFTTrainer(...)
trainer_sft.train()  # 阶段1

trainer_rm = RewardTrainer(...)
trainer_rm.train()  # 阶段2

trainer_ppo = PPOTrainer(...)
for batch in dataloader:
    responses = trainer_ppo.generate(...)
    rewards = reward_model(...)
    trainer_ppo.step(...)  # 阶段3

# DPO：简单的单阶段
trainer_dpo = DPOTrainer(...)
trainer_dpo.train()  # 一步完成！
```

### 32.2.2 稳定性提升

**RLHF的不稳定性**：
- PPO对超参敏感（learning rate, clip_range, KL coef）
- 奖励模型可能发散
- 训练曲线震荡

**DPO的稳定性**：
- 类似监督学习
- 梯度稳定
- 收敛平滑

**实验对比**（InstructGPT复现）：

```
RLHF训练曲线：
Reward: 4.2 → 5.8 → 4.1 → 6.3 → 5.5 (震荡)

DPO训练曲线：
Margin: 0.5 → 1.2 → 1.8 → 2.1 → 2.3 (平滑上升)
```

### 32.2.3 计算效率

**前向传播次数对比**（每个训练步骤）：

| 方法 | 生成阶段 | 奖励计算 | 更新阶段 | **总计** |
|------|----------|----------|----------|----------|
| RLHF | policy × 1, ref × 1 | RM × 1, value × 1 | policy × 4（PPO epochs） | **~8次** |
| DPO | - | - | policy × 1, ref × 1 | **~2次** |

**GPU内存占用**（7B模型，FP16）：

- RLHF：~60GB（4个模型）
- DPO：~28GB（2个模型）

**训练速度**：
- RLHF：~1000 samples/hour
- DPO：~5000 samples/hour（**5x faster**）

---

## 32.3 DPO 变体

### 32.3.1 IPO (Identity Preference Optimization)

**问题**：DPO中的log比率可能导致数值不稳定。

**IPO改进**：使用平方损失而非log似然。

**损失函数**：

$$
\mathcal{L}_{\text{IPO}}(\theta) = \mathbb{E} \left[ \left( \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)} - \frac{1}{2\beta} \right)^2 \right]
$$

**优势**：
- 更稳定的梯度
- 不需要sigmoid
- 理论上更接近真实偏好分布

### 32.3.2 KTO (Kahneman-Tversky Optimization)

**动机**：收集成对比较成本高，能否只用二元标签（好/坏）？

**KTO损失**：

$$
\mathcal{L}_{\text{KTO}}(\theta) = \mathbb{E}_{(x, y, r)} \left[ \mathbb{1}_{r=1} \cdot l_{\text{good}}(\theta) + \mathbb{1}_{r=0} \cdot l_{\text{bad}}(\theta) \right]
$$

其中：
- $r \in \{0, 1\}$：二元标签（0=差，1=好）
- $l_{\text{good}}$：好回复的损失
- $l_{\text{bad}}$：坏回复的损失

**实现**：

```python
def kto_loss(policy_model, ref_model, x, y, label, beta=0.1):
    """
    KTO损失（Kahneman-Tversky Optimization）
    
    Args:
        label: 0（差）或 1（好）
    """
    # 计算log比率
    policy_logp = compute_log_prob(policy_model, x, y)
    ref_logp = compute_log_prob(ref_model, x, y)
    
    log_ratio = policy_logp - ref_logp
    
    if label == 1:  # 好回复
        # 鼓励增加概率
        loss = -F.logsigmoid(beta * log_ratio)
    else:  # 差回复
        # 鼓励降低概率
        loss = -F.logsigmoid(-beta * log_ratio)
    
    return loss
```

**优势**：
- 数据收集成本低（无需成对比较）
- 可扩展性强

### 32.3.3 SPIN (Self-Play Fine-Tuning)

**核心思想**：用模型自己生成的回复作为"rejected"样本。

**流程**：

1. 使用当前策略 $\pi_t$ 生成回复 $y_t$
2. 真实数据 $y^*$ 作为"chosen"，$y_t$ 作为"rejected"
3. DPO更新：$\pi_t \rightarrow \pi_{t+1}$
4. 重复

**自博弈损失**：

$$
\mathcal{L}_{\text{SPIN}}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}, y^* \sim p^*(·|x)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y^* | x)}{\pi_{\text{old}}(y^* | x)} - \beta \log \frac{\pi_\theta(\pi_{\text{old}}(x) | x)}{\pi_{\text{old}}(\pi_{\text{old}}(x) | x)} \right) \right]
$$
 
**优势**：
- 无需人工标注
- 迭代自我改进
- 类似 AlphaGo 的自博弈

**代码示例**：

```python
def spin_iteration(policy_model, ref_model, real_data, tokenizer, beta=0.1):
    """
    SPIN单次迭代
    
    Args:
        real_data: 真实数据 {"prompt": ..., "response": ...}
    """
    # 1. 使用当前策略生成回复
    generated_responses = []
    for item in real_data:
        prompt = item['prompt']
        
        inputs = tokenizer(prompt, return_tensors='pt')
        outputs = policy_model.generate(**inputs, max_length=256)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        generated_responses.append(response)
    
    # 2. 构造偏好对
    preference_pairs = []
    for item, gen_resp in zip(real_data, generated_responses):
        preference_pairs.append({
            'prompt': item['prompt'],
            'chosen': item['response'],  # 真实数据
            'rejected': gen_resp  # 模型生成
        })
    
    # 3. DPO训练
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-7)
    
    for batch in create_batches(preference_pairs, batch_size=4):
        loss, metrics = dpo_loss(
            policy_model,
            ref_model,
            [p['prompt'] for p in batch],
            [p['chosen'] for p in batch],
            [p['rejected'] for p in batch],
            tokenizer,
            beta=beta
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 4. 更新参考模型（可选）
    ref_model.load_state_dict(policy_model.state_dict())
    ref_model.eval()
    
    return policy_model, ref_model
```

<div data-component="IterativeDPOLoop"></div>

---

## 32.4 迭代 DPO

### 32.4.1 在线偏好收集

**问题**：离线数据集可能过时，无法反映当前策略的分布。

**解决方案**：在线收集偏好数据。

**流程**：

```python
for iteration in range(num_iterations):
    # 1. 使用当前策略生成回复样本
    prompts = sample_prompts(dataset)
    responses_A = policy_model.generate(prompts, temperature=1.0)
    responses_B = policy_model.generate(prompts, temperature=1.0)
    
    # 2. 人类/AI标注偏好
    preferences = annotate(prompts, responses_A, responses_B)
    
    # 3. DPO训练
    new_policy = dpo_train(policy_model, preferences)
    
    # 4. 更新策略
    policy_model = new_policy
```

### 32.4.2 自我改进循环

**迭代DPO**：多轮DPO训练，每轮更新参考模型。

**Round  $t$ 的目标**：

$$
\pi_{t+1} = \arg\max_\pi \ \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_t} \left[ \log \sigma \left( \beta \log \frac{\pi(y_w | x)}{\pi_t(y_w | x)} - \beta \log \frac{\pi(y_l | x)}{\pi_t(y_l | x)} \right) \right]
$$

其中 $\pi_t$ 是第 $t$ 轮的参考模型。

**实践建议**：
- 每2-3 epochs更新一次参考模型
- 逐渐降低学习率
- 监控KL散度避免崩溃

### 32.4.3 分布漂移控制

**问题**：随着迭代，策略分布可能偏离原始SFT分布，导致性能下降。

**KL约束**：

$$
\max_\pi \ \mathbb{E}_{...} [...] \quad \text{s.t.} \quad  \text{KL}(\pi \| \pi_{\text{SFT}}) \leq \epsilon
$$

**实现**：在DPO损失中加入KL惩罚项。

```python
def dpo_loss_with_kl_constraint(
    policy_model,
    ref_model,
    sft_model,  # 原始SFT模型
    prompts,
    chosen,
    rejected,
    tokenizer,
    beta=0.1,
    kl_coef=0.01  # KL约束系数
):
    # 标准DPO损失
    dpo_loss, metrics = dpo_loss(
        policy_model, ref_model, prompts, chosen, rejected, tokenizer, beta
    )
    
    # 计算与SFT的KL散度
    with torch.no_grad():
        sft_logps_chosen = compute_log_probs(sft_model, chosen_tokens, ...)
    
    policy_logps_chosen = compute_log_probs(policy_model, chosen_tokens, ...)
    
    kl_penalty = (policy_logps_chosen - sft_logps_chosen).mean()
    
    # 总损失
    total_loss = dpo_loss + kl_coef * kl_penalty
    
    return total_loss, metrics
```

<div data-component="DPOLossLandscape"></div>

---

## 32.5 理论分析

### 32.5.1 与 RLHF 的等价性

**定理**（Rafailov et al., 2023）：在最优情况下，DPO的解等价于RLHF的解。

**证明梗概**：

1. RLHF最优策略满足：
   $$
   \pi^* = \arg\max_\pi \ \mathbb{E}_{x,y} [r(x, y)] - \beta \cdot \text{KL}(\pi \| \pi_{\text{ref}})
   $$

2. 闭式解（根据变分推断）：
   $$
   \pi^*(y | x) \propto \pi_{\text{ref}}(y | x) \exp\left( \frac{r^*(x, y)}{\beta} \right)
   $$

3. 反解奖励并代入Bradley-Terry，得到DPO损失

**结论**：DPO直接优化RLHF的最优策略，绕过显式奖励模型。

### 32.5.2 收敛性保证

**定理**：在凸假设下，DPO损失收敛到全局最优。

**条件**：
- 策略空间足够丰富
- 偏好数据来自真实分布
- 学习率满足Robbins-Monro条件

**收敛速度**：$O(1/\sqrt{T})$（与标准SGD相同）

### 32.5.3 样本复杂度

**问题**：需要多少偏好对才能达到 $\epsilon$-最优？

**上界**（理论）：

$$
N = \tilde{O}\left( \frac{d}{\epsilon^2} \right)
$$

其中 $d$ 是策略参数维度。

**实践**：
- GPT-3.5：~50K 偏好对
- Llama-2：~100K 偏好对
- Mistral：~30K 偏好对（更高效）

**对比RLHF**：
- RLHF需要更多数据（因为RM需要单独训练）
- DPO样本效率更高

---

## 总结

本章涵盖了DPO（直接偏好优化）：

1. **核心思想**：绕过显式奖励模型，直接从偏好优化策略
2. **理论推导**：从Bradley-Terry模型推导隐式奖励
3. **DPO优势**：简化流程、稳定性、计算效率（5x faster）
4. **变体**：IPO（平方损失）、KTO（二元标签）、SPIN（自博弈）
5. **迭代DPO**：在线偏好收集、自我改进、分布漂移控制
6. **理论**：与RLHF等价、收敛性保证、样本复杂度

**关键要点**：
- DPO是RLHF的更简单、稳定的替代方案
- 隐式奖励避免了RM训练和奖励Hacking
- 迭代DPO可以持续改进
- 样本效率高于RLHF

**下一章**：第33章探讨**Reasoning-Time RL**——在推理时扩展计算的新范式。

---

## 参考文献

- Rafailov, R., et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." *NeurIPS*.
- Azar, M. G., et al. (2023). "A General Theoretical Paradigm to Understand Learning from Human Preferences." *ICML*.
- Chen, L., et al. (2024). "Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models (SPIN)." *arXiv*.
- Ethayarajh, K., et al. (2024). "KTO: Model Alignment as Prospect Theoretic Optimization." *arXiv*.
- Ye, Q., et al. (2024). "A Theoretical Analysis of Nash Learning from Human Feedback (IPO)." *arXiv*.

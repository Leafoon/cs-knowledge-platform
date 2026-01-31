---
title: "Chapter 12. Proximal Policy Optimization (PPO)"
description: "现代强化学习的主力算法"
updated: "2026-01-29"
---

> **Learning Objectives**
> * 理解 PPO 的动机与设计思想
> * 掌握 PPO-Clip 机制与悲观界
> * 学习 PPO 完整实现（多 epoch + mini-batch）
> * 了解 PPO 在实际应用中的成功案例
> * 掌握 PPO 调参技巧与工程实践

---

## 12.1 PPO 的动机

PPO 旨在保留 TRPO 的优点，同时大幅简化实现。

### 12.1.1 简化 TRPO

**TRPO 的问题**：
- ❌ 计算开销大（共轭梯度、line search）
- ❌ 实现复杂（~500 行代码）
- ❌ 难以调试

**PPO 的目标**：
- ✅ 保留单调改进的思想
- ✅ 简化为一阶优化（普通 SGD）
- ✅ 易于实现（~150 行核心代码）

### 12.1.2 保留单调改进

**核心思想**：即使没有严格的理论保证，也要在实践中保证稳定性。

**方法**：
1. **限制策略变化**（类似 TRPO 的 KL 约束）
2. **使用简单的裁剪**（clipping）代替复杂的二阶优化

### 12.1.3 易于实现

**PPO 只需**：
- 标准的梯度下降
- 简单的 clipping 操作
- 无需 Fisher矩阵、共轭梯度等

**结果**：代码量减少 70%，速度提升 3-5 倍。

---

## 12.2 PPO-Clip

PPO-Clip 是 PPO 的主要变体，使用巧妙的裁剪机制。

### 12.2.1 Clipped Surrogate Objective

**标准策略梯度目标**：

$$
L^{PG}(\boldsymbol{\theta}) = \mathbb{E}\left[\frac{\pi_{\boldsymbol{\theta}}(a|s)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a|s)} A^{\pi_{\boldsymbol{\theta}_{\text{old}}}}(s,a)\right]
$$

**PPO-Clip 目标**：

$$
L^{CLIP}(\boldsymbol{\theta}) = \mathbb{E}\left[\min(r_t(\boldsymbol{\theta}) A_t, \text{clip}(r_t(\boldsymbol{\theta}), 1-\epsilon, 1+\epsilon) A_t)\right]
$$

其中：
- $r_t(\boldsymbol{\theta}) = \frac{\pi_{\boldsymbol{\theta}}(a_t|s_t)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a_t|s_t)}$：概率比率
- $\epsilon$：裁剪范围（通常 0.1 或 0.2）

<div data-component="PPOClipMechanism"></div>

### 12.2.2 r_t(θ) = π_θ(a|s) / π_θ_old(a|s)

**概率比率**：

$$
r_t(\boldsymbol{\theta}) = \frac{\pi_{\boldsymbol{\theta}}(a_t|s_t)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a_t|s_t)}
$$

**含义**：
- $r_t = 1$：新旧策略相同
- $r_t > 1$：新策略更倾向选择该动作
- $r_t < 1$：新策略更不倾向选择该动作

**计算**（对数空间）：

$$
r_t = \exp(\log \pi_{\boldsymbol{\theta}}(a|s) - \log \pi_{\boldsymbol{\theta}_{\text{old}}}(a|s))
$$

### 12.2.3 clip(r_t, 1-ε, 1+ε)

**裁剪函数**：

$$
\text{clip}(r, 1-\epsilon, 1+\epsilon) = \begin{cases}
1-\epsilon, & r < 1-\epsilon \\
r, & 1-\epsilon \leq r \leq 1+\epsilon \\
1+\epsilon, & r > 1+\epsilon
\end{cases}
$$

**效果**：
- 限制 $r_t$ 在 $[1-\epsilon, 1+\epsilon]$ 范围内
- 防止策略变化过大

<div data-component="RatioClippingEffect"></div>

### 12.2.4 悲观界（Pessimistic Bound）

**PPO 目标的两项**：

$$
L^{CLIP} = \mathbb{E}\left[\min\left(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\right)\right]
$$

**分析**（假设 $A_t > 0$，即好动作）：

| $r_t$ 范围 | 未裁剪 | 裁剪后 | $\min$ |
|-----------|--------|--------|---------|
| $r_t < 1-\epsilon$ | $r_t A_t$ ↓ | $(1-\epsilon) A_t$ | $(1-\epsilon) A_t$ ✅ |
| $1-\epsilon \leq r_t \leq 1+\epsilon$ | $r_t A_t$ | $r_t A_t$ | $r_t A_t$ ✅ |
| $r_t > 1+\epsilon$ | $r_t A_t$ ↑ | $(1+\epsilon) A_t$ | $(1+\epsilon) A_t$ ✅ |

**关键洞察**：
- 当 $A_t > 0$（好动作）且 $r_t > 1+\epsilon$（过度增加概率）→ 裁剪到 $1+\epsilon$
- 当 $A_t < 0$（坏动作）且 $r_t < 1-\epsilon$（过度减少概率）→ 裁剪到 $1-\epsilon$
- **取 min**：选择保守的（pessimistic）估计

**图示**：

```
Advantage > 0 (好动作):
    ↑ 目标
    |     裁剪区域
    |   /‾‾‾‾‾‾‾‾‾
    |  /
    | /
    |/_____________→ r_t
   1-ε    1   1+ε

Advantage < 0 (坏动作):
    ↑ 目标
    |\
    | \
    |  \
    |   \__________  裁剪区域
    |____________→ r_t
   1-ε    1   1+ε
```

---

## 12.3 PPO-Penalty

PPO 的另一个变体，使用自适应 KL 惩罚。

### 12.3.1 自适应 KL 惩罚

**目标函数**：

$$
L^{KLPEN}(\boldsymbol{\theta}) = \mathbb{E}\left[\frac{\pi_{\boldsymbol{\theta}}(a|s)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a|s)} A(s,a) - \beta \cdot D_{KL}(\pi_{\boldsymbol{\theta}_{\text{old}}} \| \pi_{\boldsymbol{\theta}})\right]
$$

其中 $\beta$ 是惩罚系数。

### 12.3.2 动态调整系数

**算法**：

```python
if kl > kl_target * 1.5:
    beta *= 2  # KL 过大，增加惩罚
elif kl < kl_target / 1.5:
    beta /= 2  # KL 过小，减少惩罚
```

### 12.3.3 与 PPO-Clip 对比

| 特性 | PPO-Clip | PPO-Penalty |
|------|----------|-------------|
| **实现** | 更简单 | 需要调整 β |
| **性能** | 略优 | 相当 |
| **主流** | ✅ 更常用 | 较少使用 |
| **超参数** | ε (固定) | β (动态) |

**实践建议**：优先使用 **PPO-Clip**。

---

## 12.4 PPO 实现

PPO 的完整实现包含多 epoch 更新和 mini-batch SGD。

### 12.4.1 多 epoch 更新

**与 A2C 的区别**：
- **A2C**：收集数据 → 更新 1 次 → 丢弃数据
- **PPO**：收集数据 → **更新多次**（例如 10 epochs）→ 丢弃数据

**原因**：
- PPO 的 clipping 限制了策略变化
- 允许对同一批数据多次更新
- 提高样本效率

<div data-component="MultiEpochUpdate"></div>

### 12.4.2 Mini-batch SGD

**流程**：

```python
for epoch in range(ppo_epochs):  # 例如 10
    # 打乱数据
    indices = np.random.permutation(len(states))
    
    # Mini-batch 更新
    for start in range(0, len(states), mini_batch_size):
        end = start + mini_batch_size
        batch_indices = indices[start:end]
        
        # 更新策略和价值函数
        update(states[batch_indices], ...)
```

**Mini-batch 大小**：通常 64 或 128。

### 12.4.3 GAE 优势估计

PPO 通常使用 GAE 计算 advantage：

$$
A_t^{GAE(\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}
$$

**好处**：
- 平衡偏差和方差
- $\lambda = 0.95$ 是常用值

### 12.4.4 价值函数裁剪

**动机**：价值函数也可能有大的更新。

**裁剪**：

$$
V^{CLIP}(s) = V_{\text{old}}(s) + \text{clip}(V(s) - V_{\text{old}}(s), -\epsilon_v, \epsilon_v)
$$

**价值损失**：

$$
L^{V} = \mathbb{E}\left[\max((V(s) - V^{targ})^2, (V^{CLIP}(s) - V^{targ})^2)\right]
$$

**注意**：这是可选的，不是所有实现都使用。

---

## 12.5 PPO 变体与改进

### 12.5.1 PPO-Lagrangian

使用拉格朗日乘数法处理 KL 约束：

$$
\max_{\boldsymbol{\theta}} \min_{\lambda \geq 0} \mathbb{E}[L(\boldsymbol{\theta}) - \lambda (D_{KL} - d)]
$$

### 12.5.2 PPO with Auxiliary Tasks

**添加辅助任务**：
- 价值函数预测
- Dynamics 预测
- Inverse dynamics

**好处**：提高样本效率，学习更好的表示。

### 12.5.3 Recurrent PPO (R-PPO)

**使用 LSTM/GRU** 处理部分可观测环境（POMDP）。

**修改**：
- 在 trajectory 层面计算 GAE
- 保持隐藏状态的连续性

---

## 12.6 PPO 成功案例

PPO 是现代 RL 应用最广泛的算法。

### 12.6.1 OpenAI Five (Dota 2)

**任务**：5v5 Dota 2 游戏，击败人类职业选手。

**挑战**：
- 动作空间巨大（~20,000）
- 长期规划（45 分钟游戏）
- 部分可观测

**方法**：
- PPO + Transformer
- 大规模分布式训练（256 GPU + 128,000 CPU cores）
- 每天 900 年游戏经验

**结果**：2019 年击败世界冠军队伍 OG。

### 12.6.2 ChatGPT RLHF

**任务**：根据人类偏好对齐语言模型。

**方法**：
1. 训练 reward model（从人类标注）
2. 使用 PPO 优化语言模型
3. KL 惩罚防止偏离 base model

**PPO 的作用**：
- 稳定优化大型语言模型（175B 参数）
- 平衡 reward 最大化和 KL 约束

### 12.6.3 机器人控制

**应用**：
- 灵巧操作（Dexterity）
- 四足行走
- 人形机器人

**优势**：
- 样本效率高（相对其他 on-policy 方法）
- 稳定，容易调参
- 可扩展到高维动作空间

---

## 12.7 完整 PPO 实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import numpy as np

class PPOAgent:
    """
    PPO (Proximal Policy Optimization) Agent
    
    Args:
        state_dim: 状态维度
        action_dim: 动作数量
        num_envs: 并行环境数量
        hidden_dim: 隐藏层维度
        clip_epsilon: PPO clip 范围
        ppo_epochs: 每次更新的 epoch 数
        mini_batch_size: Mini-batch 大小
        gamma: 折扣因子
        lambda_gae: GAE 参数
        lr: 学习率
    """
    def __init__(self, state_dim, action_dim, num_envs=8, hidden_dim=64,
                 clip_epsilon=0.2, ppo_epochs=10, mini_batch_size=256,
                 gamma=0.99, lambda_gae=0.95, lr=3e-4, entropy_coef=0.01,
                 value_coef=0.5, max_grad_norm=0.5):
        
        self.num_envs = num_envs
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # 共享网络
        self.model = SharedActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    
    def select_actions(self, states):
        """选择动作并返回相关信息"""
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states)
            action_probs, values = self.model(states_tensor)
            
            dist = torch.distributions.Categorical(action_probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            
            return (actions.numpy(), 
                    log_probs.numpy(), 
                    values.squeeze().numpy())
    
    def compute_gae(self, rewards, values, next_value, dones):
        """计算 GAE"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_v = next_value
            else:
                next_v = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_v * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lambda_gae * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + values
        
        return advantages, returns
    
    def train_step(self, states, actions, old_log_probs, returns, advantages):
        """
        PPO 训练步骤
        
        包含：
        - 多 epoch 更新
        - Mini-batch SGD
        - PPO-Clip
        - 价值函数更新
        - 熵正则化
        """
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # 标准化 advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多 epoch 更新
        for epoch in range(self.ppo_epochs):
            # 打乱数据
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            # Mini-batch 更新
            for start in range(0, len(states), self.mini_batch_size):
                end = min(start + self.mini_batch_size, len(states))
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # 前向传播
                action_probs, values = self.model(batch_states)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO Clip loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                values = values.squeeze()
                critic_loss = nn.MSELoss()(values, batch_returns)
                
                # 总损失
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                
                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item()
        }


def train_ppo(env_name='CartPole-v1', num_envs=8, timesteps_per_batch=2048, 
              total_timesteps=1000000):
    """
    PPO 训练主循环
    
    Args:
        env_name: 环境名称
        num_envs: 并行环境数量
        timesteps_per_batch: 每个 batch 的时间步数
        total_timesteps: 总训练时间步数
    """
    # 创建并行环境
    def make_env():
        return lambda: gym.make(env_name)
    
    envs = AsyncVectorEnv([make_env() for _ in range(num_envs)])
    
    # 初始化 agent
    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.n
    agent = PPOAgent(state_dim, action_dim, num_envs=num_envs)
    
    # 训练循环
    states = envs.reset()[0]
    global_step = 0
    
    while global_step < total_timesteps:
        # 收集经验
        batch_states = []
        batch_actions = []
        batch_log_probs = []
        batch_values = []
        batch_rewards = []
        batch_dones = []
        
        for _ in range(timesteps_per_batch // num_envs):
            # 选择动作
            actions, log_probs, values = agent.select_actions(states)
            
            # 环境交互
            next_states, rewards, dones, truncated, infos = envs.step(actions)
            dones = np.logical_or(dones, truncated)
            
            # 存储
            batch_states.append(states)
            batch_actions.append(actions)
            batch_log_probs.append(log_probs)
            batch_values.append(values)
            batch_rewards.append(rewards)
            batch_dones.append(dones)
            
            states = next_states
            global_step += num_envs
        
        # 计算 next value
        with torch.no_grad():
            _, next_value = agent.model(torch.FloatTensor(states))
            next_value = next_value.squeeze().numpy()
        
        # 展平数据
        batch_states = np.array(batch_states).reshape(-1, state_dim)
        batch_actions = np.array(batch_actions).flatten()
        batch_log_probs = np.array(batch_log_probs).flatten()
        batch_values = np.array(batch_values).flatten()
        batch_rewards = np.array(batch_rewards)
        batch_dones = np.array(batch_dones)
        
        # 计算 advantages 和 returns
        all_advantages = []
        all_returns = []
        
        for env_idx in range(num_envs):
            env_rewards = batch_rewards[:, env_idx]
            env_values = batch_values.reshape(-1, num_envs)[:, env_idx]
            env_dones = batch_dones[:, env_idx]
            
            advantages, returns = agent.compute_gae(
                env_rewards, env_values, next_value[env_idx], env_dones
            )
            all_advantages.append(advantages)
            all_returns.append(returns)
        
        all_advantages = np.concatenate(all_advantages)
        all_returns = np.concatenate(all_returns)
        
        # 训练
        metrics = agent.train_step(
            batch_states, batch_actions, batch_log_probs, 
            all_returns, all_advantages
        )
        
        # 日志
        if global_step % 10000 == 0:
            print(f"Steps: {global_step}/{total_timesteps}")
            print(f"  Actor Loss: {metrics['actor_loss']:.3f}")
            print(f"  Critic Loss: {metrics['critic_loss']:.3f}")
            print(f"  Entropy: {metrics['entropy']:.3f}")
    
    envs.close()
    return agent

# 运行训练
if __name__ == "__main__":
    agent = train_ppo(env_name='CartPole-v1', total_timesteps=500000)
```

<div data-component="PPOvsTRPO"></div>

---

## 12.8 PPO 超参数调优

**关键超参数**：

| 超参数 | 典型值 | 说明 |
|--------|--------|------|
| clip_epsilon | 0.1 - 0.2 | Clip 范围，通常 0.2 |
| ppo_epochs | 3 - 10 | 每次数据更新次数，通常 10 |
| mini_batch_size | 64 - 256 | Mini-batch 大小|
| γ (gamma) | 0.99 | 折扣因子 |
| λ (lambda_gae) | 0.95 | GAE 参数 |
| lr | 3e-4 | 学习率 |
| entropy_coef | 0.01 | 熵系数 |
| value_coef | 0.5 | 价值损失系数 |
| max_grad_norm | 0.5 | 梯度裁剪 |

**调优建议**：
1. **开始**：使用默认值（epsilon=0.2, epochs=10）
2. **不稳定**：减小 epsilon（0.1）或学习率
3. **探索不足**：增大 entropy_coef（0.02）
4. **样本效率低**：增加 ppo_epochs（15）

---

## 本章小结

在本章中，我们学习了：

✅ **PPO 动机**：简化 TRPO，保留单调改进思想  
✅ **PPO-Clip**：巧妙的裁剪机制，限制策略变化  
✅ **多 epoch 更新**：提高样本效率  
✅ **成功案例**：OpenAI Five、ChatGPT RLHF、机器人控制  
✅ **工程实践**：完整实现与超参数调优  

> [!TIP]
> **核心要点**：
> - PPO-Clip 通过简单的 min(r·A, clip(r)·A) 限制策略变化
> - 多 epoch + mini-batch 提高样本效率
> - 是现代 RL 最实用的算法（简单、稳定、高效）
> - 广泛应用于游戏、机器人、LLM 对齐等领域
> - 默认超参数（epsilon=0.2, epochs=10）通常表现良好

> [!NOTE]
> **下一步**：
> Chapter 13 将学习**最大熵强化学习与 SAC**：
> - 最大熵框架
> - Soft Actor-Critic (SAC)
> - 自动温度调整
> - 应用于连续控制
> 
> 进入 [Chapter 13. SAC](13-sac.md)

---

## 扩展阅读

- **经典论文**：
  - Schulman et al. (2017): Proximal Policy Optimization Algorithms
  - OpenAI (2019): Dota 2 with Large Scale Deep RL
- **实现资源**：
  - Stable-Baselines3: PPO
  - OpenAI Spinning Up: PPO
  - CleanRL: PPO Implementation
- **应用案例**：
  - OpenAI Five
  - ChatGPT RLHF
  - DeepMind's AlphaStar (StarCraft II)

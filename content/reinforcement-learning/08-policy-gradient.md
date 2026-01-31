---
title: "Chapter 8. 策略梯度基础（Policy Gradient Foundations）"
description: "从价值函数到策略：直接优化策略的开端"
updated: "2026-01-29"
---

> **Learning Objectives**
> * 理解为什么需要策略梯度方法
> * 掌握策略梯度定理及其证明
> * 学习 REINFORCE 算法和 baseline 技术
> * 理解 Actor-Critic 架构
> * 掌握策略梯度的优缺点和适用场景

---

## 8.1 从价值到策略

策略梯度方法标志着强化学习的重要转变：从学习价值函数到直接优化策略。

### 8.1.1 为什么直接优化策略？

**价值函数方法（Q-learning, DQN）的流程**：

```
学习 Q(s,a) → 推导策略: π(s) = argmax_a Q(s,a)
```

**问题**：

1. **离散动作空间限制**：
   - DQN 需要枚举所有动作计算 $\max_a Q(s,a)$
   - 连续动作（机器人关节角度）无法枚举
   
2. **确定性策略**：
   - argmax 产生确定性策略
   - 某些任务需要随机策略（剪刀石头布、扑克）

3**不平滑的策略改进**：
   - Q 值微小变化可能导致策略突变
   - 训练不稳定

**策略梯度解决方案**：

```
直接参数化策略: π(a|s, θ) → 梯度上升优化 θ
```

<div data-component="PolicyGradientTheorem"></div>

### 8.1.2 策略参数化 π(a|s;θ)

**离散动作空间**：Softmax 策略

$$
\pi(a|s, \boldsymbol{\theta}) = \frac{\exp(h(s, a, \boldsymbol{\theta}))}{\sum_{a'} \exp(h(s, a', \boldsymbol{\theta}))}
$$

其中 $h(s, a, \boldsymbol{\theta})$ 是偏好函数（preference），可以是：
- 线性：$h(s, a, \boldsymbol{\theta}) = \boldsymbol{\theta}^T \phi(s, a)$
- 神经网络：$h(s, a, \boldsymbol{\theta}) = \text{NN}(s; \boldsymbol{\theta})_a$

**连续动作空间**：高斯策略

$$
\pi(a|s, \boldsymbol{\theta}) = \mathcal{N}(a; \mu(s, \boldsymbol{\theta}), \sigma^2)
$$

### 8.1.3 连续动作空间的优势

**示例任务**：
- 机器人控制：关节角度、扭矩
- 自动驾驶：转向角、加速度
- 金融交易：买卖数量

**策略梯度的优势**：
- ✅ 天然处理连续动作
- ✅ 输出动作分布（均值 + 方差）
- ✅ 自然地进行探索（采样）

**DQN 的困难**：
- ❌ 无法枚举无限动作
- ❌ 离散化损失精度
- ❌ 维度灾难（多个连续动作）

### 8.1.4 随机策略的必要性

**确定性策略问题**：

在某些任务中，确定性策略次优：

**示例 1：剪刀石头布**
- 最优策略：均匀随机（1/3, 1/3, 1/3）
- 确定性策略：可被对手利用

**示例 2：信息不完全游戏（扑克）**
- 需要混合策略（bluff）
- 确定性行为可预测

**示例 3：感知混淆（Aliasing）**
- 不同状态看起来相同
- 随机策略可以打破对称

**策略梯度自然支持随机策略**！

---

## 8.2 策略梯度定理

策略梯度定理提供了计算策略梯度的核心公式。

### 8.2.1 目标函数 J(θ)

**Episodic 情况**：

$$
J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} [G(\tau)] = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} \left[\sum_{t=0}^T r_t\right]
$$

其中轨迹 $\tau = (s_0, a_0, r_1, s_1, a_1, \ldots, s_T)$。

**Continuing 情况**：

$$
J(\boldsymbol{\theta}) = \mathbb{E}_{s \sim d^{\pi_{\boldsymbol{\theta}}}} [V^{\pi_{\boldsymbol{\theta}}}(s)]
$$

其中 $d^{\pi}(s)$ 是稳态分布。

**目标**：找到最优参数

$$
\boldsymbol{\theta}^* = \arg\max_{\boldsymbol{\theta}} J(\boldsymbol{\theta})
$$

### 8.2.2 策略梯度定理推导

**定理（Policy Gradient Theorem）**：

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim \pi_{\boldsymbol{\theta}}} \left[\sum_{t=0}^T \nabla_{\boldsymbol{\theta}} \log \pi(a_t|s_t, \boldsymbol{\theta}) \cdot G_t\right]
$$

或等价形式：

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{s, a \sim \pi} \left[\nabla_{\boldsymbol{\theta}} \log \pi(a|s, \boldsymbol{\theta}) \cdot Q^{\pi}(s, a)\right]
$$

**关键点**：
- 梯度**不依赖**环境动态 $P(s'|s,a)$（model-free）
- 只需对策略求梯度
- $\nabla \log \pi$ 称为 **score function**

**证明**（简化版）：

轨迹概率：

$$
P(\tau | \boldsymbol{\theta}) = \mu(s_0) \prod_{t=0}^{T-1} \pi(a_t|s_t, \boldsymbol{\theta}) P(s_{t+1}|s_t, a_t)
$$

目标函数：

$$
J(\boldsymbol{\theta}) = \int_{\tau} P(\tau | \boldsymbol{\theta}) G(\tau) d\tau
$$

求梯度：

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \int_{\tau} \nabla_{\boldsymbol{\theta}} P(\tau | \boldsymbol{\theta}) G(\tau) d\tau
$$

使用 **log-derivative trick**：

$$
\nabla_{\boldsymbol{\theta}} P(\tau | \boldsymbol{\theta}) = P(\tau | \boldsymbol{\theta}) \nabla_{\boldsymbol{\theta}} \log P(\tau | \boldsymbol{\theta})
$$

因此：

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\tau} [G(\tau) \nabla_{\boldsymbol{\theta}} \log P(\tau | \boldsymbol{\theta})]
$$

展开 $\log P(\tau | \boldsymbol{\theta})$：

$$
\log P(\tau | \boldsymbol{\theta}) = \log \mu(s_0) + \sum_{t=0}^{T-1} \log \pi(a_t|s_t, \boldsymbol{\theta}) + \sum_{t=0}^{T-1} \log P(s_{t+1}|s_t, a_t)
$$

求梯度（$\mu$ 和 $P$ 不依赖 $\boldsymbol{\theta}$）：

$$
\nabla_{\boldsymbol{\theta}} \log P(\tau | \boldsymbol{\theta}) = \sum_{t=0}^{T-1} \nabla_{\boldsymbol{\theta}} \log \pi(a_t|s_t, \boldsymbol{\theta})
$$

最终：

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\tau} \left[G(\tau) \sum_{t=0}^{T-1} \nabla_{\boldsymbol{\theta}} \log \pi(a_t|s_t, \boldsymbol{\theta})\right]
$$

**Reward-to-go 化简**：可以证明时刻 $t$ 的梯度只依赖 $t$ 之后的奖励：

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\tau} \left[\sum_{t=0}^{T-1} \nabla_{\boldsymbol{\theta}} \log \pi(a_t|s_t, \boldsymbol{\theta}) \cdot G_t\right]
$$

其中 $G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k$。

### 8.2.3 ∇J(θ) = E[∇logπ(a|s;θ) Q^π(s,a)]

**物理意义**：

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E} \left[\nabla_{\boldsymbol{\theta}} \log \pi(a|s, \boldsymbol{\theta}) \cdot Q^{\pi}(s, a)\right]
$$

**解读**：
- $\nabla \log \pi(a|s, \boldsymbol{\theta})$：增加动作 $a$ 概率的方向
- $Q^{\pi}(s, a)$：动作 $a$ 的好坏
- 乘积：**好的动作增加概率，坏的动作减少概率**

**直觉**：
- 如果 $Q(s,a) > 0$：沿 $\nabla \log \pi$ 方向移动 → 增加 $\pi(a|s)$
- 如果 $Q(s,a) < 0$：反方向移动 → 减少 $\pi(a|s)$

### 8.2.4 Score Function Estimator

**Score function**：

$$
\nabla_{\boldsymbol{\theta}} \log \pi(a|s, \boldsymbol{\theta})
$$

**Softmax 策略的 score function**：

$$
\nabla_{\boldsymbol{\theta}} \log \pi(a|s, \boldsymbol{\theta}) = \phi(s, a) - \mathbb{E}_{a' \sim \pi}[\phi(s, a')]
$$

**高斯策略的 score function**：

$$
\nabla_{\boldsymbol{\theta}} \log \pi(a|s, \boldsymbol{\theta}) = \frac{(a - \mu(s, \boldsymbol{\theta}))}{\sigma^2} \nabla_{\boldsymbol{\theta}} \mu(s, \boldsymbol{\theta})
$$

**PyTorch 自动计算**：

```python
action_dist = torch.distributions.Categorical(action_probs)
log_prob = action_dist.log_prob(action)
# 反向传播自动计算 ∇log_prob
```

---

## 8.3 REINFORCE 算法

REINFORCE（Williams, 1992）是最经典的蒙特卡洛策略梯度算法。

### 8.3.1 蒙特卡洛策略梯度

**核心思想**：使用完整 episode 的 return $G_t$ 估计 $Q^{\pi}(s_t, a_t)$。

**算法（REINFORCE）**：

```
初始化策略参数 θ

For episode = 1, 2, ...:
    生成 episode: s₀, a₀, r₁, s₁, a₁, r₂, ..., s_T
    
    For t = 0, 1, ..., T-1:
        计算 return: G_t = Σ_{k=t}^{T-1} γ^{k-t} r_{k+1}
        更新: θ ← θ + α γ^t G_t ∇_θ log π(a_t|s_t, θ)
```

**更新公式**：

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \gamma^t G_t \nabla_{\boldsymbol{\theta}} \log \pi(a_t|s_t, \boldsymbol{\theta})
$$

<div data-component="REINFORCEVariance"></div>

### 8.3.2 完整 episode 采样

**REINFORCE 特点**：

1. **On-policy**：必须用当前策略生成数据
2. **Monte Carlo**：使用完整 return（无 bootstrapping）
3. **Unbiased**：期望无偏
4. **High variance**：方差很大（需要 baseline）

### 8.3.3 高方差问题

**问题**：Return $G_t$ 的方差很大

**原因**：
- 长 episode → 累积大量随机性
- 奖励的随机性
- 策略的随机性

**后果**：
- 梯度估计不稳定
- 需要大量样本
- 学习缓慢

**解决方案**：
1. Baseline（下一节）
2. Advantage estimation
3. 减小学习率

### 8.3.4 伪代码与实现

**完整 PyTorch 实现**：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np

class PolicyNetwork(nn.Module):
    """策略网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.network(state)

def reinforce(env_name='CartPole-v1', num_episodes=1000, gamma=0.99, lr=0.01):
    """
    REINFORCE 算法
    
    Args:
        env_name: 环境名称
        num_episodes: 训练 episode 数量
        gamma: 折扣因子
        lr: 学习率
    
    Returns:
        policy_net: 训练好的策略网络
        episode_rewards: 每个 episode 的总回报
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy_net = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        # 生成 episode
        states = []
        actions = []
        rewards = []
        
        state = env.reset()[0]
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # 采样动作
            action_probs = policy_net(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            
            next_state, reward, done, truncated, _ = env.step(action.item())
            done = done or truncated
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
        
        # 计算 returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        
        # 标准化 returns（减小方差）
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # 策略梯度更新
        policy_loss = []
        for t in range(len(states)):
            state_tensor = torch.FloatTensor(states[t]).unsqueeze(0)
            action_probs = policy_net(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            log_prob = dist.log_prob(actions[t])
            
            # -log_prob * G_t（负号因为梯度下降）
            policy_loss.append(-log_prob * returns[t])
        
        # 反向传播
        optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        
        # 记录
        episode_reward = sum(rewards)
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")
    
    env.close()
    return policy_net, episode_rewards

# 使用示例
if __name__ == "__main__":
    policy, rewards = reinforce(env_name='CartPole-v1', num_episodes=1000)
```

---

## 8.4 Baseline 技术

Baseline 是降低策略梯度方差的关键技术。

### 8.4.1 方差缩减的必要性

**问题**：REINFORCE 的高方差导致：
- 学习不稳定
- 需要大量样本
- 收敛缓慢

**解决方案**：从 return 中减去 baseline $b(s_t)$：

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E} \left[\sum_{t=0}^{T-1} \nabla_{\boldsymbol{\theta}} \log \pi(a_t|s_t, \boldsymbol{\theta}) \cdot (G_t - b(s_t))\right]
$$

### 8.4.2 状态价值函数作为 baseline

**最常用 baseline**：状态价值函数 $V^{\pi}(s_t)$

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E} \left[\sum_{t=0}^{T-1} \nabla_{\boldsymbol{\theta}} \log \pi(a_t|s_t, \boldsymbol{\theta}) \cdot (G_t - V^{\pi}(s_t))\right]
$$

定义 **Advantage 函数**：

$$
A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s)
$$

则：

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E} \left[\nabla_{\boldsymbol{\theta}} \log \pi(a|s, \boldsymbol{\theta}) \cdot A^{\pi}(s, a)\right]
$$

**优势**：
- ✅ 显著降低方差
- ✅ 保持期望不变（无偏）
- ✅ 更快收敛

<div data-component="BaselineEffect"></div>

### 8.4.3 不改变期望的证明

**定理**：对于任意不依赖动作的 baseline $b(s)$：

$$
\mathbb{E}_{a \sim \pi} \left[\nabla_{\boldsymbol{\theta}} \log \pi(a|s, \boldsymbol{\theta}) \cdot b(s)\right] = 0
$$

**证明**：

$$
\begin{align}
&\mathbb{E}_{a \sim \pi} \left[\nabla_{\boldsymbol{\theta}} \log \pi(a|s, \boldsymbol{\theta}) \cdot b(s)\right] \\
&= b(s) \sum_a \pi(a|s, \boldsymbol{\theta}) \nabla_{\boldsymbol{\theta}} \log \pi(a|s, \boldsymbol{\theta}) \\
&= b(s) \sum_a \nabla_{\boldsymbol{\theta}} \pi(a|s, \boldsymbol{\theta}) \\
&= b(s) \nabla_{\boldsymbol{\theta}} \sum_a \pi(a|s, \boldsymbol{\theta}) \\
&= b(s) \nabla_{\boldsymbol{\theta}} 1 = 0
\end{align}
$$

因此，baseline **不改变期望，但降低方差**。

### 8.4.4 最优 baseline 选择

**问题**：什么是最优 baseline？

**定理**：最小化方差的 baseline为：

$$
b^*(s) = \frac{\mathbb{E}[G_t^2 \| \nabla \log \pi \|^2 | S_t = s]}{\mathbb{E}[\| \nabla \log \pi \|^2 | S_t = s]}
$$

**实践**：通常使用 $V^{\pi}(s)$ 已经很好，接近最优。

**REINFORCE with Baseline 实现**：

```python
class ValueNetwork(nn.Module):
    """价值网络（baseline）"""
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state).squeeze()

def reinforce_with_baseline(env_name='CartPole-v1', num_episodes=1000, 
                            gamma=0.99, lr_policy=0.01, lr_value=0.01):
    """REINFORCE with Baseline"""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy_net = PolicyNetwork(state_dim, action_dim)
    value_net = ValueNetwork(state_dim)
    
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr_policy)
    value_optimizer = optim.Adam(value_net.parameters(), lr=lr_value)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        states, actions, rewards = [], [], []
        
        state = env.reset()[0]
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            action_probs = policy_net(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            
            next_state, reward, done, truncated, _ = env.step(action.item())
            done = done or truncated
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
        
        # 计算 returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        states_tensor = torch.FloatTensor(states)
        
        # 计算 baseline（价值估计）
        values = value_net(states_tensor).detach()
        
        # 计算 advantage
        advantages = returns - values
        
        # 更新价值网络
        value_loss = nn.MSELoss()(value_net(states_tensor), returns)
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()
        
        # 更新策略网络
        policy_loss = []
        for t in range(len(states)):
            state_tensor = torch.FloatTensor(states[t]).unsqueeze(0)
            action_probs = policy_net(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            log_prob = dist.log_prob(actions[t])
            
            # 使用 advantage 而非 return
            policy_loss.append(-log_prob * advantages[t])
        
        policy_optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        policy_optimizer.step()
        
        episode_reward = sum(rewards)
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")
    
    env.close()
    return policy_net, value_net, episode_rewards
```

---

## 8.5 Actor-Critic 架构

Actor-Critic 结合了策略梯度和价值函数方法的优点。

### 8.5.1 Actor（策略网络）+ Critic（价值网络）

**架构**：

```
Actor:  π(a|s; θ)  - 策略网络，选择动作
Critic: V(s; w)    - 价值网络，评估状态
```

**分工**：
- **Actor**：学习策略，决定做什么
- **Critic**：学习价值函数，评估做得如何

**优势**：
- ✅ 降低方差（使用 critic）
- ✅ 在线学习（TD bootstrap）
- ✅ 样本效率更高

<div data-component="ActorCriticArchitecture"></div>

### 8.5.2 TD error 作为优势估计

**TD error**：

$$
\delta_t = r_t + \gamma V(s_{t+1}; \mathbf{w}) - V(s_t; \mathbf{w})
$$

**关键性质**：$\delta_t$ 是 $A^{\pi}(s_t, a_t)$ 的**无偏估计**！

**证明**：

$$
\begin{align}
\mathbb{E}[\delta_t] &= \mathbb{E}[r_t + \gamma V(s_{t+1}) - V(s_t)] \\
&= Q^{\pi}(s_t, a_t) - V^{\pi}(s_t) \\
&= A^{\pi}(s_t, a_t)
\end{align}
$$

**使用 TD error 更新策略**：

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \delta_t \nabla_{\boldsymbol{\theta}} \log \pi(a_t|s_t, \boldsymbol{\theta})
$$

### 8.5.3 同步更新策略与价值

**算法（Actor-Critic）**：

```
初始化 Actor π(a|s;θ) 和 Critic V(s;w)

For each episode:
    s = env.reset()
    
    While not done:
        # Actor: 选择动作
        a ~ π(·|s; θ)
        s', r = env.step(a)
        
        # Critic: 计算 TD error
        δ = r + γV(s';w) - V(s;w)
        
        # 更新 Critic
        w ← w + α_w δ ∇_w V(s;w)
        
        # 更新 Actor
        θ ← θ + α_θ δ ∇_θ log π(a|s;θ)
        
        s ← s'
```

**PyTorch 实现**：

```python
def actor_critic(env_name='CartPole-v1', num_episodes=1000,
                 gamma=0.99, lr_actor=0.01, lr_critic=0.01):
    """Actor-Critic 算法"""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    actor = PolicyNetwork(state_dim, action_dim)
    critic = ValueNetwork(state_dim)
    
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()[0]
        episode_reward = 0
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Actor: 选择动作
            action_probs = actor(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            
            next_state, reward, done, truncated, _ = env.step(action.item())
            done = done or truncated
            episode_reward += reward
            
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            
            # Critic: 计算 TD error
            value = critic(state_tensor)
            next_value = critic(next_state_tensor)
            td_target = reward + gamma * next_value * (1 - done)
            td_error = td_target - value
            
            # 更新 Critic
            critic_loss = td_error.pow(2)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            
            # 更新 Actor
            log_prob = dist.log_prob(action)
            actor_loss = -log_prob * td_error.detach()  # detach 避免影响 critic
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}")
    
    env.close()
    return actor, critic, episode_rewards
```

### 8.5.4 收敛性分析

**Theorem (Actor-Critic Convergence)**：

在适当条件下（两时间尺度学习率），Actor-Critic 收敛到局部最优。

**要求**：
- Critic 学习率 > Actor 学习率
- Critic 先收敛，然后 Actor 在稳定的 Critic 下优化

---

## 8.6 策略梯度的优缺点

### 8.6.1 优点：连续动作、随机策略、理论保证

**优点**：

1. ✅ **连续动作空间**：天然处理，无需离散化
2. ✅ **随机策略**：直接学习概率分布
3. ✅ **平滑更新**：策略改进连续平滑
4. ✅ **理论保证**：收敛性有理论支持
5. ✅ **高维动作**：扩展到多维连续动作

### 8.6.2 缺点：高方差、样本效率低、局部最优

**缺点**：

1. ❌ **高方差**：需要大量样本
2. ❌ **样本效率低**：On-policy，数据不能重复使用
3. ❌ **局部最优**：只保证收敛到局部最优
4. ❌ **超参数敏感**：学习率、网络架构等
5. ❌ **训练不稳定**：可能发散

### 8.6.3 适用场景

**适合策略梯度的任务**：
- 连续控制（机器人、自动驾驶）
- 高维动作空间
- 需要随机策略的任务

**不适合的任务**：
- 离散动作 + 样本受限（DQN 更好）
- 需要极高样本效率的任务

---

## 本章小结

在本章中，我们学习了：

✅ **策略梯度动机**：直接优化策略，处理连续动作  
✅ **策略梯度定理**：∇J(θ) = E[∇logπ Q]  
✅ **REINFORCE**：蒙特卡洛策略梯度算法  
✅ **Baseline**：降低方差的关键技术  
✅ **Actor-Critic**：结合策略和价值，实现在线学习  

> [!TIP]
> **核心要点**：
> - 策略梯度直接优化策略参数，适合连续动作
> - 策略梯度定理提供模型无关的梯度公式
> - REINFORCE 是无偏但高方差的
> - Baseline（如价值函数）显著降低方差
> - Actor-Critic 使用 TD error 实现在线学习

> [!NOTE]
> **下一步**：
> Chapter 9 将学习 **A2C/A3C**，进一步改进 Actor-Critic：
> - Advantage Function
> - 异步并行训练
> - 多步 TD
> - 为 PPO 打下基础
> 
> 进入 [Chapter 9. A2C/A3C](09-a2c-a3c.md)

---

## 扩展阅读

- **Sutton & Barto**：Chapter 13 (Policy Gradient Methods)
- **经典论文**：
  - Williams (1992): Simple Statistical Gradient-Following Algorithms
  - Sutton et al. (2000): Policy Gradient Methods for RL with Function Approximation
- **实现资源**：
  - OpenAI Spinning Up: Policy Gradients
  - Stable-Baselines3: A2C

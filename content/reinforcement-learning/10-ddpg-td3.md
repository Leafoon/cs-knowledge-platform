---
title: "Chapter 10. 确定性策略梯度（DDPG & TD3）"
description: "连续控制的利器：从 DDPG 到 TD3"
updated: "2026-01-29"
---

> **Learning Objectives**
> * 理解确定性策略与随机策略的区别
> * 掌握确定性策略梯度（DPG）定理
> * 学习 DDPG 算法及其实现
> * 理解 TD3 的三大改进
> * 掌握连续控制任务的实践技巧

---

## 10.1 确定性策略

确定性策略是连续动作空间的自然选择。

### 10.1.1 μ(s;θ) 而非 π(a|s;θ)

**随机策略**（Policy Gradient, A2C）：

$$
\pi(a|s; \boldsymbol{\theta}): \mathcal{S} \times \mathcal{A} \to [0,1]
$$

输出动作的**概率分布**。

**确定性策略**（DDPG, TD3）：

$$
\mu(s; \boldsymbol{\theta}): \mathcal{S} \to \mathcal{A}
$$

输出**确定的动作**。

**示例**：

```python
# 随机策略（高斯）
mean, std = policy_network(state)
action = torch.normal(mean, std)  # 采样

# 确定性策略
action = deterministic_policy_network(state)  # 直接输出
```

<div data-component="DeterministicPolicyVisualization"></div>

### 10.1.2 连续动作空间的优势

**为什么确定性策略适合连续动作？**

**问题**：随机策略的梯度估计

$$
\nabla_{\boldsymbol{\theta}} J = \mathbb{E}_{a \sim \pi}\left[\nabla_{\boldsymbol{\theta}} \log \pi(a|s; \boldsymbol{\theta}) \cdot Q(s, a)\right]
$$

在连续空间中，这个期望需要对**无限多**的动作积分/采样，方差极大。

**确定性策略的优势**：
- ✅ 无需对动作空间积分
- ✅ 梯度估计方差更低
- ✅ 样本效率更高
- ✅ 易于处理高维连续动作

**适用场景**：
- 机器人控制（关节角度、扭矩）
- 自动驾驶（转向角、加速度）
- 游戏（连续移动）

### 10.1.3 探索问题

**挑战**：确定性策略**没有内在探索**。

**解决方案**：添加**探索噪声**

$$
a_t = \mu(s_t; \boldsymbol{\theta}) + \mathcal{N}_t
$$

其中 $\mathcal{N}_t$ 是噪声（例如高斯噪声或 OU 噪声）。

---

## 10.2 DPG 定理

确定性策略梯度（Deterministic Policy Gradient）定理是 DDPG 的理论基础。

### 10.2.1 确定性策略梯度定理

**定理**（Silver et al., 2014）：

$$
\nabla_{\boldsymbol{\theta}} J(\mu_{\boldsymbol{\theta}}) = \mathbb{E}_{s \sim \rho^{\mu}} \left[\nabla_{\boldsymbol{\theta}} \mu(s; \boldsymbol{\theta}) \cdot \nabla_a Q^{\mu}(s, a)|_{a=\mu(s)}\right]
$$

**解释**：
- $\nabla_{\boldsymbol{\theta}} \mu(s; \boldsymbol{\theta})$：策略对参数的导数
- $\nabla_a Q^{\mu}(s, a)|_{a=\mu(s)}$：Q 函数对动作的梯度
- 链式法则：$\frac{\partial J}{\partial \boldsymbol{\theta}} = \frac{\partial \mu}{\partial \boldsymbol{\theta}} \cdot \frac{\partial Q}{\partial a}$

**直觉**：
1. 计算当前动作的 Q 值
2. 计算 Q 对动作的梯度（哪个方向会增加 Q？）
3. 沿这个方向调整策略

### 10.2.2 ∇J(θ) = E[∇_a Q(s,a)|_{a=μ(s)} ∇_θ μ(s;θ)]

**推导**（简化版）：

目标：

$$
J(\boldsymbol{\theta}) = \mathbb{E}_{s \sim \rho^{\mu}} [Q^{\mu}(s, \mu(s; \boldsymbol{\theta}))]
$$

对 $\boldsymbol{\theta}$ 求梯度：

$$
\begin{align}
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) &= \nabla_{\boldsymbol{\theta}} \mathbb{E}_{s} [Q^{\mu}(s, \mu(s; \boldsymbol{\theta}))] \\
&= \mathbb{E}_{s} [\nabla_{\boldsymbol{\theta}} Q^{\mu}(s, \mu(s; \boldsymbol{\theta}))] \\
&= \mathbb{E}_{s} [\nabla_a Q^{\mu}(s, a)|_{a=\mu(s)} \cdot \nabla_{\boldsymbol{\theta}} \mu(s; \boldsymbol{\theta})] \quad \text{(链式法则)}
\end{align}
$$

### 10.2.3 与随机策略梯度的关系

**随机策略梯度**：

$$
\nabla J = \mathbb{E}_{s, a} [\nabla \log \pi(a|s) \cdot Q(s, a)]
$$

**确定性策略梯度**：

$$
\nabla J = \mathbb{E}_{s} [\nabla_a Q(s, a)|_{a=\mu(s)} \cdot \nabla \mu(s)]
$$

**关系**：确定性策略是随机策略在方差 $\to 0$ 时的极限情况。

**优势对比**：

| 特性 | 随机策略梯度 | 确定性策略梯度 |
|------|------------|--------------|
| 动作空间 | 离散/连续 | 主要连续 |
| 探索 | 内在（采样） | 外在（噪声） |
| 方差 | 高 | 低 |
| 样本效率 | 低 | 高 |

---

## 10.3 DDPG 算法

DDPG（Deep Deterministic Policy Gradient）是DPG的深度学习版本。

### 10.3.1 Deep Deterministic Policy Gradient

**核心思想**：
- 结合 DPG 定理和 DQN 的技巧
- Actor（确定性策略）+ Critic（Q 函数）
- Experience Replay + Target Networks

**算法结构**：

```
Actor: μ(s; θ^μ)  →  确定性动作
Critic: Q(s, a; θ^Q)  →  Q 值估计
```

<div data-component="DDPGArchitecture"></div>

### 10.3.2 Actor-Critic 架构

**Actor 网络**（策略）：

$$
a = \mu(s; \boldsymbol{\theta}^{\mu})
$$

```python
class Actor(nn.Module):
    """
    Actor 网络（确定性策略）
    
    Args:
        state_dim: 状态维度
        action_dim: 动作维度
        action_bound: 动作边界
        hidden_dims: 隐藏层维度列表
    """
    def __init__(self, state_dim, action_dim, action_bound=1.0, hidden_dims=[256, 256]):
        super(Actor, self).__init__()
        
        self.action_bound = action_bound
        
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.output = nn.Linear(input_dim, action_dim)
    
    def forward(self, state):
        """
        输出确定性动作
        
        Returns:
            action: 范围在 [-action_bound, action_bound]
        """
        x = self.network(state)
        action = torch.tanh(self.output(x)) * self.action_bound
        return action
```

**Critic 网络**（Q 函数）：

$$
Q(s, a; \boldsymbol{\theta}^Q)
$$

```python
class Critic(nn.Module):
    """
    Critic 网络（Q 函数）
    
    Args:
        state_dim: 状态维度
        action_dim: 动作维度
        hidden_dims: 隐藏层维度列表
    """
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super(Critic, self).__init__()
        
        # 状态编码
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU()
        )
        
        # 状态+动作编码
        self.sa_encoder = nn.Sequential(
            nn.Linear(hidden_dims[0] + action_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )
    
    def forward(self, state, action):
        """
        计算 Q(s, a)
        
        Returns:
            q_value: Q 值
        """
        state_features = self.state_encoder(state)
        sa_features = torch.cat([state_features, action], dim=1)
        q_value = self.sa_encoder(sa_features)
        return q_value
```

### 10.3.3 Target Networks（软更新）

**动机**：与 DQN 相同，稳定训练目标。

**软更新**（Polyak Averaging）：

$$
\boldsymbol{\theta}^{\mu'} \leftarrow \tau \boldsymbol{\theta}^{\mu} + (1 - \tau) \boldsymbol{\theta}^{\mu'}
$$

$$
\boldsymbol{\theta}^{Q'} \leftarrow \tau \boldsymbol{\theta}^{Q} + (1 - \tau) \boldsymbol{\theta}^{Q'}
$$

其中 $\tau \ll 1$（例如 0.005）。

**代码**：

```python
def soft_update(target_net, source_net, tau=0.005):
    """
    软更新目标网络
    
    Args:
        target_net: 目标网络
        source_net: 源网络
        tau: 更新系数（小→更新慢）
    """
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )
```

**与 DQN 硬更新的对比**：

| 更新类型 | 公式 | 频率 | 稳定性 |
|---------|------|------|-------|
| 硬更新 | $\theta' \leftarrow \theta$ | 每 C 步 | 较差 |
| 软更新 | $\theta' \leftarrow \tau\theta + (1-\tau)\theta'$ | 每步 | 更好 |

### 10.3.4 Ornstein-Uhlenbeck 噪声

**目的**：为探索添加时间相关的噪声。

**OU 过程**：

$$
\mathcal{N}_t = \mathcal{N}_{t-1} + \theta (\mu_{\text{noise}} - \mathcal{N}_{t-1}) \Delta t + \sigma \sqrt{\Delta t} \mathcal{W}_t
$$

其中：
- $\theta$：回归速度（例如 0.15）
- $\mu_{\text{noise}}$：长期均值（通常 0）
- $\sigma$：波动性（例如 0.2）
- $\mathcal{W}_t$：维纳过程（标准正态）

<div data-component="OUNoiseProcess"></div>

**实现**：

```python
class OUNoise:
    """
    Ornstein-Uhlenbeck 噪声过程
    
    Args:
        action_dim: 动作维度
        mu: 长期均值
        theta: 回归速度
        sigma: 波动性
    """
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu
    
    def reset(self):
        """重置噪声状态"""
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self):
        """采样噪声"""
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state
```

**使用**：

```python
# 初始化
ou_noise = OUNoise(action_dim)

# 每个 episode 开始时重置
ou_noise.reset()

# 选择动作时添加噪声
action = actor(state) + ou_noise.sample()
action = np.clip(action, -action_bound, action_bound)
```

**现代实践**：简单的高斯噪声通常也能工作得很好：

```python
# 简化版：高斯噪声
noise = np.random.normal(0, sigma, size=action_dim)
action = actor(state) + noise
```

---

## 10.4 TD3 算法

TD3（Twin Delayed DDPG）是 DDPG 的改进版，解决了其高估偏差问题。

### 10.4.1 Twin Delayed DDPG

**动机**：DDPG 像 DQN 一样会**过度估计** Q 值。

**TD3 的三大改进**：
1. **Clipped Double Q-learning**：使用两个 Critic，取最小值
2. **Delayed Policy Updates**：延迟更新 Actor
3. **Target Policy Smoothing**：目标策略加噪声

<div data-component="TD3Improvements"></div>

### 10.4.2 Clipped Double Q-learning

**问题**：单个 Q 网络倾向于过度估计。

**解决方案**：使用**两个独立的** Critic，取**最小值**。

**两个 Critic**：

$$
Q_1(s, a; \boldsymbol{\theta}_1), \quad Q_2(s, a; \boldsymbol{\theta}_2)
$$

**目标计算**：

$$
y = r + \gamma \min_{i=1,2} Q_i'(s', \mu'(s'))
$$

而非 DDPG 的：

$$
y = r + \gamma Q'(s', \mu'(s'))
$$

**代码**：

```python
class TD3Agent:
    def __init__(self, state_dim, action_dim, action_bound):
        # Actor
        self.actor = Actor(state_dim, action_dim, action_bound)
        self.actor_target = Actor(state_dim, action_dim, action_bound)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # 两个 Critic
        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_2 = Critic(state_dim, action_dim)
        
        self.critic_1_target = Critic(state_dim, action_dim)
        self.critic_2_target = Critic(state_dim, action_dim)
        
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
    
    def compute_target_q(self, next_states, rewards, dones, gamma=0.99):
        """计算目标 Q 值（使用 min）"""
        with torch.no_grad():
            # 目标动作
            next_actions = self.actor_target(next_states)
            
            # 两个 Q 值
            target_q1 = self.critic_1_target(next_states, next_actions)
            target_q2 = self.critic_2_target(next_states, next_actions)
            
            # 取最小值（保守估计）
            target_q = torch.min(target_q1, target_q2)
            
            # 计算目标
            target_q = rewards + gamma * (1 - dones) * target_q
        
        return target_q
```

**效果**：显著降低过度估计，提高稳定性。

### 10.4.3 延迟策略更新

**问题**：频繁更新策略会导致不稳定。

**解决方案**：每 $d$ 次 Critic 更新才更新一次 Actor。

**典型值**：$d = 2$（每更新 2 次 Critic，更新 1 次 Actor）

**代码**：

```python
def train_step(self, batch, step, policy_delay=2):
    """
    训练步骤
    
    Args:
        batch: 经验 batch
        step: 当前步数
        policy_delay: 策略更新延迟
    """
    # 1. 总是更新 Critic
    critic_loss = self.update_critic(batch)
    
    # 2. 仅在特定步数更新 Actor
    if step % policy_delay == 0:
        actor_loss = self.update_actor(batch)
        
        # 软更新目标网络
        soft_update(self.actor_target, self.actor)
        soft_update(self.critic_1_target, self.critic_1)
        soft_update(self.critic_2_target, self.critic_2)
    else:
        actor_loss = None
    
    return critic_loss, actor_loss
```

**原理**：
- Critic 需要更频繁更新以准确估计 Q
- Actor 更新太频繁会被不准确的 Q 误导
- 延迟让 Critic 先收敛

### 10.4.4 目标策略平滑

**问题**：确定性目标策略可能过拟合特定动作。

**解决方案**：在目标策略上添加**裁剪的噪声**。

**公式**：

$$
\tilde{a} = \mu'(s') + \epsilon, \quad \epsilon \sim \text{clip}(\mathcal{N}(0, \sigma), -c, c)
$$

其中：
- $\sigma$：噪声标准差（例如 0.2）
- $c$：裁剪范围（例如 0.5）

**代码**：

```python
def compute_target_q_with_smoothing(self, next_states, rewards, dones, 
                                   gamma=0.99, policy_noise=0.2, noise_clip=0.5):
    """
    计算带平滑的目标 Q 值
    
    Args:
        policy_noise: 目标策略噪声标准差
        noise_clip: 噪声裁剪范围
    """
    with torch.no_grad():
        # 目标动作
        next_actions = self.actor_target(next_states)
        
        # 添加裁剪的噪声
        noise = torch.randn_like(next_actions) * policy_noise
        noise = torch.clamp(noise, -noise_clip, noise_clip)
        next_actions = next_actions + noise
        next_actions = torch.clamp(next_actions, -self.action_bound, self.action_bound)
        
        # 计算目标 Q（取最小值）
        target_q1 = self.critic_1_target(next_states, next_actions)
        target_q2 = self.critic_2_target(next_states, next_actions)
        target_q = torch.min(target_q1, target_q2)
        
        target_q = rewards + gamma * (1 - dones) * target_q
    
    return target_q
```

**效果**：平滑 Q 值表面，减少过拟合。

---

## 10.5 实现细节

### 10.5.1 经验回放

与 DQN 相同，DDPG/TD3 使用 Experience Replay Buffer。

```python
class ReplayBuffer:
    """经验回放缓冲区（连续动作版本）"""
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),  # 连续动作
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).unsqueeze(1)
        )
    
    def __len__(self):
        return len(self.buffer)
```

### 10.5.2 批归一化

**作用**：稳定不同尺度的输入特征。

```python
class ActorWithBN(nn.Module):
    """带批归一化的 Actor"""
    def __init__(self, state_dim, action_dim, action_bound=1.0):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.output = nn.Linear(256, action_dim)
        self.action_bound = action_bound
    
    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        action = torch.tanh(self.output(x)) * self.action_bound
        return action
```

**注意**：
- 训练时使用 batch statistics
- 评估时使用 running statistics
- 可能降低样本效率（需要更大 batch）

### 10.5.3 超参数敏感性

**DDPG/TD3 关键超参数**：

| 超参数 | 典型值 | 说明 |
|--------|--------|------|
| Actor LR | 1e-4 | Actor 学习率 |
| Critic LR | 1e-3 | Critic 学习率（通常更大） |
| γ (gamma) | 0.99 | 折扣因子 |
| τ (tau) | 0.005 | 软更新系数 |
| Buffer Size | 1M | 经验回放容量 |
| Batch Size | 256 | 训练 batch 大小 |
| Exploration Noise | 0.1 | 探索噪声标准差 |
| Policy Delay (TD3) | 2 | 策略更新延迟 |
| Policy Noise (TD3) | 0.2 | 目标策略噪声 |
| Noise Clip (TD3) | 0.5 | 噪声裁剪范围 |

**调参建议**：
1. 从推荐值开始
2. Critic LR 通常是 Actor LR 的 10 倍
3. $\tau$ 越小越稳定但学习越慢
4. TD3 的三个参数通常不需调整

---

## 10.6 完整 TD3 实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from collections import deque
import random

class TD3:
    """
    TD3 (Twin Delayed DDPG) Agent
    
    Args:
        state_dim: 状态维度
        action_dim: 动作维度
        action_bound: 动作边界
        gamma: 折扣因子
        tau: 软更新系数
        policy_noise: 目标策略噪声
        noise_clip: 噪声裁剪
        policy_delay: 策略更新延迟
        actor_lr: Actor 学习率
        critic_lr: Critic 学习率
    """
    def __init__(self, state_dim, action_dim, action_bound=1.0,
                 gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5,
                 policy_delay=2, actor_lr=1e-4, critic_lr=1e-3):
        
        self.action_bound = action_bound
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        
        # Actor
        self.actor = Actor(state_dim, action_dim, action_bound)
        self.actor_target = Actor(state_dim, action_dim, action_bound)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Twin Critics
        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_2 = Critic(state_dim, action_dim)
        self.critic_1_target = Critic(state_dim, action_dim)
        self.critic_2_target = Critic(state_dim, action_dim)
        
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=1000000)
        
        # Training step counter
        self.total_it = 0
    
    def select_action(self, state, add_noise=False, noise_scale=0.1):
        """
        选择动作
        
        Args:
            state: 状态
            add_noise: 是否添加探索噪声
            noise_scale: 噪声尺度
        
        Returns:
            action: 选择的动作
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy()[0]
        
        if add_noise:
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = action + noise
        
        action = np.clip(action, -self.action_bound, self.action_bound)
        return action
    
    def train_step(self, batch_size=256):
        """
        训练步骤
        
        Args:
            batch_size: Batch 大小
        
        Returns:
            metrics: 训练指标
        """
        self.total_it += 1
        
        # 采样 batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # ============ 更新 Critic ============
        with torch.no_grad():
            # 目标动作（带噪声）
            noise = torch.randn_like(actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_actions = self.actor_target(next_states) + noise
            next_actions = torch.clamp(next_actions, -self.action_bound, self.action_bound)
            
            # 计算目标 Q 值（取最小值）
            target_q1 = self.critic_1_target(next_states, next_actions)
            target_q2 = self.critic_2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * (1 - dones) * target_q
        
        # 当前 Q 值
        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)
        
        # Critic 损失
        critic_1_loss = nn.MSELoss()(current_q1, target_q)
        critic_2_loss = nn.MSELoss()(current_q2, target_q)
        
        # 更新 Critic 1
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        
        # 更新 Critic 2
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        
        # ============ 延迟更新 Actor ============
        actor_loss = None
        if self.total_it % self.policy_delay == 0:
            # Actor 损失（最大化 Q 值）
            actor_loss = -self.critic_1(states, self.actor(states)).mean()
            
            # 更新 Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 软更新目标网络
            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic_1_target, self.critic_1)
            self.soft_update(self.critic_2_target, self.critic_2)
        
        return {
            'critic_1_loss': critic_1_loss.item(),
            'critic_2_loss': critic_2_loss.item(),
            'actor_loss': actor_loss.item() if actor_loss is not None else None
        }
    
    def soft_update(self, target, source):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

def train_td3(env_name='Pendulum-v1', total_timesteps=1000000, 
              start_timesteps=25000, batch_size=256, eval_freq=5000):
    """
    TD3 训练主循环
    
    Args:
        env_name: 环境名称
        total_timesteps: 总训练步数
        start_timesteps: 开始训练前的随机探索步数
        batch_size: Batch 大小
        eval_freq: 评估频率
    """
    # 创建环境
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = float(env.action_space.high[0])
    
    # 创建 agent
    agent = TD3(state_dim, action_dim, action_bound)
    
    # 训练循环
    state = env.reset()[0]
    episode_reward = 0
    episode_num = 0
    
    for t in range(total_timesteps):
        # 选择动作
        if t < start_timesteps:
            # 初期随机探索
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, add_noise=True, noise_scale=0.1)
        
        # 执行动作
        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        
        # 存储到 buffer
        agent.replay_buffer.push(state, action, reward, next_state, float(done))
        
        state = next_state
        episode_reward += reward
        
        # 训练
        if t >= start_timesteps:
            metrics = agent.train_step(batch_size)
        
        # Episode 结束
        if done:
            print(f"Total T: {t+1}, Episode {episode_num+1}, Reward: {episode_reward:.2f}")
            state = env.reset()[0]
            episode_reward = 0
            episode_num += 1
        
        # 评估
        if (t + 1) % eval_freq == 0:
            eval_rewards = []
            for _ in range(10):
                eval_state = eval_env.reset()[0]
                eval_reward = 0
                eval_done = False
                
                while not eval_done:
                    eval_action = agent.select_action(eval_state, add_noise=False)
                    eval_state, r, eval_done, truncated, _ = eval_env.step(eval_action)
                    eval_done = eval_done or truncated
                    eval_reward += r
                
                eval_rewards.append(eval_reward)
            
            print(f"Evaluation over 10 episodes: {np.mean(eval_rewards):.2f}")
    
    return agent

# 运行训练
if __name__ == "__main__":
    agent = train_td3(env_name='Pendulum-v1', total_timesteps=100000)
```

---

## 本章小结

在本章中，我们学习了：

✅ **确定性策略**：μ(s) 直接输出动作，适合连续控制  
✅ **DPG 定理**：∇J = E[∇_a Q · ∇_θ μ]  
✅ **DDPG**：DPG + Deep Learning + DQN 技巧  
✅ **TD3 三大改进**：Twin Critics, Delayed Update, Target Smoothing  
✅ **实现细节**：OU 噪声、软更新、批归一化  

> [!TIP]
> **核心要点**：
> - 确定性策略消除了对动作空间的积分，方差更低
> - DDPG 结合 Actor-Critic 和 DQN 的经验回放
> - TD3 的 Clipped Double Q-learning 显著降低过度估计
> - 延迟策略更新让 Critic 先收敛
> - 目标策略平滑防止过拟合

> [!NOTE]
> **下一步**：
> Chapter 11 将学习**近端策略优化（PPO）**：
> - 信赖域方法
> - PPO-Clip 机制
> - 现代 RL 的主力算法
> - 在线策略优化的突破
> 
> 进入 [Chapter 11. PPO](11-ppo.md)

---

## 扩展阅读

- **经典论文**：
  - Silver et al. (2014): Deterministic Policy Gradient Algorithms
  - Lillicrap et al. (2016): Continuous control with deep RL (DDPG)
  - Fujimoto et al. (2018): Addressing Function Approximation Error (TD3)
- **实现资源**：
  - Stable-Baselines3: TD3
  - OpenAI Spinning Up: DDPG/TD3
- **应用案例**：
  - MuJoCo 连续控制任务
  - 机器人操作
  - 自动驾驶控制

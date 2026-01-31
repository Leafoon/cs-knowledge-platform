---
title: "Chapter 7. Deep Q-Network (DQN)"
description: "深度强化学习的突破：从 Atari 到人类水平"
updated: "2026-01-29"
---

> **Learning Objectives**
> * 理解 DQN 的历史意义和核心创新
> * 掌握 Experience Replay 和 Target Network 机制
> * 学习 DQN 的各种改进变体（Double, Dueling, Prioritized, Rainbow）
> * 能够实现完整的 DQN 算法
> * 理解 DQN 的局限性和适用场景

---

## 7.1 DQN 的诞生

Deep Q-Network (DQN) 是深度强化学习历史上的里程碑，标志着 RL 进入深度学习时代。

### 7.1.1 Atari 游戏挑战

**背景**：2013年之前，RL 主要局限于：
- 低维状态空间（表格方法）
- 手工特征工程（线性函数逼近）
- 简单任务（CartPole, Mountain Car）

**Atari 2600 挑战**：
- **高维输入**：84×84×4 RGB 图像（~28,000维）
- **多样性**：49个不同游戏，每个有独特规则
- **人类基准**：需要达到人类玩家水平
- **端到端学习**：直接从像素到动作

**为什么困难**：
1. 状态空间 ≈ 256^(84×84×4) 近乎无限
2. 强化学习固有的不稳定性
3. 样本相关性问题
4. 延迟奖励和稀疏奖励

### 7.1.2 端到端学习的意义

**传统方法**：
```
原始像素 → [手工特征提取] → 特征向量 → Q-learning → 动作
           (人工设计)
```

**DQN 方法**：
```
原始像素 → [卷积神经网络] → 动作价值 Q(s,a) → 动作
           (端到端学习)
```

**优势**：
- ✅ 无需手工特征工程
- ✅ 自动学习视觉表示
- ✅ 可迁移到不同任务
- ✅ 随数据量增长而改进

### 7.1.3 Nature DQN (Mnih et al., 2015)

**里程碑成果**：
- 在 49 个 Atari 游戏中，**29 个超过人类水平**
- 使用**相同**网络架构和超参数
- 没有游戏特定的调整

**关键创新**：
1. **Experience Replay**：打破样本相关性
2. **Target Network**：稳定训练目标
3. **卷积网络**：端到端特征学习
4. **帧堆叠**：捕捉时间信息

**影响**：
- 开启深度强化学习时代
- AlphaGo (2016) 的基础
- 推动 RL 在实际应用中的部署

<div data-component="DQNArchitecture"></div>

---

## 7.2 DQN 核心机制

DQN 的成功源于两个关键创新：Experience Replay 和 Target Network。

### 7.2.1 Experience Replay Buffer

**动机**：在线 RL 的问题

在标准 Q-learning 中，每个经验 $(s, a, r, s')$ 只使用一次：
```python
# 标准 Q-learning（不稳定）
s, a, r, s' = env.step(a)
Q[s][a] += alpha * (r + gamma * max(Q[s']) - Q[s][a])  # 使用一次即丢弃
```

**问题**：
1. **样本效率低**：每个经验只用一次
2. **强相关性**：连续样本高度相关（违反 i.i.d. 假设）
3. **灾难性遗忘**：新经验覆盖旧知识

**Experience Replay 解决方案**：

将经验存储在 Replay Buffer 中，训练时随机采样：

```python
class ReplayBuffer:
    """
    经验回放缓冲区
    
    Args:
        capacity: 缓冲区最大容量（例如 1,000,000）
    """
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        添加一个转移到缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否终止
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        随机采样一个 batch
        
        Args:
            batch_size: 批大小（例如 32 或 64）
        
        Returns:
            batch: (states, actions, rewards, next_states, dones) 的 tuple
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)
```

**优势**：
1. ✅ **打破相关性**：随机采样使样本近似 i.i.d.
2. ✅ **提高样本效率**：每个经验可重复使用多次
3. ✅ **平滑学习**：减少震荡和发散
4. ✅ **Off-policy**：可以从旧策略的数据学习

<div data-component="ExperienceReplayVisualizer"></div>

### 7.2.2 Target Network

**动机**：Q-learning 的不稳定性

在 Q-learning 中，TD 目标依赖于 Q 网络本身：

$$
y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \boldsymbol{\theta})
$$

**问题**：
- 目标随着 $\boldsymbol{\theta}$ 更新而变化（**移动目标**）
- 类似于"追逐自己的尾巴"
- 导致振荡和发散

**Target Network 解决方案**：

使用**独立的目标网络** $Q(s, a; \boldsymbol{\theta}^-)$ 计算目标：

$$
y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \boldsymbol{\theta}^-)
$$

其中 $\boldsymbol{\theta}^-$ 是目标网络参数，**定期**从主网络复制：

```python
# 主网络（在线网络）
q_network = DQN(state_dim, action_dim)

# 目标网络（固定参数）
target_network = DQN(state_dim, action_dim)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()  # 设置为评估模式

# 训练循环
for step in range(total_steps):
    # 使用主网络选择动作
    action = select_action(q_network, state)
    
    # ...收集经验...
    
    # 计算 TD 目标（使用目标网络）
    with torch.no_grad():
        max_next_q = target_network(next_state).max(1)[0]
        td_target = reward + gamma * max_next_q * (1 - done)
    
    # 更新主网络
    current_q = q_network(state).gather(1, action.unsqueeze(1))
    loss = F.mse_loss(current_q.squeeze(), td_target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 定期更新目标网络（例如每 10,000 步）
    if step % target_update_freq == 0:
        target_network.load_state_dict(q_network.state_dict())
```

**更新策略**：
1. **硬更新**（Hard Update）：定期完全复制
   ```python
   if step % C == 0:
       target_net.load_state_dict(q_net.state_dict())
   ```

2. **软更新**（Soft Update / Polyak Averaging）：指数移动平均
   ```python
   for target_param, param in zip(target_net.parameters(), q_net.parameters()):
       target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
   ```
   其中 $\tau \ll 1$（例如 0.001）

**优势**：
- ✅ **稳定训练**：目标在一段时间内固定
- ✅ **减少振荡**：平滑目标变化
- ✅ **提高收敛性**：更稳定的梯度

<div data-component="TargetNetworkUpdate"></div>

### 7.2.3 损失函数

**TD 误差**：

$$
\delta_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \boldsymbol{\theta}^-) - Q(s_t, a_t; \boldsymbol{\theta})
$$

**损失函数**：

$$
\mathcal{L}(\boldsymbol{\theta}) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \boldsymbol{\theta}^-) - Q(s, a; \boldsymbol{\theta}) \right)^2 \right]
$$

**Huber Loss（更稳健）**：

为了处理异常值，实践中常用 Huber Loss：

$$
\mathcal{L}_{\delta}(x) = \begin{cases}
\frac{1}{2} x^2 & \text{if } |x| \leq \delta \\
\delta (|x| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}
$$

```python
import torch.nn.functional as F

# MSE Loss（标准）
loss = F.mse_loss(current_q, td_target)

# Huber Loss（更稳健）
loss = F.smooth_l1_loss(current_q, td_target)  # δ=1 的 Huber Loss
```

### 7.2.4 ε-greedy 探索

**探索策略**：

$$
a_t = \begin{cases}
\arg\max_{a} Q(s_t, a; \boldsymbol{\theta}) & \text{with probability } 1 - \epsilon \\
\text{random action} & \text{with probability } \epsilon
\end{cases}
$$

**ε 衰减策略**：

```python
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995  # 或线性衰减

# 指数衰减
epsilon = max(epsilon_end, epsilon * epsilon_decay)

# 线性衰减
epsilon = epsilon_start - (epsilon_start - epsilon_end) * (step / total_steps)
```

**Nature DQN 设置**：
- 初始：$\epsilon = 1.0$（纯探索）
- 最终：$\epsilon = 0.1$（10% 探索）
- 衰减：前 1M 帧线性衰减

---

## 7.3 DQN 算法详解

### 7.3.1 完整伪代码

**算法（DQN）**：

```
初始化:
    Replay buffer D (capacity M)
    Q-network with random weights θ
    Target network with weights θ⁻ = θ
    
For episode = 1, 2, ..., N:
    初始化状态 s₁
    
    For t = 1, 2, ..., T:
        # 选择动作（ε-greedy）
        With probability ε:
            a_t = random action
        Otherwise:
            a_t = argmax_a Q(s_t, a; θ)
        
        # 执行动作并观察
        执行 a_t，观察 r_t, s_{t+1}, done
        存储 (s_t, a_t, r_t, s_{t+1}, done) 到 D
        
        # 训练
        If |D| >= batch_size:
            采样 minibatch {(s_j, a_j, r_j, s_j', done_j)} from D
            
            # 计算目标
            For j in batch:
                if done_j:
                    y_j = r_j
                else:
                    y_j = r_j + γ max_a' Q(s_j', a'; θ⁻)
            
            # 梯度下降
            L(θ) = 1/batch_size Σ_j (y_j - Q(s_j, a_j; θ))²
            θ ← θ - α ∇_θ L(θ)
        
        # 更新目标网络
        Every C steps:
            θ⁻ ← θ
        
        s_t ← s_{t+1}
```

### 7.3.2 完整 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    """
    Deep Q-Network
    
    Args:
        state_dim: 状态维度
        action_dim: 动作数量
        hidden_dim: 隐藏层维度
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        """
        前向传播
        
        Args:
            state: Tensor of shape (batch_size, state_dim)
        
        Returns:
            q_values: Tensor of shape (batch_size, action_dim)
        """
        return self.network(state)

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

def train_dqn(env_name='CartPole-v1', 
              num_episodes=1000,
              gamma=0.99,
              epsilon_start=1.0,
              epsilon_end=0.01,
              epsilon_decay=0.995,
              learning_rate=1e-3,
              batch_size=64,
              target_update_freq=10,
              buffer_capacity=100000):
    """
    DQN 训练函数
    
    Args:
        env_name: Gym 环境名称
        num_episodes: 训练 episode 数量
        gamma: 折扣因子
        epsilon_start: 初始探索率
        epsilon_end: 最终探索率
        epsilon_decay: ε 衰减率
        learning_rate: 学习率
        batch_size: 批大小
        target_update_freq: 目标网络更新频率（episodes）
        buffer_capacity: Replay buffer 容量
    
    Returns:
        q_network: 训练好的 Q 网络
        episode_rewards: 每个 episode 的总回报
    """
    # 创建环境
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 初始化网络
    q_network = DQN(state_dim, action_dim)
    target_network = DQN(state_dim, action_dim)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()
    
    # 优化器
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    
    # Replay buffer
    replay_buffer = ReplayBuffer(buffer_capacity)
    
    # 记录
    episode_rewards = []
    epsilon = epsilon_start
    
    for episode in range(num_episodes):
        state = env.reset()[0]
        episode_reward = 0
        done = False
        
        while not done:
            # ε-greedy 选择动作
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = q_network(state_tensor)
                    action = q_values.argmax().item()
            
            # 执行动作
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            episode_reward += reward
            
            # 存储到 replay buffer
            replay_buffer.push(state, action, reward, next_state, float(done))
            
            # 训练
            if len(replay_buffer) >= batch_size:
                # 采样 batch
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # 计算当前 Q 值
                current_q = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # 计算目标 Q 值（使用目标网络）
                with torch.no_grad():
                    max_next_q = target_network(next_states).max(1)[0]
                    target_q = rewards + gamma * max_next_q * (1 - dones)
                
                # 计算损失
                loss = F.mse_loss(current_q, target_q)
                
                # 优化
                optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪（防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10.0)
                optimizer.step()
            
            state = next_state
        
        # 衰减 epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # 更新目标网络
        if (episode + 1) % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())
        
        # 记录
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Avg Reward: {avg_reward:.2f}, "
                  f"Epsilon: {epsilon:.3f}")
    
    env.close()
    return q_network, episode_rewards

# 使用示例
if __name__ == "__main__":
    q_net, rewards = train_dqn(env_name='CartPole-v1', num_episodes=1000)
    
    # 绘制学习曲线
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.3, label='Raw')
    
    # 平滑曲线
    window = 50
    smoothed = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
    plt.plot(smoothed, linewidth=2, label=f'Smoothed ({window} eps)')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Training on CartPole-v1')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
```

### 7.3.3 超参数设置

| 超参数 | Nature DQN (Atari) | CartPole | 说明 |
|--------|---------------------|----------|------|
| Buffer Size | 1,000,000 | 100,000 | Replay buffer 容量 |
| Batch Size | 32 | 64 | 每次采样大小 |
| Target Update | 10,000 steps | 10 episodes | 目标网络更新频率 |
| Learning Rate | 0.00025 | 0.001 | Adam 学习率 |
| γ (gamma) | 0.99 | 0.99 | 折扣因子 |
| ε start | 1.0 | 1.0 | 初始探索率 |
| ε end | 0.1 | 0.01 | 最终探索率 |
| ε decay | 1M frames | 0.995/episode | 衰减方式 |

**调参建议**：
1. **Buffer size**：越大越好（内存允许的情况下）
2. **Batch size**：32-64 对大多数任务有效
3. **Target update**：过频繁不稳定，过稀疏学习慢
4. **Learning rate**：从 1e-3 开始，根据loss曲线调整

---

## 7.4 DQN 变体

DQN 发表后，研究者提出了许多改进变体。

### 7.4.1 Double DQN (van Hasselt et al., 2016)

**动机**：标准 DQN 的最大化偏差

标准 DQN 目标：

$$
y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \boldsymbol{\theta}^-)
$$

问题：同一个网络既**选择**动作又**评估**动作，导致过度乐观估计。

**Double DQN 解决方案**：

使用**在线网络选择**动作，**目标网络评估**价值：

$$
y_t = r_t + \gamma Q(s_{t+1}, \arg\max_{a'} Q(s_{t+1}, a'; \boldsymbol{\theta}); \boldsymbol{\theta}^-)
$$

**代码修改**：

```python
# 标准 DQN
with torch.no_grad():
    max_next_q = target_network(next_states).max(1)[0]
    target_q = rewards + gamma * max_next_q * (1 - dones)

# Double DQN
with torch.no_grad():
    # 用在线网络选择最优动作
    best_actions = q_network(next_states).argmax(1, keepdim=True)
    # 用目标网络评估该动作的价值
    max_next_q = target_network(next_states).gather(1, best_actions).squeeze(1)
    target_q = rewards + gamma * max_next_q * (1 - dones)
```

**效果**：显著减少过度估计，提高稳定性和性能。

### 7.4.2 Dueling DQN (Wang et al., 2016)

**动机**：分离状态价值和动作优势

在许多状态下，不同动作的价值差异很小。Dueling DQN 显式分解：

$$
Q(s, a) = V(s) + A(s, a)
$$

其中：
- $V(s)$：状态价值
- $A(s, a)$：优势函数

**为了唯一性**，添加约束（减去平均优势）：

$$
Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|A|} \sum_{a'} A(s, a') \right)
$$

或使用最大值：

$$
Q(s, a) = V(s) + \left( A(s, a) - \max_{a'} A(s, a') \right)
$$

**网络架构**：

```python
class DuelingDQN(nn.Module):
    """
    Dueling DQN Architecture
    
    分离价值流和优势流
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingDQN, self).__init__()
        
        # 共享特征层
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 价值流（单一输出）
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 优势流（每个动作一个输出）
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        """
        前向传播
        
        Returns:
            q_values: Q(s,a) = V(s) + (A(s,a) - mean A(s,·))
        """
        features = self.feature_layer(state)
        
        value = self.value_stream(features)  # (batch, 1)
        advantages = self.advantage_stream(features)  # (batch, action_dim)
        
        # 结合：Q = V + (A - mean(A))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values
```

**优势**：
- ✅ 更快学习状态价值
- ✅ 在动作价值相近时更稳定
- ✅ 提高泛化能力

<div data-component="DuelingDQNDecomposition"></div>

### 7.4.3 Prioritized Experience Replay (Schaul et al., 2016)

**动机**：并非所有经验同等重要

标准 Replay 均匀采样，但：
- 高 TD error 的转移更有信息量
- 稀有转移应该重复学习

**优先级定义**：

基于 TD error：

$$
p_i = |\delta_i| + \epsilon
$$

或使用排序：

$$
p_i = \frac{1}{\text{rank}(i)}
$$

**采样概率**：

$$
P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}
$$

其中 $\alpha$ 控制优先级强度（$\alpha=0$ 退化为均匀采样）。

**重要性采样权重**（消除偏差）：

$$
w_i = \left( \frac{1}{N} \cdot \frac{1}{P(i)} \right)^\beta
$$

归一化：

$$
w_i \leftarrow \frac{w_i}{\max_j w_j}
$$

**代码实现**（简化版）：

```python
class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    
    Args:
        capacity: 缓冲区容量
        alpha: 优先级指数
        beta_start: 重要性采样初始值
        beta_frames: beta 增长到 1.0 的帧数
    """
    def __init__(self, capacity=100000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
    
    def beta_by_frame(self, frame_idx):
        """线性增长 beta 到 1.0"""
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, state, action, reward, next_state, done):
        """添加经验，初始优先级设为最大"""
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        """按优先级采样"""
        N = len(self.buffer)
        if N == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:N]
        
        # 计算采样概率
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        # 采样 indices
        indices = np.random.choice(N, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # 计算重要性采样权重
        beta = self.beta_by_frame(self.frame)
        self.frame += 1
        
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights)
        
        # 转换为 tensors
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones),
            indices,
            weights
        )
    
    def update_priorities(self, indices, priorities):
        """更新优先级"""
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio
```

**使用**：

```python
# 采样时获取权重
states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size)

# 计算 TD error
td_errors = (target_q - current_q).abs()

# 使用权重加权 loss
loss = (weights * td_errors.pow(2)).mean()

# 更新优先级
replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
```

**效果**：显著提高样本效率，特别是在稀疏奖励任务中。

<div data-component="PrioritizedReplayWeighting"></div>

### 7.4.4 Noisy DQN (Fortunato et al., 2018)

**动机**：用参数化噪声替代 ε-greedy

**Noisy Nets**：在网络权重中添加可学习的噪声：

$$
y = (W + \sigma \odot \epsilon) x + b
$$

其中：
- $W, b$：标准权重和偏置
- $\sigma$：可学习的噪声标准差
- $\epsilon$：随机噪声（从标准正态分布采样）
- $\odot$：元素乘法

**优势**：
- 自动调整探索
- 无需手动调整 ε
- 与状态相关的探索

### 7.4.5 Rainbow DQN (Hessel et al., 2018)

**Rainbow = 所有改进的组合**：

1. ✅ Double Q-learning
2. ✅ Dueling Networks
3. ✅ Prioritized Replay
4. ✅ Multi-step Learning（n-step returns）
5. ✅ Distributional RL（C51）
6. ✅ Noisy Nets

**结果**：在 Atari 上达到当时的 SOTA 性能。

**消融研究**：每个组件都有贡献，组合效果最好。

---

## 7.5 DQN 的局限性

尽管 DQN 是里程碑，但仍有明显局限：

### 7.5.1 仅适用于离散动作空间

**问题**：DQN 需要计算 $\max_a Q(s,a)$，要求动作空间可枚举。

**不适用场景**：
- 连续控制（机器人、自动驾驶）
- 高维离散动作（围棋需要约 19×19=361 个动作，尚可；但更复杂任务不行）

**解决方案**：
- 策略梯度方法（PPO, SAC）
- 动作离散化（不理想）

### 7.5.2 样本效率仍然较低

**问题**：
- Atari 训练需要 **200M 帧**（约 40 小时游戏时间）
- 人类学习同样游戏只需几分钟

**原因**：
- On-policy 数据收集（虽然 off-policy 学习）
- 需要大量探索

### 7.5.3 不稳定性问题

**问题**：
- 训练曲线震荡
- 对超参数敏感
- 容易发散（特别是复杂任务）

**缓解措施**：
- Double DQN
- Gradient clipping
- 仔细调参

---

## 本章小结

在本章中，我们学习了：

✅ **DQN 的历史意义**：开启深度强化学习时代  
✅ **核心机制**：Experience Replay 和 Target Network  
✅ **DQN 变体**：Double, Dueling, Prioritized, Noisy, Rainbow  
✅ **完整实现**：从零实现可运行的 DQN  
✅ **局限性**：仅离散动作、样本效率、稳定性  

> [!TIP]
> **核心要点**：
> - Experience Replay 打破样本相关性，提高样本效率
> - Target Network 稳定训练，防止振荡
> - Double DQN 消除最大化偏差
> - Dueling DQN 分离状态价值和优势
> - Prioritized Replay 关注重要经验
> - Rainbow 组合所有改进达到 SOTA

> [!NOTE]
> **下一步**：
> Chapter 8 将学习**策略梯度基础**，这是处理连续动作空间的关键：
> - 直接优化策略而非价值函数
> - REINFORCE 算法
> - Actor-Critic 架构
> - 为 PPO 打下基础
> 
> 进入 [Chapter 8. 策略梯度基础](08-policy-gradient.md)

---

## 扩展阅读

- **经典论文**：
  - Mnih et al. (2015): Human-level control through deep RL (Nature DQN)
  - van Hasselt et al. (2016): Deep RL with Double Q-learning
  - Wang et al. (2016): Dueling Network Architectures for Deep RL
  - Schaul et al. (2016): Prioritized Experience Replay
  - Hessel et al. (2018): Rainbow: Combining Improvements in Deep RL
- **实现资源**：
  - OpenAI Baselines DQN
  - Stable-Baselines3 DQN
  - CleanRL DQN
- **应用案例**：
  - Atari 游戏（Breakout, Pong, Space Invaders）
  - AlphaGo（结合 MCTS 和价值网络）

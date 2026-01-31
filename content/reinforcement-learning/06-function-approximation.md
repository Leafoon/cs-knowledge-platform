---
title: "Chapter 6. 函数逼近（Function Approximation）"
description: "从表格到函数：处理大规模连续状态空间"
updated: "2026-01-29"
---

> **Learning Objectives**
> * 理解为什么需要函数逼近（维度灾难）
> * 掌握线性函数逼近和特征工程
> * 学习 Semi-gradient 方法（On-policy 和 Off-policy）
> * 理解 Deadly Triad 和其影响
> * 了解深度神经网络在 RL 中的应用（DQN 预告）

---

## 6.1 为什么需要函数逼近

表格方法在大规模/连续状态空间中不可行。

### 6.1.1 维度灾难的现实

**表格方法的限制**：

| 任务 | 状态空间大小 | 表格存储需求 | 可行性 |
|------|------------|------------|--------|
| GridWorld 10×10 | 100 | 100 floats (400 bytes) | ✅ 可行 |
| Chess | ~10<sup>47</sup> | ~10<sup>47</sup> floats | ❌ 不可行 |
| Go (19×19) | ~10<sup>170</sup> | ~10<sup>170</sup> floats | ❌ 完全不可能 |
| Atari (84×84 RGB) | 256<sup>84×84×3</sup> ≈ 10<sup>50000</sup> | 不可计算 | ❌ 不可能 |
| CartPole (连续) | ∞ | ∞ | ❌ 连续状态 |

**关键问题**：
1. **存储**：状态太多，无法存储所有 Q(s,a)
2. **计算**：无法遍历所有状态
3. **泛化**：无法从已见状态泛化到未见状态
4. **连续**：连续状态空间有无穷多个状态

<div data-component="FunctionApproximationComparison"></div>

### 6.1.2 泛化能力的必要性

**泛化（Generalization）**：

从已经历的状态推广到相似的未见状态。

**示例**：

假设在 CartPole 中：
- 见过状态：$(x=0.1, \dot{x}=0.05, \theta=0.01, \dot{\theta}=0.02)$
- 未见状态：$(x=0.11, \dot{x}=0.06, \theta=0.01, \dot{\theta}=0.02)$

**问题**：如何估计未见状态的价值？

**表格方法**：无法泛化（每个状态独立）

**函数逼近**：
- 假设相似状态有相似价值
- 使用参数化函数 $\hat{V}(s, \mathbf{w})$ 或 $\hat{Q}(s, a, \mathbf{w})$
- 调整参数 $\mathbf{w}$ 来逼近真实价值

### 6.1.3 函数逼近的类型

**价值函数逼近**：

$$
\hat{V}(s, \mathbf{w}) \approx V^\pi(s)
$$

$$
\hat{Q}(s, a, \mathbf{w}) \approx Q^\pi(s, a)
$$

**常见函数类**：

1. **线性函数**：
   $$\hat{V}(s, \mathbf{w}) = \mathbf{w}^T \mathbf{x}(s) = \sum_i w_i x_i(s)$$
   
   其中 $\mathbf{x}(s)$ 是特征向量

2. **多项式**：
   $$\hat{V}(s, \mathbf{w}) = w_0 + w_1 s + w_2 s^2 + \cdots$$

3. **神经网络**：
   $$\hat{V}(s, \mathbf{w}) = \text{NN}(s; \mathbf{w})$$

4. **决策树/集成方法**：
   - Random Forest
   - Gradient Boosting

---

## 6.2 线性函数逼近

线性函数逼近是最简单且理论分析最完善的方法。

### 6.2.1 特征向量 x(s)

**定义**：

特征向量 $\mathbf{x}(s) = [x_1(s), x_2(s), \ldots, x_d(s)]^T$ 是状态 $s$ 的数值表示。

**线性价值函数**：

$$
\hat{V}(s, \mathbf{w}) = \mathbf{w}^T \mathbf{x}(s) = \sum_{i=1}^d w_i x_i(s)
$$

**参数**：
- $\mathbf{w} \in \mathbb{R}^d$：权重向量
- $d$：特征维度（远小于状态数）

<div data-component="FeatureEngineeringVisualizer"></div>

### 6.2.2 特征工程示例

**1. 多项式特征（Polynomial Features）**:

对于一维状态 $s \in [0, 1]$：

$$
\mathbf{x}(s) = [1, s, s^2, s^3, \ldots, s^k]^T
$$

**2. Tile Coding（瓦片编码）**：

```python
def tile_coding(state, num_tilings=8, tiles_per_dim=8):
    """
    Tile Coding 特征
    
    Args:
        state: 连续状态 (如 [position, velocity])
        num_tilings: 平铺数量
        tiles_per_dim: 每个维度的瓦片数
    
    Returns:
        features: 稀疏二进制特征向量
    """
    features = []
    
    for tiling in range(num_tilings):
        # 每个 tiling 偏移一点
        offset = tiling / num_tilings
        
        # 计算当前 tiling 中激活的瓦片
        tile_indices = []
        for dim, value in enumerate(state):
            # 归一化到 [0, tiles_per_dim]
            normalized = (value + offset) * tiles_per_dim
            tile_idx = int(np.floor(normalized)) % tiles_per_dim
            tile_indices.append(tile_idx)
        
        # 每个 tiling 贡献一个激活的瓦片
        tile_id = tiling * (tiles_per_dim ** len(state)) + sum(
            idx * (tiles_per_dim ** i) for i, idx in enumerate(tile_indices)
        )
        features.append(tile_id)
    
    # 创建稀疏二进制向量
    total_features = num_tilings * (tiles_per_dim ** len(state))
    feature_vector = np.zeros(total_features)
    for feat in features:
        feature_vector[feat] = 1.0
    
    return feature_vector
```

**3. Radial Basis Functions（径向基函数）**：

$$
x_i(s) = \exp\left( -\frac{\|s - c_i\|^2}{2\sigma^2} \right)
$$

其中 $c_i$ 是中心点，$\sigma$ 是宽度。

**4. Fourier Basis**：

$$
x_i(s) = \cos(\pi \mathbf{c}_i^T s)
$$

### 6.2.3 梯度下降更新

**目标**：最小化价值函数误差

$$
J(\mathbf{w}) = \mathbb{E}\left[ (V^\pi(S) - \hat{V}(S, \mathbf{w}))^2 \right]
$$

**梯度**：

$$
\nabla_\mathbf{w} J(\mathbf{w}) = -2 \mathbb{E}\left[ (V^\pi(S) - \hat{V}(S, \mathbf{w})) \nabla_\mathbf{w} \hat{V}(S, \mathbf{w}) \right]
$$

**对于线性函数**：

$$
\nabla_\mathbf{w} \hat{V}(s, \mathbf{w}) = \mathbf{x}(s)
$$

**随机梯度下降（SGD）更新**：

$$
\mathbf{w} \leftarrow \mathbf{w} + \alpha [V^\pi(s) - \hat{V}(s, \mathbf{w})] \mathbf{x}(s)
$$

**问题**：我们不知道 $V^\pi(s)$！

**解决方案**：使用 TD target 作为近似。

---

## 6.3 Semi-gradient 方法

Semi-gradient 方法使用 bootstrapping（TD target）代替真实目标。

### 6.3.1 Semi-gradient TD(0)

**更新规则**：

$$
\mathbf{w} \leftarrow \mathbf{w} + \alpha [R + \gamma \hat{V}(S', \mathbf{w}) - \hat{V}(S, \mathbf{w})] \nabla_\mathbf{w} \hat{V}(S, \mathbf{w})
$$

**为什么叫"Semi-gradient"？**

真正的梯度应该是：

$$
\nabla_\mathbf{w} [R + \gamma \hat{V}(S', \mathbf{w}) - \hat{V}(S, \mathbf{w})]^2
$$

这会包含 $\nabla_\mathbf{w} \hat{V}(S', \mathbf{w})$ 项。

**Semi-gradient 忽略了目标中的 $\mathbf{w}$ 依赖**：

$$
\nabla_\mathbf{w} [R + \gamma \hat{V}(S', \mathbf{w}) - \hat{V}(S, \mathbf{w})] \approx -\nabla_\mathbf{w} \hat{V}(S, \mathbf{w})
$$

**算法（Semi-gradient TD(0)）**：

```python
def semi_gradient_td0(env, policy, num_episodes=1000, alpha=0.01, gamma=0.99):
    """
    Semi-gradient TD(0) with linear function approximation
    
    Args:
        env: Gym 环境
        policy: 策略函数
        num_episodes: episode 数量
        alpha: 学习率
        gamma: 折扣因子
    
    Returns:
        w: 学习到的权重向量
    """
    # 初始化权重（假设 feature_dim 已知）
    feature_dim = 100  # 示例
    w = np.zeros(feature_dim)
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # 提取特征
            x = extract_features(state)
            
            # 当前价值估计
            v = np.dot(w, x)
            
            # 执行动作
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            
            # 下一状态特征和价值
            x_next = extract_features(next_state)
            v_next = np.dot(w, x_next)
            
            # TD error
            delta = reward + gamma * v_next * (1 - done) - v
            
            # Semi-gradient 更新
            w += alpha * delta * x
            
            state = next_state
    
    return w
```

### 6.3.2 Semi-gradient SARSA

**算法（Semi-gradient SARSA）**：

$$
\mathbf{w} \leftarrow \mathbf{w} + \alpha [R + \gamma \hat{Q}(S', A', \mathbf{w}) - \hat{Q}(S, A, \mathbf{w})] \nabla_\mathbf{w} \hat{Q}(S, A, \mathbf{w})
$$

**代码实现**：

```python
def semi_gradient_sarsa(env, num_episodes=1000, alpha=0.01, gamma=0.99, epsilon=0.1):
    """
    Semi-gradient SARSA with linear function approximation
    
    使用 Tile Coding 作为特征
    """
    # 初始化（假设使用 tile coding）
    num_tilings = 8
    tiles_per_dim = 8
    num_actions = env.action_space.n
    feature_dim = num_tilings * (tiles_per_dim ** 2) * num_actions
    
    w = np.zeros(feature_dim)
    
    def get_features(state, action):
        """获取状态-动作特征"""
        state_features = tile_coding(state, num_tilings, tiles_per_dim)
        # 为每个动作创建独立特征
        features = np.zeros(feature_dim)
        offset = action * len(state_features)
        features[offset:offset+len(state_features)] = state_features
        return features
    
    def get_q_value(state, action):
        """计算 Q 值"""
        features = get_features(state, action)
        return np.dot(w, features)
    
    def epsilon_greedy(state):
        """ε-greedy 策略"""
        if np.random.random() < epsilon:
            return env.action_space.sample()
        else:
            q_values = [get_q_value(state, a) for a in range(num_actions)]
            return np.argmax(q_values)
    
    for episode in range(num_episodes):
        state = env.reset()
        action = epsilon_greedy(state)
        done = False
        
        while not done:
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(next_state)
            
            # Semi-gradient SARSA 更新
            features = get_features(state, action)
            q = get_q_value(state, action)
            q_next = get_q_value(next_state, next_action) * (1 - done)
            
            delta = reward + gamma * q_next - q
            w += alpha * delta * features
            
            state = next_state
            action = next_action
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}")
    
    return w
```

### 6.3.3 收敛性问题

**线性函数逼近 + On-policy**：

✅ **保证收敛**到 $\mathbf{w}^*$ 使得：

$$
\overline{VE}(\mathbf{w}^*) \leq \frac{1}{1-\gamma} \min_\mathbf{w} \overline{VE}(\mathbf{w})
$$

其中 $\overline{VE}$ 是平均平方价值误差。

**线性函数逼近 + Off-policy**：

⚠️ **可能发散**（见 Deadly Triad）

---

## 6.4 Deadly Triad

Deadly Triad 是强化学习中最危险的组合。

### 6.4.1 三个要素

**Deadly Triad**：

1. **函数逼近**（Function Approximation）
2. **Bootstrapping**（例如 TD）
3. **Off-policy 学习**

**当这三者同时存在时，算法可能发散！**

<div data-component="DeadlyTriadDemo"></div>

### 6.4.2 Baird 反例

**Baird's Counterexample**（1995）：

一个简单的 MDP，使用线性函数逼近的 Off-policy TD(0) 会导致权重 $\mathbf{w} \to \infty$。

**设置**：
- 7 个状态
- 2 个动作（solid, dashed）
- 线性函数逼近（8 维特征）
- 行为策略：随机选择 dashed (6/7 概率)
- 目标策略：总是选择 solid

**结果**：权重指数增长，发散！

**教训**：
- Off-policy + Bootstrapping + 函数逼近 = 危险
- 需要特殊技术来稳定学习

### 6.4.3 解决方案

**1. 避免 Deadly Triad**：
- 使用 On-policy 方法（A3C, PPO）
- 使用 Monte Carlo（无 Bootstrapping）

**2. 使用稳定技术**：
- **Gradient TD**（GTD, TDC）：真正的梯度下降
- **Emphatic TD**：调整状态访问分布
- **Experience Replay + Target Network**（DQN）

**3. Trust Region 方法**：
- TRPO, PPO：限制策略变化

---

## 6.5 深度神经网络

深度神经网络是最强大的函数逼近器。

### 6.5.1 神经网络作为通用逼近器

**Universal Approximation Theorem**：

一个足够大的前馈神经网络可以以任意精度逼近任何连续函数。

**RL 中的神经网络**：

$$
\hat{Q}(s, a, \mathbf{w}) = \text{NN}(s, a; \mathbf{w})
$$

其中 $\mathbf{w}$ 是所有网络参数（权重和偏置）。

<div data-component="NeuralNetworkApproximation"></div>

### 6.5.2 网络架构设计

**典型架构**：

```python
import torch
import torch.nn as nn

class ValueNetwork(nn.Module):
    """
    深度价值网络
    
    Args:
        state_dim: 状态维度
        hidden_dims: 隐藏层维度列表
    """
    def __init__(self, state_dim, hidden_dims=[128, 128]):
        super(ValueNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """
        前向传播
        
        Args:
            state: Tensor of shape (batch_size, state_dim)
        
        Returns:
            value: Tensor of shape (batch_size, 1)
        """
        return self.network(state)

class QNetwork(nn.Module):
    """
    深度 Q 网络（DQN 架构）
    
    Args:
        state_dim: 状态维度
        action_dim: 动作数量
        hidden_dims: 隐藏层维度
    """
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128]):
        super(QNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # 输出层：每个动作一个 Q 值
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """
        前向传播
        
        Args:
            state: Tensor of shape (batch_size, state_dim)
        
        Returns:
            q_values: Tensor of shape (batch_size, action_dim)
        """
        return self.network(state)
```

### 6.5.3 训练技巧

**1. 经验回放（Experience Replay）**：

```python
from collections import deque
import random

class ReplayBuffer:
    """
    经验回放缓冲区
    
    存储 (state, action, reward, next_state, done) 转移
    """
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """添加一个转移"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样一个 batch"""
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
```

**优点**：
- 打破样本相关性
- 提高数据效率（重复使用）
- Off-policy 学习

**2. Target Network（目标网络）**：

```python
# 主网络
q_network = QNetwork(state_dim, action_dim)

# 目标网络（参数固定，定期更新）
target_network = QNetwork(state_dim, action_dim)
target_network.load_state_dict(q_network.state_dict())

# 训练循环
for step in range(num_steps):
    # 使用主网络选择动作
    action = select_action(q_network, state)
    
    # ...收集经验...
    
    # 使用目标网络计算 TD target
    with torch.no_grad():
        target_q = reward + gamma * target_network(next_state).max()
    
    # 更新主网络
    current_q = q_network(state)[action]
    loss = (current_q - target_q) ** 2
    loss.backward()
    optimizer.step()
    
    # 定期更新目标网络
    if step % target_update_freq == 0:
        target_network.load_state_dict(q_network.state_dict())
```

**优点**：
- 稳定训练（目标不频繁变化）
- 减少振荡

**3. 梯度裁剪（Gradient Clipping）**：

```python
# 防止梯度爆炸
torch.nn.utils.clip_grad_norm_(q_network.parameters(), max_norm=10.0)
```

---

## 6.6 DQN 预告

Deep Q-Network (DQN) 结合了上述所有技术。

### 6.6.1 完整 DQN 算法

**算法（DQN - Mnih et al., 2015）**：

```
初始化:
    Q-network with random weights θ
    Target network with weights θ⁻ = θ
    Replay buffer D

For each episode:
    初始化 state s
    
    For each step:
        # ε-greedy 选择动作
        a = argmax_a Q(s,a,θ) with prob 1-ε
             random action      with prob ε
        
        # 执行并存储
        执行 a，观察 r, s'
        存储 (s,a,r,s',done) 到 D
        
        # 采样并训练
        if |D| >= batch_size:
            采样 minibatch from D
            
            # 计算 target（使用目标网络）
            y = r + γ max_a' Q(s',a',θ⁻) * (1-done)
            
            # 梯度下降
            L = (Q(s,a,θ) - y)²
            θ ← θ - α ∇_θ L
        
        # 更新目标网络
        every C steps: θ⁻ ← θ
```

**关键组件**：
1. ✅ 深度神经网络（函数逼近）
2. ✅ Q-learning（Bootstrapping + Off-policy）
3. ✅ Experience Replay（打破相关性）
4. ✅ Target Network（稳定训练）

**结果**：
- 在 49 个 Atari 游戏中达到人类水平
- 开启深度强化学习时代

### 6.6.2 CartPole 示例代码

```python
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train_dqn_cartpole():
    """
    使用 DQN 解决 CartPole 任务
    """
    # 环境
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n  # 2
    
    # 网络
    q_network = QNetwork(state_dim, action_dim, hidden_dims=[64, 64])
    target_network = QNetwork(state_dim, action_dim, hidden_dims=[64, 64])
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    
    # 超参
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    batch_size = 64
    target_update = 10
    
    # Replay buffer
    replay_buffer = ReplayBuffer(capacity=10000)
    
    # 训练
    episode_rewards = []
    
    for episode in range(500):
        state = env.reset()[0]
        episode_reward = 0
        done = False
        
        while not done:
            # ε-greedy
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = q_network(state_tensor)
                    action = q_values.argmax().item()
            
            # 执行
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            episode_reward += reward
            
            # 存储
            replay_buffer.push(state, action, reward, next_state, float(done))
            
            # 训练
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # 当前 Q
                current_q = q_network(states).gather(1, actions.unsqueeze(1))
                
                # Target Q（使用目标网络）
                with torch.no_grad():
                    max_next_q = target_network(next_states).max(1)[0]
                    target_q = rewards + gamma * max_next_q * (1 - dones)
                
                # Loss
                loss = nn.MSELoss()(current_q.squeeze(), target_q)
                
                # 更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            state = next_state
        
        # 递减 ε
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # 更新目标网络
        if episode % target_update == 0:
            target_network.load_state_dict(q_network.state_dict())
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, ε: {epsilon:.3f}")
    
    env.close()
    return q_network, episode_rewards

if __name__ == "__main__":
    q_net, rewards = train_dqn_cartpole()
    
    # 绘制学习曲线
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN on CartPole-v1')
    plt.show()
```

---

## 本章小结

在本章中，我们学习了：

✅ **函数逼近动机**：解决维度灾难，实现泛化  
✅ **线性函数逼近**：特征工程（Tile Coding, RBF, Fourier）  
✅ **Semi-gradient 方法**：TD(0), SARSA 的函数逼近版本  
✅ **Deadly Triad**：函数逼近 + Bootstrapping + Off-policy 的危险  
✅ **深度神经网络**：通用函数逼近器  
✅ **DQN**：Experience Replay + Target Network，开启深度 RL 时代  

> [!TIP]
> **核心要点**：
> - 函数逼近是处理大规模问题的关键
> - 特征工程对线性方法至关重要
> - Semi-gradient 方法在 On-policy 情况下收敛
> - Deadly Triad 需要特殊技术（Replay + Target Network）
> - 深度神经网络 + RL = 深度强化学习（DQN开始）

> [!NOTE]
> **下一步**：
> 现在已经掌握了经典强化学习的核心内容！后续章节将深入：
> - Chapter 7-10：策略梯度方法（REINFORCE, Actor-Critic, PPO）
> - Chapter 11-15：深度强化学习（DQN改进、连续控制、Model-based）
> - Chapter 16+：前沿方向（RLHF, Offline RL, Multi-agent等）

---

## 扩展阅读

- **Sutton & Barto**：Chapter 9 (On-policy Prediction with Approximation), Chapter 11 (Off-policy Methods with Approximation)
- **经典论文**：
  - Mnih et al. (2015): Human-level control through deep reinforcement learning (DQN)
  - Baird (1995): Residual Algorithms: Reinforcement Learning with Function Approximation
  - Tsitsiklis & Van Roy (1997): Analysis of Temporal-Difference Learning with Function Approximation
- **深度 RL 综述**：
  - Arulkumaran et al. (2017): Deep Reinforcement Learning: A Brief Survey
  - Li (2018): Deep Reinforcement Learning

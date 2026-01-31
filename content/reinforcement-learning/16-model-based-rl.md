---
title: "Chapter 16. Model-Based Reinforcement Learning"
description: "在想象中学习与规划"
updated: "2026-01-29"
---

> **Learning Objectives**
> * 理解 Model-based RL 的优势与挑战
> * 掌握环境模型的学习方法
> * 学习 Dyna 架构与规划
> * 了解 MBPO 的短期模型滚动
> * 掌握 World Models 与 Dreamer 系列

---

## 16.1 为什么需要 Model-Based RL？

Model-based RL 通过学习环境模型来提升样本效率和规划能力。

### 16.1.1 样本效率提升

**Model-free RL 的瓶颈**：需要大量真实交互。

**Model-based 的优势**：
- 学习环境模型 $P(s'|s,a)$
- 在**想象**中生成轨迹
- 无需与真实环境交互即可训练

**示例**：
- Model-free：需要 10M 真实步
- Model-based：可能只需 100K 真实步 + 大量想象步

### 16.1.2 规划能力

**规划**（Planning）：使用模型模拟未来，选择最优动作。

**方法**：
- **前向搜索**：模拟多条轨迹，选择最佳
- **动态规划**：在模型上执行值迭代

**优势**：
- 可以提前"思考"而不是仅仅反应
- 对分布外状态有更好的泛化

### 16.1.3 与 Model-Free 的对比

<div data-component="ModelBasedVsModelFree"></div>

| 特性 | Model-Free | Model-Based |
|------|-----------|-------------|
| **样本效率** | 低 | 高 |
| **渐近性能** | 高（如果有足够样本） | 可能受限于模型误差 |
| **泛化性** | 局限于经历过的状态 | 可泛化到新状态 |
| **计算** | 简单（前向传播） | 复杂（需要规划） |
| **模型误差** | 无 | 可能导致性能下降 |

**实践建议**：
- 样本昂贵（机器人）→ Model-based
- 样本便宜（模拟器）→ Model-free
- 混合方法（MBPO）→ 两全其美

---

## 16.2 环境模型学习

学习环境的转移和奖励函数。

### 16.2.1 转移模型 P(s'|s,a)

**确定性环境**：

$$
s' = f_{\boldsymbol{\phi}}(s, a)
$$

使用神经网络 $f_{\boldsymbol{\phi}}$ 拟合：

```python
class DeterministicTransitionModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, state, action):
        """预测下一个状态"""
        x = torch.cat([state, action], dim=-1)
        delta = self.net(x)  # 预测状态变化
        next_state = state + delta
        return next_state
```

**随机性环境**：

$$
s' \sim P_{\boldsymbol{\phi}}(\cdot|s, a)
$$

使用概率分布（例如高斯）：

$$
P_{\boldsymbol{\phi}}(s'|s, a) = \mathcal{N}(\mu_{\boldsymbol{\phi}}(s, a), \Sigma_{\boldsymbol{\phi}}(s, a))
$$

### 16.2.2 奖励模型 R(s,a)

**学习奖励函数**：

$$
\hat{r} = R_{\boldsymbol{\psi}}(s, a)
$$

```python
class RewardModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        """预测奖励"""
        x = torch.cat([state, action], dim=-1)
        reward = self.net(x)
        return reward
```

### 16.2.3 监督学习方法

**训练数据**：从真实环境收集 $(s, a, s', r)$

**损失函数**：

$$
\mathcal{L}_{\text{model}} = \mathbb{E}_{(s,a,s',r) \sim \mathcal{D}} \left[\|f_{\boldsymbol{\phi}}(s,a) - s'\|^2 + \|R_{\boldsymbol{\psi}}(s,a) - r\|^2\right]
$$

**训练循环**：

```python
def train_model(model, replay_buffer, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        batch = replay_buffer.sample(batch_size=256)
        states, actions, rewards, next_states, dones = batch
        
        # 预测
        pred_next_states = model.transition(states, actions)
        pred_rewards = model.reward(states, actions)
        
        # 损失
        transition_loss = F.mse_loss(pred_next_states, next_states)
        reward_loss = F.mse_loss(pred_rewards, rewards)
        loss = transition_loss + reward_loss
        
        # 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 16.2.4 模型误差问题

**复合误差**（Compounding Error）：

- 单步误差小（例如 0.1%）
- 多步模拟：误差累积 → 轨迹偏离真实

**示例**：
- 1 步：误差 ~0.1%
- 100 步：误差可能 ~10%
- 1000 步：完全不可靠

**缓解方法**：
1. **短期滚动**：只用模型预测短期（例如 5-10 步）
2. **模型集成**：训练多个模型，取平均/最小值
3. **不确定性建模**：估计模型不确定性，避免在不确定区域规划

---

## 16.3 Dyna 架构

Dyna 结合真实经验和模拟经验。

<div data-component="DynaArchitecture"></div>

### 16.3.1 Real Experience + Simulated Experience

**Dyna 流程**：

```
1. 真实交互: (s, a, r, s') → 存入 replay buffer
2. 学习模型: 用 replay buffer 训练 model
3. 规划: 用 model 生成模拟经验
4. 学习策略: 用真实 + 模拟经验训练 policy
```

**关键思想**：
- **Real experience**：确保数据真实
- **Simulated experience**：提升样本效率

### 16.3.2 Dyna-Q 算法

**伪代码**：

```python
# Dyna-Q 算法
for episode in range(num_episodes):
    state = env.reset()
    
    while not done:
        # (a) 选择动作并执行
        action = epsilon_greedy(Q, state)
        next_state, reward, done = env.step(action)
        
        # (b) 直接更新 Q（model-free）
        Q[state, action] += alpha * (reward + gamma * max(Q[next_state]) - Q[state, action])
        
        # (c) 更新模型
        Model[state, action] = (next_state, reward)
        
        # (d) 规划：n 步模拟
        for _ in range(planning_steps):
            # 随机采样已访问过的 (s, a)
            s = random.choice(visited_states)
            a = random.choice(visited_actions[s])
            
            # 用模型模拟
            s_prime, r = Model[s, a]
            
            # 更新 Q（基于模拟）
            Q[s, a] += alpha * (r + gamma * max(Q[s_prime]) - Q[s, a])
        
        state = next_state
```

### 16.3.3 规划步数选择

**trade-off**：
- **planning_steps 小**：接近 model-free，样本效率提升不明显
- **planning_steps 大**：计算开销大，模型误差累积

**实践**：
- 通常 planning_steps = 5~50
- 取决于模型质量和计算资源

---

## 16.4 MBPO (Model-Based Policy Optimization)

结合短期模型滚动和 SAC。

### 16.4.1 短期模型滚动

**动机**：避免长期模拟的复合误差。

**MBPO 方法**：
1. 从真实状态 $s$ 开始
2. 用模型滚动 $k$ 步（例如 $k=5$）
3. 将模拟轨迹加入 replay buffer
4. 用 SAC 在混合 buffer 上训练

**公式**：

$$
s_0 \sim \mathcal{D}, \quad s_{t+1} \sim \hat{P}(s_{t+1}|s_t, a_t), \quad a_t \sim \pi(s_t), \quad t=0,\ldots,k-1
$$

**优势**：
- 短期预测更准确
- 样本效率提升
- 保持 off-policy 优势

### 16.4.2 与 SAC 结合

**MBPO 算法**：

```python
# MBPO 主循环
for step in range(total_steps):
    # (1) 真实交互
    action = policy.select_action(state)
    next_state, reward, done = env.step(action)
    real_buffer.add(state, action, reward, next_state, done)
    
    # (2) 训练模型
    if step % model_train_freq == 0:
        train_model(model, real_buffer)
    
    # (3) 生成模拟数据（短期滚动）
    for _ in range(num_rollouts):
        # 从真实 buffer 采样起始状态
        start_state = real_buffer.sample_state()
        
        # 滚动 k 步
        s = start_state
        for t in range(rollout_length):  # rollout_length = 5
            a = policy.select_action(s)
            s_next, r = model.predict(s, a)
            
            # 加入模拟 buffer
            sim_buffer.add(s, a, r, s_next, done=False)
            s = s_next
    
    # (4) 训练策略（SAC）
    combined_buffer = real_buffer + sim_buffer
    sac_agent.train_step(combined_buffer)
```

### 16.4.3 模型集成（Ensemble）

**单个模型的问题**：可能过拟合或有系统性偏差。

**Ensemble 方法**：
- 训练 $N$ 个独立模型（例如 $N=5$）
- 预测时随机选择一个模型

$$
\hat{P}_{\text{ensemble}}(s'|s,a) = \frac{1}{N} \sum_{i=1}^{N} P_i(s'|s,a)
$$

**优势**：
- 减少过拟合
- 更鲁棒的预测
- 可以估计不确定性（模型间方差）

---

## 16.5 世界模型（World Models）

学习环境的压缩表示，在潜在空间中规划。

### 16.5.1 学习压缩表示

**动机**：原始状态（例如图像）维度太高。

**World Model 架构**（Ha & Schmidhuber, 2018）：

```
VAE: 图像 → 潜在编码 z
RNN: z_t, a_t → z_{t+1}
```

1. **VAE（Variational Autoencoder）**：压缩观察
   $$
   z_t \sim \text{Encoder}(o_t), \quad \hat{o}_t = \text{Decoder}(z_t)
   $$

2. **RNN (MDN-RNN)**：预测潜在状态转移
   $$
   z_{t+1} \sim \text{RNN}(z_t, a_t, h_t)
   $$

<div data-component="WorldModelVisualization"></div>

### 16.5.2 在想象中训练

**训练 Controller**（策略）：
- 不与真实环境交互
- 仅在 World Model 中"想象"

**流程**：

```python
# 在 World Model 中训练策略
for episode in range(num_episodes):
    z = sample_initial_latent()
    
    for t in range(episode_length):
        # 策略在潜在空间选择动作
        action = controller(z)
        
        # 用 RNN 预测下一个潜在状态
        z_next = world_model.predict(z, action)
        
        # 用 controller 梦中的奖励训练
        reward = world_model.reward(z, action)
        
        # 更新策略（例如进化算法或 REINFORCE）
        controller.update(reward)
        
        z = z_next
```

### 16.5.3 Ha & Schmidhuber (2018)

**Car Racing 任务**：
- 用 10,000 轨迹训练 World Model
- 在 World Model 中用进化策略训练 Controller
- **完全无需真实环境**即可学会驾驶

**限制**：
- World Model 必须非常准确
- 难以泛化到复杂环境

---

## 16.6 Dreamer 系列

Dreamer 是现代 World Model 方法，在潜在空间执行 actor-critic。

### 16.6.1 DreamerV1: 潜在空间规划

**关键思想**：
- 学习 RSSM（Recurrent State Space Model）
- 在潜在空间执行想象滚动
- 用 actor-critic 训练策略

### 16.6.2 DreamerV2: 离散潜在表示

**改进**：
- 使用**离散潜在**（categorical distribution）
- 更好的重建质量
- 更稳定的训练

### 16.6.3 DreamerV3: 统一算法

**DreamerV3 (Hafner et al., 2023)**：
- 单一算法，无需任务特定调参
- 在 Atari、DMC、Minecraft 等多种任务表现优异
- **样本效率**达到或超过 model-free SOTA

### 16.6.4 RSSM (Recurrent State Space Model)

**RSSM 架构**：

$$
\begin{align}
\text{决定性状态:} \quad & h_t = f(h_{t-1}, z_{t-1}, a_{t-1}) \\
\text{随机状态:} \quad & z_t \sim p(z_t | h_t) \\
\text{观察重建:} \quad & \hat{o}_t \sim p(o_t | h_t, z_t) \\
\text{奖励预测:} \quad & \hat{r}_t = r(h_t, z_t)
\end{align}
$$

**训练**：
- 最大化 ELBO（Evidence Lower Bound）
- 重建观察 + 预测奖励 + KL 正则化

<div data-component="DreamerRollout"></div>

**想象轨迹**：

```python
# Dreamer 想象滚动
def imagine_trajectory(rssm, policy, start_state, horizon=15):
    states, actions, rewards = [], [], []
    
    h, z = start_state
    
    for t in range(horizon):
        # 策略选择动作（在潜在状态上）
        a = policy(h, z)
        
        # RSSM 预测下一状态
        h_next = rssm.deterministic(h, z, a)
        z_next = rssm.stochastic(h_next).sample()
        
        # 预测奖励
        r = rssm.reward(h_next, z_next)
        
        states.append((h, z))
        actions.append(a)
        rewards.append(r)
        
        h, z = h_next, z_next
    
    return states, actions, rewards
```

**训练 Actor-Critic**（在想象中）：

```python
# 用想象轨迹训练 policy 和 value
imagined_states, imagined_actions, imagined_rewards = imagine_trajectory(...)

# 计算 λ-return
returns = compute_lambda_return(imagined_rewards, values)

# Actor loss
actor_loss = -(log_probs * advantages).mean()

# Critic loss
critic_loss = F.mse_loss(values, returns)
```

---

## 本章小结

在本章中，我们学习了：

✅ **Model-based RL 优势**：样本效率、规划能力  
✅ **环境模型学习**：转移模型、奖励模型、监督学习  
✅ **Dyna 架构**：真实经验 + 模拟经验  
✅ **MBPO**：短期模型滚动 + SAC  
✅ **World Models 与 Dreamer**：潜在空间规划、RSSM  

> [!TIP]
> **核心要点**：
> - Model-based RL 通过学习环境模型大幅提升样本效率
> - 模型误差是主要挑战，短期滚动和集成是有效缓解方法
> - Dyna 结合真实和模拟经验
> - MBPO 用短期滚动避免复合误差
> - Dreamer 在潜在空间执行想象滚动，实现高样本效率
> - DreamerV3 是当前 model-based SOTA，适用于多种任务

> [!NOTE]
> **后续章节将涵盖**：
> - Chapter 17: 探索策略（ICM, RND, Go-Explore）
> - Chapter 18: 层次化 RL
> - Multi-agent RL
> - Meta-RL
> - RLHF 与大模型对齐

---

## 扩展阅读

- **经典论文**：
  - Sutton & Barto (2018): Chapter 8 - Planning and Learning
  - Janner et al. (2019): When to Trust Your Model: Model-Based Policy Optimization (MBPO)
  - Ha & Schmidhuber (2018): World Models
  - Hafner et al. (2020, 2021, 2023): Dreamer V1/V2/V3
- **实现资源**：
  - Dreamer V3: Official implementation
  - MBPO: Author's implementation
  - PlaNet, Dreamer: TensorFlow implementations
- **应用案例**：
  - 机器人控制（真实样本昂贵）
  - 自动驾驶仿真
  - Minecraft 等复杂环境

---
title: "Chapter 21. Offline Reinforcement Learning"
description: "从静态数据集学习策略"
updated: "2026-01-29"
---

> **Learning Objectives**
> * 理解 Offline RL 的动机与挑战
> * 掌握 BCQ 的批量约束方法
> * 学习 CQL 的保守 Q 学习
> * 了解 IQL 的隐式 Q 学习
> * 掌握 Decision Transformer 的序列建模方法

---

## 21.1 Offline RL 的动机

Offline RL 从预先收集的静态数据集学习策略，无需在线交互。

### 21.1.1 利用历史数据

**问题**：许多领域已有大量历史数据。

**示例**：
- 医疗：电子健康记录
- 自动驾驶：数百万公里的驾驶数据
- 机器人：多年的操作日志

**Offline RL 目标**：充分利用这些数据。

### 21.1.2 避免在线交互

**传统 RL**：需要大量在线探索

**问题**：
- ❌ 在线交互昂贵（时间、成本）
- ❌ 探索可能危险（医疗、自动驾驶）
- ❌ 无法重置环境（历史系统）

**Offline RL**：
- ✅ 无需环境交互
- ✅ 利用已有数据
- ✅ 安全训练

### 21.1.3 安全关键应用

**场景**：不能承受探索失败的应用。

**示例**：
- 医疗决策：不能拿患者生命冒险
- 自动驾驶：不能在真实道路上随机探索
- 金融交易：不能承受巨额损失

---

## 21.2 Offline RL 的挑战

Offline RL 面临独特的挑战。

<div data-component="OfflineRLChallenge"></div>

### 21.2.1 分布外动作（OOD Actions）

**问题**：数据集中未见过的 $(s, a)$ 对。

<div data-component="OODActionProblem"></div>

**示例**：
- 数据：保守驾驶员的数据
- 策略：学到激进动作（高 Q 值）
- 现实：激进动作从未在数据中出现 → Q 值不准

**原因**：Q 函数在 OOD 区域**外推误差**。

### 21.2.2 外推误差（Extrapolation Error）

**定义**：在未见过的 $(s, a)$ 对上，Q 函数估计不准确。

**Q-learning 导致外推**：

$$
\max_a Q(s, a) \text{ 可能选择 OOD 动作}
$$

**后果**：Q 值被高估 → 策略学到错误行为。

### 21.2.3 Deadly Triad 再现

**回顾 Deadly Triad**（Ch5）：
- 函数逼近
- Bootstrapping
- Off-policy

**Offline RL**：全部满足 → 不稳定。

---

## 21.3 保守策略

通过约束策略接近行为策略来避免 OOD。

### 21.3.1 BCQ (Batch-Constrained Q-learning)

**核心思想**：限制策略只选择数据集中出现过的动作。

**BCQ 策略**：

$$
\pi(a|s) \approx \arg\max_{a \sim \pi_\beta(a|s)} Q(s, a)
$$

**实现**：

1. 学习行为策略 $\pi_\beta$ 的生成模型（VAE）
2. 从 VAE 采样动作
3. 从采样中选择最大 Q 值的动作

**代码**：

```python
import torch
import torch.nn as nn

class BCQ_Actor(nn.Module):
    """
    BCQ Actor: 生成接近数据分布的动作
    使用 VAE 建模行为策略
    """
    def __init__(self, state_dim, action_dim, latent_dim=2*action_dim):
        super().__init__()
        
        # VAE Encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, 750),
            nn.ReLU(),
            nn.Linear(750, 750),
            nn.ReLU()
        )
        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)
        
        # VAE Decoder
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, 750),
            nn.ReLU(),
            nn.Linear(750, 750),
            nn.ReLU(),
            nn.Linear(750, action_dim),
            nn.Tanh()  # 假设动作在 [-1, 1]
        )
    
    def encode(self, state, action):
"""
        """编码 (s,a) 到潜在空间"""
        x = torch.cat([state, action], dim=-1)
        h = self.encoder(x)
        mean = self.mean(h)
        log_std = self.log_std(h)
        return mean, log_std
    
    def decode(self, state, z):
        """从潜在变量解码出动作"""
        x = torch.cat([state, z], dim=-1)
        return self.decoder(x)
    
    def forward(self, state, action):
        """VAE 前向传播"""
        mean, log_std = self.encode(state, action)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        recon_action = self.decode(state, z)
        return recon_action, mean, log_std
    
    def sample_action(self, state, num_samples=10):
        """
        采样多个动作并选择最佳
        
        Args:
            state: 状态
            num_samples: 采样数量
        
        Returns:
            actions: [num_samples, action_dim]
        """
        state_repeat = state.unsqueeze(1).repeat(1, num_samples, 1)
        state_repeat = state_repeat.view(-1, state.shape[-1])
        
        # 从先验采样 z
        z = torch.randn(state_repeat.shape[0], self.latent_dim).to(state.device)
        
        # 解码
        actions = self.decode(state_repeat, z)
        return actions.view(state.shape[0], num_samples, -1)
```

### 21.3.2 行为克隆正则化

**添加 BC 损失**：

$$
\mathcal{L} = \mathcal{L}_{\text{RL}} + \lambda \cdot \mathcal{L}_{\text{BC}}
$$

其中：

$$
\mathcal{L}_{\text{BC}} = -\log \pi(a|s) \quad \text{for } (s, a) \in \mathcal{D}
$$

### 21.3.3 TD3+BC

**简单有效的方法**（Fujimoto & Gu, 2021）：

在 TD3 基础上添加 BC 正则化：

$$
\mathcal{L}_\pi = -Q(s, \pi(s)) + \alpha \cdot (\pi(s) - a)^2
$$

**实现**：

```python
def td3_bc_loss(policy, critic, states, actions, alpha=2.5):
    """
    TD3+BC 策略损失
    
    Args:
        policy: 策略网络
        critic: Q 网络
        states, actions: 批次数据
        alpha: BC 正则化权重
    
    Returns:
        loss: 策略损失
    """
    # Q-learning 部分（最大化 Q 值）
    pi_actions = policy(states)
    q_loss = -critic(states, pi_actions).mean()
    
    # BC 正则化（接近数据）
    bc_loss = (pi_actions - actions).pow(2).mean()
    
    # 总损失
    loss = q_loss + alpha * bc_loss 
    return loss
```

---

## 21.4 Conservative Q-Learning (CQL)

通过保守估计 Q 值避免外推误差。

<div data-component="CQLObjective"></div>

### 21.4.1 Q 值下界估计

**核心思想**：惩罚 OOD 动作的 Q 值。

**CQL 目标**：学习 Q 函数的**下界**。

$$
\min_Q \alpha \mathbb{E}_{s \sim \mathcal{D}} \left[\log \sum_a \exp(Q(s, a)) - \mathbb{E}_{a \sim \mathcal{D}}[Q(s, a)]\right] + \mathcal{L}_{\text{TD}}
$$

**解释**：
- 第一项：降低所有动作的 Q 值（背景）
- 第二项：提升数据中动作的 Q 值
- 第三项：TD 误差

### 21.4.2 CQL 损失函数

**完整 CQL 损失**：

```python
def cql_loss(q_network, states, actions, next_states, rewards, dones, 
             alpha=1.0, num_samples=10):
    """
    CQL (Conservative Q-Learning) 损失
    
    Args:
        q_network: Q 网络
        states, actions, next_states, rewards, dones: 批次
        alpha: CQL 正则化强度
        num_samples: 采样动作数
    
    Returns:
        loss: CQL 损失
    """
    batch_size = states.shape[0]
    
    # (1) 计算当前 Q 值
    current_q = q_network(states, actions)
    
    # (2) 计算目标 Q 值（标准 TD）
    with torch.no_grad():
        next_actions = policy(next_states)  # 目标策略
        target_q = q_network_target(next_states, next_actions)
        target = rewards + (1 - dones) * gamma * target_q
    
    # (3) TD 损失
    td_loss = (current_q - target).pow(2).mean()
    
    # (4) CQL 正则化
    # 随机采样动作
    random_actions = torch.rand(batch_size, num_samples, action_dim) * 2 - 1
    random_actions = random_actions.to(states.device)
    
    # 计算所有采样动作的 Q 值
    states_repeat = states.unsqueeze(1).repeat(1, num_samples, 1)
    q_rand = q_network(
        states_repeat.view(-1, state_dim),
        random_actions.view(-1, action_dim)
    ).view(batch_size, num_samples)
    
    # logsumexp
    cql_loss = torch.logsumexp(q_rand, dim=1).mean()
    
    # 减去数据中动作的 Q 值
    cql_loss -= current_q.mean()
    
    # (5) 总损失
    loss = td_loss + alpha * cql_loss
    return loss
```

### 21.4.3 理论保证

**定理**（Kumar et al., 2020）：
CQL 学习的 Q 函数是真实 Q 函数的下界（在一定条件下）。

$$
Q_{\text{CQL}}(s, a) \leq Q^{\pi^*}(s, a) + \epsilon
$$

**好处**：避免 Q 值高估 → 策略保守但安全。

---

## 21.5 Implicit Q-Learning (IQL)

通过期望值学习避免 OOD 查询。

### 21.5.1 期望值学习

**核心思想**：不使用 $\max_a Q(s, a)$，而是学习期望值。

**IQL 值函数**：

$$
V(s) = \mathbb{E}_{a \sim \pi_\beta}[\text{Advantage}(s, a)^+]
$$

其中 $[\cdot]^+ = \max(0, \cdot)$ 是正部。

### 21.5.2 避免 OOD 查询

**关键**：IQL 不需要 $\max_a Q(s, a)$ → 不会查询 OOD 动作。

**训练流程**：

1. 学习 Q 函数
2. 学习 V 函数（期望值）
3. 策略通过 AWR (Advantage Weighted Regression) 学习

### 21.5.3 简单高效

**实现**：

```python
class IQL:
    """
    Implicit Q-Learning
    
    三个网络：
    - Q: Q 函数
    - V: 值函数
    - π: 策略
    """
    def __init__(self, state_dim, action_dim):
        self.q_net = QNetwork(state_dim, action_dim)
        self.v_net = VNetwork(state_dim)
        self.policy = Policy(state_dim, action_dim)
        
        self.expectile = 0.7  # τ for expectile regression
    
    def train_step(self, states, actions, rewards, next_states, dones):
        # (1) 训练 V 网络（expectile regression）
        q_values = self.q_net(states, actions).detach()
        v_values = self.v_net(states)
        
        v_loss = self.expectile_loss(q_values - v_values, self.expectile)
        
        # (2) 训练 Q 网络
        with torch.no_grad():
            target_v = self.v_net(next_states)
            target_q = rewards + (1 - dones) * gamma * target_v
        
        q_loss = (self.q_net(states, actions) - target_q).pow(2).mean()
        
        # (3) 训练策略（AWR）
        advantage = q_values - v_values
        weights = torch.exp(advantage / beta).clamp(max=100.0)
        
        log_prob = self.policy.log_prob(states, actions)
        policy_loss = -(weights * log_prob).mean()
        
        return v_loss, q_loss, policy_loss
    
    @staticmethod
    def expectile_loss(diff, expectile=0.7):
        """Expectile regression loss"""
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return (weight * diff.pow(2)).mean()
```

---

## 21.6 Decision Transformer

将 Offline RL 视为序列建模问题。

<div data-component="DecisionTransformerArchitecture"></div>

### 21.6.1 序列建模视角

**核心思想**：将 RL 轨迹视为序列，用 Transformer 建模。

**序列**：

$$
(\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \ldots)
$$

其中 $\hat{R}_t = \sum_{t'=t}^T r_{t'}$ 是 return-to-go。

### 21.6.2 Transformer 架构

**输入嵌入**：
- Return: MLP($\hat{R}_t$)
- State: MLP($s_t$)
- Action: MLP($a_t$)

**Transformer**：自注意力处理序列

**输出**：预测下一个动作 $a_t$

### 21.6.3 Return-Conditioned Policy

**关键**：以期望回报为条件生成动作。

$$
a_t = \text{Transformer}(s_t, a_{t-1}, \hat{R}_t, \ldots)
$$

**测试时**：设置高 return → 生成高回报轨迹。

**实现**：

```python
class DecisionTransformer(nn.Module):
    """
    Decision Transformer for Offline RL
    
    Args:
        state_dim: 状态维度
        action_dim: 动作维度
        hidden_dim: 隐藏层维度
        num_layers: Transformer 层数
        num_heads: 注意力头数
        max_length: 最大序列长度
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, 
                 num_layers=3, num_heads=1, max_length=20):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        
        # 嵌入层
        self.embed_return = nn.Linear(1, hidden_dim)
        self.embed_state = nn.Linear(state_dim, hidden_dim)
        self.embed_action = nn.Linear(action_dim, hidden_dim)
        
        # 位置编码
        self.embed_timestep = nn.Embedding(max_length, hidden_dim)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4*hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 输出层
        self.predict_action = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, returns, states, actions, timesteps):
        """
        前向传播
        
        Args:
            returns: [batch, seq_len, 1]
            states: [batch, seq_len, state_dim]
            actions: [batch, seq_len, action_dim]
            timesteps: [batch, seq_len]
        
        Returns:
            action_preds: [batch, seq_len, action_dim]
        """
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        # 嵌入
        return_embeddings = self.embed_return(returns)
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        time_embeddings = self.embed_timestep(timesteps)
        
        # 交错排列 (R, s, a)
        # [batch, seq_len*3, hidden_dim]
        sequence = torch.zeros(
            batch_size, seq_len * 3, self.hidden_dim,
            device=states.device
        )
        sequence[:, 0::3, :] = return_embeddings + time_embeddings
        sequence[:, 1::3, :] = state_embeddings + time_embeddings
        sequence[:, 2::3, :] = action_embeddings + time_embeddings
        
        # Transformer
        output = self.transformer(sequence)
        
        # 预测动作（使用状态嵌入的输出）
        action_preds = self.predict_action(output[:, 1::3, :])
        
        return action_preds
```

---

## 21.7 数据集质量

Offline RL 性能高度依赖数据集质量。

### 21.7.1 D4RL Benchmark

**D4RL**（Datasets for Deep Data-Driven RL）：标准 Offline RL 基准。

**数据集类型**：
- **Expert**：专家数据
- **Medium**：中等水平
- **Random**：随机策略
- **Medium-Replay**：训练中的回放缓冲区

### 21.7.2 数据多样性

**重要性**：数据覆盖的状态-动作空间越广，性能越好。

**问题**：单一策略数据覆盖有限。

### 21.7.3 数据增强

**方法**：
- 合成新轨迹
- 混合不同策略数据
- 对抗扰动

---

## 本章小结

在本章中，我们学习了：

✅ **Offline RL 动机**：利用历史数据、避免在线交互  
✅ **挑战**：OOD 动作、外推误差、Deadly Triad  
✅ **BCQ**：批量约束、VAE 行为克隆  
✅ **CQL**：保守 Q 学习、Q 值下界  
✅ **IQL**：隐式 Q 学习、避免 OOD 查询  
✅ **Decision Transformer**：序列建模、Return-conditioned  

> [!TIP]
> **核心要点**：
> - Offline RL 从静态数据集学习，无需环境交互
> - 主要挑战是 OOD 动作的外推误差
> - BCQ 通过限制动作在数据分布内避免 OOD
> - CQL 学习保守的 Q 函数下界
> - IQL 通过期望值学习完全避免 max 操作
> - Decision Transformer 用序列模型处理 RL
> - 数据集质量对 Offline RL 性能至关重要

> [!NOTE]
> **后续章节将涵盖**：
> - Chapter 22: 多任务与迁移学习
> - Chapter 23: 元强化学习
> - 大模型时代的 RL（RLHF、DPO）

---

## 扩展阅读

- **经典论文**：
  - Fujimoto et al. (2019): Off-Policy Deep RL without Exploration (BCQ)
  - Kumar et al. (2020): Conservative Q-Learning for Offline RL (CQL)
  - Kostrikov et al. (2022): Offline RL via Supervised Learning (IQL)
  - Chen et al. (2021): Decision Transformer: Reinforcement Learning via Sequence Modeling
  - Fujimoto & Gu (2021): A Minimalist Approach to Offline RL (TD3+BC)
- **理论基础**：
  - Levine et al. (2020): Offline Reinforcement Learning: Tutorial, Review, and Perspectives
- **数据集**：
  - Fu et al. (2020): D4RL: Datasets for Deep Data-Driven RL
  - https://github.com/Farama-Foundation/D4RL

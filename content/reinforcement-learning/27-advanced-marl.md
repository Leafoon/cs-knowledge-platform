---
title: "Chapter 27. Advanced Multi-Agent RL"
description: "值分解、MAPPO与高级多智能体算法"
updated: "2026-01-29"
---

> **Learning Objectives**
> * 理解值分解方法（VDN, QMIX）
> * 掌握 MAPPO 算法
> * 学习 MADDPG 方法
> * 了解 Mean Field RL
> * 掌握 GNN 在 MARL 中的应用

---

## 27.1 Value Decomposition

将联合 Q 值分解为每个智能体的贡献。

<div data-component="ValueDecompositionComparison"></div>

### 27.1.1 VDN (Value Decomposition Networks)

**核心思想**：联合 Q 值是各智能体 Q 值之和。

$$
Q_{\text{tot}}(\mathbf{s}, \mathbf{a}) = \sum_{i=1}^N Q^i(s, a^i)
$$

**优势**：
- ✅ 分散执行：智能体 $i$ 只需 $\arg\max_{a^i} Q^i$
- ✅ 简单实现

**局限**：
- ❌ 表达能力有限（仅可加性）

**实现**：

```python
class VDN(nn.Module):
    """
    Value Decomposition Networks
    
    Args:
        num_agents: 智能体数量
        obs_dim: 观测维度
        action_dim: 动作维度
    """
    def __init__(self, num_agents, obs_dim, action_dim):
        super().__init__()
        self.num_agents = num_agents
        
        # 每个智能体的Q网络
        self.agent_q_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            )
            for _ in range(num_agents)
        ])
    
    def forward(self, observations, actions):
        """
        计算联合Q值
        
        Args:
            observations: [batch, num_agents, obs_dim]
            actions: [batch, num_agents] (discrete)
        
        Returns:
            q_tot: [batch] 联合Q值
        """
        batch_size = observations.shape[0]
        
        # 每个智能体的Q值
        individual_q_values = []
        for i in range(self.num_agents):
            q_values = self.agent_q_networks[i](observations[:, i])  # [batch, action_dim]
            
            # 选择执行动作的Q值
            q_i = q_values.gather(1, actions[:, i].unsqueeze(1)).squeeze(1)  # [batch]
            individual_q_values.append(q_i)
        
        # VDN: 求和
        q_tot = sum(individual_q_values)
        
        return q_tot
```

### 27.1.2 QMIX

**改进**：非线性混合，保持单调性。

<div data-component="QMIXMixingNetwork"></div>

**混合网络**：

$$
Q_{\text{tot}}(\mathbf{s}, \mathbf{a}) = f_{\text{mix}}(Q^1, Q^2, \ldots, Q^N; s)
$$

**单调性约束**：

$$
\frac{\partial Q_{\text{tot}}}{\partial Q^i} \geq 0 \quad \forall i
$$

**实现**：

```python
class QMIXMixingNetwork(nn.Module):
    """
    QMIX Mixing Network
    
    非线性混合个体Q值，保持单调性
    """
    def __init__(self, num_agents, state_dim, mixing_embed_dim=32):
        super().__init__()
        self.num_agents = num_agents
        
        # 超网络：从状态生成混合网络的权重
        # 第一层权重
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, num_agents * mixing_embed_dim)
        )
        
        # 第二层权重
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, mixing_embed_dim)
        )
        
        # 偏置
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )
    
    def forward(self, agent_q_values, state):
        """
        混合个体Q值
        
        Args:
            agent_q_values: [batch, num_agents]
            state: [batch, state_dim]
        
        Returns:
            q_tot: [batch, 1]
        """
        batch_size = agent_q_values.shape[0]
        
        # 生成权重（确保非负，保持单调性）
        w1 = torch.abs(self.hyper_w1(state))  # [batch, num_agents * mixing_embed_dim]
        w1 = w1.view(batch_size, self.num_agents, -1)  # [batch, num_agents, mixing_embed_dim]
        
        b1 = self.hyper_b1(state)  # [batch, mixing_embed_dim]
        
        # 第一层混合
        hidden = F.elu(torch.bmm(agent_q_values.unsqueeze(1), w1).squeeze(1) + b1)
        # [batch, mixing_embed_dim]
        
        # 第二层权重
        w2 = torch.abs(self.hyper_w2(state))  # [batch, mixing_embed_dim]
        b2 = self.hyper_b2(state)  # [batch, 1]
        
        # 第二层混合
        q_tot = (hidden * w2).sum(dim=1, keepdim=True) + b2  # [batch, 1]
        
        return q_tot


class QMIX(nn.Module):
    """
    完整QMIX算法
    """
    def __init__(self, num_agents, obs_dim, action_dim, state_dim):
        super().__init__()
        
        # 个体Q网络
        self.agent_q_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            )
            for _ in range(num_agents)
        ])
        
        # QMIX混合网络
        self.mixing_network = QMIXMixingNetwork(num_agents, state_dim)
    
    def forward(self, observations, actions, state):
        """
        计算联合Q值
        
        Args:
            observations: [batch, num_agents, obs_dim]
            actions: [batch, num_agents]
            state: [batch, state_dim]
        
        Returns:
            q_tot: [batch, 1]
        """
        # 计算每个智能体的Q值
        agent_q_values = []
        for i in range(len(self.agent_q_networks)):
            q_i = self.agent_q_networks[i](observations[:, i])
            q_i_selected = q_i.gather(1, actions[:, i].unsqueeze(1)).squeeze(1)
            agent_q_values.append(q_i_selected)
        
        agent_q_values = torch.stack(agent_q_values, dim=1)  # [batch, num_agents]
        
        # QMIX混合
        q_tot = self.mixing_network(agent_q_values, state)
        
        return q_tot
```

### 27.1.3 QTRAN

**QTRAN**：更通用的值分解，放松单调性约束。

### 27.1.4 可加性 vs 单调性

**VDN 可加性**：$Q_{\text{tot}} = \sum Q^i$

**QMIX 单调性**：$\frac{\partial Q_{\text{tot}}}{\partial Q^i} \geq 0$

**权衡**：
- VDN 简单但表达力弱
- QMIX 更强但复杂

---

## 27.2 MAPPO (Multi-Agent PPO)

将 PPO 扩展到多智能体。

<div data-component="MAPPOArchitecture"></div>

### 27.2.1 集中式 Critic

**Critic 输入**：全局状态 + 所有动作

$$
V(\mathbf{s}) \quad \text{or} \quad Q(\mathbf{s}, \mathbf{a})
$$

### 27.2.2 分散式 Actor

**Actor 输入**：局部观测

$$
\pi^i(a^i | o^i)
$$

### 27.2.3 参数共享

**共享策略**：所有智能体共享参数 → 可扩展性。

**实现**：

```python
class MAPPO:
    """
    Multi-Agent PPO
    
    集中式Critic + 分散式Actor
    """
    def __init__(self, num_agents, obs_dim, action_dim, state_dim):
        # 共享Actor（所有智能体共享参数）
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # 集中式Critic（输入全局状态）
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
    
    def select_actions(self, observations):
        """
        每个智能体选择动作
        
        Args:
            observations: [num_agents, obs_dim]
        
        Returns:
            actions, log_probs
        """
        logits = self.actor(observations)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs
    
    def update(self, rollout_buffer):
        """
        PPO更新
        
        Args:
            rollout_buffer: 包含 (obs, actions, rewards, states, ...)
        """
        for epoch in range(ppo_epochs):
            for batch in rollout_buffer:
                obs, actions, old_log_probs, returns, advantages, states = batch
                
                # Critic更新
                values = self.critic(states).squeeze()
                critic_loss = F.mse_loss(values, returns)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                
                # Actor更新
                logits = self.actor(obs)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions)
                
                # PPO ratio
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
```

---

## 27.3 MADDPG

Multi-Agent DDPG。

### 27.3.1 集中式 Critic 输入所有观测

**Critic** for 智能体 $i$：

$$
Q^i(\mathbf{o}, \mathbf{a}) = Q^i(o^1, \ldots, o^N, a^1, \ldots, a^N)
$$

### 27.3.2 分散式 Actor

**Actor** for 智能体 $i$：

$$
\pi^i(o^i)
$$

### 27.3.3 混合合作-竞争

**适用于**：既有合作又有竞争的环境。

**示例**：多智能体粒子环境

---

## 27.4 Mean Field RL

大规模多智能体的平均场近似。

<div data-component="MeanFieldApproximation"></div>

### 27.4.1 大规模多智能体

**挑战**：$N$ 很大（成百上千）

**例子**：
- 交通仿真（数千辆车）
- 人群仿真

### 27.4.2 平均场近似

**核心思想**：用平均动作代替其他智能体的动作。

$$
Q^i(s, a^i, \bar{a}) \approx Q^i(s, a^i, a^1, \ldots, a^N)
$$

其中 $\bar{a} = \frac{1}{N} \sum_j a^j$ 是平均动作。

### 27.4.3 可扩展性

**优势**：
- ✅ 复杂度从 $O(N \times |A|^N)$ 降到 $O(N \times |A|)$
- ✅ 适用于大规模系统

---

## 27.5 Graph Neural Networks for MARL

使用 GNN 建模智能体关系。

### 27.5.1 关系建模

**图表示**：
- 节点：智能体
- 边：关系（通信、协作）

**GNN**：
- 消息传递
- 图卷积

### 27.5.2 动态拓扑

**拓扑变化**：智能体关系动态变化。

**方法**：
- 注意力机制
- 动态图

### 27.5.3 消息传递

**GNN 消息传递**：

```
1. 每个节点聚合邻居信息
2. 更新节点表示
3. 重复 k 轮
```

---

## 本章小结

在本章中，我们学习了：

✅ **Value Decomposition**：VDN、QMIX、QTRAN、可加性 vs 单调性  
✅ **MAPPO**：集中式 Critic、分散式 Actor、参数共享  
✅ **MADDPG**：集中训练、混合合作竞争  
✅ **Mean Field RL**：平均场近似、大规模可扩展性  
✅ **GNN for MARL**：关系建模、动态拓扑、消息传递  

> [!TIP]
> **核心要点**：
> - 值分解允许分散执行同时保持协调
> - VDN简单但表达力有限（可加性）
> - QMIX通过单调性约束提升表达力
> - MAPPO是MARL中最实用的算法之一
> - 参数共享提升可扩展性
> - Mean Field RL适用于大规模系统
> - GNN可以建模智能体间的复杂关系
> - CTDE是MARL的核心设计模式

> [!NOTE]
> **后续章节将涵盖**：
> - Chapter 28: Self-Play与涌现行为
> - Chapter 29: 合作多智能体任务
> - 更多前沿主题

---

## 扩展阅读

- **经典论文**：
  - Sunehag et al. (2018): Value-Decomposition Networks For Cooperative Multi-Agent Learning (VDN)
  - Rashid et al. (2018): QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent RL
  - Yu et al. (2022): The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)
  - Lowe et al. (2017): Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)
  - Yang et al. (2018): Mean Field Multi-Agent RL
- **GNN for MARL**：
  - Jiang et al. (2020): Graph Convolutional RL
  - Agarwal et al. (2021): Learning to Communicate with Deep Multi-Agent RL
- **实现资源**：
  - PyMARL: Python MARL library (VDN, QMIX)
  - EPyMARL: Extended PyMARL
  - SMAC: StarCraft Multi-Agent Challenge
  - PettingZoo: Multi-Agent RL environments

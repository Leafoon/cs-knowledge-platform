---
title: "Chapter 26. Multi-Agent RL Foundations"
description: "多智能体强化学习基础"
updated: "2026-01-29"
---

> **Learning Objectives**
> * 理解多智能体 RL 问题定义
> * 掌握博弈论基础概念
> * 学习独立学习的挑战
> * 了解 CTDE 架构
> * 掌握智能体通信机制

---

## 26.1 MARL 问题定义

多个智能体在同一环境中交互学习。

<div data-component="MARLProblemVisualization"></div>

### 26.1.1 多智能体 MDP (MMDP)

**定义**：

$$
\text{MMDP} = (N, S, \{A^i\}_{i=1}^N, P, \{R^i\}_{i=1}^N)
$$

其中：
- $N$：智能体数量
- $S$：状态空间
- $A^i$：智能体 $i$ 的动作空间
- $P(s' | s, a^1, \ldots, a^N)$：转移概率
- $R^i(s, a^1, \ldots, a^N)$：智能体 $i$ 的奖励

**联合动作**：$\mathbf{a} = (a^1, a^2, \ldots, a^N)$

### 26.1.2 部分可观测性

**Dec-POMDP** (Decentralized POMDP)：

- 每个智能体只能观测部分状态
- $o^i = O^i(s)$：智能体 $i$ 的观测

**挑战**：
- 信息不完整
- 协调困难

### 26.1.3 通信与协作

**通信**：智能体之间交换信息。

**协作**：共享目标，合作最大化团队奖励。

---

## 26.2 博弈论基础

MARL 与博弈论密切相关。

<div data-component="NashEquilibriumDemo"></div>

### 26.2.1 Nash 均衡

**定义**：策略组合 $(\pi^1, \ldots, \pi^N)$ 是 Nash 均衡，如果没有智能体单方面改变策略能获得更高回报。

$$
J^i(\pi^1, \ldots, \pi^i, \ldots, \pi^N) \geq J^i(\pi^1, \ldots, \pi'^i, \ldots, \pi^N) \quad \forall i, \pi'^i
$$

**示例**：囚徒困境

| | 合作 | 背叛 |
|---|---|---|
| **合作** | (3, 3) | (0, 5) |
| **背叛** | (5, 0) | (1, 1) |

Nash 均衡：(背叛, 背叛)

### 26.2.2 零和游戏 vs 合作游戏

**零和游戏**：

$$
\sum_i R^i = 0
$$

**合作游戏**：

$$
R^1 = R^2 = \cdots = R^N \quad \text{(共享奖励)}
$$

### 26.2.3 Pareto 最优

**Pareto 最优**：无法在不损害任何智能体的情况下改进。

**关系**：
- Nash 均衡不一定是 Pareto 最优
- 例如囚徒困境：(背叛, 背叛) 是 Nash 但不是 Pareto

---

## 26.3 独立学习

每个智能体独立学习，无协调。

### 26.3.1 Independent Q-Learning

**方法**：每个智能体使用 Q-learning，忽略其他智能体。

**算法**：

```python
class IndependentQLearning:
    """
    独立Q-learning for MARL
    
    每个智能体独立学习Q函数
    """
    def __init__(self, num_agents, state_dim, action_dim):
        self.num_agents = num_agents
        
        # 每个智能体的Q表
        self.Q = [
            {}  # Q^i[state][action]
            for _ in range(num_agents)
        ]
    
    def select_actions(self, state):
        """
        每个智能体独立选择动作
        
        Args:
            state: 全局状态
        
        Returns:
            actions: [a^1, a^2, ..., a^N]
        """
        actions = []
        for i in range(self.num_agents):
            # ε-greedy
            if np.random.rand() < epsilon:
                action = np.random.randint(action_dim)
            else:
                # 贪心选择
                q_values = [self.Q[i].get((state, a), 0.0) 
                           for a in range(action_dim)]
                action = np.argmax(q_values)
            actions.append(action)
        return actions
    
    def update(self, state, actions, rewards, next_state, done):
        """
        独立更新每个智能体的Q值
        
        Args:
            state: 当前状态
            actions: [a^1, ..., a^N]
            rewards: [r^1, ..., r^N]
            next_state: 下一个状态
            done: 是否结束
        """
        for i in range(self.num_agents):
            # 智能体i的Q更新
            if (state, actions[i]) not in self.Q[i]:
                self.Q[i][(state, actions[i])] = 0.0
            
            if not done:
                # 下一个状态的最大Q值
                next_q_values = [
                    self.Q[i].get((next_state, a), 0.0)
                    for a in range(action_dim)
                ]
                max_next_q = max(next_q_values)
            else:
                max_next_q = 0.0
            
            # TD更新
            target = rewards[i] + gamma * max_next_q
            self.Q[i][(state, actions[i])] += alpha * (
                target - self.Q[i][(state, actions[i])]
            )
```

### 26.3.2 非平稳性问题

**挑战**：其他智能体的策略在变化 → 环境非平稳。

**后果**：
- 违反 MDP 假设
- 收敛性无保证

### 26.3.3 收敛性挑战

**问题**：独立学习不一定收敛到 Nash 均衡。

**示例**：Matching Pennies（配对硬币）

| | 正面 | 反面 |
|---|---|---|
| **正面** | (1, -1) | (-1, 1) |
| **反面** | (-1, 1) | (1, -1) |

独立Q-learning 可能循环，不收敛。

---

## 26.4 集中训练分散执行（CTDE）

训练时集中信息，执行时分散。

<div data-component="CTDEArchitecture"></div>

### 26.4.1 架构设计

**CTDE 原理**：

- **训练**：Critic 可以访问全局信息
- **执行**：Actor 只用局部观测

**优势**：
- 训练时利用全局信息
- 执行时满足分散约束

### 26.4.2 信息共享

**训练时 Critic 输入**：
- 全局状态 $s$
- 所有智能体的动作 $\mathbf{a} = (a^1, \ldots, a^N)$

**执行时 Actor 输入**：
- 局部观测 $o^i$

### 26.4.3 可扩展性

**挑战**：智能体数量增加 → 动作空间指数增长

**解决**：
- 参数共享
- 值分解

---

## 26.5 通信机制

智能体之间交换信息。

<div data-component="AgentCommunication"></div>

### 26.5.1 显式通信

**方法**：智能体发送消息。

**通信协议**：
- 广播：发送给所有智能体
- 点对点：发送给特定智能体

**实现**：

```python
class CommunicationModule(nn.Module):
    """
    显式通信模块
    
    智能体生成消息并广播
    """
    def __init__(self, observation_dim, message_dim):
        super().__init__()
        
        # 消息生成网络
        self.message_encoder = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, message_dim)
        )
        
        # 消息处理网络
        self.message_processor = nn.Sequential(
            nn.Linear(message_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    
    def generate_message(self, observation):
        """
        生成消息
        
        Args:
            observation: [batch, obs_dim]
        
        Returns:
            message: [batch, message_dim]
        """
        return self.message_encoder(observation)
    
    def aggregate_messages(self, messages):
        """
        聚合接收的消息
        
        Args:
            messages: [num_agents, batch, message_dim]
        
        Returns:
            aggregated: [batch, 64]
        """
        # 平均池化
        avg_message = messages.mean(dim=0)
        return self.message_processor(avg_message)


class MultiAgentWithComm:
    """
    带通信的多智能体系统
    """
    def __init__(self, num_agents, obs_dim, action_dim, message_dim=16):
        self.num_agents = num_agents
        
        # 每个智能体的通信模块
        self.comm_modules = [
            CommunicationModule(obs_dim, message_dim)
            for _ in range(num_agents)
        ]
        
        # 每个智能体的策略（输入包含聚合消息）
        self.policies = [
            nn.Sequential(
                nn.Linear(obs_dim + 64, 128),  # obs + aggregated message
                nn.ReLU(),
                nn.Linear(128, action_dim)
            )
            for _ in range(num_agents)
        ]
    
    def forward(self, observations):
        """
        前向传播（含通信）
        
        Args:
            observations: [num_agents, batch, obs_dim]
        
        Returns:
            actions: [num_agents, batch, action_dim]
        """
        # (1) 每个智能体生成消息
        messages = []
        for i in range(self.num_agents):
            msg = self.comm_modules[i].generate_message(observations[i])
            messages.append(msg)
        
        messages = torch.stack(messages)  # [num_agents, batch, message_dim]
        
        # (2) 每个智能体聚合消息
        actions = []
        for i in range(self.num_agents):
            aggregated_msg = self.comm_modules[i].aggregate_messages(messages)
            
            # (3) 拼接观测和消息
            input_with_comm = torch.cat([observations[i], aggregated_msg], dim=-1)
            
            # (4) 策略输出动作
            action = self.policies[i](input_with_comm)
            actions.append(action)
        
        return torch.stack(actions)
```

### 26.5.2 隐式协调

**方法**：无显式消息，通过观测他人行为协调。

**示例**：
- 观测历史动作
- 学习他人策略

### 26.5.3 CommNet、TarMAC

**CommNet** (Sukhbaatar et al., 2016)：
- 连续通信
- 消息通过网络传递

**TarMAC** (Das et al., 2019)：
- 目标驱动的注意力通信
- 选择性听取消息

---

## 本章小结

在本章中，我们学习了：

✅ **MARL 问题定义**：MMDP、部分可观测、通信协作  
✅ **博弈论基础**：Nash 均衡、零和游戏、Pareto 最优  
✅ **独立学习**：Independent Q-Learning、非平稳性、收敛挑战  
✅ **CTDE**：集中训练分散执行、信息共享、可扩展性  
✅ **通信机制**：显式通信、隐式协调、CommNet、TarMAC  

> [!TIP]
> **核心要点**：
> - MARL处理多个智能体在同一环境中交互
> - 多智能体环境对单个智能体是非平稳的
> - Nash均衡是MARL的重要解概念
> - 独立学习简单但收敛性差
> - CTDE利用训练时全局信息，执行时分散
> - 通信可以显著提升协作性能
> - 参数共享和值分解提升可扩展性

> [!NOTE]
> **下一步**：
> Chapter 27 将学习**高级多智能体算法**：
> - Value Decomposition (VDN, QMIX)
> - MAPPO
> - MADDPG
> - Mean Field RL
> 
> 进入 [Chapter 27. Advanced MARL](27-advanced-marl.md)

---

## 扩展阅读

- **经典论文**：
  - Busoniu et al. (2008): A Comprehensive Survey of Multi-Agent RL
  - Lowe et al. (2017): Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)
  - Foerster et al. (2018): Counterfactual Multi-Agent Policy Gradients
  - Sukhbaatar et al. (2016): Learning Multiagent Communication with Backpropagation (CommNet)
- **博弈论**：
  - Shoham & Leyton-Brown (2009): Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations
- **实现资源**：
  - PettingZoo: Multi-Agent RL environments
  - SMAC: StarCraft Multi-Agent Challenge
  - EPyMARL: PyTorch MARL library

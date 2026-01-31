---
title: "Chapter 23. Meta-Reinforcement Learning"
description: "学会学习：快速适应新任务"
updated: "2026-01-29"
---

> **Learning Objectives**
> * 理解元学习（Meta-Learning）的概念
> * 掌握 MAML 的二阶优化方法
> * 学习 PEARL 的概率嵌入方法
> * 了解 RL² 的隐式适应
> * 掌握 Few-Shot RL 应用

---

## 23.1 元学习概念

元强化学习：学习如何快速学习新任务。

<div data-component="MetaLearningConcept"></div>

### 23.1.1 Learning to Learn

**传统 RL**：每个任务从零开始学习

**Meta-RL**：在任务分布上学习，快速适应新任务

**类比**：
- 传统 RL = 学习一门外语
- Meta-RL = 学习如何学习语言

### 23.1.2 任务分布

**关键概念**：任务不是孤立的，而是来自一个分布。

$$
\mathcal{T} \sim p(\mathcal{T})
$$

<div data-component="TaskDistributionSampling"></div>

**示例任务分布**：
- 机器人：不同重量的物体抓取
- 游戏：不同地图布局
- 推荐：不同用户偏好

**训练**：
1. 从分布采样任务 $\mathcal{T}_i \sim p(\mathcal{T})$
2. 在 $\mathcal{T}_i$ 上学习
3. 评估在新任务 $\mathcal{T}_{\text{test}}$ 的适应速度

### 23.1.3 快速适应

**目标**：用少量数据快速适应新任务。

**Few-Shot RL**：
- 5-shot：5 条轨迹
- 1-shot：1 条轨迹
- Zero-shot：0 条轨迹（直接泛化）

---

## 23.2 MAML (Model-Agnostic Meta-Learning)

通过二阶优化学习良好的初始化。

<div data-component="MAMLInnerOuterLoop"></div>

### 23.2.1 二阶优化

**核心思想**：学习一个初始参数 $\theta$，使得少量梯度步就能适应新任务。

**目标**：

$$
\min_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(\theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta))
$$

**解释**：
- 内循环：在任务 $\mathcal{T}_i$ 上梯度更新
- 外循环：优化初始参数 $\theta$

### 23.2.2 内循环 vs 外循环

**内循环**（Task-specific adaptation）：

$$
\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta)
$$

在单个任务上快速适应。

**外循环**（Meta-update）：

$$
\theta \leftarrow \theta - \beta \nabla_\theta \sum_i \mathcal{L}_{\mathcal{T}_i}(\theta'_i)
$$

更新元参数。

**算法流程**：

```
1. 初始化元参数 θ
2. for meta-iteration:
    3. 采样一批任务 {T₁, T₂, ..., Tₙ}
    4. for 每个任务 Tᵢ:
        5. (内循环) 收集数据，计算 θ'ᵢ = θ - α∇L(θ)
        6. (评估) 在 θ'ᵢ 上评估
    7. (外循环) 元更新 θ ← θ - β∇Σ L(θ'ᵢ)
```

### 23.2.3 RL-MAML

**MAML for RL** (Finn et al., 2017)：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import copy

class MAML_RL:
    """
    MAML for Reinforcement Learning
    
    Args:
        policy: 策略网络
        inner_lr: 内循环学习率
        outer_lr: 外循环学习率（元学习率）
        num_inner_steps: 内循环步数
    """
    def __init__(self, policy, inner_lr=0.01, outer_lr=0.001, num_inner_steps=1):
        self.policy = policy
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        
        # 元优化器
        self.meta_optimizer = optim.Adam(policy.parameters(), lr=outer_lr)
    
    def inner_loop(self, task_data, policy):
        """
        内循环：在单个任务上适应
        
        Args:
            task_data: [(state, action, reward), ...]
            policy: 当前策略（会被修改）
        
        Returns:
            adapted_policy: 适应后的策略
        """
        # 复制策略（避免修改原始策略）
        adapted_policy = copy.deepcopy(policy)
        
        # 在任务数据上执行梯度步
        for step in range(self.num_inner_steps):
            # 计算策略梯度损失
            loss = 0
            for state, action, reward in task_data:
                log_prob = adapted_policy.log_prob(state, action)
                loss -= log_prob * reward  # REINFORCE
            
            # 手动梯度更新（一阶近似）
            grads = torch.autograd.grad(
                loss, adapted_policy.parameters(), create_graph=True
            )
            
            # 更新参数
            for param, grad in zip(adapted_policy.parameters(), grads):
                param.data = param.data - self.inner_lr * grad
        
        return adapted_policy
    
    def meta_train_step(self, task_batch):
        """
        元训练步：在一批任务上更新元参数
        
        Args:
            task_batch: [task1_data, task2_data, ...]
        """
        meta_loss = 0
        
        # 对每个任务
        for task_train_data, task_test_data in task_batch:
            # (1) 内循环：在训练数据上适应
            adapted_policy = self.inner_loop(task_train_data, self.policy)
            
            # (2) 在测试数据上评估
            task_loss = 0
            for state, action, reward in task_test_data:
                log_prob = adapted_policy.log_prob(state, action)
                task_loss -= log_prob * reward
            
            meta_loss += task_loss
        
        # (3) 外循环：元更新
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def adapt_to_new_task(self, new_task_data):
        """
        适应到新任务
        
        Args:
            new_task_data: 新任务的少量数据
        
        Returns:
            adapted_policy: 适应后的策略
        """
        return self.inner_loop(new_task_data, self.policy)
```

**使用示例**：

```python
# 初始化
policy = PolicyNetwork(state_dim, action_dim)
maml = MAML_RL(policy, inner_lr=0.01, outer_lr=0.001)

# 元训练
for meta_iter in range(1000):
    # 采样任务批次
    task_batch = sample_task_batch(num_tasks=5)
    
    # 元训练步
    loss = maml.meta_train_step(task_batch)
    print(f"Meta-iter {meta_iter}, Loss: {loss}")

# 适应到新任务
new_task_data = collect_data_from_new_task(num_trajectories=5)
adapted_policy = maml.adapt_to_new_task(new_task_data)
```

---

## 23.3 PEARL (Probabilistic Embeddings for Actor-Critic RL)

通过学习任务嵌入实现上下文条件策略。

<div data-component="PEARLArchitecture"></div>

### 23.3.1 任务推断

**核心思想**：从轨迹推断任务嵌入 $z$。

**概率模型**：

$$
z \sim q_\phi(z | \mathcal{C})
$$

其中 $\mathcal{C}$ 是上下文（历史轨迹）。

### 23.3.2 上下文编码器

**架构**：

```
Context = {(s₁,a₁,r₁,s'₁), ..., (sₙ,aₙ,rₙ,s'ₙ)}
    ↓
Context Encoder
    ↓
z ~ N(μ, σ²)
```

**实现**：

```python
class ContextEncoder(nn.Module):
    """
    上下文编码器（用于 PEARL）
    
    输入: 上下文 (s,a,r,s')
    输出: 任务嵌入 z
    """
    def __init__(self, state_dim, action_dim, latent_dim=16):
        super().__init__()
        
        input_dim = state_dim + action_dim + 1 + state_dim  # (s,a,r,s')
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(128, latent_dim)
        self.log_std = nn.Linear(128, latent_dim)
    
    def forward(self, context):
        """
        编码上下文
        
        Args:
            context: [batch, num_transitions, (s,a,r,s')]
        
        Returns:
            z: [batch, latent_dim]
        """
        # 聚合上下文
        h = self.encoder(context)
        h = h.mean(dim=1)  # 平均池化
        
        # 输出分布参数
        mean = self.mean(h)
        log_std = self.log_std(h)
        
        # 采样 z
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        
        return z, mean, std


class PEARLAgent:
    """
    PEARL 智能体
    """
    def __init__(self, state_dim, action_dim, latent_dim=16):
        # 上下文编码器
        self.context_encoder = ContextEncoder(
            state_dim, action_dim, latent_dim
        )
        
        # 条件策略（以 z 为条件）
        self.policy = nn.Sequential(
            nn.Linear(state_dim + latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # 条件 Q 函数
        self.q_network = nn.Sequential(
            nn.Linear(state_dim + action_dim + latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def infer_task(self, context):
        """从上下文推断任务"""
        z, mean, std = self.context_encoder(context)
        return z
    
    def select_action(self, state, z):
        """选择动作（条件在任务嵌入 z 上）"""
        state_z = torch.cat([state, z], dim=-1)
        action = self.policy(state_z)
        return action
```

### 23.3.3 变分推断

**ELBO 目标**：

$$
\max_{\phi, \theta} \mathbb{E}_{z \sim q_\phi(z|\mathcal{C})} \left[R(\tau) - \text{KL}(q_\phi(z|\mathcal{C}) || p(z))\right]
$$

**训练**：
- 最大化期望回报
- KL 散度正则化

---

## 23.4 RL²

使用 RNN 隐式学习适应算法。

### 23.4.1 RNN 作为元学习器

**核心思想**：RNN 的隐状态作为"学习算法"。

**输入序列**：

$$
(s_1, a_1, r_1, s_2, a_2, r_2, \ldots)
$$

**RNN 输出**：下一个动作 $a_t$

### 23.4.2 隐式适应

**不需要显式梯度更新**：RNN 通过隐状态自动适应。

**实现**：

```python
class RL2Agent(nn.Module):
    """
    RL² 智能体（使用 RNN）
    
    Args:
        state_dim: 状态维度
        action_dim: 动作维度
        hidden_dim: RNN 隐藏层维度
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # 输入: (state, previous_action, reward)
        input_dim = state_dim + action_dim + 1
        
        # GRU 层
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        # 策略头
        self.policy_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, sequence, hidden=None):
        """
        前向传播
        
        Args:
            sequence: [batch, seq_len, (s, a_prev, r)]
            hidden: RNN 隐状态
        
        Returns:
            actions: [batch, seq_len, action_dim]
            hidden: 新的隐状态
        """
        # RNN 处理序列
        output, hidden = self.gru(sequence, hidden)
        
        # 策略输出
        actions = self.policy_head(output)
        
        return actions, hidden
    
    def reset(self):
        """重置 RNN 隐状态"""
        return None  # GRU 会自动初始化
```

**训练**：在整个任务序列上用 PPO/TRPO。

---

## 23.5 应用场景

### 23.5.1 Few-Shot RL

**场景**：用少量交互快速学习新任务。

**示例**：
- 机器人：5 次演示学会新动作
- 游戏：1 次试玩适应新关卡

### 23.5.2 机器人快速适应

**场景**：机器人遭遇新环境/损伤后快速恢复。

**方法**：
- MAML：学习鲁棒初始化
- PEARL：推断环境参数

### 23.5.3 个性化推荐

**场景**：快速适应新用户偏好。

**方法**：
- Meta-RL on 用户分布
- Few-shot 适应个人偏好

---

## 本章小结

在本章中，我们学习了：

✅ **元学习概念**：Learning to Learn、任务分布、快速适应  
✅ **MAML**：二阶优化、内外循环、RL-MAML  
✅ **PEARL**：上下文编码、任务推断、变分推断  
✅ **RL²**：RNN 元学习器、隐式适应  
✅ **应用场景**：Few-Shot RL、机器人快速适应、个性化  

> [!TIP]
> **核心要点**：
> - Meta-RL 学习如何在任务分布上快速适应
> - MAML 通过二阶优化学习良好的初始化
> - 内循环在单任务上适应，外循环优化元参数
> - PEARL 通过上下文编码推断任务嵌入
> - RL² 使用 RNN 隐式学习适应算法
> - Few-Shot RL 是 Meta-RL 的关键应用
> - Meta-RL 特别适合需要快速适应的场景

> [!NOTE]
> **下一步**：
> Chapter 24 将学习**多目标强化学习**：
> - Pareto Front
> - Scalarization
> - Multi-Objective 权衡
> 
> 进入 [Chapter 24. Multi-Objective RL](24-multi-objective-rl.md)

---

## 扩展阅读

- **经典论文**：
  - Finn et al. (2017): Model-Agnostic Meta-Learning for Fast Adaptation (MAML)
  - Rakelly et al. (2019): Efficient Off-Policy Meta-RL via Probabilistic Context Variables (PEARL)
  - Duan et al. (2016): RL²: Fast Reinforcement Learning via Slow Reinforcement Learning
  - Nagabandi et al. (2019): Learning to Adapt in Dynamic, Real-World Environments
- **理论基础**：
  - Hospedales et al. (2020): Meta-Learning in Neural Networks: A Survey
- **实现资源**：
  - learn2learn: PyTorch Meta-Learning library
  - garage: Meta-RL implementations
  - rlkit: PEARL implementation

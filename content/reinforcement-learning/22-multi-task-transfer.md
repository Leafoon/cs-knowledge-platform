---
title: "Chapter 22. Multi-Task & Transfer Learning"
description: "多任务学习与知识迁移"
updated: "2026-01-29"
---

> **Learning Objectives**
> * 理解多任务强化学习的挑战与方法
> * 掌握迁移学习的技术
> * 学习 Zero-Shot Transfer 与 Successor Features
> * 了解 Curriculum Learning 的原理
> * 掌握实际应用场景

---

## 22.1 多任务 RL

在多个相关任务上同时学习，共享知识。

### 22.1.1 共享表示学习

**核心思想**：多个任务共享底层特征表示。

<div data-component="MultiTaskLearning"></div>

**架构**：

```
输入 state → 共享编码器 → 任务特定头 → 输出 (action/value)
```

**优势**：
- ✅ 知识共享：相关任务互相帮助
- ✅ 样本效率：利用所有任务的数据
- ✅ 泛化能力：学到更通用的表示

**实现**：

```python
import torch
import torch.nn as nn

class MultiTaskPolicy(nn.Module):
    """
    多任务策略网络
    
    Args:
        state_dim: 状态维度
        action_dim: 动作维度
        num_tasks: 任务数量
        shared_dim: 共享层维度
    """
    def __init__(self, state_dim, action_dim, num_tasks, shared_dim=256):
        super().__init__()
        
        # 共享编码器
        self.shared_encoder = nn.Sequential(
            nn.Linear(state_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU()
        )
        
        # 每个任务的特定头
        self.task_heads = nn.ModuleList([
            nn.Linear(shared_dim, action_dim)
            for _ in range(num_tasks)
        ])
    
    def forward(self, state, task_id):
        """
        前向传播
        
        Args:
            state: [batch, state_dim]
            task_id: int, 任务ID
        
        Returns:
            action_logits: [batch, action_dim]
        """
        # 共享特征提取
        shared_features = self.shared_encoder(state)
        
        # 任务特定输出
        action_logits = self.task_heads[task_id](shared_features)
        
        return action_logits


def train_multi_task(envs, num_tasks, num_episodes=1000):
    """
    多任务训练
    
    Args:
        envs: 环境列表 [env0, env1, ...]
        num_tasks: 任务数量
        num_episodes: 训练轮数
    """
    policy = MultiTaskPolicy(state_dim, action_dim, num_tasks)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    
    for episode in range(num_episodes):
        # 循环采样任务
        task_id = episode % num_tasks
        env = envs[task_id]
        
        # 收集轨迹
        state = env.reset()
        done = False
        rewards = []
        log_probs = []
        
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            logits = policy(state_tensor.unsqueeze(0), task_id)
            
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            next_state, reward, done, _ = env.step(action.item())
            
            rewards.append(reward)
            log_probs.append(log_prob)
            state = next_state
        
        # REINFORCE 更新
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        loss = -sum(log_p * R for log_p, R in zip(log_probs, returns))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 22.1.2 任务干扰问题

**挑战**：不同任务的学习可能相互干扰。

**负迁移**（Negative Transfer）：
- 任务 A 的学习损害任务 B 的性能
- 共享参数的冲突更新

**解决方案**：

1. **软模块化**（Soft Modularization）：
   - 学习任务相似度
   - 动态调整共享程度

2. **渐进式训练**：
   - 先训练简单任务
   - 逐步添加复杂任务

3. **任务特定归一化**：
   - 每个任务独立的 Batch Norm

### 22.1.3 Soft Modularization

**Distral**（Teh et al., 2017）：分布式强化学习。

**核心思想**：
- 学习一个中心策略（蒸馏策略）
- 每个任务策略受中心策略约束

**目标函数**：

$$
\mathcal{L}_i = \mathcal{L}_{\text{RL}}^i + \alpha \cdot \text{KL}(\pi_i || \pi_0)
$$

其中 $\pi_0$ 是中心策略。

---

## 22.2 迁移学习

从源任务迁移知识到目标任务。

<div data-component="TransferLearningFlow"></div>

### 22.2.1 源任务 → 目标任务

**场景**：
- 源任务：已训练好的策略
- 目标任务：新任务（相关但不同）

**迁移方式**：
1. 特征迁移：复用编码器
2. 策略迁移：fine-tune 策略
3. 值函数迁移：迁移 Q/V 函数

### 22.2.2 Fine-tuning 策略

**标准流程**：

```python
def fine_tune_policy(source_policy, target_env, fine_tune_steps=10000):
    """
    Fine-tune 预训练策略到目标任务
    
    Args:
        source_policy: 源任务训练的策略
        target_env: 目标环境
        fine_tune_steps: fine-tune 步数
    
    Returns:
        fine_tuned_policy: 调优后的策略
    """
    # (1) 加载源策略
    policy = source_policy.copy()
    
    # (2) 可选：冻结部分层
    # 例如冻结共享编码器，只训练任务头
    for param in policy.shared_encoder.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=1e-4  # 较小学习率
    )
    
    # (3) 在目标任务上训练
    for step in range(fine_tune_steps):
        state = target_env.reset()
        done = False
        
        while not done:
            action = policy.select_action(state)
            next_state, reward, done, _ = target_env.step(action)
            
            # PPO/SAC 更新
            policy.update(state, action, reward, next_state)
            state = next_state
    
    return policy
```

**技巧**：
- 使用较小的学习率
- 冻结低层特征
- 早停避免过拟合

### 22.2.3 Domain Randomization

**核心思想**：在源任务训练时引入随机化，增强泛化。

**方法**：
- 环境参数随机化（重力、摩擦）
- 外观随机化（颜色、纹理）
- 动力学随机化

**示例**（机器人）：

```python
import numpy as np

class RandomizedEnv:
    """
    Domain Randomization环境包装器
    """
    def __init__(self, base_env):
        self.base_env = base_env
    
    def reset(self):
        # 随机化环境参数
        self.base_env.set_gravity(
            np.random.uniform(8.0, 12.0)  # 重力: 8-12 m/s²
        )
        self.base_env.set_friction(
            np.random.uniform(0.3, 0.7)   # 摩擦系数
        )
        
        return self.base_env.reset()
    
    def step(self, action):
        return self.base_env.step(action)
```

**优势**：
- ✅ 提升鲁棒性
- ✅ Sim-to-Real 迁移
- ✅ 无需目标域数据

---

## 22.3 Zero-Shot Transfer

不需要在目标任务上训练，直接泛化。

### 22.3.1 任务泛化

**目标**：学习能泛化到未见任务的策略。

**方法**：
- 学习任务不变的表示
- 条件策略（conditioned policy）

### 22.3.2 Successor Features

**核心思想**：分解值函数为特征和权重。

**值函数分解**：

$$
Q(s, a) = \phi(s, a)^\top w
$$

其中：
- $\phi(s, a)$：Successor Features（SF）
- $w$：任务权重向量

**Successor Features 定义**：

$$
\phi(s, a) = \mathbb{E}\left[\sum_{t=0}^\infty \gamma^t \phi(s_t) \mid s_0=s, a_0=a\right]
$$

**迁移**：
- 学习 SF $\phi$ 在源任务
- 目标任务只需估计新的 $w$

**实现**：

```python
class SuccessorFeatureAgent:
    """
    Successor Features 智能体
    
    Args:
        feature_dim: 特征维度
        state_dim: 状态维度
        action_dim: 动作维度
    """
    def __init__(self, feature_dim, state_dim, action_dim):
        # SF 网络: (s,a) → φ(s,a)
        self.sf_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
        # 任务权重 w
        self.task_weight = nn.Parameter(torch.randn(feature_dim))
    
    def compute_q(self, state, action):
        """计算 Q 值"""
        sf = self.sf_network(torch.cat([state, action], dim=-1))
        q_value = (sf * self.task_weight).sum(dim=-1)
        return q_value
    
    def transfer_to_new_task(self, new_task_weight):
        """迁移到新任务：只更新权重"""
        self.task_weight = nn.Parameter(new_task_weight)
```

### 22.3.3 Universal Value Function Approximators (UVFA)

**思想**：值函数以目标为条件。

$$
Q(s, a, g) \quad \text{(goal-conditioned Q-function)}
$$

**训练**：在多个目标上训练

**测试**：指定新目标，直接泛化

---

## 22.4 Curriculum Learning

按难度递增的顺序学习任务。

<div data-component="CurriculumProgression"></div>

### 22.4.1 任务难度递增

**动机**：直接学习困难任务效率低。

**课程设计**：
1. 简单任务 → 中等 → 困难
2. 逐步增加复杂度

**示例**（机器人抓取）：
- Level 1：抓取固定位置物体
- Level 2：抓取随机位置物体
- Level 3：抓取运动物体

### 22.4.2 自动课程生成

**Teacher-Student 框架**：

- **Student**：学习策略
- **Teacher**：生成课程（选择任务难度）

**Teacher 目标**：最大化 Student 学习进展

$$
\text{Maximize Learning Progress} = \frac{\Delta \text{Performance}}{\Delta \text{Time}}
$$

**实现**：

```python
class AutomaticCurriculum:
    """
    自动课程学习
    
    根据学习进展动态调整任务难度
    """
    def __init__(self, difficulty_range=(0.0, 1.0)):
        self.difficulty_range = difficulty_range
        self.performance_history = []
    
    def select_difficulty(self, current_performance):
        """
        选择下一个任务难度
        
        Args:
            current_performance: 当前性能
        
        Returns:
            difficulty: 选择的难度
        """
        self.performance_history.append(current_performance)
        
        if len(self.performance_history) < 10:
            # 初期：随机探索
            return np.random.uniform(*self.difficulty_range)
        
        # 计算学习进展
        recent_progress = (
            self.performance_history[-1] - 
            self.performance_history[-10]
        )
        
        if recent_progress > 0.1:
            # 进展快 → 增加难度
            return min(self.difficulty_range[1], 
                      self.current_difficulty + 0.1)
        elif recent_progress < 0.01:
            # 进展慢 → 降低难度
            return max(self.difficulty_range[0],
                      self.current_difficulty - 0.1)
        else:
            # 保持当前难度
            return self.current_difficulty
```

### 22.4.3 Teacher-Student 框架

**PopArt**（Adaptive Normalization）：
- 自动调整奖励尺度
- 适应不同难度任务

**POET**（Paired Open-Ended Trailblazer）：
- 同时演化环境和智能体
- 开放式课程生成

---

## 22.5 实际应用

### 22.5.1 机器人多技能

**场景**：机器人学习抓取、放置、推动等多个技能。

**方法**：
- 多任务学习共享视觉编码器
- 每个技能独立策略头

**优势**：
- 数据效率高
- 技能可组合

### 22.5.2 游戏 AI 泛化

**StarCraft II**：
- 在多个地图上训练
- 泛化到新地图

**Dota 2**：
- Domain Randomization（英雄随机化）
- Zero-Shot 泛化到新英雄组合

---

## 本章小结

在本章中，我们学习了：

✅ **多任务 RL**：共享表示、任务干扰、Soft Modularization  
✅ **迁移学习**：Fine-tuning、Domain Randomization  
✅ **Zero-Shot Transfer**：Successor Features、UVFA  
✅ **Curriculum Learning**：任务难度递增、自动课程  
✅ **实际应用**：机器人多技能、游戏 AI 泛化  

> [!TIP]
> **核心要点**：
> - 多任务学习通过共享表示提升样本效率
> - 任务干扰是多任务RL的主要挑战
> - 迁移学习可复用源任务知识加速目标任务学习
> - Domain Randomization 增强 sim-to-real 迁移
> - Successor Features 分解值函数实现零样本迁移
> - Curriculum Learning 通过难度递增提升学习效率
> - 自动课程生成可根据学习进展动态调整难度

> [!NOTE]
> **下一步**：
> Chapter 23 将学习**元强化学习**：
> - MAML
> - PEARL
> - RL²
> 
> 进入 [Chapter 23. Meta-RL](23-meta-rl.md)

---

## 扩展阅读

- **经典论文**：
  - Barreto et al. (2017): Successor Features for Transfer in RL
  - Teh et al. (2017): Distral: Robust Multitask RL via Slow and Fast Learning Rates
  - Taylor & Stone (2009): Transfer Learning for RL: A Survey
  - Bengio et al. (2009): Curriculum Learning
- **Domain Randomization**：
  - Tobin et al. (2017): Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World
  - Peng et al. (2018): Sim-to-Real Transfer of Robotic Control with Dynamics Randomization
- **实现资源**：
  - Meta-World: Benchmark for Multi-Task RL
  - garage: Multi-task RL toolkit

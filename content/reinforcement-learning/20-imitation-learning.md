---
title: "Chapter 20. Imitation Learning"
description: "从演示中直接学习策略"
updated: "2026-01-29"
---

> **Learning Objectives**
> * 理解行为克隆（Behavioral Cloning）的原理与局限
> * 掌握 DAgger 的迭代数据聚合方法
> * 学习从观察中学习的方法
> * 了解 One-Shot Imitation
> * 掌握模仿学习与 RL 的结合

---

## 20.1 Behavioral Cloning

直接从专家演示学习策略的监督学习方法。

### 20.1.1 监督学习方法

**基本思想**：将模仿学习视为监督学习问题。

**数据**：专家演示 $\mathcal{D} = \{(s_i, a_i)\}_{i=1}^N$

**目标**：学习策略 $\pi_\theta(a|s)$ 最小化：

$$
\mathcal{L}(\theta) = -\sum_{(s,a) \in \mathcal{D}} \log \pi_\theta(a|s)
$$

<div data-component="BehavioralCloningProcess"></div>

**实现**：

```python
import torch
import torch.nn as nn
import torch.optim as optim

class BehavioralCloningPolicy(nn.Module):
    """
    行为克隆策略网络
    
    Args:
        state_dim: 状态维度
        action_dim: 动作维度（离散）或动作空间维度（连续）
        continuous: 是否为连续动作空间
    """
    def __init__(self, state_dim, action_dim, continuous=False):
        super().__init__()
        
        self.continuous = continuous
        
        # 共享特征提取
        self.features = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        if continuous:
            # 连续动作：输出均值和标准差
            self.mean = nn.Linear(256, action_dim)
            self.log_std = nn.Linear(256, action_dim)
        else:
            # 离散动作：输出 logits
            self.logits = nn.Linear(256, action_dim)
    
    def forward(self, state):
        """前向传播"""
        features = self.features(state)
        
        if self.continuous:
            mean = self.mean(features)
            log_std = self.log_std(features)
            return mean, log_std
        else:
            logits = self.logits(features)
            return logits
    
    def get_action(self, state):
        """采样动作"""
        if self.continuous:
            mean, log_std = self.forward(state)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            return action
        else:
            logits = self.forward(state)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return action


def train_behavioral_cloning(expert_demos, state_dim, action_dim, 
                            continuous=False, epochs=100):
    """
    训练行为克隆策略
    
    Args:
        expert_demos: [(state, action), ...]
        state_dim, action_dim: 维度
        continuous: 是否连续动作
        epochs: 训练轮数
    
    Returns:
        policy: 训练好的策略
    """
    policy = BehavioralCloningPolicy(state_dim, action_dim, continuous)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    
    # 准备数据
    states = torch.tensor([s for s, a in expert_demos], dtype=torch.float32)
    actions = torch.tensor([a for s, a in expert_demos], dtype=torch.float32)
    
    for epoch in range(epochs):
        if continuous:
            # 连续动作：负对数似然
            mean, log_std = policy(states)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            loss = -dist.log_prob(actions).sum(dim=-1).mean()
        else:
            # 离散动作：交叉熵
            logits = policy(states)
            loss = nn.CrossEntropyLoss()(logits, actions.long())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return policy
```

### 20.1.2 分布漂移问题

**问题**：训练分布 ≠ 测试分布

<div data-component="DistributionShift"></div>

**原因**：
- 专家数据来自 $p_{\text{expert}}(s)$
- 学习策略访问 $p_{\pi}(s)$
- 小错误累积 → 到达未见过的状态

**示例**：
1. 专家走直线
2. BC 略有偏差 → 偏离直线
3. 偏离状态没有专家演示 → 错误累积

**理论**：性能下降与时间成二次关系

$$
\text{Error} = O(\epsilon T^2)
$$

其中 $\epsilon$ 是单步错误，$T$ 是时间步。

### 20.1.3 数据增强

**缓解分布漂移的方法**：

1. **噪声注入**：
```python
# 在专家演示中加入噪声
noisy_states = states + noise_scale * torch.randn_like(states)
```

2. **合成数据**：
- 在专家轨迹附近生成状态
- 查询专家在这些状态的动作

3. **回放策略轨迹**：
- 收集策略运行时的失败案例
- 让专家标注正确动作

---

## 20.2 DAgger (Dataset Aggregation)

通过迭代数据聚合解决分布漂移。

<div data-component="DAggerIteration"></div>

### 20.2.1 交互式数据收集

**核心思想**：让策略收集数据，专家标注动作。

**DAgger 算法**（Ross et al., 2011）：

```
初始化: D = 空数据集, π = 随机策略

for iteration = 1 to N:
    # (1) 用当前策略收集轨迹
    τ = run_policy(π)
    
    # (2) 专家标注这些状态
    for state in τ:
        expert_action = expert(state)  # 专家查询
        D.add((state, expert_action))
    
    # (3) 用聚合数据重新训练策略
    π = train_BC(D)

return π
```

**实现**：

```python
def dagger(env, expert_policy, num_iterations=20, 
           rollouts_per_iter=10, steps_per_rollout=100):
    """
    DAgger 算法实现
    
    Args:
        env: 环境
        expert_policy: 专家策略（可查询）
        num_iterations: DAgger 迭代次数
        rollouts_per_iter: 每次迭代的轨迹数
        steps_per_rollout: 每条轨迹的步数
    
    Returns:
        policy: 最终策略
    """
    # 初始化
    dataset = []
    policy = BehavioralCloningPolicy(state_dim, action_dim)
    
    for iteration in range(num_iterations):
        print(f"\n=== DAgger Iteration {iteration + 1} ===")
        
        # (1) 用当前策略收集轨迹
        for _ in range(rollouts_per_iter):
            state = env.reset()
            
            for step in range(steps_per_rollout):
                # 策略选择动作
                with torch.no_grad():
                    action = policy.get_action(
                        torch.tensor(state, dtype=torch.float32)
                    ).numpy()
                
                # 专家标注（查询专家在当前状态的动作）
                expert_action = expert_policy(state)
                
                # 添加到数据集（状态 + 专家动作）
                dataset.append((state, expert_action))
                
                # 执行动作
                state, _, done, _ = env.step(action)
                if done:
                    break
        
        # (2) 用聚合数据重新训练策略
        print(f"Dataset size: {len(dataset)}")
        policy = train_behavioral_cloning(
            dataset, state_dim, action_dim, epochs=50
        )
    
    return policy
```

### 20.2.2 专家查询

**关键**：需要专家能够对任意状态给出动作。

**挑战**：
- 人类专家：查询成本高
- 自动专家（例如 MPC）：可行

**减少查询次数**：
- β-DAgger：以概率 β 查询专家
- Active DAgger：仅在不确定状态查询

### 20.2.3 迭代改进

**性能保证**：

$$
\text{Error}_{\text{DAgger}} = O(\epsilon T)
$$

（相比 BC 的 $O(\epsilon T^2)$）

**为什么有效**：
- 策略访问的状态会被加入数据集
- 缓解分布漂移

---

## 20.3 从观察中学习

不需要动作标注，仅从观察中学习。

### 20.3.1 第三人称模仿

**场景**：专家演示来自不同视角（例如视频）。

**挑战**：
- 没有动作标签
- 视角不同（第三人称 vs 第一人称）

**方法**：
- 学习状态表示
- 对应第三人称和第一人称状态

### 20.3.2 视角转换

**TCN (Time-Contrastive Networks)**：

- 学习嵌入 $\phi(o)$
- 对齐不同视角的观察

**目标**：时间上接近的观察应该嵌入接近。

---

## 20.4 One-Shot Imitation

从单个演示学习新任务。

### 20.4.1 元学习方法

**思想**：在任务分布上训练，快速适应新任务。

**One-Shot Imitation Learning**（Duan et al., 2017）：

1. 观察一个任务的演示
2. 推断任务嵌入
3. 执行该任务

### 20.4.2 任务嵌入

**架构**：

- 演示编码器：$z = \text{encode}(\tau_{\text{demo}})$
- 条件策略：$\pi(a|s, z)$

**训练**：
- 在多个任务上训练
- 每个任务给一个演示

---

## 20.5 与 RL 结合

模仿学习作为 RL 的初始化或辅助。

### 20.5.1 预训练 + 微调

**流程**：

1. **预训练**：用 BC 从专家演示学习初始策略
2. **微调**：用 RL 进一步优化

**优势**：
- 加速 RL 训练
- 避免危险的随机探索

**示例**：

```python
# (1) 预训练
policy = train_behavioral_cloning(expert_demos)

# (2) RL 微调
for episode in range(rl_episodes):
    # 用预训练策略初始化
    state = env.reset()
    done = False
    
    while not done:
        action = policy.get_action(state)
        next_state, reward, done = env.step(action)
        
        # PPO/SAC 更新
        policy.update(state, action, reward, next_state)
        state = next_state
```

### 20.5.2 奖励塑形

**BC 作为奖励**：

$$
r_{\text{total}} = r_{\text{env}} + \lambda \cdot \log \pi_{\text{BC}}(a|s)
$$

**效果**：鼓励策略接近专家行为。

---

## 本章小结

在本章中，我们学习了：

✅ **Behavioral Cloning**：监督学习、分布漂移问题  
✅ **DAgger**：迭代数据聚合、专家查询  
✅ **从观察中学习**：第三人称模仿、视角转换  
✅ **One-Shot Imitation**：元学习、任务嵌入  
✅ **与 RL 结合**：预训练 + 微调、奖励塑形  

> [!TIP]
> **核心要点**：
> - Behavioral Cloning 简单但受分布漂移影响
> - 分布漂移导致性能下降 O(εT²)
> - DAgger 通过迭代聚合数据改进到 O(εT)
> - 需要专家可查询（人类或自动）
> - 从观察学习不需要动作标签
> - One-Shot Imitation 使用元学习快速适应
> - 模仿学习可作为 RL 的预训练

> [!NOTE]
> **下一步**：
> Chapter 21 将学习**Offline RL**：
> - BCQ、CQL、IQL
> - Decision Transformer
> - 处理 OOD 动作问题
> 
> 进入 [Chapter 21. Offline RL](21-offline-rl.md)

---

## 扩展阅读

- **经典论文**：
  - Pomerleau (1991): Efficient Training of Artificial Neural Networks (ALVINN)
  - Ross et al. (2011): A Reduction of Imitation Learning and Structured Prediction (DAgger)
  - Torabi et al. (2018): Behavioral Cloning from Observation
  - Duan et al. (2017): One-Shot Imitation Learning
- **理论基础**：
  - Ross & Bagnell (2010): Efficient Reductions for Imitation Learning
  - Syed & Schapire (2007): A Game-Theoretic Approach to Apprenticeship Learning
- **实现资源**：
  - imitation library (BC, DAgger)
  - stable-baselines3
  - OpenAI Spinning Up

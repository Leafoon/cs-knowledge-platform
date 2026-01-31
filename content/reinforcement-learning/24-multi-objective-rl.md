---
title: "Chapter 24. Multi-Objective Reinforcement Learning"
description: "平衡多个目标的强化学习"
updated: "2026-01-29"
---

> **Learning Objectives**
> * 理解多目标优化问题
> * 掌握 Pareto Front 的概念
> * 学习 Scalarization 方法
> * 了解 Pareto Q-Learning
> * 掌握多目标权衡应用

---

## 24.1 多目标优化

同时优化多个（可能冲突的）目标。

### 24.1.1 Pareto Front

**单目标 RL**：最大化单一奖励 $r$

**多目标 RL**：最大化向量奖励 $\mathbf{r} = [r_1, r_2, \ldots, r_n]$

<div data-component="ParetoFrontVisualization"></div>

**Pareto 最优**：
- 策略 $\pi$ 是 Pareto 最优的，如果不存在另一个策略 $\pi'$ 在所有目标上都不差，且至少一个目标更好。

**数学定义**：

$$
\pi \text{ is Pareto optimal} \Leftrightarrow \nexists \pi': 
\begin{cases}
J_i(\pi') \geq J_i(\pi) & \forall i \\
J_j(\pi') > J_j(\pi) & \text{for some } j
\end{cases}
$$

**Pareto Front**：所有 Pareto 最优策略的集合。

### 24.1.2 目标冲突

**示例**：自动驾驶
- 目标 1：安全性（minimize 碰撞）
- 目标 2：效率（minimize 时间）

**冲突**：
- 高安全 → 慢速驾驶 → 低效率
- 高效率 → 快速驾驶 → 低安全

### 24.1.3 偏好权衡

<div data-component="MultiObjectiveTradeoff"></div>

**用户偏好**：$\mathbf{w} = [w_1, w_2, \ldots, w_n]$

**加权回报**：

$$
J_{\mathbf{w}}(\pi) = \sum_i w_i J_i(\pi)
$$

**不同用户，不同权重**：
- 保守用户：$w_{\text{safety}} = 0.8, w_{\text{speed}} = 0.2$
- 激进用户：$w_{\text{safety}} = 0.3, w_{\text{speed}} = 0.7$

---

## 24.2 Scalarization 方法

将多目标问题转化为单目标问题。

<div data-component="ScalarizationComparison"></div>

### 24.2.1 线性加权

**最简单方法**：线性组合

$$
r_{\text{scalar}} = w_1 r_1 + w_2 r_2 + \cdots + w_n r_n
$$

**实现**：

```python
class LinearScalarization:
    """
    线性加权 Scalarization
    
    Args:
        weights: 权重向量 [w₁, w₂, ..., wₙ]
    """
    def __init__(self, weights):
        self.weights = np.array(weights)
        self.weights = self.weights / self.weights.sum()  # 归一化
    
    def scalarize(self, multi_objective_reward):
        """
        标量化多目标奖励
        
        Args:
            multi_objective_reward: [r₁, r₂, ..., rₙ]
        
        Returns:
            scalar_reward: 标量奖励
        """
        return np.dot(self.weights, multi_objective_reward)


# 使用示例
scalarizer = LinearScalarization(weights=[0.6, 0.4])  # 60% 目标1, 40% 目标2

# 环境返回多目标奖励
state, multi_reward, done, _ = env.step(action)
# multi_reward = [r₁, r₂]

# 转换为标量
scalar_reward = scalarizer.scalarize(multi_reward)
```

**局限**：
- ❌ 仅能找到 Pareto Front 的凸部分
- ❌ 对权重敏感

### 24.2.2 Chebyshev Scalarization

**改进方法**：Chebyshev 距离

$$
r_{\text{cheby}} = \min_i \left( w_i (r_i - z_i^*) \right)
$$

其中 $z^*$ 是理想点（每个目标的最大值）。

**优势**：
- ✅ 能找到 Pareto Front 的非凸部分
- ✅ 更均衡的权衡

**实现**：

```python
class ChebyshevScalarization:
    """
    Chebyshev Scalarization
    
    Args:
        weights: 权重向量
        ideal_point: 理想点（可选）
    """
    def __init__(self, weights, ideal_point=None):
        self.weights = np.array(weights)
        self.ideal_point = ideal_point
    
    def scalarize(self, multi_objective_reward):
        """Chebyshev 标量化"""
        if self.ideal_point is None:
            # 假设理想点为 0（可根据任务调整）
            self.ideal_point = np.zeros_like(multi_objective_reward)
        
        # Chebyshev 距离
        weighted_diff = self.weights * (multi_objective_reward - self.ideal_point)
        return np.min(weighted_diff)
```

### 24.2.3 动态权重

**自适应权重**：根据学习进展调整权重。

**示例**：
- 早期：更关注探索（高权重给新目标）
- 后期：更关注优化（高权重给主要目标）

---

## 24.3 Pareto Q-Learning

学习向量值 Q 函数。

### 24.3.1 向量值 Q 函数

**标准 Q**：$Q(s, a) \in \mathbb{R}$

**向量 Q**：$\mathbf{Q}(s, a) \in \mathbb{R}^n$

$$
\mathbf{Q}(s, a) = [Q_1(s, a), Q_2(s, a), \ldots, Q_n(s, a)]
$$

每个 $Q_i$ 对应一个目标。

### 24.3.2 Pareto 最优策略集

**Pareto Q-learning** 算法：

```python
class ParetoQLearning:
    """
    Pareto Q-Learning
    
    学习向量值 Q 函数，找到 Pareto 最优策略集
    
    Args:
        state_dim: 状态维度
        action_dim: 动作维度
        num_objectives: 目标数量
    """
    def __init__(self, state_dim, action_dim, num_objectives):
        self.num_objectives = num_objectives
        
        # 向量 Q 表 (如果状态/动作离散)
        # Q[s][a] = [Q₁, Q₂, ..., Qₙ]
        self.Q = {}  # 或使用神经网络
    
    def get_pareto_optimal_actions(self, state):
        """
        获取 Pareto 最优动作集合
        
        Args:
            state: 当前状态
        
        Returns:
            pareto_actions: Pareto 最优的动作列表
        """
        # 获取所有动作的 Q 向量
        q_vectors = [self.Q.get((state, a), np.zeros(self.num_objectives))
                     for a in range(self.action_dim)]
        
        # 找到 Pareto 最优集
        pareto_actions = []
        for i, q_i in enumerate(q_vectors):
            is_dominated = False
            for j, q_j in enumerate(q_vectors):
                if i != j and self.dominates(q_j, q_i):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_actions.append(i)
        
        return pareto_actions
    
    @staticmethod
    def dominates(vec_a, vec_b):
        """
        判断 vec_a 是否 Pareto 支配 vec_b
        
        Returns:
            True if vec_a >= vec_b 在所有维度，且至少一个维度 >
        """
        return (np.all(vec_a >= vec_b) and np.any(vec_a > vec_b))
    
    def update(self, state, action, multi_reward, next_state, done):
        """
        向量 Q 更新
        
        Args:
            state, action, multi_reward, next_state, done
            multi_reward: [r₁, r₂, ..., rₙ]
        """
        if (state, action) not in self.Q:
            self.Q[(state, action)] = np.zeros(self.num_objectives)
        
        if not done:
            # 获取 next_state 的 Pareto 最优动作
            pareto_actions = self.get_pareto_optimal_actions(next_state)
            
            # 选择其中一个（例如随机选择）
            next_action = np.random.choice(pareto_actions)
            next_Q = self.Q.get((next_state, next_action), 
                               np.zeros(self.num_objectives))
        else:
            next_Q = np.zeros(self.num_objectives)
        
        # 向量 TD 更新
        target = multi_reward + gamma * next_Q
        self.Q[(state, action)] += alpha * (target - self.Q[(state, action)])
```

---

## 24.4 Conditioned RL

策略以偏好为条件。

### 24.4.1 偏好条件策略

**策略形式**：

$$
\pi(a | s, \mathbf{w})
$$

策略以权重 $\mathbf{w}$ 为条件。

**训练**：
- 在不同权重下训练
- 单一策略适应多种偏好

**实现**：

```python
class PreferenceConditionedPolicy(nn.Module):
    """
    偏好条件策略
    
    Args:
        state_dim: 状态维度
        action_dim: 动作维度
        num_objectives: 目标数量
    """
    def __init__(self, state_dim, action_dim, num_objectives):
        super().__init__()
        
        # 输入: state + preference weights
        input_dim = state_dim + num_objectives
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, state, preference):
        """
        前向传播
        
        Args:
            state: [batch, state_dim]
            preference: [batch, num_objectives] (权重向量)
        
        Returns:
            action_logits: [batch, action_dim]
        """
        # 拼接状态和偏好
        x = torch.cat([state, preference], dim=-1)
        return self.network(x)
    
    def select_action(self, state, preference):
        """根据偏好选择动作"""
        logits = self.forward(state, preference)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item()


# 使用示例
policy = PreferenceConditionedPolicy(state_dim=4, action_dim=2, num_objectives=2)

# 不同用户，不同偏好
conservative_user = torch.tensor([0.8, 0.2])  # 80% 安全, 20% 速度
aggressive_user = torch.tensor([0.3, 0.7])    # 30% 安全, 70% 速度

# 根据偏好选择动作
action_conservative = policy.select_action(state, conservative_user)
action_aggressive = policy.select_action(state, aggressive_user)
```

### 24.4.2 用户偏好学习

**从反馈学习偏好**：

- 用户评分
- 比较反馈（A vs B）
- 隐式反馈（点击率）

**贝叶斯优化**：
- 建模偏好不确定性
- 主动查询用户

---

## 24.5 应用

### 24.5.1 能耗 vs 性能

**场景**：数据中心服务器调度

**目标**：
- 目标 1：最小化能耗
- 目标 2：最大化性能（吞吐量）

**方法**：
- Scalarization with 动态权重
- 白天（峰值）：高权重性能
- 夜晚（低峰）：高权重能耗

### 24.5.2 安全 vs 效率

**场景**：自动驾驶

**目标**：
- 目标 1：安全性（minimize 风险）
- 目标 2：效率（minimize 时间）

**Pareto Front**：
- 极端保守：0 风险，但很慢
- 极端激进：快速，但高风险
- 中间策略：平衡

### 24.5.3 推荐系统多样性

**场景**：视频推荐

**目标**：
- 目标 1：点击率（用户满意度）
- 目标 2：多样性（避免信息茧房）
- 目标 3：新颖性（推荐新内容）

**多目标优化**：
- 学习 Pareto 最优推荐策略
- 根据用户偏好调整权重

---

## 本章小结

在本章中，我们学习了：

✅ **多目标优化**：Pareto Front、目标冲突、偏好权衡  
✅ **Scalarization**：线性加权、Chebyshev、动态权重  
✅ **Pareto Q-Learning**：向量 Q 函数、Pareto 最优策略集  
✅ **Conditioned RL**：偏好条件策略、用户偏好学习  
✅ **应用**：能耗 vs 性能、安全 vs 效率、推荐多样性  

> [!TIP]
> **核心要点**：
> - 多目标 RL 处理多个（可能冲突的）目标
> - Pareto Front 是所有 Pareto 最优策略的集合
> - Pareto 最优指无法在不损害其他目标的情况下改进任何目标
> - Scalarization 将多目标转化为单目标问题
> - 线性加权简单但只能找到凸 Pareto Front
> - Chebyshev Scalarization 可以找到非凸部分
> - Pareto Q-Learning 学习向量值 Q 函数
> - 偏好条件策略可以适应不同用户偏好

> [!NOTE]
> **后续章节将涵盖**：
> - Chapter 25: 安全强化学习（Safe RL）
> - Chapter 26: 大模型时代的 RL（RLHF、DPO）
> - 更多前沿主题

---

## 扩展阅读

- **经典论文**：
  - Vamplew et al. (2011): Empirical Evaluation Methods for Multiobjective RL Algorithms
  - Yang et al. (2019): A Generalized Algorithm for Multi-Objective RL and Policy Adaptation
  - Van Moffaert & Nowé (2014): Multi-Objective RL using Sets of Pareto Dominating Policies
- **理论基础**：
  - Roijers et al. (2013): A Survey of Multi-Objective Sequential Decision-Making
  - Liu et al. (2015): An Algorithm for Finding a Pareto Set in Multi-Objective RL
- **应用**：
  - Abels et al. (2019): Dynamic Weights in Multi-Objective Deep RL
  - Energy-efficient datacenter scheduling with MORL

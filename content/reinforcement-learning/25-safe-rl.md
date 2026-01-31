---
title: "Chapter 25. Safe Reinforcement Learning"
description: "约束、安全与鲁棒的强化学习"
updated: "2026-01-29"
---

> **Learning Objectives**
> * 理解安全强化学习的核心概念
> * 掌握约束 MDP (CMDP) 方法
> * 学习安全探索策略
> * 了解鲁棒 RL 与风险敏感 RL
> * 掌握实际应用场景

---

## 25.1 安全性定义

安全强化学习确保智能体在学习过程中满足安全约束。

### 25.1.1 约束满足

**定义**：智能体必须满足安全约束，例如：
- 碰撞约束：不能撞到障碍物
- 成本约束：成本不超过阈值
- 状态约束：保持在安全区域

**数学表达**：

$$
\mathbb{E}\left[\sum_t c_t\right] \leq d
$$

其中 $c_t$ 是成本，$d$ 是约束阈值。

### 25.1.2 风险敏感

**标准 RL**：最大化期望回报

$$
\max_\pi \mathbb{E}[R]
$$

**风险敏感 RL**：考虑回报的分布和风险

$$
\max_\pi \text{CVaR}_\alpha(R) \quad \text{或} \quad \max_\pi \min_{\text{scenario}} R
$$

### 25.1.3 鲁棒性

**目标**：策略在扰动、不确定性下仍能工作。

**方法**：
- 对抗训练
- Domain Randomization
- Worst-case 优化

---

## 25.2 约束 MDP (CMDP)

将安全约束形式化为 CMDP。

<div data-component="SafetyConstraintVisualization"></div>

### 25.2.1 成本约束

**CMDP 定义**：

$$
\text{CMDP} = (S, A, P, R, C, d)
$$

其中：
- $R(s,a)$：奖励函数
- $C(s,a)$：成本函数
- $d$：成本约束阈值

**约束**：

$$
\mathbb{E}\left[\sum_{t=0}^\infty \gamma^t C(s_t, a_t)\right] \leq d
$$

### 25.2.2 Lagrangian 方法

**拉格朗日松弛**：

$$
\max_\pi \min_{\lambda \geq 0} \mathbb{E}_\pi[R] - \lambda \left(\mathbb{E}_\pi[C] - d\right)
$$

**算法**：
1. 固定 $\lambda$，优化策略 $\pi$
2. 更新 $\lambda$ 以满足约束

**实现**：

```python
import torch
import torch.nn as nn

class LagrangianCMDP:
    """
    Lagrangian方法求解CMDP
    
    Args:
        policy: 策略网络
        cost_limit: 成本约束阈值
        lambda_lr: λ 学习率
    """
    def __init__(self, policy, cost_limit, lambda_lr=0.01):
        self.policy = policy
        self.cost_limit = cost_limit
        
        # Lagrangian乘子（可学习参数）
        self.lam = nn.Parameter(torch.tensor(0.0))
        self.lambda_optimizer = torch.optim.Adam([self.lam], lr=lambda_lr)
    
    def compute_lagrangian_loss(self, rewards, costs):
        """
        计算Lagrangian损失
        
        Args:
            rewards: 奖励 [batch]
            costs: 成本 [batch]
        
        Returns:
            loss: Lagrangian损失
        """
        # 期望奖励
        avg_reward = rewards.mean()
        
        # 期望成本
        avg_cost = costs.mean()
        
        # Lagrangian目标（最大化）
        # L = R - λ(C - d)
        lagrangian = avg_reward - self.lam * (avg_cost - self.cost_limit)
        
        # 转为最小化
        loss = -lagrangian
        
        return loss, avg_cost
    
    def update_lambda(self, avg_cost):
        """
        更新Lagrangian乘子
        
        如果成本超过阈值，增大λ
        """
        self.lambda_optimizer.zero_grad()
        
        # λ应该最大化约束违反
        lambda_loss = -self.lam * (avg_cost - self.cost_limit)
        
        lambda_loss.backward()
        self.lambda_optimizer.step()
        
        # 保证λ非负
        self.lam.data.clamp_(min=0.0)


# 使用示例
policy = PolicyNetwork(state_dim, action_dim)
cmdp_solver = LagrangianCMDP(policy, cost_limit=10.0)

for episode in range(num_episodes):
    states, actions, rewards, costs = collect_trajectory(policy)
    
    # 计算Lagrangian损失
    loss, avg_cost = cmdp_solver.compute_lagrangian_loss(rewards, costs)
    
    # 更新策略
    policy_optimizer.zero_grad()
    loss.backward()
    policy_optimizer.step()
    
    # 更新λ
    cmdp_solver.update_lambda(avg_cost)
```

### 25.2.3 CPO (Constrained Policy Optimization)

**CPO** (Achiam et al., 2017)：约束策略优化。

**目标**：

$$
\max_\pi J(\pi) \quad \text{s.t.} \quad J_C(\pi) \leq d
$$

**Trust Region 约束**：

$$
\text{KL}(\pi_{\text{old}} || \pi) \leq \delta
$$

**算法流程**：

```
1. 收集轨迹 D
2. 估计奖励梯度 g 和成本梯度 b
3. 求解约束优化问题（二次规划）
4. 更新策略
```

---

## 25.3 Safe Exploration

在探索时保证安全。

<div data-component="SafeExplorationDemo"></div>

### 25.3.1 安全集合

**安全集合** $\mathcal{S}_{\text{safe}}$：保证安全的状态集合。

**目标**：探索时保持在 $\mathcal{S}_{\text{safe}}$ 内。

**方法**：
- 学习安全集合
- 通过可达性分析验证

### 25.3.2 Shield 机制

**Shield**：安全保护层，拦截不安全动作。

**架构**：

```
智能体策略 π → Shield → 环境
```

**Shield 功能**：
- 如果动作 $a$ 安全 → 执行 $a$
- 如果动作 $a$ 不安全 → 执行安全替代动作

**实现**：

```python
class SafetyShield:
    """
    安全Shield机制
    
    拦截不安全的动作，替换为安全动作
    """
    def __init__(self, safe_controller):
        self.safe_controller = safe_controller
    
    def is_safe(self, state, action):
        """
        检查动作是否安全
        
        Args:
            state: 当前状态
            action: 候选动作
        
        Returns:
            safe: 是否安全（bool）
        """
        # 预测执行动作后的下一个状态
        next_state = self.predict_next_state(state, action)
        
        # 检查是否在安全区域
        return self.is_in_safe_set(next_state)
    
    def filter_action(self, state, proposed_action):
        """
        过滤不安全的动作
        
        Args:
            state: 当前状态
            proposed_action: 策略提议的动作
        
        Returns:
            safe_action: 安全的动作
        """
        if self.is_safe(state, proposed_action):
            return proposed_action
        else:
            # 使用安全控制器
            return self.safe_controller.get_safe_action(state)


# 使用示例
shield = SafetyShield(safe_controller)

# 智能体提议动作
proposed_action = policy.select_action(state)

# Shield过滤
safe_action = shield.filter_action(state, proposed_action)

# 执行安全动作
next_state, reward, done, _ = env.step(safe_action)
```

### 25.3.3 Reachability Analysis

**可达性分析**：验证从当前状态能否安全到达目标。

**方法**：
- 向后可达集合（Backward Reachable Set）
- 验证安全性

---

## 25.4 Robust RL

对抗扰动和不确定性。

<div data-component="RobustPolicyComparison"></div>

### 25.4.1 对抗鲁棒性

**对抗攻击**：恶意扰动观测或动力学。

**防御方法**：
- 对抗训练
- 鲁棒策略学习

**对抗训练**：

```python
def adversarial_training(policy, env, epsilon=0.1):
    """
    对抗训练：在扰动下训练策略
    
    Args:
        policy: 策略
        env: 环境
        epsilon: 扰动幅度
    """
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # (1) 策略选择动作
            action = policy.select_action(state)
            
            # (2) 对抗性添加扰动到状态
            perturbed_state = state + epsilon * torch.randn_like(state)
            
            # (3) 在扰动状态上重新计算动作
            action_perturbed = policy.select_action(perturbed_state)
            
            # (4) 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # (5) 训练策略（最坏情况）
            policy.update(perturbed_state, action_perturbed, reward, next_state)
            
            state = next_state
```

### 25.4.2 Domain Randomization

**已在 Chapter 22 介绍**：随机化环境参数提升鲁棒性。

### 25.4.3 Worst-Case Optimization

**目标**：在最坏情况下优化。

$$
\max_\pi \min_{\omega \in \Omega} J(\pi, \omega)
$$

其中 $\omega$ 是环境参数，$\Omega$ 是不确定性集合。

---

## 25.5 风险敏感 RL

考虑回报分布的尾部风险。

<div data-component="CVaRRiskMeasure"></div>

### 25.5.1 CVaR (Conditional Value at Risk)

**CVaR 定义**：

$$
\text{CVaR}_\alpha(R) = \mathbb{E}[R \mid R \leq \text{VaR}_\alpha(R)]
$$

**解释**：最坏 $\alpha$ 情况的平均回报。

**目标**：

$$
\max_\pi \text{CVaR}_\alpha(R)
$$

**实现**：

```python
class CVaRObjective:
    """
    CVaR目标函数
    
    Args:
        alpha: 风险水平（例如 0.1 表示最坏10%）
    """
    def __init__(self, alpha=0.1):
        self.alpha = alpha
    
    def compute_cvar(self, returns):
        """
        计算CVaR
        
        Args:
            returns: 回报样本 [num_samples]
        
        Returns:
            cvar: CVaR值
        """
        # 排序回报
        sorted_returns = torch.sort(returns)[0]
        
        # 计算VaR（α分位数）
        var_index = int(self.alpha * len(returns))
        var = sorted_returns[var_index]
        
        # 计算CVaR（VaR以下的平均）
        cvar = sorted_returns[:var_index].mean()
        
        return cvar
    
    def cvar_loss(self, returns):
        """CVaR损失（最大化CVaR）"""
        cvar = self.compute_cvar(returns)
        return -cvar  # 负号因为要最大化


# 使用示例
cvar_objective = CVaRObjective(alpha=0.1)

# 收集多条轨迹的回报
returns = []
for _ in range(num_trajectories):
    trajectory_return = run_episode(policy, env)
    returns.append(trajectory_return)

returns = torch.tensor(returns)

# 计算CVaR损失
loss = cvar_objective.cvar_loss(returns)

# 优化策略
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 25.5.2 分布式 RL

**Distributional RL**（见 Chapter 12）：学习回报分布。

**优势**：
- 更丰富的信息
- 自然支持风险敏感

### 25.5.3 风险度量

**常见风险度量**：
- **VaR** (Value at Risk)：α 分位数
- **CVaR** (Conditional VaR)：尾部平均
- **标准差**：$\sigma(R)$
- **最坏情况**：$\min R$

---

## 25.6 实际应用

### 25.6.1 自动驾驶

**安全约束**：
- 不能碰撞
- 保持车道
- 遵守交通规则

**方法**：
- CMDP with 碰撞成本
- Shield 拦截危险动作

### 25.6.2 医疗决策

**安全需求**：
- 避免有害治疗
- 满足伦理约束

**方法**：
- 保守策略
- 风险敏感优化

### 25.6.3 金融交易

**风险管理**：
- 限制损失
- 控制波动

**方法**：
- CVaR 优化
- Robust RL

---

## 本章小结

在本章中，我们学习了：

✅ **安全性定义**：约束满足、风险敏感、鲁棒性  
✅ **CMDP**：成本约束、Lagrangian方法、CPO  
✅ **Safe Exploration**：安全集合、Shield机制、可达性分析  
✅ **Robust RL**：对抗鲁棒性、Domain Randomization、Worst-case  
✅ **风险敏感 RL**：CVaR、分布式RL、风险度量  
✅ **实际应用**：自动驾驶、医疗决策、金融交易  

> [!TIP]
> **核心要点**:
> - 安全RL确保智能体在学习时满足约束
> - CMDP将安全约束形式化为成本约束
> - Lagrangian方法通过乘子平衡奖励与成本
> - CPO是TRPO的约束版本
> - Shield机制在运行时拦截不安全动作
> - 对抗训练提升策略鲁棒性
> - CVaR优化考虑最坏情况风险
> - 安全RL对实际部署至关重要

> [!NOTE]
> **下一步**：
> Chapter 26 将学习**多智能体强化学习基础**：
> - MARL问题定义
> - 博弈论基础
> - CTDE架构
> - 通信机制
> 
> 进入 [Chapter 26. MARL Foundations](26-marl-foundations.md)

---

## 扩展阅读

- **经典论文**：
  - Achiam et al. (2017): Constrained Policy Optimization
  - García & Fernández (2015): A Comprehensive Survey on Safe Reinforcement Learning
  - Dalal et al. (2018): Safe Exploration in Continuous Action Spaces
  - Rockafellar & Uryasev (2000): Optimization of Conditional Value-at-Risk
- **理论基础**：
  - Altman (1999): Constrained Markov Decision Processes
  - Tamar et al. (2015): Sequential Decision Making with CVaR
- **实现资源**：
  - Safety Gym: Benchmark for Safe RL
  - safe-control-gym: Safe RL for control
  - OSQP: Quadratic Programming solver (用于CPO)

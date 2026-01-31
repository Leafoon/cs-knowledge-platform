---
title: "Chapter 11. Trust Region Policy Optimization (TRPO)"
description: "保证单调改进的策略优化方法"
updated: "2026-01-29"
---

> **Learning Objectives**
> * 理解策略优化的挑战与步长选择问题
> * 掌握 Trust Region 方法与 KL 散度约束
> * 学习 TRPO 的理论基础与单调改进保证
> * 理解共轭梯度法与 Fisher Information Matrix
> * 认识 TRPO 的局限性及其对 PPO 的启发

---

## 11.1 策略优化的挑战

在策略梯度方法中，选择合适的步长是一个核心挑战。

### 11.1.1 步长选择困难

**普通梯度下降**：

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}_t)
$$

**问题**：
- $\alpha$ 太小 → 学习慢
- $\alpha$ 太大 → 性能崩溃

**为什么策略网络对步长敏感？**

在监督学习中，步长过大顶多导致训练震荡。但在 RL 中：
1. **改变数据分布**：策略变化 → 采样分布变化 → 训练数据变化
2. **非平稳性**：价值函数和策略相互影响
3. **累积效应**：小的策略错误会被放大

### 11.1.2 性能崩溃风险

**灾难性崩溃示例**：

```
Step 100: Reward = 200 ✅
Step 101: 步长过大，策略变坏
Step 102: Reward = 50 ❌ (崩溃)
Step 103: 无法恢复...
```

**原因**：
- 糟糕的策略 → 糟糕的数据 → 更糟糕的策略
- 陷入恶性循环

### 11.1.3 单调改进的必要性

**目标**：保证每次更新后，性能**不会变差**。

$$
J(\boldsymbol{\theta}_{t+1}) \geq J(\boldsymbol{\theta}_t) \quad \text{(单调改进)}
$$

**好处**：
- ✅ 稳定训练
- ✅ 可预测的进展
- ✅ 避免崩溃

---

## 11.2 Trust Region 方法

Trust Region（信赖域）方法通过约束策略变化来保证稳定性。

### 11.2.1 约束优化问题

**TRPO 优化问题**：

$$
\begin{align}
\max_{\boldsymbol{\theta}} \quad & \mathbb{E}_{s,a \sim \pi_{\boldsymbol{\theta}_{\text{old}}}} \left[\frac{\pi_{\boldsymbol{\theta}}(a|s)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a|s)} A^{\pi_{\boldsymbol{\theta}_{\text{old}}}}(s,a)\right] \\
\text{s.t.} \quad & \mathbb{E}_{s \sim \pi_{\boldsymbol{\theta}_{\text{old}}}} [D_{KL}(\pi_{\boldsymbol{\theta}_{\text{old}}}(\cdot|s) \| \pi_{\boldsymbol{\theta}}(\cdot|s))] \leq \delta
\end{align}
$$

**解释**：
- **目标**：最大化代理目标（surrogate objective）
- **约束**：KL 散度不超过阈值 $\delta$（例如 0.01）

<div data-component="TrustRegionVisualization"></div>

### 11.2.2 KL 散度约束

**KL 散度**衡量两个分布的差异：

$$
D_{KL}(P \| Q) = \mathbb{E}_{x \sim P} \left[\log \frac{P(x)}{Q(x)}\right]
$$

**在策略优化中**：

$$
D_{KL}(\pi_{\text{old}} \| \pi_{\text{new}}) = \mathbb{E}_{a \sim \pi_{\text{old}}} \left[\log \frac{\pi_{\text{old}}(a|s)}{\pi_{\text{new}}(a|s)}\right]
$$

**作用**：
- 限制新策略与旧策略的差异
- 防止更新步长过大
- 保持在"信赖域"内

<div data-component="KLConstraintEffect"></div>

### 11.2.3 单调改进保证

**Kakade & Langford (2002) 定理**：

在 KL 约束下，TRPO 保证单调改进：

$$
J(\boldsymbol{\theta}_{\text{new}}) \geq J(\boldsymbol{\theta}_{\text{old}}) - C \cdot \delta
$$

其中 $C$ 是常数，$\delta$ 是 KL 散度上限。

**证明核心**：利用 Advantage 函数的性质和 KL 散度的界。

---

## 11.3 理论基础

TRPO 建立在严格的理论分析之上。

### 11.3.1 策略改进界（Policy Improvement Bound）

**定理**（Kakade & Langford, 2002）：

定义：

$$
\epsilon = \max_s \left|\mathbb{E}_{a \sim \pi_{\text{new}}}[A^{\pi_{\text{old}}}(s,a)]\right|
$$

$$
\alpha = D_{KL}^{\max}(\pi_{\text{old}} \| \pi_{\text{new}})
$$

则：

$$
J(\pi_{\text{new}}) \geq L_{\pi_{\text{old}}}(\pi_{\text{new}}) - \frac{4\epsilon\gamma}{(1-\gamma)^2} \alpha
$$

其中 $L$ 是代理目标。

**含义**：
- 代理目标的提升 + 一个惩罚项
- KL 散度越小，界越紧

### 11.3.2 Kakade & Langford (2002) 定理

**保守策略迭代（Conservative Policy Iteration）**：

通过限制 KL 散度，可以保证性能提升：

$$
\pi_{\text{new}}(a|s) = (1-\alpha) \pi_{\text{old}}(a|s) + \alpha \pi^*(a|s)
$$

其中 $\pi^*$ 是贪婪策略。

### 11.3.3 Surrogate Objective

**代理目标**：

$$
L_{\pi_{\boldsymbol{\theta}_{\text{old}}}}(\boldsymbol{\theta}) = \mathbb{E}_{s,a \sim \pi_{\boldsymbol{\theta}_{\text{old}}}} \left[\frac{\pi_{\boldsymbol{\theta}}(a|s)}{\pi_{\boldsymbol{\theta}_{\text{old}}}(a|s)} A^{\pi_{\boldsymbol{\theta}_{\text{old}}}}(s,a)\right]
$$

**为什么叫 Surrogate？**

真实目标 $J(\boldsymbol{\theta})$ 难以直接优化（需要在新策略下采样），而 $L$ 可以用旧策略的数据计算。

**关键性质**：

$$
\nabla_{\boldsymbol{\theta}} L_{\pi_{\text{old}}}(\boldsymbol{\theta})|_{\boldsymbol{\theta}=\boldsymbol{\theta}_{\text{old}}} = \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})|_{\boldsymbol{\theta}=\boldsymbol{\theta}_{\text{old}}}
$$

即：在 $\boldsymbol{\theta}_{\text{old}}$ 处，两者梯度相同。

---

## 11.4 TRPO 算法

TRPO 求解带 KL 约束的优化问题。

### 11.4.1 约束优化形式

**原始问题**：

$$
\begin{align}
\max_{\boldsymbol{\theta}} \quad & L(\boldsymbol{\theta}) \\
\text{s.t.} \quad & \bar{D}_{KL}(\boldsymbol{\theta}_{\text{old}}, \boldsymbol{\theta}) \leq \delta
\end{align}
$$

其中：

$$
\bar{D}_{KL}(\boldsymbol{\theta}_1, \boldsymbol{\theta}_2) = \mathbb{E}_{s} [D_{KL}(\pi_{\boldsymbol{\theta}_1}(\cdot|s) \| \pi_{\boldsymbol{\theta}_2}(\cdot|s))]
$$

### 11.4.2 共轭梯度法

**线性近似**：

在 $\boldsymbol{\theta}_{\text{old}}$ 附近泰勒展开：

$$
L(\boldsymbol{\theta}) \approx L(\boldsymbol{\theta}_{\text{old}}) + \mathbf{g}^T (\boldsymbol{\theta} - \boldsymbol{\theta}_{\text{old}})
$$

$$
\bar{D}_{KL} \approx \frac{1}{2} (\boldsymbol{\theta} - \boldsymbol{\theta}_{\text{old}})^T \mathbf{F} (\boldsymbol{\theta} - \boldsymbol{\theta}_{\text{old}})
$$

其中：
- $\mathbf{g} = \nabla_{\boldsymbol{\theta}} L|_{\boldsymbol{\theta}_{\text{old}}}$：梯度
- $\mathbf{F}$：Fisher Information Matrix

**近似问题**：

$$
\begin{align}
\max_{\mathbf{x}} \quad & \mathbf{g}^T \mathbf{x} \\
\text{s.t.} \quad & \frac{1}{2} \mathbf{x}^T \mathbf{F} \mathbf{x} \leq \delta
\end{align}
$$

**解**（拉格朗日乘数法）：

$$
\mathbf{x}^* = \sqrt{\frac{2\delta}{\mathbf{g}^T \mathbf{F}^{-1} \mathbf{g}}} \mathbf{F}^{-1} \mathbf{g}
$$

**实现**：
- 直接计算 $\mathbf{F}^{-1}$ 太贵（$O(n^3)$）
- 使用**共轭梯度法**（Conjugate Gradient）求解 $\mathbf{F} \mathbf{x} = \mathbf{g}$

<div data-component="ConjugateGradientProcess"></div>

### 11.4.3 Line Search

**问题**：二次近似可能不准确。

**解决**：在更新方向上进行 **line search**（回溯线搜索）。

**算法**：

```python
def line_search(theta_old, full_step, expected_improve, max_backtracks=10):
    """
    Line search to ensure improvement and KL constraint
    
    Args:
        theta_old: 旧参数
        full_step: 完整更新步长
        expected_improve: 预期改进
        max_backtracks: 最大回溯次数
    """
    alpha = 1.0
    for _ in range(max_backtracks):
        theta_new = theta_old + alpha * full_step
        
        # 检查改进
        new_loss = compute_loss(theta_new)
        actual_improve = new_loss - old_loss
        
        # 检查 KL 约束
        kl = compute_kl(theta_old, theta_new)
        
        # 满足条件：改进 + KL 约束
        if actual_improve > 0 and kl <= max_kl:
            return theta_new
        
        # 否则，减小步长
        alpha *= 0.5
    
    # 如果都不满足，保持不变
    return theta_old
```

### 11.4.4 Fisher Information Matrix

**定义**：

$$
\mathbf{F} = \mathbb{E}_{s,a \sim \pi} [\nabla_{\boldsymbol{\theta}} \log \pi(a|s) \cdot \nabla_{\boldsymbol{\theta}} \log \pi(a|s)^T]
$$

**性质**：
- 对称正定矩阵
- 度量参数空间的局部曲率
- 与 KL 散度的 Hessian 等价

**Hessian-Vector Product**：

计算 $\mathbf{F} \mathbf{v}$ 而不显式构造 $\mathbf{F}$：

$$
\mathbf{F} \mathbf{v} \approx \nabla_{\boldsymbol{\theta}} \left[(\nabla_{\boldsymbol{\theta}} \bar{D}_{KL})^T \mathbf{v}\right]
$$

可以用自动微分高效计算。

---

## 11.5 实现细节

### 11.5.1 自然梯度计算

**自然梯度**（Natural Gradient）：

$$
\tilde{\mathbf{g}} = \mathbf{F}^{-1} \mathbf{g}
$$

**直觉**：
- 普通梯度：在参数空间中的最陡方向
- 自然梯度：在**分布空间**中的最陡方向

**优势**：
- 对参数化不敏感
- 更好的收敛性质

### 11.5.2 Hessian-Vector Product

**实现技巧**：

```python
def fisher_vector_product(policy, states, vector):
    """
    计算 Fisher Information Matrix 与向量的乘积
    
    Args:
        policy: 策略网络
        states: 状态batch
        vector: 向量
    
    Returns:
        F * vector
    """
    # 1. 计算 KL 散度的梯度
    kl = compute_kl_divergence(policy, states)
    kl_grad = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
    kl_grad_vector = torch.cat([g.view(-1) for g in kl_grad])
    
    # 2. 计算 (∇KL)^T * vector
    grad_vector_product = (kl_grad_vector * vector).sum()
    
    # 3. 计算其对参数的梯度 → F * vector
    fvp = torch.autograd.grad(grad_vector_product, policy.parameters())
    fvp_vector = torch.cat([g.contiguous().view(-1) for g in fvp])
    
    return fvp_vector
```

### 11.5.3 计算复杂度

**共轭梯度法**：
- 迭代次数：通常 10-20 次
- 每次迭代：需要计算 Hessian-Vector Product
- 总复杂度：$O(k \cdot n)$，其中 $k$ 是迭代次数，$n$ 是参数数量

**对比**：
- 直接求逆：$O(n^3)$
- 共轭梯度：$O(k \cdot n)$，通常 $k \ll n$

---

## 11.6 TRPO 的局限性

尽管 TRPO 理论优美，但实践中存在诸多挑战。

### 11.6.1 计算开销大

**每次更新需要**：
1. 计算梯度 $\mathbf{g}$
2. 运行共轭梯度法（10-20 次 FVP 计算）
3. Line search（多次前向传播）

**对比 A2C**：
- A2C：一次前向+反向传播
- TRPO：10-20 次 + line search

**结果**：TRPO 慢 10-20 倍。

### 11.6.2 实现复杂

**需要实现**：
- Fisher-Vector Product
- 共轭梯度法
- Line search
- KL 散度计算

**代码量**：~500 行（vs A2C 的 ~100 行）

**调试困难**：
- 多个组件相互依赖
- 数值稳定性问题

### 11.6.3 引出 PPO

**TRPO 的核心价值**：
1. 理论保证（单调改进）
2. KL 约束的有效性

**问题**：太复杂了！

**启发 PPO**：
- 保留 TRPO 的核心思想（限制策略变化）
- 简化实现（去掉共轭梯度、line search）
- 使用 **Clipping** 代替 KL 约束

---

## 11.7 完整 TRPO 伪代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np

class TRPOAgent:
    """
    TRPO (Trust Region Policy Optimization) Agent
    
    注意：这是简化版实现，完整实现需要更多细节处理
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64, max_kl=0.01, damping=0.1):
        self.actor = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.critic = ValueNetwork(state_dim, hidden_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.max_kl = max_kl
        self.damping = damping
        self.cg_iters = 10
        self.backtrack_iters = 10
        self.backtrack_coeff = 0.8
    
    def compute_kl(self, states, old_policy):
        """计算 KL 散度"""
        with torch.no_grad():
            old_probs = old_policy(states)
        new_probs = self.actor(states)
        
        kl = (old_probs * (torch.log(old_probs) - torch.log(new_probs))).sum(dim=1).mean()
        return kl
    
    def fisher_vector_product(self, states, vector):
        """Fisher Information Matrix 与向量的乘积"""
        # 计算 KL 散度
        old_probs = self.actor(states).detach()
        new_probs = self.actor(states)
        kl = (old_probs * (torch.log(old_probs) - torch.log(new_probs))).sum(dim=1).mean()
        
        # 计算梯度
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([g.view(-1) for g in kl_grad])
        
        # 梯度与向量的内积
        grad_vector_product = (kl_grad_vector * vector).sum()
        
        # 计算 Hessian-vector product
        fvp = torch.autograd.grad(grad_vector_product, self.actor.parameters())
        fvp_vector = torch.cat([g.contiguous().view(-1) for g in fvp])
        
        return fvp_vector + self.damping * vector
    
    def conjugate_gradient(self, states, b):
        """
        共轭梯度法求解 Ax = b
        其中 A 是 Fisher Information Matrix
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        
        for _ in range(self.cg_iters):
            Ap = self.fisher_vector_product(states, p)
            alpha = rdotr / (torch.dot(p, Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = torch.dot(r, r)
            beta = new_rdotr / (rdotr + 1e-8)
            p = r + beta * p
            rdotr = new_rdotr
            
            if rdotr < 1e-10:
                break
        
        return x
    
    def line_search(self, states, actions, advantages, old_policy_params, full_step):
        """Line search 找到满足条件的步长"""
        old_loss = self.compute_surrogate_loss(states, actions, advantages).item()
        old_params = self.get_flat_params()
        
        for i in range(self.backtrack_iters):
            step_size = self.backtrack_coeff ** i
            new_params = old_params + step_size * full_step
            self.set_flat_params(new_params)
            
            # 检查改进
            new_loss = self.compute_surrogate_loss(states, actions, advantages).item()
            actual_improve = new_loss - old_loss
            
            # 检查 KL 约束
            kl = self.compute_kl(states, old_policy_params)
            
            if actual_improve > 0 and kl <= self.max_kl:
                return True
        
        # 回退到旧参数
        self.set_flat_params(old_params)
        return False
    
    def get_flat_params(self):
        """获取扁平化的参数向量"""
        return torch.cat([p.data.view(-1) for p in self.actor.parameters()])
    
    def set_flat_params(self, flat_params):
        """设置扁平化的参数向量"""
        offset = 0
        for p in self.actor.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset + numel].view_as(p.data))
            offset += numel
    
    def compute_surrogate_loss(self, states, actions, advantages):
        """计算代理目标"""
        action_probs = self.actor(states)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)
        
        return (log_probs * advantages).mean()
    
    def train_step(self, states, actions, returns, advantages):
        """TRPO 训练步骤"""
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # 标准化 advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 保存旧策略
        old_policy = self.actor(states).detach()
        
        # 1. 计算梯度
        loss = self.compute_surrogate_loss(states, actions, advantages)
        grads = torch.autograd.grad(loss, self.actor.parameters())
        loss_grad = torch.cat([g.view(-1) for g in grads])
        
        # 2. 共轭梯度求解
        step_dir = self.conjugate_gradient(states, loss_grad)
        
        # 3. 计算完整步长
        shs = 0.5 * torch.dot(step_dir, self.fisher_vector_product(states, step_dir))
        step_size = torch.sqrt(2 * self.max_kl / (shs + 1e-8))
        full_step = step_size * step_dir
        
        # 4. Line search
        self.line_search(states, actions, advantages, old_policy, full_step)
        
        # 5. 更新 Critic
        values = self.critic(states).squeeze()
        critic_loss = nn.MSELoss()(values, returns)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return {
            'loss': loss.item(),
            'critic_loss': critic_loss.item()
        }
```

<div data-component="MonotonicImprovement"></div>

---

## 本章小结

在本章中，我们学习了：

✅ **策略优化挑战**：步长选择困难、性能崩溃风险  
✅ **Trust Region 方法**：KL 约束保证稳定性  
✅ **TRPO 理论**：单调改进保证、策略改进界  
✅ **实现技巧**：共轭梯度法、Fisher-Vector Product、Line Search  
✅ **TRPO 局限**：计算开销大、实现复杂  

> [!TIP]
> **核心要点**：
> - TRPO 通过 KL 约束限制策略变化，保证单调改进
> - 使用共轭梯度法高效求解约束优化问题
> - 理论严谨但实现复杂，为 PPO 提供理论基础
> - Line search 确保实际改进和 KL 约束满足

> [!NOTE]
> **下一步**：
> Chapter 12 将学习**近端策略优化（PPO）**：
> - 简化 TRPO 的实现
> - PPO-Clip 机制
> - 现代 RL 的主力算法
> - 广泛应用（OpenAI Five、ChatGPT RLHF）
> 
> 进入 [Chapter 12. PPO](12-ppo.md)

---

## 扩展阅读

- **经典论文**：
  - Schulman et al. (2015): Trust Region Policy Optimization
  - Kakade & Langford (2002): Approximately Optimal Approximate RL
- **实现资源**：
  - OpenAI Spinning Up: TRPO
  - rllab: TRPO implementation
- **理论深入**：
  - Sham Kakade: Natural Policy Gradient
  - Amari: Natural Gradient Works Efficiently in Learning

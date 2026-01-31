---
title: "Chapter 14. Natural Policy Gradient (NPG)"
description: "策略空间中的最速下降方向"
updated: "2026-01-29"
---

> **Learning Objectives**
> * 理解参数空间与策略空间的区别
> * 掌握 Fisher Information Matrix 的几何意义
> * 学习自然梯度的定义与计算
> * 理解 NPG 与 TRPO 的联系
> * 了解 K-FAC 等实用优化方法

---

## 14.1 梯度下降的问题

普通梯度下降在参数空间中进行，但在策略空间中可能低效。

### 14.1.1 参数空间 vs 策略空间

**参数空间**：神经网络参数 $\boldsymbol{\theta} \in \mathbb{R}^n$

**策略空间**：概率分布 $\pi_{\boldsymbol{\theta}}(\cdot|s)$ 的空间

**问题**：相同的参数变化 $\Delta \boldsymbol{\theta}$ 可能导致**非常不同**的策略变化！

<div data-component="ParameterSpaceVsPolicySpace"></div>

**示例**（Softmax 策略）：

$$
\pi_{\boldsymbol{\theta}}(a|s) = \frac{\exp(\boldsymbol{\theta}^T \phi(s, a))}{\sum_{a'} \exp(\boldsymbol{\theta}^T \phi(s, a'))}
$$

- 当 $\|\boldsymbol{\theta}\|$ 很大时：$\pi$ 接近确定性，对 $\boldsymbol{\theta}$ 的小变化敏感
- 当 $\|\boldsymbol{\theta}\|$ 很小时：$\pi$ 接近均匀分布，对 $\boldsymbol{\theta}$ 的变化不敏感

### 14.1.2 步长选择困难

**普通梯度下降**：

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}_t)
$$

**问题**：
- **固定步长 $\alpha$**：在参数空间中是固定的，但在策略空间中是变化的
- **早期训练**：$\alpha$ 太大可能导致策略变化过大
- **后期训练**：同样的 $\alpha$ 可能导致策略几乎不变

### 14.1.3 协变量偏移

**协变量偏移**（Covariate Shift）：
- 参数化方式影响梯度方向
- 重新参数化（例如 $\boldsymbol{\theta}' = A\boldsymbol{\theta}$）会改变梯度
- 但最优策略不应依赖于参数化方式！

**期望**：一个**参数化不变**的优化方法。

---

## 14.2 自然梯度

自然梯度在策略空间（而非参数空间）中寻找最速下降方向。

### 14.2.1 Fisher Information Metric

**黎曼度量**：在策略空间中，我们需要定义距离度量。

**KL 散度**作为距离：

$$
D_{KL}(\pi \| \pi') = \mathbb{E}_{s, a \sim \pi} \left[\log \frac{\pi(a|s)}{\pi'(a|s)}\right]
$$

**局部近似**（泰勒展开）：

$$
D_{KL}(\pi_{\boldsymbol{\theta}} \| \pi_{\boldsymbol{\theta} + \Delta \boldsymbol{\theta}}) \approx \frac{1}{2} \Delta \boldsymbol{\theta}^T \mathbf{F}(\boldsymbol{\theta}) \Delta \boldsymbol{\theta}
$$

其中 $\mathbf{F}(\boldsymbol{\theta})$ 是 **Fisher Information Matrix**：

$$
\mathbf{F}(\boldsymbol{\theta}) = \mathbb{E}_{s, a \sim \pi_{\boldsymbol{\theta}}} \left[\nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a|s) \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(a|s)^T\right]
$$

<div data-component="FisherInformationMatrix"></div>

**性质**：
- $\mathbf{F}$ 是对称正定矩阵
- $\mathbf{F}$ 定义了策略空间的黎曼度量
- $\mathbf{F}$ 与参数化方式有关，但自然梯度不依赖参数化

### 14.2.2 自然梯度定义

**问题**：在策略空间中，找到使 $J(\boldsymbol{\theta})$ 增加最多的方向，约束为 KL 散度固定：

$$
\max_{\Delta \boldsymbol{\theta}} \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})^T \Delta \boldsymbol{\theta} \quad \text{s.t.} \quad \frac{1}{2} \Delta \boldsymbol{\theta}^T \mathbf{F} \Delta \boldsymbol{\theta} = c
$$

**解**（拉格朗日乘数法）：

$$
\Delta \boldsymbol{\theta} \propto \mathbf{F}^{-1} \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})
$$

**自然梯度**：

$$
\tilde{\nabla}_{\boldsymbol{\theta}} J = \mathbf{F}^{-1} \nabla_{\boldsymbol{\theta}} J
$$

**自然梯度下降**：

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha \mathbf{F}^{-1} \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}_t)
$$

<div data-component="NaturalGradientVisualization"></div>

### 14.2.3 与普通梯度的关系

**普通梯度**：在**欧几里得空间**（参数空间）中的最速下降方向

$$
\nabla_{\boldsymbol{\theta}} J
$$

**自然梯度**：在**黎曼空间**（策略空间）中的最速下降方向

$$
\tilde{\nabla}_{\boldsymbol{\theta}} J = \mathbf{F}^{-1} \nabla_{\boldsymbol{\theta}} J
$$

**预条件梯度**：自然梯度是用 $\mathbf{F}^{-1}$ **预条件**的普通梯度。

**参数化不变性**：

- 假设重新参数化 $\boldsymbol{\theta}' = A \boldsymbol{\theta}$
- 普通梯度：$\nabla_{\boldsymbol{\theta}'} J = A^{-1} \nabla_{\boldsymbol{\theta}} J$（变化）
- 自然梯度：$\tilde{\nabla}_{\boldsymbol{\theta}'} J = \tilde{\nabla}_{\boldsymbol{\theta}} J$（不变！）

---

## 14.3 NPG 算法

自然策略梯度（Natural Policy Gradient）将自然梯度应用于策略梯度。

### 14.3.1 自然策略梯度定理

**策略梯度定理**：

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\pi} [\nabla_{\boldsymbol{\theta}} \log \pi(a|s) \cdot Q^{\pi}(s, a)]
$$

**自然策略梯度**：

$$
\tilde{\nabla}_{\boldsymbol{\theta}} J = \mathbf{F}^{-1} \nabla_{\boldsymbol{\theta}} J = \mathbf{F}^{-1} \mathbb{E}_{\pi} [\nabla_{\boldsymbol{\theta}} \log \pi(a|s) \cdot Q^{\pi}(s, a)]
$$

**定理**（Kakade, 2001）：存在值函数 $w^*$ 使得

$$
\tilde{\nabla}_{\boldsymbol{\theta}} J = \mathbb{E}_{\pi} [\nabla_{\boldsymbol{\theta}} \log \pi(a|s) \cdot w^*(s, a)]
$$

其中 $w^*$ 满足 **compatible value function approximation** 条件。

### 14.3.2 Compatible Function Approximation

**条件**：如果 $w(s, a)$ 满足：

1. $w(s, a) = \nabla_{\boldsymbol{\theta}} \log \pi(a|s)^T \mathbf{v}$（线性于 log-policy 梯度的特征）
2. $\mathbf{v}$ 最小化均方误差：

$$
\min_{\mathbf{v}} \mathbb{E}_{\pi} [(Q^{\pi}(s, a) - w(s, a))^2]
$$

则：

$$
\mathbf{F} \mathbf{v} = \nabla_{\boldsymbol{\theta}} J
$$

因此：

$$
\mathbf{v} = \mathbf{F}^{-1} \nabla_{\boldsymbol{\theta}} J = \tilde{\nabla}_{\boldsymbol{\theta}} J
$$

**实践意义**：我们可以用线性回归学习 $\mathbf{v}$，而不是直接计算 $\mathbf{F}^{-1}$！

### 14.3.3 实现方法

**NPG 算法流程**：

```python
# 1. 收集轨迹，计算 Q 值
trajectories = collect_trajectories(policy)
Q_values = compute_Q_values(trajectories)

# 2. 计算策略梯度
policy_grad = compute_policy_gradient(trajectories, Q_values)

# 3. 计算自然梯度
# 方法 1: 直接求逆（小规模）
F = compute_fisher_matrix(policy, trajectories)
natural_grad = np.linalg.solve(F, policy_grad)

# 方法 2: 共轭梯度法（大规模）
natural_grad = conjugate_gradient(fisher_vector_product, policy_grad)

# 方法 3: Compatible value function（线性回归）
features = compute_log_policy_gradient_features(trajectories)
v = linear_regression(features, Q_values)
natural_grad = v

# 4. 更新策略
theta = theta + alpha * natural_grad
```

---

## 14.4 与 TRPO 的联系

NPG 与 TRPO 有密切联系。

### 14.4.1 二阶近似

**回顾 TRPO**：

$$
\max_{\boldsymbol{\theta}} L(\boldsymbol{\theta}) \quad \text{s.t.} \quad \bar{D}_{KL}(\boldsymbol{\theta}_{\text{old}}, \boldsymbol{\theta}) \leq \delta
$$

**二阶近似**：

$$
L(\boldsymbol{\theta}) \approx L(\boldsymbol{\theta}_{\text{old}}) + \mathbf{g}^T (\boldsymbol{\theta} - \boldsymbol{\theta}_{\text{old}})
$$

$$
\bar{D}_{KL} \approx \frac{1}{2} (\boldsymbol{\theta} - \boldsymbol{\theta}_{\text{old}})^T \mathbf{F} (\boldsymbol{\theta} - \boldsymbol{\theta}_{\text{old}})
$$

**TRPO 解**：

$$
\boldsymbol{\theta} - \boldsymbol{\theta}_{\text{old}} = \sqrt{\frac{2\delta}{\mathbf{g}^T \mathbf{F}^{-1} \mathbf{g}}} \mathbf{F}^{-1} \mathbf{g}
$$

这正是**自然梯度方向** $\mathbf{F}^{-1} \mathbf{g}$，步长自适应选择！

### 14.4.2 Trust Region 解释

**自然梯度更新可以理解为**：
- 在策略空间的 trust region 内
- 沿自然梯度方向
- 最大化性能提升

**TRPO = NPG + Line Search + KL 约束**

---

## 14.5 实用算法

直接计算 $\mathbf{F}^{-1}$ 的复杂度是 $O(n^3)$，对大规模网络不可行。

### 14.5.1 K-FAC (Kronecker-Factored Approximate Curvature)

**动机**：Fisher 矩阵是巨大的（参数数量平方），但有结构可以利用。

**假设**（分层网络）：对于第 $l$ 层的权重 $W^{(l)}$：

$$
\text{前向}: \quad a^{(l)} = W^{(l)} x^{(l-1)}
$$

**Fisher 可以近似分解**为 Kronecker 积：

$$
\mathbf{F}^{(l)} \approx A^{(l)} \otimes S^{(l)}
$$

其中：
- $A^{(l)} = \mathbb{E}[a^{(l)} (a^{(l)})^T]$：激活的二阶矩
- $S^{(l)} = \mathbb{E}[\nabla_{a^{(l)}} L (\nabla_{a^{(l)}} L)^T]$：梯度的二阶矩

**Kronecker 积的逆**：

$$
(A \otimes S)^{-1} = A^{-1} \otimes S^{-1}
$$

**复杂度降低**：
- 原始：$O(n^3)$，$n = d_{\text{in}} \times d_{\text{out}}$
- K-FAC：$O(d_{\text{in}}^3 + d_{\text{out}}^3)$

**K-FAC 更新**：

```python
# 1. 估计 A 和 S（在 mini-batch 上）
A = compute_activation_covariance(layer, batch)
S = compute_gradient_covariance(layer, batch)

# 2. 计算逆（可以预计算和缓存）
A_inv = np.linalg.inv(A + damping * I)
S_inv = np.linalg.inv(S + damping * I)

# 3. 预条件梯度
grad_flat = gradients[layer].reshape(-1)
precond_grad = np.kron(A_inv, S_inv) @ grad_flat
precond_grad = precond_grad.reshape(gradients[layer].shape)

# 4. 更新
params[layer] -= learning_rate * precond_grad
```

### 14.5.2 计算效率优化

**其他实用方法**：

1. **Truncated Natural Gradient**：只计算 Fisher 的前 $k$ 个特征值/向量
2. **Diagonal Approximation**：只用 Fisher 的对角元素（类似 Adagrad/Adam）
3. **Block-Diagonal Approximation**：按层分块近似
4. **Empirical Fisher**：用实际梯度估计，而非理论定义

**与 Adam 的关系**：

Adam 可以看作是 **diagonal** Fisher 的近似自然梯度：

$$
\theta \leftarrow \theta - \frac{\alpha}{\sqrt{\text{diag}(\mathbf{F}) + \epsilon}} \nabla J
$$

---

## 14.6 NPG 伪代码与实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np

class NPGAgent:
    """
    Natural Policy Gradient Agent (使用共轭梯度法)
    
    Args:
        state_dim: 状态维度
        action_dim: 动作数量
        hidden_dim: 隐藏层维度
        gamma: 折扣因子
        cg_iters: 共轭梯度迭代次数
        cg_damping: 共轭梯度阻尼
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64, gamma=0.99,
                 cg_iters=10, cg_damping=0.1):
        
        self.gamma = gamma
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        
        # Policy 网络
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        
        # Value 网络（用于 baseline）
        self.value = ValueNetwork(state_dim, hidden_dim)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=1e-3)
    
    def fisher_vector_product(self, states, vector):
        """
        Fisher Information Matrix 与向量的乘积
        
        使用 Pearlmutter's trick 高效计算 F * vector
        """
        # 1. 计算平均 KL
        action_probs = self.policy(states)
        action_dist = torch.distributions.Categorical(action_probs)
        
        # 采样一个动作以计算 KL（或用当前分布）
        with torch.no_grad():
            fixed_probs = action_probs.detach()
        
        kl = (fixed_probs * (torch.log(fixed_probs) - torch.log(action_probs))).sum(dim=1).mean()
        
        # 2. 计算 KL 的梯度
        kl_grad = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([g.view(-1) for g in kl_grad])
        
        # 3. 计算 (∇KL)^T * vector
        grad_vector_product = (kl_grad_vector * vector).sum()
        
        # 4. 计算其对参数的梯度 → F * vector
        fvp = torch.autograd.grad(grad_vector_product, self.policy.parameters())
        fvp_vector = torch.cat([g.contiguous().view(-1) for g in fvp])
        
        return fvp_vector + self.cg_damping * vector
    
    def conjugate_gradient(self, states, b):
        """
        共轭梯度法求解 Fx = b
        
        Args:
            states: 状态 batch
            b: 右侧向量（policy gradient）
        
        Returns:
            x: 自然梯度
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
    
    def compute_policy_gradient(self, states, actions, advantages):
        """计算策略梯度"""
        action_probs = self.policy(states)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)
        
        # Policy gradient
        loss = -(log_probs * advantages).mean()
        
        self.policy.zero_grad()
        loss.backward()
        
        # 将梯度展平
        policy_grad = torch.cat([p.grad.view(-1) for p in self.policy.parameters()])
        
        return policy_grad
    
    def train_step(self, states, actions, returns, step_size=0.01):
        """
        NPG 训练步骤
        
        Args:
            states: 状态列表
            actions: 动作列表
            returns: 回报列表
            step_size: 更新步长
        """
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        returns = torch.FloatTensor(returns)
        
        # 更新 value function
        values = self.value(states).squeeze()
        value_loss = nn.MSELoss()(values, returns)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # 计算 advantages
        with torch.no_grad():
            advantages = returns - self.value(states).squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 计算 policy gradient
        policy_grad = self.compute_policy_gradient(states, actions, advantages)
        
        # 用共轭梯度法求自然梯度
        natural_grad = self.conjugate_gradient(states, policy_grad)
        
        # 更新策略
        offset = 0
        for param in self.policy.parameters():
            numel = param.numel()
            param.data.add_(natural_grad[offset:offset + numel].view_as(param.data), alpha=step_size)
            offset += numel
        
        return {
            'value_loss': value_loss.item(),
            'policy_grad_norm': policy_grad.norm().item(),
            'natural_grad_norm': natural_grad.norm().item()
        }


# 简化的训练循环
def train_npg(env_name='CartPole-v1', episodes=500):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = NPGAgent(state_dim, action_dim)
    
    for episode in range(episodes):
        states, actions, rewards = [], [], []
        state = env.reset()[0]
        done = False
        
        # 收集一个 episode
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = agent.policy(state_tensor)
            action = torch.distributions.Categorical(action_probs).sample().item()
            
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
        
        # 计算 returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + agent.gamma * G
            returns.insert(0, G)
        
        # NPG 更新
        metrics = agent.train_step(states, actions, returns)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {sum(rewards)}, Value Loss: {metrics['value_loss']:.3f}")
    
    return agent
```

---

## 本章小结

在本章中，我们学习了：

✅ **参数空间 vs 策略空间**：梯度方向的几何意义  
✅ **Fisher Information Matrix**：策略空间的黎曼度量  
✅ **自然梯度**：策略空间中的最速下降方向  
✅ **NPG 算法**：自然策略梯度的实现  
✅ **K-FAC**：实用的二阶优化方法  

> [!TIP]
> **核心要点**：
> - 自然梯度在策略空间中寻找最优方向，具有参数化不变性
> - Fisher Information Matrix 定义了策略空间的距离度量
> - NPG 与 TRPO 密切相关，TRPO 是带约束的 NPG
> - K-FAC 利用 Kronecker 分解降低计算复杂度
> - 实践中可用共轭梯度法或 compatible value function 计算自然梯度

> [!NOTE]
> **下一步**：
> Chapter 15 将进入**前沿主题**
>
> 包括分布式 RL、Model-based RL、Multi-agent RL、Meta-RL 等

---

## 扩展阅读

- **经典论文**：
  - Kakade (2001): A Natural Policy Gradient
  - Amari (1998): Natural Gradient Works Efficiently in Learning
  - Martens & Grosse (2015): Optimizing Neural Networks with Kronecker-factored Approximate Curvature (K-FAC)
- **理论基础**：
  - Amari: Information Geometry
  - Nielsen & Chuang: Quantum Computation (Fisher Information)
- **实现资源**：
  - Spinning Up: Natural Policy Gradient
  - K-FAC TensorFlow/PyTorch implementations

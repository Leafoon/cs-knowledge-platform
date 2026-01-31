---
title: "Chapter 13. Soft Actor-Critic (SAC)"
description: "最大熵强化学习的实践"
updated: "2026-01-29"
---

> **Learning Objectives**
> * 理解最大熵强化学习框架
> * 掌握 Soft Bellman 方程
> * 学习 SAC 算法与实现
> * 理解自动温度调整机制
> * 掌握连续控制中的 SAC 应用

---

## 13.1 最大熵框架

最大熵强化学习（Maximum Entropy RL）在标准 RL 目标中加入熵正则化。

### 13.1.1 熵正则化目标

**标准 RL 目标**：

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t)\right]
$$

**最大熵 RL 目标**：

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[\sum_{t=0}^{\infty} \gamma^t \left(r(s_t, a_t) + \alpha H(\pi(\cdot|s_t))\right)\right]
$$

其中：
- $H(\pi(\cdot|s_t)) = -\sum_a \pi(a|s_t) \log \pi(a|s_t)$：策略的熵
- $\alpha \geq 0$：温度参数（temperature），控制熵的权重

<div data-component="MaxEntropyFramework"></div>

### 13.1.2 J(π) = E[Σ r_t + α H(π(·|s_t))]

**公式解读**：

$$
\text{最大化} \quad \underbrace{r(s_t, a_t)}_{\text{奖励}} + \underbrace{\alpha H(\pi(\cdot|s_t))}_{\text{熵 bonus}}
$$

**熵的作用**：
- $H$ 大：策略接近均匀分布，更多探索
- $H$ 小：策略接近确定性，更多利用

**对于连续动作（高斯策略）**：

$$
H(\pi) = \frac{1}{2} \log((2\pi e)^d |\Sigma|)
$$

其中 $d$ 是动作维度，$\Sigma$ 是协方差矩阵。

### 13.1.3 探索-利用的自然平衡

**传统 RL 的困境**：
- 纯利用 → 局部最优
- 显式探索（ε-greedy）→ 人工设计衰减

**最大熵 RL 的优势**：
- ✅ **内在探索**：熵 bonus 自动鼓励探索
- ✅ **自适应**：早期高熵探索，后期低熵利用
- ✅ **无需人工调度**：不需要 ε 衰减

### 13.1.4 鲁棒性提升

**为什么最大熵提升鲁棒性？**

1. **多模态解**：鼓励学习所有接近最优的策略，而非单一最优策略
2. **抗干扰**：即使某些状态-动作对失效，仍有备选
3. **泛化性**：更广泛的探索 → 更好的泛化

**示例**：
- 机器人抓取：学习多种抓取姿态，而非单一最优姿态
- 游戏：学习多样的获胜策略

---

## 13.2 Soft Bellman 方程

最大熵框架下的 Bellman 方程需要修改以包含熵项。

### 13.2.1 Soft Q-function

**定义**：

$$
Q^{\text{soft}}(s_t, a_t) = r(s_t, a_t) + \gamma \mathbb{E}_{s_{t+1}, a_{t+1}} \left[Q^{\text{soft}}(s_{t+1}, a_{t+1}) + \alpha H(\pi(\cdot|s_{t+1}))\right]
$$

**展开**：

$$
Q^{\text{soft}}(s_t, a_t) = r(s_t, a_t) + \gamma \mathbb{E}_{s_{t+1}} \left[\mathbb{E}_{a_{t+1} \sim \pi} [Q^{\text{soft}}(s_{t+1}, a_{t+1}) - \alpha \log \pi(a_{t+1}|s_{t+1})]\right]
$$

**简化形式**（定义 soft V-function）：

$$
V^{\text{soft}}(s_t) = \mathbb{E}_{a_t \sim \pi} [Q^{\text{soft}}(s_t, a_t) - \alpha \log \pi(a_t|s_t)]
$$

则：

$$
Q^{\text{soft}}(s_t, a_t) = r(s_t, a_t) + \gamma \mathbb{E}_{s_{t+1}} [V^{\text{soft}}(s_{t+1})]
$$

<div data-component="SoftBellmanEquation"></div>

### 13.2.2 Soft Value Function

**Soft V-function**：

$$
V^{\text{soft}}(s) = \alpha \log \int_{\mathcal{A}} \exp\left(\frac{1}{\alpha} Q^{\text{soft}}(s, a)\right) da
$$

这是 **LogSumExp**（也称 soft-max）的连续版本。

**离散动作版本**：

$$
V^{\text{soft}}(s) = \alpha \log \sum_a \exp\left(\frac{1}{\alpha} Q^{\text{soft}}(s, a)\right)
$$

### 13.2.3 Soft Policy Iteration

**Soft 策略评估**：对固定策略 $\pi$，计算 $Q^{\pi, \text{soft}}$。

**Soft 策略改进**：

$$
\pi_{\text{new}}(a|s) \propto \exp\left(\frac{1}{\alpha} Q^{\pi_{\text{old}}, \text{soft}}(s, a)\right)
$$

**定理**（单调改进）：Soft policy iteration 保证单调改进，收敛到最优策略。

---

## 13.3 Soft Actor-Critic (SAC)

SAC 是最大熵 RL 的实用深度学习实现。

### 13.3.1 SAC 算法框架

**SAC 包含**：
1. **Actor**（策略）：$\pi_{\boldsymbol{\theta}}(a|s)$，通常是 squashed Gaussian
2. **双 Critic**（Q 函数）：$Q_{\boldsymbol{\phi}_1}, Q_{\boldsymbol{\phi}_2}$
3. **温度参数**：$\alpha$（可学习）

<div data-component="SACArchitecture"></div>

**更新流程**：

1. **Critic 更新**：最小化 soft Bellman residual
2. **Actor 更新**：最大化 $\mathbb{E}[Q(s,a) - \alpha \log \pi(a|s)]$
3. **温度更新**：调整 $\alpha$ 以匹配目标熵

### 13.3.2 自动温度调整

**动机**：$\alpha$ 的选择很重要，但难以手动调整。

**目标**：约束熵不低于目标熵 $\bar{H}$：

$$
\max_{\pi} \mathbb{E}_{\pi} \left[\sum_t r_t\right] \quad \text{s.t.} \quad \mathbb{E}_{(s_t, a_t) \sim \rho_{\pi}}[-\log \pi(a_t|s_t)] \geq \bar{H}
$$

**拉格朗日形式**：

$$
\max_{\pi} \min_{\alpha} \mathbb{E}_{\pi} \left[\sum_t r_t + \alpha (H(\pi(\cdot|s_t)) - \bar{H})\right]
$$

**温度更新**：

$$
\alpha \leftarrow \alpha - \eta_{\alpha} \nabla_{\alpha} \mathbb{E}_{a_t \sim \pi} [\alpha (\log \pi(a_t|s_t) + \bar{H})]
$$

**实践中**：

```python
# 目标熵（启发式）
target_entropy = -action_dim

# 可学习的 log(α)
log_alpha = torch.zeros(1, requires_grad=True)

# 更新
alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
alpha_optimizer.zero_grad()
alpha_loss.backward()
alpha_optimizer.step()

alpha = log_alpha.exp()
```

<div data-component="TemperatureEffect"></div>

### 13.3.3 Reparameterization Trick

**问题**：如何对 $\mathbb{E}_{a \sim \pi_{\boldsymbol{\theta}}} [Q(s, a)]$ 求导？

**直接采样**：

$$
a \sim \pi_{\boldsymbol{\theta}}(\cdot|s)
$$

梯度无法反向传播（不可微）。

**Reparameterization Trick**：

将采样分解为：
1. 确定性函数 $f_{\boldsymbol{\theta}}$
2. 独立于 $\boldsymbol{\theta}$ 的噪声 $\epsilon$

$$
a = f_{\boldsymbol{\theta}}(s, \epsilon), \quad \epsilon \sim p(\epsilon)
$$

**高斯策略示例**：

$$
\begin{align}
\mu, \sigma &= \text{NN}_{\boldsymbol{\theta}}(s) \\
\epsilon &\sim \mathcal{N}(0, I) \\
a &= \mu + \sigma \odot \epsilon
\end{align}
$$

**梯度**：

$$
\nabla_{\boldsymbol{\theta}} \mathbb{E}_{\epsilon} [Q(s, f_{\boldsymbol{\theta}}(s, \epsilon))] = \mathbb{E}_{\epsilon} [\nabla_{\boldsymbol{\theta}} Q(s, f_{\boldsymbol{\theta}}(s, \epsilon))]
$$

可以通过采样估计并反向传播！

### 13.3.4 双 Q 网络

**动机**：与 TD3 相同，避免 Q 值过高估计。

**方法**：
- 训练两个独立的 Critic：$Q_{\boldsymbol{\phi}_1}, Q_{\boldsymbol{\phi}_2}$
- 计算目标时取**最小值**：

$$
y = r + \gamma (\min(Q_{\boldsymbol{\phi}_1'}(s', a'), Q_{\boldsymbol{\phi}_2'}(s', a')) - \alpha \log \pi(a'|s'))
$$

---

## 13.4 SAC 实现细节

### 13.4.1 Squashed Gaussian Policy

**原始高斯**：

$$
\tilde{a} \sim \mathcal{N}(\mu_{\boldsymbol{\theta}}(s), \sigma_{\boldsymbol{\theta}}(s)^2)
$$

**问题**：$\tilde{a} \in \mathbb{R}^d$，无界。

**Squashing**：使用 $\tanh$ 将动作映射到有界范围：

$$
a = \tanh(\tilde{a})
$$

**Log-probability 修正**：

$$
\log \pi(a|s) = \log \pi(\tilde{a}|s) - \sum_{i=1}^{d} \log(1 - \tanh^2(\tilde{a}_i))
$$

**实现**：

```python
class SquashedGaussianPolicy(nn.Module):
    """
    Squashed Gaussian Policy for SAC
    
    Args:
        state_dim: 状态维度
        action_dim: 动作维度
        hidden_dim: 隐藏层维度
        action_bound: 动作边界
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, action_bound=1.0):
        super().__init__()
        
        self.action_bound = action_bound
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        """输出均值和方差"""
        features = self.net(state)
        mean = self.mean(features)
        log_std = torch.clamp(self.log_std(features), -20, 2)
        return mean, log_std
    
    def sample(self, state):
        """
        采样动作（使用 reparameterization trick）
        
        Returns:
            action: 采样的动作
            log_prob: Log probability
            mean: 均值（用于评估）
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # rsample = reparameterized sample
        
        # Squash
        action = torch.tanh(x_t) * self.action_bound
        
        # Log probability with squashing correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) / (self.action_bound ** 2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        mean_action = torch.tanh(mean) * self.action_bound
        
        return action, log_prob, mean_action
```

### 13.4.2 Log-Prob 计算

**变量变换公式**：

对于 $y = g(x)$，其中 $g$ 可逆：

$$
p_Y(y) = p_X(g^{-1}(y)) \left|\det \frac{\partial g^{-1}(y)}{\partial y}\right|
$$

对数形式：

$$
\log p_Y(y) = \log p_X(g^{-1}(y)) + \log \left|\det \frac{\partial g^{-1}(y)}{\partial y}\right|
$$

**对于 $a = \tanh(\tilde{a})$**：

$$
\frac{\partial \tanh(\tilde{a})}{\partial \tilde{a}} = 1 - \tanh^2(\tilde{a}) = 1 - a^2
$$

因此：

$$
\log \pi(a|s) = \log \pi(\tilde{a}|s) - \sum_i \log(1 - a_i^2)
$$

### 13.4.3 目标熵设置

**启发式**：

$$
\bar{H} = -\text{action\_dim}
$$

**原理**：
- 对于标准高斯 $\mathcal{N}(0, 1)$：$H \approx 1.42$ nat
- 对于 $d$ 维独立高斯：$H \approx 1.42 \times d$
- 使用 $-d$ 略微保守，鼓励一定的确定性

**可调整**：
- 更多探索：$\bar{H} = -0.5 \times \text{action\_dim}$
- 更少探索：$\bar{H} = -2 \times \text{action\_dim}$

---

## 13.5 SAC 变体

### 13.5.1 Discrete SAC

**问题**：原始 SAC 用于连续动作。

**离散版本**：
- 策略：$\pi(a|s) = \text{softmax}(f_{\boldsymbol{\theta}}(s))$
- 熵：$H = -\sum_a \pi(a|s) \log \pi(a|s)$
- 无需 reparameterization（直接可微）

### 13.5.2 SAC with Automatic Entropy Tuning

**标准实现**：
- 手动设置 $\alpha$（例如 0.2）

**自动调整**：
- 学习 $\log \alpha$
- 根据目标熵动态调整
- **推荐使用**

### 13.5.3 TQC (Truncated Quantile Critics)

**动机**：进一步减少过高估计。

**方法**：
- 不只用两个 Critic，用**多个**（例如 5 个）
- 取**截断均值**（例如最小的 2 个的平均）

**优势**：
- 更鲁棒的 Q 估计
- 更好的性能（特别是高维任务）

---

## 13.6 应用与优势

### 13.6.1 样本效率

**SAC vs 其他算法**：

| 算法 | 样本效率 | 原因 |
|------|---------|------|
| PPO | ⭐⭐⭐ | On-policy，丢弃旧数据 |
| TD3 | ⭐⭐⭐⭐ | Off-policy + 经验回放 |
| SAC | ⭐⭐⭐⭐⭐ | Off-policy + 熵正则化 + 更好探索 |

**实验结果**：SAC 通常用更少的样本达到相同性能。

### 13.6.2 稳定性

**SAC 的稳定性优势**：
- ✅ **熵正则化**：防止过早收敛
- ✅ **自动温度调整**：自适应探索
- ✅ **双 Q 网络**：减少过高估计
- ✅ **Soft 更新**：平滑更新

**对超参数不敏感**：default 参数通常就能工作得很好。

### 13.6.3 机器人控制

**应用领域**：
- 灵巧操作（Dexterous manipulation）
- 四足行走
- 人形机器人
- 工业机器人

**优势**：
- **样本效率**：减少真实机器人实验次数
- **鲁棒性**：对环境扰动、模型误差鲁棒
- **多模态**：学习多种解决方案

---

## 13.7 完整 SAC 实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from collections import deque
import random

class SACAgent:
    """
    SAC (Soft Actor-Critic) Agent
    
    Args:
        state_dim: 状态维度
        action_dim: 动作维度
        action_bound: 动作边界
        hidden_dim: 隐藏层维度
        gamma: 折扣因子
        tau: 软更新系数
        alpha_lr: 温度学习率
        actor_lr: Actor 学习率
        critic_lr: Critic 学习率
        automatic_entropy_tuning: 是否自动调整温度
    """
    def __init__(self, state_dim, action_dim, action_bound=1.0, hidden_dim=256,
                 gamma=0.99, tau=0.005, alpha_lr=3e-4, actor_lr=3e-4, critic_lr=3e-4,
                 automatic_entropy_tuning=True):
        
        self.action_bound = action_bound
        self.gamma = gamma
        self.tau = tau
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Actor
        self.actor = SquashedGaussianPolicy(state_dim, action_dim, hidden_dim, action_bound)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Twin Critics
        self.critic_1 = Critic(state_dim, action_dim, hidden_dim)
        self.critic_2 = Critic(state_dim, action_dim, hidden_dim)
        self.critic_1_target = Critic(state_dim, action_dim, hidden_dim)
        self.critic_2_target = Critic(state_dim, action_dim, hidden_dim)
        
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=critic_lr)
        
        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 0.2  # Fixed alpha
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=1000000)
    
    def select_action(self, state, evaluate=False):
        """
        选择动作
        
        Args:
            state: 状态
            evaluate: 是否为评估模式（使用均值）
        
        Returns:
            action: 选择的动作
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        
        if evaluate:
            _, _, action = self.actor.sample(state)
        else:
            action, _, _ = self.actor.sample(state)
        
        return action.detach().cpu().numpy()[0]
    
    def train_step(self, batch_size=256):
        """
        SAC 训练步骤
        
        Args:
            batch_size: Batch 大小
        
        Returns:
            metrics: 训练指标
        """
        # 采样 batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # 当前 alpha
        if self.automatic_entropy_tuning:
            alpha = self.log_alpha.exp()
        else:
            alpha = self.alpha
        
        # ============ 更新 Critics ============
        with torch.no_grad():
            # 采样 next actions
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            
            # 计算 target Q (取最小值)
            target_q1 = self.critic_1_target(next_states, next_actions)
            target_q2 = self.critic_2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_probs
            target_q = rewards + self.gamma * (1 - dones) * target_q
        
        # 当前 Q 值
        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)
        
        # Critic 损失
        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)
        
        # 更新 Critic 1
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        
        # 更新 Critic 2
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()
        
        # ============ 更新 Actor ============
        # 采样 new actions
        new_actions, log_probs, _ = self.actor.sample(states)
        
        # Q 值（取最小值）
        q1 = self.critic_1(states, new_actions)
        q2 = self.critic_2(states, new_actions)
        q = torch.min(q1, q2)
        
        # Actor 损失（最大化 Q - α log π）
        actor_loss = (alpha * log_probs - q).mean()
        
        # 更新 Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ============ 更新 Alpha （可选）============
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.)
        
        # ============ 软更新 Target Networks ============
        self.soft_update(self.critic_1_target, self.critic_1)
        self.soft_update(self.critic_2_target, self.critic_2)
        
        return {
            'critic_1_loss': critic_1_loss.item(),
            'critic_2_loss': critic_2_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': alpha.item() if self.automatic_entropy_tuning else alpha
        }
    
    def soft_update(self, target, source):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )


def train_sac(env_name='Pendulum-v1', total_timesteps=1000000, 
              start_timesteps=10000, batch_size=256, eval_freq=5000):
    """
    SAC 训练主循环
    
    Args:
        env_name: 环境名称
        total_timesteps: 总训练步数
        start_timesteps: 开始训练前的随机探索步数
        batch_size: Batch 大小
        eval_freq: 评估频率
    """
    # 创建环境
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = float(env.action_space.high[0])
    
    # 创建 agent
    agent = SACAgent(state_dim, action_dim, action_bound)
    
    # 训练循环
    state = env.reset()[0]
    episode_reward = 0
    episode_num = 0
    
    for t in range(total_timesteps):
        # 选择动作
        if t < start_timesteps:
            # 初期随机探索
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)
        
        # 执行动作
        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        
        # 存储到 buffer
        agent.replay_buffer.push(state, action, reward, next_state, float(done))
        
        state = next_state
        episode_reward += reward
        
        # 训练
        if t >= start_timesteps:
            metrics = agent.train_step(batch_size)
        
        # Episode 结束
        if done:
            print(f"Total T: {t+1}, Episode {episode_num+1}, Reward: {episode_reward:.2f}")
            state = env.reset()[0]
            episode_reward = 0
            episode_num += 1
        
        # 评估
        if (t + 1) % eval_freq == 0 and t >= start_timesteps:
            eval_rewards = []
            for _ in range(10):
                eval_state = eval_env.reset()[0]
                eval_reward = 0
                eval_done = False
                
                while not eval_done:
                    eval_action = agent.select_action(eval_state, evaluate=True)
                    eval_state, r, eval_done, truncated, _ = eval_env.step(eval_action)
                    eval_done = eval_done or truncated
                    eval_reward += r
                
                eval_rewards.append(eval_reward)
            
            print(f"Evaluation over 10 episodes: {np.mean(eval_rewards):.2f}")
    
    return agent

# 运行训练
if __name__ == "__main__":
    agent = train_sac(env_name='Pendulum-v1', total_timesteps=100000)
```

---

## 本章小结

在本章中，我们学习了：

✅ **最大熵框架**：在 RL 目标中加入熵正则化  
✅ **Soft Bellman 方程**：包含熵项的价值函数  
✅ **SAC 算法**：最大熵 RL 的实用深度学习实现  
✅ **自动温度调整**：动态调整探索-利用权衡  
✅ **工程实践**：Squashed Gaussian、Reparameterization Trick  

> [!TIP]
> **核心要点**：
> - 最大熵鼓励探索的多样性，提升鲁棒性和泛化性
> - SAC 结合了 off-policy、双 Critic、熵正则化的优势
> - 自动温度调整使 SAC 对超参数不敏感
> - Reparameterization trick 使策略梯度可微
> - SAC 在连续控制任务中样本效率和稳定性都很优秀

> [!NOTE]
> **下一步**：
> Chapter 14 将学习**自然策略梯度（NPG）**：
> - 参数空间 vs 策略空间
> - Fisher Information Matrix
> - 自然梯度的几何意义
> - K-FAC 等实用算法
> 
> 进入 [Chapter 14. Natural Policy Gradient](14-npg.md)

---

## 扩展阅读

- **经典论文**：
  - Haarnoja et al. (2018): Soft Actor-Critic
  - Haarnoja et al. (2018): Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL
  - Ziebart (2010): Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy
- **实现资源**：
  - Stable-Baselines3: SAC
  - CleanRL: SAC Implementation
  - rlkit: SAC
- **应用案例**：
  - 机器人灵巧操作
  - 自动驾驶
  - 工业控制

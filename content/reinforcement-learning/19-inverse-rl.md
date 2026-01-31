---
title: "Chapter 19. Inverse Reinforcement Learning"
description: "从演示中学习奖励函数"
updated: "2026-01-29"
---

> **Learning Objectives**
> * 理解逆强化学习（IRL）的问题定义
> * 掌握 Maximum Entropy IRL 方法
> * 学习 GAIL 的对抗式模仿学习
> * 了解 AIRL 的可迁移奖励学习
> * 掌握 IRL 的应用场景

---

## 19.1 IRL 问题定义

逆强化学习（Inverse Reinforcement Learning）从专家演示中学习隐含的奖励函数。

### 19.1.1 从演示中学习奖励

**正向 RL**：已知奖励 R → 学习策略 π

**逆向 RL**：已知策略（专家演示）→ 学习奖励 R

<div data-component="IRLProblemVisualization"></div>

**动机**：
- 奖励函数难以手工设计
- 专家演示更容易获得
- 从演示中推断意图

**示例**：自动驾驶
- ❌ 难以设计：如何惩罚不舒适的驾驶？
- ✅ 专家演示：观察人类驾驶员

### 19.1.2 与模仿学习的关系

| 方法 | 学习目标 | 泛化性 |
|------|---------|--------|
| **Behavioral Cloning** | 直接克隆专家策略 | 差（仅学习状态→动作） |
| **Inverse RL** | 学习奖励函数 | 好（理解任务目标） |

**IRL 的优势**：
- ✅ 理解任务的**目标**而非仅仅动作
- ✅ 更好的泛化到新环境
- ✅ 可解释性（奖励函数有意义）

### 19.1.3 奖励函数的不确定性

**问题**：给定专家演示，奖励函数不唯一。

**平凡解**：$R(s, a) = 0 \quad \forall s, a$（所有策略都最优）

**解决**：添加约束或先验
- 最大熵原理
- 简单性先验（Occam's Razor）
- 线性特征表示

---

## 19.2 Maximum Entropy IRL

使用最大熵框架消除奖励函数的歧义。

### 19.2.1 最大熵原理

**核心思想**：在满足专家演示约束的前提下，选择**熵最大**的策略。

**最大熵策略**：

$$
\pi^*(a|s) \propto \exp(Q^*(s, a))
$$

**专家策略的概率**（轨迹 τ）：

$$
P(\tau | \theta) = \frac{1}{Z(\theta)} \exp\left(\sum_t r_\theta(s_t, a_t)\right)
$$

其中 $\theta$ 是奖励函数的参数。

### 19.2.2 特征匹配

**奖励函数参数化**（线性特征）：

$$
r_\theta(s, a) = \theta^\top \phi(s, a)
$$

**目标**：匹配专家的特征期望

$$
\mathbb{E}_{\pi_E}[\phi(s, a)] \approx \mathbb{E}_{\pi_\theta}[\phi(s, a)]
$$

**最大似然学习**：

$$
\max_\theta \mathbb{E}_{\text{demo}} \left[\sum_t r_\theta(s_t, a_t)\right] - \log Z(\theta)
$$

### 19.2.3 训练流程

**Ziebart et al. (2008) 算法**：

```python
def max_ent_irl(expert_demos, features, num_iterations=100):
    """
    Maximum Entropy Inverse RL
    
    Args:
        expert_demos: 专家演示轨迹
        features: 特征函数 phi(s,a)
        num_iterations: 迭代次数
    
    Returns:
        theta: 学习的奖励参数
    """
    # 初始化奖励参数
    theta = np.zeros(feature_dim)
    
    # 计算专家特征期望
    expert_feature_exp = compute_feature_expectation(expert_demos, features)
    
    for iteration in range(num_iterations):
        # (1) 用当前奖励执行 RL
        reward = lambda s, a: theta @ features(s, a)
        policy = run_rl(reward)  # 例如用 Soft Q-learning
        
        # (2) 计算学习策略的特征期望
        policy_feature_exp = compute_feature_expectation(policy, features)
        
        # (3) 梯度步
        grad = expert_feature_exp - policy_feature_exp
        theta += learning_rate * grad
    
    return theta
```

---

## 19.3 GAIL (Generative Adversarial Imitation Learning)

使用 GAN 框架进行模仿学习。

<div data-component="GAILArchitecture"></div>

### 19.3.1 GAN 框架应用

**核心思想**：让学习策略生成的轨迹与专家轨迹"难以区分"。

**判别器** $D$：
- 输入：$(s, a)$
- 输出：$D(s, a) \in [0, 1]$（来自专家的概率）

**生成器**（策略）$\pi$：
- 生成轨迹，欺骗判别器

### 19.3.2 判别器作为奖励

**GAIL 的奖励函数**：

$$
r(s, a) = \log D(s, a)
$$

**min-max 目标**：

$$
\min_\pi \max_D \mathbb{E}_{\pi_E}[\log D(s, a)] + \mathbb{E}_{\pi}[\log(1 - D(s, a))]
$$

**解释**：
- 判别器：区分专家 vs 学习策略
- 策略：最大化 $\log D$ = 让判别器认为自己是专家

### 19.3.3 GAIL 算法

```python
import torch
import torch.nn as nn
import torch.optim as optim

class GAILDiscriminator(nn.Module):
    """
    GAIL 判别器
    
    输入: (s, a)
    输出: D(s,a) ∈ [0,1] (来自专家的概率)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出 [0,1]
        )
    
    def forward(self, state, action):
        """前向传播"""
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
    
    def get_reward(self, state, action):
        """将判别器输出转换为奖励"""
        with torch.no_grad():
            d = self.forward(state, action)
            # GAIL 奖励: log D(s,a)
            reward = torch.log(d + 1e-8)
        return reward


def train_gail(env, expert_demos, num_iterations=1000):
    """
    GAIL 训练主循环
    
    Args:
        env: 环境
        expert_demos: 专家演示 [(s,a), ...]
        num_iterations: 训练迭代次数
    """
    # 初始化
    discriminator = GAILDiscriminator(state_dim, action_dim)
    policy = PPO(state_dim, action_dim)  # 使用 PPO 作为策略
    
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=3e-4)
    
    for iteration in range(num_iterations):
        # ========== (1) 收集策略轨迹 ==========
        policy_trajs = []
        state = env.reset()
        for step in range(traj_length):
            action = policy.select_action(state)
            next_state, _, done = env.step(action)
            policy_trajs.append((state, action))
            state = next_state
            if done:
                state = env.reset()
        
        # ========== (2) 训练判别器 ==========
        for _ in range(discriminator_steps):
            # 专家数据
            expert_batch = sample_batch(expert_demos, batch_size)
            expert_states, expert_actions = expert_batch
            
            # 策略数据
            policy_batch = sample_batch(policy_trajs, batch_size)
            policy_states, policy_actions = policy_batch
            
            # 判别器损失 (BCE)
            expert_logits = discriminator(expert_states, expert_actions)
            policy_logits = discriminator(policy_states, policy_actions)
            
            disc_loss = -(torch.log(expert_logits + 1e-8).mean() +
                         torch.log(1 - policy_logits + 1e-8).mean())
            
            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()
        
        # ========== (3) 训练策略（PPO + GAIL reward）==========
        # 计算 GAIL 奖励
        gail_rewards = []
        for (s, a) in policy_trajs:
            reward = discriminator.get_reward(
                torch.tensor(s), torch.tensor(a)
            ).item()
            gail_rewards.append(reward)
        
        # 用 PPO 更新策略
        policy.update(policy_trajs, gail_rewards)
```

### 19.3.4 与 IRL 的联系

**定理**（Ho & Ermon, 2016）：
GAIL 等价于最小化专家策略和学习策略之间的 **Jensen-Shannon 散度**。

$$
\text{GAIL} \Leftrightarrow \min_\pi \text{JSD}(\pi_E || \pi)
$$

---

## 19.4 AIRL (Adversarial Inverse RL)

学习可迁移的奖励函数。

### 19.4.1 可迁移的奖励函数

**GAIL 的问题**：判别器依赖于当前环境的动态。

**AIRL 目标**：学习不依赖于环境动态的奖励函数。

**AIRL 判别器**：

$$
D(s, a, s') = \frac{\exp(r_\theta(s, a) + \gamma V(s') - V(s))}
              {\exp(r_\theta(s, a) + \gamma V(s') - V(s)) + \pi(a|s)}
$$

### 19.4.2 解耦奖励与策略

**关键**：AIRL 显式分离奖励函数 $r_\theta$ 和值函数 $V$。

**优势**：
- ✅ 学习的奖励函数可迁移到新环境
- ✅ 不受环境动态影响

**实现**：

```python
class AIRLDiscriminator(nn.Module):
    """
    AIRL 判别器
    
    包含奖励网络和值函数网络
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # 奖励网络
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 值函数网络
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.gamma = 0.99
    
    def get_reward(self, state, action):
        """获取学习的奖励"""
        x = torch.cat([state, action], dim=-1)
        return self.reward_net(x)
    
    def forward(self, state, action, next_state, log_prob):
        """
        AIRL 判别器前向传播
        
        Args:
            state, action, next_state: 转移
            log_prob: log π(a|s)
        
        Returns:
            D(s,a,s'): 来自专家的概率
        """
        r = self.get_reward(state, action)
        v = self.value_net(state)
        v_next = self.value_net(next_state)
        
        # f = r + γV(s') - V(s)
        f = r + self.gamma * v_next - v
        
        # D = exp(f) / (exp(f) + π(a|s))
        logits = f - log_prob
        return torch.sigmoid(logits)
```

---

## 19.5 应用场景

<div data-component="RewardRecovery"></div>

### 19.5.1 机器人模仿

**场景**：让机器人学习人类的操作技能。

**方法**：
1. 人类演示操作（例如抓取物体）
2. 用 IRL/GAIL 学习奖励或策略
3. 机器人在类似场景中泛化

**优势**：无需手工设计奖励函数。

### 19.5.2 自动驾驶

**场景**：从人类驾驶数据学习驾驶策略。

**挑战**：
- 多模态行为（不同驾驶员风格）
- 安全性要求高

**应用**：
- Waymo：从人类驾驶演示学习
- Comma.ai：开源数据集

### 19.5.3 游戏 AI

**场景**：从高手玩家的游戏录像学习。

**示例**：
- StarCraft II：从职业选手录像学习策略
- Dota 2：模仿人类比赛风格

---

## 本章小结

在本章中，我们学习了：

✅ **IRL 问题定义**：从演示学习奖励，优于直接克隆  
✅ **Maximum Entropy IRL**：特征匹配、最大熵原理  
✅ **GAIL**：对抗式模仿、判别器作为奖励  
✅ **AIRL**：可迁移奖励函数、解耦奖励与策略  
✅ **应用场景**：机器人、自动驾驶、游戏 AI  

> [!TIP]
> **核心要点**：
> - 逆强化学习从专家演示中推断奖励函数
> - MaxEnt IRL 通过特征匹配学习奖励
> - GAIL 使用 GAN 框架，判别器输出作为奖励信号
> - AIRL 学习可迁移的奖励函数，不依赖环境动态
> - IRL 比行为克隆有更好的泛化性和可解释性
> - 应用于机器人模仿、自动驾驶等复杂任务

> [!NOTE]
> **下一步**：
> Chapter 20 将学习**模仿学习**：
> - Behavioral Cloning
> - DAgger
> - 从观察中学习
> 
> 进入 [Chapter 20. Imitation Learning](20-imitation-learning.md)

---

## 扩展阅读

- **经典论文**：
  - Ng & Russell (2000): Algorithms for Inverse Reinforcement Learning
  - Ziebart et al. (2008): Maximum Entropy Inverse Reinforcement Learning
  - Ho & Ermon (2016): Generative Adversarial Imitation Learning (GAIL)
  - Fu et al. (2018): Learning Robust Rewards with Adversarial Inverse RL (AIRL)
- **理论基础**：
  - Abbeel & Ng (2004): Apprenticeship Learning via Inverse RL
  - Finn et al. (2016): Guided Cost Learning
- **实现资源**：
  - imitation library (PyTorch)
  - stable-baselines3 + GAIL
  - rlkit: AIRL implementation

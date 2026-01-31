---
title: "Chapter 17. Exploration Strategies"
description: "好奇心驱动的智能探索"
updated: "2026-01-29"
---

> **Learning Objectives**
> * 理解探索-利用困境的本质
> * 掌握 Count-Based 探索方法
> * 学习 ICM（好奇心驱动探索）
> * 了解 RND 的新颖性检测
> * 掌握 Go-Explore 的记忆与返回机制

---

## 17.1 探索-利用困境

探索（Exploration）与利用（Exploitation）的平衡是 RL 的核心挑战。

### 17.1.1 Multi-Armed Bandit 问题

**问题设定**：
- K 个老虎机（arms），每个有未知的奖励分布
- 目标：最大化累积奖励

**困境**：
- **探索**：尝试新的 arm，获取信息
- **利用**：选择当前最优 arm，获取奖励

<div data-component="ExplorationVsExploitation"></div>

### 17.1.2 ε-greedy 的局限性

**ε-greedy**：
$$
a = \begin{cases}
\arg\max_a Q(s, a) & \text{概率 } 1-\epsilon \\
\text{random}(a) & \text{概率 } \epsilon
\end{cases}
$$

**问题**：
- ❌ **无目的探索**：随机动作不考虑信息增益
- ❌ **固定 ε**：不适应任务复杂度
- ❌ **高维失效**：在大状态空间中随机探索效率极低

**示例**：Montezuma's Revenge
- 状态空间：~10^9 个状态
- ε-greedy：几乎不可能随机到达有奖励的状态

### 17.1.3 探索的必要性

**稀疏奖励环境**：
- 大部分状态 r = 0
- 需要长时间探索才能找到奖励

**解决方案**：
1. **内在动机**（Intrinsic Motivation）：给予探索奖励
2. **好奇心**（Curiosity）：新颖状态 → 内在奖励
3. **记忆**（Memory）：记住有趣的状态并返回

---

## 17.2 Count-Based 探索

基于访问计数的探索方法。

### 17.2.1 访问计数奖励

**基本思想**：访问少的状态更值得探索。

**Count-based bonus**：

$$
r^+ = r + \beta \cdot \frac{1}{\sqrt{N(s)}}
$$

其中 $N(s)$ 是状态 $s$ 的访问次数。

<div data-component="CountBasedBonus"></div>

**优势**：
- 简单直观
- 鼓励访问新状态

### 17.2.2 UCB (Upper Confidence Bound)

**UCB 公式**（Bandit）：

$$
a^* = \arg\max_a \left(\bar{Q}(a) + c\sqrt{\frac{\log t}{N(a)}}\right)
$$

**两项平衡**：
- $\bar{Q}(a)$：平均奖励（利用）
- $c\sqrt{\frac{\log t}{N(a)}}$：置信区间（探索）

**在 RL 中的应用**：
- UCB-RL
- UCRL2

### 17.2.3 高维状态空间的挑战

**问题**：连续或高维状态空间中，几乎每个状态都只访问一次。

**解决方案**：
1. **Pseudo-Count**：学习密度模型估计"计数"
2. **Hash-based**：使用状态哈希近似计数
3. **学习表示**：在潜在空间计数

---

## 17.3 好奇心驱动探索

使用预测误差作为内在奖励。

### 17.3.1 内在动机（Intrinsic Motivation）

**总奖励**：

$$
r_{\text{total}} = r_{\text{extrinsic}} + \beta \cdot r_{\text{intrinsic}}
$$

**内在奖励来源**：
- 新颖性（Novelty）
- 预测误差（Prediction Error）
- 学习进步（Learning Progress）

### 17.3.2 预测误差作为奖励

**动机**：难以预测的状态 = 新颖状态。

**Forward Model**：

$$
\hat{s}_{t+1} = f(s_t, a_t)
$$

**预测误差**：

$$
r_{\text{intrinsic}} = \|s_{t+1} - \hat{s}_{t+1}\|^2
$$

**问题**：
- ❌ **随机性陷阱**：随机环境（例如白噪声）总是难以预测
- ❌ **学习饱和**：模型训练好后，所有状态都易预测

### 17.3.3 ICM (Intrinsic Curiosity Module)

**核心思想**：在特征空间（而非原始状态）预测。

<div data-component="ICMArchitecture"></div>

**ICM 架构**：

1. **Feature Encoder** $\phi$：
   $$
   \phi_t = \phi(s_t)
   $$
   将状态编码到特征空间

2. **Inverse Model**（逆向模型）：
   $$
   \hat{a}_t = g(\phi_t, \phi_{t+1})
   $$
   预测导致状态转移的动作

3. **Forward Model**（前向模型）：
   $$
   \hat{\phi}_{t+1} = f(\phi_t, a_t)
   $$
   预测下一个特征

**内在奖励**：

$$
r_{\text{intrinsic}} = \frac{\eta}{2} \|\hat{\phi}_{t+1} - \phi_{t+1}\|^2
$$

**训练损失**：

$$
\mathcal{L} = \lambda_I \mathcal{L}_I + (1 - \beta_I) \mathcal{L}_F
$$

其中：
- $\mathcal{L}_I = \|\hat{a}_t - a_t\|^2$：inverse model 损失
- $\mathcal{L}_F = \|\hat{\phi}_{t+1} - \phi_{t+1}\|^2$：forward model 损失

**关键优势**：
- ✅ **学习相关特征**：Inverse model 强制 $\phi$ 只编码与动作相关的信息
- ✅ **过滤随机性**：无法控制的随机性（例如背景噪声）不会被编码

**实现**：

```python
import torch
import torch.nn as nn

class ICMModule(nn.Module):
    """
    Intrinsic Curiosity Module
    
    Args:
        state_dim: 状态维度
        action_dim: 动作数量
        feature_dim: 特征维度
    """
    def __init__(self, state_dim, action_dim, feature_dim=256):
        super().__init__()
        
        # Feature Encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
        # Inverse Model: (φ_t, φ_{t+1}) → a_t
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # Forward Model: (φ_t, a_t) → φ_{t+1}
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
    def forward(self, state, next_state, action):
        """
        计算 ICM 损失和内在奖励
        
        Args:
            state: 当前状态
            next_state: 下一状态
            action: 动作（one-hot）
        
        Returns:
            intrinsic_reward: 内在奖励
            inverse_loss: 逆向模型损失
            forward_loss: 前向模型损失
        """
        # 编码特征
        phi = self.encoder(state)
        phi_next = self.encoder(next_state)
        
        # Inverse model
        phi_concat = torch.cat([phi, phi_next], dim=-1)
        pred_action = self.inverse_model(phi_concat)
        inverse_loss = nn.CrossEntropyLoss()(pred_action, action)
        
        # Forward model
        action_onehot = nn.functional.one_hot(action, num_classes=pred_action.shape[-1]).float()
        phi_action = torch.cat([phi, action_onehot], dim=-1)
        pred_phi_next = self.forward_model(phi_action)
        forward_loss = 0.5 * (pred_phi_next - phi_next.detach()).pow(2).mean(dim=-1)
        
        # 内在奖励 = 前向预测误差
        intrinsic_reward = forward_loss
        
        return intrinsic_reward, inverse_loss, forward_loss.mean()
```

---

## 17.4 Random Network Distillation (RND)

RND 通过随机网络蒸馏检测新颖性。

### 17.4.1 随机网络蒸馏

**核心思想**：
- 固定一个随机初始化的网络 $f$（target network）
- 训练一个预测网络 $\hat{f}$（predictor network）
- 预测误差 = 新颖性

**RND 内在奖励**：

$$
r_{\text{intrinsic}} = \|\hat{f}(s) - f(s)\|^2
$$

<div data-component="RNDNovelty"></div>

**为什么有效？**

- **常见状态**：predictor 见过多次，预测准确，误差小
- **新颖状态**：predictor 没见过，预测不准，误差大

### 17.4.2 新颖性检测

**训练**：

```python
class RNDModule(nn.Module):
    """
    Random Network Distillation
    
    Args:
        state_dim: 状态维度
        output_dim: 输出维度
    """
    def __init__(self, state_dim, output_dim=512):
        super().__init__()
        
        # Target network (固定，不训练)
        self.target_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
        # 冻结参数
        for param in self.target_net.parameters():
            param.requires_grad = False
        
        # Predictor network (训练)
        self.predictor_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, state):
        """
        计算内在奖励
        
        Args:
            state: 状态
        
        Returns:
            intrinsic_reward: 预测误差
        """
        with torch.no_grad():
            target = self.target_net(state)
        
        pred = self.predictor_net(state)
        intrinsic_reward = (pred - target).pow(2).mean(dim=-1)
        
        return intrinsic_reward
    
    def train_step(self, states):
        """训练 predictor"""
        target = self.target_net(states).detach()
        pred = self.predictor_net(states)
        loss = (pred - target).pow(2).mean()
        return loss
```

**奖励归一化**（关键！）：

```python
# Running mean and std
class RunningMeanStd:
    def __init__(self):
        self.mean = 0
        self.var = 1
        self.count = 1e-4
    
    def update(self, x):
        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean += delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count
    
    def normalize(self, x):
        return (x - self.mean) / (torch.sqrt(torch.tensor(self.var)) + 1e-8)

# 使用
reward_rms = RunningMeanStd()
intrinsic_reward = rnd.forward(state)
reward_rms.update(intrinsic_reward)
normalized_reward = reward_rms.normalize(intrinsic_reward)
```

### 17.4.3 与 ICM 的对比

| 特性 | ICM | RND |
|------|-----|-----|
| **模型** | Forward + Inverse | Random + Predictor |
| **训练** | 需要动作信息 | 只需状态 |
| **特征** | 学习的特征 | 随机特征 |
| **复杂度** | 高 | 低 |
| **效果** | 优秀 | 优秀 |

**实践建议**：
- RND 实现更简单，通常是首选
- ICM 在需要学习动作相关特征时更好

---

## 17.5 Go-Explore

通过记忆有趣状态并返回来系统探索。

### 17.5.1 记忆有趣状态

**Go-Explore 流程**：

1. **Explore**：随机探索，记录所有访问过的状态
2. **Archive**：维护一个状态档案（cell archive）
3. **Return**：选择一个有趣的状态，返回它
4. **Explore Again**：从该状态继续探索

<div data-component="GoExploreProcess"></div>

**状态表示（Cell）**：
- 低分辨率的状态描述（例如 agent 位置）
- 降低状态空间维度

### 17.5.2 返回并探索

**选择策略**：
- 优先选择访问少的 cell
- 或接近边界的 cell

**返回方法**：
1. **Phase 1（探索）**：保存轨迹，记录如何到达每个状态
2. **Return**：重放轨迹到达目标状态
3. **Continue**：从该状态继续探索

**伪代码**：

```python
def go_explore(env, max_iterations):
    archive = {}  # cell → (trajectory, reward)
    
    for iteration in range(max_iterations):
        # 1. 选择一个 cell 返回
        cell = select_cell_from_archive(archive)
        
        # 2. 返回到该 cell
        env.reset()
        trajectory = archive[cell]['trajectory']
        for action in trajectory:
            env.step(action)
        
        # 3. 从该 cell 继续探索
        while not done:
            action = random_policy()  # 或使用学习的策略
            next_state, reward, done = env.step(action)
            
            # 4. 更新 archive
            cell = downscale(next_state)
            if cell not in archive or reward > archive[cell]['reward']:
                archive[cell] = {
                    'trajectory': trajectory + [action],
                    'reward': reward
                }
```

### 17.5.3 Montezuma's Revenge 突破

**成就**：Go-Explore 在 Montezuma's Revenge 上取得了人类水平的性能。

**关键**：
- 系统的探索（不依赖随机性）
- 记忆机制（能返回有趣状态）
- 确定性环境（可以精准返回）

**局限性**：
- 需要确定性环境（或低随机性）
- 需要合适的状态表示（cell 定义）
- 返回机制可能不适用于所有环境

---

## 17.6 Noisy Networks

在参数空间加噪声实现探索。

### 17.6.1 参数空间噪声

**标准网络**：

$$
y = f(x; \theta)
$$

**Noisy Network**：

$$
y = f(x; \theta + \epsilon \odot \sigma)
$$

其中：
- $\epsilon \sim \mathcal{N}(0, 1)$：标准高斯噪声
- $\sigma$：可学习的噪声参数

**Noisy Linear Layer**：

$$
y = (\mu^w + \sigma^w \odot \epsilon^w) x + \mu^b + \sigma^b \odot \epsilon^b
$$

**实现**：

```python
class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for NoisyNet-DQN
    
    Args:
        in_features: 输入维度
        out_features: 输出维度
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # 可学习参数 μ
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        
        # 可学习参数 σ
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        
        # 注册噪声（不参与训练）
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """初始化参数"""
        mu_range = 1 / (self.in_features ** 0.5)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        
        self.weight_sigma.data.fill_(0.5 / (self.in_features ** 0.5))
        self.bias_sigma.data.fill_(0.5 / (self.out_features ** 0.5))
    
    def reset_noise(self):
        """重新采样噪声"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    @staticmethod
    def _scale_noise(size):
        """因子化噪声"""
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()
    
    def forward(self, x):
        """前向传播"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return nn.functional.linear(x, weight, bias)
```

### 17.6.2 自适应探索

**优势**：
- ✅ **状态相关探索**：不同状态有不同的探索策略
- ✅ **自适应**：网络学习何时探索、何时利用
- ✅ **无需 ε 调度**：不需要手动设置 ε 衰减

**在 DQN 中使用**：

```python
class NoisyDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            NoisyLinear(128, 128),
            nn.ReLU(),
            NoisyLinear(128, action_dim)
        )
    
    def forward(self, state):
        return self.net(state)
    
    def reset_noise(self):
        """每个 episode 开始时重置噪声"""
        for module in self.net.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()
```

---

## 17.7 Thompson Sampling

贝叶斯方法的探索策略。

### 17.7.1 后验采样

**基本思想**：
1. 维护参数的后验分布 $p(\theta | \mathcal{D})$
2. 每次采样一个参数 $\theta \sim p(\theta | \mathcal{D})$
3. 根据该参数选择动作

**优势**：自然平衡探索与利用（不确定性高 = 更多探索）。

### 17.7.2 贝叶斯 RL

**Bayes-Adaptive MDP**：
- 将信念状态（belief state）纳入状态空间
- 最优策略在信念空间中平衡探索-利用

**实践挑战**：
- 维护后验分布计算昂贵
- 高维参数空间的采样困难

**近似方法**：
- **Dropout as Bayesian Approximation**
- **Bootstrapped DQN**：训练多个 Q 网络，采样选择

---

## 本章小结

在本章中，我们学习了：

✅ **探索-利用困境**：RL 的核心挑战  
✅ **Count-Based 探索**：基于访问计数的奖励  
✅ **ICM**：好奇心驱动，学习相关特征  
✅ **RND**：随机网络蒸馏，简单高效  
✅ **Go-Explore**：记忆与返回，突破难题  
✅ **Noisy Networks**：参数空间噪声  
✅ **Thompson Sampling**：贝叶斯探索  

> [!TIP]
> **核心要点**：
> - 稀疏奖励环境需要智能探索策略
> - 内在奖励（好奇心）是强大的探索驱动
> - ICM 通过逆向模型学习相关特征，过滤无关随机性
> - RND 实现简单，效果优秀，是实践首选
> - Go-Explore 通过记忆机制实现系统探索
> - Noisy Networks 提供状态相关的自适应探索
> - 探索奖励需要归一化以保持训练稳定

> [!NOTE]
> **下一步**：
> Chapter 18 将学习**层次化 RL**：
> - Options 框架
> - Feudal RL
> - 技能发现（DIAYN）
> 
> 进入 [Chapter 18. Hierarchical RL](18-hierarchical-rl.md)

---

## 扩展阅读

- **经典论文**：
  - Pathak et al. (2017): Curiosity-driven Exploration by Self-supervised Prediction (ICM)
  - Burda et al. (2019): Exploration by Random Network Distillation (RND)
  - Ecoffet et al. (2019): First Return Then Explore (Go-Explore)
  - Fortunato et al. (2018): Noisy Networks for Exploration
- **理论基础**：
  - Sutton & Barto: Chapter 2 (Multi-armed Bandits)
  - Schmidhuber (1991): Curious model-building control systems
- **实现资源**：
  - OpenAI Baselines: PPO + ICM
  - stable-baselines3: Curiosity module
  - RND PyTorch implementation

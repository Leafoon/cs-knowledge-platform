---
title: "第0章：强化学习概览"
description: "从零开始理解强化学习：核心概念、历史发展、应用场景与环境准备"
date: "2026-01-30"
---

# 第0章：强化学习概览

欢迎来到强化学习（Reinforcement Learning, RL）的世界。在这一章中，我们将从宏观视角理解 RL 的本质、发展历程以及它与其他机器学习范式的区别，并动手搭建第一个 RL 程序。

---

## 0.1 什么是强化学习？

### 0.1.1 与监督学习、无监督学习的区别

机器学习的三大范式各有特点：

| 范式 | 数据形式 | 学习目标 | 典型应用 | 核心挑战 |
|:---|:---|:---|:---|:---|
| **监督学习** | $(x, y)$ 标注对 | 学习映射 $f: x \to y$ | 图像分类、语音识别 | 需要大量标注数据 |
| **无监督学习** | $x$ 无标签数据 | 发现数据内在结构 | 聚类、降维、生成 | 评估标准模糊 |
| **强化学习** | $(s, a, r, s')$ 交互序列 | 最大化长期累积奖励 | 游戏、机器人、推荐 | 延迟奖励、探索-利用权衡 |

**强化学习的核心特征**：
1.  **试错学习（Trial-and-Error）**：Agent 通过尝试不同的动作来发现哪些行为能带来高奖励。
2.  **延迟奖励（Delayed Reward）**：当前动作的影响可能在多步之后才显现（如下棋中的一步好棋可能在 20 步后才体现价值）。
3.  **探索-利用权衡（Exploration-Exploitation）**：既要利用已知的好策略（Exploitation）以获取高分，又要尝试未知区域（Exploration）以发现潜藏的更优策略。这就像去餐厅点菜：是点以前吃过觉得好吃的（Exploitation），还是尝试一道新菜（Exploration）？

> **如果把人生看作是一个强化学习问题...**
> 
> "小时候我们通过不停地摔倒学会走路（试错），上学备考时为了高分刷题（延迟奖励），工作后在稳定薪水和创业梦想之间抉择（探索与利用）。RL 不仅仅是一种算法，更是理解智能体如何适应世界的通用框架。"

**类比理解**：
- **监督学习** 像在学校听老师讲课，老师直接告诉你"这道题答案是 A"。
- **强化学习** 像婴儿学走路，摔倒了（负奖励），站稳了（正奖励），通过不断试错逐渐掌握平衡技巧，没有老师逐步指导。

---

#### 数据分布视角的深度对比

**监督学习的 i.i.d. 假设**

在监督学习中，我们假设训练数据 $(x_i, y_i)$ 是从某个固定的联合分布 $p_{\text{data}}(x, y)$ 中**独立同分布（i.i.d.）**采样得到的：

$$
\mathcal{D}_{\text{train}} = \{(x_1, y_1), (x_2, y_2), \ldots, (x_N, y_N)\} \sim p_{\text{data}}(x, y)
$$

学习目标是最小化经验风险（Empirical Risk）：

$$
\min_{\theta} \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f_\theta(x_i), y_i)
$$

**关键特性**：
- 数据分布 $p_{\text{data}}$ 在训练和测试时保持不变（或至少假设如此）
- 每个样本 $(x_i, y_i)$ 之间相互独立
- 模型的预测不会影响未来的数据分布

**强化学习的分布漂移（Distribution Shift）**

在强化学习中，数据分布**不是固定的**，而是由当前策略 $\pi$ 决定的。Agent 的策略会影响它访问哪些状态，从而改变数据分布。

形式化地，在策略 $\pi$ 下，状态的**折现访问频率（Discounted State Visitation Distribution）** 定义为：

$$
d^\pi(s) = (1-\gamma) \sum_{t=0}^\infty \gamma^t P(s_t = s | s_0, \pi)
$$

其中：
- $s_0$ 是初始状态分布
- $P(s_t = s | s_0, \pi)$ 是在策略 $\pi$ 下，$t$ 步后到达状态 $s$ 的概率
- $(1-\gamma)$ 是归一化常数，确保 $\sum_s d^\pi(s) = 1$

**核心洞察**：
- 当策略 $\pi$ 改变时，$d^\pi(s)$ 也会改变
- 这意味着 Agent 收集的数据分布是**非平稳的（Non-stationary）**
- 这是 RL 与 SL 最本质的区别之一

**可视化示例：策略如何影响状态分布**

考虑一个简单的网格世界（GridWorld），Agent 从左上角出发，目标是到达右下角的宝藏：

```
S . . . .
. # # # .
. . . . G
```

- `S`: 起点
- `G`: 终点（宝藏）
- `#`: 障碍物
- `.`: 可通行区域

**策略 A（随机策略）**：每步随机选择上下左右
- 状态访问分布：几乎均匀分布在所有可达状态
- 很少到达终点 `G`

**策略 B（最优策略）**：沿最短路径前进
- 状态访问分布：集中在最短路径上
- 频繁到达终点 `G`

这两个策略产生的数据分布 $d^{\pi_A}(s)$ 和 $d^{\pi_B}(s)$ 完全不同！

---

#### 反馈信号的本质差异

**监督学习：即时、精确的标签**

在监督学习中，每个输入 $x$ 都有一个对应的标签 $y$，这个标签是：
- **即时的**：在看到 $x$ 的同时就知道 $y$
- **精确的**：$y$ 明确告诉你正确答案（如分类任务中的类别，回归任务中的数值）
- **密集的**：每个样本都有标签

例如，在图像分类中：
- 输入 $x$：一张猫的图片
- 标签 $y$：类别 "猫"
- 损失函数：交叉熵 $\mathcal{L} = -\log p_\theta(\text{猫} | x)$

**强化学习：延迟、稀疏、标量奖励**

在强化学习中，Agent 收到的反馈是：
- **延迟的**：当前动作的影响可能在多步之后才显现
- **稀疏的**：很多时候奖励为 0，只有在特定时刻才有非零奖励
- **标量的**：奖励只是一个数字，不直接告诉你"应该采取哪个动作"

例如，在围棋中：
- 状态 $s$：当前棋盘局面
- 动作 $a$：在某个位置落子
- 奖励 $r$：
  - 中间步骤：$r = 0$（没有即时反馈）
  - 游戏结束：$r = +1$（赢）或 $r = -1$（输）

**Credit Assignment Problem（归因问题）**

这是 RL 最核心的挑战之一：**如何将最终的奖励归因到之前的每一步动作？**

形式化地，考虑一个长度为 $T$ 的轨迹：

$$
\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \ldots, s_{T-1}, a_{T-1}, r_T, s_T)
$$

总回报（Return）为：

$$
R(\tau) = \sum_{t=0}^{T-1} \gamma^t r_{t+1}
$$

问题是：**哪些动作 $a_t$ 对最终的 $R(\tau)$ 贡献最大？**

**实际案例：围棋中的妙手**

在 AlphaGo 与李世石的第二局中，AlphaGo 在第 37 手下出了一步"神之一手"，这步棋在当时看起来很奇怪，但在 100 多手之后才显现出其战略价值，最终帮助 AlphaGo 获胜。

如果用监督学习的思路，我们无法直接知道"第 37 手应该下在这里"，因为没有即时的标签告诉我们这步棋的好坏。只有通过 RL 的方法，利用最终的胜负结果（$r = +1$），结合价值函数估计，才能逐步学会这种深层次的战略思维。

**解决 Credit Assignment 的方法**

1. **时序差分学习（Temporal Difference Learning）**
   - 不等到游戏结束，而是利用 Bootstrapping 估计中间状态的价值
   - TD 误差：$\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$

2. **资格迹（Eligibility Traces）**
   - 追踪每个状态-动作对的"资格"（最近被访问的程度）
   - TD($\lambda$) 算法结合了多步回报的优势

3. **优势函数（Advantage Function）**
   - $A(s, a) = Q(s, a) - V(s)$
   - 衡量动作 $a$ 相对于平均水平的优势

---

#### 学习目标的形式化对比

**监督学习的优化目标**

给定数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$，监督学习的目标是找到参数 $\theta$，使得经验风险最小：

$$
\theta^* = \arg\min_\theta \frac{1}{N} \sum_{i=1}^N \mathcal{L}(f_\theta(x_i), y_i)
$$

常见的损失函数：
- **分类**：交叉熵损失 $\mathcal{L}_{\text{CE}} = -\sum_c y_c \log \hat{y}_c$
- **回归**：均方误差 $\mathcal{L}_{\text{MSE}} = (y - \hat{y})^2$

**强化学习的优化目标**

RL 的目标是找到最优策略 $\pi^*$，使得期望累积奖励最大：

$$
\pi^* = \arg\max_\pi \mathbb{E}_{\tau \sim \pi} [R(\tau)]
$$

其中轨迹 $\tau$ 的回报定义为：

$$
R(\tau) = \sum_{t=0}^{T-1} \gamma^t r_{t+1}
$$

展开期望，我们有：

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi} [R(\tau)] = \mathbb{E}_{s_0} \left[ V^\pi(s_0) \right]
$$

其中 $V^\pi(s)$ 是状态价值函数，定义为从状态 $s$ 开始，遵循策略 $\pi$ 的期望回报：

$$
V^\pi(s) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^\infty \gamma^t r_{t+1} \mid s_0 = s \right]
$$

**轨迹分布的形式化**

轨迹 $\tau = (s_0, a_0, r_1, s_1, \ldots)$ 的概率分布为：

$$
p_\pi(\tau) = p(s_0) \prod_{t=0}^{T-1} \pi(a_t | s_t) p(s_{t+1}, r_{t+1} | s_t, a_t)
$$

其中：
- $p(s_0)$：初始状态分布
- $\pi(a_t | s_t)$：策略（动作选择概率）
- $p(s_{t+1}, r_{t+1} | s_t, a_t)$：环境的转移动态（通常未知）

**关键差异总结**

| 维度 | 监督学习 | 强化学习 |
|:---|:---|:---|
| **优化目标** | 最小化损失 $\mathcal{L}$ | 最大化回报 $J(\pi)$ |
| **数据分布** | 固定 $p_{\text{data}}(x, y)$ | 策略依赖 $p_\pi(\tau)$ |
| **反馈信号** | 即时标签 $y$ | 延迟奖励 $r$ |
| **学习方式** | 单步优化 | 序列决策 |
| **评估指标** | 准确率、F1 等 | 累积奖励 |

---

#### 探索-利用困境的理论基础

**Multi-Armed Bandit（多臂老虎机）问题**

为了理解探索-利用困境，我们先考虑一个简化的 RL 问题：**Multi-Armed Bandit（MAB）**。

**问题设定**：
- 有 $K$ 个老虎机（臂），每个臂 $k$ 有一个未知的期望奖励 $\mu_k$
- 每次你可以选择拉一个臂 $a_t \in \{1, 2, \ldots, K\}$
- 拉臂 $k$ 后，你会收到一个随机奖励 $r_t \sim \mathcal{N}(\mu_k, \sigma^2)$
- 目标：在 $T$ 步内最大化总奖励 $\sum_{t=1}^T r_t$

**Regret（遗憾）的定义**

Regret 衡量你的策略与最优策略之间的差距：

$$
\text{Regret}_T = T \cdot \mu^* - \sum_{t=1}^T r_t
$$

其中 $\mu^* = \max_k \mu_k$ 是最优臂的期望奖励。

**理想情况**：如果你一开始就知道哪个臂最好，直接拉 $T$ 次，总奖励为 $T \cdot \mu^*$，Regret 为 0。

**现实情况**：你不知道 $\mu_k$，需要通过试验来估计，这就产生了探索-利用的权衡。

**三种经典策略**

**1. $\epsilon$-Greedy**

- 以概率 $1 - \epsilon$ 选择当前估计最好的臂（Exploitation）
- 以概率 $\epsilon$ 随机选择一个臂（Exploration）

$$
a_t = \begin{cases}
\arg\max_k \hat{\mu}_k & \text{with probability } 1 - \epsilon \\
\text{Uniform}(\{1, \ldots, K\}) & \text{with probability } \epsilon
\end{cases}
$$

其中 $\hat{\mu}_k = \frac{1}{N_k} \sum_{t: a_t = k} r_t$ 是臂 $k$ 的平均奖励估计，$N_k$ 是拉臂 $k$ 的次数。

**2. Upper Confidence Bound (UCB)**

- 选择"乐观估计"最高的臂
- 对不确定性高的臂给予奖励（鼓励探索）

$$
a_t = \arg\max_k \left[ \hat{\mu}_k + c \sqrt{\frac{\log t}{N_k}} \right]
$$

其中：
- $\hat{\mu}_k$：臂 $k$ 的平均奖励（Exploitation）
- $c \sqrt{\frac{\log t}{N_k}}$：置信区间上界（Exploration Bonus）
- $c$ 是超参数，控制探索强度

**直觉**：
- 如果臂 $k$ 被拉的次数 $N_k$ 很少，$\sqrt{\frac{\log t}{N_k}}$ 很大，UCB 会倾向于选择它（探索）
- 随着 $N_k$ 增加，置信区间缩小，逐渐转向利用

**3. Thompson Sampling（贝叶斯方法）**

- 为每个臂的期望奖励 $\mu_k$ 维护一个后验分布 $p(\mu_k | \mathcal{D}_k)$
- 每次从后验分布中采样 $\tilde{\mu}_k \sim p(\mu_k | \mathcal{D}_k)$
- 选择采样值最大的臂：$a_t = \arg\max_k \tilde{\mu}_k$

**代码示例：10臂老虎机的三种策略对比**

```python
import numpy as np
import matplotlib.pyplot as plt

class MultiArmedBandit:
    """K 臂老虎机环境"""
    def __init__(self, K=10, seed=42):
        np.random.seed(seed)
        self.K = K
        # 真实的期望奖励（未知）
        self.true_means = np.random.randn(K)
        self.optimal_arm = np.argmax(self.true_means)
        self.optimal_reward = self.true_means[self.optimal_arm]
    
    def pull(self, arm):
        """拉臂，返回奖励（加高斯噪声）"""
        return np.random.randn() + self.true_means[arm]

class EpsilonGreedy:
    """ε-Greedy 策略"""
    def __init__(self, K, epsilon=0.1):
        self.K = K
        self.epsilon = epsilon
        self.Q = np.zeros(K)  # 估计的期望奖励
        self.N = np.zeros(K)  # 每个臂被拉的次数
    
    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.K)  # 探索
        else:
            return np.argmax(self.Q)  # 利用
    
    def update(self, arm, reward):
        self.N[arm] += 1
        # 增量更新平均值
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]

class UCB:
    """UCB 策略"""
    def __init__(self, K, c=2.0):
        self.K = K
        self.c = c
        self.Q = np.zeros(K)
        self.N = np.zeros(K)
        self.t = 0
    
    def select_arm(self):
        self.t += 1
        # 初始化：每个臂至少拉一次
        if self.t <= self.K:
            return self.t - 1
        
        # UCB 公式
        ucb_values = self.Q + self.c * np.sqrt(np.log(self.t) / (self.N + 1e-5))
        return np.argmax(ucb_values)
    
    def update(self, arm, reward):
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]

class ThompsonSampling:
    """Thompson Sampling（假设高斯奖励）"""
    def __init__(self, K, prior_mean=0.0, prior_std=1.0):
        self.K = K
        # 后验分布参数（高斯-高斯共轭）
        self.mu = np.full(K, prior_mean)  # 后验均值
        self.tau = np.full(K, 1.0 / prior_std**2)  # 后验精度
        self.N = np.zeros(K)
    
    def select_arm(self):
        # 从每个臂的后验分布中采样
        samples = np.random.randn(self.K) / np.sqrt(self.tau) + self.mu
        return np.argmax(samples)
    
    def update(self, arm, reward):
        self.N[arm] += 1
        # 贝叶斯更新（假设奖励方差为 1）
        self.tau[arm] += 1.0
        self.mu[arm] = (self.mu[arm] * (self.tau[arm] - 1) + reward) / self.tau[arm]

def run_experiment(bandit, agent, T=1000):
    """运行实验，返回累积 Regret"""
    regrets = []
    cumulative_regret = 0
    
    for t in range(T):
        arm = agent.select_arm()
        reward = bandit.pull(arm)
        agent.update(arm, reward)
        
        # 计算 Regret
        regret = bandit.optimal_reward - reward
        cumulative_regret += regret
        regrets.append(cumulative_regret)
    
    return regrets

# 运行对比实验
K = 10
T = 1000
num_runs = 100

bandit = MultiArmedBandit(K=K)

strategies = {
    'ε-Greedy (ε=0.1)': lambda: EpsilonGreedy(K, epsilon=0.1),
    'UCB (c=2)': lambda: UCB(K, c=2.0),
    'Thompson Sampling': lambda: ThompsonSampling(K)
}

results = {name: [] for name in strategies}

for name, agent_fn in strategies.items():
    for run in range(num_runs):
        agent = agent_fn()
        regrets = run_experiment(bandit, agent, T)
        results[name].append(regrets)

# 绘图
plt.figure(figsize=(10, 6))
for name, regrets_list in results.items():
    mean_regret = np.mean(regrets_list, axis=0)
    std_regret = np.std(regrets_list, axis=0)
    plt.plot(mean_regret, label=name)
    plt.fill_between(range(T), mean_regret - std_regret, mean_regret + std_regret, alpha=0.2)

plt.xlabel('Time Step')
plt.ylabel('Cumulative Regret')
plt.title('Multi-Armed Bandit: Strategy Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**预期结果**：
- **$\epsilon$-Greedy**：Regret 线性增长（因为始终有 $\epsilon$ 概率探索次优臂）
- **UCB**：Regret 对数增长 $O(\log T)$（理论最优）
- **Thompson Sampling**：Regret 对数增长，实践中通常优于 UCB

**从 MAB 到完整 RL**

Multi-Armed Bandit 是 RL 的特例（只有一个状态）。在完整的 RL 问题中：
- 有多个状态 $s \in \mathcal{S}$
- 探索-利用权衡在每个状态都存在
- 需要平衡"探索新状态"和"利用已知好策略"

常见的探索策略：
1. **$\epsilon$-Greedy**：在动作选择时加入随机性
2. **Boltzmann Exploration**：根据 Q 值的 Softmax 分布采样
3. **Entropy Regularization**：在策略优化目标中加入熵项（如 SAC）
4. **Intrinsic Motivation**：为探索新状态提供内在奖励（如 Curiosity-driven RL）

---

**本节小结**

我们从三个维度深入对比了 RL 与监督学习/无监督学习：

1. **数据分布**：RL 的数据分布由策略决定，是非平稳的
2. **反馈信号**：RL 的奖励是延迟、稀疏的，导致 Credit Assignment 问题
3. **学习目标**：RL 优化的是期望累积奖励，而非单步损失

我们还通过 Multi-Armed Bandit 问题深入理解了探索-利用困境，并实现了三种经典策略。这些概念是理解后续所有 RL 算法的基础。

### 0.1.2 核心要素：Agent, Environment, State, Action, Reward

### 0.1.2 核心要素：Agent, Environment, State, Action, Reward

强化学习的过程可以用一个标准的五元组 $(S, A, P, R, \gamma)$ 来形式化描述，但在此概览章节，我们先从直观概念入手：

```mermaid
graph LR
    Agent[Agent<br/>(智能体)] -- Action $a_t$ --> Env[Environment<br/>(环境)]
    Env -- State $s_{t+1}$ <br/> Reward $r_{t+1}$ --> Agent
    style Agent fill:#f9f,stroke:#333,stroke-width:2px
    style Env fill:#ccf,stroke:#333,stroke-width:2px
```

<div data-component="AgentEnvironmentLoop"></div>

---

#### Agent-Environment 交互循环的深度剖析

**交互协议的形式化定义**

在每个时间步 $t$，Agent 与环境的交互遵循以下协议：

1. **Agent 观察状态**：接收环境的观测 $o_t$（或完整状态 $s_t$）
2. **Agent 选择动作**：根据策略 $\pi$ 选择动作 $a_t \sim \pi(\cdot | o_t)$
3. **环境响应**：
   - 根据转移动态 $p(s_{t+1}, r_{t+1} | s_t, a_t)$ 生成新状态和奖励
   - 返回观测 $o_{t+1}$ 和奖励 $r_{t+1}$
4. **循环继续**：$t \leftarrow t + 1$，回到步骤 1

**时间步的伪代码**

```python
# 初始化
s_0 = env.reset()
t = 0

while not done:
    # 1. Agent 选择动作
    a_t = agent.select_action(s_t)
    
    # 2. 环境执行动作
    s_{t+1}, r_{t+1}, done, info = env.step(a_t)
    
    # 3. Agent 学习（可选，取决于算法）
    agent.update(s_t, a_t, r_{t+1}, s_{t+1}, done)
    
    # 4. 更新状态
    s_t = s_{t+1}
    t += 1
```

**信息流向图（带时间下标）**

```
时刻 t:
    Agent 状态: s_t
    ↓ (策略 π)
    动作: a_t
    ↓ (环境转移 P)
    环境反馈: (s_{t+1}, r_{t+1})
    ↓
时刻 t+1:
    Agent 状态: s_{t+1}
    ...
```

**实现示例：自定义简单环境**

让我们实现一个简单的"网格世界"环境，理解环境的内部机制：

```python
import numpy as np
from typing import Tuple, Optional

class GridWorld:
    """
    简单的网格世界环境
    - 5x5 网格
    - Agent 从 (0,0) 出发，目标是到达 (4,4)
    - 动作：上下左右
    - 奖励：到达目标 +10，每步 -0.1（鼓励快速到达）
    """
    def __init__(self, size=5):
        self.size = size
        self.action_space = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.action_space)
        
        # 动作对应的位置变化
        self.action_effects = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }
        
        self.goal = (size-1, size-1)
        self.reset()
    
    def reset(self) -> Tuple[int, int]:
        """重置环境，返回初始状态"""
        self.agent_pos = (0, 0)
        return self.agent_pos
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, dict]:
        """
        执行动作，返回 (next_state, reward, done, info)
        
        这是环境的核心：转移动态 P(s', r | s, a)
        """
        # 计算新位置
        dx, dy = self.action_effects[action]
        new_x = max(0, min(self.size-1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.size-1, self.agent_pos[1] + dy))
        
        self.agent_pos = (new_x, new_y)
        
        # 计算奖励
        if self.agent_pos == self.goal:
            reward = 10.0
            done = True
        else:
            reward = -0.1  # 每步小惩罚，鼓励快速到达
            done = False
        
        info = {'steps': 1}
        return self.agent_pos, reward, done, info
    
    def render(self):
        """可视化当前状态"""
        grid = np.full((self.size, self.size), '.')
        grid[self.agent_pos] = 'A'
        grid[self.goal] = 'G'
        
        for row in grid:
            print(' '.join(row))
        print()

# 使用示例
env = GridWorld(size=5)
state = env.reset()
env.render()

for _ in range(10):
    action = np.random.randint(4)  # 随机动作
    state, reward, done, info = env.step(action)
    print(f"Action: {env.action_space[action]}, Reward: {reward:.2f}")
    env.render()
    
    if done:
        print("Goal reached!")
        break
```

**关键洞察**：
- 环境的 `step()` 函数封装了转移动态 $P(s', r | s, a)$
- Agent 无法直接访问这个函数的内部逻辑（黑盒）
- Agent 只能通过反复交互来"探测"环境的规律

---

#### State vs Observation：MDP vs POMDP

**完全可观测：马尔可夫决策过程（MDP）**

当 Agent 能够观察到环境的**完整状态** $s_t$ 时，问题被建模为 **MDP（Markov Decision Process）**。

**MDP 的形式化定义**：五元组 $(S, A, P, R, \gamma)$

1. **$S$**：状态空间（State Space）
   - 所有可能状态的集合
   - 例如：棋盘的所有可能局面

2. **$A$**：动作空间（Action Space）
   - 所有可能动作的集合
   - 例如：围棋的所有合法落子位置

3. **$P$**：状态转移概率（Transition Dynamics）
   - $P(s' | s, a) = \Pr(S_{t+1} = s' | S_t = s, A_t = a)$
   - 给定当前状态 $s$ 和动作 $a$，转移到下一状态 $s'$ 的概率
   - **马尔可夫性质**：未来只依赖于现在，与过去无关
     $$P(S_{t+1} | S_t, A_t, S_{t-1}, A_{t-1}, \ldots) = P(S_{t+1} | S_t, A_t)$$

4. **$R$**：奖励函数（Reward Function）
   - $R(s, a)$ 或 $R(s, a, s')$
   - 在状态 $s$ 采取动作 $a$ 后获得的期望奖励

5. **$\gamma$**：折现因子（Discount Factor）
   - $\gamma \in [0, 1]$
   - 控制对未来奖励的重视程度

**马尔可夫性质的直观理解**

> "现在包含了所有相关的历史信息"

例如，在国际象棋中：
- **状态 $s$**：当前棋盘局面（包含所有棋子的位置）
- **马尔可夫性**：下一步的局面只取决于当前局面和你的走法，与之前的走法序列无关
- 当前棋盘已经"总结"了所有历史信息

**部分可观测：POMDP（Partially Observable MDP）**

在很多实际问题中，Agent 无法观察到完整状态，只能看到部分信息（观测 $o_t$）。

**POMDP 的形式化定义**：七元组 $(S, A, P, R, \Omega, O, \gamma)$

在 MDP 基础上增加：
- **$\Omega$**：观测空间（Observation Space）
- **$O$**：观测函数 $O(o | s, a) = \Pr(O_t = o | S_t = s, A_{t-1} = a)$

**核心挑战**：Agent 需要根据观测序列 $o_{1:t}$ 推断真实状态 $s_t$

**Belief State（信念状态）**

由于无法直接观察 $s_t$，Agent 维护一个**信念状态** $b_t(s)$，表示对当前状态的概率分布：

$$
b_t(s) = \Pr(S_t = s | o_{1:t}, a_{1:t-1})
$$

这是一个**贝叶斯滤波**问题，可以递归更新：

$$
b_{t+1}(s') = \frac{O(o_{t+1} | s') \sum_s P(s' | s, a_t) b_t(s)}{\sum_{s'} O(o_{t+1} | s') \sum_s P(s' | s, a_t) b_t(s)}
$$

**实际案例对比**

| 环境 | 可观测性 | 说明 |
|:---|:---|:---|
| **围棋、国际象棋** | 完全可观测（MDP） | 棋盘是公开的，双方都能看到完整局面 |
| **扑克牌** | 部分可观测（POMDP） | 看不到对手的手牌，需要根据出牌历史推断 |
| **自动驾驶** | 部分可观测（POMDP） | 传感器只能看到周围环境，看不到远处或被遮挡的物体 |
| **FPS 游戏** | 部分可观测（POMDP） | 只能看到视野内的敌人，墙后的敌人需要推断 |
| **Atari 游戏（单帧）** | 部分可观测（POMDP） | 单张图片无法判断物体的速度 |
| **Atari 游戏（堆叠帧）** | 近似完全可观测 | 通过堆叠 4 帧图片，可以推断速度信息 |

**处理 POMDP 的方法**

1. **Frame Stacking（帧堆叠）**
   - 将最近的 $k$ 个观测拼接：$o_t' = [o_t, o_{t-1}, \ldots, o_{t-k+1}]$
   - DQN 使用 4 帧堆叠来推断 Atari 游戏中物体的速度

2. **循环神经网络（RNN/LSTM）**
   - 使用 LSTM 隐藏状态 $h_t$ 来"记忆"历史信息
   - $h_t = \text{LSTM}(o_t, h_{t-1})$
   - $a_t = \pi(h_t)$

3. **Transformer**
   - 使用自注意力机制处理观测序列
   - 可以捕捉长距离依赖关系

**代码示例：Frame Stacking**

```python
import numpy as np
from collections import deque

class FrameStack:
    """
    将最近的 k 帧堆叠成一个观测
    用于处理部分可观测问题（如 Atari 游戏）
    """
    def __init__(self, k=4):
        self.k = k
        self.frames = deque(maxlen=k)
    
    def reset(self, obs):
        """重置时，用初始观测填充所有帧"""
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs()
    
    def step(self, obs):
        """添加新观测，自动丢弃最旧的帧"""
        self.frames.append(obs)
        return self._get_obs()
    
    def _get_obs(self):
        """返回堆叠后的观测"""
        return np.stack(self.frames, axis=0)

# 使用示例
import gymnasium as gym

env = gym.make("Pong-v5")
frame_stack = FrameStack(k=4)

obs, info = env.reset()
stacked_obs = frame_stack.reset(obs)
print(f"单帧形状: {obs.shape}")  # (210, 160, 3)
print(f"堆叠后形状: {stacked_obs.shape}")  # (4, 210, 160, 3)

# 现在 Agent 可以从 4 帧中推断球的速度方向
```

---

#### Action Space 的设计哲学

动作空间的设计直接影响算法的选择和性能。

**离散动作空间（Discrete Action Space）**

**定义**：有限个离散选项

$$
\mathcal{A} = \{a_1, a_2, \ldots, a_n\}
$$

**表示方式**：
- One-hot encoding：$a = [0, 0, 1, 0, \ldots, 0]$（第 3 个动作）
- 整数索引：$a = 2$

**适用算法**：
- **Value-based**：DQN, Rainbow, C51
  - 对每个动作估计 Q 值：$Q(s, a_1), Q(s, a_2), \ldots, Q(s, a_n)$
  - 选择最大 Q 值的动作：$a^* = \arg\max_a Q(s, a)$

**优势**：
- 简单直观
- 可以用 $\arg\max$ 直接选择最优动作

**劣势**：
- **维度灾难**：如果有多个独立的离散选择，组合数爆炸
  - 例如：10 个开关，每个有 2 个状态，总共 $2^{10} = 1024$ 种组合
- 无法表示连续控制（如方向盘角度）

**连续动作空间（Continuous Action Space）**

**定义**：实数值向量

$$
\mathcal{A} = \mathbb{R}^d \quad \text{或} \quad \mathcal{A} = [a_{\min}, a_{\max}]^d
$$

**表示方式**：
- 高斯分布：$a \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))$
- 确定性策略：$a = \mu_\theta(s)$

**适用算法**：
- **Policy Gradient**：REINFORCE, PPO, TRPO
  - 直接输出动作分布的参数（均值和方差）
- **Deterministic Policy Gradient**：DDPG, TD3, SAC
  - 输出确定性动作，使用 Critic 估计梯度

**优势**：
- 适用于连续控制任务（机器人、自动驾驶）
- 可以精细调节动作（如油门 0.73，而非只能选 0 或 1）

**劣势**：
- 探索困难（连续空间无限大）
- 无法用 $\arg\max$ 找最优动作（需要优化算法）

**Reparameterization Trick（重参数化技巧）**

在连续动作空间中，我们通常需要从策略分布中采样：

$$
a \sim \pi_\theta(a | s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))
$$

但直接采样无法反向传播梯度。**重参数化技巧**将随机性分离出来：

$$
a = \mu_\theta(s) + \sigma_\theta(s) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

现在 $a$ 是 $\theta$ 的确定性函数，可以反向传播！

**代码示例：不同 Action Space 的处理**

```python
import torch
import torch.nn as nn
import torch.distributions as dist

class DiscretePolicy(nn.Module):
    """离散动作空间的策略网络"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state):
        logits = self.net(state)
        return dist.Categorical(logits=logits)
    
    def select_action(self, state):
        action_dist = self(state)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob

class ContinuousPolicy(nn.Module):
    """连续动作空间的策略网络（高斯策略）"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # 限制在 [-1, 1]
        )
        # 学习对数标准差（确保 σ > 0）
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        mean = self.mean_net(state)
        std = torch.exp(self.log_std)
        return dist.Normal(mean, std)
    
    def select_action(self, state):
        action_dist = self(state)
        # 重参数化采样
        action = action_dist.rsample()  # rsample 支持梯度
        log_prob = action_dist.log_prob(action).sum(dim=-1)
        return action.detach().numpy(), log_prob

# 使用示例
state = torch.randn(1, 4)  # batch_size=1, state_dim=4

# 离散动作
discrete_policy = DiscretePolicy(state_dim=4, action_dim=2)
action, log_prob = discrete_policy.select_action(state)
print(f"离散动作: {action}, log_prob: {log_prob.item():.4f}")

# 连续动作
continuous_policy = ContinuousPolicy(state_dim=4, action_dim=2)
action, log_prob = continuous_policy.select_action(state)
print(f"连续动作: {action}, log_prob: {log_prob.item():.4f}")
```

**混合动作空间的处理**

某些任务同时包含离散和连续动作（如星际争霸：选择单位 + 移动位置）。

**解决方案**：
1. **分层策略**：先选离散动作，再选连续参数
2. **Multi-head 网络**：一个网络输出多个分支
3. **Parameterized Action Space**：将离散动作参数化

---

#### Reward Engineering：艺术与科学

**The Reward Hypothesis（奖励假说）**

> "所有我们所说的'目标'和'目的'，都可以被归结为：最大化接收到的标量信号（奖励）的累积和。"  
> —— *Richard Sutton*

这是 RL 的哲学基础：**所有智能行为都可以通过奖励信号来引导**。

**但现实很复杂**：如何设计奖励函数，使得"最大化奖励"等价于"完成任务"？

**Reward Shaping（奖励塑造）**

**问题**：稀疏奖励导致学习困难

例如，在迷宫中：
- **稀疏奖励**：只有到达终点才有 +1，其他时候全是 0
- **问题**：Agent 可能永远找不到终点（探索困难）

**解决方案**：添加中间奖励（Shaping Reward）

例如：
- 每靠近终点一步，给 +0.1
- 远离终点，给 -0.1

**但要小心！错误的 Reward Shaping 会改变最优策略**

**Potential-based Shaping（基于势函数的塑造）**

Ng et al. (1999) 证明了一种**保持最优策略不变**的 Shaping 方法：

$$
F(s, a, s') = \gamma \Phi(s') - \Phi(s)
$$

其中 $\Phi: S \to \mathbb{R}$ 是**势函数**（Potential Function）。

**定理**：如果将原始奖励 $R(s, a, s')$ 替换为 $R'(s, a, s') = R(s, a, s') + F(s, a, s')$，则最优策略不变。

**直觉**：
- $\Phi(s)$ 可以理解为"状态 $s$ 离目标的距离"
- $F(s, a, s')$ 奖励"缩短距离"的动作
- 但由于折现因子 $\gamma$，长期累积奖励不变

**证明（简化版）**：

原始累积奖励：
$$
G_t = \sum_{k=0}^\infty \gamma^k R_{t+k+1}
$$

加入 Shaping 后：
$$
\begin{align}
G_t' &= \sum_{k=0}^\infty \gamma^k [R_{t+k+1} + \gamma \Phi(s_{t+k+1}) - \Phi(s_{t+k})] \\
&= \sum_{k=0}^\infty \gamma^k R_{t+k+1} + \sum_{k=0}^\infty \gamma^k [\gamma \Phi(s_{t+k+1}) - \Phi(s_{t+k})] \\
&= G_t + \sum_{k=0}^\infty [\gamma^{k+1} \Phi(s_{t+k+1}) - \gamma^k \Phi(s_{t+k})] \\
&= G_t + [-\Phi(s_t) + \lim_{k \to \infty} \gamma^{k+1} \Phi(s_{t+k+1})] \\
&= G_t - \Phi(s_t) \quad \text{(假设 } \lim_{k \to \infty} \gamma^k \Phi(s_k) = 0 \text{)}
\end{align}
$$

由于 $\Phi(s_t)$ 只依赖于初始状态，不影响策略的相对优劣！

**常见的 Reward Hacking 案例**

**1. OpenAI 的船竞速 Agent**

- **任务**：赛船比赛，目标是最快完成赛道
- **奖励设计**：每经过一个检查点 +10 分
- **问题**：Agent 学会了在起点附近打转，反复触发同一个检查点刷分，而不是完成赛道！

**2. YouTube 推荐系统**

- **目标**：最大化用户长期满意度
- **错误奖励**：点击率（CTR）
- **问题**：推荐标题党、低质内容（短期点击高，长期用户流失）
- **正确奖励**：观看时长 + 用户留存率

**3. 机器人抓取**

- **任务**：抓取物体并放入盒子
- **错误奖励**：手爪与物体的距离
- **问题**：Agent 学会了把手爪放在物体上方，但不实际抓取（距离为 0，但任务失败）
- **正确奖励**：物体是否在盒子里（稀疏但正确）

**稀疏奖励的解决方案**

1. **Curiosity-driven Exploration（好奇心驱动探索）**
   - 为"访问新状态"提供内在奖励（Intrinsic Reward）
   - ICM (Intrinsic Curiosity Module)：奖励"难以预测"的状态转移

2. **Hindsight Experience Replay (HER)**
   - "事后诸葛亮"：即使失败了，也假装"目标就是到达失败的地方"
   - 从失败中学习

3. **Reward Shaping with Domain Knowledge**
   - 利用领域知识设计中间奖励
   - 使用 Potential-based Shaping 保证正确性

**代码示例：Potential-based Reward Shaping**

```python
import numpy as np

class MazeEnv:
    """迷宫环境（稀疏奖励）"""
    def __init__(self, size=10):
        self.size = size
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        self.reset()
    
    def reset(self):
        self.pos = self.start
        return self.pos
    
    def step(self, action):
        # 移动逻辑（省略）
        new_pos = self._move(self.pos, action)
        self.pos = new_pos
        
        # 原始稀疏奖励
        if self.pos == self.goal:
            reward = 1.0
            done = True
        else:
            reward = 0.0
            done = False
        
        return self.pos, reward, done, {}
    
    def _move(self, pos, action):
        # 简化：上下左右移动
        moves = [(-1,0), (1,0), (0,-1), (0,1)]
        dx, dy = moves[action]
        new_x = max(0, min(self.size-1, pos[0] + dx))
        new_y = max(0, min(self.size-1, pos[1] + dy))
        return (new_x, new_y)

class ShapedMazeEnv(MazeEnv):
    """加入 Potential-based Shaping 的迷宫"""
    def __init__(self, size=10, gamma=0.99):
        super().__init__(size)
        self.gamma = gamma
    
    def potential(self, pos):
        """势函数：负的曼哈顿距离"""
        return -abs(pos[0] - self.goal[0]) - abs(pos[1] - self.goal[1])
    
    def step(self, action):
        old_pos = self.pos
        new_pos, reward, done, info = super().step(action)
        
        # 添加 Shaping Reward: F(s,a,s') = γΦ(s') - Φ(s)
        shaping = self.gamma * self.potential(new_pos) - self.potential(old_pos)
        shaped_reward = reward + shaping
        
        return new_pos, shaped_reward, done, info

# 对比实验
env_sparse = MazeEnv(size=10)
env_shaped = ShapedMazeEnv(size=10)

print("稀疏奖励环境：")
s = env_sparse.reset()
for _ in range(5):
    s, r, done, _ = env_sparse.step(np.random.randint(4))
    print(f"  State: {s}, Reward: {r:.2f}")

print("\nShaped 奖励环境：")
s = env_shaped.reset()
for _ in range(5):
    s, r, done, _ = env_shaped.step(np.random.randint(4))
    print(f"  State: {s}, Reward: {r:.2f}")
```

---

#### Discount Factor $\gamma$ 的深层含义

**数学意义：未来奖励的现值折现**

折现因子 $\gamma \in [0, 1]$ 控制对未来奖励的重视程度：

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^\infty \gamma^k r_{t+k+1}
$$

**直觉**：
- $\gamma = 0$：只关心即时奖励（短视）
- $\gamma = 1$：所有未来奖励同等重要（远视）
- $\gamma = 0.99$：100 步后的奖励只值现在的 $0.99^{100} \approx 0.366$ 倍

**收敛性保证**

如果 $\gamma < 1$ 且奖励有界（$|r_t| \leq R_{\max}$），则累积奖励有界：

$$
G_t \leq \sum_{k=0}^\infty \gamma^k R_{\max} = \frac{R_{\max}}{1 - \gamma}
$$

这保证了价值函数的收敛性。

**时间视野（Effective Horizon）**

$\gamma$ 决定了 Agent 的"有效时间视野"：

$$
H_{\text{eff}} = \frac{1}{1 - \gamma}
$$

例如：
- $\gamma = 0.9 \Rightarrow H_{\text{eff}} = 10$ 步
- $\gamma = 0.99 \Rightarrow H_{\text{eff}} = 100$ 步
- $\gamma = 0.999 \Rightarrow H_{\text{eff}} = 1000$ 步

**实际选择**

| 任务类型 | 推荐 $\gamma$ | 原因 |
|:---|:---|:---|
| 短期任务（CartPole） | 0.95 - 0.99 | 快速收敛 |
| 长期任务（围棋） | 0.99 - 0.999 | 需要长远规划 |
| 无限视野任务 | 0.99 | 平衡收敛性与长期性能 |

**代码示例：不同 $\gamma$ 的影响**

```python
import numpy as np
import matplotlib.pyplot as plt

def compute_return(rewards, gamma):
    """计算折现回报"""
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
    return G

# 模拟一个奖励序列
rewards = [1, 0, 0, 0, 10]  # 第 5 步有大奖励

gammas = [0.0, 0.5, 0.9, 0.99, 1.0]
returns = [compute_return(rewards, g) for g in gammas]

plt.figure(figsize=(10, 6))
plt.bar(gammas, returns)
plt.xlabel('Discount Factor γ')
plt.ylabel('Total Return')
plt.title('Impact of Discount Factor on Return')
plt.grid(True, alpha=0.3)
plt.show()

for g, G in zip(gammas, returns):
    print(f"γ = {g:.2f}: Return = {G:.2f}")
```

**预期输出**：
```
γ = 0.00: Return = 1.00   (只看第 1 步)
γ = 0.50: Return = 1.62   (第 5 步的 10 被折现为 0.62)
γ = 0.90: Return = 7.56
γ = 0.99: Return = 10.41
γ = 1.00: Return = 11.00  (所有奖励同等重要)
```

---

**本节小结**

我们深入探讨了 RL 的五大核心要素：

1. **Agent-Environment 交互**：形式化的交互协议与实现
2. **State vs Observation**：MDP vs POMDP，马尔可夫性质，Belief State
3. **Action Space**：离散 vs 连续，算法选择，重参数化技巧
4. **Reward Engineering**：Reward Shaping，Potential-based 方法，Reward Hacking 案例
5. **Discount Factor**：数学意义，时间视野，实际选择

这些概念是理解所有 RL 算法的基础。在后续章节中，我们将看到这些概念如何在具体算法中体现。

### 0.1.3 RL 的应用场景

强化学习在以下领域取得了突破性成果：

**1. 游戏 AI**
*   **Atari 游戏**：DeepMind 的 DQN 算法（2015）在 49 款 Atari 游戏中达到或超越人类水平。
*   **围棋**：AlphaGo（2016）击败世界冠军李世石，AlphaZero（2017）仅通过自我对弈超越所有人类围棋知识。
*   **星际争霸 II**：AlphaStar（2019）达到职业选手水平。
*   **Dota 2**：OpenAI Five（2019）在 5v5 团队对抗中击败世界冠军队伍。

**2. 机器人控制**
*   **机械臂抓取**：通过 RL 学习精确抓取各种形状的物体（RGB-Stacking, RoboSumo）。
*   **四足机器人步态**：训练机器狗在复杂地形中行走、跳跃（ANYmal, Spot）。
*   **人形机器人**：学习保持平衡、行走、翻滚等复杂运动（Humanoid, Atlas）。

**3. 推荐系统**
*   **YouTube 视频推荐**：将推荐问题建模为序列决策，最大化用户长期观看时长而非短期点击率。
*   **电商广告投放**：动态调整广告出价策略，平衡曝光量与转化率。

**4. 大语言模型 (LLM) 与推理 (Reasoning)**
*   **RLHF (InstructGPT)**：通过 PPO 算法，让模型学会遵循人类指令，减少有害输出。
*   **Reasoning Chains (OpenAI o1 / DeepSeek-R1)**：
    *   最新的前沿方向是 **Training-time / Test-time Scaling**。
    *   利用 RL 让模型探索思维链（Chain of Thought），这类训练通常使用**过程奖励（Process Reward）**或**结果验证（Outcome Verification）**。
    *   *Self-Correction*：模型在推理过程中通过 RL 学会了自我检查和修正错误。

**5. 其他应用**
*   **自动驾驶**：路径规划、决策制定。
*   **金融交易**：自动化交易策略优化。
*   **能源调度**：数据中心冷却系统优化（Google 使用 RL 节省 40% 能耗）。

### 0.1.4 RL 的挑战

尽管 RL 强大，但也面临诸多挑战：

| 挑战 | 描述 | 解决方向 |
|:---|:---|:---|
| **延迟奖励 (Credit Assignment)** | "功成不必在我"：很难判断哪一步操作对最终胜利贡献最大。 | 价值函数估计、TD($\lambda$)、GAE |
| **探索-利用权衡** | 走老路稳妥但无法更强，探新路可能踩坑。 | Entropy Regularization、UCB、Intrinsic Motivation (好奇心) |
| **样本效率低** | 需要海量交互（DQN 玩 Atari 需数百万帧）。 | Model-based RL、Transfer Learning |
| **Sim-to-Real Gap** | 仿真环境里练得好好的，上真机就偏瘫。 | Domain Randomization、Real-world Fine-tuning |
| **The Deadly Triad (死亡三角)** | 当同时结合 **Function Approximation**、**Bootstrapping**、**Off-policy** 时，训练极易发散。 | 目标网络 (Target Network)、Double Q-learning |

---

## 0.2 历史发展脉络

<div data-component="RLTimelineEvolution"></div>

### 0.2.1 早期：动态规划（Bellman, 1950s）

Richard Bellman 在 1957 年提出了**最优控制理论**和**动态规划（Dynamic Programming, DP）**方法，奠定了 RL 的数学基础。

**核心贡献**：
*   **Bellman 方程**：状态价值 $V(s)$ 可以被分解为"即时奖励"与"未来价值"两部分，这种递归结构是 RL 最核心的数学支柱。
    $$ V(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s') \right] $$

**局限**：需要完整的环境模型（转移概率 $P$），且计算复杂度随状态数指数增长（"维度灾难"）。

### 0.2.2 表格方法：Q-learning（Watkins, 1989）

Christopher Watkins 在其博士论文中提出了 **Q-learning** 算法，这是第一个 **model-free**（无需环境模型）且 **off-policy**（可以从历史数据中学习）的 RL 算法。

**核心思想**：
*   维护一个表格 $Q(s, a)$，记录在状态 $s$ 采取动作 $a$ 的价值。
*   通过与环境交互，逐步更新 $Q$ 值：
    $$ Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)] $$
*   **收敛性保证**：在满足一定条件下（所有状态-动作对被无限次访问），$Q$ 表格会收敛到最优值 $Q^*$。

**局限**：只能处理小规模离散状态空间（如 10x10 网格世界）。对于 Atari 游戏（$256^{84 \times 84 \times 4} \approx 10^{67970}$ 种状态），表格方法完全不可行。

### 0.2.3 深度 RL 时代：DQN（Mnih et al., 2015）

DeepMind 的 Volodymyr Mnih 等人在 *Nature* 发表了划时代的论文《Human-level control through deep reinforcement learning》，提出 **DQN（Deep Q-Network）**。

**核心创新**：
1.  **深度神经网络作为函数逼近器**：用卷积神经网络（CNN）来近似 $Q(s, a)$，输入是游戏画面（84x84 灰度图），输出是每个动作的 Q 值。
2.  **经验回放（Experience Replay）**：将 $(s, a, r, s')$ 存入回放缓冲区，训练时随机采样，打破数据相关性。
3.  **目标网络（Target Network）**：计算 TD 目标时使用一个参数固定的网络 $Q_{\text{target}}$，切断了 Bootstrapping 中的正反馈循环，显著提升了稳定性。

**成果**：在 49 款 Atari 游戏中，29 款达到或超过人类专业玩家水平。

### 0.2.4 策略优化：PPO（Schulman et al., 2017）

虽然 DQN 在 Atari 上成功，但它只适用于离散动作空间。对于连续控制（如机器人），需要**策略梯度（Policy Gradient）**方法。

John Schulman 等人在 OpenAI 提出了 **PPO（Proximal Policy Optimization）**，成为工业界默认的首选算法。

**核心思想**：
*   直接优化策略 $\pi_\theta(a|s)$，而不是中间的价值函数。
*   使用"裁剪目标函数"限制策略更新幅度，防止更新过大导致性能崩溃：
    $$ L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right] $$
    其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是新旧策略的比率。

**优势**：简单、稳健、易于调参，适用于几乎所有任务。

### 0.2.5 LLM 对齐：RLHF（OpenAI, 2022）

**RLHF（Reinforcement Learning from Human Feedback）** 是 ChatGPT 成功的关键技术之一。

**流程**：
1.  **监督微调（SFT）**：在高质量对话数据上微调预训练模型。
2.  **奖励模型训练**：人类标注者对模型输出进行排序（A 比 B 好），训练奖励模型 $r_\phi(x, y)$。
3.  **RL 优化**：使用 PPO 优化语言模型，最大化奖励同时保持与初始模型的 KL 散度约束：
    $$ \max_{\pi_\theta} \mathbb{E}_{x, y \sim \pi_\theta} [r_\phi(x, y)] - \beta \, \text{KL}(\pi_\theta || \pi_{\text{ref}}) $$

**成果**：使 GPT-3、GPT-4、Claude、Gemini 等大模型的输出更符合人类偏好，减少有害内容。

---

## 0.3 核心概念预览

在深入学习 RL 算法之前，我们需要理解一些核心概念。这些概念将在后续章节中详细展开。

### 0.3.1 价值函数 vs 策略

RL 算法的核心是学习一个"好"的策略。实现方式有两种：

| 方法 | 学习目标 | 代表算法 | 优势 | 劣势 |
|:---|:---|:---|:---|:---|
| **Value-based** | 学习价值函数 $V(s)$ 或 $Q(s,a)$ | Q-learning, DQN | 样本效率高 | 仅适用于离散动作 |
| **Policy-based** | 直接学习策略 $\pi_\theta(a|s)$ | REINFORCE, PPO | 适用于连续动作、随机策略 | 高方差、样本效率低 |
| **Actor-Critic** | 同时学习价值函数和策略 | A3C, SAC | 结合两者优点 | 实现复杂 |

**形象类比**：
*   **Value-based**：你在玩一款策略游戏，通过大量对局记住了"在这个局面下赢的概率是 80%"这样的经验（价值），下棋时选择价值最高的走法。
*   **Policy-based**：你直接学习"看到这个局面就走这一步"的反射性策略，不需要显式地评估每一步的价值。

### 0.3.2 On-policy vs Off-policy

| 类型 | 定义 | 数据来源 | 代表算法 | 优势 | 劣势 |
|:---|:---|:---|:---|:---|
| **On-policy** | 用当前策略采集数据并更新该策略 | 必须是当前策略产生的新鲜数据 | SARSA, PPO | 稳定性好 | 样本效率低（数据用一次就扔） |
| **Off-policy** | 从任意策略（包括历史数据）中学习 | 可以是其他策略甚至人类演示 | Q-learning, SAC, DQN | 样本效率高、可复用数据 | 需要重要性采样修正、可能不稳定 |

**实际意义**：Off-policy 算法可以利用 **Experience Replay Buffer**（经验回放缓冲区），大幅提升样本效率。这在高成本环境（如真实机器人）中尤为重要。

### 0.3.3 Model-free vs Model-based

| 类型 | 是否学习环境模型 | 代表算法 | 优势 | 劣势 |
|:---|:---|:---|:---|:---|
| **Model-free** | 否，直接学习价值/策略 | Q-learning, PPO, SAC | 通用性强、实现简单 | 样本效率低 |
| **Model-based** | 是，学习 $P(s'|s,a)$ 和 $R(s,a)$ | Dyna, MBPO, Dreamer | 样本效率高、可规划 | 模型误差累积、实现复杂 |

**形象类比**：
*   **Model-free**：婴儿摸到火炉知道疼（直接经验），不需要理解"火的温度"和"皮肤损伤"的因果机制。
*   **Model-based**：物理学家建立热传导模型，可以预测"如果把手放在 500°C 的表面上 3 秒会怎样"，然后规划避免接触。

### 0.3.4 Sample efficiency vs Asymptotic performance

这是 RL 算法设计中的**核心权衡**：

*   **Sample efficiency（样本效率）**：达到特定性能所需的交互样本数。
    *   **高样本效率**：Model-based RL、Off-policy 算法（如 SAC）。
    *   **低样本效率**：On-policy 算法（如 PPO）、Policy Gradient。
*   **Asymptotic performance（渐近性能）**：在样本数趋于无穷时能达到的最优性能。

通常存在 **trade-off**：
*   **PPO**：样本效率较低，但渐近性能通常很好（最终能找到接近最优的策略）。
*   **SAC**：样本效率高（因为 off-policy），但在某些任务上最终性能可能不如 PPO。

---

## 0.4 环境准备

### 0.4.1 Gymnasium：现代 RL 环境标准

**Gymnasium** 是 OpenAI Gym 的继任者，由 Farama Foundation 维护，是目前 RL 社区的事实标准。

**安装**：
```bash
pip install gymnasium[all]
```

**核心接口**：
```python
import gymnasium as gym

# 创建环境
env = gym.make("CartPole-v1", render_mode="human")

# 重置环境，返回初始状态和信息字典
observation, info = env.reset(seed=42)

# 与环境交互
for _ in range(100):
    # 采样随机动作（从动作空间中均匀采样）
    action = env.action_space.sample()
    
    # 执行动作，返回五元组
    observation, reward, terminated, truncated, info = env.step(action)
    
    # terminated: 环境达到终止状态（如游戏结束）
    # truncated: 达到最大步数限制（超时）
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

**关键概念**：
*   **`observation_space`**：状态空间的规范。
    *   `Box`：连续空间，如 `Box(low=-1, high=1, shape=(4,))` 表示 4 维连续向量，每维范围 `[-1, 1]`。
    *   `Discrete(n)`：离散空间，如 `Discrete(4)` 表示 `{0, 1, 2, 3}` 四个选项。
*   **`action_space`**：动作空间的规范，格式同上。
*   **`reward_range`**：奖励的理论范围（用于归一化）。

### 0.4.2 常用环境介绍

**1. 经典控制（Classic Control）**
*   **CartPole-v1**：平衡倒立摆，最简单的入门环境。
    *   **Obs**: 4D（位置、速度、角度、角速度）
    *   **Action**: Discrete(2)（左推、右推）
    *   **Success**: 坚持 475+ 步
*   **MountainCar-v0**：推小车上山，奖励稀疏的经典难题。
*   **Pendulum-v1**：连续控制的钟摆平衡。

**2. Atari 游戏**
```bash
pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]
```
```python
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
```
*   **特点**：高维图像输入（210x160x3 RGB），需要预处理（灰度化、裁剪、缩放至 84x84）。

**3. MuJoCo 连续控制**
```bash
pip install gymnasium[mujoco]
```
```python
env = gym.make("HalfCheetah-v4")
```
*   **特点**：物理仿真环境，用于机器人步态控制、机械臂抓取等。

### 0.4.3 PyTorch 框架

我们将使用 **PyTorch** 作为深度学习框架（也可使用 JAX/TensorFlow，但 PyTorch 更直观）。

```bash
pip install torch torchvision
```

**基础代码模板**：
```python
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    """简单的 Q 网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.net(state)

# 初始化
state_dim = 4
action_dim = 2
q_net = QNetwork(state_dim, action_dim)
optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
```

### 0.4.4 第一个 RL 程序：Random Agent

让我们运行一个完全随机的 Agent，作为后续算法的 Baseline。

```python
"""
随机 Agent：在 CartPole 环境中采取随机动作
预期结果：平均约 20-30 步就会失败（远低于 475 步的成功线）
"""
import gymnasium as gym
import numpy as np

def run_random_agent(env_name="CartPole-v1", num_episodes=10):
    env = gym.make(env_name)
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # 随机选择动作
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode+1}: Reward = {total_reward}")
    
    env.close()
    print(f"\n平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    return episode_rewards

if __name__ == "__main__":
    rewards = run_random_agent(num_episodes=10)
```

**预期输出**：
```
Episode 1: Reward = 23.0
Episode 2: Reward = 31.0
Episode 3: Reward = 18.0
...
平均奖励: 24.30 ± 7.15
```

这个 24 分就是我们的 **Baseline**。在后续章节中，我们将学习各种算法，目标是达到 475+ 的"解决"标准。

---

## 本章总结

本章我们学习了：
1.  **RL 的本质**：通过与环境交互学习最优策略，核心是试错学习和延迟奖励。
2.  **历史脉络**：从 Bellman 的动态规划，到 DQN 的深度突破，再到 RLHF 驱动的 LLM 对齐。
3.  **核心概念**：Value vs Policy、On-policy vs Off-policy、Model-free vs Model-based。
4.  **实践环境**：Gymnasium、MuJoCo、PyTorch。

**下一章预告**：我们将深入学习强化学习的数学基础——**马尔可夫决策过程（MDP）**，形式化定义状态、动作、奖励和策略，并推导 Bellman 方程。

---

## 扩展阅读

*   Sutton & Barto, *Reinforcement Learning: An Introduction*, Chapter 1
*   OpenAI Spinning Up: [Introduction to RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
*   DeepMind x UCL RL Course: Lecture 1 (David Silver)
*   论文：Mnih et al., "Human-level control through deep reinforcement learning", *Nature* 2015

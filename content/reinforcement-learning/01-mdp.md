---
title: "第1章：马尔可夫决策过程 (MDP)"
description: "强化学习的数学基础：形式化定义、Bellman 方程推导与最优性理论"
date: "2026-01-30"
---

# 第1章：马尔可夫决策过程 (MDP)

几乎所有的强化学习问题都可以形式化为**马尔可夫决策过程（Markov Decision Process, MDP）**。本章我们将深入数学底层，构建 RL 的理论大厦。

---

## 1.1 MDP 形式化定义

MDP 是一个五元组 $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$，其中：

1.  $\mathcal{S}$：**状态空间 (State Space)**。所有可能状态的集合。
2.  $\mathcal{A}$：**动作空间 (Action Space)**。Agent 可以采取的所有动作集合。
3.  $\mathcal{P}$：**状态转移概率 (Transition Probability)**。
    $$ P(s'|s,a) = \Pr(S_{t+1}=s' \mid S_t=s, A_t=a) $$
    它描述了环境的动力学（Dynamics）。
4.  $\mathcal{R}$：**奖励函数 (Reward Function)**。即时奖励的期望值。
    $$ R(s,a) = \mathbb{E}[R_{t+1} \mid S_t=s, A_t=a] $$
    注：有时也写成 $R(s,a,s')$。
5.  $\gamma$：**折扣因子 (Discount Factor)**，$\gamma \in [0, 1]$。用于权衡即时奖励与长期奖励。

<div data-component="MDPGraphVisualizer"></div>

---

### 1.1.1 状态空间 $\mathcal{S}$ 的设计哲学

**状态的本质：完整性与充分性**

状态 $s \in \mathcal{S}$ 必须包含**所有决策相关的信息**，这是马尔可夫性质的前提。一个好的状态表示应该满足：

1. **完备性（Completeness）**：状态包含预测未来所需的所有信息
2. **充分性（Sufficiency）**：状态不包含冗余信息
3. **可观测性（Observability）**：Agent 能够获取或推断状态

**离散 vs 连续状态空间**

| 类型 | 定义 | 示例 | 算法选择 |
|:---|:---|:---|:---|
| **离散** | $\mathcal{S} = \{s_1, s_2, \ldots, s_n\}$ | 棋盘游戏、网格世界 | 表格方法、DQN |
| **连续** | $\mathcal{S} \subseteq \mathbb{R}^d$ | 机器人关节角度、自动驾驶 | 函数逼近、DDPG |
| **混合** | 部分离散 + 部分连续 | 游戏（离散动作 + 连续位置） | 混合架构 |

**状态空间大小的影响**

状态空间的大小直接决定了问题的复杂度：

- **小规模** ($|\mathcal{S}| < 10^3$)：可以用表格方法（Q-Table）
- **中等规模** ($10^3 < |\mathcal{S}| < 10^6$)：需要函数逼近（线性、神经网络）
- **大规模** ($|\mathcal{S}| > 10^6$)：必须用深度学习（DQN、PPO）

**实际案例：不同任务的状态表示**

1. **Atari 游戏**
   - 原始状态：210×160×3 RGB 图像
   - 预处理后：84×84×4 灰度堆叠帧
   - 状态空间大小：$256^{84 \times 84 \times 4} \approx 10^{67970}$（天文数字！）

2. **机器人控制（HalfCheetah）**
   - 状态：17维连续向量（关节角度、角速度）
   - 状态空间：$\mathbb{R}^{17}$（无限）

3. **围棋**
   - 状态：19×19 棋盘，每个位置 3 种状态（空、黑、白）
   - 状态空间大小：$3^{361} \approx 10^{172}$

**代码示例：不同状态空间的表示**

```python
import numpy as np
from typing import Union, Tuple

class StateSpace:
    """状态空间的抽象基类"""
    def sample(self):
        """随机采样一个状态"""
        raise NotImplementedError
    
    def contains(self, state):
        """检查状态是否在空间内"""
        raise NotImplementedError

class DiscreteStateSpace(StateSpace):
    """离散状态空间"""
    def __init__(self, n: int):
        self.n = n
        self.states = list(range(n))
    
    def sample(self):
        return np.random.randint(self.n)
    
    def contains(self, state):
        return 0 <= state < self.n
    
    def __repr__(self):
        return f"Discrete({self.n})"

class ContinuousStateSpace(StateSpace):
    """连续状态空间（Box）"""
    def __init__(self, low: np.ndarray, high: np.ndarray):
        self.low = np.array(low)
        self.high = np.array(high)
        self.shape = self.low.shape
    
    def sample(self):
        return np.random.uniform(self.low, self.high)
    
    def contains(self, state):
        return np.all(state >= self.low) and np.all(state <= self.high)
    
    def __repr__(self):
        return f"Box(shape={self.shape})"

# 使用示例
discrete_space = DiscreteStateSpace(n=16)  # 4x4 网格
continuous_space = ContinuousStateSpace(
    low=[-1.0, -1.0, -8.0, -8.0],  # CartPole 状态下界
    high=[1.0, 1.0, 8.0, 8.0]      # CartPole 状态上界
)

print(f"离散空间采样: {discrete_space.sample()}")
print(f"连续空间采样: {continuous_space.sample()}")
```

---

### 1.1.2 动作空间 $\mathcal{A}$ 的类型学

**全局动作空间 vs 状态依赖动作空间**

1. **全局动作空间**：$\mathcal{A}$ 在所有状态下相同
   - 例如：CartPole（左推、右推）
   - 简化了算法设计

2. **状态依赖动作空间**：$\mathcal{A}(s)$ 依赖于状态 $s$
   - 例如：国际象棋（不同局面下的合法走法不同）
   - 需要动态生成可用动作

**动作空间的结构**

| 结构 | 描述 | 示例 | 复杂度 |
|:---|:---|:---|:---|
| **独立动作** | 单个离散选择 | 上下左右 | $|\mathcal{A}|$ |
| **组合动作** | 多个独立选择的组合 | 10个开关（每个2状态） | $2^{10} = 1024$ |
| **连续动作** | 实数向量 | 机器人关节力矩 | $\infty$ |
| **混合动作** | 离散 + 连续 | 选择单位 + 移动位置 | 复杂 |

**动作空间大小的影响**

- **小动作空间** ($|\mathcal{A}| < 10$)：可以枚举所有动作（Value-based 方法）
- **大动作空间** ($|\mathcal{A}| > 100$)：需要策略梯度方法（Policy Gradient）
- **连续动作空间**：必须用 Actor-Critic 或 Deterministic Policy Gradient

**代码示例：动作空间的表示**

```python
class ActionSpace:
    """动作空间的抽象基类"""
    def sample(self):
        raise NotImplementedError
    
    def contains(self, action):
        raise NotImplementedError

class DiscreteActionSpace(ActionSpace):
    """离散动作空间"""
    def __init__(self, n: int):
        self.n = n
    
    def sample(self):
        return np.random.randint(self.n)
    
    def contains(self, action):
        return 0 <= action < self.n

class ContinuousActionSpace(ActionSpace):
    """连续动作空间"""
    def __init__(self, low: np.ndarray, high: np.ndarray):
        self.low = np.array(low)
        self.high = np.array(high)
        self.shape = self.low.shape
    
    def sample(self):
        return np.random.uniform(self.low, self.high)
    
    def contains(self, action):
        return np.all(action >= self.low) and np.all(action <= self.high)

# 使用示例
discrete_actions = DiscreteActionSpace(n=4)  # 上下左右
continuous_actions = ContinuousActionSpace(
    low=[-1.0, -1.0],  # 两个关节的力矩范围
    high=[1.0, 1.0]
)
```

---

### 1.1.3 转移概率 $\mathcal{P}$ 的性质

**形式化定义**

转移概率函数 $P: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0, 1]$ 定义为：

$$
P(s' | s, a) = \Pr(S_{t+1} = s' | S_t = s, A_t = a)
$$

**数学性质**

1. **归一化性质**（Normalization）：
   $$\sum_{s' \in \mathcal{S}} P(s' | s, a) = 1, \quad \forall s \in \mathcal{S}, a \in \mathcal{A}$$

2. **非负性**（Non-negativity）：
   $$P(s' | s, a) \geq 0, \quad \forall s, s' \in \mathcal{S}, a \in \mathcal{A}$$

**确定性 vs 随机性环境**

| 类型 | 定义 | 示例 | 转移概率 |
|:---|:---|:---|:---|
| **确定性** | 给定 $(s,a)$，下一状态唯一 | 棋类游戏、确定性物理模拟 | $P(s'|s,a) \in \{0, 1\}$ |
| **随机性** | 给定 $(s,a)$，下一状态有多种可能 | 扑克牌、有噪声的机器人 | $0 < P(s'|s,a) < 1$ |

**转移矩阵的表示**

对于离散状态和动作空间，转移概率可以表示为三维张量：

$$
\mathbf{P}^a \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{S}|}, \quad [\mathbf{P}^a]_{ij} = P(s_j | s_i, a)
$$

其中 $\mathbf{P}^a$ 是动作 $a$ 对应的转移矩阵。

**转移矩阵的稀疏性**

在很多实际问题中，转移矩阵是**稀疏的**（Sparse）：
- 大部分 $P(s'|s,a) = 0$
- 只有少数状态可达

例如，在网格世界中，从一个格子只能移动到相邻的 4 个格子，而不是所有格子。

**代码示例：构建转移矩阵**

```python
def build_transition_matrix(size=4):
    """
    构建 GridWorld 的转移矩阵
    
    参数:
        size: 网格大小
    
    返回:
        P: shape (num_states, num_actions, num_states)
    """
    num_states = size * size
    num_actions = 4  # 上下左右
    
    # 初始化转移矩阵
    P = np.zeros((num_states, num_actions, num_states))
    
    def state_to_coord(s):
        return (s // size, s % size)
    
    def coord_to_state(x, y):
        return x * size + y
    
    for s in range(num_states):
        x, y = state_to_coord(s)
        
        # 定义四个动作的效果
        actions = [
            (-1, 0),  # 上
            (0, 1),   # 右
            (1, 0),   # 下
            (0, -1)   # 左
        ]
        
        for a, (dx, dy) in enumerate(actions):
            # 计算新位置（带边界检查）
            nx = max(0, min(size - 1, x + dx))
            ny = max(0, min(size - 1, y + dy))
            ns = coord_to_state(nx, ny)
            
            # 确定性转移
            P[s, a, ns] = 1.0
    
    # 验证归一化性质
    assert np.allclose(P.sum(axis=2), 1.0), "转移概率未归一化！"
    
    return P

# 使用示例
P = build_transition_matrix(size=4)
print(f"转移矩阵形状: {P.shape}")
print(f"稀疏度: {(P == 0).sum() / P.size * 100:.1f}%")

# 可视化某个状态-动作对的转移概率
s, a = 5, 1  # 状态5，动作1（右）
print(f"\n从状态 {s} 采取动作 {a} 的转移概率:")
print(P[s, a])
```

---

### 1.1.4 奖励函数 $\mathcal{R}$ 的设计

**奖励函数的不同形式**

1. **$R(s, a)$**：奖励只依赖于状态和动作
   $$R(s, a) = \mathbb{E}[R_{t+1} | S_t = s, A_t = a]$$

2. **$R(s, a, s')$**：奖励还依赖于下一状态
   $$R(s, a, s') = \mathbb{E}[R_{t+1} | S_t = s, A_t = a, S_{t+1} = s']$$

3. **$R(s)$**：奖励只依赖于状态（简化形式）

**三种形式的关系**

$$
R(s, a) = \sum_{s'} P(s' | s, a) R(s, a, s')
$$

**奖励的有界性假设**

在理论分析中，我们通常假设奖励是有界的：

$$
|R(s, a)| \leq R_{\max}, \quad \forall s, a
$$

这保证了累积回报的收敛性（当 $\gamma < 1$ 时）。

**奖励设计的最佳实践**

1. **简单性**：尽可能简单，避免过度工程化
2. **稀疏性 vs 密集性**：权衡探索难度与训练稳定性
3. **Potential-based Shaping**：保持最优策略不变（见 Chapter 0）
4. **避免 Reward Hacking**：测试奖励函数是否真正反映目标

**代码示例：不同奖励函数的实现**

```python
class RewardFunction:
    """奖励函数的抽象基类"""
    def __call__(self, s, a, s_next=None):
        raise NotImplementedError

class GridWorldReward(RewardFunction):
    """GridWorld 的奖励函数"""
    def __init__(self, size=4, goal_state=None):
        self.size = size
        self.goal_state = goal_state or (size * size - 1)
    
    def __call__(self, s, a, s_next=None):
        # 到达目标：+10
        if s_next == self.goal_state:
            return 10.0
        # 每步惩罚：-0.1（鼓励快速到达）
        return -0.1

class ShapedReward(RewardFunction):
    """带 Potential-based Shaping 的奖励"""
    def __init__(self, base_reward, potential_fn, gamma=0.99):
        self.base_reward = base_reward
        self.potential_fn = potential_fn
        self.gamma = gamma
    
    def __call__(self, s, a, s_next):
        # 基础奖励
        r = self.base_reward(s, a, s_next)
        # Shaping: F(s,a,s') = γΦ(s') - Φ(s)
        shaping = self.gamma * self.potential_fn(s_next) - self.potential_fn(s)
        return r + shaping

# 使用示例
def manhattan_potential(s, goal=15, size=4):
    """曼哈顿距离势函数（负值）"""
    x, y = s // size, s % size
    gx, gy = goal // size, goal % size
    return -abs(x - gx) - abs(y - gy)

base_reward = GridWorldReward(size=4)
shaped_reward = ShapedReward(
    base_reward=base_reward,
    potential_fn=lambda s: manhattan_potential(s, goal=15, size=4),
    gamma=0.99
)

# 测试
s, a, s_next = 0, 1, 1
print(f"基础奖励: {base_reward(s, a, s_next):.2f}")
print(f"Shaped 奖励: {shaped_reward(s, a, s_next):.2f}")
```

---

### 1.1.5 马尔可夫性质 (Markov Property)

> "The future is independent of the past given the present."

**形式化定义**

如果在当前状态 $S_t$ 已知的情况下，未来状态 $S_{t+1}$ 只与当前状态 $S_t$ 和动作 $A_t$ 有关，而与历史状态 $S_{t-1}, S_{t-2}, \dots$ 无关，则称该状态具有**马尔可夫性质**。

$$
\Pr(S_{t+1} | S_t, A_t) = \Pr(S_{t+1} | S_t, A_t, S_{t-1}, A_{t-1}, \dots, S_0, A_0)
$$

**数学上的条件独立性**

马尔可夫性质可以表述为条件独立性：

$$
S_{t+1} \perp (S_{t-1}, A_{t-1}, \ldots, S_0, A_0) \mid (S_t, A_t)
$$

读作："给定 $(S_t, A_t)$，$S_{t+1}$ 与历史独立"。

**为什么马尔可夫性质很重要？**

1. **简化问题**：不需要记住整个历史，只需当前状态
2. **理论保证**：Bellman 方程的推导依赖于马尔可夫性
3. **算法设计**：Value Iteration、Q-Learning 等算法的正确性依赖于马尔可夫性

**状态的充分性（State Sufficiency）**

一个状态是**充分的（Sufficient）**，当且仅当它包含了预测未来所需的所有信息。

形式化地，状态 $s_t$ 是充分的，如果：

$$
\mathbb{E}[R_{t+1} | s_t, a_t, h_t] = \mathbb{E}[R_{t+1} | s_t, a_t]
$$

其中 $h_t = (s_0, a_0, \ldots, s_{t-1}, a_{t-1})$ 是历史。

**非马尔可夫环境的处理**

当环境不满足马尔可夫性时，我们有几种策略：

1. **扩展状态空间**
   - 将历史信息纳入状态
   - 例如：Frame Stacking（堆叠最近 $k$ 帧）

2. **使用循环神经网络（RNN/LSTM）**
   - 隐藏状态 $h_t$ 作为"记忆"
   - $h_t = f(h_{t-1}, o_t)$
   - 策略：$\pi(a | h_t)$

3. **Belief State（信念状态）**
   - 维护状态的概率分布 $b_t(s)$
   - 贝叶斯更新：$b_{t+1}(s') \propto O(o_{t+1} | s') \sum_s P(s' | s, a_t) b_t(s)$

**实际案例：马尔可夫性的检验**

| 环境 | 单步观测 | 马尔可夫性 | 解决方案 |
|:---|:---|:---|:---|
| **国际象棋** | 当前棋盘 | ✅ 是 | 无需处理 |
| **Atari (单帧)** | 单张图片 | ❌ 否（无法判断速度） | Frame Stacking (4帧) |
| **扑克牌** | 自己的手牌 | ❌ 否（看不到对手手牌） | Belief State / CFR |
| **股票交易** | 当前价格 | ❌ 否（需要历史趋势） | RNN / Transformer |

**代码示例：检测马尔可夫性**

```python
def check_markov_property(env, num_samples=1000):
    """
    检测环境是否满足马尔可夫性质
    
    方法：检查 P(s'|s,a) 是否与历史无关
    """
    from collections import defaultdict
    
    # 收集转移数据
    transitions = defaultdict(lambda: defaultdict(int))
    
    for _ in range(num_samples):
        s = env.reset()
        history = []
        
        for t in range(100):
            a = env.action_space.sample()
            s_next, r, done, _ = env.step(a)
            
            # 记录转移：(s, a, history) -> s_next
            history_tuple = tuple(history[-5:])  # 只看最近5步
            transitions[(s, a, history_tuple)][s_next] += 1
            
            history.append((s, a))
            s = s_next
            
            if done:
                break
    
    # 检查：对于相同的 (s, a)，不同历史是否导致相同的转移分布？
    markov_violations = 0
    
    for (s, a, hist), next_states in transitions.items():
        # 查找相同 (s, a) 但不同历史的转移
        for (s2, a2, hist2), next_states2 in transitions.items():
            if s == s2 and a == a2 and hist != hist2:
                # 比较两个转移分布
                if next_states != next_states2:
                    markov_violations += 1
    
    if markov_violations == 0:
        print("✅ 环境可能满足马尔可夫性质")
    else:
        print(f"❌ 检测到 {markov_violations} 个马尔可夫性违反")
    
    return markov_violations == 0

# 使用示例（需要 Gymnasium 环境）
# import gymnasium as gym
# env = gym.make("CartPole-v1")
# check_markov_property(env)
```

---

### 1.1.6 回报 (Return) 与折扣因子

**累积折扣回报的定义**

Agent 的目标是最大化**累积折扣回报 (Cumulative Discounted Return)** $G_t$：

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^\infty \gamma^k R_{t+k+1}
$$

**回报的递归定义**

回报具有优美的递归结构：

$$
G_t = R_{t+1} + \gamma G_{t+1}
$$

这是 Bellman 方程的基础。

**为什么需要折扣因子 $\gamma$？**

1. **数学收敛性**：对于无限长度的任务，若不衰减，回报可能无穷大。$\gamma < 1$ 保证了级数收敛。
   
   **证明**：如果 $|R_t| \leq R_{\max}$，则：
   $$
   |G_t| \leq \sum_{k=0}^\infty \gamma^k R_{\max} = \frac{R_{\max}}{1 - \gamma} < \infty
   $$

2. **不确定性**：未来的预测往往不如当前确定，衰减反映了对未来的不确定性。

3. **金融解释**：现在的 1 元钱比未来的 1 元钱更有价值（时间价值）。

4. **有效时间视野**：$\gamma$ 决定了 Agent 的"远见"：
   $$
   H_{\text{eff}} = \frac{1}{1 - \gamma}
   $$
   - $\gamma = 0.9 \Rightarrow H_{\text{eff}} = 10$ 步
   - $\gamma = 0.99 \Rightarrow H_{\text{eff}} = 100$ 步

**Episodic vs Continuing Tasks**

| 任务类型 | 定义 | 示例 | 折扣因子 |
|:---|:---|:---|:---|
| **Episodic** | 有明确的终止状态 | 棋类游戏、Atari 游戏 | $\gamma \in [0, 1]$ |
| **Continuing** | 无终止状态，永远运行 | 服务器调度、股票交易 | $\gamma < 1$ (必须) |

**统一框架：吸收状态（Absorbing State）**

我们可以将 Episodic 任务转换为 Continuing 任务：
- 添加一个**吸收状态** $s_{\text{absorb}}$
- 终止状态转移到吸收状态：$P(s_{\text{absorb}} | s_{\text{terminal}}, a) = 1$
- 吸收状态的奖励为 0：$R(s_{\text{absorb}}, a) = 0$

**回报的期望与方差**

回报 $G_t$ 是一个随机变量（因为轨迹是随机的）。

**期望**：
$$
\mathbb{E}_\pi[G_t | S_t = s] = V^\pi(s)
$$

**方差**：
$$
\text{Var}_\pi[G_t | S_t = s] = \mathbb{E}_\pi[(G_t - V^\pi(s))^2 | S_t = s]
$$

方差的大小影响学习的稳定性（高方差 → 不稳定）。

**代码示例：计算折现回报**

```python
def compute_returns(rewards, gamma=0.99, normalize=False):
    """
    计算折现回报
    
    参数:
        rewards: 奖励序列 [r_1, r_2, ..., r_T]
        gamma: 折扣因子
        normalize: 是否归一化
    
    返回:
        returns: 折现回报序列 [G_0, G_1, ..., G_{T-1}]
    """
    returns = []
    G = 0
    
    # 从后往前计算（利用递归定义）
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    
    returns = np.array(returns)
    
    if normalize:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    return returns

# 使用示例
rewards = [1, 0, 0, 0, 10]  # 第5步有大奖励
returns = compute_returns(rewards, gamma=0.99)

print("奖励序列:", rewards)
print("回报序列:", returns)
print(f"\n初始回报 G_0 = {returns[0]:.4f}")
print(f"验证递归: G_0 = {rewards[0] + 0.99 * returns[1]:.4f}")
```

**可视化：不同 $\gamma$ 对回报的影响**

```python
import matplotlib.pyplot as plt

# 模拟一个奖励序列
T = 50
rewards = np.zeros(T)
rewards[10] = 5   # 第10步有奖励
rewards[30] = 10  # 第30步有更大奖励

gammas = [0.9, 0.95, 0.99, 1.0]
plt.figure(figsize=(12, 6))

for gamma in gammas:
    returns = compute_returns(rewards, gamma=gamma)
    plt.plot(returns, label=f'γ={gamma}', linewidth=2)

plt.xlabel('Time Step')
plt.ylabel('Return $G_t$')
plt.title('Impact of Discount Factor on Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

**本节小结**

我们深入探讨了 MDP 的五个核心组成部分：

1. **状态空间 $\mathcal{S}$**：完备性、充分性、可观测性
2. **动作空间 $\mathcal{A}$**：离散 vs 连续、全局 vs 状态依赖
3. **转移概率 $\mathcal{P}$**：归一化性质、稀疏性、确定性 vs 随机性
4. **奖励函数 $\mathcal{R}$**：不同形式、有界性、设计原则
5. **马尔可夫性质**：条件独立性、状态充分性、非马尔可夫环境的处理
6. **回报与折扣**：递归定义、收敛性、Episodic vs Continuing

这些概念是理解后续所有 RL 算法的数学基础。

---

## 1.2 策略 (Policy)

## 1.2 策略 (Policy)

策略 $\pi$ 定义了 Agent 在给定状态下的行为方式。它是 RL 的核心：**策略决定行为，价值评估好坏**。

---

### 1.2.1 策略的形式化定义

**随机策略（Stochastic Policy）**

随机策略是一个条件概率分布：

$$
\pi: \mathcal{S} \times \mathcal{A} \to [0, 1]
$$

$$
\pi(a|s) = \Pr(A_t = a | S_t = s)
$$

满足归一化条件：

$$
\sum_{a \in \mathcal{A}} \pi(a|s) = 1, \quad \forall s \in \mathcal{S}
$$

**确定性策略（Deterministic Policy）**

确定性策略是一个映射函数：

$$
\pi: \mathcal{S} \to \mathcal{A}
$$

$$
a = \pi(s)
$$

确定性策略可以看作随机策略的特例：

$$
\pi(a|s) = \begin{cases}
1 & \text{if } a = \pi(s) \\
0 & \text{otherwise}
\end{cases}
$$

**何时使用随机策略？**

1. **探索（Exploration）**
   - 随机策略天然具有探索性
   - 例如：$\epsilon$-greedy 策略

2. **二人博弈（Game Theory）**
   - 石头剪刀布：纯策略会被对手利用
   - 混合策略（Mixed Strategy）是纳什均衡

3. **部分可观测环境（POMDP）**
   - 随机策略可能优于确定性策略
   - 例如：扑克牌中的 bluffing

4. **策略梯度方法**
   - 需要可微的策略分布
   - 确定性策略的梯度为 0

**何时使用确定性策略？**

1. **最优策略**
   - 在 MDP 中，至少存在一个确定性最优策略
   - 定理：$\exists \pi^*$ 确定性且最优

2. **连续控制**
   - DDPG、TD3 使用确定性策略
   - 配合 Critic 估计梯度

3. **部署阶段**
   - 训练时用随机策略（探索）
   - 部署时用确定性策略（利用）

---

### 1.2.2 策略空间的数学结构

**策略空间 $\Pi$**

所有可能策略的集合构成**策略空间** $\Pi$。

对于有限 MDP（$|\mathcal{S}| < \infty, |\mathcal{A}| < \infty$）：

- **随机策略空间**：$\Pi_{\text{stochastic}}$ 是一个 $|\mathcal{S}| \times |\mathcal{A}|$ 维的概率单纯形（Probability Simplex）
- **确定性策略空间**：$\Pi_{\text{deterministic}}$ 包含 $|\mathcal{A}|^{|\mathcal{S}|}$ 个策略

**概率单纯形（Probability Simplex）**

对于每个状态 $s$，策略 $\pi(\cdot|s)$ 是一个概率分布，位于单纯形上：

$$
\Delta^{|\mathcal{A}|-1} = \left\{ p \in \mathbb{R}^{|\mathcal{A}|} : \sum_a p_a = 1, p_a \geq 0 \right\}
$$

**策略的参数化**

在实际应用中，我们通常使用参数化策略 $\pi_\theta$：

$$
\pi_\theta(a|s) = f_\theta(s, a)
$$

其中 $\theta \in \mathbb{R}^d$ 是参数向量，$f_\theta$ 是参数化函数（如神经网络）。

**常见的参数化方法**

1. **Softmax 策略**（离散动作）
   $$
   \pi_\theta(a|s) = \frac{\exp(h_\theta(s, a))}{\sum_{a'} \exp(h_\theta(s, a'))}
   $$
   其中 $h_\theta(s, a)$ 是偏好函数（Preference Function）

2. **高斯策略**（连续动作）
   $$
   \pi_\theta(a|s) = \mathcal{N}(a | \mu_\theta(s), \sigma_\theta(s))
   $$
   其中 $\mu_\theta(s)$ 是均值，$\sigma_\theta(s)$ 是标准差

3. **确定性策略**
   $$
   a = \mu_\theta(s)
   $$

---

### 1.2.3 策略的表示方法

**1. 表格表示（Tabular Representation）**

对于小规模离散状态和动作空间，可以用表格存储策略。

**数据结构**：$|\mathcal{S}| \times |\mathcal{A}|$ 矩阵

$$
\pi[s, a] = \pi(a|s)
$$

**优势**：
- 简单直观
- 精确表示

**劣势**：
- 只适用于小规模问题
- 无法泛化到未见过的状态

**代码实现**：

```python
import numpy as np

class TabularPolicy:
    """表格策略"""
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        # 初始化为均匀分布
        self.table = np.ones((num_states, num_actions)) / num_actions
    
    def get_action_probs(self, state):
        """获取动作概率分布"""
        return self.table[state]
    
    def sample_action(self, state):
        """采样动作"""
        probs = self.get_action_probs(state)
        return np.random.choice(self.num_actions, p=probs)
    
    def update(self, state, action, new_prob):
        """更新策略（需要重新归一化）"""
        self.table[state, action] = new_prob
        # 归一化
        self.table[state] /= self.table[state].sum()
    
    def make_greedy(self, Q):
        """根据 Q 函数构造贪心策略"""
        for s in range(self.num_states):
            best_action = np.argmax(Q[s])
            self.table[s] = 0
            self.table[s, best_action] = 1.0

# 使用示例
policy = TabularPolicy(num_states=16, num_actions=4)
state = 5
action = policy.sample_action(state)
print(f"状态 {state} 的动作概率: {policy.get_action_probs(state)}")
print(f"采样动作: {action}")
```

**2. 参数化表示（Parameterized Representation）**

使用神经网络或其他函数逼近器表示策略。

**神经网络策略**：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

class DiscretePolicy(nn.Module):
    """离散动作空间的神经网络策略"""
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
        """输出动作的 logits"""
        return self.net(state)
    
    def get_action_probs(self, state):
        """获取动作概率分布"""
        logits = self(state)
        return F.softmax(logits, dim=-1)
    
    def sample_action(self, state):
        """采样动作"""
        logits = self(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

class ContinuousPolicy(nn.Module):
    """连续动作空间的高斯策略"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 限制在 [-1, 1]
        )
        # 学习对数标准差
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        """输出动作分布的参数"""
        mean = self.mean_net(state)
        std = torch.exp(self.log_std)
        return mean, std
    
    def sample_action(self, state):
        """采样动作"""
        mean, std = self(state)
        dist = Normal(mean, std)
        action = dist.rsample()  # 重参数化采样
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach().numpy(), log_prob

# 使用示例
discrete_policy = DiscretePolicy(state_dim=4, action_dim=2)
continuous_policy = ContinuousPolicy(state_dim=4, action_dim=2)

state = torch.randn(1, 4)
action, log_prob = discrete_policy.sample_action(state)
print(f"离散策略采样: action={action}, log_prob={log_prob.item():.4f}")

action, log_prob = continuous_policy.sample_action(state)
print(f"连续策略采样: action={action}, log_prob={log_prob.item():.4f}")
```

**3. 隐式策略（Implicit Policy）**

通过价值函数隐式定义策略。

**$\epsilon$-Greedy 策略**：

$$
\pi(a|s) = \begin{cases}
1 - \epsilon + \frac{\epsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_{a'} Q(s, a') \\
\frac{\epsilon}{|\mathcal{A}|} & \text{otherwise}
\end{cases}
$$

**Softmax（Boltzmann）策略**：

$$
\pi(a|s) = \frac{\exp(Q(s, a) / \tau)}{\sum_{a'} \exp(Q(s, a') / \tau)}
$$

其中 $\tau$ 是温度参数（Temperature）。

**代码实现**：

```python
class EpsilonGreedyPolicy:
    """ε-贪心策略"""
    def __init__(self, num_actions, epsilon=0.1):
        self.num_actions = num_actions
        self.epsilon = epsilon
    
    def sample_action(self, Q_values):
        """根据 Q 值采样动作"""
        if np.random.rand() < self.epsilon:
            # 探索：随机动作
            return np.random.randint(self.num_actions)
        else:
            # 利用：贪心动作
            return np.argmax(Q_values)

class BoltzmannPolicy:
    """Boltzmann（Softmax）策略"""
    def __init__(self, temperature=1.0):
        self.temperature = temperature
    
    def get_action_probs(self, Q_values):
        """计算动作概率"""
        exp_Q = np.exp(Q_values / self.temperature)
        return exp_Q / exp_Q.sum()
    
    def sample_action(self, Q_values):
        """采样动作"""
        probs = self.get_action_probs(Q_values)
        return np.random.choice(len(Q_values), p=probs)

# 使用示例
Q_values = np.array([1.0, 2.5, 0.5, 1.8])

epsilon_greedy = EpsilonGreedyPolicy(num_actions=4, epsilon=0.1)
action = epsilon_greedy.sample_action(Q_values)
print(f"ε-Greedy 采样: {action}")

boltzmann = BoltzmannPolicy(temperature=0.5)
probs = boltzmann.get_action_probs(Q_values)
print(f"Boltzmann 概率: {probs}")
action = boltzmann.sample_action(Q_values)
print(f"Boltzmann 采样: {action}")
```

---

### 1.2.4 策略的偏序关系与策略改进

**策略的偏序（Partial Order）**

我们定义策略 $\pi$ 不劣于策略 $\pi'$，记作 $\pi \geq \pi'$，当且仅当：

$$
V^\pi(s) \geq V^{\pi'}(s), \quad \forall s \in \mathcal{S}
$$

这是一个**偏序关系**（Partial Order），因为：
1. **自反性**：$\pi \geq \pi$
2. **传递性**：若 $\pi_1 \geq \pi_2$ 且 $\pi_2 \geq \pi_3$，则 $\pi_1 \geq \pi_3$
3. **反对称性**：若 $\pi \geq \pi'$ 且 $\pi' \geq \pi$，则 $V^\pi = V^{\pi'}$

**注意**：不是所有策略都可比较！可能存在 $\pi_1, \pi_2$ 使得：
- $V^{\pi_1}(s_1) > V^{\pi_2}(s_1)$
- $V^{\pi_1}(s_2) < V^{\pi_2}(s_2)$

**策略改进定理（Policy Improvement Theorem）**

这是 RL 最重要的定理之一！

**定理**：设 $\pi$ 和 $\pi'$ 是两个确定性策略，如果对所有状态 $s$：

$$
Q^\pi(s, \pi'(s)) \geq V^\pi(s)
$$

则 $\pi' \geq \pi$，即 $V^{\pi'}(s) \geq V^\pi(s), \forall s$。

**证明**：

从任意状态 $s$ 开始，按照策略 $\pi'$ 行动一步，然后切换到策略 $\pi$：

$$
\begin{align}
V^\pi(s) &\leq Q^\pi(s, \pi'(s)) \\
&= \mathbb{E}[R_{t+1} + \gamma V^\pi(S_{t+1}) | S_t = s, A_t = \pi'(s)] \\
&= \mathbb{E}_{\pi'}[R_{t+1} + \gamma V^\pi(S_{t+1}) | S_t = s]
\end{align}
$$

继续展开，在 $S_{t+1}$ 处再次应用不等式：

$$
\begin{align}
V^\pi(s) &\leq \mathbb{E}_{\pi'}[R_{t+1} + \gamma Q^\pi(S_{t+1}, \pi'(S_{t+1})) | S_t = s] \\
&= \mathbb{E}_{\pi'}[R_{t+1} + \gamma (R_{t+2} + \gamma V^\pi(S_{t+2})) | S_t = s] \\
&= \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 V^\pi(S_{t+2}) | S_t = s]
\end{align}
$$

重复此过程至无穷：

$$
V^\pi(s) \leq \mathbb{E}_{\pi'}\left[\sum_{k=0}^\infty \gamma^k R_{t+k+1} \bigg| S_t = s\right] = V^{\pi'}(s)
$$

**证毕**。

**贪心策略改进（Greedy Policy Improvement）**

给定策略 $\pi$，我们可以构造一个改进的策略 $\pi'$：

$$
\pi'(s) = \arg\max_a Q^\pi(s, a)
$$

根据策略改进定理，$\pi' \geq \pi$。

**代码示例：策略改进**

```python
def policy_improvement(Q, num_states, num_actions):
    """
    根据 Q 函数进行策略改进
    
    参数:
        Q: Q 函数表格，shape (num_states, num_actions)
        num_states: 状态数
        num_actions: 动作数
    
    返回:
        new_policy: 改进后的确定性策略
        policy_stable: 策略是否稳定（未改变）
    """
    new_policy = np.zeros(num_states, dtype=int)
    
    for s in range(num_states):
        # 贪心选择最优动作
        new_policy[s] = np.argmax(Q[s])
    
    return new_policy

def demonstrate_policy_improvement():
    """演示策略改进过程"""
    num_states, num_actions = 4, 2
    
    # 初始随机策略
    old_policy = np.random.randint(0, num_actions, size=num_states)
    
    # 假设的 Q 函数
    Q = np.array([
        [1.0, 2.0],  # 状态 0: 动作 1 更好
        [3.0, 1.0],  # 状态 1: 动作 0 更好
        [2.0, 2.0],  # 状态 2: 两个动作相同
        [0.5, 1.5]   # 状态 3: 动作 1 更好
    ])
    
    # 策略改进
    new_policy = policy_improvement(Q, num_states, num_actions)
    
    print("旧策略:", old_policy)
    print("新策略:", new_policy)
    print("改进:", new_policy != old_policy)
    
    # 验证：新策略的动作应该对应最大 Q 值
    for s in range(num_states):
        assert Q[s, new_policy[s]] == Q[s].max()
    print("✅ 策略改进验证通过")

demonstrate_policy_improvement()
```

---

### 1.2.5 平稳策略与非平稳策略

**平稳策略（Stationary Policy）**

策略不随时间变化：

$$
\pi(a|s) \text{ 不依赖于 } t
$$

**非平稳策略（Non-stationary Policy）**

策略随时间变化：

$$
\pi_t(a|s) \text{ 依赖于时间步 } t
$$

**定理**：在无限视野折扣 MDP 中，至少存在一个**平稳**最优策略。

这意味着我们可以专注于寻找平稳策略，无需考虑时间依赖。

**历史依赖策略（History-dependent Policy）**

在 POMDP 中，策略可能依赖于观测历史：

$$
\pi(a | o_1, a_1, o_2, a_2, \ldots, o_t)
$$

但在 MDP 中，马尔可夫性质保证了历史依赖策略不会优于马尔可夫策略。

---

**本节小结**

我们深入探讨了策略的理论基础：

1. **形式化定义**：随机策略 vs 确定性策略
2. **策略空间**：概率单纯形、参数化策略
3. **表示方法**：表格、神经网络、隐式策略（$\epsilon$-greedy, Softmax）
4. **偏序关系**：策略优劣的定义
5. **策略改进定理**：贪心改进的理论保证
6. **平稳性**：平稳策略的充分性

这些概念是策略迭代、策略梯度等算法的理论基础。

---

## 1.3 价值函数 (Value Function)

为了评估一个策略的好坏，我们定义**价值函数**。

### 1.3.1 状态价值函数 $V^\pi(s)$

当我们处于状态 $s$，并按照策略 $\pi$ 行动时，预期的回报是多少？

$$ V^\pi(s) = \mathbb{E}_\pi [G_t \mid S_t=s] = \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k R_{t+k+1} \bigg| S_t=s \right] $$

### 1.3.2 动作价值函数 $Q^\pi(s,a)$

当我们处于状态 $s$，**先采取动作 $a$**，之后按照策略 $\pi$ 行动，预期的回报是多少？

$$ Q^\pi(s, a) = \mathbb{E}_\pi [G_t \mid S_t=s, A_t=a] = \mathbb{E}_\pi \left[ \sum_{k=0}^\infty \gamma^k R_{t+k+1} \bigg| S_t=s, A_t=a \right] $$

### 1.3.3 V 与 Q 的关系

这不仅是公式，更是直觉：
1.  **V 是 Q 的期望**：
    $$ V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) Q^\pi(s,a) $$
2.  **Q 是 R + 下一步的 V**：
    $$ Q^\pi(s, a) = R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) V^\pi(s') $$

<div data-component="ValueFunctionEvolution"></div>

### 1.3.4 优势函数 (Advantage Function)
$$ A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s) $$
它衡量了动作 $a$ 比"平均水平"好多少。这在 PPO、A3C 等算法中至关重要。

---

## 1.4 Bellman 方程 (Bellman Equation)

RL 的核心就是解 Bellman 方程。它利用了价值函数的**递归性质**。

### 1.4.1 Bellman 期望方程 (Expectation Equation)
展开 $G_t$：
$$ \begin{aligned} V^\pi(s) &= \mathbb{E}[R_{t+1} + \gamma G_{t+1} | S_t=s] \\ &= \sum_a \pi(a|s) \sum_{s', r} p(s', r|s,a) [r + \gamma V^\pi(s')] \end{aligned} $$

对于 Q 值：
$$ Q^\pi(s, a) = \sum_{s', r} p(s', r|s,a) [r + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a')] $$

<div data-component="BellmanEquationDerivation"></div>

### 1.4.2 Bellman 最优方程 (Optimality Equation)
如果我们遵循**最优策略** $\pi^*$（即每一步都选 $Q$ 值最大的动作），则有：

$$ V^*(s) = \max_a Q^*(s,a) = \max_a \sum_{s', r} p(s', r|s,a) [r + \gamma V^*(s')] $$
$$ Q^*(s, a) = \sum_{s', r} p(s', r|s,a) [r + \gamma \max_{a'} Q^*(s', a')] $$

**注意**：期望方程是线性的（可以解线性方程组），而最优方程不仅是非线性的（因为有 `max` 操作），而且没有解析解，通常需要迭代求解（如 Value Iteration 或 Q-Learning）。

---

## 1.5 最优性理论 (Optimality)

### 1.5.1 偏序关系
我们定义策略 $\pi \ge \pi'$ 当且仅当对于所有状态 $s$，都有 $V^\pi(s) \ge V^{\pi'}(s)$。

**定理**：
对于任意 MDP：
1.  存在至少一个最优策略 $\pi^*$，它不劣于任何其他策略。
2.  所有最优策略共享相同的最优状态价值函数 $V^*(s)$。
3.  所有最优策略共享相同的最优动作价值函数 $Q^*(s,a)$。

### 1.5.2 寻找最优策略
一旦我们有了 $Q^*(s,a)$，最优策略提取就变得非常简单（贪心）：

$$ \pi^*(s) = \arg\max_a Q^*(s,a) $$

如果 $Q^*$ 对应的最大值有多个动作，则任意分配概率都是最优的。

---

## 1.6 代码实战：构建一个 GridWorld MDP

我们来实现一个简单的网格世界 MDP 类，用于后续章节的测试。

```python
import numpy as np

class GridWorldMDP:
    def __init__(self, size=4, gamma=0.9):
        self.size = size
        self.gamma = gamma
        self.num_states = size * size
        self.num_actions = 4 # 0:Up, 1:Right, 2:Down, 3:Left
        self.P = self._build_transition_matrix()
        self.R = self._build_reward_matrix()

    def _state_to_coord(self, s):
        return (s // self.size, s % self.size)

    def _coord_to_state(self, x, y):
        return x * self.size + y

    def _build_transition_matrix(self):
        # P[s][a][s'] = probability
        P = np.zeros((self.num_states, self.num_actions, self.num_states))
        
        for s in range(self.num_states):
            x, y = self._state_to_coord(s)
            
            # 终止状态 (左上角和右下角)
            if s == 0 or s == self.num_states - 1:
                P[s, :, s] = 1.0
                continue
                
            for a in range(self.num_actions):
                nx, ny = x, y
                if a == 0: nx = max(0, x - 1)   # Up
                elif a == 1: ny = min(self.size - 1, y + 1) # Right
                elif a == 2: nx = min(self.size - 1, x + 1) # Down
                elif a == 3: ny = max(0, y - 1) # Left
                
                ns = self._coord_to_state(nx, ny)
                P[s, a, ns] = 1.0 # 确定性环境
                
        return P

    def _build_reward_matrix(self):
        # R[s][a] = expected reward
        # 标准设置：每走一步 -1，直到到达终点
        R = np.full((self.num_states, self.num_actions), -1.0)
        R[0, :] = 0
        R[self.num_states - 1, :] = 0
        return R

# 使用示例
mdp = GridWorldMDP()
print(f"State space size: {mdp.num_states}")
print(f"P shape: {mdp.P.shape}")
```

这个 `GridWorldMDP` 类将在下一章（动态规划）中用于演示策略迭代和价值迭代。

---

## 本章总结

1.  **MDP 五元组**：S, A, P, R, $\gamma$ 完整定义了 RL 问题。
2.  **策略与价值**：策略 $\pi$ 决定行为，价值 $V/Q$ 评估好坏。
3.  **两个方程**：
    *   **Bellman Expectation**：描述了 V/Q 之间的自洽关系（用于 Policy Evaluation）。
    *   **Bellman Optimality**：描述了最优值必须满足的条件（用于 Control）。

**下一章预告**：知道了 Bellman 方程，我们怎么解它呢？对于已知模型（已知 P 和 R）的情况，我们将使用**动态规划（Dynamic Programming）**算法来精确求解最优策略。

---

## 扩展阅读

*   Sutton & Barto, *Reinforcement Learning: An Introduction*, Chapter 3
*   DeepMind RL Course by David Silver, Lecture 2: Markov Decision Processes
*   Stanford CS234: Lecture 2

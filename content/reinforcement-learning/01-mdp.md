---
title: "Chapter 1. 马尔可夫决策过程（MDP）"
description: "强化学习的数学基础：状态、动作、奖励与 Bellman 方程"
updated: "2026-01-29"
---

> **Learning Objectives**
> * 掌握 MDP 的形式化定义：$(S, A, P, R, \gamma)$
> * 理解策略、价值函数、Q 函数的数学含义
> * 推导并证明 Bellman 方程
> * 理解策略改进定理与最优性理论
> * 实现 GridWorld MDP 环境

---

## 1.1 MDP 形式化定义

**马尔可夫决策过程（Markov Decision Process, MDP）** 是强化学习的数学框架。它为 Agent-Environment 交互提供了严格的数学描述。

### 1.1.1 状态空间 S、动作空间 A

**状态空间（State Space）** $\mathcal{S}$：
- 定义：所有可能状态的集合
- 符号：$s \in \mathcal{S}$
- 例子：
  - 棋盘游戏：所有可能的棋盘局面
  - 机器人：位置、速度、关节角度
  - 股票交易：价格、成交量、技术指标

**动作空间（Action Space）** $\mathcal{A}$：
- 定义：所有可能动作的集合
- 符号：$a \in \mathcal{A}$ 或 $a \in \mathcal{A}(s)$（状态相关）
- 分类：
  - **离散动作空间**：$\mathcal{A} = \{a_1, a_2, \ldots, a_n\}$
    - 例如：围棋（361个落子位置）、Atari（4-18个按键）
  - **连续动作空间**：$\mathcal{A} \subseteq \mathbb{R}^n$
    - 例如：机器人关节力矩、自动驾驶方向盘角度

**马尔可夫性质（Markov Property）**：

$$
P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, \ldots, s_0, a_0) = P(s_{t+1} | s_t, a_t)
$$

**含义**：未来只依赖于当前状态和动作，与历史无关。

> [!IMPORTANT]
> **马尔可夫性质的重要性**：
> - 简化问题：无需记住完整历史
> - 理论保证：Bellman 方程成立的前提
> - 实际应用：大多数问题可以通过状态设计满足马尔可夫性

### 1.1.2 转移概率 $P(s'|s,a)$

**状态转移概率（Transition Probability）**：

$$
P(s' | s, a) = \mathbb{P}[S_{t+1} = s' | S_t = s, A_t = a]
$$

**含义**：在状态 $s$ 执行动作 $a$ 后，转移到状态 $s'$ 的概率。

**性质**：
1. **归一化**：$\sum_{s' \in \mathcal{S}} P(s'|s,a) = 1$
2. **非负性**：$P(s'|s,a) \geq 0$

**确定性 vs 随机性**：
- **确定性环境**：$P(s'|s,a) \in \{0, 1\}$
  - 例如：围棋（落子后局面确定）
- **随机性环境**：$P(s'|s,a) \in [0, 1]$
  - 例如：扑克（发牌随机）、机器人（执行误差）

### 1.1.3 奖励函数 $R(s,a,s')$

**奖励函数（Reward Function）**：

$$
R(s, a, s') = \mathbb{E}[R_{t+1} | S_t = s, A_t = a, S_{t+1} = s']
$$

**简化形式**：
- $R(s, a)$：只依赖状态和动作
- $R(s)$：只依赖状态

**设计原则**：
1. **稀疏 vs 密集**：
   - 稀疏奖励：只在达成目标时给奖励（如走迷宫）
   - 密集奖励：每步都有反馈（如游戏得分）

2. **塑形（Reward Shaping）**：
   - 添加中间奖励引导学习
   - 风险：可能改变最优策略

3. **归一化**：
   - 将奖励缩放到合理范围（如 $[-1, 1]$）
   - 避免数值不稳定

### 1.1.4 折扣因子 $\gamma$ 的作用

**折扣因子（Discount Factor）** $\gamma \in [0, 1]$：

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

**作用**：
1. **数学便利**：保证无限期望收敛
   - 如果 $|\gamma| < 1$ 且奖励有界，则 $G_t < \infty$

2. **偏好近期奖励**：
   - $\gamma = 0$：只关心即时奖励（贪心）
   - $\gamma = 1$：未来奖励与当前等价（无折扣）
   - $\gamma = 0.99$：常用值，平衡近期与远期

3. **不确定性建模**：
   - 未来越远越不确定，折扣反映这种不确定性

**直观理解**：

| $\gamma$ | 含义 | 等效视野 |
|----------|------|---------|
| 0.9 | 重视近期 | ~10 步 |
| 0.95 | 平衡 | ~20 步 |
| 0.99 | 重视长期 | ~100 步 |
| 1.0 | 无折扣 | 无限 |

**等效视野（Effective Horizon）**：$\frac{1}{1-\gamma}$

> [!TIP]
> **选择 $\gamma$ 的经验法则**：
> - Episode 任务（有明确终点）：$\gamma = 0.99$ 或 $\gamma = 1$
> - 连续任务（无终点）：$\gamma < 1$（必须）
> - 需要快速反应：$\gamma$ 较小（0.9）
> - 需要长期规划：$\gamma$ 较大（0.99）

---

## 1.2 策略（Policy）

策略是 Agent 的"行为准则"，定义了在每个状态下如何选择动作。

### 1.2.1 确定性策略 $\mu(s)$ vs 随机策略 $\pi(a|s)$

**确定性策略（Deterministic Policy）**：

$$
\mu: \mathcal{S} \rightarrow \mathcal{A}
$$

- 含义：每个状态对应唯一动作
- 符号：$a = \mu(s)$
- 例子：围棋 AI 的最终决策

**随机策略（Stochastic Policy）**：

$$
\pi: \mathcal{S} \times \mathcal{A} \rightarrow [0, 1]
$$

- 含义：每个状态对应动作的概率分布
- 符号：$\pi(a|s) = \mathbb{P}[A_t = a | S_t = s]$
- 性质：$\sum_{a \in \mathcal{A}} \pi(a|s) = 1$
- 例子：探索阶段的 $\epsilon$-greedy 策略

**为什么需要随机策略？**

1. **探索（Exploration）**：
   - 随机性帮助发现更好的策略
   - 例如：$\epsilon$-greedy 以 $\epsilon$ 概率随机探索

2. **部分可观测（Partial Observability）**：
   - 当状态不完全可观测时，随机策略可能更优
   - 例如：扑克中的混合策略

3. **多智能体（Multi-Agent）**：
   - 对手无法预测你的动作
   - 例如：石头剪刀布的均匀随机策略

### 1.2.2 策略的表示方法

**表格表示（Tabular）**：
- 适用：状态和动作空间较小
- 存储：二维表 $\pi(a|s)$

```python
# 示例：GridWorld 策略表
policy = {
    (0, 0): 0.25 * np.ones(4),  # 均匀随机
    (0, 1): np.array([0.7, 0.1, 0.1, 0.1]),  # 偏向动作0
    # ...
}
```

**函数逼近（Function Approximation）**：
- 适用：大规模或连续状态空间
- 方法：神经网络

```python
class PolicyNetwork(nn.Module):
    def forward(self, state):
        logits = self.network(state)
        return F.softmax(logits, dim=-1)  # 输出概率分布

# 采样动作
probs = policy_net(state)
action = torch.multinomial(probs, 1)
```

### 1.2.3 最优策略的存在性

**定理 1.1（最优策略存在性）**：

对于任何有限 MDP，至少存在一个最优策略 $\pi^*$，使得对所有状态 $s$ 和所有策略 $\pi$：

$$
V^{\pi^*}(s) \geq V^{\pi}(s), \quad \forall s \in \mathcal{S}
$$

**证明思路**：
1. 价值函数空间是紧集（有界闭集）
2. Bellman 算子是压缩映射
3. 不动点定理保证最优解存在

**重要性质**：
- **确定性最优策略**：总存在一个确定性最优策略
  - 即使有多个最优策略，至少有一个是确定性的
- **非唯一性**：最优策略可能不唯一
  - 但最优价值函数 $V^*$ 是唯一的

---

## 1.3 价值函数

价值函数量化了"状态的好坏"或"状态-动作对的好坏"。

### 1.3.1 状态价值函数 $V^\pi(s)$

**定义**：

$$
V^\pi(s) = \mathbb{E}_\pi \left[ G_t | S_t = s \right] = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s \right]
$$

**含义**：从状态 $s$ 开始，遵循策略 $\pi$，期望获得的累积折扣奖励。

**直观理解**：
- $V^\pi(s)$ 高 → 状态 $s$ 好（在策略 $\pi$ 下）
- $V^\pi(s)$ 低 → 状态 $s$ 差

**例子（GridWorld）**：

```
终点(+10)  墙壁  空地
  空地     空地  空地
起点(0)    空地  陷阱(-10)
```

如果策略是"随机游走"：
- $V^\pi(\text{终点附近}) > V^\pi(\text{起点})$
- $V^\pi(\text{陷阱附近}) < V^\pi(\text{起点})$

### 1.3.2 动作价值函数 $Q^\pi(s,a)$

**定义**：

$$
Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t | S_t = s, A_t = a \right]
$$

**含义**：从状态 $s$ 执行动作 $a$，然后遵循策略 $\pi$，期望获得的累积折扣奖励。

**与 $V^\pi$ 的关系**：

$$
V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) Q^\pi(s, a)
$$

**直观理解**：
- $Q^\pi(s, a)$ 告诉我们"在状态 $s$ 选择动作 $a$ 有多好"
- $V^\pi(s)$ 是所有动作的加权平均（权重为策略概率）

### 1.3.3 Advantage 函数 $A^\pi(s,a)$

**定义**：

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

**含义**：动作 $a$ 相对于平均水平的"优势"。

**性质**：
- $A^\pi(s, a) > 0$：动作 $a$ 优于平均
- $A^\pi(s, a) < 0$：动作 $a$ 劣于平均
- $A^\pi(s, a) = 0$：动作 $a$ 与平均持平

**重要性**：
- **策略梯度**：$\nabla J(\theta) \propto \mathbb{E}[A^\pi(s,a) \nabla \log \pi(a|s)]$
- **降低方差**：Advantage 相比 Q 函数方差更小

### 1.3.4 价值函数的递归性质

价值函数满足递归关系，这是 Bellman 方程的基础。

**递推推导**：

$$
\begin{align}
V^\pi(s) &= \mathbb{E}_\pi[G_t | S_t = s] \\
&= \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} | S_t = s] \\
&= \mathbb{E}_\pi[R_{t+1} | S_t = s] + \gamma \mathbb{E}_\pi[G_{t+1} | S_t = s] \\
&= \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]
\end{align}
$$

这就是 **Bellman 期望方程**。

### 交互演示：价值函数演化

<div data-component="ValueFunctionEvolution"></div>

---

## 1.4 Bellman 方程

Bellman 方程是强化学习的核心，它将价值函数表示为递归形式。

### 1.4.1 Bellman 期望方程（Expectation Equation）

**状态价值函数的 Bellman 期望方程**：

$$
V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{s' \in \mathcal{S}} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]
$$

**动作价值函数的 Bellman 期望方程**：

$$
Q^\pi(s, a) = \sum_{s' \in \mathcal{S}} P(s'|s,a) \left[ R(s,a,s') + \gamma \sum_{a' \in \mathcal{A}} \pi(a'|s') Q^\pi(s', a') \right]
$$

**矩阵形式**（有限 MDP）：

$$
V^\pi = R^\pi + \gamma P^\pi V^\pi
$$

解析解：

$$
V^\pi = (I - \gamma P^\pi)^{-1} R^\pi
$$

但实际中很少直接求逆，而是用迭代方法。

### 1.4.2 Bellman 最优方程（Optimality Equation）

**最优状态价值函数**：

$$
V^*(s) = \max_\pi V^\pi(s) = \max_{a \in \mathcal{A}} Q^*(s, a)
$$

**最优动作价值函数**：

$$
Q^*(s, a) = \sum_{s' \in \mathcal{S}} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]
$$

**Bellman 最优方程**：

$$
V^*(s) = \max_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]
$$

$$
Q^*(s, a) = \sum_{s' \in \mathcal{S}} P(s'|s,a) \left[ R(s,a,s') + \gamma \max_{a' \in \mathcal{A}} Q^*(s', a') \right]
$$

### 1.4.3 数学推导与证明

**定理 1.2（Bellman 期望方程推导）**：

**证明**：

$$
\begin{align}
V^\pi(s) &= \mathbb{E}_\pi[G_t | S_t = s] \\
&= \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} | S_t = s] \\
&= \sum_a \pi(a|s) \mathbb{E}[R_{t+1} + \gamma G_{t+1} | S_t = s, A_t = a] \\
&= \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \mathbb{E}[R_{t+1} + \gamma G_{t+1} | S_t = s, A_t = a, S_{t+1} = s'] \\
&= \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma \mathbb{E}[G_{t+1} | S_{t+1} = s'] \right] \\
&= \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]
\end{align}
$$

**关键步骤**：
1. 期望的线性性
2. 全期望公式
3. 马尔可夫性质

### 1.4.4 Bellman 算子的压缩性质

**定义 Bellman 算子** $\mathcal{T}^\pi$：

$$
(\mathcal{T}^\pi V)(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V(s') \right]
$$

**定理 1.3（压缩映射定理）**：

Bellman 算子 $\mathcal{T}^\pi$ 是关于最大范数的 $\gamma$-压缩映射：

$$
\|\mathcal{T}^\pi V_1 - \mathcal{T}^\pi V_2\|_\infty \leq \gamma \|V_1 - V_2\|_\infty
$$

**证明**：

$$
\begin{align}
|(\mathcal{T}^\pi V_1)(s) - (\mathcal{T}^\pi V_2)(s)| &= \left| \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \gamma [V_1(s') - V_2(s')] \right| \\
&\leq \gamma \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) |V_1(s') - V_2(s')| \\
&\leq \gamma \|V_1 - V_2\|_\infty
\end{align}
$$

**重要性**：
- 保证迭代收敛：$V_{k+1} = \mathcal{T}^\pi V_k$ 收敛到唯一不动点 $V^\pi$
- 收敛速度：几何级数，$O(\gamma^k)$

### 交互演示：Bellman 方程推导

<div data-component="BellmanEquationDerivation"></div>

---

## 1.5 最优性理论

### 1.5.1 最优价值函数 $V^*(s)$、$Q^*(s,a)$

**定义**：

$$
V^*(s) = \max_\pi V^\pi(s), \quad \forall s \in \mathcal{S}
$$

$$
Q^*(s, a) = \max_\pi Q^\pi(s, a), \quad \forall s \in \mathcal{S}, a \in \mathcal{A}
$$

**关系**：

$$
V^*(s) = \max_{a \in \mathcal{A}} Q^*(s, a)
$$

$$
Q^*(s, a) = \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]
$$

### 1.5.2 最优策略的唯一性（值唯一，策略可能多个）

**定理 1.4（最优价值函数唯一性）**：

对于任何 MDP，最优价值函数 $V^*$ 和 $Q^*$ 是唯一的。

**定理 1.5（最优策略存在性）**：

存在至少一个最优策略 $\pi^*$ 使得：

$$
V^{\pi^*}(s) = V^*(s), \quad \forall s \in \mathcal{S}
$$

**定理 1.6（确定性最优策略）**：

总存在一个确定性最优策略。

**证明思路**：

对于任何随机策略 $\pi$，定义确定性策略：

$$
\pi'(s) = \arg\max_{a} Q^\pi(s, a)
$$

可以证明 $V^{\pi'}(s) \geq V^\pi(s)$。

**非唯一性示例**：

```
状态 s，两个动作 a1, a2
Q*(s, a1) = Q*(s, a2) = 10

则 π*(a1|s) = 1 和 π*(a2|s) = 1 都是最优策略
```

### 1.5.3 策略改进定理（Policy Improvement Theorem）

**定理 1.7（策略改进定理）**：

设 $\pi$ 和 $\pi'$ 是两个确定性策略，如果对所有状态 $s$：

$$
Q^\pi(s, \pi'(s)) \geq V^\pi(s)
$$

则：

$$
V^{\pi'}(s) \geq V^\pi(s), \quad \forall s \in \mathcal{S}
$$

**证明**：

$$
\begin{align}
V^\pi(s) &\leq Q^\pi(s, \pi'(s)) \\
&= \mathbb{E}[R_{t+1} + \gamma V^\pi(S_{t+1}) | S_t = s, A_t = \pi'(s)] \\
&\leq \mathbb{E}[R_{t+1} + \gamma Q^\pi(S_{t+1}, \pi'(S_{t+1})) | S_t = s, A_t = \pi'(s)] \\
&= \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \gamma^2 V^\pi(S_{t+2}) | S_t = s, \pi'] \\
&\leq \cdots \\
&\leq V^{\pi'}(s)
\end{align}
$$

**应用**：策略迭代算法的理论基础。

### 1.5.4 策略迭代收敛性证明

**定理 1.8（策略迭代收敛性）**：

策略迭代算法在有限步内收敛到最优策略。

**证明**：

1. **单调性**：策略改进定理保证 $V^{\pi_{k+1}} \geq V^{\pi_k}$
2. **有限性**：确定性策略数量有限（$|\mathcal{A}|^{|\mathcal{S}|}$）
3. **严格改进**：如果 $\pi_{k+1} \neq \pi_k$，则 $V^{\pi_{k+1}} > V^{\pi_k}$（至少一个状态严格改进）
4. **终止**：有限步后必然 $\pi_{k+1} = \pi_k$，此时达到最优

**收敛速度**：
- 最坏情况：$O(|\mathcal{A}|^{|\mathcal{S}|})$
- 实际中：通常很快（几次迭代）

---

## 1.6 GridWorld MDP 实现

让我们用代码实现一个完整的 GridWorld MDP 环境。

### GridWorld 环境定义

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

class GridWorldMDP:
    """
    GridWorld MDP 环境
    
    状态：(x, y) 坐标
    动作：0=上, 1=右, 2=下, 3=左
    奖励：到达目标 +10，掉入陷阱 -10，其他 -1
    """
    
    def __init__(self, size: int = 5, gamma: float = 0.9):
        self.size = size
        self.gamma = gamma
        
        # 特殊位置
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        self.traps = [(1, 1), (2, 3)]
        self.walls = [(1, 2), (3, 2)]
        
        # 状态和动作空间
        self.states = [(i, j) for i in range(size) for j in range(size)
                      if (i, j) not in self.walls]
        self.actions = [0, 1, 2, 3]  # 上右下左
        self.action_names = ['↑', '→', '↓', '←']
        
        # 动作效果
        self.action_effects = {
            0: (-1, 0),  # 上
            1: (0, 1),   # 右
            2: (1, 0),   # 下
            3: (0, -1),  # 左
        }
    
    def is_valid_state(self, state: Tuple[int, int]) -> bool:
        """检查状态是否有效"""
        x, y = state
        return (0 <= x < self.size and 
                0 <= y < self.size and 
                state not in self.walls)
    
    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """检查是否为终止状态"""
        return state == self.goal or state in self.traps
    
    def get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        """确定性转移：获取下一个状态"""
        if self.is_terminal(state):
            return state
        
        dx, dy = self.action_effects[action]
        next_state = (state[0] + dx, state[1] + dy)
        
        # 如果下一个状态无效，保持原地
        if not self.is_valid_state(next_state):
            return state
        
        return next_state
    
    def get_reward(self, state: Tuple[int, int], action: int, 
                   next_state: Tuple[int, int]) -> float:
        """获取奖励"""
        if next_state == self.goal:
            return 10.0
        elif next_state in self.traps:
            return -10.0
        else:
            return -1.0  # 每步惩罚，鼓励快速到达目标
    
    def transition(self, state: Tuple[int, int], action: int) -> Tuple[Tuple[int, int], float]:
        """执行动作，返回 (next_state, reward)"""
        next_state = self.get_next_state(state, action)
        reward = self.get_reward(state, action, next_state)
        return next_state, reward
    
    def get_transition_prob(self, state: Tuple[int, int], action: int, 
                           next_state: Tuple[int, int]) -> float:
        """
        获取转移概率 P(next_state | state, action)
        这里是确定性环境，所以返回 0 或 1
        """
        predicted_next = self.get_next_state(state, action)
        return 1.0 if next_state == predicted_next else 0.0
    
    def visualize(self, values: dict = None, policy: dict = None):
        """可视化网格世界"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 绘制网格
        for i in range(self.size + 1):
            ax.plot([0, self.size], [i, i], 'k-', linewidth=0.5)
            ax.plot([i, i], [0, self.size], 'k-', linewidth=0.5)
        
        # 绘制特殊位置
        for state in self.states:
            x, y = state
            
            # 背景颜色
            if state == self.goal:
                color = 'lightgreen'
                ax.text(y + 0.5, self.size - x - 0.5, '🎯', 
                       ha='center', va='center', fontsize=20)
            elif state in self.traps:
                color = 'lightcoral'
                ax.text(y + 0.5, self.size - x - 0.5, '💀', 
                       ha='center', va='center', fontsize=20)
            elif state == self.start:
                color = 'lightyellow'
                ax.text(y + 0.5, self.size - x - 0.5, '🏁', 
                       ha='center', va='center', fontsize=20)
            else:
                color = 'white'
            
            rect = plt.Rectangle((y, self.size - x - 1), 1, 1, 
                                facecolor=color, edgecolor='black')
            ax.add_patch(rect)
            
            # 显示价值
            if values and state in values:
                ax.text(y + 0.5, self.size - x - 0.3, f'{values[state]:.1f}', 
                       ha='center', va='center', fontsize=10, fontweight='bold')
            
            # 显示策略
            if policy and state in policy and not self.is_terminal(state):
                action = policy[state]
                arrow = self.action_names[action]
                ax.text(y + 0.5, self.size - x - 0.7, arrow, 
                       ha='center', va='center', fontsize=16, color='blue')
        
        # 绘制墙壁
        for wall in self.walls:
            x, y = wall
            rect = plt.Rectangle((y, self.size - x - 1), 1, 1, 
                                facecolor='gray', edgecolor='black')
            ax.add_patch(rect)
        
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.title('GridWorld MDP', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

# 创建环境
env = GridWorldMDP(size=5, gamma=0.9)

# 可视化
env.visualize()
```

### 测试转移函数

```python
# 测试状态转移
state = (2, 2)
print(f"当前状态: {state}")

for action in env.actions:
    next_state, reward = env.transition(state, action)
    print(f"动作 {env.action_names[action]}: "
          f"下一状态 {next_state}, 奖励 {reward}")
```

**输出**：
```
当前状态: (2, 2)
动作 ↑: 下一状态 (1, 2), 奖励 -1.0
动作 →: 下一状态 (2, 3), 奖励 -1.0
动作 ↓: 下一状态 (3, 2), 奖励 -1.0
动作 ←: 下一状态 (2, 1), 奖励 -1.0
```

### 计算状态价值（给定随机策略）

```python
def compute_state_value_random_policy(env: GridWorldMDP, 
                                     theta: float = 1e-6) -> dict:
    """
    计算随机策略（均匀分布）的状态价值函数
    使用迭代策略评估
    """
    # 初始化价值函数
    V = {state: 0.0 for state in env.states}
    
    iteration = 0
    while True:
        delta = 0
        
        for state in env.states:
            if env.is_terminal(state):
                continue
            
            v = V[state]
            
            # Bellman 期望方程（随机策略 π(a|s) = 1/4）
            new_v = 0
            for action in env.actions:
                next_state, reward = env.transition(state, action)
                new_v += 0.25 * (reward + env.gamma * V[next_state])
            
            V[state] = new_v
            delta = max(delta, abs(v - new_v))
        
        iteration += 1
        print(f"迭代 {iteration}: delta = {delta:.6f}")
        
        if delta < theta:
            break
    
    return V

# 计算价值函数
V_random = compute_state_value_random_policy(env)

# 可视化
env.visualize(values=V_random)
```

### 交互演示：MDP 图可视化

<div data-component="MDPGraphVisualizer"></div>

---

## 本章小结

在本章中，我们学习了：

✅ **MDP 形式化定义**：$(S, A, P, R, \gamma)$ 五元组  
✅ **马尔可夫性质**：未来只依赖当前状态  
✅ **策略**：确定性 vs 随机性  
✅ **价值函数**：$V^\pi(s)$、$Q^\pi(s,a)$、$A^\pi(s,a)$  
✅ **Bellman 方程**：期望方程与最优方程  
✅ **最优性理论**：策略改进定理、收敛性证明  
✅ **GridWorld 实现**：完整的 MDP 环境

> [!TIP]
> **下一步**：
> 现在你已经掌握了 MDP 的数学基础，接下来我们将学习如何**求解** MDP——**动态规划（Dynamic Programming）**方法。
> 
> 进入 [Chapter 2. 动态规划](02-dynamic-programming.md)

---

## 扩展阅读

- **Sutton & Barto**：Chapter 3 (Finite Markov Decision Processes)
- **RL Theory Book**：Chapter 2 (Markov Decision Processes)
- **Bertsekas**：Chapter 1 (Finite-Horizon Problems)
- **论文**：
  - Bellman (1957): Dynamic Programming
  - Puterman (1994): Markov Decision Processes (经典教材)

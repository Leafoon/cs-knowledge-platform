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

### 1.1.1 马尔可夫性质 (Markov Property)

> "The future is independent of the past given the present."

如果在当前状态 $S_t$ 已知的情况下，未来状态 $S_{t+1}$ 只与当前状态 $S_t$ 和动作 $A_t$ 有关，而与历史状态 $S_{t-1}, S_{t-2}, \dots$ 无关，则称该状态具有**马尔可夫性质**。

$$ \Pr(S_{t+1} \mid S_t, A_t) = \Pr(S_{t+1} \mid S_t, A_t, S_{t-1}, A_{t-1}, \dots) $$

**为什么这很重要？**
这意味着 state $s$ 包含了决策所需的所有信息。如果环境是非马尔可夫的（例如扑克牌中不知道对手手牌），我们通常需要通过堆叠历史帧（Frame Stacking）或使用循环神经网络（RNN）来近似恢复马尔可夫状态。

### 1.1.2 回报 (Return)

Agent 的目标是最大化**累积折扣回报 (Cumulative Discounted Return)** $G_t$：

$$ G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^\infty \gamma^k R_{t+k+1} $$

**为什么需要折扣因子 $\gamma$？**
1.  **数学收敛性**：对于无限长度的任务，若不衰减，回报可能无穷大。$\gamma < 1$ 保证了级数收敛。
2.  **不确定性**：未来的预测往往不如当前确定，衰减反映了对未来的不确定性。
3.  **金融解释**：现在的 1 元钱比未来的 1 元钱更有价值（利息）。

---

## 1.2 策略 (Policy)

策略 $\pi$ 定义了 Agent 在给定状态下的行为方式。

### 1.2.1 随机策略 vs 确定性策略

1.  **随机策略 (Stochastic Policy)**：$\pi(a|s) = \Pr(A_t=a \mid S_t=s)$
    *   输出一个动作的概率分布。
    *   适用于探索（Exploration）或二人博弈（如石头剪刀布）。
2.  **确定性策略 (Deterministic Policy)**：$a = \pi(s)$
    *   直接输出一个具体的动作。
    *   通常在最优策略收敛后使用。

### 1.2.2 策略的表示

在代码中，策略通常有两种表示：
*   **表格 (Tabular)**：一个 $|S| \times |A|$ 的矩阵（仅限小规模离散状态）。
*   **函数逼近 (Function Approximation)**：一个神经网络 $\pi_\theta(s)$，输入状态，输出动作分布或 Q 值。

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

---
title: "第2章：动态规划 (Dynamic Programming)"
description: "求解 MDP 的经典方法：策略迭代、价值迭代与广义策略迭代"
date: "2026-01-30"
---

# 第2章：动态规划 (Dynamic Programming)

在上一章我们建立了 MDP 和 Bellman 方程。本章，我们将假设**环境模型是已知的**（即已知 $P(s'|s,a)$ 和 $R(s,a)$），并使用动态规划（DP）方法精确求解最优策略。

虽然 DP 在大规模问题中往往不可行（维度灾难），但它是所有现代 RL 算法的理论基石。

---

## 2.1 策略评估 (Policy Evaluation)

**目标**：给定一个策略 $\pi$，计算其状态价值函数 $V^\pi(s)$。

根据 Bellman 期望方程：
$$ V^\pi(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r|s,a) [r + \gamma V^\pi(s')] $$

这是一个线性方程组（$|S|$ 个方程，$|S|$ 个未知数）。对于小规模 MDP 可以直接解矩阵逆，但计算量是 $O(|S|^3)$。
DP 使用**迭代法**求解，将 Bellman 方程变为更新规则：

$$ V_{k+1}(s) \leftarrow \sum_a \pi(a|s) \sum_{s', r} p(s', r|s,a) [r + \gamma V_k(s')] $$

**收敛性**：由 **Banach 不动点定理** 保证，只要 $\gamma < 1$，序列 $V_k$ 必定收敛到唯一不动点 $V^\pi$。

### 代码实现

让我们使用上一章的 `GridWorldMDP` 来说明：

```python
import numpy as np

def policy_evaluation(mdp, policy, theta=1e-6):
    """
    迭代策略评估
    Args:
        mdp: GridWorldMDP 实例
        policy: 大小为 [num_states] 的数组，存储动作索引
    Returns:
        V: 状态价值函数
    """
    V = np.zeros(mdp.num_states)
    
    while True:
        delta = 0
        # 对每个状态进行扫描
        for s in range(mdp.num_states):
            old_v = V[s]
            
            # 由于是确定性策略，直接取由 policy[s] 指定的动作
            a = policy[s]
            
            # 计算期望价值
            # GridWorld 是确定性转移，sum 去掉
            # P[s][a][s'] == 1 的那个 s'
            # 这里简化写出通用形式：
            new_v = 0
            for s_prime in range(mdp.num_states):
                prob = mdp.P[s, a, s_prime]
                if prob > 0:
                    reward = mdp.R[s, a]
                    new_v += prob * (reward + mdp.gamma * V[s_prime])
            
            V[s] = new_v
            delta = max(delta, abs(old_v - V[s]))
            
        if delta < theta:
            break
            
    return V
```

---

## 2.2 策略改进 (Policy Improvement)

**目标**：给定当前策略 $\pi$ 及其价值 $V^\pi$，找到一个更好的策略 $\pi'$。

根据 **策略改进定理 (Policy Improvement Theorem)**，如果我们在某个状态 $s$ 下，选择动作 $a \ne \pi(s)$，使得 $Q^\pi(s, a) > V^\pi(s)$，那么永久改变该状态下的策略为 $a$ 将严格提升整体策略性能。

贪心改进策略：
$$ \pi'(s) = \arg\max_a Q^\pi(s, a) = \arg\max_a \sum_{s', r} p(s', r|s,a) [r + \gamma V^\pi(s')] $$

---

## 2.3 策略迭代 (Policy Iteration)

结合评估与改进，我们得到策略迭代算法：

1.  **初始化**：任意策略 $\pi_0$，任意 $V_0$。
2.  **策略评估 (PE)**：计算 $V^{\pi_k}$。
3.  **策略改进 (PI)**：$\pi_{k+1}(s) = \arg\max_a Q^{\pi_k}(s, a)$。
4.  **终止**：若 $\pi_{k+1} = \pi_k$，则停止，此时 $\pi^* = \pi_k$。

<div data-component="PolicyIterationVisualizer"></div>

**特点**：
*   通常迭代次数很少就能收敛。
*   但每一步 PE 都需要迭代很多次直到 $V$ 收敛，计算开销大。

---

## 2.4 价值迭代 (Value Iteration)

为了解决策略迭代中 PE 耗时的问题，我们可以**截断** PE 过程——只更新一次 $V$ 就进行策略改进，或者干脆直接把 Bellman 最优方程变成迭代更新规则：

$$ V_{k+1}(s) \leftarrow \max_a \sum_{s', r} p(s', r|s,a) [r + \gamma V_k(s')] $$

这里不再显式维护策略 $\pi$，而是直接迭代 $V$。当 $V$ 收敛后，再提取最优策略。

<div data-component="ValueIterationConvergence"></div>

### 2.4.1 代码实现

```python
def value_iteration(mdp, theta=1e-6):
    """
    价值迭代算法
    Returns:
        policy: 最优确定性策略
        V: 最优状态价值
    """
    V = np.zeros(mdp.num_states)
    
    # 1. 迭代更新 V 直到收敛
    while True:
        delta = 0
        for s in range(mdp.num_states):
            old_v = V[s]
            
            # 计算所有动作的 Q 值
            q_values = []
            for a in range(mdp.num_actions):
                q = 0
                for s_prime in range(mdp.num_states):
                    p = mdp.P[s, a, s_prime]
                    if p > 0:
                        r = mdp.R[s, a]
                        q += p * (r + mdp.gamma * V[s_prime])
                q_values.append(q)
            
            # Bellman Optimality Update
            V[s] = max(q_values)
            delta = max(delta, abs(old_v - V[s]))
            
        if delta < theta:
            break
            
    # 2. 从 V 提取最优策略
    policy = np.zeros(mdp.num_states, dtype=int)
    for s in range(mdp.num_states):
        q_values = []
        for a in range(mdp.num_actions):
            q = 0
            for s_prime in range(mdp.num_states):
                p = mdp.P[s, a, s_prime]
                if p > 0:
                    r = mdp.R[s, a]
                    q += p * (r + mdp.gamma * V[s_prime])
            q_values.append(q)
        policy[s] = np.argmax(q_values)
        
    return policy, V
```

### 2.4.2 策略迭代 vs 价值迭代

| 特性 | 策略迭代 (PI) | 价值迭代 (VI) |
|:---|:---|:---|
| **核心** | 显式策略 $\pi \to V \to \pi'$ | 隐式策略 $V \to V'$ |
| **每步代价** | 高 (完整的 Policy Evaluation) | 低 (一次 Sweep) |
| **迭代步数** | 少 | 多 |
| **收敛性** | 确切收敛到最优策略 | 渐近收敛到最优值 |
| **适用性** | 动作空间较小 | 状态空间稍大时 |

---

## 2.5 广义策略迭代 (Generalized Policy Iteration, GPI)

"策略迭代"和"价值迭代"其实是两个极端。
*   PI：完全评估 $V^\pi$ 后再改进。
*   VI：仅评估一步（截断）就改进。

**GPI** 是一个统称，指代所有交替进行**策略评估**和**策略改进**的方法，无论评估进行得多么不精确。

<div data-component="GPIFramework"></div>

几乎所有的强化学习算法（包括 DQN, PPO, SAC）都可以看作是 GPI 的某种近似实现：
*   **Critic** 负责 Policy Evaluation（估计 $Q$ 或 $V$）。
*   **Actor** 负责 Policy Improvement（基于估计的价值更新策略）。

---

## 2.6 DP 的局限性

虽然 DP 很完美，但它要求：
1.  **已知环境模型**：必须知道 $P(s'|s,a)$ 和 $R(s,a)$。现实任务（如自动驾驶）很难建模。
2.  **全宽扫描 (Full-width backup)**：每次更新都要遍历所有状态。对于围棋（状态 $10^{170}$），这根本没法算。

这引出了后续章节的主题：当模型未知或状态空间太大时，我们该怎么办？
答案是：**从采样中学习（Monte Carlo 和 Temporal Difference）**。

---

## 本章总结

1.  **策略评估**：解方程求 $V^\pi$。
2.  **策略改进**：贪心选择 $Q$ 值最大的动作提升 $\pi$。
3.  **两个算法**：策略迭代（PE+PI 循环）和价值迭代（直接迭代最优方程）。
4.  **GPI**：所有 RL 算法的通用范式。

**下一章预告**：我们将扔掉完美的上帝视角（环境模型），让 Agent 像一个婴儿一样，通过与其交互、试错来学习。我们将介绍第一种无模型方法——**蒙特卡洛方法（Monte Carlo Methods）**。

---

## 扩展阅读

*   Sutton & Barto, *Reinforcement Learning: An Introduction*, Chapter 4
*   RL Theory Book (Agarwal et al.), Chapter 2: Dynamic Programming

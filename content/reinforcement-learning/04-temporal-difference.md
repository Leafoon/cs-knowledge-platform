---
title: "第4章：时序差分学习 (Temporal Difference Learning)"
description: "强化学习的核心：结合蒙特卡洛的采样与动态规划的自举，实现 SARSA 与 Q-learning"
date: "2026-01-30"
---

# 第4章：时序差分学习 (Temporal Difference Learning)

如果说蒙特卡洛（MC）方法是“不撞南墙不回头”的执着者，那么时序差分（TD）学习就是“走一步看一步”的即兴大师。

TD 学习结合了 **Monte Carlo（采样）** 和 **Dynamic Programming（自举）** 的思想，是强化学习中最核心、最独特，也是应用最广泛的方法。

---

## 4.1 TD 预测 (TD Prediction)

**问题**：如何估计 $V^\pi(s)$？

### 4.1.1 MC vs TD：更新时机的博弈

*   **MC 更新**：必须等到 Episode 结束，拿到真实的 $G_t$。
    $$ V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)] $$
*   **TD(0) 更新**：只要走一步，拿到即时奖励 $R_{t+1}$ 和下一个状态 $S_{t+1}$，就立刻更新。
    $$ V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)] $$

这里的 $R_{t+1} + \gamma V(S_{t+1})$ 被称为 **TD Target**。
这里的 $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ 被称为 **TD Error**。

**核心思想：Bootstrapping (自举)**
TD 方法用一个**估计值**（$V(S_{t+1})$）来更新另一个**估计值**（$V(S_t)$）。这听起来像是在“左脚踩右脚上天”，但在统计学上，也就是在进行 Bellman 方程的迭代求解。

<div data-component="TDUpdateVisualizer"></div>

### 4.1.2 TD 的优势

1.  **Online Learning**：可以在 Episode 进行中持续学习，不需要等待结束。这对长任务或无尽任务至关重要。
2.  **Variance Reduction**：$G_t$ 取决于未来所有随机结果，方差很大；而 TD Target 只取决于一步随机性和 $V(S_{t+1})$，方差通常更小，收敛更快。
3.  **Model-free**：不需要环境模型 P。

---

## 4.2 SARSA: On-policy TD Control

我们将 TD 思想应用到控制问题（寻找最优策略），即学习 $Q(s,a)$。

SARSA 的名字来源于其更新所使用的五元组：$(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$。

**更新规则**：
$$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)] $$

**算法流程**：
1.  在 $S$ 状态，使用 $\epsilon$-greedy 策略选择动作 $A$。
2.  执行 $A$，观察 $R, S'$。
3.  在 $S'$ 状态，再次使用 $\epsilon$-greedy 策略选择下一个动作 $A'$。
4.  使用 $Q(S', A')$ 更新 $Q(S, A)$。
5.  $S \leftarrow S', A \leftarrow A'$。

由于所有的动作（包括更新目标 $A'$）都是由当前策略（Behavior Policy）生成的，所以 SARSA 是 **On-policy** 算法。

---

## 4.3 Q-Learning: Off-policy TD Control

Q-Learning 是 RL 历史上最重要的算法之一（Watkins, 1989）。它的核心突破在于：**学习的目标动作可以与实际执行的动作不同**。

**更新规则**：
$$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)] $$

**关键区别**：
*   **实际行为**：可能是 $\epsilon$-greedy 的探索性动作。
*   **更新目标**：始终假设下一步会采取**最优动作** ($\max_a$)。

这意味着 Q-Learning 直接逼近 $Q^*$（最优 Action-Value），而不受当前探索策略的影响（只要探索得足够充分）。

### 4.3.1 实战：SARSA vs Q-Learning (Cliff Walking)

在经典的“悬崖行走”环境中，我们可以清晰地看到两者的区别。
*   **任务**：从起点走到终点，尽量快（每步 -1）。
*   **悬崖**：掉下去会扣 -100 并回到起点。

<div data-component="SARSAvsQLearning"></div>

*   **Q-Learning**：非常“勇敢”。它学到的是**最优路径**（紧贴悬崖边走）。但在训练期间，由于 $\epsilon$-greedy 的随机性，它会经常掉下悬崖。
*   **SARSA**：非常“保守”。它意识到如果走悬崖边，一旦随机探索（$\epsilon$），就会掉下去很疼。所以它学到的是**安全路径**（远离悬崖）。

**代码实现**：

```python
import numpy as np
import gymnasium as gym

def train_td(env_name="CliffWalking-v0", algo="qlearning", episodes=500, alpha=0.5, gamma=0.9):
    env = gym.make(env_name)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    for i in range(episodes):
        state, _ = env.reset()
        done = False
        
        # SARSA 需要先选动作
        action = 0 
        if algo == "sarsa":
            action = epsilon_greedy(Q, state, epsilon=0.1)
            
        while not done:
            if algo == "qlearning":
                action = epsilon_greedy(Q, state, epsilon=0.1)
                
            next_state, reward, done, _, _ = env.step(action)
            
            # TD Target 计算
            if algo == "qlearning":
                # Off-policy: Target = max Q
                target = reward + gamma * np.max(Q[next_state])
            else: # sarsa
                # On-policy: Target = Q(s', a')
                next_action = epsilon_greedy(Q, next_state, epsilon=0.1)
                target = reward + gamma * Q[next_state, next_action]
            
            # Update
            Q[state, action] += alpha * (target - Q[state, action])
            
            state = next_state
            if algo == "sarsa":
                action = next_action
                
    return Q

def epsilon_greedy(Q, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(0, Q.shape[1])
    else:
        return np.argmax(Q[state])
```

---

## 4.4 最大化偏差 (Maximization Bias)

Q-Learning 有一个致命缺陷：**最大化偏差**。

回顾更新公式：$Target = R + \gamma \max_a Q(S', a)$。
我们在**同一个样本集**上既做**估计**（Estimate）又做**最大化**（Maximization）。

由于 $\mathbb{E}[\max(X)] \ge \max(\mathbb{E}[X])$，如果 Q 值的估计存在噪声（这在初期是必然的），$\max$ 操作会倾向于**高估**那些仅仅是因为运气好而被高估的动作。

<div data-component="MaximizationBiasDemo"></div>

### 4.4.1 Double Q-Learning

为了解决这个问题，Hasselt (2010) 提出了 Double Q-Learning。
**核心思想**：使用两个独立的 Q 网络（或 Q 表）$Q_1$ 和 $Q_2$。
*   用 $Q_1$ 来**选择**动作（argmax）。
*   用 $Q_2$ 来**评估**动作（value）。

$$ a^* = \arg\max_a Q_1(S', a) $$
$$ Target = R + \gamma Q_2(S', a^*) $$

这样，即使 $Q_1$ 高估了某个动作，$Q_2$（独立分布）也不太可能同时高估它，从而抵消了偏差。这个思想后来被 DeepMind 用到了 DQN 上（Double DQN），成为标配。

---

## 本章总结

| 特性 | MC (蒙特卡洛) | TD (时序差分) | DP (动态规划) |
| :--- | :--- | :--- | :--- |
| **是否有模型** | Model-free | Model-free | Model-based |
| **更新时机** | Episode 结束 | 每一步 (Step-by-step) | 每一轮 (Sweeps) |
| **方差** | 高 | 低 | 无 (确定性) |
| **偏差** | 无 (Unbiased) | 有 (Initial Bias) | 有 |
| **代表算法** | MC Control | SARSA, Q-Learning | Value Iteration |

**实践建议**：
*   如果你的任务是连续的（无尽头），**必须用 TD**。
*   如果你的模拟器非常慢，且每一步都很昂贵，**TD 的样本效率通常比 MC 高**。
*   **Q-Learning** 通常因其 Off-policy 特性（直接学最优）而被首选，但如果安全性很重要（如机器人不想摔坏），**SARSA** 可能更好。

---

## 扩展阅读

*   Sutton & Barto, *Reinforcement Learning: An Introduction*, Chapter 6
*   [Hasselt, 2010: Double Q-learning](https://papers.nips.cc/paper/3964-double-q-learning)
*   [Watkins, 1989: Learning from Delayed Rewards (Q-learning 原始论文)](http://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf)

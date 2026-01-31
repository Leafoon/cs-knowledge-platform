---
title: "第3章：蒙特卡洛方法 (Monte Carlo Methods)"
description: "无模型学习的第一步：从经验中学习价值，探索与利用的平衡，以及 On-policy 与 Off-policy 的殊途同归"
date: "2026-01-30"
---

# 第3章：蒙特卡洛方法 (Monte Carlo Methods)

在上一章的动态规划（DP）中，我们假设拥有完美的“上帝视角”（即已知的环境模型 $P(s'|s,a)$ 和 $R(s,a)$）。然而，在现实世界中，我们往往无法预知环境的运作规律（比如自动驾驶、股票市场或复杂游戏）。

本章我们将进入 **Model-free（无模型）** 强化学习的世界。我们将从最直观的方法开始：**蒙特卡洛方法（Monte Carlo, MC）**。

简单来说，MC 方法就是：**不问前程，只看结果**。Agent 通过不断地与环境交互，生成完整的实验轨迹（Episode），然后根据最终的收益来反向推导状态的价值。

---

## 3.1 蒙特卡洛预测 (MC Prediction)

**目标**：给定一个策略 $\pi$，估算其状态价值函数 $V^\pi(s)$。

### 3.1.1 核心思想：用平均回报代替期望回报

根据定义，状态价值是期望回报：
$$ V^\pi(s) = \mathbb{E}_\pi [G_t | S_t = s] $$

在没有模型的情况下，我们无法直接计算期望（积分或求和）。但根据大数定律（Law of Large Numbers），只要采样次数足够多，**样本均值**就会收敛到**总体期望**。

这就好比计算圆周率 $\pi$：你无需知道圆的方程，只需随机向正方形内撒豆子，统计落在圆内的比例即可。

<div data-component="MCReturnEstimation"></div>

### 3.1.2 算法实现

MC 预测通常有两种变体：

1.  **First-visit MC**：在一个 Episode 中，只计算状态 $s$ **第一次**出现时的回报。
    *   **优点**：每个 Episode 为每个状态提供独立同分布（i.i.d）的样本，理论性质好。
    *   **常用**：大多数情况下使用此版本。
2.  **Every-visit MC**：在一个 Episode 中，每次经过状态 $s$ 都计算并累加。

**First-visit MC 伪代码**：

```
输入: 策略 π
初始化: V(s) 任意值, Returns(s) 空列表

循环 (针对每个 Episode):
    1. 使用策略 π 生成轨迹: S0, A0, R1, S1, A1, R2, ..., ST
    2. G ← 0
    3. 从后向前遍历 t = T-1, T-2, ..., 0:
        G ← γG + R_{t+1}
        除非 St 在 S0...S_{t-1} 中出现过 (First-visit check):
            将 G 添加到 Returns(St)
            V(St) ← average(Returns(St))
```

### 3.1.3 增量式更新 (Incremental Implementation)

我们不需要存储所有的 $G$，可以使用增量平均公式（类似梯度下降）：

$$ V(S_t) \leftarrow V(S_t) + \frac{1}{N(S_t)} (G_t - V(S_t)) $$

或者使用固定步长 $\alpha$（即使是非平稳问题也能适应）：

$$ V(S_t) \leftarrow V(S_t) + \alpha (G_t - V(S_t)) $$

这种形式揭示了 RL 学习的核心模式：
$$ \text{NewValue} \leftarrow \text{OldValue} + \text{StepSize} \times (\text{Target} - \text{OldValue}) $$
其中 $G_t$ 就是我们的 **Target（目标）**。

---

## 3.2 蒙特卡洛控制 (MC Control)

**目标**：找到最优策略 $\pi^*$。

我们沿用 DP 中的 **GPI（广义策略迭代）** 框架：
1.  **评估（Evaluation）**：使用 MC 方法估计 $Q^\pi(s,a)$（注意：Model-free 必须估计 $Q$ 而不仅仅是 $V$，因为没有模型无法从 $V$ 推导策略）。
2.  **改进（Improvement）**：贪心更新策略 $\pi(s) = \arg\max_a Q(s,a)$。

### 3.2.1 探索与利用的困境

如果我们在改进策略时直接使用完全贪心策略（Deterministic Greedy），Agent 可能会陷入局部最优，并不再探索其他可能的动作。例如，开场走了左边得到 +1，走了右边得到 0，以后就永远只走左边，错过了右边深处的 +100。

**解决方案 1：Exploring Starts（探索性出发）**
*   假设所有 $(s,a)$ 对都有非零概率作为 Episode 的起点。
*   **局限**：在现实环境中很难实现（你很难让自动驾驶汽车直接从“如果不踩刹车会怎样”的状态开始）。

**解决方案 2：$\epsilon$-Greedy 策略**
*   以 $1-\epsilon$ 的概率选择贪心动作。
*   以 $\epsilon$ 的概率随机选择动作。
*   保证所有动作都有 $\frac{\epsilon}{|A|}$ 的概率被选中，确保持续探索。

### 3.2.2 21点 (Blackjack) 实战

Blackjack 是经典的 MC 学习案例。
*   **状态**：(玩家点数 [12-21], 庄家明牌 [A-10], 玩家是否有可用 Ace [True/False])。
*   **动作**：Hit (要牌), Stick (停牌)。
*   **奖励**：赢 +1, 输 -1, 平 0。

```python
import gymnasium as gym
import numpy as np
from collections import defaultdict

def run_mc_control(num_episodes=50000, epsilon=0.1, gamma=1.0):
    env = gym.make('Blackjack-v1', sab=True) # sab=True遵循Sutton书规则
    
    # Q 表: 字典映射 state -> action_values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # 计数: N(s, a)
    Returns_count = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # 当前策略: epsilon-greedy
    def policy(state):
        if np.random.random() < epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(Q[state])

    for i in range(num_episodes):
        # 1. 生成 Episode
        episode = []
        state, _ = env.reset()
        done = False
        while not done:
            action = policy(state)
            next_state, reward, done, _, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            
        # 2. 从后向前计算 G 并更新 Q
        G = 0
        visited_sa = set()
        for t in range(len(episode)-1, -1, -1):
            s, a, r = episode[t]
            G = gamma * G + r
            
            # First-visit MC
            if (s, a) not in visited_sa:
                visited_sa.add((s, a))
                Returns_count[s][a] += 1
                # 增量更新均值
                Q[s][a] += (G - Q[s][a]) / Returns_count[s][a]
    
    return Q

# 运行训练
Q_table = run_mc_control()

# 简单的策略展示
def print_policy_summary(Q):
    print("Strategy Learnt (Subset):")
    # 挑选几个典型状态展示
    test_states = [
        (20, 10, False), # 玩家20点，庄家10点，无Ace -> 应该停牌
        (10, 10, False), # 玩家10点，庄家10点，无Ace -> 应该要牌(期望到20)
    ]
    for s in test_states:
        best_a = "Stick" if np.argmax(Q[s]) == 0 else "Hit"
        print(f"State {s}: Best Action = {best_a}")

print_policy_summary(Q_table)
```

---

## 3.3 On-policy vs Off-policy

在强化学习中，我们要处理两个不同的策略概念：
1.  **Target Policy ($\pi$)**：我们想要学习和评估的策略（通常是最优策略）。
2.  **Behavior Policy ($b$)**：用来与环境交互、产生数据的策略（通常具有探索性）。

<div data-component="OnPolicyVsOffPolicy"></div>

### 3.3.1 核心区别

*   **On-policy (同策略)**：$\pi = b$。
    *   “一边学，一边做”。
    *   例如 SARSA, $\epsilon$-greedy MC。
    *   **优点**：简单。
    *   **缺点**：为了探索，Behavior Policy 必须是软策略（如 $\epsilon$-greedy），导致我们也只能学到一个“非最优”的软策略（即便学会了，执行时还要保留 $\epsilon$ 的随机性）。

*   **Off-policy (异策略)**：$\pi \ne b$。
    *   “看着别人（或过去的自己）做，自己学”。
    *   例如 Q-learning, DQN, Off-policy MC。
    *   **优点**：
        1.  可以学习最优确定性策略 $\pi^*$，同时保持探索性行为 $b$。
        2.  可以从人类演示或其他 Agent 的经验中学习。
    *   **缺点**：方差大，收敛慢，理论复杂。

---

## 3.4 重要性采样 (Importance Sampling)

Off-policy MC 面临一个数学挑战：我们想估计 $\mathbb{E}_\pi [G_t]$，但我们的数据是从分布 $b$ 中采样的。

直接求平均是错误的，这就像用“掷骰子”的数据来估计“扔硬币”的期望。我们需要修正这个分布偏差，使用 **重要性采样比率 (Importance Sampling Ratio)**：

$$ \rho_{t:T-1} = \frac{\prod_{k=t}^{T-1} \pi(A_k|S_k) P(S_{k+1}|S_k,A_k)}{\prod_{k=t}^{T-1} b(A_k|S_k) P(S_{k+1}|S_k,A_k)} = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)} $$

**神奇之处**：环境转换概率 $P$ 在分子分母中**消掉了**！这意味着我们**不需要由模型**就能进行 Off-policy 学习，只需要知道两个策略的概率比。

<div data-component="ImportanceSamplingVisualizer"></div>

### 3.4.1 普通重要性采样 (Ordinary IS) vs 加权重要性采样 (Weighted IS)

$$ V(s) = \frac{\sum \rho G}{\sum 1} \quad \text{vs} \quad V(s) = \frac{\sum \rho G}{\sum \rho} $$

*   **Ordinary IS**：无偏估计，但方差可能无穷大（Unlimited Variance）。
*   **Weighted IS**：有偏估计（Bias），但方差有界且较小，实际上更常用。

尽管 Off-policy MC 理论上很美，但在长 Episode 任务中实用性极差（因为 $\rho$ 是连乘积，容易爆炸或消失）。这也是为什么 Q-learning (TD 方法) 在 Off-policy 场景下占据统治地位的原因。

---

## 本章总结

1.  **MC 方法**：直接从完整经验中学习，$V(s) \approx \text{Average}(G_t)$。不需要环境模型。
2.  **优势**：无偏（unbiased），能处理非马尔可夫环境（只看结果）。
3.  **劣势**：方差高（High Variance），必须等待 Episode 结束（无法处理连续任务），只能从完成的轨迹中学习。
4.  **Off-policy**：通过重要性采样，让策略 $\pi$ 从策略 $b$ 产生的数据中学习。

**下一章预告**：
如果我们不想等到 Episode 结束才学习怎么办？比如在自动驾驶中，撞车前一秒就应该意识到危险，而不是等到撞车后才更新。下一章我们将介绍强化学习中最核心的思想——**时序差分学习 (Temporal Difference Learning, TD)**，它结合了 MC 的“采样”和 DP 的“自举”优点。

---

## 扩展阅读

*   Sutton & Barto, *Reinforcement Learning: An Introduction*, Chapter 5
*   [UCL RL Course Lecture 4: Model-Free Prediction](https://www.davidsilver.uk/wp-content/uploads/2020/03/MC-TD.pdf)

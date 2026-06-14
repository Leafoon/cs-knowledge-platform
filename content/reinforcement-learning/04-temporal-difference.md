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

让我们通过对比来理解 TD 的独特之处。

**蒙特卡洛（MC）更新**

MC 方法必须等到 Episode 结束，获得完整的回报 $G_t$：

$$
V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]
$$

**符号详解**：
- $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$：从时刻 $t$ 开始的**实际回报**
- $\alpha$：学习率
- $G_t - V(S_t)$：**误差**（实际回报与估计值的差距）

**特点**：
- ✅ 无偏估计（$G_t$ 是真实回报的样本）
- ❌ 高方差（$G_t$ 受很多随机因素影响）
- ❌ 必须等到 Episode 结束
- ❌ 无法用于连续任务

**时序差分 TD(0) 更新**

TD 方法只需走一步，立即更新：

$$
V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]
$$

**符号详解**：
- $R_{t+1}$：执行动作后获得的**即时奖励**（真实观察）
- $\gamma$：折扣因子
- $V(S_{t+1})$：下一个状态的**估计价值**（不是真实值！）
- $R_{t+1} + \gamma V(S_{t+1})$：**TD 目标**（TD Target）
- $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$：**TD 误差**（TD Error）

**特点**：
- ✅ 在线学习（每步都可以更新）
- ✅ 低方差（只依赖一步随机性）
- ✅ 适用于连续任务
- ❌ 有偏估计（使用估计值 $V(S_{t+1})$）

**数值例子**：

假设：
- 当前状态 $S_t$ 的估计价值：$V(S_t) = 5$
- 执行动作后获得奖励：$R_{t+1} = 2$
- 下一个状态 $S_{t+1}$ 的估计价值：$V(S_{t+1}) = 8$
- 折扣因子：$\gamma = 0.9$
- 学习率：$\alpha = 0.1$

计算 TD 目标：
$$
\text{TD Target} = R_{t+1} + \gamma V(S_{t+1}) = 2 + 0.9 \times 8 = 2 + 7.2 = 9.2
$$

计算 TD 误差：
$$
\delta_t = 9.2 - 5 = 4.2
$$

更新价值函数：
$$
V(S_t) \leftarrow 5 + 0.1 \times 4.2 = 5 + 0.42 = 5.42
$$

**核心思想：Bootstrapping (自举)**

TD 方法用一个**估计值**（$V(S_{t+1})$）来更新另一个**估计值**（$V(S_t)$）。这听起来像是在“左脚踩右脚上天”，但实际上非常有效！

**什么是自举？**

自举（Bootstrapping）是指用一个**估计值**来更新另一个**估计值**。

在 TD 学习中：
- 我们用 $V(S_{t+1})$（估计值）来更新 $V(S_t)$（估计值）
- 这就像“左脚踩右脚上天”——听起来不靠谱，但实际上非常有效！

**为什么自举有效？**

**数学角度**：TD 更新实际上是在进行 Bellman 方程的迭代求解。

回顾 Bellman 方程：
$$
V^\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma V^\pi(S_{t+1}) | S_t = s]
$$

TD 更新可以看作：
1. 用样本 $(S_t, R_{t+1}, S_{t+1})$ 估计期望
2. 用当前估计 $V(S_{t+1})$ 代替真实值 $V^\pi(S_{t+1})$
3. 逐步逼近真实的 $V^\pi$

**直观角度**：

想象你在学习一门课程：
- **MC 方法**：等到期末考试，根据最终成绩评估整个学期的学习效果
- **TD 方法**：每次小测验后，根据成绩和对未来的预期调整学习策略

TD 方法能更快地发现问题并调整。

**自举的优缺点**：

**优点**：
- 低方差：只依赖一步随机性
- 在线学习：不需要等待 Episode 结束
- 适用于连续任务

**缺点**：
- 有偏：初始估计不准确时，误差会传播
- 收敛速度可能受初始值影响

<div data-component="TDUpdateVisualizer"></div>

### 4.1.2 TD 的优势

**1. 在线学习（Online Learning）**

- 可以在 Episode 进行中持续学习
- 不需要等待结束
- 对长任务或无尽任务至关重要

**2. 方差减少（Variance Reduction）**

- $G_t$ 取决于未来所有随机结果，方差很大
- TD Target 只取决于一步随机性和 $V(S_{t+1})$
- 方差通常更小，收敛更快

**3. 无需模型（Model-free）**

- 不需要环境模型 $\mathcal{P}$ 和 $\mathcal{R}$
- 只需要与环境交互

**4. 计算效率**

- 每步更新，不需要存储整个 Episode
- 内存需求低

---

### 4.1.3 TD(0) 算法

**完整算法（伪代码）**：

```
算法：TD(0) 预测
输入：策略 π，学习率 α，折扣因子 γ
输出：状态价值函数 V ≈ V^π

1. 初始化：
   对所有 s ∈ S:
       V(s) ← 任意值（通常为0）

2. 循环（对每个 Episode）：
   初始化状态 S
   
   循环（对 Episode 中的每一步）：
       a) 使用策略 π 选择动作 A
       b) 执行 A，观察奖励 R 和下一个状态 S'
       c) 更新价值函数：
          V(S) ← V(S) + α[R + γ·V(S') - V(S)]
       d) S ← S'
   
   直到 S 是终止状态

3. 返回 V
```

**关键步骤详解**：

**步骤 c：更新价值函数**

这是 TD 的核心。让我们逐步分解：

1. 计算 TD 目标：
   $$
   \text{Target} = R + \gamma \cdot V(S')
   $$

2. 计算 TD 误差：
   $$
   \delta = \text{Target} - V(S) = R + \gamma \cdot V(S') - V(S)
   $$

3. 更新估计：
   $$
   V(S) \leftarrow V(S) + \alpha \cdot \delta
   $$

**学习率 $\alpha$ 的作用**：

- $\alpha = 0$：不学习（$V$ 不变）
- $\alpha = 1$：完全相信新信息（$V(S) = \text{Target}$）
- $\alpha = 0.1$：保守学习（慢慢调整）

---

### 4.1.4 收敛性保证

**定理**：TD(0) 在表格表示下收敛到 $V^\pi$

**条件**：
1. 策略 $\pi$ 固定
2. 所有状态被访问无穷多次
3. 学习率满足 Robbins-Monro 条件：
   $$
   \sum_{t=1}^{\infty} \alpha_t = \infty, \quad \sum_{t=1}^{\infty} \alpha_t^2 < \infty
   $$

**Robbins-Monro 条件解释**：

**第一个条件**：$\sum_{t=1}^{\infty} \alpha_t = \infty$
- 学习率的总和必须是无穷大
- 保证能够克服任何初始条件或暂时的不良估计
- 例如：$\alpha_t = \frac{1}{t}$ 满足（调和级数发散）

**第二个条件**：$\sum_{t=1}^{\infty} \alpha_t^2 < \infty$
- 学习率平方的总和必须收敛
- 保证噪声的影响最终会消失
- 例如：$\alpha_t = \frac{1}{t}$ 满足（$\sum \frac{1}{t^2}$ 收敛）

**常见学习率**：
- $\alpha_t = \frac{1}{t}$：满足条件，理论保证收敛
- $\alpha_t = \frac{1}{1+\text{visit}(s)}$：状态特定的学习率
- $\alpha_t = 0.1$（常数）：不满足条件，但实践中效果好

**收敛速度**：

- **MC**：$O(1/\sqrt{n})$（$n$ 是样本数）
- **TD**：通常更快（低方差），但理论分析复杂

---

### 4.1.5 完整代码实现

```python
import numpy as np
import gymnasium as gym
from typing import Dict, List, Callable
import matplotlib.pyplot as plt
from collections import defaultdict

class TDPredictor:
    """TD(0) 预测器"""
    
    def __init__(self, gamma: float = 0.99, alpha: float = 0.1):
        """
        Args:
            gamma: 折扣因子
            alpha: 学习率
        """
        self.gamma = gamma
        self.alpha = alpha
        self.V = defaultdict(float)  # 状态价值函数
        self.visit_counts = defaultdict(int)  # 访问次数
        
    def update(self, state, reward: float, next_state, done: bool):
        """
        TD(0) 更新
        
        详细步骤：
        1. 计算 TD 目标：R + γ·V(S')
        2. 计算 TD 误差：δ = Target - V(S)
        3. 更新：V(S) ← V(S) + α·δ
        
        Args:
            state: 当前状态
            reward: 即时奖励
            next_state: 下一个状态
            done: 是否终止
        """
        # 计算 TD 目标
        if done:
            # 终止状态的价值为 0
            td_target = reward
        else:
            td_target = reward + self.gamma * self.V[next_state]
        
        # 计算 TD 误差
        td_error = td_target - self.V[state]
        
        # 更新价值函数
        self.V[state] += self.alpha * td_error
        
        # 记录访问
        self.visit_counts[state] += 1
        
        return td_error
    
    def predict(self, env, policy: Callable, num_episodes: int = 1000) -> Dict:
        """
        运行 TD(0) 预测
        
        Args:
            env: 环境
            policy: 策略函数 policy(state) -> action
            num_episodes: Episode 数量
            
        Returns:
            info: 包含学习曲线等信息的字典
        """
        td_errors_history = []
        value_history = []
        
        for episode_num in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_td_errors = []
            
            while not done:
                # 选择动作
                action = policy(state)
                
                # 执行动作
                next_state, reward, done, truncated, _ = env.step(action)
                
                # TD 更新
                td_error = self.update(state, reward, next_state, done or truncated)
                episode_td_errors.append(abs(td_error))
                
                state = next_state
                
                if truncated:
                    break
            
            # 记录统计信息
            avg_td_error = np.mean(episode_td_errors) if episode_td_errors else 0
            td_errors_history.append(avg_td_error)
            
            if episode_num % 100 == 0:
                value_history.append(dict(self.V))
        
        return {
            'V': dict(self.V),
            'visit_counts': dict(self.visit_counts),
            'td_errors': td_errors_history,
            'value_history': value_history
        }
    
    def visualize_convergence(self, td_errors: List[float], 
                             value_history: List[Dict]):
        """可视化收敛过程"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # TD 误差收敛
        axes[0].plot(td_errors, alpha=0.7, linewidth=1)
        
        # 计算移动平均
        window = 50
        if len(td_errors) >= window:
            moving_avg = np.convolve(td_errors, 
                                    np.ones(window)/window, 
                                    mode='valid')
            axes[0].plot(range(window-1, len(td_errors)), 
                        moving_avg, 
                        color='red', 
                        linewidth=2, 
                        label=f'{window}-Episode Moving Avg')
        
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Average |TD Error|')
        axes[0].set_title('TD Error Convergence')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 价值函数演化
        if value_history:
            states = list(value_history[0].keys())[:5]
            for state in states:
                values = [vh.get(state, 0) for vh in value_history]
                episodes = [i * 100 for i in range(len(values))]
                axes[1].plot(episodes, values, 
                           label=f'State {state}', 
                           linewidth=2, 
                           marker='o', 
                           markersize=4)
            
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('Estimated Value')
            axes[1].set_title('Value Function Evolution')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# 使用示例
if __name__ == "__main__":
    # 创建环境
    env = gym.make('FrozenLake-v1', is_slippery=False)
    
    # 定义策略（随机策略）
    def random_policy(state):
        return env.action_space.sample()
    
    # 创建 TD 预测器
    predictor = TDPredictor(gamma=0.99, alpha=0.1)
    
    # 运行预测
    print("Running TD(0) Prediction...")
    info = predictor.predict(env, random_policy, num_episodes=1000)
    
    # 打印结果
    print("\nEstimated State Values:")
    for state, value in sorted(info['V'].items()):
        print(f"State {state}: V = {value:.4f}, "
              f"Visits = {info['visit_counts'][state]}")
    
    # 可视化收敛过程
    predictor.visualize_convergence(info['td_errors'], 
                                    info['value_history'])
```

---

### 4.1.6 n-步 TD 方法

TD(0) 只向前看一步。我们可以推广到 **n-步 TD**，在 MC 和 TD(0) 之间找到平衡。

**n-步回报**：

$$
G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})
$$

**更新规则**：

$$
V(S_t) \leftarrow V(S_t) + \alpha [G_t^{(n)} - V(S_t)]
$$

**特殊情况**：
- $n=1$：TD(0)
- $n=\infty$：Monte Carlo

**权衡**：
- **小 n**：低方差，高偏差，快速更新
- **大 n**：高方差，低偏差，接近 MC

**实践建议**：
- 通常 $n=3$ 到 $n=5$ 效果较好
- 需要根据具体任务调整

---

### 4.1.7 小结

TD 预测的核心要点：

1. **核心思想**：用估计值更新估计值（自举）
2. **TD 目标**：$R_{t+1} + \gamma V(S_{t+1})$
3. **TD 误差**：$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$
4. **偏差-方差**：有偏但低方差，通常比 MC 收敛更快
5. **在线学习**：每步都可以更新，适用于连续任务
6. **收敛保证**：在 Robbins-Monro 条件下收敛

下一节我们将学习如何使用 TD 方法来寻找最优策略（TD 控制）。

---

## 4.2 SARSA: On-policy TD Control

**目标**：找到最优策略 $\pi^*$。

我们将 TD 思想应用到控制问题（寻找最优策略），即学习 $Q(s,a)$。

---

### 4.2.1 为什么要估计 Q 函数？

**关键问题**：在无模型（model-free）环境中，我们必须估计 **动作价值函数 $Q(s,a)$** 而不仅仅是状态价值函数 $V(s)$。

**原因详解**：

在动态规划中，我们可以从 $V(s)$ 推导出最优策略：

$$
\pi^*(s) = \arg\max_a \sum_{s'} \mathcal{P}(s'|s,a) \left[ \mathcal{R}(s,a) + \gamma V^*(s') \right]
$$

但这需要知道：
- $\mathcal{P}(s'|s,a)$：转移概率（环境模型）
- $\mathcal{R}(s,a)$：奖励函数

**在无模型情况下**，我们没有 $\mathcal{P}$ 和 $\mathcal{R}$，所以无法从 $V$ 计算出策略！

**解决方案**：直接估计 $Q(s,a)$

$$
Q^\pi(s,a) = \mathbb{E}_\pi [G_t | S_t = s, A_t = a]
$$

有了 $Q(s,a)$，策略改进变得简单：

$$
\pi'(s) = \arg\max_a Q(s,a)
$$

---

### 4.2.2 SARSA 更新公式

SARSA 的名字来源于其更新所使用的五元组：$(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$。

**更新规则**：

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
$$

**符号详解**：
- $S_t$：当前状态
- $A_t$：当前动作（由策略 $\pi$ 选择）
- $R_{t+1}$：执行 $A_t$ 后获得的即时奖励
- $S_{t+1}$：下一个状态
- $A_{t+1}$：在 $S_{t+1}$ 下由策略 $\pi$ 选择的下一个动作
- $\alpha$：学习率
- $\gamma$：折扣因子

**SARSA Target**：
$$
\text{SARSA Target} = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})
$$

**SARSA Error**：
$$
\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)
$$

**数值例子**：

假设：
- 当前状态-动作对：$(S_t, A_t)$，$Q(S_t, A_t) = 3$
- 执行 $A_t$ 后获得奖励：$R_{t+1} = 1$
- 下一个状态-动作对：$(S_{t+1}, A_{t+1})$，$Q(S_{t+1}, A_{t+1}) = 5$
- $\gamma = 0.9$，$\alpha = 0.1$

SARSA Target：
$$
\text{Target} = 1 + 0.9 \times 5 = 1 + 4.5 = 5.5
$$

SARSA Error：
$$
\delta_t = 5.5 - 3 = 2.5
$$

更新：
$$
Q(S_t, A_t) \leftarrow 3 + 0.1 \times 2.5 = 3 + 0.25 = 3.25
$$

---

### 4.2.3 SARSA 算法流程

**完整算法（伪代码）**：

```
算法：SARSA
输入：学习率 α，折扣因子 γ，探索率 ε
输出：近似最优策略 π ≈ π*

1. 初始化：
   对所有 s ∈ S, a ∈ A:
       Q(s,a) ← 任意值

2. 循环（对每个 Episode）：
   初始化状态 S
   使用 ε-greedy 策略选择动作 A  // 关键：先选择 A
   
   循环（对 Episode 中的每一步）：
       a) 执行动作 A，观察 R 和 S'
       
       b) 使用 ε-greedy 策略选择 A'  // 关键：再选择 A'
       
       c) 更新 Q 值：
          Q(S, A) ← Q(S, A) + α[R + γ·Q(S', A') - Q(S, A)]
       
       d) S ← S'
          A ← A'  // 关键：使用已选择的 A'
   
   直到 S 是终止状态

3. 返回 π（基于 Q 的贪心策略）
```

**关键步骤详解**：

1. **先选择 A**：在 Episode 开始时，先用 ε-greedy 选择第一个动作

2. **再选择 A'**：执行动作后，在新状态 S' 下再次用 ε-greedy 选择 A'

3. **使用 Q(S', A') 更新**：使用已经选择的 A' 的 Q 值来更新

4. **A ← A'**：下一步直接使用已选择的 A'，不需要重新选择

---

### 4.2.4 On-policy 特性

由于所有的动作（包括更新目标 $A'$）都是由当前策略（Behavior Policy）生成的，所以 SARSA 是 **On-policy** 算法。

**On-policy 的含义**：
- **生成数据的策略** = **被评估和改进的策略**
- 两者是同一个策略 $\pi$

**实践影响**：
- SARSA 学到的是当前策略（包括探索）的价值
- 如果使用 ε-greedy，学到的是“带有探索的策略”的价值
- 因此 SARSA 通常更加**保守**（避免风险）

---

## 4.3 Q-Learning: Off-policy TD Control

Q-Learning 是 RL 历史上最重要的算法之一（Watkins, 1989）。它的核心突破在于：**学习的目标动作可以与实际执行的动作不同**。

---

### 4.3.1 Q-Learning 更新公式

**更新规则**：

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]
$$

**符号详解**：
- $S_t$：当前状态
- $A_t$：当前动作（可能是探索性动作）
- $R_{t+1}$：执行 $A_t$ 后获得的奖励
- $S_{t+1}$：下一个状态
- $\max_a Q(S_{t+1}, a)$：在 $S_{t+1}$ 下所有动作中 Q 值最大的那个
- $\alpha$：学习率
- $\gamma$：折扣因子

**Q-Learning Target**：
$$
\text{Q-Learning Target} = R_{t+1} + \gamma \max_a Q(S_{t+1}, a)
$$

**关键区别**：
- **SARSA**：使用 $Q(S_{t+1}, A_{t+1})$，其中 $A_{t+1}$ 是由策略选择的
- **Q-Learning**：使用 $\max_a Q(S_{t+1}, a)$，直接取最大值

**数值例子**：

假设：
- 当前状态-动作对：$(S_t, A_t)$，$Q(S_t, A_t) = 3$
- 执行 $A_t$ 后获得奖励：$R_{t+1} = 1$
- 下一个状态 $S_{t+1}$ 有 3 个动作：
  - $Q(S_{t+1}, a_1) = 4$
  - $Q(S_{t+1}, a_2) = 6$  ← 最大
  - $Q(S_{t+1}, a_3) = 2$
- $\gamma = 0.9$，$\alpha = 0.1$

Q-Learning Target：
$$
\text{Target} = 1 + 0.9 \times \max(4, 6, 2) = 1 + 0.9 \times 6 = 1 + 5.4 = 6.4
$$

Q-Learning Error：
$$
\delta_t = 6.4 - 3 = 3.4
$$

更新：
$$
Q(S_t, A_t) \leftarrow 3 + 0.1 \times 3.4 = 3 + 0.34 = 3.34
$$

---

### 4.3.2 Off-policy 特性

**Off-policy 的含义**：
- **生成数据的策略**（Behavior Policy）≠ **被评估和改进的策略**（Target Policy）

在 Q-Learning 中：
- **Behavior Policy**：$\epsilon$-greedy（用于探索）
- **Target Policy**：完全贪心（$\pi(s) = \arg\max_a Q(s,a)$）

**为什么直接逼近 $Q^*$？**

因为 Q-Learning 的更新目标是：
$$
R_{t+1} + \gamma \max_a Q(S_{t+1}, a)
$$

这直接对应于 Bellman 最优方程：
$$
Q^*(s,a) = \mathbb{E}[R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a') | S_t = s, A_t = a]
$$

因此，Q-Learning 直接学习最优 Q 函数，而不受当前探索策略的影响（只要探索得足够充分）。

---

### 4.3.3 SARSA vs Q-Learning 深度对比

| 特性 | SARSA | Q-Learning |
|:---|:---|:---|
| **类型** | On-policy | Off-policy |
| **更新目标** | $R + \gamma Q(S', A')$ | $R + \gamma \max_a Q(S', a)$ |
| **下一动作** | 由策略选择 $A'$ | 取最大 Q 值 | 
| **学到的是** | 当前策略的价值 | 最优策略的价值 |
| **风格** | 保守（考虑探索风险） | 激进（直接学最优） |
| **适用场景** | 安全关键的任务 | 一般任务 |

**直观理解**：

- **SARSA**：“我学的是我实际会做的事情”
  - 如果策略会探索，就学会避免探索时的风险
  
- **Q-Learning**：“我学的是最优的做法，不管我现在怎么做”
  - 即使现在在探索，也学习最优路径

---

### 4.3.4 Cliff Walking 详细分析

在经典的“悬崖行走”环境中，我们可以清晰地看到两者的区别。

**环境设置**：
- **任务**：从起点走到终点，尽量快（每步 -1）
- **悬崖**：掉下去会扣 -100 并回到起点
- **地图**：
```
[S] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [ ] [G]
[ ] [C] [C] [C] [C] [C] [C] [C] [C] [C] [C] [ ]
```
S = 起点, G = 终点, C = 悬崖

<div data-component="SARSAvsQLearning"></div>

**Q-Learning 的表现**：
- **学到的路径**：紧贴悬崖边走（最短路径）
- **原因**：它学习的是 $\max_a Q(s,a)$，即最优动作
- **问题**：训练期间经常掉下悬崖（因为 $\epsilon$-greedy 探索）
- **最终策略**：完全贪心时安全（不探索）

**SARSA 的表现**：
- **学到的路径**：远离悬崖走（安全路径）
- **原因**：它学习的是 $Q(s, A')$，其中 $A'$ 由 $\epsilon$-greedy 选择
- **逻辑**：“如果我走悬崖边，一旦随机探索，就会掉下去”
- **结果**：学会避开风险区域

**数值对比**：

假设在悬崖边的状态 $s$：
- 向右走（最优）：期望回报 = -13（到终点 13 步）
- 但有 $\epsilon = 0.1$ 的概率掉下悬崖：-100

**Q-Learning**：
$$
Q(s, \text{right}) \approx -13 \quad \text{(只考虑最优动作)}
$$

**SARSA**：
$$
Q(s, \text{right}) \approx 0.9 \times (-13) + 0.1 \times (-100) = -11.7 - 10 = -21.7
$$

SARSA 的 Q 值更低，因为它考虑了探索风险！

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

---

### 4.4.1 什么是最大化偏差？

回顾 Q-Learning 更新公式：

$$
\text{Target} = R + \gamma \max_a Q(S', a)
$$

我们在**同一个样本集**上既做**估计**（Estimate）又做**最大化**（Maximization）。

**数学不等式**：

$$
\mathbb{E}[\max(X)] \ge \max(\mathbb{E}[X])
$$

**含义**：
- **左边**：先取最大值，再求期望
- **右边**：先求期望，再取最大值

由于 $\mathbb{E}[\max(X)] \ge \max(\mathbb{E}[X])$，如果 Q 值的估计存在噪声（这在初期是必然的），$\max$ 操作会倾向于**高估**那些仅仅是因为运气好而被高估的动作。

**数值例子**：

假设状态 $s$ 有 3 个动作，真实 Q 值都是 0：
- $Q^*(s, a_1) = 0$
- $Q^*(s, a_2) = 0$
- $Q^*(s, a_3) = 0$

但由于噪声，我们的估计是：
- $Q(s, a_1) = -0.5$
- $Q(s, a_2) = +0.8$  ← 运气好，被高估
- $Q(s, a_3) = -0.3$

Q-Learning 会选择：
$$
\max_a Q(s, a) = Q(s, a_2) = 0.8
$$

但真实的最大值应该是 0！这就是**高估**。

<div data-component="MaximizationBiasDemo"></div>

---

### 4.4.2 Double Q-Learning

为了解决这个问题，Hasselt (2010) 提出了 Double Q-Learning。

**核心思想**：使用两个独立的 Q 网络（或 Q 表）$Q_1$ 和 $Q_2$。

**关键原则**：
- 用 $Q_1$ 来**选择**动作（argmax）
- 用 $Q_2$ 来**评估**动作（value）

**更新规则**：

以概率 0.5 选择更新 $Q_1$ 或 $Q_2$：

**更新 $Q_1$**：
$$
a^* = \arg\max_a Q_1(S', a) \quad \text{(用 } Q_1 \text{ 选择)}
$$
$$
\text{Target} = R + \gamma Q_2(S', a^*) \quad \text{(用 } Q_2 \text{ 评估)}
$$
$$
Q_1(S, A) \leftarrow Q_1(S, A) + \alpha [\text{Target} - Q_1(S, A)]
$$

**更新 $Q_2$**：
$$
a^* = \arg\max_a Q_2(S', a) \quad \text{(用 } Q_2 \text{ 选择)}
$$
$$
\text{Target} = R + \gamma Q_1(S', a^*) \quad \text{(用 } Q_1 \text{ 评估)}
$$
$$
Q_2(S, A) \leftarrow Q_2(S, A) + \alpha [\text{Target} - Q_2(S, A)]
$$

**为什么能减少偏差？**

即使 $Q_1$ 高估了某个动作，$Q_2$（独立分布）也不太可能同时高估它，从而抵消了偏差。

**数值例子**：

回到之前的例子，真实 Q 值都是 0：

**Q-Learning**：
- $Q(s, a_1) = -0.5$
- $Q(s, a_2) = +0.8$  ← 选择并评估
- $Q(s, a_3) = -0.3$
- Target = $0.8$ （高估）

**Double Q-Learning**：
- $Q_1(s, a_1) = -0.5$
- $Q_1(s, a_2) = +0.8$  ← 用 $Q_1$ 选择
- $Q_1(s, a_3) = -0.3$

- $Q_2(s, a_1) = +0.2$
- $Q_2(s, a_2) = -0.1$  ← 用 $Q_2$ 评估
- $Q_2(s, a_3) = +0.3$

选择：$a^* = \arg\max_a Q_1(s, a) = a_2$

评估：$Q_2(s, a_2) = -0.1$

Target = $-0.1$ （更接近真实值 0）

---

### 4.4.3 实践影响

**Double Q-Learning 的优势**：
- 减少最大化偏差
- 更稳定的学习
- 更准确的 Q 值估计

**扩展到 Deep RL**：

这个思想后来被 DeepMind 用到了 DQN 上（**Double DQN**），成为标配。

在 Double DQN 中：
- 用当前网络选择动作
- 用目标网络评估动作

这大大提高了 DQN 的性能和稳定性。

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

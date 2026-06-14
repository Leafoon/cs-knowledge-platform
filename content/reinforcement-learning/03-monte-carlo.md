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

蒙特卡洛方法是强化学习中最直观的学习方法之一。它的核心思想非常简单：**通过实际经验来估计价值**。就像你想知道一家餐厅好不好，最直接的方法就是去吃几次，然后根据这几次的体验来判断。

---

### 3.1.1 核心思想：用样本均值代替期望

**回顾：状态价值的定义**

根据第1章的定义，状态价值函数是从状态 $s$ 开始，遵循策略 $\pi$ 所能获得的**期望回报**：

$$
V^\pi(s) = \mathbb{E}_\pi [G_t | S_t = s]
$$

让我们逐个符号解释：
- $V^\pi(s)$：在策略 $\pi$ 下，状态 $s$ 的价值
- $\mathbb{E}_\pi[\cdot]$：期望（Expected value），下标 $\pi$ 表示遵循策略 $\pi$
- $G_t$：从时刻 $t$ 开始的**回报**（Return）
- $S_t = s$：条件，表示在时刻 $t$ 处于状态 $s$

**什么是回报 $G_t$？**

回报是从时刻 $t$ 开始，未来所有奖励的折扣总和：

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

**符号详解**：
- $R_{t+1}$：在时刻 $t$ 执行动作后，在时刻 $t+1$ 获得的即时奖励
- $\gamma$：折扣因子（$0 \leq \gamma \leq 1$），控制未来奖励的重要性
- $\gamma^k$：$k$ 步之后的奖励会被折扣 $\gamma^k$ 倍

**直观理解**：
- 如果 $\gamma = 0$：只关心即时奖励 $R_{t+1}$
- 如果 $\gamma = 1$：所有未来奖励同等重要（无折扣）
- 如果 $\gamma = 0.9$：1步后的奖励打9折，2步后打81折，3步后打72.9折...

**例子**：假设一个 Episode 的奖励序列是 $[0, 0, 1, 0, 10]$，$\gamma = 0.9$

从 $t=0$ 开始的回报：
$$
\begin{aligned}
G_0 &= 0 + 0.9 \times 0 + 0.9^2 \times 1 + 0.9^3 \times 0 + 0.9^4 \times 10 \\
&= 0 + 0 + 0.81 + 0 + 6.561 \\
&= 7.371
\end{aligned}
$$

从 $t=2$ 开始的回报：
$$
G_2 = 1 + 0.9 \times 0 + 0.9^2 \times 10 = 1 + 0 + 8.1 = 9.1
$$

---

### 3.1.2 期望 vs 样本均值：大数定律

**问题**：在动态规划中，我们可以直接计算期望：

$$
V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} \mathcal{P}(s'|s,a) \left[ \mathcal{R}(s,a) + \gamma V^\pi(s') \right]
$$

但这需要知道环境模型 $\mathcal{P}$ 和 $\mathcal{R}$。在无模型（model-free）情况下，我们该怎么办？

**解决方案**：根据**大数定律（Law of Large Numbers）**，样本均值会收敛到期望值。

**大数定律（通俗版）**：

如果你重复做同一件随机实验很多次，这些结果的平均值会越来越接近理论上的期望值。

**数学表述**：

设 $X_1, X_2, \ldots, X_n$ 是独立同分布（i.i.d.）的随机变量，期望为 $\mu$，则：

$$
\lim_{n \to \infty} \frac{1}{n} \sum_{i=1}^n X_i = \mu
$$

**应用到强化学习**：

如果我们从状态 $s$ 开始，多次运行 Episode，每次都遵循策略 $\pi$，记录每次的回报 $G_t^{(1)}, G_t^{(2)}, \ldots, G_t^{(n)}$，那么：

$$
V^\pi(s) \approx \frac{1}{n} \sum_{i=1}^n G_t^{(i)}
$$

当 $n$ 足够大时，这个样本均值就是对 $V^\pi(s)$ 的良好估计。

**直观例子：估计圆周率 $\pi$**

这就像用蒙特卡洛方法估计圆周率：
1. 在一个边长为2的正方形内随机撒点
2. 统计落在内接圆（半径为1）内的点数
3. 圆周率 $\pi \approx 4 \times \frac{\text{圆内点数}}{\text{总点数}}$

你不需要知道圆的方程，只需要大量采样！

<div data-component="MCReturnEstimation"></div>

---

### 3.1.3 First-visit MC vs Every-visit MC

在一个 Episode 中，同一个状态可能被访问多次。我们该如何处理？

**First-visit MC（首次访问蒙特卡洛）**

只计算状态 $s$ **第一次**出现时的回报。

**例子**：Episode 轨迹为 $S_0, S_1, S_2, S_1, S_3$（状态 $S_1$ 出现了两次）

- 对于 $S_1$：只使用第一次出现（$t=1$）时的回报 $G_1$
- 忽略第二次出现（$t=3$）时的回报 $G_3$

**优点**：
- 每个 Episode 为每个状态提供**独立同分布（i.i.d.）**的样本
- 理论性质好，容易分析收敛性
- 是**无偏估计**（期望等于真实值）

**Every-visit MC（每次访问蒙特卡洛）**

计算状态 $s$ **每次**出现时的回报。

**例子**：同样的轨迹 $S_0, S_1, S_2, S_1, S_3$

- 对于 $S_1$：使用两次回报 $G_1$ 和 $G_3$

**优点**：
- 使用更多数据，可能收敛更快
- 实现更简单（不需要检查是否首次访问）

**缺点**：
- 同一 Episode 内的样本不是独立的（相关性）
- 理论分析更复杂

**实践中**：First-visit MC 更常用，因为理论保证更强。

---

### 3.1.4 算法实现

**First-visit MC 预测算法（伪代码）**

```
算法：First-visit MC 预测
输入：策略 π，折扣因子 γ
输出：状态价值函数 V

1. 初始化：
   对所有 s ∈ S:
       V(s) ← 任意值（通常为0）
       Returns(s) ← 空列表

2. 循环（对每个 Episode）：
   a) 使用策略 π 生成一个 Episode：
      S₀, A₀, R₁, S₁, A₁, R₂, ..., Sₜ₋₁, Aₜ₋₁, Rₜ
   
   b) G ← 0  // 初始化回报
   
   c) 从后向前遍历 Episode（t = T-1, T-2, ..., 0）：
      i.   G ← γ·G + Rₜ₊₁  // 递归计算回报
      ii.  如果 Sₜ 没有在 S₀, S₁, ..., Sₜ₋₁ 中出现过：
           - 将 G 添加到 Returns(Sₜ)
           - V(Sₜ) ← average(Returns(Sₜ))

3. 返回 V
```

**为什么从后向前遍历？**

因为回报的递归定义：

$$
G_t = R_{t+1} + \gamma G_{t+1}
$$

从最后一步开始：
- $G_{T-1} = R_T$（最后一步的回报就是最后的奖励）
- $G_{T-2} = R_{T-1} + \gamma G_{T-1}$
- $G_{T-3} = R_{T-2} + \gamma G_{T-2}$
- ...

这样可以高效地计算所有时刻的回报，时间复杂度 $O(T)$。

---

### 3.1.5 增量式更新公式（重要！）

**问题**：存储所有回报 $Returns(s)$ 需要大量内存。能否在线更新？

**答案**：可以！使用增量平均公式。

**推导过程**（逐步详解）：

设我们已经观察到状态 $s$ 的 $n$ 个回报：$G_1, G_2, \ldots, G_n$

当前的平均值为：

$$
V_n = \frac{1}{n} \sum_{i=1}^n G_i
$$

现在我们得到第 $n+1$ 个回报 $G_{n+1}$，新的平均值为：

$$
V_{n+1} = \frac{1}{n+1} \sum_{i=1}^{n+1} G_i
$$

**目标**：用 $V_n$ 和 $G_{n+1}$ 来表示 $V_{n+1}$，避免重新计算所有回报的和。

**步骤1**：展开 $V_{n+1}$

$$
\begin{aligned}
V_{n+1} &= \frac{1}{n+1} \sum_{i=1}^{n+1} G_i \\
&= \frac{1}{n+1} \left( \sum_{i=1}^{n} G_i + G_{n+1} \right)
\end{aligned}
$$

**步骤2**：注意到 $\sum_{i=1}^{n} G_i = n \cdot V_n$

$$
V_{n+1} = \frac{1}{n+1} \left( n \cdot V_n + G_{n+1} \right)
$$

**步骤3**：展开并整理

$$
\begin{aligned}
V_{n+1} &= \frac{n}{n+1} V_n + \frac{1}{n+1} G_{n+1} \\
&= V_n + \frac{1}{n+1} G_{n+1} - \frac{1}{n+1} V_n \\
&= V_n + \frac{1}{n+1} (G_{n+1} - V_n)
\end{aligned}
$$

**最终公式**：

$$
V_{n+1} = V_n + \frac{1}{n+1} (G_{n+1} - V_n)
$$

**公式解读**：
- $V_n$：旧的估计值
- $G_{n+1}$：新观察到的回报（**目标值**）
- $G_{n+1} - V_n$：**误差**（目标值与当前估计的差距）
- $\frac{1}{n+1}$：**步长**（学习率）
- 新估计 = 旧估计 + 步长 × 误差

**通用形式**：

$$
\text{NewEstimate} \leftarrow \text{OldEstimate} + \alpha \cdot (\text{Target} - \text{OldEstimate})
$$

其中 $\alpha$ 是学习率（步长）。

**两种步长选择**：

1. **样本平均**（Sample Average）：$\alpha = \frac{1}{n}$
   - 随着样本增多，步长减小
   - 适用于**平稳环境**（环境不变）
   - 保证收敛到真实均值

2. **固定步长**：$\alpha = \text{常数}$（如 0.1）
   - 步长固定，对新数据更敏感
   - 适用于**非平稳环境**（环境会变化）
   - 会"遗忘"旧数据，追踪最新趋势

**例子**：

假设 $V_3 = 5$（前3次的平均回报），现在观察到 $G_4 = 8$

使用 $\alpha = \frac{1}{4}$：
$$
V_4 = 5 + \frac{1}{4}(8 - 5) = 5 + 0.75 = 5.75
$$

使用 $\alpha = 0.1$：
$$
V_4 = 5 + 0.1(8 - 5) = 5 + 0.3 = 5.3
$$

---

### 3.1.6 收敛性分析

**定理**：First-visit MC 方法会收敛到真实的 $V^\pi(s)$

**条件**：
1. 每个状态被访问无穷多次
2. 使用样本平均（$\alpha = \frac{1}{n}$）

**证明思路**：

根据大数定律，当样本数 $n \to \infty$ 时：

$$
\frac{1}{n} \sum_{i=1}^n G_i \to \mathbb{E}[G] = V^\pi(s)
$$

**收敛速率**：

标准误差（Standard Error）：

$$
SE = \frac{\sigma}{\sqrt{n}}
$$

其中 $\sigma$ 是回报的标准差，$n$ 是样本数。

**含义**：要将误差减半，需要4倍的样本！

---

### 3.1.7 偏差-方差权衡

**MC 方法的特点**：

- **无偏（Unbiased）**：$\mathbb{E}[V_{MC}(s)] = V^\pi(s)$
  - 估计的期望等于真实值
  - 不依赖于初始值或其他状态的估计

- **高方差（High Variance）**：
  - 回报 $G_t$ 是整个轨迹的累积，受很多随机因素影响
  - 不同 Episode 的回报可能差异很大
  - 需要大量样本才能得到稳定估计

**直观理解**：

想象你在评估一家餐厅：
- **无偏**：你每次去都诚实记录体验，长期平均就是真实水平
- **高方差**：但每次体验可能差异很大（厨师状态、食材新鲜度等），需要去很多次才能确定

---

### 3.1.8 完整代码实现

```python
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from collections import defaultdict

class MCPredictor:
    """蒙特卡洛预测器"""
    
    def __init__(self, gamma: float = 0.99, alpha: float = None):
        """
        Args:
            gamma: 折扣因子
            alpha: 学习率（None 表示使用样本平均）
        """
        self.gamma = gamma
        self.alpha = alpha
        self.V = defaultdict(float)  # 状态价值函数
        self.returns = defaultdict(list)  # 存储每个状态的回报（用于样本平均）
        self.visit_counts = defaultdict(int)  # 访问次数
        
    def generate_episode(self, env, policy, max_steps: int = 1000) -> List[Tuple]:
        """
        生成一个 Episode
        
        Args:
            env: 环境
            policy: 策略函数 policy(state) -> action
            max_steps: 最大步数
            
        Returns:
            episode: [(state, action, reward), ...]
        """
        episode = []
        state, _ = env.reset()
        
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            
            if done or truncated:
                break
        
        return episode
    
    def update_first_visit(self, episode: List[Tuple]):
        """
        First-visit MC 更新
        
        详细步骤：
        1. 从后向前计算每个时刻的回报
        2. 对于每个状态，只在首次访问时更新
        """
        # 记录已访问的状态（用于 first-visit 检查）
        visited_states = set()
        
        # 从后向前遍历
        G = 0  # 回报初始化为0
        
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            
            # 递归计算回报：G_t = R_{t+1} + γ * G_{t+1}
            G = reward + self.gamma * G
            
            # First-visit 检查
            if state not in visited_states:
                visited_states.add(state)
                
                # 更新价值函数
                if self.alpha is None:
                    # 样本平均
                    self.returns[state].append(G)
                    self.V[state] = np.mean(self.returns[state])
                else:
                    # 固定步长
                    # V(s) ← V(s) + α[G - V(s)]
                    self.V[state] += self.alpha * (G - self.V[state])
                
                self.visit_counts[state] += 1
    
    def update_every_visit(self, episode: List[Tuple]):
        """Every-visit MC 更新"""
        G = 0
        
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = reward + self.gamma * G
            
            if self.alpha is None:
                self.returns[state].append(G)
                self.V[state] = np.mean(self.returns[state])
            else:
                self.V[state] += self.alpha * (G - self.V[state])
            
            self.visit_counts[state] += 1
    
    def predict(self, env, policy, num_episodes: int = 1000, 
                method: str = 'first-visit') -> Dict:
        """
        运行 MC 预测
        
        Args:
            env: 环境
            policy: 策略
            num_episodes: Episode 数量
            method: 'first-visit' 或 'every-visit'
            
        Returns:
            info: 包含学习曲线等信息的字典
        """
        value_history = []
        
        for episode_num in range(num_episodes):
            # 生成 Episode
            episode = self.generate_episode(env, policy)
            
            # 更新价值函数
            if method == 'first-visit':
                self.update_first_visit(episode)
            else:
                self.update_every_visit(episode)
            
            # 记录当前价值函数（用于可视化）
            if episode_num % 100 == 0:
                value_history.append(dict(self.V))
        
        return {
            'V': dict(self.V),
            'visit_counts': dict(self.visit_counts),
            'value_history': value_history
        }
    
    def visualize_convergence(self, value_history: List[Dict], 
                             true_V: Dict = None):
        """可视化价值函数的收敛过程"""
        if not value_history:
            return
        
        # 选择几个状态进行可视化
        states = list(value_history[0].keys())[:5]
        
        plt.figure(figsize=(12, 6))
        
        for state in states:
            values = [vh.get(state, 0) for vh in value_history]
            plt.plot(values, label=f'State {state}', linewidth=2)
            
            # 如果有真实值，画水平线
            if true_V and state in true_V:
                plt.axhline(y=true_V[state], linestyle='--', 
                           alpha=0.5, label=f'True V({state})')
        
        plt.xlabel('Episode (×100)', fontsize=12)
        plt.ylabel('Estimated Value', fontsize=12)
        plt.title('MC Prediction Convergence', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# 使用示例
if __name__ == "__main__":
    import gymnasium as gym
    
    # 创建环境（使用 FrozenLake）
    env = gym.make('FrozenLake-v1', is_slippery=False)
    
    # 定义一个简单策略（随机策略）
    def random_policy(state):
        return env.action_space.sample()
    
    # 创建 MC 预测器
    predictor = MCPredictor(gamma=0.99, alpha=0.1)
    
    # 运行预测
    print("Running MC Prediction...")
    info = predictor.predict(env, random_policy, num_episodes=5000)
    
    # 打印结果
    print("\nEstimated State Values:")
    for state, value in sorted(info['V'].items()):
        print(f"State {state}: V = {value:.4f}, Visits = {info['visit_counts'][state]}")
    
    # 可视化收敛过程
    predictor.visualize_convergence(info['value_history'])
```

---

### 3.1.9 实践技巧

**1. Episode 长度的影响**

- **短 Episode**：方差小，收敛快，但可能无法充分探索
- **长 Episode**：方差大，收敛慢，但能探索更多状态

**2. 初始值的选择**

- MC 方法对初始值不敏感（因为是无偏估计）
- 通常初始化为 0

**3. 学习率的选择**

- **样本平均**（$\alpha = 1/n$）：理论保证强，但对新数据不敏感
- **固定步长**（$\alpha = 0.1$）：适应性强，但可能不收敛到精确值

**4. 何时使用 MC？**

- ✅ Episode 任务（有明确的开始和结束）
- ✅ 环境模型未知
- ✅ 可以接受高方差
- ❌ 连续任务（无终止）
- ❌ 需要在线学习（每步更新）

---

### 3.1.10 小结

蒙特卡洛预测的核心要点：

1. **核心思想**：用样本均值估计期望 $V^\pi(s) \approx \frac{1}{n}\sum G_i$
2. **无偏估计**：期望正确，但方差高
3. **增量更新**：$V \leftarrow V + \alpha(G - V)$
4. **First-visit vs Every-visit**：理论 vs 实践的权衡
5. **收敛保证**：大数定律保证收敛到真实值

下一节我们将学习如何使用 MC 方法来寻找最优策略（MC 控制）。

---

## 3.2 蒙特卡洛控制 (MC Control)

**目标**：找到最优策略 $\pi^*$。

在上一节中，我们学习了如何评估一个给定的策略。现在的问题是：**如何找到最优策略？**

我们将沿用动态规划中的 **GPI（广义策略迭代）** 框架：
1. **策略评估（Policy Evaluation）**：估计当前策略的价值
2. **策略改进（Policy Improvement）**：基于价值改进策略
3. **重复**：直到收敛到最优策略

---

### 3.2.1 为什么要估计 Q 函数而不是 V 函数？

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

**直观理解**：

- $V(s)$ 告诉你"这个状态有多好"
- $Q(s,a)$ 告诉你"在这个状态下，采取这个动作有多好"

没有模型时，我们需要知道每个动作的价值才能选择最优动作。

---

### 3.2.2 探索与利用的困境 (Exploration-Exploitation Dilemma)

**核心矛盾**：

- **利用（Exploitation）**：选择当前已知最好的动作，获得高回报
- **探索（Exploration）**：尝试未知的动作，可能发现更好的策略

**问题**：如果我们总是贪心地选择当前最优动作，可能会陷入局部最优。

**具体例子**：

想象你在一个新城市找餐厅：
- **利用**：总是去你已经知道还不错的那家餐厅
- **探索**：尝试新的餐厅，可能找到更好的，也可能踩雷

如果你只利用，可能错过城市里最好的餐厅；如果只探索，每次都冒险吃难吃的。

**强化学习中的例子**：

假设一个简单的环境：
- 状态 $s$，有两个动作：$a_1$ 和 $a_2$
- 初始时，我们尝试了 $a_1$ 得到回报 +1，尝试了 $a_2$ 得到回报 0
- 如果我们使用完全贪心策略，以后永远只选 $a_1$
- 但实际上 $a_2$ 的真实期望回报可能是 +10（只是第一次运气不好）

**解决方案 1：Exploring Starts（探索性起始）**

**思想**：确保所有 $(s,a)$ 对都有非零概率作为 Episode 的起点。

**算法**：
1. 随机选择一个状态-动作对 $(s_0, a_0)$ 作为起点
2. 从 $a_0$ 开始执行，之后遵循当前策略
3. 这样保证每个动作都会被尝试

**局限性**：
- 在很多实际环境中无法实现
- 例如：自动驾驶不能从"不踩刹车"的状态开始
- 例如：机器人不能从"摔倒"的状态开始

**解决方案 2：$\epsilon$-Greedy 策略（最常用）**

**核心思想**：大部分时间利用，小部分时间探索。

**策略定义**：

$$
\pi(a|s) = \begin{cases}
1 - \epsilon + \frac{\epsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_{a'} Q(s, a') \quad \text{(贪心动作)} \\
\frac{\epsilon}{|\mathcal{A}|} & \text{otherwise} \quad \text{(其他动作)}
\end{cases}
$$

**符号详解**：
- $\epsilon$：探索概率（通常取 0.1 或 0.05）
- $|\mathcal{A}|$：动作空间的大小（有多少个动作）
- $\arg\max_{a'} Q(s, a')$：Q 值最大的动作（贪心动作）

**公式推导**：

假设有 $|\mathcal{A}| = 4$ 个动作，$\epsilon = 0.1$

**步骤1**：总概率必须为 1

我们希望：
- 以概率 $1-\epsilon = 0.9$ 选择贪心动作
- 以概率 $\epsilon = 0.1$ 随机探索

**步骤2**：随机探索时，每个动作概率相等

探索时，4个动作各有 $\frac{\epsilon}{4} = \frac{0.1}{4} = 0.025$ 的概率

**步骤3**：贪心动作的总概率

贪心动作 $a^*$ 的概率 = 贪心选择概率 + 随机探索时选到它的概率

$$
\pi(a^*|s) = (1-\epsilon) + \frac{\epsilon}{|\mathcal{A}|} = 0.9 + 0.025 = 0.925
$$

**步骤4**：其他动作的概率

非贪心动作只能通过随机探索被选中：

$$
\pi(a|s) = \frac{\epsilon}{|\mathcal{A}|} = 0.025 \quad \text{for } a \neq a^*
$$

**验证**：总概率 = $0.925 + 3 \times 0.025 = 0.925 + 0.075 = 1.0$ ✓

**直观理解**：

- 90% 的时间选择当前最优动作（利用）
- 10% 的时间随机选择（探索）
- 即使在探索时，也可能随机选到最优动作

**$\epsilon$ 的选择**：

- **大 $\epsilon$**（如 0.3）：探索多，学习快，但策略不稳定
- **小 $\epsilon$**（如 0.01）：探索少，策略稳定，但可能陷入局部最优
- **衰减 $\epsilon$**：开始大（多探索），逐渐减小（多利用）

---

### 3.2.3 GLIE 性质 (Greedy in the Limit with Infinite Exploration)

**定义**：一个策略序列 $\{\pi_k\}$ 满足 GLIE 如果：

1. **无限探索**：所有状态-动作对被访问无穷多次
   $$
   \lim_{k \to \infty} N_k(s,a) = \infty, \quad \forall s, a
   $$

2. **极限贪心**：策略在极限情况下收敛到贪心策略
   $$
   \lim_{k \to \infty} \pi_k(a|s) = \mathbb{1}(a = \arg\max_{a'} Q_k(s,a'))
   $$

**通俗解释**：

- 开始时多探索（保证每个动作都尝试很多次）
- 最终收敛到完全贪心（不再探索）

**$\epsilon$-greedy 的 GLIE 版本**：

让 $\epsilon$ 随时间衰减：

$$
\epsilon_k = \frac{1}{k}
$$

这样：
- 早期：$\epsilon$ 大，多探索
- 后期：$\epsilon \to 0$，趋向贪心

**为什么需要 GLIE？**

- 保证收敛到最优策略
- 平衡探索与利用

---

### 3.2.4 On-Policy MC Control 算法

**完整算法（伪代码）**：

```
算法：On-Policy First-Visit MC Control (ε-greedy)
输入：折扣因子 γ，探索率 ε
输出：近似最优策略 π ≈ π*

1. 初始化：
   对所有 s ∈ S, a ∈ A:
       Q(s,a) ← 任意值
       Returns(s,a) ← 空列表
   π ← 基于 Q 的 ε-greedy 策略

2. 循环（对每个 Episode）：
   a) 使用策略 π 生成 Episode：
      S₀, A₀, R₁, S₁, A₁, R₂, ..., Sₜ₋₁, Aₜ₋₁, Rₜ
   
   b) G ← 0
   
   c) 从后向前遍历（t = T-1, T-2, ..., 0）：
      i.   G ← γ·G + Rₜ₊₁
      
      ii.  如果 (Sₜ, Aₜ) 没有在 (S₀,A₀), ..., (Sₜ₋₁,Aₜ₋₁) 中出现：
           - 将 G 添加到 Returns(Sₜ, Aₜ)
           - Q(Sₜ, Aₜ) ← average(Returns(Sₜ, Aₜ))
           
           - 策略改进（ε-greedy）：
             a* ← argmax_a Q(Sₜ, a)
             对所有 a ∈ A(Sₜ):
                 if a == a*:
                     π(a|Sₜ) ← 1 - ε + ε/|A(Sₜ)|
                 else:
                     π(a|Sₜ) ← ε/|A(Sₜ)|

3. 返回 π
```

**算法特点**：

- **On-policy**：用于生成数据的策略（$\pi$）和被评估改进的策略是同一个
- **First-visit**：每个 Episode 中，每个 $(s,a)$ 对只在首次出现时更新
- **$\epsilon$-greedy**：保证持续探索

---

### 3.2.5 完整代码实现

```python
import numpy as np
import gymnasium as gym
from collections import defaultdict
from typing import Tuple, Dict
import matplotlib.pyplot as plt

class MCControl:
    """蒙特卡洛控制（On-Policy）"""
    
    def __init__(self, env, gamma: float = 0.99, epsilon: float = 0.1):
        """
        Args:
            env: Gymnasium 环境
            gamma: 折扣因子
            epsilon: 探索率
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q 函数：Q(s,a)
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        
        # 回报列表：用于计算平均
        self.returns = defaultdict(list)
        
        # 访问次数
        self.visit_counts = defaultdict(int)
        
    def epsilon_greedy_policy(self, state) -> int:
        """
        ε-greedy 策略
        
        详细步骤：
        1. 以概率 ε 随机选择动作（探索）
        2. 以概率 1-ε 选择 Q 值最大的动作（利用）
        """
        if np.random.random() < self.epsilon:
            # 探索：随机选择
            return self.env.action_space.sample()
        else:
            # 利用：选择 Q 值最大的动作
            return np.argmax(self.Q[state])
    
    def generate_episode(self) -> list:
        """
        生成一个 Episode
        
        Returns:
            episode: [(state, action, reward), ...]
        """
        episode = []
        state, _ = self.env.reset()
        done = False
        
        while not done:
            # 使用 ε-greedy 策略选择动作
            action = self.epsilon_greedy_policy(state)
            
            # 执行动作
            next_state, reward, done, truncated, _ = self.env.step(action)
            
            # 记录 (s, a, r)
            episode.append((state, action, reward))
            
            state = next_state
            
            if truncated:
                break
        
        return episode
    
    def update_q_function(self, episode: list):
        """
        更新 Q 函数（First-visit MC）
        
        详细步骤：
        1. 从后向前计算回报 G
        2. 对于首次访问的 (s,a)，更新 Q(s,a)
        """
        # 记录已访问的 (s,a) 对
        visited_sa = set()
        
        # 从后向前遍历
        G = 0
        
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            
            # 递归计算回报
            G = reward + self.gamma * G
            
            # First-visit 检查
            sa_pair = (state, action)
            if sa_pair not in visited_sa:
                visited_sa.add(sa_pair)
                
                # 更新 Q 函数（样本平均）
                self.returns[sa_pair].append(G)
                self.Q[state][action] = np.mean(self.returns[sa_pair])
                
                self.visit_counts[sa_pair] += 1
    
    def train(self, num_episodes: int = 10000) -> Dict:
        """
        训练 MC Control
        
        Args:
            num_episodes: Episode 数量
            
        Returns:
            info: 训练信息
        """
        episode_rewards = []
        episode_lengths = []
        
        for episode_num in range(num_episodes):
            # 生成 Episode
            episode = self.generate_episode()
            
            # 更新 Q 函数
            self.update_q_function(episode)
            
            # 记录统计信息
            total_reward = sum(r for _, _, r in episode)
            episode_rewards.append(total_reward)
            episode_lengths.append(len(episode))
            
            # 打印进度
            if (episode_num + 1) % 1000 == 0:
                avg_reward = np.mean(episode_rewards[-1000:])
                print(f"Episode {episode_num + 1}/{num_episodes}, "
                      f"Avg Reward: {avg_reward:.2f}")
        
        return {
            'Q': dict(self.Q),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }
    
    def get_greedy_policy(self) -> Dict:
        """提取贪心策略（用于最终评估）"""
        policy = {}
        for state in self.Q:
            policy[state] = np.argmax(self.Q[state])
        return policy
    
    def visualize_training(self, episode_rewards: list):
        """可视化训练过程"""
        # 计算移动平均
        window = 100
        moving_avg = np.convolve(episode_rewards, 
                                np.ones(window)/window, 
                                mode='valid')
        
        plt.figure(figsize=(12, 5))
        
        # 原始奖励
        plt.subplot(1, 2, 1)
        plt.plot(episode_rewards, alpha=0.3, label='Raw')
        plt.plot(range(window-1, len(episode_rewards)), 
                moving_avg, label=f'{window}-Episode Moving Avg', 
                linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Q 值分布
        plt.subplot(1, 2, 2)
        all_q_values = []
        for state_q in self.Q.values():
            all_q_values.extend(state_q)
        
        plt.hist(all_q_values, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Q Value')
        plt.ylabel('Frequency')
        plt.title('Q Value Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Blackjack 示例
if __name__ == "__main__":
    # 创建环境
    env = gym.make('Blackjack-v1', sab=True)
    
    # 创建 MC Control agent
    agent = MCControl(env, gamma=1.0, epsilon=0.1)
    
    # 训练
    print("Training MC Control on Blackjack...")
    info = agent.train(num_episodes=50000)
    
    # 可视化
    agent.visualize_training(info['episode_rewards'])
    
    # 提取学到的策略
    policy = agent.get_greedy_policy()
    
    # 展示几个典型状态的策略
    print("\nLearned Policy (Sample States):")
    test_states = [
        (20, 10, False),  # 玩家20点，庄家10，无Ace
        (16, 10, False),  # 玩家16点，庄家10，无Ace
        (12, 2, False),   # 玩家12点，庄家2，无Ace
    ]
    
    action_names = ['Stick', 'Hit']
    for state in test_states:
        if state in policy:
            action = policy[state]
            q_values = agent.Q[state]
            print(f"State {state}: {action_names[action]} "
                  f"(Q={q_values[action]:.3f})")
```

---

### 3.2.6 收敛性保证

**定理**：On-Policy First-Visit MC Control 在 GLIE 条件下收敛到最优策略。

**条件**：
1. 所有 $(s,a)$ 对被访问无穷多次
2. $\epsilon$ 衰减到 0（如 $\epsilon_k = 1/k$）

**证明思路**：

1. **策略评估**：First-visit MC 收敛到 $Q^\pi(s,a)$（大数定律）
2. **策略改进**：$\epsilon$-greedy 改进保证 $Q^{\pi_{k+1}} \geq Q^{\pi_k}$
3. **GLIE**：最终 $\epsilon \to 0$，策略收敛到贪心策略

---

### 3.2.7 实践技巧

**1. $\epsilon$ 衰减策略**

```python
# 线性衰减
epsilon = max(epsilon_min, epsilon_start - episode * decay_rate)

# 指数衰减
epsilon = epsilon_min + (epsilon_start - epsilon_min) * np.exp(-decay_rate * episode)

# 1/k 衰减（理论保证）
epsilon = 1.0 / (episode + 1)
```

**2. 学习率选择**

- **样本平均**：理论保证，但对新数据不敏感
- **固定步长**：$Q(s,a) \leftarrow Q(s,a) + \alpha[G - Q(s,a)]$，更适应非平稳环境

**3. 初始化技巧**

- **乐观初始化**：$Q(s,a) = $ 大值，鼓励探索
- **零初始化**：$Q(s,a) = 0$，中性

---

### 3.2.8 小结

蒙特卡洛控制的核心要点：

1. **估计 Q 函数**：无模型环境必须估计 $Q(s,a)$ 而非 $V(s)$
2. **探索-利用**：$\epsilon$-greedy 平衡探索与利用
3. **GLIE**：保证收敛到最优策略
4. **On-policy**：用同一个策略生成数据和学习

下一节我们将学习 On-policy 与 Off-policy 的区别。
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

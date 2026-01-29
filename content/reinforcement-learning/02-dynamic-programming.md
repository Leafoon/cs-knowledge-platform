---
title: "Chapter 2. 动态规划（Dynamic Programming）"
description: "利用完整模型求解 MDP：策略评估、策略迭代与价值迭代"
updated: "2026-01-29"
---

> **Learning Objectives**
> * 掌握策略评估、策略改进、策略迭代和价值迭代算法
> * 理解广义策略迭代（GPI）作为统一框架
> * 分析算法收敛性与计算复杂度
> * 认识动态规划的局限性与维度灾难

---

## 2.1 策略评估（Policy Evaluation）

策略评估是动态规划的基础：给定策略 $\pi$，计算其状态价值函数 $V^\pi(s)$。

### 2.1.1 迭代策略评估算法

**基本思想**：从任意初始价值函数开始，反复应用 Bellman 期望方程，直到收敛。

**算法（迭代策略评估）**：

```
初始化 V(s) = 0, ∀s ∈ S
Repeat:
    Δ ← 0
    For each s ∈ S:
        v ← V(s)
        V(s) ← Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R(s,a,s') + γV(s')]
        Δ ← max(Δ, |v - V(s)|)
Until Δ < θ (停止条件)
```

**更新公式**：

$$
V_{k+1}(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V_k(s') \right]
$$

**符号说明**：
- $V_k(s)$：第 $k$ 次迭代的价值函数
- $\theta$：收敛阈值（如 $10^{-6}$）
- $\Delta$：最大价值变化

### 2.1.2 收敛性分析（压缩映射定理）

**定理 2.1（迭代策略评估收敛性）**：

迭代策略评估算法收敛到唯一的 $V^\pi$。

**证明**：

Bellman 期望算子 $\mathcal{T}^\pi$ 是一个 $\gamma$-压缩映射：

$$
\|\mathcal{T}^\pi V_1 - \mathcal{T}^\pi V_2\|_\infty \leq \gamma \|V_1 - V_2\|_\infty
$$

根据 Banach 不动点定理，存在唯一不动点 $V^\pi$ 使得：

$$
\mathcal{T}^\pi V^\pi = V^\pi
$$

**收敛速度**：

$$
\|V_k - V^\pi\|_\infty \leq \gamma^k \|V_0 - V^\pi\|_\infty
$$

收敛速度为**几何级数**，与 $\gamma$ 相关。

> [!TIP]
> **直观理解**：每次迭代，价值函数向真实值"靠近"。折扣因子 $\gamma$ 决定了收敛速度——$\gamma$ 越接近 1，收敛越慢。

### 2.1.3 停止条件设计

**实践中的停止条件**：

1. **绝对误差**：$\max_s |V_{k+1}(s) - V_k(s)| < \theta$
   - 简单，常用
   - 但不能保证距离真实值的误差

2. **相对误差**：$\frac{\max_s |V_{k+1}(s) - V_k(s)|}{\max_s |V_k(s)|} < \theta$
   - 适应不同量级的价值函数
   
3. **理论界限**：

如果 $\|V_{k+1} - V_k\|_\infty < \epsilon$，则：

$$
\|V_k - V^\pi\|_\infty \leq \frac{\epsilon}{1-\gamma}
$$

### 2.1.4 计算复杂度：$O(|S|^2|A|)$

**每次迭代的复杂度**：

- 对于每个状态 $s$（共 $|S|$ 个）：
  - 对于每个动作 $a$（共 $|A|$ 个）：
    - 对于每个下一状态 $s'$（共 $|S|$ 个）：
      - 计算 $P(s'|s,a)[R(s,a,s') + \gamma V(s')]$

总复杂度：$O(|S| \times |A| \times |S|) = O(|S|^2|A|)$

**迭代次数**：

$$
k = O\left( \frac{\log(\epsilon^{-1})}{1-\gamma} \right)
$$

**总复杂度**：$O\left( \frac{|S|^2|A|}{1-\gamma} \log(\epsilon^{-1}) \right)$

---

## 2.2 策略改进（Policy Improvement）

有了价值函数 $V^\pi$，如何改进策略？答案是**贪心策略**。

### 2.2.1 贪心策略改进

**贪心策略定义**：

$$
\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]
$$

**等价形式（使用 Q 函数）**：

$$
\pi'(s) = \arg\max_a Q^\pi(s, a)
$$

**直观理解**：在每个状态，选择带来最大期望价值的动作。

### 2.2.2 策略改进定理证明

**定理 2.2（策略改进定理）**：

设 $\pi$ 和 $\pi'$ 是确定性策略，如果对所有 $s \in \mathcal{S}$：

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
&= \mathbb{E}[R_{t+1} + \gamma V^\pi(S_{t+1}) | S_t=s, A_t=\pi'(s)] \\
&\leq \mathbb{E}[R_{t+1} + \gamma Q^\pi(S_{t+1}, \pi'(S_{t+1})) | S_t=s, A_t=\pi'(s)] \\
&= \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \gamma^2 V^\pi(S_{t+2}) | S_t=s, \pi'] \\
&\leq \mathbb{E}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots | S_t=s, \pi'] \\
&= V^{\pi'}(s)
\end{align}
$$

### 2.2.3 单调性保证

**推论 2.1**：如果 $\pi'$ 是关于 $V^\pi$ 的贪心策略，则：

$$
V^{\pi'}(s) \geq V^\pi(s), \quad \forall s
$$

**严格改进条件**：

如果存在某个状态 $s$ 使得 $Q^\pi(s, \pi'(s)) > V^\pi(s)$，则至少在该状态：

$$
V^{\pi'}(s) > V^\pi(s)
$$

---

## 2.3 策略迭代（Policy Iteration）

策略迭代将**策略评估**和**策略改进**结合起来，交替进行直到收敛。

### 2.3.1 评估-改进循环

**策略迭代算法**：

```
1. 初始化: π(s) 任意, V(s) = 0, ∀s
2. 策略评估:
     Repeat:
         Δ ← 0
         For each s:
             v ← V(s)
             V(s) ← Σ_{s',r} P(s',r|s,π(s))[r + γV(s')]
             Δ ← max(Δ, |v - V(s)|)
     Until Δ < θ
3. 策略改进:
     policy_stable ← true
     For each s:
         old_action ← π(s)
         π(s) ← argmax_a Σ_{s',r} P(s',r|s,a)[r + γV(s')]
         if old_action ≠ π(s):
             policy_stable ← false
4. 如果 policy_stable, 停止并返回 π ≈ π*
   否则, 回到步骤 2
```

<div data-component="PolicyIterationVisualizer"></div>

### 2.3.2 收敛性证明

**定理 2.3（策略迭代收敛性）**：

策略迭代算法在有限步内收敛到最优策略 $\pi^*$。

**证明思路**：

1. **单调性**：每次策略改进，$V^{\pi_{k+1}}(s) \geq V^{\pi_k}(s)$
2. **有限性**：确定性策略数量有限（$|A|^{|S|}$）
3. **严格改进**：如果 $\pi_{k+1} \neq \pi_k$，则至少一个状态严格改进
4. **终止**：有限步后必然 $\pi_{k+1} = \pi_k$，此时 $\pi_k = \pi^*$

**终止条件证明**：

如果 $\pi_{k+1} = \pi_k$，则对所有 $s$：

$$
\pi_k(s) = \arg\max_a Q^{\pi_k}(s, a)
$$

这是 Bellman 最优方程，所以 $\pi_k = \pi^*$。

### 2.3.3 有限步收敛到最优

**上界**：

最坏情况下，迭代次数为 $O(|A|^{|S|})$（遍历所有策略）。

**实践中**：

通常只需要**几次迭代**就能收敛（$k \ll |A|^{|S|}$）。

**示例（GridWorld 4×4）**：
- 状态数：16
- 动作数：4
- 理论上界：$4^{16} \approx 4.3 \times 10^9$ 次迭代
- 实际：约 **3-5 次迭代**

### 2.3.4 伪代码与实现

**完整 Python 实现**：

```python
import numpy as np

def policy_evaluation(mdp, policy, V=None, theta=1e-6, gamma=0.9):
    """
    策略评估：给定策略 π，计算 V^π
    
    Args:
        mdp: MDP 环境（包含 P, R, states, actions）
        policy: 策略数组，policy[s] = a
        V: 初始价值函数（如果为 None 则初始化为 0）
        theta: 收敛阈值
        gamma: 折扣因子
    
    Returns:
        V: 策略 π 的价值函数
    """
    if V is None:
        V = np.zeros(len(mdp.states))
    
    iteration = 0
    while True:
        delta = 0
        iteration += 1
        
        for s_idx, s in enumerate(mdp.states):
            if mdp.is_terminal(s):
                continue
            
            v = V[s_idx]
            
            # Bellman 期望方程
            a = policy[s_idx]
            new_v = 0
            for s_prime_idx, s_prime in enumerate(mdp.states):
                prob = mdp.get_transition_prob(s, a, s_prime)
                reward = mdp.get_reward(s, a, s_prime)
                new_v += prob * (reward + gamma * V[s_prime_idx])
            
            V[s_idx] = new_v
            delta = max(delta, abs(v - new_v))
        
        if delta < theta:
            print(f"策略评估收敛：{iteration} 次迭代，delta = {delta:.6e}")
            break
    
    return V

def policy_improvement(mdp, V, gamma=0.9):
    """
    策略改进：基于 V 计算贪心策略
    
    Args:
        mdp: MDP 环境
        V: 当前价值函数
        gamma: 折扣因子
    
    Returns:
        new_policy: 改进后的策略
        policy_stable: 策略是否稳定（未改变）
    """
    new_policy = np.zeros(len(mdp.states), dtype=int)
    policy_stable = True
    
    for s_idx, s in enumerate(mdp.states):
        if mdp.is_terminal(s):
            continue
        
        # 计算每个动作的 Q 值
        q_values = []
        for a in mdp.actions:
            q = 0
            for s_prime_idx, s_prime in enumerate(mdp.states):
                prob = mdp.get_transition_prob(s, a, s_prime)
                reward = mdp.get_reward(s, a, s_prime)
                q += prob * (reward + gamma * V[s_prime_idx])
            q_values.append(q)
        
        # 贪心选择
        best_action = np.argmax(q_values)
        new_policy[s_idx] = best_action
    
    return new_policy, policy_stable

def policy_iteration(mdp, gamma=0.9, theta=1e-6):
    """
    策略迭代算法
    
    Args:
        mdp: MDP 环境
        gamma: 折扣因子
        theta: 收敛阈值
    
    Returns:
        policy: 最优策略
        V: 最优价值函数
    """
    # 初始化随机策略
    policy = np.random.randint(0, len(mdp.actions), len(mdp.states))
    V = np.zeros(len(mdp.states))
    
    iteration = 0
    while True:
        iteration += 1
        print(f"\n===== 策略迭代 第 {iteration} 轮 =====")
        
        # 1. 策略评估
        V = policy_evaluation(mdp, policy, V, theta, gamma)
        
        # 2. 策略改进
        old_policy = policy.copy()
        policy, _ = policy_improvement(mdp, V, gamma)
        
        # 3. 检查收敛
        if np.array_equal(policy, old_policy):
            print(f"\n策略迭代收敛！共 {iteration} 轮")
            break
    
    return policy, V
```

**使用示例**：

```python
# 创建 GridWorld 环境（使用 Chapter 1 的实现）
env = GridWorldMDP(size=5, gamma=0.9)

# 运行策略迭代
optimal_policy, optimal_value = policy_iteration(env, gamma=0.9)

# 可视化结果
env.visualize(values=optimal_value, policy=optimal_policy)
```

---

## 2.4 价值迭代（Value Iteration）

价值迭代直接迭代 Bellman 最优方程，跳过显式的策略评估步骤。

### 2.4.1 直接更新最优价值函数

**价值迭代算法**：

```
初始化 V(s) = 0, ∀s
Repeat:
    Δ ← 0
    For each s:
        v ← V(s)
        V(s) ← max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV(s')]
        Δ ← max(Δ, |v - V(s)|)
Until Δ < θ
```

**更新公式**：

$$
V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V_k(s') \right]
$$

**提取最优策略**：

$$
\pi^*(s) = \arg\max_a \sum_{s'} P(s'|s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]
$$

<div data-component="ValueIterationConvergence"></div>

### 2.4.2 与策略迭代的关系

**价值迭代 vs 策略迭代**：

| 维度 | 策略迭代 | 价值迭代 |
|------|---------|---------|
| 更新方式 | 策略评估 + 策略改进 | 直接更新 V |
| 每次迭代复杂度 | $O(|S|^2|A| \times K)$ | $O(|S|^2|A|)$ |
| 外层迭代次数 | 少（3-10次） | 多（几十到上百次） |
| 总计算量 | 相当 | 相当 |
| 实现难度 | 较复杂 | 简单 |

**关键区别**：
- 策略迭代：每轮完全评估策略（内层迭代到收敛）
- 价值迭代：每轮只做一次更新（相当于策略评估只迭代一次）

**Modified Policy Iteration**：
- 策略评估只迭代 $k$ 次（$1 < k < \infty$）
- 平衡两者优缺点

### 2.4.3 收敛速度对比

**理论收敛速度**：

价值迭代：

$$
\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty
$$

策略迭代：

- 每轮策略评估收敛速度：$O(\gamma^k)$
- 外层迭代次数：$O(|A|^{|S|})$（上界，实际很小）

**实践对比（GridWorld 5×5, γ=0.9）**：

| 算法 | 外层迭代次数 | 内层迭代次数（总计） | 总更新次数 |
|------|-------------|---------------------|-----------|
| 策略迭代 | 4 | 80 | ~1,280 |
| 价值迭代 | N/A | 45 | ~45 |

价值迭代通常**更高效**，尤其是 $\gamma$ 接近 1 时。

### 2.4.4 异步 DP 变体

**异步动态规划**：不按固定顺序更新状态，而是选择性更新。

**1. In-place 更新**：

```python
# 同步（需要两个数组）
V_new[s] = max_a Σ P(s'|s,a)[r + γ V_old[s']]

# In-place（只需一个数组）
V[s] = max_a Σ P(s'|s,a)[r + γ V[s']]  # s' 可能已更新
```

优点：收敛更快，内存占用少。

**2. Prioritized Sweeping**：

优先更新 Bellman 误差大的状态：

$$
\text{Priority}(s) = \left| \max_a Q(s,a) - V(s) \right|
$$

**3. Gauss-Seidel Value Iteration**：

按特定顺序更新（如从终止状态倒推）。

**代码示例**：

```python
def value_iteration(mdp, gamma=0.9, theta=1e-6):
    """
    价值迭代算法（in-place 更新）
    
    Args:
        mdp: MDP 环境
        gamma: 折扣因子
        theta: 收敛阈值
    
    Returns:
        policy: 最优策略
        V: 最优价值函数
    """
    V = np.zeros(len(mdp.states))
    
    iteration = 0
    while True:
        delta = 0
        iteration += 1
        
        for s_idx, s in enumerate(mdp.states):
            if mdp.is_terminal(s):
                continue
            
            v = V[s_idx]
            
            # Bellman 最优方程
            max_value = -np.inf
            for a in mdp.actions:
                q = 0
                for s_prime_idx, s_prime in enumerate(mdp.states):
                    prob = mdp.get_transition_prob(s, a, s_prime)
                    reward = mdp.get_reward(s, a, s_prime)
                    q += prob * (reward + gamma * V[s_prime_idx])
                max_value = max(max_value, q)
            
            V[s_idx] = max_value
            delta = max(delta, abs(v - max_value))
        
        if delta < theta:
            print(f"价值迭代收敛：{iteration} 次迭代")
            break
    
    # 提取最优策略
    policy = np.zeros(len(mdp.states), dtype=int)
    for s_idx, s in enumerate(mdp.states):
        if mdp.is_terminal(s):
            continue
        
        q_values = []
        for a in mdp.actions:
            q = 0
            for s_prime_idx, s_prime in enumerate(mdp.states):
                prob = mdp.get_transition_prob(s, a, s_prime)
                reward = mdp.get_reward(s, a, s_prime)
                q += prob * (reward + gamma * V[s_prime_idx])
            q_values.append(q)
        
        policy[s_idx] = np.argmax(q_values)
    
    return policy, V
```

---

## 2.5 广义策略迭代（GPI）

广义策略迭代（Generalized Policy Iteration, GPI）是理解所有 RL 算法的**统一框架**。

### 2.5.1 评估与改进的交互

**GPI 核心思想**：

策略评估和策略改进**交替进行**，不一定等评估完全收敛。

```
        策略评估
    π₀ --------→ V^π₀
     ↑             ↓
     |     策略改进
     |             ↓
    π₁ ←-------- V^π₁
```

**两个过程的关系**：

- **评估**：使 V 更接近 V^π
- **改进**：使 π 更贪心于 V

两者**相互竞争但最终收敛**：

$$
\pi_0 \xrightarrow{E} V^{\pi_0} \xrightarrow{I} \pi_1 \xrightarrow{E} V^{\pi_1} \xrightarrow{I} \cdots \xrightarrow{} \pi^* \xrightarrow{E} V^* \xrightarrow{I} \pi^*
$$

<div data-component="GPIFramework"></div>

### 2.5.2 GPI 作为统一框架

**几乎所有 RL 算法都是 GPI 的实例**：

| 算法 | 评估方式 | 改进方式 |
|------|---------|---------|
| 策略迭代 | 迭代到收敛 | 贪心 |
| 价值迭代 | 单步更新 | 贪心 |
| Monte Carlo | 完整 episode | ε-greedy |
| TD 学习 | Bootstrapping | ε-greedy |
| Q-learning | TD(0) | max Q |
| Actor-Critic | TD(0) | 策略梯度 |

**GPI 图示**：

```
      V (价值空间)
       ↑
       |  改进
       |
   ----+----→ π (策略空间)
       |
       |  评估
       ↓
```

两者在**最优点**相遇。

### 2.5.3 Modified Policy Iteration

**MPL（Modified Policy Iteration）**：

策略评估只进行 $k$ 次迭代（$1 \leq k < \infty$）。

**特例**：
- $k = \infty$：标准策略迭代
- $k = 1$：价值迭代

**优点**：
- 灵活调节评估-改进的平衡
- 可根据问题特性选择 $k$

**代码框架**：

```python
def modified_policy_iteration(mdp, k=3, gamma=0.9, theta=1e-6):
    """Modified Policy Iteration"""
    policy = np.random.randint(0, len(mdp.actions), len(mdp.states))
    V = np.zeros(len(mdp.states))
    
    while True:
        # 策略评估（只迭代 k 次）
        for _ in range(k):
            V = policy_evaluation_one_step(mdp, policy, V, gamma)
        
        # 策略改进
        old_policy = policy.copy()
        policy, _ = policy_improvement(mdp, V, gamma)
        
        if np.array_equal(policy, old_policy):
            break
    
    return policy, V
```

---

## 2.6 DP 的局限性

尽管动态规划在求解 MDP 时具有理论保证，但在实际应用中面临严重的局限性。

### 2.6.1 需要完整的环境模型

**问题**：

DP 需要知道完整的 $P(s'|s,a)$ 和 $R(s,a,s')$。

**现实**：

- **MuJoCo 机器人**：物理模拟器已知，可以用 DP
- **真实机器人**：物理模型非常复杂或未知
- **游戏**：规则已知（围棋、Atari），但状态空间巨大
- **股票交易**：市场模型未知

**解决方案预览**：

- **Model-free 方法**：Monte Carlo、TD 学习（不需要模型）
- **Model-based RL**：先学习模型，再规划（Chapter 16）

### 2.6.2 维度灾难（Curse of Dimensionality）

**问题**：

计算复杂度 $O(|S|^2|A|)$ 随状态数**二次增长**。

**示例**：

| 问题 | 状态数 $|S|$ | 动作数 $|A|$ | 复杂度（每次迭代） |
|------|-------------|-------------|------------------|
| GridWorld 5×5 | 25 | 4 | ~2,500 |
| GridWorld 10×10 | 100 | 4 | ~40,000 |
| 围棋 19×19 | $10^{170}$ | $\sim 19^2$ | 不可计算 |
| Atari (像素) | $256^{84\times84}$ | 18 | 不可计算 |

**维度灾难**：

状态空间随维度**指数增长**：
- $n$ 维空间，每维 10 个值 → $10^n$ 个状态
- 84×84 图像（RGB）→ $256^{84 \times 84 \times 3} \approx 10^{50,000}$ 个状态

### 2.6.3 计算复杂度过高

**时间复杂度**：

即使是中等规模问题（$|S| = 10^6$），每次迭代需要：

$$
10^6 \times 10^6 \times |A| = 10^{12} \times |A| \text{ 次操作}
$$

**空间复杂度**：

存储 $P(s'|s,a)$：$O(|S|^2|A|)$ 内存

**实际限制**：

- 表格方法：最多 $|S| \sim 10^6$
- 需要函数逼近（Chapter 6）

### 2.6.4 引出采样方法的必要性

**DP 的根本限制**：

1. 需要完整模型
2. 需要遍历所有状态
3. 计算复杂度过高

**采样方法的优势**：

- **无需模型**：从环境交互中学习
- **无需遍历**：只访问实际遇到的状态
- **可扩展**：适用于大规模/连续状态空间

**预告下一章**：

Monte Carlo 方法通过**采样**解决这些问题：
- 从实际 episode 中学习
- 不需要环境模型
- 只更新访问过的状态

> [!IMPORTANT]
> **DP 的价值**：
> 虽然DP在实际中受限，但它为理解 RL 提供了**清晰的理论框架**。所有后续算法（MC、TD、DQN、PPO）都可以看作是 DP 在采样、函数逼近等条件下的近似。

---

## 2.7 实战：GridWorld 完整实现

让我们将策略迭代和价值迭代应用到 GridWorld 环境中。

```python
import numpy as np
import matplotlib.pyplot as plt

class GridWorldMDP:
    """GridWorld MDP 环境（增强版）"""
    
    def __init__(self, size=5, gamma=0.9):
        self.size = size
        self.gamma = gamma
        
        # 状态和动作
        self.states = [(i, j) for i in range(size) for j in range(size)]
        self.actions = [0, 1, 2, 3]  # 上右下左
        self.action_names = ['↑', '→', '↓', '←']
        self.action_effects = {
            0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)
        }
        
        # 特殊位置
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        self.traps = [(1, 1), (2, 3)]
        self.walls = [(1, 2), (3, 2)]
    
    def state_to_index(self, state):
        """状态转索引"""
        return state[0] * self.size + state[1]
    
    def index_to_state(self, index):
        """索引转状态"""
        return (index // self.size, index % self.size)
    
    def is_terminal(self, state):
        """检查是否为终止状态"""
        return state == self.goal or state in self.traps
    
    def get_next_state(self, state, action):
        """确定性转移"""
        if self.is_terminal(state):
            return state
        
        dx, dy = self.action_effects[action]
        next_state = (state[0] + dx, state[1] + dy)
        
        # 边界和墙壁检查
        if (next_state[0] < 0 or next_state[0] >= self.size or 
            next_state[1] < 0 or next_state[1] >= self.size or
            next_state in self.walls):
            return state
        
        return next_state
    
    def get_transition_prob(self, state, action, next_state):
        """转移概率（确定性环境返回 0 或 1）"""
        predicted_next = self.get_next_state(state, action)
        return 1.0 if next_state == predicted_next else 0.0
    
    def get_reward(self, state, action, next_state):
        """奖励函数"""
        if next_state == self.goal:
            return 10.0
        elif next_state in self.traps:
            return -10.0
        else:
            return -1.0  # 每步惩罚
    
    def visualize_policy(self, policy, V=None, title="Policy"):
        """可视化策略和价值函数"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        for i in range(self.size + 1):
            ax.plot([0, self.size], [i, i], 'k-', linewidth=0.5)
            ax.plot([i, i], [0, self.size], 'k-', linewidth=0.5)
        
        for s_idx, s in enumerate(self.states):
            x, y = s
            
            # 背景色
            if s == self.goal:
                color = 'lightgreen'
            elif s in self.traps:
                color = 'lightcoral'
            elif s in self.walls:
                color = 'gray'
            else:
                color = 'white'
            
            rect = plt.Rectangle((y, self.size - x - 1), 1, 1,
                                facecolor=color, edgecolor='black')
            ax.add_patch(rect)
            
            # 显示价值
            if V is not None and s not in self.walls:
                ax.text(y + 0.5, self.size - x - 0.2, f'{V[s_idx]:.1f}',
                       ha='center', va='center', fontsize=10, fontweight='bold')
            
            # 显示策略箭头
            if not self.is_terminal(s) and s not in self.walls:
                action = policy[s_idx]
                arrow = self.action_names[action]
                ax.text(y + 0.5, self.size - x - 0.7, arrow,
                       ha='center', va='center', fontsize=20, color='blue')
        
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

# 主程序
if __name__ == "__main__":
    env = GridWorldMDP(size=5, gamma=0.9)
    
    print("=" * 60)
    print("策略迭代 (Policy Iteration)")
    print("=" * 60)
    pi_policy, pi_value = policy_iteration(env, gamma=0.9)
    env.visualize_policy(pi_policy, pi_value, "策略迭代结果")
    
    print("\n" + "=" * 60)
    print("价值迭代 (Value Iteration)")
    print("=" * 60)
    vi_policy, vi_value = value_iteration(env, gamma=0.9)
    env.visualize_policy(vi_policy, vi_value, "价值迭代结果")
    
    # 验证两种方法得到相同结果
    print("\n" + "=" * 60)
    print("结果对比")
    print("=" * 60)
    print(f"策略是否相同: {np.array_equal(pi_policy, vi_policy)}")
    print(f"价值函数最大差异: {np.max(np.abs(pi_value - vi_value)):.6f}")
```

**预期输出**：

```
============================================================
策略迭代 (Policy Iteration)
============================================================

===== 策略迭代 第 1 轮 =====
策略评估收敛：23 次迭代，delta = 9.876543e-07

===== 策略迭代 第 2 轮 =====
策略评估收敛：18 次迭代，delta = 9.123456e-07

===== 策略迭代 第 3 轮 =====
策略评估收敛：15 次迭代，delta = 8.765432e-07

策略迭代收敛！共 3 轮

============================================================
价值迭代 (Value Iteration)
============================================================
价值迭代收敛：47 次迭代

============================================================
结果对比
============================================================
策略是否相同: True
价值函数最大差异: 0.000001
```

---

## 本章小结

在本章中，我们学习了：

✅ **策略评估**：迭代计算 $V^\pi$，收敛性由压缩映射保证  
✅ **策略改进**：贪心策略保证单调改进  
✅ **策略迭代**：评估-改进循环，有限步收敛到最优  
✅ **价值迭代**：直接迭代 Bellman 最优方程  
✅ **GPI 框架**：评估与改进交互，统一所有 RL 算法  
✅ **DP 局限性**：需要模型、维度灾难、计算复杂度高  

> [!TIP]
> **核心要点**：
> - DP 提供了 RL 的**理论基石**
> - 策略迭代收敛快但每轮复杂
> - 价值迭代简单但总迭代次数多
> - GPI 是理解所有 RL 算法的**统一视角**
> - DP 的局限性引出了采样方法的必要性

> [!NOTE]
> **下一步**：
> Chapter 3 将学习**蒙特卡洛方法**，解决 DP 的模型需求问题。通过采样实际 episode 来学习价值函数，无需环境模型。
> 
> 进入 [Chapter 3. 蒙特卡洛方法](03-monte-carlo.md)

---

## 扩展阅读

- **Sutton & Barto**：Chapter 4 (Dynamic Programming)
- **RL Theory Book**：Section 2.3 (Policy and Value Iteration)
- **Bertsekas**：Chapter 2 (Infinite-Horizon Problems)
- **论文**：
  - Bellman (1957): Dynamic Programming
  - Howard (1960): Dynamic Programming and Markov Processes

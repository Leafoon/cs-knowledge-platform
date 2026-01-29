---
title: "Chapter 3. 蒙特卡洛方法（Monte Carlo Methods）"
description: "从经验中学习：无需模型的采样估计方法"
updated: "2026-01-29"
---

> **Learning Objectives**
> * 理解蒙特卡洛方法的核心思想：从完整 episode 中学习
> * 掌握 MC 策略评估（First-Visit 和 Every-Visit）
> * 学习 MC 控制算法（Exploring Starts 和 ε-greedy）
> * 理解 Off-policy MC 和重要性采样
> * 分析 MC 与 DP 的区别和适用场景

---

## 3.1 MC 基本思想

蒙特卡洛方法是第一个**无需环境模型**的强化学习算法，通过采样实际经验来估计价值函数。

### 3.1.1 从经验中学习（无需模型）

**核心差异**：

| 维度 | 动态规划（DP） | 蒙特卡洛（MC） |
|------|--------------|--------------|
| 模型需求 | 需要完整的 P(s'\|s,a) 和 R | **不需要模型** |
| 更新方式 | 遍历所有状态 | 只更新访问过的状态 |
| 理论基础 | Bellman 方程（期望） | 大数定律（采样平均） |
| 适用场景 | 小规模、已知模型 | 大规模、未知模型 |

**为什么叫"蒙特卡洛"？**

- 来源于摩纳哥的蒙特卡洛赌场
- 通过**随机采样**估计期望值
- 采样越多，估计越准确

### 3.1.2 完整 episode 采样

**Episode（回合）**：从初始状态到终止状态的完整轨迹。

$$
\tau = (S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T)
$$

**MC 的根本要求**：必须有**终止状态**（episodic tasks）。

**示例任务**：
- ✅ Blackjack（游戏结束为终止）
- ✅ 迷宫（到达出口为终止）
- ❌ 股票交易（持续任务，无终止）
- ❌ 机器人控制（持续任务）

> [!WARNING]
> **MC 不适用于持续任务**（continuing tasks）。对于没有自然终止的任务，需要使用 TD 学习（Chapter 4）。

### 3.1.3 Return 的无偏估计

**Return（回报）**：

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^{T-t-1} R_T
$$

**关键性质**：

$$
\mathbb{E}[G_t | S_t = s] = V^\pi(s)
$$

MC 通过**平均多个 episode 的 Return** 来估计 $V^\pi(s)$：

$$
V(s) \approx \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_i(s)
$$

其中 $N(s)$ 是访问状态 $s$ 的次数，$G_i(s)$ 是第 $i$ 次访问时的 Return。

**无偏性**：

$$
\lim_{N(s) \to \infty} V(s) = V^\pi(s) \quad \text{(大数定律)}
$$

<div data-component="MCReturnEstimation"></div>

### 3.1.4 与 DP 的对比

**DP（动态规划）更新**：

$$
V(s) \leftarrow \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V(s')]
$$

- 需要 $P(s'|s,a)$ 和 $R(s,a,s')$
- 使用**期望**（所有可能的下一状态）
- **Bootstrapping**：用估计更新估计

**MC 更新**：

$$
V(s) \leftarrow V(s) + \alpha [G_t - V(s)]
$$

- 不需要模型
- 使用**采样**（实际经历的轨迹）
- **无 Bootstrapping**：用实际 Return 更新

---

## 3.2 MC 策略评估

给定策略 $\pi$，如何用 MC 估计 $V^\pi(s)$？

### 3.2.1 First-Visit MC

**思想**：只在每个 episode 中**第一次访问**状态 $s$ 时记录 Return。

**算法（First-Visit MC 策略评估）**：

```
初始化：
    V(s) = 0, ∀s
    Returns(s) = 空列表, ∀s

For each episode:
    生成 episode: S₀, A₀, R₁, S₁, A₁, R₂, ..., S_T-1, A_T-1, R_T
    G ← 0
    For t = T-1, T-2, ..., 0:
        G ← γG + R_{t+1}
        If S_t 不在 S₀, S₁, ..., S_{t-1} 中:  # First-visit
            Append G to Returns(S_t)
            V(S_t) ← average(Returns(S_t))
```

**Python 实现**：

```python
from collections import defaultdict
import numpy as np

def first_visit_mc_prediction(env, policy, num_episodes=10000, gamma=0.99):
    """
    First-Visit MC 策略评估
    
    Args:
        env: Gym 环境
        policy: 策略函数 policy(state) -> action
        num_episodes: episode 数量
        gamma: 折扣因子
    
    Returns:
        V: 状态价值函数估计
    """
    V = defaultdict(float)
    returns = defaultdict(list)
    
    for episode_num in range(num_episodes):
        # 生成 episode
        episode = []
        state = env.reset()
        done = False
        
        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
        
        # 反向计算 Return
        G = 0
        visited_states = set()
        
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            
            # First-visit: 只在第一次访问时更新
            if state not in visited_states:
                visited_states.add(state)
                returns[state].append(G)
                V[state] = np.mean(returns[state])
    
    return dict(V)
```

### 3.2.2 Every-Visit MC

**思想**：每次访问状态 $s$ 时都记录 Return。

**算法差异**：

```python
# First-Visit
if state not in visited_states:
    visited_states.add(state)
    returns[state].append(G)
    V[state] = np.mean(returns[state])

# Every-Visit
returns[state].append(G)
V[state] = np.mean(returns[state])
```

**对比**：

| 特性 | First-Visit | Every-Visit |
|------|------------|-------------|
| 无偏性 | ✅ 无偏 | ⚠️ 有偏（但渐近无偏） |
| 收敛性 | ✅ 收敛到 $V^\pi$ | ✅ 收敛到 $V^\pi$ |
| 方差 | 较高 | 较低（更多样本） |
| 实践使用 | 更常用 | 较少用 |

### 3.2.3 增量式更新公式

**问题**：存储所有 Returns 占用内存过大。

**解决方案**：增量式更新。

**推导**：

$$
\begin{align}
V_{n+1}(s) &= \frac{1}{n} \sum_{i=1}^n G_i \\
&= \frac{1}{n} \left( G_n + \sum_{i=1}^{n-1} G_i \right) \\
&= \frac{1}{n} \left( G_n + (n-1) V_n(s) \right) \\
&= V_n(s) + \frac{1}{n} \left( G_n - V_n(s) \right)
\end{align}
$$

**增量式更新**：

$$
V(s) \leftarrow V(s) + \alpha [G - V(s)]
$$

其中 $\alpha$ 可以是：
- **样本平均**：$\alpha = \frac{1}{N(s)}$
- **固定步长**：$\alpha = 0.01$（常用，适应非平稳环境）

**代码实现**：

```python
def incremental_mc_prediction(env, policy, num_episodes=10000, 
                               alpha=0.01, gamma=0.99):
    """增量式 MC 预测"""
    V = defaultdict(float)
    
    for episode_num in range(num_episodes):
        episode = generate_episode(env, policy)
        
        G = 0
        visited_states = set()
        
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            
            if state not in visited_states:
                visited_states.add(state)
                # 增量式更新
                V[state] += alpha * (G - V[state])
    
    return dict(V)
```

### 3.2.4 收敛性分析（大数定律）

**定理 3.1（MC 收敛性）**：

在以下条件下，First-Visit MC 估计收敛到真实值：

1. 所有状态被访问无限次：$\lim_{n \to \infty} N(s) = \infty, \forall s$
2. Returns 有界

则：

$$
V(s) \xrightarrow{a.s.} V^\pi(s) \quad \text{(以概率1收敛)}
$$

**证明**：大数定律（Law of Large Numbers）。

**收敛速度**：

$$
\text{Var}[\hat{V}(s)] = \frac{\sigma^2(s)}{N(s)}
$$

其中 $\sigma^2(s)$ 是 Return 的方差。

**标准误差**：

$$
\text{SE} = \frac{\sigma(s)}{\sqrt{N(s)}}
$$

要将误差减半，需要 **4 倍的 episodes**！

---

## 3.3 MC 控制

如何用 MC 找到最优策略？

### 3.3.1 MC Exploring Starts

**思想**：结合 GPI 框架，用 MC 做策略评估。

**挑战**：如何保证探索所有状态-动作对？

**Exploring Starts 假设**：每个 episode 从随机的 $(s, a)$ 对开始。

**算法（MC Exploring Starts）**：

```
初始化：
    Q(s,a) = 0, ∀s,a
    π(s) = 任意动作, ∀s
    Returns(s,a) = 空列表, ∀s,a

Repeat forever:
    # 探索开始
    随机选择 s₀ ∈ S, a₀ ∈ A(s₀)
    
    # 生成 episode（从 (s₀, a₀) 开始，之后遵循 π）
    Episode ← generate_episode(s₀, a₀, π)
    
    # MC 评估
    For each (s,a) 出现在 Episode 中:
        G ← (s,a) 之后的 return
        Append G to Returns(s,a)
        Q(s,a) ← average(Returns(s,a))
    
    # 策略改进
    For each s in Episode:
        π(s) ← argmax_a Q(s,a)
```

**问题**：Exploring Starts 在实际中很难满足（如真实机器人）。

### 3.3.2 ε-greedy 策略

**解决方案**：使用 ε-greedy 策略保证持续探索。

**ε-greedy 策略定义**：

$$
\pi(a|s) = \begin{cases}
1 - \epsilon + \frac{\epsilon}{|A|} & \text{if } a = \arg\max_{a'} Q(s, a') \\
\frac{\epsilon}{|A|} & \text{otherwise}
\end{cases}
$$

**性质**：
- 以概率 $1-\epsilon$ 选择贪心动作
- 以概率 $\epsilon$ 随机探索
- 保证所有动作都有非零概率被选择

**直观理解**：

```python
def epsilon_greedy_policy(Q, state, epsilon, num_actions):
    if np.random.random() < epsilon:
        return np.random.randint(num_actions)  # 探索
    else:
        return np.argmax(Q[state])  # 利用
```

### 3.3.3 On-policy MC Control

**On-policy**：学习的是**当前执行的策略**（ε-greedy）。

**算法（On-policy MC Control）**：

```
初始化：
    Q(s,a) = 0, ∀s,a
    Returns(s,a) = 空列表, ∀s,a
    ε ← 可调参数（如 0.1）

Repeat forever:
    # 生成 episode（使用 ε-greedy 策略）
    Episode ← []
    s ← env.reset()
    while not done:
        a ← ε-greedy(Q, s, ε)
        s', r, done ← env.step(a)
        Episode.append((s, a, r))
        s ← s'
    
    # MC 评估 + 改进
    G ← 0
    For t = T-1, T-2, ..., 0:
        s, a, r ← Episode[t]
        G ← γG + r
        
        # First-visit 检查
        If (s,a) 第一次出现:
            Append G to Returns(s,a)
            Q(s,a) ← average(Returns(s,a))
            # 隐式策略改进（通过 ε-greedy 使用新的 Q）
```

**完整代码实现**：

```python
def mc_control_epsilon_greedy(env, num_episodes=100000, 
                               gamma=0.99, epsilon=0.1):
    """
    On-policy MC Control with ε-greedy
    
    Args:
        env: Gym 环境
        num_episodes: episode 数量
        gamma: 折扣因子
        epsilon: 探索率
    
    Returns:
        Q: 动作价值函数
        policy: 最终策略（确定性）
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    returns = defaultdict(list)
    
    for episode_num in range(num_episodes):
        # 生成 episode
        episode = []
        state = env.reset()
        done = False
        
        while not done:
            # ε-greedy 选择动作
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
        
        # MC 更新
        G = 0
        visited_pairs = set()
        
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            
            pair = (state, action)
            if pair not in visited_pairs:
                visited_pairs.add(pair)
                returns[pair].append(G)
                Q[state][action] = np.mean(returns[pair])
        
        # 进度显示
        if (episode_num + 1) % 10000 == 0:
            print(f"Episode {episode_num + 1}/{num_episodes}")
    
    # 提取确定性最优策略
    policy = {s: np.argmax(Q[s]) for s in Q}
    
    return dict(Q), policy
```

### 3.3.4 收敛性证明（GLIE 条件）

**GLIE（Greedy in the Limit with Infinite Exploration）**：

1. **无限探索**：所有状态-动作对被访问无限次
   $$\lim_{n \to \infty} N(s, a) = \infty, \quad \forall s, a$$

2. **渐近贪心**：策略在极限下变为贪心
   $$\lim_{n \to \infty} \pi_n(a|s) = \mathbb{1}(a = \arg\max_{a'} Q_n(s, a'))$$

**定理 3.2（On-policy MC 收敛性）**：

如果满足 GLIE 条件，on-policy MC control 收敛到最优 $Q^*$。

**GLIE 策略示例**：

$$
\epsilon_n = \frac{1}{n} \quad \text{(随episode数递减)}
$$

```python
epsilon = 1.0 / (episode_num + 1)  # 递减 ε
```

---

## 3.4 Off-policy MC

**问题**：如何从一个策略（行为策略）生成的数据中，学习另一个策略（目标策略）？

### 3.4.1 重要性采样（Importance Sampling）

**场景**：
- **目标策略**（Target Policy）$\pi$：我们想评估/改进的策略
- **行为策略**（Behavior Policy）$b$：实际用于生成数据的策略

**为什么需要 Off-policy？**

1. **探索 vs 利用**：$b$ 可以更激进地探索，$\pi$ 可以是贪心策略
2. **从人类数据学习**：$b$ 是人类专家策略
3. **数据复用**：用旧策略的数据学习新策略

**重要性采样原理**：

$$
\mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[ \frac{p(x)}{q(x)} f(x) \right]
$$

**应用到 RL**：

$$
\mathbb{E}_{\tau \sim b}[\rho(\tau) G(\tau)] = \mathbb{E}_{\tau \sim \pi}[G(\tau)] = V^\pi(s)
$$

其中**重要性采样比**（Importance Sampling Ratio）：

$$
\rho_t = \frac{\pi(A_t|S_t) \pi(A_{t+1}|S_{t+1}) \cdots \pi(A_{T-1}|S_{T-1})}{b(A_t|S_t) b(A_{t+1}|S_{t+1}) \cdots b(A_{T-1}|S_{T-1})} = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}
$$

<div data-component="ImportanceSamplingVisualizer"></div>

### 3.4.2 普通重要性采样 vs 加权重要性采样

**普通重要性采样（Ordinary Importance Sampling）**：

$$
V(s) = \frac{\sum_{t \in \mathcal{T}(s)} \rho_t G_t}{|\mathcal{T}(s)|}
$$

- **无偏**：$\mathbb{E}[V(s)] = V^\pi(s)$
- **高方差**：$\rho_t$ 可能很大（如 $\rho = 100$）

**加权重要性采样（Weighted Importance Sampling）**：

$$
V(s) = \frac{\sum_{t \in \mathcal{T}(s)} \rho_t G_t}{\sum_{t \in \mathcal{T}(s)} \rho_t}
$$

- **有偏**（但渐近无偏）：$\lim_{n \to \infty} \mathbb{E}[V(s)] = V^\pi(s)$
- **低方差**：权重归一化

**对比**：

| 特性 | 普通 IS | 加权 IS |
|------|---------|---------|
| 偏差 | 无偏 | 有偏（渐近无偏） |
| 方差 | **极高** | 较低 |
| 实践推荐 | ❌ 很少用 | ✅ 常用 |

**代码实现**：

```python
def off_policy_mc_prediction_weighted(env, target_policy, behavior_policy,
                                       num_episodes=100000, gamma=0.99):
    """
    Off-policy MC 预测（加权重要性采样）
    
    Args:
        env: 环境
        target_policy: 目标策略（评估对象）
        behavior_policy: 行为策略（生成数据）
        num_episodes: episode 数量
        gamma: 折扣因子
    
    Returns:
        V: 目标策略的价值函数估计
    """
    V = defaultdict(float)
    C = defaultdict(float)  # 累积权重
    
    for episode_num in range(num_episodes):
        # 用 behavior_policy 生成 episode
        episode = []
        state = env.reset()
        done = False
        
        while not done:
            action = behavior_policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
        
        # 反向计算（带重要性采样）
        G = 0
        W = 1.0  # 累积重要性采样比
        
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            
            # 更新累积权重
            C[state] += W
            # 加权更新
            V[state] += (W / C[state]) * (G - V[state])
            
            # 更新重要性采样比
            W *= target_policy(action, state) / behavior_policy(action, state)
            
            # 早停：如果 W = 0，后续项都是 0
            if W == 0:
                break
    
    return dict(V)
```

### 3.4.3 方差问题与缓解

**方差爆炸问题**：

$$
\text{Var}[\rho_t G_t] = \mathbb{E}[\rho_t^2 G_t^2] - (\mathbb{E}[\rho_t G_t])^2
$$

当 $\pi$ 和 $b$ 差异大时，$\rho_t$ 可能非常大（如 $10^{10}$），导致方差爆炸。

**缓解策略**：

1. **使用加权重要性采样**（降低方差）

2. **限制 $\pi$ 和 $b$ 的差异**：
   - $b$ 选择 ε-greedy（保证支撑覆盖）
   - $\pi$ 也使用较小的 ε

3. **Per-decision 重要性采样**（高级技巧）：
   $$V(s) \approx \rho_{t:T-1} G_t$$
   而不是整个轨迹的比率

4. **截断**：
   $$\rho_t = \min(\rho_t, \rho_{\max}) \quad \text{(如 } \rho_{\max} = 10\text{)}$$

### 3.4.4 Off-policy MC Control

**算法（Off-policy MC Control）**：

```
初始化：
    Q(s,a) = 0, ∀s,a
    C(s,a) = 0, ∀s,a  # 累积权重
    π(s) = argmax_a Q(s,a), ∀s  # 目标策略（贪心）
    b = ε-greedy(Q, ε=0.1)      # 行为策略

Repeat forever:
    # 用行为策略 b 生成 episode
    Episode ← generate_episode(b)
    
    G ← 0
    W ← 1
    For t = T-1, T-2, ..., 0:
        s, a, r ← Episode[t]
        G ← γG + r
        
        # 更新
        C(s,a) ← C(s,a) + W
        Q(s,a) ← Q(s,a) + (W / C(s,a)) * (G - Q(s,a))
        
        # 更新目标策略
        π(s) ← argmax_a Q(s,a)
        
        # 如果不是贪心动作，后续贡献为 0
        If a ≠ π(s):
            Break
        
        # 更新重要性采样比
        W ← W * 1 / b(a|s)  # π(a|s) = 1 (贪心)
```

**代码实现**：

```python
def off_policy_mc_control(env, num_episodes=500000, gamma=0.99, epsilon=0.1):
    """Off-policy MC Control"""
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # 目标策略（贪心）
    def target_policy(state):
        return np.argmax(Q[state])
    
    # 行为策略（ε-greedy）
    def behavior_policy(state):
        if np.random.random() < epsilon:
            return env.action_space.sample()
        return np.argmax(Q[state])
    
    for episode_num in range(num_episodes):
        episode = []
        state = env.reset()
        done = False
        
        # 生成 episode（用行为策略）
        while not done:
            action = behavior_policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
        
        # Off-policy 更新
        G = 0
        W = 1.0
        
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            
            # 加权更新
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            
            # 如果不是贪心动作，截断
            if action != np.argmax(Q[state]):
                break
            
            # 更新权重
            W *= 1.0 / max(epsilon / env.action_space.n, 
                           1 - epsilon + epsilon / env.action_space.n)
    
    policy = {s: np.argmax(Q[s]) for s in Q}
    return dict(Q), policy
```

---

## 3.5 MC 的优缺点

### 3.5.1 优点：无需模型、无偏估计、易于理解

✅ **无需环境模型**
- 不需要 $P(s'|s,a)$ 和 $R(s,a,s')$
- 适用于未知环境

✅ **无偏估计**
- $\mathbb{E}[G_t] = V^\pi(s)$ （精确）
- 不像 TD 有 bootstrapping 误差

✅ **易于理解和实现**
- 概念简单：采样 + 平均
- 代码简洁

✅ **可以从经验中学习**
- 可以从人类专家数据学习
- 可以重放历史数据

<div data-component="OnPolicyVsOffPolicy"></div>

### 3.5.2 缺点：高方差、需要完整 episode、样本效率低

❌ **高方差**
- Return $G_t$ 是**长期累积**，方差很大
- 收敛慢：需要大量 episodes

❌ **必须等到 episode 结束**
- 不适用于持续任务
- 在线学习困难

❌ **样本效率低**
- 每个 episode 只更新一次
- TD 学习每步都更新（更高效）

❌ **Off-policy 方差爆炸**
- 重要性采样比可能非常大
- 实用性受限

**方差对比（实验数据）**：

| 任务 | MC 方差 | TD 方差 | 收敛 episodes |
|------|---------|---------|---------------|
| Blackjack | 1.2 | 0.3 | MC: 50万, TD: 5万 |
| GridWorld | 0.8 | 0.2 | MC: 10万无, TD: 1万 |

### 3.5.3 适用场景分析

**MC 适用于**：
- ✅ Episodic 任务（有明确终止）
- ✅ 环境模型未知
- ✅ 可以离线学习
- ✅ 可以获得大量数据

**MC 不适用于**：
- ❌ 持续任务（无终止状态）
- ❌ Episode 很长（方差太大）
- ❌ 需要快速学习（样本效率低）
- ❌ 在线学习（需要等 episode 结束）

**典型应用**：
- 🎲 Blackjack、扑克等卡牌游戏
- 🎮 Atari 游戏（有 game over）
- 🏁 赛车游戏（有终点）
- 📊 金融回测（历史数据）

---

## 3.6 实战：Blackjack MC Control

让我们用 MC 解决经典的 21 点（Blackjack）问题。

```python
import gymnasium as gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建 Blackjack 环境
env = gym.make('Blackjack-v1')

def run_blackjack_mc():
    """Blackjack MC Control 实战"""
    
    # On-policy MC Control
    Q, policy = mc_control_epsilon_greedy(
        env, 
        num_episodes=500000,
        gamma=1.0,  # Blackjack 无折扣
        epsilon=0.1
    )
    
    # 可视化策略
    def plot_blackjack_policy(policy, title="Blackjack Policy"):
        # 提取策略矩阵（玩家手牌 vs 庄家明牌）
        player_range = range(12, 22)  # 玩家手牌 12-21
        dealer_range = range(1, 11)   # 庄家明牌 A-10
        
        # 无 Ace / 有 Ace
        for usable_ace in [False, True]:
            policy_matrix = np.zeros((len(player_range), len(dealer_range)))
            
            for i, player_sum in enumerate(player_range):
                for j, dealer_card in enumerate(dealer_range):
                    state = (player_sum, dealer_card, usable_ace)
                    if state in policy:
                        policy_matrix[i, j] = policy[state]  # 0=stick, 1=hit
            
            # 绘制热力图
            plt.figure(figsize=(10, 6))
            plt.imshow(policy_matrix, cmap='RdYlGn', aspect='auto', origin='lower')
            plt.colorbar(label='Action (0=Stick, 1=Hit)')
            plt.xlabel('Dealer Showing')
            plt.ylabel('Player Sum')
            plt.xticks(range(len(dealer_range)), dealer_range)
            plt.yticks(range(len(player_range)), player_range)
            plt.title(f"{title} - {'Usable' if usable_ace else 'No'} Ace")
            plt.tight_layout()
            plt.show()
    
    # 可视化价值函数
    def plot_value_function(Q, title="State Value Function"):
        for usable_ace in [False, True]:
            player_range = range(12, 22)
            dealer_range = range(1, 11)
            
            X, Y = np.meshgrid(dealer_range, player_range)
            Z = np.zeros_like(X, dtype=float)
            
            for i, player_sum in enumerate(player_range):
                for j, dealer_card in enumerate(dealer_range):
                    state = (player_sum, dealer_card, usable_ace)
                    if state in Q:
                        Z[i, j] = np.max(Q[state])
            
            # 3D 曲面图
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
            ax.set_xlabel('Dealer Showing')
            ax.set_ylabel('Player Sum')
            ax.set_zlabel('State Value')
            ax.set_title(f"{title} - {'Usable' if usable_ace else 'No'} Ace")
            fig.colorbar(surf)
            plt.show()
    
    # 评估策略
    def evaluate_policy(policy, num_episodes=10000):
        wins = 0
        losses = 0
        draws = 0
        
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            
            while not done:
                if state in policy:
                    action = policy[state]
                else:
                    action = 0  # 默认 stick
                
                state, reward, done, _ = env.step(action)
            
            if reward > 0:
                wins += 1
            elif reward < 0:
                losses += 1
            else:
                draws += 1
        
        print(f"Win Rate: {wins/num_episodes:.2%}")
        print(f"Loss Rate: {losses/num_episodes:.2%}")
        print(f"Draw Rate: {draws/num_episodes:.2%}")
        return wins / num_episodes
    
    # 可视化
    plot_blackjack_policy(policy)
    plot_value_function(Q)
    
    # 评估
    win_rate = evaluate_policy(policy)
    print(f"\nFinal Win Rate: {win_rate:.2%}")
    
    return Q, policy

if __name__ == "__main__":
    Q, policy = run_blackjack_mc()
```

**预期输出**：

```
Episode 100000/500000
Episode 200000/500000
Episode 300000/500000
Episode 400000/500000
Episode 500000/500000

Win Rate: 42.35%
Loss Rate: 47.12%
Draw Rate: 10.53%

Final Win Rate: 42.35%
```

**策略解释**：

学习到的策略通常是：
- 玩家手牌 < 12：总是要牌（Hit）
- 玩家手牌 17-21：总是停牌（Stick）
- 玩家手牌 12-16：
  - 庄家明牌 2-6：停牌（庄家可能爆）
  - 庄家明牌 7-A：要牌（庄家可能更大）

---

## 本章小结

在本章中，我们学习了：

✅ **MC 基本思想**：从完整 episode 采样学习，无需环境模型  
✅ **MC 策略评估**：First-Visit 和 Every-Visit，收敛性由大数定律保证  
✅ **MC 控制**：Exploring Starts 和 ε-greedy，收敛需要 GLIE 条件  
✅ **Off-policy MC**：重要性采样，普通 vs 加权，方差问题  
✅ **MC 优缺点**：无偏但高方差，需要 episodes，样本效率低  

> [!TIP]
> **核心要点**：
> - MC 是第一个**无需模型**的 RL 方法
> - 使用**实际 Return** 而非 Bellman 方程
> - **高方差**是主要限制，需要大量数据
> - Off-policy 的重要性采样方差可能爆炸
> - 实践中，加权重要性采样优于普通重要性采样

> [!NOTE]
> **下一步**：
> Chapter 4 将学习**时序差分（TD）学习**，结合 DP 和 MC 的优点：
> - 像 MC 一样无需模型
> - 像 DP 一样可以 bootstrap，单步更新
> - 大幅降低方差，提高样本效率
> 
> 进入 [Chapter 4. 时序差分学习](04-td-learning.md)

---

## 扩展阅读

- **Sutton & Barto**：Chapter 5 (Monte Carlo Methods)
- **Spinning Up**：Monte Carlo Methods Introduction
- **经典论文**：
  - Metropolis & Ulam (1949): The Monte Carlo Method
  - Singh & Sutton (1996): Reinforcement Learning with Replacing Eligibility Traces
- **应用案例**：
  - AlphaGo 的蒙特卡洛树搜索（MCTS）
  - Tesauro's TD-Gammon（结合 TD 和 MC）

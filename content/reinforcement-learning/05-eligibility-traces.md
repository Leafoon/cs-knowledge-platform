---
title: "Chapter 5. 资格迹（Eligibility Traces）"
description: "统一 MC 和 TD：信用分配的优雅机制"
updated: "2026-01-29"
---

> **Learning Objectives**
> * 理解资格迹的动机：加速信用分配
> * 掌握 λ-return 和 TD(λ) 预测
> * 学习 SARSA(λ) 和 Watkins's Q(λ)
> * 理解前向视角和后向视角的等价性
> * 掌握资格迹的高效实现技巧

---

## 5.1 资格迹的动机

资格迹（Eligibility Traces）是强化学习中最优雅的机制之一，它解决了一个核心问题：**信用分配**（Credit Assignment）。

### 5.1.1 信用分配问题（Credit Assignment）

**问题场景**：

假设在一个游戏中：
1. 第 1 步：选择了一个关键动作 $A_1$
2. 第 2-99 步：执行了 98 个普通动作
3. 第 100 步：获得了大奖励 +1000

**问题**：如何将这个奖励正确地分配给第 1 步的关键动作？

**Monte Carlo 的解决方案**：
- ✅ 使用完整 Return，自动包含所有未来奖励
- ❌ 必须等到 episode 结束
- ❌ 高方差

**TD(0) 的问题**：
- ✅ 每步都可以更新
- ❌ 只看一步，奖励传播**非常慢**
  - 第 100 步：$V(S_{100}) \leftarrow V(S_{100}) + \alpha [R_{100} - V(S_{100})]$
  - 第 99 步：$V(S_{99}) \leftarrow V(S_{99}) + \alpha [R_{99} + \gamma V(S_{100}) - V(S_{99})]$
  - ...
  - 第 1 步：需要迭代 99 次才能感受到奖励！

**资格迹的解决方案**：
- ✅ 在**一次 episode** 中更新**所有相关状态**
- ✅ 根据状态的"资格"（eligibility）分配信用
- ✅ 比 MC 方差小，比 TD(0) 传播快

<div data-component="EligibilityTraceEvolution"></div>

### 5.1.2 前向视角 vs 后向视角

资格迹有两种等价的视角：

**前向视角（Forward View）**：
- 从当前状态向**未来**看
- 使用 λ-return：未来多步 return 的加权平均
- 概念清晰，但需要知道未来

**后向视角（Backward View）**：
- 从当前状态向**过去**看
- 使用资格迹向量：记录每个状态的"资格"
- 可以在线实现，高效

**等价性**：两种视角在理论上等价，但实现方式不同。

<div data-component="ForwardVsBackwardView"></div>

### 5.1.3 统一 MC 和 TD

资格迹提供了 MC 和 TD 之间的**连续谱**：

- $\lambda = 0$：纯 TD(0)
- $\lambda = 1$：近似 MC
- $0 < \lambda < 1$：折中方案

**参数 λ 的作用**：
- 控制资格迹的**衰减速度**
- 控制信用分配的**范围**

---

## 5.2 λ-return

λ-return 是资格迹的核心概念，它是多步 return 的加权平均。

### 5.2.1 n-step return 的加权平均

**回顾 n-step return**：

$$
G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})
$$

**λ-return 定义**：

$$
G_t^\lambda = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}
$$

**展开**：

$$
G_t^\lambda = (1-\lambda)(G_t^{(1)} + \lambda G_t^{(2)} + \lambda^2 G_t^{(3)} + \cdots)
$$

**权重归一化**：

$$
(1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} = (1-\lambda) \cdot \frac{1}{1-\lambda} = 1
$$

<div data-component="LambdaReturnWeighting"></div>

### 5.2.2 λ 参数的作用（0 ≤ λ ≤ 1）

**特殊值**：

- **λ = 0**：
  $$G_t^{\lambda=0} = G_t^{(1)} = R_{t+1} + \gamma V(S_{t+1})$$
  纯 TD(0)

- **λ = 1**：
  $$G_t^{\lambda=1} = G_t^{(\infty)} = R_{t+1} + \gamma R_{t+2} + \cdots$$
  Monte Carlo Return（假设 episode 终止）

- **0 < λ < 1**：平衡偏差和方差

**λ 的几何直觉**：

```python
# λ-return 的权重分布
import numpy as np
import matplotlib.pyplot as plt

lambda_vals = [0.0, 0.5, 0.9, 0.99]
n_steps = 20

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, lam in enumerate(lambda_vals):
    weights = [(1 - lam) * (lam ** (n-1)) for n in range(1, n_steps+1)]
    
    axes[idx].bar(range(1, n_steps+1), weights, alpha=0.7, color='steelblue')
    axes[idx].set_title(f'λ = {lam}', fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('n-step')
    axes[idx].set_ylabel('Weight')
    axes[idx].grid(alpha=0.3)
    axes[idx].set_ylim(0, 1.1)

plt.tight_layout()
plt.show()
```

**观察**：
- λ 越小，权重越集中在近期（低方差，高偏差）
- λ 越大，权重越分散到远期（高方差，低偏差）

### 5.2.3 几何加权的合理性

**为什么用几何加权？**

1. **简单性**：权重呈指数衰减，容易计算
2. **马尔可夫性**：符合折扣的哲学（近期重要，远期衰减）
3. **数学优雅**：权重自动归一化
4. **实践效果**：在大量任务中表现良好

**其他加权方案**（较少用）：
- 均匀加权：$w_n = \frac{1}{N}$
- 线性衰减：$w_n = \frac{N - n + 1}{\sum_i i}$

---

## 5.3 TD(λ) 预测

TD(λ) 是 TD(0) 的推广，使用资格迹向量实现高效的 λ-return 估计。

### 5.3.1 资格迹向量 e<sub>t</sub>(s)

**定义**：

资格迹 $e_t(s)$ 记录状态 $s$ 在时刻 $t$ 的"资格"，用于决定该状态应该获得多少信用。

**累积迹（Accumulating Trace）**：

$$
e_t(s) = \begin{cases}
\gamma \lambda e_{t-1}(s) + 1 & \text{if } S_t = s \\
\gamma \lambda e_{t-1}(s) & \text{otherwise}
\end{cases}
$$

**替换迹（Replacing Trace）**：

$$
e_t(s) = \begin{cases}
1 & \text{if } S_t = s \\
\gamma \lambda e_{t-1}(s) & \text{otherwise}
\end{cases}
$$

**直觉**：
- 访问一个状态 → 提高其资格
- 时间流逝 → 资格按 $\gamma \lambda$ 衰减

### 5.3.2 累积迹 vs 替换迹

| 特性 | 累积迹 | 替换迹 |
|------|--------|--------|
| 重复访问 | 累加资格 | 重置为 1 |
| 理论分析 | 更常用 | 实践中常用 |
| 收敛性 | 理论保证 | 实验效果好 |
| 适用场景 | 表格方法 | 函数逼近 |

**示例**：

假设轨迹为 $S_1, S_2, S_1, S_3$（$S_1$ 被访问两次）：

**累积迹**：
```
t=1: e(S₁)=1,    e(S₂)=0,    e(S₃)=0
t=2: e(S₁)=γλ,   e(S₂)=1,    e(S₃)=0
t=3: e(S₁)=γλ·γλ+1, e(S₂)=γλ, e(S₃)=0  # S₁ 累加！
t=4: e(S₁)=γλ(...), e(S₂)=(γλ)², e(S₃)=1
```

**替换迹**：
```
t=1: e(S₁)=1,    e(S₂)=0,    e(S₃)=0
t=2: e(S₁)=γλ,   e(S₂)=1,    e(S₃)=0
t=3: e(S₁)=1,     e(S₂)=γλ,  e(S₃)=0  # S₁ 重置为 1！
t=4: e(S₁)=γλ,    e(S₂)=(γλ)², e(S₃)=1
```

### 5.3.3 TD(λ) 更新规则

**算法（TD(λ) 预测 - 后向视角）**：

```
输入：策略 π，参数 α, γ, λ
初始化 V(s) = 0, ∀s

For each episode:
    初始化 e(s) = 0, ∀s
    初始化 S
    
    For each step of episode:
        A ← 从 π(·|S) 采样
        执行 A，观察 R, S'
        
        δ ← R + γV(S') - V(S)  # TD error
        e(S) ← e(S) + 1         # 累积迹（或 e(S) ← 1 替换迹）
        
        For all s:
            V(s) ← V(s) + α δ e(s)  # 更新所有状态！
            e(s) ← γλ e(s)           # 衰减资格迹
        
        S ← S'
```

**关键点**：
1. **TD error** $\delta$ 只计算一次（当前状态）
2. **所有状态**根据其资格迹 $e(s)$ 更新
3. 资格迹每步衰减 $\gamma \lambda$

**Python 实现**：

```python
def td_lambda_prediction(env, policy, num_episodes=1000, 
                         alpha=0.1, gamma=0.99, lambda_=0.9):
    """
    TD(λ) 策略评估（累积迹）
    
    Args:
        env: Gym 环境
        policy: 策略函数 policy(state) -> action
        num_episodes: episode 数量
        alpha: 学习率
        gamma: 折扣因子
        lambda_: λ 参数
    
    Returns:
        V: 价值函数估计
    """
    V = defaultdict(float)
    
    for episode in range(num_episodes):
        # 初始化资格迹
        e = defaultdict(float)
        
        state = env.reset()
        done = False
        
        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            
            # TD error
            td_target = reward + gamma * V[next_state] * (1 - done)
            delta = td_target - V[state]
            
            # 更新当前状态的资格迹（累积迹）
            e[state] += 1.0
            
            # 更新所有状态的价值和资格迹
            for s in list(e.keys()):
                V[s] += alpha * delta * e[s]
                e[s] *= gamma * lambda_
                
                # 清理接近零的资格迹（优化）
                if e[s] < 1e-5:
                    del e[s]
            
            state = next_state
    
    return dict(V)
```

### 5.3.4 在线 vs 离线 λ-return

**离线 λ-return**（前向视角）：
- 需要完整 episode 才能计算
- 理论上清晰
- 不能在线学习

**在线 TD(λ)**（后向视角）：
- 每步都可以更新
- 使用资格迹向量
- 可以在线学习

**True Online TD(λ)**（van Seijen & Sutton, 2014）：
- 更精确的在线实现
- 考虑了价值函数的变化
- 性能更好，但实现复杂

---

## 5.4 SARSA(λ)

SARSA(λ) 将资格迹应用到控制问题（学习动作价值函数 $Q$）。

### 5.4.1 动作价值的资格迹

**资格迹扩展到 Q(s,a)**：

$$
e_t(s,a) = \begin{cases}
\gamma \lambda e_{t-1}(s,a) + 1 & \text{if } S_t = s, A_t = a \\
\gamma \lambda e_{t-1}(s,a) & \text{otherwise}
\end{cases}
$$

**替换迹**：

$$
e_t(s,a) = \begin{cases}
1 & \text{if } S_t = s, A_t = a \\
\gamma \lambda e_{t-1}(s,a) & \text{otherwise}
\end{cases}
$$

### 5.4.2 SARSA(λ) 算法

**算法（SARSA(λ) - 累积迹）**：

```
输入：α, γ, λ, ε
初始化 Q(s,a) = 0, ∀s,a

For each episode:
    初始化 e(s,a) = 0, ∀s,a
    初始化 S, A ← ε-greedy(Q, S)
    
    For each step of episode:
        执行 A，观察 R, S'
        A' ← ε-greedy(Q, S')
        
        δ ← R + γQ(S',A') - Q(S,A)  # SARSA TD error
        e(S,A) ← e(S,A) + 1          # 累积迹
        
        For all s, a:
            Q(s,a) ← Q(s,a) + α δ e(s,a)
            e(s,a) ← γλ e(s,a)
        
        S ← S'; A ← A'
```

**完整代码实现**：

```python
def sarsa_lambda(env, num_episodes=5000, alpha=0.1, gamma=0.99, 
                 lambda_=0.9, epsilon=0.1):
    """
    SARSA(λ) 算法（累积迹）
    
    Args:
        env: Gym 环境
        num_episodes: episode 数量
        alpha: 学习率
        gamma: 折扣因子
        lambda_: λ 参数
        epsilon: ε-greedy 探索率
    
    Returns:
        Q: 动作价值函数
        policy: 学习到的策略
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    for episode in range(num_episodes):
        # 初始化资格迹
        e = defaultdict(lambda: np.zeros(env.action_space.n))
        
        state = env.reset()
        
        # 选择初始动作（ε-greedy）
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        done = False
        while not done:
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 选择下一个动作（ε-greedy）
            if np.random.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])
            
            # SARSA TD error
            td_target = reward + gamma * Q[next_state][next_action] * (1 - done)
            delta = td_target - Q[state][action]
            
            # 更新当前状态-动作的资格迹（累积迹）
            e[state][action] += 1.0
            
            # 更新所有状态-动作对
            states_to_update = list(e.keys())
            for s in states_to_update:
                for a in range(env.action_space.n):
                    if e[s][a] > 1e-5:  # 只更新有显著资格的
                        Q[s][a] += alpha * delta * e[s][a]
                        e[s][a] *= gamma * lambda_
                    else:
                        e[s][a] = 0  # 清理
            
            state = next_state
            action = next_action
        
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{num_episodes}")
    
    # 提取策略
    policy = {s: np.argmax(Q[s]) for s in Q}
    return dict(Q), policy
```

### 5.4.3 True Online SARSA(λ)

**问题**：标准 SARSA(λ) 使用的是**旧的** Q 值计算 TD error。

**True Online SARSA(λ)**（van Seijen & Sutton, 2014）：

考虑 Q 值的变化，使用**修正的资格迹**：

$$
\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)
$$

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \delta_t e_t(s,a) + \alpha (Q_{old}(S_t, A_t) - Q(S_t, A_t))(e_t(s,a) - 1)
$$

**优点**：
- 更精确的在线逼近 λ-return
- 更好的性能（特别是 $\lambda$ 接近 1 时）

**缺点**：
- 实现复杂
- 计算开销略高

**实践建议**：
- 标准 SARSA(λ) 通常足够好
- 如果追求最优性能，使用 True Online 版本

---

## 5.5 Watkins's Q(λ) 与 Off-policy 资格迹

Off-policy 学习中的资格迹更复杂，因为行为策略和目标策略不同。

### 5.5.1 Off-policy 资格迹的挑战

**问题**：

在 Off-policy 中：
- 行为策略 $b$ 生成数据
- 目标策略 $\pi$ 是学习目标

**SARSA(λ) 的问题**：
- SARSA(λ) 是 On-policy，使用 $A'$ 来计算 TD target
- 如果 $A'$ 不是贪心动作，资格迹应该如何处理？

### 5.5.2 Watkins's Q(λ) 解决方案

**Watkins's Q(λ)**（Watkins, 1989）：

当执行**非贪心动作**时，**截断资格迹**。

**算法**：

```
For each step:
    δ ← R + γ max_a Q(S',a) - Q(S,A)  # Q-learning style
    e(S,A) ← e(S,A) + 1
    
    For all s,a:
        Q(s,a) ← Q(s,a) + α δ e(s,a)
        
        if A ≠ argmax_a Q(S,a):  # 非贪心动作
            e(s,a) ← 0  # 截断资格迹！
        else:
            e(s,a) ← γλ e(s,a)
```

**直觉**：
- 如果采取贪心动作 → 继续累积资格迹
- 如果采取探索动作 → 重置资格迹（因为偏离了目标策略）

**代码实现**：

```python
def watkins_q_lambda(env, num_episodes=5000, alpha=0.1, gamma=0.99,
                     lambda_=0.9, epsilon=0.1):
    """
    Watkins's Q(λ) 算法
    
    特点：Off-policy，当采取非贪心动作时截断资格迹
    """
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    for episode in range(num_episodes):
        e = defaultdict(lambda: np.zeros(env.action_space.n))
        
        state = env.reset()
        done = False
        
        while not done:
            # ε-greedy 选择动作
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, done, _ = env.step(action)
            
            # Q-learning style TD error
            td_target = reward + gamma * np.max(Q[next_state]) * (1 - done)
            delta = td_target - Q[state][action]
            
            # 更新资格迹
            e[state][action] += 1.0
            
            # 判断是否是贪心动作
            greedy_action = np.argmax(Q[state])
            is_greedy = (action == greedy_action)
            
            # 更新所有状态-动作对
            for s in list(e.keys()):
                for a in range(env.action_space.n):
                    if e[s][a] > 1e-5:
                        Q[s][a] += alpha * delta * e[s][a]
                        
                        if is_greedy:
                            e[s][a] *= gamma * lambda_
                        else:
                            e[s][a] = 0  # 截断！
            
            state = next_state
    
    return dict(Q)
```

### 5.5.3 资格迹截断的影响

**优点**：
- 理论上正确（Off-policy 学习）
- 避免错误的信用分配

**缺点**：
- 截断过于激进
- 当 $\epsilon$ 较大时，资格迹频繁被重置
- 实际上接近 Q-learning（$\lambda$ 效果不明显）

**改进方案**：
- **Peng's Q(λ)**：不截断，但使用重要性采样权重
- **Tree-Backup(λ)**：使用期望更新，避免截断
- **Q*(λ)**：混合策略

---

## 5.6 资格迹的实现技巧

高效实现资格迹对于大规模问题至关重要。

### 5.6.1 稀疏表示

**问题**：

在大状态空间中，大多数状态的资格迹为 0 或接近 0。

**解决方案**：使用稀疏字典

```python
# ❌ 密集表示（低效）
e = np.zeros(num_states)

# ✅ 稀疏表示（高效）
e = {}  # 只存储非零元素

# 更新
e[state] = e.get(state, 0.0) + 1.0

# 衰减并清理
to_delete = []
for s in e:
    e[s] *= gamma * lambda_
    if e[s] < 1e-5:
        to_delete.append(s)

for s in to_delete:
    del e[s]
```

### 5.6.2 衰减策略

**阈值清理**：

```python
# 设置阈值
TRACE_THRESHOLD = 1e-5

# 清理小于阈值的迹
e = {s: val for s, val in e.items() if val >= TRACE_THRESHOLD}
```

**固定大小迹**：

```python
# 只保留前 K 个最大的迹
K = 100
if len(e) > K:
    top_k = sorted(e.items(), key=lambda x: x[1], reverse=True)[:K]
    e = dict(top_k)
```

### 5.6.3 计算效率优化

**批量更新**：

```python
# ❌ 慢：逐个更新
for s in all_states:
    V[s] += alpha * delta * e[s]

# ✅ 快：只更新有资格的状态
for s in e.keys():
    V[s] += alpha * delta * e[s]
```

**向量化（使用 NumPy）**：

```python
# 对于表格方法，使用数组
e = np.zeros(num_states)
V = np.zeros(num_states)

# 向量化更新
V += alpha * delta * e
e *= gamma * lambda_
```

**函数逼近中的资格迹**（预告 Chapter 6）：

```python
# 梯度资格迹
e_w = np.zeros_like(w)  # 参数向量的资格迹

# 更新
e_w = gamma * lambda_ * e_w + grad_V(state, w)
w += alpha * delta * e_w
```

---

## 5.7 实战：Mountain Car with SARSA(λ)

Mountain Car 是测试资格迹效果的经典任务。

```python
import gymnasium as gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def discretize_state(state, bins=20):
    """
    将连续状态离散化
    
    Mountain Car 状态：
    - position: [-1.2, 0.6]
    - velocity: [-0.07, 0.07]
    """
    position, velocity = state
    
    pos_bins = np.linspace(-1.2, 0.6, bins)
    vel_bins = np.linspace(-0.07, 0.07, bins)
    
    pos_idx = np.digitize(position, pos_bins)
    vel_idx = np.digitize(velocity, vel_bins)
    
    return (pos_idx, vel_idx)

def run_mountain_car_sarsa_lambda():
    """
    Mountain Car 任务：SARSA(λ) vs SARSA(0)
    """
    env = gym.make('MountainCar-v0')
    
    # 离散化参数
    bins = 20
    
    algorithms = {
        'SARSA(0)': {'lambda': 0.0},
        'SARSA(0.5)': {'lambda': 0.5},
        'SARSA(0.9)': {'lambda': 0.9},
        'SARSA(0.99)': {'lambda': 0.99},
    }
    
    results = {}
    
    for name, params in algorithms.items():
        print(f"\nRunning {name}...")
        
        lambda_ = params['lambda']
        Q = defaultdict(lambda: np.zeros(env.action_space.n))
        
        episode_lengths = []
        
        for episode in range(500):
            # 初始化资格迹
            e = defaultdict(lambda: np.zeros(env.action_space.n))
            
            state = env.reset()[0]
            state_disc = discretize_state(state, bins)
            
            # ε-greedy（递减）
            epsilon = max(0.01, 1.0 - episode / 200)
            
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state_disc])
            
            done = False
            steps = 0
            
            while not done and steps < 1000:
                # 执行动作
                next_state, reward, done, truncated, _ = env.step(action)
                next_state_disc = discretize_state(next_state, bins)
                done = done or truncated
                
                # 下一个动作
                if np.random.random() < epsilon:
                    next_action = env.action_space.sample()
                else:
                    next_action = np.argmax(Q[next_state_disc])
                
                # SARSA TD error
                td_target = reward + 0.99 * Q[next_state_disc][next_action] * (1 - done)
                delta = td_target - Q[state_disc][action]
                
                # 更新资格迹
                e[state_disc][action] += 1.0
                
                # 更新 Q 和资格迹（只更新有显著迹的）
                for s in list(e.keys()):
                    for a in range(env.action_space.n):
                        if e[s][a] > 1e-5:
                            Q[s][a] += 0.1 * delta * e[s][a]
                            e[s][a] *= 0.99 * lambda_
                
                state_disc = next_state_disc
                action = next_action
                steps += 1
            
            episode_lengths.append(steps)
            
            if (episode + 1) % 100 == 0:
                avg_length = np.mean(episode_lengths[-100:])
                print(f"Episode {episode + 1}, Avg Length: {avg_length:.1f}")
        
        results[name] = episode_lengths
    
    # 绘制学习曲线
    plt.figure(figsize=(12, 6))
    for name, lengths in results.items():
        # 平滑曲线
        smoothed = np.convolve(lengths, np.ones(10)/10, mode='valid')
        plt.plot(smoothed, label=name, linewidth=2)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Episode Length (steps)', fontsize=12)
    plt.title('Mountain Car: SARSA(λ) with Different λ Values', 
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    env.close()

if __name__ == "__main__":
    run_mountain_car_sarsa_lambda()
```

**预期结果**：

- **SARSA(0)**：学习最慢（约 400 episodes 才收敛）
- **SARSA(0.5)**：显著加速（约 250 episodes）
- **SARSA(0.9)**：更快（约 150 episodes）
- **SARSA(0.99)**：最快，但可能不稳定

**关键观察**：
1. **λ 越大，学习越快**（直到某个点）
2. **λ 太接近 1 可能导致高方差**
3. **最优 λ 通常在 0.8-0.95 之间**

---

## 本章小结

在本章中，我们学习了：

✅ **资格迹动机**：加速信用分配，统一 MC 和 TD  
✅ **λ-return**：多步 return 的几何加权平均  
✅ **TD(λ)**：使用资格迹向量的高效实现  
✅ **SARSA(λ)**：资格迹在控制问题中的应用  
✅ **Off-policy 资格迹**：Watkins's Q(λ) 和截断机制  
✅ **实现技巧**：稀疏表示、阈值清理、计算优化  

> [!TIP]
> **核心要点**：
> - 资格迹是 RL 中最优雅的机制之一
> - λ 参数控制偏差-方差权衡
> - 后向视角（资格迹）等价于前向视角（λ-return）
> - 实践中 λ ∈ [0.8, 0.95] 通常效果最好
> - Off-policy 资格迹需要特殊处理（截断或重要性采样）

> [!NOTE]
> **下一步**：
> Chapter 6 将学习**函数逼近（Function Approximation）**，这是处理大规模/连续状态空间的关键：
> - 线性函数逼近和特征工程
> - 深度神经网络作为通用逼近器
> - Semi-gradient 方法和 Deadly Triad
> - 为深度强化学习（DQN）打下基础
> 
> 进入 [Chapter 6. 函数逼近](06-function-approximation.md)

---

## 扩展阅读

- **Sutton & Barto**：Chapter 12 (Eligibility Traces)
- **经典论文**：
  - Sutton (1988): Learning to Predict by the Methods of Temporal Differences
  - Van Seijen & Sutton (2014): True Online TD(λ)
  - Watkins (1989): Learning from Delayed Rewards (博士论文)
- **应用案例**：
  - TD-Gammon（Tesauro, 1995）：世界级 Backgammon 程序
  - Mountain Car 基准测试

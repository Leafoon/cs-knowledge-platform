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

策略评估是动态规划的核心操作之一，也是所有强化学习算法的基础。本节将深入探讨策略评估的理论基础、算法实现和优化技术。

---

### 2.1.1 问题形式化

给定 MDP $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$ 和策略 $\pi$，我们需要求解 Bellman 期望方程：

$$
V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} \mathcal{P}(s'|s,a) \left[ \mathcal{R}(s,a) + \gamma V^\pi(s') \right], \quad \forall s \in \mathcal{S}
$$

这是一个**线性方程组**，包含 $|\mathcal{S}|$ 个方程和 $|\mathcal{S}|$ 个未知数。

**矩阵形式**

定义向量 $\mathbf{V}^\pi \in \mathbb{R}^{|\mathcal{S}|}$ 和矩阵 $\mathbf{P}^\pi, \mathbf{R}^\pi$：

$$
[\mathbf{P}^\pi]_{ss'} = \sum_a \pi(a|s) \mathcal{P}(s'|s,a)
$$

$$
[\mathbf{R}^\pi]_s = \sum_a \pi(a|s) \mathcal{R}(s,a)
$$

则 Bellman 方程可写为：

$$
\mathbf{V}^\pi = \mathbf{R}^\pi + \gamma \mathbf{P}^\pi \mathbf{V}^\pi
$$

**直接求解**

理论上可以直接求解：

$$
\mathbf{V}^\pi = (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} \mathbf{R}^\pi
$$

但矩阵求逆的时间复杂度为 $O(|\mathcal{S}|^3)$，对于大规模问题不可行。

---

### 2.1.2 Bellman 期望算子

定义 **Bellman 期望算子** $T^\pi: \mathbb{R}^{|\mathcal{S}|} \to \mathbb{R}^{|\mathcal{S}|}$：

$$
[T^\pi V](s) = \sum_a \pi(a|s) \sum_{s'} \mathcal{P}(s'|s,a) \left[ \mathcal{R}(s,a) + \gamma V(s') \right]
$$

则 Bellman 方程等价于寻找 $T^\pi$ 的不动点：

$$
V^\pi = T^\pi V^\pi
$$

**算子的性质**

**性质 1：单调性 (Monotonicity)**

如果 $V_1(s) \leq V_2(s), \forall s$，则 $[T^\pi V_1](s) \leq [T^\pi V_2](s), \forall s$。

**证明**：
$$
\begin{aligned}
[T^\pi V_1](s) &= \sum_a \pi(a|s) \sum_{s'} \mathcal{P}(s'|s,a) [\mathcal{R}(s,a) + \gamma V_1(s')] \\
&\leq \sum_a \pi(a|s) \sum_{s'} \mathcal{P}(s'|s,a) [\mathcal{R}(s,a) + \gamma V_2(s')] \\
&= [T^\pi V_2](s)
\end{aligned}
$$

**性质 2：$\gamma$-收缩性 (Contraction)**

对于任意 $V_1, V_2$，有：

$$
\| T^\pi V_1 - T^\pi V_2 \|_\infty \leq \gamma \| V_1 - V_2 \|_\infty
$$

其中 $\|V\|_\infty = \max_s |V(s)|$ 是无穷范数。

**证明**：

$$
\begin{aligned}
|[T^\pi V_1](s) - [T^\pi V_2](s)| &= \left| \sum_a \pi(a|s) \sum_{s'} \mathcal{P}(s'|s,a) \gamma [V_1(s') - V_2(s')] \right| \\
&\leq \sum_a \pi(a|s) \sum_{s'} \mathcal{P}(s'|s,a) \gamma |V_1(s') - V_2(s')| \\
&\leq \gamma \|V_1 - V_2\|_\infty \sum_a \pi(a|s) \sum_{s'} \mathcal{P}(s'|s,a) \\
&= \gamma \|V_1 - V_2\|_\infty
\end{aligned}
$$

对所有 $s$ 取最大值：

$$
\| T^\pi V_1 - T^\pi V_2 \|_\infty \leq \gamma \| V_1 - V_2 \|_\infty
$$

---

### 2.1.3 Banach 不动点定理与收敛性

**Banach 不动点定理 (Banach Fixed-Point Theorem)**

设 $(\mathcal{X}, d)$ 是完备度量空间，$T: \mathcal{X} \to \mathcal{X}$ 是收缩映射，即存在 $0 \leq \gamma < 1$ 使得：

$$
d(T(x), T(y)) \leq \gamma \cdot d(x, y), \quad \forall x, y \in \mathcal{X}
$$

则：
1. $T$ 有**唯一不动点** $x^* \in \mathcal{X}$，即 $T(x^*) = x^*$
2. 对任意初始点 $x_0 \in \mathcal{X}$，迭代序列 $x_{k+1} = T(x_k)$ 收敛到 $x^*$
3. 收敛速率为几何级数：$d(x_k, x^*) \leq \gamma^k d(x_0, x^*)$

**应用到策略评估**

- 空间：$\mathcal{X} = \mathbb{R}^{|\mathcal{S}|}$，度量：$d(V_1, V_2) = \|V_1 - V_2\|_\infty$
- 算子：$T = T^\pi$
- 收缩系数：$\gamma$（折扣因子）

由于 $T^\pi$ 是 $\gamma$-收缩映射（$\gamma < 1$），根据 Banach 定理：

1. **唯一性**：$V^\pi$ 是唯一解
2. **收敛性**：迭代 $V_{k+1} = T^\pi V_k$ 必定收敛到 $V^\pi$
3. **收敛速率**：$\|V_k - V^\pi\|_\infty \leq \gamma^k \|V_0 - V^\pi\|_\infty$

**收敛界估计**

设初始误差上界为 $\epsilon_0 = \|V_0 - V^\pi\|_\infty$，要达到精度 $\epsilon$，需要迭代次数：

$$
k \geq \frac{\log(\epsilon / \epsilon_0)}{\log \gamma}
$$

例如，$\gamma = 0.9$，$\epsilon_0 = 100$，$\epsilon = 0.01$：

$$
k \geq \frac{\log(0.01 / 100)}{\log 0.9} = \frac{\log 10^{-4}}{\log 0.9} \approx \frac{-9.21}{-0.105} \approx 88 \text{ 次迭代}
$$

---

### 2.1.4 迭代策略评估算法

将 Bellman 方程转化为迭代更新规则：

$$
V_{k+1}(s) \leftarrow \sum_a \pi(a|s) \sum_{s'} \mathcal{P}(s'|s,a) \left[ \mathcal{R}(s,a) + \gamma V_k(s') \right]
$$

**算法伪代码**

```
Algorithm: Iterative Policy Evaluation
Input: MDP (S, A, P, R, γ), policy π, threshold θ
Output: Value function V^π

1. Initialize V(s) = 0 for all s ∈ S
2. repeat
3.     Δ ← 0
4.     for each s ∈ S do
5.         v ← V(s)
6.         V(s) ← Σ_a π(a|s) Σ_s' P(s'|s,a) [R(s,a) + γ V(s')]
7.         Δ ← max(Δ, |v - V(s)|)
8.     end for
9. until Δ < θ
10. return V
```

**时间复杂度**：每次迭代 $O(|\mathcal{S}|^2 |\mathcal{A}|)$，总复杂度 $O(|\mathcal{S}|^2 |\mathcal{A}| \cdot k)$，其中 $k$ 是迭代次数。

---

### 2.1.5 算法变体

**Two-Array vs In-Place 更新**

- **Two-Array**：使用两个数组 $V_{old}$ 和 $V_{new}$，所有状态基于 $V_{old}$ 更新到 $V_{new}$
- **In-Place**：只用一个数组，更新时立即使用新值

In-place 更新通常收敛更快，因为新值立即被后续状态使用。

**Gauss-Seidel vs Jacobi 迭代**

- **Jacobi**：所有状态并行更新（Two-Array）
- **Gauss-Seidel**：顺序更新，立即使用新值（In-Place）

Gauss-Seidel 通常收敛速度快 2 倍左右。

---

### 2.1.6 完整代码实现

```python
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt

class PolicyEvaluator:
    """策略评估器"""
    
    def __init__(self, mdp, method='in-place'):
        """
        Args:
            mdp: MDP 实例
            method: 'in-place' 或 'two-array'
        """
        self.mdp = mdp
        self.method = method
        
    def evaluate(self, policy: np.ndarray, theta: float = 1e-6, 
                 max_iterations: int = 1000) -> Tuple[np.ndarray, List[float]]:
        """
        迭代策略评估
        
        Args:
            policy: 策略，shape [num_states] (确定性) 或 [num_states, num_actions] (随机)
            theta: 收敛阈值
            max_iterations: 最大迭代次数
            
        Returns:
            V: 状态价值函数
            deltas: 每次迭代的最大误差
        """
        V = np.zeros(self.mdp.num_states)
        deltas = []
        
        # 判断策略类型
        is_stochastic = (policy.ndim == 2)
        
        for iteration in range(max_iterations):
            if self.method == 'two-array':
                V_new = np.zeros_like(V)
            
            delta = 0
            
            for s in range(self.mdp.num_states):
                v_old = V[s]
                
                # 计算期望价值
                v_new = 0
                
                if is_stochastic:
                    # 随机策略
                    for a in range(self.mdp.num_actions):
                        if policy[s, a] > 0:
                            q_sa = self._compute_q_value(s, a, V)
                            v_new += policy[s, a] * q_sa
                else:
                    # 确定性策略
                    a = policy[s]
                    v_new = self._compute_q_value(s, a, V)
                
                if self.method == 'two-array':
                    V_new[s] = v_new
                else:
                    V[s] = v_new
                
                delta = max(delta, abs(v_old - v_new))
            
            if self.method == 'two-array':
                V = V_new
            
            deltas.append(delta)
            
            if delta < theta:
                print(f"Converged in {iteration + 1} iterations")
                break
        
        return V, deltas
    
    def _compute_q_value(self, s: int, a: int, V: np.ndarray) -> float:
        """计算 Q(s, a)"""
        q = 0
        for s_prime in range(self.mdp.num_states):
            prob = self.mdp.P[s, a, s_prime]
            if prob > 0:
                reward = self.mdp.R[s, a]
                q += prob * (reward + self.mdp.gamma * V[s_prime])
        return q
    
    def evaluate_matrix(self, policy: np.ndarray) -> np.ndarray:
        """
        使用矩阵求逆直接求解（仅用于小规模问题）
        V^π = (I - γP^π)^{-1} R^π
        """
        # 构建 P^π 和 R^π
        P_pi = np.zeros((self.mdp.num_states, self.mdp.num_states))
        R_pi = np.zeros(self.mdp.num_states)
        
        is_stochastic = (policy.ndim == 2)
        
        for s in range(self.mdp.num_states):
            if is_stochastic:
                for a in range(self.mdp.num_actions):
                    if policy[s, a] > 0:
                        R_pi[s] += policy[s, a] * self.mdp.R[s, a]
                        for s_prime in range(self.mdp.num_states):
                            P_pi[s, s_prime] += policy[s, a] * self.mdp.P[s, a, s_prime]
            else:
                a = policy[s]
                R_pi[s] = self.mdp.R[s, a]
                P_pi[s, :] = self.mdp.P[s, a, :]
        
        # 求解 V = (I - γP)^{-1} R
        I = np.eye(self.mdp.num_states)
        V = np.linalg.solve(I - self.mdp.gamma * P_pi, R_pi)
        
        return V


def visualize_convergence(deltas_dict: dict, title: str = "Policy Evaluation Convergence"):
    """可视化收敛过程"""
    plt.figure(figsize=(10, 6))
    
    for method, deltas in deltas_dict.items():
        plt.semilogy(deltas, label=method, linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Max |V_{k+1} - V_k| (log scale)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == "__main__":
    from gridworld import GridWorldMDP  # 假设已定义
    
    # 创建环境
    mdp = GridWorldMDP(size=4, gamma=0.9)
    
    # 定义一个随机策略（均匀分布）
    uniform_policy = np.ones((mdp.num_states, mdp.num_actions)) / mdp.num_actions
    
    # 对比不同方法
    results = {}
    
    for method in ['in-place', 'two-array']:
        evaluator = PolicyEvaluator(mdp, method=method)
        V, deltas = evaluator.evaluate(uniform_policy, theta=1e-6)
        results[method] = deltas
        print(f"\n{method.upper()} Method:")
        print(f"Value function:\n{V.reshape(4, 4)}")
    
    # 矩阵方法（精确解）
    evaluator = PolicyEvaluator(mdp)
    V_exact = evaluator.evaluate_matrix(uniform_policy)
    print(f"\nMatrix Inversion (Exact):")
    print(f"Value function:\n{V_exact.reshape(4, 4)}")
    
    # 可视化收敛
    visualize_convergence(results)
```

---

### 2.1.7 数值稳定性与实践技巧

**停止条件选择**

- **绝对误差**：$\max_s |V_{k+1}(s) - V_k(s)| < \theta$
- **相对误差**：$\frac{\max_s |V_{k+1}(s) - V_k(s)|}{\max_s |V_k(s)|} < \theta$

对于价值函数量级差异大的问题，相对误差更合适。

**初始化策略**

- **零初始化**：$V_0(s) = 0$（最常用）
- **启发式初始化**：基于领域知识
- **随机初始化**：测试收敛鲁棒性

**浮点误差累积**

长时间迭代可能累积浮点误差，建议：
- 使用双精度浮点数（`float64`）
- 定期检查数值稳定性
- 设置最大迭代次数防止无限循环

---

### 2.1.8 小结

策略评估是动态规划的基础操作，核心要点：

1. **理论基础**：Bellman 期望算子的收缩性保证收敛
2. **算法**：迭代更新直到收敛，In-place 通常更快
3. **复杂度**：每次迭代 $O(|\mathcal{S}|^2 |\mathcal{A}|)$
4. **收敛速率**：几何收敛，速率取决于 $\gamma$

下一节我们将学习如何利用评估得到的价值函数来改进策略。

---

## 2.2 策略改进 (Policy Improvement)

**目标**：给定当前策略 $\pi$ 及其价值 $V^\pi$，找到一个更好的策略 $\pi'$。

策略改进是动态规划的第二个核心操作。本节将深入探讨策略改进定理的完整证明、贪心策略的性质以及实现技巧。

---

### 2.2.1 策略改进的动机

假设我们已经通过策略评估得到了 $V^\pi$。现在的问题是：**如何改进策略？**

**直观想法**：在每个状态 $s$，如果存在某个动作 $a$ 使得：

$$
Q^\pi(s, a) > V^\pi(s)
$$

那么在状态 $s$ 选择动作 $a$ 应该比遵循 $\pi$ 更好。

**贪心策略**

定义贪心策略 $\pi'$ 为：

$$
\pi'(s) = \arg\max_a Q^\pi(s, a) = \arg\max_a \sum_{s'} \mathcal{P}(s'|s,a) \left[ \mathcal{R}(s,a) + \gamma V^\pi(s') \right]
$$

**核心问题**：这样的贪心改进是否总能提升策略性能？

---

### 2.2.2 策略改进定理 (Policy Improvement Theorem)

**定理 (策略改进定理)**

设 $\pi$ 和 $\pi'$ 是两个确定性策略，满足：

$$
Q^\pi(s, \pi'(s)) \geq V^\pi(s), \quad \forall s \in \mathcal{S}
$$

则有：

$$
V^{\pi'}(s) \geq V^\pi(s), \quad \forall s \in \mathcal{S}
$$

即 $\pi' \geq \pi$（策略 $\pi'$ 至少和 $\pi$ 一样好）。

如果存在某个状态 $s$ 使得不等式严格成立，则 $\pi'$ 严格优于 $\pi$。

---

### 2.2.3 策略改进定理的完整证明

**证明思路**：通过展开 $V^{\pi'}$ 并利用 Bellman 方程，逐步证明不等式。

**第一步：单步改进**

从定义出发：

$$
\begin{aligned}
V^\pi(s) &\leq Q^\pi(s, \pi'(s)) \quad \text{(假设)} \\
&= \mathbb{E}_{\pi'}[R_{t+1} + \gamma V^\pi(S_{t+1}) \mid S_t = s] \\
&= \sum_{s'} \mathcal{P}(s'|s, \pi'(s)) \left[ \mathcal{R}(s, \pi'(s)) + \gamma V^\pi(s') \right]
\end{aligned}
$$

**第二步：递归展开**

现在在 $s'$ 处继续应用同样的不等式：

$$
V^\pi(s') \leq Q^\pi(s', \pi'(s')) = \sum_{s''} \mathcal{P}(s''|s', \pi'(s')) \left[ \mathcal{R}(s', \pi'(s')) + \gamma V^\pi(s'') \right]
$$

代入上式：

$$
\begin{aligned}
V^\pi(s) &\leq \sum_{s'} \mathcal{P}(s'|s, \pi'(s)) \left[ \mathcal{R}(s, \pi'(s)) + \gamma V^\pi(s') \right] \\
&\leq \sum_{s'} \mathcal{P}(s'|s, \pi'(s)) \Bigg[ \mathcal{R}(s, \pi'(s)) + \gamma \sum_{s''} \mathcal{P}(s''|s', \pi'(s')) \left[ \mathcal{R}(s', \pi'(s')) + \gamma V^\pi(s'') \right] \Bigg] \\
&= \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 V^\pi(S_{t+2}) \mid S_t = s]
\end{aligned}
$$

**第三步：无限展开**

继续递归展开 $k$ 步：

$$
V^\pi(s) \leq \mathbb{E}_{\pi'}\left[ \sum_{t=0}^{k-1} \gamma^t R_{t+1} + \gamma^k V^\pi(S_{t+k}) \mid S_0 = s \right]
$$

**第四步：取极限**

当 $k \to \infty$ 时，由于 $\gamma < 1$ 且 $V^\pi$ 有界，有 $\gamma^k V^\pi(S_{t+k}) \to 0$，因此：

$$
V^\pi(s) \leq \mathbb{E}_{\pi'}\left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \mid S_0 = s \right] = V^{\pi'}(s)
$$

**结论**：$V^{\pi'}(s) \geq V^\pi(s), \forall s$。证毕。

---

### 2.2.4 贪心策略的性质

**性质 1：确定性最优策略的存在性**

**定理**：对于任意有限 MDP，存在一个**确定性**最优策略 $\pi^*$。

**证明思路**：

1. 假设 $\pi^*$ 是随机最优策略
2. 定义确定性策略 $\pi'(s) = \arg\max_a Q^{\pi^*}(s, a)$
3. 由策略改进定理，$V^{\pi'}(s) \geq V^{\pi^*}(s)$
4. 由于 $\pi^*$ 已经最优，必有 $V^{\pi'}(s) = V^{\pi^*}(s)$
5. 因此确定性策略 $\pi'$ 也是最优的

**性质 2：贪心策略的唯一性**

贪心策略 $\pi'(s) = \arg\max_a Q^\pi(s, a)$ 可能不唯一（存在多个动作达到最大值）。

**打破平局 (Tie-Breaking)**

常见策略：
- **最小索引**：选择索引最小的动作
- **随机选择**：在最优动作中随机选择
- **领域知识**：基于任务特性选择

---

### 2.2.5 $\epsilon$-改进策略

在某些情况下，我们希望策略改进更加保守，避免过度贪心。

**定义**：$\epsilon$-改进策略

$$
\pi'(a|s) = \begin{cases}
1 - \epsilon + \frac{\epsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_{a'} Q^\pi(s, a') \\
\frac{\epsilon}{|\mathcal{A}|} & \text{otherwise}
\end{cases}
$$

这保证了探索（exploration），同时主要利用（exploitation）当前最优动作。

---

### 2.2.6 代码实现

```python
import numpy as np
from typing import Tuple

class PolicyImprover:
    """策略改进器"""
    
    def __init__(self, mdp):
        self.mdp = mdp
    
    def improve(self, V: np.ndarray, tie_breaking='min') -> Tuple[np.ndarray, bool]:
        """
        策略改进：基于价值函数 V 计算贪心策略
        
        Args:
            V: 状态价值函数
            tie_breaking: 打破平局策略 ('min', 'random', 'max')
            
        Returns:
            policy: 改进后的策略 (确定性)
            policy_stable: 策略是否稳定（未改变）
        """
        policy = np.zeros(self.mdp.num_states, dtype=int)
        policy_stable = True
        
        for s in range(self.mdp.num_states):
            # 计算所有动作的 Q 值
            q_values = self._compute_q_values(s, V)
            
            # 找到最优动作
            if tie_breaking == 'min':
                best_action = np.argmax(q_values)
            elif tie_breaking == 'random':
                max_q = np.max(q_values)
                best_actions = np.where(q_values == max_q)[0]
                best_action = np.random.choice(best_actions)
            elif tie_breaking == 'max':
                best_action = np.argmax(q_values[::-1])
                best_action = len(q_values) - 1 - best_action
            else:
                raise ValueError(f"Unknown tie_breaking: {tie_breaking}")
            
            policy[s] = best_action
        
        return policy, policy_stable
    
    def improve_stochastic(self, V: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
        """
        ε-贪心策略改进（随机策略）
        
        Args:
            V: 状态价值函数
            epsilon: 探索概率
            
        Returns:
            policy: 改进后的随机策略 [num_states, num_actions]
        """
        policy = np.zeros((self.mdp.num_states, self.mdp.num_actions))
        
        for s in range(self.mdp.num_states):
            q_values = self._compute_q_values(s, V)
            best_action = np.argmax(q_values)
            
            # ε-greedy
            for a in range(self.mdp.num_actions):
                if a == best_action:
                    policy[s, a] = 1 - epsilon + epsilon / self.mdp.num_actions
                else:
                    policy[s, a] = epsilon / self.mdp.num_actions
        
        return policy
    
    def _compute_q_values(self, s: int, V: np.ndarray) -> np.ndarray:
        """计算状态 s 下所有动作的 Q 值"""
        q_values = np.zeros(self.mdp.num_actions)
        
        for a in range(self.mdp.num_actions):
            q = 0
            for s_prime in range(self.mdp.num_states):
                prob = self.mdp.P[s, a, s_prime]
                if prob > 0:
                    reward = self.mdp.R[s, a]
                    q += prob * (reward + self.mdp.gamma * V[s_prime])
            q_values[a] = q
        
        return q_values
    
    def compute_policy_value_gap(self, V: np.ndarray, policy: np.ndarray) -> float:
        """
        计算策略改进的潜力（最大 Q 值与当前策略价值的差距）
        
        Returns:
            max_gap: 最大改进空间
        """
        max_gap = 0
        
        for s in range(self.mdp.num_states):
            q_values = self._compute_q_values(s, V)
            max_q = np.max(q_values)
            current_v = V[s]
            gap = max_q - current_v
            max_gap = max(max_gap, gap)
        
        return max_gap


def visualize_policy_improvement(mdp, V_before, policy_before, V_after, policy_after):
    """可视化策略改进前后的对比"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 重塑为网格（假设是 GridWorld）
    size = int(np.sqrt(mdp.num_states))
    V_before_grid = V_before.reshape(size, size)
    V_after_grid = V_after.reshape(size, size)
    policy_before_grid = policy_before.reshape(size, size)
    policy_after_grid = policy_after.reshape(size, size)
    
    # 改进前的价值函数
    im1 = axes[0, 0].imshow(V_before_grid, cmap='viridis')
    axes[0, 0].set_title('Value Function (Before)', fontsize=14)
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 改进后的价值函数
    im2 = axes[0, 1].imshow(V_after_grid, cmap='viridis')
    axes[0, 1].set_title('Value Function (After)', fontsize=14)
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 改进前的策略
    im3 = axes[1, 0].imshow(policy_before_grid, cmap='tab10', vmin=0, vmax=3)
    axes[1, 0].set_title('Policy (Before)', fontsize=14)
    plt.colorbar(im3, ax=axes[1, 0], ticks=[0, 1, 2, 3])
    
    # 改进后的策略
    im4 = axes[1, 1].imshow(policy_after_grid, cmap='tab10', vmin=0, vmax=3)
    axes[1, 1].set_title('Policy (After)', fontsize=14)
    plt.colorbar(im4, ax=axes[1, 1], ticks=[0, 1, 2, 3])
    
    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == "__main__":
    from gridworld import GridWorldMDP
    from policy_evaluator import PolicyEvaluator
    
    # 创建环境
    mdp = GridWorldMDP(size=4, gamma=0.9)
    
    # 初始随机策略
    policy = np.random.randint(0, mdp.num_actions, size=mdp.num_states)
    
    # 评估当前策略
    evaluator = PolicyEvaluator(mdp)
    V, _ = evaluator.evaluate(policy)
    
    print("Before Improvement:")
    print(f"Policy:\n{policy.reshape(4, 4)}")
    print(f"Value:\n{V.reshape(4, 4)}")
    
    # 策略改进
    improver = PolicyImprover(mdp)
    policy_new, stable = improver.improve(V)
    
    # 评估新策略
    V_new, _ = evaluator.evaluate(policy_new)
    
    print("\nAfter Improvement:")
    print(f"Policy:\n{policy_new.reshape(4, 4)}")
    print(f"Value:\n{V_new.reshape(4, 4)}")
    
    # 计算改进幅度
    improvement = np.mean(V_new - V)
    print(f"\nAverage Value Improvement: {improvement:.4f}")
    
    # 可视化
    visualize_policy_improvement(mdp, V, policy, V_new, policy_new)
```

---

### 2.2.7 策略改进的实践考虑

**何时停止改进？**

策略改进会在以下情况停止：

$$
Q^\pi(s, \pi'(s)) = V^\pi(s), \quad \forall s
$$

此时 $\pi' = \pi$，策略已经收敛到最优。

**改进的单调性**

策略改进保证单调性：

$$
V^{\pi_0} \leq V^{\pi_1} \leq V^{\pi_2} \leq \cdots \leq V^{\pi^*}
$$

由于有限 MDP 的策略空间有限，这个序列必定在有限步内收敛。

**计算复杂度**

策略改进的时间复杂度为 $O(|\mathcal{S}| |\mathcal{A}| |\mathcal{S}|) = O(|\mathcal{S}|^2 |\mathcal{A}|)$，与策略评估相同。

---

### 2.2.8 小结

策略改进的核心要点：

1. **贪心原则**：选择使 $Q$ 值最大的动作
2. **理论保证**：策略改进定理保证单调提升
3. **确定性最优**：存在确定性最优策略
4. **实现简单**：只需计算 Q 值并取 argmax

下一节我们将结合策略评估和策略改进，构建完整的策略迭代算法。

---

## 2.3 策略迭代 (Policy Iteration)

结合策略评估与策略改进，我们得到了强化学习中最经典的算法之一：**策略迭代**。

---

### 2.3.1 算法框架

**策略迭代算法**

```
Algorithm: Policy Iteration
Input: MDP (S, A, P, R, γ), threshold θ
Output: Optimal policy π*, optimal value V*

1. Initialize π arbitrarily (e.g., π(s) = random action)
2. repeat
3.     // Policy Evaluation
4.     V ← Evaluate(π, θ)
5.     
6.     // Policy Improvement
7.     policy_stable ← true
8.     for each s ∈ S do
9.         old_action ← π(s)
10.        π(s) ← argmax_a Σ_s' P(s'|s,a) [R(s,a) + γ V(s')]
11.        if old_action ≠ π(s) then
12.            policy_stable ← false
13.    end for
14.    
15. until policy_stable
16. return π, V
```

**核心思想**：
1. **评估**：计算当前策略的价值
2. **改进**：基于价值贪心地改进策略
3. **迭代**：重复直到策略不再改变

<div data-component="PolicyIterationVisualizer"></div>

---

### 2.3.2 收敛性分析

**定理（有限步收敛）**

对于有限 MDP，策略迭代算法在**有限步**内收敛到最优策略。

**证明**：

1. **单调性**：由策略改进定理，每次迭代都有 $V^{\pi_{k+1}} \geq V^{\pi_k}$

2. **有限性**：有限 MDP 的确定性策略数量有限（最多 $|\mathcal{A}|^{|\mathcal{S}|}$ 个）

3. **严格改进或收敛**：
   - 如果 $\pi_{k+1} \neq \pi_k$，则存在某个状态 $s$ 使得 $V^{\pi_{k+1}}(s) > V^{\pi_k}(s)$（严格改进）
   - 如果 $\pi_{k+1} = \pi_k$，则满足 Bellman 最优方程，$\pi_k = \pi^*$

4. **结论**：由于策略数量有限且每次严格改进，算法必在有限步内收敛

**最坏情况分析**

- **理论上界**：最多 $|\mathcal{A}|^{|\mathcal{S}|}$ 次迭代
- **实际表现**：通常只需 $O(|\mathcal{S}|)$ 次迭代甚至更少

**例子**：4×4 GridWorld
- 状态数：16
- 动作数：4
- 理论上界：$4^{16} \approx 4.3 \times 10^9$ 次
- 实际迭代：通常 3-5 次

---

### 2.3.3 算法优化

**优化 1：Modified Policy Iteration**

策略评估不需要完全收敛，可以截断：

```python
def modified_policy_iteration(mdp, k_eval=10, theta=1e-6):
    """
    修改的策略迭代：每次只进行 k 步策略评估
    
    Args:
        k_eval: 策略评估的最大迭代次数
    """
    policy = np.random.randint(0, mdp.num_actions, size=mdp.num_states)
    V = np.zeros(mdp.num_states)
    
    while True:
        # Truncated Policy Evaluation (k steps)
        for _ in range(k_eval):
            delta = 0
            for s in range(mdp.num_states):
                v = V[s]
                a = policy[s]
                V[s] = sum(mdp.P[s, a, s_prime] * 
                          (mdp.R[s, a] + mdp.gamma * V[s_prime])
                          for s_prime in range(mdp.num_states))
                delta = max(delta, abs(v - V[s]))
            
            if delta < theta:
                break
        
        # Policy Improvement
        policy_stable = True
        for s in range(mdp.num_states):
            old_action = policy[s]
            q_values = [sum(mdp.P[s, a, s_prime] * 
                           (mdp.R[s, a] + mdp.gamma * V[s_prime])
                           for s_prime in range(mdp.num_states))
                       for a in range(mdp.num_actions)]
            policy[s] = np.argmax(q_values)
            
            if old_action != policy[s]:
                policy_stable = False
        
        if policy_stable:
            break
    
    return policy, V
```

**优化 2：初始化策略的影响**

- **随机初始化**：收敛较慢
- **启发式初始化**：基于领域知识，收敛更快
- **贪心初始化**：基于零价值函数的贪心策略

---

### 2.3.4 完整实现

```python
import numpy as np
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt

class PolicyIterator:
    """策略迭代器"""
    
    def __init__(self, mdp):
        self.mdp = mdp
        self.history = []  # 记录迭代历史
    
    def iterate(self, theta: float = 1e-6, max_iterations: int = 100,
                k_eval: int = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        策略迭代算法
        
        Args:
            theta: 收敛阈值
            max_iterations: 最大迭代次数
            k_eval: 截断策略评估的步数（None 表示完全评估）
            
        Returns:
            policy: 最优策略
            V: 最优价值函数
            info: 迭代信息
        """
        # 初始化
        policy = np.random.randint(0, self.mdp.num_actions, 
                                  size=self.mdp.num_states)
        V = np.zeros(self.mdp.num_states)
        
        self.history = []
        total_eval_iterations = 0
        
        for iteration in range(max_iterations):
            # Policy Evaluation
            V_old = V.copy()
            V, eval_iters = self._policy_evaluation(policy, V, theta, k_eval)
            total_eval_iterations += eval_iters
            
            # Policy Improvement
            policy_old = policy.copy()
            policy, policy_stable = self._policy_improvement(V)
            
            # 记录历史
            policy_changes = np.sum(policy != policy_old)
            value_change = np.max(np.abs(V - V_old))
            
            self.history.append({
                'iteration': iteration,
                'policy_changes': policy_changes,
                'value_change': value_change,
                'eval_iterations': eval_iters,
                'V': V.copy(),
                'policy': policy.copy()
            })
            
            print(f"Iteration {iteration + 1}: "
                  f"{policy_changes} policy changes, "
                  f"max value change = {value_change:.6f}")
            
            if policy_stable:
                print(f"\nConverged in {iteration + 1} iterations")
                print(f"Total policy evaluation iterations: {total_eval_iterations}")
                break
        
        info = {
            'iterations': iteration + 1,
            'total_eval_iterations': total_eval_iterations,
            'history': self.history
        }
        
        return policy, V, info
    
    def _policy_evaluation(self, policy: np.ndarray, V_init: np.ndarray,
                          theta: float, k_max: int = None) -> Tuple[np.ndarray, int]:
        """策略评估（支持截断）"""
        V = V_init.copy()
        iterations = 0
        
        while True:
            delta = 0
            for s in range(self.mdp.num_states):
                v = V[s]
                a = policy[s]
                
                V[s] = sum(self.mdp.P[s, a, s_prime] * 
                          (self.mdp.R[s, a] + self.mdp.gamma * V[s_prime])
                          for s_prime in range(self.mdp.num_states))
                
                delta = max(delta, abs(v - V[s]))
            
            iterations += 1
            
            # 检查停止条件
            if delta < theta:
                break
            if k_max is not None and iterations >= k_max:
                break
        
        return V, iterations
    
    def _policy_improvement(self, V: np.ndarray) -> Tuple[np.ndarray, bool]:
        """策略改进"""
        policy = np.zeros(self.mdp.num_states, dtype=int)
        policy_stable = True
        
        for s in range(self.mdp.num_states):
            # 计算所有动作的 Q 值
            q_values = np.zeros(self.mdp.num_actions)
            for a in range(self.mdp.num_actions):
                q_values[a] = sum(self.mdp.P[s, a, s_prime] * 
                                 (self.mdp.R[s, a] + self.mdp.gamma * V[s_prime])
                                 for s_prime in range(self.mdp.num_states))
            
            policy[s] = np.argmax(q_values)
        
        return policy, policy_stable
    
    def visualize_convergence(self):
        """可视化收敛过程"""
        if not self.history:
            print("No history to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        iterations = [h['iteration'] for h in self.history]
        policy_changes = [h['policy_changes'] for h in self.history]
        value_changes = [h['value_change'] for h in self.history]
        eval_iters = [h['eval_iterations'] for h in self.history]
        
        # 策略变化
        axes[0, 0].plot(iterations, policy_changes, 'o-', linewidth=2)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Number of Policy Changes')
        axes[0, 0].set_title('Policy Convergence')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 价值变化
        axes[0, 1].semilogy(iterations, value_changes, 'o-', linewidth=2, color='orange')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Max Value Change (log scale)')
        axes[0, 1].set_title('Value Function Convergence')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 评估迭代次数
        axes[1, 0].bar(iterations, eval_iters, color='green', alpha=0.7)
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Policy Evaluation Iterations')
        axes[1, 0].set_title('Evaluation Cost per Iteration')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 价值函数演化（选择几个状态）
        num_states_to_plot = min(5, self.mdp.num_states)
        for s in range(num_states_to_plot):
            values = [h['V'][s] for h in self.history]
            axes[1, 1].plot(iterations, values, 'o-', label=f'State {s}', linewidth=2)
        
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Value Function Evolution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def compare_policy_iteration_variants(mdp, k_values=[None, 1, 5, 10]):
    """对比不同截断步数的策略迭代"""
    results = {}
    
    for k in k_values:
        print(f"\n{'='*50}")
        print(f"Running Policy Iteration with k_eval = {k}")
        print(f"{'='*50}")
        
        iterator = PolicyIterator(mdp)
        policy, V, info = iterator.iterate(k_eval=k)
        
        results[k] = {
            'policy': policy,
            'V': V,
            'info': info,
            'iterator': iterator
        }
    
    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for k, result in results.items():
        label = f"k={k}" if k is not None else "Full Eval"
        iterations = [h['iteration'] for h in result['info']['history']]
        value_changes = [h['value_change'] for h in result['info']['history']]
        
        axes[0].semilogy(iterations, value_changes, 'o-', label=label, linewidth=2)
    
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Max Value Change (log scale)')
    axes[0].set_title('Convergence Speed Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 总评估迭代次数对比
    labels = [f"k={k}" if k is not None else "Full" for k in k_values]
    total_evals = [results[k]['info']['total_eval_iterations'] for k in k_values]
    
    axes[1].bar(labels, total_evals, color='skyblue', alpha=0.7)
    axes[1].set_ylabel('Total Evaluation Iterations')
    axes[1].set_title('Computational Cost Comparison')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    return results


# 使用示例
if __name__ == "__main__":
    from gridworld import GridWorldMDP
    
    # 创建环境
    mdp = GridWorldMDP(size=4, gamma=0.9)
    
    # 标准策略迭代
    print("Standard Policy Iteration")
    print("="*50)
    iterator = PolicyIterator(mdp)
    policy, V, info = iterator.iterate()
    
    print(f"\nOptimal Policy:\n{policy.reshape(4, 4)}")
    print(f"\nOptimal Value:\n{V.reshape(4, 4)}")
    
    # 可视化收敛过程
    iterator.visualize_convergence()
    
    # 对比不同变体
    print("\n\nComparing Policy Iteration Variants")
    results = compare_policy_iteration_variants(mdp, k_values=[None, 1, 3, 5, 10])
```

---

### 2.3.5 实验分析

**实验 1：迭代次数 vs 环境复杂度**

| 环境 | 状态数 | 动作数 | 迭代次数 | 总评估迭代 |
|:---|:---:|:---:|:---:|:---:|
| GridWorld 4×4 | 16 | 4 | 3-4 | 120-150 |
| GridWorld 8×8 | 64 | 4 | 5-7 | 500-800 |
| FrozenLake 4×4 | 16 | 4 | 4-6 | 150-250 |
| CliffWalking | 48 | 4 | 6-8 | 300-500 |

**实验 2：截断评估的影响**

对于 GridWorld 4×4：
- **完全评估** (k=∞): 4 次策略迭代，150 次总评估
- **k=10**: 4 次策略迭代，40 次总评估
- **k=5**: 5 次策略迭代，25 次总评估
- **k=1**: 8 次策略迭代，8 次总评估（接近价值迭代）

**观察**：
- 截断评估减少计算量，但可能增加策略迭代次数
- 存在最优的 k 值平衡两者

---

### 2.3.6 策略迭代的特点

**优点**：
1. **收敛快**：策略迭代次数通常很少（3-10 次）
2. **理论保证**：有限步收敛到最优
3. **单调改进**：每次迭代都严格提升或收敛

**缺点**：
1. **评估开销大**：每次策略评估需要多次迭代
2. **需要完整模型**：必须知道 $\mathcal{P}$ 和 $\mathcal{R}$
3. **全宽扫描**：每次更新需遍历所有状态

**时间复杂度**：
- 单次策略评估：$O(|\mathcal{S}|^2 |\mathcal{A}| \cdot k_{eval})$
- 单次策略改进：$O(|\mathcal{S}|^2 |\mathcal{A}|)$
- 总复杂度：$O(|\mathcal{S}|^2 |\mathcal{A}| \cdot k_{eval} \cdot k_{PI})$

其中 $k_{eval}$ 是评估迭代次数，$k_{PI}$ 是策略迭代次数。

---

### 2.3.7 小结

策略迭代是动态规划的核心算法：

1. **框架**：评估 → 改进 → 重复
2. **收敛性**：有限步收敛到最优
3. **优化**：截断评估减少计算量
4. **实践**：通常 3-10 次迭代即可收敛

下一节我们将学习价值迭代，它是策略迭代的一个极端变体。

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

---
title: "第30章：竞争多智能体与博弈论"
description: "零和博弈、纳什均衡、扑克AI、CFR算法与对抗训练"
date: "2026-01-30"
---

# 第30章：竞争多智能体与博弈论

## 30.1 零和博弈

### 30.1.1 极小极大策略

在**零和**博弈中，一个玩家的收益恰好等于另一个玩家的损失。总收益之和为零。

**形式化定义**：

对于双人博弈，收益矩阵为 $A$：
- 玩家1选择行 $i$，获得收益 $A_{ij}$
- 玩家2选择列 $j$，获得收益 $-A_{ij}$

**极小极大定理**（von Neumann, 1928）：

$$
\max_i \min_j A_{ij} = \min_j \max_i A_{ij} = v^*
$$

其中 $v^*$ 是**博弈的值**。

**极小极大策略**：
- 玩家1：选择 $i^* = \arg\max_i \min_j A_{ij}$
- 玩家2：选择 $j^* = \arg\min_j \max_i A_{ij}$

<div data-component="ZeroSumGameVisualization"></div>

### 30.1.2 纳什均衡计算

**纳什均衡**：策略组合，其中没有玩家能通过单方面偏离获益。

对于双人零和博弈，纳什均衡 = 极小极大/极大极小策略。

**支撑枚举法**：

```python
import numpy as np
from scipy.optimize import linprog

def compute_nash_equilibrium(payoff_matrix):
    """
    使用线性规划计算双人零和博弈的纳什均衡
    
    Args:
        payoff_matrix: 玩家1的收益矩阵 (m x n)
    
    Returns:
        strategy_p1: 玩家1的混合策略（概率向量）
        strategy_p2: 玩家2的混合策略（概率向量）
        value: 博弈的值
    """
    A = payoff_matrix
    m, n = A.shape
    
    # 玩家1的极大极小问题（最大化值 v）：
    # maximize v
    # subject to: A^T * p >= v * 1, sum(p) = 1, p >= 0
    
    # 转换为LP最小化形式：minimize -v
    c = np.zeros(m + 1)
    c[-1] = -1  # v的系数
    
    # 不等式：-A^T * p + v * 1 <= 0
    A_ub = np.hstack([-A.T, np.ones((n, 1))])
    b_ub = np.zeros(n)
    
    # 等式：sum(p) = 1
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1
    b_eq = np.array([1])
    
    # 边界：p >= 0, v无界
    bounds = [(0, None) for _ in range(m)] + [(None, None)]
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if result.success:
        strategy_p1 = result.x[:m]
        value = result.x[-1]
        
        # 玩家2的极小极大（通过对称性）
        c2 = np.zeros(n + 1)
        c2[-1] = 1  # 玩家2最小化
        
        A_ub2 = np.hstack([A, -np.ones((m, 1))])
        b_ub2 = np.zeros(m)
        
        A_eq2 = np.zeros((1, n + 1))
        A_eq2[0, :n] = 1
        b_eq2 = np.array([1])
        
        bounds2 = [(0, None) for _ in range(n)] + [(None, None)]
        
        result2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2, bounds=bounds2, method='highs')
        
        if result2.success:
            strategy_p2 = result2.x[:n]
            return strategy_p1, strategy_p2, value
    
    return None, None, None


# 示例：石头-剪刀-布
rps_payoff = np.array([
    [ 0,  -1,   1],  # 石头 vs (石头, 布, 剪刀)
    [ 1,   0,  -1],  # 布 vs (石头, 布, 剪刀)
    [-1,   1,   0]   # 剪刀 vs (石头, 布, 剪刀)
])

p1_strategy, p2_strategy, game_value = compute_nash_equilibrium(rps_payoff)
print(f"玩家1策略: {p1_strategy}")  # 应为 [1/3, 1/3, 1/3]
print(f"玩家2策略: {p2_strategy}")  # 应为 [1/3, 1/3, 1/3]
print(f"博弈值: {game_value}")  # 应为 0（公平博弈）
```

### 30.1.3 可利用性

**可利用性（Exploitability）** 衡量策略距离纳什均衡有多远。

**定义**：

$$
\text{Exploitability}(\sigma) = \max_{\sigma'} u(\sigma', \sigma) - u(\sigma^*, \sigma^*)
$$

其中：
- $\sigma$ 是被评估的策略
- $\sigma^*$ 是纳什均衡策略
- $u(\sigma', \sigma)$ 是使用 $\sigma'$ 最佳应对 $\sigma$ 时的收益

**计算可利用性**：

```python
def compute_exploitability(strategy, payoff_matrix):
    """
    计算策略的可利用性
    
    Args:
        strategy: 混合策略（概率向量）
        payoff_matrix: 收益矩阵
    
    Returns:
        exploitability: 最佳应对能做得多好
    """
    # 计算针对每个纯策略的期望收益
    expected_payoffs = payoff_matrix @ strategy
    
    # 最佳应对：选择期望收益最高的纯策略
    best_response_value = np.max(expected_payoffs)
    
    # 纳什均衡值
    nash_value = strategy @ payoff_matrix @ strategy
    
    # 可利用性 = 最佳应对的潜在收益
    exploitability = best_response_value - nash_value
    
    return exploitability
```

<div data-component="ExploitabilityMeasure"></div>

---

## 30.2 扑克AI

### 30.2.1 不完全信息博弈

**扑克**是**不完全信息**博弈：玩家不知道对手的隐藏牌。

**与完全信息的关键区别**：
- **信息集**：玩家无法区分的所有状态
- **信念状态**：可能隐藏状态的概率分布
- **虚张声势（Bluffing）**：有时理性地打出次优策略来欺骗对手

**简化扑克的博弈树（Kuhn扑克）**：

- 3张牌：J（Jack）、Q（Queen）、K（King）
- 2名玩家，每人发1张牌
- 下注回合：过牌（Check）或下注（Bet）
- 收益：赢家拿走彩池

### 30.2.2 CFR（反事实后悔最小化）

**CFR** 是使超人扑克AI成为可能的突破性算法（Libratus, Pluribus）。

**核心思想**：追踪未采取每个动作的**后悔**，加权我们到达该状态的可能性（**反事实**概率）。

**信息集 $I$ 中动作 $a$ 的后悔**：

$$
R^T(I, a) = \sum_{t=1}^T \left( v^{\sigma^t}(I, a) - v^{\sigma^t}(I) \right)
$$

其中：
- $v^{\sigma^t}(I, a)$：如果我们在 $I$ 总是采取动作 $a$（其他地方遵循 $\sigma^t$）的期望值
- $v^{\sigma^t}(I)$：在策略 $\sigma^t$ 下的期望值

**策略更新**（后悔匹配）：

$$
\sigma^{T+1}(I, a) = \frac{\max(R^T(I, a), 0)}{\sum_{a'} \max(R^T(I, a'), 0)}
$$

如果所有后悔都非正，则均匀游戏。

**CFR算法**：

```python
import random
from collections import defaultdict

class KuhnPokerCFR:
    """
    Kuhn扑克的CFR（简化版3张牌扑克）
    参考：Zinkevich et al. (2007) "Regret Minimization in Games with Incomplete Information"
    """
    def __init__(self):
        self.regret_sum = defaultdict(lambda: defaultdict(float))  # I -> {action -> regret}
        self.strategy_sum = defaultdict(lambda: defaultdict(float))  # I -> {action -> 累积概率}
        self.actions = ['check', 'bet']
    
    def get_strategy(self, info_set):
        """
        使用后悔匹配获取当前策略
        
        Args:
            info_set: 信息集字符串（例如 "J", "Qb", "Kc"）
        
        Returns:
            strategy: Dict {action -> probability}
        """
        regrets = self.regret_sum[info_set]
        
        # 后悔匹配
        strategy = {}
        normalizing_sum = 0.0
        
        for action in self.actions:
            strategy[action] = max(0.0, regrets[action])
            normalizing_sum += strategy[action]
        
        # 归一化
        if normalizing_sum > 0:
            for action in self.actions:
                strategy[action] /= normalizing_sum
        else:
            # 如果所有后悔都非正，则均匀游戏
            for action in self.actions:
                strategy[action] = 1.0 / len(self.actions)
        
        return strategy
    
    def cfr(self, cards, history, p0, p1):
        """
        递归CFR计算
        
        Args:
            cards: 元组 (card0, card1)，其中 card0 是玩家0的牌
            history: 已采取动作的字符串（例如 "cb" = check then bet）
            p0: 玩家0到达此状态的概率
            p1: 玩家1到达此状态的概率
        
        Returns:
            expected_value: 玩家0的期望收益
        """
        plays = len(history)
        player = plays % 2  # 当前玩家（0或1）
        opponent = 1 - player
        
        # 终止状态（游戏结束）
        if plays > 1:
            # 双方都过牌（摊牌）
            if history[-2:] == "cc":
                return 1 if cards[0] > cards[1] else -1
            
            # 玩家在下注后弃牌
            if history[-1] == 'c' and history[-2] == 'b':
                return -1  # 下注的玩家赢
            
            # 双方都下注（摊牌，彩池更大）
            if history[-2:] == "bb":
                return 2 if cards[0] > cards[1] else -2
        
        # 非终止：递归计算值
        info_set = str(cards[player]) + history
        strategy = self.get_strategy(info_set)
        
        # 对手的反事实到达概率
        cf_prob = p1 if player == 0 else p0
        
        # 计算动作值
        action_values = {}
        node_value = 0.0
        
        for action in self.actions:
            next_history = history + ('c' if action == 'check' else 'b')
            
            if player == 0:
                action_values[action] = -self.cfr(cards, next_history, p0 * strategy[action], p1)
            else:
                action_values[action] = -self.cfr(cards, next_history, p0, p1 * strategy[action])
            
            node_value += strategy[action] * action_values[action]
        
        # 更新后悔
        for action in self.actions:
            regret = action_values[action] - node_value
            self.regret_sum[info_set][action] += cf_prob * regret
        
        # 更新策略和（用于平均）
        for action in self.actions:
            self.strategy_sum[info_set][action] += (p0 if player == 0 else p1) * strategy[action]
        
        return node_value
    
    def train(self, iterations=10000):
        """
        训练CFR N次迭代
        
        Args:
            iterations: 训练迭代次数
        """
        cards = [0, 1, 2]  # J=0, Q=1, K=2
        
        for i in range(iterations):
            # 洗牌并发牌
            random.shuffle(cards)
            player_cards = (cards[0], cards[1])
            
            # 从根节点运行CFR
            self.cfr(player_cards, "", 1.0, 1.0)
            
            if (i + 1) % 1000 == 0:
                print(f"迭代 {i+1}/{iterations}")
        
        print("\n最终平均策略:")
        self.print_strategy()
    
    def print_strategy(self):
        """打印平均策略"""
        for info_set in sorted(self.strategy_sum.keys()):
            strategy = self.get_average_strategy(info_set)
            print(f"{info_set}: {strategy}")
    
    def get_average_strategy(self, info_set):
        """获取时间平均策略（收敛到纳什均衡）"""
        strategy = {}
        normalizing_sum = sum(self.strategy_sum[info_set].values())
        
        for action in self.actions:
            if normalizing_sum > 0:
                strategy[action] = self.strategy_sum[info_set][action] / normalizing_sum
            else:
                strategy[action] = 1.0 / len(self.actions)
        
        return strategy


# 训练Kuhn扑克
cfr = KuhnPokerCFR()
cfr.train(iterations=10000)
```

<div data-component="CFRAlgorithm"></div>

### 30.2.3 Libratus与Pluribus

**Libratus**（2017）：
- 在单挑无限注德州扑克中击败顶级职业玩家
- 关键技术：
  - **抽象**：减小博弈树大小（10^161个状态 → 可管理）
  - **CFR+**：改进的CFR，更好的后悔更新
  - **子博弈求解**：游戏中的实时优化
  - **自博弈**：1500万CPU核心小时

**Pluribus**（2019）：
- 首个在6人无限注德州扑克中击败职业玩家的AI
- 创新：
  - **蓝图策略**：通过CFR离线计算
  - **深度限制搜索**：在线优化，最多向前4步
  - **线性CFR**：内存高效变体
  - 成本：仅144美元的云计算（相比Libratus的数百万）

**影响**：证明了RL可以处理不完全信息、大搜索空间和多玩家动态。

---

## 30.3 对抗训练

### 30.3.1 红队与蓝队

**对抗训练**使用竞争智能体来发现和修复漏洞。

**设置**：
- **蓝队（防御者）**：正在训练的主策略
- **红队（攻击者）**：发现蓝队策略中的漏洞

**训练循环**：
1. 在主任务上训练蓝队
2. 训练红队利用蓝队
3. 重新训练蓝队防御红队
4. 重复直到收敛

**代码示例**：

```python
class AdversarialTraining:
    """红队 vs 蓝队对抗训练"""
    def __init__(self, env):
        self.env = env
        self.blue_agent = PolicyNetwork()  # 防御者
        self.red_agent = PolicyNetwork()   # 攻击者
        self.blue_optimizer = torch.optim.Adam(self.blue_agent.parameters())
        self.red_optimizer = torch.optim.Adam(self.red_agent.parameters())
    
    def train_round(self, num_episodes=100):
        """训练双方智能体一轮"""
        # 阶段1：蓝队 vs 环境
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                action = self.blue_agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.blue_agent.update(state, action, reward)
                state = next_state
        
        # 阶段2：红队学习利用蓝队
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                # 蓝智能体行动（冻结）
                with torch.no_grad():
                    blue_action = self.blue_agent.select_action(state)
                
                # 红智能体尝试使蓝智能体失败
                # （环境被修改以让红智能体控制干扰）
                red_action = self.red_agent.select_action(state)
                
                # 应用两个动作
                next_state, blue_reward, done, _ = self.env.step_adversarial(blue_action, red_action)
                
                # 红智能体因使蓝智能体失败而获得奖励
                red_reward = -blue_reward
                self.red_agent.update(state, red_action, red_reward)
                state = next_state
        
        # 阶段3：蓝队针对红队重新训练
        for _ in range(num_episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                blue_action = self.blue_agent.select_action(state)
                red_action = self.red_agent.select_action(state)
                
                next_state, reward, done, _ = self.env.step_adversarial(blue_action, red_action)
                self.blue_agent.update(state, blue_action, reward)
                state = next_state
```

### 30.3.2 鲁棒性提升

对抗训练提高对以下方面的**鲁棒性**：
- **分布偏移**：环境变化
- **对抗样本**：精心制作的输入
- **对手适应**：策略随时间演化

**度量**：
- **最坏情况性能**：vs 发现的最强对手
- **后悔**：与预言最佳应对的性能差距

### 30.3.3 对抗样本防御

深度RL策略容易受到观测中**对抗扰动**的影响。

**攻击**：向输入添加小噪声导致灾难性失败。

**通过对抗训练进行防御**：

```python
def adversarial_training_defense(policy, env, epsilon=0.1):
    """
    训练策略以鲁棒对抗扰动
    
    Args:
        policy: RL策略网络
        env: 环境
        epsilon: 最大扰动幅度
    """
    optimizer = torch.optim.Adam(policy.parameters())
    
    for episode in range(1000):
        state = env.reset()
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            state_tensor.requires_grad = True
            
            # 从策略获取动作
            action_probs = policy(state_tensor)
            action = torch.argmax(action_probs).item()
            
            # 计算策略相对于输入的梯度（FGSM攻击）
            loss = -action_probs[0, action]
            loss.backward()
            
            # 生成对抗扰动
            perturbation = epsilon * torch.sign(state_tensor.grad)
            adversarial_state = state_tensor + perturbation
            adversarial_state = torch.clamp(adversarial_state, 0, 1)  # 保持在有效范围
            
            # 在干净和对抗状态上训练
            clean_action_probs = policy(state_tensor.detach())
            adv_action_probs = policy(adversarial_state.detach())
            
            # 策略梯度更新（简化）
            next_state, reward, done, _ = env.step(action)
            
            # 更新策略（鼓励鲁棒性）
            policy_loss = -reward * (torch.log(clean_action_probs[0, action]) + 
                                      torch.log(adv_action_probs[0, action]))
            
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()
            
            state = next_state
```

---

## 30.4 混合策略

### 30.4.1 随机化策略

**混合策略**：纯策略上的概率分布。

**为什么随机？**
- **不可预测性**：防止对手利用模式
- **纳什均衡**：许多博弈没有纯纳什均衡，只有混合

**示例：点球**

| 射手 \\ 守门员 | 左 | 右 |
|-----------------------|------|-------|
| **踢左**         | 0    | 1     |
| **踢右**        | 1    | 0     |

纳什均衡：双方都50-50随机化。

### 30.4.2 不可预测性

在竞争设置中，**可利用性随可预测性增加**。

**作为不可预测性度量的熵**：

$$
H(\pi) = -\sum_a \pi(a) \log \pi(a)
$$

更高的熵 → 更难利用。

### 30.4.3 石头-剪刀-布循环

**非传递博弈**：不存在占优策略（A胜B，B胜C，C胜A）。

**纳什均衡**：均匀随机（1/3, 1/3, 1/3）。

<div data-component="MixedStrategyNash"></div>

**使用虚拟自博弈训练**：

```python
def fictitious_self_play(num_iterations=1000):
    """
    虚拟自博弈寻找纳什均衡
    每个玩家最佳应对对手的平均历史策略
    """
    actions = ['Rock', 'Paper', 'Scissors']
    payoff = {
        ('Rock', 'Rock'): 0, ('Rock', 'Paper'): -1, ('Rock', 'Scissors'): 1,
        ('Paper', 'Rock'): 1, ('Paper', 'Paper'): 0, ('Paper', 'Scissors'): -1,
        ('Scissors', 'Rock'): -1, ('Scissors', 'Paper'): 1, ('Scissors', 'Scissors'): 0
    }
    
    # 跟踪动作计数
    p1_counts = {a: 0 for a in actions}
    p2_counts = {a: 0 for a in actions}
    
    for iteration in range(num_iterations):
        # 计算平均策略
        total_p1 = sum(p1_counts.values()) or 1
        total_p2 = sum(p2_counts.values()) or 1
        
        p1_avg = {a: count / total_p1 for a, count in p1_counts.items()}
        p2_avg = {a: count / total_p2 for a, count in p2_counts.items()}
        
        # 最佳应对对手的平均策略
        p1_best = max(actions, key=lambda a: sum(p2_avg[b] * payoff[(a, b)] for b in actions))
        p2_best = max(actions, key=lambda a: sum(p1_avg[b] * payoff[(b, a)] for b in actions))
        
        # 更新计数
        p1_counts[p1_best] += 1
        p2_counts[p2_best] += 1
    
    # 最终平均策略
    total_p1 = sum(p1_counts.values())
    total_p2 = sum(p2_counts.values())
    
    print("收敛策略:")
    print(f"玩家1: {', '.join([f'{a}: {p1_counts[a] / total_p1:.3f}' for a in actions])}")
    print(f"玩家2: {', '.join([f'{a}: {p2_counts[a] / total_p2:.3f}' for a in actions])}")
```

---

## 总结

本章涵盖了竞争多智能体RL与博弈论：

1. **零和博弈**：极小极大策略、纳什均衡计算、可利用性
2. **扑克AI**：不完全信息、CFR算法、Libratus/Pluribus成就
3. **对抗训练**：红队vs蓝队、鲁棒性提升
4. **混合策略**：随机化以降低不可预测性、非传递博弈

**关键要点**：
- CFR是不完全信息博弈的突破
- 对抗训练发现漏洞并提高鲁棒性
- 当不存在纯纳什均衡时，混合策略是必需的
- 竞争驱动智能体发现复杂策略

**下一部分**：第七部分涵盖大模型时代的RL（RLHF、DPO、推理时RL）。

---

## 参考文献

- Zinkevich, M., et al. (2007). "Regret Minimization in Games with Incomplete Information (CFR)." *NIPS*.
- Brown, N., & Sandholm, T. (2017). "Superhuman AI for Heads-up No-Limit Poker (Libratus)." *Science*.
- Brown, N., & Sandholm, T. (2019). "Superhuman AI for Multiplayer Poker (Pluribus)." *Science*.
- von Neumann, J., & Morgenstern, O. (1944). *Theory of Games and Economic Behavior*. Princeton.
- Pinto, L., et al. (2017). "Robust Adversarial RL." *ICML*.

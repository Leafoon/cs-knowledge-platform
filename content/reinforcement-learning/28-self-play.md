---
title: "第28章：自博弈与涌现行为"
description: "自博弈训练、AlphaZero、种群训练、联盟训练与涌现智能"
date: "2026-01-30"
---

# 第28章：自博弈与涌现行为

## 28.1 自博弈训练

### 28.1.1 核心思想

**自博弈（Self-Play）** 是一种强大的训练范式，智能体通过与自己的副本对战来提升策略。

**关键优势**：
- **自动课程**：随着智能体变强，对手也变强
- **无需人类数据**：从零开始学习
- **发现新策略**：超越人类知识

**形式化定义**：

在时刻 $t$，智能体 $\pi_t$ 与过去版本的自己对战：

$$
\pi_{t+1} = \arg\max_{\pi} \mathbb{E}_{\tau \sim \pi \text{ vs } \pi_t} [R(\tau)]
$$

其中 $\tau$ 是轨迹，$R(\tau)$ 是累积奖励。

### 28.1.2 AlphaZero 架构

**AlphaZero** 是自博弈的里程碑成就，掌握了国际象棋、将棋和围棋，均达到超人水平。

**核心组件**：

1. **蒙特卡洛树搜索（MCTS）**：在训练和推理时扩展搜索树
2. **神经网络**：$f_\theta(s) \rightarrow (p, v)$
   - **策略头** $p$：动作概率分布
   - **价值头** $v$：状态价值估计
3. **自博弈数据生成**：使用 MCTS + 网络生成对局
4. **网络训练**：最小化策略和价值损失

**MCTS 算法**：

对于每个状态 $s$，MCTS 通过四个步骤迭代构建搜索树：

1. **选择（Selection）**：使用 UCB 公式向下遍历树
   $$
   a^* = \arg\max_a \left( Q(s, a) + c_{\text{puct}} P(s, a) \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)} \right)
   $$
   其中：
   - $Q(s, a)$：平均动作价值
   - $P(s, a)$：先验概率（来自神经网络）
   - $N(s, a)$：访问次数
   - $c_{\text{puct}}$：探索常数

2. **扩展（Expansion）**：到达叶节点时，使用神经网络预测 $(p, v)$

3. **评估（Evaluation）**：使用网络的价值估计 $v$

4. **回溯（Backup）**：更新路径上所有节点的 $Q$ 和 $N$

**完整 AlphaZero 实现**：

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

class AlphaZeroNetwork(nn.Module):
    """
    AlphaZero 神经网络：策略-价值网络
    对于给定的游戏状态，输出动作概率分布和状态价值
    """
    def __init__(self, state_shape, action_size, hidden_dim=256):
        """
        Args:
            state_shape: 状态表示的形状（例如棋盘尺寸）
            action_size: 动作空间大小
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        # 共享特征提取器（对于图像状态，使用CNN）
        # 这里展示的是全连接版本
        state_dim = np.prod(state_shape)
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 策略头（输出动作概率）
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size)
        )
        
        # 价值头（输出标量价值）
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # 价值范围 [-1, 1]
        )
    
    def forward(self, state):
        """
        前向传播
        
        Args:
            state: 游戏状态 (batch_size, state_dim)
        
        Returns:
            policy: 动作概率 (batch_size, action_size)
            value: 状态价值 (batch_size, 1)
        """
        features = self.feature_extractor(state)
        
        # 策略输出（log概率用于训练）
        policy_logits = self.policy_head(features)
        policy = F.softmax(policy_logits, dim=-1)
        
        # 价值输出
        value = self.value_head(features)
        
        return policy, value


class MCTSNode:
    """蒙特卡洛树搜索节点"""
    def __init__(self, state, parent=None, prior_prob=1.0):
        self.state = state
        self.parent = parent
        self.children = {}  # {action: MCTSNode}
        self.prior_prob = prior_prob
        
        self.visit_count = 0
        self.value_sum = 0.0
    
    def value(self):
        """平均价值"""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def select_child(self, c_puct=1.0):
        """使用 UCB 公式选择最佳子节点"""
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        # UCB 公式
        for action, child in self.children.items():
            # Q(s,a) + U(s,a)
            q_value = child.value()
            u_value = c_puct * child.prior_prob * np.sqrt(self.visit_count) / (1 + child.visit_count)
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def expand(self, action_probs):
        """使用神经网络预测扩展节点"""
        for action, prob in enumerate(action_probs):
            if prob > 0:  # 只扩展合法动作
                self.children[action] = MCTSNode(state=None, parent=self, prior_prob=prob)
    
    def update(self, value):
        """回溯更新"""
        self.visit_count += 1
        self.value_sum += value


class MCTS:
    """蒙特卡洛树搜索"""
    def __init__(self, network, c_puct=1.0, num_simulations=800):
        """
        Args:
            network: AlphaZero 神经网络
            c_puct: 探索常数
            num_simulations: 每次搜索的模拟次数
        """
        self.network = network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
    
    def search(self, state, env):
        """
        执行 MCTS 搜索
        
        Args:
            state: 当前游戏状态
            env: 游戏环境（用于模拟）
        
        Returns:
            action_probs: 访问计数归一化后的动作概率
        """
        root = MCTSNode(state)
        
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # 1. 选择：向下遍历到叶节点
            while node.children:
                action, node = node.select_child(self.c_puct)
                search_path.append(node)
            
            # 2. 扩展和评估
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_probs, value = self.network(state_tensor)
            
            action_probs = action_probs.squeeze(0).numpy()
            value = value.item()
            
            # 检查是否为终止状态
            if not env.is_terminal(state):
                node.expand(action_probs)
            
            # 3. 回溯
            for n in reversed(search_path):
                n.update(value)
                value = -value  # 对于双人零和游戏，翻转价值
        
        # 返回访问计数作为改进的策略
        visit_counts = np.zeros(len(root.children))
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count
        
        # 归一化为概率
        action_probs = visit_counts / visit_counts.sum()
        
        return action_probs


def train_alphazero(env, num_iterations=1000, games_per_iteration=100):
    """
    AlphaZero 训练循环
    
    Args:
        env: 游戏环境
        num_iterations: 训练迭代次数
        games_per_iteration: 每次迭代的自博弈对局数
    """
    # 初始化网络
    network = AlphaZeroNetwork(
        state_shape=env.state_shape,
        action_size=env.action_size
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    
    # MCTS
    mcts = MCTS(network, num_simulations=800)
    
    replay_buffer = []
    
    for iteration in range(num_iterations):
        print(f"迭代 {iteration + 1}/{num_iterations}")
        
        # 自博弈：生成训练数据
        for game_idx in range(games_per_iteration):
            state = env.reset()
            game_data = []
            
            while not env.is_terminal(state):
                # 使用 MCTS 选择动作
                action_probs = mcts.search(state, env)
                
                # 保存 (state, action_probs) 用于训练
                game_data.append((state.copy(), action_probs.copy()))
                
                # 采样动作（训练时加入温度参数）
                action = np.random.choice(len(action_probs), p=action_probs)
                
                # 执行动作
                state, reward, done = env.step(action)
            
            # 对局结束，获得最终奖励
            final_reward = env.get_reward()
            
            # 为每个状态分配奖励（从当前玩家视角）
            for i, (s, pi) in enumerate(game_data):
                # 交替玩家：奖励符号翻转
                z = final_reward if i % 2 == (len(game_data) - 1) % 2 else -final_reward
                replay_buffer.append((s, pi, z))
        
        # 训练网络
        # 从经验回放缓冲区采样批次
        batch_size = 256
        num_epochs = 10
        
        for epoch in range(num_epochs):
            np.random.shuffle(replay_buffer)
            
            for i in range(0, len(replay_buffer), batch_size):
                batch = replay_buffer[i:i + batch_size]
                
                states = torch.FloatTensor([x[0] for x in batch])
                target_pis = torch.FloatTensor([x[1] for x in batch])
                target_values = torch.FloatTensor([[x[2]] for x in batch])
                
                # 前向传播
                pred_pis, pred_values = network(states)
                
                # 损失函数
                policy_loss = -torch.mean(torch.sum(target_pis * torch.log(pred_pis + 1e-8), dim=1))
                value_loss = F.mse_loss(pred_values, target_values)
                
                total_loss = policy_loss + value_loss
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
        
        print(f"策略损失: {policy_loss.item():.4f}, 价值损失: {value_loss.item():.4f}")
        
        # 限制缓冲区大小
        if len(replay_buffer) > 100000:
            replay_buffer = replay_buffer[-100000:]
```

<div data-component="SelfPlayEvolution"></div>

### 28.1.3 自博弈的挑战

**1. 遗忘（Forgetting）**
- 问题：智能体可能忘记如何对抗旧策略
- 解决方案：与历史快照对战

**2. 策略崩溃（Policy Collapse）**
- 问题：收敛到局部最优或循环策略
- 解决方案：种群训练、多样性正则化

**3. 样本效率**
- 问题：需要大量自博弈对局
- 解决方案：优先经验回放、模型辅助

---

## 28.2 种群训练（Population-Based Training, PBT）

### 28.2.1 核心思想

**PBT** 同时训练一个**智能体种群**，定期让表现差的智能体**复制**表现好的智能体并**变异**超参数。

**优势**：
- **自动超参数调优**：无需手动网格搜索
- **多样性维护**：种群包含不同策略
- **适应性**：超参数随训练动态调整

**算法流程**：

1. 初始化种群（$N$ 个智能体，随机超参数）
2. 并行训练所有智能体
3. 定期评估：
   - 排序智能体（按适应度）
   - 底部 $k\%$ **利用**（exploit）顶部智能体：
     - 复制权重
     - 扰动超参数（**探索**, explore）
4. 重复步骤 2-3

**超参数变异**：

$$
\theta_{\text{new}} = \begin{cases}
\theta_{\text{parent}} \times 1.2 & \text{概率} \ 0.25 \\
\theta_{\text{parent}} \times 0.8 & \text{概率} \ 0.25 \\
\theta_{\text{parent}} & \text{概率} \ 0.5
\end{cases}
$$

**代码实现**：

```python
import copy

class PBTWorker:
    """PBT 中的单个工作器（智能体+超参数）"""
    def __init__(self, agent, hyperparams):
        self.agent = agent
        self.hyperparams = hyperparams
        self.fitness = 0.0  # 性能指标（例如平均奖励）
    
    def train(self, num_steps):
        """训练智能体 num_steps 步"""
        # 使用当前超参数训练
        for step in range(num_steps):
            # 采样经验并更新策略
            experience = self.agent.collect_experience()
            self.agent.update(experience, lr=self.hyperparams['lr'])
        
        # 评估fitness
        self.fitness = self.agent.evaluate()
    
    def exploit_and_explore(self, parent_worker):
        """从更好的父代复制并扰动超参数"""
        # 利用：复制父代权重
        self.agent.load_weights(parent_worker.agent.get_weights())
        
        # 探索：扰动超参数
        for key, value in self.hyperparams.items():
            rand = np.random.rand()
            if rand < 0.25:
                self.hyperparams[key] = value * 1.2  # 增加
            elif rand < 0.5:
                self.hyperparams[key] = value * 0.8  # 减少
            # 否则保持不变


def population_based_training(population_size=10, num_generations=100, steps_per_gen=1000):
    """
    种群训练主循环
    
    Args:
        population_size: 种群中的智能体数量
        num_generations: 训练代数
        steps_per_gen: 每代训练步数
    """
    # 初始化种群：随机超参数
    population = []
    for _ in range(population_size):
        hyperparams = {
            'lr': np.random.uniform(1e-4, 1e-2),
            'gamma': np.random.uniform(0.95, 0.99),
            'entropy_coef': np.random.uniform(0.001, 0.1)
        }
        agent = PPOAgent(hyperparams)  # 假设 PPOAgent 已定义
        worker = PBTWorker(agent, hyperparams)
        population.append(worker)
    
    for generation in range(num_generations):
        print(f"代数 {generation + 1}/{num_generations}")
        
        # 1. 并行训练所有工作器
        for worker in population:
            worker.train(steps_per_gen)
        
        # 2. 按适应度排序
        population.sort(key=lambda w: w.fitness, reverse=True)
        
        # 3. 利用与探索
        # 底部 20% 从顶部 20% 复制
        cutoff = int(0.2 * population_size)
        for i in range(population_size - cutoff, population_size):
            # 随机选择一个顶部工作器作为父代
            parent_idx = np.random.randint(0, cutoff)
            population[i].exploit_and_explore(population[parent_idx])
        
        # 打印最佳适应度
        print(f"最佳适应度: {population[0].fitness:.2f}")
        print(f"最佳超参数: {population[0].hyperparams}")
    
    # 返回最佳智能体
    return population[0].agent
```

<div data-component="PopulationDiversity"></div>

---

## 28.3 联盟训练（League Training）

### 28.3.1 AlphaStar 联盟架构

**联盟训练**（由 DeepMind 的 AlphaStar 提出）维护**多个智能体种群**：

1. **主智能体（Main Agents）**
   - 持续改进的主要策略
   - 对抗多样化的对手

2. **主利用者（Main Exploiters）**
   - 专门寻找主智能体的弱点
   - 防止主智能体被利用

3. **联盟利用者（League Exploiters）**
   - 寻找整个联盟的漏洞
   - 确保鲁棒性

4. **历史智能体（Historical Players）**
   - 过去主智能体的冻结快照
   - 防止遗忘

**对手采样策略**（主智能体）：
- 35%：当前自己（自博弈）
- 50%：历史快照
- 15%：利用者

**优势**：
- **防止利用**：利用者持续测试弱点
- **多样性**：不同种群提供不同挑战
- **长期记忆**：历史快照防止遗忘

**代码示例**：

```python
class LeagueTraining:
    """AlphaStar 风格的联盟训练"""
    def __init__(self):
        self.main_agents = [Agent() for _ in range(3)]
        self.main_exploiters = [Agent() for _ in range(2)]
        self.league_exploiters = [Agent() for _ in range(2)]
        self.historical = []  # 历史快照
    
    def sample_opponent(self, agent_type):
        """
        为给定智能体类型采样对手
        
        Args:
            agent_type: 'main', 'main_exploiter', 或 'league_exploiter'
        
        Returns:
            opponent: 采样的对手智能体
        """
        if agent_type == 'main':
            # 主智能体的采样分布
            rand = np.random.rand()
            
            if rand < 0.35:
                # 35%: 自博弈
                return np.random.choice(self.main_agents)
            elif rand < 0.85:
                # 50%: 历史快照
                if self.historical:
                    return np.random.choice(self.historical)
                else:
                    return np.random.choice(self.main_agents)
            else:
                # 15%: 利用者
                all_exploiters = self.main_exploiters + self.league_exploiters
                return np.random.choice(all_exploiters)
        
        elif agent_type == 'main_exploiter':
            # 主利用者仅对抗主智能体
            return np.random.choice(self.main_agents)
        
        else:  # league_exploiter
            # 联盟利用者对抗整个联盟
            all_agents = self.main_agents + self.main_exploiters + self.historical
            return np.random.choice(all_agents)
    
    def train_step(self):
        """执行一个训练步骤"""
        # 训练主智能体
        for main_agent in self.main_agents:
            opponent = self.sample_opponent('main')
            # 对战并更新
            self.play_and_update(main_agent, opponent)
        
        # 训练主利用者
        for exploiter in self.main_exploiters:
            opponent = self.sample_opponent('main_exploiter')
            self.play_and_update(exploiter, opponent)
        
        # 训练联盟利用者
        for exploiter in self.league_exploiters:
            opponent = self.sample_opponent('league_exploiter')
            self.play_and_update(exploiter, opponent)
    
    def create_snapshot(self, interval=1000):
        """定期创建主智能体的历史快照"""
        for agent in self.main_agents:
            snapshot = copy.deepcopy(agent)
            snapshot.freeze()  # 冻结权重
            self.historical.append(snapshot)
        
        # 限制历史大小
        if len(self.historical) > 50:
            self.historical = self.historical[-50:]
```

<div data-component="LeagueTrainingArchitecture"></div>

---

## 28.4 涌现行为

### 28.4.1 什么是涌现？

**涌现行为**是指从简单规则和目标中自然产生的复杂策略，而无需显式编程。

**特征**：
- **未预见**：设计者未明确编码
- **复杂性**：超越初始规则
- **竞争驱动**：通过对抗性压力演化

### 28.4.2 OpenAI 捉迷藏案例

**环境**：
- **隐藏者（Hiders）**：建造庇护所并躲藏
- **寻找者（Seekers）**：找到隐藏者
- **对象**：可移动的箱子和斜坡

**6 个涌现阶段**（3.8亿+ 时间步）：

1. **阶段 1（0-10M）**：奔跑与追逐
   - 隐藏者随机移动
   - 寻找者追逐最近的隐藏者

2. **阶段 2（10-25M）**：庇护所建造
   - 隐藏者学会**推箱子建墙**！
   - 寻找者仍在基础追逐

3. **阶段 3（25-75M）**：斜坡工具使用
   - 寻找者发现可以**使用斜坡爬过墙**
   - 隐藏者的庇护所被破解

4. **阶段 4（75-130M）**：锁定斜坡
   - 隐藏者学会**先锁住斜坡再建庇护所**
   - 防止寻找者使用斜坡

5. **阶段 5（130-380M）**：箱子冲浪（物理漏洞利用）
   - 寻找者发现物理漏洞：**在移动的箱子上冲浪**
   - 可以绕过锁定的斜坡

6. **阶段 6（380M+）**：锁定所有对象
   - 隐藏者学会**系统性地锁住所有可移动对象**
   - 终极防御策略

<div data-component="EmergentBehaviorDemo"></div>

### 28.4.3 涌现的启示

**关键要素**：
1. **长期训练**：数亿时间步
2. **竞争压力**：对抗性迫使适应
3. **丰富环境**：提供创造性解决方案的工具
4. **最小假设**：让策略自然演化

**应用**：
- **机器人**：涌现运动技能（行走、抓取）
- **游戏 AI**：发现人类未预见的策略
- **多智能体系统**：涌现通信和协调

---

## 总结

本章涵盖了自博弈与涌现行为：

1. **自博弈训练**：AlphaZero MCTS 算法，通过与自己对战学习
2. **种群训练（PBT）**：同时训练多个智能体，自动调优超参数
3. **联盟训练**：AlphaStar 架构，使用多种群防止利用
4. **涌现行为**：OpenAI 捉迷藏中从简单目标涌现出复杂策略

**关键要点**：
- 自博弈创建自动课程，无需人类数据
- 种群方法提高鲁棒性和多样性
- 涌现行为展示 RL 发现新颖解决方案的潜力
- 长期训练和竞争压力是涌现的关键

**下一章**：第 29 章探讨**合作多智能体任务**。

---

## 参考文献

- Silver, D., et al. (2017). "Mastering Chess and Shogi by Self-Play with a General RL Algorithm (AlphaZero)." *arXiv*.
- Silver, D., et al. (2018). "A General RL Algorithm that Masters Chess, Shogi, and Go through Self-Play." *Science*.
- Jaderberg, M., et al. (2017). "Population Based Training of Neural Networks." *arXiv*.
- Vinyals, O., et al. (2019). "Grandmaster level in StarCraft II using multi-agent RL (AlphaStar)." *Nature*.
- Baker, B., et al. (2020). "Emergent Tool Use From Multi-Agent Autocurricula (Hide-and-Seek)." *ICLR*.

---
title: "第0章：强化学习概览"
description: "从零开始理解强化学习：核心概念、历史发展、应用场景与环境准备"
date: "2026-01-30"
---

# 第0章：强化学习概览

欢迎来到强化学习（Reinforcement Learning, RL）的世界。在这一章中，我们将从宏观视角理解 RL 的本质、发展历程以及它与其他机器学习范式的区别，并动手搭建第一个 RL 程序。

---

## 0.1 什么是强化学习？

### 0.1.1 与监督学习、无监督学习的区别

机器学习的三大范式各有特点：

| 范式 | 数据形式 | 学习目标 | 典型应用 | 核心挑战 |
|:---|:---|:---|:---|:---|
| **监督学习** | $(x, y)$ 标注对 | 学习映射 $f: x \to y$ | 图像分类、语音识别 | 需要大量标注数据 |
| **无监督学习** | $x$ 无标签数据 | 发现数据内在结构 | 聚类、降维、生成 | 评估标准模糊 |
| **强化学习** | $(s, a, r, s')$ 交互序列 | 最大化长期累积奖励 | 游戏、机器人、推荐 | 延迟奖励、探索-利用权衡 |

**强化学习的核心特征**：
1.  **试错学习（Trial-and-Error）**：Agent 通过尝试不同的动作来发现哪些行为能带来高奖励。
2.  **延迟奖励（Delayed Reward）**：当前动作的影响可能在多步之后才显现（如下棋中的一步好棋可能在 20 步后才体现价值）。
3.  **探索-利用权衡（Exploration-Exploitation）**：既要利用已知的好策略（Exploitation）以获取高分，又要尝试未知区域（Exploration）以发现潜藏的更优策略。这就像去餐厅点菜：是点以前吃过觉得好吃的（Exploitation），还是尝试一道新菜（Exploration）？

> **如果把人生看作是一个强化学习问题...**
> 
> "小时候我们通过不停地摔倒学会走路（试错），上学备考时为了高分刷题（延迟奖励），工作后在稳定薪水和创业梦想之间抉择（探索与利用）。RL 不仅仅是一种算法，更是理解智能体如何适应世界的通用框架。"

**类比理解**：
- **监督学习** 像在学校听老师讲课，老师直接告诉你"这道题答案是 A"。
- **强化学习** 像婴儿学走路，摔倒了（负奖励），站稳了（正奖励），通过不断试错逐渐掌握平衡技巧，没有老师逐步指导。

### 0.1.2 核心要素：Agent, Environment, State, Action, Reward

强化学习的过程可以用一个标准的五元组 $(S, A, P, R, \gamma)$ 来形式化描述，但在此概览章节，我们先从直观概念入手：

```mermaid
graph LR
    Agent[Agent<br/>(智能体)] -- Action $a_t$ --> Env[Environment<br/>(环境)]
    Env -- State $s_{t+1}$ <br/> Reward $r_{t+1}$ --> Agent
    style Agent fill:#f9f,stroke:#333,stroke-width:2px
    style Env fill:#ccf,stroke:#333,stroke-width:2px
```

<div data-component="AgentEnvironmentLoop"></div>

**1. Agent（智能体）**
感知环境并做出决策的主体。在代码中，它通常是一个神经网络（Policy Network）或一个查找表（Q-Table）。

**2. Environment（环境）**
Agent 所在的外部世界。它遵循物理定律或游戏规则，接收 Agent 的动作，反馈新的状态和奖励。环境对 Agent 来说通常是**黑盒（Black Box）**，Agent 必须透过交互来"探测"环境的反应。

**3. State（状态）$s$ vs Observation（观测）$o$**
*   **State ($s$)**：对环境状况的**完整、客观描述**。例如：AlphaGo 能看到整个棋盘。
*   **Observation ($o$)**：Agent 主观观察到的**部分信息**。例如：在 FPS 游戏中，你只能看到屏幕画面，看不到墙后的敌人。
    *   *Fully Observable (MDP)*: $o_t = s_t$
    *   *Partially Observable (POMDP)*: $o_t \neq s_t$，Agent 需要靠记忆力推断真实状态。

**4. Action（动作）$a$**
*   **离散动作 (Discrete)**：有限个选择（如：上下左右、开火）。
*   **连续动作 (Continuous)**：实数值向量（如：方向盘转角 $\in [-540^\circ, 540^\circ]$，油门力度 $\in [0, 1]$）。

**5. Reward（奖励）$r$**
环境给出的标量反馈信号，用于评估动作的好坏。
> **The Reward Hypothesis (奖励假说)**
> 
> "所有我们所说的'目标'和'目的'，都可以被归结为：最大化接收到的标量信号（奖励）的累积和。" —— *Richard Sutton*

*   **稀疏奖励 (Sparse Reward)**：很难获得反馈（如：只有赢了才有 +1，其他时候全是 0）。
*   **密集奖励 (Dense Reward)**：每一步都有反馈（如：离目标越近分越高）。

### 0.1.3 RL 的应用场景

强化学习在以下领域取得了突破性成果：

**1. 游戏 AI**
*   **Atari 游戏**：DeepMind 的 DQN 算法（2015）在 49 款 Atari 游戏中达到或超越人类水平。
*   **围棋**：AlphaGo（2016）击败世界冠军李世石，AlphaZero（2017）仅通过自我对弈超越所有人类围棋知识。
*   **星际争霸 II**：AlphaStar（2019）达到职业选手水平。
*   **Dota 2**：OpenAI Five（2019）在 5v5 团队对抗中击败世界冠军队伍。

**2. 机器人控制**
*   **机械臂抓取**：通过 RL 学习精确抓取各种形状的物体（RGB-Stacking, RoboSumo）。
*   **四足机器人步态**：训练机器狗在复杂地形中行走、跳跃（ANYmal, Spot）。
*   **人形机器人**：学习保持平衡、行走、翻滚等复杂运动（Humanoid, Atlas）。

**3. 推荐系统**
*   **YouTube 视频推荐**：将推荐问题建模为序列决策，最大化用户长期观看时长而非短期点击率。
*   **电商广告投放**：动态调整广告出价策略，平衡曝光量与转化率。

**4. 大语言模型 (LLM) 与推理 (Reasoning)**
*   **RLHF (InstructGPT)**：通过 PPO 算法，让模型学会遵循人类指令，减少有害输出。
*   **Reasoning Chains (OpenAI o1 / DeepSeek-R1)**：
    *   最新的前沿方向是 **Training-time / Test-time Scaling**。
    *   利用 RL 让模型探索思维链（Chain of Thought），这类训练通常使用**过程奖励（Process Reward）**或**结果验证（Outcome Verification）**。
    *   *Self-Correction*：模型在推理过程中通过 RL 学会了自我检查和修正错误。

**5. 其他应用**
*   **自动驾驶**：路径规划、决策制定。
*   **金融交易**：自动化交易策略优化。
*   **能源调度**：数据中心冷却系统优化（Google 使用 RL 节省 40% 能耗）。

### 0.1.4 RL 的挑战

尽管 RL 强大，但也面临诸多挑战：

| 挑战 | 描述 | 解决方向 |
|:---|:---|:---|
| **延迟奖励 (Credit Assignment)** | "功成不必在我"：很难判断哪一步操作对最终胜利贡献最大。 | 价值函数估计、TD($\lambda$)、GAE |
| **探索-利用权衡** | 走老路稳妥但无法更强，探新路可能踩坑。 | Entropy Regularization、UCB、Intrinsic Motivation (好奇心) |
| **样本效率低** | 需要海量交互（DQN 玩 Atari 需数百万帧）。 | Model-based RL、Transfer Learning |
| **Sim-to-Real Gap** | 仿真环境里练得好好的，上真机就偏瘫。 | Domain Randomization、Real-world Fine-tuning |
| **The Deadly Triad (死亡三角)** | 当同时结合 **Function Approximation**、**Bootstrapping**、**Off-policy** 时，训练极易发散。 | 目标网络 (Target Network)、Double Q-learning |

---

## 0.2 历史发展脉络

<div data-component="RLTimelineEvolution"></div>

### 0.2.1 早期：动态规划（Bellman, 1950s）

Richard Bellman 在 1957 年提出了**最优控制理论**和**动态规划（Dynamic Programming, DP）**方法，奠定了 RL 的数学基础。

**核心贡献**：
*   **Bellman 方程**：状态价值 $V(s)$ 可以被分解为"即时奖励"与"未来价值"两部分，这种递归结构是 RL 最核心的数学支柱。
    $$ V(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s') \right] $$

**局限**：需要完整的环境模型（转移概率 $P$），且计算复杂度随状态数指数增长（"维度灾难"）。

### 0.2.2 表格方法：Q-learning（Watkins, 1989）

Christopher Watkins 在其博士论文中提出了 **Q-learning** 算法，这是第一个 **model-free**（无需环境模型）且 **off-policy**（可以从历史数据中学习）的 RL 算法。

**核心思想**：
*   维护一个表格 $Q(s, a)$，记录在状态 $s$ 采取动作 $a$ 的价值。
*   通过与环境交互，逐步更新 $Q$ 值：
    $$ Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)] $$
*   **收敛性保证**：在满足一定条件下（所有状态-动作对被无限次访问），$Q$ 表格会收敛到最优值 $Q^*$。

**局限**：只能处理小规模离散状态空间（如 10x10 网格世界）。对于 Atari 游戏（$256^{84 \times 84 \times 4} \approx 10^{67970}$ 种状态），表格方法完全不可行。

### 0.2.3 深度 RL 时代：DQN（Mnih et al., 2015）

DeepMind 的 Volodymyr Mnih 等人在 *Nature* 发表了划时代的论文《Human-level control through deep reinforcement learning》，提出 **DQN（Deep Q-Network）**。

**核心创新**：
1.  **深度神经网络作为函数逼近器**：用卷积神经网络（CNN）来近似 $Q(s, a)$，输入是游戏画面（84x84 灰度图），输出是每个动作的 Q 值。
2.  **经验回放（Experience Replay）**：将 $(s, a, r, s')$ 存入回放缓冲区，训练时随机采样，打破数据相关性。
3.  **目标网络（Target Network）**：计算 TD 目标时使用一个参数固定的网络 $Q_{\text{target}}$，切断了 Bootstrapping 中的正反馈循环，显著提升了稳定性。

**成果**：在 49 款 Atari 游戏中，29 款达到或超过人类专业玩家水平。

### 0.2.4 策略优化：PPO（Schulman et al., 2017）

虽然 DQN 在 Atari 上成功，但它只适用于离散动作空间。对于连续控制（如机器人），需要**策略梯度（Policy Gradient）**方法。

John Schulman 等人在 OpenAI 提出了 **PPO（Proximal Policy Optimization）**，成为工业界默认的首选算法。

**核心思想**：
*   直接优化策略 $\pi_\theta(a|s)$，而不是中间的价值函数。
*   使用"裁剪目标函数"限制策略更新幅度，防止更新过大导致性能崩溃：
    $$ L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right] $$
    其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 是新旧策略的比率。

**优势**：简单、稳健、易于调参，适用于几乎所有任务。

### 0.2.5 LLM 对齐：RLHF（OpenAI, 2022）

**RLHF（Reinforcement Learning from Human Feedback）** 是 ChatGPT 成功的关键技术之一。

**流程**：
1.  **监督微调（SFT）**：在高质量对话数据上微调预训练模型。
2.  **奖励模型训练**：人类标注者对模型输出进行排序（A 比 B 好），训练奖励模型 $r_\phi(x, y)$。
3.  **RL 优化**：使用 PPO 优化语言模型，最大化奖励同时保持与初始模型的 KL 散度约束：
    $$ \max_{\pi_\theta} \mathbb{E}_{x, y \sim \pi_\theta} [r_\phi(x, y)] - \beta \, \text{KL}(\pi_\theta || \pi_{\text{ref}}) $$

**成果**：使 GPT-3、GPT-4、Claude、Gemini 等大模型的输出更符合人类偏好，减少有害内容。

---

## 0.3 核心概念预览

在深入学习 RL 算法之前，我们需要理解一些核心概念。这些概念将在后续章节中详细展开。

### 0.3.1 价值函数 vs 策略

RL 算法的核心是学习一个"好"的策略。实现方式有两种：

| 方法 | 学习目标 | 代表算法 | 优势 | 劣势 |
|:---|:---|:---|:---|:---|
| **Value-based** | 学习价值函数 $V(s)$ 或 $Q(s,a)$ | Q-learning, DQN | 样本效率高 | 仅适用于离散动作 |
| **Policy-based** | 直接学习策略 $\pi_\theta(a|s)$ | REINFORCE, PPO | 适用于连续动作、随机策略 | 高方差、样本效率低 |
| **Actor-Critic** | 同时学习价值函数和策略 | A3C, SAC | 结合两者优点 | 实现复杂 |

**形象类比**：
*   **Value-based**：你在玩一款策略游戏，通过大量对局记住了"在这个局面下赢的概率是 80%"这样的经验（价值），下棋时选择价值最高的走法。
*   **Policy-based**：你直接学习"看到这个局面就走这一步"的反射性策略，不需要显式地评估每一步的价值。

### 0.3.2 On-policy vs Off-policy

| 类型 | 定义 | 数据来源 | 代表算法 | 优势 | 劣势 |
|:---|:---|:---|:---|:---|
| **On-policy** | 用当前策略采集数据并更新该策略 | 必须是当前策略产生的新鲜数据 | SARSA, PPO | 稳定性好 | 样本效率低（数据用一次就扔） |
| **Off-policy** | 从任意策略（包括历史数据）中学习 | 可以是其他策略甚至人类演示 | Q-learning, SAC, DQN | 样本效率高、可复用数据 | 需要重要性采样修正、可能不稳定 |

**实际意义**：Off-policy 算法可以利用 **Experience Replay Buffer**（经验回放缓冲区），大幅提升样本效率。这在高成本环境（如真实机器人）中尤为重要。

### 0.3.3 Model-free vs Model-based

| 类型 | 是否学习环境模型 | 代表算法 | 优势 | 劣势 |
|:---|:---|:---|:---|:---|
| **Model-free** | 否，直接学习价值/策略 | Q-learning, PPO, SAC | 通用性强、实现简单 | 样本效率低 |
| **Model-based** | 是，学习 $P(s'|s,a)$ 和 $R(s,a)$ | Dyna, MBPO, Dreamer | 样本效率高、可规划 | 模型误差累积、实现复杂 |

**形象类比**：
*   **Model-free**：婴儿摸到火炉知道疼（直接经验），不需要理解"火的温度"和"皮肤损伤"的因果机制。
*   **Model-based**：物理学家建立热传导模型，可以预测"如果把手放在 500°C 的表面上 3 秒会怎样"，然后规划避免接触。

### 0.3.4 Sample efficiency vs Asymptotic performance

这是 RL 算法设计中的**核心权衡**：

*   **Sample efficiency（样本效率）**：达到特定性能所需的交互样本数。
    *   **高样本效率**：Model-based RL、Off-policy 算法（如 SAC）。
    *   **低样本效率**：On-policy 算法（如 PPO）、Policy Gradient。
*   **Asymptotic performance（渐近性能）**：在样本数趋于无穷时能达到的最优性能。

通常存在 **trade-off**：
*   **PPO**：样本效率较低，但渐近性能通常很好（最终能找到接近最优的策略）。
*   **SAC**：样本效率高（因为 off-policy），但在某些任务上最终性能可能不如 PPO。

---

## 0.4 环境准备

### 0.4.1 Gymnasium：现代 RL 环境标准

**Gymnasium** 是 OpenAI Gym 的继任者，由 Farama Foundation 维护，是目前 RL 社区的事实标准。

**安装**：
```bash
pip install gymnasium[all]
```

**核心接口**：
```python
import gymnasium as gym

# 创建环境
env = gym.make("CartPole-v1", render_mode="human")

# 重置环境，返回初始状态和信息字典
observation, info = env.reset(seed=42)

# 与环境交互
for _ in range(100):
    # 采样随机动作（从动作空间中均匀采样）
    action = env.action_space.sample()
    
    # 执行动作，返回五元组
    observation, reward, terminated, truncated, info = env.step(action)
    
    # terminated: 环境达到终止状态（如游戏结束）
    # truncated: 达到最大步数限制（超时）
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

**关键概念**：
*   **`observation_space`**：状态空间的规范。
    *   `Box`：连续空间，如 `Box(low=-1, high=1, shape=(4,))` 表示 4 维连续向量，每维范围 `[-1, 1]`。
    *   `Discrete(n)`：离散空间，如 `Discrete(4)` 表示 `{0, 1, 2, 3}` 四个选项。
*   **`action_space`**：动作空间的规范，格式同上。
*   **`reward_range`**：奖励的理论范围（用于归一化）。

### 0.4.2 常用环境介绍

**1. 经典控制（Classic Control）**
*   **CartPole-v1**：平衡倒立摆，最简单的入门环境。
    *   **Obs**: 4D（位置、速度、角度、角速度）
    *   **Action**: Discrete(2)（左推、右推）
    *   **Success**: 坚持 475+ 步
*   **MountainCar-v0**：推小车上山，奖励稀疏的经典难题。
*   **Pendulum-v1**：连续控制的钟摆平衡。

**2. Atari 游戏**
```bash
pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]
```
```python
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
```
*   **特点**：高维图像输入（210x160x3 RGB），需要预处理（灰度化、裁剪、缩放至 84x84）。

**3. MuJoCo 连续控制**
```bash
pip install gymnasium[mujoco]
```
```python
env = gym.make("HalfCheetah-v4")
```
*   **特点**：物理仿真环境，用于机器人步态控制、机械臂抓取等。

### 0.4.3 PyTorch 框架

我们将使用 **PyTorch** 作为深度学习框架（也可使用 JAX/TensorFlow，但 PyTorch 更直观）。

```bash
pip install torch torchvision
```

**基础代码模板**：
```python
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    """简单的 Q 网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.net(state)

# 初始化
state_dim = 4
action_dim = 2
q_net = QNetwork(state_dim, action_dim)
optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
```

### 0.4.4 第一个 RL 程序：Random Agent

让我们运行一个完全随机的 Agent，作为后续算法的 Baseline。

```python
"""
随机 Agent：在 CartPole 环境中采取随机动作
预期结果：平均约 20-30 步就会失败（远低于 475 步的成功线）
"""
import gymnasium as gym
import numpy as np

def run_random_agent(env_name="CartPole-v1", num_episodes=10):
    env = gym.make(env_name)
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # 随机选择动作
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(total_reward)
        print(f"Episode {episode+1}: Reward = {total_reward}")
    
    env.close()
    print(f"\n平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    return episode_rewards

if __name__ == "__main__":
    rewards = run_random_agent(num_episodes=10)
```

**预期输出**：
```
Episode 1: Reward = 23.0
Episode 2: Reward = 31.0
Episode 3: Reward = 18.0
...
平均奖励: 24.30 ± 7.15
```

这个 24 分就是我们的 **Baseline**。在后续章节中，我们将学习各种算法，目标是达到 475+ 的"解决"标准。

---

## 本章总结

本章我们学习了：
1.  **RL 的本质**：通过与环境交互学习最优策略，核心是试错学习和延迟奖励。
2.  **历史脉络**：从 Bellman 的动态规划，到 DQN 的深度突破，再到 RLHF 驱动的 LLM 对齐。
3.  **核心概念**：Value vs Policy、On-policy vs Off-policy、Model-free vs Model-based。
4.  **实践环境**：Gymnasium、MuJoCo、PyTorch。

**下一章预告**：我们将深入学习强化学习的数学基础——**马尔可夫决策过程（MDP）**，形式化定义状态、动作、奖励和策略，并推导 Bellman 方程。

---

## 扩展阅读

*   Sutton & Barto, *Reinforcement Learning: An Introduction*, Chapter 1
*   OpenAI Spinning Up: [Introduction to RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
*   DeepMind x UCL RL Course: Lecture 1 (David Silver)
*   论文：Mnih et al., "Human-level control through deep reinforcement learning", *Nature* 2015

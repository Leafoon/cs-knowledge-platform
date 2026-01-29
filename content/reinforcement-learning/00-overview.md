---
title: "Chapter 0. 强化学习概览"
description: "理解强化学习的本质、历史发展与核心概念"
updated: "2026-01-29"
---

> **Learning Objectives**
> * 理解强化学习与监督学习、无监督学习的本质区别
> * 掌握 Agent-Environment 交互循环的核心机制
> * 了解强化学习的历史发展脉络与重要里程碑
> * 配置 Gymnasium 环境并运行第一个 RL 程序

---

## 0.1 什么是强化学习？

### 0.1.1 与监督学习、无监督学习的区别

想象你正在教一个孩子学习骑自行车。你不会给他一本"正确骑车姿势大全"让他背诵（这是监督学习），也不会让他自己发现自行车的结构规律（这是无监督学习）。相反，你会让他**尝试骑车，摔倒了爬起来再试，逐渐学会平衡**——这就是强化学习。

**强化学习（Reinforcement Learning, RL）** 是一种通过**试错（Trial-and-Error）**和**延迟奖励（Delayed Reward）**来学习最优行为的机器学习范式。

| 学习范式 | 数据形式 | 学习目标 | 典型应用 |
|---------|---------|---------|---------|
| **监督学习** | (x, y) 标注对 | 学习 x → y 的映射 | 图像分类、语音识别 |
| **无监督学习** | x 无标注数据 | 发现数据结构/模式 | 聚类、降维 |
| **强化学习** | (s, a, r, s') 交互序列 | 学习最大化累积奖励的策略 | 游戏AI、机器人控制 |

**核心区别**：
- 监督学习：有"老师"告诉你正确答案
- 无监督学习：没有"老师"，自己发现规律
- 强化学习：有"评委"（环境）给你打分，但不告诉你正确答案

### 0.1.2 核心要素：Agent、Environment、State、Action、Reward

强化学习的核心是 **Agent（智能体）** 与 **Environment（环境）** 的交互循环：

```
    ┌─────────┐
    │  Agent  │
    └────┬────┘
         │ Action (a_t)
         ▼
    ┌─────────┐
    │   Env   │
    └────┬────┘
         │ State (s_{t+1}), Reward (r_{t+1})
         ▼
    ┌─────────┐
    │  Agent  │
    └─────────┘
```

**五大核心要素**：

1. **Agent（智能体）**：学习者和决策者
   - 例如：游戏中的AI玩家、自动驾驶汽车、机器人

2. **Environment（环境）**：Agent 交互的外部世界
   - 例如：游戏规则、物理世界、股票市场

3. **State（状态）** $s_t$：环境在时刻 $t$ 的表示
   - 例如：棋盘局面、机器人位置、股票价格

4. **Action（动作）** $a_t$：Agent 在状态 $s_t$ 下采取的行为
   - 例如：落子位置、方向盘角度、买入/卖出

5. **Reward（奖励）** $r_t$：环境对 Agent 动作的即时反馈
   - 例如：游戏得分、行驶距离、投资收益

### 交互演示：Agent-Environment 循环

<div data-component="AgentEnvironmentLoop"></div>

### 0.1.3 RL 的应用场景

强化学习在以下领域取得了突破性成果：

**🎮 游戏 AI**
- **AlphaGo**（2016）：击败围棋世界冠军李世石
- **OpenAI Five**（2018）：在 Dota 2 中击败人类职业队
- **AlphaStar**（2019）：达到星际争霸 II 大师级水平

**🤖 机器人控制**
- 机械臂抓取与操作
- 四足机器人步态学习
- 无人机自主飞行

**💬 大语言模型对齐**
- **ChatGPT**（2022）：通过 RLHF 对齐人类偏好
- **Claude**（2023）：Constitutional AI
- **GPT-4**（2023）：多模态对齐

**🚗 自动驾驶**
- 路径规划
- 决策控制
- 交通流优化

**💰 金融交易**
- 算法交易
- 投资组合优化
- 风险管理

**📱 推荐系统**
- 个性化推荐
- 广告投放
- 内容排序

### 0.1.4 RL 的挑战

强化学习虽然强大，但也面临诸多挑战：

1. **延迟奖励（Credit Assignment）**
   - 问题：一个动作的后果可能在很久之后才显现
   - 例如：围棋中，开局的一步棋可能影响终局胜负

2. **探索-利用困境（Exploration-Exploitation Dilemma）**
   - 探索（Exploration）：尝试新动作，可能发现更好的策略
   - 利用（Exploitation）：选择已知最优动作，获得即时奖励
   - 权衡：如何平衡两者？

3. **样本效率低（Sample Inefficiency）**
   - 问题：需要大量交互才能学到好策略
   - 例如：AlphaGo 训练了数百万局对弈

4. **奖励稀疏（Sparse Rewards）**
   - 问题：大部分时间奖励为 0，只有达成目标才有奖励
   - 例如：走迷宫，只有到达终点才有奖励

5. **非平稳性（Non-Stationarity）**
   - 问题：环境或策略在变化，目标是移动的
   - 例如：多智能体博弈中，对手策略在进化

> [!TIP]
> **理解 RL 的关键**：强化学习不是寻找"正确答案"，而是通过**试错**学习"好的策略"。这种学习方式更接近人类和动物的学习过程。

---

## 0.2 历史发展脉络

强化学习的发展历程跨越了近70年，经历了从理论奠基到深度学习革命的多个阶段。

### 0.2.1 早期：动态规划（1950s-1980s）

**Richard Bellman（1957）**：提出**动态规划（Dynamic Programming）**理论

- **Bellman 方程**：最优性的数学表达
- **价值迭代**：计算最优策略的算法
- 局限：需要完整的环境模型，计算复杂度高

**关键贡献**：
```
V*(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V*(s')]
```
这个方程奠定了现代 RL 的理论基础。

### 0.2.2 表格方法：Q-learning（1980s-1990s）

**Chris Watkins（1989）**：提出 **Q-learning** 算法

- 无需环境模型（Model-Free）
- 通过采样学习
- Off-policy 学习

**Q-learning 更新规则**：
```
Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
```

**Sutton（1988）**：提出 **TD(λ)** 和**资格迹（Eligibility Traces）**

这一时期的方法主要用于**表格型**问题（状态和动作空间较小）。

### 0.2.3 深度 RL 时代：DQN（2013-2015）

**DeepMind（2013-2015）**：**Deep Q-Network (DQN)**

- 将深度神经网络与 Q-learning 结合
- **Experience Replay**：打破数据相关性
- **Target Network**：稳定训练
- 在 Atari 游戏上达到人类水平

**Nature 论文（2015）**：
- 标题：*Human-level control through deep reinforcement learning*
- 影响：开启深度 RL 时代

**关键创新**：
```python
# Experience Replay
replay_buffer.store(s, a, r, s', done)
batch = replay_buffer.sample(batch_size)

# Target Network
target_q = r + gamma * target_net(s').max()
loss = (q_net(s)[a] - target_q)^2
```

### 0.2.4 策略优化：PPO（2015-2017）

**OpenAI（2015-2017）**：策略梯度方法的突破

- **TRPO**（2015）：Trust Region Policy Optimization
  - 单调改进保证
  - 但计算复杂

- **PPO**（2017）：Proximal Policy Optimization
  - 简化 TRPO
  - 易于实现，性能优异
  - 成为工业界标准

**PPO Clip 机制**：
```
L^CLIP(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
```

**应用**：
- OpenAI Five（Dota 2）
- 机器人控制
- ChatGPT RLHF

### 0.2.5 LLM 对齐：RLHF（2017-2023）

**OpenAI（2017-2022）**：从人类反馈中学习

- **2017**：Deep RL from Human Preferences
- **2020**：GPT-3 + 少量 RLHF
- **2022**：InstructGPT（ChatGPT 前身）
  - 监督微调（SFT）
  - 奖励模型（RM）
  - PPO 强化学习

**ChatGPT（2022.11）**：RLHF 的里程碑应用
- 对齐人类价值观
- 拒绝有害请求
- 提供有用、诚实、无害的回答

**最新进展（2023-2024）**：
- **DPO**（2023）：绕过显式奖励模型
- **Constitutional AI**（2023）：自我批评与改进
- **OpenAI o1**（2024）：推理时 RL

### 交互演示：RL 发展时间线

<div data-component="RLTimelineEvolution"></div>

> [!IMPORTANT]
> **RL 发展的三次浪潮**：
> 1. **理论奠基**（1950s-1990s）：Bellman、Q-learning
> 2. **深度革命**（2013-2017）：DQN、PPO
> 3. **LLM 对齐**（2017-至今）：RLHF、DPO

---

## 0.3 核心概念预览

在深入学习之前，我们先预览几个贯穿整个课程的核心概念。

### 0.3.1 价值函数 vs 策略

强化学习有两种主要的学习目标：

**价值函数（Value Function）**：
- 定义：状态或状态-动作对的"好坏"程度
- 符号：$V(s)$ 或 $Q(s,a)$
- 方法：Q-learning、DQN、TD

**策略（Policy）**：
- 定义：从状态到动作的映射
- 符号：$\pi(a|s)$ 或 $\mu(s)$
- 方法：REINFORCE、PPO、SAC

**对比**：

| 维度 | 价值函数方法 | 策略方法 |
|------|------------|---------|
| 学习目标 | 学习 Q(s,a) | 直接学习 π(a\|s) |
| 动作选择 | ε-greedy / argmax Q | 从 π 采样 |
| 连续动作 | 困难（需离散化） | 自然支持 |
| 随机策略 | 间接 | 直接 |
| 代表算法 | DQN、Q-learning | PPO、SAC |

**Actor-Critic**：结合两者优势
- Actor：策略网络 $\pi(a|s)$
- Critic：价值网络 $V(s)$ 或 $Q(s,a)$

### 0.3.2 On-policy vs Off-policy

**On-policy（同策略）**：
- 定义：用于学习的数据来自**当前策略**
- 特点：数据新鲜，但样本效率低
- 算法：SARSA、A2C、PPO

**Off-policy（异策略）**：
- 定义：用于学习的数据来自**其他策略**（如历史数据）
- 特点：样本效率高，但可能不稳定
- 算法：Q-learning、DQN、SAC

**对比**：

```
On-policy:  生成数据的策略 = 学习的策略
Off-policy: 生成数据的策略 ≠ 学习的策略
```

**实际影响**：
- On-policy：每次策略更新后，旧数据失效，需重新采样
- Off-policy：可以使用 Experience Replay，重复利用历史数据

### 0.3.3 Model-free vs Model-based

**Model-free（无模型）**：
- 定义：不学习环境模型，直接学习策略或价值函数
- 优点：简单，适用性广
- 缺点：样本效率低
- 算法：Q-learning、DQN、PPO、SAC

**Model-based（有模型）**：
- 定义：学习环境模型 $P(s'|s,a)$ 和 $R(s,a)$，然后规划
- 优点：样本效率高，可以想象（Imagination）
- 缺点：模型误差会累积
- 算法：Dyna、MBPO、Dreamer

**对比**：

| 维度 | Model-free | Model-based |
|------|-----------|-------------|
| 学习内容 | 策略/价值函数 | 环境模型 |
| 样本效率 | 低 | 高 |
| 计算复杂度 | 低 | 高 |
| 模型误差 | 无 | 可能累积 |
| 适用场景 | 通用 | 环境可建模 |

### 0.3.4 Sample efficiency vs Asymptotic performance

这是 RL 算法设计中的核心权衡：

**Sample efficiency（样本效率）**：
- 定义：达到一定性能所需的交互次数
- 重要性：真实世界交互成本高（机器人、自动驾驶）
- 高样本效率算法：SAC、TD3、Model-based RL

**Asymptotic performance（渐近性能）**：
- 定义：无限数据下的最终性能
- 重要性：游戏、仿真环境（交互成本低）
- 高渐近性能算法：PPO、A3C

**权衡曲线**：

```
性能
 │     ╱─────  高渐近性能算法（PPO）
 │   ╱
 │ ╱─────────  高样本效率算法（SAC）
 │╱
 └──────────────── 样本数
```

> [!TIP]
> **选择算法的经验法则**：
> - 真实世界（机器人）：优先样本效率 → SAC、Model-based
> - 仿真环境（游戏）：优先渐近性能 → PPO
> - 离线数据：Offline RL → CQL、IQL

### 交互演示：RL 生态全景图

<div data-component="RLEcosystemMap"></div>

---

## 0.4 环境准备

让我们配置 RL 开发环境，并运行第一个程序。

### 0.4.1 Gymnasium（OpenAI Gym 继任者）

**Gymnasium** 是 OpenAI Gym 的官方继任者，提供了标准化的 RL 环境接口。

**安装**：

```bash
# 基础安装
pip install gymnasium

# 包含所有环境（推荐）
pip install "gymnasium[all]"

# Atari 游戏
pip install "gymnasium[atari]"
pip install "gymnasium[accept-rom-license]"

# MuJoCo 物理仿真
pip install "gymnasium[mujoco]"
```

**核心接口**：

```python
import gymnasium as gym

# 创建环境
env = gym.make("CartPole-v1", render_mode="human")

# 重置环境
observation, info = env.reset(seed=42)

# 交互循环
for _ in range(1000):
    # 随机动作
    action = env.action_space.sample()
    
    # 执行动作
    observation, reward, terminated, truncated, info = env.step(action)
    
    # 检查是否结束
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

**环境分类**：

| 类别 | 环境 | 状态空间 | 动作空间 |
|------|------|---------|---------|
| 经典控制 | CartPole, MountainCar | 连续 | 离散 |
| Atari | Pong, Breakout | 图像 | 离散 |
| MuJoCo | HalfCheetah, Ant | 连续 | 连续 |
| Box2D | LunarLander, BipedalWalker | 连续 | 离散/连续 |

### 0.4.2 MuJoCo、Atari、Procgen 环境

**MuJoCo（Multi-Joint dynamics with Contact）**：
- 用途：机器人控制、连续控制
- 特点：高精度物理仿真
- 环境：HalfCheetah、Ant、Humanoid

```python
import gymnasium as gym

env = gym.make("HalfCheetah-v4", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # 连续动作
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

**Atari**：
- 用途：视觉 RL、DQN 基准
- 特点：高维图像输入
- 环境：Pong、Breakout、SpaceInvaders

```python
env = gym.make("ALE/Pong-v5", render_mode="human")
# 状态：(210, 160, 3) RGB 图像
# 动作：离散（6个）
```

**Procgen**：
- 用途：泛化能力测试
- 特点：程序生成关卡，每次不同
- 环境：CoinRun、StarPilot、BigFish

```python
from procgen import ProcgenEnv

env = ProcgenEnv(num_envs=1, env_name="coinrun")
# 测试泛化：训练关卡 vs 测试关卡
```

### 0.4.3 PyTorch、JAX 框架选择

**PyTorch**（推荐初学者）：
- 优点：易用、生态丰富、调试友好
- 缺点：速度稍慢
- 适用：研究、原型开发

```python
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)
```

**JAX**（追求极致性能）：
- 优点：速度快、自动向量化、GPU/TPU 友好
- 缺点：学习曲线陡峭
- 适用：大规模训练、生产部署

```python
import jax
import jax.numpy as jnp
from flax import linen as nn

class QNetwork(nn.Module):
    action_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x
```

### 0.4.4 第一个 RL 程序：Random Agent

让我们运行第一个完整的 RL 程序——一个随机智能体。

```python
import gymnasium as gym
import numpy as np

def random_agent(env_name="CartPole-v1", num_episodes=10):
    """
    随机智能体：在环境中随机选择动作
    
    Args:
        env_name: 环境名称
        num_episodes: 运行的 episode 数量
    """
    # 创建环境
    env = gym.make(env_name)
    
    print(f"环境: {env_name}")
    print(f"状态空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    print("-" * 50)
    
    # 记录统计信息
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        # 重置环境
        observation, info = env.reset()
        
        episode_reward = 0
        episode_length = 0
        terminated = False
        truncated = False
        
        # Episode 循环
        while not (terminated or truncated):
            # 随机选择动作
            action = env.action_space.sample()
            
            # 执行动作
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}: "
              f"Reward = {episode_reward:.2f}, "
              f"Length = {episode_length}")
    
    env.close()
    
    # 打印统计信息
    print("-" * 50)
    print(f"平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"平均长度: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    
    return episode_rewards, episode_lengths

# 运行
if __name__ == "__main__":
    rewards, lengths = random_agent("CartPole-v1", num_episodes=10)
```

**预期输出**：

```
环境: CartPole-v1
状态空间: Box([-4.8 -inf -0.42 -inf], [4.8 inf 0.42 inf], (4,), float32)
动作空间: Discrete(2)
--------------------------------------------------
Episode 1: Reward = 22.00, Length = 22
Episode 2: Reward = 15.00, Length = 15
Episode 3: Reward = 28.00, Length = 28
Episode 4: Reward = 19.00, Length = 19
Episode 5: Reward = 13.00, Length = 13
Episode 6: Reward = 25.00, Length = 25
Episode 7: Reward = 17.00, Length = 17
Episode 8: Reward = 21.00, Length = 21
Episode 9: Reward = 16.00, Length = 16
Episode 10: Reward = 24.00, Length = 24
--------------------------------------------------
平均奖励: 20.00 ± 4.83
平均长度: 20.00 ± 4.83
```

**代码解析**：

1. **环境创建**：`gym.make(env_name)`
2. **重置**：`env.reset()` 返回初始状态
3. **采样动作**：`env.action_space.sample()` 随机动作
4. **执行动作**：`env.step(action)` 返回 (observation, reward, terminated, truncated, info)
5. **终止条件**：`terminated`（任务完成）或 `truncated`（超时）

> [!NOTE]
> **CartPole-v1 环境**：
> - **目标**：平衡杆不倒下
> - **状态**：[位置, 速度, 角度, 角速度]
> - **动作**：0（左）或 1（右）
> - **奖励**：每步 +1（最多 500 步）
> - **终止**：杆倾斜超过 ±12°，或小车超出边界

### 进阶：可视化环境

```python
import gymnasium as gym
import matplotlib.pyplot as plt
from IPython.display import clear_output

def visualize_random_agent(env_name="CartPole-v1"):
    """可视化随机智能体"""
    env = gym.make(env_name, render_mode="rgb_array")
    observation, info = env.reset()
    
    plt.figure(figsize=(8, 6))
    
    for step in range(200):
        # 渲染
        img = env.render()
        
        # 显示
        clear_output(wait=True)
        plt.imshow(img)
        plt.title(f"Step {step}")
        plt.axis('off')
        plt.pause(0.01)
        
        # 执行动作
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    env.close()
    plt.close()

# 运行（需要 Jupyter Notebook）
# visualize_random_agent()
```

---

## 本章小结

在本章中，我们学习了：

✅ **强化学习的本质**：通过试错和延迟奖励学习最优策略  
✅ **核心要素**：Agent、Environment、State、Action、Reward  
✅ **历史发展**：从 Bellman 到 DQN，再到 RLHF  
✅ **核心概念**：价值函数 vs 策略、On-policy vs Off-policy、Model-free vs Model-based  
✅ **环境配置**：Gymnasium、MuJoCo、Atari  
✅ **第一个程序**：Random Agent

> [!TIP]
> **下一步**：
> 现在你已经了解了 RL 的全貌，接下来我们将深入学习 RL 的数学基础——**马尔可夫决策过程（MDP）**。这是理解所有 RL 算法的关键。
> 
> 进入 [Chapter 1. 马尔可夫决策过程](01-mdp.md)

---

## 扩展阅读

- **Sutton & Barto**：Chapter 1 (Introduction)
- **Spinning Up**：[Introduction to RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- **DeepMind Blog**：[AlphaGo](https://deepmind.google/technologies/alphago/)
- **OpenAI Blog**：[ChatGPT](https://openai.com/blog/chatgpt)
- **论文**：
  - Mnih et al. (2015): Human-level control through deep RL (DQN)
  - Schulman et al. (2017): Proximal Policy Optimization (PPO)
  - Ouyang et al. (2022): Training language models to follow instructions (InstructGPT)

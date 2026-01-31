---
title: "第29章：合作多智能体任务"
description: "合作式MARL，Dec-POMDP，协调机制与SMAC基准测试"
date: "2026-01-30"
---

# 第29章：合作多智能体任务

## 29.1 合作任务设计

### 29.1.1 共享奖励结构

在**合作多智能体**设置中，所有智能体共享一个共同目标并获得相同的团队奖励。

**形式化定义**：

对于 $n$ 个智能体，联合状态 $s \in \mathcal{S}$，联合动作 $\mathbf{a} = (a^1, ..., a^n)$：

$$
r_{\text{shared}}(s, \mathbf{a}) = r^1(s, \mathbf{a}) = r^2(s, \mathbf{a}) = \cdots = r^n(s, \mathbf{a})
$$

**优势**：
- 智能体目标自然对齐
- 鼓励团队协作
- 简化信用分配（所有智能体获得相同奖励）

**挑战**：
- **信用分配**：哪个智能体对成功/失败有贡献？
- **搭便车**：部分智能体可能依赖其他智能体的努力
- **协调失败**：智能体可能相互干扰

### 29.1.2 部分可观测性

现实世界的合作任务往往涉及**部分可观测性**：每个智能体只能看到局部信息。

**示例（机器人仓库）**：
- 机器人1看到它附近的货架
- 机器人2看到不同的货架
- 没有机器人拥有全局仓库视图

**影响**：
1. 智能体必须从局部观测**推断**全局状态
2. **通信**变得有价值
3. 需要**去中心化执行**

### 29.1.3 Dec-POMDP 框架

**去中心化部分可观测马尔可夫决策过程（Dec-POMDP）** 形式化了合作多智能体问题。

**组成部分**：
- $\mathcal{S}$：全局状态空间
- $\mathcal{A}^i$：智能体 $i$ 的动作空间
- $\mathcal{O}^i$：智能体 $i$ 的观测空间
- $T(s' | s, \mathbf{a})$：状态转移函数
- $O(o^i | s, a^i)$：智能体 $i$ 的观测函数
- $R(s, \mathbf{a})$：共享团队奖励

**关键性质**：智能体基于**局部观测历史**而非全局状态行动：

$$
\pi^i(a^i | \tau^i), \quad \tau^i = (o_0^i, a_0^i, o_1^i, a_1^i, \dots, o_t^i)
$$

**复杂度**：Dec-POMDP 是 NEXP-完全的（甚至比 POMDP 更难！）

<div data-component="CooperativeTaskVisualization"></div>

---

## 29.2 协调机制

### 29.2.1 角色分配

有效的团队通常表现出具有专业化角色的**分工**。

**静态角色分配**：
- 预定义角色（例如"侦察兵"、"防御者"、"攻击者"）
- 智能体 $i$ 遵循策略 $\pi_{\text{role}_i}$

**动态角色分配**：
- 角色根据情况自适应
- 使用**角色选择器**网络：$\rho(s) \rightarrow \text{角色分配}$

**实现**：

```python
import torch
import torch.nn as nn
import gymnasium as gym
from typing import List, Dict

class RoleBasedPolicy(nn.Module):
    """
    基于角色的策略，用于合作多智能体任务
    每个角色有一个专门的子策略
    """
    def __init__(self, obs_dim: int, action_dim: int, num_roles: int = 3, hidden_dim: int = 128):
        """
        Args:
            obs_dim: 每个智能体的观测维度
            action_dim: 动作空间大小
            num_roles: 不同角色的数量（例如侦察兵、防御者、攻击者）
            hidden_dim: 隐藏层大小
        """
        super().__init__()
        self.num_roles = num_roles
        
        # 角色选择器：将观测映射到角色概率
        self.role_selector = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_roles),
            nn.Softmax(dim=-1)
        )
        
        # 特定角色的策略
        self.role_policies = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1)
            ) for _ in range(num_roles)
        ])
    
    def forward(self, obs):
        """
        Args:
            obs: 智能体观测 (batch_size, obs_dim)
        
        Returns:
            action_probs: 动作概率 (batch_size, action_dim)
            role_probs: 角色分配概率 (batch_size, num_roles)
        """
        # 选择角色
        role_probs = self.role_selector(obs)
        
        # 计算所有角色的加权动作概率
        action_probs = torch.zeros(obs.size(0), self.role_policies[0][-2].out_features)
        for i, role_policy in enumerate(self.role_policies):
            action_probs += role_probs[:, i:i+1] * role_policy(obs)
        
        return action_probs, role_probs


# 示例：使用基于角色的策略训练
def train_cooperative_agents(env, num_agents=4, num_episodes=1000):
    """
    使用基于角色的策略训练合作智能体
    
    Args:
        env: 多智能体环境（例如 PettingZoo）
        num_agents: 团队中的智能体数量
        num_episodes: 训练回合数
    """
    obs_dim = 20  # 示例观测维度
    action_dim = 5
    
    # 为每个智能体创建基于角色的策略
    agents = [RoleBasedPolicy(obs_dim, action_dim, num_roles=3) for _ in range(num_agents)]
    optimizers = [torch.optim.Adam(agent.parameters(), lr=1e-3) for agent in agents]
    
    for episode in range(num_episodes):
        observations = env.reset()
        done = False
        episode_reward = 0.0
        
        episode_data = []  # 存储所有智能体的 (obs, action, reward)
        
        while not done:
            actions = []
            role_assignments = []
            
            # 获取所有智能体的动作
            for agent_id, agent in enumerate(agents):
                obs_tensor = torch.FloatTensor(observations[agent_id]).unsqueeze(0)
                
                with torch.no_grad():
                    action_probs, role_probs = agent(obs_tensor)
                
                # 采样动作
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample().item()
                actions.append(action)
                
                # 跟踪角色分配
                role = torch.argmax(role_probs).item()
                role_assignments.append(role)
            
            # 环境步进
            next_observations, rewards, dones, truncated, info = env.step(actions)
            
            # 存储所有智能体的转移（共享奖励）
            shared_reward = sum(rewards) / len(rewards)  # 平均奖励
            episode_data.append({
                'observations': observations.copy(),
                'actions': actions.copy(),
                'reward': shared_reward,
                'role_assignments': role_assignments.copy()
            })
            
            observations = next_observations
            episode_reward += shared_reward
            done = all(dones.values())
        
        # 为所有智能体进行策略梯度更新
        for agent_id, agent in enumerate(agents):
            optimizer = optimizers[agent_id]
            
            loss = 0.0
            for t, data in enumerate(episode_data):
                obs = torch.FloatTensor(data['observations'][agent_id]).unsqueeze(0)
                action = data['actions'][agent_id]
                reward = data['reward']
                
                # 计算回报（简化：使用 reward-to-go）
                returns = sum([d['reward'] for d in episode_data[t:]])
                
                # 策略梯度
                action_probs, role_probs = agent(obs)
                log_prob = torch.log(action_probs[0, action] + 1e-8)
                loss += -log_prob * returns
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if episode % 100 == 0:
            print(f"回合 {episode}, 奖励: {episode_reward:.2f}, 角色: {role_assignments}")
```

<div data-component="RoleAssignment"></div>

### 29.2.2 任务分解

复杂的合作任务可以分解为分配给不同智能体的子任务。

**分层分解**：
1. **高层规划器**：决定子任务
2. **底层执行器**：执行分配的子任务

**示例（搜救任务）**：
- 任务：寻找并疏散平民
- 子任务：
  - 侦察：探索未知区域
  - 救援：将平民运送到安全地带
  - 支援：提供补给

**优势**：
- 降低协调复杂度
- 实现专业化
- 并行化执行

### 29.2.3 动态协作

智能体必须根据当前情况调整协作模式。

**用于协作的注意力机制**：

```python
class CollaborativeAttention(nn.Module):
    """
    智能体协作的注意力机制
    每个智能体关注其他智能体的状态以进行协调  
    """
    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.query_net = nn.Linear(obs_dim, hidden_dim)
        self.key_net = nn.Linear(obs_dim, hidden_dim)
        self.value_net = nn.Linear(obs_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5
    
    def forward(self, agent_obs, team_obs):
        """
        Args:
            agent_obs: 当前智能体观测 (batch, obs_dim)
            team_obs: 所有智能体的观测 (batch, num_agents, obs_dim)
        
        Returns:
            attended_features: 聚合的团队信息 (batch, hidden_dim)
        """
        # 计算 Q, K, V
        query = self.query_net(agent_obs).unsqueeze(1)  # (batch, 1, hidden)
        keys = self.key_net(team_obs)  # (batch, num_agents, hidden)
        values = self.value_net(team_obs)  # (batch, num_agents, hidden)
        
        # 注意力分数
        scores = torch.matmul(query, keys.transpose(-2, -1)) / self.scale  # (batch, 1, num_agents)
        attention_weights = torch.softmax(scores, dim=-1)
        
        # 值的加权和
        attended = torch.matmul(attention_weights, values).squeeze(1)  # (batch, hidden)
        
        return attended, attention_weights
```

---

## 29.3 基准环境

### 29.3.1 SMAC（星际争霸多智能体挑战）

**SMAC** 是合作 MARL 的领先基准测试，基于星际争霸 II 微操作。

**环境描述**：
- **任务**：控制一队单位击败敌军
- **智能体**：3-27 个单位（机枪兵、追猎者、刺蛇等）
- **动作**：移动、攻击、停止、保持位置
- **观测**：局部视野（单位生命值、位置、敌人可见性）
- **奖励**：造成的伤害 + 敌人死亡数 - 友军死亡数

**关键场景**：
- `3m`：3 机枪兵 vs 3 机枪兵（简单）
- `8m`：8 机枪兵 vs 8 机枪兵
- `2s3z`：2 追猎者 + 3 狂热者 vs 相同配置
- `corridor`：狭窄地形，需要协调
- `MMM`：混合单位类型（机枪兵、掠夺者、医疗运输机）

**安装与使用**：

```bash
# 安装 SMAC
pip install smac

# 需要星际争霸 II（免费版）
# 下载 SC2: https://github.com/Blizzard/s2client-proto#downloads
```

```python
from smac.env import StarCraft2Env
import numpy as np

def run_smac_episode():
    """
    在 SMAC 环境中运行单个回合
    演示基本 API 使用
    """
    # 创建环境
    # map_name 选项："3m", "8m", "25m", "2s3z", "3s5z", "MMM" 等
    env = StarCraft2Env(map_name="3m", difficulty="7")  # difficulty: 1-9 (9=最难)
    
    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]  # 可控智能体数量
    n_actions = env_info["n_actions"]  # 动作空间大小
    obs_shape = env_info["obs_shape"]  # 每个智能体的观测维度
    state_shape = env_info["state_shape"]  # 全局状态维度
    
    print(f"SMAC 环境: {n_agents} 个智能体, {n_actions} 个动作, obs_dim={obs_shape}")
    
    env.reset()
    terminated = False
    episode_reward = 0
    
    while not terminated:
        # 获取观测
        obs = env.get_obs()  # 每个智能体的观测列表，形状 (n_agents, obs_shape)
        state = env.get_state()  # 全局状态，形状 (state_shape,)
        
        # 获取可用动作（某些动作可能无效）
        avail_actions = env.get_avail_actions()  # (n_agents, n_actions) 二进制掩码
        
        # 选择动作（演示用的随机策略）
        actions = []
        for agent_id in range(n_agents):
            available = np.where(avail_actions[agent_id])[0]
            action = np.random.choice(available)
            actions.append(action)
        
        # 环境步进
        reward, terminated, info = env.step(actions)
        episode_reward += reward
    
    print(f"回合结束。总奖励: {episode_reward:.2f}")
    print(f"战斗胜利: {info.get('battle_won', False)}")
    
    env.close()
    return episode_reward


# 使用 QMIX 训练（简化版）
class QMIXAgent:
    """简化的 SMAC 用 QMIX"""
    def __init__(self, n_agents, obs_dim, n_actions, state_dim):
        self.n_agents = n_agents
        
        # 个体 Q 网络
        self.q_networks = [
            nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, n_actions)
            ) for _ in range(n_agents)
        ]
        
        # 混合网络（简化版）
        self.mixer = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_agents),
            nn.Softmax(dim=-1)  # 单调权重
        )
    
    def forward(self, observations, state):
        """
        Args:
            observations: 智能体观测列表
            state: 全局状态
        
        Returns:
            q_tot: 联合动作的混合 Q 值
        """
        # 获取个体 Q 值
        q_values = []
        for i, obs in enumerate(observations):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            q = self.q_networks[i](obs_tensor)
            q_values.append(q)
        
        q_values = torch.stack(q_values, dim=1)  # (batch, n_agents, n_actions)
        
        # 混合 Q 值
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        weights = self.mixer(state_tensor)  # (batch, n_agents)
        
        # 加权和（简化，应使用超网络）
        q_tot = torch.sum(q_values * weights.unsqueeze(-1), dim=1)
        
        return q_tot
```

<div data-component="SMACEnvironment"></div>

### 29.3.2 Google Research Football

**多智能体足球**环境，具有连续控制和复杂团队动态。

**特性**：
- **智能体**：每队 11 名球员（可以控制全部或部分）
- **动作**：移动、冲刺、射门、传球、铲球、切换球员
- **观测**：球位置、球员位置、比分
- **奖励**：进球得分

**安装**：

```bash
pip install gfootball
```

**关键场景**：
- `academy_3_vs_1_with_keeper`：3 名进攻球员 vs 1 名防守球员 + 守门员
- `11_vs_11_easy_stochastic`：完整 11v11 比赛

### 29.3.3 PettingZoo

**PettingZoo** 是多智能体环境库，具有统一的 API（类似于单智能体的 Gymnasium）。

**安装**：

```bash
pip install pettingzoo[all]
```

**示例合作环境**：多粒子环境（MPE）

```python
from pettingzoo.mpe import simple_spread_v3

def run_pettingzoo_cooperative():
    """
    在 PettingZoo MPE 环境中运行合作
    任务：智能体必须分散覆盖所有地标
    """
    # 创建环境
    # simple_spread: N 个智能体必须分散覆盖 N 个地标
    env = simple_spread_v3.parallel_env(N=3, continuous_actions=False)
    
    observations, infos = env.reset()
    
    # PettingZoo 使用智能体名称作为键
    agents = env.agents  # ['agent_0', 'agent_1', 'agent_2']
    
    print(f"智能体: {agents}")
    print(f"观测空间: {env.observation_spaces}")
    print(f"动作空间: {env.action_spaces}")
    
    terminated = {agent: False for agent in agents}
    truncated = {agent: False for agent in agents}
    
    while not all(terminated.values()):
        actions = {agent: env.action_space(agent).sample() for agent in agents}
        
        observations, rewards, terminated, truncated, infos = env.step(actions)
        
        print(f"奖励: {rewards}")
    
    env.close()


# PettingZoo 合作环境列表
cooperative_envs = [
    "simple_spread",  # 智能体覆盖地标
    "simple_reference",  # 通信游戏
    "simple_speaker_listener",  # 一个智能体说话，另一个行动
    "pistonball",  # 合作推球
    "waterworld",  # 收集食物，避免毒药
]
```

---

## 29.4 现实世界应用

### 29.4.1 多机器人系统

**仓库自动化**：
- **智能体**：自主移动机器人（AMR）
- **任务**：从货架拣选物品，送到打包站
- **协调**：避免碰撞，队列管理
- **公司**：亚马逊机器人、Ocado

**关键算法**：
- **MAPPO**：用于去中心化执行的多智能体 PPO
- **CommNet**：机器人之间的通信
- **路径规划**：多智能体路径查找（MAPF）

### 29.4.2 交通控制

**自适应交通信号控制**：
- **智能体**：交叉路口的交通灯
- **观测**：车辆计数、队列长度
- **动作**：绿灯/红灯信号时间
- **奖励**：最小化平均等待时间

**挑战**：
- 大规模协调（数百个交叉路口）
- 延迟奖励
- 非平稳性（交通模式变化）

**结果**：基于 RL 的交通控制相比固定时间信号实现了 10-30% 的等待时间减少。

### 29.4.3 资源分配

**云计算**：
- **智能体**：资源调度器
- **任务**：为作业分配 CPU/内存
- **目标**：最大化吞吐量，最小化延迟

**能源网格**：
- **智能体**：发电厂、变电站
- **任务**：平衡供需
- **协调**：分布式优化，避免停电

---

## 总结

本章涵盖了合作多智能体强化学习：

1. **任务设计**：共享奖励、Dec-POMDP 形式化、部分可观测性
2. **协调机制**：角色分配、任务分解、注意力机制
3. **基准测试**：SMAC（星际争霸 II）、Google Research Football、PettingZoo
4. **应用**：多机器人系统、交通控制、资源分配

**关键要点**：
- 合作需要显式的协调机制
- Dec-POMDP 捕捉部分可观测性挑战
- SMAC 是合作 MARL 的标准基准测试
- 现实世界应用受益于专门的 RL 算法（QMIX、MAPPO）

**下一章**：第 30 章探讨**竞争**多智能体设置与博弈论和对抗训练。

---

## 参考文献

- Samvelyan, M., et al. (2019). "The StarCraft Multi-Agent Challenge." *AAMAS*.
- Terry, J. K., et al. (2021). "PettingZoo: Gym for Multi-Agent RL." *NeurIPS Datasets Track*.
- Rashid, T., et al. (2018). "QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent RL." *ICML*.
- Yu, C., et al. (2022). "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games." *NeurIPS*.
- Oliehoek, F. A., & Amato, C. (2016). *A Concise Introduction to Decentralized POMDPs*. Springer.

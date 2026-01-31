---
title: "Chapter 18. Hierarchical Reinforcement Learning"
description: "时间抽象与技能复用"
updated: "2026-01-29"
---

> **Learning Objectives**
> * 理解层次化 RL 的动机与优势
> * 掌握 Options 框架与 Semi-MDP
> * 学习 Feudal RL 的 Manager-Worker 架构
> * 了解 HAM 与 MAXQ 分解
> * 掌握技能发现方法（DIAYN）

---

## 18.1 层次化的动机

层次化强化学习（Hierarchical RL）通过时间抽象和技能复用加速学习。

### 18.1.1 长期规划

**问题**：平面 RL（flat RL）在长时域任务中效率低。

**示例**：机器人导航
- **Flat RL**：每步选择电机命令（low-level）
- **Hierarchical RL**：先选择目标房间（high-level），再执行导航技能（low-level）

**优势**：
- 减少决策复杂度
- 更有效的信用分配

### 18.1.2 技能复用

**技能**（Skill）：可复用的子策略。

**示例**：
- "抓取"技能可用于多个任务
- "导航"技能可用于不同目标

**优势**：
- 迁移学习
- 更快的新任务学习

### 18.1.3 时间抽象

**Options**：持续多个时间步的**宏动作**（macro-action）。

**对比**：
- **原始动作**：每步选择低阶动作
- **Option**：选择一个 option，执行到终止条件

**优势**：
- 简化决策
- 加速探索（大步跳跃）

---

## 18.2 Options 框架

Options 是对 MDP 的层次化扩展。

<div data-component="OptionsFramework"></div>

### 18.2.1 Option 定义（π, β, I）

**Option** $o = \langle I_o, \pi_o, \beta_o \rangle$：

1. **Initiation Set** $I_o \subseteq \mathcal{S}$：
   - 可以启动 option 的状态集合

2. **Policy** $\pi_o: \mathcal{S} \times \mathcal{A} \to [0, 1]$：
   - option 的内部策略

3. **Termination Condition** $\beta_o: \mathcal{S} \to [0, 1]$：
   - 在状态 $s$ 终止的概率

**示例**：导航到门口
- $I_o$：任何室内状态
- $\pi_o$：朝门移动
- $\beta_o(s) = 1$ 如果 $s$ 在门口，否则 $0$

### 18.2.2 Semi-MDP

**Options 下的 MDP**：

- **状态**：$s \in \mathcal{S}$
- **Option**：$o \in \mathcal{O}$（option 集合）
- **转移**：执行 option 直到终止
- **奖励**：累积奖励 $R(s, o) = \sum_{t=0}^{k-1} \gamma^t r_t$

**值函数**：

$$
Q(s, o) = \mathbb{E}\left[\sum_{t=0}^{k-1} \gamma^t r_t + \gamma^k \max_{o'} Q(s_k, o') \mid s_0 = s, o\right]
$$

### 18.2.3 Option-Critic 算法

**学习 options 的策略和终止条件。**

**High-level 策略**：

$$
\pi_{\Omega}(o|s) = \text{softmax}(Q_{\Omega}(s, o))
$$

**Intra-option 值函数**：

$$
Q_{\Omega}(s, o) = \sum_a \pi_{o}(a|s) Q_U(s, o, a)
$$

**Bellman 方程**：

$$
Q_U(s, o, a) = r(s, a) + \gamma \sum_{s'} P(s'|s, a) U(o, s')
$$

其中：

$$
U(o, s') = (1 - \beta_o(s')) Q_{\Omega}(s', o) + \beta_o(s') \max_{o'} Q_{\Omega}(s', o')
$$

**梯度**：

Option 的策略梯度：

$$
\nabla_{\theta_o} J = \mathbb{E}\left[\nabla_{\theta_o} \log \pi_o(a|s) Q_U(s, o, a)\right]
$$

Termination 梯度：

$$
\nabla_{\theta_{\beta}} J = \mathbb{E}\left[\nabla_{\theta_{\beta}} \beta_o(s) A_{\Omega}(s, o)\right]
$$

其中 $A_{\Omega}(s, o) = Q_{\Omega}(s, o) - \max_{o'} Q_{\Omega}(s, o')$。

**实现**：

```python
class OptionCritic(nn.Module):
    """
    Option-Critic Architecture
    
    Args:
        state_dim: 状态维度
        num_options: Option 数量
        action_dim: 动作数量
    """
    def __init__(self, state_dim, num_options, action_dim):
        super().__init__()
        
        self.num_options = num_options
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        
        # Option-specific policies
        self.option_policies = nn.ModuleList([
            nn.Linear(128, action_dim) for _ in range(num_options)
        ])
        
        # Termination functions
        self.terminations = nn.Linear(128, num_options)
        
        # Q-values over options
        self.q_values = nn.Linear(128, num_options)
    
    def forward(self, state, option=None):
        """
        Args:
            state: 状态
            option: 当前 option（如果在执行中）
        
        Returns:
            action_probs: 动作概率
            q_options: option 的 Q 值
            termination_prob: 终止概率
        """
        features = self.features(state)
        
        # Q-values over options
        q_options = self.q_values(features)
        
        # Termination probabilities
        termination_probs = torch.sigmoid(self.terminations(features))
        
        # Action probabilities (for current option)
        if option is not None:
            logits = self.option_policies[option](features)
            action_probs = torch.softmax(logits, dim=-1)
        else:
            action_probs = None
        
        return action_probs, q_options, termination_probs
    
    def select_option(self, state):
        """选择 option（greedy 或 softmax）"""
        _, q_options, _ = self.forward(state)
        return q_options.argmax(dim=-1).item()
    
    def select_action(self, state, option):
        """在 option 内选择动作"""
        action_probs, _, _ = self.forward(state, option)
        dist = torch.distributions.Categorical(action_probs)
        return dist.sample().item()
    
    def should_terminate(self, state, option):
        """判断 option 是否应该终止"""
        _, _, termination_probs = self.forward(state)
        return torch.rand(1).item() < termination_probs[option].item()
```

---

## 18.3 Feudal RL

Feudal RL 使用 Manager-Worker 层次结构。

<div data-component="FeudalArchitecture"></div>

### 18.3.1 Manager-Worker 架构

**两层结构**：

1. **Manager**（高层）：
   - 设定目标（goals）
   - 较长时间尺度

2. **Worker**（低层）：
   - 实现目标
   - 较短时间尺度

**信息流**：
- Manager → Worker：目标 $g_t$
- Worker → Manager：状态信息

### 18.3.2 目标设定

**Manager 学习设定目标** $g_t$：

$$
g_t = f_{\text{manager}}(s_t)
$$

**Worker 学习实现目标**：

$$
a_t = \pi_{\text{worker}}(s_t, g_t)
$$

**内在奖励**（Worker）：

$$
r_{\text{intrinsic}} = \cos(s_{t+1} - s_t, g_t)
$$

（Worker 因靠近目标方向而获得奖励）

### 18.3.3 FuN (FeUdal Networks)

**FuN 架构**（Vezhnevets et al., 2017）：

1. **Manager**：
   - 输出目标向量 $g_t \in \mathbb{R}^d$
   - 在较长时间尺度更新（例如每 c 步）

2. **Worker**：
   - 以 $(s_t, g_t)$ 为输入
   - 选择动作 $a_t$

**Transition Policy Gradient**：

Manager 通过 Worker 的累积奖励学习：

$$
\nabla J_{\text{manager}} = \mathbb{E}\left[\sum_{t} \nabla \log \pi_{\text{manager}}(g_t | s_t) R_t\right]
$$

Worker 通过内在奖励学习：

$$
\nabla J_{\text{worker}} = \mathbb{E}\left[\sum_{t} \nabla \log \pi_{\text{worker}}(a_t | s_t, g_t) r_{\text{intrinsic},t}\right]
$$

---

## 18.4 HAM (Hierarchical Abstract Machines)

基于状态机的层次化方法。

### 18.4.1 状态机层次

**HAM**：定义任务的层次化状态机。

**组成**：
- **Choice states**：需要学习的决策点
- **Action states**：执行原始动作
- **Call states**：调用子机器

**示例**：导航任务
```
Root Machine:
  ├─ Navigate to Room A
  │   ├─ Move North
  │   └─ Move East
  └─ Navigate to Room B
```

### 18.4.2 MAXQ 分解

**MAXQ**：将值函数分解为层次结构。

**分解**：

$$
Q(s, a) = V(a, s) + C(a, s)
$$

- $V(a, s)$：执行子任务 $a$ 的值
- $C(a, s)$：子任务完成后的值

<div data-component="MAXQDecomposition"></div>

**递归分解**：

$$
V(M_i, s) = \max_{a \in A_i} Q(M_i, s, a)
$$

$$
Q(M_i, s, a) = \begin{cases}
R(s, a) + \gamma \sum_{s'} P(s'|s, a) V(M_i, s') & \text{if } a \text{ is primitive} \\
V(M_a, s) + C(M_i, s, M_a) & \text{if } a \text{ is composite}
\end{cases}
$$

**优势**：
- 模块化：子任务独立学习
- 可迁移：子任务可复用

---

## 18.5 技能发现

无监督学习有用的技能。

### 18.5.1 DIAYN (Diversity is All You Need)

**核心思想**：学习多样化且可区分的技能。

<div data-component="SkillDiscovery"></div>

**目标**：

$$
\max I(Z; S) - I(Z; A|S)
$$

其中：
- $Z$：技能（离散随机变量，例如 $Z \in \{1, \ldots, K\}$）
- $S$：状态
- $A$：动作

**解释**：
- $I(Z; S)$：技能应对状态有影响（可区分性）
- $I(Z; A|S)$：技能应对动作影响小（鼓励探索）

### 18.5.2 互信息最大化

**实践目标**：

$$
\mathcal{F}(\theta) = \mathbb{E}_{z \sim p(z), \tau \sim \pi_z}\left[\sum_t \log q_{\phi}(z|s_t) + H[\pi_z]\right]
$$

其中：
- $\pi_z(a|s)$：技能 $z$ 的策略
- $q_{\phi}(z|s)$：判别器（判断状态来自哪个技能）

**伪奖励**：

$$
r(s, a, z) = \log q_{\phi}(z|s) - \log p(z)
$$

**训练流程**：

1. **采样技能**：$z \sim p(z)$（例如均匀分布）
2. **执行技能**：用 $\pi_z$ 收集轨迹
3. **训练判别器**：$q_{\phi}(z|s)$ 预测技能
4. **训练策略**：$\pi_z$ 最大化伪奖励

**实现**：

```python
class DIAYNAgent:
    """
    DIAYN: Diversity is All You Need
    
    Args:
        state_dim: 状态维度
        action_dim: 动作数量
        num_skills: 技能数量
    """
    def __init__(self, state_dim, action_dim, num_skills):
        self.num_skills = num_skills
        
        # Skill-conditioned policy
        self.policy = SkillConditionedPolicy(state_dim, action_dim, num_skills)
        
        # Discriminator q(z|s)
        self.discriminator = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_skills)
        )
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=3e-4)
        
        # 均匀先验 p(z)
        self.skill_prior = 1.0 / num_skills
    
    def train_step(self, states, actions, next_states, skills):
        """
        DIAYN 训练步骤
        
        Args:
            states, actions, next_states: 轨迹
            skills: 技能标签 z
        """
        # 训练判别器
        logits = self.discriminator(next_states)
        disc_loss = nn.CrossEntropyLoss()(logits, skills)
        
        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()
        
        # 计算伪奖励
        with torch.no_grad():
            log_q_z = nn.functional.log_softmax(logits, dim=-1)
            log_q_z = log_q_z.gather(1, skills.unsqueeze(1)).squeeze()
            pseudo_reward = log_q_z - np.log(self.skill_prior)
        
        # 训练策略（使用 SAC 或其他 RL 算法 + 伪奖励）
        # ... (SAC update with pseudo_reward)
```

### 18.5.3 无监督技能学习

**其他方法**：

1. **VIC (Variational Intrinsic Control)**：
   - 使用变分推断
   - 最大化互信息下界

2. **APT (Active Pretraining)**：
   - 主动选择有信息的技能

3. **APS (Episodic Curiosity)**：
   - 基于 episode 记忆的技能发现

**应用**：
- 预训练：先学习技能，再用于下游任务
- 迁移学习：技能可复用于新任务
- 探索：技能作为探索策略

---

## 本章小结

在本章中，我们学习了：

✅ **层次化动机**：长期规划、技能复用、时间抽象  
✅ **Options 框架**：Semi-MDP、Option-Critic 算法  
✅ **Feudal RL**：Manager-Worker 架构、目标设定  
✅ **HAM 与 MAXQ**：状态机层次、值函数分解  
✅ **技能发现**：DIAYN、互信息最大化、无监督学习  

> [!TIP]
> **核心要点**：
> - 层次化 RL 通过时间抽象简化长时域任务
> - Options 是可复用的宏动作，包含策略和终止条件
> - Feudal RL 使用 Manager-Worker 分层决策
> - MAXQ 分解允许模块化学习和技能复用
> - DIAYN 通过最大化互信息无监督学习多样技能
> - 技能发现为迁移学习和预训练提供基础

> [!NOTE]
> **后续章节将涵盖**：
> - Chapter 19: 逆强化学习（IRL, GAIL）
> - Multi-agent RL
> - Meta-RL
> - RLHF 与大模型对齐

---

## 扩展阅读

- **经典论文**：
  - Sutton et al. (1999): Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in RL
  - Bacon et al. (2017): The Option-Critic Architecture
  - Vezhnevets et al. (2017): FeUdal Networks for Hierarchical RL
  - Dietterich (2000): Hierarchical RL with the MAXQ Value Function Decomposition
  - Eysenbach et al. (2019): Diversity is All You Need: Learning Skills without a Reward Function (DIAYN)
- **理论基础**：
  - Parr & Russell (1998): Reinforcement Learning with Hierarchies of Machines
  - Precup (2000): Temporal Abstraction in RL
- **实现资源**：
  - Option-Critic: PyTorch implementation
  - DIAYN: Official code
  - Hierarchical RL库: HRL-Lib

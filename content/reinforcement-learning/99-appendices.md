---
title: "附录：深度强化学习参考手册"
description: "数学推导、环境速查表、通用训练框架、调试反模式与论文导读"
date: "2026-01-30"
---

# 附录 (Appendices)

本附录旨在成为强化学习研究者与工程师的案头参考手册。我们不仅提供基础公式，还汇集了工程实践中的“暗知识”与标准模板。

---

## Appendix A: 数学基础与推导 (Math & Derivations)

### A.1 概率与统计

**1. 常用分布与 KL 散度**
*   **高斯分布 (Gaussian)**: $\mathcal{N}(\mu, \sigma^2)$，RL 中常用于连续动作策略 $\pi_\theta(a|s)$。
    *   Log Probability: $\log \pi(a|s) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(a-\mu)^2}{2\sigma^2}$ (PPO Loss 计算常用)
*   **KL 散度 (Kullback-Leibler Divergence)**: 衡量两个分布的差异。
    *   $\text{KL}(P || Q) = \mathbb{E}_{x \sim P} [\log \frac{P(x)}{Q(x)}]$
    *   两个高斯分布间的 KL: $\text{KL}(\mathcal{N}_0 || \mathcal{N}_1) = \log\frac{\sigma_1}{\sigma_0} + \frac{\sigma_0^2 + (\mu_0 - \mu_1)^2}{2\sigma_1^2} - \frac{1}{2}$

**2. 梯度估计 (Gradient Estimation)**
*   **Score Function Estimator (REINFORCE)**:
    $$ \nabla_\theta \mathbb{E}_{x \sim p_\theta}[f(x)] = \mathbb{E}_{x \sim p_\theta}[f(x) \nabla_\theta \log p_\theta(x)] $$
    *   *直观理解*：如果 $f(x)$ (奖励) 高，就增加该样本 $x$ 的概率密度。
    *   *推导*: 利用 $\nabla p_\theta = p_\theta \nabla \log p_\theta$ (Log-derivative trick)。

**3. 重要性采样 (Importance Sampling)**
用于 Off-policy 策略评估，修正分布偏移。
$$ \mathbb{E}_{x \sim \text{target}}[f(x)] = \mathbb{E}_{x \sim \text{behavior}} \left[ \rho(x) f(x) \right], \quad \rho(x) = \frac{\pi_{\text{target}}(x)}{\pi_{\text{behavior}}(x)} $$
*   **注意**: 如果 $\rho(x)$ 方差过大，估计会极其不稳定（PPO 采用 Clip 机制来缓解此问题）。

### A.2 贝尔曼方程与收敛性

**1. Bellman 算子 (Operator)**
定义算子 $\mathcal{T}^\pi$:
$$ (\mathcal{T}^\pi V)(s) = R(s, \pi(s)) + \gamma \sum_{s'} P(s'|s, \pi(s)) V(s') $$
*   **收缩映射 (Contraction Mapping)**: $\mathcal{T}^\pi$ 是 $\gamma$-contraction，即 $ ||\mathcal{T}U - \mathcal{T}V||_\infty \le \gamma ||U - V||_\infty $。
*   **不动点定理**: 根据 Banach Fixed Point Theorem，价值迭代必收敛于唯一不动点 $V^\pi$。

---

## Appendix B: 环境与 Benchmark 速查表

### B.1 经典控制 (Classic Control)

| 环境 ID | 观测空间 (Obs) | 动作空间 (Action) | 奖励范围 | 解决标准 (Solved) | 特点 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `CartPole-v1` | Box(4): [位置, 速, 角, 角速] | Discrete(2): [左, 右] | +1/step (max 500) | 475 | 入门测试，调试必用 |
| `Pendulum-v1` | Box(3): [cos, sin, dot] | Box(1):力矩 [-2, 2] | -(θ^2 + 0.1v^2 + 0.001u^2) | Approx -150 | 连续控制入门 |
| `MountainCar-v0` | Box(2): [位置, 速度] | Discrete(3): [推左, 不动, 推右] | -1/step | -110 | 稀疏奖励，需要探索 |

### B.2 MuJoCo (v4)

所有 MuJoCo 环境均为连续动作空间 `Box(k)`，范围通常为 `[-1, 1]`。

| 环境 ID | Obs Dim | Action Dim | 描述 | SOTA 分数 (PPO) |
| :--- | :--- | :--- | :--- | :--- |
| `HalfCheetah-v4` | 17 | 6 | 二维猎豹跑 | ~5000-8000 |
| `Hopper-v4` | 11 | 3 | 单脚跳 | ~3000 |
| `Walker2d-v4` | 17 | 6 | 双足行走 | ~4000-5000 |
| `Ant-v4` | 27 | 8 | 四足蚂蚁 | ~5000-6000 |
| `Humanoid-v4` | 376 | 17 | 人形机器人 | ~6000+ (极难) |

### B.3 自定义环境模板

创建一个兼容 Gymnasium 的环境标准模板：

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CustomEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.render_mode = render_mode
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 1. 设置随机种子
        self.state = self.np_random.uniform(low=0, high=1, size=(10,)).astype(np.float32)
        info = {}
        return self.state, info

    def step(self, action):
        # 2. 状态转移逻辑
        velocity = (action - 1.5) * 0.1
        self.state = np.clip(self.state + velocity, 0, 1)
        
        # 3. 计算奖励
        reward = -np.sum((self.state - 0.5)**2)  # 目标是0.5
        
        # 4. 终止条件
        terminated = bool(np.abs(self.state[0] - 0.5) < 0.01)
        truncated = False # 超时截断
        
        return self.state, reward, terminated, truncated, {}
```

---

## Appendix C: 通用工程框架与代码片段

### C.1 现代 RL 训练器模板 (The "Trainer" Pattern)

不依赖第三方库的轻量级框架结构建议：

```python
class RLTrainer:
    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.buffer = ReplayBuffer(config.capacity)
        self.logger = SummaryWriter(config.log_dir)  # TensorBoard
        self.steps = 0

    def collect_rollouts(self, num_steps):
        """与环境交互并存入 Buffer"""
        obs, _ = self.env.reset()
        for _ in range(num_steps):
            with torch.no_grad():
                action = self.agent.select_action(obs)
            
            next_obs, reward, term, trunc, _ = self.env.step(action)
            done = term or trunc
            
            # 关键：处理 Terminated 时的 next_obs
            real_next_obs = next_obs.copy()
            if trunc: # 如果是超时，next_obs 是真实的；如果是 Terminated，可能需特殊处理
                pass 

            self.buffer.add(obs, action, reward, real_next_obs, term)
            obs = next_obs
            if done: obs, _ = self.env.reset()

    def train_step(self):
        """从 Buffer 采样并更新"""
        batch = self.buffer.sample(self.config.batch_size)
        loss_info = self.agent.update(batch)
        
        # Logging
        if self.steps % 100 == 0:
            for k, v in loss_info.items():
                self.logger.add_scalar(f"train/{k}", v, self.steps)
```

### C.2 GAE (Generalized Advantage Estimation) 实现

这是一个极易写错的关键函数：

```python
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards: [T]
    values: [T+1]  (包含最后一个状态的 V(s'))
    dones: [T]
    """
    gae = 0
    returns = []
    
    # 逆序计算
    for step in reversed(range(len(rewards))):
        # delta = r + gamma * V(s') * (1-d) - V(s)
        delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        
        # gae = delta + gamma * lambda * (1-d) * gae_next
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        
        # Return = V(s) + GAE = Q_target
        returns.insert(0, gae + values[step])
        
    return torch.tensor(returns)
```

---

## Appendix D: 调试与常见反模式 (Anti-Patterns)

### 💀 D.1 常见致命错误 (Deadly Bugs)

1.  **忘记 `optimizer.zero_grad()`**:
    *   *现象*: Loss 震荡或发散。
    *   *原因*: PyTorch 默认累加梯度，导致梯度值在几个 Batch 后变得巨大。
    
2.  **Done Flag 错误**:
    *   *错误写法*: `target = r + gamma * max_q * (1 - done)` 不区分 `TimeLimit`。
    *   *正确写法*: `truncated` (超时) 不应该被视为真正的结束（状态价值不为0），只有 `terminated` (失败/成功) 才是。
    *   *修正*: `mask = 1 - terminated` (忽略 truncated)。

3.  **Softmax 维度错误**:
    *   *错误*: `F.softmax(logits)` (默认 dim=None，旧版可能警告)。
    *   *正确*: `F.softmax(logits, dim=-1)`。

4.  **Observation 未归一化**:
    *   *现象*: 训练极其缓慢，Loss 很大。
    *   *修正*: 将图像除以 255.0，或对连续状态进行 Standard Scaling (减均值除方差)。

### 🔍 D.2 性能诊断流程

1.  **Check 0**: 随机策略的表现是多少？（作为 Baseline）。
2.  **Check 1**: 能否过拟合一个 Episode？（让 Batch Size = Episode Length，重复训练同一数据，看 Loss 是否趋近 0）。
3.  **Check 2**: 输出 Action 的分布。是否一直输出边界值（Saturation）？
    *   如果是 `Tanh` 激活，一直是 1 或 -1 -> 最后一层初始化权重过大。
4.  **Check 3**: 梯度范数 (Gradient Norm)。如果突然暴涨，检查是否有除以零的操作（如 Standardize Advantage 时 std=0）。

---

## Appendix E: 权威论文导读 (Annotated Bibliography)

### 基础算法
*   **DQN (2015)**: 首次将 CNN 与 Q-learning 结合。
    *   *Key*: Experience Replay (打破相关性), Target Network (稳定目标)。
*   **PPO (2017)**: 工业界默认首选。
    *   *Key*: Clipped Surrogate Objective (限制策略更新幅度)，简单且稳健。

### 高级主题
*   **GAE (Schulman 2015)**: *High-dimensional Continuous Control using GAE*.
    *   *Key*: Bias-Variance Tradeoff。$\lambda=1$ 是 Monte Carlo (高方差)，$\lambda=0$ 是 TD (高偏差)。
*   **SAC (2018)**: *Soft Actor-Critic*.
    *   *Key*: 最大熵 (Max Entropy) 目标 $ \mathbb{E}[R + \alpha H(\pi)] $，极大提升了探索能力和鲁棒性。

### 必读综述
*   **Spinning Up in Deep RL**: OpenAI 撰写的入门文档，包含极佳的算法伪代码和注意事项。
*   **A Survey on Offline Reinforcement Learning (Levine 2020)**: 离线 RL 的百科全书。

---

## Appendix F: 术语表 (Glossary)

*   **Episode/Rollout**: 从初始状态到终止状态的一次完整交互序列。
*   **Horizon (H)**: 一个 Episode 的最大步数。
*   **Return (G)**: 累积折现奖励 $\sum \gamma^t r_t$。
*   **On-Policy**: 训练数据的分布必须与当前策略一致（如 PPO, REINFORCE）。
*   **Off-Policy**: 可以利用历史数据（别人产生的经验）进行训练（如 DQN, SAC）。
*   **Model-Free**: 不学习 $P(s'|s,a)$，直接学 Value 或 Policy。
*   **Model-Based**: 学习环境模型，并在“脑海中”推演（Planning）。
*   **Sim-to-Real**: 仿真到现实的迁移，主要挑战是 Reality Gap。

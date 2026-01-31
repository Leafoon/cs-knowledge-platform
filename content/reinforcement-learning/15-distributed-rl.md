---
title: "Chapter 15. Distributed Reinforcement Learning"
description: "大规模并行训练加速RL"
updated: "2026-01-29"
---

> **Learning Objectives**
> * 理解分布式 RL 的必要性与优势
> * 掌握 Ape-X 的分布式架构
> * 学习 IMPALA 与 V-trace 修正
> * 了解 R2D2 的循环经验回放
> * 掌握分布式 RL 的实现架构

---

## 15.1 并行化的必要性

现代 RL 任务需要海量数据，单机训练难以满足需求。

### 15.1.1 样本效率提升

**问题**：深度 RL 通常需要数百万甚至数十亿步交互。

**示例**：
- Atari 游戏：DQN 需要 ~2亿 帧 (约 200M steps)
- Dota 2 (OpenAI Five)：900 年游戏经验/天
- StarCraft II：数千个 GPU 训练数周

**分布式解决方案**：
- 多个 Actor 并行收集数据
- 样本收集速度 × N（N = Actor 数量）

### 15.1.2 墙钟时间缩短

**墙钟时间**（Wall-clock Time）：实际流逝的时间。

**对比**：
| 方法 | 采样速度 | 训练到收敛 |
|------|---------|-----------|
| 单机 A3C | 100 steps/s | 10 小时 |
| 分布式 (100 Actors) | 10,000 steps/s | 6 分钟 |

**关键**：即使样本效率相同，墙钟时间大幅缩短！

### 15.1.3 探索多样性

**单个 Actor**：
- 探索轨迹相关性强
- 容易陷入局部探索

**多个 Actors**：
- 不同初始化 → 不同探索路径
- 更广泛的状态覆盖
- 发现罕见但有价值的状态

**示例**：蒙特祖玛的复仇（Montezuma's Revenge），需要多样化探索才能找到稀疏奖励。

<div data-component="DistributedRLArchitecture"></div>

---

## 15.2 Ape-X

Ape-X (Distributed Prioritized Experience Replay) 是分布式 DQN。

### 15.2.1 分布式经验收集

**架构**：

```
┌─────────┐  ┌─────────┐  ┌─────────┐
│ Actor 1 │  │ Actor 2 │  │ Actor N │  (并行收集经验)
└────┬────┘  └────┬────┘  └────┬────┘
     │            │            │
     └────────────┴────────────┘
                  │
           ┌──────▼──────┐
           │ Replay      │  (优先级回放 Buffer)
           │ Buffer      │
           └──────┬──────┘
                  │
           ┌──────▼──────┐
           │   Learner   │  (中心化训练)
           └──────┬──────┘
                  │
           (参数更新分发给 Actors)
```

**工作流程**：

1. **Actors**：并行与环境交互，收集 (s, a, r, s') 存入共享 buffer
2. **Replay Buffer**：维护大规模经验池（例如 1M transitions）
3. **Learner**：从 buffer 采样 batch，更新网络
4. **参数同步**：定期将新参数分发给 Actors

### 15.2.2 优先级回放

**Ape-X 使用 Prioritized Experience Replay (PER)**：

$$
P(i) = \frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}}
$$

其中优先级 $p_i$ 基于 TD 误差：

$$
p_i = |\delta_i| + \epsilon, \quad \delta_i = r + \gamma \max_{a'} Q(s', a') - Q(s, a)
$$

**分布式挑战**：
- Actors 无法实时计算 TD 误差（网络参数可能过时）
- 解决：Actors 上传经验到 buffer，Learner 异步计算优先级

### 15.2.3 中心化学习

**优势**：
- ✅ **解耦收集与训练**：Actors 专注采样，Learner 专注优化
- ✅ **硬件利用**：Actors 用 CPU，Learner 用 GPU
- ✅ **大 Batch**：Learner 可以用更大 batch size（更稳定的梯度）

**权衡**：
- ❌ **延迟**：Actor 的策略可能比 Learner 落后几个更新
- ❌ **Off-policy 程度更高**：需要修正（例如 importance sampling）

---

## 15.3 IMPALA

IMPALA (Importance Weighted Actor-Learner Architecture) 是分布式 A3C。

### 15.3.1 Importance Weighted Actor-Learner Architecture

**与 Ape-X 的区别**：
- **Ape-X**：off-policy (DQN)，有 replay buffer
- **IMPALA**：off-policy actor-critic，**无** replay buffer

**架构**：

```
┌─────────┐  ┌─────────┐  ┌─────────┐
│ Actor 1 │  │ Actor 2 │  │ Actor N │  (异步执行，使用旧策略)
└────┬────┘  └────┬────┘  └────┬────┘
     │            │            │
     │   (发送轨迹片段)        │
     └────────────┴────────────┘
                  │
           ┌──────▼──────┐
           │   Learner   │  (中心化训练，V-trace 修正)
           │   (GPU)     │
           └──────┬──────┘
                  │
           (广播新参数)
```

**关键特点**：
1. **异步 Actors**：不等待 Learner，持续收集数据
2. **轨迹片段**：Actors 发送短轨迹（例如 20 步）而非单步
3. **V-trace 修正**：补偿策略滞后（policy lag）

<div data-component="IMPALAFlow"></div>

### 15.3.2 V-trace 修正

**问题**：Actor 的策略 $\mu$ 可能与 Learner 的策略 $\pi$ 不同（policy lag）。

**V-trace 目标**：

$$
v_s = V(s) + \sum_{t=s}^{s+n-1} \gamma^{t-s} \left(\prod_{i=s}^{t-1} c_i\right) \delta_t
$$

其中：
- $\delta_t = \rho_t (r_t + \gamma V(s_{t+1}) - V(s_t))$：TD 误差
- $\rho_t = \min\left(\bar{\rho}, \frac{\pi(a_t|s_t)}{\mu(a_t|s_t)}\right)$：截断的重要性权重
- $c_i = \min\left(\bar{c}, \frac{\pi(a_i|s_i)}{\mu(a_i|s_i)}\right)$：trace 截断系数

**超参数**：
- $\bar{\rho} = 1.0$：TD 误差的截断阈值
- $\bar{c} = 1.0$：trace 的截断阈值

<div data-component="VTraceCorrection"></div>

**V-trace 的作用**：
- **修正 off-policy**：通过重要性权重调整
- **截断**：避免高方差（当 $\pi$ 和 $\mu$ 差异大时）
- **稳定训练**：即使策略滞后也能有效学习

### 15.3.3 异步 Actor-Learner

**通信模式**：

1. **Actors → Learner**：
   - 发送轨迹 $(s_t, a_t, r_t, \log \mu(a_t|s_t))$
   - 发送频率：每 N 步（例如 20 步）

2. **Learner → Actors**：
   - 广播新参数 $\boldsymbol{\theta}$
   - 更新频率：每个训练步（可异步）

**优势**：
- Actors 不阻塞等待参数更新
- Learner 总有新数据可训练
- 最大化 GPU 利用率

---

## 15.4 R2D2

R2D2 (Recurrent Experience Replay in Distributed RL) 处理部分可观测环境（POMDP）。

### 15.4.1 Recurrent Experience Replay

**动机**：许多环境是部分可观测的（例如：只看到部分地图）。

**解决方案**：使用 LSTM/GRU 记忆历史信息。

**挑战**：如何回放 LSTM 的经验？

**R2D2 的方法**：
- 存储**序列**而非单个 transition
- 回放时保持序列的时间顺序

### 15.4.2 Stored State

**存储内容**：

```python
# 传统 replay buffer
buffer.store(state, action, reward, next_state, done)

# R2D2 replay buffer
buffer.store(
    sequence_of_states,      # [s_0, s_1, ..., s_T]
    sequence_of_actions,     # [a_0, a_1, ..., a_T]
    sequence_of_rewards,     # [r_0, r_1, ..., r_T]
    initial_hidden_state     # h_0 (LSTM 的初始隐藏状态)
)
```

**关键**：保存 LSTM 的初始隐藏状态 $h_0$，以便正确重放。

### 15.4.3 Burn-in

**Burn-in**：在序列开始时，先"预热" LSTM 几步，再计算损失。

**原因**：
- 存储的 $h_0$ 可能过时（参数已更新）
- Burn-in 让 LSTM 根据当前参数重新构建隐藏状态

**实现**：

```python
# 序列长度 = burn_in + learning
burn_in_steps = 40
learning_steps = 80
total_sequence = burn_in_steps + learning_steps

# 前向传播
hidden = initial_hidden_state
for t in range(burn_in_steps):
    # Burn-in: 只前向，不计算损失
    q_values, hidden = lstm(state[t], hidden)

for t in range(burn_in_steps, total_sequence):
    # Learning: 计算 TD 误差和损失
    q_values, hidden = lstm(state[t], hidden)
    loss += compute_td_loss(q_values, ...)
```

**效果**：即使 $h_0$ 过时，经过 burn-in 后隐藏状态也能准确反映当前策略。

---

## 15.5 实现架构

分布式 RL 的实际工程实现。

### 15.5.1 Actor-Learner 分离

**Actor 进程**：

```python
class Actor:
    def __init__(self, env, parameter_server):
        self.env = env
        self.param_server = parameter_server
        self.policy = PolicyNetwork()
        
    def run(self):
        while True:
            # 1. 获取最新参数
            params = self.param_server.get_parameters()
            self.policy.load_state_dict(params)
            
            # 2. 收集轨迹
            trajectory = self.collect_trajectory()
            
            # 3. 发送到 buffer/learner
            self.param_server.send_trajectory(trajectory)
```

**Learner 进程**：

```python
class Learner:
    def __init__(self, replay_buffer, parameter_server):
        self.buffer = replay_buffer
        self.param_server = parameter_server
        self.policy = PolicyNetwork()
        self.optimizer = optim.Adam(self.policy.parameters())
        
    def run(self):
        while True:
            # 1. 从 buffer 采样
            batch = self.buffer.sample(batch_size)
            
            # 2. 训练
            loss = self.compute_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 3. 广播新参数
            self.param_server.update_parameters(self.policy.state_dict())
```

### 15.5.2 参数服务器

**功能**：
1. 存储当前最新参数
2. 接收 Actors 的参数请求
3. 接收 Learner 的参数更新

**简单实现**（使用共享内存/Redis/RPC）：

```python
import redis

class ParameterServer:
    def __init__(self):
        self.redis_client = redis.Redis()
        
    def update_parameters(self, params):
        """Learner 更新参数"""
        # 序列化参数
        params_bytes = pickle.dumps(params)
        self.redis_client.set('policy_params', params_bytes)
        
    def get_parameters(self):
        """Actors 获取参数"""
        params_bytes = self.redis_client.get('policy_params')
        return pickle.loads(params_bytes)
    
    def send_trajectory(self, trajectory):
        """Actors 发送轨迹到队列"""
        traj_bytes = pickle.dumps(trajectory)
        self.redis_client.rpush('trajectory_queue', traj_bytes)
    
    def get_trajectories(self, batch_size):
        """Learner 获取轨迹"""
        trajectories = []
        for _ in range(batch_size):
            traj_bytes = self.redis_client.lpop('trajectory_queue')
            if traj_bytes:
                trajectories.append(pickle.loads(traj_bytes))
        return trajectories
```

### 15.5.3 通信优化

**挑战**：参数传输的开销很大（例如一个网络有 10M 参数）。

**优化技术**：

1. **梯度聚合**（而非参数传输）：
   ```python
   # Actors 发送梯度而非轨迹
   grads = actor.compute_gradients(trajectory)
   param_server.aggregate_gradients(grads)
   
   # Learner 应用聚合后的梯度
   aggregated_grads = param_server.get_aggregated_gradients()
   optimizer.apply_gradients(aggregated_grads)
   ```

2. **压缩**：
   - 梯度量化（例如 float32 → int8）
   - Top-k 稀疏梯度（只传输最大的 k 个梯度）

3. **异步更新**：
   - Actors 不等待最新参数（可以用稍旧的）
   - Learner 不等待所有 Actors（收到足够数据就训练）

4. **本地 Buffer**：
   - 每个 Actor 维护小的本地 buffer
   - 批量发送到中心 buffer（减少通信次数）

---

## 15.6 完整分布式 RL 示例（简化版 IMPALA）

```python
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import gymnasium as gym
import numpy as np

class SimplifiedIMPALA:
    """
    简化版 IMPALA 实现
    
    包含:
    - 多个并行 Actors
    - 中心化 Learner
    - 参数共享（使用 multiprocessing）
    """
    def __init__(self, num_actors=4, env_name='CartPole-v1'):
        self.num_actors = num_actors
        self.env_name = env_name
        
        # 共享网络参数
        self.shared_policy = PolicyNetwork(state_dim=4, action_dim=2)
        self.shared_policy.share_memory()  # 关键：使参数可在进程间共享
        
        # 轨迹队列
        self.trajectory_queue = mp.Queue(maxsize=100)
        
    def actor_process(self, actor_id):
        """Actor 进程：收集轨迹"""
        env = gym.make(self.env_name)
        local_policy = PolicyNetwork(state_dim=4, action_dim=2)
        
        while True:
            # 同步参数
            local_policy.load_state_dict(self.shared_policy.state_dict())
            
            # 收集轨迹片段
            states, actions, rewards, log_probs = [], [], [], []
            state = env.reset()[0]
            
            for _ in range(20):  # 20 步轨迹
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs = local_policy(state_tensor)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                next_state, reward, done, truncated, _ = env.step(action.item())
                
                states.append(state)
                actions.append(action.item())
                rewards.append(reward)
                log_probs.append(log_prob.item())
                
                state = next_state
                if done or truncated:
                    state = env.reset()[0]
            
            # 发送轨迹
            trajectory = {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'log_probs': log_probs
            }
            self.trajectory_queue.put(trajectory)
    
    def learner_process(self):
        """Learner 进程：训练网络"""
        optimizer = torch.optim.Adam(self.shared_policy.parameters(), lr=1e-3)
        
        step = 0
        while step < 10000:
            # 收集一批轨迹
            batch = []
            for _ in range(4):  # batch size = 4 trajectories
                if not self.trajectory_queue.empty():
                    batch.append(self.trajectory_queue.get())
            
            if len(batch) == 0:
                continue
            
            # 计算损失（简化版，未使用 V-trace）
            total_loss = 0
            for traj in batch:
                states = torch.FloatTensor(traj['states'])
                actions = torch.LongTensor(traj['actions'])
                rewards = traj['rewards']
                
                # 计算 returns（简单累积）
                returns = []
                G = 0
                for r in reversed(rewards):
                    G = r + 0.99 * G
                    returns.insert(0, G)
                returns = torch.FloatTensor(returns)
                
                # Policy gradient loss
                action_probs = self.shared_policy(states)
                dist = torch.distributions.Categorical(action_probs)
                log_probs = dist.log_prob(actions)
                
                loss = -(log_probs * returns).mean()
                total_loss += loss
            
            # 更新共享参数
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            step += 1
            if step % 100 == 0:
                print(f"Learner step {step}, Loss: {total_loss.item():.3f}")
    
    def train(self):
        """启动分布式训练"""
        # 启动 Actors
        actor_processes = []
        for i in range(self.num_actors):
            p = mp.Process(target=self.actor_process, args=(i,))
            p.start()
            actor_processes.append(p)
        
        # 启动 Learner
        learner_p = mp.Process(target=self.learner_process)
        learner_p.start()
        
        # 等待训练完成
        learner_p.join()
        
        # 停止 Actors
        for p in actor_processes:
            p.terminate()


if __name__ == "__main__":
    # PyTorch multiprocessing 需要此设置
    mp.set_start_method('spawn', force=True)
    
    trainer = SimplifiedIMPALA(num_actors=4)
    trainer.train()
```

---

## 本章小结

在本章中，我们学习了：

✅ **分布式 RL 的必要性**：样本效率、墙钟时间、探索多样性  
✅ **Ape-X**：分布式 DQN，中心化优先级回放  
✅ **IMPALA**：异步 Actor-Learner，V-trace 修正  
✅ **R2D2**：循环经验回放，处理部分可观测性  
✅ **实现架构**：Actor-Learner 分离、参数服务器、通信优化  

> [!TIP]
> **核心要点**：
> - 分布式 RL 通过并行化大幅缩短训练时间
> - Ape-X 适合 off-policy（DQN, SAC），IMPALA 适合 on-policy（A3C, PPO）
> - V-trace 是关键创新，允许策略滞后而不失稳定性
> - R2D2 用序列回放和 burn-in 处理循环网络
> - 实际实现需要考虑通信开销和同步策略

> [!NOTE]
> **下一步**：
> Chapter 16 将学习**Model-based RL**：
> - 环境模型学习
> - Dyna 架构
> - MBPO 与世界模型
> - Dreamer 系列
> 
> 进入 [Chapter 16. Model-based RL](16-model-based-rl.md)

---

## 扩展阅读

- **经典论文**：
  - Horgan et al. (2018): Distributed Prioritized Experience Replay (Ape-X)
  - Espeholt et al. (2018): IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures
  - Kapturowski et al. (2019): Recurrent Experience Replay in Distributed RL (R2D2)
- **实现资源**：
  - Ray RLlib: Distributed RL framework
  - SEED RL: Google's scalable RL framework
  - Acme: DeepMind's RL framework
- **应用案例**：
  - AlphaStar (StarCraft II)
  - OpenAI Five (Dota 2)
  - DeepMind Lab 环境

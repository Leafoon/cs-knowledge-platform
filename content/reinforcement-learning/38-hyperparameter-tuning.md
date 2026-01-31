---
title: "第38章：超参数调优与实验设计"
description: "超参数重要性、调优方法、实验设计、性能评估、Benchmark标准、可复现性"
date: "2026-01-30"
---

# 第38章：超参数调优与实验设计

## 38.1 超参数重要性

### 38.1.1 学习率

**学习率** $\alpha$ 是最重要的超参数之一。

**影响**：
- **过大**：训练不稳定，发散
- **过小**：收敛慢，可能陷入局部最优
- **正合适**：稳定且快速收敛

**典型值**：
- **DQN**: $1 \times 10^{-4}$ to $5 \times 10^{-4}$
- **PPO**: $3 \times 10^{-4}$
- **SAC**: $3 \times 10^{-4}$
- **A3C**: $7 \times 10^{-4}$

**学习率调度**：

```python
"""
学习率调度策略
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

class LearningRateScheduler:
    """
    学习率调度器
    """
    def __init__(self, initial_lr, schedule_type='constant'):
        """
        Args:
            initial_lr: 初始学习率
            schedule_type: 'constant', 'linear', 'exponential', 'cosine', 'warmup_cosine'
        """
        self.initial_lr = initial_lr
        self.schedule_type = schedule_type
    
    def get_lr(self, step, total_steps):
        """
        根据当前步数获取学习率
        """
        if self.schedule_type == 'constant':
            return self.initial_lr
        
        elif self.schedule_type == 'linear':
            # 线性衰减
            return self.initial_lr * (1 - step / total_steps)
        
        elif self.schedule_type == 'exponential':
            # 指数衰减
            decay_rate = 0.96
            decay_steps = total_steps // 10
            return self.initial_lr * (decay_rate ** (step / decay_steps))
        
        elif self.schedule_type == 'cosine':
            # 余弦退火
            return self.initial_lr * 0.5 * (1 + np.cos(np.pi * step / total_steps))
        
        elif self.schedule_type == 'warmup_cosine':
            # Warmup + Cosine
            warmup_steps = total_steps // 10
            
            if step < warmup_steps:
                # Warmup阶段：线性增加
                return self.initial_lr * (step / warmup_steps)
            else:
                # Cosine衰减
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return self.initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
        
        else:
            return self.initial_lr
    
    def visualize(self, total_steps=10000):
        """可视化学习率变化"""
        steps = np.arange(total_steps)
        lrs = [self.get_lr(s, total_steps) for s in steps]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, lrs, linewidth=2)
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title(f'Learning Rate Schedule: {self.schedule_type}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig(f'lr_schedule_{self.schedule_type}.png', dpi=300, bbox_inches='tight')
        plt.show()


# 对比不同调度策略
schedules = ['constant', 'linear', 'exponential', 'cosine', 'warmup_cosine']

plt.figure(figsize=(14, 8))
total_steps = 10000

for i, schedule_type in enumerate(schedules):
    scheduler = LearningRateScheduler(initial_lr=3e-4, schedule_type=schedule_type)
    steps = np.arange(total_steps)
    lrs = [scheduler.get_lr(s, total_steps) for s in steps]
    
    plt.subplot(2, 3, i + 1)
    plt.plot(steps, lrs, linewidth=2, color=f'C{i}')
    plt.xlabel('Step', fontsize=10)
    plt.ylabel('LR', fontsize=10)
    plt.title(schedule_type.replace('_', ' ').title(), fontsize=12)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lr_schedules_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 38.1.2 折扣因子 γ

**折扣因子** $\gamma \in [0, 1]$ 决定未来奖励的权重。

**影响**：
- **γ → 1**：远见，长期规划（但方差大，训练慢）
- **γ → 0**：短视，即时奖励（低方差，但可能次优）

**典型值**：
- **短期任务**：$\gamma = 0.9 \sim 0.95$
- **长期任务**：$\gamma = 0.99$
- **无限水平**：$\gamma = 0.999$

**有效时间范围**：

$$
\text{Effective Horizon} = \frac{1}{1 - \gamma}
$$

- $\gamma = 0.9$ → 10步
- $\gamma = 0.99$ → 100步
- $\gamma = 0.999$ → 1000步

**敏感性分析**：

```python
"""
折扣因子敏感性分析
"""

def analyze_gamma_impact(env, policy_class, gammas, num_episodes=100):
    """
    分析不同gamma对性能的影响
    """
    results = {}
    
    for gamma in gammas:
        print(f"\n测试 γ={gamma}")
        
        # 训练策略
        policy = policy_class(gamma=gamma)
        rewards = train_policy(policy, env, num_episodes)
        
        results[gamma] = {
            'rewards': rewards,
            'final_mean': np.mean(rewards[-10:]),
            'convergence_step': find_convergence_point(rewards),
            'effective_horizon': 1 / (1 - gamma)
        }
    
    # 可视化
    plt.figure(figsize=(14, 5))
    
    # 学习曲线
    plt.subplot(1, 2, 1)
    for gamma in gammas:
        rewards = smooth(results[gamma]['rewards'], window=10)
        plt.plot(rewards, label=f'γ={gamma}')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('Learning Curves for Different γ', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 最终性能vs有效时间范围
    plt.subplot(1, 2, 2)
    final_perfs = [results[g]['final_mean'] for g in gammas]
    horizons = [results[g]['effective_horizon'] for g in gammas]
    
    plt.scatter(horizons, final_perfs, s=100, alpha=0.7)
    for g in gammas:
        plt.annotate(f'γ={g}', 
                    (results[g]['effective_horizon'], results[g]['final_mean']),
                    fontsize=10)
    plt.xlabel('Effective Horizon', fontsize=12)
    plt.ylabel('Final Performance', fontsize=12)
    plt.title('Performance vs Horizon', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gamma_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


# 运行分析
gammas = [0.9, 0.95, 0.99, 0.995, 0.999]
results = analyze_gamma_impact(env, PPO, gammas)
```

<div data-component="HyperparameterSensitivity"></div>

### 38.1.3 探索参数 ε

**ε-greedy探索**：

$$
a = \begin{cases}
\arg\max_a Q(s,a) & \text{with probability } 1-\epsilon \\
\text{random} & \text{with probability } \epsilon
\end{cases}
$$

**典型策略**：
- **初始**：$\epsilon = 1.0$（全探索）
- **最终**：$\epsilon = 0.01 \sim 0.1$
- **衰减**：线性或指数

```python
"""
ε衰减策略
"""

class EpsilonScheduler:
    """
    ε调度器
    """
    def __init__(self, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.current_eps = eps_start
    
    def step(self):
        """衰减ε"""
        self.current_eps = max(self.eps_end, self.current_eps * self.eps_decay)
        return self.current_eps
    
    def get_epsilon(self, step, schedule='exponential'):
        """
        获取当前ε值
        
        Args:
            step: 当前步数
            schedule: 'exponential', 'linear', 'inverse'
        """
        if schedule == 'exponential':
            return self.eps_end + (self.eps_start - self.eps_end) * (self.eps_decay ** step)
        
        elif schedule == 'linear':
            # 线性衰减（假设10000步）
            decay_steps = 10000
            fraction = min(step / decay_steps, 1.0)
            return self.eps_start - fraction * (self.eps_start - self.eps_end)
        
        elif schedule == 'inverse':
            # 反比衰减
            return self.eps_end + (self.eps_start - self.eps_end) / (1 + step / 1000)
        
        return self.current_eps
```

### 38.1.4 网络架构

**关键超参数**：
1. **层数**：2-4层典型
2. **每层神经元数**：64, 128, 256
3. **激活函数**：ReLU, Tanh, ELU
4. **正则化**：Dropout, Layer Norm, Weight Decay

**架构搜索**：

```python
"""
神经网络架构搜索
"""

class ArchitectureConfig:
    """网络架构配置"""
    def __init__(self, hidden_sizes, activation='relu', use_layer_norm=False):
        self.hidden_sizes = hidden_sizes  # e.g., [256, 256]
        self.activation = activation
        self.use_layer_norm = use_layer_norm
    
    def to_dict(self):
        return {
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'layer_norm': self.use_layer_norm
        }


def grid_search_architecture(env, configs, num_trials=5):
    """
    架构网格搜索
    """
    results = []
    
    for config in configs:
        print(f"\n测试架构: {config.to_dict()}")
        
        trial_rewards = []
        
        for trial in range(num_trials):
            # 训练
            policy = create_policy(config)
            rewards = train(policy, env, num_episodes=500)
            
            final_performance = np.mean(rewards[-50:])
            trial_rewards.append(final_performance)
        
        results.append({
            'config': config.to_dict(),
            'mean_perf': np.mean(trial_rewards),
            'std_perf': np.std(trial_rewards)
        })
    
    # 排序
    results.sort(key=lambda x: x['mean_perf'], reverse=True)
    
    # 打印Top 5
    print("\n=== Top 5 Architectures ===")
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. {result['config']}")
        print(f"   Performance: {result['mean_perf']:.2f} ± {result['std_perf']:.2f}")
    
    return results


# 候选架构
configs = [
    ArchitectureConfig([64, 64], 'relu'),
    ArchitectureConfig([128, 128], 'relu'),
    ArchitectureConfig([256, 256], 'relu'),
    ArchitectureConfig([256, 256, 256], 'relu'),
    ArchitectureConfig([256, 128], 'tanh'),
    ArchitectureConfig([256, 256], 'relu', use_layer_norm=True),
]

results = grid_search_architecture(env, configs)
```

---

## 38.2 调优方法

### 38.2.1 网格搜索（Grid Search）

**定义**：系统地遍历超参数空间的所有组合。

**优点**：
- 简单直观
- 完全覆盖
- 可并行

**缺点**：
- 指数级复杂度
- 很多计算浪费

**实现**：

```python
"""
网格搜索超参数调优
"""

from itertools import product
import json

class GridSearch:
    """
    网格搜索
    """
    def __init__(self, param_grid, env, algorithm_class):
        """
        Args:
            param_grid: 超参数网格，例如:
                {
                    'learning_rate': [1e-4, 3e-4, 1e-3],
                    'gamma': [0.99, 0.995],
                    'hidden_size': [128, 256]
                }
        """
        self.param_grid = param_grid
        self.env = env
        self.algorithm_class = algorithm_class
        self.results = []
    
    def run(self, num_trials=3, num_episodes=1000):
        """
        执行网格搜索
        """
        # 生成所有组合
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        all_combinations = list(product(*values))
        total_combinations = len(all_combinations)
        
        print(f"Total hyperparameter combinations: {total_combinations}")
        print(f"Total trials per combination: {num_trials}")
        print(f"Total runs: {total_combinations * num_trials}")
        
        for combo_idx, combo in enumerate(all_combinations):
            params = dict(zip(keys, combo))
            
            print(f"\n[{combo_idx + 1}/{total_combinations}] Testing: {params}")
            
            trial_results = []
            
            for trial in range(num_trials):
                # 创建算法
                agent = self.algorithm_class(**params)
                
                # 训练
                rewards, training_time = self._train(agent, num_episodes)
                
                # 记录结果
                trial_results.append({
                    'rewards': rewards,
                    'final_mean': np.mean(rewards[-50:]),
                    'auc': np.sum(rewards),  # Area Under Curve
                    'training_time': training_time
                })
            
            # 聚合trial结果
            self.results.append({
                'params': params,
                'mean_final_perf': np.mean([r['final_mean'] for r in trial_results]),
                'std_final_perf': np.std([r['final_mean'] for r in trial_results]),
                'mean_auc': np.mean([r['auc'] for r in trial_results]),
                'mean_time': np.mean([r['training_time'] for r in trial_results])
            })
        
        # 排序
        self.results.sort(key=lambda x: x['mean_final_perf'], reverse=True)
        
        return self.results
    
    def _train(self, agent, num_episodes):
        """训练agent"""
        import time
        start_time = time.time()
        
        rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                agent.update(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
            
            rewards.append(episode_reward)
        
        training_time = time.time() - start_time
        
        return rewards, training_time
    
    def get_best_params(self):
        """获取最佳超参数"""
        if not self.results:
            return None
        
        best = self.results[0]
        print(f"\n=== Best Hyperparameters ===")
        print(f"Parameters: {best['params']}")
        print(f"Performance: {best['mean_final_perf']:.2f} ± {best['std_final_perf']:.2f}")
        print(f"AUC: {best['mean_auc']:.2f}")
        print(f"Training Time: {best['mean_time']:.2f}s")
        
        return best['params']
    
    def save_results(self, filename='grid_search_results.json'):
        """保存结果"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {filename}")


# 使用示例
param_grid = {
    'learning_rate': [1e-4, 3e-4, 1e-3],
    'gamma': [0.99, 0.995, 0.999],
    'batch_size': [64, 128, 256],
    'hidden_size': [128, 256]
}

searcher = GridSearch(param_grid, env, PPO)
results = searcher.run(num_trials=3, num_episodes=500)
best_params = searcher.get_best_params()
searcher.save_results()
```

### 38.2.2 随机搜索（Random Search）

**优势**（Bergstra & Bengio, 2012）：
- 对不重要的超参数更鲁棒
- 更高效（相同预算下探索更多重要超参数的值）

**实现**：

```python
"""
随机搜索
"""

import random

class RandomSearch:
    """
    随机搜索超参数
    """
    def __init__(self, param_distributions, env, algorithm_class):
        """
        Args:
            param_distributions: 超参数分布，例如:
                {
                    'learning_rate': ('log_uniform', 1e-5, 1e-3),
                    'gamma': ('uniform', 0.95, 0.999),
                    'hidden_size': ('choice', [64, 128, 256, 512])
                }
        """
        self.param_distributions = param_distributions
        self.env = env
        self.algorithm_class = algorithm_class
        self.results = []
    
    def sample_params(self):
        """采样一组超参数"""
        params = {}
        
        for param_name, distribution in self.param_distributions.items():
            dist_type = distribution[0]
            
            if dist_type == 'uniform':
                low, high = distribution[1], distribution[2]
                params[param_name] = random.uniform(low, high)
            
            elif dist_type == 'log_uniform':
                log_low, log_high = np.log(distribution[1]), np.log(distribution[2])
                params[param_name] = np.exp(random.uniform(log_low, log_high))
            
            elif dist_type == 'choice':
                choices = distribution[1]
                params[param_name] = random.choice(choices)
            
            elif dist_type == 'int_uniform':
                low, high = distribution[1], distribution[2]
                params[param_name] = random.randint(low, high)
        
        return params
    
    def run(self, num_iterations=50, num_trials=3, num_episodes=500):
        """
        执行随机搜索
        
        Args:
            num_iterations: 采样次数
        """
        print(f"Random Search: {num_iterations} iterations, {num_trials} trials each")
        
        for iteration in range(num_iterations):
            # 采样超参数
            params = self.sample_params()
            
            print(f"\n[{iteration + 1}/{num_iterations}] Testing: {params}")
            
            trial_results = []
            
            for trial in range(num_trials):
                agent = self.algorithm_class(**params)
                rewards = train(agent, self.env, num_episodes)
                
                trial_results.append({
                    'final_mean': np.mean(rewards[-50:]),
                    'auc': np.sum(rewards)
                })
            
            self.results.append({
                'params': params,
                'mean_final_perf': np.mean([r['final_mean'] for r in trial_results]),
                'std_final_perf': np.std([r['final_mean'] for r in trial_results]),
                'mean_auc': np.mean([r['auc'] for r in trial_results])
            })
        
        # 排序
        self.results.sort(key=lambda x: x['mean_final_perf'], reverse=True)
        
        return self.results
    
    def get_best_params(self):
        """获取最佳超参数"""
        if not self.results:
            return None
        
        best = self.results[0]
        print(f"\n=== Best Hyperparameters (Random Search) ===")
        print(f"Parameters: {best['params']}")
        print(f"Performance: {best['mean_final_perf']:.2f} ± {best['std_final_perf']:.2f}")
        
        return best['params']


# 使用示例
param_distributions = {
    'learning_rate': ('log_uniform', 1e-5, 1e-2),
    'gamma': ('uniform', 0.95, 0.999),
    'batch_size': ('choice', [32, 64, 128, 256, 512]),
    'hidden_size': ('choice', [64, 128, 256, 512]),
    'epsilon': ('uniform', 0.1, 0.3)
}

random_searcher = RandomSearch(param_distributions, env, PPO)
results = random_searcher.run(num_iterations=50, num_trials=3)
best_params = random_searcher.get_best_params()
```

### 38.2.3 贝叶斯优化

**核心思想**：使用概率模型（Gaussian Process）建模超参数→性能的映射，智能选择下一个尝试的超参数。

**采集函数**：
- **Expected Improvement (EI)**
- **Upper Confidence Bound (UCB)**
- **Probability of Improvement (PI)**

**实现（使用Optuna）**：

```python
"""
贝叶斯优化using Optuna
"""

import optuna

def objective(trial):
    """
    Optuna目标函数
    
    Args:
        trial: Optuna trial对象，用于采样超参数
    
    Returns:
        performance: 要最大化的性能指标
    """
    # 采样超参数
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.95, 0.999)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256, 512])
    
    # 创建agent
    agent = PPO(
        learning_rate=lr,
        gamma=gamma,
        batch_size=batch_size,
        hidden_size=hidden_size
    )
    
    # 训练
    rewards = train(agent, env, num_episodes=500)
    
    # 返回性能（最后50个episode的平均）
    performance = np.mean(rewards[-50:])
    
    # 可选：报告中间结果（用于早停）
    for i, r in enumerate(rewards):
        trial.report(r, i)
        
        # 检查是否应该剪枝
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return performance


# 创建study
study = optuna.create_study(
    direction='maximize',  # 最大化性能
    sampler=optuna.samplers.TPESampler(),  # Tree-structured Parzen Estimator
    pruner=optuna.pruners.MedianPruner()  # 中位数剪枝
)

# 运行优化
study.optimize(objective, n_trials=100, timeout=3600)  # 100次试验或1小时

# 最佳结果
print("\n=== Best Trial ===")
print(f"Value: {study.best_value:.2f}")
print(f"Params: {study.best_params}")

# 可视化
import optuna.visualization as vis

# 优化历史
fig = vis.plot_optimization_history(study)
fig.show()

# 超参数重要性
fig = vis.plot_param_importances(study)
fig.show()

# 超参数关系
fig = vis.plot_contour(study, params=['learning_rate', 'gamma'])
fig.show()
```

继续...

### 38.2.4 Population-Based Training (PBT)

**核心思想**（Jaderberg et al., 2017）：在训练过程中**动态调整**超参数。

**机制**：
1. 并行训练多个agent（population）
2. 定期评估performance
3. **Exploit**: 表现差的复制表现好的权重
4. **Explore**: 扰动超参数

**算法**：

```python
"""
Population-Based Training
"""

import copy

class PopulationBasedTraining:
    """
    PBT for RL超参数优化
    """
    def __init__(
        self,
        population_size,
        env,
        algorithm_class,
        hyperparams_to_optimize
    ):
        """
        Args:
            population_size: population大小
            hyperparams_to_optimize: 要优化的超参数及其范围
                例如: {'learning_rate': (1e-5, 1e-2)}
        """
        self.population_size = population_size
        self.env = env
        self.algorithm_class = algorithm_class
        self.hyperparams_to_optimize = hyperparams_to_optimize
        
        # 初始化population
        self.population = self._initialize_population()
    
    def _initialize_population(self):
        """初始化population"""
        population = []
        
        for i in range(self.population_size):
            # 随机采样超参数
            hyperparams = {}
            for param, (low, high) in self.hyperparams_to_optimize.items():
                if param == 'learning_rate':
                    # log_uniform
                    hyperparams[param] = np.exp(np.random.uniform(np.log(low), np.log(high)))
                else:
                    hyperparams[param] = np.random.uniform(low, high)
            
            # 创建agent
            agent = self.algorithm_class(**hyperparams)
            
            population.append({
                'agent': agent,
                'hyperparams': hyperparams,
                'performance': 0.0,
                'history': []
            })
        
        return population
    
    def exploit_and_explore(self, member_idx):
        """
        Exploit: 复制好的agent
        Explore: 扰动超参数
        """
        # 按performance排序
        sorted_pop = sorted(self.population, key=lambda x: x['performance'], reverse=True)
        
        # 如果当前member在bottom 20%，exploit & explore
        if member_idx >= int(0.8 * self.population_size):
            # Exploit: 从top 20%随机选一个
            top_k = int(0.2 * self.population_size)
            source = np.random.choice(sorted_pop[:top_k])
            
            # 复制权重
            self.population[member_idx]['agent'].load_state_dict(
                source['agent'].state_dict()
            )
            
            # Explore: 扰动超参数（±20%）
            for param, value in source['hyperparams'].items():
                perturb_factor = np.random.choice([0.8, 1.2])  # ±20%
                new_value = value * perturb_factor
                
                # 裁剪到合法范围
                low, high = self.hyperparams_to_optimize[param]
                new_value = np.clip(new_value, low, high)
                
                self.population[member_idx]['hyperparams'][param] = new_value
            
            # 更新agent的超参数
            self.population[member_idx]['agent'].update_hyperparams(
                self.population[member_idx]['hyperparams']
            )
    
    def run(self, total_steps, eval_interval=1000):
        """
        运行PBT
        
        Args:
            total_steps: 总训练步数
            eval_interval: 评估间隔
        """
        step = 0
        
        while step < total_steps:
            # 每个member训练eval_interval步
            for member_idx, member in enumerate(self.population):
                # 训练
                rewards = []
                
                for _ in range(eval_interval // 200):  # 假设每个episode 200步
                    episode_reward = run_episode(member['agent'], self.env)
                    rewards.append(episode_reward)
                
                # 更新performance
                member['performance'] = np.mean(rewards)
                member['history'].append(member['performance'])
            
            # Exploit & Explore
            for member_idx in range(self.population_size):
                self.exploit_and_explore(member_idx)
            
            step += eval_interval
            
            # 日志
            best_perf = max([m['performance'] for m in self.population])
            print(f"Step {step}/{total_steps}, Best Performance: {best_perf:.2f}")
        
        # 返回最佳agent
        best_member = max(self.population, key=lambda x: x['performance'])
        
        print(f"\n=== PBT Complete ===")
        print(f"Best Hyperparameters: {best_member['hyperparams']}")
        print(f"Best Performance: {best_member['performance']:.2f}")
        
        return best_member


# 使用示例
hyperparams_to_optimize = {
    'learning_rate': (1e-5, 1e-2),
    'gamma': (0.95, 0.999),
    'epsilon': (0.05, 0.3)
}

pbt = PopulationBasedTraining(
    population_size=10,
    env=env,
    algorithm_class=PPO,
    hyperparams_to_optimize=hyperparams_to_optimize
)

best_member = pbt.run(total_steps=100000, eval_interval=5000)
```

---

## 38.3 实验设计

### 38.3.1 随机种子控制

**重要性**：RL结果方差很大，随机种子至关重要。

**最佳实践**：

```python
"""
随机种子控制
"""

import random
import numpy as np
import torch

def set_seed(seed):
    """
    设置所有随机种子
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # 确保CUDA确定性（可能牺牲性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Gym环境种子
    # env.seed(seed)  # 已废弃，使用env.reset(seed=seed)


# 多种子训练
def train_with_multiple_seeds(algorithm, env_fn, seeds, num_episodes=1000):
    """
    使用多个随机种子训练
    
    Args:
        seeds: 种子列表，例如 [0, 1, 2, 3, 4]
    """
    all_results = []
    
    for seed in seeds:
        print(f"\n=== Training with seed {seed} ===")
        
        # 设置种子
        set_seed(seed)
        
        # 创建环境和agent
        env = env_fn()
        agent = algorithm()
        
        # 训练
        rewards = []
        
        for episode in range(num_episodes):
            state = env.reset(seed=seed + episode)
            done = False
            episode_reward = 0
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.update(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
            
            rewards.append(episode_reward)
        
        all_results.append({
            'seed': seed,
            'rewards': rewards,
            'final_mean': np.mean(rewards[-50:])
        })
    
    return all_results


# 运行
seeds = [0, 1, 2, 3, 4]  # 至少5个种子
results = train_with_multiple_seeds(PPO, lambda: gym.make('CartPole-v1'), seeds)
```

### 38.3.2 多次运行统计

**报告标准**（Agarwal et al., 2021）：

- **均值 ± 标准差**
- **中位数 + IQR**（四分位距）
- **置信区间**

```python
"""
多次运行统计分析
"""

import scipy.stats as stats

class MultiRunStatistics:
    """
    多次运行统计
    """
    def __init__(self, results):
        """
        Args:
            results: 列表，每个元素是一次运行的rewards列表
        """
        self.results = results
        self.num_runs = len(results)
    
    def compute_statistics(self):
        """计算统计量"""
        # 最终性能（最后50个episode）
        final_perfs = [np.mean(r[-50:]) for r in self.results]
        
        stats_dict = {
            'mean': np.mean(final_perfs),
            'std': np.std(final_perfs),
            'median': np.median(final_perfs),
            '25th_percentile': np.percentile(final_perfs, 25),
            '75th_percentile': np.percentile(final_perfs, 75),
            'min': np.min(final_perfs),
            'max': np.max(final_perfs)
        }
        
        # IQR
        stats_dict['iqr'] = stats_dict['75th_percentile'] - stats_dict['25th_percentile']
        
        return stats_dict
    
    def confidence_interval(self, confidence=0.95):
        """
        计算置信区间
        
        Args:
            confidence: 置信水平（0.95 = 95%）
        """
        final_perfs = [np.mean(r[-50:]) for r in self.results]
        
        # t-分布（样本量小时更准确）
        mean = np.mean(final_perfs)
        std_err = stats.sem(final_perfs)  # 标准误
        
        # t值
        t_crit = stats.t.ppf((1 + confidence) / 2, len(final_perfs) - 1)
        
        margin_of_error = t_crit * std_err
        
        ci_lower = mean - margin_of_error
        ci_upper = mean + margin_of_error
        
        return (ci_lower, ci_upper)
    
    def plot_learning_curves(self):
        """绘制学习曲线（均值 + 标准差阴影）"""
        # 对齐长度（取最短）
        min_length = min(len(r) for r in self.results)
        aligned = np.array([r[:min_length] for r in self.results])
        
        # 计算均值和标准差
        mean_rewards = aligned.mean(axis=0)
        std_rewards = aligned.std(axis=0)
        
        # 平滑
        window = 10
        mean_smooth = np.convolve(mean_rewards, np.ones(window)/window, mode='valid')
        std_smooth = np.convolve(std_rewards, np.ones(window)/window, mode='valid')
        
        episodes = np.arange(len(mean_smooth))
        
        plt.figure(figsize=(10, 6))
        
        # 均值线
        plt.plot(episodes, mean_smooth, linewidth=2, label='Mean')
        
        # 标准差阴影
        plt.fill_between(
            episodes,
            mean_smooth - std_smooth,
            mean_smooth + std_smooth,
            alpha=0.3,
            label='±1 Std'
        )
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Reward', fontsize=12)
        plt.title(f'Learning Curve ({self.num_runs} runs)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('learning_curve_multi_run.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def report(self):
        """生成报告"""
        stats_dict = self.compute_statistics()
        ci = self.confidence_interval()
        
        print("\n" + "="*60)
        print("Multi-Run Statistics")
        print("="*60)
        print(f"Number of runs: {self.num_runs}")
        print(f"\nFinal Performance:")
        print(f"  Mean ± Std: {stats_dict['mean']:.2f} ± {stats_dict['std']:.2f}")
        print(f"  Median [IQR]: {stats_dict['median']:.2f} [{stats_dict['25th_percentile']:.2f}, {stats_dict['75th_percentile']:.2f}]")
        print(f"  95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
        print(f"  Range: [{stats_dict['min']:.2f}, {stats_dict['max']:.2f}]")
        print("="*60)


# 使用示例
results = [r['rewards'] for r in train_with_multiple_seeds(...)]
stats_analyzer = MultiRunStatistics(results)
stats_analyzer.report()
stats_analyzer.plot_learning_curves()
```

<div data-component="LearningCurveComparison"></div>

### 38.3.3 显著性检验

**目标**：判断算法A是否**显著优于**算法B。

**常用检验**：
1. **t-test**（参数检验）
2. **Mann-Whitney U test**（非参数）
3. **Bootstrap**

```python
"""
显著性检验
"""

from scipy.stats import ttest_ind, mannwhitneyu
from scipy.stats import bootstrap

def significance_test(results_A, results_B, alpha=0.05):
    """
    比较两个算法的显著性
    
    Args:
        results_A: 算法A的多次运行结果
        results_B: 算法B的多次运行结果
        alpha: 显著性水平（0.05 = 5%）
    """
    # 提取最终性能
    perfs_A = [np.mean(r[-50:]) for r in results_A]
    perfs_B = [np.mean(r[-50:]) for r in results_B]
    
    print("\n" + "="*60)
    print("Significance Test: Algorithm A vs Algorithm B")
    print("="*60)
    
    # 1. t-test（假设正态分布）
    t_stat, p_value_t = ttest_ind(perfs_A, perfs_B)
    
    print(f"\n1. Independent t-test:")
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value: {p_value_t:.4f}")
    
    if p_value_t < alpha:
        winner = "A" if np.mean(perfs_A) > np.mean(perfs_B) else "B"
        print(f"   ✅ Algorithm {winner} is significantly better (p < {alpha})")
    else:
        print(f"   ❌ No significant difference (p >= {alpha})")
    
    # 2. Mann-Whitney U test（非参数，不假设正态）
    u_stat, p_value_u = mannwhitneyu(perfs_A, perfs_B, alternative='two-sided')
    
    print(f"\n2. Mann-Whitney U test:")
    print(f"   U-statistic: {u_stat:.4f}")
    print(f"   p-value: {p_value_u:.4f}")
    
    if p_value_u < alpha:
        winner = "A" if np.median(perfs_A) > np.median(perfs_B) else "B"
        print(f"   ✅ Algorithm {winner} is significantly better (p < {alpha})")
    else:
        print(f"   ❌ No significant difference (p >= {alpha})")
    
    # 3. Effect Size (Cohen's d)
    mean_diff = np.mean(perfs_A) - np.mean(perfs_B)
    pooled_std = np.sqrt((np.std(perfs_A)**2 + np.std(perfs_B)**2) / 2)
    cohens_d = mean_diff / pooled_std
    
    print(f"\n3. Effect Size (Cohen's d): {cohens_d:.4f}")
    
    if abs(cohens_d) < 0.2:
        effect = "Small"
    elif abs(cohens_d) < 0.5:
        effect = "Medium"
    else:
        effect = "Large"
    
    print(f"   Effect: {effect}")
    
    print("="*60)
    
    return {
        't_test_p': p_value_t,
        'u_test_p': p_value_u,
        'cohens_d': cohens_d
    }
```

---

## 38.4 性能评估

### 38.4.1 学习曲线分析

**关键指标**：
- **收敛速度**：达到性能阈值的步数
- **最终性能**：收敛后的平均性能
- **样本效率**：AUC（曲线下面积）

```python
"""
学习曲线分析
"""

def analyze_learning_curve(rewards, threshold=200):
    """
    分析学习曲线
    
    Args:
        rewards: episode rewards列表
        threshold: 性能阈值
    """
    # 平滑
    window = 50
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    # 1. 收敛速度
    convergence_episode = None
    
    for i in range(len(smoothed) - 50):
        # 检查连续50个episode是否都超过阈值
        if np.all(smoothed[i:i+50] >= threshold):
            convergence_episode = i
            break
    
    # 2. 最终性能
    final_performance = np.mean(smoothed[-50:])
    
    # 3. 样本效率（AUC）
    auc = np.sum(rewards)
    
    # 4. 方差（稳定性）
    final_variance = np.var(rewards[-100:])
    
    print("\n" + "="*60)
    print("Learning Curve Analysis")
    print("="*60)
    print(f"Convergence Episode: {convergence_episode if convergence_episode else 'Not converged'}")
    print(f"Final Performance: {final_performance:.2f}")
    print(f"Sample Efficiency (AUC): {auc:.2f}")
    print(f"Final Variance: {final_variance:.2f}")
    print("="*60)
    
    return {
        'convergence_episode': convergence_episode,
        'final_performance': final_performance,
        'auc': auc,
        'final_variance': final_variance
    }
```

### 38.4.2 Ablation Study

**目的**：分析每个组件的贡献。

**示例**：PPO的ablation

```python
"""
Ablation Study
"""

def ablation_study(base_algorithm, components_to_ablate):
    """
    消融实验
    
    Args:
        base_algorithm: 完整算法
        components_to_ablate: 要消融的组件
            例如: ['clip', 'value_clip', 'gae', 'entropy_bonus']
    """
    results = {}
    
    # 1. 完整算法
    print("\n=== Full Algorithm ===")
    full_rewards = train(base_algorithm(all_components=True), env, num_episodes=500)
    results['Full'] = {
        'final_perf': np.mean(full_rewards[-50:]),
        'rewards': full_rewards
    }
    
    # 2. 逐个移除组件
    for component in components_to_ablate:
        print(f"\n=== Without {component} ===")
        
        # 创建无该组件的算法
        ablated_algo = base_algorithm(**{component: False})
        
        ablated_rewards = train(ablated_algo, env, num_episodes=500)
        
        results[f'w/o {component}'] = {
            'final_perf': np.mean(ablated_rewards[-50:]),
            'rewards': ablated_rewards
        }
    
    # 可视化
    plt.figure(figsize=(12, 6))
    
    for name, data in results.items():
        smoothed = smooth(data['rewards'], window=10)
        plt.plot(smoothed, label=name, linewidth=2)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('Ablation Study', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('ablation_study.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印贡献
    print("\n" + "="*60)
    print("Component Contributions")
    print("="*60)
    
    full_perf = results['Full']['final_perf']
    
    for component in components_to_ablate:
        ablated_perf = results[f'w/o {component}']['final_perf']
        contribution = full_perf - ablated_perf
        percentage = (contribution / full_perf) * 100
        
        print(f"{component:20s}: {contribution:+.2f} ({percentage:+.1f}%)")
    
    print("="*60)
    
    return results


# 运行
components = ['clip', 'value_clip', 'gae', 'entropy_bonus']
ablation_results = ablation_study(PPO, components)
```

<div data-component="AblationStudyVisualizer"></div>

---

## 38.5 Benchmark标准

### 38.5.1 Atari 2600

**标准**：57个游戏，200M frames训练。

**指标**：
- Human-normalized score
- Median & IQR over games

```python
"""
Atari Benchmark
"""

# Human-normalized score
def human_normalized_score(agent_score, random_score, human_score):
    """
    人类归一化分数
    
    100% = 人类水平
    0% = 随机策略
    """
    return (agent_score - random_score) / (human_score - random_score) * 100


# 示例
atari_games = {
    'Pong': {'random': -20.7, 'human': 14.6},
    'Breakout': {'random': 1.7, 'human': 30.5},
    'Seaquest': {'random': 68.4, 'human': 42054.7},
    # ... 其他游戏
}

agent_scores = {
    'Pong': 21.0,
    'Breakout': 400.0,
    'Seaquest': 5800.0
}

for game, scores in atari_games.items():
    agent_score = agent_scores[game]
    normalized = human_normalized_score(
        agent_score,
        scores['random'],
        scores['human']
    )
    print(f"{game:15s}: {normalized:.1f}% human")
```

### 38.5.2 MuJoCo连续控制

**环境**：
- HalfCheetah
- Hopper
- Walker2d
- Ant

**报告**：最终性能 + 学习曲线

<div data-component="BenchmarkLeaderboard"></div>

---

## 38.6 可复现性

### 38.6.1 代码开源

**最佳实践**：
- 完整代码 + 超参数
- README with环境配置
- Requirements.txt

```bash
# requirements.txt示例
numpy==1.24.0
torch==2.0.0
gymnasium==0.29.0
stable-baselines3==2.1.0
tensorboard==2.14.0
```

### 38.6.2 结果报告规范

**包含**：
- 超参数完整列表
- 随机种子
- 训练时间
- 硬件信息

```python
"""
实验报告生成
"""

import platform
import torch

def generate_experiment_report(config, results):
    """生成实验报告"""
    report = f"""
# Experiment Report

## Environment
- OS: {platform.system()} {platform.release()}
- CPU: {platform.processor()}
- GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}
- Python: {platform.python_version()}
- PyTorch: {torch.__version__}

## Hyperparameters
"""
    
    for key, value in config.items():
        report += f"- {key}: {value}\n"
    
    report += f"""
## Results
- Final Performance: {results['final_mean']:.2f} ± {results['final_std']:.2f}
- Training Time: {results['training_time']:.2f}s
- Seeds Used: {results['seeds']}

## Reproducibility
```bash
python train.py --seed {results['seeds'][0]} --config config.yaml
```
"""
    
    with open('experiment_report.md', 'w') as f:
        f.write(report)
    
    print("Report saved to experiment_report.md")
```

---

## 总结

本章介绍了RL的超参数调优与实验设计：

1. **超参数重要性**：学习率（调度策略）、γ（有效时间范围）、ε（探索策略）、网络架构
2. **调优方法**：网格搜索、随机搜索（更高效）、贝叶斯优化（Optuna、TPE）、PBT（动态调整）
3. **实验设计**：随机种子控制、多次运行统计、置信区间、显著性检验（t-test、Mann-Whitney）
4. **性能评估**：学习曲线分析、样本效率（AUC）、Ablation Study
5. **Benchmark标准**：Atari（人类归一化）、MuJoCo、Procgen、D4RL
6. **可复现性**：代码开源、超参数记录、环境版本、完整报告

**关键要点**：
- 学习率是最重要的超参数
- 随机搜索通常优于网格搜索
- 至少5个随机种子 + 置信区间
- 报告均值 ± 标准差 OR 中位数 + IQR
- Ablation Study揭示组件贡献
- 标准化benchmark便于比较

---

## 参考文献

- Bergstra, J., & Bengio, Y. (2012). "Random Search for Hyper-Parameter Optimization." *JMLR*.
- Jaderberg, M., et al. (2017). "Population Based Training of Neural Networks." *arXiv*.
- Henderson, P., et al. (2018). "Deep Reinforcement Learning That Matters." *AAAI*.
- Engstrom, L., et al. (2020). "Implementation Matters in Deep Policy Gradients." *ICLR*.
- Agarwal, R., et al. (2021). "Deep Reinforcement Learning at the Edge of the Statistical Precipice." *NeurIPS*.

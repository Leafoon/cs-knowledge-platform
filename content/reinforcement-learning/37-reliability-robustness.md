---
title: "第37章：可靠性与鲁棒性"
description: "分布漂移、对抗鲁棒性、不确定性量化、OOD检测、可解释性"
date: "2026-01-30"
---

# 第37章：可靠性与鲁棒性

## 37.1 分布漂移（Distribution Shift）

### 37.1.1 协变量偏移

**定义**：训练时的状态分布 $P_{\text{train}}(s)$ 与部署时的分布 $P_{\text{deploy}}(s)$ 不同。

**问题**：
- 训练环境 ≠ 真实环境
- 模拟器到现实（Sim-to-Real）
- 环境动态变化

**数学形式化**：

$$
P_{\text{train}}(s, a, r) \neq P_{\text{deploy}}(s, a, r)
$$

但转移动态可能相同：$P(s'|s,a)$ 保持不变。

**示例与检测**：

```python
"""
分布漂移检测与可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import ks_2samp, wasserstein_distance

class DistributionShiftDetector:
    """
    检测训练分布与部署分布的偏移
    """
    def __init__(self):
        self.train_states = []
        self.deploy_states = []
    
    def add_train_data(self, states):
        """收集训练数据"""
        self.train_states.extend(states)
    
    def add_deploy_data(self, states):
        """收集部署数据"""
        self.deploy_states.extend(states)
    
    def kolmogorov_smirnov_test(self, feature_idx=0):
        """
        Kolmogorov-Smirnov检验
        
        H0: 两个分布相同
        """
        train_feature = [s[feature_idx] for s in self.train_states]
        deploy_feature = [s[feature_idx] for s in self.deploy_states]
        
        statistic, p_value = ks_2samp(train_feature, deploy_feature)
        
        print(f"KS检验 (特征 {feature_idx}):")
        print(f"  统计量: {statistic:.4f}")
        print(f"  p值: {p_value:.4f}")
        print(f"  分布不同: {p_value < 0.05}")
        
        return statistic, p_value
    
    def wasserstein_distance_metric(self):
        """
        Wasserstein距离（Earth Mover's Distance）
        
        衡量两个分布之间的"移动成本"
        """
        # 降维（如果是高维状态）
        all_states = np.array(self.train_states + self.deploy_states)
        
        if all_states.shape[1] > 2:
            pca = PCA(n_components=2)
            all_states_2d = pca.fit_transform(all_states)
        else:
            all_states_2d = all_states
        
        train_2d = all_states_2d[:len(self.train_states)]
        deploy_2d = all_states_2d[len(self.train_states):]
        
        # 计算Wasserstein距离
        dist = wasserstein_distance(
            train_2d.flatten(),
            deploy_2d.flatten()
        )
        
        print(f"\nWasserstein距离: {dist:.4f}")
        
        return dist
    
    def maximum_mean_discrepancy(self, kernel_bandwidth=1.0):
        """
        Maximum Mean Discrepancy (MMD)
        
        基于核方法的分布差异度量
        """
        def rbf_kernel(X, Y, bandwidth):
            """RBF核"""
            XX = np.sum(X**2, axis=1).reshape(-1, 1)
            YY = np.sum(Y**2, axis=1).reshape(-1, 1)
            XY = np.dot(X, Y.T)
            distances = XX + YY.T - 2*XY
            return np.exp(-distances / (2 * bandwidth**2))
        
        X = np.array(self.train_states)
        Y = np.array(self.deploy_states)
        
        n, m = len(X), len(Y)
        
        # MMD² = E[k(X,X')] + E[k(Y,Y')] - 2E[k(X,Y)]
        K_XX = rbf_kernel(X, X, kernel_bandwidth)
        K_YY = rbf_kernel(Y, Y, kernel_bandwidth)
        K_XY = rbf_kernel(X, Y, kernel_bandwidth)
        
        mmd_squared = (
            (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1)) +
            (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1)) -
            2 * K_XY.sum() / (n * m)
        )
        
        mmd = np.sqrt(max(mmd_squared, 0))
        
        print(f"\nMMD: {mmd:.4f}")
        
        return mmd
    
    def visualize_shift(self):
        """
        可视化分布偏移
        """
        # PCA降维到2D
        all_states = np.array(self.train_states + self.deploy_states)
        pca = PCA(n_components=2)
        all_states_2d = pca.fit_transform(all_states)
        
        train_2d = all_states_2d[:len(self.train_states)]
        deploy_2d = all_states_2d[len(self.train_states):]
        
        plt.figure(figsize=(12, 5))
        
        # 散点图
        plt.subplot(1, 2, 1)
        plt.scatter(train_2d[:, 0], train_2d[:, 1], alpha=0.5, label='Train', s=20)
        plt.scatter(deploy_2d[:, 0], deploy_2d[:, 1], alpha=0.5, label='Deploy', s=20)
        plt.xlabel('PC1', fontsize=12)
        plt.ylabel('PC2', fontsize=12)
        plt.title('State Distribution (PCA)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 密度直方图
        plt.subplot(1, 2, 2)
        plt.hist(train_2d[:, 0], bins=30, alpha=0.5, label='Train', density=True)
        plt.hist(deploy_2d[:, 0], bins=30, alpha=0.5, label='Deploy', density=True)
        plt.xlabel('PC1 Value', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Distribution Comparison', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('distribution_shift.png', dpi=300, bbox_inches='tight')
        plt.show()


# 使用示例
detector = DistributionShiftDetector()

# 模拟训练数据（正态分布）
train_states = np.random.randn(1000, 4)
detector.add_train_data(train_states)

# 模拟部署数据（偏移的分布）
deploy_states = np.random.randn(1000, 4) + np.array([0.5, 0.3, 0.0, 0.8])
detector.add_deploy_data(deploy_states)

# 检测
detector.kolmogorov_smirnov_test(feature_idx=0)
detector.wasserstein_distance_metric()
detector.maximum_mean_discrepancy()
detector.visualize_shift()
```

<div data-component="DistributionShiftVisualization"></div>

### 37.1.2 域适应

**Domain Adaptation**：使训练在源域（source domain）的策略能够泛化到目标域（target domain）。

**方法**：

1. **Domain Randomization**（域随机化）
2. **Adversarial Domain Adaptation**
3. **Fine-tuning**

**Domain Randomization实现**：

```python
"""
Domain Randomization for Sim-to-Real Transfer
"""

import gym
import numpy as np

class DomainRandomizedEnv(gym.Wrapper):
    """
    域随机化环境包装器
    
    随机化物理参数以增强泛化能力
    """
    def __init__(self, env):
        super().__init__(env)
        self.original_params = self._get_physics_params()
    
    def _get_physics_params(self):
        """获取当前物理参数"""
        # 示例：CartPole的物理参数
        return {
            'gravity': self.env.unwrapped.gravity,
            'masspole': self.env.unwrapped.masspole,
            'length': self.env.unwrapped.length,
            'force_mag': self.env.unwrapped.force_mag
        }
    
    def _randomize_physics(self):
        """
        随机化物理参数
        
        在合理范围内扰动参数
        """
        # 重力 ±20%
        self.env.unwrapped.gravity = self.original_params['gravity'] * (
            1 + np.random.uniform(-0.2, 0.2)
        )
        
        # 杆质量 ±30%
        self.env.unwrapped.masspole = self.original_params['masspole'] * (
            1 + np.random.uniform(-0.3, 0.3)
        )
        
        # 杆长度 ±15%
        self.env.unwrapped.length = self.original_params['length'] * (
            1 + np.random.uniform(-0.15, 0.15)
        )
        
        # 力大小 ±25%
        self.env.unwrapped.force_mag = self.original_params['force_mag'] * (
            1 + np.random.uniform(-0.25, 0.25)
        )
    
    def reset(self, **kwargs):
        """每个episode开始时随机化"""
        self._randomize_physics()
        return self.env.reset(**kwargs)


# 训练with Domain Randomization
env = gym.make('CartPole-v1')
randomized_env = DomainRandomizedEnv(env)

# 训练策略（示例）
for episode in range(1000):
    state = randomized_env.reset()
    done = False
    
    while not done:
        # 策略选择动作
        action = policy(state)
        state, reward, done, info = randomized_env.step(action)
    
    # ... 训练更新


print("训练完成 - 策略已在多样化物理参数下学习，具有更好的泛化能力")
```

**对抗域适应**：

```python
"""
Adversarial Domain Adaptation for RL
"""

import torch
import torch.nn as nn

class DomainDiscriminator(nn.Module):
    """
    域判别器
    
    判断状态来自源域还是目标域
    """
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state):
        """
        Returns:
            probability that state is from target domain
        """
        return self.network(state)


class DomainAdaptivePolicy(nn.Module):
    """
    域自适应策略
    
    同时优化任务性能和域混淆
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # 特征提取器（域不变）
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        features = self.feature_extractor(state)
        action_probs = self.policy_head(features)
        return action_probs, features


def train_with_domain_adaptation(
    source_env,
    target_env,
    num_epochs=100,
    lambda_domain=0.1
):
    """
    训练域自适应策略
    
    Args:
        source_env: 源域环境（有标签反馈）
        target_env: 目标域环境（少量或无标签）
        lambda_domain: 域对抗损失权重
    """
    state_dim = source_env.observation_space.shape[0]
    action_dim = source_env.action_space.n
    
    policy = DomainAdaptivePolicy(state_dim, action_dim)
    discriminator = DomainDiscriminator(128)  # 128是特征维度
    
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        # 1. 从源域采样
        source_states = collect_states(source_env, num_samples=256)
        source_states_tensor = torch.FloatTensor(source_states)
        
        # 2. 从目标域采样
        target_states = collect_states(target_env, num_samples=256)
        target_states_tensor = torch.FloatTensor(target_states)
        
        # 3. 训练判别器（区分域）
        _, source_features = policy(source_states_tensor)
        _, target_features = policy(target_states_tensor)
        
        # 判别器损失
        source_domain_pred = discriminator(source_features.detach())
        target_domain_pred = discriminator(target_features.detach())
        
        disc_loss = (
            -torch.log(source_domain_pred + 1e-8).mean() -
            torch.log(1 - target_domain_pred + 1e-8).mean()
        )
        
        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()
        
        # 4. 训练策略（任务损失 + 域混淆损失）
        # RL任务损失（在源域有监督）
        rl_loss = compute_rl_loss(policy, source_env)
        
        # 域混淆损失（愚弄判别器）
        _, source_features = policy(source_states_tensor)
        _, target_features = policy(target_states_tensor)
        
        # 让判别器无法区分
        domain_confusion_loss = (
            torch.log(1 - source_domain_pred + 1e-8).mean() +
            torch.log(target_domain_pred + 1e-8).mean()
        )
        
        # 总损失
        total_loss = rl_loss + lambda_domain * domain_confusion_loss
        
        policy_optimizer.zero_grad()
        total_loss.backward()
        policy_optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}:")
            print(f"  RL Loss: {rl_loss.item():.4f}")
            print(f"  Domain Confusion: {domain_confusion_loss.item():.4f}")
            print(f"  Disc Loss: {disc_loss.item():.4f}")
    
    return policy
```

### 37.1.3 持续学习

**Continual Learning**：在不断变化的环境中持续学习，避免灾难性遗忘。

**挑战**：
- **灾难性遗忘**（Catastrophic Forgetting）：学习新任务时遗忘旧任务
- **可塑性-稳定性困境**：新学习能力 vs 保持已学知识

**方法**：

1. **Elastic Weight Consolidation (EWC)**
2. **Progressive Neural Networks**
3. **Experience Replay**

**EWC实现**：

```python
"""
Elastic Weight Consolidation for Continual RL
"""

import torch
import torch.nn as nn
import copy

class EWC:
    """
    Elastic Weight Consolidation
    
    通过Fisher信息矩阵约束重要参数的变化
    """
    def __init__(self, model, dataset, lambda_ewc=1000):
        """
        Args:
            model: 神经网络模型
            dataset: 旧任务的数据集
            lambda_ewc: EWC正则化系数
        """
        self.model = model
        self.lambda_ewc = lambda_ewc
        
        # 保存旧任务的参数
        self.old_params = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }
        
        # 计算Fisher信息矩阵
        self.fisher = self._compute_fisher(dataset)
    
    def _compute_fisher(self, dataset):
        """
        计算Fisher信息矩阵（对角近似）
        
        F_ii ≈ E[(∂log p(y|x;θ) / ∂θ_i)²]
        """
        fisher = {}
        
        for name, param in self.model.named_parameters():
            fisher[name] = torch.zeros_like(param)
        
        self.model.eval()
        
        for data in dataset:
            self.model.zero_grad()
            
            # 计算对数似然
            output = self.model(data['state'])
            log_likelihood = torch.nn.functional.log_softmax(output, dim=-1)[
                range(len(data['action'])),
                data['action']
            ].mean()
            
            # 梯度
            log_likelihood.backward()
            
            # 累积平方梯度
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2
        
        # 平均
        n_samples = len(dataset)
        for name in fisher:
            fisher[name] /= n_samples
        
        return fisher
    
    def penalty(self):
        """
        EWC惩罚项
        
        L_EWC = λ/2 Σ_i F_ii (θ_i - θ*_i)²
        """
        loss = 0.0
        
        for name, param in self.model.named_parameters():
            fisher_weight = self.fisher[name]
            old_param = self.old_params[name]
            
            loss += (fisher_weight * (param - old_param) ** 2).sum()
        
        return (self.lambda_ewc / 2) * loss


# 使用EWC训练多个任务
def continual_learning_with_ewc(tasks, model):
    """
    使用EWC进行持续学习
    
    Args:
        tasks: 任务列表，每个任务是一个环境
        model: 策略网络
    """
    ewc = None
    
    for task_id, task_env in enumerate(tasks):
        print(f"\n学习任务 {task_id + 1}...")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        
        for epoch in range(100):
            # 采集数据
            trajectories = collect_trajectories(task_env, model, num_episodes=10)
            
            for batch in create_batches(trajectories):
                optimizer.zero_grad()
                
                # RL损失
                rl_loss = compute_policy_gradient_loss(model, batch)
                
                # EWC惩罚（如果不是第一个任务）
                if ewc is not None:
                    ewc_loss = ewc.penalty()
                    total_loss = rl_loss + ewc_loss
                else:
                    total_loss = rl_loss
                
                total_loss.backward()
                optimizer.step()
        
        # 任务完成后，计算Fisher矩阵用于下一个任务
        dataset = collect_dataset(task_env, model)
        ewc = EWC(model, dataset, lambda_ewc=1000)
        
        # 评估所有任务性能（检查遗忘）
        print(f"任务 {task_id + 1} 完成后的性能:")
        for tid, test_task in enumerate(tasks[:task_id + 1]):
            perf = evaluate(model, test_task)
            print(f"  任务 {tid + 1}: {perf:.2f}")
```

---

## 37.2 对抗鲁棒性

### 37.2.1 对抗攻击

**定义**：在输入状态上添加精心设计的小扰动，导致策略做出错误决策。

**攻击类型**：

1. **FGSM**（Fast Gradient Sign Method）
2. **PGD**（Projected Gradient Descent）
3. **C&W Attack**

**FGSM攻击**：

$$
s_{\text{adv}} = s + \epsilon \cdot \text{sign}(\nabla_s J(s, a^*))
$$

其中 $a^* = \pi(s)$ 是原始动作，$J$ 是策略目标。

**实现**：

```python
"""
对抗攻击on RL Policies
"""

import torch
import torch.nn as nn

class AdversarialAttacker:
    """
    对RL策略的对抗攻击
    """
    def __init__(self, policy):
        """
        Args:
            policy: 目标策略网络
        """
        self.policy = policy
    
    def fgsm_attack(self, state, target_action=None, epsilon=0.1):
        """
        Fast Gradient Sign Method攻击
        
        Args:
            state: 原始状态
            target_action: 目标动作（定向攻击）或None（非定向）
            epsilon: 扰动幅度
        
        Returns:
            adversarial_state: 对抗样本
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        state_tensor.requires_grad = True
        
        # 前向传播
        action_probs = self.policy(state_tensor)
        
        if target_action is None:
            # 非定向攻击：最大化原动作的损失
            original_action = action_probs.argmax().item()
            loss = -torch.log(action_probs[0, original_action] + 1e-8)
        else:
            # 定向攻击：最大化目标动作的概率
            loss = -torch.log(action_probs[0, target_action] + 1e-8)
        
        # 反向传播
        loss.backward()
        
        # 计算扰动
        perturbation = epsilon * state_tensor.grad.sign()
        
        # 生成对抗样本
        adversarial_state = state_tensor + perturbation
        
        # 裁剪到合法范围
        adversarial_state = torch.clamp(adversarial_state, state_tensor.min(), state_tensor.max())
        
        return adversarial_state.detach().numpy()[0]
    
    def pgd_attack(
        self,
        state,
        target_action=None,
        epsilon=0.1,
        alpha=0.01,
        num_iter=40
    ):
        """
        Projected Gradient Descent攻击
        
        更强的迭代攻击
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        adv_state = state_tensor.clone()
        
        for iteration in range(num_iter):
            adv_state.requires_grad = True
            
            # 前向传播
            action_probs = self.policy(adv_state)
            
            if target_action is None:
                original_action = self.policy(state_tensor).argmax().item()
                loss = -torch.log(action_probs[0, original_action] + 1e-8)
            else:
                loss = -torch.log(action_probs[0, target_action] + 1e-8)
            
            # 梯度
            loss.backward()
            
            # 更新
            with torch.no_grad():
                adv_state = adv_state + alpha * adv_state.grad.sign()
                
                # 投影到epsilon球
                perturbation = torch.clamp(
                    adv_state - state_tensor,
                    -epsilon,
                    epsilon
                )
                adv_state = state_tensor + perturbation
                
                # 裁剪到合法范围
                adv_state = torch.clamp(adv_state, state_tensor.min(), state_tensor.max())
        
        return adv_state.detach().numpy()[0]
    
    def evaluate_robustness(self, env, num_episodes=100, epsilon=0.1):
        """
        评估策略的对抗鲁棒性
        
        Returns:
            clean_performance: 干净状态下的性能
            adversarial_performance: 对抗状态下的性能
        """
        clean_rewards = []
        adversarial_rewards = []
        
        for episode in range(num_episodes):
            # Clean episode
            state = env.reset()
            done = False
            clean_reward = 0
            
            while not done:
                action_probs = self.policy(torch.FloatTensor(state).unsqueeze(0))
                action = action_probs.argmax().item()
                state, reward, done, _ = env.step(action)
                clean_reward += reward
            
            clean_rewards.append(clean_reward)
            
            # Adversarial episode
            state = env.reset()
            done = False
            adv_reward = 0
            
            while not done:
                # 生成对抗状态
                adv_state = self.fgsm_attack(state, epsilon=epsilon)
                
                # 在对抗状态下选择动作
                action_probs = self.policy(torch.FloatTensor(adv_state).unsqueeze(0))
                action = action_probs.argmax().item()
                
                # 但在真实状态下执行
                state, reward, done, _ = env.step(action)
                adv_reward += reward
            
            adversarial_rewards.append(adv_reward)
        
        print(f"\n鲁棒性评估 (ε={epsilon}):")
        print(f"  干净性能: {np.mean(clean_rewards):.2f} ± {np.std(clean_rewards):.2f}")
        print(f"  对抗性能: {np.mean(adversarial_rewards):.2f} ± {np.std(adversarial_rewards):.2f}")
        print(f"  性能下降: {(np.mean(clean_rewards) - np.mean(adversarial_rewards)):.2f}")
        
        return np.mean(clean_rewards), np.mean(adversarial_rewards)


# 使用示例
policy = load_pretrained_policy()
attacker = AdversarialAttacker(policy)

# 单个状态攻击
state = env.reset()
adv_state = attacker.fgsm_attack(state, epsilon=0.1)

# 评估鲁棒性
clean_perf, adv_perf = attacker.evaluate_robustness(env, epsilon=0.1)
```

<div data-component="AdversarialAttackDemo"></div>

继续...

### 37.2.2 防御机制

**对抗训练**（Adversarial Training）：

$$
\min_\theta \mathbb{E}_{s \sim \mathcal{D}} \left[ \max_{\|\delta\| \leq \epsilon} L(\pi_\theta(s + \delta), a^*) \right]
$$

**实现**：

```python
"""
对抗训练提高策略鲁棒性
"""

class AdversarialTraining:
    """
    对抗训练框架
    """
    def __init__(self, policy, epsilon=0.1, attack_steps=10):
        self.policy = policy
        self.epsilon = epsilon
        self.attack_steps = attack_steps
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    
    def train_step(self, states, actions, advantages):
        """
        对抗训练步骤
        
        对每个batch，生成对抗样本并同时训练
        """
        # 1. 正常训练损失
        clean_loss = self._compute_loss(states, actions, advantages)
        
        # 2. 生成对抗样本
        adv_states = self._generate_adversarial(states, actions)
        
        # 3. 对抗样本上的损失
        adv_loss = self._compute_loss(adv_states, actions, advantages)
        
        #4. 总损失（平衡clean和adversarial）
        total_loss = 0.5 * clean_loss + 0.5 * adv_loss
        
        # 5. 优化
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def _generate_adversarial(self, states, actions):
        """
        生成对抗样本（PGD）
        """
        states_tensor = torch.FloatTensor(states)
        adv_states = states_tensor.clone()
        
        for _ in range(self.attack_steps):
            adv_states.requires_grad = True
            
            # 计算策略概率
            action_probs = self.policy(adv_states)
            
            # 损失：降低正确动作的概率
            loss = -torch.log(
                action_probs[range(len(actions)), actions] + 1e-8
            ).mean()
            
            # 梯度
            loss.backward()
            
            # PGD更新
            with torch.no_grad():
                adv_states = adv_states + (self.epsilon / self.attack_steps) * adv_states.grad.sign()
                
                # 投影
                perturbation = torch.clamp(
                    adv_states - states_tensor,
                    -self.epsilon,
                    self.epsilon
                )
                adv_states = states_tensor + perturbation
        
        return adv_states.detach()
    
    def _compute_loss(self, states, actions, advantages):
        """计算策略梯度损失"""
        action_probs = self.policy(states)
        log_probs = torch.log(
            action_probs[range(len(actions)), actions] + 1e-8
        )
        loss = -(log_probs * advantages).mean()
        return loss
```

**其他防御方法**：

1. **Certified Defense**：可证明的鲁棒性界
2. **Randomized Smoothing**
3. **Input Transformation**

---

## 37.3 不确定性量化

### 37.3.1 认知不确定性 vs 偶然不确定性

**两类不确定性**：

1. **Epistemic Uncertainty**（认知不确定性）：
   - 模型参数不确定性
   - 训练数据不足导致
   - 可以通过更多数据减少

2. **Aleatoric Uncertainty**（偶然不确定性）：
   - 环境随机性
   - 观测噪声
   - 无法减少

$$
\text{Total Uncertainty} = \underbrace{\text{Epistemic}}_{\text{model}} + \underbrace{\text{Aleatoric}}_{\text{environment}}
$$

**贝叶斯神经网络**：

```python
"""
贝叶斯神经网络for不确定性量化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianLinear(nn.Module):
    """
    贝叶斯线性层
    
    每个权重是一个分布，而不是单一值
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        
        # 权重均值和标准差
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # 偏置均值和标准差
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        # 初始化
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -5)  # log(std)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_rho, -5)
    
    def forward(self, x, sample=True):
        """
        前向传播
        
        Args:
            sample: 是否采样权重（训练时True，测试时可多次采样）
        """
        if sample:
            # 采样权重 w ~ N(μ, σ²)
            weight_std = torch.log1p(torch.exp(self.weight_rho))
            weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
            
            bias_std = torch.log1p(torch.exp(self.bias_rho))
            bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
        else:
            # 使用均值
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self):
        """
        KL散度：q(w|θ) || p(w)
        
        用于ELBO损失
        """
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        bias_std = torch.log1p(torch.exp(self.bias_rho))
        
        # KL(q||p) for Gaussian
        kl_weight = (
            torch.log(1 / weight_std) +
            (weight_std**2 + self.weight_mu**2) / 2 - 0.5
        ).sum()
        
        kl_bias = (
            torch.log(1 / bias_std) +
            (bias_std**2 + self.bias_mu**2) / 2 - 0.5
        ).sum()
        
        return kl_weight + kl_bias


class BayesianPolicy(nn.Module):
    """
    贝叶斯策略网络
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        self.fc1 = BayesianLinear(state_dim, 128)
        self.fc2 = BayesianLinear(128, 64)
        self.fc3 = BayesianLinear(64, action_dim)
    
    def forward(self, state, sample=True):
        x = F.relu(self.fc1(state, sample))
        x = F.relu(self.fc2(x, sample))
        action_probs = F.softmax(self.fc3(x, sample), dim=-1)
        return action_probs
    
    def predict_with_uncertainty(self, state, num_samples=100):
        """
        预测with不确定性估计
        
        Returns:
            mean_probs: 平均动作概率
            epistemic_uncertainty: 认知不确定性（预测方差）
        """
        self.eval()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # 每次采样不同的网络权重
                probs = self.forward(state, sample=True)
                predictions.append(probs)
        
        predictions = torch.stack(predictions)  # (num_samples, batch, action_dim)
        
        # 均值预测
        mean_probs = predictions.mean(dim=0)
        
        # 方差（认知不确定性）
        epistemic_uncertainty = predictions.var(dim=0)
        
        return mean_probs, epistemic_uncertainty
    
    def kl_divergence(self):
        """总KL散度"""
        return self.fc1.kl_divergence() + self.fc2.kl_divergence() + self.fc3.kl_divergence()


def train_bayesian_policy(env, num_episodes=1000):
    """
    训练贝叶斯策略
    
    使用变分推断（Bayes by Backprop）
    """
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy = BayesianPolicy(state_dim, action_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_rewards = []
        episode_log_probs = []
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # 采样动作
            action_probs = policy(state_tensor, sample=True)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            # 环境交互
            state, reward, done, _ = env.step(action.item())
            
            episode_rewards.append(reward)
            episode_log_probs.append(log_prob)
        
        # ELBO损失
        # ELBO = E[log p(D|w)] - KL[q(w)||p(w)]
        
        # 1. Likelihood（策略梯度部分）
        returns = compute_returns(episode_rewards)
        policy_loss = 0
        for log_prob, G in zip(episode_log_probs, returns):
            policy_loss -= log_prob * G
        
        # 2. KL散度（正则化）
        kl_loss = policy.kl_divergence() / len(episode_log_probs)
        
        # 3. 总ELBO损失
        loss = policy_loss + 0.01 * kl_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {sum(episode_rewards):.2f}")
    
    return policy


# 使用不确定性进行决策
def uncertainty_aware_decision(bayesian_policy, state, uncertainty_threshold=0.1):
    """
    考虑不确定性的决策
    
    如果不确定性高，采取保守动作
    """
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    
    mean_probs, uncertainty = bayesian_policy.predict_with_uncertainty(state_tensor, num_samples=50)
    
    # 最大不确定性
    max_uncertainty = uncertainty.max().item()
    
    if max_uncertainty > uncertainty_threshold:
        # 高不确定性：保守策略（如随机探索或安全动作）
        print(f"高不确定性 ({max_uncertainty:.4f}) - 采取保守动作")
        action = safe_action()
    else:
        # 低不确定性：使用均值预测
        action = mean_probs.argmax().item()
    
    return action, max_uncertainty
```

<div data-component="UncertaintyQuantification"></div>

### 37.3.2 Ensemble方法

**Bootstrap Ensemble**：

```python
"""
Ensemble方法for不确定性估计
"""

class EnsemblePolicy:
    """
    策略集成
    
    训练多个独立策略并聚合
    """
    def __init__(self, state_dim, action_dim, num_models=5):
        self.models = [
            create_policy(state_dim, action_dim)
            for _ in range(num_models)
        ]
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=3e-4)
            for model in self.models
        ]
    
    def train(self, env, num_episodes=1000):
        """
        训练每个模型（用不同的随机种子/bootstrap样本）
        """
        for model_id, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            print(f"\n训练模型 {model_id + 1}...")
            
            # 设置不同随机种子
            torch.manual_seed(model_id * 1000)
            np.random.seed(model_id * 1000)
            
            for episode in range(num_episodes // len(self.models)):
                # 正常RL训练
                train_one_episode(env, model, optimizer)
    
    def predict_with_disagreement(self, state):
        """
        预测with模型分歧度
        
        Returns:
            mean_probs: 平均预测
            disagreement: 模型间分歧（不确定性代理）
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                probs = model(state_tensor)
                predictions.append(probs)
        
        predictions = torch.stack(predictions)  # (num_models, batch, action_dim)
        
        # 均值
        mean_probs = predictions.mean(dim=0)
        
        # 分歧度（方差或熵）
        disagreement = predictions.var(dim=0).mean().item()
        
        return mean_probs, disagreement
    
    def mutual_information(self, state):
        """
        互信息：I(Y; θ | X)
        
        测量认知不确定性
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                probs = model(state_tensor)[0]  # (action_dim,)
                predictions.append(probs.numpy())
        
        predictions = np.array(predictions)  # (num_models, action_dim)
        
        # E[H[Y|θ,X]]：期望熵
        expected_entropy = np.mean([
            -np.sum(p * np.log(p + 1e-8))
            for p in predictions
        ])
        
        # H[E[Y|X]]：均值的熵
        mean_pred = predictions.mean(axis=0)
        entropy_of_mean = -np.sum(mean_pred * np.log(mean_pred + 1e-8))
        
        # 互信息 = H[E[Y|X]] - E[H[Y|θ,X]]
        mi = entropy_of_mean - expected_entropy
        
        return mi
```

---

## 37.4 Out-of-Distribution检测

### 37.4.1 OOD状态识别

**目标**：识别训练分布之外的状态，避免做出不可靠的预测。

**方法**：

1. **基于重构误差**（VAE）
2. **基于密度**（Normalizing Flows）
3. **基于置信度**
4. **基于Mahalanobis距离**

**Mahalanobis距离方法**：

```python
"""
OOD检测using Mahalanobis Distance
"""

class MahalanobisOODDetector:
    """
    基于Mahalanobis距离的OOD检测
    
    Lee et al., 2018: A Simple Unified Framework for Detecting OOD
    """
    def __init__(self, model, layer_name='penultimate'):
        self.model = model
        self.layer_name = layer_name
        
        # 训练数据的统计量
        self.class_means = None
        self.shared_covariance = None
    
    def fit(self, train_data):
        """
        计算训练数据的统计量
        
        Args:
            train_data: (states, labels)
        """
        states, labels = train_data
        
        # 提取特征
        features = self._extract_features(states)
        
        # 按类别计算均值
        unique_labels = np.unique(labels)
        self.class_means = {}
        
        for label in unique_labels:
            class_features = features[labels == label]
            self.class_means[label] = class_features.mean(axis=0)
        
        # 计算共享协方差矩阵
        centered_features = []
        for label in unique_labels:
            class_features = features[labels == label]
            centered = class_features - self.class_means[label]
            centered_features.append(centered)
        
        all_centered = np.vstack(centered_features)
        self.shared_covariance = np.cov(all_centered.T) + 1e-6 * np.eye(all_centered.shape[1])
    
    def _extract_features(self, states):
        """提取特征（倒数第二层）"""
        self.model.eval()
        
        features = []
        
        with torch.no_grad():
            for state in states:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                # Hook to extract features
                activation = {}
                def get_activation(name):
                    def hook(model, input, output):
                        activation[name] = output.detach()
                    return hook
                
                # Register hook
                handle = self.model.fc2.register_forward_hook(get_activation('features'))
                
                _ = self.model(state_tensor)
                features.append(activation['features'].numpy()[0])
                
                handle.remove()
        
        return np.array(features)
    
    def mahalanobis_distance(self, state):
        """
        计算Mahalanobis距离
        
        M(x) = min_c √((f(x) - μ_c)^T Σ^{-1} (f(x) - μ_c))
        """
        # 提取特征
        feature = self._extract_features([state])[0]
        
        # 计算到每个类别均值的距离
        distances = []
        
        precision = np.linalg.inv(self.shared_covariance)
        
        for class_mean in self.class_means.values():
            diff = feature - class_mean
            dist = np.sqrt(diff.T @ precision @ diff)
            distances.append(dist)
        
        # 最小距离
        min_distance = min(distances)
        
        return min_distance
    
    def is_ood(self, state, threshold):
        """
        判断是否OOD
        
        Args:
            threshold: Mahalanobis距离阈值
        """
        distance = self.mahalanobis_distance(state)
        
        return distance > threshold, distance


# 使用示例
detector = MahalanobisOODDetector(policy)

# 在训练数据上拟合
train_states, train_actions = collect_training_data()
detector.fit((train_states, train_actions))

# 设置阈值（基于验证集）
threshold = calibrate_threshold(detector, validation_set)

# 部署时检测
def safe_policy(state):
    is_ood, distance = detector.is_ood(state, threshold)
    
    if is_ood:
        print(f"OOD状态检测! 距离={distance:.4f}")
        # 安全回退策略
        return fallback_action()
    else:
        # 正常策略
        return policy(state).argmax()
```

### 37.4.2 置信度校准

**问题**：神经网络往往过度自信。

**解决**：Temperature Scaling、Platt Scaling

```python
"""
置信度校准
"""

class TemperatureScaling:
    """
    Temperature Scaling for Calibration
    
    Guo et al., 2017: On Calibration of Modern Neural Networks
    """
    def __init__(self, model):
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, state):
        """
        调整后的概率
        
        p_calibrated = softmax(logits / T)
        """
        logits = self.model.get_logits(state)
        scaled_logits = logits / self.temperature
        calibrated_probs = F.softmax(scaled_logits, dim=-1)
        
        return calibrated_probs
    
    def calibrate(self, val_data):
        """
        在验证集上优化temperature
        
        最小化负对数似然
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval():
            optimizer.zero_grad()
            loss = 0.0
            
            for state, true_action in val_data:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                calibrated_probs = self.forward(state_tensor)
                loss += F.nll_loss(
                    torch.log(calibrated_probs + 1e-8),
                    torch.LongTensor([true_action])
                )
            
            loss.backward()
            return loss
        
        optimizer.step(eval)
        
        print(f"最优温度: T = {self.temperature.item():.4f}")
        
        return self.temperature.item()
```

---

## 37.5 可解释性

### 37.5.1 策略可视化

**显著性图**（Saliency Maps）：

```python
"""
策略可解释性工具
"""

class PolicyExplainer:
    """
    策略可解释性分析
    """
    def __init__(self, policy):
        self.policy = policy
    
    def saliency_map(self, state):
        """
        显著性图：哪些输入特征对决策最重要
        
        计算 ∂π(a|s) / ∂s
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        state_tensor.requires_grad = True
        
        # 前向传播
        action_probs = self.policy(state_tensor)
        selected_action = action_probs.argmax()
        
        # 反向传播
        self.policy.zero_grad()
        action_probs[0, selected_action].backward()
        
        # 梯度即为显著性
        saliency = state_tensor.grad.abs()[0].numpy()
        
        return saliency, selected_action.item()
    
    def integrated_gradients(self, state, baseline=None, steps=50):
        """
        Integrated Gradients（更稳定的归因方法）
        
        IG = (x - x') ∫_0^1 ∂F(x' + α(x - x')) / ∂x dα
        """
        if baseline is None:
            baseline = np.zeros_like(state)
        
        state_tensor = torch.FloatTensor(state)
        baseline_tensor = torch.FloatTensor(baseline)
        
        # 路径积分
        integrated_grads = torch.zeros_like(state_tensor)
        
        for alpha in np.linspace(0, 1, steps):
            # 插值
            interpolated = baseline_tensor + alpha * (state_tensor - baseline_tensor)
            interpolated.requires_grad = True
            
            # 梯度
            action_probs = self.policy(interpolated.unsqueeze(0))
            selected_action = action_probs.argmax()
            
            self.policy.zero_grad()
            action_probs[0, selected_action].backward()
            
            integrated_grads += interpolated.grad
        
        # 平均并乘以路径
        integrated_grads = integrated_grads * (state_tensor - baseline_tensor) / steps
        
        return integrated_grads.abs().numpy()
    
    def visualize_attention(self, state, attention_layer):
        """
        可视化注意力权重（如果策略使用Attention）
        """
        # 假设模型有attention机制
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            _, attention_weights = self.policy.forward_with_attention(state_tensor)
        
        return attention_weights.numpy()


# 可视化
def visualize_saliency(explainer, state, state_names=None):
    """
    可视化显著性图
    """
    saliency, action = explainer.saliency_map(state)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(state)), state, alpha=0.5, label='State')
    if state_names:
        plt.xticks(range(len(state)), state_names, rotation=45)
    plt.ylabel('Value')
    plt.title('State')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(saliency)), saliency, color='red', alpha=0.7)
    if state_names:
        plt.xticks(range(len(saliency)), state_names, rotation=45)
    plt.ylabel('Importance')
    plt.title(f'Saliency Map (Action={action})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('saliency_map.png', dpi=300, bbox_inches='tight')
    plt.show()
```

<div data-component="PolicyExplainability"></div>

---

## 37.6 故障诊断

### 37.6.1 训练不稳定

**常见问题**：

1. **梯度爆炸/消失**
2. **奖励尺度问题**
3. **探索不足**

**诊断工具**：

```python
"""
RL训练诊断工具
"""

class RLDiagnostics:
    """
    RL训练诊断
    """
    def __init__(self):
        self.metrics = {
            'gradients': [],
            'rewards': [],
            'value_estimates': [],
            'entropy': []
        }
    
    def log_gradients(self, model):
        """记录梯度统计"""
        total_norm = 0.0
        max_grad = 0.0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                max_grad = max(max_grad, param.grad.abs().max().item())
        
        total_norm = total_norm ** 0.5
        
        self.metrics['gradients'].append({
            'total_norm': total_norm,
            'max_grad': max_grad
        })
        
        # 警告
        if total_norm > 10.0:
            print(f"⚠️ 梯度爆炸! Norm={total_norm:.2f}")
        elif total_norm < 1e-6:
            print(f"⚠️ 梯度消失! Norm={total_norm:.2e}")
    
    def analyze_rewards(self, rewards):
        """分析奖励分布"""
        self.metrics['rewards'].append(rewards)
        
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        # 检查奖励尺度
        if std_reward > 100:
            print(f"⚠️ 奖励方差过大! std={std_reward:.2f}, 考虑标准化")
        
        # 检查稀疏奖励
        if np.sum(np.array(rewards) != 0) / len(rewards) < 0.1:
            print("⚠️ 检测到稀疏奖励，考虑Reward Shaping或Hindsight ER")
    
    def check_exploration(self, policy, states):
        """检查探索充分性"""
        entropies = []
        
        for state in states:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = policy(state_tensor)[0]
            entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum().item()
            entropies.append(entropy)
        
        avg_entropy = np.mean(entropies)
        self.metrics['entropy'].append(avg_entropy)
        
        # 警告
        if avg_entropy < 0.1:
            print(f"⚠️ 熵过低! H={avg_entropy:.4f}, 策略可能过早收敛")
    
    def generate_report(self):
        """生成诊断报告"""
        print("\n" + "="*60)
        print("RL训练诊断报告")
        print("="*60)
        
        # 梯度
        if self.metrics['gradients']:
            recent_grads = self.metrics['gradients'][-100:]
            avg_norm = np.mean([g['total_norm'] for g in recent_grads])
            print(f"\n梯度统计 (最近100步):")
            print(f"  平均梯度范数: {avg_norm:.4f}")
            print(f"  最大梯度: {max(g['max_grad'] for g in recent_grads):.4f}")
        
        # 奖励
        if self.metrics['rewards']:
            recent_rewards = self.metrics['rewards'][-100:]
            flat_rewards = [r for ep in recent_rewards for r in ep]
            print(f"\n奖励统计:")
            print(f"  均值: {np.mean(flat_rewards):.2f}")
            print(f"  标准差: {np.std(flat_rewards):.2f}")
            print(f"  非零比例: {np.sum(np.array(flat_rewards) != 0) / len(flat_rewards) * 100:.1f}%")
        
        # 熵
        if self.metrics['entropy']:
            recent_entropy = self.metrics['entropy'][-100:]
            print(f"\n策略熵:")
            print(f"  当前: {recent_entropy[-1]:.4f}")
            print(f"  趋势: {'下降' if recent_entropy[-1] < recent_entropy[0] else '上升'}")
        
        print("="*60)
```

---

## 总结

本章介绍了RL的可靠性与鲁棒性：

1. **分布漂移**：KS检验、Wasserstein距离、MMD、域随机化、域适应、持续学习（EWC）
2. **对抗鲁棒性**：FGSM/PGD攻击、对抗训练、certified defense
3. **不确定性量化**：认知vs偶然不确定性、贝叶斯RL、Ensemble方法
4. **OOD检测**：Mahalanobis距离、置信度校准、安全回退
5. **可解释性**：显著性图、Integrated Gradients、注意力可视化
6. **故障诊断**：梯度监控、奖励分析、探索检查

**关键要点**：
- 真实部署需要考虑分布偏移
- 对抗攻击可以显著降低性能，需要防御
- 不确定性量化对安全决策至关重要
- OOD检测避免在未知状态下做出错误决策
- 可解释性提升信任度和调试能力

---

## 参考文献

- Pinto, L., et al. (2017). "Robust Adversarial Reinforcement Learning." *ICML*.
- Kahn, G., et al. (2017). "Uncertainty-Aware Reinforcement Learning for Collision Avoidance." *arXiv*.
- Lee, K., et al. (2018). "A Simple Unified Framework for Detecting Out-of-Distribution Samples." *NeurIPS*.
- Kirkpatrick, J., et al. (2017). "Overcoming Catastrophic Forgetting in Neural Networks." *PNAS*.
- Dulac-Arnold, G., et al. (2019). "Challenges of Real-World Reinforcement Learning." *ICML Workshop*.
- Guo, C., et al. (2017). "On Calibration of Modern Neural Networks." *ICML*.

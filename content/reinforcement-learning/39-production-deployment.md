---
title: "第39章：生产部署与工程实践"
description: "模型部署、在线学习、监控日志、数据管理、工程工具链、实际案例"
date: "2026-01-30"
---

# 第39章：生产部署与工程实践

## 39.1 模型部署

### 39.1.1 模型导出

**ONNX（Open Neural Network Exchange）**：

```python
"""
导出PyTorch模型到ONNX
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

class PolicyNetwork(nn.Module):
    """策略网络"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.network(state)


# 训练好的模型
state_dim = 4
action_dim = 2
model = PolicyNetwork(state_dim, action_dim)
model.load_state_dict(torch.load('policy_weights.pth'))
model.eval()

# 导出到ONNX
dummy_input = torch.randn(1, state_dim)

torch.onnx.export(
    model,
    dummy_input,
    "policy.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['state'],
    output_names=['action_probs'],
    dynamic_axes={
        'state': {0: 'batch_size'},
        'action_probs': {0: 'batch_size'}
    }
)

print("✅ 模型已导出到 policy.onnx")

# 验证ONNX模型
onnx_model = onnx.load("policy.onnx")
onnx.checker.check_model(onnx_model)
print("✅ ONNX模型验证通过")

# 使用ONNX Runtime推理
ort_session = ort.InferenceSession("policy.onnx")

def predict_onnx(state):
    """使用ONNX模型推理"""
    ort_inputs = {ort_session.get_inputs()[0].name: state.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    return ort_outputs[0]


# 对比PyTorch和ONNX输出
test_state = torch.randn(1, state_dim)

with torch.no_grad():
    pytorch_output = model(test_state).numpy()

onnx_output = predict_onnx(test_state)

print(f"\nPyTorch输出: {pytorch_output}")
print(f"ONNX输出: {onnx_output}")
print(f"差异: {abs(pytorch_output - onnx_output).max():.6f}")
```

**TorchScript**：

```python
"""
导出PyTorch模型到TorchScript
"""

import torch

# 方法1: Tracing（追踪）
model.eval()
example_input = torch.randn(1, state_dim)

traced_model = torch.jit.trace(model, example_input)
traced_model.save("policy_traced.pt")

print("✅ Traced模型已保存")

# 方法2: Scripting（脚本化）- 支持控制流
class PolicyNetworkWithControl(nn.Module):
    """带控制流的策略网络"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        
        # 控制流（Tracing无法处理）
        if x.sum() > 0:
            x = torch.relu(self.fc2(x))
        else:
            x = torch.tanh(self.fc2(x))
        
        return torch.softmax(self.fc3(x), dim=-1)


scripted_model = torch.jit.script(PolicyNetworkWithControl(state_dim, action_dim))
scripted_model.save("policy_scripted.pt")

print("✅ Scripted模型已保存")

# 加载并使用
loaded_model = torch.jit.load("policy_traced.pt")
loaded_model.eval()

with torch.no_grad():
    output = loaded_model(test_state)
    print(f"TorchScript输出: {output}")
```

### 39.1.2 量化与压缩

**动态量化**（推理时量化）：

```python
"""
模型量化 - 减小模型大小，加速推理
"""

import torch.quantization

# 原始模型大小
original_size = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"原始模型大小: {original_size / 1024:.2f} KB")

# 动态量化（INT8）
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},  # 量化Linear层
    dtype=torch.qint8
)

# 保存量化模型
torch.save(quantized_model.state_dict(), 'policy_quantized.pth')

# 量化后大小
quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
print(f"量化后大小: {quantized_size / 1024:.2f} KB")
print(f"压缩比: {original_size / quantized_size:.2f}x")

# 推理速度对比
import time

test_batch = torch.randn(100, state_dim)

# 原始模型
start = time.time()
with torch.no_grad():
    for _ in range(1000):
        _ = model(test_batch)
original_time = time.time() - start

# 量化模型
start = time.time()
with torch.no_grad():
    for _ in range(1000):
        _ = quantized_model(test_batch)
quantized_time = time.time() - start

print(f"\n原始模型推理时间: {original_time:.3f}s")
print(f"量化模型推理时间: {quantized_time:.3f}s")
print(f"加速比: {original_time / quantized_time:.2f}x")
```

**模型剪枝**：

```python
"""
模型剪枝 - 移除不重要的权重
"""

import torch.nn.utils.prune as prune

def prune_model(model, amount=0.3):
    """
    剪枝模型
    
    Args:
        amount: 剪枝比例（0.3 = 移除30%的权重）
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # L1 unstructured pruning
            prune.l1_unstructured(module, name='weight', amount=amount)
            
            # 永久化剪枝
            prune.remove(module, 'weight')
    
    return model


# 剪枝
pruned_model = prune_model(model, amount=0.3)

# 统计稀疏度
def count_sparsity(model):
    """计算模型稀疏度"""
    total_params = 0
    zero_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()
    
    sparsity = zero_params / total_params * 100
    return sparsity


sparsity = count_sparsity(pruned_model)
print(f"模型稀疏度: {sparsity:.2f}%")
```

### 39.1.3 推理优化

**批处理推理**：

```python
"""
批处理推理优化
"""

class BatchedInference:
    """批处理推理引擎"""
    def __init__(self, model, batch_size=32, timeout=0.01):
        """
        Args:
            batch_size: 批大小
            timeout: 等待超时（秒）
        """
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        
        self.queue = []
        self.results = {}
        
        import threading
        self.lock = threading.Lock()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def predict(self, state, request_id):
        """
        异步预测
        
        Args:
            state: 输入状态
            request_id: 请求ID
        """
        with self.lock:
            self.queue.append((request_id, state))
        
        # 等待结果
        while request_id not in self.results:
            time.sleep(0.001)
        
        result = self.results.pop(request_id)
        return result
    
    def _worker(self):
        """后台worker - 批处理推理"""
        import time
        
        while True:
            time.sleep(self.timeout)
            
            with self.lock:
                if len(self.queue) == 0:
                    continue
                
                # 取出一批
                batch = self.queue[:self.batch_size]
                self.queue = self.queue[self.batch_size:]
            
            # 批处理推理
            request_ids = [req_id for req_id, _ in batch]
            states = torch.stack([state for _, state in batch])
            
            with torch.no_grad():
                outputs = self.model(states)
            
            # 存储结果
            with self.lock:
                for req_id, output in zip(request_ids, outputs):
                    self.results[req_id] = output


# 使用
batched_engine = BatchedInference(model, batch_size=32)

# 模拟多个并发请求
import uuid

for i in range(100):
    state = torch.randn(state_dim)
    req_id = str(uuid.uuid4())
    result = batched_engine.predict(state, req_id)
    print(f"Request {i}: {result}")
```

### 39.1.4 边缘设备部署

**TensorFlow Lite转换**（用于移动端）：

```python
"""
转换为TensorFlow Lite（移动端部署）
"""

# 假设已有TensorFlow模型
import tensorflow as tf

# 转换为TFLite
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_dir')

# 优化
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 量化
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# 保存
with open('policy.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ TFLite模型已保存")

# 使用TFLite推理
interpreter = tf.lite.Interpreter(model_path="policy.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 推理
input_data = np.random.randn(1, state_dim).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(f"TFLite输出: {output_data}")
```

<div data-component="DeploymentPipeline"></div>

---

## 39.2 在线学习系统

### 39.2.1 持续训练

**在线学习架构**：

```python
"""
在线学习系统
"""

import redis
import json
from collections import deque

class OnlineLearningSystem:
    """
    在线学习系统
    
    持续从生产环境收集数据并更新模型
    """
    def __init__(self, model, redis_host='localhost', redis_port=6379):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        
        # Redis连接（用于数据队列）
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        
        # 经验缓冲
        self.buffer = deque(maxlen=10000)
        
        # 统计
        self.total_updates = 0
        self.performance_history = []
    
    def collect_experience(self, state, action, reward, next_state, done):
        """
        收集生产环境经验
        
        Args:
            state, action, reward, next_state, done: SARS'元组
        """
        experience = {
            'state': state.tolist(),
            'action': action,
            'reward': reward,
            'next_state': next_state.tolist(),
            'done': done
        }
        
        # 推送到Redis队列
        self.redis_client.rpush('experience_queue', json.dumps(experience))
        
        # 本地缓冲
        self.buffer.append(experience)
    
    def update_model(self, batch_size=64):
        """
        从缓冲区采样并更新模型
        """
        if len(self.buffer) < batch_size:
            return
        
        # 采样batch
        import random
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([exp['state'] for exp in batch])
        actions = torch.LongTensor([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        next_states = torch.FloatTensor([exp['next_state'] for exp in batch])
        dones = torch.FloatTensor([exp['done'] for exp in batch])
        
        # 计算损失（示例：简单策略梯度）
        action_probs = self.model(states)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)) + 1e-8)
        
        # 简化的损失
        loss = -(log_probs * rewards.unsqueeze(1)).mean()
        
        # 更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.total_updates += 1
        
        # 记录
        if self.total_updates % 100 == 0:
            avg_reward = rewards.mean().item()
            self.performance_history.append(avg_reward)
            print(f"Update {self.total_updates}: Avg Reward = {avg_reward:.2f}")
    
    def save_checkpoint(self, path):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_updates': self.total_updates,
            'performance_history': self.performance_history
        }
        torch.save(checkpoint, path)
        print(f"✅ 检查点已保存到 {path}")
    
    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_updates = checkpoint['total_updates']
        self.performance_history = checkpoint['performance_history']
        print(f"✅ 检查点已从 {path} 加载")


# 使用示例
online_system = OnlineLearningSystem(model)

# 模拟生产环境数据流
for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        # 使用当前模型选择动作
        with torch.no_grad():
            action_probs = model(torch.FloatTensor(state).unsqueeze(0))
            action = torch.multinomial(action_probs, 1).item()
        
        # 执行动作
        next_state, reward, done, _ = env.step(action)
        
        # 收集经验
        online_system.collect_experience(
            torch.FloatTensor(state),
            action,
            reward,
            torch.FloatTensor(next_state),
            done
        )
        
        # 定期更新
        if len(online_system.buffer) >= 64:
            online_system.update_model(batch_size=64)
        
        state = next_state
    
    # 定期保存
    if episode % 100 == 0:
        online_system.save_checkpoint(f'checkpoint_ep{episode}.pth')
```

<div data-component="OnlineLearningArchitecture"></div>

### 39.2.2 A/B测试

**A/B测试框架**：

```python
"""
A/B测试框架
"""

import numpy as np
from scipy import stats

class ABTest:
    """
    A/B测试管理器
    """
    def __init__(self, model_a, model_b, traffic_split=0.5):
        """
        Args:
            model_a: 模型A（对照组）
            model_b: 模型B（实验组）
            traffic_split: 流量分配比例（0.5 = 50/50）
        """
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split
        
        # 统计
        self.results_a = []
        self.results_b = []
    
    def select_model(self, user_id):
        """
        根据user_id分配模型（确保同一用户始终看到同一模型）
        """
        # 哈希user_id
        import hashlib
        hash_value = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
        
        # 分配
        if (hash_value % 100) < (self.traffic_split * 100):
            return 'A', self.model_a
        else:
            return 'B', self.model_b
    
    def record_result(self, variant, reward):
        """记录结果"""
        if variant == 'A':
            self.results_a.append(reward)
        else:
            self.results_b.append(reward)
    
    def analyze(self, confidence=0.95):
        """
        分析A/B测试结果
        
        Returns:
            dict: 分析结果
        """
        if len(self.results_a) < 30 or len(self.results_b) < 30:
            return {"error": "样本量不足（需要至少30个）"}
        
        # 均值和标准差
        mean_a = np.mean(self.results_a)
        mean_b = np.mean(self.results_b)
        std_a = np.std(self.results_a)
        std_b = np.std(self.results_b)
        
        # t-检验
        t_stat, p_value = stats.ttest_ind(self.results_a, self.results_b)
        
        # 效应量（Cohen's d）
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
        cohens_d = (mean_b - mean_a) / pooled_std
        
        # 置信区间
        se_diff = np.sqrt(std_a**2/len(self.results_a) + std_b**2/len(self.results_b))
        t_crit = stats.t.ppf((1 + confidence) / 2, len(self.results_a) + len(self.results_b) - 2)
        ci_lower = (mean_b - mean_a) - t_crit * se_diff
        ci_upper = (mean_b - mean_a) + t_crit * se_diff
        
        # 判断
        is_significant = p_value < (1 - confidence)
        winner = 'B' if mean_b > mean_a and is_significant else 'A' if mean_a > mean_b and is_significant else 'No clear winner'
        
        result = {
            'model_a': {
                'mean': mean_a,
                'std': std_a,
                'n': len(self.results_a)
            },
            'model_b': {
                'mean': mean_b,
                'std': std_b,
                'n': len(self.results_b)
            },
            'difference': mean_b - mean_a,
            'relative_improvement': ((mean_b - mean_a) / mean_a * 100) if mean_a != 0 else 0,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'confidence_interval': (ci_lower, ci_upper),
            'is_significant': is_significant,
            'winner': winner
        }
        
        return result
    
    def print_report(self):
        """打印测试报告"""
        result = self.analyze()
        
        if 'error' in result:
            print(f"❌ {result['error']}")
            return
        
        print("\n" + "="*60)
        print("A/B测试报告")
        print("="*60)
        
        print(f"\n模型A (对照组):")
        print(f"  样本数: {result['model_a']['n']}")
        print(f"  均值: {result['model_a']['mean']:.4f}")
        print(f"  标准差: {result['model_a']['std']:.4f}")
        
        print(f"\n模型B (实验组):")
        print(f"  样本数: {result['model_b']['n']}")
        print(f"  均值: {result['model_b']['mean']:.4f}")
        print(f"  标准差: {result['model_b']['std']:.4f}")
        
        print(f"\n差异分析:")
        print(f"  绝对差异: {result['difference']:.4f}")
        print(f"  相对提升: {result['relative_improvement']:.2f}%")
        print(f"  p值: {result['p_value']:.4f}")
        print(f"  Cohen's d: {result['cohens_d']:.4f}")
        print(f"  95% CI: [{result['confidence_interval'][0]:.4f}, {result['confidence_interval'][1]:.4f}]")
        
        print(f"\n结论:")
        if result['is_significant']:
            print(f"  ✅ 差异显著 (p < 0.05)")
            print(f"  🏆 获胜者: 模型{result['winner']}")
        else:
            print(f"  ❌ 差异不显著 (p >= 0.05)")
            print(f"  建议继续收集数据或保持现状")
        
        print("="*60)


# 使用示例
ab_test = ABTest(model_a=old_model, model_b=new_model, traffic_split=0.5)

# 模拟用户请求
for user_id in range(1000):
    variant, model = ab_test.select_model(user_id)
    
    # 模拟交互
    state = env.reset()
    total_reward = 0
    
    for _ in range(100):
        with torch.no_grad():
            action_probs = model(torch.FloatTensor(state).unsqueeze(0))
            action = torch.multinomial(action_probs, 1).item()
        
        state, reward, done, _ = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    # 记录结果
    ab_test.record_result(variant, total_reward)

# 分析
ab_test.print_report()
```

继续...

---

## 39.3 监控与日志

### 39.3.1 性能监控

**关键指标**：
1. **环境指标**：Episode Return, Episode Length, Envv Steps/sec
2. **策略指标**：Entropy, KL Divergence, Value Loss, Policy Loss
3. **系统指标**：CPU/GPU Usage, RAM, Inference Latency

<div data-component="RLMonitoringDashboard"></div>

**Prometheus监控集成**：

```python
"""
Prometheus监控集成
"""

from prometheus_client import start_http_server, Gauge, Summary, Counter
import time
import random

# 定义指标
EPISODE_REWARD = Gauge('rl_episode_reward', 'Average reward per episode')
EPISODE_LENGTH = Gauge('rl_episode_length', 'Average length per episode')
TRAINING_LOSS = Gauge('rl_training_loss', 'Current training loss')
INFERENCE_LATENCY = Summary('rl_inference_latency_seconds', 'Time spent processing inference request')
TOTAL_STEPS = Counter('rl_total_steps', 'Total environment steps')

class PrometheusLogger:
    """
    Prometheus日志记录器
    """
    def __init__(self, port=8000):
        # 启动Prometheus metrics server
        start_http_server(port)
        print(f"Prometheus metrics server started on port {port}")
    
    def log_episode(self, reward, length):
        EPISODE_REWARD.set(reward)
        EPISODE_LENGTH.set(length)
    
    def log_step(self):
        TOTAL_STEPS.inc()
    
    def log_loss(self, loss):
        TRAINING_LOSS.set(loss)
    
    @INFERENCE_LATENCY.time()
    def process_inference(self):
        """模拟推理过程"""
        time.sleep(random.uniform(0.01, 0.05))


# 使用
logger = PrometheusLogger(port=8000)

# 模拟训练循环
for step in range(100):
    # 模拟推理
    logger.process_inference()
    logger.log_step()
    
    # 模拟episode结束
    if step % 10 == 0:
        reward = random.uniform(0, 100)
        length = random.uniform(50, 200)
        logger.log_episode(reward, length)
        print(f"Logged episode: reward={reward:.2f}")
    
    # 模拟训练更新
    if step % 20 == 0:
        loss = random.uniform(0.1, 1.0)
        logger.log_loss(loss)

print("Metrics available at http://localhost:8000/metrics")
```

### 39.3.2 异常检测

**检测逻辑**：
- **性能骤降**：最近N个episode平均奖励 < 历史均值 - 3σ
- **分布漂移**：观测值分布与训练集KL散度 > 阈值
- **梯度爆炸**：梯度范数 > 阈值

```python
"""
异常检测器
"""

class AnomalyDetector:
    """RL异常检测"""
    def __init__(self, window_size=100, threshold_sigma=3.0):
        self.history = deque(maxlen=window_size)
        self.threshold_sigma = threshold_sigma
        
        # 统计量
        self.mean = 0
        self.std = 0
        self.n = 0
    
    def check(self, value):
        """
        检查新值是否异常
        
        Args:
            value: 新的监测值（如episode reward）
            
        Returns:
            bool: 是否异常
        """
        if self.n < 10:  # 预热
            self._update(value)
            return False
        
        # Z-score检测
        z_score = (value - self.mean) / (self.std + 1e-8)
        
        if abs(z_score) > self.threshold_sigma:
            print(f"⚠️ Anomaly detected! Value={value:.2f}, Mean={self.mean:.2f}, Z={z_score:.2f}")
            return True
        
        self._update(value)
        return False
    
    def _update(self, value):
        """在线更新均值和方差 (Welford's algorithm)"""
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.std = np.sqrt((self.std**2 * (self.n - 2) + delta * delta2) / (self.n - 1)) if self.n > 1 else 0
```

---

## 39.4 数据管理

### 39.4.1 经验回放存储

**存储方案**：
- **内存 (Redis)**: 高吞吐，低延迟，容量有限
- **NoSQL (Cassandra/DynamoDB)**: 大容量，分布式
- **Data Lake (S3/HDFS)**: 离线分析，批量训练

**数据结构设计**：

```json
{
  "episode_id": "uuid-v4",
  "timestamp": 1678900000,
  "steps": [
    {
      "step_id": 0,
      "state": [0.1, 0.5, -0.2, ...],
      "action": 1,
      "reward": 1.0,
      "info": {"latency": 0.02}
    },
    ...
  ],
  "metadata": {
    "model_version": "v1.2.3",
    "user_group": "A"
  }
}
```

### 39.4.2 隐私保护

**PII (Personal Identifiable Information) 移除**：
- 在状态设计时避免包含ID、IP等
- 存储前进行掩码处理
- 差分隐私（Differential Privacy）训练

---

## 39.5 工程工具链

<div data-component="ToolchainComparison"></div>

### 39.5.1 Stable-Baselines3

**特点**：
- PyTorch实现
- 接口统一 (`.learn()`, `.predict()`)
- 文档主要，社区活跃
- 适合：科研、中小型项目、教学

```python
"""
Stable-Baselines3 示例
"""

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# 并行环境
vec_env = make_vec_env("CartPole-v1", n_envs=4)

# 创建模型
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    learning_rate=3e-4,
    device="auto"  # 自动选择CPU/GPU
)

# 训练
model.learn(total_timesteps=25000)

# 保存与加载
model.save("ppo_cartpole")
loaded_model = PPO.load("ppo_cartpole")

# 评估
obs = vec_env.reset()
for _ in range(100):
    action, _ = loaded_model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
```

### 39.5.2 RLlib (Ray)

**特点**：
- 分布式训练（Scale out）
- 支持多智能体 (MARL)
- 工业级强度
- 学习曲线较陡

```python
"""
Ray RLlib 示例
"""

import ray
from ray.rllib.algorithms.ppo import PPOConfig

ray.init()

# 配置
config = (
    PPOConfig()
    .environment("CartPole-v1")
    .framework("torch")
    .rollouts(num_rollout_workers=2)  # 2个并行worker
    .training(lr=0.0003)
)

# 构建算法
algo = config.build()

# 训练循环
for i in range(10):
    result = algo.train()
    print(f"Iter {i}: reward={result['episode_reward_mean']:.2f}")

# 保存
checkpoint = algo.save()
print(f"Checkpoint saved at {checkpoint}")
```

### 39.5.3 CleanRL

**特点**：
- 单文件实现（Single-file implementation）
- 极度简洁，便于修改
- 适合：算法研究、魔改

---

## 39.6 实际案例

### 39.6.1 推荐系统

**场景**：YouTube/TikTok视频推荐
- **Action**: 从百万候选集中选出Top-k视频
- **State**: 用户历史行为序列、当前上下文
- **Reward**: 观看时长、完播率、互动（点赞/分享）

**架构**：
1. **召回 (Retrieval)**: 双塔模型，快速筛选Top-1000
2. **粗排 (Pre-ranking)**: 过滤
3. **精排 (Ranking)**: RL/精细模型打分
4. **重排 (Re-ranking)**: RL考虑多样性、长期收益（Slate Optimization）

### 39.6.2 游戏 AI

**场景**：MOBA (Dota2/王者荣耀)
- **挑战**：长时序、不完全信息、巨大状态空间
- **方案**：
  - **OpenAI Five**: PPO + Self-play + LSTM (Scale up)
  - **架构**: Teacher-Student Distillation, Surgery (模型手术)
  - **Reward Shaping**: 稠密奖励 -> 稀疏奖励

### 39.6.3 资源调度

**场景**：数据中心冷却/作业调度
- **DeepMind Google Data Center**: 控制冷却系统
- **State**: 温度传感器、负载、天气
- **Action**: 制冷设定点
- **Reward**: -能耗 (约束：温度<安全阈值)
- **效果**: 节能40%

---

## 总结

工程实践是将RL从论文带入现实的关键：
1. **部署**：ONNX导出、量化加速、边缘计算
2. **系统**：Redis队列、在线学习闭环、A/B测试
3. **监控**：全链路指标、异常检测、Prometheus/Grafana
4. **工具**：SB3(易用) vs RLlib(扩展) vs CleanRL(研究)

**生产环境RL的铁律**：
- **Start Simple**: 先用简单的Baseline (如Heuristic/Supervised Learning)
- **Data First**: 数据质量决定上限
- **Safety**: 始终设置安全回退策略 (Safety Fallback)

---

## 参考资源

- **OpenNX**: https://onnx.ai/
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **Ray RLlib**: https://docs.ray.io/en/latest/rllib/index.html
- **Prometheus**: https://prometheus.io/
- **Chip Huyen**: "Designing Machine Learning Systems"

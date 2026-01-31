---
title: "第33章：Reasoning-Time RL与Process Reward"
description: "推理时计算扩展，过程奖励，搜索增强，OpenAI o1范式"
date: "2026-01-30"
---

# 第33章：Reasoning-Time RL 与 Process Reward

## 33.1 推理时 RL (Reasoning-Time RL)

### 33.1.1 测试时计算扩展

**传统范式**：模型性能主要由**训练时计算**决定（参数量、数据量）。

**新范式**：在**推理时**投入更多计算，提升输出质量。

**核心思想**：
- 训练阶段：学习推理能力
- 测试阶段：通过搜索/采样/验证扩展计算
- **时间换质量**：更长推理时间 → 更好答案

**OpenAI o1 的突破**（2024年9月）：
- 在数学、编程、科学推理任务上，**推理时间与性能呈幂律关系**
- 通过强化学习优化"思考过程"
- 测试时可花费数十秒"思考"

<div data-component="ReasoningTimeScaling"></div>

### 33.1.2 思维链优化 (Chain-of-Thought)

**CoT提示**（Wei et al., 2022）：让模型"一步一步思考"。

**示例**：

```
Q: Roger有5个网球。他又买了2罐网球，每罐3个球。他现在有几个网球？

标准回答：
A: 11个。

CoT回答：
A: Roger开始有5个球。
   2罐网球，每罐3个，所以2×3=6个球。
   5+6=11个球。
   答案：11个。
```

**RL优化CoT**：
- 目标：优化中间推理步骤
- 奖励：每一步的正确性（Process Reward）
- 方法：用RL微调，鼓励清晰、正确的推理链

### 33.1.3 OpenAI o1 模型架构

**o1的关键特性**：

1. **长思考链**：内部生成数千token的推理过程
2. **RL训练**：大规模强化学习优化推理策略
3. **自我反思**：能够检查、修正自己的错误
4. **可扩展性**：推理时间越长，性能越好

**训练方法（推测）**：
- 使用Process Reward Model（PRM）监督每一步
- MCTS或Beam Search在推理时探索多条路径
- 自我一致性验证选择最佳答案

**性能提升**：
- AIME 2024数学竞赛：GPT-4o (13%) → o1-preview (74%)
- Codeforces编程：GPT-4 (Elo 808) → o1 (Elo 1807, 89th percentile)

---

## 33.2 Process Reward vs Outcome Reward

### 33.2.1 过程奖励的优势

**Outcome Reward（结果奖励）**：
- 只看最终答案是否正确
- 二元：对/错
- 稀疏：整个推理过程只有一个信号

**Process Reward（过程奖励）**：
- 评估**每一个中间步骤**
- 密集信号：每步都有反馈
- 更好的学习信号

**对比示例**：

| 步骤 | 推理内容 | Outcome | Process |
|------|----------|---------|---------|
| 1 | Roger有5个球 | - | ✅ 正确 |
| 2 | 买了2罐，每罐3个 | - | ✅ 正确 |
| 3 | 2×3=5（错误！） | - | ❌ 错误 |
| 4 | 5+5=10 | - | ❌ 错误 |
| 最终 | 答案：10 | ❌ 错误 | - |

**Process Reward的优势**：
- 能精确定位错误位置（步骤3）
- 提供细粒度学习信号
- 泛化性更好

<div data-component="ProcessVsOutcomeReward"></div>

### 33.2.2 中间步骤监督

**数据收集**：人工标注每个推理步骤的正确性。

**PRM800K数据集**（Lightman et al., 2023）：
- **800K**个人工标注的推理步骤
- 来自MATH数据集的数学题
- 每步标签：正确 / 错误 / 中性

**标注示例**：

```python
{
    "problem": "求解方程 2x + 3 = 11",
    "solution_steps": [
        {"step": "2x + 3 = 11", "label": "neutral"},
        {"step": "2x = 11 - 3", "label": "positive"},  # ✅ 
        {"step": "2x = 8", "label": "positive"},       # ✅
        {"step": "x = 8 / 2", "label": "positive"},    # ✅
        {"step": "x = 4", "label": "positive"}         # ✅
    ]
}
```

### 33.2.3 训练Process Reward Model

**目标**：训练模型 $\text{PRM}(s_t | s_{<t})$ 预测步骤 $s_t$ 的正确性。

**架构**：
- 基于语言模型（如GPT-3.5）
- 输入：问题 + 前面的步骤 + 当前步骤
- 输出：正确性概率 $p \in [0, 1]$

**损失函数**：

$$
\mathcal{L}_{\text{PRM}} = -\sum_{i} \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]
$$

其中 $y_i \in \{0, 1\}$ 是人工标注。

**实现**：

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class ProcessRewardModel(nn.Module):
    """
    过程奖励模型：评估推理步骤的正确性
    """
    def __init__(self, base_model_name="gpt2", hidden_dim=768):
        super().__init__()
        
        # 加载基础语言模型
        self.lm = AutoModelForCausalLM.from_pretrained(base_model_name)
        
        # 奖励头：输出[0, 1]的正确性概率
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出概率
        )
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (batch, seq_len) - 问题 + 前序步骤 + 当前步骤
            attention_mask: (batch, seq_len)
        
        Returns:
            correctness_prob: (batch,) - 当前步骤正确性概率
        """
        # LM编码
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 取最后一个token的hidden state
        last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden)
        
        # 找到每个序列的最后有效token
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        last_token_hidden = last_hidden[batch_indices, seq_lengths]  # (batch, hidden)
        
        # 预测正确性
        correctness_prob = self.reward_head(last_token_hidden).squeeze(-1)
        
        return correctness_prob


def train_prm(
    model_name="gpt2",
    prm_data_path="prm800k.jsonl",
    output_dir="./prm_model",
    epochs=3,
    batch_size=8,
    learning_rate=5e-5
):
    """
    训练过程奖励模型
    
    Args:
        prm_data_path: PRM数据路径，格式：
            {
                "problem": "...",
                "steps": ["step1", "step2", ...],
                "labels": [1, 1, 0, 1, ...]  # 1=正确, 0=错误
            }
    """
    import json
    from torch.utils.data import Dataset, DataLoader
    
    # 1. 加载数据
    with open(prm_data_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # 2. 构造训练样本
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    class PRMDataset(Dataset):
        def __init__(self, data, tokenizer, max_length=512):
            self.samples = []
            
            for item in data:
                problem = item['problem']
                steps = item['steps']
                labels = item['labels']
                
                # 为每一步创建样本
                for i, (step, label) in enumerate(zip(steps, labels)):
                    # 输入：问题 + 前序步骤 + 当前步骤
                    context = f"Problem: {problem}\n"
                    if i > 0:
                        context += "Previous steps:\n" + "\n".join(steps[:i]) + "\n"
                    context += f"Current step: {step}"
                    
                    self.samples.append({
                        'text': context,
                        'label': float(label)
                    })
            
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            sample = self.samples[idx]
            
            encoding = self.tokenizer(
                sample['text'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'label': torch.tensor(sample['label'], dtype=torch.float)
            }
    
    dataset = PRMDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. 训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProcessRewardModel(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            probs = model(input_ids, attention_mask)
            
            # 损失
            loss = criterion(probs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 准确率
            predictions = (probs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")
    
    # 保存模型
    torch.save(model.state_dict(), f"{output_dir}/prm_model.pt")
    print(f"PRM模型已保存至 {output_dir}")
```

---

## 33.3 搜索增强 RL

### 33.3.1 蒙特卡洛树搜索 (MCTS)

**MCTS for Reasoning**：将推理过程视为决策树。

**流程**：

1. **Selection（选择）**：从根节点开始，用UCB选择子节点
2. **Expansion（扩展）**：添加新的推理步骤
3. **Simulation（模拟）**：用策略模型完成推理
4. **Backpropagation（回传）**：用PRM评分更新节点价值

**UCB公式**：

$$
\text{UCB}(s, a) = Q(s, a) + c \cdot \sqrt{\frac{\log N(s)}{N(s, a)}}
$$

其中：
- $Q(s, a)$：节点价值（由PRM估计）
- $N(s)$：父节点访问次数
- $N(s, a)$：子节点访问次数
- $c$：探索系数

<div data-component="MCTSForReasoning"></div>

**实现**：

```python
import math
import numpy as np

class MCTSNode:
    """MCTS节点"""
    def __init__(self, state, parent=None):
        self.state = state  # 当前推理状态（问题+已有步骤）
        self.parent = parent
        self.children = {}  # action -> MCTSNode
        self.visits = 0
        self.value = 0.0  # 累积价值
    
    def is_fully_expanded(self, possible_actions):
        return len(self.children) == len(possible_actions)
    
    def best_child(self, c=1.4):
        """UCB选择最佳子节点"""
        choices_weights = [
            (child.value / child.visits) + c * math.sqrt(math.log(self.visits) / child.visits)
            for child in self.children.values()
        ]
        return list(self.children.values())[np.argmax(choices_weights)]
    
    def expand(self, action, next_state):
        """扩展新节点"""
        child = MCTSNode(next_state, parent=self)
        self.children[action] = child
        return child


class MCTSReasoning:
    """
    MCTS for Reasoning
    """
    def __init__(self, policy_model, prm_model, max_iterations=100):
        """
        Args:
            policy_model: 生成下一步推理的策略模型
            prm_model: 过程奖励模型（评估步骤正确性）
            max_iterations: MCTS迭代次数
        """
        self.policy_model = policy_model
        self.prm_model = prm_model
        self.max_iterations = max_iterations
    
    def search(self, problem):
        """
        对给定问题进行MCTS搜索
        
        Returns:
            best_solution: 最佳推理路径
        """
        root = MCTSNode(state={"problem": problem, "steps": []})
        
        for _ in range(self.max_iterations):
            # 1. Selection
            node = self._select(root)
            
            # 2. Expansion
            if not self._is_terminal(node):
                node = self._expand(node)
            
            # 3. Simulation
            value = self._simulate(node)
            
            # 4. Backpropagation
            self._backpropagate(node, value)
        
        # 返回访问次数最多的路径
        return self._best_solution(root)
    
    def _select(self, node):
        """选择阶段：UCB选择"""
        while not self._is_terminal(node):
            if not node.is_fully_expanded(self._get_possible_actions(node)):
                return node
            else:
                node = node.best_child()
        return node
    
    def _expand(self, node):
        """扩展阶段：生成新的推理步骤"""
        possible_actions = self._get_possible_actions(node)
        untried_actions = [a for a in possible_actions if a not in node.children]
        
        if untried_actions:
            action = np.random.choice(untried_actions)
            next_state = self._apply_action(node.state, action)
            return node.expand(action, next_state)
        
        return node
    
    def _simulate(self, node):
        """模拟阶段：用PRM评估当前步骤"""
        state = node.state
        
        # 用PRM评估最后一步
        if len(state['steps']) > 0:
            last_step = state['steps'][-1]
            context = f"Problem: {state['problem']}\n"
            context += "\n".join(state['steps'][:-1]) + f"\nCurrent: {last_step}"
            
            correctness_prob = self.prm_model.evaluate(context)
            return correctness_prob
        
        return 0.5  # 根节点
    
    def _backpropagate(self, node, value):
        """回传阶段：更新节点价值"""
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent
    
    def _get_possible_actions(self, node):
        """生成可能的下一步推理"""
        # 用策略模型生成候选步骤（beam search / sampling）
        return self.policy_model.generate_next_steps(
            node.state['problem'],
            node.state['steps'],
            num_candidates=5
        )
    
    def _apply_action(self, state, action):
        """应用动作"""
        new_steps = state['steps'] + [action]
        return {"problem": state['problem'], "steps": new_steps}
    
    def _is_terminal(self, node):
        """是否终止（答案已找到）"""
        if len(node.state['steps']) == 0:
            return False
        
        last_step = node.state['steps'][-1]
        return "答案" in last_step or "Answer" in last_step
    
    def _best_solution(self, root):
        """选择最佳解"""
        current = root
        solution_steps = []
        
        while current.children:
            # 选择访问次数最多的子节点
            current = max(current.children.values(), key=lambda c: c.visits)
            if current.state['steps']:
                solution_steps.append(current.state['steps'][-1])
        
        return solution_steps
```

### 33.3.2 Beam Search

**Beam Search**：保留top-k个最优路径，逐步扩展。

**优势**：
- 比贪心搜索更全面
- 比MCTS更简单、快速
- 适合生成任务

### 33.3.3 Best-of-N 采样

**方法**：
1. 采样N个完整解
2. 用评分模型（PRM/Verifier）打分
3. 选择最高分的解

**性能提升**：N越大，准确率越高（但计算成本↑）。

**实验结果**（GSM8K数学题）：

| N | 准确率 | 计算成本 |
|---|--------|----------|
| 1 | 65% | 1x |
| 4 | 72% | 4x |
| 16 | 78% | 16x |
| 64 | 82% | 64x |

---

## 33.4 自我验证 (Self-Verification)

### 33.4.1 生成-验证循环

**流程**：

1. **生成**：模型生成候选答案
2. **验证**：模型检查答案是否正确
3. **迭代**：如果错误，重新生成

<div data-component="SelfVerificationLoop"></div>

**实现**：

```python
def self_verification_loop(problem, model, max_attempts=5):
    """
    自我验证循环
    
    Args:
        problem: 待解决的问题
        model: 语言模型（同时用于生成和验证）
        max_attempts: 最大尝试次数
    
    Returns:
        final_answer: 验证通过的答案
    """
    for attempt in range(max_attempts):
        # 1. 生成候选答案
        candidate = model.generate(
            f"Problem: {problem}\nSolution:",
            max_tokens=512
        )
        
        # 2. 自我验证
        verification_prompt = f"""
Problem: {problem}
Proposed Solution: {candidate}

Check if the solution is correct. Think step by step:
1. Is the reasoning logical?
2. Are the calculations correct?
3. Does it answer the question?

Verification:"""
        
        verification = model.generate(verification_prompt, max_tokens=256)
        
        # 3. 判断是否通过
        if "correct" in verification.lower() or "yes" in verification.lower():
            print(f"✅ 验证通过（第{attempt+1}次尝试）")
            return candidate
        else:
            print(f"❌ 验证失败（第{attempt+1}次尝试），重新生成...")
    
    print("⚠️ 达到最大尝试次数，返回最后一次答案")
    return candidate
```

### 33.4.2 一致性检查

**Self-Consistency**（Wang et al., 2022）：

1. 生成N个推理路径（用不同采样）
2. 提取每个路径的最终答案
3. **多数投票**选择最常见的答案

**效果**：在GSM8K上从~65%提升到~74%（N=40）。

### 33.4.3 多数投票

```python
from collections import Counter

def self_consistency(problem, model, num_samples=10):
    """
    自一致性：多数投票
    """
    answers = []
    
    for i in range(num_samples):
        # 生成推理路径（高温度采样）
        solution = model.generate(
            f"Problem: {problem}\nLet's think step by step:",
            temperature=0.7,  # 增加多样性
            max_tokens=512
        )
        
        # 提取答案
        answer = extract_final_answer(solution)
        answers.append(answer)
    
    # 多数投票
    vote_counts = Counter(answers)
    most_common_answer, count = vote_counts.most_common(1)[0]
    
    confidence = count / num_samples
    
    print(f"投票结果：{most_common_answer} ({count}/{num_samples}, 置信度{confidence:.1%})")
    return most_common_answer


def extract_final_answer(solution_text):
    """从解答中提取最终答案"""
    # 简化示例：查找"答案："后的内容
    if "答案：" in solution_text:
        return solution_text.split("答案：")[1].strip().split()[0]
    elif "Answer:" in solution_text:
        return solution_text.split("Answer:")[1].strip().split()[0]
    else:
        # fallback：返回最后一行
        return solution_text.strip().split("\n")[-1]
```

---

## 33.5 数学推理与代码生成

### 33.5.1 数学数据集

**GSM8K**（小学数学）：
- 8,500道题
- 需要多步算术推理
- GPT-4准确率：~92%

**MATH**（竞赛数学）：
- 12,500道题，难度高
- 涵盖代数、几何、概率等
- GPT-4准确率：~52% → o1: ~83%

### 33.5.2 代码生成数据集

**HumanEval**：
- 164道Python编程题
- GPT-4 pass@1: ~67% → o1: ~90%

**MBPP** (Mostly Basic Python Problems)：
- 974道基础Python题

### 33.5.3 AlphaCode方法

**AlphaCode**（DeepMind, 2022）：

1. **大规模采样**：生成数百万个候选程序
2. **聚类过滤**：按功能聚类，每类选1个
3. **测试用例验证**：在公开测试上运行
4. **提交top-10**：选择最优的10个

**性能**：达到Codeforces平均水平（超过54%参赛者）。

---

## 33.6 计算-性能权衡

### 33.6.1 推理时间 vs 准确率

**实验观察**（OpenAI o1）：

| 推理时间 | AIME准确率 | 成本 |
|----------|-----------|------|
| 1秒 | 13% | 1x |
| 10秒 | 45% | 10x |
| 60秒 | 74% | 60x |

**幂律关系**：

$$
\text{Accuracy} \propto (\text{Compute})^\alpha
$$

其中 $\alpha \approx 0.3-0.5$（任务相关）。

<div data-component="ComputePerformanceTradeoff"></div>

### 33.6.2 Scaling Laws

**训练时Scaling**（经典）：

$$
\text{Loss} \propto N^{-\alpha} \cdot D^{-\beta}
$$

- $N$：参数量
- $D$：数据量

**推理时Scaling**（新）：

$$
\text{Performance} \propto C_{\text{test}}^{\gamma}
$$

- $C_{\text{test}}$：测试时计算（FLOP / 时间）
- $\gamma$：scaling指数

**启示**：可以用更小模型 + 更多推理时计算达到大模型效果。

### 33.6.3 效率优化

**减少推理成本**：

1. **Early Stopping**：简单问题少搜索，难题多搜索
2. **Adaptive Compute**：动态调整beam size / MCTS iterations
3. **缓存**：复用常见子问题的推理
4. **蒸馏**：将推理能力蒸馏回小模型

---

## 总结

本章涵盖了Reasoning-Time RL的核心内容：

1. **推理时RL**：从训练时计算转向测试时计算扩展
2. **过程奖励 vs 结果奖励**：密集信号优于稀疏信号
3. **搜索增强**：MCTS、Beam Search、Best-of-N
4. **自我验证**：生成-验证循环、一致性检查
5. **数学与代码**：GSM8K、MATH、HumanEval、AlphaCode
6. **计算权衡**：推理时间与性能的幂律关系

**关键要点**：
- OpenAI o1 开启了推理时RL的新范式
- Process Reward提供更细粒度的学习信号
- 搜索 + 验证能大幅提升复杂推理性能
- 计算-性能权衡：时间换质量

**未来方向**：
- 更高效的搜索算法
- 自动化过程奖励标注
- 多模态推理
- 混合符号-神经推理

---

## 参考文献

- Lightman, H., et al. (2023). "Let's Verify Step by Step." *arXiv*.
- OpenAI (2024). "Learning to Reason with LLMs." *OpenAI Blog*.
- Wei, J., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." *NeurIPS*.
- Wang, X., et al. (2022). "Self-Consistency Improves Chain of Thought Reasoning." *ICLR*.
- Li, Y., et al. (2022). "Competition-Level Code Generation with AlphaCode." *Science*.
- Silver, D., et al. (2017). "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm." *arXiv* (AlphaZero).

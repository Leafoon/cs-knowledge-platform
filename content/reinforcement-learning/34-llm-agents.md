---
title: "第34章：LLM Agent与工具使用"
description: "ReAct框架，工具学习，RL优化Agent，多步规划与实际应用"
date: "2026-01-30"
---

# 第34章：LLM Agent 与工具使用

## 34.1 LLM 作为 Agent

### 34.1.1 ReAct 框架

**ReAct**（Reasoning and Acting）：将推理与行动交织，让LLM既能"思考"又能"行动"。

**核心思想**：
```
Thought → Action → Observation → Thought → Action → ...
```

**传统方法问题**：
- **Chain-of-Thought (CoT)**：只思考，不与环境交互
- **Action-only**：盲目行动，缺乏推理

**ReAct优势**：
- ✅ 显式推理链
- ✅ 实时环境反馈
- ✅ 错误检测与修正
- ✅ 可解释性强

<div data-component="ReActFramework"></div>

**ReAct示例**：

```
问题：2024年诺贝尔物理学奖得主的第一篇论文发表于哪一年？

Thought 1: 我需要先找到2024年诺贝尔物理学奖得主是谁
Action 1: Search[2024 Nobel Prize Physics winner]
Observation 1: John Hopfield 和 Geoffrey Hinton 获得2024年诺贝尔物理学奖

Thought 2: 现在我知道得主是John Hopfield和Geoffrey Hinton，我需要找到他们的第一篇论文
Action 2: Search[John Hopfield first paper year]
Observation 2: John Hopfield于1982年发表了著名的Hopfield网络论文

Thought 3: 我还需要查Geoffrey Hinton的第一篇论文
Action 3: Search[Geoffrey Hinton first paper year]
Observation 3: Geoffrey Hinton的第一篇论文发表于1976年

Thought 4: 两位得主中，Hinton的论文更早(1976年)，但题目问的是"第一篇"，需要明确是指谁的
Action 4: Finish[两位得主中，Geoffrey Hinton的第一篇论文发表于1976年，John Hopfield的著名论文发表于1982年]
```

### 34.1.2 思考-行动循环

**循环架构**：

```python
def react_loop(question, max_steps=10):
    """
    ReAct循环主逻辑
    
    Args:
        question: 用户问题
        max_steps: 最大步数（防止无限循环）
    
    Returns:
        final_answer: 最终答案
    """
    context = f"Question: {question}\n"
    
    for step in range(max_steps):
        # 1. 思考（Thought）
        thought_prompt = context + f"Thought {step+1}:"
        thought = llm.generate(thought_prompt, max_tokens=100)
        context += f"Thought {step+1}: {thought}\n"
        
        # 2. 决定行动（Action）
        action_prompt = context + f"Action {step+1}:"
        action = llm.generate(action_prompt, max_tokens=50)
        context += f"Action {step+1}: {action}\n"
        
        # 3. 执行行动并观察（Observation）
        if action.startswith("Finish["):
            # 提取答案并结束
            answer = extract_answer(action)
            return answer
        
        observation = execute_action(action)
        context += f"Observation {step+1}: {observation}\n"
        
        # 4. 检查是否超时
        if step == max_steps - 1:
            return "达到最大步数，未找到答案"
    
    return "循环结束"


def execute_action(action_str):
    """
    执行具体行动
    
    支持的行动类型：
    - Search[query]: 搜索
    - Lookup[term]: 查找
    - Calculate[expression]: 计算
    - Finish[answer]: 结束并返回答案
    """
    if action_str.startswith("Search["):
        query = extract_argument(action_str)
        return search_engine.search(query)
    
    elif action_str.startswith("Lookup["):
        term = extract_argument(action_str)
        return lookup_database(term)
    
    elif action_str.startswith("Calculate["):
        expr = extract_argument(action_str)
        try:
            result = eval(expr)  # 生产环境需要安全的表达式求值
            return f"计算结果：{result}"
        except Exception as e:
            return f"计算错误：{e}"
    
    else:
        return "未知行动类型"


def extract_argument(action_str):
    """从Action[argument]中提取argument"""
    start = action_str.index('[') + 1
    end = action_str.rindex(']')
    return action_str[start:end]
```

### 34.1.3 工具调用能力

**工具定义**：

```python
from typing import Callable, Dict, Any, List
import inspect

class Tool:
    """工具基类"""
    def __init__(
        self,
        name: str,
        func: Callable,
        description: str,
        return_direct: bool = False
    ):
        """
        Args:
            name: 工具名称
            func: 工具实现函数
            description: 工具描述（供LLM理解）
            return_direct: 是否直接返回结果给用户
        """
        self.name = name
        self.func = func
        self.description = description
        self.return_direct = return_direct
        
        # 自动提取函数签名
        self.signature = inspect.signature(func)
        self.parameters = {
            name: param.annotation
            for name, param in self.signature.parameters.items()
        }
    
    def run(self, **kwargs) -> str:
        """执行工具"""
        try:
            result = self.func(**kwargs)
            return str(result)
        except Exception as e:
            return f"工具执行错误：{e}"
    
    def to_prompt(self) -> str:
        """生成工具的prompt描述"""
        params_desc = ", ".join([
            f"{name}: {dtype.__name__}"
            for name, dtype in self.parameters.items()
        ])
        return f"{self.name}({params_desc}): {self.description}"


# 示例工具定义
def wikipedia_search(query: str) -> str:
    """搜索维基百科"""
    import wikipediaapi
    wiki = wikipediaapi.Wikipedia('en')
    page = wiki.page(query)
    
    if page.exists():
        # 返回摘要（前500字符）
        return page.summary[:500]
    else:
        return f"未找到'{query}'相关的维基百科页面"


def calculator(expression: str) -> str:
    """计算数学表达式"""
    import ast
    import operator
    
    # 安全的表达式求值（只允许基本运算）
    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow
    }
    
    def eval_expr(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            op = type(node.op)
            if op not in allowed_ops:
                raise ValueError(f"不允许的操作：{op}")
            return allowed_ops[op](eval_expr(node.left), eval_expr(node.right))
        else:
            raise TypeError(f"不支持的节点类型：{type(node)}")
    
    try:
        tree = ast.parse(expression, mode='eval')
        result = eval_expr(tree.body)
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"


def python_repl(code: str) -> str:
    """执行Python代码"""
    import io
    import sys
    from contextlib import redirect_stdout
    
    # 捕获标准输出
    output = io.StringIO()
    
    try:
        with redirect_stdout(output):
            # 安全性警告：生产环境需要沙箱
            exec(code, {"__builtins__": __builtins__})
        
        result = output.getvalue()
        return result if result else "代码执行成功（无输出）"
    except Exception as e:
        return f"执行错误：{e}"


# 工具注册
tools = [
    Tool("Wikipedia", wikipedia_search, "搜索维基百科获取知识"),
    Tool("Calculator", calculator, "计算数学表达式"),
    Tool("PythonREPL", python_repl, "执行Python代码", return_direct=True)
]
```

---

## 34.2 工具学习

### 34.2.1 API 调用

**OpenAPI规范集成**：

```python
import requests
import json
from typing import Dict, Any

class APITool(Tool):
    """API调用工具"""
    def __init__(
        self,
        name: str,
        api_spec: Dict[str, Any],
        base_url: str,
        auth_token: str = None
    ):
        """
        Args:
            name: API名称
            api_spec: OpenAPI 3.0规范
            base_url: API基础URL
            auth_token: 认证token（可选）
        """
        self.api_spec = api_spec
        self.base_url = base_url
        self.auth_token = auth_token
        
        # 从spec生成描述
        description = api_spec.get('info', {}).get('description', '')
        
        super().__init__(
            name=name,
            func=self._call_api,
            description=description
        )
    
    def _call_api(self, endpoint: str, method: str = "GET", **params) -> str:
        """
        调用API
        
        Args:
            endpoint: API端点（如 /users/{id}）
            method: HTTP方法
            **params: 请求参数
        """
        url = f"{self.base_url}{endpoint}"
        
        headers = {}
        if self.auth_token:
            headers['Authorization'] = f"Bearer {self.auth_token}"
        
        try:
            if method == "GET":
                response = requests.get(url, params=params, headers=headers)
            elif method == "POST":
                response = requests.post(url, json=params, headers=headers)
            elif method == "PUT":
                response = requests.put(url, json=params, headers=headers)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers)
            else:
                return f"不支持的HTTP方法：{method}"
            
            response.raise_for_status()
            return json.dumps(response.json(), indent=2, ensure_ascii=False)
        
        except requests.exceptions.RequestException as e:
            return f"API调用失败：{e}"


# 示例：天气API
weather_api = APITool(
    name="WeatherAPI",
    api_spec={
        "info": {
            "title": "Weather API",
            "description": "获取实时天气信息"
        },
        "paths": {
            "/weather": {
                "get": {
                    "parameters": [
                        {"name": "city", "in": "query", "required": True}
                    ]
                }
            }
        }
    },
    base_url="https://api.weatherapi.com/v1"
)
```

### 34.2.2 代码执行器

**安全的代码执行环境**：

```python
import docker
import tempfile
import os

class DockerCodeExecutor(Tool):
    """
    Docker容器中执行代码（安全隔离）
    """
    def __init__(self, language="python", timeout=30):
        """
        Args:
            language: 编程语言（python, javascript, bash等）
            timeout: 执行超时（秒）
        """
        self.language = language
        self.timeout = timeout
        self.client = docker.from_env()
        
        # 语言到Docker镜像的映射
        self.image_map = {
            "python": "python:3.9-slim",
            "javascript": "node:16-slim",
            "bash": "bash:latest"
        }
        
        super().__init__(
            name=f"{language.capitalize()}Executor",
            func=self._execute_code,
            description=f"在隔离环境中执行{language}代码"
        )
    
    def _execute_code(self, code: str) -> str:
        """执行代码"""
        image = self.image_map.get(self.language)
        if not image:
            return f"不支持的语言：{self.language}"
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=f'.{self._get_extension()}',
            delete=False
        ) as f:
            f.write(code)
            code_file = f.name
        
        try:
            # 在Docker容器中执行
            container = self.client.containers.run(
                image=image,
                command=self._get_command(os.path.basename(code_file)),
                volumes={
                    os.path.dirname(code_file): {
                        'bind': '/app',
                        'mode': 'ro'  # 只读
                    }
                },
                working_dir='/app',
                detach=True,
                mem_limit='256m',  # 内存限制
                network_disabled=True  # 禁用网络
            )
            
            # 等待执行完成
            result = container.wait(timeout=self.timeout)
            logs = container.logs().decode('utf-8')
            
            # 清理
            container.remove()
            os.unlink(code_file)
            
            if result['StatusCode'] == 0:
                return logs
            else:
                return f"执行失败（退出码{result['StatusCode']}）：\n{logs}"
        
        except docker.errors.ContainerError as e:
            return f"容器错误：{e}"
        except Exception as e:
            return f"执行错误：{e}"
        finally:
            if os.path.exists(code_file):
                os.unlink(code_file)
    
    def _get_extension(self) -> str:
        """获取文件扩展名"""
        ext_map = {
            "python": "py",
            "javascript": "js",
            "bash": "sh"
        }
        return ext_map.get(self.language, "txt")
    
    def _get_command(self, filename: str) -> str:
        """获取执行命令"""
        cmd_map = {
            "python": f"python {filename}",
           "javascript": f"node {filename}",
            "bash": f"bash {filename}"
        }
        return cmd_map.get(self.language)
```

### 34.2.3 外部知识库

**向量数据库集成**：

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class VectorKnowledgeBase(Tool):
    """
    向量知识库工具：基于语义相似度检索
    """
    def __init__(
        self,
        documents: List[str],
        model_name="all-MiniLM-L6-v2",
        top_k=3
    ):
        """
        Args:
            documents: 知识库文档列表
            model_name: 句子嵌入模型（来自sentence-transformers）
            top_k: 返回top-k个最相关文档
        """
        self.documents = documents
        self.top_k = top_k
        
        # 加载嵌入模型
        self.encoder = SentenceTransformer(model_name)
        
        # 构建向量索引
        self.index = self._build_index()
        
        super().__init__(
            name="KnowledgeBase",
            func=self._retrieve,
            description="从知识库检索相关信息"
        )
    
    def _build_index(self):
        """构建FAISS索引"""
        # 编码所有文档
        embeddings = self.encoder.encode(
            self.documents,
            show_progress_bar=True
        )
        
        # 归一化（用于余弦相似度）
        faiss.normalize_L2(embeddings)
        
        # 创建FAISS索引
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product（余弦）
        index.add(embeddings.astype('float32'))
        
        return index
    
    def _retrieve(self, query: str) -> str:
        """检索相关文档"""
        # 编码查询
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # 搜索
        scores, indices = self.index.search(
            query_embedding.astype('float32'),
            self.top_k
        )
        
        # 格式化结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                'document': self.documents[idx],
                'score': float(score)
            })
        
        # 返回格式化的检索结果
        output = f"找到{len(results)}个相关文档：\n\n"
        for i, res in enumerate(results, 1):
            output += f"{i}. (相似度{res['score']:.3f}) {res['document'][:200]}...\n\n"
        
        return output


# 示例使用
knowledge_docs = [
    "强化学习是机器学习的一个分支，研究智能体如何在环境中采取行动以最大化累积奖励...",
    "深度强化学习结合了深度学习和强化学习，使用神经网络作为函数逼近器...",
    "Q-learning是一种无模型的强化学习算法，学习状态-动作对的价值函数...",
    # ... 更多文档
]

kb_tool = VectorKnowledgeBase(knowledge_docs)
```

---

## 34.3 RL 优化 Agent

### 34.3.1 轨迹级奖励

**Agent轨迹定义**：

$$
\tau = (s_0, a_0, o_0, s_1, a_1, o_1, \ldots, s_T, a_T)
$$

其中：
- $s_t$：思考（Thought）
- $a_t$：行动（Action）
- $o_t$：观察（Observation）

**轨迹级奖励信号**：

```python
def compute_trajectory_reward(trajectory, final_answer, ground_truth):
    """
    计算Agent轨迹的总奖励
    
    Args:
        trajectory: [(thought, action, observation), ...]
        final_answer: 最终答案
        ground_truth: 正确答案
    
    Returns:
        total_reward: 总奖励
        step_rewards: 每步奖励列表
    """
    step_rewards = []
    
    # 1. 中间步骤奖励
    for i, (thought, action, observation) in enumerate(trajectory):
        reward = 0.0
        
        # 推理质量
        if is_logical(thought):
            reward += 0.1
        
        # 行动有效性
        if action_is_valid(action):
            reward += 0.1
        
        # 观察有用性
        if observation_is_informative(observation):
            reward += 0.2
        
        # 进度奖励（接近目标）
        if makes_progress(action, ground_truth):
            reward += 0.3
        
        step_rewards.append(reward)
    
    # 2. 最终答案奖励（主要信号）
    if final_answer == ground_truth:
        final_reward = 10.0  # 正确答案
    elif is_partially_correct(final_answer, ground_truth):
        final_reward = 3.0  # 部分正确
    else:
        final_reward = -2.0  # 错误答案
    
    step_rewards.append(final_reward)
    
    # 3. 效率惩罚（步数太多）
    num_steps = len(trajectory)
    efficiency_penalty = -0.1 * max(0, num_steps - 5)
    
    total_reward = sum(step_rewards) + efficiency_penalty
    
    return total_reward, step_rewards
```

<div data-component="ToolSelectionProcess"></div>

### 34.3.2 工具选择优化

**工具选择策略梯度**：

```python
import torch
import torch.nn as nn
import torch.optim as optim

class ToolSelectorPolicy(nn.Module):
    """
    工具选择策略网络
    """
    def __init__(
        self,
        state_dim,
        num_tools,
        hidden_dim=256
    ):
        """
        Args:
            state_dim: 状态维度（当前上下文）
            num_tools: 工具数量
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.tool_selector = nn.Linear(hidden_dim, num_tools)
        
    def forward(self, state):
        """
        Args:
            state: (batch, state_dim) - 编码后的上下文
        
        Returns:
            tool_logits: (batch, num_tools) - 工具logits
        """
        hidden = self.encoder(state)
        tool_logits = self.tool_selector(hidden)
        return tool_logits
    
    def select_tool(self, state, temperature=1.0):
        """采样工具"""
        with torch.no_grad():
            logits = self.forward(state) / temperature
            probs = torch.softmax(logits, dim=-1)
            tool_idx = torch.multinomial(probs, num_samples=1)
        return tool_idx.item(), probs[0, tool_idx].item()


def train_tool_selector(
    policy,
    trajectories,
    tools,
    learning_rate=1e-4,
    epochs=100
):
    """
    训练工具选择策略
    
    Args:
        policy: ToolSelectorPolicy网络
        trajectories: 训练轨迹数据
            [
                {
                    'states': [...],  # 每步的状态
                    'tools': [...],   # 选择的工具
                    'reward': float   # 轨迹总奖励
                },
                ...
            ]
        tools: 工具列表
    """
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for traj in trajectories:
            states = torch.tensor(traj['states'], dtype=torch.float32)
            tools_taken = torch.tensor(traj['tools'], dtype=torch.long)
            reward = traj['reward']
            
            # 前向传播
            logits = policy(states)
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # 选择的工具的log概率
            selected_log_probs = log_probs.gather(
                1,
                tools_taken.unsqueeze(-1)
            ).squeeze(-1)
            
            # REINFORCE损失（策略梯度）
            loss = -(selected_log_probs * reward).mean()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss / len(trajectories):.4f}")
```

### 34.3.3 错误恢复

**自动错误检测与恢复**：

```python
class ErrorRecoveryAgent:
    """
    具有错误恢复能力的Agent
    """
    def __init__(self, llm, tools, max_retries=3):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_retries = max_retries
    
    def run(self, question, max_steps=10):
        """运行Agent with错误恢复"""
        context = f"Question: {question}\n"
        history = []
        
        for step in range(max_steps):
            # 生成思考
            thought = self._generate_thought(context)
            context += f"Thought {step+1}: {thought}\n"
            history.append({'type': 'thought', 'content': thought})
            
            # 生成行动
            action = self._generate_action(context)
            context += f"Action {step+1}: {action}\n"
            history.append({'type': 'action', 'content': action})
            
            # 执行行动（with 重试）
            observation, success = self._execute_with_retry(action)
            
            if not success:
                # 错误恢复
                recovery_thought = "上一个行动失败，我需要尝试不同的方法"
                context += f"Thought {step+2}: {recovery_thought}\n"
                context += f"Observation {step+1}: {observation}\n"
                history.append({'type': 'error_recovery', 'content': recovery_thought})
            else:
                context += f"Observation {step+1}: {observation}\n"
                history.append({'type': 'observation', 'content': observation})
            
            # 检查是否完成
            if "Finish[" in action:
                answer = self._extract_answer(action)
                return answer, history
        
        return "达到最大步数", history
    
    def _execute_with_retry(self, action):
        """带重试的行动执行"""
        for attempt in range(self.max_retries):
            try:
                result = self._execute_action(action)
                return result, True
            except Exception as e:
                if attempt < self.max_retries - 1:
                    # 尝试修复
                    action = self._fix_action(action, str(e))
                else:
                    return f"执行失败（{self.max_retries}次重试）：{e}", False
        
        return "未知错误", False
    
    def _fix_action(self, action, error_msg):
        """尝试修复错误的行动"""
        # 使用LLM修复
        fix_prompt = f"""
原始行动：{action}
错误信息：{error_msg}

请修正上述行动以解决错误。修正后的行动：
"""
        fixed_action = self.llm.generate(fix_prompt, max_tokens=50)
        return fixed_action.strip()
    
    def _generate_thought(self, context):
        """生成思考"""
        prompt = context + "Thought:"
        return self.llm.generate(prompt, max_tokens=100, stop=["\n"])
    
    def _generate_action(self, context):
        """生成行动"""
        available_tools = ", ".join(self.tools.keys())
        prompt = context + f"Action (可用工具: {available_tools}):"
        return self.llm.generate(prompt, max_tokens=50, stop=["\n"])
    
    def _execute_action(self, action_str):
        """执行行动"""
        # 解析行动
        if "[" not in action_str:
            raise ValueError("无效的行动格式")
        
        tool_name = action_str[:action_str.index("[")]
        args_str = action_str[action_str.index("[")+1:action_str.rindex("]")]
        
        if tool_name not in self.tools:
            raise ValueError(f"未知工具：{tool_name}")
        
        tool = self.tools[tool_name]
        return tool.run(query=args_str)
    
    def _extract_answer(self, action_str):
        """从Finish[answer]中提取答案"""
        start = action_str.index("[") + 1
        end = action_str.rindex("]")
        return action_str[start:end]
```

---

## 34.4 多步规划

### 34.4.1 任务分解

**层次化任务分解**：

```python
class HierarchicalPlanner:
    """
    层次化规划器：将复杂任务分解为子任务
    """
    def __init__(self, llm):
        self.llm = llm
    
    def decompose_task(self, task):
        """
        分解任务为子任务
        
        Args:
            task: 原始任务描述
        
        Returns:
            subtasks: 子任务列表
        """
        decompose_prompt = f"""
将以下任务分解为具体的子任务：

任务：{task}

请列出需要完成的子任务（每行一个）：
1.
"""
        
        response = self.llm.generate(decompose_prompt, max_tokens=300)
        
        # 解析子任务
        subtasks = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                # 移除行号
                subtask = line[line.index('.')+1:].strip()
                subtasks.append(subtask)
        
        return subtasks
```

<div data-component="AgentPlanningTree"></div>

### 34.4.2 子目标设定

**动态子目标生成**：

```python
def generate_subgoals(current_state, final_goal, num_subgoals=3):
    """
    生成中间子目标
    
    Args:
        current_state: 当前状态描述
        final_goal: 最终目标
        num_subgoals: 子目标数量
    
    Returns:
        subgoals: 子目标列表
    """
    prompt = f"""
当前状态：{current_state}
最终目标：{final_goal}

请设定{num_subgoals}个中间子目标，帮助逐步达成最终目标：

子目标1:
子目标2:
子目标3:
"""
    
    response = llm.generate(prompt, max_tokens=200)
    
    subgoals = []
    for line in response.split('\n'):
        if line.startswith('子目标'):
            subgoal = line.split(':')[1].strip()
            subgoals.append(subgoal)
    
    return subgoals
```

### 34.4.3 Plan-and-Execute

**规划-执行框架**：

```python
class PlanAndExecuteAgent:
    """
    先规划再执行的Agent
    """
    def __init__(self, llm, tools, executor):
        self.llm = llm
        self.tools = tools
        self.executor = executor
    
    def solve(self, task):
        """
        解决任务
        
        流程：
        1. 生成完整计划
        2. 逐步执行计划
        3. 必要时重新规划
        """
        # 阶段1：规划
        plan = self._create_plan(task)
        print(f"计划：\n{self._format_plan(plan)}\n")
        
        # 阶段2：执行
        results = []
        for i, step in enumerate(plan):
            print(f"执行步骤{i+1}/{len(plan)}: {step}")
            
            result = self._execute_step(step)
            results.append(result)
            
            # 检查是否需要重新规划
            if self._should_replan(step, result):
                print("检测到偏差，重新规划...")
                remaining_task = self._update_task(task, results)
                plan = self._create_plan(remaining_task)
        
        # 综合结果
        final_answer = self._synthesize_results(results)
        return final_answer
    
    def _create_plan(self, task):
        """生成计划"""
        plan_prompt = f"""
任务：{task}

可用工具：{', '.join([t.name for t in self.tools])}

请创建详细的执行计划（每行一个步骤）：
1.
"""
        
        response = self.llm.generate(plan_prompt, max_tokens=400)
        
        plan = []
        for line in response.split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                step = line[line.index('.')+1:].strip()
                plan.append(step)
        
        return plan
    
    def _execute_step(self, step):
        """执行单个计划步骤"""
        return self.executor.execute(step)
    
    def _should_replan(self, step, result):
        """判断是否需要重新规划"""
        # 简单启发式：如果结果包含错误信息
        error_keywords = ["错误", "失败", "未找到"]
        return any(kw in result for kw in error_keywords)
    
    def _update_task(self, original_task, completed_results):
        """更新任务描述（移除已完成部分）"""
        update_prompt = f"""
原始任务：{original_task}

已完成的步骤结果：
{'\n'.join(completed_results)}

请描述剩余需要完成的任务：
"""
        
        return self.llm.generate(update_prompt, max_tokens=200).strip()
    
    def _synthesize_results(self, results):
        """综合所有结果"""
        synthesis_prompt = f"""
将以下步骤结果综合为最终答案：

{'\n\n'.join([f"步骤{i+1}: {r}" for i, r in enumerate(results)])}

最终答案：
"""
        
        return self.llm.generate(synthesis_prompt, max_tokens=300).strip()
    
    def _format_plan(self, plan):
        """格式化计划显示"""
        return '\n'.join([f"{i+1}. {step}" for i, step in enumerate(plan)])
```

<div data-component="MultiStepExecution"></div>

---

## 34.5 实际应用

### 34.5.1 WebGPT

**WebGPT**（Nakano et al., 2021）：能够浏览网页回答问题的Agent。

**核心能力**：
1. **搜索**：使用搜索引擎查找信息
2. **浏览**：点击链接、阅读网页
3. **引用**：引用可靠来源
4. **综合**：整合多个来源的信息

**实现框架**：

```python
class WebGPTAgent:
    """
    WebGPT Agent实现
    """
    def __init__(self, llm, search_engine, browser):
        self.llm = llm
        self.search_engine = search_engine
        self.browser = browser
        self.max_browsing_steps = 10
    
    def answer_question(self, question):
        """回答问题（with web浏览）"""
        context = f"Question: {question}\n\n"
        citations = []
        
        for step in range(self.max_browsing_steps):
            # 决定下一步行动
            action = self._decide_action(context)
            
            if action['type'] == 'search':
                # 搜索
                query = action['query']
                search_results = self.search_engine.search(query)
                context += f"Search[{query}]\nResults: {search_results}\n\n"
            
            elif action['type'] == 'click':
                # 点击链接
                url = action['url']
                page_content = self.browser.get_page(url)
                context += f"Click[{url}]\nContent: {page_content[:500]}...\n\n"
                citations.append(url)
            
            elif action['type'] == 'answer':
                # 生成最终答案
                answer = action['answer']
                return {
                    'answer': answer,
                    'citations': citations,
                    'browsing_steps': step + 1
                }
        
        return {
            'answer': "无法在规定步数内找到答案",
            'citations': citations,
            'browsing_steps': self.max_browsing_steps
        }
    
    def _decide_action(self, context):
        """决定下一步行动"""
        action_prompt = context + """
可选行动：
- Search[query]: 搜索信息
- Click[url]: 浏览网页
- Answer[text]: 给出最终答案

选择行动：
"""
        
        response = self.llm.generate(action_prompt, max_tokens=100)
        
        # 解析行动
        if response.startswith("Search["):
            query = response[7:response.index("]")]
            return {'type': 'search', 'query': query}
        elif response.startswith("Click["):
            url = response[6:response.index("]")]
            return {'type': 'click', 'url': url}
        elif response.startswith("Answer["):
            answer = response[7:response.rindex("]")]
            return {'type': 'answer', 'answer': answer}
        else:
            return {'type': 'search', 'query': 'default query'}
```

### 34.5.2 Toolformer

**Toolformer**（Schick et al., 2023）：自学习工具使用的LLM。

**核心创新**：
- 自动发现何时、何地使用工具
- 通过自我监督学习工具调用

**训练流程**：

```python
def train_toolformer(
    base_model,
    tools,
    training_texts,
    num_epochs=3
):
    """
    Toolformer训练
    
    步骤：
    1. 采样工具调用位置
    2. 评估工具是否有帮助
    3. 过滤有益的工具调用
    4. 微调模型
    """
    augmented_data = []
    
    for text in training_texts:
        # 1. 采样潜在工具调用位置
        positions = sample_positions(text)
        
        for pos in positions:
            for tool in tools:
                # 2. 生成工具调用
                tool_call = generate_tool_call(
                    base_model,
                    text,
                    pos,
                    tool
                )
                
                # 3. 执行工具获取结果
                tool_result = tool.run(tool_call['args'])
                
                # 4. 评估是否有改进
                text_with_tool = insert_tool_call(
                    text,
                    pos,
                    tool_call,
                    tool_result
                )
                
                # 计算困惑度改进
                ppl_without = compute_perplexity(base_model, text)
                ppl_with = compute_perplexity(base_model, text_with_tool)
                
                # 5. 如果有帮助，则保留
                if ppl_with < ppl_without:
                    augmented_data.append(text_with_tool)
    
    # 6. 在增强数据上微调
    fine_tuned_model = fine_tune(base_model, augmented_data)
    
    return fine_tuned_model
```

### 34.5.3 AutoGPT、BabyAGI

**AutoGPT**：完全自主的Agent，能够自我设定目标、规划、执行。

**核心循环**：

```python
class AutoGPT:
    """
    AutoGPT实现：自主Agent
    """
    def __init__(self, llm, tools, memory):
        self.llm = llm
        self.tools = tools
        self.memory = memory  # 长期记忆
    
    def run(self, objective, max_iterations=100):
        """
        自主运行
        
        Args:
            objective: 高层目标
            max_iterations: 最大迭代次数
        """
        for iteration in range(max_iterations):
            # 1. 从记忆中检索相关信息
            relevant_memories = self.memory.retrieve(objective)
            
            # 2. 生成思考
            thought = self._think(objective, relevant_memories)
            print(f"[思考] {thought}")
            
            # 3. 决定行动
            action = self._decide(thought)
            print(f"[行动] {action}")
            
            # 4. 执行行动
            result = self._act(action)
            print(f"[结果] {result}")
            
            # 5. 自我批评
            criticism = self._self_criticize(thought, action, result)
            print(f"[反思] {criticism}")
            
            # 6. 更新记忆
            self.memory.store({
                'thought': thought,
                'action': action,
                'result': result,
                'criticism': criticism
            })
            
            # 7. 检查是否完成目标
            if self._is_objective_met(objective, result):
                print(f"✅ 目标达成！")
                return result
        
        print(f"⚠️ 达到最大迭代次数")
        return None
    
    def _think(self, objective, memories):
        """生成思考"""
        prompt = f"""
目标：{objective}

相关记忆：
{self._format_memories(memories)}

基于当前情况，你的思考是：
"""
        return self.llm.generate(prompt, max_tokens=200)
    
    def _decide(self, thought):
        """决定行动"""
        tools_desc = '\n'.join([
            f"- {t.name}: {t.description}"
            for t in self.tools
        ])
        
        prompt = f"""
思考：{thought}

可用工具：
{tools_desc}

下一步行动：
"""
        return self.llm.generate(prompt, max_tokens=100)
    
    def _act(self, action):
        """执行行动"""
        # 解析并执行
        for tool in self.tools:
            if action.startswith(tool.name):
                args = extract_args(action)
                return tool.run(**args)
        
        return "无法执行该行动"
    
    def_self_criticize(self, thought, action, result):
        """自我批评"""
        prompt = f"""
思考：{thought}
行动：{action}
结果：{result}

批评性地评估：这个行动是否有效？是否有更好的方法？
"""
        return self.llm.generate(prompt, max_tokens=150)
    
    def _is_objective_met(self, objective, result):
        """判断目标是否达成"""
        prompt = f"""
目标：{objective}
当前结果：{result}

目标是否已达成？回答"是"或"否"：
"""
        response = self.llm.generate(prompt, max_tokens=10).strip().lower()
        return '是' in response or 'yes' in response
```

---

## 总结

本章介绍了LLM Agent的核心技术：

1. **ReAct框架**：思考-行动循环，显式推理链
2. **工具学习**：API调用、代码执行、知识库检索
3. **RL优化**：轨迹级奖励、工具选择策略、错误恢复
4. **多步规划**：任务分解、子目标设定、Plan-and-Execute
5. **实际应用**：WebGPT、Toolformer、AutoGPT

**关键要点**：
- ReAct将推理与行动结合，提升可靠性和可解释性
- 工具使用扩展了LLM的能力边界
- RL可以优化Agent的工具选择和规划策略
- 实际应用展示了Agent在复杂任务中的潜力

**未来方向**：
- 更强大的多模态Agent
- 更高效的工具学习方法
- 更可靠的长期规划能力
- 与人类更好的协作

---

## 参考文献

- Yao, S., et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models." *ICLR*.
- Schick, T., et al. (2023). "Toolformer: Language Models Can Teach Themselves to Use Tools." *NeurIPS*.
- Nakano, R., et al. (2021). "WebGPT: Browser-assisted question-answering with human feedback." *arXiv*.
- Significant Gravitas (2023). "AutoGPT." *GitHub*.
- Nakajima, Y. (2023). "BabyAGI." *GitHub*.

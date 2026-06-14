---
title: "第1章：Agent 核心循环 — 感知-决策-执行"
description: "深入理解 Agent 的感知-决策-执行（PDA）循环，掌握状态空间、动作空间与环境建模，理解 Agent 与环境交互的完整数据流，从零构建完整 Agent Step 函数。"
date: "2026-06-11"
---

# 第1章：Agent 核心循环 — 感知-决策-执行

Agent 的核心是一个**循环**——感知环境、做出决策、执行行动、观察结果，然后再次感知。本章将深入剖析这个循环的每一个环节。

---

## 1.1 Agent 的感知-决策-执行循环

### 1.1.1 PDA 循环的完整定义

Agent 的行为可以用一个**感知-决策-执行（Perception-Decision-Action, PDA）**循环来描述：

$$
\text{Step}_t = \text{Action}(\text{Decision}(\text{Perception}(s_t, o_t)))
$$

其中：
- $s_t$：当前 Agent 状态（对话历史、任务进度、中间结果）
- $o_t$：环境观察（工具返回值、用户输入、系统事件）
- $\text{Perception}$：感知函数，将原始输入转化为内部表示
- $\text{Decision}$：决策函数（LLM），基于感知结果选择行动
- $\text{Action}$：执行函数，将决策转化为对环境的实际操作

```
┌─────────────────────────────────────────────────┐
│              Agent PDA 循环                      │
│                                                  │
│    ┌──────────┐                                  │
│    │ 用户输入  │──────┐                           │
│    └──────────┘      │                           │
│                      ▼                           │
│    ┌─────────────────────────────┐               │
│    │      Perception（感知）      │               │
│    │  • 输入解析                  │               │
│    │  • 上下文组装                │               │
│    │  • 历史摘要                  │               │
│    └──────────────┬──────────────┘               │
│                   ▼                              │
│    ┌─────────────────────────────┐               │
│    │      Decision（决策）        │               │
│    │  • LLM 推理                  │               │
│    │  • 工具选择                  │               │
│    │  • 参数构造                  │               │
│    └──────────────┬──────────────┘               │
│                   ▼                              │
│    ┌─────────────────────────────┐               │
│    │      Action（执行）          │               │
│    │  • 工具调用                  │               │
│    │  • 代码执行                  │               │
│    │  • API 请求                  │               │
│    └──────────────┬──────────────┘               │
│                   ▼                              │
│    ┌─────────────────────────────┐               │
│    │      Observation（观察）     │               │
│    │  • 结果解析                  │               │
│    │  • 错误检测                  │               │
│    │  • 状态更新                  │               │
│    └──────────────┬──────────────┘               │
│                   │                              │
│                   ▼                              │
│              ┌──────────┐                        │
│              │ 任务完成？ │──No──► 回到 Perception│
│              └────┬─────┘                        │
│                   │ Yes                          │
│                   ▼                              │
│              ┌──────────┐                        │
│              │ 最终输出  │                        │
│              └──────────┘                        │
└─────────────────────────────────────────────────┘
```

### 1.1.2 与强化学习的 Agent-Environment 循环对比

| 维度 | RL Agent | LLM Agent |
|:---|:---|:---|
| **决策依据** | 策略网络 $\pi_\theta(a|s)$ | LLM 推理 $P(a|context)$ |
| **状态表示** | 向量/图像 | 自然语言文本 |
| **动作空间** | 离散/连续 | 自然语言 + 工具调用 |
| **学习方式** | 在线学习（试错） | 上下文学习（In-Context Learning） |
| **奖励信号** | 稀疏标量奖励 | 任务完成反馈 |
| **探索策略** | $\epsilon$-greedy, UCB | 多次采样 + Self-Consistency |
| **环境交互** | 环境状态转移 $P(s'|s,a)$ | 工具返回观察 |

### 1.1.3 状态空间、动作空间与观察空间

```python
from dataclasses import dataclass, field
from typing import Any
from langchain_core.messages import BaseMessage

@dataclass
class AgentState:
    """Agent 的完整状态"""
    # 消息历史
    messages: list[BaseMessage] = field(default_factory=list)

    # 当前任务
    task: str = ""

    # 执行计划
    plan: list[str] = field(default_factory=list)
    current_step: int = 0

    # 工具调用历史
    tool_calls_history: list[dict] = field(default_factory=list)

    # 中间结果
    intermediate_results: dict[str, Any] = field(default_factory=dict)

    # 迭代控制
    iteration_count: int = 0
    max_iterations: int = 15

    # 错误记录
    errors: list[str] = field(default_factory=list)

    # Agent 思考草稿纸
    scratchpad: str = ""

    # 自定义字段
    context: dict = field(default_factory=dict)
```

---

## 1.2 感知（Perception）

### 1.2.1 用户输入解析

```python
from langchain_core.messages import HumanMessage, SystemMessage

def perceive_user_input(
    user_input: str,
    agent_state: AgentState,
    system_prompt: str
) -> list[BaseMessage]:
    """感知层：将用户输入与当前状态组装为 LLM 输入"""
    messages = []

    # 1. 系统提示词（定义 Agent 的角色和行为规范）
    messages.append(SystemMessage(content=system_prompt))

    # 2. 历史对话（根据 Token 预算裁剪）
    history_messages = agent_state.messages
    budget_messages = manage_token_budget(
        messages=history_messages,
        max_tokens=8000,
        preserve_recent=6
    )
    messages.extend(budget_messages)

    # 3. 当前用户输入
    messages.append(HumanMessage(content=user_input))

    # 4. 如果有中间结果或草稿纸，附加到上下文
    if agent_state.scratchpad:
        messages.append(SystemMessage(
            content=f"[Scratchpad]\n{agent_state.scratchpad}"
        ))

    return messages


def manage_token_budget(
    messages: list[BaseMessage],
    max_tokens: int,
    preserve_recent: int = 6
) -> list[BaseMessage]:
    """Token 预算管理：在有限窗口内最大化上下文价值"""
    import tiktoken

    encoder = tiktoken.encoding_for_model("gpt-4o")

    # 计算每条消息的 Token 数
    token_counts = []
    for msg in messages:
        tokens = len(encoder.encode(msg.content or ""))
        token_counts.append(tokens)

    total_tokens = sum(token_counts)

    # 如果总 Token 数在预算内，直接返回
    if total_tokens <= max_tokens:
        return messages

    # 保留最近的消息
    recent_messages = messages[-preserve_recent:]
    recent_tokens = sum(token_counts[-preserve_recent:])

    # 对更早的消息进行摘要
    older_messages = messages[:-preserve_recent]
    older_tokens = total_tokens - recent_tokens

    if older_tokens > max_tokens - recent_tokens:
        summary = summarize_messages(older_messages)
        return [SystemMessage(content=f"[对话历史摘要]\n{summary}")] + recent_messages

    return messages


def summarize_messages(messages: list[BaseMessage]) -> str:
    """对早期消息生成摘要"""
    contents = []
    for msg in messages:
        role = msg.type
        content = (msg.content or "")[:200]
        contents.append(f"{role}: {content}")
    return "\n".join(contents)
```

### 1.2.2 环境观察处理

```python
def perceive_tool_result(
    tool_name: str,
    raw_result: Any,
    max_content_length: int = 4000
) -> str:
    """感知层：处理工具返回的原始结果"""
    # 1. 类型转换
    if isinstance(raw_result, dict):
        import json
        content = json.dumps(raw_result, ensure_ascii=False, indent=2)
    elif isinstance(raw_result, list):
        import json
        content = json.dumps(raw_result, ensure_ascii=False, indent=2)
    else:
        content = str(raw_result)

    # 2. 长度截断
    if len(content) > max_content_length:
        content = content[:max_content_length] + f"\n... (截断，共 {len(content)} 字符)"

    return f"[工具 {tool_name} 的返回结果]\n{content}"
```

### 1.2.3 多模态感知

```python
from langchain_core.messages import HumanMessage
import base64

def perceive_multimodal_input(
    text: str = None,
    image_path: str = None,
    audio_path: str = None
) -> HumanMessage:
    """多模态感知：统一处理文本、图像、音频输入"""
    content = []

    if text:
        content.append({"type": "text", "text": text})

    if image_path:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
        })

    return HumanMessage(content=content)
```

---

## 1.3 决策（Decision / Reasoning）

### 1.3.1 推理策略分类

**直觉响应（System 1 Thinking）**：

```python
# 直觉响应：LLM 直接给出答案
user_query = "什么是 Python？"
# LLM 直接回答，无需工具调用
# 延迟低、Token 少、可靠性高
```

**链式推理（Chain-of-Thought, System 2 Thinking）**：

```python
# Chain-of-Thought 示例
# Thought 1: 这是复利计算问题
# Thought 2: 复利公式 A = P(1+r)^n
# Thought 3: P=10000, r=0.08, n=10
# Thought 4: A = 10000 * (1.08)^10 = 21589.25
```

**树状搜索（Tree of Thoughts）**：

```python
# Tree of Thoughts 示例
# Path 1: Kafka 方案 → 优点：高吞吐 → 缺点：运维复杂
# Path 2: RabbitMQ 方案 → 优点：易用 → 缺点：性能瓶颈
# Path 3: Pulsar 方案 → 优点：云原生 → 缺点：生态较小
# 评估后选择 Path 1 并详细设计
```

### 1.3.2 决策的不确定性处理

```python
def handle_uncertainty(
    decision: dict,
    confidence_threshold: float = 0.8
) -> dict:
    """处理决策不确定性"""
    confidence = decision.get("confidence", 0.5)

    if confidence >= confidence_threshold:
        return decision
    elif confidence >= 0.5:
        return {
            "type": "confirm_required",
            "suggested_action": decision,
            "message": f"我不太确定（置信度 {confidence:.0%}），是否继续执行？"
        }
    else:
        return {
            "type": "resample",
            "original_decision": decision,
            "strategy": "self_consistency",
            "sample_count": 3
        }
```

---

## 1.4 执行（Action / Execution）

### 1.4.1 工具调用执行完整流程

```python
import asyncio

class ToolExecutor:
    """工具执行器"""

    def __init__(self, tools: list):
        self.tool_map = {tool.name: tool for tool in tools}

    async def execute(
        self,
        tool_name: str,
        arguments: dict,
        timeout: float = 30.0
    ) -> dict:
        """执行单个工具调用"""
        tool = self.tool_map.get(tool_name)
        if not tool:
            return {
                "status": "error",
                "error": f"工具 '{tool_name}' 不存在。可用工具：{list(self.tool_map.keys())}"
            }

        try:
            result = await asyncio.wait_for(
                tool.ainvoke(arguments),
                timeout=timeout
            )
            return {
                "status": "success",
                "tool_name": tool_name,
                "result": result
            }
        except asyncio.TimeoutError:
            return {"status": "error", "error": f"工具 '{tool_name}' 执行超时（{timeout}秒）"}
        except Exception as e:
            return {"status": "error", "error": f"工具 '{tool_name}' 执行异常：{type(e).__name__}: {str(e)}"}

    async def execute_parallel(self, tool_calls: list) -> list[dict]:
        """并行执行多个工具调用"""
        tasks = [self.execute(tc["name"], tc["args"]) for tc in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [
            {"status": "error", "error": str(r)} if isinstance(r, Exception) else r
            for r in results
        ]
```

### 1.4.2 代码执行与沙箱

```python
class CodeSandbox:
    """代码执行沙箱"""

    def __init__(self, timeout: int = 30, max_output_length: int = 10000):
        self.timeout = timeout
        self.max_output_length = max_output_length

    def execute_python(self, code: str) -> dict:
        """在沙箱中执行 Python 代码"""
        import subprocess
        import tempfile
        import os

        # 安全检查
        dangerous_patterns = [
            "import os", "import subprocess", "import shutil",
            "os.system", "os.remove", "__import__", "eval(", "exec(",
        ]
        for pattern in dangerous_patterns:
            if pattern in code:
                return {"status": "error", "error": f"安全限制：禁止操作 '{pattern}'"}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            result = subprocess.run(
                ["python3", temp_path],
                capture_output=True, text=True, timeout=self.timeout
            )
            output = result.stdout + (f"\n[STDERR]\n{result.stderr}" if result.stderr else "")
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length] + "\n... (已截断)"
            return {
                "status": "success" if result.returncode == 0 else "error",
                "output": output
            }
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": f"执行超时（{self.timeout}秒）"}
        finally:
            os.unlink(temp_path)
```

### 1.4.3 错误处理与重试机制

```python
import time

class RetryHandler:
    """重试处理器"""

    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def should_retry(self, error: str, attempt: int) -> bool:
        if attempt >= self.max_retries:
            return False
        retryable = ["TimeoutError", "ConnectionError", "RateLimitError"]
        return any(e in error for e in retryable)

    async def execute_with_retry(self, func, *args, **kwargs):
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                if isinstance(result, dict) and result.get("status") == "error":
                    if self.should_retry(result.get("error", ""), attempt):
                        last_error = result["error"]
                        time.sleep(self.backoff_factor * (2 ** attempt))
                        continue
                return result
            except Exception as e:
                last_error = str(e)
                if self.should_retry(last_error, attempt):
                    time.sleep(self.backoff_factor * (2 ** attempt))
                    continue
                raise
        return {"status": "error", "error": f"重试失败：{last_error}"}
```

---

## 1.5 观察与反馈（Observation）

### 1.5.1 工具输出的结构化解析

```python
from langchain_core.messages import ToolMessage

def create_tool_observation(
    tool_call_id: str,
    tool_name: str,
    result: dict
) -> ToolMessage:
    """将工具执行结果转化为 LLM 可理解的观察消息"""
    if result["status"] == "success":
        content = f"[{tool_name} 执行成功]\n{result['result']}"
    else:
        content = f"[{tool_name} 执行失败]\n错误：{result['error']}"
    return ToolMessage(content=content, tool_call_id=tool_call_id)
```

### 1.5.2 观察压缩策略

```python
def compress_observation(
    observation: str,
    max_length: int = 2000,
    strategy: str = "truncate"
) -> str:
    """压缩观察结果"""
    if len(observation) <= max_length:
        return observation

    if strategy == "truncate":
        return observation[:max_length] + "\n... (已截断)"

    elif strategy == "head_tail":
        head_len = max_length // 3
        tail_len = max_length // 3
        return (
            observation[:head_len]
            + f"\n\n... (省略 {len(observation) - head_len - tail_len} 字符) ...\n\n"
            + observation[-tail_len:]
        )

    return observation[:max_length]
```

### 1.5.3 反馈循环的终止条件

```python
def should_terminate(state: AgentState, last_decision: dict) -> tuple[bool, str]:
    """判断 Agent 循环是否应该终止"""
    # 任务完成
    if last_decision["type"] == "final_answer":
        return True, "task_completed"

    # 达到最大迭代次数
    if state.iteration_count >= state.max_iterations:
        return True, "max_iterations_reached"

    # 连续错误过多
    consecutive_errors = 0
    for error in reversed(state.errors):
        if "error" in error.lower():
            consecutive_errors += 1
        else:
            break
    if consecutive_errors >= 3:
        return True, "too_many_consecutive_errors"

    return False, "continue"
```

---

## 1.6 Agent 状态管理

### 1.6.1 消息历史管理

```python
from langchain_core.messages import BaseMessage
import tiktoken

class MessageHistoryManager:
    """消息历史管理器"""

    def __init__(self, max_messages: int = 50, max_tokens: int = 16000):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.messages: list[BaseMessage] = []

    def add(self, message: BaseMessage):
        self.messages.append(message)
        self._trim()

    def get_messages(self) -> list[BaseMessage]:
        return self.messages.copy()

    def _trim(self):
        """裁剪消息历史"""
        if len(self.messages) <= self.max_messages:
            return

        system_msgs = [m for m in self.messages if m.type == "system"]
        other_msgs = [m for m in self.messages if m.type != "system"]

        recent = other_msgs[-(self.max_messages - len(system_msgs)):]

        old_msgs = other_msgs[:-(self.max_messages - len(system_msgs))]
        if old_msgs:
            summary = self._create_summary(old_msgs)
            self.messages = system_msgs + [
                SystemMessage(content=f"[早期对话摘要]\n{summary}")
            ] + recent
        else:
            self.messages = system_msgs + recent

    def _create_summary(self, messages: list[BaseMessage]) -> str:
        summary_parts = []
        for msg in messages:
            role = msg.type
            content = (msg.content or "")[:100]
            summary_parts.append(f"- {role}: {content}")
        return "\n".join(summary_parts)
```

---

## 1.7 完整 Agent Step 实现

### 1.7.1 从零构建 Agent Loop

```python
"""完整 Agent 实现：从零构建 PDA 循环"""
import openai
import json
import re

class SimpleAgent:
    """简单但完整的 Agent 实现"""

    def __init__(self, model="gpt-4o", tools=None, max_iterations=10, verbose=True):
        self.client = openai.OpenAI()
        self.model = model
        self.tools = tools or {}
        self.max_iterations = max_iterations
        self.verbose = verbose

    def run(self, user_message: str) -> str:
        messages = [
            {"role": "system", "content": "你是一个有用的 AI 助手，可以使用工具来完成任务。"},
            {"role": "user", "content": user_message}
        ]

        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n{'='*50}")
                print(f"🔄 Agent Step {iteration + 1}")

            # 调用 LLM（决策）
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[t["schema"] for t in self.tools.values()] if self.tools else None,
                tool_choice="auto",
                temperature=0,
                max_tokens=4096,
            )

            assistant_message = response.choices[0].message
            messages.append(assistant_message.model_dump())

            # 检查是否需要执行工具
            if not assistant_message.tool_calls:
                if self.verbose:
                    print(f"✅ 任务完成：直接回答")
                return assistant_message.content

            # 执行工具
            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                if self.verbose:
                    print(f"  🔧 {func_name}({func_args})")

                # 执行工具
                try:
                    func = self.tools[func_name]["func"]
                    result = func(**func_args)
                except Exception as e:
                    result = f"错误：{type(e).__name__}: {str(e)}"

                if self.verbose:
                    print(f"  📋 结果：{str(result)[:200]}...")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })

        return "达到最大迭代次数，任务未完成。"
```

### 1.7.2 使用示例

```python
# 创建 Agent
agent = SimpleAgent(model="gpt-4o", verbose=True)

# 注册工具
def search_knowledge(query: str) -> str:
    """模拟知识搜索"""
    knowledge = {
        "Python": "Python 由 Guido van Rossum 于 1991 年创建。",
        "AI Agent": "AI Agent 是能自主感知环境、做出决策的智能实体。",
    }
    for key, value in knowledge.items():
        if key.lower() in query.lower():
            return value
    return f"未找到关于 '{query}' 的信息"

def calculate(expression: str) -> str:
    """数学计算"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误：{e}"

agent.tools = {
    "search_knowledge": {
        "func": search_knowledge,
        "schema": {
            "type": "function",
            "function": {
                "name": "search_knowledge",
                "description": "搜索知识库获取信息",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
            }
        }
    },
    "calculate": {
        "func": calculate,
        "schema": {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "执行数学计算",
                "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
            }
        }
    }
}

# 运行
result = agent.run("Python 是谁创建的？请计算 2026-1991")
print(f"最终结果: {result}")
```

---

## 1.8 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| PDA 循环 | Agent 的核心是感知-决策-执行循环 |
| 与 RL 对比 | LLM Agent 使用上下文学习而非在线学习 |
| 感知层 | 负责输入解析、上下文组装、Token 预算管理 |
| 决策层 | LLM 作为推理引擎，选择工具并生成参数 |
| 执行层 | 工具调用、代码执行、错误处理、并行执行 |
| 观察层 | 结果解析、压缩、错误检测、终止条件判断 |
| 状态管理 | 消息历史、序列化/恢复、Token 预算控制 |
| 完整实现 | 从零构建 SimpleAgent，展示 PDA 循环全貌 |

> **下一章预告**
>
> 在第 2 章中，我们将深入 Agent 的"大脑"——大语言模型，理解 Transformer 架构、Token 机制、模型参数对 Agent 行为的影响。

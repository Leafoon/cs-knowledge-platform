---
title: "第19章：OpenAI Agents SDK"
description: "掌握 OpenAI Assistants API、Agents SDK 核心概念、Handoff 机制与 Guardrails 安全护栏"
updated: "2025-06-15"
---

 # 第19章：OpenAI Agents SDK

 > **学习目标**：
 > - 理解 OpenAI Assistants API 的完整流程
 > - 掌握 Agents SDK 的核心概念与架构
 > - 学会使用 Handoff 机制实现任务委派
 > - 熟练掌握 Guardrails 安全护栏配置
 > - 对比 Assistants API 与 Agents SDK 的差异
 > - 构建生产级的 AI Agent 应用

 下面的交互式演示展示了 Agent Handoff 机制：

 <div data-component="OpenAIAgentsHandoff"></div>

 ---

 ## 19.1 OpenAI Assistants API 概述

### 19.1.1 什么是 Assistants API

OpenAI Assistants API 是 OpenAI 提供的高级 API，用于构建具有持久对话、文件检索和代码执行能力的 AI 助手。它简化了构建复杂 AI 应用的流程。

> **核心思想**：通过创建持久化的 Assistant 实现多轮对话，支持文件检索、代码执行和函数调用。

### 19.1.2 Assistants API 架构

```
┌─────────────────────────────────────────────────────────┐
│                  Assistants API                          │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Assistant Layer                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────┐  │   │
│  │  │  Assistant  │  │    Thread   │  │  Run    │  │   │
│  │  │  (助手)     │  │  (线程)     │  │ (运行)  │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────┘  │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │            Capability Layer                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────┐  │   │
│  │  │  File       │  │  Code       │  │ Function│  │   │
│  │  │  Retrieval  │  │  Interpreter│  │ Calling │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────┘  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 19.1.3 核心概念速览

| 概念 | 定义 | 类比 |
|------|------|------|
| **Assistant** | AI 助手实例 | 虚拟助手 |
| **Thread** | 对话线程 | 会话记录 |
| **Message** | 消息内容 | 聊天消息 |
| **Run** | 执行过程 | API 调用 |
| **Tool** | 工具集 | 功能模块 |
| **File** | 上传文件 | 附件 |

---

## 19.2 Assistants API 完整流程

### 19.2.1 基础使用流程

```python
from openai import OpenAI

# 初始化客户端
client = OpenAI(api_key="your-api-key")

# 步骤1：创建 Assistant
assistant = client.beta.assistants.create(
    name="AI 助手",
    instructions="你是一个有帮助的 AI 助手，可以回答问题并提供专业建议。",
    model="gpt-4",
    tools=[
        {"type": "code_interpreter"},
        {"type": "file_search"},
    ],
)

print(f"Assistant ID: {assistant.id}")

# 步骤2：创建 Thread（对话线程）
thread = client.beta.threads.create()

print(f"Thread ID: {thread.id}")

# 步骤3：添加消息到 Thread
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="请解释什么是 AI Agent？",
)

print(f"Message ID: {message.id}")

# 步骤4：创建 Run（执行）
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)

print(f"Run ID: {run.id}")

# 步骤5：等待完成并获取结果
import time
while run.status in ["queued", "in_progress"]:
    time.sleep(1)
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id,
    )

# 步骤6：获取回复
messages = client.beta.threads.messages.list(thread_id=thread.id)
for msg in messages.data:
    if msg.role == "assistant":
        print(f"Assistant: {msg.content[0].text.value}")
```

### 19.2.2 文件检索功能

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

# 步骤1：上传文件
file = client.files.create(
    file=open("knowledge_base.pdf", "rb"),
    purpose="assistants",
)

print(f"File ID: {file.id}")

# 步骤2：创建带文件检索的 Assistant
assistant = client.beta.assistants.create(
    name="文档助手",
    instructions="你是文档分析专家，可以基于上传的文档回答问题。",
    model="gpt-4",
    tools=[
        {"type": "file_search"},
    ],
    tool_resources={
        "file_search": {
            "vector_store_ids": ["vs_xxx"]  # 需要先创建 vector store
        }
    },
)

# 步骤3：创建 Vector Store
vector_store = client.beta.vector_stores.create(
    name="知识库",
    file_ids=[file.id],
)

print(f"Vector Store ID: {vector_store.id}")

# 步骤4：更新 Assistant 使用新的 Vector Store
assistant = client.beta.assistants.update(
    assistant_id=assistant.id,
    tool_resources={
        "file_search": {
            "vector_store_ids": [vector_store.id]
        }
    },
)

# 步骤5：对话
thread = client.beta.threads.create()
client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="文档中提到了哪些关键概念？",
)

run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)
```

### 19.2.3 代码执行功能

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

# 创建带代码执行的 Assistant
assistant = client.beta.assistants.create(
    name="数据分析师",
    instructions="你是数据分析师，可以编写和执行 Python 代码来分析数据。",
    model="gpt-4",
    tools=[
        {"type": "code_interpreter"},
    ],
)

# 上传数据文件
file = client.files.create(
    file=open("data.csv", "rb"),
    purpose="assistants",
)

# 创建 Thread 并上传文件
thread = client.beta.threads.create(
    messages=[
        {
            "role": "user",
            "content": "请分析这个数据集并生成可视化图表。",
            "attachments": [
                {
                    "file_id": file.id,
                    "tools": [{"type": "code_interpreter"}],
                }
            ],
        }
    ],
)

# 执行
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)

# 等待完成
import time
while run.status in ["queued", "in_progress"]:
    time.sleep(1)
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id,
    )

# 获取结果
messages = client.beta.threads.messages.list(thread_id=thread.id)
for msg in messages.data:
    if msg.role == "assistant":
        for content in msg.content:
            if content.type == "text":
                print(f"文本: {content.text.value}")
            elif content.type == "image_file":
                print(f"图片: {content.image_file.file_id}")
            elif content.type == "code_interpreter_output":
                print(f"代码输出: {content.code_interpreter_output}")
```

### 19.2.4 函数调用功能

```python
from openai import OpenAI
import json

client = OpenAI(api_key="your-api-key")

# 定义函数
def get_weather(city: str) -> str:
    """获取天气信息"""
    # 模拟天气数据
    weather_data = {
        "北京": "晴天，25°C",
        "上海": "多云，22°C",
        "广州": "小雨，28°C",
    }
    return weather_data.get(city, f"未找到 {city} 的天气信息")

# 创建带函数的 Assistant
assistant = client.beta.assistants.create(
    name="天气助手",
    instructions="你是天气助手，可以帮助用户查询天气信息。",
    model="gpt-4",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称",
                        }
                    },
                    "required": ["city"],
                },
            },
        }
    ],
)

# 创建 Thread
thread = client.beta.threads.create()
client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="北京今天天气怎么样？",
)

# 执行
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id,
)

# 处理函数调用
import time
while run.status in ["queued", "in_progress"]:
    time.sleep(1)
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id,
    )
    
    # 检查是否有函数调用
    if run.status == "requires_action":
        tool_outputs = []
        for call in run.required_action.submit_tool_outputs.tool_calls:
            if call.function.name == "get_weather":
                args = json.loads(call.function.arguments)
                result = get_weather(args["city"])
                tool_outputs.append({
                    "tool_call_id": call.id,
                    "output": result,
                })
        
        # 提交函数结果
        run = client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=run.id,
            tool_outputs=tool_outputs,
        )

# 获取回复
messages = client.beta.threads.messages.list(thread_id=thread.id)
for msg in messages.data:
    if msg.role == "assistant":
        print(f"Assistant: {msg.content[0].text.value}")
```

---

## 19.3 Agents SDK 核心概念

### 19.3.1 什么是 Agents SDK

OpenAI Agents SDK 是 OpenAI 官方推出的 Python SDK，专为构建多 Agent 应用而设计。它提供了更简洁的 API 和更强大的功能，支持 Agent 之间的协作和任务委派。

> **核心思想**：通过声明式的方式定义 Agent，支持 Handoff 机制实现任务委派，内置 Guardrails 确保安全。

### 19.3.2 Agents SDK 安装与配置

```bash
# 安装 Agents SDK
pip install openai-agents

# 或从源码安装
pip install git+https://github.com/openai/openai-agents-python.git
```

```python
from agents import Agent, Runner, function_tool

# 配置 API Key
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"
```

### 19.3.3 基础 Agent 创建

```python
from agents import Agent, Runner

# 创建基础 Agent
assistant = Agent(
    name="AI 助手",
    instructions="你是一个有帮助的 AI 助手，可以回答各种问题。",
    model="gpt-4",
)

# 使用 Runner 执行
async def main():
    result = await Runner.run(
        assistant,
        messages=[{"role": "user", "content": "你好，请介绍一下自己"}],
    )
    print(result)

import asyncio
asyncio.run(main())
```

### 19.3.4 Agent 高级配置

```python
from agents import Agent, Runner, function_tool
from typing import List

# 创建带工具的 Agent
@function_tool
def search_web(query: str) -> str:
    """搜索网络信息"""
    return f"搜索结果: 关于 '{query}' 的信息..."

@function_tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"

# 创建 Agent
research_agent = Agent(
    name="研究员",
    instructions="""你是一个研究员，负责：
    1. 搜索相关信息
    2. 分析数据
    3. 提供研究结论""",
    model="gpt-4",
    tools=[search_web, calculate],
    handoffs=["writer_agent"],  # 可以委派给 writer_agent
)

writer_agent = Agent(
    name="写手",
    instructions="""你是一个写手，负责：
    1. 撰写研究报告
    2. 整理信息
    3. 输出文档""",
    model="gpt-4",
)

# 使用
async def main():
    result = await Runner.run(
        research_agent,
        messages=[{"role": "user", "content": "研究 AI Agent 的最新进展并撰写报告"}],
    )
    print(result)

import asyncio
asyncio.run(main())
```

---

## 19.4 Handoff 机制

### 19.4.1 基础 Handoff

```python
from agents import Agent, Runner

# 创建多个 Agent
triage_agent = Agent(
    name="调度员",
    instructions="""你是调度员，负责将用户请求分配给合适的 Agent。
    
    分配规则：
    - 技术问题 -> 技术专家
    - 业务问题 -> 业务专家
    - 一般问题 -> 通用助手""",
    model="gpt-4",
    handoffs=["tech_expert", "business_expert", "general_assistant"],
)

tech_expert = Agent(
    name="技术专家",
    instructions="你是技术专家，负责处理技术问题。",
    model="gpt-4",
)

business_expert = Agent(
    name="业务专家",
    instructions="你是业务专家，负责处理业务问题。",
    model="gpt-4",
)

general_assistant = Agent(
    name="通用助手",
    instructions="你是通用助手，负责处理一般问题。",
    model="gpt-4",
)

# 使用
async def main():
    result = await Runner.run(
        triage_agent,
        messages=[{"role": "user", "content": "如何优化 Python 代码性能？"}],
    )
    print(f"处理 Agent: {result.last_agent}")
    print(f"回复: {result.final_output}")

import asyncio
asyncio.run(main())
```

### 19.4.2 高级 Handoff 配置

```python
from agents import Agent, Runner, Handoff
from typing import List

# 创建带条件的 Handoff
def should_handoff_to_tech(context) -> bool:
    """判断是否应该委派给技术专家"""
    last_message = context.messages[-1]["content"] if context.messages else ""
    tech_keywords = ["代码", "编程", "Python", "API", "数据库"]
    return any(keyword in last_message for keyword in tech_keywords)

# 创建 Agent
triage_agent = Agent(
    name="智能调度员",
    instructions="你是智能调度员，根据用户需求分配任务。",
    model="gpt-4",
    handoffs=[
        Handoff(
            target_agent=tech_expert,
            condition=should_handoff_to_tech,
            description="当用户询问技术问题时委派",
        ),
        Handoff(
            target_agent=business_expert,
            condition=lambda ctx: "业务" in ctx.messages[-1].get("content", ""),
            description="当用户询问业务问题时委派",
        ),
    ],
)

# 带上下文的 Handoff
class HandoffContext:
    def __init__(self, user_id: str, session_id: str):
        self.user_id = user_id
        self.session_id = session_id

async def main():
    context = HandoffContext(user_id="user_123", session_id="session_456")
    
    result = await Runner.run(
        triage_agent,
        messages=[{"role": "user", "content": "如何部署微服务？"}],
        context=context,
    )
    print(result)

import asyncio
asyncio.run(main())
```

### 19.4.3 Handoff 链式调用

```python
from agents import Agent, Runner

# 创建 Agent 链
researcher = Agent(
    name="研究员",
    instructions="你负责收集和分析信息。",
    model="gpt-4",
    handoffs=["analyst"],
)

analyst = Agent(
    name="分析师",
    instructions="你负责分析数据并提供洞察。",
    model="gpt-4",
    handoffs=["writer"],
)

writer = Agent(
    name="写手",
    instructions="你负责撰写报告。",
    model="gpt-4",
    handoffs=[],  # 终点 Agent
)

# 使用
async def main():
    # 链式调用：researcher -> analyst -> writer
    result = await Runner.run(
        researcher,
        messages=[{"role": "user", "content": "分析 AI 市场趋势并撰写报告"}],
    )
    
    # 查看调用链
    for step in result.steps:
        print(f"Agent: {step.agent.name}")
        print(f"输出: {step.output[:100]}...")
        print("---")

import asyncio
asyncio.run(main())
```

---

## 19.5 Guardrails 安全护栏

### 19.5.1 基础 Guardrail

```python
from agents import Agent, Runner, InputGuardrail, GuardrailFunctionOutput
from pydantic import BaseModel

# 定义 Guardrail 输出
class SafetyCheck(BaseModel):
    is_safe: bool
    reason: str

# 创建 Guardrail 函数
@InputGuardrail
async def safety_guardrail(context, agent):
    """安全检查 Guardrail"""
    user_message = context.messages[-1]["content"] if context.messages else ""
    
    # 检查是否包含敏感信息
    sensitive_patterns = ["密码", "password", "信用卡", "身份证"]
    
    for pattern in sensitive_patterns:
        if pattern.lower() in user_message.lower():
            return GuardrailFunctionOutput(
                output_info=SafetyCheck(
                    is_safe=False,
                    reason=f"包含敏感信息: {pattern}"
                ),
                tripwire_triggered=True,
            )
    
    return GuardrailFunctionOutput(
        output_info=SafetyCheck(
            is_safe=True,
            reason="安全检查通过"
        ),
        tripwire_triggered=False,
    )

# 创建带 Guardrail 的 Agent
safe_agent = Agent(
    name="安全助手",
    instructions="你是一个安全的助手。",
    model="gpt-4",
    input_guardrails=[safety_guardrail],
)

# 使用
async def main():
    # 正常请求
    result = await Runner.run(
        safe_agent,
        messages=[{"role": "user", "content": "什么是 AI？"}],
    )
    print(f"正常回复: {result.final_output}")
    
    # 敏感请求（会触发 Guardrail）
    try:
        result = await Runner.run(
            safe_agent,
            messages=[{"role": "user", "content": "我的密码是什么？"}],
        )
    except Exception as e:
        print(f"Guardrail 触发: {e}")

import asyncio
asyncio.run(main())
```

### 19.5.2 高级 Guardrail 配置

```python
from agents import Agent, Runner, InputGuardrail, OutputGuardrail, GuardrailFunctionOutput
from pydantic import BaseModel
from typing import List

# 1. 内容安全 Guardrail
class ContentSafety(BaseModel):
    is_appropriate: bool
    category: str

@InputGuardrail
async def content_safety_guardrail(context, agent):
    """内容安全检查"""
    user_message = context.messages[-1]["content"]
    
    # 检查不当内容
    inappropriate_content = ["暴力", "色情", "仇恨"]
    
    for content in inappropriate_content:
        if content in user_message:
            return GuardrailFunctionOutput(
                output_info=ContentSafety(
                    is_appropriate=False,
                    category=content
                ),
                tripwire_triggered=True,
            )
    
    return GuardrailFunctionOutput(
        output_info=ContentSafety(
            is_appropriate=True,
            category="safe"
        ),
        tripwire_triggered=False,
    )

# 2. 输出质量 Guardrail
class OutputQuality(BaseModel):
    quality_score: float
    issues: List[str]

@OutputGuardrail
async def output_quality_guardrail(context, agent, output):
    """输出质量检查"""
    issues = []
    
    # 检查输出长度
    if len(output) < 10:
        issues.append("输出过短")
    
    # 检查是否包含错误信息
    error_indicators = ["错误", "失败", "无法"]
    for indicator in error_indicators:
        if indicator in output:
            issues.append(f"包含错误指示: {indicator}")
    
    quality_score = 1.0 - (len(issues) * 0.2)
    
    return GuardrailFunctionOutput(
        output_info=OutputQuality(
            quality_score=max(0, quality_score),
            issues=issues
        ),
        tripwire_triggered=quality_score < 0.5,
    )

# 创建带多个 Guardrail 的 Agent
quality_agent = Agent(
    name="高质量助手",
    instructions="你是一个提供高质量回复的助手。",
    model="gpt-4",
    input_guardrails=[content_safety_guardrail],
    output_guardrails=[output_quality_guardrail],
)

# 使用
async def main():
    result = await Runner.run(
        quality_agent,
        messages=[{"role": "user", "content": "解释量子计算"}],
    )
    print(f"回复: {result.final_output}")

import asyncio
asyncio.run(main())
```

### 19.5.3 自定义 Guardrail 类

```python
from agents import Agent, Runner, InputGuardrail, GuardrailFunctionOutput
from pydantic import BaseModel
from typing import Any

class CustomGuardrail:
    """自定义 Guardrail 类"""
    
    def __init__(self, name: str, rules: dict):
        self.name = name
        self.rules = rules
        self.violations = []
    
    async def check(self, context, agent) -> GuardrailFunctionOutput:
        """执行检查"""
        user_message = context.messages[-1]["content"] if context.messages else ""
        
        # 执行所有规则
        for rule_name, rule_func in self.rules.items():
            is_valid, reason = rule_func(user_message)
            
            if not is_valid:
                self.violations.append({
                    "rule": rule_name,
                    "reason": reason,
                    "message": user_message,
                })
                
                return GuardrailFunctionOutput(
                    output_info={"rule": rule_name, "reason": reason},
                    tripwire_triggered=True,
                )
        
        return GuardrailFunctionOutput(
            output_info={"status": "passed"},
            tripwire_triggered=False,
        )
    
    def get_violations(self) -> list:
        """获取违规记录"""
        return self.violations

# 使用示例
def no_profanity(text: str) -> tuple[bool, str]:
    """检查是否包含脏话"""
    bad_words = ["badword1", "badword2"]
    for word in bad_words:
        if word in text.lower():
            return False, f"包含不当词汇: {word}"
    return True, ""

def max_length(text: str) -> tuple[bool, str]:
    """检查最大长度"""
    if len(text) > 1000:
        return False, f"输入过长: {len(text)} 字符"
    return True, ""

# 创建 Guardrail
custom_guardrail = CustomGuardrail(
    name="自定义安全检查",
    rules={
        "no_profanity": no_profanity,
        "max_length": max_length,
    }
)

# 创建 Agent
agent = Agent(
    name="受保护的助手",
    instructions="你是一个受保护的助手。",
    model="gpt-4",
    input_guardrails=[custom_guardrail.check],
)
```

---

## 19.6 Assistants API vs Agents SDK 对比

### 19.6.1 核心差异

| 特性 | Assistants API | Agents SDK |
|------|----------------|------------|
| **设计目标** | 持久化助手 | 多 Agent 协作 |
| **状态管理** | 服务端持久化 | 客户端管理 |
| **对话模型** | Thread + Run | 直接消息 |
| **文件处理** | 内置支持 | 通过工具 |
| **代码执行** | 内置沙箱 | 通过工具 |
| **Handoff** | 不支持 | 内置支持 |
| **Guardrails** | 不支持 | 内置支持 |
| **学习曲线** | 中等 | 较低 |
| **灵活性** | 中等 | 高 |

### 19.6.2 选择指南

```python
# Assistants API 适用场景
assistants_scenarios = [
    "需要持久化对话状态",
    "文件检索和分析",
    "代码执行和调试",
    "简单的单 Agent 应用",
    "快速原型开发",
]

# Agents SDK 适用场景
agents_sdk_scenarios = [
    "多 Agent 协作",
    "任务委派和 Handoff",
    "安全护栏和 Guardrails",
    "复杂的 Agent 工作流",
    "生产级应用",
]

# 混合使用示例
def hybrid_approach():
    """混合使用 Assistants API 和 Agents SDK"""
    # 使用 Assistants API 处理文件和代码执行
    # 使用 Agents SDK 编排多 Agent 协作
    pass
```

### 19.6.3 性能对比

| 指标 | Assistants API | Agents SDK |
|------|----------------|------------|
| **延迟** | 中等 | 低 |
| **吞吐量** | 中等 | 高 |
| **成本** | 较高 | 中等 |
| **扩展性** | 优秀 | 优秀 |
| **企业支持** | 官方支持 | 社区支持 |

---

## 19.7 综合实战案例

### 19.7.1 智能客服系统

```python
from agents import Agent, Runner, InputGuardrail, GuardrailFunctionOutput
from pydantic import BaseModel

# 定义 Guardrail
class CustomerServiceGuardrail(BaseModel):
    is_appropriate: bool
    category: str

@InputGuardrail
async def customer_service_guardrail(context, agent):
    """客服 Guardrail"""
    user_message = context.messages[-1]["content"]
    
    # 检查是否是有效的客服问题
    valid_categories = ["咨询", "投诉", "建议", "技术支持"]
    
    has_valid_category = any(cat in user_message for cat in valid_categories)
    
    if not has_valid_category:
        return GuardrailFunctionOutput(
            output_info=CustomerServiceGuardrail(
                is_appropriate=False,
                category="invalid"
            ),
            tripwire_triggered=True,
        )
    
    return GuardrailFunctionOutput(
        output_info=CustomerServiceGuardrail(
            is_appropriate=True,
            category="valid"
        ),
        tripwire_triggered=False,
    )

# 创建 Agent
triage_agent = Agent(
    name="客服调度",
    instructions="""你是客服调度员，负责将客户请求分配给合适的部门。
    
    分配规则：
    - 技术问题 -> 技术支持
    - 投诉问题 -> 客户关怀
    - 一般咨询 -> 通用客服""",
    model="gpt-4",
    handoffs=["tech_support", "customer_care", "general_support"],
    input_guardrails=[customer_service_guardrail],
)

tech_support = Agent(
    name="技术支持",
    instructions="你是技术支持专家，负责解决技术问题。",
    model="gpt-4",
)

customer_care = Agent(
    name="客户关怀",
    instructions="你是客户关怀专家，负责处理投诉和关怀客户。",
    model="gpt-4",
)

general_support = Agent(
    name="通用客服",
    instructions="你是通用客服，负责处理一般咨询。",
    model="gpt-4",
)

# 使用
async def main():
    result = await Runner.run(
        triage_agent,
        messages=[{"role": "user", "content": "我的产品有技术问题，需要技术支持"}],
    )
    print(f"处理部门: {result.last_agent}")
    print(f"回复: {result.final_output}")

import asyncio
asyncio.run(main())
```

### 19.7.2 研究助手系统

```python
from agents import Agent, Runner, function_tool

# 定义工具
@function_tool
def search_academic(query: str) -> str:
    """搜索学术论文"""
    return f"学术搜索结果: 关于 '{query}' 的论文..."

@function_tool
def analyze_paper(paper_content: str) -> str:
    """分析论文内容"""
    return f"论文分析: {paper_content[:200]}..."

@function_tool
def generate_citation(paper_info: dict) -> str:
    """生成引用格式"""
    return f"引用格式: {paper_info.get('author', 'Unknown')} ({paper_info.get('year', '2024')})"

# 创建 Agent
research_agent = Agent(
    name="研究助手",
    instructions="""你是一个研究助手，负责：
    1. 搜索相关学术论文
    2. 分析论文内容
    3. 生成引用格式""",
    model="gpt-4",
    tools=[search_academic, analyze_paper, generate_citation],
    handoffs=["summary_agent"],
)

summary_agent = Agent(
    name="总结助手",
    instructions="你负责总结研究结果并生成报告。",
    model="gpt-4",
)

# 使用
async def main():
    result = await Runner.run(
        research_agent,
        messages=[{"role": "user", "content": "搜索关于 Transformer 架构的最新论文并分析"}],
    )
    print(f"研究结果: {result.final_output}")

import asyncio
asyncio.run(main())
```

---

## 19.8 高级特性与最佳实践

### 19.8.1 流式输出

```python
from agents import Agent, Runner

# 创建 Agent
agent = Agent(
    name="流式助手",
    instructions="你是一个提供流式输出的助手。",
    model="gpt-4",
)

# 流式执行
async def main():
    # 使用 Runner 的流式接口
    result = await Runner.run(
        agent,
        messages=[{"role": "user", "content": "写一首关于 AI 的诗"}],
        stream=True,  # 启用流式输出
    )
    
    # 逐步获取输出
    async for chunk in result.stream():
        print(chunk, end="", flush=True)
    
    print()  # 换行

import asyncio
asyncio.run(main())
```

### 19.8.2 多模态支持

```python
from agents import Agent, Runner

# 创建支持多模态的 Agent
multimodal_agent = Agent(
    name="多模态助手",
    instructions="""你是一个多模态助手，可以：
    1. 分析图片内容
    2. 理解图表数据
    3. 生成图像描述""",
    model="gpt-4-vision-preview",
)

# 使用多模态输入
async def main():
    result = await Runner.run(
        multimodal_agent,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请分析这张图片"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/image.jpg"
                        }
                    }
                ]
            }
        ],
    )
    print(result.final_output)

import asyncio
asyncio.run(main())
```

### 19.8.3 Agent 记忆管理

```python
from agents import Agent, Runner
from typing import List, Dict

class AgentMemory:
    """Agent 记忆管理"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.short_term: List[Dict] = []
        self.long_term: Dict = {}
    
    def add_message(self, role: str, content: str):
        """添加消息到短期记忆"""
        self.short_term.append({
            "role": role,
            "content": content,
        })
        
        # 限制历史长度
        if len(self.short_term) > self.max_history:
            self.short_term = self.short_term[-self.max_history:]
    
    def add_to_long_term(self, key: str, value: str):
        """添加到长期记忆"""
        self.long_term[key] = value
    
    def get_context(self) -> str:
        """获取记忆上下文"""
        context_parts = []
        
        # 短期记忆
        if self.short_term:
            context_parts.append("最近对话:")
            for msg in self.short_term[-5:]:
                context_parts.append(f"  {msg['role']}: {msg['content'][:50]}...")
        
        # 长期记忆
        if self.long_term:
            context_parts.append("长期记忆:")
            for key, value in self.long_term.items():
                context_parts.append(f"  {key}: {value[:50]}...")
        
        return "\n".join(context_parts)

# 使用带记忆的 Agent
class MemoryAgent:
    """带记忆的 Agent"""
    
    def __init__(self, name: str, instructions: str, model: str = "gpt-4"):
        self.agent = Agent(
            name=name,
            instructions=instructions,
            model=model,
        )
        self.memory = AgentMemory()
    
    async def chat(self, user_message: str) -> str:
        """带记忆的对话"""
        # 添加用户消息到记忆
        self.memory.add_message("user", user_message)
        
        # 获取记忆上下文
        memory_context = self.memory.get_context()
        
        # 构建带记忆的指令
        enhanced_instructions = f"{self.agent.instructions}\n\n记忆上下文:\n{memory_context}"
        
        # 创建带增强指令的 Agent
        enhanced_agent = Agent(
            name=self.agent.name,
            instructions=enhanced_instructions,
            model=self.agent.model,
        )
        
        # 执行
        result = await Runner.run(
            enhanced_agent,
            messages=[{"role": "user", "content": user_message}],
        )
        
        # 添加助手回复到记忆
        self.memory.add_message("assistant", result.final_output)
        
        return result.final_output

# 使用示例
async def main():
    agent = MemoryAgent(
        name="记忆助手",
        instructions="你是一个有记忆的助手，可以记住之前的对话。",
    )
    
    # 多轮对话
    response1 = await agent.chat("我叫张三")
    print(f"回复1: {response1}")
    
    response2 = await agent.chat("我叫什么名字？")
    print(f"回复2: {response2}")

import asyncio
asyncio.run(main())
```

### 19.8.4 错误处理与重试

```python
from agents import Agent, Runner
import asyncio
from typing import Optional

class ResilientRunner:
    """弹性 Runner"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    async def run_with_retry(
        self,
        agent: Agent,
        messages: list,
        **kwargs
    ):
        """带重试的执行"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                result = await Runner.run(
                    agent,
                    messages=messages,
                    **kwargs
                )
                return result
                
            except Exception as e:
                last_error = e
                print(f"执行失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        raise Exception(f"执行失败，已重试 {self.max_retries} 次: {last_error}")
    
    async def run_with_timeout(
        self,
        agent: Agent,
        messages: list,
        timeout: float = 30.0,
        **kwargs
    ):
        """带超时的执行"""
        try:
            result = await asyncio.wait_for(
                Runner.run(agent, messages=messages, **kwargs),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            raise Exception(f"执行超时: {timeout} 秒")

# 使用
async def main():
    runner = ResilientRunner(max_retries=3)
    
    agent = Agent(
        name="弹性助手",
        instructions="你是一个有弹性的助手。",
        model="gpt-4",
    )
    
    try:
        result = await runner.run_with_retry(
            agent,
            messages=[{"role": "user", "content": "测试重试机制"}],
        )
        print(f"成功: {result.final_output}")
    except Exception as e:
        print(f"失败: {e}")

import asyncio
asyncio.run(main())
```

### 19.8.5 成本优化

```python
from agents import Agent, Runner
from datetime import datetime

class CostOptimizer:
    """成本优化器"""
    
    def __init__(self, budget: float = 100.0):
        self.budget = budget
        self.total_cost = 0.0
        self.call_history = []
    
    def estimate_cost(self, model: str, tokens: int) -> float:
        """估算成本"""
        cost_per_1k = {
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01,
            "gpt-3.5-turbo": 0.002,
            "gpt-4-vision-preview": 0.01,
        }
        
        return (tokens / 1000) * cost_per_1k.get(model, 0.01)
    
    def track_call(self, model: str, tokens_used: int):
        """追踪调用成本"""
        cost = self.estimate_cost(model, tokens_used)
        self.total_cost += cost
        
        self.call_history.append({
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "tokens": tokens_used,
            "cost": cost,
        })
        
        # 检查预算
        if self.total_cost > self.budget:
            raise Exception(f"超出预算: ${self.total_cost:.2f} / ${self.budget:.2f}")
    
    def get_optimization_suggestions(self) -> list:
        """获取优化建议"""
        suggestions = []
        
        # 检查是否频繁使用 GPT-4
        gpt4_calls = sum(1 for h in self.call_history if h["model"] == "gpt-4")
        if gpt4_calls > 10:
            suggestions.append("考虑使用 GPT-3.5-turbo 处理简单任务")
        
        # 检查平均 token 使用
        avg_tokens = sum(h["tokens"] for h in self.call_history) / len(self.call_history) if self.call_history else 0
        if avg_tokens > 2000:
            suggestions.append("考虑优化提示词以减少 token 使用")
        
        return suggestions

# 使用
async def main():
    optimizer = CostOptimizer(budget=50.0)
    
    agent = Agent(
        name="经济助手",
        instructions="你是一个经济高效的助手。",
        model="gpt-3.5-turbo",  # 使用更便宜的模型
    )
    
    # 追踪成本
    result = await Runner.run(
        agent,
        messages=[{"role": "user", "content": "测试成本优化"}],
    )
    
    # 模拟成本追踪
    optimizer.track_call("gpt-3.5-turbo", 500)
    
    print(f"总成本: ${optimizer.total_cost:.4f}")
    print(f"优化建议: {optimizer.get_optimization_suggestions()}")

import asyncio
asyncio.run(main())
```

### 19.8.6 监控与日志

```python
from agents import Agent, Runner
from datetime import datetime
import logging

class AgentMonitor:
    """Agent 监控"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_duration": 0,
            "errors": 0,
            "handoffs": 0,
        }
        self.call_log = []
    
    def record_call(
        self,
        agent_name: str,
        duration: float,
        tokens: int,
        success: bool,
        handoff: bool = False
    ):
        """记录调用"""
        self.metrics["total_calls"] += 1
        self.metrics["total_tokens"] += tokens
        self.metrics["total_duration"] += duration
        
        if not success:
            self.metrics["errors"] += 1
        
        if handoff:
            self.metrics["handoffs"] += 1
        
        # 记录日志
        self.call_log.append({
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "duration": duration,
            "tokens": tokens,
            "success": success,
            "handoff": handoff,
        })
        
        # 输出日志
        self.logger.info(
            f"Agent Call - {agent_name}: {duration:.2f}s, {tokens} tokens, "
            f"Success: {success}, Handoff: {handoff}"
        )
    
    def get_metrics(self) -> dict:
        """获取指标"""
        avg_duration = (
            self.metrics["total_duration"] / self.metrics["total_calls"]
            if self.metrics["total_calls"] > 0
            else 0
        )
        
        error_rate = (
            self.metrics["errors"] / self.metrics["total_calls"]
            if self.metrics["total_calls"] > 0
            else 0
        )
        
        return {
            **self.metrics,
            "average_duration": avg_duration,
            "error_rate": error_rate,
        }
    
    def get_agent_stats(self) -> dict:
        """获取 Agent 统计"""
        agent_stats = {}
        
        for log in self.call_log:
            agent = log["agent"]
            if agent not in agent_stats:
                agent_stats[agent] = {
                    "calls": 0,
                    "total_duration": 0,
                    "total_tokens": 0,
                    "errors": 0,
                }
            
            agent_stats[agent]["calls"] += 1
            agent_stats[agent]["total_duration"] += log["duration"]
            agent_stats[agent]["total_tokens"] += log["tokens"]
            
            if not log["success"]:
                agent_stats[agent]["errors"] += 1
        
        return agent_stats

# 使用
async def main():
    monitor = AgentMonitor()
    
    agent = Agent(
        name="监控助手",
        instructions="你是一个被监控的助手。",
        model="gpt-4",
    )
    
    import time
    start_time = time.time()
    
    result = await Runner.run(
        agent,
        messages=[{"role": "user", "content": "测试监控"}],
    )
    
    duration = time.time() - start_time
    
    # 记录调用
    monitor.record_call(
        agent_name="监控助手",
        duration=duration,
        tokens=100,
        success=True,
    )
    
    print(f"指标: {monitor.get_metrics()}")
    print(f"Agent 统计: {monitor.get_agent_stats()}")

import asyncio
asyncio.run(main())
```

### 19.8.7 配置管理

```python
import yaml
from agents import Agent, Runner

# 配置文件结构
config_yaml = """
agents:
  assistant:
    name: "AI 助手"
    instructions: "你是一个有帮助的 AI 助手。"
    model: "gpt-4"
    tools: []
    handoffs: []
  
  researcher:
    name: "研究员"
    instructions: "你是一个研究员。"
    model: "gpt-4"
    tools:
      - "search_web"
      - "analyze_data"
    handoffs:
      - "writer"

  writer:
    name: "写手"
    instructions: "你是一个写手。"
    model: "gpt-4"
    tools: []
    handoffs: []

guardrails:
  safety:
    type: "input"
    rules:
      - "no_profanity"
      - "no_sensitive_info"
  
  quality:
    type: "output"
    rules:
      - "min_length"
      - "max_length"
"""

# 加载配置
def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# 根据配置创建 Agent
def create_agents_from_config(config: dict) -> dict:
    """根据配置创建 Agent"""
    agents = {}
    
    for agent_name, agent_config in config.get("agents", {}).items():
        agent = Agent(
            name=agent_config["name"],
            instructions=agent_config["instructions"],
            model=agent_config["model"],
            handoffs=agent_config.get("handoffs", []),
        )
        agents[agent_name] = agent
    
    return agents

# 使用
# config = load_config("agents_config.yaml")
# agents = create_agents_from_config(config)
# result = await Runner.run(agents["assistant"], messages=[...])
```

### 19.8.8 测试策略

```python
from agents import Agent, Runner
from unittest.mock import AsyncMock, patch
import pytest

# 测试 Agent
class TestAgent:
    """Agent 测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.agent = Agent(
            name="测试助手",
            instructions="你是一个测试助手。",
            model="gpt-4",
        )
    
    @pytest.mark.asyncio
    async def test_basic_chat(self):
        """测试基础对话"""
        # Mock Runner.run
        with patch('agents.Runner.run') as mock_run:
            mock_run.return_value = AsyncMock(
                final_output="测试回复"
            )
            
            result = await Runner.run(
                self.agent,
                messages=[{"role": "user", "content": "测试"}],
            )
            
            assert result.final_output == "测试回复"
            mock_run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_with_tools(self):
        """测试带工具的 Agent"""
        from agents import function_tool
        
        @function_tool
        def mock_tool(query: str) -> str:
            return f"工具结果: {query}"
        
        agent = Agent(
            name="工具助手",
            instructions="你是一个使用工具的助手。",
            model="gpt-4",
            tools=[mock_tool],
        )
        
        # 测试
        assert len(agent.tools) == 1

# 测试 Handoff
class TestHandoff:
    """Handoff 测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.agent_a = Agent(
            name="Agent A",
            instructions="你是 Agent A。",
            model="gpt-4",
            handoffs=["agent_b"],
        )
        
        self.agent_b = Agent(
            name="Agent B",
            instructions="你是 Agent B。",
            model="gpt-4",
        )
    
    @pytest.mark.asyncio
    async def test_handoff(self):
        """测试 Handoff"""
        assert "agent_b" in self.agent_a.handoffs

# 测试 Guardrail
class TestGuardrail:
    """Guardrail 测试"""
    
    @pytest.mark.asyncio
    async def test_input_guardrail(self):
        """测试输入 Guardrail"""
        from agents import InputGuardrail, GuardrailFunctionOutput
        
        @InputGuardrail
        async def mock_guardrail(context, agent):
            return GuardrailFunctionOutput(
                output_info={"status": "passed"},
                tripwire_triggered=False,
            )
        
        agent = Agent(
            name="受保护的助手",
            instructions="你是一个受保护的助手。",
            model="gpt-4",
            input_guardrails=[mock_guardrail],
        )
        
        assert len(agent.input_guardrails) == 1

# 使用 pytest 运行测试
# pytest test_agents_sdk.py -v
```

### 19.8.9 Agents SDK API 速查表

```python
# 导入核心组件
from agents import (
    Agent,
    Runner,
    function_tool,
    Handoff,
    InputGuardrail,
    OutputGuardrail,
    GuardrailFunctionOutput,
)

# Agent 创建
agent = Agent(
    name="助手",
    instructions="指令",
    model="gpt-4",
    tools=[tool1, tool2],
    handoffs=["other_agent"],
    input_guardrails=[guardrail1],
    output_guardrails=[guardrail2],
)

# Runner 执行
result = await Runner.run(
    agent,
    messages=[{"role": "user", "content": "..."}],
)

# 流式执行
async for chunk in Runner.run_streamed(agent, messages=[...]):
    print(chunk)

# 工具定义
@function_tool
def my_tool(arg: str) -> str:
    return "result"

# Handoff 定义
handoff = Handoff(
    target_agent=other_agent,
    condition=lambda ctx: True,
    description="委派描述",
)

# Guardrail 定义
@InputGuardrail
async def my_guardrail(context, agent):
    return GuardrailFunctionOutput(
        output_info={},
        tripwire_triggered=False,
    )

# 获取结果
print(result.final_output)  # 最终输出
print(result.last_agent)    # 最后处理的 Agent
print(result.steps)         # 执行步骤
```

### 19.8.10 常见问题排查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| **Agent 未响应** | API 错误 | 检查 API Key 和网络 |
| **Handoff 失败** | Agent 名称错误 | 检查 handoffs 配置 |
| **Guardrail 误触发** | 规则过于严格 | 调整规则阈值 |
| **成本过高** | 模型选择不当 | 使用更便宜的模型 |
| **响应慢** | 模型选择 | 使用更快的模型 |
| **记忆丢失** | 未持久化 | 实现记忆存储 |
| **工具调用失败** | 函数定义错误 | 检查工具定义 |

### 最佳实践总结

> **最佳实践清单**：
> 
> 1. **Agent 设计**
>    - 为每个 Agent 定义清晰的职责
>    - 使用简洁明确的指令
>    - 合理配置工具和 Handoff
> 
> 2. **Handoff 管理**
>    - 使用条件路由实现智能分配
>    - 避免循环 Handoff
>    - 记录 Handoff 历史
> 
> 3. **Guardrails**
>    - 平衡安全性和用户体验
>    - 定期审查和更新规则
>    - 监控 Guardrail 触发情况
> 
> 4. **性能优化**
>    - 选择合适的模型
>    - 启用缓存减少重复调用
>    - 使用异步执行提升吞吐量
> 
> 5. **成本控制**
>    - 监控 API 调用成本
>    - 使用更便宜的模型处理简单任务
>    - 实现预算告警
> 
> 6. **监控与日志**
>    - 记录详细的调用日志
>    - 监控关键指标
>    - 实现告警机制

### 19.8.11 最佳实践总结

> **最佳实践清单**：
> 
> 1. **Agent 设计**
>    - 为每个 Agent 定义清晰的职责
>    - 使用简洁明确的指令
>    - 合理配置工具和 Handoff
> 
> 2. **Handoff 管理**
>    - 使用条件路由实现智能分配
>    - 避免循环 Handoff
>    - 记录 Handoff 历史
> 
> 3. **Guardrails**
>    - 平衡安全性和用户体验
>    - 定期审查和更新规则
>    - 监控 Guardrail 触发情况
> 
> 4. **性能优化**
>    - 选择合适的模型
>    - 启用缓存减少重复调用
>    - 使用异步执行提升吞吐量
> 
> 5. **成本控制**
>    - 监控 API 调用成本
>    - 使用更便宜的模型处理简单任务
>    - 实现预算告警
> 
> 6. **监控与日志**
>    - 记录详细的调用日志
>    - 监控关键指标
>    - 实现告警机制

---

## 本章小结

本章深入探讨了 OpenAI Agents SDK 的核心特性：

1. **Assistants API**：持久化助手、文件检索、代码执行
2. **Agents SDK**：多 Agent 协作、Handoff 机制、Guardrails
3. **Handoff 机制**：任务委派、条件路由、链式调用
4. **Guardrails**：输入检查、输出质量、自定义规则
5. **框架对比**：Assistants API 适合单 Agent，Agents SDK 适合多 Agent

> **核心要点**：Agents SDK 的核心优势在于其多 Agent 协作能力和内置的安全护栏。通过 Handoff 机制可以实现复杂的任务委派，Guardrails 确保系统的安全性。

---

## 思考题

1. Assistants API 和 Agents SDK 各适用于什么场景？
2. 如何设计一个支持复杂 Handoff 链的系统？
3. Guardrails 如何平衡安全性和用户体验？
4. 如何优化 Agents SDK 的性能和成本？
5. 如何将 Agents SDK 与现有的企业系统集成？

---

## 附录：OpenAI Agents SDK 生态系统

### 核心组件

```
OpenAI Agents SDK 生态系统：

┌─────────────────────────────────────────────────────────┐
│                  Agents SDK Core                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Agent     │  │   Runner    │  │  Handoff    │     │
│  │   (助手)    │  │  (执行器)   │  │  (委派)     │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
├─────────────────────────────────────────────────────────┤
│                Tools & Guardrails                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Function    │  │  Input      │  │  Output     │     │
│  │ Tools       │  │  Guardrails │  │  Guardrails │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
├─────────────────────────────────────────────────────────┤
│                Integration                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ OpenAI API  │  │  Streaming  │  │  Memory     │     │
│  │             │  │             │  │  (自定义)   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### 安装与配置

```bash
# 安装 Agents SDK
pip install openai-agents

# 安装开发依赖
pip install openai-agents[dev]

# 验证安装
python -c "import agents; print(agents.__version__)"
```

### 环境变量

```bash
# OpenAI API Key
export OPENAI_API_KEY="your-api-key"

# 可选：自定义 API 端点
export OPENAI_API_BASE="https://api.openai.com/v1"

# 可选：调试模式
export AGENTS_DEBUG=true
```

### 版本兼容性

| Agents SDK 版本 | Python 版本 | 主要特性 |
|----------------|-------------|----------|
| 0.x | 3.10+ | 基础功能 |
| 1.x | 3.10+ | 稳定版，生产就绪 |

### Assistants API vs Agents SDK 详细对比

| 特性维度 | Assistants API | Agents SDK |
|---------|----------------|------------|
| **状态管理** | 服务端持久化 | 客户端管理 |
| **对话模型** | Thread + Run | 直接消息 |
| **文件处理** | 内置 Vector Store | 通过工具 |
| **代码执行** | 内置沙箱 | 通过工具 |
| **多 Agent** | 不支持 | 内置 Handoff |
| **安全护栏** | 不支持 | 内置 Guardrails |
| **流式输出** | 支持 | 支持 |
| **异步支持** | 有限 | 完整支持 |
| **适用场景** | 单 Agent 应用 | 多 Agent 协作 |
| **学习曲线** | 中等 | 较低 |
| **企业支持** | 官方支持 | 社区支持 |

### 快速开始示例

```python
from agents import Agent, Runner

# 1. 创建 Agent
agent = Agent(
    name="我的助手",
    instructions="你是一个有帮助的助手。",
    model="gpt-4",
)

# 2. 运行
import asyncio

async def main():
    result = await Runner.run(
        agent,
        messages=[{"role": "user", "content": "你好！"}],
    )
    print(result.final_output)

asyncio.run(main())
```

### 常用命令

```bash
# 运行 Agent
python my_agent.py

# 运行测试
pytest tests/ -v

# 代码检查
ruff check .

# 类型检查
mypy agents/

# 构建文档
mkdocs build

# 清理缓存
rm -rf __pycache__ .pytest_cache
```

---

## 参考资源

- [OpenAI Assistants API 文档](https://platform.openai.com/docs/assistants/overview)
- [OpenAI Agents SDK GitHub](https://github.com/openai/openai-agents-python)
- [Agents SDK 示例](https://github.com/openai/openai-agents-python/tree/main/examples)
- [OpenAI API 参考](https://platform.openai.com/docs/api-reference)
- [Agents SDK 最佳实践](https://github.com/openai/openai-agents-python/blob/main/docs/)
- [OpenAI 社区论坛](https://community.openai.com/)
- [Agents SDK 贡献指南](https://github.com/openai/openai-agents-python/blob/main/CONTRIBUTING.md)

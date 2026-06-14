---
title: "第19章：OpenAI Agents SDK 与原生 Agent"
description: "掌握 OpenAI Assistants API、Agents SDK 的架构设计、工具集成、Handoff 机制、Guardrails 安全护栏与代码解释器。"
date: "2026-06-11"
---

# 第19章：OpenAI Agents SDK 与原生 Agent

---

## 19.1 Assistants API

```python
from openai import OpenAI
client = OpenAI()

assistant = client.beta.assistants.create(
    name="Data Analyst",
    instructions="你是一个数据分析助手。",
    model="gpt-4o",
    tools=[{"type": "code_interpreter"}, {"type": "file_search"}]
)

thread = client.beta.threads.create()
client.beta.threads.messages.create(thread_id=thread.id, role="user", content="分析销售数据")
run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant.id)
```

---

## 19.2 Agents SDK

```python
from agents import Agent, Runner, function_tool

@function_tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city}：晴，25°C"

agent = Agent(
    name="Assistant",
    instructions="你是一个有帮助的助手。",
    tools=[get_weather],
    model="gpt-4o",
)
result = Runner.run_sync(agent, "北京天气怎么样？")
print(result.final_output)
```

---

## 19.3 Handoff

```python
from agents import Agent, handoff

weather_agent = Agent(name="Weather Agent", instructions="你专门处理天气查询。", tools=[get_weather])
math_agent = Agent(name="Math Agent", instructions="你专门处理数学计算。", tools=[calculator])

triage_agent = Agent(
    name="Triage Agent",
    instructions="根据用户问题类型，交接给合适的专业 Agent。",
    handoffs=[
        handoff(weather_agent, tool_name_override="transfer_to_weather"),
        handoff(math_agent, tool_name_override="transfer_to_math"),
    ],
)
```

---

## 19.4 Guardrails

```python
from agents import Agent, InputGuardrail, GuardrailFunctionOutput
from pydantic import BaseModel

class SafetyCheck(BaseModel):
    is_safe: bool
    reason: str

guardrail_agent = Agent(name="Safety Checker", instructions="检查输入是否安全。", output_type=SafetyCheck)

async def safety_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data)
    output = result.final_output_as(SafetyCheck)
    return GuardrailFunctionOutput(output_info=output, tripwire_triggered=not output.is_safe)
```

---

## 19.5 Assistants API vs Agents SDK

| 特性 | Assistants API | Agents SDK |
|:---|:---|:---|
| 定位 | 平台级服务 | 轻量级框架 |
| 状态管理 | 服务端 Thread | 本地状态 |
| Handoff | 不支持 | 原生支持 |
| Guardrails | 不支持 | 原生支持 |

---

## 19.6 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| Assistants API | 原生 Agent 平台 |
| Agents SDK | 轻量级框架 |
| Handoff | Agent 间任务交接 |
| Guardrails | 输入/输出安全护栏 |

> **下一章预告**
>
> 在第 20 章中，我们将系统梳理多智能体协作模式。

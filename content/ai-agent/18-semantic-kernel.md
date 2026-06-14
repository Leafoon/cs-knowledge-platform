---
title: "第18章：Semantic Kernel 企业级 Agent"
description: "深入 Microsoft Semantic Kernel：Kernel 架构、Plugin 系统、Planner 自动规划、Memory Store 与企业级集成模式。"
date: "2026-06-11"
---

# 第18章：Semantic Kernel 企业级 Agent

---

## 18.1 Kernel 概念

```python
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

kernel = sk.Kernel()
kernel.add_service(OpenAIChatCompletion(service_id="chat", ai_model_id="gpt-4o", api_key="sk-..."))
```

---

## 18.2 Plugin 系统

```python
from semantic_kernel.functions import kernel_function

class MathPlugin:
    @kernel_function(description="计算数学表达式")
    def calculate(self, expression: str) -> str:
        return str(eval(expression))

class WeatherPlugin:
    @kernel_function(description="获取城市天气")
    def get_weather(self, city: str) -> str:
        return f"{city}：晴，25°C"

kernel.add_plugin(MathPlugin(), "math")
kernel.add_plugin(WeatherPlugin(), "weather")
```

---

## 18.3 Planner

```python
from semantic_kernel.planners.function_calling_stepwise_planner import FunctionCallingStepwisePlanner

planner = FunctionCallingStepwisePlanner(service_id="chat")
result = await planner.invoke(kernel=kernel, question="计算 (2 + 3) * 4")
```

---

## 18.4 Memory Store

```python
from semantic_kernel.memory import SemanticTextMemory

memory = SemanticTextMemory(storage=ChromaMemoryStore(persist_directory="./chroma_sk"))
await memory.save_information(collection="prefs", text="用户喜欢简洁回答", id="pref_1")
results = await memory.search(collection="prefs", query="用户偏好？", limit=3)
```

---

## 18.5 Semantic Kernel vs LangChain

| 特性 | Semantic Kernel | LangChain |
|:---|:---|:---|
| 语言支持 | Python, C#, Java | Python, JS/TS |
| Azure 集成 | 原生深度集成 | 通过社区包 |
| 企业特性 | RBAC, 合规 | 通过 LangSmith |

---

## 18.6 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| Kernel | 核心容器 |
| Plugin | 功能封装单元 |
| Planner | 自动规划 |
| Memory | 语义记忆 |

> **下一章预告**
>
> 在第 19 章中，我们将学习 OpenAI Agents SDK。

---
title: "第13章：Agent 架构设计 — 单体到分布式"
description: "系统梳理 Agent 架构设计模式：单 Agent 循环、状态机 Agent、事件驱动 Agent、分层 Agent、多 Agent 主从/对等架构与微服务架构。"
date: "2026-06-11"
---

# 第13章：Agent 架构设计 — 单体到分布式

---

## 13.1 架构设计原则

| 原则 | 在 Agent 中的体现 |
|:---|:---|
| 单一职责 | 每个 Agent 只负责一个领域 |
| 开闭原则 | 通过工具扩展能力 |
| 接口隔离 | 工具接口简洁 |

---

## 13.2 单 Agent 架构

### 循环式

```python
class LoopAgent:
    def __init__(self, llm, tools, max_iterations=10):
        self.llm = llm.bind_tools(tools)
        self.max_iterations = max_iterations

    def run(self, task):
        messages = [SystemMessage(content="你是一个 AI Agent。"), HumanMessage(content=task)]
        for _ in range(self.max_iterations):
            response = self.llm.invoke(messages)
            messages.append(response)
            if not response.tool_calls: return response.content
            for tc in response.tool_calls:
                result = execute_tool(tc)
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        return "达到最大迭代次数"
```

### 状态机

```python
from enum import Enum

class AgentState(Enum):
    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    DONE = "done"
    ERROR = "error"
```

---

## 13.3 多 Agent 架构

### 主从式

```python
from langgraph.graph import StateGraph, MessagesState, START, END

def supervisor(state):
    llm = ChatOpenAI(model="gpt-4o")
    response = llm.invoke([SystemMessage(content="选择：research / code / finish"), *state["messages"]])
    return {"next": response.content.strip().lower()}

graph = StateGraph(MessagesState)
graph.add_node("supervisor", supervisor)
graph.add_node("research", research_node)
graph.add_node("code", code_node)
graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", lambda s: s["next"], {"research": "research", "code": "code", "finish": END})
```

### 对等协作

```python
def peer_collaboration(agents, task):
    messages = [HumanMessage(content=task)]
    for agent in agents:
        response = agent.invoke({"messages": messages})
        messages.append(AIMessage(content=f"[{agent.name}]: {response['messages'][-1].content}"))
    return messages[-1].content
```

---

## 13.4 微服务架构

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class AgentRequest(BaseModel):
    message: str
    session_id: str

@app.post("/agent/chat")
async def chat(request: AgentRequest):
    result = await agent.run(request.message, request.session_id)
    return {"response": result["answer"], "tools_used": result["tools"]}
```

---

## 13.5 架构选型

| 架构 | 复杂度 | 适用场景 |
|:---|:---|:---|
| 循环式 | 低 | 简单工具调用 |
| 状态机 | 中 | 流程明确 |
| 主从式 | 中高 | 多领域协作 |
| 对等式 | 高 | 创造性协作 |
| 微服务 | 很高 | 企业级系统 |

---

## 13.6 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| 循环式 | 最简单架构 |
| 状态机 | 有限状态机管理 |
| 主从式 | Supervisor 分配 |
| 对等式 | Agent 平等协作 |
| 微服务 | 企业级架构 |

> **下一章预告**
>
> 在第 14 章中，我们将深入 LangChain/LangGraph Agent 实战。

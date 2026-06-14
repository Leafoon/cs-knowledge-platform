---
title: "第14章：LangChain Agent 深度实战"
description: "深入 LangChain/LangGraph 的 Agent 实现：StateGraph、create_react_agent、自定义 Agent、工具集成、错误恢复与时间旅行调试。"
date: "2026-06-11"
---

# 第14章：LangChain Agent 深度实战

---

## 14.1 LangGraph 基础

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_openai import ChatOpenAI

def agent_node(state: MessagesState):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return {"messages": [llm.invoke(state["messages"])]}

def tool_node(state: MessagesState):
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        result = execute_tool(tc["name"], tc["args"])
        results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": results}

def should_continue(state):
    last = state["messages"][-1]
    return "tools" if hasattr(last, "tool_calls") and last.tool_calls else END

graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")
app = graph.compile()
```

---

## 14.2 create_react_agent

```python
from langgraph.prebuilt import create_react_agent

@tool
def search(query: str) -> str:
    """搜索互联网"""
    return f"搜索 '{query}' 的结果..."

@tool
def calculator(expression: str) -> float:
    """计算数学表达式"""
    return eval(expression)

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o", temperature=0),
    tools=[search, calculator],
)
result = agent.invoke({"messages": [("user", "2+3等于多少？")]})
```

---

## 14.3 流式执行

```python
for chunk in agent.stream({"messages": [("user", "搜索 AI Agent")]}, stream_mode="updates"):
    for node, update in chunk.items():
        print(f"\n[{node}]")
        if "messages" in update:
            for msg in update["messages"]: msg.pretty_print()
```

---

## 14.4 Human-in-the-Loop

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = graph.compile(checkpointer=memory, interrupt_before=["tools"])

config = {"configurable": {"thread_id": "user_1"}}
result = app.invoke({"messages": [("user", "删除日志文件")]}, config=config)
user_approved = True
if user_approved: result = app.invoke(None, config=config)
```

---

## 14.5 时间旅行调试

```python
history = list(app.get_state_history(config))
past_state = history[2]
app.update_state(config, past_state.values)
result = app.invoke(None, config=config)
```

---

## 14.6 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| StateGraph | 用图编排 Agent 行为 |
| create_react_agent | 最简单的 ReAct Agent |
| 流式输出 | 实时查看推理过程 |
| HITL | 人工审批机制 |
| 时间旅行 | 检查点回溯调试 |

> **下一章预告**
>
> 在第 15 章中，我们将深入 LangGraph 有状态 Agent。

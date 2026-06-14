---
title: "第15章：LangGraph 有状态 Agent — 超越线性链"
description: "掌握 LangGraph 的状态图 Agent 构建：StateGraph、条件边、循环、子图组合、检查点持久化、Human-in-the-Loop、流式执行与时间旅行调试。"
date: "2026-06-11"
---

# 第15章：LangGraph 有状态 Agent — 超越线性链

---

## 15.1 高级状态管理

```python
from typing import TypedDict, Annotated
import operator

class ComplexAgentState(TypedDict):
    messages: Annotated[list, operator.add]
    plan: list[str]
    current_step: int
    errors: list[str]
    final_answer: str

def smart_agent_node(state: ComplexAgentState):
    plan = state.get("plan", [])
    step = state.get("current_step", 0)
    errors = state.get("errors", [])
    if errors: prompt = f"之前的错误：{errors[-1]}，请调整策略。"
    elif plan and step < len(plan): prompt = f"执行步骤 {step + 1}：{plan[step]}"
    else: prompt = "请制定执行计划。"
    llm = ChatOpenAI(model="gpt-4o")
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response], "current_step": step + 1 if plan else 0}
```

---

## 15.2 子图组合

```python
def build_research_subgraph():
    graph = StateGraph(dict)
    graph.add_node("search", search_node)
    graph.add_node("analyze", analyze_node)
    graph.add_edge(START, "search")
    graph.add_edge("search", "analyze")
    graph.add_edge("analyze", END)
    return graph.compile()

main_graph = StateGraph(dict)
main_graph.add_node("researcher", build_research_subgraph())
main_graph.add_node("writer", build_writing_subgraph())
main_graph.add_edge(START, "researcher")
main_graph.add_edge("researcher", "writer")
main_graph.add_edge("writer", END)
```

---

## 15.3 流式执行

```python
# 模式 1：消息级
for chunk in app.stream(input_data, stream_mode="messages"):
    print(chunk.content, end="", flush=True)

# 模式 2：节点更新级
for chunk in app.stream(input_data, stream_mode="updates"):
    for node, update in chunk.items(): print(f"[{node}] {update}")
```

---

## 15.4 持久化

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    app = graph.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "session_1"}}
    result = app.invoke({"messages": [("user", "你好")]}, config=config)
```

---

## 15.5 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| 自定义状态 | TypedDict 定义复杂状态 |
| 子图组合 | 模块化构建 |
| 流式执行 | 三种模式 |
| 持久化 | SQLite/PostgreSQL |

> **下一章预告**
>
> 在第 16 章中，我们将深入 AutoGen 多智能体框架。

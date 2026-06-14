---
title: "第30章：工作流编排 — 复杂 Agent Pipeline"
description: "掌握复杂 Agent 工作流编排：DAG 执行、条件分支、并行处理、人工审批、错误补偿与 Saga 模式。"
date: "2026-06-11"
---

# 第30章：工作流编排 — 复杂 Agent Pipeline

---

## 30.1 DAG 编排

```python
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(AgentState)
workflow.add_node("fetch", fetch_data)
workflow.add_node("analyze", analyze)
workflow.add_node("summarize", summarize)
workflow.add_edge(START, "fetch")
workflow.add_edge("fetch", "analyze")
workflow.add_edge("analyze", "summarize")
workflow.add_edge("summarize", END)
```

---

## 30.2 条件分支

```python
def route_by_type(state):
    return "text_processor" if state["type"] == "text" else "code_processor"

workflow.add_conditional_edges("router", route_by_type, {...})
```

---

## 30.3 Human-in-the-Loop

```python
app = workflow.compile(interrupt_before=["approval"], checkpointer=MemorySaver())
result = app.invoke({"messages": [("user", "删除文件")]}, config)
# 暂停等待审批
```

---

## 30.4 Saga 模式

```python
class SagaOrchestrator:
    async def execute(self, context):
        completed = []
        for i, step in enumerate(self.steps):
            try:
                result = await step(context)
                completed.append(i)
            except Exception:
                for j in reversed(completed):
                    await self.compensations[j](context)
                raise
```

---

## 30.5 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| DAG | 有向无环图编排 |
| HITL | 人工审批 |
| Saga | 分布式事务补偿 |

> **下一章预告**
>
> 在第 31 章中，我们将学习 Agent 成本优化。

---
title: "第15章：LangGraph 有状态 Agent — 超越线性链"
description: "掌握 LangGraph 的状态图 Agent 构建：StateGraph、条件边、循环、子图组合、检查点持久化、Human-in-the-Loop、流式执行与时间旅行调试。"
date: "2026-06-11"
---

 # 第15章：LangGraph 有状态 Agent — 超越线性链

 LangGraph 是构建复杂、有状态 Agent 的标准工具。它基于图论的思想，将 Agent 的行为建模为一个有向图，其中节点代表处理步骤，边代表控制流。与传统的线性链不同，LangGraph 支持循环、条件分支、子图组合等高级特性，使得构建复杂的 Agent 工作流成为可能。本章将深入 LangGraph 的高级特性，帮助你构建生产级的有状态 Agent 系统。

LangGraph 的核心是状态图执行。下面的交互式演示展示了 LangGraph 的状态流转过程：

<div data-component="LangGraphStateFlow"></div>

ReAct 框架是 Agent 推理与行动交替的经典范式。下面的交互式演示展示了完整的 ReAct 工作流程：

<div data-component="ReActDemoV9"></div>

 ---

 ## 15.1 高级状态管理

### 15.1.1 自定义状态类型

LangGraph 使用 TypedDict 来定义 Agent 的状态结构。状态是 Agent 在整个执行过程中共享的数据容器，每个节点都可以读取和修改状态。

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph
import operator

class ComplexAgentState(TypedDict):
    """复杂 Agent 的状态定义"""
    messages: Annotated[list, operator.add]  # 消息历史（自动累加）
    plan: list[str]  # 执行计划
    current_step: int  # 当前步骤
    errors: list[str]  # 错误记录
    final_answer: str  # 最终输出
    context: dict  # 执行上下文
```

`Annotated[list, operator.add]` 告诉 LangGraph 如何合并来自不同节点的状态更新。当多个节点都向 `messages` 字段添加内容时，LangGraph 会使用 `operator.add`（即列表拼接）来合并它们。

### 15.1.2 状态更新策略

```python
def smart_agent_node(state: ComplexAgentState):
    """智能 Agent 节点：根据状态决定行为"""
    plan = state.get("plan", [])
    step = state.get("current_step", 0)
    errors = state.get("errors", [])

    if errors:
        prompt = f"之前的错误：{errors[-1]}，请调整策略。"
    elif plan and step < len(plan):
        prompt = f"执行计划步骤 {step + 1}：{plan[step]}"
    else:
        prompt = "请制定执行计划。"

    llm = ChatOpenAI(model="gpt-4o")
    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "messages": [response],
        "current_step": step + 1 if plan else 0,
        "tool_calls_count": state.get("tool_calls_count", 0) + 1,
    }
```

---

## 15.2 子图组合与模块化

### 15.2.1 子图的概念

子图是 LangGraph 中实现模块化的关键机制。通过将复杂的 Agent 系统拆分为多个子图，每个子图负责一个特定的功能，可以大幅提升代码的可维护性和可复用性。

子图的优势在于：
1. **独立测试**：每个子图可以独立测试和调试
2. **模块化**：修改一个子图不会影响其他子图
3. **复用性**：相同的子图可以在不同的主图中复用
4. **团队协作**：不同的开发者可以负责不同的子图

### 15.2.2 子图实现

```python
def build_research_subgraph():
    """研究子图：搜索 → 分析"""
    def search_node(state):
        return {"search_results": ["result1", "result2"]}
    def analyze_node(state):
        return {"analysis": "分析完成"}
    graph = StateGraph(dict)
    graph.add_node("search", search_node)
    graph.add_node("analyze", analyze_node)
    graph.add_edge(START, "search")
    graph.add_edge("search", "analyze")
    graph.add_edge("analyze", END)
    return graph.compile()

def build_writing_subgraph():
    """写作子图：草稿 → 审核"""
    def draft_node(state):
        return {"draft": "初稿完成"}
    def review_node(state):
        return {"review": "审核通过"}
    graph = StateGraph(dict)
    graph.add_node("draft", draft_node)
    graph.add_node("review", review_node)
    graph.add_edge(START, "draft")
    graph.add_edge("draft", "review")
    graph.add_edge("review", END)
    return graph.compile()

# 主图：组合子图
main_graph = StateGraph(dict)
main_graph.add_node("researcher", build_research_subgraph())
main_graph.add_node("writer", build_writing_subgraph())
main_graph.add_edge(START, "researcher")
main_graph.add_edge("researcher", "writer")
main_graph.add_edge("writer", END)
app = main_graph.compile()
```

---

## 15.3 流式执行

### 15.3.1 三种流式模式

```python
# 模式 1：消息级流式（Token 级输出）
for chunk in app.stream(input_data, stream_mode="messages"):
    print(chunk.content, end="", flush=True)

# 模式 2：节点更新级流式
for chunk in app.stream(input_data, stream_mode="updates"):
    for node, update in chunk.items():
        print(f"[{node}] {update}")

# 模式 3：值级流式
for chunk in app.stream(input_data, stream_mode="values"):
    print(chunk)
```

---

## 15.4 持久化与检查点

### 15.4.1 SQLite 检查点

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    app = graph.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "session_1"}}
    result = app.invoke({"messages": [("user", "你好")]}, config=config)
    result = app.invoke({"messages": [("user", "继续")]}, config=config)
```

### 15.4.2 PostgreSQL 持久化

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user:password@localhost:5432/agent_db"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

---

## 15.5 Human-in-the-Loop

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = graph.compile(checkpointer=memory, interrupt_before=["tools"])

config = {"configurable": {"thread_id": "user_1"}}
result = app.invoke({"messages": [("user", "删除日志文件")]}, config=config)

user_approved = True
if user_approved:
    result = app.invoke(None, config=config)
```

---

## 15.6 时间旅行调试

```python
history = list(app.get_state_history(config))
past_state = history[2]
app.update_state(config, past_state.values)
result = app.invoke(None, config=config)
```

---

## 15.7 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| 自定义状态 | TypedDict 定义复杂状态结构 |
| 子图组合 | 模块化构建，提升代码复用 |
| 流式执行 | messages/updates/values 三种模式 |
| 持久化 | SQLite/PostgreSQL 检查点 |
| HITL | interrupt_before 设置中断点 |
| 时间旅行 | 检查点历史回溯调试 |

> **下一章预告**
>
> 在第 16 章中，我们将深入 Microsoft AutoGen 多智能体框架。

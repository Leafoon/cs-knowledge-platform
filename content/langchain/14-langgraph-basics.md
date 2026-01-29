# Chapter 14: LangGraph 基础 - 状态图与控制流

> **本章目标**：深入掌握 LangGraph 的核心概念与实现原理，学习有状态、可循环的复杂 AI 应用构建方法，理解状态图、节点、边、检查点等核心机制，为构建生产级 Agent 系统打下基础。

## 14.1 为什么需要 LangGraph？

### 14.1.1 LCEL 的局限

虽然 LCEL 强大，但在构建复杂 AI 应用时存在局限：

**无法表达循环逻辑**：
```python
# LCEL 只能表达线性或并行流程
chain = prompt | llm | parser | next_step

# ❌ 无法表达：如果 X，则回到 Y 重试
# ❌ 无法表达：循环执行直到满足条件
```

**状态管理困难**：
- LCEL 链是无状态的，每次执行相互独立
- 难以维护跨步骤的共享状态
- 无法实现"记忆上次执行结果"的逻辑

**缺乏人工介入机制**：
- 无法在关键节点暂停等待人工审批
- 难以实现"生成 → 人工检查 → 修改 → 继续"的工作流

### 14.1.2 LangGraph 的核心优势

LangGraph 是**基于图的有状态编排框架**，专为复杂 AI 应用设计：

✅ **循环与条件控制**：支持循环、条件跳转、动态路由  
✅ **持久化状态**：内置 checkpoint 机制，支持状态保存与恢复  
✅ **人机协作**：支持中断、等待人工输入、时间旅行调试  
✅ **流式执行**：支持流式输出中间状态  
✅ **与 LCEL 兼容**：可以在节点中使用任何 LCEL 链

<div data-component="LangGraphArchitectureDiagram"></div>

---

## 14.2 核心概念

### 14.2.1 StateGraph - 状态图

StateGraph 是 LangGraph 的核心抽象，定义了应用的**状态结构**和**执行图**。

```python
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated
from operator import add

class AgentState(TypedDict):
    messages: Annotated[list, add]  # 消息列表（自动合并）
    user_input: str                  # 用户输入
    final_answer: str                # 最终答案
    iteration: int                   # 迭代计数

# 创建状态图
workflow = StateGraph(AgentState)
```

**状态更新策略**：
- **替换（默认）**：新值覆盖旧值
- **合并（Annotated[list, add]）**：自动追加到列表
- **自定义**：可以实现任意合并逻辑

### 14.2.2 节点（Node）

节点是状态图中的**处理单元**，接收当前状态，返回状态更新。

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

def call_llm(state: AgentState) -> AgentState:
    """调用 LLM 节点"""
    messages = state["messages"]
    response = llm.invoke(messages)
    
    # 返回状态更新
    return {
        "messages": [response],
        "iteration": state["iteration"] + 1
    }

# 添加节点
workflow.add_node("llm", call_llm)
```

**节点类型**：
- **普通节点**：执行计算、调用 LLM、处理数据
- **工具节点**：执行外部工具调用
- **条件节点**：根据状态返回不同路径

### 14.2.3 边（Edge）

边定义了节点之间的**转换关系**。

**无条件边**：
```python
# 从 A 到 B 的固定转换
workflow.add_edge("node_a", "node_b")
```

**条件边**：
```python
def should_continue(state: AgentState) -> str:
    """根据状态决定下一步"""
    if state["iteration"] > 5:
        return "end"
    if "FINAL ANSWER" in state["messages"][-1].content:
        return "end"
    return "continue"

# 条件路由
workflow.add_conditional_edges(
    "llm",
    should_continue,
    {
        "continue": "llm",     # 继续循环
        "end": END              # 结束
    }
)
```

### 14.2.4 入口点与编译

```python
from langgraph.graph import END

# 设置入口点
workflow.set_entry_point("llm")

# 编译为可执行的图
app = workflow.compile()

# 执行
result = app.invoke({
    "messages": [HumanMessage(content="What is 2+2?")],
    "iteration": 0
})
```

<div data-component="StateGraphExecution"></div>

---

## 14.3 构建第一个 LangGraph 应用

### 14.3.1 简单对话 Agent

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import TypedDict, Annotated
from operator import add

# 定义状态
class ChatState(TypedDict):
    messages: Annotated[list, add]

# 创建 LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 定义节点
def chat_node(state: ChatState):
    """聊天节点"""
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# 构建图
workflow = StateGraph(ChatState)
workflow.add_node("chat", chat_node)
workflow.set_entry_point("chat")
workflow.add_edge("chat", END)

# 编译
app = workflow.compile()

# 使用
initial_state = {
    "messages": [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is LangGraph?")
    ]
}

result = app.invoke(initial_state)
print(result["messages"][-1].content)
```

**执行流程**：
```
[入口] → chat_node → [结束]
```

### 14.3.2 带循环的 ReAct Agent

```python
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    # 模拟搜索
    return f"Search results for: {query}"

@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

tools = [search, calculator]
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list, add]
    iterations: int

def call_model(state: AgentState):
    """调用 LLM"""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "iterations": state["iterations"] + 1
    }

def execute_tools(state: AgentState):
    """执行工具"""
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    
    tool_messages = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_input = tool_call["args"]
        
        # 执行工具
        tool_output = None
        for t in tools:
            if t.name == tool_name:
                tool_output = t.invoke(tool_input)
                break
        
        # 创建工具消息
        tool_messages.append(
            ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_call["id"]
            )
        )
    
    return {"messages": tool_messages}

def should_continue(state: AgentState) -> str:
    """判断是否继续"""
    last_message = state["messages"][-1]
    
    # 超过最大迭代次数
    if state["iterations"] >= 10:
        return "end"
    
    # 没有工具调用，说明得到最终答案
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return "end"
    
    return "continue"

# 构建图
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tools)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

workflow.add_edge("tools", "agent")  # 工具执行后回到 agent

app = workflow.compile()

# 使用
result = app.invoke({
    "messages": [HumanMessage(content="What is 25 * 4? Then search for LangGraph.")],
    "iterations": 0
})

for msg in result["messages"]:
    print(f"{msg.__class__.__name__}: {msg.content}")
```

**执行流程**：
```
agent → 判断 → tools → agent → 判断 → tools → agent → 判断 → END
```

---

## 14.4 Checkpoint 与持久化

### 14.4.1 为什么需要 Checkpoint？

在长时间运行的 Agent 中，我们需要：
- **保存中间状态**：避免崩溃后从头开始
- **恢复执行**：从上次中断的地方继续
- **时间旅行调试**：回到任意历史状态检查问题
- **分支执行**：从某个历史点创建新的执行分支

### 14.4.2 内存 Checkpointer

```python
from langgraph.checkpoint.memory import MemorySaver

# 创建内存 checkpointer
memory = MemorySaver()

# 编译时传入
app = workflow.compile(checkpointer=memory)

# 执行时指定线程 ID
config = {"configurable": {"thread_id": "conversation-1"}}

# 第一次执行
result1 = app.invoke(
    {"messages": [HumanMessage(content="Hi, I'm Alice")]},
    config=config
)

# 第二次执行（会保留上次的状态）
result2 = app.invoke(
    {"messages": [HumanMessage(content="What's my name?")]},
    config=config
)

print(result2["messages"][-1].content)  # "Your name is Alice"
```

### 14.4.3 SQLite Checkpointer（持久化）

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# SQLite 持久化
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

app = workflow.compile(checkpointer=checkpointer)

# 使用方式相同
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke(initial_state, config=config)
```

### 14.4.4 获取状态快照

```python
# 获取当前状态
state_snapshot = app.get_state(config)

print("Current state:", state_snapshot.values)
print("Next node:", state_snapshot.next)
print("Checkpoint ID:", state_snapshot.config["configurable"]["checkpoint_id"])

# 获取历史状态
for state in app.get_state_history(config, limit=5):
    print(f"Checkpoint {state.config['configurable']['checkpoint_id']}")
    print(f"  Messages: {len(state.values['messages'])}")
    print(f"  Next: {state.next}")
```

<div data-component="CheckpointTimeline"></div>

---

## 14.5 Human-in-the-Loop（人机协作）

### 14.5.1 中断与恢复

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

class ApprovalState(TypedDict):
    messages: Annotated[list, add]
    draft: str
    approved: bool

def generate_draft(state: ApprovalState):
    """生成草稿"""
    response = llm.invoke(state["messages"])
    return {"draft": response.content}

def human_approval(state: ApprovalState):
    """人工审批节点（会中断）"""
    # 这个节点会自动中断，等待人工输入
    pass

def finalize(state: ApprovalState):
    """最终化"""
    return {"messages": [AIMessage(content=state["draft"])]}

workflow = StateGraph(ApprovalState)
workflow.add_node("draft", generate_draft)
workflow.add_node("approval", human_approval)
workflow.add_node("finalize", finalize)

workflow.set_entry_point("draft")
workflow.add_edge("draft", "approval")
workflow.add_conditional_edges(
    "approval",
    lambda s: "approve" if s.get("approved") else "reject",
    {
        "approve": "finalize",
        "reject": END
    }
)
workflow.add_edge("finalize", END)

# 编译时指定中断节点
app = workflow.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["approval"]  # 在 approval 前中断
)

# 第一次执行（会在 approval 前中断）
config = {"configurable": {"thread_id": "approval-1"}}
result = app.invoke(
    {"messages": [HumanMessage(content="Write a poem")]},
    config=config
)

print("Draft:", result["draft"])
print("Status: PENDING APPROVAL")

# 人工审批后继续
app.update_state(
    config,
    {"approved": True}
)

# 继续执行
final_result = app.invoke(None, config=config)
print("Final:", final_result["messages"][-1].content)
```

### 14.5.2 动态修改状态

```python
# 获取当前状态
current_state = app.get_state(config)

# 修改草稿
app.update_state(
    config,
    {"draft": "Modified draft content"}
)

# 继续执行
result = app.invoke(None, config=config)
```

---

## 14.6 流式执行

### 14.6.1 流式输出节点

```python
# 流式执行
for event in app.stream(
    {"messages": [HumanMessage(content="Tell me a story")]},
    config=config
):
    for node_name, node_output in event.items():
        print(f"\n=== {node_name} ===")
        print(node_output)
```

**输出示例**：
```
=== agent ===
{'messages': [AIMessage(content='Once upon a time...')]}

=== __end__ ===
{'messages': [HumanMessage(...), AIMessage(...)]}
```

### 14.6.2 流式 Token

```python
async for event in app.astream_events(
    {"messages": [HumanMessage(content="Hello")]},
    config=config,
    version="v1"
):
    kind = event["event"]
    
    if kind == "on_chat_model_stream":
        content = event["data"]["chunk"].content
        if content:
            print(content, end="", flush=True)
```

---

## 14.7 子图与模块化

### 14.7.1 创建子图

```python
# 子图：数据验证
def create_validation_graph():
    class ValidationState(TypedDict):
        data: dict
        is_valid: bool
        errors: list
    
    def validate(state: ValidationState):
        # 验证逻辑
        data = state["data"]
        errors = []
        
        if "email" not in data:
            errors.append("Missing email")
        if "age" not in data or data["age"] < 0:
            errors.append("Invalid age")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors
        }
    
    subgraph = StateGraph(ValidationState)
    subgraph.add_node("validate", validate)
    subgraph.set_entry_point("validate")
    subgraph.add_edge("validate", END)
    
    return subgraph.compile()

# 主图中使用子图
validation_graph = create_validation_graph()

def process_with_validation(state: MainState):
    # 调用子图
    validation_result = validation_graph.invoke({
        "data": state["user_data"]
    })
    
    if not validation_result["is_valid"]:
        return {"error": validation_result["errors"]}
    
    # 继续处理
    return {"status": "validated"}
```

---

## 14.8 实战案例：Multi-Agent 协作系统

```python
from typing import Literal

class ResearchState(TypedDict):
    messages: Annotated[list, add]
    topic: str
    research_data: dict
    draft: str
    final_report: str
    current_agent: str

# Agent 1: 研究员
def researcher(state: ResearchState):
    """收集信息"""
    topic = state["topic"]
    prompt = f"Research the topic: {topic}"
    response = llm_with_tools.invoke([HumanMessage(content=prompt)])
    
    return {
        "messages": [response],
        "research_data": {"findings": response.content},
        "current_agent": "researcher"
    }

# Agent 2: 作家
def writer(state: ResearchState):
    """撰写报告"""
    research = state["research_data"]
    prompt = f"Write a report based on: {research}"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "messages": [response],
        "draft": response.content,
        "current_agent": "writer"
    }

# Agent 3: 编辑
def editor(state: ResearchState):
    """编辑润色"""
    draft = state["draft"]
    prompt = f"Edit and improve: {draft}"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "messages": [response],
        "final_report": response.content,
        "current_agent": "editor"
    }

# 路由逻辑
def route_next(state: ResearchState) -> Literal["writer", "editor", "end"]:
    current = state.get("current_agent", "")
    
    if current == "researcher":
        return "writer"
    elif current == "writer":
        return "editor"
    else:
        return "end"

# 构建协作图
workflow = StateGraph(ResearchState)

workflow.add_node("researcher", researcher)
workflow.add_node("writer", writer)
workflow.add_node("editor", editor)

workflow.set_entry_point("researcher")

workflow.add_conditional_edges(
    "researcher",
    route_next,
    {"writer": "writer"}
)

workflow.add_conditional_edges(
    "writer",
    route_next,
    {"editor": "editor"}
)

workflow.add_conditional_edges(
    "editor",
    route_next,
    {"end": END}
)

app = workflow.compile(checkpointer=MemorySaver())

# 执行
result = app.invoke(
    {"topic": "LangGraph best practices"},
    config={"configurable": {"thread_id": "research-1"}}
)

print("Final Report:")
print(result["final_report"])
```

---

## 14.9 调试与可视化

### 14.9.1 可视化图结构

```python
from IPython.display import Image, display

# 生成 Mermaid 图
display(Image(app.get_graph().draw_mermaid_png()))
```

### 14.9.2 调试模式

```python
# 打印每个节点的输入输出
for event in app.stream(initial_state, config=config, debug=True):
    print(event)
```

### 14.9.3 时间旅行调试

```python
# 获取历史状态
history = list(app.get_state_history(config, limit=10))

# 回到第 3 个 checkpoint
third_checkpoint = history[2]

# 从该点重新执行
result = app.invoke(
    None,
    config=third_checkpoint.config
)
```

---

## 14.10 最佳实践

### 14.10.1 状态设计原则

✅ **保持状态最小化**：只存储必要信息  
✅ **使用类型注解**：利用 TypedDict 定义清晰结构  
✅ **合理使用 Annotated**：为列表字段指定合并策略  
✅ **避免大对象**：大文件用引用，不直接存储在状态中

### 14.10.2 节点设计原则

✅ **单一职责**：每个节点只做一件事  
✅ **幂等性**：相同输入产生相同输出  
✅ **错误处理**：在节点内捕获异常，返回错误状态  
✅ **日志记录**：记录节点执行情况便于调试

### 14.10.3 性能优化

```python
# 使用异步节点
async def async_node(state: AgentState):
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

# 并行执行多个节点
workflow.add_edge(START, ["node1", "node2", "node3"])
```

---

## 本章小结

本章系统学习了 LangGraph 的核心机制：

1. **StateGraph**：有状态的图编排框架，支持循环与条件控制
2. **节点与边**：定义处理逻辑和转换关系
3. **Checkpoint**：持久化状态，支持恢复与时间旅行
4. **Human-in-the-Loop**：人机协作，中断与恢复机制
5. **流式执行**：实时输出中间状态
6. **Multi-Agent**：构建复杂的多 Agent 协作系统

**下一章预告**：Chapter 15 将深入学习 Agent 系统设计，包括 ReAct、Planning、Reflection、Multi-Agent 架构等高级模式。

---

## 扩展阅读

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [Human-in-the-Loop 最佳实践](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/)
- [Multi-Agent 系统设计](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/)
- [Checkpoint 持久化指南](https://langchain-ai.github.io/langgraph/how-tos/persistence/)

# Chapter 15: Agent 系统设计

> **本章目标**：深入掌握 LangChain/LangGraph 中的 Agent 设计模式，学习 ReAct、Planning、Reflection、Multi-Agent 等高级架构，理解工具调用、错误处理、长期任务管理等生产级技术，构建可靠、可扩展的 AI Agent 系统。

## 15.1 什么是 Agent？

### 15.1.1 Agent 的核心特征

Agent（智能体）是能够**自主决策、调用工具、完成复杂任务**的 AI 系统：

**自主性**：根据环境动态选择行动，无需硬编码流程  
**工具使用**：调用外部 API、数据库、计算器等工具  
**多轮推理**：通过观察-思考-行动循环逐步解决问题  
**目标导向**：持续执行直到达成目标或超时

### 15.1.2 Agent vs Chain

| 维度 | Chain | Agent |
|------|-------|-------|
| **流程** | 预定义（线性/并行） | 动态决策 |
| **工具调用** | 固定位置 | 自主选择何时调用 |
| **复杂度** | 简单 | 高 |
| **可控性** | 强 | 弱（需要防护栏） |
| **适用场景** | 固定流程任务 | 开放式问题求解 |

<div data-component="AgentArchitectureComparison"></div>

---

## 15.2 ReAct Agent 模式

### 15.2.1 ReAct 原理

**ReAct = Reasoning（推理） + Acting（行动）**

核心循环：
```
1. Thought: 我需要做什么？
2. Action: 调用工具 X
3. Observation: 工具返回结果 Y
4. Thought: 基于 Y，我需要…
5. Action: 调用工具 Z
6. Observation: …
7. Thought: 我现在可以回答了
8. Final Answer: …
```

### 15.2.2 使用 create_react_agent

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # 模拟 API 调用
    weather_data = {
        "Beijing": "Sunny, 25°C",
        "Shanghai": "Rainy, 18°C",
        "London": "Cloudy, 15°C"
    }
    return weather_data.get(city, "Unknown city")

@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

tools = [get_weather, calculator]
llm = ChatOpenAI(model="gpt-4", temperature=0)

# 创建 ReAct Agent
agent = create_react_agent(llm, tools)

# 执行
from langchain_core.messages import HumanMessage

result = agent.invoke({
    "messages": [
        HumanMessage(content="What's the weather in Beijing? Also calculate 15 * 7.")
    ]
})

for msg in result["messages"]:
    print(f"{msg.__class__.__name__}: {msg.content}")
```

**执行轨迹**：
```
HumanMessage: What's the weather in Beijing? Also calculate 15 * 7.

AIMessage: [Thought] I need to get weather and do calculation
[Action] get_weather(city="Beijing")

ToolMessage: Sunny, 25°C

AIMessage: [Action] calculator(expression="15 * 7")

ToolMessage: 105

AIMessage: The weather in Beijing is Sunny, 25°C. 15 * 7 = 105.
```

### 15.2.3 自定义 ReAct Agent

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Literal
from operator import add
from langchain_core.messages import AIMessage, ToolMessage

class AgentState(TypedDict):
    messages: Annotated[list, add]
    iterations: int

def call_model(state: AgentState):
    """调用 LLM"""
    messages = state["messages"]
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke(messages)
    
    return {
        "messages": [response],
        "iterations": state["iterations"] + 1
    }

def execute_tools(state: AgentState):
    """执行工具调用"""
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    
    tool_messages = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # 找到对应工具
        selected_tool = None
        for t in tools:
            if t.name == tool_name:
                selected_tool = t
                break
        
        if selected_tool:
            tool_output = selected_tool.invoke(tool_args)
        else:
            tool_output = f"Error: Tool {tool_name} not found"
        
        tool_messages.append(
            ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_call["id"]
            )
        )
    
    return {"messages": tool_messages}

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """判断是否继续"""
    last_message = state["messages"][-1]
    
    # 超过最大迭代
    if state["iterations"] >= 10:
        return "end"
    
    # 有工具调用
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    return "end"

# 构建图
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tools)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

workflow.add_edge("tools", "agent")

react_agent = workflow.compile()

# 使用
result = react_agent.invoke({
    "messages": [HumanMessage(content="Calculate 25 * 4")],
    "iterations": 0
})
```

---

## 15.3 工具系统设计

### 15.3.1 工具定义最佳实践

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(..., description="City name, e.g., Beijing, London")
    unit: str = Field("celsius", description="Temperature unit: celsius or fahrenheit")

@tool(args_schema=WeatherInput)
def get_weather(city: str, unit: str = "celsius") -> str:
    """
    Get current weather for a city.
    
    This tool provides real-time weather information.
    Use it when user asks about weather conditions.
    """
    # 实现逻辑
    weather = fetch_weather_api(city)
    
    if unit == "fahrenheit":
        temp = celsius_to_fahrenheit(weather["temp"])
    else:
        temp = weather["temp"]
    
    return f"{weather['condition']}, {temp}°{unit[0].upper()}"
```

**关键要点**：
- ✅ 使用 Pydantic schema 定义参数类型
- ✅ 提供详细的 docstring（LLM 会读取）
- ✅ 在 Field 中添加 description（帮助 LLM 理解）
- ✅ 处理错误情况并返回有意义的错误信息

### 15.3.2 工具错误处理

```python
from langchain_core.tools import ToolException

@tool
def database_query(sql: str) -> str:
    """Execute SQL query on database."""
    try:
        # 验证 SQL（防止注入）
        if any(keyword in sql.upper() for keyword in ["DROP", "DELETE", "UPDATE"]):
            raise ToolException("Only SELECT queries are allowed")
        
        # 执行查询
        result = execute_sql(sql)
        return str(result)
    
    except ToolException:
        raise  # 重新抛出工具异常
    except Exception as e:
        raise ToolException(f"Database error: {str(e)}")

# 在 Agent 中捕获工具异常
def execute_tools_with_error_handling(state: AgentState):
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    
    tool_messages = []
    for tool_call in tool_calls:
        try:
            tool_output = selected_tool.invoke(tool_args)
            status = "success"
        except ToolException as e:
            tool_output = f"Tool Error: {str(e)}"
            status = "error"
        except Exception as e:
            tool_output = f"Unexpected Error: {str(e)}"
            status = "error"
        
        tool_messages.append(
            ToolMessage(
                content=tool_output,
                tool_call_id=tool_call["id"],
                status=status
            )
        )
    
    return {"messages": tool_messages}
```

### 15.3.3 工具重试机制

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@tool
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def api_call(endpoint: str) -> str:
    """Call external API with retry."""
    response = requests.get(endpoint, timeout=10)
    response.raise_for_status()
    return response.json()
```

<div data-component="ToolCallFlow"></div>

---

## 15.4 Planning Agent

### 15.4.1 Plan-and-Execute 模式

与 ReAct 不同，Planning Agent **先制定完整计划，再逐步执行**：

```python
from langgraph.graph import StateGraph, END

class PlanExecuteState(TypedDict):
    input: str
    plan: list[str]
    past_steps: Annotated[list, add]
    response: str

def plan_step(state: PlanExecuteState):
    """制定计划"""
    prompt = f"""
    For the following objective, create a step-by-step plan:
    
    Objective: {state['input']}
    
    Output a numbered list of steps.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    plan_text = response.content
    
    # 解析计划
    steps = []
    for line in plan_text.split('\n'):
        if line.strip() and line[0].isdigit():
            steps.append(line.split('.', 1)[1].strip())
    
    return {"plan": steps}

def execute_step(state: PlanExecuteState):
    """执行下一步"""
    if not state["plan"]:
        return {"response": "Plan completed"}
    
    current_step = state["plan"][0]
    remaining_plan = state["plan"][1:]
    
    prompt = f"""
    Execute this step: {current_step}
    
    Context from previous steps:
    {state['past_steps']}
    
    Available tools: {[tool.name for tool in tools]}
    """
    
    # 使用工具执行
    agent_response = react_agent.invoke({
        "messages": [HumanMessage(content=prompt)]
    })
    
    result = agent_response["messages"][-1].content
    
    return {
        "plan": remaining_plan,
        "past_steps": [f"Step: {current_step}\nResult: {result}"]
    }

def should_continue_plan(state: PlanExecuteState) -> Literal["execute", "end"]:
    if state["plan"]:
        return "execute"
    return "end"

# 构建 Plan-and-Execute 图
workflow = StateGraph(PlanExecuteState)

workflow.add_node("planner", plan_step)
workflow.add_node("executor", execute_step)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")

workflow.add_conditional_edges(
    "executor",
    should_continue_plan,
    {
        "execute": "executor",
        "end": END
    }
)

plan_agent = workflow.compile()

# 使用
result = plan_agent.invoke({
    "input": "Research LangGraph, summarize key features, and write a blog post"
})
```

**执行示例**：
```
[Plan]
1. Search for LangGraph documentation
2. Extract key features
3. Organize information
4. Write blog post draft
5. Review and finalize

[Execute Step 1] Search for LangGraph documentation → Found docs
[Execute Step 2] Extract key features → Listed 5 features
[Execute Step 3] Organize information → Created outline
[Execute Step 4] Write blog post draft → Generated 500-word draft
[Execute Step 5] Review and finalize → Polished final version
```

### 15.4.2 动态重新规划

```python
def replan_if_needed(state: PlanExecuteState) -> Literal["replan", "execute", "end"]:
    """根据执行结果决定是否重新规划"""
    if not state["plan"]:
        return "end"
    
    # 检查最后一步是否失败
    if state["past_steps"] and "Error" in state["past_steps"][-1]:
        return "replan"
    
    return "execute"

workflow.add_conditional_edges(
    "executor",
    replan_if_needed,
    {
        "replan": "planner",  # 重新规划
        "execute": "executor",
        "end": END
    }
)
```

---

## 15.5 Reflection Agent（自我批评）

### 15.5.1 生成-反思-改进循环

```python
class ReflectionState(TypedDict):
    input: str
    draft: str
    critique: str
    final_output: str
    iterations: int

def generate(state: ReflectionState):
    """生成初稿"""
    prompt = f"Write content for: {state['input']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "draft": response.content,
        "iterations": state["iterations"] + 1
    }

def reflect(state: ReflectionState):
    """反思与批评"""
    prompt = f"""
    Review the following draft and provide critique:
    
    Draft:
    {state['draft']}
    
    Identify:
    1. Factual errors
    2. Logical inconsistencies
    3. Missing information
    4. Areas for improvement
    
    If the draft is good, say "APPROVED".
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"critique": response.content}

def improve(state: ReflectionState):
    """改进草稿"""
    prompt = f"""
    Improve the draft based on this critique:
    
    Original Draft:
    {state['draft']}
    
    Critique:
    {state['critique']}
    
    Generate an improved version.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "draft": response.content,
        "iterations": state["iterations"] + 1
    }

def should_continue_reflection(state: ReflectionState) -> Literal["improve", "end"]:
    # 超过最大迭代
    if state["iterations"] >= 5:
        return "end"
    
    # 批评中包含 "APPROVED"
    if "APPROVED" in state["critique"]:
        return "end"
    
    return "improve"

# 构建 Reflection 图
workflow = StateGraph(ReflectionState)

workflow.add_node("generate", generate)
workflow.add_node("reflect", reflect)
workflow.add_node("improve", improve)

workflow.set_entry_point("generate")
workflow.add_edge("generate", "reflect")

workflow.add_conditional_edges(
    "reflect",
    should_continue_reflection,
    {
        "improve": "improve",
        "end": END
    }
)

workflow.add_edge("improve", "reflect")

reflection_agent = workflow.compile()

# 使用
result = reflection_agent.invoke({
    "input": "Explain quantum computing to a 10-year-old",
    "iterations": 0
})
```

<InteractiveComponent name="ReflectionLoopVisualizer" />

---

## 15.6 Multi-Agent 协作

### 15.6.1 Supervisor 模式

一个主控 Agent 分配任务给多个专业 Agent：

```python
class SupervisorState(TypedDict):
    messages: Annotated[list, add]
    next_agent: str

# 专业 Agent
def researcher_agent(state: SupervisorState):
    """研究员 Agent"""
    last_message = state["messages"][-1].content
    prompt = f"Research the following topic: {last_message}"
    response = llm_with_tools.invoke([HumanMessage(content=prompt)])
    
    return {"messages": [response]}

def writer_agent(state: SupervisorState):
    """作家 Agent"""
    research_content = state["messages"][-1].content
    prompt = f"Write an article based on: {research_content}"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"messages": [response]}

def editor_agent(state: SupervisorState):
    """编辑 Agent"""
    draft = state["messages"][-1].content
    prompt = f"Edit and improve: {draft}"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"messages": [response]}

# Supervisor
supervisor_prompt = """
You are a supervisor managing these workers:
- Researcher: Gathers information
- Writer: Creates content
- Editor: Polishes drafts

Given the conversation, decide which worker should act next.
Options: Researcher, Writer, Editor, FINISH
"""

def supervisor(state: SupervisorState):
    """主控 Agent"""
    messages = state["messages"]
    
    response = llm.invoke([
        SystemMessage(content=supervisor_prompt),
        *messages,
        HumanMessage(content="Who should act next?")
    ])
    
    next_agent = response.content.strip()
    
    return {"next_agent": next_agent}

def route_supervisor(state: SupervisorState) -> Literal["researcher", "writer", "editor", "end"]:
    next_agent = state["next_agent"].lower()
    
    if "researcher" in next_agent:
        return "researcher"
    elif "writer" in next_agent:
        return "writer"
    elif "editor" in next_agent:
        return "editor"
    else:
        return "end"

# 构建 Multi-Agent 图
workflow = StateGraph(SupervisorState)

workflow.add_node("supervisor", supervisor)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)
workflow.add_node("editor", editor_agent)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "researcher": "researcher",
        "writer": "writer",
        "editor": "editor",
        "end": END
    }
)

workflow.add_edge("researcher", "supervisor")
workflow.add_edge("writer", "supervisor")
workflow.add_edge("editor", "supervisor")

multi_agent = workflow.compile()

# 使用
result = multi_agent.invoke({
    "messages": [HumanMessage(content="Create a blog post about LangGraph")]
})
```

### 15.6.2 Hierarchical Agent

```python
# 高层 Agent
def manager_agent(state):
    """经理 Agent - 分解任务"""
    task = state["task"]
    
    prompt = f"""
    Break down this task into subtasks:
    {task}
    
    Assign each subtask to: DataAnalyst, Engineer, or Designer
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # 解析子任务
    subtasks = parse_subtasks(response.content)
    
    return {"subtasks": subtasks}

# 执行层 Agent
def execute_subtasks(state):
    """执行子任务"""
    results = []
    
    for subtask in state["subtasks"]:
        if subtask["assignee"] == "DataAnalyst":
            result = data_analyst_agent.invoke({"task": subtask["description"]})
        elif subtask["assignee"] == "Engineer":
            result = engineer_agent.invoke({"task": subtask["description"]})
        else:
            result = designer_agent.invoke({"task": subtask["description"]})
        
        results.append(result)
    
    return {"results": results}
```

<div data-component="MultiAgentArchitecture"></div>

---

## 15.7 长期任务与内存管理

### 15.7.1 任务状态持久化

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 长时间运行的任务
checkpointer = SqliteSaver.from_conn_string("long_task.db")

long_agent = workflow.compile(checkpointer=checkpointer)

# 开始任务
config = {"configurable": {"thread_id": "task-12345"}}

result = long_agent.invoke(
    {"messages": [HumanMessage(content="Start research")]},
    config=config
)

# 稍后恢复
later_result = long_agent.invoke(
    {"messages": [HumanMessage(content="Continue from where we left off")]},
    config=config
)
```

### 15.7.2 长期记忆（External Memory）

```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 创建长期记忆
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(["Initial knowledge"], embeddings)

memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

def agent_with_memory(state: AgentState):
    """带长期记忆的 Agent"""
    current_input = state["messages"][-1].content
    
    # 检索相关记忆
    relevant_memory = memory.load_memory_variables({"input": current_input})
    
    # 构建提示
    prompt = f"""
    Current Input: {current_input}
    
    Relevant Past Information:
    {relevant_memory['history']}
    
    How would you respond?
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # 保存新记忆
    memory.save_context(
        {"input": current_input},
        {"output": response.content}
    )
    
    return {"messages": [response]}
```

---

## 15.8 Agent 可靠性工程

### 15.8.1 防护栏（Guardrails）

```python
def validate_output(state: AgentState) -> Literal["valid", "retry"]:
    """验证 Agent 输出"""
    last_message = state["messages"][-1].content
    
    # 检查有害内容
    if contains_harmful_content(last_message):
        return "retry"
    
    # 检查格式
    if not is_valid_format(last_message):
        return "retry"
    
    # 检查事实准确性
    if not fact_check(last_message):
        return "retry"
    
    return "valid"

workflow.add_conditional_edges(
    "agent",
    validate_output,
    {
        "valid": "end",
        "retry": "agent"  # 重新生成
    }
)
```

### 15.8.2 Rate Limiting

```python
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_calls: int, window: int):
        self.max_calls = max_calls
        self.window = window
        self.calls = defaultdict(list)
    
    def allow(self, key: str) -> bool:
        now = time.time()
        self.calls[key] = [t for t in self.calls[key] if now - t < self.window]
        
        if len(self.calls[key]) >= self.max_calls:
            return False
        
        self.calls[key].append(now)
        return True

limiter = RateLimiter(max_calls=10, window=60)

def rate_limited_llm_call(state: AgentState):
    """限流的 LLM 调用"""
    user_id = state.get("user_id", "default")
    
    if not limiter.allow(user_id):
        raise Exception("Rate limit exceeded. Please wait.")
    
    return call_model(state)
```

### 15.8.3 超时控制

```python
import asyncio

async def agent_with_timeout(state: AgentState):
    """带超时的 Agent"""
    try:
        result = await asyncio.wait_for(
            async_agent.ainvoke(state),
            timeout=30.0
        )
        return result
    except asyncio.TimeoutError:
        return {
            "messages": [AIMessage(content="Task timeout. Stopping execution.")],
            "status": "timeout"
        }
```

---

## 15.9 实战案例：客服 Agent 系统

```python
class CustomerServiceState(TypedDict):
    messages: Annotated[list, add]
    customer_id: str
    intent: str
    resolved: bool
    escalated: bool

# 意图识别
def classify_intent(state: CustomerServiceState):
    """识别用户意图"""
    last_message = state["messages"][-1].content
    
    prompt = f"""
    Classify the customer's intent:
    
    Message: {last_message}
    
    Options:
    - technical_support
    - billing_question
    - product_inquiry
    - complaint
    - other
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    intent = response.content.strip().lower()
    
    return {"intent": intent}

# 技术支持 Agent
def technical_support_agent(state: CustomerServiceState):
    """处理技术问题"""
    # 查询知识库
    kb_result = knowledge_base_tool.invoke({"query": state["messages"][-1].content})
    
    prompt = f"""
    Customer issue: {state['messages'][-1].content}
    
    Knowledge base info: {kb_result}
    
    Provide a solution.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"messages": [response]}

# 账单 Agent
def billing_agent(state: CustomerServiceState):
    """处理账单问题"""
    # 查询账单系统
    billing_info = billing_system_tool.invoke({"customer_id": state["customer_id"]})
    
    response = llm.invoke([
        HumanMessage(content=f"Billing info: {billing_info}"),
        state["messages"][-1]
    ])
    
    return {"messages": [response]}

# 升级到人工
def escalate_to_human(state: CustomerServiceState):
    """升级到人工客服"""
    return {
        "messages": [AIMessage(content="Transferring to human agent...")],
        "escalated": True
    }

def route_by_intent(state: CustomerServiceState) -> str:
    intent = state["intent"]
    
    if intent == "technical_support":
        return "technical"
    elif intent == "billing_question":
        return "billing"
    elif intent == "complaint":
        return "escalate"
    else:
        return "general"

# 构建客服系统
workflow = StateGraph(CustomerServiceState)

workflow.add_node("classify", classify_intent)
workflow.add_node("technical", technical_support_agent)
workflow.add_node("billing", billing_agent)
workflow.add_node("escalate", escalate_to_human)

workflow.set_entry_point("classify")

workflow.add_conditional_edges(
    "classify",
    route_by_intent,
    {
        "technical": "technical",
        "billing": "billing",
        "escalate": "escalate",
        "general": END
    }
)

workflow.add_edge("technical", END)
workflow.add_edge("billing", END)
workflow.add_edge("escalate", END)

customer_service = workflow.compile(
    checkpointer=SqliteSaver.from_conn_string("customer_service.db")
)

# 使用
result = customer_service.invoke({
    "messages": [HumanMessage(content="My app keeps crashing")],
    "customer_id": "CUST-12345"
}, config={"configurable": {"thread_id": "session-001"}})
```

---

## 15.10 最佳实践总结

### 15.10.1 Agent 设计原则

✅ **明确任务边界**：定义 Agent 能做什么、不能做什么  
✅ **提供丰富上下文**：给 LLM 足够信息做决策  
✅ **限制最大迭代**：防止无限循环  
✅ **工具文档清晰**：帮助 LLM 正确使用工具  
✅ **错误处理完善**：优雅处理工具失败、超时等  
✅ **日志与可观测性**：记录每步决策便于调试

### 15.10.2 性能优化

```python
# 并行工具调用
async def parallel_tool_execution(tool_calls):
    tasks = [tool.ainvoke(args) for tool, args in tool_calls]
    results = await asyncio.gather(*tasks)
    return results

# 缓存工具结果
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_tool_call(tool_name: str, args: str):
    return tool.invoke(args)

# 流式输出
async for event in agent.astream_events(input_data, version="v1"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
```

### 15.10.3 安全考虑

```python
# 工具权限控制
ALLOWED_TOOLS = {
    "user_role_1": ["search", "calculator"],
    "user_role_2": ["search", "calculator", "database_query"],
    "admin": ["search", "calculator", "database_query", "system_command"]
}

def filter_tools_by_role(user_role: str):
    all_tools = [search, calculator, database_query, system_command]
    allowed = ALLOWED_TOOLS.get(user_role, [])
    return [t for t in all_tools if t.name in allowed]

# 输入验证
def validate_user_input(user_input: str) -> bool:
    # 检查恶意输入
    if contains_injection_patterns(user_input):
        return False
    
    # 长度限制
    if len(user_input) > 1000:
        return False
    
    return True
```

---

## 本章小结

本章深入学习了 Agent 系统设计的核心模式：

1. **ReAct Agent**：推理-行动循环，动态工具调用
2. **Planning Agent**：先规划后执行，支持动态重新规划
3. **Reflection Agent**：生成-反思-改进，自我批评机制
4. **Multi-Agent**：Supervisor、Hierarchical 等协作模式
5. **可靠性工程**：防护栏、限流、超时、错误处理
6. **长期任务管理**：Checkpoint、外部记忆、状态持久化

---

## 扩展阅读

- [LangGraph Agent 教程](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- [ReAct 论文](https://arxiv.org/abs/2210.03629)
- [Tool Calling 最佳实践](https://python.langchain.com/docs/how_to/tool_calling/)
- [Multi-Agent 系统设计](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/)
- [Agent 安全指南](https://python.langchain.com/docs/security/)

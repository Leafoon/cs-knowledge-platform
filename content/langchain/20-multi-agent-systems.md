# Chapter 20: 多 Agent 系统

## 本章概览

单个 Agent 虽然已经能够处理复杂任务，但在面对大规模、多领域、需要专业分工的问题时，多 Agent 协作系统展现出更强大的能力。本章将深入探讨多 Agent 架构设计模式，学习如何构建高效的 Agent 团队，掌握 Supervisor、Hierarchical、Collaborative 等协作模式，以及 Agent 间通信、任务分配、结果聚合等关键技术。

**本章重点**：
- 多 Agent 架构模式与适用场景
- Supervisor 模式：中心化调度
- Hierarchical 模式：层级委派
- Collaborative 模式：平等协作
- Agent 间通信机制
- 实战案例：研究助手、客服系统、代码生成

---

## 20.1 为什么需要多 Agent？

### 20.1.1 单 Agent 的局限性

单个 Agent 在处理复杂任务时面临的挑战：

| 挑战 | 描述 | 影响 |
|------|------|------|
| **上下文爆炸** | 复杂任务需要大量上下文信息 | 超出模型上下文窗口 |
| **专业能力不足** | 单个 LLM 难以精通所有领域 | 输出质量下降 |
| **推理链过长** | 多步骤推理容易出错 | 错误累积，成功率低 |
| **工具调用复杂** | 需要同时使用多种工具 | 选择困难，效率低下 |
| **可维护性差** | 所有逻辑集中在一个 Agent | 难以调试和优化 |

### 20.1.2 多 Agent 系统的优势

```python
# 单 Agent 方式：一个 Agent 处理所有任务
single_agent = create_react_agent(
    llm=ChatOpenAI(),
    tools=[
        search_tool,           # 搜索
        calculator_tool,       # 计算
        code_execution_tool,   # 代码执行
        database_tool,         # 数据库查询
        email_tool,           # 发送邮件
        # ... 20+ 工具
    ]
)
# 问题：工具选择困难、上下文混乱、难以优化

# 多 Agent 方式：专业化分工
research_agent = create_react_agent(llm, tools=[search_tool, summarize_tool])
analyst_agent = create_react_agent(llm, tools=[calculator_tool, visualization_tool])
writer_agent = create_react_agent(llm, tools=[grammar_tool, style_tool])
# 优势：专业化、可组合、易维护
```

**多 Agent 核心优势**：
1. **专业化分工** - 每个 Agent 专注特定领域，提高专业性
2. **任务并行** - 多个 Agent 可同时工作，提升效率
3. **容错能力** - 某个 Agent 失败不影响整体系统
4. **模块化设计** - 易于添加、替换、升级 Agent
5. **可扩展性** - 根据需求动态增加 Agent 数量

### 20.1.3 多 Agent 应用场景

- **研究助手系统** - 搜索 Agent + 分析 Agent + 写作 Agent
- **客服系统** - 路由 Agent + 专家 Agent + 升级 Agent
- **代码生成系统** - 规划 Agent + 编码 Agent + 测试 Agent + 审查 Agent
- **内容创作** - 主题研究 → 大纲生成 → 内容撰写 → 编辑校对
- **数据分析** - 数据收集 → 清洗 → 分析 → 可视化 → 报告生成

---

## 20.2 多 Agent 架构模式

### 20.2.1 三大核心模式对比

<div data-component="MultiAgentArchitectureComparison"></div>

| 模式 | 结构 | 通信方式 | 适用场景 |
|------|------|----------|----------|
| **Supervisor** | 星型（一个中心节点） | 中心化调度 | 明确的任务分解、需要统一协调 |
| **Hierarchical** | 树型（多层级） | 层级传递 | 大型组织、复杂任务分解 |
| **Collaborative** | 网状（点对点） | 平等协作 | 创意型任务、需要多方观点 |

### 20.2.2 模式选择决策树

```
任务是否有明确的子任务分解？
├─ 是 → 子任务之间有依赖关系吗？
│   ├─ 强依赖 → Supervisor 模式
│   └─ 弱依赖 → Collaborative 模式
└─ 否 → 需要多层管理吗？
    ├─ 是 → Hierarchical 模式
    └─ 否 → Collaborative 模式
```

---

## 20.3 Supervisor 模式：中心化调度

### 20.3.1 架构原理

Supervisor 模式由一个中心 Agent 负责任务分解、Agent 选择、结果聚合：

```
用户请求 → Supervisor Agent
              ├─→ Worker Agent 1 (搜索)
              ├─→ Worker Agent 2 (分析)
              └─→ Worker Agent 3 (写作)
              ↓
         结果聚合 → 最终输出
```

**核心职责**：
- **Supervisor**：任务理解、分解、路由、聚合
- **Workers**：执行具体任务，专注单一领域

### 20.3.2 实现 Supervisor Agent

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
import operator

# 定义状态
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    next: str  # 下一个要执行的 Agent

# 定义 Worker Agents
def create_research_agent():
    """搜索和研究专家"""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    from langchain_community.tools import DuckDuckGoSearchRun
    tools = [DuckDuckGoSearchRun()]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的研究助手。
        你的任务是搜索相关信息并提供准确、全面的研究结果。
        使用搜索工具查找最新、最权威的信息。"""),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def create_analyst_agent():
    """数据分析专家"""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    from langchain_community.tools import PythonREPLTool
    tools = [PythonREPLTool()]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个数据分析专家。
        你的任务是分析数据、计算统计指标、发现模式和趋势。
        使用 Python 进行数据分析和计算。"""),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def create_writer_agent():
    """写作专家"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的内容写作专家。
        你的任务是将研究和分析结果整理成清晰、专业、易读的文章。
        注重逻辑结构、语言表达和可读性。"""),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Writer Agent 不需要工具
    return llm | prompt

# 创建 Supervisor Agent
def create_supervisor_chain():
    """创建 Supervisor，负责任务路由"""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    members = ["researcher", "analyst", "writer"]
    system_prompt = f"""你是一个任务分配管理者，负责协调团队成员完成用户请求。
    
团队成员：
- researcher: 负责搜索和研究信息
- analyst: 负责数据分析和计算
- writer: 负责撰写和整理内容

你的职责：
1. 理解用户请求
2. 决定下一步需要哪个团队成员工作
3. 如果任务完成，返回 FINISH

当前可调度的成员：{members}

请只返回下一个工作成员的名字，或者 FINISH。
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "根据上述对话，下一个应该工作的成员是谁？只返回成员名或 FINISH。"),
    ])
    
    return prompt | llm

# 定义节点函数
def supervisor_node(state: AgentState):
    """Supervisor 节点：决定下一步"""
    supervisor_chain = create_supervisor_chain()
    result = supervisor_chain.invoke(state)
    
    next_agent = result.content.strip().lower()
    
    return {
        "next": next_agent if next_agent in ["researcher", "analyst", "writer"] else "FINISH"
    }

def researcher_node(state: AgentState):
    """Research Agent 节点"""
    agent = create_research_agent()
    result = agent.invoke({"messages": state["messages"]})
    
    return {
        "messages": [AIMessage(content=result["output"], name="researcher")]
    }

def analyst_node(state: AgentState):
    """Analyst Agent 节点"""
    agent = create_analyst_agent()
    result = agent.invoke({"messages": state["messages"]})
    
    return {
        "messages": [AIMessage(content=result["output"], name="analyst")]
    }

def writer_node(state: AgentState):
    """Writer Agent 节点"""
    chain = create_writer_agent()
    result = chain.invoke({"messages": state["messages"]})
    
    return {
        "messages": [AIMessage(content=result.content, name="writer")]
    }

# 构建 Supervisor Graph
def build_supervisor_graph():
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)
    
    # 定义路由逻辑
    def router(state: AgentState):
        return state["next"]
    
    # 设置边
    workflow.set_entry_point("supervisor")
    
    workflow.add_conditional_edges(
        "supervisor",
        router,
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "FINISH": END
        }
    )
    
    # Worker 完成后回到 Supervisor
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("analyst", "supervisor")
    workflow.add_edge("writer", "supervisor")
    
    return workflow.compile()

# 使用示例
graph = build_supervisor_graph()

# 执行任务
result = graph.invoke({
    "messages": [
        HumanMessage(content="""请帮我研究 LangGraph 的最新特性，
        分析其与传统 LCEL 的性能对比，并撰写一篇技术博客。""")
    ]
})

# 查看执行过程
for message in result["messages"]:
    print(f"{message.name}: {message.content}\n")
```

**执行流程**：
1. Supervisor 分析任务 → 分配给 researcher
2. Researcher 搜索 LangGraph 信息 → 返回给 Supervisor
3. Supervisor → 分配给 analyst
4. Analyst 分析性能数据 → 返回给 Supervisor
5. Supervisor → 分配给 writer
6. Writer 撰写博客 → 返回给 Supervisor
7. Supervisor → FINISH

<div data-component="SupervisorRoutingFlow"></div>

### 20.3.3 优化 Supervisor 决策

使用 Structured Output 提升路由准确性：

```python
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal

class RouteDecision(BaseModel):
    """Supervisor 的路由决策"""
    next_agent: Literal["researcher", "analyst", "writer", "FINISH"] = Field(
        description="下一个应该工作的 Agent 或 FINISH"
    )
    reason: str = Field(description="选择该 Agent 的原因")

def create_supervisor_with_structure():
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    structured_llm = llm.with_structured_output(RouteDecision)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是任务调度管理者。分析对话历史，决定下一步行动。
        
团队成员：
- researcher: 搜索和研究
- analyst: 数据分析
- writer: 内容写作

如果任务完成，返回 FINISH。"""),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    return prompt | structured_llm

# 使用
supervisor = create_supervisor_with_structure()
decision = supervisor.invoke({
    "messages": [HumanMessage(content="研究 AI Agent 的最新进展")]
})

print(f"下一步: {decision.next_agent}")
print(f"原因: {decision.reason}")
```

### 20.3.4 并行 Worker 执行

Supervisor 可以同时调度多个 Worker：

```python
from langgraph.graph import Send

def supervisor_parallel_node(state: AgentState):
    """支持并行调度的 Supervisor"""
    # 分析任务，决定需要哪些 Agent
    tasks_needed = analyze_task(state["messages"][-1].content)
    
    # 返回多个 Send 指令
    return [
        Send("researcher", state),
        Send("analyst", state)
    ]

# 在 Graph 中使用
workflow.add_conditional_edges(
    "supervisor",
    supervisor_parallel_node,
    ["researcher", "analyst", "writer"]
)
```

---

## 20.4 Hierarchical 模式：层级委派

### 20.4.1 架构原理

Hierarchical 模式模拟企业组织结构，形成管理层级：

```
                Manager (总管理者)
                    |
        +-----------+-----------+
        |           |           |
   Team Lead 1  Team Lead 2  Team Lead 3
        |           |           |
    +---+---+   +---+---+   +---+---+
    |       |   |       |   |       |
Worker1 Worker2 Worker3 Worker4 Worker5 Worker6
```

**适用场景**：
- 大型复杂项目
- 需要多层决策的任务
- 团队规模较大（6+ Agent）

### 20.4.2 实现层级系统

```python
from typing import List

class HierarchicalState(TypedDict):
    messages: Annotated[Sequence, operator.add]
    current_level: str  # 'manager', 'team_lead', 'worker'
    assigned_team: str  # 分配到的团队

# Manager Agent
def create_manager():
    llm = ChatOpenAI(model="gpt-4")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是项目总管理者。
        
你管理三个团队：
1. research_team: 负责信息搜集和研究
2. development_team: 负责代码开发和实现
3. quality_team: 负责测试和质量保证

你的职责：
- 理解项目需求
- 分解为子项目
- 分配给合适的团队
- 协调团队间的协作"""),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    return prompt | llm

# Team Lead Agents
def create_team_lead(team_name: str):
    llm = ChatOpenAI(model="gpt-4")
    
    team_info = {
        "research_team": "你管理研究团队，包括：信息搜索专家、数据收集专家",
        "development_team": "你管理开发团队，包括：前端工程师、后端工程师",
        "quality_team": "你管理质量团队，包括：测试工程师、代码审查员"
    }
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""你是 {team_name} 的团队负责人。
        {team_info[team_name]}
        
你的职责：
- 接收上级分配的任务
- 分解任务给团队成员
- 监督执行进度
- 整合团队产出"""),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    return prompt | llm

# Worker Agents（同前面的示例）

# 构建层级图
def build_hierarchical_graph():
    workflow = StateGraph(HierarchicalState)
    
    # Manager 层
    workflow.add_node("manager", manager_node)
    
    # Team Lead 层
    workflow.add_node("research_lead", team_lead_node("research_team"))
    workflow.add_node("dev_lead", team_lead_node("development_team"))
    workflow.add_node("qa_lead", team_lead_node("quality_team"))
    
    # Worker 层
    workflow.add_node("searcher", searcher_node)
    workflow.add_node("collector", collector_node)
    workflow.add_node("frontend_dev", frontend_node)
    workflow.add_node("backend_dev", backend_node)
    workflow.add_node("tester", tester_node)
    workflow.add_node("reviewer", reviewer_node)
    
    # 设置路由
    workflow.set_entry_point("manager")
    
    # Manager → Team Leads
    workflow.add_conditional_edges(
        "manager",
        route_to_team_lead,
        {
            "research": "research_lead",
            "development": "dev_lead",
            "quality": "qa_lead"
        }
    )
    
    # Team Leads → Workers
    workflow.add_conditional_edges("research_lead", route_to_worker)
    workflow.add_conditional_edges("dev_lead", route_to_worker)
    workflow.add_conditional_edges("qa_lead", route_to_worker)
    
    # Workers → Team Leads (回报)
    workflow.add_edge("searcher", "research_lead")
    workflow.add_edge("frontend_dev", "dev_lead")
    # ... 其他边
    
    return workflow.compile()
```

### 20.4.3 层级通信协议

定义标准化的任务传递格式：

```python
from pydantic import BaseModel
from datetime import datetime

class TaskAssignment(BaseModel):
    """任务分配"""
    task_id: str
    from_level: str  # 'manager', 'team_lead'
    to_agent: str
    description: str
    priority: int
    deadline: datetime

class TaskResult(BaseModel):
    """任务结果"""
    task_id: str
    agent: str
    status: Literal["completed", "failed", "needs_help"]
    output: str
    timestamp: datetime

# 使用示例
def manager_assign_task(state):
    assignment = TaskAssignment(
        task_id="T001",
        from_level="manager",
        to_agent="research_lead",
        description="研究竞品分析",
        priority=1,
        deadline=datetime.now() + timedelta(hours=2)
    )
    
    return {
        "messages": [AIMessage(content=assignment.json(), name="manager")]
    }
```

---

## 20.5 Collaborative 模式：平等协作

### 20.5.1 架构原理

Collaborative 模式中，Agents 地位平等，通过协商、投票、讨论达成共识：

```
    Agent 1 ←→ Agent 2
       ↕          ↕
    Agent 3 ←→ Agent 4
```

**适用场景**：
- 创意型任务（头脑风暴）
- 需要多角度评估的决策
- 没有明确层级的协作

### 20.5.2 实现协作系统

```python
class CollaborativeState(TypedDict):
    messages: Annotated[Sequence, operator.add]
    votes: dict  # Agent 投票结果
    consensus_reached: bool

def create_collaborative_agents():
    """创建多个平等协作的 Agent"""
    
    # Agent 1: 保守派
    conservative_agent = ChatOpenAI(model="gpt-4") | ChatPromptTemplate.from_messages([
        ("system", """你是一个保守谨慎的决策者。
        你倾向于：
        - 选择风险低、稳妥的方案
        - 关注可行性和成本
        - 提出潜在问题和风险"""),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Agent 2: 创新派
    innovative_agent = ChatOpenAI(model="gpt-4") | ChatPromptTemplate.from_messages([
        ("system", """你是一个创新激进的决策者。
        你倾向于：
        - 追求创新和突破
        - 关注长期价值
        - 提出大胆的想法"""),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Agent 3: 实用派
    pragmatic_agent = ChatOpenAI(model="gpt-4") | ChatPromptTemplate.from_messages([
        ("system", """你是一个实用主义者。
        你倾向于：
        - 平衡风险和收益
        - 关注执行效率
        - 提出可落地的方案"""),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    return {
        "conservative": conservative_agent,
        "innovative": innovative_agent,
        "pragmatic": pragmatic_agent
    }

# 实现投票机制
class VoteResult(BaseModel):
    agent_name: str
    decision: Literal["approve", "reject", "abstain"]
    reasoning: str
    confidence: float  # 0-1

def voting_round(state: CollaborativeState, question: str):
    """进行一轮投票"""
    agents = create_collaborative_agents()
    votes = []
    
    for name, agent in agents.items():
        # 每个 Agent 独立给出意见
        messages = state["messages"] + [
            HumanMessage(content=f"请对以下问题投票：{question}")
        ]
        
        structured_agent = agent.with_structured_output(VoteResult)
        vote = structured_agent.invoke({"messages": messages})
        vote.agent_name = name
        votes.append(vote)
    
    # 统计结果
    approve_count = sum(1 for v in votes if v.decision == "approve")
    total_votes = len([v for v in votes if v.decision != "abstain"])
    
    consensus = approve_count / total_votes > 0.67  # 2/3 多数
    
    return {
        "votes": {v.agent_name: v.dict() for v in votes},
        "consensus_reached": consensus
    }

# 构建协作图
def build_collaborative_graph():
    workflow = StateGraph(CollaborativeState)
    
    # 添加讨论轮次
    workflow.add_node("round_1", discussion_node)
    workflow.add_node("voting", voting_node)
    workflow.add_node("consensus_check", consensus_check_node)
    workflow.add_node("refine", refine_proposal_node)
    
    workflow.set_entry_point("round_1")
    
    workflow.add_edge("round_1", "voting")
    workflow.add_conditional_edges(
        "consensus_check",
        lambda state: "finish" if state["consensus_reached"] else "refine",
        {
            "finish": END,
            "refine": "refine"
        }
    )
    workflow.add_edge("refine", "round_1")  # 重新讨论
    
    return workflow.compile()
```

### 20.5.3 讨论与辩论机制

```python
def structured_debate(topic: str, num_rounds: int = 3):
    """结构化辩论"""
    
    # 正方和反方 Agent
    pro_agent = create_debate_agent("支持方")
    con_agent = create_debate_agent("反对方")
    moderator = create_moderator_agent()
    
    messages = [HumanMessage(content=f"辩论主题：{topic}")]
    
    for round_num in range(num_rounds):
        # 正方陈述
        pro_response = pro_agent.invoke({"messages": messages})
        messages.append(AIMessage(content=pro_response.content, name="pro"))
        
        # 反方反驳
        con_response = con_agent.invoke({"messages": messages})
        messages.append(AIMessage(content=con_response.content, name="con"))
        
        # 主持人总结
        summary = moderator.invoke({"messages": messages})
        messages.append(AIMessage(content=summary.content, name="moderator"))
    
    # 最终裁决
    final_decision = moderator.invoke({
        "messages": messages + [
            HumanMessage(content="请根据双方论述给出最终结论")
        ]
    })
    
    return final_decision
```

<div data-component="CollaborativeDebateFlow"></div>

---

## 20.6 Agent 间通信机制

### 20.6.1 消息传递 (Message Passing)

最常见的通信方式：

```python
class AgentMessage(BaseModel):
    from_agent: str
    to_agent: str
    message_type: Literal["request", "response", "notification"]
    content: str
    metadata: dict = {}

def send_message(from_agent: str, to_agent: str, content: str):
    msg = AgentMessage(
        from_agent=from_agent,
        to_agent=to_agent,
        message_type="request",
        content=content
    )
    
    return AIMessage(
        content=msg.json(),
        name=from_agent,
        additional_kwargs={"target": to_agent}
    )
```

### 20.6.2 共享状态 (Shared State)

通过共享状态实现信息交换：

```python
class SharedWorkspace(TypedDict):
    """共享工作空间"""
    research_data: dict  # Researcher 写入
    analysis_results: dict  # Analyst 读取并写入
    draft_content: str  # Writer 读取
    revision_notes: List[str]  # 所有人都可以写入

def researcher_with_shared_state(state: SharedWorkspace):
    # 读取其他 Agent 的输出
    previous_analysis = state.get("analysis_results", {})
    
    # 执行研究
    new_data = perform_research()
    
    # 更新共享状态
    return {
        "research_data": {**state.get("research_data", {}), **new_data}
    }
```

### 20.6.3 事件驱动 (Event-Driven)

基于事件的异步通信：

```python
from enum import Enum

class EventType(Enum):
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    ERROR_OCCURRED = "error_occurred"
    HELP_NEEDED = "help_needed"

class AgentEvent(BaseModel):
    event_type: EventType
    agent: str
    timestamp: datetime
    data: dict

def emit_event(agent_name: str, event_type: EventType, data: dict):
    """发出事件"""
    event = AgentEvent(
        event_type=event_type,
        agent=agent_name,
        timestamp=datetime.now(),
        data=data
    )
    
    # 广播给所有监听者
    notify_listeners(event)

# 监听器示例
def on_task_completed(event: AgentEvent):
    if event.event_type == EventType.TASK_COMPLETED:
        print(f"{event.agent} 完成任务: {event.data}")
        # 触发下一个 Agent
        next_agent.invoke(event.data)
```

---

## 20.7 实战案例

### 20.7.1 案例1：研究助手系统

构建一个完整的研究助手系统，包含搜索、分析、写作三个专业 Agent：

```python
def build_research_assistant():
    """完整的研究助手系统"""
    
    # 1. Researcher Agent - 信息搜集
    from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
    from langchain_community.utilities import WikipediaAPIWrapper
    
    research_tools = [
        WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
        DuckDuckGoSearchRun()
    ]
    
    researcher = create_react_agent(
        llm=ChatOpenAI(model="gpt-4"),
        tools=research_tools,
        prompt="""你是研究专家。任务：
        1. 使用搜索工具查找权威信息
        2. 收集多个来源的数据
        3. 整理关键事实和数据"""
    )
    
    # 2. Analyst Agent - 数据分析
    from langchain_experimental.tools import PythonREPLTool
    
    analyst_tools = [PythonREPLTool()]
    
    analyst = create_react_agent(
        llm=ChatOpenAI(model="gpt-4"),
        tools=analyst_tools,
        prompt="""你是数据分析专家。任务：
        1. 分析研究数据
        2. 计算统计指标
        3. 发现趋势和模式
        4. 生成可视化（代码）"""
    )
    
    # 3. Writer Agent - 内容撰写
    writer_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是专业技术写作专家。任务：
        1. 将研究和分析结果整理成文章
        2. 确保逻辑清晰、结构合理
        3. 使用专业但易懂的语言
        4. 添加必要的引用和数据支持
        
输出格式：
# 标题
## 引言
## 主体（多个小节）
## 结论
## 参考资料"""),
        MessagesPlaceholder(variable_name="messages")
    ])
    
    writer = ChatOpenAI(model="gpt-4", temperature=0.7) | writer_prompt
    
    # 4. 构建 Supervisor Graph
    class ResearchState(TypedDict):
        messages: Annotated[Sequence, operator.add]
        research_results: str
        analysis_results: str
        final_article: str
        next: str
    
    def supervisor_node(state: ResearchState):
        # 简化的路由逻辑
        if not state.get("research_results"):
            return {"next": "researcher"}
        elif not state.get("analysis_results"):
            return {"next": "analyst"}
        elif not state.get("final_article"):
            return {"next": "writer"}
        else:
            return {"next": "FINISH"}
    
    def researcher_node(state: ResearchState):
        result = researcher.invoke({"messages": state["messages"]})
        return {
            "research_results": result["output"],
            "messages": [AIMessage(content=result["output"], name="researcher")]
        }
    
    def analyst_node(state: ResearchState):
        # 将研究结果传给分析器
        analysis_prompt = f"请分析以下研究数据：\n{state['research_results']}"
        result = analyst.invoke({
            "messages": state["messages"] + [HumanMessage(content=analysis_prompt)]
        })
        return {
            "analysis_results": result["output"],
            "messages": [AIMessage(content=result["output"], name="analyst")]
        }
    
    def writer_node(state: ResearchState):
        # 整合研究和分析结果
        writing_prompt = f"""基于以下材料撰写文章：

研究结果：
{state['research_results']}

分析结果：
{state['analysis_results']}"""
        
        result = writer.invoke({
            "messages": state["messages"] + [HumanMessage(content=writing_prompt)]
        })
        return {
            "final_article": result.content,
            "messages": [AIMessage(content=result.content, name="writer")]
        }
    
    # 构建图
    workflow = StateGraph(ResearchState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)
    
    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            "FINISH": END
        }
    )
    
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("analyst", "supervisor")
    workflow.add_edge("writer", "supervisor")
    
    return workflow.compile()

# 使用
assistant = build_research_assistant()

result = assistant.invoke({
    "messages": [
        HumanMessage(content="请研究 LangGraph 的架构设计，分析其优势，并撰写一篇技术文章")
    ]
})

print("=== 最终文章 ===")
print(result["final_article"])
```

### 20.7.2 案例2：客服系统

实现智能客服的多 Agent 协作：

```python
def build_customer_service_system():
    """客服系统：路由 + 专家 + 升级"""
    
    # 1. Router Agent - 意图识别和路由
    class CustomerIntent(BaseModel):
        category: Literal["billing", "technical", "general", "complaint"]
        urgency: Literal["low", "medium", "high", "critical"]
        summary: str
    
    router_llm = ChatOpenAI(model="gpt-4").with_structured_output(CustomerIntent)
    
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是客服路由系统，分析客户问题并分类。"),
        ("user", "{customer_message}")
    ])
    
    router = router_prompt | router_llm
    
    # 2. 专家 Agents
    billing_agent = create_specialist_agent("billing", "账单和支付")
    technical_agent = create_specialist_agent("technical", "技术支持")
    general_agent = create_specialist_agent("general", "常见问题")
    
    # 3. Escalation Agent - 人工客服
    def escalate_to_human(issue: str, history: List):
        return {
            "status": "escalated",
            "message": "已转接人工客服",
            "issue": issue,
            "conversation_history": history
        }
    
    # 4. 构建路由逻辑
    def customer_service_graph():
        class CSState(TypedDict):
            messages: Annotated[Sequence, operator.add]
            intent: CustomerIntent
            resolution: str
            escalated: bool
        
        def router_node(state: CSState):
            intent = router.invoke({
                "customer_message": state["messages"][-1].content
            })
            return {"intent": intent}
        
        def route_to_specialist(state: CSState):
            if state["intent"].urgency == "critical":
                return "escalate"
            return state["intent"].category
        
        def billing_node(state: CSState):
            # 处理账单问题
            resolution = billing_agent.invoke(state["messages"])
            return {"resolution": resolution}
        
        def technical_node(state: CSState):
            resolution = technical_agent.invoke(state["messages"])
            return {"resolution": resolution}
        
        workflow = StateGraph(CSState)
        workflow.add_node("router", router_node)
        workflow.add_node("billing", billing_node)
        workflow.add_node("technical", technical_node)
        workflow.add_node("general", general_node)
        workflow.add_node("escalate", escalate_node)
        
        workflow.set_entry_point("router")
        workflow.add_conditional_edges("router", route_to_specialist)
        
        return workflow.compile()
    
    return customer_service_graph()
```

### 20.7.3 案例3：代码生成系统

多 Agent 协作生成高质量代码：

```python
def build_code_generation_system():
    """代码生成：规划 → 编码 → 测试 → 审查"""
    
    # 1. Planner - 任务规划
    planner_prompt = """你是软件架构师。任务：
    1. 理解需求
    2. 设计系统架构
    3. 分解为模块和函数
    4. 输出实现计划（JSON格式）"""
    
    # 2. Coder - 代码实现
    coder_prompt = """你是资深开发工程师。任务：
    1. 根据计划实现代码
    2. 遵循最佳实践
    3. 添加必要的注释
    4. 确保代码可读性"""
    
    # 3. Tester - 测试
    tester_prompt = """你是测试工程师。任务：
    1. 为代码编写单元测试
    2. 考虑边界情况
    3. 确保测试覆盖率
    4. 执行测试并报告结果"""
    
    # 4. Reviewer - 代码审查
    reviewer_prompt = """你是代码审查专家。任务：
    1. 审查代码质量
    2. 检查潜在bug
    3. 评估性能
    4. 提出改进建议"""
    
    # 构建流程
    class CodeGenState(TypedDict):
        requirement: str
        plan: str
        code: str
        tests: str
        review: dict
        approved: bool
    
    # ... 实现各个节点和流程
    
    return workflow.compile()
```

<div data-component="MultiAgentCodeGenFlow"></div>

---

## 20.8 性能优化与最佳实践

### 20.8.1 减少 Agent 间通信开销

```python
# ❌ 低效：频繁的小消息
for data_point in large_dataset:
    agent.invoke({"data": data_point})  # 100次调用

# ✅ 高效：批量处理
agent.batch([{"data": dp} for dp in large_dataset])  # 1次调用
```

### 20.8.2 Agent 结果缓存

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_agent_call(agent_name: str, input_hash: str):
    """缓存 Agent 调用结果"""
    return agents[agent_name].invoke(input_hash)
```

### 20.8.3 超时和重试策略

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def robust_agent_call(agent, input_data):
    """带重试的 Agent 调用"""
    return agent.invoke(input_data, config={"timeout": 30})
```

### 20.8.4 监控和日志

```python
import logging
from datetime import datetime

class AgentLogger:
    def __init__(self):
        self.logger = logging.getLogger("multi_agent")
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_tokens": 0
        }
    
    def log_agent_call(self, agent_name: str, status: str, tokens: int):
        self.metrics["total_calls"] += 1
        self.metrics["total_tokens"] += tokens
        
        if status == "success":
            self.metrics["successful_calls"] += 1
        else:
            self.metrics["failed_calls"] += 1
        
        self.logger.info(f"[{datetime.now()}] {agent_name}: {status} ({tokens} tokens)")
    
    def get_report(self):
        success_rate = self.metrics["successful_calls"] / self.metrics["total_calls"]
        return {
            **self.metrics,
            "success_rate": f"{success_rate:.2%}"
        }
```

---

## 20.9 常见问题与调试

### 20.9.1 Agent 陷入循环

**问题**：Supervisor 不断在同一 Agent 间循环

```python
# 解决方案：添加循环检测
class StateWithHistory(TypedDict):
    messages: Annotated[Sequence, operator.add]
    agent_history: List[str]  # 记录 Agent 调用历史

def supervisor_with_loop_detection(state: StateWithHistory):
    history = state.get("agent_history", [])
    
    # 检测循环
    if len(history) >= 3 and len(set(history[-3:])) == 1:
        # 连续3次调用同一 Agent，强制终止
        return {"next": "FINISH"}
    
    next_agent = decide_next_agent(state)
    
    return {
        "next": next_agent,
        "agent_history": history + [next_agent]
    }
```

### 20.9.2 Agent 之间信息丢失

**问题**：后续 Agent 看不到前面 Agent 的输出

```python
# 解决方案：使用显式的状态管理
class ExplicitState(TypedDict):
    user_query: str
    research_data: dict  # Researcher 输出
    analysis: dict  # Analyst 输出
    draft: str  # Writer 输出

def researcher_node(state: ExplicitState):
    result = researcher.invoke(state["user_query"])
    return {
        "research_data": {
            "sources": result.sources,
            "summary": result.summary,
            "facts": result.facts
        }
    }

def analyst_node(state: ExplicitState):
    # 明确访问 research_data
    data = state["research_data"]
    analysis = analyze(data)
    return {"analysis": analysis}
```

### 20.9.3 调试工具

```python
def visualize_agent_execution(graph, input_data):
    """可视化 Agent 执行流程"""
    
    execution_log = []
    
    for step in graph.stream(input_data):
        for node_name, node_output in step.items():
            execution_log.append({
                "timestamp": datetime.now(),
                "node": node_name,
                "output_preview": str(node_output)[:100]
            })
            print(f"[{node_name}] {node_output}")
    
    return execution_log

# 使用
log = visualize_agent_execution(multi_agent_graph, {"messages": [...]})
```

---

## 本章总结

本章深入探讨了多 Agent 系统的设计与实现：

**核心概念**：
- 多 Agent 系统适用于复杂、多领域、需要专业分工的任务
- 三大架构模式：Supervisor（中心化）、Hierarchical（层级）、Collaborative（协作）
- Agent 间通信：消息传递、共享状态、事件驱动

**实战技能**：
- 使用 LangGraph 构建 Supervisor 模式的多 Agent 系统
- 实现层级管理和任务委派
- 设计平等协作和投票机制
- 完整案例：研究助手、客服系统、代码生成

**最佳实践**：
- 合理选择架构模式
- 减少通信开销，使用批处理
- 添加循环检测和超时控制
- 完善监控和日志

**下一步**：
Chapter 21 将学习更高级的 Agent 模式：Planning（计划与执行）和 Self-Critique（自我反思），进一步提升 Agent 的智能性和可靠性。

---

## 练习题

1. **架构设计**：为一个"智能招聘系统"设计多 Agent 架构（简历筛选 + 技能评估 + 面试安排 + 决策）

2. **性能优化**：在 Supervisor 模式中，如何实现 Worker Agent 的并行执行？

3. **容错处理**：如果某个 Worker Agent 失败，Supervisor 应该如何处理？

4. **扩展挑战**：实现一个支持动态添加/删除 Agent 的系统

---

## 扩展阅读

- [LangGraph Multi-Agent Documentation](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/)
- [AutoGen: Multi-Agent Conversation Framework](https://github.com/microsoft/autogen)
- [CrewAI: Framework for orchestrating role-playing AI agents](https://github.com/joaomdmoura/crewAI)
- [Paper: Communicative Agents for Software Development](https://arxiv.org/abs/2307.07924)

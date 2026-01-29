# Chapter 28: 高级 Agent 模式与人机协作

在前面的章节中，我们已经学习了 Agent 的基础架构、ReAct 模式、多 Agent 系统以及规划与反思机制。然而，在实际的生产环境中，Agent 系统往往需要处理更加复杂的场景：如何让 Agent 在关键决策点暂停并请求人工审批？如何为 Agent 构建长期记忆以支持跨会话的上下文延续？如何优雅地编排多个工具的调用顺序与依赖关系？本章将深入探讨这些高级 Agent 模式，帮助您构建更加智能、可控、可靠的企业级 Agent 系统。

> **本章核心内容**：
> - 人机协作（Human-in-the-Loop）：中断机制、审批流程、反馈注入
> - 长期记忆系统：跨会话记忆、知识图谱、向量记忆
> - 工具编排与依赖管理：工具链、条件工具、动态工具加载
> - Agent 调试与可观测性：中间状态检查、决策路径追踪
> - 高级错误恢复：自修复、降级策略、人工接管

## 28.1 人机协作（Human-in-the-Loop）基础

### 28.1.1 为什么需要人机协作？

在许多企业场景中，完全自动化的 Agent 可能并不合适：

1. **高风险决策**：财务审批、数据删除、合同签署等操作需要人工确认
2. **不确定性处理**：当 Agent 对结果不确定时，需要人工指导
3. **合规性要求**：某些行业（金融、医疗）要求人工审核关键步骤
4. **质量控制**：在 Agent 输出最终结果前，需要人工评审
5. **持续学习**：通过人工反馈改进 Agent 行为

### 28.1.2 LangGraph 的中断机制

LangGraph 提供了内置的中断（interrupt）功能，允许 Agent 在执行过程中暂停并等待人工输入。

**基本中断示例**：

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

# 定义状态
class AgentState(TypedDict):
    messages: Annotated[list, "消息历史"]
    approval_needed: bool
    approved: bool | None

# 定义节点
def analyze_request(state: AgentState):
    """分析用户请求"""
    last_message = state["messages"][-1].content
    
    # 检测是否需要审批（例如：删除操作）
    needs_approval = "delete" in last_message.lower() or "remove" in last_message.lower()
    
    return {
        "messages": state["messages"] + [AIMessage(content="已分析请求，检测到高风险操作")],
        "approval_needed": needs_approval,
        "approved": None
    }

def request_approval(state: AgentState):
    """请求人工审批（此节点会触发中断）"""
    return {
        "messages": state["messages"] + [AIMessage(content="⚠️ 请审批此操作：是否允许执行？")],
    }

def execute_action(state: AgentState):
    """执行实际操作"""
    if state.get("approved"):
        result = "✅ 操作已执行"
    else:
        result = "❌ 操作已拒绝"
    
    return {
        "messages": state["messages"] + [AIMessage(content=result)]
    }

# 构建图
workflow = StateGraph(AgentState)
workflow.add_node("analyze", analyze_request)
workflow.add_node("request_approval", request_approval)
workflow.add_node("execute", execute_action)

# 添加边
workflow.add_edge(START, "analyze")
workflow.add_conditional_edges(
    "analyze",
    lambda x: "request_approval" if x["approval_needed"] else "execute"
)
workflow.add_edge("request_approval", "execute")
workflow.add_edge("execute", END)

# 编译时添加中断点
memory = MemorySaver()
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["execute"]  # 在执行前中断
)
```

**使用中断的 Agent**：

```python
# 配置
config = {"configurable": {"thread_id": "approval-demo-1"}}

# 第一次调用：触发中断
initial_input = {
    "messages": [HumanMessage(content="请删除用户 ID 12345 的所有数据")],
    "approval_needed": False,
    "approved": None
}

# 运行到中断点
result = app.invoke(initial_input, config)
print("中断前的状态：")
print(result["messages"][-1].content)
# 输出：⚠️ 请审批此操作：是否允许执行？

# 检查当前状态
state = app.get_state(config)
print(f"下一个节点：{state.next}")  # ('execute',)

# 人工审批：更新状态并继续
app.update_state(config, {"approved": True})

# 继续执行
final_result = app.invoke(None, config)
print("最终结果：")
print(final_result["messages"][-1].content)
# 输出：✅ 操作已执行
```

### 28.1.3 多级审批流程

在复杂的企业场景中，可能需要多级审批：

```python
from enum import Enum

class ApprovalLevel(str, Enum):
    NONE = "none"
    MANAGER = "manager"
    DIRECTOR = "director"
    CEO = "ceo"

class MultiLevelState(TypedDict):
    messages: list
    request_type: str
    amount: float | None
    current_approval_level: ApprovalLevel
    approvals: dict[ApprovalLevel, bool]

def determine_approval_level(state: MultiLevelState):
    """根据金额确定审批级别"""
    amount = state.get("amount", 0)
    
    if amount < 1000:
        level = ApprovalLevel.NONE
    elif amount < 10000:
        level = ApprovalLevel.MANAGER
    elif amount < 100000:
        level = ApprovalLevel.DIRECTOR
    else:
        level = ApprovalLevel.CEO
    
    return {"current_approval_level": level}

def manager_approval(state: MultiLevelState):
    """经理审批"""
    return {
        "messages": state["messages"] + [
            AIMessage(content=f"等待经理审批金额 ${state['amount']}")
        ]
    }

def director_approval(state: MultiLevelState):
    """总监审批"""
    return {
        "messages": state["messages"] + [
            AIMessage(content=f"等待总监审批金额 ${state['amount']}")
        ]
    }

def ceo_approval(state: MultiLevelState):
    """CEO 审批"""
    return {
        "messages": state["messages"] + [
            AIMessage(content=f"等待 CEO 审批金额 ${state['amount']}")
        ]
    }

# 构建多级审批流程
workflow = StateGraph(MultiLevelState)
workflow.add_node("determine_level", determine_approval_level)
workflow.add_node("manager_approval", manager_approval)
workflow.add_node("director_approval", director_approval)
workflow.add_node("ceo_approval", ceo_approval)
workflow.add_node("execute", execute_action)

workflow.add_edge(START, "determine_level")
workflow.add_conditional_edges(
    "determine_level",
    lambda x: {
        ApprovalLevel.NONE: "execute",
        ApprovalLevel.MANAGER: "manager_approval",
        ApprovalLevel.DIRECTOR: "director_approval",
        ApprovalLevel.CEO: "ceo_approval"
    }[x["current_approval_level"]]
)

# 所有审批节点都连接到执行节点
workflow.add_edge("manager_approval", "execute")
workflow.add_edge("director_approval", "execute")
workflow.add_edge("ceo_approval", "execute")
workflow.add_edge("execute", END)

# 编译时在所有审批节点前中断
app = workflow.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["manager_approval", "director_approval", "ceo_approval"]
)
```

<div data-component="HumanInLoopFlow"></div>

### 28.1.4 审批超时与自动降级

在实际应用中，人工审批可能因为各种原因延迟，需要设置超时机制：

```python
import time
from datetime import datetime, timedelta

class TimeoutState(TypedDict):
    messages: list
    approval_requested_at: str | None
    approval_timeout_seconds: int
    approved: bool | None

def check_timeout(state: TimeoutState):
    """检查审批是否超时"""
    if not state.get("approval_requested_at"):
        return {"approved": False}
    
    requested_at = datetime.fromisoformat(state["approval_requested_at"])
    timeout = state.get("approval_timeout_seconds", 300)  # 默认 5 分钟
    
    if datetime.now() - requested_at > timedelta(seconds=timeout):
        # 超时：自动拒绝或降级处理
        return {
            "approved": False,
            "messages": state["messages"] + [
                AIMessage(content="⏱️ 审批超时，操作已自动拒绝")
            ]
        }
    
    return {}

# 在实际应用中的使用
def request_approval_with_timeout(state: TimeoutState):
    return {
        "messages": state["messages"] + [AIMessage(content="等待审批...")],
        "approval_requested_at": datetime.now().isoformat()
    }
```

### 28.1.5 人工反馈注入

除了简单的批准/拒绝，人工还可以提供详细的反馈和指导：

```python
class FeedbackState(TypedDict):
    messages: list
    agent_proposal: str | None
    human_feedback: str | None
    revision_count: int

def generate_proposal(state: FeedbackState):
    """Agent 生成提议"""
    # 实际应用中调用 LLM
    proposal = "我建议采用方案 A，因为它成本最低"
    
    return {
        "agent_proposal": proposal,
        "messages": state["messages"] + [AIMessage(content=f"📝 提议：{proposal}")]
    }

def incorporate_feedback(state: FeedbackState):
    """根据人工反馈修订提议"""
    feedback = state.get("human_feedback")
    
    if not feedback:
        # 无反馈，使用原提议
        return {"messages": state["messages"] + [AIMessage(content="采用原方案")]}
    
    # 实际应用中调用 LLM 整合反馈
    revised = f"根据反馈'{feedback}'，我修订提议为：方案 B"
    
    return {
        "messages": state["messages"] + [AIMessage(content=f"📝 修订后：{revised}")],
        "revision_count": state.get("revision_count", 0) + 1
    }

# 构建反馈循环
workflow = StateGraph(FeedbackState)
workflow.add_node("generate", generate_proposal)
workflow.add_node("incorporate", incorporate_feedback)

workflow.add_edge(START, "generate")
workflow.add_edge("generate", "incorporate")

# 条件边：如果有反馈且未超过最大修订次数，继续修订
workflow.add_conditional_edges(
    "incorporate",
    lambda x: "generate" if x.get("human_feedback") and x.get("revision_count", 0) < 3 else END
)

app = workflow.compile(
    checkpointer=MemorySaver(),
    interrupt_after=["generate"]  # 生成后中断，等待反馈
)
```

**使用示例**：

```python
config = {"configurable": {"thread_id": "feedback-loop-1"}}

# 初始运行
initial_state = {
    "messages": [HumanMessage(content="请为新项目推荐技术方案")],
    "agent_proposal": None,
    "human_feedback": None,
    "revision_count": 0
}

result = app.invoke(initial_state, config)
print(result["agent_proposal"])
# 输出：我建议采用方案 A，因为它成本最低

# 提供人工反馈
app.update_state(config, {
    "human_feedback": "成本不是主要考虑因素，性能更重要"
})

# 继续执行
result = app.invoke(None, config)
print(result["messages"][-1].content)
# 输出：📝 修订后：根据反馈'成本不是主要考虑因素，性能更重要'，我修订提议为：方案 B
```

## 28.2 长期记忆系统

### 28.2.1 为什么 Agent 需要长期记忆？

传统的对话记忆（如 ConversationBufferMemory）只能维护单次会话的上下文，但在许多场景中，Agent 需要跨会话的长期记忆：

1. **个性化服务**：记住用户的偏好、历史行为
2. **知识积累**：从多次交互中学习新知识
3. **关系维护**：记住与用户的关系历史
4. **任务延续**：跨会话追踪长期任务的进度

### 28.2.2 长期记忆的架构设计

一个完整的长期记忆系统通常包含三个层次：

```
┌─────────────────────────────────────────┐
│         短期记忆（Working Memory）        │  ← 当前会话的上下文
│  ConversationBuffer / ConversationSummary│
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│      中期记忆（Episodic Memory）         │  ← 最近几次会话的摘要
│   向量存储 + 时间索引 + 重要性评分        │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│      长期记忆（Semantic Memory）         │  ← 结构化知识库
│    知识图谱 + 实体关系 + 持久化事实       │
└─────────────────────────────────────────┘
```

### 28.2.3 基于向量存储的中期记忆

使用向量存储保存会话摘要，并通过相似度检索相关历史：

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from datetime import datetime
import uuid

class VectorMemorySystem:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            collection_name="agent_memory",
            embedding_function=self.embeddings,
            persist_directory="./agent_memory_db"
        )
        self.llm = ChatOpenAI(temperature=0)
    
    def save_conversation(self, user_id: str, conversation: list, importance: float = 0.5):
        """保存对话摘要到向量存储"""
        # 使用 LLM 生成对话摘要
        summary_prompt = f"请用一句话总结以下对话：\n{conversation}"
        summary = self.llm.invoke(summary_prompt).content
        
        # 构建文档
        doc = Document(
            page_content=summary,
            metadata={
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "importance": importance,
                "conversation_id": str(uuid.uuid4()),
                "full_conversation": str(conversation)
            }
        )
        
        # 保存到向量存储
        self.vectorstore.add_documents([doc])
        print(f"✅ 已保存对话摘要：{summary}")
    
    def recall_relevant_memories(self, user_id: str, current_query: str, k: int = 3):
        """检索相关的历史记忆"""
        # 向量检索
        results = self.vectorstore.similarity_search(
            current_query,
            k=k,
            filter={"user_id": user_id}
        )
        
        # 按重要性和时间排序
        sorted_results = sorted(
            results,
            key=lambda x: (
                x.metadata.get("importance", 0.5),
                x.metadata.get("timestamp", "")
            ),
            reverse=True
        )
        
        return [
            {
                "summary": doc.page_content,
                "timestamp": doc.metadata["timestamp"],
                "importance": doc.metadata["importance"]
            }
            for doc in sorted_results
        ]
    
    def forget_old_memories(self, user_id: str, days_threshold: int = 30):
        """清理旧记忆（根据时间和重要性）"""
        # 实际应用中需要实现基于时间的过滤和删除
        pass

# 使用示例
memory_system = VectorMemorySystem()

# 保存对话
memory_system.save_conversation(
    user_id="user123",
    conversation=[
        "用户：我喜欢 Python 编程",
        "助手：太好了！Python 是一门很棒的语言"
    ],
    importance=0.7
)

# 检索相关记忆
memories = memory_system.recall_relevant_memories(
    user_id="user123",
    current_query="你知道我喜欢什么编程语言吗？",
    k=3
)
print("相关记忆：", memories)
```

### 28.2.4 基于知识图谱的长期记忆

对于结构化的知识，使用知识图谱更加高效：

```python
from typing import List, Tuple
import networkx as nx
import json

class KnowledgeGraphMemory:
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def add_entity(self, entity: str, entity_type: str, attributes: dict = None):
        """添加实体"""
        self.graph.add_node(
            entity,
            type=entity_type,
            attributes=attributes or {}
        )
    
    def add_relation(self, subject: str, predicate: str, object: str, metadata: dict = None):
        """添加关系三元组"""
        self.graph.add_edge(
            subject,
            object,
            relation=predicate,
            metadata=metadata or {}
        )
    
    def query_relations(self, entity: str, relation_type: str = None) -> List[Tuple]:
        """查询实体的关系"""
        results = []
        
        # 查询出边（实体作为主语）
        for target in self.graph.successors(entity):
            edge_data = self.graph[entity][target]
            if relation_type is None or edge_data.get("relation") == relation_type:
                results.append((entity, edge_data["relation"], target))
        
        # 查询入边（实体作为宾语）
        for source in self.graph.predecessors(entity):
            edge_data = self.graph[source][entity]
            if relation_type is None or edge_data.get("relation") == relation_type:
                results.append((source, edge_data["relation"], entity))
        
        return results
    
    def get_entity_context(self, entity: str, depth: int = 2) -> dict:
        """获取实体的上下文（周边关系）"""
        # 获取 N 跳内的所有邻居
        subgraph_nodes = set([entity])
        current_level = {entity}
        
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                next_level.update(self.graph.successors(node))
                next_level.update(self.graph.predecessors(node))
            subgraph_nodes.update(next_level)
            current_level = next_level
        
        subgraph = self.graph.subgraph(subgraph_nodes)
        
        return {
            "entity": entity,
            "relations": [
                (u, subgraph[u][v]["relation"], v)
                for u, v in subgraph.edges()
            ]
        }
    
    def save(self, filepath: str):
        """持久化知识图谱"""
        data = nx.node_link_data(self.graph)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str):
        """加载知识图谱"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data)

# 使用示例
kg_memory = KnowledgeGraphMemory()

# 构建用户知识图谱
kg_memory.add_entity("Alice", "Person", {"age": 28, "occupation": "Engineer"})
kg_memory.add_entity("Python", "ProgrammingLanguage", {"paradigm": "multi-paradigm"})
kg_memory.add_entity("ProjectX", "Project", {"status": "active"})

kg_memory.add_relation("Alice", "likes", "Python")
kg_memory.add_relation("Alice", "works_on", "ProjectX")
kg_memory.add_relation("ProjectX", "uses", "Python")

# 查询
relations = kg_memory.query_relations("Alice")
print("Alice 的关系：", relations)
# 输出：[('Alice', 'likes', 'Python'), ('Alice', 'works_on', 'ProjectX')]

context = kg_memory.get_entity_context("Alice", depth=2)
print("Alice 的上下文：", context)
```

<div data-component="LongTermMemoryArchitecture"></div>

### 28.2.5 混合记忆系统：整合短期、中期和长期记忆

```python
from typing import TypedDict
from langchain_core.messages import BaseMessage

class HybridMemoryState(TypedDict):
    messages: list[BaseMessage]  # 短期记忆：当前会话
    recent_memories: list[dict]  # 中期记忆：相关历史会话
    knowledge_graph: dict        # 长期记忆：结构化知识

class HybridMemoryAgent:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.vector_memory = VectorMemorySystem()
        self.kg_memory = KnowledgeGraphMemory()
        self.llm = ChatOpenAI(temperature=0.7)
    
    def process_query(self, state: HybridMemoryState, query: str):
        """处理查询，整合三层记忆"""
        # 1. 从向量存储检索相关历史
        recent_memories = self.vector_memory.recall_relevant_memories(
            self.user_id,
            query,
            k=3
        )
        
        # 2. 从知识图谱提取相关实体
        # （实际应用中需要 NER 提取查询中的实体）
        entities = ["Alice"]  # 示例
        kg_context = self.kg_memory.get_entity_context(entities[0], depth=2)
        
        # 3. 构建增强的提示
        context_prompt = f"""
相关历史记忆：
{chr(10).join([f"- {m['summary']} ({m['timestamp']})" for m in recent_memories])}

知识图谱上下文：
{chr(10).join([f"- {s} {r} {o}" for s, r, o in kg_context['relations']])}

当前对话历史：
{chr(10).join([f"{m.type}: {m.content}" for m in state['messages'][-5:]])}

用户问题：{query}

请根据以上所有上下文回答用户问题。
"""
        
        response = self.llm.invoke(context_prompt).content
        return response
    
    def save_conversation(self, conversation: list, importance: float = 0.5):
        """保存对话到中期记忆"""
        self.vector_memory.save_conversation(
            self.user_id,
            conversation,
            importance
        )
    
    def update_knowledge(self, subject: str, relation: str, object: str):
        """更新长期知识"""
        self.kg_memory.add_relation(subject, relation, object)

# 使用示例
agent = HybridMemoryAgent(user_id="user123")

state = {
    "messages": [
        HumanMessage(content="我最近在学 Rust"),
        AIMessage(content="Rust 是一门很棒的系统编程语言")
    ],
    "recent_memories": [],
    "knowledge_graph": {}
}

response = agent.process_query(state, "我之前提到过喜欢什么语言？")
print(response)

# 保存当前对话
agent.save_conversation(
    [m.content for m in state["messages"]],
    importance=0.8
)

# 更新知识图谱
agent.update_knowledge("user123", "learning", "Rust")
```

## 28.3 工具编排与依赖管理

### 28.3.1 工具链：顺序工具调用

在某些场景中，工具需要按特定顺序调用，后续工具依赖前面工具的输出：

```python
from langchain.tools import tool
from typing import List, Dict

@tool
def fetch_user_info(user_id: str) -> dict:
    """获取用户信息"""
    # 模拟数据库查询
    return {
        "user_id": user_id,
        "name": "Alice",
        "email": "alice@example.com",
        "preferences": {"language": "zh-CN"}
    }

@tool
def get_user_orders(user_id: str) -> list:
    """获取用户订单"""
    # 模拟订单查询
    return [
        {"order_id": "ORD001", "amount": 299.99, "status": "shipped"},
        {"order_id": "ORD002", "amount": 149.99, "status": "processing"}
    ]

@tool
def calculate_total_spent(orders: list) -> float:
    """计算用户总消费"""
    return sum(order["amount"] for order in orders)

@tool
def send_personalized_email(user_info: dict, total_spent: float) -> str:
    """发送个性化邮件"""
    email_body = f"""
    尊敬的 {user_info['name']}，
    
    您在我们平台的累计消费为 ${total_spent:.2f}。
    感谢您的支持！
    """
    return f"邮件已发送到 {user_info['email']}"

# 工具链编排
class ToolChain:
    def __init__(self, tools: List):
        self.tools = {tool.name: tool for tool in tools}
    
    def execute_chain(self, steps: List[Dict], initial_input: Dict):
        """执行工具链"""
        context = initial_input.copy()
        results = {}
        
        for step in steps:
            tool_name = step["tool"]
            input_mapping = step["input"]
            output_key = step["output"]
            
            # 构建工具输入（从上下文中提取）
            tool_input = {}
            for arg_name, source_key in input_mapping.items():
                if source_key in context:
                    tool_input[arg_name] = context[source_key]
                elif source_key in results:
                    tool_input[arg_name] = results[source_key]
                else:
                    raise ValueError(f"Missing input: {source_key}")
            
            # 调用工具
            tool = self.tools[tool_name]
            result = tool.invoke(tool_input)
            
            # 保存结果
            results[output_key] = result
            context[output_key] = result
            
            print(f"✅ {tool_name}: {result}")
        
        return results

# 使用示例
tools = [fetch_user_info, get_user_orders, calculate_total_spent, send_personalized_email]
chain = ToolChain(tools)

# 定义工具链步骤
steps = [
    {
        "tool": "fetch_user_info",
        "input": {"user_id": "user_id"},
        "output": "user_info"
    },
    {
        "tool": "get_user_orders",
        "input": {"user_id": "user_id"},
        "output": "orders"
    },
    {
        "tool": "calculate_total_spent",
        "input": {"orders": "orders"},
        "output": "total_spent"
    },
    {
        "tool": "send_personalized_email",
        "input": {"user_info": "user_info", "total_spent": "total_spent"},
        "output": "email_result"
    }
]

# 执行
results = chain.execute_chain(steps, initial_input={"user_id": "12345"})
print("\n最终结果：", results["email_result"])
```

### 28.3.2 条件工具调用

根据中间结果决定是否调用某个工具：

```python
def execute_conditional_chain(self, steps: List[Dict], initial_input: Dict):
    """执行带条件的工具链"""
    context = initial_input.copy()
    results = {}
    
    for step in steps:
        # 检查条件
        if "condition" in step:
            condition_func = step["condition"]
            if not condition_func(context, results):
                print(f"⏭️  跳过 {step['tool']}（条件不满足）")
                continue
        
        # 执行工具（同上）
        tool_name = step["tool"]
        tool = self.tools[tool_name]
        
        # ... 执行逻辑 ...
    
    return results

# 使用示例
conditional_steps = [
    {
        "tool": "fetch_user_info",
        "input": {"user_id": "user_id"},
        "output": "user_info"
    },
    {
        "tool": "get_user_orders",
        "input": {"user_id": "user_id"},
        "output": "orders"
    },
    {
        "tool": "send_personalized_email",
        "input": {"user_info": "user_info", "total_spent": "total_spent"},
        "output": "email_result",
        "condition": lambda ctx, res: len(res.get("orders", [])) > 0  # 仅当有订单时发送
    }
]
```

### 28.3.3 动态工具加载

根据上下文动态决定加载哪些工具：

```python
from typing import Callable

class DynamicToolLoader:
    def __init__(self):
        self.tool_registry = {}
    
    def register_tool(self, tool_name: str, tool_func: Callable, requires: List[str] = None):
        """注册工具及其依赖"""
        self.tool_registry[tool_name] = {
            "func": tool_func,
            "requires": requires or []
        }
    
    def get_available_tools(self, context: dict) -> List[str]:
        """根据当前上下文返回可用的工具"""
        available = []
        
        for tool_name, tool_info in self.tool_registry.items():
            # 检查所有依赖是否满足
            if all(req in context for req in tool_info["requires"]):
                available.append(tool_name)
        
        return available
    
    def auto_plan_execution(self, goal: str, context: dict):
        """自动规划工具执行顺序"""
        # 简化版本：使用拓扑排序
        available = self.get_available_tools(context)
        
        # 实际应用中应使用 LLM 进行智能规划
        print(f"目标：{goal}")
        print(f"可用工具：{available}")
        
        return available

# 使用示例
loader = DynamicToolLoader()

loader.register_tool("fetch_user", fetch_user_info, requires=[])
loader.register_tool("get_orders", get_user_orders, requires=["user_id"])
loader.register_tool("calculate_total", calculate_total_spent, requires=["orders"])

# 场景 1：刚开始，只有 user_id
context1 = {"user_id": "12345"}
print("阶段 1 可用工具：", loader.get_available_tools(context1))
# 输出：['fetch_user', 'get_orders']

# 场景 2：已获取订单
context2 = {"user_id": "12345", "orders": [{"amount": 100}]}
print("阶段 2 可用工具：", loader.get_available_tools(context2))
# 输出：['fetch_user', 'get_orders', 'calculate_total']
```

<div data-component="ToolOrchestrationVisualizer"></div>

### 28.3.4 工具执行的并发控制

对于独立的工具调用，可以并发执行以提高效率：

```python
import asyncio
from typing import List, Dict

class AsyncToolChain:
    def __init__(self, tools: List):
        self.tools = {tool.name: tool for tool in tools}
    
    async def execute_parallel_tools(self, tool_calls: List[Dict]) -> Dict:
        """并发执行多个独立的工具"""
        tasks = []
        
        for call in tool_calls:
            tool_name = call["tool"]
            tool_input = call["input"]
            
            # 创建异步任务
            tool = self.tools[tool_name]
            task = asyncio.create_task(
                asyncio.to_thread(tool.invoke, tool_input)
            )
            tasks.append((call["output"], task))
        
        # 等待所有任务完成
        results = {}
        for output_key, task in tasks:
            results[output_key] = await task
        
        return results

# 使用示例
async def main():
    chain = AsyncToolChain([fetch_user_info, get_user_orders])
    
    # 并发调用多个工具
    parallel_calls = [
        {"tool": "fetch_user_info", "input": {"user_id": "12345"}, "output": "user"},
        {"tool": "get_user_orders", "input": {"user_id": "12345"}, "output": "orders"}
    ]
    
    results = await chain.execute_parallel_tools(parallel_calls)
    print("并发结果：", results)

# 运行
# asyncio.run(main())
```

## 28.4 Agent 调试与可观测性

### 28.4.1 中间状态检查

LangGraph 允许在任意节点后检查状态：

```python
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

# 构建图（使用前面的示例）
workflow = StateGraph(AgentState)
# ... 添加节点和边 ...

# 编译时启用检查点
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# 执行
config = {"configurable": {"thread_id": "debug-1"}}
result = app.invoke(initial_input, config)

# 检查执行历史
state_history = app.get_state_history(config)
for i, state in enumerate(state_history):
    print(f"步骤 {i}:")
    print(f"  节点: {state.next}")
    print(f"  状态: {state.values}")
    print()
```

### 28.4.2 决策路径追踪

记录 Agent 的每个决策点：

```python
class TrackedAgentState(TypedDict):
    messages: list
    decision_log: list[dict]

def tracked_decision_node(state: TrackedAgentState):
    """记录决策的节点"""
    # 做出决策
    decision = "执行操作 A"
    
    # 记录决策
    decision_entry = {
        "timestamp": datetime.now().isoformat(),
        "decision": decision,
        "reasoning": "因为 X 条件满足",
        "confidence": 0.85
    }
    
    return {
        "decision_log": state.get("decision_log", []) + [decision_entry]
    }

# 执行后分析决策路径
def analyze_decision_path(state: TrackedAgentState):
    """分析决策路径"""
    for i, entry in enumerate(state["decision_log"]):
        print(f"决策 {i+1}: {entry['decision']}")
        print(f"  原因: {entry['reasoning']}")
        print(f"  置信度: {entry['confidence']}")
        print(f"  时间: {entry['timestamp']}")
        print()
```

### 28.4.3 性能分析

```python
import time
from functools import wraps

def measure_performance(func):
    """装饰器：测量节点执行时间"""
    @wraps(func)
    def wrapper(state):
        start_time = time.time()
        result = func(state)
        elapsed = time.time() - start_time
        
        # 将性能数据添加到状态
        perf_data = state.get("performance_log", [])
        perf_data.append({
            "node": func.__name__,
            "duration_seconds": elapsed,
            "timestamp": datetime.now().isoformat()
        })
        
        result["performance_log"] = perf_data
        return result
    
    return wrapper

# 使用示例
@measure_performance
def slow_analysis_node(state):
    """模拟耗时节点"""
    time.sleep(2)
    return {"messages": state["messages"] + [AIMessage(content="分析完成")]}

# 执行后查看性能报告
def print_performance_report(state):
    """打印性能报告"""
    total_time = sum(log["duration_seconds"] for log in state["performance_log"])
    
    print("=== 性能报告 ===")
    for log in state["performance_log"]:
        percentage = (log["duration_seconds"] / total_time) * 100
        print(f"{log['node']}: {log['duration_seconds']:.2f}s ({percentage:.1f}%)")
    print(f"总计: {total_time:.2f}s")
```

## 28.5 高级错误恢复

### 28.5.1 自修复机制

当工具调用失败时，Agent 可以尝试自动修复：

```python
class SelfHealingState(TypedDict):
    messages: list
    tool_call: dict | None
    error: str | None
    retry_count: int

def execute_tool_with_healing(state: SelfHealingState):
    """执行工具，失败时尝试自修复"""
    tool_call = state["tool_call"]
    retry_count = state.get("retry_count", 0)
    
    try:
        # 尝试执行工具
        result = execute_tool(tool_call)
        return {
            "messages": state["messages"] + [AIMessage(content=f"✅ {result}")],
            "error": None
        }
    except Exception as e:
        error_msg = str(e)
        
        # 尝试自修复
        if retry_count < 3:
            # 使用 LLM 分析错误并生成修复方案
            heal_prompt = f"""
            工具调用失败：
            工具: {tool_call['name']}
            参数: {tool_call['args']}
            错误: {error_msg}
            
            请分析错误原因并提供修复后的参数。
            只返回 JSON 格式的修复参数。
            """
            
            llm = ChatOpenAI(temperature=0)
            fixed_args = llm.invoke(heal_prompt).content
            
            return {
                "tool_call": {"name": tool_call["name"], "args": fixed_args},
                "retry_count": retry_count + 1,
                "error": error_msg,
                "messages": state["messages"] + [
                    AIMessage(content=f"⚠️ 错误：{error_msg}，正在尝试修复...")
                ]
            }
        else:
            # 放弃修复，转人工处理
            return {
                "error": error_msg,
                "messages": state["messages"] + [
                    AIMessage(content=f"❌ 自动修复失败，需要人工介入")
                ]
            }
```

### 28.5.2 降级策略

当高级功能不可用时，回退到简单方法：

```python
class FallbackState(TypedDict):
    messages: list
    strategy: str  # "advanced" | "standard" | "basic"

def advanced_strategy_node(state: FallbackState):
    """高级策略（可能失败）"""
    try:
        # 尝试高级 API
        result = call_advanced_api()
        return {"messages": state["messages"] + [AIMessage(content=result)]}
    except Exception as e:
        # 降级到标准策略
        return {"strategy": "standard", "messages": state["messages"]}

def standard_strategy_node(state: FallbackState):
    """标准策略"""
    try:
        result = call_standard_api()
        return {"messages": state["messages"] + [AIMessage(content=result)]}
    except Exception as e:
        # 再降级到基础策略
        return {"strategy": "basic", "messages": state["messages"]}

def basic_strategy_node(state: FallbackState):
    """基础策略（保证可用）"""
    result = simple_fallback_logic()
    return {"messages": state["messages"] + [AIMessage(content=result)]}

# 构建降级流程
workflow = StateGraph(FallbackState)
workflow.add_node("advanced", advanced_strategy_node)
workflow.add_node("standard", standard_strategy_node)
workflow.add_node("basic", basic_strategy_node)

workflow.add_edge(START, "advanced")
workflow.add_conditional_edges(
    "advanced",
    lambda x: "standard" if x.get("strategy") == "standard" else END
)
workflow.add_conditional_edges(
    "standard",
    lambda x: "basic" if x.get("strategy") == "basic" else END
)
workflow.add_edge("basic", END)
```

### 28.5.3 人工接管

当 Agent 遇到无法处理的情况，平滑过渡到人工客服：

```python
class HandoffState(TypedDict):
    messages: list
    confidence: float
    handoff_triggered: bool
    handoff_reason: str | None

def check_handoff_condition(state: HandoffState):
    """检查是否需要人工接管"""
    confidence = state.get("confidence", 1.0)
    
    # 低置信度 → 转人工
    if confidence < 0.3:
        return {
            "handoff_triggered": True,
            "handoff_reason": "置信度过低"
        }
    
    # 用户明确要求 → 转人工
    last_message = state["messages"][-1].content.lower()
    if "人工" in last_message or "客服" in last_message:
        return {
            "handoff_triggered": True,
            "handoff_reason": "用户请求人工服务"
        }
    
    return {"handoff_triggered": False}

def handoff_to_human(state: HandoffState):
    """转人工处理"""
    reason = state.get("handoff_reason", "未知原因")
    
    # 实际应用中：通知人工客服系统、发送工单等
    handoff_message = f"""
    已为您转接人工客服。
    原因：{reason}
    请稍等，客服人员马上为您服务。
    """
    
    return {
        "messages": state["messages"] + [AIMessage(content=handoff_message)]
    }

# 在 Agent 流程中添加检查点
workflow.add_conditional_edges(
    "agent_response",
    lambda x: "handoff" if x.get("handoff_triggered") else END
)
```

## 28.6 综合案例：企业级客服 Agent

让我们整合本章的所有概念，构建一个完整的企业级客服 Agent：

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

class CustomerServiceState(TypedDict):
    messages: list
    user_id: str
    session_id: str
    
    # 记忆系统
    recent_memories: list[dict]
    user_profile: dict
    
    # 工具执行
    pending_tools: list[dict]
    tool_results: dict
    
    # 人机协作
    needs_approval: bool
    approved: bool | None
    
    # 错误处理
    error_count: int
    confidence: float
    handoff_triggered: bool

def load_user_context(state: CustomerServiceState):
    """加载用户上下文（长期记忆）"""
    # 从向量存储检索历史对话
    memory_system = VectorMemorySystem()
    recent_memories = memory_system.recall_relevant_memories(
        state["user_id"],
        state["messages"][-1].content,
        k=3
    )
    
    # 从知识图谱加载用户画像
    kg = KnowledgeGraphMemory()
    kg.load(f"./users/{state['user_id']}_kg.json")
    user_profile = kg.get_entity_context(state["user_id"], depth=1)
    
    return {
        "recent_memories": recent_memories,
        "user_profile": user_profile
    }

def analyze_intent(state: CustomerServiceState):
    """分析用户意图"""
    llm = ChatOpenAI(temperature=0)
    
    # 构建上下文增强的提示
    context = f"""
用户历史：
{chr(10).join([f"- {m['summary']}" for m in state['recent_memories']])}

用户画像：
{state['user_profile']}

当前对话：
{chr(10).join([f"{m.type}: {m.content}" for m in state['messages'][-3:]])}

请分析用户的意图，并返回 JSON：
{{"intent": "查询订单|退款申请|技术支持|其他", "confidence": 0.0-1.0, "entities": []}}
"""
    
    result = llm.invoke(context).content
    # 解析 JSON（实际应用中需要更严格的解析）
    
    return {"confidence": 0.8}  # 示例

def plan_tool_execution(state: CustomerServiceState):
    """规划工具执行"""
    # 根据意图选择工具
    intent = "查询订单"  # 从上一步获取
    
    if intent == "查询订单":
        tools = [
            {"tool": "fetch_user_info", "input": {"user_id": state["user_id"]}},
            {"tool": "get_user_orders", "input": {"user_id": state["user_id"]}}
        ]
    elif intent == "退款申请":
        tools = [
            {"tool": "check_refund_eligibility", "input": {}},
            {"tool": "submit_refund_request", "input": {}}
        ]
        # 退款需要审批
        return {"pending_tools": tools, "needs_approval": True}
    else:
        tools = []
    
    return {"pending_tools": tools, "needs_approval": False}

def execute_tools(state: CustomerServiceState):
    """执行工具链"""
    results = {}
    
    for tool_spec in state["pending_tools"]:
        try:
            # 执行工具（简化示例）
            result = {"status": "success"}
            results[tool_spec["tool"]] = result
        except Exception as e:
            return {"error_count": state.get("error_count", 0) + 1}
    
    return {"tool_results": results}

def generate_response(state: CustomerServiceState):
    """生成响应"""
    llm = ChatOpenAI(temperature=0.7)
    
    prompt = f"""
用户问题：{state['messages'][-1].content}

工具执行结果：
{state['tool_results']}

请生成友好、专业的客服回复。
"""
    
    response = llm.invoke(prompt).content
    
    return {
        "messages": state["messages"] + [AIMessage(content=response)]
    }

def check_quality(state: CustomerServiceState):
    """质量检查：决定是否需要人工介入"""
    confidence = state.get("confidence", 1.0)
    error_count = state.get("error_count", 0)
    
    if confidence < 0.3 or error_count > 2:
        return {"handoff_triggered": True}
    
    return {}

# 构建完整的客服 Agent 图
workflow = StateGraph(CustomerServiceState)

# 添加节点
workflow.add_node("load_context", load_user_context)
workflow.add_node("analyze_intent", analyze_intent)
workflow.add_node("plan_tools", plan_tool_execution)
workflow.add_node("execute_tools", execute_tools)
workflow.add_node("generate_response", generate_response)
workflow.add_node("check_quality", check_quality)
workflow.add_node("handoff", handoff_to_human)

# 添加边
workflow.add_edge(START, "load_context")
workflow.add_edge("load_context", "analyze_intent")
workflow.add_edge("analyze_intent", "plan_tools")

# 条件边：是否需要审批
workflow.add_conditional_edges(
    "plan_tools",
    lambda x: "wait_approval" if x.get("needs_approval") else "execute_tools"
)

workflow.add_edge("execute_tools", "generate_response")
workflow.add_edge("generate_response", "check_quality")

# 条件边：是否转人工
workflow.add_conditional_edges(
    "check_quality",
    lambda x: "handoff" if x.get("handoff_triggered") else END
)

workflow.add_edge("handoff", END)

# 编译
app = workflow.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["execute_tools"]  # 审批点
)

# 使用示例
config = {"configurable": {"thread_id": "customer-123-session-456"}}

initial_state = {
    "messages": [HumanMessage(content="我要申请退款")],
    "user_id": "customer-123",
    "session_id": "session-456",
    "recent_memories": [],
    "user_profile": {},
    "pending_tools": [],
    "tool_results": {},
    "needs_approval": False,
    "approved": None,
    "error_count": 0,
    "confidence": 1.0,
    "handoff_triggered": False
}

# 执行
result = app.invoke(initial_state, config)
print("Agent 响应：", result["messages"][-1].content)
```

## 28.7 最佳实践与生产建议

### 28.7.1 人机协作的设计原则

1. **明确的权限边界**：清晰定义哪些操作必须人工审批
2. **低摩擦体验**：审批流程应简单快捷，避免过度打断
3. **超时处理**：设置合理的审批超时，避免 Agent 无限期等待
4. **审计日志**：记录所有审批决策，用于合规和分析

### 28.7.2 长期记忆的管理策略

1. **记忆衰减**：旧记忆应逐渐降低权重或删除
2. **隐私保护**：敏感信息应加密存储或定期清理
3. **记忆一致性**：向量记忆和知识图谱应保持同步
4. **成本控制**：限制记忆存储的总量，避免无限膨胀

### 28.7.3 工具编排的优化

1. **最小化工具调用**：避免不必要的 API 请求
2. **并发执行**：对于独立的工具调用，使用异步并发
3. **缓存结果**：对于频繁查询的数据，使用缓存
4. **错误传播**：合理处理工具链中的错误传播

### 28.7.4 可观测性的关键指标

1. **决策质量**：Agent 决策的准确率、置信度分布
2. **执行效率**：每个节点的耗时、工具调用延迟
3. **人工介入率**：需要人工审批或接管的比例
4. **用户满意度**：通过反馈收集用户评价

## 28.8 章节总结

本章深入探讨了高级 Agent 模式，重点涵盖：

1. **人机协作（HITL）**：
   - LangGraph 的中断机制与审批流程
   - 多级审批、超时处理、反馈注入
   - 人工接管的平滑过渡

2. **长期记忆系统**：
   - 三层记忆架构：短期、中期、长期
   - 基于向量存储的情节记忆
   - 基于知识图谱的语义记忆
   - 混合记忆系统的整合

3. **工具编排**：
   - 工具链的顺序执行与依赖管理
   - 条件工具调用与动态加载
   - 并发控制与性能优化

4. **调试与可观测性**：
   - 中间状态检查与决策路径追踪
   - 性能分析与瓶颈识别
   - LangSmith 集成（详见 Chapter 22-24）

5. **错误恢复**：
   - 自修复机制与重试策略
   - 降级方案与容错设计
   - 人工接管的触发条件

通过这些高级模式，您可以构建更加智能、可控、可靠的企业级 Agent 系统，适应复杂的生产环境需求。

下一章（Chapter 29）将探讨 LangChain 与其他框架的生态集成，以及如何平滑迁移现有项目。

---

**扩展阅读**：
- [LangGraph Human-in-the-Loop 官方文档](https://langchain-ai.github.io/langgraph/how-tos/human-in-the-loop/)
- [LangChain Memory 系统详解](https://python.langchain.com/docs/modules/memory/)
- [企业级 Agent 架构设计模式](https://www.langchain.com/blog/)

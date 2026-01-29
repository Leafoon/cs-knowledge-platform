# Chapter 29: LangChain 生态集成与迁移

在实际的企业应用中，技术栈的选择往往受到多种因素的影响：团队熟悉度、现有项目的技术债务、特定场景的性能需求等。LangChain 虽然是目前最流行的 LLM 应用框架之一，但并非所有场景的唯一选择。本章将深入探讨 LangChain 与其他主流框架（LlamaIndex、Haystack、AutoGen、CrewAI）的对比与集成，帮助您理解何时选择何种工具，以及如何在不同框架间平滑迁移。

> **本章核心内容**：
> - LangChain vs. 主流框架：定位、优势、适用场景
> - 与 LlamaIndex 的集成：RAG 系统的最佳拍档
> - 与 Haystack 的对比与互操作
> - 与 AutoGen / CrewAI 的多 Agent 系统集成
> - 迁移指南：从其他框架迁移到 LangChain
> - 反向迁移：从 LangChain 迁移到其他框架

## 29.1 LLM 应用框架生态全景

### 29.1.1 主流框架定位对比

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Application Frameworks                │
├─────────────────┬──────────────┬────────────────────────────┤
│    LangChain    │  LlamaIndex  │        Haystack            │
│  通用编排框架     │  RAG 专家     │   企业搜索 + NLP          │
│  ✓ 链式编排      │  ✓ 索引优化   │   ✓ Pipeline 架构        │
│  ✓ Agent 系统    │  ✓ 查询引擎   │   ✓ 生产部署             │
│  ✓ 记忆管理      │  ✓ 评估工具   │   ✓ REST API             │
├─────────────────┼──────────────┼────────────────────────────┤
│    AutoGen      │   CrewAI     │         Semantic Kernel    │
│  多 Agent 研究   │  角色化 Agent │   微软企业框架             │
│  ✓ Agent 对话    │  ✓ 团队协作   │   ✓ .NET / Python        │
│  ✓ Code Executor │  ✓ 任务分配   │   ✓ 企业集成             │
│  ✓ 自主对话      │  ✓ 流程编排   │   ✓ Azure 生态           │
└─────────────────┴──────────────┴────────────────────────────┘
```

### 29.1.2 框架选择决策树

```python
def choose_framework(requirements: dict) -> str:
    """根据需求推荐框架"""
    
    # RAG 为核心
    if requirements.get("primary_use_case") == "RAG":
        if requirements.get("need_advanced_indexing"):
            return "LlamaIndex（专业 RAG）"
        else:
            return "LangChain（通用 RAG + 其他能力）"
    
    # 多 Agent 系统
    elif requirements.get("primary_use_case") == "multi_agent":
        if requirements.get("research_focus"):
            return "AutoGen（研究型多 Agent）"
        elif requirements.get("role_based_workflow"):
            return "CrewAI（业务流程多 Agent）"
        else:
            return "LangChain + LangGraph（生产级多 Agent）"
    
    # 企业搜索
    elif requirements.get("primary_use_case") == "enterprise_search":
        return "Haystack（企业搜索特化）"
    
    # .NET 生态
    elif requirements.get("language") == ".NET":
        return "Semantic Kernel"
    
    # 通用场景
    else:
        return "LangChain（最成熟的通用框架）"

# 示例
requirements = {
    "primary_use_case": "RAG",
    "need_advanced_indexing": True,
    "scale": "large",
    "language": "Python"
}

recommendation = choose_framework(requirements)
print(f"推荐框架：{recommendation}")
# 输出：推荐框架：LlamaIndex（专业 RAG）
```

<div data-component="FrameworkComparisonMatrix"></div>

## 29.2 与 LlamaIndex 的深度集成

### 29.2.1 LlamaIndex 简介与优势

LlamaIndex（原 GPT Index）是专注于 RAG 系统的框架，其核心优势：

1. **高级索引结构**：Tree Index、Vector Index、List Index、Keyword Index 等多种索引
2. **智能查询引擎**：自动选择最优索引策略
3. **评估工具**：内置 RAG 评估指标（relevance、faithfulness、context recall 等）
4. **与 LangChain 的互操作性**：可无缝集成

**LlamaIndex 的典型代码**：

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 加载文档
documents = SimpleDirectoryReader("./data").load_data()

# 构建索引
index = VectorStoreIndex.from_documents(documents)

# 查询
query_engine = index.as_query_engine()
response = query_engine.query("What is LangChain?")
print(response)
```

### 29.2.2 在 LangChain 中使用 LlamaIndex

LangChain 提供了 `LlamaIndexRetriever` 和 `LlamaIndexChain`，可直接集成 LlamaIndex：

```python
from langchain.retrievers import LlamaIndexRetriever
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 1. 使用 LlamaIndex 构建索引
documents = SimpleDirectoryReader("./docs").load_data()
index = VectorStoreIndex.from_documents(documents)

# 2. 将 LlamaIndex 索引包装为 LangChain Retriever
retriever = LlamaIndexRetriever(index=index)

# 3. 在 LangChain 中使用
llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# 4. 查询
result = qa_chain.invoke({"query": "What are the key features of LangChain?"})
print(result["result"])
```

### 29.2.3 LlamaIndex 的高级索引在 LangChain 中的应用

**Tree Index（层次化索引）**：

```python
from llama_index.core import TreeIndex
from langchain.retrievers import LlamaIndexRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

# 构建 Tree Index（适合总结和层次化查询）
tree_index = TreeIndex.from_documents(documents)

# 包装为 LangChain Retriever
retriever = LlamaIndexRetriever(index=tree_index, num_chunks=3)

# LCEL 链
template = """基于以下上下文回答问题：

上下文：
{context}

问题：{question}

答案："""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

response = chain.invoke("总结所有文档的主要观点")
print(response.content)
```

**Hybrid Index（混合索引：向量 + 关键词）**：

```python
from llama_index.core import VectorStoreIndex, KeywordTableIndex
from llama_index.core.composability import ComposableGraph

# 构建向量索引
vector_index = VectorStoreIndex.from_documents(documents)

# 构建关键词索引
keyword_index = KeywordTableIndex.from_documents(documents)

# 组合为混合索引
graph = ComposableGraph.from_indices(
    indices=[vector_index, keyword_index],
    index_summaries=["向量索引：适合语义搜索", "关键词索引：适合精确匹配"]
)

# 在 LangChain 中使用
retriever = LlamaIndexRetriever(index=graph.get_index())

# ... 后续与 LangChain 集成
```

### 29.2.4 LlamaIndex 评估器与 LangSmith 集成

LlamaIndex 提供强大的 RAG 评估工具，可与 LangSmith 集成：

```python
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator
)
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from langsmith import Client as LangSmithClient

# 初始化评估器
llm = LlamaOpenAI(model="gpt-4")
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

# 评估查询结果
query = "What is LangChain?"
response = query_engine.query(query)

# 评估忠实度（是否基于上下文回答）
faithfulness_result = await faithfulness_evaluator.aevaluate(
    query=query,
    response=str(response),
    contexts=[node.text for node in response.source_nodes]
)

# 评估相关性（检索的上下文是否相关）
relevancy_result = await relevancy_evaluator.aevaluate(
    query=query,
    response=str(response),
    contexts=[node.text for node in response.source_nodes]
)

print(f"忠实度：{faithfulness_result.score}")
print(f"相关性：{relevancy_result.score}")

# 上传到 LangSmith
langsmith_client = LangSmithClient()
langsmith_client.create_feedback(
    run_id=run_id,  # 从 LangSmith trace 获取
    key="faithfulness",
    score=faithfulness_result.score,
    comment=faithfulness_result.feedback
)
```

### 29.2.5 混合 LangChain 和 LlamaIndex 的最佳实践

**推荐架构**：使用 LlamaIndex 处理复杂 RAG，LangChain 处理其他逻辑：

```python
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from llama_index.core import VectorStoreIndex

# 1. LlamaIndex 负责 RAG
documents = SimpleDirectoryReader("./knowledge_base").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# 2. 将 LlamaIndex 查询引擎包装为 LangChain Tool
def llamaindex_search(query: str) -> str:
    """搜索内部知识库"""
    response = query_engine.query(query)
    return str(response)

llamaindex_tool = Tool(
    name="knowledge_base_search",
    func=llamaindex_search,
    description="搜索内部知识库，获取产品文档、FAQ 等信息"
)

# 3. 创建 LangChain Agent，集成 LlamaIndex Tool
llm = ChatOpenAI(model="gpt-4")

tools = [
    llamaindex_tool,
    # ... 其他工具（计算器、天气 API 等）
]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手，可以访问内部知识库。"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 4. 使用
result = agent_executor.invoke({
    "input": "我们的退款政策是什么？如果用户在购买后 7 天内申请退款，应该如何处理？"
})

print(result["output"])
```

## 29.3 与 Haystack 的对比与互操作

### 29.3.1 Haystack 简介

Haystack 是由 deepset 开发的开源 NLP 框架，专注于企业级搜索和问答系统。

**Haystack 的核心概念**：
- **Pipeline**：类似 LangChain 的 Chain，但更模块化
- **DocumentStore**：文档存储（支持 Elasticsearch、Weaviate、Pinecone 等）
- **Retriever**：检索器（BM25、DPR、Embedding Retriever）
- **Reader**：阅读器（提取式 QA 模型）

**典型 Haystack Pipeline**：

```python
from haystack import Pipeline
from haystack.nodes import BM25Retriever, FARMReader
from haystack.document_stores import ElasticsearchDocumentStore

# 初始化文档存储
document_store = ElasticsearchDocumentStore(host="localhost")

# 初始化检索器
retriever = BM25Retriever(document_store=document_store)

# 初始化阅读器
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

# 构建 Pipeline
pipe = Pipeline()
pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
pipe.add_node(component=reader, name="Reader", inputs=["Retriever"])

# 查询
result = pipe.run(query="What is LangChain?", params={"Retriever": {"top_k": 5}})
```

### 29.3.2 Haystack vs. LangChain

| 维度 | Haystack | LangChain |
|------|----------|-----------|
| **定位** | 企业搜索 + QA | 通用 LLM 应用编排 |
| **核心抽象** | Pipeline（DAG） | Chain / Graph（灵活组合） |
| **检索器** | BM25、DPR、Embedding（传统 NLP 背景） | 向量存储 + Reranker（LLM 时代设计） |
| **阅读器** | 提取式 QA 模型（BERT、RoBERTa） | 生成式 LLM（GPT、Claude） |
| **部署** | REST API（内置 Haystack REST API） | LangServe（FastAPI 包装） |
| **评估** | 内置评估框架（Exact Match、F1、Recall@k） | LangSmith（追踪 + 评估） |
| **适用场景** | 传统搜索升级、大规模文档库 | LLM 原生应用、复杂 Agent 系统 |

### 29.3.3 在 LangChain 中使用 Haystack 组件

虽然直接集成较少，但可通过自定义 Retriever 桥接：

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever as HaystackBM25Retriever

class HaystackRetrieverWrapper(BaseRetriever):
    """将 Haystack Retriever 包装为 LangChain Retriever"""
    
    def __init__(self, haystack_retriever, document_store):
        self.haystack_retriever = haystack_retriever
        self.document_store = document_store
    
    def _get_relevant_documents(self, query: str) -> list[Document]:
        # 使用 Haystack 检索
        results = self.haystack_retriever.retrieve(query=query, top_k=5)
        
        # 转换为 LangChain Document
        docs = []
        for doc in results:
            docs.append(Document(
                page_content=doc.content,
                metadata={"score": doc.score, **doc.meta}
            ))
        
        return docs

# 使用示例
document_store = ElasticsearchDocumentStore(host="localhost")
haystack_retriever = HaystackBM25Retriever(document_store=document_store)

# 包装为 LangChain Retriever
langchain_retriever = HaystackRetrieverWrapper(
    haystack_retriever=haystack_retriever,
    document_store=document_store
)

# 在 LangChain 中使用
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=langchain_retriever
)

result = qa_chain.invoke({"query": "What is RAG?"})
print(result)
```

### 29.3.4 何时选择 Haystack 而非 LangChain

**选择 Haystack 的场景**：
1. **已有 Elasticsearch 基础设施**：Haystack 与 Elasticsearch 深度集成
2. **传统 NLP 模型优先**：希望使用 BERT/RoBERTa 提取式 QA 而非生成式 LLM
3. **大规模文档检索**：Haystack 的 BM25 + DPR 在亿级文档上性能更优
4. **预算有限**：避免频繁调用 GPT API，使用本地模型

**选择 LangChain 的场景**：
1. **复杂 Agent 系统**：需要 ReAct、Planning、Multi-Agent
2. **生成式 QA**：希望 LLM 生成答案而非提取
3. **快速原型**：LangChain 的高层抽象更适合快速迭代
4. **可观测性需求**：LangSmith 提供完整追踪

## 29.4 与 AutoGen 的多 Agent 集成

### 29.4.1 AutoGen 简介

AutoGen 是微软研究院开发的多 Agent 对话框架，核心特点：

1. **Agent 自主对话**：Agent 之间可自主交流，无需人工干预
2. **Code Execution**：内置安全的代码执行环境
3. **Human-in-the-Loop**：支持人工介入对话
4. **群聊模式**：多个 Agent 组成群聊，协同解决问题

**AutoGen 的典型代码**：

```python
import autogen

# 配置
config_list = [{"model": "gpt-4", "api_key": "your-api-key"}]

# 创建 AssistantAgent（助手）
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={"config_list": config_list}
)

# 创建 UserProxyAgent（用户代理 + 代码执行器）
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",  # 自动模式
    code_execution_config={"work_dir": "coding"}
)

# 发起对话
user_proxy.initiate_chat(
    assistant,
    message="请帮我用 Python 计算斐波那契数列的前 10 项"
)
```

### 29.4.2 AutoGen vs. LangChain Agent

| 维度 | AutoGen | LangChain Agent |
|------|---------|-----------------|
| **对话模式** | 多轮自主对话（Agent 之间） | 单轮工具调用（Agent ↔ Tools） |
| **Code Execution** | 内置（Docker 隔离） | 需自定义 Tool（PythonREPLTool） |
| **群聊** | 原生支持（GroupChat） | 需手动编排（LangGraph） |
| **控制粒度** | 低（Agent 自主决策） | 高（显式定义流程） |
| **适用场景** | 研究、探索性任务 | 生产级可控流程 |

### 29.4.3 在 LangChain 中模拟 AutoGen 的自主对话

使用 LangGraph 实现类似 AutoGen 的 Agent 对话：

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

class MultiAgentState(TypedDict):
    messages: Annotated[list, "对话历史"]
    next_speaker: str
    max_turns: int
    current_turn: int

def agent_1(state: MultiAgentState):
    """Agent 1: 研究员"""
    llm = ChatOpenAI(model="gpt-4")
    
    # 生成回复
    prompt = f"""你是研究员 Agent。查看以下对话历史，并提供你的见解。
如果问题已解决，回复 "TERMINATE"。

对话历史：
{chr(10).join([f"{m.type}: {m.content}" for m in state['messages'][-5:]])}
"""
    
    response = llm.invoke(prompt).content
    
    return {
        "messages": state["messages"] + [AIMessage(content=f"[研究员] {response}")],
        "next_speaker": "agent_2" if "TERMINATE" not in response else "end",
        "current_turn": state["current_turn"] + 1
    }

def agent_2(state: MultiAgentState):
    """Agent 2: 工程师"""
    llm = ChatOpenAI(model="gpt-4")
    
    prompt = f"""你是工程师 Agent。查看研究员的回复，提供技术实现方案。
如果无需进一步讨论，回复 "TERMINATE"。

对话历史：
{chr(10).join([f"{m.type}: {m.content}" for m in state['messages'][-5:]])}
"""
    
    response = llm.invoke(prompt).content
    
    return {
        "messages": state["messages"] + [AIMessage(content=f"[工程师] {response}")],
        "next_speaker": "agent_1" if "TERMINATE" not in response else "end",
        "current_turn": state["current_turn"] + 1
    }

# 构建图
workflow = StateGraph(MultiAgentState)
workflow.add_node("agent_1", agent_1)
workflow.add_node("agent_2", agent_2)

workflow.set_entry_point("agent_1")
workflow.add_conditional_edges(
    "agent_1",
    lambda x: x["next_speaker"] if x["current_turn"] < x["max_turns"] else "end"
)
workflow.add_conditional_edges(
    "agent_2",
    lambda x: x["next_speaker"] if x["current_turn"] < x["max_turns"] else "end"
)

app = workflow.compile()

# 使用
initial_state = {
    "messages": [HumanMessage(content="设计一个高性能的向量搜索系统")],
    "next_speaker": "agent_1",
    "max_turns": 10,
    "current_turn": 0
}

result = app.invoke(initial_state)
for msg in result["messages"]:
    print(f"{msg.content}\n")
```

## 29.5 与 CrewAI 的角色化 Agent 集成

### 29.5.1 CrewAI 简介

CrewAI 是一个专注于角色化 Agent 团队协作的框架，类似现实世界的项目团队。

**核心概念**：
- **Agent**：具有角色、目标、背景故事的智能体
- **Task**：分配给 Agent 的具体任务
- **Crew**：Agent 团队，协同完成复杂项目

**CrewAI 示例**：

```python
from crewai import Agent, Task, Crew

# 定义 Agent
researcher = Agent(
    role="研究员",
    goal="收集并分析关于 AI 框架的最新信息",
    backstory="你是一名经验丰富的技术研究员，擅长快速理解新技术",
    verbose=True
)

writer = Agent(
    role="技术作者",
    goal="撰写清晰易懂的技术文章",
    backstory="你是一名优秀的技术写作专家，能将复杂概念简化",
    verbose=True
)

# 定义任务
research_task = Task(
    description="研究 LangChain 和 LlamaIndex 的主要区别",
    agent=researcher
)

writing_task = Task(
    description="根据研究结果，撰写一篇对比文章",
    agent=writer
)

# 组建团队
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    verbose=True
)

# 执行
result = crew.kickoff()
print(result)
```

### 29.5.2 在 LangChain 中实现 CrewAI 风格的角色化 Agent

```python
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class CrewAIStyleState(TypedDict):
    project: str
    research_findings: str | None
    article_draft: str | None
    review_feedback: str | None

def researcher_agent(state: CrewAIStyleState):
    """研究员 Agent"""
    llm = ChatOpenAI(model="gpt-4")
    
    prompt = ChatPromptTemplate.from_template("""
你是一名技术研究员。项目需求：{project}

请进行深入研究，收集关键信息、技术细节和最佳实践。
""")
    
    chain = prompt | llm
    result = chain.invoke({"project": state["project"]})
    
    return {"research_findings": result.content}

def writer_agent(state: CrewAIStyleState):
    """作者 Agent"""
    llm = ChatOpenAI(model="gpt-4")
    
    prompt = ChatPromptTemplate.from_template("""
你是一名技术写作专家。基于以下研究成果，撰写一篇专业文章：

研究成果：
{research_findings}

要求：结构清晰、语言简洁、示例丰富。
""")
    
    chain = prompt | llm
    result = chain.invoke({"research_findings": state["research_findings"]})
    
    return {"article_draft": result.content}

def reviewer_agent(state: CrewAIStyleState):
    """审稿 Agent"""
    llm = ChatOpenAI(model="gpt-4")
    
    prompt = ChatPromptTemplate.from_template("""
你是资深编辑。请审阅以下文章草稿，提供改进建议：

文章草稿：
{article_draft}

请指出：1) 逻辑不清之处；2) 可补充的内容；3) 语言优化建议。
""")
    
    chain = prompt | llm
    result = chain.invoke({"article_draft": state["article_draft"]})
    
    return {"review_feedback": result.content}

# 构建 Crew 流程
workflow = StateGraph(CrewAIStyleState)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)
workflow.add_node("reviewer", reviewer_agent)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "reviewer")
workflow.add_edge("reviewer", END)

app = workflow.compile()

# 执行项目
result = app.invoke({
    "project": "对比 LangChain 和 LlamaIndex 的 RAG 实现",
    "research_findings": None,
    "article_draft": None,
    "review_feedback": None
})

print("研究成果：", result["research_findings"])
print("\n文章草稿：", result["article_draft"])
print("\n审稿意见：", result["review_feedback"])
```

## 29.6 迁移指南

### 29.6.1 从 LlamaIndex 迁移到 LangChain

**场景**：LlamaIndex RAG 项目需要扩展为复杂 Agent 系统。

**迁移步骤**：

1. **保留 LlamaIndex 索引，包装为 LangChain Retriever**（见 29.2.2）
2. **将 LlamaIndex 查询引擎改为 LangChain RetrievalQA**
3. **添加 LangChain Agent 和工具**

**迁移前（LlamaIndex）**：

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("What is RAG?")
print(response)
```

**迁移后（LangChain + LlamaIndex）**：

```python
from langchain.retrievers import LlamaIndexRetriever
from langchain.agents import Tool, create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 保留 LlamaIndex 索引
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# 包装为 LangChain Tool
def search_knowledge_base(query: str) -> str:
    """搜索知识库"""
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return str(response)

kb_tool = Tool(
    name="knowledge_base",
    func=search_knowledge_base,
    description="搜索内部知识库，获取 RAG 相关信息"
)

# 添加其他工具
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper

wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# 创建 Agent
llm = ChatOpenAI(model="gpt-4")
tools = [kb_tool, wikipedia_tool]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手，可以访问内部知识库和 Wikipedia。"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 使用
result = agent_executor.invoke({
    "input": "对比内部知识库中的 RAG 定义和 Wikipedia 上的定义"
})
print(result["output"])
```

### 29.6.2 从 Haystack 迁移到 LangChain

**场景**：Haystack 搜索系统需要升级为生成式 QA。

**核心变化**：
- Haystack Pipeline → LangChain Chain / LangGraph
- Haystack Retriever → LangChain VectorStore Retriever
- FARM Reader → ChatOpenAI（生成式 LLM）

**迁移映射**：

```python
# Haystack: BM25Retriever
haystack_retriever = BM25Retriever(document_store=document_store)

# LangChain: 改用向量检索（或自定义 BM25）
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
langchain_retriever = vectorstore.as_retriever()

# Haystack: Pipeline
pipe = Pipeline()
pipe.add_node(retriever, "Retriever", inputs=["Query"])
pipe.add_node(reader, "Reader", inputs=["Retriever"])

# LangChain: RetrievalQA Chain
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=langchain_retriever
)
```

<div data-component="MigrationPathGuide"></div>

<div data-component="APIMappingTable"></div>

## 29.7 最佳实践与建议

### 29.7.1 何时保持单一框架

**优先使用单一框架的场景**：
1. **团队规模小**：减少学习成本
2. **需求简单**：单一框架足以满足
3. **快速迭代**：避免维护多套系统的复杂度

### 29.7.2 何时混合使用多框架

**推荐混合使用的场景**：
1. **LangChain + LlamaIndex**：通用编排 + 高级 RAG
2. **LangChain + AutoGen**：可控流程 + 探索性对话
3. **LangChain + Haystack**：LLM 应用 + 传统搜索

### 29.7.3 迁移风险与缓解

**风险**：
- 依赖不兼容（版本锁定）
- 性能回退（不同框架的优化点不同）
- 可观测性断层（从 Haystack 评估迁移到 LangSmith）

**缓解措施**：
- 逐步迁移（先并行运行，再切换）
- 充分测试（对比新旧系统的准确率、延迟）
- 保留回滚方案（Feature Flag 控制）

## 29.8 章节总结

本章深入探讨了 LangChain 与其他主流框架的生态集成：

1. **框架对比**：
   - LangChain：通用编排框架
   - LlamaIndex：RAG 专家
   - Haystack：企业搜索
   - AutoGen：自主对话研究
   - CrewAI：角色化团队协作

2. **集成模式**：
   - LangChain + LlamaIndex：包装 LlamaIndex 索引为 LangChain Retriever
   - LangChain + Haystack：自定义 Retriever 桥接
   - LangChain + AutoGen：LangGraph 模拟自主对话
   - LangChain + CrewAI：角色化 Agent 流程

3. **迁移策略**：
   - 从 LlamaIndex 迁移：保留索引，扩展 Agent
   - 从 Haystack 迁移：替换 Reader 为生成式 LLM

通过合理选择和集成不同框架，您可以构建更加强大和灵活的 LLM 应用系统。

下一章（Chapter 30）将深入探讨性能优化与可靠性工程，帮助您构建生产级的高性能系统。

---

**扩展阅读**：
- [LlamaIndex 官方文档](https://docs.llamaindex.ai/)
- [Haystack 官方文档](https://haystack.deepset.ai/)
- [AutoGen 官方文档](https://microsoft.github.io/autogen/)
- [CrewAI 官方文档](https://docs.crewai.com/)

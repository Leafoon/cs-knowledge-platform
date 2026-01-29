# Appendix A: 常见问题与调试

> **本附录汇总 LangChain 生态中最常遇到的问题、错误提示、调试技巧与解决方案，涵盖 LangChain Core、LangGraph、LangSmith、LangServe 各环节的实战排障经验。**

---

## A.1 LangSmith Tracing 不生效

### 问题表现

执行 LangChain 代码后，LangSmith 平台上看不到任何追踪记录（Trace），或仅显示部分步骤。

### 常见原因与解决方案

#### 原因 1：环境变量未正确设置

**检查清单：**

```python
import os

# 必需的环境变量
required_vars = {
    "LANGCHAIN_TRACING_V2": "true",           # 启用追踪
    "LANGCHAIN_API_KEY": "lsv2_...",          # API Key
    "LANGCHAIN_PROJECT": "my-project",         # 项目名（可选，但建议设置）
}

# 诊断脚本
for key, expected in required_vars.items():
    actual = os.getenv(key)
    if not actual:
        print(f"❌ {key} 未设置")
    elif key == "LANGCHAIN_TRACING_V2" and actual.lower() != "true":
        print(f"⚠️  {key}={actual}（应为 'true'）")
    else:
        print(f"✅ {key} 已设置")
```

**解决方案：**

```bash
# .env 文件
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_your_key_here
LANGCHAIN_PROJECT=production-app
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com  # 通常不需要（使用默认值）
```

```python
from dotenv import load_dotenv
load_dotenv()  # 在导入 langchain 之前调用
```

#### 原因 2：代理/网络问题

在中国大陆或某些企业网络环境中，可能无法连接到 `api.smith.langchain.com`。

**验证连接性：**

```python
import requests

try:
    response = requests.get(
        "https://api.smith.langchain.com/info",
        timeout=5
    )
    print(f"✅ 连接成功: {response.status_code}")
except Exception as e:
    print(f"❌ 连接失败: {e}")
```

**解决方案：**

```python
# 设置代理
os.environ["HTTP_PROXY"] = "http://proxy.company.com:8080"
os.environ["HTTPS_PROXY"] = "http://proxy.company.com:8080"

# 或使用自托管 LangSmith（企业版）
os.environ["LANGCHAIN_ENDPOINT"] = "https://langsmith.internal.company.com"
```

#### 原因 3：Runnable 没有通过 LCEL 构建

使用传统 Chain 或自定义函数时，可能未正确继承追踪上下文。

**错误示例：**

```python
# ❌ 普通 Python 函数不会自动追踪
def my_chain(input_text):
    response = llm.invoke(input_text)
    return response.upper()

result = my_chain("Hello")  # 不会出现在 LangSmith
```

**正确示例：**

```python
from langchain_core.runnables import RunnableLambda

# ✅ 包装为 Runnable
my_chain = RunnableLambda(lambda x: llm.invoke(x).upper())
result = my_chain.invoke("Hello")  # 会追踪
```

#### 原因 4：异步代码未正确处理

在 Jupyter Notebook 或异步环境中，需确保事件循环正确管理。

**问题代码：**

```python
# ❌ 在 Jupyter 中混用 sync/async
import asyncio

async def run():
    result = await chain.ainvoke("Hello")
    return result

# 直接调用可能导致追踪丢失
asyncio.run(run())  # 可能创建新的事件循环
```

**解决方案：**

```python
# ✅ 使用 await（在异步环境中）
result = await chain.ainvoke("Hello")

# ✅ 或使用 nest_asyncio（Jupyter）
import nest_asyncio
nest_asyncio.apply()
```

#### 原因 5：批处理中的部分失败

批处理时，如果某些项失败但被忽略，可能导致追踪不完整。

**诊断：**

```python
from langchain.callbacks.tracers import ConsoleCallbackHandler

# 添加本地回调查看执行细节
chain.batch(
    ["input1", "input2", "input3"],
    config={"callbacks": [ConsoleCallbackHandler()]}
)
```

### 调试技巧

#### 使用 `langsmith.utils.tracing_context`

```python
from langsmith.run_helpers import traceable

@traceable(run_type="chain", name="custom-chain")
def my_custom_chain(input_text: str) -> str:
    """强制追踪自定义函数"""
    result = llm.invoke(input_text)
    return result.content

# 会在 LangSmith 中显示为独立的 Run
my_custom_chain("Hello")
```

#### 启用详细日志

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("langchain")
logger.setLevel(logging.DEBUG)

# 执行链后查看日志中的追踪 URL
chain.invoke("Hello")
# 输出: View trace at https://smith.langchain.com/...
```

---

## A.2 LCEL 类型推断错误

### 问题表现

IDE 提示类型不匹配、运行时出现 `AttributeError`、或链无法正确组合。

### 常见错误与解决方案

#### 错误 1：输入输出类型不匹配

**错误代码：**

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("Translate to {language}: {text}")
model = ChatOpenAI()
parser = StrOutputParser()

# ❌ 类型不匹配：prompt 需要 dict，但可能传入 str
chain = prompt | model | parser
chain.invoke("Hello")  # TypeError: Expected dict, got str
```

**解决方案：**

```python
# ✅ 传入正确的字典格式
chain.invoke({"language": "French", "text": "Hello"})

# ✅ 或添加输入适配器
from langchain_core.runnables import RunnableLambda

input_adapter = RunnableLambda(lambda x: {"language": "French", "text": x})
chain = input_adapter | prompt | model | parser
chain.invoke("Hello")  # 现在可以接受 str
```

#### 错误 2：RunnablePassthrough 使用不当

**错误代码：**

```python
from langchain_core.runnables import RunnablePassthrough

# ❌ 以为会透传整个字典
chain = RunnablePassthrough() | model
chain.invoke({"text": "Hello"})  # model 收到 dict 而非 str
```

**解决方案：**

```python
# ✅ 使用 RunnablePassthrough.assign() 添加字段
chain = (
    RunnablePassthrough.assign(
        response=model
    )
)
chain.invoke({"text": "Hello"})
# 输出: {"text": "Hello", "response": AIMessage(...)}

# ✅ 或使用 itemgetter 提取字段
from operator import itemgetter

chain = (
    {"text": itemgetter("text")}
    | ChatPromptTemplate.from_template("Translate: {text}")
    | model
)
```

#### 错误 3：Pydantic 模型验证失败

**错误代码：**

```python
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser

class Person(BaseModel):
    name: str
    age: int

parser = PydanticOutputParser(pydantic_object=Person)

# ❌ LLM 输出格式不符合 JSON
chain = model | parser
chain.invoke("Extract person from: John is 30 years old")
# JSONDecodeError: Expecting value...
```

**解决方案：**

```python
# ✅ 使用 with_structured_output（推荐）
model_with_structure = model.with_structured_output(Person)
result = model_with_structure.invoke("John is 30 years old")
print(result)  # Person(name='John', age=30)

# ✅ 或显式指导 LLM 输出 JSON
prompt = ChatPromptTemplate.from_template(
    "Extract person info as JSON:\n{text}\n\n{format_instructions}"
)
chain = (
    {"text": RunnablePassthrough(), "format_instructions": lambda _: parser.get_format_instructions()}
    | prompt
    | model
    | parser
)
```

### 类型注解最佳实践

```python
from typing import TypedDict
from langchain_core.runnables import Runnable

# ✅ 定义输入输出类型
class ChainInput(TypedDict):
    language: str
    text: str

class ChainOutput(TypedDict):
    translation: str

# 类型标注链
chain: Runnable[ChainInput, ChainOutput] = (
    prompt | model | StrOutputParser()
)

# IDE 现在可以提供自动补全
result = chain.invoke({"language": "French", "text": "Hello"})
print(result["translation"])  # ✅ IDE 知道这是 str
```

---

## A.3 LangGraph 状态更新失败

### 问题表现

- 节点执行后状态未更新
- `checkpointer.get()` 返回 `None`
- 条件边无法正确路由

### 常见原因与解决方案

#### 原因 1：状态键名不匹配

**错误代码：**

```python
from typing import TypedDict
from langgraph.graph import StateGraph

class State(TypedDict):
    messages: list
    context: str

def node_a(state: State) -> dict:
    # ❌ 返回了不在 State 中的键
    return {"message": "Hello"}  # 应为 "messages"

graph = StateGraph(State)
graph.add_node("node_a", node_a)
```

**解决方案：**

```python
def node_a(state: State) -> dict:
    # ✅ 返回符合 State 定义的键
    return {"messages": state["messages"] + ["Hello"]}
```

#### 原因 2：Reducer 配置错误

**错误代码：**

```python
from typing import Annotated
from operator import add

class State(TypedDict):
    # ❌ 使用 add 但传入的不是可加类型
    messages: Annotated[list, add]

def node_a(state: State) -> dict:
    # 返回单个 str 而非 list
    return {"messages": "Hello"}  # TypeError: can only concatenate list to list
```

**解决方案：**

```python
class State(TypedDict):
    messages: Annotated[list, add]  # ✅ 确保返回 list

def node_a(state: State) -> dict:
    return {"messages": ["Hello"]}  # ✅ 包装为列表
```

#### 原因 3：Checkpointer 未正确配置

**问题代码：**

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = StateGraph(State)
# ... 添加节点/边
app = graph.compile(checkpointer=checkpointer)

# ❌ 调用时未提供 thread_id
result = app.invoke({"messages": []})
```

**解决方案：**

```python
# ✅ 提供 config 包含 thread_id
config = {"configurable": {"thread_id": "conversation-1"}}
result = app.invoke({"messages": []}, config=config)

# 验证状态已保存
snapshot = checkpointer.get(config)
print(snapshot)  # CheckpointTuple(...)
```

#### 原因 4：条件边返回值错误

**错误代码：**

```python
def should_continue(state: State) -> str:
    if len(state["messages"]) > 10:
        return "end"  # ❌ 但图中没有名为 "end" 的节点
    return "continue"

graph.add_conditional_edges("node_a", should_continue)
```

**解决方案：**

```python
from langgraph.graph import END

def should_continue(state: State) -> str:
    if len(state["messages"]) > 10:
        return END  # ✅ 使用 LangGraph 的 END 常量
    return "node_b"  # ✅ 确保节点存在

graph.add_conditional_edges(
    "node_a",
    should_continue,
    {
        END: END,
        "node_b": "node_b"
    }
)
```

### 调试技巧

#### 1. 启用详细日志

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("langgraph")
logger.setLevel(logging.DEBUG)

# 执行图
app.invoke({"messages": []}, config=config)
# 查看日志中的状态更新
```

#### 2. 打印中间状态

```python
def debug_node(state: State) -> dict:
    print(f"🔍 Current state: {state}")
    return {}

# 在关键位置插入调试节点
graph.add_node("debug", debug_node)
graph.add_edge("node_a", "debug")
graph.add_edge("debug", "node_b")
```

#### 3. 使用 `stream` 查看执行过程

```python
for event in app.stream({"messages": []}, config=config):
    print(f"📦 Event: {event}")
    # 输出每个节点的输入/输出
```

---

## A.4 Agent 陷入无限循环

### 问题表现

Agent 反复调用相同工具、生成相同输出、或长时间不返回结果。

### 常见原因与解决方案

#### 原因 1：工具输出格式不符合预期

**错误场景：**

```python
from langchain.tools import tool

@tool
def search_tool(query: str) -> str:
    """Search the web"""
    # ❌ 返回空字符串或无用信息
    return ""

# Agent 无法判断任务完成，继续调用工具
```

**解决方案：**

```python
@tool
def search_tool(query: str) -> str:
    """Search the web"""
    results = perform_search(query)
    if not results:
        # ✅ 返回明确的失败信息
        return "No results found. Try a different query."
    # ✅ 返回结构化且有信息量的结果
    return f"Found {len(results)} results:\n" + "\n".join(results[:3])
```

#### 原因 2：缺少最大迭代限制

**错误代码：**

```python
from langgraph.prebuilt import create_react_agent

# ❌ 没有设置 max_iterations
agent = create_react_agent(model, tools)
agent.invoke({"messages": [("user", "Find info about X")]})
```

**解决方案：**

```python
# ✅ 设置最大迭代次数
agent = create_react_agent(
    model,
    tools,
    state_modifier="You must complete the task in 5 steps or less."
)

# 或在自定义图中添加循环检测
from langgraph.graph import END

def should_continue(state):
    if len(state["messages"]) > 20:  # ✅ 强制退出
        return END
    # ... 其他逻辑
```

#### 原因 3：Agent 提示词不明确

**问题提示：**

```python
system_prompt = "You are a helpful assistant."
# ❌ 没有明确何时停止
```

**改进提示：**

```python
system_prompt = """You are a helpful assistant. Follow these rules:
1. Use tools to gather information
2. Once you have enough information, provide a FINAL ANSWER
3. Do NOT call the same tool twice with the same arguments
4. If a tool returns no results, try a different approach OR admit you cannot find the answer
5. ALWAYS end your response with "FINAL ANSWER:" when task is complete
"""
```

#### 原因 4：工具依赖循环

**错误场景：**

```python
# Tool A 的输出需要 Tool B 处理
# Tool B 的输出又需要 Tool A 处理
# 形成死循环
```

**解决方案：**

```python
# ✅ 使用 LangGraph 显式定义工具调用顺序
graph = StateGraph(State)
graph.add_node("tool_a", tool_a_node)
graph.add_node("tool_b", tool_b_node)
graph.add_edge("tool_a", "tool_b")  # 强制顺序
graph.add_edge("tool_b", END)  # 防止循环
```

### 调试技巧

#### 1. 监控工具调用

```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = agent.invoke({"messages": [("user", "Query")]})
    print(f"Total calls: {cb.successful_requests}")
    print(f"Total tokens: {cb.total_tokens}")
    # 如果调用次数异常高，说明有循环
```

#### 2. 记录每次迭代

```python
class IterationTracker:
    def __init__(self):
        self.iterations = []
    
    def track(self, action: str):
        self.iterations.append(action)
        if len(self.iterations) > 10:
            raise RuntimeError("Too many iterations!")

tracker = IterationTracker()

# 在节点中调用
def agent_node(state):
    tracker.track(state["next_action"])
    # ...
```

---

## A.5 RAG 检索质量差

### 问题表现

- 检索到的文档与查询无关
- 相似度分数很低
- 明明有相关文档却检索不到

### 常见原因与解决方案

#### 原因 1：Embedding 模型不匹配

**问题代码：**

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 索引时使用 OpenAI embeddings
embeddings_v1 = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = Chroma.from_documents(docs, embeddings_v1)

# ❌ 检索时换了模型
embeddings_v2 = OpenAIEmbeddings(model="text-embedding-3-small")
retriever = vectorstore.as_retriever(embedding=embeddings_v2)
```

**解决方案：**

```python
# ✅ 始终使用同一个 embedding 实例
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 索引
vectorstore = Chroma.from_documents(docs, embeddings)

# 检索（使用相同实例）
retriever = vectorstore.as_retriever()
```

#### 原因 2：文档分块不合理

**错误分块：**

```python
from langchain.text_splitter import CharacterTextSplitter

# ❌ Chunk 太大（2000 字符）导致语义混杂
splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
docs = splitter.split_documents(raw_docs)
```

**优化分块：**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ 适中的 chunk size，带重叠
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,          # 根据 embedding 模型调整
    chunk_overlap=50,        # 保留上下文
    separators=["\n\n", "\n", "。", "！", "？", " ", ""]
)
docs = splitter.split_documents(raw_docs)
```

#### 原因 3：查询改写不足

**基础查询：**

```python
# ❌ 直接使用用户原始查询
query = "Python 怎么读文件？"
docs = retriever.get_relevant_documents(query)
```

**查询改写：**

```python
from langchain.retrievers import MultiQueryRetriever

# ✅ 生成多个变体查询
retriever_with_rewrite = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=ChatOpenAI(temperature=0)
)
docs = retriever_with_rewrite.get_relevant_documents(query)
```

#### 原因 4：未使用混合检索

**纯向量检索：**

```python
# ❌ 仅依赖语义相似度
retriever = vectorstore.as_retriever(search_type="similarity")
```

**混合检索：**

```python
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever

# ✅ 结合向量检索 + BM25 关键词检索
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 5

ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]  # 各占 50%
)
```

#### 原因 5：缺少元数据过滤

**无过滤：**

```python
# ❌ 检索所有文档，包括无关来源
docs = retriever.get_relevant_documents("2024 年政策")
```

**元数据过滤：**

```python
# ✅ 添加时间/来源过滤
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"year": {"$gte": 2024}}  # 仅检索 2024 年后的文档
    }
)
```

### 调试技巧

#### 1. 检查文档嵌入质量

```python
# 查看 embedding 向量
query_embedding = embeddings.embed_query("测试查询")
print(f"Embedding dimension: {len(query_embedding)}")
print(f"First 5 values: {query_embedding[:5]}")

# 计算相似度
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

doc_embedding = embeddings.embed_query("文档内容")
similarity = cosine_similarity(query_embedding, doc_embedding)
print(f"Similarity: {similarity:.4f}")  # 应 > 0.7 才算相关
```

#### 2. 可视化检索结果

```python
docs_with_scores = vectorstore.similarity_search_with_score(query, k=10)

for i, (doc, score) in enumerate(docs_with_scores):
    print(f"\n{'='*60}")
    print(f"Rank {i+1} | Score: {score:.4f}")
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Metadata: {doc.metadata}")
```

#### 3. A/B 测试不同配置

```python
# 对比不同配置的检索质量
configs = [
    {"chunk_size": 500, "k": 5},
    {"chunk_size": 1000, "k": 3},
    {"chunk_size": 200, "k": 10},
]

for config in configs:
    # 重新索引
    splitter = RecursiveCharacterTextSplitter(chunk_size=config["chunk_size"])
    docs = splitter.split_documents(raw_docs)
    vectorstore = Chroma.from_documents(docs, embeddings)
    
    # 检索
    retriever = vectorstore.as_retriever(search_kwargs={"k": config["k"]})
    results = retriever.get_relevant_documents(test_query)
    
    # 人工评估
    print(f"\nConfig: {config}")
    for doc in results[:3]:
        print(f"  - {doc.page_content[:100]}...")
```

---

## A.6 流式输出中断或乱码

### 问题表现

- `astream()` 中途停止
- Token 顺序错乱
- 中文字符显示为 `�`

### 解决方案

#### 问题 1：缓冲区未刷新

```python
import sys

async for chunk in chain.astream("Hello"):
    print(chunk, end="", flush=True)  # ✅ 立即刷新缓冲区
```

#### 问题 2：编码问题

```python
# ✅ 确保 UTF-8 编码
import sys
sys.stdout.reconfigure(encoding='utf-8')
```

#### 问题 3：异常未捕获

```python
try:
    async for chunk in chain.astream("Hello"):
        print(chunk, end="")
except Exception as e:
    print(f"\n❌ Stream error: {e}")
```

---

## A.7 LangServe 部署 422 错误

### 问题表现

调用 `/invoke` 端点时返回 `422 Unprocessable Entity`。

### 常见原因

#### 原因 1：请求体格式错误

```python
# ❌ 错误格式
requests.post(
    "http://localhost:8000/chain/invoke",
    json={"input": "Hello"}  # 缺少必需的 "input" 包装
)

# ✅ 正确格式
requests.post(
    "http://localhost:8000/chain/invoke",
    json={"input": {"text": "Hello"}}  # 根据链的输入结构
)
```

#### 原因 2：Schema 不匹配

```python
# 服务端链定义
class ChainInput(BaseModel):
    text: str
    language: str

# ❌ 客户端遗漏字段
requests.post(url, json={"input": {"text": "Hello"}})

# ✅ 提供完整字段
requests.post(url, json={"input": {"text": "Hello", "language": "en"}})
```

### 调试技巧

```python
# 查看链的输入 schema
response = requests.get("http://localhost:8000/chain/input_schema")
print(response.json())  # 查看期望的输入格式
```

---

## A.8 内存占用过高

### 问题表现

运行 RAG 应用或 Agent 时内存持续增长，最终 OOM。

### 常见原因与解决方案

#### 原因 1：对话历史未限制

```python
# ❌ 无限增长的历史
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
# 每次对话都追加，永不清理
```

**解决方案：**

```python
# ✅ 使用窗口记忆
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=10)  # 仅保留最近 10 轮

# ✅ 或使用摘要记忆
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)  # 自动压缩
```

#### 原因 2：向量库加载到内存

```python
# ❌ Chroma 默认在内存中
vectorstore = Chroma.from_documents(docs, embeddings)
```

**解决方案：**

```python
# ✅ 持久化到磁盘
vectorstore = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory="./chroma_db"
)
```

#### 原因 3：文档未及时释放

```python
# ✅ 使用生成器而非列表
def load_docs():
    for file in files:
        yield load_file(file)

# 而非
docs = [load_file(f) for f in files]  # ❌ 全部加载到内存
```

---

## A.9 LangGraph Checkpoint 恢复失败

### 问题表现

调用 `get_state()` 返回 `None`，或状态与预期不符。

### 解决方案

#### 1. 确认 thread_id 一致

```python
# 保存时
config1 = {"configurable": {"thread_id": "abc"}}
app.invoke(input, config=config1)

# 恢复时必须使用相同 ID
config2 = {"configurable": {"thread_id": "abc"}}  # ✅
state = app.get_state(config2)
```

#### 2. 使用持久化 Checkpointer

```python
# ❌ MemorySaver 重启后丢失
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

# ✅ SqliteSaver 持久化
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
```

---

## A.10 快速诊断清单

当遇到问题时，按以下顺序排查：

```python
# 1. 检查版本
import langchain
print(f"LangChain: {langchain.__version__}")

# 2. 检查环境变量
import os
print(f"Tracing: {os.getenv('LANGCHAIN_TRACING_V2')}")
print(f"API Key: {os.getenv('OPENAI_API_KEY')[:10]}...")

# 3. 测试基础功能
from langchain_openai import ChatOpenAI
llm = ChatOpenAI()
response = llm.invoke("Hello")
print(f"LLM working: {bool(response)}")

# 4. 检查网络连接
import requests
try:
    requests.get("https://api.openai.com", timeout=5)
    print("✅ Network OK")
except:
    print("❌ Network issue")

# 5. 查看日志
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

**下一步**：完成其他附录（B-E）以及最终集成测试。

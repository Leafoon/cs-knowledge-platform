# Appendix B: API 速查表

> **本附录提供 LangChain 生态核心 API 的快速参考，涵盖最常用的类、方法、参数，适合日常开发查阅。**

---

## B.1 Runnable 核心方法

### 基础调用

| 方法 | 签名 | 说明 | 返回值 |
|------|------|------|--------|
| `invoke()` | `invoke(input, config=None)` | 同步单次调用 | `Output` |
| `ainvoke()` | `async ainvoke(input, config=None)` | 异步单次调用 | `Output` |
| `batch()` | `batch(inputs, config=None, **kwargs)` | 同步批量调用 | `List[Output]` |
| `abatch()` | `async abatch(inputs, config=None, **kwargs)` | 异步批量调用 | `List[Output]` |
| `stream()` | `stream(input, config=None)` | 同步流式输出 | `Iterator[Output]` |
| `astream()` | `async astream(input, config=None)` | 异步流式输出 | `AsyncIterator[Output]` |
| `astream_events()` | `async astream_events(input, config=None, version="v1")` | 事件流（包含中间步骤） | `AsyncIterator[StreamEvent]` |

### 配置与组合

| 方法 | 签名 | 说明 |
|------|------|------|
| `with_config()` | `with_config(config)` | 绑定配置（tags、metadata、callbacks 等） |
| `with_types()` | `with_types(input_type, output_type)` | 显式指定类型 |
| `with_retry()` | `with_retry(retry_if_exception_type, stop_after_attempt=3, wait_exponential_jitter=True)` | 添加重试机制 |
| `with_fallbacks()` | `with_fallbacks(fallbacks)` | 添加降级备选 |
| `configurable_fields()` | `configurable_fields(**kwargs)` | 标记可配置字段 |
| `configurable_alternatives()` | `configurable_alternatives(ConfigurableField(...), **kwargs)` | 标记可切换组件 |

### 组合操作符

| 操作符 | 等价方法 | 说明 | 示例 |
|--------|----------|------|------|
| `\|` | `pipe()` | 顺序组合 | `prompt \| model \| parser` |
| - | `RunnableParallel()` | 并行组合 | `{"a": chain1, "b": chain2}` |
| - | `RunnableBranch()` | 条件分支 | `RunnableBranch((cond1, chain1), (cond2, chain2), default)` |

### 示例代码

```python
from langchain_core.runnables import RunnableLambda

# 基础调用
chain = prompt | model | parser
result = chain.invoke({"input": "Hello"})                    # 同步
result = await chain.ainvoke({"input": "Hello"})             # 异步
results = chain.batch([{"input": "A"}, {"input": "B"}])      # 批量

# 流式输出
for chunk in chain.stream({"input": "Hello"}):               # 同步流
    print(chunk, end="")

async for chunk in chain.astream({"input": "Hello"}):        # 异步流
    print(chunk, end="")

# 事件流（查看中间步骤）
async for event in chain.astream_events({"input": "Hello"}, version="v1"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")

# 配置
chain_with_config = chain.with_config(
    tags=["production"],
    metadata={"user_id": "123"},
    run_name="translation-chain"
)

# 重试 + Fallback
reliable_chain = chain.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True
).with_fallbacks([fallback_chain])
```

---

## B.2 LCEL 常用组件

### RunnablePassthrough

| 用法 | 说明 | 示例 |
|------|------|------|
| `RunnablePassthrough()` | 透传输入到输出 | `RunnablePassthrough() \| model` |
| `RunnablePassthrough.assign(key=runnable)` | 添加新字段（保留原输入） | `RunnablePassthrough.assign(summary=summarize_chain)` |

```python
from langchain_core.runnables import RunnablePassthrough

# 透传
chain = RunnablePassthrough() | model
chain.invoke("Hello")  # model 收到 "Hello"

# 添加字段
chain = RunnablePassthrough.assign(
    summary=summarize_chain,
    translation=translate_chain
)
result = chain.invoke({"text": "Long document..."})
# 输出: {"text": "...", "summary": "...", "translation": "..."}
```

### RunnableLambda

```python
from langchain_core.runnables import RunnableLambda

# 包装普通函数
def uppercase(x: str) -> str:
    return x.upper()

chain = RunnableLambda(uppercase) | model

# 或使用装饰器
from langchain_core.runnables import chain as chain_decorator

@chain_decorator
def custom_chain(x: dict) -> str:
    return x["text"].upper()
```

### RunnableParallel

```python
from langchain_core.runnables import RunnableParallel

# 并行执行多个链
parallel_chain = RunnableParallel(
    summary=summarize_chain,
    keywords=extract_keywords_chain,
    sentiment=sentiment_chain
)
result = parallel_chain.invoke({"text": "..."})
# 输出: {"summary": "...", "keywords": [...], "sentiment": "positive"}

# 简写形式（字典即为 RunnableParallel）
parallel_chain = {
    "summary": summarize_chain,
    "keywords": extract_keywords_chain
}
```

### RunnableBranch

```python
from langchain_core.runnables import RunnableBranch

# 条件路由
branch = RunnableBranch(
    (lambda x: x["lang"] == "en", english_chain),
    (lambda x: x["lang"] == "zh", chinese_chain),
    default_chain  # 默认分支
)
result = branch.invoke({"lang": "en", "text": "Hello"})
```

### itemgetter / attrgetter

```python
from operator import itemgetter, attrgetter

# 提取字典字段
chain = (
    {"text": itemgetter("input"), "lang": itemgetter("language")}
    | prompt
    | model
)
chain.invoke({"input": "Hello", "language": "French"})

# 提取对象属性
chain = attrgetter("content") | model
```

---

## B.3 Prompt Templates

### PromptTemplate

```python
from langchain_core.prompts import PromptTemplate

# 基础创建
prompt = PromptTemplate.from_template("Translate {text} to {language}")

# 完整参数
prompt = PromptTemplate(
    template="Translate {text} to {language}",
    input_variables=["text", "language"],
    partial_variables={"language": "French"}  # 部分填充
)

# 常用方法
formatted = prompt.format(text="Hello", language="French")
messages = prompt.invoke({"text": "Hello", "language": "French"})
```

### ChatPromptTemplate

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 基础创建
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", "{input}")
])

# 复杂消息组合
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}"),
    MessagesPlaceholder("history"),          # 对话历史占位符
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")  # Agent 工作区
])

# Few-Shot 提示
from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples = [
    {"input": "Hi", "output": "Hello!"},
    {"input": "Bye", "output": "Goodbye!"}
]
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt
)
```

### 从 LangChain Hub 加载

```python
from langchain import hub

# 拉取公开提示模板
prompt = hub.pull("rlm/rag-prompt")
prompt = hub.pull("hwchase17/react")

# 推送自定义模板
hub.push("my-username/my-prompt", prompt)
```

---

## B.4 Output Parsers

### 常用 Parsers

| Parser | 输出类型 | 用途 |
|--------|----------|------|
| `StrOutputParser` | `str` | 提取 AI 消息的文本内容 |
| `JsonOutputParser` | `dict` | 解析 JSON 格式输出 |
| `PydanticOutputParser` | Pydantic Model | 结构化输出 + 验证 |
| `CommaSeparatedListOutputParser` | `List[str]` | 解析逗号分隔列表 |
| `DatetimeOutputParser` | `datetime` | 解析日期时间 |
| `EnumOutputParser` | `Enum` | 限定枚举值 |

### 代码示例

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# StrOutputParser（最常用）
chain = prompt | model | StrOutputParser()

# JsonOutputParser
json_parser = JsonOutputParser()
prompt = ChatPromptTemplate.from_template(
    "Extract info as JSON: {text}\n{format_instructions}"
)
chain = (
    {"text": RunnablePassthrough(), "format_instructions": lambda _: json_parser.get_format_instructions()}
    | prompt | model | json_parser
)

# PydanticOutputParser
class Person(BaseModel):
    name: str = Field(description="Person's name")
    age: int = Field(description="Person's age")

pydantic_parser = PydanticOutputParser(pydantic_object=Person)
chain = prompt | model | pydantic_parser

# 推荐：使用 with_structured_output（更简洁）
model_with_structure = model.with_structured_output(Person)
result = model_with_structure.invoke("John is 30 years old")
print(result)  # Person(name='John', age=30)
```

---

## B.5 LangGraph API

### StateGraph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from operator import add

# 定义状态
class State(TypedDict):
    messages: Annotated[list, add]  # 使用 add reducer
    context: str

# 创建图
graph = StateGraph(State)

# 添加节点
graph.add_node("node1", node1_function)
graph.add_node("node2", node2_function)

# 添加边
graph.add_edge("node1", "node2")       # 无条件边
graph.add_edge("node2", END)           # 结束

# 条件边
graph.add_conditional_edges(
    "node1",
    route_function,                     # 返回下一个节点名
    {
        "continue": "node2",
        "end": END
    }
)

# 设置入口
graph.set_entry_point("node1")

# 编译
app = graph.compile()
```

### 节点函数签名

```python
from langgraph.graph import StateGraph

def my_node(state: State) -> dict:
    """
    节点函数必须：
    1. 接受 state 参数（完整状态）
    2. 返回 dict（部分更新）
    """
    return {
        "messages": state["messages"] + ["new message"],
        "context": "updated context"
    }
```

### Checkpointer

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# 内存 Checkpointer（测试用）
checkpointer = MemorySaver()

# SQLite 持久化
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# 编译时传入
app = graph.compile(checkpointer=checkpointer)

# 调用时提供 thread_id
config = {"configurable": {"thread_id": "conversation-123"}}
result = app.invoke({"messages": []}, config=config)

# 获取状态
snapshot = app.get_state(config)
print(snapshot.values)  # 当前状态
print(snapshot.next)    # 下一个节点

# 更新状态
app.update_state(config, {"messages": ["manual update"]})
```

### 中断与恢复（Human-in-the-Loop）

```python
from langgraph.graph import StateGraph

def approval_node(state):
    # 节点执行后会中断，等待人工输入
    return {"status": "pending_approval"}

graph.add_node("approval", approval_node)

# 编译时指定中断点
app = graph.compile(
    checkpointer=checkpointer,
    interrupt_before=["approval"]  # 在此节点前中断
)

# 首次调用（执行到中断点）
config = {"configurable": {"thread_id": "123"}}
result = app.invoke({"messages": []}, config=config)

# 人工审批后恢复
app.update_state(config, {"approved": True})
result = app.invoke(None, config=config)  # 继续执行
```

---

## B.6 LangSmith API

### 环境变量

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=lsv2_pt_...
export LANGCHAIN_PROJECT=my-project
export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com  # 可选
```

### 自定义追踪

```python
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

# 装饰器追踪
@traceable(run_type="chain", name="custom-chain")
def my_function(input_text: str) -> str:
    # 自动追踪
    result = process(input_text)
    return result

# 获取当前 Run 信息
@traceable
def my_function(input_text):
    run_tree = get_current_run_tree()
    print(f"Run ID: {run_tree.id}")
    print(f"Trace URL: {run_tree.get_url()}")
    return process(input_text)
```

### 数据集操作

```python
from langsmith import Client

client = Client()

# 创建数据集
dataset = client.create_dataset("my-dataset", description="Test dataset")

# 添加示例
client.create_example(
    dataset_id=dataset.id,
    inputs={"question": "What is LangChain?"},
    outputs={"answer": "A framework for building LLM apps"}
)

# 列出数据集
datasets = client.list_datasets()

# 读取示例
examples = client.list_examples(dataset_id=dataset.id)
```

### 评估

```python
from langsmith.evaluation import evaluate, LangChainStringEvaluator

# 定义评估器
evaluators = [
    LangChainStringEvaluator("cot_qa"),      # 内置评估器
    LangChainStringEvaluator("criteria", config={"criteria": "helpfulness"})
]

# 运行评估
results = evaluate(
    lambda inputs: my_chain.invoke(inputs),
    data="my-dataset",
    evaluators=evaluators,
    experiment_prefix="experiment-v1"
)

# 查看结果
print(results)
```

---

## B.7 LangServe API

### 添加路由

```python
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.runnables import RunnableLambda

app = FastAPI(title="My API", version="1.0")

# 添加链路由
add_routes(
    app,
    chain,
    path="/my-chain",                    # 端点路径
    enabled_endpoints=["invoke", "batch", "stream"],  # 启用的端点
    input_type=ChainInput,               # 可选：输入类型
    output_type=ChainOutput,             # 可选：输出类型
    config_keys=["configurable"]         # 可配置参数
)
```

### 可用端点

| 端点 | HTTP 方法 | 说明 | 请求体示例 |
|------|-----------|------|-----------|
| `/invoke` | POST | 单次调用 | `{"input": {"text": "Hello"}}` |
| `/batch` | POST | 批量调用 | `{"inputs": [{"text": "A"}, {"text": "B"}]}` |
| `/stream` | POST | 流式输出 | `{"input": {"text": "Hello"}}` |
| `/stream_log` | POST | 流式日志（包含中间步骤） | `{"input": {"text": "Hello"}}` |
| `/input_schema` | GET | 获取输入 Schema | - |
| `/output_schema` | GET | 获取输出 Schema | - |
| `/config_schema` | GET | 获取配置 Schema | - |
| `/playground` | GET | 交互式 Playground | - |

### 客户端调用

```python
from langserve import RemoteRunnable

# 创建远程客户端
remote_chain = RemoteRunnable("http://localhost:8000/my-chain")

# 调用（与本地 Runnable 完全一致）
result = remote_chain.invoke({"text": "Hello"})
results = remote_chain.batch([{"text": "A"}, {"text": "B"}])

async for chunk in remote_chain.astream({"text": "Hello"}):
    print(chunk, end="")
```

### 配置化调用

```python
# 服务端：定义可配置字段
chain = chain.configurable_fields(
    temperature=ConfigurableField(
        id="temperature",
        name="LLM Temperature",
        default=0.7
    )
)

# 客户端：传递配置
result = remote_chain.invoke(
    {"text": "Hello"},
    config={"configurable": {"temperature": 0.9}}
)
```

---

## B.8 Memory / Chat History

### ConversationBufferMemory

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()

# 保存对话
memory.save_context(
    {"input": "Hello"},
    {"output": "Hi there!"}
)

# 加载历史
history = memory.load_memory_variables({})
print(history["history"])  # "Human: Hello\nAI: Hi there!"
```

### ChatMessageHistory

```python
from langchain_community.chat_message_histories import ChatMessageHistory

history = ChatMessageHistory()

# 添加消息
history.add_user_message("Hello")
history.add_ai_message("Hi!")

# 获取消息
messages = history.messages  # List[BaseMessage]
```

### RunnableWithMessageHistory

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 存储每个会话的历史
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 包装链
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 调用时提供 session_id
result = chain_with_history.invoke(
    {"input": "Hello"},
    config={"configurable": {"session_id": "user123"}}
)
```

---

## B.9 Retrievers

### VectorStore Retriever

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings)

# 创建 Retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",           # 或 "mmr", "similarity_score_threshold"
    search_kwargs={"k": 5}              # 返回 Top-5
)

# 检索
docs = retriever.get_relevant_documents("query")

# 异步检索
docs = await retriever.aget_relevant_documents("query")
```

### MultiQueryRetriever

```python
from langchain.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

# 自动生成多个查询变体
docs = retriever.get_relevant_documents("How to use LangChain?")
```

### ContextualCompressionRetriever

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 检索后压缩文档（仅保留相关部分）
docs = compression_retriever.get_relevant_documents("query")
```

---

## B.10 Tools

### 定义工具

```python
from langchain.tools import tool
from pydantic import BaseModel, Field

# 方式 1：装饰器
@tool
def search_tool(query: str) -> str:
    """Search the web for information"""
    return perform_search(query)

# 方式 2：带参数验证
class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Max number of results")

@tool(args_schema=SearchInput)
def search_tool(query: str, max_results: int = 5) -> str:
    """Search the web for information"""
    return perform_search(query, max_results)

# 方式 3：从函数创建
from langchain.tools import Tool

def my_function(x: str) -> str:
    return x.upper()

tool = Tool(
    name="uppercase",
    description="Converts text to uppercase",
    func=my_function
)
```

### 使用工具

```python
# 直接调用
result = search_tool.invoke({"query": "LangChain"})

# 在 Agent 中使用
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model, [search_tool, calculator_tool])
result = agent.invoke({"messages": [("user", "Search for LangChain")]})
```

---

## B.11 Document Loaders

### 常用 Loaders

```python
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
    DirectoryLoader
)

# 文本文件
loader = TextLoader("file.txt")
docs = loader.load()

# PDF
loader = PyPDFLoader("file.pdf")
docs = loader.load()

# 网页
loader = WebBaseLoader("https://example.com")
docs = loader.load()

# 目录（批量）
loader = DirectoryLoader("./docs", glob="**/*.md")
docs = loader.load()
```

---

## B.12 Text Splitters

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter
)

# 推荐：递归分割器
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", " ", ""]
)
docs = splitter.split_documents(raw_docs)

# Token 分割器（按 token 数）
splitter = TokenTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)
```

---

## B.13 Callbacks

### 常用回调

```python
from langchain.callbacks import (
    StdOutCallbackHandler,
    FileCallbackHandler,
    get_openai_callback
)

# 标准输出
chain.invoke(input, config={"callbacks": [StdOutCallbackHandler()]})

# 文件输出
chain.invoke(input, config={"callbacks": [FileCallbackHandler("log.txt")]})

# Token 计数
with get_openai_callback() as cb:
    result = chain.invoke(input)
    print(f"Tokens: {cb.total_tokens}")
    print(f"Cost: ${cb.total_cost}")
```

### 自定义回调

```python
from langchain.callbacks.base import BaseCallbackHandler

class MyCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM started with prompts: {prompts}")
    
    def on_llm_end(self, response, **kwargs):
        print(f"LLM ended with response: {response}")
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        print(f"Chain started with inputs: {inputs}")

# 使用
chain.invoke(input, config={"callbacks": [MyCallbackHandler()]})
```

---

**本附录持续更新，更多 API 参考请查阅官方文档：**

- **LangChain Python API**: https://api.python.langchain.com/
- **LangGraph API**: https://langchain-ai.github.io/langgraph/
- **LangSmith API**: https://docs.smith.langchain.com/

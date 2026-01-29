# Appendix E: 版本迁移指南

> **本附录提供从 Legacy Chains 迁移到 LCEL、从 LangChain 0.1 升级到 0.3、以及从其他框架迁移到 LangChain 的详细指南。**

---

## E.1 从 Legacy Chains 迁移到 LCEL

### 迁移必要性

**为什么需要迁移？**

1. **Legacy Chains 已废弃**：从 LangChain 0.2 开始，`LLMChain`、`ConversationChain` 等传统链被标记为废弃
2. **性能提升**：LCEL 支持流式、批处理、并行执行等高级特性
3. **类型安全**：更好的 IDE 支持与类型推断
4. **可组合性**：更灵活的链组合与复用

### 迁移对照表

#### 1. LLMChain → LCEL

**❌ Legacy 写法**：
```python
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

prompt = PromptTemplate.from_template("Translate {text} to French")
llm = ChatOpenAI()
chain = LLMChain(llm=llm, prompt=prompt)

result = chain.run(text="Hello")
```

**✅ LCEL 写法**：
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Translate {text} to French")
llm = ChatOpenAI()
chain = prompt | llm | StrOutputParser()

result = chain.invoke({"text": "Hello"})
```

**关键差异**：
- 使用 `|` 操作符替代 `LLMChain`
- `run()` → `invoke()`
- 显式添加 `StrOutputParser()` 提取文本

---

#### 2. ConversationChain → LCEL + Memory

**❌ Legacy 写法**：
```python
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

response = conversation.predict(input="Hi, I'm Alice")
```

**✅ LCEL 写法**：
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 1. 定义提示
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

# 2. 构建链
chain = prompt | llm | StrOutputParser()

# 3. 添加记忆管理
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 4. 使用
config = {"configurable": {"session_id": "user123"}}
response = chain_with_history.invoke(
    {"input": "Hi, I'm Alice"},
    config=config
)
```

---

#### 3. SimpleSequentialChain → LCEL Pipe

**❌ Legacy 写法**：
```python
from langchain.chains import SimpleSequentialChain

chain1 = LLMChain(llm=llm, prompt=prompt1)
chain2 = LLMChain(llm=llm, prompt=prompt2)

overall_chain = SimpleSequentialChain(chains=[chain1, chain2])
result = overall_chain.run("Input text")
```

**✅ LCEL 写法**：
```python
chain1 = prompt1 | llm | StrOutputParser()
chain2 = prompt2 | llm | StrOutputParser()

overall_chain = chain1 | chain2  # 直接 pipe
result = overall_chain.invoke({"input": "Input text"})
```

---

#### 4. TransformChain → RunnableLambda

**❌ Legacy 写法**：
```python
from langchain.chains import TransformChain

def transform_func(inputs: dict) -> dict:
    return {"output": inputs["text"].upper()}

transform_chain = TransformChain(
    input_variables=["text"],
    output_variables=["output"],
    transform=transform_func
)
```

**✅ LCEL 写法**：
```python
from langchain_core.runnables import RunnableLambda

transform_chain = RunnableLambda(lambda x: x["text"].upper())

# 或使用装饰器
from langchain_core.runnables import chain

@chain
def transform_chain(x: dict) -> str:
    return x["text"].upper()
```

---

#### 5. RouterChain → RunnableBranch

**❌ Legacy 写法**：
```python
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser

destinations = [
    {"name": "physics", "description": "Physics questions"},
    {"name": "math", "description": "Math questions"}
]

router_chain = LLMRouterChain.from_llm(llm, router_prompt)
multi_prompt_chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains={"physics": physics_chain, "math": math_chain},
    default_chain=default_chain
)
```

**✅ LCEL 写法**：
```python
from langchain_core.runnables import RunnableBranch

def route(x: dict) -> str:
    # 简单分类逻辑
    if "physics" in x["question"].lower():
        return "physics"
    elif "math" in x["question"].lower():
        return "math"
    return "default"

# 使用 LLM 路由（更智能）
router = (
    prompt_router
    | llm
    | StrOutputParser()
    | RunnableLambda(lambda x: x.strip().lower())
)

branch = RunnableBranch(
    (lambda x: router.invoke(x) == "physics", physics_chain),
    (lambda x: router.invoke(x) == "math", math_chain),
    default_chain
)
```

---

### 完整迁移检查清单

- [ ] 替换所有 `LLMChain` 为 LCEL pipe
- [ ] 替换 `ConversationChain` 为 `RunnableWithMessageHistory`
- [ ] 替换 `SimpleSequentialChain` 为链式 `|` 组合
- [ ] 替换 `TransformChain` 为 `RunnableLambda`
- [ ] 替换 `RouterChain` 为 `RunnableBranch`
- [ ] 所有 `chain.run()` 改为 `chain.invoke()`
- [ ] 添加显式的 `StrOutputParser()` 或其他 parser
- [ ] 测试所有迁移后的链，确保行为一致

---

## E.2 从 LangChain 0.1 升级到 0.3

### 主要变更

#### 1. 包结构重组

**LangChain 0.1**：
```python
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
```

**LangChain 0.3**：
```python
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
```

**迁移规则**：
- 提供商特定组件移到独立包（`langchain-openai`、`langchain-anthropic` 等）
- 安装：`pip install langchain-openai langchain-chroma`

---

#### 2. Pydantic v2 升级

**LangChain 0.3 要求 Pydantic v2**（`pydantic>=2.0`）

**常见问题**：

**问题 1：`__fields__` 属性不存在**

```python
# ❌ Pydantic v1
model.schema()["properties"]
model.__fields__

# ✅ Pydantic v2
model.model_json_schema()["properties"]
model.model_fields
```

**问题 2：`Config` 类语法变化**

```python
# ❌ Pydantic v1
class MyModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True

# ✅ Pydantic v2
from pydantic import ConfigDict

class MyModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
```

---

#### 3. Callbacks 系统更新

**LangChain 0.1**：
```python
chain.run(input, callbacks=[handler])
```

**LangChain 0.3**：
```python
chain.invoke(input, config={"callbacks": [handler]})
```

**迁移**：
- 所有 `callbacks` 参数移到 `config` 字典中

---

#### 4. Memory 系统变化

**LangChain 0.1**：
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory)
```

**LangChain 0.3**（推荐使用 LangGraph 管理状态）：
```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

checkpointer = MemorySaver()
graph = StateGraph(State)
# ... 添加节点
app = graph.compile(checkpointer=checkpointer)
```

---

### 升级步骤

#### Step 1: 更新依赖

```bash
# 卸载旧版本
pip uninstall langchain langchain-community

# 安装新版本
pip install langchain==0.3.0 langchain-core langchain-community

# 安装提供商包
pip install langchain-openai langchain-anthropic langchain-chroma

# 升级 Pydantic
pip install pydantic==2.6.0
```

#### Step 2: 修复导入语句

运行自动化工具：

```bash
# 使用 langchain-cli（如果可用）
langchain migrate

# 或手动查找替换
grep -r "from langchain.llms import OpenAI" . -l | xargs sed -i 's/from langchain.llms import OpenAI/from langchain_openai import OpenAI/g'
```

#### Step 3: 运行测试

```bash
pytest tests/ -v
```

#### Step 4: 逐步迁移（建议）

- **阶段 1**：仅升级 `langchain-core`，保持其他不变
- **阶段 2**：升级提供商包（`langchain-openai` 等）
- **阶段 3**：迁移到 LCEL
- **阶段 4**：迁移到 LangGraph（如需复杂状态管理）

---

### 兼容性矩阵

| 组件 | 0.1 | 0.2 | 0.3 | 备注 |
|------|-----|-----|-----|------|
| LCEL | ⚠️ | ✅ | ✅ | 0.1 部分支持 |
| LangGraph | ❌ | ⚠️ | ✅ | 0.2 实验性 |
| Legacy Chains | ✅ | ⚠️ | ❌ | 0.3 完全移除 |
| Pydantic v1 | ✅ | ✅ | ❌ | 0.3 需 v2 |
| Pydantic v2 | ❌ | ✅ | ✅ |  |

---

## E.3 从 LlamaIndex 迁移到 LangChain

### 核心差异

| 维度 | LlamaIndex | LangChain |
|------|------------|-----------|
| **核心定位** | RAG 优先 | 通用 LLM 框架 |
| **索引抽象** | Index (GPTVectorStoreIndex 等) | VectorStore + Retriever |
| **查询接口** | `index.as_query_engine()` | `retriever.get_relevant_documents()` |
| **链组合** | Query Engine Pipeline | LCEL |
| **Agent** | ReActAgent | LangGraph / create_react_agent |

### 典型场景迁移

#### 场景 1：向量索引 + 查询

**LlamaIndex 写法**：
```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("What is LangChain?")
print(response)
```

**LangChain 等价写法**：
```python
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 加载文档
loader = DirectoryLoader("data", glob="**/*.txt")
documents = loader.load()

# 2. 分块
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(documents)

# 3. 创建向量库
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever()

# 4. 构建 RAG 链
prompt = ChatPromptTemplate.from_template("""
Answer based on context:

Context: {context}

Question: {question}
""")

llm = ChatOpenAI()

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("What is LangChain?")
print(response)
```

---

#### 场景 2：Chat Engine（对话式 RAG）

**LlamaIndex 写法**：
```python
chat_engine = index.as_chat_engine()
response = chat_engine.chat("Hello")
```

**LangChain 等价写法**：
```python
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on context: {context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}")
])

chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

store = {}
def get_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chat_chain = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="question",
    history_messages_key="history"
)

config = {"configurable": {"session_id": "user1"}}
response = chat_chain.invoke({"question": "Hello"}, config=config)
```

---

### 迁移检查清单

- [ ] 替换 `VectorStoreIndex` 为 `Chroma` / `Pinecone` / `FAISS`
- [ ] 替换 `as_query_engine()` 为 Retriever + RAG 链
- [ ] 替换 `SimpleDirectoryReader` 为 `DirectoryLoader`
- [ ] 使用 `RecursiveCharacterTextSplitter` 替代 LlamaIndex 的默认分块
- [ ] 使用 `RunnableWithMessageHistory` 替代 `as_chat_engine()`
- [ ] 调整提示模板格式

---

## E.4 从 Haystack 迁移到 LangChain

### 核心差异

| 维度 | Haystack | LangChain |
|------|----------|-----------|
| **核心抽象** | Pipeline + Nodes | Runnable + Chains |
| **文档处理** | Preprocessor Node | Text Splitter |
| **检索** | Retriever Node | Retriever |
| **生成** | PromptNode | LLM + Prompt |
| **组合方式** | `pipeline.add_node()` | LCEL `\|` 操作符 |

### 典型迁移

**Haystack 写法**：
```python
from haystack import Pipeline
from haystack.nodes import BM25Retriever, FARMReader

pipeline = Pipeline()
pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
pipeline.add_node(component=reader, name="Reader", inputs=["Retriever"])

result = pipeline.run(query="What is LangChain?")
```

**LangChain 等价写法**：
```python
from langchain.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI

retriever = BM25Retriever.from_documents(documents)
llm = ChatOpenAI()

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke("What is LangChain?")
```

---

## E.5 从 AutoGen 迁移到 LangChain

### 核心差异

| 维度 | AutoGen | LangChain |
|------|---------|-----------|
| **Agent 定义** | `AssistantAgent` / `UserProxyAgent` | `create_react_agent` |
| **对话模式** | 双向消息流 | 单向调用链 |
| **代码执行** | 内置代码执行器 | 需集成 `PythonREPLTool` |
| **人机协作** | `UserProxyAgent` | LangGraph `interrupt_before` |

### 典型迁移

**AutoGen 写法**：
```python
from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent("assistant", llm_config=llm_config)
user_proxy = UserProxyAgent("user", code_execution_config={"work_dir": "coding"})

user_proxy.initiate_chat(assistant, message="Plot a chart of stock prices")
```

**LangChain 等价写法（使用 LangGraph）**：
```python
from langgraph.prebuilt import create_react_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import ChatOpenAI

tools = [PythonREPLTool()]
llm = ChatOpenAI()

agent = create_react_agent(llm, tools)

result = agent.invoke({"messages": [("user", "Plot a chart of stock prices")]})
```

---

## E.6 版本迁移常见问题

### Q1: 升级后 Import Error

**错误**：
```
ImportError: cannot import name 'OpenAI' from 'langchain.llms'
```

**解决方案**：
```bash
pip install langchain-openai
```

```python
from langchain_openai import OpenAI  # ✅
```

---

### Q2: Pydantic 版本冲突

**错误**：
```
pydantic.errors.PydanticUserError: `__fields__` attribute not found
```

**解决方案**：
```bash
pip install pydantic==2.6.0
```

---

### Q3: Callbacks 不生效

**错误**：
```python
chain.invoke(input, callbacks=[handler])  # ❌ 不再支持
```

**解决方案**：
```python
chain.invoke(input, config={"callbacks": [handler]})  # ✅
```

---

### Q4: LangSmith Tracing 丢失

**原因**：环境变量未正确设置

**解决方案**：
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=lsv2_...
```

---

## E.7 自动化迁移工具

### LangChain CLI（实验性）

```bash
# 安装 CLI
pip install langchain-cli

# 运行迁移脚本
langchain migrate --from 0.1 --to 0.3 --path ./src

# 生成迁移报告
langchain migrate --report
```

### 手动迁移脚本

```python
# migrate.py
import re
from pathlib import Path

# 查找所有 Python 文件
files = Path("./src").rglob("*.py")

for file in files:
    content = file.read_text()
    
    # 替换导入语句
    content = re.sub(
        r"from langchain\.llms import OpenAI",
        "from langchain_openai import OpenAI",
        content
    )
    
    content = re.sub(
        r"from langchain\.chat_models import ChatOpenAI",
        "from langchain_openai import ChatOpenAI",
        content
    )
    
    # 替换方法调用
    content = re.sub(
        r"\.run\(",
        ".invoke(",
        content
    )
    
    file.write_text(content)
    print(f"Migrated: {file}")
```

---

## E.8 迁移时间估算

| 项目规模 | Legacy Chains → LCEL | 0.1 → 0.3 | LlamaIndex → LangChain |
|----------|----------------------|-----------|------------------------|
| **小型**（< 10 链） | 2-4 小时 | 1-2 小时 | 4-8 小时 |
| **中型**（10-50 链） | 1-2 天 | 0.5-1 天 | 2-3 天 |
| **大型**（> 50 链） | 3-5 天 | 1-2 天 | 1-2 周 |

**建议策略**：
- 优先迁移核心功能
- 使用分支进行迁移（保留旧版本代码）
- 逐步迁移，每次迁移一个模块并充分测试

---

**迁移过程中遇到问题？**
- 查看官方迁移指南：https://python.langchain.com/docs/guides/migrating
- 在 Discord 社区求助：https://discord.gg/langchain
- 提交 GitHub Issue：https://github.com/langchain-ai/langchain/issues

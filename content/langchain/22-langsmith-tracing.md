# Chapter 22: LangSmith Tracing 基础

## 本章概览

当 LangChain 应用变得复杂时，调试与性能优化成为最大挑战：为什么这个链失败了？哪个步骤最慢？Token 消耗在哪里？LangSmith 的 Tracing 系统通过**完整的执行追踪**和**可视化分析**，让复杂链的内部运行过程一目了然。本章将深入学习 LangSmith Tracing 的配置、结构、分析方法和自定义技术。

**本章重点**：
- LangSmith Tracing 的核心价值与应用场景
- Tracing 配置与项目管理
- Trace 结构解析（Run、Span、嵌套关系）
- Trace 查看与性能分析
- 自定义 Tracing 与 Metadata

---

## 22.1 为什么需要 LangSmith？

### 22.1.1 复杂链的调试困境

随着 LangChain 应用变得复杂，传统调试方法失效：

**问题示例**：一个 RAG 应用失败了

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 复杂的 RAG 链
vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    chain_type="stuff",
    retriever=retriever,
)

# 执行失败！但无法知道哪里出错了
result = qa_chain.invoke("What is LangChain?")
# Error: ... （错误信息模糊）
```

**调试难点**：
1. ❓ **执行路径不透明**：无法看到内部调用链
2. ❓ **错误定位困难**：不知道在哪一步失败
3. ❓ **性能瓶颈未知**：哪个步骤最慢？
4. ❓ **Token 消耗不明**：钱花在哪里了？
5. ❓ **输入输出不可见**：每一步的中间结果是什么？

### 22.1.2 生产监控需求

生产环境中，你需要回答这些问题：

| 问题 | 传统方法 | LangSmith 方案 |
|------|---------|----------------|
| 系统是否正常运行？ | 手动日志查看 | 实时 Dashboard |
| 哪些请求失败了？ | grep 错误日志 | 自动失败追踪 |
| 平均延迟多少？ | 自己写指标收集 | 内置性能分析 |
| Token 成本趋势？ | 手动计算 | 自动成本追踪 |
| 用户体验如何？ | 用户反馈 | Feedback 机制 |

### 22.1.3 LangSmith 核心价值

LangSmith 提供**三位一体**的解决方案：

```
1. 🔍 Tracing（追踪）
   ├─ 完整的执行过程可视化
   ├─ 嵌套调用链展示
   └─ 输入输出完整记录

2. 📊 Evaluation（评估）
   ├─ 数据集管理
   ├─ 批量评估
   └─ 多维度指标

3. 📈 Monitoring（监控）
   ├─ 生产环境实时追踪
   ├─ 告警与异常检测
   └─ 成本与性能分析
```

**与其他工具对比**：

| 工具 | 追踪 | 评估 | 监控 | LangChain 集成 |
|------|------|------|------|----------------|
| LangSmith | ✅ 原生 | ✅ 内置 | ✅ 实时 | ✅ 无缝 |
| Weights & Biases | ⚠️ 通用 | ✅ ML 评估 | ✅ 实验追踪 | ⚠️ 需适配 |
| MLflow | ⚠️ 通用 | ⚠️ ML 指标 | ✅ 模型管理 | ⚠️ 需适配 |
| 自建日志 | ❌ 手动 | ❌ 无 | ⚠️ 需自建 | ⚠️ 复杂 |

---

## 22.2 Tracing 配置

### 22.2.1 环境变量配置

最简单的启用方式：设置环境变量

```bash
# 1. 启用 Tracing V2（必需）
export LANGCHAIN_TRACING_V2=true

# 2. 设置 API Key（必需）
export LANGCHAIN_API_KEY="lsv2_pt_..."  # 从 https://smith.langchain.com 获取

# 3. 设置项目名称（可选，默认为 "default"）
export LANGCHAIN_PROJECT="my-rag-app"

# 4. 设置 Endpoint（可选，默认为官方服务器）
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
```

**验证配置**：

```python
import os

print("Tracing Enabled:", os.getenv("LANGCHAIN_TRACING_V2"))
print("API Key:", os.getenv("LANGCHAIN_API_KEY")[:20] + "...")
print("Project:", os.getenv("LANGCHAIN_PROJECT"))
```

### 22.2.2 代码中动态配置

更灵活的方式：在代码中控制

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 方法 1：全局启用
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_pt_..."
os.environ["LANGCHAIN_PROJECT"] = "debug-session"

# 方法 2：仅对特定链启用（推荐）
from langchain_core.tracers import LangChainTracer

tracer = LangChainTracer(project_name="experiment-v2")

chain = (
    ChatPromptTemplate.from_template("Tell me a joke about {topic}")
    | ChatOpenAI(model="gpt-4")
)

# 仅此次调用启用 tracing
result = chain.invoke(
    {"topic": "AI"},
    config={"callbacks": [tracer]}
)

# 其他调用不会被追踪
result2 = chain.invoke({"topic": "Python"})  # 不追踪
```

### 22.2.3 项目管理最佳实践

**项目命名策略**：

```python
# 按环境区分
os.environ["LANGCHAIN_PROJECT"] = "production"  # 生产
os.environ["LANGCHAIN_PROJECT"] = "staging"     # 测试
os.environ["LANGCHAIN_PROJECT"] = "dev-alice"   # 开发

# 按功能区分
os.environ["LANGCHAIN_PROJECT"] = "rag-customer-support"
os.environ["LANGCHAIN_PROJECT"] = "agent-code-gen"
os.environ["LANGCHAIN_PROJECT"] = "chatbot-hr"

# 按实验区分
os.environ["LANGCHAIN_PROJECT"] = "exp-gpt4-vs-claude"
os.environ["LANGCHAIN_PROJECT"] = "exp-prompt-v3"
```

**动态切换项目**：

```python
from contextlib import contextmanager

@contextmanager
def langsmith_project(project_name: str):
    """临时切换 LangSmith 项目"""
    old_project = os.getenv("LANGCHAIN_PROJECT")
    os.environ["LANGCHAIN_PROJECT"] = project_name
    try:
        yield
    finally:
        if old_project:
            os.environ["LANGCHAIN_PROJECT"] = old_project
        else:
            os.environ.pop("LANGCHAIN_PROJECT", None)

# 使用示例
with langsmith_project("experiment-2024-01"):
    result = chain.invoke({"topic": "LLM"})
    # 此调用会记录到 "experiment-2024-01" 项目
```

### 22.2.4 禁用 Tracing（性能优化）

生产环境中，可能需要选择性禁用：

```python
# 全局禁用
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# 或删除环境变量
os.environ.pop("LANGCHAIN_TRACING_V2", None)

# 对单次调用禁用
result = chain.invoke(
    {"topic": "AI"},
    config={"callbacks": []}  # 空 callbacks 列表
)
```

---

## 22.3 Trace 结构解析

### 22.3.1 Run（运行）：基本单位

每次 LangChain 组件执行都会生成一个 **Run**：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
result = llm.invoke("Hello!")
# ↑ 这会生成一个 "llm" 类型的 Run
```

**Run 的核心属性**：

```python
{
    "id": "run-abc123...",           # 唯一标识
    "name": "ChatOpenAI",             # 组件名称
    "run_type": "llm",                # 类型：llm/chain/tool/retriever
    "start_time": "2024-01-20T10:30:00Z",
    "end_time": "2024-01-20T10:30:02Z",
    "inputs": {"messages": [...]},    # 输入数据
    "outputs": {"generations": [...]}, # 输出数据
    "error": null,                    # 错误信息（如果失败）
    "extra": {
        "metadata": {...},            # 自定义元数据
        "tags": ["gpt-4", "chat"],    # 标签
    },
    "parent_run_id": null,            # 父 Run ID（嵌套时有值）
    "child_runs": [],                 # 子 Run 列表
}
```

### 22.3.2 Run 类型详解

LangSmith 支持 5 种主要 Run 类型：

<div data-component="TraceTreeVisualizer"></div>

**1. Chain Run**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

chain = (
    ChatPromptTemplate.from_template("Translate to {language}: {text}")
    | ChatOpenAI()
)

result = chain.invoke({"language": "French", "text": "Hello"})
# ↑ 生成一个 Chain Run，包含 2 个子 Run：
#   ├─ PromptTemplate Run
#   └─ ChatOpenAI Run
```

**2. LLM Run**

```python
llm = ChatOpenAI(model="gpt-4")
result = llm.invoke("What is 2+2?")
# ↑ 生成一个 LLM Run，记录：
#   - 模型名称（gpt-4）
#   - Token 使用量
#   - 延迟
```

**3. Tool Run**

```python
from langchain.tools import Tool

def search(query: str) -> str:
    return f"Results for: {query}"

search_tool = Tool(
    name="search",
    func=search,
    description="Search the web"
)

result = search_tool.invoke("LangChain")
# ↑ 生成一个 Tool Run
```

**4. Retriever Run**

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

docs = retriever.invoke("What is RAG?")
# ↑ 生成一个 Retriever Run，记录：
#   - 查询内容
#   - 检索到的文档数量
#   - 相似度分数
```

**5. Embedding Run**

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectors = embeddings.embed_query("Hello world")
# ↑ 生成一个 Embedding Run
```

### 22.3.3 Span（跨度）与嵌套结构

复杂链会形成**树形嵌套结构**：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 定义链
prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
llm = ChatOpenAI(model="gpt-4")
parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({"topic": "LangSmith"})
```

**生成的 Trace 树**：

```
RunnableSequence (Chain)              ← 根 Run
├─ ChatPromptTemplate (Prompt)        ← 子 Run 1
├─ ChatOpenAI (LLM)                   ← 子 Run 2
│  └─ OpenAI API Call                 ← 孙 Run
└─ StrOutputParser (Parser)           ← 子 Run 3
```

**Span 的时间关系**：

```
时间线：
0ms ─────────────────────────────────────────→ 2000ms
│
├─ Prompt [0-10ms]        ▓
├─ LLM [10-1900ms]        ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
│  └─ API Call [50-1850ms]  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
└─ Parser [1900-2000ms]                    ▓
```

### 22.3.4 Parent-Child 关系

通过 `parent_run_id` 和 `child_runs` 构建调用链：

```python
# 示例：一个 RAG 链的 Run 树
{
    "id": "run-root",
    "name": "RetrievalQA",
    "run_type": "chain",
    "child_runs": [
        {
            "id": "run-retriever",
            "name": "VectorStoreRetriever",
            "run_type": "retriever",
            "parent_run_id": "run-root",
            "child_runs": [
                {
                    "id": "run-embedding",
                    "name": "OpenAIEmbeddings",
                    "run_type": "embedding",
                    "parent_run_id": "run-retriever"
                }
            ]
        },
        {
            "id": "run-llm",
            "name": "ChatOpenAI",
            "run_type": "llm",
            "parent_run_id": "run-root"
        }
    ]
}
```

---

## 22.4 Trace 查看与分析

### 22.4.1 LangSmith UI 导航

访问 [https://smith.langchain.com](https://smith.langchain.com) 后：

**1. Projects 页面**
- 查看所有项目列表
- 切换活动项目
- 查看项目统计（总 Run 数、成功率、平均延迟）

**2. Runs 页面**
- 按时间、状态、Run 类型过滤
- 搜索特定 Run（按 ID、名称、Tag）
- 查看 Run 列表（时间、延迟、Token、状态）

**3. Run 详情页**
- **Overview**：Run 基本信息
- **Inputs/Outputs**：完整输入输出
- **Metadata**：自定义元数据
- **Timeline**：时间线视图
- **Tree**：树形结构视图

### 22.4.2 时间线视图（Timeline）

<div data-component="SpanTimelineChart"></div>

时间线视图展示**每个 Span 的执行时间**：

**示例分析**：

```
Chain Execution Timeline (Total: 3.2s)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

0.0s ────────────────────────────────→ 3.2s

Retriever      [0.0s - 1.2s]  ████████
  ├─ Embedding [0.1s - 0.6s]   ████
  └─ Search    [0.6s - 1.2s]       ████

LLM Call       [1.2s - 3.0s]           ██████████
  └─ API Wait  [1.3s - 2.9s]            █████████

Parser         [3.0s - 3.2s]                    █
```

**性能瓶颈识别**：
- ⚠️ LLM Call 占用 56% 时间（1.8s / 3.2s）
- ✅ Retriever 和 Parser 较快
- 💡 优化方向：考虑 Streaming 或异步调用

### 22.4.3 Tree 视图（树形结构）

树形视图展示**父子关系与数据流**：

```
📦 RetrievalQA Chain
├─ 📥 Input: "What is LangChain?"
├─ 🔍 VectorStoreRetriever
│  ├─ 📥 Input: "What is LangChain?"
│  ├─ 🧮 OpenAIEmbeddings
│  │  ├─ 📥 Input: "What is LangChain?"
│  │  └─ 📤 Output: [0.123, -0.456, ...]
│  ├─ 🔎 Chroma Search
│  └─ 📤 Output: [Document(page_content="LangChain is..."), ...]
├─ 🤖 ChatOpenAI
│  ├─ 📥 Input: {"context": "...", "question": "..."}
│  └─ 📤 Output: "LangChain is a framework for..."
└─ 📤 Final Output: "LangChain is a framework for..."
```

**调试价值**：
- 查看每一步的实际输入输出
- 验证数据是否按预期流动
- 定位错误发生的具体步骤

### 22.4.4 Token 消耗分析

<div data-component="TokenUsageBreakdown"></div>

LangSmith 自动统计 Token 使用量：

**Token 统计示例**：

```python
Total Tokens: 1,234
├─ Prompt Tokens: 856
│  ├─ System Prompt: 120
│  ├─ User Input: 36
│  └─ Retrieved Context: 700  ← 大头！
└─ Completion Tokens: 378
```

**成本计算**：

```
GPT-4 Pricing (2024-01):
- Prompt: $0.03 / 1K tokens
- Completion: $0.06 / 1K tokens

Cost = (856 * 0.03 + 378 * 0.06) / 1000
     = $0.0257 + $0.0227
     = $0.0484 per request
```

**优化建议**：
- 🔧 缩短 System Prompt
- 🔧 减少检索文档数量（k=3 → k=2）
- 🔧 使用 GPT-3.5-Turbo 替代 GPT-4（测试环境）

### 22.4.5 延迟热点识别

**按组件类型统计延迟**：

```python
Component Latency Breakdown:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Retriever:    1.2s  (37.5%)  ████████
LLM Call:     1.8s  (56.3%)  ████████████
Parser:       0.2s  (6.2%)   ██
─────────────────────────────
Total:        3.2s  (100%)
```

**性能优化策略**：

| 瓶颈 | 优化方案 |
|------|----------|
| Retriever 慢 | 1. 使用更快的向量数据库（FAISS）<br>2. 减少检索数量<br>3. 添加缓存层 |
| LLM 慢 | 1. Streaming（用户感知更快）<br>2. 使用更快的模型（GPT-4 Turbo）<br>3. 异步调用<br>4. 批处理 |
| Embedding 慢 | 1. 批量嵌入<br>2. 缓存常见查询 |

---

## 22.5 自定义 Tracing

### 22.5.1 @traceable 装饰器

为自定义函数添加 Tracing：

```python
from langsmith import traceable

@traceable(run_type="custom", name="DataProcessor")
def process_data(data: dict) -> dict:
    """自定义数据处理函数"""
    # 复杂的业务逻辑
    processed = {k: v.upper() for k, v in data.items()}
    return processed

# 调用时自动追踪
result = process_data({"name": "Alice", "city": "NYC"})
```

**生成的 Trace**：

```
Custom Run
├─ Name: DataProcessor
├─ Run Type: custom
├─ Input: {"name": "Alice", "city": "NYC"}
├─ Output: {"name": "ALICE", "city": "NYC"}
└─ Duration: 0.002s
```

### 22.5.2 嵌套自定义 Trace

```python
@traceable(name="Step1-FetchData")
def fetch_data(user_id: str) -> dict:
    return {"user_id": user_id, "name": "Alice"}

@traceable(name="Step2-ValidateData")
def validate_data(data: dict) -> bool:
    return "name" in data and len(data["name"]) > 0

@traceable(name="MainPipeline")
def main_pipeline(user_id: str) -> str:
    data = fetch_data(user_id)      # ← 子 Trace
    valid = validate_data(data)      # ← 子 Trace
    
    if valid:
        return f"Welcome {data['name']}!"
    else:
        return "Invalid data"

# 执行生成嵌套 Trace
result = main_pipeline("user123")
```

**Trace 树**：

```
MainPipeline
├─ Input: "user123"
├─ Step1-FetchData
│  ├─ Input: "user123"
│  └─ Output: {"user_id": "user123", "name": "Alice"}
├─ Step2-ValidateData
│  ├─ Input: {"user_id": "user123", "name": "Alice"}
│  └─ Output: true
└─ Output: "Welcome Alice!"
```

### 22.5.3 添加 Metadata 与 Tags

```python
from langsmith import traceable

@traceable(
    name="UserQuery",
    metadata={"version": "v2.1", "environment": "production"},
    tags=["user-facing", "critical"]
)
def handle_user_query(query: str) -> str:
    # 处理逻辑
    return f"Answer to: {query}"

result = handle_user_query("What is LangChain?")
```

**动态 Metadata**：

```python
from langsmith import Client
import uuid

@traceable
def process_request(request: dict) -> dict:
    # 获取当前 Run
    client = Client()
    run_id = uuid.uuid4()  # 实际会自动生成
    
    # 添加动态元数据
    client.update_run(
        run_id=run_id,
        extra={
            "metadata": {
                "user_id": request.get("user_id"),
                "request_ip": request.get("ip"),
                "timestamp": "2024-01-20T10:30:00Z"
            }
        }
    )
    
    return {"status": "success"}
```

### 22.5.4 错误追踪

```python
@traceable(name="RiskyOperation")
def risky_operation(data: dict) -> dict:
    try:
        if "required_field" not in data:
            raise ValueError("Missing required field")
        
        result = {"processed": data["required_field"].upper()}
        return result
    
    except Exception as e:
        # LangSmith 自动记录异常
        raise

# 失败的调用会在 Trace 中显示错误
try:
    risky_operation({})  # 缺少 required_field
except ValueError:
    pass
```

**Trace 中的错误信息**：

```
RiskyOperation (FAILED)
├─ Status: Error
├─ Error: ValueError("Missing required field")
├─ Stack Trace:
│  File "example.py", line 5, in risky_operation
│    raise ValueError("Missing required field")
└─ Duration: 0.001s
```

### 22.5.5 LangChain 集成的高级用法

**为链添加自定义名称**：

```python
from langchain_core.runnables import RunnableConfig

chain = prompt | llm | parser

result = chain.invoke(
    {"topic": "AI"},
    config=RunnableConfig(
        run_name="TranslationChain-v2",  # 自定义 Run 名称
        tags=["translation", "v2"],       # 添加标签
        metadata={"user": "alice"}        # 添加元数据
    )
)
```

**批量操作的 Tracing**：

```python
inputs = [
    {"topic": "AI"},
    {"topic": "ML"},
    {"topic": "LLM"}
]

# 每个输入生成独立的 Trace
results = chain.batch(
    inputs,
    config={"tags": ["batch-job", "experiment-1"]}
)
```

---

## 22.6 实战案例：调试复杂 RAG 链

### 22.6.1 问题场景

一个客服 RAG 系统响应缓慢（>5秒）且答案质量差。

**原始代码**：

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# 初始化组件
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory="./customer_kb",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}  # ← 可能的问题？
)

template = """You are a customer service agent. 
Use the following context to answer the question. 
If you don't know, say you don't know.

Context: {context}

Question: {question}

Detailed Answer:"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4", temperature=0),  # ← GPT-4 很慢
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# 启用 Tracing
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "debug-slow-rag"

# 执行查询
result = qa_chain.invoke("How do I reset my password?")
```

### 22.6.2 Trace 分析

查看 LangSmith Trace 后发现：

**时间线分析**：

```
Total Time: 5.3s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Retriever        [0.0s - 2.1s]  ████████████
  ├─ Embedding   [0.0s - 0.3s]   ██
  └─ Search      [0.3s - 2.1s]     ██████████  ← 慢！

LLM Call         [2.1s - 5.2s]               ██████████████  ← 很慢！
  └─ API Wait    [2.2s - 5.1s]                █████████████

Parser           [5.2s - 5.3s]                              █
```

**Token 分析**：

```
Prompt Tokens: 3,456  ← 异常高！
├─ System Prompt: 120
├─ User Question: 20
└─ Context: 3,316  ← 10 个文档太多了！

Completion Tokens: 287
```

**问题诊断**：
1. 🔴 检索了 10 个文档（k=10），导致 Context 过长
2. 🔴 使用 GPT-4，延迟较高
3. 🔴 Chroma 搜索较慢（可能索引问题）

### 22.6.3 优化方案

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# 优化 1：切换到 FAISS（更快）
vectorstore = FAISS.load_local(
    "customer_kb_faiss",
    embeddings,
    allow_dangerous_deserialization=True
)

# 优化 2：减少检索数量
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # 10 → 3
)

# 优化 3：使用 GPT-3.5-Turbo（更快更便宜）
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# 重新测试
result = qa_chain.invoke("How do I reset my password?")
```

**优化后的 Trace**：

```
Total Time: 1.2s  (原 5.3s，提升 77%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Retriever        [0.0s - 0.3s]  ████
  ├─ Embedding   [0.0s - 0.1s]   █
  └─ Search      [0.1s - 0.3s]    ██

LLM Call         [0.3s - 1.1s]      ████████
  └─ API Wait    [0.4s - 1.0s]       ██████

Parser           [1.1s - 1.2s]            █
```

**Token 分析**：

```
Prompt Tokens: 856  (原 3,456，减少 75%)
├─ System Prompt: 120
├─ User Question: 20
└─ Context: 716  (3 个文档而非 10 个)

Completion Tokens: 198
```

**成本对比**：

```
Before: $0.0484 per request (GPT-4 + 3,743 tokens)
After:  $0.0016 per request (GPT-3.5 + 1,054 tokens)
节省:   97% 成本
```

---

## 22.7 最佳实践

### 22.7.1 何时启用 Tracing？

| 场景 | 是否启用 | 原因 |
|------|----------|------|
| 开发调试 | ✅ 始终启用 | 快速定位问题 |
| 单元测试 | ⚠️ 选择性 | 避免大量无用 Trace |
| 集成测试 | ✅ 启用 | 验证完整流程 |
| 生产环境 | ⚠️ 采样启用 | 避免性能开销，按 1-10% 采样 |
| 性能基准 | ❌ 禁用 | 避免 Tracing 本身的开销 |

**生产采样示例**：

```python
import random

def should_trace() -> bool:
    """10% 采样"""
    return random.random() < 0.1

if should_trace():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

result = chain.invoke(input_data)
```

### 22.7.2 项目命名规范

```python
# ✅ 好的命名
"production-customer-chatbot"
"staging-rag-v2"
"dev-alice-experiment-prompt-v3"

# ❌ 不好的命名
"test"
"my-project"
"aaa"
```

### 22.7.3 标签使用策略

```python
# 为不同类型的请求打标签
tags = []

if request.get("user_type") == "premium":
    tags.append("premium-user")

if request.get("query_type") == "complex":
    tags.append("complex-query")

tags.append(f"version-{app_version}")

result = chain.invoke(
    input_data,
    config={"tags": tags}
)
```

### 22.7.4 敏感信息处理

```python
from langsmith import traceable

@traceable(
    name="ProcessUserData",
    # 隐藏敏感字段
    hide_inputs=["password", "credit_card"],
    hide_outputs=["api_key"]
)
def process_user_data(data: dict) -> dict:
    # 处理包含敏感信息的数据
    return {
        "status": "success",
        "api_key": "sk-..."  # 会被隐藏
    }
```

---

## 本章总结

**核心收获**：

1. ✅ **LangSmith Tracing 是复杂链调试的必备工具**
   - 可视化执行过程
   - 定位性能瓶颈
   - 追踪 Token 消耗

2. ✅ **Trace 结构理解**
   - Run：基本单位（Chain、LLM、Tool、Retriever）
   - Span：时间维度的执行片段
   - 嵌套关系：父子 Run 树

3. ✅ **三种视图互补使用**
   - Timeline：找性能瓶颈
   - Tree：查数据流
   - Metadata：看业务信息

4. ✅ **自定义 Tracing 扩展能力**
   - @traceable 装饰器
   - Metadata 与 Tags
   - 错误追踪

5. ✅ **生产环境最佳实践**
   - 采样策略（1-10%）
   - 项目命名规范
   - 敏感信息保护

**下一章预告**：
Chapter 23 将深入学习 **LangSmith 评估系统**，掌握数据集管理、批量评估、自定义 Evaluator、LLM-as-Judge 等技术，建立 LLM 应用的质量保障体系。

---

## 练习题

### 基础练习

1. **启用 Tracing**：为现有的聊天机器人项目启用 LangSmith Tracing，观察执行流程。

2. **性能分析**：找出你的应用中最慢的 3 个步骤，使用 Timeline 视图分析。

3. **Token 优化**：统计你的应用的平均 Token 使用量，尝试优化提示词减少消耗。

### 进阶练习

4. **自定义 Trace**：为自定义的数据处理函数添加 @traceable，记录关键业务指标。

5. **错误调试**：故意引入一个错误（如缺少环境变量），观察 LangSmith 如何记录错误信息。

6. **多项目管理**：创建 3 个不同的项目（dev、staging、production），为不同环境的请求路由到不同项目。

### 挑战练习

7. **采样策略**：实现一个智能采样策略：对失败的请求 100% 追踪，成功的请求 5% 追踪。

8. **成本监控**：编写脚本，从 LangSmith API 提取过去一周的 Token 使用量，生成成本报告。

9. **性能对比实验**：对同一个任务使用 GPT-4 和 GPT-3.5，通过 Tracing 对比延迟和成本差异。

---

## 扩展阅读

- [LangSmith Documentation - Tracing](https://docs.smith.langchain.com/tracing)
- [LangSmith API Reference](https://api.smith.langchain.com/docs)
- [LangChain Callbacks](https://python.langchain.com/docs/modules/callbacks/)
- [Observability Best Practices for LLM Applications](https://blog.langchain.dev/observability-best-practices/)

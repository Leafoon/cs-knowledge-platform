# Chapter 1: 核心抽象与基础组件

> **本章目标**：深入理解 LangChain 的底层抽象机制，掌握 Runnable 协议、Language Models、Prompt Templates、Output Parsers 等核心组件的使用方法，建立构建 AI 应用的坚实基础。

---

## 📖 本章导览

本章深入剖析 LangChain 的核心抽象层，这些概念是理解整个框架的关键。

### 🎯 学习路线图

```
Runnable 协议 → Language Models → Prompt Templates → Output Parsers → Message 系统 → 完整应用
    ↓               ↓                  ↓                   ↓              ↓
 统一接口       模型调用          提示管理            结构化输出      对话管理
```

### 🔑 核心知识点概览

| 组件 | 核心价值 | 难度 | 重要性 | 预计学习时间 |
|------|----------|------|--------|------------|
| **Runnable 协议** | 统一接口，组合基础 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 15 分钟 |
| **Language Models** | LLM 调用与配置 | ⭐⭐ | ⭐⭐⭐⭐⭐ | 20 分钟 |
| **Prompt Templates** | 提示工程标准化 | ⭐⭐ | ⭐⭐⭐⭐ | 20 分钟 |
| **Output Parsers** | 结构化输出 | ⭐⭐⭐ | ⭐⭐⭐⭐ | 15 分钟 |
| **Message 系统** | 对话管理 | ⭐⭐ | ⭐⭐⭐ | 10 分钟 |

### 📚 本章结构

1. **1.1 Runnable 协议** - 统一接口设计与组合模式
2. **1.2 Language Models** - Chat Models、LLMs 与多提供商集成
3. **1.3 Prompt Templates** - 模板系统与提示工程
4. **1.4 Output Parsers** - 结构化输出解析
5. **1.5 Message 与 Conversation** - 消息系统与对话管理
6. **1.6 高级主题** - RunnableConfig、自定义组件与性能优化

---

## 1.1 Runnable 协议

> **核心理念**：LangChain 通过 Runnable 协议统一所有组件的调用接口，实现灵活的组合与编排。

### 1.1.1 设计动机：为什么需要 Runnable？

**问题背景**：

在 LangChain 早期版本中，不同组件有不同的调用方式：
- PromptTemplate 使用 `.format()`
- LLM 使用 `.predict()` 或 `__call__()`
- Chain 使用 `.run()` 或 `.apply()`

这导致：
- ❌ 组合困难：不同组件难以无缝连接
- ❌ 学习曲线陡峭：需要记住多种调用方式
- ❌ 代码不一致：同一操作有多种写法

**解决方案：Runnable 协议**

Runnable 是一个抽象基类，定义了统一的接口标准：

```python
from abc import ABC, abstractmethod
from typing import Any, Iterator, AsyncIterator, Optional

class Runnable(ABC):
    """所有可执行组件的基类"""
    
    @abstractmethod
    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        """同步调用：阻塞直到结果返回"""
        pass
    
    @abstractmethod
    async def ainvoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        """异步调用：非阻塞，适合高并发场景"""
        pass
    
    @abstractmethod
    def stream(self, input: Any, config: Optional[RunnableConfig] = None) -> Iterator[Any]:
        """同步流式输出：逐块返回结果"""
        pass
    
    @abstractmethod
    async def astream(self, input: Any, config: Optional[RunnableConfig] = None) -> AsyncIterator[Any]:
        """异步流式输出：结合异步与流式的优势"""
        pass
    
    @abstractmethod
    def batch(self, inputs: list[Any], config: Optional[RunnableConfig] = None) -> list[Any]:
        """批量处理：一次性处理多个输入"""
        pass
    
    @abstractmethod
    async def abatch(self, inputs: list[Any], config: Optional[RunnableConfig] = None) -> list[Any]:
        """异步批量处理：高效处理大批量任务"""
        pass
```

**优势**：
- ✅ 统一接口：所有组件使用相同的调用方式
- ✅ 灵活组合：通过 `|` 操作符连接组件
- ✅ 性能优化：支持流式、批量、异步等多种执行模式
- ✅ 配置传递：RunnableConfig 在链中自动传递

### 1.1.2 核心方法详解

<div data-component="RunnableProtocolVisualizer"></div>

#### 方法 1：invoke() - 同步单次调用

**适用场景**：简单脚本、单次请求、测试代码

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 创建模型实例
model = ChatOpenAI(
    model="gpt-4o-mini",      # 模型名称
    temperature=0.7,          # 温度参数（0-2）
    timeout=30,               # 超时时间（秒）
    max_retries=2             # 重试次数
)

# 同步调用
message = HumanMessage(content="What is 2+2?")
response = model.invoke([message])

print(response.content)       # "4"
print(type(response))         # <class 'langchain_core.messages.ai.AIMessage'>
```

**参数说明**：

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `model` | `str` | 模型标识符（如 `gpt-4o`, `gpt-4o-mini`） | 必填 |
| `temperature` | `float` | 控制输出随机性（0=确定性，2=高创意） | `0.7` |
| `timeout` | `int` | 请求超时时间（秒） | `None` |
| `max_retries` | `int` | 失败重试次数 | `2` |
| `max_tokens` | `int` | 最大生成token数 | `None`（模型默认值） |
| `streaming` | `bool` | 是否启用流式输出 | `False` |

**执行流程**：

```
输入 → invoke() → 网络请求 → 等待响应 → 返回完整结果
         ↓
      阻塞主线程（同步）
```

#### 方法 2：ainvoke() - 异步单次调用

**适用场景**：Web 后端（FastAPI、Django）、高并发应用、需要同时处理多个请求

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

async def async_example():
    model = ChatOpenAI(model="gpt-4o-mini")
    
    # 异步调用
    response = await model.ainvoke([
        HumanMessage(content="What is the capital of France?")
    ])
    
    print(response.content)  # "Paris"

# 运行异步函数
asyncio.run(async_example())
```

**并发优势对比**：

```python
import time

# 同步版本：总耗时 = 单次耗时 × 请求数
def sync_version():
    model = ChatOpenAI(model="gpt-4o-mini")
    questions = ["What is 1+1?", "What is 2+2?", "What is 3+3?"]
    
    start = time.time()
    for q in questions:
        model.invoke([HumanMessage(content=q)])
    
    print(f"同步耗时: {time.time() - start:.2f}秒")  # 约 3-6 秒

# 异步版本：总耗时 ≈ 单次耗时（并发执行）
async def async_version():
    model = ChatOpenAI(model="gpt-4o-mini")
    questions = ["What is 1+1?", "What is 2+2?", "What is 3+3?"]
    
    start = time.time()
    tasks = [
        model.ainvoke([HumanMessage(content=q)])
        for q in questions
    ]
    await asyncio.gather(*tasks)  # 并发执行
    
    print(f"异步耗时: {time.time() - start:.2f}秒")  # 约 1-2 秒

asyncio.run(async_version())
```

**性能提升公式**：

$$
\text{加速比} = \frac{\text{串行总时间}}{\text{并行总时间}} = \frac{n \times t}{t + \text{overhead}} \approx n
$$

其中 $n$ 为任务数量，$t$ 为单任务耗时。

#### 方法 3：stream() - 同步流式输出

**适用场景**：聊天界面、实时反馈、渐进式显示

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="gpt-4o", streaming=True)

# 流式输出
for chunk in model.stream([HumanMessage(content="Count from 1 to 10.")]):
    print(chunk.content, end="", flush=True)
    # 输出: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 (逐个显示)
```

**用户体验对比**：

```python
# 非流式：用户等待 5 秒后一次性看到完整回复
response = model.invoke([HumanMessage(content="Write a story.")])
print(response.content)  # 一次性显示全部内容

# 流式：用户立即看到开始，逐字显示（类似 ChatGPT）
for chunk in model.stream([HumanMessage(content="Write a story.")]):
    print(chunk.content, end="", flush=True)
    time.sleep(0.05)  # 模拟打字效果
```

**实现原理**：

```
LLM 生成 → 服务器分块发送 → 客户端逐块接收 → 实时显示
             ↓                    ↓
         SSE (Server-Sent     Iterator/Generator
            Events)           (Python yield)
```

#### 方法 4：astream() - 异步流式输出

**适用场景**：异步 Web 框架（FastAPI）、WebSocket、高性能实时应用

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

async def async_stream_example():
    model = ChatOpenAI(model="gpt-4o", streaming=True)
    
    async for chunk in model.astream([HumanMessage(content="Explain AI.")]):
        print(chunk.content, end="", flush=True)

asyncio.run(async_stream_example())
```

**FastAPI 集成示例**：

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

app = FastAPI()

@app.get("/stream")
async def stream_response(question: str):
    model = ChatOpenAI(model="gpt-4o", streaming=True)
    
    async def generate():
        async for chunk in model.astream([HumanMessage(content=question)]):
            yield chunk.content
    
    return StreamingResponse(generate(), media_type="text/plain")

# 访问 http://localhost:8000/stream?question=What+is+AI
```

#### 方法 5：batch() - 批量处理

**适用场景**：数据处理、批量翻译、批量摘要、批量评估

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="gpt-4o-mini")

# 批量处理多个输入
messages_batch = [
    [HumanMessage(content="Translate 'Hello' to French")],
    [HumanMessage(content="Translate 'Goodbye' to Spanish")],
    [HumanMessage(content="Translate 'Thank you' to German")]
]

responses = model.batch(messages_batch)

for resp in responses:
    print(resp.content)
# 输出:
# Bonjour
# Adiós
# Danke
```

**性能优势**：

```python
import time

# 逐个调用（不推荐）
start = time.time()
results = []
for msg in messages_batch:
    results.append(model.invoke(msg))
print(f"逐个调用耗时: {time.time() - start:.2f}秒")  # 约 6 秒

# 批量调用（推荐）
start = time.time()
results = model.batch(messages_batch)
print(f"批量调用耗时: {time.time() - start:.2f}秒")  # 约 2 秒
```

**批量调用优化原理**：

1. **请求合并**：多个请求合并为一个 HTTP 请求
2. **并行处理**：服务器端并行处理多个输入
3. **连接复用**：减少 TCP 连接建立开销

**最佳实践**：

```python
# 批量大小建议：10-50
batch_size = 20

def process_in_batches(inputs, batch_size=20):
    results = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        results.extend(model.batch(batch))
    return results

# 处理 1000 条数据
large_dataset = [[HumanMessage(content=f"Translate {i}")] for i in range(1000)]
results = process_in_batches(large_dataset)
```

#### 方法 6：abatch() - 异步批量处理

**适用场景**：大规模数据处理、高并发批量任务

```python
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

async def async_batch_example():
    model = ChatOpenAI(model="gpt-4o-mini")
    
    messages_batch = [
        [HumanMessage(content=f"What is {i} + {i}?")]
        for i in range(1, 6)
    ]
    
    results = await model.abatch(messages_batch)
    
    for i, resp in enumerate(results, 1):
        print(f"{i} + {i} = {resp.content}")

asyncio.run(async_batch_example())
```

### 1.1.3 方法选择决策树

```
是否需要实时反馈？
├─ 是 → 是否异步环境？
│        ├─ 是 → astream()
│        └─ 否 → stream()
└─ 否 → 是否批量处理？
         ├─ 是 → 是否异步环境？
         │        ├─ 是 → abatch()
         │        └─ 否 → batch()
         └─ 否 → 是否异步环境？
                  ├─ 是 → ainvoke()
                  └─ 否 → invoke()
```

**完整对比表**：

| 方法 | 同步/异步 | 流式 | 批量 | 适用场景 | 性能 | 复杂度 |
|------|-----------|------|------|----------|------|--------|
| `invoke()` | 同步 | ❌ | ❌ | 简单脚本、测试 | ⭐⭐ | ⭐ |
| `ainvoke()` | 异步 | ❌ | ❌ | Web 后端、并发 | ⭐⭐⭐⭐ | ⭐⭐ |
| `stream()` | 同步 | ✅ | ❌ | 聊天界面 | ⭐⭐⭐ | ⭐⭐ |
| `astream()` | 异步 | ✅ | ❌ | 异步聊天 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| `batch()` | 同步 | ❌ | ✅ | 批量数据处理 | ⭐⭐⭐⭐ | ⭐⭐ |
| `abatch()` | 异步 | ❌ | ✅ | 大规模数据 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

### 1.1.4 Runnable 组合模式

#### 管道操作符（|）

**核心语法**：

```python
# 使用 | 操作符组合多个 Runnable
chain = component1 | component2 | component3

# 等价于
from langchain_core.runnables import RunnableSequence
chain = RunnableSequence(component1, component2, component3)
```

**完整示例**：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 创建 Prompt Template (Runnable)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a translator."),
    ("human", "Translate '{text}' to {language}.")
])

# 2. 创建 Model (Runnable)
model = ChatOpenAI(model="gpt-4o-mini")

# 3. 创建 Output Parser (Runnable)
parser = StrOutputParser()

# 4. 组合成链
chain = prompt | model | parser

# 5. 执行
result = chain.invoke({"text": "Hello", "language": "French"})
print(result)  # "Bonjour"
```

**执行流程可视化**：

```
输入: {"text": "Hello", "language": "French"}
  ↓
prompt.invoke() → 生成消息列表
  ↓
[SystemMessage("You are a translator."),
 HumanMessage("Translate 'Hello' to French.")]
  ↓
model.invoke() → 调用 LLM
  ↓
AIMessage(content="Bonjour")
  ↓
parser.invoke() → 提取文本
  ↓
输出: "Bonjour"
```

#### RunnablePassthrough：透传与调试

**基本用法**：

```python
from langchain_core.runnables import RunnablePassthrough

# 直接透传输入
passthrough = RunnablePassthrough()
result = passthrough.invoke({"key": "value"})
print(result)  # {"key": "value"}
```

**调试链**：

```python
# 在链中插入 passthrough 查看中间结果
chain = (
    prompt
    | RunnablePassthrough()  # 查看 prompt 输出
    | model
    | RunnablePassthrough()  # 查看 model 输出
    | parser
)
```

**添加额外字段**：

```python
chain = (
    {"input": RunnablePassthrough()}  # 保留原始输入
    | prompt
    | model
    | {"output": parser, "raw": RunnablePassthrough()}  # 同时返回解析结果和原始消息
)

result = chain.invoke({"text": "Hello"})
# {
#   "output": "Bonjour",
#   "raw": AIMessage(content="Bonjour")
# }
```

#### RunnableParallel：并行执行

**基本用法**：

```python
from langchain_core.runnables import RunnableParallel

# 并行执行多个任务
parallel = RunnableParallel(
    french=chain_french,
    spanish=chain_spanish,
    german=chain_german
)

result = parallel.invoke({"text": "Hello"})
# {
#   "french": "Bonjour",
#   "spanish": "Hola",
#   "german": "Guten Tag"
# }
```

**实际案例：多角度分析**：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# 定义多个分析链
sentiment_chain = (
    ChatPromptTemplate.from_template("Analyze the sentiment of: {text}")
    | model
    | parser
)

topic_chain = (
    ChatPromptTemplate.from_template("Extract the main topic of: {text}")
    | model
    | parser
)

summary_chain = (
    ChatPromptTemplate.from_template("Summarize in one sentence: {text}")
    | model
    | parser
)

# 并行执行
analysis_pipeline = RunnableParallel(
    sentiment=sentiment_chain,
    topic=topic_chain,
    summary=summary_chain
)

result = analysis_pipeline.invoke({
    "text": "LangChain is an amazing framework for building LLM applications!"
})

print(result)
# {
#   "sentiment": "Positive",
#   "topic": "LangChain framework",
#   "summary": "LangChain is a great tool for developing LLM-based apps."
# }
```

#### RunnableLambda：包装任意函数

**基本用法**：

```python
from langchain_core.runnables import RunnableLambda

def uppercase(text: str) -> str:
    return text.upper()

def add_prefix(text: str) -> str:
    return f"[TRANSLATED] {text}"

# 包装为 Runnable
chain = (
    prompt
    | model
    | parser
    | RunnableLambda(uppercase)
    | RunnableLambda(add_prefix)
)

result = chain.invoke({"text": "Hello", "language": "French"})
print(result)  # "[TRANSLATED] BONJOUR"
```

**复杂数据处理**：

```python
def extract_and_format(ai_message):
    """从 AIMessage 提取内容并格式化"""
    content = ai_message.content
    return {
        "text": content,
        "length": len(content),
        "word_count": len(content.split())
    }

chain = (
    prompt
    | model
    | RunnableLambda(extract_and_format)
)

result = chain.invoke({"text": "Hello", "language": "French"})
# {
#   "text": "Bonjour",
#   "length": 7,
#   "word_count": 1
# }
```

#### RunnableBranch：条件分支

**基本用法**：

```python
from langchain_core.runnables import RunnableBranch

def route_by_language(input_dict):
    """根据语言选择不同的链"""
    language = input_dict.get("language", "").lower()
    
    if language == "french":
        return chain_french
    elif language == "spanish":
        return chain_spanish
    else:
        return chain_default

# 创建分支
branch = RunnableBranch(
    (lambda x: x["language"] == "french", chain_french),
    (lambda x: x["language"] == "spanish", chain_spanish),
    chain_default  # 默认分支
)

result = branch.invoke({"text": "Hello", "language": "french"})
```

### 1.1.5 RunnableConfig：配置传递

**RunnableConfig 结构**：

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    # 回调管理
    callbacks=[StdOutCallbackHandler()],
    
    # 标签与元数据
    tags=["production", "translation"],
    metadata={"user_id": "12345", "session_id": "abc"},
    
    # 运行时配置
    run_name="translation_task",
    max_concurrency=5,
    
    # 递归限制
    recursion_limit=25
)
```

**在链中传递配置**：

```python
# 配置会自动传递给链中的所有组件
result = chain.invoke(
    {"text": "Hello", "language": "French"},
    config=config
)
```

**动态配置**：

```python
def get_config_for_user(user_id: str) -> RunnableConfig:
    """根据用户ID生成配置"""
    return RunnableConfig(
        tags=[f"user:{user_id}"],
        metadata={"user_id": user_id}
    )

# 使用
user_config = get_config_for_user("user_123")
result = chain.invoke({"text": "Hello"}, config=user_config)
```

---

## 1.2 Language Models 集成

> **核心概念**：LangChain 支持多种语言模型提供商，通过统一接口实现模型无缝切换。

### 1.2.1 Chat Models vs LLMs

**架构对比**：

| 特性 | LLM | Chat Model |
|------|-----|------------|
| **输入格式** | 字符串 | 消息列表（List[BaseMessage]） |
| **输出格式** | 字符串 | AIMessage |
| **典型模型** | GPT-3 text-davinci-003 | GPT-4, Claude-3, Llama-3 |
| **上下文管理** | 需手动拼接 | 原生支持角色区分 |
| **Function Calling** | ❌ 不支持 | ✅ 支持 |
| **推荐使用** | ❌ 已废弃 | ✅ 优先使用 |

**LLM 示例**（不推荐，仅供理解）：

```python
from langchain_openai import OpenAI

llm = OpenAI(
    model="gpt-3.5-turbo-instruct",  # 旧式模型
    temperature=0.7
)

# 输入：纯文本字符串
prompt = "Translate 'Hello' to French:"
response = llm.invoke(prompt)

print(response)  # "Bonjour"（字符串）
print(type(response))  # <class 'str'>
```

**Chat Model 示例**（推荐）：

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

chat = ChatOpenAI(
    model="gpt-4o",  # 现代对话模型
    temperature=0.7
)

# 输入：消息列表
messages = [
    SystemMessage(content="You are a professional translator."),
    HumanMessage(content="Translate 'Hello' to French.")
]

response = chat.invoke(messages)

print(response.content)  # "Bonjour"
print(type(response))    # <class 'langchain_core.messages.ai.AIMessage'>
print(response.response_metadata)  # {'model': 'gpt-4o', 'usage': {...}}
```

**为什么必须使用 Chat Model？**

1. **角色分离**：System、Human、AI 消息清晰区分
2. **对话历史**：消息列表天然支持多轮对话
3. **元数据丰富**：包含 token 使用量、模型版本等信息
4. **功能完整**：支持 Function/Tool Calling、JSON Mode、Streaming
5. **行业标准**：所有现代 LLM 都是对话模型训练

### 1.2.2 模型提供商集成

#### OpenAI（推荐）

**安装**：

```bash
# 安装 LangChain OpenAI 集成
pip install langchain-openai

# 环境变量配置
export OPENAI_API_KEY="sk-..."
```

**基础用法**：

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    # 必填参数
    model="gpt-4o",                    # 模型名称
    
    # API 配置
    api_key="sk-...",                  # API 密钥（或从环境变量读取）
    base_url="https://api.openai.com/v1",  # API 端点
    organization="org-...",            # 组织 ID（可选）
    
    # 生成参数
    temperature=0.7,                   # 温度：0-2
    max_tokens=1000,                   # 最大生成 token 数
    top_p=1.0,                         # 核采样：0-1
    frequency_penalty=0.0,             # 频率惩罚：-2 to 2
    presence_penalty=0.0,              # 存在惩罚：-2 to 2
    n=1,                               # 生成结果数量
    
    # 连接参数
    timeout=30,                        # 总超时时间（秒）
    max_retries=2,                     # 最大重试次数
    request_timeout=60,                # 单次请求超时
    
    # 流式参数
    streaming=True,                    # 启用流式输出
    
    # 额外参数
    model_kwargs={                     
        "seed": 42,                    # 随机种子（可复现）
        "response_format": {"type": "json_object"}  # JSON 模式
    }
)
```

**参数详解**：

##### temperature（温度）

控制输出的随机性和创造性。

$$
\text{probability}(token_i) = \frac{\exp(logit_i / T)}{\sum_j \exp(logit_j / T)}
$$

其中 $T$ 为温度。

| Temperature | 效果 | 适用场景 | 示例 |
|-------------|------|----------|------|
| `0` | 完全确定性 | 翻译、摘要、数据提取 | "Translate 'cat' to French" → 总是 "chat" |
| `0.3-0.5` | 轻微随机性 | 问答、分类 | 回答稍有变化但核心一致 |
| `0.7-0.9` | 平衡 | 对话、内容生成 | ChatGPT 默认值 |
| `1.0-1.5` | 高创造性 | 创意写作、头脑风暴 | 每次生成不同的故事 |
| `1.5-2.0` | 极高随机性 | 探索性实验 | 输出可能不连贯 |

**实验对比**：

```python
prompts = [HumanMessage(content="Write a 3-word slogan for a coffee shop.")]

# 低温度
model_det = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print("T=0:", model_det.invoke(prompts).content)
# 多次运行几乎相同：
# "Fresh Coffee Daily"
# "Fresh Coffee Daily"
# "Fresh Coffee Daily"

# 高温度
model_creative = ChatOpenAI(model="gpt-4o-mini", temperature=1.5)
print("T=1.5:", model_creative.invoke(prompts).content)
# 每次不同：
# "Brewed to Perfection"
# "Sip, Savor, Smile"
# "Awaken Your Senses"
```

##### top_p（核采样）

只考虑累积概率达到 $p$ 的 token 集合。

$$
\text{tokens}_\text{considered} = \{t : \sum_{i=1}^{t} P(token_i) \leq p\}
$$

| Top P | 效果 | 与 Temperature 配合 |
|-------|------|---------------------|
| `0.1` | 只考虑高概率词 | 适合低温度，确保质量 |
| `0.5` | 中等范围 | 平衡多样性和质量 |
| `0.9-1.0` | 考虑大部分词 | 高温度，最大化创造性 |

**最佳实践**：
- 翻译/摘要：`temperature=0, top_p=1`
- 对话：`temperature=0.7, top_p=0.9`
- 创作：`temperature=1.0, top_p=0.95`

##### frequency_penalty 与 presence_penalty

**frequency_penalty**：根据词频降低重复词的概率。

$$
\text{penalty} = \alpha \times \text{count}(token)
$$

**presence_penalty**：如果词已出现，降低其概率（不考虑次数）。

$$
\text{penalty} = \alpha \times \mathbb{I}(token \text{ appeared})
$$

| 参数 | 范围 | 效果 | 适用场景 |
|------|------|------|----------|
| `frequency_penalty` | -2 to 2 | 减少重复词汇 | 生成多样化文本 |
| `presence_penalty` | -2 to 2 | 鼓励新话题 | 避免偏离主题 |

**示例**：

```python
# 无惩罚
model_none = ChatOpenAI(model="gpt-4o-mini", frequency_penalty=0)
# 可能输出：
# "The cat is cute. The cat is fluffy. The cat is playful."

# 频率惩罚
model_freq = ChatOpenAI(model="gpt-4o-mini", frequency_penalty=1.0)
# 输出：
# "The cat is cute. It's fluffy. This feline is playful."
```

##### max_tokens

限制生成的最大 token 数量。

**Token 估算**：
- 英文：1 token ≈ 4 characters ≈ 0.75 words
- 中文：1 token ≈ 1-2 characters

```python
# 控制输出长度
short_model = ChatOpenAI(model="gpt-4o", max_tokens=50)
long_model = ChatOpenAI(model="gpt-4o", max_tokens=500)

prompt = [HumanMessage(content="Explain quantum physics.")]

print("Short:", short_model.invoke(prompt).content)
# 约 50 tokens，简短回答

print("Long:", long_model.invoke(prompt).content)
# 约 500 tokens，详细解释
```

#### Anthropic Claude

**安装**：

```bash
pip install langchain-anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

**用法**：

```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",  # 推荐模型
    temperature=0.7,
    max_tokens=4096,
    timeout=30,
    max_retries=2,
    api_key="sk-ant-..."
)

# 使用方式与 OpenAI 完全相同
response = model.invoke([HumanMessage(content="Hello")])
```

**Claude 模型选择**：

| 模型 | 上下文窗口 | 特点 | 适用场景 |
|------|-----------|------|----------|
| `claude-3-5-sonnet-20241022` | 200K | 最强能力，最新版本 | 复杂推理、长文档 |
| `claude-3-opus-20240229` | 200K | 最高质量 | 需要最佳性能 |
| `claude-3-haiku-20240307` | 200K | 最快速度，低成本 | 简单任务、高并发 |

#### 本地模型（Ollama）

**安装与启动**：

```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 下载模型
ollama pull llama3.2
ollama pull mistral

# 启动服务
ollama serve
```

**LangChain 集成**：

```python
from langchain_community.chat_models import ChatOllama

model = ChatOllama(
    model="llama3.2",                  # 本地模型名称
    temperature=0.7,
    base_url="http://localhost:11434"  # Ollama API 端点
)

response = model.invoke([HumanMessage(content="Hello")])
```

**优势与限制**：

| 特性 | Ollama | OpenAI/Anthropic |
|------|--------|------------------|
| **成本** | ✅ 免费 | ❌ 按 token 计费 |
| **隐私** | ✅ 本地部署 | ❌ 数据上传云端 |
| **性能** | ⚠️ 取决于硬件 | ✅ 高性能 |
| **能力** | ⚠️ 较弱 | ✅ 最强 |
| **维护** | ❌ 需自行管理 | ✅ 无需维护 |

### 1.2.3 统一模型接口（工厂模式）

**问题**：如何在不同提供商之间无缝切换？

**解决方案**：工厂函数

```python
from typing import Literal
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama

def get_model(
    provider: Literal["openai", "anthropic", "ollama"] = "openai",
    model_name: str | None = None,
    **kwargs
) -> BaseChatModel:
    """
    工厂函数：根据提供商创建模型
    
    Args:
        provider: 模型提供商（openai/anthropic/ollama）
        model_name: 模型名称（可选，使用默认值）
        **kwargs: 额外参数（temperature、max_tokens等）
    
    Returns:
        BaseChatModel: 统一接口的聊天模型
    
    Examples:
        >>> model = get_model("openai", temperature=0.5)
        >>> model = get_model("anthropic", model_name="claude-3-opus-20240229")
        >>> model = get_model("ollama", model_name="llama3.2")
    """
    # 默认模型映射
    default_models = {
        "openai": "gpt-4o",
        "anthropic": "claude-3-5-sonnet-20241022",
        "ollama": "llama3.2"
    }
    
    # 使用指定模型或默认模型
    model = model_name or default_models[provider]
    
    # 创建模型实例
    if provider == "openai":
        return ChatOpenAI(model=model, **kwargs)
    elif provider == "anthropic":
        return ChatAnthropic(model=model, **kwargs)
    elif provider == "ollama":
        return ChatOllama(model=model, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")

# 使用示例
model = get_model("anthropic", temperature=0.7)

# 无需修改其他代码
chain = prompt | model | parser
result = chain.invoke({"text": "Hello", "language": "French"})
```

**环境变量配置**：

```python
import os

def get_model_from_env(**kwargs) -> BaseChatModel:
    """从环境变量读取配置"""
    provider = os.getenv("LLM_PROVIDER", "openai")
    model_name = os.getenv("LLM_MODEL")
    
    return get_model(provider, model_name, **kwargs)

# .env 文件
# LLM_PROVIDER=anthropic
# LLM_MODEL=claude-3-5-sonnet-20241022

model = get_model_from_env(temperature=0.7)
```

### 1.2.4 Callbacks 与监控

<div data-component="CallbackFlow"></div>

#### 标准输出 Callback

```python
from langchain.callbacks import StdOutCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(
    model="gpt-4o-mini",
    callbacks=[StdOutCallbackHandler()],
    verbose=True
)

response = model.invoke([HumanMessage(content="Hello")])

# 输出：
# > Entering new ChatOpenAI chain...
# > Prompt: [HumanMessage(content='Hello')]
# > Response: AIMessage(content='Hi there! How can I assist you today?')
# > Finished chain.
```

#### 自定义 Callback：Token 计数器

```python
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

class TokenCounterCallback(BaseCallbackHandler):
    """统计 token 使用量"""
    
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
    
    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs):
        """LLM 开始时触发"""
        print(f"🚀 Starting LLM with {len(prompts)} prompts")
    
    def on_llm_end(self, response: LLMResult, **kwargs):
        """LLM 结束时触发"""
        # 提取 token 使用信息
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += total_tokens
            
            # 成本计算（GPT-4o 价格）
            cost = (prompt_tokens * 0.0025 + completion_tokens * 0.01) / 1000
            self.total_cost += cost
            
            print(f"📊 Tokens: {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} total")
            print(f"💰 Cost: ${cost:.6f}")
    
    def on_llm_error(self, error: Exception, **kwargs):
        """LLM 出错时触发"""
        print(f"❌ Error: {error}")
    
    def reset(self):
        """重置计数器"""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0

# 使用
counter = TokenCounterCallback()
model = ChatOpenAI(model="gpt-4o", callbacks=[counter])

model.invoke([HumanMessage(content="Hello")])
# 输出:
# 🚀 Starting LLM with 1 prompts
# 📊 Tokens: 8 prompt + 12 completion = 20 total
# 💰 Cost: $0.000140

# 查看累计统计
print(f"Total tokens: {counter.total_tokens}")
print(f"Total cost: ${counter.total_cost:.6f}")
```

#### 自定义 Callback：延迟监控

```python
import time
from langchain.callbacks.base import BaseCallbackHandler

class LatencyMonitorCallback(BaseCallbackHandler):
    """监控 LLM 调用延迟"""
    
    def __init__(self):
        self.start_time = None
        self.latencies = []
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """记录开始时间"""
        self.start_time = time.time()
    
    def on_llm_end(self, response, **kwargs):
        """计算延迟"""
        if self.start_time:
            latency = time.time() - self.start_time
            self.latencies.append(latency)
            print(f"⏱️  Latency: {latency:.2f}s")
            self.start_time = None
    
    def get_stats(self):
        """获取统计信息"""
        if not self.latencies:
            return {"count": 0}
        
        return {
            "count": len(self.latencies),
            "avg": sum(self.latencies) / len(self.latencies),
            "min": min(self.latencies),
            "max": max(self.latencies)
        }

# 使用
latency_monitor = LatencyMonitorCallback()
model = ChatOpenAI(model="gpt-4o-mini", callbacks=[latency_monitor])

for i in range(5):
    model.invoke([HumanMessage(content=f"Say {i}")])

print("\n统计信息：", latency_monitor.get_stats())
# {
#   "count": 5,
#   "avg": 1.23,
#   "min": 0.98,
#   "max": 1.56
# }
```

#### 组合多个 Callbacks

```python
# 同时使用多个 callback
model = ChatOpenAI(
    model="gpt-4o-mini",
    callbacks=[
        StdOutCallbackHandler(),
        TokenCounterCallback(),
        LatencyMonitorCallback()
    ]
)
```

---

## 1.3 Prompt Templates

> **核心价值**：Prompt Templates 将提示词从代码中解耦，实现复用、版本管理和协作。

### 1.3.1 PromptTemplate 基础

#### 创建方式

**方式 1：from_template()（推荐）**

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "Translate the following text to {language}: {text}"
)

# 格式化
prompt = template.format(language="French", text="Hello")
print(prompt)
# "Translate the following text to French: Hello"

# 作为 Runnable 使用
result = template.invoke({"language": "Spanish", "text": "Goodbye"})
print(result.to_string())
# "Translate the following text to Spanish: Goodbye"
```

**方式 2：构造函数**

```python
template = PromptTemplate(
    input_variables=["language", "text"],
    template="Translate the following text to {language}: {text}"
)

# 验证变量
print(template.input_variables)  # ['language', 'text']
```

**方式 3：from_file()（大型提示）**

```python
# prompts/translate.txt:
# Translate the following text to {language}:
#
# Text: {text}
#
# Translation:

template = PromptTemplate.from_file(
    "prompts/translate.txt",
    input_variables=["language", "text"]
)
```

#### 变量类型

**单变量**：

```python
template = PromptTemplate.from_template("Say {word}")
result = template.invoke({"word": "hello"})
```

**多变量**：

```python
template = PromptTemplate.from_template(
    "Translate '{text}' from {source_lang} to {target_lang}"
)

result = template.invoke({
    "text": "Bonjour",
    "source_lang": "French",
    "target_lang": "English"
})
```

**可选变量**（使用 partial）：

```python
template = PromptTemplate.from_template(
    "You are a {role}. {instruction}"
)

# 固定角色
assistant_template = template.partial(role="helpful assistant")

# 后续只需提供 instruction
result = assistant_template.invoke({"instruction": "Explain AI."})
```

#### 部分填充（Partial）

**静态部分填充**：

```python
from datetime import datetime

template = PromptTemplate.from_template(
    "Today is {date}. {question}"
)

# 固定日期
dated_template = template.partial(date="2024-01-29")

# 每次只需提供问题
result = dated_template.invoke({"question": "What is the weather?"})
```

**动态部分填充（函数）**：

```python
def get_current_date():
    """每次调用时获取当前日期"""
    return datetime.now().strftime("%Y-%m-%d")

template = PromptTemplate.from_template(
    "Current date: {date}. {question}"
)

# 使用函数动态填充
dynamic_template = template.partial(date=get_current_date)

# 每次调用时自动获取最新日期
result1 = dynamic_template.invoke({"question": "What day is it?"})
# "Current date: 2024-01-29. What day is it?"

# 一天后调用
result2 = dynamic_template.invoke({"question": "What day is it?"})
# "Current date: 2024-01-30. What day is it?"
```

### 1.3.2 ChatPromptTemplate：对话模板

<div data-component="PromptComposer"></div>

#### 基础用法

```python
from langchain_core.prompts import ChatPromptTemplate

# 方式1：from_messages（推荐）
template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("human", "{user_input}")
])

# 格式化
messages = template.invoke({
    "role": "translator",
    "user_input": "Translate 'Hello' to French"
})

print(messages)
# [
#   SystemMessage(content='You are a translator.'),
#   HumanMessage(content="Translate 'Hello' to French")
# ]
```

**方式2：使用消息类**

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a {role}."),
    HumanMessagePromptTemplate.from_template("{user_input}")
])
```

#### 消息角色

**支持的角色类型**：

| 角色 | 字符串表示 | 类 | 用途 |
|------|-----------|-----|------|
| System | `"system"` | `SystemMessage` | 系统指令、角色设定 |
| Human | `"human"`, `"user"` | `HumanMessage` | 用户输入 |
| AI | `"ai"`, `"assistant"` | `AIMessage` | AI 回复（历史） |
| Tool | `"tool"` | `ToolMessage` | 工具调用结果 |

**完整示例**：

```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant."),
    ("human", "What is {topic}?"),
    ("ai", "Let me explain {topic} in detail."),
    ("human", "Can you summarize?")
])

messages = template.invoke({"topic": "quantum physics"})
# [
#   SystemMessage(content='You are an AI assistant.'),
#   HumanMessage(content='What is quantum physics?'),
#   AIMessage(content='Let me explain quantum physics in detail.'),
#   HumanMessage(content='Can you summarize?')
# ]
```

#### 多轮对话模板

**场景：Few-Shot 示例**

```python
template = ChatPromptTemplate.from_messages([
    ("system", "You are a sentiment analyzer. Classify text as Positive, Negative, or Neutral."),
    ("human", "I love this product!"),
    ("ai", "Positive"),
    ("human", "This is terrible."),
    ("ai", "Negative"),
    ("human", "It's okay."),
    ("ai", "Neutral"),
    ("human", "{text}")  # 实际待分类文本
])

result = template.invoke({"text": "Amazing experience!"})
```

### 1.3.3 Few-Shot Prompting

#### FewShotPromptTemplate（旧式，不推荐）

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# 定义示例
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "hot", "output": "cold"}
]

# 示例模板
example_template = PromptTemplate.from_template(
    "Input: {input}\nOutput: {output}"
)

# Few-Shot 模板
few_shot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="Give the opposite of the word.",
    suffix="Input: {word}\nOutput:",
    input_variables=["word"]
)

prompt = few_shot_template.format(word="big")
print(prompt)
# Give the opposite of the word.
# Input: happy
# Output: sad
# Input: tall
# Output: short
# Input: hot
# Output: cold
# Input: big
# Output:
```

#### FewShotChatMessagePromptTemplate（推荐）

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate
)

# 定义示例（使用消息格式）
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "3+5", "output": "8"}
]

# 示例提示模板
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

# Few-Shot 模板
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt
)

# 最终模板
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a math calculator."),
    few_shot_prompt,
    ("human", "{input}")
])

# 使用
messages = final_prompt.invoke({"input": "5+7"})
# [
#   SystemMessage(content='You are a math calculator.'),
#   HumanMessage(content='2+2'),
#   AIMessage(content='4'),
#   HumanMessage(content='3+5'),
#   AIMessage(content='8'),
#   HumanMessage(content='5+7')
# ]
```

#### 动态示例选择（ExampleSelector）

**场景**：根据输入选择最相关的示例

```python
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 定义大量示例
examples = [
    {"input": "happy", "output": "😊"},
    {"input": "sad", "output": "😢"},
    {"input": "angry", "output": "😠"},
    {"input": "excited", "output": "🎉"},
    {"input": "tired", "output": "😴"}
]

# 创建示例选择器
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),  # 使用 embeddings 计算相似度
    FAISS,               # 向量数据库
    k=2                  # 选择最相关的 2 个示例
)

# Few-Shot 模板
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_selector=example_selector,  # 使用选择器
    example_prompt=ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ])
)

# 使用
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "Convert words to emojis."),
    few_shot_prompt,
    ("human", "{input}")
])

# 输入 "joyful" 会自动选择 "happy" 和 "excited" 作为示例
messages = final_prompt.invoke({"input": "joyful"})
```

### 1.3.4 LangChain Hub 集成

**LangChain Hub** 是一个提示词管理平台，类似 GitHub for Prompts。

#### 安装与配置

```bash
pip install langchainhub
export LANGCHAIN_API_KEY="ls__..."
```

#### 拉取公开提示

```python
from langchain import hub

# 拉取热门提示
prompt = hub.pull("rlm/rag-prompt")

# 查看内容
print(prompt.template)

# 使用
chain = prompt | model | parser
result = chain.invoke({
    "context": "LangChain is a framework...",
    "question": "What is LangChain?"
})
```

#### 推送自定义提示

```python
from langchain_core.prompts import ChatPromptTemplate

# 创建自定义提示
my_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {domain} consultant."),
    ("human", "{question}")
])

# 推送到 Hub（需要登录）
hub.push("my-username/expert-consultant", my_prompt)

# 他人可以拉取
prompt = hub.pull("my-username/expert-consultant")
```

#### 版本管理

```python
# 拉取特定版本
prompt_v1 = hub.pull("rlm/rag-prompt:v1")
prompt_v2 = hub.pull("rlm/rag-prompt:v2")

# 拉取最新版本
prompt_latest = hub.pull("rlm/rag-prompt")
```

---

## 1.4 Output Parsers

> **核心价值**：将 LLM 的文本输出转换为结构化数据（JSON、Python 对象等），实现类型安全。

### 1.4.1 StrOutputParser：文本提取

**最简单的解析器**：从 AIMessage 提取 `.content` 字段。

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# 不使用 parser
response = model.invoke([HumanMessage(content="Say hi")])
print(type(response))  # <class 'langchain_core.messages.ai.AIMessage'>
print(response.content)  # "Hi!"

# 使用 parser
chain = model | parser
result = chain.invoke([HumanMessage(content="Say hi")])
print(type(result))  # <class 'str'>
print(result)  # "Hi!"
```

**适用场景**：
- 简单文本生成
- 不需要结构化输出
- 快速原型开发

### 1.4.2 JsonOutputParser：JSON 解析

**基本用法**：

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Output your response as valid JSON."),
    ("human", "List 3 colors with their hex codes.")
])

chain = prompt | model | parser

result = chain.invoke({})
print(type(result))  # <class 'dict'>
print(result)
# {
#   "colors": [
#     {"name": "red", "hex": "#FF0000"},
#     {"name": "green", "hex": "#00FF00"},
#     {"name": "blue", "hex": "#0000FF"}
#   ]
# }

# 可以直接访问
print(result["colors"][0]["name"])  # "red"
```

**错误处理**：

```python
try:
    result = chain.invoke({})
except Exception as e:
    print(f"Parsing failed: {e}")
    # 可以重试或使用默认值
```

### 1.4.3 PydanticOutputParser：类型安全

**核心优势**：
- ✅ 类型检查：IDE 自动补全和类型提示
- ✅ 数据验证：自动验证字段类型和约束
- ✅ 文档生成：自动生成 schema 说明

**基础用法**：

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, field_validator

# 1. 定义数据模型
class Person(BaseModel):
    """Person information"""
    
    name: str = Field(description="Person's full name")
    age: int = Field(description="Person's age in years", ge=0, le=150)
    occupation: str = Field(description="Person's current job")
    email: str | None = Field(default=None, description="Email address")
    
    @field_validator("email")
    def validate_email(cls, v):
        """验证邮箱格式"""
        if v and "@" not in v:
            raise ValueError("Invalid email")
        return v

# 2. 创建解析器
parser = PydanticOutputParser(pydantic_object=Person)

# 3. 获取格式说明
format_instructions = parser.get_format_instructions()
print(format_instructions)
# The output should be formatted as a JSON instance that conforms to the JSON schema below.
# 
# {"properties": {"name": {"description": "Person's full name", "title": "Name", "type": "string"}, ...}}
# 
# Here is the output schema:
# ```
# {"name": "string", "age": "integer", "occupation": "string", "email": "string"}
# ```

# 4. 在提示中使用
prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract person information from the text.\n{format_instructions}"),
    ("human", "{input}")
])

chain = prompt | model | parser

# 5. 调用
result = chain.invoke({
    "format_instructions": format_instructions,
    "input": "John Doe is a 30-year-old software engineer at Google. His email is john@gmail.com."
})

print(type(result))  # <class '__main__.Person'>
print(result)
# Person(name='John Doe', age=30, occupation='software engineer', email='john@gmail.com')

# 6. 访问属性（类型安全）
print(result.name)  # "John Doe" (IDE 有自动补全)
print(result.age + 5)  # 35 (类型检查通过)
```

**复杂嵌套结构**：

```python
from typing import List
from pydantic import BaseModel, Field

class Address(BaseModel):
    """Address information"""
    street: str
    city: str
    country: str
    postal_code: str

class Company(BaseModel):
    """Company information"""
    name: str
    industry: str
    employees: int = Field(ge=1)

class Person(BaseModel):
    """Complete person profile"""
    name: str
    age: int
    addresses: List[Address] = Field(default_factory=list)
    company: Company | None = None

parser = PydanticOutputParser(pydantic_object=Person)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract detailed person information.\n{format_instructions}"),
    ("human", "{text}")
])

chain = prompt | model | parser

result = chain.invoke({
    "format_instructions": parser.get_format_instructions(),
    "text": """
    Alice Johnson, 28, works at TechCorp (a software company with 500 employees).
    She lives at 123 Main St, San Francisco, CA 94101, USA.
    She also has a vacation home at 456 Beach Rd, Miami, FL 33101, USA.
    """
})

print(result.name)  # "Alice Johnson"
print(result.company.name)  # "TechCorp"
print(len(result.addresses))  # 2
print(result.addresses[0].city)  # "San Francisco"
```

**验证与错误处理**：

```python
from pydantic import ValidationError

try:
    result = chain.invoke({
        "format_instructions": parser.get_format_instructions(),
        "input": "Invalid data"
    })
except ValidationError as e:
    print("Validation failed:")
    print(e.json())
    # [
    #   {
    #     "loc": ["age"],
    #     "msg": "field required",
    #     "type": "value_error.missing"
    #   }
    # ]
```

### 1.4.4 CommaSeparatedListOutputParser：列表解析

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Output a comma-separated list."),
    ("human", "List 5 {category}.")
])

chain = prompt | model | parser

result = chain.invoke({"category": "programming languages"})
print(type(result))  # <class 'list'>
print(result)  # ['Python', 'JavaScript', 'Java', 'C++', 'Go']

# 直接使用
for lang in result:
    print(f"- {lang}")
```

### 1.4.5 自定义 Output Parser

**场景**：解析特殊格式输出

```python
from langchain_core.output_parsers import BaseOutputParser
from typing import List
import re

class BulletPointParser(BaseOutputParser[List[str]]):
    """解析项目符号列表"""
    
    def parse(self, text: str) -> List[str]:
        """从文本中提取项目符号列表"""
        # 匹配 "- item" 或 "* item" 格式
        pattern = r'^[\-\*]\s+(.+)$'
        lines = text.split('\n')
        items = []
        
        for line in lines:
            match = re.match(pattern, line.strip())
            if match:
                items.append(match.group(1))
        
        return items
    
    @property
    def _type(self) -> str:
        return "bullet_point_parser"

# 使用
parser = BulletPointParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Output a bullet-point list using '-' or '*'."),
    ("human", "List benefits of exercise.")
])

chain = prompt | model | parser

result = chain.invoke({})
print(result)
# ['Improves cardiovascular health', 'Reduces stress', 'Increases energy']
```

**处理解析失败**：

```python
from langchain_core.output_parsers import OutputParserException

class SafeBulletPointParser(BaseOutputParser[List[str]]):
    """带错误处理的解析器"""
    
    def parse(self, text: str) -> List[str]:
        items = []
        for line in text.split('\n'):
            match = re.match(r'^[\-\*]\s+(.+)$', line.strip())
            if match:
                items.append(match.group(1))
        
        if not items:
            raise OutputParserException(
                f"No bullet points found in output: {text}",
                llm_output=text
            )
        
        return items

# 使用
try:
    result = chain.invoke({})
except OutputParserException as e:
    print(f"Parsing failed: {e}")
    # 可以重试或使用备用方案
```

---

## 1.5 Message 与 Conversation

> **核心概念**：Message 是 LangChain 对话系统的基础，理解消息类型对于构建多轮对话至关重要。

### 1.5.1 消息类型系统

<div data-component="MessageFlowDiagram"></div>

#### 完整消息类型

```python
from langchain_core.messages import (
    BaseMessage,       # 抽象基类
    SystemMessage,     # 系统指令
    HumanMessage,      # 用户输入
    AIMessage,         # AI 回复
    ToolMessage,       # 工具调用结果
    ChatMessage,       # 自定义角色
    FunctionMessage    # 已废弃，使用 ToolMessage
)

# 1. SystemMessage：设定 AI 行为
sys_msg = SystemMessage(content="You are a helpful assistant.")

# 2. HumanMessage：用户输入
human_msg = HumanMessage(content="What is LangChain?")

# 3. AIMessage：AI 回复（带元数据）
ai_msg = AIMessage(
    content="LangChain is a framework for building LLM applications.",
    additional_kwargs={"model": "gpt-4o", "finish_reason": "stop"}
)

# 4. ToolMessage：工具调用结果
tool_msg = ToolMessage(
    content="Temperature: 72°F",
    tool_call_id="call_abc123"
)

# 5. ChatMessage：自定义角色
custom_msg = ChatMessage(
    content="I am a custom role.",
    role="narrator"
)
```

#### 消息属性

```python
# 所有消息共有属性
msg = HumanMessage(content="Hello")

print(msg.content)       # "Hello"
print(msg.type)          # "human"
print(msg.additional_kwargs)  # {}

# AIMessage 特有属性
ai_msg = AIMessage(
    content="Hi there!",
    response_metadata={
        "token_usage": {"total_tokens": 20},
        "model_name": "gpt-4o"
    }
)

print(ai_msg.response_metadata)
```

### 1.5.2 对话历史管理

#### 简单对话类

```python
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

class SimpleConversation:
    """简单对话管理器"""
    
    def __init__(self, system_message: str, model_name: str = "gpt-4o-mini"):
        self.messages: List[BaseMessage] = [
            SystemMessage(content=system_message)
        ]
        self.model = ChatOpenAI(model=model_name)
    
    def add_user_message(self, content: str):
        """添加用户消息"""
        self.messages.append(HumanMessage(content=content))
    
    def add_ai_message(self, content: str):
        """添加 AI 消息"""
        self.messages.append(AIMessage(content=content))
    
    def get_messages(self) -> List[BaseMessage]:
        """获取所有消息"""
        return self.messages
    
    def chat(self, user_input: str) -> str:
        """发送消息并获取回复"""
        self.add_user_message(user_input)
        response = self.model.invoke(self.messages)
        self.add_ai_message(response.content)
        return response.content
    
    def clear(self, keep_system: bool = True):
        """清除历史（可选保留系统消息）"""
        if keep_system:
            self.messages = self.messages[:1]
        else:
            self.messages = []
    
    def get_history(self) -> str:
        """格式化显示对话历史"""
        lines = []
        for msg in self.messages:
            role = msg.type.capitalize()
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

# 使用
conv = SimpleConversation("You are a Python tutor.")

response1 = conv.chat("How do I sort a list?")
print(response1)
# "You can use the sorted() function or the .sort() method..."

response2 = conv.chat("What's the difference?")
print(response2)
# "sorted() returns a new list, while .sort() modifies in-place..."

print("\n--- History ---")
print(conv.get_history())
# System: You are a Python tutor.
# Human: How do I sort a list?
# AI: You can use the sorted() function or the .sort() method...
# Human: What's the difference?
# AI: sorted() returns a new list, while .sort() modifies in-place...
```

### 1.5.3 消息过滤与转换

#### 限制消息历史长度

```python
def trim_messages(
    messages: List[BaseMessage],
    max_tokens: int = 2000,
    keep_system: bool = True
) -> List[BaseMessage]:
    """
    保留最近的消息，确保不超过 token 限制
    
    Args:
        messages: 消息列表
        max_tokens: 最大 token 数
        keep_system: 是否保留系统消息
    
    Returns:
        修剪后的消息列表
    """
    from langchain.text_splitter import TokenTextSplitter
    
    # 简化版：保留最后 N 条消息
    max_messages = 20
    
    if keep_system and len(messages) > 0 and isinstance(messages[0], SystemMessage):
        system_msg = [messages[0]]
        other_messages = messages[1:]
        
        if len(other_messages) > max_messages:
            return system_msg + other_messages[-max_messages:]
        else:
            return messages
    else:
        if len(messages) > max_messages:
            return messages[-max_messages:]
        else:
            return messages

# 使用
class ConversationWithTrim(SimpleConversation):
    def chat(self, user_input: str) -> str:
        self.add_user_message(user_input)
        
        # 修剪历史
        self.messages = trim_messages(self.messages)
        
        response = self.model.invoke(self.messages)
        self.add_ai_message(response.content)
        return response.content
```

#### 消息格式转换

```python
def messages_to_openai_format(messages: List[BaseMessage]) -> List[dict]:
    """转换为 OpenAI API 格式"""
    return [
        {
            "role": msg.type if msg.type != "ai" else "assistant",
            "content": msg.content
        }
        for msg in messages
    ]

# 使用
api_messages = messages_to_openai_format(conv.get_messages())
print(api_messages)
# [
#   {"role": "system", "content": "You are a Python tutor."},
#   {"role": "user", "content": "How do I sort a list?"},
#   {"role": "assistant", "content": "You can use sorted()..."}
# ]
```

---

## 1.6 高级主题

### 1.6.1 RunnableConfig 深度解析

**完整配置结构**：

```python
from langchain_core.runnables import RunnableConfig
from langchain.callbacks import StdOutCallbackHandler

config = RunnableConfig(
    # Callbacks：监控与日志
    callbacks=[StdOutCallbackHandler()],
    
    # Tags：任务分类
    tags=["production", "translation", "urgent"],
    
    # Metadata：自定义元数据
    metadata={
        "user_id": "user_123",
        "session_id": "session_abc",
        "environment": "production",
        "version": "1.0.0"
    },
    
    # Run Name：运行标识
    run_name="translate_hello_to_french",
    
    # Concurrency：并发控制
    max_concurrency=5,  # 最多同时5个请求
    
    # Recursion Limit：递归深度
    recursion_limit=25,
    
    # Configurable：动态配置
    configurable={
        "model": "gpt-4o",
        "temperature": 0.7
    }
)
```

**在链中传递**：

```python
# 配置自动传递给所有组件
result = chain.invoke(
    {"text": "Hello"},
    config=config
)

# 每个组件都会收到相同的 config
```

### 1.6.2 自定义 Runnable

**场景**：实现复杂的自定义逻辑

```python
from langchain_core.runnables import Runnable, RunnableConfig
from typing import Any, Iterator

class RetryableRunnable(Runnable):
    """带重试机制的 Runnable"""
    
    def __init__(self, runnable: Runnable, max_retries: int = 3):
        self.runnable = runnable
        self.max_retries = max_retries
    
    def invoke(self, input: Any, config: RunnableConfig = None) -> Any:
        """同步调用（带重试）"""
        for attempt in range(self.max_retries):
            try:
                return self.runnable.invoke(input, config)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(2 ** attempt)  # 指数退避
    
    async def ainvoke(self, input: Any, config: RunnableConfig = None) -> Any:
        """异步调用（带重试）"""
        import asyncio
        for attempt in range(self.max_retries):
            try:
                return await self.runnable.ainvoke(input, config)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                await asyncio.sleep(2 ** attempt)

# 使用
model = ChatOpenAI(model="gpt-4o-mini")
retryable_model = RetryableRunnable(model, max_retries=3)

chain = prompt | retryable_model | parser
```

### 1.6.3 性能优化技巧

#### 1. 批量处理优化

```python
# ❌ 不推荐：逐个调用
results = []
for item in large_dataset:
    results.append(chain.invoke(item))

# ✅ 推荐：批量调用
batch_size = 20
results = []
for i in range(0, len(large_dataset), batch_size):
    batch = large_dataset[i:i+batch_size]
    results.extend(chain.batch(batch))
```

#### 2. 异步并发

```python
import asyncio

# ✅ 推荐：异步并发
async def process_all(items):
    tasks = [chain.ainvoke(item) for item in items]
    return await asyncio.gather(*tasks)

results = asyncio.run(process_all(large_dataset))
```

#### 3. 缓存策略

```python
from langchain.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache

# 内存缓存
set_llm_cache(InMemoryCache())

# 持久化缓存
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# 相同输入会直接返回缓存结果
model = ChatOpenAI(model="gpt-4o-mini", cache=True)
```

---

## 🎯 本章小结

### 核心要点回顾

| 组件 | 核心概念 | 关键方法/类 |
|------|----------|-------------|
| **Runnable** | 统一接口 | invoke, ainvoke, stream, astream, batch, abatch |
| **Language Models** | 模型调用 | ChatOpenAI, ChatAnthropic, ChatOllama |
| **Prompt Templates** | 提示管理 | PromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate |
| **Output Parsers** | 结构化输出 | StrOutputParser, JsonOutputParser, PydanticOutputParser |
| **Message** | 消息系统 | SystemMessage, HumanMessage, AIMessage |

### 掌握检查清单

完成本章学习后，你应该能够：

- [ ] **Runnable 协议**
  - [ ] 解释 Runnable 设计的动机
  - [ ] 选择合适的调用方法（invoke/stream/batch/async）
  - [ ] 使用 `|` 操作符组合多个 Runnable
  - [ ] 理解 RunnableConfig 的作用

- [ ] **Language Models**
  - [ ] 区分 Chat Model 和 LLM
  - [ ] 在 OpenAI、Anthropic、Ollama 之间切换
  - [ ] 配置 temperature、max_tokens 等参数
  - [ ] 实现自定义 Callback

- [ ] **Prompt Templates**
  - [ ] 创建 PromptTemplate 和 ChatPromptTemplate
  - [ ] 使用变量注入和部分填充
  - [ ] 实现 Few-Shot 提示
  - [ ] 从 LangChain Hub 拉取提示

- [ ] **Output Parsers**
  - [ ] 使用 PydanticOutputParser 实现类型安全
  - [ ] 处理 JSON 输出
  - [ ] 自定义解析器

- [ ] **Message 管理**
  - [ ] 理解不同消息类型的用途
  - [ ] 实现简单对话历史管理
  - [ ] 修剪和转换消息列表

### 练习题

#### 练习 1：性能对比实验

对比 `invoke()` 和 `batch()` 处理 100 条消息的耗时。

```python
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="gpt-4o-mini")
messages = [[HumanMessage(content=f"Say {i}")] for i in range(100)]

# TODO: 实现对比实验
```

#### 练习 2：智能模型选择器

实现一个函数，根据输入长度自动选择模型。

```python
def smart_model_selector(text: str) -> ChatOpenAI:
    """
    根据文本长度选择模型：
    - 短文本（<100 字符）：gpt-4o-mini
    - 长文本（>=100 字符）：gpt-4o
    """
    # TODO: 实现逻辑
    pass
```

#### 练习 3：结构化数据提取

使用 PydanticOutputParser 从文本中提取书籍信息。

```python
from pydantic import BaseModel, Field

class Book(BaseModel):
    title: str = Field(description="Book title")
    author: str = Field(description="Author name")
    year: int = Field(description="Publication year")
    genre: str = Field(description="Book genre")

# TODO: 实现提取链
text = "1984 by George Orwell, published in 1949, is a dystopian novel."
```

#### 练习 4：对话持久化

扩展 `SimpleConversation` 类，添加保存和加载功能。

```python
class PersistentConversation(SimpleConversation):
    def save_to_file(self, filepath: str):
        """保存对话历史到文件"""
        # TODO: 实现
        pass
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "PersistentConversation":
        """从文件加载对话历史"""
        # TODO: 实现
        pass
```

### 下一章预告

**Chapter 2: 简单链构建入门**

在下一章中，我们将学习如何使用 LCEL（LangChain Expression Language）构建实用的应用：
- 翻译链：多语言翻译系统
- 摘要链：智能文档摘要
- 问答链：基于上下文的问答
- 错误处理：重试、降级与日志

---

## 📚 扩展阅读

### 官方文档

- [Runnable 接口文档](https://python.langchain.com/docs/concepts/runnables) - 官方 Runnable 协议详解
- [Chat Models 集成](https://python.langchain.com/docs/integrations/chat/) - 支持的模型提供商列表
- [Prompt Templates 指南](https://python.langchain.com/docs/concepts/prompt_templates) - 提示模板完整教程
- [Output Parsers 详解](https://python.langchain.com/docs/concepts/output_parsers) - 输出解析器参考
- [Message 类型参考](https://python.langchain.com/api_reference/core/messages.html) - 消息API文档

### 进阶资源

- [LangChain Hub](https://smith.langchain.com/hub) - 提示词管理平台
- [LangSmith 文档](https://docs.smith.langchain.com/) - 可观测性与评估
- [Pydantic 文档](https://docs.pydantic.dev/) - 数据验证库
- [OpenAI API 文档](https://platform.openai.com/docs/api-reference) - OpenAI 官方API
- [Anthropic Claude 文档](https://docs.anthropic.com/) - Claude API 文档

### 社区资源

- [LangChain GitHub Discussions](https://github.com/langchain-ai/langchain/discussions) - 社区讨论
- [LangChain Discord](https://discord.gg/langchain) - 实时交流
- [LangChain Blog](https://blog.langchain.dev/) - 官方博客

---


> **本章目标**：深入理解 LangChain 的底层抽象机制，掌握 Runnable 协议、Language Models、Prompt Templates、Output Parsers 等核心组件的使用方法。

---

## 本章导览

本章深入剖析 LangChain 的核心抽象层，这些概念是理解整个框架的关键：

- **Runnable 协议**：统一的接口标准，支持同步/异步调用、流式处理、批处理等多种执行模式
- **语言模型集成**：掌握 Chat Models 与 LLMs 的区别，学习模型切换与配置最佳实践
- **提示工程**：从简单模板到 Few-Shot 学习，系统化管理提示词资产
- **输出解析**：结构化提取 LLM 响应，实现类型安全的数据处理
- **消息抽象**：理解 SystemMessage、HumanMessage、AIMessage 的设计与应用场景

这些基础组件是构建所有 LangChain 应用的基石，务必扎实掌握。

---

## 1.1 Runnable 协议

Runnable 是 LangChain 中所有可执行组件的统一接口，它定义了一套标准化的调用方法，使得不同组件可以无缝组合。

### 1.1.1 统一接口：invoke()、stream()、batch()、astream()

<div data-component="RunnableProtocolVisualizer"></div>

**Runnable 协议的核心方法**：

```python
from langchain_core.runnables import Runnable
from typing import Any, Iterator, AsyncIterator

class Runnable(ABC):
    """所有可执行组件的基类"""
    
    def invoke(self, input: Any, config: RunnableConfig = None) -> Any:
        """同步调用，阻塞直到结果返回"""
        pass
    
    async def ainvoke(self, input: Any, config: RunnableConfig = None) -> Any:
        """异步调用，非阻塞"""
        pass
    
    def stream(self, input: Any, config: RunnableConfig = None) -> Iterator[Any]:
        """同步流式输出，逐块返回"""
        pass
    
    async def astream(self, input: Any, config: RunnableConfig = None) -> AsyncIterator[Any]:
        """异步流式输出"""
        pass
    
    def batch(self, inputs: list[Any], config: RunnableConfig = None) -> list[Any]:
        """批量处理"""
        pass
    
    async def abatch(self, inputs: list[Any], config: RunnableConfig = None) -> list[Any]:
        """异步批量处理"""
        pass
```

**实际示例**：

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="gpt-4o-mini")
message = HumanMessage(content="Count from 1 to 5.")

# 1. 同步调用
response = model.invoke([message])
print(response.content)
# 输出: 1, 2, 3, 4, 5

# 2. 流式输出
for chunk in model.stream([message]):
    print(chunk.content, end="", flush=True)
# 输出: 1, 2, 3, 4, 5 (逐字显示)

# 3. 批量处理
messages_batch = [
    [HumanMessage(content="Say 'Hi'")],
    [HumanMessage(content="Say 'Hello'")],
    [HumanMessage(content="Say 'Hey'")]
]
responses = model.batch(messages_batch)
for resp in responses:
    print(resp.content)
# 输出: Hi / Hello / Hey

# 4. 异步调用
import asyncio

async def async_example():
    response = await model.ainvoke([message])
    print(response.content)

asyncio.run(async_example())
```

**方法选择指南**：

| 方法 | 使用场景 | 优势 | 劣势 |
|------|----------|------|------|
| `invoke()` | 简单脚本、单次调用 | 代码简洁 | 阻塞主线程 |
| `ainvoke()` | Web 后端、并发场景 | 高效并发 | 需要异步上下文 |
| `stream()` | 聊天界面、实时反馈 | 用户体验好 | 处理复杂 |
| `astream()` | 异步流式场景 | 高性能流式 | 最复杂 |
| `batch()` | 数据处理、批量任务 | 节省请求次数 | 内存占用大 |

### 1.1.2 Runnable 实现类

**常用 Runnable 实现**：

```python
from langchain_core.runnables import (
    RunnableLambda,      # 包装任意函数
    RunnablePassthrough, # 透传输入
    RunnableParallel,    # 并行执行
    RunnableBranch,      # 条件分支
)

# 1. RunnableLambda：包装普通函数
def add_prefix(text: str) -> str:
    return f"Translated: {text}"

prefix_runnable = RunnableLambda(add_prefix)
result = prefix_runnable.invoke("Bonjour")
print(result)  # "Translated: Bonjour"

# 2. RunnablePassthrough：透传输入（常用于调试）
from langchain_core.runnables import RunnablePassthrough

passthrough = RunnablePassthrough()
print(passthrough.invoke({"key": "value"}))  # {"key": "value"}

# 3. RunnableParallel：并行执行多个任务
from langchain_core.prompts import ChatPromptTemplate

chain1 = ChatPromptTemplate.from_template("Translate to French: {text}") | model
chain2 = ChatPromptTemplate.from_template("Translate to Spanish: {text}") | model

parallel = RunnableParallel(
    french=chain1,
    spanish=chain2
)

result = parallel.invoke({"text": "Hello"})
# {'french': AIMessage(content='Bonjour'), 
#  'spanish': AIMessage(content='Hola')}
```

**组合模式**：

```python
# 管道组合（顺序执行）
chain = prompt | model | parser

# 等价于
from langchain_core.runnables import RunnableSequence
chain = RunnableSequence(prompt, model, parser)

# 数学表示
# f(x) = parser(model(prompt(x)))
```

### 1.1.3 与 Python 生态的互操作性

**Runnable 可以直接与 Python 函数互操作**：

```python
# Python 函数自动转为 Runnable
def uppercase(text: str) -> str:
    return text.upper()

# 直接用于链中
chain = prompt | model | uppercase | parser

# 或显式包装
chain = prompt | model | RunnableLambda(uppercase) | parser
```

**类型标注与 IDE 支持**：

```python
from langchain_core.runnables import Runnable

def create_chain() -> Runnable[dict, str]:
    """返回类型标注：输入 dict，输出 str"""
    return prompt | model | parser

# IDE 会自动推断类型
chain = create_chain()
result: str = chain.invoke({"text": "test"})  # 类型检查通过
```

---

## 1.2 Prompt Templates

<div data-component="PromptTemplateBuilder"></div>

### 1.2.1 基础模板用法

PromptTemplate 是 LangChain 中用于管理和复用提示的核心组件。

**两种模型接口**：

| 特性 | LLM | ChatModel |
|------|-----|-----------|
| **输入格式** | 字符串 | 消息列表 |
| **输出格式** | 字符串 | AIMessage |
| **适用模型** | 旧式模型（GPT-3） | 现代对话模型（GPT-4、Claude） |
| **推荐使用** | ❌ 已过时 | ✅ 优先使用 |

**LLM 示例**（不推荐）：

```python
from langchain_openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct")
result = llm.invoke("Translate 'Hello' to French:")
print(result)  # "Bonjour"
```

**ChatModel 示例**（推荐）：

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

chat = ChatOpenAI(model="gpt-4o")

messages = [
    SystemMessage(content="You are a translator."),
    HumanMessage(content="Translate 'Hello' to French.")
]

result = chat.invoke(messages)
print(result.content)  # "Bonjour"
print(type(result))    # <class 'langchain_core.messages.ai.AIMessage'>
```

**为什么优先使用 ChatModel？**

1. **更好的上下文管理**：系统消息、用户消息、助手消息分离
2. **支持多轮对话**：消息列表天然支持对话历史
3. **模型对齐**：现代 LLM 都是对话模型训练
4. **功能完整**：支持 Function Calling、JSON Mode 等

### 1.2.2 模型提供商切换

**OpenAI**：

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=1000,
    timeout=30,
    max_retries=2,
    api_key="sk-...",  # 或从环境变量读取
    base_url="https://api.openai.com/v1"  # 支持代理
)
```

**Anthropic Claude**：

```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0.7,
    max_tokens=4096,
    api_key="sk-ant-..."
)
```

**本地模型（Ollama）**：

```python
from langchain_community.chat_models import ChatOllama

model = ChatOllama(
    model="llama3.2",
    temperature=0.7,
    base_url="http://localhost:11434"
)

# 需要先启动 Ollama：ollama serve
```

**统一接口**（推荐）：

```python
def get_model(provider: str = "openai"):
    """工厂模式创建模型"""
    if provider == "openai":
        return ChatOpenAI(model="gpt-4o")
    elif provider == "anthropic":
        return ChatAnthropic(model="claude-3-5-sonnet-20241022")
    elif provider == "ollama":
        return ChatOllama(model="llama3.2")
    else:
        raise ValueError(f"Unknown provider: {provider}")

# 使用
model = get_model("anthropic")
chain = prompt | model | parser
```

### 1.2.3 模型参数详解

**核心参数**：

```python
model = ChatOpenAI(
    # 必选参数
    model="gpt-4o",                    # 模型名称
    
    # 生成参数
    temperature=0.7,                   # 温度：0-2，越高越随机
    top_p=1.0,                         # 核采样：0-1
    frequency_penalty=0.0,             # 频率惩罚：-2 to 2
    presence_penalty=0.0,              # 存在惩罚：-2 to 2
    max_tokens=1000,                   # 最大生成 token 数
    
    # 连接参数
    timeout=30,                        # 超时时间（秒）
    max_retries=2,                     # 重试次数
    request_timeout=60,                # 单次请求超时
    
    # 流式参数
    streaming=True,                    # 启用流式
    
    # 其他
    model_kwargs={                     # 额外参数
        "seed": 42,                    # 随机种子
        "response_format": {"type": "json_object"}  # JSON 模式
    }
)
```

**参数效果对比**：

| 参数 | 低值 | 高值 | 使用场景 |
|------|------|------|----------|
| `temperature` | 0 (确定性) | 2 (创意性) | 0: 翻译、摘要；1: 创作、对话 |
| `top_p` | 0.1 (保守) | 1.0 (多样) | 与 temperature 配合 |
| `frequency_penalty` | 0 | 2 | 减少重复词汇 |
| `presence_penalty` | 0 | 2 | 鼓励新话题 |

**实验示例**：

```python
# Temperature 对比
prompt_text = "Write a creative story about a robot."

# 低温度（确定性）
model_deterministic = ChatOpenAI(model="gpt-4o", temperature=0)
result1 = model_deterministic.invoke([HumanMessage(content=prompt_text)])

# 高温度（创意性）
model_creative = ChatOpenAI(model="gpt-4o", temperature=1.5)
result2 = model_creative.invoke([HumanMessage(content=prompt_text)])

# 多次运行 result1 几乎一致，result2 每次不同
```

### 1.2.4 Callbacks 与日志

**Callbacks 机制**：

```python
from langchain.callbacks import StdOutCallbackHandler

model = ChatOpenAI(
    model="gpt-4o",
    callbacks=[StdOutCallbackHandler()],  # 标准输出回调
    verbose=True
)

result = model.invoke([HumanMessage(content="Hello")])

# 输出详细日志：
# > Entering new ChatOpenAI chain...
# > Prompt: [HumanMessage(content='Hello')]
# > Response: AIMessage(content='Hi there!')
# > Finished chain.
```

**自定义 Callback**：

```python
from langchain.callbacks.base import BaseCallbackHandler

class TokenCounterCallback(BaseCallbackHandler):
    """统计 token 使用量"""
    
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"🚀 Starting LLM with {len(prompts)} prompts")
    
    def on_llm_end(self, response, **kwargs):
        # 提取 token 使用信息
        usage = response.llm_output.get("token_usage", {})
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        print(f"📊 Tokens: {self.prompt_tokens} prompt + {self.completion_tokens} completion")

# 使用
counter = TokenCounterCallback()
model = ChatOpenAI(model="gpt-4o", callbacks=[counter])
model.invoke([HumanMessage(content="Hello")])
```

---

## 1.3 Prompt Templates

Prompt Templates 用于构建结构化、可复用的提示文本。

### 1.3.1 PromptTemplate 基础

**基础用法**：

```python
from langchain_core.prompts import PromptTemplate

# 方式1：from_template（推荐）
template = PromptTemplate.from_template(
    "Translate the following text to {language}: {text}"
)

# 方式2：构造函数
template = PromptTemplate(
    input_variables=["language", "text"],
    template="Translate the following text to {language}: {text}"
)

# 格式化
prompt = template.format(language="French", text="Hello")
print(prompt)
# "Translate the following text to French: Hello"

# 直接作为 Runnable 使用
result = template.invoke({"language": "Spanish", "text": "Goodbye"})
print(result)
# PromptValue(text="Translate the following text to Spanish: Goodbye")
```

**部分填充（Partial）**：

```python
# 预填充某些变量
template = PromptTemplate.from_template(
    "You are a {role}. {instruction}"
)

# 固定角色
assistant_template = template.partial(role="helpful assistant")

# 后续只需提供 instruction
result = assistant_template.invoke({"instruction": "Explain quantum physics."})
```

### 1.3.2 ChatPromptTemplate 与消息格式

**ChatPromptTemplate** 是为对话模型设计的模板。

```python
from langchain_core.prompts import ChatPromptTemplate

# 定义多角色模板
template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("human", "{user_input}"),
    ("ai", "I understand you want to know about {topic}."),
    ("human", "Yes, please explain.")
])

# 格式化
messages = template.invoke({
    "role": "science teacher",
    "user_input": "Tell me about photosynthesis",
    "topic": "photosynthesis"
})

print(messages)
# [
#   SystemMessage(content='You are a science teacher.'),
#   HumanMessage(content='Tell me about photosynthesis'),
#   AIMessage(content='I understand you want to know about photosynthesis.'),
#   HumanMessage(content='Yes, please explain.')
# ]
```

**消息类型**：

```python
from langchain_core.messages import (
    SystemMessage,    # 系统指令
    HumanMessage,     # 用户输入
    AIMessage,        # AI 回复
    FunctionMessage,  # 函数调用结果（已废弃）
    ToolMessage       # 工具调用结果（推荐）
)

# 直接构造
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is 2+2?"),
    AIMessage(content="2+2 equals 4."),
    HumanMessage(content="Thanks!")
]
```

### 1.3.3 变量注入与部分填充

**动态变量**：

```python
template = ChatPromptTemplate.from_messages([
    ("system", "Current date: {date}. You are a {role}."),
    ("human", "{input}")
])

# 使用 partial 填充日期
from datetime import datetime

template_with_date = template.partial(
    date=lambda: datetime.now().strftime("%Y-%m-%d")
)

# 每次调用时自动获取当前日期
result = template_with_date.invoke({
    "role": "assistant",
    "input": "What's the weather?"
})
```

**条件变量**：

```python
def get_system_prompt(user_level: str) -> str:
    """根据用户级别返回不同的系统提示"""
    prompts = {
        "beginner": "Explain in simple terms.",
        "expert": "Use technical terminology."
    }
    return prompts.get(user_level, prompts["beginner"])

template = ChatPromptTemplate.from_messages([
    ("system", "{system_instruction}"),
    ("human", "{question}")
])

# 动态系统提示
result = template.invoke({
    "system_instruction": get_system_prompt("expert"),
    "question": "How does TCP work?"
})
```

### 1.3.4 模板组合

**PipelinePromptTemplate**（多阶段提示）：

```python
from langchain_core.prompts import PipelinePromptTemplate

# 子模板
intro_template = PromptTemplate.from_template(
    "You are an expert in {domain}."
)

task_template = PromptTemplate.from_template(
    "{intro}\nTask: {task}"
)

# 组合
full_template = PipelinePromptTemplate(
    final_prompt=task_template,
    pipeline_prompts=[
        ("intro", intro_template)
    ]
)

result = full_template.invoke({
    "domain": "machine learning",
    "task": "Explain gradient descent"
})
```

---

## 1.4 Output Parsers

Output Parsers 将 LLM 的文本输出解析为结构化数据。

### 1.4.1 StrOutputParser：基础文本解析

```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

# 从 AIMessage 提取 content
ai_message = AIMessage(content="Hello, world!")
result = parser.invoke(ai_message)
print(result)  # "Hello, world!"

# 在链中使用
chain = prompt | model | StrOutputParser()
result = chain.invoke({"input": "Say hi"})
print(type(result))  # <class 'str'>
```

### 1.4.2 JsonOutputParser：结构化输出

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

# 提示模型输出 JSON
template = ChatPromptTemplate.from_messages([
    ("system", "Output your response as JSON."),
    ("human", "List 3 colors with their hex codes.")
])

chain = template | model | parser

result = chain.invoke({})
print(result)
# {'colors': [
#   {'name': 'red', 'hex': '#FF0000'},
#   {'name': 'green', 'hex': '#00FF00'},
#   {'name': 'blue', 'hex': '#0000FF'}
# ]}
```

### 1.4.3 PydanticOutputParser：类型安全

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 定义数据模型
class Person(BaseModel):
    name: str = Field(description="Person's name")
    age: int = Field(description="Person's age")
    occupation: str = Field(description="Person's job")

parser = PydanticOutputParser(pydantic_object=Person)

# 获取格式说明
format_instructions = parser.get_format_instructions()
print(format_instructions)
# Output your response as JSON matching this schema:
# {"name": "string", "age": "integer", "occupation": "string"}

# 在提示中使用
template = ChatPromptTemplate.from_messages([
    ("system", "Extract person information.\n{format_instructions}"),
    ("human", "{input}")
])

chain = template | model | parser

result = chain.invoke({
    "format_instructions": format_instructions,
    "input": "John is a 30-year-old engineer."
})

print(result)
# Person(name='John', age=30, occupation='engineer')
print(type(result))  # <class '__main__.Person'>
```

### 1.4.4 CommaSeparatedListOutputParser：列表解析

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

template = ChatPromptTemplate.from_messages([
    ("system", "Output a comma-separated list."),
    ("human", "List 5 programming languages.")
])

chain = template | model | parser

result = chain.invoke({})
print(result)
# ['Python', 'JavaScript', 'Java', 'C++', 'Go']
print(type(result))  # <class 'list'>
```

---

## 1.5 Message 与 Conversation

### 1.5.1 消息类型

<div data-component="MessageFlowDiagram"></div>

**完整消息类型**：

```python
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    ChatMessage
)

# 1. SystemMessage：系统指令
sys_msg = SystemMessage(content="You are a helpful assistant.")

# 2. HumanMessage：用户输入
human_msg = HumanMessage(content="What is LangChain?")

# 3. AIMessage：AI 回复
ai_msg = AIMessage(
    content="LangChain is a framework...",
    additional_kwargs={"model": "gpt-4o"}
)

# 4. ToolMessage：工具调用结果
tool_msg = ToolMessage(
    content="Search result: ...",
    tool_call_id="call_123"
)

# 5. ChatMessage：自定义角色
custom_msg = ChatMessage(
    content="...",
    role="custom_role"
)
```

### 1.5.2 消息历史管理

```python
from langchain_core.messages import BaseMessage

class SimpleConversation:
    """简单对话管理"""
    
    def __init__(self, system_message: str):
        self.messages: list[BaseMessage] = [
            SystemMessage(content=system_message)
        ]
    
    def add_user_message(self, content: str):
        self.messages.append(HumanMessage(content=content))
    
    def add_ai_message(self, content: str):
        self.messages.append(AIMessage(content=content))
    
    def get_messages(self) -> list[BaseMessage]:
        return self.messages
    
    def clear(self):
        self.messages = self.messages[:1]  # 保留系统消息

# 使用
conv = SimpleConversation("You are a coding assistant.")
conv.add_user_message("How do I sort a list in Python?")

# 调用模型
response = model.invoke(conv.get_messages())
conv.add_ai_message(response.content)

conv.add_user_message("What about in reverse order?")
response = model.invoke(conv.get_messages())
```

### 1.5.3 消息转换与过滤

**限制消息历史长度**：

```python
def trim_messages(messages: list[BaseMessage], max_tokens: int = 2000) -> list[BaseMessage]:
    """保留最近的消息，确保不超过 token 限制"""
    from langchain_openai import ChatOpenAI
    
    # 简化版：保留最后 N 条消息
    max_messages = 10
    if len(messages) > max_messages:
        return [messages[0]] + messages[-max_messages:]  # 保留系统消息
    return messages

# 使用
trimmed = trim_messages(conv.get_messages())
```

**消息格式转换**：

```python
def messages_to_dict(messages: list[BaseMessage]) -> list[dict]:
    """转换为 OpenAI API 格式"""
    return [
        {
            "role": msg.type,
            "content": msg.content
        }
        for msg in messages
    ]

# 使用
api_format = messages_to_dict(conv.get_messages())
```

---

## 🎯 本章小结

**核心要点**：

1. **Runnable 协议**：invoke、stream、batch、ainvoke 四种调用方式
2. **Language Models**：优先使用 ChatModel，支持多提供商切换
3. **Prompt Templates**：ChatPromptTemplate 用于对话，支持变量注入
4. **Output Parsers**：StrOutputParser、JsonOutputParser、PydanticOutputParser
5. **Message 管理**：SystemMessage、HumanMessage、AIMessage，手动管理历史

**掌握检查**：

- [ ] 能解释 Runnable 协议的设计意义
- [ ] 能切换不同模型提供商（OpenAI、Anthropic、Ollama）
- [ ] 能使用 ChatPromptTemplate 构建多轮对话
- [ ] 能用 PydanticOutputParser 解析结构化输出
- [ ] 能实现简单的对话历史管理

**练习题**：

1. **性能对比**：对比 `invoke()` 和 `batch()` 处理 100 条消息的耗时
2. **模型切换**：实现一个函数，根据输入长度自动选择模型（短文本用 gpt-4o-mini，长文本用 gpt-4o）
3. **结构化提取**：用 PydanticOutputParser 从文本中提取书籍信息（标题、作者、出版年份）
4. **对话记忆**：扩展 SimpleConversation 类，添加 `save_to_file()` 和 `load_from_file()` 方法

**下一章预告**：

Chapter 2 将学习如何用 LCEL 构建简单链，包括翻译链、摘要链、问答链等常见模式。

---

## 📚 扩展阅读

- [Runnable 接口文档](https://python.langchain.com/docs/concepts/runnables)
- [Chat Models 对比](https://python.langchain.com/docs/integrations/chat/)
- [Prompt Templates 指南](https://python.langchain.com/docs/concepts/prompt_templates)
- [Output Parsers 详解](https://python.langchain.com/docs/concepts/output_parsers)
- [Message 类型参考](https://python.langchain.com/api_reference/core/messages.html)

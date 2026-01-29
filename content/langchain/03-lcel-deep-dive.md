> **本章目标**：深入理解 LCEL 的底层机制，掌握 Runnable 高级操作、配置化、Fallback、Retry 等企业级特性。

---

## 本章导览

本章深入 LCEL 的高级特性与底层原理，提升应用的健壮性与性能：

- **组合数学**：理解 Pipe 操作符背后的函数组合原理 `f₄(f₃(f₂(f₁(x))))`
- **高级操作**：RunnablePassthrough、RunnableLambda、RunnableBranch 等灵活组合技巧
- **配置化开发**：通过 `configurable_fields` 和 `configurable_alternatives` 实现动态配置
- **容错机制**：Fallback 降级、Retry 重试、超时控制等生产环境必备特性
- **性能优化**：并行执行、批处理、缓存等提升吞吐量的实战技巧

掌握这些高级技术，你将能够构建企业级的健壮 LLM 应用。

---

## 3.1 Pipe 与组合

### 3.1.1 链式调用的数学基础

LCEL 的管道操作符基于**函数组合**（Function Composition）的数学概念。

**数学定义**:

$$
(f \circ g)(x) = f(g(x))
$$

在 LCEL 中:

$$
\text{chain} = f_n \circ f_{n-1} \circ \cdots \circ f_2 \circ f_1
$$

```python
# 数学表达式
# chain(x) = parser(model(prompt(x)))

# LCEL 表达式
chain = prompt | model | parser

# 等价于
def chain(x):
    return parser(model(prompt(x)))
```

<div data-component="RunnableCompositionFlow"></div>

### 组合演示

<div data-component="ParallelExecutionDemo"></div>

**组合性质**:

1. **结合律**: `(f | g) | h ≡ f | (g | h)`
2. **类型安全**: `f: A → B` 和 `g: B → C` 才能组合为 `f | g: A → C`

```python
from langchain_core.runnables import Runnable

# 类型检查示例
prompt: Runnable[dict, PromptValue] 
model: Runnable[PromptValue, AIMessage]
parser: Runnable[AIMessage, str]

# 组合后类型自动推断
chain: Runnable[dict, str] = prompt | model | parser
```

### 3.1.2 类型传递与自动推断

```python
from typing import TypedDict

class Input(TypedDict):
    text: str
    language: str

class Output(TypedDict):
    translation: str
    original: str

# 显式类型标注
def create_typed_chain() -> Runnable[Input, Output]:
    prompt = ChatPromptTemplate.from_template("Translate to {language}: {text}")
    model = ChatOpenAI(model="gpt-4o-mini")
    
    # 使用 RunnablePassthrough 保留原文
    from langchain_core.runnables import RunnablePassthrough
    
    chain = (
        RunnablePassthrough.assign(
            translation=prompt | model | StrOutputParser()
        )
        | (lambda x: {"translation": x["translation"], "original": x["text"]})
    )
    
    return chain

# IDE 自动提示类型
typed_chain = create_typed_chain()
result: Output = typed_chain.invoke({"text": "Hello", "language": "French"})
```

### 3.1.3 RunnableSequence 内部实现

```python
from langchain_core.runnables import RunnableSequence

# 管道操作符的底层实现
class RunnableSequence(Runnable):
    def __init__(self, *steps: Runnable):
        self.steps = steps
    
    def invoke(self, input, config=None):
        result = input
        for step in self.steps:
            result = step.invoke(result, config)
        return result
    
    def stream(self, input, config=None):
        # 只有最后一个组件流式输出
        result = input
        for step in self.steps[:-1]:
            result = step.invoke(result, config)
        
        for chunk in self.steps[-1].stream(result, config):
            yield chunk

# 使用
chain = RunnableSequence(prompt, model, parser)
# 等价于
chain = prompt | model | parser
```

---

## 3.2 Runnable 高级操作

### 3.2.1 RunnablePassthrough：透传输入

**用途**: 在链中保留原始输入。

```python
from langchain_core.runnables import RunnablePassthrough

# 基础透传
passthrough = RunnablePassthrough()
print(passthrough.invoke({"key": "value"}))  # {"key": "value"}

# 在链中使用
chain = (
    {"original": RunnablePassthrough(), "processed": some_chain}
)

result = chain.invoke("input")
# {'original': 'input', 'processed': <处理后的结果>}
```

**实际案例**:

```python
# 保留原文的翻译链
translation_chain = (
    RunnablePassthrough.assign(
        translation=ChatPromptTemplate.from_template("Translate to French: {text}")
        | model
        | StrOutputParser()
    )
)

result = translation_chain.invoke({"text": "Hello"})
# {'text': 'Hello', 'translation': 'Bonjour'}
```

### 3.2.2 RunnableLambda：自定义函数包装

```python
from langchain_core.runnables import RunnableLambda

# 包装普通函数
def add_prefix(text: str) -> str:
    return f"[TRANSLATED] {text}"

prefix_runnable = RunnableLambda(add_prefix)

chain = prompt | model | StrOutputParser() | prefix_runnable

result = chain.invoke({"text": "Hello", "language": "French"})
# "[TRANSLATED] Bonjour"
```

**异步函数**:

```python
import asyncio

async def async_process(text: str) -> str:
    await asyncio.sleep(0.1)  # 模拟异步操作
    return text.upper()

async_runnable = RunnableLambda(async_process)

# 支持异步调用
result = await async_runnable.ainvoke("hello")  # "HELLO"
```

### 3.2.3 RunnableBranch：条件分支

```python
from langchain_core.runnables import RunnableBranch

# 根据条件选择不同链
def is_long_text(x: dict) -> bool:
    return len(x.get("text", "")) > 100

summarize_chain = ChatPromptTemplate.from_template("Summarize: {text}") | model
direct_chain = RunnablePassthrough()

branch = RunnableBranch(
    (is_long_text, summarize_chain),    # 条件1: 长文本→摘要
    direct_chain                        # 默认: 直接透传
)

# 短文本
result1 = branch.invoke({"text": "Hi"})  # 透传

# 长文本
long_text = "a" * 150
result2 = branch.invoke({"text": long_text})  # 摘要
```

### 3.2.4 RunnableParallel：并行执行

```python
from langchain_core.runnables import RunnableParallel

# 并行执行多个链
parallel_chain = RunnableParallel(
    french=ChatPromptTemplate.from_template("Translate to French: {text}") | model,
    spanish=ChatPromptTemplate.from_template("Translate to Spanish: {text}") | model,
    german=ChatPromptTemplate.from_template("Translate to German: {text}") | model
)

result = parallel_chain.invoke({"text": "Hello"})
# {
#   'french': AIMessage(content='Bonjour'),
#   'spanish': AIMessage(content='Hola'),
#   'german': AIMessage(content='Hallo')
# }
```

**字典语法糖**:

```python
# 使用字典（更简洁）
parallel_chain = {
    "french": prompt_fr | model,
    "spanish": prompt_es | model
}
```

### 3.2.5 RunnableMap：字典映射

```python
# 映射输入到多个键
from langchain_core.runnables import RunnableParallel

chain = RunnableParallel({
    "uppercase": RunnableLambda(lambda x: x.upper()),
    "lowercase": RunnableLambda(lambda x: x.lower()),
    "length": RunnableLambda(lambda x: len(x))
})

result = chain.invoke("Hello World")
# {'uppercase': 'HELLO WORLD', 'lowercase': 'hello world', 'length': 11}
```

---

## 3.3 配置化（Configurable）

### 3.3.1 ConfigurableField：动态参数

```python
from langchain_core.runnables import ConfigurableField

# 可配置的模型
model = ChatOpenAI(model="gpt-4o-mini").configurable_fields(
    temperature=ConfigurableField(
        id="llm_temperature",
        name="LLM Temperature",
        description="Controls randomness"
    ),
    model=ConfigurableField(
        id="llm_model",
        name="LLM Model"
    )
)

chain = prompt | model | parser

# 运行时配置
result1 = chain.invoke(
    {"text": "Hello"},
    config={"configurable": {"llm_temperature": 0}}
)

result2 = chain.invoke(
    {"text": "Hello"},
    config={"configurable": {"llm_temperature": 1.5}}
)
```

### 3.3.2 ConfigurableAlternatives：模型切换

```python
from langchain_core.runnables import ConfigurableFieldAlternatives
from langchain_anthropic import ChatAnthropic

# 可切换的模型
model = ChatOpenAI(model="gpt-4o").configurable_alternatives(
    ConfigurableField(id="llm"),
    default_key="openai",
    anthropic=ChatAnthropic(model="claude-3-5-sonnet-20241022"),
    local=ChatOllama(model="llama3.2")
)

chain = prompt | model | parser

# 使用 OpenAI (默认)
result1 = chain.invoke({"text": "Hello"})

# 切换到 Anthropic
result2 = chain.invoke(
    {"text": "Hello"},
    config={"configurable": {"llm": "anthropic"}}
)

# 切换到本地模型
result3 = chain.invoke(
    {"text": "Hello"},
    config={"configurable": {"llm": "local"}}
)
```

### 3.3.3 运行时配置（RunnableConfig）

```python
from langchain_core.runnables import RunnableConfig

# 创建配置
config = RunnableConfig(
    tags=["translation", "production"],
    metadata={"user_id": "12345"},
    callbacks=[...],
    max_concurrency=5
)

result = chain.invoke({"text": "Hello"}, config=config)
```

### 3.3.4 with_config() 方法

```python
# 预配置链
production_chain = chain.with_config({
    "tags": ["production"],
    "metadata": {"env": "prod"},
    "configurable": {"llm_temperature": 0}
})

# 后续调用自动使用配置
result = production_chain.invoke({"text": "Hello"})
```

---

## 3.4 Fallback 与容错

### 3.4.1 with_fallbacks()：失败降级

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# 主模型
primary = ChatOpenAI(model="gpt-4o")

# 备用模型
fallbacks = [
    ChatOpenAI(model="gpt-4o-mini"),      # 备用1: 更便宜的模型
    ChatAnthropic(model="claude-3-5-sonnet-20241022"),  # 备用2: 不同提供商
    ChatOllama(model="llama3.2")          # 备用3: 本地模型
]

# 带降级的链
chain = prompt | primary.with_fallbacks(fallbacks) | parser

# 自动降级
try:
    result = chain.invoke({"text": "Hello"})
except Exception:
    # 如果所有降级都失败才抛出异常
    pass
```

### 3.4.2 多级 Fallback 策略

```python
# 完整链的降级
primary_chain = prompt_complex | gpt4 | parser
fallback_chain = prompt_simple | gpt3 | parser
last_resort = RunnableLambda(lambda x: "Translation unavailable")

full_chain = primary_chain.with_fallbacks([
    fallback_chain,
    last_resort
])
```

### 3.4.3 异常处理与日志记录

```python
from langchain.callbacks import StdOutCallbackHandler

class FallbackLogger(StdOutCallbackHandler):
    def on_chain_error(self, error, **kwargs):
        print(f"Primary chain failed: {error}")
        print("Trying fallback...")

chain = primary.with_fallbacks(
    fallbacks=[fallback],
    callbacks=[FallbackLogger()]
)
```

---

## 3.5 Retry 重试机制

### 3.5.1 with_retry()：自动重试

```python
from langchain_core.runnables import Runnable

# 自动重试（默认最多3次）
model_with_retry = model.with_retry()

chain = prompt | model_with_retry | parser

# 遇到临时错误自动重试
result = chain.invoke({"text": "Hello"})
```

### 3.5.2 指数退避（Exponential Backoff）

```python
# 自定义重试策略
model_with_retry = model.with_retry(
    stop_after_attempt=5,              # 最多5次
    wait_exponential_multiplier=1,     # 初始等待1秒
    wait_exponential_max=60,           # 最多等待60秒
    retry_if_exception_type=(RateLimitError,)  # 只重试特定错误
)

# 重试间隔: 1s, 2s, 4s, 8s, 16s (上限60s)
```

### 3.5.3 重试条件自定义

```python
from openai import APIError

def should_retry(error: Exception) -> bool:
    """自定义重试逻辑"""
    if isinstance(error, RateLimitError):
        return True
    if isinstance(error, APIError) and "timeout" in str(error):
        return True
    return False

model_with_custom_retry = model.with_retry(
    retry_if_exception=should_retry,
    stop_after_attempt=3
)
```

<InteractiveComponent name="FallbackPathSimulator" />
<InteractiveComponent name="RetryTimeline" />

---

## 🎯 本章小结

**核心要点**:

1. **函数组合**: LCEL 基于数学函数组合,具有结合律和类型安全
2. **高级操作**: RunnablePassthrough、RunnableLambda、RunnableBranch、RunnableParallel
3. **配置化**: ConfigurableField 和 ConfigurableAlternatives 实现运行时配置
4. **容错机制**: with_fallbacks() 实现多级降级
5. **重试策略**: with_retry() 支持指数退避和自定义条件

**掌握检查**:

- [ ] 能解释 LCEL 的函数组合本质
- [ ] 能使用 RunnableParallel 并行执行任务
- [ ] 能用 ConfigurableAlternatives 实现模型切换
- [ ] 能配置多级 Fallback 策略
- [ ] 能自定义重试条件

**练习题**:

1. **并行翻译**: 用 RunnableParallel 同时翻译到5种语言,测量总耗时
2. **智能路由**: 根据文本长度选择不同模型（<100字用mini,>100字用gpt-4）
3. **容错链**: 实现主模型→备用模型→本地模型的三级降级
4. **自定义重试**: 对 RateLimitError 重试5次,对其他错误立即失败

**下一章预告**:

Chapter 4 将学习流式处理与批处理,包括 stream()、astream()、batch()、异步编程等。

---

## 📚 扩展阅读

- [LCEL Runnable 接口](https://python.langchain.com/docs/concepts/runnables)
- [配置化文档](https://python.langchain.com/docs/how_to/configure)
- [Fallback 指南](https://python.langchain.com/docs/how_to/fallbacks)
- [Retry 策略](https://python.langchain.com/docs/how_to/retry)

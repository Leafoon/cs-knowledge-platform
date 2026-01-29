> **本章目标**：掌握 LCEL（LangChain Expression Language）的基本用法，学会构建翻译链、摘要链、问答链等常见模式，并掌握链的调试与错误处理技巧。

---

## 本章导览

本章从实战角度出发，教你使用 LCEL 构建生产级应用链：

- **LCEL vs Legacy**：对比新旧写法，理解为何官方强烈推荐 LCEL 作为标准开发范式
- **经典链模式**：翻译链、摘要链、问答链等高频场景的标准实现模板
- **调试技巧**：使用 LangSmith、verbose 模式、日志等工具快速定位问题
- **错误处理**：Retry、Fallback、超时控制等企业级可靠性保障机制
- **可视化调试**：通过 LangSmith Trace 理解链的执行流程与性能瓶颈

通过本章学习，你将能够独立构建稳定可靠的 LLM 应用链。

---

## 2.1 Legacy Chain vs LCEL

在 LangChain 早期版本中，开发者使用 `LLMChain`、`SequentialChain` 等类来构建应用。从 v0.1.0 开始，官方推荐使用 **LCEL（LangChain Expression Language）** 替代这些旧式 Chain。

### 2.1.1 LLMChain（已废弃）回顾

**旧式写法**（不推荐）：

```python
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 旧式 LLMChain
prompt = ChatPromptTemplate.from_template("Translate to {language}: {text}")
model = ChatOpenAI(model="gpt-4o-mini")

chain = LLMChain(
    llm=model,
    prompt=prompt,
    verbose=True
)

result = chain.run(language="French", text="Hello")
# 警告: LLMChain is deprecated, use LCEL instead
```

**问题**：
1. 需要记忆不同 Chain 类的 API（LLMChain、SequentialChain、TransformChain...）
2. 组合复杂链时代码冗长
3. 类型推断困难
4. 性能优化受限

### 2.1.2 为什么迁移到 LCEL？

**LCEL 的优势**：

| 特性 | Legacy Chain | LCEL |
|------|--------------|------|
| **语法** | 类实例化 | 管道操作符 `\|` |
| **类型推断** | ❌ 弱 | ✅ 强（IDE 支持） |
| **流式支持** | ❌ 部分支持 | ✅ 原生支持 |
| **并行执行** | ❌ 需手动 | ✅ RunnableParallel |
| **调试** | verbose=True | get_graph()、LangSmith |
| **性能** | ⚠️ 一般 | ✅ 优化的执行引擎 |

**设计哲学**：

$$
\text{LCEL} = \text{Functional Programming} + \text{Runnable Protocol}
$$

LCEL 将链视为函数的组合，每个组件都实现 Runnable 接口，通过管道操作符串联。

### 2.1.3 迁移指南与对比示例

<div data-component="LegacyVsLCELComparison"></div>

**迁移对比**：

```python
# ❌ 旧式写法
from langchain.chains import LLMChain

chain = LLMChain(llm=model, prompt=prompt)
result = chain.run(text="Hello", language="French")

# ✅ LCEL 写法
chain = prompt | model | StrOutputParser()
result = chain.invoke({"text": "Hello", "language": "French"})
```

**复杂链迁移**：

```python
# ❌ 旧式 SequentialChain
from langchain.chains import SequentialChain

chain1 = LLMChain(llm=model, prompt=prompt1, output_key="translation")
chain2 = LLMChain(llm=model, prompt=prompt2, output_key="summary")

sequential = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["text"],
    output_variables=["translation", "summary"]
)

# ✅ LCEL 写法
from langchain_core.runnables import RunnablePassthrough

chain = (
    {"text": RunnablePassthrough()}
    | prompt1 | model | StrOutputParser()
    | (lambda x: {"translation": x})
    | RunnablePassthrough.assign(
        summary=prompt2 | model | StrOutputParser()
    )
)
```

---

## 2.2 第一条 LCEL 链

### 2.2.1 Pipe 操作符（|）的魔力

**管道操作符** `|` 是 LCEL 的核心语法，它连接不同的 Runnable 组件。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 定义组件
prompt = ChatPromptTemplate.from_template("Translate to French: {text}")
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# 用管道连接
chain = prompt | model | parser

# 调用
result = chain.invoke({"text": "Hello, world!"})
print(result)  # "Bonjour, le monde!"
```

**执行流程**：

```
输入: {"text": "Hello, world!"}
  │
  ▼
prompt.invoke({"text": "Hello, world!"})
  │
  ▼
ChatPromptValue([HumanMessage(content="Translate to French: Hello, world!")])
  │
  ▼
model.invoke([HumanMessage(...)])
  │
  ▼
AIMessage(content="Bonjour, le monde!")
  │
  ▼
parser.invoke(AIMessage(...))
  │
  ▼
输出: "Bonjour, le monde!"
```

**数学表示**：

$$
\text{chain}(x) = \text{parser}(\text{model}(\text{prompt}(x)))
$$

### 2.2.2 Prompt → Model → Parser 基础模式

这是 LCEL 中最常见的模式：

```python
# 模式模板
chain = (
    prompt_template    # 生成提示
    | language_model   # LLM 处理
    | output_parser    # 解析输出
)
```

**实际示例**：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 1. 提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful translator."),
    ("human", "Translate '{text}' to {language}.")
])

# 2. 语言模型
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3. 输出解析器
parser = StrOutputParser()

# 4. 组合
translation_chain = prompt | model | parser

# 5. 使用
result = translation_chain.invoke({
    "text": "Good morning",
    "language": "Spanish"
})
print(result)  # "Buenos días"
```

### 2.2.3 链的类型标注与 IDE 支持

**类型标注**：

```python
from langchain_core.runnables import Runnable

# 明确输入输出类型
translation_chain: Runnable[dict, str] = prompt | model | parser

# IDE 会自动提示
result: str = translation_chain.invoke({"text": "Hi", "language": "French"})
```

**自定义类型**：

```python
from typing import TypedDict

class TranslationInput(TypedDict):
    text: str
    language: str

def create_translation_chain() -> Runnable[TranslationInput, str]:
    prompt = ChatPromptTemplate.from_template("Translate to {language}: {text}")
    model = ChatOpenAI(model="gpt-4o-mini")
    parser = StrOutputParser()
    return prompt | model | parser

# 使用
chain = create_translation_chain()
result = chain.invoke({"text": "Hello", "language": "German"})
```

---

## 2.3 常见简单链模式

### 2.3.0 错误处理策略

<div data-component="ErrorHandlingFlow"></div>

### 2.3.1 翻译链（Translation Chain）

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_translator(target_language: str):
    """创建翻译器"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a professional translator. Translate all inputs to {target_language}."),
        ("human", "{text}")
    ])
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()
    
    return prompt | model | parser

# 使用
french_translator = create_translator("French")
spanish_translator = create_translator("Spanish")

print(french_translator.invoke({"text": "Hello"}))    # "Bonjour"
print(spanish_translator.invoke({"text": "Hello"}))   # "Hola"
```

**多语言翻译**：

```python
from langchain_core.runnables import RunnableParallel

# 并行翻译到多种语言
multi_translator = RunnableParallel(
    french=create_translator("French"),
    spanish=create_translator("Spanish"),
    german=create_translator("German")
)

result = multi_translator.invoke({"text": "Good morning"})
print(result)
# {
#   'french': 'Bonjour',
#   'spanish': 'Buenos días',
#   'german': 'Guten Morgen'
# }
```

### 2.3.2 摘要链（Summarization Chain）

```python
def create_summarizer(max_words: int = 50):
    """创建摘要生成器"""
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text in {max_words} words or less:\n\n{text}"
    )
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    parser = StrOutputParser()
    
    chain = prompt | model | parser
    
    # 部分填充 max_words
    return chain.partial(max_words=max_words)

# 使用
summarizer = create_summarizer(max_words=30)

long_text = """
LangChain is a framework for developing applications powered by language models. 
It enables applications that are context-aware and reason about their actions. 
The framework consists of several parts: LangChain Libraries, LangChain Templates, 
LangServe, LangSmith, and LangChain Hub.
"""

summary = summarizer.invoke({"text": long_text})
print(summary)
# "LangChain is a framework for building context-aware, reasoning language model 
#  applications, comprising Libraries, Templates, LangServe, LangSmith, and Hub."
```

**分级摘要**：

```python
from langchain_core.runnables import RunnablePassthrough

# 两级摘要：先摘要到100词，再摘要到20词
two_level_summary = (
    create_summarizer(max_words=100)
    | (lambda x: {"text": x})
    | create_summarizer(max_words=20)
)

result = two_level_summary.invoke({"text": long_text})
```

### 2.3.3 问答链（QA Chain）

```python
def create_qa_chain():
    """创建问答链"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question based on the context provided. "
                   "If you cannot answer, say 'I don't know'."),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])
    
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    parser = StrOutputParser()
    
    return prompt | model | parser

# 使用
qa_chain = create_qa_chain()

context = """
LangChain was created by Harrison Chase in October 2022. 
It is an open-source framework that helps developers build LLM applications.
"""

answer = qa_chain.invoke({
    "context": context,
    "question": "Who created LangChain?"
})
print(answer)  # "Harrison Chase created LangChain."

answer = qa_chain.invoke({
    "context": context,
    "question": "When was LangChain released?"
})
print(answer)  # "LangChain was created in October 2022."
```

### 2.3.4 实体提取链（Entity Extraction）

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class ExtractedEntities(BaseModel):
    """实体数据模型"""
    people: list[str] = Field(description="List of people mentioned")
    organizations: list[str] = Field(description="List of organizations")
    locations: list[str] = Field(description="List of locations")

def create_entity_extractor():
    """创建实体提取器"""
    parser = PydanticOutputParser(pydantic_object=ExtractedEntities)
    
    prompt = ChatPromptTemplate.from_template(
        "Extract entities from the following text.\n"
        "{format_instructions}\n\n"
        "Text: {text}"
    )
    
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    
    chain = (
        prompt.partial(format_instructions=parser.get_format_instructions())
        | model
        | parser
    )
    
    return chain

# 使用
extractor = create_entity_extractor()

text = """
Elon Musk announced that Tesla will open a new factory in Berlin, Germany. 
The company plans to hire 10,000 employees in the next year.
"""

entities = extractor.invoke({"text": text})
print(entities)
# ExtractedEntities(
#     people=['Elon Musk'],
#     organizations=['Tesla'],
#     locations=['Berlin', 'Germany']
# )
```

---

## 2.4 链的调试与检查

### 2.4.1 get_graph()：查看链结构

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Translate to French: {text}")
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

chain = prompt | model | parser

# 获取链的图结构
graph = chain.get_graph()
print(graph.draw_ascii())
```

**输出**：

```
           +--------------+              
           | PromptInput  |              
           +--------------+              
                   *                     
                   *                     
                   *                     
         +--------------------+          
         | ChatPromptTemplate |          
         +--------------------+          
                   *                     
                   *                     
                   *                     
           +--------------+              
           | ChatOpenAI   |              
           +--------------+              
                   *                     
                   *                     
                   *                     
        +---------------------+          
        | StrOutputParser     |          
        +---------------------+          
                   *                     
                   *                     
                   *                     
          +----------------+             
          | StrOutputParser |             
          +----------------+             
```

<div data-component="ChainGraphVisualizer"></div>

### 2.4.2 verbose=True：详细日志

```python
# 启用详细日志
chain = prompt | model.with_config({"verbose": True}) | parser

result = chain.invoke({"text": "Hello"})

# 输出：
# [chain/start] Entering Chain
# [chat_model/start] Entering ChatOpenAI
# [chat_model/end] ChatOpenAI output: AIMessage(content="Bonjour")
# [chain/end] Chain output: "Bonjour"
```

**自定义回调**：

```python
from langchain.callbacks import StdOutCallbackHandler

callback = StdOutCallbackHandler()

result = chain.invoke(
    {"text": "Hello"},
    config={"callbacks": [callback]}
)
```

### 2.4.3 LangSmith Tracing 初探

**启用 LangSmith**：

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__your_api_key"
os.environ["LANGCHAIN_PROJECT"] = "my-first-project"

# 此后所有链调用自动追踪
result = chain.invoke({"text": "Hello"})

# 在 https://smith.langchain.com 查看追踪
```

**查看追踪信息**：
- 输入/输出
- 延迟时间
- Token 消耗
- 错误堆栈
- 嵌套调用关系

---

## 2.5 错误处理基础

### 2.5.1 try-except 包装

```python
try:
    result = chain.invoke({"text": "Hello"})
    print(result)
except Exception as e:
    print(f"Error: {e}")
    # 记录日志、返回默认值等
```

**常见错误**：

```python
from openai import AuthenticationError, RateLimitError

try:
    result = chain.invoke({"text": "Hello"})
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded, retry later")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### 2.5.2 Fallback 机制预览

**with_fallbacks()** 在主链失败时切换到备用链。

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# 主模型
primary_model = ChatOpenAI(model="gpt-4o")

# 备用模型
fallback_model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# 带降级的链
chain = (
    prompt 
    | primary_model.with_fallbacks([fallback_model])
    | parser
)

# 如果 GPT-4 失败，自动切换到 Claude
result = chain.invoke({"text": "Hello"})
```

### 2.5.3 重试策略

**with_retry()** 自动重试失败的调用。

```python
from langchain_core.runnables import RunnableRetry

# 自动重试（最多3次）
model_with_retry = model.with_retry(
    stop_after_attempt=3,
    wait_exponential_multiplier=1,  # 指数退避
    wait_exponential_max=10
)

chain = prompt | model_with_retry | parser

# 遇到临时错误会自动重试
result = chain.invoke({"text": "Hello"})
```

**自定义重试条件**：

```python
def should_retry(error: Exception) -> bool:
    """只对特定错误重试"""
    return isinstance(error, RateLimitError)

model_with_custom_retry = model.with_retry(
    retry_if_exception=should_retry,
    stop_after_attempt=5
)
```

---

## 🎯 本章小结

**核心要点**：

1. **LCEL 优于 Legacy Chain**：语法简洁、类型安全、性能更好
2. **管道操作符**：`prompt | model | parser` 是最基础的模式
3. **常见链**：翻译链、摘要链、问答链、实体提取链
4. **调试工具**：get_graph()、verbose、LangSmith
5. **错误处理**：try-except、fallbacks、retry

**掌握检查**：

- [ ] 能解释 LCEL 相比 Legacy Chain 的优势
- [ ] 能用 LCEL 构建翻译链和摘要链
- [ ] 能使用 PydanticOutputParser 提取结构化数据
- [ ] 能用 get_graph() 查看链结构
- [ ] 能配置 fallback 和 retry 机制

**练习题**：

1. **多步骤链**：构建一个链，先翻译文本到法语，再对法语文本进行摘要
2. **条件执行**：根据输入语言自动选择目标语言（英语→法语，法语→英语）
3. **批量处理**：用 batch() 方法同时翻译 10 条消息，测量总耗时
4. **错误恢复**：实现一个带重试和降级的翻译链，主模型失败时切换到备用模型

**下一章预告**：

Chapter 3 将深入 LCEL 的高级特性，包括 RunnablePassthrough、RunnableParallel、配置化、Fallback、Retry 等。

---

## 📚 扩展阅读

- [LCEL 完整指南](https://python.langchain.com/docs/concepts/lcel)
- [从 Legacy Chain 迁移](https://python.langchain.com/docs/versions/migrating_chains/)
- [常见链模式](https://python.langchain.com/docs/how_to/)
- [LangSmith 追踪](https://docs.smith.langchain.com/observability/how_to_guides/tracing)
- [错误处理最佳实践](https://python.langchain.com/docs/how_to/fallbacks)

> **本章目标**：深入理解对话记忆系统的设计原理，掌握 ConversationBufferMemory、ConversationSummaryMemory、VectorStoreMemory 等多种记忆类型，构建具备上下文感知能力的对话应用。

---

## 本章导览

本章聚焦于赋予 LLM 应用"记忆"能力，这是构建自然对话体验的核心：

- **记忆基础**：理解为什么 LLM 需要记忆、记忆的存储与检索机制
- **Buffer Memory**：最简单的完整历史记忆，适合短对话
- **Summary Memory**：压缩历史为摘要，节省 token 开销
- **Vector Store Memory**：语义检索历史消息，适合长对话
- **Entity Memory**：跟踪对话中的实体信息（人物、地点、事件等）
- **Window Memory**：滑动窗口记忆，平衡成本与上下文
- **多记忆组合**：结合多种记忆策略，构建复杂对话系统

掌握这些技术将让你的对话应用能够记住用户信息、维持话题连贯性、提供个性化体验。

---

## 9.1 为什么 LLM 需要记忆？

### 9.1.1 LLM 的"无状态"特性

大语言模型本质上是无状态的：每次调用都是独立的，模型不会"记住"之前的对话。

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4")

# 第一次对话
response1 = model.invoke("My name is Alice")
print(response1.content)
# "Hello Alice! How can I help you today?"

# 第二次对话（模型不记得 Alice）
response2 = model.invoke("What's my name?")
print(response2.content)
# "I don't have access to your personal information..."
```

### 9.1.2 记忆的核心作用

记忆系统解决三大问题：

1. **上下文连续性**：记住之前说了什么，避免重复询问
2. **个性化体验**：记住用户偏好、历史行为
3. **多轮推理**：基于历史信息进行复杂推理

**带记忆的对话**：

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# 创建记忆
memory = ConversationBufferMemory()

# 创建对话链
conversation = ConversationChain(
    llm=model,
    memory=memory,
    verbose=True
)

# 第一轮
print(conversation.predict(input="My name is Alice"))
# "Hello Alice! How can I help you today?"

# 第二轮（记住了名字）
print(conversation.predict(input="What's my name?"))
# "Your name is Alice."
```

### 9.1.3 记忆系统架构

```
┌─────────────┐
│ 用户输入    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│ 1. 加载记忆（Load）     │
│    - 从存储中读取历史   │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│ 2. 构建提示（Format）   │
│    - 将历史插入提示词   │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│ 3. LLM 生成（Generate） │
└──────┬──────────────────┘
       │
       ▼
┌─────────────────────────┐
│ 4. 保存记忆（Save）     │
│    - 用户输入 + LLM响应 │
└─────────────────────────┘
```

<div data-component="MemoryEvolutionTimeline"></div>

---

## 9.2 ConversationBufferMemory

### 9.2.1 基本用法

最简单的记忆类型，保存完整对话历史。

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()

# 添加消息
memory.save_context(
    {"input": "Hi, I'm Alice"},
    {"output": "Hello Alice! How can I help you?"}
)

memory.save_context(
    {"input": "I like Python programming"},
    {"output": "That's great! Python is a powerful language."}
)

# 查看历史
print(memory.load_memory_variables({}))
# {
#   'history': 'Human: Hi, I'm Alice\nAI: Hello Alice! How can I help you?\n
#               Human: I like Python programming\nAI: That's great! Python is a powerful language.'
# }
```

### 9.2.2 与链集成

```python
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""The following is a conversation between a human and an AI.

{history}
Human: {input}
AI:"""
)

memory = ConversationBufferMemory()

chain = LLMChain(
    llm=model,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# 使用
print(chain.predict(input="My favorite color is blue"))
# "That's a lovely choice! Blue is calming and peaceful."

print(chain.predict(input="What's my favorite color?"))
# "Your favorite color is blue."
```

### 9.2.3 返回消息对象

默认返回字符串，也可以返回消息列表。

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)

memory.save_context(
    {"input": "Hello"},
    {"output": "Hi there!"}
)

print(memory.load_memory_variables({}))
# {
#   'history': [
#       HumanMessage(content='Hello'),
#       AIMessage(content='Hi there!')
#   ]
# }
```

### 9.2.4 自定义键名

```python
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    output_key="answer"
)

memory.save_context(
    {"question": "What is LangChain?"},
    {"answer": "LangChain is a framework for building LLM applications."}
)

print(memory.load_memory_variables({}))
# {'chat_history': '...'}
```

---

## 9.3 ConversationSummaryMemory

### 9.3.1 为什么需要摘要记忆？

完整历史会快速消耗 token，特别是长对话。摘要记忆将历史压缩为简洁摘要。

```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=model)

# 添加多轮对话
memory.save_context(
    {"input": "Hi, my name is Alice and I'm a software engineer"},
    {"output": "Hello Alice! It's great to meet a software engineer."}
)

memory.save_context(
    {"input": "I've been working on a Python project for 2 years"},
    {"output": "That's impressive! Two years is a significant commitment."}
)

memory.save_context(
    {"input": "The project is about machine learning"},
    {"output": "Machine learning is such an exciting field!"}
)

# 查看摘要
print(memory.load_memory_variables({}))
# {
#   'history': 'Alice is a software engineer who has been working on 
#               a Python machine learning project for 2 years.'
# }
```

### 9.3.2 摘要 vs 完整历史

```python
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory

# 完整历史
buffer_memory = ConversationBufferMemory()

# 摘要记忆
summary_memory = ConversationSummaryMemory(llm=model)

# 添加相同对话
conversations = [
    ({"input": "My name is Bob"}, {"output": "Hello Bob!"}),
    ({"input": "I live in New York"}, {"output": "New York is a great city!"}),
    ({"input": "I work as a teacher"}, {"output": "Teaching is a noble profession."}),
]

for conv in conversations:
    buffer_memory.save_context(*conv)
    summary_memory.save_context(*conv)

print("Buffer Memory:")
print(buffer_memory.load_memory_variables({}))
# 完整对话（约 100 tokens）

print("\nSummary Memory:")
print(summary_memory.load_memory_variables({}))
# "Bob is a teacher living in New York." (约 10 tokens)
```

### 9.3.3 增量摘要更新

```python
memory = ConversationSummaryMemory(llm=model)

# 第一次摘要
memory.save_context(
    {"input": "I like reading books"},
    {"output": "That's wonderful!"}
)
print(memory.buffer)
# "The human enjoys reading books."

# 第二次摘要（基于之前的摘要更新）
memory.save_context(
    {"input": "Especially science fiction novels"},
    {"output": "Sci-fi is fascinating!"}
)
print(memory.buffer)
# "The human enjoys reading books, particularly science fiction novels."
```

---

## 9.4 ConversationBufferWindowMemory

### 9.4.1 滑动窗口记忆

只保留最近 N 轮对话，平衡成本与上下文。

```python
from langchain.memory import ConversationBufferWindowMemory

# 只保留最近 2 轮对话
memory = ConversationBufferWindowMemory(k=2)

memory.save_context({"input": "Message 1"}, {"output": "Response 1"})
memory.save_context({"input": "Message 2"}, {"output": "Response 2"})
memory.save_context({"input": "Message 3"}, {"output": "Response 3"})
memory.save_context({"input": "Message 4"}, {"output": "Response 4"})

print(memory.load_memory_variables({}))
# 只包含 Message 3 和 Message 4
```

### 9.4.2 动态调整窗口大小

```python
memory = ConversationBufferWindowMemory(k=3)

# 根据上下文长度动态调整
def adaptive_window(memory, max_tokens=1000):
    history = memory.load_memory_variables({})["history"]
    token_count = len(history.split())  # 简化计数
    
    if token_count > max_tokens:
        memory.k = max(1, memory.k - 1)  # 缩小窗口
    else:
        memory.k = min(10, memory.k + 1)  # 扩大窗口
    
    return memory.k

# 使用
new_k = adaptive_window(memory)
print(f"Adjusted window size: {new_k}")
```

---

## 9.5 ConversationSummaryBufferMemory

### 9.5.1 混合策略

结合摘要和窗口：保留最近消息的完整内容，旧消息压缩为摘要。

```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=model,
    max_token_limit=100  # 超过 100 tokens 开始摘要
)

# 添加多轮对话
conversations = [
    ({"input": "I'm planning a trip to Japan"}, {"output": "That sounds exciting!"}),
    ({"input": "I'll visit Tokyo and Kyoto"}, {"output": "Great choices!"}),
    ({"input": "I love sushi"}, {"output": "You'll find amazing sushi there!"}),
    ({"input": "When is the best time to visit?"}, {"output": "Spring or autumn is ideal."}),
]

for conv in conversations:
    memory.save_context(*conv)

print(memory.load_memory_variables({}))
# {
#   'history': '[摘要] The human is planning a trip to Japan, visiting Tokyo and Kyoto...\n
#               Human: When is the best time to visit?\n
#               AI: Spring or autumn is ideal.'
# }
```

### 9.5.2 自定义摘要时机

```python
memory = ConversationSummaryBufferMemory(
    llm=model,
    max_token_limit=50,
    moving_summary_buffer=""  # 初始摘要
)

# 查看摘要触发
for i in range(10):
    memory.save_context(
        {"input": f"Message {i}"},
        {"output": f"Response {i}"}
    )
    
    print(f"Round {i}: {len(memory.buffer)} messages")
    if memory.moving_summary_buffer:
        print(f"Summary: {memory.moving_summary_buffer[:50]}...")
```

---

## 9.6 VectorStoreMemory

### 9.6.1 语义检索记忆

使用向量数据库存储历史，根据语义相似度检索相关对话。

```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 创建向量存储
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(
    ["Initial context"],
    embedding=embeddings
)

# 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 创建记忆
memory = VectorStoreRetrieverMemory(retriever=retriever)

# 添加对话
memory.save_context(
    {"input": "My favorite programming language is Python"},
    {"output": "Python is great for data science and web development."}
)

memory.save_context(
    {"input": "I also like JavaScript"},
    {"output": "JavaScript is essential for web development."}
)

memory.save_context(
    {"input": "I'm learning machine learning"},
    {"output": "Machine learning is an exciting field!"}
)

# 语义检索
relevant = memory.load_memory_variables({"prompt": "What languages do I like?"})
print(relevant)
# 返回与"languages"相关的对话（Python, JavaScript）
```

### 9.6.2 混合检索策略

```python
# 结合关键词和语义检索
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever

# BM25 检索器（关键词）
bm25_retriever = BM25Retriever.from_texts([...])

# 向量检索器（语义）
vector_retriever = vectorstore.as_retriever()

# 混合检索
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

memory = VectorStoreRetrieverMemory(retriever=ensemble_retriever)
```

---

## 9.7 EntityMemory

### 9.7.1 实体跟踪

专门跟踪对话中提到的实体（人物、地点、组织等）。

```python
from langchain.memory import ConversationEntityMemory

memory = ConversationEntityMemory(llm=model)

memory.save_context(
    {"input": "My friend Alice works at Google in New York"},
    {"output": "That's interesting! Google's NYC office is impressive."}
)

memory.save_context(
    {"input": "Alice loves machine learning"},
    {"output": "Machine learning is a great field to be in!"}
)

# 查看实体
print(memory.entity_store)
# {
#   'Alice': 'Friend of the user, works at Google, interested in machine learning',
#   'Google': 'Company where Alice works, has office in New York',
#   'New York': 'Location of Google office'
# }

# 加载特定实体信息
print(memory.load_memory_variables({"input": "Tell me about Alice"}))
# 包含 Alice 相关的所有信息
```

### 9.7.2 实体更新

```python
memory = ConversationEntityMemory(llm=model)

# 第一次提及
memory.save_context(
    {"input": "John is a teacher"},
    {"output": "Teaching is rewarding!"}
)

print(memory.entity_store.get("John"))
# "John is a teacher"

# 第二次提及（更新信息）
memory.save_context(
    {"input": "John teaches mathematics at MIT"},
    {"output": "MIT is a prestigious institution!"}
)

print(memory.entity_store.get("John"))
# "John is a mathematics teacher at MIT"
```

<div data-component="MemoryTypeComparison"></div>

---

## 9.8 持久化记忆

### 9.8.1 保存到文件

```python
from langchain.memory import FileChatMessageHistory
from langchain.memory import ConversationBufferMemory

# 使用文件存储
chat_history = FileChatMessageHistory("chat_history.json")

memory = ConversationBufferMemory(
    chat_memory=chat_history,
    return_messages=True
)

# 使用记忆
memory.save_context({"input": "Hello"}, {"output": "Hi there!"})

# 记忆会自动保存到 chat_history.json
```

### 9.8.2 Redis 存储

```python
from langchain.memory import RedisChatMessageHistory

# 连接 Redis
chat_history = RedisChatMessageHistory(
    url="redis://localhost:6379",
    session_id="user_123"
)

memory = ConversationBufferMemory(chat_memory=chat_history)

# 使用
memory.save_context({"input": "Test"}, {"output": "Response"})
```

### 9.8.3 数据库存储

```python
from langchain.memory import SQLChatMessageHistory

# 使用 SQLite
chat_history = SQLChatMessageHistory(
    connection_string="sqlite:///chat_history.db",
    session_id="user_123"
)

memory = ConversationBufferMemory(chat_memory=chat_history)
```

---

## 9.9 多记忆组合

### 9.9.1 组合不同记忆类型

```python
from langchain.memory import CombinedMemory

# 短期记忆（最近 3 轮）
short_term = ConversationBufferWindowMemory(
    k=3,
    memory_key="short_term_history"
)

# 长期记忆（摘要）
long_term = ConversationSummaryMemory(
    llm=model,
    memory_key="long_term_summary"
)

# 组合
combined_memory = CombinedMemory(memories=[short_term, long_term])

# 使用
combined_memory.save_context(
    {"input": "Hello"},
    {"output": "Hi there!"}
)

print(combined_memory.load_memory_variables({}))
# {
#   'short_term_history': '...',
#   'long_term_summary': '...'
# }
```

### 9.9.2 实战：智能客服记忆系统

```python
from langchain.memory import CombinedMemory, ConversationBufferWindowMemory
from langchain.memory import ConversationEntityMemory, VectorStoreRetrieverMemory

# 1. 最近对话（窗口记忆）
recent_memory = ConversationBufferWindowMemory(
    k=5,
    memory_key="recent"
)

# 2. 用户实体信息
entity_memory = ConversationEntityMemory(
    llm=model,
    memory_key="entities"
)

# 3. 历史问题检索（向量记忆）
vector_memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(),
    memory_key="relevant_history"
)

# 组合
customer_service_memory = CombinedMemory(
    memories=[recent_memory, entity_memory, vector_memory]
)

# 提示模板
prompt = PromptTemplate(
    input_variables=["recent", "entities", "relevant_history", "input"],
    template="""You are a customer service AI.

Recent conversation:
{recent}

Customer information:
{entities}

Relevant past interactions:
{relevant_history}

Customer: {input}
AI:"""
)

# 创建链
chain = LLMChain(
    llm=model,
    prompt=prompt,
    memory=customer_service_memory
)

# 使用
response = chain.predict(input="I need help with my order")
print(response)
```

<div data-component="EntityMemoryGraph"></div>

---

## 9.10 记忆最佳实践

### 9.10.1 选择合适的记忆类型

| 场景 | 推荐记忆类型 | 理由 |
|------|-------------|------|
| 短对话（<10轮） | ConversationBufferMemory | 简单高效 |
| 长对话（>50轮） | ConversationSummaryMemory | 节省 token |
| 需要个性化 | ConversationEntityMemory | 跟踪用户信息 |
| 历史检索 | VectorStoreMemory | 语义搜索 |
| 实时聊天 | ConversationBufferWindowMemory | 平衡性能 |
| 复杂场景 | CombinedMemory | 多策略组合 |

### 9.10.2 内存管理策略

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """计算 token 数量"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

class AdaptiveMemory:
    """自适应记忆管理"""
    
    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
        self.buffer_memory = ConversationBufferMemory()
        self.summary_memory = ConversationSummaryMemory(llm=model)
        
    def save_context(self, inputs, outputs):
        # 保存到 buffer
        self.buffer_memory.save_context(inputs, outputs)
        
        # 检查 token 数量
        history = self.buffer_memory.load_memory_variables({})["history"]
        token_count = count_tokens(history)
        
        if token_count > self.max_tokens:
            # 切换到摘要记忆
            print(f"Token limit exceeded ({token_count}), switching to summary...")
            self.summary_memory.save_context(inputs, outputs)
            self.buffer_memory.clear()  # 清空 buffer
```

### 9.10.3 隐私与安全

```python
import re

class PrivacyProtectedMemory:
    """隐私保护记忆"""
    
    def __init__(self, base_memory):
        self.base_memory = base_memory
        
    def _redact_sensitive_info(self, text: str) -> str:
        """脱敏处理"""
        # 移除邮箱
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        # 移除电话
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        # 移除信用卡
        text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD]', text)
        return text
    
    def save_context(self, inputs, outputs):
        # 脱敏后保存
        clean_inputs = {k: self._redact_sensitive_info(v) for k, v in inputs.items()}
        clean_outputs = {k: self._redact_sensitive_info(v) for k, v in outputs.items()}
        self.base_memory.save_context(clean_inputs, clean_outputs)
```

---

## 本章小结

本章深入学习了对话记忆系统：

✅ **记忆基础**：理解 LLM 无状态特性，记忆系统的必要性与架构  
✅ **Buffer Memory**：完整历史记忆，适合短对话  
✅ **Summary Memory**：压缩历史为摘要，节省 token  
✅ **Window Memory**：滑动窗口，平衡成本与上下文  
✅ **Vector Store Memory**：语义检索，适合长对话  
✅ **Entity Memory**：跟踪实体信息，提供个性化  
✅ **持久化**：文件、Redis、数据库存储  
✅ **多记忆组合**：结合多种策略，构建复杂系统

这些技术是构建自然对话体验的基础，让 LLM 应用具备真正的"记忆"能力。

---

## 扩展阅读

- [Memory](https://python.langchain.com/docs/modules/memory/)
- [Chat Message History](https://python.langchain.com/docs/modules/memory/chat_messages/)
- [Memory Types](https://python.langchain.com/docs/modules/memory/types/)
- [Custom Memory](https://python.langchain.com/docs/modules/memory/custom_memory)

---
title: "第7章：记忆系统 — Agent 的记忆与学习"
description: "全面掌握 Agent 记忆架构：认知科学基础、短期记忆、工作记忆、长期记忆、情景记忆与语义记忆的分离与协同，LangChain Memory 模块，Mem0 框架。"
date: "2026-06-11"
---

# 第7章：记忆系统 — Agent 的记忆与学习

---

## 7.1 Agent 记忆的认知科学基础

### 7.1.1 人类记忆模型与 Agent 类比

| 人类记忆类型 | 持续时间 | Agent 对应 | 实现方式 |
|:---|:---|:---|:---|
| **感觉记忆** | <1秒 | API 原始返回 | 内存缓冲区 |
| **短期记忆** | ~30秒 | 对话上下文窗口 | 消息列表 |
| **工作记忆** | 中等 | Agent Scratchpad | 状态变量 |
| **情景记忆** | 长久 | 历史对话存储 | 向量数据库 |
| **语义记忆** | 长久 | 知识库/文档 | RAG 系统 |
| **程序性记忆** | 长久 | 工具使用经验 | Few-shot 示例 |

```
┌─────────────────────────────────────────────────────────────┐
│                 Agent 记忆架构                                │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              工作记忆 (Working Memory)                  │  │
│  │  • 当前任务状态    • 中间推理结果    • 执行计划          │  │
│  │  • 工具调用历史    • 错误记录        • 用户偏好         │  │
│  └───────────────────────┬───────────────────────────────┘  │
│                          │                                  │
│  ┌──────────┐    ┌───────┴───────┐    ┌──────────────┐    │
│  │ 短期记忆  │    │  检索增强     │    │   长期记忆    │    │
│  │          │    │              │    │              │    │
│  │ 对话窗口  │◄──►│  向量检索    │◄──►│  情景记忆    │    │
│  │ 最近 N 轮│    │  相似度匹配  │    │  语义记忆    │    │
│  └──────────┘    └─────────────┘    └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 7.2 短期记忆：对话上下文

### 7.2.1 对话缓冲记忆

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
```

### 7.2.2 窗口记忆

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=5, return_messages=True)
```

### 7.2.3 Token 限制记忆

```python
from langchain.memory import ConversationTokenBufferMemory

memory = ConversationTokenBufferMemory(
    llm=ChatOpenAI(model="gpt-4o"),
    max_token_limit=4000,
    return_messages=True,
)
```

---

## 7.3 长期记忆：摘要与向量存储

### 7.3.1 摘要记忆

```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    return_messages=True,
)
```

### 7.3.2 向量记忆

```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_texts(["初始化占位"], embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key="relevant_history")
```

---

## 7.4 记忆衰减与遗忘机制

```python
import math
from datetime import datetime

class MemoryWithDecay:
    def __init__(self, content: str, importance: float = 1.0):
        self.content = content
        self.importance = importance
        self.created_at = datetime.now()
        self.access_count = 0

    def get_strength(self) -> float:
        hours = (datetime.now() - self.created_at).total_seconds() / 3600
        decay = math.exp(-0.1 * hours)
        access_boost = 1 + 0.2 * self.access_count
        return self.importance * decay * access_boost

    def access(self):
        self.access_count += 1
```

---

## 7.5 LangChain Memory 模块实战

```python
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

llm = ChatOpenAI(model="gpt-4o", temperature=0)
memory = ConversationBufferWindowMemory(k=10, return_messages=True)

conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
```

---

## 7.6 Mem0：生产级 Agent 记忆框架

```python
from mem0 import Memory

m = Memory()
m.add("我喜欢用 Python 开发", user_id="user_123")
results = m.search("我用什么编程语言？", user_id="user_123")
```

---

## 7.7 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| 记忆类型 | 短期、工作、长期（情景/语义） |
| Buffer Memory | 保存完整对话 |
| Window Memory | 保留最近 N 轮 |
| Summary Memory | LLM 摘要压缩 |
| Vector Memory | 语义检索 |
| 记忆衰减 | 模拟遗忘曲线 |
| Mem0 | 生产级记忆框架 |

> **下一章预告**
>
> 在第 8 章中，我们将深入记忆系统的工程实现。

---
title: "第8章：记忆工程实现 — 从理论到生产"
description: "深入记忆系统的工程实现：统一记忆接口、混合记忆管理器、对话历史管理、摘要压缩、向量检索记忆、跨会话持久化与记忆一致性保证。"
date: "2026-06-11"
---

# 第8章：记忆工程实现 — 从理论到生产

---

## 8.1 生产级记忆管理器

### 8.1.1 统一记忆接口

```python
from abc import ABC, abstractmethod

class BaseMemoryManager(ABC):
    @abstractmethod
    def add_messages(self, messages: list) -> None: pass

    @abstractmethod
    def get_context(self, query: str, max_tokens: int = 4000) -> list: pass

    @abstractmethod
    def save_checkpoint(self) -> str: pass

    @abstractmethod
    def clear(self) -> None: pass
```

### 8.1.2 混合记忆管理器

```python
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import tiktoken

class HybridMemoryManager(BaseMemoryManager):
    def __init__(self, session_id, llm=None, buffer_size=10):
        self.session_id = session_id
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.buffer_size = buffer_size
        self.buffer = []
        self.summary = ""
        self.vectorstore = FAISS.from_texts(["初始化"], embedding=OpenAIEmbeddings())
        self.entities = {}
        self.encoder = tiktoken.encoding_for_model("gpt-4o")

    def add_messages(self, messages):
        for msg in messages:
            self.buffer.append(msg)
            self._extract_entities(msg)
            if msg.content:
                self.vectorstore.add_texts([msg.content])
        if len(self.buffer) > self.buffer_size * 2:
            self._compress_buffer()

    def get_context(self, query, max_tokens=4000):
        messages = [SystemMessage(content=self._build_system_context(query))]
        relevant = self.vectorstore.similarity_search(query, k=3)
        if relevant:
            messages.append(SystemMessage(content=f"[相关历史]\n{chr(10).join(d.page_content for d in relevant)}"))
        messages.extend(self.buffer[-(self.buffer_size * 2):])
        total = sum(len(self.encoder.encode(m.content or "")) for m in messages)
        if total > max_tokens:
            messages = self._trim_messages(messages, max_tokens)
        return messages

    def _compress_buffer(self):
        old = self.buffer[:-self.buffer_size]
        self.buffer = self.buffer[-self.buffer_size:]
        old_text = "\n".join(f"{m.type}: {(m.content or '')[:200]}" for m in old)
        response = self.llm.invoke([HumanMessage(content=f"总结以下对话：\n{old_text}")])
        self.summary = f"{self.summary}\n{response.content}" if self.summary else response.content

    def _extract_entities(self, message):
        if not message.content: return
        for kw in ["名字", "叫", "使用", "项目", "喜欢"]:
            if kw in message.content:
                self.entities[f"msg_{id(message)}"] = message.content[:100]
                break

    def _build_system_context(self, query):
        parts = ["你是 AI Agent 助手。"]
        if self.summary: parts.append(f"\n[对话摘要]\n{self.summary}")
        return "\n".join(parts)

    def _trim_messages(self, messages, max_tokens):
        result, total = [], 0
        for msg in reversed(messages):
            tokens = len(self.encoder.encode(msg.content or ""))
            if total + tokens > max_tokens: break
            result.insert(0, msg)
            total += tokens
        return result

    def save_checkpoint(self): return f"ckpt_{self.session_id}"
    def clear(self): self.buffer, self.summary, self.entities = [], "", {}
```

---

## 8.2 LangGraph 中的记忆管理

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

def agent_node(state: MessagesState):
    llm = ChatOpenAI(model="gpt-4o")
    return {"messages": [llm.invoke(state["messages"])]}

graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

app = graph.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "user_123"}}
result = app.invoke({"messages": [("user", "你好，我叫张三")]}, config=config)
result = app.invoke({"messages": [("user", "你还记得我叫什么吗？")]}, config=config)
```

---

## 8.3 混合检索策略

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

class HybridMemoryRetriever:
    def __init__(self, texts):
        self.bm25 = BM25Retriever.from_texts(texts, k=3)
        self.vector = FAISS.from_texts(texts, OpenAIEmbeddings()).as_retriever(search_kwargs={"k": 3})
        self.ensemble = EnsembleRetriever(retrievers=[self.bm25, self.vector], weights=[0.3, 0.7])

    def retrieve(self, query):
        return [doc.page_content for doc in self.ensemble.invoke(query)]
```

---

## 8.4 跨会话记忆

```python
class UserMemoryStore:
    def __init__(self, user_id, storage):
        self.user_id = user_id
        self.storage = storage

    def save_fact(self, fact, session):
        self.storage.save(namespace=f"user:{self.user_id}:facts",
                         key=f"fact_{datetime.now().timestamp()}",
                         value=json.dumps({"fact": fact, "session": session}))

    def recall_facts(self, query, top_k=5):
        results = self.storage.search(namespace=f"user:{self.user_id}:facts", query=query, top_k=top_k)
        return [json.loads(r["value"])["fact"] for r in results]
```

---

## 8.5 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| 统一接口 | BaseMemoryManager 标准接口 |
| 混合记忆 | Buffer + Summary + Vector + Entity |
| 缓冲区压缩 | 摘要释放上下文空间 |
| Checkpointer | LangGraph 内置持久化 |
| 混合检索 | BM25 + 向量检索 |
| 跨会话记忆 | 用户级偏好和事实 |

> **下一章预告**
>
> 在第 9 章中，我们将进入 RAG 的世界。

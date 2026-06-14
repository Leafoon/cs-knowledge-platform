---
title: "第9章：RAG 基础 — 检索增强生成"
description: "全面掌握 RAG 系统的核心架构：文档加载、文本分块策略、Embedding 模型选型、向量数据库对比、相似度检索、RAG 链构建与生成质量评估。"
date: "2026-06-11"
---

# 第9章：RAG 基础 — 检索增强生成

---

## 9.1 RAG 的核心原理

$$
\text{RAG}(q) = \text{LLM}(q, \text{Retrieve}(q, \mathcal{D}))
$$

| 问题 | 无 RAG | 有 RAG |
|:---|:---|:---|
| **知识截止** | 模型只知训练数据前 | 可检索最新信息 |
| **幻觉** | 可能编造信息 | 基于检索事实回答 |
| **私有数据** | 无法访问 | 可检索私有知识库 |

---

## 9.2 离线索引阶段

### 9.2.1 文档加载

```python
from langchain_community.document_loaders import TextLoader, PDFLoader, CSVLoader, WebBaseLoader

loader = TextLoader("data.txt", encoding="utf-8")
docs = loader.load()
```

### 9.2.2 文本分块

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。", " "],
)
chunks = splitter.split_documents(docs)
```

| 场景 | chunk_size | chunk_overlap |
|:---|:---|:---|
| 精确问答 | 300-500 | 50 |
| 文档摘要 | 1500-2000 | 200 |
| 代码搜索 | 500-1000 | 100 |

### 9.2.3 Embedding 模型

| 模型 | 维度 | 中文 | 价格 |
|:---|:---|:---|:---|
| text-embedding-3-small | 1536 | ✅ | $0.02/M |
| text-embedding-3-large | 3072 | ✅ | $0.13/M |
| bge-small-zh | 512 | ✅ | 免费 |

---

## 9.3 向量数据库

| 数据库 | 特点 | 适用场景 |
|:---|:---|:---|
| FAISS | 内存级，速度快 | 开发/小规模 |
| Chroma | 嵌入式，简单 | 原型/小项目 |
| Qdrant | Rust，高性能 | 生产环境 |
| Milvus | 分布式 | 企业级 |

```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_texts(texts, embedding=OpenAIEmbeddings(), metadatas=metadatas)
results = vectorstore.similarity_search("什么是智能体？", k=2)
```

---

## 9.4 RAG 链构建

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
prompt = ChatPromptTemplate.from_template("基于以下上下文回答问题。\n上下文：{context}\n问题：{question}\n回答：")
llm = ChatOpenAI(model="gpt-4o", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
```

---

## 9.5 RAG 质量评估

| 指标 | 含义 |
|:---|:---|
| Context Precision | 检索结果中相关文档比例 |
| Context Recall | 相关文档被检索到的比例 |
| Faithfulness | 生成内容与检索内容一致性 |
| Answer Relevancy | 回答与问题相关程度 |

---

## 9.6 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| RAG 原理 | 先检索再生成 |
| 文本分块 | RecursiveCharacterTextSplitter |
| Embedding | text-embedding-3-small |
| 向量数据库 | FAISS 适合开发，Qdrant 适合生产 |
| 质量评估 | RAGAS 框架 |

> **下一章预告**
>
> 在第 10 章中，我们将深入高级 RAG 策略。

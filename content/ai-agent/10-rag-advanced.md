---
title: "第10章：高级 RAG 策略 — 从 Naive 到 Production"
description: "深入高级 RAG 技术：Multi-Query、HyDE、Parent-Child、RAPTOR、重排序、上下文压缩、Self-RAG、Corrective RAG、Graph RAG 与 Agentic RAG。"
date: "2026-06-11"
---

# 第10章：高级 RAG 策略 — 从 Naive 到 Production

---

## 10.1 Query 转换与扩展

### 10.1.1 Multi-Query RAG

```python
multi_query_prompt = """给定用户问题，生成 3 个不同角度的搜索查询。

用户问题：{question}
输出 3 个查询，每行一个："""

def multi_query_retrieve(question, retriever, llm):
    chain = multi_query_prompt | llm
    response = chain.invoke({"question": question})
    queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]

    all_docs, seen = [], set()
    for query in queries:
        for doc in retriever.invoke(query):
            h = hash(doc.page_content)
            if h not in seen:
                all_docs.append(doc)
                seen.add(h)
    return all_docs
```

### 10.1.2 HyDE

```python
hyde_prompt = """针对以下问题，写一段可能的回答（用于搜索）。

问题：{question}
假设性回答："""

def hyde_retrieve(question, retriever, llm):
    hypothetical = (hyde_prompt | llm).invoke({"question": question}).content
    return retriever.invoke(hypothetical)
```

---

## 10.2 重排序（Rerank）

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
reranker = CohereRerank(model="rerank-v3.5", top_n=5)
rerank_retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)
```

---

## 10.3 Parent-Child 检索

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

parent_retriever = ParentDocumentRetriever(
    vectorstore=child_vectorstore,
    docstore=InMemoryStore(),
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
```

---

## 10.4 Self-RAG

```python
SELF_RAG_PROMPT = """
1. 是否需要检索？[Retrieve] 或 [No Retrieve]
2. 检索相关性？[Is Relevant] 或 [Is Not Relevant]
3. 回答忠实度？[Supported] 或 [Not Supported]

用户问题：{question}
检索结果：{context}
"""
```

---

## 10.5 Corrective RAG (CRAG)

```python
class CorrectiveRAG:
    def run(self, question):
        docs = self.retriever.invoke(question)
        quality = self.evaluate_retrieval(question, docs)
        if quality == "correct": return self.generate(question, docs)
        elif quality == "ambiguous": return self.generate_with_correction(question, docs)
        else: return self.generate(question, self.web_search(question))
```

---

## 10.6 Graph RAG

```python
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain

graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="password")
chain = GraphCypherQAChain.from_llm(llm=ChatOpenAI(model="gpt-4o"), graph=graph)
```

---

## 10.7 Agentic RAG

```python
from langgraph.prebuilt import create_react_agent

rag_agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o"),
    tools=[vector_retriever_tool, web_search_tool, sql_query_tool],
)
```

---

## 10.8 本章小结

| 策略 | 核心思想 | 适用场景 |
|:---|:---|:---|
| Multi-Query | 多角度查询扩展 | 提升召回率 |
| HyDE | 假设性文档检索 | 查询与文档表述差异大 |
| Rerank | 精排检索结果 | 精度要求高 |
| Parent-Child | 小块检索大块返回 | 需要上下文完整性 |
| Self-RAG | 自我评估检索需求 | 动态决定是否检索 |
| CRAG | 检索后自我纠正 | 检索质量不稳定 |
| Graph RAG | 知识图谱增强 | 实体关系密集 |
| Agentic RAG | Agent 驱动检索 | 多源检索 |

> **下一章预告**
>
> 在第 11 章中，我们将进入 Agent 规划的世界。

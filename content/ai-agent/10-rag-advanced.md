---
title: "第10章：高级 RAG 策略"
description: "Multi-Query、HyDE、查询改写、重排序、Parent-Child检索、Self-RAG、Corrective RAG、Graph RAG、Agentic RAG"
chapter: 10
module: ai-agent
order: 10
tags: ["高级RAG", "查询优化", "检索策略", "Self-RAG", "Graph RAG"]
difficulty: advanced
---

 # 第10章：高级 RAG 策略

 > **高级 RAG 策略**是提升检索增强生成系统质量的关键技术。本章将深入探讨多种优化策略，从查询改写、检索优化到自适应生成，帮助构建更智能、更可靠的 RAG 系统。

 下面的交互式演示展示了查询改写的过程：

 <div data-component="QueryRewriteDemo"></div>

 ## 10.1 概述：为什么需要高级 RAG

### 10.1.1 基础 RAG 的局限性

基础 RAG 虽然有效，但存在以下问题：

| 问题 | 表现 | 影响 |
|------|------|------|
| 查询模糊 | 用户查询与文档表述不匹配 | 检索失败 |
| 单一视角 | 仅从一个角度检索信息 | 信息不全面 |
| 噪声干扰 | 检索结果包含大量无关内容 | 生成质量下降 |
| 缺乏验证 | 无法判断检索结果的可靠性 | 可能产生幻觉 |
| 静态流程 | 无法根据查询复杂度动态调整 | 复杂问题回答不好 |

### 10.1.2 高级 RAG 策略分类

```
高级 RAG 策略
├── 查询优化
│   ├── Multi-Query（多查询）
│   ├── HyDE（假设性文档嵌入）
 │   ├── 查询扩展
 │   └── 查询改写
 ├── 检索优化
 │   ├── 重排序（Re-ranking）
 │   ├── Parent-Child 检索
 │   ├── 混合检索
 │   └── 多路召回
 ├── 自适应生成
 │   ├── Self-RAG
 │   ├── Corrective RAG (CRAG)
 │   └── Adaptive RAG
 ├── 结构化检索
 │   ├── Graph RAG
 │   └── 知识图谱增强
 └── 智能检索
     └── Agentic RAG
 ```

 高级 RAG 策略涉及多种技术。下面的交互式可视化展示了各种策略的工作流程：

 <div data-component="AdvancedRAGStrategies"></div>

工具选择是 Agent 决策中的关键环节。下面的交互式演示展示了完整的工具选择决策过程：

<div data-component="ToolSelectionDemoV8"></div>

 ---

 ## 10.2 Multi-Query（多查询）

### 10.2.1 核心思想

Multi-Query 的核心思想是：**将一个查询扩展为多个不同角度的查询，分别检索后合并结果**。

> 单一查询可能无法覆盖所有相关信息，但多个查询可以从不同角度检索，提高召回率。

### 10.2.2 工作流程

```
原始查询
    │
    ▼
┌─────────────────────────┐
│     查询生成器（LLM）    │
│  生成多个不同角度的查询   │
└─────────────────────────┘
    │
    ▼
┌─────┬─────┬─────┐
│查询1│查询2│查询3│
└──┬──┴──┬──┴──┬──┘
   │     │     │
   ▼     ▼     ▼
┌─────┬─────┬─────┐
│检索1│检索2│检索3│
└──┬──┴──┬──┴──┬──┘
   │     │     │
   ▼     ▼     ▼
┌─────────────────────────┐
│    结果去重与合并        │
└─────────────────────────┘
    │
    ▼
   最终结果
```

### 10.2.3 实现代码

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List

class MultiQueryRAG:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def _generate_queries(self, question: str) -> List[str]:
        prompt = ChatPromptTemplate.from_template(
            "请生成3个不同角度的查询来检索以下问题的信息：{question}"
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"question": question})
        queries = [q.strip() for q in response.split("\n") if q.strip()]
        return queries[:3]
    
    def _deduplicate_and_merge(self, results):
        seen = set()
        merged = []
        for result_list in results:
            for doc in result_list:
                key = doc.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)
        return merged
    
    def query(self, question: str):
        queries = self._generate_queries(question)
        all_results = [self.retriever.invoke(q) for q in queries]
        return self._deduplicate_and_merge(all_results)
```

---

## 10.3 HyDE（Hypothetical Document Embeddings）

### 10.3.1 核心思想

HyDE 的核心思想是：**先让 LLM 生成一个假设性的文档答案，然后用这个假设文档的 embedding 进行检索**。

> 查询通常很短，而文档很长。用查询去匹配文档可能不准确，但如果先生成一个"假设答案"，这个假设答案会更像真实文档。

### 10.3.2 工作流程

```
原始查询 → LLM生成假设文档 → Embedding假设文档 → 用假设文档检索 → 用原始查询+真实文档生成答案
```

### 10.3.3 实现代码

```python
class HyDERAG:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    def _generate_hypothetical_doc(self, question: str) -> str:
        prompt = ChatPromptTemplate.from_template(
            "请写一段关于以下问题的详细解释文章：{question}"
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"question": question})
    
    def query(self, question: str):
        hypothetical_doc = self._generate_hypothetical_doc(question)
        docs = self.retriever.invoke(hypothetical_doc)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = ChatPromptTemplate.from_template(
            "基于以下上下文回答问题：{context}\n问题：{question}"
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": question})
```

---

## 10.4 查询改写（Query Rewriting）

### 10.4.1 核心思想

查询改写通过 LLM 优化用户的原始查询，使其更适合检索：

| 改写类型 | 目的 | 示例 |
|---------|------|------|
| 查询简化 | 去除冗余信息 | "请问你能告诉我..." → "机器学习定义" |
| 查询扩展 | 添加相关术语 | "AI" → "人工智能 机器学习 深度学习" |
| 查询专业化 | 使用专业术语 | "电脑病毒" → "恶意软件 malware" |
| 查询分解 | 拆分为子问题 | "比较A和B" → "A是什么？" + "B是什么？" |

### 10.4.2 实现代码

```python
class QueryRewriter:
    def __init__(self, llm):
        self.llm = llm
    
    def simplify_query(self, query: str) -> str:
        prompt = ChatPromptTemplate.from_template(
            "将以下查询简化为简洁的搜索查询：{query}"
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query}).strip()
    
    def expand_query(self, query: str) -> str:
        prompt = ChatPromptTemplate.from_template(
            "将以下查询扩展为包含更多相关关键词的查询：{query}"
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query}).strip()
    
    def decompose_query(self, query: str) -> List[str]:
        prompt = ChatPromptTemplate.from_template(
            "将以下复杂问题分解为2-4个子问题：{query}"
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"query": query})
        return [q.strip() for q in response.split("\n") if q.strip()]
    
    def professionalize_query(self, query: str, domain: str) -> str:
        prompt = ChatPromptTemplate.from_template(
            "将以下查询转换为{domain}领域的专业查询：{query}"
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query, "domain": domain}).strip()
```

---

## 10.5 重排序（Re-ranking）

### 10.5.1 核心思想

重排序在初步检索后，使用更精确的模型对结果进行重新排序。

> 初步检索速度快但精度有限，重排序模型精度高但速度慢。两者结合可以兼顾效率和质量。

### 10.5.2 重排序流程

```
初步检索（Top-K） → 重排序模型 → 计算查询-文档相关性 → 按相关性重新排序 → Top-N 结果
```

### 10.5.3 实现代码

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

class Reranker:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    def rerank_with_cross_encoder(self, query: str, top_k: int = 5):
        cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        compressor = CrossEncoderReranker(model=cross_encoder, top_n=top_k)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=self.retriever
        )
        return compression_retriever.invoke(query)
    
    def reciprocal_rank_fusion(self, query: str, k: int = 60):
        results = self.retriever.invoke(query)
        doc_scores = {}
        for rank, doc in enumerate(results):
            key = doc.page_content[:100]
            if key not in doc_scores:
                doc_scores[key] = {"doc": doc, "score": 0}
            doc_scores[key]["score"] += 1 / (k + rank + 1)
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs]
```

---

## 10.6 Parent-Child 检索

### 10.6.1 核心思想

Parent-Child 检索的核心思想是：**用小分块（Child）进行精确检索，但返回其父文档（Parent）作为上下文**。

> 小分块提高检索精度，大分块提供完整上下文。

### 10.6.2 工作流程

```
文档分割为Parent Chunk(2000字符)和Child Chunk(500字符)
  │
  ├── 用Child Chunks进行向量检索
  ├── 找到最相关的Child Chunks
  └── 返回对应的Parent Chunks
```

### 10.6.3 实现代码

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import uuid

class ParentChildRetriever:
    def __init__(self, embeddings, parent_size=2000, child_size=500):
        self.embeddings = embeddings
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_size, chunk_overlap=200)
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_size, chunk_overlap=50)
        self.parent_chunks = {}
        self.child_to_parent = {}
    
    def build_index(self, documents):
        parent_chunks = []
        child_chunks = []
        for doc in documents:
            parents = self.parent_splitter.split_documents([doc])
            for parent in parents:
                parent_id = str(uuid.uuid4())
                parent.metadata["parent_id"] = parent_id
                parent_chunks.append(parent)
                children = self.child_splitter.split_documents([parent])
                for child in children:
                    child_id = str(uuid.uuid4())
                    child.metadata["parent_id"] = parent_id
                    child_chunks.append(child)
                    self.child_to_parent[child_id] = parent_id
        self.child_vectorstore = Chroma.from_documents(child_chunks, self.embeddings)
        self.parent_chunks = {doc.metadata["parent_id"]: doc for doc in parent_chunks}
    
    def retrieve(self, query: str, top_k: int = 3):
        child_results = self.child_vectorstore.similarity_search(query, k=top_k * 2)
        seen_parents = set()
        parent_results = []
        for child in child_results:
            parent_id = child.metadata.get("parent_id")
            if parent_id and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                if parent_id in self.parent_chunks:
                    parent_results.append(self.parent_chunks[parent_id])
        return parent_results[:top_k]
```

---

## 10.7 Self-RAG

### 10.7.1 核心思想

Self-RAG 是一种自适应的 RAG 方法，模型会**自主判断**是否需要检索、检索结果是否有用、以及生成的回答是否忠实。

### 10.7.2 反思标记（Reflection Tokens）

| 标记类型 | 含义 | 可选值 |
|---------|------|--------|
| [Retrieve] | 是否需要检索 | Yes / No |
| [IsRel] | 检索结果是否相关 | Relevant / Irrelevant |
| [IsSup] | 回答是否被检索结果支持 | Fully Supported / Partially Supported / No Support |
| [IsUse] | 回答是否有用 | Useful:5 / Useful:4 / ... / Useful:1 |

### 10.7.3 实现代码

```python
class SelfRAG:
    def __init__(self, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    def _should_retrieve(self, query: str) -> bool:
        prompt = ChatPromptTemplate.from_template(
            "判断以下问题是否需要检索外部知识（是/否）：{query}"
        )
        chain = prompt | self.llm | StrOutputParser()
        return "是" in chain.invoke({"query": query})
    
    def _evaluate_relevance(self, query: str, document: str) -> float:
        prompt = ChatPromptTemplate.from_template(
            "评估以下文档与查询的相关性（0-1）：查询：{query} 文档：{document}"
        )
        chain = prompt | self.llm | StrOutputParser()
        try:
            return float(chain.invoke({"query": query, "document": document}).strip())
        except:
            return 0.5
    
    def query(self, question: str):
        if not self._should_retrieve(question):
            prompt = ChatPromptTemplate.from_template("回答以下问题：{question}")
            chain = prompt | self.llm | StrOutputParser()
            return {"answer": chain.invoke({"question": question}), "retrieved": False}
        
        docs = self.retriever.invoke(question)
        relevant_docs = [doc for doc in docs if self._evaluate_relevance(question, doc.page_content) > 0.5]
        
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        prompt = ChatPromptTemplate.from_template(
            "基于以下上下文回答问题：{context}\n问题：{question}"
        )
        chain = prompt | self.llm | StrOutputParser()
        return {"answer": chain.invoke({"context": context, "question": question}), "retrieved": True}
```

---

## 10.8 Corrective RAG (CRAG)

### 10.8.1 核心思想

Corrective RAG 在 Self-RAG 的基础上增加了**纠正机制**：当检索结果质量不佳时，会自动进行额外的检索或网络搜索。

### 10.8.2 工作流程

```
原始查询 → 检索文档 → 评估检索质量
    ├── 正确 → 直接使用检索结果
    └── 不正确 → 进一步评估
        ├── 模糊 → 精炼检索结果
        └── 错误 → 触发网络搜索
```

### 10.8.3 实现代码

```python
class CorrectiveRAG:
    def __init__(self, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    def _evaluate_quality(self, query: str, documents):
        docs_text = "\n\n".join([f"[文档{i+1}]\n{doc}" for i, doc in enumerate(documents)])
        prompt = ChatPromptTemplate.from_template(
            "评估检索质量（Correct/Ambiguous/Incorrect）：查询：{query}\n文档：{docs}"
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query, "docs": docs_text}).strip()
    
    def _web_search(self, query: str) -> str:
        return f"网络搜索结果：关于'{query}'的信息..."
    
    def query(self, question: str):
        docs = self.retriever.invoke(question)
        doc_contents = [doc.page_content for doc in docs]
        quality = self._evaluate_quality(question, doc_contents)
        
        if "Correct" in quality:
            context = "\n\n".join(doc_contents)
        elif "Ambiguous" in quality:
            context = "\n\n".join(doc_contents)
        else:
            context = self._web_search(question)
        
        prompt = ChatPromptTemplate.from_template(
            "基于以下上下文回答问题：{context}\n问题：{question}"
        )
        chain = prompt | self.llm | StrOutputParser()
        return {"answer": chain.invoke({"context": context, "question": question}), "quality": quality}
```

---

## 10.9 Graph RAG

### 10.9.1 核心思想

Graph RAG 结合了知识图谱和 RAG，通过实体关系图来增强检索和推理能力。

> 传统 RAG 基于文本相似度检索，而 Graph RAG 可以基于实体关系进行推理。

### 10.9.2 工作流程

```
文档 → 实体抽取 → 关系抽取 → 知识图谱构建
                                    │
用户查询 → 查询理解 → 图检索/文本检索 → 结果融合 → 生成回答
```

### 10.9.3 实现代码

```python
class GraphRAG:
    def __init__(self, llm, vectorstore=None):
        self.llm = llm
        self.vectorstore = vectorstore
        self.entities = {}
        self.relations = []
    
    def build_knowledge_graph(self, documents):
        prompt = ChatPromptTemplate.from_template(
            "从以下文本中提取实体和关系（JSON格式）：{text}"
        )
        chain = prompt | self.llm | StrOutputParser()
        for doc in documents:
            try:
                import json
                data = json.loads(chain.invoke({"text": doc[:2000]}))
                for entity in data.get("entities", []):
                    self.entities[entity["id"]] = entity
                for rel in data.get("relations", []):
                    self.relations.append(rel)
            except:
                continue
    
    def graph_retrieve(self, query: str) -> str:
        prompt = ChatPromptTemplate.from_template("从以下问题中提取主要实体：{query}")
        chain = prompt | self.llm | StrOutputParser()
        entities_str = chain.invoke({"query": query})
        entities = [e.strip() for e in entities_str.split("\n") if e.strip()]
        
        graph_context = []
        for entity in entities:
            for rel in self.relations:
                if rel.get("subject") == entity or rel.get("object") == entity:
                    graph_context.append(f"{rel.get('subject')} --[{rel.get('predicate')}]--> {rel.get('object')}")
        return "\n".join(graph_context) if graph_context else "未找到相关图谱信息"
    
    def query(self, question: str):
        graph_context = self.graph_retrieve(question)
        text_context = ""
        if self.vectorstore:
            docs = self.vectorstore.similarity_search(question, k=3)
            text_context = "\n\n".join([doc.page_content for doc in docs])
        
         combined_context = f"知识图谱信息：\n{graph_context}\n\n文本信息：\n{text_context}"
         prompt = ChatPromptTemplate.from_template(
             "基于以下上下文回答问题：{context}\n问题：{question}"
         )
         chain = prompt | self.llm | StrOutputParser()
         return {"answer": chain.invoke({"context": combined_context, "question": question})}
 ```

 知识图谱增强的 RAG 系统能够处理复杂的实体关系查询。下面的交互式工具展示了 Graph RAG 的工作原理：

 <div data-component="GraphRAGVisualization"></div>

 ---

 ## 10.10 Agentic RAG

### 10.10.1 核心思想

Agentic RAG 将 Agent 的决策能力与 RAG 结合，让系统能够自主决定检索策略、工具使用和回答生成。

### 10.10.2 工作流程

```
用户查询 → Agent规划器 → 工具选择 → 执行与观察 → 反思与迭代 → 生成最终回答
```

### 10.10.3 实现代码

```python
class AgenticRAG:
    def __init__(self, llm, vectorstore=None):
        self.llm = llm
        self.vectorstore = vectorstore
        self.memory = []
    
    def _vector_search(self, query: str) -> str:
        if not self.vectorstore:
            return "向量库不可用"
        docs = self.vectorstore.similarity_search(query, k=5)
        return "\n\n".join([f"[文档{i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)])
    
    def _web_search(self, query: str) -> str:
        return f"网络搜索结果：关于'{query}'的信息..."
    
    def _direct_answer(self, query: str) -> str:
        return f"基于通用知识：关于'{query}'的回答..."
    
    def _plan_action(self, query: str):
        prompt = ChatPromptTemplate.from_template(
            "分析查询并决定使用哪个工具（vector_search/web_search/direct_answer）：{query}"
        )
        chain = prompt | self.llm | StrOutputParser()
        tool = chain.invoke({"query": query}).strip()
        return {"tool": tool, "query": query}
    
    def _execute_tool(self, tool_name: str, query: str) -> str:
        tools = {"vector_search": self._vector_search, "web_search": self._web_search, "direct_answer": self._direct_answer}
        return tools.get(tool_name, self._direct_answer)(query)
    
    def _generate_final_answer(self, query: str, context: str) -> str:
        prompt = ChatPromptTemplate.from_template(
            "基于以下上下文回答问题：{context}\n问题：{query}"
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "query": query})
    
    def query(self, question: str, max_iterations: int = 3):
        from langchain_core.messages import HumanMessage, AIMessage
        self.memory.append(HumanMessage(content=question))
        
        all_results = []
        for iteration in range(max_iterations):
            decision = self._plan_action(question)
            tool_result = self._execute_tool(decision["tool"], decision["query"])
            all_results.append(tool_result)
        
        combined_context = "\n\n---\n\n".join(all_results)
        final_answer = self._generate_final_answer(question, combined_context)
        self.memory.append(AIMessage(content=final_answer))
        
 return {"question": question, "answer": final_answer, "iterations": max_iterations}
 ```

Agent 规划能力对于处理复杂任务至关重要。下面的交互式可视化展示了 Agent 如何构建规划树：

<div data-component="AgentPlanningDemoV9"></div>

 ---

 ## 10.11 策略对比表

| 策略 | 核心思想 | 优点 | 缺点 | 适用场景 |
|------|---------|------|------|---------|
| Multi-Query | 多角度查询 | 提高召回率 | 增加检索次数 | 信息需求复杂 |
| HyDE | 假设文档检索 | 提高匹配度 | 需要额外生成 | 查询与文档表述差异大 |
| 查询改写 | 优化查询 | 灵活可定制 | 需要LLM调用 | 查询质量差 |
| 重排序 | 后处理排序 | 提高精度 | 增加延迟 | 对精度要求高 |
| Parent-Child | 层级检索 | 精度+上下文 | 实现复杂 | 需要完整上下文 |
| Self-RAG | 自适应检索 | 减少不必要检索 | 模型依赖 | 资源受限场景 |
| CRAG | 纠正机制 | 更可靠 | 实现复杂 | 对准确性要求高 |
| Graph RAG | 结构化知识 | 支持复杂推理 | 构建成本高 | 关系密集型知识 |
| Agentic RAG | 智能决策 | 灵活自主 | 延迟高 | 复杂多步骤任务 |

---

## 10.12 本章小结

> **核心要点回顾**：
>
> 1. **Multi-Query**：通过多角度查询提高检索召回率
> 2. **HyDE**：用假设文档进行检索，提高匹配准确性
> 3. **查询改写**：优化用户查询，使其更适合检索
> 4. **重排序**：后处理提升检索结果质量
> 5. **Parent-Child**：兼顾检索精度和上下文完整性
> 6. **Self-RAG**：自适应决策，智能选择是否检索
> 7. **CRAG**：增加纠正机制，提高可靠性
> 8. **Graph RAG**：结合知识图谱，支持复杂推理
> 9. **Agentic RAG**：Agent + RAG，自主规划执行

**下一步**：学习第11章，了解 Agent 的规划基础和思维链技术。

---

## 10.13 高级RAG评估框架

### 10.13.1 评估维度体系

高级RAG系统的评估需要从多个维度进行综合考量：

| 评估维度 | 评估指标 | 权重 | 评估方法 |
|---------|---------|------|---------|
| 检索质量 | Recall@K, Precision@K, MRR, NDCG | 0.3 | 自动化计算 |
| 生成质量 | Faithfulness, Relevance, Completeness | 0.3 | LLM评估 |
| 效率性能 | 延迟, 吞吐量, 资源消耗 | 0.2 | 基准测试 |
| 用户体验 | 满意度, 可用性, 可解释性 | 0.2 | 用户调研 |

### 10.13.2 RAG评估指标详解

#### 检索质量指标

$$\text{Recall@K} = \frac{|\{d \in \text{Retrieved}_K\} \cap \{d \in \text{Relevant}\}|}{|\{d \in \text{Relevant}\}|}$$

$$\text{Precision@K} = \frac{|\{d \in \text{Retrieved}_K\} \cap \{d \in \text{Relevant}\}|}{K}$$

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

其中：

$$\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$$

#### 生成质量指标

| 指标名称 | 计算方法 | 取值范围 | 说明 |
|---------|---------|---------|------|
| Faithfulness | LLM判断生成内容是否忠实于上下文 | 0-1 | 越高越好 |
| Answer Relevance | 生成内容与问题的相关性 | 0-1 | 越高越好 |
| Context Precision | 检索上下文的精确度 | 0-1 | 越高越好 |
| Context Recall | 检索上下文的召回率 | 0-1 | 越高越好 |
| Answer Correctness | 生成答案的正确性 | 0-1 | 越高越好 |
| Hallucination Rate | 幻觉比例 | 0-1 | 越低越好 |

### 10.13.3 完整评估框架实现

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from enum import Enum

class MetricType(Enum):
    """评估指标类型"""
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    EFFICIENCY = "efficiency"

@dataclass
class EvaluationResult:
    """评估结果"""
    metric_name: str
    metric_type: MetricType
    value: float
    details: Dict[str, Any] = field(default_factory=dict)

class AdvancedRAGEvaluator:
    """高级RAG评估框架"""
    
    def __init__(self, llm=None):
        self.llm = llm
        self.results: List[EvaluationResult] = []
    
    # ========== 检索质量评估 ==========
    
    def recall_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """计算Recall@K"""
        retrieved_set = set(retrieved[:k])
        relevant_set = set(relevant)
        return len(retrieved_set & relevant_set) / len(relevant_set) if relevant_set else 0
    
    def precision_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """计算Precision@K"""
        retrieved_set = set(retrieved[:k])
        relevant_set = set(relevant)
        return len(retrieved_set & relevant_set) / k if k > 0 else 0
    
    def mrr(self, retrieved: List[str], relevant: List[str]) -> float:
        """计算Mean Reciprocal Rank"""
        relevant_set = set(relevant)
        for i, doc in enumerate(retrieved):
            if doc in relevant_set:
                return 1 / (i + 1)
        return 0
    
    def ndcg_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """计算NDCG@K"""
        relevant_set = set(relevant)
        
        # DCG
        dcg = sum([
            1 / np.log2(i + 2) if doc in relevant_set else 0
            for i, doc in enumerate(retrieved[:k])
        ])
        
        # IDCG
        ideal_relevant = min(len(relevant_set), k)
        idcg = sum([1 / np.log2(i + 2) for i in range(ideal_relevant)])
        
        return dcg / idcg if idcg > 0 else 0
    
    def evaluate_retrieval(self, retrieved: List[str], relevant: List[str], 
                          k: int = 5) -> Dict[str, float]:
        """综合检索质量评估"""
        results = {
            f"recall@{k}": self.recall_at_k(retrieved, relevant, k),
            f"precision@{k}": self.precision_at_k(retrieved, relevant, k),
            "mrr": self.mrr(retrieved, relevant),
            f"ndcg@{k}": self.ndcg_at_k(retrieved, relevant, k)
        }
        
        for name, value in results.items():
            self.results.append(EvaluationResult(
                metric_name=name,
                metric_type=MetricType.RETRIEVAL,
                value=value
            ))
        
        return results
    
    # ========== 生成质量评估 ==========
    
    def faithfulness_score(self, answer: str, context: str) -> float:
        """评估忠实度（使用LLM）"""
        if not self.llm:
            return 0.5
        
        prompt = f"""请评估以下回答是否忠实于给定的上下文。

上下文：{context}
回答：{answer}

评分标准：
- 1.0: 完全基于上下文，无任何添加
- 0.75: 基本忠实，有少量推断
- 0.5: 部分忠实，有一些超出上下文的内容
- 0.25: 大部分不忠实
- 0.0: 完全不忠实

请只返回一个数字（0-1）："""
        
        try:
            response = self.llm.invoke(prompt).content
            return float(response.strip())
        except:
            return 0.5
    
    def answer_relevance_score(self, question: str, answer: str) -> float:
        """评估回答相关性（使用LLM）"""
        if not self.llm:
            return 0.5
        
        prompt = f"""请评估以下回答与问题的相关性。

问题：{question}
回答：{answer}

评分标准：
- 1.0: 完全回答了问题，非常相关
- 0.75: 基本回答了问题
- 0.5: 部分相关
- 0.25: 相关性较低
- 0.0: 完全不相关

请只返回一个数字（0-1）："""
        
        try:
            response = self.llm.invoke(prompt).content
            return float(response.strip())
        except:
            return 0.5
    
    def completeness_score(self, question: str, answer: str) -> float:
        """评估完整性"""
        if not self.llm:
            return 0.5
        
        prompt = f"""请评估以下回答的完整性。

问题：{question}
回答：{answer}

评分标准：
- 1.0: 完整覆盖了问题的所有方面
- 0.75: 覆盖了大部分方面
- 0.5: 覆盖了部分方面
- 0.25: 覆盖较少
- 0.0: 几乎没有覆盖

请只返回一个数字（0-1）："""
        
        try:
            response = self.llm.invoke(prompt).content
            return float(response.strip())
        except:
            return 0.5
    
    def evaluate_generation(self, question: str, answer: str, 
                           context: str) -> Dict[str, float]:
        """综合生成质量评估"""
        results = {
            "faithfulness": self.faithfulness_score(answer, context),
            "answer_relevance": self.answer_relevance_score(question, answer),
            "completeness": self.completeness_score(question, answer)
        }
        
        for name, value in results.items():
            self.results.append(EvaluationResult(
                metric_name=name,
                metric_type=MetricType.GENERATION,
                value=value
            ))
        
        return results
    
    # ========== 综合评估 ==========
    
    def comprehensive_evaluation(self, question: str, answer: str,
                                retrieved_docs: List[str], relevant_docs: List[str],
                                context: str, k: int = 5) -> Dict[str, Any]:
        """综合评估RAG系统"""
        retrieval_metrics = self.evaluate_retrieval(retrieved_docs, relevant_docs, k)
        generation_metrics = self.evaluate_generation(question, answer, context)
        
        # 计算综合分数
        weights = {
            "retrieval": 0.4,
            "generation": 0.6
        }
        
        retrieval_score = np.mean(list(retrieval_metrics.values()))
        generation_score = np.mean(list(generation_metrics.values()))
        
        overall_score = (weights["retrieval"] * retrieval_score + 
                        weights["generation"] * generation_score)
        
        return {
            "question": question,
            "retrieval_metrics": retrieval_metrics,
            "generation_metrics": generation_metrics,
            "retrieval_score": retrieval_score,
            "generation_score": generation_score,
            "overall_score": overall_score
        }
    
    def generate_report(self) -> str:
        """生成评估报告"""
        report = "=== RAG评估报告 ===\n\n"
        
        retrieval_results = [r for r in self.results if r.metric_type == MetricType.RETRIEVAL]
        generation_results = [r for r in self.results if r.metric_type == MetricType.GENERATION]
        
        report += "【检索质量指标】\n"
        for r in retrieval_results:
            report += f"  - {r.metric_name}: {r.value:.4f}\n"
        
        report += "\n【生成质量指标】\n"
        for r in generation_results:
            report += f"  - {r.metric_name}: {r.value:.4f}\n"
        
        if retrieval_results and generation_results:
            report += f"\n【综合评分】\n"
            report += f"  - 检索平均分: {np.mean([r.value for r in retrieval_results]):.4f}\n"
            report += f"  - 生成平均分: {np.mean([r.value for r in generation_results]):.4f}\n"
        
        return report

# 使用示例
evaluator = AdvancedRAGEvaluator(llm=ChatOpenAI(model="gpt-4o", temperature=0))

# 评估示例
result = evaluator.comprehensive_evaluation(
    question="什么是机器学习？",
    answer="机器学习是人工智能的一个分支，专注于让计算机从数据中学习。",
    retrieved_docs=["ML是AI的分支", "深度学习是ML的子集"],
    relevant_docs=["ML是AI的分支"],
    context="机器学习是人工智能的一个重要分支"
)

 print(evaluator.generate_report())
 ```

 RAG 系统的调优需要系统化的方法。下面的交互式指南展示了各种调优策略的效果：

 <div data-component="RAGTuningGuide"></div>

 ### 10.13.4 自动化评估管道

```python
from typing import List, Dict
import json
from datetime import datetime

class RAGEvaluationPipeline:
    """RAG自动化评估管道"""
    
    def __init__(self, evaluator: AdvancedRAGEvaluator):
        self.evaluator = evaluator
        self.evaluation_history = []
    
    def run_evaluation(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """运行完整评估"""
        results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"评估测试用例 {i+1}/{len(test_cases)}...")
            
            result = self.evaluator.comprehensive_evaluation(
                question=test_case["question"],
                answer=test_case["answer"],
                retrieved_docs=test_case["retrieved_docs"],
                relevant_docs=test_case["relevant_docs"],
                context=test_case["context"]
            )
            results.append(result)
        
        # 汇总结果
        summary = self._aggregate_results(results)
        
        # 记录历史
        evaluation_record = {
            "timestamp": datetime.now().isoformat(),
            "num_cases": len(test_cases),
            "summary": summary,
            "details": results
        }
        self.evaluation_history.append(evaluation_record)
        
        return summary
    
    def _aggregate_results(self, results: List[Dict]) -> Dict[str, float]:
        """汇总评估结果"""
        if not results:
            return {}
        
        # 收集所有指标
        all_metrics = {}
        for result in results:
            for metric_name, value in result.get("retrieval_metrics", {}).items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
            
            for metric_name, value in result.get("generation_metrics", {}).items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # 计算平均值
        summary = {name: np.mean(values) for name, values in all_metrics.items()}
        summary["overall_score"] = np.mean([r["overall_score"] for r in results])
        
        return summary
    
    def compare_systems(self, system_a_results: List[Dict], 
                       system_b_results: List[Dict]) -> Dict[str, Any]:
        """比较两个RAG系统"""
        summary_a = self._aggregate_results(system_a_results)
        summary_b = self._aggregate_results(system_b_results)
        
        comparison = {}
        all_metrics = set(summary_a.keys()) | set(summary_b.keys())
        
        for metric in all_metrics:
            value_a = summary_a.get(metric, 0)
            value_b = summary_b.get(metric, 0)
            comparison[metric] = {
                "system_a": value_a,
                "system_b": value_b,
                "difference": value_b - value_a,
                "winner": "B" if value_b > value_a else "A" if value_a > value_b else "Tie"
            }
        
        return comparison
    
    def export_results(self, filepath: str):
        """导出评估结果"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_history, f, ensure_ascii=False, indent=2)

# 使用示例
pipeline = RAGEvaluationPipeline(evaluator)

test_cases = [
    {
        "question": "什么是深度学习？",
        "answer": "深度学习是机器学习的一个子集，使用多层神经网络。",
        "retrieved_docs": ["深度学习是ML子集", "神经网络基础"],
        "relevant_docs": ["深度学习是ML子集"],
        "context": "深度学习是机器学习的重要分支"
    },
    {
        "question": "NLP的应用有哪些？",
        "answer": "NLP应用包括机器翻译、情感分析、文本分类等。",
        "retrieved_docs": ["NLP应用", "机器翻译", "情感分析"],
        "relevant_docs": ["NLP应用", "机器翻译"],
        "context": "自然语言处理有多种应用"
    }
]

summary = pipeline.run_evaluation(test_cases)
print("评估摘要:", summary)
```

---

## 10.14 RAG调优策略

### 10.14.1 分块参数调优

分块参数对检索质量有显著影响。以下是调优策略：

| 参数 | 默认值 | 调优范围 | 影响 |
|------|-------|---------|------|
| chunk_size | 1000 | 200-4000 | 小分块精确，大分块完整 |
| chunk_overlap | 200 | 50-500 | 重叠越大上下文越连续 |
| separators | ["\n\n", "\n"] | 多种组合 | 影响分割点选择 |

#### 自动化分块调优

```python
from typing import List, Dict, Tuple
import itertools

class ChunkingOptimizer:
    """分块参数优化器"""
    
    def __init__(self, documents, embeddings, evaluator):
        self.documents = documents
        self.embeddings = embeddings
        self.evaluator = evaluator
    
    def grid_search(self, test_queries: List[Dict], 
                   chunk_sizes: List[int] = [500, 1000, 1500, 2000],
                   overlap_ratios: List[float] = [0.1, 0.2, 0.3]) -> List[Dict]:
        """网格搜索最佳分块参数"""
        results = []
        
        for chunk_size in chunk_sizes:
            for overlap_ratio in overlap_ratios:
                overlap = int(chunk_size * overlap_ratio)
                
                # 创建分块器
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=overlap
                )
                
                # 分块文档
                chunks = splitter.split_documents(self.documents)
                
                # 创建向量数据库
                from langchain_community.vectorstores import Chroma
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=self.embeddings
                )
                
                # 评估检索质量
                total_score = 0
                for test_case in test_queries:
                    retrieved = vectorstore.similarity_search(
                        test_case["query"], k=5
                    )
                    retrieved_ids = [doc.page_content[:50] for doc in retrieved]
                    
                    # 计算recall
                    relevant_set = set(test_case["relevant_ids"])
                    retrieved_set = set(retrieved_ids)
                    recall = len(retrieved_set & relevant_set) / len(relevant_set) if relevant_set else 0
                    total_score += recall
                
                avg_score = total_score / len(test_queries) if test_queries else 0
                
                results.append({
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "overlap_ratio": overlap_ratio,
                    "num_chunks": len(chunks),
                    "avg_recall": avg_score
                })
        
        # 按分数排序
        results.sort(key=lambda x: x["avg_recall"], reverse=True)
        return results
    
    def find_optimal_params(self, test_queries: List[Dict]) -> Dict:
        """找到最佳参数"""
        results = self.grid_search(test_queries)
        return results[0] if results else None

# 使用示例
optimizer = ChunkingOptimizer(documents, embeddings, evaluator)

test_queries = [
    {"query": "什么是机器学习？", "relevant_ids": ["ML是AI分支"]},
    {"query": "深度学习的原理", "relevant_ids": ["深度学习原理"]}
]

optimal = optimizer.find_optimal_params(test_queries)
print(f"最佳参数: chunk_size={optimal['chunk_size']}, overlap={optimal['overlap']}")
print(f"平均召回率: {optimal['avg_recall']:.4f}")
```

### 10.14.2 Embedding模型调优

```python
from typing import List, Dict
import numpy as np

class EmbeddingOptimizer:
    """Embedding模型优化器"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def add_model(self, name: str, embeddings):
        """添加评估的Embedding模型"""
        self.models[name] = embeddings
    
    def evaluate_models(self, test_queries: List[Dict], 
                       documents: List[str]) -> Dict[str, float]:
        """评估不同Embedding模型的效果"""
        results = {}
        
        for model_name, embeddings in self.models.items():
            # 生成document embeddings
            doc_embeddings = embeddings.embed_documents(documents)
            
            scores = []
            for test_case in test_queries:
                query_embedding = embeddings.embed_query(test_case["query"])
                
                # 计算余弦相似度
                similarities = []
                for doc_emb in doc_embeddings:
                    sim = np.dot(query_embedding, doc_emb) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
                    )
                    similarities.append(sim)
                
                # 检查相关文档是否排在前面
                relevant_idx = test_case.get("relevant_idx", 0)
                if similarities:
                    rank = sorted(range(len(similarities)), 
                                key=lambda i: similarities[i], reverse=True)
                    mrr = 1 / (rank.index(relevant_idx) + 1) if relevant_idx in rank else 0
                    scores.append(mrr)
            
            results[model_name] = np.mean(scores) if scores else 0
        
        return results
    
    def compare_latency(self, test_texts: List[str]) -> Dict[str, Dict]:
        """比较不同模型的延迟"""
        import time
        
        results = {}
        for model_name, embeddings in self.models.items():
            latencies = []
            for text in test_texts:
                start = time.time()
                embeddings.embed_query(text)
                latencies.append(time.time() - start)
            
            results[model_name] = {
                "avg_latency_ms": np.mean(latencies) * 1000,
                "p95_latency_ms": np.percentile(latencies, 95) * 1000
            }
        
        return results

# 使用示例
optimizer = EmbeddingOptimizer()
optimizer.add_model("openai_small", OpenAIEmbeddings(model="text-embedding-3-small"))
optimizer.add_model("bge_large", HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-zh-v1.5"))

# 评估效果
effectiveness = optimizer.evaluate_models(test_queries, documents)
print("模型效果对比:", effectiveness)

# 比较延迟
latency = optimizer.compare_latency(["测试文本"] * 100)
print("延迟对比:", latency)
```

### 10.14.3 检索参数调优

```python
from typing import List, Dict, Any
import numpy as np

class RetrievalParameterTuner:
    """检索参数调优器"""
    
    def __init__(self, vectorstore, evaluator):
        self.vectorstore = vectorstore
        self.evaluator = evaluator
    
    def tune_k_value(self, test_queries: List[Dict], 
                    k_values: List[int] = [1, 3, 5, 10, 20]) -> List[Dict]:
        """调优top-k值"""
        results = []
        
        for k in k_values:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
            
            scores = []
            for test_case in test_queries:
                docs = retriever.invoke(test_case["query"])
                retrieved_ids = [doc.page_content[:50] for doc in docs]
                
                # 计算precision和recall
                relevant_set = set(test_case["relevant_ids"])
                retrieved_set = set(retrieved_ids)
                
                precision = len(retrieved_set & relevant_set) / k if k > 0 else 0
                recall = len(retrieved_set & relevant_set) / len(relevant_set) if relevant_set else 0
                
                # F1分数
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                scores.append(f1)
            
            results.append({
                "k": k,
                "avg_f1": np.mean(scores),
                "avg_precision": precision,
                "avg_recall": recall
            })
        
        return results
    
    def tune_search_type(self, test_queries: List[Dict]) -> Dict[str, float]:
        """比较不同检索类型的效果"""
        search_types = ["similarity", "mmr", "similarity_score_threshold"]
        results = {}
        
        for search_type in search_types:
            try:
                if search_type == "mmr":
                    retriever = self.vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.7}
                    )
                elif search_type == "similarity_score_threshold":
                    retriever = self.vectorstore.as_retriever(
                        search_type="similarity_score_threshold",
                        search_kwargs={"k": 5, "score_threshold": 0.5}
                    )
                else:
                    retriever = self.vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 5}
                    )
                
                scores = []
                for test_case in test_queries:
                    docs = retriever.invoke(test_case["query"])
                    # 简化的评分
                    scores.append(len(docs) / 5)
                
                results[search_type] = np.mean(scores)
            except Exception as e:
                results[search_type] = 0.0
        
        return results
    
    def tune_mmr_lambda(self, test_queries: List[Dict],
                       lambda_values: List[float] = [0.3, 0.5, 0.7, 0.9]) -> List[Dict]:
        """调优MMR的lambda参数"""
        results = []
        
        for lambda_mult in lambda_values:
            retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": lambda_mult}
            )
            
            scores = []
            for test_case in test_queries:
                docs = retriever.invoke(test_case["query"])
                # 评估多样性
                contents = [doc.page_content[:100] for doc in docs]
                unique_ratio = len(set(contents)) / len(contents) if contents else 0
                scores.append(unique_ratio)
            
            results.append({
                "lambda_mult": lambda_mult,
                "avg_diversity": np.mean(scores)
            })
        
        return results

# 使用示例
tuner = RetrievalParameterTuner(vectorstore, evaluator)

# 调优k值
k_results = tuner.tune_k_value(test_queries)
print("k值调优结果:", k_results)

# 调优检索类型
type_results = tuner.tune_search_type(test_queries)
print("检索类型对比:", type_results)
```

---

## 10.15 策略组合与混合RAG

### 10.15.1 策略组合原则

不同RAG策略可以组合使用以获得更好的效果。以下是常见的组合模式：

| 组合模式 | 组合策略 | 适用场景 | 预期效果 |
|---------|---------|---------|---------|
| 查询增强型 | Multi-Query + HyDE | 查询模糊 | 召回率+30% |
| 检索优化型 | 混合检索 + 重排序 | 精度要求高 | 精度+25% |
| 质量保证型 | Self-RAG + CRAG | 准确性要求高 | 幻觉率-40% |
| 全面型 | 所有策略组合 | 复杂场景 | 综合提升 |

### 10.15.2 混合检索实现

```python
from typing import List, Dict, Any
from langchain_core.documents import Document
import numpy as np

class HybridRetriever:
    """混合检索器：结合向量检索和关键词检索"""
    
    def __init__(self, vectorstore, bm25_index=None):
        self.vectorstore = vectorstore
        self.bm25_index = bm25_index
        self.vector_weight = 0.7
        self.bm25_weight = 0.3
    
    def vector_search(self, query: str, k: int = 10) -> List[Document]:
        """向量检索"""
        return self.vectorstore.similarity_search(query, k=k)
    
    def bm25_search(self, query: str, k: int = 10) -> List[Document]:
        """BM25关键词检索"""
        if not self.bm25_index:
            return []
        
        # BM25检索实现
        from rank_bm25 import BM25Okapi
        tokenized_query = query.split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # 获取top-k
        top_indices = np.argsort(scores)[::-1][:k]
        return [self.documents[i] for i in top_indices if i < len(self.documents)]
    
    def reciprocal_rank_fusion(self, 
                              vector_results: List[Document],
                              bm25_results: List[Document],
                              k: int = 60) -> List[Document]:
        """倒数排名融合（RRF）"""
        doc_scores = {}
        
        # 向量检索分数
        for rank, doc in enumerate(vector_results):
            doc_id = hash(doc.page_content[:100])
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "score": 0}
            doc_scores[doc_id]["score"] += self.vector_weight / (k + rank + 1)
        
        # BM25检索分数
        for rank, doc in enumerate(bm25_results):
            doc_id = hash(doc.page_content[:100])
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "score": 0}
            doc_scores[doc_id]["score"] += self.bm25_weight / (k + rank + 1)
        
        # 按分数排序
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs]
    
    def hybrid_search(self, query: str, k: int = 5) -> List[Document]:
        """混合检索"""
        vector_results = self.vector_search(query, k=k*2)
        bm25_results = self.bm25_search(query, k=k*2)
        
        return self.reciprocal_rank_fusion(vector_results, bm25_results, k=k)

# 使用示例
hybrid_retriever = HybridRetriever(vectorstore)
results = hybrid_retriever.hybrid_search("什么是机器学习？", k=5)
```

### 10.15.3 完整混合RAG系统

```python
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class AdvancedHybridRAG:
    """高级混合RAG系统"""
    
    def __init__(self, vectorstore, llm=None):
        self.vectorstore = vectorstore
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0)
        self.embeddings = OpenAIEmbeddings()
        
        # 初始化各种策略
        self.multi_query_enabled = True
        self.hyde_enabled = True
        self.reranking_enabled = True
    
    def _multi_query_generation(self, question: str, num_queries: int = 3) -> List[str]:
        """生成多个查询"""
        prompt = ChatPromptTemplate.from_template(
            f"请生成{num_queries}个不同角度的查询来检索以下问题的信息：{{question}}"
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"question": question})
        queries = [q.strip() for q in response.split("\n") if q.strip()]
        return queries[:num_queries]
    
    def _hyde_generation(self, question: str) -> str:
        """生成假设性文档"""
        prompt = ChatPromptTemplate.from_template(
            "请写一段关于以下问题的详细解释文章：{question}"
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"question": question})
    
    def _rerank_documents(self, query: str, documents: List[Document], 
                         top_k: int = 5) -> List[Document]:
        """重排序文档"""
        # 使用LLM进行重排序
        prompt = ChatPromptTemplate.from_template(
            """请根据以下查询对文档进行排序（最相关到最不相关）。

查询：{query}

文档列表：
{documents}

请返回排序后的文档编号（逗号分隔）："""
        )
        
        docs_text = "\n".join([f"[{i+1}] {doc.page_content[:200]}" 
                              for i, doc in enumerate(documents)])
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"query": query, "documents": docs_text})
        
        # 解析排序结果
        try:
            indices = [int(x.strip()) - 1 for x in response.split(",")]
            reranked = [documents[i] for i in indices if 0 <= i < len(documents)]
            return reranked[:top_k]
        except:
            return documents[:top_k]
    
    def advanced_retrieve(self, question: str, k: int = 5) -> List[Document]:
        """高级检索策略"""
        all_docs = []
        
        # 1. Multi-Query检索
        if self.multi_query_enabled:
            queries = self._multi_query_generation(question)
            for q in queries:
                docs = self.vectorstore.similarity_search(q, k=3)
                all_docs.extend(docs)
        
        # 2. HyDE检索
        if self.hyde_enabled:
            hypothetical_doc = self._hyde_generation(question)
            docs = self.vectorstore.similarity_search(hypothetical_doc, k=3)
            all_docs.extend(docs)
        
        # 3. 去重
        seen = set()
        unique_docs = []
        for doc in all_docs:
            doc_id = hash(doc.page_content[:100])
            if doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)
        
        # 4. 重排序
        if self.reranking_enabled and len(unique_docs) > k:
            unique_docs = self._rerank_documents(question, unique_docs, k)
        
        return unique_docs[:k]
    
    def query(self, question: str) -> Dict[str, Any]:
        """执行高级RAG查询"""
        # 检索
        docs = self.advanced_retrieve(question, k=5)
        
        # 构建上下文
        context = "\n\n".join([f"[文档{i+1}] {doc.page_content}" 
                              for i, doc in enumerate(docs)])
        
        # 生成
        prompt = ChatPromptTemplate.from_template(
            """基于以下上下文信息回答问题。

上下文：
{context}

问题：{question}

请用中文详细回答："""
        )
        
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        
        return {
            "question": question,
            "answer": answer,
            "sources": [{"content": doc.page_content[:200], 
                        "metadata": doc.metadata} for doc in docs],
            "num_sources": len(docs)
        }

# 使用示例
advanced_rag = AdvancedHybridRAG(vectorstore)
result = advanced_rag.query("什么是深度学习？")
print("回答:", result["answer"])
print("来源数量:", result["num_sources"])
```

---

## 10.16 生产环境RAG模式

### 10.16.1 分层缓存架构

```python
from typing import Any, Optional
from datetime import datetime, timedelta
import json
import hashlib

class TieredCache:
    """分层缓存架构"""
    
    def __init__(self):
        self.l1_cache = {}  # 内存缓存（热数据）
        self.l2_cache = {}  # Redis缓存（温数据）
        self.l1_ttl = timedelta(minutes=5)
        self.l2_ttl = timedelta(hours=1)
    
    def _generate_key(self, query: str, **kwargs) -> str:
        """生成缓存键"""
        cache_data = {"query": query, **kwargs}
        return hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()
    
    def get(self, query: str, **kwargs) -> Optional[Any]:
        """获取缓存"""
        key = self._generate_key(query, **kwargs)
        
        # L1缓存
        if key in self.l1_cache:
            entry = self.l1_cache[key]
            if datetime.now() - entry["timestamp"] < self.l1_ttl:
                return entry["value"]
            del self.l1_cache[key]
        
        # L2缓存
        if key in self.l2_cache:
            entry = self.l2_cache[key]
            if datetime.now() - entry["timestamp"] < self.l2_ttl:
                # 提升到L1
                self.l1_cache[key] = entry
                return entry["value"]
            del self.l2_cache[key]
        
        return None
    
    def set(self, query: str, value: Any, **kwargs):
        """设置缓存"""
        key = self._generate_key(query, **kwargs)
        entry = {"value": value, "timestamp": datetime.now()}
        
        # 写入L1
        self.l1_cache[key] = entry
        
        # 异步写入L2
        self.l2_cache[key] = entry
    
    def invalidate(self, query: str, **kwargs):
        """失效缓存"""
        key = self._generate_key(query, **kwargs)
        self.l1_cache.pop(key, None)
        self.l2_cache.pop(key, None)
    
    def clear(self):
        """清空缓存"""
        self.l1_cache.clear()
        self.l2_cache.clear()

# 使用示例
cache = TieredCache()

def cached_query(query: str, vectorstore):
    """带缓存的查询"""
    # 查缓存
    cached_result = cache.get(query)
    if cached_result:
        print("Cache hit!")
        return cached_result
    
    # 执行查询
    result = vectorstore.similarity_search(query, k=5)
    
    # 写缓存
    cache.set(query, result)
    return result
```

### 10.16.2 异步处理架构

```python
import asyncio
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

@dataclass
class RAGTask:
    """RAG任务"""
    task_id: str
    query: str
    status: str = "pending"
    result: Any = None
    created_at: float = None
    completed_at: float = None

class AsyncRAGProcessor:
    """异步RAG处理器"""
    
    def __init__(self, rag_system, max_workers: int = 4):
        self.rag_system = rag_system
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks: Dict[str, RAGTask] = {}
    
    async def process_query(self, query: str) -> str:
        """异步处理查询"""
        task_id = f"task_{int(time.time() * 1000)}"
        task = RAGTask(task_id=task_id, query=query, created_at=time.time())
        self.tasks[task_id] = task
        
        # 在线程池中执行
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor, 
            self.rag_system.query, 
            query
        )
        
        task.result = result
        task.status = "completed"
        task.completed_at = time.time()
        
        return result
    
    async def batch_process(self, queries: List[str]) -> List[Dict]:
        """批量处理查询"""
        tasks = [self.process_query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [
            {"query": q, "result": r if not isinstance(r, Exception) else str(r)}
            for q, r in zip(queries, results)
        ]
    
    def get_task_status(self, task_id: str) -> Dict:
        """获取任务状态"""
        task = self.tasks.get(task_id)
        if not task:
            return {"error": "Task not found"}
        
        return {
            "task_id": task.task_id,
            "query": task.query,
            "status": task.status,
            "created_at": task.created_at,
            "completed_at": task.completed_at,
            "duration": (task.completed_at - task.created_at) if task.completed_at else None
        }

# 使用示例
async def main():
    processor = AsyncRAGProcessor(rag_system)
    
    queries = [
        "什么是机器学习？",
        "深度学习的原理",
        "NLP的应用"
    ]
    
    results = await processor.batch_process(queries)
    for result in results:
        print(f"Query: {result['query']}")
        print(f"Result: {result['result']}\n")

# asyncio.run(main())
```

### 10.16.3 错误处理与降级策略

```python
from typing import Any, Optional, Callable
from enum import Enum
import traceback

class DegradationLevel(Enum):
    """降级级别"""
    NONE = "none"           # 正常
    LIGHT = "light"         # 轻度降级
    MODERATE = "moderate"   # 中度降级
    SEVERE = "severe"       # 重度降级

class RAGDegradationManager:
    """RAG降级管理器"""
    
    def __init__(self):
        self.degradation_level = DegradationLevel.NONE
        self.error_count = 0
        self.success_count = 0
        self.error_threshold = 5
        self.success_threshold = 10
    
    def record_success(self):
        """记录成功"""
        self.success_count += 1
        self.error_count = 0
        
        if self.degradation_level != DegradationLevel.NONE:
            # 尝试恢复
            if self.success_count >= self.success_threshold:
                self.degradation_level = DegradationLevel.NONE
                print("系统恢复正常")
    
    def record_error(self, error: Exception):
        """记录错误"""
        self.error_count += 1
        self.success_count = 0
        
        # 根据错误次数决定降级级别
        if self.error_count >= self.error_threshold * 3:
            self.degradation_level = DegradationLevel.SEVERE
        elif self.error_count >= self.error_threshold * 2:
            self.degradation_level = DegradationLevel.MODERATE
        elif self.error_count >= self.error_threshold:
            self.degradation_level = DegradationLevel.LIGHT
        
        print(f"降级级别: {self.degradation_level.value}")
    
    def get_strategy(self) -> Dict[str, Any]:
        """获取当前降级策略"""
        strategies = {
            DegradationLevel.NONE: {
                "use_cache": True,
                "use_reranking": True,
                "use_multi_query": True,
                "timeout_ms": 5000,
                "max_retries": 3
            },
            DegradationLevel.LIGHT: {
                "use_cache": True,
                "use_reranking": True,
                "use_multi_query": False,
                "timeout_ms": 3000,
                "max_retries": 2
            },
            DegradationLevel.MODERATE: {
                "use_cache": True,
                "use_reranking": False,
                "use_multi_query": False,
                "timeout_ms": 2000,
                "max_retries": 1
            },
            DegradationLevel.SEVERE: {
                "use_cache": True,
                "use_reranking": False,
                "use_multi_query": False,
                "timeout_ms": 1000,
                "max_retries": 0
            }
        }
        return strategies[self.degradation_level]

class ResilientRAGSystem:
    """具有容错能力的RAG系统"""
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.degradation_manager = RAGDegradationManager()
    
    def query_with_fallback(self, question: str) -> Dict[str, Any]:
        """带降级的查询"""
        strategy = self.degradation_manager.get_strategy()
        
        try:
            # 根据策略调整查询
            result = self.rag_system.query(question)
            
            self.degradation_manager.record_success()
            return {
                "answer": result.get("answer", ""),
                "status": "success",
                "degradation_level": self.degradation_manager.degradation_level.value
            }
            
        except Exception as e:
            self.degradation_manager.record_error(e)
            
            # 降级处理
            return self._fallback_query(question)
    
    def _fallback_query(self, question: str) -> Dict[str, Any]:
        """降级查询"""
        try:
            # 简化查询，跳过复杂步骤
            strategy = self.degradation_manager.get_strategy()
            
            # 直接使用向量检索
            docs = self.rag_system.vectorstore.similarity_search(question, k=3)
            context = "\n".join([doc.page_content for doc in docs])
            
            # 简单生成
            prompt = f"基于以下上下文回答：{context}\n问题：{question}"
            answer = self.rag_system.llm.invoke(prompt).content
            
            return {
                "answer": answer,
                "status": "degraded",
                "degradation_level": self.degradation_manager.degradation_level.value
            }
            
        except Exception as e:
            return {
                "answer": "系统暂时无法处理您的请求，请稍后重试。",
                "status": "failed",
                "error": str(e)
            }

# 使用示例
resilient_rag = ResilientRAGSystem(rag_system)
result = resilient_rag.query_with_fallback("什么是机器学习？")
print(f"状态: {result['status']}")
print(f"降级级别: {result['degradation_level']}")
```

### 10.16.4 监控与可观测性

```python
from dataclasses import dataclass
from typing import Dict, List, Any
from datetime import datetime
import time

@dataclass
class RAGTrace:
    """RAG追踪记录"""
    trace_id: str
    query: str
    start_time: float
    end_time: float = None
    stages: Dict[str, float] = None
    metadata: Dict[str, Any] = None

class RAGMonitor:
    """RAG监控器"""
    
    def __init__(self):
        self.traces: List[RAGTrace] = []
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_latency_ms": 0,
            "p95_latency_ms": 0
        }
    
    def start_trace(self, query: str) -> str:
        """开始追踪"""
        trace_id = f"trace_{int(time.time() * 1000)}"
        trace = RAGTrace(
            trace_id=trace_id,
            query=query,
            start_time=time.time(),
            stages={},
            metadata={}
        )
        self.traces.append(trace)
        self.metrics["total_queries"] += 1
        return trace_id
    
    def record_stage(self, trace_id: str, stage_name: str, duration_ms: float):
        """记录阶段耗时"""
        for trace in self.traces:
            if trace.trace_id == trace_id:
                if trace.stages is None:
                    trace.stages = {}
                trace.stages[stage_name] = duration_ms
                break
    
    def end_trace(self, trace_id: str, success: bool = True):
        """结束追踪"""
        for trace in self.traces:
            if trace.trace_id == trace_id:
                trace.end_time = time.time()
                
                if success:
                    self.metrics["successful_queries"] += 1
                else:
                    self.metrics["failed_queries"] += 1
                
                # 更新平均延迟
                total_latency = (trace.end_time - trace.start_time) * 1000
                n = self.metrics["total_queries"]
                self.metrics["avg_latency_ms"] = (
                    (self.metrics["avg_latency_ms"] * (n - 1) + total_latency) / n
                )
                break
    
    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        if not self.traces:
            return self.metrics
        
        latencies = [
            (t.end_time - t.start_time) * 1000 
            for t in self.traces 
            if t.end_time
        ]
        
        if latencies:
            latencies.sort()
            self.metrics["p95_latency_ms"] = latencies[int(len(latencies) * 0.95)]
        
        return self.metrics
    
    def export_traces(self, filepath: str):
        """导出追踪数据"""
        import json
        
        export_data = []
        for trace in self.traces:
            export_data.append({
                "trace_id": trace.trace_id,
                "query": trace.query,
                "start_time": trace.start_time,
                "end_time": trace.end_time,
                "duration_ms": (trace.end_time - trace.start_time) * 1000 if trace.end_time else None,
                "stages": trace.stages,
                "metadata": trace.metadata
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

# 使用示例
monitor = RAGMonitor()

# 追踪查询
trace_id = monitor.start_trace("什么是机器学习？")

# 记录各阶段耗时
monitor.record_stage(trace_id, "query_rewrite", 50)
monitor.record_stage(trace_id, "retrieval", 150)
monitor.record_stage(trace_id, "reranking", 100)
monitor.record_stage(trace_id, "generation", 500)

monitor.end_trace(trace_id, success=True)

print("监控摘要:", monitor.get_summary())
```

### 10.16.5 A/B测试框架

```python
from dataclasses import dataclass
from typing import Dict, List, Any
import random
from datetime import datetime

@dataclass
class ABTestExperiment:
    """A/B测试实验"""
    experiment_id: str
    name: str
    variants: List[str]
    traffic_split: Dict[str, float]  # variant_name -> percentage
    start_time: datetime
    end_time: datetime = None
    results: Dict[str, List] = None

class RAGABTestFramework:
    """RAG A/B测试框架"""
    
    def __init__(self):
        self.experiments: Dict[str, ABTestExperiment] = {}
        self.user_assignments: Dict[str, Dict[str, str]] = {}
    
    def create_experiment(self, name: str, variants: List[str], 
                         traffic_split: Dict[str, float]) -> str:
        """创建实验"""
        experiment_id = f"exp_{int(datetime.now().timestamp())}"
        
        experiment = ABTestExperiment(
            experiment_id=experiment_id,
            name=name,
            variants=variants,
            traffic_split=traffic_split,
            start_time=datetime.now(),
            results={v: [] for v in variants}
        )
        
        self.experiments[experiment_id] = experiment
        return experiment_id
    
    def assign_variant(self, experiment_id: str, user_id: str) -> str:
        """分配用户到变体"""
        if experiment_id not in self.experiments:
            raise ValueError("Experiment not found")
        
        if user_id not in self.user_assignments:
            self.user_assignments[user_id] = {}
        
        if experiment_id in self.user_assignments[user_id]:
            return self.user_assignments[user_id][experiment_id]
        
        # 基于流量比例分配
        experiment = self.experiments[experiment_id]
        rand = random.random()
        cumulative = 0
        
        for variant, percentage in experiment.traffic_split.items():
            cumulative += percentage
            if rand <= cumulative:
                self.user_assignments[user_id][experiment_id] = variant
                return variant
        
        return experiment.variants[-1]
    
    def record_metric(self, experiment_id: str, variant: str, 
                     metric_name: str, value: float):
        """记录指标"""
        if experiment_id not in self.experiments:
            return
        
        experiment = self.experiments[experiment_id]
        if variant not in experiment.results:
            experiment.results[variant] = []
        
        experiment.results[variant].append({
            "metric_name": metric_name,
            "value": value,
            "timestamp": datetime.now()
        })
    
    def analyze_results(self, experiment_id: str) -> Dict[str, Any]:
        """分析实验结果"""
        if experiment_id not in self.experiments:
            return {}
        
        experiment = self.experiments[experiment_id]
        analysis = {}
        
        for variant, records in experiment.results.items():
            if not records:
                continue
            
            # 按指标分组
            metrics = {}
            for record in records:
                metric_name = record["metric_name"]
                if metric_name not in metrics:
                    metrics[metric_name] = []
                metrics[metric_name].append(record["value"])
            
            # 计算统计量
            variant_analysis = {}
            for metric_name, values in metrics.items():
                import numpy as np
                variant_analysis[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "count": len(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
            
            analysis[variant] = variant_analysis
        
        return analysis
    
    def get_winner(self, experiment_id: str, metric_name: str) -> str:
        """获取获胜变体"""
        analysis = self.analyze_results(experiment_id)
        
        best_variant = None
        best_score = -1
        
        for variant, metrics in analysis.items():
            if metric_name in metrics:
                score = metrics[metric_name]["mean"]
                if score > best_score:
                    best_score = score
                    best_variant = variant
        
        return best_variant

# 使用示例
ab_test = RAGABTestFramework()

# 创建实验
exp_id = ab_test.create_experiment(
    name="RAG策略对比",
    variants=["baseline", "multi_query", "hyde"],
    traffic_split={"baseline": 0.33, "multi_query": 0.33, "hyde": 0.34}
)

# 记录指标
for i in range(100):
    variant = ab_test.assign_variant(exp_id, f"user_{i}")
    # 模拟指标
    score = random.uniform(0.7, 0.95) if variant != "baseline" else random.uniform(0.6, 0.9)
    ab_test.record_metric(exp_id, variant, "faithfulness", score)

# 分析结果
analysis = ab_test.analyze_results(exp_id)
print("实验分析:", analysis)

winner = ab_test.get_winner(exp_id, "faithfulness")
print("获胜变体:", winner)
```

---

## 10.17 高级RAG策略效果对比

### 10.17.1 策略效果量化对比

| 策略 | 召回率提升 | 精度提升 | 延迟增加 | 成本增加 | 实现复杂度 |
|------|-----------|---------|---------|---------|-----------|
| Multi-Query | +25-35% | +5-10% | +100ms | +2x LLM | ⭐⭐ |
| HyDE | +20-30% | +10-15% | +200ms | +1x LLM | ⭐⭐ |
| 查询改写 | +15-25% | +10-20% | +50ms | +0.5x LLM | ⭐⭐⭐ |
| 重排序 | +5-10% | +20-30% | +150ms | +0.5x LLM | ⭐⭐⭐ |
| Parent-Child | +10-15% | +15-25% | +20ms | +0.1x | ⭐⭐⭐⭐ |
| Self-RAG | +10-20% | +15-25% | +100ms | +1x LLM | ⭐⭐⭐⭐ |
| CRAG | +15-25% | +20-30% | +300ms | +2x LLM | ⭐⭐⭐⭐⭐ |
| Graph RAG | +30-40% | +25-35% | +500ms | +3x | ⭐⭐⭐⭐⭐ |
| 混合检索 | +30-40% | +20-30% | +100ms | +0.5x | ⭐⭐⭐ |

### 10.17.2 场景推荐策略

| 应用场景 | 推荐策略组合 | 预期效果 | 注意事项 |
|---------|-------------|---------|---------|
| 企业知识库 | 混合检索 + 重排序 + Self-RAG | 精度高、可靠性强 | 需要高质量文档 |
| 客服问答 | Multi-Query + 缓存 + 降级 | 响应快、稳定性好 | 关注用户体验 |
| 学术研究 | Graph RAG + HyDE + 重排序 | 知识深度强 | 构建成本高 |
| 实时搜索 | 缓存 + 异步 + 降级 | 延迟低 | 牺牲部分精度 |
| 复杂推理 | Agentic RAG + CRAG + 多轮 | 准确性高 | 延迟较高 |

### 10.17.3 最佳实践总结

| 实践领域 | 最佳实践 | 优先级 |
|---------|---------|-------|
| 分块策略 | 根据文档类型调整chunk_size，保持10-20%重叠 | 🔴 高 |
| Embedding选择 | 生产环境使用商业模型，中文场景用BGE/GTE | 🔴 高 |
| 检索优化 | 混合检索 + 重排序是最佳组合 | 🔴 高 |
| 缓存策略 | L1内存 + L2 Redis分层缓存 | 🟡 中 |
| 监控告警 | 完整的trace和metrics监控 | 🟡 中 |
| 错误处理 | 降级策略保证可用性 | 🟡 中 |
| A/B测试 | 持续优化策略参数 | 🟢 低 |
| 成本控制 | 缓存 + 异步 + 批处理 | 🟢 低 |

---

## 10.18 本章扩展小结

> **高级RAG策略核心要点**：
>
> 1. **评估框架**：建立多维度评估体系，持续监控系统质量
> 2. **参数调优**：分块参数、Embedding模型、检索参数都需要针对性优化
> 3. **策略组合**：单一策略效果有限，组合使用才能获得最佳效果
> 4. **生产模式**：缓存、异步、降级、监控是生产环境的必备组件
> 5. **持续优化**：A/B测试和数据分析驱动持续改进

 **下一步**：学习第11章，了解 Agent 的规划基础和思维链技术，探索更智能的AI系统构建方法。

不同的 Agent 实现方式各有优劣。下面的交互式对比可以帮助你选择合适的方案：

<div data-component="AgentArchitectureComparisonV8"></div>

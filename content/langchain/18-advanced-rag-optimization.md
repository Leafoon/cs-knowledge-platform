# Chapter 18: 高级 RAG 模式与优化

## 本章概览

在掌握了 RAG 基础架构和向量检索之后，本章将深入探讨**高级 RAG 模式**与**性能优化技术**，帮助你构建生产级、高质量的检索增强生成系统。我们将学习多种前沿 RAG 架构、查询优化策略、上下文压缩技术，以及如何通过重排序（Reranking）、混合检索、自查询（Self-Query）等方法显著提升检索质量。

本章重点：
- 高级 RAG 架构模式（Multi-Query、HyDE、Parent-Document、RAPTOR）
- 查询转换与优化（Query Transformation、Expansion、Decomposition）
- 上下文压缩与相关性过滤
- 重排序（Reranking）策略
- 混合检索（Dense + Sparse）
- Self-Query 与元数据过滤
- RAG 评估与优化循环

---

## 18.1 高级 RAG 架构模式

### 18.1.1 Naive RAG vs Advanced RAG

**Naive RAG（朴素 RAG）**的典型流程：
```
Query → Embedding → Vector Search → Top-K Docs → LLM → Answer
```

**痛点**：
- 查询与文档语义不匹配
- 检索到的文档噪声多
- 上下文窗口浪费
- 缺乏多跳推理能力

**Advanced RAG** 通过以下技术改进：
1. **Query Transformation**：优化查询表达
2. **Retrieval Augmentation**：多路检索、重排序
3. **Context Compression**：压缩无关信息
4. **Iterative Refinement**：多轮检索与推理

<div data-component="AdvancedRAGComparison"></div>

---

## 18.2 Multi-Query RAG

### 18.2.1 原理

用户的原始查询可能表达不精确。**Multi-Query RAG** 让 LLM 生成多个**语义相似但表达不同**的查询，扩大检索覆盖面。

**流程**：
```
Original Query → LLM 生成 N 个变体 → 并行检索 → 合并去重 → Rerank → LLM 生成答案
```

### 18.2.2 实现

```python
from langchain.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

llm = ChatOpenAI(temperature=0)
vectorstore = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

# Multi-Query Retriever
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    llm=llm,
    include_original=True  # 包含原始查询
)

# 单次调用，内部生成多个查询并合并结果
docs = retriever.get_relevant_documents("如何优化 RAG 系统性能？")

# 查看生成的查询变体
print(retriever.get_generated_queries("如何优化 RAG 系统性能？"))
# 输出示例：
# [
#   "如何优化 RAG 系统性能？",  # 原始
#   "提升检索增强生成系统效率的方法有哪些？",
#   "RAG 性能调优的最佳实践是什么？",
#   "怎样改进 RAG 的检索质量和速度？"
# ]
```

**优势**：
- 覆盖不同表达方式
- 提高召回率（Recall）
- 对模糊查询更鲁棒

**劣势**：
- 额外的 LLM 调用成本
- 可能引入噪声查询

---

## 18.3 HyDE（Hypothetical Document Embeddings）

### 18.3.1 核心思想

传统检索：`Query Embedding ↔ Document Embedding`

**HyDE**：让 LLM **先生成一个假设的答案文档**，然后用这个假设文档的 Embedding 去检索，因为**答案与答案的语义相似度更高**。

**流程**：
```
Query → LLM 生成假设答案 → Embedding(假设答案) → Vector Search → 真实文档 → LLM 生成最终答案
```

### 18.3.2 实现

```python
from langchain.chains import HypotheticalDocumentEmbedder
from langchain_openai import OpenAI, OpenAIEmbeddings

base_embeddings = OpenAIEmbeddings()
llm = OpenAI(temperature=0)

# HyDE Embedder
hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=base_embeddings,
    prompt_key="web_search"  # 内置 prompt 模板
)

# 生成假设文档并检索
vectorstore = Chroma(
    embedding_function=hyde_embeddings,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
docs = retriever.get_relevant_documents("PyTorch 的动态图和静态图有什么区别？")
```

**HyDE 内部流程**：
1. **Prompt**：
   ```
   Please write a passage to answer the question:
   Question: PyTorch 的动态图和静态图有什么区别？
   Passage:
   ```
2. **LLM 生成假设答案**（可能不准确，但语义接近真实答案）
3. **Embedding(假设答案)** → 检索相似文档
4. 用真实文档再次生成准确答案

**适用场景**：
- 专业领域问答（医疗、法律）
- 文档语言风格与查询差异大
- 需要高召回率的场景

---

## 18.4 Parent Document Retriever

### 18.4.1 问题背景

**矛盾**：
- **小 Chunk**：检索精准，但上下文不足
- **大 Chunk**：上下文完整，但检索噪声多

**解决方案**：
- **索引小 Chunk**（用于检索）
- **返回大 Chunk / 完整文档**（提供给 LLM）

### 18.4.2 实现

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# 父文档存储（内存或 Redis）
docstore = InMemoryStore()

# 大 Chunk（父文档）
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

# 小 Chunk（子文档，用于检索）
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

vectorstore = Chroma(
    collection_name="parent_docs",
    embedding_function=OpenAIEmbeddings()
)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 添加文档
retriever.add_documents(documents)

# 检索时：匹配小 Chunk，返回大 Chunk
docs = retriever.get_relevant_documents("什么是 Transformer？")
# 返回完整的父文档（2000 字符），而非匹配的子块（400 字符）
```

**优势**：
- 检索精度高（小块匹配更准确）
- 上下文完整（返回大块）
- 适合长文档场景（论文、技术文档）

---

## 18.5 RAPTOR（Recursive Abstractive Processing for Tree-Organized Retrieval）

### 18.5.1 核心思想

传统 RAG 只在**叶子节点**（原始文档块）检索。**RAPTOR** 构建**文档摘要树**，在多个层级检索：
- **叶子层**：原始文档块
- **中间层**：Cluster Summaries（多个块的摘要）
- **顶层**：Global Summary（整个文档集的摘要）

**优势**：支持**多粒度检索**，能回答高层次抽象问题。

### 18.5.2 架构

```
                  [Global Summary]
                  /              \
        [Cluster 1 Summary]   [Cluster 2 Summary]
           /     \                /     \
       [Doc1] [Doc2]          [Doc3] [Doc4]
```

检索时在所有层级并行搜索，整合不同粒度的信息。

### 18.5.3 实现思路（伪代码）

```python
# 1. 构建摘要树
def build_raptor_tree(documents, llm):
    # 叶子层：原始文档
    leaves = split_documents(documents)
    
    # 中间层：聚类并生成摘要
    clusters = cluster_embeddings(leaves, n_clusters=5)
    summaries = [llm.summarize(cluster) for cluster in clusters]
    
    # 递归构建更高层
    if len(summaries) > 1:
        upper_summaries = build_raptor_tree(summaries, llm)
    
    # 索引所有层级
    vectorstore.add_documents(leaves + summaries + upper_summaries)

# 2. 检索时在所有层级搜索
def raptor_retrieve(query, k=5):
    return vectorstore.similarity_search(query, k=k)
```

**LangChain 实现**（需自定义或使用 LlamaIndex 的 RAPTOR）：
```python
# 目前 LangChain 无内置 RAPTOR，但可通过自定义 Retriever 实现
# 参考：https://github.com/langchain-ai/langchain/discussions/15000
```

---

## 18.6 查询转换（Query Transformation）

<div data-component="QueryTransformationFlow"></div>

### 18.6.1 Query Rewriting

**目标**：优化查询表达，提高检索精度。

```python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

rewrite_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Given the following user question, rewrite it to be more specific and suitable for vector search:

Original Question: {question}

Rewritten Question:"""
)

rewrite_chain = LLMChain(llm=llm, prompt=rewrite_prompt)

# 示例
original = "这个怎么用？"
rewritten = rewrite_chain.run(question=original)
# 输出：如何使用这个软件功能？请提供详细步骤。
```

### 18.6.2 Query Decomposition（查询分解）

将复杂问题分解为多个子问题，逐步检索并回答。

```python
from langchain.chains import SequentialChain

# 1. 分解查询
decompose_prompt = PromptTemplate(
    template="""Break down the following complex question into 3 simpler sub-questions:

Question: {question}

Sub-questions:
1."""
)

# 2. 逐个回答子问题
# 3. 合并答案
```

**示例**：
- **复杂查询**："比较 PyTorch 和 TensorFlow 在分布式训练、模型部署和社区生态方面的优劣"
- **分解后**：
  1. PyTorch 和 TensorFlow 的分布式训练能力对比？
  2. 两者在模型部署方面的差异？
  3. 社区生态和工具链的成熟度对比？

### 18.6.3 Step-Back Prompting

生成更**抽象/高层次**的问题，先检索通用知识，再回答具体问题。

```python
step_back_prompt = PromptTemplate(
    template="""You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer.

Original Question: {question}

Step-back Question:"""
)

# 示例
original = "2023年诺贝尔物理学奖获得者阿秒激光的原理是什么？"
step_back = "阿秒激光的基本原理是什么？"  # 更通用，更容易检索到背景知识
```

---

## 18.7 上下文压缩（Contextual Compression）

### 18.7.1 问题

检索返回的文档包含大量**无关信息**，浪费 Token 并降低 LLM 性能。

### 18.7.2 LLMChainExtractor

使用 LLM 从检索文档中**提取与查询相关的片段**。

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 基础 Retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# 压缩器：提取相关片段
compressor = LLMChainExtractor.from_llm(llm)

# 组合
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 使用
docs = compression_retriever.get_relevant_documents(
    "LangChain 的 LCEL 是什么？"
)
# 返回：仅包含与 LCEL 相关的句子/段落，过滤掉无关内容
```

**内部流程**：
1. Base Retriever 检索 10 个文档
2. 对每个文档，Prompt LLM：
   ```
   Given the following context and question, extract only the parts relevant to answering the question.
   
   Context: [文档内容]
   Question: LangChain 的 LCEL 是什么？
   
   Relevant parts:
   ```
3. 返回压缩后的文档

### 18.7.3 EmbeddingsFilter

基于 **Embedding 相似度** 过滤文档，无需 LLM 调用（更快、更便宜）。

```python
from langchain.retrievers.document_compressors import EmbeddingsFilter

embeddings_filter = EmbeddingsFilter(
    embeddings=OpenAIEmbeddings(),
    similarity_threshold=0.76  # 相似度阈值
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=base_retriever
)
```

### 18.7.4 Pipeline Compressor（多级压缩）

```python
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter

# 1. 先分块（减少噪声）
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")

# 2. 再过滤（Embedding 相似度）
filter = EmbeddingsFilter(embeddings=OpenAIEmbeddings(), similarity_threshold=0.76)

# 3. 最后提取（LLM 精炼）
extractor = LLMChainExtractor.from_llm(llm)

# 组合
pipeline = DocumentCompressorPipeline(transformers=[splitter, filter, extractor])

compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline,
    base_retriever=base_retriever
)
```

---

## 18.8 重排序（Reranking）

### 18.8.1 为什么需要 Rerank？

向量检索（Embedding Similarity）是**粗排**，可能存在：
- 语义相似但不相关（Semantic drift）
- 缺乏对查询意图的深度理解

**Reranker** 是**精排模型**（通常是 Cross-Encoder），对候选文档重新打分。

### 18.8.2 Cohere Rerank

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.llms import Cohere

# Cohere Rerank（需 API Key）
compressor = CohereRerank(
    model="rerank-english-v2.0",
    top_n=3  # 返回 Top 3
)

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

rerank_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 流程：检索 20 个 → Rerank → 返回 Top 3
docs = rerank_retriever.get_relevant_documents("最新的 AI 突破是什么？")
```

### 18.8.3 自定义 Reranker（Cross-Encoder）

```python
from sentence_transformers import CrossEncoder
from langchain.schema import Document

# 加载 Cross-Encoder 模型
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_documents(query: str, documents: list[Document], top_k: int = 3):
    # 计算 Query-Doc 对的相关性分数
    pairs = [[query, doc.page_content] for doc in documents]
    scores = model.predict(pairs)
    
    # 排序
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]

# 使用
docs = base_retriever.get_relevant_documents("RAG 的最佳实践")
reranked_docs = rerank_documents("RAG 的最佳实践", docs, top_k=3)
```

**Rerank 收益**：
- **NDCG@10** 提升 15-30%
- **MRR** 提升 20-40%
- 减少 LLM 输入噪声

---

## 18.9 混合检索（Hybrid Search）

<div data-component="HybridSearchArchitecture"></div>

### 18.9.1 Dense + Sparse 检索

**Dense Retrieval**（向量检索）：
- 优势：语义理解强
- 劣势：对专有名词、精确匹配弱

**Sparse Retrieval**（BM25、TF-IDF）：
- 优势：精确匹配、专有名词召回好
- 劣势：无语义理解

**混合检索**：结合两者优势。

### 18.9.2 BM25 + Vector Search

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# 1. BM25 Retriever
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

# 2. Vector Retriever
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. Ensemble（加权融合）
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]  # BM25: 40%, Vector: 60%
)

# 使用
docs = ensemble_retriever.get_relevant_documents("PyTorch Lightning 教程")
# 内部：BM25 检索 5 个，Vector 检索 5 个，合并去重，加权排序
```

**权重调优**：
- 通用场景：`[0.5, 0.5]`
- 语义优先：`[0.3, 0.7]`
- 精确匹配优先：`[0.7, 0.3]`

### 18.9.3 Elasticsearch 混合检索

```python
from langchain_community.vectorstores import ElasticsearchStore

vectorstore = ElasticsearchStore(
    es_url="http://localhost:9200",
    index_name="my_index",
    embedding=OpenAIEmbeddings(),
    strategy=ElasticsearchStore.SparseVectorRetrievalStrategy()  # Hybrid
)

# Elasticsearch 内部使用 RRF（Reciprocal Rank Fusion）融合结果
docs = vectorstore.similarity_search("LangChain 教程", k=5)
```

---

## 18.10 Self-Query Retriever（自查询）

### 18.10.1 问题

用户查询常包含**元数据过滤条件**：
- "2023年后发表的关于 Transformer 的论文"
- "Python 相关的初学者教程"

传统向量检索无法利用这些**结构化过滤**。

### 18.10.2 Self-Query 原理

LLM 将自然语言查询分解为：
1. **语义查询**（用于向量检索）
2. **元数据过滤器**（SQL-like 条件）

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# 定义元数据结构
metadata_field_info = [
    AttributeInfo(
        name="year",
        description="论文发表年份",
        type="integer",
    ),
    AttributeInfo(
        name="language",
        description="编程语言",
        type="string",
    ),
    AttributeInfo(
        name="difficulty",
        description="难度等级: beginner, intermediate, advanced",
        type="string",
    ),
]

document_content_description = "技术文档和教程"

# Self-Query Retriever
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
)

# 使用
docs = retriever.get_relevant_documents(
    "2022年后发表的关于 PyTorch 的高级教程"
)

# 内部生成的结构化查询：
# {
#   "query": "PyTorch advanced tutorial",
#   "filter": {"year": {"$gte": 2022}, "difficulty": "advanced"}
# }
```

**支持的 VectorStore**：
- Chroma
- Pinecone
- Weaviate
- Qdrant
- Milvus

---

## 18.11 RAG 评估与优化

### 18.11.1 评估维度

**检索质量**：
- **Recall@K**：Top-K 中包含相关文档的比例
- **Precision@K**：Top-K 中相关文档的比例
- **MRR（Mean Reciprocal Rank）**：第一个相关文档的排名倒数
- **NDCG（Normalized Discounted Cumulative Gain）**：考虑排序的综合指标

**生成质量**：
- **Faithfulness**：答案是否忠于检索文档（无幻觉）
- **Answer Relevance**：答案是否回答了问题
- **Context Relevance**：检索文档是否相关

### 18.11.2 使用 RAGAS 评估

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

# 构建评估数据集
eval_data = {
    "question": ["What is LangChain?", "How to use LCEL?"],
    "answer": ["LangChain is a framework...", "LCEL is..."],
    "contexts": [["LangChain documentation..."], ["LCEL guide..."]],
    "ground_truths": [["LangChain is..."], ["LCEL allows..."]]
}

dataset = Dataset.from_dict(eval_data)

# 评估
result = evaluate(
    dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

print(result)
# {
#   "context_precision": 0.85,
#   "context_recall": 0.92,
#   "faithfulness": 0.88,
#   "answer_relevancy": 0.91
# }
```

### 18.11.3 优化循环

```
评估 → 发现问题 → 优化策略 → 重新评估
```

**常见问题与对策**：

| 问题 | 指标 | 优化策略 |
|------|------|----------|
| 检索不到相关文档 | Context Recall ↓ | 调整 Chunk Size、使用 Multi-Query、混合检索 |
| 检索噪声多 | Context Precision ↓ | Reranking、上下文压缩、Self-Query |
| 答案不忠于文档 | Faithfulness ↓ | 优化 Prompt、减少 LLM Temperature |
| 答案偏离问题 | Answer Relevance ↓ | Query Rewriting、Few-Shot Examples |

---

## 18.12 完整高级 RAG Pipeline 示例

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
    MultiQueryRetriever,
)
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
    CohereRerank,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 1. 加载文档并分块
documents = load_documents("./docs")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
splits = text_splitter.split_documents(documents)

# 2. 构建向量存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 3. Multi-Query Retriever（查询扩展）
llm = ChatOpenAI(temperature=0, model="gpt-4")
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

# 4. 上下文压缩 Pipeline
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
redundant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
reranker = CohereRerank(top_n=5)

pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, reranker]
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=multi_query_retriever
)

# 5. QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=compression_retriever,
    return_source_documents=True,
)

# 6. 使用
result = qa_chain({"query": "LangChain 的 LCEL 相比传统 Chain 有什么优势？"})

print("答案:", result["result"])
print("\n来源文档:")
for doc in result["source_documents"]:
    print(f"- {doc.metadata['source']}: {doc.page_content[:100]}...")
```

**完整流程**：
```
Query
  ↓
Multi-Query（生成 3-5 个查询变体）
  ↓
并行检索（每个查询检索 20 个文档）
  ↓
合并去重
  ↓
分块（300 字符）
  ↓
Embedding 过滤（相似度 > 0.76）
  ↓
Cohere Rerank（Top 5）
  ↓
LLM 生成答案
```

---

## 18.13 最佳实践与经验总结

### 18.13.1 Chunk Size 选择

| 场景 | 推荐 Chunk Size | 说明 |
|------|----------------|------|
| 问答（FAQ） | 200-400 | 精确匹配优先 |
| 长文档摘要 | 800-1500 | 需要完整上下文 |
| 代码检索 | 500-1000 | 保持函数/类完整性 |
| 聊天机器人 | 300-600 | 平衡精度与上下文 |

**动态 Chunking**：使用 Parent Document Retriever，检索用小块，返回大块。

### 18.13.2 Embedding Model 选择

| 模型 | 维度 | 速度 | 质量 | 成本 |
|------|------|------|------|------|
| OpenAI text-embedding-3-small | 1536 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | $ |
| OpenAI text-embedding-3-large | 3072 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $$ |
| Cohere embed-multilingual-v3.0 | 1024 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $ |
| sentence-transformers (本地) | 384-768 | ⭐⭐⭐⭐ | ⭐⭐⭐ | Free |

### 18.13.3 检索策略选择决策树

```
是否需要精确匹配（专有名词、代码）？
  ├─ 是 → 混合检索（BM25 + Vector）
  └─ 否 → 纯向量检索
      ├─ 查询模糊/多样？ → Multi-Query RAG
      ├─ 需要多粒度信息？ → Parent Document / RAPTOR
      ├─ 有元数据过滤需求？ → Self-Query Retriever
      └─ 上下文噪声多？ → 上下文压缩 + Reranking
```

### 18.13.4 成本优化

**Embedding 成本**（以 1M tokens 为例）：
- OpenAI ada-002: $0.10
- OpenAI text-embedding-3-small: $0.02
- 本地模型: $0

**优化策略**：
1. **缓存 Embeddings**：文档不变则复用
2. **增量索引**：只对新文档 Embedding
3. **混合检索**：BM25 免费，减少向量检索依赖
4. **批量处理**：批量 Embedding 降低延迟

### 18.13.5 延迟优化

| 优化点 | 方法 | 延迟收益 |
|--------|------|----------|
| 向量检索 | 使用 HNSW 索引、GPU 加速 | -50% |
| Reranking | 本地 Cross-Encoder（避免 API） | -70% |
| LLM 调用 | 流式输出、使用更快模型 | -30% |
| 上下文压缩 | 用 Embedding Filter 替代 LLM Extractor | -80% |

---

## 18.14 未来发展方向

1. **Fine-tuned Embeddings**：针对特定领域微调 Embedding 模型
2. **LLM-Reranker**：使用小型 LLM（如 GPT-3.5）作为 Reranker
3. **Active Retrieval**：LLM 主动决定何时检索、检索什么
4. **Multimodal RAG**：图像、表格、代码的联合检索
5. **Agentic RAG**：Agent 自主规划多轮检索策略

---

## 本章小结

本章深入探讨了高级 RAG 模式与优化技术，涵盖：

✅ **高级架构**：Multi-Query、HyDE、Parent-Document、RAPTOR  
✅ **查询优化**：Query Rewriting、Decomposition、Step-Back Prompting  
✅ **上下文压缩**：LLMChainExtractor、EmbeddingsFilter、Pipeline  
✅ **重排序**：Cohere Rerank、Cross-Encoder  
✅ **混合检索**：BM25 + Vector、Elasticsearch  
✅ **Self-Query**：自动提取元数据过滤条件  
✅ **评估与优化**：RAGAS、优化循环、最佳实践  

通过这些技术，你可以构建**生产级、高性能**的 RAG 系统，显著提升检索质量和生成准确性。

---

## 扩展阅读

- [LangChain Retrieval 文档](https://python.langchain.com/docs/modules/data_connection/retrievers/)
- [Advanced RAG Techniques (Paper)](https://arxiv.org/abs/2312.10997)
- [RAPTOR Paper](https://arxiv.org/abs/2401.18059)
- [RAGAS 评估框架](https://docs.ragas.io/)
- [Cohere Rerank API](https://docs.cohere.com/docs/reranking)

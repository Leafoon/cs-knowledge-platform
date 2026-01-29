# Chapter 13: 向量存储与检索

> **本章目标**：深入学习向量数据库（VectorStore）的核心概念与主流实现，掌握高效检索策略（相似度搜索、MMR、混合检索），理解索引构建与优化技术，为生产级 RAG 系统提供高性能检索能力。

## 13.1 VectorStore 抽象

### 13.1.1 核心接口设计

LangChain 的 `VectorStore` 抽象定义了向量数据库的统一接口：

```python
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from typing import List

class VectorStore:
    """向量存储抽象基类"""
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档（自动 Embedding）"""
        pass
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Document]:
        """相似度搜索（返回 Top-K 文档）"""
        pass
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[tuple[Document, float]]:
        """相似度搜索（带分数）"""
        pass
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[Document]:
        """MMR 搜索（平衡相关性与多样性）"""
        pass
```

**关键方法**：

| 方法 | 说明 | 使用场景 |
|------|------|----------|
| `add_documents()` | 批量添加文档 | 索引构建 |
| `similarity_search()` | 基础相似度搜索 | 最常用 |
| `similarity_search_with_score()` | 返回相似度分数 | 需要置信度 |
| `max_marginal_relevance_search()` | MMR 多样性搜索 | 避免重复结果 |
| `delete()` | 删除文档 | 索引维护 |
| `as_retriever()` | 转换为 Retriever | 链式集成 |

### 13.1.2 相似度度量

**余弦相似度（Cosine Similarity）** - 最常用

$$
\text{cosine\_similarity}(\mathbf{A}, \mathbf{B}) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}
$$

```python
import numpy as np

def cosine_similarity(vec_a, vec_b):
    """计算余弦相似度（范围：-1 到 1）"""
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

# 示例
vec_a = np.array([1, 2, 3])
vec_b = np.array([4, 5, 6])
similarity = cosine_similarity(vec_a, vec_b)
print(f"Cosine Similarity: {similarity:.4f}")  # 0.9746
```

**欧氏距离（Euclidean Distance）**

$$
\text{euclidean}(\mathbf{A}, \mathbf{B}) = \sqrt{\sum_{i=1}^{n} (A_i - B_i)^2}
$$

**点积（Dot Product）**

$$
\text{dot\_product}(\mathbf{A}, \mathbf{B}) = \sum_{i=1}^{n} A_i \cdot B_i
$$

**选择建议**：

- **余弦相似度**：适用于文本嵌入（归一化向量），主流选择
- **欧氏距离**：适用于未归一化向量
- **点积**：适用于已归一化向量（等价于余弦相似度）

### 13.1.3 异步操作

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# 同步搜索
results = vectorstore.similarity_search("LangChain tutorial")

# 异步搜索（高并发场景）
import asyncio

async def async_search():
    results = await vectorstore.asimilarity_search("LangChain tutorial", k=5)
    return results

results = asyncio.run(async_search())
```

**异步优势**：

- 并发处理多个查询
- 不阻塞主线程
- 提升吞吐量（适用于 API 服务）

---

## 13.2 主流 VectorStore 集成

### 13.2.1 Chroma - 轻量级本地向量数据库

**特点**：开源、嵌入式、适合开发与小规模部署

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader

# 加载文档
loader = TextLoader("data/docs.txt")
documents = loader.load()

# 创建 Chroma 向量库
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"  # 持久化目录
)

# 搜索
results = vectorstore.similarity_search("What is LangChain?", k=3)
for doc in results:
    print(doc.page_content[:100])
```

**持久化与加载**：

```python
# 保存（自动持久化到 persist_directory）
vectorstore.persist()

# 加载已有数据库
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
```

**增量更新**：

```python
# 添加新文档
new_docs = [Document(page_content="New content", metadata={"source": "new.txt"})]
vectorstore.add_documents(new_docs)

# 删除文档
doc_ids = vectorstore.add_documents(new_docs)
vectorstore.delete(ids=doc_ids)
```

### 13.2.2 Pinecone - 云向量数据库

**特点**：托管服务、高性能、弹性扩展、生产就绪

```python
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# 初始化 Pinecone
pc = Pinecone(api_key="YOUR_API_KEY")

# 创建索引
index_name = "langchain-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI Embeddings 维度
        metric="cosine",  # 'cosine' | 'euclidean' | 'dotproduct'
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# 创建向量库
vectorstore = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    index_name=index_name
)

# 搜索
results = vectorstore.similarity_search("RAG architecture", k=5)
```

**Namespace 隔离**（多租户）：

```python
# 不同用户使用不同 namespace
user_vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
    namespace="user_123"  # 用户隔离
)

user_vectorstore.add_documents(user_docs)
results = user_vectorstore.similarity_search("query", k=3)
```

**Metadata 过滤**：

```python
# 按元数据过滤
results = vectorstore.similarity_search(
    "LangChain",
    k=5,
    filter={"source": {"$eq": "documentation"}}
)
```

### 13.2.3 Weaviate - 开源向量搜索引擎

**特点**：支持混合搜索（BM25 + Vector）、GraphQL API、Schema 管理

```python
from langchain_weaviate import WeaviateVectorStore
import weaviate

# 连接 Weaviate
client = weaviate.Client(
    url="http://localhost:8080",
    auth_client_secret=weaviate.AuthApiKey(api_key="YOUR_API_KEY")
)

# 创建向量库
vectorstore = WeaviateVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    client=client,
    index_name="LangChainDocs",
    text_key="content"
)

# 搜索
results = vectorstore.similarity_search("vector database", k=4)
```

**混合搜索**（BM25 + Vector）：

```python
# Weaviate 原生支持
results = client.query.get(
    "LangChainDocs",
    ["content", "source"]
).with_hybrid(
    query="LangChain",
    alpha=0.5  # 0=纯BM25, 1=纯Vector, 0.5=混合
).with_limit(5).do()
```

### 13.2.4 Qdrant - 高性能向量数据库

**特点**：Rust 实现、高性能、支持过滤与分组

```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# 初始化客户端
client = QdrantClient(url="http://localhost:6333")

# 创建 Collection
collection_name = "langchain_docs"
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE
    )
)

# 创建向量库
vectorstore = QdrantVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name=collection_name,
    url="http://localhost:6333"
)

# 搜索
results = vectorstore.similarity_search("embedding models", k=3)
```

**Payload 过滤**：

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

# 复杂过滤
results = vectorstore.similarity_search(
    "LangChain",
    k=5,
    filter=Filter(
        must=[
            FieldCondition(
                key="metadata.source",
                match=MatchValue(value="official_docs")
            )
        ]
    )
)
```

### 13.2.5 FAISS - Facebook 向量索引库

**特点**：本地、高性能、适合大规模离线索引

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# 创建 FAISS 索引
vectorstore = FAISS.from_documents(documents, embeddings)

# 保存索引
vectorstore.save_local("faiss_index")

# 加载索引
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True  # 信任本地文件
)

# 搜索
results = vectorstore.similarity_search_with_score("RAG tutorial", k=3)
for doc, score in results:
    print(f"Score: {score:.4f} | {doc.page_content[:100]}")
```

**增量添加**：

```python
# 添加新文档
new_vectorstore = FAISS.from_documents(new_docs, embeddings)
vectorstore.merge_from(new_vectorstore)  # 合并索引
```

### 13.2.6 Milvus - 分布式向量数据库

**特点**：云原生、分布式、支持 GPU 加速、PB 级规模

```python
from langchain_milvus import Milvus
from pymilvus import connections

# 连接 Milvus
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

# 创建向量库
vectorstore = Milvus.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="langchain_collection",
    connection_args={"host": "localhost", "port": "19530"}
)

# 搜索
results = vectorstore.similarity_search("distributed systems", k=5)
```

**分区管理**（Partition）：

```python
# 按分区存储（如按日期分区）
vectorstore = Milvus(
    embedding_function=embeddings,
    collection_name="docs",
    partition_key_field="date"
)

vectorstore.add_documents(
    documents,
    partition_name="2024-01"
)

# 仅搜索特定分区
results = vectorstore.similarity_search(
    "query",
    k=5,
    partition_names=["2024-01"]
)
```

### 13.2.7 性能与成本对比

<div data-component="VectorStoreComparison"></div>

| VectorStore | 部署方式 | 性能 | 成本 | 适用场景 |
|-------------|---------|------|------|----------|
| **Chroma** | 本地 | 中 | 免费 | 开发、小规模 |
| **Pinecone** | 云服务 | 高 | $$ | 生产、企业 |
| **Weaviate** | 自托管/云 | 高 | $ | 混合搜索需求 |
| **Qdrant** | 自托管/云 | 非常高 | $ | 高性能需求 |
| **FAISS** | 本地 | 非常高 | 免费 | 离线批处理 |
| **Milvus** | 自托管/云 | 极高 | $$ | PB 级数据 |

---

## 13.3 Retriever 高级特性

### 13.3.1 VectorStoreRetriever 基础

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(documents, OpenAIEmbeddings())

# 转换为 Retriever（LCEL 集成）
retriever = vectorstore.as_retriever(
    search_type="similarity",  # 'similarity' | 'mmr' | 'similarity_score_threshold'
    search_kwargs={"k": 5}
)

# 在链中使用
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("""
根据以下上下文回答问题：

{context}

问题：{question}
""")

llm = ChatOpenAI()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

answer = chain.invoke("What is RAG?")
```

### 13.3.2 search_type 详解

#### **similarity - 相似度搜索**

最基础、最常用的搜索方式：

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # 返回最相似的 3 个文档
)

docs = retriever.invoke("LangChain tutorial")
```

#### **mmr - 最大边际相关性（Maximum Marginal Relevance）**

**解决问题**：避免检索结果过于相似（冗余）

$$
\text{MMR} = \arg\max_{d_i \in R \setminus S} \left[ \lambda \cdot \text{Sim}(d_i, Q) - (1-\lambda) \cdot \max_{d_j \in S} \text{Sim}(d_i, d_j) \right]
$$

其中：
- $Q$：查询向量
- $R$：候选集
- $S$：已选集
- $\lambda$：权衡参数（0=多样性优先，1=相关性优先）

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,           # 最终返回 5 个文档
        "fetch_k": 20,    # 先检索 20 个候选
        "lambda_mult": 0.5  # 平衡相关性与多样性
    }
)

docs = retriever.invoke("machine learning algorithms")
```

**lambda_mult 调优**：

- `lambda_mult=1.0`：退化为 similarity（纯相关性）
- `lambda_mult=0.5`：平衡（推荐）
- `lambda_mult=0.0`：极端多样性（可能丢失相关性）

#### **similarity_score_threshold - 阈值过滤**

仅返回相似度 **超过阈值** 的文档：

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.8,  # 仅返回相似度 >= 0.8 的文档
        "k": 10  # 最多返回 10 个
    }
)

docs = retriever.invoke("vector database")
# 如果没有文档超过 0.8，则返回空列表
```

**阈值选择**：

- **0.9+**：极高相关性（可能返回太少）
- **0.7-0.8**：高相关性（推荐）
- **0.5-0.6**：中等相关性
- **<0.5**：可能包含不相关内容

### 13.3.3 search_kwargs 配置

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,                  # 最终返回数量
        "fetch_k": 20,           # 候选集大小
        "lambda_mult": 0.5,      # MMR 权衡参数
        "filter": {"source": "official_docs"},  # Metadata 过滤
        "score_threshold": 0.7   # 最低分数阈值（部分实现支持）
    }
)
```

### 13.3.4 as_retriever() 快捷方法

```python
# 方式 1：使用默认参数
retriever = vectorstore.as_retriever()

# 方式 2：自定义参数
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.6}
)

# 方式 3：手动创建 Retriever
from langchain_core.retrievers import VectorStoreRetriever

retriever = VectorStoreRetriever(
    vectorstore=vectorstore,
    search_type="similarity",
    search_kwargs={"k": 5}
)
```

---

## 13.4 混合检索

### 13.4.1 BM25 + Vector 组合

**BM25（Best Matching 25）**：经典基于词频的检索算法

**优势组合**：

- **BM25**：擅长精确关键词匹配
- **Vector**：擅长语义相似性匹配

```python
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 1. BM25 Retriever
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

# 2. Vector Retriever
vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. Ensemble Retriever（混合）
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]  # BM25: 40%, Vector: 60%
)

# 使用
docs = ensemble_retriever.invoke("LangChain LCEL tutorial")
```

**权重调优**：

```python
# 测试不同权重组合
weight_combinations = [
    (0.3, 0.7),  # Vector 为主
    (0.5, 0.5),  # 平衡
    (0.7, 0.3),  # BM25 为主
]

for bm25_weight, vector_weight in weight_combinations:
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight]
    )
    
    docs = retriever.invoke(query)
    # 评估检索质量（Precision、Recall、NDCG 等）
```

### 13.4.2 EnsembleRetriever 详解

```python
from langchain.retrievers import EnsembleRetriever

# 支持 2+ 个检索器
ensemble = EnsembleRetriever(
    retrievers=[
        bm25_retriever,
        vector_retriever,
        another_retriever
    ],
    weights=[0.3, 0.5, 0.2],  # 权重和 = 1.0
    c=60  # RRF（Reciprocal Rank Fusion）参数
)
```

**RRF（Reciprocal Rank Fusion）**：

$$
\text{RRF}(d) = \sum_{r \in R} \frac{1}{c + \text{rank}_r(d)}
$$

其中 $c$ 通常取 60（论文推荐值）

### 13.4.3 Reranking 策略

使用 **Cross-Encoder** 模型重排序检索结果：

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

# 基础检索器
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Cohere Rerank
compressor = CohereRerank(
    model="rerank-english-v2.0",
    top_n=5  # 重排序后仅保留 Top-5
)

reranking_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

docs = reranking_retriever.invoke("deep learning frameworks")
```

**Reranking 流程**：

```
查询 → 向量检索 Top-20 → Cross-Encoder 重排序 → 返回 Top-5
```

**性能提升**：

- ✅ 精度提升 10-20%（NDCG@10）
- ❌ 延迟增加 200-500ms（Cross-Encoder 推理）
- ❌ 成本上升（Rerank API 调用）

<div data-component="HybridRetrievalFlow"></div>

---

## 13.5 索引管理

### 13.5.1 索引构建优化

#### **批量索引**

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# ❌ 低效：逐个添加
for doc in documents:
    vectorstore.add_documents([doc])

# ✅ 高效：批量添加
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    vectorstore.add_documents(batch)
```

#### **并行嵌入**

```python
import asyncio
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

async def embed_batch_async(texts, batch_size=100):
    """异步批量嵌入"""
    tasks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        task = embeddings.aembed_documents(batch)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return [vec for batch in results for vec in batch]

texts = [doc.page_content for doc in documents]
vectors = asyncio.run(embed_batch_async(texts))
```

#### **索引参数调优**

**FAISS 索引类型**：

```python
import faiss
from langchain_community.vectorstores import FAISS

# 1. Flat Index（精确搜索，速度慢）
index = faiss.IndexFlatL2(dimension)

# 2. IVF Index（近似搜索，速度快）
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist=100)  # nlist: 聚类中心数

# 3. HNSW Index（图索引，平衡精度与速度）
index = faiss.IndexHNSWFlat(dimension, M=32)  # M: 每节点连接数

# 使用自定义索引
vectorstore = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)
```

**Pinecone Pod 类型**：

```python
from pinecone import Pinecone, PodSpec

pc = Pinecone(api_key="xxx")

# 不同 Pod 类型
pc.create_index(
    name="high-performance",
    dimension=1536,
    metric="cosine",
    spec=PodSpec(
        environment="us-east1-gcp",
        pod_type="p1.x1",  # 'p1.x1' | 'p1.x2' | 'p1.x4' | 's1.x1' (高性能)
        pods=2,  # Pod 数量（并行查询）
        replicas=1  # 副本数（高可用）
    )
)
```

### 13.5.2 增量索引更新

#### **添加新文档**

```python
# Chroma
new_docs = [Document(page_content="New content", metadata={"id": "new_1"})]
vectorstore.add_documents(new_docs)

# Pinecone
vectorstore.add_documents(new_docs, namespace="production")

# FAISS（需要重新保存）
vectorstore.add_documents(new_docs)
vectorstore.save_local("faiss_index")
```

#### **更新已有文档**

```python
# 1. 删除旧文档
vectorstore.delete(ids=["doc_123"])

# 2. 添加新版本
updated_doc = Document(
    page_content="Updated content",
    metadata={"id": "doc_123", "version": 2}
)
vectorstore.add_documents([updated_doc])
```

#### **定期重建索引**

```python
from datetime import datetime, timedelta

def rebuild_index_if_needed(vectorstore, rebuild_interval_days=7):
    """每 N 天重建索引（优化性能）"""
    last_rebuild = vectorstore.metadata.get("last_rebuild")
    
    if not last_rebuild or (datetime.now() - last_rebuild) > timedelta(days=rebuild_interval_days):
        # 重建索引
        all_docs = vectorstore.get_all_documents()
        vectorstore.delete_collection()
        vectorstore = Chroma.from_documents(all_docs, embeddings)
        vectorstore.metadata["last_rebuild"] = datetime.now()
    
    return vectorstore
```

### 13.5.3 索引版本管理

```python
from datetime import datetime

class VersionedVectorStore:
    """版本化向量库"""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.current_version = self.get_latest_version()
    
    def get_latest_version(self) -> str:
        """获取最新版本号"""
        # 格式：v20240115_120000
        return datetime.now().strftime("v%Y%m%d_%H%M%S")
    
    def create_new_version(self, documents):
        """创建新版本索引"""
        version = self.get_latest_version()
        version_path = f"{self.base_path}/{version}"
        
        vectorstore = Chroma.from_documents(
            documents,
            embeddings,
            persist_directory=version_path
        )
        
        # 记录版本元数据
        with open(f"{version_path}/metadata.json", "w") as f:
            json.dump({
                "version": version,
                "doc_count": len(documents),
                "created_at": datetime.now().isoformat()
            }, f)
        
        return vectorstore
    
    def rollback(self, version: str):
        """回滚到指定版本"""
        version_path = f"{self.base_path}/{version}"
        return Chroma(
            persist_directory=version_path,
            embedding_function=embeddings
        )

# 使用
versioned_store = VersionedVectorStore("./vector_versions")
vectorstore = versioned_store.create_new_version(documents)

# 回滚
vectorstore = versioned_store.rollback("v20240101_100000")
```

### 13.5.4 索引清理与维护

#### **删除过期文档**

```python
from datetime import datetime, timedelta

def cleanup_old_documents(vectorstore, days=30):
    """删除 30 天前的文档"""
    cutoff_date = datetime.now() - timedelta(days=days)
    
    # 假设 metadata 中有 'timestamp' 字段
    all_docs = vectorstore.get()  # 获取所有文档
    
    old_doc_ids = [
        doc_id for doc_id, metadata in zip(all_docs['ids'], all_docs['metadatas'])
        if datetime.fromisoformat(metadata.get('timestamp', '9999-12-31')) < cutoff_date
    ]
    
    if old_doc_ids:
        vectorstore.delete(ids=old_doc_ids)
        print(f"Deleted {len(old_doc_ids)} old documents")
```

#### **去重**

```python
import hashlib

def deduplicate_documents(documents):
    """基于内容 hash 去重"""
    seen_hashes = set()
    unique_docs = []
    
    for doc in documents:
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)
    
    print(f"Removed {len(documents) - len(unique_docs)} duplicates")
    return unique_docs

# 使用
unique_docs = deduplicate_documents(documents)
vectorstore.add_documents(unique_docs)
```

#### **索引压缩**（FAISS）

```python
# 使用 Product Quantization 压缩索引（减少内存占用）
import faiss

dimension = 1536
index = faiss.IndexFlatL2(dimension)

# 添加文档
vectors = embeddings.embed_documents([doc.page_content for doc in documents])
index.add(np.array(vectors))

# 压缩索引
compressed_index = faiss.IndexIVFPQ(
    index,
    dimension,
    nlist=100,  # 聚类中心
    M=8,        # 子向量数（越小压缩率越高）
    nbits=8     # 每个子向量的比特数
)

compressed_index.train(np.array(vectors))
compressed_index.add(np.array(vectors))

# 内存占用减少 ~90%，检索速度提升 ~10x
```

<div data-component="SimilaritySearchDemo"></div>

---

## 本章小结

本章深入学习了向量存储与检索的核心技术，涵盖以下内容：

✅ **VectorStore 抽象**：理解统一接口设计、相似度度量（余弦/欧氏/点积）、异步操作  
✅ **主流实现**：掌握 Chroma、Pinecone、Weaviate、Qdrant、FAISS、Milvus 的使用与特点  
✅ **Retriever 高级特性**：学习 similarity、mmr、similarity_score_threshold 三种搜索模式  
✅ **混合检索**：掌握 BM25 + Vector、EnsembleRetriever、Reranking 等优化策略  
✅ **索引管理**：学习批量索引、增量更新、版本管理、去重清理等运维技术

**关键要点**：

1. **选择合适的 VectorStore**：开发用 Chroma，生产用 Pinecone/Qdrant，大规模用 Milvus
2. **MMR 平衡相关性与多样性**：避免检索结果过于相似，提升用户体验
3. **混合检索提升精度**：BM25 擅长关键词，Vector 擅长语义，组合效果最佳
4. **索引需要持续维护**：增量更新、版本管理、去重清理缺一不可

**下一章预告**：Chapter 14 将深入学习高级 RAG 技术，包括 Contextual Compression（上下文压缩）、Multi-Query Retrieval（多查询检索）、Parent Document Retrieval（父文档检索）、Self-Query（自查询）等前沿方法，以及 RAG 系统的评估指标与优化技巧。

---

## 扩展阅读

- [LangChain VectorStore 官方文档](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
- [VectorStore Retriever 完整指南](https://python.langchain.com/docs/modules/data_connection/retrievers/)
- [Pinecone 最佳实践](https://docs.pinecone.io/guides/data/understanding-hybrid-search)
- [FAISS 性能优化](https://github.com/facebookresearch/faiss/wiki)
- [Weaviate 混合搜索](https://weaviate.io/developers/weaviate/search/hybrid)
- [Maximum Marginal Relevance 论文](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf)

# Chapter 12: RAG 基础架构

> **本章目标**：全面掌握检索增强生成（RAG）的核心原理与实现方法，学习 Document Loaders、Text Splitters、Embeddings 等基础组件的使用，理解文档处理流程与优化策略，为构建生产级 RAG 系统打下坚实基础。

## 12.1 RAG 原理与动机

### 12.1.1 为什么需要 RAG？

大语言模型（LLM）虽然强大，但存在三个核心局限：

1. **知识截止日期**：模型训练数据有时间界限，无法回答最新信息
2. **领域知识缺失**：通用模型对企业私有数据、专业领域知识了解有限
3. **幻觉问题**：模型可能生成看似合理但实际错误的内容

**RAG（Retrieval-Augmented Generation）**通过**外部知识检索**解决这些问题。

**核心优势**：

- ✅ **实时性**：可随时更新知识库，无需重新训练模型
- ✅ **可追溯**：答案基于检索到的具体文档，可验证来源
- ✅ **成本低**：相比 Fine-tuning，无需大量标注数据与算力
- ✅ **可控性**：可通过知识库管理控制模型输出范围

### 12.1.2 RAG vs Fine-tuning 对比

| 维度 | RAG | FINE-TUNING |
|------|-----|-------------|
| 知识更新 | 动态更新知识库（秒级） | 需要重新训练（小时/天级） |
| 成本 | 低（仅 Embedding + 检索） | 高（需要 GPU 训练） |
| 数据需求 | 无需标注，原始文档即可 | 需要大量高质量标注数据 |
| 可解释性 | 强（可展示检索文档） | 弱（黑盒模型） |
| 适用场景 | 知识密集型、频繁更新 | 任务特化、风格适配 |

### 12.1.3 RAG 三种架构模式

<div data-component="RAGArchitectureDiagram"></div>

**Naive RAG（基础架构）**：
- 简单直接的查询 → 嵌入 → 检索 → 生成流程
- 适合快速原型，但检索精度有限

**Advanced RAG（高级架构）**：
- 增加查询改写、混合检索、Reranking、上下文压缩
- 提升检索精度和生成质量

**Modular RAG（模块化架构）**：
- 多阶段检索、Self-RAG、主动检索、引用生成
- 生产级架构，精度最高

---

## 12.2 Document Loaders

### 12.2.1 TextLoader - 纯文本文件

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/policy.txt", encoding="utf-8")
documents = loader.load()

print(documents[0].page_content)
print(documents[0].metadata)  # {'source': 'data/policy.txt'}
```

### 12.2.2 PyPDFLoader - PDF 文档

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/report.pdf")
pages = loader.load()

for i, page in enumerate(pages):
    print(f"Page {i}: {page.metadata}")
```

**PDF Loader 对比**：

| Loader | 优势 | 局限 |
|--------|------|------|
| PyPDFLoader | 轻量快速 | 不支持复杂布局 |
| PyMuPDFLoader | 速度快，保留格式 | 依赖 fitz 库 |
| PDFPlumberLoader | 表格提取强 | 速度较慢 |
| UnstructuredPDFLoader | 智能布局识别 | 需要额外依赖 |

### 12.2.3 CSVLoader - 结构化数据

```python
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(
    file_path="data/products.csv",
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
        "fieldnames": ["id", "name", "price", "description"]
    }
)
documents = loader.load()
```

### 12.2.4 WebBaseLoader - 网页内容

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://docs.langchain.com/")
documents = loader.load()

# 使用 BeautifulSoup 提取特定内容
loader = WebBaseLoader(
    web_paths=["https://example.com"],
    bs_kwargs={"parse_only": bs4.SoupStrainer(class_="main-content")}
)
```

### 12.2.5 DirectoryLoader - 批量加载

```python
from langchain_community.document_loaders import DirectoryLoader

# 加载目录下所有 .txt 文件
loader = DirectoryLoader(
    "data/",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
documents = loader.load()

print(f"Loaded {len(documents)} documents")
```

---

## 12.3 Document 数据结构

每个 Document 包含两个核心属性：

```python
from langchain.schema import Document

doc = Document(
    page_content="LangChain is a framework...",
    metadata={
        "source": "docs/intro.md",
        "title": "Introduction",
        "author": "LangChain Team",
        "date": "2024-01-15",
        "category": "tutorial"
    }
)
```

**推荐 metadata 字段**：

| 字段 | 用途 | 示例 |
|------|------|------|
| source | 文档来源 | "data/report.pdf" |
| title | 文档标题 | "Q4 Financial Report" |
| author | 作者 | "John Doe" |
| date | 日期 | "2024-01-15" |
| page | 页码 | 12 |
| category | 分类 | "finance" |

---

## 12.4 Text Splitters

### 12.4.1 为什么需要分割？

- LLM 有 token 上下文限制（GPT-4: 128K, GPT-3.5: 16K）
- 大文档直接输入会超出限制
- 分割成小块便于精准检索

<div data-component="TextSplittingVisualizer"></div>

### 12.4.2 RecursiveCharacterTextSplitter（推荐）

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 每个 chunk 的目标大小
    chunk_overlap=50,      # chunk 之间的重叠
    length_function=len,   # 长度计算函数
    separators=["\n\n", "\n", ". ", " ", ""]  # 分隔符优先级
)

texts = text_splitter.split_text(long_text)
documents = text_splitter.split_documents(documents)
```

**工作原理**：按优先级尝试分隔符
1. 首先尝试 `\n\n`（段落）
2. 如果块还太大，尝试 `\n`（行）
3. 再尝试 `. `（句子）
4. 最后尝试空格和字符

### 12.4.3 TokenTextSplitter - 精确 Token 控制

```python
from langchain.text_splitter import TokenTextSplitter

text_splitter = TokenTextSplitter(
    chunk_size=100,      # 100 tokens per chunk
    chunk_overlap=10,
    encoding_name="cl100k_base"  # GPT-4 tokenizer
)

chunks = text_splitter.split_text(text)
```

### 12.4.4 MarkdownHeaderTextSplitter - 保留层级

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
docs = splitter.split_text(markdown_text)

# metadata 会包含层级信息
# {'Header 1': 'Introduction', 'Header 2': 'Getting Started'}
```

### 12.4.5 参数调优建议

**chunk_size 选择**：

| 场景 | 推荐大小 | 理由 |
|------|---------|------|
| 问答系统 | 400-600 | 平衡上下文与精度 |
| 摘要生成 | 1000-2000 | 需要更多上下文 |
| 代码检索 | 200-400 | 代码块通常较短 |
| 长文档分析 | 800-1200 | 保留段落完整性 |

**chunk_overlap 选择**：
- 一般设置为 chunk_size 的 10-20%
- 确保上下文不会在分割处断裂
- 过大会浪费 token，过小会丢失上下文

---

## 12.5 Embeddings

### 12.5.1 OpenAI Embeddings

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # 或 text-embedding-3-large
    dimensions=1536  # 可选：降维以节省成本
)

# 嵌入单个文本
vector = embeddings.embed_query("What is LangChain?")
print(len(vector))  # 1536

# 批量嵌入
vectors = embeddings.embed_documents([
    "Document 1",
    "Document 2",
    "Document 3"
])
```

**OpenAI Embeddings 对比**：

| 模型 | 维度 | 性能 | 成本 ($/1M tokens) |
|------|------|------|-------------------|
| text-embedding-3-small | 1536 | 高 | $0.02 |
| text-embedding-3-large | 3072 | 最高 | $0.13 |
| text-embedding-ada-002 | 1536 | 中 | $0.10 |

### 12.5.2 HuggingFace Embeddings

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectors = embeddings.embed_documents(texts)
```

**推荐模型**：

| 模型 | 维度 | 语言 | 适用场景 |
|------|------|------|---------|
| all-MiniLM-L6-v2 | 384 | 英文 | 通用检索，速度快 |
| bge-large-en-v1.5 | 1024 | 英文 | 高精度检索 |
| bge-large-zh-v1.5 | 1024 | 中文 | 中文最佳 |
| multilingual-e5-large | 1024 | 多语言 | 跨语言检索 |

<div data-component="EmbeddingSpaceVisualization"></div>

### 12.5.3 批量嵌入优化

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# ❌ 低效：逐个嵌入
vectors = []
for doc in documents:
    vector = embeddings.embed_query(doc.page_content)
    vectors.append(vector)

# ✅ 高效：批量嵌入
texts = [doc.page_content for doc in documents]
vectors = embeddings.embed_documents(texts)
```

### 12.5.4 嵌入缓存

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

# 本地文件缓存
store = LocalFileStore("./cache/embeddings")

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=OpenAIEmbeddings(),
    document_embedding_cache=store,
    namespace="openai-embeddings"
)

# 第一次调用：计算并缓存
vectors1 = cached_embeddings.embed_documents(["Hello world"])

# 第二次调用：直接从缓存读取（秒级）
vectors2 = cached_embeddings.embed_documents(["Hello world"])
```

---

## 12.6 完整 RAG Pipeline 示例

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 1. 加载文档
loader = DirectoryLoader(
    "data/",
    glob="**/*.txt",
    loader_cls=TextLoader
)
documents = loader.load()
print(f"Loaded {len(documents)} documents")

# 2. 文本分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

# 3. 生成 Embeddings
embeddings = OpenAIEmbeddings()

# 4. 创建向量数据库
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. 创建 Retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 6. 创建 RAG Chain
llm = ChatOpenAI(model="gpt-4", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 7. 查询
result = qa_chain({"query": "What is LangChain?"})
print(result["result"])

# 8. 查看来源文档
for doc in result["source_documents"]:
    print(f"Source: {doc.metadata['source']}")
    print(f"Content: {doc.page_content[:200]}...\n")
```

---

## 12.7 最佳实践与优化

### 12.7.1 文档预处理

```python
def preprocess_documents(documents):
    """清理和规范化文档"""
    processed = []
    for doc in documents:
        # 移除多余空白
        content = " ".join(doc.page_content.split())
        
        # 移除特殊字符（可选）
        content = content.replace("\x00", "")
        
        # 标准化元数据
        metadata = {
            "source": doc.metadata.get("source", "unknown"),
            "title": doc.metadata.get("title", ""),
            "processed_at": datetime.now().isoformat()
        }
        
        processed.append(Document(
            page_content=content,
            metadata=metadata
        ))
    
    return processed
```

### 12.7.2 动态 Chunk Size

```python
def optimize_chunk_size(text, target_tokens=500):
    """根据文本特征动态调整 chunk size"""
    avg_word_length = sum(len(word) for word in text.split()) / len(text.split())
    estimated_chars_per_token = avg_word_length + 1
    
    chunk_size = int(target_tokens * estimated_chars_per_token)
    return chunk_size
```

### 12.7.3 成本优化

```python
# 1. 使用更便宜的 embedding 模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 2. 降维以减少存储和计算
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024  # 从 3072 降到 1024
)

# 3. 批量处理以利用 API 速率限制
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(chunk_size=1000)  # 每批 1000 个
```

---

## 本章小结

本章系统学习了 RAG 的基础架构：

1. **RAG 原理**：理解为什么需要 RAG，以及与 Fine-tuning 的对比
2. **Document Loaders**：掌握多种文档加载器的使用（Text、PDF、CSV、Web）
3. **Text Splitters**：学会选择合适的分割策略和参数调优
4. **Embeddings**：了解主流 Embedding 模型的特点和优化方法
5. **完整 Pipeline**：能够构建端到端的 RAG 系统

**下一章预告**：Chapter 13 将深入学习向量存储（VectorStore）与检索优化，包括 Chroma、Pinecone、FAISS 等主流向量数据库的使用、混合检索策略、索引管理等生产级技术。

---

## 扩展阅读

- [LangChain Document Loaders 官方文档](https://python.langchain.com/docs/modules/data_connection/document_loaders/)
- [Text Splitters 完整指南](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Embeddings 提供商对比](https://python.langchain.com/docs/modules/data_connection/text_embedding/)
- [RAG 架构设计模式](https://blog.langchain.dev/rag-from-scratch/)
- [Unstructured 文档解析库](https://github.com/Unstructured-IO/unstructured)

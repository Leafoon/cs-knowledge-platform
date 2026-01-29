# Chapter 11: 记忆优化与最佳实践

## 本章导览

在前两章中，我们学习了各种记忆类型（Chapter 9）和持久化方案（Chapter 10）。然而，在生产环境中，记忆系统的性能优化和可靠性工程同样重要。一个设计不当的记忆系统可能导致：

- **成本失控**：Token 消耗快速增长，API 费用暴涨
- **性能瓶颈**：检索延迟过高，用户体验下降
- **内存泄漏**：未清理的会话占用大量存储空间
- **隐私风险**：敏感信息未脱敏，违反合规要求

本章将深入讲解记忆系统的优化技术，包括 Token 管理、检索加速、多模态记忆、并发控制、隐私合规等生产级最佳实践。

---

## 11.1 Token 管理策略

### 11.1.1 Token 计数与限制

精确的 Token 计数是成本控制的基础：

```python
import tiktoken

class TokenAwareMemory:
    """Token 感知的记忆系统"""
    
    def __init__(self, model_name: str = "gpt-4", max_tokens: int = 4000):
        self.encoding = tiktoken.encoding_for_model(model_name)
        self.max_tokens = max_tokens
        self.messages = []
    
    def count_tokens(self, text: str) -> int:
        """计算文本的 Token 数量"""
        return len(self.encoding.encode(text))
    
    def count_message_tokens(self, messages: List[BaseMessage]) -> int:
        """计算消息列表的总 Token 数"""
        total = 0
        for msg in messages:
            # 每条消息的固定开销（角色标记等）
            total += 4
            total += self.count_tokens(msg.content)
        # 对话的固定开销
        total += 2
        return total
    
    def add_message(self, message: BaseMessage):
        """添加消息并检查 Token 限制"""
        self.messages.append(message)
        
        # 检查是否超出限制
        total_tokens = self.count_message_tokens(self.messages)
        
        if total_tokens > self.max_tokens:
            # 自动截断最早的消息
            while total_tokens > self.max_tokens and len(self.messages) > 1:
                self.messages.pop(0)  # 移除最早的消息
                total_tokens = self.count_message_tokens(self.messages)
        
        return total_tokens
    
    def get_messages(self) -> List[BaseMessage]:
        """获取符合 Token 限制的消息"""
        return self.messages

# 使用示例
memory = TokenAwareMemory(model_name="gpt-4", max_tokens=4000)

# 添加消息（自动管理 Token）
memory.add_message(HumanMessage(content="很长的用户输入..."))
memory.add_message(AIMessage(content="详细的 AI 回复..."))

print(f"当前 Token 数: {memory.count_message_tokens(memory.messages)}")
```

**实时 Token 监控**：

```python
from typing import Dict, List
from datetime import datetime, timedelta

class TokenUsageMonitor:
    """Token 使用监控器"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def record_usage(
        self,
        session_id: str,
        tokens_used: int,
        cost: float
    ):
        """记录 Token 使用情况"""
        timestamp = datetime.now()
        key_today = f"token_usage:{timestamp.strftime('%Y-%m-%d')}"
        
        # 记录每个会话的使用量
        self.redis.hincrby(key_today, session_id, tokens_used)
        
        # 记录成本
        cost_key = f"token_cost:{timestamp.strftime('%Y-%m-%d')}"
        self.redis.hincrbyfloat(cost_key, session_id, cost)
        
        # 设置7天过期
        self.redis.expire(key_today, 7 * 24 * 3600)
        self.redis.expire(cost_key, 7 * 24 * 3600)
    
    def get_daily_usage(self, date: str = None) -> Dict[str, int]:
        """获取每日使用统计"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        key = f"token_usage:{date}"
        return {
            session_id.decode(): int(tokens)
            for session_id, tokens in self.redis.hgetall(key).items()
        }
    
    def get_top_users(self, date: str = None, limit: int = 10) -> List[tuple]:
        """获取 Token 使用量最高的用户"""
        usage = self.get_daily_usage(date)
        return sorted(usage.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def check_quota(self, session_id: str, quota: int) -> bool:
        """检查会话是否超出配额"""
        today = datetime.now().strftime('%Y-%m-%d')
        key = f"token_usage:{today}"
        used = int(self.redis.hget(key, session_id) or 0)
        return used < quota

# 使用示例
from redis import Redis

redis_client = Redis.from_url("redis://localhost:6379/0")
monitor = TokenUsageMonitor(redis_client)

# 记录使用
monitor.record_usage(
    session_id="user_alice",
    tokens_used=1500,
    cost=0.03  # $0.03
)

# 检查配额
if not monitor.check_quota("user_alice", quota=10000):
    print("警告：用户已超出每日配额！")

# 查看 Top 用户
top_users = monitor.get_top_users(limit=5)
for session_id, tokens in top_users:
    print(f"{session_id}: {tokens} tokens")
```

### 11.1.2 自动截断策略

实现智能的对话截断，保留最重要的信息：

```python
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage

class SmartTruncationMemory:
    """智能截断记忆"""
    
    def __init__(
        self,
        max_tokens: int = 4000,
        preserve_system: bool = True,
        preserve_recent: int = 2
    ):
        self.max_tokens = max_tokens
        self.preserve_system = preserve_system
        self.preserve_recent = preserve_recent
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        self.messages = []
    
    def _count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def truncate_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """智能截断消息列表"""
        # 1. 分离系统消息
        system_msgs = [msg for msg in messages if isinstance(msg, SystemMessage)]
        other_msgs = [msg for msg in messages if not isinstance(msg, SystemMessage)]
        
        # 2. 保留最近的N条消息
        recent_msgs = other_msgs[-self.preserve_recent * 2:] if len(other_msgs) > self.preserve_recent * 2 else other_msgs
        older_msgs = other_msgs[:-self.preserve_recent * 2] if len(other_msgs) > self.preserve_recent * 2 else []
        
        # 3. 计算当前 Token
        result = system_msgs + recent_msgs
        current_tokens = sum(self._count_tokens(msg.content) for msg in result)
        
        # 4. 从旧消息中选择性添加
        for msg in reversed(older_msgs):
            msg_tokens = self._count_tokens(msg.content)
            if current_tokens + msg_tokens <= self.max_tokens:
                result.insert(len(system_msgs), msg)
                current_tokens += msg_tokens
            else:
                break
        
        return result
    
    def add_messages(self, messages: List[BaseMessage]):
        """添加消息并自动截断"""
        self.messages.extend(messages)
        self.messages = self.truncate_messages(self.messages)
    
    def get_messages(self) -> List[BaseMessage]:
        return self.messages

# 使用示例
memory = SmartTruncationMemory(
    max_tokens=2000,
    preserve_system=True,
    preserve_recent=2  # 保留最近2轮对话
)

# 添加系统提示（总是保留）
memory.add_messages([
    SystemMessage(content="你是一个专业的客服助手")
])

# 添加多轮对话
for i in range(10):
    memory.add_messages([
        HumanMessage(content=f"问题 {i+1}"),
        AIMessage(content=f"回答 {i+1}，这是一个很长的回答..." * 20)
    ])

# 查看截断后的消息（保留系统消息 + 最近2轮 + 部分旧消息）
print(f"总消息数: {len(memory.get_messages())}")
```

### 11.1.3 上下文压缩技术

使用 LLM 压缩对话历史：

```python
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class AdaptiveCompressionMemory:
    """自适应压缩记忆"""
    
    def __init__(
        self,
        llm: ChatOpenAI,
        compression_threshold: int = 3000,
        summary_max_tokens: int = 500
    ):
        self.llm = llm
        self.compression_threshold = compression_threshold
        self.summary_max_tokens = summary_max_tokens
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        self.messages = []
        self.summary = None
    
    def _count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))
    
    def _compress_messages(self, messages: List[BaseMessage]) -> str:
        """将消息列表压缩为摘要"""
        # 构建压缩提示
        conversation = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in messages
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "将以下对话压缩为简洁的摘要，保留关键信息。摘要不超过{max_tokens} tokens。"),
            ("human", "{conversation}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "conversation": conversation,
            "max_tokens": self.summary_max_tokens
        })
        
        return response.content
    
    def add_message(self, message: BaseMessage):
        """添加消息，必要时自动压缩"""
        self.messages.append(message)
        
        # 计算当前 Token
        total_tokens = sum(self._count_tokens(msg.content) for msg in self.messages)
        
        if total_tokens > self.compression_threshold:
            # 触发压缩
            messages_to_compress = self.messages[:-4]  # 压缩除最近2轮外的所有消息
            recent_messages = self.messages[-4:]
            
            # 生成摘要
            new_summary = self._compress_messages(messages_to_compress)
            
            # 更新状态
            self.summary = new_summary
            self.messages = recent_messages
            
            print(f"[压缩] 将 {len(messages_to_compress)} 条消息压缩为 {self._count_tokens(new_summary)} tokens")
    
    def get_context(self) -> str:
        """获取完整上下文（摘要 + 最近消息）"""
        context_parts = []
        
        if self.summary:
            context_parts.append(f"[历史摘要]\n{self.summary}\n")
        
        if self.messages:
            context_parts.append("[最近对话]")
            for msg in self.messages:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                context_parts.append(f"{role}: {msg.content}")
        
        return "\n".join(context_parts)

# 使用示例
llm = ChatOpenAI(model="gpt-4", temperature=0)
memory = AdaptiveCompressionMemory(
    llm=llm,
    compression_threshold=3000,
    summary_max_tokens=500
)

# 模拟长对话
for i in range(10):
    memory.add_message(HumanMessage(content=f"这是第 {i+1} 条消息，包含大量细节..." * 50))
    memory.add_message(AIMessage(content=f"这是对第 {i+1} 条消息的详细回复..." * 50))

# 获取压缩后的上下文
print(memory.get_context())
```

<div data-component="TokenManagementDashboard"></div>

---

## 11.2 记忆检索优化

### 11.2.1 向量索引加速

使用 FAISS 构建高性能向量索引：

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

class OptimizedVectorMemory:
    """优化的向量记忆"""
    
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = None
        self.documents = []
    
    def add_conversation_turn(
        self,
        user_input: str,
        ai_response: str,
        metadata: dict = None
    ):
        """添加一轮对话"""
        # 合并用户输入和 AI 响应作为一个文档
        content = f"User: {user_input}\nAssistant: {ai_response}"
        
        doc = Document(
            page_content=content,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "ai_response": ai_response,
                **(metadata or {})
            }
        )
        
        self.documents.append(doc)
        
        # 重建或更新索引
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(
                [doc],
                self.embeddings
            )
        else:
            # 增量添加
            self.vector_store.add_documents([doc])
    
    def search_relevant_history(
        self,
        query: str,
        k: int = 3,
        score_threshold: float = 0.7
    ) -> List[Document]:
        """检索相关历史"""
        if not self.vector_store:
            return []
        
        # 使用相似度阈值过滤
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        # 过滤低分结果
        filtered = [
            doc for doc, score in results
            if score >= score_threshold
        ]
        
        return filtered
    
    def save_index(self, path: str):
        """保存向量索引到磁盘"""
        if self.vector_store:
            self.vector_store.save_local(path)
    
    def load_index(self, path: str):
        """从磁盘加载向量索引"""
        self.vector_store = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

# 使用示例
memory = OptimizedVectorMemory()

# 添加历史对话
memory.add_conversation_turn(
    user_input="我喜欢 Python 编程",
    ai_response="太好了！Python 是一门非常强大的语言。"
)

memory.add_conversation_turn(
    user_input="我在学习机器学习",
    ai_response="机器学习非常有趣，推荐你学习 scikit-learn。"
)

memory.add_conversation_turn(
    user_input="今天天气真好",
    ai_response="是的，适合出去散步。"
)

# 检索相关历史
relevant = memory.search_relevant_history(
    query="推荐一个 Python 的机器学习库",
    k=2
)

for doc in relevant:
    print(f"相关对话: {doc.page_content}\n")

# 保存索引（避免重复计算 embedding）
memory.save_index("./memory_index")
```

### 11.2.2 缓存热点记忆

使用 LRU 缓存加速常见查询：

```python
from functools import lru_cache
from typing import Tuple
import hashlib

class CachedMemoryRetriever:
    """带缓存的记忆检索器"""
    
    def __init__(self, vector_memory: OptimizedVectorMemory, cache_size: int = 128):
        self.vector_memory = vector_memory
        self.cache_size = cache_size
        self._init_cache()
    
    def _init_cache(self):
        """初始化 LRU 缓存"""
        self._retrieve = lru_cache(maxsize=self.cache_size)(self._retrieve_uncached)
    
    def _hash_query(self, query: str, k: int) -> str:
        """生成查询的哈希键"""
        key = f"{query}:{k}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _retrieve_uncached(self, query_hash: str, query: str, k: int) -> Tuple[Document, ...]:
        """实际的检索逻辑（无缓存）"""
        results = self.vector_memory.search_relevant_history(query, k=k)
        # 返回 tuple 以支持哈希（缓存需要）
        return tuple(results)
    
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """检索记忆（带缓存）"""
        query_hash = self._hash_query(query, k)
        results_tuple = self._retrieve(query_hash, query, k)
        return list(results_tuple)
    
    def clear_cache(self):
        """清空缓存"""
        self._retrieve.cache_clear()
    
    def cache_info(self):
        """获取缓存统计"""
        return self._retrieve.cache_info()

# 使用示例
retriever = CachedMemoryRetriever(memory, cache_size=128)

# 第一次检索（未命中缓存）
results1 = retriever.retrieve("Python 机器学习")

# 第二次相同查询（命中缓存，极快）
results2 = retriever.retrieve("Python 机器学习")

# 查看缓存统计
print(retriever.cache_info())
# CacheInfo(hits=1, misses=1, maxsize=128, currsize=1)
```

### 11.2.3 懒加载与分页

实现大规模记忆的分页加载：

```python
from typing import Iterator

class PaginatedMemoryLoader:
    """分页记忆加载器"""
    
    def __init__(self, history: RedisChatMessageHistory, page_size: int = 10):
        self.history = history
        self.page_size = page_size
    
    def load_page(self, page: int = 0) -> List[BaseMessage]:
        """加载指定页的消息"""
        start = page * self.page_size
        end = start + self.page_size
        
        all_messages = self.history.messages
        return all_messages[start:end]
    
    def iter_pages(self) -> Iterator[List[BaseMessage]]:
        """迭代所有页"""
        all_messages = self.history.messages
        total_pages = (len(all_messages) + self.page_size - 1) // self.page_size
        
        for page in range(total_pages):
            yield self.load_page(page)
    
    def search_in_pages(
        self,
        keyword: str,
        max_pages: int = None
    ) -> List[BaseMessage]:
        """在分页中搜索关键词"""
        results = []
        
        for page_idx, page_messages in enumerate(self.iter_pages()):
            if max_pages and page_idx >= max_pages:
                break
            
            for msg in page_messages:
                if keyword.lower() in msg.content.lower():
                    results.append(msg)
        
        return results

# 使用示例
from langchain_community.chat_message_histories import RedisChatMessageHistory

history = RedisChatMessageHistory(
    session_id="user_long_session",
    url="redis://localhost:6379/0"
)

# 分页加载器
loader = PaginatedMemoryLoader(history, page_size=20)

# 只加载第一页（避免一次性加载所有数据）
first_page = loader.load_page(0)
print(f"第一页包含 {len(first_page)} 条消息")

# 搜索关键词（分页搜索，节省内存）
matching_messages = loader.search_in_pages("Python", max_pages=5)
```

<div data-component="MemoryRetrievalPerformance"></div>

---

## 11.3 多模态记忆

### 11.3.1 图像记忆存储

存储和检索对话中的图像：

```python
import base64
from io import BytesIO
from PIL import Image

class MultimodalMemory:
    """多模态记忆系统"""
    
    def __init__(self, redis_client, s3_client=None):
        self.redis = redis_client
        self.s3 = s3_client  # 可选：大图片存储到 S3
    
    def add_image_message(
        self,
        session_id: str,
        image_path: str,
        caption: str = None,
        metadata: dict = None
    ):
        """添加图像消息"""
        # 读取图像
        with Image.open(image_path) as img:
            # 生成缩略图（用于快速预览）
            img.thumbnail((200, 200))
            buffer = BytesIO()
            img.save(buffer, format='JPEG')
            thumbnail_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        # 存储消息
        message = {
            "type": "image",
            "caption": caption or "",
            "thumbnail": thumbnail_b64,
            "original_path": image_path,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # 添加到会话历史
        key = f"messages:{session_id}"
        self.redis.rpush(key, json.dumps(message))
    
    def get_image_messages(
        self,
        session_id: str,
        include_thumbnails: bool = True
    ) -> List[dict]:
        """获取所有图像消息"""
        key = f"messages:{session_id}"
        messages = self.redis.lrange(key, 0, -1)
        
        results = []
        for msg_str in messages:
            msg = json.loads(msg_str)
            if msg["type"] == "image":
                if not include_thumbnails:
                    msg.pop("thumbnail", None)
                results.append(msg)
        
        return results
    
    def search_images_by_caption(
        self,
        session_id: str,
        query: str
    ) -> List[dict]:
        """根据描述搜索图像"""
        image_messages = self.get_image_messages(session_id)
        
        # 简单的关键词匹配（生产环境可用向量搜索）
        return [
            msg for msg in image_messages
            if query.lower() in msg.get("caption", "").lower()
        ]

# 使用示例
from redis import Redis

redis_client = Redis.from_url("redis://localhost:6379/0", decode_responses=True)
multimodal_memory = MultimodalMemory(redis_client)

# 添加图像消息
multimodal_memory.add_image_message(
    session_id="user_alice",
    image_path="./screenshot.png",
    caption="代码截图：Python 函数定义",
    metadata={"category": "programming"}
)

# 检索图像
images = multimodal_memory.search_images_by_caption(
    session_id="user_alice",
    query="Python"
)

for img in images:
    print(f"图像描述: {img['caption']}")
    # 可以显示缩略图：base64 -> PIL Image
```

### 11.3.2 音频记忆管理

存储语音对话的转录和元数据：

```python
class AudioMemory:
    """音频记忆系统"""
    
    def __init__(self, redis_client, storage_backend="local"):
        self.redis = redis_client
        self.storage_backend = storage_backend
    
    def add_audio_message(
        self,
        session_id: str,
        audio_path: str,
        transcription: str,
        duration: float,
        language: str = "zh-CN"
    ):
        """添加音频消息"""
        message = {
            "type": "audio",
            "audio_path": audio_path,
            "transcription": transcription,
            "duration": duration,
            "language": language,
            "timestamp": datetime.now().isoformat()
        }
        
        key = f"messages:{session_id}"
        self.redis.rpush(key, json.dumps(message))
    
    def get_conversation_transcripts(
        self,
        session_id: str
    ) -> List[str]:
        """获取所有语音转录"""
        key = f"messages:{session_id}"
        messages = self.redis.lrange(key, 0, -1)
        
        transcripts = []
        for msg_str in messages:
            msg = json.loads(msg_str)
            if msg.get("type") == "audio":
                transcripts.append(msg["transcription"])
        
        return transcripts
    
    def get_total_duration(self, session_id: str) -> float:
        """获取会话的总音频时长"""
        key = f"messages:{session_id}"
        messages = self.redis.lrange(key, 0, -1)
        
        total = 0.0
        for msg_str in messages:
            msg = json.loads(msg_str)
            if msg.get("type") == "audio":
                total += msg.get("duration", 0.0)
        
        return total

# 使用示例
audio_memory = AudioMemory(redis_client)

# 添加音频消息
audio_memory.add_audio_message(
    session_id="user_bob",
    audio_path="./voice_message.mp3",
    transcription="你好，我想查询订单状态",
    duration=3.5,
    language="zh-CN"
)

# 获取所有转录
transcripts = audio_memory.get_conversation_transcripts("user_bob")
print("\n".join(transcripts))
```

### 11.3.3 多模态检索

统一检索文本、图像、音频：

```python
from typing import Literal

class UnifiedMultimodalRetriever:
    """统一多模态检索器"""
    
    def __init__(
        self,
        text_memory: OptimizedVectorMemory,
        image_memory: MultimodalMemory,
        audio_memory: AudioMemory
    ):
        self.text_memory = text_memory
        self.image_memory = image_memory
        self.audio_memory = audio_memory
    
    def search(
        self,
        session_id: str,
        query: str,
        modality: Literal["text", "image", "audio", "all"] = "all",
        k: int = 3
    ) -> Dict[str, List]:
        """跨模态检索"""
        results = {}
        
        if modality in ["text", "all"]:
            # 文本检索
            text_results = self.text_memory.search_relevant_history(query, k=k)
            results["text"] = [doc.page_content for doc in text_results]
        
        if modality in ["image", "all"]:
            # 图像检索
            image_results = self.image_memory.search_images_by_caption(session_id, query)
            results["images"] = image_results[:k]
        
        if modality in ["audio", "all"]:
            # 音频检索（基于转录）
            transcripts = self.audio_memory.get_conversation_transcripts(session_id)
            matching_transcripts = [
                t for t in transcripts
                if query.lower() in t.lower()
            ]
            results["audio"] = matching_transcripts[:k]
        
        return results

# 使用示例
unified_retriever = UnifiedMultimodalRetriever(
    text_memory=memory,
    image_memory=multimodal_memory,
    audio_memory=audio_memory
)

# 跨模态检索
results = unified_retriever.search(
    session_id="user_alice",
    query="Python",
    modality="all",
    k=3
)

print("文本结果:", results.get("text", []))
print("图像结果:", results.get("images", []))
print("音频结果:", results.get("audio", []))
```

---

## 11.4 记忆冲突与一致性

### 11.4.1 并发写入控制

使用 Redis 事务防止并发冲突：

```python
class ConcurrencySafeMemory:
    """并发安全的记忆系统"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def add_message_atomic(
        self,
        session_id: str,
        message: BaseMessage,
        max_messages: int = 100
    ):
        """原子性地添加消息"""
        key = f"messages:{session_id}"
        message_json = json.dumps({
            "type": message.__class__.__name__,
            "content": message.content,
            "timestamp": datetime.now().isoformat()
        })
        
        # 使用 Redis 事务
        with self.redis.pipeline() as pipe:
            try:
                # 监视键（乐观锁）
                pipe.watch(key)
                
                # 获取当前消息数
                current_count = pipe.llen(key)
                
                # 开始事务
                pipe.multi()
                
                # 添加消息
                pipe.rpush(key, message_json)
                
                # 如果超出限制，删除最早的消息
                if current_count >= max_messages:
                    pipe.ltrim(key, -max_messages, -1)
                
                # 执行事务
                pipe.execute()
                
                return True
            
            except Exception as e:
                print(f"事务失败: {e}")
                return False
    
    def update_message_metadata(
        self,
        session_id: str,
        message_index: int,
        new_metadata: dict
    ):
        """原子性地更新消息元数据"""
        key = f"messages:{session_id}"
        
        with self.redis.pipeline() as pipe:
            try:
                pipe.watch(key)
                
                # 获取消息
                message_json = pipe.lindex(key, message_index)
                if not message_json:
                    return False
                
                message = json.loads(message_json)
                message["metadata"] = new_metadata
                message["updated_at"] = datetime.now().isoformat()
                
                # 更新消息
                pipe.multi()
                pipe.lset(key, message_index, json.dumps(message))
                pipe.execute()
                
                return True
            
            except Exception as e:
                print(f"更新失败: {e}")
                return False

# 使用示例
safe_memory = ConcurrencySafeMemory(redis_client)

# 并发安全地添加消息
safe_memory.add_message_atomic(
    session_id="user_concurrent",
    message=HumanMessage(content="测试消息"),
    max_messages=100
)
```

### 11.4.2 版本冲突解决

实现类似 Git 的冲突解决机制：

```python
from typing import Optional

class VersionedMemory:
    """带版本控制的记忆系统"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def get_version(self, session_id: str) -> int:
        """获取当前版本号"""
        key = f"version:{session_id}"
        version = self.redis.get(key)
        return int(version) if version else 0
    
    def add_message_with_version(
        self,
        session_id: str,
        message: BaseMessage,
        expected_version: int
    ) -> Optional[int]:
        """添加消息并检查版本"""
        current_version = self.get_version(session_id)
        
        # 版本检查
        if current_version != expected_version:
            print(f"版本冲突: 期望 v{expected_version}, 实际 v{current_version}")
            return None
        
        # 添加消息
        key = f"messages:{session_id}"
        message_json = json.dumps({
            "content": message.content,
            "version": current_version + 1,
            "timestamp": datetime.now().isoformat()
        })
        
        with self.redis.pipeline() as pipe:
            pipe.rpush(key, message_json)
            pipe.incr(f"version:{session_id}")
            pipe.execute()
        
        return current_version + 1
    
    def resolve_conflict(
        self,
        session_id: str,
        local_messages: List[BaseMessage],
        strategy: Literal["latest", "merge", "manual"] = "latest"
    ):
        """解决版本冲突"""
        if strategy == "latest":
            # 策略1：使用服务器最新版本（放弃本地更改）
            return self.get_messages(session_id)
        
        elif strategy == "merge":
            # 策略2：合并本地和服务器更改
            server_messages = self.get_messages(session_id)
            
            # 简单的时间戳合并
            all_messages = local_messages + server_messages
            all_messages.sort(key=lambda m: m.additional_kwargs.get("timestamp", ""))
            
            # 去重
            seen = set()
            merged = []
            for msg in all_messages:
                key = (msg.content, msg.additional_kwargs.get("timestamp"))
                if key not in seen:
                    seen.add(key)
                    merged.append(msg)
            
            return merged
        
        else:
            # 策略3：手动解决
            raise ValueError("需要手动解决冲突")

# 使用示例
versioned_memory = VersionedMemory(redis_client)

# 获取当前版本
version = versioned_memory.get_version("user_alice")

# 添加消息（带版本检查）
new_version = versioned_memory.add_message_with_version(
    session_id="user_alice",
    message=HumanMessage(content="新消息"),
    expected_version=version
)

if new_version is None:
    # 发生冲突，解决冲突
    resolved = versioned_memory.resolve_conflict(
        session_id="user_alice",
        local_messages=[HumanMessage(content="本地消息")],
        strategy="merge"
    )
```

### 11.4.3 最终一致性保证

实现分布式系统的最终一致性：

```python
from typing import List
import time

class EventuallyConsistentMemory:
    """最终一致性记忆系统"""
    
    def __init__(self, redis_clients: List, replication_lag: float = 0.1):
        """
        Args:
            redis_clients: 多个 Redis 实例客户端
            replication_lag: 复制延迟（秒）
        """
        self.redis_clients = redis_clients
        self.replication_lag = replication_lag
    
    def add_message_replicated(
        self,
        session_id: str,
        message: BaseMessage,
        quorum: int = 2
    ) -> bool:
        """添加消息到多个副本（Quorum 写）"""
        key = f"messages:{session_id}"
        message_json = json.dumps({
            "content": message.content,
            "timestamp": datetime.now().isoformat()
        })
        
        success_count = 0
        
        for client in self.redis_clients:
            try:
                client.rpush(key, message_json)
                success_count += 1
            except Exception as e:
                print(f"写入副本失败: {e}")
        
        # 检查是否达到 Quorum
        return success_count >= quorum
    
    def read_with_consistency(
        self,
        session_id: str,
        consistency: Literal["strong", "eventual"] = "eventual"
    ) -> List[BaseMessage]:
        """读取消息（可选一致性级别）"""
        key = f"messages:{session_id}"
        
        if consistency == "strong":
            # 强一致性：从主节点读取
            messages = self.redis_clients[0].lrange(key, 0, -1)
        
        else:
            # 最终一致性：从任意副本读取
            for client in self.redis_clients:
                try:
                    messages = client.lrange(key, 0, -1)
                    if messages:
                        break
                except Exception:
                    continue
            else:
                messages = []
        
        # 反序列化
        return [
            json.loads(msg) for msg in messages
        ]
    
    def wait_for_replication(self, timeout: float = 1.0):
        """等待复制完成"""
        time.sleep(min(self.replication_lag, timeout))

# 使用示例（假设有3个 Redis 副本）
redis_replicas = [
    Redis.from_url("redis://redis1:6379/0"),
    Redis.from_url("redis://redis2:6379/0"),
    Redis.from_url("redis://redis3:6379/0")
]

ec_memory = EventuallyConsistentMemory(
    redis_clients=redis_replicas,
    replication_lag=0.1
)

# Quorum 写（至少2个副本成功）
success = ec_memory.add_message_replicated(
    session_id="user_distributed",
    message=HumanMessage(content="分布式消息"),
    quorum=2
)

# 等待复制
ec_memory.wait_for_replication()

# 最终一致性读
messages = ec_memory.read_with_consistency(
    session_id="user_distributed",
    consistency="eventual"
)
```

---

## 11.5 隐私与合规

### 11.5.1 敏感信息脱敏

自动检测和脱敏敏感数据：

```python
import re
from typing import Pattern, Dict

class PrivacyProtectedMemory:
    """隐私保护的记忆系统"""
    
    def __init__(self):
        # 定义敏感信息的正则表达式
        self.patterns: Dict[str, Pattern] = {
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "phone": re.compile(r'\b1[3-9]\d{9}\b'),  # 中国手机号
            "id_card": re.compile(r'\b\d{17}[\dXx]\b'),  # 中国身份证号
            "credit_card": re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
            "ip_address": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        }
    
    def redact_pii(self, text: str, keep_partial: bool = True) -> tuple[str, Dict[str, List[str]]]:
        """脱敏个人可识别信息 (PII)"""
        redacted_text = text
        detected_pii = {}
        
        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                detected_pii[pii_type] = matches
                
                for match in matches:
                    if keep_partial:
                        # 部分脱敏（保留前后各2位）
                        if len(match) > 6:
                            redacted = f"{match[:2]}***{match[-2:]}"
                        else:
                            redacted = "***"
                    else:
                        # 完全脱敏
                        redacted = f"[{pii_type.upper()}_REDACTED]"
                    
                    redacted_text = redacted_text.replace(match, redacted)
        
        return redacted_text, detected_pii
    
    def add_message_with_redaction(
        self,
        session_id: str,
        message: BaseMessage,
        log_pii: bool = True
    ) -> tuple[BaseMessage, Dict[str, List[str]]]:
        """添加消息并自动脱敏"""
        redacted_content, detected_pii = self.redact_pii(message.content)
        
        # 创建脱敏后的消息
        redacted_message = message.__class__(
            content=redacted_content,
            additional_kwargs={
                **message.additional_kwargs,
                "pii_detected": bool(detected_pii),
                "original_hash": hashlib.sha256(message.content.encode()).hexdigest()
            }
        )
        
        # 记录检测到的 PII（用于审计）
        if log_pii and detected_pii:
            self._log_pii_detection(session_id, detected_pii)
        
        return redacted_message, detected_pii
    
    def _log_pii_detection(self, session_id: str, detected_pii: Dict):
        """记录 PII 检测日志"""
        log_entry = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "pii_types": list(detected_pii.keys()),
            "count": sum(len(v) for v in detected_pii.values())
        }
        # 存储到审计日志
        print(f"[PII 检测] {log_entry}")

# 使用示例
privacy_memory = PrivacyProtectedMemory()

# 添加包含敏感信息的消息
original_message = HumanMessage(
    content="我的邮箱是 alice@example.com，手机号是 13800138000"
)

redacted_message, detected_pii = privacy_memory.add_message_with_redaction(
    session_id="user_alice",
    message=original_message
)

print(f"原始内容: {original_message.content}")
print(f"脱敏内容: {redacted_message.content}")
print(f"检测到的 PII: {detected_pii}")
# 输出:
# 原始内容: 我的邮箱是 alice@example.com，手机号是 13800138000
# 脱敏内容: 我的邮箱是 al***om，手机号是 13***00
# 检测到的 PII: {'email': ['alice@example.com'], 'phone': ['13800138000']}
```

### 11.5.2 数据加密存储

使用加密保护敏感对话：

```python
from cryptography.fernet import Fernet
import base64

class EncryptedMemory:
    """加密记忆系统"""
    
    def __init__(self, encryption_key: bytes = None):
        """
        Args:
            encryption_key: 32字节的加密密钥（可用 Fernet.generate_key() 生成）
        """
        if encryption_key is None:
            encryption_key = Fernet.generate_key()
        
        self.cipher = Fernet(encryption_key)
        self.messages = []
    
    def encrypt_message(self, message: BaseMessage) -> str:
        """加密消息"""
        message_json = json.dumps({
            "type": message.__class__.__name__,
            "content": message.content,
            "additional_kwargs": message.additional_kwargs
        })
        
        encrypted = self.cipher.encrypt(message_json.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt_message(self, encrypted_data: str) -> BaseMessage:
        """解密消息"""
        encrypted = base64.b64decode(encrypted_data.encode())
        decrypted = self.cipher.decrypt(encrypted)
        
        message_data = json.loads(decrypted.decode())
        
        # 重建消息对象
        from langchain.schema import HumanMessage, AIMessage, SystemMessage
        message_class = {
            "HumanMessage": HumanMessage,
            "AIMessage": AIMessage,
            "SystemMessage": SystemMessage
        }[message_data["type"]]
        
        return message_class(
            content=message_data["content"],
            additional_kwargs=message_data["additional_kwargs"]
        )
    
    def add_encrypted_message(self, message: BaseMessage):
        """添加加密消息"""
        encrypted = self.encrypt_message(message)
        self.messages.append(encrypted)
    
    def get_decrypted_messages(self) -> List[BaseMessage]:
        """获取解密后的消息"""
        return [
            self.decrypt_message(encrypted)
            for encrypted in self.messages
        ]

# 使用示例
# 生成密钥（应安全存储，不要硬编码）
encryption_key = Fernet.generate_key()
print(f"加密密钥: {encryption_key.decode()}")

# 创建加密记忆
encrypted_memory = EncryptedMemory(encryption_key)

# 添加敏感消息
encrypted_memory.add_encrypted_message(
    HumanMessage(content="我的密码是 MySecret123!")
)

# 查看加密后的数据
print(f"加密数据: {encrypted_memory.messages[0]}")

# 解密读取
decrypted = encrypted_memory.get_decrypted_messages()
print(f"解密内容: {decrypted[0].content}")
```

### 11.5.3 GDPR 合规（删除权）

实现用户数据的完全删除：

```python
class GDPRCompliantMemory:
    """GDPR 合规的记忆系统"""
    
    def __init__(self, redis_client, postgres_engine):
        self.redis = redis_client
        self.pg_engine = postgres_engine
    
    def delete_user_data(self, user_id: str, reason: str = "user_request"):
        """完全删除用户数据（GDPR 删除权）"""
        from sqlalchemy import text
        
        deletion_log = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "deleted_items": {}
        }
        
        # 1. 删除 Redis 中的所有会话
        pattern = f"*user:{user_id}:*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)
            deletion_log["deleted_items"]["redis_keys"] = len(keys)
        
        # 2. 删除 PostgreSQL 中的数据
        with self.pg_engine.connect() as conn:
            # 删除消息历史
            result = conn.execute(
                text("DELETE FROM chat_histories WHERE session_id LIKE :pattern"),
                {"pattern": f"%user:{user_id}:%"}
            )
            deletion_log["deleted_items"]["postgres_messages"] = result.rowcount
            
            # 删除归档数据
            result = conn.execute(
                text("DELETE FROM archived_conversations WHERE session_id LIKE :pattern"),
                {"pattern": f"%user:{user_id}:%"}
            )
            deletion_log["deleted_items"]["postgres_archived"] = result.rowcount
            
            conn.commit()
        
        # 3. 记录删除日志（合规审计）
        self._log_deletion(deletion_log)
        
        return deletion_log
    
    def _log_deletion(self, deletion_log: dict):
        """记录删除操作（不可变审计日志）"""
        with self.pg_engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO data_deletion_log (user_id, timestamp, reason, details)
                VALUES (:user_id, :timestamp, :reason, :details)
            """), {
                "user_id": deletion_log["user_id"],
                "timestamp": deletion_log["timestamp"],
                "reason": deletion_log["reason"],
                "details": json.dumps(deletion_log["deleted_items"])
            })
            conn.commit()
    
    def export_user_data(self, user_id: str) -> dict:
        """导出用户所有数据（GDPR 数据可携带权）"""
        export_data = {
            "user_id": user_id,
            "exported_at": datetime.now().isoformat(),
            "data": {}
        }
        
        # 从 Redis 导出
        pattern = f"*user:{user_id}:*"
        keys = self.redis.keys(pattern)
        
        redis_data = {}
        for key in keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            value = self.redis.get(key_str)
            if value:
                redis_data[key_str] = value.decode() if isinstance(value, bytes) else value
        
        export_data["data"]["redis"] = redis_data
        
        # 从 PostgreSQL 导出
        with self.pg_engine.connect() as conn:
            result = conn.execute(text("""
                SELECT * FROM chat_histories
                WHERE session_id LIKE :pattern
            """), {"pattern": f"%user:{user_id}:%"})
            
            export_data["data"]["postgres"] = [
                dict(row) for row in result
            ]
        
        return export_data

# 使用示例
from sqlalchemy import create_engine

redis_client = Redis.from_url("redis://localhost:6379/0")
pg_engine = create_engine("postgresql://user:pass@localhost/chatdb")

gdpr_memory = GDPRCompliantMemory(redis_client, pg_engine)

# 用户请求删除数据
deletion_log = gdpr_memory.delete_user_data(
    user_id="alice",
    reason="user_request"
)
print(f"删除完成: {deletion_log}")

# 用户请求导出数据
export = gdpr_memory.export_user_data("alice")
with open("user_data_export.json", "w") as f:
    json.dump(export, f, indent=2)
```

<div data-component="PrivacyComplianceFlow"></div>

---

## 本章小结

本章深入讲解了记忆系统的优化与最佳实践：

### 核心技术回顾

1. **Token 管理策略**
   - 精确计数（tiktoken）
   - 实时监控与配额控制
   - 智能截断（保留系统消息 + 最近N轮）
   - 自适应压缩（基于 LLM 的摘要）

2. **检索优化**
   - 向量索引加速（FAISS）
   - LRU 缓存热点记忆
   - 分页加载大规模历史
   - 懒加载减少内存占用

3. **多模态记忆**
   - 图像存储与缩略图
   - 音频转录与元数据
   - 统一多模态检索

4. **并发与一致性**
   - Redis 事务防止冲突
   - 版本控制与冲突解决
   - 最终一致性（Quorum 读写）

5. **隐私与合规**
   - 敏感信息自动脱敏（PII 检测）
   - 数据加密存储（Fernet）
   - GDPR 合规（删除权、数据可携带权）

### 生产环境检查清单

- ✅ 实现 Token 计数与限制
- ✅ 配置合理的截断和压缩策略
- ✅ 使用向量索引加速检索
- ✅ 添加缓存层减少延迟
- ✅ 实现并发控制机制
- ✅ 启用敏感信息脱敏
- ✅ 加密存储关键数据
- ✅ 实现 GDPR 合规功能
- ✅ 监控 Token 使用和成本
- ✅ 定期审计和清理

### 性能优化建议

1. **Token 优化**
   - 目标：控制在 max_tokens 的 70-80%
   - 策略：窗口记忆 + 摘要压缩

2. **检索优化**
   - 向量索引预热（启动时加载）
   - 缓存命中率 > 60%
   - 分页大小：20-50 条消息

3. **存储优化**
   - Redis 用于热数据（< 7天）
   - PostgreSQL 用于温数据（7-90天）
   - S3 用于冷数据（> 90天）

---

## 扩展阅读

- **官方文档**：[Memory Management](https://python.langchain.com/docs/modules/memory/)
- **最佳实践**：[Production Memory Best Practices](https://blog.langchain.dev/memory-best-practices/)
- **性能优化**：[Scaling Memory Systems](https://blog.langchain.dev/scaling-memory/)
- **隐私合规**：[GDPR Compliance in LLM Applications](https://arxiv.org/abs/2304.12345)
- **Token 优化**：[Token Management Strategies](https://platform.openai.com/docs/guides/token-management)
- **向量检索**：[FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss)

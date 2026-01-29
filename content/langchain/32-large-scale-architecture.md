# Chapter 32: 大规模生产系统架构

> **本章导读**  
> 当 LangChain 应用从原型走向生产，从百级用户扩展到百万级用户时，系统架构需要进行全面的重构与优化。本章系统讲解大规模 LLM 系统的架构设计：包括微服务拆分、异步消息队列、分布式缓存、负载均衡、自动扩缩容、多模型路由、流量削峰、熔断降级、灰度发布等核心技术，并通过交互式组件深入理解复杂分布式系统的设计决策，帮助您构建可承载千万级流量的企业级 LLM 平台。

---

## 32.1 从单体到微服务

### 32.1.1 单体架构的瓶颈

**典型单体 LangChain 应用**的特征：
- 所有功能（对话、RAG、Agent、工具调用）打包在一个进程中
- 单一数据库连接池
- 垂直扩展（增加 CPU/内存）
- 发布需停机
- 故障影响全局

```python
# 单体架构示例
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma

app = FastAPI()

# 全局资源（单体架构的典型问题）
llm = ChatOpenAI(model="gpt-4", max_tokens=2000)
vectorstore = Chroma(persist_directory="./chroma_db")
memory_store = {}  # 内存存储（无法跨实例共享）

@app.post("/chat")
async def chat(user_id: str, message: str):
    """所有功能耦合在一起"""
    # 1. 记忆加载（内存瓶颈）
    if user_id not in memory_store:
        memory_store[user_id] = ConversationBufferMemory()
    
    # 2. 向量检索（I/O 瓶颈）
    docs = vectorstore.similarity_search(message, k=5)
    
    # 3. LLM 调用（外部 API 瓶颈）
    context = "\n".join([doc.page_content for doc in docs])
    response = await llm.ainvoke(
        f"Context: {context}\n\nUser: {message}"
    )
    
    # 4. 记忆保存（内存泄漏风险）
    memory_store[user_id].save_context(
        {"input": message},
        {"output": response.content}
    )
    
    return {"response": response.content}

# 问题：
# 1. 内存不共享：多实例下用户记忆丢失
# 2. 资源竞争：向量检索和 LLM 调用共用一个事件循环
# 3. 扩展困难：无法针对热点功能单独扩容
# 4. 单点故障：任何组件崩溃导致整个服务不可用
```

**性能瓶颈分析**：

| 组件 | 瓶颈类型 | 表现 | 影响 |
|------|----------|------|------|
| LLM 调用 | I/O 密集 | 平均 2-5 秒延迟 | 阻塞其他请求 |
| 向量检索 | CPU + I/O | 大规模查询时 CPU 飙升 | 降低吞吐量 |
| 内存管理 | 内存 | 长对话积累，内存泄漏 | OOM 崩溃 |
| 工具执行 | CPU + I/O | 外部 API 调用超时 | 级联故障 |

### 32.1.2 微服务拆分策略

**按职责拆分**的微服务架构：

```
┌─────────────────────────────────────────────────────────┐
│                    API Gateway (FastAPI)                │
│  - 身份认证 - 限流 - 路由 - 协议转换                    │
└────────┬──────────┬──────────┬──────────┬───────────────┘
         │          │          │          │
    ┌────▼────┐ ┌──▼───┐ ┌────▼────┐ ┌───▼────┐
    │ Chat    │ │ RAG  │ │ Agent   │ │ Memory │
    │ Service │ │Service│ │ Service │ │ Service│
    └────┬────┘ └──┬───┘ └────┬────┘ └───┬────┘
         │         │          │          │
    ┌────▼─────────▼──────────▼──────────▼────┐
    │        Message Queue (Redis/RabbitMQ)    │
    └───────────────┬──────────────────────────┘
                    │
         ┌──────────▼──────────┐
         │  LLM Orchestrator   │  ← 智能路由
         │  - GPT-4 Pool       │
         │  - GPT-3.5 Pool     │
         │  - Claude Pool      │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │   Shared Storage    │
         │  - PostgreSQL (元数据)│
         │  - Redis (缓存/会话)│
         │  - S3 (文档/日志)   │
         └─────────────────────┘
```

**实现示例**：

```python
# 1. Chat Service（对话服务）
# chat_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import aio_pika
import json

app = FastAPI(title="Chat Service")

class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str
    model: str = "gpt-4"

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    对话服务：接收用户消息，发送到消息队列异步处理
    """
    # 1. 发布消息到队列
    connection = await aio_pika.connect_robust("amqp://guest:guest@rabbitmq/")
    async with connection:
        channel = await connection.channel()
        
        # 发送到 LLM 编排服务
        await channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps({
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                    "message": request.message,
                    "model": request.model,
                    "service": "chat"
                }).encode()
            ),
            routing_key="llm_requests"
        )
    
    # 2. 返回任务 ID（客户端轮询或 WebSocket 获取结果）
    task_id = f"{request.session_id}:{hash(request.message)}"
    return {
        "task_id": task_id,
        "status": "processing",
        "message": "Request queued successfully"
    }


# 2. RAG Service（检索增强服务）
# rag_service.py
from fastapi import FastAPI
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

app = FastAPI(title="RAG Service")

# 连接专用向量数据库集群
qdrant_client = QdrantClient(host="qdrant-cluster", port=6333)
vectorstore = Qdrant(
    client=qdrant_client,
    collection_name="knowledge_base",
    embeddings=OpenAIEmbeddings()
)

@app.post("/retrieve")
async def retrieve(query: str, top_k: int = 5, filters: dict = None):
    """
    检索服务：向量相似度搜索
    优化：
    - 独立扩容（CPU 密集）
    - 专用向量数据库集群
    - 结果缓存
    """
    # 1. 检查缓存
    cache_key = f"retrieval:{hash(query)}:{top_k}"
    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # 2. 执行检索
    docs = await vectorstore.asimilarity_search(
        query=query,
        k=top_k,
        filter=filters
    )
    
    results = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": doc.metadata.get("score", 0)
        }
        for doc in docs
    ]
    
    # 3. 缓存结果（5分钟）
    await redis_client.setex(cache_key, 300, json.dumps(results))
    
    return {"documents": results}


# 3. Agent Service（Agent 编排服务）
# agent_service.py
from fastapi import FastAPI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool

app = FastAPI(title="Agent Service")

# 工具注册表（动态加载）
tools = [
    Tool(
        name="WebSearch",
        func=lambda q: requests.get(f"https://api.search.com?q={q}").text,
        description="Search the web"
    ),
    Tool(
        name="Calculator",
        func=lambda expr: eval(expr),
        description="Perform calculations"
    )
]

@app.post("/execute")
async def execute_agent(task: str, tools_allowed: list[str] = None):
    """
    Agent 服务：工具调用与任务执行
    优化：
    - 沙箱隔离（避免恶意工具影响其他服务）
    - 超时控制
    - 工具并行执行
    """
    # 过滤允许的工具
    if tools_allowed:
        active_tools = [t for t in tools if t.name in tools_allowed]
    else:
        active_tools = tools
    
    # 创建 Agent
    agent = create_openai_functions_agent(
        llm=ChatOpenAI(model="gpt-4", timeout=30),
        tools=active_tools,
        prompt=ChatPromptTemplate.from_template("{input}")
    )
    
    executor = AgentExecutor(
        agent=agent,
        tools=active_tools,
        max_iterations=5,
        handle_parsing_errors=True
    )
    
    # 执行（带超时）
    try:
        result = await asyncio.wait_for(
            executor.ainvoke({"input": task}),
            timeout=60
        )
        return {"output": result["output"], "steps": len(result.get("intermediate_steps", []))}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Agent execution timeout")


# 4. Memory Service（记忆服务）
# memory_service.py
from fastapi import FastAPI
from redis import asyncio as aioredis
import json

app = FastAPI(title="Memory Service")

# 使用 Redis 集群存储会话
redis = aioredis.Redis(host="redis-cluster", port=6379, decode_responses=True)

@app.get("/memory/{session_id}")
async def get_memory(session_id: str, max_messages: int = 10):
    """获取对话历史"""
    messages = await redis.lrange(f"session:{session_id}", 0, max_messages - 1)
    return {"messages": [json.loads(m) for m in messages]}

@app.post("/memory/{session_id}")
async def save_memory(session_id: str, role: str, content: str):
    """保存消息"""
    message = json.dumps({"role": role, "content": content, "timestamp": time.time()})
    
    # 保存到列表
    await redis.lpush(f"session:{session_id}", message)
    
    # 限制长度（保留最近 50 条）
    await redis.ltrim(f"session:{session_id}", 0, 49)
    
    # 设置过期时间（30天）
    await redis.expire(f"session:{session_id}", 2592000)
    
    return {"status": "saved"}


# 5. LLM Orchestrator（模型编排服务）
# llm_orchestrator.py
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import asyncio
from collections import defaultdict

app = FastAPI(title="LLM Orchestrator")

# 模型池（多实例负载均衡）
model_pools = {
    "gpt-4": [
        ChatOpenAI(model="gpt-4", api_key=key, max_retries=3)
        for key in ["key1", "key2", "key3"]  # 多 API Key 轮询
    ],
    "gpt-3.5-turbo": [
        ChatOpenAI(model="gpt-3.5-turbo", api_key=key, max_retries=3)
        for key in ["key1", "key2", "key3"]
    ],
    "claude-3": [
        ChatAnthropic(model="claude-3-opus-20240229", api_key=key)
        for key in ["claude_key1", "claude_key2"]
    ]
}

# 请求计数器（用于轮询）
request_counters = defaultdict(int)

async def get_model(model_name: str):
    """负载均衡：轮询选择模型实例"""
    pool = model_pools.get(model_name)
    if not pool:
        raise ValueError(f"Model {model_name} not available")
    
    # Round-robin
    idx = request_counters[model_name] % len(pool)
    request_counters[model_name] += 1
    
    return pool[idx]

@app.post("/invoke")
async def invoke_model(model: str, messages: list[dict]):
    """
    模型调用入口
    优化：
    - 多 API Key 轮询（突破限流）
    - 自动重试（指数退避）
    - 故障转移（fallback 到备用模型）
    """
    llm = await get_model(model)
    
    try:
        response = await llm.ainvoke(messages)
        return {"content": response.content, "model": model}
    except Exception as e:
        # Fallback 策略
        if model == "gpt-4":
            print(f"GPT-4 failed: {e}, falling back to GPT-3.5")
            fallback_llm = await get_model("gpt-3.5-turbo")
            response = await fallback_llm.ainvoke(messages)
            return {"content": response.content, "model": "gpt-3.5-turbo", "fallback": True}
        raise


# 6. API Gateway（统一入口）
# api_gateway.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
import httpx

app = FastAPI(title="API Gateway")

# 限流器
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

# 服务注册表（实际生产应使用 Consul/Eureka）
SERVICE_REGISTRY = {
    "chat": "http://chat-service:8001",
    "rag": "http://rag-service:8002",
    "agent": "http://agent-service:8003",
    "memory": "http://memory-service:8004",
    "llm": "http://llm-orchestrator:8005"
}

@app.post("/api/chat")
@limiter.limit("10/minute")  # 每用户每分钟 10 次
async def chat_proxy(request: Request, user_id: str, message: str):
    """
    统一入口：路由到对应微服务
    """
    # 1. 鉴权（JWT验证）
    token = request.headers.get("Authorization")
    if not verify_jwt(token):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # 2. 路由到 Chat Service
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SERVICE_REGISTRY['chat']}/chat",
            json={"user_id": user_id, "message": message}
        )
        return response.json()

def verify_jwt(token: str) -> bool:
    # JWT 验证逻辑
    return True  # 简化示例
```

**预期效果**：

```yaml
# 微服务架构优势：
独立扩容:
  - Chat Service: 3 实例（处理高并发）
  - RAG Service: 5 实例（CPU 密集）
  - Agent Service: 2 实例（资源隔离）
  - Memory Service: 2 实例（共享状态）
  - LLM Orchestrator: 4 实例（多模型并行）

故障隔离:
  - RAG 服务崩溃不影响纯对话功能
  - Agent 超时不阻塞其他请求

资源优化:
  - RAG 服务部署在 CPU 优化实例
  - LLM Orchestrator 部署在网络优化实例
  - Memory 服务使用内存优化实例

部署灵活:
  - 各服务独立发布、回滚
  - 金丝雀发布（先发布 10% 流量测试）
  - A/B 测试（不同服务版本并存）
```

<div data-component="MicroserviceArchitecture"></div>

---

## 32.2 异步消息队列

### 32.2.1 为什么需要消息队列

LLM 应用的特殊性：
1. **长尾延迟**：GPT-4 调用可能需要 5-30 秒
2. **突发流量**：活动期间 QPS 从 10 跃升到 1000
3. **重试需求**：外部 API 偶尔失败需要重试
4. **优先级**：付费用户优先处理

**消息队列解决方案**：

```python
# 使用 Celery + Redis 实现异步任务队列
# celery_app.py
from celery import Celery
from kombu import Queue, Exchange

app = Celery(
    'langchain_tasks',
    broker='redis://redis:6379/0',
    backend='redis://redis:6379/1'
)

# 定义优先级队列
app.conf.task_queues = (
    Queue('high_priority', Exchange('tasks'), routing_key='high', priority=10),
    Queue('normal', Exchange('tasks'), routing_key='normal', priority=5),
    Queue('low_priority', Exchange('tasks'), routing_key='low', priority=1),
)

# 配置
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_acks_late=True,  # 任务执行完再确认（避免丢失）
    worker_prefetch_multiplier=1,  # 一次只取一个任务（公平分配）
    task_time_limit=300,  # 5分钟超时
    task_soft_time_limit=240,  # 4分钟软超时（graceful shutdown）
)


# 定义任务
@app.task(bind=True, max_retries=3, default_retry_delay=60)
def process_chat_request(self, user_id: str, message: str, model: str = "gpt-4"):
    """
    异步处理对话请求
    
    优势：
    - 非阻塞：立即返回，后台处理
    - 重试：失败自动重试（指数退避）
    - 限流：队列天然削峰填谷
    - 可观测：Celery Flower 监控
    """
    try:
        # 1. 加载记忆
        memory = load_memory(user_id)
        
        # 2. 调用 LLM
        llm = ChatOpenAI(model=model)
        response = llm.invoke([
            *memory.get("messages", []),
            {"role": "user", "content": message}
        ])
        
        # 3. 保存记忆
        save_memory(user_id, message, response.content)
        
        # 4. 存储结果（供客户端查询）
        redis_client.setex(
            f"result:{self.request.id}",
            3600,  # 1小时过期
            json.dumps({
                "status": "completed",
                "response": response.content,
                "model": model
            })
        )
        
        return {"status": "success", "task_id": self.request.id}
    
    except Exception as exc:
        # 重试策略
        if self.request.retries < self.max_retries:
            # 指数退避：60s, 120s, 240s
            raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
        else:
            # 最终失败，记录并通知
            redis_client.setex(
                f"result:{self.request.id}",
                3600,
                json.dumps({"status": "failed", "error": str(exc)})
            )
            raise


@app.task
def batch_embedding_task(texts: list[str], collection: str):
    """
    批量嵌入任务（适合大规模文档处理）
    """
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Qdrant
    
    embeddings = OpenAIEmbeddings()
    vectorstore = Qdrant(collection_name=collection)
    
    # 分批处理（避免超时）
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        vectorstore.add_texts(batch)
    
    return {"processed": len(texts)}


# FastAPI 集成
from fastapi import FastAPI, BackgroundTasks
from celery.result import AsyncResult

api = FastAPI()

@api.post("/chat/async")
async def async_chat(user_id: str, message: str, priority: str = "normal"):
    """
    异步对话接口
    """
    # 根据用户等级选择优先级
    routing_key = {
        "premium": "high",
        "normal": "normal",
        "free": "low"
    }.get(priority, "normal")
    
    # 提交任务到队列
    task = process_chat_request.apply_async(
        args=[user_id, message],
        queue=routing_key
    )
    
    return {
        "task_id": task.id,
        "status": "queued",
        "check_url": f"/tasks/{task.id}"
    }

@api.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """
    查询任务状态
    """
    # 先检查 Redis 缓存的结果
    result = redis_client.get(f"result:{task_id}")
    if result:
        return json.loads(result)
    
    # 查询 Celery 状态
    task = AsyncResult(task_id, app=app)
    
    if task.state == 'PENDING':
        return {"status": "pending", "message": "Task is waiting in queue"}
    elif task.state == 'STARTED':
        return {"status": "processing", "message": "Task is being processed"}
    elif task.state == 'SUCCESS':
        return {"status": "completed", "result": task.result}
    elif task.state == 'FAILURE':
        return {"status": "failed", "error": str(task.info)}
    else:
        return {"status": task.state}


# WebSocket 实时推送（可选）
from fastapi import WebSocket

@api.websocket("/ws/chat/{user_id}")
async def websocket_chat(websocket: WebSocket, user_id: str):
    """
    WebSocket 连接：实时推送任务结果
    """
    await websocket.accept()
    
    try:
        while True:
            # 接收消息
            data = await websocket.receive_json()
            message = data.get("message")
            
            # 提交任务
            task = process_chat_request.apply_async(args=[user_id, message])
            
            # 发送任务 ID
            await websocket.send_json({"task_id": task.id, "status": "queued"})
            
            # 轮询任务状态（实际生产应使用 Redis Pub/Sub）
            while True:
                await asyncio.sleep(0.5)
                
                result = redis_client.get(f"result:{task.id}")
                if result:
                    await websocket.send_json(json.loads(result))
                    break
    
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()
```

### 32.2.2 流量削峰与限流

**令牌桶算法**实现：

```python
import time
from collections import defaultdict
from threading import Lock

class TokenBucket:
    """
    令牌桶限流器
    
    原理：
    - 固定速率向桶中添加令牌（rate tokens/second）
    - 请求消耗令牌
    - 桶满时丢弃新令牌（capacity）
    - 无令牌时请求被拒绝
    """
    
    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # 令牌生成速率（tokens/second）
        self.capacity = capacity  # 桶容量
        self.tokens = capacity  # 当前令牌数
        self.last_update = time.time()
        self.lock = Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        尝试消耗令牌
        
        Returns:
            True if successful, False if rate limit exceeded
        """
        with self.lock:
            now = time.time()
            
            # 补充令牌
            elapsed = now - self.last_update
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now
            
            # 消耗令牌
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            else:
                return False

# 全局限流器（每个用户独立）
user_buckets = defaultdict(lambda: TokenBucket(rate=10, capacity=50))

@app.post("/chat")
async def chat_with_rate_limit(user_id: str, message: str):
    """
    带限流的对话接口
    """
    bucket = user_buckets[user_id]
    
    if not bucket.consume():
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "retry_after": int((1 - bucket.tokens) / bucket.rate)
            }
        )
    
    # 正常处理
    return await process_chat(user_id, message)
```

**滑动窗口限流**（更精确）：

```python
import redis.asyncio as aioredis

class SlidingWindowRateLimiter:
    """
    滑动窗口限流（基于 Redis）
    
    优势：
    - 分布式：多实例共享限流状态
    - 精确：基于时间戳，无突刺问题
    """
    
    def __init__(self, redis_url: str):
        self.redis = aioredis.from_url(redis_url)
    
    async def is_allowed(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> tuple[bool, dict]:
        """
        检查是否允许请求
        
        Args:
            key: 限流键（如 user_id）
            max_requests: 窗口内最大请求数
            window_seconds: 窗口大小（秒）
        
        Returns:
            (allowed, {"current": X, "limit": Y, "reset": Z})
        """
        now = time.time()
        window_start = now - window_seconds
        
        # Lua 脚本（原子操作）
        script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local window_start = tonumber(ARGV[2])
        local max_requests = tonumber(ARGV[3])
        local window_seconds = tonumber(ARGV[4])
        
        -- 移除过期记录
        redis.call('ZREMRANGEBYSCORE', key, 0, window_start)
        
        -- 当前请求数
        local current = redis.call('ZCARD', key)
        
        if current < max_requests then
            -- 允许请求，记录时间戳
            redis.call('ZADD', key, now, now)
            redis.call('EXPIRE', key, window_seconds)
            return {1, current + 1, max_requests, window_seconds}
        else
            -- 拒绝请求
            local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')[2]
            local reset_after = math.ceil(tonumber(oldest) + window_seconds - now)
            return {0, current, max_requests, reset_after}
        end
        """
        
        result = await self.redis.eval(
            script,
            1,
            f"rate_limit:{key}",
            now,
            window_start,
            max_requests,
            window_seconds
        )
        
        allowed = result[0] == 1
        
        return allowed, {
            "current": result[1],
            "limit": result[2],
            "reset_after": result[3]
        }

# 使用
limiter = SlidingWindowRateLimiter("redis://localhost")

@app.post("/chat")
async def chat_rate_limited(user_id: str, message: str):
    # 不同用户等级不同限流规则
    tier = get_user_tier(user_id)
    limits = {
        "free": (10, 60),      # 10 req/min
        "premium": (100, 60),  # 100 req/min
        "enterprise": (1000, 60)  # 1000 req/min
    }
    
    max_req, window = limits.get(tier, (10, 60))
    
    allowed, info = await limiter.is_allowed(
        key=f"user:{user_id}",
        max_requests=max_req,
        window_seconds=window
    )
    
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "limit": info["limit"],
                "current": info["current"],
                "reset_after": info["reset_after"]
            }
        )
    
    return await process_chat(user_id, message)
```

---

## 32.3 分布式缓存策略

### 32.3.1 多级缓存架构

```
         ┌──────────────────────────────────┐
         │    Application Memory (L1)       │
         │  - LRU Cache (1000 entries)      │
         │  - TTL: 60s                      │
         │  - Hit Rate: ~40%                │
         └──────────┬───────────────────────┘
                    │ Miss ↓
         ┌──────────▼───────────────────────┐
         │    Redis Cluster (L2)            │
         │  - Distributed Cache             │
         │  - TTL: 1 hour                   │
         │  - Hit Rate: ~35%                │
         └──────────┬───────────────────────┘
                    │ Miss ↓
         ┌──────────▼───────────────────────┐
         │    Database / Vector Store (L3)  │
         │  - PostgreSQL / Qdrant           │
         │  - Source of Truth               │
         │  - Hit Rate: 100% (fallback)     │
         └──────────────────────────────────┘
```

**实现**：

```python
from functools import lru_cache
from typing import Optional
import hashlib
import json

class MultiLevelCache:
    """
    三级缓存系统
    
    L1: 本地内存（functools.lru_cache）
    L2: Redis 分布式缓存
    L3: 原始数据源
    """
    
    def __init__(self, redis_client, ttl_l1: int = 60, ttl_l2: int = 3600):
        self.redis = redis_client
        self.ttl_l1 = ttl_l1
        self.ttl_l2 = ttl_l2
        self.stats = {"l1_hits": 0, "l2_hits": 0, "l3_hits": 0}
    
    def _make_key(self, prefix: str, **kwargs) -> str:
        """生成缓存键"""
        params = json.dumps(kwargs, sort_keys=True)
        hash_val = hashlib.md5(params.encode()).hexdigest()[:8]
        return f"{prefix}:{hash_val}"
    
    async def get_or_compute(
        self,
        key_prefix: str,
        compute_func,
        **kwargs
    ):
        """
        三级缓存查询
        """
        cache_key = self._make_key(key_prefix, **kwargs)
        
        # L1: 内存缓存
        l1_result = self._get_l1(cache_key)
        if l1_result is not None:
            self.stats["l1_hits"] += 1
            return l1_result
        
        # L2: Redis 缓存
        l2_result = await self._get_l2(cache_key)
        if l2_result is not None:
            self.stats["l2_hits"] += 1
            self._set_l1(cache_key, l2_result)  # 回填 L1
            return l2_result
        
        # L3: 计算/查询原始数据
        self.stats["l3_hits"] += 1
        result = await compute_func(**kwargs)
        
        # 回填 L2 和 L1
        await self._set_l2(cache_key, result)
        self._set_l1(cache_key, result)
        
        return result
    
    @lru_cache(maxsize=1000)
    def _get_l1(self, key: str):
        """L1 缓存（装饰器自动管理）"""
        return None  # 实际由 lru_cache 拦截
    
    def _set_l1(self, key: str, value):
        """手动设置 L1（通过调用 _get_l1）"""
        self._get_l1.__wrapped__.__setitem__(key, value)
    
    async def _get_l2(self, key: str):
        """L2 Redis 缓存"""
        value = await self.redis.get(key)
        return json.loads(value) if value else None
    
    async def _set_l2(self, key: str, value):
        """设置 L2 缓存"""
        await self.redis.setex(
            key,
            self.ttl_l2,
            json.dumps(value)
        )
    
    def get_stats(self) -> dict:
        """缓存命中率统计"""
        total = sum(self.stats.values())
        if total == 0:
            return {}
        
        return {
            "l1_hit_rate": self.stats["l1_hits"] / total,
            "l2_hit_rate": self.stats["l2_hits"] / total,
            "l3_hit_rate": self.stats["l3_hits"] / total,
            **self.stats
        }

# 使用示例
cache = MultiLevelCache(redis_client=redis)

async def expensive_retrieval(query: str, top_k: int):
    """模拟昂贵的向量检索"""
    docs = await vectorstore.asimilarity_search(query, k=top_k)
    return [doc.page_content for doc in docs]

@app.get("/search")
async def search(query: str, top_k: int = 5):
    """
    带三级缓存的检索接口
    """
    results = await cache.get_or_compute(
        key_prefix="retrieval",
        compute_func=expensive_retrieval,
        query=query,
        top_k=top_k
    )
    
    return {
        "results": results,
        "cache_stats": cache.get_stats()
    }
```

**预期效果**：

```
# 缓存命中率提升性能
Before Multi-Level Cache:
- Average Latency: 800ms
- P99 Latency: 2000ms
- QPS: 50

After Multi-Level Cache:
- Average Latency: 50ms (16x faster)
- P99 Latency: 200ms (10x faster)
- QPS: 500 (10x throughput)
- Cache Hit Rate: 75% (L1: 40%, L2: 35%)
```

### 32.3.2 语义缓存（Semantic Cache）

传统缓存要求完全匹配，语义缓存允许相似问题复用：

```python
from langchain_community.cache import RedisSemanticCache
from langchain_openai import OpenAIEmbeddings
from langchain_core.globals import set_llm_cache

# 配置语义缓存
set_llm_cache(RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=OpenAIEmbeddings(),
    score_threshold=0.85  # 相似度阈值
))

llm = ChatOpenAI(model="gpt-4")

# 第一次查询（未命中缓存）
response1 = llm.invoke("What's the capital of France?")
# → 调用 GPT-4，耗时 2s

# 相似查询（命中语义缓存！）
response2 = llm.invoke("What is France's capital city?")
# → 直接返回缓存，耗时 50ms

# 原理：
# 1. 计算问题的嵌入向量
# 2. 在 Redis 中搜索相似向量（余弦相似度 > 0.85）
# 3. 找到则返回缓存的答案，否则调用 LLM
```

**高级语义缓存**（支持参数化）：

```python
import numpy as np
from typing import Optional

class AdvancedSemanticCache:
    """
    高级语义缓存
    
    特性：
    - 参数化查询（支持变量替换）
    - 时效性控制（新闻类查询短TTL）
    - 多模态（文本+图像）
    """
    
    def __init__(
        self,
        vectorstore,
        embedding_model,
        similarity_threshold: float = 0.85
    ):
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model
        self.threshold = similarity_threshold
    
    async def get_or_compute(
        self,
        query: str,
        compute_func,
        ttl: Optional[int] = None,
        metadata: dict = None
    ):
        """
        语义缓存查询
        
        Args:
            query: 查询文本
            compute_func: 计算函数（未命中时调用）
            ttl: 缓存有效期（秒）
            metadata: 过滤条件（如 {"topic": "finance"}）
        """
        # 1. 计算查询嵌入
        query_embedding = await self.embedding_model.aembed_query(query)
        
        # 2. 向量相似度搜索
        results = await self.vectorstore.asimilarity_search_with_score(
            query=query,
            k=1,
            filter=metadata
        )
        
        if results and results[0][1] >= self.threshold:
            # 命中缓存
            cached_doc = results[0][0]
            
            # 检查是否过期
            if ttl:
                timestamp = cached_doc.metadata.get("timestamp", 0)
                if time.time() - timestamp > ttl:
                    # 过期，删除并重新计算
                    await self._invalidate(cached_doc.metadata["cache_id"])
                else:
                    return json.loads(cached_doc.metadata["response"])
            else:
                return json.loads(cached_doc.metadata["response"])
        
        # 3. 未命中，执行计算
        response = await compute_func(query)
        
        # 4. 存入缓存
        cache_id = hashlib.md5(query.encode()).hexdigest()
        await self.vectorstore.aadd_texts(
            texts=[query],
            metadatas=[{
                "cache_id": cache_id,
                "response": json.dumps(response),
                "timestamp": time.time(),
                **(metadata or {})
            }]
        )
        
        return response
    
    async def _invalidate(self, cache_id: str):
        """删除缓存条目"""
        await self.vectorstore.adelete(filter={"cache_id": cache_id})

# 使用
semantic_cache = AdvancedSemanticCache(
    vectorstore=qdrant,
    embedding_model=OpenAIEmbeddings()
)

@app.get("/ask")
async def ask_with_semantic_cache(question: str, topic: str = None):
    async def call_llm(q):
        return (await ChatOpenAI(model="gpt-4").ainvoke(q)).content
    
    # 新闻类查询：TTL=1小时
    # 知识类查询：TTL=None（永久）
    ttl = 3600 if topic == "news" else None
    
    answer = await semantic_cache.get_or_compute(
        query=question,
        compute_func=call_llm,
        ttl=ttl,
        metadata={"topic": topic} if topic else None
    )
    
    return {"answer": answer}
```

---

## 32.4 负载均衡与自动扩缩容

### 32.4.1 Kubernetes 自动扩缩容

**Horizontal Pod Autoscaler (HPA)** 配置：

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langchain-api
spec:
  replicas: 3  # 初始副本数
  selector:
    matchLabels:
      app: langchain-api
  template:
    metadata:
      labels:
        app: langchain-api
    spec:
      containers:
      - name: api
        image: langchain-api:v1.0
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis-cluster:6379"
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
# HPA 配置
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: langchain-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: langchain-api
  minReplicas: 3
  maxReplicas: 50
  metrics:
  # CPU 利用率
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70  # CPU > 70% 扩容
  
  # 内存利用率
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80  # 内存 > 80% 扩容
  
  # 自定义指标：请求队列长度
  - type: Pods
    pods:
      metric:
        name: celery_queue_length
      target:
        type: AverageValue
        averageValue: "10"  # 平均队列长度 > 10 扩容
  
  # 自定义指标：P99 延迟
  - type: Pods
    pods:
      metric:
        name: http_request_duration_p99
      target:
        type: AverageValue
        averageValue: "1000"  # P99 > 1s 扩容
  
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 0  # 立即扩容
      policies:
      - type: Percent
        value: 100  # 每次扩容 100%（翻倍）
        periodSeconds: 15
      - type: Pods
        value: 5  # 或每次增加 5 个 Pod
        periodSeconds: 15
      selectPolicy: Max  # 选择最激进的策略
    
    scaleDown:
      stabilizationWindowSeconds: 300  # 5分钟稳定期（避免抖动）
      policies:
      - type: Percent
        value: 50  # 每次缩容 50%
        periodSeconds: 60
      - type: Pods
        value: 2  # 或每次减少 2 个 Pod
        periodSeconds: 60
      selectPolicy: Min  # 选择最保守的策略

---
# Service (负载均衡)
apiVersion: v1
kind: Service
metadata:
  name: langchain-api-service
spec:
  selector:
    app: langchain-api
  type: LoadBalancer
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  sessionAffinity: ClientIP  # 会话保持（可选）
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 3600  # 1小时

---
# Ingress (七层负载均衡 + SSL)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: langchain-api-ingress
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"  # 100 req/s per IP
    nginx.ingress.kubernetes.io/limit-rps: "10"    # 10 req/s per connection
spec:
  tls:
  - hosts:
    - api.example.com
    secretName: api-tls-secret
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: langchain-api-service
            port:
              number: 80
```

**自定义指标收集**（Prometheus）：

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response
import time

# 定义指标
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]  # 自定义分位点
)

celery_queue_length = Gauge(
    'celery_queue_length',
    'Number of tasks in Celery queue',
    ['queue_name']
)

llm_token_count = Counter(
    'llm_tokens_used_total',
    'Total tokens consumed',
    ['model']
)

# 中间件：自动记录指标
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response

# Prometheus 端点
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")

# 定期更新队列长度（后台任务）
from celery import Celery

celery_app = Celery(broker='redis://redis:6379/0')

async def update_queue_metrics():
    while True:
        # 查询各队列长度
        inspect = celery_app.control.inspect()
        active = inspect.active() or {}
        reserved = inspect.reserved() or {}
        
        for queue in ['high_priority', 'normal', 'low_priority']:
            length = sum(
                len(tasks.get(queue, []))
                for tasks in [active, reserved]
            )
            celery_queue_length.labels(queue_name=queue).set(length)
        
        await asyncio.sleep(5)

# 启动后台任务
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(update_queue_metrics())
```

**预期效果**：

```
# 扩缩容场景演示

场景 1：突发流量（活动推广）
时间     | QPS | CPU | 副本数 | 动作
---------|-----|-----|--------|------
09:00:00 | 50  | 30% | 3      | 稳定
09:15:00 | 500 | 85% | 3      | 触发扩容
09:15:15 | 500 | 45% | 6      | 扩容完成（翻倍）
09:15:30 | 500 | 45% | 6      | 稳定
09:30:00 | 1000| 75% | 6      | 再次扩容
09:30:15 | 1000| 40% | 12     | 扩容完成
10:00:00 | 100 | 15% | 12     | 流量下降
10:05:00 | 100 | 15% | 6      | 缩容（5分钟稳定期）
10:10:00 | 100 | 30% | 3      | 继续缩容

场景 2：队列积压
队列长度 | 副本数 | 动作
---------|--------|------
50       | 3      | 正常
150      | 5      | 扩容（平均 > 10）
300      | 10     | 持续扩容
50       | 10     | 队列清空
20       | 5      | 缩容
10       | 3      | 恢复正常

成本优化：
- 峰值时自动扩容至 50 副本
- 低峰时缩减至 3 副本
- 平均节省 60% 资源成本
```

---

## 32.5 多模型路由与智能调度

### 32.5.1 基于任务复杂度的模型选择

```python
from enum import Enum
from typing import Literal

class TaskComplexity(Enum):
    SIMPLE = "simple"       # 简单任务：GPT-3.5
    MEDIUM = "medium"       # 中等任务：GPT-4
    COMPLEX = "complex"     # 复杂任务：GPT-4 + CoT
    SPECIALIZED = "specialized"  # 专业任务：Claude-3

class ModelRouter:
    """
    智能模型路由器
    
    策略：
    - 简单查询 → GPT-3.5（便宜、快速）
    - 需要推理 → GPT-4
    - 代码生成 → Claude-3
    - 长文本 → Claude-3 (200K context)
    """
    
    def __init__(self):
        self.models = {
            "gpt-3.5-turbo": {
                "cost_per_1k_tokens": 0.002,
                "max_tokens": 16000,
                "avg_latency": 1.5,
                "strengths": ["speed", "cost"]
            },
            "gpt-4": {
                "cost_per_1k_tokens": 0.03,
                "max_tokens": 128000,
                "avg_latency": 3.0,
                "strengths": ["reasoning", "accuracy"]
            },
            "claude-3-opus": {
                "cost_per_1k_tokens": 0.015,
                "max_tokens": 200000,
                "avg_latency": 2.5,
                "strengths": ["coding", "long_context"]
            }
        }
    
    def classify_task(self, prompt: str, context: str = "") -> TaskComplexity:
        """
        任务复杂度分类
        
        规则：
        1. 长度 < 100 字 && 无专业词汇 → SIMPLE
        2. 包含 "analyze", "compare", "explain" → MEDIUM
        3. 包含 "code", "algorithm", "debug" → SPECIALIZED
        4. context > 10K tokens → COMPLEX
        """
        prompt_lower = prompt.lower()
        total_length = len(prompt) + len(context)
        
        # 关键词检测
        simple_keywords = ["what is", "define", "translate", "summarize"]
        complex_keywords = ["analyze", "compare", "evaluate", "critique"]
        code_keywords = ["code", "function", "algorithm", "debug", "implement"]
        
        if any(kw in prompt_lower for kw in code_keywords):
            return TaskComplexity.SPECIALIZED
        
        if total_length > 10000 or any(kw in prompt_lower for kw in complex_keywords):
            return TaskComplexity.COMPLEX
        
        if len(prompt) < 100 and any(kw in prompt_lower for kw in simple_keywords):
            return TaskComplexity.SIMPLE
        
        return TaskComplexity.MEDIUM
    
    def select_model(
        self,
        prompt: str,
        context: str = "",
        user_tier: str = "free",
        priority: str = "balanced"
    ) -> str:
        """
        选择最优模型
        
        Args:
            prompt: 用户提示
            context: 上下文
            user_tier: 用户等级（free/premium/enterprise）
            priority: 优先级（cost/speed/quality/balanced）
        
        Returns:
            model_name
        """
        complexity = self.classify_task(prompt, context)
        
        # 用户等级限制
        if user_tier == "free":
            # 免费用户只能用 GPT-3.5
            return "gpt-3.5-turbo"
        
        # 根据复杂度和优先级选择
        if complexity == TaskComplexity.SIMPLE:
            return "gpt-3.5-turbo"  # 简单任务无需强模型
        
        elif complexity == TaskComplexity.SPECIALIZED:
            return "claude-3-opus"  # 代码类任务用 Claude
        
        elif complexity == TaskComplexity.COMPLEX:
            if priority == "cost":
                return "gpt-3.5-turbo"
            elif priority == "quality":
                return "gpt-4"
            else:  # balanced
                return "gpt-4" if user_tier == "enterprise" else "gpt-3.5-turbo"
        
        else:  # MEDIUM
            if priority == "speed":
                return "gpt-3.5-turbo"
            else:
                return "gpt-4"
    
    async def route_request(
        self,
        prompt: str,
        context: str = "",
        **kwargs
    ) -> dict:
        """
        路由并执行请求
        """
        model = self.select_model(prompt, context, **kwargs)
        
        # 根据模型创建 LLM 实例
        llm_map = {
            "gpt-3.5-turbo": ChatOpenAI(model="gpt-3.5-turbo"),
            "gpt-4": ChatOpenAI(model="gpt-4"),
            "claude-3-opus": ChatAnthropic(model="claude-3-opus-20240229")
        }
        
        llm = llm_map[model]
        
        # 执行
        start_time = time.time()
        response = await llm.ainvoke(prompt)
        latency = time.time() - start_time
        
        # 计算成本
        token_count = len(response.content.split())  # 简化估算
        cost = token_count / 1000 * self.models[model]["cost_per_1k_tokens"]
        
        return {
            "response": response.content,
            "model_used": model,
            "latency": latency,
            "cost": cost,
            "tokens": token_count
        }

# 使用
router = ModelRouter()

@app.post("/chat")
async def chat_with_routing(
    user_id: str,
    message: str,
    tier: str = "free",
    priority: str = "balanced"
):
    result = await router.route_request(
        prompt=message,
        user_tier=tier,
        priority=priority
    )
    
    return result
```

**预期效果**：

```
# 成本与延迟优化对比

Without Routing (All GPT-4):
- Average Cost: $0.05/request
- Average Latency: 3.2s
- Monthly Cost (1M requests): $50,000

With Smart Routing:
- Simple (60%): GPT-3.5 → $0.002/req
- Medium (30%): GPT-4 → $0.03/req
- Complex (10%): GPT-4 → $0.03/req
- Average Cost: $0.011/request (78% reduction)
- Average Latency: 2.1s (34% faster)
- Monthly Cost (1M requests): $11,000 (节省 $39,000)

Quality Metrics:
- User Satisfaction: 无显著差异（简单任务无需 GPT-4）
- Task Success Rate: 98% (vs 99% all GPT-4)
```

<div data-component="ModelRoutingFlow"></div>

---

## 32.6 灰度发布与 A/B 测试

### 32.6.1 金丝雀部署（Canary Deployment）

```yaml
# 使用 Istio 进行流量分割
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: langchain-api-vs
spec:
  hosts:
  - langchain-api-service
  http:
  - match:
    - headers:
        X-Beta-User:
          exact: "true"
    route:
    - destination:
        host: langchain-api-service
        subset: v2  # 新版本
      weight: 100
  
  - route:
    - destination:
        host: langchain-api-service
        subset: v1  # 旧版本
      weight: 95
    - destination:
        host: langchain-api-service
        subset: v2  # 新版本
      weight: 5  # 5% 流量到新版本

---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: langchain-api-dr
spec:
  host: langchain-api-service
  subsets:
  - name: v1
    labels:
      version: v1
  - name: v2
    labels:
      version: v2
```

**渐进式发布策略**：

```python
# canary_controller.py
import asyncio
from dataclasses import dataclass
from typing import Literal

@dataclass
class CanaryConfig:
    initial_traffic: float = 0.05  # 初始流量 5%
    increment: float = 0.10  # 每阶段增加 10%
    interval_minutes: int = 30  # 观察期 30 分钟
    rollback_threshold: float = 0.05  # 错误率 > 5% 回滚

class CanaryController:
    """
    金丝雀发布控制器
    
    流程：
    1. 部署新版本（0% 流量）
    2. 逐步增加流量（5% → 15% → 30% → 50% → 100%）
    3. 每个阶段观察指标（错误率、延迟、用户反馈）
    4. 异常则自动回滚
    """
    
    def __init__(self, config: CanaryConfig):
        self.config = config
        self.current_traffic = 0.0
    
    async def deploy_canary(self, new_version: str):
        """
        执行金丝雀部署
        """
        print(f"🚀 Starting canary deployment for {new_version}")
        
        # 阶段 1：初始流量
        await self._update_traffic(self.config.initial_traffic)
        await self._observe(self.config.interval_minutes)
        
        # 阶段 2-N：逐步增加
        while self.current_traffic < 1.0:
            next_traffic = min(
                1.0,
                self.current_traffic + self.config.increment
            )
            
            await self._update_traffic(next_traffic)
            
            # 观察期
            is_healthy = await self._observe(self.config.interval_minutes)
            
            if not is_healthy:
                # 回滚
                print(f"❌ Canary unhealthy, rolling back...")
                await self._rollback()
                return False
        
        print(f"✅ Canary deployment completed successfully")
        return True
    
    async def _update_traffic(self, target_traffic: float):
        """更新流量分配"""
        print(f"📊 Updating traffic: {self.current_traffic*100:.0f}% → {target_traffic*100:.0f}%")
        
        # 调用 Istio API 更新 VirtualService
        # 简化示例，实际应调用 Kubernetes API
        self.current_traffic = target_traffic
    
    async def _observe(self, minutes: int) -> bool:
        """
        观察期：监控关键指标
        
        Returns:
            True if healthy, False if should rollback
        """
        print(f"👀 Observing for {minutes} minutes...")
        
        for i in range(minutes):
            await asyncio.sleep(60)  # 每分钟检查一次
            
            # 查询 Prometheus 指标
            metrics = await self._get_metrics()
            
            # 检查健康状态
            if metrics['error_rate'] > self.config.rollback_threshold:
                print(f"🚨 Error rate too high: {metrics['error_rate']:.2%}")
                return False
            
            if metrics['p99_latency'] > metrics['baseline_p99'] * 1.5:
                print(f"🚨 Latency degradation: {metrics['p99_latency']:.2f}s")
                return False
            
            print(f"  [{i+1}/{minutes}] ✅ Metrics OK")
        
        return True
    
    async def _get_metrics(self) -> dict:
        """从 Prometheus 查询指标"""
        # 简化示例
        return {
            'error_rate': 0.02,  # 2%
            'p99_latency': 1.8,  # 1.8s
            'baseline_p99': 2.0  # baseline 2.0s
        }
    
    async def _rollback(self):
        """回滚到旧版本"""
        await self._update_traffic(0.0)
        print("✅ Rollback completed")

# 使用
controller = CanaryController(CanaryConfig())
await controller.deploy_canary("v2.0")
```

### 32.6.2 A/B 测试框架

```python
import hashlib
from typing import Literal

class ABTestFramework:
    """
    A/B 测试框架
    
    场景：
    - 测试新提示模板效果
    - 测试新模型（GPT-4 vs Claude-3）
    - 测试新检索策略
    """
    
    def __init__(self):
        self.experiments = {}
    
    def create_experiment(
        self,
        name: str,
        variants: dict,
        traffic_split: dict,
        success_metric: str
    ):
        """
        创建 A/B 实验
        
        Args:
            name: 实验名称
            variants: 变体配置 {"control": {...}, "treatment": {...}}
            traffic_split: 流量分配 {"control": 0.5, "treatment": 0.5}
            success_metric: 成功指标（如 "user_satisfaction"）
        """
        self.experiments[name] = {
            "variants": variants,
            "traffic_split": traffic_split,
            "success_metric": success_metric,
            "results": {v: {"count": 0, "sum": 0} for v in variants}
        }
    
    def assign_variant(self, experiment_name: str, user_id: str) -> str:
        """
        为用户分配变体（一致性哈希）
        
        保证：同一用户始终看到同一变体
        """
        exp = self.experiments[experiment_name]
        
        # 哈希用户 ID
        hash_val = int(hashlib.md5(
            f"{experiment_name}:{user_id}".encode()
        ).hexdigest(), 16)
        
        # 根据流量分配确定变体
        ratio = (hash_val % 100) / 100.0
        cumulative = 0.0
        
        for variant, traffic in exp["traffic_split"].items():
            cumulative += traffic
            if ratio < cumulative:
                return variant
        
        return list(exp["variants"].keys())[0]  # fallback
    
    def record_result(
        self,
        experiment_name: str,
        variant: str,
        metric_value: float
    ):
        """记录实验结果"""
        results = self.experiments[experiment_name]["results"][variant]
        results["count"] += 1
        results["sum"] += metric_value
    
    def get_results(self, experiment_name: str) -> dict:
        """获取实验统计结果"""
        exp = self.experiments[experiment_name]
        
        stats = {}
        for variant, data in exp["results"].items():
            if data["count"] > 0:
                stats[variant] = {
                    "mean": data["sum"] / data["count"],
                    "count": data["count"]
                }
            else:
                stats[variant] = {"mean": 0, "count": 0}
        
        # 计算统计显著性（简化版 t-test）
        if len(stats) == 2:
            variants = list(stats.keys())
            mean_diff = abs(stats[variants[0]]["mean"] - stats[variants[1]]["mean"])
            stats["significant"] = mean_diff > 0.05  # 简化判断
        
        return stats

# 使用示例
ab_test = ABTestFramework()

# 实验：测试新的 RAG 策略
ab_test.create_experiment(
    name="rag_strategy_test",
    variants={
        "control": {"retriever": "standard", "top_k": 5},
        "treatment": {"retriever": "hybrid", "top_k": 10}
    },
    traffic_split={"control": 0.5, "treatment": 0.5},
    success_metric="answer_quality"
)

@app.post("/ask")
async def ask(user_id: str, question: str):
    # 分配变体
    variant = ab_test.assign_variant("rag_strategy_test", user_id)
    config = ab_test.experiments["rag_strategy_test"]["variants"][variant]
    
    # 执行对应策略
    if config["retriever"] == "hybrid":
        docs = await hybrid_retriever(question, top_k=config["top_k"])
    else:
        docs = await standard_retriever(question, top_k=config["top_k"])
    
    # 生成答案
    answer = await generate_answer(question, docs)
    
    # 记录（后续用户反馈时）
    # ab_test.record_result("rag_strategy_test", variant, quality_score)
    
    return {
        "answer": answer,
        "experiment_variant": variant  # 可选：告知用户参与实验
    }

# 查看结果
print(ab_test.get_results("rag_strategy_test"))
# {'control': {'mean': 4.2, 'count': 500}, 'treatment': {'mean': 4.5, 'count': 500}, 'significant': True}
```

<div data-component="ABTestDashboard"></div>

---

## 32.7 总结

本章深入讲解了大规模 LangChain 生产系统的架构设计，覆盖：

1. **微服务架构**：职责分离、独立扩容、故障隔离
2. **异步消息队列**：削峰填谷、重试、优先级
3. **多级缓存**：L1/L2/L3、语义缓存、命中率优化
4. **负载均衡**：HPA 自动扩缩容、自定义指标
5. **模型路由**：智能调度、成本优化（78% 节省）
6. **灰度发布**：金丝雀部署、A/B 测试、自动回滚

**核心原则**：
- ✅ **解耦**：服务间松耦合，独立演进
- ✅ **弹性**：自动扩缩容，应对流量波动
- ✅ **高可用**：故障隔离、熔断降级、多副本
- ✅ **可观测**：指标监控、分布式追踪、告警
- ✅ **成本优化**：智能路由、缓存、按需扩容

**从 100 用户到 1000 万用户**的架构演进路径已清晰，下一章将探讨 LangChain 的未来演进方向与研究前沿。

---

## 扩展阅读

- [Microservices Patterns (Chris Richardson)](https://microservices.io/patterns/index.html)
- [Kubernetes HPA Documentation](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Istio Traffic Management](https://istio.io/latest/docs/concepts/traffic-management/)
- [Celery Best Practices](https://docs.celeryq.dev/en/stable/userguide/tasks.html)
- [Redis Caching Strategies](https://redis.io/docs/manual/patterns/)

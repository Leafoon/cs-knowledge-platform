# Chapter 30：性能优化与可靠性工程

在将 LangChain 应用从原型推向生产环境时，性能优化与可靠性工程成为关键考量。本章将系统讲解缓存策略、批处理、异步执行、成本控制、重试机制、降级策略、监控告警等生产级实践，并结合真实案例展示如何在保证功能的同时将系统打造成高可用、高性能、低成本的稳定服务。

---

## 30.1 缓存策略：加速重复调用

LLM 调用成本高、延迟大，缓存是最直接的优化手段。LangChain 提供多层缓存方案：内存缓存（开发/演示）、Redis 缓存（生产分布式）、SQLite 缓存（单机持久化）。

### 30.1.1 内存缓存：快速原型

**InMemoryCache** 将结果存在进程内存中，适用于开发测试或单机短期服务。

```python
from langchain_openai import ChatOpenAI
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
import time

set_llm_cache(InMemoryCache())

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 第一次调用：无缓存
start = time.time()
result1 = llm.invoke("What is the capital of France?")
print(f"第一次调用耗时: {time.time() - start:.2f}s")
print(result1.content)

# 第二次调用：命中缓存
start = time.time()
result2 = llm.invoke("What is the capital of France?")
print(f"第二次调用耗时: {time.time() - start:.2f}s")  # 几乎为 0
print(result2.content)
```

**预期输出：**
```
第一次调用耗时: 1.23s
The capital of France is Paris.
第二次调用耗时: 0.001s
The capital of France is Paris.
```

**工作原理：** LangChain 将 `(prompt, model_name, temperature, ...)` 构建哈希键，映射到响应对象，完全匹配的请求直接返回缓存。

### 30.1.2 Redis 缓存：分布式共享

**RedisCache** 适用于多实例/多进程共享缓存，部署在生产环境中可大幅减少重复调用。

```python
from langchain.cache import RedisCache
import redis

redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=False  # LangChain 需要 bytes
)

set_llm_cache(RedisCache(redis_client))

llm = ChatOpenAI(model="gpt-4o-mini")
result = llm.invoke("Translate 'Hello' to French")
# 后续相同请求可跨进程/实例直接从 Redis 读取
```

**最佳实践：**
- 设置 TTL（过期时间）避免缓存无限增长：
  ```python
  redis_client.expire("langchain:cache:abc123...", 86400)  # 1 天过期
  ```
- 使用 Redis Cluster 支持高可用与水平扩展
- 监控缓存命中率（`INFO stats` 中的 `keyspace_hits / keyspace_misses`）

### 30.1.3 SQLite 缓存：持久化单机存储

**SQLiteCache** 将缓存持久化到 SQLite 数据库文件，服务重启后缓存依然可用。

```python
from langchain.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path=".langchain.db"))

llm = ChatOpenAI(model="gpt-4o-mini")
result = llm.invoke("What is 2+2?")
# 缓存保存在 .langchain.db 文件中，重启进程后依然有效
```

**场景选择：**
| 场景 | 推荐缓存 | 原因 |
|------|---------|------|
| 本地开发/演示 | InMemoryCache | 无需外部依赖，快速启动 |
| 单机生产服务 | SQLiteCache | 持久化，重启不丢失 |
| 分布式服务 | RedisCache | 跨实例共享，支持集群 |
| 高并发 API | Redis + 分片 | 高吞吐、低延迟 |

<div data-component="CachingStrategyComparison"></div>

### 30.1.4 语义缓存：容忍语义相似请求

标准缓存要求完全匹配，**SemanticCache** 使用向量相似度匹配，可容忍语义相似的不同表述。

```python
from langchain.cache import RedisSemanticCache
from langchain_openai import OpenAIEmbeddings

embedding = OpenAIEmbeddings()
semantic_cache = RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=embedding,
    score_threshold=0.9  # 余弦相似度 >= 0.9 视为命中
)

set_llm_cache(semantic_cache)

llm = ChatOpenAI()
result1 = llm.invoke("What is Python?")
result2 = llm.invoke("Tell me about the Python programming language")
# 第二个请求可能命中第一个的缓存（取决于 embedding 相似度）
```

**权衡：**
- **优点：** 提高缓存命中率，对用户表述多样性鲁棒
- **缺点：** 需计算 embedding（额外延迟），可能返回不完全匹配的结果（需权衡精确性）

---

## 30.2 批处理：提升吞吐量

批处理将多个请求合并为单次 API 调用，减少网络往返、平摊固定开销。

### 30.2.1 LLM.batch()：并行批量调用

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

prompts = [
    "Translate 'dog' to French",
    "Translate 'cat' to Spanish",
    "Translate 'bird' to German"
]

results = llm.batch(prompts)
for i, result in enumerate(results):
    print(f"{prompts[i]} -> {result.content}")
```

**预期输出：**
```
Translate 'dog' to French -> chien
Translate 'cat' to Spanish -> gato
Translate 'bird' to German -> Vogel
```

**底层原理：** `batch()` 在内部使用并发请求（`asyncio.gather()` 或 ThreadPoolExecutor），将多个请求同时发送给 API，等待全部完成后返回。

### 30.2.2 Runnable.batch()：链批处理

LCEL 链也支持批处理：

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Tell me a {adjective} joke")
chain = prompt | llm

inputs = [
    {"adjective": "funny"},
    {"adjective": "dark"},
    {"adjective": "tech"}
]

results = chain.batch(inputs)
for inp, res in zip(inputs, results):
    print(f"{inp['adjective']}: {res.content[:50]}...")
```

**性能对比（10 个请求）：**
- 顺序调用：10 × 1.2s = 12s
- 批处理：~2s（并发执行）

### 30.2.3 批处理最佳实践

1. **批次大小控制：** 过大批次可能触发 API 速率限制或超时，建议每批 5-20 个
2. **异常处理：** 批处理中单个失败不应影响其他请求，使用 `max_concurrency` + 错误捕获：
   ```python
   results = chain.batch(inputs, config={"max_concurrency": 5})
   ```
3. **结果顺序保证：** LangChain 保证返回顺序与输入顺序一致

---

## 30.3 异步执行：提升并发能力

同步 API 在 I/O 等待时阻塞线程，异步可让单进程处理大量并发请求。

### 30.3.1 异步调用基础

LangChain 所有 `invoke()` 都有对应的 `ainvoke()`（异步版本）：

```python
import asyncio
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()

async def translate(text: str):
    result = await llm.ainvoke(f"Translate '{text}' to French")
    return result.content

async def main():
    tasks = [
        translate("hello"),
        translate("goodbye"),
        translate("thank you")
    ]
    results = await asyncio.gather(*tasks)
    print(results)

asyncio.run(main())
```

**预期输出：**
```
['bonjour', 'au revoir', 'merci']
```

### 30.3.2 异步流式生成

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Write a poem about {topic}")
chain = prompt | llm

async def stream_poem():
    async for chunk in chain.astream({"topic": "winter"}):
        print(chunk.content, end="", flush=True)

asyncio.run(stream_poem())
```

**输出：** 逐 token 流式打印诗歌内容。

### 30.3.3 异步 vs 批处理

- **批处理：** 适合已知固定数量的请求，批量提交
- **异步：** 适合动态/长期运行的并发任务（如 WebSocket 服务、多用户同时请求）

---

## 30.4 成本控制：Token 计数与模型路由

LLM 成本与 token 使用成正比，生产环境需精细化成本管理。

### 30.4.1 Token 计数

使用 `tiktoken` 库精确计算 token 数（OpenAI 模型）：

```python
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4o-mini")

text = "This is a sample text for token counting."
tokens = encoding.encode(text)
print(f"Token count: {len(tokens)}")  # 输出: 9
```

**在 LangChain 中集成：**
```python
from langchain.callbacks import get_openai_callback

with get_openai_callback() as cb:
    result = llm.invoke("Tell me a joke")
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost:.4f}")
```

**预期输出：**
```
Total Tokens: 45
Prompt Tokens: 12
Completion Tokens: 33
Total Cost (USD): $0.0003
```

### 30.4.2 模型路由：按任务选择模型

不同任务对模型能力要求不同，简单任务使用小模型（gpt-4o-mini）、复杂推理使用大模型（gpt-4o）：

```python
from langchain_core.runnables import RunnableBranch

def route_by_complexity(input_dict):
    if len(input_dict["question"]) < 50:
        return "simple"
    else:
        return "complex"

simple_llm = ChatOpenAI(model="gpt-4o-mini")  # $0.15/1M tokens
complex_llm = ChatOpenAI(model="gpt-4o")      # $2.50/1M tokens

router = RunnableBranch(
    (lambda x: route_by_complexity(x) == "simple", simple_llm),
    complex_llm  # default
)

# 简单问题走 gpt-4o-mini
result1 = router.invoke({"question": "What is 2+2?"})

# 复杂问题走 gpt-4o
result2 = router.invoke({
    "question": "Analyze the geopolitical implications of AI regulation in the EU compared to the US regulatory framework..."
})
```

**成本节省估算：** 若 70% 请求是简单任务，切换到 gpt-4o-mini 可节省约 **85% 成本**。

### 30.4.3 输出长度限制

通过 `max_tokens` 限制生成长度，避免意外高成本：

```python
llm = ChatOpenAI(model="gpt-4o", max_tokens=100)  # 最多生成 100 tokens
result = llm.invoke("Write a long essay about AI...")
# 生成会在 100 tokens 处截断
```

---

## 30.5 重试机制：应对瞬时故障

网络波动、API 限流、服务暂时不可用时，自动重试可提高成功率。

### 30.5.1 简单重试装饰器

```python
from langchain_core.runnables import RunnableRetry

llm_with_retry = llm.with_retry(
    stop_after_attempt=3,  # 最多重试 3 次
    wait_exponential_multiplier=1,  # 指数退避：1s, 2s, 4s
    wait_exponential_max=10  # 最大等待 10s
)

result = llm_with_retry.invoke("Hello")
# 若失败，自动重试最多 3 次，间隔递增
```

### 30.5.2 细粒度重试控制

使用 `tenacity` 库自定义重试策略：

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(RateLimitError),
    before_sleep=lambda retry_state: print(f"Rate limited, retrying in {retry_state.next_action.sleep}s...")
)
def call_llm_with_retry():
    return llm.invoke("Test")

result = call_llm_with_retry()
```

**退避策略说明：**
重试间隔按指数增长（1s → 2s → 4s → 8s），避免频繁请求加剧服务压力。

### 30.5.3 重试最佳实践

- **区分错误类型：** 只重试瞬时错误（网络超时、429 限流），不重试永久错误（401 认证失败、400 请求格式错误）
- **设置最大重试次数：** 避免无限重试耗尽资源
- **记录重试日志：** 监控重试频率，发现系统性问题

---

## 30.6 降级策略：保证服务可用性

当主服务不可用时，自动切换到备用方案（fallback）。

### 30.6.1 Fallback Chain

```python
from langchain_core.runnables import RunnableWithFallbacks

primary_llm = ChatOpenAI(model="gpt-4o")  # 主模型
fallback_llm = ChatOpenAI(model="gpt-4o-mini")  # 备用模型（更便宜/更快）

llm_with_fallback = primary_llm.with_fallbacks([fallback_llm])

result = llm_with_fallback.invoke("Hello")
# 若 gpt-4o 失败，自动切换到 gpt-4o-mini
```

### 30.6.2 多级降级

```python
primary = ChatOpenAI(model="gpt-4o", timeout=5)
fallback1 = ChatOpenAI(model="gpt-4o-mini", timeout=10)
fallback2 = ChatOpenAI(model="gpt-3.5-turbo", timeout=15)

chain = primary.with_fallbacks([fallback1, fallback2])

# 降级顺序: gpt-4o -> gpt-4o-mini -> gpt-3.5-turbo
```

### 30.6.3 缓存降级

当 LLM 完全不可用时,返回缓存结果或默认响应：

```python
from langchain_core.runnables import RunnableLambda

def cached_or_default(input_dict):
    # 尝试从缓存读取
    cached = check_cache(input_dict["question"])
    if cached:
        return cached
    return "Service temporarily unavailable. Please try again later."

fallback_fn = RunnableLambda(cached_or_default)

chain_with_cache_fallback = llm.with_fallbacks([fallback_fn])
```

**降级策略选择：**
根据服务可用性要求选择合适的降级方案：主模型 → 备用模型 → 缓存 → 默认响应。

---

## 30.7 监控与可观测性

生产环境需实时监控性能指标、错误率、成本等。

### 30.7.1 LangSmith 集成监控

启用 LangSmith 后，所有调用自动追踪：

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"

# 所有 invoke/batch/stream 自动上报到 LangSmith
result = chain.invoke({"question": "..."})
```

在 LangSmith Dashboard 查看：
- 延迟分布（P50/P95/P99）
- 成功率 / 错误率
- Token 使用量与成本
- 调用链可视化

### 30.7.2 自定义 Callback 监控

```python
from langchain.callbacks.base import BaseCallbackHandler

class MetricsCallback(BaseCallbackHandler):
    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.call_count += 1
    
    def on_llm_end(self, response, **kwargs):
        # 假设从 response.llm_output 读取 token 信息
        usage = response.llm_output.get("token_usage", {})
        tokens = usage.get("total_tokens", 0)
        self.total_tokens += tokens
        self.total_cost += tokens * 0.000002  # 假设 $0.002/1K tokens
    
    def summary(self):
        return {
            "calls": self.call_count,
            "tokens": self.total_tokens,
            "cost_usd": self.total_cost
        }

metrics = MetricsCallback()
result = llm.invoke("Test", callbacks=[metrics])
print(metrics.summary())
```

**预期输出：**
```python
{'calls': 1, 'tokens': 15, 'cost_usd': 0.00003}
```

### 30.7.3 Prometheus + Grafana

将指标暴露为 Prometheus 格式：

```python
from prometheus_client import Counter, Histogram, start_http_server

llm_calls_total = Counter('llm_calls_total', 'Total LLM calls', ['model', 'status'])
llm_latency = Histogram('llm_latency_seconds', 'LLM call latency', ['model'])

class PrometheusCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.start_time = time.time()
    
    def on_llm_end(self, response, **kwargs):
        latency = time.time() - self.start_time
        llm_latency.labels(model="gpt-4o-mini").observe(latency)
        llm_calls_total.labels(model="gpt-4o-mini", status="success").inc()
    
    def on_llm_error(self, error, **kwargs):
        llm_calls_total.labels(model="gpt-4o-mini", status="error").inc()

# 启动 Prometheus HTTP 服务器
start_http_server(8000)

# 使用 callback
result = llm.invoke("Test", callbacks=[PrometheusCallback()])
```

在 Grafana 中创建 Dashboard 可视化：
- 请求 QPS（Queries Per Second）
- 延迟分位数（P50/P95/P99）
- 错误率趋势

---

## 30.8 高级优化模式

### 30.8.1 请求去重

对于短时间内的重复请求（如用户连续点击），可在缓存层之上增加去重逻辑：

```python
import hashlib
import time
from threading import Lock

class RequestDeduplicator:
    def __init__(self, ttl=5):
        self.pending = {}
        self.lock = Lock()
        self.ttl = ttl
    
    def deduplicate(self, request_key, fn):
        request_hash = hashlib.md5(request_key.encode()).hexdigest()
        
        with self.lock:
            # 若请求正在处理中，等待结果
            if request_hash in self.pending:
                future = self.pending[request_hash]
                return future.result()
            
            # 创建新任务
            from concurrent.futures import Future
            future = Future()
            self.pending[request_hash] = future
        
        try:
            result = fn()
            future.set_result(result)
            return result
        finally:
            with self.lock:
                del self.pending[request_hash]

dedup = RequestDeduplicator()

def expensive_call():
    return llm.invoke("What is AI?")

# 多个并发请求只执行一次
result = dedup.deduplicate("ai_question", expensive_call)
```

### 30.8.2 预热缓存

对于高频问题，启动时预加载缓存：

```python
common_questions = [
    "What is Python?",
    "How to install LangChain?",
    "What is RAG?"
]

async def warmup_cache():
    tasks = [llm.ainvoke(q) for q in common_questions]
    await asyncio.gather(*tasks)
    print("Cache warmed up!")

# 在服务启动时调用
asyncio.run(warmup_cache())
```

### 30.8.3 压缩提示（Prompt Compression）

对于超长提示，使用 LongLLMLingua 等工具压缩：

```python
from llmlingua import PromptCompressor

compressor = PromptCompressor()

long_prompt = "..." * 1000  # 假设非常长

compressed = compressor.compress_prompt(
    long_prompt,
    rate=0.5,  # 压缩到 50% 长度
    force_tokens=['important', 'keyword']  # 保留关键词
)

result = llm.invoke(compressed["compressed_prompt"])
```

**效果：** Token 减少 50%，成本减半，延迟降低，但可能损失部分信息。

---

## 30.9 案例研究：优化前后对比

### 优化前系统

- **架构：** 同步调用 gpt-4o，无缓存，无重试
- **性能：** QPS 5，P95 延迟 3.2s，成本 $50/day

### 优化后系统

实施以下优化：
1. **Redis 缓存**（命中率 60%）
2. **模型路由**（70% 请求用 gpt-4o-mini）
3. **批处理**（每批 10 个请求）
4. **异步并发**（单进程处理 50 并发）
5. **重试 + 降级**（gpt-4o -> gpt-4o-mini -> cached response）

### 优化结果

| 指标 | 优化前 | 优化后 | 改善 |
|------|-------|-------|------|
| QPS | 5 | 50 | +900% |
| P95 延迟 | 3.2s | 0.8s | -75% |
| 成本/天 | $50 | $8 | -84% |
| 缓存命中率 | 0% | 60% | +60% |
| 可用性 | 99.5% | 99.95% | +0.45% |

**实施步骤详解：**
1. 启用 Redis 缓存，设置合理 TTL
2. 实施模型路由，简单任务使用 gpt-4o-mini
3. 批处理高频请求，每批 10-20 个
4. 启用异步处理，提升并发能力
5. 配置重试与降级策略

---

## 30.10 成本优化仪表盘

<div data-component="CostOptimizationDashboard"></div>

---

## 30.11 可靠性决策树

<div data-component="ReliabilityDecisionTree"></div>

---

## 30.12 总结

性能优化与可靠性工程是 LangChain 生产化的最后一公里。关键要点：

1. **缓存优先：** 内存 < SQLite < Redis < 语义缓存，根据场景选择
2. **批处理 + 异步：** 大幅提升吞吐量，降低延迟
3. **成本控制：** Token 计数、模型路由、输出限制
4. **重试 + 降级：** 指数退避、多级 fallback、缓存兜底
5. **可观测性：** LangSmith / Prometheus / 自定义 Callback 全方位监控
6. **持续优化：** A/B 测试不同策略，数据驱动决策

通过系统性应用这些模式，可将原型级 LangChain 应用改造为高性能、高可用、低成本的生产系统。

---

**扩展阅读：**
- [LangChain Caching Documentation](https://python.langchain.com/docs/modules/model_io/llms/llm_caching)
- [OpenAI Rate Limits](https://platform.openai.com/docs/guides/rate-limits)
- [Tenacity Retry Library](https://tenacity.readthedocs.io/)
- [Prometheus Python Client](https://github.com/prometheus/client_python)
- [LLMLingua: Prompt Compression](https://github.com/microsoft/LLMLingua)

**思考题：**
1. 语义缓存在什么场景下比精确缓存更优？存在哪些风险？
2. 批处理与异步的性能提升原理有何不同？能否结合使用？
3. 设计一个支持动态调整重试策略的系统（根据实时错误率）
4. 如何评估降级策略的业务影响？（如 gpt-4o -> gpt-4o-mini 后答案质量下降多少）
5. 成本与延迟的权衡：在什么情况下应优先考虑成本？什么情况下优先延迟？

（完）

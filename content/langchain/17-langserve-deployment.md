# Chapter 17: LangServe 部署与生产化

## 本章概览

在完成 LLM 应用的开发和优化后，下一步是将其部署到生产环境，对外提供稳定、高效的 API 服务。**LangServe** 是 LangChain 官方提供的部署框架，基于 **FastAPI**，专为 LangChain 应用设计，支持：

- **一键部署** LCEL 链和 LangGraph 图为 REST API
- **流式响应** 与批处理
- **Playground** 交互式测试界面
- **OpenAPI 文档** 自动生成
- **生产级特性**：并发控制、错误处理、监控集成

本章将深入探讨：
- LangServe 核心概念与架构
- 部署单链与复杂应用
- 流式输出与批处理优化
- 生产环境最佳实践
- 与 Docker、Kubernetes 集成

---

## 17.1 为什么需要 LangServe？

### 17.1.1 从开发到生产的挑战

在 Jupyter Notebook 或本地脚本中开发的 LLM 应用，部署到生产环境时面临诸多挑战：

| 挑战 | 开发环境 | 生产环境需求 | LangServe 解决方案 |
|------|----------|--------------|-------------------|
| **API 接口** | 函数调用 | RESTful API | 自动生成 `/invoke`、`/stream`、`/batch` 端点 |
| **并发处理** | 单线程 | 高并发请求 | FastAPI 异步处理 + uvicorn |
| **错误处理** | 抛出异常 | 优雅降级 + 错误日志 | 统一异常捕获与 HTTP 状态码映射 |
| **监控与日志** | print() | 结构化日志 + Metrics | 集成 LangSmith Tracing |
| **文档** | 手动编写 | 自动生成 | OpenAPI Spec + Swagger UI |
| **测试界面** | 无 | 方便调试 | 内置 Playground |

### 17.1.2 LangServe 核心特性

<Callout type="success">
**LangServe 的核心价值**

1. **零配置部署**：`add_routes(app, chain)` 一行代码完成部署
2. **自动端点生成**：
   - `/invoke` - 单次同步调用
   - `/stream` - 流式输出（SSE）
   - `/batch` - 批量处理
   - `/stream_log` - 流式日志（调试用）
3. **类型安全**：基于 Pydantic 的请求/响应验证
4. **Playground**：交互式 Web UI 测试
5. **生产就绪**：与 Docker、K8s、Prometheus 无缝集成
</Callout>

### 17.1.3 LangServe vs 其他部署方案

| 特性 | LangServe | FastAPI 手动 | BentoML | Ray Serve |
|------|-----------|--------------|---------|-----------|
| **LangChain 原生** | ✅ 完美集成 | ❌ 需手动适配 | ⚠️ 部分支持 | ⚠️ 部分支持 |
| **流式输出** | ✅ 原生支持 | ⚠️ 需手动实现 | ❌ 不支持 | ⚠️ 复杂 |
| **Playground** | ✅ 自动生成 | ❌ 无 | ✅ 有 | ❌ 无 |
| **部署复杂度** | ⭐ 低 | ⭐⭐ 中 | ⭐⭐⭐ 高 | ⭐⭐⭐ 高 |
| **扩展性** | ⭐⭐⭐ 良好 | ⭐⭐⭐⭐ 优秀 | ⭐⭐⭐⭐ 优秀 | ⭐⭐⭐⭐⭐ 极好 |

---

## 17.2 快速上手：第一个 LangServe 应用

### 17.2.1 环境配置

```bash
# 安装依赖
pip install "langserve[all]" langchain-openai

# 或仅安装核心
pip install langserve fastapi uvicorn[standard]
```

### 17.2.2 最简单的 LangServe 应用

创建 `app.py`：

```python
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes

# 1. 定义链
prompt = ChatPromptTemplate.from_template("将以下文本翻译为{language}：\n\n{text}")
model = ChatOpenAI(model="gpt-3.5-turbo")
chain = prompt | model | StrOutputParser()

# 2. 创建 FastAPI 应用
app = FastAPI(
    title="翻译服务 API",
    version="1.0",
    description="基于 LangServe 的翻译服务"
)

# 3. 添加 LangServe 路由
add_routes(
    app,
    chain,
    path="/translate"  # API 路径前缀
)

# 4. 启动服务
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 17.2.3 运行与测试

#### 启动服务

```bash
python app.py
```

**输出：**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345]
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

#### 访问 Playground

打开浏览器访问 `http://localhost:8000/translate/playground`：

<div data-component="LangServePlayground"></div>

#### API 调用示例

**1. 使用 cURL 调用 `/invoke`**

```bash
curl -X POST "http://localhost:8000/translate/invoke" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "language": "法语",
      "text": "Hello, how are you?"
    }
  }'
```

**响应：**
```json
{
  "output": "Bonjour, comment allez-vous?",
  "metadata": {
    "run_id": "abc123...",
    "feedback_key": null
  }
}
```

**2. 使用 Python 客户端**

```python
from langserve import RemoteRunnable

# 连接到远程服务
remote_chain = RemoteRunnable("http://localhost:8000/translate")

# 调用（同步）
result = remote_chain.invoke({
    "language": "西班牙语",
    "text": "Good morning!"
})
print(result)  # "¡Buenos días!"

# 流式调用
for chunk in remote_chain.stream({
    "language": "日语",
    "text": "Thank you very much!"
}):
    print(chunk, end="", flush=True)
```

### 17.2.4 自动生成的端点

`add_routes()` 会自动生成以下端点：

| 端点 | 方法 | 用途 | 返回类型 |
|------|------|------|----------|
| `/translate/invoke` | POST | 同步单次调用 | JSON (完整结果) |
| `/translate/batch` | POST | 批量调用 | JSON Array |
| `/translate/stream` | POST | 流式输出 | SSE (Server-Sent Events) |
| `/translate/stream_log` | POST | 流式日志（调试） | SSE |
| `/translate/input_schema` | GET | 输入 JSON Schema | JSON |
| `/translate/output_schema` | GET | 输出 JSON Schema | JSON |
| `/translate/config_schema` | GET | 配置 Schema | JSON |
| `/translate/playground` | GET | 交互式测试界面 | HTML |

---

## 17.3 流式输出（Streaming）

### 17.3.1 为什么需要流式输出？

对于生成式任务（如聊天、文本生成），流式输出可以：
- **改善用户体验**：即时反馈，减少等待感
- **降低首字节延迟（TTFB）**：不需要等待完整生成
- **支持超长内容**：避免单次响应过大

### 17.3.2 流式输出实现

**服务端（自动支持）：**

LangServe 自动为所有链提供 `/stream` 端点。

**客户端调用：**

```python
from langserve import RemoteRunnable

chain = RemoteRunnable("http://localhost:8000/translate")

# 流式调用
for chunk in chain.stream({
    "language": "中文",
    "text": "LangServe makes it easy to deploy LangChain applications as production-ready APIs."
}):
    print(chunk, end="", flush=True)
```

**输出：**
```
LangServe 使
得将 LangChain
 应用部署为
生产就绪的
 API 变得简单。
```

### 17.3.3 流式输出的内部机制

LangServe 使用 **Server-Sent Events (SSE)** 协议：

```http
POST /translate/stream HTTP/1.1
Content-Type: application/json

{"input": {"language": "中文", "text": "Hello"}}

---

HTTP/1.1 200 OK
Content-Type: text/event-stream

data: "你"

data: "好"

data: ""

event: end
data: null
```

每个 `data:` 行包含一个增量输出（chunk）。

### 17.3.4 自定义流式行为

对于复杂链，可以控制哪些部分流式输出：

```python
from langchain_core.runnables import RunnablePassthrough

# 定义流式链
chain = (
    RunnablePassthrough.assign(
        # 流式输出 LLM 部分
        translated=prompt | model.stream() | StrOutputParser()
    )
)

add_routes(app, chain, path="/translate_with_context")
```

---

## 17.4 批处理（Batching）

### 17.4.1 批处理优势

批处理适用于：
- **离线任务**：批量翻译、批量摘要
- **吞吐量优化**：减少网络往返次数
- **成本节约**：某些 LLM 提供商对批量调用有折扣

### 17.4.2 批处理 API 使用

**客户端调用：**

```python
from langserve import RemoteRunnable

chain = RemoteRunnable("http://localhost:8000/translate")

# 批量调用
inputs = [
    {"language": "法语", "text": "Hello"},
    {"language": "西班牙语", "text": "Goodbye"},
    {"language": "德语", "text": "Thank you"}
]

results = chain.batch(inputs)
for result in results:
    print(result)
```

**输出：**
```
Bonjour
Adiós
Danke
```

### 17.4.3 批处理配置

控制批处理并发度：

```python
from langserve import add_routes

add_routes(
    app,
    chain,
    path="/translate",
    config_keys=["max_concurrency"]  # 允许客户端配置
)
```

**客户端使用：**

```python
results = chain.batch(
    inputs,
    config={"max_concurrency": 5}  # 最多同时处理 5 个请求
)
```

---

## 17.5 复杂应用部署

### 17.5.1 部署多个链

同一个 FastAPI 应用可以部署多个链：

```python
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes

app = FastAPI(title="多功能 AI 服务")

# 链 1：翻译
translate_chain = (
    ChatPromptTemplate.from_template("翻译为{language}：{text}") 
    | ChatOpenAI() 
)
add_routes(app, translate_chain, path="/translate")

# 链 2：摘要
summarize_chain = (
    ChatPromptTemplate.from_template("总结以下文本：\n\n{text}") 
    | ChatOpenAI() 
)
add_routes(app, summarize_chain, path="/summarize")

# 链 3：问答
qa_chain = (
    ChatPromptTemplate.from_template("回答问题：{question}\n\n上下文：{context}") 
    | ChatOpenAI() 
)
add_routes(app, qa_chain, path="/qa")
```

### 17.5.2 部署 LangGraph 应用

LangGraph 的 `StateGraph` 可以直接部署：

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langserve import add_routes

# 定义状态
class AgentState(TypedDict):
    query: str
    result: str
    iteration: int

# 定义图
def agent_node(state: AgentState):
    # Agent 逻辑
    result = llm.invoke(state["query"])
    return {"result": result, "iteration": state["iteration"] + 1}

def should_continue(state: AgentState):
    return "end" if state["iteration"] > 3 else "agent"

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_conditional_edges("agent", should_continue, {"agent": "agent", "end": END})
graph.set_entry_point("agent")

compiled_graph = graph.compile()

# 部署图
add_routes(app, compiled_graph, path="/agent")
```

### 17.5.3 自定义输入/输出类型

使用 Pydantic 模型定义类型：

```python
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda

# 输入类型
class TranslateRequest(BaseModel):
    text: str = Field(description="待翻译文本")
    source_lang: str = Field(default="auto", description="源语言")
    target_lang: str = Field(description="目标语言")

# 输出类型
class TranslateResponse(BaseModel):
    translated_text: str
    detected_lang: str
    confidence: float

# 定义链
def translate_func(input: TranslateRequest) -> TranslateResponse:
    # 翻译逻辑
    result = llm.invoke(f"从{input.source_lang}翻译到{input.target_lang}：{input.text}")
    return TranslateResponse(
        translated_text=result,
        detected_lang="zh",
        confidence=0.95
    )

chain = RunnableLambda(translate_func)

add_routes(
    app,
    chain.with_types(input_type=TranslateRequest, output_type=TranslateResponse),
    path="/translate_v2"
)
```

现在 `/translate_v2/input_schema` 会返回完整的 Pydantic Schema，Playground 会自动渲染表单。

---

## 17.6 生产环境配置

### 17.6.1 错误处理与日志

**统一错误处理：**

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": str(exc)}
    )

# 添加链路由
add_routes(app, chain, path="/translate")
```

**结构化日志：**

```python
import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()

# 在链中记录日志
def log_input(input_data):
    logger.info("translate_request", input=input_data)
    return input_data

chain_with_logging = RunnableLambda(log_input) | chain
```

### 17.6.2 限流与并发控制

**使用 FastAPI 中间件：**

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import time
from collections import defaultdict

# 简单限流器
class RateLimiter:
    def __init__(self, max_requests=10, window=60):
        self.max_requests = max_requests
        self.window = window
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        # 清理过期记录
        self.requests[client_id] = [
            t for t in self.requests[client_id] 
            if now - t < self.window
        ]
        
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        self.requests[client_id].append(now)
        return True

rate_limiter = RateLimiter(max_requests=100, window=60)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    
    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded"}
        )
    
    response = await call_next(request)
    return response
```

**或使用 slowapi：**

```bash
pip install slowapi
```

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/translate/invoke")
@limiter.limit("10/minute")
async def limited_invoke(request: Request):
    # 处理逻辑
    pass
```

### 17.6.3 CORS 配置

允许跨域访问（用于前端集成）：

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],  # 生产环境指定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 17.6.4 健康检查端点

```python
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/ready")
async def readiness_check():
    # 检查依赖服务（数据库、模型加载等）
    try:
        # 简单测试 LLM 连接
        llm.invoke("test")
        return {"status": "ready"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "error": str(e)}
        )
```

<div data-component="DeploymentArchitecture"></div>

---

## 17.7 Docker 容器化

### 17.7.1 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**requirements.txt：**
```
langserve[all]==0.0.30
langchain==0.1.0
langchain-openai==0.0.2
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
```

### 17.7.2 构建与运行

```bash
# 构建镜像
docker build -t langserve-app:latest .

# 运行容器
docker run -d \
  --name langserve \
  -p 8000:8000 \
  -e OPENAI_API_KEY=sk-xxx \
  langserve-app:latest

# 查看日志
docker logs -f langserve

# 测试
curl http://localhost:8000/health
```

### 17.7.3 Docker Compose（多服务）

```yaml
version: '3.8'

services:
  langserve:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - langserve
    restart: unless-stopped

volumes:
  redis-data:
```

---

## 17.8 Kubernetes 部署

### 17.8.1 Deployment 配置

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langserve-deployment
  labels:
    app: langserve
spec:
  replicas: 3
  selector:
    matchLabels:
      app: langserve
  template:
    metadata:
      labels:
        app: langserve
    spec:
      containers:
      - name: langserve
        image: your-registry/langserve-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: langserve-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
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
          initialDelaySeconds: 10
          periodSeconds: 5
```

### 17.8.2 Service 配置

```yaml
apiVersion: v1
kind: Service
metadata:
  name: langserve-service
spec:
  selector:
    app: langserve
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

### 17.8.3 Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: langserve-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: langserve-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

<div data-component="KubernetesArchitecture"></div>

---

## 17.9 监控与可观测性

### 17.9.1 集成 Prometheus

**安装依赖：**

```bash
pip install prometheus-fastapi-instrumentator
```

**配置：**

```python
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

# 自动记录指标
Instrumentator().instrument(app).expose(app)

# 现在可以访问 /metrics 端点
```

**Prometheus 指标示例：**
- `http_requests_total` - 总请求数
- `http_request_duration_seconds` - 请求延迟
- `http_requests_in_progress` - 并发请求数

### 17.9.2 自定义指标

```python
from prometheus_client import Counter, Histogram

# 自定义计数器
translation_requests = Counter(
    'translation_requests_total',
    'Total translation requests',
    ['source_lang', 'target_lang']
)

# 自定义直方图（延迟分布）
translation_duration = Histogram(
    'translation_duration_seconds',
    'Translation duration',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# 在链中记录
@translation_duration.time()
def translate(input_data):
    translation_requests.labels(
        source_lang=input_data['source_lang'],
        target_lang=input_data['target_lang']
    ).inc()
    
    result = chain.invoke(input_data)
    return result
```

### 17.9.3 集成 LangSmith

LangSmith 自动追踪所有 LangServe 请求：

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_pt_xxx"
os.environ["LANGCHAIN_PROJECT"] = "production-translate-api"

# 所有请求自动记录到 LangSmith
add_routes(app, chain, path="/translate")
```

在 LangSmith UI 中可以看到：
- 每个请求的完整 Trace
- Token 消耗和成本
- 错误率和延迟分布
- 用户反馈

---

## 17.10 性能优化最佳实践

### 17.10.1 缓存策略

**使用 Redis 缓存结果：**

```python
import redis
import hashlib
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cached_chain(input_data):
    # 生成缓存键
    cache_key = hashlib.md5(
        json.dumps(input_data, sort_keys=True).encode()
    ).hexdigest()
    
    # 检查缓存
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # 执行链
    result = chain.invoke(input_data)
    
    # 存入缓存（24 小时过期）
    redis_client.setex(cache_key, 86400, json.dumps(result))
    
    return result

cached_runnable = RunnableLambda(cached_chain)
add_routes(app, cached_runnable, path="/translate_cached")
```

### 17.10.2 连接池优化

**复用 LLM 客户端：**

```python
from langchain_openai import ChatOpenAI
from functools import lru_cache

@lru_cache(maxsize=1)
def get_llm():
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        max_retries=3,
        request_timeout=30,
        # 使用连接池
        http_client=httpx.Client(
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        )
    )

llm = get_llm()
```

### 17.10.3 异步处理

对于 I/O 密集型操作，使用异步：

```python
from langchain_core.runnables import RunnableLambda

async def async_translate(input_data):
    # 异步调用 LLM
    result = await llm.ainvoke(input_data['text'])
    return result

async_chain = RunnableLambda(async_translate)
add_routes(app, async_chain, path="/translate_async")
```

---

## 17.11 安全性最佳实践

### 17.11.1 API 密钥认证

```python
from fastapi import Header, HTTPException

API_KEYS = {"key_12345", "key_67890"}

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key

# 添加依赖
@app.post("/translate/invoke", dependencies=[Depends(verify_api_key)])
async def protected_invoke():
    # 处理逻辑
    pass
```

### 17.11.2 输入验证与清理

```python
from pydantic import BaseModel, validator

class TranslateInput(BaseModel):
    text: str
    language: str
    
    @validator('text')
    def validate_text(cls, v):
        if len(v) > 5000:
            raise ValueError("Text too long (max 5000 characters)")
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()
    
    @validator('language')
    def validate_language(cls, v):
        allowed = {"中文", "英语", "法语", "西班牙语", "日语"}
        if v not in allowed:
            raise ValueError(f"Language must be one of {allowed}")
        return v
```

### 17.11.3 防止提示注入

```python
def sanitize_input(user_input: str) -> str:
    """移除可能的提示注入尝试"""
    # 移除常见注入模式
    dangerous_patterns = [
        "ignore previous instructions",
        "forget everything above",
        "system:",
        "assistant:"
    ]
    
    for pattern in dangerous_patterns:
        user_input = user_input.replace(pattern, "")
    
    return user_input.strip()

chain = (
    RunnableLambda(lambda x: {"text": sanitize_input(x["text"])})
    | prompt
    | llm
)
```

---

## 17.12 实战案例：企业级翻译 API

### 17.12.1 完整代码

```python
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from prometheus_fastapi_instrumentator import Instrumentator
import redis
import hashlib
import json
import logging

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis 缓存
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

# 输入模型
class TranslateRequest(BaseModel):
    text: str = Field(..., max_length=5000)
    source_lang: str = Field(default="auto")
    target_lang: str = Field(...)
    
    @validator('text')
    def clean_text(cls, v):
        return v.strip()

# 输出模型
class TranslateResponse(BaseModel):
    translated_text: str
    source_lang: str
    target_lang: str
    cached: bool = False

# 创建应用
app = FastAPI(
    title="企业翻译 API",
    version="2.0.0",
    description="高性能、可扩展的翻译服务"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Prometheus 监控
Instrumentator().instrument(app).expose(app)

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

# 提示
prompt = ChatPromptTemplate.from_template(
    "将以下{source_lang}文本翻译为{target_lang}，保持原意和语气：\n\n{text}"
)

# 缓存层
def cached_translate(input_data: dict) -> TranslateResponse:
    cache_key = f"translate:{hashlib.md5(json.dumps(input_data, sort_keys=True).encode()).hexdigest()}"
    
    # 检查缓存
    cached = redis_client.get(cache_key)
    if cached:
        logger.info("Cache hit", extra={"key": cache_key})
        result = json.loads(cached)
        return TranslateResponse(**result, cached=True)
    
    # 执行翻译
    logger.info("Cache miss, invoking LLM", extra={"input": input_data})
    result_text = (prompt | llm | StrOutputParser()).invoke(input_data)
    
    result = {
        "translated_text": result_text,
        "source_lang": input_data['source_lang'],
        "target_lang": input_data['target_lang']
    }
    
    # 存入缓存（1 小时）
    redis_client.setex(cache_key, 3600, json.dumps(result))
    
    return TranslateResponse(**result)

# API 密钥验证
async def verify_api_key(x_api_key: str = Header(...)):
    # 这里应该查询数据库
    if x_api_key != "demo_key_12345":
        raise HTTPException(status_code=403, detail="Invalid API key")

# 端点
@app.post("/translate", response_model=TranslateResponse, dependencies=[Depends(verify_api_key)])
async def translate_endpoint(request: TranslateRequest):
    try:
        return cached_translate(request.dict())
    except Exception as e:
        logger.error("Translation failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/stats")
async def stats():
    return {
        "cache_size": redis_client.dbsize(),
        "cache_hits": redis_client.info()['keyspace_hits'],
        "cache_misses": redis_client.info()['keyspace_misses']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
```

### 17.12.2 部署配置

**docker-compose.yml：**

```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - redis
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 2G

  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana

volumes:
  redis-data:
  prometheus-data:
  grafana-data:
```

---

## 17.13 常见问题与陷阱

### 17.13.1 流式输出中断

**问题**：客户端在流式输出过程中断开连接。

**解决方案：**

```python
from fastapi import Request

async def stream_with_disconnect_check(request: Request):
    async for chunk in chain.astream(input_data):
        if await request.is_disconnected():
            logger.info("Client disconnected, stopping stream")
            break
        yield chunk
```

### 17.13.2 内存泄漏

**问题**：长时间运行后内存占用持续增长。

**解决方案：**
- 使用 `uvicorn --workers 4` 多进程模式
- 定期重启 Worker（`uvicorn --timeout-keep-alive 5`）
- 检查缓存是否设置了过期时间

### 17.13.3 模型加载慢

**问题**：首次请求延迟高（模型需要加载）。

**解决方案：**

```python
@app.on_event("startup")
async def warmup():
    logger.info("Warming up model...")
    llm.invoke("test")
    logger.info("Model ready")
```

---

## 17.14 扩展阅读与资源

### 官方文档
- [LangServe 官方文档](https://python.langchain.com/docs/langserve)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [LangServe GitHub](https://github.com/langchain-ai/langserve)

### 示例项目
- [LangServe Templates](https://github.com/langchain-ai/langserve/tree/main/examples)
- [Production Deployment Guide](https://python.langchain.com/docs/langserve#deployment)

### 最佳实践
- [API 设计最佳实践](https://github.com/microsoft/api-guidelines)
- [FastAPI 生产部署](https://fastapi.tiangolo.com/deployment/)

---

## 本章小结

本章深入探讨了 LangServe 的核心功能和生产部署：

✅ **快速部署**：`add_routes()` 一键部署 LCEL 链和 LangGraph 图  
✅ **流式输出**：基于 SSE 的实时流式响应，改善用户体验  
✅ **批处理**：高吞吐量批量调用，优化成本和性能  
✅ **生产特性**：错误处理、限流、监控、健康检查  
✅ **容器化**：Docker + Kubernetes 部署，支持水平扩展  
✅ **可观测性**：集成 Prometheus、LangSmith 实现全链路追踪  

**关键要点：**
1. LangServe 专为 LangChain 应用设计，零配置即可部署
2. 流式输出和批处理是生产环境的关键特性
3. 缓存、连接池、异步是性能优化的三大法宝
4. 监控、日志、健康检查是稳定性的基石
5. 安全性（API 密钥、输入验证、提示注入防护）不可忽视

下一章将学习更多高级话题，包括性能优化、可靠性工程和生态集成。

---

**思考题：**
1. 流式输出 vs 批处理，各自适用于什么场景？
2. 如何设计一个支持每秒 1000 次请求的 LangServe 应用？
3. 在 Kubernetes 中如何实现金丝雀发布（Canary Deployment）？

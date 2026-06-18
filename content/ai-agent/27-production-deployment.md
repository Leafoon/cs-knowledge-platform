---
title: "第27章：生产级部署 — 从原型到上线"
description: "掌握 Agent 生产部署全流程：Docker 容器化、Kubernetes 编排、API 网关、缓存策略与成本优化。"
date: "2026-06-11"
---


下面的交互式演示展示了生产部署的检查清单：

<div data-component="ProductionDeploymentChecklist"></div>

# 第27章：生产级部署 — 从原型到上线

将 Agent 从 Jupyter Notebook 搬到生产环境，需要解决容器化、扩展性、可靠性、安全性等一系列挑战。本章系统讲解 Agent 生产部署的完整流程。

## 从原型到生产：为什么需要专门的部署策略？

在 Jupyter Notebook 中运行 Agent 和在生产环境中服务用户是完全不同的两件事。原型阶段通常只需要考虑"能不能跑"，而生产环境需要考虑：

**1. 可靠性（Reliability）**
- 系统必须 24/7 可用
- 单个组件故障不能导致整体崩溃
- 需要自动恢复机制

**2. 可扩展性（Scalability）**
- 能够处理突发的流量高峰
- 可以水平扩展以支持更多用户
- 资源使用要高效

**3. 安全性（Security）**
- 保护用户数据和 API 密钥
- 防止恶意攻击
- 符合合规要求

**4. 可观测性（Observability）**
- 能够监控系统状态
- 快速定位和解决问题
- 支持性能优化

**5. 成本控制（Cost Control）**
- 优化资源使用
- 控制 LLM API 调用成本
- 支持预算管理

Docker 和 Kubernetes 是解决这些问题的核心技术。Docker 提供了环境一致性和隔离性，Kubernetes 提供了自动化部署、扩缩容和故障恢复能力。

---

## 27.1 Docker 容器化

### 为什么选择 Docker？

Docker 是一个容器化平台，它允许我们将应用程序及其依赖打包成一个轻量级、可移植的容器。对于 Agent 系统，Docker 提供了以下优势：

**环境一致性**：在开发、测试、生产环境中运行完全相同的代码，避免"在我机器上能跑"的问题。

**依赖隔离**：每个容器有自己独立的依赖环境，避免版本冲突。

**快速部署**：容器可以在几秒钟内启动，支持快速迭代。

**资源效率**：比虚拟机更轻量，可以在同一台机器上运行多个容器。

### 27.1.1 多阶段构建

多阶段构建是 Docker 的一个高级特性，它允许我们在一个 Dockerfile 中使用多个基础镜像。这对于 Python 应用特别有用，因为我们可以：

1. 在构建阶段安装编译依赖
2. 在运行阶段只复制需要的文件
3. 最终镜像更小、更安全

```dockerfile
# 构建阶段
FROM python:3.11-slim as builder

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 运行阶段
FROM python:3.11-slim

WORKDIR /app

# 复制构建阶段的依赖
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# 复制应用代码
COPY . .

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**关键点解释**：
- `HEALTHCHECK`：告诉 Docker 如何检查容器是否健康运行
- `PYTHONUNBUFFERED=1`：确保 Python 输出立即刷新，便于日志收集
- `--workers 4`：启动 4 个 worker 进程，提高并发处理能力

```dockerfile
# 构建阶段
FROM python:3.11-slim as builder

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# 运行阶段
FROM python:3.11-slim

WORKDIR /app

# 复制构建阶段的依赖
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# 复制应用代码
COPY . .

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### 27.1.2 Docker Compose 配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Agent 服务
  agent:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://postgres:password@db:5432/agent_db
    depends_on:
      - redis
      - db
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
  
  # Redis 缓存
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
  
  # PostgreSQL 数据库
  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=agent_db
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  
  # Nginx 反向代理
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./certs:/etc/nginx/certs
    depends_on:
      - agent

volumes:
  redis_data:
  postgres_data:
```

### 27.1.3 镜像优化

```python
# requirements.txt
# 使用精确版本锁定
fastapi==0.109.0
uvicorn[standard]==0.27.0
langchain==0.1.4
openai==1.12.0
redis==5.0.1
sqlalchemy==2.0.25
pydantic==2.6.0

# 生产环境专用依赖
gunicorn==21.2.0
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
```

---

## 27.2 Kubernetes 部署

### 27.2.1 Deployment 配置

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-agent
  labels:
    app: ai-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-agent
  template:
    metadata:
      labels:
        app: ai-agent
    spec:
      containers:
      - name: agent
        image: ai-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: openai-api-key
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: agent-config
              key: redis-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
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
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 27.2.2 Service 和 Ingress

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ai-agent-service
spec:
  selector:
    app: ai-agent
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-agent-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - agent.example.com
    secretName: agent-tls
  rules:
  - host: agent.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ai-agent-service
            port:
              number: 80
```

### 27.2.3 HPA 自动扩缩容

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-agent
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
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: 1000
```

---

## 27.3 API 网关设计

### 27.3.1 Nginx 配置

```nginx
# nginx.conf
upstream agent_backend {
    least_conn;
    server agent1:8000 weight=5;
    server agent2:8000 weight=3;
    server agent3:8000 backup;
}

server {
    listen 80;
    server_name agent.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name agent.example.com;

    ssl_certificate /etc/nginx/certs/fullchain.pem;
    ssl_certificate_key /etc/nginx/certs/privkey.pem;

    # 速率限制
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    # 代理配置
    location / {
        proxy_pass http://agent_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 超时配置
        proxy_connect_timeout 30s;
        proxy_send_timeout 60s;
        proxy_read_timeout 120s;
        
        # 缓冲配置
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }

    # 健康检查
    location /health {
        proxy_pass http://agent_backend;
        access_log off;
    }
}
```

### 27.3.2 FastAPI 服务

```python
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import asyncio
import time
from contextlib import asynccontextmanager

# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化
    await init_db()
    await init_redis()
    yield
    # 关闭时清理
    await close_db()
    await close_redis()

app = FastAPI(
    title="AI Agent API",
    description="生产级 Agent 服务",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求/响应模型
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    user_id: str
    stream: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "帮我查询今天的天气",
                "user_id": "user_123",
                "stream": False
            }
        }

class ChatResponse(BaseModel):
    response: str
    session_id: str
    tools_used: List[str]
    tokens_used: int
    cost_usd: float
    latency_ms: float

# 依赖注入
async def get_current_user(request: Request):
    token = request.headers.get("Authorization")
    if not token:
        raise HTTPException(status_code=401, detail="Missing authorization token")
    # 验证 token
    user = await verify_token(token)
    return user

# 路由
@app.post("/agent/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user=Depends(get_current_user)
):
    start_time = time.time()
    
    try:
        # 检查速率限制
        await check_rate_limit(user.id)
        
        # 执行 Agent
        if request.stream:
            return StreamingResponse(
                stream_agent_response(request),
                media_type="text/event-stream"
            )
        
        result = await agent.run(
            query=request.message,
            session_id=request.session_id,
            user_id=user.id
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return ChatResponse(
            response=result["answer"],
            session_id=result["session_id"],
            tools_used=result["tools_used"],
            tokens_used=result["tokens"],
            cost_usd=result["cost"],
            latency_ms=latency_ms
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/agent/metrics")
async def get_metrics(user=Depends(get_current_user)):
    return await get_user_metrics(user.id)
```

---

## 27.4 缓存策略

### 27.4.1 多层缓存架构

```python
from typing import Optional, Any
import hashlib
import json
import asyncio
from datetime import timedelta

class MultiLevelCache:
    """多层缓存架构：L1 本地缓存 + L2 Redis 分布式缓存"""
    
    def __init__(self, redis_client, local_ttl=60, redis_ttl=3600):
        self.redis = redis_client
        self.local_cache = {}
        self.local_ttl = local_ttl
        self.redis_ttl = redis_ttl
    
    async def get(self, key: str) -> Optional[Any]:
        # L1: 本地缓存
        if key in self.local_cache:
            entry = self.local_cache[key]
            if time.time() < entry["expires"]:
                return entry["value"]
            else:
                del self.local_cache[key]
        
        # L2: Redis 缓存
        result = await self.redis.get(key)
        if result:
            value = json.loads(result)
            # 回填本地缓存
            self.local_cache[key] = {
                "value": value,
                "expires": time.time() + self.local_ttl
            }
            return value
        
        return None
    
    async def set(self, key: str, value: Any):
        # 写入 L2
        await self.redis.setex(key, self.redis_ttl, json.dumps(value))
        
        # 写入 L1
        self.local_cache[key] = {
            "value": value,
            "expires": time.time() + self.local_ttl
        }
    
    async def invalidate(self, key: str):
        await self.redis.delete(key)
        self.local_cache.pop(key, None)

class SemanticCache:
    """语义缓存：基于向量相似度的缓存"""
    
    def __init__(self, embedding_model, vector_store, similarity_threshold=0.95):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
    
    async def get_similar(self, query: str) -> Optional[dict]:
        # 生成查询向量
        query_embedding = await self.embedding_model.embed(query)
        
        # 搜索相似查询
        results = await self.vector_store.search(
            embedding=query_embedding,
            top_k=1,
            threshold=self.similarity_threshold
        )
        
        if results:
            return results[0]["metadata"]["response"]
        return None
    
    async def set(self, query: str, response: dict):
        # 生成查询向量
        query_embedding = await self.embedding_model.embed(query)
        
        # 存储到向量数据库
        await self.vector_store.upsert(
            embedding=query_embedding,
            metadata={
                "query": query,
                "response": response,
                "timestamp": time.time()
            }
        )
```

### 27.4.2 缓存策略选择

```python
class CacheStrategy:
    """缓存策略管理器"""
    
    @staticmethod
    def get_strategy(query_type: str) -> dict:
        strategies = {
            # 简单查询：精确匹配缓存
            "factual": {
                "type": "exact",
                "ttl": 3600,  # 1小时
                "invalidate_on": ["knowledge_update"]
            },
            
            # 复杂查询：语义缓存
            "complex": {
                "type": "semantic",
                "similarity_threshold": 0.92,
                "ttl": 1800,  # 30分钟
                "invalidate_on": ["user_feedback"]
            },
            
            # 代码生成：不缓存
            "code_generation": {
                "type": "none",
                "ttl": 0
            },
            
            # 对话历史：会话级缓存
            "conversation": {
                "type": "session",
                "ttl": 1800,  # 30分钟
                "max_turns": 10
            }
        }
        return strategies.get(query_type, strategies["factual"])
```

---

## 27.5 成本优化

### 27.5.1 模型路由

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict

class QueryComplexity(Enum):
    SIMPLE = "simple"      # 简单事实查询
    MODERATE = "moderate"  # 中等复杂度
    COMPLEX = "complex"    # 复杂推理任务

@dataclass
class ModelConfig:
    model_name: str
    cost_per_1k_input: float
    cost_per_1k_output: float
    max_tokens: int
    capabilities: list

class ModelRouter:
    """智能模型路由器"""
    
    def __init__(self):
        self.models = {
            QueryComplexity.SIMPLE: ModelConfig(
                model_name="gpt-4o-mini",
                cost_per_1k_input=0.00015,
                cost_per_1k_output=0.0006,
                max_tokens=4096,
                capabilities=["basic_qa", "summarization"]
            ),
            QueryComplexity.MODERATE: ModelConfig(
                model_name="gpt-4o",
                cost_per_1k_input=0.005,
                cost_per_1k_output=0.015,
                max_tokens=8192,
                capabilities=["reasoning", "code_generation", "analysis"]
            ),
            QueryComplexity.COMPLEX: ModelConfig(
                model_name="gpt-4o",
                cost_per_1k_input=0.005,
                cost_per_1k_output=0.015,
                max_tokens=16384,
                capabilities=["complex_reasoning", "multi_step", "creative"]
            )
        }
    
    async def classify_query(self, query: str) -> QueryComplexity:
        """使用小模型分类查询复杂度"""
        classification_prompt = f"""
        分类以下查询的复杂度（simple/moderate/complex）：
        
        查询：{query}
        
        简单：事实查询、简单问答
        中等：需要推理、分析、代码生成
        复杂：多步骤推理、创造性任务、复杂分析
        
        只输出一个词：simple、moderate 或 complex
        """
        
        # 使用轻量级模型分类
        response = await self.classifier_model.generate(classification_prompt)
        complexity = response.strip().lower()
        
        return QueryComplexity(complexity)
    
    async def route(self, query: str) -> ModelConfig:
        """路由到合适的模型"""
        complexity = await self.classify_query(query)
        return self.models[complexity]
    
    async def estimate_cost(self, query: str, model: ModelConfig) -> float:
        """估算成本"""
        input_tokens = len(query.split()) * 1.3  # 估算 token 数
        output_tokens = input_tokens * 2  # 假设输出是输入的 2 倍
        
        cost = (
            (input_tokens / 1000) * model.cost_per_1k_input +
            (output_tokens / 1000) * model.cost_per_1k_output
        )
        return cost
```

### 27.5.2 Prompt 优化

```python
class PromptOptimizer:
    """Prompt 优化器"""
    
    @staticmethod
    def compress_context(messages: list, max_tokens: int = 4000) -> list:
        """压缩上下文长度"""
        if sum(len(m["content"].split()) for m in messages) <= max_tokens:
            return messages
        
        # 保留系统消息和最近的消息
        system_messages = [m for m in messages if m["role"] == "system"]
        other_messages = [m for m in messages if m["role"] != "system"]
        
        # 使用摘要替换旧消息
        if len(other_messages) > 4:
            old_messages = other_messages[:-2]
            recent_messages = other_messages[-2:]
            
            summary = PromptOptimizer._summarize_messages(old_messages)
            compressed = system_messages + [{"role": "user", "content": summary}] + recent_messages
            return compressed
        
        return messages
    
    @staticmethod
    def _summarize_messages(messages: list) -> str:
        """使用小模型生成摘要"""
        conversation = "\n".join([
            f"{m['role']}: {m['content'][:200]}"
            for m in messages
        ])
        
        summary_prompt = f"""
        简要总结以下对话的关键信息：
        
        {conversation}
        
        总结：
        """
        
        # 使用小模型生成摘要
        return summary_prompt
    
    @staticmethod
    def deduplicate_tools(tools: list) -> list:
        """去重工具描述"""
        seen = set()
        unique_tools = []
        
        for tool in tools:
            tool_key = tool["function"]["name"]
            if tool_key not in seen:
                seen.add(tool_key)
                unique_tools.append(tool)
        
        return unique_tools
```

### 27.5.3 成本监控

```python
class CostTracker:
    """成本跟踪器"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def record_cost(
        self,
        user_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int
    ):
        """记录 API 调用成本"""
        cost = self._calculate_cost(model, input_tokens, output_tokens)
        
        # 记录到 Redis
        await self.redis.hincrbyfloat(f"cost:{user_id}:daily", cost)
        await self.redis.hincrbyfloat(f"cost:{user_id}:monthly", cost)
        
        # 设置过期时间
        await self.redis.expire(f"cost:{user_id}:daily", 86400)
        await self.redis.expire(f"cost:{user_id}:monthly", 2592000)
    
    async def check_budget(self, user_id: str, estimated_cost: float) -> bool:
        """检查预算"""
        daily_cost = float(await self.redis.hget(f"cost:{user_id}:daily", "total") or 0)
        monthly_cost = float(await self.redis.hget(f"cost:{user_id}:monthly", "total") or 0)
        
        daily_limit = 10.0  # 每日限额
        monthly_limit = 200.0  # 每月限额
        
        if daily_cost + estimated_cost > daily_limit:
            return False
        if monthly_cost + estimated_cost > monthly_limit:
            return False
        
        return True
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """计算成本"""
        pricing = {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "claude-3-opus": {"input": 0.015, "output": 0.075}
        }
        
        rates = pricing.get(model, {"input": 0.005, "output": 0.015})
        return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1000
```

---

## 27.6 可靠性保障

### 27.6.1 重试和熔断

```python
import asyncio
from typing import Callable, Any
from functools import wraps
import time

class CircuitBreaker:
    """熔断器"""
    
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise

def retry_with_backoff(max_retries=3, base_delay=1, max_delay=30):
    """指数退避重试装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        await asyncio.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator
```

### 27.6.2 优雅降级

```python
class GracefulDegradation:
    """优雅降级策略"""
    
    def __init__(self, primary_model, fallback_models):
        self.primary_model = primary_model
        self.fallback_models = fallback_models
        self.circuit_breaker = CircuitBreaker()
    
    async def execute(self, query: str) -> str:
        """执行查询，支持降级"""
        try:
            # 尝试主模型
            return await self.circuit_breaker.call(
                self.primary_model.generate, query
            )
        except Exception as e:
            print(f"Primary model failed: {e}")
            
            # 尝试备用模型
            for fallback_model in self.fallback_models:
                try:
                    return await fallback_model.generate(query)
                except Exception as e:
                    print(f"Fallback model {fallback_model.name} failed: {e}")
                    continue
            
            # 所有模型都失败，返回默认响应
            return "抱歉，当前服务繁忙，请稍后再试。"
```

---

## 27.7 安全考虑

### 27.7.1 认证和授权

```python
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta

security = HTTPBearer()

class AuthManager:
    """认证管理器"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.algorithm = "HS256"
    
    def create_token(self, user_id: str, expires_delta: timedelta = timedelta(hours=1)):
        """创建 JWT Token"""
        payload = {
            "sub": user_id,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + expires_delta
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> dict:
        """验证 Token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """获取当前用户"""
    auth = AuthManager(secret_key="your-secret-key")
    payload = auth.verify_token(credentials.credentials)
    return payload["sub"]
```

### 27.7.2 速率限制

```python
from fastapi import Request, HTTPException
from datetime import datetime, timedelta
import redis

class RateLimiter:
    """分布式速率限制器"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def check_rate_limit(
        self,
        user_id: str,
        limit: int = 100,
        window: int = 3600
    ) -> bool:
        """检查速率限制"""
        key = f"rate_limit:{user_id}:{datetime.utcnow().hour}"
        
        current = await self.redis.incr(key)
        if current == 1:
            await self.redis.expire(key, window)
        
        return current <= limit

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """速率限制中间件"""
    user_id = request.headers.get("X-User-ID")
    if user_id:
        limiter = RateLimiter(redis_client)
        if not await limiter.check_rate_limit(user_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    response = await call_next(request)
    return response
```

---

## 27.8 监控和告警

### 27.8.1 健康检查

```python
from fastapi import APIRouter
import psutil

router = APIRouter()

@router.get("/health")
async def health_check():
    """健康检查"""
    checks = {
        "status": "healthy",
        "timestamp": time.time(),
        "checks": {
            "memory": check_memory(),
            "cpu": check_cpu(),
            "disk": check_disk(),
            "database": await check_database(),
            "redis": await check_redis(),
            "llm": await check_llm_service()
        }
    }
    
    # 检查所有组件
    all_healthy = all(
        check["status"] == "healthy"
        for check in checks["checks"].values()
    )
    
    if not all_healthy:
        checks["status"] = "unhealthy"
    
    return checks

def check_memory():
    """检查内存使用"""
    memory = psutil.virtual_memory()
    status = "healthy" if memory.percent < 80 else "unhealthy"
    return {
        "status": status,
        "percent": memory.percent,
        "available": memory.available
    }

def check_cpu():
    """检查 CPU 使用"""
    cpu_percent = psutil.cpu_percent(interval=1)
    status = "healthy" if cpu_percent < 80 else "unhealthy"
    return {
        "status": status,
        "percent": cpu_percent
    }
```

### 27.8.2 性能指标

```python
from prometheus_client import Counter, Histogram, Gauge

# 请求指标
REQUEST_COUNT = Counter(
    'agent_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'agent_request_latency_seconds',
    'Request latency',
    ['method', 'endpoint']
)

# Agent 指标
AGENT_STEPS = Histogram(
    'agent_steps',
    'Number of steps per request',
    buckets=[1, 2, 3, 5, 10, 20]
)

LLM_CALLS = Counter(
    'agent_llm_calls_total',
    'Total LLM calls',
    ['model', 'status']
)

TOKEN_USAGE = Counter(
    'agent_tokens_total',
    'Total tokens used',
    ['model', 'type']
)

COST_USD = Counter(
    'agent_cost_usd_total',
    'Total cost in USD',
    ['model']
)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """指标收集中间件"""
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response
```

---

## 27.9 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| **Docker** | 多阶段构建、镜像优化、健康检查 |
| **Kubernetes** | Deployment、Service、HPA 自动扩缩容 |
| **API 网关** | Nginx 配置、速率限制、负载均衡 |
| **FastAPI** | 异步框架、依赖注入、中间件 |
| **缓存策略** | 多层缓存、语义缓存、缓存失效 |
| **成本优化** | 模型路由、Prompt 压缩、Token 监控 |
| **可靠性** | 重试、熔断、优雅降级 |
| **安全** | JWT 认证、速率限制、输入验证 |
| **监控** | 健康检查、性能指标、Prometheus 集成 |

---

## 27.10 高级部署策略

### 27.10.1 蓝绿部署

```python
from typing import Dict, Any, List
from dataclasses import dataclass
import time

@dataclass
class DeploymentConfig:
    """部署配置"""
    app_name: str
    image: str
    replicas: int
    port: int
    health_check_path: str = "/health"
    environment: str = "production"

class BlueGreenDeployer:
    """蓝绿部署器"""
    
    def __init__(self, k8s_client):
        self.k8s = k8s_client
        self.current_environment = "blue"
    
    async def deploy(
        self,
        config: DeploymentConfig,
        strategy: str = "blue_green"
    ) -> Dict[str, Any]:
        """执行部署"""
        # 确定目标环境
        target_env = "green" if self.current_environment == "blue" else "blue"
        
        print(f"开始 {strategy} 部署")
        print(f"当前环境: {self.current_environment}")
        print(f"目标环境: {target_env}")
        
        # 1. 部署到目标环境
        deploy_result = await self._deploy_to_environment(config, target_env)
        
        if not deploy_result["success"]:
            return deploy_result
        
        # 2. 等待健康检查通过
        health_ok = await self._wait_for_health(config.app_name, target_env)
        
        if not health_ok:
            await self._rollback(config.app_name, target_env)
            return {"success": False, "error": "健康检查失败"}
        
        # 3. 切换流量
        await self._switch_traffic(target_env)
        
        # 4. 验证部署
        verification = await self._verify_deployment(config.app_name, target_env)
        
        if verification["success"]:
            self.current_environment = target_env
            return {
                "success": True,
                "from_environment": self.current_environment,
                "to_environment": target_env,
                "verification": verification
            }
        else:
            await self._switch_traffic(self.current_environment)
            return {"success": False, "error": "部署验证失败"}
    
    async def _deploy_to_environment(self, config: DeploymentConfig, env: str) -> Dict:
        """部署到指定环境"""
        deployment_name = f"{config.app_name}-{env}"
        
        # 创建或更新部署
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": deployment_name,
                "labels": {"app": config.app_name, "environment": env}
            },
            "spec": {
                "replicas": config.replicas,
                "selector": {
                    "matchLabels": {"app": config.app_name, "environment": env}
                },
                "template": {
                    "metadata": {
                        "labels": {"app": config.app_name, "environment": env}
                    },
                    "spec": {
                        "containers": [{
                            "name": config.app_name,
                            "image": config.image,
                            "ports": [{"containerPort": config.port}],
                            "livenessProbe": {
                                "httpGet": {
                                    "path": config.health_check_path,
                                    "port": config.port
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            }
                        }]
                    }
                }
            }
        }
        
        try:
            # 这里应该调用 Kubernetes API
            # self.k8s.create_or_update_deployment(deployment)
            return {"success": True, "deployment": deployment_name}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _wait_for_health(self, app_name: str, env: str, timeout: int = 300) -> bool:
        """等待健康检查通过"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # 检查 Pod 状态
            # 这里应该调用 Kubernetes API 检查 Pod 健康状态
            # pod_status = self.k8s.get_pods(app_name, env)
            
            # 模拟健康检查
            await asyncio.sleep(5)
            
            # 如果所有 Pod 都就绪，返回成功
            if True:  # 实际应该检查 pod_status
                return True
        
        return False
    
    async def _switch_traffic(self, target_env: str):
        """切换流量"""
        # 更新 Service 指向目标环境
        service_patch = {
            "spec": {
                "selector": {
                    "environment": target_env
                }
            }
        }
        
        # self.k8s.patch_service(f"{self.app_name}-service", service_patch)
        print(f"流量已切换到 {target_env} 环境")
    
    async def _verify_deployment(self, app_name: str, env: str) -> Dict[str, Any]:
        """验证部署"""
        # 执行验证测试
        tests = [
            {"name": "健康检查", "passed": True},
            {"name": "API 响应", "passed": True},
            {"name": "功能测试", "passed": True}
        ]
        
        all_passed = all(test["passed"] for test in tests)
        
        return {
            "success": all_passed,
            "tests": tests
        }
    
    async def _rollback(self, app_name: str, env: str):
        """回滚部署"""
        # 删除失败的部署
        # self.k8s.delete_deployment(f"{app_name}-{env}")
        print(f"已回滚 {env} 环境的部署")
```

### 27.10.2 金丝雀发布

```python
class CanaryDeployer:
    """金丝雀发布器"""
    
    def __init__(self, k8s_client):
        self.k8s = k8s_client
        self.canary_config = {
            "initial_percentage": 5,
            "increment": 10,
            "interval": 300,  # 5 分钟
            "max_percentage": 100,
            "success_threshold": 0.99,
            "rollback_threshold": 0.01
        }
    
    async def deploy_canary(
        self,
        config: DeploymentConfig,
        canary_percentage: int = None
    ) -> Dict[str, Any]:
        """执行金丝雀发布"""
        percentage = canary_percentage or self.canary_config["initial_percentage"]
        
        print(f"开始金丝雀发布，初始流量比例: {percentage}%")
        
        # 1. 创建金丝雀部署
        await self._create_canary_deployment(config, percentage)
        
        # 2. 监控金丝雀
        monitoring_result = await self._monitor_canary(config.app_name)
        
        if not monitoring_result["healthy"]:
            await self._rollback_canary(config.app_name)
            return {"success": False, "error": "金丝雀健康检查失败"}
        
        # 3. 逐步增加流量
        current_percentage = percentage
        while current_percentage < self.canary_config["max_percentage"]:
            # 等待监控周期
            await asyncio.sleep(self.canary_config["interval"])
            
            # 增加流量
            current_percentage = min(
                current_percentage + self.canary_config["increment"],
                self.canary_config["max_percentage"]
            )
            
            print(f"增加金丝雀流量到 {current_percentage}%")
            await self._update_canary_percentage(config.app_name, current_percentage)
            
            # 检查健康状态
            monitoring_result = await self._monitor_canary(config.app_name)
            
            if not monitoring_result["healthy"]:
                await self._rollback_canary(config.app_name)
                return {
                    "success": False,
                    "error": "金丝雀健康检查失败",
                    "failed_at_percentage": current_percentage
                }
        
        # 4. 完成部署
        await self._finalize_canary(config.app_name)
        
        return {
            "success": True,
            "final_percentage": 100,
            "duration": time.time() - time.time()
        }
    
    async def _create_canary_deployment(self, config: DeploymentConfig, percentage: int):
        """创建金丝雀部署"""
        canary_name = f"{config.app_name}-canary"
        
        # 计算金丝雀副本数
        total_replicas = config.replicas
        canary_replicas = max(1, int(total_replicas * percentage / 100))
        stable_replicas = total_replicas - canary_replicas
        
        print(f"创建金丝雀部署: {canary_replicas} 副本")
        
        # 创建金丝雀 Deployment
        canary_deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": canary_name,
                "labels": {
                    "app": config.app_name,
                    "role": "canary"
                }
            },
            "spec": {
                "replicas": canary_replicas,
                "selector": {
                    "matchLabels": {
                        "app": config.app_name,
                        "role": "canary"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": config.app_name,
                            "role": "canary"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": config.app_name,
                            "image": config.image,
                            "ports": [{"containerPort": config.port}]
                        }]
                    }
                }
            }
        }
        
        # 更新稳定版 Deployment 的副本数
        # self.k8s.update_deployment_replicas(f"{config.app_name}-stable", stable_replicas)
        
        return canary_deployment
    
    async def _monitor_canary(self, app_name: str) -> Dict[str, Any]:
        """监控金丝雀"""
        # 监控指标
        metrics = {
            "error_rate": 0.01,  # 模拟
            "latency_p95": 200,
            "throughput": 1000
        }
        
        # 检查健康状态
        healthy = (
            metrics["error_rate"] < self.canary_config["rollback_threshold"] and
            metrics["latency_p95"] < 1000
        )
        
        return {
            "healthy": healthy,
            "metrics": metrics
        }
    
    async def _update_canary_percentage(self, app_name: str, percentage: int):
        """更新金丝雀流量比例"""
        canary_replicas = max(1, int(10 * percentage / 100))  # 假设总共 10 个副本
        stable_replicas = 10 - canary_replicas
        
        # 更新副本数
        # self.k8s.update_deployment_replicas(f"{app_name}-canary", canary_replicas)
        # self.k8s.update_deployment_replicas(f"{app_name}-stable", stable_replicas)
        
        print(f"金丝雀副本: {canary_replicas}, 稳定版副本: {stable_replicas}")
    
    async def _rollback_canary(self, app_name: str):
        """回滚金丝雀"""
        # 删除金丝雀部署
        # self.k8s.delete_deployment(f"{app_name}-canary")
        
        # 恢复稳定版副本数
        # self.k8s.update_deployment_replicas(f"{app_name}-stable", 10)
        
        print("已回滚金丝雀部署")
    
    async def _finalize_canary(self, app_name: str):
        """完成金丝雀发布"""
        # 将金丝雀提升为稳定版
        # self.k8s.delete_deployment(f"{app_name}-stable")
        # self.k8s.rename_deployment(f"{app_name}-canary", f"{app_name}-stable")
        
        print("金丝雀发布完成")
```

---

## 27.11 流量管理

### 27.11.1 流量路由

```python
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class TrafficRule:
    """流量规则"""
    name: str
    match: Dict[str, Any]  # 匹配条件
    destination: str  # 目标服务
    weight: int = 100  # 权重
    headers: Dict[str, str] = None  # 请求头修改

class TrafficManager:
    """流量管理器"""
    
    def __init__(self):
        self.rules: List[TrafficRule] = []
        self.services: Dict[str, Dict] = {}
    
    def add_service(self, name: str, config: Dict[str, Any]):
        """添加服务"""
        self.services[name] = config
    
    def add_traffic_rule(self, rule: TrafficRule):
        """添加流量规则"""
        self.rules.append(rule)
    
    def generate_nginx_config(self) -> str:
        """生成 Nginx 配置"""
        config = """
# Agent 流量路由配置
upstream agent_backend {
"""
        
        # 生成 upstream
        for service_name, service_config in self.services.items():
            servers = service_config.get("servers", [])
            for server in servers:
                config += f"    server {server['host']}:{server['port']} weight={server.get('weight', 1)};\n"
        
        config += "}\n\n"
        
        # 生成路由规则
        config += "server {\n"
        config += "    listen 80;\n"
        config += "    server_name agent.example.com;\n\n"
        
        for rule in self.rules:
            config += f"    # {rule.name}\n"
            config += f"    location {rule.match.get('path', '/')} {{\n"
            
            if rule.headers:
                for header, value in rule.headers.items():
                    config += f"        proxy_set_header {header} {value};\n"
            
            config += f"        proxy_pass http://{rule.destination};\n"
            config += "        proxy_set_header Host $host;\n"
            config += "        proxy_set_header X-Real-IP $remote_addr;\n"
            config += "    }\n\n"
        
        config += "}\n"
        
        return config
    
    def generate_istio_config(self) -> str:
        """生成 Istio 配置"""
        config = """
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: agent-service
spec:
  hosts:
  - agent.example.com
  http:
"""
        
        for rule in self.rules:
            config += f"  - match:\n"
            config += f"    - uri:\n"
            config += f"        prefix: {rule.match.get('path', '/')}\n"
            config += f"    route:\n"
            config += f"    - destination:\n"
            config += f"        host: {rule.destination}\n"
            config += f"      weight: {rule.weight}\n"
        
        return config
    
    def validate_rules(self) -> List[Dict[str, Any]]:
        """验证规则"""
        issues = []
        
        # 检查规则冲突
        for i, rule1 in enumerate(self.rules):
            for rule2 in self.rules[i+1:]:
                if self._rules_conflict(rule1, rule2):
                    issues.append({
                        "type": "conflict",
                        "rules": [rule1.name, rule2.name],
                        "description": "规则存在冲突"
                    })
        
        # 检查目标服务是否存在
        for rule in self.rules:
            if rule.destination not in self.services:
                issues.append({
                    "type": "missing_service",
                    "rule": rule.name,
                    "destination": rule.destination,
                    "description": f"目标服务 {rule.destination} 不存在"
                })
        
        return issues
    
    def _rules_conflict(self, rule1: TrafficRule, rule2: TrafficRule) -> bool:
        """检查规则是否冲突"""
        # 简化实现：检查路径是否重叠
        path1 = rule1.match.get("path", "/")
        path2 = rule2.match.get("path", "/")
        
        return path1 == path2 or path1.startswith(path2) or path2.startswith(path1)
```

---

## 27.12 数据库迁移

### 27.12.1 数据库版本管理

```python
import os
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Callable
from dataclasses import dataclass

@dataclass
class Migration:
    """数据库迁移"""
    version: str
    name: str
    up_sql: str
    down_sql: str
    created_at: datetime
    checksum: str

class DatabaseMigrationManager:
    """数据库迁移管理器"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.migrations: List[Migration] = []
        self.migration_history: List[Dict] = []
        
        self._init_migration_table()
    
    def _init_migration_table(self):
        """初始化迁移表"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            checksum VARCHAR(64) NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        # self.db.execute(create_table_sql)
    
    def add_migration(self, migration: Migration):
        """添加迁移"""
        self.migrations.append(migration)
        self.migrations.sort(key=lambda m: m.version)
    
    async def migrate(self, target_version: str = None) -> Dict[str, Any]:
        """执行迁移"""
        applied_count = 0
        
        for migration in self.migrations:
            if target_version and migration.version > target_version:
                break
            
            # 检查是否已应用
            if await self._is_applied(migration.version):
                continue
            
            # 应用迁移
            try:
                await self._apply_migration(migration)
                applied_count += 1
                print(f"已应用迁移: {migration.version} - {migration.name}")
            except Exception as e:
                print(f"迁移失败: {migration.version} - {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "applied_count": applied_count
                }
        
        return {
            "success": True,
            "applied_count": applied_count
        }
    
    async def rollback(self, target_version: str) -> Dict[str, Any]:
        """回滚迁移"""
        rollback_count = 0
        
        # 反向遍历迁移
        for migration in reversed(self.migrations):
            if migration.version <= target_version:
                break
            
            # 检查是否已应用
            if not await self._is_applied(migration.version):
                continue
            
            # 回滚迁移
            try:
                await self._rollback_migration(migration)
                rollback_count += 1
                print(f"已回滚迁移: {migration.version} - {migration.name}")
            except Exception as e:
                print(f"回滚失败: {migration.version} - {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "rollback_count": rollback_count
                }
        
        return {
            "success": True,
            "rollback_count": rollback_count
        }
    
    async def _is_applied(self, version: str) -> bool:
        """检查迁移是否已应用"""
        # query = "SELECT 1 FROM schema_migrations WHERE version = %s"
        # result = self.db.fetch_one(query, (version,))
        # return result is not None
        return False
    
    async def _apply_migration(self, migration: Migration):
        """应用迁移"""
        # 执行 SQL
        # self.db.execute(migration.up_sql)
        
        # 记录迁移
        insert_sql = """
        INSERT INTO schema_migrations (version, name, checksum)
        VALUES (%s, %s, %s)
        """
        # self.db.execute(insert_sql, (migration.version, migration.name, migration.checksum))
        
        self.migration_history.append({
            "version": migration.version,
            "action": "applied",
            "timestamp": datetime.now()
        })
    
    async def _rollback_migration(self, migration: Migration):
        """回滚迁移"""
        # 执行回滚 SQL
        # self.db.execute(migration.down_sql)
        
        # 删除迁移记录
        delete_sql = "DELETE FROM schema_migrations WHERE version = %s"
        # self.db.execute(delete_sql, (migration.version,))
        
        self.migration_history.append({
            "version": migration.version,
            "action": "rolled_back",
            "timestamp": datetime.now()
        })
    
    def get_status(self) -> Dict[str, Any]:
        """获取迁移状态"""
        return {
            "total_migrations": len(self.migrations),
            "applied_migrations": len([m for m in self.migration_history if m["action"] == "applied"]),
            "pending_migrations": len(self.migrations) - len([m for m in self.migration_history if m["action"] == "applied"]),
            "history": self.migration_history
        }

# 预定义的 Agent 数据库迁移
AGENT_MIGRATIONS = [
    Migration(
        version="001",
        name="create_sessions_table",
        up_sql="""
        CREATE TABLE sessions (
            id VARCHAR(36) PRIMARY KEY,
            user_id VARCHAR(36) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            metadata JSON,
            INDEX idx_user_id (user_id)
        );
        """,
        down_sql="DROP TABLE IF EXISTS sessions;",
        created_at=datetime.now(),
        checksum="abc123"
    ),
    Migration(
        version="002",
        name="create_messages_table",
        up_sql="""
        CREATE TABLE messages (
            id VARCHAR(36) PRIMARY KEY,
            session_id VARCHAR(36) NOT NULL,
            role ENUM('user', 'assistant', 'system') NOT NULL,
            content TEXT NOT NULL,
            token_count INT DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
            INDEX idx_session_id (session_id)
        );
        """,
        down_sql="DROP TABLE IF EXISTS messages;",
        created_at=datetime.now(),
        checksum="def456"
    ),
    Migration(
        version="003",
        name="create_tools_usage_table",
        up_sql="""
        CREATE TABLE tool_usage (
            id VARCHAR(36) PRIMARY KEY,
            session_id VARCHAR(36) NOT NULL,
            message_id VARCHAR(36),
            tool_name VARCHAR(100) NOT NULL,
            tool_input JSON,
            tool_output JSON,
            execution_time_ms INT,
            success BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
            INDEX idx_tool_name (tool_name),
            INDEX idx_created_at (created_at)
        );
        """,
        down_sql="DROP TABLE IF EXISTS tool_usage;",
        created_at=datetime.now(),
        checksum="ghi789"
    ),
]
```

---

## 27.13 多环境管理

### 27.13.1 环境配置管理

```python
from typing import Dict, Any, Optional
from dataclasses import dataclass
import os
import json

@dataclass
class EnvironmentConfig:
    """环境配置"""
    name: str
    api_base_url: str
    api_key: str
    model: str
    database_url: str
    redis_url: str
    log_level: str
    debug: bool
    replicas: int

class EnvironmentManager:
    """环境管理器"""
    
    def __init__(self):
        self.environments: Dict[str, EnvironmentConfig] = {}
        self.current_environment: Optional[str] = None
        
        self._load_default_environments()
    
    def _load_default_environments(self):
        """加载默认环境配置"""
        self.environments = {
            "development": EnvironmentConfig(
                name="development",
                api_base_url="http://localhost:8000",
                api_key="dev-key",
                model="gpt-4o-mini",
                database_url="postgresql://localhost:5432/agent_dev",
                redis_url="redis://localhost:6379/0",
                log_level="DEBUG",
                debug=True,
                replicas=1
            ),
            "staging": EnvironmentConfig(
                name="staging",
                api_base_url="https://staging-api.example.com",
                api_key=os.getenv("STAGING_API_KEY", ""),
                model="gpt-4o",
                database_url=os.getenv("STAGING_DATABASE_URL", ""),
                redis_url=os.getenv("STAGING_REDIS_URL", ""),
                log_level="INFO",
                debug=False,
                replicas=2
            ),
            "production": EnvironmentConfig(
                name="production",
                api_base_url="https://api.example.com",
                api_key=os.getenv("PRODUCTION_API_KEY", ""),
                model="gpt-4o",
                database_url=os.getenv("PRODUCTION_DATABASE_URL", ""),
                redis_url=os.getenv("PRODUCTION_REDIS_URL", ""),
                log_level="WARNING",
                debug=False,
                replicas=5
            )
        }
    
    def set_environment(self, env_name: str):
        """设置当前环境"""
        if env_name not in self.environments:
            raise ValueError(f"环境 {env_name} 不存在")
        
        self.current_environment = env_name
        os.environ["AGENT_ENVIRONMENT"] = env_name
    
    def get_config(self, env_name: str = None) -> EnvironmentConfig:
        """获取配置"""
        env_name = env_name or self.current_environment or "development"
        return self.environments.get(env_name)
    
    def add_environment(self, config: EnvironmentConfig):
        """添加环境"""
        self.environments[config.name] = config
    
    def validate_environment(self, env_name: str) -> Dict[str, Any]:
        """验证环境配置"""
        config = self.environments.get(env_name)
        if not config:
            return {"valid": False, "error": f"环境 {env_name} 不存在"}
        
        issues = []
        
        # 检查必要的配置
        if not config.api_key and env_name != "development":
            issues.append("API Key 未配置")
        
        if not config.database_url:
            issues.append("数据库 URL 未配置")
        
        if not config.redis_url:
            issues.append("Redis URL 未配置")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "config": config
        }
    
    def export_config(self, env_name: str, format: str = "env") -> str:
        """导出配置"""
        config = self.environments.get(env_name)
        if not config:
            return ""
        
        if format == "env":
            return f"""
AGENT_ENVIRONMENT={config.name}
AGENT_API_BASE_URL={config.api_base_url}
AGENT_API_KEY={config.api_key}
AGENT_MODEL={config.model}
DATABASE_URL={config.database_url}
REDIS_URL={config.redis_url}
LOG_LEVEL={config.log_level}
DEBUG={config.debug}
REPLICAS={config.replicas}
"""
        elif format == "json":
            return json.dumps({
                "name": config.name,
                "api_base_url": config.api_base_url,
                "model": config.model,
                "log_level": config.log_level,
                "debug": config.debug,
                "replicas": config.replicas
            }, indent=2)
        
        return ""
```

---

## 27.14 灾难恢复

### 27.14.1 备份和恢复策略

```python
from datetime import datetime, timedelta
from typing import List, Dict, Any
from dataclasses import dataclass
import asyncio

@dataclass
class BackupConfig:
    """备份配置"""
    backup_type: str  # full, incremental, differential
    schedule: str  # cron 表达式
    retention_days: int
    storage_location: str
    encryption_enabled: bool = True

class DisasterRecoveryManager:
    """灾难恢复管理器"""
    
    def __init__(self):
        self.backup_configs: Dict[str, BackupConfig] = {}
        self.backup_history: List[Dict] = []
        self.recovery_procedures: Dict[str, Dict] = {}
        
        self._initialize_recovery_procedures()
    
    def _initialize_recovery_procedures(self):
        """初始化恢复程序"""
        self.recovery_procedures = {
            "database_recovery": {
                "description": "数据库恢复",
                "steps": [
                    "停止应用服务",
                    "从备份恢复数据库",
                    "验证数据完整性",
                    "重启应用服务",
                    "验证服务可用性"
                ],
                "estimated_time": "30-60 分钟",
                "rto": 60,  # 恢复时间目标（分钟）
                "rpo": 15   # 恢复点目标（分钟）
            },
            "full_system_recovery": {
                "description": "完整系统恢复",
                "steps": [
                    "评估灾难范围",
                    "启动备用环境",
                    "恢复数据库",
                    "恢复应用配置",
                    "恢复用户数据",
                    "切换 DNS",
                    "验证服务可用性"
                ],
                "estimated_time": "2-4 小时",
                "rto": 240,
                "rpo": 60
            },
            "data_recovery": {
                "description": "数据恢复",
                "steps": [
                    "确定数据丢失范围",
                    "从备份恢复数据",
                    "验证数据完整性",
                    "同步增量数据"
                ],
                "estimated_time": "15-30 分钟",
                "rto": 30,
                "rpo": 5
            }
        }
    
    async def create_backup(
        self,
        backup_type: str,
        components: List[str]
    ) -> Dict[str, Any]:
        """创建备份"""
        backup_id = f"backup_{int(time.time())}"
        
        print(f"开始创建备份: {backup_id}")
        print(f"备份类型: {backup_type}")
        print(f"备份组件: {components}")
        
        backup_result = {
            "backup_id": backup_id,
            "type": backup_type,
            "components": components,
            "start_time": datetime.now(),
            "status": "in_progress",
            "details": {}
        }
        
        for component in components:
            try:
                component_result = await self._backup_component(component, backup_type)
                backup_result["details"][component] = component_result
            except Exception as e:
                backup_result["details"][component] = {
                    "status": "failed",
                    "error": str(e)
                }
                backup_result["status"] = "partial"
        
        if backup_result["status"] != "partial":
            backup_result["status"] = "completed"
        
        backup_result["end_time"] = datetime.now()
        backup_result["duration"] = (backup_result["end_time"] - backup_result["start_time"]).total_seconds()
        
        self.backup_history.append(backup_result)
        
        return backup_result
    
    async def _backup_component(self, component: str, backup_type: str) -> Dict:
        """备份组件"""
        # 模拟备份过程
        await asyncio.sleep(2)
        
        return {
            "status": "success",
            "size_mb": 1024,
            "checksum": hashlib.md5(str(time.time()).encode()).hexdigest()
        }
    
    async def restore_from_backup(
        self,
        backup_id: str,
        components: List[str] = None
    ) -> Dict[str, Any]:
        """从备份恢复"""
        # 查找备份
        backup = None
        for b in self.backup_history:
            if b["backup_id"] == backup_id:
                backup = b
                break
        
        if not backup:
            return {"success": False, "error": f"备份 {backup_id} 不存在"}
        
        print(f"开始从备份 {backup_id} 恢复")
        
        restore_result = {
            "backup_id": backup_id,
            "start_time": datetime.now(),
            "status": "in_progress",
            "details": {}
        }
        
        components_to_restore = components or list(backup["details"].keys())
        
        for component in components_to_restore:
            if component not in backup["details"]:
                restore_result["details"][component] = {
                    "status": "skipped",
                    "reason": "组件不在备份中"
                }
                continue
            
            try:
                component_result = await self._restore_component(component, backup)
                restore_result["details"][component] = component_result
            except Exception as e:
                restore_result["details"][component] = {
                    "status": "failed",
                    "error": str(e)
                }
                restore_result["status"] = "partial"
        
        if restore_result["status"] != "partial":
            restore_result["status"] = "completed"
        
        restore_result["end_time"] = datetime.now()
        restore_result["duration"] = (restore_result["end_time"] - restore_result["start_time"]).total_seconds()
        
        return restore_result
    
    async def _restore_component(self, component: str, backup: Dict) -> Dict:
        """恢复组件"""
        # 模拟恢复过程
        await asyncio.sleep(3)
        
        return {
            "status": "success",
            "restored_items": 1000
        }
    
    def get_recovery_plan(self, scenario: str) -> Dict[str, Any]:
        """获取恢复计划"""
        procedure = self.recovery_procedures.get(scenario)
        if not procedure:
            return {"error": f"未知的恢复场景: {scenario}"}
        
        return {
            "scenario": scenario,
            "procedure": procedure,
            "last_backup": self.backup_history[-1] if self.backup_history else None,
            "backup_count": len(self.backup_history),
            "recommendations": self._generate_recommendations(scenario)
        }
    
    def _generate_recommendations(self, scenario: str) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 检查备份频率
        recent_backups = [
            b for b in self.backup_history
            if b["start_time"] > datetime.now() - timedelta(days=1)
        ]
        
        if len(recent_backups) < 3:
            recommendations.append("建议增加备份频率，至少每天 3 次")
        
        # 检查备份成功率
        if self.backup_history:
            success_rate = sum(
                1 for b in self.backup_history if b["status"] == "completed"
            ) / len(self.backup_history)
            
            if success_rate < 0.95:
                recommendations.append(f"备份成功率较低 ({success_rate:.1%})，建议检查备份配置")
        
        return recommendations
    
    def test_recovery(self, scenario: str) -> Dict[str, Any]:
        """测试恢复流程"""
        print(f"开始测试恢复流程: {scenario}")
        
        procedure = self.recovery_procedures.get(scenario)
        if not procedure:
            return {"success": False, "error": "未知的恢复场景"}
        
        test_result = {
            "scenario": scenario,
            "start_time": datetime.now(),
            "steps": [],
            "success": True
        }
        
        for step in procedure["steps"]:
            step_result = {
                "step": step,
                "status": "completed",
                "duration": 5  # 模拟
            }
            test_result["steps"].append(step_result)
        
        test_result["end_time"] = datetime.now()
        test_result["total_duration"] = sum(s["duration"] for s in test_result["steps"])
        
        return test_result
```

---

## 27.15 性能调优

### 27.15.1 性能优化策略

```python
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
import asyncio

@dataclass
class PerformanceProfile:
    """性能配置"""
    name: str
    description: str
    optimizations: List[Dict[str, Any]]
    expected_improvement: str

class PerformanceTuner:
    """性能调优器"""
    
    def __init__(self):
        self.profiles: Dict[str, PerformanceProfile] = {}
        self.benchmarks: List[Dict] = []
        
        self._initialize_profiles()
    
    def _initialize_profiles(self):
        """初始化性能配置"""
        self.profiles = {
            "high_throughput": PerformanceProfile(
                name="高吞吐量",
                description="优化系统以处理更多并发请求",
                optimizations=[
                    {
                        "name": "启用连接池",
                        "description": "增加数据库和 Redis 连接池大小",
                        "config": {"db_pool_size": 20, "redis_pool_size": 10}
                    },
                    {
                        "name": "启用异步处理",
                        "description": "使用异步 I/O 处理请求",
                        "config": {"async_mode": True}
                    },
                    {
                        "name": "增加缓存",
                        "description": "启用多层缓存减少数据库查询",
                        "config": {"cache_enabled": True, "cache_ttl": 300}
                    }
                ],
                expected_improvement="吞吐量提升 50-100%"
            ),
            "low_latency": PerformanceProfile(
                name="低延迟",
                description="优化系统以减少响应时间",
                optimizations=[
                    {
                        "name": "启用响应缓存",
                        "description": "缓存常见查询的响应",
                        "config": {"response_cache": True}
                    },
                    {
                        "name": "优化数据库查询",
                        "description": "添加索引和优化查询语句",
                        "config": {"query_optimization": True}
                    },
                    {
                        "name": "使用 CDN",
                        "description": "将静态资源部署到 CDN",
                        "config": {"cdn_enabled": True}
                    }
                ],
                expected_improvement="延迟降低 30-50%"
            ),
            "cost_optimized": PerformanceProfile(
                name="成本优化",
                description="优化系统以降低运营成本",
                optimizations=[
                    {
                        "name": "模型路由",
                        "description": "根据任务复杂度选择合适的模型",
                        "config": {"model_routing": True}
                    },
                    {
                        "name": "Prompt 压缩",
                        "description": "压缩提示词减少 Token 消耗",
                        "config": {"prompt_compression": True}
                    },
                    {
                        "name": "请求批处理",
                        "description": "合并多个请求批量处理",
                        "config": {"batch_processing": True}
                    }
                ],
                expected_improvement="成本降低 40-60%"
            )
        }
    
    def get_profile(self, profile_name: str) -> PerformanceProfile:
        """获取性能配置"""
        return self.profiles.get(profile_name)
    
    async def apply_profile(
        self,
        profile_name: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """应用性能配置"""
        profile = self.profiles.get(profile_name)
        if not profile:
            return {"success": False, "error": f"配置 {profile_name} 不存在"}
        
        print(f"应用性能配置: {profile.name}")
        
        results = []
        for optimization in profile.optimizations:
            result = await self._apply_optimization(optimization, config)
            results.append({
                "optimization": optimization["name"],
                "result": result
            })
        
        return {
            "success": True,
            "profile": profile_name,
            "optimizations_applied": len(results),
            "results": results
        }
    
    async def _apply_optimization(self, optimization: Dict, config: Dict) -> Dict:
        """应用优化"""
        # 模拟应用优化
        await asyncio.sleep(1)
        
        return {
            "status": "applied",
            "config": optimization.get("config", {})
        }
    
    async def benchmark(
        self,
        test_function: Callable,
        iterations: int = 100,
        concurrency: int = 10
    ) -> Dict[str, Any]:
        """执行基准测试"""
        print(f"开始基准测试: {iterations} 次迭代, {concurrency} 并发")
        
        start_time = time.time()
        results = []
        
        async def run_single():
            iteration_start = time.time()
            await test_function()
            iteration_time = time.time() - iteration_start
            results.append(iteration_time)
        
        # 执行测试
        tasks = [run_single() for _ in range(iterations)]
        await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # 计算统计指标
        avg_latency = sum(results) / len(results)
        p50_latency = sorted(results)[len(results) // 2]
        p95_latency = sorted(results)[int(len(results) * 0.95)]
        p99_latency = sorted(results)[int(len(results) * 0.99)]
        
        throughput = iterations / total_time
        
        benchmark_result = {
            "iterations": iterations,
            "concurrency": concurrency,
            "total_time": total_time,
            "throughput": throughput,
            "latency": {
                "avg": avg_latency,
                "p50": p50_latency,
                "p95": p95_latency,
                "p99": p99_latency,
                "min": min(results),
                "max": max(results)
            }
        }
        
        self.benchmarks.append(benchmark_result)
        
        return benchmark_result
    
    def compare_benchmarks(self, benchmark1: Dict, benchmark2: Dict) -> Dict[str, Any]:
        """比较基准测试结果"""
        return {
            "throughput_change": (benchmark2["throughput"] - benchmark1["throughput"]) / benchmark1["throughput"] * 100,
            "latency_change": (benchmark2["latency"]["avg"] - benchmark1["latency"]["avg"]) / benchmark1["latency"]["avg"] * 100,
            "p95_change": (benchmark2["latency"]["p95"] - benchmark1["latency"]["p95"]) / benchmark1["latency"]["p95"] * 100,
            "improvement": benchmark2["throughput"] > benchmark1["throughput"]
        }
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """生成优化报告"""
        return {
            "available_profiles": list(self.profiles.keys()),
            "benchmark_history": self.benchmarks,
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if self.benchmarks:
            latest = self.benchmarks[-1]
            
            if latest["latency"]["p95"] > 1000:
                recommendations.append("P95 延迟较高，建议启用响应缓存和数据库优化")
            
            if latest["throughput"] < 100:
                recommendations.append("吞吐量较低，建议增加并发处理和连接池")
        
        return recommendations
```

---

## 27.16 生产环境案例

### 27.16.1 企业级 Agent 部署案例

```python
class EnterpriseDeploymentCase:
    """企业级 Agent 部署案例"""
    
    def __init__(self):
        self.case_study = {
            "company": "某大型金融机构",
            "scale": {
                "daily_requests": 5_000_000,
                "concurrent_users": 50_000,
                "agents": 20,
                "compliance_requirements": ["SOC2", "GDPR", "PCI-DSS"]
            },
            "challenges": [
                "严格的合规要求",
                "高可用性需求（99.99% SLA）",
                "数据安全和隐私保护",
                "实时性能监控"
            ],
            "solutions": [],
            "results": {}
        }
    
    async def implement_deployment(self) -> Dict[str, Any]:
        """实施部署方案"""
        solutions = [
            {
                "challenge": "合规要求",
                "solution": "实现完整的审计日志和数据加密",
                "implementation": "使用 Kubernetes NetworkPolicy 实现网络隔离，启用 Pod Security Policy",
                "result": "通过所有合规审计"
            },
            {
                "challenge": "高可用性",
                "solution": "多区域部署和自动故障转移",
                "implementation": "使用 Istio 实现服务网格，配置健康检查和自动重启",
                "result": "可用性达到 99.99%"
            },
            {
                "challenge": "数据安全",
                "solution": "端到端加密和访问控制",
                "implementation": "使用 AWS KMS 进行密钥管理，实现 RBAC 和最小权限原则",
                "result": "零安全事故"
            },
            {
                "challenge": "性能监控",
                "solution": "全链路追踪和实时告警",
                "implementation": "部署 Jaeger + Prometheus + Grafana，配置自定义告警规则",
                "result": "故障定位时间从小时级降到分钟级"
            }
        ]
        
        self.case_study["solutions"] = solutions
        
        self.case_study["results"] = {
            "availability": "99.99%",
            "avg_latency": "150ms",
            "compliance_score": "100%",
            "security_incidents": "0",
            "cost_reduction": "25%",
            "deployment_frequency": "每天 10+ 次"
        }
        
        return self.case_study
```

---

## 27.17 常见问题和解决方案

### 27.17.1 部署问题排查

```python
class DeploymentTroubleshooting:
    """部署问题排查指南"""
    
    @staticmethod
    def get_common_issues() -> List[Dict[str, Any]]:
        """获取常见问题"""
        return [
            {
                "issue": "Pod 无法启动",
                "symptoms": ["Pod 处于 CrashLoopBackOff 状态", "容器反复重启"],
                "causes": [
                    "镜像拉取失败",
                    "配置错误",
                    "资源不足",
                    "健康检查失败"
                ],
                "solutions": [
                    "检查镜像名称和标签",
                    "验证 ConfigMap 和 Secret",
                    "增加资源限制",
                    "调整健康检查参数"
                ]
            },
            {
                "issue": "服务无法访问",
                "symptoms": ["请求超时", "连接被拒绝"],
                "causes": [
                    "Service 配置错误",
                    "Ingress 规则错误",
                    "网络策略阻止",
                    "端口未暴露"
                ],
                "solutions": [
                    "检查 Service selector",
                    "验证 Ingress 配置",
                    "检查 NetworkPolicy",
                    "确认容器端口"
                ]
            },
            {
                "issue": "数据库连接失败",
                "symptoms": ["连接超时", "认证失败"],
                "causes": [
                    "数据库服务未启动",
                    "连接字符串错误",
                    "防火墙规则",
                    "连接池耗尽"
                ],
                "solutions": [
                    "检查数据库 Pod 状态",
                    "验证连接配置",
                    "检查网络策略",
                    "增加连接池大小"
                ]
            },
            {
                "issue": "性能下降",
                "symptoms": ["响应时间增加", "吞吐量下降"],
                "causes": [
                    "资源不足",
                    "内存泄漏",
                    "数据库慢查询",
                    "网络延迟"
                ],
                "solutions": [
                    "增加资源限制",
                    "检查内存使用",
                    "优化数据库查询",
                    "检查网络延迟"
                ]
            }
        ]
    
    @staticmethod
    def get_diagnostic_commands() -> Dict[str, List[str]]:
        """获取诊断命令"""
        return {
            "pod状态": [
                "kubectl get pods -n agent",
                "kubectl describe pod <pod-name> -n agent",
                "kubectl logs <pod-name> -n agent"
            ],
            "服务状态": [
                "kubectl get svc -n agent",
                "kubectl get endpoints -n agent",
                "kubectl describe svc <service-name> -n agent"
            ],
            "部署状态": [
                "kubectl get deployments -n agent",
                "kubectl rollout status deployment/<deployment-name> -n agent",
                "kubectl rollout history deployment/<deployment-name> -n agent"
            ],
            "资源使用": [
                "kubectl top nodes",
                "kubectl top pods -n agent",
                "kubectl describe nodes"
            ],
            "网络问题": [
                "kubectl get networkpolicies -n agent",
                "kubectl get ingress -n agent",
                "kubectl describe ingress <ingress-name> -n agent"
            ]
        }
    
    @staticmethod
    def get_recovery_checklist() -> List[Dict[str, str]]:
        """获取恢复清单"""
        return [
            {"step": "评估影响范围", "command": "kubectl get pods -n agent --field-selector=status.phase!=Running"},
            {"step": "检查事件日志", "command": "kubectl get events -n agent --sort-by='.lastTimestamp'"},
            {"step": "检查资源使用", "command": "kubectl top pods -n agent"},
            {"step": "检查网络连通性", "command": "kubectl exec -it <pod> -- curl -s http://localhost:8000/health"},
            {"step": "检查数据库连接", "command": "kubectl exec -it <pod> -- python -c 'import psycopg2; print(\"OK\")'"},
            {"step": "重启问题 Pod", "command": "kubectl delete pod <pod-name> -n agent"},
            {"step": "回滚部署", "command": "kubectl rollout undo deployment/<deployment-name> -n agent"}
        ]
```

---

## 27.18 本章小结（更新版）

| 知识点 | 核心要点 |
|:---|:---|
| **Docker** | 多阶段构建、镜像优化、健康检查 |
| **Kubernetes** | Deployment、Service、HPA 自动扩缩容 |
| **API 网关** | Nginx 配置、速率限制、负载均衡 |
| **FastAPI** | 异步框架、依赖注入、中间件 |
| **缓存策略** | 多层缓存、语义缓存、缓存失效 |
| **成本优化** | 模型路由、Prompt 压缩、Token 监控 |
| **可靠性** | 重试、熔断、优雅降级 |
| **安全** | JWT 认证、速率限制、输入验证 |
| **监控** | 健康检查、性能指标、Prometheus 集成 |
| **蓝绿部署** | 零停机部署、快速回滚 |
| **金丝雀发布** | 渐进式发布、风险控制 |
| **流量管理** | 路由规则、流量分割、A/B 测试 |
| **数据库迁移** | 版本控制、回滚机制 |
| **多环境管理** | 开发、测试、生产环境隔离 |
| **灾难恢复** | 备份策略、恢复流程、RTO/RPO |
| **性能调优** | 基准测试、优化策略、监控分析 |

---

## 27.19 生产环境最佳实践

### 27.19.1 部署检查清单

```python
class DeploymentChecklist:
    """部署检查清单"""
    
    @staticmethod
    def get_pre_deployment_checks() -> List[Dict[str, Any]]:
        """获取部署前检查项"""
        return [
            {
                "category": "代码质量",
                "checks": [
                    {"name": "单元测试通过", "required": True, "command": "pytest tests/"},
                    {"name": "集成测试通过", "required": True, "command": "pytest tests/integration/"},
                    {"name": "代码覆盖率 > 80%", "required": True, "command": "pytest --cov=src tests/"},
                    {"name": "代码审查完成", "required": True, "command": "Check PR approvals"},
                    {"name": "安全扫描通过", "required": True, "command": "bandit -r src/"}
                ]
            },
            {
                "category": "配置验证",
                "checks": [
                    {"name": "环境变量已配置", "required": True, "command": "Check .env files"},
                    {"name": "Secrets 已设置", "required": True, "command": "kubectl get secrets"},
                    {"name": "ConfigMap 已创建", "required": True, "command": "kubectl get configmaps"},
                    {"name": "数据库迁移已准备", "required": True, "command": "Check migration scripts"}
                ]
            },
            {
                "category": "基础设施",
                "checks": [
                    {"name": "Kubernetes 集群健康", "required": True, "command": "kubectl cluster-info"},
                    {"name": "节点资源充足", "required": True, "command": "kubectl top nodes"},
                    {"name": "存储卷可用", "required": True, "command": "kubectl get pv"},
                    {"name": "网络策略已配置", "required": False, "command": "kubectl get networkpolicies"}
                ]
            },
            {
                "category": "监控和告警",
                "checks": [
                    {"name": "健康检查端点可用", "required": True, "command": "curl http://localhost:8000/health"},
                    {"name": "Prometheus 指标正常", "required": True, "command": "Check Prometheus targets"},
                    {"name": "告警规则已配置", "required": True, "command": "Check alertmanager config"},
                    {"name": "日志收集正常", "required": True, "command": "Check Elasticsearch indices"}
                ]
            }
        ]
    
    @staticmethod
    def get_post_deployment_checks() -> List[Dict[str, Any]]:
        """获取部署后检查项"""
        return [
            {
                "category": "服务验证",
                "checks": [
                    {"name": "所有 Pod 运行正常", "command": "kubectl get pods -n agent"},
                    {"name": "Service 端点可用", "command": "kubectl get endpoints -n agent"},
                    {"name": "健康检查通过", "command": "curl http://service/health"},
                    {"name": "API 响应正常", "command": "curl http://service/api/test"}
                ]
            },
            {
                "category": "功能验证",
                "checks": [
                    {"name": "核心功能测试", "command": "Run smoke tests"},
                    {"name": "回归测试通过", "command": "Run regression tests"},
                    {"name": "性能测试达标", "command": "Run performance tests"},
                    {"name": "安全测试通过", "command": "Run security tests"}
                ]
            },
            {
                "category": "监控验证",
                "checks": [
                    {"name": "指标正常采集", "command": "Check Grafana dashboards"},
                    {"name": "日志正常记录", "command": "Check Kibana logs"},
                    {"name": "告警未触发", "command": "Check Alertmanager"},
                    {"name": "追踪数据正常", "command": "Check Jaeger traces"}
                ]
            }
        ]
```

### 27.19.2 安全最佳实践

```python
class SecurityBestPractices:
    """安全最佳实践"""
    
    @staticmethod
    def get_kubernetes_security_practices() -> Dict[str, Any]:
        """获取 Kubernetes 安全实践"""
        return {
            "pod_security": {
                "description": "Pod 安全策略",
                "practices": [
                    "使用非 root 用户运行容器",
                    "设置只读根文件系统",
                    "禁止特权升级",
                    "删除不必要的 Linux capabilities",
                    "使用 Seccomp 和 AppArmor 配置文件"
                ],
                "example_config": {
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 1000,
                        "readOnlyRootFilesystem": True,
                        "allowPrivilegeEscalation": False,
                        "capabilities": {
                            "drop": ["ALL"]
                        }
                    }
                }
            },
            "network_security": {
                "description": "网络安全",
                "practices": [
                    "使用 NetworkPolicy 限制 Pod 间通信",
                    "启用 mTLS 进行服务间加密",
                    "使用 Ingress 控制器进行 TLS 终止",
                    "配置 WAF 防护常见攻击"
                ]
            },
            "secret_management": {
                "description": "密钥管理",
                "practices": [
                    "使用 Kubernetes Secrets 或外部密钥管理系统",
                    "定期轮换密钥",
                    "限制密钥访问权限",
                    "不在镜像或配置文件中硬编码密钥"
                ]
            },
            "image_security": {
                "description": "镜像安全",
                "practices": [
                    "使用可信的基础镜像",
                    "扫描镜像漏洞",
                    "签名和验证镜像",
                    "使用最小化镜像（如 distroless）"
                ]
            }
        }
    
    @staticmethod
    def get_application_security_practices() -> Dict[str, Any]:
        """获取应用安全实践"""
        return {
            "authentication": {
                "description": "认证",
                "practices": [
                    "实施强密码策略",
                    "支持多因素认证",
                    "使用 OAuth 2.0 / OpenID Connect",
                    "实现会话管理"
                ]
            },
            "authorization": {
                "description": "授权",
                "practices": [
                    "实施 RBAC（基于角色的访问控制）",
                    "遵循最小权限原则",
                    "定期审查权限",
                    "实现 API 级别的权限控制"
                ]
            },
            "input_validation": {
                "description": "输入验证",
                "practices": [
                    "验证所有用户输入",
                    "使用参数化查询防止 SQL 注入",
                    "对输出进行编码防止 XSS",
                    "实施内容安全策略（CSP）"
                ]
            },
            "data_protection": {
                "description": "数据保护",
                "practices": [
                    "加密敏感数据（传输中和静态）",
                    "实施数据脱敏",
                    "遵循数据最小化原则",
                    "实施数据保留策略"
                ]
            }
        }
```

### 27.19.3 可观测性最佳实践

```python
class ObservabilityBestPractices:
    """可观测性最佳实践"""
    
    @staticmethod
    def get_metrics_practices() -> Dict[str, Any]:
        """获取指标最佳实践"""
        return {
            "metric_design": {
                "description": "指标设计",
                "practices": [
                    "使用有意义的指标名称（如 http_request_duration_seconds）",
                    "为指标添加标签以支持多维分析",
                    "区分计数器（counter）、直方图（histogram）和仪表（gauge）",
                    "设置合理的指标粒度"
                ]
            },
            "cardinality_management": {
                "description": "基数管理",
                "practices": [
                    "限制标签值的数量",
                    "避免使用高基数标签（如 user_id）",
                    "使用聚合减少基数",
                    "监控指标基数"
                ]
            },
            "alerting": {
                "description": "告警",
                "practices": [
                    "基于 SLO（服务级别目标）设置告警",
                    "避免告警疲劳",
                    "实施告警分级",
                    "配置告警抑制和静默"
                ]
            }
        }
    
    @staticmethod
    def get_logging_practices() -> Dict[str, Any]:
        """获取日志最佳实践"""
        return {
            "structured_logging": {
                "description": "结构化日志",
                "practices": [
                    "使用 JSON 格式的结构化日志",
                    "包含标准化字段（timestamp, level, message, trace_id）",
                    "避免在日志中记录敏感信息",
                    "使用适当的日志级别"
                ]
            },
            "log_aggregation": {
                "description": "日志聚合",
                "practices": [
                    "集中收集所有服务的日志",
                    "实施日志保留策略",
                    "使用日志采样减少存储成本",
                    "配置日志告警"
                ]
            },
            "correlation": {
                "description": "关联",
                "practices": [
                    "在日志中包含 trace_id 以便关联",
                    "在指标中包含日志上下文",
                    "实现日志和追踪的双向关联"
                ]
            }
        }
    
    @staticmethod
    def get_tracing_practices() -> Dict[str, Any]:
        """获取追踪最佳实践"""
        return {
            "trace_context": {
                "description": "追踪上下文",
                "practices": [
                    "在所有服务间传播追踪上下文",
                    "为重要的操作创建 span",
                    "在 span 中记录有意义的属性",
                    "关联日志和指标"
                ]
            },
            "sampling": {
                "description": "采样",
                "practices": [
                    "使用自适应采样策略",
                    "对错误和慢请求进行 100% 采样",
                    "监控采样率对存储的影响",
                    "根据服务重要性调整采样率"
                ]
            },
            "performance": {
                "description": "性能",
                "practices": [
                    "异步上报追踪数据",
                    "批量发送 span",
                    "监控追踪系统本身的性能",
                    "优化 span 属性大小"
                ]
            }
        }
```

---

## 27.20 案例研究：大规模 Agent 系统部署

### 27.20.1 案例概述

```python
class LargeScaleAgentDeploymentCase:
    """大规模 Agent 系统部署案例"""
    
    def __init__(self):
        self.case_study = {
            "company": "某全球科技公司",
            "scale": {
                "daily_requests": 100_000_000,
                "concurrent_users": 500_000,
                "agents": 50,
                "regions": 10,
                "languages": 20
            },
            "challenges": [
                "全球多区域部署",
                "高并发和低延迟要求",
                "多语言和多文化支持",
                "严格的合规要求（GDPR, CCPA）",
                "实时成本控制"
            ],
            "architecture": {
                "compute": "Kubernetes (GKE)",
                "database": "Cloud Spanner + Redis",
                "messaging": "Pub/Sub",
                "storage": "Cloud Storage",
                "cdn": "Cloud CDN",
                "monitoring": "Cloud Monitoring + custom dashboards"
            }
        }
    
    async def implement_solution(self) -> Dict[str, Any]:
        """实施解决方案"""
        solutions = [
            {
                "challenge": "全球多区域部署",
                "solution": "使用 GKE 多集群部署，配合 Cloud Load Balancing",
                "implementation": [
                    "在 10 个区域部署 Kubernetes 集群",
                    "使用 Cloud Load Balancing 进行全局负载均衡",
                    "实现区域故障转移",
                    "使用 Cloud Spanner 进行全球数据同步"
                ],
                "results": {
                    "latency_reduction": "60%",
                    "availability": "99.99%",
                    "disaster_recovery_time": "< 5 分钟"
                }
            },
            {
                "challenge": "高并发和低延迟",
                "solution": "多层缓存 + 异步处理 + 自动扩缩容",
                "implementation": [
                    "实现 CDN 缓存静态资源",
                    "使用 Redis 集群缓存热点数据",
                    "实现异步消息处理",
                    "配置 HPA 自动扩缩容"
                ],
                "results": {
                    "p99_latency": "< 200ms",
                    "throughput": "10,000 RPS per region",
                    "auto_scaling_time": "< 30 seconds"
                }
            },
            {
                "challenge": "实时成本控制",
                "solution": "智能模型路由 + Token 监控 + 预算告警",
                "implementation": [
                    "根据任务复杂度选择模型",
                    "实时监控 Token 使用量",
                    "设置预算告警和限制",
                    "实现请求批处理"
                ],
                "results": {
                    "cost_reduction": "45%",
                    "budget_overrun": "0%",
                    "cost_per_request": "$0.001"
                }
            }
        ]
        
        self.case_study["solutions"] = solutions
        
        self.case_study["results"] = {
            "global_availability": "99.99%",
            "avg_latency_global": "120ms",
            "daily_requests_served": "100M+",
            "cost_optimization": "45%",
            "deployment_frequency": "100+ per day",
            "mttr": "< 5 minutes",
            "compliance": "100% (GDPR, CCPA, SOC2)"
        }
        
        return self.case_study
```

---

## 27.21 本章小结（最终版）

| 知识点 | 核心要点 |
|:---|:---|
| **容器化** | Docker 多阶段构建、镜像优化、健康检查 |
| **容器编排** | Kubernetes Deployment、Service、HPA、Ingress |
| **API 网关** | Nginx/Traefik 配置、速率限制、负载均衡 |
| **应用框架** | FastAPI 异步服务、依赖注入、中间件 |
| **缓存策略** | 多层缓存、语义缓存、Redis 集群 |
| **成本优化** | 模型路由、Prompt 压缩、Token 监控、预算管理 |
| **高可用性** | 重试、熔断、优雅降级、多区域部署 |
| **安全防护** | JWT 认证、速率限制、输入验证、密钥管理 |
| **可观测性** | 健康检查、性能指标、分布式追踪、日志聚合 |
| **部署策略** | 蓝绿部署、金丝雀发布、滚动更新 |
| **流量管理** | 路由规则、流量分割、A/B 测试、灰度发布 |
| **数据管理** | 数据库迁移、备份恢复、数据同步 |
| **环境管理** | 多环境隔离、配置管理、环境变量 |
| **灾难恢复** | 备份策略、恢复流程、RTO/RPO、故障转移 |
| **性能调优** | 基准测试、性能分析、优化策略 |
| **生产实践** | 部署清单、安全审计、合规要求、最佳实践 |

> **下一章预告**
>
> 在第 28 章中，我们将学习 Agent 可观测性。

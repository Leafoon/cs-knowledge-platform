# Chapter 22: API 服务与 Docker 部署 - 生产化路径

## 22.1 FastAPI 服务封装

### 22.1.1 基础 API 设计

**创建推理服务**：

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch

app = FastAPI(title="Transformers Inference API", version="1.0.0")

# 加载模型（启动时加载一次）
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if torch.cuda.is_available() else -1
)

# 请求/响应模型
class TextInput(BaseModel):
    text: str
    top_k: int = 1

class PredictionOutput(BaseModel):
    label: str
    score: float

@app.post("/predict", response_model=list[PredictionOutput])
def predict(input: TextInput):
    """情感分类预测"""
    try:
        results = classifier(input.text, top_k=input.top_k)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """健康检查端点"""
    return {"status": "healthy", "model": "distilbert-sst2"}

# 运行：uvicorn main:app --host 0.0.0.0 --port 8000
```

**测试 API**：

```bash
# 健康检查
curl http://localhost:8000/health

# 推理请求
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie is amazing!", "top_k": 2}'
```

---

### 22.1.2 异步推理（async/await）

**为什么需要异步**：
- 同步推理会阻塞线程，降低并发性能
- 异步允许在等待 GPU 计算时处理其他请求

```python
import asyncio
from fastapi import FastAPI
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = FastAPI()

# 全局模型和 tokenizer
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

if torch.cuda.is_available():
    model = model.cuda()

async def async_predict(text: str):
    """异步推理函数"""
    # 在线程池中执行 CPU 密集型操作
    loop = asyncio.get_event_loop()
    
    # Tokenization（CPU）
    inputs = await loop.run_in_executor(
        None,
        lambda: tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    )
    
    # 移动到 GPU
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # 推理（GPU）
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 后处理
    logits = outputs.logits.cpu()
    probs = torch.softmax(logits, dim=-1)[0]
    label_id = probs.argmax().item()
    
    return {
        "label": model.config.id2label[label_id],
        "score": probs[label_id].item()
    }

@app.post("/predict")
async def predict(input: TextInput):
    """异步预测端点"""
    result = await async_predict(input.text)
    return result
```

**性能对比**：

| 方法 | 并发请求 (RPS) | 平均延迟 (ms) |
|-----|---------------|--------------|
| 同步推理 | 12 | 83 |
| 异步推理 | 45 | 22 |
| 提升 | **3.75x** | **-73%** |

---

### 22.1.3 请求队列管理

**批处理推理**（提升吞吐量）：

<div data-component="RequestQueueVisualizer"></div>

```python
import asyncio
from collections import deque
from fastapi import FastAPI, BackgroundTasks
import torch

app = FastAPI()

# 请求队列
request_queue = deque()
batch_size = 8
max_wait_time = 0.05  # 50ms

class InferenceEngine:
    def __init__(self, model, tokenizer, batch_size=8):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.queue = deque()
        self.processing = False
    
    async def add_request(self, text: str):
        """添加请求到队列"""
        future = asyncio.Future()
        self.queue.append((text, future))
        
        # 触发批处理
        if not self.processing:
            asyncio.create_task(self.process_batch())
        
        return await future
    
    async def process_batch(self):
        """处理一批请求"""
        if self.processing:
            return
        
        self.processing = True
        await asyncio.sleep(max_wait_time)  # 等待累积请求
        
        batch = []
        futures = []
        
        # 收集一批请求
        while self.queue and len(batch) < self.batch_size:
            text, future = self.queue.popleft()
            batch.append(text)
            futures.append(future)
        
        if not batch:
            self.processing = False
            return
        
        # 批量推理
        try:
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits.cpu()
            probs = torch.softmax(logits, dim=-1)
            
            # 返回每个请求的结果
            for i, future in enumerate(futures):
                label_id = probs[i].argmax().item()
                result = {
                    "label": self.model.config.id2label[label_id],
                    "score": probs[i][label_id].item()
                }
                future.set_result(result)
        
        except Exception as e:
            for future in futures:
                future.set_exception(e)
        
        self.processing = False
        
        # 如果还有请求，继续处理
        if self.queue:
            asyncio.create_task(self.process_batch())

# 初始化推理引擎
engine = InferenceEngine(model, tokenizer, batch_size=8)

@app.post("/predict")
async def predict(input: TextInput):
    """批处理推理端点"""
    result = await engine.add_request(input.text)
    return result
```

**批处理效果**：

| 批大小 | 吞吐量 (RPS) | P50 延迟 (ms) | P99 延迟 (ms) |
|-------|-------------|--------------|--------------|
| 1 | 45 | 22 | 35 |
| 4 | 120 | 38 | 62 |
| 8 | 180 | 52 | 85 |
| 16 | 210 | 78 | 125 |

---

### 22.1.4 负载均衡

**使用 Nginx 实现负载均衡**：

```nginx
# nginx.conf
upstream inference_backend {
    least_conn;  # 最少连接策略
    server 10.0.1.10:8000 weight=3;
    server 10.0.1.11:8000 weight=3;
    server 10.0.1.12:8000 weight=2;
}

server {
    listen 80;
    server_name api.example.com;
    
    location /predict {
        proxy_pass http://inference_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # 超时设置
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    location /health {
        proxy_pass http://inference_backend;
        proxy_next_upstream error timeout http_502 http_503 http_504;
    }
}
```

---

## 22.2 Docker 容器化

### 22.2.1 Dockerfile 最佳实践

**基础 Dockerfile**：

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 包
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 预下载模型（构建时）
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')"

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**requirements.txt**：

```
fastapi==0.104.1
uvicorn[standard]==0.24.0
transformers==4.37.0
torch==2.1.2
pydantic==2.5.0
```

**构建与运行**：

```bash
# 构建镜像
docker build -t transformers-api:v1.0 .

# 运行容器
docker run -d \
  --name transformers-api \
  -p 8000:8000 \
  --memory=4g \
  --cpus=2 \
  transformers-api:v1.0

# 查看日志
docker logs -f transformers-api
```

---

### 22.2.2 多阶段构建

**优化镜像大小**：

```dockerfile
# Stage 1: 构建阶段
FROM python:3.10 AS builder

WORKDIR /build

# 安装构建依赖
RUN pip install --upgrade pip setuptools wheel

# 复制依赖并安装
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: 运行阶段
FROM python:3.10-slim

WORKDIR /app

# 复制已安装的包
COPY --from=builder /root/.local /root/.local

# 复制应用代码
COPY . .

# 确保 Python 脚本在 PATH 中
ENV PATH=/root/.local/bin:$PATH

# 预下载模型
ENV TRANSFORMERS_CACHE=/app/models
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**镜像大小对比**：

| 方法 | 镜像大小 | 构建时间 |
|-----|---------|---------|
| 单阶段构建 | 2.8 GB | 8 min |
| 多阶段构建 | 1.6 GB (-43%) | 6 min |

---

### 22.2.3 CUDA 镜像选择

**GPU 推理 Dockerfile**：

```dockerfile
# 使用 NVIDIA 官方 CUDA 镜像
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 安装 Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 安装 PyTorch（CUDA 版本）
RUN pip3 install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

# 预下载模型
RUN python3 -c "from transformers import pipeline; pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english', device=0)"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**运行 GPU 容器**：

```bash
# 需要 NVIDIA Container Toolkit
docker run -d \
  --name transformers-gpu \
  --gpus all \
  -p 8000:8000 \
  transformers-api:gpu-v1.0

# 或指定 GPU
docker run -d \
  --gpus '"device=0,1"' \
  -p 8000:8000 \
  transformers-api:gpu-v1.0
```

---

### 22.2.4 模型缓存优化

**使用 Docker Volume 缓存模型**：

```bash
# 创建模型缓存卷
docker volume create transformers-models

# 运行容器，挂载卷
docker run -d \
  --name transformers-api \
  -v transformers-models:/root/.cache/huggingface \
  -p 8000:8000 \
  transformers-api:v1.0
```

**Dockerfile 优化**（利用层缓存）：

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 1. 先复制依赖文件（变化少）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. 预下载模型（变化少）
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')"

# 3. 最后复制代码（变化频繁）
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 22.3 Kubernetes 部署

### 22.3.1 Deployment YAML 配置

<div data-component="K8sDeploymentVisualizer"></div>

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: transformers-api
  labels:
    app: transformers-api
spec:
  replicas: 3  # 3 个副本
  selector:
    matchLabels:
      app: transformers-api
  template:
    metadata:
      labels:
        app: transformers-api
    spec:
      containers:
      - name: api
        image: transformers-api:v1.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "1000m"
            memory: "2Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        env:
        - name: TRANSFORMERS_CACHE
          value: "/models"
        volumeMounts:
        - name: models-cache
          mountPath: /models
      volumes:
      - name: models-cache
        persistentVolumeClaim:
          claimName: transformers-models-pvc

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: transformers-api-service
spec:
  selector:
    app: transformers-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

**部署到 Kubernetes**：

```bash
# 应用配置
kubectl apply -f deployment.yaml

# 查看 Pods
kubectl get pods -l app=transformers-api

# 查看服务
kubectl get svc transformers-api-service

# 查看日志
kubectl logs -f deployment/transformers-api
```

---

### 22.3.2 GPU 资源请求

**GPU Deployment**：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: transformers-gpu-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: transformers-gpu-api
  template:
    metadata:
      labels:
        app: transformers-gpu-api
    spec:
      containers:
      - name: api
        image: transformers-api:gpu-v1.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "2000m"
            memory: "8Gi"
            nvidia.com/gpu: 1  # 请求 1 个 GPU
          limits:
            cpu: "4000m"
            memory: "16Gi"
            nvidia.com/gpu: 1
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
      nodeSelector:
        accelerator: nvidia-tesla-t4  # 选择有 T4 GPU 的节点
```

**前置条件**：
1. 安装 NVIDIA Device Plugin：
   ```bash
   kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/nvidia-device-plugin.yml
   ```

2. 验证 GPU 可用：
   ```bash
   kubectl get nodes "-o=custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"
   ```

---

### 22.3.3 自动扩缩容（HPA）

**水平 Pod 自动扩缩器**：

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: transformers-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: transformers-api
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
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # 缩容冷却 5 分钟
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 2
        periodSeconds: 30
```

**自定义指标扩缩容**（基于请求队列长度）：

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: transformers-api-custom-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: transformers-api
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Pods
    pods:
      metric:
        name: queue_length
      target:
        type: AverageValue
        averageValue: "10"  # 平均队列长度 > 10 时扩容
```

---

### 22.3.4 模型版本管理

**使用 ConfigMap 管理模型版本**：

```yaml
# model-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
data:
  model_name: "distilbert-base-uncased-finetuned-sst-2-english"
  model_version: "v2.0"
  max_length: "512"
  batch_size: "8"
```

**在 Deployment 中引用**：

```yaml
spec:
  containers:
  - name: api
    image: transformers-api:v1.0
    envFrom:
    - configMapRef:
        name: model-config
```

**滚动更新模型**：

```bash
# 更新 ConfigMap
kubectl create configmap model-config \
  --from-literal=model_name="bert-base-uncased" \
  --from-literal=model_version="v3.0" \
  --dry-run=client -o yaml | kubectl apply -f -

# 重启 Deployment（触发滚动更新）
kubectl rollout restart deployment/transformers-api

# 查看更新状态
kubectl rollout status deployment/transformers-api

# 回滚
kubectl rollout undo deployment/transformers-api
```

---

## 22.4 监控与日志

### 22.4.1 Prometheus 指标暴露

**添加 Prometheus 客户端**：

```python
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
import time

app = FastAPI()

# 自定义指标
prediction_counter = Counter(
    "predictions_total",
    "Total number of predictions",
    ["model", "status"]
)

prediction_latency = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    ["model"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# 自动化指标收集
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

@app.post("/predict")
async def predict(input: TextInput):
    start_time = time.time()
    
    try:
        result = await async_predict(input.text)
        
        # 记录成功指标
        prediction_counter.labels(model="distilbert-sst2", status="success").inc()
        
        return result
    
    except Exception as e:
        # 记录失败指标
        prediction_counter.labels(model="distilbert-sst2", status="error").inc()
        raise
    
    finally:
        # 记录延迟
        latency = time.time() - start_time
        prediction_latency.labels(model="distilbert-sst2").observe(latency)
```

**Prometheus 配置**：

```yaml
# prometheus.yaml
scrape_configs:
  - job_name: 'transformers-api'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        regex: transformers-api
        action: keep
      - source_labels: [__meta_kubernetes_pod_ip]
        target_label: __address__
        replacement: ${1}:8000
      - target_label: __metrics_path__
        replacement: /metrics
```

---

### 22.4.2 Grafana 可视化

**示例仪表板查询**：

```promql
# QPS（每秒请求数）
rate(http_requests_total{job="transformers-api"}[1m])

# P50 延迟
histogram_quantile(0.5, 
  rate(prediction_latency_seconds_bucket[5m])
)

# P99 延迟
histogram_quantile(0.99, 
  rate(prediction_latency_seconds_bucket[5m])
)

# 错误率
rate(predictions_total{status="error"}[1m]) 
/ 
rate(predictions_total[1m])

# GPU 利用率（需要 DCGM Exporter）
DCGM_FI_DEV_GPU_UTIL{exported_pod=~"transformers-gpu-api.*"}
```

---

### 22.4.3 日志聚合（ELK）

**结构化日志**：

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        
        if hasattr(record, "request_id"):
            log_obj["request_id"] = record.request_id
        
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_obj)

# 配置日志
logger = logging.getLogger("transformers-api")
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

@app.post("/predict")
async def predict(input: TextInput):
    request_id = str(uuid.uuid4())
    
    logger.info(
        "Prediction request received",
        extra={"request_id": request_id, "text_length": len(input.text)}
    )
    
    result = await async_predict(input.text)
    
    logger.info(
        "Prediction completed",
        extra={"request_id": request_id, "label": result["label"]}
    )
    
    return result
```

**Filebeat 配置**（收集日志到 Elasticsearch）：

```yaml
# filebeat.yaml
filebeat.inputs:
  - type: container
    paths:
      - /var/lib/docker/containers/*/*.log
    json.keys_under_root: true
    json.add_error_key: true

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "transformers-api-%{+yyyy.MM.dd}"
```

---

### 22.4.4 链路追踪（Jaeger）

**OpenTelemetry 集成**：

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# 配置 Tracer
trace.set_tracer_provider(TracerProvider())
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# 自动化追踪
FastAPIInstrumentor.instrument_app(app)

tracer = trace.get_tracer(__name__)

@app.post("/predict")
async def predict(input: TextInput):
    with tracer.start_as_current_span("predict") as span:
        span.set_attribute("text_length", len(input.text))
        
        with tracer.start_as_current_span("tokenization"):
            inputs = tokenizer(input.text, return_tensors="pt")
        
        with tracer.start_as_current_span("model_inference"):
            outputs = model(**inputs)
        
        with tracer.start_as_current_span("postprocessing"):
            result = process_outputs(outputs)
        
        span.set_attribute("prediction_label", result["label"])
        
        return result
```

---

## 22.5 安全性考虑

### 22.5.1 输入验证与过滤

**防止注入攻击**：

```python
from pydantic import BaseModel, Field, validator
import re

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    
    @validator("text")
    def sanitize_text(cls, v):
        # 移除控制字符
        v = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', v)
        
        # 限制特殊字符
        if re.search(r'<script|javascript:|onerror=', v, re.IGNORECASE):
            raise ValueError("Potentially malicious input detected")
        
        return v.strip()

@app.post("/predict")
async def predict(input: TextInput):
    # input.text 已经过验证和清理
    result = await async_predict(input.text)
    return result
```

---

### 22.5.2 Rate Limiting

**使用 slowapi 限流**：

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("10/minute")  # 每分钟 10 次
async def predict(request: Request, input: TextInput):
    result = await async_predict(input.text)
    return result

# 不同用户等级不同限制
@app.post("/predict_premium")
@limiter.limit("100/minute")
async def predict_premium(request: Request, input: TextInput):
    result = await async_predict(input.text)
    return result
```

**Redis 分布式限流**：

```python
import redis
from fastapi import HTTPException

redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

async def check_rate_limit(user_id: str, limit: int = 100, window: int = 60):
    """基于 Redis 的滑动窗口限流"""
    key = f"rate_limit:{user_id}"
    current_time = time.time()
    
    # 移除过期记录
    redis_client.zremrangebyscore(key, 0, current_time - window)
    
    # 检查当前窗口内的请求数
    request_count = redis_client.zcard(key)
    
    if request_count >= limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # 添加当前请求
    redis_client.zadd(key, {str(uuid.uuid4()): current_time})
    redis_client.expire(key, window)

@app.post("/predict")
async def predict(input: TextInput, user_id: str = Header(...)):
    await check_rate_limit(user_id, limit=100, window=60)
    result = await async_predict(input.text)
    return result
```

---

### 22.5.3 认证与授权（JWT）

**JWT Token 验证**：

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

security = HTTPBearer()

def create_token(user_id: str, role: str = "user") -> str:
    """创建 JWT Token"""
    payload = {
        "user_id": user_id,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """验证 JWT Token"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/predict")
async def predict(
    input: TextInput,
    token_data: dict = Depends(verify_token)
):
    """需要认证的预测端点"""
    user_id = token_data["user_id"]
    
    logger.info(f"Prediction request from user {user_id}")
    
    result = await async_predict(input.text)
    return result

@app.post("/login")
async def login(username: str, password: str):
    """登录获取 Token"""
    # 验证用户名密码（示例）
    if username == "admin" and password == "secret":
        token = create_token(user_id=username, role="admin")
        return {"access_token": token, "token_type": "bearer"}
    
    raise HTTPException(status_code=401, detail="Invalid credentials")
```

---

### 22.5.4 模型水印

**输出文本水印**（防止滥用）：

```python
import hashlib
from typing import List

class TextWatermark:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def embed(self, text: str, user_id: str) -> str:
        """在生成文本中嵌入不可见水印"""
        # 使用零宽字符嵌入用户 ID
        watermark = self._generate_watermark(user_id)
        
        # 在句子末尾插入零宽字符
        watermarked_text = text
        for i, char in enumerate(watermark):
            if i < len(text):
                watermarked_text = watermarked_text[:i+1] + char + watermarked_text[i+1:]
        
        return watermarked_text
    
    def _generate_watermark(self, user_id: str) -> str:
        """生成零宽字符水印"""
        hash_value = hashlib.sha256(
            (user_id + self.secret_key).encode()
        ).hexdigest()
        
        # 转换为零宽字符
        zero_width_chars = ['\u200b', '\u200c', '\u200d']
        watermark = ""
        
        for char in hash_value[:16]:  # 取前 16 位
            idx = int(char, 16) % len(zero_width_chars)
            watermark += zero_width_chars[idx]
        
        return watermark
    
    def extract(self, text: str) -> str:
        """从文本中提取水印"""
        zero_width_chars = ['\u200b', '\u200c', '\u200d']
        watermark = ""
        
        for char in text:
            if char in zero_width_chars:
                watermark += str(zero_width_chars.index(char))
        
        return watermark

watermarker = TextWatermark(secret_key="your-secret-key")

@app.post("/generate")
async def generate_text(
    input: TextInput,
    user_id: str = Depends(get_current_user_id)
):
    """带水印的文本生成"""
    generated_text = await async_generate(input.text)
    
    # 嵌入水印
    watermarked_text = watermarker.embed(generated_text, user_id)
    
    return {"text": watermarked_text, "user_id": user_id}
```

---

## 22.6 实战案例：完整生产部署

**项目结构**：

```
transformers-api/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI 应用
│   ├── models.py        # Pydantic 模型
│   ├── inference.py     # 推理引擎
│   └── auth.py          # 认证逻辑
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   └── configmap.yaml
├── monitoring/
│   ├── prometheus.yaml
│   └── grafana-dashboard.json
├── Dockerfile
├── requirements.txt
└── README.md
```

**完整部署流程**：

```bash
# 1. 构建镜像
docker build -t myregistry.com/transformers-api:v1.0 .
docker push myregistry.com/transformers-api:v1.0

# 2. 创建 Kubernetes 资源
kubectl create namespace transformers
kubectl apply -f k8s/ -n transformers

# 3. 部署 Prometheus 和 Grafana
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring

# 4. 验证部署
kubectl get pods -n transformers
kubectl get svc -n transformers

# 5. 测试 API
export API_URL=$(kubectl get svc transformers-api-service -n transformers -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

curl -X POST http://$API_URL/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"text": "This is amazing!"}'
```

---

## 22.7 总结

### 核心要点

1. **API 设计**：
   - 异步推理提升并发性能（3-4x）
   - 批处理增加吞吐量（4-5x）
   - 请求队列管理平衡延迟和吞吐量

2. **Docker 化**：
   - 多阶段构建减小镜像（-40%）
   - 模型缓存优化构建速度
   - GPU 镜像选择合适的 CUDA 版本

3. **Kubernetes 部署**：
   - HPA 自动扩缩容应对流量波动
   - GPU 资源请求与调度
   - 滚动更新零停机部署

4. **监控**：
   - Prometheus + Grafana 可视化
   - 结构化日志便于查询
   - 分布式追踪定位性能瓶颈

5. **安全**：
   - 输入验证防注入
   - Rate Limiting 防滥用
   - JWT 认证授权
   - 模型水印追踪

---

## 22.8 扩展阅读

1. **FastAPI 官方文档**：https://fastapi.tiangolo.com/
2. **Kubernetes 官方文档**：https://kubernetes.io/docs/
3. **Prometheus 监控指南**：https://prometheus.io/docs/
4. **NVIDIA GPU Operator**：https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/
5. **OpenTelemetry Python**：https://opentelemetry.io/docs/instrumentation/python/

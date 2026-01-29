# Chapter 27: 容器化与云部署

在 Chapter 26 中，我们掌握了 LangServe 的高级特性，能够构建生产级的 API 服务。然而，要将应用真正部署到云端并实现高可用、可扩展，还需要掌握容器化技术和云原生部署方案。本章将深入探讨 Docker 容器化、Kubernetes 编排、以及在 AWS、GCP、Azure 等主流云平台的部署最佳实践。

## 27.1 为什么需要容器化

### 27.1.1 传统部署的痛点

在容器化之前，应用部署面临诸多挑战：

- **环境不一致**："在我机器上能跑" —— 开发环境与生产环境配置差异导致的部署失败
- **依赖冲突**：不同应用需要不同版本的 Python、库、系统依赖
- **资源隔离不足**：多个应用共享同一服务器，资源竞争和安全隔离困难
- **部署复杂**：手动配置环境、安装依赖、启动服务，容易出错且难以复现
- **扩展性差**：横向扩展需要手动复制服务器和配置

### 27.1.2 Docker 的核心优势

Docker 通过容器技术解决了这些问题：

```
应用 + 依赖 + 配置 → 打包成镜像 → 在任何支持 Docker 的环境中一致运行
```

**核心价值**：

1. **一致性**：开发、测试、生产环境完全一致
2. **可移植性**：一次构建，到处运行（云平台、本地、边缘设备）
3. **轻量级**：相比虚拟机，容器共享宿主机内核，启动快、资源占用少
4. **版本化**：镜像可以版本管理，支持回滚
5. **自动化**：与 CI/CD 无缝集成，自动构建、测试、部署

### 27.1.3 容器 vs 虚拟机

$$
\begin{align*}
\text{虚拟机} &: \text{App} \rightarrow \text{Guest OS} \rightarrow \text{Hypervisor} \rightarrow \text{Host OS} \rightarrow \text{Hardware} \\
\text{容器} &: \text{App} \rightarrow \text{Container Runtime} \rightarrow \text{Host OS} \rightarrow \text{Hardware}
\end{align*}
$$

| 特性 | 虚拟机 | Docker 容器 |
|------|--------|-------------|
| **启动时间** | 分钟级 | 秒级 |
| **资源占用** | GB 级别（需要完整 OS） | MB 级别（共享内核） |
| **隔离性** | 强（硬件级别） | 中等（进程级别） |
| **性能** | 约 90% 原生性能 | 接近 100% 原生性能 |
| **适用场景** | 需要强隔离、多 OS | 微服务、快速部署 |

---

## 27.2 为 LangServe 应用编写 Dockerfile

### 27.2.1 基础 Dockerfile

从零开始构建 LangServe 应用的镜像：

```dockerfile
# 使用官方 Python 基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（如果需要）
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**构建镜像**：

```bash
docker build -t langserve-app:latest .
```

**运行容器**：

```bash
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=sk-your-key \
  --name langserve \
  langserve-app:latest
```

### 27.2.2 多阶段构建优化镜像大小

生产环境需要最小化镜像大小以加快部署速度：

```dockerfile
# ========== 构建阶段 ==========
FROM python:3.11-slim AS builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖到特定目录
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ========== 运行阶段 ==========
FROM python:3.11-slim

WORKDIR /app

# 只复制必要的运行时依赖
COPY --from=builder /root/.local /root/.local
COPY . .

# 确保 pip 安装的可执行文件在 PATH 中
ENV PATH=/root/.local/bin:$PATH

# 创建非 root 用户（安全最佳实践）
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# 使用 exec 形式的 CMD（更好的信号处理）
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**镜像大小对比**：

```bash
# 单阶段构建
langserve-app:basic       850MB

# 多阶段构建
langserve-app:optimized   320MB

# 进一步优化（使用 alpine）
langserve-app:alpine      180MB
```

### 27.2.3 使用 .dockerignore 排除不必要文件

创建 `.dockerignore` 文件：

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Git
.git/
.gitignore

# 测试和文档
tests/
docs/
*.md
.coverage

# 日志和临时文件
*.log
tmp/
.cache/

# 敏感文件
.env
*.key
*.pem
```

### 27.2.4 健康检查

添加健康检查以便容器编排系统监控应用状态：

```dockerfile
FROM python:3.11-slim

# ... 前面的步骤 ...

# 安装 curl 用于健康检查
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

在应用中添加健康检查端点：

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/ready")
async def readiness_check():
    """就绪检查（检查依赖服务）"""
    # 检查数据库、Redis、外部 API 等
    try:
        # redis_client.ping()
        # db.execute("SELECT 1")
        return {"status": "ready"}
    except Exception as e:
        return {"status": "not_ready", "error": str(e)}, 503
```

<DockerBuildFlow />

### 27.2.5 完整的生产级 Dockerfile 示例

综合所有最佳实践：

```dockerfile
# syntax=docker/dockerfile:1

# ========== 构建阶段 ==========
FROM python:3.11-slim AS builder

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 安装编译依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制并安装依赖
COPY requirements.txt .
RUN pip install --user --no-warn-script-location -r requirements.txt

# ========== 运行阶段 ==========
FROM python:3.11-slim

# 元数据标签
LABEL maintainer="your-email@example.com" \
      version="1.0" \
      description="LangServe Translation API"

ENV PYTHONUNBUFFERED=1 \
    PATH=/root/.local/bin:$PATH \
    PORT=8000

WORKDIR /app

# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制依赖
COPY --from=builder /root/.local /root/.local

# 复制应用代码
COPY main.py .
COPY chains/ ./chains/
COPY config/ ./config/

# 创建非 root 用户
RUN groupadd -r appgroup && \
    useradd -r -g appgroup -u 1000 appuser && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE ${PORT}

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# 使用 gunicorn 作为生产服务器
CMD exec gunicorn main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:${PORT} \
    --timeout 120 \
    --keep-alive 5 \
    --access-logfile - \
    --error-logfile - \
    --log-level info
```

**requirements.txt**：

```
fastapi==0.104.1
uvicorn[standard]==0.24.0
gunicorn==21.2.0
langchain==0.1.0
langchain-openai==0.0.2
langserve==0.0.40
pydantic==2.5.0
python-dotenv==1.0.0
redis==5.0.1
```

---

## 27.3 Docker Compose：多容器编排

对于需要多个服务（应用 + Redis + Postgres）的场景，使用 Docker Compose：

### 27.3.1 基础 docker-compose.yml

```yaml
version: '3.8'

services:
  # LangServe 应用
  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_HOST=redis
      - POSTGRES_HOST=postgres
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
    networks:
      - langserve-network

  # Redis 缓存
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped
    networks:
      - langserve-network

  # PostgreSQL 数据库
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=langserve
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-changeme}
      - POSTGRES_DB=langserve_db
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: unless-stopped
    networks:
      - langserve-network

volumes:
  redis-data:
  postgres-data:

networks:
  langserve-network:
    driver: bridge
```

**启动所有服务**：

```bash
# 启动
docker-compose up -d

# 查看日志
docker-compose logs -f app

# 停止
docker-compose down

# 停止并删除数据卷
docker-compose down -v
```

### 27.3.2 添加 Prometheus 和 Grafana 监控

```yaml
version: '3.8'

services:
  app:
    # ... （同上）
    labels:
      - "prometheus.scrape=true"
      - "prometheus.port=8000"
      - "prometheus.path=/metrics"

  redis:
    # ... （同上）

  postgres:
    # ... （同上）

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped
    networks:
      - langserve-network

  # Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
    volumes:
      - grafana-data:/var/lib/grafana
    depends_on:
      - prometheus
    restart: unless-stopped
    networks:
      - langserve-network

volumes:
  redis-data:
  postgres-data:
  prometheus-data:
  grafana-data:

networks:
  langserve-network:
    driver: bridge
```

**prometheus.yml**：

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'langserve'
    static_configs:
      - targets: ['app:8000']
```

访问：
- LangServe API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

---

## 27.4 Kubernetes 部署

对于生产环境，Kubernetes 提供了更强大的编排能力：自动扩缩容、滚动更新、服务发现、负载均衡等。

### 27.4.1 核心概念

- **Pod**：最小部署单元，包含一个或多个容器
- **Deployment**：管理 Pod 的副本数量和更新策略
- **Service**：为 Pod 提供稳定的网络访问入口
- **ConfigMap**：配置数据（非敏感）
- **Secret**：敏感数据（API Key、密码）
- **Ingress**：HTTP/HTTPS 路由规则

### 27.4.2 Deployment 配置

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langserve-app
  labels:
    app: langserve
spec:
  replicas: 3  # 运行 3 个副本
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
        image: your-registry.com/langserve-app:1.0.0
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: langserve-secrets
              key: openai-api-key
        - name: REDIS_HOST
          value: redis-service
        - name: PORT
          value: "8000"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
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

### 27.4.3 Service 配置

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: langserve-service
spec:
  selector:
    app: langserve
  ports:
  - name: http
    port: 80
    targetPort: 8000
  type: LoadBalancer  # 在云平台会自动创建负载均衡器
```

### 27.4.4 ConfigMap 和 Secret

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: langserve-config
data:
  LOG_LEVEL: "info"
  MAX_WORKERS: "4"
  TIMEOUT: "120"

---
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: langserve-secrets
type: Opaque
data:
  # 值必须是 base64 编码
  # echo -n 'sk-your-openai-key' | base64
  openai-api-key: c2steW91ci1vcGVuYWkta2V5
```

创建 Secret（推荐方式）：

```bash
kubectl create secret generic langserve-secrets \
  --from-literal=openai-api-key=sk-your-actual-key
```

### 27.4.5 Ingress 配置（HTTPS 支持）

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: langserve-ingress
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.example.com
    secretName: langserve-tls
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: langserve-service
            port:
              number: 80
```

### 27.4.6 水平自动扩缩容 (HPA)

根据 CPU 使用率自动调整 Pod 数量：

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: langserve-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: langserve-app
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70  # 当 CPU > 70% 时扩容
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

**部署所有资源**：

```bash
# 应用配置
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml

# 查看状态
kubectl get pods
kubectl get svc
kubectl get ing
kubectl get hpa

# 查看日志
kubectl logs -f deployment/langserve-app

# 进入容器调试
kubectl exec -it deployment/langserve-app -- /bin/bash
```

<KubernetesArchitecture />

---

## 27.5 云平台部署

### 27.5.1 AWS 部署方案

#### 选项 1：AWS ECS (Elastic Container Service)

适合快速部署，无需管理 Kubernetes 集群：

```bash
# 1. 创建 ECR 仓库
aws ecr create-repository --repository-name langserve-app

# 2. 推送镜像
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

docker tag langserve-app:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/langserve-app:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/langserve-app:latest

# 3. 创建 ECS 任务定义 (task-definition.json)
{
  "family": "langserve-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [
    {
      "name": "langserve",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/langserve-app:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "OPENAI_API_KEY",
          "value": "{{resolve:secretsmanager:langserve/openai:SecretString:api_key}}"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/langserve",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}

# 4. 创建 ECS 服务
aws ecs create-service \
  --cluster langserve-cluster \
  --service-name langserve-service \
  --task-definition langserve-task \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=langserve,containerPort=8000"
```

#### 选项 2：AWS EKS (Elastic Kubernetes Service)

适合需要完整 Kubernetes 功能的场景：

```bash
# 1. 创建 EKS 集群
eksctl create cluster \
  --name langserve-cluster \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type t3.medium \
  --nodes 3 \
  --nodes-min 2 \
  --nodes-max 5 \
  --managed

# 2. 配置 kubectl
aws eks update-kubeconfig --region us-east-1 --name langserve-cluster

# 3. 部署应用（使用前面的 YAML 文件）
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml
```

### 27.5.2 Google Cloud Platform (GCP) 部署

#### GKE (Google Kubernetes Engine)

```bash
# 1. 创建 GKE 集群
gcloud container clusters create langserve-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-2 \
  --enable-autoscaling \
  --min-nodes 2 \
  --max-nodes 10

# 2. 配置 kubectl
gcloud container clusters get-credentials langserve-cluster --zone us-central1-a

# 3. 推送镜像到 GCR
docker tag langserve-app:latest gcr.io/[PROJECT-ID]/langserve-app:latest
docker push gcr.io/[PROJECT-ID]/langserve-app:latest

# 4. 部署应用
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

### 27.5.3 Azure 部署

#### AKS (Azure Kubernetes Service)

```bash
# 1. 创建资源组
az group create --name langserve-rg --location eastus

# 2. 创建 AKS 集群
az aks create \
  --resource-group langserve-rg \
  --name langserve-cluster \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# 3. 连接到集群
az aks get-credentials --resource-group langserve-rg --name langserve-cluster

# 4. 推送镜像到 ACR
az acr create --resource-group langserve-rg --name langserveacr --sku Basic
az acr login --name langserveacr

docker tag langserve-app:latest langserveacr.azurecr.io/langserve-app:latest
docker push langserveacr.azurecr.io/langserve-app:latest

# 5. 部署应用
kubectl apply -f deployment.yaml
```

<CloudPlatformComparison />

---

## 27.6 CI/CD 自动化部署

### 27.6.1 GitHub Actions 工作流

创建 `.github/workflows/deploy.yml`：

```yaml
name: Build and Deploy

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: pytest tests/ --cov=./

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=sha,prefix={{branch}}-
            type=semver,pattern={{version}}
      
      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}

  deploy-to-k8s:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up kubectl
        uses: azure/setup-kubectl@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Update kubeconfig
        run: |
          aws eks update-kubeconfig --region us-east-1 --name langserve-cluster
      
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/langserve-app \
            langserve=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:main-${{ github.sha }}
          kubectl rollout status deployment/langserve-app
```

### 27.6.2 GitLab CI/CD

`.gitlab-ci.yml`：

```yaml
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  IMAGE_TAG: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA

test:
  stage: test
  image: python:3.11
  script:
    - pip install -r requirements.txt
    - pip install pytest pytest-cov
    - pytest tests/ --cov=./
  only:
    - merge_requests
    - main

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $IMAGE_TAG .
    - docker push $IMAGE_TAG
    - docker tag $IMAGE_TAG $CI_REGISTRY_IMAGE:latest
    - docker push $CI_REGISTRY_IMAGE:latest
  only:
    - main

deploy:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl config set-cluster k8s --server="$KUBE_URL" --insecure-skip-tls-verify=true
    - kubectl config set-credentials admin --token="$KUBE_TOKEN"
    - kubectl config set-context default --cluster=k8s --user=admin
    - kubectl config use-context default
    - kubectl set image deployment/langserve-app langserve=$IMAGE_TAG
    - kubectl rollout status deployment/langserve-app
  only:
    - main
  when: manual  # 需要手动触发
```

---

## 27.7 监控与可观测性

### 27.7.1 Prometheus + Grafana 监控栈

在 Kubernetes 中部署完整监控方案：

```bash
# 使用 Helm 安装 kube-prometheus-stack
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false
```

为 LangServe 应用添加 ServiceMonitor：

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: langserve-metrics
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: langserve
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
```

### 27.7.2 日志聚合 (ELK / Loki)

使用 Loki + Promtail 收集日志：

```bash
helm install loki grafana/loki-stack \
  --namespace monitoring \
  --set grafana.enabled=true \
  --set prometheus.enabled=false \
  --set promtail.enabled=true
```

在 Grafana 中查询日志：

```logql
{app="langserve"} |= "error" | json
```

### 27.7.3 分布式追踪 (Jaeger)

```bash
kubectl create ns observability

kubectl apply -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/main/deploy/crds/jaegertracing.io_jaegers_crd.yaml
kubectl apply -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/main/deploy/service_account.yaml
kubectl apply -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/main/deploy/role.yaml
kubectl apply -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/main/deploy/role_binding.yaml
kubectl apply -f https://raw.githubusercontent.com/jaegertracing/jaeger-operator/main/deploy/operator.yaml

# 创建 Jaeger 实例
kubectl apply -f - <<EOF
apiVersion: jaegertracing.io/v1
kind: Jaeger
metadata:
  name: jaeger
  namespace: observability
spec:
  strategy: production
  storage:
    type: elasticsearch
EOF
```

---

## 27.8 安全加固

### 27.8.1 镜像安全扫描

使用 Trivy 扫描漏洞：

```bash
# 安装 Trivy
brew install trivy

# 扫描镜像
trivy image langserve-app:latest

# 在 CI/CD 中集成
- name: Scan image for vulnerabilities
  run: |
    trivy image --severity HIGH,CRITICAL --exit-code 1 ${{ env.IMAGE_TAG }}
```

### 27.8.2 Kubernetes 安全策略

**Pod Security Policy**：

```yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: restricted
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'secret'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

**Network Policy**：

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: langserve-network-policy
spec:
  podSelector:
    matchLabels:
      app: langserve
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          role: frontend
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to:  # 允许访问外部 API (OpenAI)
    ports:
    - protocol: TCP
      port: 443
```

---

## 27.9 成本优化

### 27.9.1 使用 Spot / Preemptible 实例

**AWS EKS**：

```bash
eksctl create nodegroup \
  --cluster langserve-cluster \
  --name spot-workers \
  --node-type t3.medium \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 5 \
  --spot
```

**GKE**：

```bash
gcloud container node-pools create spot-pool \
  --cluster langserve-cluster \
  --zone us-central1-a \
  --spot \
  --machine-type n1-standard-2 \
  --num-nodes 2
```

### 27.9.2 Vertical Pod Autoscaler (VPA)

自动调整资源请求：

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: langserve-vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: langserve-app
  updatePolicy:
    updateMode: "Auto"  # 自动调整资源配置
```

### 27.9.3 集群自动缩放

```bash
# EKS
eksctl create cluster \
  --asg-access \
  --enable-cluster-autoscaler

# GKE（默认启用）
gcloud container clusters create ... --enable-autoscaling
```

---

## 27.10 本章总结

本章深入探讨了 LangChain 应用的容器化与云部署：

### 核心要点

1. **Docker 容器化**：
   - 多阶段构建优化镜像大小
   - 健康检查保障服务可用性
   - 非 root 用户运行提升安全性

2. **Kubernetes 编排**：
   - Deployment 管理副本和更新
   - Service 提供稳定网络访问
   - HPA 实现自动扩缩容
   - ConfigMap/Secret 分离配置

3. **云平台部署**：
   - AWS ECS/EKS：成熟稳定，与 AWS 生态集成紧密
   - GCP GKE：性能优异，运维简单
   - Azure AKS：企业级支持，混合云友好

4. **CI/CD 自动化**：
   - GitHub Actions/GitLab CI 实现自动测试、构建、部署
   - 镜像扫描防止安全漏洞
   - 滚动更新保证零停机部署

5. **监控可观测性**：
   - Prometheus + Grafana 监控关键指标
   - Loki 聚合日志
   - Jaeger 分布式追踪

6. **安全与成本**：
   - Pod Security Policy 限制权限
   - Network Policy 隔离网络
   - Spot 实例降低成本
   - VPA 优化资源配置

### 生产部署检查清单

- [ ] ✅ 多阶段 Dockerfile 优化镜像大小
- [ ] ✅ 健康检查和就绪探针
- [ ] ✅ 资源限制和请求配置
- [ ] ✅ Secret 管理敏感数据
- [ ] ✅ HPA 自动扩缩容
- [ ] ✅ Ingress HTTPS 配置
- [ ] ✅ Prometheus 监控
- [ ] ✅ 日志聚合
- [ ] ✅ 镜像安全扫描
- [ ] ✅ Network Policy 网络隔离
- [ ] ✅ CI/CD 自动化部署
- [ ] ✅ 灾难恢复计划

### 下一步

Chapter 28 将学习**高级 RAG 技术**，掌握混合检索、重排序、查询改写、自适应检索等技术，构建企业级知识库问答系统。

### 扩展资源

- [Docker 官方文档](https://docs.docker.com/)
- [Kubernetes 官方文档](https://kubernetes.io/docs/)
- [AWS EKS 最佳实践](https://aws.github.io/aws-eks-best-practices/)
- [12-Factor App](https://12factor.net/)
- [CNCF Landscape](https://landscape.cncf.io/)

---
title: "第27章：生产级部署 — 从原型到上线"
description: "掌握 Agent 生产部署全流程：Docker 容器化、Kubernetes 编排、API 网关、缓存策略与成本优化。"
date: "2026-06-11"
---

# 第27章：生产级部署 — 从原型到上线

---

## 27.1 Docker 容器化

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 27.2 FastAPI Agent 服务

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class AgentRequest(BaseModel):
    message: str
    session_id: str

@app.post("/agent/chat")
async def chat(request: AgentRequest):
    result = await agent.run(request.message, request.session_id)
    return {"response": result["answer"], "tools_used": result["tools"]}
```

---

## 27.3 缓存策略

```python
import redis, hashlib, json

class AgentCache:
    def __init__(self, redis_client, ttl=3600):
        self.redis = redis_client; self.ttl = ttl
    def get(self, query):
        key = f"agent_cache:{hashlib.md5(query.encode()).hexdigest()}"
        result = self.redis.get(key)
        return json.loads(result) if result else None
    def set(self, query, response):
        key = f"agent_cache:{hashlib.md5(query.encode()).hexdigest()}"
        self.redis.setex(key, self.ttl, json.dumps(response))
```

---

## 27.4 成本优化

| 策略 | 节省比例 | 实现方式 |
|:---|:---|:---|
| 缓存 | 30-50% | 相似查询复用 |
| 模型路由 | 40-60% | 简单问题用小模型 |
| Prompt 压缩 | 20-30% | 减少上下文长度 |

---

## 27.5 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| 容器化 | Docker + K8s |
| API 服务 | FastAPI |
| 缓存 | Redis |
| 模型路由 | 根据复杂度选模型 |

> **下一章预告**
>
> 在第 28 章中，我们将学习 Agent 可观测性。

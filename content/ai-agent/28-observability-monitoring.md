---
title: "第28章：可观测性与监控 — Agent 的运维之眼"
description: "构建 Agent 可观测性体系：分布式追踪（LangSmith/Langfuse）、Prometheus 指标、日志聚合与告警规则。"
date: "2026-06-11"
---

# 第28章：可观测性与监控 — Agent 的运维之眼

---

## 28.1 可观测性三支柱

| 支柱 | 工具 | 用途 |
|:---|:---|:---|
| **Tracing** | LangSmith, Langfuse | 追踪执行链路 |
| **Metrics** | Prometheus, Grafana | 监控性能指标 |
| **Logging** | ELK Stack, Loki | 聚合分析日志 |

---

## 28.2 LangSmith 追踪

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_..."
os.environ["LANGCHAIN_PROJECT"] = "my-agent"

agent = create_react_agent(model=llm, tools=tools)
result = agent.invoke({"messages": [("user", "你好")]})
# 在 LangSmith UI 中查看追踪
```

---

## 28.3 Prometheus 指标

```python
from prometheus_client import Counter, Histogram

agent_requests = Counter('agent_requests_total', 'Total requests', ['status'])
agent_latency = Histogram('agent_latency_seconds', 'Request latency')

@app.post("/agent/chat")
async def chat(request):
    with agent_latency.time():
        result = await agent.run(request.message)
        agent_requests.labels(status='success').inc()
        return result
```

---

## 28.4 告警规则

```yaml
groups:
  - name: agent_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(agent_requests_total{status="error"}[5m]) > 0.1
        labels:
          severity: warning
```

---

## 28.5 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| 三支柱 | Tracing + Metrics + Logging |
| LangSmith | 自动追踪调用链路 |
| Prometheus | 性能指标监控 |

> **下一章预告**
>
> 在第 29 章中，我们将学习 MCP 与 A2A 协议。

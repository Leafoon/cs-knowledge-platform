---
title: "第28章：可观测性与监控 — Agent 的运维之眼"
description: "构建 Agent 可观测性体系：分布式追踪、日志聚合、指标监控、告警规则与调试工作流。"
date: "2026-06-11"
---


下面的交互式演示展示了 Agent 可观测性工具的对比：

<div data-component="AgentObservabilityTools"></div>

# 第28章：可观测性与监控 — Agent 的运维之眼

Agent 系统的复杂性远超传统应用——单次请求可能涉及多次 LLM 调用、工具执行、记忆检索和多 Agent 协作。没有可观测性，调试 Agent 就像"在黑暗中摸索"。本章系统讲解如何构建 Agent 的可观测性体系。

## 为什么 Agent 需要可观测性？

传统的 Web 应用通常遵循确定性的请求-响应模式，而 Agent 系统则完全不同。一个简单的用户查询可能触发以下复杂流程：

1. **LLM 理解意图**：解析用户输入，决定是否需要工具
2. **工具调用**：可能调用搜索引擎、数据库、API 等多个外部服务
3. **结果整合**：将多个工具的返回结果整合成连贯的回答
4. **多轮对话**：用户可能追问，Agent 需要维护上下文

在这个过程中，任何一个环节出问题都可能导致：
- 响应延迟（用户等待时间过长）
- 成本失控（Token 消耗超出预算）
- 质量下降（Agent 产生幻觉或错误回答）
- 安全风险（敏感信息泄露或注入攻击）

可观测性就是解决这些问题的关键。它让我们能够：
- **看到**系统的内部状态
- **理解**为什么会出问题
- **预测**潜在的风险
- **快速定位**和修复问题

---

## 28.1 可观测性三支柱

可观测性（Observability）是一个系统工程概念，指的是通过系统的外部输出来推断其内部状态的能力。在 Agent 系统中，可观测性由三大支柱组成，它们相互配合，提供完整的系统视图。

### 28.1.1 三大支柱详解

**1. 分布式追踪（Tracing）**

分布式追踪记录一个请求从开始到结束的完整路径。在 Agent 系统中，一个用户查询可能跨越多个服务和组件，追踪能够让我们看到：
- LLM 调用的顺序和耗时
- 工具调用的依赖关系
- 每个步骤的输入和输出
- 错误发生的具体位置

追踪的核心概念是 **Span**（跨度），它代表一个操作单元。多个 Span 组成一个 **Trace**（追踪），形成完整的调用链。

**2. 指标监控（Metrics）**

指标监控收集和聚合系统运行时的数值数据。对于 Agent 系统，关键指标包括：
- **请求量**：每秒处理多少请求
- **延迟**：请求的平均响应时间
- **错误率**：失败请求的比例
- **资源使用**：CPU、内存、网络带宽
- **业务指标**：Token 消耗、工具调用次数

指标帮助我们了解系统的整体健康状况和性能趋势。

**3. 日志记录（Logging）**

日志记录系统中发生的事件和状态变化。结构化的日志能够提供：
- **事件详情**：什么时间发生了什么
- **上下文信息**：哪个用户、哪个会话
- **错误堆栈**：出错时的完整调用栈
- **业务数据**：输入输出内容、处理结果

日志是排查问题时最重要的信息来源。

### 28.1.2 支柱对比

| 支柱 | 核心问题 | 数据类型 | 工具 | 采集方式 |
|:---|:---|:---|:---|:---|
| **Tracing** | 请求经历了什么？ | Span、Trace | LangSmith, Langfuse, Phoenix | 自动/手动埋点 |
| **Metrics** | 系统表现如何？ | 计数器、直方图、仪表 | Prometheus, Grafana, Datadog | 采样聚合 |
| **Logging** | 发生了什么事件？ | 结构化日志 | ELK Stack, Loki, Seq | 事件记录 |

### 28.1.2 三支柱如何协作

三大支柱不是孤立存在的，它们相互配合，形成完整的可观测性体系。当问题发生时，我们通常需要从一个支柱跳转到另一个支柱进行深入分析。

```
用户请求 → Tracing（记录完整调用链）
         → Metrics（统计性能指标）
         → Logging（记录关键事件）
```

**典型场景分析**：用户报告"Agent 回复很慢"

**第一步：Metrics 告警**
监控系统发现 P95 延迟从 2s 升到 10s，触发告警。这告诉我们"系统变慢了"，但不知道为什么。

**第二步：Tracing 定位**
通过追踪系统，我们找到一个具体的 Trace，发现整个请求耗时 12s，其中某次工具调用耗时 8s。这告诉我们"问题出在工具调用"。

**第三步：Logging 深入**
查看该工具的详细日志，发现错误信息："Database connection timeout after 30s"。这告诉我们"根本原因是数据库连接超时"。

**第四步：Metrics 验证**
检查数据库连接池的指标，发现连接数接近上限，验证了我们的假设。

通过这种"从宏观到微观"的分析路径，我们能够快速定位和解决问题。

### 28.1.3 Agent 特有的可观测性挑战

Agent 系统与传统 Web 应用有很大不同，这给可观测性带来了独特的挑战：

**1. 非确定性（Non-determinism）**

传统应用对于相同的输入会产生相同的输出，但 Agent 系统由于 LLM 的特性，同样的输入可能产生不同的调用链。这意味着：
- 不能简单地重放请求来复现问题
- 需要记录完整的上下文和随机种子
- 同一个问题可能需要不同的排查路径

**2. 多轮交互（Multi-turn Interaction）**

Agent 任务通常跨越多轮对话，单个用户查询可能触发：
- 多次 LLM 调用
- 多个工具执行
- 上下文的动态更新

这要求我们能够：
- 使用 Session ID 关联同一会话的所有操作
- 跟踪上下文的变化过程
- 分析多轮交互的整体效果

**3. 工具副作用（Tool Side Effects）**

工具调用可能修改外部状态，如：
- 写入数据库
- 发送邮件
- 调用第三方 API

这意味着我们需要：
- 记录工具调用前后的状态变化
- 支持操作回滚和补偿
- 审计所有外部副作用

**4. Token 消耗（Token Consumption）**

LLM 调用按 Token 计费，成本管理至关重要。我们需要：
- 实时监控 Token 使用量
- 按用户/会话/模型统计成本
- 设置预算告警和限制

**5. 幻觉检测（Hallucination Detection）**

Agent 可能生成虚假信息（幻觉），这需要：
- 对比工具返回值和 Agent 输出
- 检测不一致或矛盾
- 标记可能的幻觉内容

| 挑战 | 说明 | 解决方案 |
|:---|:---|:---|
| **非确定性** | 同样的输入可能产生不同的调用链 | 记录完整上下文和随机种子 |
| **多轮交互** | 单个任务可能跨越多轮对话 | 使用 Session ID 关联 |
| **工具副作用** | 工具调用可能修改外部状态 | 记录工具输入输出快照 |
| **Token 消耗** | LLM 调用成本高 | 监控 Token 使用量 |
| **幻觉检测** | Agent 可能生成虚假信息 | 对比工具返回值和 Agent 输出 |

---

## 28.2 分布式追踪系统

### 28.2.1 OpenTelemetry 基础

OpenTelemetry 是可观测性的标准框架，支持 Traces、Metrics 和 Logs：

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

# 配置资源标识
resource = Resource.create({
    "service.name": "ai-agent",
    "service.version": "1.0.0",
    "deployment.environment": "production"
})

# 配置 Tracer
provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)
```

### 28.2.2 Agent 追踪埋点

```python
from opentelemetry import trace
import time

class TracedAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.tracer = trace.get_tracer(__name__)
    
    async def run(self, query: str, session_id: str):
        with self.tracer.start_as_current_span(
            "agent.run",
            attributes={
                "session.id": session_id,
                "query.length": len(query),
                "agent.type": "react"
            }
        ) as span:
            # 记录输入
            span.set_attribute("input.query", query[:500])
            
            # 执行 ReAct 循环
            steps = []
            for step in range(10):
                with self.tracer.start_as_current_span(f"agent.step.{step}") as step_span:
                    # LLM 调用追踪
                    with self.tracer.start_as_current_span("llm.call") as llm_span:
                        llm_span.set_attribute("llm.model", "gpt-4o")
                        llm_span.set_attribute("llm.temperature", 0.7)
                        
                        start_time = time.time()
                        response = await self.llm.ainvoke(messages)
                        llm_latency = time.time() - start_time
                        
                        llm_span.set_attribute("llm.latency_ms", llm_latency * 1000)
                        llm_span.set_attribute("llm.tokens.input", response.usage.input_tokens)
                        llm_span.set_attribute("llm.tokens.output", response.usage.output_tokens)
                        llm_span.set_attribute("llm.content", response.content[:500])
                    
                    # 工具调用追踪
                    if should_use_tool(response):
                        with self.tracer.start_as_current_span(
                            "tool.call",
                            attributes={"tool.name": tool_name}
                        ) as tool_span:
                            tool_start = time.time()
                            tool_result = await self.tools[tool_name].ainvoke(tool_input)
                            tool_latency = time.time() - tool_start
                            
                            tool_span.set_attribute("tool.input", str(tool_input)[:500])
                            tool_span.set_attribute("tool.output", str(tool_result)[:500])
                            tool_span.set_attribute("tool.latency_ms", tool_latency * 1000)
                            tool_span.set_attribute("tool.success", True)
                    
                    steps.append({"step": step, "action": action})
                    step_span.set_attribute("steps.total", len(steps))
            
            # 记录最终结果
            span.set_attribute("output.answer", final_answer[:500])
            span.set_attribute("steps.count", len(steps))
            span.set_attribute("session.completed", True)
            
            return final_answer
```

### 28.2.3 LangSmith 深度集成

LangSmith 是 LangChain 官方的追踪平台，提供丰富的 Agent 可视化：

```python
import os
from langsmith import Client

# 配置环境变量
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_..."
os.environ["LANGCHAIN_PROJECT"] = "my-agent-production"

# 初始化客户端
client = Client()

# 方式1：自动追踪（推荐）
agent = create_react_agent(model=llm, tools=tools)
result = agent.invoke({"messages": [("user", "你好")]})

# 方式2：手动追踪
from langsmith.run_helpers import trace

@trace(name="custom_agent_run", run_type="chain")
async def custom_agent_run(query: str):
    # 你的自定义逻辑
    pass

# 方式3：添加元数据
from langsmith.run_helpers import trace, get_current_run_tree

@trace(
    name="agent_with_metadata",
    run_type="chain",
    metadata={"user_id": "user_123", "environment": "prod"}
)
async def agent_with_metadata(query: str):
    rt = get_current_run_tree()
    rt.add_metadata({"custom_field": "value"})
    # 你的逻辑
```

### 28.2.4 Langfuse 开源方案

Langfuse 是 LangSmith 的开源替代，支持自托管：

```python
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

# 初始化 Langfuse
langfuse = Langfuse(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com"
)

# 使用装饰器追踪
@observe(as_type="generation")
def call_llm(messages, model="gpt-4o"):
    response = openai.chat.completions.create(
        model=model,
        messages=messages
    )
    # 自动记录 LLM 调用
    langfuse_context.update_current_observation(
        model=model,
        usage={"input": response.usage.prompt_tokens, "output": response.usage.completion_tokens}
    )
    return response.choices[0].message.content

@observe()  # 追踪整个 Agent 执行
async def agent_run(query: str):
    messages = [{"role": "user", "content": query}]
    
    for step in range(10):
        response = call_llm(messages)
        
        if needs_tool(response):
            tool_result = await execute_tool(response)
            messages.append({"role": "tool", "content": tool_result})
        else:
            return response
    
    return "达到最大步数"

# 手动添加事件
langfuse_context.score_current_trace(
    name="user_feedback",
    value=1,
    comment="回答有帮助"
)
```

### 28.2.5 Arize Phoenix 本地追踪

Phoenix 是 Arize 提供的开源可观测性工具，特别适合本地开发：

```python
import phoenix as px
from phoenix.otel import register

# 启动本地 Phoenix 服务
px.launch_app()

# 注册 OpenTelemetry
tracer_provider = register(project_name="my-agent")

# 使用 LangChain 自动追踪
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager

tracer = LangChainTracer(project_name="my-agent")
callback_manager = CallbackManager([tracer])

agent = create_react_agent(model=llm, tools=tools, callbacks=callback_manager)
result = agent.invoke({"messages": [("user", "你好")]})

# 在浏览器中查看 http://localhost:6006
```

---

## 28.3 日志系统

### 28.3.1 结构化日志设计

```python
import structlog
from datetime import datetime
from typing import Optional, Dict, Any

# 配置 structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()

class AgentLogger:
    """Agent 专用日志记录器"""
    
    @staticmethod
    def log_agent_start(session_id: str, query: str, user_id: Optional[str] = None):
        logger.info(
            "agent.start",
            session_id=session_id,
            query=query[:500],
            user_id=user_id,
            timestamp=datetime.utcnow().isoformat()
        )
    
    @staticmethod
    def log_llm_call(
        session_id: str,
        model: str,
        messages_count: int,
        temperature: float,
        max_tokens: int
    ):
        logger.info(
            "llm.call",
            session_id=session_id,
            model=model,
            messages_count=messages_count,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    @staticmethod
    def log_llm_response(
        session_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        finish_reason: str
    ):
        logger.info(
            "llm.response",
            session_id=session_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            cost_usd=input_tokens * 0.00001 + output_tokens * 0.00003
        )
    
    @staticmethod
    def log_tool_call(
        session_id: str,
        tool_name: str,
        tool_input: Any,
        tool_output: Any,
        latency_ms: float,
        success: bool,
        error: Optional[str] = None
    ):
        log_fn = logger.info if success else logger.error
        log_fn(
            "tool.call",
            session_id=session_id,
            tool_name=tool_name,
            tool_input=str(tool_input)[:1000],
            tool_output=str(tool_output)[:1000],
            latency_ms=latency_ms,
            success=success,
            error=error
        )
    
    @staticmethod
    def log_agent_end(
        session_id: str,
        total_steps: int,
        total_latency_ms: float,
        total_tokens: int,
        total_cost_usd: float,
        success: bool
    ):
        logger.info(
            "agent.end",
            session_id=session_id,
            total_steps=total_steps,
            total_latency_ms=total_latency_ms,
            total_tokens=total_tokens,
            total_cost_usd=total_cost_usd,
            success=success
        )
    
    @staticmethod
    def log_error(session_id: str, error_type: str, error_message: str, stack_trace: str):
        logger.error(
            "agent.error",
            session_id=session_id,
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace
        )
```

### 28.3.2 ELK Stack 配置

```yaml
# docker-compose.yml
version: '3.8'
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
  
  logstash:
    image: docker.elastic.co/logstash/logstash:8.11.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - "5000:5000"
  
  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
```

```ruby
# logstash.conf
input {
  tcp {
    port => 5000
    codec => json_lines
  }
}

filter {
  # 解析 Agent 日志
  if [type] == "agent" {
    mutate {
      add_field => { "index_name" => "agent-logs-%{+YYYY.MM.dd}" }
    }
    
    # 提取关键字段
    ruby {
      code => "
        event.set('input_tokens', event.get('[llm][usage][input_tokens]').to_i)
        event.set('output_tokens', event.get('[llm][usage][output_tokens]').to_i)
        event.set('total_cost', event.get('input_tokens').to_f * 0.00001 + event.get('output_tokens').to_f * 0.00003)
      "
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "%{[index_name]}"
  }
}
```

### 28.3.3 Grafana Loki 日志聚合

```python
import logging
import json
from datetime import datetime

class LokiHandler(logging.Handler):
    """Grafana Loki 日志处理器"""
    
    def __init__(self, loki_url: str, labels: dict):
        super().__init__()
        self.loki_url = loki_url
        self.labels = labels
    
    def emit(self, record):
        log_entry = {
            "streams": [{
                "stream": self.labels,
                "values": [
                    [str(int(datetime.utcnow().timestamp() * 1e9)), self.format(record)]
                ]
            }]
        }
        
        import requests
        requests.post(f"{self.loki_url}/loki/api/v1/push", json=log_entry)

# 配置日志
handler = LokiHandler(
    loki_url="http://localhost:3100",
    labels={"service": "ai-agent", "environment": "production"}
)

logger = logging.getLogger("agent")
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

---

## 28.4 指标监控系统

### 28.4.1 核心指标设计

```python
from prometheus_client import Counter, Histogram, Gauge, Summary, Info
from prometheus_client import start_http_server
import time

class AgentMetrics:
    """Agent 指标收集器"""
    
    def __init__(self):
        # 请求计数器
        self.requests_total = Counter(
            'agent_requests_total',
            'Total agent requests',
            ['status', 'agent_type', 'user_id']
        )
        
        # 延迟直方图
        self.request_latency = Histogram(
            'agent_request_latency_seconds',
            'Agent request latency',
            ['agent_type'],
            buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60]
        )
        
        # LLM 调用指标
        self.llm_calls_total = Counter(
            'agent_llm_calls_total',
            'Total LLM calls',
            ['model', 'status']
        )
        
        self.llm_tokens_total = Counter(
            'agent_llm_tokens_total',
            'Total LLM tokens used',
            ['model', 'type']  # type: input/output
        )
        
        self.llm_latency = Histogram(
            'agent_llm_latency_seconds',
            'LLM call latency',
            ['model'],
            buckets=[0.1, 0.5, 1, 2, 5, 10]
        )
        
        # 工具调用指标
        self.tool_calls_total = Counter(
            'agent_tool_calls_total',
            'Total tool calls',
            ['tool_name', 'status']
        )
        
        self.tool_latency = Histogram(
            'agent_tool_latency_seconds',
            'Tool call latency',
            ['tool_name'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1, 5]
        )
        
        # 成本指标
        self.cost_total = Counter(
            'agent_cost_usd_total',
            'Total cost in USD',
            ['model']
        )
        
        # 活跃会话
        self.active_sessions = Gauge(
            'agent_active_sessions',
            'Number of active sessions'
        )
        
        # 错误率
        self.error_rate = Gauge(
            'agent_error_rate',
            'Current error rate'
        )
    
    def record_request(self, status: str, agent_type: str, user_id: str):
        self.requests_total.labels(status=status, agent_type=agent_type, user_id=user_id).inc()
    
    def record_latency(self, latency: float, agent_type: str):
        self.request_latency.labels(agent_type=agent_type).observe(latency)
    
    def record_llm_call(self, model: str, input_tokens: int, output_tokens: int, latency: float, cost: float):
        self.llm_calls_total.labels(model=model, status="success").inc()
        self.llm_tokens_total.labels(model=model, type="input").inc(input_tokens)
        self.llm_tokens_total.labels(model=model, type="output").inc(output_tokens)
        self.llm_latency.labels(model=model).observe(latency)
        self.cost_total.labels(model=model).inc(cost)
    
    def record_tool_call(self, tool_name: str, success: bool, latency: float):
        status = "success" if success else "error"
        self.tool_calls_total.labels(tool_name=tool_name, status=status).inc()
        self.tool_latency.labels(tool_name=tool_name).observe(latency)
```

### 28.4.2 Grafana 仪表板配置

```json
{
  "dashboard": {
    "title": "AI Agent 监控仪表板",
    "panels": [
      {
        "title": "请求速率",
        "type": "graph",
        "targets": [{
          "expr": "rate(agent_requests_total[5m])",
          "legendFormat": "{{status}} - {{agent_type}}"
        }]
      },
      {
        "title": "延迟分布",
        "type": "heatmap",
        "targets": [{
          "expr": "histogram_quantile(0.95, rate(agent_request_latency_seconds_bucket[5m]))",
          "legendFormat": "P95"
        }]
      },
      {
        "title": "错误率",
        "type": "singlestat",
        "targets": [{
          "expr": "rate(agent_requests_total{status='error'}[5m]) / rate(agent_requests_total[5m]) * 100",
          "format": "percent"
        }],
        "thresholds": [
          {"value": 1, "color": "green"},
          {"value": 5, "color": "yellow"},
          {"value": 10, "color": "red"}
        ]
      },
      {
        "title": "Token 使用量",
        "type": "graph",
        "targets": [{
          "expr": "rate(agent_llm_tokens_total[5m])",
          "legendFormat": "{{model}} - {{type}}"
        }]
      },
      {
        "title": "成本趋势",
        "type": "graph",
        "targets": [{
          "expr": "increase(agent_cost_usd_total[1h])",
          "legendFormat": "{{model}}"
        }]
      },
      {
        "title": "工具调用成功率",
        "type": "piechart",
        "targets": [{
          "expr": "sum by (tool_name) (agent_tool_calls_total{status='success'})",
          "legendFormat": "{{tool_name}}"
        }]
      }
    ]
  }
}
```

### 28.4.3 自定义导出器

```python
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import psutil
import asyncio

class AgentResourceMetrics:
    """Agent 资源使用指标"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        
        self.cpu_usage = Gauge(
            'agent_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'agent_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.gpu_usage = Gauge(
            'agent_gpu_usage_percent',
            'GPU usage percentage',
            ['device'],
            registry=self.registry
        )
        
        self.gpu_memory = Gauge(
            'agent_gpu_memory_usage_bytes',
            'GPU memory usage in bytes',
            ['device'],
            registry=self.registry
        )
    
    async def collect_metrics(self):
        """收集系统资源指标"""
        while True:
            # CPU 和内存
            self.cpu_usage.set(psutil.cpu_percent())
            self.memory_usage.set(psutil.virtual_memory().used)
            
            # GPU 指标（如果可用）
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        self.gpu_usage.labels(device=f"gpu_{i}").set(
                            torch.cuda.utilization(i)
                        )
                        self.gpu_memory.labels(device=f"gpu_{i}").set(
                            torch.cuda.memory_allocated(i)
                        )
            except ImportError:
                pass
            
            await asyncio.sleep(10)
    
    def push_metrics(self, gateway: str):
        """推送到 Pushgateway"""
        push_to_gateway(gateway, job='ai_agent', registry=self.registry)
```

---

## 28.5 告警系统

### 28.5.1 告警规则设计

```yaml
# prometheus-rules.yml
groups:
  - name: agent_alerts
    rules:
      # 错误率告警
      - alert: HighErrorRate
        expr: rate(agent_requests_total{status="error"}[5m]) / rate(agent_requests_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Agent 错误率过高"
          description: "错误率 {{ $value | humanizePercentage }} 超过 10% 阈值"
      
      # 延迟告警
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(agent_request_latency_seconds_bucket[5m])) > 30
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Agent 响应延迟过高"
          description: "P95 延迟 {{ $value }}s 超过 30s 阈值"
      
      # LLM 调用失败
      - alert: LLMCallFailure
        expr: rate(agent_llm_calls_total{status="error"}[5m]) > 0.05
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "LLM 调用失败率过高"
          description: "LLM 调用失败率 {{ $value | humanizePercentage }}"
      
      # 成本异常
      - alert: HighCost
        expr: increase(agent_cost_usd_total[1h]) > 100
        labels:
          severity: warning
        annotations:
          summary: "每小时成本超过 $100"
          description: "当前小时成本 ${{ $value }}"
      
      # 工具调用失败
      - alert: ToolCallFailure
        expr: rate(agent_tool_calls_total{status="error"}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "工具调用失败率过高"
          description: "工具 {{ $labels.tool_name }} 失败率 {{ $value | humanizePercentage }}"
      
      # 活跃会话过多
      - alert: TooManySessions
        expr: agent_active_sessions > 1000
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "活跃会话数过多"
          description: "当前活跃会话 {{ $value }} 超过 1000"
```

### 28.5.2 告警通知集成

```python
import aiohttp
import json
from typing import Dict, Any

class AlertNotifier:
    """多渠道告警通知器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def send_alert(self, alert: Dict[str, Any]):
        """发送告警到多个渠道"""
        severity = alert.get("labels", {}).get("severity", "info")
        
        # 根据严重程度选择通知渠道
        if severity == "critical":
            await self._send_pagerduty(alert)
            await self._send_slack(alert)
            await self._send_email(alert)
        elif severity == "warning":
            await self._send_slack(alert)
        else:
            await self._send_slack(alert, channel="#agent-info")
    
    async def _send_slack(self, alert: Dict[str, Any], channel: str = "#agent-alerts"):
        """发送 Slack 通知"""
        webhook_url = self.config["slack_webhook"]
        
        message = {
            "channel": channel,
            "attachments": [{
                "color": "danger" if alert.get("labels", {}).get("severity") == "critical" else "warning",
                "title": alert.get("annotations", {}).get("summary"),
                "text": alert.get("annotations", {}).get("description"),
                "fields": [
                    {"title": "Severity", "value": alert.get("labels", {}).get("severity"), "short": True},
                    {"title": "Service", "value": alert.get("labels", {}).get("service", "ai-agent"), "short": True}
                ],
                "footer": "AI Agent Monitoring",
                "ts": int(time.time())
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            await session.post(webhook_url, json=message)
    
    async def _send_pagerduty(self, alert: Dict[str, Any]):
        """发送 PagerDuty 告警"""
        # 集成 PagerDuty API
        pass
    
    async def _send_email(self, alert: Dict[str, Any]):
        """发送邮件通知"""
        # 集成邮件服务
        pass
```

---

## 28.6 调试工作流

### 28.6.1 Agent 调试最佳实践

```python
class AgentDebugger:
    """Agent 调试工具"""
    
    def __init__(self, agent, tracer):
        self.agent = agent
        self.tracer = tracer
    
    async def debug_run(self, query: str, session_id: str):
        """调试模式运行 Agent"""
        print(f"\n{'='*60}")
        print(f"调试会话: {session_id}")
        print(f"查询: {query}")
        print(f"{'='*60}\n")
        
        # 记录初始状态
        initial_state = {
            "query": query,
            "session_id": session_id,
            "start_time": time.time()
        }
        
        try:
            # 执行 Agent
            result = await self.agent.run(query, session_id)
            
            # 记录成功
            self._log_success(initial_state, result)
            return result
            
        except Exception as e:
            # 记录错误
            self._log_error(initial_state, e)
            raise
    
    def _log_success(self, state: dict, result: dict):
        """记录成功执行"""
        duration = time.time() - state["start_time"]
        print(f"\n{'='*60}")
        print(f"执行成功")
        print(f"耗时: {duration:.2f}s")
        print(f"步骤数: {result.get('steps', 0)}")
        print(f"Token 使用: {result.get('tokens', 0)}")
        print(f"成本: ${result.get('cost', 0):.4f}")
        print(f"{'='*60}\n")
    
    def _log_error(self, state: dict, error: Exception):
        """记录错误"""
        duration = time.time() - state["start_time"]
        print(f"\n{'='*60}")
        print(f"执行失败")
        print(f"耗时: {duration:.2f}s")
        print(f"错误类型: {type(error).__name__}")
        print(f"错误信息: {str(error)}")
        print(f"{'='*60}\n")
```

### 28.6.2 交互式调试

```python
import ipywidgets as widgets
from IPython.display import display, clear_output

class InteractiveDebugger:
    """交互式调试器（Jupyter Notebook）"""
    
    def __init__(self, agent):
        self.agent = agent
        self.output = widgets.Output()
        self.query_input = widgets.Textarea(
            placeholder='输入查询...',
            description='查询:',
            layout=widgets.Layout(width='100%', height='100px')
        )
        self.run_button = widgets.Button(description='运行')
        self.run_button.on_click(self._on_run)
        
        display(self.query_input, self.run_button, self.output)
    
    def _on_run(self, b):
        with self.output:
            clear_output()
            query = self.query_input.value
            print(f"执行查询: {query}\n")
            
            # 运行 Agent
            result = asyncio.run(self.agent.run(query))
            
            # 显示结果
            print(f"结果:\n{result['answer']}\n")
            print(f"步骤数: {result['steps']}")
            print(f"Token: {result['tokens']}")
```

---

## 28.7 生产环境案例

### 28.7.1 大规模 Agent 系统监控架构

```
┌─────────────────────────────────────────────────────────────┐
│                     负载均衡器 (Nginx)                        │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Agent Pod 1  │    │  Agent Pod 2  │    │  Agent Pod 3  │
│  ┌─────────┐  │    │  ┌─────────┐  │    │  ┌─────────┐  │
│  │ Agent   │  │    │  │ Agent   │  │    │  │ Agent   │  │
│  │  + OTel │  │    │  │  + OTel │  │    │  │  + OTel │  │
│  └─────────┘  │    │  └─────────┘  │    │  └─────────┘  │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Jaeger/Tempo   │
                    │  (Tracing)      │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   Prometheus    │
                    │   (Metrics)     │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │     Grafana     │
                    │ (Dashboard)     │
                    └─────────────────┘
```

### 28.7.2 成本监控和优化

```python
class CostMonitor:
    """成本监控和优化"""
    
    def __init__(self):
        self.cost_limits = {
            "daily": 100.0,  # 每日限额
            "hourly": 10.0,  # 每小时限额
            "per_request": 1.0  # 单次请求限额
        }
        self.current_costs = {
            "daily": 0.0,
            "hourly": 0.0,
            "per_request": 0.0
        }
    
    def check_cost_limit(self, estimated_cost: float) -> bool:
        """检查是否超出成本限额"""
        if self.current_costs["daily"] + estimated_cost > self.cost_limits["daily"]:
            return False
        if self.current_costs["hourly"] + estimated_cost > self.cost_limits["hourly"]:
            return False
        if estimated_cost > self.cost_limits["per_request"]:
            return False
        return True
    
    def record_cost(self, model: str, input_tokens: int, output_tokens: int):
        """记录成本"""
        cost = self._calculate_cost(model, input_tokens, output_tokens)
        self.current_costs["daily"] += cost
        self.current_costs["hourly"] += cost
        self.current_costs["per_request"] += cost
        
        # 检查告警
        self._check_alerts()
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """计算成本"""
        pricing = {
            "gpt-4o": {"input": 0.00001, "output": 0.00003},
            "gpt-4o-mini": {"input": 0.00000015, "output": 0.0000006},
            "claude-3-opus": {"input": 0.000015, "output": 0.000075}
        }
        
        rates = pricing.get(model, {"input": 0.00001, "output": 0.00003})
        return input_tokens * rates["input"] + output_tokens * rates["output"]
    
    def _check_alerts(self):
        """检查是否需要告警"""
        if self.current_costs["daily"] > self.cost_limits["daily"] * 0.8:
            self._send_cost_alert("daily", 80)
        if self.current_costs["hourly"] > self.cost_limits["hourly"] * 0.8:
            self._send_cost_alert("hourly", 80)
    
    def _send_cost_alert(self, period: str, threshold: int):
        """发送成本告警"""
        print(f"警告: {period}成本已达 {threshold}%")
```

---

## 28.8 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| **三支柱** | Tracing 追踪调用链、Metrics 监控性能、Logging 记录事件 |
| **OpenTelemetry** | 标准化的可观测性框架，支持 Traces、Metrics、Logs |
| **LangSmith** | LangChain 官方追踪平台，提供丰富的 Agent 可视化 |
| **Langfuse** | 开源替代方案，支持自托管 |
| **Prometheus + Grafana** | 指标收集和可视化，支持自定义仪表板 |
| **结构化日志** | 使用 structlog 或类似库记录结构化日志 |
| **告警系统** | 基于 Prometheus 规则的多渠道告警通知 |
| **成本监控** | 实时跟踪 Token 使用和 API 成本 |
| **调试工具** | 交互式调试器，支持断点和状态检查 |

---

## 28.9 高级监控场景

### 28.9.1 实时监控系统

```python
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class AlertSeverity(Enum):
    """告警严重程度"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class MonitoringRule:
    """监控规则"""
    name: str
    metric: str
    condition: str  # "gt", "lt", "eq", "ne"
    threshold: float
    severity: AlertSeverity
    window: int = 300  # 时间窗口（秒）
    description: str = ""

class RealTimeMonitor:
    """实时监控系统"""
    
    def __init__(self):
        self.rules: List[MonitoringRule] = []
        self.metrics_buffer: Dict[str, List[Dict]] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.notification_channels: List[callable] = []
    
    def add_rule(self, rule: MonitoringRule):
        """添加监控规则"""
        self.rules.append(rule)
        self.metrics_buffer[rule.metric] = []
    
    def add_notification_channel(self, channel: callable):
        """添加通知渠道"""
        self.notification_channels.append(channel)
    
    async def record_metric(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """记录指标"""
        if metric_name not in self.metrics_buffer:
            self.metrics_buffer[metric_name] = []
        
        self.metrics_buffer[metric_name].append({
            "value": value,
            "timestamp": time.time(),
            "labels": labels or {}
        })
        
        # 检查规则
        await self._check_rules(metric_name)
    
    async def _check_rules(self, metric_name: str):
        """检查监控规则"""
        for rule in self.rules:
            if rule.metric != metric_name:
                continue
            
            # 获取时间窗口内的数据
            window_start = time.time() - rule.window
            recent_values = [
                m["value"] for m in self.metrics_buffer[metric_name]
                if m["timestamp"] >= window_start
            ]
            
            if not recent_values:
                continue
            
            # 计算统计值
            avg_value = sum(recent_values) / len(recent_values)
            max_value = max(recent_values)
            min_value = min(recent_values)
            
            # 检查条件
            triggered = False
            current_value = avg_value
            
            if rule.condition == "gt" and avg_value > rule.threshold:
                triggered = True
            elif rule.condition == "lt" and avg_value < rule.threshold:
                triggered = True
            elif rule.condition == "eq" and abs(avg_value - rule.threshold) < 0.001:
                triggered = True
            
            if triggered:
                await self._trigger_alert(rule, current_value, {
                    "avg": avg_value,
                    "max": max_value,
                    "min": min_value,
                    "count": len(recent_values)
                })
    
    async def _trigger_alert(self, rule: MonitoringRule, current_value: float, stats: Dict):
        """触发告警"""
        alert = {
            "rule": rule.name,
            "metric": rule.metric,
            "severity": rule.severity.value,
            "current_value": current_value,
            "threshold": rule.threshold,
            "condition": rule.condition,
            "stats": stats,
            "timestamp": time.time(),
            "description": rule.description
        }
        
        self.alerts.append(alert)
        
        # 发送通知
        for channel in self.notification_channels:
            try:
                await channel(alert)
            except Exception as e:
                print(f"Notification failed: {e}")
    
    def get_active_alerts(self, severity: AlertSeverity = None) -> List[Dict]:
        """获取活跃告警"""
        if severity:
            return [a for a in self.alerts if a["severity"] == severity.value]
        return self.alerts
    
    def get_metric_statistics(self, metric_name: str, window: int = 3600) -> Dict[str, Any]:
        """获取指标统计"""
        if metric_name not in self.metrics_buffer:
            return {}
        
        window_start = time.time() - window
        values = [
            m["value"] for m in self.metrics_buffer[metric_name]
            if m["timestamp"] >= window_start
        ]
        
        if not values:
            return {"count": 0}
        
        return {
            "count": len(values),
            "avg": sum(values) / len(values),
            "max": max(values),
            "min": min(values),
            "sum": sum(values),
            "p50": sorted(values)[len(values) // 2],
            "p95": sorted(values)[int(len(values) * 0.95)],
            "p99": sorted(values)[int(len(values) * 0.99)]
        }

class MetricAggregator:
    """指标聚合器"""
    
    def __init__(self):
        self.aggregation_rules: Dict[str, Dict] = {}
        self.aggregated_metrics: Dict[str, Any] = {}
    
    def add_aggregation_rule(self, metric_name: str, aggregation_type: str, window: int):
        """添加聚合规则"""
        self.aggregation_rules[metric_name] = {
            "type": aggregation_type,  # sum, avg, max, min, count, percentile
            "window": window
        }
    
    async def aggregate(self, metric_name: str, values: List[float]) -> Dict[str, Any]:
        """执行聚合"""
        if metric_name not in self.aggregation_rules:
            return {"value": values[-1] if values else 0}
        
        rule = self.aggregation_rules[metric_name]
        window_start = time.time() - rule["window"]
        
        # 这里简化处理，实际应该根据时间戳过滤
        recent_values = values[-100:]  # 取最近100个值
        
        if not recent_values:
            return {"value": 0}
        
        if rule["type"] == "sum":
            result = sum(recent_values)
        elif rule["type"] == "avg":
            result = sum(recent_values) / len(recent_values)
        elif rule["type"] == "max":
            result = max(recent_values)
        elif rule["type"] == "min":
            result = min(recent_values)
        elif rule["type"] == "count":
            result = len(recent_values)
        elif rule["type"] == "percentile":
            sorted_values = sorted(recent_values)
            result = sorted_values[int(len(sorted_values) * 0.95)]
        else:
            result = recent_values[-1]
        
        return {
            "metric": metric_name,
            "aggregation": rule["type"],
            "value": result,
            "window": rule["window"],
            "sample_count": len(recent_values)
        }
```

### 28.9.2 异常检测系统

```python
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class AnomalyDetectionResult:
    """异常检测结果"""
    is_anomaly: bool
    score: float
    threshold: float
    expected_range: Tuple[float, float]
    actual_value: float
    description: str

class StatisticalAnomalyDetector:
    """统计异常检测器"""
    
    def __init__(self, sensitivity: float = 3.0):
        self.sensitivity = sensitivity
        self.history: Dict[str, List[float]] = {}
        self.baseline: Dict[str, Dict] = {}
    
    def update_baseline(self, metric_name: str, values: List[float]):
        """更新基线"""
        if len(values) < 10:
            return
        
        self.history[metric_name] = values
        
        mean = np.mean(values)
        std = np.std(values)
        
        self.baseline[metric_name] = {
            "mean": mean,
            "std": std,
            "min": np.percentile(values, 5),
            "max": np.percentile(values, 95),
            "iqr": np.percentile(values, 75) - np.percentile(values, 25)
        }
    
    def detect(self, metric_name: str, value: float) -> AnomalyDetectionResult:
        """检测异常"""
        if metric_name not in self.baseline:
            return AnomalyDetectionResult(
                is_anomaly=False,
                score=0,
                threshold=0,
                expected_range=(0, 0),
                actual_value=value,
                description="无基线数据"
            )
        
        baseline = self.baseline[metric_name]
        mean = baseline["mean"]
        std = baseline["std"]
        
        # Z-score 方法
        z_score = abs(value - mean) / std if std > 0 else 0
        
        # 计算预期范围
        expected_min = mean - self.sensitivity * std
        expected_max = mean + self.sensitivity * std
        
        # 判断是否异常
        is_anomaly = z_score > self.sensitivity
        
        # 计算异常分数（0-1）
        score = min(z_score / (self.sensitivity * 2), 1.0)
        
        description = "正常"
        if is_anomaly:
            if value > expected_max:
                description = f"异常高值 (超出 {z_score:.2f} 个标准差)"
            else:
                description = f"异常低值 (低于 {z_score:.2f} 个标准差)"
        
        return AnomalyDetectionResult(
            is_anomaly=is_anomaly,
            score=score,
            threshold=self.sensitivity,
            expected_range=(expected_min, expected_max),
            actual_value=value,
            description=description
        )

class TimeSeriesAnomalyDetector:
    """时间序列异常检测器"""
    
    def __init__(self, window_size: int = 24):
        self.window_size = window_size
        self.data_windows: Dict[str, List[float]] = {}
    
    def add_data_point(self, metric_name: str, value: float):
        """添加数据点"""
        if metric_name not in self.data_windows:
            self.data_windows[metric_name] = []
        
        self.data_windows[metric_name].append(value)
        
        # 保持窗口大小
        if len(self.data_windows[metric_name]) > self.window_size:
            self.data_windows[metric_name] = self.data_windows[metric_name][-self.window_size:]
    
    def detect(self, metric_name: str, value: float) -> AnomalyDetectionResult:
        """检测异常"""
        if metric_name not in self.data_windows or len(self.data_windows[metric_name]) < 10:
            return AnomalyDetectionResult(
                is_anomaly=False,
                score=0,
                threshold=0,
                expected_range=(0, 0),
                actual_value=value,
                description="数据不足"
            )
        
        window = self.data_windows[metric_name]
        
        # 计算移动平均和标准差
        moving_avg = np.mean(window)
        moving_std = np.std(window)
        
        # 计算预期范围
        expected_min = moving_avg - 2 * moving_std
        expected_max = moving_avg + 2 * moving_std
        
        # 计算偏离度
        deviation = abs(value - moving_avg)
        score = min(deviation / (moving_std * 4) if moving_std > 0 else 0, 1.0)
        
        is_anomaly = value < expected_min or value > expected_max
        
        return AnomalyDetectionResult(
            is_anomaly=is_anomaly,
            score=score,
            threshold=2.0,
            expected_range=(expected_min, expected_max),
            actual_value=value,
            description="异常" if is_anomaly else "正常"
        )

class SeasonalAnomalyDetector:
    """季节性异常检测器"""
    
    def __init__(self, season_length: int = 24):
        self.season_length = season_length
        self.seasonal_patterns: Dict[str, List[float]] = {}
        self.deviations: Dict[str, List[float]] = {}
    
    def learn_seasonal_pattern(self, metric_name: str, historical_data: List[float]):
        """学习季节性模式"""
        if len(historical_data) < self.season_length * 2:
            return
        
        # 计算季节性模式
        seasonal_pattern = []
        for i in range(self.season_length):
            season_values = historical_data[i::self.season_length]
            seasonal_pattern.append(np.mean(season_values))
        
        self.seasonal_patterns[metric_name] = seasonal_pattern
        
        # 计算每个季节的偏差
        deviations = []
        for i, value in enumerate(historical_data):
            season_idx = i % self.season_length
            expected = seasonal_pattern[season_idx]
            deviations.append(abs(value - expected))
        
        self.deviations[metric_name] = deviations
    
    def detect(self, metric_name: str, value: float, position: int) -> AnomalyDetectionResult:
        """检测异常"""
        if metric_name not in self.seasonal_patterns:
            return AnomalyDetectionResult(
                is_anomaly=False,
                score=0,
                threshold=0,
                expected_range=(0, 0),
                actual_value=value,
                description="无季节性模式"
            )
        
        pattern = self.seasonal_patterns[metric_name]
        season_idx = position % self.season_length
        expected_value = pattern[season_idx]
        
        # 计算预期偏差范围
        if metric_name in self.deviations and self.deviations[metric_name]:
            avg_deviation = np.mean(self.deviations[metric_name])
            std_deviation = np.std(self.deviations[metric_name])
            max_expected_deviation = avg_deviation + 3 * std_deviation
        else:
            max_expected_deviation = abs(expected_value) * 0.3
        
        # 检测异常
        actual_deviation = abs(value - expected_value)
        is_anomaly = actual_deviation > max_expected_deviation
        
        score = min(actual_deviation / (max_expected_deviation * 2) if max_expected_deviation > 0 else 0, 1.0)
        
        return AnomalyDetectionResult(
            is_anomaly=is_anomaly,
            score=score,
            threshold=3.0,
            expected_range=(expected_value - max_expected_deviation, expected_value + max_expected_deviation),
            actual_value=value,
            description=f"预期 {expected_value:.2f}，实际 {value:.2f}"
        )
```

---

## 28.10 性能优化监控

### 28.10.1 性能指标体系

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any
import time

@dataclass
class PerformanceMetric:
    """性能指标定义"""
    name: str
    description: str
    unit: str
    metric_type: str  # counter, gauge, histogram, summary
    labels: List[str] = field(default_factory=list)

class PerformanceMonitor:
    """性能监控系统"""
    
    def __init__(self):
        self.metrics: Dict[str, PerformanceMetric] = {}
        self.metric_values: Dict[str, List[Dict]] = {}
        self.slo_targets: Dict[str, Dict] = {}
        
        self._register_default_metrics()
    
    def _register_default_metrics(self):
        """注册默认性能指标"""
        default_metrics = [
            PerformanceMetric(
                name="request_latency",
                description="请求延迟",
                unit="ms",
                metric_type="histogram",
                labels=["endpoint", "method", "status"]
            ),
            PerformanceMetric(
                name="request_throughput",
                description="请求吞吐量",
                unit="req/s",
                metric_type="gauge",
                labels=["endpoint"]
            ),
            PerformanceMetric(
                name="error_rate",
                description="错误率",
                unit="%",
                metric_type="gauge",
                labels=["endpoint", "error_type"]
            ),
            PerformanceMetric(
                name="llm_latency",
                description="LLM 调用延迟",
                unit="ms",
                metric_type="histogram",
                labels=["model", "operation"]
            ),
            PerformanceMetric(
                name="llm_tokens_per_second",
                description="LLM Token 生成速度",
                unit="tokens/s",
                metric_type="gauge",
                labels=["model"]
            ),
            PerformanceMetric(
                name="memory_usage",
                description="内存使用量",
                unit="bytes",
                metric_type="gauge",
                labels=["component"]
            ),
            PerformanceMetric(
                name="cpu_usage",
                description="CPU 使用率",
                unit="%",
                metric_type="gauge",
                labels=["component"]
            ),
            PerformanceMetric(
                name="active_connections",
                description="活跃连接数",
                unit="connections",
                metric_type="gauge",
                labels=["service"]
            ),
        ]
        
        for metric in default_metrics:
            self.metrics[metric.name] = metric
            self.metric_values[metric.name] = []
    
    def record(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """记录指标值"""
        if metric_name not in self.metrics:
            return
        
        self.metric_values[metric_name].append({
            "value": value,
            "timestamp": time.time(),
            "labels": labels or {}
        })
        
        # 限制历史数据量
        if len(self.metric_values[metric_name]) > 10000:
            self.metric_values[metric_name] = self.metric_values[metric_name][-5000:]
    
    def set_slo_target(self, metric_name: str, target: float, window: int = 300):
        """设置 SLO 目标"""
        self.slo_targets[metric_name] = {
            "target": target,
            "window": window,
            "breaches": 0,
            "total_checks": 0
        }
    
    def check_slo(self, metric_name: str) -> Dict[str, Any]:
        """检查 SLO"""
        if metric_name not in self.slo_targets:
            return {"status": "no_target"}
        
        target_info = self.slo_targets[metric_name]
        window_start = time.time() - target_info["window"]
        
        recent_values = [
            v["value"] for v in self.metric_values.get(metric_name, [])
            if v["timestamp"] >= window_start
        ]
        
        if not recent_values:
            return {"status": "no_data"}
        
        avg_value = sum(recent_values) / len(recent_values)
        is_met = avg_value <= target_info["target"]
        
        target_info["total_checks"] += 1
        if not is_met:
            target_info["breaches"] += 1
        
        return {
            "status": "met" if is_met else "breached",
            "current_value": avg_value,
            "target": target_info["target"],
            "breach_rate": target_info["breaches"] / target_info["total_checks"] if target_info["total_checks"] > 0 else 0
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        report = {
            "timestamp": time.time(),
            "metrics": {},
            "slo_status": {},
            "recommendations": []
        }
        
        for metric_name, metric in self.metrics.items():
            recent_values = [v["value"] for v in self.metric_values.get(metric_name, [])[-100:]]
            
            if recent_values:
                report["metrics"][metric_name] = {
                    "description": metric.description,
                    "unit": metric.unit,
                    "current": recent_values[-1],
                    "avg": sum(recent_values) / len(recent_values),
                    "min": min(recent_values),
                    "max": max(recent_values),
                    "p95": sorted(recent_values)[int(len(recent_values) * 0.95)] if len(recent_values) >= 20 else recent_values[-1]
                }
        
        # 检查 SLO
        for metric_name in self.slo_targets:
            slo_status = self.check_slo(metric_name)
            report["slo_status"][metric_name] = slo_status
            
            if slo_status.get("status") == "breached":
                report["recommendations"].append(
                    f"SLO 违规: {metric_name} 当前值 {slo_status.get('current_value', 0):.2f} 超过目标 {slo_status.get('target', 0):.2f}"
                )
        
        return report

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.optimization_history: List[Dict] = []
    
    async def analyze_bottlenecks(self) -> List[Dict[str, Any]]:
        """分析性能瓶颈"""
        bottlenecks = []
        
        # 分析延迟
        latency_data = self.monitor.metric_values.get("request_latency", [])
        if latency_data:
            recent_latencies = [v["value"] for v in latency_data[-100:]]
            avg_latency = sum(recent_latencies) / len(recent_latencies)
            
            if avg_latency > 1000:  # 超过 1 秒
                bottlenecks.append({
                    "type": "high_latency",
                    "metric": "request_latency",
                    "current_value": avg_latency,
                    "threshold": 1000,
                    "severity": "high",
                    "recommendation": "考虑优化 LLM 调用或添加缓存"
                })
        
        # 分析错误率
        error_data = self.monitor.metric_values.get("error_rate", [])
        if error_data:
            recent_errors = [v["value"] for v in error_data[-100:]]
            avg_error_rate = sum(recent_errors) / len(recent_errors)
            
            if avg_error_rate > 5:  # 错误率超过 5%
                bottlenecks.append({
                    "type": "high_error_rate",
                    "metric": "error_rate",
                    "current_value": avg_error_rate,
                    "threshold": 5,
                    "severity": "critical",
                    "recommendation": "检查错误日志，修复根本原因"
                })
        
        # 分析内存使用
        memory_data = self.monitor.metric_values.get("memory_usage", [])
        if memory_data:
            recent_memory = [v["value"] for v in memory_data[-100:]]
            avg_memory = sum(recent_memory) / len(recent_memory)
            max_memory = max(recent_memory)
            
            if max_memory > 1024 * 1024 * 1024:  # 超过 1GB
                bottlenecks.append({
                    "type": "high_memory_usage",
                    "metric": "memory_usage",
                    "current_value": max_memory / (1024 * 1024 * 1024),
                    "threshold": 1,
                    "severity": "medium",
                    "recommendation": "检查内存泄漏，优化数据结构"
                })
        
        return bottlenecks
    
    async def generate_optimization_plan(self) -> Dict[str, Any]:
        """生成优化计划"""
        bottlenecks = await self.analyze_bottlenecks()
        
        plan = {
            "timestamp": time.time(),
            "bottlenecks": bottlenecks,
            "optimizations": [],
            "estimated_impact": {}
        }
        
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "high_latency":
                plan["optimizations"].append({
                    "target": "latency",
                    "actions": [
                        "启用 LLM 响应缓存",
                        "优化 Prompt 长度",
                        "使用更快的模型处理简单请求",
                        "实现异步处理"
                    ],
                    "priority": "high"
                })
            
            elif bottleneck["type"] == "high_error_rate":
                plan["optimizations"].append({
                    "target": "reliability",
                    "actions": [
                        "添加重试机制",
                        "实现熔断器",
                        "改进错误处理",
                        "添加输入验证"
                    ],
                    "priority": "critical"
                })
            
            elif bottleneck["type"] == "high_memory_usage":
                plan["optimizations"].append({
                    "target": "memory",
                    "actions": [
                        "优化数据结构",
                        "实现分页加载",
                        "添加垃圾回收",
                        "使用流式处理"
                    ],
                    "priority": "medium"
                })
        
        return plan
```

---

## 28.11 成本分析和预测

### 28.11.1 成本追踪系统

```python
from datetime import datetime, timedelta
from typing import Dict, List, Any
from dataclasses import dataclass
import json

@dataclass
class CostEntry:
    """成本条目"""
    timestamp: float
    service: str
    operation: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    user_id: str = ""
    session_id: str = ""

class CostTracker:
    """成本追踪器"""
    
    def __init__(self):
        self.entries: List[CostEntry] = []
        self.cost_limits: Dict[str, float] = {}
        self.cost_alerts: List[Dict] = []
        
        # 模型定价（每 1K tokens）
        self.pricing = {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125}
        }
    
    def record_cost(
        self,
        service: str,
        operation: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        user_id: str = "",
        session_id: str = ""
    ):
        """记录成本"""
        # 计算成本
        pricing = self.pricing.get(model, {"input": 0.005, "output": 0.015})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000
        
        entry = CostEntry(
            timestamp=time.time(),
            service=service,
            operation=operation,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            user_id=user_id,
            session_id=session_id
        )
        
        self.entries.append(entry)
        
        # 检查成本限制
        self._check_cost_limits(cost)
    
    def set_cost_limit(self, period: str, limit: float):
        """设置成本限制"""
        self.cost_limits[period] = limit
    
    def _check_cost_limits(self, new_cost: float):
        """检查成本限制"""
        for period, limit in self.cost_limits.items():
            current_cost = self.get_cost_summary(period)["total_cost"]
            
            if current_cost + new_cost > limit:
                self.cost_alerts.append({
                    "period": period,
                    "current_cost": current_cost,
                    "limit": limit,
                    "new_cost": new_cost,
                    "timestamp": time.time()
                })
    
    def get_cost_summary(
        self,
        period: str = "day",
        group_by: str = "model"
    ) -> Dict[str, Any]:
        """获取成本摘要"""
        # 计算时间范围
        now = time.time()
        if period == "hour":
            start_time = now - 3600
        elif period == "day":
            start_time = now - 86400
        elif period == "week":
            start_time = now - 604800
        elif period == "month":
            start_time = now - 2592000
        else:
            start_time = 0
        
        # 过滤条目
        relevant_entries = [e for e in self.entries if e.timestamp >= start_time]
        
        # 分组统计
        grouped_costs = {}
        total_cost = 0
        total_input_tokens = 0
        total_output_tokens = 0
        
        for entry in relevant_entries:
            group_key = getattr(entry, group_by, "unknown")
            
            if group_key not in grouped_costs:
                grouped_costs[group_key] = {
                    "cost": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "count": 0
                }
            
            grouped_costs[group_key]["cost"] += entry.cost_usd
            grouped_costs[group_key]["input_tokens"] += entry.input_tokens
            grouped_costs[group_key]["output_tokens"] += entry.output_tokens
            grouped_costs[group_key]["count"] += 1
            
            total_cost += entry.cost_usd
            total_input_tokens += entry.input_tokens
            total_output_tokens += entry.output_tokens
        
        return {
            "period": period,
            "group_by": group_by,
            "total_cost": total_cost,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_requests": len(relevant_entries),
            "grouped_costs": grouped_costs,
            "average_cost_per_request": total_cost / len(relevant_entries) if relevant_entries else 0
        }
    
    def get_cost_trend(
        self,
        days: int = 30,
        granularity: str = "day"
    ) -> List[Dict[str, Any]]:
        """获取成本趋势"""
        now = time.time()
        start_time = now - days * 86400
        
        # 按天分组
        daily_costs = {}
        
        for entry in self.entries:
            if entry.timestamp < start_time:
                continue
            
            # 获取日期
            date = datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d")
            
            if date not in daily_costs:
                daily_costs[date] = {
                    "date": date,
                    "cost": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "requests": 0
                }
            
            daily_costs[date]["cost"] += entry.cost_usd
            daily_costs[date]["input_tokens"] += entry.input_tokens
            daily_costs[date]["output_tokens"] += entry.output_tokens
            daily_costs[date]["requests"] += 1
        
        # 转换为列表并排序
        trend = sorted(daily_costs.values(), key=lambda x: x["date"])
        
        return trend
    
    def predict_cost(
        self,
        forecast_days: int = 30,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """预测成本"""
        # 获取最近 7 天的数据
        recent_trend = self.get_cost_trend(days=7)
        
        if not recent_trend:
            return {"prediction": 0, "confidence": 0}
        
        # 计算日均成本
        daily_costs = [day["cost"] for day in recent_trend]
        avg_daily_cost = sum(daily_costs) / len(daily_costs)
        
        # 计算标准差
        if len(daily_costs) > 1:
            std_dev = np.std(daily_costs)
        else:
            std_dev = avg_daily_cost * 0.2
        
        # 预测
        predicted_total = avg_daily_cost * forecast_days
        margin = std_dev * forecast_days * (1.96 if confidence_level == 0.95 else 1.645)
        
        return {
            "forecast_days": forecast_days,
            "avg_daily_cost": avg_daily_cost,
            "predicted_total": predicted_total,
            "confidence_interval": {
                "lower": predicted_total - margin,
                "upper": predicted_total + margin
            },
            "confidence_level": confidence_level
        }
    
    def generate_cost_report(self) -> Dict[str, Any]:
        """生成成本报告"""
        daily_summary = self.get_cost_summary(period="day")
        weekly_summary = self.get_cost_summary(period="week")
        monthly_summary = self.get_cost_summary(period="month")
        
        prediction = self.predict_cost(forecast_days=30)
        
        return {
            "timestamp": time.time(),
            "daily": daily_summary,
            "weekly": weekly_summary,
            "monthly": monthly_summary,
            "prediction": prediction,
            "cost_alerts": self.cost_alerts[-10:],  # 最近 10 个告警
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成成本优化建议"""
        recommendations = []
        
        # 分析模型使用
        model_costs = {}
        for entry in self.entries[-1000:]:  # 最近 1000 条
            if entry.model not in model_costs:
                model_costs[entry.model] = 0
            model_costs[entry.model] += entry.cost_usd
        
        # 检查是否可以使用更便宜的模型
        if "gpt-4o" in model_costs and model_costs["gpt-4o"] > 10:
            recommendations.append(
                "考虑将部分简单任务迁移到 gpt-4o-mini 以降低成本"
            )
        
        # 检查是否有高成本用户
        user_costs = {}
        for entry in self.entries[-1000:]:
            if entry.user_id:
                if entry.user_id not in user_costs:
                    user_costs[entry.user_id] = 0
                user_costs[entry.user_id] += entry.cost_usd
        
        if user_costs:
            max_user_cost = max(user_costs.values())
            if max_user_cost > 50:
                recommendations.append(
                    f"用户 {max(user_costs, key=user_costs.get)} 成本较高，建议检查使用模式"
                )
        
        return recommendations
```

### 28.11.2 成本可视化

```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import List, Dict

class CostVisualizer:
    """成本可视化器"""
    
    def __init__(self, cost_tracker: CostTracker):
        self.tracker = cost_tracker
    
    def plot_cost_trend(self, days: int = 30, save_path: str = None):
        """绘制成本趋势图"""
        trend = self.tracker.get_cost_trend(days=days)
        
        dates = [datetime.strptime(day["date"], "%Y-%m-%d") for day in trend]
        costs = [day["cost"] for day in trend]
        
        plt.figure(figsize=(12, 6))
        plt.plot(dates, costs, marker='o', linewidth=2, markersize=6)
        plt.fill_between(dates, costs, alpha=0.3)
        
        plt.title('Daily Cost Trend', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Cost (USD)')
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_cost_by_model(self, period: str = "month", save_path: str = None):
        """按模型绘制成本分布"""
        summary = self.tracker.get_cost_summary(period=period, group_by="model")
        
        models = list(summary["grouped_costs"].keys())
        costs = [summary["grouped_costs"][m]["cost"] for m in models]
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.Set3(range(len(models)))
        
        wedges, texts, autotexts = plt.pie(
            costs,
            labels=models,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            pctdistance=0.85
        )
        
        # 美化
        for text in texts:
            text.set_fontsize(11)
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        plt.title(f'Cost Distribution by Model ({period})', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_cost_forecast(self, forecast_days: int = 30, save_path: str = None):
        """绘制成本预测图"""
        # 获取历史数据
        historical_trend = self.tracker.get_cost_trend(days=30)
        
        # 获取预测
        prediction = self.tracker.predict_cost(forecast_days=forecast_days)
        
        # 准备数据
        historical_dates = [datetime.strptime(day["date"], "%Y-%m-%d") for day in historical_trend]
        historical_costs = [day["cost"] for day in historical_trend]
        
        # 生成预测日期
        last_date = historical_dates[-1] if historical_dates else datetime.now()
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
        forecast_costs = [prediction["avg_daily_cost"]] * forecast_days
        
        # 绘图
        plt.figure(figsize=(14, 7))
        
        # 历史数据
        plt.plot(historical_dates, historical_costs, 'b-o', linewidth=2, markersize=6, label='Historical')
        
        # 预测数据
        plt.plot(forecast_dates, forecast_costs, 'r--', linewidth=2, label='Forecast')
        
        # 置信区间
        lower_bound = prediction["confidence_interval"]["lower"] / forecast_days
        upper_bound = prediction["confidence_interval"]["upper"] / forecast_days
        plt.fill_between(
            forecast_dates,
            [lower_bound] * forecast_days,
            [upper_bound] * forecast_days,
            alpha=0.3,
            color='red',
            label='95% Confidence Interval'
        )
        
        plt.title('Cost Forecast', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Daily Cost (USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
```

---

## 28.12 安全监控

### 28.12.1 安全事件监控

```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

class SecurityEventType(Enum):
    """安全事件类型"""
    INJECTION_ATTEMPT = "injection_attempt"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION = "data_exfiltration"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_INPUT = "suspicious_input"
    TOOL_ABUSE = "tool_abuse"
    PRIVILEGE_ESCALATION = "privilege_escalation"

@dataclass
class SecurityEvent:
    """安全事件"""
    event_type: SecurityEventType
    timestamp: float
    severity: str  # low, medium, high, critical
    source: str
    description: str
    metadata: Dict[str, Any]
    user_id: str = ""
    session_id: str = ""

class SecurityMonitor:
    """安全监控系统"""
    
    def __init__(self):
        self.events: List[SecurityEvent] = []
        self.alert_rules: Dict[SecurityEventType, Dict] = {}
        self.blocked_ips: List[str] = []
        self.suspicious_patterns: List[str] = []
        
        self._initialize_rules()
    
    def _initialize_rules(self):
        """初始化告警规则"""
        self.alert_rules = {
            SecurityEventType.INJECTION_ATTEMPT: {
                "threshold": 3,
                "window": 300,
                "action": "block"
            },
            SecurityEventType.UNAUTHORIZED_ACCESS: {
                "threshold": 5,
                "window": 600,
                "action": "alert"
            },
            SecurityEventType.DATA_EXFILTRATION: {
                "threshold": 1,
                "window": 0,
                "action": "block"
            },
            SecurityEventType.RATE_LIMIT_EXCEEDED: {
                "threshold": 10,
                "window": 60,
                "action": "throttle"
            },
            SecurityEventType.SUSPICIOUS_INPUT: {
                "threshold": 5,
                "window": 300,
                "action": "alert"
            },
            SecurityEventType.TOOL_ABUSE: {
                "threshold": 3,
                "window": 300,
                "action": "block"
            }
        }
    
    def record_event(
        self,
        event_type: SecurityEventType,
        severity: str,
        source: str,
        description: str,
        metadata: Dict[str, Any] = None,
        user_id: str = "",
        session_id: str = ""
    ):
        """记录安全事件"""
        event = SecurityEvent(
            event_type=event_type,
            timestamp=time.time(),
            severity=severity,
            source=source,
            description=description,
            metadata=metadata or {},
            user_id=user_id,
            session_id=session_id
        )
        
        self.events.append(event)
        
        # 检查是否需要告警
        self._check_alert_rules(event)
        
        # 记录到日志
        self._log_event(event)
    
    def _check_alert_rules(self, event: SecurityEvent):
        """检查告警规则"""
        if event.event_type not in self.alert_rules:
            return
        
        rule = self.alert_rules[event.event_type]
        window = rule["window"]
        
        # 计算窗口内的事件数
        if window > 0:
            window_start = time.time() - window
            recent_events = [
                e for e in self.events
                if e.event_type == event.event_type and e.timestamp >= window_start
            ]
            event_count = len(recent_events)
        else:
            event_count = sum(1 for e in self.events if e.event_type == event.event_type)
        
        # 检查阈值
        if event_count >= rule["threshold"]:
            self._trigger_security_alert(event, rule["action"], event_count)
    
    def _trigger_security_alert(self, event: SecurityEvent, action: str, count: int):
        """触发安全告警"""
        alert = {
            "event_type": event.event_type.value,
            "severity": event.severity,
            "action": action,
            "event_count": count,
            "threshold": self.alert_rules[event.event_type]["threshold"],
            "timestamp": time.time(),
            "description": f"检测到 {count} 次 {event.event_type.value} 事件"
        }
        
        # 执行动作
        if action == "block":
            if event.source not in self.blocked_ips:
                self.blocked_ips.append(event.source)
                alert["action_taken"] = f"已封禁 IP: {event.source}"
        
        elif action == "throttle":
            alert["action_taken"] = f"已对 {event.source} 进行限流"
        
        # 发送告警通知
        self._send_alert_notification(alert)
    
    def _send_alert_notification(self, alert: Dict):
        """发送告警通知"""
        # 这里可以集成各种通知渠道
        # 例如：Slack, Email, PagerDuty 等
        print(f"Security Alert: {json.dumps(alert, indent=2)}")
    
    def _log_event(self, event: SecurityEvent):
        """记录事件日志"""
        log_entry = {
            "event_type": event.event_type.value,
            "timestamp": datetime.fromtimestamp(event.timestamp).isoformat(),
            "severity": event.severity,
            "source": event.source,
            "description": event.description,
            "metadata": event.metadata
        }
        
        # 写入日志文件
        with open("security_events.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def get_events(
        self,
        event_type: SecurityEventType = None,
        severity: str = None,
        start_time: float = None,
        end_time: float = None,
        limit: int = 100
    ) -> List[SecurityEvent]:
        """获取安全事件"""
        filtered = self.events
        
        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]
        
        if severity:
            filtered = [e for e in filtered if e.severity == severity]
        
        if start_time:
            filtered = [e for e in filtered if e.timestamp >= start_time]
        
        if end_time:
            filtered = [e for e in filtered if e.timestamp <= end_time]
        
        return filtered[-limit:]
    
    def get_security_report(self) -> Dict[str, Any]:
        """获取安全报告"""
        now = time.time()
        
        # 最近 24 小时的事件
        last_24h = [e for e in self.events if e.timestamp >= now - 86400]
        
        # 按类型统计
        type_stats = {}
        for event in last_24h:
            event_type = event.event_type.value
            if event_type not in type_stats:
                type_stats[event_type] = {"count": 0, "severities": {}}
            type_stats[event_type]["count"] += 1
            type_stats[event_type]["severities"][event.severity] = \
                type_stats[event_type]["severities"].get(event.severity, 0) + 1
        
        # 按严重程度统计
        severity_stats = {}
        for event in last_24h:
            severity_stats[event.severity] = severity_stats.get(event.severity, 0) + 1
        
        return {
            "timestamp": time.time(),
            "period": "24h",
            "total_events": len(last_24h),
            "events_by_type": type_stats,
            "events_by_severity": severity_stats,
            "blocked_ips": self.blocked_ips,
            "top_sources": self._get_top_sources(last_24h),
            "recommendations": self._generate_security_recommendations(last_24h)
        }
    
    def _get_top_sources(self, events: List[SecurityEvent], top_k: int = 5) -> List[Dict]:
        """获取主要来源"""
        source_counts = {}
        for event in events:
            source = event.source
            if source not in source_counts:
                source_counts[source] = 0
            source_counts[source] += 1
        
        sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [{"source": s, "count": c} for s, c in sorted_sources]
    
    def _generate_security_recommendations(self, events: List[SecurityEvent]) -> List[str]:
        """生成安全建议"""
        recommendations = []
        
        # 检查注入攻击
        injection_count = sum(1 for e in events if e.event_type == SecurityEventType.INJECTION_ATTEMPT)
        if injection_count > 5:
            recommendations.append(
                "检测到多次注入攻击尝试，建议加强输入验证和 WAF 规则"
            )
        
        # 检查未授权访问
        unauthorized_count = sum(1 for e in events if e.event_type == SecurityEventType.UNAUTHORIZED_ACCESS)
        if unauthorized_count > 3:
            recommendations.append(
                "检测到未授权访问尝试，建议检查认证机制和权限设置"
            )
        
        # 检查数据泄露
        exfiltration_count = sum(1 for e in events if e.event_type == SecurityEventType.DATA_EXFILTRATION)
        if exfiltration_count > 0:
            recommendations.append(
                "检测到数据泄露风险，建议立即审查数据访问日志"
            )
        
        return recommendations
```

---

## 28.13 合规审计

### 28.13.1 审计日志系统

```python
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class AuditAction(Enum):
    """审计操作类型"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    EXPORT = "export"
    SHARE = "share"

@dataclass
class AuditLog:
    """审计日志"""
    log_id: str
    timestamp: float
    user_id: str
    action: AuditAction
    resource_type: str
    resource_id: str
    details: Dict[str, Any]
    ip_address: str
    user_agent: str
    session_id: str

class AuditLogger:
    """审计日志系统"""
    
    def __init__(self):
        self.logs: List[AuditLog] = []
        self.retention_days: int = 90
        self.sensitive_fields: List[str] = ["password", "api_key", "secret"]
    
    def log(
        self,
        user_id: str,
        action: AuditAction,
        resource_type: str,
        resource_id: str,
        details: Dict[str, Any] = None,
        ip_address: str = "",
        user_agent: str = "",
        session_id: str = ""
    ):
        """记录审计日志"""
        log_id = f"audit_{int(time.time() * 1000)}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
        
        # 脱敏敏感信息
        sanitized_details = self._sanitize_details(details or {})
        
        audit_log = AuditLog(
            log_id=log_id,
            timestamp=time.time(),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            details=sanitized_details,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id
        )
        
        self.logs.append(audit_log)
        
        # 持久化存储
        self._persist_log(audit_log)
        
        return log_id
    
    def _sanitize_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """脱敏敏感信息"""
        sanitized = {}
        
        for key, value in details.items():
            if any(field in key.lower() for field in self.sensitive_fields):
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_details(value)
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _persist_log(self, audit_log: AuditLog):
        """持久化日志"""
        log_entry = {
            "log_id": audit_log.log_id,
            "timestamp": datetime.fromtimestamp(audit_log.timestamp).isoformat(),
            "user_id": audit_log.user_id,
            "action": audit_log.action.value,
            "resource_type": audit_log.resource_type,
            "resource_id": audit_log.resource_id,
            "details": audit_log.details,
            "ip_address": audit_log.ip_address,
            "user_agent": audit_log.user_agent,
            "session_id": audit_log.session_id
        }
        
        # 写入审计日志文件
        with open("audit_logs.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def query_logs(
        self,
        user_id: str = None,
        action: AuditAction = None,
        resource_type: str = None,
        start_time: float = None,
        end_time: float = None,
        limit: int = 1000
    ) -> List[AuditLog]:
        """查询审计日志"""
        filtered = self.logs
        
        if user_id:
            filtered = [l for l in filtered if l.user_id == user_id]
        
        if action:
            filtered = [l for l in filtered if l.action == action]
        
        if resource_type:
            filtered = [l for l in filtered if l.resource_type == resource_type]
        
        if start_time:
            filtered = [l for l in filtered if l.timestamp >= start_time]
        
        if end_time:
            filtered = [l for l in filtered if l.timestamp <= end_time]
        
        return filtered[-limit:]
    
    def generate_compliance_report(self, period: str = "month") -> Dict[str, Any]:
        """生成合规报告"""
        now = time.time()
        
        if period == "day":
            start_time = now - 86400
        elif period == "week":
            start_time = now - 604800
        elif period == "month":
            start_time = now - 2592000
        else:
            start_time = 0
        
        relevant_logs = [l for l in self.logs if l.timestamp >= start_time]
        
        # 统计分析
        user_activity = {}
        action_stats = {}
        resource_access = {}
        
        for log in relevant_logs:
            # 用户活动
            if log.user_id not in user_activity:
                user_activity[log.user_id] = {"count": 0, "actions": set()}
            user_activity[log.user_id]["count"] += 1
            user_activity[log.user_id]["actions"].add(log.action.value)
            
            # 操作统计
            action_stats[log.action.value] = action_stats.get(log.action.value, 0) + 1
            
            # 资源访问
            resource_key = f"{log.resource_type}:{log.resource_id}"
            resource_access[resource_key] = resource_access.get(resource_key, 0) + 1
        
        # 转换 set 为 list 以便 JSON 序列化
        for user_data in user_activity.values():
            user_data["actions"] = list(user_data["actions"])
        
        return {
            "period": period,
            "total_events": len(relevant_logs),
            "unique_users": len(user_activity),
            "user_activity": user_activity,
            "action_stats": action_stats,
            "top_resources": sorted(resource_access.items(), key=lambda x: x[1], reverse=True)[:10],
            "compliance_score": self._calculate_compliance_score(relevant_logs)
        }
    
    def _calculate_compliance_score(self, logs: List[AuditLog]) -> float:
        """计算合规分数"""
        if not logs:
            return 100.0
        
        # 检查各种合规要求
        checks = []
        
        # 1. 检查是否有删除操作
        delete_count = sum(1 for l in logs if l.action == AuditAction.DELETE)
        checks.append(delete_count == 0)
        
        # 2. 检查敏感操作是否有详细记录
        sensitive_actions = [AuditAction.DELETE, AuditAction.EXPORT, AuditAction.SHARE]
        sensitive_logs = [l for l in logs if l.action in sensitive_actions]
        all_have_details = all(bool(l.details) for l in sensitive_logs)
        checks.append(all_have_details)
        
        # 3. 检查是否有异常时间的操作
        now = time.time()
        recent_logs = [l for l in logs if now - l.timestamp < 3600]
        checks.append(len(recent_logs) < 100)  # 1小时内操作不超过100次
        
        # 计算分数
        passed_checks = sum(1 for c in checks if c)
        score = (passed_checks / len(checks)) * 100 if checks else 100.0
        
        return score
    
    def export_audit_logs(
        self,
        format: str = "json",
        start_time: float = None,
        end_time: float = None
    ) -> str:
        """导出审计日志"""
        filtered = self.logs
        
        if start_time:
            filtered = [l for l in filtered if l.timestamp >= start_time]
        
        if end_time:
            filtered = [l for l in filtered if l.timestamp <= end_time]
        
        if format == "json":
            return json.dumps([
                {
                    "log_id": l.log_id,
                    "timestamp": datetime.fromtimestamp(l.timestamp).isoformat(),
                    "user_id": l.user_id,
                    "action": l.action.value,
                    "resource_type": l.resource_type,
                    "resource_id": l.resource_id,
                    "details": l.details,
                    "ip_address": l.ip_address
                }
                for l in filtered
            ], indent=2)
        
        elif format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            writer.writerow([
                "log_id", "timestamp", "user_id", "action",
                "resource_type", "resource_id", "ip_address"
            ])
            
            for log in filtered:
                writer.writerow([
                    log.log_id,
                    datetime.fromtimestamp(log.timestamp).isoformat(),
                    log.user_id,
                    log.action.value,
                    log.resource_type,
                    log.resource_id,
                    log.ip_address
                ])
            
            return output.getvalue()
        
        return ""
```

---

## 28.14 监控最佳实践

### 28.14.1 监控策略框架

```python
class MonitoringStrategy:
    """监控策略框架"""
    
    def __init__(self):
        self.strategies: Dict[str, Dict] = {}
        self.best_practices: List[Dict] = []
        
        self._initialize_strategies()
    
    def _initialize_strategies(self):
        """初始化监控策略"""
        self.strategies = {
            "agent_performance": {
                "description": "Agent 性能监控",
                "metrics": [
                    "request_latency",
                    "llm_latency",
                    "token_usage",
                    "error_rate"
                ],
                "alerts": [
                    {"metric": "request_latency", "threshold": 5000, "severity": "warning"},
                    {"metric": "error_rate", "threshold": 5, "severity": "critical"}
                ],
                "dashboard": "agent_performance_dashboard"
            },
            "cost_monitoring": {
                "description": "成本监控",
                "metrics": [
                    "daily_cost",
                    "cost_per_request",
                    "token_cost"
                ],
                "alerts": [
                    {"metric": "daily_cost", "threshold": 100, "severity": "warning"}
                ],
                "dashboard": "cost_monitoring_dashboard"
            },
            "security_monitoring": {
                "description": "安全监控",
                "metrics": [
                    "injection_attempts",
                    "unauthorized_access",
                    "data_exfiltration"
                ],
                "alerts": [
                    {"metric": "injection_attempts", "threshold": 10, "severity": "critical"}
                ],
                "dashboard": "security_dashboard"
            }
        }
        
        self.best_practices = [
            {
                "category": "指标设计",
                "practices": [
                    "使用有意义的指标名称",
                    "为指标添加标签以支持多维分析",
                    "区分计数器、直方图和仪表的使用场景",
                    "设置合理的指标保留策略"
                ]
            },
            {
                "category": "告警配置",
                "practices": [
                    "设置分级告警（info, warning, critical）",
                    "配置告警抑制避免告警风暴",
                    "为告警设置合理的超时时间",
                    "定期审查和调整告警阈值"
                ]
            },
            {
                "category": "仪表板设计",
                "practices": [
                    "按角色设计不同的仪表板视图",
                    "使用一致的可视化规范",
                    "保持仪表板简洁，突出关键信息",
                    "支持下钻查看详细数据"
                ]
            },
            {
                "category": "日志管理",
                "practices": [
                    "使用结构化日志格式",
                    "实施日志分级和采样",
                    "配置日志保留和归档策略",
                    "确保日志的安全性和隐私保护"
                ]
            }
        ]
    
    def get_strategy(self, strategy_name: str) -> Dict:
        """获取监控策略"""
        return self.strategies.get(strategy_name, {})
    
    def get_best_practices(self, category: str = None) -> List[Dict]:
        """获取最佳实践"""
        if category:
            return [p for p in self.best_practices if p["category"] == category]
        return self.best_practices
    
    def validate_monitoring_setup(self, setup: Dict[str, Any]) -> Dict[str, Any]:
        """验证监控设置"""
        issues = []
        recommendations = []
        
        # 检查指标覆盖
        required_metrics = ["request_latency", "error_rate", "resource_usage"]
        for metric in required_metrics:
            if metric not in setup.get("metrics", []):
                issues.append(f"缺少必要的指标: {metric}")
                recommendations.append(f"添加 {metric} 指标监控")
        
        # 检查告警配置
        if "alerts" not in setup:
            issues.append("未配置告警")
            recommendations.append("配置关键指标的告警规则")
        
        # 检查仪表板
        if "dashboards" not in setup:
            issues.append("未配置仪表板")
            recommendations.append("创建关键指标的可视化仪表板")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations,
            "score": max(0, 100 - len(issues) * 20)
        }
```

---

## 28.15 案例研究

### 28.15.1 大规模 Agent 监控案例

```python
class LargeScaleAgentMonitoringCase:
    """大规模 Agent 监控案例"""
    
    def __init__(self):
        self.case_study = {
            "company": "某大型科技公司",
            "scale": {
                "daily_requests": 10_000_000,
                "concurrent_users": 100_000,
                "agent_types": 15,
                "regions": 5
            },
            "challenges": [
                "高并发下的性能监控",
                "多地区数据一致性",
                "实时成本控制",
                "异常行为检测"
            ],
            "solutions": [],
            "results": {}
        }
    
    async def implement_monitoring(self) -> Dict[str, Any]:
        """实施监控方案"""
        # 1. 分布式追踪
        tracing_solution = {
            "tool": "Jaeger + OpenTelemetry",
            "implementation": "全链路追踪，采样率 1%",
            "result": "定位性能瓶颈时间从 2 小时缩短到 5 分钟"
        }
        
        # 2. 实时指标
        metrics_solution = {
            "tool": "Prometheus + Grafana",
            "implementation": "自定义指标，15 秒采集间隔",
            "result": "实时监控 50+ 关键指标"
        }
        
        # 3. 日志聚合
        logging_solution = {
            "tool": "ELK Stack",
            "implementation": "结构化日志，7 天热存储",
            "result": "日志查询时间从分钟级降到秒级"
        }
        
        # 4. 成本监控
        cost_solution = {
            "tool": "自研成本监控系统",
            "implementation": "实时 Token 计数，预算告警",
            "result": "月度成本降低 30%"
        }
        
        # 5. 安全监控
        security_solution = {
            "tool": "自研安全监控",
            "implementation": "实时异常检测，自动响应",
            "result": "安全事件响应时间从小时级降到分钟级"
        }
        
        self.case_study["solutions"] = [
            tracing_solution,
            metrics_solution,
            logging_solution,
            cost_solution,
            security_solution
        ]
        
        self.case_study["results"] = {
            "uptime": "99.99%",
            "avg_latency": "200ms",
            "cost_reduction": "30%",
            "mttr": "5分钟",
            "security_incidents": "0"
        }
        
        return self.case_study
```

---

## 28.16 常见问题和解决方案

### 28.16.1 监控问题排查

```python
class MonitoringTroubleshooting:
    """监控问题排查指南"""
    
    @staticmethod
    def get_common_issues() -> List[Dict[str, Any]]:
        """获取常见问题"""
        return [
            {
                "issue": "指标数据缺失",
                "symptoms": ["仪表板显示空白", "告警未触发"],
                "causes": [
                    "指标采集器未运行",
                    "网络连接问题",
                    "指标名称配置错误",
                    "时间范围设置不当"
                ],
                "solutions": [
                    "检查采集器状态",
                    "验证网络连接",
                    "确认指标名称",
                    "调整时间范围"
                ]
            },
            {
                "issue": "告警风暴",
                "symptoms": ["大量重复告警", "告警通知过多"],
                "causes": [
                    "告警阈值设置过低",
                    "缺少告警抑制",
                    "依赖项故障导致连锁反应"
                ],
                "solutions": [
                    "调整告警阈值",
                    "配置告警抑制规则",
                    "实施告警分组",
                    "设置告警超时"
                ]
            },
            {
                "issue": "性能影响",
                "symptoms": ["系统变慢", "资源使用率高"],
                "causes": [
                    "指标采集频率过高",
                    "日志量过大",
                    "查询效率低"
                ],
                "solutions": [
                    "降低采集频率",
                    "实施日志采样",
                    "优化查询语句",
                    "增加缓存"
                ]
            },
            {
                "issue": "数据不一致",
                "symptoms": ["不同系统数据不匹配", "历史数据丢失"],
                "causes": [
                    "时区问题",
                    "数据同步延迟",
                    "存储配置问题"
                ],
                "solutions": [
                    "统一时区设置",
                    "优化同步机制",
                    "检查存储配置",
                    "实施数据校验"
                ]
            }
        ]
    
    @staticmethod
    def get_diagnostic_checklist() -> List[Dict[str, str]]:
        """获取诊断清单"""
        return [
            {"check": "确认监控服务状态", "command": "systemctl status prometheus"},
            {"check": "检查指标端点", "command": "curl http://localhost:9090/metrics"},
            {"check": "查看告警规则", "command": "curl http://localhost:9090/api/v1/rules"},
            {"check": "检查日志采集", "command": "docker logs logstash"},
            {"check": "验证网络连接", "command": "netstat -tlnp | grep 9090"},
            {"check": "查看资源使用", "command": "top -bn1 | head -20"}
        ]
```

---

## 28.17 本章小结（更新版）

| 知识点 | 核心要点 |
|:---|:---|
| **三支柱** | Tracing 追踪调用链、Metrics 监控性能、Logging 记录事件 |
| **OpenTelemetry** | 标准化的可观测性框架，支持 Traces、Metrics、Logs |
| **LangSmith** | LangChain 官方追踪平台，提供丰富的 Agent 可视化 |
| **Langfuse** | 开源替代方案，支持自托管 |
| **Prometheus + Grafana** | 指标收集和可视化，支持自定义仪表板 |
| **结构化日志** | 使用 structlog 或类似库记录结构化日志 |
| **告警系统** | 基于 Prometheus 规则的多渠道告警通知 |
| **成本监控** | 实时跟踪 Token 使用和 API 成本 |
| **调试工具** | 交互式调试器，支持断点和状态检查 |
| **异常检测** | 统计方法、时间序列分析、季节性检测 |
| **性能优化** | 瓶颈分析、优化计划、SLA 监控 |
| **安全监控** | 安全事件检测、自动响应、合规审计 |
| **最佳实践** | 监控策略框架、问题排查指南 |

> **下一章预告**
>
> 在第 29 章中，我们将学习 MCP 与 A2A 协议。

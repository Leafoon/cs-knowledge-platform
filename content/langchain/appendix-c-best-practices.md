# Appendix C: 最佳实践清单

> **本附录提供 LangChain 生态开发的最佳实践指南，涵盖提示设计、链构建、Agent 开发、生产部署等各环节的可操作检查清单。**

---

## C.1 Prompt 设计原则

### ✅ 基本原则清单

- [ ] **明确角色与任务**：在 system prompt 中清晰定义 AI 的角色、职责、约束条件
- [ ] **提供上下文**：包含必要的背景信息、示例、格式说明
- [ ] **结构化输出**：明确期望的输出格式（JSON、Markdown、列表等）
- [ ] **边界设定**：告知 AI 能做什么、不能做什么、何时应拒绝回答
- [ ] **Few-Shot 示例**：对于复杂任务，提供 2-5 个高质量示例
- [ ] **版本管理**：使用 LangChain Hub 或 Git 管理提示模板版本

### 示例：优秀 vs 糟糕的提示

**❌ 糟糕的提示**：
```python
prompt = "翻译这段话：{text}"
```

**问题**：
- 未指定目标语言
- 未定义翻译风格（正式/口语）
- 未处理特殊术语

**✅ 优秀的提示**：
```python
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional translator specializing in technical documentation.

Rules:
- Translate from {source_lang} to {target_lang}
- Maintain technical terminology in English (e.g., "API", "Docker")
- Use formal tone for documentation
- Preserve formatting (Markdown, code blocks)
- If uncertain about a term, add [译注: ...] annotation

Examples:
Input: "The API endpoint returns a JSON response."
Output: "该 API 端点返回 JSON 响应。"
"""),
    ("human", "{text}")
])
```

### Prompt 测试检查清单

- [ ] **边界测试**：用极端输入测试（空字符串、超长文本、特殊字符）
- [ ] **语言测试**：测试多种语言（中文、英文、日文等）
- [ ] **格式健壮性**：验证输出格式一致性（运行 100 次，统计解析成功率）
- [ ] **错误处理**：测试 AI 如何处理无效输入或不合理请求
- [ ] **A/B 测试**：使用 LangSmith 对比不同提示版本的效果

---

## C.2 LCEL 链设计模式

### ✅ 设计原则

- [ ] **单一职责**：每个链只做一件事（翻译 OR 摘要，不混合）
- [ ] **类型安全**：使用 Pydantic 或 TypedDict 定义输入输出
- [ ] **可组合性**：设计链时考虑与其他链的组合可能性
- [ ] **幂等性**：相同输入应产生相同输出（除非显式引入随机性）
- [ ] **错误边界**：每个链应能优雅处理异常（使用 `with_fallbacks`）

### 链构建检查清单

#### 性能优化
- [ ] 使用 `batch()` 替代循环调用 `invoke()`
- [ ] 对于 I/O 密集型操作，使用 `ainvoke()` / `abatch()`
- [ ] 将可并行的步骤用 `RunnableParallel` 组合
- [ ] 启用流式输出（`astream`）提升用户体验

#### 可观测性
- [ ] 设置有意义的 `run_name`（便于在 LangSmith 中查找）
- [ ] 添加 `tags` 标记环境（production / staging / test）
- [ ] 在关键步骤添加 `metadata`（user_id、request_id 等）
- [ ] 配置 `callbacks` 记录自定义指标

#### 可靠性
- [ ] 为 LLM 调用添加 `with_retry()`（至少 2-3 次重试）
- [ ] 为关键链配置 `with_fallbacks()`（降级到更简单的模型或规则）
- [ ] 设置 `timeout` 避免无限等待
- [ ] 验证输出格式（使用 Pydantic 或自定义验证器）

### 示例：生产级链设计

```python
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

# 1. 定义类型
class TranslationInput(BaseModel):
    text: str = Field(max_length=5000)
    target_lang: Literal["en", "zh", "ja", "fr"]

class TranslationOutput(BaseModel):
    translation: str
    confidence: float = Field(ge=0, le=1)

# 2. 主模型 + 降级模型
primary_model = ChatOpenAI(model="gpt-4", temperature=0)
fallback_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 3. 带重试和降级的模型
reliable_model = primary_model.with_retry(
    stop_after_attempt=2
).with_fallbacks([fallback_model])

# 4. 提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a translator. Translate to {target_lang}. Return JSON: {{\"translation\": \"...\", \"confidence\": 0.95}}"),
    ("human", "{text}")
])

# 5. 输出解析
parser = PydanticOutputParser(pydantic_object=TranslationOutput)

# 6. 完整链
translation_chain = (
    prompt
    | reliable_model
    | parser
).with_config(
    run_name="production-translation",
    tags=["production", "translation"],
    max_concurrency=10
)

# 7. 使用
result = translation_chain.invoke({
    "text": "Hello world",
    "target_lang": "zh"
})
print(result)  # TranslationOutput(translation="你好世界", confidence=0.98)
```

---

## C.3 Agent 可靠性检查表

### ✅ 设计阶段

- [ ] **工具选择**：仅提供必要工具（5-10 个为宜，超过 15 个会降低性能）
- [ ] **工具文档**：每个工具必须有清晰的 `description`（LLM 据此决定是否调用）
- [ ] **工具依赖**：明确工具间的调用顺序（如：先搜索再摘要）
- [ ] **循环检测**：设置最大迭代次数（`max_iterations=10`）
- [ ] **终止条件**：明确告知 Agent 何时应停止并返回最终答案

### ✅ 工具设计

- [ ] **输入验证**：使用 Pydantic `BaseModel` 验证参数
- [ ] **错误处理**：工具内部捕获异常，返回错误信息而非抛出异常
- [ ] **幂等性**：同样参数多次调用应返回相同结果（除非本质上随机）
- [ ] **超时设置**：为长时间操作设置 `timeout`
- [ ] **结果格式化**：返回结构化、信息丰富的结果（而非简单的"成功"/"失败"）

### 工具设计示例

**❌ 糟糕的工具**：
```python
@tool
def search(query: str) -> str:
    """Search the web"""
    results = api.search(query)  # 可能抛出异常
    return results  # 可能返回复杂对象
```

**✅ 优秀的工具**：
```python
from pydantic import BaseModel, Field
from langchain.tools import tool
import logging

class SearchInput(BaseModel):
    query: str = Field(description="The search query", max_length=200)
    max_results: int = Field(default=3, ge=1, le=10)

@tool(args_schema=SearchInput)
def search(query: str, max_results: int = 3) -> str:
    """
    Search the web for information.
    
    Use this when you need current information not in your training data.
    Returns a formatted list of search results with titles and snippets.
    """
    logger = logging.getLogger(__name__)
    
    try:
        results = api.search(query, limit=max_results, timeout=5)
        
        if not results:
            return f"No results found for query: '{query}'. Try rephrasing or using different keywords."
        
        formatted = [
            f"{i+1}. {r['title']}\n   {r['snippet']}\n   Source: {r['url']}"
            for i, r in enumerate(results)
        ]
        
        return f"Found {len(results)} results:\n\n" + "\n\n".join(formatted)
    
    except TimeoutError:
        logger.warning(f"Search timeout for query: {query}")
        return "Search timed out. Please try again with a simpler query."
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"Search failed due to technical error. Please try again later."
```

### ✅ 运行时监控

- [ ] **日志记录**：记录每次工具调用（输入、输出、耗时）
- [ ] **追踪**：启用 LangSmith Tracing 查看完整决策树
- [ ] **成本监控**：追踪 token 消耗（`get_openai_callback()`）
- [ ] **性能指标**：记录任务完成率、平均迭代次数、失败原因

### ✅ 人机协作（HITL）

- [ ] **关键决策点**：在执行高风险操作前中断（如：删除数据、发送邮件）
- [ ] **审批流程**：使用 LangGraph `interrupt_before` 实现审批
- [ ] **超时处理**：如果人工 24 小时未响应，自动拒绝或降级处理
- [ ] **审计日志**：记录所有人工干预的决策

---

## C.4 生产部署 Checklist

### ✅ 部署前检查

#### 代码质量
- [ ] 所有链/Agent 都有单元测试（覆盖率 > 80%）
- [ ] 集成测试覆盖主要用户场景
- [ ] 使用 `mypy` 或 `pyright` 进行类型检查
- [ ] 代码通过 `ruff` / `black` 格式化

#### 配置管理
- [ ] API Keys 存储在环境变量或密钥管理服务（如 AWS Secrets Manager）
- [ ] 使用不同配置文件区分环境（dev / staging / prod）
- [ ] 模型版本固定（如 `gpt-4-0613` 而非 `gpt-4`）
- [ ] 超时、重试、并发数等参数可配置

#### 安全性
- [ ] 输入验证：限制输入长度、过滤恶意内容
- [ ] 输出过滤：防止泄露敏感信息（PII、API Keys）
- [ ] 速率限制：防止滥用（使用 `slowapi` 或 API Gateway）
- [ ] 身份认证：API 端点需要 API Key 或 OAuth

### ✅ 基础设施

#### 容器化
- [ ] Dockerfile 使用多阶段构建（减小镜像体积）
- [ ] 镜像基于官方 Python Slim 镜像
- [ ] 依赖使用 `poetry` 或 `pip-tools` 锁定版本
- [ ] 健康检查端点（`/health`）

#### 编排（Kubernetes）
- [ ] 设置资源限制（`requests` 和 `limits`）
- [ ] 配置 HPA（水平自动扩缩容）
- [ ] 使用 Liveness / Readiness Probes
- [ ] 配置 PodDisruptionBudget（保证高可用）

#### 可观测性
- [ ] 集成日志聚合（ELK / Loki）
- [ ] 设置指标监控（Prometheus）
- [ ] 配置分布式追踪（Jaeger / Zipkin）
- [ ] 设置告警规则（响应时间 > 5s、错误率 > 1% 等）

### ✅ 部署后验证

- [ ] 冒烟测试：调用主要端点，验证基本功能
- [ ] 性能测试：使用 `Locust` 或 `k6` 进行压力测试
- [ ] 监控 Dashboard：确认所有指标正常上报
- [ ] 金丝雀发布：先将 5% 流量切到新版本，观察 1 小时后全量发布

### 示例：Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langchain-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: langchain-app
  template:
    metadata:
      labels:
        app: langchain-app
    spec:
      containers:
      - name: app
        image: my-registry/langchain-app:v1.2.3
        ports:
        - containerPort: 8000
        env:
        - name: LANGCHAIN_API_KEY
          valueFrom:
            secretKeyRef:
              name: langchain-secrets
              key: api-key
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
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: langchain-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: langchain-app
  minReplicas: 3
  maxReplicas: 20
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

---

## C.5 性能优化 Checklist

### ✅ 延迟优化

- [ ] **缓存**：使用 Redis 缓存 LLM 响应（相同输入避免重复调用）
- [ ] **批处理**：合并多个请求（`batch()` 而非多次 `invoke()`）
- [ ] **并行化**：使用 `RunnableParallel` 并行执行独立步骤
- [ ] **流式输出**：使用 `astream()` 提前返回首个 token（降低感知延迟）
- [ ] **预加载**：在应用启动时加载模型、向量库索引

### ✅ 成本优化

- [ ] **模型路由**：简单任务用 GPT-3.5，复杂任务用 GPT-4
- [ ] **Prompt 压缩**：移除冗余内容、使用缩写
- [ ] **Few-Shot 动态选择**：根据查询难度动态选择示例数量
- [ ] **缓存策略**：对于不常变化的内容（如文档摘要），长期缓存
- [ ] **Token 预算**：设置每个请求的 `max_tokens` 上限

### ✅ RAG 优化

- [ ] **Chunk 大小调优**：测试 200/500/1000 token 的效果
- [ ] **混合检索**：结合向量检索 + BM25 关键词检索
- [ ] **重排序**：使用 `Cohere Rerank` 或 `cross-encoder` 提升 Top-K 质量
- [ ] **元数据过滤**：根据时间、来源、类型预过滤文档
- [ ] **查询改写**：使用 `MultiQueryRetriever` 生成查询变体

### 性能测试示例

```python
import time
from langchain.callbacks import get_openai_callback

# 测试脚本
test_inputs = [...]  # 100 条测试数据

# 1. 基线测试
start = time.time()
with get_openai_callback() as cb:
    results = [chain.invoke(x) for x in test_inputs]
    baseline_time = time.time() - start
    baseline_tokens = cb.total_tokens
    baseline_cost = cb.total_cost

# 2. 批处理优化
start = time.time()
with get_openai_callback() as cb:
    results = chain.batch(test_inputs)
    batch_time = time.time() - start
    batch_tokens = cb.total_tokens
    batch_cost = cb.total_cost

# 3. 缓存优化
from langchain.cache import RedisCache
langchain.llm_cache = RedisCache(redis_url="redis://localhost:6379")

start = time.time()
with get_openai_callback() as cb:
    # 第二次运行（命中缓存）
    results = chain.batch(test_inputs)
    cache_time = time.time() - start
    cache_tokens = cb.total_tokens
    cache_cost = cb.total_cost

# 对比报告
print(f"""
Performance Comparison:
======================
Baseline:
  Time: {baseline_time:.2f}s
  Tokens: {baseline_tokens}
  Cost: ${baseline_cost:.4f}

Batching:
  Time: {batch_time:.2f}s ({batch_time/baseline_time*100:.1f}%)
  Tokens: {batch_tokens}
  Cost: ${batch_cost:.4f}

Caching:
  Time: {cache_time:.2f}s ({cache_time/baseline_time*100:.1f}%)
  Tokens: {cache_tokens} ({cache_tokens/baseline_tokens*100:.1f}%)
  Cost: ${cache_cost:.4f}
""")
```

---

## C.6 LangSmith 评估最佳实践

### ✅ 数据集构建

- [ ] **代表性**：数据集应覆盖主要用户场景（常见查询 + 边界情况）
- [ ] **多样性**：包含不同长度、语言、复杂度的输入
- [ ] **真实性**：优先使用生产环境的真实数据
- [ ] **标注质量**：由领域专家标注 ground truth
- [ ] **版本控制**：数据集变更时创建新版本

### ✅ 评估指标

- [ ] **任务相关指标**：
  - QA：EM（精确匹配）、F1、ROUGE、BLEU
  - RAG：Answer Relevance、Context Relevance、Faithfulness
  - 分类：Accuracy、Precision、Recall、F1
- [ ] **通用指标**：
  - 延迟（P50、P95、P99）
  - Token 消耗
  - 成本
  - 失败率
- [ ] **人工评估**：关键场景由人类评审（如：安全性、伦理性）

### ✅ 实验管理

- [ ] **基线建立**：先运行 baseline 版本，后续版本与之对比
- [ ] **单变量实验**：每次只改一个变量（Prompt / 模型 / 参数）
- [ ] **统计显著性**：使用 t-test 验证改进是否显著（p < 0.05）
- [ ] **文档记录**：记录每个实验的假设、配置、结果、结论

---

## C.7 安全与隐私检查表

### ✅ 输入验证

- [ ] **长度限制**：限制输入最大长度（防止 DoS）
- [ ] **内容过滤**：检测并拒绝恶意 Prompt（如 Prompt Injection）
- [ ] **PII 检测**：扫描输入中的敏感信息（姓名、邮箱、手机号等）
- [ ] **SQL 注入防护**：如果涉及数据库查询，使用参数化查询

### ✅ 输出过滤

- [ ] **PII 脱敏**：使用 `presidio` 或正则表达式替换敏感信息
- [ ] **敏感词过滤**：过滤政治敏感、暴力、色情内容
- [ ] **API Key 泄露检测**：扫描输出中的 API Keys、密码等

### ✅ 访问控制

- [ ] **身份认证**：所有 API 需要认证（API Key / OAuth / JWT）
- [ ] **权限管理**：基于角色的访问控制（RBAC）
- [ ] **速率限制**：每用户/IP 限制请求频率

### 示例：输入验证中间件

```python
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, validator
import re

app = FastAPI()

class ChatInput(BaseModel):
    message: str = Field(max_length=2000)
    
    @validator("message")
    def validate_message(cls, v):
        # 1. 长度检查
        if len(v.strip()) == 0:
            raise ValueError("Message cannot be empty")
        
        # 2. 恶意 Prompt 检测（简单示例）
        injection_patterns = [
            r"ignore previous instructions",
            r"system:\s*you are now",
            r"<\s*script\s*>",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError("Potential prompt injection detected")
        
        # 3. PII 检测（简化）
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        if re.search(email_pattern, v):
            raise ValueError("Please do not include email addresses")
        
        return v

@app.post("/chat")
async def chat(input: ChatInput):
    # 输入已验证，安全处理
    result = chain.invoke({"message": input.message})
    return {"response": result}
```

---

## C.8 代码审查清单

### ✅ 通用代码质量

- [ ] 代码符合 PEP 8 规范
- [ ] 所有函数/类有 Docstring
- [ ] 复杂逻辑有注释
- [ ] 无硬编码的魔法数字（使用常量）
- [ ] 无 TODO / FIXME 遗留

### ✅ LangChain 特定

- [ ] 使用 LCEL 而非 Legacy Chains
- [ ] 所有 Runnable 都有明确的类型标注
- [ ] LLM 调用有重试机制
- [ ] 敏感信息不在日志中输出
- [ ] 长对话有记忆管理策略（摘要 / 窗口）

### ✅ 测试

- [ ] 单元测试覆盖核心逻辑
- [ ] 集成测试覆盖主要场景
- [ ] 使用 `pytest-mock` 模拟 LLM 调用（避免真实 API 开销）
- [ ] 边界情况有测试（空输入、超长输入、特殊字符）

---

**本清单持续更新，建议在每个开发阶段对照检查。**

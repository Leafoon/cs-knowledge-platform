# Chapter 16: LangSmith 可观测性与调试

## 本章概览

在生产环境中运行 LLM 应用时，可观测性（Observability）是确保系统可靠性、性能和质量的关键。LangSmith 是 LangChain 官方提供的端到端开发者平台，专注于 LLM 应用的**追踪（Tracing）**、**调试（Debugging）**、**评估（Evaluation）**和**监控（Monitoring）**。

本章将深入探讨：
- LangSmith 核心概念与架构
- Tracing 机制与 Span 层级结构
- 调试工具与提示优化
- 数据集管理与离线评估
- 在线监控与反馈循环

---

## 16.1 为什么需要 LangSmith？

### 16.1.1 LLM 应用的可观测性挑战

传统软件开发中，可观测性三大支柱是**日志（Logs）**、**指标（Metrics）**和**追踪（Traces）**。但 LLM 应用具有独特挑战：

1. **非确定性输出**：同样的输入可能产生不同输出
2. **复杂链路**：多个 LLM 调用、检索、工具执行组成的多步流程
3. **高延迟与成本**：每次 LLM 调用耗时长且消耗 Token
4. **质量难量化**：输出质量没有明确的"正确答案"
5. **上下文依赖**：需要追踪完整的对话历史和状态变化

### 16.1.2 LangSmith 的核心价值

<Callout type="success">
**LangSmith 解决的核心问题**

- **透明化执行过程**：记录每一步的输入、输出、延迟、Token 消耗
- **快速定位问题**：可视化 Trace，定位失败节点、性能瓶颈
- **系统化评估**：通过数据集和评估器量化应用质量
- **持续优化**：收集反馈，迭代提示、模型、检索策略
- **生产监控**：实时监控错误率、延迟、成本，设置警报
</Callout>

### 16.1.3 LangSmith vs 其他工具

| 特性 | LangSmith | LangFuse | Weights & Biases | Arize Phoenix |
|------|-----------|----------|------------------|---------------|
| **LangChain 集成** | 原生无缝 | 需配置 | 需配置 | 需配置 |
| **Trace 可视化** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **数据集管理** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **在线评估** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Prompt Playground** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **自托管** | ❌ (云端) | ✅ | ✅ | ✅ |
| **定价** | 免费层 + 企业 | 开源 + 企业 | 付费 | 开源 |

---

## 16.2 LangSmith 快速上手

### 16.2.1 环境配置

#### 步骤 1：获取 API Key

访问 [https://smith.langchain.com](https://smith.langchain.com) 注册账号，在设置中获取 API Key。

#### 步骤 2：配置环境变量

```bash
# .env 文件
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=lsv2_pt_xxx...  # 你的 API Key
LANGCHAIN_PROJECT=my-first-project  # 项目名称
```

#### 步骤 3：安装依赖

```bash
pip install langchain langchain-openai langsmith
```

### 16.2.2 第一个 Traced 应用

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 环境变量已设置，LangSmith 自动启用
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_pt_xxx"
os.environ["LANGCHAIN_PROJECT"] = "demo-simple-chain"

# 构建简单链
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{role}。"),
    ("user", "{input}")
])

llm = ChatOpenAI(model="gpt-4", temperature=0.7)
chain = prompt | llm | StrOutputParser()

# 执行链（自动记录到 LangSmith）
result = chain.invoke({
    "role": "Python 导师",
    "input": "解释什么是装饰器？"
})

print(result)
```

**预期输出：**
```
装饰器（Decorator）是 Python 中的一种设计模式，用于在不修改原函数代码的情况下，动态地增强函数的功能...

✅ 自动记录到 LangSmith：
   - Trace ID: a7f3e9d2-...
   - 链接: https://smith.langchain.com/o/.../projects/p/.../runs/r/...
```

在 LangSmith UI 中可以看到：
- **Trace 树**：Prompt 格式化 → LLM 调用 → 输出解析
- **每个步骤的输入/输出**
- **Token 消耗**：Prompt Tokens: 28, Completion Tokens: 156
- **延迟分布**：Total: 2.3s, LLM: 2.1s

<div data-component="LangSmithTraceVisualization"></div>

---

## 16.3 Tracing 深度解析

### 16.3.1 Trace 与 Span 层级结构

LangSmith 的追踪机制基于 **OpenTelemetry** 标准，每次执行产生一个 **Trace**，内部包含多个 **Span**。

#### Trace 层级示例

```
Root Trace (Chain Execution)
├─ Span 1: PromptTemplate.format()
│  ├─ Input: {role: "Python 导师", input: "解释装饰器"}
│  └─ Output: [SystemMessage, HumanMessage]
│
├─ Span 2: ChatOpenAI.invoke()
│  ├─ Input: [SystemMessage, HumanMessage]
│  ├─ Metadata: {model: "gpt-4", temperature: 0.7}
│  ├─ Token 使用: {prompt: 28, completion: 156, total: 184}
│  └─ Output: AIMessage(content="装饰器是...")
│
└─ Span 3: StrOutputParser.parse()
   ├─ Input: AIMessage(content="装饰器是...")
   └─ Output: "装饰器是..."
```

### 16.3.2 自动 vs 手动 Tracing

#### 自动 Tracing

所有 LangChain 组件（Runnable）默认支持自动追踪：

```python
from langchain_core.runnables import RunnableLambda

# 自定义函数也会被追踪
def custom_processor(text: str) -> str:
    return text.upper()

# 包装为 Runnable 后自动追踪
runnable_processor = RunnableLambda(custom_processor)

chain = prompt | llm | runnable_processor
chain.invoke({"role": "助手", "input": "hello"})
```

#### 手动 Tracing（自定义 Span）

对于复杂业务逻辑，可手动创建 Span：

```python
from langsmith import trace

@trace(name="数据预处理", run_type="tool")
def preprocess_data(data: dict) -> dict:
    """自定义 Span 追踪数据处理过程"""
    # 复杂处理逻辑
    processed = {k: v.strip().lower() for k, v in data.items()}
    return processed

@trace(name="完整流程", run_type="chain")
def full_pipeline(user_input: str):
    data = {"input": user_input}
    data = preprocess_data(data)  # 创建子 Span
    
    result = chain.invoke(data)
    return result

# 执行时会创建嵌套 Trace
full_pipeline("  HELLO WORLD  ")
```

**Trace 结构：**
```
完整流程 (Chain)
├─ 数据预处理 (Tool)
│  └─ 输入: {"input": "  HELLO WORLD  "}
│  └─ 输出: {"input": "hello world"}
└─ Chain Execution
   ├─ PromptTemplate
   ├─ ChatOpenAI
   └─ StrOutputParser
```

### 16.3.3 Trace 元数据与标签

为 Trace 添加元数据和标签便于后续筛选和分析：

```python
from langsmith import Client

client = Client()

# 方式 1：通过配置添加元数据
chain.invoke(
    {"role": "助手", "input": "你好"},
    config={
        "metadata": {
            "user_id": "user_12345",
            "session_id": "sess_abc",
            "environment": "production"
        },
        "tags": ["customer-support", "greeting"]
    }
)

# 方式 2：通过装饰器添加
@trace(
    name="用户查询处理",
    metadata={"department": "sales"},
    tags=["high-priority"]
)
def handle_query(query: str):
    return chain.invoke({"role": "销售专家", "input": query})
```

在 LangSmith UI 中可以通过标签和元数据过滤：
- 查看特定用户的所有 Trace
- 分析生产环境 vs 测试环境的性能差异
- 统计高优先级查询的成功率

---

## 16.4 调试与 Prompt 优化

### 16.4.1 Playground：交互式提示调试

LangSmith Playground 允许你在浏览器中直接修改提示、参数，实时对比效果。

#### 使用流程

1. **选择 Trace**：在 Trace 列表中点击一个 LLM 调用
2. **打开 Playground**：点击 "Open in Playground"
3. **修改提示**：
   - 调整 System Message
   - 修改 Temperature、Top-p
   - 切换模型（GPT-4 → GPT-3.5 → Claude）
4. **对比测试**：
   - 并排对比多个版本
   - 查看 Token 消耗差异
5. **保存优化版本**：将优化后的提示保存到 Hub

#### 示例：优化翻译提示

**原始提示：**
```
Translate to French: Hello world
```

**优化后：**
```
You are a professional translator specializing in English to French translation.
Translate the following text to French, maintaining the tone and style:

Text: Hello world

Translation:
```

**对比结果：**
| 版本 | 输出 | Token | 质量评分 |
|------|------|-------|----------|
| 原始 | "Bonjour le monde" | 15 | 7/10 |
| 优化 | "Bonjour le monde" (with explanation) | 45 | 9/10 |

### 16.4.2 失败 Trace 自动标记

LangSmith 自动标记失败的 Trace（抛出异常或超时）：

```python
from langchain_core.runnables import RunnableLambda

def risky_operation(x: int) -> int:
    if x < 0:
        raise ValueError("不支持负数")
    return x * 2

chain = RunnableLambda(risky_operation)

try:
    chain.invoke(-5)
except ValueError as e:
    print(f"捕获异常: {e}")
```

在 LangSmith 中会看到：
- ❌ **Status: Error**
- **Error Type**: `ValueError`
- **Error Message**: "不支持负数"
- **Stack Trace**: 完整堆栈信息

### 16.4.3 成本与延迟分析

LangSmith 自动计算每次执行的成本和延迟：

```python
# 执行多次以收集统计数据
for i in range(10):
    chain.invoke({
        "role": "助手",
        "input": f"第 {i+1} 个问题"
    })
```

**分析视图：**
- **延迟分布图**：P50: 1.2s, P95: 2.8s, P99: 4.1s
- **成本统计**：
  - 平均每次调用: $0.0045
  - 总成本: $0.045
- **Token 分布**：
  - Prompt Tokens: 平均 32 (范围 28-35)
  - Completion Tokens: 平均 150 (范围 120-180)

---

## 16.5 数据集管理与离线评估

### 16.5.1 创建数据集

数据集是评估的基础，包含**输入**和**预期输出**（可选）。

#### 通过代码创建数据集

```python
from langsmith import Client

client = Client()

# 创建数据集
dataset_name = "customer-support-qa"
dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="客服常见问答数据集"
)

# 添加样本
examples = [
    {
        "inputs": {"question": "如何重置密码？"},
        "outputs": {"answer": "点击登录页面的'忘记密码'链接，按照邮件指引操作。"}
    },
    {
        "inputs": {"question": "支持哪些支付方式？"},
        "outputs": {"answer": "支持信用卡、PayPal、支付宝和微信支付。"}
    },
    {
        "inputs": {"question": "退货政策是什么？"},
        "outputs": {"answer": "30 天内无理由退货，需保持商品完好。"}
    }
]

for example in examples:
    client.create_example(
        dataset_id=dataset.id,
        inputs=example["inputs"],
        outputs=example["outputs"]
    )

print(f"✅ 数据集创建完成：{dataset_name}")
```

#### 从 Trace 创建数据集

在 LangSmith UI 中：
1. 选择高质量的 Trace
2. 点击 "Add to Dataset"
3. 选择目标数据集或创建新数据集

### 16.5.2 运行评估

#### 定义评估器

LangSmith 支持多种评估器：

**1. 精确匹配（Exact Match）**

```python
from langsmith.evaluation import EvaluationResult

def exact_match_evaluator(run, example):
    """检查输出是否与预期完全一致"""
    prediction = run.outputs.get("output", "")
    reference = example.outputs.get("answer", "")
    
    return EvaluationResult(
        key="exact_match",
        score=1.0 if prediction == reference else 0.0
    )
```

**2. LLM-as-Judge（使用 LLM 评估质量）**

```python
from langchain_openai import ChatOpenAI
from langsmith.evaluation import LangChainStringEvaluator

# 使用 GPT-4 评估答案质量
evaluator = LangChainStringEvaluator(
    "qa",  # QA 评估模式
    config={
        "llm": ChatOpenAI(model="gpt-4", temperature=0),
        "criteria": {
            "accuracy": "答案是否准确回答了问题？",
            "completeness": "答案是否完整？",
            "clarity": "答案是否清晰易懂？"
        }
    }
)
```

**3. 自定义评估器**

```python
def custom_evaluator(run, example):
    """自定义评估逻辑"""
    prediction = run.outputs.get("output", "")
    reference = example.outputs.get("answer", "")
    
    # 检查关键词是否出现
    keywords = ["密码", "重置", "邮件"]
    keyword_match = sum(kw in prediction for kw in keywords) / len(keywords)
    
    # 检查长度合理性
    length_ok = 20 < len(prediction) < 200
    
    return EvaluationResult(
        key="custom_quality",
        score=(keyword_match + (1.0 if length_ok else 0.0)) / 2,
        comment=f"关键词匹配: {keyword_match:.2f}, 长度合理: {length_ok}"
    )
```

#### 执行评估

```python
from langsmith.evaluation import evaluate

# 定义待评估的链
def predict(inputs: dict) -> dict:
    question = inputs["question"]
    result = chain.invoke({"role": "客服", "input": question})
    return {"output": result}

# 运行评估
results = evaluate(
    predict,
    data=dataset_name,
    evaluators=[
        exact_match_evaluator,
        custom_evaluator
    ],
    experiment_prefix="customer-support-v1",
    metadata={
        "model": "gpt-4",
        "version": "1.0.0"
    }
)

print(results)
```

**评估报告：**
```
📊 评估结果：customer-support-v1-20240115-123045

总样本数: 3
平均分数:
  - exact_match: 0.33 (1/3 完全匹配)
  - custom_quality: 0.78 (质量良好)

详细结果:
1. ✅ 如何重置密码？
   - exact_match: 0.0
   - custom_quality: 0.85
   - 评论: 关键词匹配良好，但表述与标准答案不同

2. ✅ 支持哪些支付方式？
   - exact_match: 1.0
   - custom_quality: 1.0
   
3. ❌ 退货政策是什么？
   - exact_match: 0.0
   - custom_quality: 0.50
   - 评论: 缺少关键信息"保持商品完好"
```

<div data-component="EvaluationDashboard"></div>

### 16.5.3 对比实验（A/B Testing）

比较不同提示、模型或检索策略的效果：

```python
# 实验 A：GPT-3.5 + 简单提示
chain_a = ChatPromptTemplate.from_template("回答：{question}") | \
          ChatOpenAI(model="gpt-3.5-turbo") | \
          StrOutputParser()

# 实验 B：GPT-4 + 详细提示
chain_b = ChatPromptTemplate.from_template(
    "你是专业客服，请用友好、准确的语气回答：{question}"
) | ChatOpenAI(model="gpt-4") | StrOutputParser()

# 分别评估
results_a = evaluate(
    lambda x: {"output": chain_a.invoke(x)},
    data=dataset_name,
    evaluators=[custom_evaluator],
    experiment_prefix="experiment-A-gpt35"
)

results_b = evaluate(
    lambda x: {"output": chain_b.invoke(x)},
    data=dataset_name,
    evaluators=[custom_evaluator],
    experiment_prefix="experiment-B-gpt4"
)
```

**对比结果：**

| 指标 | Experiment A (GPT-3.5) | Experiment B (GPT-4) | 改进 |
|------|------------------------|----------------------|------|
| **平均质量分** | 0.65 | 0.82 | +26% |
| **平均延迟** | 1.2s | 2.8s | +133% |
| **平均成本** | $0.0008 | $0.0045 | +463% |
| **成功率** | 66% | 100% | +34% |

**结论**：GPT-4 质量显著更高，但成本和延迟也明显增加。可考虑混合策略：简单问题用 GPT-3.5，复杂问题用 GPT-4。

---

## 16.6 生产监控与反馈循环

### 16.6.1 实时监控仪表板

LangSmith 提供实时监控面板，展示关键指标：

#### 核心监控指标

1. **吞吐量（Throughput）**
   - 每分钟请求数（RPM）
   - 每小时 Token 消耗

2. **延迟（Latency）**
   - P50、P95、P99 延迟
   - LLM 调用延迟 vs 总延迟

3. **错误率（Error Rate）**
   - 按错误类型分类（Timeout, RateLimitError, ValidationError）
   - 错误 Trace 占比

4. **成本（Cost）**
   - 每日成本趋势
   - 按模型/用户/功能分组的成本

5. **质量指标（Quality Metrics）**
   - 用户反馈评分
   - 自动评估器分数

#### 设置警报

```python
# 通过 LangSmith API 设置警报（伪代码，实际需在 UI 配置）
alert_config = {
    "name": "高错误率警报",
    "condition": "error_rate > 0.05",  # 错误率超过 5%
    "window": "5m",  # 5 分钟窗口
    "actions": [
        {"type": "email", "recipients": ["team@example.com"]},
        {"type": "slack", "webhook": "https://hooks.slack.com/..."}
    ]
}
```

### 16.6.2 用户反馈收集

#### 在线收集反馈

```python
from langsmith import Client

client = Client()

# 执行链
result = chain.invoke({"role": "助手", "input": "你好"})

# 假设用户给出反馈（在 UI 中收集）
run_id = "run_abc123"  # 从 Trace 中获取

# 记录用户反馈
client.create_feedback(
    run_id=run_id,
    key="user_rating",
    score=0.8,  # 0-1 之间
    comment="回答准确但略显冗长"
)

client.create_feedback(
    run_id=run_id,
    key="user_thumbs",
    score=1.0,  # 1 = 👍, 0 = 👎
)
```

#### 反馈驱动的迭代

1. **分析低分 Trace**：找出用户评分低的共性问题
2. **创建改进数据集**：将低分样本加入数据集
3. **调整提示/模型**：针对性优化
4. **A/B 测试验证**：对比新旧版本
5. **灰度发布**：逐步推广优化版本

### 16.6.3 持续评估（Online Evaluation）

在生产环境中持续运行评估器：

```python
from langsmith.evaluation import EvaluationResult

def online_evaluator(run):
    """生产环境实时评估"""
    output = run.outputs.get("output", "")
    
    # 检查输出长度
    length_ok = 50 < len(output) < 500
    
    # 检查是否包含不当内容（简化示例）
    inappropriate = any(word in output for word in ["脏话", "侮辱"])
    
    return EvaluationResult(
        key="production_quality",
        score=1.0 if (length_ok and not inappropriate) else 0.0,
        comment=f"长度: {len(output)}, 内容合规: {not inappropriate}"
    )

# 配置在线评估（自动应用到所有新 Trace）
# 实际需在 LangSmith UI 中配置
```

<div data-component="MonitoringDashboard"></div>

---

## 16.7 高级特性与最佳实践

### 16.7.1 自定义 Run 收集器

对于非 LangChain 应用，可手动发送 Trace：

```python
from langsmith import Client
from datetime import datetime

client = Client()

# 手动创建 Run
run_id = client.create_run(
    name="自定义推荐系统",
    run_type="chain",
    inputs={"user_id": "user_123", "context": "浏览历史"},
    start_time=datetime.now()
)

try:
    # 执行业务逻辑
    recommendations = my_custom_algorithm()
    
    # 记录成功
    client.update_run(
        run_id=run_id,
        outputs={"recommendations": recommendations},
        end_time=datetime.now()
    )
except Exception as e:
    # 记录失败
    client.update_run(
        run_id=run_id,
        error=str(e),
        end_time=datetime.now()
    )
```

### 16.7.2 批量导出 Trace 数据

用于离线分析或数据科学工作流：

```python
from langsmith import Client
import pandas as pd

client = Client()

# 查询指定时间范围的 Trace
runs = client.list_runs(
    project_name="production-app",
    start_time="2024-01-01",
    end_time="2024-01-31",
    filter='eq(status, "success")'  # 只导出成功的
)

# 转换为 DataFrame
data = []
for run in runs:
    data.append({
        "run_id": run.id,
        "name": run.name,
        "latency": run.latency,
        "total_tokens": run.total_tokens,
        "cost": run.prompt_tokens * 0.00003 + run.completion_tokens * 0.00006,
        "created_at": run.start_time
    })

df = pd.DataFrame(data)
print(df.describe())
```

### 16.7.3 Privacy & Compliance

#### 脱敏处理

```python
import re
from langsmith.run_helpers import traceable

def redact_pii(text: str) -> str:
    """移除个人身份信息"""
    # 移除邮箱
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    # 移除电话号码
    text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)
    return text

@traceable(
    name="脱敏查询处理",
    process_inputs=lambda x: {"query": redact_pii(x["query"])},
    process_outputs=lambda x: {"result": redact_pii(x["result"])}
)
def handle_query(query: str):
    result = chain.invoke({"input": query})
    return {"result": result}
```

#### 数据保留策略

在 LangSmith 设置中配置：
- **自动删除**：30 天后删除 Trace
- **采样策略**：只保留 10% 的成功 Trace，保留 100% 失败 Trace
- **地域限制**：确保数据存储在合规地区（EU/US）

---

## 16.8 实战案例：优化客服 Agent

### 16.8.1 初始版本

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("作为客服回答：{question}")
chain = prompt | ChatOpenAI(model="gpt-3.5-turbo") | StrOutputParser()
```

**初始指标（运行 1 周）：**
- 平均延迟：1.5s
- 用户满意度：65%
- 成本/查询：$0.0012

### 16.8.2 问题诊断

通过 LangSmith 发现：
1. **20% 的查询超时**（> 5s）
2. **低分 Trace 共性**：回答过于简短或不相关
3. **高成本 Trace**：复杂查询重复调用 LLM

### 16.8.3 优化措施

#### 优化 1：改进提示

```python
improved_prompt = ChatPromptTemplate.from_template("""
你是专业客服代表，遵循以下原则：
1. 友好、耐心、专业
2. 提供具体、可操作的建议
3. 如果不确定，诚实告知并建议联系人工客服

用户问题：{question}

回答：
""")
```

#### 优化 2：添加缓存

```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())  # 相同问题直接返回缓存
```

#### 优化 3：添加 Fallback

```python
primary_chain = improved_prompt | ChatOpenAI(model="gpt-4", timeout=3)
fallback_chain = improved_prompt | ChatOpenAI(model="gpt-3.5-turbo", timeout=5)

robust_chain = primary_chain.with_fallbacks([fallback_chain])
```

### 16.8.4 A/B 测试验证

```python
# 对比新旧版本
results_old = evaluate(
    lambda x: {"output": chain.invoke(x)},
    data="customer-qa-v1",
    experiment_prefix="baseline"
)

results_new = evaluate(
    lambda x: {"output": robust_chain.invoke(x)},
    data="customer-qa-v1",
    experiment_prefix="optimized"
)
```

**结果对比：**

| 指标 | 基线版本 | 优化版本 | 改进 |
|------|----------|----------|------|
| 用户满意度 | 65% | 85% | +31% |
| 平均延迟 | 1.5s | 1.2s | -20% |
| 超时率 | 20% | 2% | -90% |
| 成本/查询 | $0.0012 | $0.0018 | +50% |

**决策**：虽然成本增加，但满意度和稳定性大幅提升，决定全量上线。

---

## 16.9 常见问题与陷阱

### 16.9.1 Trace 数据量过大

**问题**：高流量应用每天产生数百万 Trace，成本和存储压力大。

**解决方案：**

1. **采样策略**
   ```python
   import random
   
   # 只追踪 10% 的请求
   if random.random() < 0.1:
       os.environ["LANGCHAIN_TRACING_V2"] = "true"
   else:
       os.environ["LANGCHAIN_TRACING_V2"] = "false"
   
   chain.invoke(...)
   ```

2. **按条件追踪**
   ```python
   # 只追踪失败或高价值用户
   should_trace = (user.is_premium or has_error)
   
   with trace(enabled=should_trace):
       chain.invoke(...)
   ```

### 16.9.2 评估器与实际用户体验不一致

**问题**：LLM-as-Judge 评分高，但用户反馈差。

**解决方案：**
- 结合**真实用户反馈**校准评估器权重
- 使用**多样化评估器**（语法、事实、用户意图）
- 定期**人工抽查**评估结果

### 16.9.3 隐私与合规问题

**问题**：Trace 包含敏感信息（PII），违反 GDPR。

**解决方案：**
- **脱敏处理**：在发送前移除 PII
- **本地部署**：考虑自托管 LangFuse 等开源方案
- **数据保留策略**：自动删除旧 Trace

---

## 16.10 扩展阅读与资源

### 官方文档
- [LangSmith 文档](https://docs.smith.langchain.com/)
- [LangSmith Python SDK](https://github.com/langchain-ai/langsmith-sdk)
- [LangSmith Cookbook](https://github.com/langchain-ai/langsmith-cookbook)

### 最佳实践指南
- [Tracing Best Practices](https://docs.smith.langchain.com/tracing/best-practices)
- [Evaluation Strategies](https://docs.smith.langchain.com/evaluation/strategies)
- [Production Monitoring](https://docs.smith.langchain.com/monitoring)

### 视频教程
- [LangSmith 快速上手](https://www.youtube.com/watch?v=xxx) (官方)
- [生产级 LLM 应用监控](https://www.youtube.com/watch?v=yyy)

---

## 本章小结

本章深入探讨了 LangSmith 的核心功能：

✅ **Tracing**：自动记录每一步执行过程，支持嵌套 Span 和自定义元数据  
✅ **Debugging**：Playground 交互式调试，快速定位失败 Trace  
✅ **Evaluation**：数据集管理 + 多样化评估器 + A/B 测试  
✅ **Monitoring**：实时监控延迟、成本、错误率，设置警报  
✅ **Feedback Loop**：收集用户反馈，持续优化提示和模型  

**关键要点：**
1. LangSmith 是 LangChain 生态的可观测性基石，生产必备
2. Trace 提供完整的执行链路透明度，便于调试和优化
3. 数据集 + 评估器实现系统化质量控制
4. 在线监控 + 用户反馈形成持续改进闭环
5. 注意隐私合规，合理采样控制成本

下一章将学习 **LangServe**，将优化后的链部署为生产级 API 服务。

---

**思考题：**
1. 如何设计一个评估器来衡量聊天机器人的"友好度"？
2. 在什么场景下应该使用 GPT-4 vs GPT-3.5？如何通过 LangSmith 数据支持决策？
3. 如果 Trace 显示 80% 的延迟来自向量检索，你会采取哪些优化措施？

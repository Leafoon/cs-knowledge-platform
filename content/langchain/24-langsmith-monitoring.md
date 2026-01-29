# Chapter 24: LangSmith 生产监控

## 本章概览

在完成了 Tracing（追踪）和 Evaluation（评估）后，生产环境还需要**持续监控**。LangSmith 的监控系统提供实时仪表盘、智能告警、在线 Playground、运行结果标注与成本分析，帮助你在生产环境中保持应用的稳定性、性能与成本可控性。本章将学习如何利用 LangSmith 构建完整的生产级可观测性体系。

**本章重点**：
- 监控面板（Dashboard）：实时指标与趋势分析
- 告警（Alerts）：自动化问题检测与通知
- Playground：在线提示调试与对比
- Annotation & Curation：运行结果标注与数据集构建
- 成本分析：Token 消耗追踪与优化

---

## 24.1 监控面板（Monitoring Dashboard）

### 24.1.1 实时请求量监控

<div data-component="MonitoringDashboard"></div>

**访问监控面板**：

1. 登录 [https://smith.langchain.com](https://smith.langchain.com)
2. 进入项目 → **Monitoring** 标签
3. 查看实时指标

**关键指标**：

| 指标 | 说明 | 目标值 |
|------|------|--------|
| **Requests/min** | 每分钟请求数 | 取决于流量 |
| **Success Rate** | 成功率 | > 99% |
| **Error Rate** | 错误率 | < 1% |
| **Avg Latency** | 平均延迟 | < 2s |
| **P95 Latency** | 95% 请求延迟 | < 5s |
| **P99 Latency** | 99% 请求延迟 | < 10s |

**代码中查询指标**：

```python
from langsmith import Client
from datetime import datetime, timedelta

client = Client()

# 查询最近 1 小时的 Runs
end_time = datetime.now()
start_time = end_time - timedelta(hours=1)

runs = client.list_runs(
    project_name="production-chatbot",
    start_time=start_time,
    end_time=end_time
)

# 计算指标
total_runs = 0
successful_runs = 0
failed_runs = 0
latencies = []

for run in runs:
    total_runs += 1
    if run.status == "success":
        successful_runs += 1
    else:
        failed_runs += 1
    
    if run.end_time and run.start_time:
        latency_ms = (run.end_time - run.start_time).total_seconds() * 1000
        latencies.append(latency_ms)

# 统计
success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
error_rate = (failed_runs / total_runs * 100) if total_runs > 0 else 0
avg_latency = sum(latencies) / len(latencies) if latencies else 0

# P95、P99 延迟
latencies_sorted = sorted(latencies)
p95_index = int(len(latencies_sorted) * 0.95)
p99_index = int(len(latencies_sorted) * 0.99)
p95_latency = latencies_sorted[p95_index] if p95_index < len(latencies_sorted) else 0
p99_latency = latencies_sorted[p99_index] if p99_index < len(latencies_sorted) else 0

print(f"📊 监控报告（最近 1 小时）")
print(f"总请求数: {total_runs}")
print(f"成功率: {success_rate:.2f}%")
print(f"错误率: {error_rate:.2f}%")
print(f"平均延迟: {avg_latency:.0f}ms")
print(f"P95 延迟: {p95_latency:.0f}ms")
print(f"P99 延迟: {p99_latency:.0f}ms")
```

### 24.1.2 延迟分布（P50、P95、P99）

**为什么使用百分位数？**

```python
# 示例：10 个请求的延迟（ms）
latencies = [100, 110, 120, 95, 105, 130, 140, 3000, 115, 125]

# 平均值会被极端值拉高
avg = sum(latencies) / len(latencies)  # 1044ms

# P95 更能反映大多数用户体验
sorted_latencies = sorted(latencies)
p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]  # 140ms

print(f"平均延迟: {avg}ms (被 3000ms 拉高)")
print(f"P95 延迟: {p95}ms (95% 用户的体验)")
```

**可视化延迟分布**：

```python
import matplotlib.pyplot as plt
import numpy as np

# 生成延迟分布直方图
plt.figure(figsize=(10, 6))
plt.hist(latencies, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(avg_latency, color='red', linestyle='--', label=f'Average: {avg_latency:.0f}ms')
plt.axvline(p95_latency, color='orange', linestyle='--', label=f'P95: {p95_latency:.0f}ms')
plt.axvline(p99_latency, color='purple', linestyle='--', label=f'P99: {p99_latency:.0f}ms')
plt.xlabel('Latency (ms)')
plt.ylabel('Frequency')
plt.title('Latency Distribution')
plt.legend()
plt.show()
```

### 24.1.3 错误率追踪

**错误分类**：

```python
from collections import defaultdict

# 按错误类型分类
error_types = defaultdict(int)

for run in runs:
    if run.error:
        # 提取错误类型
        error_message = str(run.error)
        if "RateLimitError" in error_message:
            error_types["Rate Limit"] += 1
        elif "Timeout" in error_message:
            error_types["Timeout"] += 1
        elif "AuthenticationError" in error_message:
            error_types["Authentication"] += 1
        elif "ValidationError" in error_message:
            error_types["Validation"] += 1
        else:
            error_types["Other"] += 1

# 打印错误分布
print("\n🚨 错误类型分布")
for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / failed_runs * 100) if failed_runs > 0 else 0
    print(f"{error_type:20} {count:5} ({percentage:5.1f}%)")
```

**错误趋势分析**：

```python
from datetime import datetime, timedelta
import pandas as pd

# 按小时分组统计错误
hourly_errors = {}

for run in runs:
    hour_key = run.start_time.replace(minute=0, second=0, microsecond=0)
    if hour_key not in hourly_errors:
        hourly_errors[hour_key] = {"total": 0, "errors": 0}
    
    hourly_errors[hour_key]["total"] += 1
    if run.status != "success":
        hourly_errors[hour_key]["errors"] += 1

# 转为 DataFrame
df = pd.DataFrame([
    {
        "hour": hour,
        "error_rate": (data["errors"] / data["total"] * 100) if data["total"] > 0 else 0
    }
    for hour, data in sorted(hourly_errors.items())
])

print("\n📈 错误率趋势（按小时）")
print(df)
```

### 24.1.4 Token 消耗趋势

```python
# 统计 Token 消耗
total_prompt_tokens = 0
total_completion_tokens = 0
total_cost = 0

# GPT-4 价格（示例）
PRICE_PER_1K_PROMPT = 0.03
PRICE_PER_1K_COMPLETION = 0.06

for run in runs:
    if run.outputs and "token_usage" in run.outputs:
        usage = run.outputs["token_usage"]
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        
        # 计算成本
        total_cost += (prompt_tokens / 1000 * PRICE_PER_1K_PROMPT)
        total_cost += (completion_tokens / 1000 * PRICE_PER_1K_COMPLETION)

print(f"\n💰 Token 消耗统计")
print(f"Prompt Tokens: {total_prompt_tokens:,}")
print(f"Completion Tokens: {total_completion_tokens:,}")
print(f"Total Tokens: {total_prompt_tokens + total_completion_tokens:,}")
print(f"估算成本: ${total_cost:.4f}")
```

---

## 24.2 告警（Alerts）

### 24.2.1 告警规则配置

<div data-component="AlertRuleBuilder"></div>

**在 LangSmith UI 中配置告警**：

1. 进入项目 → **Settings** → **Alerts**
2. 点击 **Create Alert**
3. 配置规则：
   - **Metric**: 选择指标（Error Rate、Latency、Token Usage）
   - **Condition**: 设置条件（> 阈值）
   - **Threshold**: 阈值（如 5%、2000ms）
   - **Duration**: 持续时间（如 5 分钟）
   - **Notifications**: 通知渠道

**示例：高错误率告警**

```yaml
Alert Name: High Error Rate
Metric: Error Rate
Condition: Greater than
Threshold: 5%
Duration: 5 minutes
Notifications:
  - Email: team@example.com
  - Slack: #production-alerts
```

### 24.2.2 阈值告警（延迟、错误率）

**编程方式配置告警**：

```python
from langsmith import Client

client = Client()

# 创建告警规则（伪代码，实际需要 UI 配置）
alert_config = {
    "name": "High Latency Alert",
    "metric": "p95_latency",
    "condition": "greater_than",
    "threshold": 3000,  # 3 秒
    "window": 300,  # 5 分钟
    "notifications": [
        {"type": "email", "to": "team@example.com"},
        {"type": "slack", "webhook": "https://hooks.slack.com/..."}
    ]
}

# 注意：LangSmith SDK 可能不支持直接创建告警，通常通过 UI 配置
```

**自定义告警逻辑**：

```python
import time
from datetime import datetime, timedelta

def monitor_and_alert(project_name: str, check_interval: int = 60):
    """自定义监控与告警"""
    client = Client()
    
    while True:
        # 查询最近 5 分钟的数据
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=5)
        
        runs = list(client.list_runs(
            project_name=project_name,
            start_time=start_time,
            end_time=end_time
        ))
        
        if not runs:
            time.sleep(check_interval)
            continue
        
        # 计算错误率
        failed = sum(1 for r in runs if r.status != "success")
        error_rate = (failed / len(runs)) * 100
        
        # 检查告警条件
        if error_rate > 5:
            send_alert(
                title="🚨 High Error Rate Detected",
                message=f"Error rate: {error_rate:.1f}% (threshold: 5%)",
                severity="high"
            )
        
        # 计算 P95 延迟
        latencies = [
            (r.end_time - r.start_time).total_seconds() * 1000
            for r in runs if r.end_time and r.start_time
        ]
        if latencies:
            latencies_sorted = sorted(latencies)
            p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
            
            if p95 > 3000:
                send_alert(
                    title="⏱️ High Latency Detected",
                    message=f"P95 latency: {p95:.0f}ms (threshold: 3000ms)",
                    severity="medium"
                )
        
        time.sleep(check_interval)

def send_alert(title: str, message: str, severity: str):
    """发送告警（集成 Slack、邮件等）"""
    print(f"\n{'='*60}")
    print(f"[{severity.upper()}] {title}")
    print(message)
    print(f"{'='*60}\n")
    
    # 实际应用：发送到 Slack / Email / PagerDuty
    # slack_webhook("https://hooks.slack.com/...", message)
    # send_email("team@example.com", title, message)
```

### 24.2.3 异常检测告警

**基于统计的异常检测**：

```python
import numpy as np

def detect_anomalies(metric_values: list, threshold_std: float = 3) -> list:
    """使用 3-sigma 规则检测异常"""
    mean = np.mean(metric_values)
    std = np.std(metric_values)
    
    anomalies = []
    for i, value in enumerate(metric_values):
        z_score = abs(value - mean) / std if std > 0 else 0
        if z_score > threshold_std:
            anomalies.append({
                "index": i,
                "value": value,
                "z_score": z_score,
                "mean": mean,
                "std": std
            })
    
    return anomalies

# 使用示例
latencies = [100, 110, 95, 105, 3000, 120, 115, 130]  # 3000 是异常值
anomalies = detect_anomalies(latencies)

for anomaly in anomalies:
    print(f"🔴 异常检测: 值 {anomaly['value']} 偏离均值 {anomaly['z_score']:.1f} 个标准差")
```

### 24.2.4 通知渠道（邮件、Slack、Webhook）

**Slack 通知**：

```python
import requests

def send_slack_alert(webhook_url: str, message: str):
    """发送 Slack 通知"""
    payload = {
        "text": message,
        "attachments": [
            {
                "color": "danger",
                "fields": [
                    {"title": "Project", "value": "production-chatbot", "short": True},
                    {"title": "Time", "value": datetime.now().isoformat(), "short": True}
                ]
            }
        ]
    }
    
    response = requests.post(webhook_url, json=payload)
    if response.status_code == 200:
        print("✅ Slack 通知已发送")
    else:
        print(f"❌ Slack 通知失败: {response.status_code}")

# 使用
slack_webhook = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
send_slack_alert(slack_webhook, "🚨 错误率超过 5%！")
```

**邮件通知**：

```python
import smtplib
from email.mime.text import MIMEText

def send_email_alert(to_email: str, subject: str, body: str):
    """发送邮件通知"""
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = 'alerts@example.com'
    msg['To'] = to_email
    
    # SMTP 配置
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    smtp_user = 'your-email@gmail.com'
    smtp_password = 'your-password'
    
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
    
    print("✅ 邮件通知已发送")
```

---

## 24.3 Playground

### 24.3.1 Prompt 在线编辑与测试

**Playground 功能**：

1. **在线编辑 Prompt**：无需修改代码即可测试不同提示
2. **即时运行**：查看输出、Token 消耗、延迟
3. **版本对比**：并排对比多个提示版本
4. **保存到 Hub**：优秀提示一键分享

**访问 Playground**：

1. 在 LangSmith UI 中选择一个 Run
2. 点击 **Open in Playground**
3. 编辑 Prompt 或参数
4. 点击 **Run** 测试

**示例：优化翻译提示**

```
# 原始 Prompt（在 Playground 中编辑）
Translate to French: {text}

# 优化后 Prompt
You are a professional French translator with expertise in cultural nuances.

Translate the following text to French while:
- Preserving the original tone and style
- Using appropriate French idioms when applicable
- Maintaining grammatical accuracy

Text: {text}

# 在 Playground 中对比两个版本
```

### 24.3.2 模型参数调优

**可调参数**：

```python
# 在 Playground 中调整这些参数
{
    "temperature": 0.7,       # 创造性：0-2（0=确定，2=随机）
    "max_tokens": 150,        # 最大输出长度
    "top_p": 0.9,            # 核采样阈值
    "frequency_penalty": 0,  # 重复词惩罚：-2 到 2
    "presence_penalty": 0,   # 新话题鼓励：-2 到 2
}
```

**参数效果对比**：

| 参数 | 值 | 效果 |
|------|-----|------|
| temperature | 0.0 | 确定性强，适合翻译、摘要 |
| temperature | 1.0 | 平衡创造性与连贯性 |
| temperature | 2.0 | 高度创造性，适合头脑风暴 |
| top_p | 0.5 | 保守，选择高概率词 |
| top_p | 0.95 | 宽松，允许多样性 |

### 24.3.3 对比不同配置

**在 Playground 中对比**：

```
配置 A:
- Model: gpt-4
- Temperature: 0.7
- Prompt: "Translate to French: {text}"

配置 B:
- Model: gpt-3.5-turbo
- Temperature: 0.3
- Prompt: "Professional French translation: {text}"

输入: "The weather is nice today."

结果对比:
配置 A: "Le temps est agréable aujourd'hui."
配置 B: "Il fait beau aujourd'hui."

Token 消耗:
配置 A: 45 tokens, $0.0027
配置 B: 35 tokens, $0.0007

选择: 配置 B（更便宜，质量相近）
```

### 24.3.4 保存为 Hub Prompt

```python
from langchain import hub

# 在 Playground 中测试并优化后，保存到 Hub
prompt = hub.pull("your-username/optimized-translation-prompt")

# 团队其他成员可以直接使用
# prompt = hub.pull("your-username/optimized-translation-prompt")
```

---

## 24.4 Annotation & Curation

### 24.4.1 运行结果标注

**为什么需要标注？**

生产环境中的运行结果是宝贵的训练数据，通过标注可以：
- 构建高质量评估数据集
- 发现边界情况与失败模式
- 持续改进模型与提示

**在 UI 中标注**：

1. 选择一个 Run
2. 点击 **Add to Dataset**
3. 选择目标数据集
4. 可选：修正输出（如果 AI 输出有误）
5. 保存

**编程方式标注**：

```python
from langsmith import Client

client = Client()

# 查询需要标注的 Runs（例如：用户给了好评的）
high_quality_runs = client.list_runs(
    project_name="production-chatbot",
    filter='feedback.user_rating.score = 1'  # 好评
)

# 添加到数据集
dataset = client.read_dataset(dataset_name="golden-responses")

for run in list(high_quality_runs)[:50]:  # 取前 50 个
    client.create_example(
        dataset_id=dataset.id,
        inputs=run.inputs,
        outputs=run.outputs,
        metadata={
            "source": "production",
            "user_rating": "positive",
            "run_id": str(run.id)
        }
    )

print(f"✅ 已添加 50 个高质量样本到数据集")
```

### 24.4.2 构建黄金数据集

**黄金数据集特征**：

- ✅ **高质量**：经过人工审核或用户好评
- ✅ **代表性**：覆盖真实场景
- ✅ **多样性**：不同类型输入
- ✅ **可维护**：定期更新

**策略 1：从用户反馈筛选**

```python
def build_golden_dataset_from_feedback(min_score: float = 0.8):
    """从用户反馈中构建黄金数据集"""
    client = Client()
    
    # 创建黄金数据集
    golden_dataset = client.create_dataset(
        dataset_name=f"golden-set-{datetime.now().strftime('%Y%m%d')}",
        description="从生产环境高分样本中构建"
    )
    
    # 查询高分 Runs
    high_rated = client.list_runs(
        project_name="production-chatbot",
        filter=f'feedback.user_rating.score >= {min_score}'
    )
    
    # 添加到数据集
    added = 0
    for run in high_rated:
        if added >= 100:  # 限制数量
            break
        
        client.create_example(
            dataset_id=golden_dataset.id,
            inputs=run.inputs,
            outputs=run.outputs
        )
        added += 1
    
    print(f"✅ 黄金数据集已创建，包含 {added} 个样本")
    return golden_dataset
```

**策略 2：主动学习（Active Learning）**

```python
def active_learning_curation(uncertainty_threshold: float = 0.6):
    """选择模型不确定的样本进行人工标注"""
    client = Client()
    
    # 查询所有 Runs
    runs = client.list_runs(project_name="production-chatbot")
    
    uncertain_runs = []
    for run in runs:
        # 假设输出中有 confidence 分数
        if run.outputs and "confidence" in run.outputs:
            confidence = run.outputs["confidence"]
            if confidence < uncertainty_threshold:
                uncertain_runs.append(run)
    
    print(f"🔍 发现 {len(uncertain_runs)} 个不确定样本，建议人工审核")
    
    # 导出待标注样本
    for run in uncertain_runs[:10]:  # 展示前 10 个
        print(f"\nRun ID: {run.id}")
        print(f"Input: {run.inputs}")
        print(f"Output: {run.outputs}")
        print(f"Confidence: {run.outputs.get('confidence')}")
        print("请人工审核并标注 ↑")
```

### 24.4.3 持续改进工作流

**完整闭环**：

```
1. 生产运行 → 收集数据
   ↓
2. 用户反馈 → 筛选高质量样本
   ↓
3. 构建数据集 → 定期评估
   ↓
4. 发现问题 → 优化提示/模型
   ↓
5. A/B 测试 → 部署改进版本
   ↓
回到第 1 步
```

**自动化脚本**：

```python
import schedule
import time

def weekly_improvement_workflow():
    """每周自动改进流程"""
    print("🔄 开始每周改进流程...")
    
    # 1. 构建黄金数据集
    golden_dataset = build_golden_dataset_from_feedback(min_score=0.8)
    
    # 2. 评估当前版本
    from langsmith.evaluation import evaluate
    
    current_results = evaluate(
        current_chain,
        data=golden_dataset.name,
        evaluators=[...],
        experiment_prefix="weekly-baseline"
    )
    
    # 3. 评估实验版本
    experimental_results = evaluate(
        experimental_chain,
        data=golden_dataset.name,
        evaluators=[...],
        experiment_prefix="weekly-experiment"
    )
    
    # 4. 决策
    if experimental_results['avg_score'] > current_results['avg_score'] * 1.03:
        print("✅ 实验版本提升 3%+，建议部署")
        # 自动部署或发送通知给团队决策
    else:
        print("⏸️ 改进不显著，保持当前版本")

# 每周一凌晨 2 点执行
schedule.every().monday.at("02:00").do(weekly_improvement_workflow)

while True:
    schedule.run_pending()
    time.sleep(3600)
```

---

## 24.5 成本分析

### 24.5.1 Token 消耗成本计算

<div data-component="CostAnalysisDashboard"></div>

**价格表（2024 年 1 月）**：

| 模型 | Prompt ($/1K tokens) | Completion ($/1K tokens) |
|------|---------------------|--------------------------|
| GPT-4 | $0.03 | $0.06 |
| GPT-4 Turbo | $0.01 | $0.03 |
| GPT-3.5 Turbo | $0.0005 | $0.0015 |
| Claude 3 Opus | $0.015 | $0.075 |
| Claude 3 Sonnet | $0.003 | $0.015 |

**计算成本**：

```python
from langsmith import Client
from datetime import datetime, timedelta

client = Client()

# 价格配置
PRICING = {
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
}

def calculate_cost(project_name: str, days: int = 7):
    """计算指定天数的成本"""
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    runs = client.list_runs(
        project_name=project_name,
        start_time=start_time,
        end_time=end_time
    )
    
    total_cost = 0
    model_costs = {}
    
    for run in runs:
        if not run.outputs or "token_usage" not in run.outputs:
            continue
        
        usage = run.outputs["token_usage"]
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        # 识别模型
        model_name = run.extra.get("invocation_params", {}).get("model", "gpt-3.5-turbo")
        
        # 查找价格
        pricing = PRICING.get(model_name, PRICING["gpt-3.5-turbo"])
        
        # 计算成本
        cost = (prompt_tokens / 1000 * pricing["prompt"]) + \
               (completion_tokens / 1000 * pricing["completion"])
        
        total_cost += cost
        model_costs[model_name] = model_costs.get(model_name, 0) + cost
    
    # 报告
    print(f"\n💰 成本分析报告（最近 {days} 天）")
    print(f"{'='*60}")
    print(f"总成本: ${total_cost:.4f}")
    print(f"\n按模型分解:")
    for model, cost in sorted(model_costs.items(), key=lambda x: x[1], reverse=True):
        percentage = (cost / total_cost * 100) if total_cost > 0 else 0
        print(f"  {model:20} ${cost:8.4f} ({percentage:5.1f}%)")
    
    return total_cost, model_costs

# 使用
total_cost, model_costs = calculate_cost("production-chatbot", days=7)
```

### 24.5.2 模型调用成本拆分

**按功能拆分**：

```python
def analyze_cost_by_function(project_name: str):
    """按功能拆分成本"""
    client = Client()
    
    runs = client.list_runs(project_name=project_name)
    
    function_costs = {}
    
    for run in runs:
        # 从 metadata 或 tags 中识别功能
        function_name = run.tags[0] if run.tags else "unknown"
        
        if run.outputs and "token_usage" in run.outputs:
            usage = run.outputs["token_usage"]
            cost = calculate_run_cost(usage)
            function_costs[function_name] = function_costs.get(function_name, 0) + cost
    
    # 排序并打印
    print("\n📊 按功能拆分成本")
    for func, cost in sorted(function_costs.items(), key=lambda x: x[1], reverse=True):
        print(f"{func:30} ${cost:.4f}")

def calculate_run_cost(usage: dict) -> float:
    """计算单个 Run 的成本"""
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    return (prompt_tokens / 1000 * 0.03) + (completion_tokens / 1000 * 0.06)
```

### 24.5.3 优化建议生成

**自动生成优化建议**：

```python
def generate_optimization_recommendations(project_name: str):
    """分析成本并生成优化建议"""
    client = Client()
    
    runs = list(client.list_runs(project_name=project_name))
    
    recommendations = []
    
    # 分析 1：是否过度使用 GPT-4
    gpt4_usage = sum(1 for r in runs if "gpt-4" in str(r.extra.get("invocation_params", {}).get("model", "")))
    gpt4_ratio = gpt4_usage / len(runs) if runs else 0
    
    if gpt4_ratio > 0.5:
        recommendations.append({
            "priority": "HIGH",
            "issue": f"GPT-4 使用率 {gpt4_ratio*100:.1f}%",
            "suggestion": "考虑对简单任务降级使用 GPT-3.5 Turbo",
            "potential_savings": "70-90%"
        })
    
    # 分析 2：Prompt 是否过长
    avg_prompt_tokens = sum(
        r.outputs.get("token_usage", {}).get("prompt_tokens", 0)
        for r in runs if r.outputs and "token_usage" in r.outputs
    ) / len(runs) if runs else 0
    
    if avg_prompt_tokens > 1000:
        recommendations.append({
            "priority": "MEDIUM",
            "issue": f"平均 Prompt 长度 {avg_prompt_tokens:.0f} tokens",
            "suggestion": "优化 Prompt 模板，移除冗余指令",
            "potential_savings": "20-40%"
        })
    
    # 分析 3：缓存命中率低
    # （需要集成缓存监控数据）
    
    # 打印建议
    print("\n🔧 成本优化建议")
    print("="*60)
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. [{rec['priority']}] {rec['issue']}")
        print(f"   💡 建议: {rec['suggestion']}")
        print(f"   💰 潜在节省: {rec['potential_savings']}")
    
    return recommendations

# 使用
recommendations = generate_optimization_recommendations("production-chatbot")
```

---

## 24.6 最佳实践

### 24.6.1 监控指标选择

**必须监控的指标**：

| 指标类别 | 具体指标 | 目标值 | 告警阈值 |
|---------|---------|--------|---------|
| **可用性** | Success Rate | > 99.5% | < 99% |
| **性能** | P95 Latency | < 2s | > 5s |
| **成本** | Daily Cost | 预算内 | > 预算 * 1.2 |
| **质量** | User Rating | > 4.0/5 | < 3.5/5 |

### 24.6.2 告警疲劳预防

```python
# ❌ 不好的告警：每次错误都告警
if error_rate > 0:
    send_alert("有错误发生")

# ✅ 好的告警：错误率超过阈值且持续一段时间
if error_rate > 5 and duration > 5_minutes:
    send_alert("错误率异常高")
```

### 24.6.3 成本控制策略

1. **预算告警**：设置每日成本上限
2. **模型降级**：简单任务使用便宜模型
3. **缓存优化**：相似问题重用结果
4. **Prompt 优化**：减少不必要的 Token
5. **批处理**：合并多个请求

---

## 本章总结

**核心收获**：

1. ✅ **监控面板**：实时追踪请求量、延迟、错误率、Token 消耗
2. ✅ **智能告警**：自动检测异常，及时通知团队
3. ✅ **在线 Playground**：快速测试提示与参数，无需部署
4. ✅ **运行结果标注**：从生产数据构建黄金数据集
5. ✅ **成本分析**：精细化成本管理与优化

**完整可观测性体系**：

```
Chapter 22 (Tracing) → 看到发生了什么（调试）
Chapter 23 (Evaluation) → 判断做得好不好（质量）
Chapter 24 (Monitoring) → 持续保持稳定（生产）
```

**下一章预告**：
Chapter 25 将学习 **LangServe 基础**，掌握如何将 LangChain 应用部署为生产级 REST API。

---

## 练习题

### 基础练习

1. **计算监控指标**：查询最近 1 小时的 Runs，计算成功率、平均延迟、P95 延迟。

2. **配置告警**：在 LangSmith UI 中配置一个高错误率告警（> 5%）。

3. **Playground 测试**：在 Playground 中测试不同 temperature 参数对输出的影响。

### 进阶练习

4. **自定义监控脚本**：编写一个脚本，每分钟检查错误率，超过 5% 时发送 Slack 通知。

5. **成本分析**：计算最近 7 天的总成本，并按模型拆分。

6. **黄金数据集构建**：从用户好评 Runs 中筛选 50 个高质量样本，构建黄金数据集。

### 挑战练习

7. **异常检测**：实现基于 3-sigma 规则的延迟异常检测。

8. **持续改进流程**：设计一个每周自动运行的脚本，从生产数据更新数据集并重新评估。

9. **成本优化方案**：分析当前成本结构，提出至少 3 个优化建议并估算节省金额。

---

## 扩展阅读

- [LangSmith Monitoring Guide](https://docs.smith.langchain.com/monitoring)
- [LangSmith Alerts Documentation](https://docs.smith.langchain.com/alerts)
- [LangSmith Playground Tutorial](https://blog.langchain.dev/langsmith-playground/)
- [Cost Optimization Best Practices](https://blog.langchain.dev/optimizing-llm-costs/)

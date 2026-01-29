# Chapter 23: LangSmith 评估系统

## 本章概览

Tracing 让你看到**发生了什么**，而 Evaluation（评估）帮你判断**做得好不好**。LangSmith 的评估系统提供完整的工具链：从数据集管理、离线批量评估、多维度指标计算，到在线用户反馈收集与 A/B 测试。本章将深入学习如何构建 LLM 应用的质量保障体系，实现可量化、可复现的持续改进。

**本章重点**：
- 数据集（Dataset）管理与版本控制
- 离线评估（Evaluation）完整流程
- 评估指标（Evaluators）：LLM-as-Judge、距离度量、自定义
- A/B 测试与对比实验
- 在线反馈收集与闭环优化

---

## 23.1 数据集管理

### 23.1.1 为什么需要数据集？

**问题场景**：如何验证提示词改进真的有效？

```python
# 改进前的提示
prompt_v1 = "Translate to French: {text}"

# 改进后的提示
prompt_v2 = """You are a professional translator. 
Translate the following text to French while preserving tone and cultural nuances:

{text}"""

# ❓ 问题：哪个更好？如何证明？
```

**没有数据集的困境**：
- 🤔 依靠主观感觉（"感觉 v2 更好"）
- 🤔 只测试 1-2 个样本（不具代表性）
- 🤔 无法复现（下次测试时忘记用什么输入）
- 🤔 无法量化改进（到底好了多少？）

**有数据集的优势**：
- ✅ 客观评估：用相同数据测试所有版本
- ✅ 代表性：覆盖各种边界情况
- ✅ 可复现：随时重新评估
- ✅ 可量化：计算准确率、BLEU 等指标

### 23.1.2 创建数据集

**方法 1：代码创建**

```python
from langsmith import Client

client = Client()

# 创建数据集
dataset_name = "translation-test-set"
dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="测试翻译质量的标准数据集"
)

# 添加示例
examples = [
    {
        "inputs": {"text": "Hello, world!"},
        "outputs": {"translation": "Bonjour, le monde !"}
    },
    {
        "inputs": {"text": "How are you?"},
        "outputs": {"translation": "Comment vas-tu ?"}
    },
    {
        "inputs": {"text": "The weather is nice today."},
        "outputs": {"translation": "Il fait beau aujourd'hui."}
    },
]

for example in examples:
    client.create_example(
        dataset_id=dataset.id,
        inputs=example["inputs"],
        outputs=example["outputs"]
    )

print(f"✅ Created dataset '{dataset_name}' with {len(examples)} examples")
```

**方法 2：从 CSV 导入**

```python
import pandas as pd

# 准备 CSV 文件
data = {
    "input_text": ["Hello", "Goodbye", "Thank you"],
    "expected_translation": ["Bonjour", "Au revoir", "Merci"]
}
df = pd.DataFrame(data)
df.to_csv("translation_dataset.csv", index=False)

# 从 CSV 创建数据集
dataset = client.create_dataset(dataset_name="translation-from-csv")

# 读取并添加示例
for _, row in df.iterrows():
    client.create_example(
        dataset_id=dataset.id,
        inputs={"text": row["input_text"]},
        outputs={"translation": row["expected_translation"]}
    )
```

**方法 3：从 Trace 创建（生产数据复用）**

```python
# 从成功的 Run 创建示例
from langsmith import Client

client = Client()

# 查询成功的 Runs
runs = client.list_runs(
    project_name="production-chatbot",
    filter='status="success" AND feedback.score > 0.8'  # 高分 Run
)

# 创建数据集
dataset = client.create_dataset(dataset_name="production-golden-set")

# 添加高质量样本
for run in runs[:50]:  # 取前 50 个
    client.create_example(
        dataset_id=dataset.id,
        inputs=run.inputs,
        outputs=run.outputs
    )
```

### 23.1.3 数据集版本管理

```python
# 创建多个版本
dataset_v1 = client.create_dataset(
    dataset_name="qa-dataset-v1.0",
    description="初始版本"
)

# 后续创建新版本（不同名称）
dataset_v2 = client.create_dataset(
    dataset_name="qa-dataset-v2.0",
    description="增加边界情况测试"
)

# 复制数据集
def clone_dataset(old_name: str, new_name: str):
    old_dataset = client.read_dataset(dataset_name=old_name)
    new_dataset = client.create_dataset(dataset_name=new_name)
    
    # 复制所有示例
    examples = client.list_examples(dataset_id=old_dataset.id)
    for example in examples:
        client.create_example(
            dataset_id=new_dataset.id,
            inputs=example.inputs,
            outputs=example.outputs
        )
    
    return new_dataset

# 使用
clone_dataset("qa-dataset-v1.0", "qa-dataset-v2.0-candidate")
```

### 23.1.4 数据集质量标准

**好的评估数据集应具备**：

| 标准 | 说明 | 示例 |
|------|------|------|
| **代表性** | 覆盖真实使用场景 | 包含简单、中等、复杂问题 |
| **多样性** | 不同类型输入 | 短文本、长文本、特殊字符 |
| **边界情况** | 极端输入 | 空输入、超长输入、歧义输入 |
| **标准答案** | 高质量参考输出 | 人工审核的"黄金标准" |
| **规模适中** | 50-500 条 | 太少不代表，太多浪费 |

**反例：糟糕的数据集**

```python
# ❌ 不好的数据集
bad_examples = [
    {"inputs": {"q": "hi"}, "outputs": {"a": "hello"}},  # 太简单
    {"inputs": {"q": "hi"}, "outputs": {"a": "hey"}},    # 重复输入
    {"inputs": {"q": "hi"}, "outputs": {"a": "hi"}},     # 重复输入
]
# 问题：缺乏多样性，无法反映真实场景
```

**正例：高质量数据集**

```python
# ✅ 好的数据集
good_examples = [
    # 简单问题
    {
        "inputs": {"question": "What is the capital of France?"},
        "outputs": {"answer": "Paris"}
    },
    # 需要推理
    {
        "inputs": {"question": "If a train leaves at 2pm and arrives at 5pm, how long is the journey?"},
        "outputs": {"answer": "3 hours"}
    },
    # 歧义问题（需要澄清）
    {
        "inputs": {"question": "What is the best programming language?"},
        "outputs": {"answer": "It depends on your use case. For web development, JavaScript is popular. For data science, Python is widely used."}
    },
    # 超出知识范围
    {
        "inputs": {"question": "What will happen tomorrow?"},
        "outputs": {"answer": "I cannot predict future events."}
    },
]
```

---

## 23.2 离线评估（Evaluation）

### 23.2.1 evaluate() 函数基础

<div data-component="EvaluationPipeline"></div>

**基本评估流程**：

```python
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 初始化
client = Client()

# 定义链
prompt = ChatPromptTemplate.from_template("Translate to French: {text}")
llm = ChatOpenAI(model="gpt-4")
chain = prompt | llm

# 运行评估
results = client.evaluate(
    lambda inputs: chain.invoke(inputs),  # 要评估的函数
    data=dataset_name,                     # 数据集名称
    evaluators=[...],                      # 评估器（下文详解）
    experiment_prefix="translation-v1"     # 实验名称前缀
)

print(f"Evaluation completed: {results['experiment_name']}")
```

**evaluate() 工作流程**：

```
1. 加载数据集 (Dataset)
   ↓
2. 对每个示例调用链 (Chain.invoke)
   ↓
3. 应用所有 Evaluators
   ↓
4. 聚合评估结果
   ↓
5. 保存到 LangSmith（可视化查看）
```

### 23.2.2 完整评估示例

```python
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.smith import RunEvalConfig

client = Client()

# 创建测试链
def create_translation_chain(model_name: str):
    prompt = ChatPromptTemplate.from_template(
        "Translate the following text to French:\n\n{text}"
    )
    llm = ChatOpenAI(model=model_name)
    return prompt | llm

# 定义评估器（下一节详解）
eval_config = RunEvalConfig(
    evaluators=[
        "qa",  # 内置 QA 评估器
        "embedding_distance",  # 嵌入距离
    ]
)

# 运行评估
chain = create_translation_chain("gpt-4")

results = evaluate(
    lambda inputs: chain.invoke(inputs).content,  # 提取文本内容
    data="translation-test-set",
    evaluators=eval_config.evaluators,
    experiment_prefix="gpt4-baseline",
)

# 查看结果
print(f"✅ Experiment: {results['experiment_name']}")
print(f"📊 Results: {results['results']}")
```

### 23.2.3 批量评估并行化

```python
from langsmith.evaluation import evaluate
from concurrent.futures import ThreadPoolExecutor

# 方法 1：evaluate() 自动并行
results = evaluate(
    chain,
    data=dataset_name,
    evaluators=evaluators,
    max_concurrency=10,  # 并行度（默认值会自动设置）
)

# 方法 2：手动控制并行
def evaluate_parallel(chains: list, dataset_name: str):
    """并行评估多个链"""
    results_list = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i, chain in enumerate(chains):
            future = executor.submit(
                evaluate,
                chain,
                data=dataset_name,
                evaluators=evaluators,
                experiment_prefix=f"chain-{i}"
            )
            futures.append(future)
        
        for future in futures:
            results_list.append(future.result())
    
    return results_list

# 使用
chains = [
    create_translation_chain("gpt-4"),
    create_translation_chain("gpt-3.5-turbo"),
    create_translation_chain("claude-3-opus"),
]

all_results = evaluate_parallel(chains, "translation-test-set")
```

### 23.2.4 评估结果查看

**在 LangSmith UI 中查看**：

1. 访问 [https://smith.langchain.com](https://smith.langchain.com)
2. 进入 **Datasets** 页面
3. 选择数据集 → **Experiments** 标签
4. 查看每个实验的：
   - **整体分数**（平均值、中位数）
   - **示例级别结果**（每条样本的得分）
   - **对比视图**（多个实验对比）

**代码中查看结果**：

```python
# 获取实验详情
experiment = client.read_project(project_name=results['experiment_name'])

# 获取所有 Runs
runs = list(client.list_runs(project_name=results['experiment_name']))

# 统计
total_runs = len(runs)
successful_runs = sum(1 for r in runs if r.status == "success")
failed_runs = total_runs - successful_runs

print(f"Total: {total_runs}, Success: {successful_runs}, Failed: {failed_runs}")

# 获取评估分数
scores = []
for run in runs:
    if run.feedback_stats:
        for key, value in run.feedback_stats.items():
            if "score" in key.lower():
                scores.append(value.get("avg", 0))

avg_score = sum(scores) / len(scores) if scores else 0
print(f"Average Score: {avg_score:.2f}")
```

---

## 23.3 评估指标（Evaluators）

### 23.3.1 LLM-as-Judge：Criteria Evaluator

使用 LLM 作为评判者，评估输出质量。

**基本用法**：

```python
from langchain.evaluation import load_evaluator

# 创建 Criteria Evaluator
criteria_eval = load_evaluator("criteria", criteria="correctness")

# 评估单个样本
result = criteria_eval.evaluate_strings(
    prediction="Paris is the capital of France.",
    reference="The capital of France is Paris.",
    input="What is the capital of France?"
)

print(result)
# {'reasoning': '...', 'value': 'Y', 'score': 1}
```

**自定义评估标准**：

```python
from langchain.evaluation import CriteriaEvalChain

# 自定义标准
custom_criteria = {
    "politeness": "Is the response polite and respectful?",
    "completeness": "Does the response fully answer the question?",
    "clarity": "Is the response clear and easy to understand?"
}

# 创建评估链
eval_chain = CriteriaEvalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4"),
    criteria=custom_criteria
)

# 评估
result = eval_chain.evaluate_strings(
    prediction="I don't know.",
    input="What is the capital of France?",
    reference="Paris"
)
```

**多维度评估**：

```python
from langsmith.evaluation import evaluate, EvaluatorType

# 定义多个评估维度
evaluators = [
    # 正确性
    {
        "type": EvaluatorType.CRITERIA,
        "criteria": "correctness",
        "llm": ChatOpenAI(model="gpt-4")
    },
    # 简洁性
    {
        "type": EvaluatorType.CRITERIA,
        "criteria": "conciseness",
        "llm": ChatOpenAI(model="gpt-4")
    },
    # 专业性
    {
        "type": EvaluatorType.CRITERIA,
        "criteria": {
            "professionalism": "Is the response professional and appropriate for a business setting?"
        },
        "llm": ChatOpenAI(model="gpt-4")
    },
]

# 评估
results = evaluate(
    chain,
    data=dataset_name,
    evaluators=evaluators,
)
```

### 23.3.2 Embedding Distance

通过嵌入向量的距离度量语义相似度。

```python
from langchain.evaluation import load_evaluator
from langchain_openai import OpenAIEmbeddings

# 创建 Embedding Distance Evaluator
embedding_eval = load_evaluator(
    "embedding_distance",
    embeddings=OpenAIEmbeddings(),
    distance_metric="cosine"  # 或 "euclidean", "manhattan"
)

# 评估
result = embedding_eval.evaluate_strings(
    prediction="Paris is the capital of France.",
    reference="The capital of France is Paris."
)

print(result)
# {'score': 0.95}  # 分数越高越相似（cosine similarity）
```

**适用场景**：
- ✅ 语义相似度判断（改写、摘要）
- ✅ 多语言评估（嵌入空间对齐）
- ❌ 精确匹配要求（如代码生成）

### 23.3.3 String Distance（编辑距离、BLEU）

**编辑距离（Levenshtein Distance）**：

```python
from langchain.evaluation import load_evaluator

string_eval = load_evaluator("string_distance", distance="levenshtein")

result = string_eval.evaluate_strings(
    prediction="Bonjour le monde",
    reference="Bonjour, le monde!"
)

print(result)
# {'score': 2}  # 编辑距离（越小越好）
```

**BLEU Score（翻译评估）**：

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def bleu_evaluator(prediction: str, reference: str) -> dict:
    """BLEU 评估器"""
    # Tokenize
    pred_tokens = prediction.split()
    ref_tokens = [reference.split()]  # BLEU 需要列表的列表
    
    # 计算 BLEU
    smoothing = SmoothingFunction().method1
    score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)
    
    return {"score": score}

# 使用
result = bleu_evaluator(
    prediction="Bonjour le monde",
    reference="Bonjour, le monde!"
)
print(result)  # {'score': 0.7071...}
```

### 23.3.4 Regex Evaluator

```python
import re

def regex_evaluator(pattern: str):
    """正则表达式评估器"""
    def evaluate(prediction: str, **kwargs) -> dict:
        match = re.search(pattern, prediction)
        return {
            "score": 1 if match else 0,
            "reasoning": f"Pattern '{pattern}' {'found' if match else 'not found'}"
        }
    return evaluate

# 示例：检查输出是否包含数字
number_eval = regex_evaluator(r'\d+')

result = number_eval(prediction="The answer is 42")
print(result)  # {'score': 1, 'reasoning': "Pattern '\\d+' found"}
```

### 23.3.5 自定义评估函数

```python
from langsmith.evaluation import EvaluationResult

def custom_length_evaluator(max_length: int):
    """自定义评估器：检查输出长度"""
    def evaluate(run, example):
        prediction = run.outputs.get("output", "")
        length = len(prediction)
        
        # 返回 EvaluationResult
        return EvaluationResult(
            key="length_check",
            score=1 if length <= max_length else 0,
            comment=f"Length: {length} ({'PASS' if length <= max_length else 'FAIL'})"
        )
    
    return evaluate

# 使用
evaluators = [
    custom_length_evaluator(max_length=200)
]

results = evaluate(
    chain,
    data=dataset_name,
    evaluators=evaluators,
)
```

**复杂自定义评估器示例**：

```python
def fact_checker_evaluator(llm):
    """事实准确性检查器"""
    def evaluate(run, example):
        prediction = run.outputs.get("output", "")
        reference = example.outputs.get("answer", "")
        
        # 使用 LLM 检查事实准确性
        prompt = f"""Compare the following two statements and check if the PREDICTION contains factual errors compared to the REFERENCE.

REFERENCE: {reference}
PREDICTION: {prediction}

Is the PREDICTION factually correct? Answer with:
- "CORRECT" if factually accurate
- "INCORRECT" if contains factual errors
- "PARTIALLY_CORRECT" if mostly correct with minor issues

Answer: """
        
        result = llm.invoke(prompt).content.strip()
        
        score_map = {
            "CORRECT": 1.0,
            "PARTIALLY_CORRECT": 0.5,
            "INCORRECT": 0.0
        }
        
        return EvaluationResult(
            key="fact_check",
            score=score_map.get(result, 0.0),
            comment=f"Fact check result: {result}"
        )
    
    return evaluate

# 使用
evaluators = [
    fact_checker_evaluator(ChatOpenAI(model="gpt-4"))
]
```

---

## 23.4 A/B 测试

### 23.4.1 对比不同提示版本

<div data-component="ABTestComparison"></div>

```python
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

client = Client()

# 版本 A：简单提示
prompt_v1 = ChatPromptTemplate.from_template("Translate to French: {text}")

# 版本 B：详细提示
prompt_v2 = ChatPromptTemplate.from_template(
    """You are a professional translator with expertise in French.
    
Translate the following text to French while:
- Preserving the original tone
- Using appropriate cultural references
- Maintaining grammatical accuracy

Text: {text}"""
)

# 创建两个链
llm = ChatOpenAI(model="gpt-4")
chain_v1 = prompt_v1 | llm
chain_v2 = prompt_v2 | llm

# 评估两个版本
results_v1 = evaluate(
    chain_v1,
    data="translation-test-set",
    evaluators=[...],
    experiment_prefix="prompt-v1-simple"
)

results_v2 = evaluate(
    chain_v2,
    data="translation-test-set",
    evaluators=[...],
    experiment_prefix="prompt-v2-detailed"
)

# 对比结果
print(f"V1 Average Score: {results_v1['avg_score']}")
print(f"V2 Average Score: {results_v2['avg_score']}")
```

### 23.4.2 对比不同模型

```python
models = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus-20240229"]

results_dict = {}

for model_name in models:
    chain = prompt | ChatOpenAI(model=model_name)
    
    results = evaluate(
        chain,
        data="translation-test-set",
        evaluators=evaluators,
        experiment_prefix=f"model-{model_name}"
    )
    
    results_dict[model_name] = results

# 生成对比报告
print("\n📊 Model Comparison Report")
print("="*60)
for model, results in results_dict.items():
    print(f"{model:30} Score: {results['avg_score']:.3f}")
```

### 23.4.3 统计显著性分析

```python
from scipy import stats

def compare_experiments(exp1_scores: list, exp2_scores: list):
    """比较两个实验的统计显著性"""
    # T-test
    t_stat, p_value = stats.ttest_ind(exp1_scores, exp2_scores)
    
    # 效应量（Cohen's d）
    mean1, mean2 = np.mean(exp1_scores), np.mean(exp2_scores)
    std = np.sqrt((np.std(exp1_scores)**2 + np.std(exp2_scores)**2) / 2)
    cohen_d = (mean2 - mean1) / std
    
    return {
        "p_value": p_value,
        "is_significant": p_value < 0.05,
        "cohen_d": cohen_d,
        "effect_size": "small" if abs(cohen_d) < 0.5 else "medium" if abs(cohen_d) < 0.8 else "large"
    }

# 使用
exp1_scores = [0.8, 0.85, 0.9, 0.75, 0.88]
exp2_scores = [0.92, 0.95, 0.93, 0.89, 0.94]

comparison = compare_experiments(exp1_scores, exp2_scores)
print(comparison)
# {
#   'p_value': 0.012,
#   'is_significant': True,
#   'cohen_d': 1.8,
#   'effect_size': 'large'
# }
```

---

## 23.5 在线评估与反馈

### 23.5.1 用户反馈收集（Feedback）

<div data-component="FeedbackDashboard"></div>

**收集 Thumbs Up/Down**：

```python
from langsmith import Client

client = Client()

# 用户给了好评
client.create_feedback(
    run_id="run-abc123",  # 从 Trace 中获取
    key="user_rating",
    score=1,  # 1 = Thumbs Up, 0 = Thumbs Down
    comment="Great response!"
)

# 用户给了差评
client.create_feedback(
    run_id="run-def456",
    key="user_rating",
    score=0,
    comment="Incorrect answer"
)
```

**集成到应用中**：

```python
from langchain_openai import ChatOpenAI
from langsmith import Client
from langsmith.run_helpers import traceable

client = Client()
llm = ChatOpenAI(model="gpt-4")

@traceable
def chatbot(question: str) -> dict:
    """聊天机器人"""
    response = llm.invoke(question)
    return {"answer": response.content}

# 在你的 Web 应用中
def handle_user_query(question: str, session_id: str):
    # 调用聊天机器人
    result = chatbot(question)
    
    # 返回答案和 run_id（用于反馈）
    return {
        "answer": result["answer"],
        "run_id": result["__run"].id  # 获取 run_id
    }

# 用户反馈端点
def submit_feedback(run_id: str, thumbs_up: bool, comment: str = ""):
    client.create_feedback(
        run_id=run_id,
        key="user_rating",
        score=1 if thumbs_up else 0,
        comment=comment
    )
```

### 23.5.2 自定义反馈 Schema

```python
# 多维度反馈
client.create_feedback(
    run_id="run-abc123",
    key="detailed_feedback",
    score=0.8,  # 总体分数
    value={
        "accuracy": 0.9,
        "relevance": 0.8,
        "completeness": 0.7,
        "clarity": 0.85
    },
    comment="Good answer but missing some details"
)

# 分类反馈
client.create_feedback(
    run_id="run-def456",
    key="issue_type",
    value="factual_error",  # 或 "off_topic", "incomplete", etc.
    comment="Stated Paris is in Germany"
)
```

### 23.5.3 反馈数据导入评估

```python
# 获取有高分反馈的 Runs
high_rated_runs = client.list_runs(
    project_name="production-chatbot",
    filter='feedback.user_rating.score > 0.8'
)

# 创建数据集
dataset = client.create_dataset(dataset_name="high-quality-prod-samples")

# 添加到数据集
for run in high_rated_runs[:100]:
    client.create_example(
        dataset_id=dataset.id,
        inputs=run.inputs,
        outputs=run.outputs
    )

# 使用此数据集评估新版本
results = evaluate(
    new_chain,
    data="high-quality-prod-samples",
    evaluators=evaluators,
)
```

### 23.5.4 反馈驱动的持续改进

**完整闭环工作流**：

```
1. 生产环境运行
   ↓ (自动追踪)
2. 收集用户反馈
   ↓ (筛选高质量样本)
3. 构建评估数据集
   ↓ (离线评估)
4. 测试改进版本
   ↓ (A/B 测试)
5. 部署获胜版本
   ↓ (循环)
回到第 1 步
```

**实现示例**：

```python
import schedule
import time

def weekly_improvement_cycle():
    """每周自动改进流程"""
    # 1. 收集上周高分样本
    last_week = datetime.now() - timedelta(days=7)
    high_rated = client.list_runs(
        project_name="production",
        filter=f'feedback.user_rating.score > 0.8 AND start_time > "{last_week.isoformat()}"'
    )
    
    # 2. 更新数据集
    dataset = client.read_dataset(dataset_name="golden-set")
    for run in high_rated[:20]:  # 每周添加 20 个
        client.create_example(
            dataset_id=dataset.id,
            inputs=run.inputs,
            outputs=run.outputs
        )
    
    # 3. 重新评估当前版本
    current_results = evaluate(
        current_chain,
        data="golden-set",
        evaluators=evaluators,
        experiment_prefix="weekly-baseline"
    )
    
    # 4. 评估实验版本
    experimental_results = evaluate(
        experimental_chain,
        data="golden-set",
        evaluators=evaluators,
        experiment_prefix="weekly-experiment"
    )
    
    # 5. 决定是否部署
    if experimental_results['avg_score'] > current_results['avg_score'] * 1.05:
        print("🎉 Experimental version is 5% better! Deploying...")
        deploy_new_version(experimental_chain)
    else:
        print("⏸️ No significant improvement. Keeping current version.")

# 定时执行
schedule.every().monday.at("02:00").do(weekly_improvement_cycle)

while True:
    schedule.run_pending()
    time.sleep(3600)
```

---

## 23.6 最佳实践

### 23.6.1 数据集构建策略

```python
# ✅ 好的策略：分层采样
def build_balanced_dataset():
    """构建平衡的数据集"""
    dataset = client.create_dataset(dataset_name="balanced-qa-set")
    
    categories = {
        "simple": 20,      # 20% 简单问题
        "medium": 50,      # 50% 中等问题
        "complex": 20,     # 20% 复杂问题
        "edge_case": 10    # 10% 边界情况
    }
    
    for category, count in categories.items():
        examples = load_examples_by_category(category, count)
        for example in examples:
            client.create_example(
                dataset_id=dataset.id,
                inputs=example["inputs"],
                outputs=example["outputs"],
                metadata={"category": category}
            )
```

### 23.6.2 评估器选择指南

| 任务类型 | 推荐评估器 | 原因 |
|---------|-----------|------|
| 翻译 | BLEU + Embedding Distance | 兼顾精确与语义 |
| 摘要 | ROUGE + LLM-as-Judge | 覆盖率 + 质量 |
| QA | Exact Match + Criteria | 准确性 + 完整性 |
| 对话 | LLM-as-Judge (多维度) | 需要主观判断 |
| 代码生成 | Execution + Unit Tests | 功能正确性 |

### 23.6.3 评估成本控制

```python
# 策略 1：使用更便宜的模型评估
cheap_evaluator = load_evaluator(
    "criteria",
    criteria="correctness",
    llm=ChatOpenAI(model="gpt-3.5-turbo")  # 而非 GPT-4
)

# 策略 2：缓存评估结果
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_evaluate(prediction: str, reference: str) -> float:
    return evaluator.evaluate_strings(
        prediction=prediction,
        reference=reference
    )["score"]

# 策略 3：采样评估
def sample_evaluate(chain, dataset_name: str, sample_rate: float = 0.2):
    """仅评估 20% 的样本"""
    examples = list(client.list_examples(dataset_name=dataset_name))
    sampled = random.sample(examples, int(len(examples) * sample_rate))
    
    # 只评估采样的示例
    # ...
```

---

## 本章总结

**核心收获**：

1. ✅ **数据集是质量保障的基石**
   - 代表性、多样性、边界情况
   - 版本管理与持续更新
   - 生产数据复用

2. ✅ **离线评估流程标准化**
   - evaluate() 一站式评估
   - 多维度评估器组合
   - 批量并行加速

3. ✅ **评估器生态丰富**
   - LLM-as-Judge：灵活但成本高
   - 距离度量：快速且便宜
   - 自定义：适配特定需求

4. ✅ **A/B 测试驱动迭代**
   - 对比不同版本
   - 统计显著性分析
   - 数据驱动决策

5. ✅ **在线反馈闭环优化**
   - 收集用户评价
   - 构建黄金数据集
   - 持续改进流程

**下一章预告**：
Chapter 24 将学习 **LangSmith 生产监控**，掌握实时 Dashboard、告警配置、Playground 使用、成本分析等生产环境必备技能。

---

## 练习题

### 基础练习

1. **创建数据集**：为你的聊天机器人创建一个包含 10 条测试样本的数据集。

2. **基础评估**：使用 `evaluate()` 函数评估一个简单的翻译链。

3. **多评估器**：组合使用 Embedding Distance 和 String Distance 评估同一个任务。

### 进阶练习

4. **自定义评估器**：实现一个检查输出是否包含特定关键词的评估器。

5. **A/B 测试**：对比两个不同的提示模板，判断哪个效果更好。

6. **反馈收集**：为你的应用添加用户反馈功能（Thumbs Up/Down）。

### 挑战练习

7. **统计分析**：实现一个函数，计算两个实验之间的统计显著性（p-value）。

8. **持续改进流程**：设计一个自动化脚本，每周从生产数据中提取高分样本更新数据集。

9. **成本优化评估**：实现一个评估策略，在保持准确性的前提下将评估成本降低 50%。

---

## 扩展阅读

- [LangSmith Evaluation Guide](https://docs.smith.langchain.com/evaluation)
- [LangChain Evaluators](https://python.langchain.com/docs/guides/evaluation/)
- [Building Quality Datasets for LLM Evaluation](https://blog.langchain.dev/building-quality-datasets/)
- [A/B Testing for LLM Applications](https://blog.langchain.dev/ab-testing-llm-apps/)

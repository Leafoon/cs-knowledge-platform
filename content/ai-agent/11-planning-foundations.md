---
title: "第11章：规划基础 — Agent 的思维链"
description: "深入理解 Agent 规划机制：Chain-of-Thought、Zero-Shot CoT、Self-Consistency、计划分解与多步推理的理论基础与工程实现。"
date: "2026-06-11"
---

# 第11章：规划基础 — Agent 的思维链

---

## 11.1 为什么 Agent 需要规划？

| 维度 | 直觉响应 | 规划响应 |
|:---|:---|:---|
| 思考深度 | 浅层模式匹配 | 深度推理分解 |
| 适用任务 | 简单问答 | 复杂多步任务 |
| 错误率 | 高 | 低 |
| Token 消耗 | 低 | 高 |

---

## 11.2 Chain-of-Thought 推理

$$
P(a|q) = \sum_{z_1, z_2, \ldots, z_n} P(z_1|q) \cdot P(z_2|q, z_1) \cdots P(a|q, z_1, \ldots, z_n)
$$

```python
STANDARD_COT_PROMPT = """
问题: {question}

推理过程:
1. 首先，理解问题的核心需求
2. 然后，识别关键信息和已知条件
3. 接下来，确定解决步骤
4. 最后，得出结论

思考过程：
"""
```

---

## 11.3 Self-Consistency

```python
from collections import Counter

def self_consistency(question, llm, n_samples=5):
    answers = []
    for _ in range(n_samples):
        response = llm.invoke([HumanMessage(content=f"{question}\n\nLet's think step by step.")])
        answer = extract_final_answer(response.content)
        answers.append(answer)
    vote_result = Counter(answers).most_common(1)[0]
    return {"answer": vote_result[0], "confidence": vote_result[1] / n_samples}
```

---

## 11.4 计划分解

```python
decomposition_prompt = """将以下任务分解为子任务列表。

任务：{task}
输出（每行一个）："""

def decompose_task(task, llm):
    response = llm.invoke([HumanMessage(content=decomposition_prompt.format(task=task))])
    return [l.strip() for l in response.content.split("\n") if l.strip()]

def recursive_decompose(task, llm, depth=0, max_depth=3):
    if depth >= max_depth: return [task]
    is_simple = "yes" in llm.invoke([HumanMessage(
        content=f"判断以下任务是否简单（Yes/No）：{task}"
    )]).content.lower()
    if is_simple: return [task]
    subtasks = decompose_task(task, llm)
    result = []
    for st in subtasks:
        result.extend(recursive_decompose(st, llm, depth + 1, max_depth))
    return result
```

---

## 11.5 Plan-and-Solve

```python
PLAN_AND_SOLVE_PROMPT = """
问题：{question}

计划：
1. 首先，[第一步]
2. 然后，[第二步]
3. 最后，[得出答案]

执行：
[逐步执行]

最终答案：[答案]
"""
```

---

## 11.6 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| CoT | 将复杂推理分解为简单步骤 |
| Self-Consistency | 多次采样+投票 |
| 任务分解 | 一次性分解和递归分解 |
| Plan-and-Solve | 先规划后执行 |

> **下一章预告**
>
> 在第 12 章中，我们将学习 ToT、GoT 等前沿规划算法。

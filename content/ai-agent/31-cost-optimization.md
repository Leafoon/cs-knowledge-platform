---
title: "第31章：成本优化 — Token 经济学"
description: "系统讲解 Agent 成本优化：Token 计数与预算、模型路由策略、缓存复用、Prompt 压缩与 ROI 分析。"
date: "2026-06-11"
---

# 第31章：成本优化 — Token 经济学

---

## 31.1 成本构成

$$
\text{Cost} = \sum_{i=1}^{N} (C_{\text{input}} \times T_{\text{input},i} + C_{\text{output}} \times T_{\text{output},i})
$$

| 模型 | 输入价格 | 输出价格 |
|:---|:---|:---|
| GPT-4o-mini | $0.15/M | $0.60/M |
| GPT-4o | $2.50/M | $10.00/M |
| Claude Sonnet 4 | $3.00/M | $15.00/M |

---

## 31.2 优化策略

```python
class SmartModelRouter:
    def route(self, query):
        complexity = self._estimate_complexity(query)
        if complexity < 0.3: return "gpt-4o-mini"
        elif complexity < 0.7: return "gpt-4o"
        else: return "claude-sonnet-4-20250514"
```

```python
class SemanticCache:
    def get(self, query):
        results = self.vectorstore.similarity_search_with_score(query, k=1)
        if results and results[0][1] > 0.95:
            return results[0][0].metadata["response"]
        return None
```

---

## 31.3 ROI 分析

```python
def calculate_roi(time_saved, hourly_rate, tasks_per_month, cost_per_task):
    value = time_saved * hourly_rate * tasks_per_month
    cost = cost_per_task * tasks_per_month
    return {"roi": (value - cost) / cost * 100}
```

---

## 31.4 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| 模型路由 | 简单问题用小模型 |
| 语义缓存 | 相似查询复用 |
| Prompt 压缩 | 减少 token 消耗 |

> **下一章预告**
>
> 在第 32 章中，我们将深入 Agent 安全工程进阶。

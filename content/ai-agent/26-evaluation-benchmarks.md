---
title: "第26章：评测与基准 — Agent 能力度量"
description: "掌握 Agent 评测体系：SWE-bench、WebArena、GAIA、AgentBench 等基准，LLM-as-Judge、任务成功率与评测管线构建。"
date: "2026-06-11"
---

# 第26章：评测与基准 — Agent 能力度量

---

## 26.1 评测维度

| 维度 | 指标 | 重要性 |
|:---|:---|:---|
| 任务完成 | Success Rate | ★★★★★ |
| 推理质量 | Reasoning Accuracy | ★★★★ |
| 工具使用 | Tool Call Accuracy | ★★★★★ |
| 效率 | Steps / Tokens / Cost | ★★★★ |
| 安全性 | Safety Score | ★★★★★ |

---

## 26.2 主流 Benchmark

| Benchmark | 维度 | 任务类型 |
|:---|:---|:---|
| SWE-bench | 代码修复 | GitHub Issue |
| WebArena | Web 操作 | 网站交互 |
| GAIA | 通用助手 | 多步推理 |
| AgentBench | 综合 | 8 种环境 |

---

## 26.3 LLM-as-Judge

```python
def llm_as_judge(question, answer, reference, llm):
    prompt = f"评估以下回答。\n问题：{question}\n参考：{reference}\n回答：{answer}\n评分（1-5）："
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"score": response.content}
```

---

## 26.4 评测管线

```python
class AgentEvaluator:
    def evaluate(self, agent, test_cases):
        import time
        results = []
        for case in test_cases:
            start = time.time()
            result = agent.run(case["input"])
            latency = time.time() - start
            score = 1.0 if case.get("expected", "") in result else 0.0
            results.append({"input": case["input"], "score": score, "latency": latency})
        scores = [r["score"] for r in results]
        return {"success_rate": sum(1 for s in scores if s >= 0.8) / len(scores),
                "avg_score": sum(scores) / len(scores)}
```

---

## 26.5 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| 评测维度 | 完成率、推理、工具使用、效率、安全 |
| Benchmark | SWE-bench、WebArena、GAIA |
| LLM-as-Judge | LLM 评估输出质量 |

> **下一章预告**
>
> 在第 27 章中，我们将学习 Agent 生产级部署。

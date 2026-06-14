---
title: "第24章：数据分析 Agent — 自动化数据洞察"
description: "构建数据分析 Agent：数据加载、探索性分析、可视化生成、统计推理与报告撰写。"
date: "2026-06-11"
---

# 第24章：数据分析 Agent — 自动化数据洞察

---

## 24.1 数据分析 Agent

```python
import pandas as pd

class DataAnalysisAgent:
    def load_data(self, file_path):
        if file_path.endswith('.csv'): self.df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'): self.df = pd.read_excel(file_path)
        return f"数据加载成功：{self.df.shape[0]} 行 × {self.df.shape[1]} 列"

    def explore(self):
        return {"shape": self.df.shape, "columns": list(self.df.columns),
                "dtypes": self.df.dtypes.to_dict(), "missing": self.df.isnull().sum().to_dict()}

    def analyze(self, question):
        schema = self.explore()
        prompt = f"数据信息：{schema}\n用户问题：{question}\n生成 pandas 代码："
        code = self.llm.invoke([HumanMessage(content=prompt)]).content
        local_vars = {"df": self.df, "pd": pd}
        exec(code, {}, local_vars)
        return str(local_vars.get("result", "分析完成"))
```

---

## 24.2 Data Interpreter

```python
class DataInterpreter:
    def run(self, data_path, goal):
        self.load_data(data_path)
        plan = self._create_plan(goal)
        results = [self.analyze(step) for step in plan]
        return self._generate_report(goal, results)
```

---

## 24.3 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| 数据加载 | 支持 CSV/Excel/JSON |
| 探索分析 | 自动统计、缺失值检测 |
| 可视化 | LLM 生成 matplotlib 代码 |

> **下一章预告**
>
> 在第 25 章中，我们将学习 Agent 安全与对齐。

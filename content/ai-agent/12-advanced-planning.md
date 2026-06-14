---
title: "第12章：高级规划算法 — ToT、GoT 与 Plan-and-Solve"
description: "掌握前沿规划算法：Tree of Thoughts、Graph of Thoughts、LATS、Self-Refine、Reflexion 的原理与完整实现，以及算法对比与选型指南。"
date: "2026-06-11"
---

# 第12章：高级规划算法 — ToT、GoT 与 Plan-and-Solve

---

## 12.1 Tree of Thoughts (ToT)

```python
class TreeOfThoughts:
    def __init__(self, llm, breadth=3, depth=3):
        self.llm = llm
        self.breadth = breadth
        self.depth = depth

    def generate_thoughts(self, problem, current_path, n=3):
        prompt = f"问题：{problem}\n当前推理：{current_path or '(开始)'}\n生成 {n} 个可能的下一步："
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return [t.strip() for t in response.content.split("\n") if t.strip()][:n]

    def evaluate_thought(self, problem, thought):
        prompt = f"评估以下推理的质量（0-10分）：\n问题：{problem}\n推理：{thought}\n分数："
        response = self.llm.invoke([HumanMessage(content=prompt)])
        try: return float(response.content.strip()) / 10
        except: return 0.5

    def search(self, problem, strategy="bfs"):
        root = ThoughtNode(content="", value=0, depth=0)
        if strategy == "bfs":
            return self._bfs(problem, root)
        else:
            best = {"value": 0, "content": ""}
            self._dfs(root, problem, best)
            return best["content"]

    def _bfs(self, problem, root):
        import math
        queue, best_solution, best_value = [root], "", 0
        while queue:
            node = queue.pop(0)
            if node.depth >= self.depth:
                if node.value > best_value:
                    best_value, best_solution = node.value, node.content
                continue
            for thought in self.generate_thoughts(problem, node.content):
                value = self.evaluate_thought(problem, f"{node.content}\n{thought}")
                child = ThoughtNode(content=f"{node.content}\n{thought}".strip(), value=value, depth=node.depth + 1)
                node.children.append(child)
                queue.append(child)
        return best_solution
```

---

## 12.2 Graph of Thoughts (GoT)

```python
class GraphOfThoughts:
    def aggregate(self, thoughts):
        prompt = f"将以下思考整合为一个更强的洞察：\n{chr(10).join(f'{i+1}. {t}' for i, t in enumerate(thoughts))}"
        return self.llm.invoke([HumanMessage(content=prompt)]).content

    def refine(self, thought, feedback):
        prompt = f"原思考：{thought}\n反馈：{feedback}\n改进后："
        return self.llm.invoke([HumanMessage(content=prompt)]).content

    def score(self, thought, problem):
        prompt = f"评估（0-10分）：\n问题：{problem}\n思考：{thought}\n分数："
        try: return float(self.llm.invoke([HumanMessage(content=prompt)]).content.strip()) / 10
        except: return 0.5
```

---

## 12.3 Reflexion

```python
class ReflexionAgent:
    def __init__(self, llm, tools, max_retries=3):
        self.llm = llm
        self.tools = tools
        self.max_retries = max_retries
        self.memory = []

    def run(self, task):
        for attempt in range(self.max_retries):
            result = self._execute_with_reflection(task)
            if self._evaluate(task, result): return result
            reflection = self._reflect(task, result, attempt)
            self.memory.append({"attempt": attempt, "reflection": reflection})
        return "达到最大重试次数"

    def _reflect(self, task, failed_result, attempt):
        memory_ctx = "\n".join(f"- 第{m['attempt']+1}次: {m['reflection'][:200]}" for m in self.memory[-3:])
        prompt = f"任务：{task}\n失败结果：{failed_result[:500]}\n历史反思：{memory_ctx}\n请反思根本原因和改进策略："
        return self.llm.invoke([HumanMessage(content=prompt)]).content

    def _evaluate(self, task, result):
        response = self.llm.invoke([HumanMessage(content=f"判断以下结果是否成功完成任务。\n任务：{task}\n结果：{result[:500]}\n回答 Yes/No：")])
        return "yes" in response.content.lower()
```

---

## 12.4 Self-Refine

```python
class SelfRefine:
    def __init__(self, llm, max_iterations=3):
        self.llm = llm
        self.max_iterations = max_iterations

    def run(self, task):
        current = self.llm.invoke([HumanMessage(content=f"请完成：{task}")]).content
        for _ in range(self.max_iterations):
            feedback = self.llm.invoke([HumanMessage(content=f"评估质量（PASS/改进点）：\n任务：{task}\n输出：{current[:1000]}")]).content
            if "PASS" in feedback.upper(): return current
            current = self.llm.invoke([HumanMessage(content=f"改进：\n任务：{task}\n输出：{current[:1000]}\n反馈：{feedback}")]).content
        return current
```

---

## 12.5 LATS

```python
class LATSAgent:
    def search(self, task, n_simulations=10):
        root = {"state": task, "children": [], "visits": 0, "value": 0}
        for _ in range(n_simulations):
            node = self._select(root)
            children = self._expand(node)
            for child in children:
                child["value"] = self._simulate(child)
                child["visits"] = 1
            self._backpropagate(root)
        return self._get_best_path(root)
```

---

## 12.6 算法对比

| 算法 | 搜索策略 | Token 效率 | 适用场景 |
|:---|:---|:---|:---|
| CoT | 线性 | 高 | 简单推理 |
| ToT | 树状 BFS/DFS | 中 | 多方案探索 |
| GoT | 图状 | 低 | 聚合洞察 |
| Reflexion | 循环迭代 | 中 | 从错误学习 |
| Self-Refine | 生成-改进 | 中 | 高质量输出 |
| LATS | MCTS | 低 | 复杂决策 |

---

## 12.7 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| ToT | 搜索树找最优路径 |
| GoT | 图状推理，支持聚合 |
| Reflexion | 执行-反思-改进 |
| Self-Refine | 生成后持续精炼 |
| LATS | MCTS + LLM |

> **下一章预告**
>
> 在第 13 章中，我们将系统梳理 Agent 架构设计模式。

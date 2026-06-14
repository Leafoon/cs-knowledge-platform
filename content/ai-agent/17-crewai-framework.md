---
title: "第17章：CrewAI 协作智能体框架"
description: "掌握 CrewAI 的角色定义、任务编排、协作流程、记忆集成、工具系统与层级管理。"
date: "2026-06-11"
---

# 第17章：CrewAI 协作智能体框架

---

## 17.1 Agent 角色定义

```python
from crewai import Agent, Task, Crew, Process

researcher = Agent(
    role="高级研究分析师",
    goal="发现关于 {topic} 的最新洞察",
    backstory="你是一位经验丰富的研究分析师。",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm="gpt-4o",
)
```

---

## 17.2 任务定义

```python
research_task = Task(
    description="深入研究 {topic} 的最新发展。",
    expected_output="一份详细的研究报告。",
    agent=researcher,
)

writing_task = Task(
    description="基于研究报告撰写技术博客。",
    expected_output="一篇高质量的技术博客。",
    agent=writer,
    context=[research_task],
)
```

---

## 17.3 组建 Crew 并执行

```python
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,
    verbose=True,
)
result = crew.kickoff(inputs={"topic": "AI Agent 智能体开发"})
```

---

## 17.4 协作流程

| 流程 | 执行方式 | 适用场景 |
|:---|:---|:---|
| Sequential | 按顺序执行 | 流程明确 |
| Hierarchical | Manager 分配 | 复杂协作 |

---

## 17.5 CrewAI vs AutoGen vs LangGraph

| 特性 | CrewAI | AutoGen | LangGraph |
|:---|:---|:---|:---|
| 核心理念 | 角色扮演 | 对话协作 | 图编排 |
| 学习曲线 | 低 | 中 | 中高 |
| 适用场景 | 内容创作 | 代码、分析 | 复杂流程 |

---

## 17.6 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| Agent 角色 | role/goal/backstory 定义人格 |
| 任务编排 | Task 定义具体工作 |
| 协作流程 | Sequential 和 Hierarchical |

> **下一章预告**
>
> 在第 18 章中，我们将学习 Semantic Kernel。

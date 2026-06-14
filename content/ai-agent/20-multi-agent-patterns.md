---
title: "第20章：多智能体协作模式"
description: "系统梳理多 Agent 协作范式：主从模式、对等协作、辩论模式、投票机制、流水线模式、黑板模式与市场机制。"
date: "2026-06-11"
---

# 第20章：多智能体协作模式

---

## 20.1 协作模式概览

| 模式 | 通信方式 | 适用场景 | 代表框架 |
|:---|:---|:---|:---|
| **主从模式** | Supervisor 分配 | 明确分工 | LangGraph |
| **对等协作** | 平等对话 | 多视角 | AutoGen |
| **辩论模式** | 交替辩论 | 高质量决策 | 自定义 |
| **投票模式** | 多数投票 | 可靠性 | Self-Consistency |
| **流水线模式** | 线性传递 | 流程化 | CrewAI |
| **黑板模式** | 共享空间 | 复杂问题 | 自定义 |

---

## 20.2 主从模式

```python
def supervisor(state):
    llm = ChatOpenAI(model="gpt-4o")
    response = llm.invoke([SystemMessage(content="选择：research / code / finish"), *state["messages"]])
    return {"next": response.content.strip().lower()}
```

---

## 20.3 辩论模式

```python
class DebateSystem:
    def debate(self, topic):
        history = [f"辩题：{topic}"]
        for _ in range(self.rounds):
            for agent in self.agents:
                prompt = f"辩题：{topic}\n历史：{chr(10).join(history)}\n发表观点："
                response = agent.invoke([HumanMessage(content=prompt)])
                history.append(f"[{agent.name}]: {response.content}")
        return llm.invoke([HumanMessage(content=f"总结辩论：{chr(10).join(history)}")]).content
```

---

## 20.4 投票模式

```python
from collections import Counter

def voting_system(agents, question):
    votes = [agent.invoke([HumanMessage(content=question)]).content.strip() for agent in agents]
    winner = Counter(votes).most_common(1)[0]
    return {"answer": winner[0], "confidence": winner[1] / len(votes)}
```

---

## 20.5 流水线模式

```python
def pipeline_execution(agents, task):
    current = task
    for agent in agents:
        current = agent.invoke([HumanMessage(content=current)]).content
    return current
```

---

## 20.6 黑板模式

```python
class Blackboard:
    def __init__(self): self.data = {}
    def write(self, key, value, author): self.data[key] = {"value": value, "author": author}
    def read_all(self): return {k: v["value"] for k, v in self.data.items()}
```

---

## 20.7 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| 主从模式 | Supervisor 分配 |
| 对等协作 | Agent 平等协作 |
| 辩论模式 | 多轮辩论 |
| 投票模式 | 多数投票 |

> **下一章预告**
>
> 在第 21 章中，我们将深入 Agent 通信协议。

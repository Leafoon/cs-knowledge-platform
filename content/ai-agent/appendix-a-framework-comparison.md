---
title: "Appendix A: Agent 框架对比速查表"
description: "LangChain vs AutoGen vs CrewAI vs Semantic Kernel vs OpenAI Agents SDK vs LlamaIndex 的全面功能对比与选型指南。"
date: "2026-06-11"
---

# Appendix A: Agent 框架对比速查表

---

## A.1 核心对比

| 特性 | LangGraph | AutoGen | CrewAI | SK | Agents SDK | LlamaIndex |
|:---|:---|:---|:---|:---|:---|:---|
| 核心理念 | 图编排 | 对话协作 | 角色扮演 | 企业编排 | 原生集成 | 数据驱动 |
| 学习曲线 | 中 | 中 | 低 | 中 | 低 | 中 |
| 生产就绪 | ★★★★★ | ★★★★ | ★★★ | ★★★★ | ★★★★ | ★★★★ |
| HITL | ✅ 原生 | ✅ | 有限 | ✅ | ✅ Guardrails | ✅ |

## A.2 功能对比

| 功能 | LangGraph | AutoGen | CrewAI | SK | Agents SDK |
|:---|:---|:---|:---|:---|:---|
| 单 Agent | ✅ | ✅ | ✅ | ✅ | ✅ |
| 多 Agent | ✅ | ✅ | ✅ | ✅ | ✅ |
| 代码执行 | 手动 | 内置 | 通过工具 | 通过插件 | 内置 |
| 持久化 | ✅ | 手动 | 手动 | ✅ | 手动 |
| 流式输出 | ✅ | ✅ | 有限 | ✅ | ✅ |

## A.3 选型决策

```
需要构建什么？
├─ 通用 Agent → LangGraph
├─ 多 Agent 对话 → AutoGen
├─ 内容创作 → CrewAI
├─ 企业 .NET/Java → Semantic Kernel
├─ OpenAI 生态 → Agents SDK
└─ 数据 RAG → LlamaIndex
```

## A.4 代码对比

```python
# LangGraph
agent = create_react_agent(model=llm, tools=[weather_tool])
result = agent.invoke({"messages": [("user", "北京天气")]})

# AutoGen
agent = ConversableAgent(name="assistant", llm_config=config)
result = agent.initiate_chat(weather_agent, message="北京天气")

# CrewAI
agent = Agent(role="助手", tools=[weather_tool], llm="gpt-4o")
crew = Crew(agents=[agent], tasks=[Task(description="查询天气", agent=agent)])
result = crew.kickoff()

# Agents SDK
agent = Agent(name="assistant", tools=[get_weather], model="gpt-4o")
result = Runner.run_sync(agent, "北京天气")
```

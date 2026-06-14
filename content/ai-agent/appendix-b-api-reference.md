---
title: "Appendix B: API 与配置速查"
description: "主流 Agent 框架的核心 API、环境变量、配置文件格式、模型参数与常用工具的快速参考。"
date: "2026-06-11"
---

# Appendix B: API 与配置速查

---

## B.1 环境变量

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Google
GOOGLE_API_KEY=AIza...

# LangSmith
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_TRACING_V2=true
```

## B.2 LangChain/LangGraph

```python
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
```

## B.3 OpenAI API

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "hello"}],
    tools=[{"type": "function", "function": {...}}],
    tool_choice="auto",
    temperature=0,
)
```

## B.4 Anthropic API

```python
import anthropic
client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    tools=[{"name": "...", "input_schema": {...}}],
    messages=[{"role": "user", "content": "..."}]
)
```

## B.5 AutoGen

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager
agent = ConversableAgent(name="assistant", llm_config=config)
```

## B.6 CrewAI

```python
from crewai import Agent, Task, Crew, Process
agent = Agent(role="...", goal="...", backstory="...", tools=[...])
crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)
result = crew.kickoff()
```

## B.7 常用工具

| 工具 | 用途 | 安装 |
|:---|:---|:---|
| Tavily | AI 搜索 | `pip install tavily-python` |
| Playwright | 浏览器 | `pip install playwright` |
| FAISS | 向量搜索 | `pip install faiss-cpu` |
| Chroma | 向量数据库 | `pip install chromadb` |
| tiktoken | Token 计数 | `pip install tiktoken` |

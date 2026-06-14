---
title: "第29章：MCP 与 A2A 协议 — Agent 互操作标准"
description: "深入 Model Context Protocol 与 Agent-to-Agent 协议：设计哲学、消息格式、工具暴露、传输层与跨框架互操作。"
date: "2026-06-11"
---

# 第29章：MCP 与 A2A 协议 — Agent 互操作标准

---

## 29.1 MCP

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("my-tools")

@server.list_tools()
async def list_tools():
    return [Tool(name="get_weather", description="获取天气",
                inputSchema={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})]

@server.call_tool()
async def call_tool(name, arguments):
    return [TextContent(type="text", text=f"{arguments['city']}：晴")]
```

---

## 29.2 A2A

```python
from a2a.server import A2AServer

server = A2AServer()

@server.task_handler()
async def handle_task(task):
    result = await process_task(task.message.parts[0].text)
    return Task(id=task.id, status=TaskState.COMPLETED)
```

---

## 29.3 MCP vs A2A

| 特性 | MCP | A2A |
|:---|:---|:---|
| 提出者 | Anthropic | Google |
| 目标 | Agent ↔ 工具 | Agent ↔ Agent |
| 状态管理 | 无状态 | 有状态 |

---

## 29.4 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| MCP | Agent-工具通信标准 |
| A2A | Agent-Agent 通信标准 |

> **下一章预告**
>
> 在第 30 章中，我们将学习复杂 Agent 工作流编排。

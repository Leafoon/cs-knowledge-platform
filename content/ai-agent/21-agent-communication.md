---
title: "第21章：Agent 通信协议与消息传递"
description: "深入 Agent 间通信机制：消息格式标准化、MCP 协议、A2A 协议、发布-订阅模式、事件总线与状态同步。"
date: "2026-06-11"
---

# 第21章：Agent 通信协议与消息传递

---

## 21.1 消息格式标准化

```python
from pydantic import BaseModel
from datetime import datetime

class AgentMessage(BaseModel):
    id: str
    sender: str
    receiver: str
    type: str
    content: Any
    metadata: dict = {}
    timestamp: datetime = datetime.now()
    correlation_id: str = ""

    def create_reply(self, content, sender):
        return AgentMessage(id=f"reply_{self.id}", sender=sender, receiver=self.sender,
                           type="response", content=content, correlation_id=self.id)
```

---

## 21.2 MCP（Model Context Protocol）

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("my-tools")

@server.list_tools()
async def list_tools():
    return [Tool(name="get_weather", description="获取城市天气",
                inputSchema={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})]

@server.call_tool()
async def call_tool(name, arguments):
    return [TextContent(type="text", text=f"{arguments['city']}：晴，25°C")]
```

---

## 21.3 A2A（Agent-to-Agent）

```python
from a2a.server import A2AServer

server = A2AServer()

@server.task_handler()
async def handle_task(task):
    result = await process_task(task.message.parts[0].text)
    return Task(id=task.id, status=TaskState.COMPLETED)
```

---

## 21.4 事件驱动架构

```python
class EventBus:
    def __init__(self): self.subscribers = {}
    def subscribe(self, event_type, handler): self.subscribers.setdefault(event_type, []).append(handler)
    async def publish(self, event_type, data):
        if event_type in self.subscribers:
            await asyncio.gather(*[h(data) for h in self.subscribers[event_type]])
```

---

## 21.5 MCP vs A2A

| 特性 | MCP | A2A |
|:---|:---|:---|
| 提出者 | Anthropic | Google |
| 目标 | Agent ↔ 工具 | Agent ↔ Agent |
| 状态管理 | 无状态 | 有状态 |

---

## 21.6 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| MCP | Agent-工具通信标准 |
| A2A | Agent-Agent 通信标准 |
| 事件总线 | 异步解耦通信 |

> **下一章预告**
>
> 在第 22 章中，我们将学习代码生成 Agent。

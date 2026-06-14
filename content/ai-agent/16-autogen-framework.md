---
title: "第16章：AutoGen 多智能体框架"
description: "深入 Microsoft AutoGen 框架：ConversableAgent、Agent 会话协议、代码执行器、GroupChat、嵌套对话与企业级部署。"
date: "2026-06-11"
---

# 第16章：AutoGen 多智能体框架

---

## 16.1 ConversableAgent

```python
from autogen import ConversableAgent

assistant = ConversableAgent(
    name="assistant",
    system_message="你是一个有帮助的 AI 助手。",
    llm_config={"config_list": [{"model": "gpt-4o"}]},
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
)
```

---

## 16.2 GroupChat

```python
from autogen import GroupChat, GroupChatManager

researcher = ConversableAgent(name="researcher", system_message="你是研究员。", llm_config=config)
writer = ConversableAgent(name="writer", system_message="你是撰稿人。", llm_config=config)
reviewer = ConversableAgent(name="reviewer", system_message="你是审稿人。", llm_config=config)

groupchat = GroupChat(agents=[researcher, writer, reviewer], messages=[], max_round=10)
manager = GroupChatManager(groupchat=groupchat, llm_config=config)
researcher.initiate_chat(manager, message="写一篇技术博客。")
```

---

## 16.3 代码执行

```python
from autogen.coding import DockerCommandLineCodeExecutor
import tempfile

with DockerCommandLineCodeExecutor(image="python:3.11-slim", timeout=60, work_dir=tempfile.mkdtemp()) as executor:
    code_agent = ConversableAgent(name="code_agent", system_message="你是代码专家。",
                                   llm_config=config, code_execution_config={"executor": executor})
```

---

## 16.4 工具集成

```python
from autogen import register_function
register_function(get_weather, caller=assistant, executor=critic, description="获取天气")
```

---

## 16.5 AutoGen vs LangGraph

| 特性 | AutoGen | LangGraph |
|:---|:---|:---|
| 核心模型 | 对话驱动 | 图驱动 |
| 代码执行 | 内置 Docker | 需自行集成 |
| 适用场景 | 多 Agent 对话 | 复杂流程编排 |

---

## 16.6 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| ConversableAgent | 核心基类 |
| GroupChat | 多 Agent 群聊 |
| 代码执行 | 内置 Docker |

> **下一章预告**
>
> 在第 17 章中，我们将学习 CrewAI。

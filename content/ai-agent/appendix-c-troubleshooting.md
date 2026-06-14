---
title: "Appendix C: 常见问题与调试指南"
description: "Agent 开发中的常见错误诊断：无限循环、工具调用失败、记忆溢出、Token 超限、幻觉问题与性能瓶颈排查。"
date: "2026-06-11"
---

# Appendix C: 常见问题与调试指南

---

## C.1 Agent 陷入无限循环

**解决方案**：
```python
agent = create_react_agent(model=llm, tools=tools)
result = agent.invoke(input, config={"recursion_limit": 15})

# 系统提示中明确终止条件
prompt = """当你得到足够信息时，直接给出最终答案。"""
```

## C.2 工具调用失败

| 原因 | 解决方案 |
|:---|:---|
| 参数格式错误 | 检查 JSON Schema |
| 工具描述不清晰 | 优化描述，添加示例 |
| 超时 | 增加 timeout |

## C.3 Token 超限

```python
from langchain.memory import ConversationSummaryBufferMemory
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=4000)

def truncate_output(output, max_len=2000):
    return output[:max_len] + "\n...(已截断)" if len(output) > max_len else output
```

## C.4 Agent 幻觉

```python
prompt = """只使用工具返回的信息回答问题。不要编造信息。"""
```

## C.5 性能瓶颈

| 瓶颈 | 优化 |
|:---|:---|
| LLM 延迟高 | 使用更快模型、压缩 Prompt |
| 工具执行慢 | 异步执行、缓存 |
| 检索质量差 | 优化分块、使用 Rerank |

## C.6 调试技巧

```python
import langchain
langchain.debug = True

import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"

for chunk in agent.stream(input, stream_mode="updates"):
    for node, update in chunk.items():
        print(f"[{node}]")
```

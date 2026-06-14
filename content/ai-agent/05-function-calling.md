---
title: "第5章：Function Calling — LLM 的工具调用机制"
description: "深入理解 LLM Function Calling 的底层机制，掌握 OpenAI、Anthropic 等主流 API 的工具定义、参数 schema、并行调用、流式调用与错误处理。"
date: "2026-06-11"
---

# 第5章：Function Calling — LLM 的工具调用机制

---

## 5.1 Function Calling 的本质

### 5.1.1 什么是 Function Calling？

Function Calling 是指 LLM 能够**根据用户意图，自主决定调用哪个函数、生成什么参数**的能力。

$$
\text{LLM}(context) \rightarrow \begin{cases} \text{text response} & \text{如果无需工具} \\ \{(f_1, args_1), \ldots, (f_n, args_n)\} & \text{如果需要工具} \end{cases}
$$

**完整生命周期**：

```
Step 1: 工具定义 → 开发者定义工具的名称、描述、参数 schema
Step 2: 工具绑定 → 将工具 schema 附加到 LLM 请求
Step 3: LLM 决策 → 分析用户输入，决定是否需要调用工具
Step 4: 参数提取 → 生成符合 JSON Schema 的参数
Step 5: 工具执行 → 应用程序调用实际的函数
Step 6: 结果返回 → 将工具执行结果返回给 LLM
Step 7: 最终响应 → LLM 综合工具结果生成回答
```

---

## 5.2 OpenAI Function Calling

### 5.2.1 工具定义与 Schema

```python
from openai import OpenAI
client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气信息。包括温度、湿度、天气状况等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，如 '北京'"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "温度单位"}
                },
                "required": ["city"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "北京天气怎么样？"}],
    tools=tools,
    tool_choice="auto"
)
message = response.choices[0].message
print(message.tool_calls)
```

### 5.2.2 tool_choice 参数详解

| 值 | 行为 | 适用场景 |
|:---|:---|:---|
| `"auto"` | LLM 自主决定 | 通用场景 |
| `"none"` | 强制不调用工具 | 只需文本回答 |
| `"required"` | 强制调用工具 | 必须使用工具 |
| `{"type":"function","function":{"name":"xxx"}}` | 强制调用指定工具 | 指定工具 |

### 5.2.3 并行 Function Calling

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "查一下北京和上海的天气"}],
    tools=tools,
    tool_choice="auto"
)
message = response.choices[0].message
# 可能返回多个并行工具调用
```

### 5.2.4 流式 Function Calling

```python
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "北京天气"}],
    tools=tools,
    stream=True
)
tool_calls = []
for chunk in stream:
    delta = chunk.choices[0].delta if chunk.choices else None
    if delta and delta.tool_calls:
        for tc in delta.tool_calls:
            if tc.index is not None:
                while len(tool_calls) <= tc.index:
                    tool_calls.append({"id": "", "name": "", "arguments": ""})
                if tc.id: tool_calls[tc.index]["id"] = tc.id
                if tc.function:
                    if tc.function.name: tool_calls[tc.index]["name"] = tc.function.name
                    if tc.function.arguments: tool_calls[tc.index]["arguments"] += tc.function.arguments
```

---

## 5.3 Anthropic Tool Use

### 5.3.1 工具定义格式

```python
import anthropic
client = anthropic.Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "获取城市天气",
        "input_schema": {  # 注意：Anthropic 使用 input_schema
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名称"}},
            "required": ["city"]
        }
    }
]

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "北京天气怎么样？"}]
)

for block in response.content:
    if block.type == "tool_use":
        print(f"工具调用: {block.name}({block.input})")
```

### 5.3.2 OpenAI vs Anthropic 对比

| 特性 | OpenAI | Anthropic |
|:---|:---|:---|
| 工具定义字段 | `parameters` | `input_schema` |
| 并行调用 | 原生支持 | 需要 `tool_choice: {"type": "any"}` |
| 工具结果格式 | `role: "tool"` | `role: "user"` + `tool_result` |
| 错误处理 | 返回错误字符串 | `is_error: true` 标记 |

---

## 5.4 Structured Output

```python
from pydantic import BaseModel

class WeatherInfo(BaseModel):
    city: str
    temperature: float
    condition: str
    humidity: int

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "北京天气"}],
    response_format=WeatherInfo
)
weather = response.choices[0].message.parsed
print(weather.city)
```

---

## 5.5 工具 Schema 设计最佳实践

| 检查项 | 说明 | 示例 |
|:---|:---|:---|
| 工具描述清晰 | 说明工具做什么、适用场景 | "搜索产品目录中的商品" |
| 参数类型准确 | 使用正确的 JSON Schema 类型 | `type: "integer"` |
| 枚举约束 | 有限选项用 enum | `enum: ["asc", "desc"]` |
| 必填标记 | 明确标记 required | `required: ["keyword"]` |
| 默认值说明 | 在描述中说明 | `description: "结果数量，默认10"` |
| 示例值 | 在描述中给出示例 | `"例如：'iPhone'"` |

---

## 5.6 错误处理

```python
def robust_tool_executor(tool_calls, tools_map):
    results = []
    for tc in tool_calls:
        func_name = tc.function.name
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError as e:
            results.append({"tool_call_id": tc.id, "content": f"参数解析错误：{str(e)}"})
            continue
        if func_name not in tools_map:
            results.append({"tool_call_id": tc.id, "content": f"工具 '{func_name}' 不存在"})
            continue
        try:
            result = tools_map[func_name](**args)
            results.append({"tool_call_id": tc.id, "content": str(result)})
        except Exception as e:
            results.append({"tool_call_id": tc.id, "content": f"执行错误：{str(e)}"})
    return results
```

---

## 5.7 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| Function Calling | LLM 输出结构化调用请求 |
| OpenAI API | `parameters` 格式，`tool_choice` 控制策略 |
| Anthropic API | `input_schema` 格式 |
| 并行调用 | 一次响应返回多个工具调用 |
| Structured Output | Pydantic 模型约束输出格式 |
| Schema 设计 | 描述清晰、类型准确、枚举约束 |

> **下一章预告**
>
> 在第 6 章中，我们将学习如何构建和管理 Agent 的工具集。

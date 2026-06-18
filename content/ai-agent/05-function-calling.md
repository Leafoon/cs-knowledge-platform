---
title: "第5章：Function Calling — LLM 的工具调用机制"
description: "深入理解 LLM Function Calling 的底层机制，掌握 OpenAI、Anthropic 等主流 API 的工具定义、参数 schema、并行调用、流式调用与错误处理。"
date: "2026-06-15"
---

 # 第5章：Function Calling — LLM 的工具调用机制

 Function Calling 是连接 LLM 推理能力与外部世界的桥梁。本章深入剖析 Function Calling 的底层机制，覆盖 OpenAI、Anthropic 等主流 API 的实现差异与最佳实践。理解 Function Calling 是构建 Agent 系统的基石——没有它，LLM 只能生成文本；有了它，LLM 才能真正"行动"。

 下面的交互式演示展示了 Function Calling 的完整流程：

 <div data-component="FunctionCallingSteps"></div>

 ## 5.1 Function Calling 的本质与定义

Function Calling 是指 LLM 能够**根据用户意图，自主决定调用哪个函数、生成什么参数**的能力。这不是 LLM 真正执行函数——而是 LLM 输出一个**结构化的调用请求**，由应用程序负责执行。

用数学语言表达：

$$
\text{LLM}(context) \rightarrow \begin{cases} \text{text response} & \text{如果无需工具} \\ \{(f_1, args_1), \ldots, (f_n, args_n)\} & \text{如果需要工具} \end{cases}
$$

其中 $f_i$ 是工具名称，$args_i$ 是符合 JSON Schema 的参数对象。

**完整生命周期**：

1. **工具定义**：开发者定义工具的名称、描述、参数 schema
2. **工具绑定**：将工具 schema 附加到 LLM 请求的系统提示中
3. **LLM 决策**：分析用户输入，决定是否需要调用工具
4. **参数提取**：生成符合 JSON Schema 的参数
5. **工具执行**：应用程序调用实际的函数
6. **结果返回**：将工具执行结果返回给 LLM
7. **最终响应**：LLM 综合工具结果生成最终回答

这个过程可以用一个状态机来表示：

$$
S = \{s_{\text{idle}}, s_{\text{calling}}, s_{\text{executing}}, s_{\text{reasoning}}\}
$$

$$
\delta: s_{\text{idle}} \xrightarrow{\text{need tool}} s_{\text{calling}} \xrightarrow{\text{generate args}} s_{\text{executing}} \xrightarrow{\text{result}} s_{\text{reasoning}} \xrightarrow{\text{answer or call again}} s_{\text{idle}}
$$

### 5.1.2 Function Calling vs Prompt Engineering

在 Function Calling 出现之前，开发者通过 Prompt Engineering 来让 LLM 输出工具调用格式（如 JSON），然后手动解析。这种方式存在严重问题：

| 对比维度 | Prompt Engineering | Function Calling |
|:---|:---|:---|
| **格式保证** | 不保证输出有效 JSON | 保证输出符合 schema |
| **参数验证** | 需要手动验证 | 自动验证参数类型 |
| **错误率** | 高（格式错误、幻觉参数） | 极低（模型专门训练过） |
| **多工具选择** | 容易混淆 | 准确选择 |
| **并行调用** | 几乎不可能 | 原生支持 |
| **安全性** | 容易 prompt injection | 有额外安全层 |

### 5.1.3 Function Calling 与 Agent 的关系

在 Agent 架构中，Function Calling 是最核心的组件之一。Agent 的感知-决策-行动循环中，"行动"阶段几乎完全依赖 Function Calling：

```
感知（Perception）→ 决策（Decision）→ 行动（Action）
                                      ↑
                              Function Calling 是这里的关键
```

没有 Function Calling，Agent 只能输出文本，无法真正与外部世界交互。有了 Function Calling，Agent 可以：
- 查询数据库获取实时信息
- 调用 API 执行操作
- 操作文件系统
- 控制浏览器
- 发送消息

### 5.1.4 Function Calling 的局限性

虽然 Function Calling 功能强大，但它也有明确的局限：

- **不能直接执行代码**：LLM 只生成调用请求，实际执行由应用程序负责
- **参数幻觉**：模型可能生成不存在的参数值（如虚构的日期）
- **上下文窗口限制**：工具定义占用 context window，太多工具会挤占有效空间
- **延迟**：每次工具调用都需要一次完整的 LLM 推理
 - **无法处理异步回调**：当前的 Function Calling 是同步的，不支持 WebSocket 等异步场景

工具调用是 Agent 与外部世界交互的核心环节。下面的交互式演示展示了工具调用的完整流程：

<div data-component="ToolCallFlowDemo"></div>

 ---

 ## 5.2 底层原理：从 Prompt 到 Function Call 的完整链路

### 5.2.1 训练阶段：如何让 LLM 学会调用工具？

Function Calling 能力来自**专门的训练数据**。在训练阶段，模型接触到大量三元组：

$$
\text{Training Data} = \{(\text{user\_msg}, \text{tools\_schema}, \text{correct\_call})\}
$$

训练过程大致分为三个阶段：

**阶段一：指令微调（Instruction Tuning）**

在大量自然语言指令-输出对上微调，让模型理解"当用户请求X时，应该调用函数Y"。例如：

- 用户："北京今天天气怎么样？" → 调用 `get_weather(city="北京")`
- 用户："帮我订明天下午3点的会议室" → 调用 `book_meeting(time="2026-06-16T15:00")`

**阶段二：工具使用微调（Tool Use Tuning）**

在包含工具定义和正确调用的对话数据上训练，让模型学会：
1. 从多个工具中选择正确的工具
2. 从用户输入中提取参数
3. 处理可选参数和默认值

**阶段三：RLHF 与对齐**

通过人类反馈强化学习，让模型学会：
1. 在不需要工具时不调用工具
2. 在工具调用失败时优雅地处理
3. 拒绝不安全的工具调用请求

### 5.2.2 推理阶段：模型如何生成 Function Call？

在推理阶段，Function Calling 的底层机制如下：

**第一步：工具 schema 注入**

工具定义被序列化为特定格式，注入到系统提示中。以 OpenAI 为例：

```
## Available Tools

### get_weather
Description: 获取指定城市的天气信息
Parameters:
  - city (string, required): 城市名称
  - unit (string, optional): 温度单位，可选 "celsius" 或 "fahrenheit"
```

**第二步：注意力计算**

当模型处理用户输入时，注意力机制会将用户意图与工具描述进行匹配。如果匹配度超过某个阈值，模型会生成特殊的 token 序列。

**第三步：结构化输出生成**

模型输出不是普通文本，而是符合预定义格式的结构化 JSON。这通过**受控解码（Constrained Decoding）**实现——在解码过程中，模型的词表概率分布被限制为只允许产生符合 JSON Schema 的 token。

用公式表示：

$$
P_{\text{constrained}}(t_i | t_1, \ldots, t_{i-1}) = \begin{cases} P(t_i | t_1, \ldots, t_{i-1}) & \text{如果 } t_i \text{ 符合 schema 约束} \\ 0 & \text{否则} \end{cases}
$$

**第四步：解析与验证**

应用程序解析模型输出，提取函数名和参数，进行类型验证和语义验证。

### 5.2.3 工具选择的决策过程

当有多个可用工具时，模型的选择过程可以建模为：

$$
P(\text{tool} = f_i | \text{context}) = \frac{e^{s(f_i, \text{context})}}{\sum_{j=1}^{n} e^{s(f_j, \text{context})}}
$$

其中 $s(f_i, \text{context})$ 是工具 $f_i$ 与当前上下文的相关性分数。这个分数由模型的注意力机制计算得出。

实际选择时，模型考虑以下因素：
1. **语义匹配度**：工具描述与用户意图的语义相似度
2. **参数可满足性**：用户输入是否包含足够的参数信息
3. **工具依赖关系**：某些工具的调用可能依赖其他工具的结果
4. **历史调用记录**：模型会参考对话历史中已调用的工具

### 5.2.4 参数提取的原理

参数提取是 Function Calling 中最具挑战性的环节。模型需要从自然语言中提取结构化参数，这个过程涉及：

**实体识别**：从文本中识别实体（如城市名、日期、数量）

**类型推断**：将识别的实体转换为正确的类型（如字符串、数字、布尔值）

**缺失参数处理**：对于可选参数，判断是否使用默认值

**歧义消解**：当多个参数可能匹配同一实体时，根据上下文选择最合适的

用一个例子说明：

```
用户输入："帮我查一下北京到上海明天的机票，经济舱"

模型需要提取：
  - origin: "北京"（出发地）
  - destination: "上海"（目的地）
  - date: "2026-06-16"（明天的日期，需要计算）
  - cabin_class: "economy"（经济舱映射）
```

### 5.2.5 JSON Schema 约束的数学原理

JSON Schema 定义了参数的结构和约束。在推理时，模型需要在这些约束下生成合法的 JSON。这个过程可以用上下文无关文法（CFG）来描述：

$$
G = (V, \Sigma, R, S)
$$

其中：
- $V$ 是非终结符集合（如 `object`, `array`, `string`, `number`）
- $\Sigma$ 是终结符集合（如 `{`, `}`, `"`, `:`）
- $R$ 是产生式规则，由 JSON Schema 定义
- $S$ 是起始符号（通常是 `object`）

受控解码确保模型生成的 token 序列始终是这个 CFG 的合法推导。

---

## 5.3 OpenAI Function Calling 实现详解

### 5.3.1 工具定义格式

OpenAI 使用 JSON Schema 格式定义工具。以下是完整的工具定义示例：

```python
import openai
import json
from datetime import datetime

# 初始化 OpenAI 客户端
client = openai.OpenAI(api_key="your-api-key")

# 定义工具列表
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气信息，包括温度、湿度、天气状况等",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 '北京'、'上海'、'New York'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位，默认为 celsius"
                    },
                    "date": {
                        "type": "string",
                        "format": "date",
                        "description": "查询日期，格式为 YYYY-MM-DD，默认为今天"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "搜索从出发地到目的地的航班信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "出发城市或机场代码"
                    },
                    "destination": {
                        "type": "string",
                        "description": "到达城市或机场代码"
                    },
                    "date": {
                        "type": "string",
                        "format": "date",
                        "description": "出发日期，格式为 YYYY-MM-DD"
                    },
                    "cabin_class": {
                        "type": "string",
                        "enum": ["economy", "business", "first"],
                        "description": "舱位等级"
                    },
                    "max_price": {
                        "type": "number",
                        "description": "最高价格（人民币）"
                    }
                },
                "required": ["origin", "destination", "date"]
            }
        }
    }
]
```

### 5.3.2 完整的对话流程

以下是一个完整的 OpenAI Function Calling 实现，包含错误处理和重试机制：

```python
import openai
import json
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field

# ============ 工具函数实现 ============

def get_weather(city: str, unit: str = "celsius", date: str = None) -> Dict[str, Any]:
    """
    获取指定城市的天气信息
    
    Args:
        city: 城市名称
        unit: 温度单位（celsius 或 fahrenheit）
        date: 查询日期（YYYY-MM-DD 格式）
    
    Returns:
        天气信息字典
    """
    # 模拟天气 API 调用（实际项目中替换为真实 API）
    # 真实场景中可能调用和风天气、OpenWeatherMap 等 API
    
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # 模拟数据
    weather_data = {
        "北京": {"temp": 28, "humidity": 65, "condition": "晴", "wind": "东北风3级"},
        "上海": {"temp": 30, "humidity": 75, "condition": "多云", "wind": "东南风4级"},
        "广州": {"temp": 32, "humidity": 80, "condition": "雷阵雨", "wind": "南风3级"},
        "深圳": {"temp": 31, "humidity": 78, "condition": "晴间多云", "wind": "西南风2级"},
    }
    
    # 获取天气数据
    data = weather_data.get(city, {"temp": 25, "humidity": 60, "condition": "晴", "wind": "微风"})
    
    # 单位转换
    if unit == "fahrenheit":
        data["temp"] = data["temp"] * 9/5 + 32
        data["unit"] = "°F"
    else:
        data["unit"] = "°C"
    
    return {
        "city": city,
        "date": date,
        "temperature": data["temp"],
        "unit": data["unit"],
        "humidity": data["humidity"],
        "condition": data["condition"],
        "wind": data["wind"]
    }


def search_flights(
    origin: str,
    destination: str,
    date: str,
    cabin_class: str = "economy",
    max_price: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    搜索航班信息
    
    Args:
        origin: 出发城市
        destination: 到达城市
        date: 出发日期
        cabin_class: 舱位等级
        max_price: 最高价格
    
    Returns:
        航班信息列表
    """
    # 模拟航班数据
    flights = [
        {
            "flight_no": "CA1234",
            "airline": "中国国航",
            "origin": origin,
            "destination": destination,
            "departure": "08:00",
            "arrival": "10:30",
            "price": 850.0,
            "cabin_class": cabin_class
        },
        {
            "flight_no": "MU5678",
            "airline": "东方航空",
            "origin": origin,
            "destination": destination,
            "departure": "14:30",
            "arrival": "17:00",
            "price": 920.0,
            "cabin_class": cabin_class
        },
        {
            "flight_no": "CZ9012",
            "airline": "南方航空",
            "origin": origin,
            "destination": destination,
            "departure": "19:15",
            "arrival": "21:45",
            "price": 780.0,
            "cabin_class": cabin_class
        }
    ]
    
    # 价格过滤
    if max_price is not None:
        flights = [f for f in flights if f["price"] <= max_price]
    
    return flights


def book_flight(
    flight_no: str,
    passenger_name: str,
    id_number: str,
    phone: str,
    seat_preference: str = "window"
) -> Dict[str, Any]:
    """
    预订航班
    
    Args:
        flight_no: 航班号
        passenger_name: 乘客姓名
        id_number: 身份证号
        phone: 手机号码
        seat_preference: 座位偏好（window/aisle/middle）
    
    Returns:
        预订结果
    """
    # 模拟预订逻辑
    booking_id = f"BK{int(time.time()) % 1000000:06d}"
    
    return {
        "booking_id": booking_id,
        "flight_no": flight_no,
        "passenger_name": passenger_name,
        "status": "confirmed",
        "seat": f"{seat_preference[0].upper()}{12}",
        "message": f"预订成功！您的预订号为 {booking_id}"
    }


# ============ 工具注册表 ============

# 将函数名映射到实际的函数对象
# 这样可以通过字符串名称动态调用函数
TOOL_REGISTRY: Dict[str, Callable] = {
    "get_weather": get_weather,
    "search_flights": search_flights,
    "book_flight": book_flight,
}

# ============ 工具 Schema 定义 ============

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位，默认 celsius"
                    },
                    "date": {
                        "type": "string",
                        "format": "date",
                        "description": "查询日期 YYYY-MM-DD"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "搜索航班信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "出发城市"
                    },
                    "destination": {
                        "type": "string",
                        "description": "到达城市"
                    },
                    "date": {
                        "type": "string",
                        "format": "date",
                        "description": "出发日期 YYYY-MM-DD"
                    },
                    "cabin_class": {
                        "type": "string",
                        "enum": ["economy", "business", "first"],
                        "description": "舱位等级"
                    },
                    "max_price": {
                        "type": "number",
                        "description": "最高价格"
                    }
                },
                "required": ["origin", "destination", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "预订航班",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_no": {
                        "type": "string",
                        "description": "航班号"
                    },
                    "passenger_name": {
                        "type": "string",
                        "description": "乘客姓名"
                    },
                    "id_number": {
                        "type": "string",
                        "description": "身份证号"
                    },
                    "phone": {
                        "type": "string",
                        "description": "手机号码"
                    },
                    "seat_preference": {
                        "type": "string",
                        "enum": ["window", "aisle", "middle"],
                        "description": "座位偏好"
                    }
                },
                "required": ["flight_no", "passenger_name", "id_number", "phone"]
            }
        }
    }
]


# ============ Function Calling Agent ============

class FunctionCallingAgent:
    """
    基于 OpenAI Function Calling 的对话代理
    
    支持：
    - 多轮对话
    - 工具调用
    - 并行调用
    - 错误处理与重试
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        tools: List[Dict] = None,
        max_retries: int = 3,
        max_tool_rounds: int = 10
    ):
        """
        初始化代理
        
        Args:
            model: 使用的模型名称
            tools: 工具 schema 列表
            max_retries: 最大重试次数
            max_tool_rounds: 最大工具调用轮次
        """
        self.client = openai.OpenAI()
        self.model = model
        self.tools = tools or TOOL_SCHEMAS
        self.max_retries = max_retries
        self.max_tool_rounds = max_tool_rounds
        self.conversation_history: List[Dict[str, Any]] = []
    
    def _execute_tool(self, tool_call: Dict[str, Any]) -> str:
        """
        执行单个工具调用
        
        Args:
            tool_call: 工具调用对象，包含 id, function.name, function.arguments
        
        Returns:
            工具执行结果的 JSON 字符串
        """
        function_name = tool_call["function"]["name"]
        arguments_str = tool_call["function"]["arguments"]
        
        try:
            # 解析参数
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError as e:
            # JSON 解析失败，返回错误信息
            error_msg = f"参数 JSON 解析失败: {str(e)}"
            return json.dumps({"error": error_msg}, ensure_ascii=False)
        
        # 查找并执行对应的函数
        if function_name not in TOOL_REGISTRY:
            error_msg = f"未知工具: {function_name}"
            return json.dumps({"error": error_msg}, ensure_ascii=False)
        
        tool_func = TOOL_REGISTRY[function_name]
        
        try:
            # 调用实际的函数
            result = tool_func(**arguments)
            # 将结果转换为 JSON 字符串
            return json.dumps(result, ensure_ascii=False)
        except TypeError as e:
            # 参数类型错误
            error_msg = f"参数类型错误: {str(e)}"
            return json.dumps({"error": error_msg}, ensure_ascii=False)
        except Exception as e:
            # 其他执行错误
            error_msg = f"工具执行失败: {str(e)}"
            return json.dumps({"error": error_msg}, ensure_ascii=False)
    
    def _call_llm(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        调用 LLM API
        
        Args:
            messages: 消息列表
        
        Returns:
            API 响应对象
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto"  # 让模型自动决定是否调用工具
                )
                return response
            except openai.APIError as e:
                if attempt < self.max_retries - 1:
                    # 指数退避重试
                    wait_time = 2 ** attempt
                    print(f"API 调用失败，{wait_time}秒后重试: {str(e)}")
                    time.sleep(wait_time)
                else:
                    raise
    
    def _process_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        处理所有工具调用（支持并行调用）
        
        Args:
            tool_calls: 工具调用列表
        
        Returns:
            工具结果消息列表
        """
        results = []
        
        for tool_call in tool_calls:
            # 执行工具调用
            result = self._execute_tool(tool_call)
            
            # 构造工具结果消息
            tool_result_message = {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result
            }
            results.append(tool_result_message)
            
            # 记录工具调用日志
            func_name = tool_call["function"]["name"]
            func_args = tool_call["function"]["arguments"]
            print(f"[工具调用] {func_name}({func_args})")
            print(f"[工具结果] {result[:200]}...")
        
        return results
    
    def chat(self, user_message: str) -> str:
        """
        与用户进行对话
        
        Args:
            user_message: 用户输入的消息
        
        Returns:
            模型的回复文本
        """
        # 将用户消息添加到对话历史
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # 工具调用轮次计数
        tool_rounds = 0
        
        while tool_rounds < self.max_tool_rounds:
            # 调用 LLM
            response = self._call_llm(self.conversation_history)
            message = response.choices[0].message
            
            # 检查是否有工具调用
            if message.tool_calls:
                tool_rounds += 1
                print(f"\n--- 工具调用轮次 {tool_rounds} ---")
                
                # 将助手消息（包含工具调用）添加到历史
                self.conversation_history.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]
                })
                
                # 处理所有工具调用
                tool_results = self._process_tool_calls([
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ])
                
                # 将工具结果添加到对话历史
                self.conversation_history.extend(tool_results)
            else:
                # 没有工具调用，返回最终回复
                final_content = message.content or ""
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_content
                })
                return final_content
        
        # 超过最大工具调用轮次
        return "抱歉，处理过程中达到了最大工具调用次数限制。"


# ============ 使用示例 ============

def main():
    """主函数：演示完整的 Function Calling 流程"""
    
    # 创建代理
    agent = FunctionCallingAgent(
        model="gpt-4",
        tools=TOOL_SCHEMAS,
        max_retries=3,
        max_tool_rounds=10
    )
    
    # 模拟多轮对话
    print("=" * 60)
    print("Function Calling Agent 启动")
    print("=" * 60)
    
    # 第一轮：查询天气
    print("\n[用户] 北京今天天气怎么样？")
    response = agent.chat("北京今天天气怎么样？")
    print(f"[助手] {response}\n")
    
    # 第二轮：查询航班
    print("\n[用户] 帮我查一下北京到上海明天的航班")
    response = agent.chat("帮我查一下北京到上海明天的航班")
    print(f"[助手] {response}\n")
    
    # 第三轮：预订航班
    print("\n[用户] 帮我预订国航的航班，我叫张三，身份证号110101199001011234，手机13800138000")
    response = agent.chat(
        "帮我预订国航的航班，我叫张三，"
        "身份证号110101199001011234，手机13800138000，"
        "要靠窗的座位"
    )
    print(f"[助手] {response}\n")


if __name__ == "__main__":
    main()
```

### 5.3.3 tool_choice 参数详解

OpenAI API 中的 `tool_choice` 参数控制模型如何选择工具：

```python
# 不同的 tool_choice 策略

# 1. auto：模型自动决定是否调用工具（默认）
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

# 2. none：强制模型不调用任何工具，只输出文本
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools,
    tool_choice="none"
)

# 3. required：强制模型必须调用至少一个工具
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools,
    tool_choice="required"
)

 # 4. 指定工具：强制模型调用特定工具
 response = client.chat.completions.create(
     model="gpt-4",
     messages=messages,
     tools=tools,
     tool_choice={"type": "function", "function": {"name": "get_weather"}}
 )
 ```

工具选择过程需要综合考虑用户意图、工具描述和历史上下文信息。下面的交互式演示展示了工具选择的完整决策过程：

<div data-component="ToolSelectionDemoV6"></div>

 ### 5.3.4 多轮对话中的上下文管理

在多轮对话中，需要正确管理对话历史，特别是工具调用和结果的记录：

```python
def build_messages_with_history(
    history: List[Dict],
    current_query: str,
    system_prompt: str = None
) -> List[Dict[str, Any]]:
    """
    构建包含历史对话的消息列表
    
    在多轮 Function Calling 中，需要保留：
    1. 所有用户消息
    2. 所有助手消息（包括工具调用）
    3. 所有工具结果
    
    Args:
        history: 历史对话记录
        current_query: 当前用户输入
        system_prompt: 系统提示
    
    Returns:
        消息列表
    """
    messages = []
    
    # 添加系统提示
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    # 添加历史对话
    messages.extend(history)
    
    # 添加当前用户消息
    messages.append({"role": "user", "content": current_query})
    
    return messages


def compress_history_if_needed(
    messages: List[Dict],
    max_tokens: int = 8000
) -> List[Dict]:
    """
    当对话历史过长时进行压缩
    
    策略：
    1. 保留系统提示
    2. 保留最近 N 轮对话
    3. 对早期对话进行摘要
    
    Args:
        messages: 消息列表
        max_tokens: 最大 token 数限制
    
    Returns:
        压缩后的消息列表
    """
    if len(messages) <= 4:
        return messages
    
    # 保留系统提示
    system_msg = messages[0] if messages[0]["role"] == "system" else None
    rest = messages[1:] if system_msg else messages
    
    # 简单策略：保留最近 10 条消息
    # 实际项目中可以使用更复杂的摘要策略
    recent = rest[-10:]
    
    if system_msg:
        return [system_msg] + recent
    return recent
```

---

## 5.4 Anthropic Tool Use 实现详解

### 5.4.1 Anthropic 的 Tool Use API

Anthropic（Claude）的工具使用 API 与 OpenAI 有显著不同。Claude 使用 `tools` 参数，并且工具定义格式略有差异：

```python
import anthropic
import json
from typing import Dict, List, Any, Optional

# 初始化 Anthropic 客户端
client = anthropic.Anthropic(api_key="your-api-key")

# Anthropic 的工具定义格式
tools = [
    {
        "name": "get_stock_price",
        "description": "获取指定股票的实时价格。当用户询问股票价格或投资建议时使用此工具。",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "股票代码，如 'AAPL', 'GOOGL', 'MSFT'"
                },
                "currency": {
                    "type": "string",
                    "enum": ["USD", "CNY", "EUR", "GBP", "JPY"],
                    "description": "报价货币，默认 USD"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "calculate_portfolio_value",
        "description": "计算投资组合的总价值。需要提供股票代码和持有数量的映射。",
        "input_schema": {
            "type": "object",
            "properties": {
                "holdings": {
                    "type": "object",
                    "description": "持仓信息，键为股票代码，值为持有数量",
                    "additionalProperties": {
                        "type": "number"
                    }
                },
                "currency": {
                    "type": "string",
                    "enum": ["USD", "CNY"],
                    "description": "结算货币"
                }
            },
            "required": ["holdings"]
        }
    },
    {
        "name": "place_order",
        "description": "下单买入或卖出股票。注意：此操作不可逆，请确认用户意图。",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "股票代码"
                },
                "action": {
                    "type": "string",
                    "enum": ["buy", "sell"],
                    "description": "买入或卖出"
                },
                "quantity": {
                    "type": "integer",
                    "description": "交易数量（股）"
                },
                "order_type": {
                    "type": "string",
                    "enum": ["market", "limit", "stop"],
                    "description": "订单类型"
                },
                "limit_price": {
                    "type": "number",
                    "description": "限价单价格（仅限价单需要）"
                }
            },
            "required": ["ticker", "action", "quantity", "order_type"]
        }
    }
]
```

### 5.4.2 Anthropic 的消息处理流程

Anthropic 的消息处理与 OpenAI 有明显差异。Claude 使用不同的消息格式和工具调用响应格式：

```python
class AnthropicToolAgent:
    """
    基于 Anthropic Claude Tool Use 的对话代理
    
    特点：
    1. 工具定义使用 input_schema 而非 parameters
    2. 工具调用通过 content block 返回
    3. 需要将工具结果作为新的用户消息返回
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        tools: List[Dict] = None,
        max_retries: int = 3,
        max_tool_rounds: int = 10,
        temperature: float = 0.7
    ):
        """
        初始化 Anthropic 工具代理
        
        Args:
            model: 模型名称
            tools: 工具定义列表
            max_retries: 最大重试次数
            max_tool_rounds: 最大工具调用轮次
            temperature: 生成温度
        """
        self.client = anthropic.Anthropic()
        self.model = model
        self.tools = tools or []
        self.max_retries = max_retries
        self.max_tool_rounds = max_tool_rounds
        self.temperature = temperature
        self.conversation_history: List[Dict[str, Any]] = []
    
    def _execute_tool(self, name: str, input_data: Dict[str, Any]) -> str:
        """
        执行工具调用
        
        Anthropic 要求工具结果以字符串形式返回
        
        Args:
            name: 工具名称
            input_data: 工具输入参数
        
        Returns:
            工具执行结果的 JSON 字符串
        """
        # 工具函数映射
        tool_functions = {
            "get_stock_price": self._get_stock_price,
            "calculate_portfolio_value": self._calculate_portfolio_value,
            "place_order": self._place_order,
        }
        
        if name not in tool_functions:
            return json.dumps({
                "error": f"未知工具: {name}"
            }, ensure_ascii=False)
        
        try:
            result = tool_functions[name](**input_data)
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({
                "error": f"工具执行失败: {str(e)}"
            }, ensure_ascii=False)
    
    def _get_stock_price(self, ticker: str, currency: str = "USD") -> Dict[str, Any]:
        """模拟获取股票价格"""
        # 实际项目中调用真实股票 API
        prices = {
            "AAPL": {"price": 189.50, "change": 2.3},
            "GOOGL": {"price": 175.20, "change": -1.5},
            "MSFT": {"price": 425.80, "change": 5.1},
            "TSLA": {"price": 245.60, "change": -8.2},
        }
        
        data = prices.get(ticker, {"price": 100.0, "change": 0.0})
        
        # 汇率转换（简化版）
        exchange_rates = {"USD": 1.0, "CNY": 7.2, "EUR": 0.92, "GBP": 0.79, "JPY": 149.5}
        rate = exchange_rates.get(currency, 1.0)
        
        return {
            "ticker": ticker,
            "price": round(data["price"] * rate, 2),
            "change_percent": data["change"],
            "currency": currency,
            "timestamp": "2026-06-15 10:30:00"
        }
    
    def _calculate_portfolio_value(
        self,
        holdings: Dict[str, float],
        currency: str = "USD"
    ) -> Dict[str, Any]:
        """计算投资组合总价值"""
        total_value = 0.0
        details = []
        
        for ticker, quantity in holdings.items():
            price_data = self._get_stock_price(ticker, currency)
            position_value = price_data["price"] * quantity
            total_value += position_value
            details.append({
                "ticker": ticker,
                "quantity": quantity,
                "price": price_data["price"],
                "value": round(position_value, 2)
            })
        
        return {
            "total_value": round(total_value, 2),
            "currency": currency,
            "positions": details,
            "position_count": len(holdings)
        }
    
    def _place_order(
        self,
        ticker: str,
        action: str,
        quantity: int,
        order_type: str,
        limit_price: float = None
    ) -> Dict[str, Any]:
        """模拟下单"""
        order_id = f"ORD{int(time.time()) % 10000000:07d}"
        
        return {
            "order_id": order_id,
            "ticker": ticker,
            "action": action,
            "quantity": quantity,
            "order_type": order_type,
            "limit_price": limit_price,
            "status": "submitted",
            "message": f"订单 {order_id} 已提交"
        }
    
    def _call_llm(self, messages: List[Dict]) -> Dict:
        """
        调用 Claude API
        
        Anthropic API 与 OpenAI 的主要差异：
        1. 使用 system 参数而非 system 消息
        2. 工具调用通过 content blocks 返回
        3. 需要将工具结果作为 user 消息返回
        
        Args:
            messages: 消息列表
        
        Returns:
            API 响应
        """
        # 提取系统提示
        system_prompt = None
        filtered_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                filtered_messages.append(msg)
        
        for attempt in range(self.max_retries):
            try:
                kwargs = {
                    "model": self.model,
                    "max_tokens": 4096,
                    "messages": filtered_messages,
                    "tools": self.tools,
                }
                
                if system_prompt:
                    kwargs["system"] = system_prompt
                
                response = self.client.messages.create(**kwargs)
                return response
            except anthropic.APIError as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"API 调用失败，{wait_time}秒后重试: {str(e)}")
                    time.sleep(wait_time)
                else:
                    raise
    
    def _extract_tool_calls(self, response) -> List[Dict[str, Any]]:
        """
        从 Claude 响应中提取工具调用
        
        Claude 的工具调用以 content blocks 形式返回：
        [
            {"type": "text", "text": "..."},
            {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
        ]
        
        Args:
            response: Claude API 响应
        
        Returns:
            工具调用列表
        """
        tool_calls = []
        
        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })
        
        return tool_calls
    
    def _build_tool_result_messages(
        self,
        tool_calls: List[Dict],
        results: List[str]
    ) -> List[Dict[str, Any]]:
        """
        构建工具结果消息
        
        Anthropic 要求将工具结果放在 user 消息中，
        使用 tool_result content block 格式
        
        Args:
            tool_calls: 工具调用列表
            results: 工具执行结果列表
        
        Returns:
            工具结果消息列表
        """
        messages = []
        
        for tc, result in zip(tool_calls, results):
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": result
                    }
                ]
            })
        
        return messages
    
    def chat(self, user_message: str) -> str:
        """
        与 Claude 进行对话
        
        Args:
            user_message: 用户输入
        
        Returns:
            Claude 的回复
        """
        # 添加用户消息
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        tool_rounds = 0
        
        while tool_rounds < self.max_tool_rounds:
            # 调用 Claude
            response = self._call_llm(self.conversation_history)
            
            # 提取工具调用
            tool_calls = self._extract_tool_calls(response)
            
            if not tool_calls:
                # 没有工具调用，提取最终回复
                final_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        final_text += block.text
                
                # 将助手回复添加到历史
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content
                })
                
                return final_text
            
            # 有工具调用
            tool_rounds += 1
            print(f"\n--- 工具调用轮次 {tool_rounds} ---")
            
            # 将助手回复（包含工具调用）添加到历史
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content
            })
            
            # 执行所有工具调用
            results = []
            for tc in tool_calls:
                result = self._execute_tool(tc["name"], tc["input"])
                results.append(result)
                print(f"[工具调用] {tc['name']}({json.dumps(tc['input'], ensure_ascii=False)})")
                print(f"[工具结果] {result[:200]}...")
            
            # 将工具结果添加到历史
            tool_result_messages = self._build_tool_result_messages(
                tool_calls, results
            )
            self.conversation_history.extend(tool_result_messages)
        
        return "抱歉，处理过程中达到了最大工具调用次数限制。"


# ============ 使用示例 ============

def demo_anthropic():
    """演示 Anthropic Tool Use"""
    
    agent = AnthropicToolAgent(
        model="claude-sonnet-4-20250514",
        tools=tools,
        temperature=0.7
    )
    
    print("=" * 60)
    print("Anthropic Tool Use Agent 启动")
    print("=" * 60)
    
    # 查询股票价格
    print("\n[用户] AAPL 现在什么价格？")
    response = agent.chat("AAPL 现在什么价格？")
    print(f"[助手] {response}\n")
    
    # 计算投资组合
    print("\n[用户] 我持有100股AAPL、50股GOOGL、30股MSFT，总共值多少钱？")
    response = agent.chat(
        "我持有100股AAPL、50股GOOGL、30股MSFT，"
        "用美元计算总共值多少钱？"
    )
    print(f"[助手] {response}\n")


if __name__ == "__main__":
    demo_anthropic()
```

### 5.4.3 Anthropic 与 OpenAI 的关键差异

理解两个 API 之间的差异对于跨平台开发至关重要：

| 差异点 | OpenAI | Anthropic |
|:---|:---|:---|
| **系统提示** | 作为 messages 中的 system 角色 | 使用独立的 system 参数 |
| **工具定义键名** | `parameters` | `input_schema` |
| **工具调用返回** | 在 message 的 `tool_calls` 字段 | 在 content blocks 中 |
| **工具结果格式** | role: tool + tool_call_id | role: user + tool_result block |
| **并行调用** | 原生支持 | 原生支持 |
| **流式输出** | SSE 格式 | SSE 格式（略有不同） |
| **JSON Schema 支持** | 部分支持 | 完整支持 |
| **Token 计算** | 工具定义计入 context | 工具定义计入 context |

---

## 5.5 OpenAI vs Anthropic 对比分析

### 5.5.1 功能对比表

| 功能特性 | OpenAI (GPT-4) | Anthropic (Claude) |
|:---|:---|:---|
| **基础工具调用** | ✅ 支持 | ✅ 支持 |
| **并行工具调用** | ✅ 原生支持 | ✅ 原生支持 |
| **流式工具调用** | ✅ 支持 | ✅ 支持 |
| **强制工具调用** | ✅ tool_choice: required | ❌ 不支持 |
| **指定工具调用** | ✅ tool_choice: function | ❌ 不支持 |
| **工具调用嵌套** | ⚠️ 部分支持 | ✅ 支持 |
| **JSON Schema 完整性** | ⚠️ 部分特性不支持 | ✅ 完整支持 |
| **最大工具数量** | 128 | 128 |
| **工具定义 Token 限制** | 取决于模型 | 取决于模型 |

### 5.5.2 性能对比

在实际应用中，两个 API 在以下方面表现不同：

**延迟方面**：
- OpenAI：首次响应通常 1-3 秒，工具调用后 2-5 秒
- Anthropic：首次响应通常 1-4 秒，工具调用后 2-6 秒

**准确性方面**：
- OpenAI GPT-4：工具选择准确率约 95%，参数提取准确率约 90%
- Anthropic Claude：工具选择准确率约 94%，参数提取准确率约 92%

**上下文利用**：
- OpenAI：更倾向于在不需要工具时直接回答
- Anthropic：更倾向于使用工具获取信息后再回答

### 5.5.3 选择建议

根据不同的应用场景，选择不同的 API：

- **需要精确控制工具调用行为**：选择 OpenAI（支持 tool_choice）
- **需要复杂参数结构**：选择 Anthropic（更完整的 JSON Schema 支持）
- **需要流式输出**：两者都支持，但实现方式不同
- **成本敏感**：需要根据实际使用量评估，两者定价策略不同
- **安全性要求高**：Anthropic 的 Constitutional AI 可能更适合

---

## 5.6 并行调用（Parallel Function Calling）

### 5.6.1 什么是并行调用？

并行调用是指 LLM 在一次响应中同时请求调用多个工具的能力。这大幅减少了多工具场景下的延迟。

在非并行调用中，如果用户问"北京和上海的天气分别怎么样？"，模型需要：
1. 第一轮：调用 `get_weather(city="北京")` → 等待结果
2. 第二轮：调用 `get_weather(city="上海")` → 等待结果
3. 第三轮：综合两个结果生成回答

而在并行调用中，模型一次性输出两个工具调用，应用程序可以同时执行：
1. 第一轮：调用 `[get_weather(city="北京"), get_weather(city="上海")]`
2. 第二轮：同时执行两个调用，综合结果生成回答

延迟从 $O(2 \cdot t_{\text{call}} + t_{\text{reason}})$ 降低到 $O(\max(t_{\text{call1}}, t_{\text{call2}}) + t_{\text{reason}})$。

### 5.6.2 并行调用的实现

```python
import openai
import json
import asyncio
import aiohttp
from typing import Dict, List, Any, Callable
from concurrent.futures import ThreadPoolExecutor
import time

# ============ 并行工具执行器 ============

class ParallelToolExecutor:
    """
    支持并行执行的工具调用器
    
    特点：
    1. 同时执行多个工具调用
    2. 支持同步和异步工具
    3. 超时控制
    4. 错误隔离
    """
    
    def __init__(
        self,
        tool_registry: Dict[str, Callable],
        max_workers: int = 10,
        timeout: float = 30.0
    ):
        """
        初始化并行执行器
        
        Args:
            tool_registry: 工具函数注册表
            max_workers: 最大并行工作线程数
            timeout: 单个工具调用的超时时间（秒）
        """
        self.tool_registry = tool_registry
        self.max_workers = max_workers
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def _execute_single_tool(
        self,
        tool_call: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        执行单个工具调用（带超时控制）
        
        Args:
            tool_call: 工具调用信息
        
        Returns:
            执行结果
        """
        function_name = tool_call["function"]["name"]
        arguments_str = tool_call["function"]["arguments"]
        tool_call_id = tool_call["id"]
        
        try:
            # 解析参数
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError as e:
            return {
                "tool_call_id": tool_call_id,
                "result": json.dumps({
                    "error": f"参数解析失败: {str(e)}"
                }, ensure_ascii=False),
                "success": False
            }
        
        # 查找工具函数
        if function_name not in self.tool_registry:
            return {
                "tool_call_id": tool_call_id,
                "result": json.dumps({
                    "error": f"未知工具: {function_name}"
                }, ensure_ascii=False),
                "success": False
            }
        
        tool_func = self.tool_registry[function_name]
        
        try:
            # 执行工具调用
            start_time = time.time()
            result = tool_func(**arguments)
            elapsed = time.time() - start_time
            
            return {
                "tool_call_id": tool_call_id,
                "result": json.dumps(result, ensure_ascii=False),
                "success": True,
                "elapsed_seconds": round(elapsed, 3)
            }
        except Exception as e:
            return {
                "tool_call_id": tool_call_id,
                "result": json.dumps({
                    "error": f"执行失败: {str(e)}"
                }, ensure_ascii=False),
                "success": False
            }
    
    def execute_parallel(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        并行执行多个工具调用
        
        使用线程池实现真正的并行执行
        
        Args:
            tool_calls: 工具调用列表
        
        Returns:
            执行结果列表
        """
        if not tool_calls:
            return []
        
        # 使用线程池并行执行
        futures = []
        for tool_call in tool_calls:
            future = self.executor.submit(
                self._execute_single_tool,
                tool_call
            )
            futures.append(future)
        
        # 收集结果
        results = []
        for future in futures:
            try:
                result = future.result(timeout=self.timeout)
                results.append(result)
            except Exception as e:
                results.append({
                    "tool_call_id": "unknown",
                    "result": json.dumps({
                        "error": f"执行超时或异常: {str(e)}"
                    }, ensure_ascii=False),
                    "success": False
                })
        
        return results
    
    def shutdown(self):
        """关闭线程池"""
        self.executor.shutdown(wait=True)


# ============ 异步并行执行器 ============

class AsyncParallelToolExecutor:
    """
    基于 asyncio 的异步并行工具执行器
    
    适用于 I/O 密集型工具（如 API 调用）
    """
    
    def __init__(
        self,
        tool_registry: Dict[str, Callable],
        timeout: float = 30.0
    ):
        self.tool_registry = tool_registry
        self.timeout = timeout
    
    async def _execute_single_tool_async(
        self,
        tool_call: Dict[str, Any]
    ) -> Dict[str, Any]:
        """异步执行单个工具调用"""
        function_name = tool_call["function"]["name"]
        arguments_str = tool_call["function"]["arguments"]
        tool_call_id = tool_call["id"]
        
        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError as e:
            return {
                "tool_call_id": tool_call_id,
                "result": json.dumps({
                    "error": f"参数解析失败: {str(e)}"
                }, ensure_ascii=False),
                "success": False
            }
        
        if function_name not in self.tool_registry:
            return {
                "tool_call_id": tool_call_id,
                "result": json.dumps({
                    "error": f"未知工具: {function_name}"
                }, ensure_ascii=False),
                "success": False
            }
        
        tool_func = self.tool_registry[function_name]
        
        try:
            start_time = time.time()
            
            # 如果是协程函数，使用 await
            if asyncio.iscoroutinefunction(tool_func):
                result = await asyncio.wait_for(
                    tool_func(**arguments),
                    timeout=self.timeout
                )
            else:
                # 否则在线程池中执行
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: tool_func(**arguments)
                    ),
                    timeout=self.timeout
                )
            
            elapsed = time.time() - start_time
            
            return {
                "tool_call_id": tool_call_id,
                "result": json.dumps(result, ensure_ascii=False),
                "success": True,
                "elapsed_seconds": round(elapsed, 3)
            }
        except asyncio.TimeoutError:
            return {
                "tool_call_id": tool_call_id,
                "result": json.dumps({
                    "error": f"工具执行超时（{self.timeout}秒）"
                }, ensure_ascii=False),
                "success": False
            }
        except Exception as e:
            return {
                "tool_call_id": tool_call_id,
                "result": json.dumps({
                    "error": f"执行失败: {str(e)}"
                }, ensure_ascii=False),
                "success": False
            }
    
    async def execute_parallel(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        异步并行执行多个工具调用
        
        Args:
            tool_calls: 工具调用列表
        
        Returns:
            执行结果列表
        """
        if not tool_calls:
            return []
        
        tasks = [
            self._execute_single_tool_async(tc)
            for tc in tool_calls
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常情况
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "tool_call_id": tool_calls[i].get("id", "unknown"),
                    "result": json.dumps({
                        "error": f"异常: {str(result)}"
                    }, ensure_ascii=False),
                    "success": False
                })
            else:
                processed_results.append(result)
        
        return processed_results


# ============ 使用示例 ============

async def demo_parallel():
    """演示并行工具执行"""
    
    # 定义工具
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取城市天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]
    
    # 工具函数
    def get_weather(city: str) -> Dict:
        # 模拟网络延迟
        time.sleep(1)
        return {"city": city, "temp": 25, "condition": "晴"}
    
    registry = {"get_weather": get_weather}
    
    # 模拟 LLM 返回的多个工具调用
    tool_calls = [
        {"id": "call_1", "function": {"name": "get_weather", "arguments": json.dumps({"city": "北京"})}},
        {"id": "call_2", "function": {"name": "get_weather", "arguments": json.dumps({"city": "上海"})}},
        {"id": "call_3", "function": {"name": "get_weather", "arguments": json.dumps({"city": "广州"})}},
        {"id": "call_4", "function": {"name": "get_weather", "arguments": json.dumps({"city": "深圳"})}},
    ]
    
    # 串行执行（基准）
    executor = ParallelToolExecutor(registry)
    start = time.time()
    results_serial = []
    for tc in tool_calls:
        results_serial.append(executor._execute_single_tool(tc))
    serial_time = time.time() - start
    print(f"串行执行时间: {serial_time:.2f}秒")
    
    # 并行执行
    start = time.time()
    results_parallel = executor.execute_parallel(tool_calls)
    parallel_time = time.time() - start
    print(f"并行执行时间: {parallel_time:.2f}秒")
    print(f"加速比: {serial_time/parallel_time:.2f}x")
    
    executor.shutdown()


# 运行示例
# asyncio.run(demo_parallel())
```

### 5.6.3 并行调用的注意事项

并行调用虽然能提升性能，但也需要注意以下问题：

1. **工具之间可能有依赖关系**：如果工具 B 的输入依赖工具 A 的输出，就不能并行执行
2. **资源竞争**：多个工具同时调用同一资源可能导致冲突
3. **结果顺序**：并行执行的结果顺序可能与请求顺序不同，需要通过 `tool_call_id` 关联
4. **错误隔离**：一个工具的失败不应该影响其他工具的执行

---

## 5.7 流式调用（Streaming Function Calling）

### 5.7.1 流式调用的概念

流式调用是指 LLM 在生成工具调用请求时，以流的方式逐步输出内容。这在长文本生成和实时交互场景中特别有用。

> **注意**：流式调用的主要优势不是减少工具调用的延迟（工具调用本身已经很快），而是让用户能够实时看到模型的思考过程和部分结果。

### 5.7.2 OpenAI 流式调用实现

```python
import openai
import json
from typing import Dict, List, Any, Generator

class StreamingFunctionCaller:
    """
    支持流式输出的 Function Calling 实现
    
    在流式模式下，工具调用的参数会分块到达，
    需要累积这些块才能得到完整的参数
    """
    
    def __init__(self, model: str = "gpt-4"):
        self.client = openai.OpenAI()
        self.model = model
    
    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto"
    ) -> Generator[Dict[str, Any], None, None]:
        """
        流式对话，逐步返回结果
        
        Args:
            messages: 消息列表
            tools: 工具定义
            tool_choice: 工具选择策略
        
        Yields:
            流式结果块
        """
        response_stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=True
        )
        
        # 用于累积工具调用参数
        tool_calls_buffer: Dict[int, Dict[str, str]] = {}
        
        for chunk in response_stream:
            if not chunk.choices:
                continue
            
            choice = chunk.choices[0]
            delta = choice.delta
            
            # 处理文本内容
            if delta.content:
                yield {
                    "type": "text",
                    "content": delta.content
                }
            
            # 处理工具调用
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    index = tc.index
                    
                    # 初始化缓冲区
                    if index not in tool_calls_buffer:
                        tool_calls_buffer[index] = {
                            "id": tc.id or "",
                            "name": tc.function.name if tc.function and tc.function.name else "",
                            "arguments": ""
                        }
                    
                    # 累积 ID
                    if tc.id:
                        tool_calls_buffer[index]["id"] = tc.id
                    
                    # 累积函数名
                    if tc.function and tc.function.name:
                        tool_calls_buffer[index]["name"] = tc.function.name
                    
                    # 累积参数
                    if tc.function and tc.function.arguments:
                        tool_calls_buffer[index]["arguments"] += tc.function.arguments
            
            # 处理结束标记
            if choice.finish_reason == "tool_calls":
                # 所有工具调用参数已接收完毕
                for index, buffer in tool_calls_buffer.items():
                    yield {
                        "type": "tool_call",
                        "tool_call": {
                            "id": buffer["id"],
                            "function": {
                                "name": buffer["name"],
                                "arguments": buffer["arguments"]
                            }
                        }
                    }
                
                # 清空缓冲区
                tool_calls_buffer.clear()
            
            elif choice.finish_reason == "stop":
                yield {
                    "type": "done",
                    "finish_reason": "stop"
                }
    
    def chat_with_streaming_tools(
        self,
        user_message: str,
        tools: List[Dict[str, Any]],
        tool_registry: Dict[str, Any]
    ) -> str:
        """
        完整的流式工具调用流程
        
        Args:
            user_message: 用户输入
            tools: 工具定义
            tool_registry: 工具函数注册表
        
        Returns:
            最终回复
        """
        messages = [{"role": "user", "content": user_message}]
        
        while True:
            full_text = ""
            tool_calls = []
            
            # 流式获取结果
            for chunk in self.chat_stream(messages, tools):
                if chunk["type"] == "text":
                    # 实时打印文本（模拟流式输出）
                    print(chunk["content"], end="", flush=True)
                    full_text += chunk["content"]
                
                elif chunk["type"] == "tool_call":
                    tool_calls.append(chunk["tool_call"])
                    tc = chunk["tool_call"]
                    print(f"\n[工具调用] {tc['function']['name']}({tc['function']['arguments']})")
                
                elif chunk["type"] == "done":
                    pass
            
            if full_text:
                print()  # 换行
            
            # 没有工具调用，返回最终文本
            if not tool_calls:
                return full_text
            
            # 有工具调用，执行并继续
            messages.append({
                "role": "assistant",
                "content": full_text if full_text else None,
                "tool_calls": tool_calls
            })
            
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                func_args = json.loads(tc["function"]["arguments"])
                
                # 执行工具
                if func_name in tool_registry:
                    result = tool_registry[func_name](**func_args)
                else:
                    result = {"error": f"未知工具: {func_name}"}
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result, ensure_ascii=False)
                })


# ============ 使用示例 ============

def demo_streaming():
    """演示流式 Function Calling"""
    
    caller = StreamingFunctionCaller(model="gpt-4")
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]
    
    def get_weather(city: str) -> Dict:
        return {"city": city, "temp": 25, "condition": "晴"}
    
    registry = {"get_weather": get_weather}
    
    print("流式 Function Calling 示例")
    print("=" * 40)
    
    response = caller.chat_with_streaming_tools(
        "北京今天天气怎么样？",
        tools,
        registry
    )
    
    print(f"\n最终回复: {response}")


 # demo_streaming()
 ```

不同的 Agent 实现方式各有优劣。下面的交互式对比可以帮助你根据需求选择最合适的方案：

<div data-component="AgentArchitectureComparisonV5"></div>

 ### 5.7.3 Anthropic 流式调用实现

```python
import anthropic
import json
from typing import Dict, List, Any, Generator

class AnthropicStreamingCaller:
    """Anthropic Claude 的流式 Function Calling 实现"""
    
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model
    
    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: str = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Anthropic 流式调用
        
        Claude 的流式输出格式与 OpenAI 不同：
        - 使用 event types: message_start, content_block_start,
          content_block_delta, content_block_stop, message_delta, message_stop
        - 工具调用通过 content blocks 传递
        
        Args:
            messages: 消息列表
            tools: 工具定义
            system: 系统提示
        
        Yields:
            流式结果
        """
        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": messages,
            "tools": tools,
            "stream": True
        }
        
        if system:
            kwargs["system"] = system
        
        # 累积当前 content block 的内容
        current_block = None
        current_tool_call = None
        
        with self.client.messages.stream(**kwargs) as stream:
            for event in stream:
                event_type = event.type
                
                if event_type == "message_start":
                    # 消息开始
                    yield {"type": "message_start"}
                
                elif event_type == "content_block_start":
                    # 新的 content block 开始
                    block = event.content_block
                    current_block = {"type": block.type}
                    
                    if block.type == "tool_use":
                        current_tool_call = {
                            "id": block.id,
                            "name": block.name,
                            "input_json": ""
                        }
                        yield {
                            "type": "tool_call_start",
                            "tool_call": current_tool_call
                        }
                    elif block.type == "text":
                        yield {"type": "text_start"}
                
                elif event_type == "content_block_delta":
                    delta = event.delta
                    
                    if hasattr(delta, "text"):
                        yield {
                            "type": "text_delta",
                            "text": delta.text
                        }
                    
                    elif hasattr(delta, "partial_json"):
                        if current_tool_call is not None:
                            current_tool_call["input_json"] += delta.partial_json
                            yield {
                                "type": "tool_call_delta",
                                "partial_json": delta.partial_json
                            }
                
                elif event_type == "content_block_stop":
                    if current_block and current_block["type"] == "tool_use":
                        if current_tool_call:
                            yield {
                                "type": "tool_call_complete",
                                "tool_call": current_tool_call
                            }
                            current_tool_call = None
                    
                    current_block = None
                
                elif event_type == "message_delta":
                    # 消息级别更新（如 stop_reason）
                    yield {
                        "type": "message_delta",
                        "stop_reason": event.delta.stop_reason
                    }
                
                elif event_type == "message_stop":
                    yield {"type": "message_stop"}
    
    def chat_with_streaming_tools(
        self,
        user_message: str,
        tools: List[Dict[str, Any]],
        tool_registry: Dict[str, Any],
        system: str = None
    ) -> str:
        """
        完整的 Anthropic 流式工具调用流程
        """
        messages = [{"role": "user", "content": user_message}]
        
        while True:
            full_text = ""
            tool_calls = []
            
            for event in self.chat_stream(messages, tools, system):
                if event["type"] == "text_delta":
                    print(event["text"], end="", flush=True)
                    full_text += event["text"]
                
                elif event["type"] == "tool_call_complete":
                    tc = event["tool_call"]
                    tool_calls.append(tc)
                    print(f"\n[工具调用] {tc['name']}")
                
                elif event["type"] == "message_delta":
                    stop_reason = event.get("stop_reason")
                    if stop_reason == "end_turn":
                        print()
                        return full_text
            
            if not tool_calls:
                return full_text
            
            # 执行工具调用并添加到消息
            messages.append({
                "role": "assistant",
                "content": full_text if full_text else None
            })
            
            tool_results = []
            for tc in tool_calls:
                func_name = tc["name"]
                func_args = json.loads(tc["input_json"])
                
                if func_name in tool_registry:
                    result = tool_registry[func_name](**func_args)
                else:
                    result = {"error": f"未知工具: {func_name}"}
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": json.dumps(result, ensure_ascii=False)
                })
            
            messages.append({
                "role": "user",
                "content": tool_results
            })
```

---

## 5.8 Structured Output 与 JSON Schema

### 5.8.1 Structured Output 的概念

Structured Output 是指 LLM 输出严格符合预定义 JSON Schema 的结构化数据。这在 Function Calling 中至关重要——工具参数必须是合法的 JSON。

$$
\text{Structured Output} \subseteq \{x \mid \text{validate}(x, \text{schema}) = \text{true}\}
$$

### 5.8.2 JSON Schema 基础

JSON Schema 是一种用于描述和验证 JSON 数据结构的标准：

```python
# 基本类型定义
basic_schema = {
    "type": "object",
    "properties": {
        # 字符串类型
        "name": {
            "type": "string",
            "description": "用户名称",
            "minLength": 1,
            "maxLength": 100,
            "pattern": "^[\\w\\s]+$"  # 正则约束
        },
        
        # 数字类型
        "age": {
            "type": "integer",
            "description": "年龄",
            "minimum": 0,
            "maximum": 150
        },
        
        # 布尔类型
        "is_active": {
            "type": "boolean",
            "description": "是否激活"
        },
        
        # 枚举类型
        "role": {
            "type": "string",
            "enum": ["admin", "user", "guest"],
            "description": "用户角色"
        },
        
        # 数组类型
        "tags": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "标签列表",
            "minItems": 1,
            "maxItems": 10
        },
        
        # 嵌套对象
        "address": {
            "type": "object",
            "properties": {
                "street": {"type": "string"},
                "city": {"type": "string"},
                "zip_code": {"type": "string", "pattern": "^\\d{6}$"}
            },
            "required": ["street", "city"]
        }
    },
    "required": ["name", "age", "role"]
}
```

### 5.8.3 OpenAI Structured Output

OpenAI 提供了 `response_format` 参数来强制模型输出符合特定 schema 的 JSON：

```python
import openai
import json
from pydantic import BaseModel, Field
from typing import List, Optional

# 使用 Pydantic 定义输出结构
class MovieReview(BaseModel):
    """电影评论结构"""
    title: str = Field(description="电影名称")
    rating: float = Field(description="评分（1-10）", ge=1, le=10)
    summary: str = Field(description="一句话总结")
    pros: List[str] = Field(description="优点列表")
    cons: List[str] = Field(description="缺点列表")
    recommend: bool = Field(description="是否推荐观看")

# 转换为 JSON Schema
schema = MovieReview.model_json_schema()
print(json.dumps(schema, indent=2, ensure_ascii=False))

# 使用 response_format 参数
client = openai.OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "user", "content": "请评价电影《盗梦空间》"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "movie_review",
            "strict": True,  # 严格模式，确保输出完全符合 schema
            "schema": schema
        }
    }
)

# 解析结果
review_json = json.loads(response.choices[0].message.content)
review = MovieReview(**review_json)
print(f"评分: {review.rating}/10")
print(f"推荐: {'是' if review.recommend else '否'}")
```

### 5.8.4 复杂 Schema 设计

对于更复杂的场景，需要设计嵌套和多态的 Schema：

```python
from typing import Union, Literal
from pydantic import BaseModel, Field

# 多态 Schema：支持不同类型的查询
class WeatherQuery(BaseModel):
    """天气查询"""
    type: Literal["weather"] = "weather"
    city: str = Field(description="城市名称")
    date: Optional[str] = Field(default=None, description="日期")

class FlightQuery(BaseModel):
    """航班查询"""
    type: Literal["flight"] = "flight"
    origin: str = Field(description="出发城市")
    destination: str = Field(description="到达城市")
    date: str = Field(description="出发日期")

class HotelQuery(BaseModel):
    """酒店查询"""
    type: Literal["hotel"] = "hotel"
    city: str = Field(description="城市")
    check_in: str = Field(description="入住日期")
    check_out: str = Field(description="退房日期")
    guests: int = Field(description="人数", ge=1)

# 使用 Union 类型
TravelQuery = Union[WeatherQuery, FlightQuery, HotelQuery]

class TravelRequest(BaseModel):
    """旅行请求：包含多个查询"""
    queries: List[TravelQuery] = Field(description="查询列表")
    budget: Optional[float] = Field(default=None, description="预算（人民币）")
    priority: Literal["price", "time", "comfort"] = Field(
        default="price",
        description="优先级"
    )

# 生成 schema
schema = TravelRequest.model_json_schema()
print(json.dumps(schema, indent=2, ensure_ascii=False))
```

---

## 5.9 Schema 设计最佳实践

### 5.9.1 工具描述的设计

工具描述是 LLM 决定是否调用工具的关键依据。好的描述应该：

1. **清晰明确**：避免模糊的描述
2. **包含使用场景**：说明何时应该使用这个工具
3. **列出限制**：说明工具不能做什么
4. **提供示例**：给出调用示例

```python
# ❌ 不好的描述
bad_tool = {
    "name": "search",
    "description": "搜索",
    "parameters": {...}
}

# ✅ 好的描述
good_tool = {
    "name": "search_products",
    "description": (
        "在电商平台搜索商品。"
        "当用户询问商品信息、比较价格或寻找特定商品时使用此工具。"
        "支持按名称、类别、价格范围搜索。"
        "注意：此工具只能搜索在售商品，不包含已下架商品。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词，如 'iPhone 15' 或 '运动鞋'"
            },
            "category": {
                "type": "string",
                "enum": ["electronics", "clothing", "food", "books"],
                "description": "商品类别（可选）"
            },
            "min_price": {
                "type": "number",
                "description": "最低价格（可选，人民币）"
            },
            "max_price": {
                "type": "number",
                "description": "最高价格（可选，人民币）"
            }
        },
        "required": ["query"]
    }
}
```

### 5.9.2 参数命名规范

参数命名应该遵循以下原则：

| 原则 | 好的例子 | 不好的例子 |
|:---|:---|:---|
| 使用 snake_case | `user_name` | `userName` 或 `user-name` |
| 语义清晰 | `departure_date` | `date1` |
| 避免缩写 | `maximum_price` | `max_p` |
| 类型明确 | `is_available` (bool) | `available` (歧义) |

### 5.9.3 必需参数 vs 可选参数

合理划分必需参数和可选参数：

```python
# 好的设计：只将真正必需的参数设为 required
search_schema = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "搜索关键词"
        },
        "limit": {
            "type": "integer",
            "description": "返回结果数量，默认10",
            "default": 10,
            "minimum": 1,
            "maximum": 100
        },
        "offset": {
            "type": "integer",
            "description": "分页偏移量，默认0",
            "default": 0
        }
    },
    "required": ["query"]  # 只有 query 是必需的
}
```

### 5.9.4 错误处理的 Schema 设计

为工具定义一个标准的错误返回格式：

```python
# 标准错误返回格式
error_response_schema = {
    "type": "object",
    "properties": {
        "success": {
            "type": "boolean",
            "description": "是否成功"
        },
        "data": {
            "description": "成功时返回的数据"
        },
        "error": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "错误代码"
                },
                "message": {
                    "type": "string",
                    "description": "错误描述"
                }
            }
        }
    },
    "required": ["success"]
}
```

---

## 5.10 错误处理与重试机制

### 5.10.1 常见错误类型

在 Function Calling 中，可能遇到以下几类错误：

| 错误类型 | 描述 | 处理策略 |
|:---|:---|:---|
| **API 错误** | 网络超时、速率限制 | 指数退避重试 |
| **参数解析错误** | LLM 输出的 JSON 无效 | 请求 LLM 重新生成 |
| **工具执行错误** | 工具函数抛出异常 | 返回错误信息给 LLM |
| **参数验证错误** | 参数值不符合约束 | 请求 LLM 修正参数 |
| **幻觉参数** | LLM 生成不存在的参数值 | 验证并拒绝 |

### 5.10.2 完整的错误处理实现

```python
import openai
import json
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolErrorType(Enum):
    """工具错误类型枚举"""
    API_ERROR = "api_error"                    # API 调用错误
    PARSE_ERROR = "parse_error"                # JSON 解析错误
    EXECUTION_ERROR = "execution_error"        # 工具执行错误
    VALIDATION_ERROR = "validation_error"      # 参数验证错误
    HALLUCINATION_ERROR = "hallucination_error"  # 幻觉参数错误
    UNKNOWN_ERROR = "unknown_error"            # 未知错误


@dataclass
class ToolError:
    """工具错误信息"""
    error_type: ToolErrorType
    message: str
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    retryable: bool = True
    details: Optional[Dict[str, Any]] = None


class ToolErrorHandler:
    """
    工具调用错误处理器
    
    负责：
    1. 识别错误类型
    2. 决定是否重试
    3. 生成错误信息给 LLM
    4. 记录错误日志
    """
    
    def __init__(self, max_retries: int = 3):
        """
        初始化错误处理器
        
        Args:
            max_retries: 最大重试次数
        """
        self.max_retries = max_retries
        self.error_counts: Dict[str, int] = {}  # 按工具名统计错误次数
    
    def classify_error(self, error: Exception, context: Dict = None) -> ToolError:
        """
        对错误进行分类
        
        Args:
            error: 异常对象
            context: 额外上下文信息
        
        Returns:
            分类后的错误信息
        """
        error_str = str(error).lower()
        
        # API 错误
        if isinstance(error, (openai.APIError, openai.APITimeoutError)):
            retryable = not isinstance(error, openai.BadRequestError)
            return ToolError(
                error_type=ToolErrorType.API_ERROR,
                message=f"API 调用失败: {str(error)}",
                retryable=retryable
            )
        
        # JSON 解析错误
        if "json" in error_str and ("parse" in error_str or "decode" in error_str):
            return ToolError(
                error_type=ToolErrorType.PARSE_ERROR,
                message=f"JSON 解析失败: {str(error)}",
                retryable=True
            )
        
        # 参数错误
        if "argument" in error_str or "parameter" in error_str:
            return ToolError(
                error_type=ToolErrorType.VALIDATION_ERROR,
                message=f"参数验证失败: {str(error)}",
                retryable=True
            )
        
        # 未知错误
        return ToolError(
            error_type=ToolErrorType.UNKNOWN_ERROR,
            message=f"未知错误: {str(error)}",
            retryable=True
        )
    
    def should_retry(self, error: ToolError, attempt: int) -> bool:
        """
        判断是否应该重试
        
        Args:
            error: 错误信息
            attempt: 当前尝试次数
        
        Returns:
            是否应该重试
        """
        if not error.retryable:
            return False
        
        if attempt >= self.max_retries:
            return False
        
        # 检查特定工具的错误次数
        if error.tool_name:
            count = self.error_counts.get(error.tool_name, 0)
            if count >= 5:  # 同一工具错误超过5次，不再重试
                return False
        
        return True
    
    def get_retry_delay(self, attempt: int, error: ToolError) -> float:
        """
        计算重试延迟时间
        
        使用指数退避策略，加上随机抖动
        
        Args:
            attempt: 当前尝试次数
            error: 错误信息
        
        Returns:
            延迟秒数
        """
        import random
        
        # 基础延迟：1, 2, 4, 8, ... 秒
        base_delay = 2 ** attempt
        
        # 随机抖动：±25%
        jitter = base_delay * 0.25 * (2 * random.random() - 1)
        
        # API 速率限制特殊处理
        if error.error_type == ToolErrorType.API_ERROR:
            # 检查是否是 429 错误（速率限制）
            if "rate" in error.message.lower() or "429" in error.message:
                base_delay = max(base_delay, 30)  # 至少等待30秒
        
        return max(0, base_delay + jitter)
    
    def record_error(self, tool_name: str):
        """记录工具错误次数"""
        self.error_counts[tool_name] = self.error_counts.get(tool_name, 0) + 1
    
    def generate_error_response(self, error: ToolError) -> str:
        """
        生成给 LLM 的错误响应
        
        帮助 LLM 理解错误原因并尝试修正
        
        Args:
            error: 错误信息
        
        Returns:
            错误响应字符串
        """
        error_messages = {
            ToolErrorType.API_ERROR: (
                "工具调用时发生 API 错误。"
                "请稍后重试或尝试其他方法。"
            ),
            ToolErrorType.PARSE_ERROR: (
                "工具返回的数据格式异常，无法解析。"
                "请尝试调用其他工具或直接回答用户。"
            ),
            ToolErrorType.EXECUTION_ERROR: (
                f"工具 {error.tool_name} 执行失败: {error.message}。"
                "请检查参数是否正确，或尝试其他方法。"
            ),
            ToolErrorType.VALIDATION_ERROR: (
                f"工具参数验证失败: {error.message}。"
                "请修正参数后重试。"
            ),
            ToolErrorType.HALLUCINATION_ERROR: (
                "检测到可能的幻觉参数。"
                "请使用更常见的参数值重试。"
            ),
        }
        
        return error_messages.get(
            error.error_type,
            f"发生未知错误: {error.message}"
        )


class RobustFunctionCaller:
    """
    带有完善错误处理的 Function Calling 实现
    
    特点：
    1. 自动重试失败的 API 调用
    2. 智能错误分类
    3. 优雅降级
    4. 详细的错误日志
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        tools: List[Dict] = None,
        tool_registry: Dict[str, Callable] = None,
        max_retries: int = 3,
        max_tool_rounds: int = 10
    ):
        self.client = openai.OpenAI()
        self.model = model
        self.tools = tools or []
        self.tool_registry = tool_registry or {}
        self.max_retries = max_retries
        self.max_tool_rounds = max_tool_rounds
        self.error_handler = ToolErrorHandler(max_retries)
        self.conversation_history: List[Dict[str, Any]] = []
    
    def _call_llm_with_retry(self, messages: List[Dict]) -> Any:
        """
        带重试的 LLM 调用
        
        Args:
            messages: 消息列表
        
        Returns:
            API 响应
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools if self.tools else None,
                    tool_choice="auto" if self.tools else None
                )
                return response
            
            except openai.APIError as e:
                last_error = e
                tool_error = self.error_handler.classify_error(e)
                
                if not self.error_handler.should_retry(tool_error, attempt):
                    raise
                
                delay = self.error_handler.get_retry_delay(attempt, tool_error)
                logger.warning(
                    f"API 调用失败 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}. "
                    f"将在 {delay:.1f}秒后重试"
                )
                time.sleep(delay)
        
        raise last_error
    
    def _execute_tool_safely(
        self,
        tool_call: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        安全地执行单个工具调用
        
        捕获所有异常，确保不会因为单个工具失败而崩溃
        
        Args:
            tool_call: 工具调用信息
        
        Returns:
            执行结果
        """
        function_name = tool_call["function"]["name"]
        arguments_str = tool_call["function"]["arguments"]
        tool_call_id = tool_call["id"]
        
        # 检查工具是否存在
        if function_name not in self.tool_registry:
            self.error_handler.record_error(function_name)
            return {
                "tool_call_id": tool_call_id,
                "result": json.dumps({
                    "success": False,
                    "error": {
                        "code": "TOOL_NOT_FOUND",
                        "message": f"工具 '{function_name}' 不存在"
                    }
                }, ensure_ascii=False),
                "success": False
            }
        
        # 解析参数
        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError as e:
            tool_error = ToolError(
                error_type=ToolErrorType.PARSE_ERROR,
                message=f"JSON 解析失败: {str(e)}",
                tool_name=function_name,
                tool_call_id=tool_call_id
            )
            self.error_handler.record_error(function_name)
            return {
                "tool_call_id": tool_call_id,
                "result": json.dumps({
                    "success": False,
                    "error": {
                        "code": "PARSE_ERROR",
                        "message": self.error_handler.generate_error_response(tool_error)
                    }
                }, ensure_ascii=False),
                "success": False
            }
        
        # 执行工具
        tool_func = self.tool_registry[function_name]
        
        try:
            start_time = time.time()
            result = tool_func(**arguments)
            elapsed = time.time() - start_time
            
            logger.info(
                f"工具 {function_name} 执行成功 "
                f"({elapsed:.2f}秒)"
            )
            
            return {
                "tool_call_id": tool_call_id,
                "result": json.dumps(result, ensure_ascii=False),
                "success": True,
                "elapsed_seconds": round(elapsed, 3)
            }
        
        except TypeError as e:
            # 参数类型错误
            tool_error = ToolError(
                error_type=ToolErrorType.VALIDATION_ERROR,
                message=f"参数错误: {str(e)}",
                tool_name=function_name,
                tool_call_id=tool_call_id
            )
            self.error_handler.record_error(function_name)
            return {
                "tool_call_id": tool_call_id,
                "result": json.dumps({
                    "success": False,
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": self.error_handler.generate_error_response(tool_error)
                    }
                }, ensure_ascii=False),
                "success": False
            }
        
        except Exception as e:
            # 其他执行错误
            tool_error = ToolError(
                error_type=ToolErrorType.EXECUTION_ERROR,
                message=str(e),
                tool_name=function_name,
                tool_call_id=tool_call_id
            )
            self.error_handler.record_error(function_name)
            logger.error(
                f"工具 {function_name} 执行失败: {str(e)}",
                exc_info=True
            )
            return {
                "tool_call_id": tool_call_id,
                "result": json.dumps({
                    "success": False,
                    "error": {
                        "code": "EXECUTION_ERROR",
                        "message": self.error_handler.generate_error_response(tool_error)
                    }
                }, ensure_ascii=False),
                "success": False
            }
    
    def chat(self, user_message: str) -> str:
        """
        带有完善错误处理的对话方法
        
        Args:
            user_message: 用户输入
        
        Returns:
            助手回复
        """
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        tool_rounds = 0
        
        while tool_rounds < self.max_tool_rounds:
            try:
                response = self._call_llm_with_retry(self.conversation_history)
            except Exception as e:
                error_msg = f"抱歉，AI 服务暂时不可用: {str(e)}"
                logger.error(f"LLM 调用最终失败: {str(e)}")
                return error_msg
            
            message = response.choices[0].message
            
            # 检查是否有工具调用
            if not message.tool_calls:
                final_content = message.content or ""
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_content
                })
                return final_content
            
            # 处理工具调用
            tool_rounds += 1
            logger.info(f"--- 工具调用轮次 {tool_rounds} ---")
            
            # 记录助手消息
            self.conversation_history.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })
            
            # 执行所有工具调用
            for tc in message.tool_calls:
                tool_call_dict = {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                
                result = self._execute_tool_safely(tool_call_dict)
                
                # 添加工具结果到对话历史
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result["result"]
                })
        
        return "抱歉，处理过程中达到了最大工具调用次数限制。"
```

### 5.10.3 重试策略的选择

不同的错误类型需要不同的重试策略：

| 错误类型 | 重试策略 | 最大重试次数 | 延迟策略 |
|:---|:---|:---|:---|
| **网络超时** | 指数退避 | 3次 | 1s, 2s, 4s |
| **速率限制 (429)** | 固定延迟 | 5次 | 30s |
| **服务器错误 (5xx)** | 指数退避 | 3次 | 2s, 4s, 8s |
| **参数错误 (400)** | 不重试 | 0次 | - |
| **认证错误 (401)** | 不重试 | 0次 | - |
 | **工具执行错误** | 有限重试 | 2次 | 1s |

Agent 规划能力取决于 LLM 的推理深度。下面的交互式可视化展示了 Agent 如何构建规划树来解决复杂问题：

<div data-component="AgentPlanningDemoV5"></div>

 ---

 ## 5.11 本章小结

本章深入探讨了 LLM Function Calling 的完整技术栈：

1. **本质理解**：Function Calling 是 LLM 输出结构化调用请求的能力，而非真正执行函数。它是 Agent 与外部世界交互的桥梁。

2. **底层原理**：从训练阶段的三元组学习，到推理阶段的受控解码，Function Calling 的核心是让模型在 JSON Schema 约束下生成合法的结构化输出。

3. **API 实现**：OpenAI 和 Anthropic 都提供了 Function Calling API，但在工具定义格式、消息结构、流式输出等方面存在差异。理解这些差异对于跨平台开发至关重要。

4. **并行调用**：并行执行多个工具调用可以显著降低延迟，但需要注意工具之间的依赖关系和资源竞争。

5. **流式调用**：流式输出让用户能够实时看到模型的思考过程，提升交互体验。

6. **Structured Output**：JSON Schema 为工具参数提供了严格的结构约束，确保 LLM 输出的参数始终合法。

7. **错误处理**：完善的错误处理机制是生产环境的必备条件，包括错误分类、重试策略、优雅降级等。

**关键公式回顾**：

$$
\text{Function Calling}: \text{context} \rightarrow \{(f_i, args_i)\} \xrightarrow{\text{execute}} \{r_i\} \xrightarrow{\text{LLM}} \text{answer}
$$

$$
\text{Parallel Speedup}: T_{\text{parallel}} = \max(T_1, T_2, \ldots, T_n) + T_{\text{reason}}
$$

$$
\text{Retry Delay}: d_n = d_0 \cdot 2^n + \text{jitter}, \quad n \leq n_{\max}
$$

---

## 5.12 思考题

1. **概念理解**：为什么说 Function Calling 是"受控文本生成"而不是"函数执行"？请从模型架构和推理过程两个角度解释。

2. **Schema 设计**：设计一个工具，用于查询用户订单状态。要求包含以下参数：订单号（必需）、查询维度（可选，支持"物流"、"支付"、"售后"）、语言偏好（可选，支持"中文"、"英文"）。写出完整的 JSON Schema。

3. **错误处理**：在一个电商 Agent 中，用户要求"帮我查询订单并申请退款"。如果查询订单成功但退款失败，应该如何处理？请设计错误处理流程和用户交互。

4. **性能优化**：假设一个 Agent 需要同时查询 5 个 API 来回答用户问题。如果每个 API 调用平均需要 2 秒，串行执行需要 10 秒。如果改为并行执行，但由于服务器限制最多只能同时处理 3 个请求，实际执行时间是多少？

5. **跨平台对比**：在 OpenAI 中使用 `tool_choice={"type": "function", "function": {"name": "get_weather"}}` 可以强制调用特定工具。Anthropic 不支持此功能。如果需要在 Anthropic 上实现类似效果，你会如何设计？

6. **Schema 最佳实践**：以下 Schema 有什么问题？如何改进？

```json
{
  "type": "object",
  "properties": {
    "d": {"type": "string"},
    "n": {"type": "number"},
    "f": {"type": "boolean"}
  },
  "required": ["d", "n"]
}
```

7. **安全考虑**：如果用户通过 prompt injection 诱导 LLM 调用敏感工具（如删除数据库），应该在哪些层面进行防护？请从模型层、API 层、应用层三个层面分析。

8. **扩展思考**：当前的 Function Calling 是同步的——模型发出调用请求，等待结果返回。如果需要支持长时间运行的任务（如"帮我生成一个视频"），你会如何设计异步 Function Calling 机制？

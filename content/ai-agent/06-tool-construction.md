---
title: "第6章：工具构建与编排 — Agent 的手和脚"
description: "深入讲解LangChain工具构建的三种核心方式、描述优化、编排模式与动态工具生成"
weight: 6
tags: ["agent", "tool", "langchain", "orchestration"]
author: "CS2 AI Agent 编写组"
date: 2026-06-15
---

 # 第6章：工具构建与编排 — Agent 的手和脚

 > "工具是智能的延伸。没有工具的 Agent，就像没有双手的巨人。" —— AI Agent 设计哲学

 下面的交互式图表展示了工具设计的核心原则：

 <div data-component="ToolDesignPrinciples"></div>

 ## 6.1 工具设计原则与哲学

工具（Tool）是 Agent 与外部世界交互的桥梁。一个设计精良的工具，不仅能让 Agent 完成特定任务，更能让整个系统具备可扩展性和可维护性。本节将从哲学高度审视工具设计的核心原则。

### 6.1.1 什么是工具？

在 AI Agent 的语境下，工具是一个可被 LLM 调用的函数或接口。它将自然语言指令转化为具体的计算操作。

$$
\text{Tool}: \mathcal{F}: \mathcal{I} \rightarrow \mathcal{O}
$$

其中：
- $\mathcal{I}$ 表示输入空间（自然语言描述的参数）
- $\mathcal{O}$ 表示输出空间（结构化结果）
- $\mathcal{F}$ 表示从输入到输出的映射函数

### 6.1.2 工具设计的五大原则

| 原则 | 描述 | 示例 |
|------|------|------|
| **单一职责** | 每个工具只做一件事 | 搜索工具不负责格式化结果 |
| **自描述性** | 工具名和描述应清晰传达功能 | `get_weather(city: str)` 而非 `process_data(x)` |
| **幂等性** | 相同输入应产生相同输出 | 查询类工具天然幂等 |
| **容错性** | 工具应优雅处理异常情况 | 网络超时时返回友好错误信息 |
| **可组合性** | 工具之间应能灵活组合 | 搜索工具 + 分析工具 = 智能研究 |

### 6.1.3 工具的分类体系

根据功能特性，工具可以分为以下几类：

```
工具分类
├── 数据获取工具 (Data Retrieval)
│   ├── API 查询工具
│   ├── 数据库查询工具
│   └── 文件读取工具
├── 数据处理工具 (Data Processing)
│   ├── 数学计算工具
│   ├── 文本处理工具
│   └── 数据转换工具
├── 外部交互工具 (External Interaction)
│   ├── HTTP 请求工具
│   ├── 邮件发送工具
│   └── 消息推送工具
└── 系统控制工具 (System Control)
    ├── 代码执行工具
    ├── 进程管理工具
    └── 文件系统操作工具
```

### 6.1.4 工具的数学表示

从信息论角度看，工具是信息的转换器：

$$
I_{out} = T(I_{in}, \theta)
$$

其中 $I_{in}$ 是输入信息，$\theta$ 是工具参数，$I_{out}$ 是输出信息。工具的价值在于降低信息的不确定性：

$$
H(O|I) = H(O) - I(O;I)
$$

 即工具通过提供输入 $I$ 来降低输出 $O$ 的熵。

Agent 工具调用完整流程包括选择工具、生成参数、执行调用、返回结果。下面的交互式演示展示了整个过程：

<div data-component="ToolCallFlowV4"></div>

 ### 6.1.5 工具与 Prompt 的关系

工具的描述本质上是 Prompt 工程的一部分。一个好的工具描述应该包含：

1. **功能说明**：工具能做什么
2. **参数说明**：每个参数的含义和格式
3. **返回值说明**：输出的结构和含义
4. **使用示例**：典型的调用场景
5. **限制说明**：工具不能做什么

```python
# 工具描述的最佳实践示例
TOOL_DESCRIPTION = """
计算两个日期之间的工作日数量（排除周末和节假日）。

参数:
    start_date (str): 开始日期，格式为 YYYY-MM-DD
    end_date (str): 结束日期，格式为 YYYY-MM-DD
    holidays (list[str], optional): 额外的节假日列表

返回值:
    dict: 包含以下字段:
        - workdays: 工作日数量
        - total_days: 总天数
        - excluded_days: 排除的天数

示例:
    count_workdays("2026-01-01", "2026-01-31")
    # 返回: {"workdays": 21, "total_days": 31, "excluded_days": 10}

限制:
    - 仅支持 2000 年以后的日期
    - 不包含时区信息
"""
```

### 6.1.6 工具设计的反模式

以下是常见的工具设计反模式，应尽量避免：

| 反模式 | 问题 | 改进方案 |
|--------|------|----------|
| **万能工具** | 一个工具做太多事情 | 拆分为多个单一职责工具 |
| **模糊命名** | 名称无法传达功能 | 使用 `动词_名词` 格式命名 |
| **缺失描述** | LLM 无法理解工具用途 | 提供详细的 docstring |
| **无错误处理** | 异常导致 Agent 崩溃 | 添加 try-except 并返回错误信息 |
| **过长参数列表** | 超过 5 个参数难以使用 | 重构为配置对象或拆分工具 |

> "好的工具设计就像好的 API 设计：简单、一致、可预测。" —— Martin Fowler

---

## 6.2 LangChain @tool 装饰器详解

`@tool` 装饰器是 LangChain 中最简单、最常用的工具定义方式。它将普通 Python 函数快速转化为可用的工具对象。

### 6.2.1 基础用法

```python
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """将两个整数相加并返回结果。

    Args:
        a: 第一个整数
        b: 第二个整数

    Returns:
        两数之和
    """
    return a + b

# 使用工具
result = add.invoke({"a": 5, "b": 3})
print(result)  # 输出: 8

# 查看工具元数据
print(add.name)        # 输出: add
print(add.description) # 输出: 将两个整数相加并返回结果。
print(add.args_schema.model_json())  # 输出: 参数的 JSON Schema
```

### 6.2.2 参数类型与验证

`@tool` 装饰器支持丰富的参数类型，LangChain 会自动生成对应的 JSON Schema：

```python
from typing import Optional, List, Literal
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# 基础类型工具
@tool
def calculateBMI(weight_kg: float, height_m: float) -> str:
    """计算身体质量指数（BMI）。

    BMI = 体重(kg) / 身高(m)^2

    Args:
        weight_kg: 体重，单位为千克
        height_m: 身高，单位为米

    Returns:
        BMI值和健康状态评估
    """
    if height_m <= 0:
        return "错误：身高必须大于0"

    bmi = weight_kg / (height_m ** 2)

    if bmi < 18.5:
        status = "偏瘦"
    elif bmi < 24:
        status = "正常"
    elif bmi < 28:
        status = "超重"
    else:
        status = "肥胖"

    return f"BMI: {bmi:.1f} ({status})"

# 复杂类型工具
@tool
def search_products(
    query: str,
    category: Optional[str] = None,
    min_price: float = 0.0,
    max_price: float = 10000.0,
    sort_by: Literal["price", "rating", "name"] = "rating",
    tags: List[str] = []
) -> dict:
    """在商品数据库中搜索产品。

    支持按类别、价格范围和标签进行筛选。

    Args:
        query: 搜索关键词
        category: 商品类别，如 'electronics', 'clothing', 'food'
        min_price: 最低价格（元）
        max_price: 最高价格（元）
        sort_by: 排序方式，可选 'price', 'rating', 'name'
        tags: 标签列表，如 ['sale', 'new', 'popular']

    Returns:
        搜索结果列表
    """
    # 模拟搜索逻辑
    results = {
        "query": query,
        "filters": {
            "category": category,
            "price_range": [min_price, max_price],
            "sort_by": sort_by,
            "tags": tags
        },
        "count": 42  # 模拟结果数量
    }
    return results
```

### 6.2.3 工具的调用方式

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str, units: str = "celsius") -> str:
    """获取指定城市的当前天气信息。

    Args:
        city: 城市名称，支持中文和英文
        units: 温度单位，'celsius' 或 'fahrenheit'

    Returns:
        天气信息字符串
    """
    # 模拟天气数据
    weather_data = {
        "北京": {"temp": 22, "condition": "晴", "humidity": 45},
        "上海": {"temp": 25, "condition": "多云", "humidity": 70},
        "广州": {"temp": 28, "condition": "阵雨", "humidity": 85},
    }

    data = weather_data.get(city, {"temp": 20, "condition": "未知", "humidity": 50})

    if units == "fahrenheit":
        data["temp"] = data["temp"] * 9/5 + 32

    return f"{city}天气: {data['condition']}, 温度: {data['temp']}°{'C' if units == 'celsius' else 'F'}, 湿度: {data['humidity']}%"

# 方式1: 使用 invoke 方法
result = get_weather.invoke({"city": "北京"})
print(result)

# 方式2: 直接调用（不推荐，但可用）
result = get_weather.invoke("上海")
print(result)

# 方式3: 批量调用
results = get_weather.batch([
    {"city": "北京", "units": "celsius"},
    {"city": "上海", "units": "fahrenheit"},
])
print(results)

# 方式4: 异步调用
import asyncio

async def async_weather():
    result = await get_weather.ainvoke({"city": "广州"})
    print(result)

asyncio.run(async_weather())
```

### 6.2.4 工具的错误处理

```python
from langchain_core.tools import tool
from typing import Annotated

@tool
def divide_numbers(
    a: Annotated[float, "被除数"],
    b: Annotated[float, "除数"]
) -> str:
    """执行除法运算。

    Args:
        a: 被除数
        b: 除数，不能为零

    Returns:
        除法运算结果
    """
    try:
        if b == 0:
            return "错误：除数不能为零"

        result = a / b

        # 保留合理精度
        if result == int(result):
            return str(int(result))
        return f"{result:.4f}"

    except Exception as e:
        return f"计算错误: {str(e)}"

# 测试错误处理
print(divide_numbers.invoke({"a": 10, "b": 3}))   # 3.3333
print(divide_numbers.invoke({"a": 10, "b": 0}))   # 错误：除数不能为零
print(divide_numbers.invoke({"a": "abc", "b": 2})) # 计算错误: ...
```

### 6.2.5 使用 Pydantic 模型定义参数

当参数结构复杂时，可以使用 Pydantic 模型来定义参数：

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class OrderInput(BaseModel):
    """订单创建的输入参数"""
    product_id: str = Field(
        description="商品ID，格式为 'PROD-XXXX'"
    )
    quantity: int = Field(
        description="购买数量",
        ge=1,  # 大于等于1
        le=100  # 小于等于100
    )
    customer_name: str = Field(
        description="客户姓名"
    )
    shipping_address: str = Field(
        description="收货地址"
    )
    priority: bool = Field(
        default=False,
        description="是否为优先订单"
    )
    notes: Optional[str] = Field(
        default=None,
        description="订单备注"
    )

class OrderOutput(BaseModel):
    """订单创建的输出结果"""
    order_id: str = Field(description="订单ID")
    status: str = Field(description="订单状态")
    total_price: float = Field(description="总价")
    estimated_delivery: str = Field(description="预计送达日期")

@tool(args_schema=OrderInput, return_direct=True)
def create_order(order: OrderInput) -> dict:
    """创建一个新的商品订单。

    根据提供的商品信息和客户信息创建订单，
    并返回订单详情和预计送达时间。

    Args:
        order: 订单创建参数，包含商品ID、数量、客户信息等

    Returns:
        订单详情，包含订单ID、状态、总价和预计送达日期
    """
    # 模拟订单创建逻辑
    order_id = f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    base_price = 99.9  # 假设单价
    total_price = base_price * order.quantity

    if order.priority:
        total_price *= 1.2  # 优先订单加价20%
        delivery_days = 2
    else:
        delivery_days = 7

    return {
        "order_id": order_id,
        "status": "已创建",
        "total_price": round(total_price, 2),
        "estimated_delivery": f"{delivery_days}个工作日内",
        "details": {
            "product_id": order.product_id,
            "quantity": order.quantity,
            "customer": order.customer_name,
            "priority": order.priority
        }
    }

# 使用 Pydantic 模型作为输入
order_input = OrderInput(
    product_id="PROD-001",
    quantity=3,
    customer_name="张三",
    shipping_address="北京市海淀区",
    priority=True,
    notes="请在工作日配送"
)

result = create_order.invoke(order_input.model_dump())
print(result)
```

### 6.2.6 工具的异步支持

```python
import asyncio
import aiohttp
from langchain_core.tools import tool
from typing import List, Dict

@tool
async def fetch_web_content(
    urls: List[str],
    timeout: int = 30
) -> List[Dict]:
    """异步获取多个网页的内容。

    Args:
        urls: 要获取的URL列表
        timeout: 超时时间（秒）

    Returns:
        包含每个URL响应的列表
    """
    async with aiohttp.ClientSession() as session:
        results = []

        for url in urls:
            try:
                async with session.get(url, timeout=timeout) as response:
                    content = await response.text()
                    results.append({
                        "url": url,
                        "status": response.status,
                        "content_length": len(content),
                        "success": True
                    })
            except Exception as e:
                results.append({
                    "url": url,
                    "status": 0,
                    "error": str(e),
                    "success": False
                })

        return results

# 异步调用示例
async def main():
    urls = [
        "https://example.com",
        "https://httpbin.org/get",
        "https://invalid-url.example"
    ]

    results = await fetch_web_content.ainvoke({
        "urls": urls,
        "timeout": 10
    })

    for result in results:
        print(f"URL: {result['url']}, Success: {result.get('success')}")

asyncio.run(main())
```

### 6.2.7 工具的流式输出

```python
from langchain_core.tools import tool
from typing import Generator
import time

@tool
def stream_text_analysis(
    text: str,
    analysis_type: str = "sentiment"
) -> Generator[str, None, None]:
    """流式分析文本内容。

    Args:
        text: 要分析的文本
        analysis_type: 分析类型 ('sentiment', 'keywords', 'summary')

    Yields:
        分析结果的各个部分
    """
    # 模拟流式处理
    words = text.split()
    total_words = len(words)

    yield f"开始分析文本，共{total_words}个单词...\n"
    yield f"分析类型: {analysis_type}\n"
    yield "-" * 40 + "\n"

    # 模拟逐词分析
    for i, word in enumerate(words):
        time.sleep(0.01)  # 模拟处理延迟
        if (i + 1) % 10 == 0:
            yield f"已处理 {i + 1}/{total_words} 个单词\n"

    yield "-" * 40 + "\n"
    yield "分析完成！\n"

    # 返回分析结果
    if analysis_type == "sentiment":
        yield "情感分析结果: 积极 (置信度: 0.85)\n"
    elif analysis_type == "keywords":
        yield "关键词: " + ", ".join(words[:5]) + "\n"
    elif analysis_type == "summary":
        yield f"摘要: 这段文本包含{total_words}个单词。\n"
```

### 6.2.8 完整的工具集合示例

```python
from langchain_core.tools import tool
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
import json

# 工具1: 文本处理
@tool
def analyze_text(text: str) -> dict:
    """分析文本的基本统计信息。

    Args:
        text: 要分析的文本内容

    Returns:
        文本统计信息，包括字符数、单词数、行数等
    """
    lines = text.split('\n')
    words = text.split()
    characters = len(text)
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')

    return {
        "total_characters": characters,
        "total_words": len(words),
        "total_lines": len(lines),
        "chinese_characters": chinese_chars,
        "average_word_length": sum(len(w) for w in words) / max(len(words), 1),
        "longest_line": max(len(line) for line in lines) if lines else 0
    }

# 工具2: 数据转换
@tool
def convert_format(
    data: str,
    from_format: str,
    to_format: str
) -> str:
    """将数据从一种格式转换为另一种格式。

    Args:
        data: 要转换的数据
        from_format: 源格式 ('json', 'csv', 'xml')
        to_format: 目标格式 ('json', 'csv', 'xml')

    Returns:
        转换后的数据
    """
    # 简化的格式转换逻辑
    supported_formats = ['json', 'csv', 'xml']

    if from_format not in supported_formats:
        return f"错误: 不支持的源格式 {from_format}"
    if to_format not in supported_formats:
        return f"错误: 不支持的目标格式 {to_format}"

    if from_format == to_format:
        return data

    # 模拟转换逻辑
    return f"[已将数据从 {from_format} 转换为 {to_format}]"

# 工具3: 数学计算
@tool
def evaluate_expression(expression: str) -> str:
    """安全地计算数学表达式。

    支持基本运算: +, -, *, /, **, %

    Args:
        expression: 数学表达式字符串

    Returns:
        计算结果
    """
    import ast
    import operator as op

    # 安全的运算符映射
    operators = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.Mod: op.mod,
        ast.USub: op.neg,
    }

    def safe_eval(node):
        if isinstance(node, ast.Expression):
            return safe_eval(node.body)
        elif isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = safe_eval(node.left)
            right = safe_eval(node.right)
            return operators[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            return operators[type(node.op)](safe_eval(node.operand))
        else:
            raise ValueError(f"不支持的表达式类型: {type(node)}")

    try:
        tree = ast.parse(expression, mode='eval')
        result = safe_eval(tree)
        return str(result)
    except Exception as e:
        return f"计算错误: {str(e)}"

# 工具4: 列表操作
@tool
def manipulate_list(
    operation: str,
    items: List[str],
    value: Optional[str] = None
) -> dict:
    """对列表执行各种操作。

    Args:
        operation: 操作类型 ('add', 'remove', 'sort', 'reverse', 'unique')
        items: 输入列表
        value: 操作值（某些操作需要）

    Returns:
        操作结果
    """
    result_items = items.copy()

    if operation == "add" and value:
        result_items.append(value)
    elif operation == "remove" and value:
        if value in result_items:
            result_items.remove(value)
    elif operation == "sort":
        result_items.sort()
    elif operation == "reverse":
        result_items.reverse()
    elif operation == "unique":
        seen = set()
        result_items = [x for x in result_items if not (x in seen or seen.add(x))]

    return {
        "operation": operation,
        "original_length": len(items),
        "result_length": len(result_items),
        "items": result_items
    }

# 工具5: 文件操作（安全的模拟）
@tool
def safe_file_operation(
    operation: str,
    filepath: str,
    content: Optional[str] = None
) -> str:
    """安全地执行文件操作。

    Args:
        operation: 操作类型 ('read', 'write', 'info')
        filepath: 文件路径
        content: 写入内容（仅写入操作需要）

    Returns:
        操作结果
    """
    import os
    import hashlib

    # 安全检查：禁止访问系统文件
    dangerous_paths = ['/etc', '/usr', '/bin', '/sbin', '/System']
    for dangerous in dangerous_paths:
        if filepath.startswith(dangerous):
            return "错误: 禁止访问系统文件"

    if operation == "info":
        if os.path.exists(filepath):
            stat = os.stat(filepath)
            return json.dumps({
                "exists": True,
                "size": stat.st_size,
                "modified": stat.st_mtime
            })
        else:
            return json.dumps({"exists": False})

    return "操作完成"

# 收集所有工具
tools = [
    analyze_text,
    convert_format,
    evaluate_expression,
    manipulate_list,
    safe_file_operation
]

# 打印工具信息
for t in tools:
    print(f"工具名: {t.name}")
    print(f"描述: {t.description}")
    print(f"参数: {list(t.args.keys())}")
    print("-" * 40)
```

> "装饰器之美在于它将复杂的工具注册逻辑隐藏在简洁的语法背后。" —— Python 设计哲学

---

## 6.3 StructuredTool详解

`StructuredTool` 提供了比 `@tool` 更灵活的工具定义方式，特别适合需要精细控制工具行为的场景。

### 6.3.1 StructuredTool 的基本结构

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Callable, Optional

class CalculatorInput(BaseModel):
    """计算器输入参数"""
    expression: str = Field(
        description="数学表达式，如 '2 + 3 * 4'"
    )
    precision: int = Field(
        default=2,
        description="结果精度（小数位数）",
        ge=0,
        le=10
    )

def calculate(expression: str, precision: int = 2) -> str:
    """计算数学表达式。

    支持基本运算: +, -, *, /, **
    """
    try:
        # 安全的表达式计算
        import ast
        import operator as op

        ops = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.Pow: op.pow,
        }

        def _eval(node):
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            elif isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                return ops[type(node.op)](_eval(node.left), _eval(node.right))
            elif isinstance(node, ast.UnaryOp):
                return ops[type(node.op)](_eval(node.operand))
            else:
                raise ValueError("不支持的表达式")

        tree = ast.parse(expression, mode='eval')
        result = _eval(tree)

        if isinstance(result, float):
            return f"{result:.{precision}f}"
        return str(result)

    except Exception as e:
        return f"计算错误: {str(e)}"

# 创建 StructuredTool
calculator = StructuredTool.from_function(
    func=calculate,
    name="calculator",
    description="安全地计算数学表达式",
    args_schema=CalculatorInput,
    return_direct=True
)

# 使用工具
print(calculator.invoke({"expression": "2 + 3 * 4", "precision": 3}))
# 输出: 14.000
```

### 6.3.2 异步 StructuredTool

```python
import asyncio
import aiohttp
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class WebSearchInput(BaseModel):
    """网页搜索输入参数"""
    query: str = Field(description="搜索关键词")
    num_results: int = Field(default=5, description="返回结果数量", ge=1, le=20)
    language: str = Field(default="zh", description="搜索语言")

class SearchResult(BaseModel):
    """单条搜索结果"""
    title: str = Field(description="结果标题")
    url: str = Field(description="结果链接")
    snippet: str = Field(description="结果摘要")

async def async_web_search(
    query: str,
    num_results: int = 5,
    language: str = "zh"
) -> List[Dict]:
    """异步执行网页搜索。

    Args:
        query: 搜索关键词
        num_results: 返回结果数量
        language: 搜索语言

    Returns:
        搜索结果列表
    """
    # 模拟异步搜索
    await asyncio.sleep(0.1)  # 模拟网络延迟

    results = []
    for i in range(num_results):
        results.append({
            "title": f"搜索结果 {i+1}: {query}",
            "url": f"https://example.com/result{i+1}",
            "snippet": f"这是关于 '{query}' 的第 {i+1} 条结果摘要。"
        })

    return results

# 创建异步 StructuredTool
async_search_tool = StructuredTool.from_function(
    func=async_web_search,
    coroutine=async_web_search,
    name="web_search",
    description="异步执行网页搜索并返回结果",
    args_schema=WebSearchInput
)

# 异步使用
async def main():
    results = await async_search_tool.ainvoke({
        "query": "Python 编程",
        "num_results": 3
    })
    for r in results:
        print(f"标题: {r['title']}")

asyncio.run(main())
```

### 6.3.3 带状态的 StructuredTool

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any
import json
from datetime import datetime

class StatefulTool:
    """带状态的工具示例"""

    def __init__(self):
        self._state: Dict[str, Any] = {}
        self._history: list = []

    def _record_operation(self, operation: str, result: Any):
        """记录操作历史"""
        self._history.append({
            "operation": operation,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })

    class GetStateInput(BaseModel):
        """获取状态的输入"""
        key: Optional[str] = Field(
            default=None,
            description="要获取的状态键，None表示获取所有状态"
        )

    class SetStateInput(BaseModel):
        """设置状态的输入"""
        key: str = Field(description="状态键")
        value: str = Field(description="状态值")

    def get_state(self, key: Optional[str] = None) -> str:
        """获取当前工具状态。"""
        if key is None:
            result = json.dumps(self._state, ensure_ascii=False, indent=2)
        else:
            result = str(self._state.get(key, "键不存在"))

        self._record_operation("get_state", result)
        return result

    def set_state(self, key: str, value: str) -> str:
        """设置工具状态。"""
        self._state[key] = value
        result = f"已设置 {key} = {value}"
        self._record_operation("set_state", result)
        return result

    class ClearHistoryInput(BaseModel):
        """清空历史的输入"""
        keep_last: int = Field(
            default=0,
            description="保留最近N条记录"
        )

    def clear_history(self, keep_last: int = 0) -> str:
        """清空操作历史。"""
        if keep_last > 0:
            self._history = self._history[-keep_last:]
        else:
            self._history = []
        return f"历史已清空，保留最近 {keep_last} 条"

    def get_tools(self) -> list:
        """获取所有工具列表"""
        return [
            StructuredTool.from_function(
                func=self.get_state,
                name="get_state",
                description="获取当前工具状态",
                args_schema=self.GetStateInput
            ),
            StructuredTool.from_function(
                func=self.set_state,
                name="set_state",
                description="设置工具状态",
                args_schema=self.SetStateInput
            ),
            StructuredTool.from_function(
                func=self.clear_history,
                name="clear_history",
                description="清空操作历史",
                args_schema=self.ClearHistoryInput
            )
        ]

# 使用示例
stateful = StatefulTool()
tools = stateful.get_tools()

# 设置状态
print(tools[1].invoke({"key": "user_id", "value": "12345"}))
print(tools[1].invoke({"key": "session", "value": "abc123"}))

# 获取状态
print(tools[0].invoke({}))  # 获取所有状态
print(tools[0].invoke({"key": "user_id"}))  # 获取特定键

# 清空历史
print(tools[2].invoke({"keep_last": 2}))
```

### 6.3.4 工具组合与链式调用

```python
from langchain_core.tools import StructuredTool, tool
from pydantic import BaseModel, Field
from typing import List, Dict, Callable
from functools import reduce

class DataProcessor:
    """数据处理器，包含多个可组合的工具"""

    def __init__(self):
        self._data: List[Dict] = []

    class FilterInput(BaseModel):
        """过滤条件"""
        field: str = Field(description="字段名")
        operator: str = Field(
            description="操作符",
            pattern="^(eq|ne|gt|lt|gte|lte|contains)$"
        )
        value: str = Field(description="比较值")

    def filter_data(self, field: str, operator: str, value: str) -> str:
        """根据条件过滤数据。

        Args:
            field: 要过滤的字段名
            operator: 操作符 (eq, ne, gt, lt, gte, lte, contains)
            value: 比较值

        Returns:
            过滤后的数据
        """
        operators = {
            "eq": lambda x, y: str(x) == str(y),
            "ne": lambda x, y: str(x) != str(y),
            "gt": lambda x, y: float(x) > float(y),
            "lt": lambda x, y: float(x) < float(y),
            "gte": lambda x, y: float(x) >= float(y),
            "lte": lambda x, y: float(x) <= float(y),
            "contains": lambda x, y: str(y) in str(x),
        }

        if operator not in operators:
            return f"错误: 不支持的操作符 {operator}"

        op_func = operators[operator]
        filtered = [
            item for item in self._data
            if field in item and op_func(item[field], value)
        ]

        return json.dumps(filtered, ensure_ascii=False, indent=2)

    class SortInput(BaseModel):
        """排序参数"""
        field: str = Field(description="排序字段")
        reverse: bool = Field(default=False, description="是否降序")

    def sort_data(self, field: str, reverse: bool = False) -> str:
        """对数据进行排序。

        Args:
            field: 排序字段
            reverse: 是否降序

        Returns:
            排序后的数据
        """
        try:
            sorted_data = sorted(
                self._data,
                key=lambda x: x.get(field, 0),
                reverse=reverse
            )
            return json.dumps(sorted_data, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"排序错误: {str(e)}"

    class AggregateInput(BaseModel):
        """聚合参数"""
        field: str = Field(description="聚合字段")
        operation: str = Field(
            description="聚合操作 (sum, avg, min, max, count)"
        )

    def aggregate_data(self, field: str, operation: str) -> str:
        """对数据执行聚合操作。

        Args:
            field: 要聚合的字段
            operation: 聚合操作

        Returns:
            聚合结果
        """
        values = [item.get(field) for item in self._data if field in item]

        if not values:
            return f"字段 {field} 无有效数据"

        operations = {
            "sum": sum,
            "avg": lambda x: sum(x) / len(x) if x else 0,
            "min": min,
            "max": max,
            "count": len
        }

        if operation not in operations:
            return f"错误: 不支持的操作 {operation}"

        try:
            result = operations[operation](values)
            return json.dumps({
                "field": field,
                "operation": operation,
                "result": result,
                "count": len(values)
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"聚合错误: {str(e)}"

    def get_tools(self) -> List[StructuredTool]:
        """获取所有数据处理工具"""
        return [
            StructuredTool.from_function(
                func=self.filter_data,
                name="filter_data",
                description="根据条件过滤数据",
                args_schema=self.FilterInput
            ),
            StructuredTool.from_function(
                func=self.sort_data,
                name="sort_data",
                description="对数据进行排序",
                args_schema=self.SortInput
            ),
            StructuredTool.from_function(
                func=self.aggregate_data,
                name="aggregate_data",
                description="对数据执行聚合操作",
                args_schema=self.AggregateInput
            )
        ]

# 使用示例
processor = DataProcessor()
processor._data = [
    {"name": "产品A", "price": 100, "stock": 50},
    {"name": "产品B", "price": 200, "stock": 30},
    {"name": "产品C", "price": 150, "stock": 80},
    {"name": "产品D", "price": 300, "stock": 20},
]

tools = processor.get_tools()

# 过滤价格大于150的产品
print(tools[0].invoke({
    "field": "price",
    "operator": "gt",
    "value": "150"
}))

# 按价格降序排序
print(tools[1].invoke({
    "field": "price",
    "reverse": True
}))

# 计算平均价格
print(tools[2].invoke({
    "field": "price",
    "operation": "avg"
}))
```

> "StructuredTool 让工具定义更接近于 API 设计，提供了完整的类型系统支持。" —— LangChain 文档

---

## 6.4 BaseTool抽象类详解

`BaseTool` 是所有工具的基类，提供了最底层的控制能力。通过继承 `BaseTool`，可以创建高度自定义的工具。

### 6.4.1 BaseTool 的核心架构

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

class CustomBaseTool(BaseTool):
    """自定义基类工具示例"""

    # 工具元数据
    name: str = "custom_tool"
    description: str = "一个自定义的基础工具"

    # 返回值是否直接呈现给用户
    return_direct: bool = False

    # 是否需要人工确认
    handle_tool_error: bool = True

    class Config:
        """工具配置"""
        arbitrary_types_allowed = True

    def _run(self, **kwargs: Any) -> str:
        """同步执行工具的主要逻辑。

        这是必须实现的核心方法。
        """
        raise NotImplementedError

    async def _arun(self, **kwargs: Any) -> str:
        """异步执行工具的主要逻辑。

         默认实现调用同步版本。
         """
         return self._run(**kwargs)
 ```

不同的 Agent 实现方式各有优劣。下面的交互式对比可以帮助你选择合适的实现方式：

<div data-component="AgentArchitectureComparisonV6"></div>

 ### 6.4.2 完整的 BaseTool 实现

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, Type
import json
import hashlib
import time
from datetime import datetime

class APIClientTool(BaseTool):
    """API 客户端工具，支持缓存和重试"""

    name: str = "api_client"
    description: str = "调用外部API并返回结果"

    # API配置
    base_url: str = Field(description="API基础URL")
    api_key: Optional[str] = Field(default=None, description="API密钥")
    timeout: int = Field(default=30, description="超时时间（秒）")
    max_retries: int = Field(default=3, description="最大重试次数")
    cache_ttl: int = Field(default=300, description="缓存过期时间（秒）")

    # 内部状态
    _cache: Dict[str, Dict] = {}
    _request_count: int = 0
    _error_count: int = 0

    class Config:
        """Pydantic配置"""
        underscore_attrs_are_private = True
        arbitrary_types_allowed = True

    def _get_cache_key(self, endpoint: str, params: Dict) -> str:
        """生成缓存键"""
        key_data = json.dumps({"endpoint": endpoint, "params": params}, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()

    def _check_cache(self, cache_key: str) -> Optional[Dict]:
        """检查缓存"""
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if time.time() - entry["timestamp"] < self.cache_ttl:
                return entry["data"]
            else:
                del self._cache[cache_key]
        return None

    def _set_cache(self, cache_key: str, data: Any):
        """设置缓存"""
        self._cache[cache_key] = {
            "data": data,
            "timestamp": time.time()
        }

    def _make_request(self, endpoint: str, method: str = "GET", **kwargs) -> Dict:
        """发起API请求（模拟）"""
        self._request_count += 1

        # 模拟API响应
        response = {
            "status": 200,
            "data": {
                "endpoint": endpoint,
                "method": method,
                "params": kwargs,
                "timestamp": datetime.now().isoformat()
            }
        }

        # 模拟偶尔的错误
        if self._request_count % 10 == 0:
            self._error_count += 1
            response["status"] = 500
            response["error"] = "服务器内部错误"

        return response

    def _run(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        use_cache: bool = True
    ) -> str:
        """执行API调用。

        Args:
            endpoint: API端点路径
            method: HTTP方法
            params: 请求参数
            use_cache: 是否使用缓存

        Returns:
            API响应结果
        """
        # 检查缓存
        if use_cache and method == "GET":
            cache_key = self._get_cache_key(endpoint, params or {})
            cached = self._check_cache(cache_key)
            if cached:
                return json.dumps(cached, ensure_ascii=False, indent=2)

        # 重试逻辑
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._make_request(endpoint, method, **(params or {}))

                if response["status"] == 200:
                    result = response["data"]

                    # 缓存结果
                    if use_cache and method == "GET":
                        cache_key = self._get_cache_key(endpoint, params or {})
                        self._set_cache(cache_key, result)

                    return json.dumps(result, ensure_ascii=False, indent=2)
                else:
                    last_error = response.get("error", "未知错误")

            except Exception as e:
                last_error = str(e)

            # 等待后重试（实际应使用指数退避）
            if attempt < self.max_retries - 1:
                time.sleep(0.1 * (attempt + 1))

        # 所有重试都失败
        return json.dumps({
            "error": f"API调用失败: {last_error}",
            "endpoint": endpoint,
            "attempts": self.max_retries,
            "stats": {
                "total_requests": self._request_count,
                "total_errors": self._error_count
            }
        }, ensure_ascii=False, indent=2)

    async def _arun(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        use_cache: bool = True
    ) -> str:
        """异步版本的API调用"""
        # 简化实现，实际应使用aiohttp
        return self._run(endpoint, method, params, use_cache)

# 使用示例
api_tool = APIClientTool(
    base_url="https://api.example.com",
    api_key="test_key_123",
    timeout=10,
    max_retries=2
)

# 调用工具
result = api_tool.invoke({
    "endpoint": "/users",
    "method": "GET",
    "params": {"page": 1, "limit": 10}
})
print(result)
```

### 6.4.3 带验证的 BaseTool

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, field_validator
from typing import Any, Optional, List
import re

class ValidatedInput(BaseModel):
    """带验证的输入模型"""

    email: str = Field(description="用户邮箱")
    phone: Optional[str] = Field(default=None, description="手机号码")
    name: str = Field(description="用户名")
    age: int = Field(description="年龄")

    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, v):
            raise ValueError(f'无效的邮箱格式: {v}')
        return v.lower()

    @field_validator('phone')
    @classmethod
    def validate_phone(cls, v):
        if v is not None:
            pattern = r'^1[3-9]\d{9}$'
            if not re.match(pattern, v):
                raise ValueError(f'无效的手机号码: {v}')
        return v

    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError(f'年龄必须在0-150之间: {v}')
        return v

class ValidatedUserTool(BaseTool):
    """带输入验证的用户管理工具"""

    name: str = "user_manager"
    description: str = "管理用户信息，支持验证和存储"

    # 内部存储
    _users: dict = {}

    class Config:
        underscore_attrs_are_private = True

    def _run(self, action: str, **kwargs: Any) -> str:
        """执行用户管理操作。

        Args:
            action: 操作类型 ('create', 'get', 'update', 'delete')
            **kwargs: 操作参数

        Returns:
            操作结果
        """
        if action == "create":
            return self._create_user(**kwargs)
        elif action == "get":
            return self._get_user(**kwargs)
        elif action == "update":
            return self._update_user(**kwargs)
        elif action == "delete":
            return self._delete_user(**kwargs)
        else:
            return f"错误: 不支持的操作 {action}"

    def _create_user(self, email: str, name: str, age: int, phone: str = None) -> str:
        """创建新用户"""
        # 验证输入
        try:
            user_data = ValidatedInput(
                email=email, name=name, age=age, phone=phone
            )
        except ValueError as e:
            return f"输入验证失败: {str(e)}"

        # 检查邮箱是否已存在
        if user_data.email in self._users:
            return f"错误: 邮箱 {email} 已被注册"

        # 创建用户
        user_id = len(self._users) + 1
        self._users[user_data.email] = {
            "id": user_id,
            **user_data.model_dump()
        }

        return json.dumps({
            "success": True,
            "message": f"用户 {name} 创建成功",
            "user_id": user_id
        }, ensure_ascii=False, indent=2)

    def _get_user(self, email: str) -> str:
        """获取用户信息"""
        if email not in self._users:
            return f"错误: 未找到邮箱为 {email} 的用户"

        user = self._users[email]
        return json.dumps(user, ensure_ascii=False, indent=2)

    def _update_user(self, email: str, **updates) -> str:
        """更新用户信息"""
        if email not in self._users:
            return f"错误: 未找到邮箱为 {email} 的用户"

        self._users[email].update(updates)
        return json.dumps({
            "success": True,
            "message": f"用户 {email} 更新成功"
        }, ensure_ascii=False, indent=2)

    def _delete_user(self, email: str) -> str:
        """删除用户"""
        if email not in self._users:
            return f"错误: 未找到邮箱为 {email} 的用户"

        del self._users[email]
        return json.dumps({
            "success": True,
            "message": f"用户 {email} 已删除"
        }, ensure_ascii=False, indent=2)

# 使用示例
user_tool = ValidatedUserTool()

# 创建用户
result = user_tool.invoke({
    "action": "create",
    "email": "zhangsan@example.com",
    "name": "张三",
    "age": 28,
    "phone": "13800138000"
})
print(result)

# 获取用户
result = user_tool.invoke({
    "action": "get",
    "email": "zhangsan@example.com"
})
print(result)
```

### 6.4.4 工具组合模式

```python
from langchain_core.tools import BaseTool, StructuredTool
from typing import List, Dict, Callable, Any
from pydantic import BaseModel, Field
from functools import wraps

class ToolChain:
    """工具链：将多个工具组合成管道"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.tools: List[BaseTool] = []
        self._pre_hooks: Dict[str, Callable] = {}
        self._post_hooks: Dict[str, Callable] = {}

    def add_tool(self, tool: BaseTool) -> 'ToolChain':
        """添加工具到链中"""
        self.tools.append(tool)
        return self

    def add_pre_hook(self, tool_name: str, hook: Callable) -> 'ToolChain':
        """添加前置钩子"""
        self._pre_hooks[tool_name] = hook
        return self

    def add_post_hook(self, tool_name: str, hook: Callable) -> 'ToolChain':
        """添加后置钩子"""
        self._post_hooks[tool_name] = hook
        return self

    def execute(self, initial_input: Dict) -> Dict:
        """执行工具链"""
        current_output = initial_input
        execution_log = []

        for tool in self.tools:
            tool_name = tool.name
            start_time = time.time()

            # 执行前置钩子
            if tool_name in self._pre_hooks:
                current_output = self._pre_hooks[tool_name](current_output)

            # 执行工具
            try:
                result = tool.invoke(current_output)
                status = "success"
            except Exception as e:
                result = str(e)
                status = "error"

            # 记录执行日志
            execution_log.append({
                "tool": tool_name,
                "input": current_output,
                "output": result,
                "status": status,
                "duration": time.time() - start_time
            })

            # 执行后置钩子
            if tool_name in self._post_hooks:
                result = self._post_hooks[tool_name](result)

            current_output = result

        return {
            "result": current_output,
            "log": execution_log
        }

    def to_tool(self) -> StructuredTool:
        """将工具链转换为单个工具"""
        def chain_func(**kwargs):
            return self.execute(kwargs)

        return StructuredTool.from_function(
            func=chain_func,
            name=self.name,
            description=self.description
        )

# 使用示例：构建数据处理管道
from langchain_core.tools import tool

@tool
def validate_data(data: str) -> str:
    """验证数据格式"""
    return f"验证通过: {data}"

@tool
def transform_data(data: str) -> str:
    """转换数据格式"""
    return f"已转换: {data.upper()}"

@tool
def aggregate_data(data: str) -> str:
    """聚合数据"""
    return f"聚合完成: {data}"

# 构建工具链
chain = ToolChain(
    name="data_pipeline",
    description="数据处理管道：验证 -> 转换 -> 聚合"
)
chain.add_tool(validate_data)
chain.add_tool(transform_data)
chain.add_tool(aggregate_data)

# 添加钩子
chain.add_pre_hook("validate_data", lambda x: {"data": str(x)})
chain.add_post_hook("transform_data", lambda x: f"处理后: {x}")

# 执行管道
result = chain.execute({"data": "hello world"})
print(json.dumps(result, ensure_ascii=False, indent=2))

# 转换为单个工具
pipeline_tool = chain.to_tool()
result = pipeline_tool.invoke({"data": "test input"})
print(result)
```

> "BaseTool 是工具构建的终极形态，给予开发者完全的控制权。" —— LangChain 架构师

---

## 6.5 描述优化数据（Tool Description Optimization）

工具描述的质量直接影响 LLM 选择和使用工具的能力。本节介绍如何优化工具描述以提升 Agent 的表现。

### 6.5.1 描述优化的理论基础

工具描述优化可以看作是一个信息检索问题：

$$
P(tool | query) \propto P(query | tool) \cdot P(tool)
$$

其中：
- $P(tool | query)$ 是给定用户查询时选择某个工具的概率
- $P(query | tool)$ 是工具描述与查询的匹配度
- $P(tool)$ 是工具的先验使用频率

优化目标是最大化正确工具被选中的概率：

$$
\max \sum_{i=1}^{N} \mathbb{1}[\text{argmax}_j P(tool_j | q_i) = t_i^*]
$$

其中 $t_i^*$ 是查询 $q_i$ 对应的正确工具。

### 6.5.2 描述优化策略

```python
from typing import List, Dict, Callable
from dataclasses import dataclass
from pydantic import Field
import re

@dataclass
class ToolDescription:
    """工具描述优化配置"""
    name: str
    original_description: str
    optimized_description: str
    keywords: List[str]
    examples: List[str]
    limitations: List[str]
    performance_score: float = 0.0

class DescriptionOptimizer:
    """工具描述优化器"""

    def __init__(self):
        self.optimization_history: List[Dict] = []

    def analyze_description(self, description: str) -> Dict:
        """分析工具描述的质量"""
        metrics = {
            "length": len(description),
            "has_examples": "示例" in description or "example" in description.lower(),
            "has_params": "参数" in description or "Args:" in description,
            "has_return": "返回" in description or "Returns:" in description,
            "has_limitations": "限制" in description or "注意" in description,
            "complexity": len(re.findall(r'[\u4e00-\u9fff]', description)),
            "specificity": len(re.findall(r'[A-Z][a-z]+', description)),
        }

        # 计算质量分数
        score = 0
        if metrics["length"] > 50:
            score += 1
        if metrics["has_examples"]:
            score += 1
        if metrics["has_params"]:
            score += 1
        if metrics["has_return"]:
            score += 1
        if metrics["has_limitations"]:
            score += 1

        metrics["quality_score"] = score / 5.0

        return metrics

    def optimize_description(
        self,
        name: str,
        description: str,
        function: Callable
    ) -> ToolDescription:
        """优化工具描述"""
        # 提取函数信息
        docstring = function.__doc__ or ""
        params = function.__annotations__

        # 分析原始描述
        metrics = self.analyze_description(description)

        # 生成优化后的描述
        optimized_parts = [
            description,
            "",
            "参数:"
        ]

        # 添加参数描述
        for param_name, param_type in params.items():
            if param_name != 'return':
                optimized_parts.append(
                    f"  - {param_name} ({param_type.__name__}): ..."
                )

        optimized_parts.extend([
            "",
            "返回值:",
            "  结构化结果",
            "",
            "使用示例:",
            f"  {name}(...)",
            "",
            "注意事项:",
            "  - 输入验证由工具内部处理",
            "  - 支持批量操作"
        ])

        optimized = "\n".join(optimized_parts)

        # 提取关键词
        keywords = re.findall(r'[\u4e00-\u9fff]+|[A-Za-z]+', description)
        keywords = list(set(keywords))

        # 生成示例
        examples = [
            f"# 基本用法\nresult = {name}(param1='value1')",
            f"# 高级用法\nresult = {name}(param1='value1', param2='value2')"
        ]

        # 记录优化历史
        self.optimization_history.append({
            "name": name,
            "original_length": len(description),
            "optimized_length": len(optimized),
            "quality_improvement": metrics["quality_score"]
        })

        return ToolDescription(
            name=name,
            original_description=description,
            optimized_description=optimized,
            keywords=keywords,
            examples=examples,
            limitations=["仅支持特定格式的输入"],
            performance_score=metrics["quality_score"]
        )

# 使用示例
optimizer = DescriptionOptimizer()

def sample_function(query: str, max_results: int = 10) -> List[Dict]:
    """搜索相关内容"""
    return []

optimized = optimizer.optimize_description(
    name="search",
    description="搜索相关内容",
    function=sample_function
)

print(f"工具名: {optimized.name}")
print(f"质量分数: {optimized.performance_score}")
print(f"关键词: {optimized.keywords}")
print(f"\n优化后描述:\n{optimized.optimized_description}")
```

### 6.5.3 A/B 测试框架

```python
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import json

@dataclass
class ABTestVariant:
    """A/B测试变体"""
    name: str
    description: str
    selection_count: int = 0
    success_count: int = 0

    @property
    def success_rate(self) -> float:
        if self.selection_count == 0:
            return 0.0
        return self.success_count / self.selection_count

class ToolDescriptionABTest:
    """工具描述A/B测试框架"""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self.variants: Dict[str, ABTestVariant] = {}
        self.test_history: List[Dict] = []

    def add_variant(self, name: str, description: str):
        """添加测试变体"""
        self.variants[name] = ABTestVariant(
            name=name,
            description=description
        )

    def select_variant(self) -> ABTestVariant:
        """随机选择一个变体"""
        if not self.variants:
            raise ValueError("没有可用的测试变体")

        # 使用Thompson Sampling进行选择
        total_success = sum(v.success_count for v in self.variants.values())
        total_trials = sum(v.selection_count for v in self.variants.values())

        if total_trials == 0:
            # 初始阶段，均匀选择
            return random.choice(list(self.variants.values()))

        # 计算每个变体的Beta分布参数
        import numpy as np
        scores = []
        for variant in self.variants.values():
            alpha = variant.success_count + 1
            beta = variant.selection_count - variant.success_count + 1
            # 从Beta分布采样
            score = np.random.beta(alpha, beta)
            scores.append(score)

        # 选择得分最高的变体
        best_idx = np.argmax(scores)
        return list(self.variants.values())[best_idx]

    def record_selection(self, variant_name: str, success: bool = True):
        """记录选择结果"""
        if variant_name not in self.variants:
            raise ValueError(f"未知的变体: {variant_name}")

        variant = self.variants[variant_name]
        variant.selection_count += 1
        if success:
            variant.success_count += 1

        # 记录历史
        self.test_history.append({
            "variant": variant_name,
            "success": success,
            "total_selections": variant.selection_count,
            "total_successes": variant.success_count
        })

    def get_results(self) -> Dict:
        """获取测试结果"""
        results = {}
        for name, variant in self.variants.items():
            results[name] = {
                "description": variant.description,
                "selections": variant.selection_count,
                "successes": variant.success_count,
                "success_rate": f"{variant.success_rate:.2%}"
            }

        # 找出最佳变体
        best_variant = max(
            self.variants.values(),
            key=lambda v: v.success_rate if v.selection_count > 0 else 0
        )

        return {
            "tool_name": self.tool_name,
            "variants": results,
            "best_variant": best_variant.name,
            "total_tests": len(self.test_history)
        }

    def print_report(self):
        """打印测试报告"""
        results = self.get_results()

        print(f"\n{'='*50}")
        print(f"A/B测试报告: {self.tool_name}")
        print(f"{'='*50}\n")

        for name, stats in results["variants"].items():
            print(f"变体: {name}")
            print(f"  描述: {stats['description'][:50]}...")
            print(f"  选择次数: {stats['selections']}")
            print(f"  成功次数: {stats['successes']}")
            print(f"  成功率: {stats['success_rate']}")
            print()

        print(f"最佳变体: {results['best_variant']}")
        print(f"总测试次数: {results['total_tests']}")

# 使用示例
ab_test = ToolDescriptionABTest("search_tool")

ab_test.add_variant(
    "v1_original",
    "搜索相关内容"
)

ab_test.add_variant(
    "v2_enhanced",
    "根据关键词搜索相关内容，支持模糊匹配和高级筛选"
)

ab_test.add_variant(
    "v3_detailed",
    "强大的搜索工具，支持多种搜索模式：精确匹配、模糊搜索、语义搜索。返回相关度排序的结果列表。"
)

 # 模拟测试
 for _ in range(100):
     variant = ab_test.select_variant()
     success = random.random() < 0.7  # 70%成功率
     ab_test.record_selection(variant.name, success)

 # 打印报告
 ab_test.print_report()
 ```

工具选择是 Agent 决策中的关键环节。下面的交互式演示展示了完整的工具选择决策过程：

<div data-component="ToolSelectionDemoV7"></div>

 ### 6.5.4 描述模板生成器

```python
from typing import Dict, List, Optional, Type
from pydantic import BaseModel
import inspect

class DescriptionTemplateGenerator:
    """工具描述模板生成器"""

    # 常用动词映射
    VERB_MAP = {
        "get": "获取",
        "create": "创建",
        "update": "更新",
        "delete": "删除",
        "search": "搜索",
        "calculate": "计算",
        "validate": "验证",
        "convert": "转换",
        "analyze": "分析",
        "process": "处理",
    }

    @classmethod
    def generate_from_function(cls, func) -> Dict:
        """从函数生成描述模板"""
        # 解析函数信息
        name = func.__name__
        docstring = func.__doc__ or ""
        sig = inspect.signature(func)

        # 从函数名推断动作
        action = name.split("_")[0] if "_" in name else name
        action_cn = cls.VERB_MAP.get(action, action)

        # 解析参数
        params_info = []
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            param_type = param.annotation
            param_type_name = "any"
            if param_type != inspect.Parameter.empty:
                param_type_name = getattr(param_type, '__name__', str(param_type))

            params_info.append({
                "name": param_name,
                "type": param_type_name,
                "default": param.default if param.default != inspect.Parameter.empty else None,
                "required": param.default == inspect.Parameter.empty
            })

        # 生成模板
        template = {
            "name": name,
            "action": action,
            "action_cn": action_cn,
            "original_docstring": docstring,
            "params": params_info,
            "suggested_description": f"{action_cn}{name.replace('_', '')}",
            "suggested_params_desc": {},
            "suggested_examples": [],
            "suggested_limitations": []
        }

        # 为每个参数生成描述建议
        for param in params_info:
            template["suggested_params_desc"][param["name"]] = {
                "description": f"{param['name']}参数",
                "type": param["type"],
                "required": param["required"],
                "example": cls._generate_example(param["type"])
            }

        # 生成示例
        example_args = ", ".join([
            f"{p['name']}={cls._generate_example(p['type'])}"
            for p in params_info[:3]  # 只取前3个参数
        ])
        template["suggested_examples"].append(
            f"result = {name}({example_args})"
        )

        return template

    @classmethod
    def _generate_example(cls, type_name: str) -> str:
        """根据类型生成示例值"""
        examples = {
            "str": '"example"',
            "int": "42",
            "float": "3.14",
            "bool": "True",
            "List": "[1, 2, 3]",
            "Dict": '{"key": "value"}',
        }
        return examples.get(type_name, "None")

    @classmethod
    def render_markdown(cls, template: Dict) -> str:
        """渲染Markdown格式的描述"""
        lines = [
            f"# {template['name']}",
            "",
            f"## 描述",
            template["suggested_description"],
            "",
            "## 参数",
            ""
        ]

        for param in template["params"]:
            required = "必需" if param["required"] else "可选"
            lines.append(f"- **{param['name']}** ({param['type']}, {required})")

            if param["name"] in template["suggested_params_desc"]:
                desc = template["suggested_params_desc"][param["name"]]
                lines.append(f"  - 描述: {desc['description']}")
                lines.append(f"  - 示例: {desc['example']}")
            lines.append("")

        lines.extend([
            "## 使用示例",
            "```python",
            *template["suggested_examples"],
            "```",
            "",
            "## 限制",
            *template.get("suggested_limitations", ["暂无限制"]),
        ])

        return "\n".join(lines)

# 使用示例
def sample_tool(query: str, max_results: int = 10, filters: Dict = None) -> List:
    """搜索工具"""
    pass

template = DescriptionTemplateGenerator.generate_from_function(sample_tool)
markdown = DescriptionTemplateGenerator.render_markdown(template)
print(markdown)
```

> "好的工具描述就像好的文档：清晰、完整、易懂。" —— 文档工程原则

---

## 6.6 编排模式（Tool Orchestration Patterns）

编排模式决定了多个工具如何协同工作来完成复杂任务。本节介绍常见的编排模式及其应用。

### 6.6.1 编排模式概览

```
编排模式分类
├── 顺序模式 (Sequential)
│   ├── 管道模式 (Pipeline)
│   └── 链式模式 (Chain)
├── 并行模式 (Parallel)
│   ├── 扇出模式 (Fan-out)
│   └── 扇入模式 (Fan-in)
├── 条件模式 (Conditional)
│   ├── 路由模式 (Router)
│   └── 守卫模式 (Guard)
├── 循环模式 (Loop)
│   ├── 重试模式 (Retry)
│   └── 迭代模式 (Iteration)
└── 混合模式 (Hybrid)
    ├── DAG模式 (Directed Acyclic Graph)
    └── 事件驱动模式 (Event-driven)
```

### 6.6.2 顺序编排

```python
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
import time

@dataclass
class PipelineStep:
    """管道步骤"""
    name: str
    tool: Callable
    description: str
    timeout: float = 30.0
    retry_count: int = 0

class SequentialOrchestrator:
    """顺序编排器：工具按顺序执行"""

    def __init__(self, name: str):
        self.name = name
        self.steps: List[PipelineStep] = []
        self._execution_history: List[Dict] = []

    def add_step(
        self,
        name: str,
        tool: Callable,
        description: str = "",
        timeout: float = 30.0
    ) -> 'SequentialOrchestrator':
        """添加步骤"""
        self.steps.append(PipelineStep(
            name=name,
            tool=tool,
            description=description,
            timeout=timeout
        ))
        return self

    def execute(self, initial_input: Any) -> Dict[str, Any]:
        """执行管道"""
        current_data = initial_input
        start_time = time.time()

        for step in self.steps:
            step_start = time.time()

            try:
                # 执行步骤
                if callable(step.tool):
                    result = step.tool(current_data)
                else:
                    result = step.tool

                # 记录执行信息
                step_duration = time.time() - step_start
                self._execution_history.append({
                    "step": step.name,
                    "input": str(current_data)[:100],
                    "output": str(result)[:100],
                    "duration": step_duration,
                    "status": "success"
                })

                current_data = result

            except Exception as e:
                # 记录错误
                step_duration = time.time() - step_start
                self._execution_history.append({
                    "step": step.name,
                    "input": str(current_data)[:100],
                    "error": str(e),
                    "duration": step_duration,
                    "status": "error"
                })

                # 处理重试
                if step.retry_count > 0:
                    step.retry_count -= 1
                    # 重新执行当前步骤（简化实现）
                    continue

                return {
                    "success": False,
                    "error": f"步骤 '{step.name}' 失败: {str(e)}",
                    "last_successful_step": step.name,
                    "execution_history": self._execution_history
                }

        total_duration = time.time() - start_time

        return {
            "success": True,
            "result": current_data,
            "total_steps": len(self.steps),
            "total_duration": total_duration,
            "execution_history": self._execution_history
        }

    def visualize(self) -> str:
        """可视化管道结构"""
        lines = [f"管道: {self.name}", "=" * 40]

        for i, step in enumerate(self.steps):
            connector = "↓" if i < len(self.steps) - 1 else "✓"
            lines.append(f"{i+1}. {step.name}")
            lines.append(f"   {step.description}")
            lines.append(f"   {connector}")

        return "\n".join(lines)

# 使用示例
def validate_input(data: Dict) -> Dict:
    """验证输入数据"""
    if "query" not in data:
        raise ValueError("缺少 'query' 字段")
    return data

def enrich_data(data: Dict) -> Dict:
    """丰富数据"""
    data["enriched"] = True
    data["timestamp"] = time.time()
    return data

def process_data(data: Dict) -> Dict:
    """处理数据"""
    data["processed"] = True
    data["result"] = f"处理结果: {data.get('query', 'N/A')}"
    return data

def format_output(data: Dict) -> str:
    """格式化输出"""
    return f"最终结果: {data.get('result', '无结果')}"

# 构建管道
pipeline = SequentialOrchestrator("数据处理管道")
pipeline.add_step("验证", validate_input, "验证输入数据格式")
pipeline.add_step("丰富", enrich_data, "添加额外信息")
pipeline.add_step("处理", process_data, "执行核心处理逻辑")
pipeline.add_step("格式化", format_output, "格式化最终输出")

# 可视化
print(pipeline.visualize())

# 执行管道
result = pipeline.execute({"query": "测试查询"})
print(f"\n执行结果: {result}")
```

### 6.6.3 并行编排

```python
import asyncio
from typing import List, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time

@dataclass
class ParallelTask:
    """并行任务"""
    name: str
    tool: Callable
    timeout: float = 30.0
    priority: int = 0

class ParallelOrchestrator:
    """并行编排器：工具并行执行"""

    def __init__(self, name: str, max_workers: int = 4):
        self.name = name
        self.max_workers = max_workers
        self.tasks: List[ParallelTask] = []
        self._execution_history: List[Dict] = []

    def add_task(
        self,
        name: str,
        tool: Callable,
        timeout: float = 30.0,
        priority: int = 0
    ) -> 'ParallelOrchestrator':
        """添加并行任务"""
        self.tasks.append(ParallelTask(
            name=name,
            tool=tool,
            timeout=timeout,
            priority=priority
        ))
        return self

    def execute(self, inputs: List[Any]) -> Dict[str, Any]:
        """并行执行所有任务"""
        start_time = time.time()
        results = {}
        errors = {}

        # 按优先级排序
        sorted_tasks = sorted(self.tasks, key=lambda t: t.priority, reverse=True)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {}
            for task, inp in zip(sorted_tasks, inputs):
                future = executor.submit(self._execute_task, task, inp)
                future_to_task[future] = task

            # 收集结果
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result(timeout=task.timeout)
                    results[task.name] = result
                    self._execution_history.append({
                        "task": task.name,
                        "status": "success",
                        "duration": result.get("duration", 0)
                    })
                except Exception as e:
                    errors[task.name] = str(e)
                    self._execution_history.append({
                        "task": task.name,
                        "status": "error",
                        "error": str(e)
                    })

        total_duration = time.time() - start_time

        return {
            "success": len(errors) == 0,
            "results": results,
            "errors": errors,
            "total_tasks": len(self.tasks),
            "successful_tasks": len(results),
            "failed_tasks": len(errors),
            "total_duration": total_duration,
            "execution_history": self._execution_history
        }

    def _execute_task(self, task: ParallelTask, input_data: Any) -> Dict:
        """执行单个任务"""
        start_time = time.time()

        try:
            result = task.tool(input_data)
            duration = time.time() - start_time

            return {
                "task": task.name,
                "result": result,
                "duration": duration,
                "success": True
            }
        except Exception as e:
            duration = time.time() - start_time
            return {
                "task": task.name,
                "error": str(e),
                "duration": duration,
                "success": False
            }

    async def aexecute(self, inputs: List[Any]) -> Dict[str, Any]:
        """异步并行执行"""
        start_time = time.time()
        results = {}
        errors = {}

        async def run_task(task: ParallelTask, input_data: Any):
            try:
                # 模拟异步执行
                await asyncio.sleep(0.1)
                if callable(task.tool):
                    result = task.tool(input_data)
                else:
                    result = task.tool
                return {"task": task.name, "result": result, "success": True}
            except Exception as e:
                return {"task": task.name, "error": str(e), "success": False}

        # 并发执行所有任务
        tasks = [
            run_task(task, inp)
            for task, inp in zip(self.tasks, inputs)
        ]

        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in task_results:
            if isinstance(result, Exception):
                errors["unknown"] = str(result)
            elif result["success"]:
                results[result["task"]] = result["result"]
            else:
                errors[result["task"]] = result["error"]

        total_duration = time.time() - start_time

        return {
            "success": len(errors) == 0,
            "results": results,
            "errors": errors,
            "total_duration": total_duration
        }

# 使用示例
def fetch_weather(city: str) -> Dict:
    """获取天气"""
    time.sleep(0.1)  # 模拟网络延迟
    return {"city": city, "temp": 22, "condition": "晴"}

def fetch_news(topic: str) -> Dict:
    """获取新闻"""
    time.sleep(0.2)  # 模拟网络延迟
    return {"topic": topic, "count": 5, "headlines": ["新闻1", "新闻2"]}

def fetch_stocks(symbols: List[str]) -> Dict:
    """获取股票信息"""
    time.sleep(0.15)  # 模拟网络延迟
    return {"symbols": symbols, "prices": {"AAPL": 150.0, "GOOGL": 2800.0}}

# 构建并行编排器
parallel = ParallelOrchestrator("信息聚合", max_workers=3)
parallel.add_task("天气", fetch_weather, timeout=10)
parallel.add_task("新闻", fetch_news, timeout=15)
parallel.add_task("股票", fetch_stocks, timeout=10)

# 执行
result = parallel.execute(["北京", "AI", ["AAPL", "GOOGL"]])
print(f"成功率: {result['successful_tasks']}/{result['total_tasks']}")
print(f"总耗时: {result['total_duration']:.2f}秒")
```

### 6.6.4 条件编排

```python
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class RouteCondition(Enum):
    """路由条件"""
    EQUALS = "equals"
    CONTAINS = "contains"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    REGEX = "regex"

@dataclass
class Route:
    """路由规则"""
    name: str
    condition: RouteCondition
    field: str
    value: Any
    target_tool: Callable
    description: str = ""

class ConditionalOrchestrator:
    """条件编排器：根据条件选择工具"""

    def __init__(self, name: str):
        self.name = name
        self.routes: List[Route] = []
        self.default_tool: Optional[Callable] = None
        self._execution_history: List[Dict] = []

    def add_route(
        self,
        name: str,
        condition: RouteCondition,
        field: str,
        value: Any,
        target_tool: Callable,
        description: str = ""
    ) -> 'ConditionalOrchestrator':
        """添加路由规则"""
        self.routes.append(Route(
            name=name,
            condition=condition,
            field=field,
            value=value,
            target_tool=target_tool,
            description=description
        ))
        return self

    def set_default(self, tool: Callable) -> 'ConditionalOrchestrator':
        """设置默认工具"""
        self.default_tool = tool
        return self

    def evaluate_condition(self, data: Dict, route: Route) -> bool:
        """评估条件"""
        field_value = data.get(route.field)

        if field_value is None:
            return False

        if route.condition == RouteCondition.EQUALS:
            return field_value == route.value
        elif route.condition == RouteCondition.CONTAINS:
            return route.value in str(field_value)
        elif route.condition == RouteCondition.GREATER_THAN:
            return float(field_value) > float(route.value)
        elif route.condition == RouteCondition.LESS_THAN:
            return float(field_value) < float(route.value)
        elif route.condition == RouteCondition.REGEX:
            import re
            return bool(re.match(route.value, str(field_value)))

        return False

    def execute(self, data: Dict) -> Dict[str, Any]:
        """执行条件编排"""
        import time
        start_time = time.time()

        # 查找匹配的路由
        matched_route = None
        for route in self.routes:
            if self.evaluate_condition(data, route):
                matched_route = route
                break

        # 执行工具
        if matched_route:
            tool = matched_route.target_tool
            route_name = matched_route.name
        elif self.default_tool:
            tool = self.default_tool
            route_name = "default"
        else:
            return {
                "success": False,
                "error": "没有匹配的路由且未设置默认工具",
                "data": data
            }

        try:
            result = tool(data)
            duration = time.time() - start_time

            self._execution_history.append({
                "route": route_name,
                "input": str(data)[:100],
                "output": str(result)[:100],
                "duration": duration,
                "status": "success"
            })

            return {
                "success": True,
                "route": route_name,
                "result": result,
                "duration": duration
            }

        except Exception as e:
            duration = time.time() - start_time

            self._execution_history.append({
                "route": route_name,
                "input": str(data)[:100],
                "error": str(e),
                "duration": duration,
                "status": "error"
            })

            return {
                "success": False,
                "route": route_name,
                "error": str(e),
                "duration": duration
            }

    def visualize_routes(self) -> str:
        """可视化路由规则"""
        lines = [f"条件编排器: {self.name}", "=" * 50]

        for i, route in enumerate(self.routes):
            lines.append(f"\n路由 {i+1}: {route.name}")
            lines.append(f"  条件: {route.field} {route.condition.value} {route.value}")
            lines.append(f"  目标: {route.target_tool.__name__}")
            if route.description:
                lines.append(f"  描述: {route.description}")

        if self.default_tool:
            lines.append(f"\n默认工具: {self.default_tool.__name__}")

        return "\n".join(lines)

# 使用示例
def handle_urgent(data: Dict) -> str:
    """处理紧急任务"""
    return f"紧急处理: {data.get('content', '')}"

def handle_normal(data: Dict) -> str:
    """处理普通任务"""
    return f"普通处理: {data.get('content', '')}"

def handle_low_priority(data: Dict) -> str:
    """处理低优先级任务"""
    return f"低优先级处理: {data.get('content', '')}"

def handle_unknown(data: Dict) -> str:
    """处理未知类型"""
    return f"未知类型处理: {data.get('content', '')}"

# 构建条件编排器
orchestrator = ConditionalOrchestrator("任务路由")

orchestrator.add_route(
    name="紧急任务",
    condition=RouteCondition.EQUALS,
    field="priority",
    value="urgent",
    target_tool=handle_urgent,
    description="处理紧急任务"
)

orchestrator.add_route(
    name="普通任务",
    condition=RouteCondition.EQUALS,
    field="priority",
    value="normal",
    target_tool=handle_normal,
    description="处理普通任务"
)

orchestrator.add_route(
    name="低优先级任务",
    condition=RouteCondition.EQUALS,
    field="priority",
    value="low",
    target_tool=handle_low_priority,
    description="处理低优先级任务"
)

orchestrator.set_default(handle_unknown)

# 可视化
print(orchestrator.visualize_routes())

# 测试路由
test_cases = [
    {"priority": "urgent", "content": "系统崩溃"},
    {"priority": "normal", "content": "功能请求"},
    {"priority": "low", "content": "界面优化"},
    {"priority": "unknown", "content": "其他问题"}
]

for case in test_cases:
    result = orchestrator.execute(case)
    print(f"\n输入: {case}")
    print(f"结果: {result.get('result', result.get('error'))}")
```

### 6.6.5 循环编排

```python
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass
import time

@dataclass
class LoopConfig:
    """循环配置"""
    max_iterations: int = 10
    timeout: float = 60.0
    convergence_threshold: float = 0.01
    early_stop: bool = True

class LoopOrchestrator:
    """循环编排器：支持迭代和重试"""

    def __init__(self, name: str, config: LoopConfig = None):
        self.name = name
        self.config = config or LoopConfig()
        self._iteration_history: list = []

    def execute_with_retry(
        self,
        tool: Callable,
        input_data: Any,
        max_retries: int = 3,
        backoff_factor: float = 2.0
    ) -> Dict[str, Any]:
        """带重试的执行"""
        start_time = time.time()
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                result = tool(input_data)
                duration = time.time() - start_time

                self._iteration_history.append({
                    "attempt": attempt + 1,
                    "status": "success",
                    "duration": duration
                })

                return {
                    "success": True,
                    "result": result,
                    "attempts": attempt + 1,
                    "duration": duration
                }

            except Exception as e:
                last_error = e
                duration = time.time() - start_time

                self._iteration_history.append({
                    "attempt": attempt + 1,
                    "status": "error",
                    "error": str(e),
                    "duration": duration
                })

                # 指数退避
                if attempt < max_retries:
                    wait_time = backoff_factor ** attempt
                    time.sleep(min(wait_time, 10))

        return {
            "success": False,
            "error": str(last_error),
            "attempts": max_retries + 1,
            "duration": time.time() - start_time
        }

    def execute_until_convergence(
        self,
        tool: Callable,
        initial_data: Any,
        check_convergence: Callable
    ) -> Dict[str, Any]:
        """执行直到收敛"""
        start_time = time.time()
        current_data = initial_data
        previous_result = None

        for iteration in range(self.config.max_iterations):
            iteration_start = time.time()

            try:
                result = tool(current_data)
                iteration_duration = time.time() - iteration_start

                # 检查收敛
                if previous_result is not None:
                    is_converged = check_convergence(previous_result, result)
                else:
                    is_converged = False

                self._iteration_history.append({
                    "iteration": iteration + 1,
                    "status": "success",
                    "converged": is_converged,
                    "duration": iteration_duration
                })

                # 检查是否应该停止
                if is_converged and self.config.early_stop:
                    return {
                        "success": True,
                        "result": result,
                        "iterations": iteration + 1,
                        "converged": True,
                        "duration": time.time() - start_time
                    }

                previous_result = result
                current_data = result

                # 检查超时
                if time.time() - start_time > self.config.timeout:
                    return {
                        "success": False,
                        "error": "执行超时",
                        "iterations": iteration + 1,
                        "duration": time.time() - start_time
                    }

            except Exception as e:
                iteration_duration = time.time() - iteration_start
                self._iteration_history.append({
                    "iteration": iteration + 1,
                    "status": "error",
                    "error": str(e),
                    "duration": iteration_duration
                })

                return {
                    "success": False,
                    "error": str(e),
                    "iterations": iteration + 1,
                    "duration": time.time() - start_time
                }

        return {
            "success": False,
            "error": "达到最大迭代次数",
            "iterations": self.config.max_iterations,
            "duration": time.time() - start_time
        }

    def visualize_history(self) -> str:
        """可视化执行历史"""
        lines = [f"循环执行历史: {self.name}", "=" * 50]

        for entry in self._iteration_history:
            iteration = entry.get("iteration", entry.get("attempt", "?"))
            status = entry["status"]
            duration = entry.get("duration", 0)

            icon = "✓" if status == "success" else "✗"
            lines.append(f"{icon} 迭代 {iteration}: {duration:.3f}秒")

            if "error" in entry:
                lines.append(f"  错误: {entry['error'][:50]}")
            if entry.get("converged"):
                lines.append(f"  已收敛")

        return "\n".join(lines)

# 使用示例
def iterative_refinement(data: Dict) -> Dict:
    """迭代优化函数"""
    # 模拟优化过程
    if "value" not in data:
        data["value"] = 100
    data["value"] = data["value"] * 0.9  # 每次减少10%
    data["iteration"] = data.get("iteration", 0) + 1
    return data

def check_convergence(prev: Dict, curr: Dict) -> bool:
    """检查是否收敛"""
    prev_val = prev.get("value", 0)
    curr_val = curr.get("value", 0)
    return abs(prev_val - curr_val) < 0.1

# 创建循环编排器
loop = LoopOrchestrator(
    name="优化循环",
    config=LoopConfig(
        max_iterations=20,
        timeout=30.0,
        convergence_threshold=0.1
    )
)

# 执行直到收敛
result = loop.execute_until_convergence(
    tool=iterative_refinement,
    initial_data={"value": 100},
    check_convergence=check_convergence
)

print(f"最终结果: {result}")
print(f"\n{loop.visualize_history()}")
```

### 6.6.6 DAG 编排

```python
from typing import Any, Callable, Dict, List, Set
from dataclasses import dataclass, field
from collections import defaultdict
import time

@dataclass
class DAGNode:
    """DAG节点"""
    name: str
    tool: Callable
    dependencies: List[str] = field(default_factory=list)
    description: str = ""

class DAGOrchestrator:
    """DAG编排器：支持复杂的依赖关系"""

    def __init__(self, name: str):
        self.name = name
        self.nodes: Dict[str, DAGNode] = {}
        self._execution_order: List[str] = []
        self._execution_history: List[Dict] = []

    def add_node(
        self,
        name: str,
        tool: Callable,
        dependencies: List[str] = None,
        description: str = ""
    ) -> 'DAGOrchestrator':
        """添加节点"""
        self.nodes[name] = DAGNode(
            name=name,
            tool=tool,
            dependencies=dependencies or [],
            description=description
        )
        return self

    def _topological_sort(self) -> List[str]:
        """拓扑排序"""
        in_degree = defaultdict(int)
        graph = defaultdict(list)

        for name, node in self.nodes.items():
            for dep in node.dependencies:
                graph[dep].append(name)
                in_degree[name] += 1

        queue = [name for name in self.nodes if in_degree[name] == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)

            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.nodes):
            raise ValueError("检测到循环依赖")

        return order

    def _check_dependencies_met(self, node_name: str, completed: Set[str]) -> bool:
        """检查依赖是否已满足"""
        node = self.nodes[node_name]
        return all(dep in completed for dep in node.dependencies)

    def execute(self, initial_inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """执行DAG"""
        start_time = time.time()
        results = {}
        completed = set()
        errors = {}

        # 获取执行顺序
        try:
            execution_order = self._topological_sort()
        except ValueError as e:
            return {
                "success": False,
                "error": str(e)
            }

        # 执行每个节点
        for node_name in execution_order:
            node = self.nodes[node_name]

            # 检查依赖
            if not self._check_dependencies_met(node_name, completed):
                errors[node_name] = "依赖未满足"
                continue

            # 准备输入
            if initial_inputs and node_name in initial_inputs:
                input_data = initial_inputs[node_name]
            else:
                # 收集依赖节点的输出
                input_data = {}
                for dep in node.dependencies:
                    if dep in results:
                        input_data[dep] = results[dep]

            # 执行节点
            node_start = time.time()
            try:
                result = node.tool(input_data) if input_data else node.tool()
                node_duration = time.time() - node_start

                results[node_name] = result
                completed.add(node_name)

                self._execution_history.append({
                    "node": node_name,
                    "status": "success",
                    "duration": node_duration
                })

            except Exception as e:
                node_duration = time.time() - node_start
                errors[node_name] = str(e)

                self._execution_history.append({
                    "node": node_name,
                    "status": "error",
                    "error": str(e),
                    "duration": node_duration
                })

        total_duration = time.time() - start_time

        return {
            "success": len(errors) == 0,
            "results": results,
            "errors": errors,
            "completed_nodes": list(completed),
            "total_nodes": len(self.nodes),
            "total_duration": total_duration,
            "execution_history": self._execution_history
        }

    def visualize(self) -> str:
        """可视化DAG结构"""
        lines = [f"DAG编排器: {self.name}", "=" * 50, ""]

        # 显示节点
        for name, node in self.nodes.items():
            deps = ", ".join(node.dependencies) if node.dependencies else "无"
            lines.append(f"节点: {name}")
            lines.append(f"  工具: {node.tool.__name__}")
            lines.append(f"  依赖: {deps}")
            if node.description:
                lines.append(f"  描述: {node.description}")
            lines.append("")

        # 显示执行顺序
        try:
            order = self._topological_sort()
            lines.append("执行顺序:")
            lines.append(" → ".join(order))
        except ValueError:
            lines.append("⚠️ 检测到循环依赖，无法排序")

        return "\n".join(lines)

# 使用示例
def prepare_data(input_data: Dict) -> Dict:
    """准备数据"""
    return {"prepared": True, "data": input_data.get("raw_data", [])}

def analyze_data(input_data: Dict) -> Dict:
    """分析数据"""
    return {"analyzed": True, "insights": ["洞察1", "洞察2"]}

def generate_report(input_data: Dict) -> Dict:
    """生成报告"""
    return {"report": "报告内容", "format": "markdown"}

def send_notification(input_data: Dict) -> str:
    """发送通知"""
    return f"通知已发送: {input_data.get('report', '')}"

# 构建DAG
dag = DAGOrchestrator("数据处理流程")
dag.add_node("prepare", prepare_data, description="准备数据")
dag.add_node("analyze", analyze_data, dependencies=["prepare"], description="分析数据")
dag.add_node("report", generate_report, dependencies=["analyze"], description="生成报告")
dag.add_node("notify", send_notification, dependencies=["report"], description="发送通知")

# 可视化
print(dag.visualize())

# 执行
result = dag.execute({"prepare": {"raw_data": [1, 2, 3]}})
print(f"\n执行结果: {result}")
```

> "编排模式是工具协作的艺术，选择合适的模式决定了系统的效率和可靠性。" —— 分布式系统设计原则

---

## 6.7 动态工具生成

动态工具生成允许在运行时根据上下文创建新的工具，为 Agent 提供更大的灵活性。

### 6.7.1 基于配置的工具生成

```python
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field, create_model
import json

class DynamicToolConfig(BaseModel):
    """动态工具配置"""
    name: str = Field(description="工具名称")
    description: str = Field(description="工具描述")
    parameters: Dict[str, Dict] = Field(
        description="参数定义，格式: {param_name: {type, description, default}}"
    )
    function_code: str = Field(description="函数代码")
    return_type: str = Field(default="str", description="返回类型")

class DynamicToolGenerator:
    """动态工具生成器"""

    def __init__(self):
        self._generated_tools: Dict[str, Any] = {}
        self._tool_configs: Dict[str, DynamicToolConfig] = {}

    def create_tool_from_config(self, config: DynamicToolConfig) -> Callable:
        """从配置创建工具"""
        # 构建参数模型
        fields = {}
        for param_name, param_def in config.parameters.items():
            param_type = self._get_type(param_def.get("type", "str"))
            default = param_def.get("default")
            description = param_def.get("description", "")

            if default is None:
                fields[param_name] = (param_type, Field(description=description))
            else:
                fields[param_name] = (
                    param_type,
                    Field(default=default, description=description)
                )

        # 创建输入模型
        InputModel = create_model(f"{config.name}Input", **fields)

        # 编译并执行函数代码
        local_vars = {}
        exec(config.function_code, globals(), local_vars)

        # 找到主函数
        main_func = None
        for name, obj in local_vars.items():
            if callable(obj) and not name.startswith('_'):
                main_func = obj
                break

        if main_func is None:
            raise ValueError("未找到可调用的主函数")

        # 包装函数以支持参数验证
        def wrapper(**kwargs):
            try:
                # 验证参数
                input_data = InputModel(**kwargs)
                return main_func(**input_data.model_dump())
            except Exception as e:
                return f"工具执行错误: {str(e)}"

        # 设置元信息
        wrapper.__name__ = config.name
        wrapper.__doc__ = config.description

        # 保存工具
        self._generated_tools[config.name] = wrapper
        self._tool_configs[config.name] = config

        return wrapper

    def _get_type(self, type_str: str) -> type:
        """获取类型对象"""
        type_map = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
        }
        return type_map.get(type_str, str)

    def create_tool_from_natural_language(
        self,
        name: str,
        description: str,
        example_usage: str
    ) -> Callable:
        """从自然语言描述创建工具（简化版）"""
        # 这里可以集成LLM来解析自然语言并生成代码
        # 简化实现：创建一个简单的包装器

        def simple_tool(**kwargs):
            """简单工具实现"""
            return {
                "tool": name,
                "input": kwargs,
                "description": description,
                "status": "executed"
            }

        simple_tool.__name__ = name
        simple_tool.__doc__ = description

        self._generated_tools[name] = simple_tool
        return simple_tool

    def get_tool(self, name: str) -> Optional[Callable]:
        """获取已生成的工具"""
        return self._generated_tools.get(name)

    def list_tools(self) -> List[str]:
        """列出所有已生成的工具"""
        return list(self._generated_tools.keys())

    def get_tool_info(self, name: str) -> Dict:
        """获取工具信息"""
        if name not in self._tool_configs:
            return {"error": f"工具 {name} 不存在"}

        config = self._tool_configs[name]
        return {
            "name": config.name,
            "description": config.description,
            "parameters": config.parameters,
            "return_type": config.return_type
        }

# 使用示例
generator = DynamicToolGenerator()

# 从配置创建工具
config = DynamicToolConfig(
    name="text_formatter",
    description="格式化文本内容",
    parameters={
        "text": {"type": "str", "description": "要格式化的文本"},
        "format": {"type": "str", "description": "格式类型", "default": "upper"},
        "max_length": {"type": "int", "description": "最大长度", "default": 100}
    },
    function_code="""
def format_text(text: str, format: str = "upper", max_length: int = 100) -> str:
    if format == "upper":
        result = text.upper()
    elif format == "lower":
        result = text.lower()
    elif format == "title":
        result = text.title()
    else:
        result = text

    if len(result) > max_length:
        result = result[:max_length] + "..."

    return result
"""
)

formatter = generator.create_tool_from_config(config)

# 使用工具
result = generator.get_tool("text_formatter")(
    text="hello world",
    format="title",
    max_length=15
)
print(f"格式化结果: {result}")

# 列出所有工具
print(f"已生成工具: {generator.list_tools()}")
```

### 6.7.2 基于上下文的工具生成

```python
from typing import Any, Callable, Dict, List
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class ContextualToolGenerator:
    """基于上下文的工具生成器"""

    def __init__(self):
        self._context: Dict[str, Any] = {}
        self._tool_templates: Dict[str, Dict] = {}

    def set_context(self, key: str, value: Any):
        """设置上下文"""
        self._context[key] = value

    def add_tool_template(self, name: str, template: Dict):
        """添加工具模板"""
        self._tool_templates[name] = template

    def generate_from_template(
        self,
        template_name: str,
        context_overrides: Dict = None
    ) -> StructuredTool:
        """从模板生成工具"""
        if template_name not in self._tool_templates:
            raise ValueError(f"模板 {template_name} 不存在")

        template = self._tool_templates[template_name]
        context = {**self._context, **(context_overrides or {})}

        # 替换模板中的占位符
        processed_template = self._process_template(template, context)

        # 创建工具
        tool_config = processed_template.get("config", {})
        tool_code = processed_template.get("code", "")

        # 动态生成工具
        local_vars = {}
        exec(tool_code, globals(), local_vars)

        # 获取主函数
        main_func = None
        for name, obj in local_vars.items():
            if callable(obj) and not name.startswith('_'):
                main_func = obj
                break

        if main_func is None:
            raise ValueError("模板中未找到可调用函数")

        return StructuredTool.from_function(
            func=main_func,
            name=tool_config.get("name", template_name),
            description=tool_config.get("description", f"基于模板 {template_name} 生成的工具")
        )

    def _process_template(self, template: Dict, context: Dict) -> Dict:
        """处理模板，替换占位符"""
        processed = {}
        for key, value in template.items():
            if isinstance(value, str):
                # 替换占位符 {{variable}}
                for ctx_key, ctx_value in context.items():
                    placeholder = f"{{{{{ctx_key}}}}}"
                    value = value.replace(placeholder, str(ctx_value))
                processed[key] = value
            elif isinstance(value, dict):
                processed[key] = self._process_template(value, context)
            else:
                processed[key] = value
        return processed

    def generate_from_context(self, task_description: str) -> List[StructuredTool]:
        """根据任务描述和上下文生成工具"""
        # 根据上下文分析需要的工具类型
        tools = []

        # 示例：根据上下文中的数据类型生成相应工具
        if "data" in self._context:
            data = self._context["data"]

            if isinstance(data, list):
                # 生成列表处理工具
                tools.append(self._create_list_tool(data))
            elif isinstance(data, dict):
                # 生成字典处理工具
                tools.append(self._create_dict_tool(data))

        if "api_base" in self._context:
            # 生成API调用工具
            tools.append(self._create_api_tool())

        return tools

    def _create_list_tool(self, data: List) -> StructuredTool:
        """创建列表处理工具"""
        def list_processor(action: str, index: int = 0, value: Any = None) -> str:
            """处理列表数据"""
            if action == "get":
                return str(data[index]) if index < len(data) else "索引越界"
            elif action == "length":
                return str(len(data))
            elif action == "sum":
                return str(sum(x for x in data if isinstance(x, (int, float))))
            elif action == "filter":
                return str([x for x in data if x])
            else:
                return f"未知操作: {action}"

        return StructuredTool.from_function(
            func=list_processor,
            name="list_processor",
            description="处理列表数据"
        )

    def _create_dict_tool(self, data: Dict) -> StructuredTool:
        """创建字典处理工具"""
        def dict_processor(action: str, key: str = "", value: Any = None) -> str:
            """处理字典数据"""
            if action == "get":
                return str(data.get(key, "键不存在"))
            elif action == "keys":
                return str(list(data.keys()))
            elif action == "values":
                return str(list(data.values()))
            elif action == "length":
                return str(len(data))
            else:
                return f"未知操作: {action}"

        return StructuredTool.from_function(
            func=dict_processor,
            name="dict_processor",
            description="处理字典数据"
        )

    def _create_api_tool(self) -> StructuredTool:
        """创建API调用工具"""
        api_base = self._context.get("api_base", "")

        def api_caller(endpoint: str, method: str = "GET", params: str = "{}") -> str:
            """调用API"""
            import json as json_lib
            try:
                parsed_params = json_lib.loads(params)
            except:
                parsed_params = {}

            return json_lib.dumps({
                "url": f"{api_base}{endpoint}",
                "method": method,
                "params": parsed_params,
                "status": "模拟调用"
            }, ensure_ascii=False)

        return StructuredTool.from_function(
            func=api_caller,
            name="api_caller",
            description="调用API接口"
        )

# 使用示例
generator = ContextualToolGenerator()

# 设置上下文
generator.set_context("data", [1, 2, 3, 4, 5])
generator.set_context("api_base", "https://api.example.com")

# 根据上下文生成工具
tools = generator.generate_from_context("处理数据并调用API")

for tool in tools:
    print(f"生成工具: {tool.name}")
    print(f"  描述: {tool.description}")
    print(f"  参数: {list(tool.args.keys())}")
    print()

# 使用生成的工具
if tools:
    result = tools[0].invoke({"action": "sum"})
    print(f"列表求和结果: {result}")
```

### 6.7.3 运行时工具注册

```python
from typing import Any, Callable, Dict, List
import importlib
import sys

class RuntimeToolRegistry:
    """运行时工具注册表"""

    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._tool_metadata: Dict[str, Dict] = {}
        self._namespaces: Dict[str, List[str]] = {}

    def register_tool(
        self,
        name: str,
        tool: Callable,
        namespace: str = "default",
        metadata: Dict = None
    ):
        """注册工具"""
        self._tools[name] = tool
        self._tool_metadata[name] = metadata or {}

        if namespace not in self._namespaces:
            self._namespaces[namespace] = []
        self._namespaces[namespace].append(name)

    def unregister_tool(self, name: str):
        """注销工具"""
        if name in self._tools:
            del self._tools[name]
            if name in self._tool_metadata:
                del self._tool_metadata[name]

            # 从命名空间中移除
            for ns, tools in self._namespaces.items():
                if name in tools:
                    tools.remove(name)

    def get_tool(self, name: str) -> Optional[Callable]:
        """获取工具"""
        return self._tools.get(name)

    def list_tools(self, namespace: str = None) -> List[str]:
        """列出工具"""
        if namespace:
            return self._namespaces.get(namespace, [])
        return list(self._tools.keys())

    def search_tools(self, query: str) -> List[str]:
        """搜索工具"""
        results = []
        query_lower = query.lower()

        for name, metadata in self._tool_metadata.items():
            # 搜索名称
            if query_lower in name.lower():
                results.append(name)
                continue

            # 搜索描述
            description = metadata.get("description", "")
            if query_lower in description.lower():
                results.append(name)

        return results

    def execute_tool(self, name: str, **kwargs) -> Any:
        """执行工具"""
        tool = self.get_tool(name)
        if tool is None:
            raise ValueError(f"工具 {name} 不存在")

        try:
            return tool(**kwargs)
        except Exception as e:
            return f"工具执行错误: {str(e)}"

    def export_tools(self, filename: str):
        """导出工具配置"""
        import json
        export_data = {
            "tools": {},
            "namespaces": self._namespaces
        }

        for name, metadata in self._tool_metadata.items():
            export_data["tools"][name] = {
                "metadata": metadata,
                "has_callable": name in self._tools
            }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

    def import_tools_from_module(self, module_path: str):
        """从模块导入工具"""
        try:
            module = importlib.import_module(module_path)
            for name in dir(module):
                obj = getattr(module, name)
                if callable(obj) and not name.startswith('_'):
                    self.register_tool(
                        name=name,
                        tool=obj,
                        namespace=module_path,
                        metadata={"source": module_path}
                    )
        except Exception as e:
            print(f"导入模块失败: {str(e)}")

# 使用示例
registry = RuntimeToolRegistry()

# 注册工具
def greet(name: str) -> str:
    """问候函数"""
    return f"你好, {name}!"

def calculate(expression: str) -> str:
    """计算表达式"""
    try:
        return str(eval(expression))
    except:
        return "计算错误"

registry.register_tool("greet", greet, namespace="utils")
registry.register_tool("calculate", calculate, namespace="math")

# 列出工具
print("所有工具:", registry.list_tools())
print("工具命名空间:", registry.list_tools("utils"))

# 执行工具
print("\n问候:", registry.execute_tool("greet", name="张三"))
print("计算:", registry.execute_tool("calculate", expression="2 + 3"))

# 搜索工具
print("\n搜索结果:", registry.search_tools("计算"))
```

> "动态工具生成让 Agent 具备了自我进化的能力，可以根据需要创造新的能力。" —— 自适应系统设计

---

## 6.8 工具注册表与管理

工具注册表是管理工具生命周期的核心组件，负责工具的注册、发现、调用和监控。

### 6.8.1 工具注册表架构

```python
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json

class ToolStatus(Enum):
    """工具状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    ERROR = "error"

class ToolCategory(Enum):
    """工具分类"""
    DATA = "data"
    COMPUTE = "compute"
    NETWORK = "network"
    SYSTEM = "system"
    CUSTOM = "custom"

@dataclass
class ToolMetadata:
    """工具元数据"""
    name: str
    description: str
    category: ToolCategory
    status: ToolStatus = ToolStatus.ACTIVE
    version: str = "1.0.0"
    author: str = ""
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    error_count: int = 0
    avg_execution_time: float = 0.0

@dataclass
class ToolRegistration:
    """工具注册信息"""
    metadata: ToolMetadata
    tool: Callable
    input_schema: Optional[Dict] = None
    output_schema: Optional[Dict] = None
    config: Dict = field(default_factory=dict)

class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools: Dict[str, ToolRegistration] = {}
        self._categories: Dict[ToolCategory, Set[str]] = {
            cat: set() for cat in ToolCategory
        }
        self._tags: Dict[str, Set[str]] = {}
        self._history: List[Dict] = []

    def register(
        self,
        name: str,
        tool: Callable,
        description: str,
        category: ToolCategory = ToolCategory.CUSTOM,
        version: str = "1.0.0",
        author: str = "",
        tags: List[str] = None,
        **kwargs
    ) -> bool:
        """注册工具"""
        if name in self._tools:
            return False

        metadata = ToolMetadata(
            name=name,
            description=description,
            category=category,
            version=version,
            author=author,
            tags=tags or []
        )

        registration = ToolRegistration(
            metadata=metadata,
            tool=tool,
            input_schema=kwargs.get("input_schema"),
            output_schema=kwargs.get("output_schema"),
            config=kwargs.get("config", {})
        )

        self._tools[name] = registration

        # 更新分类索引
        self._categories[category].add(name)

        # 更新标签索引
        for tag in (tags or []):
            if tag not in self._tags:
                self._tags[tag] = set()
            self._tags[tag].add(name)

        # 记录注册历史
        self._history.append({
            "action": "register",
            "tool": name,
            "timestamp": datetime.now().isoformat()
        })

        return True

    def unregister(self, name: str) -> bool:
        """注销工具"""
        if name not in self._tools:
            return False

        registration = self._tools[name]

        # 从分类索引中移除
        self._categories[registration.metadata.category].discard(name)

        # 从标签索引中移除
        for tag in registration.metadata.tags:
            if tag in self._tags:
                self._tags[tag].discard(name)

        # 删除工具
        del self._tools[name]

        # 记录注销历史
        self._history.append({
            "action": "unregister",
            "tool": name,
            "timestamp": datetime.now().isoformat()
        })

        return True

    def get(self, name: str) -> Optional[ToolRegistration]:
        """获取工具注册信息"""
        return self._tools.get(name)

    def list_tools(
        self,
        category: ToolCategory = None,
        status: ToolStatus = None,
        tags: List[str] = None
    ) -> List[str]:
        """列出工具"""
        if category:
            candidates = self._categories.get(category, set())
        elif tags:
            candidates = set()
            for tag in tags:
                if tag in self._tags:
                    candidates |= self._tags[tag]
        else:
            candidates = set(self._tools.keys())

        # 过滤状态
        if status:
            candidates = {
                name for name in candidates
                if self._tools[name].metadata.status == status
            }

        return sorted(candidates)

    def search(self, query: str) -> List[str]:
        """搜索工具"""
        results = []
        query_lower = query.lower()

        for name, registration in self._tools.items():
            # 搜索名称
            if query_lower in name.lower():
                results.append(name)
                continue

            # 搜索描述
            if query_lower in registration.metadata.description.lower():
                results.append(name)
                continue

            # 搜索标签
            for tag in registration.metadata.tags:
                if query_lower in tag.lower():
                    results.append(name)
                    break

        return results

    def update_status(self, name: str, status: ToolStatus) -> bool:
        """更新工具状态"""
        if name not in self._tools:
            return False

        old_status = self._tools[name].metadata.status
        self._tools[name].metadata.status = status
        self._tools[name].metadata.updated_at = datetime.now()

        # 记录状态变更历史
        self._history.append({
            "action": "status_change",
            "tool": name,
            "old_status": old_status.value,
            "new_status": status.value,
            "timestamp": datetime.now().isoformat()
        })

        return True

    def record_usage(self, name: str, execution_time: float, success: bool):
        """记录工具使用"""
        if name not in self._tools:
            return

        metadata = self._tools[name].metadata
        metadata.usage_count += 1

        if not success:
            metadata.error_count += 1

        # 更新平均执行时间
        total_time = metadata.avg_execution_time * (metadata.usage_count - 1)
        metadata.avg_execution_time = (total_time + execution_time) / metadata.usage_count

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            "total_tools": len(self._tools),
            "by_status": {},
            "by_category": {},
            "most_used": [],
            "least_used": [],
            "error_prone": []
        }

        # 按状态统计
        for status in ToolStatus:
            count = sum(
                1 for t in self._tools.values()
                if t.metadata.status == status
            )
            stats["by_status"][status.value] = count

        # 按分类统计
        for category in ToolCategory:
            stats["by_category"][category.value] = len(self._categories[category])

        # 排序工具
        tools_by_usage = sorted(
            self._tools.values(),
            key=lambda t: t.metadata.usage_count,
            reverse=True
        )

        stats["most_used"] = [
            {"name": t.metadata.name, "count": t.metadata.usage_count}
            for t in tools_by_usage[:5]
        ]

        stats["least_used"] = [
            {"name": t.metadata.name, "count": t.metadata.usage_count}
            for t in tools_by_usage[-5:]
        ]

        # 错误率最高的工具
        tools_with_errors = [
            t for t in self._tools.values()
            if t.metadata.error_count > 0
        ]
        tools_with_errors.sort(
            key=lambda t: t.metadata.error_count / max(t.metadata.usage_count, 1),
            reverse=True
        )

        stats["error_prone"] = [
            {
                "name": t.metadata.name,
                "error_rate": t.metadata.error_count / max(t.metadata.usage_count, 1)
            }
            for t in tools_with_errors[:5]
        ]

        return stats

# 使用示例
registry = ToolRegistry()

# 注册工具
def data_fetcher(url: str) -> str:
    """数据获取工具"""
    return f"从 {url} 获取数据"

def data_processor(data: str) -> str:
    """数据处理工具"""
    return f"处理数据: {data}"

def data_analyzer(data: str) -> dict:
    """数据分析工具"""
    return {"analysis": f"分析 {data}"}

registry.register(
    name="data_fetcher",
    tool=data_fetcher,
    description="从URL获取数据",
    category=ToolCategory.DATA,
    tags=["fetch", "http", "data"]
)

registry.register(
    name="data_processor",
    tool=data_processor,
    description="处理数据",
    category=ToolCategory.DATA,
    tags=["process", "transform", "data"]
)

registry.register(
    name="data_analyzer",
    tool=data_analyzer,
    description="分析数据",
    category=ToolCategory.COMPUTE,
    tags=["analyze", "statistics", "data"]
)

# 列出工具
print("所有工具:", registry.list_tools())
print("数据类工具:", registry.list_tools(category=ToolCategory.DATA))

# 搜索工具
print("\n搜索 'data':", registry.search("data"))

# 记录使用
registry.record_usage("data_fetcher", 0.5, True)
registry.record_usage("data_processor", 0.3, True)
registry.record_usage("data_analyzer", 1.2, False)

# 获取统计
stats = registry.get_statistics()
print("\n统计信息:")
print(f"  总工具数: {stats['total_tools']}")
print(f"  按状态: {stats['by_status']}")
print(f"  最常用: {stats['most_used']}")
```

### 6.8.2 工具版本管理

```python
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import semver

@dataclass
class ToolVersion:
    """工具版本"""
    version: str
    tool: Callable
    created_at: datetime
    changelog: str = ""
    is_deprecated: bool = False

class VersionedToolRegistry:
    """带版本管理的工具注册表"""

    def __init__(self):
        self._tools: Dict[str, List[ToolVersion]] = {}
        self._current_versions: Dict[str, str] = {}

    def register(
        self,
        name: str,
        tool: Callable,
        version: str = "1.0.0",
        changelog: str = ""
    ) -> bool:
        """注册工具版本"""
        if name not in self._tools:
            self._tools[name] = []

        # 验证版本号
        try:
            semver.VersionInfo.parse(version)
        except ValueError:
            return False

        # 检查版本是否已存在
        for existing in self._tools[name]:
            if existing.version == version:
                return False

        tool_version = ToolVersion(
            version=version,
            tool=tool,
            created_at=datetime.now(),
            changelog=changelog
        )

        self._tools[name].append(tool_version)

        # 按版本排序
        self._tools[name].sort(
            key=lambda x: semver.VersionInfo.parse(x.version),
            reverse=True
        )

        # 更新当前版本
        if name not in self._current_versions:
            self._current_versions[name] = version

        return True

    def get_version(self, name: str, version: str = None) -> Optional[ToolVersion]:
        """获取指定版本的工具"""
        if name not in self._tools:
            return None

        if version is None:
            # 获取当前版本
            version = self._current_versions.get(name)
            if version is None:
                return None

        for tool_version in self._tools[name]:
            if tool_version.version == version:
                return tool_version

        return None

    def get_latest(self, name: str) -> Optional[ToolVersion]:
        """获取最新版本"""
        if name not in self._tools or not self._tools[name]:
            return None
        return self._tools[name][0]

    def set_current(self, name: str, version: str) -> bool:
        """设置当前版本"""
        if name not in self._tools:
            return False

        # 检查版本是否存在
        for tool_version in self._tools[name]:
            if tool_version.version == version:
                self._current_versions[name] = version
                return True

        return False

    def list_versions(self, name: str) -> List[str]:
        """列出工具的所有版本"""
        if name not in self._tools:
            return []
        return [v.version for v in self._tools[name]]

    def deprecate(self, name: str, version: str) -> bool:
        """废弃指定版本"""
        if name not in self._tools:
            return False

        for tool_version in self._tools[name]:
            if tool_version.version == version:
                tool_version.is_deprecated = True
                return True

        return False

    def get_compatibility_matrix(self) -> Dict:
        """获取兼容性矩阵"""
        matrix = {}
        for name, versions in self._tools.items():
            matrix[name] = {}
            for v in versions:
                matrix[name][v.version] = {
                    "deprecated": v.is_deprecated,
                    "current": self._current_versions.get(name) == v.version
                }
        return matrix

# 使用示例
registry = VersionedToolRegistry()

def calculator_v1(expression: str) -> str:
    """计算器v1"""
    return f"v1计算: {expression}"

def calculator_v2(expression: str, precision: int = 2) -> str:
    """计算器v2，支持精度控制"""
    return f"v2计算: {expression} (精度: {precision})"

def calculator_v3(expression: str, precision: int = 2, validate: bool = True) -> str:
    """计算器v3，支持验证"""
    return f"v3计算: {expression} (精度: {precision}, 验证: {validate})"

# 注册不同版本
registry.register("calculator", calculator_v1, "1.0.0", "初始版本")
registry.register("calculator", calculator_v2, "2.0.0", "添加精度控制")
registry.register("calculator", calculator_v3, "3.0.0", "添加输入验证")

# 废弃旧版本
registry.deprecate("calculator", "1.0.0")
registry.deprecate("calculator", "2.0.0")

# 列出版本
print("计算器版本:", registry.list_versions("calculator"))

# 获取最新版本
latest = registry.get_latest("calculator")
if latest:
    print(f"最新版本: {latest.version}")
    print(f"变更日志: {latest.changelog}")

# 获取兼容性矩阵
matrix = registry.get_compatibility_matrix()
print(f"\n兼容性矩阵: {json.dumps(matrix, indent=2)}")
```

### 6.8.3 工具依赖管理

```python
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from enum import Enum
import json

class DependencyType(Enum):
    """依赖类型"""
    REQUIRED = "required"      # 必需依赖
    OPTIONAL = "optional"      # 可选依赖
    CONFLICT = "conflict"      # 冲突依赖

@dataclass
class Dependency:
    """依赖关系"""
    tool_name: str
    dependency_type: DependencyType
    version_range: Optional[str] = None
    reason: str = ""

class DependencyManager:
    """工具依赖管理器"""

    def __init__(self):
        self._dependencies: Dict[str, List[Dependency]] = {}
        self._dependents: Dict[str, Set[str]] = {}  # 反向依赖

    def add_dependency(
        self,
        tool_name: str,
        dependency_name: str,
        dep_type: DependencyType = DependencyType.REQUIRED,
        version_range: str = None,
        reason: str = ""
    ):
        """添加依赖关系"""
        if tool_name not in self._dependencies:
            self._dependencies[tool_name] = []

        # 检查是否已存在
        for dep in self._dependencies[tool_name]:
            if dep.tool_name == dependency_name:
                return

        dependency = Dependency(
            tool_name=dependency_name,
            dependency_type=dep_type,
            version_range=version_range,
            reason=reason
        )

        self._dependencies[tool_name].append(dependency)

        # 更新反向依赖
        if dependency_name not in self._dependents:
            self._dependents[dependency_name] = set()
        self._dependents[dependency_name].add(tool_name)

    def get_dependencies(self, tool_name: str) -> List[Dependency]:
        """获取工具的依赖"""
        return self._dependencies.get(tool_name, [])

    def get_dependents(self, tool_name: str) -> Set[str]:
        """获取依赖此工具的工具"""
        return self._dependents.get(tool_name, set())

    def resolve_dependencies(self, tool_name: str) -> List[str]:
        """解析依赖顺序（拓扑排序）"""
        visited = set()
        order = []

        def dfs(name: str):
            if name in visited:
                return
            visited.add(name)

            for dep in self.get_dependencies(name):
                if dep.dependency_type == DependencyType.REQUIRED:
                    dfs(dep.tool_name)

            order.append(name)

        dfs(tool_name)
        return order

    def check_conflicts(self, tool_name: str) -> List[str]:
        """检查依赖冲突"""
        conflicts = []
        deps = self.get_dependencies(tool_name)

        required = [d.tool_name for d in deps if d.dependency_type == DependencyType.REQUIRED]
        conflict = [d.tool_name for d in deps if d.dependency_type == DependencyType.CONFLICT]

        # 检查必需依赖是否与冲突依赖重叠
        conflicts.extend(set(required) & set(conflict))

        return conflicts

    def get_impact_analysis(self, tool_name: str) -> Dict:
        """影响分析：移除工具会影响哪些工具"""
        affected = set()
        queue = list(self.get_dependents(tool_name))

        while queue:
            current = queue.pop(0)
            if current in affected:
                continue

            affected.add(current)
            queue.extend(self.get_dependents(current))

        return {
            "tool": tool_name,
            "direct_dependents": list(self.get_dependents(tool_name)),
            "all_affected": list(affected),
            "impact_count": len(affected)
        }

    def visualize_dependencies(self, tool_name: str) -> str:
        """可视化依赖关系"""
        lines = [f"依赖关系: {tool_name}", "=" * 50]

        deps = self.get_dependencies(tool_name)
        if deps:
            lines.append("\n依赖:")
            for dep in deps:
                icon = {
                    DependencyType.REQUIRED: "●",
                    DependencyType.OPTIONAL: "○",
                    DependencyType.CONFLICT: "✗"
                }[dep.dependency_type]
                lines.append(f"  {icon} {dep.tool_name} ({dep.dependency_type.value})")
                if dep.reason:
                    lines.append(f"    原因: {dep.reason}")

        dependents = self.get_dependents(tool_name)
        if dependents:
            lines.append("\n被依赖于:")
            for d in dependents:
                lines.append(f"  ← {d}")

        return "\n".join(lines)

# 使用示例
manager = DependencyManager()

# 添加依赖关系
manager.add_dependency("web_scraper", "http_client", DependencyType.REQUIRED)
manager.add_dependency("web_scraper", "html_parser", DependencyType.REQUIRED)
manager.add_dependency("html_parser", "text_processor", DependencyType.REQUIRED)
manager.add_dependency("data_analyzer", "web_scraper", DependencyType.REQUIRED)
manager.add_dependency("report_generator", "data_analyzer", DependencyType.REQUIRED)
manager.add_dependency("report_generator", "text_processor", DependencyType.REQUIRED)

# 解析依赖顺序
print("report_generator 依赖顺序:")
print(manager.resolve_dependencies("report_generator"))

# 影响分析
print("\n影响分析 (text_processor):")
impact = manager.get_impact_analysis("text_processor")
print(f"  直接依赖者: {impact['direct_dependents']}")
print(f"  所有受影响: {impact['all_affected']}")
print(f"  影响数量: {impact['impact_count']}")

# 可视化
print(f"\n{manager.visualize_dependencies('report_generator')}")
```

> "工具管理是大型 Agent 系统的基石，良好的管理机制确保系统的可维护性和可扩展性。" —— 软件工程原则

---

## 6.9 工具测试与验证

工具测试是确保工具质量和可靠性的重要环节。本节介绍工具测试的最佳实践和自动化测试框架。

### 6.9.1 工具测试策略

```python
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import time
import traceback

class TestStatus(Enum):
    """测试状态"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class TestCase:
    """测试用例"""
    name: str
    input_data: Dict
    expected_output: Any
    expected_status: TestStatus = TestStatus.PASSED
    timeout: float = 30.0
    tags: List[str] = None

@dataclass
class TestResult:
    """测试结果"""
    test_case: TestCase
    actual_output: Any
    status: TestStatus
    duration: float
    error_message: Optional[str] = None

class ToolTestSuite:
    """工具测试套件"""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self.test_cases: List[TestCase] = []
        self.results: List[TestResult] = []

    def add_test_case(
        self,
        name: str,
        input_data: Dict,
        expected_output: Any,
        timeout: float = 30.0,
        tags: List[str] = None
    ):
        """添加测试用例"""
        self.test_cases.append(TestCase(
            name=name,
            input_data=input_data,
            expected_output=expected_output,
            timeout=timeout,
            tags=tags or []
        ))

    def add_edge_case(
        self,
        name: str,
        input_data: Dict,
        expected_status: TestStatus,
        error_pattern: str = None
    ):
        """添加边界测试用例"""
        self.test_cases.append(TestCase(
            name=name,
            input_data=input_data,
            expected_output=None,
            expected_status=expected_status,
            tags=["edge_case"]
        ))

    def run_tests(self, tool: Callable) -> List[TestResult]:
        """运行所有测试"""
        self.results = []

        for test_case in self.test_cases:
            result = self._run_single_test(tool, test_case)
            self.results.append(result)

        return self.results

    def _run_single_test(self, tool: Callable, test_case: TestCase) -> TestResult:
        """运行单个测试"""
        start_time = time.time()

        try:
            # 执行工具
            actual_output = tool(**test_case.input_data)
            duration = time.time() - start_time

            # 验证结果
            if test_case.expected_status == TestStatus.PASSED:
                # 简化的输出比较
                if self._compare_output(actual_output, test_case.expected_output):
                    status = TestStatus.PASSED
                else:
                    status = TestStatus.FAILED
            else:
                status = TestStatus.PASSED

            return TestResult(
                test_case=test_case,
                actual_output=actual_output,
                status=status,
                duration=duration
            )

        except Exception as e:
            duration = time.time() - start_time

            if test_case.expected_status == TestStatus.ERROR:
                status = TestStatus.PASSED
            else:
                status = TestStatus.ERROR

            return TestResult(
                test_case=test_case,
                actual_output=None,
                status=status,
                duration=duration,
                error_message=str(e)
            )

    def _compare_output(self, actual: Any, expected: Any) -> bool:
        """比较输出"""
        if expected is None:
            return True

        if isinstance(expected, dict) and isinstance(actual, dict):
            # 部分匹配
            for key, value in expected.items():
                if key not in actual:
                    return False
                if not self._compare_output(actual[key], value):
                    return False
            return True

        return actual == expected

    def generate_report(self) -> str:
        """生成测试报告"""
        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        errors = sum(1 for r in self.results if r.status == TestStatus.ERROR)
        total = len(self.results)

        lines = [
            f"工具测试报告: {self.tool_name}",
            "=" * 50,
            f"总测试数: {total}",
            f"通过: {passed} ({passed/total*100:.1f}%)" if total > 0 else "通过: 0",
            f"失败: {failed}",
            f"错误: {errors}",
            "",
            "详细结果:",
            "-" * 50
        ]

        for result in self.results:
            icon = "✓" if result.status == TestStatus.PASSED else "✗"
            lines.append(f"{icon} {result.test_case.name} ({result.duration:.3f}s)")

            if result.status != TestStatus.PASSED:
                if result.error_message:
                    lines.append(f"  错误: {result.error_message}")
                lines.append(f"  输入: {result.test_case.input_data}")
                lines.append(f"  期望: {result.test_case.expected_output}")
                lines.append(f"  实际: {result.actual_output}")

        return "\n".join(lines)

# 使用示例
def sample_calculator(a: int, b: int, operation: str = "add") -> int:
    """示例计算器"""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("除数不能为零")
        return a // b
    else:
        raise ValueError(f"未知操作: {operation}")

# 创建测试套件
suite = ToolTestSuite("calculator")

# 添加测试用例
suite.add_test_case(
    name="加法测试",
    input_data={"a": 2, "b": 3, "operation": "add"},
    expected_output=5
)

suite.add_test_case(
    name="减法测试",
    input_data={"a": 5, "b": 3, "operation": "subtract"},
    expected_output=2
)

suite.add_test_case(
    name="乘法测试",
    input_data={"a": 4, "b": 5, "operation": "multiply"},
    expected_output=20
)

suite.add_test_case(
    name="除法测试",
    input_data={"a": 10, "b": 2, "operation": "divide"},
    expected_output=5
)

# 添加边界测试
suite.add_edge_case(
    name="除零测试",
    input_data={"a": 10, "b": 0, "operation": "divide"},
    expected_status=TestStatus.ERROR
)

suite.add_edge_case(
    name="未知操作测试",
    input_data={"a": 1, "b": 2, "operation": "power"},
    expected_status=TestStatus.ERROR
)

# 运行测试
results = suite.run_tests(sample_calculator)

# 生成报告
print(suite.generate_report())
```

### 6.9.2 自动化测试框架

```python
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
import time
import json

@dataclass
class TestConfig:
    """测试配置"""
    tool_name: str
    tool: Callable
    test_data_file: Optional[str] = None
    mock_dependencies: Dict[str, Any] = None
    setup_script: Optional[str] = None
    teardown_script: Optional[str] = None

class AutomatedTestFramework:
    """自动化测试框架"""

    def __init__(self):
        self._test_configs: Dict[str, TestConfig] = {}
        self._global_setup: List[Callable] = []
        self._global_teardown: List[Callable] = []
        self._test_results: Dict[str, List] = {}

    def register_tool(
        self,
        tool_name: str,
        tool: Callable,
        test_data_file: str = None,
        mock_dependencies: Dict[str, Any] = None
    ):
        """注册工具进行测试"""
        self._test_configs[tool_name] = TestConfig(
            tool_name=tool_name,
            tool=tool,
            test_data_file=test_data_file,
            mock_dependencies=mock_dependencies or {}
        )

    def add_global_setup(self, setup_func: Callable):
        """添加全局设置"""
        self._global_setup.append(setup_func)

    def add_global_teardown(self, teardown_func: Callable):
        """添加全局清理"""
        self._global_teardown.append(teardown_func)

    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        results = {}

        # 执行全局设置
        for setup in self._global_setup:
            try:
                setup()
            except Exception as e:
                print(f"全局设置失败: {e}")
                return {"success": False, "error": str(e)}

        # 运行每个工具的测试
        for tool_name, config in self._test_configs.items():
            tool_results = self._run_tool_tests(config)
            results[tool_name] = tool_results
            self._test_results[tool_name] = tool_results

        # 执行全局清理
        for teardown in self._global_teardown:
            try:
                teardown()
            except Exception as e:
                print(f"全局清理失败: {e}")

        return results

    def _run_tool_tests(self, config: TestConfig) -> Dict:
        """运行单个工具的测试"""
        # 加载测试数据
        test_data = self._load_test_data(config)

        results = {
            "total": len(test_data),
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "details": []
        }

        for i, test_case in enumerate(test_data):
            try:
                # 准备测试
                input_data = test_case.get("input", {})
                expected = test_case.get("expected")
                test_name = test_case.get("name", f"test_{i}")

                # 模拟依赖
                self._apply_mocks(config.mock_dependencies)

                # 执行测试
                start_time = time.time()
                actual = config.tool(**input_data)
                duration = time.time() - start_time

                # 验证结果
                if expected is None or actual == expected:
                    results["passed"] += 1
                    status = "passed"
                else:
                    results["failed"] += 1
                    status = "failed"

                results["details"].append({
                    "name": test_name,
                    "status": status,
                    "input": input_data,
                    "expected": expected,
                    "actual": actual,
                    "duration": duration
                })

            except Exception as e:
                results["errors"] += 1
                results["details"].append({
                    "name": test_case.get("name", f"test_{i}"),
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })

        return results

    def _load_test_data(self, config: TestConfig) -> List[Dict]:
        """加载测试数据"""
        if config.test_data_file:
            try:
                with open(config.test_data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass

        # 默认测试数据
        return []

    def _apply_mocks(self, mocks: Dict[str, Any]):
        """应用模拟"""
        for name, mock_value in mocks.items():
            # 这里可以使用 unittest.mock 或其他模拟框架
            pass

    def generate_summary(self) -> str:
        """生成测试摘要"""
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_errors = 0

        lines = [
            "自动化测试摘要",
            "=" * 60,
            f"{'工具名':<30} {'通过':<10} {'失败':<10} {'错误':<10}",
            "-" * 60
        ]

        for tool_name, results in self._test_results.items():
            total = results["total"]
            passed = results["passed"]
            failed = results["failed"]
            errors = results["errors"]

            total_tests += total
            total_passed += passed
            total_failed += failed
            total_errors += errors

            lines.append(f"{tool_name:<30} {passed:<10} {failed:<10} {errors:<10}")

        lines.extend([
            "-" * 60,
            f"{'总计':<30} {total_passed:<10} {total_failed:<10} {total_errors:<10}",
            f"\n通过率: {total_passed/total_tests*100:.1f}%" if total_tests > 0 else "\n通过率: N/A"
        ])

        return "\n".join(lines)

# 使用示例
import traceback

framework = AutomatedTestFramework()

# 注册工具
def sample_tool(x: int, y: int) -> int:
    return x + y

framework.register_tool("adder", sample_tool)

# 运行测试
results = framework.run_all_tests()

# 生成摘要
print(framework.generate_summary())
```

### 6.9.3 性能测试

```python
from typing import Callable, Dict, List, Any
from dataclasses import dataclass
import time
import statistics

@dataclass
class PerformanceMetric:
    """性能指标"""
    name: str
    value: float
    unit: str
    percentile_50: float = 0.0
    percentile_95: float = 0.0
    percentile_99: float = 0.0

class PerformanceTester:
    """性能测试器"""

    def __init__(self, tool: Callable, tool_name: str):
        self.tool = tool
        self.tool_name = tool_name
        self._results: List[float] = []

    def benchmark(
        self,
        input_data: Dict,
        iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, PerformanceMetric]:
        """执行基准测试"""
        self._results = []

        # 预热
        for _ in range(warmup_iterations):
            self.tool(**input_data)

        # 正式测试
        for _ in range(iterations):
            start = time.perf_counter()
            self.tool(**input_data)
            duration = time.perf_counter() - start
            self._results.append(duration)

        return self._calculate_metrics()

    def _calculate_metrics(self) -> Dict[str, PerformanceMetric]:
        """计算性能指标"""
        if not self._results:
            return {}

        sorted_results = sorted(self._results)

        metrics = {
            "latency_avg": PerformanceMetric(
                name="平均延迟",
                value=statistics.mean(sorted_results) * 1000,
                unit="ms"
            ),
            "latency_median": PerformanceMetric(
                name="中位数延迟",
                value=statistics.median(sorted_results) * 1000,
                unit="ms"
            ),
            "latency_std": PerformanceMetric(
                name="延迟标准差",
                value=statistics.stdev(sorted_results) * 1000 if len(sorted_results) > 1 else 0,
                unit="ms"
            ),
            "latency_min": PerformanceMetric(
                name="最小延迟",
                value=min(sorted_results) * 1000,
                unit="ms"
            ),
            "latency_max": PerformanceMetric(
                name="最大延迟",
                value=max(sorted_results) * 1000,
                unit="ms"
            ),
            "throughput": PerformanceMetric(
                name="吞吐量",
                value=1000 / (statistics.mean(sorted_results) * 1000),
                unit="ops/sec"
            )
        }

        # 计算百分位数
        n = len(sorted_results)
        for percentile, key in [(50, "p50"), (95, "p95"), (99, "p99")]:
            idx = int(n * percentile / 100)
            idx = min(idx, n - 1)
            value = sorted_results[idx] * 1000

            if key == "p50":
                metrics["latency_avg"].percentile_50 = value
            elif key == "p95":
                metrics["latency_avg"].percentile_95 = value
            elif key == "p99":
                metrics["latency_avg"].percentile_99 = value

        return metrics

    def generate_report(self) -> str:
        """生成性能报告"""
        if not self._results:
            return "无测试数据"

        metrics = self._calculate_metrics()

        lines = [
            f"性能测试报告: {self.tool_name}",
            "=" * 50,
            f"测试次数: {len(self._results)}",
            "",
            "延迟统计:",
            f"  平均: {metrics['latency_avg'].value:.2f} ms",
            f"  中位数: {metrics['latency_median'].value:.2f} ms",
            f"  标准差: {metrics['latency_std'].value:.2f} ms",
            f"  最小: {metrics['latency_min'].value:.2f} ms",
            f"  最大: {metrics['latency_max'].value:.2f} ms",
            "",
            "吞吐量:",
            f"  {metrics['throughput'].value:.2f} ops/sec",
            "",
            "百分位数:",
            f"  P50: {metrics['latency_avg'].percentile_50:.2f} ms",
            f"  P95: {metrics['latency_avg'].percentile_95:.2f} ms",
            f"  P99: {metrics['latency_avg'].percentile_99:.2f} ms",
        ]

        return "\n".join(lines)

# 使用示例
def sample_computation(n: int) -> int:
    """示例计算"""
    total = 0
    for i in range(n):
        total += i * i
    return total

tester = PerformanceTester(sample_computation, "square_sum")

# 运行基准测试
metrics = tester.benchmark(
    input_data={"n": 1000},
    iterations=200,
    warmup_iterations=20
)

# 生成报告
print(tester.generate_report())
```

> "测试是质量的保证，自动化测试是效率的关键。" —— 软件测试原则

---

## 6.10 本章小结

本章深入探讨了 AI Agent 工具构建与编排的各个方面，从基础的工具设计原则到高级的动态工具生成和编排模式。

### 核心要点回顾

1. **工具设计原则**：单一职责、自描述性、幂等性、容错性、可组合性是工具设计的五大支柱。

2. **工具定义方式**：LangChain 提供了三种主要的工具定义方式：
   - `@tool` 装饰器：简单快捷，适合快速原型
   - `StructuredTool`：灵活可控，适合复杂场景
   - `BaseTool`：完全自定义，适合高级需求

3. **描述优化**：好的工具描述应包含功能说明、参数说明、返回值说明、使用示例和限制说明。通过 A/B 测试可以持续优化描述效果。

4. **编排模式**：根据任务特性选择合适的编排模式：
   - 顺序模式：适合有依赖关系的任务链
   - 并行模式：适合独立的并发任务
   - 条件模式：适合需要根据条件路由的场景
   - 循环模式：适合需要迭代优化的场景
   - DAG 模式：适合复杂的依赖关系图

5. **动态工具生成**：允许在运行时根据上下文创建新工具，为 Agent 提供更大的灵活性。

6. **工具管理**：完善的工具注册、版本管理和依赖管理是大型 Agent 系统的基础。

7. **测试验证**：自动化测试、性能测试和边界测试确保工具的质量和可靠性。

### 工具构建的数学基础

从信息论角度，工具可以表示为：

$$
T: \mathcal{X} \times \Theta \rightarrow \mathcal{Y}
$$

其中 $\mathcal{X}$ 是输入空间，$\Theta$ 是参数空间，$\mathcal{Y}$ 是输出空间。工具的价值在于最大化信息增益：

$$
\text{Value}(T) = I(Y; X | \Theta) = H(Y) - H(Y | X, \Theta)
$$

编排优化可以形式化为：

$$
\max_{\pi} \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{T} \gamma^t r(s_t, a_t) \right]
$$

其中 $\pi$ 是编排策略，$\tau$ 是执行轨迹，$\gamma$ 是折扣因子。

### 最佳实践清单

| 类别 | 最佳实践 |
|------|----------|
| **设计** | 遵循单一职责原则，保持工具功能专一 |
| **命名** | 使用清晰的 `动词_名词` 格式命名工具 |
| **描述** | 提供详细的文档字符串，包含参数和返回值说明 |
| **错误处理** | 添加适当的异常处理，返回友好的错误信息 |
| **测试** | 编写单元测试和集成测试，覆盖边界情况 |
| **文档** | 维护工具文档，包括使用示例和最佳实践 |
| **监控** | 记录工具使用情况和性能指标 |
| **版本** | 使用语义化版本管理工具变更 |
| **安全** | 实施输入验证和访问控制 |
 | **优化** | 持续优化工具性能和描述质量 |
 | **安全** | 实施输入验证和访问控制 |

Agent 规划是复杂多步任务解决的关键。下面的交互式可视化展示了 Agent 规划树的搜索过程：

<div data-component="AgentPlanningDemoV6"></div>

 ---

 ## 6.11 思考题

### 基础题

1. **工具设计原则**
   - 为什么单一职责原则对工具设计如此重要？
   - 如何判断一个工具是否违反了单一职责原则？
   - 请举例说明一个设计不良的工具，并说明如何改进。

2. **@tool 装饰器**
   - `@tool` 装饰器是如何从函数的类型注解和文档字符串生成 JSON Schema 的？
   - 如何为 `@tool` 装饰器添加参数验证？
   - 对比 `@tool` 装饰器和 `StructuredTool.from_function()` 的优缺点。

3. **工具描述**
   - 一个好的工具描述应该包含哪些要素？
   - 如何衡量工具描述的质量？
   - 请为以下工具编写描述：一个查询天气的工具。

### 进阶题

4. **编排模式选择**
   - 给定以下场景，应该选择哪种编排模式？
     - a) 处理一个包含5个步骤的数据处理管道
     - b) 同时调用3个不同的API获取信息
     - c) 根据用户输入类型选择不同的处理逻辑
     - d) 迭代优化一个数值直到收敛
   - 如何设计一个混合编排模式，结合顺序、并行和条件模式？

5. **动态工具生成**
   - 动态工具生成有哪些潜在的安全风险？
   - 如何在动态生成工具时保证类型安全？
   - 设计一个系统，允许用户通过自然语言描述创建自定义工具。

6. **工具管理**
   - 如何设计一个支持热更新的工具注册表？
   - 版本管理中如何处理向后兼容性问题？
   - 设计一个工具依赖解析算法，处理循环依赖的情况。

### 挑战题

7. **性能优化**
   - 设计一个工具缓存系统，支持：
     - 基于时间的过期策略
     - 基于使用频率的淘汰策略
     - 分布式缓存同步
   - 如何监控和优化工具链的整体性能？

8. **测试框架**
   - 设计一个完整的工具测试框架，支持：
     - 自动生成测试用例
     - 基于属性的测试
     - 模糊测试
     - 性能回归测试
   - 如何实现工具测试的持续集成？

9. **架构设计**
   - 设计一个大型 Agent 系统的工具管理架构，要求：
     - 支持工具的动态加载和卸载
     - 支持工具的版本管理和灰度发布
     - 支持工具的依赖管理和冲突检测
     - 支持工具的监控和告警
   - 绘制架构图并说明关键组件的职责。

### 开放性问题

10. **工具进化**
    - 如何让 Agent 自动发现和学习使用新工具？
    - 工具的自动组合和编排有哪些可能的实现方式？
    - 如何评估和比较不同工具的效用？

11. **安全与伦理**
    - 工具系统可能带来哪些安全风险？如何防范？
    - 如何设计一个安全的沙箱环境来执行不受信任的工具？
    - 工具使用的伦理边界在哪里？

12. **未来趋势**
    - 工具系统的发展趋势是什么？
    - 多模态工具（处理文本、图像、音频等）如何设计？
    - 工具系统与大语言模型的结合有哪些新的可能性？

> "思考题的目的不是寻找标准答案，而是激发深入思考和探索。" —— 学习的本质

---

**本章完**

下一章我们将探讨 Agent 的记忆与状态管理，学习如何让 Agent 具备长期记忆和上下文理解能力。

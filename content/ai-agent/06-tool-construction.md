---
title: "第6章：工具构建与编排 — Agent 的手和脚"
description: "掌握 Agent 工具的完整生命周期：@tool 装饰器、StructuredTool、BaseTool、工具描述优化、参数校验、错误处理、工具组合编排、动态工具生成与工具注册表管理。"
date: "2026-06-11"
---

# 第6章：工具构建与编排 — Agent 的手和脚

---

## 6.1 工具的本质与设计原则

### 6.1.1 什么是 Agent 工具？

| LLM 固有能力 | LLM 缺陷 | 工具弥补 |
|:---|:---|:---|
| 文本生成 | 知识截止日期 | 搜索引擎、实时 API |
| 推理 | 无法精确计算 | 计算器、代码执行器 |
| 理解 | 无法访问私有数据 | 数据库查询、文件读取 |
| 规划 | 无法执行真实操作 | API 调用、系统命令 |

### 6.1.2 工具设计的 SOLID 原则

| 原则 | 在工具设计中的含义 |
|:---|:---|
| **单一职责** | 每个工具只做一件事 |
| **开闭原则** | 可扩展新工具，不修改已有工具 |
| **接口隔离** | 工具参数简洁，不暴露内部细节 |
| **依赖倒置** | 依赖抽象接口，不依赖具体实现 |

---

## 6.2 使用 LangChain @tool 装饰器

```python
from langchain_core.tools import tool

@tool
def search_web(query: str, max_results: int = 5) -> str:
    """搜索互联网获取最新信息。

    Args:
        query: 搜索查询字符串，应简洁明确
        max_results: 返回结果数量，默认 5，最大 20
    """
    results = [{"title": f"结果{i}", "url": f"https://example.com/{i}"} for i in range(max_results)]
    return "\n".join([f"[{r['title']}]({r['url']})" for r in results])

@tool
def calculator(expression: str) -> float:
    """计算数学表达式。支持加减乘除、幂运算、三角函数。

    Args:
        expression: 数学表达式，如 "2 + 3 * 4" 或 "sin(3.14159/2)"
    """
    import math
    safe_dict = {"__builtins__": {}, "sin": math.sin, "cos": math.cos, "sqrt": math.sqrt, "pi": math.pi}
    return eval(expression, safe_dict)
```

### StructuredTool 高级用法

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=5, description="最大结果数", ge=1, le=20)
    language: str = Field(default="zh", description="搜索语言")

def search_impl(query: str, max_results: int = 5, language: str = "zh") -> str:
    return f"搜索 '{query}' 的 {max_results} 条结果（语言: {language}）"

search_tool = StructuredTool.from_function(
    func=search_impl,
    name="search_web",
    description="搜索互联网获取最新信息",
    args_schema=SearchInput,
)
```

---

## 6.3 自定义 BaseTool

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class DatabaseQueryInput(BaseModel):
    query: str = Field(description="自然语言查询描述")
    database: str = Field(description="数据库名称")
    limit: int = Field(default=100, description="最大返回行数")

class DatabaseQueryTool(BaseTool):
    name: str = "database_query"
    description: str = "执行数据库查询并返回结果"
    args_schema: type = DatabaseQueryInput
    connection_string: str = ""

    def _run(self, query: str, database: str, limit: int = 100) -> str:
        try:
            import sqlite3
            conn = sqlite3.connect(self.connection_string)
            cursor = conn.execute(f"SELECT * FROM {database} LIMIT {limit}")
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            conn.close()
            result = " | ".join(columns) + "\n" + "-" * 40 + "\n"
            for row in rows[:20]:
                result += " | ".join(str(v) for v in row) + "\n"
            return result
        except Exception as e:
            return f"查询错误：{str(e)}"
```

---

## 6.4 工具描述优化

| 描述质量 | 工具选择准确率 | 参数生成准确率 | 综合准确率 |
|:---|:---|:---|:---|
| 无描述 | 32% | 25% | 8% |
| 简短描述 | 65% | 55% | 36% |
| 详细描述 | 89% | 85% | 76% |
| 详细+示例 | 94% | 92% | 87% |

---

## 6.5 工具编排模式

### 顺序编排

```python
# 顺序：搜索 → 提取 → 总结
# Agent 决策：search_papers → extract_paper_info → summarize_paper
```

### 并行编排

```python
import asyncio

async def parallel_execute(tool_calls):
    tasks = [execute_tool(tc) for tc in tool_calls]
    return await asyncio.gather(*tasks)
```

### 条件编排

```python
def route_by_intent(state):
    intent = state.get("intent", "unknown")
    if intent == "weather": return "weather_tool"
    elif intent == "search": return "search_tool"
    elif intent == "calculate": return "calculator_tool"
    return "fallback_tool"
```

---

## 6.6 动态工具生成

```python
from langchain_core.tools import tool

def create_dynamic_tool(api_spec: dict) -> callable:
    name = api_spec["name"]
    description = api_spec["description"]
    endpoint = api_spec["endpoint"]

    @tool(name=name, description=description)
    def dynamic_tool(**kwargs) -> str:
        import requests
        response = requests.get(endpoint, params=kwargs, timeout=10)
        return response.text if response.status_code == 200 else f"Error: {response.status_code}"

    return dynamic_tool
```

---

## 6.7 工具注册表管理

```python
class ToolRegistry:
    def __init__(self): self._tools = {}

    def register(self, tool, category="general"):
        self._tools[tool.name] = {"tool": tool, "category": category}

    def get(self, name):
        entry = self._tools.get(name)
        return entry["tool"] if entry else None

    def get_by_category(self, category):
        return [v["tool"] for v in self._tools.values() if v["category"] == category]

    def list_tools(self):
        return [{"name": n, "category": v["category"], "description": v["tool"].description}
                for n, v in self._tools.items()]
```

---

## 6.8 本章小结

| 知识点 | 核心要点 |
|:---|:---|
| @tool 装饰器 | 最简单的工具定义方式 |
| StructuredTool | 使用 Pydantic 模型定义复杂 schema |
| BaseTool | 自定义工具基类，支持同步/异步 |
| 描述优化 | 描述质量直接影响调用准确率 |
| 编排模式 | 顺序、并行、条件编排 |
| 动态工具 | 运行时根据 API 规范自动创建 |
| 注册表 | 集中管理工具的注册、分类、查询 |

> **下一章预告**
>
> 在第 7 章中，我们将深入 Agent 的记忆系统。

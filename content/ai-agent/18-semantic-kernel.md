---
title: "第18章：Semantic Kernel 企业级 Agent"
description: "掌握 Semantic Kernel 的 Kernel 核心、Plugin 系统、Planner 规划与 Azure 集成"
updated: "2025-06-15"
---

 # 第18章：Semantic Kernel 企业级 Agent

 > **学习目标**：
 > - 理解 Semantic Kernel 的设计理念与核心架构
 > - 掌握 Kernel 核心概念与配置方法
 > - 学会使用 Plugin 系统（kernel_function）
 > - 熟练掌握 Planner 自动规划机制
 > - 实现 Memory Store 记忆存储
 > - 掌握 Azure 集成与企业级部署
 > - 对比 Semantic Kernel 与 LangChain 的优劣

 下面的交互式演示展示了 Semantic Kernel 的插件系统：

 <div data-component="SemanticKernelPlugins"></div>

 ---

 ## 18.1 Semantic Kernel 概述

### 18.1.1 什么是 Semantic Kernel

Semantic Kernel（SK）是微软开发的开源 AI 编排框架，专为将大语言模型（LLM）集成到企业应用而设计。它提供了强大的 Plugin 系统、自动规划器和内存管理，支持 C# 和 Python 两种语言。

> **核心思想**：通过 Kernel（内核）作为中心枢纽，协调 AI 服务、插件和内存，实现智能应用的编排。

### 18.1.2 Semantic Kernel 架构全景

```
┌─────────────────────────────────────────────────────────┐
│                  Semantic Kernel                         │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Kernel Layer                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────┐  │   │
│  │  │   Kernel    │  │   Plugin   │  │ Planner │  │   │
│  │  │   (内核)    │  │   (插件)   │  │ (规划器)│  │   │
│  │  └─────────────┘  └─────────────┘  └─────────┘  │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │            Service Layer                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────┐  │   │
│  │  │ AI Service  │  │   Memory    │  │  Chat   │  │   │
│  │  │ (AI 服务)   │  │  (内存)     │  │ History │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────┘  │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │            Integration Layer                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────┐  │   │
│  │  │   Azure     │  │  OpenAI     │  │ Hugging │  │   │
│  │  │   OpenAI    │  │  API        │  │  Face   │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────┘  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 18.1.3 核心概念速览

| 概念 | 定义 | 类比 |
|------|------|------|
| **Kernel** | 中心编排器，协调所有组件 | 操作系统内核 |
| **Plugin** | 功能模块，提供特定能力 | 应用程序 |
| **Function** | Plugin 中的可调用函数 | API 端点 |
| **Planner** | 自动规划执行步骤的组件 | 任务调度器 |
| **Memory** | 持久化存储和检索信息 | 数据库 |
| **AI Service** | LLM 服务接口 | 云服务 |
| **Chat History** | 对话历史管理 | 会话记录 |

---

## 18.2 Kernel 核心概念

### 18.2.1 基础 Kernel 创建

```python
# Python 版本
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

# 创建 Kernel 实例
kernel = Kernel()

# 添加 AI 服务
kernel.add_service(
    OpenAIChatCompletion(
        service_id="chat",
        ai_model_id="gpt-4",
        api_key="your-api-key",
    )
)

# 使用 Kernel
async def main():
    # 获取 AI 服务
    chat_service = kernel.get_service("chat")
    
    # 调用 AI
    result = await chat_service.complete_chat(
        messages=[{"role": "user", "content": "你好，请自我介绍"}]
    )
    print(result)

# 运行
import asyncio
asyncio.run(main())
```

### 18.2.2 Kernel 高级配置

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.memory import VolatileMemoryStore
from semantic_kernel.planners import SequentialPlanner

# 创建带完整配置的 Kernel
kernel = Kernel()

# 添加 AI 服务
kernel.add_service(
    OpenAIChatCompletion(
        service_id="chat_gpt4",
        ai_model_id="gpt-4",
        api_key="your-api-key",
    )
)

kernel.add_service(
    OpenAIChatCompletion(
        service_id="chat_gpt35",
        ai_model_id="gpt-3.5-turbo",
        api_key="your-api-key",
    )
)

# 配置内存
kernel.add_plugin(
    VolatileMemoryStore(),
    plugin_name="memory"
)

# 配置规划器
planner = SequentialPlanner(kernel)

print(f"Kernel 服务: {kernel.services}")
print(f"Kernel 插件: {kernel.plugins}")
```

### 18.2.3 Kernel 依赖注入

```python
from semantic_kernel import Kernel
from semantic_kernel.services import AIServiceSelector

# 自定义服务选择器
class CustomServiceSelector(AIServiceSelector):
    """自定义 AI 服务选择器"""
    
    def select_ai_service(self, service_id: str, **kwargs):
        """根据条件选择 AI 服务"""
        # 根据任务复杂度选择模型
        task_type = kwargs.get("task_type", "simple")
        
        if task_type == "complex":
            return self.services.get("chat_gpt4")
        else:
            return self.services.get("chat_gpt35")

# 创建带自定义选择器的 Kernel
kernel = Kernel(service_selector=CustomServiceSelector())

# 使用依赖注入
from dependency_injector import containers, providers

class SKContainer(containers.DeclarativeContainer):
    """Semantic Kernel 容器"""
    
    kernel = providers.Singleton(Kernel)
    
    # AI 服务
    chat_service = providers.Singleton(
        OpenAIChatCompletion,
        service_id="chat",
        ai_model_id="gpt-4",
        api_key="your-api-key",
    )

# 使用容器
container = SKContainer()
kernel = container.kernel()
```

### 18.2.4 Kernel 中间件

```python
from semantic_kernel import Kernel
from semantic_kernel.kernel_pinvocation import KernelInvocationContext

# 自定义中间件
class LoggingMiddleware:
    """日志中间件"""
    
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
    
    async def __call__(
        self,
        context: KernelInvocationContext,
        next
    ):
        """中间件调用"""
        print(f"[LOG] 开始执行: {context.function.name}")
        
        # 调用下一个中间件
        result = await next(context)
        
        print(f"[LOG] 执行完成: {context.function.name}")
        return result

class MetricsMiddleware:
    """指标中间件"""
    
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.metrics = {}
    
    async def __call__(
        self,
        context: KernelInvocationContext,
        next
    ):
        """收集执行指标"""
        import time
        start_time = time.time()
        
        result = await next(context)
        
        duration = time.time() - start_time
        self.metrics[context.function.name] = {
            "duration": duration,
            "success": True,
        }
        
        return result

# 创建带中间件的 Kernel
kernel = Kernel()
kernel.add_middleware(LoggingMiddleware(kernel))
kernel.add_middleware(MetricsMiddleware(kernel))
```

---

## 18.3 Plugin 系统深度解析

### 18.3.1 kernel_function 基础

```python
from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function

# 创建 Plugin 类
class MathPlugin:
    """数学计算插件"""
    
    @kernel_function(description="将两个数字相加")
    def add(self, a: float, b: float) -> float:
        """加法运算"""
        return a + b
    
    @kernel_function(description="将两个数字相乘")
    def multiply(self, a: float, b: float) -> float:
        """乘法运算"""
        return a * b
    
    @kernel_function(description="计算平方根")
    def square_root(self, number: float) -> float:
        """平方根运算"""
        import math
        return math.sqrt(number)

# 创建 Kernel 并添加 Plugin
kernel = Kernel()
kernel.add_plugin(MathPlugin(), plugin_name="math")

# 使用 Plugin
async def main():
    # 方式1：通过 Kernel 调用
    result = await kernel.invoke(
        function_name="add",
        plugin_name="math",
        a=10,
        b=20
    )
    print(f"10 + 20 = {result}")
    
    # 方式2：获取 Plugin 并调用
    math_plugin = kernel.plugins["math"]
    result = await math_plugin["multiply"].invoke(kernel, a=5, b=6)
    print(f"5 * 6 = {result}")

import asyncio
asyncio.run(main())
```

### 18.3.2 Plugin 高级配置

```python
from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function, KernelPlugin, KernelFunctionMetadata
from typing import Annotated

# 带类型注解的 Plugin
class WeatherPlugin:
    """天气查询插件"""
    
    @kernel_function(
        description="获取指定城市的天气信息",
        name="get_weather"
    )
    def get_weather(
        self,
        city: Annotated[str, "城市名称"],
        unit: Annotated[str, "温度单位（celsius/fahrenheit）"] = "celsius"
    ) -> str:
        """获取天气信息"""
        # 模拟天气数据
        weather_data = {
            "北京": {"temp": 25, "condition": "晴天"},
            "上海": {"temp": 22, "condition": "多云"},
            "广州": {"temp": 28, "condition": "小雨"},
        }
        
        data = weather_data.get(city, {"temp": 20, "condition": "未知"})
        
        if unit == "fahrenheit":
            temp = data["temp"] * 9/5 + 32
        else:
            temp = data["temp"]
        
        return f"{city}天气: {data['condition']}, 温度: {temp}°{unit[0].upper()}"

# 创建带元数据的 Plugin
plugin_metadata = KernelPlugin(
    name="weather",
    description="天气查询服务",
    functions=[
        KernelFunctionMetadata(
            name="get_weather",
            description="获取天气信息",
            parameters={
                "city": {"type": "string", "description": "城市名称"},
                "unit": {"type": "string", "description": "温度单位", "default": "celsius"},
            },
        )
    ]
)

# 注册 Plugin
kernel = Kernel()
kernel.add_plugin(WeatherPlugin(), plugin_name="weather")
```

### 18.3.3 Plugin 组合模式

```python
from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function

# Plugin 1：数据获取
class DataPlugin:
    """数据获取插件"""
    
    @kernel_function(description="获取用户数据")
    def get_user_data(self, user_id: str) -> dict:
        """获取用户数据"""
        # 模拟数据获取
        return {
            "id": user_id,
            "name": "张三",
            "email": "zhangsan@example.com",
            "age": 30,
        }

# Plugin 2：数据处理
class ProcessingPlugin:
    """数据处理插件"""
    
    @kernel_function(description="验证邮箱格式")
    def validate_email(self, email: str) -> bool:
        """验证邮箱"""
        return "@" in email and "." in email
    
    @kernel_function(description="格式化用户信息")
    def format_user_info(self, user_data: dict) -> str:
        """格式化用户信息"""
        return f"用户: {user_data['name']}, 邮箱: {user_data['email']}"

# Plugin 3：数据存储
class StoragePlugin:
    """数据存储插件"""
    
    def __init__(self):
        self.storage = {}
    
    @kernel_function(description="保存数据")
    def save_data(self, key: str, value: str) -> bool:
        """保存数据"""
        self.storage[key] = value
        return True
    
    @kernel_function(description="获取数据")
    def get_data(self, key: str) -> str:
        """获取数据"""
        return self.storage.get(key, "未找到")

# 组合 Plugin
kernel = Kernel()
kernel.add_plugin(DataPlugin(), plugin_name="data")
kernel.add_plugin(ProcessingPlugin(), plugin_name="processing")
kernel.add_plugin(StoragePlugin(), plugin_name="storage")

# 使用组合的 Plugin
async def process_user(user_id: str):
    """处理用户数据"""
    # 1. 获取用户数据
    user_data = await kernel.invoke(
        function_name="get_user_data",
        plugin_name="data",
        user_id=user_id
    )
    
    # 2. 验证邮箱
    is_valid = await kernel.invoke(
        function_name="validate_email",
        plugin_name="processing",
        email=user_data["email"]
    )
    
    if is_valid:
        # 3. 格式化信息
        formatted = await kernel.invoke(
            function_name="format_user_info",
            plugin_name="processing",
            user_data=user_data
        )
        
        # 4. 保存数据
        await kernel.invoke(
            function_name="save_data",
            plugin_name="storage",
            key=f"user_{user_id}",
            value=formatted
        )
        
        return formatted
    
    return "邮箱验证失败"

import asyncio
result = asyncio.run(process_user("001"))
print(result)
```

---

## 18.4 Planner 自动规划

### 18.4.1 SequentialPlanner

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.planners import SequentialPlanner
from semantic_kernel.functions import kernel_function

# 创建 Kernel
kernel = Kernel()
kernel.add_service(
    OpenAIChatCompletion(
        service_id="chat",
        ai_model_id="gpt-4",
        api_key="your-api-key",
    )
)

# 添加 Plugin
class TimePlugin:
    @kernel_function(description="获取当前时间")
    def current_time(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @kernel_function(description="计算时间差")
    def time_difference(self, start_time: str, end_time: str) -> str:
        return f"时间差: 计算中..."

class TextPlugin:
    @kernel_function(description="将文本转换为大写")
    def to_upper(self, text: str) -> str:
        return text.upper()
    
    @kernel_function(description="统计文本长度")
    def text_length(self, text: str) -> int:
        return len(text)

kernel.add_plugin(TimePlugin(), plugin_name="time")
kernel.add_plugin(TextPlugin(), plugin_name="text")

# 创建规划器
planner = SequentialPlanner(kernel)

# 创建计划
async def create_plan(goal: str):
    """创建执行计划"""
    plan = await planner.create_plan(goal)
    
    print(f"目标: {goal}")
    print(f"计划步骤:")
    for step in plan.steps:
        print(f"  1. {step.description}")
        print(f"     插件: {step.plugin_name}.{step.function_name}")
    
    # 执行计划
    result = await plan.invoke()
    print(f"执行结果: {result}")
    
    return result

import asyncio
asyncio.run(create_plan("获取当前时间并转换为大写"))
```

### 18.4.2 StepwisePlanner

```python
from semantic_kernel import Kernel
from semantic_kernel.planners import StepwisePlanner

# 创建 Kernel
kernel = Kernel()
kernel.add_service(
    OpenAIChatCompletion(
        service_id="chat",
        ai_model_id="gpt-4",
        api_key="your-api-key",
    )
)

# 添加多个 Plugin
class SearchPlugin:
    @kernel_function(description="搜索信息")
    def search(self, query: str) -> str:
        # 模拟搜索结果
        return f"搜索结果: 关于 '{query}' 的信息..."
    
    @kernel_function(description="获取搜索建议")
    def get_suggestions(self, query: str) -> list:
        return [f"建议1: {query}是什么", f"建议2: 如何使用{query}"]

class CalculatorPlugin:
    @kernel_function(description="计算数学表达式")
    def calculate(self, expression: str) -> str:
        try:
            result = eval(expression)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"
    
    @kernel_function(description="单位换算")
    def convert_unit(self, value: float, from_unit: str, to_unit: str) -> str:
        # 简单的单位换算
        conversions = {
            ("km", "miles"): lambda x: x * 0.621371,
            ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
        }
        
        key = (from_unit, to_unit)
        if key in conversions:
            result = conversions[key](value)
            return f"{value} {from_unit} = {result:.2f} {to_unit}"
        return f"不支持的换算: {from_unit} -> {to_unit}"

kernel.add_plugin(SearchPlugin(), plugin_name="search")
kernel.add_plugin(CalculatorPlugin(), plugin_name="calculator")

# 创建 Stepwise 规划器
stepwise_planner = StepwisePlanner(kernel)

# 执行
async def run_stepwise_plan(goal: str):
    """运行 Stepwise 规划"""
    plan = await stepwise_planner.create_plan(goal)
    result = await plan.invoke()
    
    print(f"目标: {goal}")
    print(f"执行结果: {result}")
    
    return result

import asyncio
asyncio.run(run_stepwise_plan("搜索 AI Agent 的信息，然后计算 100 英里等于多少公里"))
```

### 18.4.3 自定义 Planner

```python
from semantic_kernel import Kernel
from semantic_kernel.planners import PlannerResult
from semantic_kernel.functions import KernelPlugin, KernelFunction

class CustomPlanner:
    """自定义规划器"""
    
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.plugins = {}
    
    def register_plugin(self, name: str, plugin: KernelPlugin):
        """注册 Plugin"""
        self.plugins[name] = plugin
    
    async def create_plan(self, goal: str) -> PlannerResult:
        """创建执行计划"""
        # 分析目标
        steps = self._analyze_goal(goal)
        
        # 创建执行计划
        plan = PlannerResult(
            goal=goal,
            steps=steps,
        )
        
        return plan
    
    def _analyze_goal(self, goal: str) -> list:
        """分析目标，生成步骤"""
        # 简单的关键词匹配
        steps = []
        
        if "搜索" in goal or "search" in goal.lower():
            steps.append({
                "plugin": "search",
                "function": "search",
                "params": {"query": goal}
            })
        
        if "计算" in goal or "calculate" in goal.lower():
            steps.append({
                "plugin": "calculator",
                "function": "calculate",
                "params": {"expression": goal}
            })
        
        if "时间" in goal or "time" in goal.lower():
            steps.append({
                "plugin": "time",
                "function": "current_time",
                "params": {}
            })
        
        return steps

# 使用自定义规划器
custom_planner = CustomPlanner(kernel)
plan = await custom_planner.create_plan("搜索 AI 信息并计算相关数据")
```

---

## 18.5 Memory Store 记忆存储

### 18.5.1 内存基础

```python
from semantic_kernel import Kernel
from semantic_kernel.memory import (
    VolatileMemoryStore,
    SemanticTextMemory,
    MemoryStore,
)
from semantic_kernel.connectors.ai.open_ai import OpenAIEmbeddingGenerator

# 创建 Kernel
kernel = Kernel()

# 配置 Embedding 服务
embedding_generator = OpenAIEmbeddingGenerator(
    service_id="embedding",
    ai_model_id="text-embedding-ada-002",
    api_key="your-api-key",
)

kernel.add_service(embedding_generator)

# 创建内存存储
memory_store = VolatileMemoryStore()
text_memory = SemanticTextMemory(storage=memory_store, embedding_generator=embedding_generator)

# 添加记忆
async def add_memories():
    """添加记忆"""
    await text_memory.save_information(
        collection="ai_knowledge",
        text="Semantic Kernel 是微软开发的 AI 编排框架",
        id="sk_intro",
        description="Semantic Kernel 介绍"
    )
    
    await text_memory.save_information(
        collection="ai_knowledge",
        text="LangChain 是最流行的 LLM 应用框架",
        id="langchain_intro",
        description="LangChain 介绍"
    )
    
    await text_memory.save_information(
        collection="ai_knowledge",
        text="CrewAI 是基于角色的多智能体协作框架",
        id="crewai_intro",
        description="CrewAI 介绍"
    )

# 检索记忆
async def search_memories(query: str):
    """搜索记忆"""
    results = text_memory.search(
        collection="ai_knowledge",
        query=query,
        limit=2,
    )
    
    print(f"搜索: {query}")
    for result in results:
        print(f"  - {result.text} (相关度: {result.relevance:.2f})")

import asyncio
asyncio.run(add_memories())
asyncio.run(search_memories("AI 编排框架"))
```

### 18.5.2 高级内存配置

```python
from semantic_kernel import Kernel
from semantic_kernel.memory import VolatileMemoryStore, SemanticTextMemory
from semantic_kernel.connectors.ai.open_ai import OpenAIEmbeddingGenerator

class AdvancedMemoryManager:
    """高级内存管理器"""
    
    def __init__(self):
        self.kernel = Kernel()
        
        # 配置 Embedding
        self.embedding_generator = OpenAIEmbeddingGenerator(
            service_id="embedding",
            ai_model_id="text-embedding-ada-002",
            api_key="your-api-key",
        )
        
        # 创建内存
        self.memory_store = VolatileMemoryStore()
        self.text_memory = SemanticTextMemory(
            storage=self.memory_store,
            embedding_generator=self.embedding_generator
        )
    
    async def save_conversation(self, collection: str, messages: list[dict]):
        """保存对话到内存"""
        for i, msg in enumerate(messages):
            await self.text_memory.save_information(
                collection=collection,
                text=f"{msg['role']}: {msg['content']}",
                id=f"msg_{i}",
                description=f"对话消息 {i}"
            )
    
    async def search_similar(self, collection: str, query: str, limit: int = 5):
        """搜索相似内容"""
        results = self.text_memory.search(
            collection=collection,
            query=query,
            limit=limit,
        )
        return results
    
    async def get_collection_stats(self, collection: str):
        """获取集合统计信息"""
        # 获取集合中的所有条目
        entries = []
        async for entry in self.memory_store.get_all(collection):
            entries.append(entry)
        
        return {
            "collection": collection,
            "count": len(entries),
        }

# 使用示例
async def main():
    manager = AdvancedMemoryManager()
    
    # 保存对话
    conversation = [
        {"role": "user", "content": "什么是 Semantic Kernel?"},
        {"role": "assistant", "content": "Semantic Kernel 是微软开发的 AI 编排框架。"},
        {"role": "user", "content": "它有什么优势?"},
        {"role": "assistant", "content": "它具有强大的 Plugin 系统和自动规划能力。"},
    ]
    
    await manager.save_conversation("tech_discussion", conversation)
    
    # 搜索
    results = await manager.search_similar("tech_discussion", "AI 编排框架")
    for result in results:
        print(f"  - {result.text}")

import asyncio
asyncio.run(main())
```

### 18.5.3 外部存储集成

```python
from semantic_kernel import Kernel
from semantic_kernel.memory import SemanticTextMemory
from semantic_kernel.memory.store import MemoryStore
import sqlite3
import json

class SQLiteMemoryStore(MemoryStore):
    """SQLite 内存存储"""
    
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                collection TEXT,
                text TEXT,
                embedding BLOB,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    async def save(self, collection: str, id: str, text: str, embedding: list, metadata: dict = None):
        """保存记忆"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO memories (id, collection, text, embedding, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (id, collection, text, json.dumps(embedding), json.dumps(metadata or {})))
        
        conn.commit()
        conn.close()
    
    async def get(self, collection: str, id: str):
        """获取记忆"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT text, embedding, metadata FROM memories
            WHERE collection = ? AND id = ?
        """, (collection, id))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "text": result[0],
                "embedding": json.loads(result[1]),
                "metadata": json.loads(result[2]),
            }
        return None
    
    async def get_all(self, collection: str):
        """获取集合中的所有记忆"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, text, embedding, metadata FROM memories
            WHERE collection = ?
        """, (collection,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                "id": row[0],
                "text": row[1],
                "embedding": json.loads(row[2]),
                "metadata": json.loads(row[3]),
            }
            for row in results
        ]

# 使用 SQLite 内存存储
sqlite_store = SQLiteMemoryStore("my_memory.db")
```

---

## 18.6 Azure 集成

### 18.6.1 Azure OpenAI 集成

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, AzureChatCompletion

# 创建带 Azure OpenAI 的 Kernel
kernel = Kernel()

# 添加 Azure OpenAI 服务
kernel.add_service(
    AzureChatCompletion(
        service_id="azure_chat",
        deployment_name="gpt-4",
        endpoint="https://your-resource.openai.azure.com/",
        api_key="your-api-key",
        api_version="2024-02-15-preview",
    )
)

# 使用 Azure 服务
async def use_azure():
    chat_service = kernel.get_service("azure_chat")
    
    result = await chat_service.complete_chat(
        messages=[{"role": "user", "content": "你好，请介绍一下自己"}]
    )
    
    print(result)

import asyncio
asyncio.run(use_azure())
```

### 18.6.2 Azure AI Search 集成

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIEmbeddingGenerator
from semantic_kernel.connectors.search.azure_ai_search import AzureAISearchSettings
from semantic_kernel.memory import SemanticTextMemory, AzureAISearchMemoryStore

# 配置 Azure AI Search
search_settings = AzureAISearchSettings(
    endpoint="https://your-search.search.windows.net/",
    api_key="your-api-key",
    index_name="semantic-kernel-index",
)

# 创建内存存储
azure_store = AzureAISearchMemoryStore(
    search_settings=search_settings
)

# 创建内存
embedding_generator = OpenAIEmbeddingGenerator(
    service_id="embedding",
    ai_model_id="text-embedding-ada-002",
    api_key="your-api-key",
)

memory = SemanticTextMemory(
    storage=azure_store,
    embedding_generator=embedding_generator
)

# 使用
async def use_azure_search():
    # 保存信息
    await memory.save_information(
        collection="enterprise_knowledge",
        text="企业知识库内容...",
        id="doc_1"
    )
    
    # 搜索
    results = memory.search(
        collection="enterprise_knowledge",
        query="企业知识",
        limit=5
    )
    
    for result in results:
        print(f"  - {result.text}")

import asyncio
asyncio.run(use_azure_search())
```

### 18.6.3 Azure 部署配置

```yaml
# azure-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: semantic-kernel-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: semantic-kernel
  template:
    metadata:
      labels:
        app: semantic-kernel
    spec:
      containers:
      - name: app
        image: your-registry/semantic-kernel-app:latest
        ports:
        - containerPort: 8080
        env:
        - name: AZURE_OPENAI_ENDPOINT
          valueFrom:
            secretKeyRef:
              name: azure-secrets
              key: openai-endpoint
        - name: AZURE_OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: azure-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

---

## 18.7 SK vs LangChain 对比

### 18.7.1 核心差异

| 特性 | Semantic Kernel | LangChain |
|------|-----------------|-----------|
| **设计理念** | 企业级 AI 编排 | 通用 LLM 应用框架 |
| **语言支持** | C#, Python | Python, JS/TS |
| **Plugin 系统** | 内核级 Plugin | Tool 集成 |
| **规划能力** | 内置 Planner | 自定义 Agent |
| **企业集成** | 原生 Azure 支持 | 社区驱动 |
| **学习曲线** | 中等 | 较低 |
| **社区规模** | 中等 | 大 |
| **企业支持** | 微软官方 | 社区 |

### 18.7.2 选择指南

```python
# Semantic Kernel 适用场景
sk_scenarios = [
    "企业级 Azure 部署",
    "需要强大 Plugin 系统",
    "自动规划需求",
    "C# 开发环境",
    "微软技术栈",
    "企业安全合规要求",
]

# LangChain 适用场景
langchain_scenarios = [
    "快速原型开发",
    "Python 生态集成",
    "社区资源丰富",
    "灵活的 Agent 编排",
    "开源优先",
    "快速迭代",
]

# 混合使用示例
def hybrid_approach():
    """混合使用 SK 和 LangChain"""
    # 使用 LangChain 快速原型
    # 使用 SK 进行企业部署
    pass
```

### 18.7.3 性能对比

| 指标 | Semantic Kernel | LangChain |
|------|-----------------|-----------|
| **启动时间** | 中等 | 快 |
| **内存占用** | 中等 | 较高 |
| **并发能力** | 优秀 | 中等 |
| **扩展性** | 优秀 | 优秀 |
| **企业支持** | 优秀 | 一般 |

---

## 18.8 综合实战案例

### 18.8.1 企业知识问答系统

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.planners import SequentialPlanner
from semantic_kernel.memory import SemanticTextMemory, VolatileMemoryStore
from semantic_kernel.functions import kernel_function

# 创建 Kernel
kernel = Kernel()

# 添加 Azure OpenAI 服务
kernel.add_service(
    AzureChatCompletion(
        service_id="chat",
        deployment_name="gpt-4",
        endpoint="https://your-resource.openai.azure.com/",
        api_key="your-api-key",
    )
)

# 创建内存存储
memory_store = VolatileMemoryStore()
memory = SemanticTextMemory(
    storage=memory_store,
    embedding_generator=embedding_generator
)

# 添加知识库
async def build_knowledge_base():
    """构建知识库"""
    documents = [
        ("我们的产品是 AI 智能客服系统，支持多渠道接入。", "product_1"),
        ("系统支持自然语言理解，可以处理 80% 的常见问题。", "product_2"),
        ("部署方式支持私有化部署和 SaaS 两种模式。", "product_3"),
        ("定价根据坐席数量和月活跃用户数计算。", "pricing_1"),
        ("技术支持提供 7x24 小时在线支持。", "support_1"),
    ]
    
    for text, doc_id in documents:
        await memory.save_information(
            collection="knowledge_base",
            text=text,
            id=doc_id,
        )

# 创建 Plugin
class KnowledgePlugin:
    """知识库插件"""
    
    @kernel_function(description="搜索知识库")
    async def search_knowledge(self, query: str) -> str:
        """搜索知识库"""
        results = memory.search(
            collection="knowledge_base",
            query=query,
            limit=3,
        )
        
        if results:
            return "\n".join([f"- {r.text}" for r in results])
        return "未找到相关知识"

class ResponsePlugin:
    """响应生成插件"""
    
    @kernel_function(description="生成回答")
    async def generate_response(self, context: str, query: str) -> str:
        """基于上下文生成回答"""
        return f"基于知识库：{context}\n\n回答您的问题：{query}"

# 注册 Plugin
kernel.add_plugin(KnowledgePlugin(), plugin_name="knowledge")
kernel.add_plugin(ResponsePlugin(), plugin_name="response")

# 创建规划器
planner = SequentialPlanner(kernel)

# 处理用户问题
async def answer_question(question: str):
    """回答用户问题"""
    plan = await planner.create_plan(
        goal=f"搜索知识库并回答：{question}"
    )
    
    result = await plan.invoke()
    return result

import asyncio
asyncio.run(build_knowledge_base())
answer = asyncio.run(answer_question("你们的产品有什么特点？"))
print(answer)
```

---

## 18.9 高级特性与最佳实践

### 18.9.1 Chat History 管理

```python
from semantic_kernel import Kernel
from semantic_kernel.contents import ChatHistory, ChatMessageContent
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

class ChatHistoryManager:
    """对话历史管理器"""
    
    def __init__(self):
        self.kernel = Kernel()
        self.kernel.add_service(
            OpenAIChatCompletion(
                service_id="chat",
                ai_model_id="gpt-4",
                api_key="your-api-key",
            )
        )
        self.histories = {}
    
    def get_or_create_history(self, session_id: str) -> ChatHistory:
        """获取或创建对话历史"""
        if session_id not in self.histories:
            self.histories[session_id] = ChatHistory(
                system_message="你是一个有帮助的助手。"
            )
        return self.histories[session_id]
    
    async def chat(self, session_id: str, user_message: str) -> str:
        """进行对话"""
        history = self.get_or_create_history(session_id)
        
        # 添加用户消息
        history.add_user_message(user_message)
        
        # 获取 AI 服务
        chat_service = self.kernel.get_service("chat")
        
        # 生成回复
        result = await chat_service.complete_chat(
            history=history,
        )
        
        # 添加助手回复
        history.add_assistant_message(result)
        
        return result
    
    def clear_history(self, session_id: str):
        """清除对话历史"""
        if session_id in self.histories:
            del self.histories[session_id]
    
    def get_history_stats(self, session_id: str) -> dict:
        """获取对话历史统计"""
        history = self.get_or_create_history(session_id)
        return {
            "message_count": len(history.messages),
            "user_messages": sum(1 for m in history.messages if m.role == "user"),
            "assistant_messages": sum(1 for m in history.messages if m.role == "assistant"),
        }

# 使用示例
async def main():
    manager = ChatHistoryManager()
    
    # 多轮对话
    response1 = await manager.chat("session_1", "你好，请介绍一下自己")
    print(f"AI: {response1}")
    
    response2 = await manager.chat("session_1", "你能做什么？")
    print(f"AI: {response2}")
    
    # 查看统计
    stats = manager.get_history_stats("session_1")
    print(f"统计: {stats}")

import asyncio
asyncio.run(main())
```

### 18.9.2 流式输出

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import ChatHistory

class StreamingChat:
    """流式对话"""
    
    def __init__(self):
        self.kernel = Kernel()
        self.kernel.add_service(
            OpenAIChatCompletion(
                service_id="chat",
                ai_model_id="gpt-4",
                api_key="your-api-key",
            )
        )
        self.history = ChatHistory()
    
    async def stream_chat(self, user_message: str):
        """流式对话"""
        self.history.add_user_message(user_message)
        
        chat_service = self.kernel.get_service("chat")
        
        # 流式获取回复
        full_response = ""
        async for chunk in chat_service.complete_chat_stream(
            history=self.history,
        ):
            # 打印每个 chunk
            print(chunk, end="", flush=True)
            full_response += chunk
        
        print()  # 换行
        
        # 添加完整回复到历史
        self.history.add_assistant_message(full_response)
        
        return full_response

# 使用示例
async def main():
    streaming = StreamingChat()
    await streaming.stream_chat("请用 Python 写一个快速排序算法")

import asyncio
asyncio.run(main())
```

### 18.9.3 函数调用（Function Calling）

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function
from semantic_kernel.contents import ChatHistory

class FunctionCallingExample:
    """函数调用示例"""
    
    def __init__(self):
        self.kernel = Kernel()
        self.kernel.add_service(
            OpenAIChatCompletion(
                service_id="chat",
                ai_model_id="gpt-4",
                api_key="your-api-key",
            )
        )
        
        # 注册 Plugin
        self.kernel.add_plugin(WeatherPlugin(), plugin_name="weather")
        self.kernel.add_plugin(CalculatorPlugin(), plugin_name="calculator")
    
    async def chat_with_functions(self, user_message: str):
        """带函数调用的对话"""
        chat_service = self.kernel.get_service("chat")
        
        # 获取可用函数
        functions = self.kernel.get_full_history_metadata()
        
        # 调用 AI（可能触发函数调用）
        result = await chat_service.complete_chat(
            messages=[{"role": "user", "content": user_message}],
            functions=functions,
        )
        
        # 检查是否需要函数调用
        if result.function_call:
            function_name = result.function_call.name
            arguments = result.function_call.arguments
            
            # 执行函数
            function_result = await self.kernel.invoke(
                function_name=function_name,
                plugin_name=result.function_call.plugin_name,
                **arguments,
            )
            
            # 继续对话
            return await self.chat_with_functions(
                f"函数 {function_name} 的结果是: {function_result}"
            )
        
        return result.content

# Plugin 定义
class WeatherPlugin:
    @kernel_function(description="获取天气")
    def get_weather(self, city: str) -> str:
        return f"{city}: 晴天, 25°C"

class CalculatorPlugin:
    @kernel_function(description="计算")
    def calculate(self, expression: str) -> str:
        try:
            return str(eval(expression))
        except:
            return "计算错误"
```

### 18.9.4 Plugin 版本管理

```python
from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function
from typing import Protocol

# Plugin 接口定义
class SearchPluginProtocol(Protocol):
    """搜索 Plugin 协议"""
    
    def search(self, query: str) -> str: ...

# Plugin 版本管理
class PluginVersionManager:
    """Plugin 版本管理器"""
    
    def __init__(self):
        self.plugins = {}
        self.versions = {}
    
    def register_plugin(self, name: str, plugin, version: str = "1.0.0"):
        """注册 Plugin"""
        if name not in self.plugins:
            self.plugins[name] = {}
        
        self.plugins[name][version] = plugin
        self.versions[name] = version
    
    def get_plugin(self, name: str, version: str = None):
        """获取 Plugin"""
        if name not in self.plugins:
            raise ValueError(f"Plugin '{name}' not found")
        
        if version is None:
            version = self.versions[name]
        
        if version not in self.plugins[name]:
            raise ValueError(f"Version '{version}' not found for plugin '{name}'")
        
        return self.plugins[name][version]
    
    def list_versions(self, name: str) -> list[str]:
        """列出 Plugin 版本"""
        if name not in self.plugins:
            return []
        return list(self.plugins[name].keys())
    
    def upgrade_plugin(self, name: str, new_version: str):
        """升级 Plugin 版本"""
        if new_version in self.plugins.get(name, {}):
            self.versions[name] = new_version
        else:
            raise ValueError(f"Version '{new_version}' not found for plugin '{name}'")

# 使用示例
class SearchPluginV1:
    @kernel_function(description="搜索 V1")
    def search(self, query: str) -> str:
        return f"V1 搜索结果: {query}"

class SearchPluginV2:
    @kernel_function(description="搜索 V2")
    def search(self, query: str) -> str:
        return f"V2 搜索结果: {query} (增强版)"

# 注册版本
version_manager = PluginVersionManager()
version_manager.register_plugin("search", SearchPluginV1(), "1.0.0")
version_manager.register_plugin("search", SearchPluginV2(), "2.0.0")

# 使用特定版本
plugin_v1 = version_manager.get_plugin("search", "1.0.0")
plugin_v2 = version_manager.get_plugin("search", "2.0.0")
```

### 18.9.5 错误处理与重试

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.exceptions import ServiceResponseException
import asyncio
from typing import Optional

class ResilientKernel:
    """弹性 Kernel"""
    
    def __init__(self):
        self.kernel = Kernel()
        self.kernel.add_service(
            OpenAIChatCompletion(
                service_id="chat",
                ai_model_id="gpt-4",
                api_key="your-api-key",
            )
        )
        self.max_retries = 3
        self.retry_delay = 1.0
    
    async def invoke_with_retry(
        self,
        function_name: str,
        plugin_name: str,
        **kwargs
    ):
        """带重试的调用"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                result = await self.kernel.invoke(
                    function_name=function_name,
                    plugin_name=plugin_name,
                    **kwargs
                )
                return result
                
            except ServiceResponseException as e:
                last_error = e
                print(f"调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
            
            except Exception as e:
                last_error = e
                print(f"未预期错误: {e}")
                raise
        
        raise Exception(f"调用失败，已重试 {self.max_retries} 次: {last_error}")
    
    async def chat_with_retry(self, user_message: str) -> str:
        """带重试的对话"""
        chat_service = self.kernel.get_service("chat")
        
        for attempt in range(self.max_retries):
            try:
                result = await chat_service.complete_chat(
                    messages=[{"role": "user", "content": user_message}]
                )
                return result
                
            except ServiceResponseException as e:
                print(f"对话失败 (尝试 {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)

# 使用
async def main():
    resilient = ResilientKernel()
    result = await resilient.chat_with_retry("你好")
    print(result)

import asyncio
asyncio.run(main())
```

### 18.9.6 性能优化

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from functools import lru_cache
import hashlib

class OptimizedKernel:
    """优化的 Kernel"""
    
    def __init__(self):
        self.kernel = Kernel()
        self.kernel.add_service(
            OpenAIChatCompletion(
                service_id="chat",
                ai_model_id="gpt-4",
                api_key="your-api-key",
            )
        )
        self.cache = {}
        self.cache_ttl = 3600  # 1小时
    
    def _get_cache_key(self, messages: list) -> str:
        """生成缓存键"""
        content = str(messages)
        return hashlib.md5(content.encode()).hexdigest()
    
    async def invoke_with_cache(self, function_name: str, plugin_name: str, **kwargs):
        """带缓存的调用"""
        cache_key = self._get_cache_key([function_name, plugin_name, kwargs])
        
        # 检查缓存
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 执行调用
        result = await self.kernel.invoke(
            function_name=function_name,
            plugin_name=plugin_name,
            **kwargs
        )
        
        # 存储到缓存
        self.cache[cache_key] = result
        
        return result
    
    async def batch_invoke(self, calls: list[dict]):
        """批量调用"""
        import asyncio
        
        tasks = []
        for call in calls:
            task = self.invoke_with_cache(
                function_name=call["function_name"],
                plugin_name=call["plugin_name"],
                **call.get("kwargs", {})
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

# 使用示例
async def main():
    optimized = OptimizedKernel()
    
    # 批量调用
    calls = [
        {"function_name": "add", "plugin_name": "math", "kwargs": {"a": 1, "b": 2}},
        {"function_name": "multiply", "plugin_name": "math", "kwargs": {"a": 3, "b": 4}},
    ]
    
    results = await optimized.batch_invoke(calls)
    print(f"批量结果: {results}")

import asyncio
asyncio.run(main())
```

### 18.9.7 测试与调试

```python
from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function
from unittest.mock import AsyncMock, MagicMock
import pytest

# 测试 Plugin
class TestMathPlugin:
    """数学插件测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.plugin = MathPlugin()
    
    def test_add(self):
        """测试加法"""
        assert self.plugin.add(2, 3) == 5
        assert self.plugin.add(-1, 1) == 0
        assert self.plugin.add(0, 0) == 0
    
    def test_multiply(self):
        """测试乘法"""
        assert self.plugin.multiply(2, 3) == 6
        assert self.plugin.multiply(-1, 1) == -1
        assert self.plugin.multiply(0, 5) == 0
    
    def test_square_root(self):
        """测试平方根"""
        assert self.plugin.square_root(4) == 2
        assert self.plugin.square_root(9) == 3
        assert self.plugin.square_root(0) == 0

# 测试 Kernel
class TestKernel:
    """Kernel 测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.kernel = Kernel()
        
        # Mock AI 服务
        self.mock_chat_service = AsyncMock()
        self.mock_chat_service.complete_chat.return_value = "测试回复"
        
        self.kernel.add_service(self.mock_chat_service)
    
    @pytest.mark.asyncio
    async def test_invoke(self):
        """测试调用"""
        # 注册测试 Plugin
        self.kernel.add_plugin(MathPlugin(), plugin_name="math")
        
        # 调用
        result = await self.kernel.invoke(
            function_name="add",
            plugin_name="math",
            a=2,
            b=3
        )
        
        assert result == 5

# 使用 pytest 运行测试
# pytest test_semantic_kernel.py -v
```

### 18.9.8 SK API 速查表

```python
# 导入核心组件
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    AzureChatCompletion,
    OpenAIEmbeddingGenerator,
)
from semantic_kernel.functions import kernel_function, KernelPlugin
from semantic_kernel.planners import SequentialPlanner, StepwisePlanner
from semantic_kernel.memory import (
    VolatileMemoryStore,
    SemanticTextMemory,
    AzureAISearchMemoryStore,
)
from semantic_kernel.contents import ChatHistory, ChatMessageContent

# Kernel 创建
kernel = Kernel()
kernel.add_service(OpenAIChatCompletion(...))
kernel.add_plugin(MyPlugin(), plugin_name="my_plugin")

# 调用
result = await kernel.invoke(
    function_name="function_name",
    plugin_name="plugin_name",
    **kwargs
)

# 流式调用
async for chunk in kernel.invoke_stream(...):
    print(chunk)

# 规划
planner = SequentialPlanner(kernel)
plan = await planner.create_plan(goal="...")
result = await plan.invoke()

# 内存
memory = SemanticTextMemory(storage=VolatileMemoryStore(), ...)
await memory.save_information(collection="...", text="...", id="...")
results = memory.search(collection="...", query="...")

# 对话历史
history = ChatHistory(system_message="...")
history.add_user_message("...")
history.add_assistant_message("...")

# 函数调用
@kernel_function(description="描述")
def my_function(self, arg: str) -> str:
    return "result"
```

### 18.9.9 常见问题排查

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| **Plugin 未找到** | 名称错误 | 检查 `plugin_name` 和 `function_name` |
| **AI 服务错误** | API 配置错误 | 检查 `api_key` 和 `endpoint` |
| **内存不工作** | 未初始化 | 确保创建了 `MemoryStore` |
| **规划失败** | 函数描述不清晰 | 优化 `@kernel_function` 的 `description` |
| **流式输出中断** | 连接问题 | 检查网络和重试 |
| **Azure 部署失败** | 权限不足 | 检查 Azure RBAC 配置 |

### 18.9.10 最佳实践总结

> **最佳实践清单**：
> 
> 1. **Kernel 设计**
>    - 保持 Kernel 的单一职责
>    - 使用依赖注入管理服务
>    - 合理配置中间件
> 
> 2. **Plugin 开发**
>    - 为每个函数提供清晰的描述
>    - 使用类型注解确保类型安全
>    - 实现适当的错误处理
> 
> 3. **规划器使用**
>    - 根据任务复杂度选择规划器
>    - 提供足够的 Plugin 描述
>    - 监控规划执行过程
> 
> 4. **内存管理**
>    - 选择合适的存储后端
>    - 定期清理过期数据
>    - 优化 Embedding 生成
> 
> 5. **企业部署**
>    - 使用 Azure 原生服务
>    - 实现安全认证和授权
>    - 配置监控和告警
> 
> 6. **性能优化**
>    - 启用缓存减少 API 调用
>    - 使用异步执行提升吞吐量
>    - 批量处理多个请求

### 18.9.11 成本控制策略

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from datetime import datetime, timedelta

class CostTracker:
    """成本追踪器"""
    
    def __init__(self, budget: float = 100.0):
        self.budget = budget
        self.total_cost = 0.0
        self.daily_costs = {}
        self.call_history = []
    
    def track_call(self, model: str, tokens_used: int):
        """追踪 API 调用成本"""
        # 估算成本（简化计算）
        cost_per_1k_tokens = {
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.002,
            "text-embedding-ada-002": 0.0001,
        }
        
        cost = (tokens_used / 1000) * cost_per_1k_tokens.get(model, 0.01)
        
        self.total_cost += cost
        
        # 记录每日成本
        today = datetime.now().strftime("%Y-%m-%d")
        self.daily_costs[today] = self.daily_costs.get(today, 0) + cost
        
        # 记录调用历史
        self.call_history.append({
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "tokens": tokens_used,
            "cost": cost,
        })
        
        # 检查预算
        if self.total_cost > self.budget:
            raise Exception(f"超出预算: ${self.total_cost:.2f} / ${self.budget:.2f}")
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_cost": self.total_cost,
            "budget_remaining": self.budget - self.total_cost,
            "daily_costs": self.daily_costs,
            "call_count": len(self.call_history),
        }

# 使用
tracker = CostTracker(budget=50.0)
tracker.track_call("gpt-4", 1000)  # $0.03
tracker.track_call("gpt-3.5-turbo", 5000)  # $0.01

print(tracker.get_stats())
```

### 18.9.12 监控与告警

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from datetime import datetime
import logging

class SKMonitor:
    """Semantic Kernel 监控"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_duration": 0,
            "errors": 0,
        }
    
    def record_call(self, duration: float, tokens: int, success: bool):
        """记录调用"""
        self.metrics["total_calls"] += 1
        self.metrics["total_tokens"] += tokens
        self.metrics["total_duration"] += duration
        
        if not success:
            self.metrics["errors"] += 1
        
        # 记录日志
        self.logger.info(
            f"API Call - Duration: {duration:.2f}s, Tokens: {tokens}, Success: {success}"
        )
    
    def get_metrics(self) -> dict:
        """获取指标"""
        avg_duration = (
            self.metrics["total_duration"] / self.metrics["total_calls"]
            if self.metrics["total_calls"] > 0
            else 0
        )
        
        error_rate = (
            self.metrics["errors"] / self.metrics["total_calls"]
            if self.metrics["total_calls"] > 0
            else 0
        )
        
        return {
            **self.metrics,
            "average_duration": avg_duration,
            "error_rate": error_rate,
        }
    
    def check_alerts(self) -> list[str]:
        """检查告警条件"""
        alerts = []
        
        metrics = self.get_metrics()
        
        if metrics["error_rate"] > 0.1:
            alerts.append(f"高错误率: {metrics['error_rate']:.2%}")
        
        if metrics["average_duration"] > 5.0:
            alerts.append(f"平均响应时间过长: {metrics['average_duration']:.2f}s")
        
        return alerts

# 使用
monitor = SKMonitor()

# 记录调用
monitor.record_call(duration=1.5, tokens=500, success=True)
monitor.record_call(duration=3.2, tokens=1000, success=True)
monitor.record_call(duration=0.5, tokens=0, success=False)

# 获取指标
print(monitor.get_metrics())

# 检查告警
alerts = monitor.check_alerts()
if alerts:
    print(f"告警: {alerts}")
```

---

## 本章小结

本章深入探讨了 Semantic Kernel 企业级 Agent 框架的核心特性：

1. **Kernel 核心**：中心编排器，协调所有组件
2. **Plugin 系统**：kernel_function 实现功能模块化
3. **Planner 规划**：Sequential 和 Stepwise 自动规划
4. **Memory Store**：多种内存存储方案
5. **Azure 集成**：原生支持 Azure 服务
6. **框架对比**：SK 适合企业级，LangChain 适合快速开发

> **核心要点**：Semantic Kernel 的核心优势在于其企业级设计和 Azure 原生集成。通过强大的 Plugin 系统和自动规划能力，可以快速构建企业级 AI 应用。

---

## 思考题

1. Semantic Kernel 的 Plugin 系统与 LangChain 的 Tool 有何区别？
2. 如何设计一个支持多租户的企业级 SK 应用？
3. SequentialPlanner 和 StepwisePlanner 各适用于什么场景？
4. 如何将 SK 与现有的 .NET 企业应用集成？
5. SK 的内存系统如何与 Azure AI Search 集成？

---

## 附录：Semantic Kernel 生态系统

### 核心组件

```
Semantic Kernel 生态系统：

┌─────────────────────────────────────────────────────────┐
│                  Semantic Kernel Core                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Kernel    │  │   Plugin    │  │  Planner    │     │
│  │   (内核)    │  │   (插件)    │  │  (规划器)   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
├─────────────────────────────────────────────────────────┤
│                Connectors & Services                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Azure OpenAI│  │   OpenAI    │  │ HuggingFace │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
├─────────────────────────────────────────────────────────┤
│                Memory & Storage                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  Volatile   │  │ Azure AI    │  │  Redis      │     │
│  │  Memory     │  │  Search     │  │  Memory     │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### 安装与配置

```bash
# 安装 Semantic Kernel
pip install semantic-kernel

# 安装 Azure 集成
pip install semantic-kernel[azure]

# 安装所有可选依赖
pip install semantic-kernel[all]

# 验证安装
python -c "import semantic_kernel; print(semantic_kernel.__version__)"
```

### 环境变量

```bash
# OpenAI
export OPENAI_API_KEY="your-api-key"

# Azure OpenAI
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4"

# Azure AI Search
export AZURE_AI_SEARCH_ENDPOINT="https://your-search.search.windows.net/"
export AZURE_AI_SEARCH_API_KEY="your-api-key"

# HuggingFace
export HUGGINGFACE_API_KEY="your-api-key"
```

### 版本兼容性

| Semantic Kernel 版本 | Python 版本 | 主要特性 |
|---------------------|-------------|----------|
| 0.x | 3.10+ | 基础功能 |
| 1.x | 3.10+ | 稳定版 |
| 2.x | 3.10+ | 最新版，增强功能 |

---

## 参考资源

- [Semantic Kernel 官方文档](https://learn.microsoft.com/semantic-kernel/)
- [Semantic Kernel GitHub](https://github.com/microsoft/semantic-kernel)
- [SK Python 示例](https://github.com/microsoft/semantic-kernel/tree/main/python)
- [Azure OpenAI 文档](https://learn.microsoft.com/azure/ai-services/openai/)
- [SK Plugin 开发指南](https://learn.microsoft.com/semantic-kernel/concepts/plugins/)
- [SK Planner 文档](https://learn.microsoft.com/semantic-kernel/concepts/ai-services/planners/)
- [SK Memory 文档](https://learn.microsoft.com/semantic-kernel/concepts/ai-services/memory/)
- [SK 中文社区](https://github.com/microsoft/semantic-kernel/blob/main/CONTRIBUTING.md)

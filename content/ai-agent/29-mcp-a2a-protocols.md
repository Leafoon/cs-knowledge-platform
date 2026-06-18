---
title: "第29章：MCP 与 A2A 协议"
description: "深入解析 Model Context Protocol (MCP) 与 Agent-to-Agent (A2A) 协议的架构设计、实现细节与对比分析"
updated: "2025-06-15"
---


下面的交互式演示展示了 MCP 协议的核心流程：

<div data-component="MCPProtocolDemo"></div>

# 第29章：MCP 与 A2A 协议

> **学习目标**：
> - 理解 MCP 协议的核心架构设计与通信机制
> - 掌握 MCP Server 的完整实现流程
> - 了解 A2A 协议的设计理念与实现方式
> - 对比 MCP 与 A2A 的适用场景与技术选型
> - 能够基于协议构建跨平台 Agent 互操作方案
> - 理解协议在企业级生产环境中的部署策略

## 29.1 协议概述与背景

### 29.1.1 为什么需要 Agent 互操作协议

随着 AI Agent 生态系统的快速发展，不同框架、不同平台构建的 Agent 需要实现互操作。传统的 REST API 或 gRPC 方式在处理 LLM 驱动的动态交互时面临诸多挑战：

| 传统协议 | 传统API问题 | Agent协议解决方案 |
|----------|-------------|-------------------|
| REST | 静态端点定义 | 动态工具发现与注册 |
| gRPC | 强类型schema | 灵活的JSON-RPC交互 |
| GraphQL | 查询结构固定 | 上下文感知的动态查询 |
| WebSocket | 无状态通信 | 有状态会话管理 |

### 29.1.2 MCP：Model Context Protocol

MCP（Model Context Protocol）由 Anthropic 于 2024 年 11 月开源发布，旨在标准化 LLM 应用与外部数据源、工具之间的连接方式。其核心理念是：

- **统一接口**：为所有 LLM 应用提供标准化的工具访问接口
- **生态共享**：构建可复用的工具与数据源生态系统
- **安全隔离**：在 LLM 应用与外部系统之间建立安全边界

### 29.1.3 A2A：Agent-to-Agent Protocol

A2A（Agent-to-Agent）由 Google 于 2025 年 4 月发布，专注于解决不同 Agent 系统之间的互操作问题：

- **Agent 发现**：通过 Agent Card 描述 Agent 的能力
- **任务管理**：标准化的 Task 生命周期管理
- **跨平台协作**：不同框架构建的 Agent 可以无缝协作

## 29.2 MCP 架构设计

### 29.2.1 三层架构模型

MCP 采用经典的 Client-Server 三层架构：

```
┌─────────────────────────────────────────────────┐
│                  LLM Application                 │
│            (Claude Desktop, IDE, etc.)            │
├─────────────────────────────────────────────────┤
│                  MCP Client                      │
│         (协议适配层，连接管理)                      │
├─────────────────────────────────────────────────┤
│                  MCP Server                      │
│      (工具/资源/提示词 的提供者)                    │
├─────────────────────────────────────────────────┤
│              External Systems                    │
│     (数据库, API, 文件系统, Web服务等)             │
└─────────────────────────────────────────────────┘
```

### 29.2.2 核心组件定义

```python
# MCP 核心组件抽象定义
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum
import uuid


class MCPRole(Enum):
    """MCP 协议角色定义"""
    CLIENT = "client"    # 客户端角色
    SERVER = "server"    # 服务端角色


class ProtocolVersion(Enum):
    """MCP 协议版本"""
    V2024_11_05 = "2024-11-05"
    V2025_03_26 = "2025-03-26"


@dataclass
class MCPTool:
    """MCP 工具定义 - Server 暴露给 Client 的能力单元"""
    name: str                          # 工具唯一名称
    description: str                   # 工具功能描述
    input_schema: dict[str, Any]       # JSON Schema 格式的输入参数定义
    
    def to_dict(self) -> dict:
        """序列化为 MCP 协议格式"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema
        }


@dataclass
class MCPResource:
    """MCP 资源定义 - Server 暴露的数据源"""
    uri: str                           # 资源 URI 标识符
    name: str                          # 资源显示名称
    description: str                   # 资源描述
    mime_type: str = "text/plain"      # MIME 类型
    
    def to_dict(self) -> dict:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type
        }


@dataclass
class MCPPrompt:
    """MCP 提示词模板 - Server 暴露的提示词片段"""
    name: str                          # 提示词模板名称
    description: str                   # 提示词描述
    arguments: list[dict] = field(default_factory=list)  # 可选参数
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments
        }


@dataclass
class MCPSessionConfig:
    """MCP 会话配置"""
    server_name: str                   # 服务器名称
    protocol_version: ProtocolVersion  # 协议版本
    capabilities: dict[str, Any] = field(default_factory=dict)  # 能力协商
    
    def to_init_params(self) -> dict:
        """生成初始化请求参数"""
        return {
            "protocolVersion": self.protocol_version.value,
            "capabilities": self.capabilities,
            "clientInfo": {
                "name": self.server_name,
                "version": "1.0.0"
            }
        }
```

### 29.2.3 传输层协议

MCP 支持多种传输机制：

| 传输类型 | 适用场景 | 特点 |
|----------|----------|------|
| stdio | 本地进程通信 | 低延迟，进程隔离 |
| SSE (Server-Sent Events) | 远程HTTP服务 | 单向流，HTTP兼容 |
| Streamable HTTP | 远程HTTP服务 | 双向流，新推荐方式 |
| WebSocket | 实时双向通信 | 低延迟，全双工 |

```python
# 不同传输层的实现对比
import asyncio
import json
from abc import ABC, abstractmethod
from typing import AsyncIterator
import aiohttp


class Transport(ABC):
    """MCP 传输层抽象基类"""
    
    @abstractmethod
    async def connect(self) -> None:
        """建立连接"""
        pass
    
    @abstractmethod
    async def send(self, message: dict) -> None:
        """发送消息"""
        pass
    
    @abstractmethod
    async def receive(self) -> dict:
        """接收消息"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """关闭连接"""
        pass


class StdioTransport(Transport):
    """标准输入输出传输 - 适用于本地 MCP Server"""
    
    def __init__(self, command: str, args: list[str] = None):
        self.command = command
        self.args = args or []
        self.process = None
    
    async def connect(self) -> None:
        """启动子进程并建立 stdio 连接"""
        self.process = await asyncio.create_subprocess_exec(
            self.command, *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        # 验证进程启动成功
        if self.process.returncode is not None:
            raise ConnectionError(
                f"进程启动失败: {self.command}"
            )
    
    async def send(self, message: dict) -> None:
        """通过 stdin 发送 JSON-RPC 消息"""
        raw = json.dumps(message) + "\n"
        self.process.stdin.write(raw.encode())
        await self.process.stdin.drain()
    
    async def receive(self) -> dict:
        """从 stdout 读取 JSON-RPC 响应"""
        line = await self.process.stdout.readline()
        if not line:
            raise ConnectionError("连接已断开")
        return json.loads(line.decode().strip())
    
    async def close(self) -> None:
        """关闭子进程"""
        if self.process:
            self.process.terminate()
            await self.process.wait()


class SSETransport(Transport):
    """SSE 传输 - 适用于远程 MCP Server"""
    
    def __init__(self, endpoint: str, headers: dict = None):
        self.endpoint = endpoint
        self.headers = headers or {}
        self.session = None
        self._event_queue: asyncio.Queue = asyncio.Queue()
    
    async def connect(self) -> None:
        """建立 SSE 连接"""
        self.session = aiohttp.ClientSession()
        # 启动 SSE 监听任务
        asyncio.create_task(self._listen_sse())
    
    async def _listen_sse(self) -> None:
        """监听 SSE 事件流"""
        async with self.session.get(
            self.endpoint,
            headers={"Accept": "text/event-stream", **self.headers}
        ) as response:
            async for line in response.content:
                decoded = line.decode().strip()
                if decoded.startswith("data:"):
                    data = decoded[5:].strip()
                    if data:
                        await self._event_queue.put(json.loads(data))
    
    async def send(self, message: dict) -> None:
        """通过 POST 发送请求"""
        async with self.session.post(
            self.endpoint,
            json=message,
            headers={"Content-Type": "application/json", **self.headers}
        ) as response:
            if response.status != 200:
                raise ConnectionError(f"发送失败: {response.status}")
    
    async def receive(self) -> dict:
        """从 SSE 事件队列获取消息"""
        return await self._event_queue.get()
    
    async def close(self) -> None:
        """关闭 SSE 连接"""
        if self.session:
            await self.session.close()


class StreamableHTTPTransport(Transport):
    """Streamable HTTP 传输 - MCP 2025 推荐方式"""
    
    def __init__(self, endpoint: str, headers: dict = None):
        self.endpoint = endpoint
        self.headers = headers or {}
        self.session = None
        self.session_id: str | None = None
    
    async def connect(self) -> None:
        """建立 Streamable HTTP 连接"""
        self.session = aiohttp.ClientSession()
    
    async def send(self, message: dict) -> None:
        """发送消息并处理响应"""
        request_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self.headers
        }
        if self.session_id:
            request_headers["Mcp-Session-Id"] = self.session_id
        
        async with self.session.post(
            self.endpoint,
            json=message,
            headers=request_headers
        ) as response:
            # 捕获 Session ID
            if "Mcp-Session-Id" in response.headers:
                self.session_id = response.headers["Mcp-Session-Id"]
            
            # 处理响应体
            content_type = response.headers.get("Content-Type", "")
            if "text/event-stream" in content_type:
                async for line in response.content:
                    decoded = line.decode().strip()
                    if decoded.startswith("data:"):
                        data = decoded[5:].strip()
                        if data:
                            return json.loads(data)
            else:
                return await response.json()
    
    async def receive(self) -> dict:
        """接收推送消息（Notification）"""
        # Streamable HTTP 使用长轮询或 SSE 接收服务端推送
        raise NotImplementedError("推送接收待实现")
    
    async def close(self) -> None:
        """关闭连接"""
        if self.session:
            await self.session.close()
```

### 29.2.4 消息协议格式

MCP 采用 JSON-RPC 2.0 作为消息格式基础：

```python
# MCP JSON-RPC 消息协议实现
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class JSONRPCMessageType(Enum):
    """JSON-RPC 消息类型"""
    REQUEST = "request"       # 请求消息
    RESPONSE = "response"     # 响应消息
    NOTIFICATION = "notification"  # 通知消息（无响应）


@dataclass
class JSONRPCRequest:
    """JSON-RPC 2.0 请求消息"""
    method: str                    # 方法名
    params: dict[str, Any] = field(default_factory=dict)  # 参数
    id: str | int = field(default_factory=lambda: str(int(time.time() * 1000)))
    
    def to_dict(self) -> dict:
        return {
            "jsonrpc": "2.0",
            "method": self.method,
            "params": self.params,
            "id": self.id
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "JSONRPCRequest":
        return cls(
            method=data["method"],
            params=data.get("params", {}),
            id=data.get("id")
        )


@dataclass
class JSONRPCResponse:
    """JSON-RPC 2.0 响应消息"""
    result: Any = None             # 成功结果
    error: Optional[dict] = None   # 错误信息
    id: str | int = None           # 对应请求 ID
    
    def to_dict(self) -> dict:
        msg = {"jsonrpc": "2.0", "id": self.id}
        if self.error:
            msg["error"] = self.error
        else:
            msg["result"] = self.result
        return msg
    
    @classmethod
    def from_dict(cls, data: dict) -> "JSONRPCResponse":
        return cls(
            result=data.get("result"),
            error=data.get("error"),
            id=data.get("id")
        )


@dataclass
class JSONRPCNotification:
    """JSON-RPC 2.0 通知消息（无需响应）"""
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "jsonrpc": "2.0",
            "method": self.method,
            "params": self.params
        }


@dataclass
class MCPError:
    """MCP 标准错误码定义"""
    CODE_PARSE_ERROR = -32700          # JSON 解析错误
    CODE_INVALID_REQUEST = -32600      # 无效请求
    CODE_METHOD_NOT_FOUND = -32601     # 方法未找到
    CODE_INVALID_PARAMS = -32602       # 参数无效
    CODE_INTERNAL_ERROR = -32603       # 内部错误
    
    # MCP 专用错误码（-32000 到 -32099）
    CODE_RESOURCE_NOT_FOUND = -32001   # 资源未找到
    CODE_TOOL_NOT_FOUND = -32002       # 工具未找到
    CODE_PROMPT_NOT_FOUND = -32003     # 提示词未找到
    CODE_AUTHENTICATION_FAILED = -32004  # 认证失败
    CODE_RATE_LIMITED = -32005         # 请求限流
    
    @staticmethod
    def create_error(code: int, message: str, data: Any = None) -> dict:
        """创建标准错误响应"""
        error = {"code": code, "message": message}
        if data:
            error["data"] = data
        return error


class MCPProtocolHandler:
    """MCP 协议消息处理器"""
    
    def __init__(self):
        self._request_id_counter = 0
    
    def create_request(self, method: str, params: dict = None) -> JSONRPCRequest:
        """创建请求消息"""
        self._request_id_counter += 1
        return JSONRPCRequest(
            method=method,
            params=params or {},
            id=self._request_id_counter
        )
    
    def create_response(
        self, request_id: str | int, result: Any = None, error: dict = None
    ) -> JSONRPCResponse:
        """创建响应消息"""
        return JSONRPCResponse(result=result, error=error, id=request_id)
    
    def create_notification(
        self, method: str, params: dict = None
    ) -> JSONRPCNotification:
        """创建通知消息（无 id，无需响应）"""
        return JSONRPCNotification(method=method, params=params or {})
    
    def parse_message(self, raw: str) -> tuple[JSONRPCMessageType, Any]:
        """解析原始消息，返回消息类型和对象"""
        data = json.loads(raw)
        
        if "method" in data and "id" in data:
            return JSONRPCMessageType.REQUEST, JSONRPCRequest.from_dict(data)
        elif "method" in data and "id" not in data:
            return JSONRPCMessageType.NOTIFICATION, JSONRPCNotification.from_dict(data)
        else:
            return JSONRPCMessageType.RESPONSE, JSONRPCResponse.from_dict(data)
```

## 29.3 MCP Server 实现

### 29.3.1 Server 能力注册

```python
# MCP Server 完整实现
import asyncio
import json
import inspect
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from functools import wraps


class MCPServer:
    """MCP Server 核心实现
    
    负责管理工具、资源、提示词的注册与生命周期
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        
        # 能力注册表
        self._tools: dict[str, MCPTool] = {}
        self._resources: dict[str, MCPResource] = {}
        self._prompts: dict[str, MCPPrompt] = {}
        
        # 处理函数映射
        self._tool_handlers: dict[str, Callable] = {}
        self._resource_handlers: dict[str, Callable] = {}
        self._prompt_handlers: dict[str, Callable] = {}
        
        # 协议处理器
        self._protocol = MCPProtocolHandler()
        
        # 初始化状态
        self._initialized = False
    
    # ========== 工具注册 ==========
    
    def tool(
        self,
        name: str = None,
        description: str = None
    ) -> Callable:
        """工具装饰器 - 将函数注册为 MCP 工具
        
        用法：
            @server.tool(description="查询天气")
            def get_weather(city: str) -> str:
                return f"{city}的天气是晴天"
        """
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = description or func.__doc__ or f"执行 {tool_name}"
            
            # 从函数签名生成 JSON Schema
            input_schema = self._generate_schema(func)
            
            # 注册工具
            tool = MCPTool(
                name=tool_name,
                description=tool_desc,
                input_schema=input_schema
            )
            self._tools[tool_name] = tool
            self._tool_handlers[tool_name] = func
            
            @wraps(func)
            async def wrapper(**kwargs):
                return await func(**kwargs) if asyncio.iscoroutinefunction(func) else func(**kwargs)
            
            return wrapper
        
        return decorator
    
    def _generate_schema(self, func: Callable) -> dict:
        """从函数签名自动生成 JSON Schema"""
        sig = inspect.signature(func)
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            # 获取类型注解
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                type_map = {
                    str: "string",
                    int: "integer",
                    float: "number",
                    bool: "boolean",
                    list: "array",
                    dict: "object"
                }
                param_type = type_map.get(param.annotation, "string")
            
            properties[param_name] = {
                "type": param_type,
                "description": f"参数 {param_name}"
            }
            
            # 无默认值的参数为必填
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        return {
            "type": "object",
            "properties": properties,
            "required": required
        }
    
    # ========== 资源注册 ==========
    
    def resource(
        self,
        uri: str,
        name: str = None,
        mime_type: str = "text/plain"
    ) -> Callable:
        """资源装饰器 - 将函数注册为 MCP 资源
        
        用法：
            @server.resource("file:///docs/readme", name="README")
            def get_readme() -> str:
                return open("README.md").read()
        """
        def decorator(func: Callable) -> Callable:
            resource_name = name or func.__name__
            
            res = MCPResource(
                uri=uri,
                name=resource_name,
                description=func.__doc__ or f"资源 {resource_name}",
                mime_type=mime_type
            )
            self._resources[uri] = res
            self._resource_handlers[uri] = func
            
            return func
        
        return decorator
    
    # ========== 提示词注册 ==========
    
    def prompt(
        self,
        name: str = None,
        description: str = None,
        arguments: list[dict] = None
    ) -> Callable:
        """提示词装饰器 - 将函数注册为 MCP 提示词模板
        
        用法：
            @server.prompt(description="代码审查提示词")
            def code_review(code: str, language: str) -> str:
                return f"请审查以下{language}代码:\n{code}"
        """
        def decorator(func: Callable) -> Callable:
            prompt_name = name or func.__name__
            
            prompt = MCPPrompt(
                name=prompt_name,
                description=description or func.__doc__ or f"提示词 {prompt_name}",
                arguments=arguments or []
            )
            self._prompts[prompt_name] = prompt
            self._prompt_handlers[prompt_name] = func
            
            return func
        
        return decorator
    
    # ========== 协议处理 ==========
    
    def get_capabilities(self) -> dict:
        """返回服务器能力描述"""
        caps = {}
        if self._tools:
            caps["tools"] = {"listChanged": True}
        if self._resources:
            caps["resources"] = {
                "subscribe": True,
                "listChanged": True
            }
        if self._prompts:
            caps["prompts"] = {"listChanged": True}
        return caps
    
    async def handle_initialize(self, params: dict) -> dict:
        """处理初始化请求"""
        self._initialized = True
        return {
            "protocolVersion": ProtocolVersion.V2025_03_26.value,
            "capabilities": self.get_capabilities(),
            "serverInfo": {
                "name": self.name,
                "version": self.version
            }
        }
    
    async def handle_tools_list(self, params: dict = None) -> dict:
        """处理工具列表请求"""
        tools = [tool.to_dict() for tool in self._tools.values()]
        return {"tools": tools}
    
    async def handle_tools_call(self, params: dict) -> dict:
        """处理工具调用请求"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name not in self._tool_handlers:
            return {
                "content": [{
                    "type": "text",
                    "text": f"错误：工具 '{tool_name}' 未找到"
                }],
                "isError": True
            }
        
        try:
            handler = self._tool_handlers[tool_name]
            result = handler(**arguments)
            if asyncio.iscoroutine(result):
                result = await result
            
            return {
                "content": [{
                    "type": "text",
                    "text": str(result)
                }]
            }
        except Exception as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"执行错误: {str(e)}"
                }],
                "isError": True
            }
    
    async def handle_resources_list(self, params: dict = None) -> dict:
        """处理资源列表请求"""
        resources = [res.to_dict() for res in self._resources.values()]
        return {"resources": resources}
    
    async def handle_resources_read(self, params: dict) -> dict:
        """处理资源读取请求"""
        uri = params.get("uri")
        
        if uri not in self._resource_handlers:
            return {
                "contents": [{
                    "uri": uri,
                    "text": f"错误：资源 '{uri}' 未找到"
                }]
            }
        
        try:
            handler = self._resource_handlers[uri]
            content = handler()
            if asyncio.iscoroutine(content):
                content = await content
            
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": self._resources[uri].mime_type,
                    "text": str(content)
                }]
            }
        except Exception as e:
            return {
                "contents": [{
                    "uri": uri,
                    "text": f"读取错误: {str(e)}"
                }]
            }
    
    async def handle_prompts_list(self, params: dict = None) -> dict:
        """处理提示词列表请求"""
        prompts = [p.to_dict() for p in self._prompts.values()]
        return {"prompts": prompts}
    
    async def handle_prompts_get(self, params: dict) -> dict:
        """处理提示词获取请求"""
        prompt_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if prompt_name not in self._prompt_handlers:
            return {
                "error": f"提示词 '{prompt_name}' 未找到"
            }
        
        try:
            handler = self._prompt_handlers[prompt_name]
            content = handler(**arguments)
            if asyncio.iscoroutine(content):
                content = await content
            
            return {
                "description": self._prompts[prompt_name].description,
                "messages": [{
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": str(content)
                    }
                }]
            }
        except Exception as e:
            return {
                "error": f"获取错误: {str(e)}"
            }
    
    async def process_message(self, raw_message: str) -> str:
        """处理单条 MCP 消息"""
        try:
            msg_type, message = self._protocol.parse_message(raw_message)
            
            if msg_type == JSONRPCMessageType.REQUEST:
                method = message.method
                params = message.params
                
                # 路由到对应处理器
                handler_map = {
                    "initialize": self.handle_initialize,
                    "tools/list": self.handle_tools_list,
                    "tools/call": self.handle_tools_call,
                    "resources/list": self.handle_resources_list,
                    "resources/read": self.handle_resources_read,
                    "prompts/list": self.handle_prompts_list,
                    "prompts/get": self.handle_prompts_get,
                }
                
                if method in handler_map:
                    result = await handler_map[method](params)
                    response = self._protocol.create_response(
                        message.id, result=result
                    )
                else:
                    error = MCPError.create_error(
                        MCPError.CODE_METHOD_NOT_FOUND,
                        f"方法 '{method}' 未找到"
                    )
                    response = self._protocol.create_response(
                        message.id, error=error
                    )
                
                return json.dumps(response.to_dict())
            
            elif msg_type == JSONRPCMessageType.NOTIFICATION:
                # 通知消息无需响应
                if message.method == "notifications/initialized":
                    self._initialized = True
                return None
            
            else:
                return None  # 响应消息由 Client 处理
                
        except json.JSONDecodeError as e:
            error = MCPError.create_error(
                MCPError.CODE_PARSE_ERROR, str(e)
            )
            return json.dumps({"jsonrpc": "2.0", "error": error})
```

### 29.3.2 工具调用执行链

```python
# MCP 工具调用执行链管理
from datetime import datetime
from typing import Any
import hashlib
import hmac


class ToolExecutionChain:
    """工具调用执行链 - 管理工具调用的生命周期"""
    
    def __init__(self, server: MCPServer):
        self.server = server
        self._execution_log: list[dict] = []
    
    async def execute_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: dict = None
    ) -> dict:
        """执行工具调用并记录日志
        
        Args:
            tool_name: 工具名称
            arguments: 调用参数
            context: 调用上下文（包含调用者信息等）
        
        Returns:
            执行结果
        """
        execution_id = hashlib.md5(
            f"{tool_name}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        start_time = datetime.now()
        
        # 记录调用开始
        log_entry = {
            "execution_id": execution_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "start_time": start_time.isoformat(),
            "status": "running",
            "context": context or {}
        }
        
        try:
            # 验证参数
            self._validate_arguments(tool_name, arguments)
            
            # 执行工具
            handler = self.server._tool_handlers[tool_name]
            result = handler(**arguments)
            if asyncio.iscoroutine(result):
                result = await result
            
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            # 更新日志
            log_entry.update({
                "end_time": end_time.isoformat(),
                "duration_ms": duration_ms,
                "status": "success",
                "result_preview": str(result)[:200]
            })
            self._execution_log.append(log_entry)
            
            return {
                "content": [{"type": "text", "text": str(result)}],
                "isError": False,
                "_meta": {
                    "execution_id": execution_id,
                    "duration_ms": duration_ms
                }
            }
            
        except Exception as e:
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            log_entry.update({
                "end_time": end_time.isoformat(),
                "duration_ms": duration_ms,
                "status": "error",
                "error": str(e)
            })
            self._execution_log.append(log_entry)
            
            return {
                "content": [{"type": "text", "text": f"执行错误: {e}"}],
                "isError": True,
                "_meta": {
                    "execution_id": execution_id,
                    "duration_ms": duration_ms
                }
            }
    
    def _validate_arguments(self, tool_name: str, arguments: dict) -> None:
        """验证工具调用参数"""
        tool = self.server._tools.get(tool_name)
        if not tool:
            raise ValueError(f"工具 '{tool_name}' 不存在")
        
        schema = tool.input_schema
        required = schema.get("required", [])
        
        for field in required:
            if field not in arguments:
                raise ValueError(f"缺少必填参数: {field}")
    
    def get_execution_history(
        self, limit: int = 100
    ) -> list[dict]:
        """获取执行历史"""
        return self._execution_log[-limit:]
    
    def get_statistics(self) -> dict:
        """获取执行统计"""
        if not self._execution_log:
            return {"total": 0}
        
        total = len(self._execution_log)
        success = sum(1 for e in self._execution_log if e["status"] == "success")
        failed = sum(1 for e in self._execution_log if e["status"] == "error")
        avg_duration = (
            sum(e.get("duration_ms", 0) for e in self._execution_log) / total
        )
        
        return {
            "total": total,
            "success": success,
            "failed": failed,
            "success_rate": success / total if total > 0 else 0,
            "avg_duration_ms": round(avg_duration, 2)
        }
```

### 29.3.3 资源订阅与变更通知

```python
# MCP 资源订阅管理
from typing import Set
from collections import defaultdict


class ResourceManager:
    """MCP 资源管理器 - 处理资源订阅与变更通知"""
    
    def __init__(self):
        # URI -> 订阅者列表
        self._subscribers: dict[str, Set[str]] = defaultdict(set)
        # URI -> 最新内容
        self._resource_cache: dict[str, Any] = {}
        # URI -> 内容哈希（用于变更检测）
        self._content_hashes: dict[str, str] = {}
    
    def subscribe(
        self, uri: str, subscriber_id: str
    ) -> None:
        """订阅资源变更
        
        Args:
            uri: 资源 URI
            subscriber_id: 订阅者标识
        """
        self._subscribers[uri].add(subscriber_id)
        print(f"[资源管理] {subscriber_id} 订阅了 {uri}")
    
    def unsubscribe(
        self, uri: str, subscriber_id: str
    ) -> None:
        """取消订阅"""
        self._subscribers[uri].discard(subscriber_id)
    
    async def notify_change(
        self, uri: str, new_content: Any
    ) -> list[str]:
        """通知订阅者资源变更
        
        计算内容哈希，仅在内容实际变化时发送通知
        """
        import hashlib
        
        content_str = str(new_content)
        new_hash = hashlib.sha256(content_str.encode()).hexdigest()
        
        # 检查内容是否真的变化了
        old_hash = self._content_hashes.get(uri)
        if old_hash == new_hash:
            return []  # 内容未变化，跳过通知
        
        self._content_hashes[uri] = new_hash
        self._resource_cache[uri] = new_content
        
        # 获取订阅者列表
        subscribers = list(self._subscribers.get(uri, set()))
        
        if subscribers:
            print(
                f"[资源管理] 通知 {len(subscribers)} 个订阅者："
                f"{uri} 已更新"
            )
        
        return subscribers
    
    def get_cached_content(self, uri: str) -> Any:
        """获取缓存的资源内容"""
        return self._resource_cache.get(uri)
```

## 29.4 A2A 协议详解

### 29.4.1 A2A 核心概念

A2A 协议定义了 Agent 之间交互的核心实体：

| 核心概念 | 描述 | 生命周期 |
|----------|------|----------|
| Agent Card | Agent 的元数据描述 | 静态发布 |
| Task | Agent 间协作的任务单元 | 创建→进行→完成 |
| Artifact | Task 产出的结果 | 由 Task 生成 |
| Message | Agent 间的消息交互 | 在 Task 内流转 |
| Part | Message/Artifact 的内容单元 | 多类型支持 |

### 29.4.2 Agent Card 设计

```python
# A2A Agent Card 定义与实现
from dataclasses import dataclass, field
from typing import Any
import json


@dataclass
class AgentSkill:
    """Agent 技能描述"""
    id: str                         # 技能唯一 ID
    name: str                       # 技能名称
    description: str                # 技能描述
    tags: list[str] = field(default_factory=list)  # 标签
    examples: list[str] = field(default_factory=list)  # 示例
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "examples": self.examples
        }


@dataclass
class AgentAuthentication:
    """Agent 认证配置"""
    schemes: list[str] = field(default_factory=lambda: ["bearer"])
    credentials: str | None = None  # OAuth2 等凭证 URL
    
    def to_dict(self) -> dict:
        result = {"schemes": self.schemes}
        if self.credentials:
            result["credentials"] = self.credentials
        return result


@dataclass
class AgentProvider:
    """Agent 提供者信息"""
    organization: str               # 组织名称
    url: str                        # 组织 URL
    
    def to_dict(self) -> dict:
        return {
            "organization": self.organization,
            "url": self.url
        }


@dataclass
class AgentCard:
    """A2A Agent Card - Agent 能力的完整描述
    
    发布在 well-known URI: /.well-known/agent.json
    """
    name: str                             # Agent 名称
    description: str                      # Agent 功能描述
    url: str                              # Agent 服务地址
    version: str = "1.0.0"                # 版本号
    documentation_url: str | None = None  # 文档地址
    provider: AgentProvider | None = None  # 提供者信息
    authentication: AgentAuthentication | None = None  # 认证配置
    default_input_modes: list[str] = field(
        default_factory=lambda: ["text"]
    )  # 支持的输入模式
    default_output_modes: list[str] = field(
        default_factory=lambda: ["text"]
    )  # 支持的输出模式
    skills: list[AgentSkill] = field(default_factory=list)  # 技能列表
    capabilities: dict[str, bool] = field(default_factory=dict)  # 能力标志
    
    def to_dict(self) -> dict:
        result = {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "defaultInputModes": self.default_input_modes,
            "defaultOutputModes": self.default_output_modes,
            "skills": [s.to_dict() for s in self.skills],
            "capabilities": self.capabilities
        }
        if self.documentation_url:
            result["documentationUrl"] = self.documentation_url
        if self.provider:
            result["provider"] = self.provider.to_dict()
        if self.authentication:
            result["authentication"] = self.authentication.to_dict()
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """序列化为 JSON 字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: dict) -> "AgentCard":
        """从字典反序列化"""
        skills = [
            AgentSkill(**s) for s in data.get("skills", [])
        ]
        provider = None
        if "provider" in data:
            provider = AgentProvider(**data["provider"])
        auth = None
        if "authentication" in data:
            auth = AgentAuthentication(**data["authentication"])
        
        return cls(
            name=data["name"],
            description=data["description"],
            url=data["url"],
            version=data.get("version", "1.0.0"),
            documentation_url=data.get("documentationUrl"),
            provider=provider,
            authentication=auth,
            default_input_modes=data.get("defaultInputModes", ["text"]),
            default_output_modes=data.get("defaultOutputModes", ["text"]),
            skills=skills,
            capabilities=data.get("capabilities", {})
        )
```

### 29.4.3 Task 生命周期管理

```python
# A2A Task 生命周期管理
import asyncio
import uuid
from datetime import datetime
from enum import Enum
from typing import Any
from dataclasses import dataclass, field


class TaskState(Enum):
    """A2A Task 状态"""
    SUBMITTED = "submitted"       # 已提交
    WORKING = "working"           # 工作中
    INPUT_REQUIRED = "input-required"  # 需要输入
    COMPLETED = "completed"       # 已完成
    CANCELED = "canceled"         # 已取消
    FAILED = "failed"            # 已失败


@dataclass
class MessagePart:
    """消息内容单元"""
    type: str                     # text, file, data
    text: str | None = None       # 文本内容
    file_uri: str | None = None   # 文件 URI
    mime_type: str | None = None  # MIME 类型
    data: Any = None              # 结构化数据
    
    def to_dict(self) -> dict:
        result = {"type": self.type}
        if self.text is not None:
            result["text"] = self.text
        if self.file_uri:
            result["fileUri"] = self.file_uri
        if self.mime_type:
            result["mimeType"] = self.mime_type
        if self.data is not None:
            result["data"] = self.data
        return result


@dataclass
class A2AMessage:
    """A2A 消息"""
    role: str                          # "user" 或 "agent"
    parts: list[MessagePart]           # 消息内容
    
    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "parts": [p.to_dict() for p in self.parts]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "A2AMessage":
        parts = [
            MessagePart(**p) for p in data.get("parts", [])
        ]
        return cls(role=data["role"], parts=parts)


@dataclass
class A2AArtifact:
    """A2A 产出物"""
    name: str                          # 产出物名称
    parts: list[MessagePart]           # 产出物内容
    description: str = ""              # 描述
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parts": [p.to_dict() for p in self.parts],
            "metadata": self.metadata
        }


@dataclass
class A2ATask:
    """A2A Task - Agent 间协作的任务单元"""
    id: str                              # 任务唯一 ID
    session_id: str                      # 会话 ID
    state: TaskState = TaskState.SUBMITTED  # 当前状态
    messages: list[A2AMessage] = field(default_factory=list)
    artifacts: list[A2AArtifact] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)  # 状态变更历史
    metadata: dict = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    
    def transition(self, new_state: TaskState, message: str = "") -> None:
        """状态转换"""
        old_state = self.state
        self.state = new_state
        self.updated_at = datetime.now().isoformat()
        
        # 记录状态变更历史
        self.history.append({
            "from": old_state.value,
            "to": new_state.value,
            "timestamp": self.updated_at,
            "message": message
        })
    
    def add_message(self, message: A2AMessage) -> None:
        """添加消息"""
        self.messages.append(message)
        self.updated_at = datetime.now().isoformat()
    
    def add_artifact(self, artifact: A2AArtifact) -> None:
        """添加产出物"""
        self.artifacts.append(artifact)
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sessionId": self.session_id,
            "state": self.state.value,
            "messages": [m.to_dict() for m in self.messages],
            "artifacts": [a.to_dict() for a in self.artifacts],
            "history": self.history,
            "metadata": self.metadata,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at
        }


class A2ATaskManager:
    """A2A Task 生命周期管理器"""
    
    def __init__(self):
        self._tasks: dict[str, A2ATask] = {}
        self._session_tasks: dict[str, list[str]] = {}
    
    def create_task(
        self,
        session_id: str,
        initial_message: A2AMessage | None = None,
        metadata: dict = None
    ) -> A2ATask:
        """创建新任务"""
        task_id = str(uuid.uuid4())
        task = A2ATask(
            id=task_id,
            session_id=session_id,
            metadata=metadata or {}
        )
        
        if initial_message:
            task.add_message(initial_message)
        
        self._tasks[task_id] = task
        
        if session_id not in self._session_tasks:
            self._session_tasks[session_id] = []
        self._session_tasks[session_id].append(task_id)
        
        return task
    
    def get_task(self, task_id: str) -> A2ATask | None:
        """获取任务"""
        return self._tasks.get(task_id)
    
    def update_task_state(
        self, task_id: str, new_state: TaskState, message: str = ""
    ) -> bool:
        """更新任务状态"""
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        # 状态转换验证
        valid_transitions = {
            TaskState.SUBMITTED: {TaskState.WORKING, TaskState.CANCELED},
            TaskState.WORKING: {
                TaskState.COMPLETED, TaskState.FAILED,
                TaskState.INPUT_REQUIRED
            },
            TaskState.INPUT_REQUIRED: {TaskState.WORKING, TaskState.CANCELED},
        }
        
        allowed = valid_transitions.get(task.state, set())
        if new_state not in allowed:
            raise ValueError(
                f"无效状态转换: {task.state.value} -> {new_state.value}"
            )
        
        task.transition(new_state, message)
        return True
    
    def add_message_to_task(
        self, task_id: str, message: A2AMessage
    ) -> bool:
        """向任务添加消息"""
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        task.add_message(message)
        return True
    
    def get_session_tasks(self, session_id: str) -> list[A2ATask]:
        """获取会话的所有任务"""
        task_ids = self._session_tasks.get(session_id, [])
        return [self._tasks[tid] for tid in task_ids if tid in self._tasks]
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24) -> int:
        """清理已完成的旧任务"""
        now = datetime.now()
        cleaned = 0
        
        for task_id, task in list(self._tasks.items()):
            if task.state in (TaskState.COMPLETED, TaskState.CANCELED):
                from datetime import timedelta
                updated = datetime.fromisoformat(task.updated_at)
                if now - updated > timedelta(hours=max_age_hours):
                    del self._tasks[task_id]
                    cleaned += 1
        
        return cleaned
```

### 29.4.4 A2A Client 实现

```python
# A2A Client 实现
import aiohttp
import json
from typing import Any


class A2AClient:
    """A2A 协议客户端 - 用于与远程 Agent 交互"""
    
    def __init__(
        self,
        agent_card: AgentCard,
        auth_token: str | None = None
    ):
        self.agent_card = agent_card
        self.base_url = agent_card.url.rstrip("/")
        self.auth_token = auth_token
        self.session: aiohttp.ClientSession | None = None
    
    async def _ensure_session(self) -> None:
        """确保 HTTP 会话存在"""
        if not self.session:
            headers = {"Content-Type": "application/json"}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            self.session = aiohttp.ClientSession(headers=headers)
    
    async def send_task(
        self,
        task: A2ATask,
        message: A2AMessage
    ) -> A2ATask:
        """发送任务和消息到远程 Agent"""
        await self._ensure_session()
        
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "id": task.id,
                "sessionId": task.session_id,
                "message": message.to_dict(),
                "metadata": task.metadata
            },
            "id": str(uuid.uuid4())
        }
        
        async with self.session.post(
            f"{self.base_url}/a2a",
            json=payload
        ) as response:
            result = await response.json()
            
            if "error" in result:
                raise A2AError(result["error"]["message"])
            
            # 解析响应中的 Task
            task_data = result.get("result", {})
            return self._parse_task(task_data)
    
    async def get_task(
        self, task_id: str, session_id: str
    ) -> A2ATask:
        """获取任务状态"""
        await self._ensure_session()
        
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/get",
            "params": {
                "id": task_id,
                "sessionId": session_id
            },
            "id": str(uuid.uuid4())
        }
        
        async with self.session.post(
            f"{self.base_url}/a2a",
            json=payload
        ) as response:
            result = await response.json()
            
            if "error" in result:
                raise A2AError(result["error"]["message"])
            
            return self._parse_task(result.get("result", {}))
    
    async def cancel_task(
        self, task_id: str, session_id: str
    ) -> bool:
        """取消任务"""
        await self._ensure_session()
        
        payload = {
            "jsonrpc": "2.0",
            "method": "tasks/cancel",
            "params": {
                "id": task_id,
                "sessionId": session_id
            },
            "id": str(uuid.uuid4())
        }
        
        async with self.session.post(
            f"{self.base_url}/a2a",
            json=payload
        ) as response:
            result = await response.json()
            return "error" not in result
    
    def _parse_task(self, data: dict) -> A2ATask:
        """解析 Task 数据"""
        task = A2ATask(
            id=data.get("id", ""),
            session_id=data.get("sessionId", ""),
            state=TaskState(data.get("state", "submitted")),
            metadata=data.get("metadata", {}),
            created_at=data.get("createdAt", ""),
            updated_at=data.get("updatedAt", "")
        )
        
        for msg_data in data.get("messages", []):
            task.add_message(A2AMessage.from_dict(msg_data))
        
        for art_data in data.get("artifacts", []):
            artifact = A2AArtifact(
                name=art_data.get("name", ""),
                parts=[MessagePart(**p) for p in art_data.get("parts", [])],
                description=art_data.get("description", ""),
                metadata=art_data.get("metadata", {})
            )
            task.add_artifact(artifact)
        
        return task
    
    async def close(self) -> None:
        """关闭 HTTP 会话"""
        if self.session:
            await self.session.close()


class A2AError(Exception):
    """A2A 协议错误"""
    pass
```

### 29.4.5 A2A Server 实现

```python
# A2A Server 完整实现
from aiohttp import web
import uuid


class A2AServer:
    """A2A 协议服务端 - 提供标准 A2A 接口"""
    
    def __init__(self, agent_card: AgentCard):
        self.agent_card = agent_card
        self.task_manager = A2ATaskManager()
        self._handlers: dict[str, Callable] = {}
        
        # 注册默认处理器
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> None:
        """注册默认的任务处理器"""
        self._handlers["tasks/send"] = self._handle_send
        self._handlers["tasks/get"] = self._handle_get
        self._handlers["tasks/cancel"] = self._handle_cancel
    
    async def _handle_send(self, params: dict) -> dict:
        """处理任务发送"""
        task_id = params.get("id", str(uuid.uuid4()))
        session_id = params.get("sessionId", str(uuid.uuid4()))
        message_data = params.get("message", {})
        metadata = params.get("metadata", {})
        
        # 创建或获取任务
        task = self.task_manager.get_task(task_id)
        if not task:
            message = A2AMessage.from_dict(message_data)
            task = self.task_manager.create_task(
                session_id=session_id,
                initial_message=message,
                metadata=metadata
            )
        else:
            message = A2AMessage.from_dict(message_data)
            task.add_message(message)
        
        # 转换到工作状态
        self.task_manager.update_task_state(
            task.id, TaskState.WORKING, "开始处理任务"
        )
        
        # 调用业务逻辑处理器
        if "tasks/process" in self._handlers:
            try:
                await self._handlers["tasks/process"](task)
            except Exception as e:
                self.task_manager.update_task_state(
                    task.id, TaskState.FAILED, str(e)
                )
        
        return task.to_dict()
    
    async def _handle_get(self, params: dict) -> dict:
        """获取任务状态"""
        task_id = params.get("id")
        task = self.task_manager.get_task(task_id)
        
        if not task:
            raise A2AError(f"任务 '{task_id}' 未找到")
        
        return task.to_dict()
    
    async def _handle_cancel(self, params: dict) -> dict:
        """取消任务"""
        task_id = params.get("id")
        task = self.task_manager.get_task(task_id)
        
        if not task:
            raise A2AError(f"任务 '{task_id}' 未找到")
        
        self.task_manager.update_task_state(
            task_id, TaskState.CANCELED, "用户请求取消"
        )
        
        return {"status": "canceled"}
    
    def register_handler(
        self, method: str, handler: Callable
    ) -> None:
        """注册自定义处理器"""
        self._handlers[method] = handler
    
    async def process_request(self, request: web.Request) -> web.Response:
        """处理 A2A JSON-RPC 请求"""
        try:
            body = await request.json()
            
            method = body.get("method")
            params = body.get("params", {})
            request_id = body.get("id")
            
            if method not in self._handlers:
                return web.json_response({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": f"方法 '{method}' 未找到"
                    },
                    "id": request_id
                })
            
            result = await self._handlers[method](params)
            
            return web.json_response({
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id
            })
            
        except A2AError as e:
            return web.json_response({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32000,
                    "message": str(e)
                },
                "id": body.get("id") if 'body' in locals() else None
            })
        except Exception as e:
            return web.json_response({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"内部错误: {e}"
                },
                "id": body.get("id") if 'body' in locals() else None
            })
    
    async def agent_card_handler(
        self, request: web.Request
    ) -> web.Response:
        """提供 Agent Card 端点"""
        return web.json_response(
            self.agent_card.to_dict(),
            content_type="application/json"
        )
    
    def create_app(self) -> web.Application:
        """创建 aiohttp 应用"""
        app = web.Application()
        
        # Agent Card 端点
        app.router.add_get(
            "/.well-known/agent.json",
            self.agent_card_handler
        )
        
        # A2A JSON-RPC 端点
        app.router.add_post("/a2a", self.process_request)
        
        return app
```

## 29.5 MCP vs A2A 对比分析

### 29.5.1 核心差异对比表

| 维度 | MCP | A2A |
|------|-----|-----|
| **定位** | LLM ↔ 工具/数据源 | Agent ↔ Agent |
| **发起者** | Anthropic (2024.11) | Google (2025.04) |
| **协议基础** | JSON-RPC 2.0 | JSON-RPC 2.0 |
| **核心实体** | Tool, Resource, Prompt | Task, Message, Artifact |
| **通信模式** | Client-Server | Peer-to-Peer |
| **发现机制** | 配置文件/环境变量 | Agent Card (.well-known) |
| **状态管理** | 无状态/有状态会话 | Task 状态机 |
| **认证方式** | 由 Server 实现 | OAuth2 / Bearer Token |
| **适用场景** | 工具集成、数据访问 | 多 Agent 协作 |
| **生态成熟度** | 高（已有数百个 Server） | 早期（快速成长中） |

### 29.5.2 架构对比图

```
MCP 架构：
┌─────────┐     ┌─────────┐     ┌─────────┐
│  LLM    │────▶│  MCP    │────▶│  MCP    │
│  App    │◀────│  Client │◀────│  Server │
└─────────┘     └─────────┘     └─────────┘
                                      │
                                      ▼
                                 ┌─────────┐
                                 │外部系统  │
                                 │(DB,API) │
                                 └─────────┘

A2A 架构：
┌─────────┐                    ┌─────────┐
│ Agent A │◀──── A2A ────▶    │ Agent B │
└─────────┘                    └─────────┘
     │                              │
     ▼                              ▼
┌─────────┐                    ┌─────────┐
│ MCP     │                    │ MCP     │
│ Server  │                    │ Server  │
└─────────┘                    └─────────┘
```

### 29.5.3 选型决策指南

```python
# Agent 协议选型决策器
from dataclasses import dataclass
from enum import Enum


class IntegrationType(Enum):
    """集成类型"""
    TOOL_ACCESS = "tool_access"       # 访问外部工具
    DATA_SOURCE = "data_source"       # 访问数据源
    AGENT_COLLABORATION = "agent_collab"  # Agent 间协作
    MULTI_AGENT_SYSTEM = "multi_agent"    # 多 Agent 系统


@dataclass
class ProtocolRecommendation:
    """协议推荐结果"""
    primary: str                       # 推荐的主协议
    secondary: str | None              # 可选的辅助协议
    reasoning: str                     # 推荐理由
    implementation_notes: list[str]    # 实现注意事项


class ProtocolSelector:
    """协议选型决策器"""
    
    def recommend(
        self,
        integration_type: IntegrationType,
        num_agents: int = 1,
        needs_state: bool = False,
        cross_platform: bool = False
    ) -> ProtocolRecommendation:
        """根据需求推荐协议"""
        
        if integration_type == IntegrationType.TOOL_ACCESS:
            return ProtocolRecommendation(
                primary="MCP",
                secondary=None,
                reasoning="MCP 专为 LLM 与工具集成设计，生态成熟",
                implementation_notes=[
                    "选择 stdio 传输用于本地工具",
                    "选择 Streamable HTTP 用于远程服务",
                    "使用 @server.tool 装饰器简化注册"
                ]
            )
        
        elif integration_type == IntegrationType.DATA_SOURCE:
            return ProtocolRecommendation(
                primary="MCP",
                secondary=None,
                reasoning="MCP Resource 机制专门用于数据源访问",
                implementation_notes=[
                    "使用 URI 标识数据资源",
                    "实现资源订阅机制获取实时更新",
                    "注意资源缓存策略"
                ]
            )
        
        elif integration_type == IntegrationType.AGENT_COLLABORATION:
            return ProtocolRecommendation(
                primary="A2A",
                secondary="MCP",
                reasoning="A2A 专为 Agent 间协作设计，MCP 用于工具访问",
                implementation_notes=[
                    "发布 Agent Card 到 .well-known/agent.json",
                    "实现 Task 生命周期管理",
                    "每个 Agent 通过 MCP 访问其本地工具"
                ]
            )
        
        elif integration_type == IntegrationType.MULTI_AGENT_SYSTEM:
            return ProtocolRecommendation(
                primary="A2A",
                secondary="MCP",
                reasoning="多 Agent 系统需要 A2A 进行协调，MCP 进行工具集成",
                implementation_notes=[
                    "设计清晰的 Agent 角色分工",
                    "实现 Agent 发现与注册中心",
                    "使用 A2A Task 管理协作流程",
                    "每个 Agent 独立管理其 MCP Server"
                ]
            )
        
        raise ValueError(f"未知集成类型: {integration_type}")


# 使用示例
def demonstrate_selection():
    """演示协议选型"""
    selector = ProtocolSelector()
    
    # 场景1：构建一个查询数据库的工具
    rec = selector.recommend(IntegrationType.TOOL_ACCESS)
    print(f"场景1 - 推荐: {rec.primary}")
    print(f"  理由: {rec.reasoning}")
    
    # 场景2：构建两个 Agent 协作完成代码审查
    rec = selector.recommend(
        IntegrationType.AGENT_COLLABORATION,
        num_agents=2
    )
    print(f"场景2 - 推荐: {rec.primary} + {rec.secondary}")
    print(f"  理由: {rec.reasoning}")
    
    # 场景3：构建多 Agent 自动化流水线
    rec = selector.recommend(
        IntegrationType.MULTI_AGENT_SYSTEM,
        num_agents=5,
        needs_state=True,
        cross_platform=True
    )
    print(f"场景3 - 推荐: {rec.primary} + {rec.secondary}")
    print(f"  理由: {rec.reasoning}")
```

### 29.5.4 两者结合的最佳实践

```python
# MCP + A2A 结合使用的架构模式
class HybridAgentArchitecture:
    """混合架构模式 - MCP 处理工具访问，A2A 处理 Agent 协作"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        
        # MCP Server - 管理本地工具
        self.mcp_server = MCPServer(name=f"{agent_name}-mcp")
        
        # A2A Agent Card
        self.agent_card = AgentCard(
            name=agent_name,
            description=f"{agent_name} Agent",
            url=f"http://localhost:8000",
            skills=[],
            capabilities={"streaming": True}
        )
        
        # A2A Task Manager
        self.a2a_task_manager = A2ATaskManager()
    
    def register_tool(self, name: str, description: str) -> Callable:
        """注册本地工具（通过 MCP 暴露）"""
        return self.mcp_server.tool(name=name, description=description)
    
    def register_skill(
        self, skill_id: str, name: str, description: str
    ) -> None:
        """注册 Agent 技能（通过 A2A Card 暴露）"""
        skill = AgentSkill(
            id=skill_id,
            name=name,
            description=description
        )
        self.agent_card.skills.append(skill)
    
    async def handle_a2a_request(
        self, task: A2ATask, message: A2AMessage
    ) -> A2AArtifact:
        """处理来自其他 Agent 的 A2A 请求
        
        在内部使用 MCP 工具完成任务
        """
        # 提取用户消息
        user_text = ""
        for part in message.parts:
            if part.type == "text" and part.text:
                user_text = part.text
        
        # 使用本地 MCP 工具处理
        result = await self.mcp_server.process_message(json.dumps({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "process_request",
                "arguments": {"query": user_text}
            },
            "id": "1"
        }))
        
        # 构建产出物
        response_data = json.loads(result)
        result_text = response_data.get("result", {}).get(
            "content", [{}]
        )[0].get("text", "")
        
        artifact = A2AArtifact(
            name="response",
            parts=[MessagePart(type="text", text=result_text)],
            description="处理结果"
        )
        
        return artifact
```

## 29.6 企业级部署策略

### 29.6.1 安全考量

```python
# MCP Server 安全层
class MCPSecurityLayer:
    """MCP 安全层 - 确保工具调用安全"""
    
    def __init__(self):
        self._allowed_tools: set[str] = set()
        self._blocked_tools: set[str] = set()
        self._rate_limits: dict[str, int] = {}
        self._call_counts: dict[str, int] = {}
    
    def allow_tool(self, tool_name: str) -> None:
        """白名单工具"""
        self._allowed_tools.add(tool_name)
        self._blocked_tools.discard(tool_name)
    
    def block_tool(self, tool_name: str) -> None:
        """黑名单工具"""
        self._blocked_tools.add(tool_name)
        self._allowed_tools.discard(tool_name)
    
    def set_rate_limit(
        self, tool_name: str, max_calls: int
    ) -> None:
        """设置工具调用频率限制"""
        self._rate_limits[tool_name] = max_calls
    
    def validate_call(self, tool_name: str) -> tuple[bool, str]:
        """验证工具调用是否允许"""
        # 检查黑名单
        if tool_name in self._blocked_tools:
            return False, f"工具 '{tool_name}' 已被禁用"
        
        # 检查白名单（如果设置）
        if self._allowed_tools and tool_name not in self._allowed_tools:
            return False, f"工具 '{tool_name}' 不在允许列表中"
        
        # 检查频率限制
        if tool_name in self._rate_limits:
            count = self._call_counts.get(tool_name, 0)
            limit = self._rate_limits[tool_name]
            if count >= limit:
                return False, f"工具 '{tool_name}' 调用频率超限 ({count}/{limit})"
            self._call_counts[tool_name] = count + 1
        
        return True, "验证通过"
    
    def reset_counts(self) -> None:
        """重置调用计数"""
        self._call_counts.clear()


# 使用示例
security = MCPSecurityLayer()

# 配置安全策略
security.allow_tool("read_file")
security.allow_tool("search_web")
security.block_tool("delete_file")
security.set_rate_limit("search_web", max_calls=100)

# 验证调用
allowed, msg = security.validate_call("read_file")
print(f"read_file: {allowed} - {msg}")  # True

allowed, msg = security.validate_call("delete_file")
print(f"delete_file: {allowed} - {msg}")  # False
```

### 29.6.2 监控与可观测性

```python
# MCP/A2A 可观测性集成
from dataclasses import dataclass, field
from datetime import datetime
import time


@dataclass
class SpanContext:
    """分布式追踪 Span 上下文"""
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    operation: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    attributes: dict[str, str] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    
    def finish(self) -> None:
        """结束 Span"""
        self.end_time = time.time()
    
    @property
    def duration_ms(self) -> float:
        """计算持续时间"""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000


class MCPTracer:
    """MCP 分布式追踪器"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self._spans: list[SpanContext] = []
    
    def start_span(
        self,
        operation: str,
        parent: SpanContext | None = None,
        attributes: dict[str, str] = None
    ) -> SpanContext:
        """开始新的追踪 Span"""
        import uuid
        
        span = SpanContext(
            trace_id=parent.trace_id if parent else str(uuid.uuid4()),
            span_id=str(uuid.uuid4())[:8],
            parent_span_id=parent.span_id if parent else None,
            operation=operation,
            attributes=attributes or {}
        )
        self._spans.append(span)
        return span
    
    def get_trace_tree(self) -> list[dict]:
        """获取追踪树结构"""
        return [
            {
                "trace_id": s.trace_id,
                "span_id": s.span_id,
                "parent": s.parent_span_id,
                "operation": s.operation,
                "duration_ms": s.duration_ms,
                "attributes": s.attributes
            }
            for s in self._spans
        ]
    
    def export_otlp(self) -> list[dict]:
        """导出为 OTLP 格式"""
        return self.get_trace_tree()


class MCPMetricsCollector:
    """MCP 指标收集器"""
    
    def __init__(self):
        self._counters: dict[str, int] = {}
        self._histograms: dict[str, list[float]] = {}
        self._gauges: dict[str, float] = {}
    
    def increment(self, name: str, value: int = 1) -> None:
        """增加计数器"""
        self._counters[name] = self._counters.get(name, 0) + value
    
    def observe(self, name: str, value: float) -> None:
        """记录直方图观测值"""
        if name not in self._histograms:
            self._histograms[name] = []
        self._histograms[name].append(value)
    
    def gauge(self, name: str, value: float) -> None:
        """设置仪表盘值"""
        self._gauges[name] = value
    
    def get_summary(self) -> dict:
        """获取指标摘要"""
        summary = {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {}
        }
        
        for name, values in self._histograms.items():
            if values:
                sorted_vals = sorted(values)
                n = len(sorted_vals)
                summary["histograms"][name] = {
                    "count": n,
                    "min": sorted_vals[0],
                    "max": sorted_vals[-1],
                    "mean": sum(sorted_vals) / n,
                    "p50": sorted_vals[n // 2],
                    "p95": sorted_vals[int(n * 0.95)],
                    "p99": sorted_vals[int(n * 0.99)]
                }
        
        return summary
```

## 29.7 实战案例

### 29.7.1 构建完整 MCP Server

```python
# 完整的文件系统 MCP Server 示例
import os
from pathlib import Path


def create_filesystem_server(
    root_path: str = "."
) -> MCPServer:
    """创建文件系统 MCP Server"""
    
    server = MCPServer(
        name="filesystem-server",
        version="1.0.0"
    )
    security = MCPSecurityLayer()
    root = Path(root_path).resolve()
    
    @server.tool(description="列出目录中的文件")
    async def list_directory(path: str = ".") -> str:
        """列出指定目录的文件和子目录"""
        target = (root / path).resolve()
        
        # 安全检查：防止目录遍历
        if not str(target).startswith(str(root)):
            return "错误：访问被拒绝（路径越界）"
        
        if not target.exists():
            return f"错误：目录 '{path}' 不存在"
        
        if not target.is_dir():
            return f"错误：'{path}' 不是目录"
        
        entries = []
        for entry in sorted(target.iterdir()):
            prefix = "📁 " if entry.is_dir() else "📄 "
            size = entry.stat().st_size if entry.is_file() else 0
            entries.append(f"{prefix}{entry.name} ({size} bytes)")
        
        return "\n".join(entries) if entries else "目录为空"
    
    @server.tool(description="读取文件内容")
    async def read_file(path: str, encoding: str = "utf-8") -> str:
        """读取文件内容"""
        target = (root / path).resolve()
        
        if not str(target).startswith(str(root)):
            return "错误：访问被拒绝"
        
        if not target.exists():
            return f"错误：文件 '{path}' 不存在"
        
        if not target.is_file():
            return f"错误：'{path}' 不是文件"
        
        try:
            content = target.read_text(encoding=encoding)
            # 限制返回大小
            if len(content) > 10000:
                content = content[:10000] + "\n... (截断)"
            return content
        except UnicodeDecodeError:
            return f"错误：无法以 {encoding} 编码读取文件"
    
    @server.tool(description="搜索文件内容")
    async def search_files(
        pattern: str, query: str, max_results: int = 50
    ) -> str:
        """在文件中搜索文本"""
        import re
        results = []
        regex = re.compile(query, re.IGNORECASE)
        
        for file_path in root.rglob(pattern):
            if not file_path.is_file():
                continue
            
            try:
                content = file_path.read_text(encoding="utf-8")
                for i, line in enumerate(content.split("\n"), 1):
                    if regex.search(line):
                        rel = file_path.relative_to(root)
                        results.append(
                            f"{rel}:{i}: {line.strip()}"
                        )
                        if len(results) >= max_results:
                            break
            except (UnicodeDecodeError, PermissionError):
                continue
            
            if len(results) >= max_results:
                break
        
        if not results:
            return f"未找到匹配 '{query}' 的内容"
        
        return "\n".join(results)
    
    @server.resource("file://stats", name="目录统计")
    def get_stats() -> str:
        """获取目录统计信息"""
        total_files = 0
        total_dirs = 0
        total_size = 0
        extensions = {}
        
        for entry in root.rglob("*"):
            if entry.is_file():
                total_files += 1
                total_size += entry.stat().st_size
                ext = entry.suffix.lower()
                extensions[ext] = extensions.get(ext, 0) + 1
            elif entry.is_dir():
                total_dirs += 1
        
        stats_lines = [
            f"目录: {root}",
            f"文件数: {total_files}",
            f"子目录数: {total_dirs}",
            f"总大小: {total_size / 1024:.1f} KB",
            "",
            "文件类型分布:"
        ]
        
        for ext, count in sorted(
            extensions.items(), key=lambda x: -x[1]
        )[:10]:
            stats_lines.append(f"  {ext or '(无后缀)'}: {count}")
        
        return "\n".join(stats_lines)
    
    return server
```

### 29.7.2 构建 A2A Agent 协作系统

```python
# 多 Agent 协作系统示例
class ResearchAgentSystem:
    """研究型 Agent 协作系统"""
    
    def __init__(self):
        # 创建各个 Agent
        self.search_agent = self._create_search_agent()
        self.analysis_agent = self._create_analysis_agent()
        self.writing_agent = self._create_writing_agent()
    
    def _create_search_agent(self) -> A2AServer:
        """创建搜索 Agent"""
        card = AgentCard(
            name="SearchAgent",
            description="负责网络搜索和信息收集",
            url="http://localhost:8001",
            skills=[
                AgentSkill(
                    id="web_search",
                    name="网络搜索",
                    description="搜索互联网获取最新信息",
                    tags=["search", "web"]
                ),
                AgentSkill(
                    id="academic_search",
                    name="学术搜索",
                    description="搜索学术论文和研究资料",
                    tags=["search", "academic"]
                )
            ]
        )
        
        server = A2AServer(agent_card=card)
        
        # 注册任务处理逻辑
        async def process_search(task: A2ATask):
            for msg in task.messages:
                for part in msg.parts:
                    if part.type == "text":
                        # 执行搜索逻辑
                        results = await self._perform_search(part.text)
                        
                        artifact = A2AArtifact(
                            name="search_results",
                            parts=[MessagePart(type="text", text=results)],
                            description="搜索结果"
                        )
                        task.add_artifact(artifact)
            
            task.transition(TaskState.COMPLETED, "搜索完成")
        
        server.register_handler("tasks/process", process_search)
        return server
    
    def _create_analysis_agent(self) -> A2AServer:
        """创建分析 Agent"""
        card = AgentCard(
            name="AnalysisAgent",
            description="负责数据分析和洞察提取",
            url="http://localhost:8002",
            skills=[
                AgentSkill(
                    id="data_analysis",
                    name="数据分析",
                    description="分析数据并提取关键洞察",
                    tags=["analysis", "data"]
                )
            ]
        )
        
        server = A2AServer(agent_card=card)
        
        async def process_analysis(task: A2ATask):
            # 分析逻辑
            task.transition(TaskState.COMPLETED, "分析完成")
        
        server.register_handler("tasks/process", process_analysis)
        return server
    
    def _create_writing_agent(self) -> A2AServer:
        """创建写作 Agent"""
        card = AgentCard(
            name="WritingAgent",
            description="负责撰写研究报告和文档",
            url="http://localhost:8003",
            skills=[
                AgentSkill(
                    id="report_writing",
                    name="报告撰写",
                    description="基于分析结果撰写研究报告",
                    tags=["writing", "report"]
                )
            ]
        )
        
        server = A2AServer(agent_card=card)
        
        async def process_writing(task: A2ATask):
            # 写作逻辑
            task.transition(TaskState.COMPLETED, "撰写完成")
        
        server.register_handler("tasks/process", process_writing)
        return server
    
    async def _perform_search(self, query: str) -> str:
        """执行搜索（模拟）"""
        return f"搜索结果: 关于 '{query}' 的相关信息..."
    
    async def execute_research(self, topic: str) -> str:
        """执行完整的研究流程"""
        print(f"开始研究课题: {topic}")
        
        # 步骤1：搜索
        print("1. 搜索信息...")
        search_results = await self._perform_search(topic)
        
        # 步骤2：分析（使用 A2A）
        print("2. 分析数据...")
        
        # 步骤3：撰写（使用 A2A）
        print("3. 撰写报告...")
        
        return f"关于 '{topic}' 的研究报告已完成"
```

## 29.8 本章小结

本章深入探讨了两大核心 Agent 协议：MCP 与 A2A。

**关键要点**：

1. **MCP 协议**：专注于 LLM 应用与外部工具/数据源的标准化连接，采用三层架构（Application → Client → Server），支持 stdio、SSE、Streamable HTTP 等传输方式。

2. **A2A 协议**：专注于不同 Agent 系统之间的互操作，通过 Agent Card 发现能力、Task 管理协作生命周期、Message 传递交互内容。

3. **协议互补性**：MCP 解决"Agent 如何使用工具"，A2A 解决"Agent 如何协作"。在实际应用中，两者往往结合使用。

4. **企业部署**：需要考虑安全策略（白名单、频率限制）、可观测性（分布式追踪、指标收集）和合规要求。

## 29.9 思考题

1. 在一个同时需要访问外部 API 和与其他 Agent 协作的场景中，如何设计 MCP 与 A2A 的协作架构？

2. MCP 的 Streamable HTTP 传输相比 SSE 有哪些优势？在什么场景下应该选择哪种传输方式？

3. A2A 的 Task 状态机设计中，`INPUT_REQUIRED` 状态的作用是什么？请举例说明需要人工输入的 Agent 协作场景。

4. 如何为 MCP Server 实现细粒度的访问控制？请设计一个支持多租户的 MCP Server 安全方案。

5. 比较 MCP 的工具注册机制与 OpenAI Function Calling 的工具定义方式，分析各自的优缺点。

6. 在大规模多 Agent 系统中，如何解决 Agent 发现和负载均衡问题？A2A 协议的 Agent Card 机制是否足够？

7. 设计一个基于 MCP + A2A 的代码审查系统，其中包含代码分析 Agent、安全检查 Agent 和报告生成 Agent，描述它们之间的交互流程。

8. 讨论 MCP 协议在边缘计算场景下的应用可能性，如何在资源受限的环境中实现 MCP Server？

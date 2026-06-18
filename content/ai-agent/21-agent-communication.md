---
title: "第21章：Agent 通信协议与消息传递"
description: "深入解析 Agent 间通信的核心协议，包括 MCP、A2A、消息格式标准化与事件总线架构"
updated: "2026-06-15"
---

# 第21章：Agent 通信协议与消息传递

 > **学习目标**：
 > - 掌握 Agent 通信中消息格式标准化的设计原则与实现方式
 > - 深入理解 MCP（Model Context Protocol）的架构与实现细节
 > - 掌握 A2A（Agent-to-Agent）协议的工作原理与应用场景
 > - 理解 EventBus 事件总线在 Agent 协作中的核心作用
 > - 能够对比分析 MCP 与 A2A 的适用场景与技术取舍
 > - 掌握生产级 Agent 通信系统的设计与部署

 下面的交互式演示展示了 Agent 通信协议的核心流程：

 <div data-component="AgentProtocolDemo"></div>

 ## 21.1 Agent 通信基础概念

### 21.1.1 为什么需要通信协议

在多 Agent 系统中，每个 Agent 作为独立的智能实体，需要与其他 Agent、工具、外部服务进行高效协作。通信协议是 Agent 间协作的基础基础设施，决定了信息如何在系统中流动。

Agent 通信面临的核心挑战包括：

| 挑战 | 说明 | 影响 |
|------|------|------|
| 异构性 | 不同 Agent 可能基于不同 LLM 或框架 | 消息格式不兼容 |
| 可靠性 | 网络不稳定或 Agent 故障 | 消息丢失或重复 |
| 安全性 | Agent 可能来自不同信任域 | 数据泄露或恶意注入 |
| 可扩展性 | 系统规模增长时通信效率下降 | 性能瓶颈 |
| 语义一致性 | 不同 Agent 对同一概念理解不同 | 协作失败 |

> **核心思想**：通信协议的本质是建立 Agent 间的"共同语言"，使得异构系统能够无缝协作。正如 TCP/IP 统一了网络通信，Agent 通信协议也在走向标准化。

### 21.1.2 通信模式分类

Agent 间的通信模式可以分为以下几类：

**1. 同步通信（Synchronous）**

Agent A 发送请求后等待 Agent B 的响应。适用于需要即时反馈的场景。

```python
import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime
import uuid

@dataclass
class SyncMessage:
    """同步通信消息格式"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    receiver: str = ""
    content: Any = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    correlation_id: Optional[str] = None
    timeout: float = 30.0  # 超时时间（秒）

class SyncAgent:
    """支持同步通信的 Agent"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.pending_requests: dict[str, asyncio.Future] = {}
    
    async def send_request(self, target_agent: 'SyncAgent', 
                           content: Any, timeout: float = 30.0) -> Any:
        """发送同步请求并等待响应"""
        message = SyncMessage(
            sender=self.agent_id,
            receiver=target_agent.agent_id,
            content=content,
            timeout=timeout
        )
        
        # 创建 Future 对象用于等待响应
        future = asyncio.get_event_loop().create_future()
        self.pending_requests[message.id] = future
        
        try:
            # 发送消息到目标 Agent
            await target_agent.receive_message(message)
            
            # 等待响应，带超时
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
            
        except asyncio.TimeoutError:
            print(f"请求 {message.id} 超时 ({timeout}s)")
            return None
        finally:
            self.pending_requests.pop(message.id, None)
    
    async def receive_message(self, message: SyncMessage):
        """接收消息并处理"""
        # 模拟处理逻辑
        result = await self._process_message(message)
        
        # 发送响应
        response = SyncMessage(
            sender=self.agent_id,
            receiver=message.sender,
            content=result,
            correlation_id=message.id
        )
        await self._send_response(response)
    
    async def _process_message(self, message: SyncMessage) -> Any:
        """处理接收到的消息"""
        # 根据消息内容进行处理
        if isinstance(message.content, dict):
            task_type = message.content.get("type", "unknown")
            return {
                "status": "completed",
                "result": f"Agent {self.agent_id} 处理了 {task_type} 任务"
            }
        return {"status": "completed", "result": "处理完成"}
    
    async def _send_response(self, response: SyncMessage):
        """发送响应到请求方"""
        # 在实际系统中，这里会通过网络发送
        # 这里简化为直接调用对方的回调
        pass
```

**2. 异步通信（Asynchronous）**

Agent A 发送消息后不等待响应，继续执行后续任务。适用于非实时场景。

```python
from collections import deque
from typing import Callable
import threading
import time

class AsyncMessage:
    """异步通信消息格式"""
    
    def __init__(self, sender: str, receiver: str, content: Any,
                 priority: int = 0, ttl: float = 300.0):
        self.id = str(uuid.uuid4())
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.priority = priority
        self.ttl = ttl  # 消息存活时间
        self.created_at = time.time()
        self.delivered = False
        self.retry_count = 0
        self.max_retries = 3

class AsyncMessageQueue:
    """异步消息队列"""
    
    def __init__(self, max_size: int = 10000):
        self.queue: deque = deque(maxlen=max_size)
        self.subscribers: dict[str, list[Callable]] = {}
        self._lock = threading.Lock()
    
    def publish(self, message: AsyncMessage) -> bool:
        """发布消息到队列"""
        with self._lock:
            if len(self.queue) >= self.queue.maxlen:
                return False  # 队列已满
            self.queue.append(message)
            self._notify_subscribers(message)
            return True
    
    def subscribe(self, agent_id: str, callback: Callable):
        """订阅消息"""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(callback)
    
    def _notify_subscribers(self, message: AsyncMessage):
        """通知订阅者"""
        callbacks = self.subscribers.get(message.receiver, [])
        for callback in callbacks:
            try:
                callback(message)
            except Exception as e:
                print(f"通知订阅者失败: {e}")
    
    def get_messages(self, agent_id: str, max_count: int = 10) -> list[AsyncMessage]:
        """获取指定 Agent 的消息"""
        with self._lock:
            messages = []
            remaining = deque()
            
            while self.queue and len(messages) < max_count:
                msg = self.queue.popleft()
                # 检查消息是否过期
                if time.time() - msg.created_at > msg.ttl:
                    continue  # 丢弃过期消息
                if msg.receiver == agent_id:
                    messages.append(msg)
                else:
                    remaining.append(msg)
            
            # 将未处理的消息放回队列
            self.queue.extend(remaining)
            return messages
```

**3. 发布-订阅（Pub-Sub）**

Agent 向主题发布消息，所有订阅该主题的 Agent 都能收到。适用于广播场景。

```python
from enum import Enum
from typing import Any, Callable
import hashlib

class EventType(Enum):
    """事件类型枚举"""
    AGENT_CREATED = "agent.created"
    AGENT_DESTROYED = "agent.destroyed"
    TASK_ASSIGNED = "task.assigned"
    TASK_COMPLETED = "task.completed"
    ERROR_OCCURRED = "error.occurred"
    HEARTBEAT = "heartbeat"

class PubSubMessage:
    """发布-订阅消息"""
    
    def __init__(self, topic: str, event_type: EventType,
                 payload: Any, source_agent: str):
        self.id = str(uuid.uuid4())
        self.topic = topic
        self.event_type = event_type
        self.payload = payload
        self.source_agent = source_agent
        self.timestamp = datetime.now().isoformat()
        self.metadata = {}

class TopicBasedPubSub:
    """基于主题的发布-订阅系统"""
    
    def __init__(self):
        self.topics: dict[str, list[Callable]] = {}
        self.message_history: dict[str, list[PubSubMessage]] = {}
        self.wildcard_handlers: list[tuple[str, Callable]] = []
    
    def subscribe(self, topic: str, handler: Callable, 
                  filter_fn: Callable = None):
        """订阅主题"""
        if topic not in self.topics:
            self.topics[topic] = []
            self.message_history[topic] = []
        
        self.topics[topic].append({
            "handler": handler,
            "filter": filter_fn
        })
    
    def subscribe_wildcard(self, pattern: str, handler: Callable):
        """通配符订阅，匹配所有符合模式的主题"""
        self.wildcard_handlers.append((pattern, handler))
    
    def publish(self, message: PubSubMessage):
        """发布消息到主题"""
        topic = message.topic
        
        # 记录消息历史
        if topic not in self.message_history:
            self.message_history[topic] = []
        self.message_history[topic].append(message)
        
        # 通知直接订阅者
        if topic in self.topics:
            for subscriber in self.topics[topic]:
                handler = subscriber["handler"]
                filter_fn = subscriber["filter"]
                
                if filter_fn is None or filter_fn(message):
                    try:
                        handler(message)
                    except Exception as e:
                        print(f"处理消息失败: {e}")
        
        # 通知通配符订阅者
        for pattern, handler in self.wildcard_handlers:
            if self._match_pattern(pattern, topic):
                try:
                    handler(message)
                except Exception as e:
                    print(f"通配符处理失败: {e}")
    
    def _match_pattern(self, pattern: str, topic: str) -> bool:
        """简单通配符匹配"""
        if pattern == "*":
            return True
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return topic.startswith(prefix)
        return pattern == topic
```

**4. 请求-响应（Request-Response）**

Agent A 向 Agent B 发送请求，Agent B 处理后返回响应。适用于需要明确结果的场景。

### 21.1.3 消息传递的可靠性保障

在分布式系统中，消息传递面临网络分区、进程崩溃等挑战。需要采用以下机制保障可靠性：

| 机制 | 说明 | 实现复杂度 |
|------|------|-----------|
| 消息确认（ACK） | 接收方确认收到消息 | 低 |
| 重试机制 | 消息发送失败时重试 | 低 |
| 幂等性 | 重复消息不会产生副作用 | 中 |
| 死信队列 | 无法处理的消息转入死信队列 | 中 |
| 事务消息 | 消息发送与本地事务原子性 | 高 |

```python
import time
import random
from typing import Optional
from dataclasses import dataclass

@dataclass
class DeliveryResult:
    """消息投递结果"""
    success: bool
    message_id: str
    error: Optional[str] = None
    retry_count: int = 0

class ReliableMessageDelivery:
    """可靠消息投递系统"""
    
    def __init__(self, max_retries: int = 3, 
                 retry_delay: float = 1.0,
                 dead_letter_queue_size: int = 1000):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.dead_letter_queue: list = []
        self.max_dlq_size = dead_letter_queue_size
        self.delivery_log: dict[str, DeliveryResult] = {}
    
    async def deliver(self, message: AsyncMessage, 
                      target_agent: Any) -> DeliveryResult:
        """可靠投递消息"""
        for attempt in range(self.max_retries + 1):
            try:
                # 模拟网络延迟和失败
                if random.random() < 0.3:  # 30% 失败率
                    raise ConnectionError("网络连接失败")
                
                # 执行投递
                await target_agent.receive_message(message)
                
                result = DeliveryResult(
                    success=True,
                    message_id=message.id,
                    retry_count=attempt
                )
                self.delivery_log[message.id] = result
                return result
                
            except Exception as e:
                print(f"投递失败 (尝试 {attempt + 1}/{self.max_retries + 1}): {e}")
                
                if attempt < self.max_retries:
                    # 指数退避
                    delay = self.retry_delay * (2 ** attempt)
                    # 添加随机抖动避免惊群效应
                    delay += random.uniform(0, 0.5)
                    await asyncio.sleep(delay)
        
        # 所有重试都失败，转入死信队列
        result = DeliveryResult(
            success=False,
            message_id=message.id,
            error="超过最大重试次数",
            retry_count=self.max_retries
        )
        self._add_to_dead_letter_queue(message, result)
        self.delivery_log[message.id] = result
        return result
    
    def _add_to_dead_letter_queue(self, message: AsyncMessage, 
                                   result: DeliveryResult):
        """将消息加入死信队列"""
        if len(self.dead_letter_queue) >= self.max_dlq_size:
            # 移除最旧的消息
            self.dead_letter_queue.pop(0)
        
        self.dead_letter_queue.append({
            "message": message,
            "result": result,
            "added_at": time.time()
        })
    
    def get_delivery_stats(self) -> dict:
        """获取投递统计"""
        total = len(self.delivery_log)
        success = sum(1 for r in self.delivery_log.values() if r.success)
        failed = total - success
        avg_retries = sum(r.retry_count for r in self.delivery_log.values()) / max(total, 1)
        
        return {
            "total_messages": total,
            "success": success,
            "failed": failed,
            "success_rate": success / max(total, 1),
            "average_retries": avg_retries,
            "dead_letter_count": len(self.dead_letter_queue)
        }
```

## 21.2 消息格式标准化

### 21.2.1 统一消息格式设计

Agent 通信的核心是消息格式的标准化。一个好的消息格式应该具备自描述性、可扩展性和兼容性。

**核心消息结构**：

```json
{
  "version": "1.0",
  "id": "msg-uuid-001",
  "type": "request",
  "source": {
    "agent_id": "agent-planner-001",
    "agent_type": "planner",
    "capabilities": ["planning", "decomposition"]
  },
  "target": {
    "agent_id": "agent-coder-001",
    "agent_type": "coder",
    "capabilities": ["code_generation", "testing"]
  },
  "payload": {
    "action": "generate_code",
    "parameters": {
      "language": "python",
      "description": "实现快速排序算法",
      "constraints": ["时间复杂度 O(n log n)", "原地排序"]
    }
  },
  "metadata": {
    "correlation_id": "req-uuid-002",
    "priority": "high",
    "ttl": 300,
    "timestamp": "2026-06-15T10:30:00Z",
    "trace_id": "trace-uuid-003"
  },
  "headers": {
    "content_type": "application/json",
    "encoding": "utf-8",
    "compression": "none"
  }
}
```

```python
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum
import uuid
from datetime import datetime

class MessageType(Enum):
    """消息类型枚举"""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    ACK = "ack"

class MessagePriority(Enum):
    """消息优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class AgentIdentity:
    """Agent 身份信息"""
    agent_id: str
    agent_type: str
    capabilities: list[str] = field(default_factory=list)
    version: str = "1.0"
    metadata: dict = field(default_factory=dict)

@dataclass
class MessageMetadata:
    """消息元数据"""
    correlation_id: Optional[str] = None
    priority: MessagePriority = MessagePriority.NORMAL
    ttl: float = 300.0  # 消息存活时间（秒）
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    trace_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class AgentMessage:
    """标准化 Agent 消息格式"""
    version: str = "1.0"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.REQUEST
    source: Optional[AgentIdentity] = None
    target: Optional[AgentIdentity] = None
    payload: Any = None
    metadata: MessageMetadata = field(default_factory=MessageMetadata)
    headers: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "version": self.version,
            "id": self.id,
            "type": self.type.value,
            "source": {
                "agent_id": self.source.agent_id,
                "agent_type": self.source.agent_type,
                "capabilities": self.source.capabilities
            } if self.source else None,
            "target": {
                "agent_id": self.target.agent_id,
                "agent_type": self.target.agent_type,
                "capabilities": self.target.capabilities
            } if self.target else None,
            "payload": self.payload,
            "metadata": {
                "correlation_id": self.metadata.correlation_id,
                "priority": self.metadata.priority.value,
                "ttl": self.metadata.ttl,
                "timestamp": self.metadata.timestamp,
                "trace_id": self.metadata.trace_id
            },
            "headers": self.headers
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AgentMessage':
        """从字典格式创建消息"""
        source = None
        if data.get("source"):
            source = AgentIdentity(
                agent_id=data["source"]["agent_id"],
                agent_type=data["source"]["agent_type"],
                capabilities=data["source"].get("capabilities", [])
            )
        
        target = None
        if data.get("target"):
            target = AgentIdentity(
                agent_id=data["target"]["agent_id"],
                agent_type=data["target"]["agent_type"],
                capabilities=data["target"].get("capabilities", [])
            )
        
        metadata_data = data.get("metadata", {})
        metadata = MessageMetadata(
            correlation_id=metadata_data.get("correlation_id"),
            priority=MessagePriority(metadata_data.get("priority", 1)),
            ttl=metadata_data.get("ttl", 300.0),
            timestamp=metadata_data.get("timestamp", datetime.now().isoformat()),
            trace_id=metadata_data.get("trace_id")
        )
        
        return cls(
            version=data.get("version", "1.0"),
            id=data.get("id", str(uuid.uuid4())),
            type=MessageType(data.get("type", "request")),
            source=source,
            target=target,
            payload=data.get("payload"),
            metadata=metadata,
            headers=data.get("headers", {})
        )
    
    def is_expired(self) -> bool:
        """检查消息是否过期"""
        msg_time = datetime.fromisoformat(self.metadata.timestamp)
        elapsed = (datetime.now() - msg_time).total_seconds()
        return elapsed > self.metadata.ttl
    
    def create_response(self, payload: Any) -> 'AgentMessage':
        """创建响应消息"""
        return AgentMessage(
            version=self.version,
            type=MessageType.RESPONSE,
            source=self.target,
            target=self.source,
            payload=payload,
            metadata=MessageMetadata(
                correlation_id=self.id,
                trace_id=self.metadata.trace_id
            )
        )

class MessageSerializer:
    """消息序列化器"""
    
    @staticmethod
    def serialize(message: AgentMessage) -> str:
        """序列化消息为 JSON 字符串"""
        import json
        return json.dumps(message.to_dict(), ensure_ascii=False, indent=2)
    
    @staticmethod
    def deserialize(data: str) -> AgentMessage:
        """反序列化 JSON 字符串为消息对象"""
        import json
        return AgentMessage.from_dict(json.loads(data))
    
    @staticmethod
    def validate(message: AgentMessage) -> list[str]:
        """验证消息格式"""
        errors = []
        
        if not message.id:
            errors.append("消息 ID 不能为空")
        
        if message.source is None:
            errors.append("消息源不能为空")
        
        if message.type == MessageType.REQUEST and message.target is None:
            errors.append("请求消息必须指定目标")
        
        if message.metadata.ttl <= 0:
            errors.append("TTL 必须大于 0")
        
        return errors
```

### 21.2.2 消息版本管理

随着系统演进，消息格式需要支持版本兼容。采用语义化版本控制：

| 版本变更类型 | 版本号变化 | 示例 |
|------------|-----------|------|
| 不兼容变更 | 主版本号 +1 | v1.0.0 → v2.0.0 |
| 新增功能 | 次版本号 +1 | v1.0.0 → v1.1.0 |
| Bug 修复 | 修订号 +1 | v1.0.0 → v1.0.1 |

```python
from packaging import version

class MessageVersionManager:
    """消息版本管理器"""
    
    SUPPORTED_VERSIONS = ["1.0", "1.1", "2.0"]
    DEFAULT_VERSION = "1.0"
    
    def __init__(self):
        self.version_converters: dict[str, callable] = {}
        self._register_converters()
    
    def _register_converters(self):
        """注册版本转换器"""
        self.version_converters[("1.0", "1.1")] = self._convert_1_0_to_1_1
        self.version_converters[("1.1", "2.0")] = self._convert_1_1_to_2_0
    
    def _convert_1_0_to_1_1(self, data: dict) -> dict:
        """v1.0 → v1.1 转换"""
        # 添加 metadata.created_at 字段
        if "metadata" not in data:
            data["metadata"] = {}
        if "created_at" not in data["metadata"]:
            data["metadata"]["created_at"] = datetime.now().isoformat()
        data["version"] = "1.1"
        return data
    
    def _convert_1_1_to_2_0(self, data: dict) -> dict:
        """v1.1 → v2.0 转换（不兼容变更）"""
        # 重构 source/target 结构
        if "source_agent" in data:
            data["source"] = {
                "agent_id": data.pop("source_agent"),
                "agent_type": "unknown",
                "capabilities": []
            }
        data["version"] = "2.0"
        return data
    
    def convert(self, message_data: dict, target_version: str) -> dict:
        """将消息转换为目标版本"""
        source_version = message_data.get("version", self.DEFAULT_VERSION)
        
        if source_version == target_version:
            return message_data
        
        # 查找转换路径
        converter_key = (source_version, target_version)
        if converter_key in self.version_converters:
            return self.version_converters[converter_key](message_data)
        
        # 尝试中间版本转换
        for mid_version in self.SUPPORTED_VERSIONS:
            key1 = (source_version, mid_version)
            key2 = (mid_version, target_version)
            
            if key1 in self.version_converters and key2 in self.version_converters:
                data = self.version_converters[key1](message_data)
                return self.version_converters[key2](data)
        
        raise ValueError(f"无法从 {source_version} 转换到 {target_version}")
```

### 21.2.3 消息验证与校验

消息的正确性是系统稳定运行的基础。需要实现多层验证机制：

```python
import re
from typing import Any

class MessageValidator:
    """消息验证器"""
    
    # 正则表达式模式
    UUID_PATTERN = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    )
    TIMESTAMP_PATTERN = re.compile(
        r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z?$'
    )
    
    def __init__(self):
        self.schema_validators: dict[str, callable] = {}
        self.custom_rules: list[callable] = []
    
    def validate(self, message: AgentMessage) -> list[str]:
        """执行完整验证"""
        errors = []
        
        # 基础字段验证
        errors.extend(self._validate_basic_fields(message))
        
        # 类型特定验证
        errors.extend(self._validate_type_specific(message))
        
        # 元数据验证
        errors.extend(self._validate_metadata(message))
        
        # 自定义规则验证
        for rule in self.custom_rules:
            try:
                rule_error = rule(message)
                if rule_error:
                    errors.append(rule_error)
            except Exception as e:
                errors.append(f"自定义规则执行失败: {e}")
        
        return errors
    
    def _validate_basic_fields(self, message: AgentMessage) -> list[str]:
        """验证基础字段"""
        errors = []
        
        # 验证 ID
        if not message.id:
            errors.append("消息 ID 不能为空")
        elif not self.UUID_PATTERN.match(message.id):
            errors.append(f"消息 ID 格式无效: {message.id}")
        
        # 验证版本
        if not message.version:
            errors.append("消息版本不能为空")
        elif message.version not in MessageVersionManager.SUPPORTED_VERSIONS:
            errors.append(f"不支持的消息版本: {message.version}")
        
        # 验证类型
        if not isinstance(message.type, MessageType):
            errors.append(f"无效的消息类型: {message.type}")
        
        # 验证源 Agent
        if message.source:
            if not message.source.agent_id:
                errors.append("源 Agent ID 不能为空")
        
        return errors
    
    def _validate_type_specific(self, message: AgentMessage) -> list[str]:
        """验证特定类型的消息"""
        errors = []
        
        if message.type == MessageType.REQUEST:
            if message.target is None:
                errors.append("请求消息必须指定目标")
            elif not message.target.agent_id:
                errors.append("目标 Agent ID 不能为空")
        
        elif message.type == MessageType.RESPONSE:
            if not message.metadata.correlation_id:
                errors.append("响应消息必须包含关联 ID")
        
        elif message.type == MessageType.EVENT:
            if message.payload is None:
                errors.append("事件消息必须包含负载数据")
        
        return errors
    
    def _validate_metadata(self, message: AgentMessage) -> list[str]:
        """验证元数据"""
        errors = []
        
        # 验证时间戳格式
        if not self.TIMESTAMP_PATTERN.match(message.metadata.timestamp):
            errors.append(f"时间戳格式无效: {message.metadata.timestamp}")
        
        # 验证 TTL
        if message.metadata.ttl <= 0:
            errors.append("TTL 必须大于 0")
        elif message.metadata.ttl > 86400:  # 24小时
            errors.append("TTL 不能超过 24 小时")
        
        # 验证优先级
        if not isinstance(message.metadata.priority, MessagePriority):
            errors.append(f"无效的优先级: {message.metadata.priority}")
        
        return errors
    
    def add_custom_rule(self, rule_fn: callable):
        """添加自定义验证规则"""
        self.custom_rules.append(rule_fn)

# 使用示例
validator = MessageValidator()

# 添加自定义规则：不允许特定 Agent ID
validator.add_custom_rule(lambda msg: 
    "禁止向系统 Agent 发送消息" 
    if msg.target and msg.target.agent_id.startswith("system-") 
    else None
)
```

## 21.3 MCP 协议详解

### 21.3.1 MCP 概述

MCP（Model Context Protocol）是由 Anthropic 提出的开放协议，用于标准化 LLM 应用与外部工具、数据源之间的通信。它的核心目标是解决"每个 AI 应用都要重新实现工具集成"的问题。

**MCP 的设计哲学**：

| 原则 | 说明 |
|------|------|
| 标准化 | 统一的工具描述和调用格式 |
| 可组合 | 工具可以自由组合形成复杂工作流 |
| 安全性 | 内置权限控制和沙箱机制 |
| 可观测 | 支持追踪和日志记录 |

> **类比理解**：如果说 LLM 是"大脑"，那么 MCP 就是连接大脑与"手脚"（工具、数据源）的"神经系统"。它定义了大脑如何发现、调用和协调各种工具。

### 21.3.2 MCP 架构

MCP 采用客户端-服务器架构：

```
┌─────────────────────────────────────────────────────┐
│                    LLM Application                    │
│  ┌───────────────────────────────────────────────┐  │
│  │              MCP Client Layer                  │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐      │  │
│  │  │Tool     │  │Resource │  │Prompt   │      │  │
│  │  │Client   │  │Client   │  │Client   │      │  │
│  │  └────┬────┘  └────┬────┘  └────┬────┘      │  │
│  └───────┼─────────────┼───────────┼────────────┘  │
│          │             │           │                │
│          ▼             ▼           ▼                │
│  ┌─────────────────────────────────────────────┐    │
│  │           MCP Protocol Layer                 │    │
│  │  (JSON-RPC 2.0 / stdio / SSE / WebSocket)  │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
                         │
                         │ 标准化接口
                         ▼
┌─────────────────────────────────────────────────────┐
│                  MCP Servers                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ File     │  │ Database │  │ API      │         │
│  │ Server   │  │ Server   │  │ Server   │         │
│  └──────────┘  └──────────┘  └──────────┘         │
└─────────────────────────────────────────────────────┘
```

### 21.3.3 MCP 核心概念

**1. Tools（工具）**

工具是 MCP 最核心的概念，允许 LLM 调用外部功能：

```python
from dataclasses import dataclass, field
from typing import Any, Callable
import json

@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str = ""
    required: bool = False
    default: Any = None
    enum: list[Any] = None
    
    def to_schema(self) -> dict:
        """转换为 JSON Schema 格式"""
        schema = {
            "type": self.type,
            "description": self.description
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema

@dataclass
class MCPTool:
    """MCP 工具定义"""
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    handler: Callable = None
    annotations: dict = field(default_factory=dict)
    
    def to_definition(self) -> dict:
        """转换为 MCP 工具定义格式"""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required
            },
            "annotations": self.annotations
        }

class MCPToolRegistry:
    """MCP 工具注册表"""
    
    def __init__(self):
        self.tools: dict[str, MCPTool] = {}
    
    def register(self, tool: MCPTool):
        """注册工具"""
        self.tools[tool.name] = tool
    
    def unregister(self, tool_name: str):
        """注销工具"""
        self.tools.pop(tool_name, None)
    
    def list_tools(self) -> list[dict]:
        """列出所有可用工具"""
        return [tool.to_definition() for tool in self.tools.values()]
    
    def get_tool(self, tool_name: str) -> MCPTool:
        """获取指定工具"""
        if tool_name not in self.tools:
            raise ValueError(f"工具不存在: {tool_name}")
        return self.tools[tool_name]
    
    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """调用工具"""
        tool = self.get_tool(tool_name)
        
        if tool.handler is None:
            raise ValueError(f"工具 {tool_name} 未实现处理函数")
        
        # 验证必需参数
        for param in tool.parameters:
            if param.required and param.name not in arguments:
                raise ValueError(f"缺少必需参数: {param.name}")
        
        # 设置默认值
        for param in tool.parameters:
            if param.name not in arguments and param.default is not None:
                arguments[param.name] = param.default
        
        return await tool.handler(**arguments)

# 使用示例：注册文件读取工具
file_tool = MCPTool(
    name="read_file",
    description="读取指定路径的文件内容",
    parameters=[
        ToolParameter(name="path", type="string", 
                     description="文件路径", required=True),
        ToolParameter(name="encoding", type="string",
                     description="文件编码", default="utf-8")
    ],
    annotations={"readOnlyHint": True, "destructiveHint": False}
)

async def read_file_handler(path: str, encoding: str = "utf-8") -> str:
    """文件读取处理函数"""
    # 实际实现中会读取文件系统
    return f"文件 {path} 的内容..."

file_tool.handler = read_file_handler
```

**2. Resources（资源）**

资源是 MCP 中用于暴露数据的概念：

```python
from typing import Any
from dataclasses import dataclass
import asyncio

@dataclass
class MCPResource:
    """MCP 资源定义"""
    uri: str  # 资源 URI，如 "file:///path/to/file"
    name: str
    description: str = ""
    mime_type: str = "text/plain"
    handler: Callable = None
    
    def to_definition(self) -> dict:
        """转换为 MCP 资源定义"""
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type
        }

class MCPResourceProvider:
    """MCP 资源提供者"""
    
    def __init__(self):
        self.resources: dict[str, MCPResource] = {}
    
    def register_resource(self, resource: MCPResource):
        """注册资源"""
        self.resources[resource.uri] = resource
    
    async def read_resource(self, uri: str) -> dict:
        """读取资源内容"""
        if uri not in self.resources:
            raise ValueError(f"资源不存在: {uri}")
        
        resource = self.resources[uri]
        
        if resource.handler:
            content = await resource.handler(uri)
        else:
            content = ""
        
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": resource.mime_type,
                    "text": content
                }
            ]
        }

# 使用示例：注册数据库资源
db_resource = MCPResource(
    uri="database://users",
    name="用户数据",
    description="用户表数据",
    mime_type="application/json"
)

async def get_users(uri: str) -> str:
    """获取用户数据"""
    # 模拟数据库查询
    users = [
        {"id": 1, "name": "张三", "email": "zhangsan@example.com"},
        {"id": 2, "name": "李四", "email": "lisi@example.com"}
    ]
    return json.dumps(users, ensure_ascii=False)

db_resource.handler = get_users
```

**3. Prompts（提示模板）**

MCP 允许服务器定义可重用的提示模板：

```python
from dataclasses import dataclass, field

@dataclass
class MCPPromptArgument:
    """提示参数"""
    name: str
    description: str = ""
    required: bool = False

@dataclass
class MCPPrompt:
    """MCP 提示模板"""
    name: str
    description: str = ""
    arguments: list[MCPPromptArgument] = field(default_factory=list)
    template: str = ""
    
    def to_definition(self) -> dict:
        """转换为 MCP 提示定义"""
        return {
            "name": self.name,
            "description": self.description,
            "arguments": [
                {
                    "name": arg.name,
                    "description": arg.description,
                    "required": arg.required
                }
                for arg in self.arguments
            ]
        }

class MCPPromptRegistry:
    """MCP 提示注册表"""
    
    def __init__(self):
        self.prompts: dict[str, MCPPrompt] = {}
    
    def register_prompt(self, prompt: MCPPrompt):
        """注册提示模板"""
        self.prompts[prompt.name] = prompt
    
    async def get_prompt(self, name: str, arguments: dict) -> dict:
        """获取填充后的提示"""
        if name not in self.prompts:
            raise ValueError(f"提示不存在: {name}")
        
        prompt = self.prompts[name]
        
        # 验证必需参数
        for arg in prompt.arguments:
            if arg.required and arg.name not in arguments:
                raise ValueError(f"缺少必需参数: {arg.name}")
        
        # 填充模板
        filled_template = prompt.template
        for key, value in arguments.items():
            filled_template = filled_template.replace(f"{{{key}}}", str(value))
        
        return {
            "description": prompt.description,
            "messages": [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": filled_template
                    }
                }
            ]
        }

# 使用示例：注册代码审查提示
code_review_prompt = MCPPrompt(
    name="code_review",
    description="代码审查提示模板",
    arguments=[
        MCPPromptArgument(name="language", description="编程语言", required=True),
        MCPPromptArgument(name="code", description="待审查代码", required=True),
    ],
    template="""请审查以下 {language} 代码：

```{language}
{code}
```

请从以下方面进行审查：
1. 代码质量和可读性
2. 潜在的 bug
3. 性能优化建议
4. 安全性问题
"""
)
```

### 21.3.4 MCP 传输层

MCP 支持多种传输机制，适应不同的部署场景：

| 传输方式 | 适用场景 | 特点 |
|---------|---------|------|
| stdio | 本地进程通信 | 简单、低延迟 |
| HTTP SSE | 远程服务器 | 跨网络、可扩展 |
| WebSocket | 实时双向通信 | 低延迟、双向 |
| Streamable HTTP | 新一代远程通信 | 无状态、可扩展 |

```python
import asyncio
import json
from typing import Any

class MCPTransport:
    """MCP 传输层基类"""
    
    def __init__(self):
        self.connected = False
        self.message_handlers: dict[str, callable] = {}
    
    async def connect(self):
        """建立连接"""
        raise NotImplementedError
    
    async def disconnect(self):
        """断开连接"""
        raise NotImplementedError
    
    async def send(self, message: dict):
        """发送消息"""
        raise NotImplementedError
    
    async def receive(self) -> dict:
        """接收消息"""
        raise NotImplementedError
    
    def on_message(self, method: str, handler: callable):
        """注册消息处理器"""
        self.message_handlers[method] = handler

class StdioTransport(MCPTransport):
    """标准输入输出传输"""
    
    def __init__(self):
        super().__init__()
        self.reader = None
        self.writer = None
    
    async def connect(self):
        """连接到标准输入输出"""
        self.connected = True
    
    async def disconnect(self):
        """断开连接"""
        self.connected = False
    
    async def send(self, message: dict):
        """发送消息到标准输出"""
        data = json.dumps(message) + "\n"
        # 在实际实现中，这里会写入 stdout
        print(f"发送: {data}")
    
    async def receive(self) -> dict:
        """从标准输入接收消息"""
        # 在实际实现中，这里会读取 stdin
        # 这里简化为模拟
        return {"jsonrpc": "2.0", "method": "ping", "id": 1}

class SSETransport(MCPTransport):
    """Server-Sent Events 传输"""
    
    def __init__(self, endpoint: str):
        super().__init__()
        self.endpoint = endpoint
        self.session_id = None
        self.event_queue: asyncio.Queue = asyncio.Queue()
    
    async def connect(self):
        """建立 SSE 连接"""
        # 在实际实现中，这里会建立 HTTP 连接
        self.connected = True
        # 模拟接收事件
        asyncio.create_task(self._simulate_events())
    
    async def _simulate_events(self):
        """模拟接收 SSE 事件"""
        while self.connected:
            await asyncio.sleep(1)
            event = {
                "event": "message",
                "data": json.dumps({"type": "ping"})
            }
            await self.event_queue.put(event)
    
    async def disconnect(self):
        """断开连接"""
        self.connected = False
    
    async def send(self, message: dict):
        """发送消息"""
        # 通过 HTTP POST 发送
        headers = {
            "Content-Type": "application/json"
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        # 在实际实现中，这里会发起 HTTP 请求
        print(f"SSE 发送: {json.dumps(message)}")
    
    async def receive(self) -> dict:
        """接收 SSE 事件"""
        event = await self.event_queue.get()
        return json.loads(event.get("data", "{}"))

class WebSocketTransport(MCPTransport):
    """WebSocket 传输"""
    
    def __init__(self, url: str):
        super().__init__()
        self.url = url
        self.ws = None
        self.send_queue: asyncio.Queue = asyncio.Queue()
        self.receive_queue: asyncio.Queue = asyncio.Queue()
    
    async def connect(self):
        """建立 WebSocket 连接"""
        # 在实际实现中，这里会建立 WebSocket 连接
        self.connected = True
        # 启动消息收发循环
        asyncio.create_task(self._message_loop())
    
    async def _message_loop(self):
        """消息收发循环"""
        while self.connected:
            # 发送队列中的消息
            while not self.send_queue.empty():
                msg = await self.send_queue.get()
                # 在实际实现中，这里会通过 WebSocket 发送
                print(f"WebSocket 发送: {msg}")
            
            await asyncio.sleep(0.1)
    
    async def disconnect(self):
        """断开连接"""
        self.connected = False
    
    async def send(self, message: dict):
        """发送消息"""
        await self.send_queue.put(json.dumps(message))
    
    async def receive(self) -> dict:
        """接收消息"""
        return await self.receive_queue.get()
```

### 21.3.5 MCP 服务器实现

完整的 MCP 服务器实现：

```python
from typing import Any, Callable, Optional
import asyncio
import uuid

class MCPServer:
    """MCP 服务器实现"""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.tool_registry = MCPToolRegistry()
        self.resource_provider = MCPResourceProvider()
        self.prompt_registry = MCPPromptRegistry()
        self.transport: Optional[MCPTransport] = None
        self.initialized = False
    
    def set_transport(self, transport: MCPTransport):
        """设置传输层"""
        self.transport = transport
        self.transport.on_message("initialize", self._handle_initialize)
        self.transport.on_message("tools/list", self._handle_list_tools)
        self.transport.on_message("tools/call", self._handle_call_tool)
        self.transport.on_message("resources/list", self._handle_list_resources)
        self.transport.on_message("resources/read", self._handle_read_resource)
        self.transport.on_message("prompts/list", self._handle_list_prompts)
        self.transport.on_message("prompts/get", self._handle_get_prompt)
    
    async def start(self):
        """启动服务器"""
        if not self.transport:
            raise ValueError("未设置传输层")
        
        await self.transport.connect()
        print(f"MCP 服务器 {self.name} v{self.version} 已启动")
    
    async def stop(self):
        """停止服务器"""
        if self.transport:
            await self.transport.disconnect()
        print(f"MCP 服务器 {self.name} 已停止")
    
    async def _handle_initialize(self, params: dict) -> dict:
        """处理初始化请求"""
        self.initialized = True
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
                "resources": {},
                "prompts": {}
            },
            "serverInfo": {
                "name": self.name,
                "version": self.version
            }
        }
    
    async def _handle_list_tools(self, params: dict) -> dict:
        """处理工具列表请求"""
        return {
            "tools": self.tool_registry.list_tools()
        }
    
    async def _handle_call_tool(self, params: dict) -> dict:
        """处理工具调用请求"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        try:
            result = await self.tool_registry.call_tool(tool_name, arguments)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": str(result)
                    }
                ]
            }
        except Exception as e:
            return {
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": f"工具调用失败: {str(e)}"
                    }
                ]
            }
    
    async def _handle_list_resources(self, params: dict) -> dict:
        """处理资源列表请求"""
        return {
            "resources": [
                resource.to_definition() 
                for resource in self.resource_provider.resources.values()
            ]
        }
    
    async def _handle_read_resource(self, params: dict) -> dict:
        """处理资源读取请求"""
        uri = params.get("uri")
        return await self.resource_provider.read_resource(uri)
    
    async def _handle_list_prompts(self, params: dict) -> dict:
        """处理提示列表请求"""
        return {
            "prompts": [
                prompt.to_definition()
                for prompt in self.prompt_registry.prompts.values()
            ]
        }
    
    async def _handle_get_prompt(self, params: dict) -> dict:
        """处理提示获取请求"""
        name = params.get("name")
        arguments = params.get("arguments", {})
        return await self.prompt_registry.get_prompt(name, arguments)

# 使用示例
async def main():
    # 创建服务器
    server = MCPServer("my-mcp-server", "1.0.0")
    
    # 注册工具
    server.tool_registry.register(file_tool)
    server.tool_registry.register(MCPTool(
        name="execute_sql",
        description="执行 SQL 查询",
        parameters=[
            ToolParameter(name="query", type="string", 
                         description="SQL 查询语句", required=True),
            ToolParameter(name="database", type="string",
                         description="数据库名称", default="default")
        ],
        handler=lambda query, database="default": f"执行查询: {query} 在数据库 {database}"
    ))
    
    # 注册资源
    server.resource_provider.register_resource(db_resource)
    
    # 注册提示
    server.prompt_registry.register_prompt(code_review_prompt)
    
    # 设置传输层并启动
    transport = StdioTransport()
    server.set_transport(transport)
    await server.start()

# asyncio.run(main())
```

## 21.4 A2A 协议详解

### 21.4.1 A2A 概述

A2A（Agent-to-Agent）是由 Google 提出的开放协议，用于 Agent 之间的直接通信。与 MCP 不同，A2A 专注于 Agent 间的协作，而不是 Agent 与工具的集成。

**A2A 的核心特点**：

| 特点 | 说明 |
|------|------|
| Agent 发现 | 自动发现网络中的其他 Agent |
| 任务委派 | 将复杂任务委派给其他 Agent |
| 能力协商 | Agent 间协商各自的能力 |
| 状态同步 | 实时同步任务执行状态 |

> **关键区别**：MCP 解决的是"Agent 如何使用工具"，A2A 解决的是"Agent 如何与其他 Agent 协作"。两者是互补的。

### 21.4.2 A2A 架构

```
┌──────────────────────────────────────────────────────────┐
│                      Agent Network                       │
│                                                          │
│  ┌──────────────┐      A2A Protocol      ┌──────────────┐│
│  │  Agent A     │◄──────────────────────►│  Agent B     ││
│  │  (Planner)   │                        │  (Researcher) ││
│  └──────┬───────┘                        └──────┬───────┘│
│         │                                        │        │
│         │           A2A Protocol                 │        │
│         │                                        │        │
│  ┌──────▼───────┐                        ┌──────▼───────┐│
│  │  Agent C     │◄──────────────────────►│  Agent D     ││
│  │  (Coder)     │                        │  (Tester)    ││
│  └──────────────┘                        └──────────────┘│
└──────────────────────────────────────────────────────────┘
```

### 21.4.3 A2A 核心概念

**1. Agent Card（Agent 名片）**

每个 Agent 通过 Agent Card 描述自己的能力和身份：

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class AgentCapability:
    """Agent 能力"""
    name: str
    description: str
    input_schema: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)

@dataclass
class AgentEndpoint:
    """Agent 端点"""
    url: str
    transport: str = "https"  # https, websocket, grpc
    authentication: dict = field(default_factory=dict)

@dataclass
class AgentCard:
    """Agent 名片"""
    name: str
    description: str
    version: str = "1.0.0"
    url: str = ""
    capabilities: list[AgentCapability] = field(default_factory=list)
    endpoints: list[AgentEndpoint] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    def to_json(self) -> dict:
        """转换为 JSON 格式"""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "url": self.url,
            "capabilities": [
                {
                    "name": cap.name,
                    "description": cap.description,
                    "inputSchema": cap.input_schema,
                    "outputSchema": cap.output_schema
                }
                for cap in self.capabilities
            ],
            "endpoints": [
                {
                    "url": ep.url,
                    "transport": ep.transport,
                    "authentication": ep.authentication
                }
                for ep in self.endpoints
            ],
            "skills": self.skills,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_json(cls, data: dict) -> 'AgentCard':
        """从 JSON 创建 Agent 名片"""
        capabilities = [
            AgentCapability(
                name=cap["name"],
                description=cap.get("description", ""),
                input_schema=cap.get("inputSchema", {}),
                output_schema=cap.get("outputSchema", {})
            )
            for cap in data.get("capabilities", [])
        ]
        
        endpoints = [
            AgentEndpoint(
                url=ep["url"],
                transport=ep.get("transport", "https"),
                authentication=ep.get("authentication", {})
            )
            for ep in data.get("endpoints", [])
        ]
        
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            url=data.get("url", ""),
            capabilities=capabilities,
            endpoints=endpoints,
            skills=data.get("skills", []),
            metadata=data.get("metadata", {})
        )

# 创建示例 Agent Card
planner_card = AgentCard(
    name="PlannerAgent",
    description="任务规划 Agent，负责将复杂任务分解为子任务",
    version="1.0.0",
    capabilities=[
        AgentCapability(
            name="decompose_task",
            description="将复杂任务分解为可执行的子任务",
            input_schema={
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "任务描述"}
                },
                "required": ["task"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "subtasks": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        )
    ],
    skills=["task_planning", "task_decomposition", "priority_analysis"]
)
```

**2. Task（任务）**

A2A 中的任务是 Agent 间协作的基本单位：

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime

class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    """任务优先级"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    URGENT = 3

@dataclass
class TaskArtifact:
    """任务产物"""
    name: str
    content: Any
    mime_type: str = "text/plain"
    metadata: dict = field(default_factory=dict)

@dataclass
class A2ATask:
    """A2A 任务"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    input: Any = None
    output: Any = None
    artifacts: list[TaskArtifact] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    assigned_to: Optional[str] = None
    created_by: Optional[str] = None
    
    def to_json(self) -> dict:
        """转换为 JSON 格式"""
        return {
            "id": self.id,
            "status": self.status.value,
            "priority": self.priority.value,
            "input": self.input,
            "output": self.output,
            "artifacts": [
                {
                    "name": art.name,
                    "content": art.content,
                    "mimeType": art.mime_type,
                    "metadata": art.metadata
                }
                for art in self.artifacts
            ],
            "metadata": self.metadata,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "assignedTo": self.assigned_to,
            "createdBy": self.created_by
        }
    
    def update_status(self, new_status: TaskStatus):
        """更新任务状态"""
        self.status = new_status
        self.updated_at = datetime.now().isoformat()
    
    def add_artifact(self, artifact: TaskArtifact):
        """添加任务产物"""
        self.artifacts.append(artifact)
        self.updated_at = datetime.now().isoformat()

# 使用示例
task = A2ATask(
    input={"task": "分析用户行为数据"},
    priority=TaskPriority.HIGH,
    assigned_to="data-analyst-agent",
    created_by="planner-agent"
)
```

**3. Message（消息）**

A2A 中的消息用于 Agent 间的通信：

```python
from dataclasses import dataclass, field
from typing import Any

class A2AMessageType(Enum):
    """A2A 消息类型"""
    TASK_CREATE = "task.create"
    TASK_UPDATE = "task.update"
    TASK_CANCEL = "task.cancel"
    MESSAGE = "message"
    HEARTBEAT = "heartbeat"

@dataclass
class A2AMessage:
    """A2A 消息"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: A2AMessageType = A2AMessageType.MESSAGE
    sender: str = ""
    receiver: str = ""
    task_id: Optional[str] = None
    content: Any = None
    metadata: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_json(self) -> dict:
        """转换为 JSON 格式"""
        return {
            "id": self.id,
            "type": self.type.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "taskId": self.task_id,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
```

### 21.4.4 A2A 协议流程

典型的 A2A 协作流程：

```
┌─────────────────────────────────────────────────────────────┐
│                    A2A 协作流程                               │
│                                                             │
│  1. 发现阶段 (Discovery)                                     │
│     ┌─────────┐         ┌─────────┐                         │
│     │ Agent A │ ──────► │ Registry│ ◄────── Agent B        │
│     │         │         │         │                         │
│     └─────────┘         └─────────┘                         │
│          │                  ▲                               │
│          │    查询能力      │    注册能力                     │
│          └──────────────────┘                               │
│                                                             │
│  2. 协商阶段 (Negotiation)                                   │
│     ┌─────────┐    能力查询    ┌─────────┐                  │
│     │ Agent A │ ─────────────►│ Agent B │                  │
│     │         │◄─────────────│         │                  │
│     └─────────┘    能力响应    └─────────┘                  │
│                                                             │
│  3. 执行阶段 (Execution)                                     │
│     ┌─────────┐    任务委派    ┌─────────┐                  │
│     │ Agent A │ ─────────────►│ Agent B │                  │
│     │         │◄─────────────│         │                  │
│     └─────────┘    状态更新    └─────────┘                  │
│                                                             │
│  4. 完成阶段 (Completion)                                    │
│     ┌─────────┐    结果返回    ┌─────────┐                  │
│     │ Agent A │◄─────────────│ Agent B │                  │
│     │         │              │         │                  │
│     └─────────┘              └─────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

```python
class A2AClient:
    """A2A 客户端"""
    
    def __init__(self, agent_card: AgentCard):
        self.agent_card = agent_card
        self.discovered_agents: dict[str, AgentCard] = {}
        self.active_tasks: dict[str, A2ATask] = {}
    
    async def discover_agents(self, registry_url: str) -> list[AgentCard]:
        """发现网络中的 Agent"""
        # 在实际实现中，这里会向注册中心查询
        # 这里简化为模拟
        print(f"从 {registry_url} 发现 Agent...")
        return []
    
    async def get_agent_card(self, agent_url: str) -> AgentCard:
        """获取 Agent 的名片"""
        # 在实际实现中，这里会请求 Agent 的 /.well-known/agent.json
        # 这里简化为模拟
        return AgentCard(
            name="Unknown Agent",
            description="未知 Agent"
        )
    
    async def create_task(self, agent_url: str, 
                          task: A2ATask) -> A2ATask:
        """创建任务"""
        # 在实际实现中，这里会通过 HTTP POST 发送任务
        message = A2AMessage(
            type=A2AMessageType.TASK_CREATE,
            sender=self.agent_card.name,
            task_id=task.id,
            content=task.to_json()
        )
        
        # 发送消息
        await self._send_message(agent_url, message)
        
        # 记录任务
        self.active_tasks[task.id] = task
        
        return task
    
    async def update_task(self, agent_url: str, 
                          task_id: str, 
                          status: TaskStatus,
                          output: Any = None) -> bool:
        """更新任务状态"""
        message = A2AMessage(
            type=A2AMessageType.TASK_UPDATE,
            sender=self.agent_card.name,
            task_id=task_id,
            content={
                "status": status.value,
                "output": output
            }
        )
        
        await self._send_message(agent_url, message)
        return True
    
    async def cancel_task(self, agent_url: str, 
                          task_id: str) -> bool:
        """取消任务"""
        message = A2AMessage(
            type=A2AMessageType.TASK_CANCEL,
            sender=self.agent_card.name,
            task_id=task_id,
            content={"reason": "用户取消"}
        )
        
        await self._send_message(agent_url, message)
        return True
    
    async def _send_message(self, agent_url: str, 
                            message: A2AMessage):
        """发送消息到 Agent"""
        # 在实际实现中，这里会通过 HTTP/WebSocket 发送
        print(f"发送消息到 {agent_url}: {message.type.value}")
```

### 21.4.5 A2A 服务端实现

```python
class A2AServer:
    """A2A 服务端"""
    
    def __init__(self, agent_card: AgentCard):
        self.agent_card = agent_card
        self.task_handlers: dict[str, Callable] = {}
        self.task_store: dict[str, A2ATask] = {}
        self.transport: Optional[MCPTransport] = None
    
    def register_task_handler(self, capability_name: str, 
                              handler: Callable):
        """注册任务处理器"""
        self.task_handlers[capability_name] = handler
    
    async def handle_message(self, message: A2AMessage) -> Optional[A2AMessage]:
        """处理接收到的消息"""
        if message.type == A2AMessageType.TASK_CREATE:
            return await self._handle_task_create(message)
        elif message.type == A2AMessageType.TASK_UPDATE:
            return await self._handle_task_update(message)
        elif message.type == A2AMessageType.TASK_CANCEL:
            return await self._handle_task_cancel(message)
        elif message.type == A2AMessageType.MESSAGE:
            return await self._handle_message(message)
        elif message.type == A2AMessageType.HEARTBEAT:
            return await self._handle_heartbeat(message)
        
        return None
    
    async def _handle_task_create(self, message: A2AMessage) -> A2AMessage:
        """处理任务创建"""
        task_data = message.content
        task = A2ATask(
            id=task_data.get("id", str(uuid.uuid4())),
            input=task_data.get("input"),
            priority=TaskPriority(task_data.get("priority", 1)),
            created_by=message.sender,
            assigned_to=self.agent_card.name
        )
        
        # 存储任务
        self.task_store[task.id] = task
        
        # 异步执行任务
        asyncio.create_task(self._execute_task(task))
        
        # 返回确认
        return A2AMessage(
            type=A2AMessageType.TASK_UPDATE,
            sender=self.agent_card.name,
            receiver=message.sender,
            task_id=task.id,
            content={"status": "accepted", "task": task.to_json()}
        )
    
    async def _execute_task(self, task: A2ATask):
        """执行任务"""
        try:
            # 更新状态为进行中
            task.update_status(TaskStatus.IN_PROGRESS)
            
            # 查找合适的处理器
            handler = None
            for capability in self.agent_card.capabilities:
                if capability.name in self.task_handlers:
                    handler = self.task_handlers[capability.name]
                    break
            
            if handler:
                result = await handler(task.input)
                task.output = result
                task.update_status(TaskStatus.COMPLETED)
            else:
                task.update_status(TaskStatus.FAILED)
                task.metadata["error"] = "未找到合适的处理器"
                
        except Exception as e:
            task.update_status(TaskStatus.FAILED)
            task.metadata["error"] = str(e)
    
    async def _handle_task_update(self, message: A2AMessage) -> A2AMessage:
        """处理任务更新"""
        task_id = message.task_id
        if task_id in self.task_store:
            task = self.task_store[task_id]
            # 更新任务状态
            new_status = message.content.get("status")
            if new_status:
                task.update_status(TaskStatus(new_status))
            
            return A2AMessage(
                type=A2AMessageType.MESSAGE,
                sender=self.agent_card.name,
                receiver=message.sender,
                task_id=task_id,
                content={"status": "updated"}
            )
        return None
    
    async def _handle_task_cancel(self, message: A2AMessage) -> A2AMessage:
        """处理任务取消"""
        task_id = message.task_id
        if task_id in self.task_store:
            task = self.task_store[task_id]
            task.update_status(TaskStatus.CANCELLED)
            
            return A2AMessage(
                type=A2AMessageType.MESSAGE,
                sender=self.agent_card.name,
                receiver=message.sender,
                task_id=task_id,
                content={"status": "cancelled"}
            )
        return None
    
    async def _handle_message(self, message: A2AMessage) -> A2AMessage:
        """处理普通消息"""
        return A2AMessage(
            type=A2AMessageType.MESSAGE,
            sender=self.agent_card.name,
            receiver=message.sender,
            content={"status": "received"}
        )
    
    async def _handle_heartbeat(self, message: A2AMessage) -> A2AMessage:
        """处理心跳"""
        return A2AMessage(
            type=A2AMessageType.HEARTBEAT,
            sender=self.agent_card.name,
            receiver=message.sender,
            content={"status": "alive"}
        )

# 使用示例
async def main():
    # 创建 Agent Card
    agent_card = AgentCard(
        name="DataAnalystAgent",
        description="数据分析 Agent",
        capabilities=[
            AgentCapability(
                name="analyze_data",
                description="分析数据并生成报告",
                input_schema={
                    "type": "object",
                    "properties": {
                        "data_source": {"type": "string"},
                        "analysis_type": {"type": "string"}
                    }
                }
            )
        ]
    )
    
    # 创建服务器
    server = A2AServer(agent_card)
    
    # 注册处理器
    async def analyze_handler(input_data):
        # 实际的数据分析逻辑
        return {"report": "分析完成"}
    
    server.register_task_handler("analyze_data", analyze_handler)
    
    print("A2A 服务器已启动")

# asyncio.run(main())
```

## 21.5 EventBus 事件总线

### 21.5.1 EventBus 概念与设计

EventBus 是 Agent 通信的核心基础设施，实现了解耦、异步、可扩展的消息传递。

```
┌─────────────────────────────────────────────────────────┐
│                    EventBus 架构                         │
│                                                         │
│  ┌─────────┐                                           │
│  │ Agent A │──── publish ────┐                         │
│  └─────────┘                 │                         │
│                              ▼                         │
│                    ┌─────────────────┐                 │
│                    │    EventBus     │                 │
│                    │  ┌───────────┐  │                 │
│                    │  │ Topic A   │──┼──► Agent X     │
│                    │  │ Topic B   │──┼──► Agent Y     │
│                    │  │ Topic C   │──┼──► Agent Z     │
│                    │  └───────────┘  │                 │
│                    └─────────────────┘                 │
│                              ▲                         │
│  ┌─────────┐                 │                         │
│  │ Agent B │──── publish ────┘                         │
│  └─────────┘                                           │
└─────────────────────────────────────────────────────────┘
```

### 21.5.2 核心 EventBus 实现

```python
from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio
import time
import uuid

@dataclass
class Event:
    """事件"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic: str = ""
    event_type: str = ""
    payload: Any = None
    source: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)

class EventBus:
    """事件总线"""
    
    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size
        self.subscribers: dict[str, list[dict]] = defaultdict(list)
        self.wildcard_subscribers: list[dict] = []
        self.event_history: list[Event] = []
        self.max_history_size = 1000
        self._lock = asyncio.Lock()
        self.metrics = EventBusMetrics()
    
    async def subscribe(self, topic: str, handler: Callable,
                        filter_fn: Callable = None,
                        priority: int = 0):
        """订阅主题"""
        async with self._lock:
            subscriber = {
                "handler": handler,
                "filter": filter_fn,
                "priority": priority,
                "subscribed_at": datetime.now().isoformat()
            }
            
            if "*" in topic or "?" in topic:
                # 通配符订阅
                self.wildcard_subscribers.append(subscriber)
                self.wildcard_subscribers.sort(
                    key=lambda x: x["priority"], reverse=True
                )
            else:
                self.subscribers[topic].append(subscriber)
                self.subscribers[topic].sort(
                    key=lambda x: x["priority"], reverse=True
                )
            
            self.metrics.record_subscription(topic)
    
    async def unsubscribe(self, topic: str, handler: Callable):
        """取消订阅"""
        async with self._lock:
            if topic in self.subscribers:
                self.subscribers[topic] = [
                    sub for sub in self.subscribers[topic]
                    if sub["handler"] != handler
                ]
    
    async def publish(self, event: Event):
        """发布事件"""
        async with self._lock:
            # 记录事件历史
            self.event_history.append(event)
            if len(self.event_history) > self.max_history_size:
                self.event_history = self.event_history[-self.max_history_size:]
            
            self.metrics.record_event(event.topic)
        
        # 获取匹配的订阅者
        matching_subscribers = self._get_matching_subscribers(event)
        
        # 通知所有匹配的订阅者
        for subscriber in matching_subscribers:
            try:
                handler = subscriber["handler"]
                filter_fn = subscriber.get("filter")
                
                if filter_fn is None or filter_fn(event):
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                    
                    self.metrics.record_delivery(event.topic, success=True)
            except Exception as e:
                self.metrics.record_delivery(event.topic, success=False)
                print(f"事件处理失败: {e}")
    
    def _get_matching_subscribers(self, event: Event) -> list[dict]:
        """获取匹配事件的订阅者"""
        subscribers = []
        
        # 精确匹配
        if event.topic in self.subscribers:
            subscribers.extend(self.subscribers[event.topic])
        
        # 通配符匹配
        for sub in self.wildcard_subscribers:
            pattern = sub.get("pattern", "*")
            if self._match_pattern(pattern, event.topic):
                subscribers.append(sub)
        
        return subscribers
    
    def _match_pattern(self, pattern: str, topic: str) -> bool:
        """通配符匹配"""
        if pattern == "*":
            return True
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return topic.startswith(prefix)
        if "?" in pattern:
            # 简单的 ? 匹配单个字符
            if len(pattern) != len(topic):
                return False
            for p, t in zip(pattern, topic):
                if p != "?" and p != t:
                    return False
            return True
        return pattern == topic
    
    def get_event_history(self, topic: str = None, 
                          limit: int = 100) -> list[Event]:
        """获取事件历史"""
        if topic:
            events = [e for e in self.event_history if e.topic == topic]
        else:
            events = self.event_history
        
        return events[-limit:]

class EventBusMetrics:
    """事件总线指标"""
    
    def __init__(self):
        self.total_events = 0
        self.total_subscriptions = 0
        self.events_by_topic: dict[str, int] = defaultdict(int)
        self.deliveries_by_topic: dict[str, dict] = defaultdict(
            lambda: {"success": 0, "failed": 0}
        )
    
    def record_event(self, topic: str):
        """记录事件"""
        self.total_events += 1
        self.events_by_topic[topic] += 1
    
    def record_subscription(self, topic: str):
        """记录订阅"""
        self.total_subscriptions += 1
    
    def record_delivery(self, topic: str, success: bool):
        """记录投递"""
        if success:
            self.deliveries_by_topic[topic]["success"] += 1
        else:
            self.deliveries_by_topic[topic]["failed"] += 1
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_events": self.total_events,
            "total_subscriptions": self.total_subscriptions,
            "events_by_topic": dict(self.events_by_topic),
            "deliveries_by_topic": dict(self.deliveries_by_topic)
        }
```

### 21.5.3 事件过滤与路由

```python
class EventFilter:
    """事件过滤器"""
    
    def __init__(self):
        self.filters: list[Callable] = []
    
    def add_filter(self, filter_fn: Callable):
        """添加过滤器"""
        self.filters.append(filter_fn)
    
    def should_process(self, event: Event) -> bool:
        """检查事件是否应该被处理"""
        for filter_fn in self.filters:
            if not filter_fn(event):
                return False
        return True
    
    @staticmethod
    def type_filter(*event_types: str):
        """类型过滤器"""
        def filter_fn(event: Event) -> bool:
            return event.event_type in event_types
        return filter_fn
    
    @staticmethod
    def source_filter(*sources: str):
        """来源过滤器"""
        def filter_fn(event: Event) -> bool:
            return event.source in sources
        return filter_fn
    
    @staticmethod
    def payload_filter(**conditions):
        """负载过滤器"""
        def filter_fn(event: Event) -> bool:
            if not isinstance(event.payload, dict):
                return False
            for key, value in conditions.items():
                if event.payload.get(key) != value:
                    return False
            return True
        return filter_fn

class EventRouter:
    """事件路由器"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.routes: dict[str, dict] = {}
    
    def add_route(self, source_pattern: str, target_topic: str,
                  transform: Callable = None):
        """添加路由规则"""
        self.routes[source_pattern] = {
            "target": target_topic,
            "transform": transform
        }
    
    async def route_event(self, event: Event):
        """路由事件"""
        for pattern, route in self.routes.items():
            if self._match_pattern(pattern, event.topic):
                target_topic = route["target"]
                transform = route.get("transform")
                
                if transform:
                    event = transform(event)
                
                event.topic = target_topic
                await self.event_bus.publish(event)
                break
    
    def _match_pattern(self, pattern: str, topic: str) -> bool:
        """通配符匹配"""
        if pattern == "*":
            return True
        if pattern.endswith(".*"):
            return topic.startswith(pattern[:-2])
        return pattern == topic

# 使用示例
async def main():
    # 创建事件总线
    event_bus = EventBus()
    
    # 创建过滤器
    event_filter = EventFilter()
    event_filter.add_filter(EventFilter.type_filter("error", "warning"))
    
    # 创建路由器
    router = EventRouter(event_bus)
    router.add_route("agent.*", "system.events")
    
    # 订阅事件
    async def error_handler(event: Event):
        print(f"错误事件: {event.payload}")
    
    await event_bus.subscribe("error", error_handler)
    
    # 发布事件
    await event_bus.publish(Event(
        topic="error",
        event_type="error",
        payload={"message": "Agent 执行失败"},
        source="agent-001"
    ))

# asyncio.run(main())
```

### 21.5.4 事件持久化与重放

```python
import json
import os
from pathlib import Path

class EventPersistence:
    """事件持久化"""
    
    def __init__(self, storage_path: str = "./event_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.current_file = None
        self.max_file_size = 10 * 1024 * 1024  # 10MB
    
    async def persist(self, event: Event):
        """持久化事件"""
        file_path = self._get_file_path()
        
        # 检查文件大小
        if file_path.exists() and file_path.stat().st_size > self.max_file_size:
            # 创建新文件
            timestamp = int(time.time())
            file_path = self.storage_path / f"events_{timestamp}.jsonl"
        
        # 追加写入
        with open(file_path, "a") as f:
            event_data = {
                "id": event.id,
                "topic": event.topic,
                "event_type": event.event_type,
                "payload": event.payload,
                "source": event.source,
                "timestamp": event.timestamp,
                "metadata": event.metadata
            }
            f.write(json.dumps(event_data, ensure_ascii=False) + "\n")
    
    def _get_file_path(self) -> Path:
        """获取当前存储文件路径"""
        if self.current_file and self.current_file.exists():
            return self.current_file
        
        # 查找最新的文件
        files = sorted(self.storage_path.glob("events_*.jsonl"))
        if files:
            self.current_file = files[-1]
            return self.current_file
        
        # 创建新文件
        timestamp = int(time.time())
        self.current_file = self.storage_path / f"events_{timestamp}.jsonl"
        return self.current_file
    
    def replay(self, topic: str = None, 
               start_time: str = None,
               end_time: str = None) -> list[Event]:
        """重放事件"""
        events = []
        
        for file_path in sorted(self.storage_path.glob("events_*.jsonl")):
            with open(file_path, "r") as f:
                for line in f:
                    if line.strip():
                        event_data = json.loads(line)
                        
                        # 应用过滤条件
                        if topic and event_data.get("topic") != topic:
                            continue
                        
                        if start_time and event_data.get("timestamp") < start_time:
                            continue
                        
                        if end_time and event_data.get("timestamp") > end_time:
                            continue
                        
                        events.append(Event(
                            id=event_data["id"],
                            topic=event_data["topic"],
                            event_type=event_data["event_type"],
                            payload=event_data.get("payload"),
                            source=event_data.get("source", ""),
                            timestamp=event_data.get("timestamp", ""),
                            metadata=event_data.get("metadata", {})
                        ))
        
        return events
    
    def cleanup(self, max_age_days: int = 30):
        """清理过期文件"""
        cutoff_time = time.time() - (max_age_days * 86400)
        
        for file_path in self.storage_path.glob("events_*.jsonl"):
            if file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                print(f"已删除过期文件: {file_path}")

class PersistentEventBus(EventBus):
    """支持持久化的事件总线"""
    
    def __init__(self, persistence: EventPersistence = None):
        super().__init__()
        self.persistence = persistence
    
    async def publish(self, event: Event):
        """发布事件（带持久化）"""
        # 先持久化
        if self.persistence:
            await self.persistence.persist(event)
        
        # 再发布
        await super().publish(event)
    
    async def replay_events(self, topic: str = None,
                           start_time: str = None):
        """重放历史事件"""
        if not self.persistence:
            return
        
        events = self.persistence.replay(topic, start_time)
        for event in events:
            await super().publish(event)
```

## 21.6 MCP vs A2A 对比分析

### 21.6.1 核心差异对比

| 维度 | MCP | A2A |
|------|-----|-----|
| **定位** | Agent 与工具的通信 | Agent 与 Agent 的通信 |
| **发起者** | LLM 应用 | Agent |
| **通信对象** | 外部工具、数据源 | 其他 Agent |
| **核心概念** | Tools, Resources, Prompts | Agent Card, Task, Message |
| **交互模式** | 请求-响应 | 任务委派 |
| **能力发现** | 工具列表 | Agent 名片 |
| **状态管理** | 无状态 | 有状态（任务状态） |
| **典型场景** | 读取文件、查询数据库 | 协作完成复杂任务 |

### 21.6.2 架构对比

```
MCP 架构：
┌─────────────────────────────────────┐
│           LLM Application           │
│                 │                   │
│                 ▼                   │
│           MCP Client                │
│                 │                   │
│     ┌───────────┼───────────┐       │
│     ▼           ▼           ▼       │
│  ┌─────┐   ┌─────┐   ┌─────┐      │
│  │Tool │   │Tool │   │Tool │      │
│  │  A  │   │  B  │   │  C  │      │
│  └─────┘   └─────┘   └─────┘      │
└─────────────────────────────────────┘

A2A 架构：
┌─────────────────────────────────────┐
│           Agent Network             │
│                                     │
│  ┌─────┐     ┌─────┐     ┌─────┐  │
│  │Agent│◄───►│Agent│◄───►│Agent│  │
│  │  A  │     │  B  │     │  C  │  │
│  └─────┘     └─────┘     └─────┘  │
│                                     │
│         Agent Registry              │
└─────────────────────────────────────┘
```

### 21.6.3 适用场景

**MCP 适用场景**：

| 场景 | 说明 |
|------|------|
| 工具集成 | LLM 需要调用外部工具 |
| 数据访问 | 需要读取外部数据源 |
| 本地操作 | 文件系统、数据库操作 |
| API 调用 | 调用第三方 API |

**A2A 适用场景**：

| 场景 | 说明 |
|------|------|
| 多 Agent 协作 | 多个 Agent 协作完成复杂任务 |
| 任务委派 | 将子任务委派给专业 Agent |
| 能力组合 | 组合不同 Agent 的能力 |
| 分布式系统 | Agent 分布在不同服务中 |

### 21.6.4 结合使用

MCP 和 A2A 可以结合使用，构建更强大的 Agent 系统：

```python
class HybridAgentSystem:
    """混合 Agent 系统（MCP + A2A）"""
    
    def __init__(self):
        self.mcp_server = MCPServer("hybrid-agent", "1.0.0")
        self.a2a_server = A2AServer(AgentCard(
            name="HybridAgent",
            description="混合 Agent，支持 MCP 和 A2A"
        ))
        self.event_bus = EventBus()
    
    async def initialize(self):
        """初始化系统"""
        # 注册 MCP 工具
        self.mcp_server.tool_registry.register(MCPTool(
            name="delegate_to_agent",
            description="将任务委派给其他 Agent",
            parameters=[
                ToolParameter(
                    name="target_agent",
                    type="string",
                    description="目标 Agent 名称",
                    required=True
                ),
                ToolParameter(
                    name="task_description",
                    type="string",
                    description="任务描述",
                    required=True
                )
            ],
            handler=self._delegate_task
        ))
        
        # 注册 A2A 任务处理器
        self.a2a_server.register_task_handler(
            "process_data",
            self._process_data
        )
    
    async def _delegate_task(self, target_agent: str, 
                            task_description: str) -> str:
        """委派任务（MCP 工具）"""
        # 创建 A2A 任务
        task = A2ATask(
            input={"description": task_description},
            assigned_to=target_agent
        )
        
        # 通过 A2A 发送任务
        await self.a2a_server.handle_message(A2AMessage(
            type=A2AMessageType.TASK_CREATE,
            sender=self.mcp_server.name,
            task_id=task.id,
            content=task.to_json()
        ))
        
        return f"任务已委派给 {target_agent}，任务 ID: {task.id}"
    
    async def _process_data(self, input_data: Any) -> Any:
        """处理数据（A2A 任务处理器）"""
        # 调用 MCP 工具处理数据
        result = await self.mcp_server.tool_registry.call_tool(
            "analyze_data",
            {"data": input_data}
        )
        return result
```

### 21.6.5 技术选型建议

| 需求 | 推荐方案 | 理由 |
|------|---------|------|
| LLM 调用外部工具 | MCP | 标准化的工具描述和调用 |
| 多 Agent 协作 | A2A | 专为 Agent 间协作设计 |
| 本地文件操作 | MCP | 简单高效的本地通信 |
| 分布式 Agent 系统 | A2A + MCP | 结合两者优势 |
| 实时协作 | A2A | 支持任务状态同步 |
| 工具集成 | MCP | 丰富的工具生态 |

## 21.7 生产级 Agent 通信系统

### 21.7.1 系统架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                  生产级 Agent 通信系统                        │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    API Gateway                        │  │
│  │              (认证、限流、路由)                         │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                 │
│                           ▼                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  Message Broker                       │  │
│  │           (Kafka / RabbitMQ / Redis)                  │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                 │
│        ┌──────────────────┼──────────────────┐              │
│        ▼                  ▼                  ▼              │
│  ┌──────────┐       ┌──────────┐       ┌──────────┐       │
│  │ Agent A  │       │ Agent B  │       │ Agent C  │       │
│  │ (MCP)    │       │ (A2A)    │       │ (混合)   │       │
│  └──────────┘       └──────────┘       └──────────┘       │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Observability Layer                       │  │
│  │         (Tracing / Metrics / Logging)                  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 21.7.2 消息代理集成

```python
from typing import Any, Callable
import asyncio
import json

class MessageBroker:
    """消息代理基类"""
    
    async def connect(self):
        """连接到消息代理"""
        raise NotImplementedError
    
    async def disconnect(self):
        """断开连接"""
        raise NotImplementedError
    
    async def publish(self, topic: str, message: dict):
        """发布消息"""
        raise NotImplementedError
    
    async def subscribe(self, topic: str, handler: Callable):
        """订阅主题"""
        raise NotImplementedError

class RedisMessageBroker(MessageBroker):
    """Redis 消息代理"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis = None
        self.pubsub = None
    
    async def connect(self):
        """连接到 Redis"""
        # 在实际实现中，这里会使用 redis.asyncio
        print(f"连接到 Redis: {self.redis_url}")
        self.connected = True
    
    async def disconnect(self):
        """断开连接"""
        self.connected = False
    
    async def publish(self, topic: str, message: dict):
        """发布消息到 Redis 频道"""
        data = json.dumps(message, ensure_ascii=False)
        # 在实际实现中，这里会调用 redis.publish
        print(f"发布到 {topic}: {data[:100]}...")
    
    async def subscribe(self, topic: str, handler: Callable):
        """订阅 Redis 频道"""
        # 在实际实现中，这里会使用 redis.pubsub
        print(f"订阅频道: {topic}")

class KafkaMessageBroker(MessageBroker):
    """Kafka 消息代理"""
    
    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self.consumer = None
    
    async def connect(self):
        """连接到 Kafka"""
        # 在实际实现中，这里会使用 aiokafka
        print(f"连接到 Kafka: {self.bootstrap_servers}")
        self.connected = True
    
    async def disconnect(self):
        """断开连接"""
        self.connected = False
    
    async def publish(self, topic: str, message: dict):
        """发布消息到 Kafka Topic"""
        data = json.dumps(message, ensure_ascii=False).encode("utf-8")
        # 在实际实现中，这里会调用 producer.send
        print(f"发送到 Kafka topic {topic}: {len(data)} bytes")
    
    async def subscribe(self, topic: str, handler: Callable):
        """订阅 Kafka Topic"""
        # 在实际实现中，这里会使用 consumer
        print(f"订阅 Kafka topic: {topic}")

class AgentCommunicationLayer:
    """Agent 通信层"""
    
    def __init__(self, broker: MessageBroker):
        self.broker = broker
        self.handlers: dict[str, Callable] = {}
        self.event_bus = EventBus()
    
    async def initialize(self):
        """初始化通信层"""
        await self.broker.connect()
    
    async def shutdown(self):
        """关闭通信层"""
        await self.broker.disconnect()
    
    async def send_message(self, target: str, message: AgentMessage):
        """发送消息"""
        topic = f"agent.{target}"
        await self.broker.publish(topic, message.to_dict())
    
    async def register_handler(self, agent_id: str, 
                               handler: Callable):
        """注册消息处理器"""
        topic = f"agent.{agent_id}"
        self.handlers[agent_id] = handler
        
        async def wrapped_handler(data: dict):
            message = AgentMessage.from_dict(data)
            await handler(message)
        
        await self.broker.subscribe(topic, wrapped_handler)
    
    async def broadcast(self, event: Event):
        """广播事件"""
        await self.event_bus.publish(event)
```

### 21.7.3 安全机制

```python
import hashlib
import hmac
from typing import Optional

class MessageSecurity:
    """消息安全机制"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.allowed_agents: set = set()
        self.blocked_agents: set = set()
    
    def sign_message(self, message: AgentMessage) -> str:
        """消息签名"""
        message_dict = message.to_dict()
        message_str = json.dumps(message_dict, sort_keys=True)
        signature = hmac.new(
            self.secret_key.encode(),
            message_str.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def verify_signature(self, message: AgentMessage, 
                        signature: str) -> bool:
        """验证消息签名"""
        expected = self.sign_message(message)
        return hmac.compare_digest(expected, signature)
    
    def check_access(self, agent_id: str) -> bool:
        """检查访问权限"""
        if agent_id in self.blocked_agents:
            return False
        if self.allowed_agents and agent_id not in self.allowed_agents:
            return False
        return True
    
    def add_allowed_agent(self, agent_id: str):
        """添加允许的 Agent"""
        self.allowed_agents.add(agent_id)
    
    def block_agent(self, agent_id: str):
        """阻止 Agent"""
        self.blocked_agents.add(agent_id)

class MessageEncryption:
    """消息加密"""
    
    def __init__(self, encryption_key: str):
        self.encryption_key = encryption_key
    
    def encrypt(self, data: str) -> str:
        """加密数据"""
        # 在实际实现中，这里会使用 AES 等加密算法
        # 这里简化为 base64 编码
        import base64
        return base64.b64encode(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """解密数据"""
        import base64
        return base64.b64decode(encrypted_data.encode()).decode()
```

### 21.7.4 监控与追踪

```python
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class TraceSpan:
    """追踪 Span"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "pending"
    attributes: dict = None
    
    def finish(self, status: str = "ok"):
        """完成 Span"""
        self.end_time = time.time()
        self.status = status
    
    @property
    def duration_ms(self) -> float:
        """获取持续时间（毫秒）"""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0

class AgentTracer:
    """Agent 追踪器"""
    
    def __init__(self):
        self.spans: dict[str, TraceSpan] = {}
        self.traces: dict[str, list[TraceSpan]] = {}
    
    def start_trace(self, operation: str) -> TraceSpan:
        """开始追踪"""
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            operation=operation,
            start_time=time.time()
        )
        
        self.spans[span_id] = span
        self.traces[trace_id] = [span]
        
        return span
    
    def start_span(self, trace_id: str, parent_span_id: str,
                   operation: str) -> TraceSpan:
        """开始子 Span"""
        span_id = str(uuid.uuid4())
        
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation=operation,
            start_time=time.time()
        )
        
        self.spans[span_id] = span
        
        if trace_id in self.traces:
            self.traces[trace_id].append(span)
        
        return span
    
    def get_trace(self, trace_id: str) -> list[TraceSpan]:
        """获取追踪的所有 Span"""
        return self.traces.get(trace_id, [])
    
    def get_trace_duration(self, trace_id: str) -> float:
        """获取追踪总时长（毫秒）"""
        spans = self.get_trace(trace_id)
        if not spans:
            return 0
        
        start = min(s.start_time for s in spans)
        end = max(s.end_time for s in spans if s.end_time)
        
        return (end - start) * 1000

# 使用示例
tracer = AgentTracer()

# 开始追踪
root_span = tracer.start_trace("用户请求处理")

# 子 Span
child_span = tracer.start_span(
    root_span.trace_id,
    root_span.span_id,
    "Agent 协作"
)

# 完成追踪
child_span.finish("ok")
root_span.finish("ok")

# 获取追踪信息
duration = tracer.get_trace_duration(root_span.trace_id)
print(f"追踪时长: {duration:.2f}ms")
```

## 21.8 本章小结

本章深入探讨了 Agent 通信协议与消息传递的核心概念和实现：

1. **通信基础**：Agent 通信面临异构性、可靠性、安全性等挑战，需要标准化的协议支持
2. **消息格式标准化**：统一的消息格式设计、版本管理和验证机制是系统稳定运行的基础
3. **MCP 协议**：Anthropic 提出的开放协议，标准化了 LLM 应用与工具的通信，支持 Tools、Resources、Prompts 三大概念
4. **A2A 协议**：Google 提出的开放协议，专注于 Agent 间的协作，支持 Agent 发现、任务委派、能力协商
5. **EventBus**：事件总线实现了 Agent 间的解耦通信，支持发布-订阅、事件过滤、持久化重放
6. **MCP vs A2A**：两者是互补的协议，MCP 解决"Agent 如何使用工具"，A2A 解决"Agent 如何与其他 Agent 协作"
7. **生产级系统**：需要考虑消息代理集成、安全机制、监控追踪等生产级需求

## 21.9 思考题

1. 在一个由 10 个 Agent 组成的系统中，如何设计通信架构以保证可扩展性和可靠性？
2. MCP 的工具注册机制如何支持动态工具发现？请设计一个支持热更新的工具注册系统。
3. A2A 协议中的任务委派如何处理 Agent 故障？请设计一个容错机制。
4. EventBus 如何保证消息的顺序性？在分布式环境下如何实现？
5. 如何设计一个支持百万级 Agent 通信的系统架构？需要考虑哪些关键瓶颈？
6. MCP 和 A2A 在安全性方面分别面临哪些挑战？如何设计统一的安全框架？
7. 如何监控和调试一个复杂的多 Agent 通信系统？需要收集哪些关键指标？

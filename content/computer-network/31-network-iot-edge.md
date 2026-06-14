# Chapter 31: 物联网与边缘计算网络

> **学习目标**：
> - 理解物联网（IoT）的三层/四层体系架构，掌握感知层、网络层、平台层和应用层的功能与数据流
> - 掌握 MQTT、CoAP、AMQP 三大物联网通信协议的报文格式、QoS 机制与适用场景
> - 深入理解 LoRa/LoRaWAN、Sigfox、NB-IoT、LTE-M 等 LPWAN 技术的调制原理与覆盖能力
> - 掌握 6LoWPAN 报头压缩与分片重组机制，理解受限网络的协议适配策略
> - 理解 MEC 与雾计算的边缘计算架构，掌握资源编排与任务调度算法
> - 掌握 5G 网络切片的 eMBB/URLLC/mMTC 三大场景与端到端资源隔离机制
> - 了解物联网安全中的轻量级加密算法、设备认证与安全启动流程

---

## 31.1 物联网概述与体系架构

### 31.1.1 物联网的定义与演进

**物联网**（Internet of Things, IoT）是指通过各种信息传感器、射频识别技术、全球定位系统等装置与技术，实时采集任何需要监控、连接、互动的物体或过程的信息，实现物与物、物与人的泛在连接。国际电信联盟（ITU）在 ITU-T Y.2060 中将其定义为：

> 物联网是信息社会的一个全球基础设施，它通过现有和不断发展的可互操作的信息和通信技术，实现物理世界和虚拟世界的连接。

物联网的演进经历了三个阶段：

| 阶段 | 时间 | 特征 | 代表技术 |
|------|------|------|----------|
| **IoT 1.0** | 2008-2014 | 设备联网，远程监控 | RFID, ZigBee, 传感器网络 |
| **IoT 2.0** | 2014-2020 | 云平台接入，数据分析 | MQTT, AWS IoT, Azure IoT |
| **IoT 3.0** | 2020-至今 | 边缘智能，自主决策 | Edge AI, 数字孪生, 5G |

### 31.1.2 三层与四层架构

物联网体系架构通常分为**三层架构**或**四层架构**：

```
物联网四层架构：
┌─────────────────────────────────────────────────────────────┐
│  应用层 (Application Layer)                                  │
│  智能家居 · 工业物联网 · 智慧城市 · 智慧医疗                    │
├─────────────────────────────────────────────────────────────┤
│  平台层 (Platform Layer)                                     │
│  设备管理 · 数据存储 · 规则引擎 · API服务 · 数据分析            │
├─────────────────────────────────────────────────────────────┤
│  网络层 (Network Layer)                                      │
│  MQTT · CoAP · AMQP · LoRa · NB-IoT · 5G · WiFi            │
├─────────────────────────────────────────────────────────────┤
│  感知层 (Perception Layer)                                   │
│  传感器 · 执行器 · RFID · 摄像头 · GPS                        │
└─────────────────────────────────────────────────────────────┘
```

各层核心功能：

| 层级 | 核心功能 | 关键技术 | 数据方向 |
|------|----------|----------|----------|
| **感知层** | 数据采集与执行 | ADC/DAC、MEMS 传感器 | 物理世界 → 数字世界 |
| **网络层** | 数据传输与路由 | MQTT、CoAP、LoRa、NB-IoT | 上行：采集 → 平台 |
| **平台层** | 数据汇聚与处理 | 时序数据库、规则引擎、ML | 存储、分析、决策 |
| **应用层** | 业务逻辑与展示 | Dashboard、API、数字孪生 | 平台 → 用户/设备 |

<div data-component="IoTArchExplorer"></div>

### 31.1.3 物联网数据流模型

物联网系统的典型数据流遵循**感知 → 传输 → 处理 → 反馈**的闭环模型：

```
数据流全景：
                    ┌──────────────────┐
                    │    云端平台       │
                    │ ┌──────────────┐ │
                    │ │  数据分析引擎 │ │
                    │ │  ML/AI 模型  │ │
                    │ └──────┬───────┘ │
                    │        │ 决策    │
                    └────────┼─────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
    ┌─────▼─────┐     ┌─────▼─────┐     ┌─────▼─────┐
    │  边缘网关  │     │  边缘网关  │     │  边缘网关  │
    │  (本地推理) │     │  (协议转换) │     │  (数据聚合) │
    └─────┬─────┘     └─────┬─────┘     └─────┬─────┘
          │                  │                  │
    ┌─────┴─────┐     ┌─────┴─────┐     ┌─────┴─────┐
    │传感器/执行器│     │传感器/执行器│     │传感器/执行器│
    └───────────┘     └───────────┘     └───────────┘
```

上行数据流（Upstream）的典型处理链：

$$\text{Sensor} \xrightarrow{\text{ADC}} \text{Raw Data} \xrightarrow{\text{Filter}} \text{Clean Data} \xrightarrow{\text{Compress}} \text{Packet} \xrightarrow{\text{MQTT/CoAP}} \text{Broker} \xrightarrow{\text{Rule Engine}} \text{Storage}$$

### 31.1.4 与传统互联网的核心区别

物联网设备与传统互联网终端（PC、智能手机）存在本质差异：

| 约束维度 | 传统互联网终端 | 物联网设备 | 影响 |
|----------|---------------|-----------|------|
| **功耗** | 持续供电（>5W） | 电池供电（μW~mW） | 需要低功耗协议和休眠机制 |
| **计算** | 多核 GHz CPU | 微控制器（MHz级） | 无法运行复杂协议栈 |
| **存储** | GB~TB 级 | KB~MB 级 | 需要报头压缩和精简协议 |
| **带宽** | Mbps~Gbps | Kbps~Mbps | 数据压缩和聚合至关重要 |
| **连接** | 有线/WiFi/5G | LoRa/NB-IoT/ZigBee | 需要专用低功耗广域网 |
| **安全** | TLS/SSL 标准化 | 资源受限难以加密 | 需要轻量级加密算法 |
| **数量** | 单用户少量设备 | 海量设备（百万级） | 需要可扩展的管理平台 |

```python
# 物联网设备资源约束示例
class IoTDeviceProfile:
    """典型物联网设备资源画像"""
    def __init__(self, device_type):
        profiles = {
            "temperature_sensor": {
                "cpu_mhz": 16,
                "ram_kb": 2,
                "flash_kb": 32,
                "battery_mah": 1000,
                "tx_power_dbm": 14,
                "duty_cycle_pct": 1,    # 占空比仅 1%
                "protocol": "LoRa",
                "sleep_current_ua": 1.5,
                "tx_current_ma": 40,
            },
            "smart_camera": {
                "cpu_mhz": 600,
                "ram_mb": 256,
                "flash_mb": 512,
                "battery_mah": None,     # 持续供电
                "tx_power_dbm": 20,
                "duty_cycle_pct": 100,
                "protocol": "WiFi",
                "sleep_current_ua": None,
                "tx_current_ma": 200,
            },
            "wearable_health": {
                "cpu_mhz": 64,
                "ram_kb": 64,
                "flash_kb": 512,
                "battery_mah": 300,
                "tx_power_dbm": 10,
                "duty_cycle_pct": 5,
                "protocol": "BLE",
                "sleep_current_ua": 3,
                "tx_current_ma": 15,
            }
        }
        self.profile = profiles.get(device_type, {})
    
    def estimate_battery_life(self, report_interval_sec=3600):
        """估算电池寿命（仅温度传感器示例）"""
        if self.profile.get("battery_mah") is None:
            return float("inf")  # 持续供电
        
        battery_ua = self.profile["battery_mah"] * 1000  # 转换为 μAh
        sleep_ua = self.profile["sleep_current_ua"]
        tx_ma = self.profile["tx_current_ma"]
        tx_duration_ms = 100  # 假设发射持续 100ms
        
        # 每个周期的平均电流
        tx_time_s = tx_duration_ms / 1000
        sleep_time_s = report_interval_sec - tx_time_s
        
        avg_current_ua = (
            (tx_ma * 1000 * tx_time_s + sleep_ua * sleep_time_s)
            / report_interval_sec
        )
        
        life_hours = battery_ua / avg_current_ua
        return life_hours / 24 / 365  # 转换为年


# 计算不同设备的电池寿命
for device_type in ["temperature_sensor", "wearable_health"]:
    profile = IoTDeviceProfile(device_type)
    life_years = profile.estimate_battery_life()
    print(f"{device_type}: 电池寿命 ≈ {life_years:.1f} 年")
```

<div data-component="IoTBatteryCalculator"></div>

---

## 31.2 物联网通信协议

物联网通信协议是连接感知层与平台层的桥梁。与传统 HTTP 相比，IoT 协议需要满足低带宽、低功耗、高并发设备连接等约束。

### 31.2.1 MQTT：发布-订阅模型

**MQTT**（Message Queuing Telemetry Transport）是 IBM 于 1999 年提出的轻量级发布-订阅协议，已成为物联网的事实标准（OASIS 标准）。

#### **MQTT 报文格式**

MQTT 报文结构极其精简，固定头部仅 2 字节：

```
MQTT 报文结构：
┌─────────────────────────────────────────────────┐
│              固定头部 (Fixed Header)              │
│  ┌───────────────┬───────────────────────────┐  │
│  │  报文类型(4bit)│ 标志位(4bit) │ 剩余长度    │  │
│  │  0x3 = PUBLISH│ DUP|QoS|RETAIN│ 变长编码   │  │
│  └───────────────┴───────────────────────────┘  │
├─────────────────────────────────────────────────┤
│         可变头部 (Variable Header)               │
│  ┌───────────────────────────────────────────┐  │
│  │ 协议名("MQTT") │ 协议级别 │ 连接标志 │ 心跳 │  │
│  └───────────────────────────────────────────┘  │
├─────────────────────────────────────────────────┤
│              负载 (Payload)                      │
│  ┌───────────────────────────────────────────┐  │
│  │              应用数据                       │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

#### **MQTT QoS 三个等级**

| QoS 等级 | 名称 | 描述 | 传输保证 | 典型场景 |
|----------|------|------|---------|----------|
| **QoS 0** | At most once | 最多一次，发后即忘 | 可能丢失 | 温湿度周期上报 |
| **QoS 1** | At least once | 至少一次，有重传 | 不丢失，可能重复 | 传感器报警 |
| **QoS 2** | Exactly once | 恰好一次，四步握手 | 不丢失不重复 | 计费/OTA 关键指令 |

QoS 2 的四步握手流程：

```
QoS 2 四步握手（确保恰好一次投递）：
Publisher                  Broker                 Subscriber
    │                        │                        │
    │─── PUBLISH (QoS2) ───>│                        │
    │                        │─── PUBLISH (QoS2) ───>│
    │<── PUBREC ────────────│                        │
    │                        │<── PUBREC ────────────│
    │─── PUBREL ───────────>│                        │
    │                        │─── PUBREL ───────────>│
    │                        │<── PUBCOMP ───────────│
    │<── PUBCOMP ───────────│                        │
    │                        │                        │
```

#### **核心组件：MQTT Broker 消息路由引擎与 QoS 管理器**

MQTT Broker 是整个物联网消息系统的**中枢神经**，负责连接管理、消息路由和 QoS 保障。其内部架构如下：

```
MQTT Broker 内部架构：
┌─────────────────────────────────────────────────────────────┐
│                    MQTT Broker 核心                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              连接管理器 (Connection Manager)              ││
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              ││
│  │  │ TCP/TLS  │  │ WebSocket│  │ 认证引擎  │              ││
│  │  │ Accept   │  │ Upgrade  │  │ ACL检查   │              ││
│  │  └──────────┘  └──────────┘  └──────────┘              ││
│  └──────────────────────┬──────────────────────────────────┘│
│                         │                                    │
│  ┌──────────────────────▼──────────────────────────────────┐│
│  │          消息路由引擎 (Message Routing Engine)            ││
│  │  ┌──────────────────────────────────────────────────┐   ││
│  │  │           Topic Trie（主题前缀树）                 │   ││
│  │  │  root ──┬── "home" ──┬── "sensor" ─── "temp"     │   ││
│  │  │         │            └── "light" ─── "status"     │   ││
│  │  │         └── "factory" ──┬── "motor" ── "rpm"      │   ││
│  │  │                        └── "alarm" ── "critical"  │   ││
│  │  └──────────────────────────────────────────────────┘   ││
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              ││
│  │  │ 通配符匹配│  │ 消息去重 │  │ 持久化   │              ││
│  │  │ +/# 匹配 │  │ MsgID表  │  │ Retained │              ││
│  │  └──────────┘  └──────────┘  └──────────┘              ││
│  └──────────────────────┬──────────────────────────────────┘│
│                         │                                    │
│  ┌──────────────────────▼──────────────────────────────────┐│
│  │           QoS 管理器 (QoS Manager)                      ││
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              ││
│  │  │ QoS 0    │  │ QoS 1    │  │ QoS 2    │              ││
│  │  │ 即发即弃  │  │ ACK/重传 │  │ 四步握手  │              ││
│  │  │ 队列     │  │ Inflight │  │ 状态机   │              ││
│  │  │          │  │ Window   │  │          │              ││
│  │  └──────────┘  └──────────┘  └──────────┘              ││
│  │  ┌──────────────────────────────────────────────────┐   ││
│  │  │ Inflight Window（滑动窗口）                       │   ││
│  │  │ max_inflight = 20  → 并发未确认消息上限            │   ││
│  │  │ 重传定时器: T_retry = base * 2^retry_count       │   ││
│  │  └──────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────┘│
│                                                              │
│  性能指标：                                                    │
│  • 单节点并发连接: 100K+ (Erlang/OTP实现)                     │
│  • 消息吞吐: 100K msg/s (QoS 0), 50K msg/s (QoS 1)          │
│  • 路由延迟: < 1ms (P99)                                     │
│  • 内存开销: ~2KB/连接 (空闲), ~10KB/连接 (活跃)              │
└─────────────────────────────────────────────────────────────┘
```

**Topic Trie 路由算法**：

```python
class TopicTrie:
    """MQTT Broker 核心 —— Topic Trie 主题匹配引擎"""
    
    def __init__(self):
        self.root = {"children": {}, "subscribers": set()}
    
    def subscribe(self, topic_filter, client_id):
        """订阅主题（支持 + 和 # 通配符）"""
        parts = topic_filter.split("/")
        node = self.root
        for part in parts:
            if part not in node["children"]:
                node["children"][part] = {"children": {}, "subscribers": set()}
            node = node["children"][part]
        node["subscribers"].add(client_id)
    
    def match(self, topic_name):
        """发布消息时，匹配所有符合的订阅者"""
        parts = topic_name.split("/")
        result = set()
        self._match_recursive(self.root, parts, 0, result)
        return result
    
    def _match_recursive(self, node, parts, idx, result):
        if idx == len(parts):
            result.update(node["subscribers"])
            return
        
        current = parts[idx]
        
        # 精确匹配
        if current in node["children"]:
            self._match_recursive(node["children"][current], parts, idx + 1, result)
        
        # + 通配符：匹配单层任意主题
        if "+" in node["children"]:
            self._match_recursive(node["children"]["+"], parts, idx + 1, result)
        
        # # 通配符：匹配零层或多层任意主题
        if "#" in node["children"]:
            result.update(node["children"]["#"]["subscribers"])


# 模拟 Broker 运行
trie = TopicTrie()

# 订阅
trie.subscribe("home/+/temperature", "dashboard_01")
trie.subscribe("home/sensor/#", "data_collector")
trie.subscribe("factory/motor/rpm", "monitor_system")
trie.subscribe("#", "admin_logger")

# 发布消息匹配
matches = trie.match("home/living_room/temperature")
print(f"home/living_room/temperature → {matches}")
# 输出: {'dashboard_01', 'data_collector', 'admin_logger'}

matches = trie.match("factory/motor/rpm")
print(f"factory/motor/rpm → {matches}")
# 输出: {'monitor_system', 'admin_logger'}
```

**QoS 管理器状态机**：

```python
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Optional
import time

class QoS2State(Enum):
    """QoS 2 四步握手状态机"""
    INIT = auto()
    PUBLISH_SENT = auto()      # 已发送 PUBLISH
    PUBREC_RECEIVED = auto()   # 已收到 PUBREC
    PUBREL_SENT = auto()       # 已发送 PUBREL
    COMPLETED = auto()         # 握手完成

@dataclass
class InflightMessage:
    """飞行窗口中的消息"""
    msg_id: int
    topic: str
    payload: bytes
    qos: int
    state: QoS2State = QoS2State.INIT
    retry_count: int = 0
    last_sent: float = field(default_factory=time.time)
    created: float = field(default_factory=time.time)

class QoSManager:
    """
    MQTT Broker QoS 管理器
    负责 QoS 1/2 的消息确认、重传和去重
    """
    def __init__(self, max_inflight: int = 20, base_retry_ms: int = 1000):
        self.max_inflight = max_inflight
        self.base_retry_ms = base_retry_ms
        self.inflight: Dict[int, InflightMessage] = {}
        self.delivered_ids: set = set()  # 已投递消息ID（去重用）
        self.next_msg_id = 1
    
    def allocate_msg_id(self) -> int:
        msg_id = self.next_msg_id
        self.next_msg_id = (self.next_msg_id % 65535) + 1
        return msg_id
    
    def can_send(self) -> bool:
        """检查飞行窗口是否有空位"""
        return len(self.inflight) < self.max_inflight
    
    def on_publish_received(self, msg_id: int, topic: str, 
                            payload: bytes, qos: int) -> Optional[str]:
        """
        收到 PUBLISH 消息后的处理
        返回需要发送的响应类型: "none", "PUBACK", "PUBREC"
        """
        # 去重检查
        if qos > 0 and msg_id in self.delivered_ids:
            return "none"  # 已投递，忽略重复
        
        if qos == 0:
            return "none"  # QoS 0 无需确认
        
        if qos == 1:
            self.delivered_ids.add(msg_id)
            return "PUBACK"
        
        if qos == 2:
            # QoS 2：收到 PUBLISH → 回复 PUBREC
            self.inflight[msg_id] = InflightMessage(
                msg_id=msg_id, topic=topic, 
                payload=payload, qos=2,
                state=QoS2State.PUBLISH_SENT
            )
            return "PUBREC"
        
        return "none"
    
    def on_pubrec_received(self, msg_id: int) -> Optional[str]:
        """QoS 2: 收到 PUBREC → 发送 PUBREL"""
        if msg_id in self.inflight:
            msg = self.inflight[msg_id]
            msg.state = QoS2State.PUBREC_RECEIVED
            msg.state = QoS2State.PUBREL_SENT
            return "PUBREL"
        return None
    
    def on_pubcomp_received(self, msg_id: int) -> bool:
        """QoS 2: 收到 PUBCOMP → 握手完成"""
        if msg_id in self.inflight:
            self.delivered_ids.add(msg_id)
            del self.inflight[msg_id]
            return True
        return False
    
    def check_retransmissions(self) -> list:
        """检查需要重传的消息（QoS 1 和 QoS 2）"""
        now = time.time()
        retransmit = []
        for msg_id, msg in list(self.inflight.items()):
            timeout = self.base_retry_ms * (2 ** msg.retry_count) / 1000
            if now - msg.last_sent > timeout:
                msg.retry_count += 1
                msg.last_sent = now
                retransmit.append(msg)
        return retransmit


# 模拟 QoS 管理器
qos_mgr = QoSManager(max_inflight=10)

# QoS 0: 无需确认
resp = qos_mgr.on_publish_received(0, "sensor/temp", b"23.5", 0)
print(f"QoS 0 响应: {resp}")  # "none"

# QoS 1: 收到 → 回复 PUBACK
resp = qos_mgr.on_publish_received(1001, "sensor/humidity", b"65", 1)
print(f"QoS 1 响应: {resp}")  # "PUBACK"

# QoS 2: 四步握手
resp = qos_mgr.on_publish_received(2001, "cmd/restart", b"now", 2)
print(f"QoS 2 第1步: {resp}")  # "PUBREC"

resp = qos_mgr.on_pubrec_received(2001)
print(f"QoS 2 第2步: {resp}")  # "PUBREL"

resp = qos_mgr.on_pubcomp_received(2001)
print(f"QoS 2 完成: {resp}")  # True
```

<div data-component="MQTTQoSExplorer"></div>

### 31.2.2 CoAP：受限应用协议

**CoAP**（Constrained Application Protocol, RFC 7252）是专为受限设备设计的 RESTful 协议，运行在 UDP 之上。

#### **CoAP 报文格式**

```
CoAP 报文格式（固定 4 字节头部）：
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|Ver| T |  TKL  |      Code     |          Message ID           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|   Token (0-8 bytes, length = TKL)                             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|   Options (可选, 变长)                                         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|1 1 1 1 1 1 1 1|    Payload (可选)                              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

Ver  = 2 (协议版本)
T    = 0-3 (Confirmable/Non-confirmable/Acknowledgment/Reset)
Code = 分类.详情 (如 2.05 = Content, 4.04 = Not Found)
```

#### **CoAP 确认模式与观察模式**

```
Confirmable (CON) — 可靠传输：
Client                          Server
  │── CON [MID=0x7a3d] GET ────>│
  │   /sensors/temperature       │
  │<── ACK [MID=0x7a3d] 2.05 ──│
  │   Payload: 23.5              │

Non-confirmable (NON) — 不可靠传输：
Client                          Server
  │── NON [MID=0x7a3e] PUT ────>│
  │   /actuators/led             │
  │   Payload: on                │
  │   (无 ACK)                   │

Observe 模式 — 推送通知：
Client                          Server
  │── CON GET /sensors/temp ────>│
  │   Observe: 0 (注册)          │
  │<── ACK 2.05 Observe:0 ─────│
  │   Payload: 23.5              │
  │                              │
  │   (数据变化时主动推送)         │
  │<── CON [MID=0x7a40] 2.05 ──│
  │   Observe: 1                 │
  │   Payload: 24.1              │
  │── ACK [MID=0x7a40] ────────>│
```

#### **MQTT vs CoAP 对比**

| 特性 | MQTT | CoAP |
|------|------|------|
| **传输层** | TCP | UDP |
| **通信模型** | 发布-订阅 | 请求-响应（RESTful） |
| **消息开销** | 最小 2 字节 | 最小 4 字节 |
| **可靠性** | QoS 0/1/2 | CON/NON + 重传 |
| **推送能力** | 原生支持 | Observe 扩展 |
| **多播** | 不支持 | 支持（UDP） |
| **代理** | Broker 模式 | Proxy 模式 |
| **适用场景** | 设备-云通信 | 受限设备 REST API |
| **安全性** | TLS | DTLS |

### 31.2.3 AMQP：高级消息队列协议

**AMQP**（Advanced Message Queuing Protocol, ISO/IEC 19464）是企业级消息中间件标准协议，提供可靠的消息投递保障。

#### **AMQP 核心架构**

```
AMQP 消息路由模型：
┌──────────────────────────────────────────────────────────────┐
│  Producer                                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │ Sensor A │  │ Sensor B │  │ Sensor C │                   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                   │
│       │              │              │                         │
│  ┌────▼──────────────▼──────────────▼────┐                   │
│  │         Connection (TCP连接)           │                   │
│  │  ┌──────────┐  ┌──────────┐           │                   │
│  │  │ Channel 1│  │ Channel 2│  ...      │                   │
│  │  └────┬─────┘  └────┬─────┘           │                   │
│  └───────┼──────────────┼────────────────┘                   │
│          │              │                                     │
│  ┌───────▼──────────────▼────────────────────────────────┐   │
│  │                   Broker                                │   │
│  │  ┌──────────────────────────────────────────────────┐  │   │
│  │  │  Exchange (交换器)                                │  │   │
│  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐         │  │   │
│  │  │  │  Direct │  │  Topic  │  │  Fanout │         │  │   │
│  │  │  │ 精确路由 │  │ 主题路由 │  │ 广播    │         │  │   │
│  │  │  └────┬────┘  └────┬────┘  └────┬────┘         │  │   │
│  │  └───────┼────────────┼────────────┼───────────────┘  │   │
│  │          │ Binding    │            │                   │   │
│  │  ┌───────▼────────────▼────────────▼───────────────┐  │   │
│  │  │              Queue（消息队列）                    │  │   │
│  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐         │  │   │
│  │  │  │ Queue 1 │  │ Queue 2 │  │ Queue 3 │         │  │   │
│  │  │  │(持久化)  │  │(内存)   │  │(优先级)  │         │  │   │
│  │  │  └─────────┘  └─────────┘  └─────────┘         │  │   │
│  │  └──────────────────────────────────────────────────┘  │   │
│  └────────────────────────────────────────────────────────┘   │
│          │              │                                     │
│  ┌───────▼──────────────▼────────────────┐                   │
│  │           Consumer                     │                   │
│  │  ┌──────────┐  ┌──────────┐           │                   │
│  │  │ Service A│  │ Service B│           │                   │
│  │  └──────────┘  └──────────┘           │                   │
│  └───────────────────────────────────────┘                   │
└──────────────────────────────────────────────────────────────┘
```

| Exchange 类型 | 路由规则 | 示例 |
|--------------|---------|------|
| **Direct** | 精确匹配 Routing Key | `routing_key = "sensor.temp"` → 精确匹配 |
| **Topic** | 通配符匹配 | `sensor.*` 匹配 `sensor.temp`，`sensor.#` 匹配 `sensor.temp.raw` |
| **Fanout** | 广播到所有绑定队列 | 忽略 Routing Key，所有队列都收到 |
| **Headers** | 基于消息头匹配 | 按自定义 header 字段过滤 |

---

## 31.3 低功耗广域网（LPWAN）

### 31.3.1 LoRa/LoRaWAN

**LoRa**（Long Range）是 Semtech 公司开发的物理层调制技术，使用 **CSS**（Chirp Spread Spectrum，啁啾扩频）调制。**LoRaWAN** 是基于 LoRa 的 MAC 层协议。

#### **CSS 扩频调制原理**

CSS 调制通过线性频率啁啾（Chirp）信号实现扩频：

$$SF = \log_2\left(\frac{BW}{R_s}\right)$$

其中 $SF$ 为扩频因子（7-12），$BW$ 为带宽（125/250/500 kHz），$R_s$ 为符号速率。

```
CSS 扫频信号（时间-频率图）：
频率 ↑
     │    ╱‾‾‾‾‾‾‾‾‾‾╲         上啁啾（Up-chirp）
BW   │   ╱              ╲        频率从低到高线性扫描
     │  ╱                ╲
     │ ╱                  ╲
     │╱                    ╲
     └────────────────────────→ 时间
          一个 Chirp 符号持续时间 T = 2^SF / BW

频率 ↑
     │╲                    ╱
BW   │ ╲                  ╱       下啁啾（Down-chirp）
     │  ╲                ╱        频率从高到低线性扫描
     │   ╲              ╱
     │    ╲____________╱
     └────────────────────────→ 时间
```

**LoRa 数据速率计算**：

$$R_b = SF \times \frac{BW}{2^{SF}} \times CR$$

其中 $CR$ 为编码率（4/5, 4/6, 4/7, 4/8）。

| SF | BW (kHz) | 灵敏度 (dBm) | 比特率 (kbps) | 范围 (km) | TOA 20B (ms) |
|----|----------|--------------|--------------|-----------|-------------|
| 7 | 125 | -123 | 5.47 | 2 | 36 |
| 8 | 125 | -126 | 3.13 | 4 | 72 |
| 9 | 125 | -129 | 1.76 | 6 | 144 |
| 10 | 125 | -132 | 0.98 | 10 | 288 |
| 11 | 125 | -134 | 0.54 | 14 | 576 |
| 12 | 125 | -137 | 0.29 | 20 | 1152 |

```python
import math

def lora_data_rate(sf: int, bw_khz: float, cr: float = 4/5) -> float:
    """计算 LoRa 数据速率 (kbps)"""
    bw_hz = bw_khz * 1000
    rb = sf * (bw_hz / (2 ** sf)) * cr
    return rb / 1000  # 转换为 kbps

def lora_time_on_air(sf: int, bw_khz: float, payload_bytes: int = 20,
                     cr: float = 4/5, preamble: int = 8,
                     de: bool = True, crc: bool = True,
                     ih: int = 0) -> float:
    """
    计算 LoRa 空中传输时间（ms）
    de: 低数据速率优化（DR≤0 时启用）
    """
    bw_hz = bw_khz * 1000
    tsym = (2 ** sf) / bw_hz  # 符号持续时间
    
    # 前导码时间
    t_preamble = (preamble + 4.25) * tsym
    
    # 负载符号数计算
    de_val = 1 if de else 0
    crc_val = 1 if crc else 0
    
    # 有效负载比特数
    payload_bits = 8 * payload_bytes - 4 * sf + 8 + 16 * crc_val + 20 * ih
    payload_symbols = math.ceil(payload_bits / (4 * (sf - 2 * de_val)))
    payload_symbols = max(payload_symbols + 8, 0)  # +8 for explicit header
    
    t_payload = payload_symbols * tsym
    
    return (t_preamble + t_payload) * 1000  # 转换为 ms

def lora_link_budget(distance_km: float, sf: int,
                     tx_power_dbm: int = 14,
                     freq_mhz: float = 868.0) -> dict:
    """
    LoRa 链路预算计算
    使用自由空间路径损耗模型
    """
    # 自由空间路径损耗 (dB)
    fspl = 20 * math.log10(distance_km * 1000) + \
           20 * math.log10(freq_mhz * 1e6) - 147.55
    
    # 灵敏度参考值 (dBm)
    sensitivity = {7: -123, 8: -126, 9: -129, 10: -132, 11: -134, 12: -137}
    rx_sens = sensitivity.get(sf, -130)
    
    # 链路预算
    link_margin = tx_power_dbm - fspl - rx_sens
    
    return {
        "distance_km": distance_km,
        "fspl_db": round(fspl, 2),
        "tx_power_dbm": tx_power_dbm,
        "sensitivity_dbm": rx_sens,
        "link_margin_db": round(link_margin, 2),
        "feasible": link_margin > 0
    }


# 计算不同 SF 的参数
print("SF | 速率(kbps) | TOA(ms) | 灵敏度(dBm)")
print("-" * 50)
for sf in range(7, 13):
    rate = lora_data_rate(sf, 125)
    toa = lora_time_on_air(sf, 125, 20)
    sens = {7: -123, 8: -126, 9: -129, 10: -132, 11: -134, 12: -137}
    print(f" {sf} |  {rate:8.2f} | {toa:7.1f} | {sens[sf]}")

# 链路预算示例
print("\n链路预算分析 (SF=10, 868MHz, 14dBm):")
for d in [1, 5, 10, 15, 20]:
    result = lora_link_budget(d, sf=10, tx_power_dbm=14, freq_mhz=868)
    status = "✓" if result["feasible"] else "✗"
    print(f"  {d:2d}km: 余量={result['link_margin_db']:6.1f}dB {status}")
```

#### **LoRaWAN 设备类别**

| 类别 | 下行窗口 | 上行 | 延迟 | 功耗 | 应用场景 |
|------|---------|------|------|------|----------|
| **Class A** | 仅发送后开两个短窗 | 随时 | 高（分钟级） | 最低 | 传感器周期上报 |
| **Class B** | 定时信标窗口 | 随时 | 中（秒级） | 低 | 定时控制/读表 |
| **Class C** | 持续监听 | 随时 | 最低（实时） | 最高 | 路灯/执行器控制 |

#### **核心组件：LoRa 网关扩频调制/解调器与信道活动检测器**

LoRa 网关是连接终端设备与网络服务器的关键设备，其核心组件包括：

```
LoRa 网关内部架构：
┌──────────────────────────────────────────────────────────────┐
│                    LoRa 网关                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              RF 前端 (RF Front-End)                      │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │ │
│  │  │ LNA      │  │ SAW Filter│ │ PA       │              │ │
│  │  │ 低噪放大  │  │ 声表面滤波│  │ 功率放大  │              │ │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘              │ │
│  └───────┼──────────────┼──────────────┼────────────────────┘ │
│          │              │              │                       │
│  ┌───────▼──────────────▼──────────────┼────────────────────┐ │
│  │              LoRa 收发器 (SX1301/SX1302)                  │ │
│  │  ┌──────────────────────────────────────────────────┐    │ │
│  │  │  扩频解调器 (CSS Demodulator)                     │    │ │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐       │    │ │
│  │  │  │ 去啁啾    │  │ FFT/     │  │ 符号解码  │       │    │ │
│  │  │  │ De-chirp │  │ 频谱分析  │  │ FEC解码  │       │    │ │
│  │  │  └──────────┘  └──────────┘  └──────────┘       │    │ │
│  │  │  支持 8 个并行解调通道 (8-Channel Gateway)        │    │ │
│  │  └──────────────────────────────────────────────────┘    │ │
│  │  ┌──────────────────────────────────────────────────┐    │ │
│  │  │  信道活动检测器 (CAD - Channel Activity Detection)│    │ │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐       │    │ │
│  │  │  │ 能量检测  │  │ Chirp    │  │ 判决逻辑  │       │    │ │
│  │  │  │ RSSI >   │  │ 相关检测  │  │ 虚警/     │       │    │ │
│  │  │  │ threshold │  │ SF匹配   │  │ 漏检平衡  │       │    │ │
│  │  │  └──────────┘  └──────────┘  └──────────┘       │    │ │
│  │  │  CAD 性能：                                      │    │ │
│  │  │  • 检测概率: Pd > 99% (SNR > -20dB)             │    │ │
│  │  │  • 虚警概率: Pfa < 1%                            │    │ │
│  │  │  • 检测时间: T_cad ≈ 2^SF / BW * (SF+1)        │    │ │
│  │  └──────────────────────────────────────────────────┘    │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              协议引擎 (Packet Forwarder)                  │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │ │
│  │  │ MAC解析  │  │ 上行转发  │  │ 下行调度  │              │ │
│  │  │ MHDR/FHDR│  │ →NS(UDP) │  │ RX1/RX2  │              │ │
│  │  └──────────┘  └──────────┘  └──────────┘              │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  性能指标：                                                     │
│  • 并发通道: 8 (EU868) / 64 (US915, 使用64信道子带)            │
│  • 最大吞吐: ~15000 packets/hour (SF7) / ~300 packets/hour (SF12)│
│  • 接收灵敏度: -137 dBm (SF12, BW125)                         │
│  • CAD 时间: ~12ms (SF10, BW125)                               │
└──────────────────────────────────────────────────────────────┘
```

**CAD 工作流程**：

```python
import numpy as np

def channel_activity_detection(signal_samples, sf, bw, 
                                snr_threshold_db=-20):
    """
    信道活动检测 (CAD) 模拟
    通过去啁啾（De-chirp）相关检测判断信道是否有 LoRa 信号
    
    参数:
        signal_samples: 接收信号采样（复数）
        sf: 扩频因子
        bw: 带宽 (Hz)
        snr_threshold_db: 判决门限
    
    返回:
        (检测结果, 估计SNR, 估计频率偏移)
    """
    n_samples = len(signal_samples)
    
    # 生成本地参考 Chirp
    t = np.arange(n_samples) / bw
    ref_chirp = np.exp(1j * np.pi * (bw / (2**sf)) * t**2)
    
    # 去啁啾：接收信号与参考 Chirp 共轭相乘
    dechirped = signal_samples * np.conj(ref_chirp)
    
    # FFT 频谱分析
    spectrum = np.abs(np.fft.fft(dechirped))
    peak_idx = np.argmax(spectrum)
    peak_power = spectrum[peak_idx] ** 2
    
    # 估计噪声功率（排除峰值附近的信号带宽）
    noise_mask = np.ones(len(spectrum), dtype=bool)
    guard = max(3, n_samples // (2**sf))
    noise_mask[max(0, peak_idx-guard):min(len(spectrum), peak_idx+guard)] = False
    noise_power = np.mean(spectrum[noise_mask] ** 2) + 1e-12
    
    # SNR 估计
    snr_db = 10 * np.log10(peak_power / noise_power)
    
    # 估计频率偏移
    freq_offset = (peak_idx / n_samples - 0.5) * bw
    
    # 判决
    detected = snr_db > snr_threshold_db
    
    return detected, snr_db, freq_offset

# CAD 模拟
sf, bw = 10, 125000
n = 2**sf
np.random.seed(42)

# 无信号（纯噪声）
noise = np.random.randn(n) + 1j * np.random.randn(n)
detected, snr, foff = channel_activity_detection(noise, sf, bw)
print(f"纯噪声: detected={detected}, SNR={snr:.1f}dB, Δf={foff:.0f}Hz")

# 有信号（SNR ≈ -15dB）
t = np.arange(n) / bw
signal = 0.02 * np.exp(1j * np.pi * (bw/(2**sf)) * t**2 + 1j * 0.5)
noisy_signal = signal + noise
detected, snr, foff = channel_activity_detection(noisy_signal, sf, bw)
print(f"含信号: detected={detected}, SNR={snr:.1f}dB, Δf={foff:.0f}Hz")
```

<div data-component="LoRaRangeCalculator"></div>

### 31.3.2 Sigfox：超窄带技术

**Sigfox** 采用**超窄带**（Ultra-Narrow Band, UNB）调制，信道带宽仅 100 Hz：

| 参数 | 值 |
|------|-----|
| **带宽** | 100 Hz（超窄带） |
| **调制** | DBPSK（差分二进制相移键控） |
| **上行消息** | 最大 12 字节负载 + 13 字节头部 |
| **下行消息** | 最大 8 字节负载 |
| **每日上行** | 140 条消息 |
| **每日下行** | 4 条消息 |
| **覆盖距离** | 城市 3-10 km，郊区 30-50 km |
| **功耗** | ~50 μJ/bit |

### 31.3.3 NB-IoT：窄带物联网

**NB-IoT**（Narrowband IoT, 3GPP R13）基于 LTE 演进，部署模式有三种：

```
NB-IoT 三种部署模式：
┌─────────────────────────────────────────────────┐
│  Standalone（独立部署）                           │
│  ┌─────────────────────────────────────┐        │
│  │  180 kHz 带宽                       │        │
│  │  独立频段，可部署于任何运营商频谱      │        │
│  └─────────────────────────────────────┘        │
│                                                  │
│  In-band（带内部署）                              │
│  ┌─────────────────────────────────────┐        │
│  │  LTE 20MHz 带宽                     │        │
│  │  ┌──┬──┬──┬NB┬──┬──┬──┬──┬──┬──┐   │        │
│  │  │  │  │  │IoT│  │  │  │  │  │  │   │        │
│  │  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘   │        │
│  │  占用 1 个 LTE PRB (180kHz)          │        │
│  └─────────────────────────────────────┘        │
│                                                  │
│  Guard-band（保护带部署）                         │
│  ┌─────────────────────────────────────┐        │
│  │  LTE 20MHz 带宽                     │        │
│  │  [LTE 资源块]│NB-IoT│[保护带]       │        │
│  │  利用 LTE 频段边缘的保护带资源        │        │
│  └─────────────────────────────────────┘        │
└─────────────────────────────────────────────────┘
```

| 特性 | NB-IoT | LTE-M |
|------|--------|-------|
| **带宽** | 180 kHz | 1.4 MHz |
| **数据速率** | ~250 kbps | ~1 Mbps |
| **延迟** | 1.5-10 s | 50-100 ms |
| **移动性** | 不支持 | 支持（小区切换） |
| **语音** | 不支持 | 支持（VoLTE） |
| **覆盖增强** | 20 dB (MCL 164dB) | 15 dB (MCL 156dB) |
| **模块成本** | ~$3-5 | ~$5-10 |
| **适用场景** | 固定传感器/智能抄表 | 资产追踪/可穿戴 |

### 31.3.4 LPWAN 技术全面对比

| 技术 | 频段 | 带宽 | 范围 | 数据速率 | 功耗 | 部署 |
|------|------|------|------|---------|------|------|
| **LoRa/LoRaWAN** | 868/915 MHz ISM | 125-500 kHz | 2-20 km | 0.3-50 kbps | 极低 | 私有/公有 |
| **Sigfox** | 868/915 MHz ISM | 100 Hz | 3-50 km | 100-600 bps | 极低 | 仅公有 |
| **NB-IoT** | 授权频段 | 180 kHz | 1-10 km | 250 kbps | 低 | 运营商 |
| **LTE-M** | 授权频段 | 1.4 MHz | 1-10 km | 1 Mbps | 中 | 运营商 |
| **ZigBee** | 2.4 GHz ISM | 2 MHz | 10-100 m | 250 kbps | 低 | 私有 |

---

## 31.4 6LoWPAN 与受限网络

**6LoWPAN**（IPv6 over Low-Power Wireless Personal Area Networks, RFC 4919/4944）解决了 IPv6 在 IEEE 802.15.4 网络上的适配问题。

### 31.4.1 报头压缩

IPv6 标准报头为 40 字节，而 IEEE 802.15.4 帧的最大负载仅 127 字节。6LoWPAN 通过 **HC1**（Header Compression 1）和 **HC2** 压缩技术大幅减小开销：

```
IPv6 报头压缩过程：
┌──────────────────────────────────────────────────────┐
│ IPv6 标准报头 (40 字节)                               │
│ ┌────────┬──────┬──────┬──────────────────────────┐  │
│ │Version │TC    │Flow  │       20B → 2B (HC1)     │  │
│ │4b=0110 │8b    │20b   │ Payload Length→0(隐含)   │  │
│ ├────────┴──────┴──────┼──────────────────────────┤  │
│ │Next Hdr │Hop Limit   │ Next Hdr→UDP(隐含)       │  │
│ │8b       │8b          │ Hop Limit→FF(隐含)       │  │
│ ├──────────────────────┼──────────────────────────┤  │
│ │Source Address (128bit)│ 16B → 0-8B (HC1)        │  │
│ │ link-local + EUI-64  │ 可完全压缩               │  │
│ ├──────────────────────┼──────────────────────────┤  │
│ │Dest Address (128bit) │ 16B → 0-8B (HC1)        │  │
│ └──────────────────────┴──────────────────────────┘  │
└──────────────────────────────────────────────────────┘
         │  HC1 压缩
         ▼
┌──────────────────────────────────────────────────────┐
│ 6LoWPAN 压缩后 (最小 ~7 字节)                         │
│ ┌──────┬────────────────────────────────────────┐    │
│ │Dispatch│ HC1 编码字节 │ UDP HC2 │ 压缩地址/负载 │    │
│ │1B     │ 1B           │ 1-2B    │ 0-16B       │    │
│ └──────┴────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

### 31.4.2 分片与重组

当 IPv6 数据包超过 802.15.4 帧负载限制时，6LoWPAN 需要分片：

```
6LoWPAN 分片格式：
第一个分片：
┌─────────┬────────┬────────┬─────────────────────┐
│Dispatch │Datagram│Datagram│      Payload        │
│(11100001│ Size   │ Tag    │                     │
│=0xE1)   │(11bit) │(16bit) │                     │
└─────────┴────────┴────────┴─────────────────────┘

后续分片：
┌─────────┬────────┬────────┬────────┬─────────────┐
│Dispatch │Datagram│Datagram│Offset  │  Payload    │
│(11100000│ Size   │ Tag    │(8bit)  │             │
│=0xE0)   │(11bit) │(16bit) │        │             │
└─────────┴────────┴────────┴────────┴─────────────┘
```

### 31.4.3 Mesh-under vs Route-over

| 特性 | Mesh-under | Route-over |
|------|-----------|-----------|
| **路由层** | 链路层（L2） | 网络层（L3） |
| **路由协议** | HWMP（类似AODV） | RPL（IPv6路由） |
| **分片处理** | 端到端重组 | 逐跳重组 |
| **适用场景** | 小型网络 | 大规模网络 |
| **开销** | 低 | 较高 |

---

## 31.5 边缘计算网络架构

### 31.5.1 MEC：多接入边缘计算

**MEC**（Multi-access Edge Computing）由 ETSI 定义，将计算能力下沉到无线接入网（RAN）边缘。

#### **ETSI MEC 参考架构**

```
ETSI MEC 参考架构：
┌──────────────────────────────────────────────────────────────┐
│                      MEC 主机 (MEC Host)                      │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              MEC 应用层                                   │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │ │
│  │  │ App 1    │  │ App 2    │  │ App 3    │              │ │
│  │  │(视频缓存) │  │(AR渲染)  │  │(IoT聚合) │              │ │
│  │  └──────────┘  └──────────┘  └──────────┘              │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              MEC 平台层                                   │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │ │
│  │  │ 路由规则  │  │ DNS 规则 │  │ 无线网络  │              │ │
│  │  │ Engine   │  │ Engine   │  │ 信息API   │              │ │
│  │  └──────────┘  └──────────┘  └──────────┘              │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │ │
│  │  │ 流量规则  │  │ 带宽管理  │  │ 计费/     │              │ │
│  │  │ API      │  │ API      │  │ 计量      │              │ │
│  │  └──────────┘  └──────────┘  └──────────┘              │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              虚拟化基础设施 (NFVI)                        │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              │ │
│  │  │ 计算     │  │ 存储     │  │ 网络虚拟化 │              │ │
│  │  │ (VM/容器) │  │ (分布式) │  │ (vSwitch) │              │ │
│  │  └──────────┘  └──────────┘  └──────────┘              │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
         │                    │                    │
  ┌──────▼──────┐     ┌──────▼──────┐     ┌──────▼──────┐
  │  MEC 编排器  │     │ 运维支撑    │     │  BSS/OSS    │
  │  (MEC       │     │ (Operations │     │             │
  │  Orchestrator│    │  Support)   │     │             │
  └─────────────┘     └─────────────┘     └─────────────┘
```

**MEC 应用生命周期**：

1. **Onboarding**：应用包（.csar）上架到 App Registry
2. **Instantiation**：MEC Orchestrator 分配资源并部署 VNF/CNF
3. **Configuration**：配置 DNS 规则、路由规则和流量规则
4. **Running**：应用处理流量，上报无线网络信息
5. **Termination**：资源释放，规则清理

### 31.5.2 雾计算

**雾计算**（Fog Computing）由 Cisco 于 2012 年提出，强调从云到物之间的连续计算层次：

```
Cisco 雾计算架构：
┌───────────────────────────────────────────────────────┐
│                    云 (Cloud)                          │
│  大规模存储、全局分析、深度学习训练                       │
└───────────────────────┬───────────────────────────────┘
                        │
┌───────────────────────▼───────────────────────────────┐
│                 核心雾节点 (Fog Core)                   │
│  区域数据中心、数据聚合、模型推理                         │
└───────────────────────┬───────────────────────────────┘
                        │
┌───────────────────────▼───────────────────────────────┐
│                 边缘雾节点 (Fog Edge)                   │
│  基站/路由器/网关、实时决策、本地缓存                     │
└───────────────────────┬───────────────────────────────┘
                        │
┌───────────────────────▼───────────────────────────────┐
│               终端设备 (End Devices)                    │
│  传感器、执行器、穿戴设备                                │
└───────────────────────────────────────────────────────┘
```

| 特性 | MEC | 雾计算 |
|------|-----|--------|
| **标准化** | ETSI | OpenFog Consortium → IIC |
| **部署位置** | RAN 边缘（靠近基站） | 云与物之间的任意层次 |
| **架构风格** | 平台即服务 | 分布式计算层次 |
| **核心关注** | 低延迟无线接入 | 异构分布式计算 |
| **编排** | MANO/MEC Orchestrator | 分布式编排 |

### 31.5.3 核心组件：边缘计算资源编排器与任务调度器

边缘计算系统的核心挑战在于如何在资源受限、分布异构的边缘节点上高效调度计算任务。

#### **资源模型**

```
边缘计算资源层次模型：
┌──────────────────────────────────────────────────────────────┐
│  资源编排器 (Resource Orchestrator)                            │
│  ┌──────────────────────────────────────────────────────────┐│
│  │            全局资源视图 (Global Resource View)             ││
│  │                                                          ││
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    ││
│  │  │ Edge    │  │ Edge    │  │ Edge    │  │ Cloud   │    ││
│  │  │ Node A  │  │ Node B  │  │ Node C  │  │ DC      │    ││
│  │  │ 4C/8G   │  │ 2C/4G   │  │ 8C/16G  │  │ 64C/256G│    ││
│  │  │ GPU:T4  │  │ 无GPU   │  │ GPU:Jetson│ │ GPU:A100│    ││
│  │  │ 延迟2ms │  │ 延迟5ms │  │ 延迟1ms │  │ 延迟50ms│    ││
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘    ││
│  └──────────────────────────────────────────────────────────┘│
│  ┌──────────────────────────────────────────────────────────┐│
│  │            任务调度器 (Task Scheduler)                     ││
│  │  ┌──────────────────────────────────────────────────┐    ││
│  │  │ 任务 DAG (有向无环图)                             │    ││
│  │  │                                                  │    ││
│  │  │    ┌───┐     ┌───┐     ┌───┐                    │    ││
│  │  │    │ T1│────>│ T2│────>│ T4│                    │    ││
│  │  │    │采集│     │推理│     │聚合│                    │    ││
│  │  │    └─┬─┘     └─┬─┘     └───┘                    │    ││
│  │  │      │         │                                 │    ││
│  │  │      ▼         ▼                                 │    ││
│  │  │    ┌───┐     ┌───┐                              │    ││
│  │  │    │ T3│────>│ T5│                              │    ││
│  │  │    │预处理│   │存储│                              │    ││
│  │  │    └───┘     └───┘                              │    ││
│  │  └──────────────────────────────────────────────────┘    ││
│  │  ┌──────────────────────────────────────────────────┐    ││
│  │  │  调度策略                                         │    ││
│  │  │  • 最小延迟 (Latency-first)                      │    ││
│  │  │  • 最小能耗 (Energy-aware)                       │    ││
│  │  │  • 成本最优 (Cost-optimal)                       │    ││
│  │  │  • 负载均衡 (Load-balanced)                      │    ││
│  │  └──────────────────────────────────────────────────┘    ││
│  └──────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

#### **任务 DAG 调度算法**

边缘计算任务通常表示为 DAG（Directed Acyclic Graph），调度问题是 NP-hard 的。常用启发式算法包括 HEFT（Heterogeneous Earliest Finish Time）：

```python
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

@dataclass
class EdgeNode:
    """边缘计算节点"""
    node_id: str
    cpu_cores: float
    memory_gb: float
    gpu_type: Optional[str]
    latency_ms: float
    cost_per_hour: float
    available_at: float = 0.0  # 节点空闲时间点

@dataclass
class Task:
    """计算任务"""
    task_id: str
    cpu_req: float      # CPU 核心数需求
    mem_req: float      # 内存需求 (GB)
    gpu_req: bool       # 是否需要 GPU
    data_size_mb: float # 输入数据大小
    predecessors: List[str] = field(default_factory=list)
    execution_times: Dict[str, float] = field(default_factory=dict)  # node_id → exec_time

@dataclass
class ScheduleResult:
    """调度结果"""
    task_id: str
    node_id: str
    start_time: float
    finish_time: float
    transfer_time: float

class HEFTScheduler:
    """
    HEFT (Heterogeneous Earliest Finish Time) 调度算法
    
    核心思想：
    1. 自底向上计算每个任务的向上排序值 (Upward Rank)
    2. 按 Rank 降序排列任务
    3. 为每个任务选择最早完成的节点
    """
    
    def __init__(self, nodes: List[EdgeNode], tasks: Dict[str, Task],
                 bandwidth_mbps: float = 100):
        self.nodes = {n.node_id: n for n in nodes}
        self.tasks = tasks
        self.bandwidth = bandwidth_mbps
        self.schedule: Dict[str, ScheduleResult] = {}
        self._comm_costs: Dict[Tuple[str, str], float] = {}
    
    def _communication_cost(self, task_from: str, task_to: str, 
                           node_from: str, node_to: str) -> float:
        """计算通信开销（不同节点间的数据传输时间）"""
        if node_from == node_to:
            return 0.0  # 同节点无通信开销
        data_mb = self.tasks[task_from].data_size_mb
        return data_mb * 8 / self.bandwidth  # 转换为秒
    
    def _compute_upward_rank(self, task_id: str, 
                             memo: Dict[str, float]) -> float:
        """递归计算向上排序值"""
        if task_id in memo:
            return memo[task_id]
        
        task = self.tasks[task_id]
        # 平均执行时间
        avg_exec = sum(task.execution_times.values()) / len(task.execution_times)
        
        # max over successors of (comm_cost + rank_successor)
        max_successor = 0.0
        for succ_id, succ_task in self.tasks.items():
            if task_id in succ_task.predecessors:
                avg_comm = 0.0
                for n1 in self.nodes:
                    for n2 in self.nodes:
                        if n1 != n2:
                            avg_comm += self._communication_cost(
                                task_id, succ_id, n1, n2)
                n_pairs = len(self.nodes) * (len(self.nodes) - 1)
                avg_comm = avg_comm / n_pairs if n_pairs > 0 else 0
                
                succ_rank = self._compute_upward_rank(succ_id, memo)
                max_successor = max(max_successor, avg_comm + succ_rank)
        
        memo[task_id] = avg_exec + max_successor
        return memo[task_id]
    
    def schedule(self) -> List[ScheduleResult]:
        """执行 HEFT 调度"""
        # 1. 计算所有任务的 upward rank
        ranks: Dict[str, float] = {}
        for task_id in self.tasks:
            self._compute_upward_rank(task_id, ranks)
        
        # 2. 按 rank 降序排序
        sorted_tasks = sorted(ranks.keys(), key=lambda t: ranks[t], reverse=True)
        
        # 3. 依次调度每个任务
        for task_id in sorted_tasks:
            task = self.tasks[task_id]
            best_result = None
            best_finish = float("inf")
            
            for node_id, node in self.nodes.items():
                # 检查资源约束
                if task.gpu_req and node.gpu_type is None:
                    continue
                if task.cpu_req > node.cpu_cores:
                    continue
                if task.mem_req > node.memory_gb:
                    continue
                
                # 计算最早开始时间
                earliest_start = node.available_at
                
                # 考虑前驱任务的通信开销
                for pred_id in task.predecessors:
                    if pred_id in self.schedule:
                        pred_result = self.schedule[pred_id]
                        comm_time = self._communication_cost(
                            pred_id, task_id,
                            pred_result.node_id, node_id
                        )
                        ready_time = pred_result.finish_time + comm_time
                        earliest_start = max(earliest_start, ready_time)
                
                # 执行时间
                exec_time = task.execution_times.get(node_id, float("inf"))
                finish_time = earliest_start + exec_time
                
                if finish_time < best_finish:
                    best_finish = finish_time
                    best_result = ScheduleResult(
                        task_id=task_id,
                        node_id=node_id,
                        start_time=earliest_start,
                        finish_time=finish_time,
                        transfer_time=0  # 简化
                    )
            
            if best_result:
                self.schedule[task_id] = best_result
                self.nodes[best_result.node_id].available_at = best_result.finish_time
        
        return list(self.schedule.values())


# 创建边缘节点
nodes = [
    EdgeNode("edge_A", cpu_cores=4, memory_gb=8, 
             gpu_type="T4", latency_ms=2, cost_per_hour=0.5),
    EdgeNode("edge_B", cpu_cores=2, memory_gb=4, 
             gpu_type=None, latency_ms=5, cost_per_hour=0.2),
    EdgeNode("edge_C", cpu_cores=8, memory_gb=16, 
             gpu_type="Jetson", latency_ms=1, cost_per_hour=1.0),
    EdgeNode("cloud", cpu_cores=64, memory_gb=256, 
             gpu_type="A100", latency_ms=50, cost_per_hour=5.0),
]

# 创建任务 DAG
tasks = {
    "T1": Task("T1", 1, 1, False, 10, predecessors=[],
               execution_times={"edge_A": 2, "edge_B": 3, "edge_C": 1, "cloud": 0.5}),
    "T2": Task("T2", 4, 4, True, 50, predecessors=["T1"],
               execution_times={"edge_A": 5, "edge_B": 15, "edge_C": 3, "cloud": 1}),
    "T3": Task("T3", 2, 2, False, 30, predecessors=["T1"],
               execution_times={"edge_A": 3, "edge_B": 4, "edge_C": 2, "cloud": 0.8}),
    "T4": Task("T4", 2, 2, False, 20, predecessors=["T2", "T3"],
               execution_times={"edge_A": 2, "edge_B": 3, "edge_C": 1.5, "cloud": 0.5}),
    "T5": Task("T5", 1, 1, False, 10, predecessors=["T3"],
               execution_times={"edge_A": 1, "edge_B": 2, "edge_C": 0.8, "cloud": 0.3}),
}

# 运行 HEFT 调度
scheduler = HEFTScheduler(nodes, tasks, bandwidth_mbps=100)
results = scheduler.schedule()

print("HEFT 调度结果：")
print(f"{'任务':<6} {'节点':<10} {'开始时间':<10} {'完成时间':<10}")
print("-" * 40)
for r in sorted(results, key=lambda x: x.start_time):
    print(f"{r.task_id:<6} {r.node_id:<10} {r.start_time:<10.2f} {r.finish_time:<10.2f}")

makespan = max(r.finish_time for r in results)
print(f"\n总完工时间 (Makespan): {makespan:.2f} 秒")
```

#### **容器编排与边缘部署**

边缘计算场景下，Kubernetes 的轻量化变体（如 K3s、KubeEdge）用于容器编排：

```yaml
# KubeEdge 边缘应用部署示例
apiVersion: apps/v1
kind: Deployment
metadata:
  name: edge-inference-app
  namespace: edge-apps
spec:
  replicas: 3
  selector:
    matchLabels:
      app: inference
  template:
    metadata:
      labels:
        app: inference
        node-type: edge
    spec:
      nodeSelector:
        kubernetes.io/hostname: edge-node-01  # 绑定到特定边缘节点
      containers:
      - name: inference
        image: registry.example.com/inference:v1.2
        resources:
          limits:
            cpu: "2"
            memory: "2Gi"
            nvidia.com/gpu: "1"  # GPU 资源
          requests:
            cpu: "500m"
            memory: "512Mi"
        env:
        - name: MODEL_PATH
          value: "/models/yolov5s.onnx"
        - name: INFERENCE_INTERVAL
          value: "100ms"
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: edge-model-pvc
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: edge-inference-netpol
spec:
  podSelector:
    matchLabels:
      app: inference
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: gateway
    ports:
    - port: 8080
      protocol: TCP
```

<div data-component="EdgeSchedulerSimulator"></div>

---

## 31.6 5G 网络切片

### 31.6.1 三大应用场景

5G 网络切片通过在同一物理基础设施上创建多个**端到端逻辑网络**来满足不同业务需求：

```
5G 三大切片场景：
┌──────────────────────────────────────────────────────────────────┐
│                    5G 物理基础设施                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ RAN (gNB)│  │ 传输网   │  │ 核心网   │  │ 边缘DC   │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│       ║              ║              ║              ║              │
│  ═════╬══════════════╬══════════════╬══════════════╬══════════    │
│       ║              ║              ║              ║              │
│  ┌────╨──────────────╨──────────────╨──────────────╨──────────┐  │
│  │                    网络切片管理器                              │  │
│  └────╥──────────────╥──────────────╥──────────────╥──────────┘  │
│       ║              ║              ║              ║              │
│  ┌────╨────┐   ┌─────╨────┐   ┌─────╨────┐                       │
│  │  eMBB   │   │  URLLC   │   │  mMTC    │                       │
│  │ 增强移动 │   │ 超可靠低 │   │ 海量机器 │                       │
│  │ 宽带    │   │ 延迟通信 │   │ 类通信   │                       │
│  │         │   │          │   │          │                       │
│  │ 4K视频  │   │ 自动驾驶 │   │ 智能抄表 │                       │
│  │ VR/AR   │   │ 远程手术 │   │ 智慧农业 │                       │
│  │ 游戏    │   │ 工业控制 │   │ 物联网   │                       │
│  └─────────┘   └──────────┘   └──────────┘                       │
└──────────────────────────────────────────────────────────────────┘
```

| 参数 | eMBB | URLLC | mMTC |
|------|------|-------|------|
| **带宽** | >1 Gbps | 10-100 Mbps | <1 Mbps |
| **延迟** | <10 ms | <1 ms | <10 s |
| **可靠性** | 99.9% | 99.999% | 99% |
| **连接密度** | 10³/km² | 10⁴/km² | 10⁶/km² |
| **移动性** | 高速 | 低速/固定 | 固定/低速 |
| **典型应用** | 视频/VR | 远程手术/车联网 | 智能抄表/传感器 |

### 31.6.2 核心组件：5G 切片管理器与资源隔离保障

```
5G 切片管理器内部架构：
┌──────────────────────────────────────────────────────────────┐
│                 切片管理器 (Slice Manager)                      │
│  ┌──────────────────────────────────────────────────────────┐│
│  │            切片生命周期管理                                 ││
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              ││
│  │  │ 切片创建  │  │ 切片修改  │  │ 切片删除  │              ││
│  │  │ (Create)  │  │ (Modify) │  │ (Delete) │              ││
│  │  └──────────┘  └──────────┘  └──────────┘              ││
│  └──────────────────────────────────────────────────────────┘│
│  ┌──────────────────────────────────────────────────────────┐│
│  │            资源隔离引擎 (Resource Isolation Engine)        ││
│  │                                                          ││
│  │  ┌──────────────────────────────────────────────────┐    ││
│  │  │  RAN 资源隔离                                      │    ││
│  │  │  • 频谱资源: RB (Resource Block) 静态/动态分配    │    ││
│  │  │  • 时域资源: 时隙 (Slot) 预留/抢占                │    ││
│  │  │  • 空域资源: 波束 (Beam) 专用/共享                │    ││
│  │  └──────────────────────────────────────────────────┘    ││
│  │  ┌──────────────────────────────────────────────────┐    ││
│  │  │  核心网资源隔离                                     │    ││
│  │  │  • UPF (User Plane Function) 实例隔离            │    ││
│  │  │  • 带宽保障: 最小保证带宽 + 最大带宽限制           │    ││
│  │  │  • 计算资源: CPU/Memory 容器级隔离                │    ││
│  │  └──────────────────────────────────────────────────┘    ││
│  │  ┌──────────────────────────────────────────────────┐    ││
│  │  │  传输网资源隔离                                     │    ││
│  │  │  • FlexE (灵活以太网) 硬管道隔离                  │    ││
│  │  │  • VLAN/MPLS 标签隔离                             │    ││
│  │  │  • SDN 流表隔离                                    │    ││
│  │  └──────────────────────────────────────────────────┘    ││
│  └──────────────────────────────────────────────────────────┘│
│  ┌──────────────────────────────────────────────────────────┐│
│  │            SLA 保障引擎 (SLA Assurance Engine)            ││
│  │                                                          ││
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐              ││
│  │  │ SLA 模板 │  │ 监控引擎  │  │ 自愈引擎  │              ││
│  │  │ 管理    │  │ (KPI     │  │ (Self-   │              ││
│  │  │         │  │ Monitor) │  │ healing) │              ││
│  │  └──────────┘  └──────────┘  └──────────┘              ││
│  │                                                          ││
│  │  SLA 指标监控：                                          ││
│  │  • 吞吐量: 实际 vs 保证 → 触发扩容                      ││
│  │  • 延迟: 实际 vs 目标 → 触发迁移                        ││
│  │  • 丢包率: 实际 vs 阈值 → 触发切换                      ││
│  │  • 可用性: 实际 vs SLA → 触发自愈                       ││
│  │                                                          ││
│  │  自愈动作链：                                             ││
│  │  SLA违规检测 → 根因分析 → 执行修复 → 验证恢复            ││
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐              ││
│  │  │扩容/缩容│   │切片迁移  │   │流量切换  │              ││
│  │  │Scale   │   │Migrate │   │Failover │              ││
│  │  └─────────┘   └─────────┘   └─────────┘              ││
│  └──────────────────────────────────────────────────────────┘│
│                                                                │
│  性能指标：                                                     │
│  • 切片创建时间: < 30 秒 (轻量级) / < 5 分钟 (完整)            │
│  • 资源隔离粒度: RAN(RB级) / CN(UPF实例) / Transport(FlexE)   │
│  • SLA 监控周期: 100ms (URLLC) / 1s (eMBB) / 10s (mMTC)      │
│  • 故障自愈时间: < 10 秒 (自动化)                               │
└──────────────────────────────────────────────────────────────┘
```

**切片 SLA 模型**：

$$\text{SLA}_i = \{(R_{\min}^i, R_{\max}^i), (D_{\max}^i), (P_{\max}^i), (A_{\min}^i)\}$$

其中 $R_{\min/\max}$ 为最小/最大吞吐量，$D_{\max}$ 为最大延迟，$P_{\max}$ 为最大丢包率，$A_{\min}$ 为最低可用性。

**资源分配优化**：

$$\min \sum_{i=1}^{N} C_i \cdot x_i \quad \text{s.t.} \quad \begin{cases} \sum_{i} x_i \cdot r_i \leq R_{\text{total}} \\ R_i(x_i) \geq R_{\min}^i, \forall i \\ D_i(x_i) \leq D_{\max}^i, \forall i \end{cases}$$

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time

@dataclass
class SliceTemplate:
    """5G 网络切片 SLA 模板"""
    slice_id: str
    slice_type: str           # eMBB / URLLC / mMTC
    min_throughput_mbps: float
    max_throughput_mbps: float
    max_latency_ms: float
    max_packet_loss: float    # 丢包率 (0-1)
    min_availability: float   # 可用性 (0-1)
    priority: int             # 优先级 (1=最高)
    max_devices: int

@dataclass
class ResourceAllocation:
    """切片资源分配"""
    slice_id: str
    ran_rb_count: int         # RAN 资源块数
    upf_cpu_cores: float      # UPF CPU 核心数
    upf_memory_gb: float      # UPF 内存
    bandwidth_mbps: float     # 传输网带宽
    edge_nodes: List[str] = field(default_factory=list)

@dataclass
class KPIReading:
    """监控指标读数"""
    timestamp: float
    throughput_mbps: float
    latency_ms: float
    packet_loss: float
    active_devices: int

class SliceManager:
    """5G 网络切片管理器"""
    
    def __init__(self, total_rb: int = 273, total_bw_gbps: float = 10):
        self.total_rb = total_rb        # 总资源块 (100MHz, 30kHz SCS)
        self.total_bw = total_bw_gbps * 1000  # 转换为 Mbps
        self.slices: Dict[str, SliceTemplate] = {}
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.kpi_history: Dict[str, List[KPIReading]] = {}
    
    def create_slice(self, template: SliceTemplate, 
                     allocation: ResourceAllocation) -> bool:
        """创建网络切片"""
        # 检查资源是否足够
        used_rb = sum(a.ran_rb_count for a in self.allocations.values())
        used_bw = sum(a.bandwidth_mbps for a in self.allocations.values())
        
        if used_rb + allocation.ran_rb_count > self.total_rb:
            return False  # RAN 资源不足
        if used_bw + allocation.bandwidth_mbps > self.total_bw:
            return False  # 带宽不足
        
        self.slices[template.slice_id] = template
        self.allocations[template.slice_id] = allocation
        self.kpi_history[template.slice_id] = []
        return True
    
    def check_sla(self, slice_id: str, kpi: KPIReading) -> Dict[str, bool]:
        """检查 SLA 合规性"""
        template = self.slices.get(slice_id)
        if not template:
            return {}
        
        violations = {
            "throughput": kpi.throughput_mbps < template.min_throughput_mbps,
            "latency": kpi.latency_ms > template.max_latency_ms,
            "packet_loss": kpi.packet_loss > template.max_packet_loss,
            "devices": kpi.active_devices > template.max_devices,
        }
        
        # 记录 KPI
        self.kpi_history[slice_id].append(kpi)
        
        return violations
    
    def auto_heal(self, slice_id: str, violations: Dict[str, bool]) -> str:
        """自动修复 SLA 违规"""
        actions = []
        
        if violations.get("throughput"):
            # 扩容：增加资源块分配
            alloc = self.allocations[slice_id]
            alloc.ran_rb_count = min(alloc.ran_rb_count + 10, self.total_rb // 2)
            alloc.bandwidth_mbps *= 1.2
            actions.append("RB+10, BW+20%")
        
        if violations.get("latency"):
            # 迁移到边缘节点
            actions.append("Migrate UPF to MEC")
        
        if violations.get("packet_loss"):
            # 切换冗余路径
            actions.append("Activate backup path")
        
        return "; ".join(actions) if actions else "No action needed"
    
    def get_resource_utilization(self) -> Dict:
        """获取资源使用率"""
        used_rb = sum(a.ran_rb_count for a in self.allocations.values())
        used_bw = sum(a.bandwidth_mbps for a in self.allocations.values())
        
        return {
            "ran_rb_used": used_rb,
            "ran_rb_total": self.total_rb,
            "ran_utilization": used_rb / self.total_rb,
            "bandwidth_used_mbps": used_bw,
            "bandwidth_total_mbps": self.total_bw,
            "bandwidth_utilization": used_bw / self.total_bw,
            "active_slices": len(self.slices),
        }


# 模拟切片管理
sm = SliceManager(total_rb=273, total_bw_gbps=10)

# 创建 eMBB 切片
embb = SliceTemplate(
    slice_id="eMBB-001", slice_type="eMBB",
    min_throughput_mbps=100, max_throughput_mbps=1000,
    max_latency_ms=10, max_packet_loss=0.001,
    min_availability=0.999, priority=2, max_devices=1000
)
embb_alloc = ResourceAllocation(
    slice_id="eMBB-001", ran_rb_count=100,
    upf_cpu_cores=8, upf_memory_gb=16,
    bandwidth_mbps=3000, edge_nodes=["MEC-01"]
)

# 创建 URLLC 切片
urllc = SliceTemplate(
    slice_id="URLLC-001", slice_type="URLLC",
    min_throughput_mbps=10, max_throughput_mbps=100,
    max_latency_ms=1, max_packet_loss=0.00001,
    min_availability=0.99999, priority=1, max_devices=500
)
urllc_alloc = ResourceAllocation(
    slice_id="URLLC-001", ran_rb_count=50,
    upf_cpu_cores=4, upf_memory_gb=8,
    bandwidth_mbps=1000, edge_nodes=["MEC-01", "MEC-02"]
)

# 创建 mMTC 切片
mmtc = SliceTemplate(
    slice_id="mMTC-001", slice_type="mMTC",
    min_throughput_mbps=0.1, max_throughput_mbps=10,
    max_latency_ms=10000, max_packet_loss=0.01,
    min_availability=0.99, priority=3, max_devices=100000
)
mmtc_alloc = ResourceAllocation(
    slice_id="mMTC-001", ran_rb_count=80,
    upf_cpu_cores=2, upf_memory_gb=4,
    bandwidth_mbps=500
)

# 部署切片
for template, alloc in [(embb, embb_alloc), (urllc, urllc_alloc), (mmtc, mmtc_alloc)]:
    ok = sm.create_slice(template, alloc)
    print(f"切片 {template.slice_id}: {'创建成功' if ok else '资源不足'}")

# 检查资源使用率
util = sm.get_resource_utilization()
print(f"\n资源使用率:")
print(f"  RAN RB: {util['ran_rb_used']}/{util['ran_rb_total']} "
      f"({util['ran_utilization']:.1%})")
print(f"  带宽: {util['bandwidth_used_mbps']:.0f}/{util['bandwidth_total_mbps']:.0f} Mbps "
      f"({util['bandwidth_utilization']:.1%})")

# SLA 监控模拟
kpi = KPIReading(
    timestamp=time.time(),
    throughput_mbps=80,  # 低于 eMBB 最小要求 100 Mbps
    latency_ms=8,
    packet_loss=0.0005,
    active_devices=500
)
violations = sm.check_sla("eMBB-001", kpi)
print(f"\neMBB-001 SLA 检查: {violations}")
if any(violations.values()):
    action = sm.auto_heal("eMBB-001", violations)
    print(f"自愈动作: {action}")
```

<div data-component="5GSliceExplorer"></div>

---

## 31.7 物联网安全

### 31.7.1 轻量级加密算法

传统加密算法（AES-128）对资源受限设备仍可能过重。轻量级密码学提供更小面积和更低功耗的替代方案：

| 算法 | 类型 | 密钥长度 | 块大小 | 硬件面积 (GE) | 软件周期 |
|------|------|---------|--------|-------------|---------|
| **PRESENT** | SPN-Block | 80/128 bit | 64 bit | ~1570 GE | ~1000 |
| **SIMON** | Feistel | 64/96/128 bit | 32/64/128 bit | ~838 GE | ~500 |
| **SPECK** | ARX | 64/96/128 bit | 32/64/128 bit | ~843 GE | ~400 |
| **AES-128** | SPN-Block | 128 bit | 128 bit | ~2400 GE | ~3000 |

> **GE**（Gate Equivalent）：逻辑门等效数，1 GE ≈ NAND2 门的面积。GE 越小，硬件开销越低。

```
PRESENT 加密轮函数结构（31 轮）：
┌─────────────────────────────────────────────┐
│ 明文 (64 bit)                                │
│         │                                    │
│    ┌────▼────┐                               │
│    │  轮密钥  │  ← 密钥寄存器 (80 bit)        │
│    │  XOR    │    ┌──────────────────────┐   │
│    └────┬────┘    │ 密钥更新：            │   │
│         │         │ Rotate Left 61       │   │
│    ┌────▼────┐    │ S-box[79:76]         │   │
│    │  S-Box  │    │ XOR round_counter   │   │
│ 4-bit 替换盒│    └──────────────────────┘   │
│ (4→4 bit)  │                                │
│    └────┬────┘                               │
│         │                                    │
│    ┌────▼────┐                               │
│    │ P-Layer │  ← 置换层：bit[i] → bit[i×16 mod 63] │
│    └────┬────┘                               │
│         │                                    │
│    重复 31 轮                                 │
│         │                                    │
│    ┌────▼────┐                               │
│    │ 密文输出 │  (64 bit)                    │
│    └─────────┘                               │
└─────────────────────────────────────────────┘
```

### 31.7.2 设备身份认证与安全启动

```
IoT 安全启动流程：
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Boot ROM │→│ Stage 1  │→│ Stage 2  │→│ 应用层    │
│ (不可变)  │  │ Bootloader│ │ OS Kernel│  │ App      │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │             │
     ▼             ▼             ▼             ▼
 ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
 │ 验证   │   │ 验证   │   │ 验证   │   │ 验证   │
 │ Stage1 │   │ Stage2 │   │ App    │   │ 运行时 │
 │ 签名   │   │ 签名   │   │ 签名   │   │ 完整性 │
 └────────┘   └────────┘   └────────┘   └────────┘
 信任链：Root of Trust → Boot ROM → Bootloader → OS → App
```

**设备认证流程（基于 DTLS）**：

```
DTLS 握手（CoAP + DTLS）：
Client (IoT Device)            Server (Gateway)
    │                              │
    │── ClientHello ─────────────>│
    │   (PSK/RPK/Certificate)     │
    │                              │
    │<── ServerHello ─────────────│
    │<── Certificate (optional) ──│
    │<── ServerKeyExchange ───────│
    │<── ServerHelloDone ────────│
    │                              │
    │── ClientKeyExchange ───────>│
    │── ChangeCipherSpec ────────>│
    │── Finished ────────────────>│
    │                              │
    │<── ChangeCipherSpec ────────│
    │<── Finished ────────────────│
    │                              │
    │◄═══ 加密 CoAP 通信 ════════►│
```

### 31.7.3 OTA 固件更新

```python
import hashlib
import json
from dataclasses import dataclass
from typing import Optional

@dataclass
class OTAUpdate:
    """OTA 固件更新包"""
    firmware_version: str
    target_device: str
    firmware_url: str
    firmware_size: int
    sha256_hash: str
    signature: str       # 数字签名（Ed25519/ECDSA）
    delta_update: bool   # 是否为差分更新
    rollback_version: Optional[str] = None
    max_retry: int = 3

    def verify_integrity(self, downloaded_bytes: bytes) -> bool:
        """验证固件完整性"""
        computed_hash = hashlib.sha256(downloaded_bytes).hexdigest()
        return computed_hash == self.sha256_hash

    def to_manifest(self) -> str:
        """生成更新清单"""
        return json.dumps({
            "version": self.firmware_version,
            "target": self.target_device,
            "url": self.firmware_url,
            "size": self.firmware_size,
            "hash": self.sha256_hash,
            "delta": self.delta_update,
        }, indent=2)


# OTA 更新流程模拟
update = OTAUpdate(
    firmware_version="2.1.0",
    target_device="sensor_node_v2",
    firmware_url="https://ota.example.com/firmware/v2.1.0.bin",
    firmware_size=262144,  # 256 KB
    sha256_hash="a1b2c3d4e5f6...",
    signature="ed25519_signature_here",
    delta_update=True,
    rollback_version="2.0.0",
)

print(update.to_manifest())
```

---

## 31.8 应用案例

### 31.8.1 智能家居

```
智能家居网络架构：
┌──────────────────────────────────────────────────────────────┐
│                      云端平台                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ 设备管理  │  │ 场景引擎  │  │ AI 分析  │  │ 用户APP  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────┬────────────────────────────────────┘
                          │ MQTT/HTTPS
┌─────────────────────────▼────────────────────────────────────┐
│                   家庭网关 (Home Gateway)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ ZigBee   │  │ BLE Mesh │  │ WiFi     │  │ Z-Wave   │    │
│  │ 协调器   │  │ Proxy    │  │ AP       │  │ 控制器    │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
└───────┼──────────────┼──────────────┼──────────────┼──────────┘
        │              │              │              │
   ┌────┴────┐    ┌────┴────┐   ┌────┴────┐   ┌────┴────┐
   │ 温湿度  │    │ 门锁    │   │ 摄像头   │   │ 灯光    │
   │ 传感器  │    │         │   │          │   │ 控制器  │
   │(ZigBee) │    │(BLE)    │   │(WiFi)   │   │(Z-Wave) │
   └─────────┘    └─────────┘   └─────────┘   └─────────┘
```

### 31.8.2 工业物联网（IIoT）与 TSN

**TSN**（Time-Sensitive Networking）为工业物联网提供确定性以太网传输：

| TSN 标准 | 功能 | 关键参数 |
|----------|------|---------|
| **802.1AS** | 时间同步（gPTP） | 同步精度 < 1 μs |
| **802.1Qbv** | 时间感知调度 | 门控周期 < 125 μs |
| **802.1Qci** | 流量过滤与监管 | 每流过滤 |
| **802.1CB** | 帧复制与消除 | 零丢包冗余 |
| **802.1Qcc** | 流预留协议 (SRP) | 集中式/分布式配置 |

### 31.8.3 智慧城市

```
智慧城市数据流全景：
┌───────────┬───────────┬───────────┬───────────┐
│ 智能交通   │ 环境监测   │ 智能照明   │ 公共安全   │
│ 车流量检测 │ PM2.5/噪声 │ 路灯控制   │ 视频监控   │
│ 红绿灯优化 │ 气象站    │ 能耗优化   │ 应急响应   │
└─────┬─────┴─────┬─────┴─────┬─────┴─────┬─────┘
      │           │           │           │
  ┌───▼───┐   ┌───▼───┐   ┌───▼───┐   ┌───▼───┐
  │NB-IoT │   │LoRa   │   │NB-IoT │   │5G     │
  │       │   │       │   │       │   │       │
  └───┬───┘   └───┬───┘   └───┬───┘   └───┬───┘
      │           │           │           │
  ┌───▼───────────▼───────────▼───────────▼───┐
  │          城市级物联网平台 (City IoT Platform)│
  │  ┌────────┐  ┌────────┐  ┌────────┐       │
  │  │数据湖  │  │规则引擎 │  │数字孪生│       │
  │  └────────┘  └────────┘  └────────┘       │
  └───────────────────┬───────────────────────┘
                      │
  ┌───────────────────▼───────────────────────┐
  │         城市运营中心 (City Operations)       │
  │  交通优化 · 能源管理 · 应急指挥 · 市民服务    │
  └────────────────────────────────────────────┘
```

<div data-component="SmartCityDashboard"></div>

---

## 本章小结

| 主题 | 核心要点 |
|------|---------|
| **IoT 架构** | 感知-网络-平台-应用四层，约束驱动设计 |
| **MQTT** | 发布-订阅，QoS 0/1/2，Broker 路由 |
| **CoAP** | RESTful/UDP，CON/NON，Observe |
| **AMQP** | Exchange-Queue，可靠企业消息 |
| **LoRa/LoRaWAN** | CSS 扩频，Class A/B/C，km 级覆盖 |
| **NB-IoT** | 授权频段，覆盖增强 20dB |
| **6LoWPAN** | IPv6 压缩，分片重组 |
| **MEC/雾计算** | 边缘资源编排，任务 DAG 调度 |
| **5G 切片** | eMBB/URLLC/mMTC，SLA 保障 |
| **IoT 安全** | 轻量加密，安全启动，DTLS |

## 思考题

1. **MQTT vs CoAP**：在以下场景中应选择哪种协议？说明理由。
   - 每 10 分钟上报一次温湿度数据（电池供电传感器）
   - 远程控制智能灯泡（需要实时响应）
   - 工厂设备状态监控（数百台设备并发）

2. **LoRa SF 选择**：在一个面积 5 km² 的农业园区部署 LoRa 传感器网络，应选择 SF=7 还是 SF=10？考虑覆盖范围、数据速率、功耗和网络容量的权衡。

3. **边缘调度**：解释 HEFT 算法中"向上排序值"（Upward Rank）的物理含义，为什么高 Rank 的任务应该优先调度？

4. **5G 切片隔离**：对于自动驾驶场景，应创建哪种类型的切片？如果切片 SLA 延迟超标，切片管理器应采取哪些自愈措施？

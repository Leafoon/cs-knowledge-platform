# Chapter 8: 网络层 — IP 协议与编址

> **学习目标**：
> - 掌握 IPv4 数据报的完整格式及每个字段的含义
> - 理解 IP 分片与重组的机制（MF、DF、分片偏移）
> - 掌握 IP 编址体系（有类别编址、子网划分、CIDR）
> - 理解 NAT 的工作原理（类型、ALG、UPnP）及端口映射机制
> - 深入掌握 ARP 协议的工作流程（请求、应答、代理 ARP、免费 ARP）
> - 理解 DHCP 的 DORA 过程及地址池管理
> - 掌握 IP 转发算法的实现原理

---

## 8.1 IPv4 数据报格式

### 8.1.1 IPv4 首部结构

IPv4 数据报首部最小 20 字节，最大 60 字节（包含选项字段）：

```
  0                   1                   2                   3
  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |Version|  IHL  |    DSCP   |ECN|         Total Length          |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |         Identification        |Flags|      Fragment Offset    |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |  Time to Live |    Protocol   |         Header Checksum       |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                       Source Address                          |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                    Destination Address                        |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                    Options                    |    Padding    |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### 8.1.2 逐字段详解

| 字段 | 位数 | 说明 |
|------|------|------|
| **Version** | 4 | IP 版本号，IPv4 = 4 |
| **IHL** | 4 | 首部长度，以 4 字节为单位。最小 5（20 字节），最大 15（60 字节） |
| **DSCP** | 6 | 差分服务代码点（Differentiated Services Code Point），用于 QoS |
| **ECN** | 2 | 显式拥塞通知（Explicit Congestion Notification） |
| **Total Length** | 16 | 数据报总长度（首部+数据），最大 65535 字节 |
| **Identification** | 16 | 标识符，用于分片重组。同一数据报的所有分片共享同一标识符 |
| **Flags** | 3 | 标志位：保留(0)、DF(Don't Fragment)、MF(More Fragments) |
| **Fragment Offset** | 13 | 分片偏移，以 8 字节为单位。指示该分片在原始数据报中的位置 |
| **Time to Live** | 8 | 生存时间，每经过一个路由器减 1，到 0 时丢弃并发送 ICMP 超时消息 |
| **Protocol** | 8 | 上层协议号：1=ICMP, 6=TCP, 17=UDP, 89=OSPF |
| **Header Checksum** | 16 | 首部校验和，每经过一个路由器重新计算（因为 TTL 变化） |
| **Source Address** | 32 | 源 IP 地址 |
| **Destination Address** | 32 | 目的 IP 地址 |
| **Options** | 可变 | 选项字段，包括记录路由、时间戳、源路由等 |

**重要协议号**：

| 协议号 | 协议 |
|--------|------|
| 1 | ICMP |
| 2 | IGMP |
| 6 | TCP |
| 17 | UDP |
| 47 | GRE |
| 89 | OSPF |
| 132 | SCTP |

### 8.1.3 首部校验和计算

IPv4 首部校验和使用**反码求和**算法：

```python
def ipv4_checksum(header_bytes):
    """计算 IPv4 首部校验和"""
    # 首部长度必须是 2 字节的倍数
    assert len(header_bytes) % 2 == 0

    # 1. 将校验和字段置零
    header_bytes = bytearray(header_bytes)
    header_bytes[10] = 0
    header_bytes[11] = 0

    # 2. 按 16 位分组求和
    checksum = 0
    for i in range(0, len(header_bytes), 2):
        word = (header_bytes[i] << 8) + header_bytes[i + 1]
        checksum += word

    # 3. 将进位加回低 16 位
    while checksum >> 16:
        checksum = (checksum & 0xFFFF) + (checksum >> 16)

    # 4. 取反码
    checksum = ~checksum & 0xFFFF

    return checksum

# 验证校验和
def verify_checksum(header_bytes):
    """验证 IPv4 首部校验和"""
    checksum = 0
    for i in range(0, len(header_bytes), 2):
        word = (header_bytes[i] << 8) + header_bytes[i + 1]
        checksum += word
    while checksum >> 16:
        checksum = (checksum & 0xFFFF) + (checksum >> 16)
    return checksum == 0xFFFF
```

<div data-component="IPv4HeaderParser"></div>

---

## 8.2 IP 分片与重组

### 8.2.1 分片的必要性

不同网络技术有不同的**最大传输单元（MTU）**：

| 网络技术 | MTU |
|---------|-----|
| 以太网 | 1500 字节 |
| PPPoE | 1492 字节 |
| 802.11 (Wi-Fi) | 2304 字节 |
| FDDI | 4352 字节 |
| Token Ring | 17914 字节 |

当 IP 数据报大于链路的 MTU 时，路由器必须将其**分片（Fragmentation）**。分片在目的主机**重组（Reassembly）**。

### 8.2.2 分片过程

**关键字段**：
- **Identification**：同一数据报的所有分片共享同一标识符
- **DF（Don't Fragment）**：1 = 禁止分片（若需分片则丢弃并发 ICMP 错误）
- **MF（More Fragments）**：1 = 后面还有分片，0 = 最后一个分片
- **Fragment Offset**：该分片在原始数据报中的偏移（以 8 字节为单位）

**分片规则**：
1. 每个分片（除最后一个）的数据长度必须是 8 字节的倍数
2. 每个分片的总长度不能超过 MTU
3. 分片偏移以 8 字节为单位

**示例**：将 4000 字节的 IP 数据报（20 字节首部 + 3980 字节数据）通过 MTU=1500 的链路：

```
原始数据报: 总长=4000, ID=1234, DF=0, MF=0, Offset=0
数据: 3980 字节

分片 1: 总长=1500, ID=1234, DF=0, MF=1, Offset=0
        首部=20 字节, 数据=1480 字节 (字节 0-1479)

分片 2: 总长=1500, ID=1234, DF=0, MF=1, Offset=185
        首部=20 字节, 数据=1480 字节 (字节 1480-2959)
        偏移=185 (1480/8)

分片 3: 总长=1060, ID=1234, DF=0, MF=0, Offset=370
        首部=20 字节, 数据=1020 字节 (字节 2960-3979)
        偏移=370 (2960/8)
```

```python
def fragment_ip_packet(total_length, mtu, identification, data):
    """IP 数据报分片"""
    header_size = 20
    max_data_per_fragment = mtu - header_size
    # 分片数据长度必须是 8 字节的倍数
    max_data_per_fragment = (max_data_per_fragment // 8) * 8

    fragments = []
    offset = 0
    remaining = len(data)

    while remaining > 0:
        fragment_data_size = min(max_data_per_fragment, remaining)
        is_last = (remaining <= max_data_per_fragment)

        fragment = {
            'identification': identification,
            'DF': 0,
            'MF': 0 if is_last else 1,
            'fragment_offset': offset // 8,  # 以 8 字节为单位
            'total_length': header_size + fragment_data_size,
            'data': data[offset:offset + fragment_data_size],
        }

        fragments.append(fragment)
        offset += fragment_data_size
        remaining -= fragment_data_size

    return fragments
```

### 8.2.3 重组过程

目的主机收到分片后，通过 Identification 字段识别属于同一数据报的分片，然后根据 Fragment Offset 和 MF 重组：

```python
class IPReassembler:
    """IP 数据报重组器"""

    def __init__(self, timeout=30):
        self.fragments = {}  # identification -> list of fragments
        self.timeout = timeout

    def receive_fragment(self, packet):
        """收到一个分片"""
        ip_id = packet['identification']

        if ip_id not in self.fragments:
            self.fragments[ip_id] = {
                'fragments': [],
                'timestamp': time.time(),
                'total_length': None
            }

        self.fragments[ip_id]['fragments'].append(packet)

        if packet['MF'] == 0:
            # 最后一个分片到达，可以确定总长度
            self.fragments[ip_id]['total_length'] = \
                packet['fragment_offset'] * 8 + len(packet['data'])

        # 尝试重组
        return self._try_reassemble(ip_id)

    def _try_reassemble(self, ip_id):
        """尝试重组"""
        entry = self.fragments[ip_id]
        if entry['total_length'] is None:
            return None  # 还没收到最后一个分片

        # 按偏移排序
        frags = sorted(entry['fragments'], key=lambda f: f['fragment_offset'])

        # 检查是否连续覆盖
        expected_offset = 0
        result = bytearray()
        for frag in frags:
            if frag['fragment_offset'] * 8 != expected_offset:
                return None  # 有缺失的分片
            result.extend(frag['data'])
            expected_offset += len(frag['data'])

        if expected_offset == entry['total_length']:
            del self.fragments[ip_id]
            return bytes(result)

        return None
```

<div data-component="IPFragmentationDemo"></div>

---

## 8.3 IP 编址

### 8.3.1 有类别编址（Classful Addressing）

早期的 IP 编址将地址空间分为 5 类：

| 类别 | 前缀 | 网络号位数 | 主机号位数 | 地址范围 |
|------|------|-----------|-----------|---------|
| A 类 | 0 | 7 | 24 | 0.0.0.0 - 127.255.255.255 |
| B 类 | 10 | 14 | 16 | 128.0.0.0 - 191.255.255.255 |
| C 类 | 110 | 21 | 8 | 192.0.0.0 - 223.255.255.255 |
| D 类 | 1110 | — | — | 224.0.0.0 - 239.255.255.255（组播） |
| E 类 | 1111 | — | — | 240.0.0.0 - 255.255.255.255（保留） |

**问题**：
- A 类网络太大（1600 万主机），C 类太小（254 主机）
- 地址浪费严重
- 路由表膨胀（每个网络号一条路由）

### 8.3.2 子网划分（Subnetting）

在有类别编址的基础上，从主机号中借用若干位作为**子网号**：

```
原始 B 类地址:
┌─────────────────┬────────────────────────┐
│   网络号 (16位) │      主机号 (16位)       │
└─────────────────┴────────────────────────┘

子网划分后:
┌─────────────────┬───────────┬────────────┐
│   网络号 (16位) │子网号(8位)│主机号 (8位) │
└─────────────────┴───────────┴────────────┘
```

**子网掩码**：用于区分网络部分和主机部分。掩码中 1 对应网络+子网，0 对应主机。

```
子网掩码: 255.255.255.0 = 11111111.11111111.11111111.00000000
                     即 /24
```

### 8.3.3 CIDR（无类别域间路由）

CIDR（Classless Inter-Derm Routing）消除了有类别编址的限制，使用**可变长子网掩码（VLSM）**。

**表示法**：`IP地址/前缀长度`，如 `192.168.1.0/24`

**CIDR 的优势**：
1. 路由聚合（Route Aggregation）：将多个子网合并为一条路由
2. 减少路由表条目
3. 灵活分配地址空间

```
路由聚合示例:

192.168.0.0/24  ─┐
192.168.1.0/24   │
192.168.2.0/24   ├──► 聚合为 192.168.0.0/22
192.168.3.0/24  ─┘

/22 表示前 22 位是网络部分，后 10 位是主机部分
可容纳 2^10 = 1024 个地址（4 个 /24 子网）
```

```python
import ipaddress

def cidr_examples():
    """CIDR 编址示例"""

    # 创建网络
    network = ipaddress.IPv4Network('192.168.1.0/24')
    print(f"网络地址: {network.network_address}")
    print(f"广播地址: {network.broadcast_address}")
    print(f"子网掩码: {network.netmask}")
    print(f"主机数: {network.num_addresses - 2}")  # 减去网络地址和广播地址

    # 检查 IP 是否属于某网络
    ip = ipaddress.IPv4Address('192.168.1.100')
    print(f"{ip} 属于 {network}: {ip in network}")

    # 路由聚合
    networks = [
        ipaddress.IPv4Network('192.168.0.0/24'),
        ipaddress.IPv4Network('192.168.1.0/24'),
        ipaddress.IPv4Network('192.168.2.0/24'),
        ipaddress.IPv4Network('192.168.3.0/24'),
    ]
    aggregated = list(ipaddress.collapse_addresses(networks))
    print(f"聚合结果: {aggregated}")  # [IPv4Network('192.168.0.0/22')]

cidr_examples()
```

### 8.3.4 特殊 IP 地址

| 地址 | 用途 |
|------|------|
| 0.0.0.0/8 | 本网络 |
| 10.0.0.0/8 | A 类私有地址 |
| 127.0.0.0/8 | 环回地址 |
| 169.254.0.0/16 | 链路本地地址（APIPA） |
| 172.16.0.0/12 | B 类私有地址 |
| 192.168.0.0/16 | C 类私有地址 |
| 224.0.0.0/4 | 组播地址 |
| 255.255.255.255 | 受限广播地址 |

<div data-component="IPSubnetCalculator"></div>

---

## 8.4 NAT — 网络地址转换

### 8.4.1 NAT 的动机

IPv4 地址只有 32 位，全球地址空间约 43 亿，远不够用。NAT（Network Address Translation）允许多个私有地址共享一个公网地址。

### 8.4.2 NAT 的类型

**1. 静态 NAT（一对一）**：
- 一个私有地址永久映射到一个公网地址
- 用于需要从外网访问内网服务器的场景

**2. 动态 NAT（多对多）**：
- 从公网地址池中动态分配
- 私有地址数不能超过公网地址池大小

**3. NAPT/PAT（网络地址端口转换）**：
- 最常用，多个私有地址共享一个公网地址
- 通过不同的端口号区分不同的内部主机

### 8.4.3 NAPT 工作原理

```
内部主机 192.168.1.10:5000 ──► NAT ──► 203.0.113.1:40001 ──► 服务器
内部主机 192.168.1.11:5000 ──► NAT ──► 203.0.113.1:40002 ──► 服务器

NAT 转换表:
┌──────────────────┬──────────────────┬───────────┐
│ 内部地址:端口    │ 外部地址:端口    │ 协议      │
├──────────────────┼──────────────────┼───────────┤
│ 192.168.1.10:5000│ 203.0.113.1:40001│ TCP       │
│ 192.168.1.11:5000│ 203.0.113.1:40002│ TCP       │
└──────────────────┴──────────────────┴───────────┘
```

```python
class NAPT:
    """网络地址端口转换器"""

    def __init__(self, public_ip):
        self.public_ip = public_ip
        self.translation_table = {}  # (内部IP, 内部端口, 协议) -> (外部端口, 时间)
        self.reverse_table = {}      # (外部端口, 协议) -> (内部IP, 内部端口)
        self.port_counter = 40000    # 起始端口号

    def translate_outbound(self, src_ip, src_port, protocol):
        """出方向转换（内部→外部）"""
        key = (src_ip, src_port, protocol)

        if key in self.translation_table:
            ext_port = self.translation_table[key]
        else:
            ext_port = self.port_counter
            self.port_counter += 1
            self.translation_table[key] = ext_port
            self.reverse_table[(ext_port, protocol)] = (src_ip, src_port)

        return self.public_ip, ext_port

    def translate_inbound(self, dst_port, protocol):
        """入方向转换（外部→内部）"""
        key = (dst_port, protocol)
        if key in self.reverse_table:
            return self.reverse_table[key]
        return None  # 无映射，丢弃
```

### 8.4.4 ALG（应用层网关）

某些应用层协议（如 FTP、SIP）在数据载荷中嵌入了 IP 地址信息。NAT 需要 **ALG** 来修改这些嵌入的地址。

```
FTP PORT 命令示例:
客户端发送: PORT 192,168,1,10,19,137
（即 192.168.1.10:5001，19*256+137=5001）

NAT ALG 修改为: PORT 203,0,113,1,156,65
（即 203.0.113.1:40001）
```

### 8.4.5 UPnP（通用即插即用）

UPnP 允许内网主机自动配置 NAT 端口映射，无需管理员手动配置。

**UPnP NAT 穿越流程**：
1. 内网主机发现 NAT 网关（通过 SSDP 协议）
2. 使用 SOAP 协议发送端口映射请求
3. NAT 网关创建静态端口映射
4. 外部主机可通过公网地址:端口访问内网服务

<div data-component="NATSimulator"></div>

---

## 8.5 ARP 协议详解

### 8.5.1 ARP 请求与应答

**ARP 请求**（广播）：
- 目标 MAC: FF:FF:FF:FF:FF:FF（广播）
- 操作码: 1
- 发送方填入自己的 MAC 和 IP
- 目标 MAC 填 00:00:00:00:00:00（未知）

**ARP 应答**（单播）：
- 目标 MAC: 请求方的 MAC
- 操作码: 2
- 发送方填入自己的 MAC 和 IP

```
ARP 请求过程:

主机 A (IP: 192.168.1.10, MAC: AA:AA:AA:00:00:01)
主机 B (IP: 192.168.1.20, MAC: BB:BB:BB:00:00:02)

Step 1: A 要发送数据给 B，需要 B 的 MAC 地址
Step 2: A 发送 ARP 请求（广播）:
        "谁的 IP 是 192.168.1.20？请告诉 192.168.1.10 (AA:AA:AA:00:00:01)"
Step 3: 所有主机收到广播，只有 B 回应
Step 4: B 发送 ARP 应答（单播）:
        "192.168.1.20 的 MAC 是 BB:BB:BB:00:00:02"
Step 5: A 收到应答，更新 ARP 缓存
Step 6: A 使用 B 的 MAC 地址发送数据帧
```

### 8.5.2 代理 ARP 详解

```
场景: 主机 A (192.168.1.10/24) 访问主机 B (192.168.2.20/24)

Step 1: A 的子网掩码是 /24，判断 192.168.2.20 不在本地子网
Step 2: A 将数据发给默认网关（路由器 R）
Step 3: R 收到后，向 192.168.2.0/24 网段发送 ARP 请求
Step 4: B 回应 ARP 应答
Step 5: R 将数据转发给 B

如果启用了代理 ARP:
Step 1: A 不知道 B 不在同一子网（或没有默认网关）
Step 2: A 广播 ARP 请求: "192.168.2.20 的 MAC 是什么？"
Step 3: R 配置了代理 ARP，用自己的 MAC 回应
Step 4: A 以为 R 的 MAC 就是 B 的 MAC
Step 5: A 将数据发给 R，R 再转发给 B
```

### 8.5.3 免费 ARP（Gratuitous ARP）

免费 ARP 是主机主动发送的 ARP 应答（或请求），不需要回应。

**用途**：
1. **IP 冲突检测**：新设备上线时广播自己的 IP-MAC 映射
2. **更新 ARP 缓存**：设备更换网卡后通知其他设备
3. **VRRP/HSRP**：虚拟路由器切换时更新其他设备的 ARP 表

```python
def send_gratuitous_arp(ip_addr, mac_addr):
    """发送免费 ARP"""
    arp_packet = {
        'opcode': 2,  # ARP 应答
        'sender_mac': mac_addr,
        'sender_ip': ip_addr,
        'target_mac': 'ff:ff:ff:ff:ff:ff',  # 广播
        'target_ip': ip_addr,  # 目标 IP = 自己的 IP
    }
    # 构造以太网帧并发送
    ethernet_frame = {
        'dst_mac': 'ff:ff:ff:ff:ff:ff',  # 广播
        'src_mac': mac_addr,
        'type': 0x0806,  # ARP
        'payload': arp_packet,
    }
    send_frame(ethernet_frame)
```

<div data-component="ARPDemo"></div>

---

## 8.6 DHCP 协议

### 8.6.1 DHCP 概述

DHCP（Dynamic Host Configuration Protocol）自动为网络中的主机分配 IP 地址及其他网络配置参数。

**DHCP 分配的参数**：
- IP 地址
- 子网掩码
- 默认网关
- DNS 服务器地址
- 租约时间

### 8.6.2 DORA 过程

DHCP 使用 4 个步骤完成地址分配：**D**iscover → **O**ffer → **R**equest → **A**cknowledge

```
客户端                                              服务器
   │                                                   │
   │  ① DHCP Discover (广播)                          │
   │  "有没有 DHCP 服务器？我需要 IP 地址"             │
   │  源IP: 0.0.0.0  目的IP: 255.255.255.255          │
   │ ─────────────────────────────────────────────────►│
   │                                                   │
   │  ② DHCP Offer (广播)                             │
   │  "我可以给你 192.168.1.100"                       │
   │  源IP: 服务器IP  目的IP: 255.255.255.255          │
   │◄───────────────────────────────────────────────── │
   │                                                   │
   │  ③ DHCP Request (广播)                            │
   │  "我要使用 192.168.1.100"                         │
   │  源IP: 0.0.0.0  目的IP: 255.255.255.255          │
   │ ─────────────────────────────────────────────────►│
   │                                                   │
   │  ④ DHCP ACK (广播)                               │
   │  "确认！192.168.1.100 归你了，租期 24 小时"       │
   │  源IP: 服务器IP  目的IP: 255.255.255.255          │
   │◄───────────────────────────────────────────────── │
   │                                                   │
   │  客户端配置 IP 地址，开始通信                     │
```

**为什么都是广播？**
- Discover 和 Request：客户端还没有 IP 地址，也不知道服务器的 MAC
- Offer 和 ACK：服务器可能有多个客户端需要通知

### 8.6.3 DHCP 消息格式

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|     op (1)    |   htype (1)   |   hlen (1)    |   hops (1)    |
+---------------+---------------+---------------+---------------+
|                            xid (4)                            |
+---------------------------------------------------------------+
|           secs (2)            |           flags (2)           |
+---------------------------------------------------------------+
|                          ciaddr (4)                           |
+---------------------------------------------------------------+
|                          yiaddr (4)                           |
+---------------------------------------------------------------+
|                          siaddr (4)                           |
+---------------------------------------------------------------+
|                          giaddr (4)                           |
+---------------------------------------------------------------+
|                                                               |
|                          chaddr (16)                          |
|                                                               |
+---------------------------------------------------------------+
|                                                               |
|                          sname (64)                           |
|                                                               |
+---------------------------------------------------------------+
|                                                               |
|                          file (128)                           |
|                                                               |
+---------------------------------------------------------------+
|                                                               |
|                          options (可变)                       |
|                                                               |
+---------------------------------------------------------------+
```

| 字段 | 说明 |
|------|------|
| op | 1=请求, 2=应答 |
| xid | 事务 ID，用于匹配请求和应答 |
| ciaddr | 客户端 IP 地址（续租时使用） |
| yiaddr | "你的" IP 地址（服务器分配给客户端的） |
| siaddr | 服务器 IP 地址 |
| giaddr | 网关 IP 地址（跨子网 DHCP 中继） |
| chaddr | 客户端硬件地址（MAC 地址） |

<div data-component="DHCPSimulator"></div>

---

## 8.7 IP 转发算法

### 8.7.1 转发 vs 路由

| 特性 | 转发（Forwarding） | 路由（Routing） |
|------|-------------------|----------------|
| 层次 | 数据平面 | 控制平面 |
| 功能 | 每个数据报的下一跳选择 | 计算和维护路由表 |
| 速度 | 纳秒级（硬件） | 秒级（软件） |
| 频率 | 每个数据报 | 路由更新时 |

### 8.7.2 路由查找算法

**最长前缀匹配（Longest Prefix Match）**：当多个路由条目匹配时，选择前缀最长（最具体）的条目。

```
路由表:
┌────────────────┬──────────┬────────────┐
│ 目的网络       │ 下一跳   │ 接口       │
├────────────────┼──────────┼────────────┤
│ 0.0.0.0/0      │ 10.0.0.1 │ eth0       │ (默认路由)
│ 192.168.0.0/16 │ 10.0.0.2 │ eth1       │
│ 192.168.1.0/24 │ 10.0.0.3 │ eth2       │
│ 192.168.1.128/25│ 10.0.0.4│ eth3       │
└────────────────┴──────────┴────────────┘

目的 IP: 192.168.1.200

匹配的条目:
- 0.0.0.0/0        (0 位匹配)
- 192.168.0.0/16   (16 位匹配)
- 192.168.1.0/24   (24 位匹配)
- 192.168.1.128/25 (25 位匹配) ← 最长匹配！

下一跳: 10.0.0.4, 出接口: eth3
```

### 8.7.3 Trie 树路由查找

Trie（前缀树）是实现最长前缀匹配的经典数据结构：

```python
class TrieNode:
    def __init__(self):
        self.children = {}  # 0 或 1
        self.next_hop = None
        self.interface = None

class RouteTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, prefix, prefix_len, next_hop, interface):
        """插入路由条目"""
        node = self.root
        for i in range(prefix_len):
            bit = (prefix >> (31 - i)) & 1
            if bit not in node.children:
                node.children[bit] = TrieNode()
            node = node.children[bit]
        node.next_hop = next_hop
        node.interface = interface

    def lookup(self, ip_addr):
        """最长前缀匹配查找"""
        node = self.root
        best_match = None

        for i in range(32):
            # 记录当前节点的最佳匹配
            if node.next_hop is not None:
                best_match = (node.next_hop, node.interface)

            bit = (ip_addr >> (31 - i)) & 1
            if bit not in node.children:
                break
            node = node.children[bit]

        # 检查最后一个节点
        if node.next_hop is not None:
            best_match = (node.next_hop, node.interface)

        return best_match

# 使用示例
trie = RouteTrie()
# 插入: 192.168.1.0/24 -> 10.0.0.3, eth2
trie.insert(0xC0A80100, 24, '10.0.0.3', 'eth2')
# 插入默认路由: 0.0.0.0/0 -> 10.0.0.1, eth0
trie.insert(0x00000000, 0, '10.0.0.1', 'eth0')

# 查找 192.168.1.200
result = trie.lookup(0xC0A801C8)  # 192.168.1.200
print(f"下一跳: {result[0]}, 接口: {result[1]}")
```

### 8.7.4 路由查找性能

| 数据结构 | 查找时间 | 插入时间 | 空间 |
|---------|---------|---------|------|
| 线性表 | O(N) | O(1) | O(N) |
| 二叉 Trie | O(32) | O(32) | O(N×32) |
| 压缩 Trie | O(32) | O(32) | 更少 |
| TCAM | O(1) | O(1) | 固定 |

<div data-component="RouteLookupDemo"></div>

---

## 8.8 IP 数据报处理器的分片与重组逻辑

### 8.8.1 路由器分片处理流程

```
IP 数据报到达路由器:

┌─────────────────────────────────────────┐
│              入端口处理                  │
│  1. 接收帧                              │
│  2. 验证 FCS                            │
│  3. 解析 IP 首部                        │
│  4. 验证首部校验和                      │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│              路由查找                    │
│  1. 最长前缀匹配                        │
│  2. 确定下一跳和出接口                  │
│  3. TTL 减 1，重新计算校验和            │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│              分片决策                    │
│                                         │
│  数据报长度 > 出接口 MTU ?              │
│     │                                    │
│  ┌──┴──┐                                │
│  是    否                               │
│  │     │                                │
│  ▼     ▼                                │
│  DF=1?  直接转发                        │
│  │                                    │
│  ┌──┴──┐                               │
│  是    否                               │
│  │     │                                │
│  ▼     ▼                                │
│  丢弃  分片                             │
│  发送  按MTU切分                        │
│  ICMP  设置MF/Offset                   │
│  需要分片错误                           │
└─────────────────────────────────────────┘
```

### 8.8.2 重组器的实现

```c
/* IP 分片重组器 - 使用哈希表管理重组上下文 */

#define MAX_REASSEMBLY  128
#define REASSEMBLY_TIMEOUT 30  /* 秒 */

struct reassembly_context {
    uint16_t    identification;
    uint32_t    src_ip;
    uint32_t    dst_ip;
    uint8_t     protocol;
    time_t      timestamp;
    uint8_t     *buffer;           /* 重组缓冲区 */
    uint16_t    total_length;      /* 预期总长度 */
    uint32_t    received_bitmap;   /* 已接收分片位图 */
    uint16_t    max_offset;        /* 最大偏移 */
    uint8_t     complete;          /* 重组完成标志 */
};

struct reassembly_table {
    struct reassembly_context contexts[MAX_REASSEMBLY];
    int count;
    pthread_mutex_t lock;
};

struct ip_packet* reassemble(struct reassembly_table *table,
                              struct ip_packet *fragment) {
    uint32_t key = hash_key(fragment->identification,
                            fragment->src_ip,
                            fragment->dst_ip);

    struct reassembly_context *ctx = find_or_create(table, key);

    /* 复制分片数据到重组缓冲区 */
    uint16_t offset = fragment->fragment_offset * 8;
    memcpy(ctx->buffer + offset, fragment->data, fragment->data_length);

    /* 更新已接收分片位图 */
    uint16_t frag_index = offset / 8;
    ctx->received_bitmap |= (1 << (frag_index % 32));

    /* 检查是否是最后一个分片 */
    if (fragment->MF == 0) {
        ctx->total_length = offset + fragment->data_length;
        ctx->complete = 1;
    }

    /* 检查是否所有分片都已接收 */
    if (ctx->complete && all_fragments_received(ctx)) {
        struct ip_packet *result = create_packet(ctx);
        remove_context(table, ctx);
        return result;
    }

    return NULL;  /* 重组未完成 */
}
```

<div data-component="IPReassemblyDemo"></div>

---

## 8.9 路由查找引擎的硬件实现

### 8.9.1 TCAM（三态内容寻址存储器）

TCAM 是实现高速路由查找的硬件设备。与普通 CAM 不同，TCAM 支持三态匹配（0, 1, X）。

```
TCAM 查找过程:

输入: 目的 IP = 192.168.1.200 (0xC0A801C8)

┌───────────────────────────────────────────────────────────┐
│ TCAM 阵列                                                │
│ ┌──────┬─────────────────────────┬──────────┬──────────┐ │
│ │ 条目 │ 前缀（三态）            │ 结果     │ 有效位   │ │
│ ├──────┼─────────────────────────┼──────────┼──────────┤ │
│ │  0   │ 00000000...00000000 XXXX│ 下一跳A  │    1    │ │ (默认路由)
│ │  1   │ 11000000.10101000.00000000.XXXXXXXX│ 下一跳B│ │ (192.168.0.0/24)
│ │  2   │ 11000000.10101000.00000001.1XXXXXXX│ 下一跳C│ │ (192.168.1.128/25)
│ │  3   │ 11000000.10101000.00000001.XXXXXXXX│ 下一跳D│ │ (192.168.1.0/24)
│ └──────┴─────────────────────────┴──────────┴──────────┘ │
│                                                           │
│ 并行比较所有条目:                                         │
│ 条目 0: 匹配（0位前缀，总是匹配）                        │
│ 条目 1: 匹配（前16位相同）                               │
│ 条目 2: 匹配（前25位相同）                               │
│ 条目 3: 匹配（前24位相同）                               │
│                                                           │
│ 优先级编码器选择最高优先级（最长匹配）→ 条目 2           │
│ 输出: 下一跳C                                            │
└───────────────────────────────────────────────────────────┘
```

**TCAM 的关键特性**：
- 查找速度：一个时钟周期（O(1)）
- 规则数量：通常数千到数万条
- 优先级：条目按顺序排列，低位条目优先级更高
- 功耗：比 SRAM 高得多

### 8.9.2 路由表的数据结构

```c
/* 路由表条目 */
struct route_entry {
    uint32_t    prefix;         /* 网络前缀 */
    uint8_t     prefix_len;     /* 前缀长度 */
    uint32_t    next_hop;       /* 下一跳 IP */
    uint8_t     out_interface;  /* 出接口 */
    uint32_t    metric;         /* 路由度量 */
    uint8_t     protocol;       /* 路由协议来源 */
    time_t      last_update;    /* 最后更新时间 */
};

/* 路由表 */
#define MAX_ROUTES 1024

struct routing_table {
    struct route_entry entries[MAX_ROUTES];
    int count;
    pthread_rwlock_t lock;

    /* Trie 索引（用于快速查找） */
    struct trie_node *trie_root;
};

/* 路由查找 - 最长前缀匹配 */
struct route_entry* route_lookup(struct routing_table *table,
                                  uint32_t dest_ip) {
    struct route_entry *best = NULL;
    int best_len = -1;

    /* 使用 Trie 进行快速查找 */
    struct trie_node *node = table->trie_root;
    for (int i = 0; i < 32; i++) {
        if (node->route_entry != NULL) {
            if (node->prefix_len > best_len) {
                best = node->route_entry;
                best_len = node->prefix_len;
            }
        }
        int bit = (dest_ip >> (31 - i)) & 1;
        if (node->children[bit] == NULL) break;
        node = node->children[bit];
    }
    if (node->route_entry != NULL && node->prefix_len > best_len) {
        best = node->route_entry;
    }

    return best;
}
```

---

## 8.10 NAT 转换表的结构与端口映射机制

### 8.10.1 NAT 转换表的硬件实现

```c
/* NAT 转换表 - 使用哈希表实现 */

#define NAT_TABLE_SIZE  65536
#define NAT_PORT_RANGE  40000-65535

struct nat_entry {
    uint32_t    internal_ip;
    uint16_t    internal_port;
    uint32_t    external_ip;
    uint16_t    external_port;
    uint8_t     protocol;       /* TCP, UDP, ICMP */
    uint8_t     state;          /* NEW, ESTABLISHED, FIN_WAIT */
    time_t      timestamp;
    uint64_t    bytes_in;
    uint64_t    bytes_out;
    uint32_t    flags;
};

struct nat_table {
    /* 正向表: (内部IP, 内部端口, 协议) -> 外部端口 */
    struct nat_entry *forward[NAT_TABLE_SIZE];
    /* 反向表: (外部端口, 协议) -> (内部IP, 内部端口) */
    struct nat_entry *reverse[NAT_TABLE_SIZE];
    int count;
    uint16_t next_port;
    pthread_rwlock_t lock;
};

/* 出方向转换 */
struct nat_entry* nat_translate_out(struct nat_table *table,
                                     uint32_t src_ip,
                                     uint16_t src_port,
                                     uint8_t protocol) {
    /* 计算正向哈希 */
    uint32_t hash = nat_hash(src_ip, src_port, protocol);
    struct nat_entry *entry = table->forward[hash];

    /* 查找现有映射 */
    while (entry) {
        if (entry->internal_ip == src_ip &&
            entry->internal_port == src_port &&
            entry->protocol == protocol) {
            entry->timestamp = time(NULL);
            return entry;
        }
        entry = entry->next;
    }

    /* 创建新映射 */
    entry = malloc(sizeof(struct nat_entry));
    entry->internal_ip = src_ip;
    entry->internal_port = src_port;
    entry->external_ip = table->public_ip;
    entry->external_port = allocate_port(table);
    entry->protocol = protocol;
    entry->state = NAT_NEW;
    entry->timestamp = time(NULL);

    /* 插入正向和反向表 */
    insert_forward(table, entry);
    insert_reverse(table, entry);

    return entry;
}
```

<div data-component="NATTableDemo"></div>

---

## 8.11 DHCP 服务器的状态机与地址池管理

### 8.11.1 DHCP 服务器状态机

```python
class DHCPState:
    INIT = 'INIT'
    SELECTING = 'SELECTING'
    REQUESTING = 'REQUESTING'
    BOUND = 'BOUND'
    RENEWING = 'RENEWING'
    REBINDING = 'REBINDING'
    RELEASED = 'RELEASED'

class DHCPServerStateMachine:
    """DHCP 服务器状态机"""

    def __init__(self, address_pool):
        self.pool = address_pool
        self.bindings = {}  # MAC -> (IP, lease_end, state)

    def handle_discover(self, client_mac, requested_ip=None):
        """处理 DHCP Discover"""
        # 优先分配之前使用的 IP
        if requested_ip and self.pool.is_available(requested_ip):
            offered_ip = requested_ip
        elif client_mac in self.bindings:
            old_ip = self.bindings[client_mac][0]
            if self.pool.is_available(old_ip):
                offered_ip = old_ip
            else:
                offered_ip = self.pool.allocate()
        else:
            offered_ip = self.pool.allocate()

        if offered_ip is None:
            return None  # 地址池耗尽

        self.bindings[client_mac] = (offered_ip, None, DHCPState.SELECTING)
        return self._make_offer(client_mac, offered_ip)

    def handle_request(self, client_mac, requested_ip, server_id):
        """处理 DHCP Request"""
        if client_mac not in self.bindings:
            return self._make_nak(client_mac)

        offered_ip, _, state = self.bindings[client_mac]

        if requested_ip != offered_ip:
            # 客户端选择了其他服务器的 offer
            self.pool.release(offered_ip)
            del self.bindings[client_mac]
            return None

        # 分配成功
        lease_time = 86400  # 24 小时
        lease_end = time.time() + lease_time
        self.bindings[client_mac] = (offered_ip, lease_end, DHCPState.BOUND)

        return self._make_ack(client_mac, offered_ip, lease_time)

    def handle_release(self, client_mac):
        """处理 DHCP Release"""
        if client_mac in self.bindings:
            ip, _, _ = self.bindings[client_mac]
            self.pool.release(ip)
            del self.bindings[client_mac]
```

### 8.11.2 地址池管理

```python
class AddressPool:
    """DHCP 地址池管理"""

    def __init__(self, network, start_ip, end_ip, excluded=[]):
        self.network = network
        self.available = set()
        self.allocated = {}  # IP -> MAC
        self.reserved = {}   # MAC -> IP (静态绑定)

        # 初始化可用地址池
        start = int(ipaddress.IPv4Address(start_ip))
        end = int(ipaddress.IPv4Address(end_ip))
        for ip_int in range(start, end + 1):
            ip = ipaddress.IPv4Address(ip_int)
            if ip not in excluded:
                self.available.add(ip)

    def allocate(self, preferred_ip=None):
        """分配 IP 地址"""
        if preferred_ip and preferred_ip in self.available:
            self.available.remove(preferred_ip)
            return preferred_ip

        if self.available:
            ip = min(self.available)  # 分配最小的可用地址
            self.available.remove(ip)
            return ip

        return None  # 地址池耗尽

    def release(self, ip):
        """释放 IP 地址"""
        if ip in self.allocated:
            del self.allocated[ip]
        self.available.add(ip)

    def is_available(self, ip):
        """检查地址是否可用"""
        return ip in self.available
```

<div data-component="DHCPPoolDemo"></div>

---

## 8.12 本章总结

| 概念 | 核心要点 |
|------|---------|
| IPv4 首部 | 最小 20 字节，最大 60 字节，Version/IHL/TTL/Protocol 等关键字段 |
| 分片 | DF/MF 标志、分片偏移（8字节为单位），目的主机重组 |
| 编址 | A/B/C/D/E 类 → 子网划分 → CIDR |
| NAT | NAPT 最常用，通过端口号区分内部主机 |
| ARP | IP→MAC 映射，请求广播/应答单播，缓存+超时 |
| DHCP | DORA 四步过程，自动分配 IP/掩码/网关/DNS |
| 转发 | 最长前缀匹配，Trie/TCAM 实现 |
| TTL | 每跳减 1，防止环路，traceroute 原理 |

<div data-component="ChapterSummaryQuiz"></div>

---

## 练习题

**1. 计算题**：将 4000 字节的 IP 数据报（20 字节首部）通过 MTU=1500 的链路分片。列出每个分片的 Identification、MF、Offset 和 Total Length。

**2. 子网划分**：给定 192.168.10.0/24，划分为 4 个等大小的子网，写出每个子网的网络地址、广播地址和可用主机范围。

**3. NAT 分析**：某内网有 100 台主机，使用一个公网 IP 地址进行 NAPT。如果每台主机最多同时有 10 个 TCP 连接和 10 个 UDP 连接，端口号范围 40000-65535，是否有足够的端口号？

**4. 实现题**：实现一个简化版的 IP 转发表，支持 CIDR 路由条目的插入和最长前缀匹配查找。

**5. 分析题**：解释为什么 DHCP 使用广播而不是单播发送 Discover 和 Request 消息。

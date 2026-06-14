# Chapter 10: 网络层 — ICMP、DHCP 与网络管理

> **学习目标**：
> - 掌握 ICMP 消息格式（类型、代码）及差错与查询消息的区别
> - 理解 ICMPv6 相比 ICMPv4 的改进
> - 深入理解 ping 和 traceroute 的底层实现原理
> - 掌握 DHCP 协议的 DORA 详细流程及地址池管理
> - 理解 SNMP 协议的 MIB 树结构、SMI 和操作
> - 掌握网络故障排查的方法论和工具使用
> - 深入分析 ICMP 消息处理器、DHCP 地址分配引擎、SNMP 代理 MIB 树和诊断工具的内部实现

---

## 10.1 ICMP 协议

### 10.1.1 ICMP 概述

ICMP（Internet Control Message Protocol）是网络层的辅助协议，用于在 IP 网络中传递控制消息和错误报告。ICMP 消息封装在 IP 数据报中（协议号=1）。

**ICMP 的两大类消息**：
1. **差错消息（Error Messages）**：报告数据报传输中的错误
2. **查询消息（Query Messages）**：用于网络诊断和信息查询

### 10.1.2 ICMP 消息格式

```
  0                   1                   2                   3
  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |     Type      |     Code      |          Checksum             |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 |                     消息体（取决于类型和代码）                  |
 +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

| 字段 | 长度 | 说明 |
|------|------|------|
| Type | 1 字节 | 消息类型 |
| Code | 1 字节 | 消息子类型 |
| Checksum | 2 字节 | ICMP 校验和（覆盖整个 ICMP 消息） |
| 消息体 | 可变 | 取决于消息类型 |

### 10.1.3 ICMP 差错消息

| Type | Code | 名称 | 说明 |
|------|------|------|------|
| 3 | 0 | 网络不可达 | 路由器找不到到目标网络的路由 |
| 3 | 1 | 主机不可达 | 目标主机不可达（ARP 失败等） |
| 3 | 2 | 协议不可达 | 目标主机上层协议不可用 |
| 3 | 3 | 端口不可达 | 目标端口没有进程监听 |
| 3 | 4 | 需要分片但设置了 DF | 路由器需要分片但 DF=1 |
| 3 | 5 | 源路由失败 | 源路由选项指定的路由失败 |
| 3 | 13 | 通信被管理性禁止 | 被防火墙等策略阻止 |
| 4 | 0 | 源站抑制（已废弃） | 拥塞控制，要求发送方降低速率 |
| 5 | 0-3 | 重定向 | 路由器通知主机更好的下一跳 |
| 11 | 0 | TTL 超时（传输中） | TTL 减为 0 时发送 |
| 11 | 1 | TTL 超时（重组中） | 重组超时 |
| 12 | 0 | IP 首部错误 | 首部参数有问题 |
| 12 | 1 | 缺少必要选项 | 缺少必需的 IP 选项 |

**ICMP 差错消息的限制**：
- 不对 ICMP 差错消息发送 ICMP 差错消息（防止无限循环）
- 不对广播/组播数据报发送 ICMP 差错消息
- 不对分片（非第一个分片）发送 ICMP 差错消息

### 10.1.4 ICMP 差错消息的数据部分

ICMP 差错消息包含引发错误的原始 IP 数据报的首部和前 8 字节数据（包含 TCP/UDP 端口号）：

```
ICMP 差错消息内容:
┌─────────────────────────────────┐
│ ICMP 首部 (Type, Code, Checksum)│ 4 字节
├─────────────────────────────────┤
│ 未使用 / 特定信息               │ 4 字节
├─────────────────────────────────┤
│ 原始 IP 数据报首部              │ 20+ 字节
├─────────────────────────────────┤
│ 原始数据报数据的前 8 字节       │ 8 字节
│ (包含 TCP/UDP 源和目的端口)    │
└─────────────────────────────────┘
```

### 10.1.5 ICMP 查询消息

| Type | 名称 | 说明 |
|------|------|------|
| 8 | Echo Request | ping 请求 |
| 0 | Echo Reply | ping 应答 |
| 13 | Timestamp Request | 时间戳请求 |
| 14 | Timestamp Reply | 时间戳应答 |
| 9 | Router Advertisement | 路由器通告 |
| 10 | Router Solicitation | 路由器请求 |

```python
import struct
import socket

class ICMPMessage:
    """ICMP 消息构造与解析"""

    # ICMP 类型常量
    ECHO_REPLY = 0
    DEST_UNREACHABLE = 3
    SOURCE_QUENCH = 4
    REDIRECT = 5
    ECHO_REQUEST = 8
    TIME_EXCEEDED = 11

    @staticmethod
    def build_echo_request(identifier, sequence, data=b''):
        """构造 ICMP Echo Request"""
        msg_type = 8  # Echo Request
        code = 0
        checksum = 0

        # 构造 ICMP 消息（不包含校验和）
        icmp_packet = struct.pack('!BBHHH',
                                   msg_type, code, checksum,
                                   identifier, sequence)
        icmp_packet += data

        # 计算校验和
        checksum = ICMPMessage._calculate_checksum(icmp_packet)

        # 重新构造消息（包含正确的校验和）
        icmp_packet = struct.pack('!BBHHH',
                                   msg_type, code, checksum,
                                   identifier, sequence)
        icmp_packet += data

        return icmp_packet

    @staticmethod
    def parse_message(data):
        """解析 ICMP 消息"""
        msg_type, code, checksum, identifier, sequence = \
            struct.unpack('!BBHHH', data[:8])

        return {
            'type': msg_type,
            'code': code,
            'checksum': checksum,
            'identifier': identifier,
            'sequence': sequence,
            'data': data[8:],
        }

    @staticmethod
    def _calculate_checksum(data):
        """计算 ICMP 校验和"""
        if len(data) % 2:
            data += b'\x00'

        s = 0
        for i in range(0, len(data), 2):
            w = (data[i] << 8) + data[i + 1]
            s += w

        while s >> 16:
            s = (s & 0xFFFF) + (s >> 16)

        return ~s & 0xFFFF
```

<div data-component="ICMPMessageParser"></div>

---

## 10.2 ICMPv6

### 10.2.1 ICMPv6 概述

ICMPv6（RFC 4443）是 IPv6 的 ICMP 协议，整合了 ICMPv4、IGMP 和 ARP 的功能。

### 10.2.2 ICMPv6 消息类型

| Type | 名称 | 类别 |
|------|------|------|
| 1 | 目的不可达 | 差错 |
| 2 | 数据报过大 | 差错 |
| 3 | 超时 | 差错 |
| 4 | 参数问题 | 差错 |
| 128 | Echo Request | 查询 |
| 129 | Echo Reply | 查询 |
| 133 | 路由器请求（RS） | 查询 |
| 134 | 路由器通告（RA） | 查询 |
| 135 | 邻居请求（NS） | 查询 |
| 136 | 邻居通告（NA） | 查询 |
| 137 | 重定向 | 查询 |

### 10.2.3 邻居发现协议（NDP）

ICMPv6 的邻居发现协议替代了 IPv4 中的 ARP，用于地址解析和邻居检测：

```
IPv6 地址解析（替代 ARP）:

主机 A 想知道主机 B 的 MAC 地址:

Step 1: A 发送邻居请求（NS, Type=135）
        - 目标: B 的被请求节点组播地址
        - 内容: "谁的 IPv6 地址是 fe80::1? 我是 fe80::2"

Step 2: B 回复邻居通告（NA, Type=136）
        - 目标: A 的单播地址
        - 内容: "fe80::1 的 MAC 是 AA:BB:CC:DD:EE:FF"

被请求节点组播地址:
  FF02::1:FFxx:xxxx (后 24 位从目标 IPv6 地址复制)
  大大减少了不必要的处理
```

<div dataComponent="ICMPv6Demo"></div>

---

## 10.3 Ping 的实现原理

### 10.3.1 Ping 的工作流程

Ping 使用 ICMP Echo Request/Reply 测试主机可达性和往返时间：

```
ping 命令执行过程:

Step 1: 构造 ICMP Echo Request 消息
        Type=8, Code=0
        Identifier=进程ID, Sequence=递增序号
        Data=时间戳或填充数据

Step 2: 将 ICMP 消息封装在 IP 数据报中
        Source IP = 本机 IP
        Destination IP = 目标 IP
        TTL = 默认值（如 64 或 128）

Step 3: 发送数据报，启动计时器

Step 4: 等待 ICMP Echo Reply
        - 收到回复：计算 RTT = 当前时间 - 发送时间
        - 超时未收到：报告请求超时

Step 5: 重复 Step 3-4（通常 4 次）

Step 6: 统计结果
        - 发送/接收/丢失包数
        - 最小/平均/最大 RTT
```

### 10.3.2 Ping 的 C 实现

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/ip.h>
#include <netinet/ip_icmp.h>
#include <arpa/inet.h>

#define PACKET_SIZE     64
#define ICMP_DATA_LEN   (PACKET_SIZE - sizeof(struct icmphdr))

unsigned short checksum(void *b, int len) {
    unsigned short *buf = b;
    unsigned int sum = 0;
    unsigned short result;

    for (sum = 0; len > 1; len -= 2)
        sum += *buf++;
    if (len == 1)
        sum += *(unsigned char *)buf;
    sum = (sum >> 16) + (sum & 0xFFFF);
    sum += (sum >> 16);
    result = ~sum;
    return result;
}

int send_ping(int sockfd, struct sockaddr_in *dest_addr, int seq) {
    char packet[PACKET_SIZE];
    struct icmphdr *icmp = (struct icmphdr *)packet;

    memset(packet, 0, PACKET_SIZE);

    icmp->type = ICMP_ECHO;
    icmp->code = 0;
    icmp->un.echo.id = getpid();
    icmp->un.echo.sequence = seq;

    /* 填充数据（包含时间戳） */
    struct timeval *tv = (struct timeval *)(packet + sizeof(struct icmphdr));
    gettimeofday(tv, NULL);

    icmp->checksum = 0;
    icmp->checksum = checksum(packet, PACKET_SIZE);

    return sendto(sockfd, packet, PACKET_SIZE, 0,
                  (struct sockaddr *)dest_addr, sizeof(*dest_addr));
}

int receive_ping(int sockfd, int seq, double *rtt) {
    char buffer[1024];
    struct sockaddr_in from;
    socklen_t from_len = sizeof(from);

    struct timeval timeout = {2, 0};  /* 2 秒超时 */
    setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    int bytes = recvfrom(sockfd, buffer, sizeof(buffer), 0,
                         (struct sockaddr *)&from, &from_len);

    if (bytes <= 0) return -1;  /* 超时 */

    struct iphdr *ip = (struct iphdr *)buffer;
    struct icmphdr *icmp = (struct icmphdr *)(buffer + (ip->ihl * 4));

    if (icmp->type == ICMP_ECHOREPLY &&
        icmp->un.echo.id == getpid() &&
        icmp->un.echo.sequence == seq) {
        struct timeval *send_time = (struct timeval *)
            (buffer + (ip->ihl * 4) + sizeof(struct icmphdr));
        struct timeval recv_time;
        gettimeofday(&recv_time, NULL);

        *rtt = (recv_time.tv_sec - send_time->tv_sec) * 1000.0 +
               (recv_time.tv_usec - send_time->tv_usec) / 1000.0;
        return 0;
    }

    return -1;  /* 不匹配的包 */
}
```

<div dataComponent="PingDemo"></div>

---

## 10.4 Traceroute 的实现原理

### 10.4.1 Traceroute 的基本思想

Traceroute 利用 IP 数据报的 **TTL（Time to Live）** 字段来发现从源到目的的路径上的每一跳路由器。

**原理**：
1. 发送 TTL=1 的数据报 → 第一个路由器将 TTL 减为 0，丢弃并发送 ICMP 超时消息
2. 发送 TTL=2 的数据报 → 第二个路由器发送 ICMP 超时消息
3. 递增 TTL，直到到达目的地（收到 ICMP 端口不可达或 Echo Reply）

```
traceroute 工作过程:

发送方                                        路由器1   路由器2   目的主机
   │                                              │         │         │
   │  UDP (TTL=1, port=33434)                     │         │         │
   │─────────────────────────────────────────────►│         │         │
   │                                              │         │         │
   │  ICMP Time Exceeded (from Router1)           │         │         │
   │◄─────────────────────────────────────────────│         │         │
   │                                              │         │         │
   │  UDP (TTL=2, port=33434)                     │         │         │
   │──────────────────────────────────────────────┼────────►│         │
   │                                              │         │         │
   │  ICMP Time Exceeded (from Router2)           │         │         │
   │◄─────────────────────────────────────────────┼─────────│         │
   │                                              │         │         │
   │  UDP (TTL=3, port=33434)                     │         │         │
   │──────────────────────────────────────────────┼─────────┼────────►│
   │                                              │         │         │
   │  ICMP Port Unreachable (from Destination)    │         │         │
   │◄─────────────────────────────────────────────┼─────────┼─────────│
```

### 10.4.2 Unix/Linux traceroute 实现

```python
import socket
import struct
import time

def traceroute(dest_host, max_hops=30, timeout=2):
    """简单的 traceroute 实现"""
    dest_ip = socket.gethostbyname(dest_host)
    print(f"traceroute to {dest_host} ({dest_ip}), {max_hops} hops max")

    for ttl in range(1, max_hops + 1):
        # 创建 UDP 发送套接字和 ICMP 接收套接字
        send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM,
                                   socket.IPPROTO_UDP)
        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_RAW,
                                   socket.IPPROTO_ICMP)

        send_sock.setsockopt(socket.IPPROTO_UDP, socket.IP_TTL, ttl)
        recv_sock.settimeout(timeout)

        port = 33434 + ttl
        send_time = time.time()

        # 发送 UDP 数据报
        send_sock.sendto(b'', (dest_ip, port))

        try:
            # 等待 ICMP 响应
            data, addr = recv_sock.recvfrom(1024)
            recv_time = time.time()
            rtt = (recv_time - send_time) * 1000

            # 解析 ICMP 消息
            icmp_type = data[20]
            icmp_code = data[21]

            if icmp_type == 11 and icmp_code == 0:
                # ICMP Time Exceeded - 中间路由器
                print(f"  {ttl:2d}  {addr[0]:15s}  {rtt:.3f} ms")
            elif icmp_type == 3 and icmp_code == 3:
                # ICMP Port Unreachable - 到达目的地
                print(f"  {ttl:2d}  {addr[0]:15s}  {rtt:.3f} ms (到达目的地)")
                break
            else:
                print(f"  {ttl:2d}  {addr[0]:15s}  ICMP type={icmp_type} code={icmp_code}")

        except socket.timeout:
            print(f"  {ttl:2d}  * * *")

        finally:
            send_sock.close()
            recv_sock.close()

# 使用示例
traceroute("www.example.com")
```

### 10.4.3 Windows tracert 实现

Windows 的 tracert 使用 **ICMP Echo Request**（而非 UDP），原理相同：

```
Unix traceroute vs Windows tracert:

| 特性 | Unix traceroute | Windows tracert |
|------|----------------|-----------------|
| 发送协议 | UDP (端口 33434+) | ICMP Echo Request |
| 探测包类型 | UDP 数据报 | ICMP Type=8 |
| 到达目的判断 | ICMP Port Unreachable | ICMP Echo Reply |
| 防火墙友好性 | 较好（高端口） | 较差（常被过滤） |
```

<div dataComponent="TracerouteDemo"></div>

---

## 10.5 DHCP 协议详解

### 10.5.1 DHCP 消息类型

| 消息类型 | 方向 | 说明 |
|---------|------|------|
| DHCPDISCOVER | 客户端→服务器 | 客户端广播寻找 DHCP 服务器 |
| DHCPOFFER | 服务器→客户端 | 服务器提供 IP 地址 |
| DHCPREQUEST | 客户端→服务器 | 客户端请求特定 IP 地址 |
| DHCPACK | 服务器→客户端 | 服务器确认分配 |
| DHCPNAK | 服务器→客户端 | 服务器拒绝请求 |
| DHCPDECLINE | 客户端→服务器 | 客户端拒绝提供的地址（IP冲突） |
| DHCPRELEASE | 客户端→服务器 | 客户端释放 IP 地址 |
| DHCPINFORM | 客户端→服务器 | 客户端已有 IP，请求其他配置参数 |

### 10.5.2 DORA 详细流程

```
详细 DORA 流程:

客户端 (MAC: AA:BB:CC:DD:EE:01)           DHCP 服务器 (192.168.1.1)
        │                                        │
Step 1: │ DHCPDISCOVER (广播)                    │
        │ ┌────────────────────────────────┐     │
        │ │ op=1 (BOOTREQUEST)             │     │
        │ │ htype=1 (Ethernet)             │     │
        │ │ hlen=6 (MAC 长度)              │     │
        │ │ xid=0x12345678 (事务ID)        │     │
        │ │ ciaddr=0.0.0.0                 │     │
        │ │ yiaddr=0.0.0.0                 │     │
        │ │ chaddr=AA:BB:CC:DD:EE:01       │     │
        │ │ Options:                       │     │
        │ │   DHCP Message Type=DISCOVER   │     │
        │ │   Parameter Request List       │     │
        │ │   Client Identifier            │     │
        │ └────────────────────────────────┘     │
        │────────────────────────────────────────►│
        │                                        │
Step 2: │ DHCPOFFER (广播)                      │
        │ ┌────────────────────────────────┐     │
        │ │ op=2 (BOOTREPLY)               │     │
        │ │ xid=0x12345678 (匹配事务ID)    │     │
        │ │ yiaddr=192.168.1.100           │     │
        │ │ siaddr=192.168.1.1             │     │
        │ │ chaddr=AA:BB:CC:DD:EE:01       │     │
        │ │ Options:                       │     │
        │ │   DHCP Message Type=OFFER      │     │
        │ │   Subnet Mask=255.255.255.0    │     │
        │ │   Router=192.168.1.1           │     │
        │ │   DNS Server=8.8.8.8           │     │
        │ │   IP Address Lease Time=86400  │     │
        │ │   DHCP Server Identifier       │     │
        │ └────────────────────────────────┘     │
        │◄────────────────────────────────────────│
        │                                        │
Step 3: │ DHCPREQUEST (广播)                     │
        │ ┌────────────────────────────────┐     │
        │ │ xid=0x12345678                 │     │
        │ │ ciaddr=0.0.0.0                 │     │
        │ │ yiaddr=0.0.0.0                 │     │
        │ │ Options:                       │     │
        │ │   DHCP Message Type=REQUEST    │     │
        │ │   Requested IP=192.168.1.100   │     │
        │ │   DHCP Server Identifier       │     │
        │ └────────────────────────────────┘     │
        │────────────────────────────────────────►│
        │                                        │
Step 4: │ DHCPACK (广播)                        │
        │ ┌────────────────────────────────┐     │
        │ │ xid=0x12345678                 │     │
        │ │ yiaddr=192.168.1.100           │     │
        │ │ Options:                       │     │
        │ │   DHCP Message Type=ACK        │     │
        │ │   所有配置参数                  │     │
        │ └────────────────────────────────┘     │
        │◄────────────────────────────────────────│
        │                                        │
        │ 客户端配置 IP 地址                      │
        │ 192.168.1.100/24                       │
        │ 网关: 192.168.1.1                       │
        │ DNS: 8.8.8.8                           │
```

<div dataComponent="DHCPDetailedDemo"></div>

---

## 10.6 SNMP 协议

### 10.6.1 SNMP 概述

SNMP（Simple Network Management Protocol）是网络管理的标准协议，用于监控和管理网络设备。

**SNMP 体系结构**：
- **管理站（Manager）**：运行管理软件，查询和配置被管理设备
- **代理（Agent）**：运行在被管理设备上，响应管理站的查询
- **MIB（Management Information Base）**：管理信息库，定义可管理的对象
- **SMI（Structure of Management Information）**：管理信息结构，定义对象的编码规则

### 10.6.2 MIB 树结构

MIB 采用树状层次结构，每个节点用对象标识符（OID）标识：

```
MIB 树:

iso(1)
 ├── org(3)
 │   ├── dod(6)
 │   │   └── internet(1)
 │   │       ├── mgmt(2)
 │   │       │   └── mib-2(1)
 │   │       │       ├── system(1)
 │   │       │       │   ├── sysDescr(1)      -- 系统描述
 │   │       │       │   ├── sysObjectID(2)   -- 系统对象ID
 │   │       │       │   ├── sysUpTime(3)     -- 系统运行时间
 │   │       │       │   ├── sysContact(4)    -- 联系人
 │   │       │       │   ├── sysName(5)       -- 系统名称
 │   │       │       │   └── sysLocation(6)   -- 位置
 │   │       │       ├── interfaces(2)
 │   │       │       │   ├── ifNumber(1)      -- 接口数量
 │   │       │       │   └── ifTable(2)
 │   │       │       │       └── ifEntry(1)
 │   │       │       │           ├── ifIndex(1)
 │   │       │       │           ├── ifDescr(2)
 │   │       │       │           ├── ifType(3)
 │   │       │       │           ├── ifSpeed(5)
 │   │       │       │           ├── ifInOctets(10)
 │   │       │       │           └── ifOutOctets(16)
 │   │       │       ├── ip(4)
 │   │       │       │   └── ipForwarding(1)
 │   │       │       ├── icmp(5)
 │   │       │       ├── tcp(6)
 │   │       │       └── udp(7)
 │   │       ├── private(4)
 │   │       │   └── enterprises(1)
 │   │       │       └── 厂商私有 MIB
 │   │       └── experimental(3)
 │   └── ...
 └── ...
```

### 10.6.3 SNMP 操作

| 操作 | 说明 | 方向 |
|------|------|------|
| GET | 查询一个或多个 MIB 对象 | Manager → Agent |
| GET-NEXT | 查询下一个 MIB 对象（遍历表） | Manager → Agent |
| GET-BULK | 批量查询（SNMPv2c+） | Manager → Agent |
| SET | 设置 MIB 对象的值 | Manager → Agent |
| TRAP | 代理主动发送事件通知 | Agent → Manager |
| INFORM | 需要确认的 TRAP（SNMPv2c+） | Agent → Manager |

### 10.6.4 SNMP 版本

| 版本 | 安全特性 | 认证方式 |
|------|---------|---------|
| SNMPv1 | 无加密 | 社区名（Community String） |
| SNMPv2c | 无加密 | 社区名 |
| SNMPv3 | 加密+认证 | USM（用户安全模型） |

```python
class MIBNode:
    """MIB 树节点"""

    def __init__(self, oid, name, access='read-only', value_type='integer'):
        self.oid = oid           # 对象标识符 (如 "1.3.6.1.2.1.1.1")
        self.name = name         # 对象名称 (如 "sysDescr")
        self.access = access     # 访问权限
        self.value_type = value_type  # 值类型
        self.value = None        # 当前值
        self.children = {}       # 子节点

class MIBTree:
    """MIB 树结构"""

    def __init__(self):
        self.root = MIBNode('', 'root')
        self.oid_index = {}  # OID -> MIBNode 快速索引

    def add_node(self, oid, name, access='read-only', value_type='integer'):
        """添加 MIB 节点"""
        parts = oid.split('.')
        node = self.root
        current_oid = ''

        for part in parts:
            current_oid = current_oid + '.' + part if current_oid else part
            if part not in node.children:
                child = MIBNode(current_oid, '', access, value_type)
                node.children[part] = child
            node = node.children[part]

        node.name = name
        node.access = access
        node.value_type = value_type
        self.oid_index[oid] = node

    def get(self, oid):
        """GET 操作"""
        if oid in self.oid_index:
            node = self.oid_index[oid]
            return (node.oid, node.value)
        return None

    def get_next(self, oid):
        """GET-NEXT 操作"""
        sorted_oids = sorted(self.oid_index.keys())
        for sorted_oid in sorted_oids:
            if sorted_oid > oid:
                node = self.oid_index[sorted_oid]
                return (node.oid, node.value)
        return None

    def set(self, oid, value):
        """SET 操作"""
        if oid in self.oid_index:
            node = self.oid_index[oid]
            if 'write' in node.access:
                node.value = value
                return True
        return False

# 创建 MIB 树示例
mib = MIBTree()
mib.add_node('1.3.6.1.2.1.1.1', 'sysDescr', 'read-only', 'octet_string')
mib.add_node('1.3.6.1.2.1.1.3', 'sysUpTime', 'read-only', 'timeticks')
mib.add_node('1.3.6.1.2.1.1.5', 'sysName', 'read-write', 'octet_string')

# 模拟查询
mib.oid_index['1.3.6.1.2.1.1.1'].value = 'Linux Server 5.4.0'
mib.oid_index['1.3.6.1.2.1.1.3'].value = 12345600  # 以百分之一秒为单位

result = mib.get('1.3.6.1.2.1.1.1')
print(f"sysDescr = {result[1]}")
```

<div dataComponent="MIBTreeDemo"></div>

---

## 10.7 网络故障排查方法论

### 10.7.1 OSI 分层排查法

```
网络故障排查 - 自底向上法:

物理层 (L1):
  ├── 检查线缆是否连接
  ├── 检查接口 LED 指示灯
  ├── 检查接口状态 (up/down)
  └── 工具: show interfaces, ethtool

数据链路层 (L2):
  ├── 检查 MAC 地址表
  ├── 检查 VLAN 配置
  ├── 检查 STP 状态
  ├── 检查 ARP 表
  └── 工具: show mac address-table, show arp

网络层 (L3):
  ├── 检查 IP 地址配置
  ├── 检查路由表
  ├── 测试 ping
  ├── 追踪 traceroute
  └── 工具: show ip route, ping, traceroute

传输层 (L4):
  ├── 检查端口是否开放
  ├── 检查防火墙规则
  ├── 检查 TCP 连接状态
  └── 工具: netstat, ss, telnet, nmap

应用层 (L7):
  ├── 检查服务是否运行
  ├── 检查 DNS 解析
  ├── 检查应用日志
  └── 工具: nslookup, dig, curl, wget
```

### 10.7.2 常用诊断命令

```bash
# 1. 连通性测试
ping -c 4 8.8.8.8
ping6 -c 4 2001:4860:4860::8888

# 2. 路由追踪
traceroute 8.8.8.8
traceroute -I 8.8.8.8  # 使用 ICMP
tracert 8.8.8.8         # Windows

# 3. DNS 查询
nslookup www.example.com
dig www.example.com A +short
dig @8.8.8.8 www.example.com

# 4. 端口扫描
nmap -sT -p 80,443 192.168.1.1
telnet 192.168.1.1 80
nc -zv 192.168.1.1 80

# 5. 网络接口信息
ip addr show
ip route show
arp -a

# 6. 连接状态
ss -tuln          # 监听的端口
ss -tp            # TCP 连接
netstat -tuln     # 同上（旧版）
```

### 10.7.3 故障排查决策树

```
网络故障排查决策树:

问题: 无法访问目标服务器

1. ping 目标 IP
   ├── 成功 → 跳到 4
   └── 失败 → 继续 2

2. ping 默认网关
   ├── 成功 → 跳到 3
   └── 失败 → 检查物理连接、接口状态、ARP

3. traceroute 到目标
   ├── 在某个路由器超时 → 检查该路由器配置
   └── 到达目标后超时 → 检查目标防火墙

4. telnet 目标端口
   ├── 成功 → 应用层问题
   └── 失败 → 检查服务是否运行、防火墙规则

5. 检查 DNS
   ├── nslookup 解析失败 → 检查 DNS 配置和服务器
   └── 解析成功 → 检查应用配置
```

<div dataComponent="NetworkDiagnostics"></div>

---

## 10.8 核心组件深入分析

### 10.8.1 ICMP 消息生成器与处理器的数据通路

```c
/* ICMP 消息处理器 */

enum icmp_action {
    ICMP_SEND_REPLY,      /* 发送应答（如 Echo Reply） */
    ICMP_NOTIFY_SENDER,   /* 通知发送方（如 Source Quench） */
    ICMP_FORWARD,         /* 转发给上层协议 */
    ICMP_DROP,            /* 丢弃 */
};

struct icmp_handler {
    /* 消息类型处理函数表 */
    void (*handlers[256])(struct icmp_message *msg,
                          struct ip_packet *original);

    /* 统计计数器 */
    uint64_t sent_error;
    uint64_t sent_query;
    uint64_t recv_error;
    uint64_t recv_query;
    uint64_t rate_limited;  /* 被速率限制的消息数 */
};

/* ICMP 速率限制 - 防止 ICMP 风暴 */
struct icmp_rate_limiter {
    uint64_t tokens;
    uint64_t max_tokens;
    uint64_t refill_rate;   /* 每秒补充的令牌数 */
    time_t last_refill;
    pthread_mutex_t lock;
};

int icmp_rate_check(struct icmp_rate_limiter *limiter) {
    time_t now = time(NULL);
    int elapsed = now - limiter->last_refill;

    /* 补充令牌 */
    limiter->tokens += elapsed * limiter->refill_rate;
    if (limiter->tokens > limiter->max_tokens)
        limiter->tokens = limiter->max_tokens;
    limiter->last_refill = now;

    /* 消耗令牌 */
    if (limiter->tokens > 0) {
        limiter->tokens--;
        return 1;  /* 允许发送 */
    }
    return 0;  /* 速率限制 */
}

/* ICMP 消息处理入口 */
void icmp_process(struct icmp_handler *handler,
                   struct ip_packet *ip_pkt,
                   struct icmp_message *icmp_msg) {
    /* 1. 验证校验和 */
    if (icmp_checksum(icmp_msg) != 0) {
        handler->stats.bad_checksum++;
        return;
    }

    /* 2. 根据类型分发处理 */
    switch (icmp_msg->type) {
    case ICMP_ECHO_REQUEST:
        /* Echo Request - 生成 Echo Reply */
        if (icmp_rate_check(&handler->rate_limiter)) {
            icmp_send_echo_reply(ip_pkt, icmp_msg);
            handler->stats.sent_query++;
        } else {
            handler->stats.rate_limited++;
        }
        break;

    case ICMP_DEST_UNREACHABLE:
        /* 差错消息 - 通知传输层 */
        handler->stats.recv_error++;
        transport_notify_error(icmp_msg);
        break;

    case ICMP_TIME_EXCEEDED:
        /* TTL 超时 - traceroute 使用 */
        handler->stats.recv_error++;
        transport_notify_error(icmp_msg);
        break;

    default:
        /* 未知类型 - 丢弃 */
        handler->stats.unknown_type++;
        break;
    }
}
```

### 10.8.2 DHCP 服务器的地址分配引擎

```c
/* DHCP 地址分配引擎 */

struct dhcp_binding {
    uint32_t    ip_addr;
    uint8_t     mac_addr[6];
    uint8_t     state;          /* FREE, OFFERED, ASSIGNED, EXPIRED */
    time_t      lease_start;
    uint32_t    lease_time;     /* 租约时间（秒） */
    uint8_t     flags;          /* STATIC, DYNAMIC */
    char        hostname[64];
};

struct dhcp_pool {
    uint32_t    network;        /* 网络地址 */
    uint32_t    netmask;        /* 子网掩码 */
    uint32_t    gateway;        /* 默认网关 */
    uint32_t    dns_server;     /* DNS 服务器 */
    uint32_t    range_start;    /* 地址范围起始 */
    uint32_t    range_end;      /* 地址范围结束 */
    uint32_t    default_lease;  /* 默认租约时间 */
    uint32_t    max_lease;      /* 最大租约时间 */

    struct dhcp_binding *bindings;   /* 绑定数组 */
    int         binding_count;

    /* 统计 */
    int         total_addresses;
    int         active_bindings;
    int         expired_bindings;
    int         free_addresses;
};

/* 地址分配算法 */
uint32_t dhcp_allocate_address(struct dhcp_pool *pool,
                                uint8_t *client_mac,
                                uint32_t requested_ip) {
    /* 1. 检查客户端是否已有绑定 */
    for (int i = 0; i < pool->binding_count; i++) {
        if (memcmp(pool->bindings[i].mac_addr, client_mac, 6) == 0) {
            if (pool->bindings[i].state == ASSIGNED) {
                /* 续租 */
                pool->bindings[i].lease_start = time(NULL);
                return pool->bindings[i].ip_addr;
            }
        }
    }

    /* 2. 检查请求的 IP 是否可用 */
    if (requested_ip != 0) {
        if (is_in_range(pool, requested_ip) &&
            is_address_free(pool, requested_ip)) {
            return create_binding(pool, client_mac, requested_ip);
        }
    }

    /* 3. 分配最小可用地址 */
    for (uint32_t ip = pool->range_start; ip <= pool->range_end; ip++) {
        if (is_address_free(pool, ip)) {
            return create_binding(pool, client_mac, ip);
        }
    }

    /* 4. 检查过期的绑定 */
    for (int i = 0; i < pool->binding_count; i++) {
        if (pool->bindings[i].state == EXPIRED) {
            pool->bindings[i].state = FREE;
            if (is_in_range(pool, pool->bindings[i].ip_addr)) {
                return create_binding(pool, client_mac,
                                      pool->bindings[i].ip_addr);
            }
        }
    }

    return 0;  /* 地址池耗尽 */
}

/* 租约检查定时器 */
void dhcp_lease_check(struct dhcp_pool *pool) {
    time_t now = time(NULL);

    for (int i = 0; i < pool->binding_count; i++) {
        if (pool->bindings[i].state == ASSIGNED) {
            time_t elapsed = now - pool->bindings[i].lease_start;

            if (elapsed > pool->bindings[i].lease_time) {
                /* 租约过期 */
                pool->bindings[i].state = EXPIRED;
                pool->active_bindings--;
                pool->expired_bindings++;
            }
        }
    }
}
```

### 10.8.3 SNMP 代理的 MIB 树结构与查询处理器

```c
/* SNMP 代理 MIB 树实现 */

#define MAX_OID_LEN     128

struct mib_object {
    uint32_t    oid[MAX_OID_LEN];
    int         oid_len;
    char        name[64];
    uint8_t     type;           /* INTEGER, OCTET_STRING, OID, ... */
    uint8_t     access;         /* READ_ONLY, READ_WRITE, ... */
    union {
        int32_t     integer;
        uint8_t     string[256];
        uint32_t    oid_val[MAX_OID_LEN];
        uint64_t    counter64;
        uint32_t    timeticks;
    } value;
    void        (*get_callback)(struct mib_object *obj);
    int         (*set_callback)(struct mib_object *obj, void *value);
};

struct mib_tree {
    struct mib_object *objects;
    int object_count;
    int object_capacity;
    pthread_rwlock_t lock;
};

/* SNMP GET 处理 */
int snmp_handle_get(struct mib_tree *tree,
                     uint32_t *oid, int oid_len,
                     struct snmp_variable *result) {
    pthread_rwlock_rdlock(&tree->lock);

    for (int i = 0; i < tree->object_count; i++) {
        if (oid_equal(tree->objects[i].oid, tree->objects[i].oid_len,
                      oid, oid_len)) {
            /* 找到对象 */
            if (tree->objects[i].get_callback) {
                tree->objects[i].get_callback(&tree->objects[i]);
            }
            result->oid = tree->objects[i].oid;
            result->oid_len = tree->objects[i].oid_len;
            result->type = tree->objects[i].type;
            result->value = tree->objects[i].value;
            pthread_rwlock_unlock(&tree->lock);
            return SNMP_ERR_NO_ERROR;
        }
    }

    pthread_rwlock_unlock(&tree->lock);
    return SNMP_ERR_NO_SUCH_NAME;
}

/* SNMP GET-NEXT 处理 */
int snmp_handle_get_next(struct mib_tree *tree,
                          uint32_t *oid, int oid_len,
                          struct snmp_variable *result) {
    pthread_rwlock_rdlock(&tree->lock);

    struct mib_object *best = NULL;

    for (int i = 0; i < tree->object_count; i++) {
        if (oid_compare(tree->objects[i].oid, tree->objects[i].oid_len,
                        oid, oid_len) > 0) {
            if (best == NULL ||
                oid_compare(tree->objects[i].oid, tree->objects[i].oid_len,
                            best->oid, best->oid_len) < 0) {
                best = &tree->objects[i];
            }
        }
    }

    if (best) {
        if (best->get_callback) {
            best->get_callback(best);
        }
        result->oid = best->oid;
        result->oid_len = best->oid_len;
        result->type = best->type;
        result->value = best->value;
        pthread_rwlock_unlock(&tree->lock);
        return SNMP_ERR_NO_ERROR;
    }

    pthread_rwlock_unlock(&tree->lock);
    return SNMP_ERR_END_OF_MIB;
}
```

### 10.8.4 网络诊断工具的底层实现原理

**ping 的底层实现**：
- 使用原始套接字（Raw Socket）构造 ICMP 包
- 设置 IP_HDRINCL 选项以自定义 IP 首部
- 使用 setsockopt 设置 TTL、超时等参数
- 通过 recvfrom 接收 ICMP 应答

**traceroute 的底层实现**：
- Linux 版本使用 UDP 发送到高端口（33434+），设置不同 TTL
- Windows 版本使用 ICMP Echo Request，设置不同 TTL
- 中间路由器返回 ICMP Time Exceeded
- 目的主机返回 ICMP Port Unreachable（UDP）或 Echo Reply（ICMP）

**tcpdump/libpcap 的底层实现**：
- 使用 BPF（Berkeley Packet Filter）捕获网络数据包
- BPF 在内核中过滤数据包，减少用户态拷贝
- 使用 mmap 提高捕获效率

```python
# 使用原始套接字实现简单的端口扫描
import socket
import struct

def tcp_syn_scan(target_ip, port, timeout=1):
    """TCP SYN 扫描（半开扫描）"""
    try:
        # 创建原始套接字
        sock = socket.socket(socket.AF_INET, socket.SOCK_RAW,
                              socket.IPPROTO_TCP)
        sock.settimeout(timeout)

        # 构造 TCP SYN 包（简化版）
        # 实际实现需要手动构造 IP 和 TCP 首部

        # 发送 SYN
        sock.connect((target_ip, port))

        # 等待 SYN-ACK 或 RST
        response = sock.recv(1024)

        # 解析响应
        tcp_flags = response[13]  # TCP 标志位
        if tcp_flags & 0x12 == 0x12:  # SYN-ACK
            return 'open'
        elif tcp_flags & 0x04:  # RST
            return 'closed'

    except socket.timeout:
        return 'filtered'  # 被防火墙过滤
    except Exception as e:
        return f'error: {e}'
    finally:
        sock.close()
```

<div dataComponent="NetworkToolsDemo"></div>

---

## 10.9 SNMP 安全与性能

### 10.9.1 SNMP 安全最佳实践

| 措施 | 说明 |
|------|------|
| 使用 SNMPv3 | 支持加密和认证 |
| 修改默认社区名 | 避免使用 public/private |
| 限制访问源 IP | ACL 限制 SNMP 访问 |
| 只读权限 | 除非必要，使用 read-only |
| 禁用不必要的 MIB | 减少攻击面 |
| 监控 SNMP 访问 | 记录异常查询 |

### 10.9.2 SNMP 性能优化

```python
class SNMPBulkRetriever:
    """SNMP 批量查询优化"""

    def get_bulk(self, agent, start_oid, max_repetitions=10):
        """使用 GET-BULK 批量获取 MIB 对象"""
        results = []
        current_oid = start_oid

        while True:
            # 发送 GET-BULK 请求
            response = snmp_get_bulk(agent, current_oid, max_repetitions)

            if not response or response.error:
                break

            for var_bind in response.var_binds:
                # 检查是否超出目标子树
                if not var_bind.oid.startswith(start_oid):
                    return results
                results.append(var_bind)
                current_oid = var_bind.oid

            if len(response.var_binds) < max_repetitions:
                break

        return results
```

---

## 10.10 本章总结

| 概念 | 核心要点 |
|------|---------|
| ICMP | 差错消息+查询消息，封装在 IP 中（协议号=1） |
| ICMP 差错消息 | 网络/主机/协议/端口不可达、重定向、TTL超时 |
| ICMP 查询消息 | Echo Request/Reply（ping）、Timestamp |
| ICMPv6 | 整合 ARP 功能，邻居发现协议（NDP） |
| ping | ICMP Echo，测量 RTT 和可达性 |
| traceroute | 递增 TTL，ICMP 超时消息发现路径 |
| DHCP | DORA 四步，地址池管理，租约机制 |
| SNMP | MIB 树结构，GET/SET/TRAP 操作 |
| 故障排查 | 分层排查法，ping→traceroute→telnet→DNS |

<div dataComponent="ChapterSummaryQuiz"></div>

---

## 练习题

**1. 分析题**：解释为什么 ICMP 差错消息中要包含原始数据报的首部和前 8 字节数据。

**2. 实现题**：用 Python 实现一个简单的 ping 工具，支持指定目标主机、TTL 和超时时间。

**3. 对比题**：比较 Unix traceroute 和 Windows tracert 的实现差异，分析各自的优缺点。

**4. 设计题**：设计一个 DHCP 服务器的地址池管理模块，支持地址分配、续租、过期回收和静态绑定。

**5. SNMP 题**：解释 MIB 树中 OID 的含义，描述 GET-NEXT 操作如何遍历 MIB 表。

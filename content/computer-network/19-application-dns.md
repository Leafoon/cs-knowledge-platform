# Chapter 19: 应用层 — DNS 域名系统

> **学习目标**：
> - 理解 DNS 的设计动机与域名层次结构
> - 掌握 DNS 服务器层次结构：根服务器、TLD 服务器、权威服务器、递归解析器
> - 区分递归查询与迭代查询的工作流程
> - 掌握 DNS 记录类型：A/AAAA/CNAME/MX/NS/TXT/SOA/PTR/SRV
> - 理解 DNS 消息格式：头部、问题区、回答区、权威区、附加区
> - 掌握 DNS 缓存机制与 TTL 管理
> - 理解 DNSSEC 链式信任模型与 DNS over HTTPS (DoH) 的安全设计

---

## 19.1 DNS 概述

### 19.1.1 为什么需要 DNS

互联网上的设备通过 IP 地址（如 142.250.80.46）通信，但人类更容易记住有意义的名字（如 www.google.com）。DNS（Domain Name System，域名系统）就是互联网的"电话簿"，将人类可读的域名转换为机器可识别的 IP 地址。

**DNS 的历史**：
- 1983 年由 Paul Mockapetris 设计（RFC 882/883）
- 1987 年更新为 RFC 1034/1035，至今仍是核心标准
- 最初解决的是 hosts.txt 文件的扩展性问题

**没有 DNS 的世界**：

```
要访问 Google，你需要记住:
  142.250.80.46    ← Google 主页
  142.250.190.46   ← YouTube
  157.240.1.35     ← Facebook
  104.16.132.229   ← Cloudflare

有了 DNS:
  www.google.com → 自动解析为 142.250.80.46
  www.youtube.com → 自动解析为 142.250.190.46
```

### 19.1.2 DNS 的设计目标

DNS 系统的设计目标包括：

1. **分布式数据库**：没有单一的中央服务器，数据分布在全球数百万台服务器上
2. **层次化命名**：域名采用树形结构，便于管理和分配
3. **可扩展性**：支持数十亿条记录，每秒处理数万亿次查询
4. **高可用性**：通过冗余和缓存实现 99.999% 的可用性
5. **协议无关**：可以解析任意类型的资源记录，不仅仅是 IP 地址

---

## 19.2 域名层次结构

### 19.2.1 域名空间

DNS 域名空间是一个**树形结构**，从根节点开始，逐层向下分支：

```
                        . (根)
                       / | \
                      /  |  \
                    com edu org net gov ...
                   / | \
                  /  |  \
               google apple amazon ...
              / | \
             /  |  \
           www mail maps ...
```

**域名的读法**：从叶节点到根节点，用点号分隔

```
www.google.com.
│   │      │  │
│   │      │  └── 根域 (root)
│   │      └───── 顶级域 (TLD): com
│   └────────── 二级域 (SLD): google
└────────────── 子域 (subdomain): www
```

### 19.2.2 顶级域名分类

| 类别 | 示例 | 管理机构 |
|------|------|---------|
| 通用顶级域 (gTLD) | .com, .org, .net, .info | ICANN 注册局 |
| 国家/地区顶级域 (ccTLD) | .cn, .uk, .jp, .de | 各国注册局 |
| 基础设施顶级域 | .arpa | IANA |
| 新通用顶级域 (new gTLD) | .app, .dev, .cloud, .xyz | ICANN |

### 19.2.3 域名注册流程

```
用户想注册 example.com:
  1. 在域名注册商（如 Namecheap、GoDaddy）查询域名可用性
  2. 支付年费，获得域名使用权
  3. 注册商将域名信息提交到 .com TLD 注册局（Verisign）
  4. 配置 DNS 记录（通过权威 DNS 服务商如 Cloudflare、AWS Route 53）
  5. TLD 服务器添加 NS 记录，指向你的权威 DNS 服务器

注意：你并不"拥有"域名，只是获得使用权
```

<div data-component="DomainTreeExplorer"></div>

---

## 19.3 DNS 服务器层次结构

### 19.3.1 服务器类型概览

```
┌─────────────────────────────────────────────────────────────────┐
│                     DNS 服务器层次结构                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  递归解析器 (Recursive Resolver)                                │
│  ├── 由 ISP 或公共 DNS 提供（8.8.8.8, 1.1.1.1）               │
│  ├── 代表客户端完成完整的域名解析                               │
│  └── 维护缓存，减少重复查询                                     │
│                                                                 │
│  根域名服务器 (Root Name Servers)                               │
│  ├── 全球 13 组根服务器（A-M），实际有数百个实例               │
│  ├── 知道所有 TLD 服务器的地址                                 │
│  └── 由 ICANN 授权的 12 个独立组织运营                         │
│                                                                 │
│  TLD 服务器 (TLD Name Servers)                                 │
│  ├── 管理特定顶级域的所有域名                                   │
│  ├── 知道各二级域的权威服务器地址                               │
│  └── 例如：.com TLD 服务器知道 google.com 的 NS 记录           │
│                                                                 │
│  权威域名服务器 (Authoritative Name Servers)                    │
│  ├── 持有域名的最终 DNS 记录                                   │
│  ├── 对查询给出权威回答                                        │
│  └── 通常由 DNS 托管服务商运营                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 19.3.2 根域名服务器

全球根域名服务器是 DNS 系统的起点：

```
根服务器列表（13 组，标识符 A-M）:
  A: Verisign                    198.41.0.4
  B: USC-ISI                     199.9.14.201
  C: Cogent                      192.33.4.12
  D: University of Maryland      199.7.91.13
  C: NASA                        192.203.230.10
  F: ISC                         192.5.5.241
  G: US DoD                      192.112.36.4
  H: US Army                     198.97.190.53
  I: Netnod                      192.36.148.17
  J: Verisign                    192.58.128.30
  K: RIPE NCC                    193.0.14.129
  L: ICANN                       199.7.83.42
  M: WIDE Project                202.12.27.33

每个"服务器"实际上是分布式集群:
  - 全球有超过 1500 个实例
  - 使用 Anycast 路由，用户连接到最近的实例
  - 单个根服务器每秒可处理数百万查询
```

### 19.3.3 递归解析器

递归解析器是用户 DNS 查询的第一站，它负责完成完整的解析过程：

```python
class RecursiveResolver:
    """DNS 递归解析器"""
    def __init__(self):
        self.cache = DNSCache(max_size=100000)
        self.root_servers = ROOT_SERVER_IPS

    def resolve(self, domain: str, record_type: str) -> list:
        # 1. 检查缓存
        cached = self.cache.get(domain, record_type)
        if cached and not cached.is_expired():
            return cached.records

        # 2. 递归解析
        result = self._recursive_resolve(domain, record_type, depth=0)

        # 3. 缓存结果
        if result:
            self.cache.put(domain, record_type, result)

        return result

    def _recursive_resolve(self, domain: str, rtype: str, depth: int) -> list:
        if depth > 15:  # 防止无限递归
            raise DNSResolutionError("Max recursion depth exceeded")

        # 从根服务器开始
        nameservers = self.root_servers

        while True:
            # 向当前层的名称服务器发送查询
            for ns in nameservers:
                response = self._send_query(ns, domain, rtype)

                if response.is_authoritative():
                    # 找到权威回答
                    return response.answers

                if response.answers:
                    # 可能是 CNAME，需要继续解析
                    cname = self._extract_cname(response.answers)
                    if cname:
                        return self._recursive_resolve(cname, rtype, depth + 1)

                if response.authorities:
                    # 获取下一层的名称服务器
                    nameservers = self._get_ns_addresses(response)
                    break

                if response.additional:
                    # 从附加区获取 NS 的 IP
                    nameservers = self._extract_additional_ips(response.additional)
                    break

            else:
                raise DNSResolutionError("No nameserver responded")

    def _send_query(self, server: str, domain: str, rtype: str) -> DNSResponse:
        """向指定 DNS 服务器发送查询"""
        query = DNSQuery(domain, rtype)
        # 设置超时和重试
        for attempt in range(3):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(2.0)  # 2 秒超时
                sock.sendto(query.to_bytes(), (server, 53))
                data, _ = sock.recvfrom(4096)
                return DNSResponse.from_bytes(data)
            except socket.timeout:
                continue
            finally:
                sock.close()
        raise DNSResolutionError(f"Failed to query {server}")
```

<div data-component="DNSResolverFlow"></div>

---

## 19.4 DNS 查询类型

### 19.4.1 递归查询 vs 迭代查询

**递归查询（Recursive Query）**：

```
客户端 → 递归解析器: "请帮我解析 www.example.com"
递归解析器 → 根服务器: "www.example.com 的 IP 是什么？"
根服务器 → 递归解析器: "去问 .com TLD 服务器"
递归解析器 → TLD 服务器: "www.example.com 的 IP 是什么？"
TLD 服务器 → 递归解析器: "去问 example.com 权威服务器"
递归解析器 → 权威服务器: "www.example.com 的 A 记录是什么？"
权威服务器 → 递归解析器: "93.184.216.34"
递归解析器 → 客户端: "www.example.com 的 IP 是 93.184.216.34"
```

**迭代查询（Iterative Query）**：

```
客户端 → 递归解析器: "请解析 www.example.com"
递归解析器 → 根服务器: "www.example.com 的 A 记录？"
根服务器 → 递归解析器: "我不知道，但 .com TLD 在 192.5.6.30"
递归解析器 → 192.5.6.30: "www.example.com 的 A 记录？"
TLD 服务器 → 递归解析器: "我不知道，但 example.com 权威在 93.184.216.1"
递归解析器 → 93.184.216.1: "www.example.com 的 A 记录？"
权威服务器 → 递归解析器: "93.184.216.34"
递归解析器 → 客户端: "结果是 93.184.216.34"
```

### 19.4.2 反向查询

反向查询将 IP 地址解析为域名：

```
正向查询: www.google.com → 142.250.80.46
反向查询: 142.250.80.46 → www.google.com

实现方式:
  IP 地址 142.250.80.46
  转换为: 46.80.250.142.in-addr.arpa
  查询 PTR 记录

  对于 IPv6:
  2001:db8::1
  转换为: 1.0.0.0...0.0.0.0.8.b.d.0.1.0.0.2.ip6.arpa
```

---

## 19.5 DNS 记录类型

### 19.5.1 常见记录类型

| 记录类型 | 用途 | 示例 |
|---------|------|------|
| A | IPv4 地址 | example.com → 93.184.216.34 |
| AAAA | IPv6 地址 | example.com → 2606:2800:220:1:... |
| CNAME | 别名 | www.example.com → example.com |
| MX | 邮件服务器 | example.com → mail.example.com (优先级 10) |
| NS | 权威名称服务器 | example.com → ns1.example.com |
| TXT | 文本记录 | SPF、DKIM、DMARC 等 |
| SOA | 起始授权 | 区域的管理信息 |
| PTR | 反向解析 | IP → 域名 |
| SRV | 服务发现 | _http._tcp.example.com → host:port |

### 19.5.2 记录格式详解

**A 记录**：

```
example.com.  300  IN  A  93.184.216.34
│            │   │  │  │
│            │   │  │  └── 记录值（IPv4 地址）
│            │   │  └── 记录类型
│            │   └── 类别（IN = Internet）
│            └── TTL（生存时间，秒）
└── 域名
```

**MX 记录**：

```
example.com.  3600  IN  MX  10 mail1.example.com.
example.com.  3600  IN  MX  20 mail2.example.com.

优先级：数字越小，优先级越高
  - 优先尝试 mail1（优先级 10）
  - 如果 mail1 不可用，使用 mail2（优先级 20）
```

**CNAME 记录**：

```
www.example.com.  300  IN  CNAME  example.com.

工作原理:
  查询 www.example.com → 发现 CNAME → 重定向到 example.com
  然后查询 example.com 的 A 记录

限制:
  - CNAME 不能与其他记录共存（同一域名同一类型）
  - 根域名（example.com）不能设置 CNAME
  - CNAME 链不宜过长（影响性能）
```

**SOA 记录**：

```
example.com.  86400  IN  SOA  ns1.example.com. admin.example.com. (
    2024010101  ; 序列号（用于区域传送）
    3600        ; 刷新间隔（从服务器多久检查更新）
    900         ; 重试间隔（刷新失败后多久重试）
    1209600     ; 过期时间（从服务器数据过期时间）
    86400       ; 否定缓存 TTL（NXDOMAIN 的缓存时间）
)
```

**TXT 记录**：

```
example.com.  300  IN  TXT  "v=spf1 include:_spf.google.com ~all"

常见用途:
  - SPF (Sender Policy Framework): 防止邮件伪造
  - DKIM (DomainKeys Identified Mail): 邮件签名验证
  - DMARC (Domain-based Message Authentication): 邮件认证策略
  - 域名验证（ACME、SSL 证书验证）
  - Google/Microsoft 等服务的域名所有权验证
```

<div data-component="DNSRecordExplorer"></div>

### 19.5.3 SRV 记录详解

SRV 记录用于服务发现，是现代微服务架构的重要组成部分：

```
_service._proto.name. TTL IN SRV priority weight port target

示例:
_sip._tcp.example.com. 3600 IN SRV 10 60 5060 sip1.example.com.
_sip._tcp.example.com. 3600 IN SRV 10 40 5060 sip2.example.com.

字段含义:
  _sip:      服务名称（SIP 协议）
  _tcp:      协议（TCP 或 UDP）
  priority:  优先级（越小越优先）
  weight:    权重（相同优先级时的负载分配比例）
  port:      服务端口
  target:    服务主机名

负载分配:
  sip1 权重 60，sip2 权重 40
  → 60% 流量到 sip1，40% 流量到 sip2
```

---

## 19.6 DNS 消息格式

### 19.6.1 消息结构

DNS 消息由五部分组成：

```
+---------------------+
|      Header         |  ← 12 字节固定头部
+---------------------+
|     Question        |  ← 查询的问题
+---------------------+
|      Answer         |  ← 回答记录
+---------------------+
|     Authority       |  ← 权威记录
+---------------------+
|     Additional      |  ← 附加记录
+---------------------+
```

### 19.6.2 头部格式

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|          Transaction ID       |        Flags                 |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  QDCOUNT (问题数)             |  ANCOUNT (回答数)             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  NSCOUNT (权威数)             |  ARCOUNT (附加数)             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

Flags 字段:
  QR:     0=查询, 1=响应
  Opcode: 0=标准, 1=反向, 2=服务器状态
  AA:     权威回答标志
  TC:     截断标志（响应超过 512 字节）
  RD:     期望递归（Recursion Desired）
  RA:     支持递归（Recursion Available）
  RCODE:  响应码（0=无错误, 3=NXDOMAIN, 2=服务器失败）
```

### 19.6.3 DNS 报文解析实现

```python
import struct
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DNSHeader:
    id: int
    flags: int
    qdcount: int  # 问题数
    ancount: int  # 回答数
    nscount: int  # 权威数
    arcount: int  # 附加数

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> 'DNSHeader':
        id_, flags, qd, an, ns, ar = struct.unpack_from('!HHHHHH', data, offset)
        return cls(id=id_, flags=flags, qdcount=qd, ancount=an,
                   nscount=ns, arcount=ar)

    def is_response(self) -> bool:
        return bool(self.flags & 0x8000)

    def is_authoritative(self) -> bool:
        return bool(self.flags & 0x0400)

    def is_truncated(self) -> bool:
        return bool(self.flags & 0x0200)

    def rcode(self) -> int:
        return self.flags & 0x000F

@dataclass
class DNSQuestion:
    name: str
    qtype: int
    qclass: int = 1  # IN

    TYPE_MAP = {
        1: 'A', 2: 'NS', 5: 'CNAME', 6: 'SOA',
        15: 'MX', 16: 'TXT', 28: 'AAAA', 33: 'SRV', 12: 'PTR'
    }

    def type_str(self) -> str:
        return self.TYPE_MAP.get(self.qtype, f'TYPE{self.qtype}')

@dataclass
class DNSRecord:
    name: str
    rtype: int
    rclass: int
    ttl: int
    rdlength: int
    rdata: bytes

    def a_record(self) -> Optional[str]:
        if self.rtype == 1 and self.rdlength == 4:
            return '.'.join(str(b) for b in self.rdata)
        return None

    def aaaa_record(self) -> Optional[str]:
        if self.rtype == 28 and self.rdlength == 16:
            import ipaddress
            return str(ipaddress.IPv6Address(self.rdata))
        return None

    def cname_record(self) -> Optional[str]:
        if self.rtype == 5:
            return decode_name(self.rdata, 0)[0]
        return None

    def mx_record(self) -> Optional[tuple]:
        if self.rtype == 15:
            priority = struct.unpack('!H', self.rdata[:2])[0]
            exchange = decode_name(self.rdata, 2)[0]
            return (priority, exchange)
        return None

class DNSMessage:
    def __init__(self):
        self.header: Optional[DNSHeader] = None
        self.questions: List[DNSQuestion] = []
        self.answers: List[DNSRecord] = []
        self.authorities: List[DNSRecord] = []
        self.additionals: List[DNSRecord] = []

    @classmethod
    def from_bytes(cls, data: bytes) -> 'DNSMessage':
        msg = cls()
        offset = 0

        # 解析头部
        msg.header = DNSHeader.from_bytes(data, offset)
        offset += 12

        # 解析问题区
        for _ in range(msg.header.qdcount):
            name, offset = decode_name(data, offset)
            qtype, qclass = struct.unpack_from('!HH', data, offset)
            offset += 4
            msg.questions.append(DNSQuestion(name, qtype, qclass))

        # 解析回答区、权威区、附加区
        for section, count in [(msg.answers, msg.header.ancount),
                               (msg.authorities, msg.header.nscount),
                               (msg.additionals, msg.header.arcount)]:
            for _ in range(count):
                name, offset = decode_name(data, offset)
                rtype, rclass, ttl, rdlength = struct.unpack_from('!HHIH', data, offset)
                offset += 10
                rdata = data[offset:offset + rdlength]
                offset += rdlength
                section.append(DNSRecord(name, rtype, rclass, ttl, rdlength, rdata))

        return msg

def decode_name(data: bytes, offset: int) -> tuple:
    """解码 DNS 域名（支持压缩指针）"""
    parts = []
    jumped = False
    max_offset = offset

    while True:
        if offset >= len(data):
            break

        length = data[offset]

        if (length & 0xC0) == 0xC0:
            # 压缩指针
            if not jumped:
                max_offset = offset + 2
            pointer = struct.unpack('!H', data[offset:offset + 2])[0] & 0x3FFF
            offset = pointer
            jumped = True
            continue

        if length == 0:
            offset += 1
            break

        offset += 1
        parts.append(data[offset:offset + length].decode('ascii'))
        offset += length

    name = '.'.join(parts)
    return name, max_offset if jumped else offset
```

<div data-component="DNSMessageParser"></div>

---

## 19.7 DNS 缓存机制

### 19.7.1 缓存的作用

DNS 缓存是 DNS 系统高性能的关键：

```
无缓存:
  每次访问 www.google.com 都需要 4-8 次 DNS 查询
  延迟: 50-200ms

有缓存:
  首次查询后，结果缓存到 TTL 过期
  后续查询延迟: < 1ms
  缓存命中率通常 > 90%
```

### 19.7.2 缓存层级

```
DNS 缓存层级:
  1. 浏览器缓存
     ├── Chrome: chrome://net-internals/#dns
     ├── 缓存时间: 通常 1 分钟
     └── 最近访问的域名

  2. 操作系统缓存
     ├── macOS: mDNSResponder
     ├── Linux: systemd-resolved 或 nscd
     └── Windows: DNS Client Service

  3. 路由器缓存
     ├── 家庭路由器的 DNS 缓存
     └── 通常缓存时间较短

  4. ISP 递归解析器缓存
     ├── 企业级缓存，容量大
     ├── 缓存时间由 TTL 决定
     └── 服务数千用户的共享缓存

  5. 公共 DNS 缓存
     ├── Google Public DNS (8.8.8.8)
     ├── Cloudflare DNS (1.1.1.1)
     └── 大规模分布式缓存
```

### 19.7.3 TTL 管理

```python
class DNSCache:
    """DNS 缓存管理器"""
    def __init__(self, max_size: int = 100000):
        self.cache = {}  # {(domain, rtype): CacheEntry}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, domain: str, rtype: str) -> Optional['CacheEntry']:
        key = (domain.lower(), rtype)
        entry = self.cache.get(key)

        if entry and not entry.is_expired():
            self.hits += 1
            # 重新排序 LRU
            entry.last_access = time.time()
            return entry

        if entry and entry.is_expired():
            del self.cache[key]

        self.misses += 1
        return None

    def put(self, domain: str, rtype: str, records: list, ttl: int):
        # 负缓存也缓存（NXDOMAIN）
        if self._is_full():
            self._evict()

        key = (domain.lower(), rtype)
        self.cache[key] = CacheEntry(
            records=records,
            ttl=ttl,
            inserted_at=time.time(),
            last_access=time.time()
        )

    def _is_full(self) -> bool:
        return len(self.cache) >= self.max_size

    def _evict(self):
        """LRU 淘汰策略"""
        if not self.cache:
            return
        oldest_key = min(self.cache, key=lambda k: self.cache[k].last_access)
        del self.cache[oldest_key]

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class CacheEntry:
    def __init__(self, records: list, ttl: int, inserted_at: float, last_access: float):
        self.records = records
        self.original_ttl = ttl
        self.inserted_at = inserted_at
        self.last_access = last_access

    def is_expired(self) -> bool:
        return time.time() - self.inserted_at > self.original_ttl

    def remaining_ttl(self) -> int:
        return max(0, int(self.original_ttl - (time.time() - self.inserted_at)))
```

### 19.7.4 负缓存

DNS 还支持**负缓存**（Negative Caching），缓存域名不存在的结果：

```
查询 nonexistent.example.com:
  权威服务器返回 NXDOMAIN（域名不存在）
  这个结果也被缓存，TTL 由 SOA 记录的 minimum TTL 决定

好处: 减少对权威服务器的重复查询
坏处: 新注册的域名可能在缓存过期前无法解析
```

<div data-component="DNSCacheDemo"></div>

---

## 19.8 DNS 权威服务器的区域数据存储

### 19.8.1 区域文件格式

权威 DNS 服务器使用**区域文件**（Zone File）存储域名记录：

```bash
; example.com 区域文件
$ORIGIN example.com.        ; 默认域名后缀
$TTL 3600                   ; 默认 TTL

; SOA 记录
@   IN  SOA  ns1.example.com. admin.example.com. (
        2024010101  ; 序列号
        3600        ; 刷新
        900         ; 重试
        1209600     ; 过期
        86400       ; 否定缓存 TTL
)

; NS 记录
@       IN  NS   ns1.example.com.
@       IN  NS   ns2.example.com.

; A 记录
@       IN  A    93.184.216.34
www     IN  A    93.184.216.34
mail    IN  A    93.184.216.35
ns1     IN  A    93.184.216.2
ns2     IN  A    93.184.216.3

; AAAA 记录
@       IN  AAAA 2606:2800:220:1:248:1893:25c8:1946

; MX 记录
@       IN  MX   10 mail.example.com.

; CNAME 记录
www     IN  CNAME example.com.   ; 注意：实际中 www 和 A 不能同时存在

; TXT 记录
@       IN  TXT  "v=spf1 mx ~all"
```

### 19.8.2 区域数据存储结构

```c
// 区域数据存储的数据结构
typedef struct RRSet {
    char        name[256];       // 域名
    uint16_t    type;            // 记录类型
    uint16_t    rclass;          // 类别
    uint32_t    ttl;             // 生存时间
    uint16_t    rdlength;        // 数据长度
    uint8_t     rdata[512];      // 记录数据
    struct RRSet *next;          // 同域名同类型的下一条记录
} RRSet;

typedef struct Zone {
    char        origin[256];     // 区域原点（如 example.com.）
    RRSet       *soa;            // SOA 记录
    RRSet       *ns;             // NS 记录集

    // 哈希表存储域名 → 记录集的映射
    HashTable   *records;        // key: "name:type", value: RRSet*

    // 用于区域传送的序列号
    uint32_t    serial;
    time_t      last_updated;
} Zone;

// 查找域名记录
RRSet* zone_lookup(Zone *zone, const char *name, uint16_t type) {
    char key[512];
    snprintf(key, sizeof(key), "%s:%u", name, type);
    return hash_table_get(zone->records, key);
}

// 区域传送（AXFR）
int zone_transfer(Zone *zone, int client_fd) {
    // 发送 SOA 记录开始
    send_rr(zone->soa, client_fd);

    // 发送所有记录
    HashTableIterator iter;
    hash_table_iter_init(zone->records, &iter);
    while (hash_table_iter_next(&iter)) {
        RRSet *rr = hash_table_iter_value(&iter);
        send_rr(rr, client_fd);
    }

    // 发送 SOA 记录结束
    send_rr(zone->soa, client_fd);
    return 0;
}
```

<div data-component="ZoneFileEditor"></div>

---

## 19.9 DNSSEC：DNS 安全扩展

### 19.9.1 DNS 面临的安全威胁

DNS 最初设计时没有考虑安全性，面临多种威胁：

```
1. DNS 缓存投毒（Cache Poisoning）
   攻击者伪造 DNS 响应，将用户引导到恶意网站
   例如：将 bank.com 解析到攻击者的服务器

2. DNS 欺骗（DNS Spoofing）
   中间人攻击，拦截和修改 DNS 查询

3. DNS 放大攻击
   利用 DNS 响应大于查询的特点进行 DDoS

4. 域名劫持
   未经授权修改域名的 DNS 记录
```

### 19.9.2 DNSSEC 链式信任模型

DNSSEC 通过**数字签名**保证 DNS 数据的完整性和真实性：

```
DNSSEC 信任链:
  根域 (Root)
    └── 签发 .com 的 DS 记录
          └── .com TLD
                └── 签发 example.com 的 DS 记录
                      └── example.com
                            └── 签发 www.example.com 的 A 记录

每一层使用公钥加密签名，形成信任链
```

**DNSSEC 记录类型**：

| 记录类型 | 用途 |
|---------|------|
| RRSIG | 资源记录签名（对 RRSet 的数字签名） |
| DNSKEY | 公钥（用于验证签名） |
| DS | 委托签名者（子域公钥的哈希） |
| NSEC/NSEC3 | 下一安全记录（证明域名不存在） |

### 19.9.3 DNSSEC 验证流程

```python
class DNSSECValidator:
    """DNSSEC 验证器"""
    def __init__(self, trust_anchors: dict):
        # 信任锚点（根域的 DNSKEY）
        self.trust_anchors = trust_anchors

    def validate(self, response: DNSMessage) -> bool:
        """验证 DNSSEC 签名链"""
        # 1. 提取 RRSIG 记录
        rrsigs = [r for r in response.answers if r.rtype == 46]  # RRSIG
        if not rrsigs:
            return False  # 没有签名

        # 2. 对每个 RRSIG 进行验证
        for rrsig in rrsigs:
            if not self._verify_rrsig(rrsig, response.answers):
                return False

        # 3. 验证 DS 记录链
        if not self._verify_ds_chain(response):
            return False

        return True

    def _verify_rrsig(self, rrsig: DNSRecord, records: list) -> bool:
        """验证单个 RRSIG"""
        # 解析 RRSIG 字段
        sig_data = self._parse_rrsig(rrsig.rdata)

        # 查找对应的 DNSKEY
        dnskey = self._find_dnskey(sig_data['signer'], records)
        if not dnskey:
            return False

        # 构建待验证的数据
        signed_data = self._build_signed_data(sig_data, records)

        # 使用公钥验证签名
        public_key = self._extract_public_key(dnskey)
        return self._rsa_verify(public_key, signed_data, sig_data['signature'])

    def _verify_ds_chain(self, response: DNSMessage) -> bool:
        """验证 DS 记录的信任链"""
        # 从根域开始，逐层验证
        current_domain = '.'
        for ds_record in self._get_ds_records(response):
            # DS 记录包含子域 DNSKEY 的摘要
            expected_hash = ds_record.rdata
            child_dnskey = self._find_dnskey_for_domain(ds_record.name)

            if not child_dnskey:
                return False

            # 计算 DNSKEY 的摘要
            computed_hash = self._compute_ds_hash(child_dnskey, ds_record.rdata)

            if computed_hash != expected_hash:
                return False  # 信任链断裂

        return True
```

<div data-component="DNSSECTrustChain"></div>

---

## 19.10 DNS 负载均衡

### 19.10.1 DNS 负载均衡原理

DNS 负载均衡是通过返回不同的 IP 地址来分配流量：

```
用户查询 www.example.com:
  查询 1 → 返回 93.184.216.34（服务器 A）
  查询 2 → 返回 93.184.216.35（服务器 B）
  查询 3 → 返回 93.184.216.36（服务器 C）
  查询 4 → 返回 93.184.216.34（服务器 A，循环）
```

### 19.10.2 负载均衡算法

```python
class DNSLoadBalancer:
    """DNS 负载均衡器"""
    def __init__(self):
        self.backends = []  # [(ip, weight, health_status)]

    def add_backend(self, ip: str, weight: int = 1):
        self.backends.append({
            'ip': ip,
            'weight': weight,
            'healthy': True,
            'current_weight': 0
        })

    def round_robin(self) -> str:
        """简单轮询"""
        healthy = [b for b in self.backends if b['healthy']]
        if not healthy:
            raise Exception("No healthy backends")
        # 轮询
        self._rr_index = getattr(self, '_rr_index', -1) + 1
        return healthy[self._rr_index % len(healthy)]['ip']

    def weighted_round_robin(self) -> str:
        """加权轮询"""
        healthy = [b for b in self.backends if b['healthy']]
        if not healthy:
            raise Exception("No healthy backends")

        # 平滑加权轮询（Smooth Weighted Round-Robin）
        total_weight = sum(b['weight'] for b in healthy)
        for b in healthy:
            b['current_weight'] += b['weight']

        selected = max(healthy, key=lambda b: b['current_weight'])
        selected['current_weight'] -= total_weight
        return selected['ip']

    def least_connections(self, conn_counts: dict) -> str:
        """最少连接"""
        healthy = [b for b in self.backends if b['healthy']]
        if not healthy:
            raise Exception("No healthy backends")
        return min(healthy, key=lambda b: conn_counts.get(b['ip'], 0))['ip']

    def geographic(self, client_ip: str, geo_db: dict) -> str:
        """地理位置感知"""
        client_region = geo_db.get(client_ip, 'default')
        region_backends = [b for b in self.backends
                          if b['healthy'] and b.get('region') == client_region]
        if region_backends:
            return self.weighted_round_robin_from(region_backends)
        return self.weighted_round_robin()
```

### 19.10.3 健康检查

```python
class HealthChecker:
    """后端健康检查器"""
    def __init__(self, backends: list, check_interval: int = 30):
        self.backends = backends
        self.check_interval = check_interval

    def check_health(self, backend: dict) -> bool:
        """检查单个后端的健康状态"""
        try:
            # TCP 检查
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((backend['ip'], 80))
            sock.close()
            return result == 0
        except:
            return False

    def run_health_checks(self):
        """运行所有健康检查"""
        for backend in self.backends:
            was_healthy = backend['healthy']
            is_healthy = self.check_health(backend)
            backend['healthy'] = is_healthy

            if was_healthy and not is_healthy:
                self._on_backend_down(backend)
            elif not was_healthy and is_healthy:
                self._on_backend_up(backend)
```

<div data-component="DNSLoadBalancerDemo"></div>

---

## 19.11 DNS over HTTPS (DoH)

### 19.11.1 传统 DNS 的隐私问题

```
传统 DNS 查询:
  客户端 --[明文 UDP]--> 递归解析器 --[明文 UDP]--> 权威服务器

问题:
  1. 中间人可以看到所有 DNS 查询
  2. ISP 可以记录用户的访问历史
  3. 政府可以审查和屏蔽域名
  4. 企业网络可以监控员工上网行为
```

### 19.11.2 DoH 工作原理

DNS over HTTPS (DoH) 将 DNS 查询封装在 HTTPS 请求中：

```
DoH 查询:
  客户端 --[HTTPS]--> DoH 服务器 --[DNS]--> 权威服务器

请求格式:
  POST /dns-query HTTP/2
  Content-Type: application/dns-message
  Body: <DNS 查询的二进制编码>

  或
  GET /dns-query?dns=<base64url编码的DNS查询>
  Accept: application/dns-message
```

```python
import requests
import base64

def doh_query(domain: str, record_type: str, doh_server: str = "https://1.1.1.1/dns-query") -> dict:
    """DNS over HTTPS 查询"""
    # 构建 DNS 查询报文
    query = build_dns_query(domain, record_type)

    # 编码为 base64url
    encoded = base64.urlsafe_b64encode(query).rstrip(b'=').decode()

    # 发送 HTTPS GET 请求
    response = requests.get(
        doh_server,
        params={'dns': encoded},
        headers={'Accept': 'application/dns-message'}
    )

    # 解析 DNS 响应
    return parse_dns_response(response.content)
```

### 19.11.3 DoH vs DoT

| 特性 | DoH (DNS over HTTPS) | DoT (DNS over TLS) |
|------|---------------------|-------------------|
| 端口 | 443（与 HTTPS 相同） | 853 |
| 协议 | HTTPS/2 | TLS |
| 隐蔽性 | 高（与普通 HTTPS 流量混合） | 低（专用端口，易被封锁） |
| 性能 | 与 HTTPS 共享连接 | 需要独立 TLS 连接 |
| 部署 | 浏览器原生支持 | 需要操作系统支持 |

<div data-component="DoHDemo"></div>

---

## 19.12 章节小结

本章详细介绍了 DNS 域名系统的各个方面：

1. **设计动机**：DNS 是互联网的电话簿，将域名转换为 IP 地址
2. **层次结构**：域名空间的树形结构，从根域到 TLD 到二级域
3. **服务器层次**：根服务器、TLD 服务器、权威服务器、递归解析器的协作
4. **查询流程**：递归查询与迭代查询的区别和实现
5. **记录类型**：A/AAAA/CNAME/MX/NS/TXT/SOA/PTR/SRV 等记录的用途和格式
6. **消息格式**：DNS 报文的五部分结构和二进制编码
7. **缓存机制**：多级缓存、TTL 管理、负缓存
8. **安全扩展**：DNSSEC 的链式信任模型和验证流程
9. **负载均衡**：轮询、加权轮询、地理位置感知等算法
10. **隐私保护**：DoH 和 DoT 的工作原理和对比

<div data-component="ChapterSummary"></div>
<div data-component="KnowledgeCheck"></div>

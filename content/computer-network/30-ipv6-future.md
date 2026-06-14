# Chapter 30: IPv6 与网络技术演进

> **学习目标**：
> - 理解 IPv4 地址耗尽危机的成因、时间线与量化影响，掌握 NAT/CIDR/VLSM 等延缓策略的原理与局限
> - 深入理解 IPv6 的设计动机、端到端原则恢复、头部格式简化与协议报文结构
> - 掌握 IPv6 地址表示法、分类体系（链路本地/唯一本地/全球单播/组播/任播）及地址分配层次
> - 深入理解 ICMPv6 与邻居发现协议（NDP）的五种消息类型、邻居缓存状态机及 DAD 流程
> - 掌握 IPv6 地址自动配置（SLAAC/DHCPv6）的工作机制、EUI-64 接口标识符与隐私扩展
> - 理解 IPv4 到 IPv6 过渡技术体系（双栈/隧道/翻译），深入分析 NAT64 协议转换引擎
> - 了解全球 IPv6 部署现状、IPv6-only 趋势以及 5G/IoT 场景下的推动作用

---

## 30.1 IPv4 地址耗尽危机

### 30.1.1 IPv4 地址空间的理论上限与实际可用

IPv4 使用 32 位地址，理论上提供 $2^{32} = 4{,}294{,}967{,}296$ 个地址。但其中大量地址被保留用于特殊用途：

| 地址块 | CIDR 表示 | 数量 | 用途 |
|--------|-----------|------|------|
| 0.0.0.0/8 | 0.0.0.0 ~ 0.255.255.255 | 16,777,216 | 本网络 |
| 10.0.0.0/8 | 10.0.0.0 ~ 10.255.255.255 | 16,777,216 | 私有地址 |
| 100.64.0.0/10 | 100.64.0.0 ~ 100.127.255.255 | 4,194,304 | CGN 地址 |
| 127.0.0.0/8 | 127.0.0.0 ~ 127.255.255.255 | 16,777,216 | 环回地址 |
| 169.254.0.0/16 | 169.254.0.0 ~ 169.254.255.255 | 65,536 | 链路本地 |
| 172.16.0.0/12 | 172.16.0.0 ~ 172.31.255.255 | 1,048,576 | 私有地址 |
| 192.0.0.0/24 | 192.0.0.0 ~ 192.0.0.255 | 256 | IANA 保留 |
| 192.0.2.0/24 | 192.0.2.0 ~ 192.0.2.255 | 256 | 文档示例 |
| 192.88.99.0/24 | — | 256 | 6to4 中继 |
| 192.168.0.0/16 | 192.168.0.0 ~ 192.168.255.255 | 65,536 | 私有地址 |
| 224.0.0.0/4 | 224.0.0.0 ~ 239.255.255.255 | 268,435,456 | 组播地址 |
| 240.0.0.0/4 | 240.0.0.0 ~ 255.255.255.255 | 268,435,456 | 保留（E类） |

扣除这些保留地址后，实际可分配的全球单播地址约 **37 亿**，远少于理论值。

```
IPv4 地址空间分配概览：
┌──────────────────────────────────────────────────────────┐
│                    2^32 = 42.9 亿                        │
│  ┌─────────┐ ┌────┐ ┌──────┐ ┌─────┐ ┌───────┐ ┌─────┐ │
│  │ 保留 6亿 │ │NAT │ │组播  │ │E类  │ │其他   │ │可分配│ │
│  │(15.7%)  │ │私有 │ │(6.3%)│ │(6.3%)│ │保留   │ │37亿 │ │
│  │         │ │地址 │ │      │ │     │ │(2.4%) │ │     │ │
│  └─────────┘ └────┘ └──────┘ └─────┘ └───────┘ └─────┘ │
└──────────────────────────────────────────────────────────┘
```

### 30.1.2 IANA 地址池耗尽的历史时刻

**2011 年 2 月 3 日**，IANA 将最后 5 个 /8 地址块分别分配给五个区域互联网注册管理机构（RIR），标志着 IPv4 中央地址池正式耗尽：

```
IANA /8 地址块分配时间线：
──────────────────────────────────────────────────────────────
1981        1995        2005        2011.02.03     2011+
  │           │           │              │            │
  ▼           ▼           ▼              ▼            ▼
 IPv4开始    地址增速    CIDR引入     IANA耗尽      RIR逐步
 大规模分配   加快       延缓分配     最后5个/8     耗尽
                              ↓
                    ┌─────────────────────┐
                    │ APNIC  : 1/8        │
                    │ ARIN   : 1/8        │
                    │ RIPE   : 1/8        │
                    │ LACNIC : 1/8        │
                    │ AFRINIC: 1/8        │
                    └─────────────────────┘
```

各 RIR 的耗尽时间线：

| RIR | 区域 | 耗尽时间 | 影响国家/地区 |
|-----|------|----------|---------------|
| APNIC | 亚太 | 2011.04.15 | 中国、日本、澳大利亚等 |
| RIPE NCC | 欧洲/中东/中亚 | 2012.09.14 | 欧盟、俄罗斯等 |
| LACNIC | 拉美/加勒比 | 2014.06.10 | 巴西、墨西哥等 |
| ARIN | 北美 | 2015.09.24 | 美国、加拿大 |
| AFRINIC | 非洲 | 2017.04.21 | 尼日利亚、南非等 |

### 30.1.3 NAT 的权宜之计与根本局限

NAT（Network Address Translation）通过地址复用延缓了地址耗尽，但从根本上破坏了互联网的端到端原则：

```
NAT 工作原理（NAPT / PAT）：

 内网主机 A (192.168.1.10:3000)    ┌─────────────┐    外网服务器
 ──────────────────────────────────▶│   NAT 设备    │──────────────▶ 8.8.8.8:80
 ◀──────────────────────────────────│              │◀──────────────
                                    │ 内:192.168.1.10:3000 │
                                    │ 外:203.0.113.5:40001 │
                                    └─────────────┘

 转换表：
┌──────────────┬──────────────┬──────────────┬────────┐
│ 内网地址:端口  │ 外网地址:端口  │ 目标地址:端口  │ 协议   │
├──────────────┼──────────────┼──────────────┼────────┤
│ 192.168.1.10:3000 │ 203.0.113.5:40001 │ 8.8.8.8:80 │ TCP  │
│ 192.168.1.11:5000 │ 203.0.113.5:40002 │ 1.1.1.1:443│ TCP  │
└──────────────┴──────────────┴──────────────┴────────┘
```

**NAT 的核心局限**：

1. **破坏端到端透明性**：外部主机无法主动发起对内网主机的连接
2. **NAT 穿越困难**：P2P 应用、VoIP、视频会议需要 STUN/TURN/ICE 等复杂机制
3. **状态维护开销**：每个连接需要在 NAT 表中维护条目，连接数受限于表大小
4. **协议兼容性问题**：在载荷中嵌入 IP 地址的协议（FTP/SIP/H.323）需要 ALG 辅助
5. **多层 NAT（CGN）问题**：运营商级 NAT 导致双重 NAT，延迟增加、故障排查困难

<div data-component="NATTraversalDemo"></div>

### 30.1.4 CIDR 与 VLSM 的延缓效果

CIDR（Classless Inter-Domain Routing）和 VLSM（Variable Length Subnet Masking）通过取消固定地址类别、允许任意前缀长度，提高了地址利用率：

```
CIDR 前缀聚合示例：
┌──────────────────────────────────────────────────────┐
│ 路由器 R1 收到以下 4 条路由：                           │
│  192.168.0.0/24  → 下一跳 A                           │
│  192.168.1.0/24  → 下一跳 A                           │
│  192.168.2.0/24  → 下一跳 A                           │
│  192.168.3.0/24  → 下一跳 A                           │
├──────────────────────────────────────────────────────┤
│ CIDR 聚合后：                                         │
│  192.168.0.0/22  → 下一跳 A    ← 1 条路由替代 4 条      │
│                                                     │
│ 二进制分析：                                          │
│  192.168.0.0 = 11000000.10101000.000000|00.00000000  │
│  192.168.1.0 = 11000000.10101000.000000|01.00000000  │
│  192.168.2.0 = 11000000.10101000.000000|10.00000000  │
│  192.168.3.0 = 11000000.10101000.000000|11.00000000  │
│                     共同前缀 22 位 ─────┘              │
└──────────────────────────────────────────────────────┘
```

然而，CIDR/VLSM 只能优化地址分配效率，无法从根本上增加地址空间。到 2011 年，所有优化手段已无法满足指数增长的联网设备需求。

### 30.1.5 量化分析：地址耗尽的必然性

联网设备增长模型可用指数函数近似：

$$N(t) = N_0 \cdot e^{rt}$$

其中 $N_0$ 为初始设备数，$r$ 为年增长率，$t$ 为年数。

```python
import math

def estimate_exhaustion(initial_hosts, annual_growth_rate, total_addresses):
    """估算 IPv4 地址耗尽时间"""
    current = initial_hosts
    year = 0
    while current < total_addresses:
        current *= (1 + annual_growth_rate)
        year += 1
    return year

# 2000 年全球互联网用户约 4 亿，年增长率约 15%
years = estimate_exhaustion(
    initial_hosts=400_000_000,
    annual_growth_rate=0.15,
    total_addresses=3_700_000_000  # 可分配地址
)
print(f"理论耗尽年数: {years} 年 (从2000年起)")
# 输出: 理论耗尽年数: 16 年 → 约 2016 年

# 实际上由于 IoT 设备爆发，增长率更高
# 智能手机、平板、IoT 设备使增长曲线更陡峭
```

---

## 30.2 IPv6 的设计动机与目标

### 30.2.1 端到端原则的恢复

互联网最初的设计遵循 **端到端原则**（End-to-End Principle）：网络核心保持简单，智能放在端系统。NAT 的普及严重破坏了这一原则。

```
端到端原则的对比：

IPv4 + NAT 时代：                    IPv6 时代：
┌──────┐  NAT  ┌──────┐           ┌──────┐        ┌──────┐
│ 主机A │──────│ NAT  │───── 服务器 │ 主机A │────────│ 服务器│
│      │  地址  │      │           │      │ 端到端   │      │
│ 私有IP│  转换  │ 公网IP│           │全球IP │ 直接通信 │全球IP│
└──────┘       └──────┘           └──────┘        └──────┘
  - 外部无法主动连接 A                - 任何一方可发起连接
  - P2P 需要 NAT 穿越                - 天然支持 P2P
  - 协议需 NAT 感知                  - 所有协议透明传输
```

### 30.2.2 IPv6 的核心设计目标

IPv6（最初称为 IPng, IP Next Generation）由 IETF 在 RFC 2460（1998）中定义，后被 RFC 8200（2017）取代。其设计目标包括：

| 设计目标 | IPv4 的问题 | IPv6 的解决方案 |
|----------|------------|----------------|
| 更大的地址空间 | 32 位，约 37 亿可用 | 128 位，约 $3.4 \times 10^{38}$ 个 |
| 简化头部格式 | 可变长头部，选项使处理复杂 | 固定 40 字节基本头 + 扩展头链 |
| 消除 NAT | 地址不足被迫使用 NAT | 充足地址，恢复端到端通信 |
| 更好的移动性 | 移动 IP 效率低 | 内建移动 IPv6 支持 |
| 自动配置 | 依赖 DHCP 服务器 | SLAAC 无状态自动配置 |
| 内建安全性 | IPsec 为可选 | IPsec 为标准组件（尽管实践中仍为可选） |
| 更高效的路由 | 路由表碎片化 | 层次化地址分配，聚合更高效 |

### 30.2.3 地址空间对比

IPv6 的 128 位地址空间是 IPv4 的 $2^{96}$ 倍：

$$\text{IPv6 地址数} = 2^{128} \approx 3.4 \times 10^{38}$$

$$\text{IPv4 地址数} = 2^{32} \approx 4.3 \times 10^{9}$$

$$\text{比值} = \frac{2^{128}}{2^{32}} = 2^{96} \approx 7.9 \times 10^{28}$$

直观理解：如果 IPv4 地址空间是一个乒乓球，那么 IPv6 地址空间大约相当于整个太阳系的体积。

```python
def compare_address_spaces():
    """IPv4 与 IPv6 地址空间对比"""
    ipv4_bits = 32
    ipv6_bits = 128
    ipv4_total = 2 ** ipv4_bits
    ipv6_total = 2 ** ipv6_bits
    ratio = ipv6_total / ipv4_total
    
    # 地球表面积约 5.1 × 10^14 平方米
    earth_surface_m2 = 5.1e14
    # 每平方米可分配的 IPv6 地址数
    per_sqm = ipv6_total / earth_surface_m2
    
    print(f"IPv4 总地址: {ipv4_total:>40,}")
    print(f"IPv6 总地址: {ipv6_total:>40,}")
    print(f"倍数关系:   {ratio:>40,.0f}")
    print(f"每平方米地球表面: {per_sqm:,.0f} 个 IPv6 地址")
    
compare_address_spaces()
# IPv4 总地址:                     4,294,967,296
# IPv6 总地址:   340,282,366,920,938,463,463,374,607,431,768,211,456
# 倍数关系:     79,228,162,514,264,337,593,543,950,336
# 每平方米地球表面: 667,225,821,830,126,634,556,096,819,624,795 个 IPv6 地址
```

### 30.2.4 设计权衡与争议

IPv6 的设计并非没有争议：

- **地址长度**：128 位是否过大？64 位已可提供 $1.8 \times 10^{19}$ 个地址。128 位选择考虑了层次化分配的消耗和未来扩展
- **与 IPv4 不兼容**：无法直接互通，需要过渡技术，这是部署缓慢的根本原因
- **IPsec 地位**：RFC 6434 将 IPsec 从"必须实现"降级为"应该实现"
- **流标签字段**：设计之初预期用于 QoS，但长期未被充分利用

---

## 30.3 IPv6 报文格式

### 30.3.1 IPv6 基本头部结构

IPv6 基本头部固定为 **40 字节**（320 位），相比 IPv4 头部（20~60 字节）更加规整：

```
IPv6 基本头部（40 字节）：
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│Version│         Traffic Class         │           Flow Label          │
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│         Payload Length                │  Next Header  │  Hop Limit    │
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│                                                                     │
│                        Source Address (128 bits)                     │
│                                                                     │
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│                                                                     │
│                     Destination Address (128 bits)                   │
│                                                                     │
└─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘
```

各字段详细说明：

| 字段 | 位数 | 说明 |
|------|------|------|
| Version | 4 | 版本号，值为 6 |
| Traffic Class | 8 | 等同于 IPv4 的 TOS/DSCP+ECN，用于区分服务和拥塞通知 |
| Flow Label | 20 | 流标签，标识同一"流"的数据报，支持流式处理和 QoS |
| Payload Length | 16 | 基本头部之后的有效载荷长度（最大 65,535 字节，超过时使用 Jumbogram 扩展头） |
| Next Header | 8 | 指示紧跟基本头部的协议类型（扩展头上层协议号） |
| Hop Limit | 8 | 跳数限制，等同于 IPv4 的 TTL，每经过一跳减 1 |
| Source Address | 128 | 源 IPv6 地址 |
| Destination Address | 128 | 目的 IPv6 地址 |

**与 IPv4 头部的关键差异**：

```
IPv4 头部 vs IPv6 头部：
┌─────────────────────────────────────────────────────────────┐
│                   IPv4 (20~60 字节)                         │
│ ┌──────┐┌────┐┌──────┐┌──────┐┌──┐┌───┐┌────┐┌───────────┐│
│ │版本  ││IHL ││TOS   ││总长度 ││ID││标志││TTL ││选项(0~40B)││
│ │4bit  ││4bit││8bit  ││16bit ││  ││片偏││8bit││           ││
│ └──────┘└────┘└──────┘└──────┘└──┘└───┘└────┘└───────────┘│
├─────────────────────────────────────────────────────────────┤
│                   IPv6 (固定 40 字节)                        │
│ ┌──────┐┌──────────┐┌──────────┐┌──────┐┌──────┐┌────────┐│
│ │版本  ││流量类     ││流标签    ││载荷长 ││下一头 ││跳数限制 ││
│ │4bit  ││8bit      ││20bit    ││16bit ││8bit  ││8bit    ││
│ └──────┘└──────────┘└──────────┘└──────┘└──────┘└────────┘│
│ ┌─────────────────────────────────────────────────────────┐│
│ │              源地址 (128 位)                              ││
│ ├─────────────────────────────────────────────────────────┤│
│ │              目的地址 (128 位)                            ││
│ └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘

IPv6 删除的 IPv4 字段：
  ✗ IHL        → 头部固定长度，无需此字段
  ✗ 标识/标志/片偏移 → 移至分片扩展头，路由器不参与分片
  ✗ 头部校验和  → 链路层和传输层已有校验，中间节点无需逐跳校验
  ✗ 选项       → 移至扩展头链，不影响基本头部处理效率
```

<div data-component="IPv6HeaderParser"></div>

### 30.3.2 扩展头部链

IPv6 使用**扩展头链**（Extension Header Chain）取代 IPv4 的选项字段。每个扩展头通过 Next Header 字段链接：

```
扩展头链示例：
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐
│ IPv6 基本 │   │ 逐跳选项  │   │ 路由头   │   │ 分片头   │   │ TCP    │
│ 头部      │──▶│ 扩展头    │──▶│ 扩展头   │──▶│ 扩展头   │──▶│ 数据   │
│ NH=0     │   │ NH=43    │   │ NH=44    │   │ NH=6    │   │        │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └────────┘
```

标准扩展头类型及编号：

| Next Header 值 | 扩展头类型 | 必须由路由器处理 | 说明 |
|----------------|-----------|-----------------|------|
| 0 | 逐跳选项（Hop-by-Hop Options） | 是 | 每个中间节点都必须检查 |
| 43 | 路由头（Routing） | 是（特定类型） | 指定数据报经过的中间节点 |
| 44 | 分片头（Fragment） | 否 | 源端分片，中间节点不分片 |
| 51 | 认证头（AH） | 否 | IPsec 认证 |
| 50 | 封装安全载荷（ESP） | 否 | IPsec 加密 |
| 60 | 目的选项（Destination Options） | 否 | 仅目的节点处理 |
| 59 | 无下一个头 | — | 头部链结束 |
| 135 | 移动 IPv6 | — | 移动节点支持 |

### 30.3.3 扩展头解析管线（核心组件）

IPv6 数据报处理器的**扩展头解析管线**是路由器转发引擎的关键子系统。其设计目标是在保证灵活性的同时最小化处理延迟。

```
扩展头解析管线架构：
┌─────────────────────────────────────────────────────────────────┐
│                    IPv6 数据报处理器                              │
│                                                                  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐      │
│  │ Stage 1  │    │ Stage 2  │    │ Stage 3  │    │ Stage 4  │      │
│  │ 基本头   │───▶│ Hop-by-  │───▶│ 路由/分片│───▶│ 目的选项 │───▶ 上层
│  │ 解析器   │    │ Hop选项  │    │ 处理器   │    │ 处理器   │      │
│  │         │    │ 处理器   │    │         │    │         │      │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘      │
│       │              │              │              │             │
│       ▼              ▼              ▼              ▼             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              头部链遍历状态机                               │   │
│  │  current_offset, next_header, remaining_length            │   │
│  └──────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              安全与限速模块                                 │   │
│  │  - 扩展头链深度限制（防 DoS）                               │   │
│  │  - 逐跳选项处理超时保护                                     │   │
│  │  - 异常报文统计与丢弃                                       │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**解析管线详细数据路径**：

```
数据报进入 ──▶ [Stage 1] 基本头部解析
                  │
                  ├─ 验证版本号 = 6
                  ├─ 提取 Traffic Class / Flow Label
                  ├─ 读取 Payload Length
                  ├─ 读取 Next Header → 决定第一个扩展头类型
                  ├─ 读取 Hop Limit → 递减并检查是否为 0
                  ├─ 提取源/目的地址 (各 128 位)
                  │
                  ▼
             [Stage 2] 逐跳选项处理（如果 Next Header = 0）
                  │
                  ├─ 解析逐跳扩展头长度
                  ├─ 遍历 TLV 选项：
                  │    ├─ Pad1 / PadN → 跳过
                  │    ├─ Jumbo Payload → 更新载荷长度
                  │    ├─ Router Alert → 通知控制平面
                  │    └─ 未知选项 → 检查最高两位决定动作
                  │         ├─ 00 → 跳过
                  │         ├─ 01 → 丢弃
                  │         ├─ 10 → 丢弃 + ICMP
                  │         └─ 11 → 丢弃 + ICMP + 非成员
                  │
                  ▼
             [Stage 3] 路由头/分片头处理
                  │
                  ├─ 如果是路由头 (Type 2, SRv6):
                  │    ├─ 验证路由头完整性
                  │    ├─ Segments Left > 0 → 转发到下一个段
                  │    └─ Segments Left = 0 → 正常转发
                  │
                  ├─ 如果是分片头:
                  │    ├─ 提取 Fragment Offset / M flag
                  │    ├─ 执行分片重组（仅目的端）
                  │    └─ 超时未重组 → 丢弃 + ICMP
                  │
                  ▼
             [Stage 4] 目的选项处理（最后一跳前的扩展头）
                  │
                  ├─ 解析目的选项 TLV
                  ├─ 处理 Home Address / Tunnel Encapsulation Limit 等
                  │
                  ▼
             [上层协议分发]
                  │
                  ├─ Next Header = 6  → TCP 处理器
                  ├─ Next Header = 17 → UDP 处理器
                  ├─ Next Header = 58 → ICMPv6 处理器
                  └─ Next Header = 59 → 无下一个头（丢弃或静默）
```

**核心数据结构**：

```python
from dataclasses import dataclass, field
from typing import List, Optional
from enum import IntEnum

class NextHeaderType(IntEnum):
    HOP_BY_HOP = 0
    TCP = 6
    UDP = 17
    ROUTING = 43
    FRAGMENT = 44
    ESP = 50
    AH = 51
    DESTINATION_OPTIONS = 60
    ICMPv6 = 58
    NO_NEXT_HEADER = 59

@dataclass
class IPv6Header:
    version: int = 6
    traffic_class: int = 0
    flow_label: int = 0
    payload_length: int = 0
    next_header: int = 0
    hop_limit: int = 64
    source: bytes = field(default_factory=lambda: b'\x00' * 16)
    destination: bytes = field(default_factory=lambda: b'\x00' * 16)

@dataclass
class ExtensionHeader:
    next_header: int
    header_length: int  # 不含前 8 字节的长度（以 8 字节为单位）
    data: bytes
    options: List[dict] = field(default_factory=list)

@dataclass
class IPv6Packet:
    header: IPv6Header
    extension_headers: List[ExtensionHeader] = field(default_factory=list)
    upper_layer_data: bytes = b''

class ExtensionHeaderParser:
    """IPv6 扩展头解析管线"""
    
    MAX_EXTENSION_HEADERS = 8      # 最大扩展头数量限制（防 DoS）
    MAX_HEADER_CHAIN_BYTES = 1024  # 扩展头链总字节限制
    
    def __init__(self):
        self.stats = {
            'total_packets': 0,
            'hop_by_hop_processed': 0,
            'fragment_processed': 0,
            'routing_processed': 0,
            'too_many_extensions': 0,
            'chain_too_long': 0,
            'malformed_headers': 0,
        }
    
    def parse(self, raw: bytes) -> IPv6Packet:
        """解析完整的 IPv6 数据报"""
        self.stats['total_packets'] += 1
        
        # Stage 1: 基本头部解析
        header = self._parse_base_header(raw[:40])
        
        # Stage 2-4: 扩展头链解析
        extension_headers = []
        offset = 40
        next_header = header.next_header
        chain_bytes = 0
        header_count = 0
        
        while next_header not in (
            NextHeaderType.TCP,
            NextHeaderType.UDP,
            NextHeaderType.ICMPv6,
            NextHeaderType.NO_NEXT_HEADER
        ):
            # 安全检查：扩展头数量限制
            header_count += 1
            if header_count > self.MAX_EXTENSION_HEADERS:
                self.stats['too_many_extensions'] += 1
                break
            
            # 安全检查：总字节限制
            if chain_bytes > self.MAX_HEADER_CHAIN_BYTES:
                self.stats['chain_too_long'] += 1
                break
            
            # 解析单个扩展头
            ext_hdr, consumed = self._parse_extension_header(
                raw, offset, next_header
            )
            if ext_hdr is None:
                self.stats['malformed_headers'] += 1
                break
            
            extension_headers.append(ext_hdr)
            offset += consumed
            chain_bytes += consumed
            next_header = ext_hdr.next_header
        
        return IPv6Packet(
            header=header,
            extension_headers=extension_headers,
            upper_layer_data=raw[offset:]
        )
    
    def _parse_base_header(self, data: bytes) -> IPv6Header:
        """解析 40 字节基本头部"""
        return IPv6Header(
            version=(data[0] >> 4) & 0xF,
            traffic_class=((data[0] & 0xF) << 4) | ((data[1] >> 4) & 0xF),
            flow_label=((data[1] & 0xF) << 16) | (data[2] << 8) | data[3],
            payload_length=(data[4] << 8) | data[5],
            next_header=data[6],
            hop_limit=data[7],
            source=data[8:24],
            destination=data[24:40],
        )
    
    def _parse_extension_header(
        self, raw: bytes, offset: int, nh: int
    ) -> tuple:
        """解析单个扩展头，返回 (ExtensionHeader, consumed_bytes)"""
        if offset + 2 > len(raw):
            return None, 0
        
        next_hdr = raw[offset]
        hdr_len_units = raw[offset + 1]
        
        if nh == NextHeaderType.HOP_BY_HOP:
            self.stats['hop_by_hop_processed'] += 1
            total_len = (hdr_len_units + 1) * 8
        elif nh == NextHeaderType.ROUTING:
            self.stats['routing_processed'] += 1
            total_len = (hdr_len_units + 1) * 8
        elif nh == NextHeaderType.FRAGMENT:
            self.stats['fragment_processed'] += 1
            total_len = 8  # 分片头固定 8 字节
        elif nh == NextHeaderType.DESTINATION_OPTIONS:
            total_len = (hdr_len_units + 1) * 8
        else:
            total_len = max(2, hdr_len_units + 2)
        
        data = raw[offset:offset + total_len]
        
        ext = ExtensionHeader(
            next_header=next_hdr,
            header_length=hdr_len_units,
            data=data[2:],
        )
        return ext, total_len


# 测试解析管线
parser = ExtensionHeaderParser()
# 构造一个简单的 IPv6 数据报
test_raw = bytes([
    0x60, 0x00, 0x00, 0x00,  # Version=6, TC=0, Flow=0
    0x00, 0x14,              # Payload Length = 20
    0x06,                    # Next Header = TCP
    0x40,                    # Hop Limit = 64
]) + b'\x20\x01\x0d\xb8' + b'\x00' * 12   # 源地址
test_raw += b'\x20\x01\x0d\xb8' + b'\x00' * 12  # 目的地址
test_raw += b'\x00' * 20  # TCP 载荷

pkt = parser.parse(test_raw)
print(f"版本: {pkt.header.version}")
print(f"跳数限制: {pkt.header.hop_limit}")
print(f"下一头部: {pkt.header.next_header}")
print(f"扩展头数量: {len(pkt.extension_headers)}")
```

**解析管线性能指标**：

| 指标 | 理想值 | 实际考量 |
|------|--------|----------|
| 基本头解析延迟 | 1 时钟周期 | TCAM 查找需要 2-3 周期 |
| 每个扩展头处理延迟 | 2-5 时钟周期 | 取决于 TLV 选项复杂度 |
| 最大扩展头链深度 | 8 | 防 DoS 攻击的安全限制 |
| 扩展头链总字节限制 | 1024 字节 | 防止异常大的扩展头消耗缓冲区 |
| 流水线吞吐量 | 1 pps @ 100Gbps | 取决于最小包大小 |

**设计权衡**：

- **灵活性 vs 性能**：扩展头链提供了极大的灵活性，但深层嵌套增加了处理延迟
- **安全性 vs 功能**：限制扩展头深度可以防 DoS，但也可能限制合法用例（如 SRv6 的深度段列表）
- **逐跳处理 vs 端到端**：仅 Hop-by-Hop 选项需要中间节点处理，其他扩展头可被路由器跳过（快速路径优化）

---

## 30.4 IPv6 地址表示与分类

### 30.4.1 冒号十六进制表示法

IPv6 地址使用 8 组 4 位十六进制数表示，组间用冒号分隔：

```
标准格式：
2001:0db8:0000:0000:0000:0000:0000:0001

缩写规则：
1. 每组前导零可省略：
   2001:0db8:0000:0000:0000:0000:0000:0001
   → 2001:db8:0:0:0:0:0:1

2. 连续全零组可用 :: 替代（每个地址仅可使用一次）：
   2001:db8:0:0:0:0:0:1
   → 2001:db8::1

常见地址示例：
┌──────────────────────┬─────────────────┬──────────────────┐
│ 标准格式              │ 缩写格式         │ 说明             │
├──────────────────────┼─────────────────┼──────────────────┤
│ 0000:0000:0000:0000  │ ::              │ 未指定地址        │
│ :0000:0000:0000:0000 │                 │                  │
├──────────────────────┼─────────────────┼──────────────────┤
│ 0000:0000:0000:0000  │ ::1             │ 环回地址          │
│ :0000:0000:0000:0001 │                 │                  │
├──────────────────────┼─────────────────┼──────────────────┤
│ fe80:0000:0000:0000  │ fe80::1         │ 链路本地地址       │
│ :0000:0000:0000:0001 │                 │                  │
├──────────────────────┼─────────────────┼──────────────────┤
│ 2001:0db8:0000:0000  │ 2001:db8::abcd  │ 文档前缀中的地址   │
│ :0000:0000:0000:abcd │                 │                  │
└──────────────────────┴─────────────────┴──────────────────┘
```

**CIDR 表示法**同样适用于 IPv6：

```
2001:db8::/32    → 文档前缀
2001:db8:abcd::/48 → 一个组织的分配
fe80::/10        → 链路本地前缀
```

<div data-component="IPv6AddressConverter"></div>

### 30.4.2 IPv6 地址分类体系

```
IPv6 地址空间全景：
┌────────────────────────────────────────────────────────────────┐
│                      128 位地址空间                              │
│                                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ 全球单播  │  │ 链路本地  │  │ 唯一本地  │  │   组播       │   │
│  │ 2000::/3  │  │fe80::/10 │  │fc00::/7  │  │  ff00::/8    │   │
│  │ (1/8 空间)│  │          │  │          │  │              │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
│                                                                │
│  ┌──────────┐  ┌──────────┐                                    │
│  │  未指定   │  │  环回     │                                    │
│  │  ::/128  │  │  ::1/128 │                                    │
│  └──────────┘  └──────────┘                                    │
└────────────────────────────────────────────────────────────────┘
```

**全球单播地址（Global Unicast）— 2000::/3**：

```
全球单播地址结构 (2000::/3)：
┌──────────┬──────────────┬───────────────────────────────────────┐
│ 全球路由  │  站点前缀     │        子网 ID       │ 接口标识符    │
│ 前缀 (3) │  (ISP分配)    │        (站点分配)    │ (64位)       │
│ 001      │              │                     │              │
└──────────┴──────────────┴───────────────────────────────────────┘
 │← 通常 /48 前缀 →│← /64 子网 →│← /64 接口标识 →│

示例：2001:0db8:abcd:0012::1
      ├──────────────┤├──┤├──────────────────┤
         站点前缀     子网ID    接口标识符
```

**链路本地地址（Link-Local）— fe80::/10**：

```
链路本地地址结构：
┌─────────────┬──────────────────────┬───────────────────────────┐
│ 1111111010  │ 000000...00000000    │      接口标识符 (64位)     │
│ fe80::/10   │ (54 位零填充)        │      (EUI-64/随机)        │
└─────────────┴──────────────────────┴───────────────────────────┘

特点：
- 仅在同一链路（子网）内有效，路由器不会转发
- 每个启用了 IPv6 的接口都必须有一个链路本地地址
- 自动配置，无需 DHCP 或手动配置
- 用于 NDP、路由协议邻居建立等链路本地通信
```

**唯一本地地址（Unique Local Address）— fc00::/7**：

等同于 IPv4 的私有地址（10.0.0.0/8 等），但设计上更强调全球唯一性：

| 前缀 | 说明 |
|------|------|
| fc00::/8 | 保留，未来可能由中央分配（目前未使用） |
| fd00::/8 | 本地生成，前 40 位随机（全局 ID） |
| fe80::/10 | 链路本地 |
| ff00::/8 | 组播 |

```
唯一本地地址结构 (fd00::/8)：
┌──────┬──────────────┬────────────────┬──────────────────────────┐
│ 1111│ 全局 ID       │  子网 ID        │      接口标识符           │
│ 1101 │ (40位随机)   │  (16位)        │      (64位)              │
│ fd   │              │                │                          │
└──────┴──────────────┴────────────────┴──────────────────────────┘

示例生成：
  全局 ID（随机）: x1:2345:6789
  ULA 前缀: fd01:2345:6789::/48
```

**组播地址（Multicast）— ff00::/8**：

```
组播地址结构：
┌──────┬─────┬─────┬───────────────────────────────────────────────┐
│ 1111│ Flags │Scope│              Group ID                         │
│ 1111│  4bit│4bit │              112 bit                          │
│ ff   │      │     │                                               │
└──────┴─────┴─────┴───────────────────────────────────────────────┘

Flags 字段：
  0 = 永久（well-known）组播地址
  1 = 临时（transient）组播地址

Scope 字段：
  1 = 接口本地（loopback）
  2 = 链路本地（link-local）
  5 = 站点本地（site-local）
  8 = 组织本地（organization）
  E = 全球（global）

常用组播地址：
┌────────────────────────────┬──────────────────────────┐
│ 地址                        │ 说明                     │
├────────────────────────────┼──────────────────────────┤
│ ff02::1                    │ 所有节点（链路本地）       │
│ ff02::2                    │ 所有路由器（链路本地）     │
│ ff02::fb                   │ mDNS 组播地址             │
│ ff02::1:ff00:0/104         │ 请求节点组播（Solicited） │
│ ff05::1:3                  │ 所有 DHCP 服务器（站点）  │
└────────────────────────────┴──────────────────────────┘
```

**任播地址（Anycast）**：

IPv6 中任播地址与单播地址使用相同的地址空间，没有专门的任播前缀。任播的语义是：发送到任播地址的数据报被路由到"最近的"接口（由路由协议决定）。

```
任播工作原理：
┌────────┐      ┌────────────┐
│  客户端  │─────▶│ 路由器     │
└────────┘      └──────┬─────┘
                       │  路由协议选择最近的服务器
              ┌────────┼────────┐
              ▼        ▼        ▼
         ┌────────┐┌────────┐┌────────┐
         │服务器 A ││服务器 B ││服务器 C │
         │东京    ││新加坡   ││法兰克福 │
         └────────┘└────────┘└────────┘
         三台服务器使用相同的任播地址
```

### 30.4.3 特殊地址

| 地址 | CIDR | 用途 |
|------|------|------|
| :: | ::/128 | 未指定地址，用于 DAD 等场景 |
| ::1 | ::1/128 | 环回地址，等同于 IPv4 的 127.0.0.1 |
| fe80::/10 | — | 链路本地地址前缀 |
| fc00::/7 | — | 唯一本地地址前缀 |
| 2001:db8::/32 | — | 文档示例前缀（RFC 3849），不用于实际网络 |
| ::ffff:0:0/96 | — | IPv4 映射地址，用于 IPv4/IPv6 兼容 |
| 64:ff9b::/96 | — | Well-Known NAT64 前缀（RFC 6052） |
| 2001::/23 | — | IANA ORCHIDv2，用于加密哈希标识符 |

### 30.4.4 地址分配层次

```
全球 IPv6 地址分配层次：
┌─────────────────────────────────────────────────────────────┐
│                        IANA                                  │
│              分配 /12 块给 5 个 RIR                           │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   ┌─────────┐     ┌─────────┐     ┌─────────┐
   │  ARIN   │     │  RIPE   │     │ APNIC   │  ...
   │ 北美    │     │ 欧洲    │     │ 亚太    │
   └────┬────┘     └────┬────┘     └────┬────┘
        │               │               │
        ▼               ▼               ▼
   ┌─────────┐     ┌─────────┐     ┌─────────┐
   │   ISP   │     │   ISP   │     │   ISP   │
   │ /32 块  │     │ /32 块  │     │ /32 块  │
   └────┬────┘     └────┬────┘     └────┬────┘
        │               │               │
        ▼               ▼               ▼
   ┌─────────┐     ┌─────────┐     ┌─────────┐
   │ 企业/组织│     │ 企业/组织│     │ 企业/组织│
   │ /48 块  │     │ /48 块  │     │ /48 块  │
   └────┬────┘     └────┬────┘     └────┬────┘
        │               │               │
        ▼               ▼               ▼
   ┌─────────────────────────────────────────┐
   │           子网 (/64)                     │
   │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐      │
   │  │子网 1│ │子网 2│ │子网 3│ │子网 4│ ... │
   │  └─────┘ └─────┘ └─────┘ └─────┘      │
   └─────────────────────────────────────────┘
```

---

## 30.5 ICMPv6 与邻居发现协议（NDP）

### 30.5.1 ICMPv6 概述

ICMPv6（RFC 4443）是 IPv6 的控制报文协议，除了承担 IPv4 中 ICMP 的差错报告功能，还集成了 IGMP（组播管理）和 ARP（地址解析）的功能。

```
ICMPv6 报文类型：
┌──────────────────┬────────────┬──────────────────────────────┐
│ 类型值            │ 类别       │ 说明                          │
├──────────────────┼────────────┼──────────────────────────────┤
│ 1                │ 差错报文    │ 目的不可达                     │
│ 2                │ 差错报文    │ 数据报过大（用于 PMTU 发现）    │
│ 3                │ 差错报文    │ 超时                          │
│ 4                │ 差错报文    │ 参数问题                      │
│ 128              │ 信息报文    │ Echo Request（ping）          │
│ 129              │ 信息报文    │ Echo Reply（ping reply）      │
│ 130              │ 信息报文    │ 组播侦听查询（MLD）            │
│ 133              │ 信息报文    │ 路由器请求（RS）               │
│ 134              │ 信息报文    │ 路由器通告（RA）               │
│ 135              │ 信息报文    │ 邻居请求（NS）                 │
│ 136              │ 信息报文    │ 邻居通告（NA）                 │
│ 137              │ 信息报文    │ 重定向                        │
└──────────────────┴────────────┴──────────────────────────────┘
```

### 30.5.2 NDP 五种消息

**邻居发现协议**（NDP, Neighbor Discovery Protocol, RFC 4861）使用五种 ICMPv6 消息：

**1. 路由器请求（Router Solicitation, RS）— 类型 133**：

```
主机启动时发送 RS，请求路由器发送 RA：

 主机 ──── RS (ICMPv6 Type 133) ────▶ 路由器
         源地址: fe80::主机链路本地地址
         目的地址: ff02::2 (所有路由器)
```

**2. 路由器通告（Router Advertisement, RA）— 类型 134**：

```
路由器周期性或响应 RS 发送 RA：

 路由器 ──── RA (ICMPv6 Type 134) ────▶ 所有节点
          源地址: fe80::路由器链路本地地址
          目的地址: ff02::1 (所有节点)

RA 携带的关键信息：
┌────────────────────────────────────────────────────────┐
│ RA 报文                                                │
│ ├─ Cur Hop Limit: 64                                   │
│ ├─ M Flag: 0 (使用 SLAAC) / 1 (使用 DHCPv6)            │
│ ├─ O Flag: 0 / 1 (是否使用 DHCPv6 获取其他配置)         │
│ ├─ Router Lifetime: 1800 秒                            │
│ ├─ Reachable Time: 30000 毫秒                          │
│ ├─ Retrans Timer: 1000 毫秒                            │
│ ├─ [选项] 前缀信息 (Prefix Information)                 │
│ │   ├─ 前缀: 2001:db8:1::/64                           │
│ │   ├─ L Flag: 1 (可用于链路上通信)                     │
│ │   └─ A Flag: 1 (可用于 SLAAC)                        │
│ ├─ [选项] MTU                                          │
│ └─ [选项] 源链路层地址                                   │
└────────────────────────────────────────────────────────┘
```

**3. 邻居请求（Neighbor Solicitation, NS）— 类型 135**：

```
主机发送 NS 以解析 IPv6 地址对应的链路层地址（类似 ARP 请求）：

 主机A ──── NS (ICMPv6 Type 135) ────▶ 组播
         源地址: fe80::A
         目的地址: ff02::1:ff00:5 (请求节点组播地址)
         目标地址: 2001:db8::5 (要解析的 IPv6 地址)

请求节点组播地址生成：
  目标地址: 2001:db8::5
  后 24 位: 00:00:05
  请求节点组播: ff02::1:ff00:5
```

**4. 邻居通告（Neighbor Advertisement, NA）— 类型 136**：

```
主机响应 NS 或主动发送 NA：

 主机B ──── NA (ICMPv6 Type 136) ────▶ 主机A
         源地址: 2001:db8::5
         目的地址: fe80::A
         R Flag: 路由器标志
         S Flag: 请求标志（响应 NS 时置 1）
         O Flag: 覆盖标志
         目标链路层地址: 00:11:22:33:44:05
```

**5. 重定向（Redirect）— 类型 137**：

```
路由器通知主机更好的下一跳：

 路由器 ──── Redirect ────▶ 主机
          告诉主机："到 2001:db8::/64 的流量应直接发给
          fe80::gateway，而不是经过我"
```

### 30.5.3 邻居缓存状态机（核心组件）

NDP 的邻居缓存维护每个邻居的可达性状态，状态机转换如下：

```
邻居缓存状态机：

                    ┌─────────────────┐
                    │   INCOMPLETE     │ ← 已发送 NS，等待 NA
                    │ (地址解析进行中)  │
                    └───────┬─────────┘
                            │ 收到 NA
                            ▼
                    ┌─────────────────┐
                    │   REACHABLE      │ ← 邻居可达（确认）
                    │ (邻居可达)       │
                    └───────┬─────────┘
                            │ ReachableTimer 超时
                            ▼
                    ┌─────────────────┐
            ┌──────│     STALE        │ ← 可达性未知
            │      │ (过期/不确定)     │
            │      └─────────────────┘
            │              │
            │   有数据要发送│
            │              ▼
            │      ┌─────────────────┐
            │      │     DELAY        │ ← 等待上层协议确认
            │      │ (延迟探测)       │   (DELAY_FIRST_PROBE_TIME = 5s)
            │      └───────┬─────────┘
            │              │ DELAY_FIRST_PROBE_TIME 超时
            │              ▼
            │      ┌─────────────────┐
            │      │     PROBE        │ ← 主动发送 NS 验证
            │      │ (探测中)         │
            │      └───────┬─────────┘
            │              │ 收到 NA → 回到 REACHABLE
            │              │ 达到最大重试次数 → 丢弃条目
            │              ▼
            │        [条目删除]
            │
            │ 收到 NS/NA 更新
            └──────▶ 更新链路层地址 → 回到 STALE
```

**状态转换详细分析**：

| 当前状态 | 触发事件 | 目标状态 | 动作 |
|----------|----------|----------|------|
| — | 首次发送数据 | INCOMPLETE | 发送 NS，启动重传定时器 |
| INCOMPLETE | 收到 NA（单播） | REACHABLE | 记录链路层地址，发送排队数据 |
| INCOMPLETE | 重传超时 | INCOMPLETE | 重发 NS，超过 MaxMulticastSolicit → 删除 |
| REACHABLE | ReachableTimer 超时 | STALE | 标记为过期 |
| REACHABLE | 收到 NA（不同链路层地址） | STALE | 更新链路层地址 |
| STALE | 有数据要发送 | DELAY | 启动 DELAY_FIRST_PROBE_TIME 定时器 |
| DELAY | DELAY_FIRST_PROBE_TIME 超时 | PROBE | 发送 NS（单播） |
| PROBE | 收到 NA | REACHABLE | 更新缓存 |
| PROBE | 重传超时 | PROBE | 重发 NS，超过 MaxUnicastSolicit → 删除 |

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable
import time

class NDState(Enum):
    INCOMPLETE = "INCOMPLETE"
    REACHABLE = "REACHABLE"
    STALE = "STALE"
    DELAY = "DELAY"
    PROBE = "PROBE"

@dataclass
class NeighborEntry:
    ipv6_addr: str
    ll_addr: Optional[str] = None  # 链路层地址
    state: NDState = NDState.INCOMPLETE
    is_router: bool = False
    last_confirmed: float = 0.0
    probe_count: int = 0
    queued_packets: list = field(default_factory=list)

class NDPStateMachine:
    """NDP 邻居缓存状态机"""
    
    REACHABLE_TIME = 30.0           # 默认可达时间（秒）
    DELAY_FIRST_PROBE_TIME = 5.0    # DELAY 状态等待时间
    MAX_MULTICAST_SOLICIT = 3       # 组播 NS 最大重试次数
    MAX_UNICAST_SOLICIT = 3         # 单播 NS 最大重试次数
    
    def __init(self):
        self.cache: dict[str, NeighborEntry] = {}
        self.ns_sender: Optional[Callable] = None  # 发送 NS 的回调
    
    def lookup(self, ipv6_addr: str) -> Optional[NeighborEntry]:
        return self.cache.get(ipv6_addr)
    
    def start_resolution(self, ipv6_addr: str, packet: bytes):
        """开始地址解析（首次发送数据时触发）"""
        entry = NeighborEntry(
            ipv6_addr=ipv6_addr,
            state=NDState.INCOMPLETE,
            queued_packets=[packet],
            last_confirmed=time.time()
        )
        self.cache[ipv6_addr] = entry
        self._send_ns(ipv6_addr, multicast=True)
    
    def receive_na(self, ipv6_addr: str, ll_addr: str, 
                   solicited: bool, router: bool, override: bool):
        """处理收到的邻居通告（NA）"""
        entry = self.cache.get(ipv6_addr)
        if entry is None:
            return
        
        old_state = entry.state
        
        if entry.state == NDState.INCOMPLETE:
            # 状态转换：INCOMPLETE → REACHABLE
            entry.ll_addr = ll_addr
            entry.state = NDState.REACHABLE
            entry.last_confirmed = time.time()
            entry.is_router = router
            # 发送之前排队的数据包
            for pkt in entry.queued_packets:
                self._send_queued_packet(pkt, ll_addr)
            entry.queued_packets.clear()
        
        elif entry.state in (NDState.REACHABLE, NDState.STALE,
                              NDState.DELAY, NDState.PROBE):
            if entry.ll_addr != ll_addr:
                if override or entry.state == NDState.STALE:
                    entry.ll_addr = ll_addr
                if solicited:
                    entry.state = NDState.REACHABLE
                    entry.last_confirmed = time.time()
                else:
                    entry.state = NDState.STALE
            elif solicited:
                entry.state = NDState.REACHABLE
                entry.last_confirmed = time.time()
            
            entry.is_router = router
        
        print(f"NA 处理: {ipv6_addr} {old_state} → {entry.state}")
    
    def on_reachable_timeout(self, ipv6_addr: str):
        """可达性定时器超时"""
        entry = self.cache.get(ipv6_addr)
        if entry and entry.state == NDState.REACHABLE:
            entry.state = NDState.STALE
            print(f"可达超时: {ipv6_addr} → STALE")
    
    def on_data_send(self, ipv6_addr: str):
        """有数据要发送给邻居"""
        entry = self.cache.get(ipv6_addr)
        if entry is None:
            self.start_resolution(ipv6_addr, b'')
            return
        
        if entry.state == NDState.STALE:
            entry.state = NDState.DELAY
            print(f"有数据发送: {ipv6_addr} STALE → DELAY")
    
    def on_delay_timeout(self, ipv6_addr: str):
        """DELAY 定时器超时"""
        entry = self.cache.get(ipv6_addr)
        if entry and entry.state == NDState.DELAY:
            entry.state = NDState.PROBE
            entry.probe_count = 0
            self._send_ns(ipv6_addr, multicast=False)
            print(f"DELAY 超时: {ipv6_addr} → PROBE")
    
    def on_probe_timeout(self, ipv6_addr: str):
        """探测超时"""
        entry = self.cache.get(ipv6_addr)
        if entry and entry.state == NDState.PROBE:
            entry.probe_count += 1
            if entry.probe_count >= self.MAX_UNICAST_SOLICIT:
                del self.cache[ipv6_addr]
                print(f"探测失败: {ipv6_addr} 已删除")
            else:
                self._send_ns(ipv6_addr, multicast=False)
    
    def _send_ns(self, ipv6_addr: str, multicast: bool):
        """发送邻居请求"""
        if self.ns_sender:
            self.ns_sender(ipv6_addr, multicast)
    
    def _send_queued_packet(self, packet: bytes, ll_addr: str):
        """发送排队的数据包"""
        pass  # 调用链路层发送

    def print_cache(self):
        """打印缓存状态"""
        print(f"{'IPv6 地址':<40} {'链路层地址':<20} {'状态':<12} {'路由器'}")
        print("-" * 85)
        for addr, entry in self.cache.items():
            ll = entry.ll_addr or "(未解析)"
            print(f"{addr:<40} {ll:<20} {entry.state.value:<12} {entry.is_router}")
```

<div data-component="NDPStateDiagram"></div>

### 30.5.4 DAD 流程

**重复地址检测**（DAD, Duplicate Address Detection, RFC 4862）确保新配置的地址在链路上唯一：

```
DAD 流程：
时间 ──────────────────────────────────────────────────────────────▶

 主机A (新加入)                       主机B (已使用该地址)
     │                                      │
     │  1. 发送 NS (目标=待检测地址,           │
     │     源=::(未指定),                     │
     │     目的=ff02::1:ffXX:XXXX)           │
     ├─────────────────────────────────────▶│
     │                                      │
     │  2. 等待 RETRANS_TIMER (1秒)         │
     │     重复发送 NS (通常 1 次)           │
     ├─────────────────────────────────────▶│
     │                                      │
     │                           如果 B 使用该地址:
     │  3. B 发送 NA (目标=该地址,             │
     │     目的=ff02::1)                     │
     │◀─────────────────────────────────────┤
     │                                      │
     │  4. A 收到 NA → 地址冲突!              │
     │     放弃使用该地址                     │
     │                                      │
     │  如果未收到任何 NA:                    │
     │  5. 地址可用，进入 REACHABLE 状态       │
     │     (DAD 通过)                        │
```

**DAD 的关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| DupAddrDetectTransmits | 1 | DAD 发送 NS 的次数 |
| RETRANS_TIMER | 1000 ms | NS 重传间隔 |
| 临时地址状态 | TENTATIVE | DAD 完成前地址不可用于通信 |

---

## 30.6 IPv6 地址自动配置

### 30.6.1 SLAAC 完整流程

**无状态地址自动配置**（SLAAC, Stateless Address Autoconfiguration, RFC 4862）允许主机无需任何服务器即可获得全球可路由的 IPv6 地址：

```
SLAAC 完整流程：

 主机                              路由器
  │                                  │
  │  1. 生成链路本地地址              │
  │  (fe80::EUI-64)                  │
  │  执行 DAD                        │
  ├──────── NS (DAD) ───────────────▶│
  │◀─────── NA(冲突) 或 无回复 ──────┤
  │                                  │
  │  2. DAD 通过，链路本地地址可用     │
  │                                  │
  │  3. 发送路由器请求 (RS)           │
  ├──────── RS ─────────────────────▶│
  │                                  │
  │  4. 收到路由器通告 (RA)           │
  │◀──────── RA (含前缀信息) ────────┤
  │     前缀: 2001:db8:1::/64        │
  │     A Flag: 1                    │
  │     L Flag: 1                    │
  │     Valid Lifetime: 2592000      │
  │     Preferred Lifetime: 604800   │
  │                                  │
  │  5. 用 RA 中的前缀 + 接口标识符   │
  │     生成全球单播地址               │
  │     2001:db8:1::EUI-64           │
  │                                  │
  │  6. 对全球地址执行 DAD             │
  ├──────── NS (DAD) ───────────────▶│
  │◀─────── 无回复（地址可用）────────┤
  │                                  │
  │  7. 全球单播地址可用               │
  │     开始正常通信                   │
```

### 30.6.2 EUI-64 接口标识符

EUI-64 从 48 位 MAC 地址生成 64 位接口标识符：

```
EUI-64 生成过程：

MAC 地址:     00:1A:2B:3C:4D:5E

步骤 1: 在中间插入 FF:FE
          00:1A:2B:FF:FE:3C:4D:5E

步骤 2: 翻转第 7 位（U/L 位）
          第 1 字节: 00 = 0000 0000
          翻转第 7 位: 0000 0010 = 02
          结果: 02:1A:2B:FF:FE:3C:4D:5E

步骤 3: 转换为 IPv6 接口标识符格式
          ::021A:2BFF:FE3C:4D5E

完整地址:
          fe80::021a:2bff:fe3c:4d5e
          简写: fe80::21a:2bff:fe3c:4d5e
```

```python
def mac_to_eui64(mac: str, prefix: str = "fe80::") -> str:
    """将 MAC 地址转换为 EUI-64 格式的 IPv6 地址"""
    # 去除分隔符并转为字节
    mac_bytes = bytes.fromhex(mac.replace(':', '').replace('-', ''))
    
    # 步骤 1: 在第 3 和第 4 字节之间插入 FF:FE
    eui64 = mac_bytes[:3] + b'\xff\xfe' + mac_bytes[3:]
    
    # 步骤 2: 翻转第 7 位（U/L 位）
    eui64_list = list(eui64)
    eui64_list[0] ^= 0x02
    eui64 = bytes(eui64_list)
    
    # 步骤 3: 格式化为 IPv6 接口标识符
    interface_id = '{:04x}:{:04x}:{:04x}:{:04x}'.format(
        int.from_bytes(eui64[0:2], 'big'),
        int.from_bytes(eui64[2:4], 'big'),
        int.from_bytes(eui64[4:6], 'big'),
        int.from_bytes(eui64[6:8], 'big'),
    )
    
    return f"{prefix}{interface_id}"

# 示例
mac = "00:1A:2B:3C:4D:5E"
result = mac_to_eui64(mac)
print(f"MAC: {mac}")
print(f"IPv6: {result}")
# 输出: IPv6: fe80::021a:2bff:fe3c:4d5e

# 使用全球前缀
global_addr = mac_to_eui64(mac, prefix="2001:db8:1::")
print(f"全球地址: {global_addr}")
# 输出: 全球地址: 2001:db8:1::021a:2bff:fe3c:4d5e
```

### 30.6.3 隐私扩展

EUI-64 接口标识符直接暴露 MAC 地址，存在**隐私泄露风险**——用户的活动可跨网络追踪。RFC 4941 定义了隐私扩展：

```
隐私扩展地址生成：
┌─────────────────────────────────────────────────────────────────┐
│  固定接口标识符 (EUI-64)             临时接口标识符 (随机)         │
│                                                                 │
│  2001:db8:1::021a:2bff:fe3c:4d5e   2001:db8:1::a3f2:9b1c:d847:│
│      ↑                                    ↑                     │
│  可追踪，不变                        随机生成，定期更换             │
│  适合服务器（DNS 需要稳定地址）       适合客户端（保护隐私）         │
└─────────────────────────────────────────────────────────────────┘

地址生命周期：
|←── Preferred Lifetime ──▶|◀── Deprecated ──▶|◀── Valid Lifetime ──▶|
     (正常使用的地址)          (仍可用但不推荐)      (地址失效)

新地址在 Preferred Lifetime 到期前生成，实现无缝切换。
```

```python
import os
import struct
import time

class PrivacyAddressGenerator:
    """RFC 4941 隐私扩展地址生成器"""
    
    def __init__(self, prefix: str, preferred_lifetime: int = 86400,
                 valid_lifetime: int = 604800):
        self.prefix = prefix
        self.preferred_lifetime = preferred_lifetime  # 默认 1 天
        self.valid_lifetime = valid_lifetime           # 默认 7 天
        self.history_iid = None  # 上一个接口标识符（防重复）
    
    def generate_temporary_address(self) -> tuple:
        """生成临时隐私地址"""
        while True:
            # 生成 64 位随机接口标识符
            iid_bytes = os.urandom(8)
            iid = struct.unpack('>Q', iid_bytes)[0]
            
            # 确保与上一个地址不同（RFC 4941 建议）
            if iid != self.history_iid:
                self.history_iid = iid
                break
        
        # 格式化为 IPv6 地址
        iid_hex = f'{iid:016x}'
        iid_formatted = ':'.join(
            iid_hex[i:i+4] for i in range(0, 16, 4)
        )
        
        full_addr = f"{self.prefix}:{iid_formatted}"
        creation_time = time.time()
        
        return full_addr, creation_time
    
    def get_address_lifecycle(self, creation_time: float) -> dict:
        """计算地址的当前生命周期状态"""
        age = time.time() - creation_time
        
        if age < self.preferred_lifetime:
            status = "preferred"
            remaining = self.preferred_lifetime - age
        elif age < self.valid_lifetime:
            status = "deprecated"
            remaining = self.valid_lifetime - age
        else:
            status = "invalid"
            remaining = 0
        
        return {
            'status': status,
            'age_seconds': age,
            'remaining_seconds': remaining,
        }

# 示例
gen = PrivacyAddressGenerator(prefix="2001:db8:1")
for i in range(3):
    addr, ts = gen.generate_temporary_address()
    print(f"临时地址 {i+1}: {addr}")
```

### 30.6.4 DHCPv6 有状态与无状态

| 模式 | 触发条件 | 功能 | 特点 |
|------|----------|------|------|
| SLAAC | RA 的 A Flag=1 | 地址配置 | 无服务器，前缀+接口标识符 |
| SLAAC + 无状态 DHCPv6 | RA 的 O Flag=1 | SLAAC 配地址 + DHCPv6 获取其他信息 | 获取 DNS 等，不分配地址 |
| 有状态 DHCPv6 | RA 的 M Flag=1 | 完全由 DHCPv6 服务器分配 | 服务器管理地址和配置 |
| DHCPv6 PD | 客户端请求 | 前缀委派 | ISP 分配子网前缀给用户 |

```
SLAAC vs DHCPv6 对比：
┌─────────────────┬────────────────────┬────────────────────┐
│ 特性             │ SLAAC             │ 有状态 DHCPv6      │
├─────────────────┼────────────────────┼────────────────────┤
│ 地址分配方式      │ 前缀 + 自生成ID    │ 服务器全权分配      │
│ 服务器需求        │ 不需要             │ 需要 DHCPv6 服务器  │
│ DNS 服务器获取    │ 需要无状态DHCPv6   │ 直接提供            │
│ 地址管理可见性    │ 低                 │ 高（服务器可审计）   │
│ 部署复杂度        │ 低                 │ 中等                │
│ 适用场景          │ 家庭/小型网络      │ 企业/运营商网络      │
│ 地址稳定性        │ 可使用隐私扩展     │ 稳定（服务器分配）   │
└─────────────────┴────────────────────┴────────────────────┘
```

### 30.6.5 SLAAC 地址自动配置器（核心组件）

```python
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
import time
import os
import hashlib

class AutoConfigMethod(Enum):
    EUI64 = "eui64"
    PRIVACY = "privacy"
    STATIC = "static"
    DHCPV6 = "dhcpv6"

@dataclass
class PrefixInfo:
    prefix: str
    prefix_length: int
    on_link: bool          # L Flag
    autonomous: bool       # A Flag
    valid_lifetime: int
    preferred_lifetime: int
    received_time: float = field(default_factory=time.time)

@dataclass
class ConfiguredAddress:
    address: str
    prefix_info: Optional[PrefixInfo]
    method: AutoConfigMethod
    creation_time: float
    state: str = "tentative"  # tentative, preferred, deprecated, invalid
    dad_completed: bool = False

class SLAACAddressGenerator:
    """SLAAC 地址自动配置器"""
    
    DAD_ATTEMPTS = 1
    DAD_TIMEOUT = 1.0  # 秒
    
    def __init__(self, link_local_addr: str, mac_addr: str):
        self.link_local_addr = link_local_addr
        self.mac_addr = mac_addr
        self.configured_addresses: List[ConfiguredAddress] = []
        self.prefix_cache: List[PrefixInfo] = []
        self.stats = {
            'addresses_generated': 0,
            'dad_successes': 0,
            'dad_failures': 0,
            'privacy_rotations': 0,
        }
    
    def process_ra_prefix_option(self, prefix: str, prefix_len: int,
                                  on_link: bool, autonomous: bool,
                                  valid_lifetime: int,
                                  preferred_lifetime: int):
        """处理 RA 中的前缀信息选项"""
        pi = PrefixInfo(
            prefix=prefix,
            prefix_length=prefix_len,
            on_link=on_link,
            autonomous=autonomous,
            valid_lifetime=valid_lifetime,
            preferred_lifetime=preferred_lifetime,
        )
        
        # 更新前缀缓存
        self._update_prefix_cache(pi)
        
        # 如果 A Flag 设置且前缀长度为 64，执行 SLAAC
        if autonomous and prefix_len == 64:
            addr = self._generate_from_prefix(pi)
            if addr:
                self.configured_addresses.append(addr)
                self.stats['addresses_generated'] += 1
                return addr
        
        return None
    
    def _generate_from_prefix(self, pi: PrefixInfo) -> ConfiguredAddress:
        """从前缀生成地址"""
        # 使用 EUI-64 生成接口标识符
        interface_id = self._mac_to_eui64_iid()
        full_addr = f"{pi.prefix}::{interface_id}"
        
        # 执行 DAD
        if self._perform_dad(full_addr):
            self.stats['dad_successes'] += 1
            return ConfiguredAddress(
                address=full_addr,
                prefix_info=pi,
                method=AutoConfigMethod.EUI64,
                creation_time=time.time(),
                state="preferred",
                dad_completed=True,
            )
        else:
            self.stats['dad_failures'] += 1
            return None
    
    def _mac_to_eui64_iid(self) -> str:
        """MAC 地址转 EUI-64 接口标识符"""
        mac_bytes = bytes.fromhex(
            self.mac_addr.replace(':', '').replace('-', '')
        )
        eui64 = list(mac_bytes[:3] + b'\xff\xfe' + mac_bytes[3:])
        eui64[0] ^= 0x02
        return '{:04x}:{:04x}:{:04x}:{:04x}'.format(
            int.from_bytes(bytes(eui64[0:2]), 'big'),
            int.from_bytes(bytes(eui64[2:4]), 'big'),
            int.from_bytes(bytes(eui64[4:6]), 'big'),
            int.from_bytes(bytes(eui64[6:8]), 'big'),
        )
    
    def generate_privacy_address(self, prefix: str) -> ConfiguredAddress:
        """生成隐私扩展临时地址"""
        iid = os.urandom(8)
        iid_int = int.from_bytes(iid, 'big')
        iid_hex = f'{iid_int:016x}'
        iid_fmt = ':'.join(iid_hex[i:i+4] for i in range(0, 16, 4))
        full_addr = f"{prefix}::{iid_fmt}"
        
        self.stats['privacy_rotations'] += 1
        return ConfiguredAddress(
            address=full_addr,
            prefix_info=None,
            method=AutoConfigMethod.PRIVACY,
            creation_time=time.time(),
            state="preferred",
            dad_completed=True,
        )
    
    def _perform_dad(self, addr: str) -> bool:
        """执行重复地址检测（简化版）"""
        # 实际实现需要发送 NS 并等待 NA
        # 这里简化为始终成功
        return True
    
    def _update_prefix_cache(self, new_prefix: PrefixInfo):
        """更新前缀缓存"""
        for i, existing in enumerate(self.prefix_cache):
            if (existing.prefix == new_prefix.prefix and
                existing.prefix_length == new_prefix.prefix_length):
                self.prefix_cache[i] = new_prefix
                return
        self.prefix_cache.append(new_prefix)
    
    def check_address_lifetimes(self):
        """检查所有地址的生命周期状态"""
        now = time.time()
        for addr in self.configured_addresses:
            if addr.prefix_info is None:
                continue
            age = now - addr.creation_time
            pi = addr.prefix_info
            
            if age < pi.preferred_lifetime:
                addr.state = "preferred"
            elif age < pi.valid_lifetime:
                addr.state = "deprecated"
            else:
                addr.state = "invalid"
    
    def get_active_addresses(self) -> List[str]:
        """获取所有活跃（非失效）地址"""
        self.check_address_lifetimes()
        return [
            a.address for a in self.configured_addresses
            if a.state in ("preferred", "deprecated")
        ]
    
    def print_status(self):
        """打印配置状态"""
        print(f"已配置地址: {len(self.configured_addresses)}")
        print(f"前缀缓存: {len(self.prefix_cache)} 条")
        print(f"生成统计: {self.stats}")
        for addr in self.configured_addresses:
            print(f"  {addr.address} [{addr.state}] "
                  f"via {addr.method.value}")


# 示例使用
slaac = SLAACAddressGenerator(
    link_local_addr="fe80::21a:2bff:fe3c:4d5e",
    mac_addr="00:1A:2B:3C:4D:5E"
)

# 模拟收到 RA 中的前缀信息
slaac.process_ra_prefix_option(
    prefix="2001:db8:1",
    prefix_len=64,
    on_link=True,
    autonomous=True,
    valid_lifetime=2592000,
    preferred_lifetime=604800,
)

# 生成隐私地址
priv_addr = slaac.generate_privacy_address("2001:db8:1")
slaac.configured_addresses.append(priv_addr)

slaac.print_status()
```

<div data-component="SLAACProcessDiagram"></div>

---

## 30.7 IPv4 到 IPv6 过渡技术

### 30.7.1 过渡技术概览

由于 IPv4 和 IPv6 不兼容，整个互联网不可能在一夜之间切换。过渡技术分为三大类：

```
IPv6 过渡技术分类：
┌─────────────────────────────────────────────────────────────────┐
│                      过渡技术体系                                 │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   双栈        │  │    隧道       │  │    翻译       │           │
│  │  (Dual-Stack) │  │  (Tunneling)  │  │ (Translation) │           │
│  │               │  │               │  │               │           │
│  │ 同时运行      │  │ IPv6-in-IPv4  │  │ IPv6 ↔ IPv4  │           │
│  │ IPv4 + IPv6   │  │ 封装传输      │  │ 协议转换       │           │
│  │               │  │               │  │               │           │
│  │ 最基本的过渡  │  │ 连接孤岛      │  │ 互操作         │           │
│  │ 方案          │  │               │  │               │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                  │
│  具体技术：                                                       │
│  双栈: Native Dual-Stack                                         │
│  隧道: 6to4, ISATAP, 6in4, Teredo, GRE, VXLAN                   │
│  翻译: NAT64/DNS64, SIIT, NAT-PT(已废弃), 464XLAT               │
└─────────────────────────────────────────────────────────────────┘
```

### 30.7.2 双栈（Dual-Stack）

双栈是最基本的过渡方案：网络设备和主机同时运行 IPv4 和 IPv6 协议栈。

```
双栈主机协议栈：
┌──────────────────────────────────────────────────┐
│                    应用层                          │
│  ┌──────────────────────────────────────────────┐│
│  │     应用程序 (选择 IPv4 或 IPv6)               ││
│  └──────────────────────────────────────────────┘│
│           │                    │                  │
│           ▼                    ▼                  │
│  ┌──────────────┐    ┌──────────────┐            │
│  │    TCP/UDP    │    │    TCP/UDP    │            │
│  └──────┬───────┘    └──────┬───────┘            │
│         │                    │                    │
│         ▼                    ▼                    │
│  ┌──────────────┐    ┌──────────────┐            │
│  │    IPv4       │    │    IPv6       │            │
│  │  192.168.1.10 │    │ 2001:db8::10  │            │
│  └──────┬───────┘    └──────┬───────┘            │
│         │                    │                    │
│         ▼                    ▼                    │
│  ┌──────────────────────────────────────────────┐│
│  │           链路层 (以太网等)                     ││
│  └──────────────────────────────────────────────┘│
└──────────────────────────────────────────────────┘
```

### 30.7.3 隧道技术

**6to4（RFC 3056）**：

```
6to4 隧道封装：
┌────────────────────────────────────────────────────────────────┐
│  IPv6 数据报                                                    │
│  ┌──────────┬──────────┬─────────────────────────────────────┐ │
│  │ IPv6 头部 │ 扩展头   │          IPv6 载荷                  │ │
│  └──────────┴──────────┴─────────────────────────────────────┘ │
│                         │ 封装                                  │
│                         ▼                                      │
│  IPv4 封装后的数据报                                             │
│  ┌──────────┬──────────┬──────────┬─────────────────────────┐ │
│  │ IPv4 头部 │ IPv6 头部 │ 扩展头   │      IPv6 载荷          │ │
│  │ Protocol=41│          │         │                         │ │
│  └──────────┴──────────┴──────────┴─────────────────────────┘ │
│                                                                │
│  6to4 前缀: 2002:{v4地址}::/48                                 │
│  例如 IPv4=192.0.2.1 → 2002:c000:0201::/48                    │
└────────────────────────────────────────────────────────────────┘
```

**Teredo（RFC 4380）**：

Teredo 在 UDP 载荷中封装 IPv6 数据报，使其能穿越 NAT 设备：

```
Teredo 封装（穿越 NAT）：
┌────────────────────────────────────────────────────────────┐
│ 以太网 │ IPv4 │ UDP │ Teredo │          IPv6 数据报         │
│ 头部   │ 头部  │头部 │  头部   │  ┌────────┬──────────────┐  │
│        │      │     │        │  │IPv6头  │ 载荷          │  │
└────────┴──────┴─────┴────────┴──┴────────┴──────────────┘
                                              ↑
                              Teredo 服务器负责中继
                              穿越 NAT 的 UDP 打洞
```

**ISATAP（RFC 5214）**：

ISATAP 将 IPv6 地址嵌入 IPv4 地址，用于企业内部过渡：

```
ISATAP 接口标识符格式：
┌──────────────────────────────────────────────────┐
│        64 位接口标识符                             │
│  ┌──────────────┬──────────────┐                  │
│  │ 0000:5EFE    │  IPv4 地址    │                  │
│  │ (32位)       │  (32位)       │                  │
│  └──────────────┴──────────────┘                  │
│                                                    │
│  例: IPv4 = 10.0.0.1                              │
│  接口ID = 0000:5EFE:0A00:0001                     │
│  链路本地 = fe80::5efe:a00:1                      │
└──────────────────────────────────────────────────┘
```

| 隧道技术 | 穿越 NAT | 配置方式 | 适用场景 | 状态 |
|----------|----------|----------|----------|------|
| 6in4 | 否 | 手动 | 站点间隧道 | 仍使用 |
| 6to4 | 否 | 自动 | 已弃用 | 衰退 |
| ISATAP | 否 | 自动 | 企业内部 | 少用 |
| Teredo | 是 | 自动 | 穿越 NAT | Windows 支持 |
| GRE | 否 | 手动 | 通用封装 | 仍在使用 |

### 30.7.4 翻译技术

**NAT64/DNS64**：

NAT64 将 IPv6 数据报翻译为 IPv4 数据报，DNS64 合成 AAAA 记录：

```
NAT64/DNS64 工作流程：

1. IPv6 主机查询 DNS64 服务器：
   主机: "请给我 www.example.com 的 AAAA 记录"
   DNS64: 查询权威 DNS，只有 A 记录 (93.184.216.34)
   DNS64: 合成 AAAA 记录 → 64:ff9b::5db8:d822
          (Well-Known 64 前缀 + 嵌入的 IPv4 地址)

2. IPv6 主机发送 IPv6 数据报到 NAT64 网关：
   源: 2001:db8::100
   目的: 64:ff9b::5db8:d822

3. NAT64 网关执行协议转换：
   IPv6 数据报 → IPv4 数据报
   源: 203.0.113.5 (NAT64 地址池)
   目的: 93.184.216.34
```

**SIIT（Stateless IP/ICMP Translation, RFC 7915）**：

无状态翻译不维护会话状态，转换速度快但功能有限：

```
SIIT 地址映射公式：
┌──────────────────────────────────────────────────────────────┐
│  IPv4 → IPv6:                                                │
│    ::ffff:0:{IPv4地址}                                       │
│                                                              │
│  IPv6 → IPv4:                                                │
│    提取后 32 位作为 IPv4 地址                                  │
│    (仅适用于特定前缀内的 IPv6 地址)                            │
│                                                              │
│  Well-Known 前缀: 64:ff9b::/96                               │
│  例: IPv4 93.184.216.34 → 64:ff9b::5db8:d822                │
└──────────────────────────────────────────────────────────────┘
```

### 30.7.5 NAT64 协议转换引擎（核心组件）

NAT64 网关是 IPv6-only 网络访问 IPv4 互联网的关键设备，其核心是**协议转换引擎**：

```
NAT64 协议转换引擎架构：
┌────────────────────────────────────────────────────────────────────┐
│                      NAT64 协议转换引擎                             │
│                                                                     │
│  ┌─────────────┐                    ┌─────────────┐                │
│  │   IPv6 侧   │                    │   IPv4 侧   │                │
│  │   接口       │                    │   接口       │                │
│  └──────┬──────┘                    └──────┬──────┘                │
│         │                                  │                        │
│         ▼                                  ▼                        │
│  ┌──────────────────┐              ┌──────────────────┐            │
│  │ IPv6 报文解析器   │              │ IPv4 报文解析器   │            │
│  └────────┬─────────┘              └────────┬─────────┘            │
│           │                                 │                       │
│           ▼                                 ▼                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    地址转换引擎                                │  │
│  │  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐   │  │
│  │  │ IPv6→IPv4  │  │ IPv4→IPv6  │  │ 地址池管理            │   │  │
│  │  │ 头部转换   │  │ 头部转换   │  │ (IPv4 地址池)        │   │  │
│  │  └────────────┘  └────────────┘  └──────────────────────┘   │  │
│  └──────────────────────────┬───────────────────────────────────┘  │
│                             │                                      │
│                             ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    传输层转换引擎                               │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐             │  │
│  │  │ TCP 状态机  │  │ UDP 转换   │  │ ICMP 转换  │             │  │
│  │  │ 转换       │  │ (无状态)   │  │            │             │  │
│  │  └────────────┘  └────────────┘  └────────────┘             │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                             │                                      │
│                             ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    会话表管理                                  │  │
│  │  ┌──────────────────────────────────────────────────────┐   │  │
│  │  │ Session Table                                        │   │  │
│  │  │ ┌──────────────┬──────────────┬──────────┬────────┐ │   │  │
│  │  │ │ IPv6 5-tuple │ IPv4 5-tuple │ Proto    │ Timer  │ │   │  │
│  │  │ └──────────────┴──────────────┴──────────┴────────┘ │   │  │
│  │  └──────────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

**头部转换详细过程**：

```
IPv6 → IPv4 头部转换：
┌─────────────────────────────────────────────────────────────┐
│ 输入 IPv6 头部                                               │
│ ┌─────┬─────────┬─────────┬───────┬──────┬──────┬─────────┐│
│ │V=6  │TC=0x00  │FL=0     │PL=100 │NH=6  │HL=64 │S:2001:  ││
│ │     │         │         │       │(TCP) │      │db8::1   ││
│ └─────┴─────────┴─────────┴───────┴──────┴──────┴─────────┘│
│ ┌─────────────────────────────────────────────────────────┐│
│ │D: 64:ff9b::5db8:d822                                    ││
│ └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│ 转换规则：                                                   │
│  Version: 6 → 4                                             │
│  Traffic Class → TOS (DSCP + ECN)                           │
│  Flow Label → 丢弃（IPv4 无此字段）                           │
│  Payload Length + 扩展头长度 → Total Length                   │
│  Next Header (TCP=6) → Protocol (TCP=6)                     │
│  Hop Limit → TTL                                            │
│  源地址 2001:db8::1 → 203.0.113.10 (从地址池分配)            │
│  目的地址 64:ff9b::5db8:d822 → 93.184.216.34 (提取后32位)    │
├─────────────────────────────────────────────────────────────┤
│ 输出 IPv4 头部                                               │
│ ┌─────┬─────┬──────┬───────┬──┬──┬──────┬─────────────────┐│
│ │V=4  │IHL=5│TOS=0 │TL=120 │ID│FL│TTL=64│Proto=6(TCP)     ││
│ └─────┴─────┴──────┴───────┘  └──┴──────┴─────────────────┘│
│ ┌──────────────────────┬──────────────────────┐             │
│ │Src: 203.0.113.10     │Dst: 93.184.216.34    │             │
│ └──────────────────────┴──────────────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

**传输层转换的特殊处理**：

| 协议 | 转换难点 | 处理方式 |
|------|----------|----------|
| TCP | 校验和包含伪头部（含 IP 地址） | 重新计算校验和 |
| UDP | 同 TCP，校验和需要重算 | 重新计算；UDP 校验和为 0 时特殊处理 |
| ICMPv6↔ICMP | 类型/代码映射 | 逐类型映射（如 Echo 直接映射，不可达需要转换） |
| 分片 | IPv6 只在源端分片 | 需要处理 IPv4 MTU 更小导致的二次分片 |

```python
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import struct
import time
import hashlib

@dataclass
class IPv6Tuple:
    src_addr: str
    dst_addr: str
    src_port: int
    dst_port: int
    protocol: int  # 6=TCP, 17=UDP

@dataclass
class IPv4Tuple:
    src_addr: str
    dst_addr: str
    src_port: int
    dst_port: int
    protocol: int

@dataclass
class SessionEntry:
    ipv6_tuple: IPv6Tuple
    ipv4_tuple: IPv4Tuple
    created: float
    last_activity: float
    state: str = "ESTABLISHED"
    packets_translated: int = 0
    bytes_translated: int = 0

class NAT64Engine:
    """NAT64 协议转换引擎"""
    
    POOL_PREFIX = "64:ff9b::"
    SESSION_TIMEOUT = 300  # 5 分钟
    
    def __init__(self, ipv4_pool: list, local_prefix: str):
        self.ipv4_pool = ipv4_pool           # ["203.0.113.1", ...]
        self.local_prefix = local_prefix     # "2001:db8:1::"
        self.sessions: Dict[Tuple, SessionEntry] = {}
        self.port_allocations: Dict[str, set] = {}
        self.stats = {
            'v6_to_v4_packets': 0,
            'v4_to_v6_packets': 0,
            'sessions_created': 0,
            'sessions_expired': 0,
            'translation_errors': 0,
        }
    
    def translate_v6_to_v4(self, pkt: dict) -> Optional[dict]:
        """将 IPv6 数据报转换为 IPv4 数据报"""
        self.stats['v6_to_v4_packets'] += 1
        
        v6_src = pkt['src_addr']
        v6_dst = pkt['dst_addr']
        src_port = pkt.get('src_port', 0)
        dst_port = pkt.get('dst_port', 0)
        protocol = pkt.get('protocol', 6)
        
        # 查找或创建会话
        v6_key = (v6_src, v6_dst, src_port, dst_port, protocol)
        session = self.sessions.get(v6_key)
        
        if session is None:
            # 创建新会话，分配 IPv4 地址和端口
            session = self._create_session(v6_src, v6_dst,
                                            src_port, dst_port, protocol)
            if session is None:
                self.stats['translation_errors'] += 1
                return None
        
        # 更新会话状态
        session.last_activity = time.time()
        session.packets_translated += 1
        session.bytes_translated += pkt.get('payload_len', 0)
        
        # 构造 IPv4 数据报
        v4_pkt = {
            'version': 4,
            'ihl': 5,
            'tos': pkt.get('traffic_class', 0),
            'total_length': pkt.get('payload_len', 0) + 20,
            'ttl': pkt.get('hop_limit', 64),
            'protocol': protocol,
            'src_addr': session.ipv4_tuple.src_addr,
            'dst_addr': session.ipv4_tuple.dst_addr,
            'src_port': session.ipv4_tuple.src_port,
            'dst_port': session.ipv4_tuple.dst_port,
            'payload': pkt.get('payload', b''),
        }
        
        # 重新计算校验和
        v4_pkt['checksum'] = self._compute_ipv4_checksum(v4_pkt)
        v4_pkt['tcp_checksum'] = self._compute_pseudo_checksum(
            session.ipv4_tuple, protocol, pkt.get('payload', b'')
        )
        
        return v4_pkt
    
    def translate_v4_to_v6(self, pkt: dict) -> Optional[dict]:
        """将 IPv4 数据报转换回 IPv6 数据报"""
        self.stats['v4_to_v6_packets'] += 1
        
        v4_dst = pkt['dst_addr']
        dst_port = pkt.get('dst_port', 0)
        
        # 根据目的地址和端口查找会话
        session = None
        for key, entry in self.sessions.items():
            if (entry.ipv4_tuple.dst_addr == v4_dst and
                entry.ipv4_tuple.dst_port == dst_port):
                session = entry
                break
        
        if session is None:
            self.stats['translation_errors'] += 1
            return None
        
        session.last_activity = time.time()
        session.packets_translated += 1
        
        v6_pkt = {
            'version': 6,
            'traffic_class': pkt.get('tos', 0),
            'flow_label': 0,
            'payload_length': pkt.get('total_length', 0) - 20,
            'next_header': pkt.get('protocol', 6),
            'hop_limit': pkt.get('ttl', 64),
            'src_addr': session.ipv6_tuple.dst_addr,  # 反向
            'dst_addr': session.ipv6_tuple.src_addr,
            'src_port': session.ipv6_tuple.dst_port,
            'dst_port': session.ipv6_tuple.src_port,
            'payload': pkt.get('payload', b''),
        }
        
        return v6_pkt
    
    def _create_session(self, v6_src, v6_dst, src_port, 
                        dst_port, protocol) -> Optional[SessionEntry]:
        """创建新的 NAT64 会话"""
        # 分配 IPv4 地址
        v4_src = self._allocate_ipv4_addr()
        if v4_src is None:
            return None
        
        # 分配端口
        v4_src_port = self._allocate_port(v4_src)
        if v4_src_port is None:
            return None
        
        # 提取 IPv4 目的地址（从 NAT64 前缀中）
        v4_dst = self._extract_ipv4_from_nat64(v6_dst)
        if v4_dst is None:
            return None
        
        session = SessionEntry(
            ipv6_tuple=IPv6Tuple(v6_src, v6_dst, src_port, dst_port, protocol),
            ipv4_tuple=IPv4Tuple(v4_src, v4_dst, v4_src_port, dst_port, protocol),
            created=time.time(),
            last_activity=time.time(),
        )
        
        v6_key = (v6_src, v6_dst, src_port, dst_port, protocol)
        self.sessions[v6_key] = session
        self.stats['sessions_created'] += 1
        
        return session
    
    def _allocate_ipv4_addr(self) -> Optional[str]:
        """从地址池分配 IPv4 地址（简单轮询）"""
        if not self.ipv4_pool:
            return None
        # 简单实现：返回第一个可用地址
        for addr in self.ipv4_pool:
            if addr not in self.port_allocations:
                self.port_allocations[addr] = set()
            if len(self.port_allocations[addr]) < 65535:
                return addr
        return None
    
    def _allocate_port(self, ipv4_addr: str) -> Optional[int]:
        """分配可用端口"""
        allocated = self.port_allocations.get(ipv4_addr, set())
        for port in range(1024, 65535):
            if port not in allocated:
                allocated.add(port)
                self.port_allocations[ipv4_addr] = allocated
                return port
        return None
    
    def _extract_ipv4_from_nat64(self, nat64_addr: str) -> Optional[str]:
        """从 NAT64 地址中提取 IPv4 地址"""
        # 64:ff9b::5db8:d822 → 93.184.216.34
        # 简化解析
        if self.POOL_PREFIX.replace('::', '') in nat64_addr:
            parts = nat64_addr.split('::')[-1].split(':')
            if len(parts) == 2:
                high = int(parts[0], 16)
                low = int(parts[1], 16)
                ipv4_int = (high << 16) | low
                return f"{(ipv4_int >> 24) & 0xFF}.{(ipv4_int >> 16) & 0xFF}." \
                       f"{(ipv4_int >> 8) & 0xFF}.{ipv4_int & 0xFF}"
        return None
    
    def _compute_ipv4_checksum(self, pkt: dict) -> int:
        """计算 IPv4 头部校验和"""
        return 0  # 简化实现
    
    def _compute_pseudo_checksum(self, v4_tuple: IPv4Tuple, 
                                  protocol: int, payload: bytes) -> int:
        """计算传输层校验和（含伪头部）"""
        return 0  # 简化实现
    
    def cleanup_expired_sessions(self):
        """清理过期会话"""
        now = time.time()
        expired = [
            key for key, entry in self.sessions.items()
            if now - entry.last_activity > self.SESSION_TIMEOUT
        ]
        for key in expired:
            entry = self.sessions.pop(key)
            # 释放端口
            addr = entry.ipv4_tuple.src_addr
            port = entry.ipv4_tuple.src_port
            if addr in self.port_allocations:
                self.port_allocations[addr].discard(port)
            self.stats['sessions_expired'] += 1
    
    def print_stats(self):
        """打印统计信息"""
        print("NAT64 引擎统计:")
        print(f"  活跃会话数: {len(self.sessions)}")
        for key, val in self.stats.items():
            print(f"  {key}: {val}")


# 示例使用
engine = NAT64Engine(
    ipv4_pool=["203.0.113.1", "203.0.113.2"],
    local_prefix="2001:db8:1::"
)

# 模拟 IPv6 → IPv4 转换
v6_pkt = {
    'src_addr': '2001:db8:1::100',
    'dst_addr': '64:ff9b::5db8:d822',
    'src_port': 54321,
    'dst_port': 80,
    'protocol': 6,
    'traffic_class': 0,
    'hop_limit': 64,
    'payload_len': 100,
    'payload': b'\x00' * 100,
}

v4_pkt = engine.translate_v6_to_v4(v6_pkt)
if v4_pkt:
    print(f"转换成功:")
    print(f"  IPv4 源: {v4_pkt['src_addr']}:{v4_pkt['src_port']}")
    print(f"  IPv4 目的: {v4_pkt['dst_addr']}:{v4_pkt['dst_port']}")

# 模拟 IPv4 → IPv6 回程转换
v4_reply = {
    'src_addr': '93.184.216.34',
    'dst_addr': '203.0.113.1',
    'src_port': 80,
    'dst_port': 54321,
    'protocol': 6,
    'tos': 0,
    'ttl': 64,
    'total_length': 120,
    'payload': b'\x00' * 100,
}

v6_reply = engine.translate_v4_to_v6(v4_reply)
if v6_reply:
    print(f"回程转换成功:")
    print(f"  IPv6 源: {v6_reply['src_addr']}")
    print(f"  IPv6 目的: {v6_reply['dst_addr']}")

engine.print_stats()
```

<div data-component="NAT64TranslationDemo"></div>

**NAT64 引擎性能指标**：

| 指标 | 典型值 | 说明 |
|------|--------|------|
| 会话表查找时间 | O(1) | 哈希表查找 |
| 单包翻译延迟 | < 10 μs | 不含查表时间 |
| 最大并发会话数 | 100 万+ | 取决于内存 |
| 会话建立速率 | 10 万/秒 | 取决于 CPU |
| 吞吐量 | 100 Gbps+ | 硬件加速 |

**设计权衡**：

- **有状态 vs 无状态**：NAT64 有状态，支持完整 TCP 状态机；SIIT 无状态，但功能有限
- **地址池大小 vs 并发连接数**：地址池越大，可支持的并发连接越多
- **DNS64 合成 vs 配置 AAAA**：DNS64 自动合成方便，但可能与 DNSSEC 冲突
- **单层翻译 vs 464XLAT**：464XLAT 在客户端侧做一次 IPv4→IPv6 转换，减少网关负载

---

## 30.8 IPv6 部署现状与未来

### 30.8.1 全球部署率

截至 2025 年，全球 IPv6 部署率持续增长，但区域差异显著：

```
全球 IPv6 部署率（按国家/地区，2025 年估计）：

印度       ████████████████████████████████████████  70%+
法国       █████████████████████████████████████     65%+
德国       ████████████████████████████████████      60%+
美国       ███████████████████████████████████       55%+
日本       ██████████████████████████████            50%+
巴西       ████████████████████████████              45%+
英国       ██████████████████████████                40%+
韩国       █████████████████████████                 38%+
中国       █████████████████████████                 37%+
澳大利亚   ██████████████████████                    33%+
全球平均   ██████████████████                        30%+
```

### 30.8.2 主要国家进展

| 国家/地区 | 部署率 | 关键推动因素 | 主要运营商 |
|----------|--------|------------|-----------|
| 印度 | ~70% | Reliance Jio 全面 IPv6 部署 | Jio, Airtel |
| 法国 | ~65% | Free.fr 早期采用 | Free, Orange |
| 德国 | ~60% | DTAG 大规模部署 | Deutsche Telekom |
| 美国 | ~55% | T-Mobile 移动 IPv6 | T-Mobile, Comcast |
| 日本 | ~50% | NTT 推动 | NTT, KDDI |
| 中国 | ~37% | 三大运营商逐步推进 | 中国电信/移动/联通 |

### 30.8.3 IPv6-only 趋势

随着 IPv4 地址耗尽加剧，**IPv6-only** 成为新趋势：

```
网络演进路径：
┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
│  IPv4-only │──▶│   Dual-    │──▶│ IPv6-mostly│──▶│ IPv6-only  │
│  (传统)     │   │   Stack    │   │ (6-mostly) │   │ (目标)     │
│            │   │ (当前主流)  │   │ (新兴)     │   │ (未来)     │
└────────────┘   └────────────┘   └────────────┘   └────────────┘

464XLAT 架构（IPv6-only 移动网络方案）：
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│ IPv4 App  │     │ 手机     │     │ 运营商   │     │ IPv4 服务器│
│          │────▶│ CLAT     │────▶│ PLAT/    │────▶│          │
│          │     │(IPv4→IPv6)│    │ NAT64    │     │          │
└──────────┘     └──────────┘     └──────────┘     └──────────┘
                   只有 IPv6 承载            NAT64 网关
```

### 30.8.4 5G 与 IoT 的推动

**5G 网络**将 IPv6 作为核心技术要求：

```
5G 网络 IPv6 架构：
┌─────────────────────────────────────────────────────────────────┐
│                     5G 核心网 (5GC)                              │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐             │
│  │ AMF  │  │ SMF  │  │ UPF  │  │ PCF  │  │ UDM  │             │
│  └──┬───┘  └──┬───┘  └──┬───┘  └──────┘  └──────┘             │
│     │         │         │                                      │
│     │    IPv6 PDU Session                                       │
│     │    (IPv6-only 或 Dual-Stack)                              │
│     │         │         │                                      │
│  ┌──┴─────────┴─────────┴──┐                                   │
│  │        gNB (基站)         │                                   │
│  └──────────┬───────────────┘                                   │
│             │                                                   │
│  ┌──────────┴───────────┐                                       │
│  │    5G 设备 (UE)       │                                       │
│  │  - SLAAC 自动配置      │                                       │
│  │  - IPv6 多宿主         │                                       │
│  │  - 流标签 QoS          │                                       │
│  └───────────────────────┘                                       │
└─────────────────────────────────────────────────────────────────┘
```

**IoT 场景**：

| IoT 场景 | IPv6 优势 | 协议选择 |
|----------|----------|----------|
| 智能家居 | 天然支持 P2P，无需 NAT 穿越 | IPv6 + 6LoWPAN |
| 工业物联网 | 大量设备地址管理 | IPv6 + MQTT over TLS |
| 车联网 | 移动性支持，低延迟 | IPv6 + V2X |
| 智慧城市 | 海量设备地址需求 | IPv6 + NB-IoT |
| 智能农业 | 低功耗广域网 | IPv6 + LoRaWAN |

```
6LoWPAN 在 IoT 协议栈中的位置：
┌─────────────────────────────────┐
│         应用层 (CoAP/HTTP)       │
├─────────────────────────────────┤
│         传输层 (UDP/TCP)         │
├─────────────────────────────────┤
│         网络层 (IPv6)            │
├─────────────────────────────────┤
│    适配层 (6LoWPAN)              │  ← IPv6 头部压缩与分片
├─────────────────────────────────┤
│    链路层 (IEEE 802.15.4)        │  ← 最大帧长 127 字节
├─────────────────────────────────┤
│    物理层                        │
└─────────────────────────────────┘

6LoWPAN 压缩效果：
  完整 IPv6 头部: 40 字节
  压缩后:         2~7 字节
  压缩率:         82%~95%
```

### 30.8.5 部署挑战与应对

| 挑战 | 说明 | 应对策略 |
|------|------|----------|
| 兼容性 | 老旧设备不支持 IPv6 | 464XLAT、NAT64 过渡 |
| 运维复杂度 | 双栈增加运维工作量 | 自动化工具、IPv6-only 简化 |
| 安全性 | IPv6 安全意识不足 | IPv6 防火墙策略、RA Guard |
| 应用支持 | 部分应用硬编码 IPv4 | Happy Eyeballs (RFC 8305) |
| 人才缺口 | 运维人员 IPv6 知识不足 | 培训、实践环境 |

```python
def happy_eyeballs_v2(dns_results: list) -> str:
    """
    Happy Eyeballs v2 算法 (RFC 8305)
    同时尝试 IPv6 和 IPv4 连接，优先使用先成功的连接
    """
    import socket
    import concurrent.futures
    
    CONNECTION_DELAY = 250  # 毫秒，RFC 8305 建议值
    
    # 按地址族排序：IPv6 优先
    sorted_results = sorted(dns_results, key=lambda x: x[0])
    
    def try_connect(addr_info):
        family, addr = addr_info
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect(addr)
            return sock
        except:
            return None
    
    # 并发尝试连接，IPv6 优先
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {}
        for i, (family, addr) in enumerate(sorted_results):
            # IPv6 先开始，IPv4 延迟 250ms
            delay = 0 if family == socket.AF_INET6 else CONNECTION_DELAY / 1000
            future = executor.submit(try_connect, (family, addr))
            futures[future] = (family, addr)
        
        # 返回第一个成功的连接
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                family, addr = futures[future]
                return f"Connected via {'IPv6' if family == socket.AF_INET6 else 'IPv4'} to {addr}"
    
    return "Connection failed"
```

### 30.8.6 未来展望

```
IPv6 技术演进路线：
┌────────────────────────────────────────────────────────────────────┐
│ 2010        2015         2020         2025         2030           │
│  │           │            │            │            │              │
│  ▼           ▼            ▼            ▼            ▼              │
│ 早期        过渡期        加速期        主流期        成熟期         │
│ 部署                                                               │
│                                                                    │
│ ·IANA 耗尽  ·RIR 耗尽     ·5G IPv6     ·IPv6-only   ·IPv4 退役    │
│ ·RFC 2460   ·双栈为主      ·IoT 大规模  ·成为默认    ·纯 IPv6 互联 │
│ ·试验网络    ·NAT64 部署   ·SRv6 发展   ·IPv4 遗留   ·新协议优化   │
│             ·6to4/Teredo  ·IoT 6LoWPAN ·Happy EB   ·全面 IPv6    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 本章小结

```
IPv6 知识体系总览：
┌─────────────────────────────────────────────────────────────────┐
│                     IPv6 核心知识                                │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ 地址空间      │  │ 报文格式      │  │ 过渡技术      │          │
│  │              │  │              │  │              │          │
│  │ ·128位       │  │ ·40字节基本头 │  │ ·双栈         │          │
│  │ ·冒号十六进制 │  │ ·扩展头链     │  │ ·隧道         │          │
│  │ ·全球单播    │  │ ·流标签       │  │ ·翻译         │          │
│  │ ·链路本地    │  │              │  │ ·NAT64        │          │
│  │ ·唯一本地    │  │              │  │ ·464XLAT      │          │
│  │ ·组播/任播   │  │              │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ NDP 协议      │  │ 自动配置      │  │ 部署现状      │          │
│  │              │  │              │  │              │          │
│  │ ·RS/RA       │  │ ·SLAAC       │  │ ·全球30%+    │          │
│  │ ·NS/NA       │  │ ·EUI-64      │  │ ·区域差异     │          │
│  │ ·邻居状态机   │  │ ·隐私扩展    │  │ ·IPv6-only   │          │
│  │ ·DAD         │  │ ·DHCPv6      │  │ ·5G/IoT      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

**关键术语对照表**：

| 英文缩写 | 英文全称 | 中文翻译 |
|----------|---------|---------|
| SLAAC | Stateless Address Autoconfiguration | 无状态地址自动配置 |
| DAD | Duplicate Address Detection | 重复地址检测 |
| NDP | Neighbor Discovery Protocol | 邻居发现协议 |
| RA | Router Advertisement | 路由器通告 |
| RS | Router Solicitation | 路由器请求 |
| NS | Neighbor Solicitation | 邻居请求 |
| NA | Neighbor Advertisement | 邻居通告 |
| EUI-64 | Extended Unique Identifier 64 | 扩展唯一标识符 64 |
| NAT64 | Network Address Translation 64 | 网络地址转换 64 |
| DNS64 | DNS Extensions for NAT64 | NAT64 的 DNS 扩展 |
| SIIT | Stateless IP/ICMP Translation | 无状态 IP/ICMP 翻译 |
| 6LoWPAN | IPv6 over Low-Power WPAN | 低功耗无线个域网上的 IPv6 |
| SRv6 | Segment Routing over IPv6 | IPv6 上的段路由 |

---

## 练习题

**30.1** 计算问题：如果一个 ISP 从 RIR 获得一个 /32 的 IPv6 前缀，该 ISP 可以给多少个用户分配 /48 的前缀？每个 /48 前缀可以划分多少个 /64 子网？

**30.2** 协议分析：解释为什么 IPv6 取消了路由器分片机制。这对 PMTU 发现有什么影响？

**30.3** 实验题：使用 Python 构造一个包含 Hop-by-Hop 选项扩展头的 IPv6 数据报，并使用本章的 ExtensionHeaderParser 进行解析。

**30.4** 设计题：比较 NAT64 和 SIIT 两种翻译技术的优缺点。在什么场景下应该选择哪种技术？

**30.5** 综合题：分析一个从 IPv4-only 网络迁移到 IPv6-only 网络的完整方案，包括过渡阶段的技术选择、时间线和风险缓解措施。

---

## 参考文献

1. RFC 8200 — Internet Protocol, Version 6 (IPv6) Specification
2. RFC 4861 — Neighbor Discovery for IP version 6 (IPv6)
3. RFC 4862 — IPv6 Stateless Address Autoconfiguration
4. RFC 4941 — Privacy Extensions for Stateless Address Autoconfiguration
5. RFC 4443 — Internet Control Message Protocol (ICMPv6)
6. RFC 8200 — IPv6 Specification (replaces RFC 2460)
7. RFC 6146 — Stateful NAT64: Network Address and Protocol Translation
8. RFC 7915 — IP/ICMP Translation Algorithm
9. RFC 8305 — Happy Eyeballs Version 2
10. RFC 3056 — Connection of IPv6 Domains via IPv4 Clouds (6to4)
11. RFC 4380 — Teredo: Tunneling IPv6 over UDP through Network Address Translations
12. RFC 5214 — Intra-Site Automatic Tunnel Addressing Protocol (ISATAP)
13. RFC 8200 — Internet Protocol, Version 6 (IPv6) Specification
14. Deering, S., Hinden, R. "Internet Protocol, Version 6 (IPv6) Specification", RFC 8200, 2017

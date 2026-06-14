# Chapter 14: 传输层 — TCP 协议基础

> **学习目标**：
> - 理解 TCP 的服务模型及其与 UDP 的根本区别
> - 掌握 TCP 段格式的每个字段含义，包括序列号、确认号、标志位和选项
> - 深入理解三次握手（SYN/SYN-ACK/ACK）的状态转换过程及设计原因
> - 掌握四次挥手（FIN/ACK/FIN/ACK）的连接终止过程及半关闭概念
> - 熟记 TCP 状态机的 11 个状态及状态转换条件
> - 理解同时打开、同时关闭、TIME_WAIT、RST 处理等边界情况
> - 掌握 TCP 控制块（TCB）和选项处理器的数据结构设计

---

## 14.1 TCP 服务模型

### 14.1.1 TCP 的核心特性

TCP（Transmission Control Protocol，传输控制协议）是互联网中最重要的传输层协议，提供**面向连接的、可靠的、有序的字节流**传输服务：

```
TCP 服务模型的核心特性：

┌─────────────────────────────────────────────────────────────┐
│                    TCP 服务模型                               │
│                                                             │
│  1. 面向连接 (Connection-Oriented)                           │
│     ┌─────────────────────────────────────────────────────┐ │
│     │  通信前需要建立连接（三次握手）                         │ │
│     │  连接状态由两端维护                                    │ │
│     │  连接是点对点的（不支持多播/广播）                      │ │
│     └─────────────────────────────────────────────────────┘ │
│                                                             │
│  2. 可靠传输 (Reliable Transfer)                             │
│     ┌─────────────────────────────────────────────────────┐ │
│     │  序列号 + 确认号保证数据不丢失、不重复                  │ │
│     │  校验和保证数据不损坏                                  │ │
│     │  超时重传机制                                         │ │
│     └─────────────────────────────────────────────────────┘ │
│                                                             │
│  3. 有序交付 (Ordered Delivery)                              │
│     ┌─────────────────────────────────────────────────────┐ │
│     │  接收方按序列号重组乱序数据                             │ │
│     │  应用层收到的数据与发送顺序一致                         │ │
│     └─────────────────────────────────────────────────────┘ │
│                                                             │
│  4. 字节流服务 (Byte Stream Service)                         │
│     ┌─────────────────────────────────────────────────────┐ │
│     │  TCP 不保留应用层的消息边界                             │ │
│     │  应用层的多次 write 可能被合并为一个 TCP 段             │ │
│     │  一个 TCP 段可能被应用层多次 read 才读完                │ │
│     └─────────────────────────────────────────────────────┘ │
│                                                             │
│  5. 流量控制 (Flow Control)                                  │
│     ┌─────────────────────────────────────────────────────┐ │
│     │  接收方通过窗口大小通告发送方自己的接收能力              │ │
│     │  防止发送方发送过快导致接收方缓冲区溢出                  │ │
│     └─────────────────────────────────────────────────────┘ │
│                                                             │
│  6. 拥塞控制 (Congestion Control)                            │
│     ┌─────────────────────────────────────────────────────┐ │
│     │  TCP 感知网络拥塞并主动降低发送速率                     │ │
│     │  防止过多数据注入网络导致路由器排队溢出                  │ │
│     └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 14.1.2 TCP 字节流 vs UDP 数据报

```
TCP 字节流服务 vs UDP 数据报服务：

发送方调用：
  write("Hello", 5)
  write("World", 5)

TCP 接收方可能看到的（字节流，无边界）：
  ┌──────────────────────────────────────────┐
  │ H e l l o W o r l d │
  └──────────────────────────────────────────┘
  一次 read(10) 就能读到 "HelloWorld"

  或者：
  ┌────────────┐ ┌────────────┐
  │ H e l l o W│ │ o r l d    │
  └────────────┘ └────────────┘
  需要两次 read 才能读完

UDP 接收方看到的（保留消息边界）：
  第一次 recvfrom: "Hello"
  第二次 recvfrom: "World"
  每次 sendto 对应一次 recvfrom

这种差异的根本原因：
TCP 是字节流协议，数据没有内在边界
UDP 是数据报协议，每个数据报是独立的传输单元
```

### 14.1.3 TCP 连接的端点标识

```
TCP 连接由四元组唯一标识：

┌─────────────────────────────────────────────────────────────┐
│  TCP 连接四元组:                                              │
│  (源 IP 地址, 源端口号, 目的 IP 地址, 目的端口号)              │
│                                                             │
│  示例：                                                      │
│  客户端 10.0.0.1:5000 连接到 服务器 10.0.0.2:80              │
│  → 连接标识: (10.0.0.1, 5000, 10.0.0.2, 80)                │
│                                                             │
│  客户端 10.0.0.1:5001 连接到 服务器 10.0.0.2:80              │
│  → 连接标识: (10.0.0.1, 5001, 10.0.0.2, 80)                │
│                                                             │
│  这是两个不同的连接！同一个服务器端口可以同时服务多个客户端    │
└─────────────────────────────────────────────────────────────┘
```

---

## 14.2 TCP 段格式详解

### 14.2.1 TCP 段结构

TCP 段（Segment）由**头部**和**数据**两部分组成。头部最小 20 字节，最大 60 字节：

```
TCP 段格式：
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│          Source Port          │       Destination Port         │
├───────────────────────────────┼────────────────────────────────┤
│                        Sequence Number                         │
├────────────────────────────────────────────────────────────────┤
│                     Acknowledgment Number                      │
├───────┬─────┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─────────────────────┤
│  Data │ Rsrv│C│E│U│A│P│R│S│F│                               │
│ Offset│     │W│C│R│C│S│S│Y│I│            Window              │
│       │     │R│E│G│K│H│T│N│N│                               │
├───────┴─────┼─┼─┼─┼─┼─┼─┼─┼─┼──────────────────────────────┤
│           Checksum            │       Urgent Pointer          │
├───────────────────────────────┴──────────────────────────────┤
│                    Options (0-40 bytes)                       │
├──────────────────────────────────────────────────────────────┤
│                         Data                                  │
└──────────────────────────────────────────────────────────────┘
```

### 14.2.2 每个字段的详细说明

| 字段 | 位数 | 描述 | 详细说明 |
|------|------|------|---------|
| **Source Port** | 16 | 源端口号 | 标识发送方进程 |
| **Destination Port** | 16 | 目的端口号 | 标识接收方进程 |
| **Sequence Number** | 32 | 序列号 | 本次发送数据的第一个字节的编号 |
| **Acknowledgment Number** | 32 | 确认号 | 期望收到的下一个字节的编号 |
| **Data Offset** | 4 | 数据偏移 | TCP 头部长度，单位为 4 字节（32 位字） |
| **Reserved** | 3 | 保留位 | 必须为 0 |
| **CWR** | 1 | 拥塞窗口减小 | ECN 相关 |
| **ECE** | 1 | ECN-Echo | ECN 相关 |
| **URG** | 1 | 紧急指针有效 | 置 1 时 Urgent Pointer 字段有效 |
| **ACK** | 1 | 确认号有效 | 置 1 时 Acknowledgment Number 字段有效 |
| **PSH** | 1 | 推送标志 | 要求接收方尽快将数据交付应用层 |
| **RST** | 1 | 复位连接 | 重置连接，通常表示异常 |
| **SYN** | 1 | 同步序列号 | 用于建立连接 |
| **FIN** | 1 | 终止连接 | 发送方不再有数据发送 |
| **Window** | 16 | 接收窗口大小 | 接收方还能接收的字节数（流量控制） |
| **Checksum** | 16 | 校验和 | 覆盖头部和数据，包含伪头部 |
| **Urgent Pointer** | 16 | 紧急指针 | 指向紧急数据的最后一个字节 |
| **Options** | 变长 | 选项 | MSS、窗口缩放、时间戳、SACK 等 |
| **Data** | 变长 | 数据 | 应用层数据 |

### 14.2.3 序列号与确认号

```
序列号与确认号的工作原理：

发送方 A                                    接收方 B
    │                                          │
    │  seq=1000, len=500                       │
    │  (发送字节 1000-1499)                     │
    │ ──────────────────────────────────────→   │
    │                                          │
    │  seq=1500, len=500                       │
    │  (发送字节 1500-1999)                     │
    │ ──────────────────────────────────────→   │
    │                                          │
    │       ack=2000                           │
    │  (确认收到字节 0-1999，期望字节 2000)      │
    │ ←──────────────────────────────────────   │
    │                                          │
    │  seq=2000, len=1000                      │
    │  (发送字节 2000-2999)                     │
    │ ──────────────────────────────────────→   │
    │                                          │
    │       ack=3000                           │
    │  (确认收到字节 0-2999，期望字节 3000)      │
    │ ←──────────────────────────────────────   │

关键理解：
1. 序列号 = 数据流中第一个字节的编号
2. 确认号 = 期望收到的下一个字节的编号
3. 确认号表示"此编号之前的所有字节都已正确收到"
4. 这种机制称为"累积确认"（Cumulative ACK）
```

### 14.2.4 TCP 选项详解

```c
/* TCP 选项类型定义 */
#define TCP_OPT_EOL       0   /* 选项列表结束 */
#define TCP_OPT_NOP       1   /* 无操作（用于填充对齐） */
#define TCP_OPT_MSS       2   /* 最大段大小 */
#define TCP_OPT_WSCALE    3   /* 窗口缩放 */
#define TCP_OPT_SACK_PERM 4   /* SACK 允许 */
#define TCP_OPT_SACK      5   /* SACK 块 */
#define TCP_OPT_TIMESTAMP 8   /* 时间戳 */

/* MSS 选项结构 */
struct tcp_opt_mss {
    uint8_t  kind;    /* 2 */
    uint8_t  length;  /* 4 */
    uint16_t mss;     /* 最大段大小（字节） */
};

/* 窗口缩放选项结构 */
struct tcp_opt_wscale {
    uint8_t kind;     /* 3 */
    uint8_t length;   /* 3 */
    uint8_t shift;    /* 移位计数（0-14） */
};

/* 时间戳选项结构 */
struct tcp_opt_timestamp {
    uint8_t  kind;      /* 8 */
    uint8_t  length;    /* 10 */
    uint32_t ts_val;    /* 时间戳值 */
    uint32_t ts_ecr;    /* 回显时间戳 */
};

/* SACK 选项结构 */
struct tcp_opt_sack {
    uint8_t  kind;      /* 5 */
    uint8_t  length;    /* 可变 */
    struct {
        uint32_t left;   /* SACK 块左边界 */
        uint32_t right;  /* SACK 块右边界 */
    } blocks[0];         /* 可变数量的 SACK 块 */
};
```

**TCP 选项的处理逻辑**：

```c
/* TCP 选项解析器 */
struct tcp_options {
    uint16_t mss;           /* MSS 值 */
    uint8_t  wscale;        /* 窗口缩放值 */
    uint8_t  sack_perm;     /* SACK 允许 */
    uint32_t ts_val;        /* 时间戳值 */
    uint32_t ts_ecr;        /* 回显时间戳 */
    uint32_t *sack_blocks;  /* SACK 块数组 */
    int      num_sack;      /* SACK 块数量 */
};

int tcp_parse_options(const uint8_t *opt_data, int opt_len,
                      struct tcp_options *opts) {
    int i = 0;

    /* 初始化默认值 */
    opts->mss = 536;        /* RFC 默认 MSS */
    opts->wscale = 0;       /* 无缩放 */
    opts->sack_perm = 0;
    opts->ts_val = 0;
    opts->ts_ecr = 0;
    opts->num_sack = 0;

    while (i < opt_len) {
        uint8_t kind = opt_data[i];

        if (kind == TCP_OPT_EOL) break;  /* 选项结束 */
        if (kind == TCP_OPT_NOP) { i++; continue; }  /* 无操作 */

        if (i + 1 >= opt_len) break;  /* 防止越界 */
        uint8_t length = opt_data[i + 1];

        switch (kind) {
            case TCP_OPT_MSS:
                if (length == 4 && i + 3 < opt_len) {
                    opts->mss = (opt_data[i+2] << 8) | opt_data[i+3];
                }
                break;

            case TCP_OPT_WSCALE:
                if (length == 3 && i + 2 < opt_len) {
                    opts->wscale = opt_data[i+2];
                    if (opts->wscale > 14) opts->wscale = 14;
                }
                break;

            case TCP_OPT_SACK_PERM:
                if (length == 2) {
                    opts->sack_perm = 1;
                }
                break;

            case TCP_OPT_SACK:
                if (length >= 10) {  /* 至少一个 SACK 块 */
                    opts->num_sack = (length - 2) / 8;
                    opts->sack_blocks = (uint32_t *)(opt_data + i + 2);
                }
                break;

            case TCP_OPT_TIMESTAMP:
                if (length == 10) {
                    opts->ts_val = (opt_data[i+2] << 24) |
                                   (opt_data[i+3] << 16) |
                                   (opt_data[i+4] << 8)  |
                                   opt_data[i+5];
                    opts->ts_ecr = (opt_data[i+6] << 24) |
                                   (opt_data[i+7] << 16) |
                                   (opt_data[i+8] << 8)  |
                                   opt_data[i+9];
                }
                break;
        }

        i += length;
    }

    return 0;
}
```

<div data-component="TCPSegmentVisualizer"></div>

---

## 14.3 TCP 三次握手

### 14.3.1 三次握手的过程

TCP 使用**三次握手（Three-Way Handshake）**建立连接：

```
三次握手过程：

    客户端 (Client)                           服务器 (Server)
         │                                        │
         │     ① SYN                               │
         │     seq=x                               │
         │     (SYN=1, seq=x)                      │
         │ ──────────────────────────────────────→  │
         │                                        │  SYN_RCVD
         │                                        │
         │     ② SYN + ACK                         │
         │     seq=y, ack=x+1                      │
         │     (SYN=1, ACK=1, seq=y, ack=x+1)     │
         │ ←──────────────────────────────────────  │
    ESTABLISHED                                    │
         │                                        │
         │     ③ ACK                               │
         │     ack=y+1                             │
         │     (ACK=1, seq=x+1, ack=y+1)          │
         │ ──────────────────────────────────────→  │
         │                                   ESTABLISHED
         │                                        │
         │        数据传输开始                       │
         │ ←─────────────────────────────────────→  │

状态转换：
  客户端: CLOSED → SYN_SENT → ESTABLISHED
  服务器: CLOSED → LISTEN → SYN_RCVD → ESTABLISHED
```

### 14.3.2 三次握手的设计原因

**为什么需要三次握手，而不是两次？**

```
两次握手的问题（假设只有两次握手）：

场景：旧的重复 SYN 报文

    客户端                                    服务器
         │                                      │
         │  ① SYN (seq=100)                     │
         │ ────────────────────────────────→     │
         │                                      │
         │     [网络延迟，SYN 未到达]              │
         │                                      │
         │  ① SYN (seq=200)  ← 新的连接请求      │
         │ ────────────────────────────────→     │
         │                                      │
         │  ② SYN+ACK (seq=300, ack=201)        │
         │ ←────────────────────────────────     │
         │                                      │
         │  客户端收到 ACK，连接建立               │
         │                                      │
         │     [旧的 SYN 终于到达]                │
         │  ① SYN (seq=100)  ← 重复的旧 SYN     │
         │ ────────────────────────────────→     │
         │                                      │
         │  ② SYN+ACK (seq=400, ack=101)        │
         │ ←────────────────────────────────     │
         │                                      │
         │  服务器误以为建立了新连接！              │
         │  但客户端不会响应这个 ACK               │
         │  → 服务器资源被浪费                     │

三次握手如何解决：

    客户端                                    服务器
         │                                      │
         │  [旧的 SYN 到达]                      │
         │  ① SYN (seq=100)                     │
         │ ────────────────────────────────→     │
         │                                      │
         │  ② SYN+ACK (seq=300, ack=101)        │
         │ ←────────────────────────────────     │
         │                                      │
         │  客户端收到旧 SYN 的响应                │
         │  客户端发现 ack=101 不是期望的 201      │
         │  客户端发送 RST 拒绝连接               │
         │  ③ RST                                │
         │ ────────────────────────────────→     │
         │                                      │
         │  服务器收到 RST，关闭连接               │
         │  → 不会建立无效连接                     │
```

### 14.3.3 三次握手中的选项协商

```
三次握手中的选项协商：

    客户端                                    服务器
         │                                      │
         │  ① SYN                                │
         │     seq=1000                          │
         │     MSS=1460                          │
         │     WScale=7                          │
         │     SACK-Perm                         │
         │     Timestamp: val=1000               │
         │ ──────────────────────────────────────→│
         │                                      │
         │  ② SYN+ACK                            │
         │     seq=5000                          │
         │     ack=1001                          │
         │     MSS=1460                          │
         │     WScale=6                          │
         │     SACK-Perm                         │
         │     Timestamp: val=5000, ecr=1000     │
         │ ←──────────────────────────────────────│
         │                                      │
         │  ③ ACK                                │
         │     seq=1001                          │
         │     ack=5001                          │
         │     Timestamp: val=1001, ecr=5000     │
         │ ──────────────────────────────────────→│
         │                                      │

协商结果：
  MSS: min(1460, 1460) = 1460 字节
  WScale: 发送方用 7，接收方用 6
  SACK: 双方都支持
  Timestamp: 双方都支持
```

---

## 14.4 TCP 四次挥手

### 14.4.1 四次挥手的过程

TCP 使用**四次挥手（Four-Way Teardown）**终止连接：

```
四次挥手过程：

    主动关闭方 (Client)                        被动关闭方 (Server)
         │                                        │
         │     ① FIN                               │
         │     seq=u                               │
         │     (FIN=1, seq=u)                      │
         │ ──────────────────────────────────────→  │
         │                                        │  CLOSE_WAIT
         │                                        │  (可能还有数据要发送)
         │     ② ACK                               │
         │     ack=u+1                             │
         │     (ACK=1, ack=u+1)                    │
         │ ←──────────────────────────────────────  │
    FIN_WAIT_1                                     │
         │                                        │
         │     [Server 可能继续发送数据]             │
         │ ←──────────────────────────────────────  │
         │                                        │
         │     ③ FIN                               │
         │     seq=w                               │
         │     (FIN=1, seq=w)                      │
         │ ←──────────────────────────────────────  │
         │                                        │  LAST_ACK
    FIN_WAIT_2                                     │
         │                                        │
         │     ④ ACK                               │
         │     ack=w+1                             │
         │     (ACK=1, ack=w+1)                    │
         │ ──────────────────────────────────────→  │
         │                                   CLOSED
         │                                        │
    TIME_WAIT                                      │
    (等待 2MSL)                                    │
         │                                        │
    CLOSED                                         │

状态转换：
  主动关闭方: ESTABLISHED → FIN_WAIT_1 → FIN_WAIT_2 → TIME_WAIT → CLOSED
  被动关闭方: ESTABLISHED → CLOSE_WAIT → LAST_ACK → CLOSED
```

### 14.4.2 半关闭（Half-Close）

```
半关闭状态：

    客户端                                    服务器
         │                                      │
         │  FIN (客户端不再发送数据)              │
         │ ──────────────────────────────→      │
         │                                      │
         │  ACK                                  │
         │ ←──────────────────────────────      │
         │                                      │
         │  此时连接处于"半关闭"状态：            │
         │  • 客户端不能发送数据                  │
         │  • 服务器可以继续发送数据              │
         │  • 客户端可以接收服务器的数据          │
         │                                      │
         │  数据 (服务器继续发送)                 │
         │ ←──────────────────────────────      │
         │                                      │
         │  ACK                                  │
         │ ──────────────────────────────→      │
         │                                      │
         │  FIN (服务器也关闭)                    │
         │ ←──────────────────────────────      │
         │                                      │
         │  ACK                                  │
         │ ──────────────────────────────→      │

为什么需要四次挥手而不是三次？
因为 TCP 是全双工的，每个方向的关闭需要独立进行
FIN 只表示"我不再发送数据"，不代表不再接收数据
```

### 14.4.3 TIME_WAIT 状态

```
TIME_WAIT 状态的作用：

主动关闭方在发送最后一个 ACK 后进入 TIME_WAIT 状态
等待时间为 2MSL（Maximum Segment Lifetime，通常 2 分钟）

为什么需要 TIME_WAIT？

1. 确保最后的 ACK 能到达对方
   ┌─────────────────────────────────────────────────────┐
   │ 如果最后的 ACK 丢失，被动关闭方会重传 FIN              │
   │ 主动关闭方需要在 TIME_WAIT 状态下重发 ACK             │
   │ 如果没有 TIME_WAIT，主动关闭方可能已经关闭             │
   │ → 收到重传的 FIN 会返回 RST，导致连接异常终止          │
   └─────────────────────────────────────────────────────┘

2. 确保旧连接的数据段在网络中消亡
   ┌─────────────────────────────────────────────────────┐
   │ 网络中可能存在延迟的数据段                             │
   │ 如果立即建立新连接（相同四元组），旧数据段可能被误接收  │
   │ TIME_WAIT = 2MSL 确保所有旧数据段都已超时消亡          │
   └─────────────────────────────────────────────────────┘

TIME_WAIT 累积问题：
大量短连接（如 HTTP 1.0）会导致 TIME_WAIT 状态堆积
解决方法：SO_REUSEADDR、SO_REUSEPORT、连接池
```

---

## 14.5 TCP 状态机

### 14.5.1 完整的 TCP 状态机

```
TCP 状态机（11 个状态）：

┌─────────────────────────────────────────────────────────────────┐
│                        TCP 状态机                                │
│                                                                 │
│                         ┌──────────┐                            │
│                         │  CLOSED   │                            │
│                         └────┬─────┘                            │
│                              │                                  │
│              ┌───────────────┼───────────────┐                  │
│              │ 被动打开        │ 主动打开        │                  │
│              ▼               │               ▼                  │
│         ┌──────────┐        │          ┌──────────┐            │
│         │  LISTEN   │        │          │ SYN_SENT │            │
│         └────┬─────┘        │          └────┬─────┘            │
│              │ 收到 SYN      │               │ 收到 SYN+ACK     │
│              ▼               │               ▼                  │
│         ┌──────────┐        │          ┌──────────┐            │
│         │ SYN_RCVD │        │          │ESTABLISHED│            │
│         └────┬─────┘        │          └────┬─────┘            │
│              │ 收到 ACK      │               │                  │
│              ▼               │               │                  │
│         ┌──────────┐        │               │                  │
│         │ESTABLISHED│        │               │                  │
│         └────┬─────┘        │               │                  │
│              │               │               │                  │
│              │  ┌────────────────────────────┘                  │
│              │  │                                               │
│              ▼  ▼                                               │
│         ┌──────────┐                                            │
│         │ 主动关闭  │                                            │
│         └────┬─────┘                                            │
│              │ 发送 FIN                                         │
│              ▼                                                  │
│         ┌──────────┐                                            │
│         │FIN_WAIT_1│                                            │
│         └────┬─────┘                                            │
│              │ 收到 ACK                                         │
│              ▼                                                  │
│         ┌──────────┐                                            │
│         │FIN_WAIT_2│                                            │
│         └────┬─────┘                                            │
│              │ 收到 FIN                                         │
│              ▼                                                  │
│         ┌──────────┐                                            │
│         │TIME_WAIT │                                            │
│         └────┬─────┘                                            │
│              │ 2MSL 超时                                         │
│              ▼                                                  │
│         ┌──────────┐                                            │
│         │  CLOSED   │                                            │
│         └──────────┘                                            │
│                                                                 │
│  被动关闭路径：                                                   │
│  ESTABLISHED → CLOSE_WAIT → LAST_ACK → CLOSED                  │
└─────────────────────────────────────────────────────────────────┘
```

### 14.5.2 状态转换表

| 当前状态 | 事件 | 动作 | 下一状态 |
|---------|------|------|---------|
| CLOSED | 主动打开（connect） | 发送 SYN | SYN_SENT |
| CLOSED | 被动打开（listen） | 等待 SYN | LISTEN |
| LISTEN | 收到 SYN | 发送 SYN+ACK | SYN_RCVD |
| SYN_SENT | 收到 SYN+ACK | 发送 ACK | ESTABLISHED |
| SYN_SENT | 收到 SYN（同时打开） | 发送 SYN+ACK | SYN_RCVD |
| SYN_RCVD | 收到 ACK | — | ESTABLISHED |
| ESTABLISHED | 主动关闭（close） | 发送 FIN | FIN_WAIT_1 |
| ESTABLISHED | 收到 FIN | 发送 ACK | CLOSE_WAIT |
| FIN_WAIT_1 | 收到 ACK | — | FIN_WAIT_2 |
| FIN_WAIT_1 | 收到 FIN | 发送 ACK | CLOSING |
| FIN_WAIT_2 | 收到 FIN | 发送 ACK | TIME_WAIT |
| CLOSE_WAIT | 被动关闭（close） | 发送 FIN | LAST_ACK |
| LAST_ACK | 收到 ACK | — | CLOSED |
| TIME_WAIT | 2MSL 超时 | — | CLOSED |
| CLOSING | 收到 ACK | — | TIME_WAIT |

### 14.5.3 TCP 状态机的软件实现

```c
/* TCP 状态机的软件实现 */

enum tcp_state {
    TCP_CLOSED,
    TCP_LISTEN,
    TCP_SYN_SENT,
    TCP_SYN_RCVD,
    TCP_ESTABLISHED,
    TCP_FIN_WAIT_1,
    TCP_FIN_WAIT_2,
    TCP_CLOSING,
    TCP_TIME_WAIT,
    TCP_CLOSE_WAIT,
    TCP_LAST_ACK
};

/* 事件类型 */
enum tcp_event {
    EVT_ACTIVE_OPEN,    /* connect() */
    EVT_PASSIVE_OPEN,   /* listen() */
    EVT_RECV_SYN,       /* 收到 SYN */
    EVT_RECV_SYN_ACK,   /* 收到 SYN+ACK */
    EVT_RECV_ACK,       /* 收到 ACK */
    EVT_RECV_FIN,       /* 收到 FIN */
    EVT_CLOSE,          /* close() */
    EVT_TIMEOUT,        /* 超时 */
    EVT_RECV_RST        /* 收到 RST */
};

/* 状态转换表 */
struct tcp_transition {
    enum tcp_state current;
    enum tcp_event event;
    enum tcp_state next;
    void (*action)(struct tcp_pcb *pcb);
};

static struct tcp_transition transitions[] = {
    /* CLOSED 状态 */
    {TCP_CLOSED, EVT_ACTIVE_OPEN, TCP_SYN_SENT, tcp_send_syn},
    {TCP_CLOSED, EVT_PASSIVE_OPEN, TCP_LISTEN, NULL},

    /* LISTEN 状态 */
    {TCP_LISTEN, EVT_RECV_SYN, TCP_SYN_RCVD, tcp_send_syn_ack},

    /* SYN_SENT 状态 */
    {TCP_SYN_SENT, EVT_RECV_SYN_ACK, TCP_ESTABLISHED, tcp_send_ack},
    {TCP_SYN_SENT, EVT_RECV_SYN, TCP_SYN_RCVD, tcp_send_syn_ack},

    /* SYN_RCVD 状态 */
    {TCP_SYN_RCVD, EVT_RECV_ACK, TCP_ESTABLISHED, NULL},

    /* ESTABLISHED 状态 */
    {TCP_ESTABLISHED, EVT_CLOSE, TCP_FIN_WAIT_1, tcp_send_fin},
    {TCP_ESTABLISHED, EVT_RECV_FIN, TCP_CLOSE_WAIT, tcp_send_ack},

    /* FIN_WAIT_1 状态 */
    {TCP_FIN_WAIT_1, EVT_RECV_ACK, TCP_FIN_WAIT_2, NULL},
    {TCP_FIN_WAIT_1, EVT_RECV_FIN, TCP_CLOSING, tcp_send_ack},

    /* FIN_WAIT_2 状态 */
    {TCP_FIN_WAIT_2, EVT_RECV_FIN, TCP_TIME_WAIT, tcp_send_ack},

    /* CLOSING 状态 */
    {TCP_CLOSING, EVT_RECV_ACK, TCP_TIME_WAIT, NULL},

    /* CLOSE_WAIT 状态 */
    {TCP_CLOSE_WAIT, EVT_CLOSE, TCP_LAST_ACK, tcp_send_fin},

    /* LAST_ACK 状态 */
    {TCP_LAST_ACK, EVT_RECV_ACK, TCP_CLOSED, NULL},

    /* TIME_WAIT 状态 */
    {TCP_TIME_WAIT, EVT_TIMEOUT, TCP_CLOSED, NULL},
};

#define NUM_TRANSITIONS (sizeof(transitions) / sizeof(transitions[0]))

/* 处理状态转换 */
int tcp_handle_event(struct tcp_pcb *pcb, enum tcp_event event) {
    for (int i = 0; i < NUM_TRANSITIONS; i++) {
        if (transitions[i].current == pcb->state &&
            transitions[i].event == event) {

            printf("TCP: %s + %s -> %s\n",
                   tcp_state_str(pcb->state),
                   tcp_event_str(event),
                   tcp_state_str(transitions[i].next));

            pcb->state = transitions[i].next;

            if (transitions[i].action) {
                transitions[i].action(pcb);
            }

            return 0;
        }
    }

    printf("TCP: 无效转换 %s + %s\n",
           tcp_state_str(pcb->state),
           tcp_event_str(event));
    return -1;
}
```

<div data-component="TCPStateMachineVisualizer"></div>

---

## 14.6 TCP 控制块（TCB）

### 14.6.1 TCB 的数据结构

TCP 控制块（Transmission Control Block, TCB）是维护 TCP 连接状态的核心数据结构：

```c
/* TCP 控制块 (TCB) 完整定义 */
struct tcp_pcb {
    /* 连接标识 */
    uint32_t local_ip;
    uint16_t local_port;
    uint32_t remote_ip;
    uint16_t remote_port;

    /* 连接状态 */
    enum tcp_state state;

    /* 序列号管理 */
    uint32_t snd_nxt;      /* 下一个要发送的序列号 */
    uint32_t snd_una;      /* 最早未确认的序列号 */
    uint32_t snd_wnd;      /* 发送窗口大小 */
    uint32_t snd_wl1;      /* 上次窗口更新的序列号 */
    uint32_t snd_wl2;      /* 上次窗口更新的确认号 */
    uint32_t iss;          /* 初始发送序列号 */

    /* 接收序列号管理 */
    uint32_t rcv_nxt;      /* 期望收到的下一个序列号 */
    uint32_t rcv_wnd;      /* 接收窗口大小 */
    uint32_t rcv_adv;      /* 通告的窗口右边界 */
    uint32_t irs;          /* 初始接收序列号 */

    /* 发送缓冲区 */
    struct tcp_send_buffer {
        uint8_t *data;
        size_t   len;
        size_t   used;
        uint32_t seq_start;
    } snd_buf;

    /* 接收缓冲区 */
    struct tcp_recv_buffer {
        uint8_t *data;
        size_t   len;
        size_t   used;
        uint32_t seq_start;
    } rcv_buf;

    /* 重传队列 */
    struct tcp_retransmit_queue {
        struct tcp_segment *head;
        struct tcp_segment *tail;
        int count;
    } retransmit_queue;

    /* 定时器 */
    uint32_t rto;          /* 重传超时 */
    uint32_t srtt;         /* 平滑 RTT */
    uint32_t rttvar;       /* RTT 变化量 */
    uint32_t rtt_seq;      /* 用于 RTT 测量的序列号 */
    uint32_t rtt_time;     /* RTT 测量开始时间 */

    /* 拥塞控制 */
    uint32_t cwnd;         /* 拥塞窗口 */
    uint32_t ssthresh;     /* 慢启动阈值 */
    uint32_t mss;          /* 最大段大小 */

    /* TCP 选项 */
    struct tcp_options {
        uint16_t mss;
        uint8_t  wscale_send;   /* 发送窗口缩放 */
        uint8_t  wscale_recv;   /* 接收窗口缩放 */
        uint8_t  sack_perm;     /* SACK 允许 */
        uint32_t ts_recent;     /* 最近收到的时间戳 */
        uint32_t ts_last_ack;   /* 上次 ACK 的时间戳 */
    } options;

    /* 统计信息 */
    struct tcp_stats {
        uint64_t bytes_sent;
        uint64_t bytes_recv;
        uint64_t segs_sent;
        uint64_t segs_recv;
        uint64_t segs_retrans;
        uint32_t timeout_count;
    } stats;

    /* 回调函数 */
    void (*recv_callback)(struct tcp_pcb *pcb, struct tcp_segment *seg);
    void (*send_callback)(struct tcp_pcb *pcb);
    void (*error_callback)(struct tcp_pcb *pcb, int err);

    /* 链表指针（用于 PCB 池管理） */
    struct tcp_pcb *next;
};
```

### 14.6.2 TCB 的生命周期管理

```python
# TCB 生命周期管理（Python 模拟）

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable
import time

class TCPState(Enum):
    CLOSED = "CLOSED"
    LISTEN = "LISTEN"
    SYN_SENT = "SYN_SENT"
    SYN_RCVD = "SYN_RCVD"
    ESTABLISHED = "ESTABLISHED"
    FIN_WAIT_1 = "FIN_WAIT_1"
    FIN_WAIT_2 = "FIN_WAIT_2"
    CLOSING = "CLOSING"
    TIME_WAIT = "TIME_WAIT"
    CLOSE_WAIT = "CLOSE_WAIT"
    LAST_ACK = "LAST_ACK"

@dataclass
class TCB:
    """TCP 控制块"""
    # 连接标识
    local_ip: str = "0.0.0.0"
    local_port: int = 0
    remote_ip: str = "0.0.0.0"
    remote_port: int = 0

    # 状态
    state: TCPState = TCPState.CLOSED

    # 序列号
    snd_nxt: int = 0
    snd_una: int = 0
    rcv_nxt: int = 0
    iss: int = 0
    irs: int = 0

    # 窗口
    snd_wnd: int = 65535
    rcv_wnd: int = 65535
    cwnd: int = 1460  # 初始拥塞窗口

    # 选项
    mss: int = 536
    wscale_send: int = 0
    wscale_recv: int = 0
    sack_perm: bool = False

    # 定时器
    rto: float = 1.0  # 初始 RTO = 1 秒
    srtt: float = 0.0
    rttvar: float = 0.0

    # 统计
    created_at: float = field(default_factory=time.time)
    bytes_sent: int = 0
    bytes_recv: int = 0

    def __str__(self):
        return (f"TCB({self.local_ip}:{self.local_port} -> "
                f"{self.remote_ip}:{self.remote_port}, "
                f"state={self.state.value})")


class TCPControlBlockPool:
    """TCB 池管理器"""

    def __init__(self, max_connections=10000):
        self.pool: dict[tuple, TCB] = {}
        self.listen_queue: dict[int, list[TCB]] = {}  # 端口 -> 半连接队列
        self.max_connections = max_connections

    def create_tcb(self, local_ip, local_port, remote_ip, remote_port) -> TCB:
        """创建新的 TCB"""
        key = (local_ip, local_port, remote_ip, remote_port)
        if key in self.pool:
            raise ValueError(f"Connection already exists: {key}")

        if len(self.pool) >= self.max_connections:
            raise ValueError("Connection limit reached")

        tcb = TCB(
            local_ip=local_ip,
            local_port=local_port,
            remote_ip=remote_ip,
            remote_port=remote_port
        )
        self.pool[key] = tcb
        return tcb

    def find_tcb(self, local_ip, local_port, remote_ip, remote_port) -> Optional[TCB]:
        """查找 TCB"""
        key = (local_ip, local_port, remote_ip, remote_port)
        return self.pool.get(key)

    def remove_tcb(self, tcb: TCB):
        """移除 TCB"""
        key = (tcb.local_ip, tcb.local_port, tcb.remote_ip, tcb.remote_port)
        self.pool.pop(key, None)

    def get_established_count(self) -> int:
        """获取已建立连接数"""
        return sum(1 for tcb in self.pool.values()
                   if tcb.state == TCPState.ESTABLISHED)
```

---

## 14.7 同时打开与同时关闭

### 14.7.1 同时打开

```
同时打开（Simultaneous Open）：

    主机 A                                    主机 B
         │                                        │
         │  主动打开 (connect)                     │  主动打开 (connect)
         │  状态: SYN_SENT                         │  状态: SYN_SENT
         │                                        │
         │     ① SYN (seq=x)                      │
         │ ──────────────────────────────────────→  │
         │                                        │
         │     ① SYN (seq=y)                      │
         │ ←──────────────────────────────────────  │
         │                                        │
         │  收到对方的 SYN                          │  收到对方的 SYN
         │  状态: SYN_RCVD                         │  状态: SYN_RCVD
         │                                        │
         │     ② SYN+ACK (seq=x+1, ack=y+1)      │
         │ ──────────────────────────────────────→  │
         │                                        │
         │     ② SYN+ACK (seq=y+1, ack=x+1)      │
         │ ←──────────────────────────────────────  │
         │                                        │
         │  收到 SYN+ACK                           │  收到 SYN+ACK
         │  状态: ESTABLISHED                      │  状态: ESTABLISHED

注意：同时打开只需要三次报文交换，而非四次
因为双方的 SYN 和 SYN+ACK 合并了
```

### 14.7.2 同时关闭

```
同时关闭（Simultaneous Close）：

    主机 A                                    主机 B
         │                                        │
         │  主动关闭                                │  主动关闭
         │  状态: FIN_WAIT_1                       │  状态: FIN_WAIT_1
         │                                        │
         │     ① FIN (seq=x)                      │
         │ ──────────────────────────────────────→  │
         │                                        │
         │     ① FIN (seq=y)                      │
         │ ←──────────────────────────────────────  │
         │                                        │
         │  收到 FIN                               │  收到 FIN
         │  状态: CLOSING                          │  状态: CLOSING
         │                                        │
         │     ② ACK (ack=y+1)                    │
         │ ──────────────────────────────────────→  │
         │                                        │
         │     ② ACK (ack=x+1)                    │
         │ ←──────────────────────────────────────  │
         │                                        │
         │  收到 ACK                               │  收到 ACK
         │  状态: TIME_WAIT                        │  状态: TIME_WAIT
         │                                        │
         │  2MSL 超时                               │  2MSL 超时
         │  状态: CLOSED                           │  状态: CLOSED
```

---

## 14.8 RST 处理

### 14.8.1 RST 的产生场景

```
RST（Reset）报文的产生场景：

1. 连接到不存在的端口
   ┌─────────────────────────────────────────────────────┐
   │ 客户端 SYN → 服务器                                  │
   │ 服务器端口没有监听                                    │
   │ 服务器返回 RST                                      │
   └─────────────────────────────────────────────────────┘

2. 异常终止连接
   ┌─────────────────────────────────────────────────────┐
   │ 任何一方发送 RST 可以立即终止连接                      │
   │ 不需要等待对方确认                                    │
   │ 相比正常关闭（FIN），RST 是"粗暴"的                   │
   └─────────────────────────────────────────────────────┘

3. 收到无效的段
   ┌─────────────────────────────────────────────────────┐
   │ 收到序列号不在窗口内的段                              │
   │ 收到无效的 ACK（确认号不在合理范围）                   │
   └─────────────────────────────────────────────────────┘

4. 半开连接检测
   ┌─────────────────────────────────────────────────────┐
   │ 一端崩溃重启后，另一端发送数据                        │
   │ 重启的一端已经没有该连接的 TCB                        │
   │ 返回 RST 通知对方连接已失效                          │
   └─────────────────────────────────────────────────────┘
```

### 14.8.2 RST 的处理逻辑

```c
/* RST 处理函数 */
void tcp_handle_rst(struct tcp_pcb *pcb, struct tcp_segment *seg) {
    switch (pcb->state) {
        case TCP_SYN_SENT:
            /* SYN_SENT 状态收到 RST */
            if (seg->ack && seg->ack_num == pcb->snd_nxt) {
                /* 有效 RST：连接被拒绝 */
                pcb->state = TCP_CLOSED;
                tcp_notify_error(pcb, ECONNREFUSED);
                tcp_free_pcb(pcb);
            }
            break;

        case TCP_SYN_RCVD:
            /* SYN_RCVD 状态收到 RST */
            pcb->state = TCP_CLOSED;
            tcp_free_pcb(pcb);
            break;

        case TCP_ESTABLISHED:
        case TCP_FIN_WAIT_1:
        case TCP_FIN_WAIT_2:
        case TCP_CLOSE_WAIT:
            /* 已建立连接收到 RST */
            pcb->state = TCP_CLOSED;
            tcp_notify_error(pcb, ECONNRESET);
            tcp_flush_buffers(pcb);
            tcp_free_pcb(pcb);
            break;

        case TCP_CLOSING:
        case TCP_LAST_ACK:
        case TCP_TIME_WAIT:
            /* 关闭过程中收到 RST */
            pcb->state = TCP_CLOSED;
            tcp_free_pcb(pcb);
            break;

        default:
            /* 其他状态忽略 RST */
            break;
    }
}
```

<div data-component="TCPResetAnalyzer"></div>

---

## 14.9 TCP 与 UDP 的协议栈实现对比

### 14.9.1 协议处理开销对比

```
TCP vs UDP 协议处理开销：

UDP 处理流程（简单）：
  收到数据段 → 校验和检查 → 查找套接字 → 交付应用
  ┌─────────────────────────────────────────────────────┐
  │ 处理步骤: 4 步                                       │
  │ 头部解析: 8 字节                                      │
  │ 状态维护: 无                                         │
  │ 内存分配: 仅接收缓冲区                                │
  │ 典型延迟: < 10μs                                     │
  └─────────────────────────────────────────────────────┘

TCP 处理流程（复杂）：
  收到数据段 → 校验和检查 → 查找 TCB → 状态机处理 →
  序列号检查 → 确认处理 → 数据重组 → 窗口更新 →
  定时器管理 → 可能触发重传 → 交付应用
  ┌─────────────────────────────────────────────────────┐
  │ 处理步骤: 10+ 步                                     │
  │ 头部解析: 20-60 字节（含选项）                        │
  │ 状态维护: TCB (数百字节)                              │
  │ 内存分配: 发送/接收缓冲区 + 重传队列                   │
  │ 典型延迟: 10-100μs                                   │
  └─────────────────────────────────────────────────────┘
```

### 14.9.2 代码实现复杂度对比

```python
# UDP 服务器（简单）
import socket

def udp_echo_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', 9999))
    while True:
        data, addr = sock.recvfrom(4096)
        sock.sendto(data, addr)  # 回显

# TCP 服务器（相对复杂）
import socket
import threading

def tcp_echo_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('', 9999))
    server.listen(128)  # 需要监听队列

    while True:
        client, addr = server.accept()  # 三次握手完成
        threading.Thread(target=handle_client, args=(client,)).start()

def handle_client(client):
    try:
        while True:
            data = client.recv(4096)  # 可能收到不完整的数据
            if not data:
                break  # 对端关闭连接
            client.sendall(data)  # 确保所有数据都发送
    finally:
        client.close()  # 四次挥手
```

---

## 本章小结

本章深入介绍了 TCP 协议的基础知识：

1. **TCP 服务模型**：面向连接、可靠、有序、字节流服务
2. **TCP 段格式**：20-60 字节头部，包含序列号、确认号、标志位和选项
3. **三次握手**：解决旧 SYN 重复问题，协商连接参数
4. **四次挥手**：支持半关闭，TIME_WAIT 确保连接正确终止
5. **TCP 状态机**：11 个状态，精确控制连接生命周期
6. **TCB 数据结构**：维护连接的所有状态信息
7. **边界情况**：同时打开/关闭、RST 处理

**核心设计思想**：

- **状态机驱动**：TCP 的行为完全由状态机控制，每个事件触发状态转换和动作
- **可靠性设计**：通过序列号、确认号、超时重传等机制保证数据可靠传输
- **协商机制**：通过选项协商，两端可以自适应不同的网络环境

---

> **下一章预告**：第 15 章将深入探讨 TCP 的可靠传输机制，包括重传、滑动窗口和流量控制。

# Chapter 11: 网络层 — 路由器架构与工作原理

> **学习目标**：
> - 理解路由器的整体架构（输入端口、交换矩阵、输出端口、路由处理器）
> - 掌握输入端口的分层处理管线（物理层、链路层、网络层）
> - 区分转发（Forwarding）与路由（Routing）的概念和实现
> - 理解最长前缀匹配的硬件实现（TCAM 和 Trie 树）
> - 掌握三种交换矩阵架构（内存、总线、交叉开关）及其调度算法
> - 理解输出端口的排队策略与缓冲区管理（RED、WFQ）
> - 了解 HOL 阻塞问题及其解决方案
> - 深入分析现代路由器（Cisco、Juniper）的内部实现

---

## 11.1 路由器架构概述

### 11.1.1 路由器的功能

路由器是网络层的核心设备，负责：
1. **路由（Routing）**：运行路由协议，计算路由表（控制平面）
2. **转发（Forwarding）**：根据路由表将数据报从输入端口转移到输出端口（数据平面）
3. **其他功能**：分片、NAT、QoS、ACL、流量工程等

### 11.1.2 通用路由器架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        路由器总体架构                            │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     ┌──────────┐    │
│  │ 输入端口 │  │ 输入端口 │  │ 输入端口 │     │ 输入端口 │    │
│  │   1      │  │   2      │  │   3      │ ... │   N      │    │
│  │┌────────┐│  │┌────────┐│  │┌────────┐│     │┌────────┐│    │
│  ││物理层  ││  ││物理层  ││  ││物理层  ││     ││物理层  ││    │
│  │├────────┤│  │├────────┤│  │├────────┤│     │├────────┤│    │
│  ││链路层  ││  ││链路层  ││  ││链路层  ││     ││链路层  ││    │
│  │├────────┤│  │├────────┤│  │├────────┤│     │├────────┤│    │
│  ││查找/  ││  ││查找/  ││  ││查找/  ││     ││查找/  ││    │
│  ││转发   ││  ││转发   ││  ││转发   ││     ││转发   ││    │
│  │└────────┘│  │└────────┘│  │└────────┘│     │└────────┘│    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘     └────┬─────┘    │
│       │              │              │                │          │
│       ▼              ▼              ▼                ▼          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                     交换矩阵 (Switch Fabric)             │  │
│  │            (内存 / 总线 / 交叉开关)                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│       │              │              │                │          │
│       ▼              ▼              ▼                ▼          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     ┌──────────┐    │
│  │ 输出端口 │  │ 输出端口 │  │ 输出端口 │     │ 输出端口 │    │
│  │   1      │  │   2      │  │   3      │ ... │   N      │    │
│  │┌────────┐│  │┌────────┐│  │┌────────┐│     │┌────────┐│    │
│  ││排队/  ││  ││排队/  ││  ││排队/  ││     ││排队/  ││    │
│  ││调度   ││  ││调度   ││  ││调度   ││     ││调度   ││    │
│  │├────────┤│  │├────────┤│  │├────────┤│     │├────────┤│    │
│  ││链路层  ││  ││链路层  ││  ││链路层  ││     ││链路层  ││    │
│  │├────────┤│  │├────────┤│  │├────────┤│     │├────────┤│    │
│  ││物理层  ││  ││物理层  ││  ││物理层  ││     ││物理层  ││    │
│  │└────────┘│  │└────────┘│  │└────────┘│     │└────────┘│    │
│  └──────────┘  └──────────┘  └──────────┘     └──────────┘    │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              路由处理器 (Route Processor)                 │  │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐              │  │
│  │  │路由协议   │ │路由表     │ │转发表     │              │  │
│  │  │(OSPF/BGP)│ │管理       │ │计算       │              │  │
│  │  └───────────┘ └───────────┘ └───────────┘              │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 11.1.3 控制平面 vs 数据平面

| 特性 | 控制平面 | 数据平面 |
|------|---------|---------|
| 功能 | 运行路由协议，计算路由表 | 根据转发表转发数据报 |
| 实现 | 软件（CPU） | 硬件（ASIC/NP） |
| 速度 | 秒级收敛 | 纳秒级转发 |
| 处理量 | 少量路由更新 | 大量数据报 |
| 可编程性 | 高 | 低（硬件固化） |

<div dataComponent="RouterArchitectureDemo"></div>

---

## 11.2 输入端口处理

### 11.2.1 分层处理管线

输入端口按照 OSI 模型的层次进行处理：

```
输入端口处理管线:

物理层处理:
┌─────────────────────────────────────────────┐
│ 1. 光电转换（光纤→电信号）                   │
│ 2. 时钟恢复和同步                            │
│ 3. 线路编码/解码                             │
│ 4. 物理层帧定界                              │
└────────────────────────┬────────────────────┘
                         │
                         ▼
链路层处理:
┌─────────────────────────────────────────────┐
│ 1. 帧定界（识别帧开始和结束）                │
│ 2. FCS 校验（CRC-32 验证）                   │
│ 3. MAC 地址检查（是否发给本机）              │
│ 4. 802.1Q VLAN 标签处理                     │
│ 5. 链路层解封装                              │
└────────────────────────┬────────────────────┘
                         │
                         ▼
网络层处理（关键！）:
┌─────────────────────────────────────────────┐
│ 1. IP 首部校验和验证                         │
│ 2. 最长前缀匹配查找（TCAM/Trie）            │
│    → 确定输出端口号                         │
│ 3. TTL 减 1，重新计算校验和                  │
│ 4. 分片检查和处理                            │
│ 5. ACL/QoS 分类                              │
│ 6. 更新统计计数器                            │
└────────────────────────┬────────────────────┘
                         │
                         ▼
                   交换矩阵入口
```

### 11.2.2 输入端口处理的性能要求

输入端口必须以**线速（Line Rate）** 处理数据报，否则会造成丢包。

**线速处理时间计算**：

| 链路速率 | 最小帧长 | 线速 | 每帧处理时间 |
|---------|---------|------|------------|
| 1 Gbps | 64 字节 | 1,488,095 pps | 672 ns |
| 10 Gbps | 64 字节 | 14,880,952 pps | 67.2 ns |
| 100 Gbps | 64 字节 | 148,809,523 pps | 6.72 ns |

在 100 Gbps 速率下，每个数据报的处理时间只有 **6.72 纳秒**！这要求硬件实现（ASIC 或网络处理器）。

### 11.2.3 输入端口的硬件实现

```c
/* 输入端口处理管线（硬件描述） */

struct input_port_pipeline {
    /* 物理层 */
    struct phy_interface    *phy;
    uint8_t                 *rx_buffer;

    /* 链路层 */
    uint8_t                 mac_addr[6];
    struct mac_filter       *mac_filter;
    struct vlan_processor   *vlan_proc;

    /* 网络层 */
    struct route_lookup     *route_lookup;   /* TCAM 或 Trie */
    struct acl_engine       *acl;
    struct qos_classifier   *qos;

    /* 统计计数器 */
    struct port_stats       stats;
};

/* 输入端口处理主函数 */
int input_port_process(struct input_port_pipeline *pipeline,
                        struct packet *pkt) {
    /* 1. 链路层处理 */
    if (link_layer_process(pipeline, pkt) != 0) {
        pipeline->stats.link_errors++;
        return -1;
    }

    /* 2. VLAN 处理 */
    uint16_t vlan_id = vlan_ingress_process(pipeline->vlan_proc, pkt);
    pkt->vlan_id = vlan_id;

    /* 3. ACL 检查 */
    if (acl_check(pipeline->acl, pkt) == ACL_DENY) {
        pipeline->stats.acl_drops++;
        return -1;
    }

    /* 4. 路由查找 - 最关键的步骤 */
    struct route_entry *route = route_lookup(pipeline->route_lookup,
                                              pkt->dst_ip);
    if (route == NULL) {
        pipeline->stats.no_route++;
        return -1;
    }

    /* 5. 确定输出端口和下一跳信息 */
    pkt->output_port = route->out_interface;
    pkt->next_hop_ip = route->next_hop;

    /* 6. TTL 处理 */
    pkt->ttl--;
    if (pkt->ttl == 0) {
        icmp_send_time_exceeded(pkt);
        pipeline->stats.ttl_expired++;
        return -1;
    }

    /* 7. 更新统计 */
    pipeline->stats.packets_received++;
    pipeline->stats.bytes_received += pkt->length;

    return 0;  /* 成功 */
}
```

<div dataComponent="InputPortPipeline"></div>

---

## 11.3 转发 vs 路由

### 11.3.1 概念区分

| 特性 | 转发（Forwarding） | 路由（Routing） |
|------|-------------------|----------------|
| 定义 | 将数据报从入端口移到出端口 | 计算和维护路由表 |
| 层次 | 数据平面 | 控制平面 |
| 速度 | 纳秒级（硬件） | 秒级（软件） |
| 频率 | 每个数据报 | 路由更新时 |
| 输入 | 数据报目的 IP | 路由更新消息 |
| 输出 | 输出端口号 | 路由表 |
| 实现 | ASIC/TCAM | CPU/软件 |

### 11.3.2 转发表 vs 路由表

```
路由表（控制平面）:
┌────────────────┬──────────┬────────────┬────────┐
│ 目的网络       │ 下一跳   │ 接口       │ 度量   │
├────────────────┼──────────┼────────────┼────────┤
│ 10.0.0.0/8     │ 10.0.0.1 │ eth0       │ OSPF 10│
│ 172.16.0.0/12  │ 10.0.0.2 │ eth1       │ OSPF 20│
│ 192.168.0.0/16 │ 10.0.0.3 │ eth2       │ BGP    │
│ 0.0.0.0/0      │ 10.0.0.4 │ eth3       │ Static │
└────────────────┴──────────┴────────────┴────────┘

转发表（数据平面 - TCAM）:
┌──────────────────────────┬────────────┐
│ 前缀（三态格式）          │ 输出端口   │
├──────────────────────────┼────────────┤
│ 00001010.XXXXXXXX...     │ eth0       │
│ 10101100.00010000.XXXX...│ eth1       │
│ 11000000.10101000.XXXX...│ eth2       │
│ XXXXXXXX.XXXXXXXX...     │ eth3       │
└──────────────────────────┴────────────┘
```

---

## 11.4 最长前缀匹配

### 11.4.1 最长前缀匹配算法

转发表查找的核心问题是**最长前缀匹配（Longest Prefix Match, LPM）**：

给定一个目的 IP 地址，在转发表中找到与该地址匹配的最长前缀。

```
示例:

目的 IP: 192.168.1.200 (0xC0A801C8)

转发表:
条目 1: 0.0.0.0/0        → eth0 (默认路由)
条目 2: 192.168.0.0/16   → eth1
条目 3: 192.168.1.0/24   → eth2
条目 4: 192.168.1.128/25 → eth3

匹配分析:
- 条目 1: 前 0 位匹配 (总是匹配)
- 条目 2: 前 16 位匹配 (192.168)
- 条目 3: 前 24 位匹配 (192.168.1)
- 条目 4: 前 25 位匹配 (192.168.1.1xxxxxxx)
  192.168.1.200 = 192.168.1.11001000
  192.168.1.128 = 192.168.1.10000000
  前 25 位: 11000000.10101000.00000001.1xxxxxxx ✓

最长匹配: 条目 4 (25 位) → eth3
```

### 11.4.2 TCAM 实现

**TCAM（Ternary Content Addressable Memory）** 是实现 LPM 的首选硬件：

```
TCAM 工作原理:

每个 TCAM 条目存储 3 种状态: 0, 1, X (无关)

条目 1 (/25):  11000000.10101000.00000001.10000000  XXXXXXXX.XXXXXXXX.XXXXXXXX.XXXXXXXX
                ├─────────────────────────────────┤
                前 25 位必须匹配                     后 7 位任意

查询: 192.168.1.200 = 11000000.10101000.00000001.11001000

并行比较所有条目:
  条目 1 (/25): 前 25 位 11000000.10101000.00000001.1 = 11000000.10101000.00000001.1 ✓ 匹配
  条目 2 (/24): 前 24 位 11000000.10101000.00000001   = 11000000.10101000.00000001   ✓ 匹配
  条目 3 (/16): 前 16 位 11000000.10101000             = 11000000.10101000             ✓ 匹配
  条目 4 (/0):  前 0 位                                 =                               ✓ 匹配

优先级编码器: 选择最高优先级（条目 1, /25）→ 输出端口 eth3
```

**TCAM 的局限性**：
- 容量有限（通常数千到数万条目）
- 功耗高（并行比较所有条目）
- 成本高
- 不支持范围匹配（需要分解为多个前缀）

### 11.4.3 Trie 树实现

对于软件路由器或路由查找的辅助方案，Trie 树是常用的数据结构：

```python
class MultiBitTrie:
    """多比特 Trie（Stride=4）- 提高查找效率"""

    def __init__(self):
        self.root = TrieNode()

    def insert(self, prefix, prefix_len, next_hop):
        """插入路由条目"""
        node = self.root
        bits_processed = 0

        while bits_processed < prefix_len:
            stride = min(4, prefix_len - bits_processed)
            # 提取 stride 位作为子节点索引
            shift = 32 - bits_processed - stride
            index = (prefix >> shift) & ((1 << stride) - 1)

            if stride == 4:
                # 内部节点，继续下探
                if index not in node.children:
                    node.children[index] = TrieNode()
                node = node.children[index]
            else:
                # 叶子节点，存储路由
                node.next_hop = next_hop
                node.prefix_len = prefix_len

            bits_processed += stride

    def lookup(self, ip_addr):
        """最长前缀匹配查找"""
        node = self.root
        best_match = None
        bits_processed = 0

        while node is not None and bits_processed < 32:
            if node.next_hop is not None:
                best_match = (node.next_hop, node.prefix_len)

            stride = 4
            shift = 32 - bits_processed - stride
            index = (ip_addr >> shift) & 0xF

            if index in node.children:
                node = node.children[index]
                bits_processed += stride
            else:
                break

        return best_match
```

### 11.4.4 查找性能对比

| 方案 | 查找时间 | 插入时间 | 空间 | 硬件支持 |
|------|---------|---------|------|---------|
| TCAM | O(1) | O(1) | 固定 | 硬件 |
| 线性表 | O(N) | O(1) | O(N) | 软件 |
| 二叉 Trie | O(W) | O(W) | O(N×W) | 软件 |
| 多比特 Trie | O(W/k) | O(W/k) | O(N×2^k) | 软件 |
| 压缩 Trie | O(W) | O(W) | 更少 | 软件 |

其中 W=地址宽度（32），N=路由条目数，k=stride 宽度

<div dataComponent="TCAMLookupDemo"></div>

---

## 11.5 交换矩阵

### 11.5.1 交换矩阵的分类

交换矩阵（Switch Fabric）是路由器内部连接输入端口和输出端口的核心组件。

```
三种交换矩阵架构:

1. 基于内存 (Memory-based)
2. 基于总线 (Bus-based)
3. 基于交叉开关 (Crossbar)
```

### 11.5.2 基于内存的交换矩阵

```
┌──────────────────────────────────────┐
│            共享内存                   │
│                                      │
│  输入端口1 ──►┌─────────┐──► 输出端口1│
│  输入端口2 ──►│ 共享    │──► 输出端口2│
│  输入端口3 ──►│ 缓冲区  │──► 输出端口3│
│  输入端口4 ──►│         │──► 输出端口4│
│              └─────────┘            │
│                                      │
│  路由处理器 (CPU) 控制转发决策        │
└──────────────────────────────────────┘
```

**特点**：
- 最简单的架构，早期路由器使用
- CPU 从输入端口读取数据报，查找转发表，写入输出端口
- 速度受 CPU 和内存带宽限制
- 适合低速路由器

**带宽限制**：
$$\text{总带宽} = \frac{\text{内存带宽}}{2}$$
（因子 2 因为每个数据报需要一次写入和一次读出）

### 11.5.3 基于总线的交换矩阵

```
┌──────────────────────────────────────────────┐
│               共享总线                        │
│  ═══════════════════════════════════════════  │
│     │          │          │          │       │
│  输入端口1  输入端口2  输入端口3  输入端口4   │
│     │          │          │          │       │
│  输出端口1  输出端口2  输出端口3  输出端口4   │
└──────────────────────────────────────────────┘
```

**特点**：
- 所有端口共享一条总线
- 同一时刻只有一个端口可以发送
- 需要总线仲裁机制
- 带宽受总线速度限制

**总线带宽计算**：
$$\text{总线带宽} \geq \sum_{i=1}^{N} \text{端口}_i\text{的速率}$$

### 11.5.4 基于交叉开关的交换矩阵

```
交叉开关矩阵 (Crossbar Switch):

              输出端口
              1    2    3    4
           ┌────┬────┬────┬────┐
    输入   │ ×  │    │    │    │  1
    端口   ├────┼────┼────┼────┤
           │    │ ×  │    │ ×  │  2
           ├────┼────┼────┼────┤
           │    │    │ ×  │    │  3
           ├────┼────┼────┼────┤
           │ ×  │    │    │ ×  │  4
           └────┴────┴────┴────┘
    × = 交叉点激活（建立连接）

非阻塞特性:
- 多个输入可以同时向不同的输出发送数据
- N×N 矩阵可以同时建立 N 个不冲突的连接
- 仅当多个输入竞争同一输出时需要调度
```

**交叉开关的关键优势**：
- **非阻塞**：只要输入输出不冲突，多个连接可同时建立
- **高带宽**：总带宽 = N × 端口速率（N 为端口数）
- **可扩展**：通过多级交叉开关构建大规模交换矩阵

<div dataComponent="SwitchFabricDemo"></div>

---

## 11.6 交叉开关调度算法

### 11.6.1 iSLIP 调度算法

iSLIP（Iterative Round-Robin Matching with SLIP）是高性能路由器中最广泛使用的调度算法。

**算法详解**：

```
iSLIP 调度算法（一轮迭代）:

假设 4×4 交叉开关:
  输入端口 1 需要发送到输出端口 2
  输入端口 2 需要发送到输出端口 1
  输入端口 3 需要发送到输出端口 2
  输入端口 4 需要发送到输出端口 4

Step 1: 请求（Request）
  输入 1 → 输出 2: 请求 ✓
  输入 2 → 输出 1: 请求 ✓
  输入 3 → 输出 2: 请求 ✓
  输入 4 → 输出 4: 请求 ✓

Step 2: 授权（Grant）
  输出 1 收到请求: 输入 2
    → 授权给输入 2（轮询指针指向 2，从 2 开始）
  输出 2 收到请求: 输入 1, 输入 3
    → 授权给输入 1（轮询指针指向 1，从 1 开始，1 先被选中）
  输出 3 收到请求: 无
  输出 4 收到请求: 输入 4
    → 授权给输入 4

Step 3: 接受（Accept）
  输入 1 收到授权: 输出 2
    → 接受输出 2
  输入 2 收到授权: 输出 1
    → 接受输出 1
  输入 3 收到授权: 无
    → 等待下一轮
  输入 4 收到授权: 输出 4
    → 接受输出 4

Step 4: 更新指针
  输出 1 的授权指针: 2 → 3（接受成功，指针移动到下一个）
  输出 2 的授权指针: 1 → 2
  输入 1 的接受指针: 2 → 3
  输入 2 的接受指针: 1 → 2
  输入 4 的接受指针: 4 → 1

结果:
  输入 1 → 输出 2 ✓
  输入 2 → 输出 1 ✓
  输入 3 → 输出 2 ✗（冲突，下一轮再试）
  输入 4 → 输出 4 ✓
```

```python
class iSLIPScheduler:
    """iSLIP 调度算法完整实现"""

    def __init__(self, num_ports):
        self.n = num_ports
        self.grant_ptr = [0] * num_ports    # 输出端口授权指针
        self.accept_ptr = [0] * num_ports   # 输入端口接受指针

    def schedule(self, requests):
        """
        requests: N×N 矩阵, requests[i][j]=1 表示输入 i 请求输出 j
        返回: match[i] = j 或 -1
        """
        match = [-1] * self.n
        output_matched = [False] * self.n

        # 多轮迭代提高调度质量
        for iteration in range(1):  # 通常 1 轮足够
            # 阶段 1: 授权 - 每个输出从请求中选择一个输入
            grants = [-1] * self.n
            for j in range(self.n):
                if output_matched[j]:
                    continue
                for offset in range(self.n):
                    i = (self.grant_ptr[j] + offset) % self.n
                    if requests[i][j] == 1 and match[i] == -1:
                        grants[j] = i
                        break

            # 阶段 2: 接受 - 每个输入从授权中选择一个输出
            for i in range(self.n):
                if match[i] != -1:
                    continue
                for offset in range(self.n):
                    j = (self.accept_ptr[i] + offset) % self.n
                    if grants[j] == i:
                        match[i] = j
                        output_matched[j] = True

                        # 更新指针（关键！仅首轮成功时更新）
                        if iteration == 0:
                            self.grant_ptr[j] = (j + 1) % self.n
                            self.accept_ptr[i] = (i + 1) % self.n
                        break

        return match

# 模拟示例
scheduler = iSLIPScheduler(4)
requests = [
    [0, 1, 0, 0],  # 输入 0 请求输出 1
    [1, 0, 0, 0],  # 输入 1 请求输出 0
    [0, 1, 0, 0],  # 输入 2 请求输出 1
    [0, 0, 0, 1],  # 输入 3 请求输出 3
]
match = scheduler.schedule(requests)
print(f"调度结果: {match}")
# 输出: [1, 0, -1, 3]  (输入 2 未匹配，下一轮再试)
```

### 11.6.2 ESLIP 调度算法

ESLIP（Enhanced SLIP）在 iSLIP 基础上改进，支持优先级调度：

```python
class ESLIPScheduler:
    """ESLIP - 支持优先级的调度算法"""

    def __init__(self, num_ports, num_priorities=8):
        self.n = num_ports
        self.priorities = num_priorities
        # 每个优先级有独立的指针
        self.grant_ptr = [[0] * num_ports for _ in range(num_priorities)]
        self.accept_ptr = [[0] * num_ports for _ in range(num_priorities)]

    def schedule(self, requests):
        """
        requests[i][j] = priority (0-7, 0=最高)
        requests[i][j] = -1 表示无请求
        """
        match = [-1] * self.n

        # 按优先级从高到低调度
        for prio in range(self.priorities):
            prio_requests = [[1 if requests[i][j] == prio else 0
                              for j in range(self.n)]
                             for i in range(self.n)]

            # 对当前优先级运行 iSLIP
            result = self._islip_for_priority(prio_requests, prio)

            # 更新匹配结果
            for i in range(self.n):
                if match[i] == -1 and result[i] != -1:
                    match[i] = result[i]

        return match

    def _islip_for_priority(self, requests, prio):
        """对特定优先级运行 iSLIP"""
        match = [-1] * self.n
        output_matched = [False] * self.n

        grants = [-1] * self.n
        for j in range(self.n):
            if output_matched[j]:
                continue
            for offset in range(self.n):
                i = (self.grant_ptr[prio][j] + offset) % self.n
                if requests[i][j] == 1 and match[i] == -1:
                    grants[j] = i
                    break

        for i in range(self.n):
            if match[i] != -1:
                continue
            for offset in range(self.n):
                j = (self.accept_ptr[prio][i] + offset) % self.n
                if grants[j] == i:
                    match[i] = j
                    output_matched[j] = True
                    self.grant_ptr[prio][j] = (j + 1) % self.n
                    self.accept_ptr[prio][i] = (i + 1) % self.n
                    break

        return match
```

<div dataComponent="iSLIPSchedulerDemo"></div>

---

## 11.7 输出端口处理

### 11.7.1 输出端口管线

```
输出端口处理管线:

来自交换矩阵的数据报
          │
          ▼
┌─────────────────────────────────────┐
│         排队和缓冲                   │
│  ┌─────────────────────────────┐    │
│  │  队列 1 (高优先级)           │    │
│  │  队列 2 (中优先级)           │    │
│  │  队列 3 (低优先级)           │    │
│  │  队列 4 (尽力而为)           │    │
│  └─────────────────────────────┘    │
│                                     │
│  调度器: WFQ / WRR / Priority     │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│         链路层处理                   │
│  1. 添加链路层首部                   │
│  2. 生成 FCS (CRC-32)              │
│  3. 802.1Q VLAN 标签处理           │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│         物理层处理                   │
│  1. 线路编码                        │
│  2. 电光转换                        │
│  3. 发送到物理链路                   │
└─────────────────────────────────────┘
```

### 11.7.2 HOL 队头阻塞（Head-of-Line Blocking）

**HOL 阻塞** 是输入排队交换矩阵中的严重问题：

```
HOL 阻塞示例:

输入端口 1: 队列 [→输出2] [→输出1] [→输出3]
输入端口 2: 队列 [→输出2] [→输出3] [→输出1]
输入端口 3: 队列 [→输出1] [→输出2] [→输出3]

时间 1:
  输入 1 的队头: →输出2 (可以发送)
  输入 2 的队头: →输出2 (冲突！)
  输入 3 的队头: →输出1 (可以发送)

假设输入 1 赢得输出 2:
  输入 1: 发送 →输出2 ✓
  输入 2: 队头阻塞！虽然 →输出3 可以发送，但被队头的 →output2 挡住
  输入 3: 发送 →output1 ✓

结果: 输入 2 的 →output3 被无端阻塞
```

**解决方案**：

1. **虚拟输出队列（VOQ）**：每个输入端口为每个输出端口维护一个独立的队列
2. **输出排队**：数据报到达输出端口后再排队（需要更高的交换矩阵速度）
3. **组合输入输出排队（CIOQ）**：结合输入和输出排队的优点

```python
class VirtualOutputQueue:
    """虚拟输出队列 (VOQ) - 解决 HOL 阻塞"""

    def __init__(self, num_ports):
        self.n = num_ports
        # 每个输入端口有 N 个虚拟输出队列
        self.voq = [[[] for _ in range(num_ports)] for _ in range(num_ports)]

    def enqueue(self, input_port, output_port, packet):
        """将数据报放入对应的虚拟输出队列"""
        self.voq[input_port][output_port].append(packet)

    def dequeue(self, input_port, output_port):
        """从指定虚拟输出队列取出数据报"""
        if self.voq[input_port][output_port]:
            return self.voq[input_port][output_port].pop(0)
        return None

    def get_request_matrix(self):
        """生成请求矩阵（用于 iSLIP 调度）"""
        requests = [[0] * self.n for _ in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                if self.voq[i][j]:
                    requests[i][j] = 1
        return requests
```

<div dataComponent="HOLBlockingDemo"></div>

---

## 11.8 队列管理

### 11.8.1 主动队列管理（AQM）

**尾部丢弃（Tail Drop）** 的问题：
- 当队列满时，丢弃所有新到达的数据报
- 导致多个 TCP 连接同时进入拥塞控制（全局同步）
- 队列延迟增大

**RED（Random Early Detection）** 算法：

```python
class REDQueueManager:
    """RED (Random Early Detection) 主动队列管理"""

    def __init__(self, queue_size, min_th, max_th, max_p, w_q):
        self.queue_size = queue_size
        self.min_th = min_th        # 最小阈值
        self.max_th = max_th        # 最大阈值
        self.max_p = max_p          # 最大丢弃概率
        self.w_q = w_q              # 平均队列长度权重
        self.avg_queue_len = 0      # 平均队列长度
        self.count = 0              # 自上次丢弃以来的包数

    def process_arrival(self, packet):
        """处理到达的数据报"""

        # 1. 计算平均队列长度（指数加权移动平均）
        current_len = self.get_current_queue_length()
        self.avg_queue_len = (1 - self.w_q) * self.avg_queue_len + \
                             self.w_q * current_len

        # 2. 根据平均队列长度决定是否丢弃
        if self.avg_queue_len < self.min_th:
            # 低于最小阈值：不丢弃
            self.count = 0
            return 'enqueue'

        elif self.avg_queue_len >= self.max_th:
            # 高于最大阈值：丢弃所有
            return 'drop'

        else:
            # 在两个阈值之间：概率丢弃
            self.count += 1
            # 计算丢弃概率
            pb = self.max_p * (self.avg_queue_len - self.min_th) / \
                 (self.max_th - self.min_th)
            pa = pb / (1 - self.count * pb)

            # 随机决定是否丢弃
            if random.random() < pa:
                self.count = 0
                return 'drop'
            else:
                return 'enqueue'

    def get_current_queue_length(self):
        """获取当前队列长度"""
        return len(self.queue)
```

### 11.8.2 加权公平排队（WFQ）

**WFQ（Weighted Fair Queuing）** 为每个流维护一个独立的队列，并按权重分配带宽：

```python
class WFQScheduler:
    """加权公平排队 (WFQ) 调度器"""

    def __init__(self):
        self.flows = {}  # flow_id -> (weight, queue, virtual_time)
        self.virtual_time = 0
        self.last_virtual_time = {}

    def add_flow(self, flow_id, weight):
        """添加流"""
        self.flows[flow_id] = {
            'weight': weight,
            'queue': [],
            'virtual_finish_time': 0,
        }

    def enqueue(self, flow_id, packet):
        """数据报入队"""
        if flow_id not in self.flows:
            return False

        flow = self.flows[flow_id]
        # 计算虚拟完成时间
        virtual_service_time = len(packet) / flow['weight']
        flow['virtual_finish_time'] = max(
            self.virtual_time,
            flow['virtual_finish_time']
        ) + virtual_service_time

        flow['queue'].append((flow['virtual_finish_time'], packet))
        return True

    def dequeue(self):
        """选择虚拟完成时间最小的流"""
        best_flow = None
        best_finish_time = float('inf')

        for flow_id, flow in self.flows.items():
            if flow['queue']:
                finish_time = flow['queue'][0][0]
                if finish_time < best_finish_time:
                    best_finish_time = finish_time
                    best_flow = flow_id

        if best_flow is not None:
            self.virtual_time = best_finish_time
            _, packet = self.flows[best_flow]['queue'].pop(0)
            return best_flow, packet

        return None, None

# 示例
wfq = WFQScheduler()
wfq.add_flow('voice', weight=5)    # 语音流，权重 5
wfq.add_flow('data', weight=1)     # 数据流，权重 1
wfq.add_flow('video', weight=3)    # 视频流，权重 3

# 语音流获得 5/(5+1+3) = 55.6% 带宽
# 视频流获得 3/(5+1+3) = 33.3% 带宽
# 数据流获得 1/(5+1+3) = 11.1% 带宽
```

### 11.8.3 优先级排队 vs WFQ

| 特性 | 优先级排队 | WFQ |
|------|----------|-----|
| 调度策略 | 严格优先级 | 按权重公平分配 |
| 延迟保证 | 高优先级有最低延迟 | 每个流有公平的延迟 |
| 带宽分配 | 高优先级可能饿死低优先级 | 按权重分配 |
| 适用场景 | 语音、视频等实时流量 | 混合流量环境 |
| 实现复杂度 | 简单 | 中等 |

<div dataComponent="QueueManagementDemo"></div>

---

## 11.9 缓冲区管理

### 11.9.1 缓冲区大小计算

路由器输出端口的缓冲区大小对性能至关重要。

**经验法则（Rule of Thumb）**：

$$\text{缓冲区大小} = \frac{RTT \times C}{\sqrt{N}}$$

其中：
- $RTT$ = 平均往返时间（如 250ms）
- $C$ = 链路容量（如 10 Gbps）
- $N$ = 流的数量

例如：RTT=250ms, C=10Gbps, N=10000 流

$$\text{缓冲区} = \frac{0.25 \times 10 \times 10^9}{\sqrt{10000}} = \frac{2.5 \times 10^9}{100} = 25 \times 10^6 \text{bits} = 3.125 \text{MB}$$

### 11.9.2 缓冲区管理策略

```python
class BufferManager:
    """路由器缓冲区管理器"""

    def __init__(self, total_size):
        self.total_size = total_size
        self.used = 0
        self.queues = {}  # port -> queue

    def allocate(self, port, packet_size):
        """尝试为数据报分配缓冲区"""

        # 策略 1: 共享缓冲池（所有端口共享）
        if self.used + packet_size <= self.total_size:
            self.used += packet_size
            return True

        # 策略 2: 尝试从最低优先级队列借用
        lowest_prio = self._find_lowest_priority_queue()
        if lowest_prio and self._can_evict(lowest_prio, packet_size):
            self._evict(lowest_prio, packet_size)
            self.used += packet_size
            return True

        return False  # 缓冲区不足，丢弃

    def release(self, packet_size):
        """释放缓冲区"""
        self.used = max(0, self.used - packet_size)
```

---

## 11.10 现代路由器架构

### 11.10.1 Cisco 路由器架构

**Cisco CRS-1（运营商级路由器）**：

```
Cisco CRS-1 架构:

┌─────────────────────────────────────────────────────┐
│                    线路卡 (Line Cards)                │
│  ┌──────┐  ┌──────┐  ┌──────┐       ┌──────┐       │
│  │ LC 1 │  │ LC 2 │  │ LC 3 │  ...  │ LC N │       │
│  │      │  │      │  │      │       │      │       │
│  │输入/ │  │输入/ │  │输入/ │       │输入/ │       │
│  │输出  │  │输出  │  │输出  │       │输出  │       │
│  │端口  │  │端口  │  │端口  │       │端口  │       │
│  └──┬───┘  └──┬───┘  └──┬───┘       └──┬───┘       │
│     │         │         │              │            │
├─────┼─────────┼─────────┼──────────────┼────────────┤
│     ▼         ▼         ▼              ▼            │
│  ┌──────────────────────────────────────────────┐   │
│  │         三级 Benes 交换矩阵                   │   │
│  │    (3-Stage Benes Switch Fabric)              │   │
│  │                                               │   │
│  │  Stage1    Stage2    Stage3                   │   │
│  │  ┌────┐   ┌────┐   ┌────┐                   │   │
│  │  │    │──►│    │──►│    │                   │   │
│  │  └────┘   └────┘   └────┘                   │   │
│  └──────────────────────────────────────────────┘   │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │       路由处理器 (Route Processor)            │   │
│  │  运行 IOS XR，执行路由协议                    │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘

特点:
- 三级 Benes 交换矩阵，支持无阻塞交换
- 分布式转发：每个线路卡独立转发
- 冗余设计：所有组件可热备份
- 支持 72 Tbps 交换容量
```

### 11.10.2 Juniper 路由器架构

**Juniper T4000（核心路由器）**：

```
Juniper T4000 架构:

┌──────────────────────────────────────────────────────┐
│                   交换矩阵 (SFEB)                    │
│  ┌──────────────────────────────────────────────┐    │
│  │         中央交换矩阵 (Switch Fabric)          │    │
│  │         多级交叉开关                          │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │              FPC (Flexible PIC Concentrator)    │  │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐  │  │
│  │  │ PIC 1  │ │ PIC 2  │ │ PIC 3  │ │ PIC 4  │  │  │
│  │  │(接口卡)│ │(接口卡)│ │(接口卡)│ │(接口卡)│  │  │
│  │  └────────┘ └────────┘ └────────┘ └────────┘  │  │
│  │                                                │  │
│  │  I-Chip (Ingress) → 查找/分类/调度             │  │
│  │  PFE (Packet Forwarding Engine)                │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │            RE (Routing Engine)                │    │
│  │  运行 JUNOS，执行路由协议                     │    │
│  └──────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘

特点:
- 分离的控制平面和数据平面
- I-Chip 和 PFE 芯片负责转发
- 支持 4+ Tbps 吞吐量
- JUNOS 操作系统
```

### 11.10.3 软件定义路由器（SDN）

SDN 将控制平面从路由器中分离出来，集中管理：

```
传统路由器:                 SDN 架构:
┌──────────────┐           ┌──────────────┐
│ 控制平面     │           │   SDN 控制器  │
│ (路由协议)   │           │   (集中式)    │
├──────────────┤           └──────┬───────┘
│ 数据平面     │                  │ OpenFlow
│ (转发引擎)   │                  │
└──────────────┘           ┌──────┴───────┐
                           │  数据平面    │
                           │  (交换机)    │
                           └──────────────┘
```

<div dataComponent="ModernRouterDemo"></div>

---

## 11.11 路由器性能指标

### 11.11.1 关键性能参数

| 指标 | 说明 | 典型值 |
|------|------|--------|
| 吞吐量 | 每秒转发的数据量 | 接入: 100Gbps, 核心: 数 Tbps |
| 包转发率 | 每秒转发的数据报数 (pps) | 100Gbps线速: 148.8Mpps |
| 路由表容量 | 可存储的路由条目数 | 数十万到数百万 |
| 延迟 | 数据报通过路由器的时间 | 10-100 μs |
| 抖动 | 延迟的变化程度 | 1-10 μs |
| 丢包率 | 被丢弃的数据报比例 | < 0.01% (正常负载) |
| 背板带宽 | 交换矩阵的总带宽 | 数 Tbps |

### 11.11.2 线速转发计算

```python
def calculate_router_performance(link_speed_gbps, num_ports):
    """计算路由器性能指标"""

    # 最小帧大小: 64 字节 + 8 字节前导码 + 12 字节帧间隙 = 84 字节
    min_frame_size = 84 * 8  # 比特

    # 单端口线速
    wire_speed_pps = link_speed_gbps * 1e9 / min_frame_size

    # 总吞吐量
    total_throughput = link_speed_gbps * num_ports

    # 总包转发率（假设所有端口同时线速）
    total_pps = wire_speed_pps * num_ports

    # 每个数据报的处理时间
    processing_time_ns = 1e9 / wire_speed_pps

    return {
        'wire_speed_pps': wire_speed_pps,
        'total_throughput_gbps': total_throughput,
        'total_pps': total_pps,
        'processing_time_ns': processing_time_ns,
    }

# 示例: 48 口千兆路由器
result = calculate_router_performance(1, 48)
print(f"单端口线速: {result['wire_speed_pps']:,.0f} pps")
print(f"总吞吐量: {result['total_throughput_gbps']} Gbps")
print(f"总包转发率: {result['total_pps']:,.0f} pps")
print(f"每包处理时间: {result['processing_time_ns']:.1f} ns")

# 示例: 4 口万兆核心路由器
result = calculate_router_performance(10, 4)
print(f"\n单端口线速: {result['wire_speed_pps']:,.0f} pps")
print(f"总吞吐量: {result['total_throughput_gbps']} Gbps")
print(f"每包处理时间: {result['processing_time_ns']:.1f} ns")
```

---

## 11.12 本章总结

| 概念 | 核心要点 |
|------|---------|
| 路由器架构 | 输入端口 + 交换矩阵 + 输出端口 + 路由处理器 |
| 输入端口 | 三层处理管线：物理→链路→网络（TCAM查找） |
| 转发 vs 路由 | 数据平面（纳秒）vs 控制平面（秒级） |
| 最长前缀匹配 | TCAM（硬件O(1)）或 Trie 树（软件） |
| 交换矩阵 | 内存/总线/交叉开关，iSLIP 调度算法 |
| HOL 阻塞 | VOQ 虚拟输出队列解决方案 |
| 队列管理 | RED 主动队列管理，WFQ 加权公平排队 |
| 缓冲区 | 经验公式 RTT×C/√N，共享缓冲池 |
| 现代路由器 | Cisco CRS-1, Juniper T4000, SDN |

<div dataComponent="ChapterSummaryQuiz"></div>

---

## 练习题

**1. 分析题**：画出路由器输入端口的分层处理管线，说明每一层的主要功能。

**2. 算法题**：对以下转发表，使用 Trie 树实现最长前缀匹配查找：
   - 10.0.0.0/8 → eth0
   - 10.1.0.0/16 → eth1
   - 10.1.1.0/24 → eth2
   - 10.1.1.128/25 → eth3

**3. 调度题**：用 iSLIP 算法对 4×4 交叉开关进行调度，输入端口请求如下：
   - 输入 0 → 输出 2
   - 输入 1 → 输出 0
   - 输入 2 → 输出 2
   - 输入 3 → 输出 3

**4. 计算题**：一个 10 Gbps 链路的路由器，RTT 为 100ms，有 10000 个活跃流。计算所需的缓冲区大小。

**5. 比较题**：对比 RED 和尾部丢弃两种队列管理策略的优缺点。

**6. 设计题**：设计一个简化的路由器数据平面，支持 IPv4 转发、最长前缀匹配和 FIFO 排队。

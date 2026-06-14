# Chapter 7: 数据链路层 — 网桥与交换技术

> **学习目标**：
> - 理解透明网桥的工作原理和学习型网桥的转发/过滤/学习算法
> - 掌握生成树协议（STP）的详细步骤：根桥选举、端口角色、端口状态转换
> - 了解快速生成树协议（RSTP）相比 STP 的改进
> - 理解交换机的两大架构（共享内存与交叉开关）及其调度算法
> - 掌握 VLAN Trunking 和链路聚合（LACP）的原理与配置
> - 了解 MAC-in-MAC 技术在运营商网络中的应用
> - 深入分析网桥转发引擎、STP 状态机、交换矩阵调度算法和 VLAN 处理器的内部实现

---

## 7.1 透明网桥

### 7.1.1 网桥的起源

在以太网规模扩大时，单个冲突域的效率急剧下降。网桥（Bridge）应运而生，用于连接多个以太网段，隔离冲突域，同时保持广播域的连通性。

**透明网桥（Transparent Bridge）** 的"透明"含义是：网桥的存在对网络中的主机完全透明——主机无需知道网桥的存在，也无需任何配置。网桥自动完成帧的转发决策。

### 7.1.2 学习型网桥算法

学习型网桥维护一个 **MAC 地址表**（也称转发表），记录每个 MAC 地址位于哪个端口。表项通过**自学习**过程自动建立。

**三个核心操作：学习（Learn）、转发（Forward）、过滤（Filter）**：

```
收到帧（从端口 P 进入）：

1. 【学习】将帧的源 MAC 地址与端口 P 的映射记录到 MAC 表
   MAC表[源MAC] = (端口P, 时间戳)

2. 【查找】在 MAC 表中查找目的 MAC 地址

3. 【决策】
   - 如果找到且出端口 ≠ P：
     → 【转发】将帧从出端口发送
   - 如果找到且出端口 = P：
     → 【过滤】丢弃帧（源和目的在同一网段）
   - 如果未找到：
     → 【泛洪】将帧从除 P 外的所有端口发送（未知单播泛洪）
   - 如果目的 MAC 是广播/组播：
     → 【泛洪】将帧从除 P 外的所有端口发送
```

```python
class LearningBridge:
    """学习型网桥模拟"""

    def __init__(self, num_ports):
        self.mac_table = {}  # MAC -> (port, timestamp)
        self.num_ports = num_ports
        self.aging_time = 300  # 5 分钟老化

    def process_frame(self, frame, in_port):
        """处理收到的帧"""
        src_mac = frame['src_mac']
        dst_mac = frame['dst_mac']

        # 1. 学习：记录源 MAC 与入端口的映射
        self.mac_table[src_mac] = (in_port, time.time())

        # 2. 查找目的 MAC
        if dst_mac == 'ff:ff:ff:ff:ff:ff':
            # 广播帧：泛洪
            return self._flood(in_port)

        if dst_mac in self.mac_table:
            out_port, ts = self.mac_table[dst_mac]
            if time.time() - ts > self.aging_time:
                # 表项过期，删除后泛洪
                del self.mac_table[dst_mac]
                return self._flood(in_port)

            if out_port == in_port:
                # 源和目的在同一端口：过滤（丢弃）
                return []
            else:
                # 转发到已知端口
                return [out_port]
        else:
            # 未知单播：泛洪
            return self._flood(in_port)

    def _flood(self, in_port):
        """泛洪：除入端口外的所有端口"""
        return [p for p in range(self.num_ports) if p != in_port]
```

### 7.1.3 MAC 地址表老化

MAC 表条目有生存时间（通常 300 秒），超时后自动删除。原因：
1. 设备可能移动到其他端口
2. 设备可能下线
3. 防止表空间被永久占用

```c
/* MAC 地址表老化定时器（每秒执行一次） */
void mac_table_aging(struct mac_table *table) {
    time_t now = time(NULL);
    for (int i = 0; i < table->capacity; i++) {
        if (table->entries[i].valid) {
            if (now - table->entries[i].timestamp > AGING_TIME) {
                table->entries[i].valid = 0;
                table->count--;
            }
        }
    }
}
```

<div data-component="LearningBridgeDemo"></div>

### 7.1.4 网桥的优缺点

| 优点 | 缺点 |
|------|------|
| 隔离冲突域 | 不能隔离广播域 |
| 自学习，无需配置 | 泛洪造成带宽浪费 |
| 可连接不同速率的网段 | 无法连接不同拓扑的网络 |
| 帧过滤减少不必要流量 | 广播风暴风险 |

---

## 7.2 生成树协议（STP）

### 7.2.1 冗余与环路问题

为了提高可靠性，网络中常部署冗余链路。但冗余会导致**二层环路**，引发严重问题：

1. **广播风暴**：广播帧在环路中无限循环
2. **MAC 表震荡**：同一 MAC 在不同端口间不断切换
3. **重复帧接收**：主机收到同一帧的多个副本

STP（Spanning Tree Protocol, IEEE 802.1D）通过逻辑上阻塞某些端口，将网状拓扑变成无环的树状拓扑。

### 7.2.2 STP 的核心概念

**BPDU（Bridge Protocol Data Unit）**：网桥之间交换的控制帧，携带拓扑信息。

BPDU 关键字段：
| 字段 | 说明 |
|------|------|
| Root Bridge ID | 发送者认为的根桥 ID |
| Root Path Cost | 发送者到根桥的路径开销 |
| Bridge ID | 发送者的桥 ID |
| Port ID | 发送端口的 ID |
| Message Age | BPDU 的年龄 |
| Max Age | BPDU 的最大存活时间（默认 20 秒） |
| Hello Time | BPDU 发送间隔（默认 2 秒） |
| Forward Delay | 端口状态转换延迟（默认 15 秒） |

**桥 ID（Bridge ID）** = 优先级（2 字节）+ MAC 地址（6 字节）。优先级范围 0-65535，默认 32768，越小越优先。

### 7.2.3 根桥选举

1. 初始时，所有网桥都声称自己是根桥
2. 网桥比较收到的 BPDU 中的 Root Bridge ID 与自己的 Bridge ID
3. 选择 Bridge ID 最小的网桥为根桥
4. 如果收到更优（更小）的 Bridge ID，更新自己的 Root Bridge ID 并转发

```
选举过程示例：

    Bridge A (ID=1000)        Bridge B (ID=2000)        Bridge C (ID=3000)
         │                         │                         │
    初始状态：三个网桥都认为自己是根桥
         │                         │                         │
    交换 BPDU 后：
    A 认为自己是根 → B 和 C 收到 A 的 BPDU → B 和 C 认可 A 是根
         │
    最终：Bridge A 成为根桥（ID 最小）
```

### 7.2.4 端口角色

| 端口角色 | 说明 | 转发状态 |
|---------|------|---------|
| 根端口（Root Port, RP） | 每个非根网桥上到达根桥路径开销最小的端口 | 转发 |
| 指定端口（Designated Port, DP） | 每个网段上到根桥路径开销最小的网桥的端口 | 转发 |
| 非指定端口（Non-designated Port） | 既不是根端口也不是指定端口 | 阻塞 |

**路径开销**（Path Cost）与链路速率相关：

| 链路速率 | STP 开销（802.1D） | STP 开销（802.1t） |
|---------|-------------------|-------------------|
| 10 Mbps | 100 | 2,000,000 |
| 100 Mbps | 19 | 200,000 |
| 1 Gbps | 4 | 20,000 |
| 10 Gbps | 2 | 2,000 |
| 100 Gbps | — | 200 |

### 7.2.5 端口状态转换

STP 的端口有 5 种状态：

```
                    ┌─────────────┐
                    │  Disabled   │
                    └──────┬──────┘
                           │ 启用端口
                           ▼
                    ┌─────────────┐
              ┌─────│ Blocking    │◄────┐
              │     └──────┬──────┘     │ Max Age 超时
              │            │ 收到更优BPDU│ 或拓扑变化
              │            ▼            │
              │     ┌─────────────┐     │
              │     │ Listening   │─────┘
              │     └──────┬──────┘
              │            │ Forward Delay (15s)
              │            ▼
              │     ┌─────────────┐
              │     │ Learning    │
              │     └──────┬──────┘
              │            │ Forward Delay (15s)
              │            ▼
              │     ┌─────────────┐
              └────►│ Forwarding  │
                    └─────────────┘
```

| 状态 | 持续时间 | 行为 |
|------|---------|------|
| Disabled | — | 端口关闭，不参与 STP |
| Blocking | 20 秒（Max Age） | 不转发数据帧，只接收 BPDU |
| Listening | 15 秒（Forward Delay） | 不转发数据帧，发送和接收 BPDU，选举端口角色 |
| Learning | 15 秒（Forward Delay） | 不转发数据帧，学习 MAC 地址 |
| Forwarding | 持续 | 正常转发数据帧，学习 MAC 地址 |

从 Blocking 到 Forwarding 需要 **30-50 秒**，这是 STP 的主要缺点。

### 7.2.6 STP 收敛过程示例

```
网络拓扑:
       ┌─────────┐
       │Bridge A  │ (Root Bridge, ID=1000)
       │  ID=1000 │
       └──┬───┬──┘
     P1   │   │   P2
          │   │
    ┌─────┘   └─────┐
    │               │
┌───┴────┐     ┌────┴───┐
│Bridge B│     │Bridge C│
│ID=2000 │     │ID=3000 │
└───┬────┘     └────┬───┘
    │   P3          │   P4
    └───────┬───────┘
            │
       ┌────┴────┐
       │Bridge D │
       │ID=4000  │
       └─────────┘

收敛步骤:
1. 根桥选举：A (ID=1000) 成为根桥

2. 非根网桥确定根端口：
   - B 的 RP: 到 A 直连端口 (开销=4, 1Gbps)
   - C 的 RP: 到 A 直连端口 (开销=4, 1Gbps)
   - D 的 RP: 通过 B 到 A (开销=4+4=8) 或通过 C 到 A (开销=4+4=8)
     假设 D 选择通过 B 的端口

3. 每个网段确定指定端口：
   - A-B 段: A 的 P1 (指定端口)
   - A-C 段: A 的 P2 (指定端口)
   - B-D 段: B 的 P3 (到 D 开销=8 < C 到 D 开销=8+4=12)
   - C-D 段: C 的 P4 (到 D 开销=8 < B 到 D 开销=4+4=8)
     如果开销相等，比较 Bridge ID

4. 阻塞端口：
   - C-D 段上 D 的端口被阻塞（C 的 P4 是指定端口）
   - 或 B-D 段上 D 的端口被阻塞（取决于具体开销比较）
```

<div data-component="STPSimulator"></div>

---

## 7.3 快速生成树协议（RSTP）

### 7.3.1 STP 的问题

| 问题 | 说明 |
|------|------|
| 收敛慢 | Blocking → Forwarding 需要 30-50 秒 |
| 无主动通知 | 依赖 BPDU 超时检测拓扑变化 |
| 端口状态多 | 5 种状态，转换过程复杂 |
| 单棵生成树 | 所有 VLAN 共享一棵树，无法实现负载均衡 |

### 7.3.2 RSTP 的改进（IEEE 802.1w）

RSTP 将收敛时间缩短到 **1-2 秒**，主要改进：

**1. 新的端口角色**：

| RSTP 端口角色 | 说明 |
|-------------|------|
| Root Port (RP) | 同 STP |
| Designated Port (DP) | 同 STP |
| Alternate Port | 根端口的备份，到根桥的替代路径 |
| Backup Port | 指定端口的备份，同一网段的替代端口 |
| Disabled Port | 管理关闭的端口 |

**2. 简化的端口状态**：

| STP 状态 | RSTP 状态 | 说明 |
|---------|----------|------|
| Disabled | Discarding | 不学习不转发 |
| Blocking | Discarding | 不学习不转发 |
| Listening | Discarding | 不学习不转发 |
| Learning | Learning | 学习但不转发 |
| Forwarding | Forwarding | 学习并转发 |

**3. BPDU 处理改进**：
- STP：非根桥只转发根桥的 BPDU
- RSTP：每个网桥自己发送 BPDU（充当 Hello），连续 3 个 Hello 未收到即认为邻居丢失

**4. Proposal/Agreement 机制**：
- 新链路激活时，指定端口立即发送 Proposal BPDU
- 下游网桥收到后阻塞所有非边缘端口，回复 Agreement
- 指定端口收到 Agreement 后立即进入 Forwarding 状态

```python
class RSTPPort:
    """RSTP 端口状态机"""

    def __init__(self):
        self.role = 'Disabled'
        self.state = 'Discarding'
        self.proposal = False
        self.agreement = False

    def receive_bpdu(self, bpdu):
        """收到 BPDU 时的状态转换"""
        if self.role == 'Root Port':
            if bpdu.is_proposal:
                # 阻塞所有非边缘端口，发送 Agreement
                self.send_agreement()
                self.state = 'Forwarding'

    def become_designated(self):
        """成为指定端口"""
        self.role = 'Designated'
        self.state = 'Discarding'
        self.proposal = True
        # 发送 Proposal BPDU
        self.send_bpdu(proposal=True)
        # 等待 Agreement 后转为 Forwarding

    def receive_agreement(self):
        """收到 Agreement"""
        if self.role == 'Designated' and self.proposal:
            self.state = 'Forwarding'
            self.proposal = False
```

### 7.3.3 RSTP vs STP 对比

| 特性 | STP (802.1D) | RSTP (802.1w) |
|------|-------------|---------------|
| 收敛时间 | 30-50 秒 | 1-2 秒 |
| 端口角色 | 3 种 | 5 种 |
| 端口状态 | 5 种 | 3 种 |
| BPDU 发送 | 仅根桥主动 | 每个网桥主动 |
| 拓扑变化检测 | 依赖超时 | 主动通知 |
| 向后兼容 | — | 兼容 STP |

<div data-component="RSTPComparison"></div>

---

## 7.4 交换机架构

### 7.4.1 交换机与网桥的区别

| 特性 | 网桥 | 交换机 |
|------|------|--------|
| 端口数 | 2-16 | 24-48（接入）到数百（核心） |
| 转发方式 | 软件 | 硬件（ASIC） |
| 速度 | 较慢 | 线速转发 |
| 功能 | 基本转发 | VLAN、QoS、ACL、镜像等 |

### 7.4.2 共享内存架构

所有端口共享一个中央内存池，帧存入内存后通过查找转发表决定从哪个端口输出。

```
         端口1  端口2  端口3  ...  端口N
           │      │      │           │
           ▼      ▼      ▼           ▼
        ┌─────────────────────────────────┐
        │       共享内存缓冲区            │
        │  ┌───┬───┬───┬───┬───┬───┬───┐ │
        │  │帧1│帧2│帧3│帧4│   │   │   │ │
        │  └───┴───┴───┴───┴───┴───┴───┘ │
        │                                 │
        │  转发决策引擎 (查找 MAC 表)     │
        └─────────────────────────────────┘
           │      │      │           │
           ▼      ▼      ▼           ▼
         端口1  端口2  端口3  ...  端口N
```

**优点**：
- 端口间共享内存，灵活分配缓冲区
- 实现简单，成本低

**缺点**：
- 内存带宽是瓶颈：所有端口读写同一内存，带宽 = 端口数 × 端口速率
- 适合端口数较少（<24）或速率较低的场景

**内存带宽计算**：
$$\text{所需带宽} = 2 \times N \times \text{端口速率}$$

（因子 2 是因为每个帧需要一次写入和一次读出）

例如 48 口千兆交换机：$2 \times 48 \times 1 \text{Gbps} = 96 \text{Gbps}$

### 7.4.3 交叉开关架构（Crossbar）

交叉开关矩阵是一种非阻塞交换结构，允许多个输入端口同时向不同的输出端口发送数据。

```
              输出端口
              1    2    3    4
           ┌────┬────┬────┬────┐
    输入   │ ╳  │    │    │    │  1
    端口   ├────┼────┼────┼────┤
           │    │ ╳  │    │ ╳  │  2
           ├────┼────┼────┼────┤
           │    │    │ ╳  │    │  3
           ├────┼────┼────┼────┤
           │ ╳  │    │    │ ╳  │  4
           └────┴────┴────┴────┘
    ╳ = 交叉点（crosspoint），可开可关
```

**关键特性**：
- **非阻塞**：只要输入和输出不冲突，多个连接可同时建立
- **扇出冲突**：同一输出端口同时被多个输入端口请求时需调度
- **交叉点数**：$N \times N$ 矩阵需要 $N^2$ 个交叉点

```python
class CrossbarSwitch:
    """交叉开关交换矩阵"""

    def __init__(self, num_ports):
        self.n = num_ports
        # 调度矩阵: input[i] -> output[schedule[i]]
        self.schedule = [-1] * num_ports

    def schedule_simple(self, requests):
        """
        requests[i] = j 表示输入端口 i 要发送到输出端口 j
        requests[i] = -1 表示输入端口 i 无请求
        返回: 调度结果 schedule[i] = 分配给输入 i 的输出端口
        """
        self.schedule = [-1] * self.n
        output_used = [False] * self.n

        # 简单轮询调度
        for i in range(self.n):
            if requests[i] != -1 and not output_used[requests[i]]:
                self.schedule[i] = requests[i]
                output_used[requests[i]] = True

        return self.schedule
```

### 7.4.4 iSLIP 调度算法

iSLIP（Iterative Round-Robin Matching with SLIP）是高性能交换机中最广泛使用的调度算法。

**算法步骤**（一轮迭代）：

1. **请求（Request）**：每个输入端口向其所需输出端口发送请求
2. **授权（Grant）**：每个输出端口从收到的请求中，按轮询指针选择一个授权
3. **接受（Accept）**：每个输入端口从收到的授权中，按轮询指针选择一个接受
4. **更新指针**：只有在匹配成功时，才将授权/接受指针更新到下一个位置

```python
class iSLIPScheduler:
    """iSLIP 调度算法"""

    def __init__(self, num_ports):
        self.n = num_ports
        self.grant_ptr = [0] * num_ports    # 输出端口的授权指针
        self.accept_ptr = [0] * num_ports   # 输入端口的接受指针

    def schedule(self, requests, num_iterations=1):
        """
        requests[i][j] = 1 表示输入 i 请求输出 j
        返回: match[i] = j 表示输入 i 匹配到输出 j
        """
        match = [-1] * self.n
        output_matched = [False] * self.n

        for iteration in range(num_iterations):
            # 步骤 1: 请求 - 每个输入向所有请求的输出发送请求
            # requests 已经给出了请求信息

            # 步骤 2: 授权 - 每个输出从请求中选择
            grants = [-1] * self.n
            for j in range(self.n):
                if output_matched[j]:
                    continue
                # 从 grant_ptr[j] 开始轮询
                for offset in range(self.n):
                    i = (self.grant_ptr[j] + offset) % self.n
                    if requests[i][j] == 1 and match[i] == -1:
                        grants[j] = i
                        break

            # 步骤 3: 接受 - 每个输入从授权中选择
            for i in range(self.n):
                if match[i] != -1:
                    continue
                # 从 accept_ptr[i] 开始轮询
                for offset in range(self.n):
                    j = (self.accept_ptr[i] + offset) % self.n
                    if grants[j] == i:
                        match[i] = j
                        output_matched[j] = True

                        # 步骤 4: 更新指针（仅在匹配成功时）
                        self.grant_ptr[j] = (j + 1) % self.n
                        self.accept_ptr[i] = (i + 1) % self.n
                        break

        return match
```

<div data-component="CrossbarSchedulerDemo"></div>

### 7.4.5 交换机架构对比

| 特性 | 共享内存 | 交叉开关 |
|------|---------|---------|
| 端口扩展性 | 受内存带宽限制 | 可扩展到数百端口 |
| 带宽 | 受限于内存带宽 | 每个交叉点独立带宽 |
| 成本 | 较低 | 交叉点成本 O(N²) |
| 调度复杂度 | 无需调度 | 需要调度算法 |
| 应用场景 | 中低端交换机 | 核心交换机、路由器 |

---

## 7.5 VLAN Trunking

### 7.5.1 Trunk 的概念

当多个交换机上的相同 VLAN 需要互通时，使用 **Trunk（干道）** 链路连接交换机。Trunk 链路上的帧带有 VLAN 标签，标识它属于哪个 VLAN。

```
交换机 A                          交换机 B
┌──────────────┐               ┌──────────────┐
│ VLAN 10: 端口1-8             │ VLAN 10: 端口1-8
│ VLAN 20: 端口9-16   Trunk   │ VLAN 20: 端口9-16
│              │───────────────│              │
│ 端口24(Trunk)│               │ 端口24(Trunk)│
└──────────────┘               └──────────────┘

VLAN 10 的帧在 Trunk 上携带 802.1Q 标签 (VID=10)
VLAN 20 的帧在 Trunk 上携带 802.1Q 标签 (VID=20)
```

### 7.5.2 Native VLAN

Trunk 端口有一个 **Native VLAN**（默认 VLAN 1）。属于 Native VLAN 的帧在 Trunk 上**不携带标签**发送。接收端收到无标签帧时，将其归入 Native VLAN。

**安全提醒**：建议将 Native VLAN 设置为一个不使用的 VLAN ID，防止 VLAN 跳转攻击（VLAN Hopping）。

### 7.5.3 VLAN Trunk 配置示例

```bash
# Cisco IOS 交换机配置

# 创建 VLAN
switch(config)# vlan 10
switch(config-vlan)# name Engineering
switch(config)# vlan 20
switch(config-vlan)# name Marketing

# 配置 Access 端口
switch(config)# interface FastEthernet 0/1
switch(config-if)# switchport mode access
switch(config-if)# switchport access vlan 10

# 配置 Trunk 端口
switch(config)# interface GigabitEthernet 0/1
switch(config-if)# switchport mode trunk
switch(config-if)# switchport trunk allowed vlan 10,20
switch(config-if)# switchport trunk native vlan 999

# 查看 VLAN 信息
switch# show vlan brief
switch# show interfaces trunk
```

<div data-component="VLANTrunkDemo"></div>

---

## 7.6 链路聚合（Link Aggregation）

### 7.6.1 链路聚合的概念

链路聚合将多条物理链路捆绑为一条逻辑链路，实现：
- **带宽增加**：多条链路的总带宽
- **冗余备份**：一条链路故障，流量自动切换到其他链路
- **负载均衡**：流量在多条链路上均匀分布

IEEE 802.3ad（后更正为 802.1AX）定义了链路聚合标准。

### 7.6.2 LACP（Link Aggregation Control Protocol）

LACP 是链路聚合的协商协议，通过交换 LACPDU（LACP Data Unit）自动协商聚合组成员。

**LACP 工作流程**：
1. 端口发送 LACPDU，包含系统 ID、端口优先级、端口编号等
2. 对端收到 LACPDU 后，比较系统 ID 和端口信息
3. 双方确认匹配后，将端口加入聚合组
4. 定期交换 LACPDU 维护聚合状态

```python
class LACPPort:
    """LACP 端口模拟"""

    def __init__(self, port_id, system_id, system_priority, port_priority):
        self.port_id = port_id
        self.system_id = system_id
        self.system_priority = system_priority
        self.port_priority = port_priority
        self.state = 'Inactive'
        self.partner_info = None
        self.lacp_timer = 0

    def send_lacpdu(self):
        """发送 LACPDU"""
        return {
            'system_id': self.system_id,
            'system_priority': self.system_priority,
            'port_id': self.port_id,
            'port_priority': self.port_priority,
            'state': self.state,
        }

    def receive_lacpdu(self, lacpdu):
        """收到 LACPDU"""
        self.partner_info = lacpdu
        self.lacp_timer = time.time()

        # 检查是否可以聚合
        if self._can_aggregate(lacpdu):
            self.state = 'Active'
            return True
        return False

    def _can_aggregate(self, lacpdu):
        """判断是否可以聚合"""
        # 系统 ID 不同（不能和自己聚合）
        if lacpdu['system_id'] == self.system_id:
            return False
        # 端口参数匹配
        return True
```

### 7.6.3 负载均衡策略

| 策略 | 基于 | 说明 |
|------|------|------|
| 基于源 MAC | 源 MAC 地址哈希 | 同一源的流量始终走同一链路 |
| 基于目的 MAC | 目的 MAC 地址哈希 | 同一目的地的流量走同一链路 |
| 基于源-目的 MAC | 双方 MAC 哈希 | 更均匀的分布 |
| 基于源-目的 IP | 双方 IP 哈希 | 最精细的负载均衡 |
| 基于 L3+L4 | IP + 端口号哈希 | 适用于三层交换机 |

---

## 7.7 MAC-in-MAC（IEEE 802.1ah）

### 7.6.1 运营商以太网的需求

在运营商网络中，需要在客户 VLAN 之上再加一层标识，用于：
- **扩展 VLAN 空间**：802.1Q 只有 4094 个 VLAN，不够用
- **客户隔离**：不同客户的相同 VLAN 号互不干扰
- **简化运营商配置**：运营商不需要知道客户的 VLAN 详情

### 7.7.2 MAC-in-MAC 帧格式

```
┌─────────────────────────────────────────────────────────────────┐
│ 运营商以太网头（Service Tag）                                    │
│ ┌──────────┬──────────┬─────────────────┬──────────────────────┐│
│ │运营商目的│运营商源  │EtherType 0x88E7│ Service Instance Tag││
│ │MAC (6B)  │MAC (6B)  │(2B)            │ (SID + PRI + DEI)   ││
│ └──────────┴──────────┴─────────────────┴──────────────────────┘│
│                                                                 │
│ 客户以太网帧                                                    │
│ ┌──────────┬──────────┬──────────┬─────────┬─────┐              │
│ │客户目的  │客户源    │ 802.1Q   │ 客户    │ FCS │              │
│ │MAC (6B)  │MAC (6B)  │ 标签     │ 数据    │     │              │
│ └──────────┴──────────┴──────────┴─────────┴─────┘              │
└─────────────────────────────────────────────────────────────────┘
```

**关键优势**：运营商只需维护自己的 MAC 地址（通常几十个），而客户有成千上万的 MAC 地址。查找运营商 MAC 比查找客户 MAC 快得多。

<div data-component="MACInMACDemo"></div>

---

## 7.8 核心组件深入分析

### 7.8.1 网桥转发引擎与 MAC 学习表的硬件实现

**CAM（Content Addressable Memory）实现 MAC 表**：

CAM 是一种特殊的存储器，可以并行比较所有条目，实现 O(1) 查找。

```
CAM 查找过程:

输入: 目的 MAC = AA:BB:CC:DD:EE:FF

┌─────────────────────────────────────────────┐
│               CAM 阵列                       │
│  ┌─────┬────────────────┬──────┬──────────┐ │
│  │ 条目│ MAC 地址       │ 端口 │ 有效位   │ │
│  ├─────┼────────────────┼──────┼──────────┤ │
│  │  0  │ AA:BB:CC:11:22:33│  1  │    1    │ │
│  │  1  │ AA:BB:CC:DD:EE:FF│  3  │    1    │ │ ← 匹配！
│  │  2  │ FF:FF:FF:FF:FF:FF│  -1 │    1    │ │ (广播)
│  │  3  │ 00:00:00:00:00:00│  0  │    0    │ │ (无效)
│  └─────┴────────────────┴──────┴──────────┘ │
│                                             │
│  匹配信号: ▶ 条目1 匹配 → 输出端口 3       │
└─────────────────────────────────────────────┘
```

**硬件实现的关键技术**：

```c
/* CAM 条目结构 */
struct cam_entry {
    uint8_t     mac_addr[6];    /* 存储的 MAC 地址 */
    uint8_t     port;           /* 关联的端口号 */
    uint8_t     valid;          /* 有效位 */
    uint16_t    vlan_id;        /* VLAN ID */
    uint32_t    timestamp;      /* 最后使用时间 */
};

/* CAM 查找（硬件并行比较，这里是软件模拟） */
int cam_lookup(struct cam_entry *cam, int size, uint8_t *mac) {
    for (int i = 0; i < size; i++) {
        if (cam[i].valid && memcmp(cam[i].mac_addr, mac, 6) == 0) {
            return cam[i].port;
        }
    }
    return -1;  /* 未找到 */
}
```

### 7.8.2 STP 协议状态机的转换逻辑

STP 端口状态机的完整转换逻辑：

```c
/* STP 端口状态机 */
enum stp_port_state {
    STP_DISABLED,
    STP_BLOCKING,
    STP_LISTENING,
    STP_LEARNING,
    STP_FORWARDING
};

enum stp_port_role {
    ROLE_ROOT,
    ROLE_DESIGNATED,
    ROLE_ALTERNATE,
    ROLE_DISABLED
};

struct stp_port {
    enum stp_port_state state;
    enum stp_port_role  role;
    uint32_t            path_cost;
    uint16_t            port_id;
    uint8_t             mac_addr[6];
    struct bpdu         last_bpdu;
    timer_t             message_age_timer;
    timer_t             forward_delay_timer;
    timer_t             max_age_timer;
    timer_t             hello_timer;
};

/* 状态转换函数 */
void stp_port_state_machine(struct stp_port *port, struct stp_bridge *bridge) {
    switch (port->state) {
    case STP_BLOCKING:
        /* Blocking → Listening: 当端口被选为根端口或指定端口 */
        if (port->role == ROLE_ROOT || port->role == ROLE_DESIGNATED) {
            port->state = STP_LISTENING;
            start_timer(&port->forward_delay_timer, FORWARD_DELAY);
        }
        /* 超时检测 */
        if (timer_expired(&port->max_age_timer)) {
            /* 拓扑变化，重新计算 */
            bridge_recalculate(bridge);
        }
        break;

    case STP_LISTENING:
        /* Listening → Learning: Forward Delay 超时 */
        if (timer_expired(&port->forward_delay_timer)) {
            port->state = STP_LEARNING;
            start_timer(&port->forward_delay_timer, FORWARD_DELAY);
        }
        /* 收到更优 BPDU 可能回到 Blocking */
        if (received_inferior_bpdu(port)) {
            /* 无操作，但更新 BPDU */
        }
        break;

    case STP_LEARNING:
        /* Learning → Forwarding: Forward Delay 超时 */
        if (timer_expired(&port->forward_delay_timer)) {
            port->state = STP_FORWARDING;
        }
        break;

    case STP_FORWARDING:
        /* 正常工作状态 */
        if (port->role == ROLE_ROOT || port->role == ROLE_DESIGNATED) {
            /* 发送 BPDU */
            if (timer_expired(&port->hello_timer)) {
                send_bpdu(port, bridge);
                start_timer(&port->hello_timer, HELLO_TIME);
            }
        }
        /* 收到更优 BPDU → 回到 Blocking */
        if (received_superior_bpdu(port)) {
            port->state = STP_BLOCKING;
            port->role = ROLE_ALTERNATE;
        }
        break;
    }
}
```

### 7.8.3 交叉开关矩阵的调度算法

**iSLIP 调度算法的详细实现**：

```c
/* iSLIP 调度器硬件实现（伪代码） */

#define MAX_PORTS 64

struct islip_scheduler {
    int grant_ptr[MAX_PORTS];   /* 每个输出端口的授权轮询指针 */
    int accept_ptr[MAX_PORTS];  /* 每个输入端口的接受轮询指针 */
};

void islip_schedule(struct islip_scheduler *sched,
                    int requests[MAX_PORTS][MAX_PORTS],
                    int match[MAX_PORTS],
                    int num_ports,
                    int iterations) {
    memset(match, -1, sizeof(int) * num_ports);
    int output_matched[MAX_PORTS] = {0};

    for (int iter = 0; iter < iterations; iter++) {
        /* 阶段 1: 请求 - 输入端口向所有需要的输出发送请求 */
        /* requests[i][j] = 1 表示输入 i 请求输出 j */

        /* 阶段 2: 授权 - 输出端口选择一个请求 */
        int grants[MAX_PORTS];
        memset(grants, -1, sizeof(int) * num_ports);

        for (int j = 0; j < num_ports; j++) {
            if (output_matched[j]) continue;

            for (int offset = 0; offset < num_ports; offset++) {
                int i = (sched->grant_ptr[j] + offset) % num_ports;
                if (requests[i][j] && match[i] == -1) {
                    grants[j] = i;
                    break;
                }
            }
        }

        /* 阶段 3: 接受 - 输入端口选择一个授权 */
        for (int i = 0; i < num_ports; i++) {
            if (match[i] != -1) continue;

            for (int offset = 0; offset < num_ports; offset++) {
                int j = (sched->accept_ptr[i] + offset) % num_ports;
                if (grants[j] == i) {
                    match[i] = j;
                    output_matched[j] = 1;

                    /* 更新指针（仅首轮成功匹配时更新） */
                    if (iter == 0) {
                        sched->grant_ptr[j] = (j + 1) % num_ports;
                        sched->accept_ptr[i] = (i + 1) % num_ports;
                    }
                    break;
                }
            }
        }
    }
}
```

**iSLIP 的性能特性**：
- 单轮迭代达到 100% 吞吐量（均匀流量下）
- 多轮迭代改善非均匀流量下的性能
- 公平性：轮询指针确保每个端口都有机会
- 时间复杂度：每轮迭代 O(N²)，通常 1-3 轮足够

### 7.8.4 VLAN 标记处理器的数据通路

VLAN 标记处理器是交换机中处理 VLAN 标签的关键模块：

```
入方向数据通路:

帧到达 → 解析以太网头 → 是否有 802.1Q 标签？
                           │
                     ┌─────┴─────┐
                     │           │
                    是           否
                     │           │
                     ▼           ▼
              提取 VID     使用端口 PVID
              提取 PCP     PCP = 端口默认优先级
                     │           │
                     └─────┬─────┘
                           │
                           ▼
                    VLAN 表查找
                    (VID 是否允许?)
                           │
                     ┌─────┴─────┐
                     │           │
                   允许         不允许
                     │           │
                     ▼           ▼
               转发引擎       丢弃帧
               (MAC查找)

出方向数据通路:

转发引擎决定输出端口 → 输出端口类型?
                           │
                     ┌─────┼─────┐
                     │     │     │
                   Access  Trunk  Hybrid
                     │     │     │
                     ▼     ▼     ▼
                   去标签  保留标签  按规则
                   (strip) (keep)  (tag/untag)
                     │     │     │
                     ▼     ▼     ▼
                   发送帧到物理端口
```

```c
/* VLAN 标签处理器 */
struct vlan_processor {
    uint16_t port_pvid[MAX_PORTS];        /* 端口默认 VLAN */
    uint8_t  port_type[MAX_PORTS];        /* ACCESS / TRUNK / HYBRID */
    uint32_t allowed_vlans[MAX_PORTS];    /* 位图，每个位对应一个 VID */
    uint8_t  default_priority[MAX_PORTS]; /* 端口默认优先级 */
};

int vlan_ingress_process(struct vlan_processor *vp,
                          int in_port,
                          struct ethernet_frame *frame,
                          uint16_t *vid_out,
                          uint8_t *pcp_out) {
    /* 检查是否有 VLAN 标签 */
    uint16_t tpid = ntohs(*(uint16_t*)(frame->data + 12));

    if (tpid == 0x8100) {
        /* 有 802.1Q 标签 */
        uint16_t tci = ntohs(*(uint16_t*)(frame->data + 14));
        *vid_out = tci & 0x0FFF;
        *pcp_out = (tci >> 13) & 0x07;
    } else {
        /* 无标签，使用端口 PVID */
        *vid_out = vp->port_pvid[in_port];
        *pcp_out = vp->default_priority[in_port];
    }

    /* 检查 VLAN 是否允许 */
    if (!(vp->allowed_vlans[in_port] & (1 << *vid_out))) {
        return -1;  /* 丢弃 */
    }

    return 0;  /* 允许 */
}
```

<div data-component="VLANProcessorDemo"></div>

---

## 7.9 交换机性能指标

### 7.9.1 关键性能参数

| 指标 | 说明 | 典型值 |
|------|------|--------|
| 背板带宽 | 交换矩阵的总带宽 | 接入: 100Gbps, 核心: 数 Tbps |
| 包转发率 | 每秒转发的帧数（pps） | 千兆线速: 1,488,095 pps |
| MAC 表容量 | 可学习的 MAC 地址数 | 8K-256K |
| VLAN 数量 | 支持的 VLAN 数 | 最大 4094 |
| 延迟 | 帧从入端口到出端口的时间 | 存储转发: 10-100μs |
| 缓冲区大小 | 入/出端口的帧缓冲区 | 数 MB 到数十 MB |

### 7.9.2 线速转发计算

以太网帧最小 64 字节 + 8 字节前导码 + 12 字节帧间隙 = 84 字节

$$\text{线速 pps} = \frac{\text{链路速率}}{84 \times 8}$$

| 链路速率 | 最小帧线速 |
|---------|-----------|
| 10 Mbps | 14,880 pps |
| 100 Mbps | 148,809 pps |
| 1 Gbps | 1,488,095 pps |
| 10 Gbps | 14,880,952 pps |
| 100 Gbps | 148,809,523 pps |

```python
def calculate_wire_speed(link_speed_bps, frame_size_bytes=64):
    """计算线速转发速率"""
    # 以太网帧额外开销: 8 字节前导码 + 12 字节帧间隙
    total_size = frame_size_bytes + 8 + 12
    wire_speed = link_speed_bps / (total_size * 8)
    return wire_speed

# 示例
for speed in [10e6, 100e6, 1e9, 10e9, 100e9]:
    pps = calculate_wire_speed(speed)
    print(f"{speed/1e6:6.0f} Mbps → {pps:>15,.0f} pps")
```

---

## 7.10 本章总结

| 概念 | 核心要点 |
|------|---------|
| 透明网桥 | 自学习 MAC 表，转发/过滤/学习三动作 |
| STP | 根桥选举(最小 BID)、根端口(最小路径开销)、指定端口、阻塞端口 |
| STP 状态 | Blocking → Listening → Learning → Forwarding (30-50秒) |
| RSTP | 3 种状态、Proposal/Agreement 快速收敛 |
| 共享内存 | 端口共享内存池，受带宽限制 |
| 交叉开关 | 非阻塞交换，iSLIP 调度算法 |
| VLAN Trunk | 802.1Q 标签，Access/Trunk 端口 |
| 链路聚合 | LACP 协商，多链路捆绑 |
| MAC-in-MAC | 运营商以太网，双层 MAC 地址 |

<div data-component="ChapterSummaryQuiz"></div>

---

## 练习题

**1. 分析题**：某网络有 4 个网桥 A(ID=1000), B(ID=2000), C(ID=3000), D(ID=4000)，链路开销均为 1。画出拓扑图并确定 STP 收敛后的根桥、根端口、指定端口和阻塞端口。

**2. 计算题**：一个 48 口千兆交换机使用共享内存架构，每个端口需要同时收发。计算所需的内存带宽。

**3. 实现题**：实现一个简单的 STP 模拟器，支持根桥选举、端口角色计算和状态转换。

**4. 比较题**：对比共享内存和交叉开关两种交换架构的优缺点，说明各自适用的场景。

**5. 设计题**：设计一个支持 4094 个 VLAN 的交换机 MAC 表数据结构，要求支持高效的 VLAN+MAC 联合查找。

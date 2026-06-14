# Chapter 6: 数据链路层 — MAC 与以太网

> **学习目标**：
> - 理解信道分配问题的本质与分类，掌握静态与动态分配策略的适用场景
> - 掌握 ALOHA 协议（纯 ALOHA、时隙 ALOHA）的工作原理、吞吐量推导与性能极限
> - 理解 CSMA 协议族（1-坚持、非坚持、p-坚持）的侦听与退避策略差异
> - 深入掌握 CSMA/CD 的碰撞检测机制与二进制指数退避算法
> - 掌握以太网帧格式（DIX Ethernet II / IEEE 802.3）的每个字段含义与区别
> - 理解 MAC 地址的结构、分类与 ARP 协议的工作流程（含缓存、超时、代理 ARP）
> - 了解 VLAN（802.1Q 标签）和 PoE 的基本原理与应用场景

---

## 6.1 信道分配问题

### 6.1.1 问题的本质

在多台计算机共享同一条通信介质（如总线型以太网、无线信道）时，核心问题是：**如何在多个竞争用户之间公平、高效地分配信道的使用权？** 这就是信道分配问题（Channel Allocation Problem）。

想象一个会议室场景：多人同时想发言，如果没有规则，所有人都在说话，谁也听不清——这就是**碰撞（Collision）**。我们需要某种机制来协调，使得每次只有一个人说话，而且大家都能轮流发言。

信道分配策略可分为两大类：

| 特性 | 静态分配 | 动态分配 |
|------|---------|---------|
| 方法 | FDM、TDM | ALOHA、CSMA、CSMA/CD |
| 适用场景 | 用户数固定、流量稳定 | 用户数变化、流量突发 |
| 信道利用率 | 负载低时浪费 | 负载低时高效 |
| 碰撞处理 | 无碰撞 | 需处理碰撞 |
| 延迟特性 | 确定性延迟 | 统计性延迟 |

### 6.1.2 静态信道分配

**频分多路复用（FDM）** 将总带宽分成若干子频段，每个用户独占一个频段。**时分多路复用（TDM）** 将时间划分为固定长度的时隙，每个用户轮流使用。

对于 $N$ 个用户，每个用户分得带宽 $W/N$，最大容量为：

$$C = \frac{W}{N} \log_2\left(1 + \frac{S}{N_0 \cdot W/N}\right)$$

当 $N$ 增大时，每个用户可用带宽急剧下降。更严重的是，当大多数用户在大部分时间没有数据发送时，分配给它们的信道资源完全浪费。

### 6.1.3 动态信道分配的三个假设

大多数动态分配协议基于以下假设（Abramson, 1970）：

1. **独立流量模型**：各站点独立产生帧，到达过程服从参数为 $\lambda$ 的泊松分布
2. **单信道假设**：所有通信都通过同一个信道完成，没有专用控制信道
3. **冲突假设**：两帧同时传输则发生碰撞，碰撞帧必须重传

<div data-component="ChannelAllocationComparison"></div>

---

## 6.2 ALOHA 协议

### 6.2.1 纯 ALOHA（Pure ALOHA）

纯 ALOHA 由 Norman Abramson 于 1970 年在夏威夷大学提出，是最简单的随机接入协议。

**工作机制**：
1. 任何站点有数据就立即发送，无需侦听信道
2. 发送后等待确认（ACK）
3. 若未收到 ACK（认为发生了碰撞），等待一个随机时间后重传
4. 随机等待时间的选择避免反复碰撞

**脆弱期（Vulnerable Period）**：一个帧的传输时间为 $t$。如果在该帧发送前一个 $t$ 时间内或发送期间有其他帧开始发送，就会发生碰撞。因此脆弱期为 $2t$。

**吞吐量推导**：

设帧的到达服从泊松分布，每帧时间 $t$ 内产生的帧数（包括新帧和重传帧）为 $G$。在脆弱期 $2t$ 内恰好没有其他帧发送的概率为：

$$P_0 = e^{-2G}$$

吞吐量 $S = G \cdot P_0 = G \cdot e^{-2G}$

对 $G$ 求导令其为零：

$$\frac{dS}{dG} = e^{-2G}(1 - 2G) = 0$$

解得 $G = 0.5$ 时，$S_{max} = 0.5 \times e^{-1} \approx 0.184$

**纯 ALOHA 的最大吞吐量仅为 18.4%**，这意味着信道利用率很低。

```python
import numpy as np
import matplotlib.pyplot as plt

G = np.linspace(0, 3, 1000)
S_pure_aloha = G * np.exp(-2 * G)
S_slotted_aloha = G * np.exp(- G)

plt.figure(figsize=(10, 6))
plt.plot(G, S_pure_aloha, label='Pure ALOHA', linewidth=2)
plt.plot(G, S_slotted_aloha, label='Slotted ALOHA', linewidth=2)
plt.xlabel('Offered Load G')
plt.ylabel('Throughput S')
plt.title('ALOHA Throughput vs Offered Load')
plt.legend()
plt.grid(True)
plt.axhline(y=0.184, color='r', linestyle='--', alpha=0.5, label='Pure ALOHA max=0.184')
plt.axhline(y=0.368, color='g', linestyle='--', alpha=0.5, label='Slotted ALOHA max=0.368')
plt.show()
```

### 6.2.2 时隙 ALOHA（Slotted ALOHA）

Roberts（1972）对纯 ALOHA 的改进：将时间划分为等长时隙，每个时隙长度等于一帧的传输时间。站点只能在时隙开始时发送。

**改进效果**：脆弱期从 $2t$ 缩短为 $t$，吞吐量为：

$$S = G \cdot e^{-G}$$

最大吞吐量在 $G=1$ 时取得：$S_{max} = e^{-1} \approx 0.368$，是纯 ALOHA 的两倍。

<div data-component="ALOHASimulator"></div>

### 6.2.3 ALOHA 的局限

| 问题 | 说明 |
|------|------|
| 低吞吐量 | 最大 18.4%/36.8%，无法满足高负载需求 |
| 不公平性 | 运气好的站点可能长期占用信道 |
| 无碰撞检测 | 发送后才知道是否成功，浪费带宽 |
| 适用范围 | 卫星通信、无线网络等传播延迟大的场景 |

---

## 6.3 CSMA 协议

### 6.3.1 载波侦听多路访问（Carrier Sense Multiple Access）

CSMA 的核心思想是**"先听后说"（Listen Before Talk）**。在发送前先侦听信道是否空闲，减少碰撞概率。

然而，即使侦听到空闲，由于**传播延迟**的存在，仍可能发生碰撞。设信道传播延迟为 $\tau$，帧传输时间为 $t$。当一个站点刚开始发送时，距离最远的站点在 $\tau$ 时间后才能感知到信号。如果在这段时间内开始发送，就会碰撞。

定义 $a = \tau / t$（传播延迟与传输时间之比），$a$ 越小，信道利用率越高。

### 6.3.2 1-坚持 CSMA

**策略**：侦听到信道忙则持续侦听，一旦空闲立即以概率 1 发送。

```
站点有数据要发送：
    while 信道忙:
        持续侦听
    信道空闲 → 立即发送
    if 碰撞:
        等待随机时间后重试
```

**优点**：最大限度利用信道（一空闲就发）。**缺点**：多个站点同时等待时，信道一空闲它们都会立即发送，碰撞概率高。

### 6.3.3 非坚持 CSMA（Non-persistent CSMA）

**策略**：侦听到信道忙则不持续侦听，而是等待一个随机时间后再侦听。

```
站点有数据要发送：
    if 信道空闲:
        立即发送
    else:
        等待一个随机时间
        重新侦听
```

**优点**：减少了碰撞概率（随机退避分散了重传时间）。**缺点**：信道可能在等待期间空闲，导致利用率降低。

### 6.3.4 p-坚持 CSMA

**策略**（用于时隙信道）：侦听到信道空闲时，以概率 $p$ 发送，以概率 $1-p$ 延迟到下一个时隙。如果下一时隙仍空闲，再次以概率 $p$ 发送。

```
站点有数据要发送（时隙信道）：
    if 信道空闲:
        以概率 p 发送
        以概率 1-p 延迟到下一个时隙
            if 下一时隙仍空闲:
                重复此过程
    else:
        等待直到下一时隙空闲
        再以概率 p 发送
```

**$p$ 值的选择**：$p$ 过大，碰撞增加；$p$ 过小，延迟增加。最优 $p$ 值与活跃站点数 $N$ 的关系为 $p = 1/N$。

<div data-component="CSMAComparison"></div>

### 6.3.5 三种 CSMA 策略对比

| 策略 | 信道忙时 | 信道闲时 | 碰撞概率 | 信道利用率 |
|------|---------|---------|---------|-----------|
| 1-坚持 | 持续侦听 | 立即发送(p=1) | 高 | 中 |
| 非坚持 | 随机等待 | 立即发送 | 低 | 中低 |
| p-坚持 | 等下一时隙 | 概率p发送 | 适中 | 较高 |

---

## 6.4 CSMA/CD 协议

### 6.4.1 带碰撞检测的 CSMA

CSMA/CD（CSMA with Collision Detection）是经典以太网（IEEE 802.3）的核心协议。它在 CSMA 的基础上增加了**碰撞检测**能力：边发送边监听，如果检测到碰撞立即停止发送。

**关键改进**：
- **"边说边听"**：发送站点在发送的同时监听信道，比较发送的信号与收到的信号
- **碰撞后立即停止**：检测到碰撞后发送一个**干扰信号（Jam Signal）**（32-48 位），确保所有站点都感知到碰撞
- **节省带宽**：碰撞帧只发送了一部分就被截断，减少了浪费

### 6.4.2 最小帧长约束

CSMA/CD 正常工作的前提是：**发送站点必须在发送完最后一比特之前检测到碰撞**。

设信道的最大传播延迟为 $\tau$（往返时间为 $2\tau$），帧的传输时间为 $t_{frame}$。

最坏情况：站点 A 在发送帧的最后一比特前 $\epsilon$ 时间，最远端的碰撞信号尚未返回。为确保检测到碰撞：

$$t_{frame} \geq 2\tau$$

对于 10 Mbps 以太网，最大段长 2500m，传播速度约 $2 \times 10^8$ m/s：

$$\tau = \frac{2500}{2 \times 10^8} = 12.5 \mu s$$

$$t_{frame} \geq 25 \mu s \Rightarrow \text{最小帧长} = 10 \text{Mbps} \times 25 \mu s = 250 \text{bits} = 31.25 \text{Bytes}$$

实际以太网取最小帧长为 **64 字节（512 比特）**，留有余量。

### 6.4.3 二进制指数退避算法（Binary Exponential Backoff）

碰撞后，站点需要等待一个随机时间再重传。以太网使用**二进制指数退避算法**：

**算法步骤**：
1. 碰撞后，设 $n$ = 已碰撞次数（$n \leq 10$）
2. 从 $\{0, 1, 2, ..., 2^{\min(n,10)} - 1\}$ 中均匀随机选择一个整数 $k$
3. 等待 $k \times 51.2 \mu s$（对于 10 Mbps 以太网，51.2 μs = 512 比特时间 = 时隙时间）
4. 若 $n > 16$，放弃发送，报告错误

**特点**：
- **自适应性**：碰撞次数少时退避时间短（快速重试），碰撞次数多时退避时间长（减轻拥塞）
- **指数增长**：退避窗口 $W = 2^n \times 51.2 \mu s$，其中 $n = \min(\text{碰撞次数}, 10)$
- **公平性**：每增加一次碰撞，可选等待时间翻倍，分散重传

```c
/* 二进制指数退避算法实现 */
#define SLOT_TIME      51.2e-6   /* 10Mbps 以太网时隙时间 */
#define MAX_COLLISIONS 16
#define MAX_BACKOFF_EXP 10

int calculate_backoff(int collision_count) {
    if (collision_count > MAX_COLLISIONS) {
        return -1;  /* 放弃发送 */
    }

    int exp = (collision_count < MAX_BACKOFF_EXP)
              ? collision_count
              : MAX_BACKOFF_EXP;

    int max_slots = (1 << exp) - 1;  /* 2^exp - 1 */
    int k = rand() % (max_slots + 1); /* 随机选择 0..max_slots */

    return (int)(k * SLOT_TIME * 1e6); /* 返回微秒数 */
}
```

### 6.4.4 CSMA/CD 的完整工作流程

```
CSMA/CD 发送流程:
┌─────────────────┐
│  有帧要发送     │
└────────┬────────┘
         ▼
┌─────────────────┐
│  侦听信道       │◄──────────┐
└────────┬────────┘           │
         ▼                    │
    信道忙？──是──► 继续侦听 ─┘
         │否
         ▼
┌─────────────────┐
│  开始发送       │
│  同时监听信道   │
└────────┬────────┘
         ▼
    检测到碰撞？──否──► 发送完成，成功
         │是
         ▼
┌─────────────────┐
│  停止发送       │
│  发送干扰信号   │
└────────┬────────┘
         ▼
┌─────────────────┐
│  碰撞次数 +1    │
│  >16? → 放弃    │
└────────┬────────┘
         ▼
┌─────────────────┐
│  二进制指数退避  │
│  等待随机时隙数  │
└────────┬────────┘
         ▼
       重试发送
```

<div data-component="CSMA_CDSimulator"></div>

### 6.4.5 CSMA/CD 的性能分析

设 $N$ 个站点竞争信道，每个站点在任意时隙发送的概率为 $p$。

一个时隙成功传输的概率为：

$$A = N \cdot p \cdot (1-p)^{N-1}$$

当 $p = 1/N$ 时，$A$ 取最大值：

$$A_{max} = \left(1 - \frac{1}{N}\right)^{N-1} \xrightarrow{N \to \infty} \frac{1}{e} \approx 0.368$$

信道效率（考虑 $a = \tau/t$）：

$$\text{Efficiency} = \frac{1}{1 + 2a \cdot e \cdot \text{(退避开销)}}$$

当 $a$ 很小时（局域网典型情况），CSMA/CD 效率接近 1。

---

## 6.5 以太网帧格式

### 6.5.1 DIX Ethernet II 帧格式

DIX（DEC-Intel-Xerox）Ethernet II 是实际使用最广泛的以太网帧格式：

```
┌──────────┬──────────┬──────────┬─────────┬─────┬─────┐
│ 目的MAC  │ 源MAC    │ 类型     │ 数据    │ 填充 │ FCS │
│ 6字节    │ 6字节    │ 2字节    │46-1500B │ 可选 │ 4B  │
└──────────┴──────────┴──────────┴─────────┴─────┴─────┘
```

| 字段 | 长度 | 说明 |
|------|------|------|
| 目的 MAC 地址 | 6 字节 | 接收方的硬件地址 |
| 源 MAC 地址 | 6 字节 | 发送方的硬件地址 |
| 类型（Type） | 2 字节 | 上层协议标识，如 0x0800=IPv4, 0x0806=ARP, 0x86DD=IPv6 |
| 数据（Payload） | 46-1500 字节 | 上层数据，最小 46 字节（保证帧长 ≥ 64 字节） |
| FCS | 4 字节 | CRC-32 校验和，覆盖目的地址到数据字段 |

**帧长度**：最小 64 字节（14 + 46 + 4），最大 1518 字节（14 + 1500 + 4）。

### 6.5.2 IEEE 802.3 帧格式

IEEE 802.3 将 2 字节的"类型"字段改为"长度"字段，并在数据字段前增加 LLC/SNAP 头部：

```
┌──────────┬──────────┬────────┬──────────┬─────────┬─────┬─────┐
│ 目的MAC  │ 源MAC    │ 长度   │ LLC/SNAP │ 数据    │ 填充 │ FCS │
│ 6字节    │ 6字节    │ 2字节  │ 3-8字节  │ 可变   │ 可选 │ 4B  │
└──────────┴──────────┴────────┴──────────┴─────────┴─────┴─────┘
```

**LLC（Logical Link Control）头**：
- DSAP（1 字节）：目的服务访问点
- SSAP（1 字节）：源服务访问点
- Control（1 字节）：控制字段

**SNAP（Sub-Network Access Protocol）扩展**：
- OUI（3 字节）：组织唯一标识符
- Type（2 字节）：协议类型（兼容 Ethernet II）

### 6.5.3 Ethernet II 与 802.3 的区分

由于 2 字节字段既可以表示类型也可以表示长度，需要区分：

- 若该字段值 **≥ 0x0600（1536）**：解释为**类型**→ Ethernet II 帧
- 若该字段值 **≤ 0x05DC（1500）**：解释为**长度**→ IEEE 802.3 帧

这是一种巧妙的向后兼容设计。

```python
def parse_ethernet_frame(raw_bytes):
    """解析以太网帧"""
    # 目的 MAC 地址 (6 字节)
    dst_mac = ':'.join(f'{b:02x}' for b in raw_bytes[0:6])
    # 源 MAC 地址 (6 字节)
    src_mac = ':'.join(f'{b:02x}' for b in raw_bytes[6:12])
    # 类型/长度字段 (2 字节)
    type_or_len = int.from_bytes(raw_bytes[12:14], byteorder='big')

    if type_or_len >= 0x0600:
        # Ethernet II 帧
        frame_type = "Ethernet II"
        protocol = {0x0800: "IPv4", 0x0806: "ARP", 0x86DD: "IPv6"}.get(type_or_len, "Unknown")
        payload = raw_bytes[14:-4]
    else:
        # IEEE 802.3 帧
        frame_type = "IEEE 802.3"
        length = type_or_len
        payload = raw_bytes[14:14+length]

    # FCS 校验 (最后 4 字节)
    fcs = int.from_bytes(raw_bytes[-4:], byteorder='little')

    return {
        "frame_type": frame_type,
        "dst_mac": dst_mac,
        "src_mac": src_mac,
        "protocol": protocol if frame_type == "Ethernet II" else "LLC",
        "payload_len": len(payload),
        "fcs": fcs,
    }
```

<div data-component="EthernetFrameParser"></div>

---

## 6.6 MAC 地址

### 6.6.1 MAC 地址的结构

MAC 地址（Media Access Control Address）是 48 位（6 字节）的硬件地址，通常表示为冒号分隔的十六进制，如 `00:1A:2B:3C:4D:5E`。

```
  位:  8      8      8      8      8      8
     ┌────────────────────────────────────────┐
     │  OUI (前 3 字节)  │  设备标识 (后 3 字节) │
     └────────────────────────────────────────┘
      I/G  U/L
      位   位
```

**I/G 位（Individual/Group）**：第 1 字节的最低位
- 0 = 单播地址（发给特定设备）
- 1 = 组播地址（发给一组设备），全 1 即广播 `FF:FF:FF:FF:FF:FF`

**U/L 位（Universal/Local）**：第 1 字节的次低位
- 0 = 全球唯一（由 IEEE 分配给厂商）
- 1 = 本地管理（管理员手动设置）

### 6.6.2 MAC 地址分类

| 类型 | 地址范围 | 说明 |
|------|---------|------|
| 单播 | 第 1 字节最低位 = 0 | 唯一标识一个网络接口 |
| 组播 | 第 1 字节最低位 = 1 | 发给一组设备 |
| 广播 | FF:FF:FF:FF:FF:FF | 发给所有设备 |

**特殊 MAC 地址**：
- `FF:FF:FF:FF:FF:FF`：广播地址
- `01:00:5E:xx:xx:xx`：IPv4 组播 MAC 地址范围
- `33:33:xx:xx:xx:xx`：IPv6 组播 MAC 地址范围
- `01:80:C2:00:00:00`到 `01:80:C2:00:00:0F`：保留给协议使用（如 STP）

### 6.6.3 MAC 地址 vs IP 地址

| 特性 | MAC 地址 | IP 地址 |
|------|---------|---------|
| 层次 | 数据链路层（L2） | 网络层（L3） |
| 长度 | 48 位 | 32 位（IPv4）/ 128 位（IPv6） |
| 分配方式 | 厂商固化 | 网络管理员/DHCP 分配 |
| 可变性 | 通常不变（可软件修改） | 随网络变化 |
| 寻址范围 | 本地链路 | 全局可达 |
| 作用 | 同一物理网段内的设备识别 | 跨网络的端到端路由 |

---

## 6.7 ARP 协议

### 6.7.1 ARP 的基本原理

ARP（Address Resolution Protocol）解决的问题是：**已知目标设备的 IP 地址，如何获取其 MAC 地址？**

工作流程：
1. 主机 A 要发送数据给 IP 为 `192.168.1.100` 的主机 B，但不知道 B 的 MAC 地址
2. A 发送 **ARP 请求**（广播帧），内容："`192.168.1.100` 的 MAC 地址是什么？我是 `192.168.1.1`，MAC 是 `AA:BB:CC:DD:EE:01`"
3. 网段内所有主机都收到该广播，只有 `192.168.1.100` 回应
4. B 发送 **ARP 应答**（单播帧），内容："`192.168.1.100` 的 MAC 是 `AA:BB:CC:DD:EE:02`"
5. A 收到应答，将映射关系存入 ARP 缓存

### 6.7.2 ARP 帧格式

```
┌────────┬────────┬──────┬──────┬────────┬──────────┬──────────┬──────────┐
│硬件类型│协议类型│硬件地│协议地│操作码  │发送方MAC │发送方IP  │目标MAC   │
│ 2字节  │ 2字节  │址长度│址长度│ 2字节  │ 6字节    │ 4字节    │ 6字节    │
│        │        │1字节 │1字节 │        │          │          │          │
├────────┴────────┴──────┴──────┼────────┼──────────┼──────────┼──────────┤
│                               │        │          │          │目标IP    │
│                               │        │          │          │ 4字节    │
└───────────────────────────────┴────────┴──────────┴──────────┴──────────┘
```

- **硬件类型**：1 = 以太网
- **协议类型**：0x0800 = IPv4
- **操作码**：1 = ARP 请求，2 = ARP 应答

### 6.7.3 ARP 缓存

为避免每次通信都发 ARP 广播，主机维护一个 **ARP 缓存表**，存储最近的 IP-MAC 映射。

```c
/* ARP 缓存表的数据结构 */
#define ARP_CACHE_SIZE  256
#define ARP_TIMEOUT     300  /* 秒 */

struct arp_entry {
    uint32_t    ip_addr;       /* IP 地址 */
    uint8_t     mac_addr[6];   /* MAC 地址 */
    time_t      timestamp;     /* 创建/更新时间 */
    uint8_t     state;         /* 状态: RESOLVED, PENDING, STALE */
    uint8_t     retry_count;   /* 重试计数 */
};

struct arp_cache {
    struct arp_entry entries[ARP_CACHE_SIZE];
    int count;
    pthread_mutex_t lock;      /* 并发保护 */
};

/* ARP 缓存查找 */
struct arp_entry* arp_cache_lookup(struct arp_cache *cache, uint32_t ip) {
    time_t now = time(NULL);

    for (int i = 0; i < cache->count; i++) {
        if (cache->entries[i].ip_addr == ip) {
            if (now - cache->entries[i].timestamp < ARP_TIMEOUT &&
                cache->entries[i].state == RESOLVED) {
                return &cache->entries[i];  /* 缓存命中 */
            } else {
                cache->entries[i].state = STALE;  /* 过期 */
                return NULL;
            }
        }
    }
    return NULL;  /* 缓存未命中 */
}
```

### 6.7.4 ARP 缓存超时机制

| 参数 | 典型值 | 说明 |
|------|--------|------|
| 成功解析缓存时间 | 15-20 分钟 | Linux 默认 60 秒（gc_timeout） |
| 未解析条目超时 | 3 秒 | PENDING 状态的等待时间 |
| 重试次数 | 3 次 | 超过则放弃 |

### 6.7.5 代理 ARP（Proxy ARP）

当路由器代替一个子网回答 ARP 请求时，称为代理 ARP。

**场景**：主机 A（`192.168.1.10/24`）要访问主机 B（`192.168.2.20/24`），A 不知道 B 不在同一子网。

1. A 广播 ARP 请求："`192.168.2.20` 的 MAC 是什么？"
2. 路由器配置了代理 ARP，用自己的 MAC 地址回应
3. A 以为路由器的 MAC 就是 B 的 MAC，将数据发给路由器
4. 路由器转发数据给真正的 B

### 6.7.6 免费 ARP（Gratuitous ARP）

免费 ARP 是一种特殊的 ARP 应答，用于：
1. **IP 冲突检测**：设备上线时主动广播自己的 IP-MAC 映射，如果收到回应说明有冲突
2. **更新其他设备的 ARP 缓存**：设备更换网卡后广播新的 MAC 地址

<div data-component="ARPFlowDiagram"></div>

---

## 6.8 VLAN — 虚拟局域网

### 6.8.1 VLAN 的概念

VLAN（Virtual Local Area Network）将一个物理交换机划分为多个**逻辑广播域**。不同 VLAN 之间的通信需要经过路由器或三层交换机。

**为什么需要 VLAN？**
- **安全性**：财务部和研发部在不同 VLAN，互相不能直接访问
- **广播控制**：广播帧只在同一 VLAN 内传播，减少不必要的流量
- **灵活性**：逻辑分组不受物理位置限制

### 6.8.2 IEEE 802.1Q 标签

802.1Q 在以太网帧的源 MAC 和类型字段之间插入一个 4 字节的 VLAN 标签：

```
标准以太网帧:
┌──────────┬──────────┬──────────┬─────────┬─────┐
│ 目的MAC  │ 源MAC    │ 类型     │ 数据    │ FCS │
└──────────┴──────────┴──────────┴─────────┴─────┘

802.1Q 标签帧:
┌──────────┬──────────┬─────────────────┬──────────┬─────────┬─────┐
│ 目的MAC  │ 源MAC    │ 802.1Q 标签(4B) │ 类型     │ 数据    │ FCS │
└──────────┴──────────┴─────────────────┴──────────┴─────────┴─────┘
```

**802.1Q 标签结构（4 字节 = 32 位）**：

```
┌──────────────┬──────┬──────────────┐
│ TPID (16位)  │ PRI  │ VLAN ID      │
│ 0x8100       │ 3位  │ 12位         │
│              │ CFI  │              │
│              │ 1位  │              │
└──────────────┴──────┴──────────────┘
```

| 字段 | 位数 | 说明 |
|------|------|------|
| TPID | 16 | 标签协议标识，值为 0x8100，标识这是一个 802.1Q 帧 |
| PCP（Priority） | 3 | 优先级（0-7），用于 QoS |
| CFI | 1 | 规范格式指示（以太网中通常为 0） |
| VID（VLAN ID） | 12 | VLAN 标识（0-4095），0 和 4095 保留，可用 1-4094 |

### 6.8.3 端口类型

| 端口类型 | 说明 | 典型连接 |
|---------|------|---------|
| Access 端口 | 只属于一个 VLAN，收发无标签帧 | 连接 PC、服务器 |
| Trunk 端口 | 可承载多个 VLAN，使用 802.1Q 标签 | 交换机之间互联 |
| Hybrid 端口 | 可同时发送有标签和无标签帧 | 特殊场景 |

```python
class VlANTagProcessor:
    """VLAN 标签处理器模拟"""

    def __init__(self):
        self.port_vlans = {}  # port -> set of VLAN IDs
        self.port_types = {}  # port -> 'access' | 'trunk'
        self.pvid = {}        # port -> default VLAN ID (for access ports)

    def configure_access_port(self, port, vlan_id):
        self.port_types[port] = 'access'
        self.pvid[port] = vlan_id
        self.port_vlans[port] = {vlan_id}

    def configure_trunk_port(self, port, allowed_vlans):
        self.port_types[port] = 'trunk'
        self.port_vlans[port] = set(allowed_vlans)

    def ingress_process(self, port, frame):
        """入端口处理：添加 VLAN 标签"""
        if self.port_types[port] == 'access':
            # Access 端口：添加 PVID 标签
            vlan_id = self.pvid[port]
            return self.add_tag(frame, vlan_id)
        elif self.port_types[port] == 'trunk':
            # Trunk 端口：已有标签，检查 VLAN 是否允许
            vlan_id = self.extract_vlan_id(frame)
            if vlan_id in self.port_vlans[port]:
                return frame
            return None  # 丢弃

    def egress_process(self, port, frame):
        """出端口处理：可能需要去除标签"""
        vlan_id = self.extract_vlan_id(frame)
        if vlan_id not in self.port_vlans.get(port, set()):
            return None  # 该端口不属于此 VLAN

        if self.port_types.get(port) == 'access':
            # Access 端口：去除标签发送
            return self.remove_tag(frame)
        else:
            # Trunk 端口：保留标签发送
            return frame
```

<div data-component="VLANSimulator"></div>

---

## 6.9 PoE — 以太网供电

### 6.9.1 PoE 概述

PoE（Power over Ethernet）通过以太网线缆同时传输数据和电力，省去了单独的电源线。

| 标准 | 名称 | 最大供电功率 | 典型应用 |
|------|------|------------|---------|
| IEEE 802.3af | PoE | 15.4W | IP 电话、无线 AP |
| IEEE 802.3at | PoE+ | 30W | PTZ 摄像头、视频电话 |
| IEEE 802.3bt | PoE++ | 60W/100W | LED 照明、数字标牌 |

### 6.9.2 PoE 工作原理

**供电检测阶段**（分级）：
1. PSE（供电设备）发送低电压探测信号
2. PD（受电设备）通过特征电阻响应自己的功率等级
3. PSE 确认后开始供电

**供电方式**：
- **模式 A**（End-span）：通过数据线对供电（1/2, 3/6 线对）
- **模式 B**（Mid-span）：通过空闲线对供电（4/5, 7/8 线对）
- **4PPoE**（802.3bt）：同时使用所有 4 对线

---

## 6.10 核心网络组件深入分析

### 6.10.1 以太网 MAC 控制器的内部结构

以太网 MAC 控制器是网络接口卡（NIC）的核心部件，负责帧的发送与接收。

**内部模块结构**：

```
┌─────────────────────────────────────────────────────────┐
│                    以太网 MAC 控制器                      │
│                                                         │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ 系统总线 │  │ 发送 DMA     │  │ 发送 FIFO        │  │
│  │ 接口     │─►│ 引擎         │─►│ (Tx FIFO)        │  │
│  │ (PCIe)   │  │              │  │                  │──┼──► PHY
│  └──────────┘  └──────────────┘  └──────────────────┘  │
│       │         ┌──────────────┐  ┌──────────────────┐  │
│       │         │ 接收 DMA     │  │ 接收 FIFO        │  │
│       │◄────────│ 引擎         │◄─│ (Rx FIFO)        │◄─┼── PHY
│       │         │              │  │                  │  │
│  ┌────▼─────┐  └──────────────┘  └──────────────────┘  │
│  │ MAC 地址 │  ┌──────────────┐  ┌──────────────────┐  │
│  │ 过滤器   │  │ CRC 校验     │  │ MII/RMII/GMII   │  │
│  │          │  │ 生成/验证    │  │ 接口             │  │
│  └──────────┘  └──────────────┘  └──────────────────┘  │
│  ┌──────────┐  ┌──────────────┐                        │
│  │ 中断控制 │  │ 寄存器组     │                        │
│  │ 器       │  │ (CSR)        │                        │
│  └──────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────┘
```

**发送 FIFO（Tx FIFO）**：
- 典型大小：2KB-8KB
- 作用：缓冲待发送的帧，解决系统总线与网络接口速率不匹配
- 溢出策略：暂停 DMA 传输或丢弃帧

**接收 FIFO（Rx FIFO）**：
- 典型大小：2KB-16KB
- 作用：缓冲接收到的帧，等待 DMA 搬运到主机内存
- 溢出处理：触发溢出中断，统计丢包计数

**MAC 地址过滤器**：
- 实现精确匹配和哈希匹配
- 精确匹配：多个 CAM（Content Addressable Memory）条目
- 哈希匹配：对组播地址做 CRC 哈希，查 64 位哈希表
- 过滤逻辑：接收目标 MAC 是本机单播、广播、匹配组播哈希的帧

```c
/* MAC 地址过滤器逻辑（伪代码） */
int mac_filter(uint8_t *dst_mac, struct mac_filter_table *filter) {
    /* 1. 广播地址：总是接收 */
    if (is_broadcast(dst_mac))
        return ACCEPT;

    /* 2. 精确匹配：本机单播地址 */
    for (int i = 0; i < filter->unicast_count; i++) {
        if (mac_equal(dst_mac, filter->unicast_addrs[i]))
            return ACCEPT;
    }

    /* 3. 组播哈希匹配 */
    if (is_multicast(dst_mac)) {
        uint32_t hash = crc32(dst_mac, 6) & 0x3F;  /* 6-bit hash */
        if (filter->multicast_hash_table & (1ULL << hash))
            return ACCEPT;
    }

    /* 4. 混杂模式检查 */
    if (filter->promiscuous_mode)
        return ACCEPT;

    return REJECT;
}
```

### 6.10.2 CSMA/CD 碰撞检测电路

在 10BASE5 和 10BASE2 同轴电缆以太网中，碰撞检测通过以下机制实现：

**同轴电缆（10BASE5）**：
- 发送器在电缆上注入信号的同时，接收器也在监听
- 比较发送信号与接收信号的幅度：若接收信号幅度明显大于发送信号，说明有其他站点也在发送
- 检测电路使用**差分比较器**，当信号幅度超过阈值时产生碰撞中断

**双绞线（10BASE-T）**：
- 使用独立的发送和接收线对
- 碰撞定义为：在发送的同时在接收线上检测到信号
- 需要通过回波抵消（Echo Cancellation）技术区分本端信号和远端信号

```
碰撞检测电路（概念性）:
┌───────────────────────────────────────┐
│                                       │
│  发送信号 ──►┌──────────┐             │
│             │ 差分比较器 │──► 碰撞指示 │
│  接收信号 ──►│          │             │
│             └──────────┘             │
│                   ▲                   │
│                   │                   │
│            阈值检测器                  │
│            (幅度 > 阈值 → 碰撞)       │
│                                       │
└───────────────────────────────────────┘
```

### 6.10.3 交换机的 MAC 地址表与转发引擎

**MAC 地址表的硬件实现**：

现代交换机使用 **CAM（Content Addressable Memory）** 或 **TCAM（Ternary CAM）** 实现 MAC 地址表的高速查找。

| 特性 | CAM | TCAM |
|------|-----|------|
| 匹配方式 | 精确匹配 | 三态匹配（0, 1, X） |
| 查找速度 | 一个时钟周期 | 一个时钟周期 |
| 容量 | 较大（64K-256K条目） | 较小（数千条目） |
| 成本 | 较低 | 较高 |
| 应用 | MAC 地址查找 | ACL、QoS 规则 |

**转发引擎的数据通路**：

```
入端口处理 → MAC 地址学习 → MAC 地址查找 → 转发决策 → 出端口排队
    │              │              │              │
    ▼              ▼              ▼              ▼
  解析帧头      更新CAM表      CAM匹配        复制/转发
  提取MAC       源MAC→端口     目的MAC→端口    丢弃(未知单播→泛洪)
```

```python
class SwitchForwardingEngine:
    """交换机转发引擎模拟"""

    def __init__(self, num_ports):
        self.mac_table = {}       # MAC -> (port, timestamp)
        self.num_ports = num_ports
        self.aging_time = 300     # 5 分钟

    def learn(self, src_mac, in_port):
        """MAC 地址学习"""
        self.mac_table[src_mac] = (in_port, time.time())

    def lookup(self, dst_mac):
        """MAC 地址查找"""
        if dst_mac in self.mac_table:
            port, ts = self.mac_table[dst_mac]
            if time.time() - ts < self.aging_time:
                return port       # 转发到已知端口
            else:
                del self.mac_table[dst_mac]  # 老化删除
        return None               # 未知

    def forward(self, frame, in_port):
        """转发决策"""
        src_mac = frame['src_mac']
        dst_mac = frame['dst_mac']

        # 1. 学习源 MAC
        self.learn(src_mac, in_port)

        # 2. 查找目的 MAC
        out_port = self.lookup(dst_mac)

        if out_port is not None:
            if out_port == in_port:
                return []         # 同端口，丢弃
            return [out_port]     # 单播转发
        else:
            # 3. 未知单播：泛洪（除入端口外的所有端口）
            return [p for p in range(self.num_ports) if p != in_port]
```

### 6.10.4 ARP 缓存表的数据结构

ARP 缓存表在实际系统中的实现需要考虑并发、老化、溢出等问题：

```c
/* 高效 ARP 缓存表实现（使用哈希表 + 双向链表） */

struct arp_cache_entry {
    uint32_t        ip_addr;
    uint8_t         mac_addr[6];
    uint8_t         state;          /* COMPLETE, INCOMPLETE, STALE */
    uint8_t         flags;
    time_t          last_used;
    time_t          last_updated;
    struct sk_buff_head pending;    /* 等待解析的包队列 */
    struct arp_cache_entry *hash_next;
    struct arp_cache_entry *lru_prev;
    struct arp_cache_entry *lru_next;
};

struct arp_cache_table {
    struct arp_cache_entry *hash_table[ARP_HASH_SIZE];  /* 哈希桶 */
    struct arp_cache_entry *lru_head;   /* LRU 链表头（最久未用） */
    struct arp_cache_entry *lru_tail;   /* LRU 链表尾（最近使用） */
    int entry_count;
    int max_entries;
    rwlock_t lock;
};

/* 查找过程：哈希 O(1) + LRU 更新 */
struct arp_cache_entry* arp_lookup(struct arp_cache_table *table,
                                    uint32_t ip) {
    uint32_t hash = arp_hash(ip, ARP_HASH_SIZE);
    struct arp_cache_entry *entry = table->hash_table[hash];

    while (entry) {
        if (entry->ip_addr == ip && entry->state == COMPLETE) {
            /* 检查是否过期 */
            if (time(NULL) - entry->last_updated > ARP_TIMEOUT) {
                entry->state = STALE;
                return NULL;
            }
            /* 移到 LRU 尾部（最近使用） */
            lru_move_to_tail(table, entry);
            entry->last_used = time(NULL);
            return entry;
        }
        entry = entry->hash_next;
    }
    return NULL;
}
```

**性能指标**：
- 查找时间：O(1) 哈希查找
- 空间复杂度：O(N)，N 为缓存条目数
- 并发支持：读写锁分离，读操作可并行
- 老化策略：LRU + 定时器，超时条目被清除

---

## 6.11 本章总结

| 概念 | 核心要点 |
|------|---------|
| 信道分配 | 静态（FDM/TDM）适合稳定流量，动态适合突发流量 |
| 纯 ALOHA | 最大吞吐 18.4%，脆弱期 2t |
| 时隙 ALOHA | 最大吞吐 36.8%，脆弱期 t |
| CSMA | 先听后说，1-坚持/非坚持/p-坚持 |
| CSMA/CD | 边说边听，最小帧 64 字节，二进制指数退避 |
| 以太网帧 | DIX（类型）vs 802.3（长度），最小 64B 最大 1518B |
| MAC 地址 | 48 位，I/G 位 + U/L 位 + OUI + 设备标识 |
| ARP | IP→MAC 映射，缓存+超时+代理 ARP |
| VLAN | 802.1Q 标签，4 字节，12 位 VID |
| PoE | 通过网线供电，802.3af/at/bt |

<div data-component="ChapterSummaryQuiz"></div>

---

## 6.12 以太网物理层技术演进

### 6.12.1 以太网速率演进

| 标准 | 名称 | 速率 | 介质 | 最大距离 |
|------|------|------|------|---------|
| 802.3 | 10BASE5 | 10 Mbps | 粗同轴电缆 | 500m |
| 802.3u | 100BASE-TX | 100 Mbps | Cat5 双绞线 | 100m |
| 802.3ab | 1000BASE-T | 1 Gbps | Cat5e 双绞线 | 100m |
| 802.3ae | 10GBASE-SR | 10 Gbps | 多模光纤 | 300m |
| 802.3ba | 40GBASE-SR4 | 40 Gbps | 多模光纤 | 150m |
| 802.3bs | 100GBASE-SR4 | 100 Gbps | 多模光纤 | 100m |
| 802.3cu | 400GBASE-SR8 | 400 Gbps | 多模光纤 | 100m |

### 6.12.2 全双工与半双工

| 模式 | 说明 | 碰撞检测 | 适用场景 |
|------|------|---------|---------|
| 半双工 | 同一时刻只能收或发 | 需要 CSMA/CD | 集线器（Hub）连接 |
| 全双工 | 可同时收发 | 不需要 CSMA/CD | 交换机连接 |

全双工以太网使用独立的发送和接收线对，消除了碰撞域，不需要 CSMA/CD。现代以太网几乎都工作在全双工模式。

### 6.12.3 自动协商（Auto-Negotiation）

以太网设备通过自动协商确定双工模式和速率：

```
自动协商过程 (1000BASE-T):

1. 设备发送快速链路脉冲 (FLP) 突发
   - 包含 17 个时钟脉冲和 16 个数据脉冲
   - 编码设备的能力（速率、双工模式等）

2. 对端设备接收并解析 FLP

3. 双方选择共同支持的最高性能模式
   优先级: 1000BASE-T 全双工 > 1000BASE-T 半双工
         > 100BASE-TX 全双工 > 100BASE-TX 半双工
         > 10BASE-T 全双工 > 1BASE-T 半双工

4. 链路建立，开始数据传输
```

### 6.12.4 巨型帧（Jumbo Frame）

标准以太网帧最大 1500 字节（MTU）。巨型帧将 MTU 扩展到 9000 字节：

| 特性 | 标准帧 (MTU=1500) | 巨型帧 (MTU=9000) |
|------|-------------------|-------------------|
| 最大帧长 | 1518 字节 | 9018 字节 |
| 有效载荷效率 | 96.2% | 99.1% |
| CPU 开销 | 高（每字节处理次数多） | 低 |
| 兼容性 | 所有设备 | 需要全路径支持 |
| 适用场景 | 通用 | 数据中心内部 |

```python
def calculate_efficiency(mtu, header_overhead=38):
    """
    计算以太网帧的有效载荷效率
    header_overhead: 以太网头(14) + VLAN标签(4) + IP头(20) = 38 字节
    """
    total_frame_size = mtu + header_overhead + 4  # +4 for FCS
    efficiency = mtu / total_frame_size
    return efficiency * 100

print(f"标准帧效率: {calculate_efficiency(1500):.1f}%")
print(f"巨型帧效率: {calculate_efficiency(9000):.1f}%")
```

<div dataComponent="EthernetEvolutionDemo"></div>

---

## 练习题

**1. 计算题**：纯 ALOHA 的信道带宽为 9.6 kbps，帧长为 1000 比特。求最大吞吐量（帧/秒）。

**2. 分析题**：某以太网最大段长为 500m，信号传播速度为 $2 \times 10^8$ m/s。求最小帧长（比特）。

**3. 设计题**：设计一个简单的 ARP 缓存表，支持插入、查找、删除和超时清除操作，要求使用哈希表实现，分析时间复杂度。

**4. 比较题**：对比 1-坚持 CSMA、非坚持 CSMA 和 p-坚持 CSMA 的优缺点，说明各适合什么场景。

**5. 实践题**：用 Wireshark 抓取 ARP 请求和应答报文，分析其帧格式中各字段的值。

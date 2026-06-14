# Chapter 16: 传输层 — TCP 拥塞控制

> **学习目标**：
> - 理解网络拥塞的原因、表现形式和度量指标
> - 掌握 AIMD（加性增/乘性减）原理及其对公平性的影响
> - 深入理解 TCP Tahoe 的慢启动、拥塞避免和快速重传机制
> - 掌握 TCP Reno 的快速恢复算法及其与 Tahoe 的区别
> - 理解 TCP NewReno 对 Reno 的改进，解决多个丢包问题
> - 掌握 TCP CUBIC 的三次函数设计和窗口增长策略
> - 了解 TCP BBR 的基于带宽-延迟模型的拥塞控制思路
> - 理解 ECN 和主动队列管理（RED/CoDel/FQ-PIE）的工作原理

---

## 16.1 网络拥塞概述

### 16.1.1 拥塞的原因

网络拥塞是指网络中的数据量超过了网络的处理能力：

```
拥塞的根本原因：

┌─────────────────────────────────────────────────────────────┐
│                    网络拥塞示意                               │
│                                                             │
│         发送方 A ──→ ┌──────────┐                            │
│                      │  路由器   │──→ 接收方                  │
│         发送方 B ──→ │ (瓶颈)   │                            │
│                      └──────────┘                            │
│                                                             │
│  瓶颈路由器的处理能力：100 Mbps                              │
│  发送方 A 的发送速率：60 Mbps                                │
│  发送方 B 的发送速率：60 Mbps                                │
│  总输入速率：120 Mbps > 100 Mbps → 拥塞！                    │
│                                                             │
│  结果：                                                     │
│  • 路由器队列增长                                            │
│  • 新到达的数据包被丢弃                                      │
│  • 所有发送方的重传增加网络负载                               │
│  • 吞吐量下降（拥塞崩溃）                                    │
└─────────────────────────────────────────────────────────────┘
```

### 16.1.2 拥塞的表现

```
拥塞的四个阶段：

1. 轻度拥塞
   ┌─────────────────────────────────────────────────────┐
   │ • 队列长度增加                                       │
   │ • 排队延迟增加                                       │
   │ • 吞吐量基本不变                                     │
   │ • RTT 增加                                           │
   └─────────────────────────────────────────────────────┘

2. 中度拥塞
   ┌─────────────────────────────────────────────────────┐
   │ • 队列接近满                                         │
   │ • 部分数据包被丢弃                                   │
   │ • 发送方检测到丢包，开始重传                         │
   │ • 吞吐量开始下降                                     │
   └─────────────────────────────────────────────────────┘

3. 严重拥塞
   ┌─────────────────────────────────────────────────────┐
   │ • 队列完全满                                         │
   │ • 大量数据包被丢弃                                   │
   │ • 发送方频繁超时重传                                 │
   │ • 吞吐量急剧下降                                     │
   └─────────────────────────────────────────────────────┘

4. 拥塞崩溃（Congestion Collapse）
   ┌─────────────────────────────────────────────────────┐
   │ • 网络中大部分是重传包                               │
   │ • 有效吞吐量接近零                                   │
   │ • 1986 年 NSFnet 曾经历拥塞崩溃                      │
   │ • 吞吐量从 32Kbps 降到 40bps                         │
   └─────────────────────────────────────────────────────┘
```

### 16.1.3 拥塞控制 vs 流量控制

```
拥塞控制 vs 流量控制：

┌──────────────────────┬────────────────────────────────────┐
│       流量控制         │           拥塞控制                  │
├──────────────────────┼────────────────────────────────────┤
│ 解决接收方处理能力不足 │ 解决网络传输能力不足                 │
│ 端到端问题            │ 全局问题                            │
│ 基于接收窗口 rwnd     │ 基于拥塞窗口 cwnd                   │
│ 接收方通告窗口大小     │ 发送方根据网络状况调整               │
│ 防止接收方缓冲区溢出   │ 防止网络路由器队列溢出               │
│ 只涉及收发双方        │ 涉及所有共享网络的发送方              │
└──────────────────────┴────────────────────────────────────┘

发送窗口 = min(rwnd, cwnd)
发送方的实际发送速率受两个窗口的共同限制
```

<div data-component="CongestionSimulator"></div>

---

## 16.2 AIMD 原理

### 16.2.1 AIMD 的基本思想

AIMD（Additive Increase Multiplicative Decrease，加性增/乘性减）是 TCP 拥塞控制的核心策略：

```
AIMD 工作原理：

┌─────────────────────────────────────────────────────────────┐
│                    AIMD 策略                                  │
│                                                             │
│  加性增加 (Additive Increase)：                              │
│  • 每个 RTT 增加 1 MSS                                      │
│  • 线性增长，探测可用带宽                                    │
│  • 缓慢但稳定地增加发送速率                                  │
│                                                             │
│  乘性减 (Multiplicative Decrease)：                          │
│  • 检测到拥塞时，窗口减半                                    │
│  • 快速降低发送速率                                          │
│  • 对拥塞做出快速反应                                        │
│                                                             │
│  窗口变化图示：                                               │
│                                                             │
│  cwnd                                                        │
│  ▲                                                          │
│  │      /\      /\      /\                                  │
│  │     /  \    /  \    /  \                                 │
│  │    /    \  /    \  /    \                                │
│  │   /      \/      \/      \                               │
│  │  /                                                  \    │
│  └──────────────────────────────────────────────────────→  │
│                        时间                                  │
│                                                             │
│  这就是著名的"锯齿"模式                                       │
└─────────────────────────────────────────────────────────────┘
```

### 16.2.2 AIMD 的公平性

```
AIMD 的公平性分析：

两条连接共享一条链路，带宽为 R

连接 1 的窗口：cwnd1
连接 2 的窗口：cwnd2

公平时：cwnd1 = cwnd2 = R/2

AIMD 如何收敛到公平：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  cwnd2                                                      │
│  ▲                                                          │
│  │                                                          │
│  │    ╲  公平线 (cwnd1 = cwnd2)                              │
│  │     ╲                                                    │
│  │      ╲                                                   │
│  │       ╲←── 先减窗口较大的连接                             │
│  │        ╲                                                 │
│  │    ●───→╲●                                               │
│  │          ╲←── 然后两者都线性增加                           │
│  │           ╲                                              │
│  │            ●───→●                                        │
│  │                 ╲                                        │
│  │                  ●───→ 公平点                             │
│  │                                                          │
│  └──────────────────────────────────────────────────────→  │
│                         cwnd1                               │
│                                                             │
│  结论：AIMD 是公平的，会收敛到公平分配                        │
│  但 AI(加性增) 和 MD(乘性减) 的比例影响公平性和效率           │
└─────────────────────────────────────────────────────────────┘
```

---

## 16.3 TCP Tahoe

### 16.3.1 Tahoe 的算法概述

TCP Tahoe（1988 年由 Van Jacobson 提出）是第一个现代 TCP 拥塞控制算法：

```
TCP Tahoe 的三个阶段：

┌─────────────────────────────────────────────────────────────┐
│                    TCP Tahoe                                  │
│                                                             │
│  1. 慢启动 (Slow Start)                                      │
│     • cwnd 从 1 MSS 开始                                    │
│     • 每收到一个 ACK，cwnd += 1 MSS                          │
│     • 每个 RTT，cwnd 翻倍（指数增长）                        │
│     • 直到 cwnd >= ssthresh                                  │
│                                                             │
│  2. 拥塞避免 (Congestion Avoidance)                          │
│     • 每个 RTT，cwnd += 1 MSS                                │
│     • 线性增长                                               │
│     • 直到检测到丢包                                         │
│                                                             │
│  3. 丢包检测后：                                              │
│     • ssthresh = cwnd / 2                                    │
│     • cwnd = 1 MSS                                          │
│     • 重新进入慢启动                                         │
│                                                             │
│  cwnd                                                        │
│  ▲                                                          │
│  │                                                          │
│  │          ssthresh                                        │
│  │            ↓                                             │
│  │   慢启动   │ 拥塞避免                                      │
│  │  (指数增)  │ (线性增)                                      │
│  │   /\      │       /                                      │
│  │  /  \     │      /                                        │
│  │ /    \    │     /                                         │
│  │/      \   │    /                                          │
│  │        \  │   /    丢包                                    │
│  │         \ │  /   ──→ cwnd=1, ssthresh=cwnd/2              │
│  │          \│ /                                             │
│  │           \/                                              │
│  │           │ 重新慢启动                                     │
│  └──────────────────────────────────────────────────────→  │
│                        时间                                  │
└─────────────────────────────────────────────────────────────┘
```

### 16.3.2 Tahoe 的实现

```python
class TCPTahoe:
    """TCP Tahoe 拥塞控制算法"""

    MSS = 1460  # 最大段大小

    def __init__(self):
        self.cwnd = self.MSS        # 拥塞窗口
        self.ssthresh = 65535       # 慢启动阈值
        self.state = "SLOW_START"   # 当前状态
        self.dup_ack_count = 0      # 重复 ACK 计数

    def on_ack_received(self, ack_num: int):
        """收到 ACK 时的处理"""
        if self.state == "SLOW_START":
            # 慢启动：每个 ACK 增加 1 MSS
            self.cwnd += self.MSS

            # 检查是否进入拥塞避免
            if self.cwnd >= self.ssthresh:
                self.state = "CONGESTION_AVOIDANCE"

        elif self.state == "CONGESTION_AVOIDANCE":
            # 拥塞避免：每个 RTT 增加 1 MSS
            # 实际实现：每个 ACK 增加 MSS/cwnd
            self.cwnd += self.MSS * (self.MSS / self.cwnd)

        self.dup_ack_count = 0

    def on_duplicate_ack(self, ack_num: int):
        """收到重复 ACK 时的处理"""
        self.dup_ack_count += 1

        if self.dup_ack_count == 3:
            # 快速重传
            self._fast_retransmit()

    def on_timeout(self):
        """超时处理"""
        self.ssthresh = self.cwnd / 2
        self.cwnd = self.MSS
        self.state = "SLOW_START"
        self.dup_ack_count = 0

    def _fast_retransmit(self):
        """快速重传"""
        self.ssthresh = self.cwnd / 2
        self.cwnd = self.MSS
        self.state = "SLOW_START"
        # 重传丢失的段

    def get_cwnd(self) -> float:
        """获取当前拥塞窗口"""
        return self.cwnd

    def get_state(self) -> str:
        """获取当前状态"""
        return self.state
```

---

## 16.4 TCP Reno

### 16.4.1 Reno 的改进

TCP Reno 在 Tahoe 的基础上增加了**快速恢复（Fast Recovery）**：

```
TCP Reno vs Tahoe：

┌─────────────────────────────────────────────────────────────┐
│                    TCP Reno                                   │
│                                                             │
│  Reno 新增：快速恢复 (Fast Recovery)                         │
│                                                             │
│  核心思想：                                                  │
│  • 收到 3 个重复 ACK 时，不回到慢启动                        │
│  • 而是进入快速恢复状态                                      │
│  • cwnd = ssthresh + 3（而不是 cwnd = 1）                   │
│  • 每收到一个重复 ACK，cwnd += 1 MSS                        │
│  • 收到新 ACK 后，进入拥塞避免                               │
│                                                             │
│  Reno 的三个阶段：                                           │
│  1. 慢启动 (Slow Start)                                      │
│  2. 拥塞避免 (Congestion Avoidance)                          │
│  3. 快速恢复 (Fast Recovery) ← 新增                          │
│                                                             │
│  Reno 的优势：                                               │
│  • 快速恢复避免了快速重传后回到慢启动                         │
│  • 在单个丢包情况下性能更好                                   │
│  • 保持较高的吞吐量                                          │
└─────────────────────────────────────────────────────────────┘
```

### 16.4.2 Reno 的实现

```python
class TCPReno:
    """TCP Reno 拥塞控制算法"""

    MSS = 1460

    def __init__(self):
        self.cwnd = self.MSS
        self.ssthresh = 65535
        self.state = "SLOW_START"
        self.dup_ack_count = 0

    def on_ack_received(self, ack_num: int):
        """收到 ACK 时的处理"""
        if self.state == "SLOW_START":
            self.cwnd += self.MSS
            if self.cwnd >= self.ssthresh:
                self.state = "CONGESTION_AVOIDANCE"

        elif self.state == "CONGESTION_AVOIDANCE":
            self.cwnd += self.MSS * (self.MSS / self.cwnd)

        elif self.state == "FAST_RECOVERY":
            # 快速恢复中收到新 ACK
            self.cwnd = self.ssthresh
            self.state = "CONGESTION_AVOIDANCE"

        self.dup_ack_count = 0

    def on_duplicate_ack(self, ack_num: int):
        """收到重复 ACK 时的处理"""
        self.dup_ack_count += 1

        if self.dup_ack_count == 3:
            # 进入快速恢复
            self.ssthresh = self.cwnd / 2
            self.cwnd = self.ssthresh + 3 * self.MSS
            self.state = "FAST_RECOVERY"
            # 重传丢失的段

        elif self.state == "FAST_RECOVERY":
            # 快速恢复中收到重复 ACK
            self.cwnd += self.MSS

    def on_timeout(self):
        """超时处理"""
        self.ssthresh = self.cwnd / 2
        self.cwnd = self.MSS
        self.state = "SLOW_START"
        self.dup_ack_count = 0
```

### 16.4.3 Reno 的问题

```
TCP Reno 的局限性：

问题：多个丢包时性能下降

场景：发送 10 个段，段 3 和段 7 丢失

Reno 的处理：
1. 收到段 1-2 的 ACK
2. 段 3 丢失
3. 收到段 4-6 → 发送重复 ACK
4. 收到 3 个重复 ACK → 快速重传段 3
5. 进入快速恢复
6. 收到段 7-10 → 发送重复 ACK
7. 段 7 丢失
8. Reno 只能重传段 3，不知道段 7 也丢失了
9. 段 7 的重传要等到超时

解决方案：
• NewReno：改进快速恢复，处理多个丢包
• SACK：选择确认，明确哪些段已收到
```

---

## 16.5 TCP NewReno

### 16.5.1 NewReno 的改进

TCP NewReno 解决了 Reno 在多个丢包时的性能问题：

```
NewReno 的核心改进：

1. 区分"完整 ACK"和"部分 ACK"
   • 完整 ACK：确认了所有在快速恢复前发送的数据
   • 部分 ACK：只确认了部分数据（还有丢包）

2. 收到部分 ACK 时：
   • 不退出快速恢复
   • 重传下一个可能丢失的段
   • 保持 cwnd 不变（或适当调整）

NewReno 的快速恢复算法：
┌─────────────────────────────────────────────────────────────┐
│  当收到 3 个重复 ACK 时：                                    │
│  1. ssthresh = cwnd / 2                                     │
│  2. cwnd = ssthresh + 3 * MSS                               │
│  3. 重传丢失的段                                             │
│  4. 记录恢复点：recovery_point = snd_nxt                     │
│                                                             │
│  在快速恢复期间：                                             │
│  • 收到重复 ACK：cwnd += MSS                                 │
│  • 收到部分 ACK（ack < recovery_point）：                     │
│    - 重传 ack 指向的段                                       │
│    - cwnd = ssthresh                                        │
│  • 收到完整 ACK（ack >= recovery_point）：                    │
│    - cwnd = ssthresh                                        │
│    - 退出快速恢复，进入拥塞避免                               │
└─────────────────────────────────────────────────────────────┘
```

### 16.5.2 NewReno 的实现

```python
class TCPNewReno:
    """TCP NewReno 拥塞控制算法"""

    MSS = 1460

    def __init__(self):
        self.cwnd = self.MSS
        self.ssthresh = 65535
        self.state = "SLOW_START"
        self.dup_ack_count = 0
        self.recovery_point = 0  # 恢复点
        self.high_seq = 0        # 快速恢复前的最高序列号

    def on_ack_received(self, ack_num: int):
        """收到 ACK 时的处理"""
        if self.state == "SLOW_START":
            self.cwnd += self.MSS
            if self.cwnd >= self.ssthresh:
                self.state = "CONGESTION_AVOIDANCE"

        elif self.state == "CONGESTION_AVOIDANCE":
            self.cwnd += self.MSS * (self.MSS / self.cwnd)

        elif self.state == "FAST_RECOVERY":
            if ack_num >= self.recovery_point:
                # 完整 ACK：退出快速恢复
                self.cwnd = self.ssthresh
                self.state = "CONGESTION_AVOIDANCE"
            else:
                # 部分 ACK：重传下一个丢失的段
                self.cwnd = self.ssthresh
                # 重传 ack_num 指向的段

        self.dup_ack_count = 0

    def on_duplicate_ack(self, ack_num: int):
        """收到重复 ACK 时的处理"""
        self.dup_ack_count += 1

        if self.dup_ack_count == 3:
            self.ssthresh = self.cwnd / 2
            self.cwnd = self.ssthresh + 3 * self.MSS
            self.state = "FAST_RECOVERY"
            self.recovery_point = self.high_seq
            # 重传丢失的段

        elif self.state == "FAST_RECOVERY":
            self.cwnd += self.MSS

    def on_timeout(self):
        """超时处理"""
        self.ssthresh = self.cwnd / 2
        self.cwnd = self.MSS
        self.state = "SLOW_START"
        self.dup_ack_count = 0
```

<div data-component="TCPCongestionComparison"></div>

---

## 16.6 TCP CUBIC

### 16.6.1 CUBIC 的设计动机

TCP CUBIC 是目前 Linux 内核默认的拥塞控制算法（2004 年提出）：

```
CUBIC 的设计目标：

1. 更好的可扩展性
   • Reno 在高带宽长延迟网络中增长太慢
   • 10Gbps 网络，从 cwnd=1 增长到 10000+ 需要很长时间

2. 更好的 RTT 公平性
   • Reno 中，RTT 短的连接增长更快
   • 导致不同 RTT 的连接之间不公平

3. 更高的链路利用率
   • Reno 在丢包后窗口减半，利用率下降
   • CUBIC 在丢包后保持较大的窗口

4. 更好的稳定性
   • Reno 的锯齿形波动大
   • CUBIC 的窗口变化更平滑
```

### 16.6.2 CUBIC 的三次函数

CUBIC 使用三次函数来控制窗口增长：

```
CUBIC 窗口增长函数：

W(t) = C * (t - K)^3 + W_max

其中：
• W(t): 时间 t 时的窗口大小
• C: 缩放常数（默认 0.4）
• t: 距上次丢包的时间
• K = (W_max * β / C)^(1/3)
• W_max: 上次丢包时的窗口大小
• β: 乘性减系数（默认 0.7，即丢包后窗口变为 70%）

CUBIC 的三个阶段：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  cwnd                                                        │
│  ▲                                                          │
│  │                                                          │
│  │          W_max                                           │
│  │           ↓                                              │
│  │           ●─────●                                        │
│  │          ╱       ╲                                       │
│  │         ╱         ╲                                      │
│  │        ╱           ╲                                     │
│  │       ╱  立即增长    ╲  稳定区域                          │
│  │      ╱   (快速恢复)   ╲ (探测带宽)                        │
│  │     ╱                 ╲                                  │
│  │    ╱                   ●                                 │
│  │   ╱                    │                                 │
│  │  ●                     │                                 │
│  │  │                     │                                 │
│  │  │←────── convex ─────→│←──── concave ───→│             │
│  │  │                     │                   │             │
│  └──┼─────────────────────┼───────────────────┼───────────→│
│     │                     │                   │       时间   │
│     丢包点                 K                   │             │
│                                         W_max 再次到达      │
│                                                             │
│  特点：                                                     │
│  1. 丢包后立即快速恢复到接近 W_max                           │
│  2. 在 W_max 附近增长变慢（稳定区域）                        │
│  3. 超过 W_max 后继续探测更高带宽                            │
└─────────────────────────────────────────────────────────────┘
```

### 16.6.3 CUBIC 的实现

```python
import math

class TCPCubic:
    """TCP CUBIC 拥塞控制算法"""

    MSS = 1460
    C = 0.4           # 缩放常数
    BETA = 0.7        # 乘性减系数

    def __init__(self):
        self.cwnd = self.MSS
        self.ssthresh = 65535
        self.w_max = 0         # 上次丢包时的窗口
        self.k = 0             # 时间偏移
        self.epoch_start = 0   # 当前 epoch 开始时间
        self.tcp_friendliness = True  # TCP 友好模式

    def _calculate_k(self):
        """计算 K 值"""
        self.k = math.pow(self.w_max * (1 - self.BETA) / self.C, 1/3)

    def _cubic_function(self, t: float) -> float:
        """计算 CUBIC 函数值"""
        return self.C * math.pow(t - self.k, 3) + self.w_max

    def on_ack_received(self, ack_num: int, current_time: float):
        """收到 ACK 时的处理"""
        if self.epoch_start == 0:
            self.epoch_start = current_time
            self._calculate_k()

        t = current_time - self.epoch_start

        # 计算 CUBIC 窗口
        cubic_wnd = self._cubic_function(t)

        # TCP 友好模式：与 Reno 比较
        if self.tcp_friendliness:
            # Reno 的窗口增长
            reno_wnd = self.w_max * self.BETA + \
                       (1 - self.BETA) * (t / (self.cwnd / self.MSS))
            # 取较大值
            target_wnd = max(cubic_wnd, reno_wnd)
        else:
            target_wnd = cubic_wnd

        # 更新 cwnd
        if target_wnd > self.cwnd:
            # 增长
            self.cwnd = min(target_wnd, self.cwnd + self.MSS)
        else:
            # 减少或保持
            self.cwnd = max(target_wnd, self.cwnd - self.MSS)

    def on_packet_loss(self, current_time: float):
        """丢包处理"""
        if self.cwnd < self.w_max:
            # 快速恢复
            self.w_max = self.cwnd
            self._calculate_k()
        else:
            # 窗口已超过上次最大值
            self.w_max = self.cwnd

        self.ssthresh = self.cwnd * self.BETA
        self.cwnd = self.cwnd * self.BETA
        self.epoch_start = current_time

    def on_timeout(self):
        """超时处理"""
        self.ssthresh = self.cwnd * self.BETA
        self.cwnd = self.MSS
        self.epoch_start = 0

    def get_cwnd(self) -> float:
        """获取当前拥塞窗口"""
        return self.cwnd

    def get_w_max(self) -> float:
        """获取 W_max"""
        return self.w_max


# 示例：CUBIC 窗口变化
def simulate_cubic():
    """模拟 CUBIC 算法的窗口变化"""
    cubic = TCPCubic()
    cubic.w_max = 100000  # 假设上次丢包时窗口为 100KB

    results = []
    for t in range(0, 100):
        cubic.on_ack_received(0, t)
        results.append((t, cubic.cwnd / cubic.MSS))

    return results
```

<div data-component="CUBICVisualizer"></div>

---

## 16.7 TCP BBR

### 16.7.1 BBR 的设计理念

TCP BBR（Bottleneck Bandwidth and Round-trip propagation time）由 Google 于 2016 年提出，是一种基于模型的拥塞控制算法：

```
BBR 的核心思想：

传统算法（Reno/CUBIC）：
• 基于丢包的拥塞信号
• 只有丢包才降低速率
• 在浅缓冲区网络中过度降低速率

BBR：
• 基于带宽和延迟的测量
• 主动探测最优操作点
• 在浅缓冲区网络中表现更好

BBR 的两个关键参数：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  1. 瓶颈带宽 (BtlBw)                                        │
│     • 测量最近一段时间的最大交付速率                          │
│     • 使用滑动窗口最大值滤波器                               │
│     • 单位：bytes/s                                         │
│                                                             │
│  2. 最小 RTT (RTprop)                                       │
│     • 测量最近一段时间的最小 RTT                              │
│     • 使用滑动窗口最小值滤波器                               │
│     • 单位：秒                                               │
│                                                             │
│  BBR 的目标：                                                │
│  操作在 BtlBw 和 RTprop 的交汇点                             │
│  即：发送速率 = BtlBw，发送量 = BtlBw * RTprop               │
│                                                             │
│         │                                                    │
│  吞吐量  │          /                                        │
│  ▲      │         /  ← 应用延迟限制                          │
│  │      │        /                                           │
│  │      │       /                                            │
│  │      │      ● ← BBR 操作点                                │
│  │      │     /│                                             │
│  │      │    / │                                             │
│  │      │   /  │ ← 缓冲区延迟限制                            │
│  │      │  /   │                                             │
│  │      │ /    │                                             │
│  │      │/     │                                             │
│  │      ●──────┼───────────────────→                       │
│  │    RTprop    │                                    发送速率  │
│  │              │                                            │
│  └──────────────┴────────────────────────────────────────── │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 16.7.2 BBR 的状态机

```
BBR 的四个状态：

┌─────────────────────────────────────────────────────────────┐
│                    BBR 状态机                                 │
│                                                             │
│  1. Startup（启动）                                          │
│     • 类似慢启动                                             │
│     • 指数增长探测带宽                                        │
│     • 当带宽不再增长时退出                                    │
│                                                             │
│  2. Drain（排空）                                            │
│     • 降低发送速率                                           │
│     • 排空 Startup 阶段堆积的队列                             │
│     • 当 inflight <= BtlBw * RTprop 时退出                   │
│                                                             │
│  3. ProbeBW（带宽探测）                                      │
│     • 主要工作状态                                           │
│     • 周期性探测更高带宽                                      │
│     • 8 个阶段循环：↑ → → → → → → ↓                        │
│     • 增益系数：[1.25, 0.75, 1, 1, 1, 1, 1, 1]             │
│                                                             │
│  4. ProbeRTT（RTT 探测）                                     │
│     • 周期性进入                                             │
│     • 降低 cwnd 到 4                                         │
│     • 测量最小 RTT                                           │
│     • 持续至少 200ms                                         │
│                                                             │
│  状态转换：                                                   │
│  Startup → Drain → ProbeBW ↔ ProbeRTT                      │
└─────────────────────────────────────────────────────────────┘
```

### 16.7.3 BBR 的带宽估计器

```python
class BBRBandwidthEstimator:
    """BBR 带宽估计器"""

    def __init__(self, window_size=10):
        self.window_size = window_size  # 滑动窗口大小（RTT 数）
        self.delivered = 0              # 已交付的总字节数
        self.delivered_time = 0         # 最后交付的时间
        self.bw_samples = []            # 带宽样本
        self.max_bw = 0                 # 最大带宽

    def on_ack(self, bytes_delivered: int, current_time: float):
        """收到 ACK 时更新带宽估计"""
        self.delivered += bytes_delivered

        # 计算瞬时带宽
        if self.delivered_time > 0:
            time_delta = current_time - self.delivered_time
            if time_delta > 0:
                bw = bytes_delivered / time_delta
                self.bw_samples.append(bw)

                # 保持窗口大小
                if len(self.bw_samples) > self.window_size:
                    self.bw_samples.pop(0)

                # 更新最大带宽
                self.max_bw = max(self.bw_samples)

        self.delivered_time = current_time

    def get_max_bw(self) -> float:
        """获取估计的最大带宽"""
        return self.max_bw


class BBRRTTEstimator:
    """BBR RTT 估计器"""

    def __init__(self, window_size=10):
        self.window_size = window_size
        self.rtt_samples = []
        self.min_rtt = float('inf')
        self.min_rtt_timestamp = 0
        self.min_rtt_validity = 600  # 600 秒有效期

    def on_rtt_sample(self, rtt: float, current_time: float):
        """更新 RTT 样本"""
        self.rtt_samples.append(rtt)

        # 保持窗口大小
        if len(self.rtt_samples) > self.window_size:
            self.rtt_samples.pop(0)

        # 更新最小 RTT
        if rtt < self.min_rtt:
            self.min_rtt = rtt
            self.min_rtt_timestamp = current_time

        # 检查最小 RTT 是否过期
        if current_time - self.min_rtt_timestamp > self.min_rtt_validity:
            self.min_rtt = float('inf')

    def get_min_rtt(self) -> float:
        """获取最小 RTT"""
        return self.min_rtt


class TCPBBR:
    """TCP BBR 拥塞控制算法"""

    MSS = 1460

    def __init__(self):
        self.state = "STARTUP"
        self.bw_estimator = BBRBandwidthEstimator()
        self.rtt_estimator = BBRRTTEstimator()
        self.pacing_gain = 2.885  # Startup 增益
        self.cwnd_gain = 2.0
        self.cwnd = self.MSS
        self.round_count = 0
        self.probe_rtt_expired = False

    def on_ack(self, bytes_delivered: int, rtt: float, current_time: float):
        """收到 ACK 时的处理"""
        self.bw_estimator.on_ack(bytes_delivered, current_time)
        self.rtt_estimator.on_rtt_sample(rtt, current_time)

        # 计算目标 cwnd
        bdp = self.bw_estimator.get_max_bw() * self.rtt_estimator.get_min_rtt()
        target_cwnd = bdp * self.cwnd_gain

        # 更新 cwnd
        self.cwnd = max(self.cwnd, target_cwnd)

        # 状态转换
        if self.state == "STARTUP":
            if self.bw_estimator.get_max_bw() > 0:
                # 检查带宽是否停止增长
                self.state = "DRAIN"
                self.pacing_gain = 0.5  # 排空增益

        elif self.state == "DRAIN":
            if self._inflight() <= bdp:
                self.state = "PROBE_BW"
                self.pacing_gain = 1.0

        elif self.state == "PROBE_BW":
            # 周期性探测
            pass

    def get_pacing_rate(self) -> float:
        """获取发送速率"""
        return self.bw_estimator.get_max_bw() * self.pacing_gain

    def _inflight(self) -> float:
        """计算在途数据量"""
        return self.cwnd  # 简化实现
```

---

## 16.8 ECN（显式拥塞通知）

### 16.8.1 ECN 的工作原理

ECN（Explicit Congestion Notification）允许路由器在拥塞时**标记**数据包，而不是丢弃：

```
ECN 工作流程：

1. IP 头部的 ECN 字段（2 位）：
   • 00: 不支持 ECN
   • 01: 支持 ECN (ECT(1))
   • 10: 支持 ECN (ECT(0))
   • 11: 拥塞经历 (CE)

2. TCP 头部的 ECN 标志：
   • CWR: 拥塞窗口减小
   • ECE: ECN-Echo

ECN 交互过程：
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  发送方                                    接收方            │
│     │                                        │              │
│     │  数据包 (ECT=01)                       │              │
│     │ ────────────────────────────────────→  │              │
│     │                                        │              │
│     │        路由器检测到拥塞                  │              │
│     │        标记数据包 (CE=11)               │              │
│     │                                        │              │
│     │  数据包 (CE=11)                        │              │
│     │ ────────────────────────────────────→  │              │
│     │                                        │              │
│     │  ACK (ECE=1)                           │              │
│     │ ←────────────────────────────────────  │              │
│     │                                        │              │
│     │  收到 ECE，降低 cwnd                    │              │
│     │  数据包 (CWR=1)                        │              │
│     │ ────────────────────────────────────→  │              │
│     │                                        │              │
│     │  正常 ACK                               │              │
│     │ ←────────────────────────────────────  │              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 16.8.2 ECN 标记处理器

```c
/* 路由器中的 ECN 标记处理器 */

struct ecn_marker {
    uint32_t queue_threshold;  /* 队列长度阈值 */
    uint32_t mark_probability; /* 标记概率 */
    uint64_t total_packets;    /* 总包数 */
    uint64_t marked_packets;   /* 标记包数 */
};

/* 检查是否需要标记 ECN */
int ecn_should_mark(struct ecn_marker *marker,
                    uint32_t queue_length,
                    uint8_t ecn_field) {
    /* 只标记支持 ECN 的包 */
    if (ecn_field == 0x00)  /* 不支持 ECN */
        return 0;

    /* 队列长度超过阈值时标记 */
    if (queue_length > marker->queue_threshold) {
        /* 概率标记 */
        if (random() % 100 < marker->mark_probability) {
            marker->marked_packets++;
            return 1;  /* 标记为 CE (11) */
        }
    }

    return 0;  /* 不标记 */
}

/* 处理 ECN 标记的 TCP 段 */
void tcp_handle_ecn(struct tcp_pcb *pcb, uint8_t ecn_bits) {
    if (ecn_bits == 0x03) {  /* CE = 11 */
        /* 收到拥塞通知 */
        pcb->ecn_ce_count++;

        /* 降低拥塞窗口 */
        pcb->ssthresh = pcb->cwnd / 2;
        pcb->cwnd = pcb->ssthresh;

        /* 设置 ECE 标志，通知发送方 */
        pcb->snd_ecn_echo = 1;
    }
}
```

<div data-component="ECNVisualizer"></div>

---

## 16.9 主动队列管理（AQM）

### 16.9.1 RED（Random Early Detection）

RED 是最经典的 AQM 算法：

```
RED 算法原理：

┌─────────────────────────────────────────────────────────────┐
│                    RED 算法                                   │
│                                                             │
│  1. 计算平均队列长度                                           │
│     avg = (1 - w) * avg + w * instantaneous_queue_length    │
│     w: 权重因子（通常 0.002）                                │
│                                                             │
│  2. 丢包/标记决策                                             │
│     if avg < min_thresh:                                    │
│         不丢包                                               │
│     elif avg < max_thresh:                                  │
│         以概率 p 丢包                                        │
│         p = max_p * (avg - min_thresh) / (max_thresh - min_thresh)│
│     else:                                                   │
│         丢弃所有包                                           │
│                                                             │
│  丢包概率                                                     │
│  ▲                                                          │
│  │                    ┌─────────────                        │
│  │                   /│                                     │
│  │                  / │                                     │
│  │                 /  │                                     │
│  │                /   │                                     │
│  │               /    │                                     │
│  │              /     │                                     │
│  │             /      │                                     │
│  │            /       │                                     │
│  │           /        │                                     │
│  │──────────/─────────┼───────────────────→                │
│  │     min_thresh     max_thresh         平均队列长度        │
│  │                                                          │
└─────────────────────────────────────────────────────────────┘
```

### 16.9.2 RED 的实现

```python
class REDQueue:
    """RED 队列管理器"""

    def __init__(self, min_thresh: float, max_thresh: float,
                 max_p: float = 0.1, w: float = 0.002):
        self.min_thresh = min_thresh    # 最小阈值
        self.max_thresh = max_thresh    # 最大阈值
        self.max_p = max_p              # 最大丢包概率
        self.w = w                      # 权重因子
        self.avg_queue_len = 0.0        # 平均队列长度
        self.count = -1                 # 自上次丢包以来的包数
        self.last_update_time = 0       # 上次更新时间

    def update_avg_queue(self, current_queue_len: int, current_time: float):
        """更新平均队列长度（指数加权移动平均）"""
        # 计算时间间隔
        time_delta = current_time - self.last_update_time
        if time_delta > 0:
            # 根据时间调整权重
            w = 1 - math.pow(1 - self.w, time_delta)
            self.avg_queue_len = (1 - w) * self.avg_queue_len + \
                                 w * current_queue_len
        self.last_update_time = current_time

    def should_drop(self) -> bool:
        """决定是否丢包"""
        if self.avg_queue_len < self.min_thresh:
            # 低于最小阈值，不丢包
            self.count = -1
            return False

        elif self.avg_queue_len < self.max_thresh:
            # 在阈值之间，概率丢包
            self.count += 1

            # 计算丢包概率
            pb = self.max_p * (self.avg_queue_len - self.min_thresh) / \
                 (self.max_thresh - self.min_thresh)

            # 调整概率（考虑自上次丢包以来的包数）
            pa = pb / (1 - self.count * pb)

            # 随机决定是否丢包
            if random.random() < pa:
                self.count = 0
                return True
            return False

        else:
            # 超过最大阈值，丢弃所有包
            self.count = 0
            return True


# 使用示例
red = REDQueue(min_thresh=10, max_thresh=50, max_p=0.1)

for i in range(1000):
    queue_len = random.randint(0, 100)
    red.update_avg_queue(queue_len, i)

    if red.should_drop():
        print(f"时间 {i}: 丢包 (avg_queue={red.avg_queue_len:.2f})")
```

### 16.9.3 CoDel 和 FQ-PIE

```
CoDel (Controlled Delay)：

核心思想：基于延迟而非队列长度

┌─────────────────────────────────────────────────────────────┐
│  CoDel 算法：                                                │
│  • 目标延迟：5ms                                             │
│  • 间隔时间：100ms                                           │
│  • 如果数据包在队列中等待超过目标延迟，开始丢包               │
│  • 使用"间歇性"丢包，避免连续丢包                             │
│                                                             │
│  优势：                                                      │
│  • 不需要手动调整参数                                        │
│  • 对突发流量更友好                                          │
│  • 延迟更稳定                                                │
└─────────────────────────────────────────────────────────────┘

FQ-PIE (Fair Queuing + PIE)：

核心思想：结合公平队列和 PIE 拥塞控制

┌─────────────────────────────────────────────────────────────┐
│  FQ-PIE 架构：                                               │
│  • 每个流一个队列                                            │
│  • PIE 算法控制总队列延迟                                    │
│  • 公平性由 FQ 保证                                          │
│  • PIE 使用概率丢包控制延迟                                   │
│                                                             │
│  PIE 算法：                                                  │
│  • 基于当前延迟和目标延迟的差异                               │
│  • 动态调整丢包概率                                          │
│  • 比 RED 更稳定                                             │
└─────────────────────────────────────────────────────────────┘
```

<div data-component="AQMComparison"></div>

---

## 16.10 拥塞控制算法对比

### 16.10.1 算法特性对比

| 特性 | Tahoe | Reno | NewReno | CUBIC | BBR |
|------|-------|------|---------|-------|-----|
| 拥塞信号 | 丢包 | 丢包 | 丢包 | 丢包 | 带宽/延迟 |
| 快速恢复 | 无 | 有 | 改进 | 有 | N/A |
| 多丢包处理 | 差 | 差 | 好 | 好 | 好 |
| 高带宽网络 | 差 | 差 | 中 | 好 | 极好 |
| RTT 公平性 | 差 | 差 | 差 | 好 | 好 |
| 缓冲区大小 | 中 | 中 | 中 | 大 | 小 |
| Linux 默认 | 2.0-2.2 | 2.2-2.4 | 2.4-2.6 | 2.6.19+ | 4.9+ |

### 16.10.2 性能特征图示

```
不同算法在不同网络环境下的表现：

1. 低带宽低延迟网络（LAN）：
   所有算法表现相似

2. 高带宽长延迟网络（WAN）：
   Reno: 增长慢，利用率低
   CUBIC: 增长快，利用率高
   BBR: 最优，主动探测带宽

3. 浅缓冲区网络（数据中心）：
   Reno: 频繁丢包，性能差
   CUBIC: 频繁丢包，性能中
   BBR: 避免填满缓冲区，性能好

4. 深缓冲区网络（ISP）：
   Reno: 填满缓冲区，延迟高
   CUBIC: 填满缓冲区，延迟高
   BBR: 避免填满缓冲区，延迟低
```

---

## 本章小结

本章全面介绍了 TCP 拥塞控制技术：

1. **拥塞原因**：网络数据量超过处理能力
2. **AIMD 原理**：加性增/乘性减，保证公平性
3. **TCP Tahoe**：慢启动、拥塞避免、快速重传
4. **TCP Reno**：增加快速恢复
5. **TCP NewReno**：改进多丢包处理
6. **TCP CUBIC**：三次函数，更好的可扩展性
7. **TCP BBR**：基于带宽-延迟模型，主动探测
8. **ECN**：显式拥塞通知，避免丢包
9. **AQM**：RED/CoDel/FQ-PIE，主动队列管理

**核心设计思想**：

- **端到端 vs 网络辅助**：TCP 在端系统实现拥塞控制，ECN 提供网络辅助
- **保守 vs 激进**：Reno/CUBIC 保守（丢包才降速），BBR 主动探测
- **公平性 vs 效率**：AIMD 保证公平，CUBIC/BBR 追求效率

---

> **下一章预告**：第 17 章将介绍 TCP 定时器管理与性能优化。

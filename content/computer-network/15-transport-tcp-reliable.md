# Chapter 15: 传输层 — TCP 可靠传输机制

> **学习目标**：
> - 理解可靠数据传输的基本原理和设计挑战
> - 掌握 TCP 超时重传机制，理解 RTO 的计算方法
> - 理解重复 ACK 和快速重传的工作原理
> - 掌握累积确认与选择确认（SACK）的区别和实现
> - 深入理解 TCP 滑动窗口的发送窗口、接收窗口和可用窗口
> - 掌握流量控制（rwnd）和糊涂窗口综合征（SWS）的解决方案
> - 理解 Nagle 算法、延迟确认和 PSH 标志的设计意图

---

## 15.1 可靠数据传输原理

### 15.1.1 可靠传输的基本问题

在网络通信中，数据可能面临以下问题：

```
网络传输中的错误类型：
┌─────────────────────────────────────────────────────────────┐
│                    网络传输错误                               │
│                                                             │
│  1. 比特错误 (Bit Errors)                                    │
│     ┌─────────────────────────────────────────────────────┐ │
│     │ 原因：电磁干扰、信号衰减、硬件故障                     │ │
│     │ 特征：随机比特翻转                                   │ │
│     │ 检测：校验和、CRC                                    │ │
│     │ 纠正：前向纠错（FEC）或重传                           │ │
│     └─────────────────────────────────────────────────────┘ │
│                                                             │
│  2. 丢包 (Packet Loss)                                       │
│     ┌─────────────────────────────────────────────────────┐ │
│     │ 原因：路由器队列溢出、链路故障、TTL 过期               │ │
│     │ 特征：数据段完全丢失                                 │ │
│     │ 检测：超时、重复 ACK                                 │ │
│     │ 恢复：重传                                           │ │
│     └─────────────────────────────────────────────────────┘ │
│                                                             │
│  3. 乱序 (Out-of-Order Delivery)                             │
│     ┌─────────────────────────────────────────────────────┐ │
│     │ 原因：多路径路由、负载均衡                            │ │
│     │ 特征：数据段到达顺序与发送顺序不同                    │ │
│     │ 检测：序列号                                         │ │
│     │ 恢复：接收方重组                                     │ │
│     └─────────────────────────────────────────────────────┘ │
│                                                             │
│  4. 重复 (Duplicate)                                         │
│     ┌─────────────────────────────────────────────────────┐ │
│     │ 原因：重传机制、网络路由环路                          │ │
│     │ 特征：同一数据段多次到达                             │ │
│     │ 检测：序列号                                         │ │
│     │ 恢复：丢弃重复数据                                   │ │
│     └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 15.1.2 可靠传输协议的演进

```
可靠传输协议的演进（从简单到复杂）：

Rdt 1.0：完全可靠信道
┌─────────────────────────────────────────────────────────────┐
│  假设：信道完全可靠，不会丢包、不会出错、不会乱序            │
│  发送方：直接发送                                            │
│  接收方：直接接收                                            │
│  实现：不需要任何可靠性机制                                   │
└─────────────────────────────────────────────────────────────┘

Rdt 2.0：具有比特错误的信道
┌─────────────────────────────────────────────────────────────┐
│  假设：信道可能翻转比特，但不会丢包                          │
│  机制：校验和 + 确认(ACK) + 否定确认(NAK) + 重传            │
│  问题：ACK/NAK 本身可能损坏                                  │
└─────────────────────────────────────────────────────────────┘

Rdt 2.1：处理损坏的 ACK/NAK
┌─────────────────────────────────────────────────────────────┐
│  机制：序列号（0/1）+ 校验和 + ACK/NAK + 重传               │
│  发送方：等待 ACK 时超时则重传                               │
│  接收方：检测重复序列号，丢弃重复数据                        │
└─────────────────────────────────────────────────────────────┘

Rdt 3.0：具有比特错误和丢包的信道
┌─────────────────────────────────────────────────────────────┐
│  机制：序列号 + 校验和 + ACK + 超时重传                      │
│  这是 TCP 的基本模型                                         │
│  问题：停等协议，信道利用率低                                 │
└─────────────────────────────────────────────────────────────┘

流水线协议（Pipelining）：
┌─────────────────────────────────────────────────────────────┐
│  机制：允许发送方连续发送多个数据段                           │
│  需要：更大的序列号空间、更大的窗口、缓冲区                   │
│  实现：Go-Back-N（GBN）或选择重传（SR）                      │
│  TCP 采用：累积确认 + 选择确认（SACK）的混合方案              │
└─────────────────────────────────────────────────────────────┘
```

<div data-component="ReliableTransferProtocolSimulator"></div>

---

## 15.2 TCP 超时重传机制

### 15.2.1 重传定时器的工作原理

TCP 为每个已发送但未确认的数据段维护一个**重传定时器（Retransmission Timeout, RTO）**：

```
重传定时器工作流程：

发送方                                    接收方
    │                                        │
    │  发送数据段 (seq=1000, len=500)         │
    │  启动重传定时器 (RTO = 1s)              │
    │ ──────────────────────────────────────→ │
    │                                        │
    │     [等待 ACK]                          │
    │     [定时器倒计时...]                    │
    │                                        │
    │     ┌─────────────────────────────┐    │
    │     │ 情况 A: ACK 在 RTO 内到达     │    │
    │     │ → 取消定时器                   │    │
    │     │ → 发送下一个数据段             │    │
    │     └─────────────────────────────┘    │
    │                                        │
    │     ┌─────────────────────────────┐    │
    │     │ 情况 B: RTO 超时              │    │
    │     │ → 重传数据段                   │    │
    │     │ → RTO = RTO * 2 (指数退避)    │    │
    │     │ → 重新启动定时器               │    │
    │     └─────────────────────────────┘    │
    │                                        │
    │  ACK (ack=1500)                        │
    │ ←────────────────────────────────────── │

重传定时器超时时间的选择：
• 太短：不必要的重传，浪费带宽
• 太长：丢包后等待时间过长，降低吞吐量
• 需要根据网络 RTT 动态调整
```

### 15.2.2 RTO 计算：Jacobson 算法

RFC 6298 定义的 RTO 计算算法：

```
RTT 测量与 RTO 计算：

1. 首次测量：
   RTT = 测量到的往返时间
   SRTT = RTT
   RTTVAR = RTT / 2
   RTO = SRTT + 4 * RTTVAR

2. 后续测量：
   RTT = 新测量到的往返时间
   RTTVAR = (1 - β) * RTTVAR + β * |SRTT - RTT|
   SRTT = (1 - α) * SRTT + α * RTT
   RTO = SRTT + max(4 * RTTVAR, 1 tick)

   其中：α = 1/8, β = 1/4

3. RTO 边界：
   RTO = max(RTO_min, min(RTO_max, RTO))
   典型值：RTO_min = 1s, RTO_max = 60s

4. 重传时的指数退避：
   每次超时重传：RTO = RTO * 2
   收到新 ACK 后：RTO 恢复为计算值
```

### 15.2.3 RTO 计算的代码实现

```python
class RTTEstimator:
    """RTT 估计器（Jacobson 算法）"""

    def __init__(self, initial_rto=1.0):
        self.srtt = 0.0      # 平滑 RTT
        self.rttvar = 0.0     # RTT 变化量
        self.rto = initial_rto # 重传超时
        self.first_measure = True

        # 常量
        self.alpha = 1/8      # SRTT 平滑因子
        self.beta = 1/4       # RTTVAR 平滑因子
        self.rto_min = 0.2    # 最小 RTO (200ms)
        self.rto_max = 60.0   # 最大 RTO (60s)
        self.k = 4            # RTTVAR 系数
        self.granularity = 0.01  # 时钟粒度 (10ms)

    def update_rtt(self, measured_rtt: float):
        """更新 RTT 测量值"""
        if self.first_measure:
            # 首次测量
            self.srtt = measured_rtt
            self.rttvar = measured_rtt / 2
            self.first_measure = False
        else:
            # 后续测量
            self.rttvar = (1 - self.beta) * self.rttvar + \
                          self.beta * abs(self.srtt - measured_rtt)
            self.srtt = (1 - self.alpha) * self.srtt + \
                        self.alpha * measured_rtt

        # 计算 RTO
        self.rto = self.srtt + max(self.k * self.rttvar, self.granularity)
        self.rto = max(self.rto_min, min(self.rto_max, self.rto))

        return self.rto

    def on_timeout(self):
        """超时后的指数退避"""
        self.rto = min(self.rto * 2, self.rto_max)
        return self.rto

    def get_rto(self) -> float:
        """获取当前 RTO"""
        return self.rto


# 示例使用
estimator = RTTEstimator()

# 模拟 RTT 测量
rtt_samples = [0.05, 0.06, 0.055, 0.07, 0.05, 0.045, 0.06, 0.1, 0.08]

for i, rtt in enumerate(rtt_samples):
    rto = estimator.update_rtt(rtt)
    print(f"样本 {i+1}: RTT={rtt*1000:.1f}ms, "
          f"SRTT={estimator.srtt*1000:.1f}ms, "
          f"RTTVAR={estimator.rttvar*1000:.1f}ms, "
          f"RTO={rto*1000:.1f}ms")

# 模拟超时
estimator.on_timeout()
print(f"超时后 RTO: {estimator.rto*1000:.1f}ms")
```

### 15.2.4 Karn's 算法

```
Karn's 算法：解决重传 RTT 测量歧义问题

问题场景：
    发送方                                    接收方
         │                                        │
         │  发送 seq=1000                          │
         │  记录发送时间 T1                         │
         │ ──────────────────────────────────────→ │
         │                                        │
         │     [数据包丢失]                         │
         │                                        │
         │  RTO 超时                                │
         │  重传 seq=1000                          │
         │  记录发送时间 T2                         │
         │ ──────────────────────────────────────→ │
         │                                        │
         │  ACK (ack=1500)                         │
         │ ←────────────────────────────────────── │
         │                                        │
         │  问题：这个 ACK 是针对原始包还是重传包？   │
         │  • 如果是原始包: RTT = T_ack - T1        │
         │  • 如果是重传包: RTT = T_ack - T2        │
         │  无法确定！                               │

Karn's 算法规则：
┌─────────────────────────────────────────────────────────────┐
│  1. 不使用重传段的 ACK 来更新 RTT 估计                       │
│  2. 只使用非重传段的 ACK 来更新 RTT                          │
│  3. 超时后 RTO 指数退避                                      │
│  4. 收到新的（非重传的）ACK 后，RTO 恢复为计算值              │
└─────────────────────────────────────────────────────────────┘

TCP 时间戳选项如何解决这个问题：
┌─────────────────────────────────────────────────────────────┐
│  时间戳选项允许精确测量每个 ACK 的 RTT                       │
│  发送方在段中放入发送时间戳 TSval                             │
│  接收方在 ACK 中回显该时间戳 TSecr                           │
│  发送方通过 TSecr 确定 ACK 对应的是哪个发送段                 │
│  → 可以安全地测量重传段的 RTT                                │
└─────────────────────────────────────────────────────────────┘
```

<div data-component="RTTEstimatorVisualizer"></div>

---

## 15.3 重复 ACK 与快速重传

### 15.3.1 重复 ACK 的产生

当接收方收到乱序的数据段时，会发送重复的 ACK：

```
重复 ACK 的产生过程：

发送方                                    接收方
    │                                        │
    │  seq=1000, len=500                     │
    │ ──────────────────────────────────────→ │
    │                                        │  正常接收
    │  ACK (ack=1500)                        │
    │ ←────────────────────────────────────── │
    │                                        │
    │  seq=1500, len=500  [丢失]              │
    │ ───────────────×                        │
    │                                        │
    │  seq=2000, len=500                     │
    │ ──────────────────────────────────────→ │
    │                                        │  收到 seq=2000，但期望 1500
    │  ACK (ack=1500) ← 重复 ACK #1          │  "我期望 1500，不是 2000"
    │ ←────────────────────────────────────── │
    │                                        │
    │  seq=2500, len=500                     │
    │ ──────────────────────────────────────→ │
    │                                        │  收到 seq=2500，仍期望 1500
    │  ACK (ack=1500) ← 重复 ACK #2          │
    │ ←────────────────────────────────────── │
    │                                        │
    │  seq=3000, len=500                     │
    │ ──────────────────────────────────────→ │
    │                                        │  收到 seq=3000，仍期望 1500
    │  ACK (ack=1500) ← 重复 ACK #3          │
    │ ←────────────────────────────────────── │
    │                                        │
    │  收到 3 个重复 ACK → 触发快速重传！      │
    │  立即重传 seq=1500                      │
    │ ──────────────────────────────────────→ │
```

### 15.3.2 快速重传（Fast Retransmit）

```
快速重传机制：

┌─────────────────────────────────────────────────────────────┐
│                    快速重传算法                               │
│                                                             │
│  触发条件：收到 3 个重复 ACK（即总共 4 个相同的 ACK）         │
│                                                             │
│  行为：                                                     │
│  1. 立即重传丢失的数据段                                     │
│  2. 不等待 RTO 超时                                         │
│  3. 减少等待时间，提高吞吐量                                 │
│                                                             │
│  为什么是 3 个重复 ACK？                                     │
│  • 1-2 个重复 ACK 可能是网络乱序引起                         │
│  • 3 个重复 ACK 很可能是真正的丢包                           │
│  • 这是一个经验阈值，在检测速度和误判率之间取得平衡           │
│                                                             │
│  快速重传 vs 超时重传：                                      │
│  ┌───────────────┬──────────────┬─────────────────────┐     │
│  │               │ 超时重传      │ 快速重传             │     │
│  ├───────────────┼──────────────┼─────────────────────┤     │
│  │ 触发条件       │ RTO 超时      │ 3 个重复 ACK         │     │
│  │ 等待时间       │ RTO (数百ms) │ 3 个 RTT (更快)      │     │
│  │ 拥塞窗口调整   │ cwnd=1 MSS   │ 取决于具体算法        │     │
│  │ 适用场景       │ 严重丢包      │ 轻度丢包             │     │
│  └───────────────┴──────────────┴─────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## 15.4 累积确认与选择确认（SACK）

### 15.4.1 累积确认

TCP 默认使用**累积确认（Cumulative ACK）**：

```
累积确认的工作方式：

发送方                                    接收方
    │                                        │
    │  seq=1000, len=500                     │
    │ ──────────────────────────────────────→ │
    │                                        │
    │  seq=1500, len=500  [丢失]              │
    │ ───────────────×                        │
    │                                        │
    │  seq=2000, len=500                     │
    │ ──────────────────────────────────────→ │
    │                                        │
    │  seq=2500, len=500                     │
    │ ──────────────────────────────────────→ │
    │                                        │
    │  ACK (ack=1500)                        │
    │  ← 表示"我已收到 1500 之前的所有字节"    │
    │  ← 但没有说明 2000 和 2500 是否收到     │
    │ ←────────────────────────────────────── │

累积确认的问题：
发送方不知道哪些数据已经到达，哪些丢失
只能重传从丢失点开始的所有数据（Go-Back-N）
或者保守地只重传可能丢失的数据
```

### 15.4.2 选择确认（SACK）

SACK（Selective Acknowledgment）允许接收方告诉发送方**具体收到了哪些数据块**：

```
SACK 选项格式：
┌──────────────────────────────────────────────────────────────┐
│  Kind=5 │ Length │ Left Edge 1 │ Right Edge 1 │ ...          │
│  (1B)   │ (1B)   │   (4B)      │   (4B)       │              │
└──────────────────────────────────────────────────────────────┘

SACK 工作示例：

发送方                                    接收方
    │                                        │
    │  发送段 1-5（每段 500 字节）             │
    │  seq=1000 (段1)  ✓ 到达                 │
    │  seq=1500 (段2)  ✗ 丢失                 │
    │  seq=2000 (段3)  ✓ 到达                 │
    │  seq=2500 (段4)  ✓ 到达                 │
    │  seq=3000 (段5)  ✗ 丢失                 │
    │                                        │
    │  ACK (ack=1500, SACK=[2000-3000])      │
    │ ←────────────────────────────────────── │
    │                                        │
    │  发送方现在知道：                         │
    │  • 段 1 (1000-1499) 已收到              │
    │  • 段 2 (1500-1999) 丢失 → 需要重传     │
    │  • 段 3-4 (2000-2999) 已收到            │
    │  • 段 5 (3000-3499) 未知                │
    │                                        │
    │  只需重传段 2，无需重传段 3-4            │
```

### 15.4.3 SACK 处理与重传决策引擎

```python
class SACKProcessor:
    """SACK 信息处理器"""

    def __init__(self):
        self.sacked_blocks = []  # 已确认的 SACK 块
        self.lost_blocks = []    # 推断的丢失块

    def process_sack_option(self, ack_num: int, sack_blocks: list):
        """
        处理收到的 SACK 选项
        参数：
            ack_num: 累积确认号
            sack_blocks: [(left1, right1), (left2, right2), ...]
        """
        self.sacked_blocks = sack_blocks
        self.lost_blocks = self._infer_lost_blocks(ack_num, sack_blocks)

    def _infer_lost_blocks(self, cum_ack: int, sack_blocks: list) -> list:
        """
        根据 SACK 块推断可能丢失的数据块
        """
        if not sack_blocks:
            return []

        lost = []
        # 累积确认点到第一个 SACK 块之间可能有丢失
        if cum_ack < sack_blocks[0][0]:
            lost.append((cum_ack, sack_blocks[0][0]))

        # SACK 块之间可能有丢失
        for i in range(len(sack_blocks) - 1):
            gap_start = sack_blocks[i][1]
            gap_end = sack_blocks[i + 1][0]
            if gap_start < gap_end:
                lost.append((gap_start, gap_end))

        return lost

    def get_retransmit_segments(self, mss: int) -> list:
        """
        获取需要重传的段列表
        """
        segments = []
        for left, right in self.lost_blocks:
            offset = left
            while offset < right:
                length = min(mss, right - offset)
                segments.append((offset, length))
                offset += length
        return segments


class RetransmissionDecisionEngine:
    """重传决策引擎"""

    def __init__(self, mss=1460):
        self.mss = mss
        self.sack_processor = SACKProcessor()
        self.retransmit_queue = []  # 重传队列
        self.sent_segments = {}     # 已发送段记录 {seq: (data, timestamp, retransmitted)}

    def on_segment_sent(self, seq: int, data: bytes):
        """记录已发送的段"""
        self.sent_segments[seq] = {
            "data": data,
            "timestamp": time.time(),
            "retransmitted": False,
            "retransmit_count": 0
        }

    def on_ack_received(self, ack_num: int, sack_blocks: list = None):
        """处理收到的 ACK"""
        # 1. 清除已确认的段
        self._remove_acked_segments(ack_num)

        # 2. 处理 SACK 信息
        if sack_blocks:
            self.sack_processor.process_sack_option(ack_num, sack_blocks)
            lost_segments = self.sack_processor.get_retransmit_segments(self.mss)
            for seq, length in lost_segments:
                self._queue_retransmit(seq)

    def on_timeout(self, seq: int):
        """超时重传"""
        if seq in self.sent_segments:
            self._queue_retransmit(seq)
            self.sent_segments[seq]["retransmit_count"] += 1

    def on_duplicate_ack(self, ack_num: int, dup_count: int):
        """处理重复 ACK"""
        if dup_count >= 3:
            # 快速重传
            self._queue_retransmit(ack_num)

    def _remove_acked_segments(self, ack_num: int):
        """移除已确认的段"""
        to_remove = [seq for seq in self.sent_segments if seq < ack_num]
        for seq in to_remove:
            del self.sent_segments[seq]

    def _queue_retransmit(self, seq: int):
        """将段加入重传队列"""
        if seq in self.sent_segments and not self.sent_segments[seq]["retransmitted"]:
            self.retransmit_queue.append(seq)
            self.sent_segments[seq]["retransmitted"] = True
            self.sent_segments[seq]["retransmit_count"] += 1

    def get_next_retransmit(self) -> tuple:
        """获取下一个要重传的段"""
        if self.retransmit_queue:
            seq = self.retransmit_queue.pop(0)
            if seq in self.sent_segments:
                return seq, self.sent_segments[seq]["data"]
        return None, None
```

<div data-component="SACKVisualizer"></div>

---

## 15.5 TCP 滑动窗口

### 15.5.1 发送窗口

TCP 使用**滑动窗口**机制实现流量控制和可靠传输：

```
发送窗口结构：

序列号空间：
[─────────────────────────────────────────────────────────────────]
0                                                            2^32-1

发送窗口：
        ┌─────────────────────────────────────────────────┐
        │            发送窗口 (SND.WND)                    │
        │                                                 │
        ├───┬───────────────┬───────────────┬─────────────┤
        │已确认│  已发送未确认  │   可发送      │  不可发送    │
        │     │  (In-Flight)  │  (Usable)     │  (Future)   │
        ├───┼───────────────┼───────────────┼─────────────┤
        │   │               │               │             │
        │ SND.UNA           │ SND.NXT       │ SND.UNA+SND.WND
        │                   │               │             │
        └───┴───────────────┴───────────────┴─────────────┘

SND.UNA: 最早未确认的序列号（Send Unacknowledged）
SND.NXT: 下一个要发送的序列号（Send Next）
SND.WND: 发送窗口大小（由接收方通告）

可用窗口大小 = SND.UNA + SND.WND - SND.NXT
```

### 15.5.2 接收窗口

```
接收窗口结构：

        ┌─────────────────────────────────────────────────┐
        │            接收窗口 (RCV.WND)                    │
        │                                                 │
        ├───┬───────────────┬───────────────┬─────────────┤
        │已确认│  已接收未确认  │   可接收      │  不可接收    │
        │     │  (Buffered)   │  (Free)       │  (Future)   │
        ├───┼───────────────┼───────────────┼─────────────┤
        │   │               │               │             │
        │ RCV.NXT           │ RCV.NXT+RCV.WND             │
        │                   │               │             │
        └───┴───────────────┴───────────────┴─────────────┘

RCV.NXT: 期望收到的下一个序列号（Receive Next）
RCV.WND: 接收窗口大小（接收缓冲区剩余空间）

接收窗口通告：
发送方通过 TCP 段的 Window 字段通告接收窗口大小
Window 字段为 16 位，配合窗口缩放选项可支持更大的窗口
```

### 15.5.3 滑动窗口的发送/接收逻辑

```c
/* TCP 滑动窗口控制器 */

struct tcp_window {
    /* 发送窗口 */
    uint32_t snd_una;      /* 最早未确认的序列号 */
    uint32_t snd_nxt;      /* 下一个要发送的序列号 */
    uint32_t snd_wnd;      /* 发送窗口大小 */
    uint32_t snd_wl1;      /* 上次窗口更新的序列号 */
    uint32_t snd_wl2;      /* 上次窗口更新的确认号 */

    /* 接收窗口 */
    uint32_t rcv_nxt;      /* 期望收到的下一个序列号 */
    uint32_t rcv_wnd;      /* 接收窗口大小 */

    /* 拥塞窗口 */
    uint32_t cwnd;         /* 拥塞窗口 */
    uint32_t ssthresh;     /* 慢启动阈值 */
};

/* 计算可用发送窗口 */
uint32_t tcp_usable_window(struct tcp_window *win) {
    uint32_t effective_wnd = min(win->snd_wnd, win->cwnd);
    uint32_t in_flight = win->snd_nxt - win->snd_una;

    if (in_flight >= effective_wnd)
        return 0;  /* 窗口已满，不能发送 */

    return effective_wnd - in_flight;
}

/* 检查是否可以发送数据 */
int tcp_can_send(struct tcp_window *win, uint32_t data_len) {
    uint32_t usable = tcp_usable_window(win);
    return data_len <= usable;
}

/* 处理 ACK，滑动窗口前移 */
void tcp_process_ack(struct tcp_window *win, uint32_t ack_num,
                     uint32_t new_wnd) {
    /* 检查 ACK 是否在有效范围内 */
    if (ack_num > win->snd_una && ack_num <= win->snd_nxt) {
        /* ACK 确认了新数据，滑动窗口前移 */
        win->snd_una = ack_num;
    }

    /* 更新窗口大小（防止窗口缩小导致的死锁） */
    if (win->snd_wl1 == 0 ||
        ack_num > win->snd_wl2 ||
        (ack_num == win->snd_wl2 && new_wnd > win->snd_wnd)) {
        win->snd_wnd = new_wnd;
        win->snd_wl1 = 0;  /* 简化处理 */
        win->snd_wl2 = ack_num;
    }
}

/* 发送数据时更新窗口 */
void tcp_send_data(struct tcp_window *win, uint32_t data_len) {
    win->snd_nxt += data_len;
}

/* 接收数据时更新接收窗口 */
void tcp_receive_data(struct tcp_window *win, uint32_t seq,
                      uint32_t data_len) {
    if (seq == win->rcv_nxt) {
        /* 按序到达 */
        win->rcv_nxt += data_len;
        win->rcv_wnd -= data_len;
    }
    /* 乱序到达：暂存，等待缺失数据 */
}
```

<div data-component="SlidingWindowSimulator"></div>

---

## 15.6 流量控制

### 15.6.1 基于窗口的流量控制

TCP 使用**接收窗口（rwnd）**实现流量控制：

```
流量控制过程：

发送方 (快)                                接收方 (慢)
    │                                        │
    │  rwnd=4096 (接收方通告窗口)              │
    │                                        │
    │  发送 1000 字节                          │
    │ ──────────────────────────────────────→ │
    │                                        │
    │  发送 1000 字节                          │
    │ ──────────────────────────────────────→ │
    │                                        │
    │  发送 1000 字节                          │
    │ ──────────────────────────────────────→ │
    │                                        │
    │  发送 1000 字节                          │
    │ ──────────────────────────────────────→ │
    │                                        │
    │  ACK (ack=4000, rwnd=1096)             │
    │  ← 窗口缩小：应用读取了部分数据           │
    │ ←────────────────────────────────────── │
    │                                        │
    │  发送 1000 字节                          │
    │ ──────────────────────────────────────→ │
    │                                        │
    │  ACK (ack=5000, rwnd=96)               │
    │  ← 窗口很小，发送方应该停止发送           │
    │ ←────────────────────────────────────── │
    │                                        │
    │  [发送方暂停，等待窗口更新]               │
    │                                        │
    │  ACK (ack=5000, rwnd=4096)             │
    │  ← 应用读取了数据，窗口重新打开           │
    │ ←────────────────────────────────────── │
    │                                        │
    │  恢复发送                                │
    │ ──────────────────────────────────────→ │
```

### 15.6.2 糊涂窗口综合征（Silly Window Syndrome）

```
糊涂窗口综合征（SWS）：

问题描述：
当接收方每次只读取少量数据时，会通告很小的窗口
发送方如果发送很小的数据段，会导致：
• 头部开销比例过大（20 字节头部 + 1 字节数据）
• 大量小段增加网络和处理开销
• 效率极低

示例：
┌─────────────────────────────────────────────────────────────┐
│  MSS = 1460 字节                                             │
│  接收方每次只读取 1 字节                                      │
│  发送方发送 1 字节数据段                                      │
│  效率：1 / (20 + 1) = 4.8%                                   │
│  头部开销：20 / 21 = 95.2%                                   │
└─────────────────────────────────────────────────────────────┘

解决方案：

1. 接收方策略（Clark's 方案）：
   当可用窗口 < MSS 时，通告窗口为 0
   直到可用窗口 >= MSS 或缓冲区有 50% 空间时才通告

2. 发送方策略（Nagle 算法，下一节详述）：
   限制发送小数据段
```

### 15.6.3 接收缓冲区管理器

```python
class TCPReceiveBuffer:
    """TCP 接收缓冲区管理器"""

    def __init__(self, capacity: int = 65536):
        self.capacity = capacity      # 缓冲区总容量
        self.buffer = bytearray(capacity)
        self.write_pos = 0            # 写指针
        self.read_pos = 0             # 读指针
        self.used = 0                 # 已使用空间
        self.rcv_nxt = 0              # 期望的下一个序列号
        self.out_of_order = {}        # 乱序数据缓存 {seq: data}

    @property
    def available_space(self) -> int:
        """可用接收空间"""
        return self.capacity - self.used

    @property
    def window_size(self) -> int:
        """通告窗口大小"""
        # 应用 Clark's 策略：窗口太小时通告为 0
        if self.available_space < self.mss:
            return 0
        return self.available_space

    def receive_segment(self, seq: int, data: bytes) -> list:
        """
        接收数据段，返回按序交付的数据列表
        """
        mss = len(data)
        delivered = []

        if seq == self.rcv_nxt:
            # 按序到达，直接写入缓冲区
            self._write_to_buffer(data)
            self.rcv_nxt += len(data)
            delivered.append(data)

            # 检查是否有乱序数据可以交付
            while self.rcv_nxt in self.out_of_order:
                ooo_data = self.out_of_order.pop(self.rcv_nxt)
                self._write_to_buffer(ooo_data)
                self.rcv_nxt += len(ooo_data)
                delivered.append(ooo_data)

        elif seq > self.rcv_nxt:
            # 乱序到达，缓存
            self.out_of_order[seq] = data

        else:
            # 重复数据，丢弃
            pass

        return delivered

    def read_data(self, length: int) -> bytes:
        """应用层读取数据"""
        to_read = min(length, self.used)
        result = bytes(self.buffer[self.read_pos:self.read_pos + to_read])
        self.read_pos = (self.read_pos + to_read) % self.capacity
        self.used -= to_read
        return result

    def _write_to_buffer(self, data: bytes):
        """写入数据到缓冲区"""
        for byte in data:
            self.buffer[self.write_pos] = byte
            self.write_pos = (self.write_pos + 1) % self.capacity
        self.used += len(data)
```

---

## 15.7 Nagle 算法与延迟确认

### 15.7.1 Nagle 算法

Nagle 算法是 TCP 发送方减少小数据段的策略：

```
Nagle 算法规则：
┌─────────────────────────────────────────────────────────────┐
│  当有数据要发送时：                                           │
│                                                             │
│  如果满足以下任一条件，立即发送：                               │
│  1. 数据量 >= MSS（可以发送完整的段）                          │
│  2. 没有未确认的数据（可以立即发送）                           │
│  3. 收到了对方的 ACK（窗口打开）                               │
│                                                             │
│  否则：                                                     │
│  将数据缓冲，等待上述条件满足后再发送                          │
│                                                             │
│  效果：                                                     │
│  • 减少网络中的小数据段数量                                   │
│  • 提高带宽利用率                                            │
│  • 对交互式应用（如 SSH、游戏）增加延迟                       │
└─────────────────────────────────────────────────────────────┘

Nagle 算法示例：

应用发送：    A(1字节) B(1字节) C(1字节) D(1字节)

不使用 Nagle：4 个小段
  [A] [B] [C] [D]  → 4 次发送，每次 21 字节（20 头 + 1 数据）

使用 Nagle：
  发送 [A]，等待 ACK
  收到 ACK 后，合并 BCD
  发送 [BCD]  → 2 次发送

带宽节省：(4*21 - 2*22) / (4*21) = 47.6%
```

### 15.7.2 延迟确认（Delayed ACK）

接收方的延迟确认策略：

```
延迟确认的工作方式：

发送方                                    接收方
    │                                        │
    │  数据段 A                               │
    │ ──────────────────────────────────────→ │
    │                                        │
    │                                        │  启动延迟确认定时器（200ms）
    │                                        │
    │  数据段 B                               │
    │ ──────────────────────────────────────→ │
    │                                        │
    │  ACK (ack=A+B)                         │  合并确认 A 和 B
    │ ←────────────────────────────────────── │
    │                                        │

延迟确认的目的：
1. 减少 ACK 数量（每个 ACK 也是 40 字节的开销）
2. 如果接收方有数据要发送，可以将 ACK 捎带在数据段中
3. 给应用层时间处理数据，可能产生响应数据

延迟确认的问题：
与 Nagle 算法配合时可能导致延迟增加
解决：TCP_NODELAY 选项禁用 Nagle 算法
```

### 15.7.3 PSH 标志

```
PSH（Push）标志的作用：

┌─────────────────────────────────────────────────────────────┐
│  发送方设置 PSH：                                             │
│  • 告诉接收方"立即将数据交付应用层"                            │
│  • 不要等待缓冲区满或延迟确认定时器超时                        │
│                                                             │
│  典型场景：                                                   │
│  • 交互式应用（Telnet、SSH）每次按键都设置 PSH                │
│  • HTTP 请求/响应的最后一个段设置 PSH                         │
│  • 实时流媒体每个帧的最后一个段设置 PSH                       │
│                                                             │
│  接收方收到 PSH：                                             │
│  • 立即将接收缓冲区中的数据交付应用层                          │
│  • 不等待更多数据                                            │
│  • 立即发送 ACK（不使用延迟确认）                              │
└─────────────────────────────────────────────────────────────┘
```

<div data-component="NagleAlgorithmDemo"></div>

---

## 15.8 重传定时器管理

### 15.8.1 定时器管理的复杂性

```
TCP 重传定时器管理的挑战：

1. 一个连接可能有多个未确认的段
   ┌─────────────────────────────────────────────────────────────┐
   │  发送段 1, 2, 3, 4, 5                                       │
   │  段 2 丢失，段 1, 3, 4, 5 正常到达                           │
   │  应该为段 2 启动定时器，还是为所有段启动？                     │
   │  → TCP 通常只维护一个重传定时器                              │
   └─────────────────────────────────────────────────────────────┘

2. 定时器超时时，应该重传哪个段？
   ┌─────────────────────────────────────────────────────────────┐
   │  传统做法：重传最早的未确认段（SND.UNA 指向的段）             │
   │  SACK 做法：重传 SACK 推断的丢失段                           │
   └─────────────────────────────────────────────────────────────┘

3. 收到 ACK 后如何处理定时器？
   ┌─────────────────────────────────────────────────────────────┐
   │  如果还有未确认的段：重启定时器                              │
   │  如果所有段都已确认：取消定时器                              │
   │  部分确认：更新定时器的起始序列号                             │
   └─────────────────────────────────────────────────────────────┘
```

### 15.8.2 重传定时器的硬件实现

```c
/* TCP 重传定时器的管理 */

struct tcp_retransmit_timer {
    uint32_t rto;           /* 当前 RTO（毫秒） */
    uint32_t seq;           /* 定时器对应的序列号 */
    uint32_t fire_time;     /* 触发时间 */
    int      active;        /* 定时器是否激活 */
    int      retransmit_count; /* 重传次数 */
};

/* 启动重传定时器 */
void tcp_timer_start(struct tcp_retransmit_timer *timer,
                     uint32_t seq, uint32_t rto) {
    timer->seq = seq;
    timer->rto = rto;
    timer->fire_time = current_time_ms() + rto;
    timer->active = 1;
    timer->retransmit_count = 0;
}

/* 停止重传定时器 */
void tcp_timer_stop(struct tcp_retransmit_timer *timer) {
    timer->active = 0;
}

/* 检查定时器是否超时 */
int tcp_timer_expired(struct tcp_retransmit_timer *timer) {
    if (!timer->active)
        return 0;
    return current_time_ms() >= timer->fire_time;
}

/* 超时处理 */
void tcp_timer_on_timeout(struct tcp_retransmit_timer *timer,
                          struct tcp_pcb *pcb) {
    timer->retransmit_count++;

    /* 指数退避 */
    timer->rto = min(timer->rto * 2, TCP_RTO_MAX);

    /* 重传最早的未确认段 */
    tcp_retransmit_segment(pcb, timer->seq);

    /* 重启定时器 */
    tcp_timer_start(timer, timer->seq, timer->rto);

    /* 超过最大重传次数，放弃连接 */
    if (timer->retransmit_count >= TCP_MAX_RETRIES) {
        pcb->state = TCP_CLOSED;
        tcp_notify_error(pcb, ETIMEDOUT);
    }
}
```

---

## 15.9 可靠传输的性能分析

### 15.9.1 吞吐量分析

```
TCP 吞吐量公式：

理想情况（无丢包）：
Throughput = Window Size / RTT

示例：
Window Size = 65535 bytes (无窗口缩放)
RTT = 50ms
Throughput = 65535 / 0.05 = 1,310,700 bytes/s ≈ 10.5 Mbps

考虑丢包的吞吐量（Mathis 公式）：
Throughput ≈ (MSS / RTT) * (1 / √p)

其中 p = 丢包率

示例：
MSS = 1460 bytes
RTT = 50ms
p = 0.01 (1% 丢包率)
Throughput ≈ (1460 / 0.05) * (1 / √0.01)
           = 29200 * 10
           = 292,000 bytes/s ≈ 2.3 Mbps

结论：
• RTT 越小，吞吐量越高
• 丢包率对吞吐量影响很大（1% 丢包导致吞吐量下降 ~10 倍）
• 这解释了为什么 TCP 在高延迟/高丢包网络中性能差
```

### 15.9.2 延迟分析

```
TCP 数据传输延迟组成：

总延迟 = 传播延迟 + 传输延迟 + 处理延迟 + 排队延迟

1. 传播延迟 (Propagation Delay)
   = 距离 / 光速
   • 跨大西洋：~30ms
   • 跨太平洋：~60ms
   • 地球同步卫星：~250ms

2. 传输延迟 (Transmission Delay)
   = 数据量 / 带宽
   • 1MB 数据 / 100Mbps = 80ms

3. 处理延迟 (Processing Delay)
   • TCP 头部解析：~10μs
   • 校验和计算：~5μs
   • 状态机处理：~20μs

4. 排队延迟 (Queuing Delay)
   • 取决于网络拥塞程度
   • 轻负载：~0
   • 重负载：数百 ms

TCP 特有的延迟：
• 三次握手：+1 RTT
• 慢启动：初始几个 RTT 吞吐量低
• 丢包重传：+RTO
```

---

## 本章小结

本章深入介绍了 TCP 的可靠传输机制：

1. **可靠传输原理**：处理比特错误、丢包、乱序、重复
2. **超时重传**：RTO 动态计算（Jacobson 算法），Karn's 算法
3. **快速重传**：3 个重复 ACK 触发，减少等待时间
4. **累积确认 vs SACK**：SACK 提供更精确的丢失信息
5. **滑动窗口**：发送窗口、接收窗口、可用窗口
6. **流量控制**：基于 rwnd 的接收方驱动流控
7. **Nagle 算法与延迟确认**：减少小数据段和 ACK 数量

**核心设计思想**：

- **端到端可靠性**：TCP 在端系统实现可靠性，中间节点不参与
- **自适应机制**：RTO、窗口大小根据网络状况动态调整
- **效率与正确性平衡**：SACK、快速重传在保证可靠性的前提下提高效率

---

> **下一章预告**：第 16 章将介绍 TCP 拥塞控制，包括 AIMD、Tahoe、Reno、CUBIC 和 BBR。

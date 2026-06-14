# Chapter 5: 数据链路层 — 滑动窗口协议

> **学习目标**：
> - 理解停等协议的工作原理，掌握超时重传与序号机制，分析其在高延迟信道下的效率瓶颈
> - 掌握滑动窗口协议的核心思想——流水线（Pipelining），理解发送窗口与接收窗口的协调机制
> - 深入掌握回退N帧协议（GBN）的发送方与接收方行为、窗口约束 $N \leq 2^m - 1$ 的证明与反例
> - 深入掌握选择重传协议（SR）的缓存机制、窗口约束 $W_{send} + W_{recv} \leq 2^m$ 的证明与序号歧义分析
> - 能够进行链路利用率的定量计算，理解带宽延迟积对协议设计的影响
> - 理解捎带确认（Piggybacking）与 Nagle 算法的基本原理
> - 了解协议验证方法（FSM、Petri 网）与实际链路层协议（HDLC、PPP）中的滑动窗口应用

---

## 5.1 停等协议（Stop-and-Wait）

### 5.1.1 无限制的简单协议

最简单的可靠传输协议是**停等协议**（Stop-and-Wait Protocol）。其核心思想可以用一个日常生活中的类比来理解：你给朋友发短信，朋友回复"收到了"，然后你才发下一条。这种"一问一答"的模式虽然简单，但在网络通信中是理解所有更复杂协议的基础。

**工作流程**：

1. **发送方**：从网络层获取一个数据包，封装成帧，发送到信道上，然后**停下来等待**
2. **接收方**：正确接收帧后，向发送方返回一个**确认帧（ACK）**
3. **发送方**：收到 ACK 后，才从网络层获取下一个数据包，重复上述过程

这种协议假设信道是**理想的**——既不会丢失帧，也不会产生比特错误。在这种理想条件下，协议的 FSM（有限状态机）描述如下：

**发送方 FSM**：

```
状态 S0（就绪）：
  ── 从网络层获取数据包
  ── 封装为帧
  ── 发送帧
  ── 进入状态 S1（等待ACK）

状态 S1（等待ACK）：
  ── 收到 ACK → 返回状态 S0
  ── （理想信道，不会超时）
```

**接收方 FSM**：

```
状态 R0（就绪）：
  ── 收到帧
  ── 提取数据包交给网络层
  ── 发送 ACK
  ── 返回状态 R0
```

### 5.1.2 有噪声信道的停等协议

现实中的信道是有噪声的，帧可能在传输过程中被损坏或丢失。停等协议必须处理以下异常情况：

#### **帧丢失（Frame Loss）**

发送方发出的帧在信道中丢失，接收方根本没有收到。此时发送方会永远等待 ACK——这就是**死锁（Deadlock）**。

**解决方案：超时重传（Timeout and Retransmit）**

发送方在发送帧的同时启动一个**计时器（Timer）**。如果在预设的**超时时间（Timeout）**内没有收到 ACK，发送方认为帧已丢失，自动重传该帧。

```
发送方行为：
  发送帧 → 启动计时器
  ┌─ 收到 ACK → 取消计时器 → 发送下一帧
  └─ 超时（Timeout）→ 重传同一帧 → 重启计时器
```

#### **ACK 丢失（ACK Loss）**

接收方正确收到了帧并发送了 ACK，但 ACK 在返回途中丢失。发送方超时后重传同一帧，但接收方已经处理过这个帧了。

这就引出了一个关键问题：**接收方如何区分这是一个新帧还是重传的旧帧？**

#### **序号（Seq）与确认号（ACK Num）的作用**

为了解决重复帧的问题，我们需要给每个帧加上**序号（Sequence Number）**：

- **发送方**：在帧头中填入当前帧的序号 `Seq`
- **接收方**：检查收到帧的序号，如果与期望的序号不同，则说明是重复帧，丢弃但重发 ACK

对于停等协议，**1 比特序号（0 和 1 交替）就足够了**。为什么？

因为在任何时刻，信道上最多只有两个帧：一个正在传输的数据帧和一个正在返回的 ACK。发送方交替使用序号 0 和 1，接收方只需检查收到的序号是否等于期望值。

```
正常情况：
  发送方：发送 Seq=0 → 接收方收到，发送 ACK=1 → 发送方收到 ACK
  发送方：发送 Seq=1 → 接收方收到，发送 ACK=0 → 发送方收到 ACK

帧丢失：
  发送方：发送 Seq=0 → [帧丢失] → 超时 → 重传 Seq=0
  接收方收到 Seq=0 → 发送 ACK=1 → 发送方收到 ACK=1（表示确认了Seq=0）

ACK丢失：
  发送方：发送 Seq=0 → 接收方收到 → 发送 ACK=1 → [ACK丢失]
  发送方超时 → 重传 Seq=0 → 接收方发现 Seq=0 已处理过
  接收方：丢弃数据帧，但仍然重发 ACK=1 → 发送方收到 ACK=1
```

**为什么 1 比特序号就够了？** 关键观察：当发送方发送 Seq=0 的帧并等待时，它不可能收到一个"旧的" ACK=0，因为上一个使用 Seq=0 的帧早已被确认。因此接收方只需区分"当前帧"和"上一帧"，1 比特足以。

### 5.1.3 效率分析

停等协议的最大问题在于**效率低下**。发送方大部分时间都在等待，而不是在发送数据。

**信道利用率的计算**：

定义以下时间参数：
- $T_{frame}$：发送一个帧所需的时间 = 帧长度 / 信道带宽 = $L / R$
- $T_{prop}$：单程传播延迟
- $T_{ack}$：发送一个 ACK 帧所需的时间（通常很小，可忽略）
- $T_{proc}$：接收方的处理时间（通常很小，可忽略）

一个完整的发送-接收周期时间为：

$$T_{total} = T_{frame} + T_{prop} + T_{ack} + T_{prop} + T_{proc} \approx T_{frame} + 2 \cdot T_{prop}$$

信道利用率（发送方利用率）为：

$$U = \frac{T_{frame}}{T_{total}} = \frac{L/R}{L/R + 2 \cdot T_{prop}}$$

**与管道（Pipeline）的类比**：

想象一根很长的水管，你在一端注入水（数据帧），水需要一段时间才能流到另一端。在停等协议中，你注入一滴水后就停下来等另一端的水龙头滴出一滴水（ACK）回来，然后才注入下一滴。当水管很长（传播延迟大）时，你绝大部分时间都在等待，水管大部分时间是空的——这就是效率低下的本质。

**数值例子**：

假设链路带宽 $R = 1 \text{ Gbps}$，传播延迟 $T_{prop} = 50 \text{ ms}$，帧长度 $L = 1500 \text{ bytes}$。

$$T_{frame} = \frac{1500 \times 8}{1 \times 10^9} = 12 \text{ μs}$$

$$U = \frac{12 \text{ μs}}{12 \text{ μs} + 100 \text{ ms}} \approx \frac{12}{100012} \approx 0.012\%$$

信道利用率仅为 **0.012%**！这意味着 99.988% 的时间信道是空闲的。这就像一条 10 车道的高速公路，但每次只允许一辆车通过，等它到达目的地后才放行下一辆。

这就是我们需要**滑动窗口协议**的根本原因——通过允许多个帧同时"在管道中"传输来提高信道利用率。

<div data-component="StopAndWaitSimulator"></div>

---

## 5.2 滑动窗口协议基本概念

### 5.2.1 流水线（Pipelining）的思想

停等协议的效率问题根源在于：发送方在等待 ACK 时，信道是空闲的。**流水线**的核心思想是：**不等前一帧的 ACK，就发送后续的帧**。

类比工厂流水线：如果每个工人都要等前一个工人完成整件产品才开始工作，效率极低。正确的做法是让每个工人专注于自己的工序，产品在流水线上连续流动。

在网络通信中，这意味着：
- 发送方可以连续发送多个帧而不需要等待 ACK
- 这些帧像流水线上的产品一样同时在信道中传输
- ACK 也在"反向流水线"中返回

为了实现流水线，我们需要解决两个关键问题：
1. **发送方如何知道哪些帧已被确认？** → 需要序号和窗口机制
2. **出错时如何处理？** → 需要不同的重传策略（GBN 或 SR）

### 5.2.2 发送窗口与接收窗口

#### **发送窗口（Send Window）**

发送窗口定义了发送方**可以发送但尚未确认的帧的范围**。窗口由两个边界定义：

```
发送窗口：[send_base, send_base + N - 1]
  ── send_base：窗口的下界，最早未确认帧的序号
  ── N：窗口大小，允许同时在管道中的最大帧数
  ── send_base + N - 1：窗口的上界
```

- 序号在窗口内的帧可以被发送（如果尚未发送）
- 序号小于 `send_base` 的帧已被确认，不需要再考虑
- 序号大于等于 `send_base + N` 的帧不能发送（窗口未扩展到那里）

当收到 ACK 确认了 `send_base` 帧时，窗口**向前滑动**：

```
收到 ACK(确认 send_base)：
  send_base = send_base + 1
  窗口右移一格
```

#### **接收窗口（Receive Window）**

接收窗口定义了接收方**愿意接受的帧的序号范围**：

```
接收窗口：[recv_base, recv_base + W - 1]
  ── recv_base：期望接收的下一个按序帧的序号
  ── W：接收窗口大小
```

- 序号在接收窗口内的帧被接受并缓存
- 序号等于 `recv_base` 的帧被交付给上层，窗口前移
- 序号在窗口外的帧被丢弃

### 5.2.3 窗口大小的意义

| 协议 | 发送窗口 $N$ | 接收窗口 $W$ | 说明 |
|------|-------------|-------------|------|
| 停等协议 | 1 | 1 | 最简单，效率最低 |
| GBN | $N > 1$ | 1 | 接收方只接受按序帧 |
| SR | $N > 1$ | $W > 1$ | 接收方缓存乱序帧 |

窗口大小直接决定了协议的效率：窗口越大，允许在管道中的帧越多，信道利用率越高。但窗口也不能无限大，受限于序号空间和缓冲区。

### 5.2.4 序号空间与窗口的关系

序号用 $m$ 比特表示，序号空间为 $[0, 2^m - 1]$，是**循环的**——序号 $2^m - 1$ 之后回到 $0$。

窗口大小和序号空间之间有严格的约束关系，这是为了避免**序号歧义（Ambiguity）**——即接收方无法区分一个帧是新的还是旧的重传。具体约束在后续 GBN 和 SR 部分详细证明。

### 5.2.5 确认方式

#### **累计确认（Cumulative ACK）**

一个 ACK(n) 表示序号 n 及之前的所有帧都已正确接收。即使中间某些帧的 ACK 丢失，后续的 ACK 也能覆盖。

- 优点：容错性好，单个 ACK 丢失不会导致重传
- 缺点：无法精确知道哪一帧出错（GBN 使用此方式）

#### **选择确认（Selective ACK）**

每个 ACK 只确认对应的单个帧。接收方可以发送关于特定帧的确认。

- 优点：精确，只重传出错的帧
- 缺点：实现复杂，需要更多缓冲区（SR 使用此方式）

### 5.2.6 超时与重传策略

**超时时间的选择**是一个关键问题：
- 太短：不必要的重传，浪费带宽
- 太长：出错时等待时间过长，降低吞吐量

理想的超时时间应略大于**往返时间（RTT）**：

$$\text{Timeout} = \text{RTT} + \text{安全余量}$$

RTT 的估计需要考虑其动态变化，常用方法包括指数加权移动平均（EWMA）和 Jacobson/Karels 算法（详见 5.6 节定时器部分）。

<div data-component="SlidingWindowConcept"></div>

---

## 5.3 回退N帧协议（Go-Back-N, GBN）

### 5.3.1 发送方行为

GBN 协议中，发送方维护一个大小为 $N$ 的发送窗口，可以连续发送窗口内的所有帧。发送方的详细行为如下：

**发送方状态变量**：
- `base`：最早未确认帧的序号
- `nextseqnum`：下一个待发送帧的序号
- `N`：窗口大小

**发送方 FSM（文字描述）**：

```
初始状态：base = 0, nextseqnum = 0

事件1：上层有数据要发送
  ── 检查：nextseqnum < base + N（窗口是否还有空间？）
  ├─ 是：封装帧（seq = nextseqnum），发送，nextseqnum++
  │       如果是窗口内第一个帧，启动计时器
  └─ 否：拒绝发送（或缓存等待）——称为"窗口满"

事件2：收到 ACK(n)
  ── 采用累计确认：ACK(n) 表示序号 ≤ n 的所有帧都已确认
  ── 更新 base = n + 1
  ── 如果 base == nextseqnum（所有帧都已确认），停止计时器
  └─ 否则，重启计时器（还有未确认的帧）

事件3：超时
  ── 重传从 base 到 nextseqnum - 1 的所有帧！（这是 GBN 的关键特征）
  ── 重启计时器
```

**关键点**：GBN 的"回退N帧"体现在事件3——超时时不是只重传一帧，而是重传所有已发送但未确认的帧。这就是"Go-Back-N"名称的由来：回退到序号 base 的位置，从那里开始全部重传。

### 5.3.2 接收方行为

GBN 的接收方非常简单——**只接受按序到达的帧**。

**接收方 FSM（文字描述）：

```
初始状态：expectedseqnum = 0

事件：收到帧（seq = n）
  ── 检查：n == expectedseqnum？
  ├─ 是（按序到达）：
  │   ── 将数据交付上层
  │   ── 发送 ACK(expectedseqnum)
  │   ── expectedseqnum++
  └─ 否（乱序到达）：
      ── 直接丢弃该帧！
      ── 重发对上一个按序帧的 ACK（即 ACK(expectedseqnum - 1)）
```

**为什么丢弃乱序帧？** 因为 GBN 的接收窗口大小为 1，接收方没有缓存来保存乱序帧。即使收到了帧 5 而期望帧 3，帧 5 也会被丢弃。当发送方最终重传帧 3 时，帧 4、5、6... 也会被一起重传。

这是 GBN 的一个缺点：在高误码率环境下，一个帧的丢失会导致大量后续帧被丢弃和重传。

### 5.3.3 窗口大小约束：$N \leq 2^m - 1$

**定理**：在 GBN 协议中，若序号用 $m$ 比特表示，则发送窗口大小必须满足 $N \leq 2^m - 1$。

**证明（用反例）**：

假设 $m = 2$，序号空间为 $\{0, 1, 2, 3\}$（共 $2^2 = 4$ 个序号）。假设窗口大小 $N = 4 = 2^2$（违反约束）。

**场景**：发送方连续发送帧 0、1、2、3。

1. 发送方发送帧 0、1、2、3
2. 接收方正确收到所有帧，发送 ACK 0、ACK 1、ACK 2、ACK 3
3. **所有 ACK 都在信道中丢失了**
4. 发送方超时，准备重传

此时发送方的窗口状态：`base = 0, nextseqnum = 4`

**问题来了**：发送方超时后重传帧 0，接收方收到一个序号为 0 的帧。但接收方此时已经回到了期望帧 0 的状态（因为上一轮的帧 0、1、2、3 都已确认并交付，`expectedseqnum = 0`）。

接收方无法区分：
- 这是第一轮的帧 0（应该接受并交付）
- 这是重传的帧 0（不应该再交付一次，否则数据重复）

由于 ACK 全部丢失，接收方认为这是一个新的帧 0，错误地将重复数据交付给上层。

**根本原因**：当 $N = 2^m$ 时，旧帧的序号和新帧的序号完全重叠，接收方无法区分新旧帧。

**正确约束**：$N \leq 2^m - 1$，这样即使所有 ACK 丢失，旧窗口和新窗口的序号范围也不会完全重叠，接收方总能区分。

| $m$（比特数） | 序号空间大小 | 最大窗口 $N$ |
|:---:|:---:|:---:|
| 1 | 2 | 1（退化为停等协议） |
| 2 | 4 | 3 |
| 3 | 8 | 7 |
| 4 | 16 | 15 |

### 5.3.4 详细例子

**场景设置**：序号 $m = 3$ 比特，序号空间 $\{0,1,2,3,4,5,6,7\}$，窗口大小 $N = 7$。

#### **情况1：正常传输**

```
时间线：
  发送方                              接收方
  ─────                              ─────
  发送 Frame[0] ─────────────────→ 收到 Frame[0]，交付上层
  发送 Frame[1] ─────────────────→ 收到 Frame[1]，交付上层
  发送 Frame[2] ─────────────────→ 收到 Frame[2]，交付上层
  发送 Frame[3] ─────────────────→ 收到 Frame[3]，交付上层
  发送 Frame[4] ─────────────────→ 收到 Frame[4]，交付上层
  发送 Frame[5] ─────────────────→ 收到 Frame[5]，交付上层
  发送 Frame[6] ─────────────────→ 收到 Frame[6]，交付上层
                ←───────────────── ACK(6) [累计确认]
  收到 ACK(6)，base = 7
  窗口滑动到 [7, 13%8] = [7, 5]
```

#### **情况2：帧丢失**

```
时间线：
  发送方                              接收方
  ─────                              ─────
  发送 Frame[0] ─────────────────→ 收到 Frame[0]，交付，ACK(0)
  发送 Frame[1] ─── [丢失] ───✕
  发送 Frame[2] ─────────────────→ 期望 Frame[1]，收到 Frame[2]，丢弃
  发送 Frame[3] ─────────────────→ 期望 Frame[1]，收到 Frame[3]，丢弃
  发送 Frame[4] ─────────────────→ 期望 Frame[1]，收到 Frame[4]，丢弃
  发送 Frame[5] ─────────────────→ 期望 Frame[1]，收到 Frame[5]，丢弃
  发送 Frame[6] ─────────────────→ 期望 Frame[1]，收到 Frame[6]，丢弃
  
  [此时发送窗口已满：base=0, nextseqnum=7, N=7]
  [发送方无法发送新帧，等待 ACK]
  
  [ACK(0) 到达发送方，base=1]
  [超时：重传 Frame[1..6] 全部6个帧！]
  
  重传 Frame[1] ─────────────────→ 收到 Frame[1]，交付，ACK(1)
  重传 Frame[2] ─────────────────→ 收到 Frame[2]，交付，ACK(2)
  ...（依此类推）
```

注意 GBN 的浪费：Frame[2..6] 都被正确接收过，但因为 Frame[1] 丢失，它们全部被丢弃并需要重传。

#### **情况3：ACK 丢失**

```
时间线：
  发送方                              接收方
  ─────                              ─────
  发送 Frame[0] ─────────────────→ 收到 Frame[0]，交付
  发送 Frame[1] ─────────────────→ 收到 Frame[1]，交付
  发送 Frame[2] ─────────────────→ 收到 Frame[2]，交付
                ←── ACK(2) [丢失] ──
  发送 Frame[3] ─────────────────→ 收到 Frame[3]，交付
                ←── ACK(3) ──────── 累积确认 Frame[0..3]
  
  收到 ACK(3)，base = 4
  窗口滑动，之前的 ACK(2) 丢失没有影响！
```

这是累计确认的优势：即使 ACK(2) 丢失，ACK(3) 的累积确认功能保证 Frame[0..3] 都被正确确认。

### 5.3.5 GBN 的优缺点分析

| 维度 | 分析 |
|------|------|
| **优点** | 实现简单，接收方无需缓存乱序帧 |
| **优点** | 累计确认对 ACK 丢失有天然容错能力 |
| **缺点** | 单帧丢失导致窗口内所有后续帧重传，浪费带宽 |
| **缺点** | 在高误码率信道上性能急剧下降 |
| **缺点** | 重传的帧中可能有很多已经被正确接收过 |
| **适用场景** | 信道质量好、误码率低的环境 |

<div data-component="GBNSimulator"></div>

---

## 5.4 选择重传协议（Selective Repeat, SR）

### 5.4.1 发送方行为

SR 协议的核心思想是：**只重传出错的帧，而不是整个窗口**。这需要发送方和接收方都维护更大的窗口和更多的状态。

**发送方状态变量**：
- `send_base`：发送窗口下界
- `N`：发送窗口大小
- 每个帧有独立的计时器

**发送方 FSM（文字描述）**：

```
初始状态：send_base = 0

事件1：上层有数据要发送
  ── 检查：nextseqnum < send_base + N
  ├─ 是：封装帧（seq = nextseqnum），发送，为该帧启动独立计时器，nextseqnum++
  └─ 否：拒绝（窗口满）

事件2：收到 ACK(n)
  ── 标记帧 n 为"已确认"
  ── 如果 n == send_base：
  │   ── 窗口滑动：找到连续已确认帧的最大序号 k
  │   ── send_base = k + 1
  │   ── 取消 send_base 到 k 之间所有帧的计时器
  └─ 否则：只标记，不滑动窗口（等待按序确认）

事件3：帧 n 超时
  ── 只重传帧 n！（不是整个窗口）
  ── 为帧 n 重启计时器
```

**关键区别**：
- 每个帧有**独立的计时器**（GBN 只有一个计时器）
- 超时时**只重传超时的那一帧**（GBN 重传整个窗口）
- ACK 是**选择确认**，只确认单个帧（GBN 是累计确认）

### 5.4.2 接收方行为

SR 的接收方比 GBN 复杂得多——它需要**缓存乱序到达的帧**。

**接收方状态变量**：
- `recv_base`：接收窗口下界（期望的最小未确认帧序号）
- `W`：接收窗口大小

**接收方 FSM（文字描述）**：

```
初始状态：recv_base = 0

事件：收到帧（seq = n）
  ── 检查：recv_base ≤ n < recv_base + W（n 在接收窗口内？）
  ├─ 是（窗口内）：
  │   ├─ 如果是新帧（之前没收到过）：
  │   │   ── 缓存该帧
  │   │   ── 发送 ACK(n)
  │   └─ 如果是重复帧（已缓存）：
  │       ── 丢弃数据，但仍然发送 ACK(n)
  │   
  │   ── 检查：n == recv_base？
  │   ├─ 是：将 recv_base 及后续连续已缓存的帧都交付上层
  │   │       recv_base 滑动到第一个未缓存帧的位置
  │   └─ 否：不交付，等待后续帧填补空缺
  │
  └─ 否（窗口外）：
      ── 丢弃
      ── 不发送 ACK（或发送一个旧的 ACK）
```

**接收方缓存的重要性**：

假设接收方已经收到帧 0、1、2、4、5，但帧 3 丢失：
- 帧 0、1、2 已经按序交付给上层，`recv_base = 3`
- 帧 4、5 被缓存（不交付，因为帧 3 还没到）
- 当帧 3 最终到达时，帧 3、4、5 一起按序交付
- `recv_base` 滑动到 6

### 5.4.3 窗口大小约束：$W_{send} + W_{recv} \leq 2^m$

**定理**：在 SR 协议中，发送窗口和接收窗口的大小之和不能超过序号空间的大小：

$$W_{send} + W_{recv} \leq 2^m$$

通常取 $W_{send} = W_{recv} = W$，则 $W \leq 2^{m-1}$。

**证明（用序号歧义的反例）**：

假设 $m = 2$，序号空间 $\{0, 1, 2, 3\}$，$W_{send} = 3$，$W_{recv} = 3$。

$W_{send} + W_{recv} = 6 > 4 = 2^2$，违反约束。

**场景构造**：

1. 发送方发送帧 0、1、2（窗口 [0, 2]）
2. 接收方正确收到帧 0、1、2，发送 ACK 0、1、2
3. 接收方交付帧 0、1、2，接收窗口滑动到 [3, 5%4] = [3, 1]，即 {3, 0, 1}
4. **所有 ACK 丢失**
5. 发送方超时，重传帧 0

此时：
- 发送方认为帧 0、1、2 未确认，窗口仍为 [0, 2]
- 接收方已经滑动窗口到 [3, 1]，即接受 {3, 0, 1}

接收方收到序号为 0 的帧：
- 0 在接收窗口 [3, 1] 内
- 但接收方**无法区分**：
  - 这是发送方重传的旧帧 0（应该丢弃，因为已经收到并交付过）
  - 这是发送方新发送的帧 0（新窗口 [4, 6%4] = [0, 2] 中的帧 0，应该接受）

**根本原因**：当窗口重叠时，旧窗口中的帧和新窗口中的帧可能具有相同的序号，接收方无法区分。

**约束推导**：要避免歧义，需要保证在最坏情况下（所有 ACK 丢失），旧窗口和新窗口的序号范围不重叠。这要求：

$$W_{send} + W_{recv} \leq 2^m$$

即两个窗口的大小之和不能超过序号空间。

| $m$ | 序号空间 | 最大 $W_{send} + W_{recv}$ | 若 $W_{send} = W_{recv}$ |
|:---:|:---:|:---:|:---:|
| 2 | 4 | 4 | $W \leq 2$ |
| 3 | 8 | 8 | $W \leq 4$ |
| 4 | 16 | 16 | $W \leq 8$ |

### 5.4.4 详细例子

**场景设置**：$m = 3$，序号空间 $\{0..7\}$，$W_{send} = 4$，$W_{recv} = 4$。

**传输过程**：

```
时间   发送方行为                        信道            接收方行为
────   ────────                        ────            ────────
t0     发送 Frame[0]                   ──────→         收到，缓存，ACK(0)
t1     发送 Frame[1]                   ──────→         收到，缓存，ACK(1)
t2     发送 Frame[2]                   ───[丢失]──✕
t3     发送 Frame[3]                   ──────→         收到，缓存，ACK(3)
                                                        [Frame[0,1,3]已缓存，Frame[2]缺失]
                                                        [不交付，等待Frame[2]]
t4     收到 ACK(0)                                   
       send_base 仍为 0（Frame[2]未确认）
t5     收到 ACK(1)
t6     收到 ACK(3)
       [Frame[2]的计时器超时]
t7     只重传 Frame[2]！               ──────→         收到 Frame[2]，缓存
                                                        [Frame[0,1,2,3]都已缓存，按序]
                                                        交付 Frame[0,1,2,3 给上层
                                                        recv_base 滑动到 4
                                                        发送 ACK(2)
t8     收到 ACK(2)
       send_base 滑动到 4
       窗口变为 [4, 7]
```

注意与 GBN 的对比：GBN 在 Frame[2] 超时时会重传 Frame[2,3]（整个窗口），而 SR 只重传 Frame[2]。

### 5.4.5 SR 的优缺点分析

| 维度 | 分析 |
|------|------|
| **优点** | 只重传出错帧，带宽利用率高 |
| **优点** | 在高误码率信道上性能远优于 GBN |
| **优点** | 不需要等待整个窗口重传完成 |
| **缺点** | 接收方需要缓存乱序帧，内存开销大 |
| **缺点** | 每个帧需要独立计时器，实现复杂 |
| **缺点** | 发送方和接收方都需要更复杂的状态管理 |
| **适用场景** | 信道质量差、误码率高的环境 |

### 5.4.6 SR vs GBN 详细比较

| 比较维度 | GBN | SR |
|---------|-----|-----|
| 重传策略 | 重传整个窗口 | 只重传出错帧 |
| 接收窗口 | 1 | $W > 1$ |
| 接收方缓存 | 不需要 | 需要缓存乱序帧 |
| 确认方式 | 累计确认 | 选择确认 |
| 计时器 | 一个全局计时器 | 每帧一个独立计时器 |
| 窗口约束 | $N \leq 2^m - 1$ | $W_{send} + W_{recv} \leq 2^m$ |
| 实现复杂度 | 低 | 高 |
| 高误码率性能 | 差（大量冗余重传） | 好（精确重传） |
| 低误码率性能 | 好（几乎无需重传） | 好（略复杂但无额外开销） |
| 缓冲区需求 | 少 | 多 |

**实际选择建议**：
- 信道质量好（误码率 $< 10^{-6}$）：GBN 简单高效
- 信道质量差（误码率 $> 10^{-4}$）：SR 性能显著优于 GBN
- 现代网络中：TCP 使用的选择确认（SACK）本质上是 SR 的变体

<div data-component="SRSimulator"></div>

---

## 5.5 协议性能分析

### 5.5.1 链路利用率计算

链路利用率是衡量协议效率的核心指标。定义：

$$U_{sender} = \frac{\text{发送数据的时间}}{\text{总时间}}$$

对于一个帧长为 $L$ 比特、链路带宽为 $R$ bps、往返时间为 RTT 的链路：

$$U = \frac{L/R}{L/R + RTT + ACK_{size}/R}$$

如果 ACK 帧很小（可以忽略），简化为：

$$U = \frac{L/R}{L/R + RTT}$$

定义**带宽延迟积（Bandwidth-Delay Product, BDP）**：

$$BDP = R \times RTT$$

BDP 表示在任何时刻"管道中"可以容纳的最大比特数。利用率可以写成：

$$U = \frac{L}{L + BDP}$$

### 5.5.2 停等协议利用率

$$U_{SW} = \frac{L/R}{L/R + RTT}$$

当 $L/R \ll RTT$（高带宽长延迟链路）时，$U_{SW} \approx \frac{L}{R \cdot RTT} = \frac{L}{BDP}$，利用率极低。

### 5.5.3 GBN 协议利用率

GBN 允许连续发送 $N$ 个帧，在理想情况下（无丢包）：

$$U_{GBN} = \min\left(1, \frac{N \cdot L/R}{L/R + RTT}\right) = \min\left(1, \frac{N \cdot L}{L + BDP}\right)$$

当 $N \cdot L \geq BDP$ 时，$U_{GBN} = 1$，即管道被完全填满。

因此，要达到 100% 利用率，需要：

$$N \geq \frac{BDP}{L} = \frac{R \times RTT}{L}$$

### 5.5.4 SR 协议利用率

在无丢包情况下，SR 的利用率与 GBN 相同。但在有丢包时，SR 的优势在于只重传出错帧：

$$U_{SR} = \frac{L/R}{L/R + RTT} \times \frac{1}{1 - p + p/N_{retrans}}$$

其中 $p$ 为丢包率，$N_{retrans}$ 为平均每次重传的帧数（SR 为 1，GBN 为整个窗口）。

### 5.5.5 带宽延迟积的影响

**高 BDP 网络**（如卫星链路、跨洋光纤）：

| 参数 | 典型值 |
|------|--------|
| 带宽 $R$ | 1 Gbps |
| RTT | 200 ms（卫星） |
| BDP | $10^9 \times 0.2 = 2 \times 10^8$ bits = 25 MB |
| 帧长 $L$ | 12000 bits（1500 bytes） |

停等协议利用率：$U = \frac{12000}{12000 + 2 \times 10^8} \approx 0.006\%$

要达到 100% 利用率：$N \geq \frac{2 \times 10^8}{12000} \approx 16667$ 帧

这意味着需要至少 16667 个帧的窗口大小——需要至少 15 比特的序号空间。

### 5.5.6 例题

**题目**：一条链路带宽为 10 Mbps，单程传播延迟为 50 ms，帧长为 1000 bytes。分别计算停等协议和窗口大小为 100 的 GBN 协议的链路利用率。

**解答**：

$$L/R = \frac{1000 \times 8}{10 \times 10^6} = 0.8 \text{ ms}$$

$$RTT = 2 \times 50 = 100 \text{ ms}$$

**停等协议**：

$$U_{SW} = \frac{0.8}{0.8 + 100} = \frac{0.8}{100.8} \approx 0.79\%$$

**GBN（N = 100）**：

$$U_{GBN} = \min\left(1, \frac{100 \times 0.8}{0.8 + 100}\right) = \min\left(1, \frac{80}{100.8}\right) \approx 79.4\%$$

将窗口从 1 增大到 100，利用率提升了约 100 倍。

**达到 100% 利用率需要的窗口大小**：

$$N \geq \frac{BDP}{L} = \frac{10 \times 10^6 \times 0.1}{8000} = \frac{10^6}{8000} = 125$$

<div data-component="ProtocolPerformanceCalculator"></div>

---

## 5.6 捎带确认（Piggybacking）

### 5.6.1 双向数据传输

前面讨论的都是**单向**数据传输：A 发送数据给 B，B 只返回 ACK。但在实际通信中，数据通常是**双向**的——A 和 B 互相发送数据。

如果 ACK 和数据帧分别发送，会造成信道浪费。**捎带确认（Piggybacking）**的思想是：**将 ACK 搭载在数据帧中一起发送**。

```
无捎带：
  A → B: Data Frame[0]
  B → A: ACK Frame（独立的 ACK 帧，有帧头开销）
  A → B: Data Frame[1]
  B → A: ACK Frame

有捎带：
  A → B: Data Frame[0]
  B → A: Data Frame + ACK(0) [ACK 搭载在 B 的数据帧中]
  A → B: Data Frame[1] + ACK(B的数据帧)
```

### 5.6.2 延迟 ACK 的问题

捎带确认面临一个权衡：**等多久才捎带？**

如果 B 暂时没有数据要发送给 A，ACK 就不能被捎带。此时有两个选择：
1. **立即发送独立 ACK 帧**：增加帧数和开销
2. **等待一小段时间**（通常 50-200 ms），期望有数据帧可以捎带：增加延迟

大多数协议采用折中方案：**设置一个延迟 ACK 定时器**。如果在定时器超时前有数据要发送，就捎带 ACK；否则超时后发送独立 ACK。

### 5.6.3 Nagle 算法简介

Nagle 算法（RFC 896）是 TCP 中用于减少小包数量的算法，与捎带确认配合使用：

**核心规则**：
1. 如果发送缓冲区中的数据达到一个 MSS（最大报文段长度），立即发送
2. 如果没有待确认的数据（即上一个发送的数据已被 ACK），立即发送
3. 否则，将数据留在缓冲区，等到 ACK 到达或缓冲区满时再发送

**效果**：在低速交互式应用（如 Telnet）中，Nagle 算法将多个小的按键数据合并成一个大帧发送，显著减少了帧数。

**缺点**：会增加延迟，在实时交互应用（如游戏）中可能不可接受。可通过 `TCP_NODELAY` 选项禁用。

---

## 5.7 协议验证与测试

### 5.7.1 有限状态机（FSM）建模

协议的正确性可以通过**有限状态机（Finite State Machine, FSM）**来建模和验证。

一个协议的 FSM 包含：
- **状态集合** $S$：协议可能处于的所有状态
- **输入集合** $I$：所有可能的事件（收到帧、超时、上层数据到达等）
- **输出集合** $O$：所有可能的动作（发送帧、交付数据、启动计时器等）
- **转移函数** $\delta: S \times I \rightarrow S \times O$
- **初始状态** $s_0$

**停等协议发送方的 FSM 示例**（状态转移表）：

| 当前状态 | 输入 | 输出 | 下一状态 |
|---------|------|------|---------|
| Ready | 上层数据到达 | 发送 Frame(seq=0)，启动 Timer | WaitACK0 |
| WaitACK0 | 收到 ACK(1) | 取消 Timer | Ready |
| WaitACK0 | 超时 | 重传 Frame(seq=0)，重启 Timer | WaitACK0 |
| Ready | 上层数据到达 | 发送 Frame(seq=1)，启动 Timer | WaitACK1 |
| WaitACK1 | 收到 ACK(0) | 取消 Timer | Ready |
| WaitACK1 | 超时 | 重传 Frame(seq=1)，重启 Timer | WaitACK1 |

**验证方法**：遍历所有状态-输入组合，检查：
1. **活性（Liveness）**：协议不会进入死锁状态
2. **安全性（Safety）**：不会出现重复交付或数据丢失
3. **完整性（Completeness）**：每个状态的每个输入都有定义的转移

### 5.7.2 Petri 网简介

**Petri 网**是一种用于建模并发系统的图形化数学工具，特别适合分析协议中的**并行性**和**同步**问题。

Petri 网由以下元素组成：
- **库所（Place）**：用圆圈表示，可以包含令牌（token）
- **变迁（Transition）**：用方框或粗线表示
- **弧（Arc）**：连接库所和变迁
- **令牌（Token）**：库所中的黑点，表示状态

**变迁的触发规则**：
- 当一个变迁的所有输入库所都有令牌时，变迁可以触发（fire）
- 触发后，从每个输入库所移除一个令牌，向每个输出库所添加一个令牌

Petri 网可以发现 FSM 难以检测的问题，如**活锁（Livelock）**和**资源竞争**。

### 5.7.3 协议的正确性证明

协议正确性通常需要证明以下性质：

**安全性性质（不变式）**：
1. 接收方不会交付重复的数据包
2. 接收方不会交付乱序的数据包（对于按序交付的服务）
3. 发送窗口和接收窗口始终满足大小约束

**活性性质（最终性）**：
1. 每个发送的帧最终都会被接收方正确接收
2. 每个数据包最终都会被交付给上层
3. 协议不会死锁

**形式化验证工具**：
- **模型检测（Model Checking）**：自动穷举状态空间，如 SPIN、NuSMV
- **定理证明（Theorem Proving）**：用逻辑推导证明性质，如 Coq、Isabelle
- **进程代数**：用代数方法描述协议行为，如 CCS、CSP

---

## 5.8 实际应用中的链路层协议

### 5.8.1 HDLC 协议

**HDLC（High-level Data Link Control）** 是 ISO 制定的面向比特的链路层协议，是许多现代协议的基础。

**HDLC 帧格式**：

```
┌──────┬──────┬────────┬──────┬──────┐
│ Flag │ Addr │ Control│ Data │  FCS │ Flag │
│ 8bit │ 8bit │ 8/16bit│ 可变 │16/32│ 8bit │
└──────┴──────┴────────┴──────┴──────┘
```

- **Flag**：`01111110`，帧定界符
- **Address**：站地址
- **Control**：帧类型和序号信息
- **FCS**：帧校验序列（CRC）

**Control 字段的三种格式**：

| 格式 | 帧类型 | 字段内容 |
|------|--------|---------|
| Information（I帧） | 数据帧 | N(S)发送序号 + N(R)接收序号 + P/F + 数据 |
| Supervisory（S帧） | 控制帧 | 类型（RR/RNR/REJ/SREJ）+ N(R) + P/F |
| Unnumbered（U帧） | 控制帧 | 命令/响应编码 + P/F |

HDLC 支持三种操作模式：
- **NRM（Normal Response Mode）**：主从模式
- **ARM（Asynchronous Response Mode）**：异步响应
- **ABM（Asynchronous Balanced Mode）**：对等模式，最常用

HDLC 的滑动窗口机制：
- 序号默认 3 比特（可扩展到 7 比特）
- 窗口大小默认 7（可扩展到 127）
- 支持 Go-Back-N 和 Selective Reject

### 5.8.2 PPP 协议

**PPP（Point-to-Point Protocol）** 是用于点对点链路的数据链路层协议，广泛用于拨号上网和专线连接。

**PPP 帧格式**：

```
┌──────┬──────┬────────┬──────┬──────┐
│ Flag │ Addr │ Control│ Proto│ Data │  FCS │ Flag │
│ 0x7E │ 0xFF │ 0x03   │ 2byte│ 可变 │ 2byte│ 0x7E │
└──────┴──────┴────────┴──────┴──────┘
```

PPP 的特点：
- **无滑动窗口**：PPP 本身不提供可靠传输，依赖上层协议（如 TCP）
- **Address 和 Control 字段固定**：简化实现
- **Protocol 字段**：标识上层协议（如 0x0021 = IP，0xC021 = LCP）
- **字节填充**：用 `0x7D` 作为转义字符

PPP 协议族包含：
- **LCP（Link Control Protocol）**：链路建立、配置、测试
- **NCP（Network Control Protocol）**：网络层协议配置（如 IPCP 配置 IP 地址）
- **PAP/CHAP**：认证协议

### 5.8.3 以太网的滑动窗口应用

现代以太网（IEEE 802.3）在数据链路层**不使用滑动窗口协议**——它提供的是**无确认、无连接**的服务。以太网假设：
- 误码率极低（光纤链路 BER < $10^{-12}$）
- 上层协议（TCP）会处理可靠传输

但以太网使用了一些与流控相关的机制：

**PAUSE 帧（IEEE 802.3x）**：
- 当接收方缓冲区快满时，发送 PAUSE 帧通知发送方暂停发送
- PAUSE 帧指定暂停时间（以 512 比特时间为单位）
- 这是一种简单的流控机制，不是滑动窗口

**滑动窗口在以太网环境中的真正应用**是在 TCP 层：
- TCP 连接的两端维护滑动窗口
- TCP 的窗口大小可动态调整（流量控制 + 拥塞控制）
- TCP 的 SACK 选项是选择重传机制的实现
- TCP 的序号空间为 32 比特，窗口最大可达 $2^{30}$（约 1GB）

---

## 5.9 组件分析

### 组件A：发送窗口与接收窗口的硬件寄存器实现

#### **窗口边界的寄存器表示**

在硬件实现中，发送窗口和接收窗口通过专用寄存器来管理。每个窗口至少需要以下寄存器：

**发送窗口寄存器组**：

```
┌──────────────────────────────────────────────────┐
│ 发送窗口寄存器组                                    │
├─────────────┬──────────────┬─────────────────────┤
│ send_base   │ nextseqnum   │ send_window_size    │
│ (m bits)    │ (m bits)     │ (m bits)            │
├─────────────┼──────────────┼─────────────────────┤
│ 最早未确认   │ 下一个待发送   │ 窗口大小 N          │
│ 帧的序号    │ 帧的序号      │                     │
└─────────────┴──────────────┴─────────────────────┘
```

`send_base` 寄存器保存最早未被确认的帧序号。每次收到一个有效的累计 ACK，该寄存器递增。`nextseqnum` 寄存器保存下一个可以被赋予的序号。每发送一帧，该寄存器递增。`send_window_size` 寄存器保存窗口大小 $N$，在协议运行过程中通常固定不变（但可以动态调整）。

发送条件的硬件判断逻辑：

```
发送允许 = (nextseqnum - send_base) mod 2^m < send_window_size
```

这通过一个模减法器和比较器实现。模运算确保序号回绕时逻辑正确。

**接收窗口寄存器组**：

```
┌──────────────────────────────────────────────────┐
│ 接收窗口寄存器组                                    │
├─────────────┬──────────────┬─────────────────────┤
│ recv_base   │ recv_window  │ expected_seq        │
│ (m bits)    │ _size (m)    │ (m bits, for GBN)   │
├─────────────┼──────────────┼─────────────────────┤
│ 窗口下界     │ 窗口大小 W   │ 期望的下一帧序号     │
└─────────────┴──────────────┴─────────────────────┘
```

#### **用循环缓冲区实现发送窗口**

发送窗口在物理内存中通常用**循环缓冲区（Circular Buffer）**实现。缓冲区大小等于窗口大小 $N$，每个槽位对应一个序号。

```
缓冲区结构（N=8，m=3）：
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│slot0│slot1│slot2│slot3│slot4│slot5│slot6│slot7│
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│Frame│Frame│Frame│     │     │     │     │     │
│  0  │  1  │  2  │空闲  │空闲  │空闲  │空闲  │空闲 │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  ↑send_base=0            ↑nextseqnum=3
```

**序号到缓冲区地址的映射**：

$$\text{buffer\_index} = \text{seq\_num} \mod N$$

这通过简单的位截断实现（当 $N = 2^k$ 时，取序号的低 $k$ 位）。如果 $N$ 不是 2 的幂，则需要模运算电路。

**注意**：GBN 中 $N \leq 2^m - 1$，不一定等于 2 的幂。因此通用实现需要模 $N$ 运算，而不是简单的位截断。

**窗口滑动的硬件逻辑**：

```
收到 ACK(n) 时的处理流程：
1. 检查：n 是否在 [send_base, nextseqnum-1] 范围内？
   ── 使用模比较器：(n - send_base) mod 2^m < (nextseqnum - send_base) mod 2^m
2. 如果有效：
   ── 标记 buffer[n mod N] 的状态为"已确认"
   ── 对于 GBN（累计确认）：
   │   ── send_base = n + 1（模 2^m）
   │   ── 释放 buffer[旧 send_base] 到 buffer[n mod N] 的所有槽位
   └─ 对于 SR（选择确认）：
       ── 只标记该帧，不移动 send_base
       ── 循环检查从 send_base 开始的连续已确认帧，移动 send_base
```

#### **用循环缓冲区实现接收窗口**

接收缓冲区的结构类似，但用途不同：

**GBN**：接收窗口大小为 1，只需要一个缓冲区槽位。

**SR**：接收窗口大小为 $W$，需要 $W$ 个槽位来缓存乱序帧。

```
SR 接收缓冲区（W=4，m=3）：
┌─────┬─────┬─────┬─────┐
│slot0│slot1│slot2│slot3│
├─────┼─────┼─────┼─────┤
│F[0]✓│F[1]✓│     │F[3]✓│
│已交付│已交付│空闲  │已缓存│
└─────┴─────┴─────┴─────┘
  recv_base=2（等待 Frame[2]）
  Frame[3] 已缓存，等 Frame[2] 到达后一起交付
```

#### **窗口满/空的判断逻辑**

**发送窗口满**：

```
窗口满 = ((nextseqnum - send_base) mod 2^m) >= send_window_size
```

**发送窗口空**：

```
窗口空 = (nextseqnum == send_base)  // 所有帧都已确认
```

**接收缓冲区满**（SR）：

```
缓冲区满 = 已使用槽位数 >= recv_window_size
```

硬件实现时，"已使用槽位数"通过一个计数器维护：收到新帧时加 1，交付帧给上层时减 1。

#### **与内存管理的交互**

在高速网络中，发送缓冲区和接收缓冲区通常位于**网卡（NIC）的专用 SRAM** 中，而不是主存中。这避免了 DMA 和总线延迟。

缓冲区管理策略：
- **静态分配**：每个连接预分配固定大小的缓冲区
- **动态分配**：从共享缓冲区池中按需分配
- **零拷贝（Zero-Copy）**：发送时直接从用户空间 DMA 到网卡，避免内核缓冲区拷贝

---

### 组件B：序号管理器的设计

#### **序号空间**

$m$ 比特的序号可以表示 $2^m$ 个不同的值：$\{0, 1, 2, \ldots, 2^m - 1\}$。序号空间是**循环的**——$2^m - 1$ 之后回到 $0$。

```
m=3 时的序号环：

      0
    7   1
   6     2
    5   3
      4

方向：0 → 1 → 2 → ... → 7 → 0 → 1 → ...
```

#### **序号递增器**

序号递增器的硬件实现：

```
┌─────────────────────────────────────────────┐
│ 序号递增器                                    │
│                                              │
│  当前序号 ──→ [模 2^m 加法器] ──→ 新序号     │
│                  ↑                           │
│              +1（固定输入）                    │
│                                              │
│  Verilog:                                    │
│  next_seq = (current_seq + 1) & ((1<<m)-1); │
│  // 当 m=3 时，mask = 0b111                  │
│  // 7+1 = 8 → 8 & 7 = 0（回绕到0）          │
└─────────────────────────────────────────────┘
```

#### **序号比较器（处理回绕）**

序号比较是硬件实现中最微妙的部分。由于序号空间是循环的，普通的整数比较会出错。

**问题**：假设 $m = 3$，序号 7 和序号 1，哪个"更新"？

如果用普通比较：$7 > 1$，所以 7 更新。但如果发送方已经发到序号 1（回绕了一圈），那么 1 实际上比 7 更新。

**解决方案：模 $2^m$ 的有符号差值比较**：

```c
// 判断 seq_a 是否在 seq_b 之后（即 seq_a 更新）
// 使用模 2^m 的有符号差值
int is_newer(int seq_a, int seq_b, int m) {
    int half = 1 << (m - 1);  // 2^(m-1)
    int diff = (seq_a - seq_b) & ((1 << m) - 1);  // 差值，模 2^m
    return (diff > 0) && (diff < half);
}
```

**原理**：如果 $a - b$ 的差值在 $(0, 2^{m-1})$ 范围内，说明 $a$ 在 $b$ 之后（更新）。如果差值在 $(2^{m-1}, 2^m)$ 范围内，说明 $b$ 实际上在 $a$ 之后（$a$ 是旧的回绕值）。

硬件实现：一个 $m$ 位减法器 + 一个判断结果是否在 $(0, 2^{m-1})$ 的比较器。

#### **序号空间与窗口大小的关系**

这是序号管理器设计中最关键的约束：

| 协议 | 约束 | 原因 |
|------|------|------|
| GBN | $N \leq 2^m - 1$ | 防止 ACK 全部丢失时序号歧义 |
| SR | $W_{send} + W_{recv} \leq 2^m$ | 防止窗口重叠时新旧帧序号歧义 |

#### **为什么 GBN 需要 $N \leq 2^m - 1$**

用 $m = 2$ 的反例说明。假设 $N = 4 = 2^2$：

1. 发送方发完帧 0、1、2、3（窗口 [0, 3]）
2. 所有 ACK 丢失
3. 发送方超时，重传帧 0
4. 接收方此时已经确认了帧 0、1、2、3，期望帧 0（新一圈的帧 0）

接收方无法区分重传的旧帧 0 和新一圈的帧 0。如果 $N = 3 = 2^2 - 1$：

1. 发送方发完帧 0、1、2（窗口 [0, 2]）
2. ACK(0) 到达，窗口滑动到 [1, 3]
3. 发送方发送帧 3
4. 所有后续 ACK 丢失
5. 发送方超时，重传帧 1、2、3

此时接收方的期望值为 1（因为之前的帧 0 已确认），可以正确区分重传帧和新帧。

#### **为什么 SR 需要 $W \leq 2^{m-1}$**

SR 中发送方和接收方各自独立滑动窗口。最坏情况：发送方的窗口和接收方的窗口"面对面"接近但不重叠。

如果 $W_{send} = W_{recv} = 2^{m-1}$，则两个窗口大小之和恰好为 $2^m$，刚好覆盖整个序号空间。再大一个就重叠了。

#### **硬件实现：模运算电路**

```verilog
// 模 2^m 递增器
module seq_incrementer #(
    parameter M = 3  // 序号比特数
)(
    input  [M-1:0] current_seq,
    output [M-1:0] next_seq
);
    assign next_seq = current_seq + 1;  // 自然溢出即为模 2^m
endmodule

// 模 2^m 比较器：判断 a 是否在 (low, low+window) 范围内
module seq_in_window #(
    parameter M = 3
)(
    input  [M-1:0] seq,      // 待检查序号
    input  [M-1:0] base,     // 窗口下界
    input  [M-1:0] wsize,    // 窗口大小
    output         in_window
);
    wire [M-1:0] diff = seq - base;  // 模 2^m 差值
    assign in_window = (diff < wsize);
endmodule
```

---

### 组件C：ARQ 定时器的工作原理

#### **定时器的硬件实现**

ARQ 定时器通常用**向下计数器 + 比较器**实现：

```
┌──────────────────────────────────────────────┐
│ ARQ 定时器硬件结构                              │
│                                               │
│  时钟信号 ──→ [向下计数器] ──→ 当前值           │
│                   ↑                            │
│              [初始值寄存器]                      │
│              (超时时间 T)                       │
│                                               │
│  当前值 ──→ [零检测器] ──→ 超时中断信号         │
│                                               │
│  启动：加载初始值 T，开始递减                    │
│  取消：停止计数器                               │
│  超时：计数器递减到 0，触发中断                  │
└──────────────────────────────────────────────┘
```

硬件实现代码（Verilog 概念）：

```verilog
module arq_timer #(
    parameter WIDTH = 32  // 计数器位宽
)(
    input  clk,
    input  rst_n,
    input  start,           // 启动定时器
    input  cancel,          // 取消定时器
    input  [WIDTH-1:0] timeout_val,  // 超时值
    output reg timeout      // 超时信号
);
    reg [WIDTH-1:0] counter;
    reg running;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            counter <= 0;
            running <= 0;
            timeout <= 0;
        end else if (start) begin
            counter <= timeout_val;
            running <= 1;
            timeout <= 0;
        end else if (cancel) begin
            running <= 0;
            timeout <= 0;
        end else if (running) begin
            if (counter == 0) begin
                timeout <= 1;
                running <= 0;
            end else begin
                counter <= counter - 1;
            end
        end
    end
endmodule
```

#### **超时时间的选择：RTO 计算**

超时时间（Retransmission Timeout, RTO）的选择是一个关键问题：

- **太短**：导致不必要的重传，浪费带宽，甚至加剧拥塞
- **太长**：帧丢失时等待时间过长，降低吞吐量

理想的 RTO 应该略大于 RTT：

$$\text{RTO} = \text{EstimatedRTT} + \text{SafetyMargin}$$

#### **Karn/Partridge 算法**

Karn/Partridge 算法（1987）是最简单的 RTT 估计算法：

```
1. 对每个发送的帧（非重传），记录发送时间
2. 收到该帧的 ACK 时，计算 SampleRTT = 当前时间 - 发送时间
3. 用指数加权移动平均（EWMA）更新估计值：
   EstimatedRTT = (1 - α) × EstimatedRTT + α × SampleRTT
   典型值：α = 0.125（即 1/8）
4. RTO = 2 × EstimatedRTT（安全系数为 2）

关键规则：重传的帧不参与 RTT 估算！
原因：收到 ACK 时无法确定它确认的是原始帧还是重传帧，
      如果确认的是重传帧，SampleRTT 会偏大。
```

#### **Jacobson/Karels 算法**

Jacobson/Karels 算法（1988）是更精确的 RTT 估计算法，被 TCP 采用。它不仅估计 RTT 的均值，还估计 RTT 的**变化（方差）**：

```
1. 采样：SampleRTT = 确认时间 - 发送时间

2. 计算误差：Err = SampleRTT - EstimatedRTT

3. 更新估计值：
   EstimatedRTT = EstimatedRTT + δ × Err         （δ = 1/8）
   DevRTT = DevRTT + δ × (|Err| - DevRTT)        （δ = 1/4）

4. 计算 RTO：
   RTO = EstimatedRTT + 4 × DevRTT

其中：
  - EstimatedRTT：RTT 的指数加权移动平均
  - DevRTT：RTT 偏差的指数加权移动平均
  - 4 × DevRTT：安全余量，自适应于 RTT 的波动程度
```

当 RTT 波动大时（DevRTT 大），RTO 自动增大以避免虚假重传。当 RTT 稳定时（DevRTT 小），RTO 接近 EstimatedRTT，对丢包响应更快。

#### **每帧一个定时器（SR）vs 单一定时器（GBN）**

| 方面 | GBN（单一定时器） | SR（每帧一个定时器） |
|------|-----------------|---------------------|
| 硬件开销 | 1 个计数器 + 1 个比较器 | $N$ 个计数器 + $N$ 个比较器 |
| 触发条件 | 最早未确认帧超时 | 任意帧超时 |
| 重传行为 | 重传整个窗口 | 只重传超时帧 |
| 实现复杂度 | 低 | 高 |

**GBN 只需一个定时器**：因为超时时要重传整个窗口，只需要跟踪最早未确认帧的超时即可。

**SR 需要每帧一个定时器**：因为每个帧独立超时和重传。在硬件中，可以用定时器池（Timer Wheel）来高效管理大量定时器。

**定时器轮（Timer Wheel）**：

```
定时器轮（8 槽）：
      ┌───┐
  ┌───┤ 0 ├───┐
  │   └───┘   │
┌─┴─┐       ┌─┴─┐
│ 7 │       │ 1 │
└─┬─┘       └─┬─┘
  │   ┌───┐   │
  ├───┤ 2 ├───┤
  │   └───┘   │
┌─┴─┐       ┌─┴─┐
│ 6 │       │ 3 │
└─┬─┘       └─┬─┘
  │   ┌───┐   │
  ├───┤ 4 ├───┤
  │   └───┘   │
  └───┤ 5 ├───┘
      └───┘

指针每 tick 顺时针旋转一格。
每个槽是一个链表，挂载在该 tick 到期的定时器。
新定时器挂在 (当前指针 + 超时 ticks) mod 8 的槽上。
时间复杂度：O(1) 插入，O(1) 触发。
```

#### **定时器与中断机制**

定时器超时通过**硬件中断**通知协议处理软件：

```
定时器超时 → 触发中断请求（IRQ）
           → CPU 保存当前上下文
           → 跳转到中断服务程序（ISR）
           → ISR：重传对应帧，重启定时器
           → 恢复上下文，继续正常处理
```

在高速网络中，定时器中断频率可能非常高。优化方法：
- **中断合并（Interrupt Coalescing）**：多个超时合并为一次中断
- **轮询模式（Polling）**：在高负载时切换到轮询，避免中断风暴

#### **定时器精度对协议性能的影响**

定时器的精度（粒度）直接影响协议性能：

- **精度太高**（如 1 μs）：硬件成本高，但 RTO 精确
- **精度太低**（如 100 ms）：硬件成本低，但 RTO 可能远大于实际 RTT，导致帧丢失时等待过久

**实际系统中的定时器精度**：
- Linux 内核：定时器精度取决于 `HZ` 配置，通常 1 ms 到 10 ms
- 高速网络适配器：硬件定时器精度可达 1 μs 或更高
- 嵌入式系统：通常 1 ms 到 10 ms

定时器精度 $\Delta t$ 对 RTO 的影响：

$$\text{实际 RTO} \geq \text{计算 RTO}$$
$$\text{实际 RTO} \leq \text{计算 RTO} + \Delta t$$

如果 $\Delta t$ 与 RTT 同数量级（如 $\Delta t = 10$ ms，RTT = 20 ms），则 RTO 的误差可达 50%，对协议效率影响显著。

<div data-component="ARQTimerDemo"></div>

---

### 组件D：协议性能模拟器

#### **用 Python 实现 GBN/SR 模拟器**

以下是一个完整的 Python 模拟器，可以模拟 GBN 和 SR 协议在不同信道条件下的性能表现。

```python
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


class ProtocolType(Enum):
    GBN = "Go-Back-N"
    SR = "Selective Repeat"


@dataclass
class Frame:
    seq_num: int
    send_time: float
    is_ack: bool = False
    ack_num: int = -1


@dataclass
class Timer:
    seq_num: int
    expire_time: float
    active: bool = True


class Channel:
    """模拟有噪声的信道"""

    def __init__(self, delay: float, loss_rate: float, corrupt_rate: float = 0.0):
        self.delay = delay  # 单程传播延迟（秒）
        self.loss_rate = loss_rate  # 帧丢失率
        self.corrupt_rate = corrupt_rate  # 帧损坏率

    def transmit(self, frame: Frame, current_time: float) -> Optional[Frame]:
        """模拟帧通过信道传输，返回到达对端的帧（或 None 如果丢失）"""
        if random.random() < self.loss_rate:
            return None  # 帧丢失
        if random.random() < self.corrupt_rate:
            return None  # 帧损坏（简化为等同于丢失）
        arrival_time = current_time + self.delay
        frame_copy = Frame(
            seq_num=frame.seq_num,
            send_time=arrival_time,
            is_ack=frame.is_ack,
            ack_num=frame.ack_num,
        )
        return frame_copy


class GBNSender:
    """GBN 发送方"""

    def __init__(self, window_size: int, num_bits: int, timeout: float):
        self.N = window_size
        self.num_bits = num_bits
        self.max_seq = 2**num_bits
        self.timeout = timeout
        self.base = 0
        self.next_seq = 0
        self.timer: Optional[Timer] = None
        self.frames_sent = 0
        self.retransmissions = 0

    def can_send(self) -> bool:
        return (self.next_seq - self.base) % self.max_seq < self.N

    def send_frame(self, current_time: float) -> Optional[Frame]:
        if not self.can_send():
            return None
        frame = Frame(seq_num=self.next_seq, send_time=current_time)
        self.next_seq = (self.next_seq + 1) % self.max_seq
        self.frames_sent += 1
        if self.timer is None:
            self.timer = Timer(self.base, current_time + self.timeout)
        return frame

    def receive_ack(self, ack_num: int, current_time: float):
        self.base = (ack_num + 1) % self.max_seq
        if self.base == self.next_seq:
            self.timer = None
        else:
            self.timer = Timer(self.base, current_time + self.timeout)

    def check_timeout(self, current_time: float) -> List[Frame]:
        if self.timer and current_time >= self.timer.expire_time:
            retransmit_frames = []
            seq = self.base
            while seq != self.next_seq:
                frame = Frame(seq_num=seq, send_time=current_time)
                retransmit_frames.append(frame)
                self.retransmissions += 1
                seq = (seq + 1) % self.max_seq
            self.timer = Timer(self.base, current_time + self.timeout)
            return retransmit_frames
        return []


class GBNReceiver:
    """GBN 接收方"""

    def __init__(self, num_bits: int):
        self.num_bits = num_bits
        self.max_seq = 2**num_bits
        self.expected_seq = 0
        self.frames_received = 0

    def receive_frame(self, frame: Frame) -> Optional[Frame]:
        if frame.seq_num == self.expected_seq:
            self.expected_seq = (self.expected_seq + 1) % self.max_seq
            self.frames_received += 1
            return Frame(
                seq_num=0,
                send_time=0,
                is_ack=True,
                ack_num=frame.seq_num,
            )
        else:
            if self.expected_seq > 0:
                return Frame(
                    seq_num=0,
                    send_time=0,
                    is_ack=True,
                    ack_num=(self.expected_seq - 1) % self.max_seq,
                )
            return None


class SRSender:
    """SR 发送方"""

    def __init__(self, window_size: int, num_bits: int, timeout: float):
        self.N = window_size
        self.num_bits = num_bits
        self.max_seq = 2**num_bits
        self.timeout = timeout
        self.base = 0
        self.next_seq = 0
        self.acked: Dict[int, bool] = {}
        self.timers: Dict[int, Timer] = {}
        self.frames_sent = 0
        self.retransmissions = 0

    def can_send(self) -> bool:
        return (self.next_seq - self.base) % self.max_seq < self.N

    def send_frame(self, current_time: float) -> Optional[Frame]:
        if not self.can_send():
            return None
        seq = self.next_seq
        frame = Frame(seq_num=seq, send_time=current_time)
        self.acked[seq] = False
        self.timers[seq] = Timer(seq, current_time + self.timeout)
        self.next_seq = (self.next_seq + 1) % self.max_seq
        self.frames_sent += 1
        return frame

    def receive_ack(self, ack_num: int, current_time: float):
        self.acked[ack_num] = True
        if ack_num in self.timers:
            del self.timers[ack_num]
        while self.base in self.acked and self.acked[self.base]:
            del self.acked[self.base]
            self.base = (self.base + 1) % self.max_seq

    def check_timeout(self, current_time: float) -> List[Frame]:
        retransmit = []
        expired = []
        for seq, timer in self.timers.items():
            if timer.active and current_time >= timer.expire_time:
                expired.append(seq)
        for seq in expired:
            frame = Frame(seq_num=seq, send_time=current_time)
            retransmit.append(frame)
            self.timers[seq] = Timer(seq, current_time + self.timeout)
            self.retransmissions += 1
        return retransmit


class SRReceiver:
    """SR 接收方"""

    def __init__(self, window_size: int, num_bits: int):
        self.W = window_size
        self.num_bits = num_bits
        self.max_seq = 2**num_bits
        self.base = 0
        self.received: Dict[int, bool] = {}
        self.frames_delivered = 0

    def receive_frame(self, frame: Frame) -> Optional[Frame]:
        seq = frame.seq_num
        diff = (seq - self.base) % self.max_seq
        if diff < self.W and seq not in self.received:
            self.received[seq] = True
            while self.base in self.received and self.received[self.base]:
                del self.received[self.base]
                self.frames_delivered += 1
                self.base = (self.base + 1) % self.max_seq
            return Frame(
                seq_num=0, send_time=0, is_ack=True, ack_num=seq
            )
        return Frame(
            seq_num=0, send_time=0, is_ack=True, ack_num=seq
        )


def simulate(
    protocol: ProtocolType,
    window_size: int,
    num_frames: int,
    num_bits: int,
    timeout: float,
    channel_delay: float,
    loss_rate: float,
    max_time: float = 10000.0,
) -> dict:
    """运行协议模拟"""
    channel = Channel(delay=channel_delay, loss_rate=loss_rate)

    if protocol == ProtocolType.GBN:
        sender = GBNSender(window_size, num_bits, timeout)
        receiver = GBNReceiver(num_bits)
    else:
        sender = SRSender(window_size, num_bits, timeout)
        receiver = SRReceiver(window_size, num_bits)

    events: List[tuple] = []
    current_time = 0.0
    dt = 0.01

    results = {
        "throughput": 0,
        "utilization": 0,
        "retransmissions": 0,
        "total_sent": 0,
        "frames_delivered": 0,
        "time": 0,
    }

    while receiver.frames_delivered < num_frames and current_time < max_time:
        while sender.can_send() and sender.next_seq < num_frames:
            frame = sender.send_frame(current_time)
            if frame:
                transmitted = channel.transmit(frame, current_time)
                if transmitted:
                    events.append(("data", transmitted))

        timeout_frames = sender.check_timeout(current_time)
        for frame in timeout_frames:
            transmitted = channel.transmit(frame, current_time)
            if transmitted:
                events.append(("data", transmitted))

        new_events = []
        for event_type, event_frame in events:
            if event_frame.send_time <= current_time:
                if not event_frame.is_ack:
                    ack = receiver.receive_frame(event_frame)
                    if ack:
                        transmitted = channel.transmit(ack, current_time)
                        if transmitted:
                            new_events.append(("ack", transmitted))
                else:
                    sender.receive_ack(event_frame.ack_num, current_time)
            else:
                new_events.append((event_type, event_frame))
        events = new_events

        current_time += dt

    results["throughput"] = receiver.frames_delivered / current_time
    t_frame = 0.001
    results["utilization"] = min(
        1.0, (window_size * t_frame) / (t_frame + 2 * channel_delay)
    )
    results["retransmissions"] = sender.retransmissions
    results["total_sent"] = sender.frames_sent
    results["frames_delivered"] = receiver.frames_delivered
    results["time"] = current_time
    return results


def run_comparison():
    """运行 GBN 和 SR 的性能对比"""
    num_frames = 100
    num_bits = 4
    timeout = 2.0
    channel_delay = 0.5

    loss_rates = [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3]
    window_sizes = [1, 2, 4, 8]

    print("=" * 70)
    print("滑动窗口协议性能模拟器")
    print("=" * 70)
    print(f"参数：帧数={num_frames}, 序号比特={num_bits}, 超时={timeout}s")
    print(f"      信道延迟={channel_delay}s")
    print()

    print("-" * 70)
    print("实验1：不同窗口大小下的 GBN 性能（丢包率=5%）")
    print("-" * 70)
    print(f"{'窗口大小':<10} {'吞吐量(frame/s)':<18} {'重传次数':<12} {'总发送':<10}")
    for ws in window_sizes:
        if ws <= 2**num_bits - 1:
            result = simulate(
                ProtocolType.GBN,
                ws,
                num_frames,
                num_bits,
                timeout,
                channel_delay,
                0.05,
            )
            print(
                f"{ws:<10} {result['throughput']:<18.4f} "
                f"{result['retransmissions']:<12} {result['total_sent']:<10}"
            )

    print()
    print("-" * 70)
    print("实验2：不同丢包率下 GBN vs SR 对比（窗口大小=7）")
    print("-" * 70)
    print(f"{'丢包率':<10} {'GBN吞吐量':<15} {'SR吞吐量':<15} {'GBN重传':<10} {'SR重传':<10}")
    for lr in loss_rates:
        gbn = simulate(
            ProtocolType.GBN, 7, num_frames, num_bits, timeout, channel_delay, lr
        )
        sr = simulate(
            ProtocolType.SR, 4, num_frames, num_bits, timeout, channel_delay, lr
        )
        print(
            f"{lr:<10.0%} {gbn['throughput']:<15.4f} {sr['throughput']:<15.4f} "
            f"{gbn['retransmissions']:<10} {sr['retransmissions']:<10}"
        )

    print()
    print("-" * 70)
    print("实验3：信道利用率理论值 vs 模拟值")
    print("-" * 70)
    t_frame = 0.001
    print(f"{'窗口大小':<10} {'理论利用率':<15} {'模拟利用率':<15}")
    for ws in [1, 2, 4, 8, 16]:
        theory = min(1.0, (ws * t_frame) / (t_frame + 2 * channel_delay))
        if ws <= 2**num_bits - 1:
            result = simulate(
                ProtocolType.GBN,
                ws,
                num_frames,
                num_bits,
                timeout,
                channel_delay,
                0.0,
            )
            print(f"{ws:<10} {theory:<15.4f} {result['utilization']:<15.4f}")


if __name__ == "__main__":
    run_comparison()
```

#### **模拟结果分析**

运行上述模拟器，可以观察到以下关键现象：

**1. 窗口大小对吞吐量的影响**

当窗口大小从 1 增大到 $2^m - 1$ 时，吞吐量近似线性增长。但当窗口足够大（填满管道）后，继续增大窗口不再提升吞吐量。

```
吞吐量
  ^
  │            ┌──────────────── 理论上限
  │           /
  │          /
  │         /
  │        /
  │       /
  │      /
  │     /
  │    /
  │   /
  │  /
  │ /
  │/
  └──────────────────────────→ 窗口大小 N
  1    4    8   16   32
```

**2. 丢包率对 GBN 和 SR 的影响**

在低丢包率时，GBN 和 SR 性能接近。随着丢包率增大，SR 的优势越来越明显，因为 GBN 需要重传整个窗口。

```
吞吐量
  ^
  │\
  │ \  SR
  │  \
  │   \_________
  │    \
  │     \ GBN
  │      \
  │       ↘___________
  │
  └──────────────────────→ 丢包率
  0%    5%   10%   20%   30%
```

**3. 超时时间的选择**

超时时间对性能的影响呈 U 型曲线：
- 太短：大量虚假重传，浪费带宽
- 太长：帧丢失时等待过久，吞吐量下降
- 最优值：略大于 RTT

<div data-component="ProtocolPerformanceSimulator"></div>

---

## 5.10 本章小结

### 关键概念回顾

| 概念 | 核心要点 |
|------|---------|
| 停等协议 | 发一帧等一个 ACK，简单但效率低 |
| 超时重传 | 解决帧丢失问题的核心机制 |
| 1比特序号 | 停等协议只需 1 比特序号即可区分新旧帧 |
| 流水线 | 允许多帧同时在管道中，提高效率 |
| 发送窗口 | 允许连续发送的帧范围 |
| 接收窗口 | 愿意接受的帧序号范围 |
| GBN | 累计确认，丢弃乱序帧，超时重传整个窗口 |
| SR | 选择确认，缓存乱序帧，超时只重传一帧 |
| GBN 约束 | $N \leq 2^m - 1$ |
| SR 约束 | $W_{send} + W_{recv} \leq 2^m$ |
| 链路利用率 | $U = \frac{L/R}{L/R + RTT}$ |
| 带宽延迟积 | $BDP = R \times RTT$，管道容量 |

### 协议选择决策树

```
                    信道质量如何？
                   /              \
              好（低误码率）    差（高误码率）
                 |                  |
           延迟大？              使用 SR
           /      \              （精确重传）
         是        否
          |         |
     需要大窗口   停等协议
      GBN/SR     足够高效
```

### 从链路层到传输层

滑动窗口协议不仅是链路层的核心技术，其思想直接延续到传输层的 TCP 协议中：

- TCP 的序号是**字节序号**而非帧序号
- TCP 的窗口大小可达 $2^{30}$ 字节（约 1 GB）
- TCP 的 SACK 选项是 SR 思想的实现
- TCP 的 RTO 计算使用 Jacobson/Karels 算法
- TCP 的拥塞控制在滑动窗口基础上增加了拥塞窗口

理解本章的滑动窗口协议，是理解 TCP 可靠传输和拥塞控制的基础。

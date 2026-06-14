# Chapter 29: 网络性能分析与测量

> **学习目标**：
> - 掌握网络性能的核心指标体系：吞吐量、延迟、抖动、丢包率和带宽延迟积
> - 理解 Little 定律的数学本质及其在排队系统和网络测量中的应用
> - 掌握 M/M/1、M/M/1/K、M/M/c 排队模型的推导与计算
> - 熟练使用 ping、traceroute、iperf3、Wireshark、tcpdump 等网络测量工具
> - 了解网络层析成像（Network Tomography）的基本原理与方法
> - 掌握分层故障排查方法论与性能排障流程
> - 能够运用排队论分析路由器缓冲区设计和服务器负载

---

## 29.1 网络性能指标体系

网络性能的量化评估是网络工程的基础。我们需要一套完整的指标体系来描述网络的健康状况。

### 29.1.1 吞吐量（Throughput）

吞吐量是指单位时间内成功传输的数据量，通常以 bps（bits per second）为单位。

#### **瞬时吞吐量**

瞬时吞吐量是在某一时刻观测到的数据传输速率：

$$S_{inst}(t) = \frac{\Delta D(t)}{\Delta t}$$

其中 $\Delta D(t)$ 是在时间窗口 $\Delta t$ 内传输的数据量。

#### **平均吞吐量**

在时间段 $[t_1, t_2]$ 内的平均吞吐量：

$$S_{avg} = \frac{\int_{t_1}^{t_2} S_{inst}(t) \, dt}{t_2 - t_1} = \frac{D_{total}}{t_2 - t_1}$$

#### **瓶颈吞吐量**

端到端吞吐量受路径上最慢链路的限制：

$$S_{e2e} = \min(C_1, C_2, \ldots, C_n)$$

其中 $C_i$ 是第 $i$ 条链路的容量。这就是所谓的**瓶颈链路**（bottleneck link）。

实际吞吐量还受以下因素影响：

$$S_{actual} = \min\left(C_{bottleneck}, \frac{W_{max}}{RTT}\right)$$

其中 $W_{max}$ 是发送窗口大小。

### 29.1.2 延迟（Delay）

端到端延迟由四个部分组成：

$$d_{end-to-end} = d_{proc} + d_{queue} + d_{trans} + d_{prop}$$

#### **处理延迟（Processing Delay）**

路由器检查分组头部并决定转发路径的时间：

- 典型值：微秒级（μs）
- 包括：查找转发表、差错检验、TTL 递减等

#### **排队延迟（Queuing Delay）**

分组在路由器缓冲区中等待发送的时间：

$$d_{queue} = \frac{L}{R} \cdot \frac{1}{1 - \rho} \cdot \rho$$

（M/M/1 模型下的期望排队延迟，后续详细推导）

#### **传输延迟（Transmission Delay）**

将分组的所有比特推送到链路上的时间：

$$d_{trans} = \frac{L}{R}$$

其中 $L$ 是分组长度（bits），$R$ 是链路带宽（bps）。

#### **传播延迟（Propagation Delay）**

信号在物理介质中传播的时间：

$$d_{prop} = \frac{d}{s}$$

其中 $d$ 是链路长度（m），$s$ 是信号传播速度（光纤中约 $2 \times 10^8$ m/s）。

<div data-component="NetworkDelaySimulator"></div>

### 29.1.3 抖动（Jitter）

抖动是延迟的变化量，也称为**包延迟变异**（Packet Delay Variation, PDV）。

#### **PDV 的定义**

RFC 3393 定义了 PDV 的测量方法。对于连续发送的分组序列，PDV 定义为：

$$PDV_i = d_i - d_{ref}$$

其中 $d_i$ 是第 $i$ 个分组的延迟，$d_{ref}$ 是参考分组（通常选第一个或延迟最小的分组）的延迟。

#### **统计指标**

$$J_{mean} = \frac{1}{n-1}\sum_{i=2}^{n}|d_i - d_{i-1}|$$

这是 Cisco 路由器中常用的抖动计算方法（间隔抖动）。

$$J_{std} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(d_i - \bar{d})^2}$$

标准差形式的抖动衡量延迟的离散程度。

#### **抖动对实时应用的影响**

| 应用类型 | 可接受抖动 | 说明 |
|---------|-----------|------|
| VoIP | < 30ms | 超过会导致语音断裂 |
| 视频会议 | < 50ms | 超过会导致画面卡顿 |
| 在线游戏 | < 20ms | 超过会导致操作延迟 |
| 直播流媒体 | < 100ms | 通过缓冲区补偿 |

### 29.1.4 丢包率（Packet Loss Rate）

$$PLR = \frac{N_{lost}}{N_{sent}} \times 100\%$$

丢包的主要原因：

1. **缓冲区溢出**：队列满时新到达的分组被丢弃（最常见）
2. **差错检测**：CRC 校验失败的分组被丢弃
3. **TTL 超时**：TTL 减为 0 时分组被丢弃
4. **主动队列管理（AQM）**：如 RED（Random Early Detection）主动丢包

### 29.1.5 带宽延迟积（Bandwidth-Delay Product）

$$BDP = C \times RTT$$

其中 $C$ 是链路带宽，$RTT$ 是往返时间。

BDP 的物理意义：**网络管道中能容纳的最大数据量**。

例如，一条 1 Gbps 链路的 RTT 为 100ms：

$$BDP = 1 \times 10^9 \times 0.1 = 100 \text{ Mbits} = 12.5 \text{ MB}$$

这意味着要充分利用这条链路，TCP 窗口至少需要 12.5 MB。

#### **BDP 与 TCP 性能**

$$S_{max} = \frac{W}{RTT}$$

当 $W < BDP$ 时，链路未被充分利用；当 $W \geq BDP$ 时，达到最大吞吐量。

<div data-component="BDPCalculator"></div>

---

## 29.2 Little 定律

Little 定律是排队论中最基本、最通用的定理之一。

### 29.2.1 定律陈述

$$L = \lambda W$$

- $L$：系统中平均实体数（队列中的平均顾客数）
- $\lambda$：平均到达率
- $W$：每个实体在系统中的平均逗留时间

### 29.2.2 数学推导

考虑一个稳定运行的排队系统，在时间区间 $[0, T]$ 内观测。

设 $A(t)$ 为时间 $[0, t]$ 内到达的顾客总数，$D(t)$ 为时间 $[0, t]$ 内离开的顾客总数。

系统中的顾客数为：

$$N(t) = A(t) - D(t)$$

在 $[0, T]$ 内的平均顾客数：

$$L_T = \frac{1}{T}\int_0^T N(t) \, dt = \frac{1}{T}\int_0^T [A(t) - D(t)] \, dt$$

设第 $i$ 个顾客的到达时间为 $a_i$，离开时间为 $d_i$，则该顾客在系统中的逗留时间为 $w_i = d_i - a_i$。

将积分按顾客分解：

$$\int_0^T N(t) \, dt = \sum_{i=1}^{A(T)} w_i$$

因此：

$$L_T = \frac{A(T)}{T} \cdot \frac{1}{A(T)}\sum_{i=1}^{A(T)} w_i$$

当 $T \to \infty$ 时：

- $\frac{A(T)}{T} \to \lambda$（到达率）
- $\frac{1}{A(T)}\sum w_i \to W$（平均逗留时间）

$$L = \lambda W$$

### 29.2.3 在排队系统中的应用

对于 M/M/1 队列：

- $L = \frac{\rho}{1-\rho}$（系统中的平均顾客数）
- $\lambda$ = 到达率
- $W = \frac{L}{\lambda} = \frac{1}{\mu - \lambda}$（平均逗留时间）

验证：

$$W = \frac{\rho / (1-\rho)}{\lambda} = \frac{\lambda / \mu}{(1 - \lambda/\mu) \cdot \lambda} = \frac{1}{\mu - \lambda} \quad \checkmark$$

### 29.2.4 在网络测量中的实用价值

Little 定律在网络中的实际应用：

**1. TCP 连接中的在途数据量**

$$L_{inflight} = \lambda_{throughput} \times RTT$$

这就是 TCP 拥塞窗口（cwnd）应该设置的值。

**2. 路由器缓冲区大小**

$$B = C \times RTT = BDP$$

对于 $N$ 条 TCP 流共享的链路：

$$B \geq \frac{RTT \times C}{\sqrt{N}}$$

这就是 Appenzeller 的缓冲区规则。

**3. 服务器并发连接数**

$$L_{conn} = \lambda_{req} \times T_{service}$$

这是 Web 服务器容量规划的基础。

---

## 29.3 排队论基础

### 29.3.1 M/M/1 模型

M/M/1 是最基础的排队模型，其假设如下：

- **到达过程**：泊松过程，到达率 $\lambda$
- **服务时间**：指数分布，服务率 $\mu$
- **服务台数量**：1
- **队列容量**：无限
- **顾客来源**：无限

#### **状态转移图**

系统状态为队列中的顾客数 $n$，状态转移率为：

$$n \xrightarrow{\lambda} n+1 \quad \text{（到达）}$$
$$n+1 \xrightarrow{\mu} n \quad \text{（离开）}$$

#### **稳态概率推导**

在稳态下，进入状态 $n$ 的速率等于离开状态 $n$ 的速率（**平衡方程**）：

对于状态 0：
$$\lambda P(0) = \mu P(1)$$

对于状态 $n \geq 1$：
$$\lambda P(n-1) + \mu P(n+1) = (\lambda + \mu) P(n)$$

即：
$$\lambda P(n-1) = \mu P(n)$$

递推求解：
$$P(1) = \frac{\lambda}{\mu} P(0) = \rho P(0)$$
$$P(2) = \rho P(1) = \rho^2 P(0)$$
$$P(n) = \rho^n P(0)$$

由归一化条件 $\sum_{n=0}^{\infty} P(n) = 1$：

$$P(0) \sum_{n=0}^{\infty} \rho^n = P(0) \cdot \frac{1}{1-\rho} = 1$$

因此：
$$P(0) = 1 - \rho$$
$$\boxed{P(n) = (1-\rho)\rho^n}$$

其中 $\rho = \frac{\lambda}{\mu}$ 为**利用率**，要求 $\rho < 1$（否则队列无限增长）。

#### **关键性能指标**

**平均队列长度**（等待中的顾客数，不含正在服务的）：

$$L_q = \sum_{n=1}^{\infty} (n-1) P(n) = \sum_{n=1}^{\infty} (n-1)(1-\rho)\rho^n$$

令 $k = n-1$：

$$L_q = (1-\rho)\rho \sum_{k=0}^{\infty} k \rho^k = (1-\rho)\rho \cdot \frac{\rho}{(1-\rho)^2}$$

$$\boxed{L_q = \frac{\rho^2}{1-\rho}}$$

**系统中的平均顾客数**（含正在服务的）：

$$L = L_q + \rho = \frac{\rho^2}{1-\rho} + \rho = \frac{\rho}{1-\rho}$$

**平均等待时间**（队列中的等待时间，不含服务时间）：

由 Little 定律 $L_q = \lambda W_q$：

$$\boxed{W_q = \frac{L_q}{\lambda} = \frac{\rho}{\mu - \lambda}}$$

**平均逗留时间**（等待时间 + 服务时间）：

$$W = W_q + \frac{1}{\mu} = \frac{1}{\mu - \lambda}$$

**用 Little 定律验证**：

$$L = \lambda W = \lambda \cdot \frac{1}{\mu - \lambda} = \frac{\lambda}{\mu - \lambda} = \frac{\rho}{1-\rho} \quad \checkmark$$

<div data-component="QueuingSystemSimulator"></div>

### 29.3.2 M/M/1/K 模型

M/M/1/K 模型假设系统容量为 $K$（队列 + 正在服务的顾客总数不超过 $K$）。

#### **稳态概率**

平衡方程的解为：

$$P(n) = \frac{(1-\rho)\rho^n}{1-\rho^{K+1}}, \quad 0 \leq n \leq K$$

#### **阻塞概率**

当系统满时，新到达的顾客被拒绝：

$$P_{block} = P(K) = \frac{(1-\rho)\rho^K}{1-\rho^{K+1}}$$

#### **有效到达率**

由于阻塞，有效到达率为：

$$\lambda_{eff} = \lambda(1 - P(K))$$

#### **吞吐量**

$$\text{Throughput} = \mu(1 - P(0)) = \lambda_{eff}$$

#### **M/M/1/K 的 Little 定律应用**

$$L = \sum_{n=0}^{K} n \cdot P(n)$$

$$W = \frac{L}{\lambda_{eff}}$$

### 29.3.3 M/M/c 模型

M/M/c 模型有 $c$ 个并行服务台，适用于多核服务器、多链路聚合等场景。

#### **Erlang C 公式**

等待概率（顾客到达时所有服务台都繁忙）：

$$C(c, \rho) = \frac{\frac{(c\rho)^c}{c!} \cdot \frac{1}{1-\rho}}{\sum_{k=0}^{c-1}\frac{(c\rho)^k}{k!} + \frac{(c\rho)^c}{c!} \cdot \frac{1}{1-\rho}}$$

其中 $\rho = \frac{\lambda}{c\mu}$（每个服务台的利用率），要求 $\rho < 1$。

#### **平均等待时间**

$$W_q = \frac{C(c, \rho)}{c\mu - \lambda}$$

#### **平均队列长度**

$$L_q = \lambda W_q = \frac{\rho \cdot C(c, \rho)}{1 - \rho}$$

#### **Erlang B 公式（M/M/c/c 模型）**

当系统容量等于服务台数（无等待队列）时的阻塞概率：

$$B(c, a) = \frac{\frac{a^c}{c!}}{\sum_{k=0}^{c}\frac{a^k}{k!}}$$

其中 $a = \frac{\lambda}{\mu}$ 为**话务量强度**（Erlang）。

### 29.3.4 排队论在网络中的应用

#### **路由器缓冲区设计**

路由器缓冲区大小的选择直接影响网络性能：

1. **传统规则**：$B = C \times RTT = BDP$
2. **Appenzeller 规则**（多条 TCP 流共享）：$B = \frac{RTT \times C}{\sqrt{N}}$
3. **过小缓冲区**：导致不必要的丢包，降低吞吐量
4. **过大缓冲区**：增加延迟（Bufferbloat 问题）

#### **服务器负载分析**

假设 Web 服务器的请求到达服从泊松分布，服务时间服从指数分布：

- 到达率：$\lambda = 100$ 请求/秒
- 平均服务时间：$1/\mu = 8$ ms
- 利用率：$\rho = \lambda/\mu = 0.8$

则：
- 平均响应时间：$W = \frac{1}{\mu - \lambda} = \frac{1}{125 - 100} = 40$ ms
- 平均队列长度：$L = \frac{\rho}{1-\rho} = 4$
- 超过 100ms 响应时间的概率：$P(W > 0.1) = e^{-(\mu-\lambda) \times 0.1} = e^{-2.5} \approx 0.082$

---

## 29.4 网络测量工具详解

### 29.4.1 ping

ping 是最基本的网络连通性和延迟测量工具，基于 ICMP Echo Request/Reply。

#### **工作原理**

1. 发送端构造 ICMP Echo Request 报文（Type=8）
2. 接收端收到后返回 ICMP Echo Reply 报文（Type=0）
3. 发送端计算 RTT = 收到 Reply 的时间 - 发出 Request 的时间

#### **ICMP 报文结构**

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|     Type      |     Code      |          Checksum             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|        Identifier             |       Sequence Number         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Data ...                              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

#### **常用选项**

```bash
# 基本 ping
ping www.example.com

# 指定次数
ping -c 10 www.example.com

# 设置包大小（字节）
ping -s 1472 www.example.com

# Flood ping（需要 root，用于压力测试）
sudo ping -f www.example.com

# 设置间隔（秒）
ping -i 0.2 www.example.com

# 指定源地址
ping -I eth0 www.example.com

# 设置 TTL
ping -t 64 www.example.com
```

#### **ping 结果解读**

```
PING www.example.com (93.184.216.34): 56 data bytes
64 bytes from 93.184.216.34: icmp_seq=0 ttl=56 time=11.632 ms
64 bytes from 93.184.216.34: icmp_seq=1 ttl=56 time=11.726 ms
64 bytes from 93.184.216.34: icmp_seq=2 ttl=56 time=10.624 ms

--- www.example.com ping statistics ---
3 packets transmitted, 3 packets received, 0.0% packet loss
round-trip min/avg/max/stddev = 10.624/11.327/11.726/0.498 ms
```

#### **Python 实现 ping**

```python
import socket
import struct
import time
import os
import select

def checksum(data):
    """计算 ICMP 校验和"""
    if len(data) % 2:
        data += b'\x00'
    s = 0
    for i in range(0, len(data), 2):
        w = (data[i] << 8) + data[i+1]
        s += w
    s = (s >> 16) + (s & 0xffff)
    s += (s >> 16)
    return ~s & 0xffff

def create_icmp_packet(seq, payload_size=56):
    """构造 ICMP Echo Request 报文"""
    icmp_type = 8  # Echo Request
    icmp_code = 0
    icmp_id = os.getpid() & 0xFFFF
    icmp_seq = seq

    # ICMP 头部（8字节）+ 数据
    header = struct.pack('!BBHHH', icmp_type, icmp_code, 0, icmp_id, icmp_seq)
    data = b'Q' * payload_size

    # 计算校验和
    cksum = checksum(header + data)
    header = struct.pack('!BBHHH', icmp_type, icmp_code, cksum, icmp_id, icmp_seq)

    return header + data

def ping(dest_addr, count=4, timeout=1):
    """简单的 ping 实现"""
    try:
        dest_ip = socket.gethostbyname(dest_addr)
    except socket.gaierror:
        print(f"无法解析主机名: {dest_addr}")
        return

    icmp_socket = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
    icmp_socket.settimeout(timeout)

    rtts = []
    sent = 0
    received = 0

    print(f"PING {dest_addr} ({dest_ip}): 56 data bytes")

    for seq in range(count):
        packet = create_icmp_packet(seq)
        sent += 1

        send_time = time.time()
        icmp_socket.sendto(packet, (dest_ip, 0))

        try:
            ready = select.select([icmp_socket], [], [], timeout)
            if ready[0]:
                recv_packet, addr = icmp_socket.recvfrom(1024)
                recv_time = time.time()

                # 解析 ICMP Echo Reply
                icmp_header = recv_packet[20:28]
                icmp_type, code, _, recv_id, recv_seq = struct.unpack('!BBHHH', icmp_header)

                if icmp_type == 0 and recv_seq == seq:
                    rtt = (recv_time - send_time) * 1000  # 毫秒
                    rtts.append(rtt)
                    received += 1
                    print(f"64 bytes from {dest_ip}: icmp_seq={seq} "
                          f"ttl=64 time={rtt:.3f} ms")
            else:
                print(f"Request timeout for icmp_seq {seq}")
        except socket.timeout:
            print(f"Request timeout for icmp_seq {seq}")

        time.sleep(1)

    icmp_socket.close()

    # 统计结果
    print(f"\n--- {dest_addr} ping statistics ---")
    loss_rate = (sent - received) / sent * 100
    print(f"{sent} packets transmitted, {received} received, {loss_rate:.1f}% packet loss")

    if rtts:
        print(f"rtt min/avg/max = {min(rtts):.3f}/{sum(rtts)/len(rtts):.3f}/"
              f"{max(rtts):.3f} ms")

if __name__ == '__main__':
    ping('8.8.8.8', count=5)
```

### 29.4.2 traceroute

traceroute 通过递增 TTL 来发现到目的地路径上的每一跳路由器。

#### **TTL 递增原理**

1. 发送 TTL=1 的分组 → 第一跳路由器返回 ICMP Time Exceeded
2. 发送 TTL=2 的分组 → 第二跳路由器返回 ICMP Time Exceeded
3. ... 以此类推，直到到达目的地

#### **每跳 RTT 测量**

traceroute 对每一跳发送 3 个探测包，显示每个包的 RTT：

```
traceroute to www.example.com (93.184.216.34), 30 hops max, 60 byte packets
 1  gateway (192.168.1.1)  1.234 ms  1.123 ms  1.089 ms
 2  10.0.0.1 (10.0.0.1)  5.456 ms  5.234 ms  5.345 ms
 3  * * *
 4  72.14.215.85 (72.14.215.85)  12.345 ms  12.234 ms  12.456 ms
```

`* * *` 表示该跳不响应（防火墙丢弃或不返回 ICMP）。

#### **UDP/ICMP 变体**

- **Unix/Linux 默认**：UDP 探测包（目标端口从 33434 开始递增）
- **Windows tracert**：ICMP Echo Request
- **tcptraceroute**：TCP SYN 包（能穿越只允许 TCP 的防火墙）

#### **Paris Traceroute**

传统 traceroute 的问题：多路径环境下，不同 TTL 的包可能走不同路径。

Paris Traceroute 通过固定 IP 头部的关键字段（如 UDP 端口或 ICMP 序列号），确保所有探测包走同一条路径，从而获得一致的路径视图。

#### **Python 实现 traceroute**

```python
import socket
import struct
import time
import select

def traceroute(dest_addr, max_hops=30, timeout=2, probes=3):
    """简单的 traceroute 实现"""
    try:
        dest_ip = socket.gethostbyname(dest_addr)
    except socket.gaierror:
        print(f"无法解析主机名: {dest_addr}")
        return

    print(f"traceroute to {dest_addr} ({dest_ip}), {max_hops} hops max")

    for ttl in range(1, max_hops + 1):
        rtts = []
        last_addr = None

        for probe in range(probes):
            # 创建 UDP 发送套接字
            send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            send_sock.setsockopt(socket.IPPROTO_IP, socket.IP_TTL, ttl)

            # 创建 ICMP 接收套接字
            recv_sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
            recv_sock.settimeout(timeout)

            # 绑定到一个临时端口
            port = 33434 + ttl * probes + probe

            send_time = time.time()
            send_sock.sendto(b'', (dest_ip, port))

            try:
                ready = select.select([recv_sock], [], [], timeout)
                if ready[0]:
                    data, addr = recv_sock.recvfrom(1024)
                    recv_time = time.time()
                    rtt = (recv_time - send_time) * 1000
                    rtts.append(rtt)
                    last_addr = addr[0]

                    # 检查是否到达目的地
                    icmp_type = data[20]
                    if icmp_type == 3:  # Destination Unreachable
                        break
                else:
                    rtts.append(None)
            except socket.timeout:
                rtts.append(None)
            finally:
                send_sock.close()
                recv_sock.close()

        # 输出该跳的结果
        if last_addr:
            rtt_strs = [f"{r:.3f} ms" if r else "*" for r in rtts]
            try:
                hostname = socket.gethostbyaddr(last_addr)[0]
                print(f"{ttl:2d}  {hostname} ({last_addr})  {'  '.join(rtt_strs)}")
            except socket.herror:
                print(f"{ttl:2d}  {last_addr}  {'  '.join(rtt_strs)}")
        else:
            print(f"{ttl:2d}  * * *")

        # 检查是否到达目的地
        if last_addr == dest_ip:
            break

if __name__ == '__main__':
    traceroute('8.8.8.8')
```

### 29.4.3 iperf3

iperf3 是网络带宽测量的标准工具，支持 TCP 和 UDP 测试。

#### **TCP 带宽测试**

```bash
# 服务端启动
iperf3 -s

# 客户端测试（默认 10 秒）
iperf3 -c server_ip

# 指定测试时间
iperf3 -c server_ip -t 30

# 并行流（多连接）
iperf3 -c server_ip -P 4

# 反向测试（服务端发送）
iperf3 -c server_ip -R

# 指定窗口大小
iperf3 -c server_ip -w 256K

# 双向测试
iperf3 -c server_ip --bidir

# 绑定 CPU
iperf3 -c server_ip -A 0
```

#### **UDP 测试**

```bash
# UDP 指定带宽
iperf3 -c server_ip -u -b 100M

# UDP 双向
iperf3 -c server_ip -u -b 100M --bidir
```

#### **JSON 输出分析**

```bash
iperf3 -c server_ip -J > result.json
```

iperf3 TCP 输出解读：

```
[ ID] Interval           Transfer     Bitrate         Retr  Cwnd
[  5]   0.00-1.00   sec   112 MBytes   940 Mbits/sec    0   427 KBytes
[  5]   1.00-2.00   sec   112 MBytes   939 Mbits/sec    0   427 KBytes
[  5]   2.00-3.00   sec   112 MBytes   940 Mbits/sec    0   427 KBytes
...
[  5]   0.00-10.00  sec  1.09 GBytes   939 Mbits/sec    0             sender
[  5]   0.00-10.00  sec  1.09 GBytes   939 Mbits/sec                  receiver
```

- **Transfer**：该时间段内传输的数据量
- **Bitrate**：吞吐量
- **Retr**：TCP 重传次数
- **Cwnd**：拥塞窗口大小

#### **iperf3 带宽测量引擎**

iperf3 的核心测量逻辑：

1. **发送端**：使用定时器控制发送速率，记录每个时间窗口的发送量
2. **接收端**：统计接收的数据量，计算吞吐量
3. **统计窗口**：默认每秒报告一次
4. **最终报告**：汇总所有窗口的统计

<div data-component="BandwidthMeasurementEngine"></div>

#### **Python 带宽测量实现**

```python
import socket
import time
import threading

class ThroughputMeasurer:
    """简单的吞吐量测量工具"""

    def __init__(self):
        self.bytes_sent = 0
        self.bytes_received = 0
        self.intervals = []
        self.lock = threading.Lock()
        self.running = False

    def server(self, host='0.0.0.0', port=5201, duration=10):
        """测量接收端吞吐量"""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((host, port))
        server_sock.listen(1)

        print(f"服务端监听 {host}:{port}")
        conn, addr = server_sock.accept()
        print(f"客户端连接: {addr}")

        self.running = True
        start_time = time.time()
        interval_start = start_time
        interval_bytes = 0

        conn.settimeout(1.0)

        while self.running and (time.time() - start_time) < duration:
            try:
                data = conn.recv(65536)
                if not data:
                    break
                now = time.time()
                with self.lock:
                    self.bytes_received += len(data)
                    interval_bytes += len(data)

                # 每秒报告
                if now - interval_start >= 1.0:
                    throughput = interval_bytes * 8 / (now - interval_start) / 1e6
                    elapsed = now - start_time
                    self.intervals.append({
                        'time': elapsed,
                        'throughput_mbps': throughput
                    })
                    print(f"[{elapsed:.1f}-{elapsed+1:.1f}] sec  "
                          f"{interval_bytes/1e6:.1f} MBytes  "
                          f"{throughput:.1f} Mbits/sec")
                    interval_start = now
                    interval_bytes = 0
            except socket.timeout:
                continue

        elapsed = time.time() - start_time
        avg_throughput = self.bytes_received * 8 / elapsed / 1e6
        print(f"\n[  0.00-{elapsed:.2f}] sec  "
              f"{self.bytes_received/1e6:.1f} MBytes  "
              f"{avg_throughput:.1f} Mbits/sec  receiver")

        conn.close()
        server_sock.close()

    def client(self, host, port=5201, duration=10, buffer_size=65536):
        """测量发送端吞吐量"""
        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_sock.connect((host, port))
        print(f"连接到 {host}:{port}")

        data = b'X' * buffer_size
        self.running = True
        start_time = time.time()

        while self.running and (time.time() - start_time) < duration:
            try:
                client_sock.sendall(data)
                with self.lock:
                    self.bytes_sent += len(data)
            except BrokenPipeError:
                break

        elapsed = time.time() - start_time
        avg_throughput = self.bytes_sent * 8 / elapsed / 1e6
        print(f"\n[  0.00-{elapsed:.2f}] sec  "
              f"{self.bytes_sent/1e6:.1f} MBytes  "
              f"{avg_throughput:.1f} Mbits/sec  sender")

        client_sock.close()
```

### 29.4.4 Wireshark

Wireshark 是最强大的网络协议分析工具。

#### **捕获过滤器语法（BPF）**

捕获过滤器在数据包被捕获时应用，使用 BPF（Berkeley Packet Filter）语法：

```bash
# 只捕获 TCP 流量
tcp

# 只捕获特定主机
host 192.168.1.100

# 只捕获特定端口
port 80

# 只捕获特定网段
net 192.168.1.0/24

# 组合条件
host 192.168.1.100 and port 80
src host 10.0.0.1 and dst port 443
not arp and not dns

# 只捕获 HTTP GET 请求
tcp port 80 and tcp[((tcp[12:1] & 0xf0) >> 2):4] = 0x47455420
```

#### **显示过滤器语法**

显示过滤器在已捕获的数据包上应用：

```
# 协议过滤
tcp
udp
http
dns
icmp
tls

# IP 地址过滤
ip.addr == 192.168.1.100
ip.src == 10.0.0.1
ip.dst == 10.0.0.2

# 端口过滤
tcp.port == 80
tcp.dstport == 443
udp.port == 53

# HTTP 过滤
http.request.method == "GET"
http.response.code == 200
http.host == "www.example.com"
http.content_type contains "text"

# TCP 过滤
tcp.flags.syn == 1
tcp.flags.reset == 1
tcp.analysis.retransmission
tcp.analysis.duplicate_ack
tcp.window_size < 1024

# 组合条件
ip.addr == 192.168.1.100 and tcp.port == 80
(http.request or http.response) and ip.src == 10.0.0.1
tcp.analysis.retransmission and tcp.stream eq 5

# 时间过滤
frame.time >= "2024-01-01 12:00:00"
```

#### **协议解析树**

Wireshark 的协议解析器采用树形结构：

```
Ethernet II
├── Destination: aa:bb:cc:dd:ee:ff
├── Source: 11:22:33:44:55:66
└── Type: IPv4 (0x0800)
    Internet Protocol Version 4
    ├── Version: 4
    ├── Header Length: 20 bytes
    ├── Total Length: 60
    ├── TTL: 64
    ├── Protocol: TCP (6)
    └── Source/Destination
        Transmission Control Protocol
        ├── Source Port: 54321
        ├── Destination Port: 80
        ├── Sequence Number: 1
        ├── Flags: 0x002 (SYN)
        └── Window Size: 65535
            Hypertext Transfer Protocol
            ├── GET /index.html HTTP/1.1
            └── Host: www.example.com
```

<div data-component="PacketCaptureEngine"></div>

#### **Wireshark IO 图表**

Wireshark 的 IO 图表功能可以可视化网络流量模式：

Statistics → IO Graphs 可以设置：
- X 轴时间粒度（ms/s/min）
- Y 轴统计方式（packets/bytes/bits）
- 过滤条件（如只看 TCP 重传）

#### **流跟踪（Follow Stream）**

右键点击一个 TCP 包 → Follow → TCP Stream，可以看到完整的会话内容：

```
GET /index.html HTTP/1.1
Host: www.example.com
Connection: keep-alive
User-Agent: Mozilla/5.0

HTTP/1.1 200 OK
Content-Type: text/html
Content-Length: 1234

<!DOCTYPE html>...
```

### 29.4.5 tcpdump

tcpdump 是命令行抓包工具，适合远程服务器使用。

#### **基本用法**

```bash
# 抓取所有包
sudo tcpdump

# 指定接口
sudo tcpdump -i eth0

# 显示详细信息
sudo tcpdump -v -i eth0

# 显示包内容（十六进制 + ASCII）
sudo tcpdump -X -i eth0

# 显示绝对序列号
sudo tcpdump -S -i eth0

# 不解析主机名和端口名
sudo tcpdump -n -i eth0

# 保存到文件
sudo tcpdump -i eth0 -w capture.pcap

# 从文件读取
tcpdump -r capture.pcap

# 限制抓包数量
sudo tcpdump -i eth0 -c 100

# 显示时间戳
sudo tcpdump -tttt -i eth0
```

#### **过滤表达式**

```bash
# 按主机
sudo tcpdump host 192.168.1.100
sudo tcpdump src host 192.168.1.100
sudo tcpdump dst host 192.168.1.100

# 按端口
sudo tcpdump port 80
sudo tcpdump src port 443
sudo tcpdump dst port 53

# 按协议
sudo tcpdump tcp
sudo tcpdump udp
sudo tcpdump icmp
sudo tcpdump arp

# 按网络
sudo tcpdump net 192.168.1.0/24

# 组合
sudo tcpdump 'host 192.168.1.100 and port 80'
sudo tcpdump 'tcp and port 80 and host 10.0.0.1'
sudo tcpdump 'not arp and not port 22'

# 高级过滤
sudo tcpdump 'tcp[tcpflags] & (tcp-syn) != 0'  # SYN 包
sudo tcpdump 'tcp[tcpflags] & (tcp-rst) != 0'  # RST 包
sudo tcpdump 'tcp[tcpflags] & (tcp-fin) != 0'  # FIN 包
sudo tcpdump 'greater 1000'  # 大于 1000 字节的包
```

### 29.4.6 ss 和 netstat

```bash
# 显示所有 TCP 连接
ss -t

# 显示监听端口
ss -tln

# 显示所有连接和进程信息
ss -tunap

# 显示套接字统计信息
ss -s

# 显示 TCP 内部信息
ss -ti

# 显示拥塞控制信息
ss -ti | grep -E "cwnd|rtt"

# netstat 用法
netstat -tulnp          # 显示监听端口和进程
netstat -anp | grep ESTABLISHED  # 已建立的连接
netstat -s              # 协议统计
netstat -i              # 网络接口统计
```

### 29.4.7 NetFlow/sFlow

NetFlow 和 sFlow 是网络流量监控的两种主要技术。

#### **NetFlow**

NetFlow 由 Cisco 开发，记录经过路由器的每条流（flow）：

一条 flow 由以下五元组定义：
- 源 IP 地址
- 目的 IP 地址
- 源端口
- 目的端口
- 协议号

NetFlow 记录包含：
- 流的起止时间
- 传输的字节数和包数
- TCP 标志
- ToS/DSCP 值

#### **sFlow**

sFlow 使用采样方式监控流量：

- 通常每 N 个包采样一个（如 1:1000）
- 采样数据包含包头和交换机/路由器的接口统计
- 适合高速网络（10Gbps+），开销比 NetFlow 低

#### **Flow 数据的分析应用**

1. **流量工程**：识别流量模式，优化链路负载
2. **安全检测**：发现 DDoS 攻击、异常流量
3. **容量规划**：基于历史数据预测带宽需求
4. **计费**：按流量计费的基础数据

---

## 29.5 网络层析成像（Network Tomography）

### 29.5.1 基本概念

网络层析成像是一种通过端到端测量来推断网络内部链路性能的技术，类似于医学中的 CT 扫描。

目标：在无法直接访问内部路由器的情况下，推断每条链路的：
- 丢包率
- 延迟分布
- 带宽利用率

### 29.5.2 多播方法

利用多播（multicast）的树形结构：

1. 源节点向多个接收者发送多播包
2. 每条链路上的丢包会影响该链路以下所有接收者
3. 通过接收者的丢包统计推断每条链路的丢包率

设链路 $i$ 的丢包率为 $p_i$，则从源到接收者 $j$ 的路径上的端到端丢包率：

$$P_j = 1 - \prod_{i \in path(j)} (1 - p_i)$$

### 29.5.3 单播方法

单播方法不需要网络支持多播，使用单播探测包：

1. **基于背靠背包对**：发送两个连续包到不同目的地，共享路径上的包具有相关性
2. **基于 TOMOGRAPHER 的方法**：利用最大似然估计推断链路丢包率
3. **压缩感知方法**：利用网络拓扑的稀疏性，用较少测量推断更多链路性能

### 29.5.4 实际应用

- **网络诊断**：定位性能瓶颈和故障链路
- **流量矩阵估计**：推断 OD（Origin-Destination）对之间的流量
- **服务质量监控**：SLA 验证

---

## 29.6 故障隔离方法

### 29.6.1 分层排查法

网络故障排查遵循 OSI 分层模型，从底层开始逐层排查：

```
┌─────────────────────────────────────────────────┐
│                  应用层 (L7)                      │
│  • 检查服务是否运行 (systemctl status)            │
│  • 检查端口监听 (ss -tlnp)                       │
│  • 检查应用日志 (/var/log/)                      │
├─────────────────────────────────────────────────┤
│                  传输层 (L4)                      │
│  • 检查 TCP 连接状态 (ss -tn)                    │
│  • 检查防火墙规则 (iptables -L)                  │
│  • 测试端口连通性 (nc -zv host port)             │
├─────────────────────────────────────────────────┤
│                  网络层 (L3)                      │
│  • 检查 IP 配置 (ip addr show)                   │
│  • 检查路由表 (ip route show)                    │
│  • 测试可达性 (ping)                             │
│  • 检查路径 (traceroute)                         │
├─────────────────────────────────────────────────┤
│               数据链路层 (L2)                     │
│  • 检查接口状态 (ip link show)                   │
│  • 检查 MAC 地址学习 (bridge fdb show)           │
│  • 检查 VLAN 配置                               │
├─────────────────────────────────────────────────┤
│                  物理层 (L1)                      │
│  • 检查网线连接                                  │
│  • 检查接口灯状态                                │
│  • 检查光功率 (ethtool)                          │
│  • 检查接口错误 (ip -s link)                     │
└─────────────────────────────────────────────────┘
```

### 29.6.2 二分搜索法

当路径较长时，使用二分搜索法加速定位：

1. 测试到路径中点的连通性
2. 如果中点可达，问题在后半段
3. 如果中点不可达，问题在前半段
4. 重复直到定位到故障点

```python
def binary_search_fault(path, is_reachable):
    """二分搜索法定位网络故障"""
    left, right = 0, len(path) - 1

    while left < right:
        mid = (left + right) // 2
        if is_reachable(path[mid]):
            left = mid + 1
        else:
            right = mid

    return path[left - 1] if left > 0 else None, path[left]
```

### 29.6.3 网络健康检查清单

```bash
# 1. 物理层检查
ip link show                          # 接口状态
ethtool eth0                          # 链路状态、速度
ip -s link show eth0                  # 错误计数

# 2. 数据链路层检查
arp -n                                # ARP 表
bridge fdb show                       # MAC 地址表（如果有交换功能）

# 3. 网络层检查
ip addr show                          # IP 地址
ip route show                         # 路由表
ping -c 3 gateway                     # 到网关的连通性
ping -c 3 8.8.8.8                     # 到外网的连通性
traceroute 8.8.8.8                    # 路径跟踪

# 4. 传输层检查
ss -tlnp                              # 监听端口
ss -tn state established              # 已建立的连接
iptables -L -n                        # 防火墙规则

# 5. 应用层检查
curl -v http://localhost:8080/health  # 应用健康检查
systemctl status myapp                # 服务状态
journalctl -u myapp --since "1h ago"  # 最近一小时的日志

# 6. 性能检查
ss -ti | grep cwnd                    # TCP 拥塞窗口
iperf3 -c target -t 10               # 带宽测试
ping -c 100 target                    # 延迟和丢包统计
```

<div data-component="NetworkFaultDiagnosticEngine"></div>

---

## 29.7 性能排障方法论

### 29.7.1 结构化排障流程

```
┌──────────────────────────────────────────────────────────────┐
│                     问题定义（Problem Definition）            │
│  • 明确症状：什么功能不工作？                                  │
│  • 确定范围：影响哪些用户/设备？                               │
│  • 量化指标：延迟多少？丢包率多少？                             │
│  • 确认时间：什么时候开始的？是间歇性的吗？                      │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                     信息收集（Information Gathering）          │
│  • 网络拓扑图                                                 │
│  • 最近的配置变更记录                                          │
│  • 监控系统数据（SNMP、NetFlow）                              │
│  • 用户报告                                                  │
│  • 网络测量工具数据                                           │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                     原因分析（Root Cause Analysis）            │
│  • 分层排查（物理→应用层）                                    │
│  • 二分法定位                                                 │
│  • 排队论模型分析                                             │
│  • 对比正常与异常状态的数据                                    │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                     方案制定（Solution Planning）              │
│  • 评估多个方案的可行性和风险                                  │
│  • 选择影响最小的方案                                         │
│  • 准备回退方案                                               │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                     实施验证（Implementation & Verification） │
│  • 在维护窗口实施变更                                         │
│  • 验证问题是否解决                                           │
│  • 监控是否有副作用                                           │
└──────────────────────┬───────────────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                     文档记录（Documentation）                  │
│  • 记录问题描述、原因和解决方案                                 │
│  • 更新网络文档                                               │
│  • 更新监控告警规则                                           │
│  • 知识库沉淀                                                 │
└──────────────────────────────────────────────────────────────┘
```

### 29.7.2 常见性能问题诊断

#### **问题 1：高延迟**

诊断步骤：
1. 使用 ping 测量 RTT
2. 使用 traceroute 定位高延迟的跳
3. 检查该跳路由器的 CPU 和队列长度
4. 分析是否有拥塞或配置问题

#### **问题 2：低吞吐量**

诊断步骤：
1. 使用 iperf3 测试基线带宽
2. 检查 TCP 窗口大小（`ss -ti`）
3. 检查是否有丢包（`netstat -s | grep retransmit`）
4. 使用 Wireshark 分析 TCP 流

```bash
# 检查 TCP 重传
netstat -s | grep -i retransmit

# 检查 TCP 窗口大小
ss -ti dst 10.0.0.1

# 检查网络接口错误
ip -s link show eth0
```

#### **问题 3：间歇性丢包**

诊断步骤：
1. 长时间 ping 收集统计数据
2. 分析丢包的时间模式
3. 检查是否有 CRC 错误或接口重置
4. 检查是否有链路翻动（flapping）

```bash
# 长时间 ping 统计
ping -c 1000 -i 0.1 target > ping_results.txt

# 分析丢包模式
awk -F'=' '/time=/{print $2}' ping_results.txt | \
  python3 -c "import sys; data=[float(x) for x in sys.stdin]; \
  print(f'avg={sum(data)/len(data):.2f}ms, max={max(data):.2f}ms')"

# 检查接口错误和丢包
ip -s link show eth0
ethtool -S eth0 | grep -E 'drop|error|crc'
```

#### **问题 4：Bufferbloat**

Bufferbloat 是由于路由器缓冲区过大导致的高延迟问题。

```bash
# 使用 flent 测试 bufferbloat
flent rrul -p all_scaled -l 60 -H server_ip

# 使用 ping 在下载时测量延迟
# 终端 1：开始下载
iperf3 -c server_ip -t 60

# 终端 2：同时测量延迟
ping -c 60 -i 1 target
```

### 29.7.3 排队论辅助诊断

当怀疑排队延迟过高时，可以使用排队论进行分析：

```python
import math

def mm1_analysis(arrival_rate, service_rate):
    """M/M/1 排队模型分析"""
    rho = arrival_rate / service_rate

    if rho >= 1:
        print("警告：系统不稳定（ρ >= 1）！队列将无限增长。")
        print(f"利用率 ρ = {rho:.4f}")
        print("建议：降低到达率或增加服务率。")
        return None

    # M/M/1 性能指标
    L = rho / (1 - rho)                     # 系统平均顾客数
    Lq = rho**2 / (1 - rho)                 # 队列平均顾客数
    W = 1 / (service_rate - arrival_rate)   # 平均逗留时间
    Wq = rho / (service_rate - arrival_rate) # 平均等待时间

    results = {
        'utilization': rho,
        'L': L, 'Lq': Lq,
        'W': W, 'Wq': Wq
    }

    print(f"=== M/M/1 排队分析 ===")
    print(f"到达率 λ = {arrival_rate} 请求/秒")
    print(f"服务率 μ = {service_rate} 请求/秒")
    print(f"利用率 ρ = {rho:.4f}")
    print(f"系统平均顾客数 L = {L:.4f}")
    print(f"队列平均顾客数 Lq = {Lq:.4f}")
    print(f"平均逗留时间 W = {W*1000:.4f} ms")
    print(f"平均等待时间 Wq = {Wq*1000:.4f} ms")

    # 用 Little 定律验证
    L_verify = arrival_rate * W
    print(f"\nLittle 定律验证: λ×W = {L_verify:.4f}, L = {L:.4f} ✓")

    return results

def mmmc_analysis(arrival_rate, service_rate, num_servers):
    """M/M/c 排队模型分析"""
    a = arrival_rate / service_rate  # 话务量强度
    c = num_servers
    rho = a / c  # 每个服务台的利用率

    if rho >= 1:
        print("警告：系统不稳定！")
        return None

    # 计算 Erlang C 公式中的 P0
    sum_terms = sum((a**k) / math.factorial(k) for k in range(c))
    last_term = ((a**c) / math.factorial(c)) * (1 / (1 - rho))
    P0 = 1 / (sum_terms + last_term)

    # Erlang C 公式：等待概率
    C_ca = (last_term * P0)

    # M/M/c 性能指标
    Lq = C_ca * rho / (1 - rho)
    Wq = Lq / arrival_rate
    W = Wq + 1 / service_rate
    L = arrival_rate * W

    print(f"=== M/M/{c} 排队分析 ===")
    print(f"到达率 λ = {arrival_rate} 请求/秒")
    print(f"服务率 μ = {service_rate} 请求/秒/台")
    print(f"服务器数 c = {c}")
    print(f"话务量强度 a = {a:.4f} Erlang")
    print(f"每台利用率 ρ = {rho:.4f}")
    print(f"等待概率 C(c,a) = {C_ca:.4f}")
    print(f"系统平均顾客数 L = {L:.4f}")
    print(f"队列平均顾客数 Lq = {Lq:.4f}")
    print(f"平均等待时间 Wq = {Wq*1000:.4f} ms")
    print(f"平均逗留时间 W = {W*1000:.4f} ms")

    return {'L': L, 'Lq': Lq, 'W': W, 'Wq': Wq, 'C_ca': C_ca}
```

### 29.7.4 网络测量自动化脚本

```python
import subprocess
import json
import time
import re

class NetworkHealthCheck:
    """自动化网络健康检查"""

    def __init__(self, targets, gateway):
        self.targets = targets
        self.gateway = gateway
        self.results = {}

    def run_ping(self, target, count=10):
        """执行 ping 测试"""
        try:
            output = subprocess.run(
                ['ping', '-c', str(count), '-W', '2', target],
                capture_output=True, text=True, timeout=30
            )
            # 解析结果
            stats = output.stdout
            loss_match = re.search(r'(\d+)% packet loss', stats)
            rtt_match = re.search(
                r'rtt min/avg/max/mdev = ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)',
                stats
            )

            result = {
                'target': target,
                'packet_loss': int(loss_match.group(1)) if loss_match else 100,
            }

            if rtt_match:
                result['rtt_min'] = float(rtt_match.group(1))
                result['rtt_avg'] = float(rtt_match.group(2))
                result['rtt_max'] = float(rtt_match.group(3))
                result['rtt_mdev'] = float(rtt_match.group(4))

            return result
        except (subprocess.TimeoutExpired, Exception) as e:
            return {'target': target, 'error': str(e)}

    def run_traceroute(self, target):
        """执行 traceroute"""
        try:
            output = subprocess.run(
                ['traceroute', '-n', '-w', '2', '-q', '1', target],
                capture_output=True, text=True, timeout=120
            )
            hops = []
            for line in output.stdout.split('\n')[1:]:
                parts = line.strip().split()
                if parts and parts[0].isdigit():
                    hop = {
                        'hop': int(parts[0]),
                        'ip': parts[1] if parts[1] != '*' else None,
                        'rtt': float(parts[2]) if parts[2] != '*' else None
                    }
                    hops.append(hop)
            return {'target': target, 'hops': hops}
        except Exception as e:
            return {'target': target, 'error': str(e)}

    def check_interface_errors(self):
        """检查网络接口错误"""
        try:
            output = subprocess.run(
                ['ip', '-s', 'link'], capture_output=True, text=True
            )
            interfaces = {}
            current_iface = None
            for line in output.stdout.split('\n'):
                iface_match = re.match(r'^\d+: (\S+):', line)
                if iface_match:
                    current_iface = iface_match.group(1)
                    interfaces[current_iface] = {}
                elif 'RX:' in line and current_iface:
                    pass  # RX stats header
                elif 'TX:' in line and current_iface:
                    pass  # TX stats header
            return interfaces
        except Exception as e:
            return {'error': str(e)}

    def full_check(self):
        """执行完整健康检查"""
        print("=" * 60)
        print("网络健康检查报告")
        print(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # 1. 网关连通性
        print("\n[1] 网关连通性测试")
        gw_result = self.run_ping(self.gateway, count=5)
        self.results['gateway'] = gw_result
        print(f"  网关 {self.gateway}: 丢包={gw_result.get('packet_loss', '?')}% "
              f"延迟={gw_result.get('rtt_avg', '?')} ms")

        # 2. 目标连通性
        print("\n[2] 目标连通性测试")
        for target in self.targets:
            result = self.run_ping(target, count=10)
            self.results[f'ping_{target}'] = result
            status = "✓" if result.get('packet_loss', 100) < 5 else "✗"
            print(f"  {status} {target}: 丢包={result.get('packet_loss', '?')}% "
                  f"延迟={result.get('rtt_avg', '?')} ms")

        # 3. 路径分析
        print("\n[3] 路径分析")
        if self.targets:
            traceroute_result = self.run_traceroute(self.targets[0])
            self.results['traceroute'] = traceroute_result
            if 'hops' in traceroute_result:
                print(f"  到 {self.targets[0]} 共 {len(traceroute_result['hops'])} 跳")

        print("\n" + "=" * 60)
        print("检查完成")
        return self.results

if __name__ == '__main__':
    checker = NetworkHealthCheck(
        targets=['8.8.8.8', '1.1.1.1'],
        gateway='192.168.1.1'
    )
    checker.full_check()
```

### 29.7.5 M/M/1 事件驱动仿真器

```python
import heapq
import random
import math

class MM1Simulator:
    """M/M/1 排队系统事件驱动仿真器"""

    def __init__(self, arrival_rate, service_rate, seed=42):
        self.lam = arrival_rate
        self.mu = service_rate
        self.rho = arrival_rate / service_rate
        self.rng = random.Random(seed)

        # 事件队列：(时间, 事件类型, 顾客ID)
        self.events = []
        self.next_customer_id = 0

        # 系统状态
        self.num_in_system = 0  # 系统中的顾客数
        self.server_busy = False

        # 统计计数器
        self.area_n = 0.0       # L 的积分面积
        self.area_q = 0.0       # Lq 的积分面积
        self.last_event_time = 0.0

        # 顾客统计
        self.customers_served = 0
        self.total_wait_time = 0.0
        self.total_system_time = 0.0
        self.wait_times = []

        # 状态分布统计
        self.state_counts = {}  # 每个状态的停留时间

    def schedule_event(self, time, event_type, customer_id):
        heapq.heappush(self.events, (time, event_type, customer_id))

    def exponential(self, rate):
        """指数分布随机数"""
        return -math.log(self.rng.random()) / rate

    def update_areas(self, current_time):
        """更新积分面积"""
        dt = current_time - self.last_event_time
        self.area_n += self.num_in_system * dt
        self.area_q += max(0, self.num_in_system - (1 if self.server_busy else 0)) * dt

        # 更新状态计数
        state = self.num_in_system
        self.state_counts[state] = self.state_counts.get(state, 0) + dt

        self.last_event_time = current_time

    def handle_arrival(self, time, customer_id):
        """处理到达事件"""
        self.update_areas(time)

        self.num_in_system += 1

        if not self.server_busy:
            # 服务台空闲，立即开始服务
            self.server_busy = True
            service_time = self.exponential(self.mu)
            # 记录等待时间（为 0）
            if customer_id not in self._arrival_times:
                self._arrival_times[customer_id] = time
            self.schedule_event(time + service_time, 'departure', customer_id)
        else:
            # 进入队列等待
            pass

        # 安排下一个到达
        self.next_customer_id += 1
        next_arrival_time = time + self.exponential(self.lam)
        self._arrival_times[self.next_customer_id] = next_arrival_time
        self.schedule_event(next_arrival_time, 'arrival', self.next_customer_id)

    def handle_departure(self, time, customer_id):
        """处理离开事件"""
        self.update_areas(time)

        self.num_in_system -= 1
        self.customers_served += 1

        # 记录统计
        arrival_time = self._arrival_times.get(customer_id, time)
        system_time = time - arrival_time
        self.total_system_time += system_time
        self.wait_times.append(system_time)

        if self.num_in_system > 0:
            # 还有顾客在等待，开始服务下一个
            service_time = self.exponential(self.mu)
            # 这里简化处理，不跟踪具体排队顺序
            next_id = customer_id + 1
            self.schedule_event(time + service_time, 'departure', next_id)
        else:
            self.server_busy = False

    def run(self, max_time=100000):
        """运行仿真"""
        self._arrival_times = {}

        # 初始化：安排第一个到达
        self.next_customer_id = 0
        first_arrival = self.exponential(self.lam)
        self._arrival_times[0] = first_arrival
        self.schedule_event(first_arrival, 'arrival', 0)

        while self.events and self.last_event_time < max_time:
            time, event_type, customer_id = heapq.heappop(self.events)
            if time > max_time:
                break

            if event_type == 'arrival':
                self.handle_arrival(time, customer_id)
            elif event_type == 'departure':
                self.handle_departure(time, customer_id)

    def get_statistics(self):
        """获取仿真统计结果"""
        total_time = self.last_event_time

        # 仿真结果
        sim_L = self.area_n / total_time
        sim_Lq = self.area_q / total_time
        sim_W = self.total_system_time / max(1, self.customers_served)
        sim_Wq = sim_W - 1 / self.mu

        # 理论值
        theo_L = self.rho / (1 - self.rho)
        theo_Lq = self.rho**2 / (1 - self.rho)
        theo_W = 1 / (self.mu - self.lam)
        theo_Wq = self.rho / (self.mu - self.lam)

        # 状态分布（归一化）
        state_probs = {}
        for state, duration in self.state_counts.items():
            state_probs[state] = duration / total_time

        return {
            'simulation': {
                'L': sim_L, 'Lq': sim_Lq,
                'W': sim_W, 'Wq': sim_Wq,
                'customers_served': self.customers_served,
                'total_time': total_time
            },
            'theory': {
                'L': theo_L, 'Lq': theo_Lq,
                'W': theo_W, 'Wq': theo_Wq
            },
            'state_probabilities': state_probs
        }

    def print_report(self):
        """打印仿真报告"""
        stats = self.get_statistics()

        print("=" * 60)
        print("M/M/1 仿真报告")
        print("=" * 60)
        print(f"参数: λ={self.lam}, μ={self.mu}, ρ={self.rho:.4f}")
        print(f"仿真时长: {stats['simulation']['total_time']:.2f}")
        print(f"服务顾客数: {stats['simulation']['customers_served']}")
        print()
        print("指标          仿真值      理论值      误差")
        print("-" * 50)

        for key in ['L', 'Lq', 'W', 'Wq']:
            sim = stats['simulation'][key]
            theo = stats['theory'][key]
            error = abs(sim - theo) / theo * 100 if theo > 0 else 0
            print(f"  {key:6s}    {sim:10.4f}  {theo:10.4f}  {error:6.2f}%")

        print()
        print("状态分布 (前 10 个状态):")
        for state in sorted(stats['state_probabilities'].keys())[:10]:
            prob = stats['state_probabilities'][state]
            theo_prob = (1 - self.rho) * self.rho**state
            print(f"  P({state:2d}) = {prob:.4f}  (理论: {theo_prob:.4f})")


# 运行仿真
if __name__ == '__main__':
    sim = MM1Simulator(arrival_rate=0.8, service_rate=1.0)
    sim.run(max_time=100000)
    sim.print_report()
```

### 29.7.6 网络故障诊断决策树

```python
class NetworkDiagnosticEngine:
    """网络故障诊断引擎 - 基于决策树"""

    def __init__(self):
        self.symptoms = []
        self.diagnosis = []

    def check_physical_layer(self):
        """物理层检查"""
        issues = []

        # 模拟检查接口状态
        interface_status = self._run_command("ip link show")

        if "state DOWN" in interface_status:
            issues.append({
                'layer': 'L1-物理层',
                'issue': '接口状态 DOWN',
                'suggestion': '检查网线连接、交换机端口、光模块'
            })

        # 检查 CRC 错误
        error_stats = self._run_command("ip -s link show")
        if "errors" in error_stats and self._parse_errors(error_stats) > 0:
            issues.append({
                'layer': 'L1-物理层',
                'issue': '检测到 CRC/帧错误',
                'suggestion': '检查网线质量、电磁干扰、网卡硬件'
            })

        return issues

    def check_datalink_layer(self):
        """数据链路层检查"""
        issues = []

        # 检查 ARP 表
        arp_table = self._run_command("arp -n")

        # 检查 MAC 地址表溢出
        # ... 实际实现会检查交换机 MAC 表

        return issues

    def check_network_layer(self):
        """网络层检查"""
        issues = []

        # 检查 IP 配置
        ip_config = self._run_command("ip addr show")

        # 检查路由表
        routes = self._run_command("ip route show")
        if "default" not in routes:
            issues.append({
                'layer': 'L3-网络层',
                'issue': '缺少默认路由',
                'suggestion': '添加默认网关: ip route add default via <gateway>'
            })

        return issues

    def check_transport_layer(self):
        """传输层检查"""
        issues = []

        # 检查连接状态
        connections = self._run_command("ss -tn")

        # 统计 TIME_WAIT 数量
        time_wait_count = connections.count("TIME-WAIT")
        if time_wait_count > 10000:
            issues.append({
                'layer': 'L4-传输层',
                'issue': f'TIME_WAIT 过多 ({time_wait_count})',
                'suggestion': '调整内核参数: net.ipv4.tcp_tw_reuse = 1'
            })

        # 检查重传
        tcp_stats = self._run_command("netstat -s | grep retransmit")

        return issues

    def check_application_layer(self):
        """应用层检查"""
        issues = []

        # 检查服务状态
        # 检查端口监听
        listeners = self._run_command("ss -tln")

        return issues

    def diagnose(self, symptom_description=""):
        """执行完整诊断"""
        print("=" * 60)
        print("网络故障诊断报告")
        print("=" * 60)

        all_issues = []

        layers = [
            ("物理层", self.check_physical_layer),
            ("数据链路层", self.check_datalink_layer),
            ("网络层", self.check_network_layer),
            ("传输层", self.check_transport_layer),
            ("应用层", self.check_application_layer),
        ]

        for layer_name, check_func in layers:
            print(f"\n检查 {layer_name}...")
            issues = check_func()
            if issues:
                for issue in issues:
                    print(f"  ⚠ {issue['issue']}")
                    print(f"    建议: {issue['suggestion']}")
                all_issues.extend(issues)
            else:
                print(f"  ✓ 未发现问题")

        print("\n" + "=" * 60)
        if all_issues:
            print(f"共发现 {len(all_issues)} 个问题")
        else:
            print("未发现明显问题，可能需要更深入的分析")

        return all_issues

    def _run_command(self, cmd):
        """执行系统命令（模拟）"""
        try:
            import subprocess
            result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=5)
            return result.stdout
        except Exception:
            return ""

    def _parse_errors(self, stats_output):
        """解析接口错误统计"""
        import re
        errors = re.findall(r'(\d+)\s+errors', stats_output)
        return sum(int(e) for e in errors)
```

---

## 29.8 综合案例：端到端性能分析

### 29.8.1 案例描述

某公司的 Web 应用响应缓慢，用户体验差。用户报告页面加载时间从 2 秒增加到 8 秒。

### 29.8.2 分析过程

#### **第一步：信息收集**

```bash
# 网络拓扑
# 客户端 → 交换机 → 路由器 → WAN → 数据中心 → 负载均衡 → Web服务器

# 基础连通性测试
ping -c 100 web-server.example.com

# 路径分析
traceroute web-server.example.com

# 带宽测试
iperf3 -c web-server.example.com -t 30 -P 4
```

#### **第二步：延迟分解**

```python
def analyze_latency(rtt_measurements):
    """分析 RTT 组成"""
    avg_rtt = sum(rtt_measurements) / len(rtt_measurements)
    jitter = max(rtt_measurements) - min(rtt_measurements)
    loss = sum(1 for r in rtt_measurements if r is None) / len(rtt_measurements)

    print(f"平均 RTT: {avg_rtt:.2f} ms")
    print(f"抖动 (max-min): {jitter:.2f} ms")
    print(f"丢包率: {loss*100:.1f}%")

    # 延迟组成估算
    # 假设传播延迟固定（基于光速）
    propagation_delay = 30  # ms, 假设跨洲传输
    transmission_delay = 0.5  # ms, 1500字节/1Gbps
    processing_delay = 0.1  # ms
    queuing_delay = avg_rtt - propagation_delay - transmission_delay - processing_delay

    print(f"\n延迟分解估算:")
    print(f"  传播延迟: ~{propagation_delay} ms")
    print(f"  传输延迟: ~{transmission_delay} ms")
    print(f"  处理延迟: ~{processing_delay} ms")
    print(f"  排队延迟: ~{max(0, queuing_delay):.2f} ms (估算)")

    if queuing_delay > 50:
        print("\n⚠ 排队延迟过高！可能存在拥塞。")

    return {
        'avg_rtt': avg_rtt,
        'jitter': jitter,
        'loss': loss,
        'queuing_delay': queuing_delay
    }
```

#### **第三步：瓶颈定位**

使用 M/M/1 模型分析路由器缓冲区：

```python
def analyze_router_buffer(arrival_rate_mbps, link_capacity_mbps, avg_packet_size_bytes):
    """分析路由器缓冲区性能"""

    # 转换为包/秒
    packet_size_bits = avg_packet_size_bytes * 8
    arrival_rate_pps = arrival_rate_mbps * 1e6 / packet_size_bits
    service_rate_pps = link_capacity_mbps * 1e6 / packet_size_bits

    print(f"到达率: {arrival_rate_pps:.0f} packets/sec")
    print(f"服务率: {service_rate_pps:.0f} packets/sec")

    results = mm1_analysis(arrival_rate_pps, service_rate_pps)

    if results:
        # 建议缓冲区大小
        bdp_bytes = link_capacity_mbps * 1e6 * 0.1 / 8  # 假设 RTT = 100ms
        print(f"\n建议缓冲区大小: {bdp_bytes/1024:.0f} KB (基于 BDP)")

    return results
```

#### **第四步：解决方案**

1. **短期方案**：
   - 调整 TCP 窗口大小
   - 启用 TCP 窗口缩放（Window Scaling）
   - 检查并修复丢包问题

2. **中期方案**：
   - 升级瓶颈链路带宽
   - 部署 CDN
   - 启用 HTTP/2

3. **长期方案**：
   - 网络架构优化
   - 部署 SD-WAN
   - 实施 QoS 策略

---

## 29.9 进阶话题

### 29.9.1 自相似流量模型

传统排队论假设流量服从泊松分布，但实际网络流量具有**自相似性**（Self-Similarity）：

- 流量在不同时间尺度上表现出相似的统计特性
- 突发流量不会随聚合而平滑
- 对排队性能的影响比泊松模型预测的更严重

Hurst 参数 $H$ 衡量自相似程度：
- $H = 0.5$：无自相似性（类似泊松过程）
- $0.5 < H < 1$：具有自相似性，$H$ 越大自相似性越强

### 29.9.2 带宽测量技术

#### **包对（Packet Pair）技术**

1. 发送两个背靠背的大小相同的包
2. 在瓶颈链路处，两个包被序列化，间隔变为 $\Delta_{out} = L / C_{bottleneck}$
3. 在接收端测量包间隔，推断瓶颈带宽

$$C_{bottleneck} = \frac{L}{\Delta_{out}}$$

#### **包列车（Packet Train）技术**

发送一系列等间隔的包，通过分析接收端的间隔变化来估计可用带宽。

### 29.9.3 TCP 性能优化

```bash
# 查看当前 TCP 参数
sysctl net.ipv4.tcp_window_scaling
sysctl net.ipv4.tcp_timestamps
sysctl net.ipv4.tcp_sack
sysctl net.core.rmem_max
sysctl net.core.wmem_max

# 优化 TCP 参数
sysctl -w net.ipv4.tcp_window_scaling=1
sysctl -w net.ipv4.tcp_timestamps=1
sysctl -w net.ipv4.tcp_sack=1
sysctl -w net.core.rmem_max=16777216
sysctl -w net.core.wmem_max=16777216
sysctl -w net.ipv4.tcp_rmem="4096 87380 16777216"
sysctl -w net.ipv4.tcp_wmem="4096 65536 16777216"
sysctl -w net.ipv4.tcp_congestion_control=bbr

# 查看当前拥塞控制算法
sysctl net.ipv4.tcp_congestion_control

# 查看可用的拥塞控制算法
sysctl net.ipv4.tcp_available_congestion_control
```

---

## 29.10 本章小结

### 核心概念回顾

| 概念 | 关键公式 | 应用场景 |
|------|---------|---------|
| 吞吐量 | $S = \min(C, W/RTT)$ | 带宽评估 |
| 延迟 | $d = d_{proc} + d_{queue} + d_{trans} + d_{prop}$ | 延迟分析 |
| BDP | $BDP = C \times RTT$ | 窗口/缓冲区设计 |
| Little 定律 | $L = \lambda W$ | 容量规划 |
| M/M/1 | $L = \rho/(1-\rho)$ | 排队分析 |
| Erlang C | $C(c, \rho)$ | 多服务台分析 |

### 工具速查表

| 工具 | 用途 | 常用选项 |
|------|------|---------|
| ping | 连通性、RTT | `-c`, `-s`, `-f` |
| traceroute | 路径发现 | `-n`, `-T`, `-w` |
| iperf3 | 带宽测量 | `-c`, `-P`, `-R`, `-u` |
| tcpdump | 抓包 | `-i`, `-w`, `-r`, `-n` |
| Wireshark | 协议分析 | 显示过滤器、IO 图表 |
| ss | 连接状态 | `-t`, `-l`, `-n`, `-i` |

### 排障流程速记

```
物理层 → 数据链路层 → 网络层 → 传输层 → 应用层
   ↓          ↓          ↓         ↓         ↓
 网线/光纤  ARP/MAC   IP/路由   TCP/UDP   HTTP/DNS
```

---

## 练习题

### 基础题

1. 一条 100 Mbps 链路的 RTT 为 50ms，计算 BDP。要充分利用该链路，TCP 窗口至少需要多大？

2. 某 Web 服务器的请求到达率为 200 req/s，平均服务时间为 4ms。使用 M/M/1 模型计算：
   - 利用率 ρ
   - 平均队列长度
   - 平均响应时间

3. 解释为什么 `ping` 测得的 RTT 不等于 `traceroute` 所有跳 RTT 之和？

### 进阶题

4. 某路由器连接 1000 条 TCP 流，链路带宽为 10 Gbps，RTT 为 100ms。使用 Appenzeller 规则计算所需的缓冲区大小。

5. 设计一个 Python 脚本，同时运行 iperf3 带宽测试和 ping 延迟测试，分析带宽与延迟的关系（检测 Bufferbloat）。

6. 在 M/M/c 模型中，如果到达率 λ=100 req/s，服务率 μ=50 req/s/台，需要多少台服务器才能使平均等待时间小于 10ms？

---

## 参考资料

- RFC 3393: IP Packet Delay Variation Metric for IP Performance Metrics (IPPM)
- Kurose, J. F., & Ross, K. W. *Computer Networking: A Top-Down Approach*
- Kleinrock, L. *Queueing Systems, Volume 1: Theory*
- Appenzeller, G., Keslassy, I., & McKeown, N. *Sizing Router Buffers (SIGCOMM 2004)*
- iperf3 官方文档: https://iperf.fr/
- Wireshark 用户指南: https://www.wireshark.org/docs/

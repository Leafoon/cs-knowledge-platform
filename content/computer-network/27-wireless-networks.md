# Chapter 27: 无线与移动网络

> **学习目标**：
> - 理解无线信道的传播特性，包括大尺度衰落、小尺度衰落、多径效应，以及SNR与BER的关系
> - 掌握IEEE 802.11协议族的MAC层机制，包括CSMA/CA、RTS/CTS、退避算法与NAV虚拟载波侦听
> - 熟悉WiFi帧格式的各个字段含义，特别是Frame Control、地址字段和QoS控制
> - 了解无线安全从WEP到WPA3的演进过程及各代协议的核心缺陷与改进
> - 掌握蜂窝网络从2G到5G的技术演进，理解LTE EPC架构核心网元的功能
> - 理解移动IP的工作机制，包括家乡代理、外地代理、注册过程与路由优化
> - 掌握切换管理的基本概念，包括水平切换、垂直切换及LTE中的X2/S1切换

---

## 27.1 无线信道特性

无线通信与有线通信的根本区别在于信道特性。无线信道是一个时变、频变、空变的复杂传播环境，理解其特性是设计无线网络协议的基础。

### 27.1.1 无线信号传播与路径损耗

无线信号从发射天线到接收天线的传播过程中，信号功率会随距离增大而衰减。最基本的传播模型是**自由空间路径损耗模型**（Free Space Path Loss, FSPL）：

$$P_r = P_t \cdot G_t \cdot G_r \cdot \left(\frac{\lambda}{4\pi d}\right)^2$$

其中：
- $P_r$：接收功率
- $P_t$：发射功率
- $G_t, G_r$：发射和接收天线增益
- $\lambda$：信号波长
- $d$：传播距离

以分贝表示的路径损耗：

$$PL(dB) = 32.44 + 20\log_{10}(d) + 20\log_{10}(f)$$

其中 $d$ 为距离（km），$f$ 为频率（MHz）。

```python
import math

def free_space_path_loss(d_km, f_mhz):
    """计算自由空间路径损耗 (dB)
    
    Args:
        d_km: 距离，单位 km
        f_mhz: 频率，单位 MHz
    Returns:
        路径损耗，单位 dB
    """
    return 32.44 + 20 * math.log10(d_km) + 20 * math.log10(f_mhz)

def received_power_dbm(pt_dbm, gt_dbi, gr_dbi, d_km, f_mhz):
    """计算接收功率 (dBm)
    
    Args:
        pt_dbm: 发射功率 dBm
        gt_dbi: 发射天线增益 dBi
        gr_dbi: 接收天线增益 dBi
        d_km: 距离 km
        f_mhz: 频率 MHz
    Returns:
        接收功率 dBm
    """
    pl = free_space_path_loss(d_km, f_mhz)
    return pt_dbm + gt_dbi + gr_dbi - pl

# 示例：2.4 GHz WiFi，距离100m，发射功率20dBm，天线增益0dBi
f = 2400  # MHz
d = 0.1   # km (100m)
pt = 20   # dBm
gt = 0    # dBi
gr = 0    # dBi

pl = free_space_path_loss(d, f)
pr = received_power_dbm(pt, gt, gr, d, f)
print(f"自由空间路径损耗: {pl:.2f} dB")
print(f"接收功率: {pr:.2f} dBm")

# 不同距离下的接收功率
print("\n距离(m) | 路径损耗(dB) | 接收功率(dBm)")
print("-" * 45)
for dist in [10, 50, 100, 200, 500, 1000]:
    d_km = dist / 1000
    loss = free_space_path_loss(d_km, f)
    recv = pt + gt + gr - loss
    print(f"{dist:>6} | {loss:>12.2f} | {recv:>11.2f}")
```

### 27.1.2 大尺度衰落

**大尺度衰落**（Large-Scale Fading）描述信号在较大距离范围（数十至数百米）内的功率变化，主要由以下因素引起：

**对数距离路径损耗模型**：

$$PL(d) = PL(d_0) + 10n\log_{10}\left(\frac{d}{d_0}\right) + X_\sigma$$

其中：
- $d_0$：参考距离（通常为1m或100m）
- $n$：路径损耗指数，取决于传播环境
- $X_\sigma$：零均值高斯随机变量，代表阴影衰落

| 环境类型 | 路径损耗指数 $n$ |
|---------|----------------|
| 自由空间 | 2 |
| 城市宏蜂窝 | 3.7 ~ 4.0 |
| 城市微蜂窝 | 2.7 ~ 3.5 |
| 室内视距(LOS) | 1.6 ~ 1.8 |
| 室内非视距(NLOS) | 4.0 ~ 6.0 |
| 工厂NLOS | 2.0 ~ 3.0 |

**阴影衰落**（Shadow Fading）由建筑物、地形等障碍物遮挡引起，接收信号功率的对数服从正态分布，标准差 $\sigma$ 通常为4~12 dB。

### 27.1.3 小尺度衰落与多径传播

**小尺度衰落**（Small-Scale Fading）描述信号在极短距离（半个波长）或极短时间内的快速波动。主要原因是**多径传播**（Multipath Propagation）：

信号从发射端到接收端经过多条路径（直射、反射、绕射、散射），每条路径的幅度、相位和时延不同，在接收端叠加产生：

1. **时间色散**：不同时延的多径分量造成信号在时间上的展宽
2. **频率选择性衰落**：信道对不同频率分量产生不同的衰减
3. **平坦衰落**：信号带宽远小于信道相干带宽时，所有频率分量经历相似衰落
4. **瑞利衰落**（Rayleigh Fading）：无直射路径时，信号包络服从瑞利分布
5. **莱斯衰落**（Rician Fading）：存在直射路径时，信号包络服从莱斯分布

**相干带宽** $B_c$ 与**时延扩展** $\tau_{rms}$ 的关系：

$$B_c \approx \frac{1}{5\tau_{rms}}$$

当信号带宽 $B_s > B_c$ 时，信号经历频率选择性衰落；当 $B_s < B_c$ 时，信号经历平坦衰落。

**多普勒效应**：移动终端的运动导致接收信号频率偏移：

$$f_d = \frac{v \cdot f_c}{c}$$

其中 $v$ 为移动速度，$f_c$ 为载波频率，$c$ 为光速。多普勒扩展导致**时间选择性衰落**。

```python
import numpy as np

def rayleigh_fading(num_samples, num_paths=20):
    """生成瑞利衰落信道样本
    
    Args:
        num_samples: 样本数量
        num_paths: 多径数量
    Returns:
        信道增益 (线性值)
    """
    # 每条路径的实部和虚部分量服从正态分布
    real_parts = np.random.randn(num_samples, num_paths)
    imag_parts = np.random.randn(num_samples, num_paths)
    
    # 各路径叠加
    h_real = np.sum(real_parts, axis=1) / np.sqrt(num_paths)
    h_imag = np.sum(imag_parts, axis=1) / np.sqrt(num_paths)
    
    # 信道增益幅度
    h_mag = np.sqrt(h_real**2 + h_imag**2)
    return h_mag

def rician_fading(num_samples, k_factor=5, num_paths=20):
    """生成莱斯衰落信道样本
    
    Args:
        num_samples: 样本数量
        k_factor: 莱斯K因子 (直射/散射功率比)
        num_paths: 散射路径数量
    Returns:
        信道增益 (线性值)
    """
    # 直射分量
    direct = np.sqrt(k_factor / (k_factor + 1))
    # 散射分量
    scatter = np.sqrt(1 / (k_factor + 1))
    
    h_rayleigh = rayleigh_fading(num_samples, num_paths)
    h_rician = np.sqrt((direct + scatter * np.random.randn(num_samples))**2 + 
                       (scatter * np.random.randn(num_samples))**2)
    return h_rician

# 生成衰落信道并计算统计特性
N = 10000
h_rayleigh = rayleigh_fading(N)
h_rician = rician_fading(N, k_factor=5)

print("瑞利衰落统计:")
print(f"  均值: {np.mean(h_rayleigh):.4f}")
print(f"  理论均值: {np.sqrt(np.pi/2)/np.sqrt(20)*np.sqrt(20):.4f} (sqrt(pi/2)/sqrt(K))")
print(f"  标准差: {np.std(h_rayleigh):.4f}")

print("\n莱斯衰落统计 (K=5):")
print(f"  均值: {np.mean(h_rician):.4f}")
print(f"  标准差: {np.std(h_rician):.4f}")
```

### 27.1.4 隐藏终端与暴露终端问题

无线网络中的**载波侦听**机制面临两个经典问题：

**隐藏终端问题**（Hidden Terminal Problem）：

```
    A ------- B ------- C
    |         |         |
    | 范围    | 范围    | 范围
    |         |         |
    
    A 和 C 都在 B 的通信范围内
    但 A 和 C 彼此不在对方的通信范围内
```

- A 向 B 发送数据时，C 感知不到信道忙碌
- C 也可能向 B 发送数据，导致 B 处发生**碰撞**
- A 对 C 来说是"隐藏终端"

**暴露终端问题**（Exposed Terminal Problem）：

```
    B ------- A ------- C ------- D
    |         |         |         |
    | 范围    | 范围    | 范围    |
    |         |         |         |
    
    A 在 B 的范围内，C 也在 A 的范围内
    但 B 和 D 不在彼此范围内
```

- B 向 A 发送数据时，C 侦听到信道忙碌
- C 实际上可以同时向 D 发送数据（不会与 B→A 冲突）
- 但 C 被"暴露"在 B 的传输范围内，被迫退避

这两个问题是设计 RTS/CTS 机制的重要动机。

### 27.1.5 SNR与BER关系

**信噪比**（Signal-to-Noise Ratio, SNR）是衡量无线信道质量的核心指标：

$$SNR = \frac{P_{signal}}{P_{noise}} = \frac{P_r}{N_0 \cdot B}$$

其中 $N_0$ 为噪声功率谱密度，$B$ 为带宽。

SNR 以分贝表示：$SNR_{dB} = 10\log_{10}(SNR)$

**误比特率**（Bit Error Rate, BER）与 SNR 的关系取决于调制方式：

| 调制方式 | BER 公式 |
|---------|---------|
| BPSK/QPSK | $BER = \frac{1}{2}\text{erfc}(\sqrt{SNR})$ |
| 16-QAM | $BER \approx \frac{3}{8}\text{erfc}\left(\sqrt{\frac{SNR}{5}}\right)$ |
| 64-QAM | $BER \approx \frac{7}{24}\text{erfc}\left(\sqrt{\frac{SNR}{21}}\right)$ |

### 27.1.6 Shannon容量公式在无线中的应用

**Shannon-Hartley定理**给出信道的理论最大传输速率：

$$C = B \cdot \log_2(1 + SNR) \quad \text{(bits/s)}$$

在无线信道中，由于信道随时间变化，**遍历容量**（Ergodic Capacity）为：

$$C_{ergodic} = \mathbb{E}\left[B \cdot \log_2\left(1 + |h|^2 \cdot SNR\right)\right]$$

其中 $h$ 为随机信道增益，期望对信道分布取平均。

```python
import math

def shannon_capacity(bandwidth_hz, snr_linear):
    """计算Shannon信道容量
    
    Args:
        bandwidth_hz: 带宽 Hz
        snr_linear: 信噪比 (线性值)
    Returns:
        信道容量 bits/s
    """
    return bandwidth_hz * math.log2(1 + snr_linear)

def snr_to_capacity_comparison():
    """对比不同调制方式在不同SNR下的实际速率与Shannon极限"""
    bandwidth = 20e6  # 20 MHz
    
    snr_db_values = [0, 5, 10, 15, 20, 25, 30]
    
    print(f"带宽: {bandwidth/1e6} MHz")
    print(f"{'SNR(dB)':>8} | {'Shannon极限':>14} | {'BPSK实际':>12} | {'16QAM实际':>12} | {'64QAM实际':>12}")
    print("-" * 72)
    
    for snr_db in snr_db_values:
        snr_linear = 10 ** (snr_db / 10)
        shannon = shannon_capacity(bandwidth, snr_linear)
        
        # 各调制方式的频谱效率 (bits/s/Hz)
        bpsk_efficiency = 1 * (1 - 0.5 * math.erfc(math.sqrt(snr_linear)))
        qam16_efficiency = 4 * (1 - 3/8 * math.erfc(math.sqrt(snr_linear/5)))
        qam64_efficiency = 6 * (1 - 7/24 * math.erfc(math.sqrt(snr_linear/21)))
        
        bpsk_rate = bpsk_efficiency * bandwidth
        qam16_rate = qam16_efficiency * bandwidth
        qam64_rate = qam64_efficiency * bandwidth
        
        print(f"{snr_db:>8} | {shannon/1e6:>11.2f} Mbps | {bpsk_rate/1e6:>9.2f} Mbps | {qam16_rate/1e6:>9.2f} Mbps | {qam64_rate/1e6:>9.2f} Mbps")

snr_to_capacity_comparison()
```

<div data-component="WiFiMACController"></div>

---

## 27.2 IEEE 802.11标准族

IEEE 802.11是无线局域网（WLAN）的基础标准族，定义了物理层（PHY）和介质访问控制层（MAC）的规范。

### 27.2.1 802.11标准概览

| 标准 | 名称 | 频段 | 最大速率 | 发布年份 |
|------|------|------|---------|---------|
| 802.11 | — | 2.4 GHz | 2 Mbps | 1997 |
| 802.11b | — | 2.4 GHz | 11 Mbps | 1999 |
| 802.11a | — | 5 GHz | 54 Mbps | 1999 |
| 802.11g | — | 2.4 GHz | 54 Mbps | 2003 |
| 802.11n | HT | 2.4/5 GHz | 600 Mbps | 2009 |
| 802.11ac | VHT | 5 GHz | 6.93 Gbps | 2013 |
| 802.11ax | HE (WiFi 6) | 2.4/5/6 GHz | 9.6 Gbps | 2020 |
| 802.11be | EHT (WiFi 7) | 2.4/5/6 GHz | 46 Gbps | 2024 |

### 27.2.2 DCF与CSMA/CA机制

**分布式协调功能**（Distributed Coordination Function, DCF）是802.11 MAC层的基本接入方式，基于**CSMA/CA**（Carrier Sense Multiple Access with Collision Avoidance）协议。

CSMA/CA 的基本流程：

```
站点有数据要发送
    │
    ▼
侦听信道是否空闲？
    │
    ├── 否 ──→ 等待信道空闲
    │           │
    │           ▼
    │        信道空闲 + DIFS时间
    │           │
    │           ▼
    │        生成随机退避时间
    │           │
    │           ▼
    └── 是 ──→ 等待 DIFS 时间
                │
                ▼
            信道仍然空闲？
                │
                ├── 否 ──→ 冻结退避计数器，回到侦听
                │
                └── 是 ──→ 开始递减退避计数器
                            │
                            ▼
                        退避计数器到0？
                            │
                            ├── 否 ──→ 继续递减
                            │
                            └── 是 ──→ 发送数据帧
                                        │
                                        ▼
                                    等待ACK（SIFS内）
                                        │
                                        ├── 收到ACK ──→ 传输成功
                                        │
                                        └── 未收到ACK ──→ 重传
                                                            │
                                                            ▼
                                                        CW加倍（BEB）
```

**帧间间隔**（Inter-Frame Space, IFS）类型：

| IFS类型 | 缩写 | 典型时长 (802.11b) | 用途 |
|---------|------|------------------|------|
| Short IFS | SIFS | 10 μs | ACK、CTS、轮询响应 |
| PCF IFS | PIFS | 30 μs | PCF无竞争期 |
| DCF IFS | DIFS | 50 μs | DCF数据帧发送前 |
| Extended IFS | EIFS | 动态 | 接收错误帧后的等待 |

### 27.2.3 退避算法：BEB

**二进制指数退避**（Binary Exponential Backoff, BEB）是CSMA/CA碰撞处理的核心算法：

$$\text{Backoff Time} = \text{Random}(0, CW) \times \text{Slot Time}$$

- **竞争窗口** $CW$ 初始值为 $CW_{min}$（通常为15或31）
- 每次传输失败（未收到ACK），$CW$ 翻倍：$CW = \min(2 \times (CW+1) - 1, CW_{max})$
- 传输成功后，$CW$ 重置为 $CW_{min}$
- $CW_{max}$ 通常为1023或1024

| 重传次数 | CW（802.11b） | 退避时间范围 |
|---------|-------------|------------|
| 0 | 31 | 0 ~ 310 μs |
| 1 | 63 | 0 ~ 630 μs |
| 2 | 127 | 0 ~ 1270 μs |
| 3 | 255 | 0 ~ 2550 μs |
| 4 | 511 | 0 ~ 5110 μs |
| 5+ | 1023 | 0 ~ 10230 μs |

```python
import random

class BackoffEngine:
    """CSMA/CA 退避引擎实现"""
    
    def __init__(self, cw_min=15, cw_max=1023, slot_time=20e-6, retry_limit=7):
        """
        Args:
            cw_min: 最小竞争窗口
            cw_max: 最大竞争窗口
            slot_time: 时隙时间（秒），802.11b为20μs
            retry_limit: 最大重传次数
        """
        self.cw_min = cw_min
        self.cw_max = cw_max
        self.slot_time = slot_time
        self.retry_limit = retry_limit
        
        self.cw = cw_min
        self.retry_count = 0
        self.backoff_counter = 0
        self.backoff_active = False
    
    def start_backoff(self):
        """启动退避过程"""
        self.backoff_counter = random.randint(0, self.cw)
        self.backoff_active = True
        return self.backoff_counter
    
    def decrement(self):
        """每个空闲时隙递减退避计数器"""
        if self.backoff_active and self.backoff_counter > 0:
            self.backoff_counter -= 1
        return self.backoff_counter
    
    def on_success(self):
        """传输成功，重置竞争窗口"""
        self.cw = self.cw_min
        self.retry_count = 0
        self.backoff_active = False
    
    def on_collision(self):
        """传输失败（碰撞），竞争窗口加倍"""
        self.retry_count += 1
        if self.retry_count >= self.retry_limit:
            # 超过重传上限，丢弃帧
            self.cw = self.cw_min
            self.retry_count = 0
            return False  # 丢弃
        
        # 二进制指数退避
        new_cw = min(2 * (self.cw + 1) - 1, self.cw_max)
        self.cw = new_cw
        self.start_backoff()
        return True  # 继续重传
    
    def get_status(self):
        """获取当前退避引擎状态"""
        return {
            'cw': self.cw,
            'retry_count': self.retry_count,
            'backoff_counter': self.backoff_counter,
            'backoff_active': self.backoff_active,
            'backoff_time_us': self.backoff_counter * self.slot_time * 1e6
        }

# 模拟退避过程
engine = BackoffEngine()
print("=== CSMA/CA 退避引擎模拟 ===\n")

for attempt in range(8):
    engine.start_backoff()
    status = engine.get_status()
    print(f"尝试 {attempt}: CW={status['cw']}, "
          f"退避计数器={status['backoff_counter']}, "
          f"退避时间={status['backoff_time_us']:.0f} μs")
    
    # 模拟碰撞
    if attempt < 5:
        success = engine.on_collision()
        if not success:
            print(f"  -> 超过重传上限，丢弃帧")
            break
    else:
        engine.on_success()
        print(f"  -> 传输成功，CW重置为{engine.cw_min}")
```

### 27.2.4 NAV虚拟载波侦听

**网络分配矢量**（Network Allocation Vector, NAV）是一种虚拟载波侦听机制，用于解决隐藏终端问题：

- 每个数据帧和RTS/CTS帧都携带一个**Duration**字段
- 其他站点收到这些帧后，将NAV设置为Duration指定的时间
- 在NAV非零期间，站点认为信道忙碌，不尝试发送
- NAV是一个倒计时器，以微秒递减

NAV的工作原理：

```
时间轴 →

站点A:   [RTS(B,Duration=X)]-------->[CTS(A)]-------->[Data]---->[ACK]
站点B:   [<--侦听到RTS-->|<---NAV=X--->|<--SIFS-->|<---NAV=Y-->|]
站点C:   [侦听到CTS,设置NAV]-------->|<---NAV=Z---->|<---NAV=W-->|]

站点B: 收到A的RTS → 设置NAV = X → 期间不发送
站点C: 收到B的CTS → 设置NAV → 期间不发送
```

### 27.2.5 RTS/CTS机制

**请求发送/清除发送**（RTS/CTS）机制用于解决隐藏终端问题，是DCF的可选增强：

```
发送方                              接收方
  │                                  │
  │──── RTS (Duration=3×SIFS+CTS+DATA+ACK) ────>│
  │                                  │
  │<──── CTS (Duration=2×SIFS+DATA+ACK) ────────│
  │                                  │
  │          (其他站点设置NAV)         │
  │                                  │
  │──── Data Frame ───────────────────>│
  │                                  │
  │<──── ACK ─────────────────────────│
  │                                  │
```

RTS/CTS的优缺点：

| 方面 | 说明 |
|------|------|
| 优势 | 解决隐藏终端问题；碰撞仅发生在RTS帧（短帧），减少信道浪费 |
| 优势 | RTS/CTS帧很短，碰撞代价远小于长数据帧 |
| 劣势 | 增加了每次传输的开销（RTS + CTS + 额外SIFS） |
| 劣势 | 对短数据帧可能不划算 |
| 实践 | 通常设置RTS阈值，仅当数据帧大于阈值时启用RTS/CTS |

### 27.2.6 PCF点协调功能

**点协调功能**（Point Coordination Function, PCF）是一种可选的集中式接入控制机制：

- 由**接入点**（AP）作为点协调器（PC）
- AP在**无竞争期**（CFP）轮询各站点
- CFP与**竞争期**（CP）交替出现，组成超帧
- PCF支持时间敏感业务，但实际部署很少使用

```
|<────── 超帧 (Superframe) ──────>|
|<-- CFP -->|<──── CP (DCF) ────>|
|  PCF模式  |    CSMA/CA竞争      |
| 轮询/CF-Poll | 正常DCF操作     |
```

### 27.2.7 802.11n/ac/ax技术改进

**802.11n (HT - High Throughput)**：
- **MIMO**：最多4×4天线配置，支持空间复用
- **信道绑定**：将两个20MHz信道绑定为40MHz
- **帧聚合**：A-MPDU和A-MSDU减少MAC开销
- **块确认**（Block ACK）：一次确认多个帧
- 最大速率：600 Mbps（4×4, 40MHz, MCS31）

**802.11ac (VHT - Very High Throughput)**：
- 仅工作在5GHz频段
- 信道带宽：80/160MHz
- 最多8×8 MIMO
- **MU-MIMO**：下行多用户MIMO（最多4用户）
- 最大调制：256-QAM
- 最大速率：6.93 Gbps

**802.11ax (HE - High Efficiency, WiFi 6)**：
- **OFDMA**：正交频分多址，类似蜂窝网络的资源分配
- **MU-MIMO增强**：上下行均支持MU-MIMO
- **1024-QAM**：更高阶调制
- **BSS Coloring**：减少同频干扰
- **TWT**（Target Wake Time）：降低功耗
- 最大速率：9.6 Gbps

<div data-component="OFDMModulator"></div>

### 27.2.8 WiFi 6E与WiFi 7新特性

**WiFi 6E**（802.11ax扩展）：
- 扩展到6GHz频段（5.925-7.125 GHz）
- 提供最多1200MHz额外频谱
- 仅支持WiFi 6E设备使用，无旧设备干扰
- 低延迟、高吞吐

**WiFi 7**（802.11be - EHT, Extremely High Throughput）：
- **320MHz信道带宽**
- **4096-QAM**（4K-QAM）
- **16×16 MIMO**
- **多链路操作**（MLO）：同时使用多个频段
- **多RUS**（Resource Unit）：一个用户可以同时使用多个RU
- 最大速率：46 Gbps

```bash
# Linux下查看WiFi接口信息
iwconfig wlan0

# 扫描可用的WiFi网络
sudo iw dev wlan0 scan | grep -E "SSID|freq|signal"

# 查看当前连接的详细信息
iw dev wlan0 link

# 查看支持的频率和信道
iw dev wlan0 info

# 设置信道
sudo iw dev wlan0 set channel 36 HT40+

# 查看站点统计信息（信号强度、速率等）
iw dev wlan0 station dump

# 使用nmcli管理WiFi连接（NetworkManager）
nmcli device wifi list
nmcli device wifi connect "SSID_NAME" password "PASSWORD"
```

---

## 27.3 WiFi帧格式详细

802.11帧格式比以太网帧复杂得多，支持无线环境特有的功能。

### 27.3.1 帧总体结构

```
|<--- 2字节 --->|<--- 2字节 --->|<--- 6字节 --->| 变长 | 0-8字节 | 4字节 |
| Frame Control |  Duration/ID  |    Address 1   | Seq  |  QoS    |  FCS  |
|               |               |    (RA)        | Ctrl |  Ctrl   |       |
+---------------+---------------+----------------+------+------+--------+
|                MAC Header (24-30字节)                         | Payload | FCS |
```

帧控制字段（Frame Control）是理解802.11帧的关键：

### 27.3.2 Frame Control字段解析

Frame Control字段占2字节（16位），结构如下：

```
 位: | 0-1  | 2-3  |  4   |  5   |  6   |  7   | 8-9 | 10-11 | 12-13 | 14-15 |
     | Prot | Type |Sub   |To DS |From  |More  |Retry|Pwr   |More   |Order  |
     | Ver  |      |type  |      | DS   |Frag  |     |Mgmt  |Data   |       |
     +------+------+------+------+------+------+-----+------+-------+-------+
     | 00   | 2b   | 4b   |  1b  |  1b  |  1b  | 1b  |  1b  |  1b   |  1b   |
```

**Type与Subtype字段**：

| Type (2位) | 含义 | Subtype示例 | 含义 |
|-----------|------|------------|------|
| 00 | Management | 0000 | Association Request |
| 00 | Management | 0100 | Probe Request |
| 00 | Management | 1000 | Beacon |
| 00 | Management | 1010 | Disassociation |
| 00 | Management | 1011 | Authentication |
| 01 | Control | 1011 | RTS |
| 01 | Control | 1100 | CTS |
| 01 | Control | 1101 | ACK |
| 01 | Control | 1000 | Block ACK |
| 10 | Data | 0000 | Data |
| 10 | Data | 0100 | Null Data |
| 10 | Data | 1000 | QoS Data |
| 11 | Extension | — | DMG/HE扩展 |

**To DS与From DS标志**决定地址字段的含义：

| To DS | From DS | 含义 | Addr1 | Addr2 | Addr3 | Addr4 |
|-------|---------|------|-------|-------|-------|-------|
| 0 | 0 | IBSS（Ad-hoc） | DA | SA | BSSID | 无 |
| 0 | 1 | AP→STA（下行） | DA | BSSID | SA | 无 |
| 1 | 0 | STA→AP（上行） | BSSID | SA | DA | 无 |
| 1 | 1 | WDS（桥接） | RA | TA | DA | SA |

- **DA**（Destination Address）：最终目的MAC地址
- **SA**（Source Address）：源MAC地址
- **BSSID**：基本服务集标识（通常是AP的MAC地址）
- **RA**（Receiver Address）：直接接收方MAC地址
- **TA**（Transmitter Address）：直接发送方MAC地址

```python
import struct

class WiFiFrameParser:
    """802.11 WiFi帧解析器"""
    
    # Type字段映射
    TYPE_MAP = {0: 'Management', 1: 'Control', 2: 'Data', 3: 'Extension'}
    
    # Subtype映射
    SUBTYPE_MAP = {
        (0, 0): 'Association Request',
        (0, 1): 'Association Response',
        (0, 2): 'Reassociation Request',
        (0, 3): 'Reassociation Response',
        (0, 4): 'Probe Request',
        (0, 5): 'Probe Response',
        (0, 8): 'Beacon',
        (0, 10): 'Disassociation',
        (0, 11): 'Authentication',
        (0, 12): 'Deauthentication',
        (1, 9): 'Block ACK Request',
        (1, 10): 'Block ACK',
        (1, 11): 'RTS',
        (1, 12): 'CTS',
        (1, 13): 'ACK',
        (2, 0): 'Data',
        (2, 4): 'Null Data',
        (2, 8): 'QoS Data',
    }
    
    @staticmethod
    def mac_to_str(mac_bytes):
        """将6字节MAC地址转换为字符串"""
        return ':'.join(f'{b:02x}' for b in mac_bytes)
    
    @staticmethod
    def parse_frame_control(fc_bytes):
        """解析Frame Control字段
        
        Args:
            fc_bytes: 2字节的Frame Control数据
        Returns:
            字典，包含各字段解析结果
        """
        fc = struct.unpack('<H', fc_bytes)[0]
        
        protocol_version = fc & 0x03
        frame_type = (fc >> 2) & 0x03
        frame_subtype = (fc >> 4) & 0x0F
        to_ds = (fc >> 8) & 0x01
        from_ds = (fc >> 9) & 0x01
        more_frag = (fc >> 10) & 0x01
        retry = (fc >> 11) & 0x01
        power_mgmt = (fc >> 12) & 0x01
        more_data = (fc >> 13) & 0x01
        protected = (fc >> 14) & 0x01
        order = (fc >> 15) & 0x01
        
        type_name = WiFiFrameParser.TYPE_MAP.get(frame_type, 'Unknown')
        subtype_name = WiFiFrameParser.SUBTYPE_MAP.get(
            (frame_type, frame_subtype), f'Subtype {frame_subtype}'
        )
        
        return {
            'protocol_version': protocol_version,
            'type': frame_type,
            'type_name': type_name,
            'subtype': frame_subtype,
            'subtype_name': subtype_name,
            'to_ds': to_ds,
            'from_ds': from_ds,
            'more_frag': more_frag,
            'retry': retry,
            'power_mgmt': power_mgmt,
            'more_data': more_data,
            'protected_frame': protected,
            'order': order,
        }
    
    @staticmethod
    def get_address_mode(to_ds, from_ds):
        """根据To DS/From DS标志确定地址字段含义"""
        if to_ds == 0 and from_ds == 0:
            return 'IBSS', ['DA', 'SA', 'BSSID', None]
        elif to_ds == 0 and from_ds == 1:
            return 'From AP', ['DA', 'BSSID', 'SA', None]
        elif to_ds == 1 and from_ds == 0:
            return 'To AP', ['BSSID', 'SA', 'DA', None]
        else:
            return 'WDS', ['RA', 'TA', 'DA', 'SA']
    
    @classmethod
    def parse_frame(cls, frame_bytes):
        """解析完整的802.11帧
        
        Args:
            frame_bytes: 帧的原始字节
        Returns:
            解析结果字典
        """
        if len(frame_bytes) < 24:
            raise ValueError(f"帧长度不足: {len(frame_bytes)} 字节 (最少24字节)")
        
        # 解析Frame Control
        fc = cls.parse_frame_control(frame_bytes[0:2])
        
        # Duration/ID
        duration = struct.unpack('<H', frame_bytes[2:4])[0]
        
        # 地址字段
        addr1 = cls.mac_to_str(frame_bytes[4:10])
        addr2 = cls.mac_to_str(frame_bytes[10:16])
        addr3 = cls.mac_to_str(frame_bytes[16:22])
        
        # 序列号控制
        seq_ctrl = struct.unpack('<H', frame_bytes[22:24])[0]
        seq_num = seq_ctrl >> 4
        frag_num = seq_ctrl & 0x0F
        
        # 确定地址模式
        mode, addr_names = cls.get_address_mode(fc['to_ds'], fc['from_ds'])
        
        result = {
            'frame_control': fc,
            'duration_id': duration,
            'sequence_number': seq_num,
            'fragment_number': frag_num,
            'address_mode': mode,
            'addr1 (RA)': addr1,
            'addr2 (TA)': addr2,
            'addr3': addr3,
        }
        
        # WDS模式有第四个地址
        if fc['to_ds'] and fc['from_ds']:
            if len(frame_bytes) >= 30:
                addr4 = cls.mac_to_str(frame_bytes[24:30])
                result['addr4'] = addr4
        
        # QoS数据帧有QoS Control字段
        if fc['type'] == 2 and (fc['subtype'] & 0x08):
            qos_offset = 24
            if fc['to_ds'] and fc['from_ds']:
                qos_offset = 30
            if len(frame_bytes) >= qos_offset + 2:
                qos_ctrl = struct.unpack('<H', frame_bytes[qos_offset:qos_offset+2])[0]
                tid = qos_ctrl & 0x0F
                ack_policy = (qos_ctrl >> 5) & 0x03
                result['qos'] = {
                    'tid': tid,
                    'ack_policy': ['Normal ACK', 'No ACK', 'No explicit ACK', 'Block ACK'][ack_policy],
                    'eosp': (qos_ctrl >> 4) & 0x01,
                }
        
        return result


# 示例：构造并解析一个QoS Data帧
def create_sample_frame():
    """创建一个示例802.11帧"""
    # Frame Control: Type=Data(10), Subtype=QoS Data(1000), To DS=1, From DS=0
    fc = 0x0088  # Protocol=0, Type=2(Data), Subtype=8(QoS), To DS=1
    duration = 0x006E  # 110 μs
    addr1 = bytes([0x00, 0x11, 0x22, 0x33, 0x44, 0x55])  # BSSID (AP)
    addr2 = bytes([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF])  # SA (Station)
    addr3 = bytes([0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB])  # DA (Destination)
    seq_ctrl = 0x01A3  # Seq=26, Frag=3
    qos_ctrl = 0x0007  # TID=7, Normal ACK
    
    frame = struct.pack('<HH', fc, duration) + addr1 + addr2 + addr3 + struct.pack('<H', seq_ctrl) + struct.pack('<H', qos_ctrl)
    
    return frame

# 解析示例帧
sample = create_sample_frame()
result = WiFiFrameParser.parse_frame(sample)

print("=== 802.11 帧解析结果 ===\n")
for key, value in result.items():
    if isinstance(value, dict):
        print(f"{key}:")
        for k, v in value.items():
            print(f"  {k}: {v}")
    else:
        print(f"{key}: {value}")
```

### 27.3.3 序列号与分段

序列控制字段（Sequence Control）占16位：

```
|<-- 12位 -->|<-- 4位 -->|
|  Seq Num   | Frag Num  |
```

- **序列号**（Seq Num）：0~4095循环，用于帧的顺序和去重
- **分段号**（Frag Num）：0~15，用于帧分段重组
- 接收方维护**重复检测缓存**，根据（SA, Seq Num, Frag Num）判断是否为重复帧

### 27.3.4 QoS控制

802.11e引入的QoS扩展定义了**接入类别**（Access Category, AC）：

| 优先级 | 接入类别 | 映射业务 | CWmin | CWmax | AIFS |
|-------|---------|---------|-------|-------|------|
| 最高 | AC_VO (Voice) | TID 6,7 | 3 | 7 | 34μs |
| 高 | AC_VI (Video) | TID 4,5 | 7 | 15 | 34μs |
| 中 | AC_BE (Best Effort) | TID 0,3 | 15 | 1023 | 43μs |
| 低 | AC_BK (Background) | TID 1,2 | 15 | 1023 | 79μs |

---

## 27.4 无线安全演进

无线网络的安全面临特殊挑战：信号在开放空间传播，任何在覆盖范围内的设备都可以接收到无线信号。因此，无线安全协议的演进是一个持续修补漏洞的过程。

### 27.4.1 WEP（Wired Equivalent Privacy）

WEP是802.11最初定义的安全协议，目标是提供与有线网络等价的安全性。

**WEP加密流程**：

```
明文消息
    │
    ├──→ CRC-32 ──→ ICV (完整性校验值)
    │                    │
    └────────────────────┤
                         ▼
              明文 || ICV (拼接)
                         │
    ┌────────────────────┤
    │                    ▼
密钥 (24位IV || 40/104位密钥)
    │                    │
    ▼                    ▼
  RC4 ──→ 密钥流 ──→ XOR ──→ 密文
                              │
    ┌─────────────────────────┘
    │
    ▼
  [IV | 密文] ──→ 发送
```

**WEP的致命缺陷**：

1. **IV太短**：仅24位，约1677万个IV值后必然重复。高流量下几小时内就会出现IV重用
2. **RC4弱点**：存在弱密钥（FMS攻击），可在收集足够IV后恢复密钥
3. **CRC-32非加密**：CRC-32是线性函数，攻击者可以翻转密文位并正确更新ICV
4. **无重放保护**：缺少序列号等防重放机制
5. **密钥管理缺失**：通常全网共享一个静态密钥

### 27.4.2 WPA（WiFi Protected Access）

WPA作为WEP的过渡方案，引入了TKIP（Temporal Key Integrity Protocol）：

**TKIP的主要改进**：
1. **48位IV**：避免IV重复
2. **每帧密钥**：通过IV+MAC地址+临时密钥混合生成每帧的RC4密钥
3. **Michael MIC**：替代CRC-32，提供更强的完整性保护
4. **序列号检查**：防止重放攻击
5. **仍使用RC4**：但通过密钥混合避免了弱密钥问题

### 27.4.3 WPA2（802.11i）

WPA2采用全新的加密框架CCMP：

**CCMP（Counter Mode CBC-MAC Protocol）**：
- 基于**AES-128**（高级加密标准，128位密钥）
- 使用**CTR模式**加密
- 使用**CBC-MAC**生成消息完整性校验码（MIC）
- 48位PN（Packet Number）防重放
- 8字节MIC提供强完整性保护

**四次握手过程**（4-Way Handshake）：

```
认证者(AP)                          请求者(STA)
    │                                    │
    │──── ANonce ───────────────────────>│  (消息1)
    │                                    │
    │              计算PTK = PRF(PMK, ANonce, SNonce, AA, SPA)
    │                                    │
    │<──── SNonce + MIC ─────────────────│  (消息2)
    │                                    │
    │   计算PTK, 安装密钥                   │
    │                                    │
    │──── GTK(加密) + MIC ──────────────>│  (消息3)
    │                                    │
    │              安装密钥                   │
    │                                    │
    │<──── ACK + MIC ────────────────────│  (消息4)
    │                                    │
    │         双方安装密钥，开始加密通信         │
```

**密钥层次结构**：

```
PSK/密码
    │
    ▼ (PBKDF2-SHA1, 4096轮)
PMK (Pairwise Master Key, 256位)
    │
    ▼ (PRF, 四次握手)
PTK (Pairwise Transient Key)
    │
    ├── KCK (Key Confirmation Key, 128位) → MIC计算
    ├── KEK (Key Encryption Key, 128位) → GTK加密传输
    └── TK (Temporal Key, 128/256位) → 数据加密
```

### 27.4.4 WPA3

WPA3在2018年发布，解决了WPA2的已知弱点：

**WPA3-Personal改进**：
1. **SAE握手**（Simultaneous Authentication of Equals）：替代PSK
   - 基于Dragonfly密钥交换协议
   - 抵抗离线字典攻击
   - 前向保密（Forward Secrecy）
2. **192位安全套件**（WPA3-Enterprise）：
   - GCMP-256加密
   - SHA-384用于密钥派生
   - 384位HMAC

**WPA3其他特性**：
- **OWE**（Opportunistic Wireless Encryption）：开放网络也加密
- **PMF**（Protected Management Frames）：强制保护管理帧
- **Transition Disable**：阻止降级攻击

```python
"""
无线安全协议对比分析
"""

class WirelessSecurity:
    """无线安全协议特性对比"""
    
    protocols = {
        'WEP': {
            'year': 1997,
            'encryption': 'RC4',
            'key_size': '64/128 bits',
            'iv_size': 24,
            'integrity': 'CRC-32',
            'key_mgmt': '静态密钥',
            'auth': 'Open System / Shared Key',
            'vulnerabilities': [
                'IV过短导致重用',
                'RC4弱密钥攻击 (FMS)',
                'CRC-32可篡改',
                '无前向保密',
                '无防重放',
            ],
            'status': '已淘汰，不安全'
        },
        'WPA': {
            'year': 2003,
            'encryption': 'RC4 (TKIP)',
            'key_size': '128 bits',
            'iv_size': 48,
            'integrity': 'Michael MIC',
            'key_mgmt': '动态密钥（四次握手）',
            'auth': 'PSK / 802.1X',
            'vulnerabilities': [
                '仍使用RC4',
                'Michael MIC存在碰撞攻击',
                '无前向保密',
            ],
            'status': '过渡方案，建议升级'
        },
        'WPA2': {
            'year': 2004,
            'encryption': 'AES-128 (CCMP)',
            'key_size': '128 bits',
            'iv_size': 48,
            'integrity': 'CBC-MAC (8字节MIC)',
            'key_mgmt': '四次握手',
            'auth': 'PSK / 802.1X',
            'vulnerabilities': [
                'KRACK攻击（密钥重装）',
                'PMKID离线字典攻击',
                '无前向保密（PSK模式）',
            ],
            'status': '广泛使用，安全可靠'
        },
        'WPA3': {
            'year': 2018,
            'encryption': 'AES-128/256 (CCMP/GCMP)',
            'key_size': '128/192/256 bits',
            'iv_size': 48,
            'integrity': 'GCMP / CCMP',
            'key_mgmt': 'SAE握手',
            'auth': 'SAE / 802.1X',
            'vulnerabilities': [
                'Dragonblood攻击（SAE实现缺陷）',
                '降级攻击（Transition模式）',
            ],
            'status': '最新标准，推荐使用'
        }
    }
    
    @classmethod
    def compare(cls):
        """打印安全协议对比表"""
        headers = ['特性', 'WEP', 'WPA', 'WPA2', 'WPA3']
        rows = [
            ('发布年份', ['year']),
            ('加密算法', ['encryption']),
            ('密钥长度', ['key_size']),
            ('IV/PN长度(位)', ['iv_size']),
            ('完整性校验', ['integrity']),
            ('密钥管理', ['key_mgmt']),
            ('认证方式', ['auth']),
            ('当前状态', ['status']),
        ]
        
        print("无线安全协议对比")
        print("=" * 90)
        print(f"{'特性':<16} {'WEP':<18} {'WPA':<18} {'WPA2':<18} {'WPA3':<18}")
        print("-" * 90)
        
        for label, keys in rows:
            values = []
            for proto in ['WEP', 'WPA', 'WPA2', 'WPA3']:
                val = cls.protocols[proto][keys[0]]
                values.append(str(val))
            print(f"{label:<16} {values[0]:<18} {values[1]:<18} {values[2]:<18} {values[3]:<18}")
        
        print("\n各协议已知漏洞：")
        for proto in ['WEP', 'WPA', 'WPA2', 'WPA3']:
            print(f"\n  {proto}:")
            for vuln in cls.protocols[proto]['vulnerabilities']:
                print(f"    - {vuln}")

WirelessSecurity.compare()
```

---

## 27.5 蜂窝网络演进

蜂窝网络经历了从1G到5G的演进，每一代都带来了根本性的技术变革。

### 27.5.1 2G GSM

**全球移动通信系统**（Global System for Mobile Communications, GSM）是第二代数字蜂窝标准：

**多址方式**：
- **FDMA**（频分多址）：将频谱划分为多个载波（每个200kHz）
- **TDMA**（时分多址）：每个载波上划分8个时隙

```
频率
  ▲
  │  ┌──┬──┬──┬──┬──┬──┬──┬──┐
f2 │  │T1│T2│T3│T4│T5│T6│T7│T8│  ← 载波2
  │  └──┴──┴──┴──┴──┴──┴──┴──┘
  │  ┌──┬──┬──┬──┬──┬──┬──┬──┐
f1 │  │T1│T2│T3│T4│T5│T6│T7│T8│  ← 载波1
  │  └──┴──┴──┴──┴──┴──┴──┴──┘
  └──────────────────────────→ 时间
       TDMA帧 (4.615ms)
       每时隙 0.577ms
```

**GSM架构组件**：
- **MS**（Mobile Station）：手机
- **BTS**（Base Transceiver Station）：基站收发器
- **BSC**（Base Station Controller）：基站控制器
- **MSC**（Mobile Switching Center）：移动交换中心
- **HLR**（Home Location Register）：归属位置寄存器
- **VLR**（Visitor Location Register）：访问位置寄存器

### 27.5.2 3G UMTS

**通用移动通信系统**（Universal Mobile Telecommunications System, UMTS）采用**WCDMA**（宽带码分多址）：

**关键技术**：
- **CDMA**：使用扩频码区分用户
- **码片速率**：3.84 Mcps
- **带宽**：5MHz
- **最大速率**：2 Mbps（HSPA+可达42 Mbps下行）

**UMTS网络架构**：
- **Node B**：对应GSM的BTS
- **RNC**（Radio Network Controller）：无线网络控制器
- **CN**（Core Network）：核心网（CS域+PS域）

### 27.5.3 4G LTE

**长期演进**（Long-Term Evolution, LTE）是第四代移动通信标准，提供全IP化网络。

**关键技术**：
- **OFDMA**（下行）：正交频分多址
- **SC-FDMA**（上行）：单载波频分多址
- **MIMO**：最多8×8下行，4×4上行
- **带宽**：1.4/3/5/10/15/20 MHz
- **峰值速率**：下行300 Mbps，上行75 Mbps（Cat 4）

**LTE OFDMA资源分配**：

```
频率(RB)
  ▲
  │  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
12 │  │  │  │  │  │  │  │  │  │  │  │ ← 子载波12
   │  ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
 3 │  │  │  │  │  │  │  │  │  │  │  │ ← 子载波3
 2 │  │  │  │  │  │  │  │  │  │  │  │ ← 子载波2
 1 │  │U1│U2│U3│U1│U4│U2│U3│U1│U4│U2│ ← 子载波1
   │  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
   └──────────────────────────────────→ 时间
      TTI1  TTI2  TTI3  TTI4  TTI5 ...
      
   每个方格 = 1个资源块(RB)的一部分
   U1, U2, U3, U4 = 不同用户的资源分配
```

<div data-component="CellularScheduler"></div>

### 27.5.4 LTE EPC架构核心网元

**演进分组核心网**（Evolved Packet Core, EPC）是LTE的核心网络架构：

```
                    ┌─────────────┐
                    │     HSS     │ ← 归属用户服务器
                    │  (用户签约   │    存储用户签约信息
                    │   数据库)    │
                    └──────┬──────┘
                           │ S6a
                           ▼
┌────────┐  S1-MME  ┌─────────────┐  S11    ┌──────────┐
│        │─────────→│     MME     │────────→│   SGW    │
│ eNodeB │          │ (移动性管理  │         │(服务网关) │
│  (基站)│          │   实体)     │         │          │
│        │─────────→│             │         │          │
└────────┘  S1-U    └─────────────┘         └────┬─────┘
                               S5                │
                               ▼                 │
                         ┌──────────┐            │
                         │   PGW    │←───────────┘
                         │(PDN网关) │  S5/S8
                         │          │
                         └────┬─────┘
                              │ SGi
                              ▼
                         外部网络
                        (Internet)
```

**核心网元功能详解**：

**MME（Mobility Management Entity，移动性管理实体）**：
- 控制面核心节点，不处理用户数据
- **附着/去附着**（Attach/Detach）：管理终端与网络的连接
- **跟踪区管理**（TAI List）：维护终端的位置信息
- **承载管理**：建立/修改/删除EPS承载
- **鉴权**：与HSS配合完成AKA鉴权
- **寻呼**：在网络侧发起呼叫时寻呼终端
- **切换信令**：处理X2和S1切换的控制面

**SGW（Serving Gateway，服务网关）**：
- 用户面锚点
- **数据转发**：在eNodeB之间切换时缓存下行数据
- **计费**：收集用户面流量信息
- **合法监听**：支持用户面数据的合法监听
- 在**S1切换**时作为用户面锚点，避免核心网重路由

**PGW（PDN Gateway，PDN网关）**：
- 连接外部数据网络（Internet）的网关
- **IP地址分配**：为UE分配IP地址
- **策略执行**：执行QoS策略和流量过滤
- **计费**：在线/离线计费
- **NAT**：网络地址转换
- **深度包检测**（DPI）：识别应用类型

**HSS（Home Subscriber Server，归属用户服务器）**：
- 用户签约数据库
- **用户标识**：IMSI、MSISDN
- **鉴权向量**：生成和存储鉴权五元组
- **签约数据**：APN、QoS profile、漫游限制
- **位置信息**：用户当前注册的MME

### 27.5.5 5G NR

**5G新空口**（New Radio, NR）定义了三大应用场景：

| 场景 | 全称 | 典型应用 | 关键指标 |
|------|------|---------|---------|
| eMBB | Enhanced Mobile Broadband | 4K/8K视频、VR/AR | 峰值20Gbps |
| URLLC | Ultra-Reliable Low-Latency | 自动驾驶、远程手术 | 时延<1ms |
| mMTC | Massive Machine-Type Comm. | 物联网、智慧城市 | 100万设备/km² |

**5G关键技术**：

**毫米波（mmWave）**：
- 频段：FR1（410MHz-7.125GHz）、FR2（24.25-52.6GHz）
- mmWave频段带宽大（400MHz+），但路径损耗大、穿透力弱
- 需要大规模天线阵列（Massive MIMO）和波束赋形

**网络切片（Network Slicing）**：
- 在同一物理基础设施上创建多个虚拟网络
- 每个切片可独立配置：QoS、安全策略、资源分配
- 三大场景可在同一网络上同时服务

**CU/DU分离架构**：

```
┌─────────────────────────────────────────────┐
│                    gNB                       │
│  ┌─────────┐         ┌────────────────────┐ │
│  │   CU    │  F1接口  │       DU           │ │
│  │(集中单元)│────────→│(分布式单元)         │ │
│  │         │         │  ┌────┐ ┌────┐     │ │
│  │ ┌─────┐ │         │  │DU1 │ │DU2 │ ...│ │
│  │ │CU-CP│ │         │  └──┬─┘ └──┬─┘     │ │
│  │ └─────┘ │         └─────┼──────┼───────┘ │
│  │ ┌─────┐ │               │      │         │
│  │ │CU-UP│ │          eCPRI │      │ eCPRI  │
│  │ └─────┘ │               │      │         │
│  └─────────┘          ┌────┴─┐ ┌──┴───┐    │
│                       │ RRU1 │ │ RRU2 │    │
│                       │(射频)│ │(射频)│    │
│                       └──────┘ └──────┘    │
└─────────────────────────────────────────────┘
```

- **CU-CP**（Centralized Unit - Control Plane）：集中式控制面处理
- **CU-UP**（Centralized Unit - User Plane）：集中式用户面处理
- **DU**（Distributed Unit）：分布式单元，处理实时性要求高的功能
- **RRU**（Remote Radio Unit）：远端射频单元

---

## 27.6 移动IP

移动IP（Mobile IP）允许移动节点在改变网络接入点时保持IP连接不变，是移动通信网络支持IP移动性的基础协议。

### 27.6.1 基本概念与实体

**移动IP的核心实体**：

1. **移动节点**（Mobile Node, MN）：改变接入点时仍需保持通信的设备
2. **家乡代理**（Home Agent, HA）：位于家乡网络的路由器，维护MN的位置信息
3. **外地代理**（Foreign Agent, FA）：位于外地网络的路由器，为MN提供路由服务
4. **通信对端**（Correspondent Node, CN）：与MN通信的对端节点

**地址概念**：
- **家乡地址**（Home Address）：MN的永久IP地址，属于家乡网络的地址空间
- **转交地址**（Care-of Address, CoA）：MN在外地网络获得的临时地址

### 27.6.2 代理发现

移动节点需要发现自己是否在家乡网络，以及外地网络的信息：

**代理通告**（Agent Advertisement）：
- HA和FA周期性广播ICMP Router Advertisement消息
- 消息中携带**移动性扩展**信息
- 包含：代理类型（HA/FA）、家乡网络前缀、转交地址、注册生命期等

**代理请求**（Agent Solicitation）：
- MN主动发送ICMP Router Solicitation
- 要求代理立即回复通告消息

```
代理通告消息格式：

   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |     Type      |     Code      |           Checksum            |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |   Num Addrs   |Addr Entry Size|           Lifetime            |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |                       Router Address[1]                       |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |                      Preference Level[1]                      |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |    移动性扩展 (Type=16)                                        |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |     Type=16   |    Length      |    Sequence Number            |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |R|B|H|F|M|G|rsv|  Reg Lifetime |          ...                  |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
   |                     Care-of Address[1]                        |
   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

标志位含义：
- **H**：Home Agent（家乡代理标志）
- **F**：Foreign Agent（外地代理标志）
- **M**：支持最小封装
- **G**：支持GRE封装
- **B**：愿意接收反向隧道的数据

### 27.6.3 注册过程

当MN发现FA并获得CoA后，需要向HA注册：

**通过FA注册的过程**：

```
MN                    FA                    HA
│                      │                     │
│── Registration Req ─→│                     │
│  (CoA, HA addr,      │                     │
│   MN addr, Lifetime) │                     │
│                      │── Registration Req ─→│
│                      │   (转发MN的请求)      │
│                      │                     │
│                      │   HA创建/更新绑定表项   │
│                      │   (MN addr → CoA)    │
│                      │                     │
│                      │←─ Registration Rep ──│
│                      │   (接受/拒绝)         │
│←─ Registration Rep ──│                     │
│   (结果通知MN)        │                     │
│                      │                     │
│  注册成功！            │                     │
│  HA现在知道MN在CoA    │                     │
```

**直接向HA注册**（无需FA，使用Co-located CoA）：

```
MN                                        HA
│                                          │
│── Registration Request ─────────────────→│
│  (CoA, HA addr, MN addr, Lifetime,       │
│   MN-HA Authentication Extension)        │
│                                          │
│    HA创建/更新绑定表项                       │
│    (MN addr → CoA)                        │
│                                          │
│←── Registration Reply ───────────────────│
│    (成功/失败 + HA-MN Auth Extension)     │
│                                          │
│  注册成功，直接通信                          │
```

### 27.6.4 数据传输与三角路由

**三角路由**（Triangle Routing）是Mobile IPv4的基本数据传输方式：

```
              CN (通信对端)
             /            \
            /              \
           /    三角路由      \
          /                  \
         ▼                    ▼
       MN ──────────────────→ HA
     (外地网络)  (隧道封装)  (家乡网络)
         ▲
         │
         └── 直接回复（反向三角路由，可选）
```

**三角路由过程**：

1. **CN → MN**：CN将数据包发往MN的家乡地址 → 包被路由到家乡网络 → HA截获 → HA通过隧道将包转发给MN的CoA
2. **MN → CN**：MN直接将数据包发往CN（源地址为家乡地址）

**三角路由的问题**：
- CN → MN的路径绕经HA，增加延迟
- HA成为瓶颈，负载集中
- 不是最优路径

### 27.6.5 路由优化

为解决三角路由问题，Mobile IP定义了路由优化扩展：

```
CN (通信对端)
    │
    │ 1. CN发送绑定更新请求给HA
    │    HA返回MN的绑定信息（CoA）
    │
    │ 2. CN缓存MN的绑定（家乡地址 → CoA）
    │
    │ 3. CN直接通过隧道发往MN的CoA
    │    (不再绕经HA)
    │
    ▼
MN (外地网络)
```

路由优化后：
- CN直接将数据发往MN的CoA，不绕经HA
- 需要CN支持Mobile IP扩展
- 需要**绑定更新**（Binding Update）机制

### 27.6.6 双向隧道

当MN不希望暴露自己的位置信息时，可以使用**双向隧道**：

```
CN ←→ HA ←→ MN
         ↑
      所有流量都经过HA
```

- MN发出的数据也通过反向隧道经HA转发
- CN看到的源地址始终是MN的家乡地址
- 保护了MN的位置隐私，但增加了路径长度

```python
"""
移动IP隧道封装模拟
"""

import struct
import socket
from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class IPHeader:
    """IP头部"""
    version: int = 4
    ihl: int = 5
    tos: int = 0
    total_length: int = 0
    identification: int = 0
    flags: int = 0
    fragment_offset: int = 0
    ttl: int = 64
    protocol: int = 0  # 6=TCP, 17=UDP, 4=IP-in-IP
    checksum: int = 0
    src_ip: str = '0.0.0.0'
    dst_ip: str = '0.0.0.0'
    
    def to_bytes(self):
        """序列化为字节"""
        src = socket.inet_aton(self.src_ip)
        dst = socket.inet_aton(self.dst_ip)
        header = struct.pack('!BBHHHBBH4s4s',
            (self.version << 4) | self.ihl,
            self.tos,
            self.total_length,
            self.identification,
            (self.flags << 13) | self.fragment_offset,
            self.ttl,
            self.protocol,
            self.checksum,
            src,
            dst
        )
        return header
    
    @classmethod
    def from_bytes(cls, data):
        """从字节反序列化"""
        fields = struct.unpack('!BBHHHBBH4s4s', data[:20])
        return cls(
            version=fields[0] >> 4,
            ihl=fields[0] & 0x0F,
            tos=fields[1],
            total_length=fields[2],
            identification=fields[3],
            flags=fields[4] >> 13,
            fragment_offset=fields[4] & 0x1FFF,
            ttl=fields[5],
            protocol=fields[6],
            checksum=fields[7],
            src_ip=socket.inet_ntoa(fields[8]),
            dst_ip=socket.inet_ntoa(fields[9])
        )


class HomeAgent:
    """家乡代理实现"""
    
    def __init__(self, ha_ip, ha_interface_ip):
        """
        Args:
            ha_ip: HA的IP地址
            ha_interface_ip: HA接口地址
        """
        self.ha_ip = ha_ip
        self.interface_ip = ha_interface_ip
        self.binding_table: Dict[str, dict] = {}  # MN家乡地址 → 绑定信息
    
    def register(self, mn_home_addr, coa, lifetime, auth_ext=None):
        """处理MN的注册请求
        
        Args:
            mn_home_addr: MN的家乡地址
            coa: MN的转交地址
            lifetime: 注册生命期（秒）
            auth_ext: 认证扩展
        Returns:
            注册结果 (code, message)
        """
        # 验证认证扩展
        if auth_ext is None:
            return (131, "缺少认证扩展")
        
        # 创建或更新绑定表项
        self.binding_table[mn_home_addr] = {
            'coa': coa,
            'lifetime': lifetime,
            'sequence': self.binding_table.get(mn_home_addr, {}).get('sequence', 0) + 1,
            'registered_at': 0,  # 简化：实际应记录时间戳
        }
        
        print(f"[HA] 注册成功: {mn_home_addr} -> {coa}, 生命期={lifetime}s")
        return (0, "注册成功")
    
    def encapsulate_packet(self, original_packet):
        """IP-in-IP隧道封装
        
        将发往MN家乡地址的包封装后发往MN的CoA
        """
        # 解析原始IP头
        orig_header = IPHeader.from_bytes(original_packet)
        
        # 查找绑定
        if orig_header.dst_ip not in self.binding_table:
            print(f"[HA] 无绑定信息，丢弃数据包（目的: {orig_header.dst_ip}）")
            return None
        
        coa = self.binding_table[orig_header.dst_ip]['coa']
        
        # 创建外部IP头（隧道头）
        tunnel_header = IPHeader(
            version=4,
            ihl=5,
            total_length=20 + len(original_packet),  # 外部头+原始包
            ttl=64,
            protocol=4,  # IP-in-IP封装
            src_ip=self.ha_ip,
            dst_ip=coa
        )
        
        # 封装：外部头 + 原始IP包
        encapsulated = tunnel_header.to_bytes() + original_packet
        
        print(f"[HA] 隧道封装:")
        print(f"  外部: {self.ha_ip} -> {coa}")
        print(f"  内部: {orig_header.src_ip} -> {orig_header.dst_ip}")
        print(f"  封装后大小: {len(encapsulated)} 字节")
        
        return encapsulated
    
    def decapsulate_packet(self, encapsulated_packet):
        """隧道解封装
        
        从外部IP头提取原始IP包
        """
        outer_header = IPHeader.from_bytes(encapsulated_packet)
        
        if outer_header.protocol != 4:
            print("[HA] 非IP-in-IP封装包")
            return None
        
        # 提取内部包
        inner_offset = outer_header.ihl * 4
        inner_packet = encapsulated_packet[inner_offset:]
        inner_header = IPHeader.from_bytes(inner_packet)
        
        print(f"[HA] 解封装:")
        print(f"  外部: {outer_header.src_ip} -> {outer_header.dst_ip}")
        print(f"  内部: {inner_header.src_ip} -> {inner_header.dst_ip}")
        
        return inner_packet
    
    def get_binding_table(self):
        """获取绑定表"""
        return self.binding_table


class MobileNode:
    """移动节点"""
    
    def __init__(self, home_addr, home_network):
        self.home_addr = home_addr
        self.home_network = home_network
        self.current_coa = None
        self.current_network = None
    
    def move_to(self, new_network, foreign_agent_ip=None):
        """移动到新网络"""
        self.current_network = new_network
        if foreign_agent_ip:
            self.current_coa = foreign_agent_ip
        else:
            # 使用co-located CoA
            self.current_coa = f"10.0.{new_network}.100"
        
        print(f"[MN] 移动到网络 {new_network}")
        print(f"[MN] 获得CoA: {self.current_coa}")
    
    def send_registration(self, ha):
        """向HA发送注册请求"""
        print(f"[MN] 向HA ({ha.ha_ip}) 发送注册请求")
        print(f"[MN]   家乡地址: {self.home_addr}")
        print(f"[MN]   转交地址: {self.current_coa}")
        
        auth = {'spi': 100, 'key': 'secret_key'}
        code, msg = ha.register(
            self.home_addr,
            self.current_coa,
            lifetime=3600,
            auth_ext=auth
        )
        
        if code == 0:
            print(f"[MN] 注册成功: {msg}")
        else:
            print(f"[MN] 注册失败 (code={code}): {msg}")
    
    def send_packet_to(self, cn_ip, ha, payload="Hello"):
        """通过HA向CN发送数据包"""
        # MN构造原始包（源地址为家乡地址）
        packet = IPHeader(
            total_length=20 + len(payload),
            protocol=17,
            src_ip=self.home_addr,
            dst_ip=cn_ip
        ).to_bytes() + payload.encode()
        
        if self.current_network != self.home_network:
            # 在外地网络：通过反向隧道发送
            print(f"[MN] 通过反向隧道发送到HA")
            tunnel_header = IPHeader(
                total_length=20 + len(packet),
                protocol=4,
                src_ip=self.current_coa,
                dst_ip=ha.ha_ip
            ).to_bytes()
            tunneled = tunnel_header + packet
            # 实际网络中tunneled会被发送到HA
            print(f"[MN] 发送: {self.current_coa} -> {ha.ha_ip} (封装)")
        else:
            print(f"[MN] 在家乡网络，直接发送到 {cn_ip}")


# 模拟移动IP过程
print("=" * 60)
print("移动IP模拟")
print("=" * 60)

# 创建实体
ha = HomeAgent(ha_ip="192.168.1.1", ha_interface_ip="192.168.1.1")
mn = MobileNode(home_addr="192.168.1.100", home_network=1)

# CN的IP
cn_ip = "10.0.5.200"

print("\n--- 场景1：MN在家乡网络 ---")
mn.send_packet_to(cn_ip, ha)

print("\n--- 场景2：MN移动到外地网络 ---")
mn.move_to(new_network=5, foreign_agent_ip="10.0.5.1")
mn.send_registration(ha)

print("\n--- 场景3：CN发送数据给MN（三角路由）---")
# CN发出的包
original = IPHeader(
    total_length=20 + 5,
    protocol=17,
    src_ip=cn_ip,
    dst_ip=mn.home_addr
).to_bytes() + b"Hello"

print(f"\n[CN] 发送数据: {cn_ip} -> {mn.home_addr}")
print("[HA] 拦截数据包，执行隧道封装")
encapsulated = ha.encapsulate_packet(original)

print("\n[MN] 收到封装包，解封装")
if encapsulated:
    inner = ha.decapsulate_packet(encapsulated)

print("\n--- 场景4：查看绑定表 ---")
bindings = ha.get_binding_table()
for home_addr, info in bindings.items():
    print(f"  {home_addr} -> CoA: {info['coa']}, "
          f"生命期: {info['lifetime']}s, "
          f"序列号: {info['sequence']}")
```

<div data-component="MobileIPTunnelEngine"></div>

---

## 27.7 切换管理

切换（Handover/Handoff）是移动网络中的核心机制，确保用户在移动过程中保持连续的通信服务。

### 27.7.1 切换分类

**按网络类型分类**：

| 类型 | 定义 | 示例 |
|------|------|------|
| 水平切换（Horizontal） | 同一接入技术内的切换 | LTE → LTE |
| 垂直切换（Vertical） | 不同接入技术间的切换 | WiFi → LTE |

**按决策方式分类**：

| 类型 | 决策方 | 特点 |
|------|--------|------|
| 移动台控制（MCHO） | MN/UE | 终端决定切换，通知网络 |
| 网络控制（NCHO） | 网络 | 网络控制切换，终端被动执行 |
| 移动台辅助（MAHO） | 网络+终端 | 终端测量，网络决策（LTE） |

**按切换时连接方式分类**：
- **硬切换**（Hard Handover）：先断后连（Break-before-Make）
- **软切换**（Soft Handover）：先连后断（Make-before-Break，CDMA特有）

### 27.7.2 切换过程

切换过程通常分为三个阶段：

```
┌──────────────────────────────────────────────┐
│                 切换过程                        │
│                                              │
│  ┌────────────┐  ┌────────────┐  ┌─────────┐│
│  │  测量阶段    │→│  决策阶段    │→│  执行阶段 ││
│  │ (Measurement)│ │ (Decision)  │ │(Execution)│
│  └────────────┘  └────────────┘  └─────────┘│
│       │               │              │       │
│  UE测量RSRP/    网络根据测量     执行切换信令   │
│  RSRQ/SINR,     报告和算法       和数据迁移    │
│  上报测量报告    判断是否切换                   │
└──────────────────────────────────────────────┘
```

**测量阶段**：
- UE周期性测量服务小区和邻区的信号质量
- 测量指标：**RSRP**（参考信号接收功率）、**RSQ**（参考信号接收质量）、**SINR**（信号与干扰噪声比）
- 测量结果通过**测量报告**（Measurement Report）上报给基站

**决策阶段**：
- 基站根据测量报告和预设算法决定是否触发切换
- 常见决策准则：
  - **A3事件**：邻区RSRP比服务小区好于一个偏移量
  - **A5事件**：服务小区低于阈值1，邻区高于阈值2
  - **TTT**（Time to Trigger）：满足条件持续一定时间才触发

**执行阶段**：
- 建立新基站的连接
- 迁移上下文信息和数据通路
- 释放旧基站的连接

### 27.7.3 LTE切换

LTE中的切换是**UE辅助的网络控制切换**（MAHO + NCHO），全部在基站控制下完成。

**X2切换**（基站间切换）：

```
Source eNB                      Target eNB
    │                               │
    │  1. UE上报测量报告              │
    │←──────────────────────────────│
    │                               │
    │  2. Source决定切换              │
    │  ──── Handover Request ──────→│
    │                               │
    │  3. Target准备资源              │
    │←──── Handover Request ACK ────│
    │                               │
    │  4. 通知UE切换                  │
    │──── RRC Connection Reconfig ─→│  (含mobilityControlInfo)
    │                               │
    │  5. UE接入Target小区            │
    │                               │
    │  6. 数据前转                    │
    │─── SN Status Transfer ───────→│
    │─── Data Forwarding ──────────→│
    │                               │
    │  7. MME/SGW更新路径              │
    │            MME                  │
    │──── Path Switch Request ─────→│
    │←─── Path Switch ACK ─────────│
    │                               │
    │  8. 释放Source资源              │
    │←──── UE Context Release ──────│
    │                               │
```

**X2切换的条件**：
- Source eNB和Target eNB之间有X2接口连接
- 适用于同一MME下的基站间切换
- 用户面数据直接在基站间前转，不经过核心网

**S1切换**（通过核心网切换）：

```
Source eNB        MME          Target eNB
    │              │               │
    │── Handover Required ───────→│
    │              │               │
    │              │── Handover Request ──→│
    │              │               │
    │              │←─ Handover Request ACK ─│
    │              │               │
    │←─ Handover Command ─────────│
    │              │               │
    │── SN Status Transfer ──────→│
    │              │               │
    │── Data Forwarding ─────────→│ (经SGW)
    │              │               │
    │              │   UE接入Target │
    │              │               │
    │              │←─ Handover Notify ────│
    │              │               │
    │              │  更新SGW路径     │
    │              │               │
```

**S1切换的使用场景**：
- 基站间无X2接口
- 跨MME切换
- X2切换失败的回退方案

```python
"""
LTE切换过程模拟
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class MeasurementReport:
    """UE测量报告"""
    serving_rsrp: float  # dBm
    serving_rsrq: float  # dB
    serving_sinr: float  # dB
    neighbor_cells: List[dict] = field(default_factory=list)
    timestamp: float = 0.0

@dataclass
class Cell:
    """基站小区"""
    cell_id: str
    pci: int  # Physical Cell ID
    earfcn: int  # 频点
    rsrp_base: float  # 基础RSRP
    x2_neighbors: List[str] = field(default_factory=list)
    load: float = 0.0  # 负载率 0~1

class HandoverManager:
    """切换管理器"""
    
    def __init__(self, a3_offset=3.0, ttt_ms=320, hysteresis=1.0):
        """
        Args:
            a3_offset: A3事件偏移量 (dB)
            ttt_ms: Time to Trigger (ms)
            hysteresis: 迟滞值 (dB)
        """
        self.a3_offset = a3_offset
        self.ttt_ms = ttt_ms
        self.hysteresis = hysteresis
    
    def evaluate_a3_event(self, report: MeasurementReport) -> Optional[str]:
        """评估A3事件：邻区RSRP > 服务小区RSRP + offset
        
        Returns:
            触发切换的目标小区ID，或None
        """
        best_neighbor = None
        best_rsrp = float('-inf')
        
        for neighbor in report.neighbor_cells:
            if neighbor['rsrp'] > report.serving_rsrp + self.a3_offset - self.hysteresis:
                if neighbor['rsrp'] > best_rsrp:
                    best_rsrp = neighbor['rsrp']
                    best_neighbor = neighbor['cell_id']
        
        return best_neighbor
    
    def evaluate_a5_event(self, report: MeasurementReport, 
                          threshold1=-110, threshold2=-105) -> Optional[str]:
        """评估A5事件：服务小区 < Thresh1 且 邻区 > Thresh2
        
        Returns:
            触发切换的目标小区ID，或None
        """
        if report.serving_rsrp >= threshold1:
            return None
        
        best_neighbor = None
        best_rsrp = float('-inf')
        
        for neighbor in report.neighbor_cells:
            if neighbor['rsrp'] > threshold2:
                if neighbor['rsrp'] > best_rsrp:
                    best_rsrp = neighbor['rsrp']
                    best_neighbor = neighbor['cell_id']
        
        return best_neighbor
    
    def decide_handover_type(self, source: Cell, target: Cell) -> str:
        """决定切换类型（X2或S1）"""
        if target.cell_id in source.x2_neighbors:
            return 'X2'
        else:
            return 'S1'


class LTEHandoverSimulator:
    """LTE切换模拟器"""
    
    def __init__(self):
        self.cells = {}
        self.handover_count = 0
        self.handover_stats = {'X2': 0, 'S1': 0, 'failed': 0}
        self.manager = HandoverManager()
    
    def add_cell(self, cell: Cell):
        """添加基站小区"""
        self.cells[cell.cell_id] = cell
    
    def simulate_path_loss(self, distance_km, frequency_mhz=2600):
        """简化的路径损耗模型"""
        # Okumura-Hata 城市模型简化
        if distance_km <= 0:
            return -50
        pl = 128.1 + 37.6 * (10 ** 0) * __import__('math').log10(max(distance_km, 0.01))
        return -pl
    
    def generate_measurement_report(self, ue_x, ue_y, serving_cell_id):
        """生成UE的测量报告
        
        Args:
            ue_x, ue_y: UE位置
            serving_cell_id: 服务小区ID
        Returns:
            MeasurementReport
        """
        # 简化：假设各小区在固定位置
        cell_positions = {
            'cell_1': (0, 0),
            'cell_2': (500, 0),
            'cell_3': (250, 433),
            'cell_4': (750, 433),
        }
        
        import math
        
        measurements = []
        serving_rsrp = None
        
        for cell_id, (cx, cy) in cell_positions.items():
            if cell_id not in self.cells:
                continue
            
            dist = math.sqrt((ue_x - cx)**2 + (ue_y - cy)**2)
            dist_km = dist / 1000
            
            # 简化的RSRP计算
            base_rsrp = self.cells[cell_id].rsrp_base
            path_loss = 128.1 + 37.6 * math.log10(max(dist_km, 0.01))
            rsrp = base_rsrp - path_loss + random.uniform(-2, 2)
            
            if cell_id == serving_cell_id:
                serving_rsrp = rsrp
            else:
                measurements.append({
                    'cell_id': cell_id,
                    'rsrp': rsrp,
                    'distance': dist
                })
        
        if serving_rsrp is None:
            serving_rsrp = -120
        
        return MeasurementReport(
            serving_rsrp=serving_rsrp,
            serving_rsrq=serving_rsrp - 10 + random.uniform(-1, 1),
            serving_sinr=serving_rsrp + 174 + random.uniform(-2, 2),
            neighbor_cells=measurements,
            timestamp=self.handover_count * 0.1
        )
    
    def perform_handover(self, source_id, target_id, method):
        """执行切换"""
        source = self.cells[source_id]
        target = self.cells[target_id]
        
        print(f"  执行 {method} 切换: {source_id} -> {target_id}")
        
        if method == 'X2':
            print(f"    1. Source eNB 发送 Handover Request 到 Target eNB")
            print(f"    2. Target eNB 准备资源，回复 Handover ACK")
            print(f"    3. 发送 RRC Connection Reconfiguration 给 UE")
            print(f"    4. UE 接入 Target 小区 (PCI={target.pci})")
            print(f"    5. SN Status Transfer: 数据前转")
            print(f"    6. Path Switch: 更新 SGW 路径")
            print(f"    7. UE Context Release: 释放 Source 资源")
            self.handover_stats['X2'] += 1
        else:
            print(f"    1. Source eNB 发送 Handover Required 到 MME")
            print(f"    2. MME 发送 Handover Request 到 Target eNB")
            print(f"    3. Target eNB 准备资源，回复 ACK")
            print(f"    4. MME 下发 Handover Command 给 Source")
            print(f"    5. UE 接入 Target 小区 (PCI={target.pci})")
            print(f"    6. 数据经 SGW 前转")
            print(f"    7. Path Switch 完成")
            self.handover_stats['S1'] += 1
        
        self.handover_count += 1
    
    def run_simulation(self, serving_cell='cell_1'):
        """运行切换模拟"""
        print("=" * 60)
        print("LTE 切换模拟")
        print("=" * 60)
        
        # 模拟UE移动轨迹：从cell_1向cell_2方向移动
        positions = [(100, 50), (200, 60), (300, 50), (400, 40), (500, 30)]
        current_serving = serving_cell
        
        for i, (x, y) in enumerate(positions):
            print(f"\n--- 时刻 {i+1}: UE位于 ({x}, {y}) ---")
            
            # 生成测量报告
            report = self.generate_measurement_report(x, y, current_serving)
            print(f"  服务小区 ({current_serving}): RSRP={report.serving_rsrp:.1f} dBm")
            for nb in report.neighbor_cells:
                print(f"  邻区 ({nb['cell_id']}): RSRP={nb['rsrp']:.1f} dBm")
            
            # 评估A3事件
            target = self.manager.evaluate_a3_event(report)
            
            if target:
                print(f"  ** A3事件触发: 目标小区 {target} **")
                method = self.manager.decide_handover_type(
                    self.cells[current_serving],
                    self.cells[target]
                )
                self.perform_handover(current_serving, target, method)
                current_serving = target
            else:
                print(f"  未触发切换")
        
        print(f"\n{'=' * 60}")
        print(f"切换统计:")
        print(f"  总切换次数: {self.handover_count}")
        print(f"  X2切换: {self.handover_stats['X2']}")
        print(f"  S1切换: {self.handover_stats['S1']}")
        print(f"  失败: {self.handover_stats['failed']}")


# 运行模拟
sim = LTEHandoverSimulator()

# 添加基站小区
sim.add_cell(Cell('cell_1', pci=101, earfcn=3000, rsrp_base=-40, x2_neighbors=['cell_2', 'cell_3']))
sim.add_cell(Cell('cell_2', pci=102, earfcn=3000, rsrp_base=-40, x2_neighbors=['cell_1', 'cell_4']))
sim.add_cell(Cell('cell_3', pci=103, earfcn=3050, rsrp_base=-40, x2_neighbors=['cell_1', 'cell_4']))
sim.add_cell(Cell('cell_4', pci=104, earfcn=3050, rsrp_base=-40, x2_neighbors=['cell_2', 'cell_3']))

sim.run_simulation()
```

<div data-component="HandoverSimulation"></div>

### 27.7.4 移动性管理与寻呼

**跟踪区更新**（Tracking Area Update, TAU）：
- UE在不同跟踪区（TA）之间移动时触发
- 更新MME中的UE位置信息
- TAU完成后，网络知道UE当前所在的TA

**寻呼**（Paging）：
- 当有下行数据到达UE但UE处于空闲态时
- MME在UE注册的TA List内所有小区广播寻呼消息
- UE收到寻呼后，执行服务请求过程，恢复连接

```bash
# 查看无线网卡信息
iwconfig wlan0

# 扫描WiFi网络
sudo iw dev wlan0 scan | grep -E 'BSS|SSID|signal|freq|WPA|RSN'

# 连接到指定WiFi
sudo iw dev wlan0 connect "NetworkName"

# 查看已连接的详细信息
iw dev wlan0 link

# 查看站点统计信息（信号强度、比特率等）
iw dev wlan0 station dump

# 设置WiFi功率管理（省电模式）
sudo iw dev wlan0 set power_save on

# 查看支持的频率
iw phy phy0 info | grep -A 20 "Frequencies"

# 使用tcpdump抓取WiFi管理帧
sudo tcpdump -i wlan0 -e -s 256 'type mgt'

# 使用airodump-ng扫描（需要aircrack-ng套件）
# sudo airodump-ng wlan0mon
```

---

## 27.8 综合实例与习题

### 27.8.1 综合实例：WiFi帧捕获与分析

```python
"""
WiFi帧捕获与分析工具（模拟）
演示如何解析实际的WiFi帧结构
"""

import struct
import random

def generate_random_mac():
    """生成随机MAC地址"""
    return bytes([random.randint(0, 255) for _ in range(6)])

def mac_to_str(mac_bytes):
    """MAC地址转字符串"""
    return ':'.join(f'{b:02x}' for b in mac_bytes)

def create_beacon_frame(ssid, bssid=None, channel=6):
    """创建一个Beacon帧"""
    if bssid is None:
        bssid = generate_random_mac()
    
    # Frame Control: Management(00), Beacon(1000)
    fc = (0 << 0) | (0 << 2) | (8 << 4) | (0 << 8) | (0 << 9)
    duration = 0x0000
    
    # 地址: DA=broadcast, SA=BSSID, BSSID=BSSID
    da = bytes([0xFF]*6)
    sa = bssid
    bssid_bytes = bssid
    
    seq_num = random.randint(0, 4095)
    seq_ctrl = (seq_num << 4) | 0
    
    # Beacon帧头
    frame = struct.pack('<HH', fc, duration)
    frame += da + sa + bssid_bytes
    frame += struct.pack('<H', seq_ctrl)
    
    # 固定字段 (12字节)
    timestamp = struct.pack('<Q', 0)  # 8字节时间戳
    beacon_interval = struct.pack('<H', 100)  # 100 TU
    capability = struct.pack('<H', 0x0431)  # ESS+Privacy+Short Preamble
    frame += timestamp + beacon_interval + capability
    
    # SSID Tag (Tag 0)
    ssid_bytes = ssid.encode('utf-8')
    frame += struct.pack('BB', 0, len(ssid_bytes)) + ssid_bytes
    
    # Supported Rates Tag (Tag 1)
    rates = bytes([0x82, 0x84, 0x8B, 0x96, 0x0C, 0x12, 0x18, 0x24])
    frame += struct.pack('BB', 1, len(rates)) + rates
    
    # Channel Tag (Tag 3)
    frame += struct.pack('BBB', 3, 1, channel)
    
    # RSN Tag (Tag 48) - WPA2
    rsn_data = bytes([
        0x01, 0x00,  # Version 1
        0x00, 0x0F, 0xAC, 0x04,  # Group Cipher: CCMP
        0x01, 0x00,  # Pairwise Cipher Count
        0x00, 0x0F, 0xAC, 0x04,  # Pairwise Cipher: CCMP
        0x01, 0x00,  # AKM Count
        0x00, 0x0F, 0xAC, 0x02,  # AKM: PSK
        0x00, 0x00,  # RSN Capabilities
    ])
    frame += struct.pack('BB', 48, len(rsn_data)) + rsn_data
    
    # FCS (简化)
    frame += struct.pack('<I', 0x12345678)
    
    return frame

def create_data_frame(src_mac, dst_mac, bssid, payload, to_ds=True, from_ds=False):
    """创建一个QoS Data帧"""
    # Frame Control
    subtype = 0x08  # QoS Data
    fc = (0 << 0) | (2 << 2) | (subtype << 4)
    if to_ds:
        fc |= (1 << 8)
    if from_ds:
        fc |= (1 << 9)
    
    duration = 0x006E
    
    # 地址字段根据To DS/From DS设置
    if to_ds and not from_ds:
        # STA -> AP: Addr1=BSSID, Addr2=SA, Addr3=DA
        addr1, addr2, addr3 = bssid, src_mac, dst_mac
    elif not to_ds and from_ds:
        # AP -> STA: Addr1=DA, Addr2=BSSID, Addr3=SA
        addr1, addr2, addr3 = dst_mac, bssid, src_mac
    else:
        addr1, addr2, addr3 = dst_mac, src_mac, bssid
    
    seq_num = random.randint(0, 4095)
    seq_ctrl = (seq_num << 4) | 0
    qos_ctrl = 0x0007  # TID=7 (Voice)
    
    frame = struct.pack('<HH', fc, duration)
    frame += addr1 + addr2 + addr3
    frame += struct.pack('<H', seq_ctrl)
    frame += struct.pack('<H', qos_ctrl)
    frame += payload.encode('utf-8')
    frame += struct.pack('<I', 0xDEADBEEF)  # FCS
    
    return frame

def analyze_frame(frame_bytes, label=""):
    """详细分析WiFi帧"""
    print(f"\n{'='*60}")
    print(f"帧分析: {label}")
    print(f"总长度: {len(frame_bytes)} 字节")
    print(f"{'='*60}")
    
    # Frame Control
    fc = struct.unpack('<H', frame_bytes[0:2])[0]
    protocol = fc & 0x03
    frame_type = (fc >> 2) & 0x03
    subtype = (fc >> 4) & 0x0F
    to_ds = (fc >> 8) & 0x01
    from_ds = (fc >> 9) & 0x01
    more_frag = (fc >> 10) & 0x01
    retry = (fc >> 11) & 0x01
    pwr_mgmt = (fc >> 12) & 0x01
    more_data = (fc >> 13) & 0x01
    protected = (fc >> 14) & 0x01
    order = (fc >> 15) & 0x01
    
    type_names = {0: 'Management', 1: 'Control', 2: 'Data', 3: 'Extension'}
    
    print(f"\n--- Frame Control (0x{fc:04X}) ---")
    print(f"  Protocol Version: {protocol}")
    print(f"  Type: {frame_type} ({type_names.get(frame_type, 'Unknown')})")
    print(f"  Subtype: {subtype}")
    print(f"  To DS: {to_ds}, From DS: {from_ds}")
    print(f"  More Frag: {more_frag}, Retry: {retry}")
    print(f"  Power Mgmt: {pwr_mgmt}, More Data: {more_data}")
    print(f"  Protected Frame: {protected}, Order: {order}")
    
    # Duration
    duration = struct.unpack('<H', frame_bytes[2:4])[0]
    print(f"\n--- Duration/ID: {duration} μs ---")
    
    # 地址字段
    addr1 = mac_to_str(frame_bytes[4:10])
    addr2 = mac_to_str(frame_bytes[10:16])
    addr3 = mac_to_str(frame_bytes[16:22])
    
    print(f"\n--- 地址字段 ---")
    if frame_type == 0:  # Management
        print(f"  DA: {addr1}")
        print(f"  SA: {addr2}")
        print(f"  BSSID: {addr3}")
    elif frame_type == 2:  # Data
        if to_ds == 0 and from_ds == 0:
            print(f"  DA: {addr1}")
            print(f"  SA: {addr2}")
            print(f"  BSSID: {addr3}")
        elif to_ds == 0 and from_ds == 1:
            print(f"  DA: {addr1}")
            print(f"  BSSID: {addr2}")
            print(f"  SA: {addr3}")
        elif to_ds == 1 and from_ds == 0:
            print(f"  BSSID: {addr1}")
            print(f"  SA: {addr2}")
            print(f"  DA: {addr3}")
    
    # 序列控制
    seq_ctrl = struct.unpack('<H', frame_bytes[22:24])[0]
    seq_num = seq_ctrl >> 4
    frag_num = seq_ctrl & 0x0F
    print(f"\n--- 序列控制 ---")
    print(f"  序列号: {seq_num}")
    print(f"  分段号: {frag_num}")
    
    # QoS Control (if applicable)
    if frame_type == 2 and (subtype & 0x08):
        qos_offset = 24
        qos = struct.unpack('<H', frame_bytes[qos_offset:qos_offset+2])[0]
        tid = qos & 0x0F
        ack_policy = (qos >> 5) & 0x03
        ac_names = {0: 'BE', 1: 'BK', 2: 'BK', 3: 'BE', 
                    4: 'VI', 5: 'VI', 6: 'VO', 7: 'VO'}
        print(f"\n--- QoS Control ---")
        print(f"  TID: {tid} (AC: {ac_names.get(tid, '?')})")
        print(f"  ACK Policy: {ack_policy}")


# 运行分析
print("=" * 60)
print("WiFi 帧捕获与分析工具")
print("=" * 60)

# 分析Beacon帧
ap_mac = bytes([0x00, 0x11, 0x22, 0x33, 0x44, 0x55])
beacon = create_beacon_frame("MyWiFi_Network", ap_mac, channel=11)
analyze_frame(beacon, "Beacon帧 (AP广播)")

# 分析QoS Data帧
sta_mac = bytes([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF])
dst_mac = bytes([0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB])
data = create_data_frame(sta_mac, dst_mac, ap_mac, "Hello WiFi!", to_ds=True, from_ds=False)
analyze_frame(data, "QoS Data帧 (STA→AP, To DS)")

print("\n\n=== 帧结构汇总 ===")
print(f"Beacon帧长度: {len(beacon)} 字节")
print(f"Data帧长度: {len(data)} 字节")
```

### 27.8.2 习题

**概念题**：

1. 解释隐藏终端问题和暴露终端问题的区别，以及RTS/CTS机制如何解决隐藏终端问题。

2. 比较CSMA/CD（以太网）和CSMA/CA（WiFi）的异同：
   - 为什么WiFi不能使用碰撞检测？
   - 为什么WiFi需要RTS/CTS而以太网不需要？

3. 画出WPA2四次握手的完整流程图，说明每条消息的作用以及各密钥的派生关系。

4. 比较移动IP中的三角路由和路由优化两种方案的优缺点。

**计算题**：

5. 一个WiFi接入点使用20MHz带宽，信噪比为25dB。
   - (a) 计算Shannon信道容量
   - (b) 如果使用64-QAM、3/4编码率，实际峰值速率是多少？
   - (c) 计算效率（实际速率/Shannon极限）

6. 一个LTE系统使用20MHz带宽，每个资源块（RB）包含12个子载波（子载波间隔15kHz）。
   - (a) 总共有多少个RB？
   - (b) 如果使用64-QAM、2×2 MIMO，峰值下行速率是多少？
   - (c) 与Shannon极限相比，效率如何？

**设计题**：

7. 设计一个简单的WiFi接入控制算法，要求：
   - 保证语音业务（AC_VO）的延迟<20ms
   - 视频业务（AC_VI）的吞吐不低于总带宽的40%
   - 最佳努力业务（AC_BE）公平分享剩余带宽
   - 使用优先级队列和加权公平调度

8. 设计一个移动IP的模拟实验：
   - 包含1个HA、3个FA、2个MN和1个CN
   - 模拟MN在FA之间移动的过程
   - 比较三角路由和路由优化的端到端延迟
   - 统计HA的处理负载

---

## 27.9 本章小结

本章涵盖了无线与移动网络的核心知识：

1. **无线信道**：理解大尺度/小尺度衰落、多径传播、隐藏/暴露终端问题是设计无线协议的基础
2. **802.11协议**：CSMA/CA、BEB退避、NAV虚拟载波侦听、RTS/CTS共同构成了WiFi的MAC层机制
3. **WiFi帧格式**：Frame Control字段的Type/Subtype/To DS/From DS决定了帧的类型和地址解释
4. **无线安全**：从WEP的致命缺陷到WPA3的SAE握手，安全协议的演进反映了对无线安全威胁认识的深化
5. **蜂窝网络**：从2G TDMA到5G NR的OFDMA，每一代都在频谱效率、速率和延迟上取得飞跃
6. **移动IP**：家乡代理/外地代理、注册、三角路由构成了IP层移动性的基础
7. **切换管理**：测量→决策→执行三阶段，X2/S1切换是LTE移动性的核心机制

<div data-component="ChapterSummary"></div>

---

## 扩展阅读

- IEEE 802.11-2020 Standard: https://standards.ieee.org/standard/802_11-2020.html
- 3GPP TS 36.300: E-UTRA and E-UTRAN Overall Description
- RFC 5944: IP Mobility Support for IPv4
- RFC 6275: Mobility Support in IPv6
- Kurose & Ross, "Computer Networking: A Top-Down Approach", Chapter 7
- Goldsmith, "Wireless Communications", Cambridge University Press

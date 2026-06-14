# Chapter 28: 多媒体网络与服务质量

> **学习目标**：
> - 掌握多媒体应用的三大分类（流式存储音视频、实时交互、批量传输）及其 QoS 需求差异
> - 理解音视频数字化全流程：采样定理（Nyquist）、量化、编码（PCM/AAC/H.264/H.265）与压缩原理
> - 深入掌握 HTTP 自适应流（DASH）的工作机制，包括 MPD 文件结构、ABR 算法与码率切换策略
> - 掌握 RTP/RTCP/RTSP 协议的头部格式、时间戳语义与交互流程
> - 理解 QoS 需求分析框架：延迟分解、抖动消除、丢包率与吞吐量保障
> - 掌握调度算法（FIFO、优先级队列、WFQ）与流量整形算法（漏桶、令牌桶）的数学原理与实现
> - 理解 IntServ 与 DiffServ 两种服务模型的设计思想、协议机制与适用场景

---

## 28.1 多媒体应用概述

### 28.1.1 多媒体应用分类

互联网上的多媒体应用可按照**交互性**和**实时性**分为三大类：

| 类型 | 典型应用 | 延迟要求 | 丢包容忍 | 带宽需求 |
|------|---------|---------|---------|---------|
| 流式存储音视频 | Netflix、YouTube、Spotify | 秒级（可缓冲） | 中等 | 高（自适应） |
| 实时交互 | VoIP、视频会议、在线游戏 | <150ms | 低 | 中等 |
| 批量传输 | 视频上传、文件下载 | 无实时要求 | 零容忍 | 尽力而为 |

**流式存储音视频（Streaming Stored Audio/Video）** 是当前互联网流量的主体。这类应用的特点是内容预先存储在服务器上，客户端可以通过缓冲来吸收网络抖动。关键挑战在于如何在不同网络条件下提供流畅的播放体验。

**实时交互应用（Real-time Interactive）** 包括 VoIP（如 Skype、微信语音）和视频会议（如 Zoom、腾讯会议）。这类应用对端到端延迟极为敏感——ITU-T G.114 建议单向延迟不超过 150ms。同时对抖动（Jitter）也有严格要求，通常需要抖动缓冲区来平滑播放。

**批量传输（Bulk Transfer）** 如视频上传到云端、大文件下载等，对延迟不敏感，但要求数据完整可靠传输，通常使用 TCP 协议。

### 28.1.2 多媒体流量特征

多媒体流量具有以下显著特征：

1. **高带宽**：标清视频约 1-2 Mbps，高清视频 5-10 Mbps，4K 视频可达 25-50 Mbps
2. **突发性**：视频帧大小不一（I 帧远大于 P/B 帧），导致流量突发
3. **周期性**：视频帧以固定帧率（如 24/30/60 fps）产生，具有周期性特征
4. **可伸缩性**：现代编码器支持多种码率，可根据网络状况调整

<div data-component="MultimediaTrafficAnalyzer"></div>

---

## 28.2 音视频数字化

### 28.2.1 采样定理（Nyquist-Shannon）

模拟信号数字化的第一步是**采样（Sampling）**。Nyquist-Shannon 采样定理指出：

> 如果连续信号 $x(t)$ 的最高频率分量为 $f_{max}$，则以采样率 $f_s > 2f_{max}$ 进行采样，可以从采样值无失真地恢复原始信号。

$$f_s \geq 2 \cdot f_{max}$$

其中 $2f_{max}$ 称为 **Nyquist 率**。

**实例**：
- 电话语音：带宽限制在 3400Hz，采样率 8000 Hz（每 125μs 采样一次）
- CD 音质：带宽 20kHz，采样率 44100 Hz
- 高保真音频：采样率 96kHz 或 192kHz

**混叠（Aliasing）**：如果采样率低于 Nyquist 率，高频分量会被"折叠"到低频段，产生无法消除的失真。因此在采样前通常需要一个**抗混叠滤波器（Anti-aliasing Filter）**。

### 28.2.2 量化与编码

采样后的连续幅值需要**量化（Quantization）** 为有限个离散值。

**均匀量化**：将信号幅值范围均匀划分为 $2^n$ 个区间，用 $n$ 位二进制数表示：

$$\text{量化步长} \Delta = \frac{V_{max} - V_{min}}{2^n}$$

**量化噪声**：量化过程引入的误差称为量化噪声。对于均匀量化，信噪比（SNR）为：

$$SNR = 6.02n + 1.76 \text{ dB}$$

每增加 1 位量化精度，SNR 提高约 6 dB。

**脉冲编码调制（PCM）** 是最基本的数字化方法：

```
模拟信号 → 采样 → 量化 → 编码 → 数字比特流
```

PCM 数据率计算：

$$R = f_s \times n \times c$$

其中 $f_s$ 为采样率，$n$ 为量化位数，$c$ 为声道数。

| 标准 | 采样率 | 量化位数 | 声道 | 数据率 |
|------|--------|---------|------|--------|
| 电话（G.711） | 8 kHz | 8 bit | 1 | 64 kbps |
| CD | 44.1 kHz | 16 bit | 2 | 1411.2 kbps |
| 高保真 | 96 kHz | 24 bit | 2 | 4608 kbps |

### 28.2.3 音频编码标准

**AAC（Advanced Audio Coding）**：
- MPEG-2/MPEG-4 标准的音频编码格式
- 使用 MDCT（改进离散余弦变换）进行频域编码
- 支持 8-96 kHz 采样率，1-48 声道
- 典型码率：128-256 kbps（立体声），音质接近 CD

**Opus**：
- IETF RFC 6716 标准，融合 SILK（语音）和 CELT（音乐）编码
- 带宽范围 6-510 kbps，延迟可低至 5ms
- 广泛用于 WebRTC、Discord 等实时通信场景

### 28.2.4 视频编码原理

视频编码的核心思想是利用**空间冗余**和**时间冗余**来压缩数据。

**帧内预测（Intra-frame Prediction）**：
利用同一帧内相邻像素的相关性。当前块的像素值可以用其上方、左方已编码块的像素来预测，只编码预测残差。

**帧间预测（Inter-frame Prediction）**：
利用相邻帧之间的时间相关性。通过**运动估计（Motion Estimation）** 找到当前帧中每个块在参考帧中的最佳匹配位置，只编码**运动矢量（Motion Vector）** 和预测残差。

**DCT 变换（Discrete Cosine Transform）**：
将空间域的残差块变换到频域，使能量集中在低频系数上。8×8 DCT 变换：

$$F(u,v) = \frac{1}{4} C(u)C(v) \sum_{x=0}^{7}\sum_{y=0}^{7} f(x,y) \cos\frac{(2x+1)u\pi}{16} \cos\frac{(2y+1)v\pi}{16}$$

其中 $C(0) = 1/\sqrt{2}$，$C(k) = 1$（$k>0$）。

**量化与熵编码**：DCT 系数经过量化（有损压缩的核心步骤）后，使用 CABAC 或 CAVLC 进行熵编码。

### 28.2.5 H.264/AVC 编码标准

H.264 是目前应用最广泛的视频编码标准，引入了多项关键技术：

1. **多参考帧运动补偿**：可使用多个已编码帧作为参考，提高预测精度
2. **可变块大小运动补偿**：宏块可分割为 16×16 到 4×4 的多种大小
3. **环路去块滤波器**：在编码环路中加入去块效应滤波器
4. **熵编码**：支持 CAVLC 和 CABAC 两种熵编码方式

H.264 的编码效率比 MPEG-2 提高约 50%，即在相同画质下码率减半。

### 28.2.6 H.265/HEVC 编码标准

H.265（High Efficiency Video Coding）是 H.264 的后继者：

- **编码树单元（CTU）**：最大 64×64，支持递归四叉树分割
- **更精细的运动补偿**：支持 35 种角度的帧内预测模式
- **改进的环路滤波**：新增 SAO（Sample Adaptive Offset）滤波
- 编码效率比 H.264 再提高约 50%

**帧类型与编码结构**：

```
I帧（关键帧）  ：完全自包含，可独立解码
P帧（预测帧）  ：参考前面的 I/P 帧进行帧间预测
B帧（双向帧）  ：参考前后两帧进行双向预测

典型 GOP 结构：I B B P B B P B B P B B I ...
```

<div data-component="VideoCodecComparison"></div>

---

## 28.3 流式存储视频

### 28.3.1 客户端缓冲策略

流式视频播放的关键在于客户端缓冲区管理。基本流程：

```
服务器 → [网络] → 客户端缓冲区 → 解码器 → 播放器
```

**缓冲区的作用**：
1. **吸收网络抖动**：网络传输延迟不均匀，缓冲区平滑这些波动
2. **应对突发丢包**：缓冲数据可用于填补丢包导致的间隙
3. **支持码率切换**：缓冲区充盈度是 ABR 算法的关键输入

**缓冲区水位管理**：
- **低水位（Low Watermark）**：低于此阈值时暂停播放或降低码率
- **高水位（High Watermark）**：高于此阈值时停止下载或切换高码率
- **目标水位**：缓冲区维持在目标值附近

### 28.3.2 HTTP 自适应流（DASH）

**DASH（Dynamic Adaptive Streaming over HTTP）** 是目前主流的流媒体传输技术，标准化为 ISO/IEC 23009-1。

**核心思想**：将视频内容预处理为多种码率的分片（Segment），客户端根据网络状况自适应选择码率。

**系统架构**：

```
原始视频 → 编码器（多码率） → 切片器 → HTTP 服务器/CDN
                                         ↓
                                    MPD 文件（清单）
                                         ↓
                                    DASH 客户端
```

**MPD（Media Presentation Description）文件**：

MPD 是一个 XML 格式的描述文件，包含视频的所有元信息：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<MPD xmlns="urn:mpeg:dash:schema:mpd:2011"
     type="dynamic"
     minimumUpdatePeriod="PT10S"
     minBufferTime="PT2S"
     availabilityStartTime="2024-01-01T00:00:00Z">
  <Period id="1" start="PT0S">
    <AdaptationSet mimeType="video/mp4" contentType="video">
      <Representation id="1" bandwidth="500000" width="640" height="360">
        <BaseURL>video_low/</BaseURL>
        <SegmentTemplate timescale="90000"
                         initialization="init.mp4"
                         media="seg_$Number$.m4s"
                         startNumber="1">
          <SegmentTimeline>
            <S t="0" d="90000" r="29"/>
          </SegmentTimeline>
        </SegmentTemplate>
      </Representation>
      <Representation id="2" bandwidth="2000000" width="1280" height="720">
        <BaseURL>video_hd/</BaseURL>
        <!-- ... -->
      </Representation>
      <Representation id="3" bandwidth="5000000" width="1920" height="1080">
        <BaseURL>video_fhd/</BaseURL>
        <!-- ... -->
      </Representation>
    </AdaptationSet>
    <AdaptationSet mimeType="audio/mp4" contentType="audio">
      <Representation id="a1" bandwidth="128000">
        <BaseURL>audio/</BaseURL>
        <!-- ... -->
      </Representation>
    </AdaptationSet>
  </Period>
</MPD>
```

### 28.3.3 ABR 算法（Adaptive Bitrate Algorithm）

ABR 算法是 DASH 的核心，决定客户端在每个分片时刻选择哪种码率。主要策略：

**基于吞吐量（Throughput-based）**：

$$r_{next} = \max\{r_i : r_i \leq \alpha \cdot T_{est}\}$$

其中 $T_{est}$ 是估计的可用带宽，$\alpha$ 是安全系数（通常 0.7-0.9）。

**基于缓冲区（Buffer-based）**：

$$r_{next} = f(B_{current})$$

当缓冲区水位高时选择高码率，水位低时选择低码率。典型实现如 BBA（Buffer-Based Approach）。

**混合策略（Hybrid）**：结合吞吐量和缓冲区信息，如 BOLA、MPC（Model Predictive Control）等。

**码率切换的挑战**：
- **振荡（Oscillation）**：频繁在高低码率间切换，影响用户体验
- **公平性**：多个客户端竞争带宽时的公平分配
- **启动延迟**：初始码率选择影响起播时间

<div data-component="DASHPlayerSimulator"></div>

### 28.3.4 CDN（Content Delivery Network）

CDN 是流式视频分发的基础设施，通过在全球部署边缘服务器来降低延迟、提高可用性。

**CDN 工作原理**：

1. **内容分发**：源站内容被推送到全球各地的边缘服务器（Edge Server）
2. **用户请求重定向**：通过 DNS 重定向或 HTTP 重定向将用户引导到最近的边缘服务器
3. **就近服务**：边缘服务器直接响应用户请求

**DNS 重定向流程**：

```
用户 → 本地DNS → 权威DNS（返回CDN域名）
                → CDN权威DNS（根据用户IP选择最优边缘服务器）
                → 返回边缘服务器IP
用户 → 边缘服务器（获取内容）
```

**任播（Anycast）**：多个边缘服务器共享同一个 IP 地址，BGP 路由自动将用户请求导向最近的服务器。这种方式无需修改 DNS，切换速度快。

**CDN 缓存策略**：
- **热度驱动**：热门内容缓存在更多边缘节点
- **地理感知**：根据区域流行度决定缓存位置
- **分层缓存**：边缘 → 区域 → 核心，逐层回源

---

## 28.4 RTP 协议

### 28.4.1 RTP 概述

**RTP（Real-time Transport Protocol）** 定义在 RFC 3550 中，是为实时多媒体数据传输设计的传输层协议。RTP 通常运行在 UDP 之上，提供时间戳、序列号等机制，但不保证可靠性或顺序交付——这些功能由应用层处理。

**RTP 的设计哲学**：将复杂性推到端系统。网络核心保持简单（best-effort），端系统负责处理丢包、乱序、抖动等问题。

### 28.4.2 RTP 头部格式

RTP 头部格式如下（最小 12 字节）：

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|V=2|P|X|  CC   |M|     PT      |       sequence number         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                           timestamp                           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                             SSRC                              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                             CSRC                              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

**字段详细解析**：

| 字段 | 位数 | 含义 |
|------|------|------|
| V（Version） | 2 bit | RTP 版本号，当前为 2 |
| P（Padding） | 1 bit | 填充标志。为 1 时，包尾有填充字节，最后一个字节指示填充长度 |
| X（Extension） | 1 bit | 扩展标志。为 1 时，固定头部后紧跟一个扩展头部 |
| CC（CSRC Count） | 4 bit | CSRC 标识符的数量（0-15） |
| M（Marker） | 1 bit | 标记位。对于视频，标记一帧的最后一个包；对于音频，标记会话开始后的第一个包 |
| PT（Payload Type） | 7 bit | 载荷类型。标识 RTP 载荷的编码格式 |

**常见 Payload Type**：

| PT 值 | 编码格式 | 采样率 | 说明 |
|--------|---------|--------|------|
| 0 | PCMU (G.711 μ-law) | 8 kHz | 电话语音 |
| 8 | PCMA (G.711 A-law) | 8 kHz | 电话语音 |
| 9 | G.722 | 8 kHz | 宽带语音 |
| 18 | G.729 | 8 kHz | 低码率语音 |
| 26 | JPEG | - | 视频 |
| 31 | H.261 | 90 kHz | 视频 |
| 32 | MPEG-1/2 | 90 kHz | 视频 |
| 96-127 | 动态 | - | 动态分配 |

**序列号（Sequence Number）**：16 位，每个 RTP 包递增 1。用于接收端检测丢包和重新排序。初始值随机选取，增加到 65535 后回绕到 0。

**时间戳（Timestamp）**：32 位，表示 RTP 载荷中第一个采样的生成时刻。**注意：时间戳的单位取决于载荷类型**：
- 音频：通常使用 8 kHz 时钟（每 125μs 递增 1），即每个采样递增 1
- 视频：通常使用 90 kHz 时钟，每个视频帧的采样时刻递增帧间隔对应的计数

**SSRC（Synchronization Source）**：32 位，唯一标识一个 RTP 源。在同一 RTP 会话中，每个发送者必须选择一个随机的、全局唯一的 SSRC。

**CSRC（Contributing Source）**：每个 32 位，数量由 CC 字段指示。用于标识经过混音器（Mixer）处理后对当前包有贡献的源。

### 28.4.3 时间戳语义详解

**音频时间戳**：

假设音频编码器每 20ms 产生一个帧，采样率 8000 Hz，则每帧包含 160 个采样：

```
帧 0: timestamp = 0
帧 1: timestamp = 160
帧 2: timestamp = 320
帧 3: timestamp = 480
...
```

**视频时间戳**：

假设视频帧率为 30fps，时钟 90kHz，则帧间隔为 3000 个时钟单位：

```
帧 0: timestamp = 0
帧 1: timestamp = 3000
帧 2: timestamp = 6000
帧 3: timestamp = 9000
...
```

当一帧视频过大被拆分为多个 RTP 包时，这些包共享相同的时间戳：

```
帧 3 的包 0: timestamp = 9000, seq = 100
帧 3 的包 1: timestamp = 9000, seq = 101
帧 3 的包 2: timestamp = 9000, seq = 102, M = 1（标记帧结束）
```

### 28.4.4 RTP 头扩展

当 X 位为 1 时，RTP 固定头部后紧跟一个扩展头部：

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|      defined by profile       |           length              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        header extension                       |
|                             ...                               |
```

常见的 RTP 扩展包括：
- **传输时间戳（Transport-wide Sequence Number）**：WebRTC 用于带宽估计
- **绝对发送时间（Absolute Send Time）**：精确的发送时刻
- **音量级别（Audio Level Indication）**：音频活动检测

<div data-component="RTPPacketParser"></div>

### 28.4.5 RTP 打包与重组

**打包逻辑**：将编码帧拆分为 RTP 包

```python
import struct
import random

class RTPPacket:
    """RTP 数据包解析与构造"""
    
    # Payload Type 常量
    PT_PCMU = 0
    PT_PCMA = 8
    PT_DYNAMIC = 96
    
    def __init__(self, payload_type=0, sequence_number=None,
                 timestamp=0, ssrc=None, marker=False, payload=b''):
        self.version = 2
        self.padding = False
        self.extension = False
        self.cc = 0
        self.marker = marker
        self.payload_type = payload_type
        self.sequence_number = sequence_number if sequence_number is not None \
                               else random.randint(0, 65535)
        self.timestamp = timestamp
        self.ssrc = ssrc if ssrc is not None else random.randint(0, 0xFFFFFFFF)
        self.payload = payload
    
    def serialize(self):
        """序列化为字节流"""
        # 第一个字节: V(2) | P(1) | X(1) | CC(4)
        byte0 = (self.version << 6) | \
                (int(self.padding) << 5) | \
                (int(self.extension) << 4) | \
                self.cc
        # 第二个字节: M(1) | PT(7)
        byte1 = (int(self.marker) << 7) | self.payload_type
        header = struct.pack('!BBHII',
                            byte0, byte1,
                            self.sequence_number,
                            self.timestamp,
                            self.ssrc)
        return header + self.payload
    
    @classmethod
    def parse(cls, data):
        """从字节流解析 RTP 包"""
        if len(data) < 12:
            raise ValueError("RTP packet too short")
        
        byte0, byte1 = struct.unpack('!BB', data[0:2])
        version = (byte0 >> 6) & 0x03
        padding = bool((byte0 >> 5) & 0x01)
        extension = bool((byte0 >> 4) & 0x01)
        cc = byte0 & 0x0F
        
        marker = bool((byte1 >> 7) & 0x01)
        payload_type = byte1 & 0x7F
        
        seq_num, timestamp, ssrc = struct.unpack('!HII', data[2:12])
        
        # 跳过 CSRC
        offset = 12 + cc * 4
        
        # 处理扩展头部
        if extension and len(data) > offset + 4:
            ext_profile, ext_length = struct.unpack('!HH', data[offset:offset+4])
            offset += 4 + ext_length * 4
        
        payload = data[offset:]
        
        pkt = cls(payload_type, seq_num, timestamp, ssrc, marker, payload)
        pkt.version = version
        pkt.padding = padding
        pkt.extension = extension
        pkt.cc = cc
        return pkt
    
    def __repr__(self):
        return (f"RTP(v={self.version}, PT={self.payload_type}, "
                f"seq={self.sequence_number}, ts={self.timestamp}, "
                f"ssrc=0x{self.ssrc:08X}, marker={self.marker}, "
                f"payload_len={len(self.payload)})")


class RTPPacker:
    """将编码帧打包为 RTP 包"""
    
    def __init__(self, ssrc, payload_type=RTPPacket.PT_DYNAMIC,
                 clock_rate=90000, max_payload_size=1400):
        self.ssrc = ssrc
        self.payload_type = payload_type
        self.clock_rate = clock_rate
        self.max_payload_size = max_payload_size
        self.seq_num = random.randint(0, 65535)
        self.base_ts = 0
    
    def pack_frame(self, frame_data, frame_duration_ms=None, is_last_in_frame=True):
        """将一个编码帧打包为一个或多个 RTP 包
        
        Args:
            frame_data: 编码帧数据
            frame_duration_ms: 帧持续时间（毫秒），用于计算时间戳增量
            is_last_in_frame: 是否是当前帧的最后一个 RTP 包
        
        Returns:
            list of RTPPacket
        """
        packets = []
        
        # 计算时间戳增量
        if frame_duration_ms is not None:
            ts_increment = int(self.clock_rate * frame_duration_ms / 1000)
        else:
            ts_increment = 0
        
        # 分片打包
        offset = 0
        while offset < len(frame_data):
            chunk_size = min(self.max_payload_size, len(frame_data) - offset)
            is_last_chunk = (offset + chunk_size >= len(frame_data))
            
            marker = is_last_chunk and is_last_in_frame
            
            pkt = RTPPacket(
                payload_type=self.payload_type,
                sequence_number=self.seq_num,
                timestamp=self.base_ts,
                ssrc=self.ssrc,
                marker=marker,
                payload=frame_data[offset:offset + chunk_size]
            )
            packets.append(pkt)
            
            self.seq_num = (self.seq_num + 1) % 65536
            offset += chunk_size
        
        self.base_ts = (self.base_ts + ts_increment) % (2**32)
        return packets
```

**重组逻辑**：去重、排序、时间戳恢复

```python
from collections import defaultdict
import heapq
import time

class RTPDepacketizer:
    """RTP 包重组器：去重、排序、抖动缓冲"""
    
    def __init__(self, jitter_buffer_ms=100):
        self.jitter_buffer_ms = jitter_buffer_ms
        self.buffers = defaultdict(list)  # 按时间戳分组
        self.seen_seq = set()             # 去重
        self.last_playout_ts = None
        self.jitter_estimate = 0
        self.last_transit = 0
    
    def add_packet(self, pkt):
        """添加一个 RTP 包到重组缓冲区"""
        # 去重
        if pkt.sequence_number in self.seen_seq:
            return
        self.seen_seq.add(pkt.sequence_number)
        
        # 计算抖动（RFC 3550 A.8）
        arrival_time = time.time() * 1000  # ms
        self._update_jitter(pkt.timestamp, arrival_time)
        
        # 按时间戳分组存入缓冲区
        self.buffers[pkt.timestamp].append(pkt)
    
    def _update_jitter(self, rtp_ts, arrival_ms):
        """更新抖动估计"""
        # 简化的抖动计算
        transit = arrival_ms - rtp_ts
        if self.last_transit != 0:
            d = abs(transit - self.last_transit)
            self.jitter_estimate += (d - self.jitter_estimate) / 16.0
        self.last_transit = transit
    
    def get_complete_frames(self):
        """获取完整的帧（marker=1 的包已到达的帧）"""
        complete = []
        for ts in sorted(self.buffers.keys()):
            pkts = self.buffers[ts]
            # 检查是否有 marker 包（帧的最后一个包）
            has_marker = any(p.marker for p in pkts)
            if has_marker:
                # 按序列号排序并拼接
                pkts.sort(key=lambda p: p.sequence_number)
                frame_data = b''.join(p.payload for p in pkts)
                complete.append((ts, frame_data))
                del self.buffers[ts]
        return complete
    
    def get_jitter_ms(self):
        """返回当前抖动估计（毫秒）"""
        return self.jitter_estimate
```

---

## 28.5 RTCP 协议

### 28.5.1 RTCP 概述

**RTCP（RTP Control Protocol）** 与 RTP 配合使用，提供会话控制和质量反馈。RTCP 不传输媒体数据，而是周期性地交换控制信息。

**RTCP 的五大功能**：
1. 服务质量反馈（丢包率、抖动、延迟）
2. 会话参与者标识（CNAME）
3. 会议规模估计
4. 最小控制信息
5. 独立于 RTP 的传输

### 28.5.2 RTCP 包类型

**SR（Sender Report，发送方报告）**：

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|V=2|P|  RC   |   PT=SR=200   |             length              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         SSRC of sender                        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|              NTP timestamp, most significant word              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|              NTP timestamp, least significant word             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         RTP timestamp                          |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     sender's packet count                     |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                      sender's octet count                     |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

**SR 关键字段**：
- **NTP 时间戳**：64 位，发送 SR 时的 NTP 绝对时间。用于计算 RTT
- **RTP 时间戳**：与 NTP 时间戳对应的 RTP 时间戳，用于音视频同步
- **发送包数/字节数**：从会话开始到发送 SR 时的累计值

**RR（Receiver Report，接收方报告）**：

RR 包含多个报告块（Report Block），每个块对应一个发送源：

```
报告块字段：
- SSRC_n: 被报告的源标识
- 丢包率(Fraction Lost): 上次报告以来的丢包比例（8位）
- 累计丢包数(Cumulative Lost): 24位，有符号
- 最高序列号扩展: 32位（循环计数 + 最高序列号）
- 抖动(Jitter): 32位，时间戳单位的抖动估计
- LSR: 最后收到的 SR 的 NTP 时间戳中间 32 位
- DLSR: 从收到 LSR 到发送 RR 的延迟（1/65536 秒）
```

**RTT 计算**：利用 SR 和 RR 中的时间戳可以计算往返时间：

```
发送方发送 SR (包含 NTP 时间戳 T1)
          ↓
接收方收到 SR，记录接收时间 T2
接收方发送 RR (LSR = T1 的中间32位, DLSR = T2 - T2')
          ↓
发送方收到 RR 于时间 T3
RTT = T3 - T1 - DLSR
```

**SDES（Source Description）**：
包含 CNAME（规范名）、NAME、EMAIL 等标识信息。CNAME 是必需的，格式为 `user@host`。

**BYE**：参与者离开会话时发送。

**APP**：应用自定义的 RTCP 包。

### 28.5.3 RTCP 带宽控制

RTCP 的总带宽应控制在会话总带宽的 5% 以内：

$$B_{RTCP} = 0.05 \times B_{total}$$

当参与者数量为 $N$ 时，每个参与者每 5 秒至少发送一个 RTCP 包。但随着 $N$ 增大，发送间隔需要增长以避免 RTCP 洪泛：

$$T_{min} = \frac{N}{B_{RTCP}} \times S_{avg}$$

其中 $S_{avg}$ 是平均 RTCP 包大小。

---

## 28.6 RTSP 协议

### 28.6.1 RTSP 概述

**RTSP（Real Time Streaming Protocol）** 定义在 RFC 7826 中，是流媒体会话的控制协议。RTSP 的作用类似于"遥控器"，控制媒体流的播放、暂停、停止等操作。

**RTSP 与 RTP 的关系**：
- **RTSP**：控制面协议，负责建立和控制会话
- **RTP**：数据面协议，负责传输媒体数据
- **RTCP**：监控面协议，负责质量反馈

```
客户端 ←——RTSP——→ 服务器（控制信令）
客户端 ←——RTP———→ 服务器（媒体数据）
客户端 ←——RTCP——→ 服务器（质量反馈）
```

### 28.6.2 RTSP 方法

| 方法 | 方向 | 功能 |
|------|------|------|
| DESCRIBE | C→S | 获取媒体描述信息（SDP 格式） |
| SETUP | C→S | 建立传输会话，协商传输参数 |
| PLAY | C→S | 开始或恢复媒体播放 |
| PAUSE | C→S | 暂停媒体播放 |
| TEARDOWN | C→S | 结束会话，释放资源 |
| GET_PARAMETER | C↔S | 查询参数 |
| SET_PARAMETER | C→S | 设置参数 |
| ANNOUNCE | C→S | 更新媒体描述 |
| REDIRECT | S→C | 重定向到新服务器 |

### 28.6.3 RTSP 会话流程

一个典型的 RTSP 会话流程：

```
C → S: DESCRIBE rtsp://example.com/movie RTSP/1.0
        CSeq: 1

S → C: RTSP/1.0 200 OK
        CSeq: 1
        Content-Type: application/sdp
        Content-Length: ...
        
        v=0
        o=- 2890844526 2890844526 IN IP4 192.168.1.100
        s=Movie
        m=video 0 RTP/AVP 96
        a=rtpmap:96 H264/90000
        m=audio 0 RTP/AVP 97
        a=rtpmap:97 PCMU/8000

C → S: SETUP rtsp://example.com/movie/video RTSP/1.0
        CSeq: 2
        Transport: RTP/AVP;unicast;client_port=3000-3001

S → C: RTSP/1.0 200 OK
        CSeq: 2
        Transport: RTP/AVP;unicast;client_port=3000-3001;
                   server_port=5000-5001
        Session: 12345678

C → S: PLAY rtsp://example.com/movie RTSP/1.0
        CSeq: 3
        Session: 12345678
        Range: npt=0.000-

S → C: RTSP/1.0 200 OK
        CSeq: 3
        Session: 12345678
        Range: npt=0.000-
        RTP-Info: url=rtsp://example.com/movie/video;
                  seq=1;rtptime=0

[媒体流通过 RTP 传输...]

C → S: TEARDOWN rtsp://example.com/movie RTSP/1.0
        CSeq: 4
        Session: 12345678

S → C: RTSP/1.0 200 OK
        CSeq: 4
```

### 28.6.4 RTSP 状态机

RTSP 会话有以下状态：

```
                    ┌──────────┐
        SETUP       │          │  TEARDOWN
    ┌──────────────→│  Init    │──────────────→
    │               │          │
    │               └──────────┘
    │                    │
    │                    │ SETUP (成功)
    │                    ↓
    │               ┌──────────┐
    │    PLAY       │          │  TEARDOWN
    ├──────────────→│  Ready   │──────────────→
    │               │          │
    │               └──────────┘
    │                    │
    │                    │ PLAY (成功)
    │                    ↓
    │               ┌──────────┐
    │    PAUSE      │          │  TEARDOWN
    ├──────────────→│ Playing  │──────────────→
    │               │          │
    │               └──────────┘
    │                    │
    │                    │ PAUSE (成功)
    │                    ↓
    │               ┌──────────┐
    └───────────────│  Ready   │
        PLAY        │          │
                    └──────────┘
```

---

## 28.7 QoS 需求分析

### 28.7.1 端到端延迟分解

多媒体应用的端到端延迟由多个环节组成：

$$D_{total} = D_{encoding} + D_{packetization} + D_{network} + D_{playout}$$

**各环节延迟分析**：

| 延迟环节 | 典型值 | 说明 |
|---------|--------|------|
| 编码延迟 $D_{enc}$ | 10-100 ms | 视频编码器需要缓存多帧才能编码 |
| 打包延迟 $D_{pkt}$ | 20-40 ms | 音频帧积累到一个 RTP 包的时间 |
| 网络传输延迟 $D_{net}$ | 10-200 ms | 包含传播延迟 + 排队延迟 + 处理延迟 |
| 抖动缓冲延迟 $D_{jbuf}$ | 50-200 ms | 消除网络抖动的缓冲延迟 |
| 解码延迟 $D_{dec}$ | 5-50 ms | 解码器处理一帧的时间 |
| 渲染延迟 $D_{render}$ | 0-16 ms | 等待下一个渲染时刻 |

**ITU-T G.114 建议**：
- 单向延迟 < 150ms：用户感知良好
- 单向延迟 150-400ms：可接受但有影响
- 单向延迟 > 400ms：不可接受

### 28.7.2 抖动（Jitter）

**抖动定义**：相邻数据包到达时间间隔的变化量。

$$J_i = |(a_i - a_{i-1}) - (s_i - s_{i-1})|$$

其中 $a_i$ 是第 $i$ 个包的到达时间，$s_i$ 是发送时间。

**RFC 3550 中的抖动估计**（指数移动平均）：

$$D_i = |(a_i - s_i) - (a_{i-1} - s_{i-1})| = |(a_i - a_{i-1}) - (s_i - s_{i-1})|$$

$$J_i = J_{i-1} + \frac{1}{16}(|D_i| - J_{i-1})$$

**抖动缓冲区**：接收端设置一个缓冲区，将到达的包暂存一段时间后再播放。缓冲区大小 $B$ 与抖动的关系：

$$P_{playout\ loss} = P(\text{packet delay} > D_{fixed} + B)$$

缓冲区越大，丢包率越低，但延迟越大——这是一个**延迟-丢包权衡**。

### 28.7.3 丢包率

实时多媒体应用的丢包率要求：

| 应用类型 | 可接受丢包率 | 丢包处理策略 |
|---------|-------------|------------|
| VoIP | < 3% | 插入静音/前一帧 |
| 视频会议 | < 1% | FEC/重传/隐藏 |
| 流式视频 | < 0.1% | 重传/FEC |

**丢包恢复技术**：
- **FEC（前向纠错）**：发送冗余数据，接收端用于恢复丢失的包
- **交织（Interleaving）**：将连续的采样分散到不同的包中
- **重传（Retransmission）**：对实时性要求不高的场景
- **错误隐藏（Error Concealment）**：用相邻帧/频段数据估算丢失内容

### 28.7.4 吞吐量

多媒体应用需要足够的吞吐量来传输编码数据。**Little's Law** 在多媒体系统中的应用：

$$L = \lambda \times W$$

其中 $L$ 是系统中的平均数据量，$\lambda$ 是到达率，$W$ 是平均逗留时间。

**应用实例**：如果视频码率为 5 Mbps，用户希望缓冲 10 秒的内容，则缓冲区需要：

$$L = 5 \times 10 = 50 \text{ Mbit} = 6.25 \text{ MB}$$

---

## 28.8 调度算法

### 28.8.1 FIFO 队列

**FIFO（First-In First-Out）** 是最简单的调度算法，按到达顺序服务。

**M/M/1 排队模型**：到达服从泊松分布（率 $\lambda$），服务时间服从指数分布（率 $\mu$）。

**Little's Law** 在 M/M/1 队列中的应用：

$$L = \frac{\rho}{1 - \rho}$$

$$W = \frac{1}{\mu - \lambda}$$

其中 $\rho = \lambda / \mu$ 是利用率。

**平均队列长度**：

$$L_q = \frac{\rho^2}{1 - \rho}$$

**平均等待时间**：

$$W_q = \frac{\rho}{\mu(1-\rho)}$$

FIFO 的问题：无法区分不同优先级的流量。一个发送大量数据的流会占用队列空间，影响其他流。

### 28.8.2 优先级队列

优先级队列将流量分为不同优先级，高优先级包先于低优先级包发送。

**实现方式**：维护多个逻辑队列（通常 2-8 个），调度器始终从最高非空优先级队列取包。

**严格优先级调度**的问题：
- **饥饿（Starvation）**：高优先级流量持续到达时，低优先级流量永远得不到服务
- **带宽不公平**：无法保证最低带宽

### 28.8.3 WFQ（加权公平队列）

**WFQ（Weighted Fair Queuing）** 是一种基于虚拟时钟的调度算法，能为每个流提供公平的带宽份额。

**核心思想**：为每个流维护一个"虚拟时钟"，虚拟完成时间最小的包优先发送。

**虚拟时钟计算**：

定义虚拟时间函数 $V(t)$，满足：

$$\frac{dV(t)}{dt} = \frac{1}{\sum_{i \in B(t)} \phi_i}$$

其中 $B(t)$ 是时间 $t$ 时有数据等待发送的流的集合，$\phi_i$ 是流 $i$ 的权重。

对于流 $i$ 的第 $k$ 个包，虚拟开始时间 $S_i^k$ 和虚拟完成时间 $F_i^k$：

$$S_i^k = \max(F_i^{k-1}, V(a_i^k))$$

$$F_i^k = S_i^k + \frac{L_i^k}{\phi_i}$$

其中 $a_i^k$ 是包到达时间，$L_i^k$ 是包长度，$\phi_i$ 是流 $i$ 的权重。

**WFQ 的带宽保证**：

流 $i$ 在 $[t_1, t_2]$ 时间内获得的服务量满足：

$$W_i(t_1, t_2) \geq \frac{\phi_i}{\sum_{j} \phi_j} \cdot C \cdot (t_2 - t_1) - \frac{L_{max}}{\sum_{j} \phi_j} \cdot \phi_i$$

其中 $C$ 是链路容量，$L_{max}$ 是最大包长。

**WFQ 的公平性**：在最坏情况下，WFQ 与理想公平队列（GPS，Generalized Processor Sharing）的偏差不超过一个最大包长的传输时间。

<div data-component="WFQSchedulerSimulator"></div>

**WFQ 调度器 Python 实现**：

```python
import heapq
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass(order=True)
class Packet:
    """数据包"""
    virtual_finish_time: float = field(compare=True)
    flow_id: int = field(compare=False)
    size: int = field(compare=False)
    arrival_time: float = field(compare=False)
    seq: int = field(compare=False)

class WFQScheduler:
    """加权公平队列调度器"""
    
    def __init__(self, link_capacity: float):
        """
        Args:
            link_capacity: 链路容量 (bits/sec)
        """
        self.link_capacity = link_capacity
        self.flows = {}           # flow_id -> weight
        self.queue = []           # 优先级队列 (虚拟完成时间)
        self.virtual_time = 0.0   # 虚拟时钟
        self.last_virtual_time = 0.0
        self.last_update_time = 0.0
        self.last_finish = {}     # flow_id -> 上一个包的虚拟完成时间
        self.pkt_counter = 0
    
    def add_flow(self, flow_id: int, weight: float):
        """添加一个流"""
        self.flows[flow_id] = weight
        self.last_finish[flow_id] = 0.0
    
    def _update_virtual_time(self, current_time: float):
        """更新虚拟时间"""
        if self.queue:
            # 有包等待时，虚拟时间按权重和推进
            total_weight = sum(
                self.flows.get(p.flow_id, 1) 
                for p in self.queue
            )
            if total_weight > 0:
                self.virtual_time += (current_time - self.last_update_time) / total_weight
        self.last_update_time = current_time
    
    def enqueue(self, flow_id: int, size: int, arrival_time: float):
        """入队一个包
        
        Args:
            flow_id: 流标识
            size: 包大小 (bits)
            arrival_time: 到达时间 (秒)
        """
        if flow_id not in self.flows:
            raise ValueError(f"Flow {flow_id} not registered")
        
        weight = self.flows[flow_id]
        
        # 更新虚拟时间
        self._update_virtual_time(arrival_time)
        
        # 计算虚拟开始时间
        vs = max(self.last_finish.get(flow_id, 0), self.virtual_time)
        
        # 计算虚拟完成时间
        vf = vs + size / weight
        
        self.last_finish[flow_id] = vf
        
        # 创建包并入队
        pkt = Packet(
            virtual_finish_time=vf,
            flow_id=flow_id,
            size=size,
            arrival_time=arrival_time,
            seq=self.pkt_counter
        )
        self.pkt_counter += 1
        
        heapq.heappush(self.queue, pkt)
    
    def dequeue(self) -> Optional[Packet]:
        """出队一个包（虚拟完成时间最小的包）"""
        if not self.queue:
            return None
        return heapq.heappop(self.queue)
    
    def simulate(self, packets: List[dict]) -> List[dict]:
        """模拟 WFQ 调度
        
        Args:
            packets: 包列表，每个包为 {'flow_id': int, 'size': int, 'arrival': float}
        
        Returns:
            调度结果列表
        """
        results = []
        current_time = 0.0
        
        sorted_pkts = sorted(packets, key=lambda p: p['arrival'])
        
        for pkt_info in sorted_pkts:
            self.enqueue(pkt_info['flow_id'], pkt_info['size'], pkt_info['arrival'])
        
        while self.queue:
            pkt = self.dequeue()
            service_time = pkt.size / self.link_capacity
            current_time = max(current_time, pkt.arrival_time) + service_time
            results.append({
                'flow_id': pkt.flow_id,
                'size': pkt.size,
                'arrival': pkt.arrival_time,
                'departure': current_time,
                'virtual_finish': pkt.virtual_finish_time
            })
        
        return results


# 示例：3个流共享 10 Mbps 链路
if __name__ == '__main__':
    scheduler = WFQScheduler(link_capacity=10_000_000)  # 10 Mbps
    scheduler.add_flow(flow_id=1, weight=4)  # 40%
    scheduler.add_flow(flow_id=2, weight=3)  # 30%
    scheduler.add_flow(flow_id=3, weight=3)  # 30%
    
    packets = []
    for i in range(5):
        packets.append({'flow_id': 1, 'size': 1500*8, 'arrival': i*0.001})
        packets.append({'flow_id': 2, 'size': 1000*8, 'arrival': i*0.001})
        packets.append({'flow_id': 3, 'size': 800*8, 'arrival': i*0.001})
    
    results = scheduler.simulate(packets)
    for r in results:
        print(f"Flow {r['flow_id']}: arrive={r['arrival']:.4f}, "
              f"depart={r['departure']:.6f}, vf={r['virtual_finish']:.2f}")
```

### 28.8.4 WRR（加权轮询）

**WRR（Weighted Round Robin）** 是 WFQ 的简化版本。每个流分配一个权重，调度器按权重比例轮询服务。

**工作方式**：
1. 为每个流维护一个计数器（初始值为权重）
2. 每次服务一个流时，计数器减 1
3. 计数器为 0 时重置为权重，轮到下一个流

WRR 的缺点是按包为单位调度，不考虑包大小。如果包大小不均匀，实际带宽分配可能偏离权重比例。

---

## 28.9 IntServ 模型

### 28.9.1 IntServ 概述

**IntServ（Integrated Services）** 是 IETF 在 RFC 1633 中定义的服务模型，目标是为每个流提供端到端的 QoS 保证。

**IntServ 的两种服务类别**：

1. **保证服务（Guaranteed Service）**：提供确定性的延迟上界和带宽保证。适用于对延迟有严格要求的应用。

2. **受控负载服务（Controlled-Load Service）**：在网络负载较轻时提供与 best-effort 相近的性能。不提供绝对保证，但在大多数情况下满足需求。

### 28.9.2 RSVP 协议

**RSVP（Resource Reservation Protocol）** 是 IntServ 的信令协议，用于在端到端路径上预留资源。

**RSVP 的关键概念**：
- **流（Flow）**：从源到目的的单向数据流，由源 IP、目的 IP、源端口、目的端口、协议号五元组标识
- **流规格（Flowspec）**：描述流的 QoS 需求，包括带宽、延迟、丢包率等参数
- **软状态（Soft State）**：预留状态需要周期性刷新，否则超时后自动删除

**PATH 消息**：从发送方到接收方，沿途记录路径信息

```
发送方 → PATH → 路由器1 → PATH → 路由器2 → PATH → 接收方
         (记录前一跳地址和上游接口)
```

PATH 消息包含：
- **Sender Template**：发送方 IP 和端口
- **Sender Tspec**：发送方流量特征（令牌桶参数：速率、突发大小）
- **Adspec**：沿途收集的服务特性

**RESV 消息**：从接收方到发送方，沿途预留资源

```
接收方 → RESV → 路由器2 → RESV → 路由器1 → RESV → 发送方
         (在每个路由器上安装预留状态)
```

RESV 消息包含：
- **FlowSpec**：请求的 QoS 参数
- **FilterSpec**：标识需要哪些包享受预留的服务

**RSVP 会话示例**：

```
时间 →
─────────────────────────────────────────────────
发送方           路由器A          路由器B          接收方
  │                │                │                │
  │─── PATH ──────→│                │                │
  │                │─── PATH ──────→│                │
  │                │                │─── PATH ──────→│
  │                │                │                │
  │                │                │←── RESV ───────│
  │                │←── RESV ───────│                │
  │←── RESV ───────│                │                │
  │                │                │                │
  │═══ 数据流 ════→│═══ 数据流 ════→│═══ 数据流 ════→│
  │                │                │                │
  │── PATH 刷新 ──→│                │                │
  │                │── PATH 刷新 ──→│                │
  │                │                │── PATH 刷新 ──→│
```

### 28.9.3 IntServ 的局限性

1. **可扩展性差**：每个路由器需要为每个流维护状态，路由器的流表会非常大
2. **信令开销大**：RSVP 需要周期性刷新软状态
3. **部署困难**：要求所有路由器都支持 IntServ，目前互联网上几乎未部署
4. **复杂性高**：每个路由器都需要流分类、准入控制、包调度等功能

---

## 28.10 DiffServ 模型

### 28.10.1 DiffServ 概述

**DiffServ（Differentiated Services）** 是 IETF 在 RFC 2474/2475 中定义的服务模型，解决了 IntServ 的可扩展性问题。

**DiffServ 的核心思想**：
- **分类在网络边缘**：在边界路由器对流量进行分类和标记
- **核心简单转发**：核心路由器根据标记进行简单转发，无需维护每流状态
- **聚合行为**：对同类流量聚合提供相同的转发行为

### 28.10.2 DSCP 标记

IP 头部的 ToS（Type of Service）字段被重新定义为 DS（Differentiated Services）字段：

```
 0   1   2   3   4   5   6   7
+---+---+---+---+---+---+---+---+
|         DSCP          |  CU   |
+---+---+---+---+---+---+---+---+
```

**DSCP（Differentiated Services Code Point）** 占 6 位，可表示 64 个不同的值（0-63）。每个 DSCP 值对应一种**每跳行为（PHB，Per-Hop Behavior）**。

**常用 DSCP 值**：

| DSCP 值 | 二进制 | PHB 类别 | 名称 | 用途 |
|---------|--------|---------|------|------|
| 0 | 000000 | BE | 尽力而为 | 默认 |
| 10 | 001010 | AF11 | 确保转发 | 低优先级数据 |
| 12 | 001100 | AF12 | 确保转发 | 低优先级数据 |
| 14 | 001110 | AF13 | 确保转发 | 低优先级数据 |
| 18 | 010010 | AF21 | 确保转发 | 中优先级数据 |
| 26 | 011010 | AF31 | 确保转发 | 高优先级数据 |
| 34 | 100010 | AF41 | 确保转发 | 信令 |
| 46 | 101110 | EF | 加速转发 | 语音 |
| 48 | 110000 | CS6 | 类选择器 | 网络控制 |
| 56 | 111000 | CS7 | 类选择器 | 网络控制 |

### 28.10.3 PHB（每跳行为详解）

**EF（Expedited Forwarding，加速转发）**：

EF 提供低延迟、低抖动、低丢包的服务，DSCP = 46（101110）。

**数学定义**：对于任何时间间隔 $[t, t+T]$，离开路由器的 EF 流量应满足：

$$\frac{S(t+T) - S(t)}{T} \leq r$$

其中 $S(t)$ 是到时间 $t$ 已发送的 EF 字节数，$r$ 是 EF 的配置速率。

**实现方式**：
- 使用严格优先级队列
- 在入口处限速，确保 EF 流量不超过配置速率
- EF 包到达时立即调度，不排队等待

**AF（Assured Forwarding，确保转发）**：

AF 定义了 4 个类别（AF1x-AF4x），每个类别有 3 个丢弃优先级：

```
AF1: AF11(low) → AF12(medium) → AF13(high)
AF2: AF21(low) → AF22(medium) → AF23(high)
AF3: AF31(low) → AF32(medium) → AF33(high)
AF4: AF41(low) → AF42(medium) → AF43(high)
```

**AF 的工作原理**：
1. 在网络入口，标记器根据流量 profile 标记包的颜色（绿/黄/红）
2. 绿色包（符合 profile）标记为低丢弃优先级
3. 红色包（超出 profile）标记为高丢弃优先级
4. 在拥塞时，优先丢弃高丢弃优先级的包

**BE（Best-Effort，尽力而为）**：

DSCP = 0，与传统 IP 服务相同，无任何保证。

<div data-component="DiffServClassifier"></div>

### 28.10.4 DiffServ 实现架构

**边界路由器功能**：
1. **分类（Classification）**：根据包头字段（源/目的 IP、端口、协议号等）将包分类
2. **计量（Metering）**：测量流量是否符合流量 profile
3. **标记（Marking）**：根据分类和计量结果设置 DSCP 值
4. **整形/丢弃（Shaping/Dropping）**：对超出 profile 的流量进行整形或丢弃

**核心路由器功能**：
1. 根据 DSCP 值将包映射到对应的 PHB
2. 使用调度算法（如 WFQ）为不同 PHB 提供不同的服务

---

## 28.11 流量整形

### 28.11.1 流量整形的必要性

流量整形（Traffic Shaping）控制数据发送的速率，使流量符合预定的 profile。主要目的：
1. 避免网络拥塞
2. 确保 QoS 承诺
3. 平滑突发流量

### 28.11.2 漏桶算法（Leaky Bucket）

**基本思想**：将不规则的输入流量转化为规则的输出流量。想象一个底部有固定大小孔的桶：
- 水（数据包）从上方倒入
- 水以恒定速率从孔中流出
- 桶满时，多余的水溢出（丢包）

**数学模型**：

设桶的容量为 $B$（字节），输出速率为 $r$（字节/秒）。

在时间间隔 $[t, t+\tau]$ 内，输出数据量为：

$$O(t, t+\tau) = \min(W(t), r\tau) + \min(B, \text{输入量} - W(t))$$

其中 $W(t)$ 是时间 $t$ 时桶中的水量。

**漏桶的约束条件**：任何时间间隔 $\tau$ 内的输入数据量必须满足：

$$\text{Input}(t, t+\tau) \leq B + r\tau$$

这是漏桶算法的**流量约束**，定义了一个 $(B, r)$ 的流量 profile。

**突发容忍**：漏桶可以容忍最大 $B$ 字节的突发。如果桶为空，可以立即接受 $B$ 字节的突发数据，然后以速率 $r$ 持续输出。

```python
import time
from dataclasses import dataclass, field
from typing import List, Tuple

class LeakyBucket:
    """漏桶流量整形器"""
    
    def __init__(self, rate: float, bucket_size: float):
        """
        Args:
            rate: 漏出速率 (bytes/sec)
            bucket_size: 桶容量 (bytes)
        """
        self.rate = rate
        self.bucket_size = bucket_size
        self.water_level = 0.0      # 当前水位
        self.last_leak_time = time.time()
        self.stats = {
            'accepted': 0,
            'dropped': 0,
            'leaked': 0
        }
    
    def _leak(self, current_time: float):
        """计算从上次到现在漏出的水量"""
        elapsed = current_time - self.last_leak_time
        leaked = elapsed * self.rate
        self.water_level = max(0, self.water_level - leaked)
        self.last_leak_time = current_time
    
    def arrive(self, data_size: float, current_time: float = None) -> bool:
        """数据到达事件
        
        Args:
            data_size: 到达数据大小 (bytes)
            current_time: 当前时间 (秒)
        
        Returns:
            True 如果数据被接受，False 如果被丢弃
        """
        if current_time is None:
            current_time = time.time()
        
        self._leak(current_time)
        
        if self.water_level + data_size <= self.bucket_size:
            self.water_level += data_size
            self.stats['accepted'] += 1
            return True
        else:
            self.stats['dropped'] += 1
            return False
    
    def get_output_rate(self) -> float:
        """获取当前输出速率"""
        return self.rate
    
    def get_water_level(self) -> float:
        """获取当前水位"""
        return self.water_level


class LeakyBucketSimulator:
    """漏桶仿真器"""
    
    def __init__(self, rate: float, bucket_size: float):
        self.bucket = LeakyBucket(rate, bucket_size)
        self.events: List[Tuple[float, float, bool, float]] = []
    
    def simulate(self, arrivals: List[Tuple[float, float]]) -> dict:
        """仿真一组到达事件
        
        Args:
            arrivals: [(到达时间, 数据大小), ...]
        
        Returns:
            仿真结果
        """
        accepted = 0
        dropped = 0
        total_delay = 0
        
        for arrival_time, data_size in arrivals:
            result = self.bucket.arrive(data_size, arrival_time)
            water = self.bucket.get_water_level()
            
            if result:
                accepted += 1
                # 计算排队延迟
                delay = water / self.bucket.rate
                total_delay += delay
            else:
                dropped += 1
            
            self.events.append((arrival_time, data_size, result, water))
        
        return {
            'accepted': accepted,
            'dropped': dropped,
            'drop_rate': dropped / len(arrivals) if arrivals else 0,
            'avg_delay': total_delay / accepted if accepted > 0 else 0,
            'events': self.events
        }


# 仿真示例
if __name__ == '__main__':
    sim = LeakyBucketSimulator(rate=1000, bucket_size=5000)
    
    arrivals = [
        (0.0, 2000),    # 正常流量
        (0.5, 3000),    # 突发
        (1.0, 1500),    # 突发持续
        (1.5, 4000),    # 超大突发
        (2.0, 1000),    # 恢复正常
        (3.0, 2000),
        (4.0, 1500),
    ]
    
    result = sim.simulate(arrivals)
    print(f"Accepted: {result['accepted']}, Dropped: {result['dropped']}")
    print(f"Drop rate: {result['drop_rate']:.2%}")
    print(f"Average delay: {result['avg_delay']:.4f}s")
    
    for t, size, ok, water in result['events']:
        status = "OK" if ok else "DROP"
        print(f"  t={t:.1f}, size={size}, {status}, water={water:.0f}")
```

### 28.11.3 令牌桶算法（Token Bucket）

**基本思想**：令牌以恒定速率产生并放入桶中。每个数据包发送时需要消耗一个令牌。桶满时令牌被丢弃。

**与漏桶的区别**：
- 漏桶：输出速率恒定，完全平滑
- 令牌桶：允许突发（桶中有积累的令牌时可以一次性发送）

**令牌桶参数**：
- **CBS（Committed Burst Size）**：承诺突发大小
- **PBS（Peak Burst Size）**：峰值突发大小
- **CIR（Committed Information Rate）**：承诺信息速率
- **PIR（Peak Information Rate）**：峰值信息速率

**双速率三色标记（trTCM）**：

```
如果到达包大小 ≤ 当前 CBS 令牌：标记为绿色（低丢弃优先级）
如果到达包大小 ≤ 当前 PBS 令牌：标记为黄色（中丢弃优先级）
否则：标记为红色（高丢弃优先级）
```

**令牌桶数学分析**：

设令牌产生速率为 $r$，桶容量为 $b$。如果系统空闲了 $t$ 时间，桶中的令牌数为：

$$\text{tokens} = \min(b, r \cdot t)$$

突发传输能力：如果桶满（$b$ 个令牌），可以在瞬间发送 $b$ 字节的数据，之后持续速率为 $r$。

**令牌桶与漏桶的级联**：

```
输入 → [令牌桶] → [漏桶] → 输出

令牌桶：允许突发，平均速率 ≤ r_t
漏桶：平滑输出，输出速率 = r_l (r_l ≥ r_t)
```

<div data-component="TokenBucketShaper"></div>

### 28.11.4 令牌桶实现

```python
class TokenBucket:
    """令牌桶流量整形器"""
    
    def __init__(self, rate: float, bucket_size: float):
        """
        Args:
            rate: 令牌产生速率 (tokens/sec)
            bucket_size: 桶容量 (tokens)
        """
        self.rate = rate
        self.bucket_size = bucket_size
        self.tokens = bucket_size  # 初始满桶
        self.last_refill = time.time()
    
    def _refill(self, current_time: float):
        """补充令牌"""
        elapsed = current_time - self.last_refill
        new_tokens = elapsed * self.rate
        self.tokens = min(self.bucket_size, self.tokens + new_tokens)
        self.last_refill = current_time
    
    def consume(self, size: float, current_time: float = None) -> bool:
        """尝试消费令牌
        
        Args:
            size: 需要的令牌数
            current_time: 当前时间
        
        Returns:
            True 如果令牌足够，False 否则
        """
        if current_time is None:
            current_time = time.time()
        
        self._refill(current_time)
        
        if self.tokens >= size:
            self.tokens -= size
            return True
        return False
    
    def get_tokens(self) -> float:
        """获取当前令牌数"""
        return self.tokens


class DualTokenBucket:
    """双速率三色标记器 (trTCM) - RFC 2698"""
    
    def __init__(self, cir: float, cbs: float, pir: float, pbs: float):
        """
        Args:
            cir: 承诺信息速率 (bytes/sec)
            cbs: 承诺突发大小 (bytes)
            pir: 峰值信息速率 (bytes/sec)
            pbs: 峰值突发大小 (bytes)
        """
        self.cir = cir
        self.cbs = cbs
        self.pir = pir
        self.pbs = pbs
        self.c_tokens = cbs    # 承诺桶
        self.p_tokens = pbs    # 峰值桶
        self.last_update = time.time()
    
    def _refill(self, current_time: float):
        elapsed = current_time - self.last_update
        self.c_tokens = min(self.cbs, self.c_tokens + elapsed * self.cir)
        self.p_tokens = min(self.pbs, self.p_tokens + elapsed * self.pir)
        self.last_update = current_time
    
    def mark(self, packet_size: float, current_time: float = None) -> str:
        """标记数据包颜色
        
        Args:
            packet_size: 包大小 (bytes)
            current_time: 当前时间
        
        Returns:
            'green', 'yellow', 或 'red'
        """
        if current_time is None:
            current_time = time.time()
        
        self._refill(current_time)
        
        if self.c_tokens >= packet_size:
            self.c_tokens -= packet_size
            self.p_tokens -= min(self.p_tokens, packet_size)
            return 'green'
        elif self.p_tokens >= packet_size:
            self.p_tokens -= packet_size
            return 'yellow'
        else:
            return 'red'


# 测试
if __name__ == '__main__':
    # 单令牌桶
    bucket = TokenBucket(rate=1000, bucket_size=5000)
    print("Token Bucket Test:")
    for i in range(10):
        ok = bucket.consume(1500)
        print(f"  consume 1500: {'OK' if ok else 'REJECT'}, "
              f"tokens={bucket.get_tokens():.0f}")
    
    # 双速率三色标记
    print("\nDual Token Bucket (trTCM) Test:")
    trtcm = DualTokenBucket(cir=1000, cbs=2000, pir=2000, pbs=4000)
    test_sizes = [500, 1000, 800, 1500, 2000, 500, 1000]
    for size in test_sizes:
        color = trtcm.mark(size)
        print(f"  packet {size}: {color}")
```

### 28.11.5 漏桶与令牌桶的数学对比

| 特性 | 漏桶 | 令牌桶 |
|------|------|--------|
| 输出速率 | 恒定 $r$ | 平均 $r$，可突发 |
| 突发容忍 | $B$ 字节（桶容量） | $b$ 字节（令牌桶容量） |
| 最大瞬时速率 | $r$ | 无限制（桶满时） |
| 流量约束 | $A(t, t+\tau) \leq B + r\tau$ | $A(t, t+\tau) \leq b + r\tau$ |
| 平滑效果 | 强 | 弱（允许突发） |

**漏桶的流量约束推导**：

设漏桶参数为 $(B, r)$，其中 $B$ 为桶容量，$r$ 为输出速率。

在任意时间间隔 $[t_1, t_2]$ 内：
- 最大输出量 = $r(t_2 - t_1)$
- 桶中初始水量 ≤ $B$
- 输入量 = 输出量 + 桶中水量变化

$$A(t_1, t_2) = O(t_1, t_2) + W(t_2) - W(t_1)$$

$$A(t_1, t_2) \leq r(t_2 - t_1) + B - 0 = B + r(t_2 - t_1)$$

因此，对于任何时间间隔 $\tau$：

$$A(t, t+\tau) \leq B + r\tau$$

这就是漏桶算法的**到达曲线（Arrival Curve）**。

---

## 28.12 综合案例：多媒体网络系统设计

### 28.12.1 端到端 QoS 保证

一个完整的多媒体网络系统需要在每个环节提供 QoS 保证：

```
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐
│  编码器  │───→│ 打包/RTP │───→│  网络传输 │───→│ 播放器  │
│H.264/AAC│    │          │    │ QoS机制  │    │缓冲+解码│
└─────────┘    └──────────┘    └──────────┘    └─────────┘
     ↓               ↓              ↓               ↓
  编码延迟       打包延迟      网络延迟+抖动     缓冲延迟
```

**QoS 参数映射**：

| 应用需求 | 编码器参数 | 网络参数 | 播放器参数 |
|---------|-----------|---------|-----------|
| 延迟 < 150ms | 低延迟编码模式 | DiffServ EF | 小缓冲区 |
| 丢包 < 1% | 内置 FEC | AF 类别 | 错误隐藏 |
| 抖动 < 30ms | 固定帧大小 | 流量整形 | 自适应缓冲 |

### 28.12.2 DASH 客户端仿真

```python
import random
import math

class DASHClient:
    """DASH 自适应流客户端仿真"""
    
    # 可用码率 (bps)
    BITRATES = [500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]
    SEGMENT_DURATION = 2.0  # 秒
    MAX_BUFFER = 30.0       # 最大缓冲时间 (秒)
    MIN_BUFFER = 2.0        # 最小缓冲时间 (秒)
    
    def __init__(self, initial_bitrate_idx=0):
        self.bitrate_idx = initial_bitrate_idx
        self.buffer_level = 0.0
        self.downloaded_segments = 0
        self.total_bytes = 0
        self.bitrate_history = []
        self.buffer_history = []
    
    def _estimate_bandwidth(self, download_size: int, download_time: float) -> float:
        """估计可用带宽"""
        return download_size * 8 / download_time
    
    def _select_bitrate_bba(self) -> int:
        """基于缓冲区的码率选择 (BBA)"""
        buffer_ratio = self.buffer_level / self.MAX_BUFFER
        
        # 将缓冲区水平映射到码率索引
        n = len(self.BITRATES)
        idx = int(buffer_ratio * n)
        return min(max(idx, 0), n - 1)
    
    def _select_bitrate_throughput(self, estimated_bw: float) -> int:
        """基于吞吐量的码率选择"""
        # 选择不超过估计带宽 80% 的最高码率
        safe_bw = estimated_bw * 0.8
        best_idx = 0
        for i, br in enumerate(self.BITRATES):
            if br <= safe_bw:
                best_idx = i
        return best_idx
    
    def _select_bitrate_hybrid(self, estimated_bw: float) -> int:
        """混合码率选择策略"""
        bw_idx = self._select_bitrate_throughput(estimated_bw)
        buf_idx = self._select_bitrate_bba()
        
        # 当缓冲区低时偏向吞吐量策略，高时偏向缓冲区策略
        if self.buffer_level < self.MIN_BUFFER * 2:
            return min(bw_idx, buf_idx)
        elif self.buffer_level > self.MAX_BUFFER * 0.7:
            return max(bw_idx, buf_idx)
        else:
            return (bw_idx + buf_idx) // 2
    
    def download_segment(self, available_bw: float) -> float:
        """下载一个视频分片
        
        Args:
            available_bw: 可用带宽 (bps)
        
        Returns:
            下载时间 (秒)
        """
        bitrate = self.BITRATES[self.bitrate_idx]
        segment_size = bitrate * self.SEGMENT_DURATION / 8  # bytes
        
        # 模拟下载（带宽可能低于码率）
        effective_bw = min(available_bw, bitrate * 1.2)
        download_time = segment_size * 8 / effective_bw
        
        # 更新状态
        self.downloaded_segments += 1
        self.total_bytes += segment_size
        self.buffer_level += self.SEGMENT_DURATION
        
        # 限制缓冲区大小
        if self.buffer_level > self.MAX_BUFFER:
            self.buffer_level = self.MAX_BUFFER
        
        self.bitrate_history.append(self.BITRATES[self.bitrate_idx])
        self.buffer_history.append(self.buffer_level)
        
        return download_time
    
    def consume_buffer(self, duration: float):
        """消费缓冲区（播放）"""
        self.buffer_level = max(0, self.buffer_level - duration)
    
    def simulate(self, bandwidth_trace: list, strategy='hybrid') -> dict:
        """仿真 DASH 播放
        
        Args:
            bandwidth_trace: 带宽变化轨迹 [(时间, 带宽), ...]
            strategy: 'throughput', 'buffer', 'hybrid'
        
        Returns:
            仿真结果
        """
        total_time = 0
        rebuffer_count = 0
        rebuffer_time = 0
        estimated_bw = bandwidth_trace[0][1] if bandwidth_trace else 1_000_000
        
        for i, (t, bw) in enumerate(bandwidth_trace):
            # 选择码率
            if strategy == 'throughput':
                self.bitrate_idx = self._select_bitrate_throughput(estimated_bw)
            elif strategy == 'buffer':
                self.bitrate_idx = self._select_bitrate_bba()
            else:
                self.bitrate_idx = self._select_bitrate_hybrid(estimated_bw)
            
            # 检查是否需要缓冲等待
            if self.buffer_level < self.MIN_BUFFER:
                rebuffer_count += 1
                wait_time = self.MIN_BUFFER - self.buffer_level
                rebuffer_time += wait_time
                total_time += wait_time
            
            # 下载分片
            download_time = self.download_segment(bw)
            total_time += download_time
            
            # 更新带宽估计
            segment_size = self.BITRATES[self.bitrate_idx] * self.SEGMENT_DURATION / 8
            estimated_bw = self._estimate_bandwidth(segment_size, download_time)
            
            # 消费缓冲区（播放时间）
            self.consume_buffer(self.SEGMENT_DURATION)
        
        return {
            'total_segments': self.downloaded_segments,
            'total_bytes': self.total_bytes,
            'total_time': total_time,
            'rebuffer_count': rebuffer_count,
            'rebuffer_time': rebuffer_time,
            'avg_bitrate': sum(self.bitrate_history) / len(self.bitrate_history) 
                           if self.bitrate_history else 0,
            'bitrate_switches': sum(1 for i in range(1, len(self.bitrate_history))
                                   if self.bitrate_history[i] != self.bitrate_history[i-1]),
            'bitrate_history': self.bitrate_history,
            'buffer_history': self.buffer_history
        }


# 仿真示例
if __name__ == '__main__':
    # 模拟带宽波动
    bandwidth_trace = []
    for i in range(50):
        t = i * 2.0
        # 带宽在 1-15 Mbps 之间波动
        bw = 3_000_000 + 7_000_000 * abs(math.sin(i * 0.3))
        bandwidth_trace.append((t, bw))
    
    for strategy in ['throughput', 'buffer', 'hybrid']:
        client = DASHClient()
        result = client.simulate(bandwidth_trace, strategy)
        print(f"\nStrategy: {strategy}")
        print(f"  Avg bitrate: {result['avg_bitrate']/1e6:.2f} Mbps")
        print(f"  Rebuffer count: {result['rebuffer_count']}")
        print(f"  Rebuffer time: {result['rebuffer_time']:.2f}s")
        print(f"  Bitrate switches: {result['bitrate_switches']}")
        print(f"  Total downloaded: {result['total_bytes']/1e6:.2f} MB")
```

### 28.12.3 抖动缓冲区自适应

```python
class AdaptiveJitterBuffer:
    """自适应抖动缓冲区"""
    
    def __init__(self, min_delay_ms=20, max_delay_ms=400):
        self.min_delay_ms = min_delay_ms
        self.max_delay_ms = max_delay_ms
        self.target_delay_ms = 60   # 初始目标延迟
        self.jitter_ema = 0.0       # 抖动的指数移动平均
        self.alpha = 0.05           # EMA 系数
        self.packets = []           # 缓冲区
        self.stats = {'adapted': 0, 'underrun': 0, 'overrun': 0}
    
    def update_jitter(self, jitter_sample_ms: float):
        """更新抖动估计"""
        self.jitter_ema = (1 - self.alpha) * self.jitter_ema + \
                          self.alpha * jitter_sample_ms
        
        # 根据抖动调整目标延迟
        # 目标延迟 = 安全系数 × 抖动估计 + 最小延迟
        new_target = 3.0 * self.jitter_ema + self.min_delay_ms
        new_target = max(self.min_delay_ms, min(self.max_delay_ms, new_target))
        
        if abs(new_target - self.target_delay_ms) > 5:
            self.target_delay_ms = new_target
            self.stats['adapted'] += 1
    
    def get_playout_delay(self) -> float:
        """获取当前播放延迟"""
        return self.target_delay_ms
```

---

## 28.13 本章小结

本章全面介绍了多媒体网络与服务质量的核心概念和技术：

| 主题 | 关键技术 | 核心要点 |
|------|---------|---------|
| 音视频数字化 | Nyquist 采样、PCM、H.264/H.265 | 采样率 ≥ 2f_max；帧内/帧间预测+DCT |
| 流式视频 | DASH、ABR、CDN | MPD 描述、自适应码率、边缘缓存 |
| 传输协议 | RTP/RTCP/RTSP | 时间戳语义、质量反馈、会话控制 |
| QoS 分析 | 延迟分解、抖动、丢包 | 端到端延迟 <150ms；抖动缓冲区 |
| 调度算法 | FIFO、WFQ | 虚拟时钟、带权公平分配 |
| 服务模型 | IntServ、DiffServ | 流级保证 vs 类级保证 |
| 流量整形 | 漏桶、令牌桶 | 流量约束 $A \leq B + r\tau$；突发容忍 |

**IntServ vs DiffServ 对比**：

| 特性 | IntServ | DiffServ |
|------|---------|----------|
| 粒度 | 每流 | 每类 |
| 状态维护 | 每流状态 | 仅边缘 |
| 信令 | RSVP | 配置/边缘标记 |
| 可扩展性 | 差 | 好 |
| QoS 保证 | 确定性 | 统计性 |
| 部署情况 | 几乎未部署 | 广泛部署 |

**关键公式汇总**：

- 采样定理：$f_s \geq 2f_{max}$
- PCM 数据率：$R = f_s \times n \times c$
- 量化 SNR：$SNR = 6.02n + 1.76$ dB
- M/M/1 平均延迟：$W = \frac{1}{\mu - \lambda}$
- WFQ 虚拟完成时间：$F_i^k = S_i^k + \frac{L_i^k}{\phi_i}$
- 漏桶约束：$A(t, t+\tau) \leq B + r\tau$
- RTT 计算：$RTT = T_3 - T_1 - DLSR$
- Little's Law：$L = \lambda W$

---

## 28.14 练习题

### 概念题

1. 解释为什么实时多媒体应用通常使用 UDP 而非 TCP？TCP 的哪些机制对实时应用是不利的？

2. 描述 RTP 时间戳与 NTP 时间戳的区别和联系。为什么 RTCP SR 中同时包含这两种时间戳？

3. 比较 IntServ 和 DiffServ 的优缺点。为什么 DiffServ 在互联网上得到更广泛的部署？

### 计算题

4. 一个 VoIP 应用使用 G.711 编码（8kHz 采样，8bit 量化），每 20ms 生成一个语音帧。计算：
   - (a) PCM 数据率
   - (b) 每个 RTP 包的载荷大小
   - (c) 包含 IP/UDP/RTP 头部后的总包大小
   - (d) 链路层效率（载荷/总包大小）

5. 一个 WFQ 调度器有 3 个流，权重分别为 2:1:1，链路容量为 10 Mbps。流 1 持续以 6 Mbps 发送，流 2 和流 3 各以 2 Mbps 发送。计算每个流获得的实际带宽。

6. 一个漏桶参数为 $B=5000$ 字节，$r=1000$ 字节/秒。如果在 $t=0$ 时刻桶为空，突然到达 8000 字节的数据，计算所有数据完全输出所需的时间。

### 分析题

7. 分析 DASH 中 ABR 算法面临的"公平性"问题：当多个 DASH 客户端共享同一条瓶颈链路时，可能出现什么问题？如何解决？

8. 设计一个简单的实验，用 Python 模证 WFQ 的公平性。假设 3 个流共享 1 Mbps 链路，权重为 3:2:1，包大小均为 1000 字节。验证每个流在 10 秒内获得的带宽是否符合权重比例。

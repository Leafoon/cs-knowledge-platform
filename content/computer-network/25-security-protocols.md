# Chapter 25: 网络安全协议 — TLS、IPsec 与防火墙

> **学习目标**：
> - 掌握 TLS 1.2 握手的完整流程，理解每一步的字段含义与密钥协商机制
> - 理解 TLS 1.3 相对于 1.2 的关键改进，包括 1-RTT 握手、0-RTT 恢复与 HKDF 密钥派生
> - 了解 TLS 记录协议的分片、加密与 MAC 计算过程
> - 掌握 IPsec 的 AH/ESP 协议头格式、传输模式与隧道模式的区别
> - 理解 IKEv2 的 SA 协商与密钥交换过程
> - 了解防火墙的演进：从包过滤到下一代防火墙（NGFW）
> - 掌握 IDS/IPS 的工作原理、部署模式与 Snort 规则编写

---

## 25.1 TLS 协议概述

TLS（Transport Layer Security）是目前互联网上最广泛使用的传输层安全协议，其前身是 Netscape 开发的 SSL（Secure Sockets Layer）。TLS 的核心目标是为 TCP 连接提供三个安全保障：

1. **机密性（Confidentiality）**：通过加密防止窃听
2. **完整性（Integrity）**：通过 MAC 防止篡改
3. **身份认证（Authentication）**：通过数字证书验证对端身份

### 25.1.1 TLS 协议栈

TLS 协议由多个子协议组成，构建在 TCP 之上：

```
┌─────────────────────────────────────────────┐
│              应用层数据 (HTTP, SMTP...)       │
├─────────────────────────────────────────────┤
│           TLS 握手协议 (Handshake)           │
│           TLS 密码变更协议 (ChangeCipherSpec) │
│           TLS 告警协议 (Alert)               │
├─────────────────────────────────────────────┤
│           TLS 记录协议 (Record)              │
├─────────────────────────────────────────────┤
│                 TCP                          │
├─────────────────────────────────────────────┤
│                 IP                           │
└─────────────────────────────────────────────┘
```

- **握手协议**：协商密码套件、交换证书、生成会话密钥
- **密码变更协议**：通知对端切换到协商好的加密参数
- **告警协议**：报告错误和关闭通知
- **记录协议**：对上层数据进行分片、压缩（可选）、加密和认证

### 25.1.2 TLS 版本演进

| 版本 | 年份 | 关键特性 | 当前状态 |
|------|------|----------|----------|
| SSL 2.0 | 1995 | 首个公开版本 | 已弃用（严重安全漏洞） |
| SSL 3.0 | 1996 | 修复 SSL 2.0 问题 | 已弃用（POODLE 攻击） |
| TLS 1.0 | 1999 | 基于 SSL 3.0，IANA 标准化 | 已弃用（BEAST 攻击） |
| TLS 1.1 | 2006 | 显式 IV，分片修复 | 已弃用 |
| TLS 1.2 | 2008 | SHA-256、AEAD 密码套件 | 广泛使用 |
| TLS 1.3 | 2018 | 1-RTT 握手、移除不安全算法 | 最新标准，快速普及中 |

---

## 25.2 TLS 1.2 握手详细过程

TLS 1.2 的完整握手需要 **两个往返（2-RTT）** 才能完成，之后客户端才能发送应用数据。

<div data-component="TLShandshakeengine"></div>

### 25.2.1 握手流程概览

```
客户端 (Client)                                服务器 (Server)
    │                                              │
    │  ──── 1. ClientHello ──────────────────────> │
    │                                              │
    │  <──── 2. ServerHello ─────────────────────  │
    │  <──── 3. Certificate ─────────────────────  │
    │  <──── 4. ServerKeyExchange ───────────────  │
    │  <──── 5. CertificateRequest (可选) ────────  │
    │  <──── 6. ServerHelloDone ─────────────────  │
    │                                              │
    │  ──── 7. Certificate (可选) ───────────────> │
    │  ──── 8. ClientKeyExchange ────────────────> │
    │  ──── 9. CertificateVerify (可选) ─────────> │
    │  ──── 10. ChangeCipherSpec ────────────────> │
    │  ──── 11. Finished ───────────────────────> │
    │                                              │
    │  <──── 12. ChangeCipherSpec ───────────────  │
    │  <──── 13. Finished ──────────────────────  │
    │                                              │
    │  ◄════ 应用数据（加密传输）════════════════> │
```

### 25.2.2 ClientHello 消息

客户端发起握手，发送自己支持的参数：

```
ClientHello 消息格式:
┌──────────────────────────────────────────────────────┐
│ handshake_type (1 byte) = 0x01 (ClientHello)         │
│ length (3 bytes)                                      │
├──────────────────────────────────────────────────────┤
│ client_version (2 bytes) = 0x0303 (TLS 1.2)          │
│ random (32 bytes):                                    │
│   ├── gmt_unix_time (4 bytes)                         │
│   └── random_bytes (28 bytes)                         │
│ session_id_length (1 byte)                            │
│ session_id (0-32 bytes)                               │
│ cipher_suites_length (2 bytes)                        │
│ cipher_suites (2*N bytes):                            │
│   ├── TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256 (0xC02F)│
│   ├── TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384 (0xC030)│
│   ├── TLS_RSA_WITH_AES_128_CBC_SHA (0x002F)          │
│   └── ...                                             │
│ compression_methods_length (1 byte)                   │
│ compression_methods (1*N bytes):                      │
│   └── 0x00 (null compression)                         │
│ extensions_length (2 bytes)                           │
│ extensions:                                           │
│   ├── server_name (SNI)                               │
│   ├── supported_groups (elliptic_curves)              │
│   ├── ec_point_formats                                │
│   ├── signature_algorithms                            │
│   ├── application_layer_protocol_negotiation (ALPN)   │
│   └── ...                                             │
└──────────────────────────────────────────────────────┘
```

**关键字段解析**：

- `random`：32 字节随机数，用于后续密钥生成。前 4 字节是 Unix 时间戳，后 28 字节是 CSPRNG 生成的随机数
- `cipher_suites`：客户端支持的密码套件列表，按优先级排序。每个套件定义了密钥交换、认证、加密和 MAC 算法
- `extensions`：扩展字段，SNI（Server Name Indication）允许一个 IP 地址托管多个 HTTPS 站点

### 25.2.3 ServerHello 消息

服务器从客户端列表中选择一个密码套件：

```
ServerHello 消息格式:
┌──────────────────────────────────────────────────────┐
│ handshake_type (1 byte) = 0x02 (ServerHello)         │
│ length (3 bytes)                                      │
├──────────────────────────────────────────────────────┤
│ server_version (2 bytes) = 0x0303 (TLS 1.2)          │
│ random (32 bytes)                                     │
│ session_id_length (1 byte)                            │
│ session_id (0-32 bytes)                               │
│ cipher_suite (2 bytes) = 0xC02F (选定的套件)          │
│ compression_method (1 byte) = 0x00                    │
│ extensions                                            │
└──────────────────────────────────────────────────────┘
```

### 25.2.4 Certificate 消息

服务器发送其 X.509 证书链：

```
Certificate 消息:
┌──────────────────────────────────────────────────────┐
│ handshake_type (1 byte) = 0x0B                       │
│ length (3 bytes)                                      │
├──────────────────────────────────────────────────────┤
│ certificates_length (3 bytes)                         │
│ certificate_list:                                     │
│   ├── certificate_1_length (3 bytes)                  │
│   │   └── X.509 certificate (leaf/server cert)       │
│   ├── certificate_2_length (3 bytes)                  │
│   │   └── X.509 certificate (intermediate CA)        │
│   └── certificate_3_length (3 bytes)                  │
│       └── X.509 certificate (root CA, 可选)           │
└──────────────────────────────────────────────────────┘
```

客户端收到证书后执行验证链：
1. 检查证书是否由受信任的 CA 签发
2. 检查证书是否过期
3. 检查证书中的域名是否匹配请求的服务器
4. 检查证书是否被吊销（CRL 或 OCSP）

### 25.2.5 ServerKeyExchange 消息

当使用 DHE 或 ECDHE 密钥交换时，服务器发送此消息：

```
对于 ECDHE_RSA:
┌──────────────────────────────────────────────────────┐
│ ECCurveType (1 byte) = 0x03 (named_curve)            │
│ named_curve (2 bytes) = 0x0017 (secp256r1)           │
│ public_key_length (1 byte)                            │
│ public_key (point)                                    │
│ signature_algorithm (2 bytes)                         │
│ signature_length (2 bytes)                            │
│ signature (数字签名覆盖 server.random + client.random │
│            + curve_params + public_key)               │
└──────────────────────────────────────────────────────┘
```

### 25.2.6 密钥协商过程

以 ECDHE_RSA 为例，密钥协商分为三个阶段：

**阶段一：Pre-Master Secret 生成**

```
客户端:
  1. 生成 ECDH 临时密钥对 (client_privkey, client_pubkey)
  2. 用 server_pubkey 和 client_privkey 计算共享密钥:
     shared_secret = ECDH(client_privkey, server_pubkey)

服务器:
  1. 生成 ECDH 临时密钥对 (server_privkey, server_pubkey)
  2. 用 client_pubkey 和 server_privkey 计算共享密钥:
     shared_secret = ECDH(server_privkey, client_pubkey)

双方共享: pre_master_secret = shared_secret (32 bytes for P-256)
```

**阶段二：Master Secret 推导**

```
master_secret = PRF(pre_master_secret,
                    "master secret",
                    client_random + server_random)[0..47]

PRF (Pseudo-Random Function):
  使用 HMAC-SHA256 进行扩展:
  P_hash(secret, seed) = HMAC_hash(secret, A(1) + seed) +
                          HMAC_hash(secret, A(2) + seed) + ...
  其中 A(0) = seed, A(i) = HMAC_hash(secret, A(i-1))
```

**阶段三：密钥块生成**

```
key_block = PRF(master_secret,
                "key expansion",
                server_random + client_random)[0..所需长度]

密钥块分割:
┌──────────────────────────────────────────────────┐
│ client_write_MAC_key (20 bytes, SHA-1)           │
│ server_write_MAC_key (20 bytes, SHA-1)           │
│ client_write_key (16 bytes, AES-128)             │
│ server_write_key (16 bytes, AES-128)             │
│ client_write_IV (4 bytes, GCM nonce)             │
│ server_write_IV (4 bytes, GCM nonce)             │
└──────────────────────────────────────────────────┘
```

各密钥用途说明：

| 密钥 | 用途 |
|------|------|
| `client_write_MAC_key` | 客户端发送数据的 MAC 计算密钥 |
| `server_write_MAC_key` | 服务器发送数据的 MAC 计算密钥 |
| `client_write_key` | 客户端到服务器方向的加密密钥 |
| `server_write_key` | 服务器到客户端方向的加密密钥 |
| `client_write_IV` | 客户端加密的初始化向量 |
| `server_write_IV` | 服务器加密的初始化向量 |

### 25.2.7 ClientKeyExchange 消息

```
ClientKeyExchange (ECDHE):
┌──────────────────────────────────────────────────────┐
│ handshake_type (1 byte) = 0x10                       │
│ length (3 bytes)                                      │
├──────────────────────────────────────────────────────┤
│ public_key_length (1 byte)                            │
│ client_public_key (ECDH 公钥点)                       │
└──────────────────────────────────────────────────────┘
```

### 25.2.8 ChangeCipherSpec 与 Finished

```
ChangeCipherSpec 消息:
┌──────────────────────────────────────────────────────┐
│ type (1 byte) = 0x14                                 │
│ version (2 bytes) = 0x0303                           │
│ length (2 bytes) = 0x0001                            │
│ change_cipher_spec (1 byte) = 0x01                   │
└──────────────────────────────────────────────────────┘

Finished 消息 (加密后的握手消息):
┌──────────────────────────────────────────────────────┐
│ handshake_type (1 byte) = 0x14                       │
│ length (3 bytes) = 0x0020 (32 bytes)                 │
│ verify_data = PRF(master_secret,                     │
│                   "client finished" / "server finished",│
│                   SHA-256(handshake_messages))[0..11] │
│                   → 12 bytes for TLS 1.2              │
└──────────────────────────────────────────────────────┘
```

Finished 消息是第一个被新协商的密钥保护的消息。它包含了所有之前握手消息的哈希，用于验证密钥协商的正确性和防止降级攻击。

---

## 25.3 TLS 1.3 改进

### 25.3.1 TLS 1.3 vs TLS 1.2 核心变化

<div data-component="TLSKDFpipeline"></div>

TLS 1.3 带来了重大改进：

1. **1-RTT 握手**：将握手减少到一个往返
2. **0-RTT 恢复**：支持提前发送数据
3. **移除不安全算法**：删除 RSA 密钥交换、静态 DH、CBC 模式密码等
4. **仅保留 (EC)DHE**：强制使用前向保密
5. **新的密钥派生**：使用 HKDF 代替 PRF
6. **PSK 模式**：支持预共享密钥恢复会话

```
TLS 1.2 密码套件 (已移除的标记 ✗):
  ✗ TLS_RSA_WITH_AES_128_CBC_SHA          (无前向保密)
  ✗ TLS_DHE_RSA_WITH_AES_128_CBC_SHA      (静态 DH)
  ✓ TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256 (保留)

TLS 1.3 密码套件 (简化):
  ✓ TLS_AES_128_GCM_SHA256
  ✓ TLS_AES_256_GCM_SHA384
  ✓ TLS_CHACHA20_POLY1305_SHA256
  (密钥交换和签名算法通过扩展协商)
```

### 25.3.2 TLS 1.3 的 1-RTT 握手

```
客户端 (Client)                                服务器 (Server)
    │                                              │
    │  ──── ClientHello ─────────────────────────> │
    │       + supported_groups (x25519, P-256)      │
    │       + key_share (客户端密钥共享)              │
    │       + signature_algorithms                   │
    │       + supported_versions = 0x0304            │
    │       + psk_key_exchange_modes                 │
    │                                              │
    │  <──── ServerHello ─────────────────────────  │
    │       + key_share (服务器密钥共享)              │
    │       + supported_versions = 0x0304            │
    │  <──── {EncryptedExtensions} ───────────────  │
    │  <──── {Certificate} ───────────────────────  │
    │  <──── {CertificateVerify} ────────────────  │
    │  <──── {Finished} ─────────────────────────  │
    │                                              │
    │  ──── {Finished} ──────────────────────────> │
    │  ──── {Application Data} ──────────────────> │
    │                                              │
    │  ◄════ {应用数据（加密传输）} ═══════════════> │

    注: {} 表示使用握手流量密钥加密
```

**关键改进**：
- ClientHello 中直接附带 `key_share` 扩展（包含客户端的 DH 公钥）
- 服务器在 ServerHello 中返回自己的 DH 公钥
- 此后的所有消息（包括证书）都使用握手流量密钥加密
- 整个握手只需 **1-RTT**

### 25.3.3 0-RTT 恢复

TLS 1.3 支持客户端在首次发送 ClientHello 时就附带应用数据（0-RTT 数据），前提是客户端之前与该服务器建立过连接并缓存了 PSK：

```
客户端 (Client)                                服务器 (Server)
    │                                              │
    │  ──── ClientHello ─────────────────────────> │
    │       + pre_shared_key (PSK identity)         │
    │       + early_data (0-RTT)                     │
    │  ──── {0-RTT Application Data} ────────────> │
    │                                              │
    │  <──── ServerHello ─────────────────────────  │
    │       + pre_shared_key (选定 PSK)              │
    │  <──── {EncryptedExtensions} ───────────────  │
    │  <──── {Finished} ─────────────────────────  │
    │                                              │
    │  ──── {Finished} ──────────────────────────> │
    │  ──── {Application Data} ──────────────────> │
```

**0-RTT 的安全限制**：
- 0-RTT 数据 **没有前向保密**
- 0-RTT 数据可能被 **重放攻击**，因此不应携带非幂等的操作
- 服务器必须实现重放检测机制（如单次使用令牌）

### 25.3.4 HKDF 密钥派生

TLS 1.3 使用 HKDF（HMAC-based Key Derivation Function）取代了 TLS 1.2 的 PRF：

```
HKDF 由两个阶段组成:

阶段一: HKDF-Extract（提取）
  HKDF-Extract(salt, IKM) = HMAC-Hash(salt, IKM)
  
  salt: 由上下文决定（可以是全零或前一阶段的输出）
  IKM (Input Keying Material): 输入密钥材料
  输出: PRK (Pseudorandom Key, 32 bytes for SHA-256)

阶段二: HKDF-Expand（扩展）
  HKDF-Expand(PRK, info, L) = T(1) || T(2) || ... || T(N)
  其中:
    T(0) = 空字符串
    T(i) = HMAC-Hash(PRK, T(i-1) || info || i)
    L = 所需输出长度
    info = 上下文标签 + 手写数据
```

**TLS 1.3 的密钥派生树**：

```
PSK / Early Secret
    │
    ├── HKDF-Extract(0, PSK)
    │   → Early Secret
    │
    ├── Derive-Secret(Early Secret, "c e traffic", ClientHello)
    │   → client_early_traffic_secret (用于加密 0-RTT 数据)
    │
    └── Derive-Secret(Early Secret, "derived", "")
        → Handshake Secret (via HKDF-Extract with DH shared secret)
        │
        ├── Derive-Secret(Handshake Secret, "c hs traffic", CH..SH)
        │   → client_handshake_traffic_secret
        │
        ├── Derive-Secret(Handshake Secret, "s hs traffic", CH..SH)
        │   → server_handshake_traffic_secret
        │
        └── Derive-Secret(Handshake Secret, "derived", "")
            → Master Secret (via HKDF-Extract with 0)
            │
            ├── Derive-Secret(Master Secret, "c ap traffic", ...)
            │   → client_application_traffic_secret_0
            │
            ├── Derive-Secret(Master Secret, "s ap traffic", ...)
            │   → server_application_traffic_secret_0
            │
            └── Derive-Secret(Master Secret, "res master", ...)
                → resumption_master_secret (用于 PSK 恢复)
```

### 25.3.5 PSK 模式详解

PSK（Pre-Shared Key）模式在 TLS 1.3 中有两种用途：

1. **PSK-only 模式**：仅使用 PSK 进行认证（不需要证书）
2. **PSK with (EC)DHE**：PSK 提供认证，DH 提供前向保密（更安全）

```
PSK 模式密码套件 (psk_key_exchange_modes):
  - psk_ke: 仅 PSK (无前向保密)
  - psk_dhe_ke: PSK + (EC)DHE (推荐，有前向保密)
```

---

## 25.4 TLS 记录协议

### 25.4.1 记录协议概述

TLS 记录协议负责将上层数据封装为 TLS 记录，包括分片、压缩（TLS 1.3 中已移除）、加密和认证。

```
TLS 记录格式 (TLS 1.2):
┌──────────────────────────────────────────────────────┐
│ content_type (1 byte)                                │
│   0x16 = handshake, 0x17 = application_data          │
│   0x15 = alert, 0x14 = change_cipher_spec            │
│ legacy_version (2 bytes) = 0x0303                    │
│ length (2 bytes) = 2^14 (最大 16384 bytes)           │
├──────────────────────────────────────────────────────┤
│ fragment (加密后的数据)                                │
│   ┌──────────────────────────────────────────┐       │
│   │ Content (原始数据，分片后)                  │       │
│   │ MAC (20 bytes, SHA-1) 或 AEAD tag         │       │
│   │ Padding (块加密时需要)                      │       │
│   └──────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────┘

TLS 记录格式 (TLS 1.3):
┌──────────────────────────────────────────────────────┐
│ content_type (1 byte) = 0x17 (伪装为 application_data)│
│ legacy_version (2 bytes) = 0x0303                    │
│ length (2 bytes)                                      │
├──────────────────────────────────────────────────────┤
│ encrypted_data (包含真实 content_type)                 │
│   ┌──────────────────────────────────────────┐       │
│   │ Content (原始数据)                         │       │
│   │ 真实 content_type (1 byte)                │       │
│   │ AEAD tag (16 bytes)                       │       │
│   └──────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────┘
```

### 25.4.2 分片过程

```
原始应用数据 (可能很大):
[数据块 1][数据块 2][数据块 3]...[数据块 N]

分片后 (每片最大 2^14 = 16384 bytes):
记录 1: [数据块 1...数据块 K] (≤16384 bytes)
记录 2: [数据块 K+1...数据块 M] (≤16384 bytes)
...
记录 R: [数据块 L+1...数据块 N] (≤16384 bytes)
```

### 25.4.3 AEAD 加密

TLS 1.2 和 1.3 推荐使用 AEAD（Authenticated Encryption with Associated Data）加密，如 AES-GCM 或 ChaCha20-Poly1305：

```
AES-GCM 加密过程:
┌─────────────────────────────────────────────────────┐
│                                                     │
│  nonce (12 bytes) = IV ⊥ seq_num                    │
│  (对于 TLS 1.2: IV + sequence_number)                │
│  (对于 TLS 1.3: IV ⊕ sequence_number)                │
│                                                     │
│  additional_data =                                  │
│    content_type || version || length                 │
│                                                     │
│  plaintext = content || content_type (TLS 1.3)      │
│                                                     │
│  ciphertext, tag = AES-GCM(key, nonce,              │
│                            additional_data,          │
│                            plaintext)               │
│                                                     │
│  输出: ciphertext (variable) || tag (16 bytes)       │
└─────────────────────────────────────────────────────┘
```

### 25.4.4 MAC 计算 (非 AEAD 套件)

对于非 AEAD 套件（如 CBC 模式），TLS 使用显式 MAC：

```
MAC = HMAC_hash(MAC_key,
    seq_num ||         # 8 bytes 序列号
    content_type ||    # 1 byte
    version ||         # 2 bytes
    length ||          # 2 bytes
    fragment           # 实际数据
)

HMAC-SHA1 输出: 20 bytes
HMAC-SHA256 输出: 32 bytes
```

---

## 25.5 IPsec 概述

IPsec（IP Security）是一组协议套件，在网络层为 IP 数据包提供安全保护。与 TLS 不同，IPsec 工作在 IP 层，可以保护任何基于 IP 的通信。

### 25.5.1 IPsec 协议族

```
IPsec 协议栈:
┌──────────────────────────────────────────────────┐
│  IKEv2 (Internet Key Exchange)                    │
│  - SA 协商与密钥交换                               │
│  - 自动密钥管理                                    │
├──────────────────────────────────────────────────┤
│  AH (Authentication Header)                       │
│  - 数据完整性认证                                  │
│  - 防重放保护                                      │
│  - 不提供加密                                      │
├──────────────────────────────────────────────────┤
│  ESP (Encapsulating Security Payload)             │
│  - 数据加密 (机密性)                               │
│  - 数据完整性认证                                  │
│  - 防重放保护                                      │
└──────────────────────────────────────────────────┘
```

### 25.5.2 AH 协议头格式

```
AH 头 (RFC 4302):
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
│ Next Header   │  Payload Len  │          Reserved             │
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
│                 Security Parameters Index (SPI)                │
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
│                    Sequence Number                             │
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
│                                                               │
│                Integrity Check Value (ICV)                     │
│                    (variable length)                           │
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

字段说明:
- Next Header (8 bits): 被保护的上层协议类型 (如 TCP=6, UDP=17)
- Payload Len (8 bits): AH 头长度 (以 4 字节为单位减 2)
- SPI (32 bits): 安全参数索引，用于查找安全关联
- Sequence Number (32 bits): 单调递增计数器，防重放
- ICV (variable): 完整性校验值，覆盖整个 IP 包（除可变字段）
```

### 25.5.3 ESP 协议头格式

```
ESP 头和尾 (RFC 4303):
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
│               Security Parameters Index (SPI)                 │
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
│                    Sequence Number                             │
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                    IV (Initialization Vector)                  |
|                         (variable, 8-16 bytes)                 |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
│                                                               |
~                 Encrypted Payload (variable)                   ~
│                                                               |
+   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+   +-----------+
|   |     Padding (0-255 bytes)                 |   | Pad Length  |
+   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+   +-----------+
|   | Next Header                               |   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
~              Integrity Check Value (ICV)                      ~
|                    (variable, 12-16 bytes)                    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

字段说明:
- SPI (32 bits): 安全参数索引
- Sequence Number (32 bits): 防重放计数器
- IV: 初始化向量，用于加密
- Encrypted Payload: 加密的原始数据
- Padding: 填充到块大小对齐
- Pad Length (8 bits): 填充字节数
- Next Header (8 bits): 原始上层协议类型
- ICV: 完整性校验值（覆盖 SPI 到 ICV 之前的所有数据）
```

### 25.5.4 传输模式 vs 隧道模式

<div data-component="IPsecSAengine"></div>

**传输模式**：只加密/认证 IP 包的载荷，IP 头不变。适用于主机到主机的通信。

```
传输模式 - ESP 加密前:
┌──────────┬─────────────┬───────────────┐
│ 原 IP 头  │   TCP 头     │    应用数据    │
└──────────┴─────────────┴───────────────┘

传输模式 - ESP 加密后:
┌──────────┬──────────┬──────┬───────────────┬──────┬──────────┐
│ 原 IP 头  │ ESP 头    │  IV  │ 加密的 TCP+数据│ ESP尾│ ICV      │
└──────────┴──────────┴──────┴───────────────┴──────┴──────────┘
```

**隧道模式**：加密整个原始 IP 包，并添加新的 IP 头。适用于 VPN 网关间的通信。

```
隧道模式 - ESP 加密前:
┌──────────┬─────────────┬───────────────┐
│ 原 IP 头  │   TCP 头     │    应用数据    │
└──────────┴─────────────┴───────────────┘

隧道模式 - ESP 加密后:
┌──────────┬──────────┬──────┬──────────────────────────┬──────┬──────┐
│ 新 IP 头  │ ESP 头    │  IV  │ 加密的(原IP头+TCP+数据)   │ ESP尾│ ICV  │
└──────────┴──────────┴──────┴──────────────────────────┴──────┴──────┘
  ↑                            ↑
  外部头 (传输路径)              整个原始包被加密保护
```

**对比总结**：

| 特性 | 传输模式 | 隧道模式 |
|------|----------|----------|
| 加密范围 | 仅载荷 | 整个原始包 |
| IP 头 | 原始 IP 头可见 | 添加新 IP 头，原始头被加密 |
| 典型应用 | 主机到主机 | VPN 网关间 |
| 额外开销 | 较小 | 较大（新 IP 头 20-40 bytes） |
| 源/目的地址保护 | 不保护 | 保护（隐藏在加密部分） |

---

## 25.6 IKEv2 协议

IKEv2（Internet Key Exchange version 2）是 IPsec 的密钥管理协议，负责自动协商安全关联（SA）和生成密钥材料。

### 25.6.1 IKEv2 消息交换

IKEv2 的 SA 建立分为两个阶段：

**阶段一：IKE_SA_INIT（2 条消息）**

```
发起方 (Initiator)                    响应方 (Responder)
    │                                      │
    │ ──── Message 1 ────────────────────> │
    │  HDR, SAi1, KEi, Ni                  │
    │  (IKE SA 提议, DH 公钥, Nonce)        │
    │                                      │
    │ <──── Message 2 ─────────────────── │
    │  HDR, SAr1, KEr, Nr                  │
    │  (选定 SA, DH 公钥, Nonce)            │
    │                                      │
    │ 此时双方已计算出:                       │
    │ SKEYSEED = PRF(Ni || Nr, g^ir)       │
    │ {SK_d, SK_ai, SK_ar, SK_ei, SK_er,   │
    │  SK_pi, SK_pr} = PRF+(SKEYSEED,      │
    │                       Ni||Nr||SPIi||SPIr)│
```

**阶段二：IKE_AUTH（2 条消息，加密传输）**

```
    │ ──── Message 3 ────────────────────> │
    │  HDR, SK{IDi, AUTH, SAi2, TSi, TSr,  │
    │          CP, N, [CERT]}               │
    │  (身份, 认证数据, CHILD_SA 提议,       │
    │   流量选择器, 配置请求, 证书)            │
    │                                      │
    │ <──── Message 4 ─────────────────── │
    │  HDR, SK{IDr, AUTH, SAr2, TSi, TSr,  │
    │          CP, N, [CERT]}               │
    │  (身份, 认证数据, 选定 CHILD_SA,       │
    │   流量选择器, 配置回复, 证书)            │
```

### 25.6.2 SA 协商结构

```
SA 载荷结构:
┌─────────────────────────────────────────────────┐
│ SA 载荷头 (4 bytes)                               │
├─────────────────────────────────────────────────┤
│ Proposal 1:                                      │
│   ├── Protocol ID = IKE (1) / AH (2) / ESP (3)  │
│   ├── SPI (0 for IKE, 4 bytes for AH/ESP)       │
│   ├── Transform 1: ENCR (加密算法)               │
│   │   └── AES-256-GCM (ENCR_AES_GCM_16)        │
│   ├── Transform 2: PRF (伪随机函数)              │
│   │   └── HMAC-SHA256                            │
│   ├── Transform 3: INTEG (完整性算法)            │
│   │   └── HMAC-SHA256-128                        │
│   └── Transform 4: DH (密钥交换组)               │
│       └── ECP-256 (group 19)                     │
├─────────────────────────────────────────────────┤
│ Proposal 2 (备选):                               │
│   └── ...                                        │
└─────────────────────────────────────────────────┘
```

### 25.6.3 CHILD_SA 建立

IKE_AUTH 完成后，已建立一个 CHILD_SA。后续可通过 CREATE_CHILD_SA 交换创建额外的 CHILD_SA：

```
CREATE_CHILD_SA 交换:
    │ ──── Message ─────────────────────────> │
    │  HDR, SK{SA, Ni, KEi, [N(NONCE)]}       │
    │  (新 CHILD_SA 提议, 新 Nonce, 新 DH)      │
    │                                         │
    │ <──── Message ──────────────────────── │
    │  HDR, SK{SA, Nr, KEr, [N(NONCE)]}       │
    │  (选定 SA, 新 Nonce, 新 DH)               │
```

---

## 25.7 SSL VPN vs IPsec VPN

### 25.7.1 概述对比

| 特性 | SSL VPN | IPsec VPN |
|------|---------|-----------|
| 工作层 | 传输层/应用层 (TLS) | 网络层 (IPsec) |
| 客户端 | 浏览器或轻量客户端 | 需要专用客户端软件 |
| 访问粒度 | 应用级别 | 全网络/子网级别 |
| NAT 穿越 | 天然支持 (TCP/443) | 需要 NAT-T 封装 |
| 防火墙穿越 | 通常无问题 (HTTPS 流量) | 可能被防火墙阻止 |
| 性能 | 加密在应用层，开销稍大 | 硬件加速支持好 |
| 部署难度 | 简单，无需客户端配置 | 较复杂，需要安全策略配置 |
| 安全性 | 依赖浏览器安全 | 协议层安全，可控性强 |

### 25.7.2 SSL VPN 的两种模式

```
1. 无客户端模式 (Clientless / Web VPN):
   浏览器 ──── HTTPS ────> SSL VPN 网关 ────> 内网应用
   (通过浏览器访问 Web 应用，无需安装客户端)

2. 全隧道模式 (Full Tunnel):
   VPN 客户端 ──── SSL/TLS ────> SSL VPN 网关 ────> 内网
   (所有流量通过 VPN 隧道，类似 IPsec VPN)
```

### 25.7.3 IPsec VPN 的典型部署

```
站点到站点 (Site-to-Site):
  总部网络 ── IPsec VPN 网关 ══════ 互联网 ══════ IPsec VPN 网关 ── 分支网络
  (192.168.1.0/24)                                                    (192.168.2.0/24)

远程访问 (Remote Access):
  远程用户 ── IPsec 客户端 ══════ 互联网 ══════ IPsec VPN 网关 ── 企业内网
```

---

## 25.8 防火墙类型

### 25.8.1 防火墙演进

```
防火墙技术演进:
  1980s ── 包过滤防火墙 (Packet Filtering)
  1990s ── 状态检测防火墙 (Stateful Inspection)
  2000s ── 应用代理防火墙 (Application Proxy)
  2010s ── 下一代防火墙 (NGFW)
```

### 25.8.2 包过滤防火墙

包过滤防火墙基于 IP/TCP/UDP 头部信息对每个数据包独立做出允许/拒绝决策：

```
包过滤规则示例:
┌─────────┬───────────┬─────────┬─────────┬─────────┬────────┐
│ 规则 #   │ 源 IP      │ 目的 IP  │ 源端口   │ 目的端口 │ 动作    │
├─────────┼───────────┼─────────┼─────────┼─────────┼────────┤
│ 1       │ 10.0.0.0/8│ any     │ any     │ 80      │ ALLOW  │
│ 2       │ 10.0.0.0/8│ any     │ any     │ 443     │ ALLOW  │
│ 3       │ any       │ 10.0.0.1│ any     │ 22      │ ALLOW  │
│ 4       │ any       │ any     │ any     │ any     │ DENY   │
└─────────┴───────────┴─────────┴─────────┴─────────┴────────┘
```

**优点**：速度快、实现简单
**缺点**：
- 无状态检查，无法跟踪连接
- 无法检查应用层内容
- 难以处理动态端口（如 FTP 的 PASV 模式）

### 25.8.3 状态检测防火墙

<div data-component="FirewallTCAMengine"></div>

状态检测防火墙维护一个 **连接状态表**，跟踪每个连接的状态：

```
连接状态表 (State Table):
┌───────────────────┬─────────────┬───────────┬─────────┬─────────┬──────────┐
│ 协议               │ 源 IP:端口   │ 目的 IP:端口│ 状态     │ 超时     │ 标志位    │
├───────────────────┼─────────────┼───────────┼─────────┼─────────┼──────────┤
│ TCP               │ 10.0.0.5:34567│ 93.184.216.34:443│ ESTABLISHED│ 3600s│ SYN/ACK  │
│ TCP               │ 10.0.0.8:54321│ 151.101.1.69:80  │ SYN_SENT │ 60s   │ SYN      │
│ UDP               │ 10.0.0.5:5353│ 224.0.0.251:5353 │ ACTIVE   │ 120s  │          │
└───────────────────┴─────────────┴───────────┴─────────┴─────────┴──────────┘

TCP 状态机跟踪:
  CLOSED ──> SYN_SENT ──> ESTABLISHED ──> FIN_WAIT_1 ──> FIN_WAIT_2 ──> TIME_WAIT
     ↑              │            │              │              │              │
     └──────────────┴────────────┴──────────────┴──────────────┴──────────────┘

UDP 连接跟踪:
  - 无状态协议，基于超时机制
  - 收到 UDP 包后创建状态条目，超时后删除
```

**状态检测规则匹配流程**：

```
收到数据包:
    │
    ▼
检查连接状态表 (五元组匹配):
    ├── 命中已有连接 → 检查是否符合当前状态
    │   ├── 符合 → ALLOW (更新状态表)
    │   └── 不符合 → DENY (可能的攻击)
    └── 未命中 → 检查规则表
        ├── 匹配 ALLOW 规则 → 创建新状态条目 → ALLOW
        └── 匹配 DENY 规则或默认 → DENY
```

### 25.8.4 应用代理防火墙

应用代理防火墙在应用层检查并转发数据，客户端与代理建立连接，代理再与服务器建立连接：

```
应用代理架构:
  客户端 <──TCP──> [代理防火墙] <──TCP──> 服务器
         TLS          应用层检查          TLS

代理功能:
  - 解密 TLS (中间人检查)
  - 深度内容检查 (HTTP 头、URL、请求体)
  - 病毒/恶意软件扫描
  - 内容过滤 (URL 黑名单)
  - 日志记录 (完整会话记录)
```

### 25.8.5 下一代防火墙 (NGFW)

NGFW 结合了传统防火墙功能与深度包检测（DPI）、应用识别等高级功能：

```
NGFW 功能层次:
┌─────────────────────────────────────────────────┐
│ 应用层识别与控制                                   │
│   - DPI 深度包检测                                 │
│   - 应用指纹识别 (不只是端口)                       │
│   - 用户身份集成 (LDAP/AD)                         │
├─────────────────────────────────────────────────┤
│ 入侵防御系统 (IPS)                                │
│   - 签名检测                                      │
│   - 异常检测                                      │
│   - 虚拟补丁                                      │
├─────────────────────────────────────────────────┤
│ 威胁情报集成                                      │
│   - 恶意 IP/域名黑名单                             │
│   - 沙箱分析                                      │
│   - 云威胁情报                                    │
├─────────────────────────────────────────────────┤
│ 传统状态检测防火墙                                  │
│   - 五元组规则                                     │
│   - 连接状态跟踪                                   │
│   - NAT                                          │
└─────────────────────────────────────────────────┘
```

---

## 25.9 IDS/IPS

### 25.9.1 IDS vs IPS

```
IDS (Intrusion Detection System) - 入侵检测系统:
  流量 ────> 镜像端口 ────> IDS 分析引擎 ────> 告警
  (旁路部署，不阻断流量)

IPS (Intrusion Prevention System) - 入侵防御系统:
  流量 ────> IPS 分析引擎 ────> 正常流量 ────> 目的
  (在线部署，可阻断恶意流量)
```

| 特性 | IDS | IPS |
|------|-----|-----|
| 部署方式 | 旁路 (镜像端口) | 在线 (串联) |
| 功能 | 检测 + 告警 | 检测 + 阻断 |
| 延迟 | 无影响 | 增加处理延迟 |
| 误报影响 | 仅告警 | 可能阻断合法流量 |
| 典型工具 | Snort (IDS 模式) | Snort (IPS 模式), Suricata |

### 25.9.2 检测方法

**基于签名的检测**：

```
签名 (Signature) 示例:
  - 已知攻击模式的特征码
  - 特定的字节序列
  - 特定的协议行为

优点:
  - 误报率低
  - 检测已知攻击效率高

缺点:
  - 无法检测零日攻击 (0-day)
  - 签名库需要持续更新
  - 签名数量增长导致性能下降
```

**基于异常的检测**：

```
异常检测流程:
  1. 建立正常流量基线 (Baseline)
     - 连接频率、数据量、协议分布
     - 用户行为模式
  2. 实时监测偏离基线的行为
  3. 偏离超过阈值时触发告警

优点:
  - 可检测零日攻击
  - 不依赖已知签名

缺点:
  - 误报率较高
  - 基线建立需要时间
  - 正常行为变化可能导致误报
```

### 25.9.3 深度包检测 (DPI) 引擎

<div data-component="IDSDPIengine"></div>

DPI 引擎是 IDS/IPS 的核心，负责对数据包内容进行深度分析：

```
DPI 引擎架构:
┌─────────────────────────────────────────────────────┐
│                    输入数据包                         │
│                        │                            │
│                        ▼                            │
│  ┌──────────────────────────────────────────────┐  │
│  │            协议解析器 (Protocol Parser)        │  │
│  │  - 解析各层协议头                               │  │
│  │  - 重组 TCP 流                                 │  │
│  │  - 提取应用层载荷                               │  │
│  └──────────────────────────────────────────────┘  │
│                        │                            │
│                        ▼                            │
│  ┌──────────────────────────────────────────────┐  │
│  │         模式匹配引擎 (Pattern Matching)        │  │
│  │  - AC 自动机 (Aho-Corasick)                   │  │
│  │  - 正则表达式引擎                              │  │
│  │  - 多模式同时匹配                              │  │
│  └──────────────────────────────────────────────┘  │
│                        │                            │
│                        ▼                            │
│  ┌──────────────────────────────────────────────┐  │
│  │            规则匹配与告警生成                   │  │
│  │  - 规则优先级评估                              │  │
│  │  - 告警阈值检查                                │  │
│  │  - 响应动作 (告警/阻断/记录)                   │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

**AC 自动机（Aho-Corasick）算法**：

AC 自动机是多模式字符串匹配的经典算法，用于在文本中同时查找多个模式串：

```
构建 AC 自动机:
  1. 将所有规则签名构建成 Trie 树
  2. 为每个节点计算失败指针 (Failure Pointer)
  3. 每个节点记录匹配的规则列表

匹配过程:
  输入: "GET /admin/config.php HTTP/1.1"
  
  模式库:
    P1: "/admin/"
    P2: "config.php"
    P3: "cmd.exe"
    P4: "/etc/passwd"
  
  匹配结果: 命中 P1 和 P2 → 触发相关规则
```

### 25.9.4 Snort 规则格式

Snort 是最流行的开源 IDS/IPS，其规则格式如下：

```
Snort 规则基本结构:
  action source_ip source_port -> dest_ip dest_port (rule_options)

规则示例:

# 检测 SQL 注入攻击
alert tcp $EXTERNAL_NET any -> $HTTP_SERVERS $HTTP_PORTS (
    msg:"SQL Injection Attempt - UNION SELECT";
    flow:established,to_server;
    content:"UNION";
    nocase;
    content:"SELECT";
    nocase;
    distance:0;
    classtype:web-application-attack;
    sid:1000001;
    rev:1;
)

# 检测 Shellcode
alert tcp $EXTERNAL_NET any -> $HOME_NET any (
    msg:"Shellcode Detected - NOP Sled";
    content:"|90 90 90 90 90 90 90 90 90 90 90 90|";
    classtype:shellcode-detect;
    sid:1000002;
    rev:1;
)

# 检测端口扫描
alert icmp $EXTERNAL_NET any -> $HOME_NET any (
    msg:"ICMP Port Scan Detected";
    icode:0;
    itype:8;
    threshold:type both,track by_src,count 50,seconds 60;
    classtype:attempted-recon;
    sid:1000003;
    rev:1;
)
```

**Snort 规则选项说明**：

| 选项 | 说明 |
|------|------|
| `msg` | 告警消息 |
| `flow` | TCP 流方向 (`to_server`, `to_client`, `established`) |
| `content` | 匹配特定字节序列 |
| `nocase` | 大小写不敏感匹配 |
| `pcre` | 正则表达式匹配 |
| `depth` / `offset` | 限制搜索范围 |
| `threshold` | 告警阈值 (防洪泛) |
| `sid` | 规则 ID |
| `rev` | 规则版本号 |
| `classtype` | 规则分类 |

### 25.9.5 IDS/IPS 部署模式

```
部署模式:

1. 旁路模式 (Passive / IDS):
   互联网 ────> 核心交换机 ────> 内部网络
                    │ (镜像/SPAN)
                    ▼
                IDS 传感器 ────> 管理控制台

2. 在线模式 (Inline / IPS):
   互联网 ────> IPS 传感器 ────> 核心交换机 ────> 内部网络
               (可阻断流量)

3. 混合模式:
   互联网 ────> IPS (关键链路) ────> 核心交换机 ────> 内部网络
                    │ (SPAN)
                    ▼
               IDS (旁路分析)
```

---

## 25.10 网络准入控制 (NAC / 802.1X)

### 25.10.1 802.1X 概述

802.1X 是 IEEE 定义的基于端口的网络访问控制标准，用于在设备接入网络前进行身份认证和安全检查。

```
802.1X 三个角色:
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  请求者        │    │  认证者        │    │  认证服务器    │
│  (Supplicant) │    │  (Authenticator)│   │  (Authentication│
│              │    │              │    │   Server)     │
│  客户端设备    │    │  交换机/AP     │    │  RADIUS 服务器 │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       │   EAPoL          │    RADIUS          │
       │  (二层帧)         │   (UDP 1812/1813)  │
       ├──────────────────>│                   │
       │                   ├──────────────────>│
       │                   │                   │
       │                   │<──────────────────┤
       │<──────────────────┤                   │
       │                   │                   │
```

### 25.10.2 802.1X 认证流程

```
802.1X 认证步骤:

1. 端口初始化:
   交换机端口设为 "未授权" 状态，只允许 EAPoL 帧通过

2. EAP 协商:
   请求者 ──── EAPoL-Start ────> 认证者
   认证者 ──── EAP-Request/Identity ────> 请求者
   请求者 ──── EAP-Response/Identity ────> 认证者
        (包含用户名)

3. RADIUS 转发:
   认证者 ──── RADIUS Access-Request ────> RADIUS 服务器
   (封装 EAP 消息)

4. EAP 方法执行:
   常见 EAP 方法:
   - EAP-TLS: 证书认证 (最安全)
   - EAP-PEAP: TLS 隧道内的密码认证
   - EAP-TTLS: 类似 PEAP，更灵活
   - EAP-MD5: 简单密码 (不推荐)

5. 认证结果:
   RADIUS 服务器 ──── RADIUS Access-Accept ────> 认证者
   (包含 VLAN 分配、ACL 等属性)
   认证者 ──── 端口设为 "已授权" ────> 允许网络访问
```

### 25.10.3 NAC 完整流程

NAC 不仅认证身份，还检查设备安全状态：

```
NAC 检查流程:
┌─────────────────────────────────────────────────────────┐
│ 1. 设备接入网络                                           │
│    └── 触发 802.1X 认证                                   │
│                                                         │
│ 2. 身份认证 (802.1X/EAP)                                │
│    ├── 用户名/密码                                        │
│    ├── 数字证书                                          │
│    └── 或其他凭证                                        │
│                                                         │
│ 3. 安全状态评估                                          │
│    ├── 操作系统版本/补丁状态                               │
│    ├── 防病毒软件状态                                     │
│    ├── 防火墙启用状态                                     │
│    └── 设备合规性检查                                     │
│                                                         │
│ 4. 策略执行                                              │
│    ├── 完全合规 → 分配到生产 VLAN                         │
│    ├── 部分合规 → 分配到修复 VLAN                         │
│    │   (只能访问补丁服务器、AV 更新服务器)                  │
│    └── 不合规/未认证 → 分配到隔离 VLAN 或拒绝接入          │
└─────────────────────────────────────────────────────────┘
```

---

## 25.11 代码示例

### 25.11.1 Python TLS 客户端示例

```python
import ssl
import socket

def tls_client(host, port=443):
    context = ssl.create_default_context()
    
    with socket.create_connection((host, port)) as sock:
        with context.wrap_socket(sock, server_hostname=host) as ssock:
            print(f"TLS 版本: {ssock.version()}")
            print(f"密码套件: {ssock.cipher()}")
            print(f"服务器证书:")
            cert = ssock.getpeercert()
            for field in cert.get('subject', ()):
                for key, value in field:
                    print(f"  {key}: {value}")
            
            ssock.sendall(f"GET / HTTP/1.1\r\nHost: {host}\r\n\r\n".encode())
            response = ssock.recv(4096)
            print(f"\n响应前 200 字节:\n{response[:200].decode(errors='replace')}")

tls_client("www.example.com")
```

### 25.11.2 Python IPsec 配置示例

```python
import subprocess
import json

class IPsecConfig:
    """IPsec 配置管理器 - 生成 strongSwan 配置"""
    
    def __init__(self, local_ip, remote_ip, local_subnet, remote_subnet):
        self.local_ip = local_ip
        self.remote_ip = remote_ip
        self.local_subnet = local_subnet
        self.remote_subnet = remote_subnet
    
    def generate_ipsec_conf(self):
        config = f"""# IPsec 连接配置
config setup
    charondebug="ike 2, knl 2, cfg 2"

conn my-vpn
    type=tunnel
    left={self.local_ip}
    leftsubnet={self.local_subnet}
    leftfirewall=yes
    right={self.remote_ip}
    rightsubnet={self.remote_subnet}
    ike=aes256-sha256-modp2048!
    esp=aes256-sha256!
    keyexchange=ikev2
    ikelifetime=8h
    lifetime=1h
    dpddelay=30s
    dpdtimeout=120s
    dpdaction=restart
    authby=secret
    auto=start
"""
        return config
    
    def generate_secrets_conf(self):
        return f"""# IPsec 预共享密钥
{self.local_ip} {self.remote_ip} : PSK "YourStrongPreSharedKey123!"
"""
    
    def apply_config(self, dry_run=True):
        ipsec_conf = self.generate_ipsec_conf()
        secrets_conf = self.generate_secrets_conf()
        
        if dry_run:
            print("=== ipsec.conf ===")
            print(ipsec_conf)
            print("=== ipsec.secrets ===")
            print(secrets_conf)
        else:
            with open("/etc/ipsec.conf", "w") as f:
                f.write(ipsec_conf)
            with open("/etc/ipsec.secrets", "w") as f:
                f.write(secrets_conf)
            subprocess.run(["ipsec", "restart"], check=True)

config = IPsecConfig(
    local_ip="203.0.113.1",
    remote_ip="198.51.100.1",
    local_subnet="192.168.1.0/24",
    remote_subnet="192.168.2.0/24"
)
config.apply_config(dry_run=True)
```

### 25.11.3 Python 数据包构造示例

```python
import struct
import socket

def craft_ipsec_esp_packet(src_ip, dst_ip, spi, seq_num, payload):
    """构造 IPsec ESP 封装的数据包（仅结构演示，非实际加密）"""
    
    # IP 头 (20 bytes)
    ip_header = struct.pack(
        '!BBHHHBBH4s4s',
        0x45,           # Version(4) + IHL(5)
        0,              # DSCP + ECN
        20 + 8 + len(payload) + 16,  # Total Length
        0,              # Identification
        0x4000,         # Flags (Don't Fragment) + Fragment Offset
        64,             # TTL
        50,             # Protocol = ESP (50)
        0,              # Header Checksum (0 for now)
        socket.inet_aton(src_ip),  # Source IP
        socket.inet_aton(dst_ip)   # Destination IP
    )
    
    # ESP 头 (8 bytes)
    esp_header = struct.pack('!II', spi, seq_num)
    
    # ESP 尾 (模拟)
    pad_len = (4 - (len(payload) % 4)) % 4
    padding = b'\x00' * pad_len
    esp_trailer = struct.pack('!BB', pad_len, 50)  # Pad Length + Next Header (50=ESP)
    
    # ICV (模拟, 16 bytes)
    icv = b'\x00' * 16
    
    packet = ip_header + esp_header + payload + padding + esp_trailer + icv
    return packet

packet = craft_ipsec_esp_packet(
    src_ip="192.168.1.1",
    dst_ip="192.168.2.1",
    spi=0x12345678,
    seq_num=1,
    payload=b"Hello IPsec!"
)
print(f"构造的数据包大小: {len(packet)} bytes")
print(f"十六进制: {packet.hex()}")
```

### 25.11.4 Bash iptables 规则示例

```bash
#!/bin/bash
# iptables 防火墙规则配置示例

# 清除现有规则
iptables -F
iptables -X
iptables -t nat -F

# 默认策略: 拒绝所有入站，允许所有出站
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# 允许回环接口
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# 允许已建立的连接和相关连接 (状态检测)
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A FORWARD -m state --state ESTABLISHED,RELATED -j ACCEPT

# 允许 SSH 访问 (仅限管理网段)
iptables -A INPUT -p tcp -s 10.0.0.0/24 --dport 22 \
    -m state --state NEW -j ACCEPT

# 允许 HTTP/HTTPS 访问
iptables -A INPUT -p tcp --dport 80 \
    -m state --state NEW -j ACCEPT
iptables -A INPUT -p tcp --dport 443 \
    -m state --state NEW -j ACCEPT

# 允许 IPsec (IKE + ESP)
iptables -A INPUT -p udp --dport 500 -j ACCEPT    # IKE
iptables -A INPUT -p udp --dport 4500 -j ACCEPT   # NAT-T
iptables -A INPUT -p 50 -j ACCEPT                  # ESP

# 防 SYN 洪泛攻击
iptables -A INPUT -p tcp --syn -m limit \
    --limit 10/second --limit-burst 20 -j ACCEPT
iptables -A INPUT -p tcp --syn -j DROP

# 防 ICMP 洪泛
iptables -A INPUT -p icmp --icmp-type echo-request \
    -m limit --limit 1/second --limit-burst 4 -j ACCEPT
iptables -A INPUT -p icmp --icmp-type echo-request -j DROP

# 记录并丢弃其他流量
iptables -A INPUT -j LOG --log-prefix "IPTables-Dropped: " \
    --log-level 4
iptables -A INPUT -j DROP

# NAT 配置 (内网访问互联网)
iptables -t nat -A POSTROUTING -s 192.168.1.0/24 \
    -o eth0 -j MASQUERADE

# 端口转发 (将外部 8443 转发到内部 Web 服务器)
iptables -t nat -A PREROUTING -i eth0 -p tcp --dport 8443 \
    -j DNAT --to-destination 192.168.1.100:443
iptables -A FORWARD -p tcp -d 192.168.1.100 --dport 443 \
    -m state --state NEW -j ACCEPT

# 保存规则
iptables-save > /etc/iptables/rules.v4

echo "防火墙规则配置完成"
```

### 25.11.5 Bash OpenSSL 命令示例

```bash
#!/bin/bash
# OpenSSL 常用命令示例

# 1. 测试 TLS 连接
echo "=== 测试 TLS 连接 ==="
openssl s_client -connect www.example.com:443 -servername www.example.com \
    -tls1_2 -brief 2>&1 | head -20

# 2. 查看服务器证书
echo -e "\n=== 查看服务器证书 ==="
openssl s_client -connect www.example.com:443 2>/dev/null | \
    openssl x509 -noout -subject -issuer -dates -serial

# 3. 生成 RSA 密钥对和自签名证书
echo -e "\n=== 生成自签名证书 ==="
openssl req -x509 -newkey rsa:2048 -keyout server.key -out server.crt \
    -days 365 -nodes \
    -subj "/C=CN/ST=Beijing/L=Beijing/O=Example/CN=example.com"

# 4. 查看证书详细信息
echo -e "\n=== 证书详细信息 ==="
openssl x509 -in server.crt -noout -text | head -30

# 5. 验证证书链
echo -e "\n=== 验证证书链 ==="
openssl verify -CAfile server.crt server.crt

# 6. 生成 CSR (证书签名请求)
echo -e "\n=== 生成 CSR ==="
openssl req -new -key server.key -out server.csr \
    -subj "/C=CN/ST=Beijing/L=Beijing/O=Example/CN=www.example.com"

# 7. 查看 CSR 内容
echo -e "\n=== CSR 内容 ==="
openssl req -in server.csr -noout -text | head -20

# 8. 计算文件的 SHA256 指纹
echo -e "\n=== 证书指纹 ==="
openssl x509 -in server.crt -noout -fingerprint -sha256

# 9. 测试支持的密码套件
echo -e "\n=== 测试密码套件 ==="
openssl ciphers -v 'ECDHE+AESGCM' | head -5

# 10. 查看 TLS 握手过程详情
echo -e "\n=== TLS 握手详情 ==="
openssl s_client -connect www.example.com:443 -msg -state 2>&1 | \
    grep -E "^(SSL_connect|SSL_accept|Protocol|Cipher|Session-ID)" | head -10
```

### 25.11.6 Python Snort 规则解析器

```python
class SnortRule:
    """简单的 Snort 规则解析器"""
    
    def __init__(self, rule_text):
        self.raw = rule_text.strip()
        self.action = None
        self.protocol = None
        self.src_ip = None
        self.src_port = None
        self.direction = None
        self.dst_ip = None
        self.dst_port = None
        self.options = {}
        self.parse()
    
    def parse(self):
        # 分割规则头部和选项
        if '(' in self.raw:
            header_part = self.raw[:self.raw.index('(')].strip()
            options_part = self.raw[self.raw.index('(')+1:-1]
        else:
            header_part = self.raw
            options_part = ""
        
        # 解析头部
        tokens = header_part.split()
        if len(tokens) >= 7:
            self.action = tokens[0]
            self.protocol = tokens[1]
            self.src_ip = tokens[2]
            self.src_port = tokens[3]
            self.direction = tokens[4]
            self.dst_ip = tokens[5]
            self.dst_port = tokens[6]
        
        # 解析选项
        for opt in options_part.split(';'):
            opt = opt.strip()
            if ':' in opt:
                key = opt[:opt.index(':')].strip()
                value = opt[opt.index(':')+1:].strip()
                self.options[key] = value
    
    def __str__(self):
        return (f"Rule: {self.action} {self.protocol} "
                f"{self.src_ip}:{self.src_port} {self.direction} "
                f"{self.dst_ip}:{self.dst_port}\n"
                f"Options: {self.options}")


# 测试规则解析
rule1 = SnortRule(
    'alert tcp $EXTERNAL_NET any -> $HTTP_SERVERS $HTTP_PORTS '
    '(msg:"SQL Injection Attempt"; flow:established,to_server; '
    'content:"UNION"; nocase; content:"SELECT"; nocase; '
    'sid:1000001; rev:1;)'
)
print(rule1)

rule2 = SnortRule(
    'alert icmp any any -> $HOME_NET any '
    '(msg:"ICMP Ping Detected"; itype:8; icode:0; '
    'sid:1000004; rev:1;)'
)
print(rule2)
```

---

## 25.12 安全协议对比总结

### 25.12.1 TLS vs IPsec

| 特性 | TLS | IPsec |
|------|-----|-------|
| 工作层 | 传输层 (Layer 4) | 网络层 (Layer 3) |
| 保护对象 | 应用层数据 | 整个 IP 包 |
| 典型用途 | Web 浏览 (HTTPS)、邮件 | VPN、站点互联 |
| 认证方式 | 证书、PSK | 证书、PSK、EAP |
| NAT 穿越 | 天然支持 | 需要 NAT-T |
| 客户端要求 | 浏览器内置 | 需要配置或客户端 |
| 性能 | 软件实现 | 可硬件加速 |

### 25.12.2 AH vs ESP

| 特性 | AH | ESP |
|------|-----|-----|
| 加密 | 不支持 | 支持 |
| 完整性 | 支持 (含 IP 头) | 支载荷部分) |
| NAT 穿越 | 不支持 (IP 头被保护) | 支持 (传输模式受限) |
| 当前推荐 | 不推荐使用 | 推荐使用 |

### 25.12.3 防火墙类型对比

| 类型 | 检查层 | 状态跟踪 | 速度 | 精度 |
|------|--------|----------|------|------|
| 包过滤 | L3/L4 | 无 | 最快 | 低 |
| 状态检测 | L3/L4 | 有 | 快 | 中 |
| 应用代理 | L7 | 有 | 慢 | 高 |
| NGFW | L3-L7 | 有 | 中 | 最高 |

---

## 25.13 习题

1. **概念题**：解释 TLS 1.2 握手中 `Pre-Master Secret`、`Master Secret` 和 `Key Block` 三者的关系和推导过程。

2. **分析题**：比较 TLS 1.2 的 2-RTT 握手和 TLS 1.3 的 1-RTT 握手，说明 TLS 1.3 如何减少了一个往返。提示：考虑 key_share 扩展的作用。

3. **配置题**：为以下场景编写 iptables 规则：
   - Web 服务器（允许 80/443 端口入站）
   - 禁止外部访问 SSH
   - 允许内网 10.0.0.0/8 网段的 SSH 访问
   - 防 SYN 洪泛攻击

4. **实践题**：使用 OpenSSL 命令行工具：
   - 生成一个 RSA 2048 位密钥对
   - 创建自签名证书（有效期 365 天）
   - 查看证书的详细信息
   - 测试与远程服务器的 TLS 连接

5. **设计题**：为一个企业设计网络安全架构，需要考虑：
   - 总部与分支之间的 VPN 连接方案（IPsec vs SSL VPN 的选择）
   - 防火墙部署位置和类型选择
   - IDS/IPS 的部署模式
   - 远程办公人员的接入方式

---

## 25.14 参考资料

- RFC 5246: The Transport Layer Security (TLS) Protocol Version 1.2
- RFC 8446: The Transport Layer Security (TLS) Protocol Version 1.3
- RFC 4301: Security Architecture for the Internet Protocol
- RFC 4302: IP Authentication Header (AH)
- RFC 4303: IP Encapsulating Security Payload (ESP)
- RFC 7296: Internet Key Exchange Protocol Version 2 (IKEv2)
- RFC 5880: Bidirectional Forwarding Detection (BFD)
- Snort Users Manual (https://www.snort.org/documents)
- NIST SP 800-77: Guide to IPsec VPNs
- NIST SP 800-52: Guidelines for the Selection, Configuration, and Use of TLS Implementations

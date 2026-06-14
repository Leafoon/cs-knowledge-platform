# Chapter 20: 应用层 — HTTP 与 Web 技术

> **学习目标**：
> - 理解 HTTP 请求-响应模型与无状态协议特性
> - 掌握 HTTP 方法（GET/POST/PUT/DELETE/HEAD/OPTIONS）的语义与安全属性
> - 熟悉 HTTP 状态码（1xx-5xx）的分类与常见状态码的含义
> - 理解 HTTP 头部的作用：内容协商、缓存控制、身份认证
> - 掌握 HTTP/1.0 到 HTTP/3 的演进：持久连接、流水线、帧多路复用
> - 理解 TLS 握手过程与 HTTPS 的安全机制
> - 掌握 Web 缓存与 CDN 的工作原理

---

## 20.1 HTTP 协议概述

### 20.1.1 Web 的基础架构

HTTP（HyperText Transfer Protocol，超文本传输协议）是万维网的基础。Tim Berners-Lee 在 1989 年发明了 HTTP，用于在 CERN 的研究人员之间共享文档。

```
Web 的基本模型:
  ┌──────────┐          ┌──────────┐
  │  浏览器   │ ◄──────► │  服务器   │
  │ (客户端)  │  HTTP    │ (Web 服务器)
  └──────────┘          └──────────┘

  浏览器: Chrome, Firefox, Safari, Edge
  服务器: Apache, Nginx, IIS, Tomcat

HTTP 的特点:
  1. 请求-响应模型：客户端发起请求，服务器返回响应
  2. 无状态协议：每个请求独立，服务器不记住之前的请求
  3. 可扩展：通过头部字段扩展功能
  4. 应用层协议：通常运行在 TCP 之上（HTTP/3 使用 QUIC）
```

### 20.1.2 HTTP 版本历史

| 版本 | 年份 | 主要特性 |
|------|------|---------|
| HTTP/0.9 | 1991 | 最简单的 GET 请求，无头部 |
| HTTP/1.0 | 1996 | 添加头部、状态码、Content-Type |
| HTTP/1.1 | 1997 | 持久连接、Host 头部、分块传输 |
| HTTP/2 | 2015 | 二进制帧、多路复用、头部压缩 |
| HTTP/3 | 2022 | 基于 QUIC，解决 TCP 队头阻塞 |

---

## 20.2 HTTP 请求-响应模型

### 20.2.1 请求报文格式

```
请求报文结构:
  ┌─────────────────────────────────────┐
  │ 请求行 (Request Line)               │
  │   方法 SP URL SP 版本 CRLF          │
  ├─────────────────────────────────────┤
  │ 请求头部 (Request Headers)          │
  │   头部名: 值 CRLF                   │
  │   ...                               │
  │   CRLF                              │
  ├─────────────────────────────────────┤
  │ 空行                                │
  ├─────────────────────────────────────┤
  │ 请求体 (Request Body) [可选]        │
  └─────────────────────────────────────┘

示例:
GET /index.html HTTP/1.1
Host: www.example.com
User-Agent: Mozilla/5.0
Accept: text/html
Accept-Language: zh-CN
Connection: keep-alive
```

### 20.2.2 响应报文格式

```
响应报文结构:
  ┌─────────────────────────────────────┐
  │ 状态行 (Status Line)                │
  │   版本 SP 状态码 SP 原因短语 CRLF    │
  ├─────────────────────────────────────┤
  │ 响应头部 (Response Headers)         │
  │   头部名: 值 CRLF                   │
  │   ...                               │
  │   CRLF                              │
  ├─────────────────────────────────────┤
  │ 空行                                │
  ├─────────────────────────────────────┤
  │ 响应体 (Response Body) [可选]       │
  └─────────────────────────────────────┘

示例:
HTTP/1.1 200 OK
Date: Mon, 01 Jan 2024 00:00:00 GMT
Content-Type: text/html; charset=UTF-8
Content-Length: 1234
Connection: keep-alive

<html>
<head><title>Example</title></head>
<body><h1>Hello World</h1></body>
</html>
```

---

## 20.3 HTTP 方法

### 20.3.1 方法语义

| 方法 | 语义 | 幂等 | 安全 | 有请求体 | 有响应体 |
|------|------|------|------|---------|---------|
| GET | 获取资源 | 是 | 是 | 否 | 是 |
| POST | 创建资源/提交数据 | 否 | 否 | 是 | 是 |
| PUT | 替换资源 | 是 | 否 | 是 | 可选 |
| DELETE | 删除资源 | 是 | 否 | 可选 | 可选 |
| HEAD | 获取头部信息 | 是 | 是 | 否 | 否 |
| OPTIONS | 获取支持的方法 | 是 | 是 | 可选 | 是 |
| PATCH | 部分修改资源 | 否 | 否 | 是 | 是 |
| TRACE | 回显请求 | 是 | 是 | 否 | 是 |

**幂等性**：同一个请求执行多次，效果与执行一次相同。
**安全性**：请求不会修改服务器上的资源。

### 20.3.2 GET 与 POST 的区别

```
GET 请求:
  - 参数在 URL 中（?key=value&key2=value2）
  - 有长度限制（通常 2048 字符）
  - 可被缓存
  - 可被收藏为书签
  - 不应修改服务器状态
  - 有安全风险（参数在 URL 中可见）

POST 请求:
  - 参数在请求体中
  - 没有大小限制
  - 不被缓存（默认）
  - 不可收藏为书签
  - 用于创建或修改资源
  - 相对更安全（参数不在 URL 中）
```

```python
# GET 请求示例
GET /api/users?page=1&limit=10 HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>

# POST 请求示例
POST /api/users HTTP/1.1
Host: api.example.com
Content-Type: application/json
Authorization: Bearer <token>

{
    "name": "张三",
    "email": "zhangsan@example.com"
}
```

<div data-component="HTTPMethodDemo"></div>

---

## 20.4 HTTP 状态码

### 20.4.1 状态码分类

```
状态码由 3 位数字组成，第一位表示类别:

1xx (信息性): 请求已被接收，继续处理
2xx (成功): 请求已成功被服务器接收、理解、并接受
3xx (重定向): 需要后续操作才能完成这一请求
4xx (客户端错误): 请求含有词法错误或者无法被执行
5xx (服务器错误): 服务器在处理某个正确请求时发生错误
```

### 20.4.2 常见状态码详解

**2xx 成功**：

```
200 OK
  请求成功。响应体包含请求的资源。

201 Created
  请求成功，并因此创建了一个新资源。
  Location 头部包含新资源的 URL。

204 No Content
  请求成功，但没有内容返回。
  常用于 DELETE 请求。

206 Partial Content
  服务器成功处理了部分 GET 请求（断点续传）。
  Content-Range 头部指定返回的数据范围。
```

**3xx 重定向**：

```
301 Moved Permanently
  资源已永久移动到新位置。
  浏览器会自动重定向，搜索引擎更新链接。

302 Found (临时重定向)
  资源临时位于其他位置。
  浏览器会自动重定向，但搜索引擎不更新链接。

304 Not Modified
  资源未修改，使用缓存版本。
  条件请求（If-Modified-Since 或 If-None-Match）的结果。

307 Temporary Redirect
  临时重定向，保持请求方法不变。
  与 302 的区别：307 不允许将 POST 改为 GET。

308 Permanent Redirect
  永久重定向，保持请求方法不变。
  与 301 的区别：308 不允许将 POST 改为 GET。
```

**4xx 客户端错误**：

```
400 Bad Request
  请求语法错误，服务器无法理解。

401 Unauthorized
  请求需要身份认证。
  响应必须包含 WWW-Authenticate 头部。

403 Forbidden
  服务器理解请求，但拒绝执行。
  与 401 的区别：403 不会因为提供认证信息而改变。

404 Not Found
  请求的资源不存在。
  可能是 URL 错误，或资源已被删除。

405 Method Not Allowed
  请求方法不被允许。
  响应必须包含 Allow 头部，列出支持的方法。

429 Too Many Requests
  客户端发送了太多请求（限流）。
  响应可以包含 Retry-After 头部。

451 Unavailable For Legal Reasons
  因法律原因不可用（如版权、审查）。
```

**5xx 服务器错误**：

```
500 Internal Server Error
  服务器内部错误。
  通常是服务器代码的 Bug。

502 Bad Gateway
  网关或代理服务器从上游服务器收到无效响应。

503 Service Unavailable
  服务器暂时不可用（过载或维护）。
  响应可以包含 Retry-After 头部。

504 Gateway Timeout
  网关或代理服务器未能及时从上游服务器收到响应。
```

<div data-component="HTTPStatusCodeExplorer"></div>

---

## 20.5 HTTP 头部字段

### 20.5.1 通用头部

```
Date: Mon, 01 Jan 2024 00:00:00 GMT
  消息产生的日期和时间

Connection: keep-alive
  连接管理选项
  keep-alive: 保持连接（HTTP/1.1 默认）
  close: 关闭连接

Cache-Control: max-age=3600
  缓存控制指令
  no-cache: 使用缓存前必须验证
  no-store: 不缓存
  max-age: 缓存有效时间（秒）
  public: 可被任何缓存存储
  private: 只能被浏览器缓存

Transfer-Encoding: chunked
  传输编码
  chunked: 分块传输（数据长度未知时）
```

### 20.5.2 请求头部

```
Host: www.example.com
  目标主机名（HTTP/1.1 必须）
  虚拟主机的关键

User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)
  客户端软件信息

Accept: text/html, application/json
  客户端可接受的媒体类型

Accept-Language: zh-CN, en-US
  客户端可接受的语言

Accept-Encoding: gzip, deflate, br
  客户端可接受的内容编码

Authorization: Bearer <token>
  身份认证凭证

Cookie: session=abc123; user=zhangsan
  客户端存储的 Cookie

If-Modified-Since: Wed, 21 Oct 2024 07:28:00 GMT
  条件请求：只在资源修改后返回

If-None-Match: "etag-value"
  条件请求：ETag 匹配时返回 304

Referer: https://www.example.com/previous-page
  请求来源页面
```

### 20.5.3 响应头部

```
Content-Type: text/html; charset=UTF-8
  响应体的媒体类型

Content-Length: 1234
  响应体的字节长度

Content-Encoding: gzip
  响应体的编码方式

ETag: "abc123"
  资源的实体标签（用于缓存验证）

Last-Modified: Wed, 21 Oct 2024 07:28:00 GMT
  资源最后修改时间

Location: /new-url
  重定向目标 URL

Set-Cookie: session=abc123; HttpOnly; Secure
  设置 Cookie

Access-Control-Allow-Origin: https://trusted.com
  CORS 跨域资源共享策略

Strict-Transport-Security: max-age=31536000; includeSubDomains
  HSTS 安全策略
```

### 20.5.4 内容协商

```
内容协商机制:
  客户端通过 Accept-* 头部表达偏好
  服务器选择最佳匹配的资源表示

Accept: text/html, application/json
  → 服务器返回 HTML 或 JSON

Accept-Language: zh-CN, en-US;q=0.9
  → 优先中文，其次英文（q 值表示权重）

Accept-Encoding: gzip, br
  → 服务器使用 gzip 或 Brotli 压缩

Accept-Charset: UTF-8
  → 服务器使用 UTF-8 编码
```

<div data-component="HTTPHeaderExplorer"></div>

---

## 20.6 HTTP 协议演进

### 20.6.1 HTTP/1.0 的问题

```
HTTP/1.0 的主要问题:
  1. 每个请求都需要新的 TCP 连接
     - TCP 三次握手: 1.5 RTT
     - TLS 握手: 1-2 RTT
     - 请求/响应: 1 RTT
     - 总计: 3.5-4.5 RTT per request!

  2. 无法流水线处理
     - 请求必须串行发送

  3. 头部冗余
     - 每个请求都发送完整的头部
```

### 20.6.2 HTTP/1.1 的改进

```
HTTP/1.1 的关键改进:

1. 持久连接（Persistent Connection）
   Connection: keep-alive（默认启用）
   同一个 TCP 连接上可以发送多个请求

2. 管道化（Pipelining）
   客户端可以连续发送多个请求，不必等待响应
   但响应必须按请求顺序返回（队头阻塞！）

3. Host 头部（必需）
   支持虚拟主机（一个 IP 多个网站）

4. 分块传输编码
   Transfer-Encoding: chunked
   无需提前知道内容长度

5. 缓存增强
   ETag、If-None-Match、Cache-Control
```

**HTTP/1.1 管道化的问题**：

```
请求: [Req1] [Req2] [Req3]
响应: [Res1] [Res2] [Res3]  ← 必须按顺序！

如果 Res1 很慢:
  即使 Res2、Res3 已准备好，也必须等待 Res1
  → 队头阻塞（应用层）
```

### 20.6.3 HTTP/2 帧处理器的流管理与 HPACK 压缩器

HTTP/2 的核心创新是**二进制帧层**：

```
HTTP/2 帧格式:
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                 Length (24)                    |
+-+-+-+-+-+-+-+-+               +-------------------------------+
|   Type (8)    |   Flags (8)   |
+-+-------------+---------------+-------------------------------+
|R|                 Stream Identifier (31)                       |
+-+-------------------------------------------------------------+
|                     Frame Payload ...                          |
+---------------------------------------------------------------+

帧类型:
  0x0 - DATA:        传输数据
  0x1 - HEADERS:     传输头部
  0x2 - PRIORITY:    设置流优先级
  0x3 - RST_STREAM:  终止流
  0x4 - SETTINGS:    连接设置
  0x5 - PUSH_PROMISE: 服务器推送
  0x6 - PING:        心跳检测
  0x7 - GOAWAY:      优雅关闭
  0x8 - WINDOW_UPDATE: 流量控制
  0x9 - CONTINUATION: 头部延续
```

**HTTP/2 流管理**：

```python
class HTTP2Stream:
    """HTTP/2 流状态管理"""
    IDLE = 0
    OPEN = 1
    HALF_CLOSED_LOCAL = 2
    HALF_CLOSED_REMOTE = 3
    CLOSED = 4

    def __init__(self, stream_id: int, weight: int = 16):
        self.stream_id = stream_id
        self.state = self.IDLE
        self.weight = weight
        self.send_window = 65535      # 发送窗口
        self.recv_window = 65535      # 接收窗口
        self.headers = {}             # 头部字段
        self.data_buffer = bytearray()
        self.parent = None
        self.children = []

    def send_headers(self, headers: dict, end_stream: bool = False):
        if self.state == self.IDLE:
            self.state = self.OPEN
        self.headers.update(headers)
        if end_stream:
            self.state = self.HALF_CLOSED_LOCAL

    def recv_headers(self, headers: dict, end_stream: bool = False):
        if self.state == self.IDLE:
            self.state = self.OPEN
        if end_stream:
            self.state = self.HALF_CLOSED_REMOTE

    def send_data(self, data: bytes, end_stream: bool = False):
        if self.send_window < len(data):
            raise FlowControlError("Send window exceeded")
        self.send_window -= len(data)
        if end_stream:
            self.state = self.HALF_CLOSED_LOCAL

    def recv_data(self, data: bytes, end_stream: bool = False):
        if self.recv_window < len(data):
            raise FlowControlError("Receive window exceeded")
        self.recv_window -= len(data)
        self.data_buffer.extend(data)
        if end_stream:
            self.state = self.HALF_CLOSED_REMOTE


class HTTP2Connection:
    """HTTP/2 连接管理"""
    def __init__(self):
        self.streams = {}
        self.next_stream_id = 1       # 客户端奇数，服务器偶数
        self.settings = {
            'SETTINGS_HEADER_TABLE_SIZE': 4096,
            'SETTINGS_ENABLE_PUSH': 1,
            'SETTINGS_MAX_CONCURRENT_STREAMS': 100,
            'SETTINGS_INITIAL_WINDOW_SIZE': 65535,
            'SETTINGS_MAX_FRAME_SIZE': 16384,
            'SETTINGS_MAX_HEADER_LIST_SIZE': 8192,
        }
        self.hp_encoder = HPACKEncoder()
        self.hp_decoder = HPACKDecoder()

    def create_stream(self, weight: int = 16) -> HTTP2Stream:
        stream_id = self.next_stream_id
        self.next_stream_id += 2  # 客户端发起的流使用奇数 ID
        stream = HTTP2Stream(stream_id, weight)
        self.streams[stream_id] = stream
        return stream

    def send_request(self, method: str, path: str, headers: dict) -> HTTP2Stream:
        stream = self.create_stream()
        pseudo_headers = {
            ':method': method,
            ':path': path,
            ':scheme': 'https',
            ':authority': headers.pop('Host', '')
        }
        all_headers = {**pseudo_headers, **headers}
        encoded = self.hp_encoder.encode(all_headers)
        stream.send_headers(all_headers, end_stream=(method == 'GET'))
        return stream

    def handle_frame(self, frame: bytes):
        length = int.from_bytes(frame[0:3], 'big')
        frame_type = frame[3]
        flags = frame[4]
        stream_id = int.from_bytes(frame[5:9], 'big') & 0x7FFFFFFF

        if frame_type == 0x1:  # HEADERS
            self._handle_headers(stream_id, frame[9:9+length], flags)
        elif frame_type == 0x0:  # DATA
            self._handle_data(stream_id, frame[9:9+length], flags)
        elif frame_type == 0x4:  # SETTINGS
            self._handle_settings(frame[9:9+length], flags)
        elif frame_type == 0x8:  # WINDOW_UPDATE
            self._handle_window_update(stream_id, frame[9:9+length])

    def _handle_headers(self, stream_id: int, payload: bytes, flags: int):
        if stream_id not in self.streams:
            self.streams[stream_id] = HTTP2Stream(stream_id)
        stream = self.streams[stream_id]
        headers = self.hp_decoder.decode(payload)
        end_stream = bool(flags & 0x1)
        stream.recv_headers(headers, end_stream)
```

**HPACK 头部压缩**：

```
HPACK 核心机制:
  1. 静态表（Static Table）
     - 61 个预定义的常用头部
     - 如: :method: GET (索引 2), :status: 200 (索引 8)

  2. 动态表（Dynamic Table）
     - 连接期间动态添加的头部
     - 最大大小可配置（SETTINGS_HEADER_TABLE_SIZE）

  3. 霍夫曼编码（Huffman Coding）
     - 对头部值进行压缩
     - 常见字符使用短编码

示例:
  第一次请求:
    :method: GET       → 索引 2（静态表）
    :path: /index.html → 字面量编码
    Host: example.com  → 字面量编码，添加到动态表（索引 62）

  第二次请求:
    :method: GET       → 索引 2（静态表）
    :path: /about      → 字面量编码
    Host: example.com  → 索引 62（动态表，0 字节！）
```

<div data-component="HTTP2FrameDemo"></div>

### 20.6.4 HTTP/3 基于 QUIC

HTTP/3 解决了 HTTP/2 的 TCP 队头阻塞问题：

```
HTTP/2 的问题:
  虽然应用层可以并行处理多个流
  但 TCP 层仍然有队头阻塞
  一个 TCP 包丢失 → 所有流都受影响

HTTP/3 的解决方案:
  使用 QUIC 替代 TCP
  每个流独立可靠交付
  一个流的丢包不影响其他流

  HTTP/2 over TCP:  流多路复用 + TCP 字节流 = 潜在队头阻塞
  HTTP/3 over QUIC: 流多路复用 + QUIC 流 = 无队头阻塞
```

<div data-component="HTTPVersionComparison"></div>

---

## 20.7 TLS 握手引擎与 HTTPS

### 20.7.1 HTTPS 概述

HTTPS = HTTP + TLS（Transport Layer Security），提供：

```
HTTPS 的安全保证:
  1. 机密性（Confidentiality）
     - 数据加密传输，中间人无法读取

  2. 完整性（Integrity）
     - 数据被篡改会被检测到

  3. 身份认证（Authentication）
     - 通过证书验证服务器身份
```

### 20.7.2 TLS 1.3 握手过程

```
TLS 1.3 握手（1-RTT）:

客户端                                           服务器
  |                                                |
  |--- ClientHello --------------------------------->|
  |    - 支持的 TLS 版本                            |
  |    - 支持的密码套件                              |
  |    - 客户端随机数                                |
  |    - 密钥共享（密钥交换参数）                    |
  |    - 支持的签名算法                              |
  |                                                |
  |<-- ServerHello ----------------------------------|
  |    - 选择的密码套件                              |
  |    - 服务器随机数                                |
  |    - 密钥共享（密钥交换参数）                    |
  |<-- EncryptedExtensions --------------------------|
  |    - 协商的应用层协议（ALPN）                    |
  |<-- Certificate ----------------------------------|
  |    - 服务器证书链                                |
  |<-- CertificateVerify ---------------------------|
  |    - 证书签名                                    |
  |<-- Finished -------------------------------------|
  |    - 握手完成验证                                |
  |                                                |
  |--- Finished ------------------------------------>|
  |                                                |
  |=== 应用数据 ====================================|
```

### 20.7.3 TLS 1.3 密钥协商管线

```python
class TLSHandshake:
    """TLS 1.3 握手引擎（简化版）"""
    def __init__(self):
        self.client_random = b''
        self.server_random = b''
        self.shared_secret = b''
        self.handshake_transcript = bytearray()

    def client_hello(self) -> dict:
        """构建 ClientHello 消息"""
        self.client_random = generate_random(32)
        self.client_key_share = self._generate_key_share()

        client_hello = {
            'version': 0x0304,  # TLS 1.3
            'random': self.client_random,
            'cipher_suites': [
                'TLS_AES_256_GCM_SHA384',
                'TLS_CHACHA20_POLY1305_SHA256',
                'TLS_AES_128_GCM_SHA256'
            ],
            'extensions': {
                'supported_versions': [0x0304],
                'key_share': self.client_key_share,
                'signature_algorithms': ['ecdsa_secp256r1_sha256', 'rsa_pss_rsae_sha256'],
                'server_name': 'www.example.com',  # SNI
            }
        }
        self._update_transcript(client_hello)
        return client_hello

    def server_hello(self, client_hello: dict) -> dict:
        """处理 ClientHello，构建 ServerHello"""
        self.server_random = generate_random(32)
        cipher_suite = self._select_cipher_suite(client_hello['cipher_suites'])
        self.server_key_share = self._generate_key_share()

        # 计算共享密钥
        self.shared_secret = self._key_exchange(
            client_hello['extensions']['key_share'],
            self.server_key_share
        )

        server_hello = {
            'version': 0x0304,
            'random': self.server_random,
            'cipher_suite': cipher_suite,
            'extensions': {
                'key_share': self.server_key_share,
            }
        }
        self._update_transcript(server_hello)
        return server_hello

    def _key_exchange(self, client_share: bytes, server_share: bytes) -> bytes:
        """ECDHE 密钥交换"""
        # 使用 X25519 或 P-256 曲线
        private_key = generate_ecdh_private_key()
        public_key = ecdh_public_key(private_key)
        shared_secret = ecdh_compute_shared(private_key, client_share)
        return shared_secret

    def derive_keys(self) -> dict:
        """从共享密钥派生 TLS 会话密钥"""
        # HKDF-Expand-Label
        early_secret = hkdf_extract(b'\x00' * 32, b'')
        handshake_secret = hkdf_extract(early_secret, self.shared_secret)

        # 派生各类密钥
        keys = {
            'client_handshake_key': hkdf_expand_label(handshake_secret, b'c hs traffic', self._transcript_hash(), 32),
            'server_handshake_key': hkdf_expand_label(handshake_secret, b's hs traffic', self._transcript_hash(), 32),
            'client_app_key': hkdf_expand_label(handshake_secret, b'c ap traffic', self._transcript_hash(), 32),
            'server_app_key': hkdf_expand_label(handshake_secret, b's ap traffic', self._transcript_hash(), 32),
        }
        return keys
```

### 20.7.4 证书验证

```python
class CertificateValidator:
    """TLS 证书验证器"""
    def __init__(self, trusted_roots: list):
        self.trusted_roots = trusted_roots

    def validate(self, cert_chain: list, server_name: str) -> bool:
        """验证证书链"""
        # 1. 验证证书链完整性
        for i in range(len(cert_chain) - 1):
            if not self._verify_signature(cert_chain[i], cert_chain[i + 1]):
                return False

        # 2. 验证根证书是否可信
        root_cert = cert_chain[-1]
        if not self._is_trusted_root(root_cert):
            return False

        # 3. 验证域名匹配
        leaf_cert = cert_chain[0]
        if not self._match_hostname(leaf_cert, server_name):
            return False

        # 4. 验证有效期
        if not self._check_validity(leaf_cert):
            return False

        # 5. 检查吊销状态（CRL 或 OCSP）
        if self._is_revoked(leaf_cert):
            return False

        return True

    def _match_hostname(self, cert: dict, hostname: str) -> bool:
        """验证证书中的域名与请求的域名匹配"""
        # 检查 Subject Alternative Name (SAN)
        sans = cert.get('subject_alt_names', [])
        for san in sans:
            if san.startswith('*.'):
                # 通配符匹配
                pattern = san[2:]
                if hostname.endswith(pattern):
                    # 确保只匹配一级子域名
                    prefix = hostname[:-len(pattern)]
                    if '.' not in prefix.rstrip('.'):
                        return True
            elif san == hostname:
                return True
        return False
```

<div data-component="TLSHandshakeDemo"></div>

---

## 20.8 Web 缓存与 CDN

### 20.8.1 缓存的作用

```
Web 缓存的好处:
  1. 减少延迟
     - 从缓存读取比从源服务器快得多
     - 用户体验更好

  2. 减少带宽消耗
     - 避免重复传输相同内容
     - 节省服务器和网络资源

  3. 减轻服务器负载
     - 大量请求被缓存处理
     - 服务器可以处理更多用户

  4. 提高可用性
     - 即使源服务器故障，缓存仍可提供服务
```

### 20.8.2 条件 GET 请求

```
条件 GET 工作流程:

第一次请求:
  客户端: GET /image.jpg HTTP/1.1
  服务器: HTTP/1.1 200 OK
          ETag: "abc123"
          Last-Modified: Wed, 01 Jan 2024 00:00:00 GMT
          Content-Length: 12345
          [图片数据]

第二次请求（验证缓存）:
  客户端: GET /image.jpg HTTP/1.1
          If-None-Match: "abc123"
          If-Modified-Since: Wed, 01 Jan 2024 00:00:00 GMT

  如果未修改:
    服务器: HTTP/1.1 304 Not Modified
            （无响应体，节省带宽）

  如果已修改:
    服务器: HTTP/1.1 200 OK
            ETag: "def456"
            [新的图片数据]
```

### 20.8.3 CDN 边缘节点的缓存替换策略

```python
from collections import OrderedDict
import heapq

class CDNEdgeCache:
    """CDN 边缘节点缓存"""
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.current_size = 0

    def get(self, url: str) -> Optional[bytes]:
        raise NotImplementedError

    def put(self, url: str, content: bytes, headers: dict):
        raise NotImplementedError


class LRUCache(CDNEdgeCache):
    """LRU (Least Recently Used) 缓存"""
    def __init__(self, max_size: int):
        super().__init__(max_size)
        self.cache = OrderedDict()  # {url: (content, headers)}

    def get(self, url: str) -> Optional[bytes]:
        if url in self.cache:
            # 移到末尾（最近使用）
            self.cache.move_to_end(url)
            return self.cache[url][0]
        return None

    def put(self, url: str, content: bytes, headers: dict):
        content_size = len(content)
        if content_size > self.max_size:
            return  # 单个内容超过缓存大小

        if url in self.cache:
            self.cache.move_to_end(url)
            self.current_size -= len(self.cache[url][0])
        else:
            # 淘汰最久未使用的
            while self.current_size + content_size > self.max_size:
                _, (evicted_content, _) = self.cache.popitem(last=False)
                self.current_size -= len(evicted_content)

        self.cache[url] = (content, headers)
        self.current_size += content_size


class LFUCache(CDNEdgeCache):
    """LFU (Least Frequently Used) 缓存"""
    def __init__(self, max_size: int):
        super().__init__(max_size)
        self.cache = {}           # {url: (content, headers)}
        self.freq = {}            # {url: frequency}
        self.freq_buckets = {}    # {freq: set(urls)}

    def get(self, url: str) -> Optional[bytes]:
        if url in self.cache:
            self._increment_freq(url)
            return self.cache[url][0]
        return None

    def put(self, url: str, content: bytes, headers: dict):
        content_size = len(content)
        if content_size > self.max_size:
            return

        if url in self.cache:
            self.current_size -= len(self.cache[url][0])
            self._increment_freq(url)
        else:
            while self.current_size + content_size > self.max_size:
                self._evict_lfu()

        self.cache[url] = (content, headers)
        self.current_size += content_size
        self.freq[url] = 1
        self.freq_buckets.setdefault(1, set()).add(url)

    def _increment_freq(self, url: str):
        old_freq = self.freq[url]
        self.freq[url] = old_freq + 1
        self.freq_buckets[old_freq].discard(url)
        if not self.freq_buckets[old_freq]:
            del self.freq_buckets[old_freq]
        self.freq_buckets.setdefault(old_freq + 1, set()).add(url)

    def _evict_lfu(self):
        min_freq = min(self.freq_buckets.keys())
        evict_url = self.freq_buckets[min_freq].pop()
        if not self.freq_buckets[min_freq]:
            del self.freq_buckets[min_freq]
        evicted_content, _ = self.cache.pop(evict_url)
        self.current_size -= len(evicted_content)
        del self.freq[evict_url]


class AdaptiveCache(CDNEdgeCache):
    """自适应缓存（结合 LRU 和 LFU）"""
    def __init__(self, max_size: int):
        super().__init__(max_size)
        self.lru = LRUCache(max_size)
        self.lfu = LFUCache(max_size)
        self.access_count = {}  # {url: count}

    def get(self, url: str) -> Optional[bytes]:
        self.access_count[url] = self.access_count.get(url, 0) + 1
        # 热点内容使用 LFU，普通内容使用 LRU
        if self.access_count[url] > 10:
            return self.lfu.get(url)
        return self.lru.get(url)

    def put(self, url: str, content: bytes, headers: dict):
        self.lru.put(url, content, headers)
        self.lfu.put(url, content, headers)
```

<div data-component="CacheStrategyDemo"></div>

---

## 20.9 HTTP/2 服务器推送

### 20.9.1 推送机制

```
服务器推送工作原理:

传统方式:
  1. 客户端请求 index.html
  2. 服务器返回 index.html
  3. 客户端解析，发现需要 style.css
  4. 客户端请求 style.css
  5. 服务器返回 style.css
  总共需要 2 个请求-响应周期

服务器推送:
  1. 客户端请求 index.html
  2. 服务器返回 index.html
  3. 服务器主动推送 style.css（无需客户端请求）
  总共只需要 1 个请求-响应周期

  节省了 1 个 RTT
```

### 20.9.2 推送限制

```
服务器推送的限制:
  1. 只能推送客户端可能需要的资源
  2. 客户端可以拒绝推送（RST_STREAM）
  3. 不能推送 POST 请求
  4. 推送的资源必须来自同一源（origin）

  注意: HTTP/3 已弃用服务器推送，因为收益有限
```

---

## 20.10 WebSocket 协议

### 20.10.1 WebSocket 概述

WebSocket 是在单个 TCP 连接上提供**全双工**通信的协议：

```
HTTP 通信（半双工）:
  客户端 → 服务器: 请求
  客户端 ← 服务器: 响应
  （服务器不能主动发送数据）

WebSocket 通信（全双工）:
  客户端 ↔ 服务器: 双向任意通信
  （服务器可以主动推送数据）
```

### 20.10.2 WebSocket 握手

```
WebSocket 握手（基于 HTTP 升级）:

客户端请求:
  GET /chat HTTP/1.1
  Host: server.example.com
  Upgrade: websocket
  Connection: Upgrade
  Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
  Sec-WebSocket-Version: 13

服务器响应:
  HTTP/1.1 101 Switching Protocols
  Upgrade: websocket
  Connection: Upgrade
  Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=

握手完成后，连接升级为 WebSocket 协议
```

<div data-component="WebSocketDemo"></div>

---

## 20.11 HTTP 请求解析器实现

### 20.11.1 HTTP 请求解析

```python
class HTTPRequestParser:
    """HTTP 请求解析器"""
    def __init__(self):
        self.buffer = b''
        self.state = 'REQUEST_LINE'
        self.request = None

    def feed(self, data: bytes) -> Optional['HTTPRequest']:
        self.buffer += data

        if self.state == 'REQUEST_LINE':
            if b'\r\n' in self.buffer:
                line, self.buffer = self.buffer.split(b'\r\n', 1)
                self.request = HTTPRequest()
                self._parse_request_line(line)
                self.state = 'HEADERS'

        if self.state == 'HEADERS':
            while b'\r\n' in self.buffer:
                line, self.buffer = self.buffer.split(b'\r\n', 1)
                if line == b'':
                    # 空行，头部结束
                    self.state = 'BODY'
                    break
                self._parse_header(line)

        if self.state == 'BODY':
            content_length = self.request.headers.get('Content-Length')
            if content_length:
                if len(self.buffer) >= int(content_length):
                    self.request.body = self.buffer[:int(content_length)]
                    self.buffer = self.buffer[int(content_length):]
                    self.state = 'COMPLETE'
                    return self.request
            else:
                self.state = 'COMPLETE'
                return self.request

        return None

    def _parse_request_line(self, line: bytes):
        parts = line.decode().split(' ')
        self.request.method = parts[0]
        self.request.path = parts[1]
        self.request.version = parts[2]

    def _parse_header(self, line: bytes):
        key, value = line.decode().split(': ', 1)
        self.request.headers[key] = value


class HTTPResponseBuilder:
    """HTTP 响应构建器"""
    def __init__(self):
        self.status_code = 200
        self.reason = 'OK'
        self.headers = {}
        self.body = b''

    def set_status(self, code: int, reason: str = None):
        self.status_code = code
        self.reason = reason or self._default_reason(code)
        return self

    def set_header(self, key: str, value: str):
        self.headers[key] = value
        return self

    def set_body(self, body: bytes, content_type: str = 'text/html'):
        self.body = body
        self.headers['Content-Type'] = content_type
        self.headers['Content-Length'] = str(len(body))
        return self

    def build(self) -> bytes:
        status_line = f'HTTP/1.1 {self.status_code} {self.reason}\r\n'
        headers = ''.join(f'{k}: {v}\r\n' for k, v in self.headers.items())
        return (status_line + headers + '\r\n').encode() + self.body

    def _default_reason(self, code: int) -> str:
        reasons = {200: 'OK', 301: 'Moved Permanently', 304: 'Not Modified',
                   400: 'Bad Request', 404: 'Not Found', 500: 'Internal Server Error'}
        return reasons.get(code, 'Unknown')
```

---

## 20.12 章节小结

本章详细介绍了 HTTP 协议与 Web 技术的各个方面：

1. **请求-响应模型**：HTTP 的基本工作原理与报文格式
2. **HTTP 方法**：GET/POST/PUT/DELETE 等方法的语义与特性
3. **状态码**：1xx-5xx 的分类与常见状态码的含义
4. **头部字段**：内容协商、缓存控制、身份认证等机制
5. **协议演进**：从 HTTP/1.0 到 HTTP/3 的技术进步
6. **TLS 握手**：HTTPS 的安全机制与密钥协商
7. **Web 缓存**：LRU/LFU 缓存策略与条件请求
8. **服务器推送与 WebSocket**：现代 Web 通信技术

<div data-component="ChapterSummary"></div>
<div data-component="KnowledgeCheck"></div>

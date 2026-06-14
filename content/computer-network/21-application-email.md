# Chapter 21: 应用层 — 电子邮件系统

> **学习目标**：
> - 理解电子邮件系统的整体架构：用户代理、邮件传输代理、邮件投递代理
> - 掌握 SMTP 协议的命令、响应与邮件传输流程
> - 理解 MIME 标准：Content-Type、编码方式（Base64、Quoted-Printable）
> - 掌握 POP3 和 IMAP 协议的区别与使用场景
> - 理解 Webmail 的架构与工作原理
> - 掌握垃圾邮件过滤技术：贝叶斯分类器、SPF、DKIM、DMARC
> - 了解邮件安全：PGP 和 S/MIME 的加密与签名机制

---

## 21.1 电子邮件系统架构

### 21.1.1 整体架构

电子邮件系统由三个核心组件组成：

```
┌─────────────────────────────────────────────────────────────────┐
│                    电子邮件系统架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  发件人                                                          │
│  ┌──────────────┐                                               │
│  │  用户代理     │  ← Outlook, Thunderbird, Apple Mail          │
│  │  (User Agent)│     编写、阅读、管理邮件                       │
│  └──────┬───────┘                                               │
│         │ SMTP                                                  │
│         ▼                                                       │
│  ┌──────────────┐                                               │
│  │  邮件传输代理 │  ← Postfix, Sendmail, Exchange               │
│  │  (MTA)       │     路由、转发、中继邮件                       │
│  └──────┬───────┘                                               │
│         │ SMTP                                                  │
│         ▼                                                       │
│  ┌──────────────┐   ┌──────────────┐                            │
│  │  邮件投递代理 │──►│  收件人邮箱   │                            │
│  │  (MDA)       │   │  (Mailbox)   │                            │
│  └──────────────┘   └──────┬───────┘                            │
│                            │ POP3/IMAP                          │
│                            ▼                                    │
│                     ┌──────────────┐                            │
│                     │  用户代理     │                            │
│                     │  (收件人)     │                            │
│                     └──────────────┘                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 21.1.2 邮件传输流程

```
完整邮件传输流程:
  Alice (alice@gmail.com) 发送邮件给 Bob (bob@yahoo.com)

  1. Alice 使用 Gmail 用户代理编写邮件
  2. 用户代理通过 SMTP 将邮件发送到 Gmail 的 MTA
  3. Gmail MTA 查询 DNS，获取 yahoo.com 的 MX 记录
  4. Gmail MTA 通过 SMTP 将邮件转发到 Yahoo 的 MTA
  5. Yahoo MTA 将邮件投递到 Bob 的邮箱（通过 MDA）
  6. Bob 使用 Yahoo 用户代理通过 POP3/IMAP 读取邮件

  涉及的协议:
    SMTP: 用户代理 → MTA → MTA（发送）
    POP3/IMAP: MDA → 用户代理（接收）
```

### 21.1.3 邮件地址格式

```
邮件地址: local-part@domain

示例:
  alice@gmail.com
  │     │
  │     └── 域名部分（邮件服务器地址）
  └── 本地部分（邮箱用户名）

地址规范 (RFC 5321):
  local-part: 最长 64 字符
  domain: 最长 255 字符
  总长度: 最长 320 字符

特殊字符:
  local-part 可以包含: 字母、数字、.!#$%&'*+/=?^_`{|}~-
  domain 可以包含: 字母、数字、连字符
```

<div data-component="EmailArchitectureDiagram"></div>

---

## 21.2 SMTP 协议

### 21.2.1 SMTP 概述

SMTP（Simple Mail Transfer Protocol，简单邮件传输协议）是电子邮件传输的核心协议。

```
SMTP 基本特性:
  - 基于 TCP，端口 25（MTA 之间）或 587（用户提交）
  - 使用命令-响应模型
  - 7 位 ASCII 文本传输（8 位通过扩展支持）
  - 支持邮件中继

SMTP vs HTTP:
  ┌──────────────┬──────────────┬──────────────┐
  │    特性       │     SMTP     │     HTTP     │
  ├──────────────┼──────────────┼──────────────┤
  │ 方向         │ 推 (Push)    │ 拉 (Pull)    │
  │ 数据格式     │ 7-bit ASCII  │ 任意         │
  │ 持久连接     │ 可选         │ 默认         │
  │ 状态         │ 有状态       │ 无状态       │
  └──────────────┴──────────────┴──────────────┘
```

### 21.2.2 SMTP 命令与响应

**基本命令**：

| 命令 | 语法 | 说明 |
|------|------|------|
| HELO | HELO hostname | 识别客户端 |
| EHLO | EHLO hostname | 扩展问候（推荐） |
| MAIL | MAIL FROM:<addr> | 指定发件人 |
| RCPT | RCPT TO:<addr> | 指定收件人 |
| DATA | DATA | 开始邮件内容 |
| RSET | RSET | 重置会话 |
| QUIT | QUIT | 结束会话 |
| AUTH | AUTH mechanism | 身份认证 |
| STARTTLS | STARTTLS | 启动 TLS 加密 |

**响应码**：

```
响应码格式: 3位数字 + 空格 + 文本

220 服务就绪
250 请求的操作已成功完成
354 开始邮件输入，以<CRLF>.<CRLF>结束
421 服务不可用
450 邮箱忙
451 处理中出错
452 磁盘空间不足
500 语法错误
550 邮箱不存在
553 邮箱名无效
```

### 21.2.3 SMTP 邮件传输流程

```
完整的 SMTP 会话示例:

S: 220 mail.example.com ESMTP Postfix
C: EHLO client.example.com
S: 250-mail.example.com
S: 250-PIPELINING
S: 250-SIZE 10240000
S: 250-STARTTLS
S: 250-AUTH PLAIN LOGIN
S: 250-ENHANCEDSTATUSCODES
S: 250-8BITMIME
S: 250 DSN
C: AUTH LOGIN
S: 334 VXNlcm5hbWU6
C: YWxpY2U=                    ← base64("alice")
S: 334 UGFzc3dvcmQ6
C: cGFzc3dvcmQ=                ← base64("password")
S: 235 2.7.0 Authentication successful
C: MAIL FROM:<alice@example.com>
S: 250 2.1.0 Ok
C: RCPT TO:<bob@example.org>
S: 250 2.1.5 Ok
C: DATA
S: 354 End data with <CR><LF>.<CR><LF>
C: From: alice@example.com
C: To: bob@example.org
C: Subject: Meeting Tomorrow
C: Date: Mon, 01 Jan 2024 10:00:00 +0800
C: MIME-Version: 1.0
C: Content-Type: text/plain; charset=UTF-8
C:
C: Hi Bob,
C:
C: Let's meet tomorrow at 2pm.
C:
C: Best,
C: Alice
C: .
S: 250 2.0.0 Ok: queued as ABC123
C: QUIT
S: 221 2.0.0 Bye
```

### 21.2.4 SMTP 服务器的邮件队列管理器与投递引擎

```python
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

class MailStatus(Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DELIVERED = "delivered"
    DEFERRED = "deferred"
    BOUNCED = "bounced"

@dataclass
class EmailMessage:
    sender: str
    recipients: List[str]
    subject: str
    body: str
    headers: dict = field(default_factory=dict)
    message_id: str = ""
    status: MailStatus = MailStatus.QUEUED
    retry_count: int = 0
    max_retries: int = 5
    next_retry: float = 0
    created_at: float = field(default_factory=time.time)

class MailQueue:
    """邮件队列管理器"""
    def __init__(self, max_size: int = 100000):
        self.queue = queue.PriorityQueue(maxsize=max_size)
        self.active = {}      # {message_id: EmailMessage}
        self.deferred = {}    # {message_id: EmailMessage}
        self.delivered = []   # 已投递记录
        self.lock = threading.Lock()

    def enqueue(self, message: EmailMessage) -> str:
        """将邮件加入队列"""
        message.message_id = self._generate_id()
        message.status = MailStatus.QUEUED
        priority = self._calculate_priority(message)
        self.queue.put((priority, time.time(), message))
        return message.message_id

    def dequeue(self) -> Optional[EmailMessage]:
        """从队列取出待处理邮件"""
        try:
            priority, timestamp, message = self.queue.get_nowait()
            message.status = MailStatus.PROCESSING
            with self.lock:
                self.active[message.message_id] = message
            return message
        except queue.Empty:
            # 检查延迟队列
            self._process_deferred()
            return None

    def mark_delivered(self, message_id: str):
        """标记邮件已投递"""
        with self.lock:
            if message_id in self.active:
                msg = self.active.pop(message_id)
                msg.status = MailStatus.DELIVERED
                self.delivered.append(msg)

    def mark_deferred(self, message_id: str, delay: int):
        """标记邮件延迟重试"""
        with self.lock:
            if message_id in self.active:
                msg = self.active.pop(message_id)
                msg.status = MailStatus.DEFERRED
                msg.retry_count += 1
                msg.next_retry = time.time() + delay
                self.deferred[message_id] = msg

    def mark_bounced(self, message_id: str, reason: str):
        """标记邮件退回"""
        with self.lock:
            if message_id in self.active:
                msg = self.active.pop(message_id)
                msg.status = MailStatus.BOUNCED
                # 发送退回通知给发件人
                self._send_bounce_notification(msg, reason)

    def _process_deferred(self):
        """处理延迟队列中到期的邮件"""
        now = time.time()
        with self.lock:
            ready = [mid for mid, msg in self.deferred.items()
                     if msg.next_retry <= now]
            for mid in ready:
                msg = self.deferred.pop(mid)
                if msg.retry_count >= msg.max_retries:
                    self.mark_bounced(mid, "Max retries exceeded")
                else:
                    self.enqueue(msg)

    def _calculate_priority(self, message: EmailMessage) -> int:
        """计算邮件优先级（数字越小优先级越高）"""
        # 紧急邮件优先级高
        if message.headers.get('X-Priority') == '1':
            return 0
        # 退回通知优先级高
        if 'mailer-daemon' in message.sender.lower():
            return 1
        return 5

class DeliveryEngine:
    """邮件投递引擎"""
    def __init__(self, queue: MailQueue):
        self.queue = queue
        self.workers = []
        self.max_workers = 10

    def start(self):
        """启动投递工作线程"""
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._delivery_loop, daemon=True)
            worker.start()
            self.workers.append(worker)

    def _delivery_loop(self):
        """投递主循环"""
        while True:
            message = self.queue.dequeue()
            if message:
                try:
                    self._deliver(message)
                    self.queue.mark_delivered(message.message_id)
                except TemporaryFailure as e:
                    delay = self._calculate_backoff(message.retry_count)
                    self.queue.mark_deferred(message.message_id, delay)
                except PermanentFailure as e:
                    self.queue.mark_bounced(message.message_id, str(e))
            else:
                time.sleep(1)  # 队列为空，等待

    def _deliver(self, message: EmailMessage):
        """投递单封邮件"""
        for recipient in message.recipients:
            domain = recipient.split('@')[1]
            mx_hosts = self._get_mx_records(domain)

            for mx in mx_hosts:
                try:
                    self._smtp_deliver(mx, message.sender, recipient, message)
                    return
                except TemporaryFailure:
                    continue
                except PermanentFailure:
                    raise

            raise TemporaryFailure(f"All MX servers for {domain} failed")

    def _smtp_deliver(self, host: str, sender: str, recipient: str, message: EmailMessage):
        """通过 SMTP 投递邮件"""
        import smtplib
        try:
            with smtplib.SMTP(host, 25, timeout=30) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.ehlo()
                smtp.sendmail(sender, [recipient], message.body)
        except smtplib.SMTPServerDisconnected:
            raise TemporaryFailure("Connection lost")
        except smtplib.SMTPRecipientsRefused:
            raise PermanentFailure("Recipient refused")

    def _get_mx_records(self, domain: str) -> list:
        """查询 MX 记录"""
        import dns.resolver
        try:
            mx_records = dns.resolver.resolve(domain, 'MX')
            return sorted([(r.preference, str(r.exchange)) for r in mx_records])
        except:
            return [(0, domain)]

    def _calculate_backoff(self, retry_count: int) -> int:
        """计算退避时间（指数退避）"""
        return min(300, 30 * (2 ** retry_count))  # 最大 5 分钟
```

<div data-component="SMTPFlowDemo"></div>

---

## 21.3 MIME 多用途互联网邮件扩展

### 21.3.1 MIME 概述

原始 SMTP 只支持 7 位 ASCII 文本，MIME（Multipurpose Internet Extensions）扩展了邮件的能力：

```
MIME 解决的问题:
  1. 非 ASCII 文本（中文、日文等）
  2. 二进制附件（图片、文档、视频）
  3. 多部分消息（文本 + HTML + 附件）
  4. 非文本头部（国际化）

MIME 头部:
  MIME-Version: 1.0
  Content-Type: ...
  Content-Transfer-Encoding: ...
  Content-Disposition: ...
```

### 21.3.2 Content-Type 类型

```
主要类型 (type/subtype):

text/plain          纯文本
text/html           HTML 文档
text/csv            CSV 数据

image/jpeg          JPEG 图片
image/png           PNG 图片
image/gif           GIF 图片

audio/mpeg          MP3 音频
audio/wav           WAV 音频

video/mp4           MP4 视频

application/octet-stream  任意二进制数据
application/pdf           PDF 文档
application/zip           ZIP 压缩包
application/json          JSON 数据

multipart/mixed           多部分混合（附件）
multipart/related         多部分关联（内嵌图片）
multipart/alternative     多部分替代（纯文本 + HTML）
multipart/form-data       表单数据
```

### 21.3.3 多部分消息格式

```
多部分邮件示例:

From: alice@example.com
To: bob@example.com
Subject: Meeting Notes
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="boundary123"

--boundary123
Content-Type: text/plain; charset=UTF-8
Content-Transfer-Encoding: 7bit

Hi Bob,

Please find the meeting notes attached.

Best,
Alice

--boundary123
Content-Type: application/pdf; name="notes.pdf"
Content-Transfer-Encoding: base64
Content-Disposition: attachment; filename="notes.pdf"

JVBERi0xLjQKMSAwIG9iago8PAovVHlwZSAvQ2F0YWxvZwovUGFn...

--boundary123--
```

### 21.3.4 Base64 编码

```python
import base64

def base64_encode(data: bytes) -> str:
    """Base64 编码"""
    encoded = base64.b64encode(data).decode('ascii')
    # 每 76 个字符换行（MIME 规范）
    lines = [encoded[i:i+76] for i in range(0, len(encoded), 76)]
    return '\r\n'.join(lines)

def base64_decode(encoded: str) -> bytes:
    """Base64 解码"""
    # 移除换行符
    clean = encoded.replace('\r\n', '').replace('\n', '')
    return base64.b64decode(clean)

# 示例
data = "Hello, 世界！".encode('utf-8')
encoded = base64_encode(data)
print(encoded)
# SGVsbG8sIOS4lueVjO+8IQ==

decoded = base64_decode(encoded)
print(decoded.decode('utf-8'))
# Hello, 世界！
```

### 21.3.5 Quoted-Printable 编码

```python
def quoted_printable_encode(text: str) -> str:
    """Quoted-Printable 编码"""
    result = []
    for char in text:
        code = ord(char)
        if code > 126 or code < 32 or char in '=?_':
            # 非 ASCII 字符或特殊字符，编码为 =XX
            for byte in char.encode('utf-8'):
                result.append(f'={byte:02X}')
        else:
            result.append(char)

    # 每行不超过 76 个字符
    encoded = ''.join(result)
    lines = []
    while len(encoded) > 76:
        # 在编码字符边界断行
        cut = 76
        while cut > 0 and encoded[cut - 1] == '=':
            cut -= 1
        lines.append(encoded[:cut] + '=')
        encoded = encoded[cut:]
    lines.append(encoded)
    return '\r\n'.join(lines)

# 示例
text = "Hello, 世界！"
encoded = quoted_printable_encode(text)
print(encoded)
# Hello, =E4=B8=96=E7=95=8C=EF=BC=81
```

<div data-component="MIMEDemo"></div>

### 21.3.6 MIME 编码/解码器的处理管线

```python
class MIMEProcessor:
    """MIME 编码/解码处理器"""
    def __init__(self):
        self.encoders = {
            'base64': self._base64_encode,
            'quoted-printable': self._qp_encode,
            '7bit': self._identity,
            '8bit': self._identity,
        }
        self.decoders = {
            'base64': self._base64_decode,
            'quoted-printable': self._qp_decode,
            '7bit': self._identity,
            '8bit': self._identity,
        }

    def encode_part(self, content: bytes, content_type: str, encoding: str) -> bytes:
        """编码 MIME 部分"""
        encoder = self.encoders.get(encoding, self._identity)
        return encoder(content)

    def decode_part(self, content: bytes, encoding: str) -> bytes:
        """解码 MIME 部分"""
        decoder = self.decoders.get(encoding, self._identity)
        return decoder(content)

    def create_multipart(self, parts: list, subtype: str = 'mixed') -> bytes:
        """创建多部分消息"""
        boundary = self._generate_boundary()
        body = f'Content-Type: multipart/{subtype}; boundary="{boundary}"\r\n\r\n'

        for part in parts:
            body += f'--{boundary}\r\n'
            body += f'Content-Type: {part["type"]}\r\n'
            if 'encoding' in part:
                body += f'Content-Transfer-Encoding: {part["encoding"]}\r\n'
            if 'disposition' in part:
                body += f'Content-Disposition: {part["disposition"]}\r\n'
            body += '\r\n'
            body += part['content'].decode('utf-8', errors='replace')
            body += '\r\n'

        body += f'--{boundary}--\r\n'
        return body.encode('utf-8')

    def parse_multipart(self, content: bytes, boundary: str) -> list:
        """解析多部分消息"""
        parts = []
        text = content.decode('utf-8', errors='replace')
        sections = text.split(f'--{boundary}')

        for section in sections[1:-1]:  # 跳过前后的边界
            if section.strip() == '':
                continue
            # 分离头部和内容
            header_end = section.find('\r\n\r\n')
            if header_end == -1:
                continue
            headers_text = section[:header_end]
            body = section[header_end + 4:]

            # 解析头部
            headers = self._parse_headers(headers_text)
            parts.append({
                'type': headers.get('Content-Type', 'text/plain'),
                'encoding': headers.get('Content-Transfer-Encoding', '7bit'),
                'content': body.encode('utf-8'),
            })

        return parts

    def _generate_boundary(self) -> str:
        import uuid
        return f'----=_Part_{uuid.uuid4().hex[:20]}'

    def _parse_headers(self, text: str) -> dict:
        headers = {}
        for line in text.split('\r\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                headers[key.strip()] = value.strip()
        return headers
```

---

## 21.4 POP3 协议

### 21.4.1 POP3 概述

POP3（Post Office Protocol version 3）是最简单的邮件接收协议：

```
POP3 特点:
  - 端口 110（明文）或 995（TLS）
  - 下载并删除模式（默认）
  - 简单，适合单设备使用
  - 不支持服务器端文件夹
  - 不支持部分下载
```

### 21.4.2 POP3 命令

```
POP3 会话示例:

S: +OK POP3 server ready
C: USER alice
S: +OK
C: PASS password123
S: +OK Logged in

C: STAT
S: +OK 3 10240    ← 3 封邮件，共 10240 字节

C: LIST
S: +OK 3 messages:
S: 1 3072
S: 2 4096
S: 3 3072
S: .

C: RETR 1
S: +OK 3072 octets
S: [邮件内容...]
S: .

C: DELE 1
S: +OK Message 1 deleted

C: QUIT
S: +OK POP3 server signing off
```

**POP3 命令表**：

| 命令 | 说明 |
|------|------|
| USER | 指定用户名 |
| PASS | 指定密码 |
| STAT | 获取邮箱状态（邮件数、总大小） |
| LIST | 列出所有邮件 |
| RETR n | 获取第 n 封邮件 |
| DELE n | 删除第 n 封邮件 |
| NOOP | 空操作（保持连接） |
| RSET | 重置删除标记 |
| QUIT | 结束会话 |

---

## 21.5 IMAP 协议

### 21.5.1 IMAP 概述

IMAP（Internet Message Access Protocol）是功能更强大的邮件接收协议：

```
IMAP 特点:
  - 端口 143（明文）或 993（TLS）
  - 邮件保留在服务器上
  - 支持多设备同步
  - 支持服务器端文件夹
  - 支持部分下载（节省带宽）
  - 支持邮件标记（已读、星标等）
```

### 21.5.2 IMAP vs POP3

```
┌──────────────────────┬──────────────────┬──────────────────┐
│       特性           │      POP3        │      IMAP        │
├──────────────────────┼──────────────────┼──────────────────┤
│ 邮件存储位置         │ 客户端           │ 服务器           │
│ 多设备同步           │ 不支持           │ 支持             │
│ 服务器文件夹         │ 不支持           │ 支持             │
│ 部分下载             │ 不支持           │ 支持             │
│ 服务器搜索           │ 不支持           │ 支持             │
│ 离线访问             │ 原生支持         │ 需要缓存         │
│ 服务器存储需求       │ 低               │ 高               │
│ 复杂度               │ 简单             │ 复杂             │
└──────────────────────┴──────────────────┴──────────────────┘
```

### 21.5.3 IMAP 命令

```
IMAP 会话示例:

S: * OK IMAP server ready
C: A001 LOGIN alice password123
S: A001 OK Logged in

C: A002 LIST "" "*"
S: * LIST (\HasNoChildren) "." "INBOX"
S: * LIST (\HasChildren) "." "Sent"
S: * LIST (\HasNoChildren) "." "Drafts"
S: A002 OK LIST completed

C: A003 SELECT INBOX
S: * 4 EXISTS
S: * 1 RECENT
S: * FLAGS (\Answered \Flagged \Deleted \Seen \Draft)
S: * OK [UIDVALIDITY 1234567890]
S: A003 OK SELECT completed

C: A004 FETCH 1 (BODY[HEADER])
S: * 1 FETCH (BODY[HEADER] {256}
S: From: bob@example.com
S: To: alice@example.com
S: Subject: Hello
S: )
S: A004 OK FETCH completed

C: A005 STORE 1 +FLAGS (\Seen)
S: * 1 FETCH (FLAGS (\Seen))
S: A005 OK STORE completed

C: A006 SEARCH FROM "bob@example.com"
S: * SEARCH 1 3
S: A006 OK SEARCH completed

C: A007 LOGOUT
S: * BYE IMAP server logging out
S: A007 OK LOGOUT completed
```

<div data-component="IMAPvsPOP3Demo"></div>

---

## 21.6 Webmail 架构

### 21.6.1 Webmail 系统架构

```
Webmail 架构:

  ┌──────────────┐
  │   浏览器      │
  │  (前端 SPA)  │
  └──────┬───────┘
         │ HTTPS (REST API / WebSocket)
         ▼
  ┌──────────────┐
  │   Web 服务器  │
  │  (后端 API)  │
  └──────┬───────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│ SMTP  │ │ IMAP  │
│ 客户端 │ │ 客户端 │
└───┬───┘ └───┬───┘
    │         │
    ▼         ▼
┌───────┐ ┌───────┐
│ SMTP  │ │ IMAP  │
│ 服务器 │ │ 服务器 │
└───────┘ └───────┘
```

### 21.6.2 Webmail 功能

```python
class WebmailService:
    """Webmail 服务"""
    def __init__(self):
        self.smtp_client = SMTPClient()
        self.imap_client = IMAPClient()
        self.user_sessions = {}

    def send_email(self, session_id: str, to: list, subject: str, body: str, attachments: list = None):
        """发送邮件"""
        user = self.user_sessions[session_id]
        message = self._compose_message(user, to, subject, body, attachments)
        self.smtp_client.send(user.smtp_server, user.email, to, message)

    def get_inbox(self, session_id: str, page: int = 1, page_size: int = 20):
        """获取收件箱"""
        user = self.user_sessions[session_id]
        return self.imap_client.list_messages(
            user.imap_server, user.email, user.password,
            folder='INBOX', offset=(page-1)*page_size, limit=page_size
        )

    def read_email(self, session_id: str, message_id: str):
        """读取邮件"""
        user = self.user_sessions[session_id]
        return self.imap_client.fetch_message(
            user.imap_server, user.email, user.password, message_id
        )

    def search_emails(self, session_id: str, query: str):
        """搜索邮件"""
        user = self.user_sessions[session_id]
        return self.imap_client.search(
            user.imap_server, user.email, user.password, query
        )
```

<div data-component="WebmailDemo"></div>

---

## 21.7 垃圾邮件过滤

### 21.7.1 垃圾邮件问题

```
垃圾邮件（Spam）的影响:
  - 占全球邮件流量的 45-85%
  - 浪费带宽和存储资源
  - 包含钓鱼链接和恶意软件
  - 降低用户工作效率
```

### 21.7.2 贝叶斯垃圾邮件过滤器

```python
import re
import math
from collections import defaultdict
from typing import Dict, Tuple

class BayesianSpamFilter:
    """贝叶斯垃圾邮件过滤器"""
    def __init__(self):
        self.word_counts = {
            'spam': defaultdict(int),    # 垃圾邮件中的词频
            'ham': defaultdict(int),     # 正常邮件中的词频
        }
        self.total_words = {'spam': 0, 'ham': 0}
        self.email_count = {'spam': 0, 'ham': 0}

    def train(self, email: str, label: str):
        """训练分类器"""
        words = self._tokenize(email)
        self.email_count[label] += 1

        for word in words:
            self.word_counts[label][word] += 1
            self.total_words[label] += 1

    def classify(self, email: str) -> Tuple[str, float]:
        """分类邮件"""
        words = self._tokenize(email)
        spam_score = 0
        ham_score = 0

        total_emails = sum(self.email_count.values())
        spam_prior = self.email_count['spam'] / total_emails
        ham_prior = self.email_count['ham'] / total_emails

        # 计算对数概率
        spam_score += math.log(spam_prior)
        ham_score += math.log(ham_prior)

        for word in words:
            # 使用拉普拉斯平滑
            spam_prob = (self.word_counts['spam'].get(word, 0) + 1) / \
                       (self.total_words['spam'] + len(self.word_counts['spam']))
            ham_prob = (self.word_counts['ham'].get(word, 0) + 1) / \
                      (self.total_words['ham'] + len(self.word_counts['ham']))

            spam_score += math.log(spam_prob)
            ham_score += math.log(ham_prob)

        # 计算置信度
        if spam_score > ham_score:
            confidence = 1 / (1 + math.exp(ham_score - spam_score))
            return 'spam', confidence
        else:
            confidence = 1 / (1 + math.exp(spam_score - ham_score))
            return 'ham', confidence

    def _tokenize(self, text: str) -> list:
        """分词"""
        # 转换为小写，移除标点
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        # 分词并过滤停用词
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at'}
        words = [w for w in text.split() if w not in stop_words and len(w) > 2]
        return words

    def get_spam_indicators(self, email: str, top_n: int = 10) -> list:
        """获取垃圾邮件指标词"""
        words = self._tokenize(email)
        indicators = []

        for word in set(words):
            spam_count = self.word_counts['spam'].get(word, 0) + 1
            ham_count = self.word_counts['ham'].get(word, 0) + 1
            spam_prob = spam_count / (spam_count + ham_count)
            indicators.append((word, spam_prob))

        indicators.sort(key=lambda x: x[1], reverse=True)
        return indicators[:top_n]
```

### 21.7.3 SPF（Sender Policy Framework）

```
SPF 工作原理:
  1. 发件域名在 DNS 中发布 SPF 记录
  2. 收件服务器查询发件域名的 SPF 记录
  3. 验证发件 IP 是否被授权

SPF 记录示例:
  example.com. TXT "v=spf1 ip4:192.168.1.0/24 include:_spf.google.com ~all"

  v=spf1: SPF 版本 1
  ip4:192.168.1.0/24: 允许的 IP 范围
  include:_spf.google.com: 包含 Google 的 SPF 记录
  ~all: 软失败（不在列表中的 IP 标记为可疑）

SPF 验证结果:
  pass: 发件 IP 在授权列表中
  fail: 发件 IP 不在授权列表中
  softfail: 发件 IP 不在列表中，但不强制拒绝
  neutral: 无法判断
```

### 21.7.4 DKIM（DomainKeys Identified Mail）

```
DKIM 工作原理:
  1. 发件服务器用私钥对邮件头部和正文签名
  2. 签名添加到 DKIM-Signature 头部
  3. 收件服务器从 DNS 获取公钥
  4. 使用公钥验证签名

DKIM 签名示例:
  DKIM-Signature: v=1; a=rsa-sha256; d=example.com; s=selector;
    c=relaxed/simple; h=From:To:Subject:Date;
    bh=abc123...; b=xyz789...;

  v=1: DKIM 版本
  a=rsa-sha256: 签名算法
  d=example.com: 发件域名
  s=selector: 选择器（用于查找公钥）
  c=relaxed/simple: 规范化方法
  h=From:To:Subject:Date: 签名的头部
  bh=: 正文哈希
  b=: 签名值

DNS 中的公钥:
  selector._domainkey.example.com TXT "v=DKIM1; k=rsa; p=MIGfMA..."
```

### 21.7.5 DMARC（Domain-based Message Authentication）

```
DMARC 工作原理:
  1. 基于 SPF 和 DKIM 的验证结果
  2. 检查域名对齐（From 头部与 SPF/DKIM 域名匹配）
  3. 根据策略决定如何处理未通过验证的邮件

DMARC 记录示例:
  _dmarc.example.com TXT "v=DMARC1; p=quarantine; rua=mailto:dmarc@example.com; pct=100"

  v=DMARC1: DMARC 版本
  p=quarantine: 策略（quarantine=隔离，reject=拒绝，none=仅报告）
  rua=mailto:...: 聚合报告接收地址
  pct=100: 应用策略的百分比

DMARC 验证流程:
  1. 检查 SPF 验证结果
  2. 检查 DKIM 验证结果
  3. 检查域名对齐
  4. 应用 DMARC 策略
```

<div data-component="SpamFilterDemo"></div>

---

## 21.8 邮件传输代理的路由查找逻辑

### 21.8.1 MTA 路由查找

```python
class MTARouter:
    """邮件传输代理路由查找"""
    def __init__(self):
        self.relay_domains = set()    # 本地域名
        self.transport_table = {}     # 路由表
        self.dns_cache = DNSCache()

    def route_message(self, sender: str, recipient: str) -> list:
        """路由邮件到目标服务器"""
        domain = recipient.split('@')[1]

        # 1. 检查是否是本地域名
        if domain in self.relay_domains:
            return [('local', 'localhost', 25)]

        # 2. 检查路由表（手动配置的路由）
        if domain in self.transport_table:
            entry = self.transport_table[domain]
            return [(entry['type'], entry['host'], entry['port'])]

        # 3. 查询 MX 记录
        mx_records = self._lookup_mx(domain)
        if mx_records:
            return [('smtp', mx, 25) for _, mx in mx_records]

        # 4. 回退到 A 记录
        a_record = self._lookup_a(domain)
        if a_record:
            return [('smtp', domain, 25)]

        # 5. 无法路由
        raise RoutingError(f"No route to {domain}")

    def _lookup_mx(self, domain: str) -> list:
        """查询 MX 记录"""
        cached = self.dns_cache.get(domain, 'MX')
        if cached:
            return cached

        try:
            import dns.resolver
            mx_records = dns.resolver.resolve(domain, 'MX')
            result = sorted([(r.preference, str(r.exchange)) for r in mx_records])
            self.dns_cache.put(domain, 'MX', result, ttl=3600)
            return result
        except:
            return []

    def _lookup_a(self, domain: str) -> Optional[str]:
        """查询 A 记录"""
        cached = self.dns_cache.get(domain, 'A')
        if cached:
            return cached[0]

        try:
            import dns.resolver
            a_records = dns.resolver.resolve(domain, 'A')
            result = [str(r) for r in a_records]
            self.dns_cache.put(domain, 'A', result, ttl=300)
            return result[0] if result else None
        except:
            return None
```

<div data-component="MTARoutingDemo"></div>

---

## 21.9 邮件安全

### 21.9.1 PGP（Pretty Good Privacy）

```
PGP 加密流程:
  1. 发件人获取收件人的公钥
  2. 生成随机会话密钥
  3. 用会话密钥加密邮件内容（对称加密）
  4. 用收件人公钥加密会话密钥（非对称加密）
  5. 将加密的会话密钥和加密的邮件一起发送

PGP 签名流程:
  1. 计算邮件内容的哈希值
  2. 用发件人私钥加密哈希值（签名）
  3. 将签名附加到邮件中
  4. 收件人用发件人公钥解密签名
  5. 比较哈希值，验证完整性和身份
```

### 21.9.2 S/MIME

```
S/MIME 与 PGP 的区别:
  ┌──────────────────────┬──────────────────┬──────────────────┐
  │       特性           │      PGP         │     S/MIME       │
  ├──────────────────────┼──────────────────┼──────────────────┤
  │ 信任模型             │ Web of Trust     │ CA 层次结构      │
  │ 证书格式             │ PGP 公钥         │ X.509 证书       │
  │ 密钥管理             │ 自管理           │ CA 管理          │
  │ 兼容性               │ 需要 PGP 软件    │ 邮件客户端原生   │
  │ 企业使用             │ 较少             │ 广泛             │
  └──────────────────────┴──────────────────┴──────────────────┘
```

<div data-component="EmailSecurityDemo"></div>

---

## 21.10 章节小结

本章详细介绍了电子邮件系统的各个方面：

1. **系统架构**：用户代理、MTA、MDA 的协作
2. **SMTP 协议**：命令、响应、邮件传输流程
3. **MIME 标准**：Content-Type、Base64、Quoted-Printable 编码
4. **POP3 协议**：简单的邮件下载协议
5. **IMAP 协议**：功能强大的邮件访问协议
6. **Webmail**：基于 Web 的邮件系统架构
7. **垃圾邮件过滤**：贝叶斯分类器、SPF、DKIM、DMARC
8. **邮件安全**：PGP 和 S/MIME 的加密与签名

<div data-component="ChapterSummary"></div>
<div data-component="KnowledgeCheck"></div>

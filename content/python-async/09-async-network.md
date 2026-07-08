---
title: "异步网络请求"
description: "使用 aiohttp 进行异步 HTTP 请求，掌握并发请求、错误处理、流式下载等实战技巧"
updated: "2026-07-07"
---

# 异步网络请求

> **学习目标**：
> - 理解为什么网络请求是异步编程的最佳应用场景
> - 掌握 aiohttp 的基本用法，包括 ClientSession、GET/POST 请求
> - 能够使用 gather 实现并发多个请求
> - 掌握网络请求中的错误处理和超时控制
> - 使用 Semaphore 限制并发数，避免目标服务器过载
> - 实现流式文件下载
> - 构建一个完整的批量爬虫实战项目
> - 了解 httpx 作为替代方案的优劣

网络请求是异步编程的"杀手级应用"。一个 HTTP 请求从发出到收到响应，通常需要几十毫秒到几秒的时间，而这段时间里 CPU 几乎什么都不做——它在等网络数据到达。这正是异步编程最擅长的事情。

---

## 9.1 为什么网络请求适合异步

在所有常见的 I/O 操作中，网络请求的延迟是最高的：

| 操作类型 | 典型延迟 | 等待期间 CPU 在做什么 |
|---------|---------|-------------------|
| 内存读写 | 纳秒级（~100ns） | 什么都不等 |
| SSD 读写 | 微秒级（~100μs） | 等磁盘响应 |
| 网络请求（本地） | 毫秒级（~1ms） | 等网络响应 |
| 网络请求（跨城） | 毫秒级（~10-50ms） | 等网络响应 |
| 网络请求（跨洲） | 百毫秒级（~100-300ms） | 等网络响应 |
| 网络请求（慢接口） | 秒级（1-10s） | 等网络响应 |

一个典型的 HTTP 请求过程：

```
客户端                          服务器
  │                               │
  ├── DNS 解析 ─────────────────→ │     ~10-100ms
  │                               │
  ├── TCP 握手 ─────────────────→ │     ~10-50ms
  │                               │
  ├── TLS 握手 ─────────────────→ │     ~20-100ms（HTTPS）
  │                               │
  ├── 发送请求数据 ─────────────→ │     ~1-10ms
  │                               │
  │   （CPU 空闲，等待响应）        ├── 处理请求
  │   （CPU 空闲，等待响应）        │     ~10ms-10s
  │                               │
  │←──────────────── 返回响应 ────┤     ~1-10ms
  │                               │
  ├── 解析响应数据                 │
```

在整个过程中，真正"工作"的时间（发送数据、解析响应）可能只有几毫秒，但等待的时间可能是几百毫秒甚至几秒。

**这就是为什么网络请求特别适合异步：**

```python
# 同步方式：一次只能等一个请求
import requests

def fetch_all(urls):
    results = []
    for url in urls:
        resp = requests.get(url)  # 每个请求等 200ms
        results.append(resp.text)
    return results

# 10 个请求 × 200ms = 2000ms（2秒）
```

```python
# 异步方式：同时等所有请求
import aiohttp
import asyncio

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [session.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return [await r.text() for r in responses]

# 10 个请求同时发出，总时间 ≈ max(200ms) = 200ms
```

性能差距是 10 倍。如果请求更多、延迟更大，差距会更明显。

<div data-component="AsyncHttpRequestFlow"></div>

---

</div>
## 9.2 aiohttp 基础

`aiohttp` 是 Python 生态中最流行的异步 HTTP 客户端库。它的核心设计思想是：**整个连接生命周期由一个 `ClientSession` 管理**。

### 为什么需要 ClientSession？

很多人第一次用 aiohttp 时会问：为什么不能像 requests 那样直接 `aiohttp.get(url)`？

原因有二：

1. **连接池复用**：ClientSession 内部维护一个连接池。对同一个服务器的多次请求可以复用 TCP 连接，省去重复的握手开销。
2. **统一配置**：超时、认证、headers、cookie 等配置只需要在 Session 上设置一次。

```python
import aiohttp
import asyncio

async def main():
    # 创建 session（相当于打开了一个浏览器窗口）
    async with aiohttp.ClientSession() as session:
        # 用这个 session 发请求（相当于在这个窗口里访问网页）
        async with session.get('https://httpbin.org/get') as response:
            print(response.status)   # 状态码
            print(await response.text())  # 响应体

asyncio.run(main())
```

### aiohttp 的层次结构

```
ClientSession（会话）
  ├── 连接池（自动管理）
  ├── Cookie Jar（自动管理）
  ├── 超时配置
  ├── Headers 配置
  │
  ├── .get(url)    → ClientResponse
  ├── .post(url)   → ClientResponse
  ├── .put(url)    → ClientResponse
  ├── .delete(url) → ClientResponse
  ├── .patch(url)  → ClientResponse
  └── .ws_connect(url) → ClientWebSocketResponse

ClientResponse（响应）
  ├── .status       → int（状态码）
  ├── .headers      → CIMultiDictProxy（响应头）
  ├── .content      → StreamReader（流式读取）
  ├── .text()       → str（文本响应体）
  ├── .json()       → dict（JSON 响应体）
  ├── .read()       → bytes（二进制响应体）
  └── .release()    → None（释放连接）
```

### ClientSession 的生命周期

```python
# 推荐写法：用 async with 管理生命周期
async with aiohttp.ClientSession() as session:
    # session 在这个块内有效
    async with session.get(url) as resp:
        data = await resp.text()
# 离开 with 块后，session 自动关闭，连接池释放
```

```python
# 也可以手动管理（不推荐）
session = aiohttp.ClientSession()
try:
    async with session.get(url) as resp:
        data = await resp.text()
finally:
    await session.close()  # 必须手动关闭
```

**最佳实践：每个程序创建一个 Session，在最顶层用 `async with` 管理。**

---

## 9.3 安装和基本用法

### 安装

```bash
# 基础安装
pip install aiohttp

# 如果需要更快的 DNS 解析
pip install aiohttp[speedups]

# 如果需要 HTTPS 证书验证
pip install aiohttp[ssl]
```

验证安装：

```python
import aiohttp
print(aiohttp.__version__)  # 应该输出 3.x
```

### 最简单的例子

```python
import aiohttp
import asyncio

async def hello():
    async with aiohttp.ClientSession() as session:
        async with session.get('https://httpbin.org/get') as resp:
            status = resp.status
            body = await resp.text()
            print(f"状态码: {status}")
            print(f"响应体前 100 字符: {body[:100]}")

asyncio.run(hello())
```

输出类似：

```
状态码: 200
响应体前 100 字符: {
  "args": {},
  "headers": {
    "Host": "httpbin.org",
    "User-Agent": "Python-ai
```

### 解析 JSON 响应

```python
import aiohttp
import asyncio

async def get_json():
    async with aiohttp.ClientSession() as session:
        async with session.get('https://httpbin.org/json') as resp:
            data = await resp.json()  # 直接解析 JSON
            print(type(data))  # <class 'dict'>
            print(data['slideshow']['title'])

asyncio.run(get_json())
```

### 带查询参数

```python
async def search():
    params = {'q': 'python async', 'page': 1}
    async with aiohttp.ClientSession() as session:
        async with session.get('https://httpbin.org/get', params=params) as resp:
            data = await resp.json()
            print(data['args'])  # {'q': 'python async', 'page': '1'}
```

### 自定义 Headers

```python
async def with_headers():
    headers = {
        'Authorization': 'Bearer my-token',
        'Accept': 'application/json',
    }
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get('https://httpbin.org/headers') as resp:
            data = await resp.json()
            print(data['headers']['Authorization'])
```

---

## 9.4 单个请求示例

让我们用一个完整的例子展示 GET 请求的方方面面：

```python
import aiohttp
import asyncio

async def fetch_single_url():
    """演示单个 GET 请求的各种细节"""
    url = 'https://httpbin.org/get'
    
    # 自定义 headers
    headers = {
        'User-Agent': 'MyAsyncApp/1.0',
        'Accept': 'application/json',
    }
    
    # 查询参数
    params = {
        'name': '张三',
        'lang': 'python',
    }
    
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url, params=params) as resp:
            # 1. 检查状态码
            print(f"状态码: {resp.status}")
            print(f"原因: {resp.reason}")
            
            # 2. 查看响应头
            print(f"Content-Type: {resp.content_type}")
            print(f"Content-Length: {resp.content_length}")
            
            # 3. 获取响应 URL（可能因重定向而变化）
            print(f"最终 URL: {resp.url}")
            
            # 4. 检查是否成功
            resp.raise_for_status()  # 非 2xx 会抛异常
            
            # 5. 读取响应体
            if resp.content_type == 'application/json':
                data = await resp.json()
                print(f"JSON 数据: {data}")
            else:
                text = await resp.text()
                print(f"文本数据: {text[:200]}")

asyncio.run(fetch_single_url())
```

### 读取二进制数据

```python
async def fetch_binary():
    """下载一张图片（二进制数据）"""
    url = 'https://httpbin.org/image/png'
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                image_data = await resp.read()  # 返回 bytes
                print(f"图片大小: {len(image_data)} 字节")
                
                with open('test.png', 'wb') as f:
                    f.write(image_data)
                print("图片已保存")

asyncio.run(fetch_binary())
```

### 查看请求信息

```python
async def inspect_request():
    """查看实际发出的请求信息"""
    async with aiohttp.ClientSession() as session:
        async with session.get('https://httpbin.org/anything',
                               params={'key': 'value'}) as resp:
            data = await resp.json()
            
            print(f"请求方法: {data['method']}")
            print(f"请求 URL: {data['url']}")
            print(f"请求头: {data['headers']}")
            print(f"查询参数: {data['args']}")
            print(f"来源 IP: {data['origin']}")

asyncio.run(inspect_request())
```

### 使用 HEAD 请求检查资源

```python
async def check_resource():
    """用 HEAD 请求检查资源是否存在（不下载内容）"""
    url = 'https://httpbin.org/get'
    
    async with aiohttp.ClientSession() as session:
        async with session.head(url) as resp:
            print(f"状态码: {resp.status}")
            print(f"Content-Type: {resp.headers.get('Content-Type')}")
            print(f"Content-Length: {resp.headers.get('Content-Length')}")
            # HEAD 请求没有响应体

asyncio.run(check_resource())
```

---

## 9.5 并发多个请求

这是 aiohttp 真正发光的地方：同时发出多个请求，然后等待所有结果。

### 使用 gather

```python
import aiohttp
import asyncio
import time

async def fetch(session, url):
    """获取单个 URL 的内容"""
    async with session.get(url) as resp:
        text = await resp.text()
        return resp.status, len(text)

async def main():
    urls = [
        'https://httpbin.org/get',
        'https://httpbin.org/post',
        'https://httpbin.org/put',
        'https://httpbin.org/delete',
        'https://httpbin.org/headers',
    ]
    
    start = time.time()
    
    async with aiohttp.ClientSession() as session:
        # 创建所有任务
        tasks = [fetch(session, url) for url in urls]
        # 并发执行
        results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    
    for url, (status, length) in zip(urls, results):
        print(f"{url}: status={status}, length={length}")
    
    print(f"\n总共用时: {elapsed:.2f}秒")
    print(f"如果串行: {len(urls) * 0.2:.2f}秒（估算）")

asyncio.run(main())
```

<div data-component="ConcurrentRequestsDemo"></div>

### 带进度的并发请求

```python
async def fetch_with_progress(session, url, index, total):
    """获取并打印进度"""
    async with session.get(url) as resp:
        text = await resp.text()
        print(f"[{index + 1}/{total}] 完成: {url} ({resp.status})")
        return text

async def fetch_all_with_progress():
    urls = [f'https://httpbin.org/get?page={i}' for i in range(20)]
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_with_progress(session, url, i, len(urls))
            for i, url in enumerate(urls)
        ]
        results = await asyncio.gather(*tasks)
    
    print(f"\n全部完成，共获取 {len(results)} 个页面")

asyncio.run(fetch_all_with_progress())
```

### 用 asyncio.create_task 手动管理

```python
async def manual_task_management():
    """手动创建任务，更灵活地控制"""
    async with aiohttp.ClientSession() as session:
        # 创建任务
        task1 = asyncio.create_task(fetch(session, 'https://httpbin.org/get'))
        task2 = asyncio.create_task(fetch(session, 'https://httpbin.org/headers'))
        task3 = asyncio.create_task(fetch(session, 'https://httpbin.org/ip'))
        
        # 等待所有任务完成
        results = await asyncio.gather(task1, task2, task3)
        
        for status, length in results:
            print(f"status={status}, length={length}")

asyncio.run(manual_task_management())
```

### as_completed：先完成先处理

```python
async def process_as_completed():
    """先完成的请求先处理"""
    urls = [
        'https://httpbin.org/delay/1',
        'https://httpbin.org/delay/3',
        'https://httpbin.org/delay/2',
        'https://httpbin.org/get',
    ]
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        
        # as_completed 返回一个迭代器
        # 先完成的 future 先被 yield
        for coro in asyncio.as_completed(tasks):
            status, length = await coro
            print(f"完成: status={status}, length={length}")

asyncio.run(process_as_completed())
```

注意 `as_completed` 的顺序与原始列表不同——它按照"谁先完成"的顺序返回。

</div>
---

## 9.6 错误处理

网络请求充满了不确定性。服务器可能宕机、网络可能中断、响应可能超时。健壮的错误处理是异步网络编程的基本功。

### 常见异常类型

```python
import aiohttp
import asyncio

async def handle_errors():
    url = 'https://httpbin.org/get'
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                resp.raise_for_status()  # 非 2xx 抛 ClientResponseError
                data = await resp.text()
                
    except aiohttp.ClientResponseError as e:
        # HTTP 错误（4xx, 5xx）
        print(f"HTTP 错误: {e.status} {e.message}")
        
    except aiohttp.ClientConnectionError as e:
        # 连接错误（DNS 解析失败、连接被拒绝等）
        print(f"连接错误: {e}")
        
    except aiohttp.ClientError as e:
        # aiohttp 所有客户端错误的基类
        print(f"客户端错误: {e}")
        
    except asyncio.TimeoutError:
        # 超时
        print("请求超时")
        
    except Exception as e:
        # 其他未知错误
        print(f"未知错误: {e}")
```

<div data-component="ErrorHandlingDemo"></div>

### 超时控制

超时是最常用的错误处理机制。aiohttp 支持多种超时设置：

```python
async def timeout_example():
    # 方法 1：整个请求的总超时
    timeout = aiohttp.ClientTimeout(total=10)  # 10 秒总超时
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get('https://httpbin.org/delay/5') as resp:
            print(await resp.text())
    
    # 方法 2：分阶段超时
    timeout = aiohttp.ClientTimeout(
        total=30,         # 整个请求最长 30 秒
        connect=10,       # 建立连接最长 10 秒
        sock_read=10,     # 读取响应最长 10 秒
        sock_connect=5,   # TCP 连接最长 5 秒
    )
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get('https://httpbin.org/delay/5') as resp:
            print(await resp.text())

asyncio.run(timeout_example())
```

### 单个请求的超时覆盖

```python
async def per_request_timeout():
    """为单个请求覆盖默认超时"""
    default_timeout = aiohttp.ClientTimeout(total=10)
    
    async with aiohttp.ClientSession(timeout=default_timeout) as session:
        # 这个请求使用默认 10 秒超时
        async with session.get('https://httpbin.org/get') as resp:
            print("普通请求:", resp.status)
        
        # 这个请求需要更长的超时
        long_timeout = aiohttp.ClientTimeout(total=60)
        async with session.get(
            'https://httpbin.org/delay/30',
            timeout=long_timeout
        ) as resp:
            print("长请求:", resp.status)
        
        # 这个请求只需要很短的超时
        short_timeout = aiohttp.ClientTimeout(total=2)
        try:
            async with session.get(
                'https://httpbin.org/delay/5',
                timeout=short_timeout
            ) as resp:
                print("短请求:", resp.status)
        except asyncio.TimeoutError:
            print("短请求超时了（预期之中）")

asyncio.run(per_request_timeout())
```

### 带重试的请求

```python
async def fetch_with_retry(session, url, max_retries=3, delay=1):
    """带重试的请求"""
    for attempt in range(max_retries):
        try:
            async with session.get(url) as resp:
                resp.raise_for_status()
                return await resp.text()
                
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt < max_retries - 1:
                wait = delay * (2 ** attempt)  # 指数退避
                print(f"请求失败 ({e})，{wait}秒后重试...")
                await asyncio.sleep(wait)
            else:
                print(f"请求失败，已重试 {max_retries} 次，放弃")
                raise

async def retry_demo():
    async with aiohttp.ClientSession() as session:
        # 这个 URL 会返回 500 错误
        try:
            text = await fetch_with_retry(
                session,
                'https://httpbin.org/status/500',
                max_retries=3,
                delay=0.5
            )
        except aiohttp.ClientResponseError:
            print("最终失败")

asyncio.run(retry_demo())
```
</div>

---

## 9.7 限制并发数

虽然异步可以同时发起大量请求，但不受限制的并发会导致：
1. 目标服务器过载（可能封禁你的 IP）
2. 本地文件描述符耗尽
3. 内存占用过高

### 使用 Semaphore

```python
import aiohttp
import asyncio
import time

async def fetch_limited(sem, session, url):
    """用 Semaphore 限制并发"""
    async with sem:  # 获取信号量（可能阻塞）
        print(f"开始请求: {url}")
        async with session.get(url) as resp:
            text = await resp.text()
            print(f"完成请求: {url} ({len(text)} bytes)")
            return resp.status, len(text)

async def limited_concurrent_requests():
    urls = [f'https://httpbin.org/delay/1' for _ in range(20)]
    
    # 限制最多 5 个并发
    sem = asyncio.Semaphore(5)
    
    start = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_limited(sem, session, url) for url in urls]
        results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    print(f"\n20 个请求，最多 5 并发，用时: {elapsed:.2f}秒")
    # 理论时间: ceil(20/5) × 1秒 = 4秒

asyncio.run(limited_concurrent_requests())
```

<div data-component="RateLimiterDemo"></div>

### 封装一个并发控制器

```python
class ConcurrencyLimiter:
    """并发控制器"""
    
    def __init__(self, max_concurrent):
        self.sem = asyncio.Semaphore(max_concurrent)
    
    async def run(self, coro):
        """运行一个协程，受并发限制"""
        async with self.sem:
            return await coro

async def controlled_requests():
    limiter = ConcurrencyLimiter(max_concurrent=3)
    
    async def slow_request(session, url):
        async with session.get(url) as resp:
            return resp.status
    
    async with aiohttp.ClientSession() as session:
        urls = [f'https://httpbin.org/delay/1' for _ in range(10)]
        tasks = [limiter.run(slow_request(session, url)) for url in urls]
        results = await asyncio.gather(*tasks)
        print(f"结果: {results}")

asyncio.run(controlled_requests())
```

### 带速率限制的并发控制

```python
import time

class RateLimiter:
    """速率限制器：限制每秒请求数"""
    
    def __init__(self, requests_per_second):
        self.rate = requests_per_second
        self.tokens = requests_per_second
        self.last_refill = time.monotonic()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.rate,
                self.tokens + elapsed * self.rate
            )
            self.last_refill = now
            
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1

async def rate_limited_fetch():
    limiter = RateLimiter(requests_per_second=5)
    
    async def fetch(session, url):
        await limiter.acquire()
        async with session.get(url) as resp:
            return resp.status
    
    async with aiohttp.ClientSession() as session:
        urls = [f'https://httpbin.org/get?id={i}' for i in range(15)]
        tasks = [fetch(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        print(f"完成 {len(results)} 个请求")

asyncio.run(rate_limited_fetch())
```
</div>

---

## 9.8 会话管理

正确管理 ClientSession 的生命周期非常重要。不当的管理会导致资源泄漏。

### 一个 Session 多个请求

```python
# 正确：一个 session 复用多个请求
async def correct_session_usage():
    async with aiohttp.ClientSession() as session:
        # 所有请求共享同一个连接池
        r1 = await session.get('https://httpbin.org/get')
        r2 = await session.get('https://httpbin.org/headers')
        r3 = await session.get('https://httpbin.org/ip')
        
        # 注意：每个 response 也需要关闭
        async with r1:
            data1 = await r1.text()
        async with r2:
            data2 = await r2.text()
        async with r3:
            data3 = await r3.text()
        
        print(data1[:50], data2[:50], data3[:50])
```

### 全局 Session 模式

```python
# 创建一个全局 session，在整个应用中复用
_session = None

async def get_session():
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session

async def fetch(url):
    session = await get_session()
    async with session.get(url) as resp:
        return await resp.text()

async def main():
    try:
        results = await asyncio.gather(
            fetch('https://httpbin.org/get'),
            fetch('https://httpbin.org/headers'),
            fetch('https://httpbin.org/ip'),
        )
        print(len(results))
    finally:
        if _session and not _session.closed:
            await _session.close()

asyncio.run(main())
```

### 配置 Session

```python
async def configured_session():
    # 连接器配置
    connector = aiohttp.TCPConnector(
        limit=100,          # 总连接数上限
        limit_per_host=10,  # 每个主机的连接数上限
        ttl_dns_cache=300,  # DNS 缓存 TTL（秒）
        enable_cleanup_closed=True,  # 自动清理关闭的连接
    )
    
    # 超时配置
    timeout = aiohttp.ClientTimeout(
        total=30,
        connect=10,
        sock_read=10,
    )
    
    # Cookie 配置
    jar = aiohttp.CookieJar(unsafe=True)  # 允许 IP 地址的 cookie
    
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        cookie_jar=jar,
        headers={'User-Agent': 'MyApp/1.0'},
    ) as session:
        async with session.get('https://httpbin.org/get') as resp:
            print(resp.status)

asyncio.run(configured_session())
```

---

## 9.9 POST 请求和 JSON

### 发送 JSON 数据

```python
async def post_json():
    """发送 JSON 数据"""
    data = {
        'username': 'zhangsan',
        'password': 'secret123',
        'email': 'zhangsan@example.com',
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            'https://httpbin.org/post',
            json=data,  # 自动设置 Content-Type: application/json
        ) as resp:
            result = await resp.json()
            print(f"状态码: {resp.status}")
            print(f"服务器收到的 JSON: {result['json']}")
            print(f"Content-Type: {result['headers']['Content-Type']}")

asyncio.run(post_json())
```

### 发送表单数据

```python
async def post_form():
    """发送表单数据（application/x-www-form-urlencoded）"""
    data = aiohttp.FormData()
    data.add_field('username', 'zhangsan')
    data.add_field('password', 'secret123')
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            'https://httpbin.org/post',
            data=data,
        ) as resp:
            result = await resp.json()
            print(f"服务器收到的表单: {result['form']}")

asyncio.run(post_form())
```

### 上传文件

```python
async def upload_file():
    """上传文件"""
    # 创建一个测试文件
    with open('test.txt', 'w') as f:
        f.write('Hello, async world!')
    
    data = aiohttp.FormData()
    data.add_field(
        'file',
        open('test.txt', 'rb'),
        filename='test.txt',
        content_type='text/plain',
    )
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            'https://httpbin.org/post',
            data=data,
        ) as resp:
            result = await resp.json()
            print(f"上传的文件: {result['files']}")

asyncio.run(upload_file())
```

### PUT、DELETE、PATCH 请求

```python
async def other_methods():
    async with aiohttp.ClientSession() as session:
        # PUT
        async with session.put(
            'https://httpbin.org/put',
            json={'key': 'value'}
        ) as resp:
            print(f"PUT: {resp.status}")
        
        # DELETE
        async with session.delete(
            'https://httpbin.org/delete'
        ) as resp:
            print(f"DELETE: {resp.status}")
        
        # PATCH
        async with session.patch(
            'https://httpbin.org/patch',
            json={'update': 'data'}
        ) as resp:
            print(f"PATCH: {resp.status}")

asyncio.run(other_methods())
```

### 带认证的请求

```python
async def authenticated_request():
    """基本认证"""
    async with aiohttp.ClientSession() as session:
        auth = aiohttp.BasicAuth('user', 'password')
        async with session.get(
            'https://httpbin.org/basic-auth/user/password',
            auth=auth,
        ) as resp:
            result = await resp.json()
            print(f"认证结果: {result}")

asyncio.run(authenticated_request())
```

---

## 9.10 下载文件

大文件下载需要用流式读取，避免一次性加载整个文件到内存。

### 流式下载

```python
import aiohttp
import asyncio

async def download_file(url, filename):
    """流式下载文件"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            
            total_size = int(resp.headers.get('Content-Length', 0))
            downloaded = 0
            
            with open(filename, 'wb') as f:
                async for chunk in resp.content.iter_chunked(8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size:
                        progress = downloaded / total_size * 100
                        print(f"\r下载进度: {progress:.1f}% "
                              f"({downloaded}/{total_size})", end='')
            
            print(f"\n下载完成: {filename}")

asyncio.run(download_file(
    'https://httpbin.org/bytes/102400',
    'test_download.bin'
))
```

<div data-component="DownloadProgressDemo"></div>

### 带进度条的下载

```python
async def download_with_progress(url, filename):
    """带详细进度的下载"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            
            total = int(resp.headers.get('Content-Length', 0))
            downloaded = 0
            start_time = time.time()
            
            with open(filename, 'wb') as f:
                async for chunk in resp.content.iter_chunked(8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    elapsed = time.time() - start_time
                    speed = downloaded / elapsed if elapsed > 0 else 0
                    
                    bar_length = 40
                    filled = int(bar_length * downloaded / total)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    
                    print(f'\r{bar} {downloaded}/{total} '
                          f'({speed/1024:.1f} KB/s)', end='')
            
            total_time = time.time() - start_time
            print(f'\n下载完成，用时 {total_time:.2f}秒')
```

### 同时下载多个文件

```python
async def download_multiple():
    """并发下载多个文件"""
    files = [
        ('https://httpbin.org/bytes/10240', 'file1.bin'),
        ('https://httpbin.org/bytes/20480', 'file2.bin'),
        ('https://httpbin.org/bytes/30720', 'file3.bin'),
    ]
    
    async def download_one(session, url, filename):
        async with session.get(url) as resp:
            resp.raise_for_status()
            data = await resp.read()
            with open(filename, 'wb') as f:
                f.write(data)
            return filename, len(data)
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            download_one(session, url, fname)
            for url, fname in files
        ]
        results = await asyncio.gather(*tasks)
        
        for fname, size in results:
            print(f"{fname}: {size} bytes")

asyncio.run(download_multiple())
```

### 下载到内存中

```python
async def download_to_memory(url):
    """下载到内存（适合小文件）"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            # read() 一次性读取所有内容
            data = await resp.read()
            return data

# 使用
data = asyncio.run(download_to_memory('https://httpbin.org/bytes/1024'))
print(f"下载了 {len(data)} 字节")
```
</div>

---

## 9.11 实战：批量爬虫

现在我们把前面学到的知识整合起来，构建一个完整的批量网页爬虫。

### 需求

爬取一个网站的多个页面，提取信息，并保存结果。

### 完整代码

```python
import aiohttp
import asyncio
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class PageInfo:
    """页面信息"""
    url: str
    status: int
    title: str
    content_length: int
    load_time: float
    error: Optional[str] = None

class AsyncCrawler:
    """异步爬虫"""
    
    def __init__(self, max_concurrent=10, timeout=30):
        self.sem = asyncio.Semaphore(max_concurrent)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.results: list[PageInfo] = []
    
    async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> PageInfo:
        """获取单个页面"""
        start = time.time()
        
        async with self.sem:
            try:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    text = await resp.text()
                    load_time = time.time() - start
                    
                    # 简单提取标题
                    title = ''
                    if '<title>' in text:
                        start_idx = text.index('<title>') + 7
                        end_idx = text.index('</title>')
                        title = text[start_idx:end_idx].strip()
                    
                    return PageInfo(
                        url=url,
                        status=resp.status,
                        title=title,
                        content_length=len(text),
                        load_time=load_time,
                    )
            except Exception as e:
                load_time = time.time() - start
                return PageInfo(
                    url=url,
                    status=0,
                    title='',
                    content_length=0,
                    load_time=load_time,
                    error=str(e),
                )
    
    async def crawl(self, urls: list[str]) -> list[PageInfo]:
        """爬取多个页面"""
        connector = aiohttp.TCPConnector(
            limit=50,
            limit_per_host=10,
        )
        
        headers = {
            'User-Agent': 'AsyncCrawler/1.0 (Educational)',
            'Accept': 'text/html',
        }
        
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout,
            headers=headers,
        ) as session:
            tasks = [self.fetch_page(session, url) for url in urls]
            
            # 使用 as_completed 来实时处理结果
            completed = 0
            for coro in asyncio.as_completed(tasks):
                result = await coro
                completed += 1
                status_icon = '✓' if result.error is None else '✗'
                print(f"[{completed}/{len(urls)}] {status_icon} "
                      f"{result.url} ({result.status}) "
                      f"{result.load_time:.2f}s")
                self.results.append(result)
        
        return self.results
    
    def save_results(self, filename: str):
        """保存结果到 JSON 文件"""
        data = [asdict(r) for r in self.results]
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到 {filename}")
    
    def print_summary(self):
        """打印汇总"""
        total = len(self.results)
        success = sum(1 for r in self.results if r.error is None)
        failed = total - success
        
        if success > 0:
            avg_time = sum(r.load_time for r in self.results if r.error is None) / success
            total_size = sum(r.content_length for r in self.results if r.error is None)
        else:
            avg_time = 0
            total_size = 0
        
        print(f"\n{'='*50}")
        print(f"爬取完成!")
        print(f"总计: {total} 页面")
        print(f"成功: {success} 页面")
        print(f"失败: {failed} 页面")
        print(f"平均加载时间: {avg_time:.2f}秒")
        print(f"总数据量: {total_size/1024:.1f} KB")
        print(f"{'='*50}")


async def main():
    # 示例 URL 列表
    urls = [
        'https://httpbin.org/get',
        'https://httpbin.org/headers',
        'https://httpbin.org/ip',
        'https://httpbin.org/user-agent',
        'https://httpbin.org/anything',
        'https://httpbin.org/encoding/utf8',
        'https://httpbin.org/html',
        'https://httpbin.org/json',
        'https://httpbin.org/xml',
        'https://httpbin.org/robots.txt',
    ]
    
    crawler = AsyncCrawler(max_concurrent=5)
    
    start = time.time()
    results = await crawler.crawl(urls)
    elapsed = time.time() - start
    
    crawler.print_summary()
    print(f"总用时: {elapsed:.2f}秒")
    
    crawler.save_results('crawl_results.json')

asyncio.run(main())
```

### 爬虫的改进版本：支持递归爬取

```python
class RecursiveCrawler:
    """支持递归的爬虫"""
    
    def __init__(self, max_depth=2, max_concurrent=5):
        self.max_depth = max_depth
        self.sem = asyncio.Semaphore(max_concurrent)
        self.visited: set[str] = set()
        self.results: list[PageInfo] = []
    
    async def crawl_page(self, session, url, depth):
        """爬取单个页面并提取链接"""
        if depth > self.max_depth or url in self.visited:
            return []
        
        self.visited.add(url)
        
        async with self.sem:
            try:
                async with session.get(url) as resp:
                    text = await resp.text()
                    
                    # 提取链接（简化版）
                    links = []
                    import re
                    for match in re.finditer(r'href="(https?://[^"]+)"', text):
                        link = match.group(1)
                        if link not in self.visited:
                            links.append(link)
                    
                    self.results.append(PageInfo(
                        url=url,
                        status=resp.status,
                        title='',
                        content_length=len(text),
                        load_time=0,
                    ))
                    
                    return [(link, depth + 1) for link in links[:5]]
                    
            except Exception:
                return []
    
    async def crawl(self, start_url):
        """从起始 URL 开始递归爬取"""
        async with aiohttp.ClientSession() as session:
            # 第一层
            links = await self.crawl_page(session, start_url, 0)
            
            # 递归后续层
            for link, depth in links:
                await self.crawl_page(session, link, depth)
        
        return self.results
```

---

## 9.12 性能对比：同步 vs 异步

让我们做一个严格的性能对比实验。

### 实验设计

```python
import aiohttp
import asyncio
import requests
import time

# 测试 URL 列表
TEST_URLS = [
    'https://httpbin.org/delay/1',
] * 10  # 10 个请求，每个延迟 1 秒

# 同步版本
def sync_fetch_all(urls):
    results = []
    for url in urls:
        resp = requests.get(url)
        results.append(resp.status_code)
    return results

# 异步版本
async def async_fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            tasks.append(fetch_one(session, url))
        return await asyncio.gather(*tasks)

async def fetch_one(session, url):
    async with session.get(url) as resp:
        return resp.status

# 运行对比
def benchmark():
    print("=== 性能对比 ===")
    
    # 同步测试
    start = time.time()
    sync_results = sync_fetch_all(TEST_URLS)
    sync_time = time.time() - start
    print(f"同步用时: {sync_time:.2f}秒")
    
    # 异步测试
    start = time.time()
    async_results = asyncio.run(async_fetch_all(TEST_URLS))
    async_time = time.time() - start
    print(f"异步用时: {async_time:.2f}秒")
    
    # 计算加速比
    speedup = sync_time / async_time
    print(f"加速比: {speedup:.1f}x")

benchmark()
```

### 预期结果

```
=== 性能对比 ===
同步用时: 10.32秒
异步用时: 1.15秒
加速比: 9.0x
```

同步版本是串行的：10 个请求 × 1 秒 = 10 秒。
异步版本是并发的：所有请求同时发出，总时间 ≈ 1 秒。

### 更完整的基准测试

```python
import aiohttp
import asyncio
import requests
import time
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    method: str
    num_requests: int
    total_time: float
    requests_per_second: float
    avg_time_per_request: float

def benchmark_sync(urls):
    """同步基准测试"""
    start = time.time()
    for url in urls:
        resp = requests.get(url)
        resp.raise_for_status()
    elapsed = time.time() - start
    
    return BenchmarkResult(
        method='同步 requests',
        num_requests=len(urls),
        total_time=elapsed,
        requests_per_second=len(urls) / elapsed,
        avg_time_per_request=elapsed / len(urls),
    )

async def benchmark_async(urls, max_concurrent=None):
    """异步基准测试"""
    sem = asyncio.Semaphore(max_concurrent) if max_concurrent else None
    
    async def fetch(session, url):
        if sem:
            async with sem:
                async with session.get(url) as resp:
                    return resp.status
        else:
            async with session.get(url) as resp:
                return resp.status
    
    start = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        await asyncio.gather(*tasks)
    elapsed = time.time() - start
    
    label = f'异步 aiohttp (max={max_concurrent})' if max_concurrent else '异步 aiohttp (无限)'
    
    return BenchmarkResult(
        method=label,
        num_requests=len(urls),
        total_time=elapsed,
        requests_per_second=len(urls) / elapsed,
        avg_time_per_request=elapsed / len(urls),
    )

async def run_full_benchmark():
    urls = ['https://httpbin.org/delay/0.5'] * 20
    
    print("=" * 60)
    print(f"基准测试: {len(urls)} 个请求，每个延迟 0.5 秒")
    print("=" * 60)
    
    # 同步
    sync_result = benchmark_sync(urls)
    print(f"{sync_result.method}: {sync_result.total_time:.2f}秒")
    
    # 异步（无限并发）
    async_result = await benchmark_async(urls)
    print(f"{async_result.method}: {async_result.total_time:.2f}秒")
    
    # 异步（限制并发）
    for limit in [5, 10, 20]:
        result = await benchmark_async(urls, max_concurrent=limit)
        print(f"{result.method}: {result.total_time:.2f}秒")
    
    print("=" * 60)
    print(f"加速比（无限并发）: "
          f"{sync_result.total_time / async_result.total_time:.1f}x")

asyncio.run(run_full_benchmark())
```

---

## 9.13 httpx 作为替代方案

`httpx` 是另一个优秀的异步 HTTP 客户端，它的 API 设计与 `requests` 非常相似。

### 为什么选择 httpx？

| 特性 | aiohttp | httpx |
|------|---------|-------|
| API 风格 | 自有风格 | 类似 requests |
| HTTP/2 支持 | 不支持 | 支持 |
| 同步/异步 | 仅异步 | 两者都支持 |
| WebSocket | 支持 | 不支持 |
| 成熟度 | 非常成熟 | 较新但稳定 |
| 依赖 | 无外部依赖 | 有一些依赖 |

### httpx 基础用法

```python
import httpx
import asyncio

async def httpx_example():
    # 方式 1：使用 async with 自动管理
    async with httpx.AsyncClient() as client:
        resp = await client.get('https://httpbin.org/get')
        print(resp.status_code)
        print(resp.json())
    
    # 方式 2：手动管理
    client = httpx.AsyncClient()
    try:
        resp = await client.get('https://httpbin.org/get')
        print(resp.status_code)
    finally:
        await client.aclose()

asyncio.run(httpx_example())
```

### httpx vs aiohttp 对比

```python
import aiohttp
import httpx
import asyncio

# aiohttp 风格
async def with_aiohttp():
    async with aiohttp.ClientSession() as session:
        async with session.get('https://httpbin.org/get') as resp:
            data = await resp.json()
            return data

# httpx 风格
async def with_httpx():
    async with httpx.AsyncClient() as client:
        resp = await client.get('https://httpbin.org/get')
        data = resp.json()  # 注意：httpx 的 json() 不需要 await
        return data

# 两者非常相似，主要区别：
# 1. aiohttp 用 session.get() 返回 context manager
# 2. httpx 用 client.get() 直接返回 response
# 3. aiohttp 的 resp.json() 需要 await
# 4. httpx 的 resp.json() 不需要 await
```

### httpx 的 HTTP/2 支持

```python
import httpx
import asyncio

async def http2_example():
    # httpx 支持 HTTP/2（需要安装 httpx[http2]）
    async with httpx.AsyncClient(http2=True) as client:
        resp = await client.get('https://httpbin.org/get')
        print(f"HTTP 版本: {resp.http_version}")
        print(f"状态码: {resp.status_code}")

asyncio.run(http2_example())
```

### httpx 并发请求

```python
import httpx
import asyncio

async def httpx_concurrent():
    urls = [
        'https://httpbin.org/get',
        'https://httpbin.org/headers',
        'https://httpbin.org/ip',
    ]
    
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        
        for resp in responses:
            print(f"{resp.url}: {resp.status_code}")

asyncio.run(httpx_concurrent())
```

### 何时选择哪个？

- **选 aiohttp**：
  - 需要 WebSocket 支持
  - 需要更成熟的异步 HTTP 实现
  - 不需要 HTTP/2
  - 项目已经在用 aiohttp

- **选 httpx**：
  - 需要 HTTP/2 支持
  - 想要与 requests 类似的 API
  - 需要在同步和异步之间切换
  - 新项目，没有历史包袱

---

## 9.14 常见误区

### 误区 1：忘记 await

```python
# 错误：忘记 await
async def wrong():
    async with aiohttp.ClientSession() as session:
        resp = session.get('https://httpbin.org/get')  # 缺少 await
        print(resp)  # 打印的是协程对象，不是响应

# 正确
async def correct():
    async with aiohttp.ClientSession() as session:
        resp = await session.get('https://httpbin.org/get')
        print(resp.status)
```

### 误区 2：在异步代码中使用同步库

```python
# 错误：在异步代码中用 requests（会阻塞事件循环）
async def wrong():
    import requests
    resp = requests.get('https://httpbin.org/get')  # 阻塞！
    print(resp.status_code)

# 正确：用 aiohttp 或 httpx
async def correct():
    async with aiohttp.ClientSession() as session:
        async with session.get('https://httpbin.org/get') as resp:
            print(resp.status)

# 如果必须用同步库，用 run_in_executor
async def workaround():
    import requests
    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(
        None,  # 使用默认线程池
        requests.get,
        'https://httpbin.org/get'
    )
    print(resp.status_code)
```

### 误区 3：不关闭 response

```python
# 错误：不关闭 response
async def wrong():
    async with aiohttp.ClientSession() as session:
        resp = await session.get('https://httpbin.org/get')
        text = await resp.text()
        # resp 没有关闭，连接可能泄漏

# 正确：使用 async with
async def correct():
    async with aiohttp.ClientSession() as session:
        async with session.get('https://httpbin.org/get') as resp:
            text = await resp.text()
        # 离开 with 块后自动关闭
```

### 误区 4：创建过多的 Session

```python
# 错误：每个请求都创建新 session
async def wrong(urls):
    results = []
    for url in urls:
        async with aiohttp.ClientSession() as session:  # 每次都新建
            async with session.get(url) as resp:
                results.append(await resp.text())
    return results

# 正确：复用 session
async def correct(urls):
    async with aiohttp.ClientSession() as session:
        results = []
        for url in urls:
            async with session.get(url) as resp:
                results.append(await resp.text())
        return results
```

### 误区 5：不在异步上下文中调用

```python
# 错误：在普通函数中调用异步代码
def wrong():
    session = aiohttp.ClientSession()  # 不能这样用
    resp = session.get('https://httpbin.org/get')  # 不能这样用

# 正确：在异步函数中使用
async def correct():
    async with aiohttp.ClientSession() as session:
        async with session.get('https://httpbin.org/get') as resp:
            print(resp.status)

# 正确：在同步代码中运行异步函数
def run():
    asyncio.run(correct())
```

### 误区 6：不处理超时

```python
# 错误：不设置超时
async def wrong():
    async with aiohttp.ClientSession() as session:
        # 如果服务器不响应，这里会永远等待
        async with session.get('https://slow-server.com/api') as resp:
            return await resp.text()

# 正确：设置超时
async def correct():
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get('https://slow-server.com/api') as resp:
                return await resp.text()
        except asyncio.TimeoutError:
            print("请求超时")
            return None
```

### 误区 7：错误的并发控制

```python
# 错误：不受限制的并发
async def wrong(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [session.get(url) for url in urls]
        # 如果 urls 有 10000 个，会同时发起 10000 个请求
        responses = await asyncio.gather(*tasks)
        return [await r.text() for r in responses]

# 正确：限制并发数
async def correct(urls, max_concurrent=10):
    sem = asyncio.Semaphore(max_concurrent)
    
    async def fetch(session, url):
        async with sem:
            async with session.get(url) as resp:
                return await resp.text()
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        return await asyncio.gather(*tasks)
```

### 误区 8：忽略异常处理

```python
# 错误：不处理异常
async def wrong(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [session.get(url) for url in urls]
        return await asyncio.gather(*tasks)  # 一个失败全部失败

# 正确：使用 return_exceptions 或单独处理
async def correct(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [session.get(url) for url in urls]
        # 方式 1：return_exceptions=True
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 方式 2：单独处理每个请求
        async def safe_fetch(url):
            try:
                async with session.get(url) as resp:
                    return await resp.text()
            except Exception as e:
                return f"Error: {e}"
        
        tasks = [safe_fetch(url) for url in urls]
        return await asyncio.gather(*tasks)
```

### 误区 9：在非异步函数中直接 await

```python
# 错误：在普通函数中用 await
def wrong():
    result = await some_async_func()  # 语法错误

# 正确：用 asyncio.run()
def correct():
    result = asyncio.run(some_async_func())
    return result

# 正确：用事件循环
def correct_loop():
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(some_async_func())
    return result
```

### 误区 10：忽略连接池耗尽

```python
# 问题：连接池被耗尽时会怎样
async def connection_exhaustion():
    # 默认连接池限制是 100
    connector = aiohttp.TCPConnector(limit=5)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        # 创建 20 个并发请求，但连接池只有 5
        tasks = []
        for i in range(20):
            tasks.append(session.get(f'https://httpbin.org/delay/1'))
        
        # 不会报错，但多余的请求会排队等待
        responses = await asyncio.gather(*tasks)
        print(f"完成 {len(responses)} 个请求")

asyncio.run(connection_exhaustion())
```

---

## 补充：实际项目中的最佳实践

### 项目结构示例

```
my_async_project/
├── main.py              # 入口
├── client.py            # HTTP 客户端封装
├── config.py            # 配置（超时、并发数等）
├── exceptions.py        # 自定义异常
├── middleware.py         # 中间件（重试、限流等）
└── utils.py             # 工具函数
```

### 封装一个生产级 HTTP 客户端

```python
# client.py
import aiohttp
import asyncio
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class Response:
    """统一的响应封装"""
    status: int
    data: Any
    headers: dict
    url: str

class AsyncHTTPClient:
    """生产级异步 HTTP 客户端"""
    
    def __init__(
        self,
        base_url: str = '',
        max_concurrent: int = 10,
        timeout: int = 30,
        max_retries: int = 3,
        headers: Optional[dict] = None,
    ):
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.default_headers = headers or {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._sem: Optional[asyncio.Semaphore] = None
    
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=self.max_concurrent,
        )
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout,
            headers=self.default_headers,
        )
        self._sem = asyncio.Semaphore(self.max_concurrent)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
    
    async def _request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> Response:
        """发送请求（带重试）"""
        full_url = f"{self.base_url}{url}" if self.base_url else url
        
        for attempt in range(self.max_retries):
            try:
                async with self._sem:
                    async with self._session.request(
                        method, full_url, **kwargs
                    ) as resp:
                        resp.raise_for_status()
                        
                        content_type = resp.content_type
                        if content_type == 'application/json':
                            data = await resp.json()
                        else:
                            data = await resp.text()
                        
                        return Response(
                            status=resp.status,
                            data=data,
                            headers=dict(resp.headers),
                            url=str(resp.url),
                        )
                        
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    await asyncio.sleep(wait)
                else:
                    raise
    
    async def get(self, url: str, **kwargs) -> Response:
        return await self._request('GET', url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> Response:
        return await self._request('POST', url, **kwargs)
    
    async def put(self, url: str, **kwargs) -> Response:
        return await self._request('PUT', url, **kwargs)
    
    async def delete(self, url: str, **kwargs) -> Response:
        return await self._request('DELETE', url, **kwargs)

# 使用示例
async def main():
    async with AsyncHTTPClient(
        base_url='https://api.example.com',
        max_concurrent=5,
        timeout=15,
    ) as client:
        # 串行请求
        user = await client.get('/users/1')
        print(f"用户: {user.data}")
        
        # 并发请求
        tasks = [
            client.get(f'/users/{i}')
            for i in range(1, 11)
        ]
        users = await asyncio.gather(*tasks)
        print(f"获取了 {len(users)} 个用户")

asyncio.run(main())
```

### 中间件模式

```python
# middleware.py
import asyncio
import time
from functools import wraps

def retry(max_retries=3, delay=1, backoff=2):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = delay * (backoff ** attempt)
                        await asyncio.sleep(wait)
                    else:
                        raise
        return wrapper
    return decorator

def rate_limit(calls_per_second):
    """速率限制装饰器"""
    min_interval = 1.0 / calls_per_second
    
    def decorator(func):
        last_call = 0
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal last_call
            now = time.monotonic()
            elapsed = now - last_call
            
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            
            last_call = time.monotonic()
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def log_requests(func):
    """请求日志装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start
            print(f"[{func.__name__}] 成功 ({elapsed:.2f}s)")
            return result
        except Exception as e:
            elapsed = time.time() - start
            print(f"[{func.__name__}] 失败 ({elapsed:.2f}s): {e}")
            raise
    return wrapper

# 组合使用
@retry(max_retries=3)
@rate_limit(calls_per_second=10)
@log_requests
async def fetch_data(session, url):
    async with session.get(url) as resp:
        return await resp.json()
```

### 配置管理

```python
# config.py
from dataclasses import dataclass
import os

@dataclass
class HTTPConfig:
    """HTTP 客户端配置"""
    # 超时设置
    total_timeout: int = 30
    connect_timeout: int = 10
    read_timeout: int = 10
    
    # 并发控制
    max_concurrent: int = 10
    max_concurrent_per_host: int = 5
    
    # 重试设置
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    
    # 连接池
    pool_limit: int = 100
    dns_cache_ttl: int = 300
    
    # 默认 headers
    user_agent: str = 'AsyncHTTPClient/1.0'
    
    @classmethod
    def from_env(cls):
        """从环境变量加载配置"""
        return cls(
            total_timeout=int(os.getenv('HTTP_TIMEOUT', 30)),
            max_concurrent=int(os.getenv('HTTP_MAX_CONCURRENT', 10)),
            max_retries=int(os.getenv('HTTP_MAX_RETRIES', 3)),
        )

# 使用
config = HTTPConfig.from_env()
```

### 错误处理的完整模式

```python
# exceptions.py
class HTTPClientError(Exception):
    """HTTP 客户端基础异常"""
    pass

class RequestTimeoutError(HTTPClientError):
    """请求超时"""
    pass

class ConnectionError(HTTPClientError):
    """连接错误"""
    pass

class RateLimitExceededError(HTTPClientError):
    """超出速率限制"""
    pass

class RetryExhaustedError(HTTPClientError):
    """重试次数耗尽"""
    def __init__(self, last_error, attempts):
        self.last_error = last_error
        self.attempts = attempts
        super().__init__(
            f"重试 {attempts} 次后失败: {last_error}"
        )

# 在客户端中使用
class RobustHTTPClient:
    async def _request(self, method, url, **kwargs):
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                async with self._sem:
                    async with self._session.request(
                        method, url, **kwargs
                    ) as resp:
                        resp.raise_for_status()
                        return await resp.json()
                        
            except asyncio.TimeoutError as e:
                last_error = RequestTimeoutError(str(e))
            except aiohttp.ClientConnectionError as e:
                last_error = ConnectionError(str(e))
            except aiohttp.ClientResponseError as e:
                if e.status == 429:
                    last_error = RateLimitExceededError(str(e))
                elif e.status >= 500:
                    last_error = e
                else:
                    raise
            
            if attempt < self.max_retries - 1:
                wait = self.retry_delay * (self.retry_backoff ** attempt)
                await asyncio.sleep(wait)
        
        raise RetryExhaustedError(last_error, self.max_retries)
```

---

## 本章小结

本章深入探讨了 Python 异步网络请求的方方面面：

1. **网络请求的本质**：大部分时间都在等待网络响应，CPU 空闲。这正是异步编程最擅长处理的场景。

2. **aiohttp 基础**：ClientSession 是核心，管理连接池和配置。使用 `async with` 管理生命周期是最佳实践。

3. **并发请求**：使用 `asyncio.gather()` 并发执行多个请求，比串行执行快 N 倍。

4. **错误处理**：网络请求充满不确定性。超时控制、异常捕获、重试机制缺一不可。

5. **并发控制**：使用 Semaphore 限制并发数，避免目标服务器过载和本地资源耗尽。

6. **流式下载**：大文件要用流式读取，避免内存溢出。

7. **Session 管理**：一个应用一个 Session，复用连接池。

8. **替代方案**：httpx 提供了类似 requests 的 API 和 HTTP/2 支持，是 aiohttp 的有力竞争者。

9. **常见误区**：忘记 await、同步库阻塞、不关闭资源、无限并发等都是新手容易犯的错误。

**关键数字**：
- 异步并发比串行快 N 倍（N = 请求数）
- 一个 Session 的默认连接数限制是 100
- 建议每主机并发数限制在 5-10
- 网络请求的典型延迟是 100-500ms

**核心原则**：
- 网络 I/O → 用异步
- 一个 Session 复用
- 必须设置超时
- 必须限制并发
- 必须处理异常

---

## 思考题

1. **基础理解**：为什么网络请求特别适合异步编程？与文件 I/O 相比，网络请求有什么特点使得异步的收益更大？

2. **代码实践**：编写一个异步函数，同时请求 5 个不同的 API 端点，收集所有响应的 JSON 数据，并返回一个合并的字典。

3. **错误处理**：编写一个带指数退避重试的请求函数，当遇到 5xx 错误或超时时自动重试，最多重试 3 次。

4. **并发控制**：实现一个 `RateLimiter` 类，支持"每秒最多 N 个请求"的限制，并用它来爬取一个网站的 100 个页面。

5. **性能分析**：如果有 100 个请求，每个需要 200ms，分别计算串行执行、10 并发、50 并发、无限并发的理论用时。

6. **资源管理**：解释为什么在异步代码中，每个请求都创建新的 `ClientSession` 是一个坏习惯。这样做会导致什么问题？

7. **方案选型**：你的项目需要同时支持 WebSocket 和 HTTP/2，你会选择 aiohttp 还是 httpx？为什么？有没有可能两者都用？

8. **实战挑战**：编写一个异步文件下载器，支持以下功能：
   - 并发下载多个文件
   - 每个文件支持断点续传（Range 请求）
   - 实时显示下载进度
   - 限制总带宽（可选）

9. **深入思考**：在什么场景下，异步网络请求可能反而比同步更慢？提示：考虑服务器的速率限制和本地资源限制。

10. **架构设计**：设计一个异步爬虫框架，支持以下特性：
    - 可配置的并发数和速率限制
    - 自动重试和错误处理
    - 结果持久化（保存到文件或数据库）
    - 礼貌爬取（遵守 robots.txt，设置合理的请求间隔）

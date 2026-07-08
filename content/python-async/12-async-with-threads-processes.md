# 第12章 Async 与线程/进程的结合

> **"异步不是万能的，线程和进程也不是过时的——真正的力量在于三者的协作。"**

在前面的章节中，我们深入学习了 `asyncio` 的各种模式。但在实际工程中，纯异步代码经常会遇到瓶颈：文件 I/O、数据库驱动、CPU 密集计算……这些场景下，单靠事件循环并不能解决问题。本章将系统讲解如何将 async 与 threading、multiprocessing 结合，构建真正高性能的 Python 应用。

---

## 12.1 为什么需要结合 — async alone can't do everything

### 12.1.1 异步的边界

`asyncio` 的核心优势在于高效管理大量 I/O 等待——网络请求、WebSocket 连接、消息队列消费等。但它有一个根本限制：

**事件循环是单线程的。**

这意味着当一个协程执行 CPU 密集计算时，整个事件循环都会被阻塞：

```python
import asyncio
import time

async def cpu_heavy():
    """这个函数会阻塞事件循环！"""
    total = 0
    for i in range(10_000_000):
        total += i * i
    return total

async def other_tasks():
    """这个协程在 cpu_heavy 运行期间完全得不到执行机会"""
    while True:
        print(f"[{time.strftime('%H:%M:%S')}] 其他任务正常运行")
        await asyncio.sleep(1)

async def main():
    # cpu_heavy 会阻塞事件循环约 2-3 秒
    # 在此期间 other_tasks 完全无法执行
    await asyncio.gather(cpu_heavy(), other_tasks())

asyncio.run(main())
```

输出结果会显示 `other_tasks` 的打印在 `cpu_heavy` 完成前完全停滞。

### 12.1.2 三类"异步不友好"的场景

| 场景 | 问题 | 解决方案 |
|------|------|----------|
| **CPU 密集计算** | 计算期间事件循环被阻塞 | `ProcessPoolExecutor` |
| **同步 I/O 库** | 如 `open()`、`requests`、某些数据库驱动 | `ThreadPoolExecutor` 或 `to_thread()` |
| **C 扩展/FFI 调用** | 底层 C 代码会释放 GIL 但也可能阻塞 | `ProcessPoolExecutor` 或线程 |

### 12.1.3 Python GIL 的影响

理解为什么需要结合三种并发模型，必须理解 GIL（全局解释器锁）：

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python 进程                              │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                      GIL (全局解释器锁)                   │   │
│  │                                                          │   │
│  │  同一时刻只有一个线程能执行 Python 字节码                   │   │
│  │  但 I/O 操作、C 扩展可以释放 GIL                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ 线程 1   │  │ 线程 2   │  │ 线程 3   │  │ 线程 4   │      │
│  │ 执行     │  │ 等待 GIL │  │ I/O 等待 │  │ 执行     │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

- **线程**：适合 I/O 密集（释放 GIL 时其他线程可以运行）
- **进程**：适合 CPU 密集（每个进程有自己的 GIL）
- **协程**：适合大量并发 I/O（单线程，零切换开销）

### 12.1.4 结合的必要性总结

```
你的应用需要什么？
│
├── 大量网络 I/O 并发？ ──────→ 纯 asyncio 就够了
│
├── 一些阻塞库无法替换？ ────→ asyncio + 线程
│
├── CPU 密集计算？ ───────────→ asyncio + 进程
│
└── 以上都有？ ───────────────→ asyncio + 线程 + 进程（混合模式）
```

---

## 12.2 asyncio.to_thread() — run sync function in thread

<div data-component="ToThreadDemo"></div>

### 12.2.1 基本用法

`asyncio.to_thread()` 是 Python 3.9 引入的最简单方式，将同步函数放到线程池中执行：

```python
import asyncio

def blocking_io():
    """一个阻塞的同步函数"""
    with open('/dev/null', 'r') as f:
        return f.read()

async def main():
    # 在线程中运行阻塞函数，不阻塞事件循环
    result = await asyncio.to_thread(blocking_io)
    print(f"结果: {result!r}")

asyncio.run(main())
```

### 12.2.2 函数签名

```python
asyncio.to_thread(func, /, *args, **kwargs)
```

- `func`：要在线程中运行的同步函数（位置参数）
- `*args, **kwargs`：传递给函数的参数
- 返回一个协程，`await` 后得到函数的返回值
- 内部使用默认的 `ThreadPoolExecutor`

### 12.2.3 参数传递

```python
import asyncio

def read_file(path, encoding='utf-8'):
    with open(path, 'r', encoding=encoding) as f:
        return f.read()

async def main():
    # 传递位置参数和关键字参数
    content = await asyncio.to_thread(
        read_file, 
        '/path/to/file.txt',
        encoding='latin-1'
    )
    print(content)

asyncio.run(main())
```

### 12.2.4 to_thread() 的内部实现

理解其原理有助于正确使用：

```python
# CPython 源码简化版 (Lib/asyncio/threads.py)
async def to_thread(func, /, *args, **kwargs):
    loop = asyncio.get_running_loop()
    # 使用默认的线程池执行器
    ctx = contextvars.copy_context()
    func_call = functools.partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(None, func_call)
```

关键点：
1. 它实际上是 `loop.run_in_executor()` 的语法糖
2. 自动复制 `contextvars` 上下文到新线程
3. 使用默认线程池执行器（`None` 表示默认）

### 12.2.5 并发执行多个阻塞调用

```python
import asyncio
import time

def blocking_task(task_id, duration):
    """模拟阻塞操作"""
    print(f"  任务 {task_id} 开始，需要 {duration} 秒")
    time.sleep(duration)
    return f"任务 {task_id} 完成"

async def main():
    start = time.time()
    
    # 顺序执行 — 总耗时 = 各任务之和
    # r1 = await asyncio.to_thread(blocking_task, 1, 2)
    # r2 = await asyncio.to_thread(blocking_task, 2, 3)
    # r3 = await asyncio.to_thread(blocking_task, 3, 1)
    
    # 并发执行 — 总耗时 = 最长的那个
    results = await asyncio.gather(
        asyncio.to_thread(blocking_task, 1, 2),
        asyncio.to_thread(blocking_task, 2, 3),
        asyncio.to_thread(blocking_task, 3, 1),
    )
    
    elapsed = time.time() - start
    print(f"结果: {results}")
    print(f"总耗时: {elapsed:.1f}秒 (而非 {2+3+1}秒)")

asyncio.run(main())
```

输出：
```
  任务 1 开始，需要 2 秒
  任务 2 开始，需要 3 秒
  任务 3 开始，需要 1 秒
结果: ['任务 1 完成', '任务 2 完成', '任务 3 完成']
总耗时: 3.0秒 (而非 6秒)
```
</div>

---

## 12.3 to_thread() 示例 — file I/O, blocking library

### 12.3.1 文件 I/O 示例

```python
import asyncio
import json

def load_config(path):
    """同步的配置文件加载"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_report(path, data):
    """同步的报告保存"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

async def process_and_save(config_path, report_path):
    # 在线程中加载配置（不阻塞事件循环）
    config = await asyncio.to_thread(load_config, config_path)
    
    # ... 进行一些异步处理 ...
    result = {"processed": True, "config": config}
    
    # 在线程中保存报告
    await asyncio.to_thread(save_report, report_path, result)

asyncio.run(process_and_save('config.json', 'report.json'))
```

### 12.3.2 使用 requests 库（同步 HTTP）

```python
import asyncio
import requests

def fetch_url(url, timeout=10):
    """使用同步 requests 库获取 URL"""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text

async def fetch_multiple(urls):
    """并发获取多个 URL（使用线程）"""
    tasks = [
        asyncio.to_thread(fetch_url, url)
        for url in urls
    ]
    return await asyncio.gather(*tasks, return_exceptions=True)

async def main():
    urls = [
        "https://httpbin.org/get",
        "https://httpbin.org/delay/1",
        "https://httpbin.org/delay/2",
    ]
    results = await fetch_multiple(urls)
    for url, result in zip(urls, results):
        if isinstance(result, Exception):
            print(f"  {url}: 错误 - {result}")
        else:
            print(f"  {url}: 成功 ({len(result)} 字节)")

asyncio.run(main())
```

### 12.3.3 数据库操作（同步驱动）

```python
import asyncio
import sqlite3

def init_db(db_path):
    """初始化数据库（同步）"""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE
        )
    """)
    conn.commit()
    return conn

def insert_user(conn, name, email):
    """插入用户（同步）"""
    conn.execute(
        "INSERT OR IGNORE INTO users (name, email) VALUES (?, ?)",
        (name, email)
    )
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

def query_users(conn, limit=100):
    """查询用户（同步）"""
    cursor = conn.execute(
        "SELECT id, name, email FROM users LIMIT ?", (limit,)
    )
    return cursor.fetchall()

async def main():
    db_path = "example.db"
    
    # 在线程中初始化数据库
    conn = await asyncio.to_thread(init_db, db_path)
    
    # 并发插入多个用户
    users_to_insert = [
        ("Alice", "alice@example.com"),
        ("Bob", "bob@example.com"),
        ("Charlie", "charlie@example.com"),
    ]
    
    insert_tasks = [
        asyncio.to_thread(insert_user, conn, name, email)
        for name, email in users_to_insert
    ]
    user_ids = await asyncio.gather(*insert_tasks)
    print(f"插入的用户 ID: {user_ids}")
    
    # 查询用户
    users = await asyncio.to_thread(query_users, conn)
    for uid, name, email in users:
        print(f"  [{uid}] {name} <{email}>")
    
    conn.close()

asyncio.run(main())
```

### 12.3.4 带进度回调的长时间操作

```python
import asyncio
import time

def long_running_task(total_steps, progress_callback=None):
    """一个长时间运行的同步任务，支持进度回调"""
    result = []
    for i in range(total_steps):
        time.sleep(0.1)  # 模拟工作
        result.append(i * i)
        if progress_callback:
            progress_callback(i + 1, total_steps)
    return result

async def main_with_progress():
    loop = asyncio.get_running_loop()
    
    def on_progress(current, total):
        # 注意：这个回调在工作线程中执行
        # 使用 call_soon_threadsafe 安全地通知事件循环
        loop.call_soon_threadsafe(
            lambda: print(f"进度: {current}/{total} ({current*100//total}%)")
        )
    
    # 使用 to_thread 不直接支持回调，需要用 run_in_executor
    result = await asyncio.to_thread(
        long_running_task, 10, on_progress
    )
    print(f"完成！结果长度: {len(result)}")

asyncio.run(main_with_progress())
```

---

## 12.4 run_in_executor() — more general, thread or process pool

### 12.4.1 基本用法

`loop.run_in_executor()` 是更底层、更灵活的 API：

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def cpu_bound(n):
    """CPU 密集计算"""
    return sum(i * i for i in range(n))

async def main():
    loop = asyncio.get_running_loop()
    
    # 使用默认线程池
    result = await loop.run_in_executor(None, cpu_bound, 1_000_000)
    print(f"默认线程池结果: {result}")
    
    # 使用自定义线程池
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = await loop.run_in_executor(pool, cpu_bound, 1_000_000)
        print(f"自定义线程池结果: {result}")
    
    # 使用进程池
    with ProcessPoolExecutor(max_workers=4) as pool:
        result = await loop.run_in_executor(pool, cpu_bound, 1_000_000)
        print(f"进程池结果: {result}")

asyncio.run(main())
```

### 12.4.2 函数签名与参数

```python
coroutine loop.run_in_executor(executor, func, *args)
```

- `executor`：`None`（使用默认线程池）、`ThreadPoolExecutor` 或 `ProcessPoolExecutor`
- `func`：**必须是普通函数**，不能是协程
- `*args`：传递给函数的位置参数（不支持关键字参数！）
- 返回一个可 `await` 的 Future

### 12.4.3 传递关键字参数

`run_in_executor` 不直接支持关键字参数，需要使用 `functools.partial`：

```python
import asyncio
import functools

def compute(data, multiplier=1, offset=0):
    return sum(x * multiplier + offset for x in data)

async def main():
    loop = asyncio.get_running_loop()
    
    # 方法1：使用 functools.partial
    func = functools.partial(compute, [1, 2, 3, 4, 5], multiplier=2, offset=10)
    result = await loop.run_in_executor(None, func)
    print(f"结果: {result}")  # (1*2+10) + (2*2+10) + ... = 80
    
    # 方法2：使用 lambda
    result = await loop.run_in_executor(
        None, 
        lambda: compute([1, 2, 3, 4, 5], multiplier=3)
    )
    print(f"结果: {result}")

asyncio.run(main())
```

### 12.4.4 to_thread() vs run_in_executor() 对比

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

def blocking_func(x):
    return x * x

async def main():
    loop = asyncio.get_running_loop()
    
    # to_thread: 简洁，自动复制 contextvars
    r1 = await asyncio.to_thread(blocking_func, 10)
    
    # run_in_executor: 更灵活，可指定执行器
    r2 = await loop.run_in_executor(None, blocking_func, 10)
    
    # to_thread 内部大致等价于：
    # ctx = contextvars.copy_context()
    # func = functools.partial(ctx.run, blocking_func, 10)
    # await loop.run_in_executor(None, func)

asyncio.run(main())
```

| 特性 | `to_thread()` | `run_in_executor()` |
|------|---------------|---------------------|
| Python 版本 | 3.9+ | 3.4+ |
| 关键字参数 | 原生支持 | 需要 `partial` |
| contextvars | 自动复制 | 需手动处理 |
| 指定执行器 | 使用默认线程池 | 可指定任意执行器 |
| 进程池支持 | 不支持 | 支持 |
| 简洁性 | 更简洁 | 更灵活 |

---

## 12.5 ThreadPoolExecutor with async

<div data-component="ExecutorComparison"></div>

### 12.5.1 自定义线程池配置

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time

def io_task(task_id):
    """模拟 I/O 任务"""
    thread_name = threading.current_thread().name
    print(f"  任务 {task_id} 在线程 {thread_name} 中开始")
    time.sleep(1)  # 模拟 I/O 等待
    return f"任务 {task_id} 完成 (线程: {thread_name})"

async def main():
    # 创建自定义线程池
    executor = ThreadPoolExecutor(
        max_workers=3,           # 最多 3 个工作线程
        thread_name_prefix="io"  # 线程名前缀
    )
    
    loop = asyncio.get_running_loop()
    
    # 提交 6 个任务，但只有 3 个线程，所以分两批执行
    tasks = [
        loop.run_in_executor(executor, io_task, i)
        for i in range(6)
    ]
    
    results = await asyncio.gather(*tasks)
    for r in results:
        print(f"  {r}")
    
    executor.shutdown(wait=True)

asyncio.run(main())
```

输出示例：
```
  任务 0 在线程 io_0 中开始
  任务 1 在线程 io_1 中开始
  任务 2 在线程 io_2 中开始
  任务 3 在线程 io_0 中开始  # 线程被复用
  任务 4 在线程 io_1 中开始
  任务 5 在线程 io_2 中开始
  任务 0 完成 (线程: io_0)
  ...
```

### 12.5.2 线程池大小选择

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

def get_optimal_thread_count(task_type="io"):
    """根据任务类型推荐线程数"""
    cpu_count = os.cpu_count() or 4
    
    if task_type == "io":
        # I/O 密集：可以开更多线程
        return min(32, cpu_count * 5)
    elif task_type == "compute":
        # 计算密集：线程数 ≈ CPU 核心数
        return cpu_count
    else:
        return cpu_count * 2

async def main():
    io_threads = get_optimal_thread_count("io")
    compute_threads = get_optimal_thread_count("compute")
    
    print(f"CPU 核心数: {os.cpu_count()}")
    print(f"I/O 任务推荐线程数: {io_threads}")
    print(f"计算任务推荐线程数: {compute_threads}")
    
    # 使用推荐配置
    io_executor = ThreadPoolExecutor(max_workers=io_threads, thread_name_prefix="io")
    compute_executor = ThreadPoolExecutor(max_workers=compute_threads, thread_name_prefix="compute")
    
    # ... 使用执行器 ...
    
    io_executor.shutdown()
    compute_executor.shutdown()

asyncio.run(main())
```

### 12.5.3 作为应用级资源管理

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

@asynccontextmanager
async def managed_thread_pool(max_workers=None):
    """作为异步上下文管理器管理线程池生命周期"""
    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        yield pool
    finally:
        # 在关闭时等待所有任务完成
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, pool.shutdown, True)

async def app():
    async with managed_thread_pool(max_workers=10) as pool:
        loop = asyncio.get_running_loop()
        
        # 使用池执行任务
        results = await asyncio.gather(
            loop.run_in_executor(pool, lambda: "task1"),
            loop.run_in_executor(pool, lambda: "task2"),
            loop.run_in_executor(pool, lambda: "task3"),
        )
        print(f"结果: {results}")
    # 这里池已被安全关闭

asyncio.run(app())
```

### 12.5.4 实际应用：并发文件处理

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import hashlib

def calculate_file_hash(filepath):
    """计算文件的 MD5 哈希（同步 I/O）"""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return filepath, hasher.hexdigest()

async def hash_directory(directory, max_workers=8):
    """并发计算目录中所有文件的哈希"""
    # 收集所有文件路径
    files = []
    for root, _, filenames in os.walk(directory):
        for name in filenames:
            files.append(os.path.join(root, name))
    
    if not files:
        return {}
    
    # 使用线程池并发计算
    executor = ThreadPoolExecutor(max_workers=max_workers)
    loop = asyncio.get_running_loop()
    
    try:
        tasks = [
            loop.run_in_executor(executor, calculate_file_hash, f)
            for f in files
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        hashes = {}
        for result in results:
            if isinstance(result, Exception):
                print(f"错误: {result}")
            else:
                filepath, hash_val = result
                hashes[filepath] = hash_val
        
        return hashes
    finally:
        executor.shutdown(wait=False)

async def main():
    hashes = await hash_directory("/tmp")
    for path, h in list(hashes.items())[:5]:
        print(f"  {h}  {path}")
    print(f"  ... 共 {len(hashes)} 个文件")

asyncio.run(main())
```
</div>

---

## 12.6 ProcessPoolExecutor with async — CPU-bound tasks

### 12.6.1 基本用法

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor
import time

def fibonacci(n):
    """CPU 密集：计算斐波那契数列"""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

async def main():
    start = time.time()
    
    loop = asyncio.get_running_loop()
    
    # 使用进程池并行计算
    with ProcessPoolExecutor(max_workers=4) as pool:
        tasks = [
            loop.run_in_executor(pool, fibonacci, n)
            for n in [30_000, 35_000, 40_000, 45_000]
        ]
        results = await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    for n, result in zip([30000, 35000, 40000, 45000], results):
        print(f"  fib({n}) = {str(result)[:20]}... ({len(str(result))} 位)")
    print(f"总耗时: {elapsed:.2f}秒")

asyncio.run(main())
```

### 12.6.2 线程池 vs 进程池性能对比

```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def cpu_heavy(n):
    """CPU 密集任务"""
    total = 0
    for i in range(n):
        total += i * i
    return total

async def benchmark(executor_type, workers, tasks_count, n):
    """基准测试"""
    loop = asyncio.get_running_loop()
    
    if executor_type == "thread":
        executor = ThreadPoolExecutor(max_workers=workers)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
    
    start = time.time()
    
    with executor:
        tasks = [
            loop.run_in_executor(executor, cpu_heavy, n)
            for _ in range(tasks_count)
        ]
        await asyncio.gather(*tasks)
    
    return time.time() - start

async def main():
    tasks_count = 8
    n = 5_000_000
    
    print("CPU 密集任务性能对比:")
    print("-" * 50)
    
    # 单线程顺序执行（基准）
    start = time.time()
    for _ in range(tasks_count):
        cpu_heavy(n)
    sequential = time.time() - start
    print(f"  顺序执行:         {sequential:.2f}秒")
    
    # 线程池
    thread_time = await benchmark("thread", 4, tasks_count, n)
    print(f"  线程池 (4 workers): {thread_time:.2f}秒 "
          f"({sequential/thread_time:.1f}x)")
    
    # 进程池
    process_time = await benchmark("process", 4, tasks_count, n)
    print(f"  进程池 (4 workers): {process_time:.2f}秒 "
          f"({sequential/process_time:.1f}x)")

asyncio.run(main())
```

典型输出：
```
CPU 密集任务性能对比:
--------------------------------------------------
  顺序执行:         8.12秒
  线程池 (4 workers): 8.35秒 (1.0x)  ← GIL 限制，几乎无加速
  进程池 (4 workers): 2.45秒 (3.3x)  ← 真正的并行
```

### 12.6.3 进程池的注意事项

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

# 注意：进程池中的函数必须是可 pickle 的
# 也就是说，必须在模块顶层定义，不能是 lambda 或闭包

def process_data(data_chunk):
    """必须在模块顶层定义，不能嵌套"""
    return [x ** 2 for x in data_chunk]

async def main():
    data = list(range(1000))
    chunk_size = 250
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    
    loop = asyncio.get_running_loop()
    
    with ProcessPoolExecutor(max_workers=4) as pool:
        # 每个进程处理一个数据块
        tasks = [
            loop.run_in_executor(pool, process_data, chunk)
            for chunk in chunks
        ]
        results = await asyncio.gather(*tasks)
    
    # 合并结果
    all_results = []
    for r in results:
        all_results.extend(r)
    
    print(f"处理了 {len(all_results)} 个元素")
    print(f"前 10 个结果: {all_results[:10]}")

asyncio.run(main())
```

### 12.6.4 使用 initializer 初始化进程

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor
import sqlite3

# 全局连接（每个进程一个）
_db_connection = None

def init_worker(db_path):
    """进程初始化函数：每个 worker 进程启动时调用一次"""
    global _db_connection
    _db_connection = sqlite3.connect(db_path)
    print(f"  进程初始化: 连接到 {db_path}")

def query_in_process(sql):
    """在进程中执行查询"""
    global _db_connection
    cursor = _db_connection.execute(sql)
    return cursor.fetchall()

async def main():
    loop = asyncio.get_running_loop()
    
    with ProcessPoolExecutor(
        max_workers=2,
        initializer=init_worker,
        initargs=("example.db",)  # 传递给 initializer 的参数
    ) as pool:
        tasks = [
            loop.run_in_executor(pool, query_in_process, 
                               "SELECT * FROM users LIMIT 10"),
            loop.run_in_executor(pool, query_in_process,
                               "SELECT COUNT(*) FROM users"),
        ]
        results = await asyncio.gather(*tasks)
    
    for r in results:
        print(r)

# asyncio.run(main())  # 需要数据库文件
```

### 12.6.5 共享状态与进程间通信

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def compute_with_shared_state(data, counter):
    """使用共享计数器"""
    result = []
    for item in data:
        result.append(item ** 2)
        with counter.get_lock():
            counter.value += 1
    return result

async def main():
    data_chunks = [list(range(i, i+100)) for i in range(0, 400, 100)]
    
    # 创建共享计数器
    counter = mp.Value('i', 0)
    
    loop = asyncio.get_running_loop()
    
    with ProcessPoolExecutor(max_workers=4) as pool:
        tasks = [
            loop.run_in_executor(pool, compute_with_shared_state, chunk, counter)
            for chunk in data_chunks
        ]
        await asyncio.gather(*tasks)
    
    print(f"总处理项数: {counter.value}")

asyncio.run(main())
```

---

## 12.7 混合模式实战 — download + process + save

<div data-component="HybridModeDiagram"></div>

### 12.7.1 典型数据管道

真实应用往往同时需要网络 I/O、CPU 处理和文件 I/O。下面是一个完整的混合模式示例：

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import time
import hashlib

# ========== 模拟外部依赖 ==========

async def download_data(url):
    """网络下载（异步）"""
    await asyncio.sleep(0.5)  # 模拟网络延迟
    return {"url": url, "data": list(range(100))}

# ========== 同步处理函数 ==========

def transform_data(raw_data):
    """CPU 密集：数据转换（在进程中运行）"""
    result = []
    for item in raw_data["data"]:
        transformed = item ** 2 + hash(raw_data["url"]) % 100
        result.append(transformed)
    return {
        "source": raw_data["url"],
        "transformed": result,
        "checksum": hashlib.md5(str(result).encode()).hexdigest()
    }

def save_to_file(data, output_dir):
    """文件 I/O：保存结果（在线程中运行）"""
    filename = f"{output_dir}/{data['checksum'][:8]}.json"
    # 模拟文件写入
    time.sleep(0.1)
    return filename

# ========== 管道组装 ==========

async def process_url(url, process_pool, io_pool, output_dir):
    """处理单个 URL 的完整管道"""
    loop = asyncio.get_running_loop()
    
    # 步骤 1：异步下载
    raw = await download_data(url)
    
    # 步骤 2：CPU 密集处理（进程池）
    processed = await loop.run_in_executor(process_pool, transform_data, raw)
    
    # 步骤 3：文件保存（线程池）
    filepath = await loop.run_in_executor(io_pool, save_to_file, processed, output_dir)
    
    return filepath

async def main():
    urls = [f"https://api.example.com/data/{i}" for i in range(20)]
    output_dir = "/tmp/output"
    
    # 创建不同用途的执行器
    process_pool = ProcessPoolExecutor(max_workers=4, thread_name_prefix="cpu")
    io_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="io")
    
    try:
        start = time.time()
        
        # 并发处理所有 URL
        tasks = [
            process_url(url, process_pool, io_pool, output_dir)
            for url in urls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        elapsed = time.time() - start
        
        success = [r for r in results if not isinstance(r, Exception)]
        failed = [r for r in results if isinstance(r, Exception)]
        
        print(f"处理完成:")
        print(f"  成功: {len(success)}")
        print(f"  失败: {len(failed)}")
        print(f"  耗时: {elapsed:.2f}秒")
    finally:
        process_pool.shutdown(wait=False)
        io_pool.shutdown(wait=False)

asyncio.run(main())
```

### 12.7.2 流水线架构

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import deque

class AsyncPipeline:
    """异步流水线：下载 → 处理 → 保存"""
    
    def __init__(self, max_download=10, max_process=4, max_save=8):
        self.download_sem = asyncio.Semaphore(max_download)
        self.process_pool = ProcessPoolExecutor(max_workers=max_process)
        self.save_pool = ThreadPoolExecutor(max_workers=max_save)
        self.loop = None
    
    async def start(self):
        self.loop = asyncio.get_running_loop()
    
    async def download(self, url):
        """阶段 1：并发下载（限流）"""
        async with self.download_sem:
            # 模拟下载
            await asyncio.sleep(0.3)
            return {"url": url, "bytes": b"x" * 1000}
    
    def process(self, raw_data):
        """阶段 2：CPU 密集处理（进程池）"""
        # 模拟数据处理
        data = raw_data["bytes"]
        processed = bytes(b ^ 0xFF for b in data)
        return {"url": raw_data["url"], "result": len(processed)}
    
    def save(self, processed_data):
        """阶段 3：保存到文件（线程池）"""
        # 模拟文件写入
        import time
        time.sleep(0.05)
        return f"saved_{processed_data['url'].split('/')[-1]}"
    
    async def process_url(self, url):
        """单个 URL 的完整处理流程"""
        # 阶段 1
        raw = await self.download(url)
        
        # 阶段 2（进程池）
        processed = await self.loop.run_in_executor(
            self.process_pool, self.process, raw
        )
        
        # 阶段 3（线程池）
        result = await self.loop.run_in_executor(
            self.save_pool, self.save, processed
        )
        
        return result
    
    async def run(self, urls):
        """运行流水线"""
        await self.start()
        tasks = [self.process_url(url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def shutdown(self):
        self.process_pool.shutdown(wait=False)
        self.save_pool.shutdown(wait=False)

async def main():
    pipeline = AsyncPipeline(max_download=5, max_process=4, max_save=8)
    
    urls = [f"https://example.com/file/{i}" for i in range(50)]
    
    try:
        start = time.time()
        results = await pipeline.run(urls)
        elapsed = time.time() - start
        
        ok = [r for r in results if not isinstance(r, Exception)]
        err = [r for r in results if isinstance(r, Exception)]
        
        print(f"流水线完成: {len(ok)} 成功, {len(err)} 失败, {elapsed:.2f}秒")
    finally:
        pipeline.shutdown()

asyncio.run(main())
```

### 12.7.3 带背压的生产者-消费者模式

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

async def producer(queue, urls):
    """生产者：从网络下载数据"""
    for url in urls:
        # 模拟下载
        await asyncio.sleep(0.1)
        data = {"url": url, "content": f"data from {url}"}
        await queue.put(data)  # 如果队列满，会自动等待（背压）
        print(f"  生产: {url}")
    
    # 发送结束信号
    await queue.put(None)

def process_data(data):
    """消费者处理函数（CPU 密集）"""
    import time
    time.sleep(0.2)  # 模拟处理
    return f"processed: {data['url']}"

async def consumer(queue, process_pool, results):
    """消费者：处理数据"""
    loop = asyncio.get_running_loop()
    
    while True:
        data = await queue.get()
        if data is None:
            await queue.put(None)  # 通知其他消费者
            break
        
        result = await loop.run_in_executor(process_pool, process_data, data)
        results.append(result)
        print(f"  消费: {result}")

async def main():
    # 使用有界队列实现背压
    queue = asyncio.Queue(maxsize=5)
    results = []
    
    urls = [f"https://api.example.com/item/{i}" for i in range(20)]
    
    with ProcessPoolExecutor(max_workers=3) as pool:
        # 1 个生产者，3 个消费者
        await asyncio.gather(
            producer(queue, urls),
            consumer(queue, pool, results),
            consumer(queue, pool, results),
            consumer(queue, pool, results),
        )
    
    print(f"\n处理完成: {len(results)} 项")

asyncio.run(main())
```
</div>

---

## 12.8 事件循环与线程的交互

### 12.8.1 每个线程的事件循环

关键规则：**每个线程最多运行一个事件循环**。

```python
import asyncio
import threading

async def async_work():
    return threading.current_thread().name

def thread_worker():
    """在新线程中运行事件循环"""
    # 每个线程可以有自己的事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(async_work())
        print(f"线程 {threading.current_thread().name}: 结果 = {result}")
    finally:
        loop.close()

async def main():
    # 主线程已经有事件循环
    print(f"主线程: {threading.current_thread().name}")
    
    # 在新线程中运行独立的事件循环
    t = threading.Thread(target=thread_worker, name="worker-thread")
    t.start()
    t.join()

asyncio.run(main())
```

### 12.8.2 事件循环的线程安全性

```python
import asyncio
import threading

# 事件循环本身不是线程安全的
# 只能在拥有事件循环的线程中调用大多数方法
# 跨线程操作必须使用 call_soon_threadsafe

async def main():
    loop = asyncio.get_running_loop()
    
    def schedule_from_other_thread():
        """从其他线程安全地调度回调"""
        loop.call_soon_threadsafe(print, "从另一个线程调度的回调！")
    
    t = threading.Thread(target=schedule_from_other_thread)
    t.start()
    
    await asyncio.sleep(0.1)  # 给回调时间执行
    t.join()

asyncio.run(main())
```

### 12.8.3 线程局部存储与事件循环

```python
import asyncio
import threading

# 线程局部存储：每个线程独立的存储
_thread_local = threading.local()

def get_thread_loop():
    """获取当前线程的事件循环"""
    if not hasattr(_thread_local, 'loop'):
        _thread_local.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_thread_local.loop)
    return _thread_local.loop

async def main():
    # 主线程的循环
    main_loop = asyncio.get_running_loop()
    print(f"主循环 ID: {id(main_loop)}")

asyncio.run(main())
```

---

## 12.9 跨线程调用 — loop.call_soon_threadsafe

<div data-component="CrossThreadCallDemo"></div>

### 12.9.1 问题场景

当工作线程需要通知事件循环时，不能直接操作事件循环——必须使用 `call_soon_threadsafe`：

```python
import asyncio
import threading
import time

def background_worker(loop, result_future):
    """在后台线程中工作，完成后通知事件循环"""
    print(f"  后台线程开始工作...")
    time.sleep(2)  # 模拟耗时操作
    
    result = "后台任务的结果"
    
    # 安全地在事件循环线程中设置结果
    loop.call_soon_threadsafe(result_future.set_result, result)
    print(f"  后台线程: 已通知事件循环")

async def main():
    loop = asyncio.get_running_loop()
    
    # 创建一个 Future 用于线程间通信
    future = loop.create_future()
    
    # 启动后台线程
    t = threading.Thread(target=background_worker, args=(loop, future))
    t.start()
    
    # 等待后台线程完成
    result = await future
    print(f"  主协程收到结果: {result}")
    
    t.join()

asyncio.run(main())
```

### 12.9.2 ToThreadDemo 组件：完整的跨线程通信模式

```python
import asyncio
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

@dataclass
class ThreadTask:
    """表示一个在线程中执行的任务"""
    task_id: str
    func: Callable
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    result: Any = None
    error: Optional[Exception] = None
    completed: bool = False

class ToThreadDemo:
    """演示各种跨线程调用模式"""
    
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._results: dict[str, asyncio.Future] = {}
    
    async def run_task(self, task_id: str, func: Callable, *args, **kwargs) -> Any:
        """在线程池中运行任务，返回结果"""
        self._loop = asyncio.get_running_loop()
        
        # 方法1：使用 asyncio.to_thread（最简单）
        return await asyncio.to_thread(func, *args, **kwargs)
    
    async def run_with_callback(self, task_id: str, func: Callable,
                                 on_progress: Callable[[int, int], None],
                                 *args, **kwargs) -> Any:
        """在线程中运行任务，支持进度回调"""
        self._loop = asyncio.get_running_loop()
        future = self._loop.create_future()
        
        def thread_work():
            try:
                def progress_wrapper(current, total):
                    # 安全地从线程调用事件循环
                    self._loop.call_soon_threadsafe(on_progress, current, total)
                
                result = func(*args, progress_callback=progress_wrapper, **kwargs)
                self._loop.call_soon_threadsafe(future.set_result, result)
            except Exception as e:
                self._loop.call_soon_threadsafe(future.set_exception, e)
        
        import threading
        t = threading.Thread(target=thread_work, name=f"task-{task_id}")
        t.start()
        
        return await future
    
    async def run_batch(self, tasks: list[ThreadTask]) -> list[ThreadTask]:
        """批量运行任务"""
        async def run_one(task: ThreadTask) -> ThreadTask:
            try:
                task.result = await asyncio.to_thread(
                    task.func, *task.args, **task.kwargs
                )
                task.completed = True
            except Exception as e:
                task.error = e
            return task
        
        return await asyncio.gather(*[run_one(t) for t in tasks])


# === 使用示例 ===

def heavy_computation(n, progress_callback=None):
    """带进度回调的计算"""
    total = 0
    for i in range(n):
        total += i * i
        if progress_callback and i % (n // 10) == 0:
            progress_callback(i, n)
    if progress_callback:
        progress_callback(n, n)
    return total

async def main():
    demo = ToThreadDemo()
    
    print("=== 示例 1: 简单任务 ===")
    result = await demo.run_task("calc", heavy_computation, 1000000)
    print(f"  结果: {result}")
    
    print("\n=== 示例 2: 带进度回调的任务 ===")
    def show_progress(current, total):
        print(f"  进度: {current}/{total} ({current*100//total}%)")
    
    result = await demo.run_with_callback(
        "progress-calc", heavy_computation, show_progress, 10000000
    )
    print(f"  最终结果: {result}")
    
    print("\n=== 示例 3: 批量任务 ===")
    tasks = [
        ThreadTask(f"task-{i}", heavy_computation, (100000 * (i + 1),))
        for i in range(5)
    ]
    results = await demo.run_batch(tasks)
    for task in results:
        status = "✓" if task.completed else "✗"
        print(f"  {status} {task.task_id}: {task.result}")

asyncio.run(main())
```

### 12.9.3 使用 concurrent.futures.Future 桥接

```python
import asyncio
from concurrent.futures import Future as ConcurrentFuture
import threading
import time

class ThreadBridge:
    """将 concurrent.futures.Future 桥接到 asyncio.Future"""
    
    def __init__(self):
        self._loop: asyncio.AbstractEventLoop = None
    
    async def submit(self, func, *args, **kwargs):
        """提交任务到线程并返回结果"""
        self._loop = asyncio.get_running_loop()
        
        # 创建两种 Future
        async_future = self._loop.create_future()
        concurrent_future = ConcurrentFuture()
        
        def run_in_thread():
            try:
                result = func(*args, **kwargs)
                concurrent_future.set_result(result)
            except Exception as e:
                concurrent_future.set_exception(e)
        
        # 当 concurrent future 完成时，设置 asyncio future
        def on_concurrent_done(fut):
            try:
                result = fut.result()
                self._loop.call_soon_threadsafe(async_future.set_result, result)
            except Exception as e:
                self._loop.call_soon_threadsafe(async_future.set_exception, e)
        
        concurrent_future.add_done_callback(on_concurrent_done)
        
        # 启动线程
        t = threading.Thread(target=run_in_thread)
        t.start()
        
        return await async_future

async def main():
    bridge = ThreadBridge()
    
    def blocking_operation(x):
        time.sleep(1)
        return x ** 2
    
    result = await bridge.submit(blocking_operation, 42)
    print(f"结果: {result}")

asyncio.run(main())
```

### 12.9.4 CrossThreadCallDemo 组件

```python
import asyncio
import threading
import time
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum

class CallMode(Enum):
    """跨线程调用模式"""
    FIRE_AND_FORGET = "fire_and_forget"      # 发射后不管
    WAIT_RESULT = "wait_result"              # 等待结果
    CALLBACK = "callback"                     # 回调通知
    QUEUE_BASED = "queue_based"              # 基于队列

@dataclass
class CrossThreadMessage:
    """跨线程消息"""
    source_thread: str
    target_thread: str
    payload: Any
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[asyncio.Future] = None

class CrossThreadCallDemo:
    """演示各种跨线程调用模式"""
    
    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._handlers: Dict[str, Callable] = {}
    
    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
    
    # ---- 模式 1: Fire and Forget ----
    
    def fire_and_forget(self, func: Callable, *args, **kwargs):
        """从任意线程发射一个任务，不等待结果"""
        if self._loop is None:
            raise RuntimeError("事件循环未设置")
        
        def wrapper():
            try:
                func(*args, **kwargs)
            except Exception as e:
                print(f"  [fire_and_forget] 错误: {e}")
        
        self._loop.call_soon_threadsafe(wrapper)
    
    # ---- 模式 2: 等待结果 ----
    
    async def call_from_thread(self, func: Callable, *args, **kwargs) -> Any:
        """从工作线程调用事件循环中的协程并等待结果"""
        if self._loop is None:
            raise RuntimeError("事件循环未设置")
        
        future = self._loop.create_future()
        
        def schedule():
            asyncio.ensure_future(self._run_and_set(future, func, *args, **kwargs))
        
        self._loop.call_soon_threadsafe(schedule)
        return await future
    
    async def _run_and_set(self, future, func, *args, **kwargs):
        try:
            result = await func(*args, **kwargs)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
    
    # ---- 模式 3: 回调通知 ----
    
    def notify_from_thread(self, event_name: str, data: Any):
        """从工作线程发送事件通知到事件循环"""
        if self._loop is None:
            raise RuntimeError("事件循环未设置")
        
        def dispatch():
            handler = self._handlers.get(event_name)
            if handler:
                handler(data)
            else:
                print(f"  [通知] 未处理的事件: {event_name}")
        
        self._loop.call_soon_threadsafe(dispatch)
    
    def on(self, event_name: str, handler: Callable):
        """注册事件处理器"""
        self._handlers[event_name] = handler
    
    # ---- 模式 4: 基于队列 ----
    
    def send_to_queue(self, message: Any):
        """从工作线程发送消息到异步队列"""
        if self._loop is None:
            raise RuntimeError("事件循环未设置")
        
        self._loop.call_soon_threadsafe(self._message_queue.put_nowait, message)
    
    async def receive_from_queue(self) -> Any:
        """从异步队列接收消息"""
        return await self._message_queue.get()


# === 演示 ===

def worker_thread(demo: CrossThreadCallDemo, thread_id: int):
    """工作线程：演示各种调用模式"""
    thread_name = f"Worker-{thread_id}"
    
    # 模式 1: 发射后不管
    demo.fire_and_forget(
        lambda: print(f"  [{thread_name}] fire-and-forget 执行完毕")
    )
    
    # 模式 3: 通知事件
    demo.notify_from_thread("progress", {"thread": thread_id, "percent": 100})
    
    # 模式 4: 发送消息到队列
    demo.send_to_queue(f"来自 {thread_name} 的消息")

async def main():
    demo = CrossThreadCallDemo()
    demo.set_loop(asyncio.get_running_loop())
    
    # 注册事件处理器
    demo.on("progress", lambda data: print(f"  [事件] 进度更新: {data}"))
    
    # 启动工作线程
    threads = []
    for i in range(3):
        t = threading.Thread(target=worker_thread, args=(demo, i))
        t.start()
        threads.append(t)
    
    # 从队列接收消息
    for _ in range(3):
        msg = await asyncio.wait_for(demo.receive_from_queue(), timeout=5)
        print(f"  [队列] 收到: {msg}")
    
    for t in threads:
        t.join()

asyncio.run(main())
```
</div>

---

## 12.10 async in non-async context — nest_asyncio

### 12.10.1 问题：Jupyter Notebook 中的事件循环

在 Jupyter Notebook 或 IPython 中，已经有一个事件循环在运行。直接调用 `asyncio.run()` 会报错：

```python
# 在 Jupyter 中运行会报错：
# RuntimeError: asyncio.run() cannot be called from a running event loop
asyncio.run(some_coroutine())
```

### 12.10.2 解决方案：nest_asyncio

```python
import nest_asyncio
nest_asyncio.apply()

# 现在可以在已有事件循环中运行嵌套的 asyncio
async def nested():
    await asyncio.sleep(1)
    return "done"

# 在 Jupyter 中直接 await
# result = await nested()

# 或者使用 asyncio.run（现在不会报错了）
result = asyncio.run(nested())
```

### 12.10.3 nest_asyncio 的原理

```python
# nest_asyncio 的核心思想：
# 1. 修补 (patch) 事件循环的 run_until_complete 方法
# 2. 允许在已有事件循环中"嵌套"运行新的异步代码
# 3. 使用 reentrant（可重入）锁替代普通锁

import asyncio
import nest_asyncio

# 查看修补前后的差异
print("修补前:")
print(f"  loop.run_until_complete 类型: {type(asyncio.get_event_loop().run_until_complete)}")

nest_asyncio.apply()

print("修补后:")
print(f"  loop.run_until_complete 类型: {type(asyncio.get_event_loop().run_until_complete)}")
```

### 12.10.4 在同步代码中调用异步函数

```python
import asyncio
import nest_asyncio

# 应用补丁
nest_asyncio.apply()

class DataService:
    """同时支持同步和异步接口的服务"""
    
    async def fetch_async(self, url):
        """异步接口"""
        await asyncio.sleep(0.1)
        return f"data from {url}"
    
    def fetch_sync(self, url):
        """同步接口：内部调用异步方法"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.fetch_async(url))

# 在已有事件循环的环境中使用
service = DataService()

# 异步使用
async def async_main():
    result = await service.fetch_async("http://example.com")
    print(f"异步结果: {result}")

asyncio.run(async_main())

# 同步使用
result = service.fetch_sync("http://example.com")
print(f"同步结果: {result}")
```

### 12.10.5 不使用 nest_asyncio 的替代方案

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

def run_async_in_sync(coro):
    """在同步上下文中运行异步代码（不使用 nest_asyncio）"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop and loop.is_running():
        # 已有事件循环在运行（如 Jupyter），使用线程
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        # 没有运行中的事件循环，直接运行
        return asyncio.run(coro)

# 使用示例
async def my_coroutine():
    await asyncio.sleep(0.1)
    return 42

# 无论是否有运行中的事件循环都可以调用
result = run_async_in_sync(my_coroutine())
print(f"结果: {result}")
```

### 12.10.6 FastAPI/Flask 中混合同步与异步

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# ========== FastAPI 示例 ==========

# from fastapi import FastAPI
# app = FastAPI()
# 
# # FastAPI 原生支持异步端点
# @app.get("/async-endpoint")
# async def async_handler():
#     result = await some_async_operation()
#     return {"result": result}
# 
# # 同步端点会自动在线程中运行
# @app.get("/sync-endpoint")
# def sync_handler():
#     result = some_sync_operation()  # 不会阻塞事件循环
#     return {"result": result}

# ========== Flask + 异步支持示例 ==========

# from flask import Flask
# import asyncio
# 
# app = Flask(__name__)
# 
# def run_async(coro):
#     """在 Flask 路由中运行异步代码"""
#     loop = asyncio.new_event_loop()
#     try:
#         return loop.run_until_complete(coro)
#     finally:
#         loop.close()
# 
# @app.route("/data")
# def get_data():
#     result = run_async(fetch_data_async())
#     return {"data": result}
```

---

## 12.11 何时用什么 — decision guide

<div data-component="DecisionGuideFlowchart"></div>

### 12.11.1 DecisionGuideFlowchart 组件：选择指南

```
你的任务是什么类型？
│
├── 网络 I/O（HTTP、WebSocket、数据库）
│   ├── 有 async 驱动？ → 纯 asyncio（最佳）
│   └── 只有同步库？ → asyncio + 线程（to_thread / run_in_executor）
│
├── 文件 I/O
│   ├── 小文件少量？ → 纯 asyncio（aiofiles）
│   ├── 大文件/大量？ → asyncio + 线程
│   └── 内存映射？ → asyncio + 线程
│
├── CPU 密集计算
│   ├── 计算量小？ → asyncio + 线程（释放 GIL 时有效）
│   ├── 计算量大？ → asyncio + 进程（ProcessPoolExecutor）
│   └── 需要共享状态？ → 进程 + multiprocessing.Manager
│
└── 混合场景
    └── asyncio + 线程池（I/O）+ 进程池（CPU）
```

### 12.11.2 DecisionGuideFlowchart 组件代码

```python
from enum import Enum
from typing import Optional, Tuple

class TaskType(Enum):
    NETWORK_IO = "network_io"
    FILE_IO = "file_io"
    CPU_BOUND = "cpu_bound"
    MIXED = "mixed"

class LibraryType(Enum):
    ASYNC_NATIVE = "async_native"    # 有 async 驱动
    SYNC_ONLY = "sync_only"          # 只有同步库

class Scale(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

class Solution(Enum):
    PURE_ASYNC = "纯 asyncio"
    ASYNC_THREAD = "asyncio + 线程"
    ASYNC_PROCESS = "asyncio + 进程"
    ASYNC_BOTH = "asyncio + 线程 + 进程"

class DecisionGuideFlowchart:
    """异步编程方案选择指南"""
    
    @staticmethod
    def recommend(task_type: TaskType,
                  library: Optional[LibraryType] = None,
                  scale: Optional[Scale] = None) -> Tuple[Solution, str]:
        """根据任务特征推荐最佳方案"""
        
        if task_type == TaskType.NETWORK_IO:
            if library == LibraryType.ASYNC_NATIVE:
                return Solution.PURE_ASYNC, "使用原生异步库，性能最优"
            else:
                return Solution.ASYNC_THREAD, "使用 to_thread 包装同步库"
        
        elif task_type == TaskType.FILE_IO:
            if scale == Scale.SMALL:
                return Solution.PURE_ASYNC, "少量小文件，aiofiles 足够"
            else:
                return Solution.ASYNC_THREAD, "大量文件或大文件，用线程池"
        
        elif task_type == TaskType.CPU_BOUND:
            if scale == Scale.SMALL:
                return Solution.ASYNC_THREAD, "计算量小，线程池足够"
            else:
                return Solution.ASYNC_PROCESS, "计算量大，必须用进程池"
        
        elif task_type == TaskType.MIXED:
            return Solution.ASYNC_BOTH, "混合场景，三种并发模型结合"
        
        return Solution.PURE_ASYNC, "默认推荐纯异步"
    
    @staticmethod
    def print_guide():
        """打印完整的选择指南"""
        scenarios = [
            (TaskType.NETWORK_IO, LibraryType.ASYNC_NATIVE, Scale.LARGE),
            (TaskType.NETWORK_IO, LibraryType.SYNC_ONLY, Scale.MEDIUM),
            (TaskType.FILE_IO, None, Scale.SMALL),
            (TaskType.FILE_IO, None, Scale.LARGE),
            (TaskType.CPU_BOUND, None, Scale.SMALL),
            (TaskType.CPU_BOUND, None, Scale.LARGE),
            (TaskType.MIXED, None, Scale.LARGE),
        ]
        
        print("=" * 60)
        print("异步编程方案选择指南")
        print("=" * 60)
        
        for task, lib, scale in scenarios:
            solution, reason = DecisionGuideFlowchart.recommend(task, lib, scale)
            print(f"\n场景: {task.value}")
            if lib:
                print(f"  库类型: {lib.value}")
            print(f"  规模: {scale.value}")
            print(f"  推荐: {solution.value}")
            print(f"  原因: {reason}")

# 运行指南
DecisionGuideFlowchart.print_guide()
```

### 12.11.3 具体场景对照表

| 场景 | 推荐方案 | 原因 |
|------|----------|------|
| HTTP API 调用 (aiohttp) | 纯 asyncio | 原生异步，最高效 |
| HTTP API 调用 (requests) | asyncio + to_thread | 同步库，用线程包装 |
| 大量小文件读写 | asyncio + 线程池 | 文件 I/O 受 GIL 影响小 |
| 图片处理/视频转码 | asyncio + 进程池 | CPU 密集，需要绕过 GIL |
| 数据库查询 (asyncpg) | 纯 asyncio | 原生异步驱动 |
| 数据库查询 (sqlite3) | asyncio + to_thread | 同步驱动 |
| ML 模型推理 | asyncio + 进程池 | CPU/GPU 密集 |
| 日志文件解析 | asyncio + 线程池 | I/O + 轻量处理 |
| 网页爬虫 | asyncio + 线程池 | 大量并发网络 I/O |
| 数据管道 (ETL) | asyncio + 线程 + 进程 | 混合负载 |

### 12.11.4 流程图可视化

```python
def visualize_decision_flow():
    """生成决策流程图的文本表示"""
    diagram = """
    ┌─────────────────────────────────────┐
    │         你的任务是什么类型？           │
    └──────────────┬──────────────────────┘
                   │
         ┌─────────┼─────────┬──────────┐
         │         │         │          │
         ▼         ▼         ▼          ▼
    ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
    │网络 I/O│ │文件 I/O│ │CPU密集 │ │ 混合   │
    └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
        │          │          │          │
        ▼          ▼          ▼          ▼
    有async库?  规模大小?   计算量?    ┌─────────┐
     │  │       │   │      │   │     │async    │
    是  否     小   大    小   大    │+ 线程   │
     │  │       │   │      │   │    │+ 进程   │
     ▼  ▼       ▼   ▼      ▼   ▼    └─────────┘
    纯  +线程  纯  +线程  +线程 +进程
    async      async
    
    图例:
    纯 async = asyncio 原生
    +线程    = asyncio + ThreadPoolExecutor
    +进程    = asyncio + ProcessPoolExecutor
    """
    print(diagram)

visualize_decision_flow()
```
</div>

---

## 12.12 性能考量

### 12.12.1 线程创建的开销

```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

def trivial_task():
    """极轻量的任务"""
    return 42

async def benchmark_thread_creation():
    """测量线程调度的开销"""
    loop = asyncio.get_running_loop()
    
    # 方式 1: 每次创建新线程（通过 to_thread）
    start = time.perf_counter()
    for _ in range(100):
        await asyncio.to_thread(trivial_task)
    no_pool = time.perf_counter() - start
    
    # 方式 2: 使用线程池
    with ThreadPoolExecutor(max_workers=10) as pool:
        start = time.perf_counter()
        for _ in range(100):
            await loop.run_in_executor(pool, trivial_task)
        with_pool = time.perf_counter() - start
    
    # 方式 3: 纯异步对比
    async def async_trivial():
        return 42
    
    start = time.perf_counter()
    for _ in range(100):
        await async_trivial()
    pure_async = time.perf_counter() - start
    
    print(f"100 次调用:")
    print(f"  to_thread (每次新线程):  {no_pool*1000:.1f}ms")
    print(f"  线程池:                  {with_pool*1000:.1f}ms")
    print(f"  纯异步:                  {pure_async*1000:.1f}ms")

asyncio.run(benchmark_thread_creation())
```

典型输出：
```
100 次调用:
  to_thread (每次新线程):  156.3ms
  线程池:                  12.5ms
  纯异步:                  0.3ms
```

### 12.12.2 进程创建的开销

```python
import asyncio
import time
from concurrent.futures import ProcessPoolExecutor

def cpu_task(n):
    return sum(i*i for i in range(n))

async def benchmark_process_pool():
    """测量进程池的开销"""
    loop = asyncio.get_running_loop()
    
    # 冷启动：第一次使用进程池
    with ProcessPoolExecutor(max_workers=4) as pool:
        start = time.perf_counter()
        await loop.run_in_executor(pool, cpu_task, 1000)
        cold_start = time.perf_counter() - start
        
        # 热启动：进程已存在
        start = time.perf_counter()
        await loop.run_in_executor(pool, cpu_task, 1000)
        warm_start = time.perf_counter() - start
    
    print(f"进程池:")
    print(f"  冷启动: {cold_start*1000:.1f}ms")
    print(f"  热启动: {warm_start*1000:.1f}ms")
    print(f"  开销比: {cold_start/warm_start:.1f}x")

asyncio.run(benchmark_process_pool())
```

### 12.12.3 数据传输开销

```python
import asyncio
import time
import pickle
from concurrent.futures import ProcessPoolExecutor

def process_large_data(data):
    """处理大数据"""
    return sum(data)

async def benchmark_data_transfer():
    """测量进程间数据传输的开销"""
    loop = asyncio.get_running_loop()
    
    sizes = [1_000, 100_000, 1_000_000, 10_000_000]
    
    with ProcessPoolExecutor(max_workers=1) as pool:
        for size in sizes:
            data = list(range(size))
            data_mb = len(pickle.dumps(data)) / (1024 * 1024)
            
            start = time.perf_counter()
            result = await loop.run_in_executor(pool, process_large_data, data)
            elapsed = time.perf_counter() - start
            
            print(f"  数据: {size:>10,} 元素 ({data_mb:.1f}MB) → {elapsed*1000:.1f}ms")

asyncio.run(benchmark_data_transfer())
```

### 12.12.4 最佳实践总结

```python
"""
性能优化最佳实践：
"""

# 1. 重用执行器（避免频繁创建销毁）
class App:
    def __init__(self):
        self._thread_pool = ThreadPoolExecutor(max_workers=8)
        self._process_pool = ProcessPoolExecutor(max_workers=4)
    
    async def run_io_task(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._thread_pool, func, *args)
    
    async def run_cpu_task(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._process_pool, func, *args)
    
    def shutdown(self):
        self._thread_pool.shutdown()
        self._process_pool.shutdown()

# 2. 控制并发数量
async def bounded_tasks(tasks, max_concurrent=10):
    """使用信号量限制并发数"""
    sem = asyncio.Semaphore(max_concurrent)
    
    async def bounded(task):
        async with sem:
            return await task
    
    return await asyncio.gather(*[bounded(t) for t in tasks])

# 3. 批量处理减少进程通信开销
async def batch_process(items, batch_size=1000):
    """将小任务批量打包后发送到进程池"""
    loop = asyncio.get_running_loop()
    
    def process_batch(batch):
        return [item ** 2 for item in batch]
    
    batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
    
    with ProcessPoolExecutor() as pool:
        tasks = [loop.run_in_executor(pool, process_batch, b) for b in batches]
        results = await asyncio.gather(*tasks)
    
    # 展平结果
    return [item for batch in results for item in batch]

# 4. 超时保护
async def safe_thread_call(func, *args, timeout=30):
    """带超时的线程调用"""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(func, *args),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        raise TimeoutError(f"任务超时 ({timeout}秒)")
```

---

## 12.13 常见误区

### 12.13.1 误区 1：在协程中调用阻塞函数

```python
# ❌ 错误：直接在协程中调用阻塞函数
async def bad_example():
    import time
    time.sleep(5)  # 这会阻塞整个事件循环！
    return "done"

# ✅ 正确：使用 to_thread
async def good_example():
    await asyncio.sleep(5)  # 如果是异步等待
    # 或者
    await asyncio.to_thread(time.sleep, 5)  # 如果必须用同步函数
    return "done"
```

### 12.13.2 误区 2：过度使用线程/进程

```python
# ❌ 错误：所有任务都用线程
async def bad_pattern():
    # 网络请求本可以用 aiohttp 直接异步
    results = await asyncio.gather(
        asyncio.to_thread(requests.get, url1),
        asyncio.to_thread(requests.get, url2),
        asyncio.to_thread(requests.get, url3),
    )

# ✅ 正确：优先使用原生异步库
async def good_pattern():
    import aiohttp
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            session.get(url1).then(lambda r: r.text()),
            session.get(url2).then(lambda r: r.text()),
            session.get(url3).then(lambda r: r.text()),
        )
```

### 12.13.3 误区 3：忘记进程池中的可 pickle 要求

```python
# ❌ 错误：在进程池中使用 lambda 或闭包
async def bad_process():
    with ProcessPoolExecutor() as pool:
        loop = asyncio.get_running_loop()
        # lambda 不能被 pickle！
        result = await loop.run_in_executor(pool, lambda x: x*2, 10)

# ✅ 正确：使用模块顶层定义的函数
def double(x):
    return x * 2

async def good_process():
    with ProcessPoolExecutor() as pool:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(pool, double, 10)
```

### 12.13.4 误区 4：在错误的线程中操作事件循环

```python
# ❌ 错误：从工作线程直接调用协程
def bad_thread_worker(loop):
    # 这会导致未定义行为！
    asyncio.run_coroutine_threadsafe(some_coro(), loop)

# ✅ 正确：使用 call_soon_threadsafe 或 run_coroutine_threadsafe
def good_thread_worker(loop):
    # 方法 1：调度一个普通回调
    loop.call_soon_threadsafe(some_callback)
    
    # 方法 2：调度一个协程
    future = asyncio.run_coroutine_threadsafe(some_coro(), loop)
    result = future.result()  # 在线程中等待结果
```

### 12.13.5 误区 5：忽略 GIL 对线程的影响

```python
# ❌ 误解：线程能加速 CPU 密集任务
async def bad_cpu_with_threads():
    # 线程无法绕过 GIL，对 CPU 密集任务没有加速效果
    results = await asyncio.gather(
        asyncio.to_thread(cpu_heavy, 1000000),
        asyncio.to_thread(cpu_heavy, 1000000),
        asyncio.to_thread(cpu_heavy, 1000000),
    )

# ✅ 正确：CPU 密集任务使用进程池
async def good_cpu_with_processes():
    with ProcessPoolExecutor() as pool:
        loop = asyncio.get_running_loop()
        results = await asyncio.gather(
            loop.run_in_executor(pool, cpu_heavy, 1000000),
            loop.run_in_executor(pool, cpu_heavy, 1000000),
            loop.run_in_executor(pool, cpu_heavy, 1000000),
        )
```

### 12.13.6 误区 6：线程安全问题

```python
# ❌ 错误：多线程共享可变状态
shared_list = []

async def bad_shared_state():
    def append_item(item):
        shared_list.append(item)  # 列表操作不是线程安全的！
    
    await asyncio.gather(
        asyncio.to_thread(append_item, 1),
        asyncio.to_thread(append_item, 2),
        asyncio.to_thread(append_item, 3),
    )

# ✅ 正确：使用线程安全的数据结构
import queue
import threading

async def good_shared_state():
    result_queue = queue.Queue()
    lock = threading.Lock()
    results = []
    
    def safe_append(item):
        with lock:
            results.append(item)
    
    await asyncio.gather(
        asyncio.to_thread(safe_append, 1),
        asyncio.to_thread(safe_append, 2),
        asyncio.to_thread(safe_append, 3),
    )
```

### 12.13.7 误区 7：不正确的异常处理

```python
# ❌ 错误：忽略线程/进程中的异常
async def bad_exception_handling():
    with ProcessPoolExecutor() as pool:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(pool, risky_function)
        # 异常可能被静默吞掉
        result = future.result()  # 如果不 await，异常不会传播

# ✅ 正确：正确处理异常
async def good_exception_handling():
    with ProcessPoolExecutor() as pool:
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(pool, risky_function)
        except Exception as e:
            print(f"任务失败: {e}")
            # 处理异常或重新抛出
            raise
```

### 12.13.8 误区 8：资源泄漏

```python
# ❌ 错误：忘记关闭执行器
async def bad_resource_management():
    pool = ProcessPoolExecutor(max_workers=4)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(pool, some_task)
    # pool 从未被关闭！进程会泄漏

# ✅ 正确：使用上下文管理器
async def good_resource_management():
    with ProcessPoolExecutor(max_workers=4) as pool:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(pool, some_task)
    # 离开 with 块时自动关闭

# ✅ 也可以：在应用关闭时手动关闭
class App:
    def __init__(self):
        self._pool = ProcessPoolExecutor(max_workers=4)
    
    async def cleanup(self):
        self._pool.shutdown(wait=True)
```

---

## 本章小结

本章系统讲解了如何将 `asyncio` 与线程、进程结合使用：

1. **asyncio 的边界**：事件循环是单线程的，CPU 密集和同步 I/O 操作会阻塞它
2. **`to_thread()`**：Python 3.9+ 的简洁 API，适合包装同步函数
3. **`run_in_executor()`**：更底层、更灵活的接口，支持自定义线程池和进程池
4. **线程池**：适合 I/O 密集的同步操作，可开较多线程
5. **进程池**：适合 CPU 密集计算，能绕过 GIL 实现真正并行
6. **混合模式**：实际应用常需同时使用网络异步 + 线程 I/O + 进程计算
7. **跨线程通信**：`call_soon_threadsafe` 和 `run_coroutine_threadsafe` 是关键
8. **`nest_asyncio`**：在已有事件循环（如 Jupyter）中运行异步代码的解决方案
9. **性能考量**：执行器有开销，重用池、批量处理、控制并发数
10. **常见误区**：GIL 影响、pickle 要求、线程安全、资源泄漏等

**核心原则：**
- 优先使用原生异步库（纯 asyncio 最优）
- 只在必要时引入线程/进程（包装同步库、CPU 密集计算）
- 始终注意资源管理和异常处理
- 理解 GIL 的影响，选择正确的并发模型

---

## 思考题

1. **基础理解**：为什么 `asyncio.to_thread()` 内部要复制 `contextvars`？如果不复制会有什么问题？

2. **性能分析**：假设有 1000 个 HTTP 请求需要发送，使用 `aiohttp`（异步）和 `requests + to_thread`（同步+线程）各有什么优劣？在什么情况下选择后者更合理？

3. **设计题**：你需要设计一个日志分析系统，需要：(a) 从 Kafka 异步消费日志，(b) 对日志进行 CPU 密集的正则匹配和统计，(c) 将结果写入 Elasticsearch。请设计一个混合并发架构。

4. **调试题**：以下代码在 Jupyter Notebook 中运行时报错 `RuntimeError: This event loop is already running`，请解释原因并给出两种修复方案：
   ```python
   async def fetch():
       await asyncio.sleep(1)
       return 42
   
   result = asyncio.run(fetch())
   ```

5. **进阶题**：`ProcessPoolExecutor` 中的函数必须是可 pickle 的。如果你需要处理的数据结构包含不可 pickle 的对象（如文件句柄、数据库连接），有什么解决方案？

6. **架构题**：在一个微服务网关中，需要同时处理 10000 个并发连接，每个连接可能需要调用 CPU 密集的认证加密、同步的旧版数据库、和异步的缓存服务。请设计一个线程池+进程池的配置方案，并解释每个参数的选择理由。

7. **对比实验**：编写一个基准测试，比较以下三种方式处理 100 个文件（每个文件 10MB）的性能：
   - 纯同步顺序读取
   - asyncio + ThreadPoolExecutor
   - asyncio + ProcessPoolExecutor
   分析结果并解释为什么进程池在这种场景下可能不如线程池。

8. **陷阱识别**：以下代码有什么问题？如何修复？
   ```python
   async def process_items(items):
       pool = ProcessPoolExecutor(max_workers=4)
       loop = asyncio.get_running_loop()
       results = await asyncio.gather(
           *[loop.run_in_executor(pool, lambda x: x*2, item) 
             for item in items]
       )
       return results
   ```

---
title: "常见错误与调试技巧"
description: "系统梳理 Python asyncio 编程中的典型错误、调试方法与最佳实践"
updated: "2026-07-07"
---

# 常见错误与调试技巧

> **学习目标**：
> - 识别并修复 asyncio 编程中最常见的 10+ 种错误
> - 理解每种错误的根本原因，而非仅仅记住修复方法
> - 掌握 asyncio debug mode、logging、性能分析等调试工具
> - 建立异步编程的最佳实践清单，从源头避免错误

异步代码的错误往往比同步代码更难调试。原因很简单：执行顺序不确定，错误可能在"意想不到的地方"出现，而且很多错误在小规模测试时根本不会触发，只在高并发时才暴露。

本章会把最常见的错误逐一拆解，每个错误都会给出：
1. **触发代码** — 什么样的写法会出问题
2. **错误信息** — Python 给你的提示长什么样
3. **根本原因** — 为什么会出这个问题
4. **修复方法** — 怎么改才对
5. **预防措施** — 怎么从源头避免

---

## 11.1 coroutine was never awaited — 忘记 await

这是异步编程中最常见的错误，没有之一。

### 错误重现

```python
import asyncio

async def fetch_data():
    await asyncio.sleep(1)
    return {"user": "Alice", "score": 95}

async def main():
    # 忘记 await
    result = fetch_data()    # ← 这里返回的是协程对象，不是结果！
    print(result)
    print(result["user"])    # ← TypeError!

asyncio.run(main())
```

输出：

```
<coroutine object fetch_data at 0x10a5b8c40>
/sys:1: RuntimeWarning: coroutine 'fetch_data' was never awaited
TypeError: 'coroutine' object is not subscriptable
```

### 根本原因

当你调用一个 `async def` 函数时，**不会立即执行函数体**。它返回的是一个协程对象（coroutine object），你必须用 `await` 来驱动它执行。

```
调用 async 函数
    │
    ▼
返回 coroutine 对象   ← 代码还没执行！
    │
    ▼
await coroutine        ← 现在才开始执行
    │
    ▼
得到返回值
```

### 修复方法

```python
async def main():
    # 正确：加上 await
    result = await fetch_data()
    print(result["user"])  # Alice
```

### 更隐蔽的变体

有时候你不会直接看到这个错误，因为问题藏在更深的地方：

```python
async def get_user_name():
    await asyncio.sleep(0.5)
    return "Alice"

async def get_greeting():
    # 这里忘了 await
    name = get_user_name()   # ← coroutine object!
    return f"Hello, {name}"  # "Hello, <coroutine object ...>"

async def main():
    greeting = await get_greeting()
    print(greeting)

asyncio.run(main())
```

输出：

```
/sys:1: RuntimeWarning: coroutine 'get_user_name' was never awaited
Hello, <coroutine object get_user_name at 0x10a5b8d00>
```

这段代码没有报错！它只是悄悄地返回了一个错误的结果。这种 bug 在生产环境中特别难发现，因为程序"能跑"，只是结果不对。

<div data-component="CommonErrorCatalog"></div>

### 预防措施

1. **IDE 配置**：PyCharm 和 VS Code 都能检测未 await 的协程，确保警告开启
2. **类型标注**：使用返回类型标注，类型检查器会发现类型不匹配

```python
async def get_greeting() -> str:
    name = get_user_name()   # mypy 会警告：Coroutine[Any, Any, str] 赋值给 str
    return f"Hello, {name}"
```

3. **养成条件反射**：看到 `async def` 的调用，第一反应就是加 `await`

</div>
---

## 11.2 RuntimeError: Event loop is closed — 事件循环关闭后使用

### 错误重现

```python
import asyncio

async def hello():
    print("Hello")

# 错误：在 run() 之后尝试使用协程
loop = asyncio.new_event_loop()
loop.run_until_complete(hello())
loop.close()

# 再次使用已关闭的循环
loop.run_until_complete(hello())  # ← RuntimeError!
```

输出：

```
Hello
RuntimeError: Event loop is closed
```

### 更常见的变体：跨线程使用

```python
import asyncio
import threading

loop = asyncio.new_event_loop()

def run_in_thread():
    # 在另一个线程中使用主循环
    future = asyncio.run_coroutine_threadsafe(hello(), loop)
    result = future.result()
    return result

threading.Thread(target=run_in_thread).start()
```

如果循环已经被关闭，`run_coroutine_threadsafe` 会抛出 `RuntimeError`。

### 根本原因

`asyncio.run()` 内部会创建一个新的事件循环，执行完毕后自动关闭它。`loop.close()` 是不可逆的——一旦关闭，就不能再用。

```
asyncio.run(main())
    │
    ├── 创建新事件循环
    ├── 执行 main()
    ├── 关闭事件循环    ← 不可逆！
    └── 返回
```

### 修复方法

```python
# 方法 1：使用 asyncio.run()，不手动管理循环
async def main():
    await hello()
    await hello()  # 可以多次 await

asyncio.run(main())

# 方法 2：如果必须手动管理，确保不重复使用
loop = asyncio.new_event_loop()
try:
    loop.run_until_complete(hello())
finally:
    loop.close()  # 只在最后关闭
```

### Jupyter Notebook 中的陷阱

Jupyter 和 IPython 已经运行了一个事件循环。直接用 `asyncio.run()` 会报错：

```python
# 在 Jupyter 中
import asyncio

async def hello():
    print("Hello")

asyncio.run(hello())  # RuntimeError: asyncio.run() cannot be called from a running event loop
```

解决方法：

```python
# 方法 1：使用 await 直接调用（Jupyter 自带循环）
await hello()

# 方法 2：使用 nest_asyncio（允许嵌套循环）
import nest_asyncio
nest_asyncio.apply()
asyncio.run(hello())  # 现在可以了
```

---

## 11.3 阻塞事件循环 — 在 async 上下文中使用 time.sleep

### 错误重现

```python
import asyncio
import time

async def slow_task(name, delay):
    print(f"[{name}] 开始")
    time.sleep(delay)          # ← 这会阻塞整个事件循环！
    print(f"[{name}] 结束")

async def main():
    # 期望并发执行
    await asyncio.gather(
        slow_task("A", 3),
        slow_task("B", 2),
        slow_task("C", 1),
    )

start = time.time()
asyncio.run(main())
print(f"总耗时: {time.time() - start:.1f}s")
```

输出：

```
[A] 开始
[A] 结束
[B] 开始
[B] 结束
[C] 开始
[C] 结束
总耗时: 6.0s    ← 串行执行！应该是 3s 才对
```

### 根本原因

`time.sleep()` 是一个阻塞调用。在 asyncio 中，事件循环是单线程的，**所有协程共享同一个线程**。当一个协程调用 `time.sleep()` 时，整个线程被阻塞，其他协程无法执行。

```
事件循环线程
    │
    ├── 执行协程 A
    │   └── time.sleep(3)  ← 线程被阻塞 3 秒
    │                         其他协程全部等待
    ├── 执行协程 B          ← A 结束后才能执行
    │   └── time.sleep(2)  ← 线程被阻塞 2 秒
    │
    └── 执行协程 C          ← B 结束后才能执行
        └── time.sleep(1)  ← 线程被阻塞 1 秒

总耗时 = 3 + 2 + 1 = 6 秒
```

### 修复方法

```python
async def fast_task(name, delay):
    print(f"[{name}] 开始")
    await asyncio.sleep(delay)  # ← 使用 asyncio.sleep
    print(f"[{name}] 结束")

async def main():
    await asyncio.gather(
        fast_task("A", 3),
        fast_task("B", 2),
        fast_task("C", 1),
    )

start = time.time()
asyncio.run(main())
print(f"总耗时: {time.time() - start:.1f}s")  # 3.0s
```

### 常见的阻塞陷阱

不只 `time.sleep()`，很多操作都会阻塞事件循环：

```python
# 阻塞操作一览
import time
import requests
import subprocess

async def blocking_examples():
    # 1. time.sleep
    time.sleep(1)                    # ❌ 阻塞

    # 2. 同步网络请求
    requests.get("https://...")      # ❌ 阻塞

    # 3. 同步文件 I/O
    with open("bigfile.txt") as f:
        data = f.read()              # ❌ 阻塞（对于大文件）

    # 4. subprocess.run（同步版本）
    subprocess.run(["ls", "-la"])    # ❌ 阻塞

    # 5. CPU 密集计算
    total = sum(range(10**8))        # ❌ 阻塞

    # 6. 同步数据库查询
    cursor.execute("SELECT ...")     # ❌ 阻塞
    rows = cursor.fetchall()
```

### 正确的替代方案

```python
import asyncio
import aiohttp
import aiofiles
import asyncio.subprocess

async def non_blocking_examples():
    # 1. asyncio.sleep
    await asyncio.sleep(1)                          # ✅ 非阻塞

    # 2. 异步 HTTP
    async with aiohttp.ClientSession() as session:
        async with session.get("https://...") as resp:
            data = await resp.json()                # ✅ 非阻塞

    # 3. 异步文件 I/O
    async with aiofiles.open("bigfile.txt") as f:
        data = await f.read()                       # ✅ 非阻塞

    # 4. asyncio.subprocess
    proc = await asyncio.create_subprocess_exec(
        "ls", "-la",
        stdout=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()            # ✅ 非阻塞

    # 5. CPU 密集：用 run_in_executor
    loop = asyncio.get_running_loop()
    total = await loop.run_in_executor(
        None, lambda: sum(range(10**8)))            # ✅ 非阻塞

    # 6. 异步数据库（如 asyncpg）
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT ...")       # ✅ 非阻塞
```

<div data-component="DebugModeDemo"></div>

---

</div>
## 11.4 重复使用协程对象 — cannot reuse coroutine

### 错误重现

```python
import asyncio

async def compute(x):
    await asyncio.sleep(0.1)
    return x * 2

async def main():
    coro = compute(5)      # 创建协程对象

    result1 = await coro   # 第一次 await：正常
    print(result1)         # 10

    result2 = await coro   # 第二次 await：RuntimeError!
    print(result2)

asyncio.run(main())
```

输出：

```
10
RuntimeError: cannot reuse already awaited coroutine
```

### 根本原因

协程对象是一次性的。当你 `await` 一个协程时，它的函数体被执行完毕，结果被缓存。再次 `await` 同一个协程对象没有意义——因为它已经"跑完了"。

```
coro = compute(5)
    │
    ▼
协程对象（未启动）
    │
    ├── await coro  ──→ 执行函数体 ──→ 返回 10  ──→ 协程已结束
    │
    └── await coro  ──→ RuntimeError: 协程已经结束，不能再用
```

### 修复方法

```python
async def main():
    # 每次需要结果时，创建新的协程
    result1 = await compute(5)
    result2 = await compute(5)

    # 或者把协程封装成可重复调用的函数
    async def reusable(x):
        return await compute(x)

    result3 = await reusable(5)
    result4 = await reusable(5)
```

### 与 Task 的对比

Task 是可以多次查询结果的，因为它代表的是"正在进行或已完成的工作"：

```python
async def main():
    # Task 可以多次查询
    task = asyncio.create_task(compute(5))

    result = await task   # 等待完成
    print(result)         # 10

    # Task 已完成，但 .result() 仍然可以调用
    print(task.result())  # 10  ✅ 不报错
```

### 常见的误用模式

```python
# ❌ 错误：把协程存起来复用
async def main():
    coro = fetch_user_data()    # 创建一次
    data1 = await coro          # 第一次成功
    data2 = await coro          # 💥 RuntimeError

# ✅ 正确：每次重新调用
async def main():
    data1 = await fetch_user_data()
    data2 = await fetch_user_data()
```

---

## 11.5 忘记等待任务 — task created but not awaited

### 错误重现

```python
import asyncio

async def background_job():
    print("后台任务开始")
    await asyncio.sleep(2)
    print("后台任务完成")
    return 42

async def main():
    # 创建任务但忘记等待
    task = asyncio.create_task(background_job())
    print("主函数继续执行")
    # 函数直接结束，后台任务可能未完成！

asyncio.run(main())
```

输出：

```
主函数继续执行
后台任务开始
```

注意：`后台任务完成` 和 `42` 不见了！因为 `main()` 结束后，`asyncio.run()` 会取消所有未完成的任务。

### 根本原因

`create_task()` 把协程注册到事件循环中并发执行，但**不会阻塞当前协程**。如果你不 `await` 这个 task，你无法知道它是否完成，也无法获取它的返回值。

```
main()                          background_job()
    │                               │
    ├── create_task()  ───────────→ │ 开始执行
    │                               │
    ├── print("主函数继续")         │ await sleep(2)
    │                               │
    └── main() 返回                 │ 还在 sleep...
                                    │
                                    └── 被取消！（main 已退出）
```

### 修复方法

```python
async def main():
    task = asyncio.create_task(background_job())

    # 方法 1：等待任务完成
    result = await task
    print(f"结果: {result}")

    # 方法 2：在适当的时候等待
    # 做一些其他工作...
    await asyncio.sleep(1)
    result = await task  # 然后再等
```

### 批量任务的正确等待

```python
async def main():
    # 创建多个任务
    tasks = [
        asyncio.create_task(background_job())
        for _ in range(5)
    ]

    # 方法 1：等待所有完成
    results = await asyncio.gather(*tasks)

    # 方法 2：逐个等待
    for task in tasks:
        result = await task
        print(result)

    # 方法 3：使用 asyncio.wait（更灵活）
    done, pending = await asyncio.wait(tasks)
    for task in done:
        print(task.result())
```

### 不需要等待的场景

有些场景确实不需要等待 task，但你需要知道后果：

```python
async def main():
    # 场景：发送通知，不关心结果
    # 用 shield 或 ensure_future 存储到某处
    fire_and_forget = asyncio.create_task(send_notification())

    # 如果 main() 即将结束，需要给任务一点时间
    await asyncio.sleep(0.1)  # 让事件循环有机会执行其他任务
```

更好的模式是用一个"管理器"来收集所有后台任务：

```python
class TaskManager:
    def __init__(self):
        self.tasks = set()

    def create_task(self, coro):
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    async def wait_all(self):
        if self.tasks:
            await asyncio.gather(*self.tasks)

async def main():
    manager = TaskManager()
    manager.create_task(background_job())
    manager.create_task(background_job())

    # 做其他工作...

    # 结束前等待所有任务
    await manager.wait_all()
```

---

## 11.6 异常被吞掉 — unhandled task exceptions

### 错误重现

```python
import asyncio

async def risky_operation():
    await asyncio.sleep(0.5)
    raise ValueError("出错了！")

async def main():
    task = asyncio.create_task(risky_operation())

    # 做其他事情...
    await asyncio.sleep(1)
    print("主函数完成")

asyncio.run(main())
```

输出：

```
主函数完成
Task exception was never retrieved
future: <Task finished name='Task-2' coro=<risky_operation() done, defined at ...> exception=ValueError('出错了！')>
ValueError: 出错了！
```

注意：异常信息是作为**警告**打印的，不是作为程序崩溃。如果你不看终端日志，根本不知道出了问题。

### 根本原因

Task 中的异常不会自动传播到父协程。Task 完成时如果包含异常，这个异常会被存储在 Task 对象中。如果你不 `await` 这个 Task，异常就"静默"了。

```
risky_operation()           main()
    │                           │
    ├── raise ValueError        ├── create_task()
    │   │                       │
    │   ▼                       │ 继续执行...
    │   异常存储在 Task 中       │
    │                           ├── print("主函数完成")
    │                           └── main() 返回
    │
    └── 警告：exception was never retrieved
```

### 修复方法

```python
async def main():
    task = asyncio.create_task(risky_operation())

    try:
        result = await task
    except ValueError as e:
        print(f"捕获到异常: {e}")

asyncio.run(main())
```

### gather 中的异常处理

```python
async def main():
    # gather 默认在第一个异常时抛出，其他任务继续执行
    tasks = [
        asyncio.create_task(risky_operation()),
        asyncio.create_task(risky_operation()),
    ]

    # 方法 1：gather + return_exceptions
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            print(f"任务失败: {result}")
        else:
            print(f"任务成功: {result}")

    # 方法 2：gather 默认行为
    try:
        results = await asyncio.gather(*tasks)
    except ValueError as e:
        print(f"至少一个任务失败: {e}")
        # 其他任务仍在运行！
```

### 全局异常处理器

```python
import asyncio
import logging

logging.basicConfig(level=logging.WARNING)

def handle_exception(loop, context):
    # 自定义异常处理
    exception = context.get("exception")
    message = context.get("message")

    if exception:
        logging.error(f"未处理的异常: {exception}")
    else:
        logging.error(f"异常消息: {message}")

async def main():
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(handle_exception)

    task = asyncio.create_task(risky_operation())
    await asyncio.sleep(1)

asyncio.run(main())
```

<div data-component="CommonErrorCatalog"></div>

---

</div>
## 11.7 死锁 — Lock acquired twice

### 错误重现

```python
import asyncio

async def main():
    lock = asyncio.Lock()

    async with lock:
        print("外层锁获取成功")
        async with lock:    # ← 死锁！永远等不到锁释放
            print("内层锁获取成功")

asyncio.run(main())
```

这个程序会**永远挂起**，不会超时，不会报错，只是静静地等待。

### 根本原因

`asyncio.Lock` 不是可重入锁（reentrant lock）。当你用 `async with lock` 获取锁时，锁的状态从"未锁定"变为"锁定"。当你再次尝试获取时，当前协程会等待锁释放——但锁是被自己持有的，形成死锁。

```
协程 main
    │
    ├── 获取 lock  ✓
    │   │
    │   └── 再次获取 lock
    │       │
    │       └── 等待 lock 释放...
    │           │
    │           └── lock 被 main 持有
    │               │
    │               └── main 在等待自己释放锁
    │                   │
    │                   └── 永远等不到！（死锁）
```

### 修复方法

```python
# 方法 1：避免嵌套锁
async def main():
    lock = asyncio.Lock()

    async with lock:
        print("获取锁，执行操作")
        # 所有需要锁的操作都在这一层
        print("释放锁")

# 方法 2：重构代码，拆分需要锁的部分
async def main():
    lock = asyncio.Lock()

    async def inner_operation():
        async with lock:
            return "inner result"

    async with lock:
        print("外层操作")
        # 需要调用 inner 的逻辑移到锁外面
    result = await inner_operation()
    print(f"内层结果: {result}")

# 方法 3：使用可重入锁（注意：asyncio 没有内置 RLock）
import threading
# 如果必须用可重入锁，考虑用 threading.RLock（但这会阻塞事件循环）
# 更好的方案是重构代码避免嵌套
```

### 更隐蔽的死锁

```python
async def transfer(from_account, to_account, amount):
    # 两个账户之间转账，需要锁定两个账户
    async with from_account.lock:
        async with to_account.lock:  # ← 如果两个协程同时调用
            from_account.balance -= amount
            to_account.balance += amount

# 死锁场景：
# 协程 A: transfer(alice, bob, 100)  — 获取 alice.lock，等待 bob.lock
# 协程 B: transfer(bob, alice, 50)   — 获取 bob.lock，等待 alice.lock
# 互相等待 → 死锁
```

修复方案——按固定顺序获取锁：

```python
async def transfer(from_account, to_account, amount):
    # 按 ID 排序，确保获取锁的顺序一致
    accounts = sorted([from_account, to_account], key=lambda a: a.id)
    first, second = accounts

    async with first.lock:
        async with second.lock:
            from_account.balance -= amount
            to_account.balance += amount
```

### 设置超时避免永久阻塞

```python
async def main():
    lock = asyncio.Lock()

    try:
        # 等待锁最多 5 秒
        await asyncio.wait_for(lock.acquire(), timeout=5.0)
        try:
            print("获取到锁")
        finally:
            lock.release()
    except asyncio.TimeoutError:
        print("获取锁超时，可能存在死锁")
```

---

## 11.8 竞态条件 — shared state without Lock

### 错误重现

```python
import asyncio

counter = 0

async def increment(n):
    global counter
    for _ in range(n):
        # 读取当前值
        current = counter
        # 模拟一些处理
        await asyncio.sleep(0.001)
        # 写回新值
        counter = current + 1

async def main():
    await asyncio.gather(
        increment(100),
        increment(100),
        increment(100),
    )
    print(f"期望: 300, 实际: {counter}")

asyncio.run(main())
```

输出：

```
期望: 300, 实际: 87   ← 每次运行结果不同！
```

### 根本原因

在 `current = counter` 和 `counter = current + 1` 之间，另一个协程可能已经修改了 `counter`。这就是经典的"读-改-写"竞态条件。

```
协程 A                              协程 B
    │                                   │
    ├── current = counter (=10)         │
    │                                   ├── current = counter (=10)
    │   await sleep...                  │
    │                                   │   await sleep...
    ├── counter = current + 1 (=11)     │
    │                                   ├── counter = current + 1 (=11)
    │
    └── 期望 12，实际 11 ← 丢失了一次增量
```

<div data-component="RaceConditionDemo"></div>

### 修复方法

```python
import asyncio

counter = 0
counter_lock = asyncio.Lock()

async def safe_increment(n):
    global counter
    for _ in range(n):
        async with counter_lock:
            current = counter
            await asyncio.sleep(0.001)  # 即使有 await，锁保护了临界区
            counter = current + 1

async def main():
    await asyncio.gather(
        safe_increment(100),
        safe_increment(100),
        safe_increment(100),
    )
    print(f"期望: 300, 实际: {counter}")  # 总是 300

asyncio.run(main())
```

### 什么时候不需要锁？

如果临界区中**没有 `await`**，在单线程 asyncio 中实际上是安全的：

```python
async def safe_increment_no_await(n):
    global counter
    for _ in range(n):
        # 没有 await，不会被打断
        counter += 1  # 这在 CPython 中是原子的
```

但是！依赖这一点是危险的：
1. 未来可能有人添加 `await`
2. 不同 Python 实现可能不同
3. 代码意图不明确

**最佳实践**：只要涉及共享可变状态，就用 Lock。

### 竞态条件的其他形式

```python
# 1. 检查后执行（TOCTOU）
async def withdraw(account, amount):
    if account.balance >= amount:    # 检查
        await asyncio.sleep(0.1)     # 期间余额可能改变
        account.balance -= amount    # 执行 ← 可能透支！

# 2. 文件写入竞态
async def write_config(data):
    with open("config.json", "r") as f:
        config = json.load(f)
    config.update(data)
    await asyncio.sleep(0.1)         # 期间另一个协程可能也在写
    with open("config.json", "w") as f:
        json.dump(config, f)         # 覆盖了另一个协程的写入
```
</div>

---

## 11.9 内存泄漏 — tasks never collected

### 错误重现

```python
import asyncio
import weakref

async def long_running_task():
    await asyncio.sleep(3600)  # 模拟长时间运行

async def main():
    # 不断创建任务但不管理它们
    tasks = []
    for i in range(10000):
        task = asyncio.create_task(long_running_task())
        tasks.append(task)  # 保持引用
        # 从不 await，也从不取消

    # 主函数结束，但 10000 个任务仍在运行
    await asyncio.sleep(1)

asyncio.run(main())
```

### 根本原因

每个 `create_task()` 都会在事件循环中注册一个 Task 对象。如果你创建了任务但：
- 不 `await` 它们
- 不取消它们
- 不释放对它们的引用

这些 Task 对象会一直占用内存，直到事件循环结束。

<div data-component="MemoryLeakDetector"></div>

### 常见的泄漏模式

```python
# 模式 1：无限制的任务创建
class Server:
    def __init__(self):
        self.connections = []  # 不断增长！

    async def handle_connection(self, reader, writer):
        # 每个连接创建一个任务
        task = asyncio.create_task(self._process(reader, writer))
        self.connections.append(task)  # 只 append，从不清理

# 模式 2：缓存协程结果但不清理
class Cache:
    def __init__(self):
        self.cache = {}

    async def get(self, key):
        if key not in self.cache:
            # 创建任务并缓存
            self.cache[key] = asyncio.create_task(self._fetch(key))
        return await self.cache[key]  # 缓存从不清除！

# 模式 3：事件监听器不移除
class EventEmitter:
    def __init__(self):
        self.listeners = defaultdict(list)

    def on(self, event, callback):
        self.listeners[event].append(callback)  # 只添加，不移除
```

### 修复方法

```python
# 修复模式 1：使用任务组或限制并发
class Server:
    def __init__(self):
        self.active_tasks = set()

    async def handle_connection(self, reader, writer):
        task = asyncio.create_task(self._process(reader, writer))
        self.active_tasks.add(task)
        task.add_done_callback(self.active_tasks.discard)

    async def shutdown(self):
        for task in self.active_tasks:
            task.cancel()
        await asyncio.gather(*self.active_tasks, return_exceptions=True)

# 修复模式 2：使用 LRU 缓存
from functools import lru_cache

class Cache:
    def __init__(self, maxsize=128):
        self.cache = {}

    async def get(self, key):
        if key not in self.cache:
            if len(self.cache) >= 128:
                # 移除最旧的
                oldest = next(iter(self.cache))
                del self.cache[oldest]
            self.cache[key] = await self._fetch(key)
        return self.cache[key]

# 修复模式 3：提供移除方法
class EventEmitter:
    def on(self, event, callback):
        self.listeners[event].append(callback)
        # 返回一个取消函数
        def unsubscribe():
            self.listeners[event].remove(callback)
        return unsubscribe
```

### 检测内存泄漏

```python
import asyncio
import gc
import tracemalloc

async def main():
    # 开始追踪内存
    tracemalloc.start()

    # 创建一些任务
    tasks = [asyncio.create_task(asyncio.sleep(1)) for _ in range(100)]

    # 查看内存使用
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("[ Top 10 memory consumers ]")
    for stat in top_stats[:10]:
        print(stat)

    # 清理
    for task in tasks:
        task.cancel()

asyncio.run(main())
```
</div>

---

## 11.10 调试工具 — asyncio debug mode

asyncio 内置了强大的调试模式，可以自动发现很多常见问题。

### 启用方式

```python
# 方法 1：环境变量
# PYTHONASYNCIODEBUG=1 python script.py

# 方法 2：代码中启用
asyncio.run(main(), debug=True)

# 方法 3：手动设置
loop = asyncio.get_event_loop()
loop.set_debug(True)
```

### Debug 模式会检查什么？

<div data-component="DebugModeDemo"></div>

```python
import asyncio
import time

async def bad_coroutine():
    # 故意的阻塞操作
    time.sleep(1)  # 在 debug 模式下会有警告

async def unawaited_coroutine():
    await asyncio.sleep(1)

async def main():
    # 1. 检测未等待的协程
    coro = unawaited_coroutine()  # ← 警告：coroutine was never awaited

    # 2. 检测阻塞调用
    await bad_coroutine()  # ← 警告：Executing bad_coroutine took 1.0 seconds

    # 3. 检测长时间运行的回调
    # 如果某个回调执行超过 100ms，会打印警告

asyncio.run(main(), debug=True)
```

### Debug 模式的输出

```
/sys:1: RuntimeWarning: coroutine 'unawaited_coroutine' was never awaited
Executing <Task finished name='Task-2' coro=<bad_coroutine() ...>> took 1.000 seconds
```

### 自定义慢回调检测

```python
import asyncio
import logging

logging.basicConfig(level=logging.WARNING)

async def main():
    loop = asyncio.get_running_loop()

    # 设置慢回调阈值（秒）
    loop.slow_callback_duration = 0.1  # 100ms

    # 所有执行超过 100ms 的回调都会被记录
    await asyncio.sleep(0.05)   # 正常
    time.sleep(0.2)             # ← 警告：took 200ms

asyncio.run(main(), debug=True)
```

### 使用 faulthandler 调试挂起

```python
import asyncio
import faulthandler
import signal

# 启用 faulthandler
faulthandler.enable()

# 或者注册信号处理器
def print_traceback(sig, frame):
    faulthandler.dump_traceback()

# 按 Ctrl+\ (SIGQUIT) 打印所有线程的调用栈
signal.signal(signal.SIGQUIT, print_traceback)

async def main():
    # 如果程序挂起，按 Ctrl+\ 查看调用栈
    lock = asyncio.Lock()
    async with lock:
        async with lock:  # 死锁！
            pass

asyncio.run(main())
```
</div>

---

## 11.11 logging 模块 — async-aware logging

### 基本设置

```python
import asyncio
import logging

# 配置 logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

async def fetch_data(url):
    logger.info(f"开始获取 {url}")
    await asyncio.sleep(1)
    logger.info(f"完成获取 {url}")
    return {"data": "..."}

async def main():
    logger.info("主函数开始")
    result = await fetch_data("https://example.com")
    logger.info(f"结果: {result}")

asyncio.run(main())
```

输出：

```
2026-07-07 10:00:00 [__main__] INFO: 主函数开始
2026-07-07 10:00:00 [__main__] INFO: 开始获取 https://example.com
2026-07-07 10:00:01 [__main__] INFO: 完成获取 https://example.com
2026-07-07 10:00:01 [__main__] INFO: 结果: {'data': '...'}
```

### 在异步上下文中使用 logging 的注意事项

```python
import asyncio
import logging
import time

logger = logging.getLogger(__name__)

async def main():
    # ❌ 避免在日志中进行阻塞操作
    # 如果日志处理器涉及文件 I/O，在高频场景下可能成为瓶颈

    # ❌ 避免在日志消息中调用异步函数
    # logger.info(f"结果: {await fetch_data()}")  # 不推荐

    # ✅ 先获取数据，再记录
    result = await fetch_data()
    logger.info(f"结果: {result}")

    # ✅ 使用 lazy formatting
    logger.debug("处理用户 %s, ID: %d", username, user_id)
    # 而不是
    logger.debug(f"处理用户 {username}, ID: {user_id}")  # 即使 DEBUG 关闭也会计算
```

### 创建异步友好的日志处理器

```python
import asyncio
import logging
import queue
from logging.handlers import QueueHandler, QueueListener

def setup_async_logging():
    # 创建队列
    log_queue = queue.Queue()

    # 创建队列处理器
    queue_handler = QueueHandler(log_queue)

    # 创建实际的处理器（文件、控制台等）
    file_handler = logging.FileHandler("app.log")
    console_handler = logging.StreamHandler()

    # 创建监听器，异步处理日志
    listener = QueueListener(
        log_queue,
        file_handler,
        console_handler,
        respect_handler_level=True
    )

    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(queue_handler)

    # 启动监听器
    listener.start()

    return listener

async def main():
    listener = setup_async_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info("应用启动")
        await asyncio.sleep(1)
        logger.info("应用结束")
    finally:
        listener.stop()

asyncio.run(main())
```

### 记录协程执行时间

```python
import asyncio
import logging
import functools
import time

logger = logging.getLogger(__name__)

def log_execution_time(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.monotonic()
        try:
            result = await func(*args, **kwargs)
            elapsed = time.monotonic() - start
            logger.info(f"{func.__name__} 完成，耗时 {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.monotonic() - start
            logger.error(f"{func.__name__} 失败，耗时 {elapsed:.3f}s，错误: {e}")
            raise
    return wrapper

@log_execution_time
async def slow_operation():
    await asyncio.sleep(2)
    return "done"
```

---

## 11.12 性能分析 — cProfile with async

### 标准 cProfile 的局限

```python
import cProfile
import asyncio

async def main():
    await asyncio.sleep(1)
    await asyncio.gather(
        asyncio.sleep(0.5),
        asyncio.sleep(0.5),
    )

# cProfile 对 async 代码的分析结果有限
cProfile.run('asyncio.run(main())')
```

输出会显示 `asyncio.run` 和 `_run_once` 的耗时，但不会细分到每个协程。

### 使用 py-spy 进行采样分析

```bash
# 安装 py-spy
pip install py-spy

# 实时监控
py-spy top -- python script.py

# 生成火焰图
py-spy record -o profile.svg -- python script.py
```

### 自定义协程级性能分析

```python
import asyncio
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List

class AsyncProfiler:
    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self._start_times: Dict[str, float] = {}

    @contextmanager
    def track(self, name: str):
        start = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - start
            self.timings[name].append(elapsed)

    async def track_async(self, name: str, coro):
        start = time.monotonic()
        try:
            return await coro
        finally:
            elapsed = time.monotonic() - start
            self.timings[name].append(elapsed)

    def report(self):
        print("\n" + "=" * 60)
        print(f"{'操作':<30} {'次数':>6} {'总耗时':>10} {'平均':>10} {'最大':>10}")
        print("-" * 60)

        for name, times in sorted(self.timings.items()):
            count = len(times)
            total = sum(times)
            avg = total / count
            max_time = max(times)
            print(f"{name:<30} {count:>6} {total:>10.3f}s {avg:>10.3f}s {max_time:>10.3f}s")

        print("=" * 60)

# 使用示例
profiler = AsyncProfiler()

async def fetch_user(user_id):
    async with profiler.track_async(f"fetch_user_{user_id}", asyncio.sleep(0.1)):
        return {"id": user_id, "name": f"User_{user_id}"}

async def main():
    # 使用 profiler
    with profiler.track("total_main"):
        users = await asyncio.gather(*[
            fetch_user(i) for i in range(10)
        ])

    profiler.report()

asyncio.run(main())
```

输出：

```
============================================================
操作                              次数       总耗时         平均         最大
------------------------------------------------------------
fetch_user_0                         1      0.100s      0.100s      0.100s
fetch_user_1                         1      0.100s      0.100s      0.100s
...
total_main                           1      0.102s      0.102s      0.102s
============================================================
```

### 使用 cProfile + asyncio 的正确姿势

```python
import cProfile
import asyncio
import pstats
from io import StringIO

async def workload():
    tasks = [asyncio.sleep(0.01) for _ in range(100)]
    await asyncio.gather(*tasks)

def profile_async():
    profiler = cProfile.Profile()
    profiler.enable()

    asyncio.run(workload())

    profiler.disable()

    # 打印统计信息
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    print(stream.getvalue())

profile_async()
```

### 使用 yappi（Yet Another Python Profiler）

```bash
pip install yappi
```

```python
import asyncio
import yappi

async def cpu_bound():
    return sum(range(10**6))

async def io_bound():
    await asyncio.sleep(0.1)

async def main():
    await asyncio.gather(
        cpu_bound(),
        io_bound(),
        cpu_bound(),
    )

# 启动 yappi
yappi.set_clock_type("wall")  # 使用 wall 时间
yappi.start()

asyncio.run(main())

# 停止并打印结果
yappi.stop()

# 获取协程级别的统计
print("协程级统计:")
yappi.get_func_stats().print_all()

# 获取线程级统计
print("\n线程级统计:")
yappi.get_thread_stats().print_all()
```

---

## 11.13 最佳实践总结

<div data-component="BestPracticesChecklist"></div>

### 原则 1：始终使用 await

```python
# ❌ 危险
result = async_function()

# ✅ 安全
result = await async_function()
```

### 原则 2：不要阻塞事件循环

```python
# ❌ 阻塞
time.sleep(1)
requests.get(url)

# ✅ 非阻塞
await asyncio.sleep(1)
async with aiohttp.ClientSession() as session:
    async with session.get(url) as resp:
        await resp.json()
```

### 原则 3：正确处理异常

```python
# ❌ 忽略异常
asyncio.create_task(risky_operation())

# ✅ 等待并处理
try:
    await asyncio.create_task(risky_operation())
except Exception as e:
    logger.error(f"操作失败: {e}")
```

### 原则 4：保护共享状态

```python
# ❌ 竞态条件
counter += 1

# ✅ 使用锁
async with lock:
    counter += 1
```

### 原则 5：限制并发数量

```python
# ❌ 无限并发
await asyncio.gather(*[fetch(url) for url in urls])

# ✅ 使用信号量
sem = asyncio.Semaphore(10)

async def limited_fetch(url):
    async with sem:
        return await fetch(url)

await asyncio.gather(*[limited_fetch(url) for url in urls])
```

### 原则 6：正确关闭资源

```python
# ❌ 不关闭
session = aiohttp.ClientSession()
await session.get(url)

# ✅ 使用 async with
async with aiohttp.ClientSession() as session:
    async with session.get(url) as resp:
        data = await resp.json()
```

### 原则 7：使用超时

```python
# ❌ 无超时
result = await fetch_data()

# ✅ 设置超时
try:
    result = await asyncio.wait_for(fetch_data(), timeout=5.0)
except asyncio.TimeoutError:
    logger.error("请求超时")
```

### 原则 8：使用 TaskGroup（Python 3.11+）

```python
# Python 3.11+ 推荐
async def main():
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(fetch_data("url1"))
        task2 = tg.create_task(fetch_data("url2"))
        task3 = tg.create_task(fetch_data("url3"))

    # 所有任务完成或至少一个失败时退出
    results = [task1.result(), task2.result(), task3.result()]
```

### 原则 9：记录关键操作

```python
import logging

logger = logging.getLogger(__name__)

async def important_operation():
    logger.info("开始重要操作")
    try:
        result = await do_work()
        logger.info(f"操作成功: {result}")
        return result
    except Exception as e:
        logger.exception(f"操作失败: {e}")
        raise
```

### 原则 10：测试异步代码

```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_fetch_data():
    result = await fetch_data()
    assert result is not None

@pytest.mark.asyncio
async def test_timeout():
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(slow_operation(), timeout=0.1)
```
</div>

---

## 11.14 常见误区

### 误区 1：async 代码一定比同步快

**事实**：async 代码在 I/O 密集型任务中有优势，但在 CPU 密集型任务中可能更慢（因为协程调度的开销）。

```python
# ❌ 误用：CPU 密集型任务用 async
async def cpu_intensive():
    return sum(range(10**8))  # 没有任何 I/O，用 async 没意义

# ✅ 正确：CPU 密集型用 ProcessPoolExecutor
import concurrent.futures

async def main():
    loop = asyncio.get_running_loop()
    with concurrent.futures.ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, lambda: sum(range(10**8)))
```

### 误区 2：await 会创建新线程

**事实**：`await` 只是暂停当前协程，让出控制权给事件循环。它不会创建新线程。

```python
async def main():
    # 这些都在同一个线程中执行
    await asyncio.sleep(1)     # 暂停，事件循环去执行其他协程
    await fetch_data()         # 暂停，等待网络响应

    # 如果需要新线程，显式使用
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, blocking_function)
```

### 误区 3：所有库都有异步版本

**事实**：很多库只有同步版本。如果必须使用，需要在 executor 中运行。

```python
# 同步库的正确使用方式
import asyncio
import requests  # 同步库

async def fetch_url(url):
    loop = asyncio.get_running_loop()
    # 在线程池中执行同步调用
    response = await loop.run_in_executor(
        None,
        lambda: requests.get(url)
    )
    return response.json()
```

### 误区 4：asyncio.gather 会自动限制并发

**事实**：`gather` 会立即启动所有协程，不会限制并发数。

```python
# ❌ 危险：如果 urls 有 10000 个，会同时发起 10000 个请求
results = await asyncio.gather(*[fetch(url) for url in urls])

# ✅ 正确：使用信号量限制并发
sem = asyncio.Semaphore(100)

async def safe_fetch(url):
    async with sem:
        return await fetch(url)

results = await asyncio.gather(*[safe_fetch(url) for url in urls])
```

### 误区 5：异步代码不需要锁

**事实**：虽然 asyncio 是单线程的，但 `await` 会让出控制权，导致竞态条件。

```python
# ❌ 危险：await 让出控制权
async def increment():
    global counter
    current = counter
    await asyncio.sleep(0)  # 让出控制权，其他协程可能修改 counter
    counter = current + 1   # 结果不确定

# ✅ 安全：使用锁
async def safe_increment():
    global counter
    async with lock:
        current = counter
        await asyncio.sleep(0)
        counter = current + 1
```

### 误区 6：异常会自动传播

**事实**：Task 中的异常不会自动传播，需要显式 await 或设置异常处理器。

```python
# ❌ 异常被吞掉
async def main():
    task = asyncio.create_task(risky_operation())
    await asyncio.sleep(1)  # 如果 task 抛异常，这里不会知道

# ✅ 正确处理
async def main():
    task = asyncio.create_task(risky_operation())
    try:
        await task
    except Exception as e:
        print(f"任务失败: {e}")
```

### 误区 7：close() 可以取消所有任务

**事实**：`loop.close()` 不会取消任务，它只是关闭事件循环。

```python
async def main():
    task = asyncio.create_task(long_running())

    await asyncio.sleep(1)
    # 不需要手动 close，asyncio.run() 会处理
```

### 误区 8：同步代码可以直接在异步函数中调用

**事实**：可以调用，但如果它是阻塞的，会阻塞整个事件循环。

```python
async def main():
    # 技术上可以，但会阻塞事件循环
    import time
    time.sleep(1)  # ❌ 阻塞

    # 正确做法
    await asyncio.sleep(1)  # ✅ 非阻塞

    # 或者在 executor 中运行
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, time.sleep, 1)  # ✅ 非阻塞
```

---

## 本章小结

本章系统梳理了 Python asyncio 编程中的常见错误、调试工具和最佳实践：

| 错误类型 | 根本原因 | 修复方法 |
|---------|---------|---------|
| coroutine was never awaited | 忘记 `await` | 对所有异步调用加 `await` |
| Event loop is closed | 重复使用已关闭的循环 | 使用 `asyncio.run()` 代替手动管理 |
| 阻塞事件循环 | `time.sleep` 等同步调用 | 使用异步替代或 `run_in_executor` |
| cannot reuse coroutine | 重复 await 同一协程对象 | 每次需要结果时重新调用 |
| task not awaited | 创建任务但不等待 | 使用 `gather` 或显式 `await` |
| 异常被吞掉 | Task 异常未被检索 | `await` Task 或设置异常处理器 |
| 死锁 | 锁的嵌套获取 | 避免嵌套或按固定顺序获取锁 |
| 竞态条件 | 共享状态无保护 | 使用 `asyncio.Lock` 保护临界区 |
| 内存泄漏 | 任务未清理 | 使用任务组、设置超时、及时取消 |

**调试工具链**：
- `asyncio.run(debug=True)`：启用内置调试模式
- `logging`：记录关键操作和异常
- `py-spy` / `yappi`：协程级性能分析
- `tracemalloc`：内存泄漏检测
- `faulthandler`：死锁时打印调用栈

**核心最佳实践**：
1. 始终 `await` 异步调用
2. 不在 async 上下文中使用阻塞操作
3. 正确处理和传播异常
4. 保护共享可变状态
5. 限制并发数量
6. 设置超时
7. 正确关闭资源
8. 记录关键操作
9. 测试边界情况
10. 使用 Python 3.11+ 的 `TaskGroup`

---

## 思考题

1. **基础理解**：以下代码有什么问题？如何修复？

```python
import asyncio

async def get_data():
    await asyncio.sleep(1)
    return [1, 2, 3]

async def main():
    data = get_data()
    for item in data:
        print(item)

asyncio.run(main())
```

2. **调试分析**：下面的代码运行时输出 `期望: 1000, 实际: ??`，实际值每次不同。为什么？如何修复？

```python
import asyncio

counter = 0

async def increment(n):
    global counter
    for _ in range(n):
        current = counter
        await asyncio.sleep(0.001)
        counter = current + 1

async def main():
    await asyncio.gather(*[increment(100) for _ in range(10)])
    print(f"期望: 1000, 实际: {counter}")

asyncio.run(main())
```

3. **死锁检测**：以下代码会怎样？为什么？

```python
import asyncio

async def worker(lock):
    async with lock:
        print("获取锁")
        await asyncio.sleep(1)
        async with lock:
            print("再次获取锁")

async def main():
    lock = asyncio.Lock()
    await worker(lock)

asyncio.run(main())
```

4. **内存泄漏**：这个 Web 服务器会有什么问题？如何修复？

```python
import asyncio

class WebServer:
    def __init__(self):
        self.handlers = []

    async def handle_request(self, request):
        await asyncio.sleep(0.1)  # 模拟处理
        return "response"

    async def run(self):
        while True:
            request = await self.get_request()
            # 每个请求创建一个任务
            task = asyncio.create_task(self.handle_request(request))
            self.handlers.append(task)  # 只添加，不清理
```

5. **最佳实践**：重构以下代码，使其符合异步编程最佳实践：

```python
import asyncio
import requests
import time

async def fetch_all(urls):
    results = []
    for url in urls:
        response = requests.get(url)  # 同步调用
        results.append(response.json())
        time.sleep(1)  # 限速
    return results

async def main():
    urls = [f"https://api.example.com/{i}" for i in range(100)]
    data = await fetch_all(urls)
    print(f"获取了 {len(data)} 条数据")

asyncio.run(main())
```

6. **异常处理**：以下代码中，如果 `fetch_data` 抛出异常，会发生什么？如何确保所有异常都被正确处理？

```python
import asyncio

async def fetch_data(url):
    if url == "https://error.com":
        raise ValueError("无效的 URL")
    await asyncio.sleep(0.1)
    return {"url": url}

async def main():
    urls = [
        "https://good1.com",
        "https://error.com",
        "https://good2.com",
    ]

    tasks = [asyncio.create_task(fetch_data(url)) for url in urls]
    # 没有 await，也没有异常处理

    print("主函数完成")

asyncio.run(main())
```

7. **性能优化**：分析以下代码的性能问题，并提出优化方案：

```python
import asyncio

async def process_item(item):
    # 模拟 CPU 密集型计算
    result = sum(range(item * 10000))
    # 模拟 I/O 操作
    await asyncio.sleep(0.1)
    return result

async def main():
    items = list(range(1000))
    results = await asyncio.gather(*[process_item(item) for item in items])
    print(f"处理了 {len(results)} 个项目")

asyncio.run(main())
```

8. **设计思考**：为什么 Python 的 asyncio.Lock 不是可重入锁？在什么场景下可重入锁是必要的？如何在 asyncio 中实现类似的功能？

9. **高级场景**：设计一个带有以下功能的异步任务管理器：
   - 限制最大并发数
   - 任务超时处理
   - 失败重试（最多 3 次）
   - 优雅关闭（等待正在运行的任务完成）
   - 任务进度追踪

10. **生产实践**：假设你正在开发一个高并发的 Web 爬虫，需要：
    - 每秒最多爬取 10 个页面
    - 每个页面最多等待 5 秒
    - 遇到错误自动重试 3 次
    - 记录爬取进度和统计信息
    - 支持优雅中断（Ctrl+C）
    - 检测并避免重复爬取

    请设计这个爬虫的核心架构，使用本章学到的最佳实践。

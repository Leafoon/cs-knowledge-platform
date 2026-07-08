# 第八章 异步同步原语

异步编程赋予了我们高并发的能力，但当多个协程共享同一份数据时，"谁先谁后"的问题并不会因为
换了异步模型就消失。本章系统介绍 `asyncio` 提供的同步原语——Lock、Semaphore、Event、
Condition、Queue、Barrier——并用大量示例演示它们的正确用法和常见陷阱。

---

## 8.1 为什么异步也需要同步

很多初学者有一个误解：既然协程是"协作式"的（只在 `await` 处切换），那是不是就不会有
并发问题？答案是否定的。

### 8.1.1 共享状态的风险

当多个协程同时读写同一变量时，即使没有线程切换，`await` 点仍然会导致交错执行：

```python
import asyncio

counter = 0

async def increment(n):
    global counter
    for _ in range(n):
        temp = counter          # 读
        await asyncio.sleep(0)  # ← 这里可能发生切换
        counter = temp + 1      # 写

async def main():
    await asyncio.gather(
        increment(1000),
        increment(1000),
    )
    print(f"counter = {counter}")  # 期望 2000，实际往往更少

asyncio.run(main())
```

关键点在于 `await asyncio.sleep(0)` 这一行——它是一个显式的让出点（yield point）。
在 `await` 前后，另一个协程可能已经修改了 `counter`，导致写回时覆盖了对方的更新。

### 8.1.2 竞态条件的定义

**竞态条件**（race condition）：程序的最终结果依赖于协程的执行顺序，而这个顺序是不确定的。

- **读-改-写**（read-modify-write）是最经典的竞态模式
- 即使没有 `await`，如果数据被多个任务共享，未来的维护者可能加入 `await` 操作
- 防御原则：**任何共享可变状态都应该被同步原语保护**

### 8.1.3 协程中的让出点

只有遇到 `await` 时，协程才会让出控制权。这意味着两个 `await` 之间的代码是"原子"的——
但这并不意味着你可以放心大胆地不做同步：

```python
# 这段代码是安全的，因为中间没有 await
counter += 1  # 读-改-写在一行内完成，没有让出点

# 但下面这段就有风险：
temp = counter
await something()  # 让出点
counter = temp + 1
```

### 8.1.4 为什么不能用 threading.Lock

`threading.Lock()` 是为操作系统线程设计的。在协程中使用它：

- 会阻塞整个事件循环线程（因为 `lock.acquire()` 不是 `await`）
- 违背了异步编程"不阻塞"的核心理念
- 可能导致死锁：持有线程锁的协程被阻塞，事件循环无法调度其他协程释放资源

`asyncio` 提供了专门的同步原语，它们的 `acquire()` 方法是协程，会在等待时让出控制权。

### 8.1.5 异步同步原语的设计哲学

`asyncio` 的同步原语与 `threading` 模块一一对应，但有根本区别：

| 特性 | threading 原语 | asyncio 原语 |
|------|---------------|-------------|
| 等待方式 | 阻塞操作系统线程 | `await` 让出控制权 |
| 安全性 | 跨线程安全 | 单线程内协作 |
| 开销 | 涉及内核调度 | 事件循环内调度 |
| 死锁恢复 | 几乎不可能 | 理论上可以取消 |
| 使用场景 | 真正的并行 | I/O 并发 |

### 8.1.6 何时需要同步原语

并非所有异步代码都需要同步。以下场景不需要：

- 每个协程处理独立的数据
- 只读共享数据
- 使用消息传递（Queue）代替共享状态

以下场景必须使用：

- 多个协程修改同一资源（计数器、缓存、文件）
- 限制并发数量（连接池、速率限制）
- 协调多个协程的执行顺序（等待信号、生产者-消费者）

---

## 8.2 asyncio.Lock — 互斥锁

<div data-component="LockDemo"></div>

`asyncio.Lock` 是最基本的同步原语，确保同一时刻只有一个协程可以访问临界区。

### 8.2.1 Lock 的基本语义

```
Lock 有两种状态：
  - unlocked（未锁定）：任何协程可以 acquire
  - locked（已锁定）：其他协程必须等待

acquire(): 如果未锁定 → 立即锁定并返回 True
           如果已锁定 → await 等待直到解锁
release(): 解锁，唤醒一个等待者
```

### 8.2.2 创建与基本使用

```python
import asyncio

lock = asyncio.Lock()

async def critical_section(name):
    print(f"{name} 等待获取锁...")
    await lock.acquire()
    try:
        print(f"{name} 已获取锁")
        await asyncio.sleep(1)  # 模拟耗时操作
        print(f"{name} 释放锁")
    finally:
        lock.release()

async def main():
    await asyncio.gather(
        critical_section("A"),
        critical_section("B"),
        critical_section("C"),
    )

asyncio.run(main())
```

输出示例：

```
A 等待获取锁...
A 已获取锁
B 等待获取锁...
C 等待获取锁...
A 释放锁
B 已获取锁
B 释放锁
C 已获取锁
C 释放锁
```

### 8.2.3 Lock 的内部实现

`asyncio.Lock` 内部维护：

- `_locked: bool` — 当前是否被锁定
- `_waiters: deque[Future]` — 等待获取锁的协程队列

`acquire()` 的简化逻辑：

```python
async def acquire(self):
    if not self._locked:
        self._locked = True
        return True

    fut = self._loop.create_future()
    self._waiters.append(fut)
    try:
        await fut  # 让出控制权，等待被唤醒
        return True
    except:
        fut.cancel()
        raise
```

`release()` 的简化逻辑：

```python
def release(self):
    if not self._locked:
        raise RuntimeError("Lock is not acquired")
    self._locked = False
    # 唤醒队列中第一个等待者
    for fut in self._waiters:
        if not fut.done():
            fut.set_result(True)
            break
```

### 8.2.4 Lock 是公平的

`asyncio.Lock` 使用 FIFO 队列管理等待者。先调用 `acquire()` 的协程先获得锁。
这避免了"饥饿"问题——不会出现某个协程永远抢不到锁的情况。

### 8.2.5 Lock 不可重入

与 `threading.RLock` 不同，`asyncio.Lock` 不是可重入的：

```python
lock = asyncio.Lock()

async def nested():
    await lock.acquire()
    try:
        print("外层锁")
        await lock.acquire()  # ← 死锁！自己等自己
        print("永远不会到这里")
    finally:
        lock.release()
```

`asyncio` 没有提供 `RLock`。如果需要重入，应该重构代码，将嵌套的临界区合并，
或者使用标志变量手动实现。

### 8.2.6 Lock 的生命周期

`Lock` 对象的生命周期应该与它保护的资源一致：

```python
# ✅ 全局锁，保护全局资源
global_lock = asyncio.Lock()

# ✅ 实例锁，保护实例资源
class Cache:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._data = {}

# ❌ 在函数内创建锁（每次调用都是新锁，无法保护）
async def bad():
    lock = asyncio.Lock()  # 每次调用创建新锁
    async with lock:
        ...
```
</div>

---

## 8.3 Lock 使用模式

### 8.3.1 async with 模式（推荐）

`asyncio.Lock` 支持异步上下文管理器，这是最推荐的使用方式：

```python
async def safe_increment():
    async with lock:  # 自动 acquire，退出时自动 release
        global counter
        temp = counter
        await asyncio.sleep(0)
        counter = temp + 1
```

`async with lock` 等价于：

```python
await lock.acquire()
try:
    ...
finally:
    lock.release()
```

优势：
- 不可能忘记 `release()`
- 异常安全：即使发生异常也会释放锁
- 代码更简洁

### 8.3.2 保护临界区的最小化

锁的持有时间应该尽可能短：

```python
# ❌ 持有锁时间过长
async def bad_fetch():
    async with lock:
        data = await slow_network_request()  # 网络请求期间也持有锁
        process(data)

# ✅ 只在需要时持有锁
async def good_fetch():
    data = await slow_network_request()  # 不在锁内
    async with lock:
        process(data)  # 只在处理共享数据时持有锁
```

### 8.3.3 LockDemo 组件

下面的 `LockDemo` 演示了 Lock 在多个场景下的使用：

```python
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


class LockDemo:
    """演示 asyncio.Lock 的各种使用模式。"""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._shared_counter = 0
        self._shared_list: list[int] = []
        self._log: list[str] = []

    def _record(self, msg: str):
        self._log.append(msg)

    @property
    def counter(self) -> int:
        return self._shared_counter

    @property
    def log(self) -> list[str]:
        return list(self._log)

    async def unsafe_increment(self, n: int):
        """不使用锁的递增——会产生竞态条件。"""
        for _ in range(n):
            temp = self._shared_counter
            await asyncio.sleep(0)  # 让出点
            self._shared_counter = temp + 1

    async def safe_increment(self, n: int):
        """使用锁的递增——线程安全。"""
        for _ in range(n):
            async with self._lock:
                temp = self._shared_counter
                await asyncio.sleep(0)
                self._shared_counter = temp + 1

    async def safe_append(self, value: int):
        """安全地向共享列表追加元素。"""
        async with self._lock:
            self._shared_list.append(value)
            self._record(f"append({value}), len={len(self._shared_list)}")

    async def transfer(self, from_idx: int, to_idx: int, amount: int,
                       balances: list[int]):
        """模拟转账——需要同时锁定两个账户。"""
        # 按固定顺序获取锁，避免死锁
        first, second = sorted([from_idx, to_idx])
        async with self._locks[first]:
            async with self._locks[second]:
                if balances[from_idx] >= amount:
                    balances[from_idx] -= amount
                    balances[to_idx] += amount
                    self._record(
                        f"transfer {amount}: {from_idx}→{to_idx}"
                    )

    async def rate_limited_operation(self, semaphore: asyncio.Semaphore):
        """配合信号量使用的限速操作。"""
        async with semaphore:
            self._record("operation started")
            await asyncio.sleep(0.1)
            self._record("operation done")

    async def demonstrate_race_condition(self):
        """演示竞态条件的影响。"""
        self._shared_counter = 0
        await asyncio.gather(
            self.unsafe_increment(100),
            self.unsafe_increment(100),
        )
        result = self._shared_counter
        self._record(f"unsafe result: {result} (expected 200)")
        return result

    async def demonstrate_lock_protection(self):
        """演示 Lock 如何保护共享状态。"""
        self._shared_counter = 0
        await asyncio.gather(
            self.safe_increment(50),
            self.safe_increment(50),
        )
        result = self._shared_counter
        self._record(f"safe result: {result} (expected 100)")
        return result


async def demo_lock_basic():
    """基本 Lock 演示。"""
    demo = LockDemo()

    # 竞态条件演示
    result1 = await demo.demonstrate_race_condition()
    print(f"无锁结果: {result1} (可能 ≠ 200)")

    # Lock 保护演示
    result2 = await demo.demonstrate_lock_protection()
    print(f"有锁结果: {result2} (= 100)")

    for entry in demo.log:
        print(f"  {entry}")


if __name__ == "__main__":
    asyncio.run(demo_lock_basic())
```

### 8.3.4 条件性获取锁

有时你需要"尝试获取锁"而不是"阻塞等待"：

```python
async def try_update():
    acquired = lock.locked()  # 检查是否被锁定
    if acquired:
        print("锁已被占用，跳过本次更新")
        return

    async with lock:
        await do_update()
```

注意：`lock.locked()` 只检查当前状态，不保证接下来不被其他协程抢占。
真正的"非阻塞获取"需要用更复杂的模式。

### 8.3.5 带超时的锁获取

```python
async def update_with_timeout():
    try:
        # asyncio.wait_for 可以给任何协程加超时
        await asyncio.wait_for(lock.acquire(), timeout=5.0)
        try:
            await do_update()
        finally:
            lock.release()
    except asyncio.TimeoutError:
        print("获取锁超时，跳过更新")
```

### 8.3.6 使用 Lock 保护文件写入

```python
class AsyncFileWriter:
    def __init__(self, filepath: str):
        self._filepath = filepath
        self._lock = asyncio.Lock()

    async def write(self, data: str):
        async with self._lock:
            async with open(self._filepath, "a") as f:
                await f.write(data + "\n")
```

---

## 8.4 死锁概念

### 8.4.1 什么是死锁

死锁（deadlock）是指两个或多个协程互相等待对方释放锁，导致所有协程都无法继续执行。

经典的死锁场景——两个协程以不同顺序获取两把锁：

```python
lock_a = asyncio.Lock()
lock_b = asyncio.Lock()

async def coroutine_1():
    async with lock_a:          # 获取锁 A
        await asyncio.sleep(0)  # 让出控制权
        async with lock_b:      # 等待锁 B（被 coroutine_2 持有）
            print("coroutine_1 done")

async def coroutine_2():
    async with lock_b:          # 获取锁 B
        await asyncio.sleep(0)  # 让出控制权
        async with lock_a:      # 等待锁 A（被 coroutine_1 持有）
            print("coroutine_2 done")
```

执行流程：

```
时间   coroutine_1              coroutine_2
─────────────────────────────────────────────
t1     acquire lock_a ✓
t2                            acquire lock_b ✓
t3     await sleep → 让出
t4                            await sleep → 让出
t5     尝试 acquire lock_b    （等待中...）
t6                            尝试 acquire lock_a（等待中...）
t7     永远等待               永远等待
```

### 8.4.2 死锁的四个必要条件

根据 Coffman 条件，死锁需要同时满足：

1. **互斥**（Mutual Exclusion）：资源不能同时被多个协程使用
2. **持有并等待**（Hold and Wait）：协程持有资源的同时等待其他资源
3. **不可抢占**（No Preemption）：资源只能由持有者主动释放
4. **循环等待**（Circular Wait）：存在一个协程的循环等待链

打破任何一个条件就可以预防死锁。

### 8.4.3 预防死锁：固定顺序获取锁

最简单的方法——所有协程以相同的顺序获取锁：

```python
lock_a = asyncio.Lock()
lock_b = asyncio.Lock()

async def coroutine_1():
    async with lock_a:          # 先 A 后 B
        async with lock_b:
            print("coroutine_1 done")

async def coroutine_2():
    async with lock_a:          # 也是先 A 后 B
        async with lock_b:
            print("coroutine_2 done")
```

如果锁的获取顺序需要动态决定，可以给锁编号：

```python
async def safe_acquire(*locks):
    """按固定顺序获取多把锁。"""
    sorted_locks = sorted(locks, key=id)
    for lock in sorted_locks:
        await lock.acquire()
    return sorted_locks

async def safe_release(locks):
    """释放所有锁。"""
    for lock in reversed(locks):
        lock.release()
```

### 8.4.4 预防死锁：使用 try_acquire + 超时

```python
async def transfer_with_timeout(lock_a, lock_b, timeout=5.0):
    """带超时的转账，避免无限等待。"""
    try:
        await asyncio.wait_for(lock_a.acquire(), timeout=timeout)
        try:
            await asyncio.wait_for(lock_b.acquire(), timeout=timeout)
            try:
                # 执行转账逻辑
                pass
            finally:
                lock_b.release()
        except asyncio.TimeoutError:
            print("获取第二把锁超时")
        finally:
            if lock_a.locked():
                lock_a.release()
    except asyncio.TimeoutError:
        print("获取第一把锁超时")
```

### 8.4.5 检测死锁

在 asyncio 中，死锁不会像线程那样导致整个程序挂起——你仍然可以观察到某些协程
长时间不返回。检测方法：

```python
async def monitor_deadlock(tasks, timeout=10.0):
    """监控任务是否超时（可能是死锁）。"""
    done, pending = await asyncio.wait(tasks, timeout=timeout)
    if pending:
        print(f"警告：{len(pending)} 个任务可能死锁")
        for task in pending:
            print(f"  - {task.get_name()}")
        # 取消可疑任务
        for task in pending:
            task.cancel()
```

### 8.4.6 asyncio 中死锁的特点

在 asyncio 的单线程模型中，死锁有一些独特特征：

- **不会导致线程阻塞**：事件循环仍然运行，其他不相关的协程不受影响
- **可以通过取消恢复**：使用 `task.cancel()` 可以打破死锁
- **更容易调试**：可以用 `asyncio.all_tasks()` 查看所有挂起的任务
- **不会损坏数据**：与线程不同，死锁时数据仍然是一致的

### 8.4.7 活锁与饥饿

除了死锁，还需要注意：

- **活锁**（livelock）：协程不断重试但始终无法推进——类似两个人在走廊里互相让路
- **饥饿**（starvation）：某个协程永远无法获取资源——FIFO 锁可以避免此问题

```python
# 活锁示例：两个协程不断重试
async def livelock_coroutine(name, my_lock, other_lock):
    while True:
        async with my_lock:
            if other_lock.locked():
                print(f"{name}: 对方持有锁，重试...")
                await asyncio.sleep(0.01)  # 短暂等待后重试
                continue  # 释放 my_lock，重新开始
            async with other_lock:
                print(f"{name}: 完成")
                break
```

---

## 8.5 asyncio.Semaphore — 限制并发

<div data-component="SemaphoreDemo"></div>

信号量（Semaphore）用于限制同时访问某资源的协程数量。它是 Lock 的泛化——
Lock 只允许 1 个协程访问，Semaphore 允许 N 个。

### 8.5.1 Semaphore 的语义

```
Semaphore 内部维护一个计数器（初始值 N）：
  - acquire(): 计数器 > 0 → 减 1，立即返回
               计数器 = 0 → await 等待
  - release(): 计数器加 1，唤醒一个等待者
```

### 8.5.2 创建与基本使用

```python
import asyncio

# 允许最多 3 个协程同时访问
sem = asyncio.Semaphore(3)

async def access_resource(name, duration):
    print(f"{name} 等待访问...")
    async with sem:
        print(f"{name} 正在访问资源")
        await asyncio.sleep(duration)
        print(f"{name} 访问完毕")

async def main():
    tasks = [access_resource(f"Task-{i}", 1) for i in range(10)]
    await asyncio.gather(*tasks)

asyncio.run(main())
```

输出模式（注意每次最多 3 个同时运行）：

```
Task-0 正在访问资源
Task-1 正在访问资源
Task-2 正在访问资源
Task-0 访问完毕
Task-3 正在访问资源
...
```

### 8.5.3 Semaphore 的内部实现

```python
class Semaphore:
    def __init__(self, value=1):
        self._value = value      # 当前可用"名额"
        self._waiters = deque()  # 等待队列

    async def acquire(self):
        self._value -= 1
        if self._value >= 0:
            return True

        # 名额不足，加入等待队列
        fut = self._loop.create_future()
        self._waiters.append(fut)
        try:
            await fut
            return True
        except:
            self._value += 1
            raise

    def release(self):
        self._value += 1
        # 唤醒一个等待者
        for fut in self._waiters:
            if not fut.done():
                fut.set_result(True)
                break
```

### 8.5.4 Semaphore(1) 等价于 Lock

```python
# 这两行效果相同
lock = asyncio.Lock()
sem = asyncio.Semaphore(1)
```

但语义不同：Lock 强调"互斥"，Semaphore 强调"并发度控制"。
代码中应根据意图选择。

### 8.5.5 Semaphore 的公平性

与 Lock 类似，`asyncio.Semaphore` 也是 FIFO 公平的——先等待的协程先获得名额。

</div>
---

## 8.6 Semaphore 示例

### 8.6.1 速率限制器（Rate Limiter）

```python
import asyncio
import time


class AsyncRateLimiter:
    """基于信号量的异步速率限制器。"""

    def __init__(self, max_requests: int, time_window: float):
        self._semaphore = asyncio.Semaphore(max_requests)
        self._time_window = time_window
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        await self._semaphore.acquire()

    def release(self):
        self._semaphore.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *args):
        await asyncio.sleep(self._time_window)
        self.release()


async def rate_limited_api_call(rate_limiter, url):
    """受速率限制的 API 调用。"""
    async with rate_limiter:
        print(f"  请求 {url} @ {time.strftime('%H:%M:%S')}")
        await asyncio.sleep(0.5)  # 模拟网络延迟
        return f"响应: {url}"


async def demo_rate_limiter():
    """演示速率限制器。"""
    # 每 2 秒最多 3 个请求
    limiter = AsyncRateLimiter(max_requests=3, time_window=2.0)

    urls = [f"https://api.example.com/data/{i}" for i in range(10)]
    tasks = [rate_limited_api_call(limiter, url) for url in urls]

    results = await asyncio.gather(*tasks)
    print(f"完成 {len(results)} 个请求")


if __name__ == "__main__":
    asyncio.run(demo_rate_limiter())
```

### 8.6.2 连接池（Connection Pool）

```python
import asyncio
from typing import Optional


class AsyncConnectionPool:
    """基于 Semaphore 的异步连接池。"""

    def __init__(self, max_connections: int = 5):
        self._semaphore = asyncio.Semaphore(max_connections)
        self._pool: list[dict] = []
        self._lock = asyncio.Lock()
        self._counter = 0

    async def _create_connection(self) -> dict:
        """创建新连接（模拟）。"""
        async with self._lock:
            self._counter += 1
            conn_id = self._counter
        await asyncio.sleep(0.1)  # 模拟连接建立
        return {"id": conn_id, "active": True}

    async def acquire(self) -> dict:
        """获取一个连接。"""
        await self._semaphore.acquire()
        async with self._lock:
            if self._pool:
                return self._pool.pop()
        return await self._create_connection()

    def release(self, conn: dict):
        """归还连接。"""
        async with self._lock:
            self._pool.append(conn)
        self._semaphore.release()

    async def __aenter__(self) -> dict:
        self._conn = await self.acquire()
        return self._conn

    async def __aexit__(self, *args):
        self.release(self._conn)


async def db_query(pool: AsyncConnectionPool, query: str):
    """使用连接池执行查询。"""
    async with pool as conn:
        print(f"  连接 {conn['id']} 执行: {query}")
        await asyncio.sleep(0.2)  # 模拟查询
        return f"结果: {query}"


async def demo_connection_pool():
    """演示连接池。"""
    pool = AsyncConnectionPool(max_connections=3)

    queries = [f"SELECT * FROM table_{i}" for i in range(8)]
    tasks = [db_query(pool, q) for q in queries]

    results = await asyncio.gather(*tasks)
    print(f"完成 {len(results)} 个查询")


if __name__ == "__main__":
    asyncio.run(demo_connection_pool())
```

### 8.6.3 并发下载器

```python
import asyncio
import aiofiles


async def download_file(sem: asyncio.Semaphore, url: str, dest: str):
    """受并发限制的文件下载。"""
    async with sem:
        print(f"开始下载: {url}")
        # 模拟下载
        await asyncio.sleep(1)
        # async with aiofiles.open(dest, "wb") as f:
        #     await f.write(content)
        print(f"下载完成: {dest}")
        return dest


async def bulk_download():
    """批量下载，最多同时 5 个。"""
    sem = asyncio.Semaphore(5)
    urls = [f"https://example.com/file_{i}.zip" for i in range(20)]

    tasks = [
        download_file(sem, url, f"/tmp/file_{i}.zip")
        for i, url in enumerate(urls)
    ]

    results = await asyncio.gather(*tasks)
    print(f"共下载 {len(results)} 个文件")
```

### 8.6.4 Semaphore 与 Lock 的对比

```python
# Lock：一次只允许一个
lock = asyncio.Lock()
async with lock:
    # 只有 1 个协程能进入
    pass

# Semaphore(5)：一次允许 5 个
sem = asyncio.Semaphore(5)
async with sem:
    # 最多 5 个协程能同时进入
    pass

# Semaphore(1)：等价于 Lock
sem1 = asyncio.Semaphore(1)
async with sem1:
    # 只有 1 个协程能进入
    pass
```

---

## 8.7 asyncio.BoundedSemaphore

### 8.7.1 BoundedSemaphore 与 Semaphore 的区别

`BoundedSemaphore` 是 `Semaphore` 的安全版本——它限制 `release()` 的次数不能超过
初始值。这可以防止"过度释放"的 bug：

```python
# 普通 Semaphore：release 可以无限调用
sem = asyncio.Semaphore(3)
sem.release()  # 计数器变成 4（可能不是你想要的）
sem.release()  # 计数器变成 5（溢出）

# BoundedSemaphore：release 超过初始值会报错
bsem = asyncio.BoundedSemaphore(3)
bsem.release()  # ValueError: BoundedSemaphore released too many times
```

### 8.7.2 何时使用 BoundedSemaphore

当信号量的初始值有明确的物理含义时（如连接池大小、磁盘空间），使用
`BoundedSemaphore` 可以尽早发现 bug：

```python
class DatabasePool:
    def __init__(self, max_conn: int):
        # 使用 BoundedSemaphore，防止归还超过 max_conn 个连接
        self._sem = asyncio.BoundedSemaphore(max_conn)
        self._connections = []

    async def get_connection(self):
        await self._sem.acquire()
        return self._connections.pop()

    def return_connection(self, conn):
        self._connections.append(conn)
        self._sem.release()  # 如果多归还了，会立即报错
```

### 8.7.3 acquire-release 配对

使用 `BoundedSemaphore` 时，必须确保 `acquire` 和 `release` 严格配对：

```python
# ✅ 正确：async with 自动配对
async with bsem:
    await do_work()

# ✅ 正确：try/finally 手动配对
await bsem.acquire()
try:
    await do_work()
finally:
    bsem.release()

# ❌ 错误：条件分支可能导致不配对
if condition:
    await bsem.acquire()
# ... 可能忘记 release
bsem.release()  # 如果没有 acquire，这里会报错
```

---

## 8.8 asyncio.Event — 协程间信号

<div data-component="EventDemo"></div>

Event 是最简单的信号机制——一个协程发出信号，其他协程等待这个信号。

### 8.8.1 Event 的语义

```
Event 有两种状态：
  - cleared（未设置）：wait() 的协程会阻塞
  - set（已设置）：wait() 的协程立即返回

set():    将事件设为"已设置"，唤醒所有等待者
clear():  将事件设为"未设置"
wait():   如果已设置 → 立即返回；否则等待
is_set(): 查询当前状态
```

### 8.8.2 基本使用

```python
import asyncio

async def waiter(event: asyncio.Event, name: str):
    print(f"{name} 等待事件...")
    await event.wait()
    print(f"{name} 收到事件，继续执行")

async def setter(event: asyncio.Event):
    await asyncio.sleep(2)
    print("设置事件")
    event.set()

async def main():
    event = asyncio.Event()

    await asyncio.gather(
        waiter(event, "W1"),
        waiter(event, "W2"),
        waiter(event, "W3"),
        setter(event),
    )

asyncio.run(main())
```

输出：

```
W1 等待事件...
W2 等待事件...
W3 等待事件...
（2 秒后）
设置事件
W1 收到事件，继续执行
W2 收到事件，继续执行
W3 收到事件，继续执行
```

### 8.8.3 Event 的内部实现

```python
class Event:
    def __init__(self):
        self._value = False
        self._waiters = deque()

    def is_set(self):
        return self._value

    def set(self):
        self._value = True
        for fut in self._waiters:
            if not fut.done():
                fut.set_result(True)
        self._waiters.clear()

    def clear(self):
        self._value = False

    async def wait(self):
        if self._value:
            return True
        fut = self._loop.create_future()
        self._waiters.append(fut)
        try:
            await fut
            return True
        except:
            self._waiters.remove(fut)
            raise
```
</div>

---

## 8.9 Event 使用模式

### 8.9.1 启动信号（Startup Signal）

```python
startup_event = asyncio.Event()

async def worker(name):
    """等待系统启动完成后开始工作。"""
    await startup_event.wait()
    print(f"{name} 开始工作")
    # ...

async def initialize():
    """系统初始化。"""
    print("系统初始化中...")
    await asyncio.sleep(2)
    print("初始化完成")
    startup_event.set()

async def main():
    await asyncio.gather(
        initialize(),
        worker("W1"),
        worker("W2"),
        worker("W3"),
    )
```

### 8.9.2 关闭信号（Shutdown Signal）

```python
shutdown_event = asyncio.Event()

async def long_running_worker(name):
    """持续运行的工作者，收到关闭信号后退出。"""
    while not shutdown_event.is_set():
        print(f"{name} 处理任务...")
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
    print(f"{name} 收到关闭信号，退出")

async def shutdown_after(seconds):
    """定时关闭。"""
    await asyncio.sleep(seconds)
    print("发送关闭信号")
    shutdown_event.set()
```

### 8.9.3 条件等待（Conditional Wait）

```python
data_ready = asyncio.Event()
data = None

async def producer():
    global data
    await asyncio.sleep(1)
    data = {"value": 42}
    data_ready.set()

async def consumer():
    await data_ready.wait()
    print(f"收到数据: {data}")
```

### 8.9.4 一次性屏障（One-shot Barrier）

```python
async def one_shot_barrier(event, n):
    """用 Event 实现一次性屏障。"""
    event._count = getattr(event, '_count', 0) + 1
    if event._count >= n:
        event.set()
    else:
        await event.wait()
```

### 8.9.5 EventDemo 组件

```python
import asyncio
from enum import Enum, auto


class SystemState(Enum):
    INITIALIZING = auto()
    RUNNING = auto()
    SHUTTING_DOWN = auto()
    STOPPED = auto()


class EventDemo:
    """演示 asyncio.Event 的各种使用模式。"""

    def __init__(self):
        self._startup_event = asyncio.Event()
        self._shutdown_event = asyncio.Event()
        self._data_events: dict[str, asyncio.Event] = {}
        self._log: list[str] = []

    def _record(self, msg: str):
        self._log.append(msg)

    @property
    def log(self) -> list[str]:
        return list(self._log)

    async def wait_for_startup(self, name: str):
        """等待系统启动完成。"""
        self._record(f"{name}: 等待启动...")
        await self._startup_event.wait()
        self._record(f"{name}: 系统已启动，开始工作")

    async def signal_startup(self, delay: float = 0):
        """发出启动信号。"""
        if delay > 0:
            await asyncio.sleep(delay)
        self._record("系统启动完成")
        self._startup_event.set()

    async def wait_for_shutdown(self, name: str):
        """等待关闭信号。"""
        self._record(f"{name}: 等待关闭信号...")
        await self._shutdown_event.wait()
        self._record(f"{name}: 收到关闭信号，清理资源")
        await asyncio.sleep(0.1)  # 模拟清理
        self._record(f"{name}: 资源已清理")

    async def signal_shutdown(self, delay: float = 0):
        """发出关闭信号。"""
        if delay > 0:
            await asyncio.sleep(delay)
        self._record("发送关闭信号")
        self._shutdown_event.set()

    async def wait_for_data(self, key: str):
        """等待特定数据准备好。"""
        if key not in self._data_events:
            self._data_events[key] = asyncio.Event()
        self._record(f"等待数据: {key}")
        await self._data_events[key].wait()
        self._record(f"数据已就绪: {key}")

    def signal_data_ready(self, key: str):
        """标记数据已准备好。"""
        if key not in self._data_events:
            self._data_events[key] = asyncio.Event()
        self._record(f"数据准备好: {key}")
        self._data_events[key].set()

    async def run_startup_sequence(self):
        """演示启动序列。"""
        await asyncio.gather(
            self.wait_for_startup("Worker-A"),
            self.wait_for_startup("Worker-B"),
            self.signal_startup(delay=1),
        )

    async def run_shutdown_sequence(self):
        """演示关闭序列。"""
        await asyncio.gather(
            self.wait_for_shutdown("Service-X"),
            self.wait_for_shutdown("Service-Y"),
            self.signal_shutdown(delay=1),
        )

    async def run_data_wait(self):
        """演示数据等待模式。"""
        async def delayed_signal(key, delay):
            await asyncio.sleep(delay)
            self.signal_data_ready(key)

        await asyncio.gather(
            self.wait_for_data("config"),
            self.wait_for_data("database"),
            delayed_signal("config", 0.5),
            delayed_signal("database", 1.0),
        )


async def demo_event():
    """Event 使用演示。"""
    demo = EventDemo()

    print("=== 启动序列 ===")
    await demo.run_startup_sequence()
    for entry in demo.log:
        print(f"  {entry}")

    demo._log.clear()
    print("\n=== 关闭序列 ===")
    await demo.run_shutdown_sequence()
    for entry in demo.log:
        print(f"  {entry}")

    demo._log.clear()
    print("\n=== 数据等待 ===")
    await demo.run_data_wait()
    for entry in demo.log:
        print(f"  {entry}")


if __name__ == "__main__":
    asyncio.run(demo_event())
```

---

## 8.10 asyncio.Condition — 复杂信号

Condition（条件变量）是 Lock + Event 的结合体。它允许协程等待某个条件成立，
并在条件成立时被唤醒。

### 8.10.1 Condition 的语义

```
Condition 基于 Lock，增加以下方法：
  - wait():       释放锁，等待通知，重新获取锁后返回
  - wait_for(predicate):  循环 wait 直到 predicate() 为 True
  - notify(n=1):  唤醒 n 个等待者
  - notify_all(): 唤醒所有等待者
```

### 8.10.2 基本使用

```python
import asyncio

condition = asyncio.Condition()
items = []

async def producer():
    for i in range(5):
        async with condition:
            items.append(i)
            print(f"生产: {i}")
            condition.notify()  # 通知消费者
        await asyncio.sleep(0.5)

async def consumer(name):
    while True:
        async with condition:
            while not items:  # 用 while 而不是 if（防止虚假唤醒）
                await condition.wait()
            item = items.pop(0)
            print(f"{name} 消费: {item}")
        await asyncio.sleep(0.1)

async def main():
    await asyncio.gather(
        producer(),
        consumer("C1"),
        consumer("C2"),
    )

asyncio.run(main())
```

### 8.10.3 wait_for 模式

`wait_for` 是 Condition 最强大的功能——它接受一个谓词函数，只有当谓词返回 True 时
才会唤醒：

```python
buffer = []
MAX_SIZE = 5

async def producer_cv():
    for i in range(20):
        async with condition:
            # 等待缓冲区有空间
            await condition.wait_for(lambda: len(buffer) < MAX_SIZE)
            buffer.append(i)
            print(f"生产: {i}, 缓冲区: {len(buffer)}")
            condition.notify_all()

async def consumer_cv():
    while True:
        async with condition:
            # 等待缓冲区有数据
            await condition.wait_for(lambda: len(buffer) > 0)
            item = buffer.pop(0)
            print(f"消费: {item}, 缓冲区: {len(buffer)}")
            condition.notify_all()
```

### 8.10.4 Condition 与 Event 的对比

| 特性 | Event | Condition |
|------|-------|-----------|
| 等待 | 无条件等待 | 可带谓词等待 |
| 通知 | 唤醒所有等待者 | 可选择唤醒 N 个 |
| 保护 | 无数据保护 | 基于 Lock 保护 |
| 用途 | 简单信号 | 生产者-消费者 |
| 复杂度 | 低 | 中等 |

### 8.10.5 使用 Condition 实现有界缓冲区

```python
class AsyncBoundedBuffer:
    """使用 Condition 实现的有界缓冲区。"""

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._buffer: list = []
        self._condition = asyncio.Condition()

    async def put(self, item):
        async with self._condition:
            await self._condition.wait_for(
                lambda: len(self._buffer) < self._capacity
            )
            self._buffer.append(item)
            self._condition.notify_all()

    async def get(self):
        async with self._condition:
            await self._condition.wait_for(
                lambda: len(self._buffer) > 0
            )
            item = self._buffer.pop(0)
            self._condition.notify_all()
            return item

    @property
    def size(self):
        return len(self._buffer)
```

---

## 8.11 asyncio.Queue — 生产者-消费者

<div data-component="QueueDemo"></div>

Queue 是 asyncio 中最常用的同步原语之一，专门用于生产者-消费者模式。

### 8.11.1 为什么用 Queue

与 Lock + Condition 相比，Queue 的优势：

- **接口简洁**：`put()` 和 `get()` 就够了
- **内置有界支持**：`maxsize` 参数自动控制背压
- **天然线程安全**：不需要额外的锁
- **支持取消**：`task_done()` + `join()` 支持优雅关闭

### 8.11.2 创建 Queue

```python
import asyncio

# 无界队列（可以无限 put）
queue = asyncio.Queue()

# 有界队列（最多容纳 10 个元素）
bounded_queue = asyncio.Queue(maxsize=10)

# 优先级队列
priority_queue = asyncio.PriorityQueue()

# 后进先出队列（栈）
lifo_queue = asyncio.LifoQueue()
```

### 8.11.3 Queue 的内部结构

```python
class Queue:
    def __init__(self, maxsize=0):
        self._maxsize = maxsize
        self._queue = deque()
        self._getters = deque()    # 等待 get 的协程
        self._putters = deque()    # 等待 put 的协程
        self._unfinished_tasks = 0

    @property
    def maxsize(self):
        return self._maxsize

    def qsize(self):
        return len(self._queue)

    def empty(self):
        return not self._queue

    def full(self):
        if self._maxsize <= 0:
            return False
        return self.qsize() >= self._maxsize
```
</div>

---

## 8.12 Queue 方法详解

### 8.12.1 put() 与 put_nowait()

```python
# put() —— 协程版本，队列满时等待
await queue.put(item)

# put_nowait() —— 非阻塞版本，队列满时抛出异常
try:
    queue.put_nowait(item)
except asyncio.QueueFull:
    print("队列已满")

# 带超时的 put
try:
    await asyncio.wait_for(queue.put(item), timeout=5.0)
except asyncio.TimeoutError:
    print("put 超时")
```

### 8.12.2 get() 与 get_nowait()

```python
# get() —— 协程版本，队列空时等待
item = await queue.get()

# get_nowait() —— 非阻塞版本，队列空时抛出异常
try:
    item = queue.get_nowait()
except asyncio.QueueEmpty:
    print("队列为空")
```

### 8.12.3 task_done() 与 join()

这对方法用于追踪队列中"未完成的任务"：

```python
async def worker(queue):
    while True:
        item = await queue.get()
        try:
            await process(item)
        finally:
            queue.task_done()  # 标记任务完成

async def main():
    queue = asyncio.Queue()

    # 启动工作者
    workers = [asyncio.create_task(worker(queue)) for _ in range(3)]

    # 添加任务
    for i in range(10):
        await queue.put(i)

    # 等待所有任务完成
    await queue.join()  # 阻塞直到所有 task_done() 被调用

    # 取消工作者
    for w in workers:
        w.cancel()
```

### 8.12.4 task_done 的计数机制

```
put() → _unfinished_tasks += 1
task_done() → _unfinished_tasks -= 1
join() → 等待 _unfinished_tasks == 0
```

注意：如果 `task_done()` 的调用次数超过 `put()` 的次数，会抛出 `ValueError`。

---

## 8.13 Queue 示例

### 8.13.1 基本生产者-消费者

```python
import asyncio
import random


async def producer(queue: asyncio.Queue, name: str, n: int):
    """生产者：产生 n 个任务。"""
    for i in range(n):
        item = f"{name}-item-{i}"
        await queue.put(item)
        print(f"[{name}] 生产: {item}")
        await asyncio.sleep(random.uniform(0.1, 0.3))
    print(f"[{name}] 生产完毕")

async def consumer(queue: asyncio.Queue, name: str):
    """消费者：持续消费任务。"""
    while True:
        item = await queue.get()
        print(f"[{name}] 消费: {item}")
        await asyncio.sleep(random.uniform(0.1, 0.5))
        queue.task_done()

async def main():
    queue = asyncio.Queue(maxsize=5)

    # 启动 2 个生产者和 3 个消费者
    producers = [
        asyncio.create_task(producer(queue, f"P{i}", 5))
        for i in range(2)
    ]
    consumers = [
        asyncio.create_task(consumer(queue, f"C{i}"))
        for i in range(3)
    ]

    # 等待生产者完成
    await asyncio.gather(*producers)

    # 等待队列清空
    await queue.join()

    # 取消消费者
    for c in consumers:
        c.cancel()

    print("全部完成")

asyncio.run(main())
```

### 8.13.2 工作队列模式（Work Queue）

```python
import asyncio
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class WorkItem:
    """工作项。"""
    id: int
    payload: Any
    callback: Callable | None = None


class AsyncWorkQueue:
    """异步工作队列。"""

    def __init__(self, max_workers: int = 4, maxsize: int = 100):
        self._queue = asyncio.Queue(maxsize=maxsize)
        self._max_workers = max_workers
        self._workers: list[asyncio.Task] = []
        self._results: dict[int, Any] = {}
        self._counter = 0
        self._lock = asyncio.Lock()

    async def submit(self, payload: Any) -> int:
        """提交工作项，返回 ID。"""
        async with self._lock:
            self._counter += 1
            item_id = self._counter

        item = WorkItem(id=item_id, payload=payload)
        await self._queue.put(item)
        return item_id

    async def _worker(self, name: str, handler: Callable):
        """工作协程。"""
        while True:
            item = await self._queue.get()
            try:
                result = await handler(item.payload)
                self._results[item.id] = result
                if item.callback:
                    await item.callback(result)
            except Exception as e:
                self._results[item.id] = e
            finally:
                self._queue.task_done()

    async def start(self, handler: Callable):
        """启动工作队列。"""
        for i in range(self._max_workers):
            task = asyncio.create_task(
                self._worker(f"Worker-{i}", handler)
            )
            self._workers.append(task)

    async def stop(self):
        """停止工作队列。"""
        await self._queue.join()
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)

    def get_result(self, item_id: int) -> Any:
        """获取结果。"""
        return self._results.get(item_id)


async def demo_work_queue():
    """演示工作队列。"""

    async def process(payload):
        """处理函数。"""
        await asyncio.sleep(0.1)
        return f"processed: {payload}"

    queue = AsyncWorkQueue(max_workers=3)
    await queue.start(process)

    # 提交任务
    ids = []
    for i in range(10):
        item_id = await queue.submit(f"data-{i}")
        ids.append(item_id)

    # 等待完成
    await queue.stop()

    # 获取结果
    for item_id in ids:
        print(f"  任务 {item_id}: {queue.get_result(item_id)}")
```

### 8.13.3 Pipeline 模式

```python
import asyncio


async def stage_1(input_queue, output_queue):
    """阶段 1：数据获取。"""
    while True:
        url = await input_queue.get()
        # 模拟获取数据
        data = f"data_from_{url}"
        await output_queue.put(data)
        input_queue.task_done()

async def stage_2(input_queue, output_queue):
    """阶段 2：数据处理。"""
    while True:
        data = await input_queue.get()
        # 模拟处理
        result = f"processed_{data}"
        await output_queue.put(result)
        input_queue.task_done()

async def stage_3(input_queue, results):
    """阶段 3：结果收集。"""
    while True:
        result = await input_queue.get()
        results.append(result)
        input_queue.task_done()


async def run_pipeline():
    """运行三级 Pipeline。"""
    q1 = asyncio.Queue()  # 输入
    q2 = asyncio.Queue()  # stage1 → stage2
    q3 = asyncio.Queue()  # stage2 → stage3
    results = []

    # 启动各级
    workers = []
    for _ in range(2):
        workers.append(asyncio.create_task(stage_1(q1, q2)))
    for _ in range(3):
        workers.append(asyncio.create_task(stage_2(q2, q3)))
    workers.append(asyncio.create_task(stage_3(q3, results)))

    # 输入数据
    for i in range(10):
        await q1.put(f"url_{i}")

    # 等待完成
    await q1.join()
    await q2.join()
    await q3.join()

    # 清理
    for w in workers:
        w.cancel()

    print(f"处理结果: {results}")
```

### 8.13.4 QueueDemo 组件

```python
import asyncio
from collections import defaultdict


class QueueDemo:
    """演示 asyncio.Queue 的各种使用模式。"""

    def __init__(self):
        self._log: list[str] = []
        self._stats: dict[str, int] = defaultdict(int)

    def _record(self, msg: str):
        self._log.append(msg)

    @property
    def log(self) -> list[str]:
        return list(self._log)

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    async def basic_producer_consumer(self):
        """基本生产者-消费者演示。"""
        queue = asyncio.Queue(maxsize=3)
        results = []

        async def producer():
            for i in range(6):
                await queue.put(i)
                self._record(f"生产: {i} (size={queue.qsize()})")
            self._record("生产者完成")

        async def consumer(name):
            for _ in range(3):
                item = await queue.get()
                results.append(item)
                self._record(f"{name} 消费: {item}")
                queue.task_done()

        await asyncio.gather(
            producer(),
            consumer("C1"),
            consumer("C2"),
        )
        return results

    async def priority_queue_demo(self):
        """优先级队列演示。"""
        pq = asyncio.PriorityQueue()

        # 添加带优先级的项目
        items = [(3, "低优先级"), (1, "高优先级"), (2, "中优先级")]
        for priority, name in items:
            await pq.put((priority, name))

        results = []
        while not pq.empty():
            priority, name = await pq.get()
            results.append((priority, name))
            self._record(f"取出: {name} (优先级={priority})")

        return results

    async def lifo_queue_demo(self):
        """后进先出队列演示。"""
        lifo = asyncio.LifoQueue()

        for i in range(5):
            await lifo.put(i)
            self._record(f"压入: {i}")

        results = []
        while not lifo.empty():
            item = await lifo.get()
            results.append(item)
            self._record(f"弹出: {item}")

        return results

    async def cancellation_safe_queue(self):
        """安全取消的队列消费。"""
        queue = asyncio.Queue()
        processed = []

        async def safe_consumer(name):
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=1.0)
                    processed.append(item)
                    self._record(f"{name} 处理: {item}")
                    queue.task_done()
                except asyncio.TimeoutError:
                    self._record(f"{name} 超时，退出")
                    break
                except asyncio.CancelledError:
                    self._record(f"{name} 被取消")
                    raise

        # 添加少量数据
        for i in range(3):
            await queue.put(i)

        # 启动消费者，等待完成后取消
        tasks = [
            asyncio.create_task(safe_consumer(f"W{i}"))
            for i in range(2)
        ]

        await asyncio.sleep(2)  # 等待消费者超时退出
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        return processed


async def demo_queue():
    """Queue 使用演示。"""
    demo = QueueDemo()

    print("=== 基本生产者-消费者 ===")
    results = await demo.basic_producer_consumer()
    print(f"  结果: {results}")
    for entry in demo.log[-10:]:
        print(f"    {entry}")

    demo._log.clear()
    print("\n=== 优先级队列 ===")
    results = await demo.priority_queue_demo()
    print(f"  结果: {results}")

    demo._log.clear()
    print("\n=== LIFO 队列 ===")
    results = await demo.lifo_queue_demo()
    print(f"  结果: {results}")


if __name__ == "__main__":
    asyncio.run(demo_queue())
```

---

## 8.14 Barrier — 同步 N 个协程

`asyncio.Barrier`（Python 3.11+）用于让 N 个协程在某个点同步——所有协程都到达屏障后
才能继续执行。

### 8.14.1 Barrier 的语义

```
Barrier(parties=N)：
  - parties：需要同步的协程数量
  - wait(): 到达屏障点并等待
            当 N 个协程都调用 wait() 后，所有协程同时被释放
  - reset(): 重置屏障
  - abort(): 中止屏障（等待中的协程会收到 BrokenBarrierError）
```

### 8.14.2 基本使用

```python
import asyncio

async def phase(barrier, name, delay):
    """每个协程执行两个阶段，中间同步。"""
    print(f"{name}: 阶段 1 开始")
    await asyncio.sleep(delay)
    print(f"{name}: 阶段 1 完成，等待其他协程...")

    await barrier.wait()  # 屏障点

    print(f"{name}: 阶段 2 开始")
    await asyncio.sleep(delay * 0.5)
    print(f"{name}: 阶段 2 完成")

async def main():
    barrier = asyncio.Barrier(3)

    await asyncio.gather(
        phase(barrier, "A", 1.0),
        phase(barrier, "B", 1.5),
        phase(barrier, "C", 0.5),
    )

asyncio.run(main())
```

输出：

```
A: 阶段 1 开始
B: 阶段 1 开始
C: 阶段 1 开始
C: 阶段 1 完成，等待其他协程...
A: 阶段 1 完成，等待其他协程...
B: 阶段 1 完成，等待其他协程...
（所有协程同时释放）
A: 阶段 2 开始
B: 阶段 2 开始
C: 阶段 2 开始
C: 阶段 2 完成
A: 阶段 2 完成
B: 阶段 2 完成
```

### 8.14.3 带动作的 Barrier

```python
async def main_with_action():
    """在所有协程到达后执行一个回调。"""

    def on_all_arrived():
        print(">>> 所有协程已到达，执行同步动作")

    barrier = asyncio.Barrier(3, action=on_all_arrived)
    # ...
```

### 8.14.4 Barrier 的应用场景

- **分布式计算的同步点**：所有 worker 完成一轮计算后同步
- **测试中的协程同步**：确保所有并发操作同时开始
- **多阶段处理**：每个阶段完成后同步，再进入下一阶段
- **资源预加载**：等待所有资源加载完成后一起开始服务

### 8.14.5 Barrier 与 Event 的对比

```python
# 用 Event 模拟 Barrier（繁琐）
ready_events = {f"W{i}": asyncio.Event() for i in range(3)}
# 每个协程设置自己的 Event，然后等待所有其他协程的 Event
# ... 代码复杂

# 用 Barrier（简洁）
barrier = asyncio.Barrier(3)
await barrier.wait()  # 一行搞定
```

---

## 8.15 选择正确的同步原语

<div data-component="SyncPrimitivesComparison"></div>

### 8.15.1 决策流程图

```
需要同步协程？
│
├─ 需要保护共享数据？
│  ├─ 只允许一个访问 → Lock
│  └─ 允许 N 个并发访问 → Semaphore
│
├─ 需要协调执行顺序？
│  ├─ 简单信号（开始/停止）→ Event
│  ├─ 复杂条件等待 → Condition
│  └─ N 个协程同步 → Barrier
│
├─ 需要在协程间传递数据？
│  ├─ 生产者-消费者 → Queue
│  ├─ 带优先级 → PriorityQueue
│  └─ 后进先出 → LifoQueue
│
└─ 需要限流？
   └─ Semaphore
```

### 8.15.2 SyncPrimitivesComparison 组件

```python
import asyncio
import time
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """基准测试结果。"""
    primitive: str
    operations: int
    elapsed: float
    ops_per_sec: float


class SyncPrimitivesComparison:
    """同步原语性能与特性对比。"""

    def __init__(self):
        self._results: list[BenchmarkResult] = []

    @property
    def results(self) -> list[BenchmarkResult]:
        return list(self._results)

    async def benchmark_lock(self, n_ops: int = 1000):
        """Lock 基准测试。"""
        lock = asyncio.Lock()
        counter = 0

        async def increment():
            nonlocal counter
            async with lock:
                counter += 1

        start = time.perf_counter()
        await asyncio.gather(*[increment() for _ in range(n_ops)])
        elapsed = time.perf_counter() - start

        self._results.append(BenchmarkResult(
            primitive="Lock",
            operations=n_ops,
            elapsed=elapsed,
            ops_per_sec=n_ops / elapsed,
        ))

    async def benchmark_semaphore(self, n_ops: int = 1000, value: int = 10):
        """Semaphore 基准测试。"""
        sem = asyncio.Semaphore(value)
        counter = 0
        lock = asyncio.Lock()

        async def increment():
            nonlocal counter
            async with sem:
                async with lock:
                    counter += 1

        start = time.perf_counter()
        await asyncio.gather(*[increment() for _ in range(n_ops)])
        elapsed = time.perf_counter() - start

        self._results.append(BenchmarkResult(
            primitive=f"Semaphore({value})",
            operations=n_ops,
            elapsed=elapsed,
            ops_per_sec=n_ops / elapsed,
        ))

    async def benchmark_event(self, n_ops: int = 1000):
        """Event 基准测试。"""
        counter = 0

        async def signal_wait(event, result_event):
            nonlocal counter
            await event.wait()
            counter += 1
            result_event.set()

        start = time.perf_counter()
        for _ in range(n_ops):
            event = asyncio.Event()
            result = asyncio.Event()
            task = asyncio.create_task(signal_wait(event, result))
            event.set()
            await result.wait()
            await task
        elapsed = time.perf_counter() - start

        self._results.append(BenchmarkResult(
            primitive="Event",
            operations=n_ops,
            elapsed=elapsed,
            ops_per_sec=n_ops / elapsed,
        ))

    async def benchmark_queue(self, n_ops: int = 1000):
        """Queue 基准测试。"""
        queue = asyncio.Queue()

        async def producer():
            for i in range(n_ops):
                await queue.put(i)

        async def consumer():
            for _ in range(n_ops):
                await queue.get()
                queue.task_done()

        start = time.perf_counter()
        p = asyncio.create_task(producer())
        c = asyncio.create_task(consumer())
        await asyncio.gather(p, c)
        elapsed = time.perf_counter() - start

        self._results.append(BenchmarkResult(
            primitive="Queue",
            operations=n_ops,
            elapsed=elapsed,
            ops_per_sec=n_ops / elapsed,
        ))

    async def run_all_benchmarks(self, n_ops: int = 1000):
        """运行所有基准测试。"""
        self._results.clear()
        await self.benchmark_lock(n_ops)
        await self.benchmark_semaphore(n_ops)
        await self.benchmark_event(n_ops)
        await self.benchmark_queue(n_ops)

    def print_results(self):
        """打印对比结果。"""
        print(f"{'原语':<20} {'操作数':<10} {'耗时(s)':<12} {'ops/s':<12}")
        print("-" * 54)
        for r in self._results:
            print(f"{r.primitive:<20} {r.operations:<10} "
                  f"{r.elapsed:<12.4f} {r.ops_per_sec:<12.0f}")

    @staticmethod
    def feature_comparison() -> str:
        """特性对比表。"""
        table = """
| 原语          | 用途             | 等待机制     | 通知方式       | Python 版本 |
|--------------|-----------------|-------------|---------------|-------------|
| Lock         | 互斥访问         | acquire()   | release()     | 3.4+        |
| Semaphore    | 并发度控制        | acquire()   | release()     | 3.4+        |
| BoundedSemaphore | 安全并发控制  | acquire()   | release()     | 3.4+        |
| Event        | 简单信号          | wait()      | set()/clear() | 3.4+        |
| Condition    | 复杂条件等待      | wait/wait_for | notify/all  | 3.4+        |
| Queue        | 生产者-消费者     | get()       | put()         | 3.4+        |
| Barrier      | N 协程同步        | wait()      | 自动          | 3.11+       |
"""
        return table


async def demo_comparison():
    """演示同步原语对比。"""
    comparison = SyncPrimitivesComparison()

    print("=== 特性对比 ===")
    print(comparison.feature_comparison())

    print("\n=== 性能基准 ===")
    await comparison.run_all_benchmarks(500)
    comparison.print_results()


if __name__ == "__main__":
    asyncio.run(demo_comparison())
```

### 8.15.3 选择原则

1. **优先使用 Queue**：大多数协程间通信都可以用 Queue 解决，它最不容易出错
2. **Lock 保护共享状态**：当必须共享可变状态时，用 Lock
3. **Semaphore 限制并发**：连接池、速率限制等场景
4. **Event 用于简单信号**：启动/关闭、数据就绪等
5. **Condition 用于复杂条件**：有界缓冲区、条件等待
6. **Barrier 用于阶段同步**：需要 N 个协程同时到达某点

### 8.15.4 组合使用

实际项目中，常常需要组合多种同步原语：

```python
class AsyncDatabase:
    """组合使用多种同步原语。"""

    def __init__(self, max_connections: int = 10):
        # 用 Semaphore 限制连接数
        self._conn_semaphore = asyncio.BoundedSemaphore(max_connections)
        # 用 Lock 保护缓存
        self._cache_lock = asyncio.Lock()
        # 用 Event 表示就绪状态
        self._ready_event = asyncio.Event()
        # 用 Queue 传递查询结果
        self._result_queue = asyncio.Queue()
        self._cache: dict = {}

    async def initialize(self):
        """初始化数据库连接。"""
        # ... 初始化逻辑 ...
        self._ready_event.set()

    async def query(self, sql: str) -> Any:
        """执行查询。"""
        await self._ready_event.wait()  # 等待初始化完成

        # 检查缓存
        async with self._cache_lock:
            if sql in self._cache:
                return self._cache[sql]

        # 获取连接
        async with self._conn_semaphore:
            result = await self._execute_query(sql)

        # 更新缓存
        async with self._cache_lock:
            self._cache[sql] = result

        return result

    async def _execute_query(self, sql: str) -> Any:
        # ... 执行查询 ...
        pass
```
</div>

---

## 8.16 常见误区

### 8.16.1 误区一：异步代码不需要同步

```python
# ❌ 错误：认为协程不会并发
counter = 0
async def increment():
    global counter
    for _ in range(1000):
        counter += 1
        await asyncio.sleep(0)  # 这里会切换！

# ✅ 正确：使用 Lock
lock = asyncio.Lock()
async def safe_increment():
    global counter
    for _ in range(1000):
        async with lock:
            counter += 1
```

### 8.16.2 误区二：在协程中使用 threading.Lock

```python
# ❌ 错误：会阻塞事件循环
import threading
thread_lock = threading.Lock()
async def bad():
    with thread_lock:  # 这会阻塞整个事件循环！
        await do_something()

# ✅ 正确：使用 asyncio.Lock
async_lock = asyncio.Lock()
async def good():
    async with async_lock:
        await do_something()
```

### 8.16.3 误区三：忘记释放锁

```python
# ❌ 错误：异常时不释放锁
async def bad():
    await lock.acquire()
    await risky_operation()  # 如果异常，锁永远不会释放
    lock.release()

# ✅ 正确：使用 async with 或 try/finally
async def good():
    async with lock:
        await risky_operation()

# ✅ 或者
async def also_good():
    await lock.acquire()
    try:
        await risky_operation()
    finally:
        lock.release()
```

### 8.16.4 误区四：持有锁时执行长时间操作

```python
# ❌ 错误：持有锁时做网络请求
async def bad():
    async with lock:
        data = await http_get("https://api.example.com")  # 长时间阻塞
        process(data)

# ✅ 正确：先获取数据，再用锁处理
async def good():
    data = await http_get("https://api.example.com")  # 不在锁内
    async with lock:
        process(data)  # 只在处理时持有锁
```

### 8.16.5 误区五：用 if 而不是 while 检查条件

```python
# ❌ 错误：虚假唤醒问题
async def bad_consumer(condition):
    async with condition:
        if not items:  # 可能虚假唤醒后 items 仍为空
            await condition.wait()
        item = items.pop(0)  # 可能 IndexError

# ✅ 正确：用 while 循环
async def good_consumer(condition):
    async with condition:
        while not items:  # 虚假唤醒后会重新检查
            await condition.wait()
        item = items.pop(0)
```

### 8.16.6 误区六：不检查 is_set 就 wait

```python
# ❌ 可能永远等待
async def bad():
    await event.wait()  # 如果 event 已经 set 了，这会立即返回
    # 但如果 event 在 wait 之前就被 clear 了...

# ✅ 好的做法
async def good():
    while not event.is_set():
        await event.wait()
    # 确保事件确实被设置了
```

### 8.16.7 误区七：Queue 的 task_done 不配对

```python
# ❌ 错误：不调用 task_done
async def bad_consumer(queue):
    while True:
        item = await queue.get()
        await process(item)
        # 忘记调用 task_done()，join() 会永远等待

# ✅ 正确
async def good_consumer(queue):
    while True:
        item = await queue.get()
        try:
            await process(item)
        finally:
            queue.task_done()  # 确保调用
```

### 8.16.8 误区八：创建过多的锁

```python
# ❌ 错误：每行代码都加锁
async def bad():
    async with lock:
        a = get_a()
    async with lock:
        b = get_b()
    async with lock:
        c = a + b
    # 不需要这么多锁，只有共享状态需要保护

# ✅ 正确：只保护必要的临界区
async def good():
    a = get_a()
    b = get_b()
    async with lock:
        shared_data.append(a + b)
```

### 8.16.9 误区九：使用 Semaphore 作为 Lock

```python
# ❌ 语义不清
sem = asyncio.Semaphore(1)
async with sem:
    # 这等价于 Lock，但意图不明确
    pass

# ✅ 明确意图
lock = asyncio.Lock()
async with lock:
    # 明确表示互斥访问
    pass
```

### 8.16.10 误区十：在 finally 之外 release

```python
# ❌ 错误
async def bad():
    await lock.acquire()
    result = await do_work()
    lock.release()
    return result  # 如果 do_work 抛异常，锁不会释放

# ✅ 正确
async def good():
    await lock.acquire()
    try:
        result = await do_work()
        return result
    finally:
        lock.release()

# ✅ 更好
async def better():
    async with lock:
        return await do_work()
```

### 8.16.11 调试技巧

```python
import asyncio
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DebugLock(asyncio.Lock):
    """带调试日志的 Lock。"""

    async def acquire(self):
        logger.debug(f"尝试获取锁 {id(self)}")
        result = await super().acquire()
        logger.debug(f"获取锁 {id(self)} 成功")
        return result

    def release(self):
        logger.debug(f"释放锁 {id(self)}")
        super().release()


class DebugSemaphore(asyncio.Semaphore):
    """带调试日志的 Semaphore。"""

    def __init__(self, value=1):
        super().__init__(value)
        self._initial_value = value

    async def acquire(self):
        logger.debug(
            f"信号量 {id(self)}: "
            f"{self._value}/{self._initial_value} 可用"
        )
        result = await super().acquire()
        logger.debug(f"信号量 {id(self)}: 获取成功")
        return result

    def release(self):
        super().release()
        logger.debug(
            f"信号量 {id(self)}: "
            f"{self._value}/{self._initial_value} 可用"
        )
```

### 8.16.12 性能注意事项

```python
# Lock 的开销很小（约 0.001ms / acquire-release）
# 不要为了避免 Lock 而写出不安全的代码

# ❌ 过早优化
async def premature_optimization():
    if not lock.locked():  # 先检查再获取——竞态条件！
        async with lock:
            do_work()

# ✅ 正确做法
async def correct():
    async with lock:
        do_work()

# Semaphore 的开销也很小
# 限制并发数通常比不限制要好（避免资源耗尽）
```

---

## 本章小结

本章系统介绍了 `asyncio` 提供的同步原语，它们是编写正确异步程序的基石：

1. **asyncio.Lock**：最基本的互斥锁，保护共享状态不被并发修改。使用 `async with lock`
   模式可以避免忘记释放锁的问题。

2. **asyncio.Semaphore**：限制并发访问的数量。常用于连接池、速率限制等场景。
   `BoundedSemaphore` 提供额外的安全检查。

3. **asyncio.Event**：最简单的信号机制，一个协程 `set()`，其他协程 `wait()`。
   适用于启动/关闭信号、数据就绪通知等。

4. **asyncio.Condition**：Lock + Event 的结合体，支持谓词等待和选择性通知。
   适用于生产者-消费者、有界缓冲区等复杂场景。

5. **asyncio.Queue**：生产者-消费者模式的最佳选择。提供 `put()/get()/task_done()/join()`
   完整的操作集，支持有界队列、优先级队列、LIFO 队列。

6. **asyncio.Barrier**（3.11+）：让 N 个协程在某点同步。适用于多阶段计算、测试同步等。

**选择原则**：
- 数据传递用 Queue
- 共享状态保护用 Lock
- 并发度控制用 Semaphore
- 简单信号用 Event
- 复杂条件等待用 Condition
- 多协程同步用 Barrier

**核心安全规则**：
- 始终使用 `async with` 获取锁
- 最小化临界区范围
- 固定顺序获取多把锁以避免死锁
- 在 `wait()` 时使用 `while` 循环而非 `if` 判断

---

## 思考题

1. **竞态条件识别**：分析以下代码是否存在竞态条件。如果存在，如何修复？

   ```python
   cache = {}

   async def get_or_create(key, factory):
       if key not in cache:
           value = await factory()
           cache[key] = value
       return cache[key]
   ```

2. **死锁预防**：设计一个函数 `transfer_funds(lock_a, lock_b, amount)`，
   要求能安全地在两个账户间转账，不会发生死锁。

3. **Semaphore 实现**：尝试用 `asyncio.Lock` 和计数器手动实现一个 `Semaphore`，
   确保其行为与标准库一致。

4. **Queue 背压**：生产者速度远快于消费者时，如何使用有界队列实现背压机制？
   编写一个生产者-消费者系统，要求生产者在队列满时自动减速。

5. **Condition 应用**：使用 `asyncio.Condition` 实现一个异步的"哲学家就餐"问题
   解决方案（5 个哲学家，5 根叉子，每人需要两根叉子才能就餐）。

6. **Event 替代方案**：以下代码使用 Event 实现了一个"等待数据就绪"的模式，
   能否用 Condition 或 Queue 重写？对比三种方案的优劣。

   ```python
   data_event = asyncio.Event()
   shared_data = None

   async def producer():
       global shared_data
       shared_data = await fetch_data()
       data_event.set()

   async def consumer():
       await data_event.wait()
       process(shared_data)
   ```

7. **性能分析**：使用 `SyncPrimitivesComparison` 组件，比较不同同步原语在
   1000、10000、100000 次操作下的性能差异。什么因素会影响性能？

8. **真实场景设计**：设计一个异步的日志收集系统，要求：
   - 多个日志源并发产生日志
   - 日志通过队列传递给消费者
   - 消费者数量可配置
   - 系统关闭时确保所有日志都被处理
   - 限制同时写入文件的协程数量

9. **Barrier 应用**：编写一个并行排序算法，使用 Barrier 确保每一轮比较-交换操作
   完成后再进入下一轮。

10. **调试练习**：以下代码偶尔会输出错误结果（总和不等于预期值）。找出 bug 并修复：

    ```python
    import asyncio

    total = 0

    async def add_values(values):
        global total
        for v in values:
            temp = total
            await asyncio.sleep(0)
            total = temp + v

    async def main():
        global total
        total = 0
        await asyncio.gather(
            add_values([1, 2, 3, 4, 5]),
            add_values([10, 20, 30, 40, 50]),
        )
        print(f"Total: {total}")  # 期望 165，有时不是
    ```

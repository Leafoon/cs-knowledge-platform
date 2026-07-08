---
title: "超时控制、任务取消与异常处理"
description: "全面掌握 asyncio 中的超时机制、任务取消传播、异常处理模式，以及 shield()、gather() 等关键 API 的正确用法"
updated: "2026-07-07"
---

# 超时控制、任务取消与异常处理

> **学习目标**：
> - 理解为什么异步程序需要超时控制，以及"永远挂起"的风险
> - 掌握 `asyncio.wait_for()` 的用法、边界行为和常见陷阱
> - 深入理解 `CancelledError` 的传播机制和嵌套取消的处理
> - 学会使用 `asyncio.shield()` 保护关键操作不被取消
> - 掌握协程中的 `try/except/finally` 异常处理模式
> - 理解 `gather()` 的 `return_exceptions` 参数及其影响
> - 能够处理"Task exception was never retrieved"警告
> - 掌握超时+降级、优雅关闭等实际工程模式

在真实的异步应用中，网络请求可能永远没有响应，数据库查询可能超时，外部服务可能宕机。如果没有超时控制和异常处理机制，一个卡住的协程可能拖垮整个应用。这一章将系统讲解如何让异步程序在面对各种异常情况时依然健壮可靠。

---

## 7.1 为什么需要超时

<div data-component="TimeoutDemo"></div>

### 永远挂起的噩梦

在同步编程中，一个阻塞的调用最多卡住当前线程。但在异步编程中，一个永不返回的协程会占用事件循环的资源，甚至导致整个程序无法正常退出。

```python
import asyncio

async def fetch_from_slow_server():
    """模拟一个永远不响应的服务器"""
    await asyncio.sleep(float('inf'))  # 永远等待...
    return "data"

async def main():
    # 如果没有超时，这个调用永远不会返回
    result = await fetch_from_slow_server()  # 程序卡在这里
    print(result)  # 永远不会执行到这里
```

### 为什么会出现"永远挂起"

```
┌─────────────────────────────────────────────────────────────────┐
│                    导致协程永远不返回的原因                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 网络问题                                                    │
│     ├── 服务器不响应（宕机、过载）                                │
│     ├── 网络中断但 TCP 连接未断开                                │
│     └── DNS 解析卡住                                            │
│                                                                 │
│  2. 资源竞争                                                    │
│     ├── 死锁：多个协程互相等待对方释放资源                        │
│     ├── 信号丢失：等待的事件永远不会被设置                        │
│     └── 队列满/空：生产者或消费者停止工作                         │
│                                                                 │
│  3. 逻辑错误                                                    │
│     ├── 循环条件永远为真                                         │
│     ├── await 的 Future 永远不会被 resolve                       │
│     └── 忘记调用 event.set() 或 future.set_result()              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 超时的作用

超时机制本质上是一种**时间维度的资源保护**。它确保即使某个操作出了问题，程序也不会无限期地等待下去。

```python
import asyncio

async def fetch_data():
    """有超时保护的数据获取"""
    try:
        # 最多等 5 秒，超时则抛出 TimeoutError
        data = await asyncio.wait_for(
            fetch_from_slow_server(),
            timeout=5.0
        )
        return data
    except asyncio.TimeoutError:
        print("请求超时，使用缓存数据")
        return get_cached_data()
```

### 超时的层次

在实际应用中，超时通常存在于多个层次：

```python
# 层次 1：单个操作超时
async def fetch_one(url):
    async with aiohttp.ClientSession() as session:
        # 连接超时 + 读取超时
        async with session.get(url, timeout=aiohttp.ClientTimeout(
            connect=5,    # 连接超时
            sock_read=10  # 读取超时
        )) as resp:
            return await resp.json()

# 层次 2：任务级超时
async def fetch_with_task_timeout(url):
    try:
        return await asyncio.wait_for(fetch_one(url), timeout=30)
    except asyncio.TimeoutError:
        return None

# 层次 3：整体流程超时
async def process_all():
    try:
        await asyncio.wait_for(
            asyncio.gather(
                fetch_with_task_timeout("http://api1.example.com"),
                fetch_with_task_timeout("http://api2.example.com"),
            ),
            timeout=60  # 整体最多 60 秒
        )
    except asyncio.TimeoutError:
        print("整体流程超时")
```
</div>

---

## 7.2 asyncio.wait_for()

### 基本用法

`asyncio.wait_for()` 是最常用的超时控制工具，它接受一个协程和一个超时时间，返回一个带超时的协程。

```python
import asyncio

async def slow_operation():
    await asyncio.sleep(10)
    return "done"

async def main():
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=3.0)
        print(result)
    except asyncio.TimeoutError:
        print("操作超时！")

asyncio.run(main())
# 输出：操作超时！（约 3 秒后）
```

### wait_for() 的工作原理

```
┌─────────────────────────────────────────────────────────────────┐
│                asyncio.wait_for() 内部机制                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  wait_for(coro, timeout=5.0)                                    │
│      │                                                          │
│      ├── 1. 将 coro 包装成 Task                                 │
│      │                                                          │
│      ├── 2. 设置一个延迟 5 秒的 Timer Handle                     │
│      │                                                          │
│      ├── 3. 等待 Task 完成 或 Timer 触发                         │
│      │      │                                                   │
│      │      ├── 如果 Task 先完成 → 返回结果                      │
│      │      │                                                   │
│      │      └── 如果 Timer 先触发 → 取消 Task → 抛出 TimeoutError │
│      │                                                          │
│      └── 4. 清理资源，返回结果或异常                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 超时时间为 None

当 `timeout` 参数为 `None` 时，`wait_for()` 不会设置超时，等同于直接 `await`：

```python
async def main():
    # 以下两种写法等效
    result = await asyncio.wait_for(slow_operation(), timeout=None)
    result = await slow_operation()
```

### 超时时间为 0 或负数

```python
async def main():
    # timeout=0 的特殊行为：立即检查是否完成
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=0)
    except asyncio.TimeoutError:
        print("立即超时")  # 会执行这里

    # timeout 为负数也会立即超时
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=-1)
    except asyncio.TimeoutError:
        print("负数超时也是立即超时")
```

### wait_for() 与 await 的区别

```python
import asyncio

async def demo():
    coro = slow_operation()

    # 直接 await：没有超时保护
    # result = await coro  # 可能永远等待

    # wait_for：有超时保护
    result = await asyncio.wait_for(coro, timeout=5.0)
    return result
```

### Python 3.11+ 的新行为

在 Python 3.11 之前，`wait_for()` 在超时时会先取消内部任务，然后立即抛出 `TimeoutError`，不等待取消完成。这可能导致资源清理不完全。

Python 3.11 修复了这个问题，`wait_for()` 现在会等待内部任务完成取消过程后再抛出异常：

```python
# Python 3.11+ 的行为
async def main():
    try:
        await asyncio.wait_for(slow_operation(), timeout=3.0)
    except asyncio.TimeoutError:
        # 在 3.11+ 中，到这里时 slow_operation 的取消已经完成
        print("超时，且资源已清理")
```

---

## 7.3 TimeoutError 异常

### 捕获 TimeoutError

`asyncio.TimeoutError` 是 `TimeoutError` 的子类，继承自 `OSError`：

```python
import asyncio

async def main():
    try:
        await asyncio.wait_for(slow_operation(), timeout=3.0)
    except asyncio.TimeoutError:
        print("asyncio 超时")  # 精确匹配
    except TimeoutError:
        print("通用超时")  # 也能捕获（Python 3.11+ 中 asyncio.TimeoutError 就是 TimeoutError）
    except OSError:
        print("OS 错误")  # 也能捕获，但通常不建议
```

### 超时后的降级策略

```python
import asyncio

async def fetch_from_primary():
    """从主服务器获取数据"""
    await asyncio.sleep(10)  # 模拟慢请求
    return {"source": "primary", "data": [1, 2, 3]}

async def fetch_from_backup():
    """从备份服务器获取数据"""
    await asyncio.sleep(1)
    return {"source": "backup", "data": [1, 2, 3]}

async def get_cached_data():
    """从缓存获取数据"""
    return {"source": "cache", "data": [1, 2, 3]}

async def fetch_with_fallback():
    """多级降级策略"""
    # 第一选择：主服务器（5 秒超时）
    try:
        return await asyncio.wait_for(fetch_from_primary(), timeout=5.0)
    except asyncio.TimeoutError:
        print("主服务器超时，尝试备份服务器...")

    # 第二选择：备份服务器（3 秒超时）
    try:
        return await asyncio.wait_for(fetch_from_backup(), timeout=3.0)
    except asyncio.TimeoutError:
        print("备份服务器也超时，使用缓存...")

    # 最后选择：缓存
    return get_cached_data()

async def main():
    result = await fetch_with_fallback()
    print(f"获取到数据: {result}")

asyncio.run(main())
```

### 嵌套超时

```python
import asyncio

async def inner_operation():
    await asyncio.sleep(2)
    return "inner"

async def outer_operation():
    # 内层超时：3 秒
    inner = await asyncio.wait_for(inner_operation(), timeout=3.0)
    await asyncio.sleep(5)  # 外层自己的操作
    return f"outer({inner})"

async def main():
    try:
        # 外层超时：2 秒（比内层短）
        # 外层会在 2 秒后超时，此时内层还没超时
        result = await asyncio.wait_for(outer_operation(), timeout=2.0)
        print(result)
    except asyncio.TimeoutError:
        print("外层超时")

asyncio.run(main())
# 输出：外层超时（2 秒后）
```

### 超时异常的上下文信息

```python
import asyncio

async def main():
    try:
        await asyncio.wait_for(slow_operation(), timeout=3.0)
    except asyncio.TimeoutError as e:
        print(f"异常类型: {type(e).__name__}")
        print(f"异常消息: {e}")
        print(f"异常链: {e.__cause__}")  # 通常为 None
        print(f"异常上下文: {e.__context__}")  # 通常为 None

asyncio.run(main())
```

---

## 7.4 任务取消基础

<div data-component="TaskCancellationFlow"></div>

### 为什么要取消任务

在很多场景下，我们需要主动取消正在运行的任务：

```
┌─────────────────────────────────────────────────────────────────┐
│                    需要取消任务的常见场景                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 用户主动取消                                                 │
│     ├── 用户关闭了下载页面                                       │
│     ├── 用户取消了正在进行的搜索                                 │
│     └── 用户关闭了应用程序                                       │
│                                                                 │
│  2. 超时触发                                                    │
│     ├── wait_for() 超时后自动取消内部任务                        │
│     └── 自定义超时逻辑                                           │
│                                                                 │
│  3. 依赖条件变化                                                 │
│     ├── 其他任务失败，需要取消相关任务                            │
│     ├── 资源不足，需要取消低优先级任务                            │
│     └── 数据更新，需要取消基于旧数据的任务                        │
│                                                                 │
│  4. 程序关闭                                                    │
│     ├── 优雅关闭（graceful shutdown）                            │
│     └── 强制关闭（需要清理资源）                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### task.cancel() 基本用法

```python
import asyncio

async def long_running_task():
    try:
        print("任务开始")
        await asyncio.sleep(100)  # 长时间运行
        print("任务完成")
    except asyncio.CancelledError:
        print("任务被取消")
        raise  # 重新抛出，让调用者知道任务被取消了

async def main():
    task = asyncio.create_task(long_running_task())

    # 等待一小段时间
    await asyncio.sleep(1)

    # 取消任务
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        print("主函数：任务已被取消")

asyncio.run(main())
# 输出：
# 任务开始
# 任务被取消
# 主函数：任务已被取消
```

### cancel() 的返回值

`task.cancel()` 返回一个布尔值，表示取消请求是否成功发出：

```python
async def main():
    task = asyncio.create_task(some_coroutine())

    # 任务还没开始，取消请求成功
    result = task.cancel()  # True
    print(f"取消结果: {result}")

    # 已经完成的任务无法取消
    task = asyncio.create_task(asyncio.sleep(0))
    await asyncio.sleep(0.1)  # 等待任务完成
    result = task.cancel()  # False
    print(f"取消已完成任务: {result}")
```

### cancel() 的参数 (msg)

Python 3.9+ 中，`cancel()` 接受一个可选的 `msg` 参数：

```python
async def main():
    task = asyncio.create_task(some_coroutine())

    # 带消息的取消
    task.cancel(msg="用户请求取消")

    try:
        await task
    except asyncio.CancelledError as e:
        print(f"取消消息: {e}")  # 可能包含取消消息
```

### 取消的状态检查

```python
async def main():
    task = asyncio.create_task(some_coroutine())

    print(f"任务状态: {task.done()}")       # False
    print(f"任务已取消: {task.cancelled()}")  # False

    task.cancel()

    # 注意：cancel() 只是发出取消请求
    # 在 await task 之前，任务可能还没真正被取消
    print(f"任务已取消: {task.cancelled()}")  # 可能还是 False

    try:
        await task
    except asyncio.CancelledError:
        pass

    print(f"任务完成: {task.done()}")         # True
    print(f"任务已取消: {task.cancelled()}")  # True
```
</div>

---

## 7.5 CancelledError

### CancelledError 的本质

`CancelledError` 是一种特殊的异常，它不是真正的"错误"，而是取消信号的载体：

```python
import asyncio

async def demonstrate_cancelled_error():
    try:
        await asyncio.sleep(float('inf'))
    except asyncio.CancelledError:
        print(f"异常类型: {type(asyncio.CancelledError)}")
        print(f"是否是 BaseException: {issubclass(asyncio.CancelledError, BaseException)}")
        print(f"是否是 Exception: {issubclass(asyncio.CancelledError, Exception)}")
        raise

async def main():
    task = asyncio.create_task(demonstrate_cancelled_error())
    await asyncio.sleep(1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print("捕获到 CancelledError")

asyncio.run(main())
```

### Python 3.9 的变化

在 Python 3.9 之前，`CancelledError` 继承自 `concurrent.futures.CancelledError`。
从 Python 3.9 开始，`asyncio.CancelledError` 直接继承自 `BaseException`（和 `KeyboardInterrupt` 同级），这意味着：

```python
# Python 3.9+ 的继承链
# BaseException
# ├── KeyboardInterrupt
# ├── SystemExit
# ├── GeneratorExit
# └── CancelledError  ← 现在在这里
#     └── Exception
#         ├── ...
#         └── TimeoutError
```

这个变化意味着**裸 `except Exception:` 不会捕获 `CancelledError`**：

```python
async def main():
    task = asyncio.create_task(some_coroutine())
    task.cancel()

    try:
        await task
    except Exception:  # Python 3.9+ 中不会捕获 CancelledError！
        print("不会执行到这里")
    except asyncio.CancelledError:
        print("正确捕获取消")  # 需要显式捕获
```

### 重新抛出 CancelledError

捕获 `CancelledError` 后，通常应该重新抛出：

```python
async def cleanup_task():
    try:
        await asyncio.sleep(100)
    except asyncio.CancelledError:
        print("执行清理操作...")
        await cleanup_resources()
        raise  # 重新抛出，让框架知道任务确实被取消了
```

### 不重新抛出的风险

```python
async def bad_example():
    try:
        await asyncio.sleep(100)
    except asyncio.CancelledError:
        print("捕获了取消，但没有重新抛出")
        # 不重新抛出意味着：任务"吞掉"了取消信号
        # 调用者会认为任务正常完成（而不是被取消）

async def main():
    task = asyncio.create_task(bad_example())
    await asyncio.sleep(1)
    task.cancel()
    await task

    # task.cancelled() 返回 False！
    # 因为 CancelledError 被吞掉了，任务看起来像是正常完成
    print(f"任务已取消: {task.cancelled()}")  # False
    print(f"任务完成: {task.done()}")         # True
    print(f"任务结果: {task.result()}")       # None
```

---

## 7.6 取消的传播

### 嵌套协程中的取消

当一个任务被取消时，取消信号会沿着 await 链向上传播：

```python
import asyncio

async def level_3():
    print("  进入 level_3")
    try:
        await asyncio.sleep(100)
    except asyncio.CancelledError:
        print("  level_3 收到取消信号")
        raise

async def level_2():
    print(" 进入 level_2")
    try:
        result = await level_3()
    except asyncio.CancelledError:
        print(" level_2 收到取消信号")
        raise

async def level_1():
    print("进入 level_1")
    try:
        result = await level_2()
    except asyncio.CancelledError:
        print("level_1 收到取消信号")
        raise

async def main():
    task = asyncio.create_task(level_1())
    await asyncio.sleep(1)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        print("主函数：任务被取消")

asyncio.run(main())
# 输出：
# 进入 level_1
#  进入 level_2
#   进入 level_3
#   level_3 收到取消信号
#  level_2 收到取消信号
# level_1 收到取消信号
# 主函数：任务被取消
```

### 取消传播的时序

```
┌─────────────────────────────────────────────────────────────────┐
│                    取消传播时序图                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  时间 ──────────────────────────────────────────────────────►   │
│                                                                 │
│  task.cancel()                                                  │
│      │                                                          │
│      ├──► level_3 中的 await sleep(100) 被唤醒                   │
│      │      │                                                   │
│      │      ├── 抛出 CancelledError                             │
│      │      └── 重新抛出（如果捕获了）                            │
│      │                                                          │
│      ├──► level_2 收到 CancelledError                           │
│      │      │                                                   │
│      │      └── 重新抛出                                        │
│      │                                                          │
│      ├──► level_1 收到 CancelledError                           │
│      │      │                                                   │
│      │      └── 重新抛出                                        │
│      │                                                          │
│      └──► task 完成，状态为 cancelled                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### gather() 与取消传播

当 `gather()` 被取消时，它会取消所有子任务：

```python
import asyncio

async def task_func(name, duration):
    try:
        print(f"任务 {name} 开始")
        await asyncio.sleep(duration)
        print(f"任务 {name} 完成")
        return name
    except asyncio.CancelledError:
        print(f"任务 {name} 被取消")
        raise

async def main():
    tasks = [
        task_func("A", 10),
        task_func("B", 20),
        task_func("C", 30),
    ]

    gather_task = asyncio.gather(*tasks)
    await asyncio.sleep(2)

    # 取消 gather 会取消所有子任务
    gather_task.cancel()

    try:
        await gather_task
    except asyncio.CancelledError:
        print("gather 被取消")

asyncio.run(main())
# 输出：
# 任务 A 开始
# 任务 B 开始
# 任务 C 开始
# 任务 A 被取消
# 任务 B 被取消
# 任务 C 被取消
# gather 被取消
```

### wait() 与取消

`asyncio.wait()` 的行为与 `gather()` 不同——它不会自动取消子任务：

```python
import asyncio

async def main():
    tasks = [
        asyncio.create_task(task_func("A", 10)),
        asyncio.create_task(task_func("B", 20)),
    ]

    # wait 返回时不会取消其他任务
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED
    )

    print(f"完成: {[t.get_name() for t in done]}")
    print(f"待定: {[t.get_name() for t in pending]}")

    # 需要手动取消待定的任务
    for task in pending:
        task.cancel()

asyncio.run(main())
```

---

## 7.7 屏蔽取消

<div data-component="ShieldDemo"></div>

### asyncio.shield() 的作用

有时候，我们希望保护某个协程不被外部取消。`asyncio.shield()` 可以做到这一点：

```python
import asyncio

async def critical_operation():
    """一个不能被取消的关键操作"""
    print("开始关键操作")
    await asyncio.sleep(5)
    print("关键操作完成")
    return "critical result"

async def main():
    try:
        # 使用 shield 保护关键操作
        result = await asyncio.shield(critical_operation())
        print(f"结果: {result}")
    except asyncio.CancelledError:
        print("外层被取消，但关键操作会继续运行")

asyncio.run(main())
```

### shield() 的工作原理

```
┌─────────────────────────────────────────────────────────────────┐
│                asyncio.shield() 内部机制                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  shield(coro)                                                   │
│      │                                                          │
│      ├── 1. 创建内部 Task 来运行 coro                           │
│      │                                                          │
│      ├── 2. 返回一个"包装"的 Future（shield_future）             │
│      │                                                          │
│      ├── 3. 当 shield_future 被取消时：                         │
│      │      ├── 取消 shield_future 本身                         │
│      │      └── 但 不取消 内部 Task                              │
│      │                                                          │
│      └── 4. 内部 Task 继续运行直到完成                           │
│             └── 其结果/异常会传播给 shield_future（如果还在等待） │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### shield() 的关键行为

```python
import asyncio

async def shielded_operation():
    print("shielded 操作开始")
    await asyncio.sleep(3)
    print("shielded 操作完成")
    return "shielded result"

async def main():
    # 创建一个会取消的任务
    async def canceller(task):
        await asyncio.sleep(1)
        task.cancel()

    # shield 保护的操作
    shielded = asyncio.shield(shielded_operation())

    # 同时启动取消器
    asyncio.create_task(canceller(asyncio.current_task()))

    try:
        result = await shielded
        print(f"结果: {result}")
    except asyncio.CancelledError:
        print("await 被取消了，但 shielded_operation 仍在后台运行")
        # shielded_operation 会继续运行直到完成

asyncio.run(main())
```

### shield() 的陷阱

**陷阱 1：shield() 不能阻止所有取消**

```python
import asyncio

async def main():
    # shield 只能阻止通过 shield_future 发出的取消
    # 如果内部任务本身被直接取消，shield 无法阻止
    inner_task = asyncio.create_task(critical_operation())
    shielded = asyncio.shield(inner_task)

    # 直接取消内部任务（绕过 shield）
    inner_task.cancel()  # 这会真正取消内部任务

    try:
        await shielded
    except asyncio.CancelledError:
        print("内部任务被直接取消了")
```

**陷阱 2：shield() 的结果可能丢失**

```python
import asyncio

async def main():
    shielded = asyncio.shield(critical_operation())

    # 不 await shielded，直接取消
    shielded.cancel()

    # critical_operation 仍在后台运行
    # 但其结果会被丢弃（没有人等待它）
    # 这可能导致"Task exception was never retrieved"警告

    await asyncio.sleep(10)  # 等待后台任务完成

asyncio.run(main())
```

**陷阱 3：shield() 不能防止异常传播**

```python
import asyncio

async def failing_operation():
    await asyncio.sleep(1)
    raise ValueError("操作失败")

async def main():
    try:
        # shield 不能阻止内部异常传播
        await asyncio.shield(failing_operation())
    except ValueError as e:
        print(f"捕获到内部异常: {e}")  # 会执行

asyncio.run(main())
```

### 何时使用 shield()

```
┌─────────────────────────────────────────────────────────────────┐
│                    shield() 适用场景                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ✓ 适用场景：                                                   │
│    ├── 日志记录：取消时仍需记录日志                               │
│    ├── 资源清理：关闭连接、释放锁等                               │
│    ├── 状态保存：将当前进度持久化                                 │
│    └── 通知：告知其他系统操作已取消                               │
│                                                                 │
│  ✗ 不适用场景：                                                  │
│    ├── 需要真正"不可取消"的操作（应使用 ensure_future + 独立管理） │
│    ├── 长时间运行的后台任务（应创建独立任务）                      │
│    └── 关键业务逻辑（取消后应该真正停止）                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
</div>

---

## 7.8 异常处理基础

<div data-component="ExceptionPropagationDemo"></div>

### 协程中的 try/except

协程中的异常处理与普通函数类似，但有一些特殊的考虑：

```python
import asyncio

async def risky_operation():
    """可能抛出异常的操作"""
    await asyncio.sleep(1)
    if random.random() < 0.5:
        raise ValueError("随机失败")
    return "success"

async def main():
    try:
        result = await risky_operation()
        print(f"结果: {result}")
    except ValueError as e:
        print(f"捕获到错误: {e}")
    except asyncio.CancelledError:
        print("操作被取消")
        raise

asyncio.run(main())
```

### 异常在协程中的传播

```python
import asyncio

async def inner():
    raise ValueError("内部错误")

async def middle():
    await inner()  # 异常会从这里传播出去

async def outer():
    await middle()  # 继续传播

async def main():
    try:
        await outer()
    except ValueError as e:
        print(f"在 main 中捕获: {e}")
        # 可以查看完整的异常链
        import traceback
        traceback.print_exc()

asyncio.run(main())
```

输出：

```
在 main 中捕获: 内部错误
Traceback (most recent call last):
  File "example.py", line 14, in main
    await outer()
  File "example.py", line 10, in outer
    await middle()
  File "example.py", line 7, in middle
    await inner()
  File "example.py", line 4, in inner
    raise ValueError("内部错误")
ValueError: 内部错误
```

### 未捕获的异常

如果协程中的异常没有被捕获，它会在 `await` 时重新抛出：

```python
import asyncio

async def failing_coroutine():
    raise ValueError("未捕获的异常")

async def main():
    # 如果不在这里 try/except，程序会崩溃
    await failing_coroutine()
    # ValueError: 未捕获的异常
    # 程序终止

asyncio.run(main())
```

### Task 中的异常

当异常发生在 Task 中且未被捕获时，它会被存储在 Task 对象中：

```python
import asyncio

async def failing_task():
    await asyncio.sleep(1)
    raise ValueError("任务失败")

async def main():
    task = asyncio.create_task(failing_task())

    # 等待任务完成
    await asyncio.sleep(2)

    # 检查任务状态
    print(f"任务完成: {task.done()}")      # True
    print(f"任务已取消: {task.cancelled()}")  # False

    # 获取异常（不抛出）
    exception = task.exception()
    print(f"异常类型: {type(exception).__name__}")
    print(f"异常消息: {exception}")

    # 或者调用 result() 会重新抛出异常
    try:
        task.result()
    except ValueError as e:
        print(f"重新抛出: {e}")

asyncio.run(main())
```

### add_done_callback 与异常

```python
import asyncio

def task_callback(task):
    """任务完成时的回调"""
    if task.cancelled():
        print("回调: 任务被取消")
    elif task.exception():
        print(f"回调: 任务出错 - {task.exception()}")
    else:
        print(f"回调: 任务成功 - {task.result()}")

async def main():
    task = asyncio.create_task(failing_task())
    task.add_done_callback(task_callback)

    # 等待回调执行
    await asyncio.sleep(2)

asyncio.run(main())
```
</div>

---

## 7.9 gather() 的异常处理

<div data-component="GatherExceptionHandling"></div>

### 默认行为：第一个异常立即传播

```python
import asyncio

async def task_a():
    await asyncio.sleep(1)
    return "A"

async def task_b():
    await asyncio.sleep(2)
    raise ValueError("B 失败")

async def task_c():
    await asyncio.sleep(3)
    return "C"

async def main():
    try:
        # 默认行为：第一个异常会立即传播
        results = await asyncio.gather(task_a(), task_b(), task_c())
        print(results)
    except ValueError as e:
        print(f"捕获到异常: {e}")
        # 此时 task_c 可能还在运行，但 gather 已经返回了

asyncio.run(main())
# 输出：捕获到异常: B 失败
```

### return_exceptions=True

使用 `return_exceptions=True` 可以让所有任务都完成，异常作为结果返回：

```python
import asyncio

async def main():
    # return_exceptions=True：异常作为结果返回
    results = await asyncio.gather(
        task_a(),
        task_b(),
        task_c(),
        return_exceptions=True
    )

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"任务 {i}: 失败 - {result}")
        else:
            print(f"任务 {i}: 成功 - {result}")

asyncio.run(main())
# 输出：
# 任务 0: 成功 - A
# 任务 1: 失败 - B 失败
# 任务 2: 成功 - C
```

### 两种模式的对比

```
┌─────────────────────────────────────────────────────────────────┐
│          gather() 异常处理模式对比                                │
├───────────────────────────────────┬─────────────────────────────┤
│  return_exceptions=False（默认）   │  return_exceptions=True     │
├───────────────────────────────────┼─────────────────────────────┤
│  第一个异常立即传播                 │  所有任务都运行到完成         │
│  其他任务继续运行（但结果被忽略）   │  异常作为结果返回，不抛出     │
│  适合：任务必须全部成功             │  适合：允许部分失败           │
│  捕获异常需要 try/except           │  检查结果列表中的异常         │
└───────────────────────────────────┴─────────────────────────────┘
```

### 实际应用：部分失败的批量请求

```python
import asyncio
import aiohttp

async def fetch_url(session, url):
    """获取单个 URL"""
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
            return await response.json()
    except Exception as e:
        return {"error": str(e), "url": url}

async def fetch_all(urls):
    """批量获取，允许部分失败"""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 分类结果
        successes = []
        failures = []
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                failures.append({"url": url, "error": str(result)})
            elif isinstance(result, dict) and "error" in result:
                failures.append(result)
            else:
                successes.append({"url": url, "data": result})

        return successes, failures
```

### gather() 的取消与异常

```python
import asyncio

async def main():
    # 当 gather 被取消时，所有未完成的任务都会被取消
    async def slow_task(i):
        try:
            await asyncio.sleep(10)
            return i
        except asyncio.CancelledError:
            print(f"任务 {i} 被取消")
            raise

    gather_coro = asyncio.gather(
        slow_task(1),
        slow_task(2),
        slow_task(3)
    )

    task = asyncio.create_task(gather_coro)
    await asyncio.sleep(1)

    # 取消 gather
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        print("gather 被取消")

asyncio.run(main())
```
</div>

---

## 7.10 未处理的异常

### "Task exception was never retrieved" 警告

当一个 Task 有未检索的异常时，Python 会在垃圾回收时打印警告：

```python
import asyncio

async def failing():
    raise ValueError("未处理的异常")

async def main():
    # 创建任务但不 await
    task = asyncio.create_task(failing())

    # 等待任务完成
    await asyncio.sleep(1)

    # 不调用 task.result() 或 task.exception()
    # 当 task 被垃圾回收时，会看到警告：
    # Task exception was never retrieved
    # future: <Task finished coro=<failing() done, defined at ...> exception=ValueError('未处理的异常')>
    # ValueError: 未处理的异常

asyncio.run(main())
```

### 如何避免这个警告

**方法 1：显式获取异常**

```python
async def main():
    task = asyncio.create_task(failing())
    await asyncio.sleep(1)

    # 方法 1：调用 exception()
    try:
        exc = task.exception()
        if exc:
            print(f"任务失败: {exc}")
    except asyncio.CancelledError:
        print("任务被取消")

    # 方法 2：调用 result()（会重新抛出）
    try:
        result = task.result()
    except ValueError as e:
        print(f"任务失败: {e}")
```

**方法 2：添加回调**

```python
async def main():
    def handle_task_result(task):
        try:
            task.result()
        except Exception as e:
            print(f"任务失败: {e}")

    task = asyncio.create_task(failing())
    task.add_done_callback(handle_task_result)

    await asyncio.sleep(1)
```

**方法 3：使用 asyncio.TaskGroup (Python 3.11+)**

```python
async def main():
    try:
        async with asyncio.TaskGroup() as tg:
            task1 = tg.create_task(failing())
            task2 = tg.create_task(another_task())
    except* ValueError as eg:
        # eg 是 ExceptionGroup，包含所有 ValueError
        for exc in eg.exceptions:
            print(f"捕获到: {exc}")
```

### 何时会出现未检索异常

```
┌─────────────────────────────────────────────────────────────────┐
│           "Task exception was never retrieved" 触发条件          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 创建了 Task 但从未 await                                    │
│     task = asyncio.create_task(may_fail())                      │
│     # 忘记 await task                                           │
│                                                                 │
│  2. gather() 的 return_exceptions=False 时部分任务失败           │
│     # 只捕获了第一个异常，其他任务的异常未检索                    │
│                                                                 │
│  3. Task 被取消但 exception() 未被调用                           │
│     task.cancel()                                               │
│     # 没有 await task 或调用 task.exception()                    │
│                                                                 │
│  4. fire-and-forget 模式                                        │
│     asyncio.create_task(background_task())                      │
│     # 如果 background_task() 失败，异常无人处理                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### fire-and-forget 的正确做法

```python
import asyncio
import logging

logger = logging.getLogger(__name__)

def fire_and_forget(coro):
    """安全的 fire-and-forget 封装"""
    async def wrapper():
        try:
            await coro
        except Exception:
            logger.exception("后台任务失败")
        except asyncio.CancelledError:
            logger.info("后台任务被取消")

    task = asyncio.create_task(wrapper())
    return task

async def background_cleanup():
    """可能失败的后台任务"""
    await asyncio.sleep(10)
    raise RuntimeError("清理失败")

async def main():
    # 安全的 fire-and-forget
    fire_and_forget(background_cleanup())

    # 主逻辑继续运行
    await asyncio.sleep(1)

asyncio.run(main())
```

---

## 7.11 异常与取消的关系

### 取消也是一种"异常"

```python
import asyncio

async def main():
    task = asyncio.create_task(some_coroutine())
    task.cancel()

    try:
        await task
    except asyncio.CancelledError as e:
        # CancelledError 是 BaseException 的子类
        print(f"异常类型: {type(e)}")
        print(f"是 BaseException: {isinstance(e, BaseException)}")
        print(f"是 Exception: {isinstance(e, Exception)}")  # Python 3.9+ 中为 False

asyncio.run(main())
```

### 异常处理中的取消

```python
import asyncio

async def operation_with_cleanup():
    """带清理的异常处理"""
    resource = await acquire_resource()
    try:
        result = await use_resource(resource)
        return result
    except ValueError:
        # 处理业务异常
        logger.error("值错误")
        return None
    except asyncio.CancelledError:
        # 取消时也要清理资源
        logger.info("操作被取消，正在清理...")
        await cleanup_resource(resource)
        raise  # 重新抛出取消信号
    finally:
        # finally 块在取消时也会执行
        release_resource(resource)
```

### 异常链和取消

```python
import asyncio

async def inner():
    try:
        await asyncio.sleep(100)
    except asyncio.CancelledError:
        raise ValueError("转换为业务异常") from None

async def main():
    task = asyncio.create_task(inner())
    await asyncio.sleep(1)
    task.cancel()

    try:
        await task
    except ValueError as e:
        print(f"捕获到转换后的异常: {e}")

asyncio.run(main())
```

### 取消后异常的处理顺序

```python
import asyncio

async def complex_operation():
    """复杂的异常处理场景"""
    try:
        await asyncio.sleep(100)
    except asyncio.CancelledError:
        print("收到取消信号")
        # 执行清理操作
        await save_state()
        raise
    except Exception as e:
        print(f"其他异常: {e}")
        raise
    finally:
        print("finally 块执行")  # 取消时也会执行

async def save_state():
    """保存状态"""
    print("保存状态...")
    await asyncio.sleep(0.1)
    print("状态已保存")

async def main():
    task = asyncio.create_task(complex_operation())
    await asyncio.sleep(1)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        print("主函数：任务被取消")

asyncio.run(main())
# 输出：
# 收到取消信号
# 保存状态...
# 状态已保存
# finally 块执行
# 主函数：任务被取消
```

---

## 7.12 资源清理

### try/finally 与异步

```python
import asyncio

async def acquire_database_connection():
    """模拟获取数据库连接"""
    print("获取数据库连接...")
    await asyncio.sleep(0.5)
    return {"connection": "db_conn_123"}

async def release_database_connection(conn):
    """模拟释放数据库连接"""
    print(f"释放数据库连接 {conn['connection']}...")
    await asyncio.sleep(0.2)
    print("连接已释放")

async def query_database(conn):
    """模拟数据库查询"""
    await asyncio.sleep(1)
    return [{"id": 1, "name": "Alice"}]

async def safe_database_operation():
    """带资源清理的数据库操作"""
    conn = await acquire_database_connection()
    try:
        result = await query_database(conn)
        return result
    finally:
        # 无论成功、失败还是取消，都会执行清理
        await release_database_connection(conn)

async def main():
    # 正常情况
    result = await safe_database_operation()
    print(f"结果: {result}")

    # 取消情况
    task = asyncio.create_task(safe_database_operation())
    await asyncio.sleep(0.6)  # 获取连接后
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        print("操作被取消，但资源已清理")

asyncio.run(main())
```

### 异步上下文管理器

```python
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def database_connection():
    """异步上下文管理器"""
    print("获取连接")
    conn = await acquire_database_connection()
    try:
        yield conn
    except asyncio.CancelledError:
        print("操作被取消")
        raise
    finally:
        print("释放连接")
        await release_database_connection(conn)

async def main():
    # 使用 async with
    async with database_connection() as conn:
        result = await query_database(conn)
        print(f"结果: {result}")

    # 取消时也会正确清理
    async def operation():
        async with database_connection() as conn:
            await asyncio.sleep(100)  # 长时间操作

    task = asyncio.create_task(operation())
    await asyncio.sleep(1)
    task.cancel()

    try:
        await task
    except asyncio.CancelledError:
        print("任务取消，资源已清理")

asyncio.run(main())
```

### 多资源清理

```python
import asyncio

async def complex_operation():
    """多个资源的清理"""
    conn = await acquire_database_connection()
    cache = await acquire_cache_connection()
    lock = await acquire_lock()

    try:
        # 使用资源
        data = await query_database(conn)
        await cache_set(cache, "result", data)
        return data
    finally:
        # 按相反顺序释放资源
        await release_lock(lock)
        await release_cache_connection(cache)
        await release_database_connection(conn)
```

### 清理时的异常处理

```python
import asyncio

async def risky_cleanup():
    """可能失败的清理操作"""
    try:
        await asyncio.sleep(100)
    except asyncio.CancelledError:
        print("收到取消信号")
        try:
            await save_to_disk()
        except Exception as e:
            # 清理失败，记录但不吞掉取消信号
            print(f"清理失败: {e}")
        raise  # 仍然重新抛出 CancelledError

async def save_to_disk():
    raise IOError("磁盘满")
```

---

## 7.13 实际模式

### 模式 1：超时 + 降级

```python
import asyncio
from typing import Optional, Any

async def fetch_with_timeout_and_fallback(
    primary_coro,
    fallback_coro,
    timeout: float
) -> Any:
    """超时后降级"""
    try:
        return await asyncio.wait_for(primary_coro, timeout=timeout)
    except asyncio.TimeoutError:
        return await fallback_coro

async def get_user_data(user_id: int) -> dict:
    """获取用户数据（带降级）"""
    # 第一选择：从 API 获取
    api_data = await fetch_with_timeout_and_fallback(
        primary_coro=fetch_from_api(user_id),
        fallback_coro=get_from_cache(user_id),
        timeout=5.0
    )

    if api_data:
        return api_data

    # 第二选择：从缓存获取（已经是降级了）
    cache_data = await get_from_cache(user_id)
    if cache_data:
        return cache_data

    # 最后选择：返回默认值
    return {"user_id": user_id, "name": "Unknown", "cached": False}
```

### 模式 2：批量操作的超时

```python
import asyncio
from typing import List, Optional

async def fetch_with_individual_timeout(
    urls: List[str],
    timeout: float
) -> List[Optional[dict]]:
    """每个请求独立超时"""
    async def fetch_one(url: str) -> Optional[dict]:
        try:
            return await asyncio.wait_for(
                fetch_from_url(url),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None
        except Exception:
            return None

    tasks = [fetch_one(url) for url in urls]
    return await asyncio.gather(*tasks)

async def fetch_with_total_timeout(
    urls: List[str],
    timeout: float
) -> List[Optional[dict]]:
    """整体超时"""
    async def fetch_one(url: str) -> Optional[dict]:
        return await fetch_from_url(url)

    tasks = [fetch_one(url) for url in urls]
    try:
        return await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        # 超时后，gather 中的任务会被取消
        return [None] * len(urls)
```

### 模式 3：优雅关闭

```python
import asyncio
import signal

class GracefulShutdown:
    """优雅关闭管理器"""

    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.tasks: List[asyncio.Task] = []

    def register_task(self, task: asyncio.Task):
        self.tasks.append(task)

    async def shutdown(self, timeout: float = 10.0):
        """执行优雅关闭"""
        print("开始优雅关闭...")

        # 1. 设置关闭事件
        self.shutdown_event.set()

        # 2. 等待所有任务完成（带超时）
        if self.tasks:
            done, pending = await asyncio.wait(
                self.tasks,
                timeout=timeout
            )

            # 3. 取消超时的任务
            for task in pending:
                print(f"取消超时任务: {task.get_name()}")
                task.cancel()

            # 4. 等待取消完成
            if pending:
                await asyncio.wait(pending)

        print("优雅关闭完成")

async def worker(shutdown: GracefulShutdown, name: str):
    """工作协程"""
    while not shutdown.shutdown_event.is_set():
        try:
            await asyncio.sleep(1)
            print(f"{name}: 工作中...")
        except asyncio.CancelledError:
            print(f"{name}: 被取消，执行清理...")
            await asyncio.sleep(0.5)  # 模拟清理
            raise

async def main():
    shutdown = GracefulShutdown()

    # 启动多个工作者
    tasks = []
    for i in range(3):
        task = asyncio.create_task(worker(shutdown, f"Worker-{i}"))
        shutdown.register_task(task)
        tasks.append(task)

    # 运行一段时间
    await asyncio.sleep(5)

    # 优雅关闭
    await shutdown.shutdown(timeout=3.0)

asyncio.run(main())
```

### 模式 4：取消令牌模式

```python
import asyncio

class CancellationToken:
    """取消令牌"""

    def __init__(self):
        self._cancelled = False
        self._event = asyncio.Event()

    def cancel(self):
        self._cancelled = True
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    async def wait_for_cancel(self):
        await self._event.wait()

    async def check_cancelled(self):
        """检查是否已取消，如果是则抛出 CancelledError"""
        if self._cancelled:
            raise asyncio.CancelledError()

async def cancellable_operation(token: CancellationToken, name: str):
    """可取消的操作"""
    for i in range(10):
        await token.check_cancelled()
        await asyncio.sleep(1)
        print(f"{name}: 步骤 {i}")

async def main():
    token = CancellationToken()

    # 启动多个操作
    tasks = [
        asyncio.create_task(cancellable_operation(token, f"Op-{i}"))
        for i in range(3)
    ]

    # 3 秒后取消所有操作
    await asyncio.sleep(3)
    token.cancel()

    # 等待所有任务完成
    await asyncio.gather(*tasks, return_exceptions=True)

asyncio.run(main())
```

### 模式 5：重试与超时结合

```python
import asyncio
from typing import TypeVar, Callable, Optional

T = TypeVar('T')

async def retry_with_timeout(
    coro_factory: Callable[[], T],
    max_retries: int = 3,
    timeout: float = 5.0,
    delay: float = 1.0
) -> T:
    """带超时的重试"""
    last_exception = None

    for attempt in range(max_retries):
        try:
            return await asyncio.wait_for(coro_factory(), timeout=timeout)
        except asyncio.TimeoutError:
            last_exception = TimeoutError(f"超时 (尝试 {attempt + 1}/{max_retries})")
            print(f"尝试 {attempt + 1} 超时")
        except Exception as e:
            last_exception = e
            print(f"尝试 {attempt + 1} 失败: {e}")

        if attempt < max_retries - 1:
            await asyncio.sleep(delay * (attempt + 1))  # 指数退避

    raise last_exception

async def unstable_api():
    """不稳定的 API"""
    import random
    await asyncio.sleep(random.uniform(0.1, 6))
    if random.random() < 0.3:
        raise ConnectionError("连接失败")
    return {"data": "success"}

async def main():
    try:
        result = await retry_with_timeout(unstable_api, max_retries=3, timeout=2.0)
        print(f"结果: {result}")
    except Exception as e:
        print(f"所有重试都失败: {e}")

asyncio.run(main())
```

---

## 7.14 常见误区

### 误区 1：cancel() 会立即停止任务

```python
# ❌ 错误理解：cancel() 会立即停止任务
task.cancel()
print(task.done())  # 期望 True，实际可能是 False

# ✓ 正确理解：cancel() 只是发出取消请求
task.cancel()
try:
    await task  # 需要 await 才能等待取消完成
except asyncio.CancelledError:
    pass
print(task.done())  # 现在是 True
```

### 误区 2：CancelledError 可以被 Exception 捕获

```python
# ❌ Python 3.9+ 中不会捕获 CancelledError
try:
    await asyncio.sleep(100)
except Exception:  # CancelledError 继承自 BaseException
    print("不会执行")

# ✓ 需要显式捕获
try:
    await asyncio.sleep(100)
except asyncio.CancelledError:
    print("正确捕获")
    raise
```

### 误区 3：shield() 可以完全阻止取消

```python
# ❌ shield() 只能阻止通过它的取消，不能阻止直接取消
inner_task = asyncio.create_task(critical_operation())
shielded = asyncio.shield(inner_task)
inner_task.cancel()  # 这会真正取消

# ✓ 如果需要完全独立的任务，直接创建 Task
independent_task = asyncio.create_task(critical_operation())
```

### 误区 4：wait_for() 超时后任务立即停止

```python
# ❌ 超时后任务可能仍在运行（Python 3.10-）
result = await asyncio.wait_for(slow_task(), timeout=5.0)
# slow_task 可能还在运行

# ✓ Python 3.11+ 修复了这个问题
# 但仍然建议在 slow_task 中检查取消信号
```

### 误区 5：吞掉 CancelledError

```python
# ❌ 吞掉取消信号
async def bad():
    try:
        await asyncio.sleep(100)
    except asyncio.CancelledError:
        print("被取消了")
        # 没有重新抛出！任务看起来正常完成

# ✓ 重新抛出
async def good():
    try:
        await asyncio.sleep(100)
    except asyncio.CancelledError:
        print("被取消了")
        raise  # 重新抛出
```

### 误区 6：忘记处理未检索的异常

```python
# ❌ 可能导致 "Task exception was never retrieved"
async def main():
    asyncio.create_task(failing_coroutine())
    await asyncio.sleep(1)

# ✓ 正确处理
async def main():
    task = asyncio.create_task(failing_coroutine())
    task.add_done_callback(handle_exception)
    await asyncio.sleep(1)
```

### 误区 7：gather() 会等待所有任务完成

```python
# ❌ 默认情况下，第一个异常就会中断 gather
results = await asyncio.gather(
    task_a(),  # 成功
    task_b(),  # 失败
    task_c(),  # 可能还没开始或被取消
)

# ✓ 使用 return_exceptions=True 等待所有任务
results = await asyncio.gather(
    task_a(),
    task_b(),
    task_c(),
    return_exceptions=True
)
```

### 误区 8：超时时间为 0 表示"不超时"

```python
# ❌ timeout=0 会立即超时
result = await asyncio.wait_for(slow_task(), timeout=0)  # 超时

# ✓ 不超时应该使用 timeout=None
result = await asyncio.wait_for(slow_task(), timeout=None)
```

### 误区 9：在 finally 中抛出异常

```python
# ❌ finally 中的异常会覆盖原始异常
async def bad():
    try:
        await risky_operation()
    except asyncio.CancelledError:
        raise
    finally:
        raise ValueError("清理失败")  # 会吞掉 CancelledError！

# ✓ 在 finally 中捕获并记录异常
async def good():
    try:
        await risky_operation()
    except asyncio.CancelledError:
        raise
    finally:
        try:
            await cleanup()
        except Exception as e:
            print(f"清理失败: {e}")  # 记录但不抛出
```

### 误区 10：直接 await shield() 的结果

```python
# ❌ 这样写没问题，但容易误解
result = await asyncio.shield(critical_operation())

# 误解：以为 shield 创建了一个"不可取消"的 Future
# 实际：shield 只保护内部 Task 不被通过 shield_future 取消
# 如果当前 Task 被取消，shield_future 也会被取消（但内部 Task 继续）
```

---

## 本章小结

本章系统讲解了 Python 异步编程中的超时控制、任务取消和异常处理机制。以下是关键要点：

**超时控制**：
- `asyncio.wait_for()` 是最常用的超时工具，它会在超时时取消内部任务并抛出 `TimeoutError`
- Python 3.11+ 修复了 `wait_for()` 在超时时不等待任务取消完成的问题
- 超时可以嵌套，外层超时会取消内层的所有操作

**任务取消**：
- `task.cancel()` 只是发出取消请求，不会立即停止任务
- `CancelledError` 在 Python 3.9+ 中继承自 `BaseException`，需要显式捕获
- 取消信号会沿着 `await` 链传播，所有等待中的协程都会收到
- 捕获 `CancelledError` 后通常应该重新抛出

**shield() 保护**：
- `asyncio.shield()` 保护内部 Task 不被通过 shield_future 取消
- shield 不能阻止直接取消内部任务
- shield 不能阻止内部异常传播

**异常处理**：
- `gather()` 默认在第一个异常时中断，使用 `return_exceptions=True` 等待所有任务
- "Task exception was never retrieved" 警告表示有未处理的异常
- 使用回调、await 或 TaskGroup 来处理异常

**资源清理**：
- `try/finally` 在取消时也会执行
- 异步上下文管理器（`async with`）是管理资源的推荐方式
- 清理操作中的异常不应吞掉原始异常

**最佳实践**：
- 总是为外部调用设置超时
- 捕获 `CancelledError` 后执行清理并重新抛出
- 使用 `return_exceptions=True` 处理批量操作
- 为 fire-and-forget 任务添加异常处理

---

## 思考题

### 基础题

1. **超时机制**：为什么 `asyncio.wait_for()` 在 timeout=0 时会立即超时？这个行为在什么场景下有用？

2. **取消传播**：当一个嵌套 3 层的协程被取消时，取消信号是如何传播的？如果中间某一层捕获了 `CancelledError` 但没有重新抛出，会发生什么？

3. **shield() 行为**：解释 `asyncio.shield()` 的工作原理。它能保护内部任务不被直接取消吗？

4. **gather() 异常**：`gather()` 的 `return_exceptions` 参数默认是什么值？两种值分别适合什么场景？

5. **CancelledError 继承**：Python 3.9 中 `CancelledError` 的继承链是什么？为什么这个变化很重要？

### 进阶题

6. **实现超时降级**：编写一个 `fetch_with_fallback()` 函数，实现以下逻辑：
   - 先尝试从主服务器获取（5 秒超时）
   - 如果超时，尝试从备份服务器获取（3 秒超时）
   - 如果还是超时，从缓存获取
   - 要求：每一步的超时和降级都要有日志

7. **优雅关闭**：实现一个 `GracefulShutdown` 类，支持：
   - 注册多个需要优雅关闭的任务
   - 发送关闭信号后等待所有任务完成（带超时）
   - 超时后强制取消未完成的任务
   - 确保所有资源都被正确清理

8. **取消令牌**：实现一个 `CancellationToken` 类，支持：
   - 多个任务共享同一个令牌
   - 调用 `cancel()` 后所有任务都会收到通知
   - 任务可以在每个步骤中检查是否被取消
   - 支持等待取消事件

9. **重试与超时**：实现一个 `retry_with_timeout()` 函数：
   - 支持设置最大重试次数
   - 每次重试有独立的超时时间
   - 支持指数退避
   - 返回最后一次成功的结果或抛出最后一次的异常

10. **异常聚合**：使用 `gather()` 和 `return_exceptions=True` 实现一个批量操作函数：
    - 并发执行多个任务
    - 收集所有成功和失败的结果
    - 返回一个包含成功列表和失败列表的结果
    - 处理超时和取消的情况

### 思辨题

11. **设计决策**：为什么 Python 3.9 将 `CancelledError` 改为继承自 `BaseException` 而不是 `Exception`？这个变化带来了什么好处和坏处？

12. **取消 vs 异常**：在什么情况下应该使用取消（cancel）而不是抛出异常？在什么情况下应该使用异常而不是取消？

13. **资源清理优先级**：当取消和异常同时发生时（例如在清理过程中又发生了取消），应该如何处理？Python 的异常处理机制如何支持这种情况？

14. **超时层次设计**：在一个复杂系统中，应该在哪些层次设置超时？如何避免超时冲突（例如外层超时比内层短）？

15. **fire-and-forget**：什么场景下适合使用 fire-and-forget 模式？如何平衡"不关心结果"和"需要处理异常"的需求？

---
title: "Coroutine、Task 与 Future 的关系"
description: "深入理解 Python asyncio 中协程对象、Task 和 Future 三种核心类型的区别与联系，掌握它们的创建方式、状态机、异常处理与实际应用模式"
updated: "2026-07-07"
---

# Coroutine、Task 与 Future 的关系

> **学习目标**：
> - 理解协程对象（Coroutine）的本质：`async def` 调用后产生的对象，而非立即执行的代码
> - 掌握 Task 的创建方式（`create_task`）及其在事件循环中的调度机制
> - 理解 Future 作为"结果占位符"的设计理念及其与回调机制的关系
> - 清晰区分三者的层级关系：Task 是特殊的 Future，Future 包装协程
> - 掌握 Task 的四种状态（Pending / Running / Done / Cancelled）及状态转换
> - 学会正确获取 Task 结果（`await task` vs `task.result()`）和处理异常
> - 理解协程对象不能重复 await、Task 可以多次 await 的区别
> - 识别 Coroutine / Task / Future 使用中的常见误区

在前面的章节中，我们已经知道 `async def` 定义的是一个协程函数，调用它会得到一个协程对象。但仅仅有协程对象是不够的——协程对象自己不会运行，它需要被"提交"给事件循环。

这一章要回答以下核心问题：

1. 协程对象（Coroutine）到底是什么？它和普通的函数调用结果有什么不同？
2. Task 是什么？为什么需要把协程包装成 Task？
3. Future 又是什么？它和 Task 是什么关系？
4. 这三者之间如何互相转换、互相配合？

搞清楚这三个概念的关系，是真正掌握 asyncio 的关键一步。

---

## 4.1 现实类比：外卖订单

<div data-component="CoroutineTaskFutureDiagram"></div>

在深入技术细节之前，我们先用一个生活中的例子来建立直觉。想象你在一个外卖平台上点餐。

### 三种角色

```
┌─────────────────────────────────────────────────────────────────────┐
│                       外卖点餐的三个概念                              │
│                                                                     │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│   │   菜谱        │    │   已提交的订单 │    │   结果占位符  │         │
│   │  (Coroutine)  │    │  (Task)       │    │  (Future)    │         │
│   │              │    │              │    │              │         │
│   │  告诉你怎么做  │    │  告诉厨房开始  │    │  你的取餐码   │         │
│   │  但还没开始做  │    │  做这道菜了    │    │  会在做好后   │         │
│   │              │    │              │    │  被填入结果    │         │
│   └──────────────┘    └──────────────┘    └──────────────┘         │
│                                                                     │
│   菜谱本身不会自动         订单被厨房调度执       取餐码是一个           │
│   产生食物。你需要         行，厨师会按顺序       "未来会有结果"         │
│   提交订单才能启动。       处理多个订单。         的承诺。               │
└─────────────────────────────────────────────────────────────────────┘
```

### 完整流程

```
你（程序员）              外卖平台（事件循环）          厨房（执行环境）
│                            │                          │
├── 看菜谱                    │                          │
│   (定义 async def)          │                          │
│                            │                          │
├── 选菜，提交订单 ──────────→│                          │
│   (create_task)             │                          │
│                            ├── 分配取餐码              │
│←── 拿到取餐码 ─────────────┤  (返回 Task 对象)         │
│   (Task = Future)           │                          │
│                            ├── 把订单发给厨房 ────────→│
│                            │                          ├── 开始做菜
│                            │                          │   (协程开始执行)
├── 刷手机、做其他事          │                          │
│   (当前协程继续执行)         │                          │
│                            │                          ├── 做好了
│                            │←──────────────────────────┤
│                            ├── 更新取餐码状态为"完成"   │
│                            │                          │
├── 用取餐码取餐              │                          │
│   (await task 获取结果)      │                          │
│                            │                          │
```

### 三者的对应关系

| 外卖概念 | Python 概念 | 本质 |
|---------|------------|------|
| 菜谱 | Coroutine（协程对象） | 描述"做什么"的蓝图，本身不执行 |
| 已提交的订单 | Task | 已被事件循环调度，正在或即将执行 |
| 取餐码 | Future | 结果的占位符，最终会被填入值 |

**关键洞察**：
- 你不会拿着菜谱就直接吃到菜——菜谱只是说明怎么做（**协程对象不会自动执行**）
- 你需要把菜谱提交给平台，平台才会安排厨师做（**`create_task` 把协程交给事件循环**）
- 提交后你拿到一个取餐码，这个码代表"未来会有结果"（**Task 是一个 Future**）
- 你可以先做别的事，等饿了再用取餐码取餐（**`await task` 获取最终结果**）

</div>
---

## 4.2 Coroutine 是什么

Coroutine（协程对象）是调用 `async def` 函数后返回的对象。它**不是执行结果**，而是一个**可以被暂停和恢复的执行计划**。

### 协程函数 vs 协程对象

很多人混淆"协程函数"和"协程对象"。它们是两个不同的东西：

```python
import asyncio


# 这是协程函数（coroutine function）
async def fetch_data(url: str) -> dict:
    """定义一个协程函数"""
    print(f"开始请求 {url}")
    await asyncio.sleep(1)  # 模拟网络请求
    return {"url": url, "status": 200}


# 调用协程函数，得到协程对象
coro = fetch_data("https://example.com")

print(type(coro))       # <class 'coroutine'>
print(type(fetch_data)) # <class 'function'>

# 协程对象不会自动执行！
# 如果你只是创建了它，什么都不会发生
```

**关键区别**：

| | 协程函数 | 协程对象 |
|--|--------|--------|
| 定义方式 | `async def func():` | `func()` 调用后产生 |
| 类型 | `function` | `coroutine` |
| 能否执行 | 是蓝图，不能直接执行 | 可以被 `await` 或交给事件循环 |
| 类比 | 菜谱 | 一份具体的、准备做的菜 |

### 协程对象的生命周期

```
┌─────────────────────────────────────────────────────────────────┐
│                    协程对象的生命周期                              │
│                                                                 │
│   async def foo():                                              │
│       ...                                                       │
│                                                                 │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐              │
│   │  创建     │ ──→ │  等待调度  │ ──→ │  执行中   │              │
│   │ foo()    │     │ (挂起)    │     │ (运行)    │              │
│   └──────────┘     └──────────┘     └──────────┘              │
│   coro = foo()     还没有被           被 await 或              │
│                    await              create_task               │
│                                                                 │
│   ⚠️ 如果协程对象创建后从未被 await，Python 会发出警告：          │
│   RuntimeWarning: coroutine 'foo' was never awaited             │
└─────────────────────────────────────────────────────────────────┘
```

### 普通函数调用 vs 协程函数调用

```python
import asyncio


# 普通函数：调用即执行，返回的是结果
def sync_add(a, b):
    return a + b


result = sync_add(1, 2)
print(result)  # 3 ← 立即得到结果


# 协程函数：调用得到的是协程对象，不是结果
async def async_add(a, b):
    await asyncio.sleep(0.1)
    return a + b


coro = async_add(1, 2)
print(coro)  # <coroutine object async_add at 0x...> ← 不是 3！

# 要得到结果，必须 await
result = asyncio.run(async_add(1, 2))
print(result)  # 3
```

**核心要点**：`async def` 函数调用时，函数体内的代码**一行都不会执行**。它只是创建了一个协程对象，把函数的参数和执行位置保存起来，等待被调度。

### 验证：协程对象创建时不执行

```python
import asyncio


async def demonstrate():
    print("这条消息说明协程开始执行了")
    await asyncio.sleep(1)
    return 42


# 方式 1：直接创建协程对象
coro = demonstrate()
print("协程对象已创建，但函数体还没执行")
print(f"协程对象类型: {type(coro)}")
print(f"协程对象: {coro}")

# 方式 2：await 协程对象，此时才会执行
result = asyncio.run(demonstrate())
print(f"执行结果: {result}")
```

输出：

```
协程对象已创建，但函数体还没执行
协程对象类型: <class 'coroutine'>
协程对象: <coroutine object demonstrate at 0x...>
这条消息说明协程开始执行了
执行结果: 42
```

注意"这条消息说明协程开始执行了"只打印了一次——在 `asyncio.run()` 调用时，而不是在 `demonstrate()` 调用时。

---

## 4.3 Task 是什么

Task 是对协程对象的**包装**。它把协程提交给事件循环，让事件循环来调度执行。

### 创建 Task

```python
import asyncio


async def fetch_user(user_id: int) -> dict:
    """模拟获取用户信息"""
    await asyncio.sleep(1)
    return {"id": user_id, "name": f"用户{user_id}"}


async def fetch_order(order_id: int) -> dict:
    """模拟获取订单信息"""
    await asyncio.sleep(0.5)
    return {"order_id": order_id, "amount": 99.9}


async def main():
    # 创建 Task：把协程提交给事件循环
    user_task = asyncio.create_task(fetch_user(1))
    order_task = asyncio.create_task(fetch_order(100))

    # 此时两个任务已经开始并发执行了！
    # 我们可以做其他事情...

    # 等待结果
    user = await user_task
    order = await order_task

    print(f"用户: {user}")
    print(f"订单: {order}")


asyncio.run(main())
```

**`create_task` 做了什么？**

```
┌─────────────────────────────────────────────────────────────────┐
│                  asyncio.create_task() 的作用                    │
│                                                                 │
│   协程对象 (Coroutine)          Task                             │
│   ┌──────────────┐             ┌──────────────────┐            │
│   │ fetch_user(1) │ ─────────→ │ Task 对象          │            │
│   │              │  包装        │                  │            │
│   │ 函数体未执行   │             │ · 包装了协程       │            │
│   └──────────────┘             │ · 已提交给事件循环  │            │
│                                │ · 会被调度执行     │            │
│                                │ · 是 Future 子类   │            │
│                                └──────────────────┘            │
│                                                                 │
│   create_task 不是"计划以后做"，而是"现在就安排做"。               │
│   调用 create_task 后，协程会在事件循环的下一个迭代中开始执行。      │
└─────────────────────────────────────────────────────────────────┘
```

### Task vs 直接 await 协程

```python
import asyncio
import time


async def slow_operation(name: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return f"{name} 完成"


async def without_task():
    """不用 Task：顺序执行，总耗时 = 1 + 2 = 3 秒"""
    start = time.perf_counter()

    result1 = await slow_operation("A", 1)
    result2 = await slow_operation("B", 2)

    elapsed = time.perf_counter() - start
    print(f"[不用 Task] 结果: {result1}, {result2}, 耗时: {elapsed:.1f}s")


async def with_task():
    """用 Task：并发执行，总耗时 = max(1, 2) = 2 秒"""
    start = time.perf_counter()

    # 创建 Task，两个任务立即开始并发
    task1 = asyncio.create_task(slow_operation("A", 1))
    task2 = asyncio.create_task(slow_operation("B", 2))

    # 等待两个任务完成
    result1 = await task1
    result2 = await task2

    elapsed = time.perf_counter() - start
    print(f"[用 Task]    结果: {result1}, {result2}, 耗时: {elapsed:.1f}s")


async def main():
    await without_task()
    await with_task()


asyncio.run(main())
```

输出：

```
[不用 Task] 结果: A 完成, B 完成, 耗时: 3.0s
[用 Task]    结果: A 完成, B 完成, 耗时: 2.0s
```

这就是 Task 的价值：**让多个协程并发执行，而不是串行等待**。

---

## 4.4 为什么需要 Task

你可能会问：既然可以直接 `await` 协程对象，为什么还要绕一圈创建 Task？

### 直接 await 的问题

```python
async def main():
    # 问题：这样写是顺序执行的！
    result1 = await fetch_user(1)       # 等 1 秒
    result2 = await fetch_user(2)       # 再等 1 秒
    result3 = await fetch_user(3)       # 再等 1 秒
    # 总耗时：3 秒

    # 正确做法：用 create_task 并发执行
    task1 = asyncio.create_task(fetch_user(1))
    task2 = asyncio.create_task(fetch_user(2))
    task3 = asyncio.create_task(fetch_user(3))

    result1 = await task1   # 三个任务已经并发执行
    result2 = await task2   # 这里可能已经完成了
    result3 = await task3   # 这里也可能已经完成了
    # 总耗时：约 1 秒
```

### Task 的核心价值

```
┌─────────────────────────────────────────────────────────────────┐
│                    为什么需要 Task？                              │
│                                                                 │
│  1. 并发执行                                                     │
│     · 直接 await 协程 → 顺序执行                                 │
│     · create_task 创建 Task → 并发执行                           │
│                                                                 │
│  2. 立即调度                                                     │
│     · create_task 调用后，协程在下一个事件循环迭代中开始执行        │
│     · 不需要等到 await 时才开始                                   │
│                                                                 │
│  3. 可管理                                                       │
│     · Task 有状态（pending / running / done / cancelled）        │
│     · 可以取消、查询结果、处理异常                                │
│     · 可以被多个地方 await                                        │
│                                                                 │
│  4. 组合使用                                                     │
│     · asyncio.gather() 接收多个 Task 或协程                      │
│     · asyncio.wait() 接收多个 Task                               │
│     · asyncio.as_completed() 接收多个 Task                       │
└─────────────────────────────────────────────────────────────────┘
```

### 什么时候需要 create_task，什么时候直接 await

| 场景 | 方式 | 原因 |
|------|------|------|
| 顺序依赖：B 需要 A 的结果 | `result = await coro` | 必须等 A 完成才能开始 B |
| 独立任务：A 和 B 互不依赖 | `task = create_task(coro)` | 可以并发执行 |
| 后台任务：不需要等待结果 | `task = create_task(coro)` | 启动后让事件循环自行调度 |
| 需要取消的任务 | `task = create_task(coro)` | Task 支持 `cancel()` |

---

## 4.5 Task 有状态

<div data-component="TaskStateVisualizer"></div>

Task 对象有四种状态，理解状态转换是正确使用 Task 的基础。

### 四种状态

```
┌─────────────────────────────────────────────────────────────────┐
│                      Task 的四种状态                              │
│                                                                 │
│  ┌──────────┐                                                  │
│  │ Pending   │  创建后、开始执行前的状态                          │
│  │          │  · 协程已提交给事件循环                             │
│  │          │  · 等待被调度                                      │
│  └────┬─────┘                                                  │
│       │                                                        │
│       │ 事件循环调度执行                                         │
│       ▼                                                        │
│  ┌──────────┐                                                  │
│  │ Running   │  正在执行中                                       │
│  │          │  · 协程正在运行                                    │
│  │          │  · 遇到 await 可能暂停，但状态仍是 Running          │
│  └────┬─────┘                                                  │
│       │                                                        │
│       │ 执行完成 或 遇到异常                                     │
│       ▼                                                        │
│  ┌──────────┐                                                  │
│  │ Done      │  已完成                                          │
│  │          │  · 正常返回结果                                    │
│  │          │  · 或抛出了异常                                    │
│  │          │  · 结果/异常被保存在 Task 中                        │
│  └──────────┘                                                  │
│                                                                 │
│  ┌──────────┐                                                  │
│  │ Cancelled │  被取消                                          │
│  │          │  · 调用了 task.cancel()                           │
│  │          │  · 下次遇到 await 时抛出 CancelledError           │
│  └──────────┘                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 状态转换图

```
                         create_task()
                              │
                              ▼
                        ┌───────────┐
                        │  Pending   │
                        └─────┬─────┘
                              │
                    事件循环开始执行
                              │
                              ▼
                        ┌───────────┐
              ┌────────→│  Running   │←────────┐
              │         └─────┬─────┘         │
              │               │               │
         遇到 await       正常返回         遇到异常
         (暂时让出)        │               │
              │            ▼               ▼
              │     ┌───────────┐    ┌───────────┐
              │     │   Done     │    │   Done     │
              └─────┤(有结果)    │    │(有异常)    │
                    └───────────┘    └───────────┘


         任何时候调用 cancel()
                  │
                  ▼
            ┌───────────┐
            │ Cancelled  │
            └───────────┘
```

### 状态示例

```python
import asyncio


async def example_task():
    print("Task 开始执行")
    await asyncio.sleep(1)
    print("Task 执行完成")
    return "结果"


async def main():
    # 创建 Task 后，它立即进入 Pending 状态（实际上很快变为 Running）
    task = asyncio.create_task(example_task())
    print(f"刚创建: {task.done()}")  # False

    # 在 main 协程中做其他事情
    await asyncio.sleep(0.5)
    print(f"执行中: {task.done()}")  # False（还在 sleep 中）

    # 等待 Task 完成
    result = await task
    print(f"完成后: {task.done()}")  # True
    print(f"结果: {result}")


asyncio.run(main())
```

输出：

```
Task 开始执行
刚创建: False
执行中: False
Task 执行完成
完成后: True
结果: 结果
```
</div>

---

## 4.6 查看 Task 状态

<div data-component="TaskStateVisualizer"></div>

Task 对象提供了几个方法来查询它的状态和结果。

### done() — 是否完成

```python
import asyncio


async def slow():
    await asyncio.sleep(2)
    return "完成"


async def main():
    task = asyncio.create_task(slow())

    # done() 返回 True 当且仅当 Task 已完成（成功或失败）
    print(f"1秒前: {task.done()}")  # False

    await asyncio.sleep(1)
    print(f"1秒后: {task.done()}")  # False（还在执行）

    await asyncio.sleep(1.5)
    print(f"2.5秒后: {task.done()}")  # True


asyncio.run(main())
```

### result() — 获取结果

```python
import asyncio


async def compute():
    await asyncio.sleep(1)
    return 42


async def main():
    task = asyncio.create_task(compute())

    # 在 Task 完成之前调用 result() 会抛出异常
    try:
        task.result()  # ❌ InvalidStateError
    except asyncio.InvalidStateError:
        print("Task 还没完成，不能获取结果")

    await task

    # Task 完成后可以安全获取结果
    print(f"结果: {task.result()}")  # ✅ 42


asyncio.run(main())
```

### exception() — 获取异常

```python
import asyncio


async def failing():
    await asyncio.sleep(1)
    raise ValueError("出错了！")


async def main():
    task = asyncio.create_task(failing())

    # 等待 Task 完成（会捕获异常，不会抛出）
    try:
        await task
    except ValueError:
        print("捕获到异常")

    # exception() 返回 Task 中的异常对象
    exc = task.exception()
    print(f"异常类型: {type(exc).__name__}")
    print(f"异常消息: {exc}")

    # 如果 Task 正常完成（没有异常），exception() 返回 None
    success_task = asyncio.create_task(asyncio.sleep(0, result="ok"))
    await success_task
    print(f"成功任务的异常: {success_task.exception()}")  # None


asyncio.run(main())
```

输出：

```
捕获到异常
异常类型: ValueError
异常消息: 出错了！
成功任务的异常: None
```

### cancel() — 取消 Task

```python
import asyncio


async def long_running():
    try:
        print("长时间任务开始")
        await asyncio.sleep(100)  # 很长的等待
        print("长时间任务完成")
    except asyncio.CancelledError:
        print("任务被取消了！")
        raise  # 重新抛出，让 Task 确认取消


async def main():
    task = asyncio.create_task(long_running())

    # 等一下，然后取消
    await asyncio.sleep(1)
    task.cancel()  # 请求取消

    try:
        await task
    except asyncio.CancelledError:
        print("确认任务已被取消")

    print(f"任务状态: done={task.done()}, cancelled={task.cancelled()}")


asyncio.run(main())
```

输出：

```
长时间任务开始
任务被取消了！
确认任务已被取消
任务状态: done=True, cancelled=True
```

### 状态查询方法总结

| 方法 | 返回值 | 何时使用 |
|------|--------|---------|
| `task.done()` | `bool` | 检查 Task 是否已完成（成功或失败） |
| `task.result()` | 任意值 | 获取 Task 的返回值（必须 done，否则抛异常） |
| `task.exception()` | `Exception` 或 `None` | 获取 Task 中的异常（必须 done，否则抛异常） |
| `task.cancelled()` | `bool` | 检查 Task 是否被取消 |
| `task.cancel()` | `bool` | 请求取消 Task |

</div>
---

## 4.7 Task 如何取得结果

获取 Task 结果有两种方式，它们的适用场景不同。

### 方式一：await task（推荐）

```python
import asyncio


async def fetch_data():
    await asyncio.sleep(1)
    return {"data": "hello"}


async def main():
    task = asyncio.create_task(fetch_data())

    # 方式一：await（推荐）
    result = await task
    print(f"结果: {result}")

    # await 会自动处理异常：如果 Task 抛出异常，await 会重新抛出


asyncio.run(main())
```

`await task` 的行为：
- 如果 Task 正常完成：返回结果值
- 如果 Task 抛出异常：重新抛出该异常
- 如果 Task 被取消：抛出 `CancelledError`

### 方式二：task.result()（特殊场景）

```python
import asyncio


async def fetch_data():
    await asyncio.sleep(1)
    return {"data": "hello"}


async def main():
    task = asyncio.create_task(fetch_data())

    # 先确保 Task 完成
    await task

    # 方式二：直接调用 result()（此时 Task 已完成）
    result = task.result()
    print(f"结果: {result}")


asyncio.run(main())
```

`task.result()` 的行为：
- Task 未完成时调用：抛出 `InvalidStateError`
- Task 正常完成：返回结果值
- Task 抛出异常：重新抛出该异常
- Task 被取消：抛出 `CancelledError`

### 两种方式的对比

```
┌─────────────────────────────────────────────────────────────────┐
│              await task  vs  task.result()                      │
│                                                                 │
│  await task:                                                    │
│  ┌──────────────────────────────────────┐                      │
│  │ 1. 挂起当前协程                       │                      │
│  │ 2. 让事件循环去执行其他任务             │                      │
│  │ 3. Task 完成后，当前协程恢复执行        │                      │
│  │ 4. 返回结果或抛出异常                  │                      │
│  └──────────────────────────────────────┘                      │
│  ✓ 非阻塞，让出控制权                                           │
│  ✓ 可以在 Task 未完成时使用                                      │
│  ✓ 推荐方式                                                     │
│                                                                 │
│  task.result():                                                 │
│  ┌──────────────────────────────────────┐                      │
│  │ 1. 直接读取保存的结果                  │                      │
│  │ 2. 如果 Task 没完成 → InvalidStateError│                     │
│  │ 3. 如果有异常 → 抛出异常               │                      │
│  └──────────────────────────────────────┘                      │
│  ✗ 不会等待 Task 完成                                           │
│  ✗ 必须确保 Task 已完成才能调用                                   │
│  ✓ 适合在回调函数中使用                                          │
│  ✓ 适合多次读取同一个 Task 的结果                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 实际选择指南

```python
import asyncio


async def process_data(task: asyncio.Task):
    """处理 Task 结果的几种模式"""

    # 模式 1：直接 await（最常用）
    result = await task

    # 模式 2：先做其他事，再 await
    await asyncio.sleep(1)
    result = await task  # 此时 Task 可能已完成，不会阻塞

    # 模式 3：用 result() 多次读取（适合已完成的 Task）
    await task  # 确保完成
    r1 = task.result()
    r2 = task.result()  # 可以反复读取，结果被缓存

    # 模式 4：在回调中使用 result()
    def on_done(t: asyncio.Task):
        try:
            res = t.result()
            print(f"回调中获取结果: {res}")
        except Exception as e:
            print(f"回调中获取异常: {e}")

    task.add_done_callback(on_done)
```

---

## 4.8 Future 是什么

<div data-component="FutureResultDemo"></div>

Future 是 asyncio 中最基础的"结果占位符"。它代表一个**将来会完成的操作的结果**。

### Future 的概念

```
┌─────────────────────────────────────────────────────────────────┐
│                      Future 的本质                               │
│                                                                 │
│   Future 是一个"空盒子"，最终会被放入一个值（结果或异常）。         │
│                                                                 │
│   创建时：                                                       │
│   ┌──────────┐                                                  │
│   │  Future   │  空的，还没有结果                                │
│   │  状态：    │                                                 │
│   │  Pending  │                                                 │
│   └──────────┘                                                  │
│                                                                 │
│   设置结果后：                                                    │
│   ┌──────────┐                                                  │
│   │  Future   │  包含结果                                        │
│   │  状态：    │                                                 │
│   │  Done     │  → result = 42                                  │
│   └──────────┘                                                  │
│                                                                 │
│   任何 await 这个 Future 的协程都会在它完成时收到结果。             │
└─────────────────────────────────────────────────────────────────┘
```

### Future vs Coroutine

| | Coroutine | Future |
|--|----------|--------|
| 创建方式 | `async def` 函数调用 | `loop.create_future()` |
| 代表什么 | 一段可以执行的代码 | 一个结果的占位符 |
| 如何完成 | 被执行（`await` 或 Task 调度） | 手动设置结果（`set_result`） |
| 谁控制完成 | 协程自己的逻辑 | 外部代码设置结果 |
| 典型用途 | 定义异步操作 | 连接回调和 await |

</div>
---

## 4.9 Future 基本示例

### 创建和使用 Future

```python
import asyncio


async def set_future_result(future: asyncio.Future, value, delay: float):
    """模拟一个异步操作，延迟后设置 Future 的结果"""
    await asyncio.sleep(delay)
    future.set_result(value)
    print(f"Future 结果已设置: {value}")


async def main():
    # 获取事件循环
    loop = asyncio.get_event_loop()

    # 创建一个 Future（结果占位符）
    future = loop.create_future()
    print(f"Future 创建完成，状态: done={future.done()}")  # False

    # 启动一个任务来设置 Future 的结果
    asyncio.create_task(set_future_result(future, "Hello, Future!", 2))

    # 等待 Future 完成（这里会阻塞，直到 set_result 被调用）
    result = await future
    print(f"收到结果: {result}")
    print(f"Future 状态: done={future.done()}")  # True


asyncio.run(main())
```

输出：

```
Future 创建完成，状态: done=False
Future 结果已设置: Hello, Future!
收到结果: Hello, Future!
Future 状态: done=True
```

### set_result 和 set_exception

```python
import asyncio


async def main():
    loop = asyncio.get_event_loop()

    # 示例 1：设置正常结果
    future1 = loop.create_future()
    future1.set_result(42)
    print(f"结果: {future1.result()}")  # 42

    # 示例 2：设置异常
    future2 = loop.create_future()
    future2.set_exception(ValueError("计算出错"))

    try:
        future2.result()  # 会抛出异常
    except ValueError as e:
        print(f"异常: {e}")  # 计算出错

    # 示例 3：await Future 获取结果
    future3 = loop.create_future()

    async def resolve():
        await asyncio.sleep(1)
        future3.set_result("异步结果")

    asyncio.create_task(resolve())
    result = await future3
    print(f"await 结果: {result}")


asyncio.run(main())
```

### Future 不能重复设置结果

```python
import asyncio


async def main():
    loop = asyncio.get_event_loop()
    future = loop.create_future()

    future.set_result("第一次设置")

    try:
        future.set_result("第二次设置")  # ❌ 抛出异常
    except asyncio.InvalidStateError:
        print("Future 已经有结果了，不能重复设置")

    print(f"结果仍然是: {future.result()}")  # 第一次设置


asyncio.run(main())
```

**关键规则**：一个 Future 只能设置一次结果（`set_result` 或 `set_exception`），之后不能再修改。这保证了结果的确定性——任何 await 这个 Future 的协程都得到相同的结果。

---

## 4.10 Future 为什么有用

你可能会觉得 Future 很奇怪——为什么要手动设置结果？直接从协程返回不好吗？

### 连接回调与 await

Future 最大的价值是**连接两种世界**：回调风格的代码和 `await` 风格的代码。

```python
import asyncio


def legacy_api_call(callback):
    """一个使用回调风格的老 API"""
    import threading

    def worker():
        import time
        time.sleep(1)  # 模拟耗时操作
        callback("来自老 API 的结果")

    thread = threading.Thread(target=worker)
    thread.start()


async def wrap_legacy_api() -> str:
    """用 Future 把回调风格的 API 包装成 async/await 风格"""
    loop = asyncio.get_event_loop()
    future = loop.create_future()

    def on_complete(result):
        # 回调在另一个线程中被调用，需要线程安全地设置结果
        loop.call_soon_threadsafe(future.set_result, result)

    legacy_api_call(on_complete)

    # 现在可以用 await 等待结果了
    return await future


async def main():
    result = await wrap_legacy_api()
    print(f"结果: {result}")


asyncio.run(main())
```

### Future 的核心价值

```
┌─────────────────────────────────────────────────────────────────┐
│                  Future 为什么有用？                              │
│                                                                 │
│  1. 结果占位符                                                   │
│     · 先创建 Future，稍后设置结果                                 │
│     · 代码可以在结果就绪前就开始等待                               │
│                                                                 │
│  2. 连接回调与 await                                             │
│     · 回调函数调用 future.set_result()                           │
│     · 其他代码 await future 获取结果                              │
│     · 完美桥接两种编程风格                                        │
│                                                                 │
│  3. 跨线程/跨进程通信                                            │
│     · 一个线程设置结果，另一个线程 await 结果                      │
│     · 配合 loop.call_soon_threadsafe() 使用                      │
│                                                                 │
│  4. 底层构建块                                                   │
│     · Task 内部就是用 Future 实现的                               │
│     · asyncio.gather() 内部也使用 Future                         │
│     · 很多高级 API 的底层都是 Future                              │
└─────────────────────────────────────────────────────────────────┘
```

### 实际应用：封装同步 API

```python
import asyncio
import requests


async def async_get(url: str) -> str:
    """把同步的 requests.get 包装成异步函数"""
    loop = asyncio.get_event_loop()
    future = loop.create_future()

    def do_request():
        try:
            response = requests.get(url)
            loop.call_soon_threadsafe(future.set_result, response.text)
        except Exception as e:
            loop.call_soon_threadsafe(future.set_exception, e)

    # 在线程池中执行同步请求
    loop.run_in_executor(None, do_request)

    return await future


async def main():
    # 现在可以异步使用了
    html = await async_get("https://example.com")
    print(f"获取到 {len(html)} 字节的数据")


asyncio.run(main())
```

---

## 4.11 Task 和 Future 的关系

<div data-component="CoroutineTaskFutureDiagram"></div>

Task 是 Future 的**子类**。这意味着 Task 继承了 Future 的所有能力，同时增加了协程调度的功能。

### 继承关系

```python
import asyncio
import inspect


# Task 是 Future 的子类
print(issubclass(asyncio.Task, asyncio.Future))  # True

# 验证
async def dummy():
    pass

task = asyncio.create_task(dummy())
print(isinstance(task, asyncio.Future))  # True
print(isinstance(task, asyncio.Task))    # True

# Task 继承了 Future 的所有方法
print(hasattr(task, 'set_result'))    # True（继承自 Future）
print(hasattr(task, 'set_exception')) # True（继承自 Future）
print(hasattr(task, 'done'))          # True（继承自 Future）
print(hasattr(task, 'result'))        # True（继承自 Future）
print(hasattr(task, 'cancel'))        # True（继承自 Future）
print(hasattr(task, 'cancelled'))     # True（继承自 Future）
```

### Task 比 Future 多了什么

```python
import asyncio


async def example():
    await asyncio.sleep(1)
    return 42


async def main():
    # Future：需要手动设置结果
    loop = asyncio.get_event_loop()
    future = loop.create_future()

    # 你需要手动调用 set_result，否则 await 永远不会返回
    # future.set_result(42)

    # Task：自动管理协程执行和结果
    task = asyncio.create_task(example())

    # Task 会自动执行协程，完成后自动设置结果
    result = await task  # 自动获取协程的返回值
    print(f"结果: {result}")

    # Task 还有额外的属性
    print(f"Task 的协程: {task.get_coro()}")
    print(f"Task 的名称: {task.get_name()}")


asyncio.run(main())
```

### Task 和 Future 的对比

```
┌─────────────────────────────────────────────────────────────────┐
│               Task vs Future 对比                                │
│                                                                 │
│   Future（基础层）            Task（应用层）                      │
│   ┌──────────────────┐       ┌──────────────────┐              │
│   │ 结果占位符         │       │ 协程调度器 + 结果  │              │
│   │                  │       │ 占位符            │              │
│   │ · set_result()   │       │                  │              │
│   │ · set_exception()│       │ · 自动执行协程     │              │
│   │ · done()         │ 继承   │ · 自动设置结果     │              │
│   │ · result()       │ ←──── │ · cancel()        │              │
│   │ · exception()    │       │ · get_coro()      │              │
│   │ · add_callback() │       │ · get_name()      │              │
│   │                  │       │ · add_callback()   │              │
│   └──────────────────┘       └──────────────────┘              │
│                                                                 │
│   Future = 空盒子，你来填                                        │
│   Task = 你给我协程，我自动执行并填入结果                          │
└─────────────────────────────────────────────────────────────────┘
```
</div>

---

## 4.12 三者关系图

<div data-component="CoroutineTaskFutureDiagram"></div>

### 完整关系图

```
┌─────────────────────────────────────────────────────────────────────┐
│                Coroutine、Task、Future 三者关系                       │
│                                                                     │
│                                                                     │
│   ┌─────────────────┐                                               │
│   │  async def func():│                                              │
│   │      ...         │                                               │
│   │      return val  │                                               │
│   └────────┬────────┘                                               │
│            │                                                        │
│            │ 调用                                                    │
│            ▼                                                        │
│   ┌─────────────────┐     asyncio.create_task()    ┌──────────────┐│
│   │  Coroutine       │ ─────────────────────────→  │    Task       ││
│   │  (协程对象)       │                              │              ││
│   │                  │     自动包装成 Task            │  · 继承 Future││
│   │  · 函数体未执行   │                              │  · 调度协程   ││
│   │  · 需要被调度     │                              │  · 管理状态   ││
│   │  · 不能重复 await │                              │  · 存储结果   ││
│   └─────────────────┘                              └──────┬───────┘│
│                                                           │        │
│                                            isinstance()   │        │
│                                                           ▼        │
│                                                  ┌──────────────┐  │
│                                                  │   Future      │  │
│                                                  │              │  │
│                                                  │  · 结果占位符  │  │
│                                                  │  · set_result │  │
│                                                  │  · done()     │  │
│                                                  │  · result()   │  │
│                                                  └──────────────┘  │
│                                                                     │
│   层级关系：                                                         │
│   ┌──────────────────────────────────────────────────────┐         │
│   │  Coroutine  ──包装──→  Task  ──是子类──→  Future      │         │
│   │  (代码蓝图)            (调度器)          (结果占位符)   │         │
│   └──────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
```

### 一句话总结

- **Coroutine**：一段可以暂停和恢复的代码，本身不执行
- **Task**：把 Coroutine 包装起来，交给事件循环调度执行
- **Future**：一个结果的占位符，Task 是它的子类

**Task = Coroutine 的调度器 + Future 的结果管理**

</div>
---

## 4.13 任务抛出异常时

<div data-component="TaskExceptionDemo"></div>

当 Task 中的协程抛出异常时，异常会被捕获并保存在 Task 对象中，而不是立即传播。

### 异常不会立即传播

```python
import asyncio


async def failing_task():
    print("任务开始")
    await asyncio.sleep(1)
    raise ValueError("任务执行出错！")
    # 异常被 Task 捕获，不会立即传播


async def main():
    task = asyncio.create_task(failing_task())
    print("Task 已创建，异常还没发生")

    # 在 Task 完成前，我们可以做其他事情
    await asyncio.sleep(0.5)
    print("Task 还在执行中...")

    # await Task 时，异常才会被重新抛出
    try:
        await task
    except ValueError as e:
        print(f"捕获到异常: {e}")

    print(f"Task 完成: {task.done()}")


asyncio.run(main())
```

输出：

```
Task 已创建，异常还没发生
任务开始
Task 还在执行中...
捕获到异常: 任务执行出错！
Task 完成: True
```

### 未被 await 的异常

```python
import asyncio


async def failing_task():
    await asyncio.sleep(1)
    raise ValueError("出错了")


async def main():
    task = asyncio.create_task(failing_task())

    # 如果不 await 这个 Task，异常会被事件循环记录
    # Python 会在程序结束时打印警告
    await asyncio.sleep(2)

    # Task 已经完成了，异常被保存在 Task 中
    print(f"Task 完成: {task.done()}")

    # 可以通过 exception() 检查异常
    exc = task.exception()
    if exc:
        print(f"发现未处理的异常: {exc}")


asyncio.run(main())
```

### 多个 Task 的异常处理

```python
import asyncio


async def success():
    await asyncio.sleep(1)
    return "成功"


async def fail():
    await asyncio.sleep(0.5)
    raise RuntimeError("失败了")


async def main():
    # 使用 gather 时，一个 Task 失败会取消其他 Task
    try:
        results = await asyncio.gather(
            success(),
            fail(),
            success(),
        )
    except RuntimeError as e:
        print(f"gather 捕获到异常: {e}")

    # 使用 return_exceptions=True 可以收集所有结果（包括异常）
    results = await asyncio.gather(
        success(),
        fail(),
        success(),
        return_exceptions=True,
    )

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"任务 {i}: 异常 - {result}")
        else:
            print(f"任务 {i}: 成功 - {result}")


asyncio.run(main())
```

输出：

```
gather 捕获到异常: 失败了
任务 0: 成功 - 成功
任务 1: 异常 - 失败了
任务 2: 成功 - 成功
```
</div>

---

## 4.14 查看 Task 异常

<div data-component="TaskExceptionDemo"></div>

### task.exception() 的使用

```python
import asyncio


async def divide(a, b):
    await asyncio.sleep(0.1)
    return a / b


async def main():
    # 正常任务
    task1 = asyncio.create_task(divide(10, 2))
    await task1
    print(f"task1 异常: {task1.exception()}")  # None

    # 会抛异常的任务
    task2 = asyncio.create_task(divide(10, 0))
    try:
        await task2
    except ZeroDivisionError:
        pass  # 捕获异常，不让它传播

    # 通过 exception() 查看异常详情
    exc = task2.exception()
    print(f"task2 异常类型: {type(exc).__name__}")
    print(f"task2 异常消息: {exc}")

    # 未完成的 Task 调用 exception() 会抛出 InvalidStateError
    task3 = asyncio.create_task(divide(10, 3))
    try:
        task3.exception()  # Task 还没完成
    except asyncio.InvalidStateError:
        print("task3 还没完成，不能查看异常")

    await task3


asyncio.run(main())
```

输出：

```
task1 异常: None
task2 异常类型: ZeroDivisionError
task2 异常消息: division by zero
task3 还没完成，不能查看异常
```

### 使用 done 回调处理异常

```python
import asyncio


async def risky_operation():
    await asyncio.sleep(1)
    raise ConnectionError("网络连接失败")


def on_task_done(task: asyncio.Task):
    """Task 完成时的回调函数"""
    if task.cancelled():
        print(f"任务 {task.get_name()} 被取消")
    elif task.exception():
        print(f"任务 {task.get_name()} 失败: {task.exception()}")
    else:
        print(f"任务 {task.get_name()} 成功: {task.result()}")


async def main():
    task = asyncio.create_task(
        risky_operation(),
        name="网络请求"
    )

    # 添加完成回调
    task.add_done_callback(on_task_done)

    # 即使不 await，回调也会在 Task 完成时被调用
    await asyncio.sleep(2)


asyncio.run(main())
```

输出：

```
任务 网络请求 失败: 网络连接失败
```

### 全局异常处理

```python
import asyncio


async def main():
    # 创建一个会失败的 Task，但不 await 它
    asyncio.create_task(failing_coroutine())

    # 等待足够长的时间让 Task 完成
    await asyncio.sleep(2)


async def failing_coroutine():
    await asyncio.sleep(1)
    raise RuntimeError("未处理的异常")


# 设置全局异常处理器
def handle_exception(loop, context):
    """处理未捕获的异常"""
    exception = context.get('exception')
    message = context.get('message')
    print(f"全局异常处理器捕获到: {message}")
    if exception:
        print(f"异常类型: {type(exception).__name__}")
        print(f"异常消息: {exception}")


loop = asyncio.new_event_loop()
loop.set_exception_handler(handle_exception)
loop.run_until_complete(main())
loop.close()
```
</div>

---

## 4.15 协程对象不能重复等待

一个协程对象只能被 `await` 一次。第二次 `await` 同一个协程对象会抛出 `RuntimeError`。

### 错误示例

```python
import asyncio


async def get_data():
    await asyncio.sleep(1)
    return "数据"


async def main():
    coro = get_data()  # 创建协程对象

    # 第一次 await：正常执行
    result1 = await coro
    print(f"第一次: {result1}")

    # 第二次 await：抛出 RuntimeError
    try:
        result2 = await coro
    except RuntimeError as e:
        print(f"错误: {e}")  # cannot reuse already awaited coroutine


asyncio.run(main())
```

输出：

```
第一次: 数据
错误: cannot reuse already awaited coroutine
```

### 为什么不能重复 await

```
┌─────────────────────────────────────────────────────────────────┐
│              为什么协程对象不能重复 await？                        │
│                                                                 │
│   协程对象是"一次性的执行计划"：                                   │
│                                                                 │
│   1. 调用 async def 函数 → 创建协程对象（保存参数和执行位置）       │
│   2. await 协程对象 → 执行函数体，得到返回值                       │
│   3. 执行完成后，协程对象的状态变为"已结束"                        │
│   4. 已结束的协程对象不能重新执行                                  │
│                                                                 │
│   类比：                                                         │
│   · 协程对象像一张电影票，看完电影后票就作废了                      │
│   · 你想再看一遍？需要买一张新票（创建新的协程对象）                 │
└─────────────────────────────────────────────────────────────────┘
```

### 正确做法：每次需要时创建新的协程对象

```python
import asyncio


async def get_data(source: str):
    await asyncio.sleep(1)
    return f"来自 {source} 的数据"


async def main():
    # 正确：每次创建新的协程对象
    result1 = await get_data("数据库")
    result2 = await get_data("缓存")
    result3 = await get_data("API")

    print(result1)
    print(result2)
    print(result3)

    # 或者用 Task（Task 会缓存结果，可以多次 await）
    task = asyncio.create_task(get_data("任务"))
    r1 = await task
    r2 = await task  # ✅ Task 可以多次 await
    print(f"Task 结果: {r1}, {r2}")


asyncio.run(main())
```

---

## 4.16 Task 可以被多次 await

与协程对象不同，Task 可以被多次 `await`，因为 Task 会缓存执行结果。

### Task 多次 await

```python
import asyncio


async def compute():
    await asyncio.sleep(1)
    print("计算执行了一次")
    return 42


async def worker_a(task: asyncio.Task):
    """工作者 A 等待 Task"""
    result = await task
    print(f"工作者 A 得到结果: {result}")


async def worker_b(task: asyncio.Task):
    """工作者 B 也等待同一个 Task"""
    result = await task
    print(f"工作者 B 得到结果: {result}")


async def main():
    task = asyncio.create_task(compute())

    # 多个协程可以同时 await 同一个 Task
    await asyncio.gather(
        worker_a(task),
        worker_b(task),
    )

    # 计算只执行了一次！
    # Task 的结果被缓存，多次 await 得到相同的结果


asyncio.run(main())
```

输出：

```
计算执行了一次
工作者 A 得到结果: 42
工作者 B 得到结果: 42
```

### 为什么 Task 可以多次 await

```
┌─────────────────────────────────────────────────────────────────┐
│           协程对象 vs Task：重复 await 的区别                      │
│                                                                 │
│   协程对象：                                                     │
│   ┌─────────────────────────────────────┐                      │
│   │ await coro → 执行函数体 → 得到结果    │                      │
│   │ coro 状态变为"已结束"                 │                      │
│   │ 再次 await → RuntimeError           │                      │
│   └─────────────────────────────────────┘                      │
│   · 协程对象是"执行计划"，执行完就没了                             │
│                                                                 │
│   Task：                                                         │
│   ┌─────────────────────────────────────┐                      │
│   │ create_task(coro) → 包装协程          │                      │
│   │ Task 自动执行协程 → 结果存入 Task      │                      │
│   │ await task → 返回缓存的结果           │                      │
│   │ 再次 await task → 返回同一个缓存结果   │                      │
│   └─────────────────────────────────────┘                      │
│   · Task 是"结果容器"，结果可以被反复读取                          │
└─────────────────────────────────────────────────────────────────┘
```

### 实际应用场景

```python
import asyncio


async def fetch_config():
    """获取配置（只执行一次）"""
    await asyncio.sleep(1)  # 模拟网络请求
    return {"db_host": "localhost", "db_port": 5432}


async def service_a(config_task: asyncio.Task):
    """服务 A 需要配置"""
    config = await config_task  # 获取配置
    print(f"服务 A 使用配置: {config}")


async def service_b(config_task: asyncio.Task):
    """服务 B 也需要同一个配置"""
    config = await config_task  # 获取同一个配置
    print(f"服务 B 使用配置: {config}")


async def main():
    # 只获取一次配置，多个服务共享
    config_task = asyncio.create_task(fetch_config())

    # 两个服务并发执行，但配置只获取一次
    await asyncio.gather(
        service_a(config_task),
        service_b(config_task),
    )


asyncio.run(main())
```

输出：

```
服务 A 使用配置: {'db_host': 'localhost', 'db_port': 5432}
服务 B 使用配置: {'db_host': 'localhost', 'db_port': 5432}
```

---

## 4.17 为什么通常不手动创建 Future

在大多数情况下，你不需要手动创建 Future。asyncio 提供了更高级的 API 来替代手动 Future 管理。

### 手动创建 Future 的问题

```python
import asyncio


async def manual_future_example():
    """手动创建 Future 的示例（不推荐）"""
    loop = asyncio.get_event_loop()
    future = loop.create_future()

    # 你需要手动管理 Future 的生命周期
    async def resolve():
        await asyncio.sleep(1)
        future.set_result(42)

    asyncio.create_task(resolve())

    # 等待结果
    result = await future
    return result


async def better_approach():
    """更好的方式：直接用协程（推荐）"""
    await asyncio.sleep(1)
    return 42
```

### 什么时候需要手动 Future

| 场景 | 是否需要 Future | 替代方案 |
|------|---------------|---------|
| 包装回调风格的 API | 是 | 无直接替代 |
| 实现自定义的等待机制 | 是 | 无直接替代 |
| 跨线程/跨进程通信 | 是 | `asyncio.run_coroutine_threadsafe()` |
| 普通的异步操作 | **否** | 直接用协程 + `await` |
| 并发执行多个任务 | **否** | `asyncio.gather()` |
| 等待多个任务中的第一个 | **否** | `asyncio.wait()` / `as_completed()` |

### 常见的 Future 替代方案

```python
import asyncio


# ❌ 不推荐：手动创建 Future
async def bad_pattern():
    loop = asyncio.get_event_loop()
    future = loop.create_future()

    async def compute():
        await asyncio.sleep(1)
        future.set_result(42)

    asyncio.create_task(compute())
    return await future


# ✅ 推荐：直接用协程
async def good_pattern():
    await asyncio.sleep(1)
    return 42


# ✅ 推荐：用 Task
async def also_good():
    async def compute():
        await asyncio.sleep(1)
        return 42

    task = asyncio.create_task(compute())
    return await task


asyncio.run(bad_pattern())
asyncio.run(good_pattern())
asyncio.run(also_good())
```

### asyncio 内部使用 Future 的地方

虽然你通常不直接创建 Future，但 asyncio 内部大量使用它：

```python
import asyncio


async def main():
    # asyncio.sleep() 内部使用 Future
    await asyncio.sleep(1)

    # asyncio.gather() 内部使用 Future 来收集结果
    await asyncio.gather(
        asyncio.sleep(1),
        asyncio.sleep(2),
    )

    # loop.run_in_executor() 返回 Future
    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(None, lambda: 42)
    result = await future

    # asyncio.wait() 内部使用 Future
    tasks = [asyncio.create_task(asyncio.sleep(i)) for i in range(3)]
    done, pending = await asyncio.wait(tasks)

    print(f"完成: {len(done)}, 待处理: {len(pending)}")


asyncio.run(main())
```

---

## 4.18 常见误区

<div data-component="TaskVsCoroutineComparison"></div>

### 误区一：async def 会立即执行

```python
import asyncio


async def greet():
    print("你好！")
    return "问候完成"


# ❌ 错误理解：调用 greet() 会打印 "你好！"
coro = greet()  # 什么都不会打印！

# ✅ 正确理解：调用 greet() 只创建协程对象
# 要执行它，需要 await 或 create_task
result = asyncio.run(greet())  # 这时才会打印 "你好！"
```

### 误区二：create_task 会阻塞当前协程

```python
import asyncio


async def slow():
    await asyncio.sleep(5)
    return "完成"


async def main():
    # ❌ 错误理解：create_task 会等 slow() 完成
    task = asyncio.create_task(slow())

    # ✅ 正确理解：create_task 立即返回，slow() 在后台执行
    print("create_task 已返回，slow() 还在执行中")

    # 如果需要结果，再 await
    result = await task


asyncio.run(main())
```

### 误区三：await 协程和 await Task 一样

```python
import asyncio


async def compute():
    await asyncio.sleep(1)
    return 42


async def main():
    # 协程对象：每次 await 都是新的执行
    coro1 = compute()
    r1 = await coro1  # 执行一次
    # r1_again = await coro1  # ❌ RuntimeError！

    # Task：执行一次，结果被缓存
    task = asyncio.create_task(compute())
    r2 = await task      # 执行并缓存
    r2_again = await task # ✅ 返回缓存的结果


asyncio.run(main())
```

### 误区四：Task 完成后还能修改结果

```python
import asyncio


async def main():
    async def get_value():
        return 42

    task = asyncio.create_task(get_value())
    await task

    # ❌ 错误：Task 完成后结果不可变
    # task.result() 返回 42，你不能改成 100

    # ✅ 正确：如果需要不同的结果，创建新的 Task
    new_task = asyncio.create_task(get_value())
    await new_task


asyncio.run(main())
```

### 误区五：不用 create_task 也能并发

```python
import asyncio


async def fetch(n):
    await asyncio.sleep(1)
    return n


async def wrong():
    """❌ 这不是并发，是顺序执行"""
    r1 = await fetch(1)  # 等 1 秒
    r2 = await fetch(2)  # 再等 1 秒
    r3 = await fetch(3)  # 再等 1 秒
    # 总耗时：3 秒


async def correct():
    """✅ 用 create_task 实现并发"""
    t1 = asyncio.create_task(fetch(1))
    t2 = asyncio.create_task(fetch(2))
    t3 = asyncio.create_task(fetch(3))

    r1 = await t1
    r2 = await t2
    r3 = await t3
    # 总耗时：约 1 秒


async def also_correct():
    """✅ 用 gather 实现并发"""
    r1, r2, r3 = await asyncio.gather(
        fetch(1),
        fetch(2),
        fetch(3),
    )
    # 总耗时：约 1 秒


asyncio.run(wrong())
asyncio.run(correct())
asyncio.run(also_correct())
```

### 误区六：Task 的异常会丢失

```python
import asyncio


async def main():
    async def fail():
        raise ValueError("出错了")

    # Task 中的异常不会丢失，但需要被处理
    task = asyncio.create_task(fail())

    # 方式 1：await 会抛出异常
    try:
        await task
    except ValueError:
        print("方式 1：捕获到异常")

    # 方式 2：exception() 可以检查异常
    task2 = asyncio.create_task(fail())
    await asyncio.sleep(1)  # 等待 task2 完成
    if task2.exception():
        print(f"方式 2：发现异常 {task2.exception()}")

    # 方式 3：done 回调可以处理异常
    def on_done(t):
        if t.exception():
            print(f"方式 3：回调捕获异常 {t.exception()}")

    task3 = asyncio.create_task(fail())
    task3.add_done_callback(on_done)
    await asyncio.sleep(1)


asyncio.run(main())
```

### 误区七：所有异步操作都需要 create_task

```python
import asyncio


async def fetch_user(user_id):
    await asyncio.sleep(1)
    return {"id": user_id, "name": "Alice"}


async def fetch_orders(user_id):
    await asyncio.sleep(1)
    return [{"order_id": 1}, {"order_id": 2}]


async def main():
    # ❌ 不必要的 create_task：如果 B 依赖 A 的结果
    user_task = asyncio.create_task(fetch_user(1))
    user = await user_task
    # 这里必须等 user_task 完成，因为 fetch_orders 需要 user_id
    orders = await fetch_orders(user["id"])

    # ✅ 正确做法：有依赖关系时直接 await
    user = await fetch_user(1)
    orders = await fetch_orders(user["id"])

    # ✅ 无依赖关系时才用 create_task
    user_task = asyncio.create_task(fetch_user(1))
    config_task = asyncio.create_task(fetch_config())  # 假设这个函数存在
    user = await user_task
    config = await config_task


asyncio.run(main())
```
</div>

---

## 本章小结

本章深入探讨了 Python asyncio 中三个核心概念的关系：Coroutine（协程对象）、Task 和 Future。

### 三者的本质

| 概念 | 本质 | 创建方式 | 特点 |
|------|------|---------|------|
| Coroutine | 可暂停/恢复的执行计划 | `async def` 函数调用 | 不自动执行，不能重复 await |
| Task | 协程的调度器 + 结果容器 | `asyncio.create_task()` | 自动执行协程，结果可缓存 |
| Future | 结果的占位符 | `loop.create_future()` | 需要手动设置结果 |

### 关键关系

```
Coroutine ──包装──→ Task ──是子类──→ Future
(代码蓝图)         (调度器)          (结果占位符)
```

### 核心要点

1. **协程对象不会自动执行**：`async def` 函数调用只是创建协程对象，函数体内的代码不会执行
2. **Task 是协程的包装器**：`create_task()` 把协程提交给事件循环，使其被调度执行
3. **Task 是 Future 的子类**：Task 继承了 Future 的所有能力，增加了协程调度功能
4. **协程对象不能重复 await**：执行一次后就"作废"了，需要再次调用函数创建新的
5. **Task 可以多次 await**：结果被缓存在 Task 中，可以反复读取
6. **Future 是底层构建块**：通常不需要直接创建，asyncio 提供了更高级的 API
7. **异常会被保存在 Task 中**：不会立即传播，需要 await 或 exception() 来处理

### 使用建议

| 场景 | 推荐方式 |
|------|---------|
| 需要并发执行多个独立任务 | `asyncio.create_task()` 或 `asyncio.gather()` |
| 有依赖关系的顺序执行 | 直接 `await` 协程 |
| 包装回调风格的 API | 手动创建 `Future` |
| 后台执行不需要等待 | `asyncio.create_task()` |
| 等待多个任务中的第一个 | `asyncio.wait()` 或 `asyncio.as_completed()` |

---

## 思考题

### 1. Coroutine 和 Task 最大的区别是什么？

Coroutine 是"执行计划"——它描述了要做什么，但本身不会执行。调用 `async def` 函数只是创建了一个协程对象，保存了参数和执行位置。

Task 是"执行器 + 结果容器"——它把协程包装起来，提交给事件循环调度执行，并在执行完成后保存结果。

关键区别：
- Coroutine 需要外部 `await` 或 Task 来驱动执行
- Task 被事件循环自动调度，不需要额外的 `await` 来启动
- Coroutine 不能重复 `await`，Task 可以（结果被缓存）

### 2. 为什么 Task 是 Future 的子类？

因为 Task 本质上就是一个"异步操作的结果占位符"。Future 定义了结果占位符的基本接口（`set_result`、`result`、`done` 等），Task 在此基础上增加了协程调度的功能。

这种设计的好处：
- 任何接受 Future 的代码也接受 Task（里氏替换原则）
- Task 可以用在所有需要 Future 的地方
- 代码复用：Task 不需要重新实现结果管理的逻辑

### 3. 什么时候应该手动创建 Future？

手动创建 Future 的典型场景是**包装回调风格的 API**。当一个库使用回调函数来通知操作完成时，你可以创建一个 Future，在回调中调用 `set_result()`，然后用 `await` 等待结果。

其他场景通常有更高级的替代方案：
- 普通异步操作 → 直接用协程
- 并发执行 → `asyncio.gather()`
- 等待多个任务 → `asyncio.wait()`

### 4. 为什么协程对象不能重复 await？

协程对象是"一次性的执行计划"。当你 `await` 一个协程对象时，Python 会执行函数体中的代码，得到返回值。执行完成后，协程对象的状态变为"已结束"，函数体中的代码已经执行过了，不可能再执行一遍。

这就像一张电影票——看完电影后票就作废了。如果你想再看一遍，需要买一张新票（创建新的协程对象）。

Task 不同：Task 是"结果容器"，它会缓存协程的执行结果。多次 `await` 同一个 Task，得到的都是同一个缓存的结果，而不是重新执行协程。

### 5. 如果不 await 一个 Task，会发生什么？

Task 仍然会被事件循环调度执行，协程中的代码仍然会运行。但你无法获得执行结果，也无法处理可能抛出的异常。

如果 Task 中的协程抛出了异常，而你没有 `await` 它，Python 会在 Task 被垃圾回收时打印一条警告：

```
Task exception was never retrieved
```

最佳实践：始终确保 Task 的结果或异常被处理。可以使用 `await`、`add_done_callback()` 或将 Task 添加到一个集合中保持引用。

### 6. 如何在多个协程之间共享同一个 Task 的结果？

直接让多个协程 `await` 同一个 Task 对象即可。Task 的协程只会执行一次，结果被缓存，所有 `await` 它的协程都会得到相同的结果。

```python
import asyncio


async def compute():
    await asyncio.sleep(1)
    print("计算执行了一次")
    return 42


async def user_a(task):
    result = await task
    print(f"用户 A: {result}")


async def user_b(task):
    result = await task
    print(f"用户 B: {result}")


async def main():
    task = asyncio.create_task(compute())
    await asyncio.gather(user_a(task), user_b(task))
    # "计算执行了一次" 只打印一次


asyncio.run(main())
```

### 7. asyncio.create_task() 和 asyncio.ensure_future() 有什么区别？

`asyncio.create_task()`（Python 3.7+）是推荐的方式，它只接受协程对象，返回 Task 对象。

`asyncio.ensure_future()` 是更老的 API，它可以接受协程对象、Future 或 Task：
- 如果传入协程对象 → 调用 `create_task()` 包装成 Task
- 如果传入 Future/Task → 直接返回

在大多数情况下，应该使用 `create_task()`。`ensure_future()` 主要用于需要兼容不同输入类型的底层代码。

```python
import asyncio


async def example():
    return 42


async def main():
    # 推荐：create_task
    task = asyncio.create_task(example())

    # 也可以：ensure_future（但没必要）
    future = asyncio.ensure_future(example())

    # 两者结果相同
    print(await task)    # 42
    print(await future)  # 42


asyncio.run(main())
```

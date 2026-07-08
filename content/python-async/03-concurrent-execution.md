---
title: "真正让多个协程并发执行"
description: "掌握 asyncio.create_task() 和 asyncio.gather() 的用法，理解协程并发调度的本质机制，区分顺序执行与真正并发的差异"
updated: "2026-07-07"
---

# 真正让多个协程并发执行

> **学习目标**：
> - 理解顺序 await 为什么不能并发，总耗时等于各任务延迟之和
> - 掌握 `asyncio.create_task()` 将协程包装为 Task 并实现并发
> - 理解并发执行的总耗时为什么只取决于最慢的任务
> - 区分"创建 Task"和"Task 开始执行"这两个不同的时刻
> - 掌握 `await asyncio.sleep(0)` 主动让出控制权的机制
> - 熟练使用 `asyncio.gather()` 批量并发并收集结果
> - 理解 gather() 返回结果的顺序与任务完成顺序的关系
> - 区分 `create_task()` 与 `gather()` 的使用场景
> - 深刻理解协程对象与 Task 对象的本质区别
> - 了解未等待的 Task 会被自动取消的机制
> - 明白 Task 不能将阻塞式代码变成异步
> - 能够量化分析顺序执行与并发执行的耗时差异
> - 识别并纠正常见的异步并发误区

上一章我们写了第一个 `async/await` 程序，知道了 `await` 会暂停当前协程、把控制权交还事件循环。但有一个关键问题还没解决：如果我们用 `await` 一个接一个地等待多个协程，它们其实是**顺序执行**的——一个做完才开始下一个。

这一章要解决的核心问题是：**怎么让多个协程真正并发执行？**

---

## 3.1 先看顺序版本：await A; await B; await C = 各延迟之和

我们先写一个看起来"好像并发了"但其实没有的版本。这是很多人刚学 asyncio 时的第一个陷阱。

### 三个慢操作

假设我们有三个模拟的异步任务，每个都需要不同的时间来完成：

```python
import asyncio
import time

async def fetch_data_a() -> str:
    """模拟从数据源 A 获取数据，耗时 3 秒"""
    print("[A] 开始获取数据...")
    await asyncio.sleep(3)
    print("[A] 数据获取完成")
    return "数据A"

async def fetch_data_b() -> str:
    """模拟从数据源 B 获取数据，耗时 2 秒"""
    print("[B] 开始获取数据...")
    await asyncio.sleep(2)
    print("[B] 数据获取完成")
    return "数据B"

async def fetch_data_c() -> str:
    """模拟从数据源 C 获取数据，耗时 4 秒"""
    print("[C] 开始获取数据...")
    await asyncio.sleep(4)
    print("[C] 数据获取完成")
    return "数据C"
```

三个任务分别需要 3 秒、2 秒、4 秒。如果并发执行，理论上只需要 `max(3, 2, 4) = 4` 秒。

### 顺序版本：一个接一个

```python
async def sequential_version() -> None:
    """顺序版本：依次等待每个任务"""
    start = time.perf_counter()

    a = await fetch_data_a()   # 等 3 秒
    b = await fetch_data_b()   # 等 2 秒
    c = await fetch_data_c()   # 等 4 秒

    elapsed = time.perf_counter() - start
    print(f"结果: {a}, {b}, {c}")
    print(f"总耗时: {elapsed:.1f} 秒")   # 3 + 2 + 4 = 9 秒

asyncio.run(sequential_version())
```

运行结果：

```
[A] 开始获取数据...
[A] 数据获取完成
[B] 开始获取数据...
[B] 数据获取完成
[C] 开始获取数据...
[C] 数据获取完成
结果: 数据A, 数据B, 数据C
总耗时: 9.0 秒
```

总耗时 9 秒 = 3 + 2 + 4。

### 为什么是顺序的？

来看时间线：

```
时间轴（秒）
0         1         2         3         4         5         6         7         8         9
├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│                                         fetch_data_a (3s)
                                │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│                 fetch_data_b (2s)
                                                          │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  fetch_data_c (4s)
```

关键原因：**每个 `await` 都会等待该协程完成后才执行下一行代码。**

```python
a = await fetch_data_a()   # 执行到这里时，整个流程暂停，直到 A 完成
b = await fetch_data_b()   # 只有 A 完成后，才会开始 B
c = await fetch_data_c()   # 只有 B 完成后，才会开始 C
```

`await` 的语义是"暂停当前协程，等待这个可等待对象完成"。在等待期间，虽然事件循环可以去执行别的协程，但**当前协程后面的代码不会被执行**——因为当前协程还在 `await` 那一行停着。

这就是顺序执行的本质：**一行 `await` 就是一个阻塞点，必须等到结果回来才能继续。**

### 深入理解：await 的暂停点模型

可以把每个 `await` 想象成一个**检查站**：

```python
async def sequential() -> None:
    # ── 检查站 1 ──
    a = await fetch_data_a()    # 暂停，等 A 完成
    # ← A 完成后，从这里恢复

    # ── 检查站 2 ──
    b = await fetch_data_b()    # 暂停，等 B 完成
    # ← B 完成后，从这里恢复

    # ── 检查站 3 ──
    c = await fetch_data_c()    # 暂停，等 C 完成
    # ← C 完成后，从这里恢复
```

在每个检查站，当前协程"冻结"了——它的执行状态被保存起来，控制权交还给事件循环。事件循环可以去执行别的协程，但当前协程必须等到它等待的对象完成后才能从检查站恢复。

### 一个生活类比：洗衣服

想象你要洗三桶衣服，每桶需要不同的洗涤时间：

```
顺序方式（一台洗衣机）：
  洗第一桶（30分钟）→ 洗第二桶（20分钟）→ 洗第三桶（40分钟）
  总时间 = 30 + 20 + 40 = 90 分钟

你会站在洗衣机前面等第一桶洗完，然后才放第二桶进去。
```

这显然很蠢。正常人会怎么做？

```
并发方式（三台洗衣机同时启动）：
  同时启动三台洗衣机
  等最慢的那桶洗完
  总时间 = max(30, 20, 40) = 40 分钟
```

顺序 `await` 就像只有一台洗衣机，一个接一个地洗。`create_task()` 就像同时启动多台洗衣机。

### 实际场景：Web 应用中的数据库查询

```python
# ❌ 顺序执行：用户要等所有查询的总时间
async def render_dashboard(user_id: int) -> dict:
    profile = await db.fetch_profile(user_id)     # 50ms
    orders = await db.fetch_orders(user_id)       # 80ms
    notifications = await db.fetch_notifications(user_id)  # 30ms
    recommendations = await ml.get_recommendations(user_id)  # 120ms
    # 总耗时 ≈ 50 + 80 + 30 + 120 = 280ms
    return {"profile": profile, "orders": orders, ...}

# ✅ 并发执行：用户只需要等最慢的那个查询
async def render_dashboard(user_id: int) -> dict:
    profile, orders, notifications, recommendations = await asyncio.gather(
        db.fetch_profile(user_id),
        db.fetch_orders(user_id),
        db.fetch_notifications(user_id),
        ml.get_recommendations(user_id),
    )
    # 总耗时 ≈ max(50, 80, 30, 120) = 120ms
    return {"profile": profile, "orders": orders, ...}
```

在这个真实场景中，从 280ms 降到 120ms——用户体验提升超过一倍。

<div data-component="GatherVsSequential"></div>

---

</div>
## 3.2 asyncio.create_task()：把协程包装成 Task

要实现并发，我们不能一个接一个地 `await`。我们需要**同时启动**多个任务，然后再等它们全部完成。

`asyncio.create_task()` 就是做这件事的：它把一个协程包装成一个 **Task** 对象，然后**立刻**把它交给事件循环去调度。

### 基本用法

```python
import asyncio
import time

async def fetch_data_a() -> str:
    await asyncio.sleep(3)
    return "数据A"

async def fetch_data_b() -> str:
    await asyncio.sleep(2)
    return "数据B"

async def fetch_data_c() -> str:
    await asyncio.sleep(4)
    return "数据C"

async def concurrent_version() -> None:
    start = time.perf_counter()

    # 第一步：创建 Task（不等待，只是注册）
    task_a = asyncio.create_task(fetch_data_a())
    task_b = asyncio.create_task(fetch_data_b())
    task_c = asyncio.create_task(fetch_data_c())

    # 第二步：等待所有 Task 完成
    a = await task_a
    b = await task_b
    c = await task_c

    elapsed = time.perf_counter() - start
    print(f"结果: {a}, {b}, {c}")
    print(f"总耗时: {elapsed:.1f} 秒")   # 约 4 秒

asyncio.run(concurrent_version())
```

运行结果：

```
结果: 数据A, 数据B, 数据C
总耗时: 4.0 秒
```

总耗时只有 4 秒，不是 9 秒！

### 时间线对比

```
时间轴（秒）
0         1         2         3         4
├─────────┼─────────┼─────────┼─────────┤
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│       fetch_data_a (3s)
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│               fetch_data_b (2s)
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  fetch_data_c (4s)
```

三个任务**同时开始**，同时推进，总耗时 = `max(3, 2, 4)` = 4 秒。

### create_task() 做了什么？

`asyncio.create_task()` 的核心动作：

1. **接收一个协程对象**作为参数
2. **创建一个 Task 对象**（Task 是 Future 的子类）
3. **把 Task 注册到当前事件循环**，事件循环会在下一个可用时机开始执行它
4. **立刻返回 Task 对象**——不会等待协程执行完成

```python
# 这行代码会立刻返回，不会等待 3 秒
task_a = asyncio.create_task(fetch_data_a())

# 此时 fetch_data_a() 可能还没开始执行，也可能刚开始
# 但它已经被"安排"了，事件循环会在合适的时候执行它
```

### 用 time.perf_counter() 验证"立刻返回"

```python
async def verify_immediate_return() -> None:
    """验证 create_task() 确实立刻返回"""
    start = time.perf_counter()

    # 创建 Task — 这应该几乎不花时间
    task = asyncio.create_task(fetch_data_a())  # fetch_data_a 需要 3 秒

    create_time = time.perf_counter() - start
    print(f"create_task() 耗时: {create_time * 1000:.3f}ms")  # 约 0.01ms

    # 等待 Task — 这才需要 3 秒
    result = await task

    total_time = time.perf_counter() - start
    print(f"await task 耗时: {total_time:.1f}s")  # 约 3.0s
    print(f"结果: {result}")

asyncio.run(verify_immediate_return())
```

运行结果：

```
create_task() 耗时: 0.012ms
await task 耗时: 3.0s
结果: 数据A
```

`create_task()` 只花了 0.012 毫秒就返回了，真正的 3 秒等待发生在 `await task` 的时候。

### Task 对象的常用属性

创建 Task 后，你可以查询它的状态：

```python
async def inspect_task() -> None:
    """检查 Task 对象的属性"""

    async def slow_work() -> str:
        await asyncio.sleep(2)
        return "完成"

    task = asyncio.create_task(slow_work())

    # 检查是否完成
    print(f"已完成? {task.done()}")          # False

    # 检查是否被取消
    print(f"已取消? {task.cancelled()}")     # False

    # 获取任务名称（Python 3.8+）
    print(f"任务名: {task.get_name()}")      # "Task-1" 或自定义名

    # 等待完成
    result = await task

    # 再次检查
    print(f"已完成? {task.done()}")          # True
    print(f"结果: {result}")

asyncio.run(inspect_task())
```

### 自定义 Task 名称

给 Task 起个名字，方便调试：

```python
async def named_tasks() -> None:
    """给 Task 命名，方便调试和日志"""
    task_a = asyncio.create_task(
        fetch_data_a(),
        name="fetch-user-profile"
    )
    task_b = asyncio.create_task(
        fetch_data_b(),
        name="fetch-orders"
    )

    print(f"Task A 名称: {task_a.get_name()}")  # "fetch-user-profile"
    print(f"Task B 名称: {task_b.get_name()}")  # "fetch-orders"

    await asyncio.gather(task_a, task_b)
```

### create_task() 必须在事件循环中调用

`create_task()` 需要一个正在运行的事件循环，所以它只能在 `async def` 函数内部调用：

```python
# ❌ 错误：在普通函数中调用
def setup():
    task = asyncio.create_task(some_coroutine())  # RuntimeError!

# ❌ 错误：在事件循环启动前调用
task = asyncio.create_task(some_coroutine())  # RuntimeError!
asyncio.run(main())

# ✅ 正确：在 async 函数中调用
async def main():
    task = asyncio.create_task(some_coroutine())  # 正确
    await task
```

<div data-component="CreateTaskDemo"></div>

</div>
---

## 3.3 为什么只需要 max(delay)：三个等待重叠

理解了 `create_task()` 之后，我们来深入分析为什么并发版本只需要 4 秒。

### 重叠的等待

在顺序版本中：

```
等待 A（3秒）→ 等待 B（2秒）→ 等待 C（4秒）
总时间 = 3 + 2 + 4 = 9 秒
```

在并发版本中：

```
同时启动 A、B、C
等待 max(3, 2, 4) = 4 秒
```

关键区别在于**等待是重叠的**。当我们在等待 A 完成的同时，B 和 C 也在执行。当我们最终 `await task_c` 时，如果 C 已经完成了（因为它只需要 4 秒，而我们已经等了 4 秒），那就立刻返回。

### 用代码验证

```python
async def demonstrate_overlap() -> None:
    """演示三个任务的执行是重叠的"""
    start = time.perf_counter()

    def elapsed() -> float:
        return time.perf_counter() - start

    async def timed_task(name: str, delay: float) -> str:
        print(f"[{elapsed():.1f}s] {name} 开始执行")
        await asyncio.sleep(delay)
        print(f"[{elapsed():.1f}s] {name} 执行完成（耗时 {delay}s）")
        return name

    # 创建三个 Task
    task_a = asyncio.create_task(timed_task("A", 3))
    task_b = asyncio.create_task(timed_task("B", 2))
    task_c = asyncio.create_task(timed_task("C", 4))

    # 等待所有完成
    await task_a
    await task_b
    await task_c

    print(f"[{elapsed():.1f}s] 全部完成")

asyncio.run(demonstrate_overlap())
```

运行结果：

```
[0.0s] A 开始执行
[0.0s] B 开始执行
[0.0s] C 开始执行
[2.0s] B 执行完成（耗时 2s）
[3.0s] A 执行完成（耗时 3s）
[4.0s] C 执行完成（耗时 4s）
[4.0s] 全部完成
```

注意：三个任务的"开始执行"几乎是同一时刻（都在 0.0 秒）。这就是**真正的并发**——它们同时在推进。

### 为什么是 max 而不是 sum？

想象你在等三壶水烧开：

- 壶 A 需要 3 分钟
- 壶 B 需要 2 分钟
- 壶 C 需要 4 分钟

如果你一壶一壶地等：3 + 2 + 4 = 9 分钟。

如果你同时把三壶水放上去烧，然后等最慢的那壶：4 分钟。

`asyncio.create_task()` 就是"同时把三壶水放上去"的动作。`await task_c` 就是"等最慢的那壶"。

**并发的本质：让多个任务的等待时间重叠，总耗时由最慢的任务决定。**

### 数学分析

设 n 个任务的耗时分别为 $d_1, d_2, \ldots, d_n$：

```
顺序执行总耗时 = d₁ + d₂ + ... + dₙ = Σdᵢ
并发执行总耗时 = max(d₁, d₂, ..., dₙ)
```

加速比：

```
加速比 = Σdᵢ / max(dᵢ)
```

以我们的例子为例：

```
顺序: 3 + 2 + 4 = 9 秒
并发: max(3, 2, 4) = 4 秒
加速比: 9 / 4 = 2.25 倍
```

如果任务耗时更均匀，加速比更大：

```
10 个任务，每个 1 秒：
  顺序: 10 秒
  并发: 1 秒
  加速比: 10 倍

100 个任务，每个 0.5 秒：
  顺序: 50 秒
  并发: 0.5 秒
  加速比: 100 倍
```

这就是为什么在 I/O 密集型场景中（比如同时请求 100 个 API），并发能带来巨大的性能提升。

### 瓶颈任务的影响

并发的总耗时由**最慢的任务**决定。如果有一个任务特别慢，它会拖累整体：

```python
async def bottleneck_demo() -> None:
    """演示瓶颈任务对并发总耗时的影响"""
    start = time.perf_counter()

    # 4 个快速任务 + 1 个慢任务
    await asyncio.gather(
        asyncio.create_task(asyncio.sleep(1)),   # 快
        asyncio.create_task(asyncio.sleep(1)),   # 快
        asyncio.create_task(asyncio.sleep(1)),   # 快
        asyncio.create_task(asyncio.sleep(1)),   # 快
        asyncio.create_task(asyncio.sleep(10)),  # ← 瓶颈！
    )

    elapsed = time.perf_counter() - start
    print(f"总耗时: {elapsed:.1f}s")   # 10 秒，不是 1 秒
    # 因为最慢的任务需要 10 秒，所有其他任务必须等它

asyncio.run(bottleneck_demo())
```

**优化提示**：如果你发现并发执行仍然很慢，检查是否有"瓶颈任务"——某个任务的耗时远超其他任务。优化这个瓶颈任务，整体性能就会提升。

---

## 3.4 创建 Task 不等于立刻抢占执行

很多人有一个误解：以为 `create_task()` 之后，新的 Task 会**立刻中断**当前协程、抢占执行。

事实并非如此。Task 被创建后，它只是被**安排**到事件循环的待办列表中。当前协程会继续执行，直到它主动让出控制权（遇到 `await`）。

### 演示：创建 Task 后继续执行

```python
async def show_scheduling() -> None:
    """演示创建 Task 后，当前协程继续执行"""

    async def background() -> None:
        print("[background] 我开始执行了")
        await asyncio.sleep(1)
        print("[background] 我执行完了")

    print("[main] 1. 创建 Task 之前")
    task = asyncio.create_task(background())
    print("[main] 2. 创建 Task 之后，Task 还没开始执行")
    print("[main] 3. 继续做其他事情...")
    print("[main] 4. 即将 await Task")

    await task
    print("[main] 5. Task 已完成")

asyncio.run(show_scheduling())
```

运行结果：

```
[main] 1. 创建 Task 之前
[main] 2. 创建 Task 之后，Task 还没开始执行
[main] 3. 继续做其他事情...
[main] 4. 即将 await Task
[background] 我开始执行了
[background] 我执行完了
[main] 5. Task 已完成
```

注意第 2、3、4 行——它们在 `background` 开始执行**之前**就打印出来了。这说明 `create_task()` 之后，当前协程继续执行了三行代码，然后才在 `await task` 处让出控制权，事件循环这才开始执行 `background`。

### 执行顺序的原理

```
main 协程                          事件循环
│                                   │
├── print "1. 创建 Task 之前"       │
├── create_task(background)         │   ← 把 background 加入待办列表
├── print "2. 创建 Task 之后"       │   ← main 继续执行，没有让出
├── print "3. 继续做其他事情"        │   ← main 继续执行，没有让出
├── print "4. 即将 await Task"      │   ← main 继续执行，没有让出
├── await task ─────────────────────→│   ← main 让出控制权
│                                   ├── 开始执行 background
│                                   ├── print "我开始执行了"
│                                   ├── sleep(1)
│   （等待中...）                    │   ← 事件循环去干别的
│                                   ├── print "我执行完了"
│←────────────────── task 完成 ──────┤
├── print "5. Task 已完成"          │
```

**关键规则：`create_task()` 只是"安排"，不是"立刻执行"。当前协程会继续运行，直到遇到下一个 `await`。**

<div data-component="TaskSchedulingOrder"></div>

---

</div>
## 3.5 await asyncio.sleep(0)：主动让出控制权

有时候你需要在没有自然 `await` 点的地方，主动给事件循环一个执行其他 Task 的机会。这时候可以用 `await asyncio.sleep(0)`。

### 为什么要主动让出？

```python
async def no_yield() -> None:
    """不让出控制权的例子"""

    async def quick_task() -> None:
        print("[quick] 我执行了！")

    task = asyncio.create_task(quick_task())

    # 这里没有 await，所以 quick_task 没有机会执行
    # 下面这段代码跑完之前，事件循环拿不回控制权
    total = 0
    for i in range(10_000_000):
        total += i

    print(f"[main] 计算完成: {total}")
    await task  # 这里才让出，quick_task 才有机会执行

asyncio.run(no_yield())
```

在这个例子中，`quick_task` 要等到 `for` 循环执行完、`await task` 让出控制权之后才有机会执行。如果循环很长，`quick_task` 会被"饿死"。

### sleep(0) 的作用

`await asyncio.sleep(0)` 的意思是"立即让出控制权，但不需要等待任何时间"。它的效果是：

1. 暂停当前协程
2. 把控制权交还事件循环
3. 事件循环检查有没有其他就绪的 Task，如果有就执行
4. 当前协程在下一个调度周期恢复执行

```python
async def with_yield() -> None:
    """主动让出控制权的例子"""

    async def quick_task() -> None:
        print("[quick] 我执行了！")

    task = asyncio.create_task(quick_task())

    # 主动让出控制权
    await asyncio.sleep(0)   # ← 事件循环有机会执行 quick_task 了

    print("[main] 继续执行")
    await task

asyncio.run(with_yield())
```

运行结果：

```
[quick] 我执行了！
[main] 继续执行
```

`quick_task` 在 `sleep(0)` 之后就被执行了。

### sleep(0) 的本质

```python
await asyncio.sleep(0)
```

等价于：

> "我现在不需要等待任何 I/O，但我想给其他协程一个执行的机会。"

它不会真的暂停 0 秒——它会暂停当前协程，让事件循环轮转一圈，然后在下一个可用时机恢复。

**什么时候用 sleep(0)？**
- 长时间计算的循环中，定期让出控制权
- 需要确保刚创建的 Task 有机会开始执行
- 实现协作式调度的"公平"机制

```python
async def fair_loop() -> None:
    """在长时间循环中定期让出"""
    for i in range(100_000_000):
        # 每 10000 次迭代让出一次
        if i % 10_000 == 0:
            await asyncio.sleep(0)
        # ... 做一些计算 ...
```

### sleep(0) 的实际应用场景

**场景 1：进度报告**

```python
async def process_items(items: list) -> list:
    """处理大量项目时，定期让出以更新 UI 或响应心跳"""
    results = []
    for i, item in enumerate(items):
        result = await process_single_item(item)
        results.append(result)

        # 每处理 100 个项目，让出一次
        # 这样事件循环可以处理 WebSocket 心跳、HTTP 健康检查等
        if i % 100 == 0:
            await asyncio.sleep(0)
            print(f"进度: {i}/{len(items)}")
    return results
```

**场景 2：确保并发 Task 有执行机会**

```python
async def producer_consumer() -> None:
    """生产者-消费者模式中，确保消费者有机会执行"""
    queue = asyncio.Queue()

    async def producer() -> None:
        for i in range(10):
            await queue.put(i)
            print(f"生产: {i}")
            await asyncio.sleep(0)  # 让消费者有机会取数据

    async def consumer() -> None:
        while True:
            item = await queue.get()
            print(f"消费: {item}")
            if item == 9:
                break

    task_c = asyncio.create_task(consumer())
    await producer()
    await task_c
```

**场景 3：调试和观察调度顺序**

```python
async def debug_scheduling() -> None:
    """用 sleep(0) 观察 Task 的调度顺序"""

    async def task(name: str) -> None:
        print(f"[{name}] 开始")
        await asyncio.sleep(0)
        print(f"[{name}] 第二步")
        await asyncio.sleep(0)
        print(f"[{name}] 完成")

    t1 = asyncio.create_task(task("A"))
    t2 = asyncio.create_task(task("B"))

    # 不加 sleep(0)，A 和 B 的"开始"可能在 main 让出后才打印
    await asyncio.sleep(0)  # 让 A 和 B 都有机会开始
    print("[main] 中间检查点")
    await asyncio.gather(t1, t2)
```

运行结果：

```
[A] 开始
[B] 开始
[main] 中间检查点
[A] 第二步
[B] 第二步
[A] 完成
[B] 完成
```

注意 A 和 B 的步骤是**交替执行**的——这就是协作式调度的效果。每次 `await asyncio.sleep(0)` 都是一次调度机会，事件循环会轮转到下一个就绪的 Task。

### sleep(0) vs sleep(epsilon)

你可能会想：`sleep(0)` 和 `sleep(0.001)` 有什么区别？

```python
await asyncio.sleep(0)      # 立即让出，下一个循环周期恢复
await asyncio.sleep(0.001)  # 让出，至少等 1ms 才恢复
```

`sleep(0)` 是最快的"让出-恢复"方式，它只消耗一个事件循环周期。`sleep(0.001)` 会真的暂停至少 1 毫秒。

**一般情况下用 `sleep(0)` 就够了。** 只有在你需要"稍微等一下"（比如等待某个外部条件变化）时才用 `sleep(小正数)`。

---

## 3.6 asyncio.gather()：并发等待，收集结果

`create_task()` + 逐个 `await` 虽然能实现并发，但写起来有点繁琐。`asyncio.gather()` 提供了更简洁的方式：**一次并发运行多个协程，等待全部完成，然后收集结果。**

### 基本用法

```python
import asyncio
import time

async def fetch_data_a() -> str:
    await asyncio.sleep(3)
    return "数据A"

async def fetch_data_b() -> str:
    await asyncio.sleep(2)
    return "数据B"

async def fetch_data_c() -> str:
    await asyncio.sleep(4)
    return "数据C"

async def gather_version() -> None:
    start = time.perf_counter()

    # 一行代码实现并发
    results = await asyncio.gather(
        fetch_data_a(),
        fetch_data_b(),
        fetch_data_c(),
    )

    elapsed = time.perf_counter() - start
    print(f"结果: {results}")
    print(f"总耗时: {elapsed:.1f} 秒")   # 约 4 秒

asyncio.run(gather_version())
```

运行结果：

```
结果: ['数据A', '数据B', '数据C']
总耗时: 4.0 秒
```

### gather() 的返回值

`asyncio.gather()` 返回一个列表，包含了每个协程的返回值，**顺序与传入的顺序一致**：

```python
results = await asyncio.gather(
    fetch_data_a(),   # 索引 0
    fetch_data_b(),   # 索引 1
    fetch_data_c(),   # 索引 2
)
# results[0] = "数据A"
# results[1] = "数据B"
# results[2] = "数据C"
```

不管哪个任务先完成，结果的顺序始终是传入的顺序。

### 逐行对比

```python
# 使用 create_task() — 6 行
task_a = asyncio.create_task(fetch_data_a())
task_b = asyncio.create_task(fetch_data_b())
task_c = asyncio.create_task(fetch_data_c())
a = await task_a
b = await task_b
c = await task_c

# 使用 gather() — 1 行
a, b, c = await asyncio.gather(
    fetch_data_a(),
    fetch_data_b(),
    fetch_data_c(),
)
```

`gather()` 是 `create_task()` + `await all` 的语法糖。

### gather() 也可以接收已创建的 Task

`gather()` 不仅可以接收协程，也可以接收已经创建好的 Task：

```python
async def gather_with_tasks() -> None:
    """gather() 可以接收协程和 Task 混合"""
    # 先创建一些 Task
    task_a = asyncio.create_task(fetch_data_a())
    task_c = asyncio.create_task(fetch_data_c())

    # gather() 可以混合接收协程和 Task
    results = await asyncio.gather(
        task_a,           # 已创建的 Task
        fetch_data_b(),   # 直接传协程
        task_c,           # 已创建的 Task
    )
    print(results)
```

### gather() 与列表推导式

当需要并发处理一批相似的任务时，`gather()` 配合列表推导式非常优雅：

```python
async def fetch_url(url: str) -> str:
    await asyncio.sleep(1)  # 模拟网络请求
    return f"内容来自 {url}"

async def fetch_all_urls() -> None:
    """并发获取一批 URL"""
    urls = [
        "https://api.example.com/users",
        "https://api.example.com/orders",
        "https://api.example.com/products",
        "https://api.example.com/reviews",
        "https://api.example.com/stats",
    ]

    # 一行代码并发获取所有 URL
    results = await asyncio.gather(*[fetch_url(url) for url in urls])

    for url, content in zip(urls, results):
        print(f"{url}: {content}")
```

### gather() 的 return_exceptions 参数

默认情况下，如果某个任务抛出异常，`gather()` 会重新抛出它。但有时你希望拿到所有结果（包括异常）：

```python
async def may_fail(name: str, should_fail: bool) -> str:
    await asyncio.sleep(1)
    if should_fail:
        raise ValueError(f"{name} 失败了！")
    return f"{name} 成功"

async def gather_with_exceptions() -> None:
    """return_exceptions=True 时，异常也会作为结果返回"""

    # 默认行为：遇到异常就抛出
    try:
        results = await asyncio.gather(
            may_fail("A", False),
            may_fail("B", True),    # 会抛出异常
            may_fail("C", False),
        )
    except ValueError as e:
        print(f"捕获异常: {e}")  # 只能捕获第一个异常

    # 使用 return_exceptions=True
    results = await asyncio.gather(
        may_fail("A", False),
        may_fail("B", True),
        may_fail("C", False),
        return_exceptions=True,   # ← 异常作为结果返回，不抛出
    )

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"任务 {i} 失败: {result}")
        else:
            print(f"任务 {i} 成功: {result}")

asyncio.run(gather_with_exceptions())
```

运行结果：

```
捕获异常: B 失败了！
任务 0 成功: A 成功
任务 1 失败: B 失败了！
任务 2 成功: C 成功
```

`return_exceptions=True` 让你可以拿到所有结果，然后逐个判断哪些成功、哪些失败。这在"部分失败也值得继续"的场景中很有用。

---

## 3.7 gather() 背后做了什么

表面上 `gather()` 只是一行代码，但它内部做了四件事。理解这四件事有助于你判断什么时候该用 `gather()`，什么时候该用 `create_task()`。

### 第一步：安排（Arrange）

```python
asyncio.gather(fetch_data_a(), fetch_data_b(), fetch_data_c())
```

`gather()` 接收多个协程（或 Task）作为参数。它会为每个协程创建一个 Task（如果传入的还不是 Task 的话），并把它们注册到事件循环。

### 第二步：并发（Concurrent）

所有 Task 同时开始执行。它们在事件循环中交替推进，就像我们用 `create_task()` 手动创建的那样。

### 第三步：等待全部（Wait All）

`gather()` 会等待**所有** Task 完成。如果任何一个 Task 抛出异常，默认情况下 `gather()` 会等待其余 Task 也完成，然后重新抛出第一个异常。

### 第四步：收集结果（Collect）

所有 Task 完成后，`gather()` 按照**传入顺序**收集每个 Task 的返回值，打包成一个列表返回。

```
gather() 的内部流程：

    协程A ──→ TaskA ──→ 执行 ──→ 结果A ──┐
    协程B ──→ TaskB ──→ 执行 ──→ 结果B ──┼──→ [结果A, 结果B, 结果C]
    协程C ──→ TaskC ──→ 执行 ──→ 结果C ──┘
```

### 伪代码实现

如果让你自己实现一个简化版的 `gather()`，大概是这样：

```python
async def my_gather(*coroutines) -> list:
    """简化版 gather() 实现（仅示意，非真实实现）"""
    # 第一步：安排 — 为每个协程创建 Task
    tasks = [asyncio.create_task(coro) for coro in coroutines]

    # 第二步+第三步：并发执行并等待全部完成
    results = []
    for task in tasks:
        result = await task        # 等待每个 Task
        results.append(result)

    # 第四步：收集结果（已在上面的循环中完成）
    return results
```

真实的 `gather()` 实现要复杂得多（涉及异常处理、取消传播、return_exceptions 参数等），但核心逻辑就是这四步。

---

## 3.8 完成顺序 vs 结果顺序

这是一个非常重要的细节：**任务完成的顺序和 `gather()` 返回结果的顺序可能不同。**

### 演示

```python
async def task_with_delay(name: str, delay: float) -> str:
    """带延迟的任务，返回名称和完成时间"""
    start = time.perf_counter()
    await asyncio.sleep(delay)
    elapsed = time.perf_counter() - start
    print(f"[{elapsed:.1f}s] {name} 完成（延迟 {delay}s）")
    return name

async def completion_order_demo() -> None:
    start = time.perf_counter()

    results = await asyncio.gather(
        task_with_delay("A-慢", 3.0),
        task_with_delay("B-快", 1.0),
        task_with_delay("C-中", 2.0),
    )

    print(f"\n完成耗时: {time.perf_counter() - start:.1f}s")
    print(f"gather 返回顺序: {results}")

asyncio.run(completion_order_demo())
```

运行结果：

```
[1.0s] B-快 完成（延迟 1.0s）
[2.0s] C-中 完成（延迟 2.0s）
[3.0s] A-慢 完成（延迟 3.0s）

完成耗时: 3.0s
gather 返回顺序: ['A-慢', 'B-快', 'C-中']
```

注意：

- **完成顺序**：B → C → A（按实际完成时间排列）
- **结果顺序**：A, B, C（按传入顺序排列）

`gather()` 的返回结果**始终按照传入的顺序**，而不是完成的顺序。这是 `gather()` 的设计保证。

### 为什么保持传入顺序很重要？

```python
# 我们可以安全地用解构赋值
urls, titles, contents = await asyncio.gather(
    fetch_urls(),
    fetch_titles(),
    fetch_contents(),
)
# urls 就是 fetch_urls 的结果，不用担心顺序对不上
```

如果 `gather()` 按完成顺序返回，你就不知道哪个结果对应哪个任务了。

<div data-component="CompletionVsResultOrder"></div>

---

</div>
## 3.9 create_task() vs gather()：个体 vs 批量

`create_task()` 和 `gather()` 都能实现并发，但适用场景不同。

### create_task()：精细控制

```python
async def fine_grained_control() -> None:
    """需要精细控制时使用 create_task()"""

    # 可以在不同时间点创建 Task
    task_a = asyncio.create_task(fetch_data_a())

    # 做一些准备工作...
    await asyncio.sleep(0.5)

    # 然后再创建其他 Task
    task_b = asyncio.create_task(fetch_data_b())

    # 可以单独检查某个 Task 的状态
    print(f"task_a 完成了吗? {task_a.done()}")

    # 可以按任意顺序等待
    b = await task_b   # 先等 B
    a = await task_a   # 再等 A
```

**适用场景**：
- 需要在不同时间点启动任务
- 需要单独取消某个任务
- 需要检查任务状态（`done()`, `cancelled()`）
- 需要给任务加回调（`add_done_callback()`）
- 任务之间有依赖关系，等待顺序不固定

### gather()：批量操作

```python
async def batch_operation() -> None:
    """不需要精细控制时使用 gather()"""

    # 一次启动并等待所有任务
    results = await asyncio.gather(
        fetch_data_a(),
        fetch_data_b(),
        fetch_data_c(),
    )

    # 直接拿到结果列表
    for name, result in zip(["A", "B", "C"], results):
        print(f"{name}: {result}")
```

**适用场景**：
- 同时启动一批任务，等待全部完成
- 不需要单独控制每个任务
- 想要简洁的代码
- 需要按传入顺序收集结果

### 对比表

| 特性 | `create_task()` | `gather()` |
|------|----------------|------------|
| 语法 | 分步创建，分步 await | 一步到位 |
| 返回值 | 每个 Task 单独返回 | 返回结果列表 |
| 启动时机 | 创建时立刻注册 | 调用时立刻注册 |
| 精细控制 | 可以单独取消、检查状态 | 只能整体操作 |
| 异常处理 | 每个 Task 独立处理 | 默认重新抛出第一个异常 |
| 取消支持 | 单独取消某个 Task | 取消会传播到所有子 Task |
| 代码量 | 较多 | 较少 |

### 如何选择？

```python
# 场景 1：一批独立任务，全部完成后处理结果 → 用 gather
results = await asyncio.gather(*[fetch(url) for url in urls])

# 场景 2：任务需要在不同时间点启动 → 用 create_task
task = asyncio.create_task(monitor())
# ... 做其他事情 ...
await some_other_work()
# ... 然后才需要 monitor 的结果 ...
await task

# 场景 3：需要取消某个任务 → 用 create_task
task = asyncio.create_task(long_running_task())
await asyncio.sleep(5)
if not task.done():
    task.cancel()
```

---

## 3.10 协程对象 vs Task：描述 vs 已安排

这是理解 asyncio 最容易混淆的概念之一：**协程对象**和 **Task 对象**看起来很像，但本质完全不同。

### 协程对象：只是"描述"

```python
async def greet(name: str) -> str:
    await asyncio.sleep(1)
    return f"Hello, {name}"

# 这只是创建了一个协程对象，还没有执行任何代码
coro = greet("World")

# 协程对象是一个"描述"：它描述了"要做什么"
# 就像一份食谱，还没开始做菜
```

协程对象的特点：
- 它是一个**惰性**的可等待对象
- 创建它不会执行任何代码
- 它只是描述了"要做的一系列操作"
- 必须被 `await` 或包装成 Task 才会执行
- 如果创建后没有 `await`，Python 会发出警告

### Task：已安排执行

```python
# 把协程包装成 Task
task = asyncio.create_task(greet("World"))

# 此时 Task 已经被安排到事件循环
# 事件循环会在下一个可用时机开始执行它
# 就像菜谱已经交给厨师，厨师开始做菜了
```

Task 对象的特点：
- 它是一个**已在事件循环中注册**的可等待对象
- 创建它时就开始安排执行（但不一定立刻执行）
- 它有状态：pending → running → done
- 可以检查状态：`task.done()`, `task.cancelled()`
- 可以取消：`task.cancel()`
- 可以加回调：`task.add_done_callback()`

### 关键对比

```python
async def compare_coroutine_and_task() -> None:
    # 创建协程对象
    coro = greet("协程")
    print(f"协程类型: {type(coro)}")   # <class 'coroutine'>

    # 创建 Task
    task = asyncio.create_task(greet("Task"))
    print(f"Task 类型: {type(task)}")  # <class '_asyncio.Task'>

    # Task 有状态，协程没有
    print(f"Task 完成了吗? {task.done()}")   # False
    # print(coro.done())   # ← AttributeError! 协程没有 done() 方法

    # 两者都可以 await
    result1 = await coro
    result2 = await task

asyncio.run(compare_coroutine_and_task())
```

### 图解

```
协程对象（描述）              Task（已安排）
┌─────────────────┐         ┌─────────────────────┐
│  greet("World") │         │  Task(greet("World"))│
│                 │         │                     │
│  状态：无        │ create_task() │  状态：pending    │
│  只是一份计划    │ ──────────→ │  已注册到事件循环   │
│                 │         │  等待被调度执行       │
└─────────────────┘         └─────────────────────┘
```

**一句话总结：协程是"菜谱"，Task 是"已经交给厨师的订单"。**

### 常见陷阱：对协程调用 create_task()

```python
async def trap_double_call() -> None:
    """陷阱：对协程函数调用的结果再调用 create_task"""

    async def work() -> str:
        await asyncio.sleep(1)
        return "done"

    # ✅ 正确：把协程对象传给 create_task
    coro = work()                        # 创建协程对象
    task = asyncio.create_task(coro)     # 包装成 Task
    result = await task

    # ❌ 错误：忘记调用 work()，把函数本身传进去了
    # task = asyncio.create_task(work)   # TypeError! work 是函数，不是协程

    # ❌ 错误：调用了两次 work()
    # task = asyncio.create_task(work())  # 这行是对的
    # task2 = asyncio.create_task(work()) # 但如果你以为 task2 == task，那就错了
    # 这是两个独立的 Task，会各自执行一次 work()
```

### 协程只能被 await 一次

```python
async def coroutine_once() -> None:
    """协程对象只能 await 一次"""
    coro = work()

    result1 = await coro   # 第一次 await：正常执行
    # result2 = await coro  # 第二次 await：RuntimeError!
    # "cannot reuse already awaited coroutine"
```

Task 对象也一样——一个 Task 只能 `await` 一次（结果），但多次 `await` 同一个 Task 不会报错，只是后续的 `await` 会立刻返回相同的结果：

```python
async def task_multiple_await() -> None:
    """Task 可以被多次 await，但结果相同"""
    task = asyncio.create_task(work())

    result1 = await task   # 等待 Task 完成
    result2 = await task   # 立刻返回，不会重新执行
    print(result1 == result2)  # True
```

### 什么时候用协程，什么时候用 Task？

```python
# 用协程（直接 await）：只需要执行一次，不需要并发
result = await fetch_data()

# 用 Task（create_task）：需要并发，或者需要延迟等待
task = asyncio.create_task(fetch_data())
# ... 做其他事情 ...
result = await task

# 用 gather：批量并发
results = await asyncio.gather(fetch_a(), fetch_b(), fetch_c())
```

---

## 3.11 不要忘记等待 Task：未完成的 Task 会被取消

创建了 Task 但忘记 `await` 会怎样？

### 忘记等待的后果

```python
async def forgotten_task() -> str:
    """一个被遗忘的 Task"""
    print("[forgotten] 我开始执行了")
    await asyncio.sleep(5)
    print("[forgotten] 我执行完了")   # 这行可能永远不会执行
    return "结果"

async def main() -> None:
    # 创建了 Task 但没有 await
    task = asyncio.create_task(forgotten_task())

    # 主协程很快结束，没有等待 task
    print("[main] 主协程结束")

asyncio.run(main())
```

运行结果：

```
[forgotten] 我开始执行了
[main] 主协程结束
RuntimeWarning: coroutine 'forgotten_task' was never awaited
```

当 `main()` 结束后，事件循环关闭。所有未完成的 Task 会被**取消**，它们的协程也会被**垃圾回收**。

### gather() 也会忘记等待吗？

`gather()` 本身会等待所有任务完成。但如果你手动创建 Task 后忘记 `await`：

```python
async def main() -> None:
    # 这些 Task 会被 gather 等待 — 没问题
    results = await asyncio.gather(
        fetch_data_a(),
        fetch_data_b(),
    )

    # 这个 Task 被遗忘了 — 有问题
    forgotten = asyncio.create_task(fetch_data_c())

    # main 结束时，forgotten 还没完成，会被取消
```

### 如何避免遗忘？

```python
async def safe_pattern() -> None:
    """安全的模式：创建 Task 后立刻记录，最后统一等待"""
    tasks = []

    tasks.append(asyncio.create_task(fetch_data_a()))
    tasks.append(asyncio.create_task(fetch_data_b()))
    tasks.append(asyncio.create_task(fetch_data_c()))

    # 统一等待所有 Task
    results = await asyncio.gather(*tasks)
```

或者更简单——直接用 `gather()`，不要单独创建 Task。

**最佳实践：要么 `gather()` 一步到位，要么把 Task 收集到列表中统一 `await`。永远不要创建 Task 后不管它。**

---

## 3.12 Task 不会让阻塞代码变异步

这是一个非常常见的误解：以为把代码放到 Task 里，它就变成异步的了。

### 阻塞代码依然是阻塞的

```python
import time
import asyncio

async def blocking_task(name: str, delay: float) -> str:
    """使用 time.sleep 而不是 asyncio.sleep"""
    print(f"[{name}] 开始")
    time.sleep(delay)         # ← 这是同步阻塞！不是异步！
    print(f"[{name}] 完成")
    return name

async def main() -> None:
    start = time.perf_counter()

    # 尝试用 create_task 并发执行
    task_a = asyncio.create_task(blocking_task("A", 3))
    task_b = asyncio.create_task(blocking_task("B", 2))

    results = await asyncio.gather(task_a, task_b)

    elapsed = time.perf_counter() - start
    print(f"结果: {results}")
    print(f"总耗时: {elapsed:.1f} 秒")   # 5 秒，不是 3 秒！

asyncio.run(main())
```

运行结果：

```
[A] 开始
[A] 完成
[B] 开始
[B] 完成
结果: ['A', 'B']
总耗时: 5.0 秒
```

总耗时是 5 秒（3 + 2），不是 3 秒（max）。**并发失败了！**

### 为什么 time.sleep 会阻塞整个事件循环？

```
time.sleep(3) 的工作方式：
    ├── 线程级别：让当前线程暂停 3 秒
    ├── 效果：整个线程被阻塞，包括事件循环
    └── 结果：事件循环无法调度其他 Task

asyncio.sleep(3) 的工作方式：
    ├── 协程级别：让当前协程暂停 3 秒
    ├── 效果：当前协程暂停，事件循环继续运行
    └── 结果：事件循环可以去执行其他协程
```

`time.sleep()` 是**同步阻塞调用**，它会阻塞整个线程。在 asyncio 的世界里，事件循环和所有协程都运行在**同一个线程**中。所以当 `time.sleep()` 阻塞了这个线程，事件循环也停了，其他 Task 自然无法执行。

### 有哪些操作会阻塞事件循环？

```python
# 这些都是同步阻塞的，会让事件循环卡住：
time.sleep(5)                    # 阻塞式等待
requests.get(url)                # 同步网络请求
open("file.txt").read()          # 同步文件读取
result = heavy_computation()     # 长时间 CPU 计算

# 这些才是异步的，不会阻塞事件循环：
await asyncio.sleep(5)           # 异步等待
await aiohttp.ClientSession().get(url)  # 异步网络请求
await aiofiles.open("file.txt")  # 异步文件读取
```

### 如何处理阻塞代码？

如果必须使用阻塞代码（比如调用一个同步库），需要用 `asyncio.to_thread()` 把它放到单独的线程中：

```python
import asyncio
import time

def blocking_io() -> str:
    """一个同步阻塞的 I/O 操作"""
    time.sleep(3)   # 阻塞 3 秒
    return "IO完成"

async def main() -> None:
    start = time.perf_counter()

    # ❌ 错误：直接在 async 中调用阻塞代码
    # result = blocking_io()   # 会阻塞事件循环 3 秒

    # ✅ 正确：用 to_thread 放到线程中
    result = await asyncio.to_thread(blocking_io)

    print(f"结果: {result}, 耗时: {time.perf_counter() - start:.1f}s")

asyncio.run(main())
```

### 用 to_thread() 实现并发的阻塞操作

```python
async def concurrent_blocking() -> None:
    """多个阻塞操作也可以并发（通过线程）"""
    start = time.perf_counter()

    # 三个阻塞操作在不同线程中并发执行
    results = await asyncio.gather(
        asyncio.to_thread(blocking_io),
        asyncio.to_thread(blocking_io),
        asyncio.to_thread(blocking_io),
    )

    elapsed = time.perf_counter() - start
    print(f"结果: {results}")
    print(f"耗时: {elapsed:.1f}s")   # 约 3 秒，不是 9 秒

asyncio.run(concurrent_blocking())
```

`to_thread()` 会把同步函数放到一个线程池中执行，不阻塞事件循环。这样即使你必须使用同步库（比如 `requests`、`pandas`），也能实现并发。

### 常见阻塞操作速查表

```
同步阻塞（会卡事件循环）      异步替代方案
─────────────────────────    ─────────────────────────
time.sleep(n)                asyncio.sleep(n)
requests.get(url)            aiohttp / httpx async
open().read()                aiofiles.open()
sqlite3.connect()            aiosqlite
subprocess.run()             asyncio.create_subprocess()
socket.recv()                asyncio streams
input()                      (用线程或专用方案)
```

**核心原则：在 asyncio 的世界里，所有阻塞操作都必须用异步版本替代，或者放到线程中执行。Task 不会魔法般地把同步代码变成异步。**

---

## 3.13 完整对比：顺序执行 vs 并发执行

让我们用一个完整的例子，直观对比两种方式的差异。

### 完整示例代码

```python
import asyncio
import time

async def simulate_api_call(name: str, delay: float) -> dict:
    """模拟一个 API 调用"""
    print(f"  [{name}] 请求发送")
    await asyncio.sleep(delay)
    result = {"source": name, "data": f"{name}的数据", "delay": delay}
    print(f"  [{name}] 响应接收（耗时 {delay}s）")
    return result

async def sequential() -> list:
    """顺序版本"""
    print("\n=== 顺序执行 ===")
    start = time.perf_counter()

    r1 = await simulate_api_call("用户服务", 2.0)
    r2 = await simulate_api_call("订单服务", 1.5)
    r3 = await simulate_api_call("商品服务", 3.0)
    r4 = await simulate_api_call("推荐服务", 0.8)

    elapsed = time.perf_counter() - start
    results = [r1, r2, r3, r4]
    print(f"  总耗时: {elapsed:.1f}s (期望: {2.0 + 1.5 + 3.0 + 0.8:.1f}s)")
    return results

async def concurrent_create_task() -> list:
    """并发版本：create_task()"""
    print("\n=== 并发执行（create_task）===")
    start = time.perf_counter()

    t1 = asyncio.create_task(simulate_api_call("用户服务", 2.0))
    t2 = asyncio.create_task(simulate_api_call("订单服务", 1.5))
    t3 = asyncio.create_task(simulate_api_call("商品服务", 3.0))
    t4 = asyncio.create_task(simulate_api_call("推荐服务", 0.8))

    r1 = await t1
    r2 = await t2
    r3 = await t3
    r4 = await t4

    elapsed = time.perf_counter() - start
    results = [r1, r2, r3, r4]
    print(f"  总耗时: {elapsed:.1f}s (期望: {max(2.0, 1.5, 3.0, 0.8):.1f}s)")
    return results

async def concurrent_gather() -> list:
    """并发版本：gather()"""
    print("\n=== 并发执行（gather）===")
    start = time.perf_counter()

    results = await asyncio.gather(
        simulate_api_call("用户服务", 2.0),
        simulate_api_call("订单服务", 1.5),
        simulate_api_call("商品服务", 3.0),
        simulate_api_call("推荐服务", 0.8),
    )

    elapsed = time.perf_counter() - start
    print(f"  总耗时: {elapsed:.1f}s (期望: {max(2.0, 1.5, 3.0, 0.8):.1f}s)")
    return list(results)

async def main() -> None:
    print("=" * 50)
    print("顺序 vs 并发 执行对比")
    print("=" * 50)

    await sequential()
    await concurrent_create_task()
    await concurrent_gather()

    print("\n" + "=" * 50)
    print("对比总结:")
    print(f"  顺序: 2.0 + 1.5 + 3.0 + 0.8 = {2.0 + 1.5 + 3.0 + 0.8}s")
    print(f"  并发: max(2.0, 1.5, 3.0, 0.8) = {max(2.0, 1.5, 3.0, 0.8)}s")
    print(f"  加速比: {(2.0 + 1.5 + 3.0 + 0.8) / max(2.0, 1.5, 3.0, 0.8):.1f}x")
    print("=" * 50)

asyncio.run(main())
```

### 时间线对比图

```
顺序执行（7.3 秒）：

0s        1s        2s        3s        4s        5s        6s        7s
├─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│                                  用户服务 (2.0s)
                                │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│         订单服务 (1.5s)
                                                          │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  商品服务 (3.0s)
                                                                                                       │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  推荐服务 (0.8s)


并发执行（3.0 秒）：

0s        1s        2s        3s
├─────────┼─────────┼─────────┤
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│           用户服务 (2.0s)
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│                 订单服务 (1.5s)
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  商品服务 (3.0s)
│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│                                 推荐服务 (0.8s)
```

<div data-component="ConcurrencyRequirements"></div>

### 并发的三个必要条件

要实现真正的并发，必须同时满足三个条件：

1. **任务必须是 I/O 密集型的**：并发的好处来自"等待 I/O 时可以做别的事"。CPU 密集型任务没有等待时间可以重叠。

2. **必须使用异步 I/O 操作**：`await asyncio.sleep()` 而不是 `time.sleep()`，`aiohttp` 而不是 `requests`。同步阻塞会卡住事件循环。

3. **必须同时启动多个 Task**：用 `create_task()` 或 `gather()`，而不是一个接一个地 `await`。顺序 await 就是顺序执行。

### 条件缺失时会发生什么？

```python
# ❌ 缺少条件 1：CPU 密集型任务，无法并发
async def cpu_heavy(n: int) -> int:
    return sum(i * i for i in range(n))

# 即使用 gather，两个 CPU 密集型任务也不会真正并发
# 因为它们都在同一个线程中运行，无法重叠
await asyncio.gather(cpu_heavy(10**7), cpu_heavy(10**7))


# ❌ 缺少条件 2：使用同步阻塞操作
async def bad_download(url: str) -> bytes:
    import requests
    return requests.get(url).content   # 同步阻塞！

# 即使用 gather，requests.get 也会卡住事件循环
await asyncio.gather(bad_download(url1), bad_download(url2))


# ❌ 缺少条件 3：顺序 await
async def sequential_gather() -> None:
    # 这不是并发！每个 await 都在等前一个完成
    a = await fetch_a()
    b = await fetch_b()
    c = await fetch_c()
```

### 快速诊断清单

当你的异步代码"好像没有并发"时，按这个清单排查：

```
□ 检查是否用了 create_task() 或 gather()
  → 顺序 await 不是并发

□ 检查是否用了同步阻塞操作
  → time.sleep, requests.get, open().read() 等

□ 检查任务是否是 I/O 密集型
  → CPU 密集型任务无法在单线程中并发

□ 检查总耗时是否等于 max(各任务耗时)
  → 如果等于 sum，说明没有真正并发

□ 检查是否有异常被静默吞掉
  → gather 默认会抛出第一个异常，可能中断其他任务
```

### 真实项目中的并发模式

在实际项目中，最常用的并发模式是"扇出-汇聚"（Fan-Out / Fan-In）：

```python
async def fan_out_fan_in() -> None:
    """
    扇出：同时发起多个独立请求
    汇聚：等待所有结果，合并处理
    """

    # ── 扇出（Fan-Out）：同时发起所有请求 ──
    user_task = asyncio.create_task(fetch_user(123))
    orders_task = asyncio.create_task(fetch_orders(123))
    products_task = asyncio.create_task(fetch_products([1, 2, 3]))
    config_task = asyncio.create_task(fetch_config())

    # ── 汇聚（Fan-In）：等待所有结果 ──
    user, orders, products, config = await asyncio.gather(
        user_task, orders_task, products_task, config_task
    )

    # ── 合并处理 ──
    return render_page(user, orders, products, config)
```

或者更简洁的写法：

```python
async def fan_out_fan_in_concise() -> None:
    user, orders, products, config = await asyncio.gather(
        fetch_user(123),
        fetch_orders(123),
        fetch_products([1, 2, 3]),
        fetch_config(),
    )
    return render_page(user, orders, products, config)
```
</div>

---

## 3.14 常见误区

### 误区 1：await 了就是并发

```python
# ❌ 错误：以为这样就能并发
result_a = await fetch_a()
result_b = await fetch_b()

# ✅ 正确：这才是并发
task_a = asyncio.create_task(fetch_a())
task_b = asyncio.create_task(fetch_b())
result_a = await task_a
result_b = await task_b

# ✅ 或者更简洁
result_a, result_b = await asyncio.gather(fetch_a(), fetch_b())
```

**理解要点**：`await` 是"等待完成"，不是"启动执行"。顺序 `await` 就是顺序执行。

### 误区 2：create_task 会立刻执行

```python
# ❌ 错误：以为 create_task 后 Task 就在执行了
task = asyncio.create_task(slow_task())
print("Task 已经在执行了")   # 不一定！当前协程还没让出控制权

# ✅ 正确：理解 Task 只是被安排了
task = asyncio.create_task(slow_task())
print("Task 已被安排，但可能还没开始")
await asyncio.sleep(0)   # 让出控制权，Task 才有机会开始
print("现在 Task 可能已经开始执行了")
```

**理解要点**：`create_task()` 只是"安排"，Task 要等到当前协程让出控制权后才开始执行。

### 误区 3：所有代码都能并发

```python
# ❌ 错误：以为任何代码放到 Task 里就能并发
async def cpu_heavy(n: int) -> int:
    total = 0
    for i in range(n):
        total += i * i
    return total

# 这两个 Task 不能真正并发（因为是 CPU 密集型）
task_a = asyncio.create_task(cpu_heavy(10_000_000))
task_b = asyncio.create_task(cpu_heavy(10_000_000))
await asyncio.gather(task_a, task_b)

# ✅ 正确：CPU 密集型任务应该用多进程
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor() as pool:
    loop = asyncio.get_event_loop()
    result_a = loop.run_in_executor(pool, cpu_heavy, 10_000_000)
    result_b = loop.run_in_executor(pool, cpu_heavy, 10_000_000)
```

**理解要点**：asyncio 的并发是 I/O 并发，不是 CPU 并发。CPU 密集型任务需要多进程。

### 误区 4：忘记等待 Task

```python
# ❌ 错误：创建了 Task 但没有 await
async def main() -> None:
    task = asyncio.create_task(some_work())
    # 忘记 await task
    print("done")   # main 结束后，task 被取消

# ✅ 正确：确保等待所有 Task
async def main() -> None:
    task = asyncio.create_task(some_work())
    result = await task   # 显式等待
    print(f"done: {result}")
```

**理解要点**：未等待的 Task 在事件循环关闭时会被取消。永远不要创建 Task 后丢弃它。

### 误区 5：gather 的结果顺序

```python
# ❌ 错误：以为 gather 按完成顺序返回
results = await asyncio.gather(
    slow_task(),   # 最后完成
    fast_task(),   # 最先完成
)
# results[0] 是 slow_task 的结果，不是 fast_task 的

# ✅ 正确：记住 gather 按传入顺序返回
slow_result, fast_result = await asyncio.gather(
    slow_task(),
    fast_task(),
)
```

**理解要点**：`gather()` 保证结果顺序与传入顺序一致，与完成顺序无关。

### 误区 6：time.sleep 混用

```python
# ❌ 错误：在异步代码中使用 time.sleep
async def bad_example() -> None:
    time.sleep(5)   # 阻塞整个事件循环！

# ✅ 正确：使用 asyncio.sleep
async def good_example() -> None:
    await asyncio.sleep(5)   # 只暂停当前协程
```

**理解要点**：在 asyncio 中，所有"等待"操作都必须使用异步版本。同步阻塞会卡住事件循环，让并发失效。

### 误区 7：认为 create_task 比 gather 高级

```python
# ❌ 错误：以为 create_task 更"底层"所以更好
tasks = [asyncio.create_task(work(i)) for i in range(10)]
results = []
for task in tasks:
    results.append(await task)

# ✅ 正确：简单场景直接用 gather
results = await asyncio.gather(*[work(i) for i in range(10)])
```

**理解要点**：`gather()` 是 `create_task()` 的封装，简单场景用 `gather()` 更简洁，需要精细控制时才用 `create_task()`。

### 误区 8：以为 gather 会处理异常

```python
# ❌ 错误：以为 gather 会"吞掉"异常继续执行
results = await asyncio.gather(
    work("A"),           # 成功
    work("B", fail=True), # 失败！
    work("C"),           # 不会执行到这里的结果
)
# 实际上：异常会被重新抛出，results 拿不到

# ✅ 正确：明确处理异常
results = await asyncio.gather(
    work("A"),
    work("B", fail=True),
    work("C"),
    return_exceptions=True,   # 异常作为结果返回
)
for i, r in enumerate(results):
    if isinstance(r, Exception):
        print(f"任务 {i} 失败: {r}")
    else:
        print(f"任务 {i} 成功: {r}")
```

**理解要点**：默认情况下 `gather()` 遇到异常会重新抛出。用 `return_exceptions=True` 可以拿到所有结果（包括异常）。

### 误区 9：在非 async 上下文中调用 create_task

```python
# ❌ 错误：在同步函数中调用 create_task
def setup_tasks():
    task = asyncio.create_task(background_work())  # RuntimeError!

# ✅ 正确：确保在事件循环运行时调用
async def setup_tasks():
    task = asyncio.create_task(background_work())
    return task

# 或者在 asyncio.run() 内部
asyncio.run(setup_tasks())
```

**理解要点**：`create_task()` 需要一个正在运行的事件循环。它只能在 `async def` 函数内部调用。

### 误区 10：以为并发一定更快

```python
# 如果任务本身很快（比如 < 1ms），并发的开销可能大于收益
async def tiny_task() -> int:
    return 42

# 并发版本反而更慢（Task 创建和调度有开销）
results = await asyncio.gather(*[tiny_task() for _ in range(1000)])

# 对于非常快的操作，直接顺序执行可能更好
results = [await tiny_task() for _ in range(1000)]
```

**理解要点**：并发有开销（Task 创建、调度、上下文切换）。对于极快的操作，这个开销可能比任务本身还大。只在 I/O 等待时间显著时才使用并发。

---

## 本章小结

本章我们学习了如何让多个协程真正并发执行。核心要点：

1. **顺序 await 是顺序执行**：`await A; await B; await C` 的总耗时 = `delay_A + delay_B + delay_C`

2. **create_task() 实现并发**：把协程包装成 Task，注册到事件循环，立刻返回。多个 Task 同时推进，总耗时 = `max(delays)`

3. **Task 不会立刻执行**：Task 被"安排"后，要等到当前协程让出控制权（遇到 `await`）才开始执行

4. **sleep(0) 主动让出**：`await asyncio.sleep(0)` 可以在没有自然 `await` 点时，给事件循环一个调度其他 Task 的机会

5. **gather() 简化并发**：一步完成"创建 Task + 等待全部 + 收集结果"，按传入顺序返回结果

6. **协程 vs Task**：协程是"描述"（惰性），Task 是"已安排"（被事件循环管理）

7. **不要遗忘 Task**：未等待的 Task 会被取消。要么 `gather()` 一步到位，要么收集到列表统一 `await`

8. **Task 不会魔法般异步化**：`time.sleep()` 依然阻塞事件循环，必须用 `asyncio.sleep()` 替代

9. **并发的三个条件**：I/O 密集型 + 异步操作 + 同时启动多个 Task

---

## 思考题

1. **基础理解**：以下代码的总耗时是多少？为什么？

   ```python
   async def main() -> None:
       await asyncio.sleep(3)
       await asyncio.sleep(2)
       await asyncio.sleep(1)
   ```

2. **create_task 时机**：以下代码中，`background` 什么时候开始执行？为什么？

   ```python
   async def main() -> None:
       task = asyncio.create_task(background())
       print("一行")
       print("两行")
       print("三行")
       await task
   ```

3. **gather 结果顺序**：以下代码中，`results` 的值是什么？

   ```python
   async def fast() -> str:
       await asyncio.sleep(1)
       return "fast"

   async def slow() -> str:
       await asyncio.sleep(3)
       return "slow"

   async def main() -> None:
       results = await asyncio.gather(slow(), fast())
       print(results)   # ???
   ```

4. **阻塞陷阱**：以下代码能否实现并发？为什么？如何修复？

   ```python
   async def download(url: str) -> bytes:
       import requests
       response = requests.get(url)
       return response.content

   async def main() -> None:
       results = await asyncio.gather(
           download("https://a.com"),
           download("https://b.com"),
       )
   ```

5. **遗忘 Task**：以下代码有什么问题？`some_work()` 的结果会被使用吗？

   ```python
   async def main() -> None:
       task = asyncio.create_task(some_work())
       print("主流程完成")
   ```

6. **设计思考**：如果你需要并发获取 100 个 URL 的内容，用 `create_task()` 还是 `gather()` 更合适？如果其中某些请求失败了，你希望其他请求继续完成，应该怎么做？

7. **进阶理解**：为什么 asyncio 的并发是"协作式"的，而不是"抢占式"的？这对你的代码编写有什么影响？

8. **性能分析**：假设你有 10 个 API 调用，每个耗时 1 秒。顺序执行需要 10 秒，`gather()` 并发执行需要约 1 秒。如果某个 API 调用需要 5 秒，其他 9 个各需要 1 秒，顺序和并发分别需要多少秒？

9. **sleep(0) 的价值**：在什么场景下，`await asyncio.sleep(0)` 是必要的？如果不使用它，可能会出现什么问题？

10. **综合应用**：设计一个函数 `fetch_with_retry(url, retries=3)`，它并发地对同一个 URL 发起最多 3 次请求，只要有一个成功就返回结果。使用 `create_task()` 和 `asyncio.wait()` 来实现（提示：可以参考后续章节的 `wait()` 用法）。

11. **阻塞检测**：如何判断你的异步代码中是否有"隐藏的"同步阻塞操作？有哪些工具或方法可以帮助你检测？

12. **并发数控制**：如果你需要并发获取 10000 个 URL，但不想同时发起 10000 个请求（会耗尽资源），你会怎么设计？（提示：信号量 Semaphore，可以参考后续章节）

13. **异常传播**：以下代码中，如果 `fetch_b()` 抛出异常，`fetch_a()` 和 `fetch_c()` 的结果会怎样？

    ```python
    results = await asyncio.gather(
        fetch_a(),
        fetch_b(),   # 抛出异常
        fetch_c(),
    )
    ```

    如果改成 `return_exceptions=True` 呢？

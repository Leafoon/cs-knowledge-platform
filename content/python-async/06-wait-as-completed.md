---
title: "asyncio.wait() 与 as_completed()"
description: "掌握 asyncio.wait() 的三种 return_when 策略和 as_completed() 的按完成顺序迭代，学会在部分完成、异常处理、超时控制等场景中灵活选择并发等待方式"
updated: "2026-07-07"
---

# asyncio.wait() 与 as_completed()

> **学习目标**：
> - 理解 `gather()` 的"全有或全无"局限性，认识更灵活的等待需求
> - 掌握 `asyncio.wait()` 的三种 `return_when` 策略及其适用场景
> - 学会使用 `as_completed()` 按完成顺序处理结果
> - 能够区分 `wait()`、`gather()`、`as_completed()` 的特点并正确选型
> - 掌握超时控制和 pending 任务的处理技巧
> - 能在实际项目（爬虫、API 聚合器）中应用这些工具

在前面的章节中，我们学习了 `asyncio.gather()` 来并发执行多个协程。`gather()` 简单好用，但它有一个根本性的问题：**必须等所有任务全部完成才能拿到结果**。

想象你同时点了 10 道菜，其中 3 道已经做好了，但服务员坚持要等 10 道全部做好才一起端上来——菜都凉了。我们需要更灵活的方式：**谁先好就先吃谁**。

本章将介绍 `asyncio.wait()` 和 `as_completed()` 这两个更强大的并发控制工具。

---

## 6.1 gather() 的局限

<div data-component="WaitVsGatherComparison"></div>

在深入新工具之前，我们先搞清楚 `gather()` 到底有什么不足。

### gather() 的"全有或全无"

`asyncio.gather()` 的行为模式很简单：**要么全部成功，要么遇到第一个异常就（默认）抛出**。

```python
import asyncio


async def fetch_user(user_id: int) -> dict:
    """模拟获取用户信息"""
    delays = {1: 1, 2: 3, 3: 0.5, 4: 2, 5: 1.5}
    delay = delays.get(user_id, 1)
    await asyncio.sleep(delay)
    if user_id == 4:
        raise ConnectionError(f"用户 {user_id} 的数据源不可用")
    return {"id": user_id, "name": f"用户{user_id}", "delay": delay}


async def main():
    # gather() 会等待所有任务完成
    results = await asyncio.gather(
        fetch_user(1),  # 1 秒完成
        fetch_user(2),  # 3 秒完成
        fetch_user(3),  # 0.5 秒完成
        fetch_user(4),  # 2 秒后抛异常
        fetch_user(5),  # 1.5 秒完成
    )
    print(results)


asyncio.run(main())
```

输出：

```
Traceback (most recent call last):
  ...
ConnectionError: 用户 4 的数据源不可用
```

**问题来了：**

```
时间线：
0.0s ──── 启动 5 个任务 ────────────────────────────────────────────►
0.5s ──── 任务3 完成 ✓ ────────────────────────────────────────────►
1.0s ──── 任务1 完成 ✓ ────────────────────────────────────────────►
1.5s ──── 任务5 完成 ✓ ────────────────────────────────────────────►
2.0s ──── 任务4 异常 ✗ ────────────────────────────────────────────►
3.0s ──── 任务2 完成 ✓ ────────────────────────────────────────────►

gather() 的行为：
  - 任务 3、1、5 已经完成了，它们的结果呢？
  - 任务 4 抛了异常，gather() 直接把异常往上抛
  - 任务 2 的结果也拿不到了
  - 已经完成的 3 个任务的结果全部丢失！
```

### gather() 的三大局限

```
┌─────────────────────────────────────────────────────────────────────┐
│                   gather() 的三大局限                                │
│                                                                     │
│  局限 1：全有或全无                                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  10 个任务，9 个成功，1 个失败                                │   │
│  │  gather() 默认行为：抛出异常，9 个成功结果全部丢失             │   │
│  │  即使 return_exceptions=True，也要等全部完成才能处理          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  局限 2：无法按完成顺序处理                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  任务 C 在 0.5 秒就完成了                                    │   │
│  │  但 gather() 要等最慢的任务完成后，才按提交顺序返回结果        │   │
│  │  无法在任务完成的"第一时间"就开始处理                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  局限 3：无法精细控制等待策略                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  "只要有一个成功就够了" → 做不到                              │   │
│  │  "有一个失败就停止"    → 做不到（需要额外逻辑）                │   │
│  │  "最多等 3 秒"         → 做不到（需要 asyncio.wait_for）     │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### gather() 的返回顺序

`gather()` 的结果**严格按照参数顺序返回**，而不是按完成顺序：

```python
import asyncio
import time


async def slow_task(name: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return f"{name} 完成"


async def main():
    start = time.time()

    results = await asyncio.gather(
        slow_task("A", 2.0),   # 最慢
        slow_task("B", 0.5),   # 最快
        slow_task("C", 1.0),   # 中等
    )

    elapsed = time.time() - start
    print(f"总耗时: {elapsed:.1f}s")
    print(f"结果: {results}")
    # 结果按提交顺序排列，不是完成顺序！


asyncio.run(main())
```

输出：

```
总耗时: 2.0s
结果: ['A 完成', 'B 完成', 'C 完成']
```

B 先完成，C 次之，A 最后。但结果列表里还是 A、B、C 的顺序。如果我们想**谁先完成就先处理谁**，`gather()` 做不到。

### 什么时候需要更灵活的工具？

```
┌─────────────────────────────────────────────────────────────────────┐
│              需要 wait() / as_completed() 的场景                     │
│                                                                     │
│  ✓ 只要一个成功结果就够了（竞速请求）                                 │
│  ✓ 想在第一个任务完成时就开始处理，而不是等全部完成                    │
│  ✓ 需要区分"成功"和"失败"，分别处理                                  │
│  ✓ 需要设置超时，超时后取消未完成的任务                               │
│  ✓ 有大量任务，想分批处理结果                                        │
│  ✓ 某个任务失败后，不想影响其他已完成任务的结果收集                    │
│                                                                     │
│  gather() 仍然适合的场景：                                           │
│  ✗ 所有任务必须全部成功，缺一不可                                    │
│  ✗ 任务数量不多，异常概率低                                          │
│  ✗ 代码简洁性优先，不需要精细控制                                     │
└─────────────────────────────────────────────────────────────────────┘
```
</div>

---

## 6.2 asyncio.wait() 基础

`asyncio.wait()` 是 Python 标准库提供的更灵活的并发等待工具。它的核心特点是可以**按条件决定何时返回**。

### 基本语法

```python
done, pending = await asyncio.wait(
    aws,                    # 可等待对象的集合
    return_when=...,        # 返回条件（核心参数）
    timeout=None,           # 超时时间（秒）
)
```

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `aws` | Iterable[Awaitable] | 一组可等待对象（协程、Task、Future） |
| `return_when` | str | 决定何时返回的条件（见下文） |
| `timeout` | float or None | 超时秒数，None 表示不限时 |

**返回值：** 一个元组 `(done, pending)`，两个都是 **set** 类型。

### return_when 的三个选项

```python
import asyncio

# 三种返回策略
asyncio.FIRST_COMPLETED   # 任意一个任务完成就返回
asyncio.FIRST_EXCEPTION   # 有异常发生时返回（没有异常则等全部完成）
asyncio.ALL_COMPLETED     # 等所有任务完成再返回（默认值）
```

```
┌─────────────────────────────────────────────────────────────────────┐
│                   return_when 三种策略对比                            │
│                                                                     │
│  策略               │ 何时返回              │ 使用场景               │
│  ──────────────────┼──────────────────────┼────────────────────────│
│  FIRST_COMPLETED   │ 第一个任务完成时       │ 竞速、快速响应          │
│  FIRST_EXCEPTION   │ 第一个异常发生时       │ 错误监控、快速失败      │
│  ALL_COMPLETED     │ 所有任务完成时         │ 全量结果收集（默认）    │
│                                                                     │
│  注意：FIRST_EXCEPTION 在没有异常时，行为等同于 ALL_COMPLETED        │
└─────────────────────────────────────────────────────────────────────┘
```

### wait() 需要 Task，不能直接传协程

这是一个重要的细节：`asyncio.wait()` 在 Python 3.11+ 中**推荐传入 Task 对象**，而不是裸协程。

```python
import asyncio


async def fetch(url: str) -> str:
    await asyncio.sleep(0.5)
    return f"来自 {url} 的数据"


async def main():
    # ⚠️ Python 3.11+ 中传裸协程会发出 DeprecationWarning
    # coros = [fetch("http://a.com"), fetch("http://b.com")]
    # done, pending = await asyncio.wait(coros)  # 不推荐

    # ✓ 正确做法：先包装成 Task
    tasks = [
        asyncio.create_task(fetch("http://a.com")),
        asyncio.create_task(fetch("http://b.com")),
    ]
    done, pending = await asyncio.wait(tasks)

    for task in done:
        print(task.result())


asyncio.run(main())
```

**为什么要这样设计？**

```
协程 vs Task：
  协程：只是一个"待执行的函数调用"，还没有被事件循环管理
  Task：已被事件循环调度，可以被取消、查询状态、等待完成

  asyncio.wait() 需要：
  - 查询每个任务的状态（是否完成）
  - 可能需要取消未完成的任务
  - 需要在事件循环中注册回调

  这些操作只有 Task 才支持，协程不行。
```

### 一个最小示例

```python
import asyncio


async def work(name: str, delay: float) -> str:
    print(f"  任务 {name} 开始，预计 {delay}s")
    await asyncio.sleep(delay)
    print(f"  任务 {name} 完成！")
    return f"{name} 的结果"


async def main():
    tasks = {
        asyncio.create_task(work("A", 2.0)),
        asyncio.create_task(work("B", 1.0)),
        asyncio.create_task(work("C", 0.5)),
    }

    print("等待所有任务完成...")
    done, pending = await asyncio.wait(tasks)

    print(f"\n完成的任务: {len(done)} 个")
    print(f"未完成的任务: {len(pending)} 个")

    for task in done:
        print(f"  结果: {task.result()}")


asyncio.run(main())
```

输出：

```
等待所有任务完成...
  任务 A 开始，预计 2.0s
  任务 B 开始，预计 1.0s
  任务 C 开始，预计 0.5s
  任务 C 完成！
  任务 B 完成！
  任务 A 完成！

完成的任务: 3 个
未完成的任务: 0 个
  结果: C 的结果
  结果: B 的结果
  结果: A 的结果
```

注意结果的打印顺序是随机的（set 是无序的），但三个任务确实并发执行了。

---

## 6.3 FIRST_COMPLETED — 有一个完成就返回

<div data-component="FirstCompletedDemo"></div>

`FIRST_COMPLETED` 是最灵活的策略：**只要有一个任务完成（成功或失败），立即返回**。

### 基本用法

```python
import asyncio


async def fetch_from_source(name: str, delay: float) -> str:
    """从不同数据源获取数据，延迟不同"""
    print(f"  [{name}] 开始请求...")
    await asyncio.sleep(delay)
    print(f"  [{name}] 收到响应（耗时 {delay}s）")
    return f"{name}: 数据内容"


async def main():
    # 同时请求三个数据源，只要有一个返回就用那个
    tasks = [
        asyncio.create_task(fetch_from_source("主服务器", 2.0)),
        asyncio.create_task(fetch_from_source("CDN-1", 0.8)),
        asyncio.create_task(fetch_from_source("CDN-2", 1.2)),
    ]

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    # done 里只有一个任务（最先完成的那个）
    winner = done.pop()
    print(f"\n最快的数据源: {winner.result()}")

    # pending 里还有两个没完成的任务
    print(f"还有 {len(pending)} 个任务未完成，需要处理")


asyncio.run(main())
```

输出：

```
  [主服务器] 开始请求...
  [CDN-1] 开始请求...
  [CDN-2] 开始请求...
  [CDN-1] 收到响应（耗时 0.8s）

最快的数据源: CDN-1: 数据内容
还有 2 个任务未完成，需要处理
```

### 竞速请求模式

FIRST_COMPLETED 最经典的用途是**竞速请求**：同时请求多个镜像源，用最快返回的那个。

```python
import asyncio
import random


async def mirror_request(mirror: str) -> str:
    """模拟向镜像服务器发送请求"""
    delay = random.uniform(0.5, 3.0)
    await asyncio.sleep(delay)

    # 模拟某些镜像可能失败
    if random.random() < 0.2:
        raise ConnectionError(f"{mirror} 不可用")

    return f"来自 {mirror} 的数据 (耗时 {delay:.2f}s)"


async def race_requests(mirrors: list[str]) -> str:
    """竞速请求：用最快成功的镜像"""
    tasks = [asyncio.create_task(mirror_request(m)) for m in mirrors]

    while tasks:
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            try:
                result = task.result()
                # 取消剩余任务
                for p in pending:
                    p.cancel()
                return result
            except Exception as e:
                print(f"  一个镜像失败: {e}")
                # 继续等剩余任务
                tasks = list(pending)
                break

    raise RuntimeError("所有镜像都失败了")


async def main():
    mirrors = ["镜像A", "镜像B", "镜像C", "镜像D"]
    try:
        result = await race_requests(mirrors)
        print(f"最终结果: {result}")
    except RuntimeError as e:
        print(f"错误: {e}")


asyncio.run(main())
```

### 竞速模式的优化版本

上面的代码有个问题：如果第一个完成的任务恰好失败了，我们又回到了等待循环。更高效的写法是一次性等所有完成/失败的最终结果：

```python
import asyncio


async def fast_mirror(mirror: str) -> str:
    delays = {"CloudFlare": 0.3, "AWS": 0.8, "阿里云": 0.5}
    delay = delays.get(mirror, 1.0)
    await asyncio.sleep(delay)
    return f"来自 {mirror} 的数据"


async def main():
    mirrors = ["CloudFlare", "AWS", "阿里云"]
    tasks = [asyncio.create_task(fast_mirror(m)) for m in mirrors]

    # 只要第一个完成的
    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_COMPLETED
    )

    # 拿到最快的结果
    fastest = done.pop()
    print(f"最快结果: {fastest.result()}")

    # 取消剩余任务（重要！否则它们会继续运行占用资源）
    for task in pending:
        task.cancel()

    # 可选：等待取消操作完成
    if pending:
        await asyncio.wait(pending)
        print(f"已取消 {len(pending)} 个未完成的任务")


asyncio.run(main())
```

输出：

```
最快结果: 来自 CloudFlare 的数据
已取消 2 个未完成的任务
```

### 不要忘记取消 pending 任务

这是一个**关键的注意事项**：

```python
# ⚠️ 错误示范：不取消 pending 任务
done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
result = done.pop().result()
# pending 里的任务还在运行！它们会继续占用资源
# 直到 main() 返回，事件循环关闭，这些任务才会被清理

# ✓ 正确做法：取消 pending 任务
for task in pending:
    task.cancel()
```

```
┌─────────────────────────────────────────────────────────────────────┐
│                   为什么要取消 pending 任务？                         │
│                                                                     │
│  1. 资源浪费：任务继续运行，占用网络连接、内存等资源                   │
│  2. 副作用：任务可能修改文件、发送请求、写数据库                       │
│  3. 泄漏：大量未取消的任务积累，可能导致内存泄漏                       │
│  4. 关闭延迟：事件循环关闭时要等所有任务结束                           │
│                                                                     │
│  规则：用完 wait(FIRST_COMPLETED) 后，永远要取消 pending 任务         │
└─────────────────────────────────────────────────────────────────────┘
```
</div>

---

## 6.4 FIRST_EXCEPTION — 第一个异常就返回

`FIRST_EXCEPTION` 策略的行为是：**当有任务抛出异常时立即返回，如果没有异常则等所有任务完成**。

### 基本用法

```python
import asyncio


async def risky_fetch(name: str, should_fail: bool) -> str:
    await asyncio.sleep(0.5)
    if should_fail:
        raise ValueError(f"{name} 出错了！")
    return f"{name}: 成功"


async def main():
    tasks = [
        asyncio.create_task(risky_fetch("A", should_fail=False)),
        asyncio.create_task(risky_fetch("B", should_fail=True)),   # 会失败
        asyncio.create_task(risky_fetch("C", should_fail=False)),
    ]

    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_EXCEPTION
    )

    print(f"完成的任务数: {len(done)}")
    print(f"未完成的任务数: {len(pending)}")

    for task in done:
        if task.exception():
            print(f"  异常: {task.exception()}")
        else:
            print(f"  成功: {task.result()}")


asyncio.run(main())
```

输出：

```
完成的任务数: 3
未完成的任务数: 0
  异常: B 出错了！
  成功: A: 成功
  成功: C: 成功
```

等一下——为什么 done 里有 3 个？因为 0.5 秒内 A、B、C 都完成了，B 抛了异常，但 A 和 C 也在差不多同一时间完成了。

### FIRST_EXCEPTION 的精确行为

```
场景 1：有异常，且异常先于其他任务完成
  ─────────────────────────────────────────────────────────► 时间
  任务 A: ════════════════════════════════════════════════ (3s)
  任务 B: ══════✗ (0.5s 异常)
  任务 C: ═══════════════ (1.5s)

  wait(FIRST_EXCEPTION) 在 0.5s 返回
  done = {B (异常)}
  pending = {A, C}


场景 2：有异常，但正常任务先完成
  ─────────────────────────────────────────────────────────► 时间
  任务 A: ═══ (0.3s 完成)
  任务 B: ══════════════════✗ (1s 异常)
  任务 C: ═══════ (0.5s 完成)

  wait(FIRST_EXCEPTION) 在 1s 返回（等异常出现）
  done = {A, B, C}（全部完成）


场景 3：没有异常
  ─────────────────────────────────────────────────────────► 时间
  任务 A: ═══ (0.3s)
  任务 B: ═══════ (0.5s)
  任务 C: ═══════════ (0.8s)

  wait(FIRST_EXCEPTION) 在 0.8s 返回（等全部完成）
  done = {A, B, C}
  pending = {}
```

### 用 FIRST_EXCEPTION 做错误监控

```python
import asyncio


async def process_batch(batch_id: int, items: list[int]) -> list[int]:
    """处理一批数据，某些数据可能出错"""
    results = []
    for item in items:
        if item < 0:
            raise ValueError(f"批次 {batch_id}: 遇到非法值 {item}")
        await asyncio.sleep(0.1)  # 模拟处理
        results.append(item * 2)
    return results


async def main():
    batches = [
        [1, 2, 3, 4, 5],
        [10, 20, 30],
        [100, -1, 200],  # 这个批次有非法值
        [7, 8, 9],
    ]

    tasks = [
        asyncio.create_task(process_batch(i, batch))
        for i, batch in enumerate(batches)
    ]

    done, pending = await asyncio.wait(
        tasks,
        return_when=asyncio.FIRST_EXCEPTION
    )

    # 检查是否有异常
    errors = [t for t in done if t.exception()]
    successes = [t for t in done if not t.exception()]

    if errors:
        print(f"发现 {len(errors)} 个错误:")
        for t in errors:
            print(f"  {t.exception()}")

        # 取消还在处理中的批次
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.wait(pending)
            print(f"已取消 {len(pending)} 个未完成的批次")
    else:
        print("所有批次都成功了")
        for t in successes:
            print(f"  结果: {t.result()}")


asyncio.run(main())
```

输出：

```
发现 1 个错误:
  批次 2: 遇到非法值 -1
已取消 1 个未完成的批次
```

### FIRST_EXCEPTION vs gather(return_exceptions=True)

```python
# gather + return_exceptions：等所有完成，异常作为值返回
results = await asyncio.gather(*tasks, return_exceptions=True)
for r in results:
    if isinstance(r, Exception):
        print(f"异常: {r}")
    else:
        print(f"成功: {r}")
# 必须等所有任务完成

# wait + FIRST_EXCEPTION：出现异常就返回，不用等所有完成
done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
# 可以更早开始处理错误
```

```
┌─────────────────────────────────────────────────────────────────────┐
│          FIRST_EXCEPTION vs gather(return_exceptions=True)           │
│                                                                     │
│  gather(return_exceptions=True):                                    │
│    - 等所有任务完成                                                  │
│    - 异常作为结果值返回，不抛出                                       │
│    - 结果保持原始顺序                                                │
│    - 无法在第一个异常时就停止等待                                     │
│                                                                     │
│  wait(FIRST_EXCEPTION):                                            │
│    - 出现异常就返回（或全部完成时返回）                               │
│    - 需要手动检查 task.exception()                                   │
│    - 结果是无序的 set                                                │
│    - 可以更早处理错误、取消剩余任务                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6.5 ALL_COMPLETED — 等待全部完成

`ALL_COMPLETED` 是 `wait()` 的默认行为，等同于 `asyncio.wait(tasks)` 不传 `return_when` 参数。

### 基本用法

```python
import asyncio


async def task(name: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return f"{name}: 完成"


async def main():
    tasks = [
        asyncio.create_task(task("A", 1.0)),
        asyncio.create_task(task("B", 0.5)),
        asyncio.create_task(task("C", 0.8)),
    ]

    # 默认就是 ALL_COMPLETED
    done, pending = await asyncio.wait(tasks)

    print(f"完成: {len(done)}, 未完成: {len(pending)}")
    for t in done:
        print(f"  {t.result()}")


asyncio.run(main())
```

### ALL_COMPLETED 与 gather() 的区别

既然 ALL_COMPLETED 也是等全部完成，它和 `gather()` 有什么区别？

```python
import asyncio


async def fetch(url: str) -> str:
    await asyncio.sleep(0.5)
    return f"来自 {url}"


async def main():
    urls = ["http://a.com", "http://b.com", "http://c.com"]

    # gather() 方式
    results = await asyncio.gather(*[fetch(u) for u in urls])
    print("gather 结果:", results)
    # 结果保持顺序: ['来自 http://a.com', '来自 http://b.com', '来自 http://c.com']

    # wait() 方式
    tasks = [asyncio.create_task(fetch(u)) for u in urls]
    done, pending = await asyncio.wait(tasks)
    wait_results = [t.result() for t in done]
    print("wait 结果:", wait_results)
    # 结果顺序不确定（set 是无序的）


asyncio.run(main())
```

```
┌─────────────────────────────────────────────────────────────────────┐
│            ALL_COMPLETED vs gather() 的核心区别                      │
│                                                                     │
│  特性              │ gather()              │ wait(ALL_COMPLETED)    │
│  ─────────────────┼───────────────────────┼────────────────────────│
│  返回类型          │ list（有序）           │ (done_set, pending_set)│
│  结果顺序          │ 保持参数顺序          │ 无序                   │
│  异常处理          │ 默认立即抛出          │ 存在 task.exception()  │
│  返回对象          │ 直接是结果值          │ 是 Task 对象           │
│  代码简洁度        │ 更简洁                │ 需要手动提取结果        │
│  附加功能          │ 无                    │ 支持 timeout           │
└─────────────────────────────────────────────────────────────────────┘
```

### 保持顺序的 wait() 写法

如果用 `wait()` 但又想保持顺序，需要自己做映射：

```python
import asyncio


async def fetch(url: str) -> str:
    await asyncio.sleep(0.5)
    return f"数据: {url}"


async def main():
    urls = ["http://a.com", "http://b.com", "http://c.com"]

    # 创建 Task 时记住它的来源
    task_to_url = {}
    tasks = []
    for url in urls:
        task = asyncio.create_task(fetch(url))
        task_to_url[task] = url
        tasks.append(task)

    done, pending = await asyncio.wait(tasks)

    # 按原始顺序收集结果
    results = {}
    for task in done:
        url = task_to_url[task]
        results[url] = task.result()

    # 按原始 URL 顺序输出
    ordered_results = [results[url] for url in urls]
    print(ordered_results)


asyncio.run(main())
```

---

## 6.6 wait() 返回值 — (done, pending) 集合

<div data-component="WaitReturnSets"></div>

`wait()` 的返回值是两个 **set**：`(done, pending)`。理解这两个集合的性质很重要。

### done 和 pending 的类型

```python
import asyncio


async def dummy():
    await asyncio.sleep(0.1)
    return 42


async def main():
    task = asyncio.create_task(dummy())
    done, pending = await asyncio.wait([task])

    print(type(done))     # <class 'set'>
    print(type(pending))  # <class 'set'>

    # done 里的元素是 Task 对象
    for t in done:
        print(type(t))              # <class 'asyncio.tasks.Task'>
        print(t.done())             # True
        print(t.result())           # 42
        print(t.exception())        # None


asyncio.run(main())
```

### done 集合中任务的状态

```python
import asyncio


async def success_task():
    await asyncio.sleep(0.1)
    return "成功"


async def fail_task():
    await asyncio.sleep(0.1)
    raise ValueError("失败了")


async def main():
    t1 = asyncio.create_task(success_task())
    t2 = asyncio.create_task(fail_task())

    done, _ = await asyncio.wait([t1, t2])

    for task in done:
        print(f"任务: {task.get_name()}")
        print(f"  done(): {task.done()}")           # True（已完成）
        print(f"  cancelled(): {task.cancelled()}")  # False（不是被取消的）

        if task.exception():
            print(f"  状态: 失败 - {task.exception()}")
        else:
            print(f"  状态: 成功 - {task.result()}")
        print()


asyncio.run(main())
```

输出：

```
任务: Task-1
  done(): True
  cancelled(): False
  状态: 成功 - 成功

任务: Task-2
  done(): True
  cancelled(): False
  状态: 失败 - 失败了
```

### pending 集合中任务的状态

```python
import asyncio


async def slow_task():
    await asyncio.sleep(10)
    return "慢任务完成"


async def main():
    t1 = asyncio.create_task(slow_task())
    t2 = asyncio.create_task(slow_task())

    # 设置 0.5 秒超时
    done, pending = await asyncio.wait([t1, t2], timeout=0.5)

    print(f"完成: {len(done)}, 未完成: {len(pending)}")

    for task in pending:
        print(f"任务: {task.get_name()}")
        print(f"  done(): {task.done()}")           # False（未完成）
        print(f"  cancelled(): {task.cancelled()}")  # False（未被取消）
        print(f"  状态: 仍在运行")

    # 通常需要取消 pending 任务
    for task in pending:
        task.cancel()

    # 等待取消生效
    await asyncio.wait(pending)

    for task in pending:
        print(f"\n取消后:")
        print(f"  done(): {task.done()}")           # True
        print(f"  cancelled(): {task.cancelled()}")  # True


asyncio.run(main())
```

输出：

```
完成: 0, 未完成: 2
任务: Task-1
  done(): False
  cancelled(): False
  状态: 仍在运行
任务: Task-2
  done(): False
  cancelled(): False
  状态: 仍在运行

取消后:
  done(): True
  cancelled(): True

取消后:
  done(): True
  cancelled(): True
```

### 从 done 集合中提取结果的模式

```python
import asyncio


async def fetch(n: int) -> int:
    await asyncio.sleep(0.1)
    return n * n


async def main():
    tasks = [asyncio.create_task(fetch(i)) for i in range(5)]
    done, _ = await asyncio.wait(tasks)

    # 模式 1：直接遍历结果
    for task in done:
        print(task.result())

    # 模式 2：收集为列表
    results = [t.result() for t in done]

    # 模式 3：分离成功和失败
    successes = [t.result() for t in done if not t.exception()]
    failures = [t.exception() for t in done if t.exception()]

    # 模式 4：用 task.result() 时先检查异常
    for task in done:
        try:
            value = task.result()
            print(f"成功: {value}")
        except Exception as e:
            print(f"失败: {e}")


asyncio.run(main())
```
</div>

---

## 6.7 as_completed() — 按完成顺序迭代

<div data-component="AsCompletedDemo"></div>

`asyncio.as_completed()` 返回一个**可迭代对象**，产生的 Future 按**完成顺序**排列——先完成的先产出。

### 基本语法

```python
for coro in asyncio.as_completed(aws, timeout=None):
    result = await coro
    # 处理结果
```

### 与 wait() 的关键区别

```python
import asyncio
import time


async def work(name: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return f"{name}(耗时{delay}s)"


async def main():
    tasks = [
        asyncio.create_task(work("A", 2.0)),
        asyncio.create_task(work("B", 0.5)),
        asyncio.create_task(work("C", 1.0)),
    ]

    # as_completed：按完成顺序处理
    start = time.time()
    for coro in asyncio.as_completed(tasks):
        result = await coro
        elapsed = time.time() - start
        print(f"  {elapsed:.1f}s 时收到: {result}")


asyncio.run(main())
```

输出：

```
  0.5s 时收到: B(耗时0.5s)
  1.0s 时收到: C(耗时1.0s)
  2.0s 时收到: A(耗时2.0s)
```

**注意结果是按完成时间排序的，不是按提交顺序！**

```
时间线对比：

gather() 的结果顺序：
  [A 的结果, B 的结果, C 的结果]    ← 按提交顺序

as_completed() 的处理顺序：
  B 的结果 → C 的结果 → A 的结果    ← 按完成顺序
```

### as_completed() 返回的是协程

这是一个容易混淆的点：`as_completed()` 产生的每个元素本身就是一个**协程**，需要 `await` 才能拿到结果。

```python
import asyncio


async def fetch(n: int) -> int:
    await asyncio.sleep(0.1)
    return n * n


async def main():
    tasks = [asyncio.create_task(fetch(i)) for i in range(5)]

    # as_completed 返回的是协程的可迭代对象
    for coro in asyncio.as_completed(tasks):
        # 每个 coro 是一个协程，需要 await
        result = await coro
        print(f"结果: {result}")


asyncio.run(main())
```

```
┌─────────────────────────────────────────────────────────────────────┐
│              as_completed() 的内部机制                               │
│                                                                     │
│  asyncio.as_completed(tasks) 返回一个迭代器                          │
│                                                                     │
│  每次迭代（或 for 循环的每次循环）：                                  │
│    1. 迭代器产生一个协程                                             │
│    2. 你 await 这个协程                                              │
│    3. 如果对应的 task 已完成 → 立即返回结果                           │
│    4. 如果对应的 task 未完成 → 等待它完成再返回                       │
│                                                                     │
│  关键：迭代器按完成顺序产生协程，不是按提交顺序                       │
│                                                                     │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                         │
│  │ 任务 A   │    │ 任务 B   │    │ 任务 C   │                        │
│  │ (2s)     │    │ (0.5s)   │    │ (1s)     │                        │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘                       │
│       │               │               │                             │
│       │          先完成 │               │                             │
│       │               ▼               │                             │
│       │         ┌───────────┐         │                             │
│       │         │ 产出协程 B │         │                             │
│       │         └───────────┘         │                             │
│       │                    次完成 │    │                             │
│       │                         ▼    ▼                             │
│       │                   ┌───────────┐                            │
│       │                   │ 产出协程 C │                            │
│       │                   └───────────┘                            │
│       ▼                                                             │
│  ┌───────────┐                                                      │
│  │ 产出协程 A │  ← 最后完成                                         │
│  └───────────┘                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### as_completed() 传入的参数

和 `wait()` 类似，`as_completed()` 推荐传入 **Task** 对象：

```python
import asyncio


async def compute(n: int) -> int:
    await asyncio.sleep(0.1)
    return n * n


async def main():
    # 推荐：先创建 Task
    tasks = [asyncio.create_task(compute(i)) for i in range(5)]
    for coro in asyncio.as_completed(tasks):
        result = await coro
        print(result)


asyncio.run(main())
```
</div>

---

## 6.8 as_completed() 示例

<div data-component="CompletionOrderVisualizer"></div>

### 示例 1：实时显示进度

```python
import asyncio
import time


async def download_file(filename: str, size_mb: float) -> str:
    """模拟文件下载"""
    # 模拟下载时间与文件大小成正比
    delay = size_mb * 0.1
    await asyncio.sleep(delay)
    return f"{filename} ({size_mb}MB)"


async def main():
    files = [
        ("photo.jpg", 2.0),
        ("video.mp4", 10.0),
        ("doc.pdf", 0.5),
        ("music.mp3", 3.0),
        ("archive.zip", 5.0),
    ]

    tasks = [
        asyncio.create_task(download_file(name, size))
        for name, size in files
    ]

    total = len(tasks)
    completed = 0
    start = time.time()

    for coro in asyncio.as_completed(tasks):
        result = await coro
        completed += 1
        elapsed = time.time() - start
        print(f"  [{elapsed:.1f}s] 完成 ({completed}/{total}): {result}")

    print(f"\n全部完成，总耗时: {time.time() - start:.1f}s")


asyncio.run(main())
```

输出：

```
  [0.1s] 完成 (1/5): doc.pdf (0.5MB)
  [0.2s] 完成 (2/5): photo.jpg (2.0MB)
  [0.3s] 完成 (3/5): music.mp3 (3.0MB)
  [0.5s] 完成 (4/5): archive.zip (5.0MB)
  [1.0s] 完成 (5/5): video.mp4 (10.0MB)

全部完成，总耗时: 1.0s
```

### 示例 2：结果聚合与排序

```python
import asyncio
import random


async def search_engine(name: str, query: str) -> list[str]:
    """模拟搜索引擎返回结果"""
    delay = random.uniform(0.1, 1.0)
    await asyncio.sleep(delay)
    # 模拟不同引擎返回不同结果
    return [f"{name}-结果-{i}" for i in range(3)]


async def main():
    engines = ["Google", "Bing", "DuckDuckGo", "Baidu"]
    query = "Python asyncio"

    tasks = [
        asyncio.create_task(search_engine(engine, query))
        for engine in engines
    ]

    all_results = []
    for coro in asyncio.as_completed(tasks):
        results = await coro
        all_results.extend(results)
        print(f"收到 {len(results)} 条结果，累计 {len(all_results)} 条")

    print(f"\n总共收集到 {len(all_results)} 条搜索结果")


asyncio.run(main())
```

### 示例 3：提前终止

如果只需要前 N 个结果，可以在收集够后取消剩余任务：

```python
import asyncio


async def worker(n: int) -> int:
    await asyncio.sleep(abs(5 - n) * 0.2)  # 不同延迟
    return n * n


async def main():
    tasks = [asyncio.create_task(worker(i)) for i in range(10)]
    needed = 3  # 只需要 3 个结果
    results = []

    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        print(f"收到: {result} (已收集 {len(results)}/{needed})")

        if len(results) >= needed:
            print("已收集够，取消剩余任务")
            for task in tasks:
                if not task.done():
                    task.cancel()
            break

    print(f"\n最终结果: {results}")


asyncio.run(main())
```

注意：`as_completed()` 本身不提供取消机制，需要手动遍历原始 tasks 列表来取消。

### 示例 4：带异常处理的 as_completed()

```python
import asyncio


async def risky_task(n: int) -> int:
    await asyncio.sleep(0.1)
    if n == 3:
        raise ValueError(f"任务 {n} 失败")
    return n * 10


async def main():
    tasks = [asyncio.create_task(risky_task(i)) for i in range(6)]

    successes = []
    failures = []

    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            successes.append(result)
            print(f"  成功: {result}")
        except ValueError as e:
            failures.append(e)
            print(f"  失败: {e}")

    print(f"\n成功: {len(successes)}, 失败: {len(failures)}")
    print(f"成功的结果: {successes}")


asyncio.run(main())
```

输出：

```
  成功: 0
  成功: 10
  成功: 20
  失败: 任务 3 失败
  成功: 40
  成功: 50

成功: 5, 失败: 1
成功的结果: [0, 10, 20, 40, 50]
```

### 示例 5：与 gather 混合使用的对比

```python
import asyncio
import time


async def api_call(name: str, delay: float) -> dict:
    await asyncio.sleep(delay)
    return {"source": name, "data": f"{name}的数据", "latency": delay}


async def main():
    apis = [
        ("快速API", 0.2),
        ("中速API", 0.5),
        ("慢速API", 1.0),
        ("超慢API", 2.0),
    ]

    # 方式 1：gather（等全部完成）
    print("=== gather 方式 ===")
    start = time.time()
    results = await asyncio.gather(*[api_call(n, d) for n, d in apis])
    for r in results:
        t = time.time() - start
        print(f"  {t:.1f}s: {r['source']} ({r['latency']}s)")
    print(f"  总耗时: {time.time() - start:.1f}s")

    # 方式 2：as_completed（按完成顺序）
    print("\n=== as_completed 方式 ===")
    start = time.time()
    tasks = [asyncio.create_task(api_call(n, d)) for n, d in apis]
    for coro in asyncio.as_completed(tasks):
        r = await coro
        t = time.time() - start
        print(f"  {t:.1f}s: {r['source']} ({r['latency']}s)")
    print(f"  总耗时: {time.time() - start:.1f}s")


asyncio.run(main())
```

输出：

```
=== gather 方式 ===
  2.0s: 快速API (0.2s)
  2.0s: 中速API (0.5s)
  2.0s: 慢速API (1.0s)
  2.0s: 超慢API (2.0s)
  总耗时: 2.0s

=== as_completed 方式 ===
  0.2s: 快速API (0.2s)
  0.5s: 中速API (0.5s)
  1.0s: 慢速API (1.0s)
  2.0s: 超慢API (2.0s)
  总耗时: 2.0s
```

总耗时相同，但 `as_completed()` 让你能在**每个任务完成时立即处理**，而不是等到最后。

</div>
---

## 6.9 wait() vs gather() vs as_completed() 对比表

### 核心对比

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    wait() vs gather() vs as_completed()                       │
├──────────────┬──────────────────┬──────────────────┬─────────────────────────┤
│  特性         │  gather()        │  wait()          │  as_completed()         │
├──────────────┼──────────────────┼──────────────────┼─────────────────────────┤
│  返回类型     │  list[T]         │  (done, pending) │  迭代器[协程]            │
│  结果顺序     │  按参数顺序       │  无序(set)        │  按完成顺序              │
│  等待策略     │  全部完成         │  可配置           │  全部完成                │
│  异常处理     │  默认抛出         │  存在 exception() │  await 时抛出            │
│  超时支持     │  需配合 wait_for  │  内置 timeout     │  内置 timeout            │
│  取消支持     │  无内置           │  pending 集合     │  需手动取消              │
│  返回 Task    │  否(返回值)       │  是               │  否(返回值)              │
│  代码简洁度   │  最简洁           │  较繁琐           │  中等                   │
│  适用场景     │  全量结果         │  精细控制         │  逐个处理                │
├──────────────┴──────────────────┴──────────────────┴─────────────────────────┤
│                                                                              │
│  选择口诀：                                                                   │
│    全部都要、顺序重要    →  gather()                                         │
│    谁先好先处理          →  as_completed()                                   │
│    需要精细控制等待条件  →  wait()                                           │
│    竞速、超时、部分完成  →  wait(FIRST_COMPLETED)                            │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 使用场景速查

```python
# 场景 1：并发请求 5 个 API，全部成功才继续
results = await asyncio.gather(*[fetch(url) for url in urls])

# 场景 2：并发请求 5 个 API，哪个先回来就先处理哪个
tasks = [asyncio.create_task(fetch(url)) for url in urls]
for coro in asyncio.as_completed(tasks):
    result = await coro
    process(result)

# 场景 3：并发请求 5 个 API，只要一个成功就够了
tasks = [asyncio.create_task(fetch(url)) for url in urls]
done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
winner = done.pop().result()
for p in pending:
    p.cancel()

# 场景 4：并发请求 5 个 API，有错误立即停止
tasks = [asyncio.create_task(fetch(url)) for url in urls]
done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
errors = [t for t in done if t.exception()]
if errors:
    handle_error(errors[0])

# 场景 5：并发请求 5 个 API，最多等 3 秒
tasks = [asyncio.create_task(fetch(url)) for url in urls]
done, pending = await asyncio.wait(tasks, timeout=3.0)
for p in pending:
    p.cancel()
```

### 异常处理对比

```python
import asyncio


async def sometimes_fails(n: int) -> int:
    await asyncio.sleep(0.1)
    if n == 2:
        raise ValueError(f"任务 {n} 失败")
    return n


async def main():
    # ── gather 默认行为：第一个异常就抛出 ──
    try:
        results = await asyncio.gather(*[sometimes_fails(i) for i in range(5)])
    except ValueError as e:
        print(f"gather 捕获异常: {e}")

    # ── gather + return_exceptions：异常作为值返回 ──
    results = await asyncio.gather(
        *[sometimes_fails(i) for i in range(5)],
        return_exceptions=True
    )
    for r in results:
        if isinstance(r, Exception):
            print(f"  异常: {r}")
        else:
            print(f"  成功: {r}")

    # ── wait：手动检查异常 ──
    tasks = [asyncio.create_task(sometimes_fails(i)) for i in range(5)]
    done, _ = await asyncio.wait(tasks)
    for t in done:
        if t.exception():
            print(f"  wait 异常: {t.exception()}")
        else:
            print(f"  wait 成功: {t.result()}")

    # ── as_completed：try/except 处理 ──
    tasks = [asyncio.create_task(sometimes_fails(i)) for i in range(5)]
    for coro in asyncio.as_completed(tasks):
        try:
            result = await coro
            print(f"  as_completed 成功: {result}")
        except ValueError as e:
            print(f"  as_completed 异常: {e}")


asyncio.run(main())
```

---

## 6.10 超时控制

### wait() 的 timeout 参数

`wait()` 内置了 `timeout` 参数，这是它相比 `gather()` 的一个重要优势：

```python
import asyncio


async def slow_task(name: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return f"{name}: 完成"


async def main():
    tasks = [
        asyncio.create_task(slow_task("快", 0.5)),
        asyncio.create_task(slow_task("中", 2.0)),
        asyncio.create_task(slow_task("慢", 5.0)),
    ]

    # 最多等 1 秒
    done, pending = await asyncio.wait(tasks, timeout=1.0)

    print(f"完成: {len(done)} 个")
    print(f"超时未完成: {len(pending)} 个")

    for t in done:
        print(f"  完成: {t.result()}")
    for t in pending:
        print(f"  超时: {t.get_name()}")

    # 清理 pending 任务
    for t in pending:
        t.cancel()


asyncio.run(main())
```

输出：

```
完成: 1 个
超时未完成: 2 个
  完成: 快: 完成
  超时: Task-2
  超时: Task-3
```

### as_completed() 的 timeout 参数

`as_completed()` 也支持 `timeout`，但它是在**整个迭代过程**上设置超时：

```python
import asyncio


async def slow_task(n: int) -> int:
    await asyncio.sleep(n * 0.5)
    return n


async def main():
    tasks = [asyncio.create_task(slow_task(i)) for i in range(5)]

    try:
        for coro in asyncio.as_completed(tasks, timeout=1.0):
            result = await coro
            print(f"收到: {result}")
    except asyncio.TimeoutError:
        print("超时了！已处理的结果可能不完整")

    # 注意：超时后剩余任务仍在运行，需要手动取消
    for task in tasks:
        if not task.done():
            task.cancel()


asyncio.run(main())
```

输出：

```
收到: 0
收到: 1
收到: 2
超时了！已处理的结果可能不完整
```

### gather() 配合 wait_for() 实现超时

`gather()` 没有内置超时，需要配合 `asyncio.wait_for()` 使用：

```python
import asyncio


async def slow_task(n: int) -> int:
    await asyncio.sleep(n)
    return n


async def main():
    # 方式 1：给整个 gather 设超时
    try:
        results = await asyncio.wait_for(
            asyncio.gather(slow_task(1), slow_task(5), slow_task(3)),
            timeout=2.0
        )
    except asyncio.TimeoutError:
        print("gather 超时了")
        # ⚠️ 注意：wait_for 超时后会取消所有子任务

    # 方式 2：给单个任务设超时
    results = await asyncio.gather(
        asyncio.wait_for(slow_task(1), timeout=2.0),
        asyncio.wait_for(slow_task(5), timeout=2.0),  # 这个会超时
        asyncio.wait_for(slow_task(3), timeout=2.0),
        return_exceptions=True,
    )
    for r in results:
        if isinstance(r, asyncio.TimeoutError):
            print(f"  超时")
        else:
            print(f"  结果: {r}")


asyncio.run(main())
```

### 超时策略对比

```
┌─────────────────────────────────────────────────────────────────────┐
│                      超时策略对比                                    │
│                                                                     │
│  方式                    │ 超时粒度      │ 超时后行为                │
│  ───────────────────────┼──────────────┼──────────────────────────│
│  wait(timeout=3)        │ 整体         │ 返回已 done + pending     │
│  as_completed(timeout=3)│ 整体         │ 抛 TimeoutError           │
│  wait_for(gather(...))  │ 整体         │ 取消所有子任务并抛异常     │
│  wait_for(单任务)        │ 单个         │ 取消该任务并抛异常         │
│                                                                     │
│  推荐：                                                              │
│  - 需要"超时后还能拿到已完成的结果" → wait(timeout)                  │
│  - 需要"超时就报错"               → as_completed(timeout)           │
│  - 给整体设超时且失败即报错        → wait_for(gather(...))           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6.11 处理 pending 任务

当 `wait()` 因为 `FIRST_COMPLETED`、`FIRST_EXCEPTION` 或 `timeout` 提前返回时，`pending` 集合中会包含未完成的任务。如何处理这些任务是一个重要话题。

### 策略 1：取消（Cancel）

最常见的做法：

```python
import asyncio


async def long_task(name: str) -> str:
    print(f"  [{name}] 开始")
    try:
        await asyncio.sleep(10)
        return f"{name}: 完成"
    except asyncio.CancelledError:
        print(f"  [{name}] 被取消了")
        raise


async def main():
    tasks = [
        asyncio.create_task(long_task("A")),
        asyncio.create_task(long_task("B")),
        asyncio.create_task(long_task("C")),
    ]

    done, pending = await asyncio.wait(tasks, timeout=1.0)

    print(f"完成: {len(done)}, 未完成: {len(pending)}")

    # 取消所有 pending 任务
    for task in pending:
        task.cancel()

    # 等待取消操作完成（让每个任务有机会清理）
    if pending:
        await asyncio.wait(pending)
        print("所有 pending 任务已取消")


asyncio.run(main())
```

输出：

```
  [A] 开始
  [B] 开始
  [C] 开始
完成: 0, 未完成: 3
  [A] 被取消了
  [B] 被取消了
  [C] 被取消了
所有 pending 任务已取消
```

### 策略 2：稍后等待（Await Later）

有时候你不想取消，而是想先处理已完成的，再回来等剩余的：

```python
import asyncio


async def task_with_priority(name: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return f"{name}: 完成"


async def main():
    tasks = [
        asyncio.create_task(task_with_priority("紧急A", 0.5)),
        asyncio.create_task(task_with_priority("紧急B", 0.8)),
        asyncio.create_task(task_with_priority("普通C", 2.0)),
        asyncio.create_task(task_with_priority("普通D", 3.0)),
    ]

    # 第一轮：等 1 秒，处理先完成的
    done, pending = await asyncio.wait(tasks, timeout=1.0)

    print("=== 第一轮结果 ===")
    for t in done:
        print(f"  {t.result()}")

    # 第二轮：继续等待剩余任务
    if pending:
        print(f"\n等待剩余 {len(pending)} 个任务...")
        done2, pending2 = await asyncio.wait(pending)  # 等全部完成

        print("=== 第二轮结果 ===")
        for t in done2:
            print(f"  {t.result()}")


asyncio.run(main())
```

输出：

```
=== 第一轮结果 ===
  紧急A: 完成
  紧急B: 完成

等待剩余 2 个任务...
=== 第二轮结果 ===
  普通C: 完成
  普通D: 完成
```

### 策略 3：分批处理

```python
import asyncio


async def process_item(item_id: int) -> dict:
    delay = (item_id % 3 + 1) * 0.2
    await asyncio.sleep(delay)
    return {"id": item_id, "processed": True}


async def batch_process(items: list[int], batch_timeout: float = 1.0):
    """分批处理：每批等固定时间，处理已完成的，剩余进入下一批"""
    tasks = [asyncio.create_task(process_item(i)) for i in items]
    all_results = []
    batch = 0

    while tasks:
        batch += 1
        done, pending = await asyncio.wait(tasks, timeout=batch_timeout)

        print(f"批次 {batch}: 完成 {len(done)}, 剩余 {len(pending)}")
        for t in done:
            all_results.append(t.result())

        tasks = list(pending)  # 剩余任务进入下一批

    return all_results


async def main():
    items = list(range(20))
    results = await batch_process(items, batch_timeout=0.5)
    print(f"\n处理完成，共 {len(results)} 个结果")


asyncio.run(main())
```

### 策略 4：设置优先级超时

```python
import asyncio


async def fetch(url: str) -> str:
    await asyncio.sleep(0.5)
    return f"数据: {url}"


async def main():
    urls = [f"http://api{i}.example.com/data" for i in range(10)]
    tasks = [asyncio.create_task(fetch(url)) for url in urls]

    # 第一轮：快速获取（100ms 超时）
    fast, slow = await asyncio.wait(tasks, timeout=0.1)
    print(f"快速完成: {len(fast)} 个")

    # 处理快速完成的结果
    for t in fast:
        print(f"  快速: {t.result()}")

    # 第二轮：给剩余的更多时间（1s 超时）
    if slow:
        medium, still_slow = await asyncio.wait(slow, timeout=1.0)
        print(f"中速完成: {len(medium)} 个")
        for t in medium:
            print(f"  中速: {t.result()}")

        # 最后：取消仍然太慢的
        for t in still_slow:
            t.cancel()
        if still_slow:
            await asyncio.wait(still_slow)
            print(f"取消: {len(still_slow)} 个")


asyncio.run(main())
```

---

## 6.12 实际应用场景

### 应用 1：Web 爬虫 — 并发抓取与进度显示

```python
import asyncio
import time


class WebCrawler:
    def __init__(self, max_concurrent: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.results = {}
        self.errors = {}

    async def fetch_page(self, url: str) -> dict:
        """抓取单个页面（受信号量限制并发数）"""
        async with self.semaphore:
            # 模拟网络请求
            delay = hash(url) % 5 * 0.2 + 0.1
            await asyncio.sleep(delay)

            # 模拟某些请求失败
            if "error" in url:
                raise ConnectionError(f"无法访问 {url}")

            return {"url": url, "content": f"{url} 的内容", "time": delay}

    async def crawl(self, urls: list[str]) -> dict:
        """并发抓取所有 URL，实时报告进度"""
        tasks = [asyncio.create_task(self.fetch_page(url)) for url in urls]
        total = len(tasks)
        completed = 0
        start = time.time()

        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                completed += 1
                elapsed = time.time() - start
                url = result["url"]
                self.results[url] = result
                print(f"  [{elapsed:.1f}s] ({completed}/{total}) ✓ {url}")
            except Exception as e:
                completed += 1
                elapsed = time.time() - start
                self.errors[str(e)] = str(e)
                print(f"  [{elapsed:.1f}s] ({completed}/{total}) ✗ {e}")

        return {
            "total": total,
            "success": len(self.results),
            "failed": len(self.errors),
            "time": time.time() - start,
        }


async def main():
    urls = [
        "http://example.com/page1",
        "http://example.com/page2",
        "http://example.com/error/page3",
        "http://example.com/page4",
        "http://example.com/page5",
        "http://example.com/error/page6",
        "http://example.com/page7",
        "http://example.com/page8",
    ]

    crawler = WebCrawler(max_concurrent=3)
    stats = await crawler.crawl(urls)

    print(f"\n抓取统计:")
    print(f"  总计: {stats['total']}")
    print(f"  成功: {stats['success']}")
    print(f"  失败: {stats['failed']}")
    print(f"  耗时: {stats['time']:.1f}s")


asyncio.run(main())
```

### 应用 2：API 聚合器 — 多源数据聚合

```python
import asyncio
import time


class APIAggregator:
    """聚合多个数据源的结果"""

    async def fetch_weather(self, city: str) -> dict:
        await asyncio.sleep(0.3)
        return {"source": "天气API", "city": city, "temp": 25, "weather": "晴"}

    async def fetch_news(self, city: str) -> list:
        await asyncio.sleep(0.5)
        return {"source": "新闻API", "city": city, "headlines": ["新闻1", "新闻2"]}

    async def fetch_traffic(self, city: str) -> dict:
        await asyncio.sleep(0.2)
        return {"source": "交通API", "city": city, "status": "畅通"}

    async def fetch_stock(self) -> dict:
        await asyncio.sleep(0.8)
        return {"source": "股票API", "index": 3500, "change": "+1.2%"}

    async def aggregate(self, city: str, timeout: float = 1.0) -> dict:
        """聚合多个数据源，允许部分超时"""
        tasks = {
            "weather": asyncio.create_task(self.fetch_weather(city)),
            "news": asyncio.create_task(self.fetch_news(city)),
            "traffic": asyncio.create_task(self.fetch_traffic(city)),
            "stock": asyncio.create_task(self.fetch_stock()),
        }

        task_list = list(tasks.values())
        done, pending = await asyncio.wait(task_list, timeout=timeout)

        # 取消超时的任务
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.wait(pending)

        # 构建结果（反向映射 task -> name）
        task_to_name = {v: k for k, v in tasks.items()}
        result = {"city": city, "data": {}, "partial": len(pending) > 0}

        for t in done:
            name = task_to_name[t]
            try:
                result["data"][name] = t.result()
            except Exception as e:
                result["data"][name] = {"error": str(e)}

        for t in pending:
            name = task_to_name[t]
            result["data"][name] = {"error": "超时"}

        return result


async def main():
    aggregator = APIAggregator()

    print("聚合数据中...")
    start = time.time()
    result = await aggregator.aggregate("北京", timeout=0.6)
    elapsed = time.time() - start

    print(f"\n城市: {result['city']}")
    print(f"是否部分结果: {result['partial']}")
    print(f"耗时: {elapsed:.1f}s")
    print(f"\n数据:")
    for source, data in result["data"].items():
        status = "✓" if "error" not in data else "✗"
        print(f"  {status} {source}: {data}")


asyncio.run(main())
```

### 应用 3：竞速下载 — 多镜像源

```python
import asyncio
import random


async def download_from_mirror(mirror: str, file: str) -> bytes:
    """从镜像源下载文件"""
    # 模拟不同镜像的速度
    speed_map = {"CloudFlare": 0.3, "AWS": 0.5, "阿里云": 0.4, "腾讯云": 0.6}
    base_delay = speed_map.get(mirror, 1.0)

    # 加入随机波动
    delay = base_delay * random.uniform(0.8, 1.5)
    await asyncio.sleep(delay)

    # 模拟某些镜像可能失败
    if random.random() < 0.15:
        raise ConnectionError(f"{mirror} 连接失败")

    return f"{file} 的内容（来自 {mirror}）".encode()


async def race_download(mirrors: list[str], file: str) -> bytes:
    """竞速下载：用最快成功的镜像"""
    tasks = [asyncio.create_task(download_from_mirror(m, file)) for m in mirrors]
    completed_tasks = set()

    while tasks:
        done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            completed_tasks.add(task)
            try:
                result = task.result()
                # 成功了，取消剩余任务
                for p in pending:
                    p.cancel()
                if pending:
                    await asyncio.wait(pending)
                return result
            except Exception:
                # 这个镜像失败了，继续等其他的
                pass

        tasks = list(pending)

    raise RuntimeError(f"所有镜像都失败了: {mirrors}")


async def main():
    mirrors = ["CloudFlare", "AWS", "阿里云", "腾讯云"]

    print(f"从 {len(mirrors)} 个镜像竞速下载...")
    try:
        content = await race_download(mirrors, "package-v1.0.tar.gz")
        print(f"下载成功: {content.decode()}")
    except RuntimeError as e:
        print(f"下载失败: {e}")


asyncio.run(main())
```

### 应用 4：健康检查 — 并发检测服务状态

```python
import asyncio
import time


async def check_service(name: str, url: str, timeout: float = 2.0) -> dict:
    """检查单个服务的健康状态"""
    try:
        # 模拟健康检查请求
        delay = hash(url) % 3 * 0.3 + 0.1
        await asyncio.sleep(delay)

        # 模拟某些服务不健康
        if "down" in name.lower():
            raise ConnectionError(f"{name} 无响应")

        return {
            "service": name,
            "status": "healthy",
            "latency": delay,
        }
    except asyncio.CancelledError:
        return {"service": name, "status": "timeout", "latency": timeout}
    except Exception as e:
        return {"service": name, "status": "unhealthy", "error": str(e)}


async def health_check_all(services: dict[str, str], timeout: float = 2.0) -> list:
    """并发健康检查所有服务"""
    tasks = [
        asyncio.create_task(check_service(name, url, timeout))
        for name, url in services.items()
    ]

    results = []
    for coro in asyncio.as_completed(tasks, timeout=timeout):
        try:
            result = await coro
            results.append(result)
        except asyncio.TimeoutError:
            break

    return results


async def main():
    services = {
        "用户服务": "http://user-service:8080/health",
        "订单服务": "http://order-service:8081/health",
        "支付服务": "http://payment-service:8082/health",
        "通知服务-down": "http://notification-service:8083/health",
        "搜索服务": "http://search-service:8084/health",
    }

    print("开始健康检查...")
    start = time.time()
    results = await health_check_all(services, timeout=3.0)
    elapsed = time.time() - start

    healthy = [r for r in results if r["status"] == "healthy"]
    unhealthy = [r for r in results if r["status"] != "healthy"]

    print(f"\n健康检查完成 ({elapsed:.1f}s):")
    print(f"  正常: {len(healthy)}/{len(services)}")
    for r in healthy:
        print(f"    ✓ {r['service']} ({r['latency']:.2f}s)")

    if unhealthy:
        print(f"  异常: {len(unhealthy)}/{len(services)}")
        for r in unhealthy:
            print(f"    ✗ {r['service']}: {r.get('error', r['status'])}")


asyncio.run(main())
```

---

## 6.13 常见误区

### 误区 1：忘记取消 pending 任务

```python
# ❌ 错误：不取消 pending 任务
async def bad_example():
    tasks = [asyncio.create_task(slow()) for _ in range(100)]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    result = done.pop().result()
    # pending 里 99 个任务还在运行！资源泄漏！

# ✓ 正确：取消 pending 任务
async def good_example():
    tasks = [asyncio.create_task(slow()) for _ in range(100)]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    result = done.pop().result()
    for t in pending:
        t.cancel()
    if pending:
        await asyncio.wait(pending)
    return result
```

### 误区 2：直接传协程给 wait()

```python
# ❌ 错误：Python 3.11+ 会发出 DeprecationWarning
async def bad_example():
    coros = [fetch(url) for url in urls]
    done, pending = await asyncio.wait(coros)

# ✓ 正确：先创建 Task
async def good_example():
    tasks = [asyncio.create_task(fetch(url)) for url in urls]
    done, pending = await asyncio.wait(tasks)
```

### 误区 3：混淆 as_completed() 的返回类型

```python
# ❌ 错误：以为 as_completed 直接返回结果
async def bad_example():
    tasks = [asyncio.create_task(fetch(url)) for url in urls]
    for result in asyncio.as_completed(tasks):  # result 是协程，不是结果！
        print(result)  # 打印的是协程对象

# ✓ 正确：需要 await
async def good_example():
    tasks = [asyncio.create_task(fetch(url)) for url in urls]
    for coro in asyncio.as_completed(tasks):
        result = await coro  # await 才能拿到结果
        print(result)
```

### 误区 4：以为 wait() 的结果有顺序

```python
# ❌ 错误：假设 done 集合有顺序
async def bad_example():
    tasks = [asyncio.create_task(fetch(url)) for url in urls]
    done, _ = await asyncio.wait(tasks)
    first_result = list(done)[0].result()  # 不一定是第一个完成的！

# ✓ 正确：如果需要顺序，用 as_completed 或 gather
async def good_example():
    tasks = [asyncio.create_task(fetch(url)) for url in urls]
    for coro in asyncio.as_completed(tasks):
        result = await coro  # 保证按完成顺序
        process(result)
```

### 误区 5：as_completed() 超时后不取消任务

```python
# ❌ 错误：超时后任务还在跑
async def bad_example():
    tasks = [asyncio.create_task(slow_task()) for _ in range(100)]
    try:
        for coro in asyncio.as_completed(tasks, timeout=5.0):
            result = await coro
    except asyncio.TimeoutError:
        pass  # 99 个任务可能还在跑！

# ✓ 正确：超时后取消任务
async def good_example():
    tasks = [asyncio.create_task(slow_task()) for _ in range(100)]
    try:
        for coro in asyncio.as_completed(tasks, timeout=5.0):
            result = await coro
    except asyncio.TimeoutError:
        for task in tasks:
            if not task.done():
                task.cancel()
```

### 误区 6：在 as_completed 中修改任务列表

```python
# ❌ 错误：在迭代过程中修改 tasks 列表
async def bad_example():
    tasks = [asyncio.create_task(fetch(url)) for url in urls]
    for coro in asyncio.as_completed(tasks):
        result = await coro
        tasks.append(asyncio.create_task(fetch("new_url")))  # 不安全！

# ✓ 正确：先准备好所有任务，再开始迭代
async def good_example():
    all_urls = get_all_urls()  # 先确定所有 URL
    tasks = [asyncio.create_task(fetch(url)) for url in all_urls]
    for coro in asyncio.as_completed(tasks):
        result = await coro
        process(result)
```

### 误区 7：混淆 wait() 和 wait_for()

```python
# wait(aws, timeout=N) — 等待一组任务，超时后返回 (done, pending)
done, pending = await asyncio.wait(tasks, timeout=5.0)

# wait_for(aw, timeout=N) — 等待单个可等待对象，超时后抛 TimeoutError
try:
    result = await asyncio.wait_for(single_coro(), timeout=5.0)
except asyncio.TimeoutError:
    print("超时")

# 关键区别：
# - wait() 返回两个集合，不抛超时异常
# - wait_for() 返回单个结果，超时抛异常
# - wait() 接受一组任务，wait_for() 接受单个可等待对象
```

### 常见错误速查表

```
┌─────────────────────────────────────────────────────────────────────┐
│                      常见错误速查表                                   │
│                                                                     │
│  错误                            │ 后果               │ 正确做法    │
│  ──────────────────────────────┼────────────────────┼─────────────│
│  不取消 pending 任务             │ 资源泄漏           │ cancel()    │
│  传裸协程给 wait()               │ DeprecationWarning │ create_task│
│  不 await as_completed 的协程    │ 拿到协程对象       │ await coro  │
│  假设 wait() 结果有序            │ 逻辑错误           │ 用 gather   │
│  as_completed 超时不取消         │ 任务继续跑         │ 手动 cancel │
│  迭代中修改任务列表              │ 未定义行为         │ 先准备好    │
│  混淆 wait 和 wait_for           │ API 用错           │ 查文档      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 本章小结

```
┌─────────────────────────────────────────────────────────────────────┐
│                        本章核心要点                                   │
│                                                                     │
│  1. gather() 的局限                                                  │
│     - 全有或全无，无法处理部分结果                                     │
│     - 按提交顺序返回，无法按完成顺序处理                               │
│     - 缺少内置超时和等待策略控制                                      │
│                                                                     │
│  2. asyncio.wait() 的三种策略                                        │
│     - FIRST_COMPLETED：有一个完成就返回（竞速请求）                    │
│     - FIRST_EXCEPTION：有异常就返回（错误监控）                       │
│     - ALL_COMPLETED：全部完成才返回（默认）                           │
│                                                                     │
│  3. wait() 的返回值                                                  │
│     - (done, pending) 两个 set                                       │
│     - done 中的 task 已完成，用 .result() 取值                       │
│     - pending 中的 task 仍在运行，通常需要 cancel()                   │
│                                                                     │
│  4. as_completed()                                                   │
│     - 按完成顺序迭代结果                                              │
│     - 返回的是协程，需要 await                                        │
│     - 适合"实时处理"场景                                              │
│                                                                     │
│  5. 超时控制                                                         │
│     - wait(timeout=N)：返回 (done, pending)，不抛异常                │
│     - as_completed(timeout=N)：超时抛 TimeoutError                   │
│     - wait_for()：包装单个可等待对象                                  │
│                                                                     │
│  6. 关键实践                                                         │
│     - 用完 wait(FIRST_COMPLETED) 后必须取消 pending 任务              │
│     - wait() 推荐传 Task 对象，不要传裸协程                           │
│     - as_completed() 迭代时不要修改任务列表                           │
│                                                                     │
│  选择口诀：                                                           │
│    全部都要用 gather，谁先好用 as_completed，                         │
│    需要控制用 wait，超时取消别忘记。                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 思考题

### 1. gather() 和 wait(ALL_COMPLETED) 都是等所有任务完成，它们有什么区别？

`gather()` 返回有序列表，直接是结果值；`wait()` 返回无序的 `(done, pending)` 集合，需要通过 `task.result()` 提取值。`gather()` 代码更简洁，`wait()` 更灵活（支持超时参数、可以切换 return_when 策略）。如果不需要超时和策略切换，优先用 `gather()`。

### 2. 为什么 wait(FIRST_COMPLETED) 返回后必须取消 pending 任务？

因为 pending 中的任务仍在事件循环中运行，继续占用网络连接、内存等资源。如果不取消，这些任务会一直运行到完成（或程序退出），造成资源泄漏。在高频调用场景下（如竞速请求），大量未取消的 pending 任务可能导致严重性能问题。

### 3. as_completed() 返回的是什么？为什么需要 await？

`as_completed()` 返回一个**异步可迭代对象**，每次迭代产生一个**协程**（不是结果值）。这个协程包装了对底层 Future 的等待逻辑：如果对应的任务已完成则立即返回结果，否则等待它完成。所以必须 `await` 才能拿到实际结果。

### 4. 如何用 as_completed() 只取前 N 个结果？

在 `for` 循环中计数，收集够 N 个后 `break`，然后遍历原始 tasks 列表取消未完成的任务。注意 `as_completed()` 本身不提供取消机制，需要持有原始 tasks 引用。

### 5. FIRST_EXCEPTION 在没有异常时的行为是什么？

当所有任务都正常完成（没有异常）时，`FIRST_EXCEPTION` 的行为等同于 `ALL_COMPLETED`——等待所有任务完成后返回。它只在有任务抛出异常时才会"提前"返回。

### 6. wait() 和 wait_for() 有什么区别？

`asyncio.wait(aws, timeout=N)` 接受一组可等待对象，返回 `(done, pending)` 元组，不会抛超时异常。`asyncio.wait_for(aw, timeout=N)` 接受单个可等待对象，返回单个结果，超时时抛出 `TimeoutError` 并自动取消该任务。两者用途不同：`wait()` 用于多任务的批量控制，`wait_for()` 用于给单个操作加超时。

### 7. 在实际项目中，什么时候用 gather()，什么时候用 wait()，什么时候用 as_completed()？

- **gather()**：所有任务必须全部成功，结果需要保持顺序，代码简洁优先
- **wait()**：需要竞速（FIRST_COMPLETED）、错误监控（FIRST_EXCEPTION）、超时控制、或需要区分 done/pending
- **as_completed()**：需要按完成顺序逐个处理结果，适合进度显示、实时日志、流式处理

实际工程中经常组合使用：用 `wait()` 做竞速和超时控制，用 `as_completed()` 做进度展示，用 `gather()` 做简单的批量操作。

### 8. 如果一个任务在 wait() 返回 pending 集合后、cancel() 之前就完成了，cancel() 会怎样？

`cancel()` 会对已经完成的任务调用，但不会有效果——已完成的 Task 无法被取消，`cancel()` 返回 `False`。这是一种安全的操作，不需要额外检查 `task.done()`。

---
title: "预备知识：阻塞、同步、异步、并发与并行"
description: "理解阻塞/非阻塞、同步/异步、并发/并行的核心区别，建立异步编程的思维基础"
updated: "2026-07-07"
---

# 预备知识：阻塞、同步、异步、并发与并行

> **学习目标**：
> - 理解阻塞与非阻塞的区别，并能判断代码属于哪一种
> - 理解同步与异步的本质含义，掌握"结果如何交付"这一核心问题
> - 理解并发与行的区别，能用时间线分析任务执行过程
> - 掌握六个核心概念（同步、异步、阻塞、非阻塞、并发、并行）之间的关系
> - 能够判断 I/O 密集型与 CPU 密集型任务分别适合什么并发模型
> - 能够识别并纠正常见的异步编程误区

这一节非常关键。很多人学异步时，最容易把这六个词揉成一团。

先记住一个总原则：**它们描述的不是同一个维度。**

可以先分成三组：

| 分组 | 概念 | 描述的维度 |
|------|------|-----------|
| 第一组 | **同步 / 异步** | 结果怎么交付给调用者 |
| 第二组 | **阻塞 / 非阻塞** | 等待期间当前线程是否停住 |
| 第三组 | **并发 / 并行** | 多个任务在时间上如何推进 |

把这三组搞清楚，后面的 asyncio 学起来会顺畅很多。

---

## 0.1 现实类比：去餐厅点餐

理解抽象概念最好的方式是用现实场景打比方。我们用"去餐厅点餐"这个日常经历，把六个概念全部过一遍。

<div data-component="RestaurantAnalogy"></div>

### 情况 1：站在窗口一直等（同步阻塞）

你去面馆点了一份牛肉面。点完之后，你站在取餐窗口前面，哪儿也不去，什么也不做，就盯着厨房，直到面做好、端出来、拿到手里，然后才离开。

```
你                          厨房
│                            │
├── 点餐 ──────────────────→ │
│                            ├── 开始做面
│   （站着等待）              │   （煮面、切肉、加料）
│   不能玩手机                │
│   不能做任何事              │
│                            ├── 面做好了
│←──────────────── 拿到面 ────┤
│                            │
├── 离开                     │
```

**关键特征**：
- 你在拿到结果之前，什么都做不了（**阻塞**）
- 你在同一个流程中发起请求并获得结果（**同步**）
- 你只有一个线程（你自己），等待期间它被完全占住

这对应：**同步 + 阻塞**。

```python
# 对应代码
def order_and_wait():
    result = cook_beef_noodles()   # 发起请求
    # ← 执行流停在这里，直到面做好
    eat(result)                    # 拿到结果后才能继续
```

### 情况 2：点餐后拿号码牌（异步 + 面通知）

你点完餐，服务员给你一个号码牌，然后你可以回座位玩手机、聊天、甚至出门买杯奶茶。面做好后，服务员会喊号或者震动号码牌通知你，你再去取餐。

```
你                          厨房
│                            │
├── 点餐 ──────────────────→ │
│←── 拿到号码牌 ─────────────┤
│                            ├── 开始做面
├── 回座位玩手机              │   （煮面、切肉、加料）
├── 刷了一会儿短视频          │
├── 买了一杯奶茶              │
│                            ├── 面做好了
│←──── 通知：您的面好了！ ────┤
│                            │
├── 取餐、开吃               │
```

**关键特征**：
- 发起请求后，你不需要一直等（**非阻塞**）
- 结果通过通知/回调的方式在未来某个时刻交付（**异步**）
- 等待期间你可以做其他事情

这对应：**异步 + 非阻塞**。

```python
# 对应代码（伪代码）
async def order_and_do_other_things():
    ticket = start_order()           # 发起请求，拿到号码牌
    await play_with_phone()          # 做其他事情
    await buy_tea()                  # 做其他事情
    result = await wait_for(ticket)  # 收到通知后取餐
    eat(result)
```

### 情况 3：你不断去窗口问（同步非阻塞）

你点完餐后，没有站在窗口死等，但你也不放心去做别的事情。你每隔 30 秒走到窗口问一次："我的面好了吗？" 如果没好，你走开几步，但很快又回来问。

```
你                          厨房
│                            │
├── 点餐 ──────────────────→ │
│                            ├── 开始做面
├── 走到窗口：好了吗？        │
│←── 还没                     │
├── 走开几步                  │
├── 又走到窗口：好了吗？      │
│←── 还没                     │
├── 走开几步                  │
├── 又走到窗口：好了吗？      │
│←── 好了！ ─────────────────┤
├── 取餐、开吃               │
```

**关键特征**：
- 每次查询都不会让你一直卡在窗口（**非阻塞**——单次调用立即返回）
- 但你在主动反复检查结果（**同步**——你必须自己去拿结果，不是通知送过来）
- 这种模式叫**轮询（Polling）**

这对应：**同步 + 非阻塞**。

```python
# 对应代码（伪代码）
def order_and_poll():
    ticket = start_order()
    while True:
        result = check_status(ticket)   # 非阻塞：立即返回
        if result is ready:
            break
        do_something_else_briefly()     # 短暂做点别的
    eat(result)
```

### 情况 4：一个厨师交替做三份菜（并发）

现在场景变了。餐厅只有一个厨师（一个 CPU 核心），但同时来了三桌客人，分别点了牛肉面、蛋炒饭和酸辣粉。

厨师很聪明，他知道煮面需要等水烧开，炒饭需要等锅热。于是他的做法是：

```
厨师的操作时间线：

0:00  开始烧水（牛肉面）
0:30  水还没开，去切蛋（蛋炒饭）
1:00  蛋切好了，去切酸辣粉的配料
1:30  水开了！回去煮面
2:00  面在煮，去热锅（蛋炒饭）
2:30  锅热了，开始炒饭
3:00  炒饭在翻炒，去看面好了没
3:30  面好了！开始做酸辣粉
...
```

**关键特征**：
- 一个厨师（一个线程）在多个任务之间**交替切换**
- 在某一个瞬间，厨师只在做一件事
- 但在一段时间内，三份菜都在**推进**
- 通过利用等待时间（等水开、等锅热），总时间大幅缩短

这对应：**并发（Concurrency）**。

```
时间轴（一个厨师）：

任务 A（牛肉面）： ██░░░░██░░░░░░██ 完成
任务 B（蛋炒饭）： ░░██░░░░██░░░░░░██ 完成
任务 C（酸辣粉）： ░░░░██░░░░██░░░░░░██ 完成
                   ──────────────────────→ 时间
                   ██ = 正在执行  ░░ = 等待
```

### 情况 5：三个厨师同时做三份菜（并行）

餐厅升级了，现在有三个厨师、三个灶台。三桌客人同时点餐，三个厨师各自领一份，同时开始做。

```
厨师 1（核心 1）：全程做牛肉面
厨师 2（核心 2）：全程做蛋炒饭
厨师 3（核心 3）：全程做酸辣粉
```

**关键特征**：
- 三个厨师（三个 CPU 核心）在同一时刻**真正在执行**不同的任务
- 没有切换，没有等待重叠，就是纯粹的同时进行

这对应：**并行（Parallelism）**。

```
时间轴（三个厨师）：

厨师 1（核心 1）： ██████████████████ 完成（任务 A）
厨师 2（核心 2）： ██████████████████ 完成（任务 B）
厨师 3（核心 3）： ██████████████████ 完成（任务 C）
                   ──────────────────────→ 时间
```

### 五种情况对比表

| 情况 | 同步/异步 | 阻塞/非阻塞 | 并发/并行 | 核心特征 |
|------|----------|------------|----------|---------|
| 站窗口等 | 同步 | 阻塞 | 无 | 等待期间什么都不能做 |
| 拿号码牌 | 异步 | 非阻塞 | 无 | 通知送达后去取结果 |
| 反复去问 | 同步 | 非阻塞 | 无 | 主动轮询，每次查询立即返回 |
| 一个厨师做三份 | 异步 | 非阻塞 | 并发 | 交替执行，利用等待间隙 |
| 三个厨师做三份 | 异步 | 非阻塞 | 并行 | 真正同时执行 |

</div>
---

## 0.2 同步与异步

同步和异步主要描述：**调用者如何获得任务结果。**

<div data-component="SyncVsAsyncTimeline"></div>

### 1. 同步（Synchronous）

同步调用表示：调用任务后，必须在当前调用流程中等待结果返回，拿到结果后才能继续后面的逻辑。

```python
def get_data():
    # 假设这里要从数据库查询，耗时 1 秒
    time.sleep(1)
    return "用户数据"

result = get_data()       # 调用 → 等待 → 拿到结果
print(result)             # 拿到结果后才能执行
print("继续执行")
```

执行顺序：

```
调用 get_data()
  ↓
等待 1 秒……
  ↓
获得结果 "用户数据"
  ↓
赋值给 result
  ↓
打印 "用户数据"
  ↓
打印 "继续执行"
```

同步的重点不是"一定很慢"，而是：**调用和结果处于同一条连续执行流程中。**

下面这个同步函数虽然瞬间返回，但它仍然是同步的：

```python
def add(a, b):
    return a + b

result = add(1, 2)  # 同步调用，瞬间完成
print(result)        # 3
```

**为什么它也是同步？** 因为调用 `add(1, 2)` 后，结果立即在当前流程中返回。调用者不需要"以后再拿"。同步关注的是**结果交付方式**，不是速度快慢。

### 2. 异步（Asynchronous）

异步调用表示：发起任务后，结果不会立即获得，调用者可以先继续处理其他事情，之后再获取结果。

```python
import asyncio

async def fetch_user_data():
    """模拟网络请求，耗时 2 秒"""
    await asyncio.sleep(2)
    return "用户数据"

async def main():
    # 发起任务，但不立即等待结果
    task = asyncio.create_task(fetch_user_data())

    # 可以先做其他事情
    print("我先做别的事")
    await asyncio.sleep(1)
    print("别的事做完了")

    # 现在再获取结果
    result = await task
    print(f"拿到结果：{result}")

asyncio.run(main())
```

输出：

```
我先做别的事
别的事做完了
拿到结果：用户数据
```

时间线：

```
时间 0s：发起 fetch_user_data 任务
时间 0s：打印 "我先做别的事"
时间 0s～1s：执行其他事情
时间 1s：打印 "别的事做完了"
时间 2s：fetch_user_data 完成，拿到结果
```

**异步的核心：调用和结果不在同一条连续执行流程中。** 中间可以插入其他工作。

### 3. 同步 vs 异步——更精确的定义

很多人用"快/慢"来区分同步和异步，这是不准确的。更精确的区分标准是：

| 问题 | 同步 | 异步 |
|------|------|------|
| 发起调用后，结果在哪里？ | 在当前执行流中立即获得 | 在未来某个时刻通过某种机制获得 |
| 调用者需要做什么？ | 等待结果返回后继续 | 可以先做别的事，结果会"送达" |
| 调用和结果的关系 | 同一条执行流（same flow） | 不同的执行流或时间点（different flow） |

```python
# 同步：调用 → 等待 → 结果（同一执行流）
result = download("https://example.com")

# 异步：发起 → 做别的事 → 结果在未来送达（不同执行流）
task = start_download("https://example.com")
do_other_work()
result = await task
```

### 4. 同步不等于慢

一个常见的误解是"同步一定慢"。实际上：

```python
# 这是同步的，而且非常快
def add(a, b):
    return a + b

result = add(1, 2)  # 纳秒级完成
```

```python
# 这是异步的，而且很慢
async def very_slow_task():
    await asyncio.sleep(3600)  # 等待 1 小时
    return "done"
```

同步/异步描述的是**结果交付机制**，不是速度。异步的优势在于**多个任务可以重叠等待时间**，而不是单个任务变快。

### 5. 生活中的同步与异步

| 场景 | 同步 | 异步 |
|------|------|------|
| 打电话 | 打过去对方不挂，你一直举着电话等 | 发短信，对方稍后回复 |
| 取快递 | 在快递站排队等叫号 | 快递放驿站，发短信通知你去取 |
| 看医生 | 挂号后在诊室门口等 | 挂号后拿到号码，到号了广播通知 |
| 下载文件 | 程序卡住直到下载完成 | 后台下载，完成后弹通知 |

</div>
---

## 0.3 阻塞与非阻塞

阻塞和非阻塞主要描述：**当前执行线程在等待结果时，是否还能继续执行其他代码。**

<div data-component="BlockingVsNonBlocking"></div>

### 1. 阻塞（Blocking）

阻塞调用会让当前线程停下来，直到操作完成。在等待期间，这个线程什么都做不了。

```python
import time

print("开始执行")
time.sleep(3)        # ← 阻塞！当前线程停住 3 秒
print("3 秒后继续")   # ← 必须等 sleep 结束才能执行
```

时间线：

```
线程 1：

|█ 开始执行 █|░░░░░░░ 等待 3 秒 ░░░░░░░|█ 3秒后继续 █|
              ↑                           ↑
         time.sleep(3)                sleep 结束
         线程被挂起                   线程恢复
```

**为什么叫"阻塞"？** 因为当前线程像被一堵墙挡住了，完全无法前进。操作系统会把这个线程标记为"等待"状态，不分配 CPU 时间片给它。

### 2. 非阻塞（Non-blocking）

非阻塞调用不会一直等到任务完成，它会**立即返回**某种状态（可能是"还没好"），调用者可以决定接下来做什么。

```python
import socket

# 创建非阻塞 socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setblocking(False)  # 设置为非阻塞模式

try:
    data = sock.recv(1024)  # 非阻塞：立即返回
except BlockingIOError:
    print("没有数据可读，去做别的事情")
    # 线程可以继续执行其他代码
```

非阻塞的关键：**调用立即返回，不让你等。** 但返回的可能是"没有准备好"的结果。

### 3. 阻塞与非阻塞的对比

| 特征 | 阻塞 | 非阻塞 |
|------|------|--------|
| 调用后线程状态 | 挂起，等待完成 | 立即返回，线程继续 |
| 返回时机 | 任务完成后才返回 | 立即返回（可能没有结果） |
| 线程利用率 | 低（等待期间空闲） | 高（可以做其他事） |
| 编程复杂度 | 简单（顺序逻辑） | 较复杂（需要检查状态） |

```python
# 阻塞版本
def blocking_read():
    data = socket.recv(1024)      # 卡住直到有数据
    process(data)

# 非阻塞版本
def non_blocking_read():
    data = socket.recv(1024)      # 立即返回
    if data:
        process(data)
    else:
        do_something_else()       # 没数据，先做别的
```

### 4. 一个容易误解的地方：非阻塞 ≠ 异步

很多人把"非阻塞"和"异步"画等号，这是不对的。看这个例子：

```python
# 非阻塞 + 同步 = 轮询（Polling）
import time

def poll_for_result():
    while True:
        result = try_get_result()   # 非阻塞：立即返回
        if result is not None:
            return result           # 拿到结果了
        # 没拿到？继续轮询
        time.sleep(0.1)             # 等一小会儿再问

result = poll_for_result()  # 同步：在当前流程中拿到结果
```

这段代码中：
- `try_get_result()` 是**非阻塞**的——每次调用立即返回
- 但整个过程是**同步**的——你在当前执行流中主动反复检查直到拿到结果
- 这叫**轮询（Polling）**，效率不高

```python
# 异步 + 非阻塞 = 事件通知
async def async_fetch():
    result = await get_result()  # 非阻塞 + 异步
    # await 会让出控制权，事件循环去执行其他协程
    # 当结果准备好时，事件循环会恢复这个协程
    return result
```

**区分要点**：

| | 调用方式 | 等待方式 | 结果获取 |
|--|---------|---------|---------|
| 同步阻塞 | 调用后一直等 | 线程挂起 | 等待结束自动获得 |
| 同步非阻塞 | 调用后反复查 | 线程不挂，但反复查 | 主动轮询 |
| 异步非阻塞 | 调用后去做别的 | 线程/协程不挂 | 事件通知/回调 |

</div>
---

## 0.4 同步/异步 × 阻塞/非阻塞 组合

同步/异步和阻塞/非阻塞是两个独立的维度，它们可以组合出四种情况。

### 组合 1：同步阻塞（最常见、最简单）

```python
import time

def download_file(url):
    print(f"开始下载 {url}")
    time.sleep(3)                # 阻塞：线程停住
    print(f"下载完成 {url}")
    return "file_content"

# 同步：结果在当前流程中返回
result = download_file("https://example.com/file.txt")
print(result)
```

**特点**：最简单直观，但效率最低——等待期间什么都做不了。

时间线：

```
|█ 下载请求 █|░░░░░ 等待 3 秒（阻塞）░░░░░|█ 拿到结果 █|█ 处理 █|
```

### 组合 2：同步非阻塞（轮询）

```python
import time

class DownloadTask:
    def __init__(self, url):
        self.url = url
        self._progress = 0
        self._start_time = time.time()

    def check(self):
        """非阻塞检查：立即返回当前状态"""
        elapsed = time.time() - self._start_time
        self._progress = min(elapsed / 3.0, 1.0)  # 3 秒完成
        if self._progress >= 1.0:
            return "完成", self._progress
        return "进行中", self._progress

# 同步非阻塞使用方式
task = DownloadTask("https://example.com/file.txt")

while True:
    status, progress = task.check()  # 非阻塞：立即返回
    print(f"状态：{status}，进度：{progress:.0%}")
    if status == "完成":
        break
    time.sleep(0.5)  # 不一直问，每隔 0.5 秒问一次
```

**特点**：调用不会卡住线程，但需要主动轮询，编程复杂度高。

### 组合 3：异步非阻塞（asyncio 的标准模式）

```python
import asyncio

async def download_file(url):
    print(f"开始下载 {url}")
    await asyncio.sleep(3)        # 非阻塞：让出控制权
    print(f"下载完成 {url}")
    return "file_content"

async def main():
    # 异步：结果在将来通过 await 获得
    task = asyncio.create_task(download_file("https://example.com/file.txt"))

    # 非阻塞：上面的代码不会卡住，可以继续执行
    print("我可以做别的事情！")
    await asyncio.sleep(1)
    print("别的事情做完了")

    # 在需要的时候获取结果
    result = await task
    print(f"结果：{result}")

asyncio.run(main())
```

**特点**：最高效的 I/O 等待方式——等待期间可以做其他事情，不需要轮询。

### 组合 4：异步中夹杂阻塞（常见错误！）

```python
import asyncio
import time

async def bad_download(url):
    """这是一个陷阱！"""
    print(f"开始下载 {url}")
    time.sleep(3)                 # 错误！这是同步阻塞！
    print(f"下载完成 {url}")
    return "file_content"

async def main():
    # 看起来是异步的，实际上是阻塞的
    task1 = asyncio.create_task(bad_download("url1"))
    task2 = asyncio.create_task(bad_download("url2"))

    await asyncio.gather(task1, task2)
    # 实际总耗时 6 秒，不是 3 秒！

asyncio.run(main())
```

**为什么是 6 秒？** 因为 `time.sleep(3)` 是同步阻塞函数，它会阻塞整个线程（包括事件循环）。即使在 `async def` 中，它也不会让出控制权。

时间线（错误版本）：

```
任务 1：|█ 开始 █|░░░ 阻塞 3 秒（time.sleep）░░░|█ 完成 █|
任务 2：                                           |█ 开始 █|░░░ 阻塞 3 秒 ░░░|█ 完成 █|
                        总耗时 = 6 秒
```

时间线（正确版本）：

```
任务 1：|█ 开始 █|░░░ 等待 3 秒（asyncio.sleep）░░░|█ 完成 █|
任务 2：|█ 开始 █|░░░ 等待 3 秒（asyncio.sleep）░░░|█ 完成 █|
                        总耗时 = 3 秒
```

### 四种组合总结表

| 组合 | 同步/异步 | 阻塞/非阻塞 | Python 表现 | 常见程度 |
|------|----------|------------|-------------|---------|
| 同步阻塞 | 同步 | 阻塞 | 普通函数 + `time.sleep()` | 最常见 |
| 同步非阻塞 | 同步 | 非阻塞 | 轮询循环 + 状态检查 | 较少见 |
| 异步非阻塞 | 异步 | 非阻塞 | `async/await` + `asyncio.sleep()` | asyncio 标准 |
| 异步夹杂阻塞 | 异步 | 阻塞（错误） | `async def` 中误用 `time.sleep()` | 常见错误 |

---

## 0.5 并发与并行

<div data-component="ConcurrencyVsParallelism"></div>

### 1. 并发（Concurrency）

并发是：多个任务在同一段时间内**交替推进**。在任意一个瞬间，可能只有一个任务在执行，但在宏观时间尺度上，多个任务都在向前推进。

```
单核 CPU 上的并发：

时间 →  0   1   2   3   4   5   6   7   8   9  10
任务 A：██  ░░  ░░  ██  ░░  ░░  ██  ██  ░░  ░░  ██ 完成
任务 B：░░  ██  ░░  ░░  ██  ░░  ░░  ░░  ██  ░░  ██ 完成
任务 C：░░  ░░  ██  ░░  ░░  ██  ░░  ░░  ░░  ██  ██ 完成

██ = 执行  ░░ = 等待/让出
```

**为什么能交替？** 因为很多任务不是一直在计算，它们经常需要等待（等网络、等磁盘、等用户输入）。在等待的间隙，CPU 可以去执行别的任务。

#### 并发的 Python 实现方式

```python
# 方式 1：asyncio（单线程并发）
import asyncio

async def task_a():
    for i in range(3):
        print(f"A-{i}")
        await asyncio.sleep(0.1)

async def task_b():
    for i in range(3):
        print(f"B-{i}")
        await asyncio.sleep(0.1)

async def main():
    await asyncio.gather(task_a(), task_b())

asyncio.run(main())
```

```python
# 方式 2：threading（多线程并发）
import threading
import time

def task_a():
    for i in range(3):
        print(f"A-{i}")
        time.sleep(0.1)

def task_b():
    for i in range(3):
        print(f"B-{i}")
        time.sleep(0.1)

t1 = threading.Thread(target=task_a)
t2 = threading.Thread(target=task_b)
t1.start()
t2.start()
t1.join()
t2.join()
```

### 2. 并行（Parallelism）

并行是：多个任务在**同一时刻真正执行**。需要多个 CPU 核心。

```
多核 CPU 上的并行：

核心 1：████████████████████████  任务 A 完成
核心 2：████████████████████████  任务 B 完成
核心 3：████████████████████████  任务 C 完成
        ─────────────────────────→ 时间
```

#### 并行的 Python 实现方式

```python
# 方式：multiprocessing（多进程并行）
from multiprocessing import Process
import time

def task_a():
    for i in range(3):
        print(f"A-{i}")
        time.sleep(0.1)

def task_b():
    for i in range(3):
        print(f"B-{i}")
        time.sleep(0.1)

if __name__ == "__main__":
    p1 = Process(target=task_a)
    p2 = Process(target=task_b)
    p1.start()
    p2.start()
    p1.join()
    p2.join()
```

### 3. 并发 ≠ 并行——核心区别

| 特征 | 并发（Concurrency） | 并行（Parallelism） |
|------|-------------------|-------------------|
| 同一时刻 | 通常只有一个任务在执行 | 多个任务同时执行 |
| 需要的硬件 | 单核即可 | 需要多核 |
| 本质 | 任务交替推进 | 任务同时推进 |
| Python 实现 | asyncio、threading | multiprocessing |
| 适合场景 | I/O 密集型 | CPU 密集型 |
| 关注重点 | 结构（如何组织多个任务） | 执行（如何同时计算） |

一个经典比喻：

> **并发**是一个厨师在三口锅之间来回切换。
> **并行**是三个厨师各管一口锅。

### 4. Rob Pike 的经典定义

Go 语言创始人 Rob Pike 给出过简洁的区分：

> - **Concurrency** is about dealing with lots of things at once.
> - **Parallelism** is about doing lots of things at once.

翻译：

> - **并发**是同时**处理**多件事（结构上安排）。
> - **并行**是同时**做**多件事（物理上同时）。

### 5. GIL 与 Python 的并发限制

Python 的 GIL（Global Interpreter Lock，全局解释器锁）导致：

```
CPython 中：

asyncio 并发：  ✓ 单线程，GIL 不影响
threading 并发：✓ 受 GIL 限制，但 I/O 等待时会释放 GIL
multiprocessing 并行：✓ 每个进程有自己的 GIL
```

```python
# GIL 的影响示例
import threading
import time

def cpu_work():
    """CPU 密集型工作"""
    total = 0
    for i in range(10_000_000):
        total += i
    return total

# 多线程执行 CPU 密集型任务（受 GIL 限制）
start = time.perf_counter()
t1 = threading.Thread(target=cpu_work)
t2 = threading.Thread(target=cpu_work)
t1.start(); t2.start()
t1.join(); t2.join()
print(f"多线程：{time.perf_counter() - start:.2f} 秒")  # ≈ 顺序时间

# 多进程执行 CPU 密集型任务（绕过 GIL）
from multiprocessing import Process
start = time.perf_counter()
p1 = Process(target=cpu_work)
p2 = Process(target=cpu_work)
p1.start(); p2.start()
p1.join(); p2.join()
print(f"多进程：{time.perf_counter() - start:.2f} 秒")  # ≈ 一半时间
```
</div>

---

## 0.6 时间线比较：顺序执行 vs 并发执行

<div data-component="SequentialVsConcurrentTimeline"></div>

假设有三个任务，每个任务需要网络请求，各等待 2 秒。

### 顺序执行时间线

```
任务 A：|█ 请求 █|░░░ 等待 2s ░░░|█ 处理 █|
任务 B：                          |█ 请求 █|░░░ 等待 2s ░░░|█ 处理 █|
任务 C：                                                    |█ 请求 █|░░░ 等待 2s ░░░|█ 处理 █|
        ──────────────────────────────────────────────────────────────────────────────────→ 时间
        0s                      2s                      4s                      6s

总耗时 = 2 + 2 + 2 = 6 秒
```

代码验证：

```python
import time

def fetch(name):
    print(f"[{name}] 开始请求")
    time.sleep(2)  # 模拟 2 秒网络等待
    print(f"[{name}] 请求完成")

start = time.perf_counter()
fetch("A")
fetch("B")
fetch("C")
print(f"总耗时：{time.perf_counter() - start:.1f} 秒")
```

输出：

```
[A] 开始请求
[A] 请求完成
[B] 开始请求
[B] 请求完成
[C] 开始请求
[C] 请求完成
总耗时：6.0 秒
```

### 并发执行时间线

```
任务 A：|█ 请求 █|░░░░░░░ 等待 2s ░░░░░░░|█ 处理 █|
任务 B：|█ 请求 █|░░░░░░░ 等待 2s ░░░░░░░|█ 处理 █|
任务 C：|█ 请求 █|░░░░░░░ 等待 2s ░░░░░░░|█ 处理 █|
        ──────────────────────────────────────────→ 时间
        0s                                    2s

总耗时 = max(2, 2, 2) = 2 秒
```

代码验证：

```python
import asyncio
import time

async def fetch(name):
    print(f"[{name}] 开始请求")
    await asyncio.sleep(2)  # 异步等待 2 秒
    print(f"[{name}] 请求完成")

async def main():
    start = time.perf_counter()
    await asyncio.gather(
        fetch("A"),
        fetch("B"),
        fetch("C"),
    )
    print(f"总耗时：{time.perf_counter() - start:.1f} 秒")

asyncio.run(main())
```

输出：

```
[A] 开始请求
[B] 开始请求
[C] 开始请求
[A] 请求完成
[B] 请求完成
[C] 请求完成
总耗时：2.0 秒
```

### 为什么快了 3 倍？

这里不是把每个任务从 2 秒压缩成了 0.67 秒。**每个任务的网络等待时间仍然是 2 秒。**

真正发生的是：**三个任务的等待时间重叠了。**

```
时间 0s：三个任务同时发起请求
时间 0s～2s：三个请求都在网络上"飞"，CPU 空闲
时间 2s：三个请求几乎同时到达
```

这就是异步并发的价值——**不是让单个任务变快，而是让多个任务的等待时间重叠。**

### 当任务耗时不同时

如果三个任务分别需要 1 秒、2 秒、3 秒：

```
顺序执行：1 + 2 + 3 = 6 秒
并发执行：max(1, 2, 3) = 3 秒
```

```
任务 A：|█|░ 1s ░|█|
任务 B：|█|░░ 2s ░░|█|
任务 C：|█|░░░ 3s ░░░|█|
        ─────────────────→ 时间
        0s           3s
```

**规律**：并发总耗时 = max(各任务耗时)。顺序总耗时 = sum(各任务耗时)。

</div>
---

## 0.7 同步版本代码详解

让我们写一个完整的同步版本程序，模拟三个 API 请求，每个需要不同的等待时间。

```python
import time

def fetch_user_profile(user_id):
    """模拟获取用户资料的 API 请求"""
    print(f"  [用户{user_id}] 开始请求用户资料...")
    time.sleep(1.5)  # 模拟网络延迟 1.5 秒
    print(f"  [用户{user_id}] 用户资料获取完成")
    return {"id": user_id, "name": f"用户{user_id}"}

def fetch_user_orders(user_id):
    """模拟获取用户订单的 API 请求"""
    print(f"  [用户{user_id}] 开始请求订单数据...")
    time.sleep(2.0)  # 模拟网络延迟 2 秒
    print(f"  [用户{user_id}] 订单数据获取完成")
    return [{"order_id": 1, "item": "商品A"}, {"order_id": 2, "item": "商品B"}]

def fetch_recommendations(user_id):
    """模拟获取推荐内容的 API 请求"""
    print(f"  [用户{user_id}] 开始请求推荐内容...")
    time.sleep(1.0)  # 模拟网络延迟 1 秒
    print(f"  [用户{user_id}] 推荐内容获取完成")
    return ["推荐商品1", "推荐商品2", "推荐商品3"]

def build_user_dashboard(user_id):
    """构建用户主页数据（同步版本）"""
    print(f"\n[构建] 开始构建用户{user_id}的主页...")

    # 三个请求顺序执行
    profile = fetch_user_profile(user_id)       # 等待 1.5 秒
    orders = fetch_user_orders(user_id)          # 等待 2.0 秒
    recs = fetch_recommendations(user_id)        # 等待 1.0 秒

    print(f"[构建] 用户{user_id}主页构建完成\n")
    return {
        "profile": profile,
        "orders": orders,
        "recommendations": recs,
    }

# 执行
start = time.perf_counter()
dashboard = build_user_dashboard(1)
elapsed = time.perf_counter() - start

print(f"总耗时：{elapsed:.2f} 秒")
print(f"预期耗时：1.5 + 2.0 + 1.0 = 4.5 秒")
```

输出：

```
[构建] 开始构建用户1的主页...
  [用户1] 开始请求用户资料...
  [用户1] 用户资料获取完成
  [用户1] 开始请求订单数据...
  [用户1] 订单数据获取完成
  [用户1] 开始请求推荐内容...
  [用户1] 推荐内容获取完成
[构建] 用户1主页构建完成

总耗时：4.50 秒
预期耗时：1.5 + 2.0 + 1.0 = 4.5 秒
```

时间线分析：

```
fetch_user_profile：|█ 请求 █|░░░ 1.5s ░░░|█ 完成 █|
fetch_user_orders：                          |█ 请求 █|░░░ 2.0s ░░░|█ 完成 █|
fetch_recommendations：                                                   |█ 请求 █|░ 1.0s |█ 完成 █|
                    ────────────────────────────────────────────────────────────────────────→ 时间
                    0s         1.5s           3.5s                       4.5s
```

**问题所在**：三个请求之间没有数据依赖（推荐内容不需要等订单数据完成），但因为是同步执行，它们只能排队。

---

## 0.8 异步并发版本详解

把上面的同步版本改造成异步并发版本。

### 完整代码

```python
import asyncio
import time

async def fetch_user_profile(user_id):
    """模拟获取用户资料的 API 请求"""
    print(f"  [用户{user_id}] 开始请求用户资料...")
    await asyncio.sleep(1.5)  # 异步等待 1.5 秒
    print(f"  [用户{user_id}] 用户资料获取完成")
    return {"id": user_id, "name": f"用户{user_id}"}

async def fetch_user_orders(user_id):
    """模拟获取用户订单的 API 请求"""
    print(f"  [用户{user_id}] 开始请求订单数据...")
    await asyncio.sleep(2.0)  # 异步等待 2 秒
    print(f"  [用户{user_id}] 订单数据获取完成")
    return [{"order_id": 1, "item": "商品A"}, {"order_id": 2, "item": "商品B"}]

async def fetch_recommendations(user_id):
    """模拟获取推荐内容的 API 请求"""
    print(f"  [用户{user_id}] 开始请求推荐内容...")
    await asyncio.sleep(1.0)  # 异步等待 1 秒
    print(f"  [用户{user_id}] 推荐内容获取完成")
    return ["推荐商品1", "推荐商品2", "推荐商品3"]

async def build_user_dashboard(user_id):
    """构建用户主页数据（异步并发版本）"""
    print(f"\n[构建] 开始构建用户{user_id}的主页...")

    # 三个请求并发执行！
    profile, orders, recs = await asyncio.gather(
        fetch_user_profile(user_id),
        fetch_user_orders(user_id),
        fetch_recommendations(user_id),
    )

    print(f"[构建] 用户{user_id}主页构建完成\n")
    return {
        "profile": profile,
        "orders": orders,
        "recommendations": recs,
    }

async def main():
    start = time.perf_counter()
    dashboard = await build_user_dashboard(1)
    elapsed = time.perf_counter() - start

    print(f"总耗时：{elapsed:.2f} 秒")
    print(f"预期耗时：max(1.5, 2.0, 1.0) = 2.0 秒")

asyncio.run(main())
```

输出：

```
[构建] 开始构建用户1的主页...
  [用户1] 开始请求用户资料...
  [用户1] 开始请求订单数据...
  [用户1] 开始请求推荐内容...
  [用户1] 推荐内容获取完成
  [用户1] 用户资料获取完成
  [用户1] 订单数据获取完成
[构建] 用户1主页构建完成

总耗时：2.00 秒
预期耗时：max(1.5, 2.0, 1.0) = 2.0 秒
```

时间线分析：

```
fetch_user_profile：   |█|░░░░ 1.5s ░░░░|█|
fetch_user_orders：    |█|░░░░░ 2.0s ░░░░░|█|
fetch_recommendations：|█|░ 1.0s ░|█|
                       ─────────────────────→ 时间
                       0s                 2.0s

三个请求几乎同时发出，等待时间完全重叠！
总耗时 = max(1.5, 2.0, 1.0) = 2.0 秒
```

### asyncio.gather() 的执行过程

让我们详细看看 `asyncio.gather()` 内部发生了什么：

```python
async def main():
    # Step 1: 创建三个协程对象（还没有执行）
    coro1 = fetch_user_profile(1)
    coro2 = fetch_user_orders(1)
    coro3 = fetch_recommendations(1)

    # Step 2: gather() 把三个协程包装成任务，提交给事件循环
    results = await asyncio.gather(coro1, coro2, coro3)

    # Step 3: 事件循环调度执行
    #   时间 0.0s：三个任务都开始执行，各自打印"开始请求"
    #   时间 0.0s：三个任务都遇到 await asyncio.sleep()
    #   时间 0.0s：事件循环发现三个任务都在等待，切换到下一个
    #   时间 1.0s：推荐内容完成，事件循环恢复对应协程
    #   时间 1.5s：用户资料完成，事件循环恢复对应协程
    #   时间 2.0s：订单数据完成，事件循环恢复对应协程
    #   时间 2.0s：三个任务全部完成，gather() 返回结果列表

    # Step 4: results 是一个列表，顺序与传入的协程顺序一致
    profile, orders, recs = results
    return {"profile": profile, "orders": orders, "recommendations": recs}
```

### create_task() vs gather()

两种方式都可以并发执行任务，但有细微区别：

```python
# 方式 1：asyncio.gather() —— 一次性提交所有任务
results = await asyncio.gather(
    fetch_user_profile(1),
    fetch_user_orders(1),
    fetch_recommendations(1),
)
```

```python
# 方式 2：asyncio.create_task() —— 逐个创建任务
task1 = asyncio.create_task(fetch_user_profile(1))
task2 = asyncio.create_task(fetch_user_orders(1))
task3 = asyncio.create_task(fetch_recommendations(1))

# 任务已经在后台运行了，可以做其他事情
print("任务已提交")

# 在需要结果的时候 await
results = await asyncio.gather(task1, task2, task3)
```

**区别**：
- `gather()` 接受协程，内部自动创建任务
- `create_task()` 立即创建任务并开始调度，可以更精细地控制
- `create_task()` 后可以插入其他代码再 `await`

---

## 0.9 time.sleep() 和 asyncio.sleep() 的区别

<div data-component="TimeSleepVsAsyncioSleep"></div>

这是异步编程初学者最常犯的错误之一。理解这两个函数的区别，是掌握 asyncio 的关键。

### 核心区别

| 函数 | 类型 | 行为 | 对事件循环的影响 |
|------|------|------|------------------|
| `time.sleep(2)` | 同步阻塞 | 当前线程暂停 2 秒 | 卡住整个事件循环，所有协程停止 |
| `await asyncio.sleep(2)` | 异步非阻塞 | 当前协程暂停 2 秒 | 事件循环继续运行，其他协程可以执行 |

### 详细对比实验

#### 实验 1：使用 asyncio.sleep()（正确）

```python
import asyncio
import time

async def task(name, delay):
    print(f"[{name}] 开始，等待 {delay} 秒")
    await asyncio.sleep(delay)    # 异步等待
    print(f"[{name}] 完成")

async def main():
    start = time.perf_counter()

    await asyncio.gather(
        task("A", 2),
        task("B", 2),
        task("C", 2),
    )

    elapsed = time.perf_counter() - start
    print(f"总耗时：{elapsed:.2f} 秒")

asyncio.run(main())
```

输出：

```
[A] 开始，等待 2 秒
[B] 开始，等待 2 秒
[C] 开始，等待 2 秒
[A] 完成
[B] 完成
[C] 完成
总耗时：2.00 秒
```

#### 实验 2：使用 time.sleep()（错误）

```python
import asyncio
import time

async def task(name, delay):
    print(f"[{name}] 开始，等待 {delay} 秒")
    time.sleep(delay)             # 错误！同步阻塞！
    print(f"[{name}] 完成")

async def main():
    start = time.perf_counter()

    await asyncio.gather(
        task("A", 2),
        task("B", 2),
        task("C", 2),
    )

    elapsed = time.perf_counter() - start
    print(f"总耗时：{elapsed:.2f} 秒")

asyncio.run(main())
```

输出：

```
[A] 开始，等待 2 秒
[A] 完成
[B] 开始，等待 2 秒
[B] 完成
[C] 开始，等待 2 秒
[C] 完成
总耗时：6.00 秒      ← 变成了顺序执行！
```

### 为什么 time.sleep() 会卡住事件循环？

```
事件循环的工作方式：

正常状态：
┌─────────────────────────────────────────┐
│ 事件循环                                 │
│  ┌─────┐  ┌─────┐  ┌─────┐            │
│  │协程A │→│协程B │→│协程C │→ 检查事件 →│
│  └─────┘  └─────┘  └─────┘            │
│  遇到 await 就切换到下一个协程           │
└─────────────────────────────────────────┘

使用 time.sleep() 时：
┌─────────────────────────────────────────┐
│ 事件循环                                 │
│  ┌─────┐                                │
│  │协程A │→ time.sleep(2) → 线程挂起！   │
│  └─────┘   整个事件循环被阻塞            │
│  协程B 无法执行                          │
│  协程C 无法执行                          │
│  （所有人都在等协程 A 的 sleep 结束）     │
└─────────────────────────────────────────┘
```

**关键理解**：asyncio 的事件循环运行在**单个线程**中。`time.sleep()` 阻塞的是这个线程，线程被阻塞后，事件循环无法调度任何协程。

### 混合使用的陷阱

有时候代码看起来没问题，但某个第三方库内部可能使用了同步阻塞操作：

```python
import asyncio
import requests  # 同步 HTTP 库

async def fetch_data(url):
    print(f"请求 {url}")
    response = requests.get(url)    # 隐式阻塞！
    return response.text

async def main():
    # 看起来是并发，实际是顺序执行
    results = await asyncio.gather(
        fetch_data("https://api.example.com/a"),
        fetch_data("https://api.example.com/b"),
        fetch_data("https://api.example.com/c"),
    )
```

**解决方案**：使用异步 HTTP 库（如 `aiohttp`）或用 `asyncio.to_thread()` 包装同步调用。

```python
import asyncio
import aiohttp  # 异步 HTTP 库

async def fetch_data(url):
    print(f"请求 {url}")
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

# 或者用 to_thread 包装同步调用
async def fetch_data_sync_compat(url):
    import requests
    return await asyncio.to_thread(requests.get, url)
```

### 如何检测阻塞调用

```python
import asyncio
import time
import logging

# 方法 1：使用 asyncio 的调试模式
asyncio.run(main(), debug=True)

# 方法 2：手动记录慢操作
async def monitored_sleep(delay):
    start = time.monotonic()
    await asyncio.sleep(delay)
    actual = time.monotonic() - start
    if actual > delay + 0.1:  # 超过预期 100ms
        logging.warning(f"sleep 耗时 {actual:.2f}s，预期 {delay}s")

# 方法 3：使用第三方工具
# pip install aiomonitor
# 可以实时查看哪些协程阻塞了事件循环
```
</div>

---

## 0.10 I/O 等待为什么适合异步

### I/O 操作的时间构成

一个典型的网络请求：

```
总耗时 200ms 的 HTTP 请求：

DNS 查询：     ██ 10ms
TCP 连接：     ██ 15ms
发送请求：     █ 5ms
等待响应：     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 150ms  ← 主要时间！
接收数据：     ██ 10ms
处理数据：     █ 5ms
               ─────────────────────────────────────→
               计算时间：~35ms（17.5%）
               等待时间：~150ms（75%）← CPU 空闲！
```

**关键观察**：I/O 操作中，CPU 大部分时间在等待，真正计算只用了很短时间。

### 异步如何利用等待时间

```python
import asyncio
import time

async def fetch(url, delay):
    """模拟 I/O 请求"""
    # 计算阶段（很短）
    print(f"  [{url}] 构造请求...")
    await asyncio.sleep(0.01)  # 模拟 10ms 计算

    # 等待阶段（很长）—— CPU 空闲
    print(f"  [{url}] 等待响应...")
    await asyncio.sleep(delay)  # 模拟网络等待

    # 处理阶段（很短）
    print(f"  [{url}] 处理响应...")
    await asyncio.sleep(0.01)  # 模拟 10ms 处理
    return f"{url} 的数据"

async def main():
    start = time.perf_counter()

    # 三个请求并发
    results = await asyncio.gather(
        fetch("api/users", 1.0),
        fetch("api/orders", 1.5),
        fetch("api/products", 0.8),
    )

    elapsed = time.perf_counter() - start
    print(f"\n总耗时：{elapsed:.2f} 秒")
    print(f"如果顺序执行：{1.0 + 1.5 + 0.8:.2f} 秒")

asyncio.run(main())
```

时间线：

```
api/users：    |█计算█|░░░░░ 等待 1.0s ░░░░░|█处理█|
api/orders：   |█计算█|░░░░░░ 等待 1.5s ░░░░░░|█处理█|
api/products： |█计算█|░░░ 等待 0.8s ░░░|█处理█|
               ──────────────────────────────────────→
               0s                           1.5s

总耗时 ≈ 1.5 秒（而不是 3.3 秒）
```

### 适合异步的 I/O 场景

| 场景 | 等待时间 | 异步收益 |
|------|---------|---------|
| HTTP API 调用 | 50ms～2s | 高 |
| 数据库查询 | 1ms～500ms | 中～高 |
| 文件读写 | 0.1ms～100ms | 低～中 |
| Redis 缓存 | 0.1ms～10ms | 低 |
| 消息队列 | 1ms～100ms | 中 |

**规律**：等待时间越长、并发请求数越多，异步的收益越大。

---

## 0.11 CPU 密集型任务为什么不适合 asyncio

### CPU 密集型任务的特征

CPU 密集型任务的特点是：**几乎不等待，一直在计算。**

```python
# CPU 密集型：大数计算
import time

def calculate_primes(n):
    """计算 n 以内的素数个数"""
    count = 0
    for num in range(2, n):
        is_prime = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            count += 1
    return count

start = time.perf_counter()
result = calculate_primes(1_000_000)
print(f"素数个数：{result}，耗时：{time.perf_counter() - start:.2f} 秒")
```

时间线：

```
calculate_primes：
|████████████████████████████████████████████████████████|  持续计算
                                                          ↑
                                                     没有任何等待点！
                                                     CPU 始终在工作
```

### 为什么 asyncio 不适合 CPU 密集型

asyncio 的事件循环需要在任务之间切换，但切换的前提是任务遇到 `await`。

```python
async def cpu_intensive_task():
    total = 0
    for i in range(100_000_000):
        total += i           # 纯计算，没有 await
        # ← 事件循环无法在这里切换到其他任务
    return total
```

```
问题：

协程 A（CPU 密集）：|████████████████████████████████████████| 完成
协程 B（等待 I/O）：                                            |█|░等待░|█|
                    ────────────────────────────────────────────→
                    整个计算期间，协程 B 完全无法执行！

事件循环：我想要切换，但是协程 A 一直在跑，不给我机会！
```

### 对比：适合 vs 不适合

```python
# 适合 asyncio 的任务
async def io_task():
    data = await fetch_from_api()     # 等待 200ms（让出控制权）
    processed = quick_process(data)   # 计算 1ms
    await save_to_db(processed)       # 等待 50ms（让出控制权）
    # 总等待 250ms，计算 1ms → 非常适合异步

# 不适合 asyncio 的任务
async def cpu_task():
    result = heavy_computation()      # 计算 5 秒，没有 await
    # 总等待 0ms，计算 5s → 不适合异步，应该用多进程
```

### CPU 密集型任务的正确方案

```python
# 方案 1：使用多进程
from multiprocessing import Pool

def cpu_work(n):
    """CPU 密集型工作"""
    return sum(i * i for i in range(n))

if __name__ == "__main__":
    with Pool(4) as pool:  # 4 个进程并行
        results = pool.map(cpu_work, [10_000_000] * 4)
    print(f"结果：{results}")
```

```python
# 方案 2：在 asyncio 中使用进程池（混合方案）
import asyncio
from concurrent.futures import ProcessPoolExecutor

def cpu_work(n):
    """CPU 密集型工作（普通函数）"""
    return sum(i * i for i in range(n))

async def main():
    loop = asyncio.get_event_loop()

    # 把 CPU 密集型任务放到进程池
    with ProcessPoolExecutor() as pool:
        results = await asyncio.gather(
            loop.run_in_executor(pool, cpu_work, 10_000_000),
            loop.run_in_executor(pool, cpu_work, 10_000_000),
            loop.run_in_executor(pool, cpu_work, 10_000_000),
        )

    print(f"结果：{results}")

asyncio.run(main())
```

### 任务类型选择指南

```
你的任务是什么类型？
       │
       ├── I/O 密集型（网络、文件、数据库）
       │    │
       │    ├── 任务数量多（>10）────→ asyncio
       │    └── 任务数量少（<10）────→ threading 也可以
       │
       └── CPU 密集型（计算、加密、图像处理）
            │
            ├── 需要结果集成 ────────→ multiprocessing.Pool
            └── 可以独立运行 ────────→ ProcessPoolExecutor
```

---

## 0.12 六个概念的准确总结

<div data-component="ConceptRelationshipDiagram"></div>

### 定义表

| 概念 | 定义 | 关键词 |
|------|------|--------|
| **同步（Synchronous）** | 调用者在当前流程中等待任务结果，结果在同一条执行流中返回 | 同一流程、顺序、等待 |
| **异步（Asynchronous）** | 结果不会立即获得，调用者可以先处理其他工作，结果通过通知/回调/await 在未来送达 | 不同流程、通知、将来 |
| **阻塞（Blocking）** | 当前线程因为等待 I/O 或其他操作而无法继续执行 | 线程挂起、无法前进 |
| **非阻塞（Non-blocking）** | 调用立即返回，不管操作是否完成，线程可以继续执行 | 立即返回、线程不挂 |
| **并发（Concurrency）** | 多个任务在同一段时间内交替推进，强调结构上的同时处理 | 交替、轮流、单核可实现 |
| **并行（Parallelism）** | 多个任务在同一时刻真正同时执行，需要多个 CPU 核心 | 同时、多核、物理并行 |

### 关系图

```
                    ┌─────────────────────────────────────┐
                    │         六个概念的关系               │
                    └─────────────────────────────────────┘

  ┌──────────────┐                              ┌──────────────┐
  │    同步       │←──── 结果交付方式 ────→│    异步       │
  │  (Synchronous)│                              │ (Asynchronous)│
  │ 调用者等待结果│                              │ 结果将来送达  │
  └──────┬───────┘                              └──────┬───────┘
         │                                              │
         │ 可以搭配                                      │ 可以搭配
         ↓                                              ↓
  ┌──────────────┐                              ┌──────────────┐
  │    阻塞       │←──── 线程是否挂起 ────→│   非阻塞      │
  │  (Blocking)   │                              │ (Non-blocking)│
  │ 线程停下来等  │                              │ 调用立即返回  │
  └──────────────┘                              └──────────────┘

  ┌──────────────┐                              ┌──────────────┐
  │    并发       │←──── 任务执行方式 ────→│    并行       │
  │ (Concurrency) │                              │ (Parallelism) │
  │ 交替推进      │                              │ 同时执行      │
  └──────────────┘                              └──────────────┘
```

### 维度归类

```
维度 1：结果交付
  ├── 同步：在当前流程中获得结果
  └── 异步：结果将来通过通知/回调/await 获得

维度 2：线程行为
  ├── 阻塞：线程停下来等
  └── 非阻塞：线程不等，调用立即返回

维度 3：任务推进
  ├── 并发：多个任务交替推进（单核可实现）
  └── 并行：多个任务同时推进（需要多核）
```

### Python 中的典型对应

| 场景 | 典型技术 | 涉及的概念 |
|------|---------|-----------|
| `requests.get()` | 同步阻塞 | 同步 + 阻塞 |
| 轮询 `socket.recv()` | 同步非阻塞 | 同步 + 非阻塞 |
| `await asyncio.sleep()` | 异步非阻塞 | 异步 + 非阻塞 |
| `time.sleep()` 在 `async def` 中 | 异步（形式）+ 阻塞（实际） | 错误用法！ |
| `asyncio.gather()` 多个 I/O 任务 | 异步非阻塞 + 并发 | asyncio 标准模式 |
| `multiprocessing.Pool` | 同步阻塞 + 并行 | CPU 密集型方案 |

</div>
---

## 0.13 常见误区

### 误区 1：异步一定比同步快

**错误认识**："异步代码更快，所以我应该把所有代码都改成异步。"

**事实**：异步主要提升 **I/O 密集型** 场景的吞吐量，对 CPU 密集型任务没有帮助，甚至可能更慢（因为有调度开销）。

```python
# 这个任务用异步没有意义
async def compute():
    total = 0
    for i in range(100_000_000):
        total += i
    return total

# 因为没有 I/O 等待，asyncio 无法在中途切换
# 反而多了事件循环的调度开销
```

### 误区 2：并发就是并行

**错误认识**："并发和并行是一样的，都表示同时执行。"

**事实**：并发是**结构上**同时处理多个任务（可以单核交替），并行是**物理上**同时执行多个任务（需要多核）。

```python
# 并发：单线程交替执行
async def main():
    await asyncio.gather(task_a(), task_b(), task_c())
    # 实际上是交替执行，不是同时执行

# 并行：多进程真正同时执行
from multiprocessing import Pool
with Pool(3) as p:
    p.map(heavy_work, [data_a, data_b, data_c])
    # 三个进程在同一时刻各自计算
```

### 误区 3：async def 里面的代码天然不会阻塞

**错误认识**："只要函数是 `async def` 定义的，里面的代码就不会阻塞。"

**事实**：`async def` 只是定义了一个协程函数，里面的代码是否会阻塞取决于具体调用了什么。

```python
import asyncio
import time

async def bad_task():
    """这是 async def，但会阻塞！"""
    time.sleep(5)          # 阻塞！整个事件循环卡住 5 秒
    return "done"

async def good_task():
    """这才是真正的异步"""
    await asyncio.sleep(5) # 非阻塞，其他协程可以运行
    return "done"
```

**判断标准**：只有遇到 `await` 时，协程才会让出控制权。没有 `await` 的代码段会一直占用事件循环。

### 误区 4：异步会让网络请求本身变快

**错误认识**："用 asyncio 后，网络请求会更快。"

**事实**：网络请求的速度取决于网络延迟，与是否使用异步无关。异步只是让多个请求的等待时间可以**重叠**，从而减少总等待时间。

```python
# 每个请求仍然需要 2 秒
# 异步只是让 3 个 2 秒的请求重叠成 1 个 2 秒

# 顺序：2 + 2 + 2 = 6 秒
# 并发：max(2, 2, 2) = 2 秒

# 单个请求的速度没有变化！
```

### 误区 5：CPU 计算也可以通过 await 自动并行

**错误认识**："我在 for 循环里加 `await`，CPU 计算就能并行了。"

**事实**：`await` 只能让出 I/O 等待时间，不能让 CPU 计算并行。CPU 并行需要多进程。

```python
# 这样做没有用
async def compute(n):
    result = 0
    for i in range(n):
        result += i        # 纯计算
        # await 在这里没意义，因为没有异步操作
    return result

# 正确方案：用多进程
from concurrent.futures import ProcessPoolExecutor

async def main():
    with ProcessPoolExecutor() as pool:
        loop = asyncio.get_event_loop()
        results = await asyncio.gather(
            loop.run_in_executor(pool, compute, 10_000_000),
            loop.run_in_executor(pool, compute, 10_000_000),
        )
```

### 误区 6：在 async def 中调用其他 async 函数就会自动并发

**错误认识**："只要调用的是 async 函数，就会自动并发执行。"

**事实**：调用 `async` 函数只是创建了一个协程对象，必须用 `await`、`create_task()` 或 `gather()` 才能实际执行和并发。

```python
async def fetch_a():
    await asyncio.sleep(2)
    return "A"

async def fetch_b():
    await asyncio.sleep(2)
    return "B"

async def wrong():
    """顺序执行，不是并发！"""
    result_a = await fetch_a()  # 等 A 完成
    result_b = await fetch_b()  # 再等 B 完成
    # 总耗时 4 秒

async def right():
    """并发执行"""
    result_a, result_b = await asyncio.gather(
        fetch_a(), fetch_b()    # 同时开始
    )
    # 总耗时 2 秒
```

---

## 本节检查题

### 第 1 题

下面代码属于阻塞还是非阻塞？当前线程在这三秒里能否继续执行后面的代码？

```python
import time
print("开始")
time.sleep(3)
print("结束")
```

<details>
<summary>查看答案</summary>

**阻塞。** `time.sleep(3)` 是同步阻塞调用，当前线程会完全停住 3 秒，无法执行任何后续代码。3 秒后才会打印"结束"。

</details>

### 第 2 题

三个网络请求每个都需要等待两秒。同步顺序执行大约需要几秒？异步并发大约需要几秒？

<details>
<summary>查看答案</summary>

- **同步顺序执行**：2 + 2 + 2 = **6 秒**（三个请求排队执行）
- **异步并发执行**：max(2, 2, 2) = **2 秒**（三个请求同时发出，等待时间重叠）

</details>

### 第 3 题

判断：一个线程在任务 A、B、C 之间切换执行属于并发还是并行？三个 CPU 核心分别同时执行任务 A、B、C 呢？

<details>
<summary>查看答案</summary>

- **一个线程切换执行**：这是**并发（Concurrency）**。在任意瞬间只有一个任务在执行，但一段时间内三个任务都在推进。
- **三个核心同时执行**：这是**并行（Parallelism）**。在同一时刻，三个任务真正在同时执行。

</details>

### 第 4 题

下面这个异步函数有什么问题？如何修改？

```python
import time
async def download():
    print("开始下载")
    time.sleep(5)
    print("下载完成")
```

<details>
<summary>查看答案</summary>

**问题**：`time.sleep(5)` 是同步阻塞函数，会阻塞整个事件循环 5 秒，导致其他协程无法执行。

**修改**：

```python
import asyncio

async def download():
    print("开始下载")
    await asyncio.sleep(5)  # 改为异步等待
    print("下载完成")
```

</details>

### 第 5 题

下面代码总耗时大约是多少？

```python
import asyncio
async def main():
    await asyncio.gather(
        asyncio.sleep(2),
        asyncio.sleep(5),
        asyncio.sleep(1),
    )
asyncio.run(main())
```

<details>
<summary>查看答案</summary>

**约 5 秒。** `asyncio.gather()` 并发执行所有任务，总耗时等于最慢任务的耗时：max(2, 5, 1) = 5 秒。

</details>

### 第 6 题

下面的代码是同步非阻塞还是异步非阻塞？为什么？

```python
import socket
import time

sock = socket.socket()
sock.setblocking(False)

while True:
    try:
        data = sock.recv(1024)
        if data:
            break
    except BlockingIOError:
        time.sleep(0.1)
```

<details>
<summary>查看答案</summary>

**同步非阻塞。** `sock.recv()` 是非阻塞的（立即返回），但整个过程是同步的——调用者在当前流程中主动轮询检查结果，而不是通过事件通知获得结果。

</details>

### 第 7 题

下面哪段代码是真正的并发？为什么？

```python
# 代码 A
async def main():
    a = await fetch_a()
    b = await fetch_b()
    c = await fetch_c()

# 代码 B
async def main():
    a, b, c = await asyncio.gather(
        fetch_a(), fetch_b(), fetch_c()
    )
```

<details>
<summary>查看答案</summary>

**代码 B 是并发。** `asyncio.gather()` 同时启动三个任务，它们的等待时间可以重叠。

代码 A 是顺序执行：`await fetch_a()` 完成后才开始 `await fetch_b()`，以此类推。每个 `await` 都会等待前一个任务完成。

</details>

---

## 编程练习

### 练习 1：同步版本计时

实现同步任务计时程序，记录三个任务的总耗时：

```python
import time

def task(name, delay):
    print(f"任务 {name} 开始执行")
    time.sleep(delay)
    print(f"任务 {name} 执行完成")

start = time.perf_counter()
task("A", 1)
task("B", 2)
task("C", 1)
elapsed = time.perf_counter() - start
print(f"总耗时：{elapsed:.2f} 秒")
```

预期输出：总耗时 = 1 + 2 + 1 = **4 秒**。

### 练习 2：异步并发版本

将练习 1 改为异步并发版本，观察总耗时变化：

```python
import asyncio
import time

async def task(name, delay):
    print(f"任务 {name} 开始执行")
    await asyncio.sleep(delay)
    print(f"任务 {name} 执行完成")

async def main():
    start = time.perf_counter()
    await asyncio.gather(
        task("A", 1),
        task("B", 2),
        task("C", 1),
    )
    elapsed = time.perf_counter() - start
    print(f"总耗时：{elapsed:.2f} 秒")

asyncio.run(main())
```

预期输出：总耗时 = max(1, 2, 1) = **2 秒**。

### 练习 3：检测阻塞错误

下面代码有问题，请找出并修复：

```python
import asyncio
import time

async def worker(name, delay):
    print(f"[{name}] 开始工作")
    time.sleep(delay)  # 有什么问题？
    print(f"[{name}] 工作完成")

async def main():
    start = time.perf_counter()
    await asyncio.gather(
        worker("A", 2),
        worker("B", 2),
        worker("C", 2),
    )
    print(f"总耗时：{time.perf_counter() - start:.2f} 秒")

asyncio.run(main())
```

**任务**：
1. 运行代码，记录实际耗时
2. 找出问题
3. 修复代码
4. 再次运行，确认耗时为 2 秒

### 练习 4：思考并发边界

假设你要同时下载 1000 个文件，每个文件需要 1 秒。直接用 `asyncio.gather()` 同时创建 1000 个任务，可能会有什么问题？应该如何处理？

<details>
<summary>提示</summary>

同时打开 1000 个网络连接可能导致：
- 服务器拒绝连接（连接数限制）
- 本地端口耗尽
- 内存占用过高

解决方案：使用信号量（`asyncio.Semaphore`）限制并发数量。

```python
import asyncio

async def download(sem, url):
    async with sem:  # 限制并发数
        print(f"下载 {url}")
        await asyncio.sleep(1)
        return f"{url} 完成"

async def main():
    sem = asyncio.Semaphore(10)  # 最多同时 10 个
    urls = [f"https://example.com/file{i}" for i in range(1000)]

    results = await asyncio.gather(
        *[download(sem, url) for url in urls]
    )
    print(f"完成 {len(results)} 个下载")

asyncio.run(main())
```

</details>

---

## 本章小结

本章介绍了异步编程的六个核心概念。它们分属三个不同的维度：

| 维度 | 概念 | 核心问题 |
|------|------|---------|
| 结果交付 | 同步 vs 异步 | 调用者如何获得结果？ |
| 线程行为 | 阻塞 vs 非阻塞 | 等待期间线程是否停住？ |
| 任务推进 | 并发 vs 并行 | 多个任务如何在时间上推进？ |

**关键要点**：

1. **阻塞/非阻塞**描述等待期间线程是否被卡住
2. **同步/异步**描述结果如何交付给调用者
3. **并发/并行**描述多个任务在时间上如何推进
4. 同步程序中 I/O 等待是串行的，异步程序中可以重叠
5. `time.sleep()` 阻塞事件循环，`await asyncio.sleep()` 不会
6. 异步主要提升 I/O 密集型场景的吞吐量
7. CPU 密集型任务应该使用多进程，而不是 asyncio
8. `async def` 不代表代码不会阻塞，关键是看有没有 `await` 异步操作

**任务类型与技术选择**：

```
I/O 密集型 ──→ asyncio（高并发）或 threading（少量任务）
CPU 密集型 ──→ multiprocessing（绕过 GIL）
混合型    ──→ asyncio + ProcessPoolExecutor
```

---

## 思考题

### 1. 如果一个程序只有一个小任务，用异步有意义吗？

没有明显意义。异步的优势在于多个任务的等待时间可以重叠。只有一个任务时，总耗时不变，反而多了事件循环的调度开销。这就像你为了取一个快递专门建立了一套"号码牌通知系统"——完全没必要。

### 2. 为什么在 async def 中使用 time.sleep() 是错误的？

因为 `time.sleep()` 是同步阻塞函数，它会阻塞当前线程（也就是事件循环所在的线程）。事件循环被阻塞后，无法调度其他协程执行，导致所有协程都停住。应该使用 `await asyncio.sleep()`，它会在等待期间让出控制权，让事件循环去执行其他协程。

### 3. 并发一定需要多线程吗？

不一定。Python asyncio 通过**单线程事件循环**实现并发，多个协程在一个线程中交替执行。切换发生在协程遇到 `await` 时，由事件循环负责调度。这比多线程更轻量，因为没有线程切换的开销和锁的问题。

### 4. 异步编程适合什么样的任务？

适合 **I/O 密集型** 任务，具体来说：
- 任务数量多（>10 个并发任务效果明显）
- 任务之间相互独立（不需要等待彼此的结果）
- 大量时间在等待外部资源（网络、磁盘、数据库）

典型场景：Web 服务器、爬虫、API 网关、消息处理。

### 5. 同步非阻塞和异步非阻塞有什么区别？

| | 同步非阻塞 | 异步非阻塞 |
|--|----------|----------|
| 结果获取 | 调用者主动轮询（Polling） | 事件通知 / 回调 / await |
| 调用者行为 | 必须反复检查状态 | 可以完全去做其他事 |
| 效率 | 较低（轮询有开销） | 较高（事件驱动） |
| 典型代码 | `while not ready: check()` | `await result` |

### 6. 如果一个程序同时需要 I/O 和 CPU 计算，应该怎么设计？

混合方案：用 asyncio 处理 I/O 等待部分，用 `ProcessPoolExecutor` 处理 CPU 计算部分。

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

def cpu_heavy(data):
    """CPU 密集型计算"""
    return sum(i * i for i in data)

async def main():
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor() as pool:
        # I/O 任务用 asyncio
        io_data = await fetch_from_api()

        # CPU 计算用进程池
        result = await loop.run_in_executor(pool, cpu_heavy, io_data)

        # I/O 任务用 asyncio
        await save_to_db(result)
```

这样既利用了 asyncio 的 I/O 并发优势，又利用了多进程的 CPU 并行能力。

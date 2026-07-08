---
title: "事件循环如何调度 Task"
description: "深入理解事件循环的调度机制、执行顺序、sleep(0) 的作用、阻塞检测与事件循环生命周期"
updated: "2026-07-07"
---

# 事件循环如何调度 Task

> **学习目标**：
> - 理解事件循环的本质——它是 asyncio 的心脏，所有协程的调度都由它驱动
> - 掌握事件循环一轮调度的完整过程：选取就绪任务 → 执行到 yield → 下一个任务
> - 理解 `create_task()` 不等于立即执行，任务何时真正开始取决于调度时机
> - 通过实验观察多个任务的调度顺序，理解 FIFO 就绪队列
> - 掌握 `await asyncio.sleep(0)` 的作用——立即让出控制权
> - 深刻理解事件循环是单线程的，同一时刻只运行一个任务
> - 了解长时间运行的任务如何阻塞事件循环，导致其他任务饿死
> - 学会使用 `enable_debug()` 和慢回调检测来发现阻塞
> - 掌握 `get_running_loop()` 的用法与适用场景
> - 理解事件循环的完整生命周期：创建、运行、关闭
> - 区分底层 API 与高层 API 的适用场景
> - 识别并纠正常见的事件循环误区

前面几章我们学会了 `async/await`、`create_task()`、`gather()`，也理解了协程、Task、Future 的关系。但有一个核心问题一直没有深入：**事件循环到底是怎么决定"下一个该执行谁"的？**

这一章就把事件循环的调度机制彻底搞清楚。

---

## 5.1 什么是事件循环——asyncio 的心脏

如果把 asyncio 比作一家餐厅，事件循环就是那位**前台调度员**。他不做具体的工作（不煮面、不切菜），但他决定**谁先做、谁后做、什么时候切换**。

### 事件循环的核心职责

```
┌─────────────────────────────────────────────────────────────────┐
│                        事件循环 (Event Loop)                     │
│                                                                 │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                 │
│   │  Task A   │    │  Task B   │    │  Task C   │                │
│   │  (就绪)   │    │  (等待中)  │    │  (就绪)   │                │
│   └─────┬────┘    └─────┬────┘    └─────┬────┘                 │
│         │               │               │                       │
│         └───────────────┼───────────────┘                       │
│                         ▼                                       │
│              ┌─────────────────────┐                            │
│              │    调度器核心循环     │                            │
│              │                     │                            │
│              │  1. 从就绪队列取任务  │                            │
│              │  2. 执行该任务       │                            │
│              │  3. 任务 await？     │                            │
│              │     → 移出就绪队列   │                            │
│              │  4. 检查完成的回调   │                            │
│              │  5. 检查 I/O 事件    │                            │
│              │  6. 回到第 1 步      │                            │
│              └─────────────────────┘                            │
│                                                                 │
│   同一时刻只执行一个任务，单线程，不存在竞态条件                     │
└─────────────────────────────────────────────────────────────────┘
```

### 事件循环管理的三类事物

| 类别 | 说明 | 例子 |
|------|------|------|
| **就绪的协程/Task** | 可以立即执行的协程 | 刚创建的 Task、sleep 到期的 Task |
| **I/O 事件** | 等待网络、文件等 I/O 完成 | socket 可读、HTTP 响应到达 |
| **定时器** | 在未来某个时间点触发 | `asyncio.sleep(2)` 注册的 2 秒后回调 |

事件循环每一轮都会检查这三类事物，按优先级调度执行。

### 代码视角

```python
import asyncio

async def hello():
    print("Hello")
    await asyncio.sleep(1)
    print("World")

asyncio.run(hello())
```

当 `asyncio.run(hello())` 执行时，背后发生了什么？

```
asyncio.run()
    │
    ├── 创建一个新的事件循环
    │
    ├── 创建一个 Task 来运行 hello() 协程
    │
    ├── 进入事件循环主循环 ─────────────────────────────┐
    │   │                                               │
    │   │   循环体：                                     │
    │   │   ┌─────────────────────────────────────┐     │
    │   │   │ 1. 就绪队列有 Task？                  │     │
    │   │   │    → 取出，执行到遇到 await            │     │
    │   │   │ 2. 有到期的定时器？                    │     │
    │   │   │    → 触发回调，对应 Task 移入就绪队列   │     │
    │   │   │ 3. 有 I/O 事件？                      │     │
    │   │   │    → 触发回调，对应 Task 移入就绪队列   │     │
    │   │   │ 4. 就绪队列为空且没有待处理事件？       │     │
    │   │   │    → 退出循环                          │     │
    │   │   └─────────────────────────────────────┘     │
    │   │                                               │
    │   └───────────────────────────────────────────────┘
    │
    └── 关闭事件循环
```

<div data-component="EventLoopVisualization"></div>

### 关键认知

事件循环**不是**一个独立运行的后台线程。它就是当前线程中的一段 `while` 循环。当你调用 `asyncio.run()` 时，你的代码就"进入"了这个循环，直到所有任务完成才会退出。

```python
# 伪代码：事件循环的核心逻辑
class EventLoop:
    def run_forever(self):
        while self._ready or self._scheduled:
            # 1. 执行所有就绪的回调
            while self._ready:
                callback = self._ready.popleft()
                callback()

            # 2. 计算最近的定时器到期时间
            timeout = self._calculate_timeout()

            # 3. 用 select/poll 等待 I/O 事件（或超时）
            self._poll(timeout)

            # 4. 将到期的定时器对应的回调移入就绪队列
            self._process_scheduled()
```

这不是真实代码，但反映了事件循环的核心逻辑。后面我们会逐层深入。

</div>
---

## 5.2 事件循环的一轮调度——pick ready → run until yield → next task

理解事件循环最好的方式是跟踪它的一轮完整调度过程。

### 一轮调度的完整流程

```
事件循环开始新一轮
    │
    ▼
┌─ 就绪队列 (Ready Queue) ─────────────────┐
│                                           │
│   [Task A] [Task C] [Task D]              │
│    ↑ 队首                                  │
│                                           │
└───────────────────────────────────────────┘
    │
    │ 取出 Task A
    ▼
┌─ 执行 Task A ─────────────────────────────┐
│                                           │
│   async def task_a():                     │
│       print("A: step 1")    ← 执行到这里   │
│       await asyncio.sleep(1) ← 遇到 await │
│       print("A: step 2")                  │
│                                           │
└───────────────────────────────────────────┘
    │
    │ Task A 遇到 await，暂停
    │ → 注册 sleep(1) 的定时器
    │ → Task A 不再就绪
    │
    ▼
┌─ 就绪队列更新 ────────────────────────────┐
│                                           │
│   [Task C] [Task D]                       │
│    ↑ 队首                                  │
│                                           │
└───────────────────────────────────────────┘
    │
    │ 取出 Task C
    ▼
┌─ 执行 Task C ─────────────────────────────┐
│                                           │
│   async def task_c():                     │
│       print("C: step 1")    ← 执行到这里   │
│       await some_io()       ← 遇到 await  │
│       print("C: step 2")                  │
│                                           │
└───────────────────────────────────────────┘
    │
    │ Task C 遇到 await，暂停
    │ → 注册 I/O 等待
    │ → Task C 不再就绪
    │
    ▼
  ... 继续处理 Task D ...
```

### 三种调度结果

当事件循环执行一个 Task 时，有三种可能的结果：

| 结果 | 说明 | Task 的去向 |
|------|------|------------|
| **Task 完成** | 协程执行完毕，返回值或抛出异常 | 从就绪队列移除，触发 done 回调 |
| **Task 遇到 await** | 协程遇到 `await` 某个未完成的东西 | 暂停，等待被唤醒后重新进入就绪队列 |
| **Task 主动 yield** | 协程调用 `await asyncio.sleep(0)` | 暂停，**立即**重新进入就绪队列末尾 |

### 完整代码跟踪

```python
import asyncio

async def task_a():
    print("[A] 开始执行")
    await asyncio.sleep(0)  # 让出控制权
    print("[A] 恢复执行")
    return "A 完成"

async def task_b():
    print("[B] 开始执行")
    await asyncio.sleep(0)  # 让出控制权
    print("[B] 恢复执行")
    return "B 完成"

async def main():
    # 创建两个 Task
    ta = asyncio.create_task(task_a())
    tb = asyncio.create_task(task_b())

    # 等待两个 Task 完成
    await asyncio.gather(ta, tb)

asyncio.run(main())
```

输出：

```
[A] 开始执行
[B] 开始执行
[A] 恢复执行
[B] 恢复执行
```

### 调度过程解析

```
时间线：
                                                       
  事件循环开始                                          
  │                                                   
  ├── 取出 main 的 Task，执行到 create_task(task_a)     
  │   → task_a 被加入就绪队列                            
  │                                                   
  ├── 继续执行 main，到 create_task(task_b)             
  │   → task_b 被加入就绪队列                            
  │                                                   
  ├── 继续执行 main，到 await gather(ta, tb)            
  │   → main 暂停（等待 gather 完成）                    
  │                                                   
  ├── 就绪队列：[task_a, task_b]                        
  │                                                   
  ├── 取出 task_a，执行 print("[A] 开始执行")            
  │   遇到 await sleep(0) → task_a 暂停                 
  │   sleep(0) 立即到期 → task_a 重新进入就绪队列末尾     
  │                                                   
  ├── 就绪队列：[task_b, task_a]                        
  │                                                   
  ├── 取出 task_b，执行 print("[B] 开始执行")            
  │   遇到 await sleep(0) → task_b 暂停                 
  │   sleep(0) 立即到期 → task_b 重新进入就绪队列末尾     
  │                                                   
  ├── 就绪队列：[task_a, task_b]                        
  │                                                   
  ├── 取出 task_a，执行 print("[A] 恢复执行")            
  │   task_a 完成 → 从就绪队列移除                       
  │                                                   
  ├── 就绪队列：[task_b]                               
  │                                                   
  ├── 取出 task_b，执行 print("[B] 恢复执行")            
  │   task_b 完成 → 从就绪队列移除                       
  │                                                   
  ├── gather 完成，main 恢复                            
  │   main 完成                                        
  │                                                   
  └── 就绪队列为空，事件循环退出                          
```

---

## 5.3 任务何时开始执行——create_task 不意味着立即执行

这是一个非常常见的误区：很多人以为 `create_task()` 会让任务立即开始运行。事实并非如此。

### create_task() 做了什么

```python
task = asyncio.create_task(some_coroutine())
```

这行代码做了两件事：

1. 把协程包装成一个 Task 对象
2. 把 Task **加入事件循环的就绪队列**

注意：**加入就绪队列不等于立即执行**。Task 要等到事件循环调度到它时才会真正运行。

### 误解演示

```python
import asyncio

async def worker(name):
    print(f"[{name}] 开始工作")
    await asyncio.sleep(1)
    print(f"[{name}] 工作完成")

async def main():
    print("[main] 创建任务之前")

    task = asyncio.create_task(worker("A"))

    print("[main] 创建任务之后")
    print("[main] 任务还没有开始执行！")

    # 如果到这里就结束了，worker 可能根本没机会执行
    # 因为 main 完成后事件循环就退出了

asyncio.run(main())
```

你可能期望看到 `[A] 开始工作`，但实际上**可能看不到**。因为 `create_task()` 只是把 Task 放进了就绪队列，而 `main()` 在打印完之后就结束了，事件循环随之退出。

### 正确的做法

```python
import asyncio

async def worker(name):
    print(f"[{name}] 开始工作")
    await asyncio.sleep(1)
    print(f"[{name}] 工作完成")

async def main():
    print("[main] 创建任务之前")

    task = asyncio.create_task(worker("A"))

    print("[main] 创建任务之后")

    # 必须 await 任务，确保它有机会执行
    await task

asyncio.run(main())
```

输出：

```
[main] 创建任务之前
[main] 创建任务之后
[A] 开始工作
[A] 工作完成
```

### 关键区分

<div data-component="TaskSchedulingDemo"></div>

| 操作 | 效果 |
|------|------|
| `coro = worker()` | 创建协程对象，**不执行任何代码** |
| `task = create_task(worker())` | 创建 Task，加入就绪队列，**但不执行** |
| `await task` | 等待 Task 完成，**期间事件循环会调度它** |
| `await worker()` | 直接等待协程完成，**当前协程暂停直到它完成** |

### 时序对比

```python
# 情况 1：create_task 后立即 await
async def case1():
    task = asyncio.create_task(worker("A"))
    await task  # 立即等待，A 会在 await 处开始执行

# 情况 2：create_task 后做其他事，再 await
async def case2():
    task = asyncio.create_task(worker("A"))
    print("做其他事情...")  # A 还没开始执行
    print("继续做其他事情...")  # A 还没开始执行
    await task  # 这里才会调度 A 执行

# 情况 3：create_task 后不 await（危险！）
async def case3():
    task = asyncio.create_task(worker("A"))
    # 不 await，main 结束后事件循环退出
    # Task A 可能被取消或根本没执行
```

### 一个微妙的时序问题

```python
import asyncio

async def worker(name):
    print(f"[{name}] 执行")
    return name

async def main():
    # 创建两个任务
    t1 = asyncio.create_task(worker("A"))
    t2 = asyncio.create_task(worker("B"))

    # 此时 A 和 B 都在就绪队列中，但都还没执行
    print("[main] 两个任务已创建")

    # 当执行到 await 时，事件循环才开始调度
    await asyncio.gather(t1, t2)

asyncio.run(main())
```

输出：

```
[main] 两个任务已创建
[A] 执行
[B] 执行
```

注意 `[main] 两个任务已创建` 打印在 `[A] 执行` 之前。这证明 `create_task()` 不会立即执行任务。

</div>
---

## 5.4 调度顺序实验——多个 create_task，观察顺序

让我们通过实验来观察事件循环的调度顺序。

### 实验 1：基本调度顺序

```python
import asyncio

async def task(name, steps):
    for i in range(steps):
        print(f"[{name}] 步骤 {i+1}")
        await asyncio.sleep(0)  # 让出控制权
    print(f"[{name}] 完成")

async def main():
    t1 = asyncio.create_task(task("A", 3))
    t2 = asyncio.create_task(task("B", 3))
    t3 = asyncio.create_task(task("C", 3))

    await asyncio.gather(t1, t2, t3)

asyncio.run(main())
```

输出：

```
[A] 步骤 1
[B] 步骤 1
[C] 步骤 1
[A] 步骤 2
[B] 步骤 2
[C] 步骤 2
[A] 步骤 3
[B] 步骤 3
[C] 步骤 3
[A] 完成
[B] 完成
[C] 完成
```

**规律**：A → B → C → A → B → C，严格轮转。

**原因**：
1. `create_task(task("A", 3))` → A 进入就绪队列
2. `create_task(task("B", 3))` → B 进入就绪队列
3. `create_task(task("C", 3))` → C 进入就绪队列
4. 就绪队列：[A, B, C]
5. 取出 A，执行步骤 1，遇到 `sleep(0)` → A 重新进入队列末尾
6. 就绪队列：[B, C, A]
7. 取出 B，执行步骤 1，遇到 `sleep(0)` → B 重新进入队列末尾
8. 就绪队列：[C, A, B]
9. ...以此类推

### 实验 2：不同步长的任务

```python
import asyncio

async def task(name, steps):
    for i in range(steps):
        print(f"[{name}] 步骤 {i+1}")
        await asyncio.sleep(0)
    print(f"[{name}] 完成")

async def main():
    t1 = asyncio.create_task(task("A", 1))  # 只有 1 步
    t2 = asyncio.create_task(task("B", 3))  # 有 3 步
    t3 = asyncio.create_task(task("C", 2))  # 有 2 步

    await asyncio.gather(t1, t2, t3)

asyncio.run(main())
```

输出：

```
[A] 步骤 1
[B] 步骤 1
[C] 步骤 1
[A] 完成
[B] 步骤 2
[C] 步骤 2
[B] 步骤 3
[C] 完成
[B] 完成
```

**观察**：
- A 只有一步，第一次执行就完成了，之后不再参与调度
- B 和 C 继续轮转，直到各自完成
- 完成顺序：A → C → B（取决于步数）

### 实验 3：没有 sleep(0) 的情况

```python
import asyncio

async def task(name, steps):
    for i in range(steps):
        print(f"[{name}] 步骤 {i+1}")
        # 没有 await，不让出控制权
    print(f"[{name}] 完成")

async def main():
    t1 = asyncio.create_task(task("A", 3))
    t2 = asyncio.create_task(task("B", 3))
    t3 = asyncio.create_task(task("C", 3))

    await asyncio.gather(t1, t2, t3)

asyncio.run(main())
```

输出：

```
[A] 步骤 1
[A] 步骤 2
[A] 步骤 3
[A] 完成
[B] 步骤 1
[B] 步骤 2
[B] 步骤 3
[B] 完成
[C] 步骤 1
[C] 步骤 2
[C] 步骤 3
[C] 完成
```

**完全不同！** A 执行完所有步骤后，才轮到 B，然后是 C。

**原因**：没有 `await`，协程不会让出控制权，事件循环无法切换到其他任务。

### 实验 4：交错的 create_task

```python
import asyncio

async def task(name):
    print(f"[{name}] 开始")
    await asyncio.sleep(0)
    print(f"[{name}] 结束")

async def main():
    t1 = asyncio.create_task(task("A"))
    print("[main] A 已创建")

    t2 = asyncio.create_task(task("B"))
    print("[main] B 已创建")

    t3 = asyncio.create_task(task("C"))
    print("[main] C 已创建")

    await asyncio.gather(t1, t2, t3)

asyncio.run(main())
```

输出：

```
[main] A 已创建
[main] B 已创建
[main] C 已创建
[A] 开始
[B] 开始
[C] 开始
[A] 结束
[B] 结束
[C] 结束
```

**关键发现**：三个 `[main]` 打印在三个 `[X] 开始` 之前。

这再次证明：`create_task()` 只是把任务放入就绪队列，**当前协程（main）继续执行**，直到 main 遇到 `await gather(...)` 时才让出控制权，事件循环才开始调度就绪队列中的任务。

### 调度顺序总结

| 场景 | 调度顺序 |
|------|---------|
| 多个 `create_task` + 有 `await` | FIFO 轮转，严格交替 |
| 多个 `create_task` + 无 `await` | 按创建顺序串行执行 |
| `create_task` 之间有其他代码 | 先执行完当前代码，再调度任务 |
| 任务完成时间不同 | 完成的任务退出调度，剩余继续轮转 |

<div data-component="SleepZeroDemo"></div>

---

</div>
## 5.5 await asyncio.sleep(0) 的作用——立即让出控制权

`await asyncio.sleep(0)` 是 asyncio 中一个非常特殊的用法。它的含义是："我**现在**就让出控制权，但**立即**让我重新就绪。"

### sleep(0) 的完整含义

```python
await asyncio.sleep(0)
```

这行代码做了三件事：

1. **暂停当前协程**——让出 CPU 给事件循环
2. **注册一个 0 秒的定时器**——"立即"到期
3. **事件循环处理完当前轮次后**，发现定时器到期，把当前协程重新放入就绪队列末尾

### 为什么需要 sleep(0)

```python
# 没有 sleep(0)：独占 CPU
async def greedy():
    while True:
        do_some_work()  # 永不让出控制权
        # 其他任务永远得不到执行！

# 有 sleep(0)：协作式让出
async def cooperative():
    while True:
        do_some_work()
        await asyncio.sleep(0)  # 每次迭代让出一次
        # 其他任务有机会执行
```

### 使用场景

**场景 1：长时间循环中让出控制权**

```python
import asyncio

async def process_items(items):
    results = []
    for i, item in enumerate(items):
        result = heavy_computation(item)
        results.append(result)

        # 每处理 100 个项目让出一次控制权
        if i % 100 == 0:
            await asyncio.sleep(0)

    return results

async def background_monitor():
    while True:
        print("[监控] 系统正常运行中...")
        await asyncio.sleep(2)

async def main():
    # 两个任务并发执行
    items = list(range(10000))
    t1 = asyncio.create_task(process_items(items))
    t2 = asyncio.create_task(background_monitor())

    result = await t1
    t2.cancel()
    print(f"处理了 {len(result)} 个项目")

asyncio.run(main())
```

**场景 2：确保其他任务有机会执行**

```python
import asyncio

async def producer(queue):
    for i in range(5):
        await queue.put(i)
        print(f"[生产者] 生产了 {i}")
        await asyncio.sleep(0)  # 让消费者有机会执行

async def consumer(queue):
    while True:
        item = await queue.get()
        print(f"[消费者] 消费了 {item}")
        queue.task_done()

async def main():
    queue = asyncio.Queue()
    consumer_task = asyncio.create_task(consumer(queue))

    await producer(queue)
    await queue.join()
    consumer_task.cancel()

asyncio.run(main())
```

**场景 3：打破递归/循环的"僵局"**

```python
import asyncio

async def task_a():
    print("[A] 第一步")
    await asyncio.sleep(0)
    print("[A] 第二步")

async def task_b():
    print("[B] 第一步")
    await asyncio.sleep(0)
    print("[B] 第二步")

async def main():
    t1 = asyncio.create_task(task_a())
    t2 = asyncio.create_task(task_b())
    await asyncio.gather(t1, t2)

asyncio.run(main())
```

输出：

```
[A] 第一步
[B] 第一步
[A] 第二步
[B] 第二步
```

没有 `sleep(0)`，输出会是 `A 第一步 → A 第二步 → B 第一步 → B 第二步`。

### sleep(0) vs sleep(正数)

| | `sleep(0)` | `sleep(1)` |
|--|-----------|-----------|
| 让出控制权 | 是 | 是 |
| 等待时间 | 0 秒（立即） | 1 秒 |
| 重新就绪时机 | 当前轮次结束后 | 1 秒后 |
| 主要用途 | 协作式调度 | 模拟 I/O 等待 |

### 重要提醒

`sleep(0)` 不等于"立即执行下一步"。它意味着"让其他就绪的任务先执行一轮，然后我再继续"。

```python
import asyncio

async def task(name, count):
    for i in range(count):
        print(f"[{name}] 第 {i+1} 次")
        await asyncio.sleep(0)
    print(f"[{name}] 完成")

async def main():
    t1 = asyncio.create_task(task("A", 2))
    t2 = asyncio.create_task(task("B", 2))
    t3 = asyncio.create_task(task("C", 2))
    await asyncio.gather(t1, t2, t3)

asyncio.run(main())
```

输出：

```
[A] 第 1 次
[B] 第 1 次
[C] 第 1 次
[A] 第 2 次
[B] 第 2 次
[C] 第 2 次
[A] 完成
[B] 完成
[C] 完成
```

每个任务执行一次后让出，三个任务严格轮转。

---

## 5.6 事件循环只调度一个任务——单线程，同一时刻一个

这是 asyncio 最重要的特性之一：**事件循环是单线程的，同一时刻只运行一个任务**。

### 为什么是单线程

```
多线程：                          asyncio 单线程：
                                  
  线程 1: ████░░░░████            事件循环: ████░░████░░████
  线程 2: ░░████░░░░██            
  线程 3: ░░░░░░████░░            任务 A:   ████    ████
                                  任务 B:         ████    ████
  → 多个 CPU 核心同时执行          
  → 需要锁来保护共享资源            → 一个线程，交替执行
                                   → 不需要锁（理论上）
```

### 单线程的好处

| 优势 | 说明 |
|------|------|
| **无竞态条件** | 同一时刻只有一个任务在运行，不会出现两个任务同时修改变量 |
| **无需加锁** | 不需要 `threading.Lock()`，代码更简洁 |
| **切换开销小** | 协程切换只涉及保存/恢复少量状态，比线程切换快 100 倍以上 |
| **内存占用低** | 一个线程管理成千上万个协程，每个协程只需几 KB |

### 单线程的限制

| 限制 | 说明 |
|------|------|
| **不能利用多核 CPU** | 一个线程只能用一个 CPU 核心 |
| **CPU 密集型任务会阻塞** | 长时间计算会卡住整个事件循环 |
| **阻塞调用会卡住一切** | `time.sleep()`、同步 I/O 都会阻塞事件循环 |

### 验证：同一时刻只有一个任务

```python
import asyncio
import time

async def task(name):
    print(f"[{name}] 开始 {time.strftime('%H:%M:%S')}")
    # 模拟 CPU 密集型工作
    count = 0
    for i in range(10_000_000):
        count += i
    print(f"[{name}] 结束 {time.strftime('%H:%M:%S')}")

async def main():
    t1 = asyncio.create_task(task("A"))
    t2 = asyncio.create_task(task("B"))

    await asyncio.gather(t1, t2)

asyncio.run(main())
```

输出：

```
[A] 开始 14:30:00
[A] 结束 14:30:02    ← A 花了 2 秒
[B] 开始 14:30:02    ← B 在 A 结束后才开始
[B] 结束 14:30:04    ← B 也花了 2 秒
```

总耗时 4 秒（2 + 2），而不是 2 秒（max）。因为没有 `await`，A 独占了 CPU，B 只能等待。

### 与多线程对比

```python
import threading
import time

def task(name):
    print(f"[{name}] 开始 {time.strftime('%H:%M:%S')}")
    count = 0
    for i in range(10_000_000):
        count += i
    print(f"[{name}] 结束 {time.strftime('%H:%M:%S')}")

t1 = threading.Thread(target=task, args=("A",))
t2 = threading.Thread(target=task, args=("B",))

t1.start()
t2.start()
t1.join()
t2.join()
```

在多核机器上，输出可能是：

```
[A] 开始 14:30:00
[B] 开始 14:30:00    ← 同时开始
[A] 结束 14:30:02
[B] 结束 14:30:02    ← 同时结束
```

总耗时 2 秒（并行执行）。但对于 I/O 密集型任务，asyncio 的效率远高于多线程。

### 核心原则

```
asyncio 的单线程模型：
                                  
  ┌─────────────────────────────────────────┐
  │           单个线程                        │
  │                                         │
  │   ┌──────┐ ┌──────┐ ┌──────┐           │
  │   │Task A│ │Task B│ │Task C│           │
  │   └──┬───┘ └──┬───┘ └──┬───┘           │
  │      │        │        │                │
  │      └────────┼────────┘                │
  │               ▼                         │
  │        ┌─────────────┐                  │
  │        │  事件循环     │                  │
  │        │  一次只执行   │                  │
  │        │  一个任务     │                  │
  │        └─────────────┘                  │
  │                                         │
  │   I/O 等待时切换 → 高效                   │
  │   CPU 计算时阻塞 → 低效                   │
  └─────────────────────────────────────────┘
```

---

## 5.7 长时间运行的任务——阻塞事件循环，其他任务饿死

如果一个任务长时间不让出控制权，其他任务就会"饿死"。

### 阻塞的后果

```python
import asyncio
import time

async def blocking_task():
    """模拟一个 CPU 密集型任务，不让出控制权"""
    print("[阻塞任务] 开始")
    start = time.perf_counter()

    # CPU 密集型计算，没有 await
    count = 0
    for i in range(50_000_000):
        count += i

    elapsed = time.perf_counter() - start
    print(f"[阻塞任务] 完成，耗时 {elapsed:.2f} 秒")

async def background_task():
    """一个需要定期执行的后台任务"""
    while True:
        print(f"[后台] 心跳 {time.strftime('%H:%M:%S')}")
        await asyncio.sleep(1)

async def main():
    # 创建后台任务
    bg = asyncio.create_task(background_task())

    # 执行阻塞任务
    await blocking_task()

    # 取消后台任务
    bg.cancel()

asyncio.run(main())
```

输出：

```
[阻塞任务] 开始
[阻塞任务] 完成，耗时 3.50 秒
```

**后台任务的心跳完全没有打印！** 因为 `blocking_task()` 独占了 CPU 3.5 秒，期间事件循环无法调度 `background_task()`。

### 影响范围

当一个任务阻塞事件循环时：

```
正常情况：                        阻塞情况：
                                  
Task A: ██░░██░░██                Task A: ████████████ (阻塞)
Task B: ░░██░░██░░                Task B: ░░░░░░░░░░░░ (饿死)
Task C: ██░░██░░██                Task C: ░░░░░░░░░░░░ (饿死)
                                  
所有任务正常轮转                    Task A 独占，B 和 C 完全停顿
```

**所有**依赖事件循环的操作都会受影响：

- `asyncio.sleep()` 不会按时触发
- 网络回调不会被处理
- 定时器不会被触发
- `asyncio.Queue` 的 `get()` 不会返回

### 如何解决

**方案 1：在计算密集任务中定期让出**

```python
async def cooperative_heavy_task(items):
    results = []
    for i, item in enumerate(items):
        result = compute(item)
        results.append(result)

        # 每 1000 次迭代让出一次
        if i % 1000 == 0:
            await asyncio.sleep(0)

    return results
```

**方案 2：使用 run_in_executor 把 CPU 密集任务放到线程池**

```python
import asyncio

def cpu_heavy(n):
    """同步的 CPU 密集型函数"""
    return sum(i * i for i in range(n))

async def main():
    loop = asyncio.get_running_loop()

    # 在线程池中执行 CPU 密集型任务
    result = await loop.run_in_executor(None, cpu_heavy, 10_000_000)

    print(f"结果: {result}")

asyncio.run(main())
```

**方案 3：使用 ProcessPoolExecutor 真正并行**

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor

def cpu_heavy(n):
    return sum(i * i for i in range(n))

async def main():
    loop = asyncio.get_running_loop()

    with ProcessPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, cpu_heavy, 10_000_000)

    print(f"结果: {result}")

asyncio.run(main())
```

### 各方案对比

| 方案 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| 定期 `sleep(0)` | 可以分段的计算 | 简单，不离开事件循环 | 需要手动插入让出点 |
| `run_in_executor(None, ...)` | CPU 密集但不想改代码 | 代码不变 | 受 GIL 限制，不适合真正的 CPU 并行 |
| `run_in_executor(pool, ...)` | 真正的 CPU 并行 | 绕过 GIL | 进程间通信有开销 |

---

## 5.8 如何检测阻塞——enable_debug、慢回调检测

当程序变慢时，如何判断是不是有任务阻塞了事件循环？

### 方法 1：启用调试模式

```python
import asyncio

async def slow_task():
    """一个意外阻塞的任务"""
    # 这里有一个同步的数据库查询
    import time
    time.sleep(0.5)  # 阻塞 0.5 秒！
    return "done"

async def main():
    await slow_task()

# 启用调试模式
asyncio.run(main(), debug=True)
```

调试模式下，asyncio 会输出更多警告信息，帮助你发现问题。

### 方法 2：设置慢回调检测

<div data-component="BlockingDetectionDemo"></div>

```python
import asyncio
import logging
import time

logging.basicConfig(level=logging.DEBUG)

async def slow_callback():
    """模拟一个慢回调"""
    # 同步阻塞 0.2 秒
    time.sleep(0.2)

async def main():
    await slow_callback()

# 方法一：通过环境变量
# PYTHONASYNCIODEBUG=1 python script.py

# 方法二：通过代码设置
loop = asyncio.new_event_loop()
loop.set_debug(True)
loop.slow_callback_duration = 0.1  # 超过 0.1 秒的回调会触发警告

try:
    loop.run_until_complete(main())
finally:
    loop.close()
```

### 方法 3：手动计时检测

```python
import asyncio
import time

async def monitor_loop(threshold=0.1):
    """监控事件循环是否有阻塞"""
    while True:
        start = time.perf_counter()
        await asyncio.sleep(0)  # 让出控制权
        elapsed = time.perf_counter() - start

        if elapsed > threshold:
            print(f"⚠️ 事件循环延迟 {elapsed:.3f} 秒（阈值 {threshold} 秒）")
            print(f"   可能有任务阻塞了事件循环！")

async def blocking_work():
    """一个阻塞的任务"""
    time.sleep(0.5)  # 故意阻塞

async def main():
    monitor = asyncio.create_task(monitor_loop())

    # 执行一些工作
    await asyncio.sleep(1)

    # 执行阻塞的工作
    await blocking_work()

    monitor.cancel()

asyncio.run(main())
```

### 方法 4：使用第三方库 aiomonitor

```python
# pip install aiomonitor
import asyncio
import aiomonitor

async def main():
    with aiomonitor.start_monitor(loop=asyncio.get_running_loop()):
        # 你的异步代码
        await asyncio.sleep(100)

asyncio.run(main())
# 然后可以通过 telnet localhost 50101 连接监控
```

### 方法 5：使用 uvloop 的内置检测

```python
# pip install uvloop
import asyncio
import uvloop

async def main():
    # uvloop 内置了更好的性能监控
    await some_work()

uvloop.install()
asyncio.run(main())
```

### 阻塞检测清单

| 检测方法 | 侵入性 | 精确度 | 适用场景 |
|---------|--------|--------|---------|
| `debug=True` | 低 | 中 | 开发阶段通用 |
| `slow_callback_duration` | 低 | 高 | 检测慢回调 |
| 手动计时监控 | 中 | 高 | 生产环境 |
| aiomonitor | 中 | 高 | 长时间运行的服务 |
| uvloop | 低 | 高 | 高性能场景 |

### 常见的隐式阻塞

有些操作看起来是异步的，实际上会阻塞：

```python
# ❌ 这些会阻塞事件循环
time.sleep(1)                    # 同步 sleep
requests.get(url)                # 同步 HTTP 请求
open("file.txt").read()          # 同步文件读取
subprocess.run(["cmd"])          # 同步子进程
json.loads(huge_string)          # 大 JSON 解析
pickle.loads(huge_data)          # 大数据反序列化

# ✅ 这些不会阻塞
await asyncio.sleep(1)           # 异步 sleep
await aiohttp.ClientSession().get(url)  # 异步 HTTP
await aiofiles.open("file.txt")  # 异步文件
await asyncio.create_subprocess_exec("cmd")  # 异步子进程
```
</div>

---

## 5.9 get_running_loop()——访问当前事件循环

在异步代码中，有时你需要直接操作事件循环本身。`asyncio.get_running_loop()` 是获取当前运行中事件循环的标准方式。

### 基本用法

```python
import asyncio

async def main():
    # 获取当前运行的事件循环
    loop = asyncio.get_running_loop()
    print(f"事件循环类型: {type(loop)}")
    print(f"是否运行中: {loop.is_running()}")

asyncio.run(main())
```

### get_running_loop() vs get_event_loop()

| 函数 | 使用场景 | 行为 |
|------|---------|------|
| `get_running_loop()` | 在异步代码中 | 返回当前运行的事件循环，没有则抛出异常 |
| `get_event_loop()` | 在同步代码中 | 返回当前线程的事件循环，没有则创建一个（已弃用） |

```python
import asyncio

async def main():
    # ✅ 正确：在异步代码中使用
    loop = asyncio.get_running_loop()
    print(f"循环: {loop}")

# ❌ 不推荐：在同步代码中使用 get_event_loop()
# loop = asyncio.get_event_loop()  # Python 3.10+ 弃用
```

### 实际用途

**用途 1：run_in_executor 执行同步代码**

```python
import asyncio

def blocking_io():
    """同步的阻塞 I/O"""
    import time
    time.sleep(2)
    return "I/O 完成"

async def main():
    loop = asyncio.get_running_loop()

    # 在线程池中执行阻塞函数
    result = await loop.run_in_executor(None, blocking_io)
    print(result)

asyncio.run(main())
```

**用途 2：添加回调**

```python
import asyncio

async def main():
    loop = asyncio.get_running_loop()

    # 在事件循环中安排一个回调
    loop.call_soon(lambda: print("立即执行的回调"))

    # 延迟 1 秒执行
    loop.call_later(1, lambda: print("1 秒后执行的回调"))

    await asyncio.sleep(2)

asyncio.run(main())
```

**用途 3：注册信号处理**

```python
import asyncio
import signal

async def main():
    loop = asyncio.get_running_loop()

    # 注册信号处理
    loop.add_signal_handler(signal.SIGINT, lambda: print("收到 Ctrl+C"))

    print("按 Ctrl+C 测试...")
    await asyncio.sleep(60)

asyncio.run(main())
```

**用途 4：创建 DNS 解析器**

```python
import asyncio

async def main():
    loop = asyncio.get_running_loop()

    # 异步 DNS 解析
    result = await loop.getaddrinfo("example.com", 80)
    print(f"解析结果: {result[0]}")

asyncio.run(main())
```

### 重要提醒

```python
# ❌ 错误：在同步代码中调用
def sync_function():
    loop = asyncio.get_running_loop()  # RuntimeError!

# ✅ 正确：传递 loop 作为参数
async def main():
    loop = asyncio.get_running_loop()
    sync_function(loop)

def sync_function(loop):
    # 使用传入的 loop
    pass
```

---

## 5.10 事件循环的生命周期——创建、运行、关闭

理解事件循环的完整生命周期，有助于正确使用和管理资源。

### 生命周期概览

<div data-component="EventLoopLifecycle"></div>

```
┌─────────────────────────────────────────────────────────────┐
│                   事件循环的生命周期                          │
│                                                             │
│   1. 创建 (Create)                                          │
│   │   loop = asyncio.new_event_loop()                       │
│   │   或 asyncio.run() 自动创建                              │
│   │                                                         │
│   ▼                                                         │
│   2. 配置 (Configure)                                       │
│   │   loop.set_debug(True)                                  │
│   │   loop.slow_callback_duration = 0.1                     │
│   │                                                         │
│   ▼                                                         │
│   3. 运行 (Run)                                             │
│   │   loop.run_until_complete(coro)                         │
│   │   或 loop.run_forever()                                 │
│   │   事件循环在这里执行所有任务                               │
│   │                                                         │
│   ▼                                                         │
│   4. 关闭 (Close)                                           │
│   │   loop.close()                                          │
│   │   释放所有资源                                           │
│   │                                                         │
│   ▼                                                         │
│   5. 结束 (End)                                             │
│       事件循环不再可用                                        │
└─────────────────────────────────────────────────────────────┘
```

### 使用 asyncio.run()（推荐）

```python
import asyncio

async def main():
    print("事件循环正在运行")
    await asyncio.sleep(1)
    print("完成")

# asyncio.run() 自动处理创建和关闭
asyncio.run(main())
```

`asyncio.run()` 做了什么：

```python
# 伪代码：asyncio.run() 的内部实现
def run(coro):
    # 1. 创建事件循环
    loop = new_event_loop()

    try:
        # 2. 运行协程直到完成
        result = loop.run_until_complete(coro)
        return result
    finally:
        # 3. 关闭事件循环
        loop.close()
```

### 手动管理事件循环

```python
import asyncio

async def main():
    print("任务开始")
    await asyncio.sleep(1)
    print("任务完成")
    return 42

# 手动创建、运行、关闭
loop = asyncio.new_event_loop()

try:
    # 运行直到协程完成
    result = loop.run_until_complete(main())
    print(f"结果: {result}")
finally:
    # 确保关闭
    loop.close()
```

### run_until_complete() vs run_forever()

```python
import asyncio

async def task():
    await asyncio.sleep(1)
    return "done"

# run_until_complete：等待协程完成后返回
loop = asyncio.new_event_loop()
result = loop.run_until_complete(task())
print(f"结果: {result}")
loop.close()
# 输出：结果: done

# run_forever：永远运行，直到手动停止
loop = asyncio.new_event_loop()

async def stop_later():
    await asyncio.sleep(1)
    loop.stop()  # 停止事件循环

# 安排 1 秒后停止
loop.create_task(stop_later())
loop.run_forever()
print("事件循环已停止")
loop.close()
```

### 事件循环的状态

```python
import asyncio

loop = asyncio.new_event_loop()

print(f"是否关闭: {loop.is_closed()}")  # False

# 运行任务
async def hello():
    print("Hello")
loop.run_until_complete(hello())

print(f"是否运行中: {loop.is_running()}")  # False（已结束运行）

loop.close()
print(f"是否关闭: {loop.is_closed()}")  # True
```

### 资源清理

```python
import asyncio

async def main():
    # 创建一些需要清理的资源
    reader, writer = await asyncio.open_connection("example.com", 80)

    try:
        writer.write(b"GET / HTTP/1.0\r\nHost: example.com\r\n\r\n")
        await writer.drain()

        data = await reader.read(1000)
        print(f"收到 {len(data)} 字节")
    finally:
        # 确保关闭连接
        writer.close()
        await writer.wait_closed()

asyncio.run(main())
# asyncio.run() 会确保事件循环关闭
```

### 多个事件循环

```python
import asyncio

# 每个线程可以有自己的事件循环
def thread_func():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def work():
        print("在线程中运行")
        await asyncio.sleep(1)

    loop.run_until_complete(work())
    loop.close()

import threading
t = threading.Thread(target=thread_func)
t.start()
t.join()
```
</div>

---

## 5.11 底层 API vs 高层 API

asyncio 提供了两层 API：底层的事件循环 API 和高层的协程/Task API。

### 高层 API（推荐使用）

```python
import asyncio

async def main():
    # 高层 API：简洁、Pythonic

    # 创建和等待任务
    task = asyncio.create_task(some_coroutine())
    result = await task

    # 并发执行多个任务
    results = await asyncio.gather(
        coro1(),
        coro2(),
        coro3(),
    )

    # 超时控制
    try:
        result = await asyncio.wait_for(some_coroutine(), timeout=5.0)
    except asyncio.TimeoutError:
        print("超时了")

    # 异步迭代
    async for item in async_generator():
        process(item)

    # 异步上下文管理器
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.json()

asyncio.run(main())
```

### 底层 API（事件循环直接操作）

```python
import asyncio

async def main():
    loop = asyncio.get_running_loop()

    # 底层 API：更灵活，但更复杂

    # 创建 Future
    future = loop.create_future()
    future.set_result(42)

    # 添加回调
    loop.call_soon(lambda: print("立即执行"))
    loop.call_later(1, lambda: print("1 秒后执行"))
    loop.call_at(loop.time() + 2, lambda: print("2 秒后执行"))

    # 注册文件描述符的 I/O 监听
    # loop.add_reader(fd, callback)
    # loop.add_writer(fd, callback)

    # 在线程池中执行
    result = await loop.run_in_executor(None, blocking_func)

asyncio.run(main())
```

### 对比

| 方面 | 高层 API | 底层 API |
|------|---------|---------|
| **易用性** | 简洁，Pythonic | 复杂，需要理解事件循环 |
| **功能** | 覆盖 90% 的场景 | 提供更细粒度的控制 |
| **适用场景** | 日常异步编程 | 框架开发、特殊需求 |
| **可移植性** | 标准 API，跨实现 | 可能依赖特定实现 |
| **推荐度** | ✅ 首选 | 仅在必要时使用 |

### 何时使用底层 API

| 场景 | 使用的底层 API |
|------|---------------|
| 需要直接操作事件循环 | `get_running_loop()` |
| 集成同步库 | `run_in_executor()` |
| 注册信号处理 | `add_signal_handler()` |
| 实现自定义协议 | `create_server()` |
| 精确控制回调时机 | `call_soon()`, `call_later()` |
| 实现自定义等待机制 | `create_future()` |

### 例子：用底层 API 实现高层功能

```python
import asyncio

async def my_wait_for(coro, timeout):
    """用底层 API 实现 wait_for"""
    loop = asyncio.get_running_loop()

    # 创建一个 Future 来存储结果
    result_future = loop.create_future()

    # 创建任务
    task = loop.create_task(coro)

    # 超时回调
    def on_timeout():
        if not task.done():
            task.cancel()

    # 设置超时
    timer = loop.call_later(timeout, on_timeout)

    try:
        return await task
    except asyncio.CancelledError:
        raise asyncio.TimeoutError()
    finally:
        timer.cancel()

async def main():
    try:
        result = await my_wait_for(asyncio.sleep(10), timeout=2.0)
    except asyncio.TimeoutError:
        print("超时！")

asyncio.run(main())
```

### 推荐原则

```
日常开发：
  使用 asyncio.run()、create_task()、gather()、wait_for()
  ↓
  这些是高层 API，简洁且足够

需要更细粒度控制时：
  使用 get_running_loop() 获取事件循环
  ↓
  调用底层方法

开发框架时：
  直接操作事件循环的底层 API
```

---

## 5.12 常见误区

学习事件循环时，很多人会陷入一些误区。这里逐一澄清。

### 误区 1：create_task 会立即执行任务

```python
# ❌ 错误理解
task = asyncio.create_task(worker())  # worker 立即开始执行？

# ✅ 正确理解
task = asyncio.create_task(worker())  # worker 被加入就绪队列，等待调度
# 此时 main() 继续执行，worker 还没开始
await task  # 这里才会调度 worker 执行
```

### 误区 2：asyncio 是多线程的

```python
# ❌ 错误理解
# "asyncio 可以利用多核 CPU"

# ✅ 正确理解
# asyncio 是单线程的，同一时刻只执行一个任务
# 要利用多核，需要结合 ProcessPoolExecutor
```

### 误区 3：await 会让当前线程暂停

```python
# ❌ 错误理解
# "await asyncio.sleep(1) 会暂停当前线程 1 秒"

# ✅ 正确理解
# "await asyncio.sleep(1) 会让当前协程暂停 1 秒"
# "线程继续运行，去执行其他协程"
```

### 误区 4：事件循环是一个独立的后台线程

```python
# ❌ 错误理解
# "事件循环在一个单独的线程中运行"

# ✅ 正确理解
# 事件循环就是当前线程中的一段 while 循环
# 调用 asyncio.run() 的线程就是事件循环所在的线程
```

### 误区 5：async def 中的代码都是异步的

```python
# ❌ 错误理解
async def process():
    data = read_huge_file()  # 这是同步阻塞！
    result = json.loads(data)  # 这也是同步阻塞！
    return result

# ✅ 正确理解
async def process():
    # 只有 await 才是真正异步的
    data = await aiofiles.read("huge_file.txt")  # 异步读取
    result = await asyncio.to_thread(json.loads, data)  # 放到线程池
    return result
```

### 误区 6：gather 和 create_task 效果完全一样

```python
# ❌ 不完全一样
# gather() 接受协程，内部自动创建任务
results = await asyncio.gather(coro1(), coro2())

# create_task() 手动创建任务，可以更精细地控制
t1 = asyncio.create_task(coro1())
t2 = asyncio.create_task(coro2())
# 可以在 await 之前做其他事情
results = await asyncio.gather(t1, t2)
```

### 误区 7：事件循环可以"后台运行"

```python
# ❌ 错误理解
# "我可以启动事件循环，然后继续执行其他同步代码"

# ✅ 正确理解
# asyncio.run() 会阻塞直到所有任务完成
# 要同时做其他事，需要用线程

import threading

def run_async():
    asyncio.run(main())

# 在另一个线程中运行事件循环
thread = threading.Thread(target=run_async)
thread.start()

# 主线程继续做其他事
do_other_work()

thread.join()
```

### 误区 8：sleep(0) 等于"立即继续"

```python
# ❌ 错误理解
await asyncio.sleep(0)  # 立即继续执行下一步？

# ✅ 正确理解
await asyncio.sleep(0)  # 让出控制权，其他就绪的任务先执行一轮
# 然后我再重新就绪
```

### 误区 9：关闭事件循环后可以重新使用

```python
# ❌ 错误理解
loop.close()
loop.run_until_complete(some_coro())  # RuntimeError!

# ✅ 正确理解
# 关闭后的事件循环不可重用
# 需要创建新的事件循环
loop = asyncio.new_event_loop()
loop.run_until_complete(some_coro())
loop.close()
```

### 误区 10：异步代码一定比同步代码快

```python
# ❌ 错误理解
# "用 asyncio 就能让代码变快"

# ✅ 正确理解
# asyncio 主要提升 I/O 密集型任务的吞吐量
# 对于 CPU 密集型任务，可能更慢（因为事件循环开销）
# 对于少量 I/O 任务，可能没有明显提升

# 适合：Web 服务器、爬虫、API 网关（大量并发 I/O）
# 不适合：科学计算、图像处理（CPU 密集）
```

### 避坑清单

| 误区 | 正确理解 |
|------|---------|
| `create_task` 立即执行 | 只是加入就绪队列 |
| asyncio 是多线程 | 单线程，协作式调度 |
| `await` 暂停线程 | 只暂停当前协程 |
| 事件循环是后台线程 | 就是当前线程的主循环 |
| `async def` 中都是异步 | 只有 `await` 才异步 |
| `gather` = `create_task` | `gather` 更简洁，`create_task` 更灵活 |
| `sleep(0)` 立即继续 | 让出后重新排队 |
| 关闭后可重用 | 需要新建 |
| 异步一定更快 | 取决于任务类型 |

---

## 本章小结

本章深入剖析了 asyncio 事件循环的调度机制。事件循环是 asyncio 的心脏，所有协程的调度都由它驱动。

**核心要点**：

| 概念 | 说明 |
|------|------|
| **事件循环** | 单线程的调度中心，管理所有协程的执行 |
| **就绪队列** | FIFO 队列，存放可以立即执行的任务 |
| **一轮调度** | 从就绪队列取任务 → 执行到 await → 下一个任务 |
| **create_task** | 不会立即执行，只是加入就绪队列 |
| **sleep(0)** | 让出控制权，立即重新就绪 |
| **单线程** | 同一时刻只执行一个任务，不需要锁 |
| **阻塞检测** | 使用 debug 模式、慢回调检测发现阻塞 |
| **get_running_loop** | 在异步代码中获取当前事件循环 |
| **生命周期** | 创建 → 配置 → 运行 → 关闭 |
| **高层 vs 底层** | 日常用高层 API，框架开发用底层 API |

**调度规则**：

```
1. 事件循环从就绪队列头部取出任务
2. 执行任务直到遇到 await
3. 如果 await 的操作未完成，任务暂停
4. 如果 await 的操作已完成（如 sleep(0)），任务进入队列末尾
5. 重复步骤 1-4，直到所有任务完成
```

**关键公式**：

```
I/O 等待重叠：总时间 ≈ max(任务1, 任务2, ..., 任务N)
CPU 串行执行：总时间 ≈ 任务1 + 任务2 + ... + 任务N
```

---

## 思考题

### 1. 为什么 asyncio 选择单线程模型而不是多线程？

单线程模型避免了多线程编程中最棘手的问题：竞态条件、死锁、资源同步。由于同一时刻只有一个协程在运行，不需要加锁保护共享状态。这使得异步代码更容易编写和维护。对于 I/O 密集型任务，单线程事件循环通过协作式调度实现了高并发，性能甚至优于多线程。

代价是不能利用多核 CPU，但对于 I/O 密集型场景（Web 服务器、爬虫），瓶颈通常在网络等待而非 CPU 计算。

### 2. 如果一个协程从不 await，会发生什么？

它会独占事件循环，直到执行完毕。其他所有就绪的任务都会被饿死，无法得到调度。

```python
import asyncio

async def greedy():
    for i in range(3):
        print(f"[greedy] {i}")
        # 没有 await，不让出控制权

async def other():
    print("[other] 执行")

async def main():
    t1 = asyncio.create_task(greedy())
    t2 = asyncio.create_task(other())

    await asyncio.gather(t1, t2)

asyncio.run(main())
# 输出：
# [greedy] 0
# [greedy] 1
# [greedy] 2
# [other] 执行    ← other 在 greedy 完成后才执行
```

解决方案：在长时间循环中定期 `await asyncio.sleep(0)`。

### 3. `create_task` 和直接 `await` 协程有什么区别？

| | `await coro()` | `create_task(coro())` |
|--|---------------|----------------------|
| 执行时机 | 立即开始执行 | 加入就绪队列，等待调度 |
| 并发性 | 阻塞当前协程 | 不阻塞，可以继续创建其他任务 |
| 控制粒度 | 串行执行 | 可以并发执行多个任务 |

```python
# 串行：A 完成后 B 才开始
await task_a()
await task_b()

# 并发：A 和 B 同时进行
t1 = asyncio.create_task(task_a())
t2 = asyncio.create_task(task_b())
await asyncio.gather(t1, t2)
```

### 4. 为什么 `await asyncio.sleep(0)` 能让任务交替执行？

`sleep(0)` 做了三件事：(1) 暂停当前协程；(2) 注册一个 0 秒的定时器；(3) 让事件循环去执行其他就绪任务。由于定时器立即到期，当前协程会在其他就绪任务执行一轮后重新就绪。

这实现了**协作式多任务**：每个任务主动让出控制权，事件循环公平地轮转调度。

### 5. 如何判断一个异步函数是否会阻塞事件循环？

检查函数内部是否有**同步阻塞调用**：

```python
# 会阻塞
async def bad():
    time.sleep(1)           # 同步 sleep
    requests.get(url)       # 同步 HTTP
    open(f).read()          # 同步文件读取
    subprocess.run(cmd)     # 同步子进程

# 不会阻塞
async def good():
    await asyncio.sleep(1)  # 异步 sleep
    await session.get(url)  # 异步 HTTP
    await aiofiles.open(f)  # 异步文件
```

也可以使用 `loop.set_debug(True)` 和 `slow_callback_duration` 自动检测。

### 6. 事件循环的就绪队列是公平的吗？

是的，就绪队列使用 FIFO（先进先出）策略。但"公平"的前提是每个任务都会主动让出控制权。如果一个任务不让出（没有 await），它会独占事件循环，导致其他任务饿死。

```python
# 公平轮转
async def task(name):
    for i in range(3):
        print(f"[{name}] {i}")
        await asyncio.sleep(0)  # 让出

# 不公平：greedy 独占
async def greedy():
    for i in range(1000000):
        pass  # 没有 await
```

### 7. 为什么推荐使用 `asyncio.run()` 而不是手动创建事件循环？

`asyncio.run()` 自动处理了事件循环的创建和关闭，避免了资源泄漏。手动管理需要确保在所有代码路径中都正确关闭事件循环。

```python
# 推荐：简洁安全
asyncio.run(main())

# 手动：容易出错
loop = asyncio.new_event_loop()
try:
    loop.run_until_complete(main())
finally:
    loop.close()  # 容易忘记
```

`asyncio.run()` 还处理了信号、键盘中断等边界情况，是 Python 官方推荐的方式。

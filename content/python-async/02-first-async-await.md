---
title: "2. 第一个 async/await 程序"
description: "从零开始编写第一个异步程序，理解 async def、await 和 asyncio.run() 的工作原理"
updated: 2026-07-07
---

> **学习目标**：
> 1. 能够独立编写一个包含 `async def` 和 `await` 的最小异步程序
> 2. 理解 `async def` 定义的是协程函数，调用后产生协程对象而非立即执行
> 3. 掌握 `asyncio.run()` 的作用：创建事件循环、运行协程、关闭事件循环
> 4. 区分协程函数（coroutine function）和协程对象（coroutine object）的概念
> 5. 理解 `await` 的含义：暂停当前协程，让出控制权给事件循环
> 6. 认识常见的异步编程错误及其修复方法

---

## 2.1 最小异步示例 — async def + asyncio.run()

让我们从一个最简单的异步程序开始。这个程序虽然简单，但包含了异步编程的两个核心要素：`async def` 和 `asyncio.run()`。

### 2.1.1 代码示例

```python
# minimal_async.py — 最小异步程序
import asyncio

async def main():
    print("你好，异步世界！")
    await asyncio.sleep(1)
    print("程序结束")

asyncio.run(main())
```

**运行结果：**
```
你好，异步世界！
（等待 1 秒）
程序结束
```

### 2.1.2 代码解析

让我们逐行分析这个程序：

| 行号 | 代码 | 说明 |
|------|------|------|
| 1 | `import asyncio` | 导入 Python 内置的异步 I/O 库 |
| 3 | `async def main():` | 定义一个**协程函数**（coroutine function） |
| 4 | `print("你好，异步世界！")` | 同步代码，在协程内部也可以使用普通代码 |
| 5 | `await asyncio.sleep(1)` | **挂起**当前协程 1 秒，让出控制权 |
| 6 | `print("程序结束")` | 恢复执行后的代码 |
| 8 | `asyncio.run(main())` | 启动异步程序的入口点 |

### 2.1.3 执行流程图

```
程序启动
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  asyncio.run(main())                                │
│  ┌───────────────────────────────────────────────┐  │
│  │  1. 创建新的事件循环                           │  │
│  │  2. 将 main() 协程加入事件循环                 │  │
│  │  3. 运行事件循环直到 main() 完成               │  │
│  │  4. 关闭事件循环                               │  │
│  └───────────────────────────────────────────────┘  │
│                          │                          │
│                          ▼                          │
│              ┌───────────────────────┐              │
│              │  执行 main() 协程     │              │
│              │                       │              │
│              │  print("你好")        │              │
│              │         │             │              │
│              │         ▼             │              │
│              │  await sleep(1)       │              │
│              │  ┌─────────────────┐  │              │
│              │  │ 挂起 1 秒       │  │              │
│              │  │ 控制权→事件循环  │  │              │
│              │  └─────────────────┘  │              │
│              │         │             │              │
│              │         ▼             │              │
│              │  print("程序结束")    │              │
│              └───────────────────────┘              │
│                          │                          │
│                          ▼                          │
│                  关闭事件循环，程序退出               │
└─────────────────────────────────────────────────────┘
```

### 2.1.4 与同步版本对比

同步版本的等价写法：

```python
# sync_version.py — 同步版本
import time

def main():
    print("你好，异步世界！")
    time.sleep(1)  # 阻塞式等待
    print("程序结束")

main()
```

**关键区别：**

| 特性 | 同步版本 | 异步版本 |
|------|---------|---------|
| 函数定义 | `def main():` | `async def main():` |
| 等待方式 | `time.sleep(1)` | `await asyncio.sleep(1)` |
| 启动方式 | `main()` | `asyncio.run(main())` |
| 等待时行为 | 阻塞整个线程 | 挂起协程，线程可处理其他任务 |
| 适用场景 | 简单脚本 | I/O 密集型应用 |


```python
# 动手试试：修改 sleep 时间，观察执行顺序
import asyncio

async def main():
    print("开始")
    await asyncio.sleep(1)  # 尝试改成 0.5 或 2
    print("结束")

asyncio.run(main())
```

---

## 2.2 async def 是什么 — 协程函数定义

`async def` 是 Python 3.5 引入的语法，用于定义**协程函数**（coroutine function）。它是异步编程的基石。

### 2.2.1 语法定义

```python
# 语法格式
async def 函数名(参数列表):
    函数体
    # 可以包含 await 表达式
```

### 2.2.2 协程函数 vs 普通函数

```python
# 普通函数
def normal_func():
    return "我是普通函数"

# 协程函数
async def coroutine_func():
    return "我是协程函数"
```

**类型检查：**

```python
import inspect

def normal_func():
    pass

async def coroutine_func():
    pass

print(inspect.isfunction(normal_func))        # True
print(inspect.iscoroutinefunction(coroutine_func))  # True
print(inspect.isfunction(coroutine_func))      # False
print(inspect.iscoroutinefunction(normal_func))  # False
```

### 2.2.3 协程函数的特征

```python
async def example():
    """协程函数的特征"""
    # 1. 可以使用 await
    await asyncio.sleep(0)
    
    # 2. 可以包含普通代码
    x = 1 + 2
    print(x)
    
    # 3. 可以有返回值
    return x
    
    # 4. 可以抛出异常
    # raise ValueError("错误")
```

### 2.2.4 类中的协程方法

```python
class AsyncWorker:
    """异步工作器"""
    
    async def start(self):
        """启动工作器"""
        print("启动中...")
        await self.initialize()
        print("启动完成")
    
    async def initialize(self):
        """初始化操作"""
        await asyncio.sleep(1)
        print("初始化完成")
    
    async def process(self, data):
        """处理数据"""
        await asyncio.sleep(0.5)
        return f"处理完成: {data}"

# 使用
async def main():
    worker = AsyncWorker()
    await worker.start()
    result = await worker.process("测试数据")
    print(result)

asyncio.run(main())
```

### 2.2.5 async def 的限制

```python
# ❌ 错误：不能在普通函数中使用 await
def normal_function():
    await asyncio.sleep(1)  # SyntaxError!

# ❌ 错误：async def 中的 lambda 不支持
async_lambda = lambda: await asyncio.sleep(1)  # SyntaxError!

# ✅ 正确：只能在 async def 中使用 await
async def correct_function():
    await asyncio.sleep(1)  # 正确
```

### 2.2.6 协程函数的元信息

```python
async def my_coroutine(x, y):
    """这是一个协程函数"""
    await asyncio.sleep(x)
    return y

# 查看函数信息
print(my_coroutine.__name__)      # 'my_coroutine'
print(my_coroutine.__doc__)       # '这是一个协程函数'
print(my_coroutine.__code__.co_varnames)  # ('x', 'y')

# 查看函数签名
import inspect
sig = inspect.signature(my_coroutine)
print(sig)  # (x, y)
print(sig.parameters)  # {'x': <Parameter "x">, 'y': <Parameter "y">}
```

<div data-component="CoroutineLifecycleDiagram"></div>

---

## 2.3 调用协程函数不会立即执行 — 只创建协程对象

这是初学者最容易犯的错误之一。调用协程函数（使用 `()` 语法）**不会立即执行函数体**，而是创建一个**协程对象**。

### 2.3.1 演示：调用但不执行

```python
import asyncio

async def say_hello():
    print("你好！")
    return "问候完成"

# 调用协程函数
coroutine = say_hello()
print(f"协程对象: {coroutine}")
print(f"类型: {type(coroutine)}")

# 注意：上面没有打印 "你好！"
# 说明函数体没有被执行

# 必须通过 await 或 asyncio.run() 才能执行
result = asyncio.run(coroutine)
print(result)
```

**输出：**
```
协程对象: <coroutine object say_hello at 0x...>
类型: <class 'coroutine'>
你好！
问候完成
```

### 2.3.2 常见错误：忘记 await

```python
import asyncio

async def get_data():
    await asyncio.sleep(1)
    return "数据"

async def main():
    # ❌ 错误：忘记 await
    result = get_data()
    print(result)  # <coroutine object get_data at 0x...>
    # 得到的是协程对象，不是数据！
    
    # ✅ 正确：使用 await
    result = await get_data()
    print(result)  # "数据"

asyncio.run(main())
```

### 2.3.3 与生成器的对比

协程和生成器有相似之处，都是"惰性执行"的：

```python
# 生成器函数
def my_generator():
    print("生成器开始")
    yield 1
    print("生成器继续")
    yield 2

# 协程函数
async def my_coroutine():
    print("协程开始")
    await asyncio.sleep(0)
    print("协程继续")
    return "完成"

# 调用生成器函数 — 创建生成器对象
gen = my_generator()
print(type(gen))  # <class 'generator'>
# 没有打印 "生成器开始"

# 调用协程函数 — 创建协程对象
coro = my_coroutine()
print(type(coro))  # <class 'coroutine'>
# 没有打印 "协程开始"

# 执行生成器
next(gen)  # 打印 "生成器开始"，返回 1
next(gen)  # 打印 "生成器继续"，返回 2

# 执行协程
asyncio.run(coro)  # 打印 "协程开始"、"协程继续"，返回 "完成"
```

### 2.3.4 对比表：函数调用 vs 协程调用

| 操作 | 普通函数 | 协程函数 | 生成器函数 |
|------|---------|---------|-----------|
| 定义 | `def f():` | `async def f():` | `def f(): yield` |
| 调用 | `f()` | `f()` | `f()` |
| 调用结果 | **立即执行**，返回返回值 | 返回**协程对象** | 返回**生成器对象** |
| 触发执行 | 直接调用 | `await f()` 或 `asyncio.run(f())` | `next(g)` 或 `for x in g` |

### 2.3.5 实际案例：为什么会出问题

```python
import asyncio

async def fetch_user(user_id):
    """模拟获取用户信息"""
    await asyncio.sleep(0.5)
    return {"id": user_id, "name": f"用户{user_id}"}

async def main():
    # ❌ 错误写法
    user = fetch_user(1)
    print(user)  # <coroutine object fetch_user at 0x...>
    # print(user["name"])  # TypeError!

    # ✅ 正确写法
    user = await fetch_user(1)
    print(user)  # {"id": 1, "name": "用户1"}
    print(user["name"])  # "用户1"

asyncio.run(main())
```

### 2.3.6 RuntimeWarning：未被 await 的协程

```python
import asyncio

async def forgotten_coroutine():
    await asyncio.sleep(1)
    return "忘记了"

async def main():
    # 创建协程但忘记 await
    coro = forgotten_coroutine()
    # 程序结束时会看到警告：
    # RuntimeWarning: coroutine 'forgotten_coroutine' was never awaited
    
    # 如何避免？
    # 方案 1：正确 await
    result = await coro
    
    # 方案 2：明确取消
    coro = forgotten_coroutine()
    coro.close()  # 显式关闭协程

asyncio.run(main())
```


```python
# 动手试试：观察协程对象的创建
import asyncio

async def demo(x):
    await asyncio.sleep(0.1)
    return x * 2

# 尝试以下代码，观察输出
coro = demo(5)
print(f"coro 的类型: {type(coro)}")
print(f"coro 的 repr: {coro}")

# 使用 await 获取结果
result = asyncio.run(coro)
print(f"最终结果: {result}")
```

---

## 2.4 asyncio.run() 做什么 — 创建事件循环、运行、关闭

`asyncio.run()` 是 Python 3.7 引入的高级 API，用于运行异步程序的入口点。它封装了事件循环的创建、运行和清理工作。

### 2.4.1 asyncio.run() 的内部过程

```python
import asyncio

async def main():
    print("主函数开始")
    await asyncio.sleep(1)
    print("主函数结束")
    return 42

# 使用 asyncio.run() 运行
result = asyncio.run(main())
print(f"结果: {result}")
```

**asyncio.run() 内部做了以下事情：**

```
asyncio.run(main())
    │
    ▼
┌─────────────────────────────────────────────────┐
│ 1. 创建新的事件循环                              │
│    loop = asyncio.new_event_loop()               │
│                                                  │
│ 2. 设置为当前事件循环                             │
│    asyncio.set_event_loop(loop)                  │
│                                                  │
│ 3. 运行直到协程完成                               │
│    result = loop.run_until_complete(main())      │
│                                                  │
│ 4. 关闭事件循环                                   │
│    loop.close()                                  │
│                                                  │
│ 5. 返回结果                                       │
│    return result                                 │
└─────────────────────────────────────────────────┘
```

### 2.4.2 等价的低级写法

```python
import asyncio

async def main():
    print("主函数开始")
    await asyncio.sleep(1)
    print("主函数结束")
    return 42

# 使用 asyncio.run() — 推荐方式
result = asyncio.run(main())

# 等价的低级写法
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
try:
    result = loop.run_until_complete(main())
finally:
    loop.close()
```

### 2.4.3 为什么需要 asyncio.run()

```python
import asyncio

async def main():
    return "完成"

# ❌ 不推荐：手动管理事件循环
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
result = loop.run_until_complete(main())
loop.close()

# ✅ 推荐：使用 asyncio.run()
result = asyncio.run(main())
```

**原因：**
1. **简洁**：一行代码替代多行
2. **安全**：自动处理异常和清理
3. **标准化**：官方推荐的入口点

### 2.4.4 asyncio.run() 的参数

```python
import asyncio

async def main():
    # 获取事件循环
    loop = asyncio.get_running_loop()
    print(f"事件循环: {loop}")
    return "完成"

# 使用 debug 模式
asyncio.run(main(), debug=True)
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `main` | Coroutine | 必需 | 要运行的协程 |
| `debug` | bool | `None` | 是否启用调试模式 |

### 2.4.5 调试模式

```python
import asyncio
import logging

# 启用日志
logging.basicConfig(level=logging.DEBUG)

async def main():
    # 在调试模式下，会检测到常见错误
    loop = asyncio.get_running_loop()
    loop.call_soon(lambda: print("同步回调"))
    
    await asyncio.sleep(0.1)
    return "完成"

# 启用调试模式
asyncio.run(main(), debug=True)
```

### 2.4.6 事件循环的生命周期

```python
import asyncio

async def phase1():
    print("阶段 1 开始")
    await asyncio.sleep(0.5)
    print("阶段 1 结束")
    return "阶段1结果"

async def phase2():
    print("阶段 2 开始")
    await asyncio.sleep(0.3)
    print("阶段 2 结束")
    return "阶段2结果"

async def main():
    """主函数"""
    print("=== 程序开始 ===")
    
    # 运行阶段 1
    result1 = await phase1()
    print(f"阶段 1 结果: {result1}")
    
    # 运行阶段 2
    result2 = await phase2()
    print(f"阶段 2 结果: {result2}")
    
    print("=== 程序结束 ===")
    return [result1, result2]

# asyncio.run() 管理整个生命周期
results = asyncio.run(main())
print(f"所有结果: {results}")
```

**输出：**
```
=== 程序开始 ===
阶段 1 开始
阶段 1 结束
阶段 1 结果: 阶段1结果
阶段 2 开始
阶段 2 结束
阶段 2 结果: 阶段2结果
=== 程序结束 ===
所有结果: ['阶段1结果', '阶段2结果']
```


```python
# 动手试试：观察事件循环的创建和销毁
import asyncio

async def check_loop():
    loop = asyncio.get_running_loop()
    print(f"事件循环类型: {type(loop)}")
    print(f"事件循环是否运行: {loop.is_running()}")
    return "检查完成"

# 运行并观察
result = asyncio.run(check_loop())
print(f"结果: {result}")

# 注意：在 asyncio.run() 外部，没有运行中的事件循环
try:
    asyncio.get_running_loop()
except RuntimeError as e:
    print(f"错误: {e}")
```

---

## 2.5 协程函数 vs 协程对象 — CoroutineVsFunction 组件

理解协程函数和协程对象的区别是掌握异步编程的关键。

### 2.5.1 概念定义

```python
import asyncio

# 协程函数 — 用 async def 定义的函数
async def my_coroutine_function():
    """这是一个协程函数"""
    await asyncio.sleep(1)
    return "结果"

# 协程对象 — 调用协程函数后产生的对象
coroutine_object = my_coroutine_function()

print(f"协程函数: {my_coroutine_function}")
print(f"协程对象: {coroutine_object}")
print(f"协程函数类型: {type(my_coroutine_function)}")
print(f"协程对象类型: {type(coroutine_object)}")
```

**输出：**
```
协程函数: <function my_coroutine_function at 0x...>
协程对象: <coroutine object my_coroutine_function at 0x...>
协程函数类型: <class 'function'>
协程对象类型: <class 'coroutine'>
```

### 2.5.2 详细对比

| 特性 | 协程函数 | 协程对象 |
|------|---------|---------|
| **定义方式** | `async def` 定义 | 调用协程函数产生 |
| **类型** | `<class 'function'>` | `<class 'coroutine'>` |
| **是否可调用** | 是（`func()`） | 否（不能 `obj()`） |
| **是否可迭代** | 否 | 否（但可用 `await`） |
| **是否可等待** | 否 | 是（`await obj`） |
| **可以多次使用** | 是（每次调用产生新对象） | 否（只能 await 一次） |
| **生命周期** | 永久存在 | 一次性，完成后销毁 |

### 2.5.3 多次调用产生多个协程对象

```python
import asyncio

async def greet(name):
    """问候函数"""
    await asyncio.sleep(0.1)
    return f"你好，{name}！"

# 同一个协程函数可以产生多个协程对象
coro1 = greet("Alice")
coro2 = greet("Bob")
coro3 = greet("Charlie")

print(f"coro1: {coro1}")  # 不同的协程对象
print(f"coro2: {coro2}")
print(f"coro3: {coro3}")

# 每个协程对象可以独立执行
async def main():
    result1 = await coro1
    result2 = await coro2
    result3 = await coro3
    print(f"{result1}, {result2}, {result3}")

asyncio.run(main())
```

### 2.5.4 协程对象的属性和方法

```python
import asyncio

async def sample_coroutine():
    await asyncio.sleep(1)
    return "完成"

coro = sample_coroutine()

# 协程对象的属性
print(f"__name__: {coro.__name__}")          # 'sample_coroutine'
print(f"__qualname__: {coro.__qualname__}")  # 'sample_coroutine'
print(f"cr_await: {coro.cr_await}")          # None (未开始执行)
print(f"cr_frame: {coro.cr_frame}")          # 帧对象
print(f"cr_running: {coro.cr_running}")      # False

# 关闭协程对象（不执行）
coro.close()
```

### 2.5.5 实际应用：协程工厂模式

```python
import asyncio

class TaskFactory:
    """任务工厂 — 创建不同的协程对象"""
    
    @staticmethod
    async def create_download_task(url):
        """创建下载任务"""
        await asyncio.sleep(1)  # 模拟下载
        return f"下载完成: {url}"
    
    @staticmethod
    async def create_process_task(data):
        """创建处理任务"""
        await asyncio.sleep(0.5)  # 模拟处理
        return f"处理完成: {data}"

async def main():
    # 创建多个协程对象
    download_coro = TaskFactory.create_download_task("http://example.com")
    process_coro = TaskFactory.create_process_task("数据")
    
    # 执行任务
    download_result = await download_coro
    process_result = await process_coro
    
    print(download_result)
    print(process_result)

asyncio.run(main())
```

<div data-component="CoroutineVsFunction"></div>

---

## 2.6 给协程传参数 — 与普通函数相同

协程函数的参数传递方式与普通函数完全相同，支持位置参数、关键字参数、默认值等。

### 2.6.1 基本参数传递

```python
import asyncio

# 位置参数
async def add(x, y):
    await asyncio.sleep(0.1)
    return x + y

# 关键字参数
async def greet(name, greeting="你好"):
    await asyncio.sleep(0.1)
    return f"{greeting}，{name}！"

# 默认参数
async def power(base, exponent=2):
    await asyncio.sleep(0.1)
    return base ** exponent

async def main():
    # 位置参数
    result1 = await add(3, 5)
    print(f"3 + 5 = {result1}")
    
    # 关键字参数
    result2 = await greet("Alice", greeting="早上好")
    print(result2)
    
    # 使用默认参数
    result3 = await power(3)
    print(f"3² = {result3}")
    
    # 混合使用
    result4 = await power(2, 3)
    print(f"2³ = {result4}")

asyncio.run(main())
```

### 2.6.2 可变参数

```python
import asyncio

# *args
async def sum_all(*args):
    await asyncio.sleep(0.1)
    return sum(args)

# **kwargs
async def print_info(**kwargs):
    await asyncio.sleep(0.1)
    for key, value in kwargs.items():
        print(f"{key}: {value}")

# 混合使用
async def flexible(a, b, *args, **kwargs):
    await asyncio.sleep(0.1)
    print(f"a={a}, b={b}")
    print(f"args: {args}")
    print(f"kwargs: {kwargs}")

async def main():
    # 使用 *args
    result1 = await sum_all(1, 2, 3, 4, 5)
    print(f"总和: {result1}")
    
    # 使用 **kwargs
    await print_info(name="Alice", age=25, city="北京")
    
    # 混合使用
    await flexible(1, 2, 3, 4, x=5, y=6)

asyncio.run(main())
```

### 2.6.3 类型注解

```python
import asyncio
from typing import List, Dict, Optional

async def process_data(
    data: List[int],
    multiplier: float = 1.0,
    verbose: bool = False
) -> Dict[str, float]:
    """处理数据并返回统计信息"""
    await asyncio.sleep(0.1)
    
    result = [x * multiplier for x in data]
    
    if verbose:
        print(f"处理结果: {result}")
    
    return {
        "sum": sum(result),
        "avg": sum(result) / len(result),
        "min": min(result),
        "max": max(result)
    }

async def main():
    stats = await process_data([1, 2, 3, 4, 5], multiplier=2.0, verbose=True)
    print(f"统计: {stats}")

asyncio.run(main())
```

### 2.6.4 复杂对象作为参数

```python
import asyncio
from dataclasses import dataclass
from typing import List

@dataclass
class Task:
    """任务数据类"""
    id: int
    name: str
    priority: int = 0

async def execute_task(task: Task) -> str:
    """执行任务"""
    await asyncio.sleep(0.1 * task.priority)
    return f"任务 {task.name} (ID: {task.id}) 执行完成"

async def execute_batch(tasks: List[Task]) -> List[str]:
    """批量执行任务"""
    results = []
    for task in tasks:
        result = await execute_task(task)
        results.append(result)
    return results

async def main():
    # 创建任务列表
    tasks = [
        Task(id=1, name="下载", priority=2),
        Task(id=2, name="处理", priority=1),
        Task(id=3, name="上传", priority=3),
    ]
    
    # 批量执行
    results = await execute_batch(tasks)
    for result in results:
        print(result)

asyncio.run(main())
```

### 2.6.5 参数验证

```python
import asyncio

async def divide(x: float, y: float) -> float:
    """除法函数，带参数验证"""
    # 在协程内部也可以进行参数验证
    if y == 0:
        raise ValueError("除数不能为零")
    
    await asyncio.sleep(0.1)
    return x / y

async def main():
    try:
        result = await divide(10, 2)
        print(f"10 / 2 = {result}")
        
        result = await divide(10, 0)  # 会抛出异常
    except ValueError as e:
        print(f"错误: {e}")

asyncio.run(main())
```


```python
# 动手试试：创建一个带参数的协程函数
import asyncio

async def repeat_message(message: str, times: int = 1, delay: float = 0.1):
    """重复打印消息"""
    for i in range(times):
        print(f"[{i+1}] {message}")
        await asyncio.sleep(delay)

async def main():
    await repeat_message("Hello", times=3, delay=0.2)
    await repeat_message("World", times=2)

asyncio.run(main())
```

---

## 2.7 协程也可以返回结果 — return value via asyncio.run()

协程函数可以像普通函数一样返回结果，通过 `await` 或 `asyncio.run()` 获取返回值。

### 2.7.1 基本返回值

```python
import asyncio

async def get_greeting(name: str) -> str:
    """返回问候语"""
    await asyncio.sleep(0.1)
    return f"你好，{name}！"

async def main():
    # 使用 await 获取返回值
    greeting = await get_greeting("Alice")
    print(greeting)

# 使用 asyncio.run() 获取返回值
result = asyncio.run(get_greeting("Bob"))
print(result)
```

### 2.7.2 返回复杂数据结构

```python
import asyncio
from typing import Dict, List

async def get_user_profile(user_id: int) -> Dict:
    """获取用户资料"""
    await asyncio.sleep(0.5)
    
    # 模拟数据库查询
    users = {
        1: {"name": "Alice", "age": 25, "hobbies": ["reading", "coding"]},
        2: {"name": "Bob", "age": 30, "hobbies": ["gaming", "music"]},
    }
    
    return users.get(user_id, {"error": "用户不存在"})

async def get_user_ids() -> List[int]:
    """获取所有用户 ID"""
    await asyncio.sleep(0.2)
    return [1, 2, 3]

async def main():
    # 获取复杂数据结构
    profile = await get_user_profile(1)
    print(f"用户资料: {profile}")
    
    # 获取列表
    user_ids = await get_user_ids()
    print(f"用户ID列表: {user_ids}")

asyncio.run(main())
```

### 2.7.3 返回多个值（元组）

```python
import asyncio

async def divide_with_remainder(x: int, y: int):
    """返回商和余数"""
    await asyncio.sleep(0.1)
    quotient = x // y
    remainder = x % y
    return quotient, remainder  # 返回元组

async def main():
    # 解包返回值
    q, r = await divide_with_remainder(17, 5)
    print(f"17 ÷ 5 = {q} 余 {r}")

asyncio.run(main())
```

### 2.7.4 条件返回

```python
import asyncio

async def check_password(password: str) -> dict:
    """检查密码强度"""
    await asyncio.sleep(0.1)
    
    if len(password) < 8:
        return {"valid": False, "reason": "密码太短"}
    elif not any(c.isupper() for c in password):
        return {"valid": False, "reason": "需要大写字母"}
    elif not any(c.isdigit() for c in password):
        return {"valid": False, "reason": "需要数字"}
    else:
        return {"valid": True, "reason": "密码合格"}

async def main():
    # 测试不同密码
    passwords = ["abc", "abcdefgh", "Abcdefg", "Abcdefg1"]
    
    for pwd in passwords:
        result = await check_password(pwd)
        status = "✓" if result["valid"] else "✗"
        print(f"{status} {pwd}: {result['reason']}")

asyncio.run(main())
```

### 2.7.5 返回值与异常

```python
import asyncio

async def risky_operation(may_fail: bool) -> str:
    """可能失败的操作"""
    await asyncio.sleep(0.1)
    
    if may_fail:
        raise ValueError("操作失败")
    
    return "操作成功"

async def main():
    # 成功的情况
    result = await risky_operation(False)
    print(f"结果: {result}")
    
    # 失败的情况
    try:
        result = await risky_operation(True)
    except ValueError as e:
        print(f"错误: {e}")

asyncio.run(main())
```

### 2.7.6 没有 return 语句时返回 None

```python
import asyncio

async def no_return():
    """没有 return 语句的协程"""
    await asyncio.sleep(0.1)
    print("执行完毕")
    # 隐式返回 None

async def explicit_none():
    """显式返回 None"""
    await asyncio.sleep(0.1)
    return None

async def main():
    result1 = await no_return()
    print(f"no_return 结果: {result1}")  # None
    
    result2 = await explicit_none()
    print(f"explicit_none 结果: {result2}")  # None

asyncio.run(main())
```


```python
# 动手试试：创建一个返回结果的协程
import asyncio

async def fibonacci(n: int) -> list:
    """计算斐波那契数列的前 n 项"""
    await asyncio.sleep(0.1)  # 模拟一些异步工作
    
    if n <= 0:
        return []
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    
    return fib[:n]

async def main():
    # 获取返回值
    result = await fibonacci(10)
    print(f"斐波那契前10项: {result}")
    
    result = await fibonacci(20)
    print(f"斐波那契前20项: {result}")

asyncio.run(main())
```

---

## 2.8 第一个 await — await asyncio.sleep(2)，FirstAwaitDemo 组件

`await` 是异步编程的关键字，用于暂停当前协程的执行，等待一个可等待对象（awaitable）完成。

### 2.8.1 await 的基本用法

```python
import asyncio

async def demo():
    print("开始")
    
    # await 暂停当前协程 2 秒
    await asyncio.sleep(2)
    
    print("2 秒后继续")

asyncio.run(demo())
```

### 2.8.2 await asyncio.sleep() 的行为

```python
import asyncio
import time

async def timed_demo():
    """演示 await 的时间行为"""
    start = time.time()
    
    print(f"开始: {time.time() - start:.2f}秒")
    
    await asyncio.sleep(1)
    print(f"第 1 秒: {time.time() - start:.2f}秒")
    
    await asyncio.sleep(1)
    print(f"第 2 秒: {time.time() - start:.2f}秒")
    
    await asyncio.sleep(1)
    print(f"第 3 秒: {time.time() - start:.2f}秒")
    
    print(f"结束: {time.time() - start:.2f}秒")

asyncio.run(timed_demo())
```

**输出：**
```
开始: 0.00秒
第 1 秒: 1.00秒
第 2 秒: 2.00秒
第 3 秒: 3.00秒
结束: 3.00秒
```

### 2.8.3 await 不同的可等待对象

```python
import asyncio

# 1. 等待协程
async def wait_coroutine():
    await asyncio.sleep(0.5)
    return "协程完成"

# 2. 等待 Future
async def wait_future():
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    
    # 0.5 秒后设置结果
    loop.call_later(0.5, future.set_result, "Future 完成")
    
    result = await future
    return result

# 3. 等待 Task
async def wait_task():
    task = asyncio.create_task(slow_operation())
    result = await task
    return result

async def slow_operation():
    await asyncio.sleep(0.5)
    return "任务完成"

async def main():
    # 测试等待协程
    result1 = await wait_coroutine()
    print(result1)
    
    # 测试等待 Future
    result2 = await wait_future()
    print(result2)
    
    # 测试等待 Task
    result3 = await wait_task()
    print(result3)

asyncio.run(main())
```

### 2.8.4 await 的执行顺序

```python
import asyncio

async def step(n):
    print(f"步骤 {n}: 开始")
    await asyncio.sleep(0.5)
    print(f"步骤 {n}: 结束")
    return n

async def main():
    print("主函数开始")
    
    # 顺序执行
    result1 = await step(1)
    result2 = await step(2)
    result3 = await step(3)
    
    print(f"结果: {result1}, {result2}, {result3}")
    print("主函数结束")

asyncio.run(main())
```

**输出：**
```
主函数开始
步骤 1: 开始
步骤 1: 结束
步骤 2: 开始
步骤 2: 结束
步骤 3: 开始
步骤 3: 结束
结果: 1, 2, 3
主函数结束
```

### 2.8.5 嵌套 await

```python
import asyncio

async def inner():
    print("  内层开始")
    await asyncio.sleep(0.5)
    print("  内层结束")
    return "内层结果"

async def middle():
    print(" 中层开始")
    result = await inner()  # 等待内层完成
    print(f" 中层得到: {result}")
    await asyncio.sleep(0.3)
    print(" 中层结束")
    return f"中层结果({result})"

async def outer():
    print("外层开始")
    result = await middle()  # 等待中层完成
    print(f"外层得到: {result}")
    print("外层结束")
    return f"外层结果({result})"

async def main():
    result = await outer()
    print(f"\n最终结果: {result}")

asyncio.run(main())
```

<div data-component="FirstAwaitDemo"></div>

---

## 2.9 await 的含义 — 暂停协程，让出控制权给事件循环

理解 `await` 的本质是掌握异步编程的关键。`await` 不仅仅是"等待"，更是"暂停并让出控制权"。

### 2.9.1 await 的本质

```
当执行到 await 时：

当前协程                    事件循环
    │                          │
    │  执行到 await            │
    ├─────────────────────────>│
    │                          │
    │  记录当前位置             │
    │  暂停执行                │
    │                          │
    │                          ├──> 检查是否有其他就绪任务
    │                          │    执行其他任务
    │                          │
    │  await 的对象完成        │
    │<─────────────────────────┤
    │                          │
    │  从暂停处恢复执行         │
    │  await 返回结果          │
    │                          │
```

### 2.9.2 代码演示

```python
import asyncio

async def task_a():
    print("A: 开始")
    print("A: 即将 await，让出控制权")
    await asyncio.sleep(1)
    print("A: 恢复执行")
    print("A: 结束")

async def task_b():
    print("B: 开始")
    print("B: 即将 await，让出控制权")
    await asyncio.sleep(1)
    print("B: 恢复执行")
    print("B: 结束")

async def main():
    # 如果顺序 await，不会让出控制权给对方
    await task_a()
    await task_b()

asyncio.run(main())
```

**输出（顺序执行）：**
```
A: 开始
A: 即将 await，让出控制权
A: 恢复执行
A: 结束
B: 开始
B: 即将 await，让出控制权
B: 恢复执行
B: 结束
```

### 2.9.3 并发执行时的控制权让出

```python
import asyncio

async def task_a():
    print("A: 开始")
    await asyncio.sleep(1)
    print("A: 恢复执行")
    print("A: 结束")

async def task_b():
    print("B: 开始")
    await asyncio.sleep(1)
    print("B: 恢复执行")
    print("B: 结束")

async def main():
    # 创建任务，让它们并发执行
    task1 = asyncio.create_task(task_a())
    task2 = asyncio.create_task(task_b())
    
    # 等待两个任务完成
    await task1
    await task2

asyncio.run(main())
```

**输出（并发执行）：**
```
A: 开始
B: 开始
A: 恢复执行
A: 结束
B: 恢复执行
B: 结束
```

### 2.9.4 事件循环的工作原理

```python
import asyncio

async def simulated_event_loop_demo():
    """模拟事件循环的工作原理"""
    
    async def worker(name, delay):
        print(f"  [{name}] 开始工作")
        print(f"  [{name}] 需要 {delay} 秒")
        
        # 模拟 await：暂停当前协程
        await asyncio.sleep(delay)
        
        print(f"  [{name}] 工作完成")
        return f"{name}的结果"
    
    print("事件循环：创建任务...")
    
    # 创建多个任务
    tasks = [
        asyncio.create_task(worker("下载", 2)),
        asyncio.create_task(worker("处理", 1)),
        asyncio.create_task(worker("上传", 3)),
    ]
    
    print("事件循环：等待所有任务完成...")
    
    # 等待所有任务
    results = await asyncio.gather(*tasks)
    
    print(f"事件循环：所有任务完成，结果: {results}")

asyncio.run(simulated_event_loop_demo())
```

### 2.9.5 await 与阻塞的区别

```python
import asyncio
import time

async def blocking_wait():
    """❌ 错误：使用阻塞式等待"""
    print("阻塞式等待开始")
    time.sleep(3)  # 阻塞整个线程！
    print("阻塞式等待结束")

async def non_blocking_wait():
    """✅ 正确：使用异步等待"""
    print("异步等待开始")
    await asyncio.sleep(3)  # 只暂停当前协程
    print("异步等待结束")

async def main():
    # 使用阻塞式等待
    start = time.time()
    await blocking_wait()
    print(f"阻塞式等待耗时: {time.time() - start:.2f}秒\n")
    
    # 使用异步等待
    start = time.time()
    await non_blocking_wait()
    print(f"异步等待耗时: {time.time() - start:.2f}秒")

asyncio.run(main())
```


```python
# 动手试试：观察 await 时的控制权让出
import asyncio

async def show_await_behavior():
    """展示 await 的行为"""
    print("1. 协程开始")
    print("2. 即将执行 await asyncio.sleep(0)")
    
    # sleep(0) 会立即让出控制权，然后恢复
    await asyncio.sleep(0)
    
    print("3. await 之后，协程恢复")
    print("4. 协程结束")

async def main():
    print("=== 开始 ===")
    await show_await_behavior()
    print("=== 结束 ===")

asyncio.run(main())
```

---

## 2.10 只有一个任务时异步有优势吗 — 没有

当只有一个任务时，异步编程没有性能优势。异步的优势在于**并发**执行多个任务。

### 2.10.1 单任务对比

```python
import asyncio
import time

# 同步版本
def sync_task():
    time.sleep(2)
    return "完成"

# 异步版本
async def async_task():
    await asyncio.sleep(2)
    return "完成"

# 测试同步版本
start = time.time()
result = sync_task()
sync_time = time.time() - start
print(f"同步耗时: {sync_time:.2f}秒")

# 测试异步版本
start = time.time()
result = asyncio.run(async_task())
async_time = time.time() - start
print(f"异步耗时: {async_time:.2f}秒")

print(f"\n结论：单任务时，异步没有优势，甚至可能略慢")
```

### 2.10.2 多任务时的优势

```python
import asyncio
import time

async def task(n):
    """模拟 I/O 任务"""
    await asyncio.sleep(1)  # 模拟 I/O 等待
    return f"任务 {n} 完成"

async def sequential():
    """顺序执行"""
    results = []
    for i in range(5):
        result = await task(i)
        results.append(result)
    return results

async def concurrent():
    """并发执行"""
    tasks = [asyncio.create_task(task(i)) for i in range(5)]
    results = await asyncio.gather(*tasks)
    return results

async def main():
    # 顺序执行
    start = time.time()
    results = await sequential()
    sequential_time = time.time() - start
    print(f"顺序执行耗时: {sequential_time:.2f}秒")
    print(f"结果: {results}\n")
    
    # 并发执行
    start = time.time()
    results = await concurrent()
    concurrent_time = time.time() - start
    print(f"并发执行耗时: {concurrent_time:.2f}秒")
    print(f"结果: {results}\n")
    
    # 性能对比
    speedup = sequential_time / concurrent_time
    print(f"性能提升: {speedup:.1f}倍")

asyncio.run(main())
```

**输出：**
```
顺序执行耗时: 5.01秒
结果: ['任务 0 完成', '任务 1 完成', '任务 2 完成', '任务 3 完成', '任务 4 完成']

并发执行耗时: 1.00秒
结果: ['任务 0 完成', '任务 1 完成', '任务 2 完成', '任务 3 完成', '任务 4 完成']

性能提升: 5.0倍
```

### 2.10.3 何时使用异步

| 场景 | 是否适合异步 | 原因 |
|------|-------------|------|
| 单个 I/O 任务 | ❌ 不适合 | 没有并发需求 |
| 多个独立 I/O 任务 | ✅ 适合 | 可以并发执行 |
| CPU 密集型任务 | ❌ 不适合 | 应使用多进程 |
| 混合任务 | ✅ 部分适合 | I/O 部分用异步 |

### 2.10.4 异步的开销

```python
import asyncio
import time

async def minimal_task():
    """最小任务"""
    return 42

async def main():
    # 测量异步开销
    iterations = 100000
    
    start = time.time()
    for _ in range(iterations):
        await minimal_task()
    elapsed = time.time() - start
    
    print(f"执行 {iterations} 次异步调用")
    print(f"总耗时: {elapsed:.2f}秒")
    print(f"每次调用: {elapsed/iterations*1000:.4f}毫秒")

asyncio.run(main())
```


```python
# 动手试试：比较多任务顺序 vs 并发
import asyncio
import time

async def download(url, delay):
    """模拟下载"""
    print(f"开始下载: {url}")
    await asyncio.sleep(delay)
    print(f"下载完成: {url}")
    return f"{url} 的内容"

async def main():
    urls = [
        ("http://api1.com", 2),
        ("http://api2.com", 1),
        ("http://api3.com", 3),
    ]
    
    # 顺序下载
    start = time.time()
    results = []
    for url, delay in urls:
        result = await download(url, delay)
        results.append(result)
    sequential_time = time.time() - start
    
    # 并发下载
    start = time.time()
    tasks = [download(url, delay) for url, delay in urls]
    results = await asyncio.gather(*tasks)
    concurrent_time = time.time() - start
    
    print(f"\n顺序下载耗时: {sequential_time:.2f}秒")
    print(f"并发下载耗时: {concurrent_time:.2f}秒")

asyncio.run(main())
```

---

## 2.11 await 只能写在 async def 中 — AwaitPositionRestriction 组件

`await` 关键字只能在 `async def` 定义的协程函数内部使用，这是 Python 的语法限制。

### 2.11.1 语法错误演示

```python
import asyncio

# ❌ 错误：在普通函数中使用 await
def normal_function():
    await asyncio.sleep(1)  # SyntaxError!

# ❌ 错误：在 lambda 中使用 await
my_lambda = lambda: await asyncio.sleep(1)  # SyntaxError!

# ❌ 错误：在同步推导式中使用 await
result = [await asyncio.sleep(1) for _ in range(3)]  # SyntaxError!

# ✅ 正确：在 async def 中使用 await
async def correct_function():
    await asyncio.sleep(1)  # 正确！
```

### 2.11.2 正确和错误的对比

```python
import asyncio

# ✅ 正确：async def 中使用 await
async def correct():
    result = await asyncio.sleep(1)
    return result

# ❌ 错误：普通 def 中使用 await
def wrong():
    result = await asyncio.sleep(1)  # SyntaxError
    return result

# ✅ 正确：在 async def 中调用协程
async def caller():
    result = await correct()  # 正确
    return result

# ❌ 错误：在普通函数中调用协程并 await
def sync_caller():
    result = await correct()  # SyntaxError
    return result
```

### 2.11.3 异步推导式（Python 3.6+）

```python
import asyncio

async def fetch(x):
    await asyncio.sleep(0.1)
    return x * 2

async def main():
    # ❌ 错误：普通推导式不能使用 await
    # result = [await fetch(i) for i in range(5)]  # SyntaxError!
    
    # ✅ 正确：使用 asyncio.gather
    tasks = [fetch(i) for i in range(5)]
    results = await asyncio.gather(*tasks)
    print(results)  # [0, 2, 4, 6, 8]
    
    # ✅ 正确：使用 async for（需要异步迭代器）
    async def async_range(n):
        for i in range(n):
            await asyncio.sleep(0.01)
            yield i
    
    results = []
    async for i in async_range(5):
        results.append(await fetch(i))
    print(results)  # [0, 2, 4, 6, 8]

asyncio.run(main())
```

### 2.11.4 嵌套函数中的 await

```python
import asyncio

async def outer():
    """外层 async 函数"""
    
    # ❌ 错误：内部普通函数不能使用 await
    def inner_normal():
        # await asyncio.sleep(1)  # SyntaxError!
        pass
    
    # ✅ 正确：内部 async 函数可以使用 await
    async def inner_async():
        await asyncio.sleep(0.1)
        return "内部结果"
    
    # 调用内部 async 函数
    result = await inner_async()
    print(f"内部结果: {result}")
    
    return "外层结果"

asyncio.run(outer())
```

### 2.11.5 类方法中的 await

```python
import asyncio

class MyClass:
    # ❌ 错误：普通方法不能使用 await
    def sync_method(self):
        # await asyncio.sleep(1)  # SyntaxError!
        pass
    
    # ✅ 正确：async 方法可以使用 await
    async def async_method(self):
        await asyncio.sleep(0.1)
        return "异步方法结果"
    
    # ✅ 正确：classmethod 中使用 await（需要 Python 3.10+）
    # @classmethod
    # async def class_async(cls):
    #     await asyncio.sleep(0.1)

async def main():
    obj = MyClass()
    result = await obj.async_method()
    print(result)

asyncio.run(main())
```

<div data-component="AwaitPositionRestriction"></div>

---

## 2.12 await 后面只能放可等待对象 — AwaitableTypeChecker 组件

`await` 后面只能跟**可等待对象**（awaitable），包括协程、Task、Future 和实现了 `__await__()` 方法的对象。

### 2.12.1 可等待对象类型

```python
import asyncio

# 1. 协程对象 — 最常见的可等待对象
async def my_coroutine():
    await asyncio.sleep(0.1)
    return "协程结果"

# 2. Task 对象 — 通过 asyncio.create_task() 创建
async def my_task():
    await asyncio.sleep(0.1)
    return "任务结果"

# 3. Future 对象 — 低级 API
async def my_future():
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    
    # 0.5 秒后设置结果
    loop.call_later(0.5, future.set_result, "Future 结果")
    
    return await future

# 4. 自定义可等待对象
class CustomAwaitable:
    def __await__(self):
        # 必须返回一个迭代器
        yield from asyncio.sleep(0.1).__await__()
        return "自定义结果"

async def main():
    # 测试协程对象
    result1 = await my_coroutine()
    print(f"协程: {result1}")
    
    # 测试 Task
    task = asyncio.create_task(my_task())
    result2 = await task
    print(f"任务: {result2}")
    
    # 测试 Future
    result3 = await my_future()
    print(f"Future: {result3}")
    
    # 测试自定义可等待对象
    result4 = await CustomAwaitable()
    print(f"自定义: {result4}")

asyncio.run(main())
```

### 2.12.2 不可等待的对象

```python
import asyncio

async def main():
    # ❌ 错误：整数不可等待
    try:
        await 42  # TypeError
    except TypeError as e:
        print(f"整数: {e}")
    
    # ❌ 错误：字符串不可等待
    try:
        await "hello"  # TypeError
    except TypeError as e:
        print(f"字符串: {e}")
    
    # ❌ 错误：列表不可等待
    try:
        await [1, 2, 3]  # TypeError
    except TypeError as e:
        print(f"列表: {e}")
    
    # ❌ 错误：None 不可等待
    try:
        await None  # TypeError
    except TypeError as e:
        print(f"None: {e}")
    
    # ❌ 错误：普通函数不可等待
    def normal_func():
        return 42
    
    try:
        await normal_func()  # TypeError (返回值 42 不可等待)
    except TypeError as e:
        print(f"普通函数: {e}")

asyncio.run(main())
```

### 2.12.3 检查对象是否可等待

```python
import asyncio
import inspect

async def my_coroutine():
    await asyncio.sleep(0.1)
    return "结果"

async def main():
    # 检查协程对象
    coro = my_coroutine()
    print(f"协程对象可等待: {inspect.isawaitable(coro)}")  # True
    
    # 检查 Task
    task = asyncio.create_task(my_coroutine())
    print(f"Task 可等待: {inspect.isawaitable(task)}")  # True
    
    # 检查 Future
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    print(f"Future 可等待: {inspect.isawaitable(future)}")  # True
    
    # 检查普通对象
    print(f"整数可等待: {inspect.isawaitable(42)}")  # False
    print(f"字符串可等待: {inspect.isawaitable('hello')}")  # False
    
    # 清理
    coro.close()
    task.cancel()

asyncio.run(main())
```

### 2.12.4 自定义可等待对象

```python
import asyncio

class AsyncTimer:
    """自定义异步计时器"""
    
    def __init__(self, seconds):
        self.seconds = seconds
    
    def __await__(self):
        # 使用 asyncio.sleep 的 __await__ 方法
        yield from asyncio.sleep(self.seconds).__await__()
        return f"等待了 {self.seconds} 秒"

class AsyncOperation:
    """自定义异步操作"""
    
    def __init__(self, value):
        self.value = value
    
    def __await__(self):
        # 模拟一些异步操作
        yield  # 让出控制权一次
        return self.value * 2

async def main():
    # 使用自定义异步计时器
    timer = AsyncTimer(1)
    result = await timer
    print(result)  # "等待了 1 秒"
    
    # 使用自定义异步操作
    op = AsyncOperation(21)
    result = await op
    print(result)  # 42

asyncio.run(main())
```

### 2.12.5 常见错误和解决方案

```python
import asyncio

async def fetch_data():
    await asyncio.sleep(0.1)
    return "数据"

async def main():
    # ❌ 错误：忘记 await
    result = fetch_data()
    print(type(result))  # <class 'coroutine'>
    # result 是协程对象，不是 "数据"
    
    # ✅ 正确：使用 await
    result = await fetch_data()
    print(result)  # "数据"
    
    # ❌ 错误：await 一个返回非可等待对象的函数
    def get_value():
        return 42
    
    try:
        await get_value()  # TypeError
    except TypeError as e:
        print(f"错误: {e}")
    
    # ✅ 正确：如果需要 await，函数应该返回协程
    async def get_value_async():
        return 42
    
    result = await get_value_async()
    print(result)  # 42

asyncio.run(main())
```

<div data-component="AwaitableTypeChecker"></div>

---

## 2.13 多个协程的嵌套调用 — coroutine awaiting coroutine

协程可以调用其他协程，形成嵌套的异步调用链。

### 2.13.1 基本嵌套

```python
import asyncio

async def step1():
    print("步骤 1: 开始")
    await asyncio.sleep(0.5)
    print("步骤 1: 完成")
    return "步骤1结果"

async def step2():
    print("步骤 2: 开始")
    await asyncio.sleep(0.3)
    print("步骤 2: 完成")
    return "步骤2结果"

async def step3():
    print("步骤 3: 开始")
    await asyncio.sleep(0.2)
    print("步骤 3: 完成")
    return "步骤3结果"

async def pipeline():
    """流水线：依次执行多个步骤"""
    result1 = await step1()
    result2 = await step2()
    result3 = await step3()
    return f"流水线结果: {result1}, {result2}, {result3}"

async def main():
    result = await pipeline()
    print(result)

asyncio.run(main())
```

### 2.13.2 传递数据的嵌套调用

```python
import asyncio

async def fetch_user(user_id: int) -> dict:
    """获取用户信息"""
    await asyncio.sleep(0.3)
    return {"id": user_id, "name": f"用户{user_id}"}

async def fetch_posts(user_id: int) -> list:
    """获取用户的帖子"""
    await asyncio.sleep(0.2)
    return [f"帖子{i}" for i in range(3)]

async def fetch_user_with_posts(user_id: int) -> dict:
    """获取用户信息及其帖子"""
    # 调用其他协程
    user = await fetch_user(user_id)
    posts = await fetch_posts(user_id)
    
    return {
        "user": user,
        "posts": posts
    }

async def main():
    result = await fetch_user_with_posts(1)
    print(f"用户: {result['user']['name']}")
    print(f"帖子: {result['posts']}")

asyncio.run(main())
```

### 2.13.3 深层嵌套

```python
import asyncio

async def level_3():
    print("    层级 3: 开始")
    await asyncio.sleep(0.1)
    print("    层级 3: 完成")
    return "L3"

async def level_2():
    print("   层级 2: 开始")
    result = await level_3()
    await asyncio.sleep(0.1)
    print(f"   层级 2: 完成，得到 {result}")
    return f"L2({result})"

async def level_1():
    print("  层级 1: 开始")
    result = await level_2()
    await asyncio.sleep(0.1)
    print(f"  层级 1: 完成，得到 {result}")
    return f"L1({result})"

async def main():
    print("主函数: 开始")
    result = await level_1()
    print(f"主函数: 完成，最终结果 {result}")

asyncio.run(main())
```

### 2.13.4 嵌套调用中的错误传播

```python
import asyncio

async def risky_operation():
    """可能出错的操作"""
    await asyncio.sleep(0.1)
    raise ValueError("操作失败")

async def safe_wrapper():
    """安全的包装函数"""
    try:
        result = await risky_operation()
        return {"success": True, "data": result}
    except ValueError as e:
        return {"success": False, "error": str(e)}

async def main():
    # 错误被包装在 safe_wrapper 中
    result = await safe_wrapper()
    print(result)  # {"success": False, "error": "操作失败"}

asyncio.run(main())
```

### 2.13.5 递归协程

```python
import asyncio

async def recursive_countdown(n: int):
    """递归倒计时"""
    print(f"倒计时: {n}")
    await asyncio.sleep(0.2)
    
    if n > 0:
        # 递归调用自身
        await recursive_countdown(n - 1)
    else:
        print("发射！")

async def main():
    await recursive_countdown(5)

asyncio.run(main())
```


```python
# 动手试试：实现一个异步的 Web 请求模拟器
import asyncio

async def http_get(url: str) -> str:
    """模拟 HTTP GET 请求"""
    print(f"请求: {url}")
    await asyncio.sleep(0.5)  # 模拟网络延迟
    return f"<html>{url} 的内容</html>"

async def fetch_all_pages(base_url: str, pages: int) -> list:
    """获取多个页面"""
    results = []
    for i in range(pages):
        url = f"{base_url}/page/{i+1}"
        content = await http_get(url)
        results.append(content)
    return results

async def scrape_website(domain: str):
    """爬取网站"""
    print(f"开始爬取 {domain}")
    pages = await fetch_all_pages(f"http://{domain}", 3)
    print(f"爬取完成，获取 {len(pages)} 个页面")
    return pages

async def main():
    result = await scrape_website("example.com")
    for i, page in enumerate(result):
        print(f"页面 {i+1}: {page[:30]}...")

asyncio.run(main())
```

---

## 2.14 await 不等于创建并发 — sequential await is sequential

连续使用 `await` 会顺序执行，不会自动并发。

### 2.14.1 顺序执行 vs 并发执行

```python
import asyncio
import time

async def task_a():
    print("A: 开始")
    await asyncio.sleep(2)
    print("A: 完成")
    return "A"

async def task_b():
    print("B: 开始")
    await asyncio.sleep(1)
    print("B: 完成")
    return "B"

async def sequential():
    """顺序执行 — 总共需要 3 秒"""
    result_a = await task_a()  # 等待 A 完成（2秒）
    result_b = await task_b()  # 等待 B 完成（1秒）
    return result_a, result_b

async def concurrent():
    """并发执行 — 总共需要 2 秒"""
    # 创建任务但不立即 await
    task_a_obj = asyncio.create_task(task_a())
    task_b_obj = asyncio.create_task(task_b())
    
    # 两个任务同时运行
    result_a = await task_a_obj
    result_b = await task_b_obj
    return result_a, result_b

async def main():
    # 顺序执行
    start = time.time()
    results = await sequential()
    sequential_time = time.time() - start
    print(f"顺序执行: {sequential_time:.2f}秒, 结果: {results}\n")
    
    # 并发执行
    start = time.time()
    results = await concurrent()
    concurrent_time = time.time() - start
    print(f"并发执行: {concurrent_time:.2f}秒, 结果: {results}")

asyncio.run(main())
```

**输出：**
```
A: 开始
A: 完成
B: 开始
B: 完成
顺序执行: 3.00秒, 结果: ('A', 'B')

A: 开始
B: 开始
B: 完成
A: 完成
并发执行: 2.00秒, 结果: ('A', 'B')
```

### 2.14.2 时间线对比

```
顺序执行：
时间: 0s    1s    2s    3s
      |-----|-----|-----|
任务A |██████████████████|  (2秒)
任务B                     |████████|  (1秒)
      总耗时: 3秒

并发执行：
时间: 0s    1s    2s
      |-----|-----|
任务A |██████████████|  (2秒)
任务B |████████|  (1秒)
      总耗时: 2秒
```

### 2.14.3 使用 asyncio.gather() 实现并发

```python
import asyncio
import time

async def download(url: str) -> str:
    """模拟下载"""
    delay = len(url) % 3 + 1  # 1-3秒
    await asyncio.sleep(delay)
    return f"{url} 下载完成 ({delay}秒)"

async def main():
    urls = [
        "http://api.example.com",
        "http://images.example.com",
        "http://data.example.com",
    ]
    
    # 方式 1：顺序下载
    start = time.time()
    results = []
    for url in urls:
        result = await download(url)
        results.append(result)
    sequential_time = time.time() - start
    print(f"顺序下载: {sequential_time:.2f}秒")
    
    # 方式 2：使用 gather 并发下载
    start = time.time()
    results = await asyncio.gather(*[download(url) for url in urls])
    gather_time = time.time() - start
    print(f"并发下载: {gather_time:.2f}秒")
    
    for result in results:
        print(f"  {result}")

asyncio.run(main())
```

### 2.14.4 常见误解

```python
import asyncio

async def main():
    # ❌ 误解 1：连续 await 会并发
    # 实际上是顺序执行
    result1 = await fetch_data("A")
    result2 = await fetch_data("B")
    result3 = await fetch_data("C")
    # 总耗时 = A + B + C
    
    # ✅ 正确：使用 gather 实现并发
    results = await asyncio.gather(
        fetch_data("A"),
        fetch_data("B"),
        fetch_data("C")
    )
    # 总耗时 = max(A, B, C)

async def fetch_data(name):
    await asyncio.sleep(1)
    return f"{name} 的数据"

asyncio.run(main())
```


```python
# 动手试试：比较多任务顺序 vs 并发
import asyncio
import time

async def process_order(order_id: int) -> str:
    """处理订单"""
    print(f"订单 {order_id}: 开始处理")
    await asyncio.sleep(1)  # 模拟处理时间
    print(f"订单 {order_id}: 处理完成")
    return f"订单{order_id}已完成"

async def main():
    orders = [1, 2, 3, 4, 5]
    
    # 顺序处理
    start = time.time()
    results = []
    for order_id in orders:
        result = await process_order(order_id)
        results.append(result)
    print(f"\n顺序处理耗时: {time.time() - start:.2f}秒")
    
    # 并发处理
    start = time.time()
    tasks = [process_order(order_id) for order_id in orders]
    results = await asyncio.gather(*tasks)
    print(f"并发处理耗时: {time.time() - start:.2f}秒")

asyncio.run(main())
```

---

## 2.15 普通函数 vs 协程函数对比表

以下是普通函数和协程函数的全面对比：

### 2.15.1 语法对比

```python
# 普通函数
def normal_func():
    return "普通函数"

# 协程函数
async def coroutine_func():
    return "协程函数"
```

### 2.15.2 完整对比表

| 特性 | 普通函数 | 协程函数 |
|------|---------|---------|
| **定义语法** | `def func():` | `async def func():` |
| **调用结果** | 立即执行，返回返回值 | 返回协程对象 |
| **执行方式** | 直接调用 `func()` | `await func()` 或 `asyncio.run(func())` |
| **返回值类型** | 函数返回值 | 协程对象 |
| **类型检查** | `callable(func)` → True | `callable(func)` → True |
| **类型检查** | `inspect.isfunction(func)` → True | `inspect.isfunction(func)` → False |
| **类型检查** | `inspect.iscoroutinefunction(func)` → False | `inspect.iscoroutinefunction(func)` → True |
| **可以使用 await** | ❌ 不可以 | ✅ 可以 |
| **可以使用 yield** | ✅ 可以（变成生成器） | ✅ 可以（变成异步生成器） |
| **可以在普通函数中调用** | ✅ 可以 | ✅ 可以（但不能 await） |
| **可以在协程函数中调用** | ✅ 可以 | ✅ 可以（可以 await） |
| **异常处理** | `try-except` | `try-except`（相同） |
| **装饰器支持** | ✅ 支持 | ✅ 支持 |

### 2.15.3 代码对比示例

```python
import asyncio
import inspect

# 普通函数
def add(x, y):
    return x + y

# 协程函数
async def async_add(x, y):
    await asyncio.sleep(0.1)
    return x + y

# 对比调用方式
result1 = add(1, 2)  # 直接返回 3
print(f"普通函数结果: {result1}")

# result2 = async_add(1, 2)  # 返回协程对象
# print(f"协程函数结果: {result2}")  # 协程对象，不是 3

async def main():
    result2 = await async_add(1, 2)  # 使用 await 获取结果
    print(f"协程函数结果: {result2}")

asyncio.run(main())

# 对比类型检查
print(f"\n=== 类型检查 ===")
print(f"add 是函数: {inspect.isfunction(add)}")  # True
print(f"async_add 是函数: {inspect.isfunction(async_add)}")  # False
print(f"add 是协程函数: {inspect.iscoroutinefunction(add)}")  # False
print(f"async_add 是协程函数: {inspect.iscoroutinefunction(async_add)}")  # True
```

### 2.15.4 混合使用示例

```python
import asyncio

# 普通函数：数据处理
def process_data(data: list) -> list:
    """处理数据"""
    return [x * 2 for x in data]

# 协程函数：获取数据
async def fetch_data() -> list:
    """获取数据"""
    await asyncio.sleep(1)  # 模拟网络请求
    return [1, 2, 3, 4, 5]

# 协程函数：保存数据
async def save_data(data: list) -> None:
    """保存数据"""
    await asyncio.sleep(0.5)  # 模拟数据库写入
    print(f"数据已保存: {data}")

# 主函数：组合使用
async def main():
    # 1. 获取数据（异步）
    raw_data = await fetch_data()
    
    # 2. 处理数据（同步）
    processed_data = process_data(raw_data)
    
    # 3. 保存数据（异步）
    await save_data(processed_data)

asyncio.run(main())
```

---

## 2.16 常见错误汇总

### 错误 1：忘记 await 协程

```python
import asyncio

async def get_data():
    await asyncio.sleep(1)
    return "数据"

async def main():
    # ❌ 错误：忘记 await
    result = get_data()
    print(result)  # <coroutine object get_data at 0x...>
    # 警告：RuntimeWarning: coroutine 'get_data' was never awaited
    
    # ✅ 正确
    result = await get_data()
    print(result)  # "数据"

asyncio.run(main())
```

**修复方法：** 始终使用 `await` 等待协程对象。

### 错误 2：在普通函数中使用 await

```python
import asyncio

# ❌ 错误
def normal_function():
    await asyncio.sleep(1)  # SyntaxError

# ✅ 正确
async def correct_function():
    await asyncio.sleep(1)

# 或者使用 asyncio.run()
def main():
    asyncio.run(correct_function())
```

**修复方法：** 将函数定义为 `async def`，或使用 `asyncio.run()` 调用。

### 错误 3：忘记调用 asyncio.run()

```python
import asyncio

async def main():
    print("Hello")

# ❌ 错误：直接调用协程函数
# main()  # 只创建协程对象，不执行

# ❌ 错误：await 但没有事件循环
# await main()  # SyntaxError

# ✅ 正确
asyncio.run(main())
```

**修复方法：** 使用 `asyncio.run()` 启动异步程序。

### 错误 4：在已运行的事件循环中调用 asyncio.run()

```python
import asyncio

async def nested():
    # ❌ 错误：在已有事件循环中调用 asyncio.run()
    # asyncio.run(another_coroutine())  # RuntimeError
    
    # ✅ 正确：直接 await
    result = await another_coroutine()
    return result

async def another_coroutine():
    await asyncio.sleep(0.1)
    return "结果"

async def main():
    result = await nested()
    print(result)

asyncio.run(main())
```

**修复方法：** 在协程内部直接使用 `await`，而不是 `asyncio.run()`。

### 错误 5：误解顺序 await 的并发性

```python
import asyncio

async def task(n):
    await asyncio.sleep(1)
    return n

async def main():
    # ❌ 误解：认为这样会并发
    results = []
    for i in range(5):
        result = await task(i)  # 实际上是顺序执行
        results.append(result)
    # 总耗时: 5秒
    
    # ✅ 正确：使用 gather 实现并发
    tasks = [task(i) for i in range(5)]
    results = await asyncio.gather(*tasks)
    # 总耗时: 1秒

asyncio.run(main())
```

**修复方法：** 使用 `asyncio.gather()` 或 `asyncio.create_task()` 实现并发。

---

## 本章小结

本章我们学习了异步编程的基础知识：

1. **async def** 用于定义协程函数，调用后返回协程对象而非立即执行
2. **await** 用于暂停当前协程，等待可等待对象完成，让出控制权给事件循环
3. **asyncio.run()** 是启动异步程序的入口点，自动管理事件循环的生命周期
4. 协程函数和普通函数的调用方式不同，需要使用 `await` 获取结果
5. 连续的 `await` 会顺序执行，使用 `asyncio.gather()` 可实现并发
6. `await` 只能在 `async def` 中使用，且后面只能跟可等待对象

**关键概念：**

| 概念 | 说明 |
|------|------|
| 协程函数 | 用 `async def` 定义的函数 |
| 协程对象 | 调用协程函数产生的对象 |
| 可等待对象 | 协程、Task、Future 或实现 `__await__` 的对象 |
| 事件循环 | 管理和调度协程的机制 |
| await | 暂停协程并让出控制权 |

---

## 思考题

### 题目 1：基础理解

**问题：** 以下代码的输出是什么？为什么？

```python
import asyncio

async def hello():
    print("Hello")
    await asyncio.sleep(1)
    print("World")

asyncio.run(hello())
```

**答案：**
```
Hello
（等待 1 秒）
World
```

**解释：** `asyncio.run()` 启动事件循环并执行 `hello()` 协程。`await asyncio.sleep(1)` 暂停协程 1 秒，然后恢复执行打印 "World"。

---

### 题目 2：协程对象

**问题：** 以下代码有什么问题？如何修复？

```python
import asyncio

async def get_number():
    return 42

result = get_number()
print(result)
```

**答案：**
```
问题：忘记 await，result 是协程对象而不是数字 42
输出：<coroutine object get_number at 0x...>
```

**修复：**
```python
import asyncio

async def get_number():
    return 42

async def main():
    result = await get_number()
    print(result)  # 42

asyncio.run(main())
```

---

### 题目 3：顺序 vs 并发

**问题：** 以下两种写法的总耗时分别是多少？

```python
# 写法 A
async def main():
    await asyncio.sleep(2)
    await asyncio.sleep(2)

# 写法 B
async def main():
    await asyncio.gather(
        asyncio.sleep(2),
        asyncio.sleep(2)
    )
```

**答案：**
- 写法 A：4 秒（顺序执行，2+2）
- 写法 B：2 秒（并发执行，max(2,2)）

---

### 题目 4：await 的位置

**问题：** 以下代码能正常运行吗？为什么？

```python
import asyncio

def wrapper():
    return asyncio.sleep(1)

async def main():
    await wrapper()

asyncio.run(main())
```

**答案：**
不能正常运行。`wrapper()` 是普通函数，返回的是 `asyncio.sleep(1)` 的协程对象。但是 `await wrapper()` 实际上是可以工作的，因为 `await` 等待的是 `wrapper()` 的返回值（协程对象），而不是 `wrapper` 本身。

**修正答案：** 代码可以运行，`await` 会等待 `wrapper()` 返回的协程对象。

---

### 题目 5：嵌套协程

**问题：** 以下代码的输出顺序是什么？

```python
import asyncio

async def a():
    print("A 开始")
    await asyncio.sleep(1)
    print("A 结束")

async def b():
    print("B 开始")
    await a()
    print("B 结束")

async def main():
    await b()

asyncio.run(main())
```

**答案：**
```
B 开始
A 开始
A 结束
B 结束
```

**解释：** `main()` 调用 `b()`，`b()` 调用 `a()`。`a()` 执行完成后，`b()` 继续执行，最后 `main()` 完成。

---

### 题目 6：综合应用

**问题：** 编写一个异步函数，同时获取 3 个 API 的数据（模拟），并返回所有结果的列表。

**答案：**
```python
import asyncio

async def fetch_api(api_id: int) -> dict:
    """模拟获取 API 数据"""
    await asyncio.sleep(1)  # 模拟网络延迟
    return {"api": api_id, "data": f"数据{api_id}"}

async def fetch_all_apis() -> list:
    """并发获取所有 API 数据"""
    tasks = [fetch_api(i) for i in range(1, 4)]
    results = await asyncio.gather(*tasks)
    return results

async def main():
    results = await fetch_all_apis()
    for result in results:
        print(result)

asyncio.run(main())
```

**输出：**
```
{'api': 1, 'data': '数据1'}
{'api': 2, 'data': '数据2'}
{'api': 3, 'data': '数据3'}
```

---

### 题目 7：错误处理

**问题：** 以下代码有什么问题？

```python
import asyncio

async def main():
    await 42

asyncio.run(main())
```

**答案：**
```
TypeError: object int can't be used in 'await' expression
```

**解释：** `await` 后面只能跟可等待对象（协程、Task、Future 或实现 `__await__` 的对象）。`42` 是整数，不是可等待对象。

**修复：**
```python
import asyncio

async def main():
    # 如果需要返回一个值，应该直接 return
    result = 42
    print(result)

asyncio.run(main())
```

---

**下一章预告：** 第 3 章将介绍 `asyncio.create_task()` 和并发执行多个任务的方法。

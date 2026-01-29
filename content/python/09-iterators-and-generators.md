---
title: "Chapter 9. 迭代器与生成器：惰性计算的艺术"
description: "深入理解迭代器协议、生成器原理、yield 关键字、生成器表达式以及异步生成器"
updated: "2026-01-22"
---

> **Learning Objectives**
> * 理解迭代器协议 (`__iter__` 和 `__next__`)
> * 掌握可迭代对象 (Iterable) 与迭代器 (Iterator) 的区别
> * 深入生成器 (Generator) 的执行机制与状态保存
> * 熟练使用 `yield`、`yield from` 和生成器表达式
> * 理解生成器的优势：内存效率与惰性求值
> * 掌握 `itertools` 模块的强大工具

迭代器和生成器是 Python 中实现**惰性计算** (Lazy Evaluation) 的核心机制，允许我们处理大型数据集而不必一次性将所有数据载入内存。

---

## 9.1 迭代器协议

迭代 (Iteration) 是编程中最基础的操作之一——遍历集合中的每个元素。Python 通过**迭代器协议** (Iterator Protocol) 提供了一套统一的迭代机制，这使得列表、元组、字典、文件等不同类型的对象都可以用相同的方式遍历。

### 9.1.1 可迭代对象 vs 迭代器

**设计思想：为什么需要区分？**

Python 将"可以被迭代的对象"和"执行迭代的对象"分离开来，这是一个精妙的设计。想象一本书和书签：
- **书** (可迭代对象)：包含所有内容，可以被多次阅读
- **书签** (迭代器)：记录当前阅读位置，每次使用都会前进

这种分离带来几个好处：
1. **可重复迭代**：一个列表可以被多次遍历，每次都从头开始
2. **多重迭代**：可以同时用多个迭代器遍历同一个对象
3. **状态隔离**：迭代器维护自己的状态，互不干扰

```python
# 可迭代对象 (Iterable): 实现了 __iter__() 方法
numbers = [1, 2, 3, 4, 5]
print(hasattr(numbers, '__iter__'))  # True

# 获取迭代器
iterator = iter(numbers)  # 调用 numbers.__iter__()
print(type(iterator))  # <class 'list_iterator'>

# 迭代器 (Iterator): 实现了 __iter__() 和 __next__() 方法
print(hasattr(iterator, '__next__'))  # True

# 手动迭代
print(next(iterator))  # 1
print(next(iterator))  # 2
print(next(iterator))  # 3

# 迭代器耗尽后抛出 StopIteration
# next(iterator)  # StopIteration
```

### 9.1.2 for 循环的内部机制

`for` 循环是迭代器协议的语法糖。下面的可视化展示了 `for` 循环如何调用迭代器方法：

<div data-component="IteratorProtocolVisualizer"></div>

```python
# for 循环实际上是以下代码的语法糖:
numbers = [1, 2, 3]

# 等价于:
iterator = iter(numbers)  # 调用 __iter__()
while True:
    try:
        item = next(iterator)  # 调用 __next__()
        print(item)
    except StopIteration:
        break
```

### 9.1.3 自定义迭代器

```python
class Countdown:
    """倒计时迭代器"""
    def __init__(self, start):
        self.current = start
    
    def __iter__(self):
        return self  # 返回自身
    
    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

# 使用
for num in Countdown(5):
    print(num)  # 5, 4, 3, 2, 1
```

### 9.1.4 分离可迭代对象与迭代器

```python
class MyRange:
    """模拟 range() 的可迭代对象"""
    def __init__(self, start, end):
        self.start = start
        self.end = end
    
    def __iter__(self):
        return MyRangeIterator(self.start, self.end)

class MyRangeIterator:
    """MyRange 的迭代器"""
    def __init__(self, start, end):
        self.current = start
        self.end = end
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        value = self.current
        self.current += 1
        return value

# 测试：可以多次迭代
my_range = MyRange(1, 5)
for i in my_range:
    print(i, end=' ')  # 1 2 3 4
print()

for i in my_range:
    print(i, end=' ')  # 1 2 3 4 (可以再次迭代)
```

---

## 9.2 生成器基础

生成器是 Python 中最优雅的特性之一，它代表了一种完全不同的编程思维方式：**惰性计算** (Lazy Evaluation)。

### 9.2.1 什么是生成器？

**传统编程的困境**：

假设你需要处理一个包含 1000 万条记录的数据集。传统方法是先将所有数据加载到内存中的列表，然后逐个处理。但这会遇到两个问题：
1. **内存消耗巨大**：1000 万条记录可能占用数 GB 内存
2. **启动延迟**：必须等所有数据加载完才能开始处理

**生成器的解决方案**：

生成器采用**按需生成** (On-Demand Generation) 的策略：不一次性创建所有数据，而是每次只生成一个值，用完即丢弃。这就像流水线生产——不需要先生产 1000 万件产品再包装，而是生产一件包装一件。

**技术定义**：

生成器是一种特殊的迭代器，它通过 `yield` 关键字定义。与普通函数不同，生成器函数：
1. **不立即执行**：调用生成器函数时，函数体不会执行，而是返回一个生成器对象
2. **状态保存**：每次 `yield` 后，函数的执行状态（局部变量、执行位置）会被冻结
3. **延续执行**：下次调用 `next()` 时，从上次 `yield` 的位置继续执行

这种机制使得生成器能够实现**协程** (Coroutine) 般的暂停-恢复执行模式。

```python
def simple_generator():
    print("开始")
    yield 1
    print("继续")
    yield 2
    print("结束")
    yield 3

# 创建生成器（此时不执行函数体）
gen = simple_generator()
print(type(gen))  # <class 'generator'>

# 每次调用 next() 执行到下一个 yield
print(next(gen))
# 输出: 开始
# 输出: 1

print(next(gen))
# 输出: 继续
# 输出: 2

print(next(gen))
# 输出: 结束
# 输出: 3

# next(gen)  # StopIteration
```

### 9.2.2 生成器 vs 列表

```python
# 列表：立即计算所有值（占用内存）
def create_list(n):
    result = []
    for i in range(n):
        result.append(i ** 2)
    return result

lst = create_list(1000000)  # 立即占用大量内存

# 生成器：按需生成（内存高效）
def create_generator(n):
    for i in range(n):
        yield i ** 2

gen = create_generator(1000000)  # 几乎不占内存

# 测试内存占用
import sys
print(f"列表大小: {sys.getsizeof(lst)} bytes")  # 约 8MB
print(f"生成器大小: {sys.getsizeof(gen)} bytes")  # 约 200 bytes
```

### 9.2.3 yield 的执行流程

生成器使用 `yield` 关键字实现**状态暂停和恢复**。下面的可视化展示了生成器的执行状态机：

<div data-component="GeneratorStateVisualizer"></div>

```python
def fibonacci():
    """无限斐波那契数列生成器"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# 生成前 10 个斐波那契数
fib = fibonacci()
for _ in range(10):
    print(next(fib), end=' ')
# 0 1 1 2 3 5 8 13 21 34
```

---

## 9.3 生成器表达式

### 9.3.1 语法与用法

```python
# 列表推导式 - 立即计算
squares_list = [x**2 for x in range(10)]
print(type(squares_list))  # <class 'list'>

# 生成器表达式 - 惰性计算（使用圆括号）
squares_gen = (x**2 for x in range(10))
print(type(squares_gen))  # <class 'generator'>

# 遍历
for square in squares_gen:
    print(square, end=' ')  # 0 1 4 9 16 25 36 49 64 81
```

### 9.3.2 性能对比

```python
import time

n = 10000000

# 列表推导式
start = time.time()
sum([x**2 for x in range(n)])
print(f"列表推导式: {time.time() - start:.3f}秒")

# 生成器表达式
start = time.time()
sum((x**2 for x in range(n)))
print(f"生成器表达式: {time.time() - start:.3f}秒")

# 输出示例:
# 列表推导式: 2.123秒 (需要创建完整列表)
# 生成器表达式: 1.856秒 (逐个生成，无列表开销)
```

### 9.3.3 在函数调用中省略括号

```python
# 生成器表达式作为唯一参数时可省略括号
sum(x**2 for x in range(10))  # 等价于 sum((x**2 for x in range(10)))

max(x for x in range(100) if x % 7 == 0)

any(x > 100 for x in range(50))  # False
```

---

## 9.4 高级生成器特性

### 9.4.1 生成器的方法：send()

```python
def echo():
    """接收并回显数据的生成器"""
    while True:
        value = yield
        print(f"接收到: {value}")

gen = echo()
next(gen)  # 启动生成器（执行到第一个 yield）
gen.send("Hello")  # 接收到: Hello
gen.send(42)       # 接收到: 42
```

### 9.4.2 双向通信

```python
def running_average():
    """计算累积平均值"""
    total = 0
    count = 0
    average = None
    
    while True:
        value = yield average
        total += value
        count += 1
        average = total / count

avg = running_average()
next(avg)  # 启动生成器

print(avg.send(10))  # 10.0
print(avg.send(20))  # 15.0
print(avg.send(30))  # 20.0
```

### 9.4.3 throw() 和 close()

```python
def generator_with_exception():
    try:
        yield 1
        yield 2
        yield 3
    except ValueError as e:
        print(f"捕获异常: {e}")
        yield "错误处理"
    finally:
        print("清理资源")

gen = generator_with_exception()
print(next(gen))  # 1
print(gen.throw(ValueError, "测试异常"))  # 捕获异常: 测试异常, 输出: 错误处理
# gen.close()  # 触发 GeneratorExit，执行 finally
```

### 9.4.4 yield from 委托生成器

```python
def generator1():
    yield 1
    yield 2

def generator2():
    yield 'a'
    yield 'b'

# 不使用 yield from
def combined_old():
    for item in generator1():
        yield item
    for item in generator2():
        yield item

# 使用 yield from（更简洁）
def combined_new():
    yield from generator1()
    yield from generator2()

print(list(combined_new()))  # [1, 2, 'a', 'b']
```

#### yield from 的深层作用

```python
def averager_subgen():
    """子生成器"""
    total = 0
    count = 0
    
    while True:
        value = yield
        if value is None:
            break
        total += value
        count += 1
    
    return total / count if count else 0

def delegating_gen():
    """委托生成器"""
    result = yield from averager_subgen()
    print(f"平均值: {result}")

gen = delegating_gen()
next(gen)
gen.send(10)
gen.send(20)
gen.send(30)
try:
    gen.send(None)  # 触发子生成器 return
except StopIteration:
    pass
# 输出: 平均值: 20.0
```

---

## 9.5 实战案例

### 案例 1: 大文件逐行读取

```python
def read_large_file(file_path):
    """内存高效的文件读取"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield line.strip()

# 使用
# for line in read_large_file('large_file.txt'):
#     process(line)  # 一次处理一行，不占用大量内存
```

### 案例 2: 数据管道 (Pipeline)

```python
def read_numbers(filename):
    """读取数字"""
    with open(filename) as f:
        for line in f:
            yield float(line.strip())

def filter_positive(numbers):
    """过滤正数"""
    for num in numbers:
        if num > 0:
            yield num

def square(numbers):
    """平方"""
    for num in numbers:
        yield num ** 2

# 组合管道
# pipeline = square(filter_positive(read_numbers('data.txt')))
# for result in pipeline:
#     print(result)
```

### 案例 3: 无限序列生成器

```python
def count(start=0, step=1):
    """无限计数器（类似 itertools.count）"""
    while True:
        yield start
        start += step

def take(n, iterable):
    """取前 n 个元素"""
    for i, item in enumerate(iterable):
        if i >= n:
            break
        yield item

# 获取前 5 个偶数
evens = (x for x in count() if x % 2 == 0)
print(list(take(5, evens)))  # [0, 2, 4, 6, 8]
```

### 案例 4: 树的遍历

```python
class TreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def inorder_traversal(root):
    """中序遍历生成器"""
    if root:
        yield from inorder_traversal(root.left)
        yield root.value
        yield from inorder_traversal(root.right)

# 构建二叉树:     4
#               /   \
#              2     6
#             / \   / \
#            1   3 5   7

tree = TreeNode(4,
    TreeNode(2, TreeNode(1), TreeNode(3)),
    TreeNode(6, TreeNode(5), TreeNode(7))
)

print(list(inorder_traversal(tree)))  # [1, 2, 3, 4, 5, 6, 7]
```

---

## 9.6 itertools 模块

### 9.6.1 无限迭代器

```python
from itertools import count, cycle, repeat

# count(start, step): 无限计数
for i in count(10, 2):
    if i > 20:
        break
    print(i, end=' ')  # 10 12 14 16 18 20
print()

# cycle(iterable): 无限循环
counter = 0
for item in cycle(['A', 'B', 'C']):
    print(item, end=' ')
    counter += 1
    if counter >= 7:
        break  # A B C A B C A
print()

# repeat(value, times): 重复值
print(list(repeat('Python', 3)))  # ['Python', 'Python', 'Python']
```

### 9.6.2 组合迭代器

```python
from itertools import chain, zip_longest, product, permutations, combinations

# chain: 串联多个可迭代对象
print(list(chain([1, 2], ['a', 'b'])))  # [1, 2, 'a', 'b']

# zip_longest: 类似 zip，但用填充值处理不等长序列
print(list(zip_longest([1, 2], ['a', 'b', 'c'], fillvalue=0)))
# [(1, 'a'), (2, 'b'), (0, 'c')]

# product: 笛卡尔积
print(list(product([1, 2], ['a', 'b'])))
# [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]

# permutations: 排列
print(list(permutations([1, 2, 3], 2)))
# [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]

# combinations: 组合
print(list(combinations([1, 2, 3], 2)))
# [(1, 2), (1, 3), (2, 3)]
```

### 9.6.3 过滤与切片

```python
from itertools import islice, filterfalse, takewhile, dropwhile

numbers = range(10)

# islice: 切片（支持无限序列）
print(list(islice(numbers, 2, 7)))  # [2, 3, 4, 5, 6]

# filterfalse: 过滤掉满足条件的元素
print(list(filterfalse(lambda x: x % 2 == 0, numbers)))
# [1, 3, 5, 7, 9]

# takewhile: 获取元素直到条件不满足
print(list(takewhile(lambda x: x < 5, numbers)))
# [0, 1, 2, 3, 4]

# dropwhile: 丢弃元素直到条件不满足
print(list(dropwhile(lambda x: x < 5, numbers)))
# [5, 6, 7, 8, 9]
```

### 9.6.4 分组

```python
from itertools import groupby

# groupby: 按键分组（需要先排序）
data = [
    ('Alice', 'A'),
    ('Bob', 'B'),
    ('Charlie', 'A'),
    ('David', 'B'),
]

data.sort(key=lambda x: x[1])  # 按第二个元素排序

for key, group in groupby(data, key=lambda x: x[1]):
    print(f"{key}: {list(group)}")

# A: [('Alice', 'A'), ('Charlie', 'A')]
# B: [('Bob', 'B'), ('David', 'B')]
```

---

## 9.7 生成器的内存优势

### 9.7.1 大数据处理

```python
import sys

# 列表：占用大量内存
def sum_squares_list(n):
    return sum([x**2 for x in range(n)])

# 生成器：内存友好
def sum_squares_gen(n):
    return sum(x**2 for x in range(n))

n = 10000000

# 内存对比
list_comp = [x**2 for x in range(1000)]
gen_exp = (x**2 for x in range(1000))

print(f"列表占用: {sys.getsizeof(list_comp)} bytes")  # 约 9KB
print(f"生成器占用: {sys.getsizeof(gen_exp)} bytes")  # 约 200 bytes
```

### 9.7.2 流式处理

```python
def process_logs(filename):
    """流式处理日志文件"""
    with open(filename) as f:
        for line in f:
            # 解析
            if 'ERROR' in line:
                yield parse_error(line)

def parse_error(line):
    # 解析错误信息
    return {"line": line.strip()}

# 处理超大日志文件，无需一次性加载
# for error in process_logs('huge_log.txt'):
#     handle_error(error)
```

---

## 9.8 协程预览（与生成器的关系）

```python
# 协程是生成器的扩展（Python 3.5+ 使用 async/await）
async def async_generator():
    """异步生成器"""
    for i in range(5):
        await asyncio.sleep(0.1)
        yield i

# 详见异步编程章节
```

---

## 9.9 本章小结

在本章中,我们深入学习了：

1. ✅ 迭代器协议与可迭代对象
2. ✅ 生成器的定义与执行机制
3. ✅ yield、send()、throw()、yield from
4. ✅ 生成器表达式与内存优势
5. ✅ itertools 模块的强大工具
6. ✅ 实战案例：大文件处理、数据管道

> [!TIP]
> **关键要点**:
> - 生成器是惰性的，按需生成值
> - 适用于大数据集和无限序列
> - 比列表更节省内存
> - `yield from` 简化嵌套生成器

> **下一步学习建议**:
> - [Chapter 10. 上下文管理器](10-context-managers.md)
> - [Chapter 13. 异步编程](13-async-programming.md)

---

## 参考资源

- [PEP 255: Simple Generators](https://peps.python.org/pep-0255/)
- [PEP 342: Coroutines via Enhanced Generators](https://peps.python.org/pep-0342/)
- [PEP 380: Syntax for Delegating to a Subgenerator](https://peps.python.org/pep-0380/)
- [itertools 文档](https://docs.python.org/3/library/itertools.html)

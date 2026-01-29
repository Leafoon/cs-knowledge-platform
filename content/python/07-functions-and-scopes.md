---
title: "Chapter 7. 函数与作用域深度解析"
description: "深入理解 Python 函数机制、作用域规则 (LEGB)、参数传递、闭包原理以及函数式编程"
updated: "2026-01-22"
---

> **Learning Objectives**
> * 理解函数是第一类对象 (First-Class Object) 的含义
> * 掌握 LEGB 作用域查找规则与命名空间机制
> * 深入参数传递机制：位置参数、关键字参数、可变参数
> * 理解函数调用栈与执行上下文 (Call Stack & Execution Context)
> * 掌握闭包原理与自由变量捕获
> * 熟练使用函数式编程工具：lambda, map, filter, reduce

在 Python 中，**函数是第一类对象**，这意味着函数可以：
- 赋值给变量
- 作为参数传递给其他函数
- 作为函数的返回值
- 存储在数据结构中

---

## 7.1 函数基础

### 7.1.1 函数定义与调用

```python
# 基本函数定义
def greet(name):
    """
    这是文档字符串 (docstring)
    可通过 help(greet) 或 greet.__doc__ 访问
    """
    return f"Hello, {name}!"

# 调用函数
message = greet("Alice")
print(message)  # Hello, Alice!

# 访问函数属性
print(greet.__name__)  # greet
print(greet.__doc__)   # 这是文档字符串...
```

### 7.1.2 函数是对象

```python
def add(a, b):
    return a + b

# 赋值给变量
plus = add
print(plus(3, 5))  # 8

# 存储在列表中
operations = [add, lambda x, y: x - y, lambda x, y: x * y]
print(operations[0](10, 5))  # 15
print(operations[1](10, 5))  # 5

# 作为参数传递
def apply_operation(func, x, y):
    return func(x, y)

result = apply_operation(add, 3, 4)
print(result)  # 7
```

### 7.1.3 函数的内部结构

```python
def example(x, y):
    z = x + y
    return z

# 查看函数的 code 对象
print(example.__code__.co_varnames)  # ('x', 'y', 'z')
print(example.__code__.co_argcount)  # 2
print(example.__code__.co_nlocals)   # 3

# 查看字节码
import dis
dis.dis(example)
```

<div data-component="FunctionCallStackVisualizer"></div>

---

## 7.2 参数传递机制

### 7.2.1 位置参数与关键字参数

```python
def describe_person(name, age, city):
    return f"{name}, {age}岁, 来自{city}"

# 1. 位置参数
print(describe_person("Alice", 25, "Beijing"))

# 2. 关键字参数
print(describe_person(age=25, city="Beijing", name="Alice"))

# 3. 混合使用（位置参数必须在前）
print(describe_person("Alice", city="Beijing", age=25))

# ❌ 错误：位置参数在关键字参数后面
# print(describe_person(name="Alice", 25, "Beijing"))  # SyntaxError
```

### 7.2.2 默认参数

```python
def power(base, exponent=2):
    return base ** exponent

print(power(5))       # 25 (使用默认值 2)
print(power(5, 3))    # 125

# ⚠️ 陷阱：可变默认参数
def append_to(element, target=[]):  # ❌ 危险！
    target.append(element)
    return target

print(append_to(1))  # [1]
print(append_to(2))  # [1, 2] ❌ 不是期望的 [2]

# ✅ 正确做法
def append_to(element, target=None):
    if target is None:
        target = []
    target.append(element)
    return target

print(append_to(1))  # [1]
print(append_to(2))  # [2] ✅
```

> [!IMPORTANT]
> **默认参数陷阱解释**:
> 默认参数在函数**定义时求值一次**，而非每次调用时。
> 可变对象（如列表、字典）作为默认参数会在所有调用间共享。

### 7.2.3 可变参数：*args 和 **kwargs

```python
# *args: 接收任意数量的位置参数（元组）
def sum_all(*args):
    print(f"args 类型: {type(args)}")  # <class 'tuple'>
    return sum(args)

print(sum_all(1, 2, 3, 4, 5))  # 15

# **kwargs: 接收任意数量的关键字参数（字典）
def print_info(**kwargs):
    print(f"kwargs 类型: {type(kwargs)}")  # <class 'dict'>
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=25, city="Beijing")

# 混合使用（顺序：位置参数 → *args → 关键字参数 → **kwargs）
def complex_func(a, b, *args, x=10, **kwargs):
    print(f"a={a}, b={b}")
    print(f"args={args}")
    print(f"x={x}")
    print(f"kwargs={kwargs}")

complex_func(1, 2, 3, 4, 5, x=20, name="Alice", city="Beijing")
# a=1, b=2
# args=(3, 4, 5)
# x=20
# kwargs={'name': 'Alice', 'city': 'Beijing'}
```

### 7.2.4 强制关键字参数 (Keyword-Only Arguments)

```python
# * 后面的参数必须使用关键字传递
def create_user(name, *, age, email):
    return {"name": name, "age": age, "email": email}

# ✅ 正确
user = create_user("Alice", age=25, email="alice@example.com")

# ❌ 错误
# user = create_user("Alice", 25, "alice@example.com")  # TypeError
```

### 7.2.5 位置参数专用 (Positional-Only Parameters, Python 3.8+)

```python
# / 之前的参数只能按位置传递
def divide(a, b, /):
    return a / b

print(divide(10, 2))  # ✅ 5.0

# ❌ 错误
# print(divide(a=10, b=2))  # TypeError
```

### 7.2.6 参数解包

```python
# * 用于解包序列
def add(a, b, c):
    return a + b + c

numbers = [1, 2, 3]
print(add(*numbers))  # 6，等价于 add(1, 2, 3)

# ** 用于解包字典
def greet(name, age):
    return f"{name} is {age} years old"

person = {"name": "Alice", "age": 25}
print(greet(**person))  # Alice is 25 years old
```

---

## 7.3 作用域与命名空间 (LEGB 规则)

**作用域** (Scope) 是程序设计中的一个核心概念，它决定了变量的可见性和生命周期。理解作用域规则对于避免命名冲突、编写可维护的代码至关重要。

### 7.3.1 LEGB 查找顺序

在 Python 中，当你引用一个变量名时，解释器并不是随意查找的，而是遵循一套严格的规则——这就是著名的 **LEGB 规则**。这个缩写代表了 Python 查找变量的四个层次：

**为什么需要作用域层次？**

想象一下，如果所有变量都在同一个命名空间中，会发生什么？你在函数内部定义的临时变量可能会意外覆盖全局变量，不同模块的变量可能会相互冲突。作用域机制通过建立**层次化的命名空间**，实现了变量的**封装**和**隔离**。

**LEGB 的四个层次**：

1. **L (Local)**: **局部作用域** - 函数内部定义的变量，仅在函数执行期间存在
2. **E (Enclosing)**: **嵌套作用域** - 外层函数的局部作用域，实现闭包的关键
3. **G (Global)**: **全局作用域** - 模块级别的变量，在整个模块中可见
4. **B (Built-in)**: **内置作用域** - Python 预定义的名字（如 `print`, `len`, `int`）

查找顺序是**从内向外**的：Local → Enclosing → Global → Built-in。一旦找到就停止查找，这就是为什么你可以在函数内部定义与全局变量同名的局部变量而不会冲突。

<div data-component="LEGBScopeVisualizer"></div>

```python
# B: Built-in
print(len([1, 2, 3]))  # len 来自内置作用域

# G: Global
x = "global"

def outer():
    # E: Enclosing
    x = "enclosing"
    
    def inner():
        # L: Local
        x = "local"
        print(x)  # local
    
    inner()
    print(x)  # enclosing

outer()
print(x)  # global
```

### 7.3.2 global 关键字

```python
count = 0

def increment():
    global count  # 声明使用全局变量
    count += 1

increment()
print(count)  # 1

# 不使用 global 会报错
def bad_increment():
    # count += 1  # ❌ UnboundLocalError
    pass
```

### 7.3.3 nonlocal 关键字

```python
def outer():
    x = 10
    
    def inner():
        nonlocal x  # 修改外层函数的变量
        x += 5
        print(f"inner: x = {x}")
    
    inner()
    print(f"outer: x = {x}")

outer()
# inner: x = 15
# outer: x = 15
```

### 7.3.4 命名空间可视化

```python
# globals() 返回全局命名空间
x = 100
print("x" in globals())  # True

def func():
    # locals() 返回局部命名空间
    y = 200
    print(locals())  # {'y': 200}

func()
```

---

## 7.4 闭包 (Closure)

闭包是函数式编程中的一个强大概念，也是 Python 高级特性的基石。理解闭包不仅能帮助你掌握装饰器、迭代器等高级特性，还能让你写出更加优雅和简洁的代码。

### 7.4.1 闭包的定义

**什么是闭包？**

在计算机科学中，**闭包** (Closure) 是指一个函数对象，它能够记住并访问其定义时所在作用域的变量，即使这个作用域已经不存在了。

更通俗地说：闭包是一个函数，加上它"记住"的外部变量。就像一个背包客，不仅带着自己的装备（局部变量），还携带着从家里带来的物品（外部变量）。

**闭包的三个要素**：
1. **嵌套函数**：必须有一个内部函数（也叫闭包函数）
2. **外部引用**：内部函数引用了外部函数的变量（这些变量称为**自由变量**）
3. **函数返回**：外部函数返回内部函数对象（而不是调用它）

**为什么需要闭包？**

闭包解决了一个经典的编程问题：如何在不使用全局变量或类的情况下，让函数"记住"某些状态？闭包提供了一种轻量级的方式来实现数据封装和状态保持。

```python
def make_multiplier(factor):
    """创建一个乘法器闭包"""
    def multiplier(x):
        return x * factor  # factor 是自由变量
    return multiplier

# 创建闭包
times_3 = make_multiplier(3)
times_5 = make_multiplier(5)

print(times_3(10))  # 30
print(times_5(10))  # 50

# 查看闭包捕获的变量
print(times_3.__closure__)  # (<cell at ...: int object at ...>,)
print(times_3.__closure__[0].cell_contents)  # 3
```

### 7.4.2 闭包的实际应用

#### 案例 1: 计数器

```python
def make_counter():
    count = 0
    
    def counter():
        nonlocal count
        count += 1
        return count
    
    return counter

c1 = make_counter()
c2 = make_counter()

print(c1())  # 1
print(c1())  # 2
print(c2())  # 1（独立的计数器）
```

#### 案例 2: 延迟计算

```python
def make_averager():
    """创建一个累积平均值计算器"""
    values = []
    
    def averager(new_value):
        values.append(new_value)
        return sum(values) / len(values)
    
    return averager

avg = make_averager()
print(avg(10))  # 10.0
print(avg(20))  # 15.0
print(avg(30))  # 20.0
```

#### 案例 3: 函数工厂

```python
def make_tag(tag_name):
    """创建 HTML 标签生成器"""
    def make_element(content):
        return f"<{tag_name}>{content}</{tag_name}>"
    return make_element

h1 = make_tag("h1")
p = make_tag("p")

print(h1("标题"))     # <h1>标题</h1>
print(p("段落内容"))  # <p>段落内容</p>
```

### 7.4.3 闭包的陷阱：延迟绑定

```python
# ❌ 常见错误
def create_multipliers():
    return [lambda x: x * i for i in range(5)]

funcs = create_multipliers()
print(funcs[0](10))  # 期望 0，实际 40 ❌
print(funcs[1](10))  # 期望 10，实际 40 ❌
print(funcs[4](10))  # 期望 40，实际 40 ✅

# 原因：闭包捕获的是变量引用，而非值
# 所有 lambda 都引用同一个 i，循环结束时 i=4

# ✅ 解决方案1：使用默认参数立即绑定
def create_multipliers():
    return [lambda x, i=i: x * i for i in range(5)]

funcs = create_multipliers()
print(funcs[0](10))  # 0 ✅
print(funcs[1](10))  # 10 ✅

# ✅ 解决方案2：使用函数工厂
def create_multipliers():
    def make_multiplier(factor):
        return lambda x: x * factor
    return [make_multiplier(i) for i in range(5)]
```

---

## 7.5 函数式编程

### 7.5.1 Lambda 表达式

```python
# lambda 参数: 表达式
add = lambda x, y: x + y
print(add(3, 5))  # 8

# 常用于排序
students = [
    ("Alice", 85),
    ("Bob", 92),
    ("Charlie", 78)
]
students.sort(key=lambda s: s[1], reverse=True)
print(students)  # [('Bob', 92), ('Alice', 85), ('Charlie', 78)]

# 限制：只能包含单个表达式，不能有语句
# square = lambda x: return x**2  # ❌ SyntaxError
square = lambda x: x**2  # ✅
```

### 7.5.2 map() - 映射

```python
# map(function, iterable)
numbers = [1, 2, 3, 4, 5]

# 平方
squares = map(lambda x: x**2, numbers)
print(list(squares))  # [1, 4, 9, 16, 25]

# 多个可迭代对象
a = [1, 2, 3]
b = [10, 20, 30]
sums = map(lambda x, y: x + y, a, b)
print(list(sums))  # [11, 22, 33]

# 等价的列表推导式（更 Pythonic）
squares = [x**2 for x in numbers]
```

### 7.5.3 filter() - 过滤

```python
# filter(function, iterable)
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 过滤偶数
evens = filter(lambda x: x % 2 == 0, numbers)
print(list(evens))  # [2, 4, 6, 8, 10]

# 等价的列表推导式
evens = [x for x in numbers if x % 2 == 0]
```

### 7.5.4 reduce() - 归约

```python
from functools import reduce

# reduce(function, iterable[, initial])
numbers = [1, 2, 3, 4, 5]

# 求和
total = reduce(lambda x, y: x + y, numbers)
print(total)  # 15

# 等价于
# total = 1 + 2 = 3
# total = 3 + 3 = 6
# total = 6 + 4 = 10
# total = 10 + 5 = 15

# 求积
product = reduce(lambda x, y: x * y, numbers)
print(product)  # 120

# 提供初始值
total = reduce(lambda x, y: x + y, numbers, 100)
print(total)  # 115
```

### 7.5.5 高阶函数组合

```python
# 案例：找出列表中所有偶数的平方和
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 函数式风格
from functools import reduce
result = reduce(
    lambda x, y: x + y,
    map(lambda x: x**2, filter(lambda x: x % 2 == 0, numbers))
)
print(result)  # 220 (4 + 16 + 36 + 64 + 100)

# Pythonic 风格（推荐）
result = sum(x**2 for x in numbers if x % 2 == 0)
print(result)  # 220
```

---

## 7.6 递归函数

### 7.6.1 递归基础

```python
# 经典案例：阶乘
def factorial(n):
    # 基线条件（递归出口）
    if n == 0 or n == 1:
        return 1
    # 递归条件
    return n * factorial(n - 1)

print(factorial(5))  # 120

# 执行过程:
# factorial(5) = 5 * factorial(4)
# factorial(4) = 4 * factorial(3)
# factorial(3) = 3 * factorial(2)
# factorial(2) = 2 * factorial(1)
# factorial(1) = 1
# 回溯: 2*1 = 2, 3*2 = 6, 4*6 = 24, 5*24 = 120
```

### 7.6.2 斐波那契数列

```python
# 朴素递归（效率低，大量重复计算）
def fib_naive(n):
    if n <= 1:
        return n
    return fib_naive(n-1) + fib_naive(n-2)

# ✅ 使用缓存优化 (Memoization)
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

print(fib(100))  # 354224848179261915075
# 朴素版本计算 fib(100) 需要几十亿次递归
# 缓存版本瞬间完成！
```

### 7.6.3 尾递归与 Python 限制

```python
# Python 不支持尾递归优化
def factorial_tail(n, acc=1):
    if n == 0:
        return acc
    return factorial_tail(n - 1, n * acc)

# 超过递归深度限制会报错
import sys
print(sys.getrecursionlimit())  # 默认 1000

# ❌ 这会导致 RecursionError
# factorial_tail(2000)

# 可以修改限制（不推荐）
# sys.setrecursionlimit(3000)

# ✅ 更好的方案：使用迭代
def factorial_iterative(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
```

---

## 7.7 装饰器预览

装饰器本质是接收函数并返回函数的高阶函数。详细内容见 [Chapter 8. 装饰器与闭包](08-decorators-and-closures.md)。

```python
# 简单装饰器示例
def timer(func):
    """测量函数执行时间"""
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} 执行时间: {end - start:.4f}秒")
        return result
    return wrapper

@timer
def slow_function():
    import time
    time.sleep(1)
    return "完成"

slow_function()  # slow_function 执行时间: 1.0001秒
```

---

## 7.8 实战案例

### 案例 1: 实现 reduce (从零开始)

```python
def my_reduce(function, iterable, initial=None):
    """手动实现 reduce 函数"""
    it = iter(iterable)
    
    if initial is None:
        try:
            value = next(it)
        except StopIteration:
            raise TypeError("reduce() of empty sequence")
    else:
        value = initial
    
    for element in it:
        value = function(value, element)
    
    return value

# 测试
numbers = [1, 2, 3, 4, 5]
print(my_reduce(lambda x, y: x + y, numbers))  # 15
```

### 案例 2: 柯里化 (Currying)

```python
def curry(func):
    """将多参数函数转换为单参数函数链"""
    def curried(x):
        def inner(y):
            return func(x, y)
        return inner
    return curried

# 普通函数
def add(x, y):
    return x + y

# 柯里化
curried_add = curry(add)
add_5 = curried_add(5)
print(add_5(10))  # 15

# 或者直接
print(curried_add(3)(7))  # 10
```

### 案例 3: 函数组合

```python
def compose(*functions):
    """组合多个函数: compose(f, g, h)(x) = f(g(h(x)))"""
    def inner(arg):
        result = arg
        for func in reversed(functions):
            result = func(result)
        return result
    return inner

# 示例
def add_10(x):
    return x + 10

def multiply_2(x):
    return x * 2

def square(x):
    return x ** 2

# 组合: square(multiply_2(add_10(x)))
pipeline = compose(square, multiply_2, add_10)
print(pipeline(5))  # ((5 + 10) * 2) ** 2 = 900
```

---

## 7.9 本章小结

在本章中，我们深入学习了：

1. ✅ 函数作为第一类对象的特性
2. ✅ 参数传递：位置、关键字、默认、可变参数
3. ✅ LEGB 作用域规则与命名空间
4. ✅ 闭包原理与自由变量捕获
5. ✅ 函数式编程：lambda, map, filter, reduce
6. ✅ 递归与尾递归优化
7. ✅ 高阶函数与函数组合

> [!TIP]
> **下一步学习建议**:
> - [Chapter 8. 装饰器与闭包](08-decorators-and-closures.md) - 深入装饰器模式
> - [Chapter 11. 迭代器与生成器](11-iterators-and-generators.md) - 惰性计算

---

## 参考资源

- [PEP 3102: Keyword-Only Arguments](https://peps.python.org/pep-3102/)
- [PEP 570: Positional-Only Parameters](https://peps.python.org/pep-0570/)
- [Functional Programming HOWTO](https://docs.python.org/3/howto/functional.html)

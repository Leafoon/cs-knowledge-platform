---
title: "Chapter 8. 装饰器与闭包高级应用"
description: "深入理解装饰器原理、实现模式、常用装饰器库以及装饰器工厂"
updated: "2026-01-22"
---

> **Learning Objectives**
> * 理解装饰器的本质：语法糖背后的函数包装
> * 掌握装饰器的实现模式：保留元数据、参数传递
> * 深入装饰器工厂：带参数的装饰器
> * 熟练使用标准库装饰器：`@staticmethod`, `@classmethod`, `@property`
> * 实现高级装饰器：缓存、重试、权限验证、性能分析

装饰器是 Python 中最强大的特性之一，它允许在不修改原函数代码的前提下，为函数添加额外功能。

---

## 8.1 装饰器基础

装饰器是 Python 中最优雅也最强大的特性之一。它体现了 Python "简洁胜于复杂" 的设计哲学，让我们能够以简洁的语法为函数添加功能。

### 8.1.1 装饰器的本质

**什么是装饰器？为什么需要它？**

在软件开发中，我们经常需要为现有函数添加额外功能，比如：
- 记录函数执行日志
- 测量函数执行时间
- 验证函数参数
- 缓存函数结果
- 权限检查

传统做法是修改函数内部代码，但这违反了**开闭原则** (Open-Closed Principle)：软件实体应该对扩展开放，对修改封闭。更糟糕的是，如果你需要为数十个函数添加相同的功能，就需要重复修改每个函数。

**装饰器的核心思想**：不修改原函数，而是通过"包装"的方式在外部添加功能。这就像给手机套上保护壳——手机本身没有改变，但获得了额外的保护功能。

从技术角度讲，装饰器本质上是一个**高阶函数** (Higher-Order Function)，它：
1. **接收**一个函数作为参数
2. **返回**一个新的函数（通常是包装后的版本）
3. 新函数在调用原函数的前后添加额外功能

```python
# 手动装饰
def greet():
    return "Hello!"

def make_bold(func):
    def wrapper():
        return f"<b>{func()}</b>"
    return wrapper

greet = make_bold(greet)  # 手动包装
print(greet())  # <b>Hello!</b>
```

### 8.1.2 @语法糖

```python
def make_bold(func):
    def wrapper():
        return f"<b>{func()}</b>"
    return wrapper

# 使用 @ 语法糖（等价于上面的手动装饰）
@make_bold
def greet():
    return "Hello!"

print(greet())  # <b>Hello!</b>

# 等价于:
# greet = make_bold(greet)
```

### 8.1.3 执行流程可视化

装饰器的执行分为两个阶段：**定义时**包装和**调用时**执行。下面的可视化展示了完整的执行流程：

<div data-component="DecoratorExecutionFlow"></div>

```python
def trace(func):
    print(f"装饰器执行: 包装 {func.__name__}")
    
    def wrapper():
        print(f"调用前: {func.__name__}")
        result = func()
        print(f"调用后: {func.__name__}")
        return result
    
    return wrapper

@trace
def say_hello():
    print("Hello!")
    return "完成"

# 输出顺序:
# 装饰器执行: 包装 say_hello  (定义时)

print("--- 开始调用函数 ---")
say_hello()
# 调用前: say_hello
# Hello!
# 调用后: say_hello
```

---

## 8.2 装饰带参数的函数

### 8.2.1 使用 *args 和 **kwargs

```python
def logger(func):
    def wrapper(*args, **kwargs):
        print(f"调用 {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} 返回 {result}")
        return result
    return wrapper

@logger
def add(a, b):
    return a + b

@logger
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

add(3, 5)
# 调用 add with args=(3, 5), kwargs={}
# add 返回 8

greet("Alice", greeting="Hi")
# 调用 greet with args=('Alice',), kwargs={'greeting': 'Hi'}
# greet 返回 Hi, Alice!
```

### 8.2.2 保留函数元数据

```python
def simple_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@simple_decorator
def greet(name):
    """问候函数"""
    return f"Hello, {name}!"

# ❌ 问题：元数据丢失
print(greet.__name__)  # wrapper (不是 greet)
print(greet.__doc__)   # None (文档字符串丢失)

# ✅ 解决方案：使用 functools.wraps
from functools import wraps

def better_decorator(func):
    @wraps(func)  # 复制元数据
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@better_decorator
def greet(name):
    """问候函数"""
    return f"Hello, {name}!"

print(greet.__name__)  # greet ✅
print(greet.__doc__)   # 问候函数 ✅
```

---

## 8.3 装饰器工厂（带参数的装饰器）

### 8.3.1 三层嵌套结构

装饰器工厂使用**三层嵌套函数**实现参数化装饰器。下面的可视化展示了这种嵌套结构的执行过程：

<div data-component="DecoratorFactoryVisualizer"></div>

```python
def repeat(times):
    """装饰器工厂：重复执行函数"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            results = []
            for _ in range(times):
                results.append(func(*args, **kwargs))
            return results
        return wrapper
    return decorator

@repeat(times=3)
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))
# ['Hello, Alice!', 'Hello, Alice!', 'Hello, Alice!']

# 等价于:
# greet = repeat(3)(greet)
```

### 8.3.2 带可选参数的装饰器

```python
from functools import wraps

def optional_arg_decorator(func=None, *, prefix=">>>"):
    """
    可以带参数也可以不带参数的装饰器
    @optional_arg_decorator          # 不带参数
    @optional_arg_decorator(prefix=">>")  # 带参数
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            result = f(*args, **kwargs)
            return f"{prefix} {result}"
        return wrapper
    
    if func is None:
        # 带参数调用: @optional_arg_decorator(prefix=">>")
        return decorator
    else:
        # 不带参数调用: @optional_arg_decorator
        return decorator(func)

@optional_arg_decorator
def greet1(name):
    return f"Hello, {name}"

@optional_arg_decorator(prefix="***")
def greet2(name):
    return f"Hello, {name}"

print(greet1("Alice"))  # >>> Hello, Alice
print(greet2("Bob"))    # *** Hello, Bob
```

---

## 8.4 常用装饰器模式

### 8.4.1 计时装饰器

```python
import time
from functools import wraps

def timer(func):
    """测量函数执行时间"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} 执行时间: {end - start:.4f}秒")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "完成"

slow_function()  # slow_function 执行时间: 1.0001秒
```

### 8.4.2 缓存装饰器 (Memoization)

```python
from functools import wraps

def memoize(func):
    """缓存函数结果"""
    cache = {}
    
    @wraps(func)
    def wrapper(*args):
        if args not in cache:
            print(f"计算 {func.__name__}{args}")
            cache[args] = func(*args)
        else:
            print(f"从缓存读取 {func.__name__}{args}")
        return cache[args]
    
    return wrapper

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
# 只计算一次每个值，后续从缓存读取

# 标准库版本（更强大）
from functools import lru_cache

@lru_cache(maxsize=128)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)
```

### 8.4.3 重试装饰器

```python
import time
from functools import wraps

def retry(max_attempts=3, delay=1):
    """失败时重试"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise
                    print(f"尝试 {attempts} 失败: {e}，{delay}秒后重试...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=2)
def unstable_network_call():
    import random
    if random.random() < 0.7:  # 70% 失败率
        raise ConnectionError("网络错误")
    return "成功"

# unstable_network_call()  # 会自动重试
```

### 8.4.4 权限验证装饰器

```python
from functools import wraps

def require_permission(permission):
    """检查用户权限"""
    def decorator(func):
        @wraps(func)
        def wrapper(user, *args, **kwargs):
            if permission not in user.get("permissions", []):
                raise PermissionError(f"需要权限: {permission}")
            return func(user, *args, **kwargs)
        return wrapper
    return decorator

@require_permission("admin")
def delete_user(user, user_id):
    return f"用户 {user_id} 已删除"

# 测试
admin = {"name": "Alice", "permissions": ["admin", "read"]}
normal_user = {"name": "Bob", "permissions": ["read"]}

print(delete_user(admin, 123))  # 用户 123 已删除
# delete_user(normal_user, 123)  # ❌ PermissionError
```

### 8.4.5 单例模式装饰器

```python
from functools import wraps

def singleton(cls):
    """将类转换为单例"""
    instances = {}
    
    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

@singleton
class Database:
    def __init__(self, url):
        print(f"连接数据库: {url}")
        self.url = url

# 测试
db1 = Database("localhost:5432")  # 连接数据库: localhost:5432
db2 = Database("localhost:5432")  # 不会再次连接
print(db1 is db2)  # True (同一个实例)
```

### 8.4.6 类型检查装饰器

```python
from functools import wraps
from typing import get_type_hints

def type_check(func):
    """运行时类型检查"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        hints = get_type_hints(func)
        
        # 检查位置参数
        arg_names = func.__code__.co_varnames[:len(args)]
        for arg_name, arg_value in zip(arg_names, args):
            if arg_name in hints:
                expected_type = hints[arg_name]
                if not isinstance(arg_value, expected_type):
                    raise TypeError(
                        f"{arg_name} 应为 {expected_type}，实际为 {type(arg_value)}"
                    )
        
        result = func(*args, **kwargs)
        
        # 检查返回值
        if "return" in hints:
            expected_type = hints["return"]
            if not isinstance(result, expected_type):
                raise TypeError(
                    f"返回值应为 {expected_type}，实际为 {type(result)}"
                )
        
        return result
    return wrapper

@type_check
def add(a: int, b: int) -> int:
    return a + b

print(add(3, 5))  # 8
# add(3, "5")  # ❌ TypeError: b 应为 <class 'int'>，实际为 <class 'str'>
```

---

## 8.5 类装饰器

### 8.5.1 使用类实现装饰器

```python
class CountCalls:
    """统计函数调用次数"""
    def __init__(self, func):
        self.func = func
        self.count = 0
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"调用 {self.func.__name__} 第 {self.count} 次")
        return self.func(*args, **kwargs)

@CountCalls
def greet(name):
    return f"Hello, {name}!"

greet("Alice")  # 调用 greet 第 1 次
greet("Bob")    # 调用 greet 第 2 次
print(greet.count)  # 2
```

### 8.5.2 装饰类

```python
def add_repr(cls):
    """为类添加 __repr__ 方法"""
    def __repr__(self):
        attrs = ', '.join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{cls.__name__}({attrs})"
    
    cls.__repr__ = __repr__
    return cls

@add_repr
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

p = Point(3, 4)
print(p)  # Point(x=3, y=4)
```

---

## 8.6 多个装饰器的叠加

### 8.6.1 执行顺序

```python
def decorator1(func):
    print("应用 decorator1")
    def wrapper(*args, **kwargs):
        print("decorator1 调用前")
        result = func(*args, **kwargs)
        print("decorator1 调用后")
        return result
    return wrapper

def decorator2(func):
    print("应用 decorator2")
    def wrapper(*args, **kwargs):
        print("decorator2 调用前")
        result = func(*args, **kwargs)
        print("decorator2 调用后")
        return result
    return wrapper

@decorator1
@decorator2
def greet():
    print("Hello!")

# 定义时输出:
# 应用 decorator2
# 应用 decorator1

greet()
# decorator1 调用前
# decorator2 调用前
# Hello!
# decorator2 调用后
# decorator1 调用后

# 等价于: greet = decorator1(decorator2(greet))
# 装饰顺序：从下到上
# 执行顺序：从上到下
```

---

## 8.7 标准库装饰器

### 8.7.1 @staticmethod 和 @classmethod

```python
class MyClass:
    count = 0
    
    def __init__(self, name):
        self.name = name
        MyClass.count += 1
    
    # 实例方法（默认）
    def instance_method(self):
        return f"实例方法: {self.name}"
    
    # 类方法：第一个参数是类本身
    @classmethod
    def class_method(cls):
        return f"类方法: 共有 {cls.count} 个实例"
    
    # 静态方法：无特殊参数
    @staticmethod
    def static_method(x, y):
        return x + y

# 测试
obj1 = MyClass("A")
obj2 = MyClass("B")

print(obj1.instance_method())  # 实例方法: A
print(MyClass.class_method())  # 类方法: 共有 2 个实例
print(MyClass.static_method(3, 5))  # 8
```

### 8.7.2 @property

```python
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        """Getter"""
        return self._radius
    
    @radius.setter
    def radius(self, value):
        """Setter"""
        if value < 0:
            raise ValueError("半径不能为负数")
        self._radius = value
    
    @property
    def area(self):
        """只读属性"""
        import math
        return math.pi * self._radius ** 2

# 使用
c = Circle(5)
print(c.radius)  # 5
print(c.area)    # 78.53981633974483

c.radius = 10    # 调用 setter
print(c.area)    # 314.1592653589793

# c.area = 100   # ❌ AttributeError: 只读属性
```

### 8.7.3 @dataclass (Python 3.7+)

```python
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int
    city: str = "北京"  # 默认值
    
    def greet(self):
        return f"我是 {self.name}，今年 {self.age} 岁"

# 自动生成 __init__, __repr__, __eq__ 等方法
p1 = Person("Alice", 25)
p2 = Person("Alice", 25)

print(p1)  # Person(name='Alice', age=25, city='北京')
print(p1 == p2)  # True
```

---

## 8.8 实战案例

### 案例 1: 完整的缓存装饰器

```python
from functools import wraps
import time

def cache_with_ttl(ttl_seconds=60):
    """带过期时间的缓存"""
    def decorator(func):
        cache = {}
        
        @wraps(func)
        def wrapper(*args):
            now = time.time()
            
            # 清理过期缓存
            expired_keys = [k for k, (_, timestamp) in cache.items() 
                          if now - timestamp > ttl_seconds]
            for k in expired_keys:
                del cache[k]
            
            # 从缓存读取或计算
            if args in cache:
                value, _ = cache[args]
                print(f"缓存命中: {args}")
                return value
            
            print(f"计算: {args}")
            value = func(*args)
            cache[args] = (value, now)
            return value
        
        wrapper.cache_info = lambda: {
            "size": len(cache),
            "keys": list(cache.keys())
        }
        
        return wrapper
    return decorator

@cache_with_ttl(ttl_seconds=5)
def expensive_computation(n):
    time.sleep(1)  # 模拟耗时操作
    return n ** 2

print(expensive_computation(10))  # 计算: (10,)
print(expensive_computation(10))  # 缓存命中: (10,)
time.sleep(6)
print(expensive_computation(10))  # 计算: (10,) (缓存已过期)
```

### 案例 2: 日志装饰器

```python
import logging
from functools import wraps

logging.basicConfig(level=logging.INFO)

def log_calls(level=logging.INFO):
    """记录函数调用日志"""
    def decorator(func):
        logger = logging.getLogger(func.__module__)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            
            logger.log(level, f"调用 {func.__name__}({signature})")
            
            try:
                result = func(*args, **kwargs)
                logger.log(level, f"{func.__name__} 返回 {result!r}")
                return result
            except Exception as e:
                logger.exception(f"{func.__name__} 抛出异常: {e}")
                raise
        
        return wrapper
    return decorator

@log_calls(level=logging.DEBUG)
def divide(a, b):
    return a / b

divide(10, 2)
# INFO:__main__:调用 divide(10, 2)
# INFO:__main__:divide 返回 5.0
```

---

## 8.9 本章小结

在本章中，我们深入学习了：

1. ✅ 装饰器的本质与语法糖
2. ✅ 保留函数元数据 (`@wraps`)
3. ✅ 装饰器工厂：带参数的装饰器
4. ✅ 常用装饰器模式：缓存、计时、重试、权限验证
5. ✅ 类装饰器与装饰类
6. ✅ 标准库装饰器：`@property`, `@staticmethod`, `@dataclass`

> [!TIP]
> **下一步学习建议**:
> - [Chapter 9. 面向对象编程基础](09-oop-basics.md)
> - [Chapter 11. 迭代器与生成器](11-iterators-and-generators.md)

---

## 参考资源

- [PEP 318: Decorators for Functions and Methods](https://peps.python.org/pep-0318/)
- [functools 标准库文档](https://docs.python.org/3/library/functools.html)

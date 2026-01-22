---
title: "Chapter 10. 异常处理与错误管理"
description: "深入理解 Python 异常机制、异常层次结构、自定义异常、异常链以及最佳实践"
updated: "2026-01-22"
---

# Chapter 10. 异常处理与错误管理

> **Learning Objectives**
> * 理解异常的本质与 Python 异常层次结构
> * 掌握 try-except-else-finally 完整语法
> * 学会自定义异常类与异常链
> * 理解异常处理的最佳实践与性能考量
> * 掌握上下文管理器与资源清理
> * 熟悉常见异常类型及其处理方法

异常处理是编写健壮程序的关键。Python 采用"请求原谅比请求许可更容易" (EAFP) 的哲学，鼓励使用异常处理而非事先检查。

---

## 10.1 异常基础

**错误处理**是区分专业程序员和业余爱好者的重要标志。一个健壮的程序不仅要能在理想情况下正确运行，还要能优雅地处理各种意外情况。

### 10.1.1 什么是异常？

**从现实世界理解异常**：

想象你在餐厅点餐：
- **正常流程**：点菜 → 厨师做菜 → 上菜 → 用餐
- **异常情况**：食材用完了、厨师生病了、停电了...

在编程中，**异常** (Exception) 就是这些"意外情况"。它是程序执行过程中发生的错误事件，会**中断正常的程序流程**。如果不处理，程序就会崩溃。

**异常 vs 语法错误**：

初学者容易混淆这两个概念：
- **语法错误** (Syntax Error)：代码违反了 Python 语法规则，程序根本无法运行
  ```python
  if True  # 缺少冒号，语法错误
      print("Hello")
  ```

- **异常**：代码语法正确，但运行时发生错误
  ```python
  result = 10 / 0  # 语法正确，但运行时抛出 ZeroDivisionError
  ```

异常是运行时的概念，只有在代码执行到特定位置时才会发生。

```python
# 示例：除零错误
def divide(a, b):
    return a / b

result = divide(10, 0)  # ❌ ZeroDivisionError: division by zero
```

### 10.1.2 基本异常处理

```python
def safe_divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("错误：不能除以零")
        result = None
    return result

print(safe_divide(10, 2))  # 5.0
print(safe_divide(10, 0))  # 错误：不能除以零, None
```

### 10.1.3 捕获异常对象

```python
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"异常类型: {type(e).__name__}")
    print(f"异常信息: {e}")
    print(f"异常参数: {e.args}")

# 输出:
# 异常类型: ZeroDivisionError
# 异常信息: division by zero
# 异常参数: ('division by zero',)
```

---

## 10.2 异常层次结构

Python 的所有异常都继承自 `BaseException`。理解异常层次结构对于正确捕获和处理异常至关重要：

<div data-component="ExceptionHierarchyTree"></div>

```
BaseException
├── SystemExit          (sys.exit() 触发)
├── KeyboardInterrupt   (Ctrl+C 触发)
├── GeneratorExit       (生成器关闭)
└── Exception           (常规异常的基类)
    ├── StopIteration
    ├── ArithmeticError
    │   ├── ZeroDivisionError
    │   ├── OverflowError
    │   └── FloatingPointError
    ├── LookupError
    │   ├── IndexError
    │   └── KeyError
    ├── ValueError
    ├── TypeError
    ├── AttributeError
    ├── NameError
    ├── OSError
    │   ├── FileNotFoundError
    │   ├── PermissionError
    │   └── ...
    └── ...
```

```python
# 常见异常示例
try:
    lst = [1, 2, 3]
    print(lst[10])  # IndexError
except LookupError as e:  # IndexError 的父类
    print(f"查找错误: {e}")

try:
    int("abc")  # ValueError
except ValueError as e:
    print(f"值错误: {e}")
```

---

## 10.3 完整的异常处理语法

### 10.3.1 try-except-else-finally

```python
def read_number_from_file(filename):
    try:
        # 可能抛出异常的代码
        f = open(filename, 'r')
        content = f.read()
        number = int(content)
    except FileNotFoundError:
        # 文件不存在
        print(f"文件 {filename} 不存在")
        return None
    except ValueError:
        # 内容不是数字
        print("文件内容不是有效数字")
        return None
    except Exception as e:
        # 捕获其他所有异常
        print(f"未知错误: {e}")
        return None
    else:
        # 没有异常时执行
        print("文件读取成功")
        return number
    finally:
        # 无论是否异常都执行（常用于清理资源）
        try:
            f.close()
            print("文件已关闭")
        except:
            pass

# 测试
# read_number_from_file('number.txt')
```

### 10.3.2 执行流程

`try-except-else-finally` 语句的执行流程较为复杂。下面的可视化展示了不同情况下的执行路径：

<div data-component="ExceptionFlowVisualizer"></div>

```python
def demonstrate_flow():
    try:
        print("1. try 块开始")
        # return "try 中返回"  # 如果有 return，先记录，等 finally 执行后才返回
        print("2. try 块结束")
    except Exception:
        print("3. except 块")
    else:
        print("4. else 块（无异常时执行）")
    finally:
        print("5. finally 块（总是执行）")
    
    print("6. 函数结束")

demonstrate_flow()
# 输出:
# 1. try 块开始
# 2. try 块结束
# 4. else 块（无异常时执行）
# 5. finally 块（总是执行）
# 6. 函数结束
```

---

## 10.4 捕获多个异常

### 10.4.1 方式1：分别捕获

```python
try:
    value = int(input("输入数字: "))
    result = 100 / value
except ValueError:
    print("输入不是数字")
except ZeroDivisionError:
    print("不能除以零")
```

### 10.4.2 方式2：元组捕获

```python
try:
    value = int(input("输入数字: "))
    result = 100 / value
except (ValueError, ZeroDivisionError) as e:
    print(f"输入错误: {e}")
```

### 10.4.3 捕获异常顺序

```python
# ✅ 正确：子类在前
try:
    # ...
    pass
except FileNotFoundError:  # 子类
    print("文件不存在")
except OSError:  # 父类
    print("OS 错误")

# ❌ 错误：父类在前会捕获所有子类异常
try:
    # ...
    pass
except OSError:  # 父类捕获了所有 OSError 子类
    print("OS 错误")
except FileNotFoundError:  # 永远不会执行！
    print("文件不存在")
```

---

## 10.5 抛出异常

### 10.5.1 raise 语句

```python
def validate_age(age):
    if age < 0:
        raise ValueError("年龄不能为负数")
    if age > 150:
        raise ValueError("年龄不合理")
    return age

try:
    validate_age(-5)
except ValueError as e:
    print(f"验证失败: {e}")
```

### 10.5.2 重新抛出异常

```python
def process_data(data):
    try:
        # 处理数据
        result = int(data)
    except ValueError:
        print("记录错误日志...")
        raise  # 重新抛出相同的异常

try:
    process_data("abc")
except ValueError:
    print("上层捕获异常")
```

### 10.5.3 异常链 (Exception Chaining)

```python
# raise ... from ... 语法（Python 3）
def load_config(filename):
    try:
        with open(filename) as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise RuntimeError("配置文件加载失败") from e

# 异常会保留原始异常信息
try:
    load_config("missing.json")
except RuntimeError as e:
    print(f"错误: {e}")
    print(f"原因: {e.__cause__}")  # 访问原始异常
```

---

## 10.6 自定义异常

### 10.6.1 简单自定义异常

```python
class InvalidEmailError(Exception):
    """邮箱格式错误"""
    pass

def validate_email(email):
    if '@' not in email:
        raise InvalidEmailError(f"无效的邮箱: {email}")
    return email

try:
    validate_email("invalid-email")
except InvalidEmailError as e:
    print(f"邮箱验证失败: {e}")
```

### 10.6.2 带额外信息的自定义异常

```python
class ValidationError(Exception):
    """数据验证错误"""
    def __init__(self, field, message):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")

def validate_user(data):
    if 'name' not in data:
        raise ValidationError('name', '姓名是必填项')
    if len(data['name']) < 2:
        raise ValidationError('name', '姓名至少2个字符')
    return data

try:
    validate_user({'name': 'A'})
except ValidationError as e:
    print(f"字段 {e.field} 验证失败: {e.message}")
```

### 10.6.3 异常层次结构设计

```python
# 业务异常基类
class BusinessError(Exception):
    """业务逻辑错误基类"""
    pass

class AuthenticationError(BusinessError):
    """认证失败"""
    pass

class AuthorizationError(BusinessError):
    """权限不足"""
    pass

class ResourceNotFoundError(BusinessError):
    """资源不存在"""
    pass

# 使用
def access_resource(user, resource_id):
    if not user.is_authenticated:
        raise AuthenticationError("用户未登录")
    if not user.has_permission(resource_id):
        raise AuthorizationError("权限不足")
    # ...
    
try:
    access_resource(user, 123)
except BusinessError as e:  # 可以统一捕获所有业务异常
    print(f"业务错误: {e}")
```

---

## 10.7 异常处理最佳实践

### 10.7.1 EAFP vs LBYL：两种编程哲学

在错误处理领域，存在两种截然不同的编程哲学，它们反映了对程序流程控制的不同思考方式。

**LBYL: Look Before You Leap (三思而后行)**

这是传统编程语言（如 C、Java）常用的方式：在执行操作前，先检查所有可能的错误条件。就像过马路前先看看有没有车。

**优点**：
- 逻辑直观，容易理解
- 避免了异常处理的开销

**缺点**：
- 存在**竞态条件** (Race Condition)：检查和操作之间状态可能改变
- 代码冗长，充满 if-else 检查
- 防御性编程可能导致过度检查

**EAFP: Easier to Ask for Forgiveness than Permission (请求原谅比请求许可更容易)**

这是 Python 推崇的方式：直接尝试操作，如果失败则处理异常。就像直接冲进门，如果锁着就说声抱歉。

**优点**：
- 避免竞态条件：操作是原子的
- 代码简洁，关注正常流程
- 符合 Python 的"鸭子类型"哲学

**缺点**：
- 对习惯 LBYL 的程序员不够直观
- 异常处理有一定性能开销（但通常可忽略）

**为什么 Python 选择 EAFP？**

Python 的设计哲学强调代码的可读性和简洁性。EAFP 让你专注于"快乐路径" (Happy Path)，而不是在代码中到处设置检查点。更重要的是，在多线程或异步环境中，EAFP 能够避免检查和操作之间的时间窗口问题。

```python
# LBYL (Look Before You Leap) - 检查再行动
d = {'key': 'value'}
if 'key' in d:
    print(d['key'])

# EAFP (Easier to Ask for Forgiveness than Permission) - Python 风格
try:
    print(d['key'])
except KeyError:
    pass

# EAFP 的优势：避免竞态条件
# LBYL 示例（可能失败）
# if os.path.exists(filename):  # 检查文件存在
#     with open(filename) as f:  # 此时文件可能已被删除！
#         ...

# EAFP 示例（更安全）
try:
    with open(filename) as f:
        # ...
        pass
except FileNotFoundError:
    pass
```

### 10.7.2 捕获具体异常

```python
# ❌ 不好：捕获所有异常
try:
    # ...
    pass
except Exception:
    pass  # 吞掉所有错误，难以调试

# ✅ 好：捕获具体异常
try:
    value = int(user_input)
except ValueError:
    print("输入必须是数字")
```

### 10.7.3 避免裸 except

```python
# ❌ 非常危险：连 KeyboardInterrupt 都会捕获
try:
    # ...
    pass
except:  # 捕获所有异常包括 SystemExit
    pass

# ✅ 如果必须捕获所有异常，明确指定 Exception
try:
    # ...
    pass
except Exception as e:
    log.error(f"错误: {e}")
    raise  # 记录后重新抛出
```

### 10.7.4 不要滥用异常

```python
# ❌ 不好：用异常控制流程
def find_index_bad(lst, value):
    try:
        return lst.index(value)
    except ValueError:
        return -1

# ✅ 好：直接使用条件判断
def find_index_good(lst, value):
    return lst.index(value) if value in lst else -1

# 异常应该用于异常情况，不应用于正常控制流
```

---

## 10.8 上下文管理器 (Context Manager)

### 10.8.1 with 语句

```python
# 传统方式：手动清理资源
f = open('file.txt')
try:
    data = f.read()
finally:
    f.close()

# 使用 with（自动清理）
with open('file.txt') as f:
    data = f.read()
# 文件自动关闭，即使发生异常
```

### 10.8.2 自定义上下文管理器

```python
class Timer:
    """计时器上下文管理器"""
    def __enter__(self):
        import time
        self.start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end = time.time()
        print(f"执行时间: {self.end - self.start:.4f}秒")
        return False  # 不压制异常

# 使用
with Timer():
    import time
    time.sleep(1)
    print("执行中...")

# 输出:
# 执行中...
# 执行时间: 1.0001秒
```

### 10.8.3 使用 @contextmanager 装饰器

```python
from contextlib import contextmanager

@contextmanager
def temporary_directory():
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    print(f"创建临时目录: {temp_dir}")
    
    try:
        yield temp_dir  # 提供给 with 语句
    finally:
        shutil.rmtree(temp_dir)
        print(f"删除临时目录: {temp_dir}")

# 使用
with temporary_directory() as temp_dir:
    print(f"使用临时目录: {temp_dir}")
    # 在这里创建临时文件...
```

### 10.8.4 多个上下文管理器

```python
# Python 3.10+ 支持括号分组
with (
    open('input.txt') as input_file,
    open('output.txt', 'w') as output_file
):
    output_file.write(input_file.read())
```

---

## 10.9 实战案例

### 案例 1: 健壮的配置加载器

```python
import json
from pathlib import Path

class ConfigurationError(Exception):
    """配置错误"""
    pass

def load_config(config_file):
    """加载配置文件"""
    path = Path(config_file)
    
    try:
        if not path.exists():
            raise ConfigurationError(f"配置文件不存在: {config_file}")
        
        with path.open() as f:
            config = json.load(f)
        
        # 验证必需字段
        required_fields = ['database', 'api_key']
        for field in required_fields:
            if field not in config:
                raise ConfigurationError(f"缺少必需字段: {field}")
        
        return config
    
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"JSON 格式错误: {e}") from e
    except PermissionError:
        raise ConfigurationError(f"无权限读取文件: {config_file}")

# 使用
try:
    config = load_config('config.json')
except ConfigurationError as e:
    print(f"配置加载失败: {e}")
    # 使用默认配置或退出程序
```

### 案例 2: 重试装饰器

```python
import time
from functools import wraps

def retry(max_attempts=3, delay=1, exceptions=(Exception,)):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        raise
                    print(f"尝试 {attempt}/{max_attempts} 失败: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=2, exceptions=(ConnectionError, TimeoutError))
def fetch_data(url):
    """模拟网络请求"""
    import random
    if random.random() < 0.7:
        raise ConnectionError("网络连接失败")
    return f"数据来自 {url}"

# fetch_data("https://api.example.com")
```

### 案例 3: 事务管理器

```python
class Transaction:
    """数据库事务上下文管理器"""
    def __init__(self, connection):
        self.connection = connection
    
    def __enter__(self):
        self.connection.begin_transaction()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # 无异常，提交事务
            self.connection.commit()
            print("事务已提交")
        else:
            # 发生异常，回滚事务
            self.connection.rollback()
            print(f"事务已回滚: {exc_val}")
        return False  # 不压制异常

# 使用
# with Transaction(db_connection):
#     db_connection.insert(...)
#     db_connection.update(...)
    # 自动提交或回滚
```

---

## 10.10 调试技巧

### 10.10.1 获取异常堆栈

```python
import traceback

try:
    1 / 0
except Exception:
    # 打印完整堆栈
    traceback.print_exc()
    
    # 获取堆栈字符串
    stack_trace = traceback.format_exc()
    print("堆栈信息:", stack_trace)
```

### 10.10.2 自定义异常钩子

```python
import sys

def custom_exception_handler(exc_type, exc_value, exc_traceback):
    """全局异常处理器"""
    if issubclass(exc_type, KeyboardInterrupt):
        # 不处理 Ctrl+C
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    print("=== 捕获未处理异常 ===")
    print(f"类型: {exc_type.__name__}")
    print(f"信息: {exc_value}")
    traceback.print_tb(exc_traceback)

# 设置全局异常钩子
sys.excepthook = custom_exception_handler

# 测试
# raise ValueError("测试异常")
```

---

## 10.11 本章小结

在本章中，我们深入学习了：

1. ✅ 异常的本质与层次结构
2. ✅ try-except-else-finally 完整语法
3. ✅ 自定义异常类与异常链
4. ✅ 异常处理最佳实践 (EAFP vs LBYL)
5. ✅ 上下文管理器与资源清理
6. ✅ 实战案例：配置加载、重试机制、事务管理

> [!TIP]
> **关键要点**:
> - 捕获具体异常，避免裸 except
> - 使用 with 语句管理资源
> - EAFP 优于 LBYL
> - 异常应该用于异常情况

> **下一步学习建议**:
> - [Chapter 11. 文件与 I/O 操作](11-file-io.md)
> - [Chapter 12. 并发编程](12-concurrency.md)

---

## 参考资源

- [Python 异常层次结构](https://docs.python.org/3/library/exceptions.html#exception-hierarchy)
- [PEP 343: The "with" Statement](https://peps.python.org/pep-0343/)
- [contextlib 标准库](https://docs.python.org/3/library/contextlib.html)

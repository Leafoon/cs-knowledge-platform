---
title: "Chapter 0. Python 基础入门"
description: "从零开始学习 Python：变量、数据类型、运算符、控制流程以及基础数据结构"
updated: "2026-01-22"
---

# Chapter 0. Python 基础入门

> **Learning Objectives**
> * 理解 Python 的设计哲学与应用场景
> * 掌握 Python 基本语法：变量、数据类型、运算符
> * 熟练使用条件语句、循环和基本数据结构
> * 建立良好的代码风格习惯（PEP 8）

---

## 0.1 为什么选择 Python？

### 0.1.1 Python 的设计哲学

Python 遵循 "The Zen of Python"（Python 之禅）的核心理念：

```python
import this
```

运行上述代码，你会看到 Python 的 19 条设计原则，其中最核心的几条：

- **Beautiful is better than ugly** (优美胜于丑陋)
- **Explicit is better than implicit** (明确胜于隐晦)
- **Simple is better than complex** (简洁胜于复杂)
- **Readability counts** (可读性很重要)

### 0.1.2 Python 的应用领域

| 领域 | 典型应用 | 主流框架/库 |
|------|---------|------------|
| **Web 开发** | 网站后端、API 服务 | Django, Flask, FastAPI |
| **数据科学** | 数据分析、可视化 | Pandas, Matplotlib, Seaborn |
| **机器学习** | 模型训练、深度学习 | TensorFlow, PyTorch, scikit-learn |
| **自动化脚本** | 运维、测试、爬虫 | Selenium, Scrapy, Requests |
| **科学计算** | 数值计算、仿真 | NumPy, SciPy, SymPy |
| **游戏开发** | 2D 游戏、原型 | Pygame, Panda3D |

---

## 0.2 你的第一个 Python 程序

### 0.2.1 Hello, World!

```python
# 这是注释，解释器会忽略这一行
print("Hello, World!")  # 输出：Hello, World!
```

**代码解析**：
- `print()` 是 Python 的内置函数，用于输出内容到控制台
- 字符串用引号包裹（单引号或双引号均可）
- `#` 开头的是单行注释

### 0.2.2 交互式解释器 (REPL)

打开终端，输入 `python3` 或 `python`，进入交互式模式：

```python
>>> 2 + 3
5
>>> name = "Alice"
>>> f"Hello, {name}!"
'Hello, Alice!'
>>> exit()  # 退出 REPL
```

> [!TIP]
> **REPL** 是 "Read-Eval-Print Loop" 的缩写，非常适合快速测试代码片段。

---

## 0.3 变量与数据类型

### 0.3.1 变量的本质

在 Python 中，变量是**标签**而非容器（详见 [Chapter 2. Python 对象模型与内存机制](02-object-model.md)）。

```python
# 变量赋值
age = 25
name = "Bob"
is_student = True

# 多重赋值
x, y, z = 1, 2, 3

# 交换变量（Python 特色语法）
a, b = 10, 20
a, b = b, a  # 交换后：a=20, b=10
```

### 0.3.2 基本数据类型

#### 1. 数值类型

```python
# 整数 (int) - 无限精度
big_number = 123456789012345678901234567890
print(type(big_number))  # <class 'int'>

# 浮点数 (float) - 双精度
pi = 3.14159
scientific = 1.5e-3  # 科学记数法: 0.0015

# 复数 (complex)
z = 3 + 4j
print(z.real, z.imag)  # 3.0 4.0
```

#### 2. 字符串 (str)

```python
# 单引号和双引号等价
s1 = 'Hello'
s2 = "World"

# 三引号：多行字符串
multiline = """
这是一个
多行字符串
"""

# 字符串操作
greeting = s1 + " " + s2  # 拼接: "Hello World"
repeated = "Ha" * 3       # 重复: "HaHaHa"
length = len(greeting)    # 长度: 11

# 字符串索引和切片
s = "Python"
print(s[0])      # 'P' (索引从 0 开始)
print(s[-1])     # 'n' (负索引从末尾开始)
print(s[0:3])    # 'Pyt' (切片: [起始:结束))
print(s[::2])    # 'Pto' (步长为 2)
```

#### 3. 布尔值 (bool)

```python
is_valid = True
is_empty = False

# 布尔运算
print(True and False)  # False
print(True or False)   # True
print(not True)        # False

# 真值测试：以下值被视为 False
print(bool(0))         # False
print(bool(""))        # False (空字符串)
print(bool([]))        # False (空列表)
print(bool(None))      # False
```

#### 4. None 类型

```python
# None 表示"无值"或"空值"
result = None

# 常用于初始化变量或函数默认返回值
def do_nothing():
    pass  # 不做任何事

print(do_nothing())  # None
```

### 0.3.3 类型转换

```python
# 显式类型转换
x = int("42")       # 字符串 -> 整数: 42
y = float("3.14")   # 字符串 -> 浮点数: 3.14
s = str(100)        # 整数 -> 字符串: "100"

# 自动类型转换（隐式）
result = 3 + 2.5    # int + float -> float: 5.5
```

---

## 0.4 运算符

### 0.4.1 算术运算符

```python
a, b = 10, 3

print(a + b)   # 加法: 13
print(a - b)   # 减法: 7
print(a * b)   # 乘法: 30
print(a / b)   # 除法: 3.3333... (总是返回浮点数)
print(a // b)  # 整除: 3 (向下取整)
print(a % b)   # 取模: 1 (余数)
print(a ** b)  # 幂运算: 1000 (10的3次方)
```

### 0.4.2 比较运算符

```python
x, y = 5, 10

print(x == y)  # 等于: False
print(x != y)  # 不等于: True
print(x > y)   # 大于: False
print(x < y)   # 小于: True
print(x >= 5)  # 大于等于: True
print(x <= 5)  # 小于等于: True

# 链式比较（Python 特色）
age = 25
print(18 <= age < 60)  # True
```

### 0.4.3 逻辑运算符

```python
# and, or, not
print(True and True)   # True
print(True and False)  # False
print(True or False)   # True
print(not True)        # False

# 短路求值
x = 0
result = (x != 0) and (10 / x)  # 不会执行 10/x，避免除零错误
```

### 0.4.4 成员运算符

```python
# in, not in
numbers = [1, 2, 3, 4, 5]
print(3 in numbers)      # True
print(10 not in numbers) # True

text = "Hello World"
print("World" in text)   # True
```

---

## 0.5 控制流程

### 0.5.1 条件语句 (if-elif-else)

```python
# 基本 if 语句
age = 18
if age >= 18:
    print("成年人")

# if-else
score = 75
if score >= 60:
    print("及格")
else:
    print("不及格")

# if-elif-else（多条件判断）
score = 85
if score >= 90:
    grade = 'A'
elif score >= 80:
    grade = 'B'
elif score >= 70:
    grade = 'C'
elif score >= 60:
    grade = 'D'
else:
    grade = 'F'
print(f"成绩等级: {grade}")

# 三元表达式（简洁写法）
age = 20
status = "成年人" if age >= 18 else "未成年人"
```

### 0.5.2 循环语句

#### while 循环

```python
# 基本 while 循环
count = 0
while count < 5:
    print(f"计数: {count}")
    count += 1

# 无限循环（需要手动 break）
while True:
    user_input = input("输入 'quit' 退出: ")
    if user_input == 'quit':
        break
    print(f"你输入了: {user_input}")
```

#### for 循环

```python
# 遍历列表
fruits = ['apple', 'banana', 'cherry']
for fruit in fruits:
    print(fruit)

# 遍历字符串
for char in "Python":
    print(char)

# range() 函数
for i in range(5):      # 0, 1, 2, 3, 4
    print(i)

for i in range(2, 8):   # 2, 3, 4, 5, 6, 7
    print(i)

for i in range(0, 10, 2):  # 0, 2, 4, 6, 8 (步长为2)
    print(i)

# enumerate() - 获取索引和值
fruits = ['apple', 'banana', 'cherry']
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
# 输出:
# 0: apple
# 1: banana
# 2: cherry
```

### 0.5.3 循环控制语句

```python
# break: 跳出整个循环
for i in range(10):
    if i == 5:
        break
    print(i)  # 输出 0-4

# continue: 跳过当前迭代，继续下一次
for i in range(5):
    if i == 2:
        continue
    print(i)  # 输出 0, 1, 3, 4

# else: 循环正常结束后执行（没有 break）
for i in range(5):
    print(i)
else:
    print("循环正常结束")

# 如果有 break，else 不执行
for i in range(5):
    if i == 3:
        break
    print(i)
else:
    print("这句不会执行")
```

---

## 0.6 基础数据结构

### 0.6.1 列表 (List)

```python
# 创建列表
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]  # 可以混合类型
empty = []

# 访问元素
print(numbers[0])   # 1
print(numbers[-1])  # 5 (最后一个)

# 切片
print(numbers[1:4])  # [2, 3, 4]
print(numbers[:3])   # [1, 2, 3]
print(numbers[3:])   # [4, 5]

# 修改元素
numbers[0] = 10
print(numbers)  # [10, 2, 3, 4, 5]

# 常用方法
numbers.append(6)        # 末尾添加: [10, 2, 3, 4, 5, 6]
numbers.insert(0, 0)     # 指定位置插入: [0, 10, 2, 3, 4, 5, 6]
numbers.remove(10)       # 删除第一个匹配项: [0, 2, 3, 4, 5, 6]
popped = numbers.pop()   # 删除并返回最后一个: 6
numbers.extend([7, 8])   # 扩展列表: [0, 2, 3, 4, 5, 7, 8]
numbers.sort()           # 排序: [0, 2, 3, 4, 5, 7, 8]
numbers.reverse()        # 反转: [8, 7, 5, 4, 3, 2, 0]

# 列表推导式（高效创建列表）
squares = [x**2 for x in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# 带条件的列表推导式
evens = [x for x in range(10) if x % 2 == 0]
print(evens)  # [0, 2, 4, 6, 8]
```

### 0.6.2 元组 (Tuple)

```python
# 创建元组（不可变）
point = (3, 4)
single = (5,)  # 单元素元组需要逗号
empty = ()

# 访问元素（与列表相同）
print(point[0])  # 3

# 元组解包
x, y = point
print(x, y)  # 3 4

# 多返回值（实际上返回的是元组）
def get_coordinates():
    return 10, 20

x, y = get_coordinates()
```

### 0.6.3 字典 (Dict)

```python
# 创建字典
person = {
    "name": "Alice",
    "age": 25,
    "city": "Beijing"
}

# 访问元素
print(person["name"])        # Alice
print(person.get("age"))     # 25
print(person.get("job", "未知"))  # 未知 (键不存在返回默认值)

# 修改和添加
person["age"] = 26           # 修改
person["job"] = "Engineer"   # 添加新键值对

# 删除
del person["city"]
removed = person.pop("job")

# 遍历
for key in person:
    print(key, person[key])

for key, value in person.items():
    print(f"{key}: {value}")

# 字典推导式
squares = {x: x**2 for x in range(5)}
print(squares)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

### 0.6.4 集合 (Set)

```python
# 创建集合（无序、不重复）
fruits = {"apple", "banana", "cherry"}
numbers = {1, 2, 3, 3, 3}  # 自动去重: {1, 2, 3}

# 集合运算
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

print(a | b)  # 并集: {1, 2, 3, 4, 5, 6}
print(a & b)  # 交集: {3, 4}
print(a - b)  # 差集: {1, 2}
print(a ^ b)  # 对称差: {1, 2, 5, 6}

# 常用方法
fruits.add("orange")        # 添加
fruits.remove("banana")     # 删除（不存在会报错）
fruits.discard("grape")     # 删除（不存在不报错）
```

---

## 0.7 输入输出

### 0.7.1 标准输入

```python
# input() 函数总是返回字符串
name = input("请输入你的名字: ")
print(f"你好, {name}!")

# 类型转换
age = int(input("请输入你的年龄: "))
height = float(input("请输入你的身高(米): "))
```

### 0.7.2 格式化输出

```python
name = "Alice"
age = 25
score = 95.5

# 1. f-string (推荐, Python 3.6+)
print(f"姓名: {name}, 年龄: {age}")
print(f"分数: {score:.1f}")  # 保留1位小数: 95.5

# 2. str.format()
print("姓名: {}, 年龄: {}".format(name, age))
print("姓名: {n}, 年龄: {a}".format(n=name, a=age))

# 3. % 格式化（旧式，不推荐）
print("姓名: %s, 年龄: %d" % (name, age))

# f-string 高级用法
width = 10
print(f"|{name:^{width}}|")  # 居中对齐: |  Alice   |
print(f"{1234567:,}")        # 千位分隔符: 1,234,567
```

---

## 0.8 代码风格 (PEP 8)

### 0.8.1 命名规范

```python
# 1. 变量和函数：小写+下划线
user_name = "Alice"
def calculate_sum(a, b):
    return a + b

# 2. 类名：大驼峰（每个单词首字母大写）
class UserAccount:
    pass

# 3. 常量：全大写+下划线
MAX_SIZE = 100
PI = 3.14159

# 4. 私有变量/方法：单下划线开头
_internal_value = 42

# 5. 避免使用的名字
l = 1  # ❌ 小写L容易与1混淆
O = 0  # ❌ 大写O容易与0混淆
```

### 0.8.2 代码布局

```python
# 缩进：4个空格（不要用Tab）
def example():
    if True:
        print("正确的缩进")

# 每行最多79个字符
long_text = (
    "这是一个非常长的字符串，"
    "我们将它分成多行以保持"
    "代码的可读性。"
)

# 运算符两侧加空格
x = 1 + 2
result = (a + b) * (c - d)

# 函数定义前后空两行
def function1():
    pass


def function2():
    pass

# 类定义前后空两行
class MyClass:
    pass
```

---

## 0.9 实战练习

### 练习 1: 简单计算器

```python
def calculator():
    """实现一个简单的四则运算计算器"""
    num1 = float(input("输入第一个数字: "))
    operator = input("输入运算符 (+, -, *, /): ")
    num2 = float(input("输入第二个数字: "))
    
    if operator == '+':
        result = num1 + num2
    elif operator == '-':
        result = num1 - num2
    elif operator == '*':
        result = num1 * num2
    elif operator == '/':
        if num2 != 0:
            result = num1 / num2
        else:
            return "错误: 除数不能为0"
    else:
        return "错误: 无效的运算符"
    
    return f"{num1} {operator} {num2} = {result}"

# calculator()  # 取消注释运行
```

### 练习 2: 猜数字游戏

```python
import random

def guess_number():
    """猜数字游戏：计算机随机生成1-100的数字，用户猜"""
    target = random.randint(1, 100)
    attempts = 0
    max_attempts = 7
    
    print("我想了一个1到100之间的数字，你有7次机会猜！")
    
    while attempts < max_attempts:
        attempts += 1
        guess = int(input(f"第{attempts}次猜测: "))
        
        if guess == target:
            print(f"恭喜你！用了{attempts}次猜对了！")
            return
        elif guess < target:
            print("太小了！")
        else:
            print("太大了！")
    
    print(f"游戏结束！答案是{target}")

# guess_number()  # 取消注释运行
```

### 练习 3: 统计字符频率

```python
def count_characters(text):
    """统计字符串中每个字符出现的次数"""
    freq = {}
    for char in text:
        if char in freq:
            freq[char] += 1
        else:
            freq[char] = 1
    
    # 或使用更简洁的写法
    # freq = {}
    # for char in text:
    #     freq[char] = freq.get(char, 0) + 1
    
    return freq

# 测试
text = "hello world"
result = count_characters(text)
for char, count in sorted(result.items()):
    print(f"'{char}': {count}")
```

---

## 0.10 本章小结

在本章中，我们学习了：

1. ✅ Python 的基本数据类型：int, float, str, bool, None
2. ✅ 运算符：算术、比较、逻辑、成员运算符
3. ✅ 控制流程：if-elif-else, for, while, break, continue
4. ✅ 基础数据结构：list, tuple, dict, set
5. ✅ 输入输出：input(), print(), f-string 格式化
6. ✅ 代码风格：PEP 8 命名规范和布局建议

这些是 Python 编程的基石。掌握这些基础后，你就可以继续深入学习更高级的主题了。

> [!TIP]
> **下一步学习建议**:
> - [Chapter 1. Python 解释器与运行环境](01-interpreter-and-env.md) - 深入理解 Python 执行流水线
> - [Chapter 2. Python 对象模型与内存机制](02-object-model.md) - 理解变量的本质
> - 练习、练习、再练习！编程是一门实践的艺术

---

## 参考资源

- [Python 官方文档](https://docs.python.org/zh-cn/3/)
- [PEP 8 风格指南](https://peps.python.org/pep-0008/)
- [Python Tutorial for Beginners](https://www.python.org/about/gettingstarted/)

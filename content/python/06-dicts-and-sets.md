---
title: "Chapter 6. 字典与集合：哈希表的力量"
description: "深入剖析 Python 字典和集合的哈希表实现、冲突解决策略、性能优化以及高级用法"
updated: "2026-01-22"
---

# Chapter 6. 字典与集合：哈希表的力量

> **Learning Objectives**
> * 理解哈希表 (Hash Table) 的核心原理与 Python 实现
> * 掌握字典的底层结构：`PyDictObject` 与开放寻址法
> * 深入哈希冲突解决机制与负载因子 (Load Factor)
> * 理解集合 (Set) 与字典的关系及性能特性
> * 掌握字典和集合的高级用法与最佳实践

字典 (`dict`) 和集合 (`set`) 是 Python 中最强大的数据结构之一。它们的底层都基于**哈希表**实现，提供 O(1) 平均时间复杂度的查找、插入和删除操作。

---

## 6.1 哈希表基础：从理论到实现

### 6.1.1 什么是哈希表？

想象一个图书馆，如果书籍随意摆放，你要找一本书就需要逐个书架查找，这是线性查找，效率很低。但如果图书馆采用分类系统：通过书名的首字母直接定位到对应书架，查找速度就会大大提升。**哈希表**正是基于这种思想设计的数据结构。

**哈希表** (Hash Table) 也称为散列表，是一种实现**关联数组** (Associative Array) 的数据结构。它的核心思想是：通过**哈希函数** (Hash Function) 将键 (Key) 转换为数组索引 (Index)，从而实现快速的数据存储和检索。

**基本工作流程**：
```
Key → Hash Function → Index → Value
        (计算)      (定位)   (存取)
```

这种设计使得哈希表的查找、插入和删除操作在理想情况下都可以达到 **O(1)** 的时间复杂度，这是相比链表 O(n) 和二叉搜索树 O(log n) 的巨大优势。正因如此，哈希表成为了现代编程中最常用的数据结构之一。

#### 核心概念

1. **哈希函数 (Hash Function)**:
   - 将任意类型的键转换为整数（哈希值）
   - Python 中使用 `hash()` 函数
   
2. **桶 (Bucket)**:
   - 哈希表内部的数组槽位
   - 通过 `hash(key) % table_size` 计算索引

3. **哈希冲突 (Collision)**:
   - 不同的键映射到同一索引
   - 需要冲突解决机制

<div data-component="HashTableVisualizer"></div>

```python
# 哈希函数示例
print(hash("apple"))   # 每次运行可能不同 (哈希随机化)
print(hash(42))        # 整数的哈希值通常是自身
print(hash((1, 2)))    # 元组可哈希
# print(hash([1, 2]))  # ❌ TypeError: 列表不可哈希
```

### 6.1.2 可哈希性 (Hashable)

在深入哈希表的实现之前，我们需要理解一个关键概念：**可哈希性**。

**为什么只有不可变对象才能作为字典的键？**

想象这样一个场景：你用一个列表 `[1, 2, 3]` 作为字典的键，存储了一个值。之后你修改了这个列表，变成 `[1, 2, 3, 4]`。此时问题来了：
1. 列表内容改变后，它的哈希值也会改变
2. 哈希值改变意味着它在哈希表中的存储位置也变了
3. 你将无法再通过原来的键找到之前存储的值

这就是为什么 Python 要求：**只有不可变对象才可作为字典的键或集合的元素**。不可变对象的哈希值在其生命周期内保持不变，确保了哈希表的正确性。

| 类型 | 可哈希？ | 原因 |
|------|---------|------|
| `int`, `float`, `str` | ✅ | 不可变 |
| `tuple` | ✅ (如果元素都可哈希) | 不可变 |
| `list`, `dict`, `set` | ❌ | 可变 |
| 自定义类 | ✅ (默认) | 可覆盖 `__hash__()` |

```python
# 自定义可哈希类
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

p1 = Point(1, 2)
p2 = Point(1, 2)
print(hash(p1) == hash(p2))  # True

# 可作为字典键
points = {p1: "Origin"}
print(points[p2])  # "Origin"
```

> [!IMPORTANT]
> 如果自定义 `__eq__()`，必须同时自定义 `__hash__()`，且满足：
> `a == b` ⇒ `hash(a) == hash(b)`

---

## 6.2 字典的底层实现

### 6.2.1 PyDictObject 结构

理解 Python 字典的底层实现，对于编写高性能代码至关重要。Python 字典的实现经历了多次优化，每次优化都在平衡内存占用和访问速度。

**历史演进**：
- **Python 3.5 及之前**：字典使用传统的稀疏哈希表，每个槽位存储完整的键值对信息。这种设计简单直观，但存在严重的**内存浪费**问题——为了减少哈希冲突，需要保持较低的负载因子（通常是 2/3），这意味着 1/3 的空间是空置的。

- **Python 3.6 引入的革命性改进**：Raymond Hettinger 提出了**紧凑哈希表** (Compact Hash Table) 设计。这个改进不仅减少了 20-25% 的内存占用，还带来了一个意外的好处——字典能够保持**插入顺序**！这个特性在 Python 3.7 中被正式纳入语言规范。

**现代 Python 字典的双数组结构**：

在 CPython 3.6+ 中，字典采用**紧凑哈希表**实现，巧妙地将存储分为两个部分：

```c
typedef struct {
    PyObject_HEAD
    Py_ssize_t ma_used;      // 已使用的键值对数量
    PyDictKeysObject *ma_keys;  // 键表（共享）
    PyObject **ma_values;    // 值数组
} PyDictObject;
```

#### 存储结构演进

**旧版 (Python 3.5-)**: 稀疏数组，浪费内存

```
index:  0     1     2     3     4     5
       [-, -, Key1, -, Key2, -]
```

**新版 (Python 3.6+)**: 紧凑数组 + 索引表

```
indices: [-, 0, -, 1, -, 2]  # 稀疏索引表
entries: [                    # 紧凑键值对数组
    (hash1, key1, value1),
    (hash2, key2, value2),
    (hash3, key3, value3)
]
```

**优势**：
- 内存占用减少 20-25%
- 保持插入顺序（Python 3.7+ 正式特性）
- 更好的缓存局部性

### 6.2.2 开放寻址法 (Open Addressing)

Python 使用**探测序列**解决哈希冲突，而非链表法。下面的可视化展示了插入、查找和删除操作时如何探测哈希槽位：

<div data-component="OpenAddressingVisualizer"></div>

```python
# 伪代码：查找键的过程
def lookup(key):
    i = hash(key) % table_size
    perturb = hash(key)
    
    while True:
        entry = table[i]
        if entry is EMPTY:
            return KEY_NOT_FOUND
        if entry.hash == hash(key) and entry.key == key:
            return entry.value
        
        # 探测下一个位置（伪随机序列）
        i = (5 * i + perturb + 1) % table_size
        perturb >>= 5
```

**探测公式**：

$$
i = (5i + 1 + \text{perturb}) \mod \text{size}
$$

其中 $\text{perturb}$ 初始值为 `hash(key)`，每次右移 5 位。

### 6.2.3 动态扩容与负载因子

当负载因子 (used / size) 超过 **2/3** 时，字典会扩容（通常变为原来的 2 倍）。

```python
import sys

# 观察字典的内存增长
d = {}
for i in range(100):
    d[i] = i
    if i in [0, 5, 10, 20, 40, 80]:
        print(f"元素数: {i+1}, 大小: {sys.getsizeof(d)} bytes")

# 输出示例:
# 元素数: 1, 大小: 232 bytes
# 元素数: 6, 大小: 360 bytes   (首次扩容)
# 元素数: 11, 大小: 360 bytes
# 元素数: 21, 大小: 648 bytes  (第二次扩容)
# ...
```

---

## 6.3 字典的基本操作

### 6.3.1 创建字典

```python
# 1. 字面量语法
person = {"name": "Alice", "age": 25}

# 2. dict() 构造函数
person = dict(name="Alice", age=25)

# 3. 从键值对列表创建
pairs = [("name", "Alice"), ("age", 25)]
person = dict(pairs)

# 4. 字典推导式
squares = {x: x**2 for x in range(6)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# 5. fromkeys() - 批量初始化
keys = ["a", "b", "c"]
default_dict = dict.fromkeys(keys, 0)
# {'a': 0, 'b': 0, 'c': 0}
```

### 6.3.2 访问元素

```python
person = {"name": "Alice", "age": 25, "city": "Beijing"}

# 1. 方括号访问（键不存在会抛出 KeyError）
print(person["name"])  # "Alice"
# print(person["job"])  # ❌ KeyError

# 2. get() - 安全访问
print(person.get("name"))        # "Alice"
print(person.get("job"))         # None
print(person.get("job", "未知"))  # "未知"

# 3. setdefault() - 获取或设置默认值
job = person.setdefault("job", "Engineer")
print(person)  # {'name': 'Alice', 'age': 25, 'city': 'Beijing', 'job': 'Engineer'}
```

### 6.3.3 修改和删除

```python
person = {"name": "Alice", "age": 25}

# 修改
person["age"] = 26

# 添加
person["city"] = "Shanghai"

# 删除方法对比
value = person.pop("city")        # 删除并返回值
# person.pop("xxx")                # ❌ KeyError
person.pop("xxx", None)           # 安全删除
del person["age"]                 # 直接删除
person.clear()                    # 清空所有键值对

# popitem() - 删除并返回最后插入的键值对 (LIFO, Python 3.7+)
person = {"a": 1, "b": 2, "c": 3}
last = person.popitem()  # ('c', 3)
```

### 6.3.4 遍历字典

```python
scores = {"Alice": 95, "Bob": 87, "Charlie": 92}

# 1. 遍历键
for name in scores:
    print(name)

# 2. 遍历键（显式）
for name in scores.keys():
    print(name)

# 3. 遍历值
for score in scores.values():
    print(score)

# 4. 遍历键值对
for name, score in scores.items():
    print(f"{name}: {score}")

# 5. 按键排序遍历
for name in sorted(scores):
    print(f"{name}: {scores[name]}")

# 6. 按值排序遍历
for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {score}")
# 输出: Alice: 95, Charlie: 92, Bob: 87
```

---

## 6.4 字典的高级用法

### 6.4.1 默认字典 (defaultdict)

自动为不存在的键提供默认值，避免 KeyError。

```python
from collections import defaultdict

# 统计单词频率
text = "apple banana apple cherry banana apple"
word_count = defaultdict(int)  # 默认值为 0

for word in text.split():
    word_count[word] += 1

print(dict(word_count))  # {'apple': 3, 'banana': 2, 'cherry': 1}

# 分组数据
students = [
    ("Alice", "Math"),
    ("Bob", "Physics"),
    ("Charlie", "Math"),
    ("David", "Physics")
]

groups = defaultdict(list)
for name, subject in students:
    groups[subject].append(name)

print(dict(groups))
# {'Math': ['Alice', 'Charlie'], 'Physics': ['Bob', 'David']}
```

### 6.4.2 有序字典 (OrderedDict)

> [!NOTE]
> Python 3.7+ 普通字典已保持插入顺序，`OrderedDict` 主要用于向后兼容或需要 `move_to_end()` 等额外方法。

```python
from collections import OrderedDict

# 实现 LRU 缓存的简单示例
class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)  # 移到末尾（最近使用）
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # 删除最旧的

# 测试
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))  # 1
cache.put(3, 3)      # 淘汰键 2
print(cache.get(2))  # -1 (已淘汰)
```

### 6.4.3 计数器 (Counter)

专门用于计数的字典子类。

```python
from collections import Counter

# 1. 统计频率
words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
counter = Counter(words)
print(counter)  # Counter({'apple': 3, 'banana': 2, 'cherry': 1})

# 2. 最常见元素
print(counter.most_common(2))  # [('apple', 3), ('banana', 2)]

# 3. 元素相加
c1 = Counter(a=3, b=1)
c2 = Counter(a=1, b=2)
print(c1 + c2)  # Counter({'a': 4, 'b': 3})

# 4. 统计字符串字符
text = "hello world"
char_count = Counter(text)
print(char_count.most_common(3))  # [('l', 3), ('o', 2), ('h', 1)]
```

### 6.4.4 链式映射 (ChainMap)

将多个字典合并为一个逻辑视图。

```python
from collections import ChainMap

# 场景：配置管理（命令行参数 > 环境变量 > 默认配置）
defaults = {"color": "red", "user": "guest"}
environ = {"user": "admin"}
cmdline = {"color": "blue"}

config = ChainMap(cmdline, environ, defaults)
print(config["color"])  # "blue" (从 cmdline)
print(config["user"])   # "admin" (从 environ)
print(config.get("theme", "dark"))  # "dark" (默认值)

# 修改只影响第一个字典
config["color"] = "green"
print(cmdline)  # {'color': 'green'}
```

---

## 6.5 集合 (Set)

### 6.5.1 集合的底层实现

集合本质上是**只有键、没有值的字典**。

```c
// 简化表示
typedef struct {
    PyObject_HEAD
    Py_ssize_t ma_used;
    PyObject **table;  // 只存储键，值为 dummy
} PySetObject;
```

### 6.5.2 基本操作

```python
# 创建集合
fruits = {"apple", "banana", "cherry"}
empty_set = set()  # 注意：{} 是空字典！

# 添加元素
fruits.add("orange")
fruits.update(["mango", "grape"])  # 批量添加

# 删除元素
fruits.remove("banana")    # 不存在会报错
fruits.discard("xxx")      # 安全删除
popped = fruits.pop()      # 随机删除并返回

# 成员测试 - O(1)
print("apple" in fruits)   # True
```

### 6.5.3 集合运算

```python
a = {1, 2, 3, 4, 5}
b = {4, 5, 6, 7, 8}

# 并集 (Union)
print(a | b)           # {1, 2, 3, 4, 5, 6, 7, 8}
print(a.union(b))      # 同上

# 交集 (Intersection)
print(a & b)           # {4, 5}
print(a.intersection(b))

# 差集 (Difference)
print(a - b)           # {1, 2, 3}
print(a.difference(b))

# 对称差 (Symmetric Difference)
print(a ^ b)           # {1, 2, 3, 6, 7, 8}
print(a.symmetric_difference(b))

# 子集/超集判断
c = {1, 2}
print(c < a)           # True (c 是 a 的真子集)
print(c.issubset(a))   # True
print(a > c)           # True (a 是 c 的超集)
```

### 6.5.4 集合推导式

```python
# 提取所有偶数
numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
evens = {x for x in numbers if x % 2 == 0}
print(evens)  # {2, 4, 6, 8, 10}

# 字符串去重（保留唯一字符）
text = "hello"
unique_chars = {char for char in text}
print(unique_chars)  # {'h', 'e', 'l', 'o'}
```

### 6.5.5 不可变集合 (frozenset)

```python
# frozenset 可作为字典的键或集合的元素
s1 = frozenset([1, 2, 3])
s2 = frozenset([2, 3, 4])

# 可哈希
print(hash(s1))  # 可计算哈希值

# 可作为字典键
graph = {
    frozenset([1, 2]): "edge1",
    frozenset([2, 3]): "edge2"
}

# 集合的集合
set_of_sets = {frozenset([1, 2]), frozenset([3, 4])}
```

---

## 6.6 性能分析与优化

### 6.6.1 时间复杂度

| 操作 | 字典 | 集合 | 列表 |
|------|------|------|------|
| 查找 | O(1) | O(1) | O(n) |
| 插入 | O(1)* | O(1)* | O(1) (末尾) / O(n) (头部) |
| 删除 | O(1) | O(1) | O(n) |
| 遍历 | O(n) | O(n) | O(n) |

\* 平摊时间复杂度，最坏情况 O(n) (扩容)

### 6.6.2 实战性能测试

```python
import time

# 测试：在大量数据中查找元素
data_size = 1000000

# 列表查找
lst = list(range(data_size))
start = time.time()
999999 in lst
print(f"列表查找: {time.time() - start:.6f}秒")

# 集合查找
s = set(range(data_size))
start = time.time()
999999 in s
print(f"集合查找: {time.time() - start:.6f}秒")

# 输出示例:
# 列表查找: 0.015000秒
# 集合查找: 0.000001秒  (快 15000 倍！)
```

### 6.6.3 内存优化技巧

```python
# 1. 使用 __slots__ 减少字典开销
class Point:
    __slots__ = ['x', 'y']  # 禁用 __dict__
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 2. 共享键 (Key-Sharing Dictionary)
# 相同结构的字典会共享键表，节省内存
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

p1 = Person("Alice", 25)
p2 = Person("Bob", 30)
# p1.__dict__ 和 p2.__dict__ 共享键表

# 3. 使用生成器而非列表推导式
# ❌ 内存占用大
big_dict = {i: i**2 for i in range(1000000)}

# ✅ 按需生成
def gen_dict(n):
    for i in range(n):
        yield (i, i**2)

for k, v in gen_dict(1000000):
    pass  # 处理
```

---

## 6.7 实战案例

### 案例 1: 两数之和 (Two Sum)

```python
def two_sum(nums, target):
    """
    给定整数数组和目标值，返回两个数的索引，使它们相加等于目标值。
    时间复杂度: O(n), 空间复杂度: O(n)
    """
    seen = {}  # 存储 {值: 索引}
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    
    return []

# 测试
print(two_sum([2, 7, 11, 15], 9))  # [0, 1]
print(two_sum([3, 2, 4], 6))       # [1, 2]
```

### 案例 2: 字母异位词分组

```python
from collections import defaultdict

def group_anagrams(words):
    """
    将字母异位词分组
    例如: ["eat", "tea", "tan", "ate", "nat", "bat"]
    输出: [["eat","tea","ate"], ["tan","nat"], ["bat"]]
    """
    groups = defaultdict(list)
    
    for word in words:
        # 排序后的字符串作为键
        key = ''.join(sorted(word))
        groups[key].append(word)
    
    return list(groups.values())

# 测试
words = ["eat", "tea", "tan", "ate", "nat", "bat"]
print(group_anagrams(words))
# [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
```

### 案例 3: LRU 缓存实现

```python
from collections import OrderedDict

class LRUCache:
    """最近最少使用缓存"""
    
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return -1
        # 移到末尾（标记为最近使用）
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        # 超出容量，删除最久未使用的
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

# 测试
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))    # 1
cache.put(3, 3)        # 淘汰键 2
print(cache.get(2))    # -1
cache.put(4, 4)        # 淘汰键 1
print(cache.get(1))    # -1
print(cache.get(3))    # 3
print(cache.get(4))    # 4
```

### 案例 4: 拓扑排序（课程表问题）

```python
from collections import defaultdict, deque

def can_finish(num_courses, prerequisites):
    """
    判断能否完成所有课程（检测有向图是否有环）
    例如: numCourses = 2, prerequisites = [[1,0]]
    表示学习课程 1 前必须先学习课程 0
    """
    # 构建邻接表和入度表
    graph = defaultdict(list)
    in_degree = [0] * num_courses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    # BFS 拓扑排序
    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    completed = 0
    
    while queue:
        course = queue.popleft()
        completed += 1
        
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)
    
    return completed == num_courses

# 测试
print(can_finish(2, [[1, 0]]))        # True
print(can_finish(2, [[1, 0], [0, 1]]))  # False (循环依赖)
```

---

## 6.8 最佳实践与陷阱

### 6.8.1 字典键的选择

```python
# ✅ 好的实践
good_dict = {
    "name": "Alice",
    42: "answer",
    (1, 2): "point"
}

# ❌ 常见错误
# bad_dict = {[1, 2]: "list"}  # TypeError: 列表不可哈希
# bad_dict = {{1, 2}: "set"}   # TypeError: 集合不可哈希
```

### 6.8.2 默认值陷阱

```python
# ❌ 危险：可变默认值
def add_item(item, items={}):  # 所有调用共享同一字典！
    items[item] = True
    return items

print(add_item("apple"))   # {'apple': True}
print(add_item("banana"))  # {'apple': True, 'banana': True} ❌

# ✅ 正确做法
def add_item(item, items=None):
    if items is None:
        items = {}
    items[item] = True
    return items
```

### 6.8.3 字典视图的动态性

```python
d = {"a": 1, "b": 2}
keys = d.keys()  # 视图对象，不是列表
print(keys)      # dict_keys(['a', 'b'])

# 视图会随字典变化
d["c"] = 3
print(keys)      # dict_keys(['a', 'b', 'c'])

# 如需固定快照，转为列表
keys_snapshot = list(d.keys())
```

---

## 6.9 本章小结

在本章中，我们深入学习了：

1. ✅ 哈希表原理：哈希函数、冲突解决、负载因子
2. ✅ 字典底层实现：紧凑哈希表、开放寻址法、动态扩容
3. ✅ 字典操作：创建、访问、修改、遍历
4. ✅ 高级容器：`defaultdict`, `OrderedDict`, `Counter`, `ChainMap`
5. ✅ 集合运算：并集、交集、差集、对称差
6. ✅ 性能优化：时间复杂度分析、内存优化技巧
7. ✅ 实战案例：LRU 缓存、拓扑排序、字母异位词分组

> [!TIP]
> **下一步学习建议**:
> - [Chapter 7. 函数与作用域深度解析](07-functions-and-scopes.md) - 深入理解函数机制
> - [Chapter 11. 迭代器与生成器](11-iterators-and-generators.md) - 惰性计算的艺术
> - 实战练习：LeetCode 哈希表标签题目

---

## 参考资源

- [PEP 412: Key-Sharing Dictionary](https://peps.python.org/pep-0412/)
- [Python's Dictionary Implementation](https://docs.python.org/3/faq/design.html#how-are-dictionaries-implemented-in-cpython)
- [Time Complexity of Python Operations](https://wiki.python.org/moin/TimeComplexity)

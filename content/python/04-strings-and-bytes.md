---
title: "Chapter 4. 字符串与编码：Unicode 的世界"
description: "深入剖析 Python 字符串的 Unicode 模型、UTF-8/16/32 编码机制、字符串驻留优化以及 bytes 类型"
updated: "2024-03-22"
---

# Chapter 4. 字符串与编码：Unicode 的世界

> **Learning Objectives**
> *   理解 Unicode 与字符编码的本质区别（字符集 vs 编码方案）
> *   掌握 UTF-8, UTF-16, UTF-32 的编码规则与性能差异
> *   深入字符串驻留 (String Interning) 机制及其应用场景
> *   区分 `str` (文本) 与 `bytes` (二进制数据) 的边界
> *   理解 f-string 的编译期优化原理

## 4.0 引言：巴别塔的倒塌

在 Python 2 的时代，字符串处理简直是开发者的噩梦：`str` 和 `unicode` 混用导致 `UnicodeDecodeError` 满天飞。这就像一栋大楼里的人说着不同的方言，沟通完全依靠猜测。

Python 3 做出了一个艰难但正确的决定：**彻底重构字符串模型**。现在，所有的字符串 (`str`) 在内存中都统一是 **Unicode**。无论你输入的是中文、英文还是 emoji，它们在 Python 眼中地位平等。

但要真正理解字符串，我们必须区分两个核心概念：**你是谁（字符）** 和 **你怎么存（字节）**。

---

## 4.1 Unicode：字符的唯一身份证

### 4.1.1 什么是 Unicode？

**Unicode** 不是一种编码格式，它更像是一个**巨大的字典**或**字符集 (Character Set)**。它的目标是给全世界乃至包括外星符号在内的所有字符，都分配一个唯一的编号，称为 **码点 (Code Point)**。

*   **抽象的存在**：Unicode 码点（如 `U+4E2D`）是字符的灵魂，是抽象的 ID。
*   **物理的存在**：字节（如 `\xe4\xb8\xad`）是字符在磁盘上的肉身，是具体的存储形式。

想象一下：
*   **字符 'A'** 就像是柏拉图理念世界中的"完美的A"。
*   **码点 65** 是它在 Unicode 字典里的页码。
*   **二进制 01000001** 是我们为了把这个页码写在硬盘上而发明的一种墨水记录方式。

*   格式：`U+XXXX` (十六进制表示)
*   范围：`U+0000` ~ `U+10FFFF` (超过 100 万个码点)

```python
ord('A')    # 65 (U+0041)
ord('中')   # 20013 (U+4E2D)
ord('🐍')   # 128013 (U+1F40D)
```

### 4.1.2 码点 ≠ 字节

码点只是"抽象的编号"，要存储到磁盘或网络传输，必须进行**编码 (Encoding)**。编码就是将抽象的数字转换为具体的字节序列的**算法**。

<div data-component="UnicodeEncodingVisualizer"></div>

---

## 4.2 编码方案：UTF-8 / UTF-16 / UTF-32

### 4.2.1 UTF-8: 变长编码 (1-4 字节)

UTF-8 是**最流行的编码**，因为它对 ASCII 完全兼容：

*   `U+0000` ~ `U+007F` (ASCII): 1 字节
*   `U+0080` ~ `U+07FF`: 2 字节
*   `U+0800` ~ `U+FFFF`: 3 字节 (包含大部分中文)
*   `U+10000` ~ `U+10FFFF`: 4 字节 (Emoji, 生僻字)

```python
"Hello".encode('utf-8')    # b'Hello' (5 bytes)
"你好".encode('utf-8')      # b'\xe4\xbd\xa0\xe5\xa5\xbd' (6 bytes, 每个汉字3字节)
"🐍".encode('utf-8')        # b'\xf0\x9f\x90\x8d' (4 bytes)
```

> [!TIP]
> **为什么 UTF-8 统治互联网？**
> 1. ASCII 兼容性：英文网站无需额外空间
> 2. 节省带宽：中文网站比 UTF-16 小 33%
> 3. 无字节序问题：UTF-8 无需 BOM (Byte Order Mark)

### 4.2.2 UTF-16: Java 与 Windows 的选择

UTF-16 使用 2 或 4 字节：

*   `U+0000` ~ `U+FFFF` (BMP 基本多文种平面): 2 字节
*   `U+10000` ~ `U+10FFFF`: 4 字节 (代理对机制)

```python
"Hello".encode('utf-16')   # b'\xff\xfeH\x00e\x00l\x00l\x00o\x00' (12 bytes，含BOM)
"你好".encode('utf-16')     # b'\xff\xfe`O}Y' (8 bytes)
```

### 4.2.3 UTF-32: 简单但浪费

UTF-32 固定每个字符 4 字节：

*   优点：随机访问 O(1)，无需解码即可索引
*   缺点：空间浪费严重（英文文本膨胀 4 倍）

```python
"Hello".encode('utf-32')   # b'\xff\xfe\x00\x00H\x00\x00\x00...' (24 bytes!)
```

---

## 4.3 字符串驻留 (String Interning)

### 4.3.1 什么是驻留？

**字符串驻留**是 Python 的优化机制：对于相同内容的字符串，解释器会让多个变量共享同一内存地址。

```python
a = "hello"
b = "hello"
print(a is b)  # True (指向同一对象)
```

### 4.3.2 驻留的规则

Python **自动驻留**以下字符串：

1.  **编译时常量** (代码中直接写出的字符串字面量)
2.  **标识符风格字符串** (仅包含字母、数字、下划线)

但以下情况**不会驻留**：

```python
a = "hello world"  # 包含空格，不驻留
b = "hello world"
print(a is b)      # False
```

### 4.3.3 手动驻留

使用 `sys.intern()` 可以强制驻留：

```python
import sys

a = sys.intern("hello world")
b = sys.intern("hello world")
print(a is b)  # True
```

**使用场景**：字典键优化。如果有大量重复的字符串作为键，驻留可以加速 `==` 比较为 `is` 比较（指针比较，O(1)）。

> [!WARNING]
> 驻留的字符串**永远不会被垃圾回收**（直到程序结束），滥用会导致内存泄漏。

---

## 4.4 str vs bytes: 文本与二进制的边界

### 4.4.1 str: 文本数据

`str` 存储的是 **Unicode 码点序列** (抽象的字符)，不涉及字节。

```python
s = "你好"
print(len(s))      # 2 (两个字符)
print(type(s))     # <class 'str'>
```

### 4.4.2 bytes: 二进制数据

`bytes` 存储的是 **8-bit 字节序列** (0-255 的整数)。

```python
b = b"hello"       # 字面量前缀 b
print(len(b))      # 5 (5个字节)
print(type(b))     # <class 'bytes'>
print(b[0])        # 104 (字符 'h' 的 ASCII 码)
```

### 4.4.3 编码与解码

*   **编码 (encode)**: `str` → `bytes` (文本 → 二进制)
*   **解码 (decode)**: `bytes` → `str` (二进制 → 文本)

```python
text = "你好"
data = text.encode('utf-8')   # str → bytes
print(data)                   # b'\xe4\xbd\xa0\xe5\xa5\xbd'

recovered = data.decode('utf-8')  # bytes → str
print(recovered)              # "你好"
```

> [!CAUTION]
> **编解码不匹配会导致乱码或崩溃**：
> ```python
> b'\xe4\xbd\xa0'.decode('gbk')  # 乱码
> b'\xff'.decode('utf-8')        # UnicodeDecodeError
> ```

---

## 4.5 字符串性能优化

### 4.5.1 字符串是不可变的

Python 的 `str` 是**不可变对象** (Immutable)。任何"修改"操作都会创建新字符串。

```python
s = "hello"
s += " world"  # 创建新字符串，旧的 "hello" 变成垃圾
```

**性能陷阱**：循环拼接字符串

```python
# ❌ Bad: O(n²) 复杂度
result = ""
for word in words:
    result += word  # 每次都复制整个 result
```

```python
# ✅ Good: O(n)
result = "".join(words)  # 只分配一次内存
```

### 4.5.2 f-string 的编译期优化

f-string (Python 3.6+) 在**编译时**就被转换为高效的格式化代码。

```python
name = "Alice"
age = 30
s = f"My name is {name}, I'm {age} years old."
```

等价于（编译器生成的字节码）：

```python
s = "My name is " + str(name) + ", I'm " + str(age) + " years old."
```

这比旧的 `%` 格式化和 `.format()` 都更快，因为减少了函数调用开销。

---

## 4.6 高级技巧

### 4.6.1 原始字符串 (Raw String)

前缀 `r` 让字符串**忽略转义**，常用于正则表达式和文件路径：

```python
path = r"C:\new\file.txt"  # 不会把 \n 当成换行符
regex = r"\d{3}-\d{4}"    # 正则表达式中的 \ 不需要双写
```

### 4.6.2 字符串池化与性能

CPython 内部维护了一个**字符串池** (String Pool)。短字符串（通常 < 20 个字符）如果符合标识符规则，会被自动驻留在池中。

查看驻留情况：

```python
import sys

a = "short"
b = "short"
print(sys.getrefcount(a))  # 引用计数会显示有多少个变量指向它
```

---

## Summary

*   **Unicode** 是字符集（抽象编号），**UTF-8/16/32** 是编码方案（字节表示）。
*   **UTF-8** 是互联网标准（ASCII 兼容 + 节省空间）。
*   **字符串驻留** 可优化字典查询，但需谨慎使用。
*   **str** 是文本，**bytes** 是二进制数据，两者必须通过 `encode/decode` 转换。
*   字符串是**不可变的**，拼接大量字符串使用 `"".join()`。

下一章，我们将进入 Python 最核心的数据结构：**列表 (List) 与元组 (Tuple)**，深入剖析动态数组的扩容策略以及内存布局。

---
title: "Chapter 2. Python 对象模型与内存机制"
description: "深入理解 PyObject 结构体、引用计数机制、变量与对象的本质关系以及可变性原理"
updated: "2024-03-22"
---

# Chapter 2. Python 对象模型与内存机制

> **Learning Objectives**
> *   彻底打破 "变量是盒子" 的旧思维，建立 "变量是标签" 的核心认知
> *   理解 Python 对象的基石：`PyObject` C 结构体 (`ob_refcnt`, `ob_type`)
> *   掌握对象的三个核心属性：Identity (`id`), Type (`type`), Value
> *   深入引用计数机制 (Reference Counting) 与内存管理

## 2.0 引言：一切皆对象

在 Python 的世界观里，有一条至高无上的法则：**"一切皆对象" (Everything is an Object)**。 

这句话不是空洞的口号，而是贯穿解释器实现的物理事实。从简单的整数 `1`，到函数、类、甚至模块本身，它们在 CPython 内部都是一个实实在在的 C 语言结构体。理解了对象模型，你就理解了 Python 运行时的灵魂。

---

## 2.1 变量不是盒子 (Variables are Labels)

这是初学者最容易产生误解的地方。

### 2.1.1 传统思维：变量是盒子
在 C 或 Java 等静态语言中，声明 `int a = 1` 通常意味着：
1.  系统在内存中画出一个固定大小的方框（盒子）。
2.  给这个盒子起名叫 `a`。
3.  把整数 `1` 的二进制数据塞进这个盒子里。
如果你接着写 `a = 2`，系统会把盒子里的数据擦除，换成 `2`。

### 2.1.2 Python 思维：变量是标签 (便利贴)
但在 Python 中，代码 `a = 1` 的含义截然不同：
1.  系统首先在内存的堆区（Heap）创建一个对象 `1`。
2.  然后拿一张写着 `a` 的便利贴（标签），贴在这个对象 `1` 上。

如果我们接着写 `a = 2`：
1.  系统创建新的对象 `2`。
2.  把标签 `a` 从对象 `1` 上撕下来，贴到对象 `2` 上。
3.  对象 `1` 依然存在（直到被垃圾回收），只是不再被 `a` 引用了。

**变量本身没有类型，也没有值。它只是一个指向对象的引用。**

### 2.1.3 交互式实验：引用的本质

下面的实验展示了变量赋值如何仅仅是改变了"引用指向"，而不是复制对象。

1.  尝试点击 **console** 中的按钮，将变量 `a` 绑定到不同的对象。
2.  观察 **Object Heap** 中的引用计数 (`Ref Cnt`) 如何变化。

<div data-component="PythonObjectVisualizer"></div>

当你执行 `a = b` 时，Python **并没有复制** `b` 的值传给 `a`。它只是让 `a` 这个标签，指向了 `b` 目前所指向的同一个对象。此时，这两个变量互为**别名 (Alias)**。

---

## 2.2 剖析 PyObject：对象的肉身

所有的 Python 对象，在 C 语言层面都有一个公共的头部，定义在 `Include/object.h` 中。

### 2.2.1 PyObject 结构体
任何对象（即便是整数）都至少包含两个字段：

```c
typedef struct _object {
    Py_ssize_t ob_refcnt;   // 引用计数
    PyTypeObject *ob_type;  // 类型指针
} PyObject;
```

1.  **`ob_refcnt` (Reference Count)**:
    *   记录有多少个变量（引用）指向这个对象。
    *   当计数变为 0 时，对象会被立即回收（垃圾回收机制的基础）。
    
2.  **`ob_type` (Type Pointer)**:
    *   指向该对象的**类型对象**（也是一个对象）。
    *   它告诉解释器："我是个整数" 或 "我是个列表"，并定义了这个对象支持的操作（如加法、长度获取）。

### 2.2.2 变长对象 (PyVarObject)
对于列表 (`list`)、字符串 (`str`) 这种长度可变的对象，头部会多一个字段：

```c
typedef struct {
    PyObject ob_base;
    Py_ssize_t ob_size; /* Number of items in variable part */
} PyVarObject;
```
*   `ob_size`: 记录容器中元素的数量（即 `len()` 的返回值）。注意，这就是为什么 `len()` 是 O(1) 操作 —— 它不需要遍历，直接读这个字段即可。

---

## 2.3 对象的奥义：身份、类型与值

在 Python 中，每个对象都有三个属性：

#### 1. Identity (身份)
*   对象在内存中的地址。
*   使用 `id(obj)` 获取。
*   `is` 运算符就是比较两个对象的 `id()` 是否相同。

#### 2. Type (类型)
*   决定了对象支持哪些方法和操作。
*   使用 `type(obj)` 获取。
*   对象创建后，类型通常是不可变的（尽管可以通过黑魔法修改 `__class__`，但不推荐）。

#### 3. Value (值)
*   对象存储的具体数据。
*   有些对象的值是可变的（Mutable，如 `list`），有些是不可变的（Immutable，如 `int`, `tuple`）。

---

## 2.4 引用计数机制 (Reference Counting)

这是 Python 内存管理的核心。

```python
import sys

a = [1, 2, 3]
print(sys.getrefcount(a)) 
# 输出通常是 2。
# 1. 变量 'a' 引用了它
# 2. getrefcount(a) 的参数 'a' 也临时引用了它
```

### 循环引用 (Reference Cycles) 问题
如果两个对象互相引用（例如 `A` 引用 `B`，`B` 引用 `A`），它们的引用计数永远不会归零。
为了解决这个问题，Python 引入了 **标记-清除 (Mark and Sweep)** 垃圾回收器作为引用计数的补充，专门处理循环引用。我们将在"内存管理进阶"章节深入探讨。

---

## 2.5 可变性 (Mutability) 的陷阱

为什么 `list` 可以修改，而 `int` 不能？

```python
# Case 1: Mutable
x = [1, 2]
y = x
x.append(3)
print(y) # 输出 [1, 2, 3] -> y 随之改变，因为 y 和 x 指向同一个对象

# Case 2: Immutable
x = 10
y = x
x = x + 1
print(y) # 输出 10 -> y 没有变，因为整数不可变，x = x + 1 创建了一个新的整数对象 11
```

*   **Immutable**: 任何看似修改的操作，底层的本质都是**创建新对象并改变指向**。
*   **Mutable**: 操作是在**原对象**的内存空间上进行的。

---

## Summary

*   **变量是标签**：赋值是绑定行为，不是复制行为。
*   **PyObject**: 所有对象都有 `ob_refcnt` 和 `ob_type`。
*   **len() 是 O(1)**: 因为 `ob_size` 存储在头部。
*   **不可变对象**: 修改等于创建新对象。

在 **Chapter 3** 中，我们将深入探讨 Python 中最常用的基础数据类型——**数字 (Numeric Types)**，并揭示为什么 Python 的整数可以无限大。

---
title: "Chapter 3. 数值类型与精度陷阱"
description: "深入剖析 Python 任意精度整数实现、IEEE 754 浮点数陷阱以及 Decimal 高精度计算"
updated: "2024-03-22"
---

> **Learning Objectives**
> *   理解 Python 整数为何不会溢出（底层 `digit` 数组实现）
> *   掌握浮点数的 IEEE 754 标准与精度丢失本质 (`0.1 + 0.2 != 0.3`)
> *   学会使用 `decimal` 和 `fractions` 进行金融级精确计算
> *   理解 `bool` 是 `int` 的子类这一设计哲学

## 3.0 引言：数字的假象

在大多数编程语言中，数字似乎是最简单的部分。但在 Python 中，即便是一个简单的整数，背后也隐藏着复杂的工程设计。

在 C 或 Java 中，`int` 就像汽车的**里程表** —— 它有固定的位数（比如 32 位或 64 位）。如果你一直跑，一旦超过了 `999999`，它就会"回绕"归零或变成负数（溢出）。

但在 Python 中，整数更像是一串**珍珠项链**。如果你需要表示更大的数，Python 会自动往项链上再穿一颗珍珠。理论上，只要你的内存足够大，这串项链可以无限长。

---

## 3.1 整数 (int)：无限精度的秘密

Python 3 彻底取消了 C 语言式的固定长度 `long` 类型，统一使用 `int`。这意味着你永远不需要担心整数溢出问题（除非内存耗尽）。

### 3.1.1 自动扩容机制
在 CPython 内部，`int` 并不是直接对应 CPU 的 64 位整数寄存器。它是一个**变长结构体** (`PyLongObject`)。

```c
struct _longobject {
    PyObject_VAR_HEAD
    digit ob_digit[1]; // 实际上不仅仅是1，而是一个数组
};
```

*   **ob_digit**: 这是一个数组，存储着整数的"绝对值"。Python 将大整数视为以 $2^{30}$ (30-bit) 为基数的"超级大进制"数。
*   **无限大**: 当数值变大时，Python 运行时会自动重新分配内存，扩大 `ob_digit` 数组的长度，这就是为什么 `int` 仿佛没有上限。

> [!NOTE]
> 这种设计被称为 **Arbitrary-precision arithmetic (任意精度算术)**。它实际上是用软件模拟了手算加减法的过程（进位、借位），因此可以处理比 CPU 寄存器大得多的数字。

### 3.1.2 性能代价
天下没有免费的午餐。由于所有整数运算（即使是简单的 `a + b`）都需要经过 Python 的大数运算库逻辑，而不是直接映射为 CPU 的 `ADD` 指令，因此 Python 的整数运算速度远慢于 C 语言的原生机器指令运算。

<div data-component="IntegerMemoryLayout"></div>

> [!TIP]
> 如果你需要进行数百万次密集数值计算，请使用 **NumPy**。NumPy 使用 C 语言原生的定长整数（如 `int64`），速度快 10-100 倍，但由于它是固定长度的，所以会发生溢出。

---

## 3.2 浮点数 (float)：IEEE 754 的诅咒

Python 的 `float` 直接对应 C 语言的 `double`（双精度浮点数），遵循 **IEEE 754** 标准。这意味着它只有 53 位的有效精度。

### 3.2.1 经典的 0.3 问题

```python
>>> 0.1 + 0.2
0.30000000000000004
>>> 0.1 + 0.2 == 0.3
False
```

这是因为 `0.1` (1/10) 在二进制中是一个**无限循环小数** (`0.0001100110011...`)。计算机截断存储导致了误差。

### 3.2.2 永远不要直接比较 float
由于精度误差，禁止使用 `==` 比较两个浮点数。应该使用 `math.isclose()`。

```python
import math
math.isclose(0.1 + 0.2, 0.3)  # True
```

---

## 3.3 精确计算：Decimal 与 Fraction

对于金融、科学计算等不允许有任何误差的场景，必须放弃 `float`。

### 3.3.1 Decimal (十进制定点数)
`decimal` 模块提供了任意精度的十进制算术。

```python
from decimal import Decimal

a = Decimal("0.1")  # 注意：必须传字符串！传 float 0.1 依然会带有初始误差
b = Decimal("0.2")
print(a + b)        # Decimal("0.3")
```

### 3.3.2 Fraction (有理数/分数)
`fractions` 模块直接存储分子和分母，保证绝对精确。

```python
from fractions import Fraction

f = Fraction(1, 3)  # 1/3
print(f * 3)        # 1 (完全精确，不是 0.99999)
```

---

## 3.4 布尔值 (bool)：int 的伪装

在 Python 中，`bool` 是 `int` 的子类。

```python
issubclass(bool, int) # True
True == 1             # True
False == 0            # True
```

这意味着你可以（虽然不推荐）将布尔值直接参与数学运算：

```python
count = True + True   # 结果是 2
elements = ["a", "b"]
print(elements[False]) # elements[0] -> "a"
```

> [!WARNING]
> 虽然 Python 允许这样做，但在代码中把布尔值当整数用通常被视为 **Bad Smell**（坏味道），严重降低可读性。

---

## Summary

*   **int**: 自动变长，不会溢出，但比原生整数慢。
*   **float**: 双精度浮点（C double），存在精度丢失，禁止直接用 `==`。
*   **decimal**: 使用字符串初始化的十进制数，用于金融计算。
*   **bool**: 就是特殊的 `0` 和 `1`。

下一章，我们将讨论 Python 中最复杂也最强大的内置类型——**字符串 (Strings)**，包括 Unicode 编码模型、字符串驻留机制 (Interning) 以及 f-string 的优化原理。

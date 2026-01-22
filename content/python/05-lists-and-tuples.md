---
title: "Chapter 5. 列表与元组：动态数组的艺术"
description: "深入剖析 Python List 的动态数组实现、内存预分配策略、Tuple 不可变性优化以及性能对比"
updated: "2024-03-22"
---

# Chapter 5. 列表与元组：动态数组的艺术

> **Learning Objectives**
> *   理解 List 的动态数组实现与扩容策略（平摊 O(1) 复杂度）
> *   掌握 List 的内存布局（`PyListObject` 结构体）
> *   深入 Tuple 的不可变性优化（对象池、常量折叠）
> *   掌握 List 与 Tuple 的性能差异与使用场景

## 5.0 引言：容器的选择

`list` 和 `tuple` 是 Python 最常用的容器类型。虽然它们在语法层面非常相似（一个用方括号 `[]`，一个用圆括号 `()`），但在底层实现上有着本质区别。

选择它们不仅仅是风格问题，更是性能问题。就像你需要搬家：
*   **List (列表)** 像是一个**可调节大小的储物架**。你可以随时把东西（元素）放进去，不够用了架子会自动变宽（扩容）。灵活，但结构稍微复杂一点。
*   **Tuple (元组)** 像是一个**封死的快递箱**。一旦打包好（创建），里面的东西就不能换了，大小也不能改了。但正因为它是封死的，它非常结实、轻便，而且可以堆叠。

---

## 5.1 List：可变的动态数组

### 5.1.1 底层实现：PyListObject

很多初学者认为 List 是链表（Linked List），因为它可以变长。但在 CPython 源码中，`list` 实际上是一个**动态数组** (Dynamic Array)。

这意味着 List 在内存中是一块**连续**的区域。

```c
typedef struct {
    PyObject_VAR_HEAD
    PyObject **ob_item;   // 指向元素数组的指针（注意：这是指针数组）
    Py_ssize_t allocated; // 已分配的容量
} PyListObject;
```

*   **`ob_item`**: 这是一个**指针数组**。List 并不直接存储对象本身，而是存储指向对象的指针。这就是为什么 List 可以混合存储不同类型的对象（整数、字符串、自定义对象），因为它存的只是地址。
*   **`allocated`**: List 当前实际占用的内存槽位。
*   **`ob_size`**: List 当前看起来的长度（即 `len()`）。通常 `allocated >= ob_size`。

### 5.1.2 动态扩容：Over-Allocation 策略

当 `append()` 导致列表满载（即 `ob_size == allocated`）时，Python 不会只申请这一个新元素的空间，而是会**重新分配**一块更大的内存，预留一些空位给未来使用。这个过程叫 **Over-Allocation (超额分配)**。

<div data-component="ListResizingVisualizer"></div>

#### 酒店类比
想象你在经营一家酒店（内存管理器）：
1.  客人（元素）来了，你不会每次只扩建一间房。
2.  因为扩建工程（`realloc`）很费时费力。
3.  所以，如果当前 8 间房满了，你会直接扩建到 16 间。
4.  这样接下来的 8 位客人入住如丝般顺滑（O(1)），直到 16 间再次住满。

#### CPython 的扩容公式

```c
new_allocated = (size_t)newsize + (newsize >> 3) + (newsize < 9 ? 3 : 6);
```

翻译为数学表达式：

$$
\text{new\_capacity} = n + \left\lfloor \frac{n}{8} \right\rfloor + \begin{cases} 3 & \text{if } n < 9 \\ 6 & \text{otherwise} \end{cases}
$$

**核心思想**：每次增加约 **12.5% (1/8)** 的额外空间。

> [!IMPORTANT]
> **为什么不是 2 倍扩容？**
> 
> C++ `std::vector` 通常使用 2 倍扩容（1.5 倍或 2 倍），但 Python 选择了更保守的 1.125 倍。
> 
> 原因：
> 1. **内存节约**：避免过度浪费（2 倍扩容最高会浪费 50% 内存）
> 2. **平摊 O(1)**：1.125 倍依然保证平摊常数时间插入，只是扩容频率稍高一点点。
> 3. **内存碎片**：小步扩容对内存分配器更友好。

### 5.1.3 删除操作的陷阱

```python
lst = [1, 2, 3, 4, 5]
del lst[2]  # 删除索引 2 的元素
```

**底层操作**：将索引 3~4 的元素向前移动一位（`memmove`），时间复杂度 **O(n)**。

| 操作 | 平均复杂度 | 最坏复杂度 |
|------|-----------|-----------|
| `append()` | O(1) 平摊 | O(n) (扩容时) |
| `pop()` | O(1) | O(1) |
| `pop(0)` | O(n) | O(n) |
| `insert(0, x)` | O(n) | O(n) |
| `del lst[i]` | O(n) | O(n) |

> [!TIP]
> **高频头部插入/删除**：使用 `collections.deque`（双端队列），O(1) 头尾操作。

---

## 5.2 Tuple：不可变的轻量级序列

### 5.2.1 为什么需要 Tuple？

1.  **性能**：创建和访问速度比 List 快（无需担心修改，优化空间更大）
2.  **哈希性**：Tuple 可作为字典键（`list` 不可哈希）
3.  **安全性**：防止意外修改（函数返回值、常量数据）

### 5.2.2 不可变 ≠ 内容不可变

Tuple 本身不可变，但如果元素是可变对象（如 `list`），元素的**内容**仍可修改。

```python
t = ([1, 2], 3)
t[0].append(4)  # 合法！Tuple 的引用不变，但 list 内容变了
print(t)        # ([1, 2, 4], 3)
```

### 5.2.3 Tuple 的内部优化

#### 1. 对象池 (Object Pool)

Python 会缓存长度为 0~20 的空 Tuple，避免重复创建。

```python
a = ()
b = ()
print(a is b)  # True（共享同一个 empty tuple）
```

#### 2. 常量折叠 (Constant Folding)

在编译期，字面量 Tuple 会被预先计算并存储为常量。

```python
import dis

def func():
    return (1, 2, 3)

dis.dis(func)
```

输出（简化）：

```text
LOAD_CONST    1 ((1, 2, 3))  # Tuple 在编译时已经创建好了
RETURN_VALUE
```

---

## 5.3 List vs Tuple 性能对比

### 5.3.1 创建速度

```python
import timeit

# List
timeit.timeit('[1, 2, 3, 4, 5]', number=10_000_000)  # ~0.25s

# Tuple
timeit.timeit('(1, 2, 3, 4, 5)', number=10_000_000)  # ~0.08s
```

**Tuple 快 3 倍**，因为：
1.  不需要分配额外容量（无 over-allocation）
2.  编译器可直接内联常量

### 5.3.2 内存占用

```python
import sys

lst = [1, 2, 3]
tup = (1, 2, 3)

print(sys.getsizeof(lst))  # 88 bytes
print(sys.getsizeof(tup))  # 64 bytes
```

**Tuple 省 27% 内存**（无需 `allocated` 字段和额外预留空间）。

---

## 5.4 列表推导式与生成器表达式

### 5.4.1 列表推导式 (List Comprehension)

```python
squares = [x**2 for x in range(10)]
```

**底层优化**：Python 会预先分配足够的空间（如果能推断出长度），避免频繁扩容。

### 5.4.2 生成器表达式 (Generator Expression)

```python
squares = (x**2 for x in range(10))  # 注意：圆括号
```

**惰性求值 (Lazy Evaluation)**：不会立即创建列表，而是按需生成。

| 特性 | 列表推导式 | 生成器表达式 |
|------|-----------|-------------|
| 内存 | O(n) | O(1) |
| 速度 | 一次性创建 | 按需生成 |
| 可迭代次数 | 无限 | 仅一次 |

---

## 5.5 高级技巧

### 5.5.1 切片复制与浅拷贝

```python
original = [1, [2, 3], 4]
shallow = original[:]  # 等价于 list(original) 或 original.copy()

shallow[1].append(99)
print(original)  # [1, [2, 3, 99], 4] ← 内部列表共享！
```

**深拷贝** (Deep Copy)：

```python
import copy
deep = copy.deepcopy(original)
```

### 5.5.2 Tuple Packing & Unpacking

```python
# Packing
coords = 10, 20  # 自动打包为 tuple

# Unpacking
x, y = coords
a, *rest, b = [1, 2, 3, 4, 5]  # a=1, rest=[2,3,4], b=5
```

### 5.5.3 List 的原地修改技巧

```python
# ❌ Bad: 创建新列表
lst = lst + [4, 5, 6]

# ✅ Good: 原地扩展
lst.extend([4, 5, 6])
# 或
lst += [4, 5, 6]  # 等价于 extend
```

---

## 5.6 何时使用 List vs Tuple？

| 场景 | 推荐类型 | 原因 |
|------|---------|------|
| 固定数据（如坐标、RGB） | Tuple | 不可变，性能更好 |
| 动态数据（如待办事项） | List | 可修改 |
| 函数返回多个值 | Tuple | 轻量级，语义清晰 |
| 字典键 | Tuple | 可哈希 |
| 大量小型集合 | Tuple | 内存省 |

---

## Summary

*   **List** 是动态数组，扩容策略：`new_capacity ≈ 1.125 × length`
*   **Tuple** 不可变，创建速度快 3 倍，内存省 27%
*   **删除操作** (`del`, `pop(0)`) 是 O(n)，避免在头部操作
*   **生成器表达式** 适合处理大数据（惰性求值）
*   **Tuple Unpacking** 是 Python 最优雅的特性之一

下一章，我们将探索 Python 最强大的数据结构：**字典 (dict) 与集合 (set)**，揭秘哈希表的实现原理与碰撞解决策略。

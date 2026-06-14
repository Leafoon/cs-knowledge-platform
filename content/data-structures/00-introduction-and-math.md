---
title: "Chapter 0: 课程导引与数学准备"
description: "建立学习 DSA 的动机与路径规划，掌握贯穿全课程的数学工具：求和公式、对数、递推式、概率基础与正确性证明方法。"
updated: "2026-03-09"
---

# Chapter 0: 课程导引与数学准备

> **学习目标**：
> - 理解为什么 DSA（Data Structures and Algorithms，数据结构与算法）是计算机科学中"最值得投资"的技能
> - 掌握贯穿全课程的数学工具：求和公式、对数运算、递推式、概率期望
> - 熟悉 CLRS 伪代码规范与 Python 实现风格
> - 建立正确性证明框架：数学归纳法、循环不变式、反证法

---

## 0.1 为什么学习数据结构与算法？

### 0.1.1 算法在工程与研究中的核心地位

先来看一个真实的例子。假设你要在一个包含 **10 亿**个数字的列表里，查找某一个目标数字。

| 方法 | 每步操作数 | 假设每步 1 纳秒 | 需要多久？ |
|------|-----------|----------------|-----------|
| **暴力遍历**（Linear Search）| 最坏 1,000,000,000 步 | 10⁹ ns | **约 1 秒** |
| **二分查找**（Binary Search）| 最坏 30 步（⌈log₂ 10⁹⌉）| 30 ns | **0.00000003 秒** |

同样的任务，算法不同，速度相差 **3000 万倍**！这就是 DSA 的魔力：**聪明的方法，远比堆硬件更有效。**

算法工程师要的不只是"写出来能跑"，而是：

- ✅ **正确性（Correctness）**：对所有合法输入都给出正确答案
- ✅ **效率（Efficiency）**：尽量少消耗时间与内存资源
- ✅ **可维护性（Maintainability）**：代码清晰，边界处理完善

> **💡 生活类比**：做菜和做"好菜"的区别。都能"做好"，但顶级厨师知道什么时候用什么技法，花最少时间做出最好的结果。算法工程师的目标一样。

### 0.1.2 "正确性"与"效率"的双重目标

很多初学者写代码只问"能不能出结果"，而忽视了：

```python
# ❌ 错误示范：看起来能跑，但对空列表会崩溃
def find_max(arr):
    max_val = arr[0]          # 若 arr 为空列表，IndexError！
    for x in arr:
        if x > max_val:
            max_val = x
    return max_val

# ✅ 正确示范：处理边界条件
def find_max(arr: list) -> int:
    if not arr:               # 边界条件：空列表
        raise ValueError("空列表没有最大值")
    max_val = arr[0]
    for x in arr:
        if x > max_val:
            max_val = x
    return max_val
```

**正确性**和**效率**是贯穿本课程的两条主线，缺一不可。

### 0.1.3 DSA 与系统设计、竞赛、面试的关系

```
          ┌─────────────────────────────────────────┐
          │         DSA 是一切的基础                  │
          └─────────────┬───────────────────────────┘
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
   🏆 算法竞赛     💼 技术面试     🏗️ 系统设计
  (LeetCode,     (FAANG,字节,     (数据库索引,
   Codeforces)    腾讯, Google)    搜索引擎,
                                   操作系统调度)
```

- **技术面试**：Google、Meta、字节跳动等顶级公司的面试核心就是 DSA，每道题背后都是对某个算法模式的考察。
- **算法竞赛**：ICPC、NOI、Codeforces 竞赛直接考察 DSA 的深度和广度。
- **系统设计**：Redis 的 ZSet 用跳表，MySQL 索引用 B+ 树，Google PageRank 是图算法——DSA 渗透到所有工业场景。

### 0.1.4 学习路径建议 & 本课程结构速览

<div data-component="LearningPathNavigator"></div>

**本课程完整结构**（共 12 个 Part，42+ 章）：

```
Part I   (Ch 0-2)：基础概念与复杂度分析  ← 你现在在这里
Part II  (Ch 3-6)：线性数据结构（数组、链表、栈、队列）
Part III (Ch 7-11)：树与优先队列（二叉树、BST、堆、AVL、红黑树）
Part IV  (Ch 12-13)：哈希与集合
Part V   (Ch 14-17)：排序与搜索
Part VI  (Ch 18-20)：图基础与遍历
Part VII (Ch 21-25)：图高级算法（最短路、网络流、强连通分量）
Part VIII(Ch 26-30)：算法设计范式（分治、动态规划、贪心、回溯）
Part IX  (Ch 31-34)：字符串算法
Part X   (Ch 35-38)：高级数据结构与摊销分析
Part XI  (Ch 39-40)：计算几何基础
Part XII (Ch 41-42)：计算复杂性与 NP
```

**建议学习节奏**：每天 1-2 小时，坚持 6 个月可覆盖全部内容。

---

## 0.2 数学基础回顾

> **为什么要学数学？** 复杂度分析（大 O 记号）和算法正确性证明，都需要精确的数学工具。这一节是"工具箱"，以后分析算法时会反复用到。

### 0.2.1 求和公式

#### 等差数列求和

$$\sum_{k=1}^{n} k = 1 + 2 + 3 + \cdots + n = \frac{n(n+1)}{2}$$

**为什么要记这个？** 分析嵌套循环的时间复杂度时，内层循环执行 $1 + 2 + \cdots + n$ 次，总次数就是这个公式。

**直观推导**（高斯的故事）：

```
正序：   1 +  2 +  3 + ... + n
逆序：   n + (n-1)+(n-2)+ ... + 1
─────────────────────────────────
两行之和：(n+1)+(n+1)+(n+1)+...+(n+1) = n(n+1)

所以原来的一行之和 = n(n+1)/2
```

#### 等比数列求和

$$\sum_{k=0}^{n} r^k = \frac{r^{n+1} - 1}{r - 1} \quad (r \neq 1)$$

当 $r = 2$（在计算机里最常见）：

$$1 + 2 + 4 + \cdots + 2^n = 2^{n+1} - 1$$

> **💡 记法口诀**：等比数列求和 = "最后一项乘 r 再减第一项，除以 (r-1)"。

#### 调和级数（Harmonic Series）—— 重要！

$$H_n = \sum_{k=1}^{n} \frac{1}{k} = 1 + \frac{1}{2} + \frac{1}{3} + \cdots + \frac{1}{n} \approx \ln n$$

更精确地：$H_n = \ln n + \gamma + O(1/n)$，其中 $\gamma \approx 0.5772$（欧拉-马歇罗尼常数）。

**为什么重要？** 快速排序的平均比较次数、哈希表的期望探测次数等许多分析都会用到调和级数。

<div data-component="HarmonicSeriesDemo"></div>

**常用求和公式速查**：

| 公式 | 结果 | 出现场景 |
|------|------|---------|
| $\sum_{k=1}^{n} k$ | $\frac{n(n+1)}{2} = \Theta(n^2)$ | 冒泡/选择排序 |
| $\sum_{k=1}^{n} k^2$ | $\frac{n(n+1)(2n+1)}{6} = \Theta(n^3)$ | 三层嵌套循环 |
| $\sum_{k=0}^{\log n} 2^k$ | $2n - 1 = \Theta(n)$ | 完全二叉树节点 |
| $\sum_{k=1}^{n} \frac{1}{k}$ | $\Theta(\log n)$ | 快速排序、哈希 |
| $\sum_{k=0}^{n} 2^{-k}$ | $< 2$ | 收敛等比（递归分析） |

### 0.2.2 对数与指数运算规则

**定义**：$\log_a b = c \Leftrightarrow a^c = b$（即"$a$ 的几次方等于 $b$"）

在算法分析中，对数的底数通常是 2（二分、二叉树），写作 $\lg n$。

**关键规则**：

$$\log_a (xy) = \log_a x + \log_a y \qquad \text{（乘法变加法）}$$

$$\log_a \frac{x}{y} = \log_a x - \log_a y \qquad \text{（除法变减法）}$$

$$\log_a x^b = b \cdot \log_a x \qquad \text{（指数提前）}$$

$$\log_a b = \frac{\log_c b}{\log_c a} \qquad \text{（换底公式）}$$

$$\log_a b \cdot \log_b c = \log_a c \qquad \text{（换底的链式法则）}$$

> **💡 关于底数的重要结论**：
> $$\log_2 n = \frac{\ln n}{\ln 2} = \frac{\log_{10} n}{\log_{10} 2}$$
> 不同底数之间只差一个**常数倍**，在大 O 记号中可以忽略。所以 $O(\log_2 n) = O(\log_{10} n) = O(\ln n)$，统一写作 $O(\log n)$。

**指数运算快速回顾**：

$$a^m \cdot a^n = a^{m+n}, \quad (a^m)^n = a^{mn}, \quad a^{-n} = \frac{1}{a^n}$$

### 0.2.3 递推式基础

**递推式（Recurrence）** 是定义算法时间复杂度的常用方式，尤其适合描述分治算法。

**示例——二分查找的时间复杂度**：

$$T(n) = T\!\left(\frac{n}{2}\right) + O(1), \quad T(1) = O(1)$$

用"递归展开"法求解：

$$T(n) = T\!\left(\frac{n}{2}\right) + 1 = T\!\left(\frac{n}{4}\right) + 1 + 1 = \cdots = T(1) + \log_2 n = O(\log n)$$

**示例——归并排序的时间复杂度**：

$$T(n) = 2T\!\left(\frac{n}{2}\right) + O(n), \quad T(1) = O(1)$$

这是经典的主定理（Master Theorem）应用场景，结果为 $T(n) = O(n \log n)$（Chapter 2 详细证明）。

> **⚠️ 常见陷阱**：递推式的展开需要假设 $n$ 是完美的某个幂次（如 $n = 2^k$），处理一般情况需要取整（$\lfloor n/2 \rfloor$ 或 $\lceil n/2 \rceil$）。

### 0.2.4 概率基础

#### 期望值（Expected Value）

若随机变量 $X$ 取值 $x_1, x_2, \ldots, x_k$，概率分别为 $p_1, p_2, \ldots, p_k$，则：

$$E[X] = \sum_{i=1}^{k} x_i \cdot p_i$$

**期望的线性性（Linearity of Expectation）**—— 极其有用的工具：

$$E[X + Y] = E[X] + E[Y]$$

**无论 $X$ 和 $Y$ 是否独立，线性性都成立！**

#### 指示随机变量法（Indicator Random Variable）

这是分析随机算法（如随机化快速排序）的利器。

**定义**：$I_A$ 是事件 $A$ 的指示变量：$I_A = \begin{cases} 1 & \text{若 } A \text{ 发生} \\ 0 & \text{否则} \end{cases}$

**关键性质**：$E[I_A] = P(A)$（指示变量的期望等于事件概率）

**例子**：计算从 $\{1, 2, \ldots, n\}$ 中以随机顺序排列时，期望的逆序对个数。

- 对每对 $(i, j)$（$i < j$），令 $I_{ij}$ 为"$j$ 在 $i$ 之前出现"的指示变量
- $P(I_{ij} = 1) = 1/2$（对称性）
- 期望逆序对数 $= \sum_{i < j} E[I_{ij}] = \binom{n}{2} \cdot \frac{1}{2} = \frac{n(n-1)}{4}$

### 0.2.5 取整函数、模运算与位运算技巧

#### 取整函数

$$\lfloor x \rfloor = \text{不超过 } x \text{ 的最大整数（向下取整）}, \quad \lceil x \rceil = \text{不小于 } x \text{ 的最小整数（向上取整）}$$

**常用不等式**：$x - 1 < \lfloor x \rfloor \leq x \leq \lceil x \rceil < x + 1$

**编程中的整除**：

```python
# Python 中的取整
import math
print(math.floor(7/2))   # 3  ← 向下取整
print(math.ceil(7/2))    # 4  ← 向上取整
print(7 // 2)            # 3  ← Python // 是向下取整（注意负数！）
print(-7 // 2)           # -4 ← Python // 向下取整，不是"截断"
```

```cpp
// C++ 中的整除（截断向零）
#include <cmath>
int a = 7 / 2;           // 3  ← 截断取整（向零）
int b = -7 / 2;          // -3 ← 截断向零，与 Python 不同！
int c = (int)floor(7.0 / 2); // 3
```

> **⚠️ 陷阱**：Python 的 `//` 是**向下取整**，C/C++ 的 `/` 是**截断取整**（向零），负数时行为不同！

#### 模运算

$$a \equiv b \pmod{m} \Longleftrightarrow m \mid (a - b)$$

在哈希表中，我们用 $h = k \bmod m$ 将键值映射到 $[0, m-1]$。

#### 位运算技巧（竞赛/面试高频）

| 操作 | 位运算 | 示例（n=12，即 1100₂） |
|------|--------|----------------------|
| 判断奇偶 | `n & 1` | `12 & 1 = 0`（偶数）|
| 除以 2 | `n >> 1` | `12 >> 1 = 6` |
| 乘以 2 | `n << 1` | `12 << 1 = 24` |
| 取第 k 位 | `(n >> k) & 1` | `(12 >> 2) & 1 = 1` |
| 清除最低 1 位 | `n & (n-1)` | `12 & 11 = 8`（1000₂）|
| 保留最低 1 位 | `n & (-n)` | `12 & (-12) = 4` |
| 判断 2 的幂次 | `n > 0 && (n & (n-1)) == 0` | `8 & 7 = 0` ✓ |

### 0.2.6 集合论基础

**集合（Set）**：一组不重复元素的无序集合，如 $S = \{1, 2, 3\}$。

| 符号 | 含义 | 例子 |
|------|------|------|
| $\|S\|$ | 集合的大小（基数）| $\|\{1,2,3\}\| = 3$ |
| $A \subseteq B$ | A 是 B 的子集 | $\{1,2\} \subseteq \{1,2,3\}$ |
| $A \cup B$ | 并集 | $\{1,2\} \cup \{2,3\} = \{1,2,3\}$ |
| $A \cap B$ | 交集 | $\{1,2\} \cap \{2,3\} = \{2\}$ |
| $A \setminus B$ | 差集 | $\{1,2,3\} \setminus \{2\} = \{1,3\}$ |
| $2^S$ | 幂集（所有子集）| $2^{\{1,2\}} = \{\emptyset,\{1\},\{2\},\{1,2\}\}$ |
| $\binom{n}{k}$ | 组合数（从 $n$ 中选 $k$ 个）| $\binom{4}{2} = 6$ |

**函数映射**：$f: A \rightarrow B$ 表示从集合 $A$（定义域）到集合 $B$（值域）的映射规则。

---

## 0.3 算法的描述方式

### 0.3.1 自然语言 vs 伪代码 vs 代码

同一个算法，三种描述方式：

**自然语言**（模糊，易误解）：
> "找到数组中最大的那个数，返回它。"

**CLRS 伪代码**（精确，语言无关）：

```
FIND-MAX(A, n)
1  max ← A[1]
2  for i ← 2 to n
3      if A[i] > max
4          max ← A[i]
5  return max
```

**Python 实现**（可直接运行）：

```python
def find_max(arr: list[int]) -> int:
    """
    找到数组中的最大值。
    
    时间复杂度：O(n)
    空间复杂度：O(1)
    
    边界条件：
    - arr 为空时抛出 ValueError
    """
    if not arr:
        raise ValueError("输入数组不能为空")
    
    max_val = arr[0]           # 初始化：假设第一个元素最大
    for x in arr[1:]:         # 从第二个开始比较，避免重复比较第一个
        if x > max_val:
            max_val = x
    return max_val

# 测试
print(find_max([3, 1, 4, 1, 5, 9, 2, 6]))  # 输出：9
print(find_max([-3, -1, -4]))               # 输出：-1（负数也能正确处理）
```

```cpp
#include <vector>
#include <stdexcept>
using namespace std;

// C++ 版本：find_max
// 时间复杂度：O(n)，空间复杂度：O(1)
int find_max(const vector<int>& arr) {
    if (arr.empty()) {
        throw invalid_argument("输入数组不能为空");
    }
    
    int max_val = arr[0];      // 初始化
    for (int i = 1; i < (int)arr.size(); ++i) {  // 注意：从 1 开始
        if (arr[i] > max_val) {
            max_val = arr[i];
        }
    }
    return max_val;
}
```

> **⚠️ C++ 注意事项**：`arr.size()` 返回 `size_t`（无符号整数），与 `int` 比较可能出现隐式转换问题，建议显式转换 `(int)arr.size()`。

### 0.3.2 CLRS 伪代码风格规范

本课程遵循 CLRS 第4版的伪代码约定：

| 约定 | 说明 | 示例 |
|------|------|------|
| **数组下标从 1 开始** | 与大多数数学教材一致 | `A[1..n]` |
| **赋值用 `←`** | 区别于等号判断 | `x ← 5` |
| **注释用 `▷`** | 说明该行作用 | `▷ 初始化最大值` |
| **`to` 表示循环区间** | 包含两端 | `for i ← 1 to n` |
| **缩进表示代码块** | 无大括号 | 像 Python |
| **可以调用全局函数** | 如 `SORT(A)`、`SWAP(A,i,j)` | 代表已知操作 |

> **⚠️ 注意**：CLRS 中数组从 1 开始，但 **Python 从 0 开始**。翻译成代码时务必调整下标！

### 0.3.3 Python 实现惯例与风格

**类型注解**（Python 3.9+）：

```python
from typing import Optional

def binary_search(arr: list[int], target: int) -> Optional[int]:
    """返回 target 在 arr 中的下标，不存在则返回 None。"""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2  # 防止整数溢出（虽然 Python 不会溢出，但是好习惯）
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return None
```

**Python 中常用的算法技巧**：

```python
# 1. 多重赋值（swap 不需要临时变量）
a, b = b, a

# 2. 列表推导式（快速构造）
squares = [x**2 for x in range(10)]

# 3. enumerate（同时获取下标和值）
for i, val in enumerate(arr):
    print(f"arr[{i}] = {val}")

# 4. 使用 float('inf') 表示无穷大
INF = float('inf')
dist = [INF] * n
dist[start] = 0

# 5. heapq（Python 的最小堆）
import heapq
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
print(heapq.heappop(heap))  # 1（最小值）
```

### 0.3.4 循环不变式（Loop Invariant）入门

**循环不变式**是一个在循环每次迭代前后都保持为**真**的断言（命题），是证明算法正确性的核心工具。

> **💡 生活类比**：你在整理一副扑克牌。每次从无序区取一张牌，插入已排序区的正确位置。在每次操作结束后，已排序区始终是排好序的——这个"始终已排序"就是循环不变式！

以**插入排序**为例：

```python
def insertion_sort(arr: list[int]) -> list[int]:
    """
    插入排序。
    
    循环不变式：在第 i 次迭代开始时，arr[0..i-1] 是【已排序的】。
    
    - 初始化：i=1 时，arr[0..0] 只有一个元素，天然有序 ✓
    - 保持：每次将 arr[i] 插入 arr[0..i-1] 的正确位置后，arr[0..i] 有序 ✓
    - 终止：i=n 时，arr[0..n-1] 即整个数组有序 ✓
    """
    n = len(arr)
    for i in range(1, n):
        key = arr[i]          # 取出当前要插入的牌
        j = i - 1
        # 将 arr[0..i-1] 中比 key 大的元素向右移一位
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key      # 插入到正确位置
    return arr

# 测试
print(insertion_sort([5, 3, 8, 1, 2]))  # [1, 2, 3, 5, 8]
```

```cpp
// 插入排序 C++ 版本（原地，稳定，O(n²)）
// 循环不变式：每次外层迭代后，arr[0..i] 已排序
void insertion_sort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 1; i < n; i++) {
        int key = arr[i];      // 取出当前要插入的元素
        int j = i - 1;
        // 将 arr[0..i-1] 中比 key 大的元素向右移一位
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;      // 插入到正确位置
    }
}
// 测试：{ 5, 3, 8, 1, 2 } → { 1, 2, 3, 5, 8 }
```

<div data-component="LoopInvariantWalkthrough"></div>

---

## 0.4 正确性证明方法预览

### 0.4.1 数学归纳法

**弱归纳**（最常用）：

1. **Base Case（基础步）**：证明命题对 $n = 1$ 成立
2. **Inductive Step（归纳步）**：假设对 $n = k$ 成立，证明对 $n = k+1$ 也成立
3. **结论**：命题对所有正整数 $n$ 成立

**例题**：证明 $\sum_{k=1}^{n} k = \frac{n(n+1)}{2}$

- **Base Case**：$n=1$ 时，左边 $= 1$，右边 $= \frac{1 \cdot 2}{2} = 1$，成立 ✓
- **Inductive Step**：假设对 $n=k$ 成立，即 $\sum_{i=1}^{k} i = \frac{k(k+1)}{2}$，则：

$$\sum_{i=1}^{k+1} i = \sum_{i=1}^{k} i + (k+1) = \frac{k(k+1)}{2} + (k+1) = \frac{(k+1)(k+2)}{2}$$

这正好是 $n = k+1$ 时的结论，归纳步成立 ✓

**强归纳**：归纳假设是"对所有 $m \leq k$ 成立"，适用于递归结构（如：分析某函数的时间复杂度）。

### 0.4.2 循环不变式三要素

| 要素 | 说明 |
|------|------|
| **初始化（Initialization）** | 在第一次迭代开始前，不变式成立 |
| **保持（Maintenance）** | 若第 $i$ 次迭代开始前成立，则第 $i$ 次迭代结束后仍成立 |
| **终止（Termination）** | 循环结束时，不变式给我们提供了算法正确性的证明 |

这三步对应数学归纳法的：Base Case → Inductive Step → 结论。

> **💡 类比数学归纳法**：
> - 初始化 ↔ Base Case（$n=1$ 时成立）
> - 保持 ↔ Inductive Step（$k \to k+1$）
> - 终止 ↔ 结论（最终命题）

### 0.4.3 反证法（Proof by Contradiction）

**假设结论不成立，推导出矛盾，从而证明结论成立。**

**经典应用**：证明"比较排序至少需要 $\Omega(n \log n)$ 次比较"

- 假设存在比较排序只需 $o(n \log n)$ 次比较
- $n!$ 种排列中每种都对应决策树的一条路径
- 决策树的叶节点数 $\geq n!$，故深度 $\geq \log_2(n!) \approx n \log n$（Stirling 近似）
- 与假设矛盾！因此比较排序至少需要 $\Omega(n \log n)$ 次比较 ✓

### 0.4.4 交换论证（Exchange Argument）

**用于证明贪心算法正确性**：若最优解中存在"不满足贪心准则"的相邻元素，则将其交换后解不变差，从而可以把最优解逐步变换成贪心解。

**例子**（活动选择问题 Chapter 27）：证明"选结束时间最早的活动"是最优策略。

---

## 0.5 常用数学结论速查表

<div data-component="MathFormulaCheatSheet"></div>

### 0.5.1 常用求和公式汇总

$$\sum_{k=1}^{n} k = \frac{n(n+1)}{2} \approx \frac{n^2}{2}$$

$$\sum_{k=1}^{n} k^2 = \frac{n(n+1)(2n+1)}{6} \approx \frac{n^3}{3}$$

$$\sum_{k=0}^{n} r^k = \frac{r^{n+1}-1}{r-1} \quad (r \neq 1)$$

$$\sum_{k=1}^{n} \frac{1}{k} = H_n \approx \ln n + 0.5772$$

$$\sum_{k=1}^{n} k \cdot x^k = \frac{x(1-(n+1)x^n+nx^{n+1})}{(1-x)^2} \quad (x \neq 1)$$

### 0.5.2 常用指数/对数不等式

对所有足够大的 $n$：

$$\log n \leq \sqrt{n} \leq n \leq n \log n \leq n^2 \leq n^3 \leq 2^n \leq n!$$

注意这些是**渐进关系**（当 $n \to \infty$ 时），不是任何具体 $n$ 下的数值比较。

**常用不等式**：

$$1 + x \leq e^x \quad (\text{对所有 } x \in \mathbb{R})$$

$$\left(1 - \frac{1}{n}\right)^n \leq \frac{1}{e} \leq \left(1 + \frac{1}{n}\right)^n \quad (n \geq 1)$$

$$\ln(1+x) \leq x \quad (x > -1)$$

### 0.5.3 Stirling 近似

$$n! \approx \sqrt{2\pi n} \left(\frac{n}{e}\right)^n$$

取对数：

$$\ln(n!) = n \ln n - n + O(\log n) \implies \log_2(n!) = \Theta(n \log n)$$

**应用**：比较排序下界的证明（见 0.4.3 节反证法）。

### 0.5.4 概率工具快速回顾

**Markov 不等式**：若随机变量 $X \geq 0$，则

$$P(X \geq a) \leq \frac{E[X]}{a}$$

**Chebyshev 不等式**：

$$P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}$$

其中 $\mu = E[X]$，$\sigma^2 = \text{Var}(X)$。

**生日悖论**：从 $n$ 个元素中有放回地抽取，约抽 $\Theta(\sqrt{n})$ 次后，期望出现重复。这直接影响哈希冲突分析！

**Union Bound（布尔不等式）**：

$$P(A_1 \cup A_2 \cup \cdots \cup A_k) \leq \sum_{i=1}^{k} P(A_i)$$

---

## ✅ 本章小结

| 知识点 | 核心内容 | 后续章节的用处 |
|--------|----------|---------------|
| **求和公式** | 等差 $\frac{n(n+1)}{2}$，调和 $H_n \approx \ln n$ | 分析嵌套循环、哈希碰撞 |
| **对数规则** | 换底公式，$O(\log n)$ 与底数无关 | 二分查找、堆操作、平衡树 |
| **递推式** | 展开法，主定理（Ch2 详述） | 分治算法复杂度 |
| **概率工具** | 期望线性性，指示随机变量 | 随机化算法分析 |
| **位运算** | `&`、`\|`、`^`、`>>`、`<<` | 哈希、位图、竞赛技巧 |
| **循环不变式** | 初始化、保持、终止三要素 | 证明所有迭代算法 |
| **数学归纳法** | 弱归纳与强归纳 | 递归算法正确性 |

---

## 🏋️ 练习题

**基础练习**：

1. 计算 $\sum_{k=1}^{100} (2k - 1)$（前 100 个奇数之和），用等差数列公式验证。
2. 证明：$\sum_{k=0}^{\log_2 n} 2^k = 2n - 1$（完全二叉树的节点总数）。
3. 用位运算实现：判断整数 $n$ 是否恰好是 2 的某次幂。

**进阶练习**：

4. 设随机变量 $X$ 是掷一个公平骰子的点数，计算 $E[X]$ 和 $E[X^2]$，并验证 $\text{Var}(X) = E[X^2] - (E[X])^2$。
5. 用循环不变式证明如下"选择排序"的正确性：
   ```python
   def selection_sort(arr):
       for i in range(len(arr)):
           min_idx = i
           for j in range(i+1, len(arr)):
               if arr[j] < arr[min_idx]:
                   min_idx = j
           arr[i], arr[min_idx] = arr[min_idx], arr[i]
   ```
   （提示：不变式是"循环第 $i$ 次迭代开始前，$\text{arr}[0..i-1]$ 是整个数组中最小的 $i$ 个元素，且已排序"。）

**思考题**：

6. 为什么说"对数的底数在大 O 记号中不重要"？能否举反例说明，在具体的数值计算中底数非常重要？
7. Stirling 近似的意义是什么？为什么说"比较排序的下界是 $\Omega(n \log n)$"？

---

## 📚 扩展阅读

| 资料 | 章节 | 内容 |
|------|------|------|
| CLRS 第4版 | 附录 A（求和）、附录 B（集合）、附录 C（概率） | 本章数学工具的权威参考 |
| CLRS 第4版 | 第2章 §2.1（插入排序、循环不变式） | 本章循环不变式的完整示例 |
| Sedgewick 第4版 | 第1章 §1.4（算法分析基础） | 实际编程语言视角的数学分析 |
| MIT 6.006 Lecture 1 | [课程主页](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/) | 算法课入门演讲，强烈推荐 |
| Khan Academy | [对数](https://www.khanacademy.org/math/algebra2/x2ec2f6f830c9fb89:logs) | 对数基础复习，有交互练习 |

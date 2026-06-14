---
title: "Chapter 2: 算法设计基本思想与正确性"
description: "掌握迭代与递归的本质区别、循环不变式完整证明框架，深入理解贪心/分治/动态规划/回溯四大范式的适用场景，学会问题规约与高效实现技巧。"
updated: "2026-03-09"
---

# Chapter 2: 算法设计基本思想与正确性

> **学习目标**：
> - 能清楚说明递归与迭代的区别，包括调用栈、尾递归优化、递归改迭代
> - 能写出循环不变式并完整证明简单算法的正确性
> - 理解贪心/分治/DP/回溯四大范式的本质区别与适用条件
> - 掌握问题规约的思想，学会常见的位运算技巧

---

## 2.1 迭代与递归

### 2.1.1 迭代算法结构

**迭代（Iteration）**是用循环来重复执行某段代码，直到满足终止条件。这是我们最熟悉的编程方式：

```python
# 用迭代计算 n 的阶乘
# 时间复杂度：O(n)，空间复杂度：O(1) ← 没有额外调用栈
def factorial_iter(n: int) -> int:
    result = 1
    for i in range(1, n + 1):   # i 从 1 到 n
        result *= i
    return result

print(factorial_iter(5))   # 120
```

```cpp
// C++ 迭代版 — 原地 O(1) 空间
long long factorial_iter(int n) {
    long long result = 1;
    for (int i = 1; i <= n; i++) {  // i 从 1 到 n
        result *= i;
    }
    return result;
}
// factorial_iter(5) == 120
```

迭代的核心要素：
1. **初始状态**：`result = 1`
2. **循环体**：每次更新状态
3. **终止条件**：`i > n` 时退出
4. **结果读取**：循环结束后 `result` 即答案

> **💡 为什么迭代通常更"稳"？** 迭代不需要调用栈，不会栈溢出，内存占用固定。对于能迭代也能递归的问题，工程中通常优先迭代。

---

### 2.1.2 递归的定义与调用栈

**递归（Recursion）**是函数调用自身来解决更"小"版本的同一问题。每次调用都会在**调用栈（Call Stack）**上压入一个新的**栈帧（Stack Frame）**。

```python
# 用递归计算 n 的阶乘
# 时间复杂度：O(n)，空间复杂度：O(n) ← n 层调用栈
def factorial_rec(n: int) -> int:
    # ① 递归基（Base Case）：n=0 时停止，否则永远调用下去
    if n == 0:
        return 1
    # ② 递归调用：问题规模缩小（关键！必须能收敛）
    return n * factorial_rec(n - 1)

# factorial_rec(4) 的执行过程：
# factorial_rec(4)
#   → 4 * factorial_rec(3)
#         → 3 * factorial_rec(2)
#               → 2 * factorial_rec(1)
#                     → 1 * factorial_rec(0)
#                               → return 1        ← 触底
#                     → 1 * 1 = 1                 ← 回溯
#               → 2 * 1 = 2
#         → 3 * 2 = 6
#   → 4 * 6 = 24
```

```cpp
// C++ 递归版 — 每次调用占 O(1) 栈帧，共 n 层 → O(n) 总空间
long long factorial_rec(int n) {
    if (n == 0) return 1;          // ① 递归基
    return (long long)n * factorial_rec(n - 1);  // ② 递归
}
```

<div data-component="RecursionCallStack"></div>

**调用栈的工作方式：**

每次调用 `factorial_rec(n)` 时，系统会在栈上分配一个**栈帧**，保存：
- 当前函数的局部变量（`n` 的值）
- 返回地址（调用完成后跳回哪里继续执行）

| 调用时刻 | 栈顶 → 栈底 |
|---------|------------|
| 调用 `f(4)` | `f(4)` |
| 调用 `f(3)` | `f(3)` → `f(4)` |
| 调用 `f(2)` | `f(2)` → `f(3)` → `f(4)` |
| 调用 `f(1)` | `f(1)` → `f(2)` → `f(3)` → `f(4)` |
| 调用 `f(0)` | `f(0)` → `f(1)` → ... → `f(4)` ← 栈最深 |
| `f(0)` 返回 1 | 弹出 `f(0)`，`f(1)` 继续执行 |
| ... 逐层弹出 | ... |
| 所有返回完毕 | 栈空 |

> **🔑 两个必要条件**：任何递归函数都必须满足：
> 1. **有递归基（Base Case）**：在某个最小规模直接返回答案
> 2. **问题规模严格缩小**：每次递归调用的参数更"小"（如 n-1, n/2）

---

### 2.1.3 尾递归与编译器优化

当**递归调用是函数的最后一个操作**时，这种递归称为**尾递归（Tail Recursion）**。

普通版 `factorial_rec(n)` **不是**尾递归，因为 `return n * factorial_rec(n-1)` 中：
- 先调用 `factorial_rec(n-1)`
- 返回后**还要**做一次乘法

怎么改成尾递归？用一个**累加器（Accumulator）**：

```python
# 带累加器的尾递归版
# acc 存储"到目前为止已经算好的部分积"
def factorial_tail(n: int, acc: int = 1) -> int:
    if n == 0:
        return acc         # ← 直接返回累加器，不再做额外计算
    return factorial_tail(n - 1, n * acc)  # ← 最后一步就是递归调用（尾递归）

# factorial_tail(4, 1)
#   → factorial_tail(3, 4)     # acc = 4*1 = 4
#     → factorial_tail(2, 12)  # acc = 3*4 = 12
#       → factorial_tail(1, 24)# acc = 2*12= 24
#         → factorial_tail(0, 24) → return 24
```

```cpp
// C++ 尾递归版（Clang/GCC 开 -O2 后会被优化为迭代）
long long factorial_tail(int n, long long acc = 1) {
    if (n == 0) return acc;
    return factorial_tail(n - 1, (long long)n * acc);  // 尾调用
}
```

| 特性 | 普通递归 | 尾递归（经编译器优化后） |
|------|---------|----------------------|
| 栈帧占用 | $O(n)$ | $O(1)$（复用同一帧）|
| 额外计算 | 返回后还要相乘 | 返回即最终答案 |
| 溢出风险 | 大 n 时爆栈 | 无 |
| Python 支持 | ✅ 有递归但无优化 | ❌ CPython 不做 TCO |
| C++ 支持 | ✅ | ✅（开 -O2 后自动优化）|

> ⚠️ **Python 没有尾调用优化（TCO）**！即使写成尾递归形式，Python 仍会为每次调用分配新栈帧。Python 默认栈深上限约为 1000 层（`sys.getrecursionlimit()`）。

---

### 2.1.4 递归改迭代（使用显式栈模拟）

任何递归算法都可以改写为迭代 + **显式栈（Explicit Stack）**。这在栈深很大、语言不支持 TCO 时非常有用。

以**前序遍历二叉树**为例：

```python
# 递归版 — 简洁但有栈溢出风险
def preorder_rec(root):
    if root is None:
        return
    print(root.val)            # 处理当前节点
    preorder_rec(root.left)    # 递归左子树
    preorder_rec(root.right)   # 递归右子树

# 迭代版 — 用显式栈模拟调用栈行为
def preorder_iter(root):
    if root is None:
        return
    stack = [root]             # 用列表模拟栈，初始压入根节点
    while stack:
        node = stack.pop()     # 弹出栈顶（LIFO 特性）
        print(node.val)        # 处理当前节点
        # ⚠️ 先压右子树，再压左子树
        # 这样弹出时先弹左（因为栈是 LIFO，后压先弹）
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
```

```cpp
// C++ 迭代版前序遍历
// 原理：显式栈模拟系统调用栈；先压右、后压左
void preorder_iter(TreeNode* root) {
    if (!root) return;
    stack<TreeNode*> stk;
    stk.push(root);
    while (!stk.empty()) {
        TreeNode* node = stk.top(); stk.pop();
        cout << node->val << " ";
        if (node->right) stk.push(node->right);   // 先压右
        if (node->left)  stk.push(node->left);    // 后压左 → 先弹左
    }
}
```

**递归改迭代的通用步骤**：
1. 用一个**显式栈**模拟函数调用栈
2. 将初始调用压栈
3. 每次迭代：弹出栈顶，处理，然后把"子问题"压栈（注意压栈顺序与处理顺序相反）

---

### 2.1.5 递归深度与栈溢出风险

```python
import sys

# Python 默认递归深度限制
print(sys.getrecursionlimit())  # 通常是 1000

# 临时提升上限（谨慎使用！）
sys.setrecursionlimit(10000)

# 观察栈溢出：
# factorial_rec(10000) 在默认设置下会引发 RecursionError
```

```cpp
// C++ 栈大小通常为 1MB～8MB，一个栈帧约 32～128 字节
// 递归深度大约能到 10,000 ～ 100,000 层
// 超过后直接 Segmentation Fault（段错误）
// 竞赛中常用：在主线程手动创建大栈
#include <sys/resource.h>
// 扩大栈大小为 256MB（Linux 下有效）
// setrlimit(RLIMIT_STACK, &rl);
```

> **工程建议**：当递归深度可能超过 $10^4$ 时，务必改用迭代或显式栈。竞赛中 DFS 遍历大图时尤其要注意。

---

## 2.2 循环不变式详解

### 2.2.1 插入排序的循环不变式证明（完整版）

Chapter 0 已经简介了循环不变式的三要素（初始化、保持、终止），这里用**插入排序**做完整证明。

先回顾算法：

```python
def insertion_sort(arr: list[int]) -> list[int]:
    n = len(arr)
    for i in range(1, n):       # i 从 1 到 n-1
        key = arr[i]            # 当前要插入的元素
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
```

```cpp
void insertion_sort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}
```

**设计循环不变式：**

> **不变式（Invariant）**：在外层 `for` 循环每次**迭代开始时**，子数组 `arr[0..i-1]` **已按升序排列**，且包含原来 `arr[0..i-1]` 中的所有元素（只是顺序改变了）。

**三要素证明：**

| 要素 | 证明内容 |
|------|---------|
| **初始化（Initialization）** | 第一次迭代开始前，$i = 1$，子数组 `arr[0..0]` 只有一个元素，单元素数组天然有序 ✓ |
| **保持（Maintenance）** | 假设第 $i$ 次迭代开始前 `arr[0..i-1]` 有序。内层 while 循环将 `arr[i]` 插入到正确位置，使 `arr[0..i]` 有序，且未丢失任何元素——因此第 $i+1$ 次迭代开始前不变式依然成立 ✓ |
| **终止（Termination）** | 循环在 $i = n$ 时终止。由不变式，`arr[0..n-1]` 即整个数组已排好序 ✓ |

**为什么这个证明是严格的？**  
关键在"保持"步：内层 while 循环的作用是**向右移位**（而不是删除元素），所以元素总数不变，且最终 `key` 被插入到正确位置，维持了有序性。

---

### 2.2.2 归纳法与不变式的关系

循环不变式本质上就是**数学归纳法**在算法证明中的应用：

$$\underbrace{\text{初始化}}_{\text{归纳基础}} \quad + \quad \underbrace{\text{保持}}_{\text{归纳步}} \quad \Rightarrow \quad \underbrace{\text{终止时结论成立}}_{\text{归纳推论}}$$

- **归纳基础**：$n = 0$（或某个起始值）时命题成立
- **归纳步**：若命题对 $n = k$ 成立，则对 $n = k+1$ 也成立
- **对应关系**：每次外层循环的迭代 $\leftrightarrow$ 归纳中的 $n = k$

---

### 2.2.3 不变式设计技巧

设计一个好的循环不变式是算法证明中最难但最关键的一步。

**技巧一：盯住"已完成的工作"**

通常，不变式的形式是："执行 $i$ 次循环后，前 $i$ 个元素满足某性质"。

**技巧二：不变式要"足够强"**

不变式必须强到能推出终止后的结论。太弱的不变式在"保持"步能成立，但"终止"时推不出最终结论。

**反例**——下面这个不变式就太弱了：
> "不变式：数组元素之和不变"  
> 这个在排序中确实保持，但排序结束时推不出数组有序，因为太多其他操作也满足这个性质。

**技巧三：不变式要"足够弱"**

同时，不变式不能太强——太强了在某次迭代就可能打破。

```python
# 示例：找数组最大值的循环不变式设计
def find_max(arr):
    max_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
    return max_val

# ✅ 好的不变式：
# "每次迭代开始前，max_val 是 arr[0..i-1] 中的最大值"
# → 初始化（i=1）：arr[0..0] 的最大值是 arr[0] = max_val ✓
# → 保持：若 arr[i] > max_val，则更新；否则 max_val 不变。两种情况都保持不变式 ✓
# → 终止（i=n）：max_val 是 arr[0..n-1] 的最大值 = 整个数组的最大值 ✓

# ❌ 坏的不变式（太强）：
# "每次迭代后，max_val 是所有元素的最大值"
# → 在第一次迭代前（只看了 arr[0]）就未必是整个数组最大值，保持步骤可能不成立
```

```cpp
// C++ 找最大值
int find_max(const vector<int>& arr) {
    int max_val = arr[0];
    // 不变式：max_val == max(arr[0..i-1])
    for (int i = 1; i < (int)arr.size(); i++) {
        if (arr[i] > max_val)
            max_val = arr[i];
    }
    return max_val;
}
```

---

### 2.2.4 如何针对同一算法构造多个不变式

同一个算法可以有多个正确的不变式，不同的不变式可以证明不同的性质。

以**选择排序**为例：

```python
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
```

```cpp
void selection_sort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[min_idx])
                min_idx = j;
        }
        swap(arr[i], arr[min_idx]);
    }
}
```

对于这个算法，可以构造两个不同的不变式：

**不变式 A（证明有序性）**：
> 每次迭代开始时，`arr[0..i-1]` 已排序，且 `arr[0..i-1]` 包含整个数组中最小的 $i$ 个元素。

**不变式 B（证明元素不丢失）**：
> 每次迭代开始时，`arr` 仍然包含原始数组的所有元素（只是顺序改变了）。

两个不变式合起来才能完整证明选择排序的正确性：A 证明结果有序，B 证明没有元素被"凭空删除或创造"。

---

## 2.3 贪心、分治、动态规划预览

这是算法领域最重要的四大思想范式。本章只做直觉性预览——后续章节会深入展开每一种。

---

### 2.3.1 贪心（Greedy）：局部最优 → 全局最优

**核心思想**：每一步都做**当前看起来最好的选择**，不回头。

**生活中的类比**：找零钱。面对 6 角，先用最大面值（5 角），再用 1 角——这就是贪心。

**经典例题**：区间调度（活动选择问题）

有 $n$ 个活动，每个活动有开始时间 $s_i$ 和结束时间 $f_i$，同一时间只能参加一个活动，求最多能参加几个？

**贪心策略**：总是选**结束时间最早**的活动。

```python
def activity_selection(activities):
    """
    活动选择：贪心算法
    activities: [(start, end), ...]
    返回最多可参加的活动数量
    
    贪心选择性质：选结束最早的活动，不影响为后续活动留出最大空间
    """
    # 按结束时间排序（贪心的关键第一步）
    activities.sort(key=lambda x: x[1])
    
    count = 1              # 至少选第一个（结束最早）
    last_end = activities[0][1]   # 上一个选中的活动结束时间
    
    for start, end in activities[1:]:
        if start >= last_end:      # 不冲突 → 贪心地选
            count += 1
            last_end = end
    
    return count

# 示例：[(1,4),(3,5),(0,6),(5,7),(3,9),(5,9),(6,10),(8,11),(8,12),(2,14),(12,16)]
# 贪心结果：选 4 个活动（最优）
```

```cpp
// C++ 活动选择贪心
int activity_selection(vector<pair<int,int>> acts) {
    // 按结束时间升序排列
    sort(acts.begin(), acts.end(), [](auto& a, auto& b){ return a.second < b.second; });
    
    int count = 1;
    int last_end = acts[0].second;
    for (int i = 1; i < (int)acts.size(); i++) {
        if (acts[i].first >= last_end) {   // 不冲突 → 选
            count++;
            last_end = acts[i].second;
        }
    }
    return count;
}
```

**贪心适用的必要条件**：
- **贪心选择性质**：局部最优选择不会妨碍全局最优（需要严格证明！）
- **最优子结构**：问题的最优解包含子问题的最优解

> **⚠️ 贪心的最大误区**：觉得"感觉局部最优"就能推出全局最优。许多问题看似适合贪心，实际上不行（如背包问题）。贪心策略必须通过严格的**交换论证（Exchange Argument）**来证明。

---

### 2.3.2 分治（Divide & Conquer）：分解 → 解决 → 合并

**核心思想**：把一个大问题拆成若干个**规模更小的同类子问题**，分别求解，再合并结果。

**生活中的类比**：二分查字典——先翻到中间，判断目标词在前半还是后半，然后只在那半部分查找。

**三个步骤**：
1. **Divide（分解）**：将原问题分成若干子问题（通常是原规模的一半）
2. **Conquer（解决）**：递归解决每个子问题（直到规模小到可以直接返回）
3. **Combine（合并）**：将子问题的解合并为原问题的解

```python
# 经典分治：归并排序（Merge Sort）
def merge_sort(arr):
    """
    分解：一分为二
    解决：递归排序两边
    合并：两个有序数组合并
    
    时间复杂度：O(n log n)  ← 由主定理：T(n) = 2T(n/2) + O(n)，Case 2
    空间复杂度：O(n)         ← 合并时需要临时数组
    """
    if len(arr) <= 1:        # 递归基：1个元素天然有序
        return arr
    
    mid = len(arr) // 2
    left  = merge_sort(arr[:mid])   # ← 解决左子问题
    right = merge_sort(arr[mid:])   # ← 解决右子问题
    return merge(left, right)       # ← 合并

def merge(left, right):
    """两个有序数组合并为一个有序数组"""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

```cpp
// C++ 归并排序（原地版）
void merge(vector<int>& arr, int l, int m, int r) {
    // 合并 arr[l..m] 和 arr[m+1..r]（两段均已排序）
    vector<int> tmp(arr.begin() + l, arr.begin() + r + 1);
    int i = 0, j = m - l + 1, k = l;
    while (i <= m - l && j <= r - l) {
        if (tmp[i] <= tmp[j]) arr[k++] = tmp[i++];
        else                  arr[k++] = tmp[j++];
    }
    while (i <= m - l) arr[k++] = tmp[i++];
    while (j <= r - l)  arr[k++] = tmp[j++];
}

void merge_sort(vector<int>& arr, int l, int r) {
    if (l >= r) return;              // 递归基
    int m = l + (r - l) / 2;        // 防止 (l+r)/2 整数溢出
    merge_sort(arr, l, m);
    merge_sort(arr, m + 1, r);
    merge(arr, l, m, r);
}
```

**分治适用的条件**：
- 问题可以分解为**同类子问题**
- 子问题之间**相互独立**（不重叠！否则用 DP）
- 有高效的**合并步骤**

---

### 2.3.3 动态规划（Dynamic Programming, DP）：重叠子问题 + 最优子结构

**核心思想**：像分治一样分解问题，但子问题之间**可能重叠**——把已经算过的子问题答案**存起来（Memoization）**，避免重复计算。

**生活中的类比**：爬楼梯。到第 $n$ 阶的方法数 = 到第 $n-1$ 阶的方法数 + 到第 $n-2$ 阶的方法数。如果每次都重新算，会有大量重复；把子问题答案存起来，就省了大量时间。

**Fibonacci 数列——从暴力递归到 DP**：

```python
# ❌ 暴力递归：O(2^n)，大量重复！
def fib_naive(n):
    if n <= 1: return n
    return fib_naive(n-1) + fib_naive(n-2)
# fib_naive(40) 要算约 3.3 亿次！

# ✅ 带记忆化的 DP（Top-Down）：O(n)，每个子问题只算一次
from functools import lru_cache
@lru_cache(maxsize=None)
def fib_memo(n):
    if n <= 1: return n
    return fib_memo(n-1) + fib_memo(n-2)

# ✅ 自底向上的 DP（Bottom-Up）：O(n) 时间，O(1) 空间
def fib_dp(n):
    if n <= 1: return n
    a, b = 0, 1          # a = fib(n-2), b = fib(n-1)
    for _ in range(2, n+1):
        a, b = b, a + b  # 滚动更新，只保留最近两个值
    return b
```

```cpp
// C++ Bottom-Up DP（O(n) 时间，O(1) 空间）
long long fib_dp(int n) {
    if (n <= 1) return n;
    long long a = 0, b = 1;
    for (int i = 2; i <= n; i++) {
        long long c = a + b;
        a = b; b = c;      // 滚动更新
    }
    return b;
}
```

**DP 的两大必要条件**：

| 条件 | 解释 |
|------|------|
| **最优子结构** | 原问题的最优解由子问题的最优解构成（可以"无后效地"分解） |
| **重叠子问题** | 同一子问题在递归树中出现多次（若无重叠，分治已足够）|

---

### 2.3.4 回溯（Backtracking）：系统性枚举 + 剪枝

**核心思想**：通过**深度优先搜索（DFS）**系统地枚举所有可能的解，一旦发现当前路径不可能得到有效解，就**回退（Backtrack）**并尝试另一条路径。

**生活中的类比**：走迷宫。沿某条路走，发现是死路就退回到上一个岔口，换另一条路继续。

```python
# 经典回溯：全排列
def permutations(nums):
    """
    生成 nums 的所有全排列（回溯法）
    时间复杂度：O(n! × n)  — n! 个排列，每个需 O(n) 拷贝
    """
    result = []
    
    def backtrack(path, remaining):
        # 终止条件：没有剩余元素 → 当前排列完成
        if not remaining:
            result.append(path[:])   # 记录当前答案
            return
        
        for i, num in enumerate(remaining):
            path.append(num)                         # 做选择
            backtrack(path, remaining[:i] + remaining[i+1:])  # 递归
            path.pop()                               # 撤销选择（回溯！）
    
    backtrack([], nums)
    return result

print(permutations([1, 2, 3]))
# [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

```cpp
// C++ 全排列（回溯）
void backtrack(vector<int>& nums, vector<bool>& used,
               vector<int>& path, vector<vector<int>>& result) {
    if ((int)path.size() == (int)nums.size()) {
        result.push_back(path);   // 记录完整排列
        return;
    }
    for (int i = 0; i < (int)nums.size(); i++) {
        if (used[i]) continue;        // 已选过 → 跳过
        used[i] = true;
        path.push_back(nums[i]);      // 做选择
        backtrack(nums, used, path, result);
        path.pop_back();              // 撤销选择
        used[i] = false;
    }
}

vector<vector<int>> permutations(vector<int> nums) {
    vector<vector<int>> result;
    vector<bool> used(nums.size(), false);
    vector<int> path;
    backtrack(nums, used, path, result);
    return result;
}
```

**剪枝（Pruning）**是回溯的精髓：提前判断当前路径不可能成功，及时中止，避免无效搜索。

---

### 2.3.5 四种范式适用条件对比表

<div data-component="AlgorithmParadigmExplorer"></div>

| 范式 | 核心特征 | 典型问题 | 时间复杂度参考 | 正确性证明方式 |
|------|---------|---------|--------------|--------------|
| **贪心** | 局部最优=全局最优，不回头 | 活动选择、Dijkstra（非负权）、哈夫曼编码 | $O(n \log n)$（通常）| 交换论证（Exchange Argument）|
| **分治** | 独立子问题，必须合并 | 归并排序、快速排序、二分查找、卡拉苏巴乘法 | $O(n \log n)$（典型）| 递归归纳 |
| **动态规划** | 重叠子问题 + 最优子结构 | 背包、最长公共子序列、最短路（Bellman-Ford）| $O(n^2)$～$O(n^3)$（典型）| 最优子结构证明 |
| **回溯** | 系统枚举，有剪枝 | 全排列、N 皇后、数独、组合总和 | $O(n!)$ 最坏（剪枝后大幅减少）| 搜索树完整性 |

**选择范式的经验法则**：

```
问题有"无后效性"且子问题重叠？
    ↳ 是 → 尝试 DP
    ↳ 否 → 子问题独立且需合并？
             ↳ 是 → 尝试分治
             ↳ 否 → 有贪心选择性质？
                      ↳ 是 → 尝试贪心（需严格证明！）
                      ↳ 否 → 回溯 + 剪枝
```

---

## 2.4 问题规约（Reduction）

### 2.4.1 规约的基本思想

**规约（Reduction）**的核心思想是：**把一个未知问题转化为一个已知问题来解决**。

> **类比**：你不会算 $17 \times 24$，但你知道 $17 \times 24 = 17 \times 20 + 17 \times 4$——这就是把乘法规约到更简单的乘法加上加法。

规约的记号：$A \leq_p B$ 表示"问题 $A$ 可以规约到问题 $B$"，意味着"$B$ 的求解器能用来解决 $A$"。

**规约的三步流程**：

```
输入(问题A)
    ↓  变换（Transformation，O(f(n)) 时间）
输入(问题B)
    ↓  调用 B 的算法
输出(问题B的解)
    ↓  逆变换（可能需要）
输出(问题A的解)
```

**总时间** = 变换时间 + B 的求解时间 + 逆变换时间

---

### 2.4.2 将新问题规约为已知问题

**例子 1：用排序解决"找中位数"**

```python
def median(arr: list[float]) -> float:
    """
    我不会直接找中位数，但我会排序（已知问题）！
    
    规约：median 问题 → sorting 问题
    总复杂度：O(n log n)（受排序瓶颈限制）
    """
    sorted_arr = sorted(arr)        # 规约：调用已知的排序算法
    n = len(sorted_arr)
    if n % 2 == 1:
        return sorted_arr[n // 2]
    else:
        return (sorted_arr[n // 2 - 1] + sorted_arr[n // 2]) / 2.0
```

```cpp
// C++ 用排序规约到中位数问题
double median(vector<double> arr) {
    sort(arr.begin(), arr.end());    // 规约核心
    int n = arr.size();
    if (n % 2 == 1) return arr[n / 2];
    return (arr[n / 2 - 1] + arr[n / 2]) / 2.0;
}
```

**例子 2：用排序实现"去重（Unique）"**

```python
def unique(arr: list) -> list:
    """
    先排序（规约），再线性扫描相邻元素。
    O(n log n) 解决去重，比原生哈希表 O(n) 慢但更节省内存。
    """
    if not arr:
        return []
    arr_sorted = sorted(arr)
    result = [arr_sorted[0]]
    for x in arr_sorted[1:]:
        if x != result[-1]:     # 只要与前一个不同就加入
            result.append(x)
    return result
```

```cpp
// C++ 排序后去重
vector<int> unique_elements(vector<int> arr) {
    sort(arr.begin(), arr.end());
    arr.erase(unique(arr.begin(), arr.end()), arr.end());  // STL unique
    return arr;
}
```

---

### 2.4.3 规约在复杂度理论中的作用

规约不仅是算法设计工具，更是**复杂度理论**的基石：

- 若 $A \leq_p B$（$A$ 能规约到 $B$），且 $B$ 很"简单"（多项式时间可解），则 $A$ 也简单。
- 反过来说：若 $A \leq_p B$ 且 $A$ 已知很"难"（如 NP-hard），则 $B$ 也至少同样难。

> **例**：3-SAT $\leq_p$ 顶点覆盖（Vertex Cover）——这告诉我们顶点覆盖问题"至少和 3-SAT 一样难"，即 NP-hard。详见 Chapter 41–42（计算复杂性理论）。

---

## 2.5 算法高效实现小技巧

### 2.5.1 Early Exit（提前退出）

提前退出的核心思想：**一旦确定答案，立即返回，不做多余的计算**。

```python
# ❌ 低效版：即使找到也不停
def contains_slow(arr, target):
    found = False
    for x in arr:
        if x == target:
            found = True    # 找到了，但还继续循环！
    return found

# ✅ 高效版：找到即返回
def contains_fast(arr, target):
    for x in arr:
        if x == target:
            return True     # 『Early Exit』立即退出
    return False

# 对比：对于长度 n=1,000,000 的数组，target 在第 1 个元素
# 低效版：仍然遍历 1,000,000 次
# 高效版：仅 1 次！
```

```cpp
// C++ Early Exit
bool contains_fast(const vector<int>& arr, int target) {
    for (int x : arr) {
        if (x == target) return true;   // 找到立即退出
    }
    return false;
}

// STL 版本：std::find 内部也用 Early Exit
bool found = std::find(arr.begin(), arr.end(), target) != arr.end();
```

**判断质数的 Early Exit 优化**：

```python
def is_prime(n: int) -> bool:
    """
    只需检查到 √n：若 n 有因子 d > √n，则 n/d < √n 必然已被发现。
    Early Exit：发现任何一个因子立即返回 False。
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:          # 偶数（除 2 外）快速判断
        return False
    i = 3
    while i * i <= n:        # 只检查到 √n
        if n % i == 0:
            return False     # Early Exit！
        i += 2               # 只检查奇数
    return True
```

```cpp
// C++ 质数判断（Early Exit + √n 优化）
bool is_prime(int n) {
    if (n < 2)     return false;
    if (n == 2)    return true;
    if (n % 2 == 0) return false;
    for (int i = 3; (long long)i * i <= n; i += 2) {
        if (n % i == 0) return false;   // Early Exit
    }
    return true;
}
```

---

### 2.5.2 哨兵技巧（Sentinel Node）与边界处理

**哨兵（Sentinel）**是一个人为添加的**虚拟元素**，它的具体值符合某种条件，专门用来简化边界判断，消除特殊情况处理。

**哨兵的好处**：
- 统一了"空链表"和"非空链表"的代码逻辑
- 消除了"是否到达链表头/尾"的额外判断
- 代码更简洁，bug 更少

**例子 1：在数组末尾添加哨兵**

```python
# 线性搜索 —— 无哨兵版（每次都要判断两个条件）
def linear_search_naive(arr, target):
    i = 0
    while i < len(arr) and arr[i] != target:   # 两个条件！
        i += 1
    return i if i < len(arr) else -1

# 线性搜索 —— 哨兵版（每次只判断一个条件）
def linear_search_sentinel(arr, target):
    arr_with_sentinel = arr + [target]   # 在末尾加哨兵（目标值本身）
    i = 0
    while arr_with_sentinel[i] != target:   # 只需判断一个条件！哨兵保证一定能找到
        i += 1
    return i if i < len(arr) else -1        # 找到的是哨兵 → 原数组中不存在
```

```cpp
// C++ 哨兵优化线性搜索
// 注意：需要在 arr 末尾有额外空间放哨兵
int linear_search_sentinel(vector<int>& arr, int target) {
    int n = arr.size();
    arr.push_back(target);   // 加哨兵（改变了数组！实际中要小心）
    int i = 0;
    while (arr[i] != target) i++;   // 只判断一次 — 哨兵保证循环必然终止
    arr.pop_back();           // 恢复数组
    return (i < n) ? i : -1;
}
```

**例子 2：链表的哑节点（Dummy Node）**

链表操作中，**头节点（Head）** 往往是最容易出 bug 的地方（删除头节点、在头部插入等）。

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# ❌ 无哑节点 —— 头节点是特殊情况，需要额外处理
def remove_all_naive(head, val):
    """删除链表中所有 val"""
    while head and head.val == val:   # 特殊处理头节点
        head = head.next
    curr = head
    while curr and curr.next:
        if curr.next.val == val:
            curr.next = curr.next.next
        else:
            curr = curr.next
    return head

# ✅ 有哑节点 —— 头节点与其他节点一视同仁
def remove_all_sentinel(head, val):
    """哑节点（Dummy）让头节点操作统一化"""
    dummy = ListNode(-1)     # 值为 -1 的哑节点（哨兵），永远在最前面
    dummy.next = head
    curr = dummy             # 从哑节点开始遍历
    while curr.next:
        if curr.next.val == val:
            curr.next = curr.next.next   # 删除 —— 无论是不是头节点，逻辑一样
        else:
            curr = curr.next
    return dummy.next        # 哑节点的下一个才是真正的头
```

```cpp
// C++ 哑节点删除链表元素
ListNode* remove_all(ListNode* head, int val) {
    ListNode dummy(0);        // 哑节点（栈上分配，自动销毁）
    dummy.next = head;
    ListNode* curr = &dummy;
    while (curr->next) {
        if (curr->next->val == val) {
            ListNode* to_del = curr->next;
            curr->next = curr->next->next;
            delete to_del;    // C++ 需要手动释放内存
        } else {
            curr = curr->next;
        }
    }
    return dummy.next;
}
```

---

### 2.5.3 位运算小技巧

位运算直接操作二进制位，速度极快（单个指令），在算法竞赛和底层优化中大量使用。

| 技巧 | 表达式 | 作用 |
|------|--------|------|
| 判断奇偶 | `n & 1` | `1` 为奇，`0` 为偶（比 `n % 2` 更快） |
| 判断 2 的幂次 | `n > 0 and (n & (n-1)) == 0` | `n` 是 2 的某次幂 |
| 清除最低位的 1 | `n & (n - 1)` | 统计 1 的个数（Brian Kernighan 算法）|
| 取最低位的 1 | `n & (-n)` | lowbit 操作（树状数组核心）|
| 快速乘以 2 | `n << 1` | 等价于 `n * 2`（位移更快）|
| 快速除以 2 | `n >> 1` | 等价于 `n // 2`（整数，向下取整）|
| 交换两个整数 | `a ^= b; b ^= a; a ^= b` | 无需临时变量（注意 a≠b 时才正确）|
| 取第 i 位 | `(n >> i) & 1` | 获取第 $i$ 位（从 0 开始）的值 |
| 置第 i 位为 1 | `n \| (1 << i)` | 将第 $i$ 位设为 1 |
| 清第 i 位为 0 | `n & ~(1 << i)` | 将第 $i$ 位设为 0 |

```python
# 位运算实战：统计整数 n 的二进制中有多少个 1（popcount）
def count_ones(n: int) -> int:
    """
    Brian Kernighan 算法：每次 n & (n-1) 消去最低位的 1
    循环次数 = n 中 1 的个数
    
    例：n = 13 = 1101
        13 & 12 = 1101 & 1100 = 1100 (=12)  → 消去了最低位的1
        12 & 11 = 1100 & 1011 = 1000 (=8)   → 消去了次低位的1
         8 &  7 = 1000 & 0111 = 0000 (=0)   → 消去了第三位的1
        循环 3 次 → 13 的二进制中有 3 个 1 ✓
    """
    count = 0
    while n:
        n &= (n - 1)   # 消去最低位的 1
        count += 1
    return count

# 快速判断一个数是否是 2 的幂
def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0
    # 解释：2 的幂的二进制形如 10...0（只有 1 个 1）
    # n-1 的形如 01...1
    # 两者相与为 0

# 测试
print(count_ones(13))           # 3  (1101₂ 有 3 个 1)
print(is_power_of_two(16))      # True  (16 = 2⁴)
print(is_power_of_two(6))       # False (6 = 110₂，有 2 个 1)
```

```cpp
// C++ 位运算
int count_ones(int n) {
    int count = 0;
    while (n) {
        n &= (n - 1);   // Brian Kernighan
        count++;
    }
    return count;
}
// 或者直接用编译器内置函数（更快）：
// int count = __builtin_popcount(n);   // GCC/Clang

bool is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}
```

**快速幂（Exponentiation by Squaring）**——位运算的经典应用：

```python
def fast_pow(base: int, exp: int, mod: int) -> int:
    """
    计算 base^exp % mod，利用位运算实现 O(log exp) 的快速幂。
    
    思想：exp 的二进制表示中，每一位对应 base 的某个平方幂。
    例：3^13 = 3^(1101₂) = 3^8 × 3^4 × 3^1
              ↑第3位    ↑第2位（0，跳过）  ↑第0位
    """
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:           # 当前位是 1
            result = result * base % mod
        base = base * base % mod  # 平方
        exp >>= 1             # 右移一位，处理下一位
    return result

print(fast_pow(2, 10, 1000))  # 1024 % 1000 = 24
```

```cpp
// C++ 快速幂（O(log exp)）
long long fast_pow(long long base, long long exp, long long mod) {
    long long result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) result = result * base % mod;   // 当前位为 1
        base = base * base % mod;                    // 平方
        exp >>= 1;                                   // 移到下一位
    }
    return result;
}
// fast_pow(2, 10, 1000) = 24
```

<div data-component="BitOperationPlayground"></div>

---

## 📚 本章小结

| 知识点 | 核心概念 | 后续章节应用 |
|--------|---------|------------|
| **迭代 vs 递归** | 调用栈、尾递归优化、显式栈模拟 | Part II 链表/树遍历，Part VI 图 DFS |
| **循环不变式** | 初始化、保持、终止三要素 | 每个迭代算法的正确性证明 |
| **贪心** | 局部最优 = 全局最优（需证明）| Chapter 21 最小生成树、Chapter 22 Dijkstra |
| **分治** | 独立子问题，必须合并 | Chapter 14 归并排序，Chapter 15 快排 |
| **DP** | 重叠子问题 + 最优子结构 | Chapter 26–28 DP 专题 |
| **回溯** | DFS + 剪枝，系统枚举 | Chapter 30 回溯专题，Chapter 42 NP |
| **问题规约** | $A \leq_p B$，转化为已知问题 | Chapter 41–42 复杂性理论 |
| **位运算** | `&`, `\|`, `^`, `>>`, `<<`, `lowbit` | Chapter 35 树状数组，竞赛优化 |

---

## 🏋️ 练习题

**基础练习**：

1. 将 `fib_naive`（指数递归）改写为带哑节点的迭代版，并分析空间复杂度的变化。
2. 用循环不变式证明**冒泡排序**的正确性（外层循环结束时，最大的 $i$ 个元素已就位于末尾）。
3. 用位运算判断一个整数 $n$ 是否是 $4$ 的幂次（提示：先判断是 $2$ 的幂，再检查 $1$ 在偶数位）。

**进阶练习**：

4. 实现归并排序的**非递归版本**（把递归 "自顶向下" 改为迭代 "自底向上"：先合并长度为 1 的段，再合并长度为 2 的段……）。
5. 设计一个回溯算法解决 **N 皇后问题**（$n \times n$ 的棋盘放 $n$ 个皇后互不攻击），并加入剪枝减少搜索空间。
6. 证明：贪心算法（选冻结时间最早的活动）对活动选择问题能给出最优解（用交换论证）。

**思考题**：

7. 分治与 DP 的本质区别是什么？能不能给出一个既可以用分治解也可以用 DP 解的问题？两者的时间复杂度有何差异？
8. 哨兵技巧不只是简化代码——它实际上消除了一类**边界条件（Edge Case）**。请思考：哨兵会不会在某些情况下引入新的 bug？在什么场合不适合用哨兵？
9. 快速幂是将 $O(n)$ 的乘法问题规约到 $O(\log n)$ 操作的典范。能否用类似思想设计 $O(\log n)$ 的矩阵幂运算，并说明其应用场景（斐波那契数列的 $O(\log n)$ 算法）？

---

**参考资料**：CLRS 第4版 Chapter 2（插入排序与正确性）、Chapter 4（分治）；MIT 6.006 Lecture 3–4；Skiena Chapter 1（范式概览）

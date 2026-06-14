---
title: "Chapter 1: 渐进复杂度分析"
description: "掌握 O/Ω/Θ 三大渐进记号的严格定义、主定理三种情形及递推式求解方法，能对任意程序片段进行精确的时间与空间复杂度分析。"
updated: "2026-03-09"
---

# Chapter 1: 渐进复杂度分析

> **学习目标**：
> - 能形式化定义并证明 O/Ω/Θ 关系，区分 o/ω 严格记号
> - 掌握主定理三种情形，能用递归树法、代入法求解递推式
> - 能对任意循环/递归片段做完整的时间/空间复杂度分析

---

## 1.1 算法效率的度量

### 1.1.1 时间复杂度 vs 空间复杂度

**为什么不直接测运行时间？**

直接测"运行了多少毫秒"有两大致命缺陷：

1. **机器依赖**：同一段代码在 M3 芯片上跑 10ms，在树莓派上跑 200ms，结论完全不同。
2. **输入依赖**：n = 100 时快，不代表 n = 10⁶ 时也快。

所以我们需要一个**与机器无关、与输入规模相关**的度量方式——这就是渐进复杂度（Asymptotic Complexity）。

> **💡 核心约定**：用 $n$ 表示输入规模（如数组长度、图的顶点数），分析"当 $n$ 趋向无穷大时，算法所需操作数的增长趋势"。

| 度量维度 | 定义 | 衡量方式 |
|---------|------|---------|
| **时间复杂度** | 算法执行的基本操作数（加法、比较、赋值等） | 关于输入规模 $n$ 的函数 |
| **空间复杂度** | 算法占用的额外内存 | 关于 $n$ 的函数（不含输入本身） |

```python
def sum_array(arr: list[int]) -> int:
    """
    时间复杂度：O(n)  — 遍历一次，n 次加法
    空间复杂度：O(1)  — 只用了 total 这一个额外变量
    """
    total = 0
    for x in arr:   # 循环 n 次
        total += x  # 每次 O(1) 操作
    return total
```

```cpp
// C++ 版本
int sum_array(const vector<int>& arr) {
    int total = 0;           // O(1) 空间
    for (int x : arr) {      // O(n) 时间
        total += x;
    }
    return total;
}
```

### 1.1.2 最坏情况 / 平均情况 / 最好情况

同一个算法，对不同的输入，运行时间可能截然不同。以**线性搜索**为例：

```python
def linear_search(arr: list[int], target: int) -> int:
    for i, x in enumerate(arr):
        if x == target:
            return i   # 找到就返回
    return -1          # 找不到
```

```cpp
int linear_search(const vector<int>& arr, int target) {
    for (int i = 0; i < (int)arr.size(); i++) {
        if (arr[i] == target)
            return i;   // 找到就返回
    }
    return -1;          // 找不到
}
```

| 情况 | 发生条件 | 比较次数 |
|------|---------|---------|
| **最好情况**（Best Case）| target 在第 1 位 | 1 次，$O(1)$ |
| **平均情况**（Average Case）| target 均匀分布在数组中 | $\frac{n+1}{2}$ 次，$O(n)$ |
| **最坏情况**（Worst Case）| target 不存在或在最后 | $n$ 次，$O(n)$ |

> **🏆 工程/竞赛优先关注最坏情况**：用户的输入不可预测，我们需要保证"任何情况下都不会太慢"。面试时说"时间复杂度 $O(n)$"默认指最坏情况。

> **📊 研究/随机算法关注平均情况**：快速排序的最坏情况是 $O(n^2)$，但平均情况是 $O(n \log n)$，工程中确实表现优秀。

### 1.1.3 摊销复杂度预览（详见 Chapter 35）

**摊销复杂度（Amortized Complexity）** 分析的是**一系列操作**的平均代价，而非单次操作的最坏代价。

> **💡 直觉**：ATM 取款，偶尔需要从总行调钱（很慢），但平均到每次取款，总体是快的。

经典例子：动态数组（Python 的 `list`）的 `append` 操作：
- 通常是 $O(1)$，但偶尔需要扩容（$O(n)$）
- 摊销分析可以证明：$n$ 次 `append` 总共是 $O(n)$，即每次**摊销** $O(1)$

```python
# 动态数组扩容示例（概念演示）
import sys

lst = []
for i in range(20):
    old_size = sys.getsizeof(lst)
    lst.append(i)
    new_size = sys.getsizeof(lst)
    if new_size != old_size:
        print(f"  → 扩容！元素数={i+1}, 内存 {old_size} → {new_size} bytes")
```

---

## 1.2 渐进记号正式定义

> **核心思想**：只关心函数在 $n \to \infty$ 时的**增长趋势**，忽略常数系数和低阶项。

### 1.2.1 O 记号（Big-O）：上界

**定义**（CLRS 3.1）：

$$f(n) = O(g(n)) \iff \exists\, c > 0,\ n_0 > 0\ 使得\ \forall n \geq n_0:\ f(n) \leq c \cdot g(n)$$

**直白翻译**：从某个 $n_0$ 开始，$f(n)$ 永远不超过 $g(n)$ 的某个常数倍。$g(n)$ 是 $f(n)$ 的**渐进上界**。

**例：证明 $f(n) = 2n^2 + 3n + 1 = O(n^2)$**

取 $c = 6$，$n_0 = 1$，则对所有 $n \geq 1$：
$$2n^2 + 3n + 1 \leq 2n^2 + 3n^2 + n^2 = 6n^2$$
（因为 $n \geq 1$ 时 $3n \leq 3n^2$ 且 $1 \leq n^2$）✓

> **⚠️ 常见误解**：$O$ 是**上界**，不是"等于"。$O(n^2)$ 可以描述一个实际上是 $O(n)$ 的函数。所以说"插入排序是 $O(n^2)$"是對的，但说"是 $\Theta(n^2)$"才是精确的。

### 1.2.2 Ω 记号（Big-Omega）：下界

**定义**：

$$f(n) = \Omega(g(n)) \iff \exists\, c > 0,\ n_0 > 0\ 使得\ \forall n \geq n_0:\ f(n) \geq c \cdot g(n)$$

**直白翻译**：从某个 $n_0$ 开始，$f(n)$ 至少是 $g(n)$ 的某个常数倍。$g(n)$ 是 $f(n)$ 的**渐进下界**。

**对称关系**：$f(n) = \Omega(g(n)) \iff g(n) = O(f(n))$

**应用**：证明算法的**下界**——"任何解决该问题的算法至少需要 $\Omega(n \log n)$ 时间"。

### 1.2.3 Θ 记号（Big-Theta）：精确界

**定义**：

$$f(n) = \Theta(g(n)) \iff f(n) = O(g(n)) \text{ 且 } f(n) = \Omega(g(n))$$

等价地：$\exists\, c_1, c_2 > 0,\ n_0 > 0$ 使得 $\forall n \geq n_0$：

$$c_1 \cdot g(n) \leq f(n) \leq c_2 \cdot g(n)$$

**直白翻译**：$f(n)$ 和 $g(n)$ 增长速度**本质相同**（只差常数倍）。

**例**：$f(n) = 3n^2 + 2n - 1 = \Theta(n^2)$
- 上界（$O$）：$3n^2 + 2n - 1 \leq 4n^2$，取 $c_2 = 4$，$n_0 = 2$ ✓
- 下界（$\Omega$）：$3n^2 + 2n - 1 \geq n^2$，取 $c_1 = 1$，$n_0 = 1$ ✓

### 1.2.4 小 o 与小 ω：严格渐进关系

| 记号 | 直觉含义 | 严格定义 |
|------|---------|---------|
| $f = o(g)$ | $f$ 比 $g$ **严格慢**（"上界但不精确"） | $\lim_{n\to\infty} \frac{f(n)}{g(n)} = 0$ |
| $f = \omega(g)$ | $f$ 比 $g$ **严格快**（"下界但不精确"） | $\lim_{n\to\infty} \frac{f(n)}{g(n)} = \infty$ |

**类比数学中的不等号**：

| 渐进记号 | 类比数字比较 |
|---------|-----------| 
| $f = O(g)$ | $f \leq g$（渐进）|
| $f = \Omega(g)$ | $f \geq g$（渐进）|
| $f = \Theta(g)$ | $f = g$（渐进）|
| $f = o(g)$ | $f < g$（严格）|
| $f = \omega(g)$ | $f > g$（严格）|

**例**：$n = o(n^2)$，因为 $\lim_{n\to\infty} \frac{n}{n^2} = \lim_{n\to\infty} \frac{1}{n} = 0$。

### 1.2.5 常见函数增长率排序与对比表

从慢到快（$n \to \infty$）：

$$O(1) \subset O(\log \log n) \subset O(\log n) \subset O(\sqrt{n}) \subset O(n) \subset O(n \log n) \subset O(n^2) \subset O(n^3) \subset O(2^n) \subset O(n!)$$

<div data-component="ComplexityGrowthChart"></div>

<div data-component="ComplexityComparisonTable"></div>

### 1.2.6 多变量渐进分析

当问题有多个规模参数时（如图算法中顶点数 $V$ 和边数 $E$），需要多变量渐进分析。

**BFS（广度优先搜索）为例**：

```python
from collections import deque

def bfs(graph: dict, start: int) -> list[int]:
    """
    时间复杂度：O(V + E)
    - 每个顶点入队/出队各一次：O(V)
    - 每条边检查一次：O(E)
    空间复杂度：O(V)  — 队列 + visited 数组
    """
    visited = set([start])
    queue = deque([start])
    order = []

    while queue:
        v = queue.popleft()     # O(1)
        order.append(v)
        for neighbor in graph[v]:  # 遍历 v 的所有邻居
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return order
```

```cpp
// 时间复杂度 O(V+E)，空间复杂度 O(V)
vector<int> bfs(const vector<vector<int>>& graph, int start) {
    int V = graph.size();
    vector<bool> visited(V, false);
    vector<int> order;
    queue<int> q;

    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int v = q.front(); q.pop();   // O(1)
        order.push_back(v);
        for (int neighbor : graph[v]) {    // 遍历 v 的所有邻居
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
    return order;
}
```

> **💡 为什么是 $O(V + E)$ 而不是 $O(V \cdot E)$？**
> 每个顶点只被加入队列**一次**（visited 标记防止重复），每条边只被检查**一次**（在其端点出队时）。总操作 = 顶点处理次数 + 边检查次数 = $V + E$。

---

## 1.3 递推式求解

递推式（Recurrence Relation）是描述分治算法等递归结构时间复杂度的标准工具。

### 1.3.1 递归树展开法（Recursion Tree）

**思路**：画出递归调用树，计算每层的总代价，然后对所有层求和。

**以 $T(n) = 2T(n/2) + n$ 为例**（归并排序）：

```
层 0（根）：          n                          代价: n
层 1：            n/2    n/2                     代价: n/2 + n/2 = n
层 2：         n/4 n/4 n/4 n/4                  代价: 4 × n/4 = n
...
层 k：       2^k 个节点，每个规模 n/2^k          代价: 2^k × n/2^k = n
...
层 log n：  n 个叶子，每个规模 1                 代价: n × O(1) = n
```

- **树高**：$\log_2 n$ 层（每次规模减半）
- **每层代价**：$n$（常数）
- **总代价**：$n \times (\log n + 1) = \Theta(n \log n)$

<div data-component="RecursionTreeBuilder"></div>

### 1.3.2 代入法（Substitution Method）

代入法是**假设答案，然后用数学归纳法验证**。

**步骤**：
1. 猜测（Guess）：凭经验/递归树猜测解的形式，如 $T(n) = O(n \log n)$
2. 归纳假设（Inductive Hypothesis）：假设对 $n/2$ 成立
3. 验证（Verify）：代入递推式，证明对 $n$ 也成立

**例：验证 $T(n) = 2T(\lfloor n/2 \rfloor) + n$ 的解是 $O(n \log n)$**

猜测 $T(n) \leq c \cdot n \log n$，需要证明此假设对 $n$ 成立：

假设对 $\lfloor n/2 \rfloor$ 成立，即 $T(\lfloor n/2 \rfloor) \leq c \cdot \lfloor n/2 \rfloor \cdot \log \lfloor n/2 \rfloor$，则：

$$T(n) = 2T(\lfloor n/2 \rfloor) + n \leq 2c \cdot \frac{n}{2} \log \frac{n}{2} + n = cn(\log n - 1) + n = cn \log n - cn + n$$

当 $c \geq 1$ 时，$-cn + n = n(1-c) \leq 0$，所以 $T(n) \leq cn \log n$ ✓

> **⚠️ 代入法陷阱**：猜测 $T(n) = O(n)$ 然后"证明" $T(n) \leq cn$ 时，代入后得 $T(n) \leq cn - \text{something}$，若减的项不能保证为正，则证明失败。要改猜。

### 1.3.3 主定理（Master Theorem）三种情形及证明思路

**主定理**（CLRS 定理 4.1）：设 $T(n) = aT(n/b) + f(n)$，其中 $a \geq 1, b > 1$ 为常数，$f(n)$ 为渐进正函数。令 $d = \log_b a$（关键值），则：

| 情形 | 条件 | 结论 | 直觉 |
|------|------|------|------|
| **情形 1** | $f(n) = O(n^{d-\varepsilon})$（$f$ 比 $n^d$ 慢） | $T(n) = \Theta(n^d)$ | 叶子层主导，根部开销可忽略 |
| **情形 2** | $f(n) = \Theta(n^d \log^k n)$（$f$ 与 $n^d$ 同阶） | $T(n) = \Theta(n^d \log^{k+1} n)$ | 每层代价相同，乘以层数 |
| **情形 3** | $f(n) = \Omega(n^{d+\varepsilon})$（$f$ 比 $n^d$ 快）且满足正则条件 | $T(n) = \Theta(f(n))$ | 根部开销主导，递归部分可忽略 |

**"关键值" $d = \log_b a$ 的直觉**：叶子数 = $a^{\log_b n} = n^{\log_b a} = n^d$。

<div data-component="MasterTheoremCalculator"></div>

**经典实例**：

| 递推式 | $a$ | $b$ | $f(n)$ | $n^d$ | 情形 | 结论 |
|--------|-----|-----|--------|--------|------|------|
| 二分搜索 $T(n) = T(n/2) + 1$ | 1 | 2 | $1$ | $n^0 = 1$ | 情形 2（$k=0$）| $\Theta(\log n)$ |
| 归并排序 $T(n) = 2T(n/2) + n$ | 2 | 2 | $n$ | $n^1 = n$ | 情形 2（$k=0$）| $\Theta(n \log n)$ |
| Strassen 矩阵 $T(n) = 7T(n/2) + n^2$ | 7 | 2 | $n^2$ | $n^{\log_2 7} \approx n^{2.81}$ | 情形 1 | $\Theta(n^{\log_2 7})$ |
| 三路分治 $T(n) = 3T(n/3) + n$ | 3 | 3 | $n$ | $n^1 = n$ | 情形 2 | $\Theta(n \log n)$ |

### 1.3.4 主定理扩展（Akra-Bazzi 方法）

当递推式中各子问题规模**不相等**时（如 $T(n) = T(n/3) + T(2n/3) + n$），主定理失效，需要 Akra-Bazzi 方法：

$$T(n) = \Theta\!\left(n^p \left(1 + \int_1^n \frac{f(u)}{u^{p+1}} du\right)\right)$$

其中 $p$ 满足 $\sum_i a_i \cdot b_i^p = 1$。

**例**：$T(n) = T(n/3) + T(2n/3) + n$
- $p$ 满足：$(1/3)^p + (2/3)^p = 1$，解出 $p = 1$
- 代入公式：$T(n) = \Theta(n \cdot (1 + \int_1^n \frac{u}{u^2} du)) = \Theta(n \log n)$

> 这是快速排序在**最坏分割比例** $1:2$ 时的递推式，结果仍是 $O(n \log n)$！

### 1.3.5 常见递推式解一览表

| 递推式 | 解 | 对应算法 |
|--------|-----|--------|
| $T(n) = T(n-1) + O(1)$ | $O(n)$ | 线性遍历 |
| $T(n) = T(n-1) + O(n)$ | $O(n^2)$ | 选择/冒泡排序 |
| $T(n) = T(n/2) + O(1)$ | $O(\log n)$ | 二分搜索 |
| $T(n) = T(n/2) + O(n)$ | $O(n)$ | 线性时间中位数（大部分工作） |
| $T(n) = 2T(n/2) + O(1)$ | $O(n)$ | 完全二叉树遍历 |
| $T(n) = 2T(n/2) + O(n)$ | $O(n \log n)$ | 归并排序 |
| $T(n) = T(n-1) + T(n-2)$ | $O(\phi^n)$（指数级）| 朴素 Fibonacci |
| $T(n) = 2T(n-1) + O(1)$ | $O(2^n)$ | 汉诺塔，朴素枚举 |
| $T(n) = 9T(n/3) + O(n^2)$ | $O(n^2 \log n)$ | 矩阵乘法某些变种 |

### 1.3.6 主定理失效的场景

**情形四（主定理不适用）**：$f(n) = \Theta(n^d)$ 但带有对数因子，且不完全符合情形 2。

**例**：$T(n) = 2T(n/2) + \frac{n}{\log n}$

- $a=2, b=2, d = \log_2 2 = 1$，$f(n) = n/\log n$
- $f(n) = O(n^{1-\varepsilon})$？不行，因为 $n/\log n$ 比任何 $n^{1-\varepsilon}$（$\varepsilon > 0$）都大
- $f(n) = \Theta(n)$？不行，因为 $n/\log n \neq \Theta(n)$
- **主定理三种情形均不适用**，需要递归树法，结果是 $\Theta(n \log \log n)$

---

## 1.4 复杂度分析实战

### 1.4.1 单层 / 嵌套循环分析

**规则**：顺序代码复杂度相加，嵌套代码复杂度相乘。

**单层循环**：

```python
# O(n) — 一层循环，每次 O(1)
for i in range(n):
    print(i)          # O(1)

# O(n) — 每次步长为 2，循环 n/2 次，仍是 O(n)
i = 0
while i < n:
    print(i)
    i += 2

# O(log n) — 每次 i 翻倍，循环 log₂n 次
i = 1
while i < n:
    print(i)
    i *= 2
```

```cpp
// O(n) — 一层循环
for (int i = 0; i < n; i++)
    cout << i;              // O(1)

// O(n) — 步长 2，仍是 O(n)
for (int i = 0; i < n; i += 2)
    cout << i;

// O(log n) — 每次 i 翻倍，共 log₂n 次
for (int i = 1; i < n; i *= 2)
    cout << i;
```

**嵌套循环**：

```python
# O(n²) — 经典双层嵌套
for i in range(n):       # n 次
    for j in range(n):   # n 次
        pass             # → n×n = n²

# O(n²) — 内层循环依赖外层：Σ(k=1→n) k = n(n+1)/2 = Θ(n²)
for i in range(n):        # n 次
    for j in range(i):    # i 次（0, 1, ..., n-1）
        pass

# O(n log n) — 内层是对数
for i in range(n):        # n 次
    j = 1
    while j < n:          # log n 次
        j *= 2
```

```cpp
// O(n²) — 经典双层嵌套
for (int i = 0; i < n; i++)       // n 次
    for (int j = 0; j < n; j++) { // n 次
    }                              // → n×n = n²

// O(n²) — 内层依赖外层：Σ k = n(n-1)/2 = Θ(n²)
for (int i = 0; i < n; i++)       // n 次
    for (int j = 0; j < i; j++) { // i 次
    }

// O(n log n) — 内层对数
for (int i = 0; i < n; i++)         // n 次
    for (int j = 1; j < n; j *= 2)  // log n 次
        ;
```

<div data-component="NestLoopAnalyzer"></div>

**三层嵌套（矩阵乘法）**：

```python
def matrix_multiply(A, B, n):
    """
    C = A × B，其中 A、B 均为 n×n 矩阵
    时间复杂度：O(n³)
    空间复杂度：O(n²)（结果矩阵 C）
    """
    C = [[0] * n for _ in range(n)]
    for i in range(n):           # n 次
        for j in range(n):       # n 次
            for k in range(n):   # n 次
                C[i][j] += A[i][k] * B[k][j]
    return C
```

```cpp
// 时间复杂度 O(n³)，空间复杂度 O(n²)
vector<vector<long long>> matrix_multiply(
        const vector<vector<long long>>& A,
        const vector<vector<long long>>& B, int n) {
    vector<vector<long long>> C(n, vector<long long>(n, 0));
    for (int i = 0; i < n; i++)           // n 次
        for (int j = 0; j < n; j++)       // n 次
            for (int k = 0; k < n; k++)   // n 次
                C[i][j] += A[i][k] * B[k][j];
    return C;
}
```

### 1.4.2 递归算法复杂度推导

**① 朴素递归 — $O(\phi^n)$ 指数级**：

```python
def fib_naive(n: int) -> int:
    if n <= 1:
        return n
    return fib_naive(n-1) + fib_naive(n-2)
# 递推式：T(n) = T(n-1) + T(n-2) + O(1) → T(n) = O(φⁿ) ≈ O(1.618ⁿ)
# n=50 时约需 2²⁵ ≈ 3×10⁷ 次调用，已很慢
```

```cpp
// 朴素递归 — 指数级时间，O(n) 调用栈
long long fib_naive(int n) {
    if (n <= 1) return n;
    return fib_naive(n-1) + fib_naive(n-2);
    // T(n) = T(n-1) + T(n-2) + O(1) → O(φⁿ)
}
```

**② 记忆化递归 — $O(n)$ 时间，$O(n)$ 空间**：

```python
def fib_memo(n: int, memo: dict = {}) -> int:
    """
    时间复杂度：O(n)  — 每个 fib(k) 只计算一次
    空间复杂度：O(n)  — 记忆化字典 + 调用栈
    """
    if n <= 1:
        return n
    if n not in memo:
        memo[n] = fib_memo(n-1, memo) + fib_memo(n-2, memo)
    return memo[n]
```

```cpp
// 记忆化递归 — O(n) 时间，O(n) 空间
unordered_map<int, long long> memo;
long long fib_memo(int n) {
    if (n <= 1) return n;
    if (memo.count(n)) return memo[n];
    return memo[n] = fib_memo(n-1) + fib_memo(n-2);
}
```

**③ 迭代法 — $O(n)$ 时间，$O(1)$ 空间（最优）**：

```python
def fib_iterative(n: int) -> int:
    """时间 O(n)，空间 O(1)"""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```

```cpp
// 迭代版本（最优）— O(n) 时间，O(1) 空间
long long fib_iterative(int n) {
    if (n <= 1) return n;
    long long a = 0, b = 1;
    for (int i = 2; i <= n; i++) {
        long long c = a + b;
        a = b;
        b = c;
    }
    return b;
}
```

### 1.4.3 对数复杂度的直觉理解（二分搜索）

**为什么二分搜索是 $O(\log n)$？**

每次比较后，搜索区间减半：

$$n \to \frac{n}{2} \to \frac{n}{4} \to \cdots \to 1$$

经过 $k$ 次后区间大小为 $n/2^k$，当 $n/2^k = 1$ 时，$k = \log_2 n$。

**核心直觉**：$\log_2(10^9) \approx 30$。10 亿个元素中，最多比较 30 次！

```python
def binary_search(arr: list[int], target: int) -> int:
    """
    前提：arr 已排序
    时间复杂度：O(log n)
    空间复杂度：O(1)（迭代版）
    
    递推式：T(n) = T(n/2) + O(1) → 主定理情形 2 → O(log n)
    """
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2   # 防止 (lo+hi) 溢出（Python 不会但 C++ 会）
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1             # 排除左半（包含 mid）
        else:
            hi = mid - 1             # 排除右半（包含 mid）
    return -1
```

```cpp
// C++ 版本（注意整数溢出防护）
int binary_search(const vector<int>& arr, int target) {
    int lo = 0, hi = (int)arr.size() - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;  // ⚠️ 防止 lo+hi 溢出
        if (arr[mid] == target) return mid;
        else if (arr[mid] < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}
```

### 1.4.4 $n \log n$ 的由来（归并排序）

归并排序的递推式 $T(n) = 2T(n/2) + O(n)$ 来自：

1. **分**：将数组一分为二——$O(1)$（单纯是计算中点）
2. **治**：递归排序左右两半——$2T(n/2)$
3. **合并**：将两个有序子数组合并——$O(n)$（线性扫描）

```python
def merge_sort(arr: list[int]) -> list[int]:
    """
    时间复杂度：Θ(n log n)  — 所有情况（最好/平均/最坏）
    空间复杂度：O(n)         — 辅助数组
    稳定性：稳定（相等元素保持原始相对顺序）
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])    # 递归排序左半  T(n/2)
    right = merge_sort(arr[mid:])   # 递归排序右半  T(n/2)
    return merge(left, right)       # 合并           O(n)

def merge(left: list[int], right: list[int]) -> list[int]:
    """合并两个有序数组，O(n) 时间，O(n) 空间"""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        # ≤ 保证稳定性（相等时优先取左边）
        if left[i] <= right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# 测试
arr = [38, 27, 43, 3, 9, 82, 10]
print(merge_sort(arr))  # [3, 9, 10, 27, 38, 43, 82]
```

```cpp
// 归并排序 — Θ(n log n)，O(n) 额外空间，稳定
void merge(vector<int>& arr, int lo, int mid, int hi) {
    vector<int> tmp(arr.begin()+lo, arr.begin()+hi+1);
    int i = 0, j = mid-lo+1, k = lo;
    while (i <= mid-lo && j <= hi-lo)
        arr[k++] = (tmp[i] <= tmp[j]) ? tmp[i++] : tmp[j++]; // ≤ 保证稳定
    while (i <= mid-lo) arr[k++] = tmp[i++];
    while (j <= hi-lo)  arr[k++] = tmp[j++];
}

void merge_sort(vector<int>& arr, int lo, int hi) {
    if (lo >= hi) return;
    int mid = lo + (hi-lo)/2;
    merge_sort(arr, lo, mid);      // T(n/2)
    merge_sort(arr, mid+1, hi);    // T(n/2)
    merge(arr, lo, mid, hi);       // O(n)
}
// 调用：merge_sort(arr, 0, arr.size()-1)
```

### 1.4.5 常数系数的工程意义：$10n$ vs $2n$

**理论算法分析忽略常数，但工程中常数至关重要。**

| 算法 A | 算法 B | $n = 10^4$ | $n = 10^6$ | 结论 |
|--------|--------|-----------|-----------|------|
| $T_A = 10n$（快速简单实现）| $T_B = 2n$（SIMD 向量化）| A:100k, B:20k | A:10M, B:2M | B 快 5 倍 |
| $T_A = 2n^2$（暴力）| $T_B = 100 n \log n$（聪明算法）| A:2×10⁸, B:5.3×10⁶ | A:2×10¹², B:4×10⁸ | 当 $n > 50$ 时 B 更快 |

> **💡 实践建议**：
> - 先用渐进复杂度筛掉明显劣势的算法
> - 同一渐进阶的算法中，关注缓存友好性（Cache Locality）、分支预测等常数因子

---

## 1.5 空间复杂度分析

### 1.5.1 原地算法（In-place）vs 额外空间

**原地算法（In-place Algorithm）**：算法使用的**额外**空间为 $O(1)$（不随输入规模增长），即直接在输入数据上操作。

| 算法 | 时间复杂度 | 额外空间 | 是否原地 |
|------|-----------|---------|--------|
| 冒泡/选择/插入排序 | $O(n^2)$ | $O(1)$ | ✅ |
| 归并排序 | $O(n \log n)$ | $O(n)$ | ❌ |
| 快速排序 | $O(n \log n)$ 平均 | $O(\log n)$ 调用栈 | ✅（近似）|
| 堆排序 | $O(n \log n)$ | $O(1)$ | ✅ |
| 计数排序 | $O(n + k)$ | $O(k)$ | ❌ |

```python
# 选择排序 — 原地，额外空间 O(1)
def selection_sort(arr: list[int]) -> None:
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]  # 只用 O(1) 额外空间

# 归并排序 — 非原地，额外空间 O(n)
def merge_sort_extra(arr):
    if len(arr) <= 1: return arr
    mid = len(arr) // 2
    return merge(merge_sort_extra(arr[:mid]),   # 创建新数组 → O(n) 额外空间
                 merge_sort_extra(arr[mid:]))
```

```cpp
// 选择排序 — 原地，O(1) 额外空间
void selection_sort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        int min_idx = i;
        for (int j = i+1; j < n; j++)
            if (arr[j] < arr[min_idx]) min_idx = j;
        swap(arr[i], arr[min_idx]);  // O(1) 额外空间
    }
}

// 归并排序 — 需要 O(n) 额外辅助数组
void merge_with_buf(vector<int>& arr, int lo, int mid, int hi,
                    vector<int>& tmp) {
    // tmp 是提前分配的 O(n) 辅助空间
    copy(arr.begin()+lo, arr.begin()+hi+1, tmp.begin()+lo);
    int i = lo, j = mid+1, k = lo;
    while (i <= mid && j <= hi)
        arr[k++] = (tmp[i] <= tmp[j]) ? tmp[i++] : tmp[j++];
    while (i <= mid) arr[k++] = tmp[i++];
    while (j <= hi)  arr[k++] = tmp[j++];
}
```

### 1.5.2 递归调用栈深度

每次函数调用都会在**调用栈（Call Stack）**上分配一个**栈帧（Stack Frame）**，存储局部变量、返回地址等。

| 递归模式 | 调用栈深度 | 空间复杂度 |
|---------|-----------|---------| 
| 线性递归（如 `factorial(n-1)`）| $n$ | $O(n)$ |
| 二分递归（如二分搜索）| $\log n$ | $O(\log n)$ |
| 归并排序 | $\log n$（树高）| $O(\log n)$ 栈 + $O(n)$ 合并 = $O(n)$ |
| 朴素 Fibonacci | $n$（深度优先调用链）| $O(n)$ |

```python
import sys
print(sys.getrecursionlimit())  # Python 默认最大递归深度：1000

# ⚠️ 危险！n=10000 时会栈溢出（RecursionError）
def dangerous_sum(n):
    if n == 0: return 0
    return n + dangerous_sum(n - 1)  # 调用栈深度 = n，O(n) 栈空间

# ✅ 安全！迭代版本，调用栈深度 = O(1)
def safe_sum(n):
    return n * (n + 1) // 2  # 数学公式，O(1) 时间 + O(1) 空间
```

```cpp
// ⚠️ 危险！n≥100000 时会栈溢出（Stack Overflow）
// 每个栈帧 ~32-64 bytes，默认栈大小约 1-8 MB
long long dangerous_sum(int n) {
    if (n == 0) return 0;
    return n + dangerous_sum(n - 1);  // 调用栈深度 = n
}

// ✅ 安全！迭代版本，O(1) 栈空间
long long safe_sum(int n) {
    return (long long)n * (n + 1) / 2;
}

// 尾递归版本（-O2 时编译器可能优化掉栈帧）
long long tail_sum(int n, long long acc = 0) {
    if (n == 0) return acc;
    return tail_sum(n - 1, acc + n);
}
```

### 1.5.3 辅助空间与总空间的区分

在分析空间复杂度时，需明确：

- **辅助空间（Auxiliary Space）**：算法使用的**额外**空间，不含输入本身
- **总空间（Total Space）**：辅助空间 + 输入占用的空间

```python
def find_max(arr: list[int]) -> int:
    """
    输入空间：O(n)  — arr 本身
    辅助空间：O(1)  — 只有 max_val 一个变量
    通常说"空间复杂度 O(1)"指的是辅助空间
    """
    max_val = arr[0]
    for x in arr:
        if x > max_val:
            max_val = x
    return max_val

def copy_and_sort(arr: list[int]) -> list[int]:
    """辅助空间：O(n)  — 复制了整个数组"""
    arr_copy = arr[:]    # 额外 O(n) 空间
    arr_copy.sort()
    return arr_copy
```

```cpp
// 辅助空间 O(1)（只用了 max_val 一个变量）
int find_max(const vector<int>& arr) {
    int max_val = arr[0];
    for (int x : arr)
        if (x > max_val) max_val = x;
    return max_val;
}

// 辅助空间 O(n) — 复制后排序
vector<int> copy_and_sort(const vector<int>& arr) {
    vector<int> arr_copy = arr;              // 额外 O(n) 空间
    sort(arr_copy.begin(), arr_copy.end());
    return arr_copy;
}
```

---

## ✅ 本章小结

| 知识点 | 核心内容 | 常见考点 |
|--------|---------|---------|
| **三大渐进记号** | $O$（上界）、$\Omega$（下界）、$\Theta$（精确）| 证明 $f(n) = O(g(n))$，找 $c$ 和 $n_0$ |
| **主定理** | 三情形，关键值 $d = \log_b a$ | 识别属于哪种情形，$f(n)$ 与 $n^d$ 的大小 |
| **递归树** | 每层代价求和，共 $\log_b n$ 层 | 归并排序、快速排序分析 |
| **代入法** | 猜答案 + 归纳证明 | 验证/推翻猜测 |
| **循环分析** | 顺序相加，嵌套相乘 | 看内层循环边界 |
| **空间分析** | 辅助空间是否随 $n$ 增长 | 递归深度常被忽视 |

---

## 🏋️ 练习题

**基础练习**：

1. 用 $c, n_0$ 的形式证明：$\frac{n^2}{2} - 3n = \Theta(n^2)$（同时找出 $O$ 和 $\Omega$ 的常数）。
2. 下列哪些关系成立？给出理由或反例：
   - $n^2 = O(n^3)$
   - $n^3 = O(n^2)$
   - $2^{2n} = O(2^n)$
   - $\log_2 n = \Theta(\log_{10} n)$
3. 用主定理求解：$T(n) = 4T(n/2) + n^2$，$T(n) = 3T(n/4) + n \log n$

**进阶练习**：

4. 分析以下代码的时间复杂度：
   ```python
   def mystery(n):
       count = 0
       i = n
       while i > 0:
           j = 0
           while j < i:
               count += 1
               j += 1
           i //= 2
       return count
   ```
   （提示：外层循环次数为 $\log n$，内层各次为 $n, n/2, n/4, \ldots$，总和是等比数列）

5. 快速排序的递推式为 $T(n) = T(k-1) + T(n-k) + O(n)$，其中 $k$ 是主元位置。
   - 最坏情况（$k=1$ 或 $k=n$）复杂度？
   - 最好情况（$k = n/2$）复杂度？
   - 用期望证明平均情况是 $O(n \log n)$？

**思考题**：

6. 为什么学习 DSA 时要区分"渐进复杂度"和"实际运行时间"？请举一个 $O(n^2)$ 实际比 $O(n \log n)$ 更快的场景。（提示：想想 $n$ 足够小时的情况，以及常数系数差异。）

7. 如果主定理三种情形都不适用，应该用哪种方法？全展开（代入法）和递归树各在什么场景下更直观？

---

## 📚 扩展阅读

| 资料 | 章节 | 内容 |
|------|------|------|
| CLRS 第4版 | Chapter 3（渐进记号）、Chapter 4（递推式）| 本章所有内容的权威来源 |
| CLRS 第4版 | 附录 D（矩阵基础，含 Strassen 应用） | 多维递推的完整分析 |
| Sedgewick 第4版 | Chapter 1.4（算法分析基础）| 编程语言视角，包含精确计数 |
| MIT 6.006 | [Lecture 1-2](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/) | 经典课程视频，强烈推荐 |
| Skiena 第2版 | Chapter 2（算法分析）| 更多实例和工程导向分析 |

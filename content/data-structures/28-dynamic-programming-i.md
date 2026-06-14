# Chapter 28: 动态规划基础（Dynamic Programming I）

## 章节导读

在 Chapter 27 我们学习了贪心：每一步局部最优，全局也最优。但现实中大量问题，贪心根本无从下手——比如"零钱兑换"、"编辑距离"，随便选一步都可能让后面的子问题变得更难。

这一章我们进入 **动态规划（Dynamic Programming，简称 DP）**。

DP 在 DSA 学习者中的口碑两极分化：

- 有人觉得它"神奇"——为什么填个表就解决了原本看起来指数级难的问题？
- 有人觉得它"难学"——每道题的状态设计都不一样，怎么找规律？

本章的目标是：**让你真正理解 DP 为什么有效，而不是背模板**。

学完这一章，你将能够：

1. 判断一道题是否适合 DP；
2. 从零开始设计 DP 状态和转移方程；
3. 读懂并写出 LCS、编辑距离、LIS、背包等经典 DP 的全量代码。

---

## 28.1 动态规划的本质

### 28.1.1 重叠子问题（Overlapping Subproblems）：Fibonacci 的指数重复计算

我们先用 Fibonacci 数列来感受"重叠子问题"带来的问题。

**斐波那契数列**：$F(0) = 0, F(1) = 1, F(n) = F(n-1) + F(n-2)$

如果你用最朴素的递归计算 $F(5)$：

```
F(5)
├── F(4)
│   ├── F(3)
│   │   ├── F(2)  ← 重复计算！
│   │   └── F(1)
│   └── F(2)      ← 重复计算！
│       ├── F(1)
│       └── F(0)
└── F(3)          ← 整棵子树重复！
    ├── F(2)
    └── F(1)
```

你会发现 $F(3)$ 被算了 2 次，$F(2)$ 被算了 3 次，$F(1)$ 被算了 5 次……

这棵递归树有指数级节点数：$O(2^n)$。计算 $F(50)$ 需要约 $2^{50} \approx 10^{15}$ 次运算——这对任何计算机来说都太慢了。

**核心矛盾**：同一个子问题被反复求解，而每次求解做的是完全一样的计算。

**解决思路**：把已经算过的值记录下来，下次遇到直接查表。这就是 DP 的核心。

> **重叠子问题**（Overlapping Subproblems）：递归分解时，多个不同的子问题路径汇聚到同一个更小的子问题。贪心和普通分治的子问题通常是独立的，DP 的子问题通常是重叠的。

---

### 28.1.2 最优子结构（Optimal Substructure）：局部最优组合全局最优

仅有"重叠子问题"还不够——你还需要**最优子结构**。

> **最优子结构**（Optimal Substructure）：原问题的最优解，建立在子问题的最优解之上。换句话说，你可以先把子问题解到最优，然后从这些最优子答案组合出原问题的最优答案。

**正面例子——最短路经过中间节点**：

若从 $A$ 到 $C$ 的最短路经过 $B$，那么 $A \to B$ 的那段一定也是 $A$ 到 $B$ 的最短路。如果存在更短的 $A \to B$，把它换进去整条路就更短了——矛盾。

**反面例子——最长路没有最优子结构**：

从 $s$ 到 $t$ 的最长简单路（无环），不一定包含 $s$ 到中间节点的最长路。因为"最长路"会绕圈，子问题之间不独立，贪心也不行。

---

### 28.1.3 DP 与分治的区别

| 对比维度 | DP | 分治（如归并排序） |
|---|---|---|
| 子问题是否重叠 | **是**（核心特征） | **否**（互相独立） |
| 是否缓存子问题答案 | 是（记忆化/填表） | 不需要（或不值得） |
| 代表算法 | LCS、背包、编辑距离 | 归并排序、快速幂 |
| 时间复杂度改进 | 指数 → 多项式 | 多项式（不能进一步减少） |

分治的"分"是把问题切开，两半**完全独立求解**，互不影响。DP 的子问题是**共享的**，所以必须缓存避免重算。

---

### 28.1.4 DP 与贪心的区别

贪心每次只做一个"局部最好"的决策，不回头。DP 则会系统地考察所有可能的子问题答案。

| 对比维度 | 贪心 | DP |
|---|---|---|
| 决策方式 | 每步选局部最优、不回头 | 枚举所有子状态取全局最优 |
| 是否需要记录历史 | 不需要 | 需要（表格/缓存） |
| 正确前提 | 需要贪心选择性质 + 最优子结构 | 只需要最优子结构 |
| 适用范围 | 更窄（需要额外证明） | 更广 |
| 时间复杂度 | 往往 $O(n \log n)$ 或更低 | 往往 $O(n^2)$ 或更高 |

**直觉总结**：

- 贪心像聪明的"选手"：每步只看当前最好，赌它能赢到底。
- DP 像"考官"：把所有可能路径都演算一遍，取最好结果。

---

### 28.1.5 DP 解题四步骤

无论多复杂的 DP 题，解题流程都可以归结为这四步：

**第一步 — 定义状态** $dp[i]$ 或 $dp[i][j]$：

精确描述 $dp[i]$ 代表什么（原问题规约到什么子问题的最优解）。

**第二步 — 写状态转移方程**：

枚举"最后一步"的所有可能，用小规模子问题的答案表达大规模问题的答案。

**第三步 — 确定初始值（Base Cases）**：

当子问题规模为 0 时（空字符串、空背包），答案直接已知。

**第四步 — 确定计算顺序（枚举顺序）**：

确保计算 $dp[i][j]$ 时所有依赖的 $dp$ 值已经算好了。

> ⚠️ **注意**：四步顺序不能颠倒。没有明确的状态定义，直接写转移方程必然出错。

---

## 28.2 记忆化递归（Top-Down）vs 迭代填表（Bottom-Up）

DP 的实现有两大范式，两者等价但各有优劣。

### 28.2.1 自顶向下（Top-Down / Memoization）：递归 + 缓存

**核心思想**：保持递归的思路不变，加一个字典/数组记录已算过的答案。

以斐波那契为例：

```python
# Top-Down：记忆化递归 + 哈希表缓存
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n: int) -> int:
    """
    Top-Down DP 计算斐波那契数

    【Edge Cases】
    - n == 0: 直接返回 0
    - n == 1: 直接返回 1
    - n < 0: 通常不处理（题目保证非负）

    【设计考量】
    - @lru_cache 自动把递归结果缓存在内存中
    - 等价于手写 memo = {}，if n in memo: return memo[n]
    """
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
```

```cpp
#include <bits/stdc++.h>
using namespace std;

// Top-Down：递归 + unordered_map 缓存
unordered_map<int, long long> memo;

long long fib(int n) {
    // Base Case
    if (n <= 1) return n;
    // 已缓存直接返回
    if (memo.count(n)) return memo[n];
    // 递归计算并存入缓存
    memo[n] = fib(n - 1) + fib(n - 2);
    return memo[n];
}
```

**每个子问题只计算一次**，递归树变成 $O(n)$ 个节点，时间复杂度 $O(n)$。

---

### 28.2.2 自底向上（Bottom-Up / Tabulation）：填表法

**核心思想**：从最小的子问题开始，按顺序往上填表，直到填出原问题的答案。

同样的斐波那契：

```python
def fib_bu(n: int) -> int:
    """
    Bottom-Up DP 计算斐波那契数

    【设计考量】
    - 显式循环，无递归栈开销
    - 只需保留前两项，空间 O(1)
    """
    if n <= 1:
        return n
    a, b = 0, 1  # dp[0], dp[1]
    for _ in range(2, n + 1):
        a, b = b, a + b  # dp[i] = dp[i-1] + dp[i-2]
    return b
```

```cpp
long long fib_bu(int n) {
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

---

### 28.2.3 两种方式的对比

| 对比维度 | Top-Down（记忆化递归） | Bottom-Up（迭代填表） |
|---|---|---|
| 代码思路 | 保留自然递归逻辑，加缓存 | 显式循环，手动控制计算顺序 |
| 栈溢出风险 | 有（n 很大时 Python 默认栈深度 1000） | 无 |
| 只算必要子问题 | ✅（子问题空间稀疏时更省时间） | ❌（需枚举所有状态） |
| 空间优化（滚动数组） | 较难 | 较容易 |
| 渐进复杂度 | 通常相同 | 通常相同 |

**经验法则**：

- 初学或状态复杂时，用 **Top-Down** 更容易实现；
- 需要空间优化（如背包从 $O(nW)$ 压缩到 $O(W)$）时，首选 **Bottom-Up**；
- 竞赛或面试时，两种都能写出来是最理想的。

<div data-component="TopDownVsBottomUp"></div>

---

### 28.2.4 何时选 Top-Down：子问题空间稀疏

如果状态数量巨大但实际只用到其中一小部分，Top-Down 的优势就体现出来了。

例如：**三维 DP**（位置 × 物品 × 颜色），若只有角落状态才有意义，Bottom-Up 会浪费大量时间填无用格子，而 Top-Down 只在需求时才递归计算。

---

### 28.2.5 何时选 Bottom-Up：子问题密集 + 需要空间优化

背包问题需要覆盖几乎所有 $(i, w)$ 状态，Bottom-Up 不亏。而且滚动数组压缩空间时，只能用 Bottom-Up（递归版本的滚动数组非常难控制）。

---

## 28.3 经典 DP 问题

### 28.3.1 斐波那契数列的进阶：矩阵快速幂 $O(\log n)$

普通 DP 是 $O(n)$，但还有更快的方法：**矩阵快速幂（Matrix Exponentiation）**，时间复杂度 $O(\log n)$。

原理：斐波那契可以表示为矩阵乘法：

$$\begin{pmatrix} F(n+1) \\ F(n) \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix}^n \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$

用快速幂计算矩阵的 $n$ 次方，即可在 $O(\log n)$ 时间内得到 $F(n)$。

```python
def matrix_mult(A, B):
    """2x2 矩阵乘法"""
    return [
        [A[0][0]*B[0][0] + A[0][1]*B[1][0],
         A[0][0]*B[0][1] + A[0][1]*B[1][1]],
        [A[1][0]*B[0][0] + A[1][1]*B[1][0],
         A[1][0]*B[0][1] + A[1][1]*B[1][1]],
    ]

def matrix_pow(M, n):
    """
    矩阵快速幂：计算 M^n

    思想：类似整数快速幂，n 是奇数时多乘一次 M
    时间复杂度：O(log n) 次矩阵乘法，每次 O(1)（2×2固定大小）
    """
    result = [[1, 0], [0, 1]]  # 单位矩阵（类似整数的"1"）
    while n > 0:
        if n % 2 == 1:
            result = matrix_mult(result, M)
        M = matrix_mult(M, M)
        n //= 2
    return result

def fib_matrix(n: int) -> int:
    """
    矩阵快速幂计算 F(n)

    【Edge Cases】
    - n == 0: F(0) = 0
    - n == 1: F(1) = 1

    【设计考量】
    - 对 n 极大时（如 n=10^18），O(n) DP 完全不可行，必须用矩阵快速幂
    """
    if n <= 1:
        return n
    M = [[1, 1], [1, 0]]
    R = matrix_pow(M, n - 1)
    return R[0][0]  # 结果在左上角
```

```cpp
typedef vector<vector<long long>> Matrix;
const int MOD = 1e9 + 7;

Matrix multiply(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, vector<long long>(n, 0));
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            if (A[i][k])
                for (int j = 0; j < n; j++)
                    C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % MOD;
    return C;
}

Matrix matpow(Matrix M, long long n) {
    int sz = M.size();
    Matrix result(sz, vector<long long>(sz, 0));
    // 单位矩阵
    for (int i = 0; i < sz; i++) result[i][i] = 1;
    while (n > 0) {
        if (n & 1) result = multiply(result, M);
        M = multiply(M, M);
        n >>= 1;
    }
    return result;
}

long long fib_matrix(long long n) {
    if (n <= 1) return n;
    Matrix M = {{1, 1}, {1, 0}};
    Matrix R = matpow(M, n - 1);
    return R[0][0];
}
```

> 📌 **面试提示**：当 $n$ 最大到 $10^{18}$，且需要对大质数取模时，矩阵快速幂是唯一可行方案。

---

### 28.3.2 最长公共子序列（LCS）

**LCS（Longest Common Subsequence）** 是 DP 领域最经典的二维 DP 题目。

#### 生活比喻

想象你有两段 DNA 序列，想找它们共有的最长相同片段（不要求连续）。比如 `ABCBDAB` 和 `BDCABA`，找出最长的公共子序列。

#### 问题定义

给定字符串 $X = x_1 x_2 \cdots x_m$ 和 $Y = y_1 y_2 \cdots y_n$，求最长公共子序列的长度。

**子序列**：从原字符串中**删去若干字符**（不改变相对顺序）得到的新字符串。可以不连续。

#### 状态定义

令 $dp[i][j]$ = $X$ 的前 $i$ 个字符 与 $Y$ 的前 $j$ 个字符的 LCS 长度。

#### 状态转移方程

$$dp[i][j] = \begin{cases} dp[i-1][j-1] + 1 & \text{if } X[i] = Y[j] \\ \max(dp[i-1][j],\; dp[i][j-1]) & \text{if } X[i] \neq Y[j] \end{cases}$$

#### 理解转移方程

- 若 $X[i] = Y[j]$：两边最后一个字符配对，那这对字符一定在 LCS 中（至少有一个最优解包含它），LCS 长度 = 前缀各减一的 LCS + 1。
- 若不等：最后一个字符至少有一方不参与 LCS，取两种舍弃方案的较大值。

#### 初始化

$dp[0][j] = 0$（$X$ 的前 0 个字符与任何串的 LCS = 0），$dp[i][0] = 0$。

<div data-component="LCSDPTableFill"></div>

```python
def lcs(X: str, Y: str) -> int:
    """
    最长公共子序列（LCS）长度

    时间复杂度：O(m * n)
    空间复杂度：O(m * n)，可优化至 O(min(m, n))

    【Edge Cases】
    - 任意一个字符串为空，LCS = 0
    - 两字符串完全相同，LCS = len(X)

    【设计考量】
    - dp[i][j] 定义为"前 i / 前 j 个字符的 LCS"，下标从 1 开始
    - 开辟 (m+1)*(n+1) 的表格，第 0 行 / 第 0 列自然初始化为 0
    """
    m, n = len(X), len(Y)
    # dp[i][j] = LCS of X[:i] and Y[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i-1] == Y[j-1]:     # 注意：X[i-1] 是第 i 个字符
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]


def lcs_backtrack(X: str, Y: str) -> str:
    """
    LCS 回溯：找出一条实际的 LCS 字符串

    从 dp[m][n] 沿转移方向往左上角追溯
    """
    m, n = len(X), len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i-1] == Y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    # 回溯
    result = []
    i, j = m, n
    while i > 0 and j > 0:
        if X[i-1] == Y[j-1]:
            result.append(X[i-1])
            i -= 1; j -= 1
        elif dp[i-1][j] >= dp[i][j-1]:
            i -= 1
        else:
            j -= 1

    return ''.join(reversed(result))


# 示例
X, Y = "ABCBDAB", "BDCABA"
print(lcs(X, Y))           # 4
print(lcs_backtrack(X, Y)) # BCBA 或 BDAB（不唯一）
```

```cpp
#include <bits/stdc++.h>
using namespace std;

int lcs(const string& X, const string& Y) {
    int m = X.size(), n = Y.size();
    // dp[i][j] = LCS of X[0..i-1] and Y[0..j-1]
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (X[i-1] == Y[j-1]) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }
    return dp[m][n];
}

string lcs_str(const string& X, const string& Y) {
    int m = X.size(), n = Y.size();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= n; j++)
            dp[i][j] = (X[i-1] == Y[j-1])
                ? dp[i-1][j-1] + 1
                : max(dp[i-1][j], dp[i][j-1]);

    // 回溯
    string res;
    int i = m, j = n;
    while (i > 0 && j > 0) {
        if (X[i-1] == Y[j-1]) { res += X[i-1]; i--; j--; }
        else if (dp[i-1][j] >= dp[i][j-1]) i--;
        else j--;
    }
    reverse(res.begin(), res.end());
    return res;
}
```

**复杂度**：时间 $O(mn)$，空间 $O(mn)$（可优化至 $O(\min(m,n))$，见 28.4.3）。

> **思考题**：LCS 和编辑距离有什么关系？如果 $\text{LCS}(s,t) = l$，那么只允许插入/删除的编辑距离是多少？（答案在本章末尾）

---

### 28.3.3 编辑距离（Edit Distance / Levenshtein Distance）

#### 生活比喻

你在手机上打字打错了，"kitten" 想改成 "sitting"。每次操作可以：

- **插入**一个字符（+1 步）
- **删除**一个字符（+1 步）
- **替换**一个字符（+1 步）

最少需要几步？

这就是**编辑距离**，又叫 Levenshtein Distance。

#### 状态定义

$dp[i][j]$ = 将 $s$ 的前 $i$ 个字符 转换成 $t$ 的前 $j$ 个字符所需的最少操作数。

#### 状态转移方程

$$dp[i][j] = \begin{cases} dp[i-1][j-1] & \text{if } s[i] = t[j] \quad \text{（字符相同，无需操作）} \\ 1 + \min\begin{cases} dp[i-1][j] & \text{删除 } s[i] \\ dp[i][j-1] & \text{插入 } t[j] \\ dp[i-1][j-1] & \text{替换} \end{cases} & \text{if } s[i] \neq t[j] \end{cases}$$

#### 初始化

- $dp[i][0] = i$（将 $s$ 前 $i$ 个字符删光，需 $i$ 次删除）
- $dp[0][j] = j$（从空串插入 $t$ 的前 $j$ 个字符，需 $j$ 次插入）

> ⚠️ **最常见错误**：把 $dp[0][j]$ 全部初始化为 0！这会导致结果完全错误。

<div data-component="EditDistanceTable"></div>

```python
def edit_distance(s: str, t: str) -> int:
    """
    编辑距离（Levenshtein Distance）

    支持三种操作：插入、删除、替换，每次代价均为 1

    时间复杂度：O(m * n)
    空间复杂度：O(m * n)，可优化至 O(min(m, n))

    【Edge Cases】
    - s 为空：返回 len(t)（需插入 len(t) 次）
    - t 为空：返回 len(s)（需删除 len(s) 次）
    - s == t：返回 0

    【内存易错点】
    - dp[0][j] = j（不是 0！），dp[i][0] = i（不是 0！）
    """
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初始化：第一行和第一列
    for i in range(m + 1): dp[i][0] = i  # 删光 s 的前 i 个字符
    for j in range(n + 1): dp[0][j] = j  # 从空串插入 t 的前 j 个字符

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i-1] == t[j-1]:
                dp[i][j] = dp[i-1][j-1]  # 字符匹配，不操作
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # 删除 s[i-1]
                    dp[i][j-1],    # 插入 t[j-1]
                    dp[i-1][j-1],  # 替换 s[i-1] 为 t[j-1]
                )

    return dp[m][n]


print(edit_distance("kitten", "sitting"))  # 3
print(edit_distance("intention", "execution"))  # 5
```

```cpp
int editDistance(const string& s, const string& t) {
    int m = s.size(), n = t.size();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1));

    // 初始化
    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s[i-1] == t[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = 1 + min({
                    dp[i-1][j],    // 删除
                    dp[i][j-1],    // 插入
                    dp[i-1][j-1]   // 替换
                });
            }
        }
    }
    return dp[m][n];
}
```

**实际应用**：

- 搜索引擎的拼写纠错（"pytohn" → "python"）；
- Git diff 算法的核心原语；
- 生物信息学 DNA/蛋白质序列比对；
- 语音识别后处理。

---

### 28.3.4 最长递增子序列（LIS）

**LIS（Longest Increasing Subsequence）** 是综合考查 DP 设计和二分优化的经典题。

#### 生活比喻

你有一列数，想从中挑出尽可能多的数，要求每个都比前一个严格更大（可以跳着挑，不必连续）。这个最长的递增序列就是 LIS。

例如：`[10, 9, 2, 5, 3, 7, 101, 18]` 的 LIS 是 `[2, 3, 7, 101]`，长度 4。

#### 方法 1：O(n²) DP

**状态定义**：$dp[i]$ = 以 $\text{nums}[i]$ 结尾的 LIS 长度。

**转移方程**：

$$dp[i] = 1 + \max_{j < i, \text{nums}[j] < \text{nums}[i]} dp[j]$$

扫描 $i$ 之前所有比 $\text{nums}[i]$ 小的位置 $j$，取最大 $dp[j]$。

```python
def lis_n2(nums: list[int]) -> int:
    """
    LIS（O(n²) DP）

    时间复杂度：O(n²)
    空间复杂度：O(n)

    【设计考量】
    - dp[i] 表示"以 nums[i] 结尾的 LIS 长度"
    - dp[i] 至少为 1（只有 nums[i] 自身）
    - 最终答案是 max(dp) 而不是 dp[-1]（LIS 不要求以最后一个元素结尾）
    """
    if not nums:
        return 0
    n = len(nums)
    dp = [1] * n  # 每个元素自身就是长度为 1 的 LIS

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:  # 严格递增
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)
```

#### 方法 2：O(n log n) 耐心排序（Patience Sorting）

这个方法用一个神奇的辅助数组 `tails`：

- `tails[k]` 存储"当前发现的所有长度为 $k+1$ 的递增子序列中，结尾元素的最小值"。
- 对每个新数 $x$，用二分查找找 `tails` 中第一个大于等于 $x$ 的位置，替换它；若 $x$ 比所有元素都大则追加。

> ⚠️ **注意**：`tails` 数组本身并不是真实的 LIS！它只保证长度正确。如果要重建 LIS，需要额外的 parent 数组。

```python
import bisect

def lis_nlogn(nums: list[int]) -> int:
    """
    LIS（O(n log n) 耐心排序）

    【核心思想】
    tails[i] = 长度为 i+1 的递增子序列中，结尾元素的最小可能值
    对每个新元素 x：
      - 如果 x > tails[-1]，则延长 LIS（tails.append(x)）
      - 否则用二分找到第一个 >= x 的位置，替换（保持 tails 尽量小）

    【正确性直觉】
    tails 始终是严格递增的（可证明），所以二分有效
    替换不破坏 LIS 长度的维护：将来可能以更小的结尾继续延伸

    【Edge Cases】
    - 空数组返回 0
    - 全相同元素（如 [5,5,5]）返回 1（严格递增）

    时间复杂度：O(n log n)
    空间复杂度：O(n)
    """
    tails = []  # tails[k] = 长度 k+1 的 LIS 最小结尾

    for x in nums:
        # bisect_left 找到 tails 中第一个 >= x 的位置
        pos = bisect.bisect_left(tails, x)
        if pos == len(tails):
            tails.append(x)   # x 比所有结尾都大，延伸 LIS
        else:
            tails[pos] = x    # 贪心替换：保持结尾尽量小

    return len(tails)


print(lis_nlogn([10, 9, 2, 5, 3, 7, 101, 18]))  # 4
print(lis_nlogn([0, 1, 0, 3, 2, 3]))             # 4
```

```cpp
int lis_nlogn(vector<int>& nums) {
    vector<int> tails;  // tails[i] = 长度 i+1 的 LIS 最小结尾

    for (int x : nums) {
        // lower_bound：第一个 >= x 的位置
        auto it = lower_bound(tails.begin(), tails.end(), x);
        if (it == tails.end()) {
            tails.push_back(x);  // 延伸
        } else {
            *it = x;             // 贪心替换
        }
    }
    return tails.size();
}
```

<div data-component="LIS_PatienceSort"></div>

**复杂度对比**：

| 方法 | 时间 | 空间 | 适用场景 |
|---|---|---|---|
| $O(n^2)$ DP | $O(n^2)$ | $O(n)$ | $n \leq 3000$，代码简单 |
| $O(n \log n)$ 耐心排序 | $O(n \log n)$ | $O(n)$ | $n \leq 10^5$，面试高频 |

---

### 28.3.5 0/1 背包（0/1 Knapsack）

**背包问题（Knapsack Problem）** 是 DP 中最重要的一类问题，变体极多，掌握透彻是大前提。

#### 生活比喻

你去旅游，只有一个背包，最多能装 $W$ 千克的物品。你面前有 $n$ 件宝贝，第 $i$ 件重量为 $w_i$，价值为 $v_i$。**每件物品只能拿一次（0 或 1）**，怎样打包价值最大？

#### 状态定义

$dp[i][j]$ = 前 $i$ 件物品，在容量为 $j$ 时的最大价值。

#### 状态转移方程

$$dp[i][j] = \begin{cases} dp[i-1][j] & \text{if } w_i > j \quad \text{（第 i 件放不下，不选）} \\ \max\!\bigl(dp[i-1][j],\; dp[i-1][j - w_i] + v_i\bigr) & \text{if } w_i \leq j \quad \text{（选或不选取较大值）} \end{cases}$$

**关键理解**：

- **不选第 $i$ 件**：价值 = $dp[i-1][j]$（前 $i-1$ 件在容量 $j$ 下的最优值）。
- **选第 $i$ 件**：价值 = $v_i$ + 前 $i-1$ 件在容量 $j - w_i$ 下的最优值（即 $dp[i-1][j-w_i]$）。

<div data-component="KnapsackDPTrace"></div>

```python
def knapsack_01(weights: list[int], values: list[int], W: int) -> int:
    """
    0/1 背包（二维 DP，O(nW) 时间，O(nW) 空间）

    【Edge Cases】
    - W == 0：返回 0
    - weights / values 为空：返回 0
    - 所有物品重量都超过 W：返回 0

    【设计考量】
    - dp[i][j] 只依赖 dp[i-1][...]，可以压缩为一维（见空间优化版本）
    - "只用前 i 件物品"比"恰好用 i 件"更好定义，初始化为 0 自然对应"空"
    """
    n = len(weights)
    # dp[i][j]: 前 i 件物品、容量 j 的最大价值
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        w_i, v_i = weights[i-1], values[i-1]
        for j in range(W + 1):
            dp[i][j] = dp[i-1][j]  # 不选第 i 件
            if j >= w_i:
                # 选第 i 件
                dp[i][j] = max(dp[i][j], dp[i-1][j - w_i] + v_i)

    return dp[n][W]


def knapsack_01_optimized(weights: list[int], values: list[int], W: int) -> int:
    """
    0/1 背包（一维滚动数组，O(W) 空间）

    【关键！！】容量必须从大到小枚举
    原因：若从小到大，dp[j - w_i] 已经被本轮（第 i 件）更新过了，
          等价于允许同一物品拿多次（变成完全背包了！）

    【内存易错点】
    - 倒序枚举：range(W, w_i - 1, -1)
    - 不能写 range(W, 0, -1)，要从 W 到 w_i 止（更小的容量不可能放得下 w_i）
    """
    n = len(weights)
    dp = [0] * (W + 1)  # dp[j] = 容量 j 的最大价值

    for i in range(n):
        # ⚠️ 必须倒序！防止同一物品被多次选取
        for j in range(W, weights[i] - 1, -1):
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])

    return dp[W]


# 示例
weights = [2, 3, 4, 5]
values  = [3, 4, 5, 6]
W = 8
print(knapsack_01(weights, values, W))           # 10
print(knapsack_01_optimized(weights, values, W)) # 10
```

```cpp
int knapsack01(vector<int>& w, vector<int>& v, int W) {
    int n = w.size();
    // 一维滚动数组，O(W) 空间
    vector<int> dp(W + 1, 0);
    for (int i = 0; i < n; i++) {
        // 必须倒序！0/1 背包的关键
        for (int j = W; j >= w[i]; j--) {
            dp[j] = max(dp[j], dp[j - w[i]] + v[i]);
        }
    }
    return dp[W];
}
```

---

### 28.3.6 完全背包（Unbounded Knapsack）

完全背包与 0/1 背包唯一的区别：**每件物品可以取无限次**。

#### 状态转移方程变化

$$dp[j] = \max(dp[j],\; dp[j - w_i] + v_i)$$

转移方程与 0/1 背包一模一样，**只有枚举容量的顺序不同**：

- 0/1 背包：**倒序**（$j$ 从 $W$ 到 $w_i$）
- 完全背包：**正序**（$j$ 从 $w_i$ 到 $W$）

#### 为什么正序就允许取无限次？

正序枚举时，当计算 $dp[j]$ 时，$dp[j - w_i]$ 已经是本轮更新后的值（可能已经包含了第 $i$ 件物品）。也就是说，第 $i$ 件物品可以被重复使用——这正是完全背包想要的！

```python
def knapsack_unbounded(weights: list[int], values: list[int], W: int) -> int:
    """
    完全背包（每件物品可取无限次）

    【关键对比 0/1 背包】
    - 0/1 背包：for j in range(W, w[i]-1, -1)  ← 倒序
    - 完全背包：for j in range(w[i], W+1)        ← 正序

    时间复杂度：O(n * W)
    空间复杂度：O(W)
    """
    dp = [0] * (W + 1)

    for i in range(len(weights)):
        # ⚠️ 正序！允许同一物品被重复选取
        for j in range(weights[i], W + 1):
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])

    return dp[W]
```

```cpp
int knapsackUnbounded(vector<int>& w, vector<int>& v, int W) {
    vector<int> dp(W + 1, 0);
    for (int i = 0; i < (int)w.size(); i++) {
        // 正序枚举：允许重复取
        for (int j = w[i]; j <= W; j++) {
            dp[j] = max(dp[j], dp[j - w[i]] + v[i]);
        }
    }
    return dp[W];
}
```

---

### 28.3.7 零钱兑换（Coin Change）：完全背包变体

LeetCode #322 是完全背包的最直接应用。

**问题**：给定面值数组 `coins` 和目标金额 `amount`，求使用最少硬币的组合方式。若不可达，返回 -1。

```python
def coin_change(coins: list[int], amount: int) -> int:
    """
    零钱兑换（LeetCode #322）

    完全背包变体：求最少硬币数（而非最大价值）
    只需把"max 价值"改成"min 个数"，初始化从 0 改成 INF

    【Edge Cases】
    - amount == 0：返回 0（零个硬币）
    - 无法凑成：返回 -1

    【设计考量】
    - dp[j] = 凑成金额 j 的最少硬币数
    - 初始化为 amount + 1（当作正无穷，因为最多用 amount 枚 1 元硬币）
    - 正序枚举（完全背包）

    时间复杂度：O(n * amount)
    空间复杂度：O(amount)
    """
    INF = amount + 1
    dp = [INF] * (amount + 1)
    dp[0] = 0  # 凑成 0 元需要 0 枚硬币

    for coin in coins:
        for j in range(coin, amount + 1):
            dp[j] = min(dp[j], dp[j - coin] + 1)

    return dp[amount] if dp[amount] < INF else -1


print(coin_change([1, 5, 10, 25], 30))  # 2 (25+5)
print(coin_change([1, 3, 4], 6))         # 2 (3+3)
print(coin_change([2], 3))               # -1
```

```cpp
int coinChange(vector<int>& coins, int amount) {
    int INF = amount + 1;
    vector<int> dp(amount + 1, INF);
    dp[0] = 0;

    for (int coin : coins) {
        for (int j = coin; j <= amount; j++) {
            if (dp[j - coin] < INF) {
                dp[j] = min(dp[j], dp[j - coin] + 1);
            }
        }
    }
    return dp[amount] >= INF ? -1 : dp[amount];
}
```

---

### 28.3.8 分割等和子集（LeetCode #416）：背包判断型

**问题**：给正整数数组 `nums`，判断是否能将其分割为两个和相等的子集。

**关键转化**：若总和为 `S`，则问题变成"能否从 `nums` 中选出若干元素，使其和恰好为 `S/2`"——这是 **0/1 背包判断型**。

```python
def can_partition(nums: list[int]) -> bool:
    """
    分割等和子集（LeetCode #416）

    转化为 0/1 背包：是否能选出若干数字和为 total // 2？

    【Edge Cases】
    - total 为奇数：必然不可分，直接返回 False
    - nums 为空或长度为 1：返回 False（无法分成两个非空子集）

    【设计考量】
    - dp[j] = True 表示"能否从前 i 个数中选出若干，使和为 j"
    - 0/1 背包：倒序枚举
    - 一旦 dp[target] 变成 True，可提前返回
    """
    total = sum(nums)
    if total % 2 != 0:
        return False
    target = total // 2

    dp = [False] * (target + 1)
    dp[0] = True  # 选 0 个数，和为 0，可达

    for num in nums:
        # 倒序：0/1 背包
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]

    return dp[target]


print(can_partition([1, 5, 11, 5]))  # True  ([1,5,5] 和 [11])
print(can_partition([1, 2, 3, 5]))   # False
```

```cpp
bool canPartition(vector<int>& nums) {
    int total = accumulate(nums.begin(), nums.end(), 0);
    if (total % 2 != 0) return false;
    int target = total / 2;

    vector<bool> dp(target + 1, false);
    dp[0] = true;

    for (int num : nums) {
        for (int j = target; j >= num; j--) {
            dp[j] = dp[j] || dp[j - num];
        }
    }
    return dp[target];
}
```

---

## 28.4 DP 状态设计方法论

### 28.4.1 "最后一步"思维法

**这是 DP 设计的核心思维工具。**

设计 $dp[i][j]$ 的转移时，不问"前面我经历了什么"，而问"最后一步我做了什么"。

- **爬楼梯（#70）**：到达第 $i$ 阶，最后一步要么来自 $i-1$，要么来自 $i-2$。
  - $dp[i] = dp[i-1] + dp[i-2]$

- **编辑距离**：将 $s[0..i-1]$ 转成 $t[0..j-1]$，最后一步要么是删除了 $s[i-1]$，要么插入了 $t[j-1]$，要么替换了最后一对字符。

- **背包**：考虑第 $i$ 件物品，最后一步要么选了它，要么没选。

**模板**：找"最后一步的所有可能"→ 每种可能对应一个更小规模的子问题 → 取所有选项的最优 → 写出转移方程。

---

### 28.4.2 如何精确定义 dp[i] 或 dp[i][j]

DP 设计失败的最常见原因不是转移写错，而是**状态定义不够精确**。

**三种主流状态定义类型**：

| 类型 | 含义 | 典型题目 |
|---|---|---|
| **前缀型** $dp[i]$ | 以 $i$ 结尾（或到 $i$ 为止）的最优值 | LIS、爬楼梯 |
| **区间型** $dp[l][r]$ | 区间 $[l, r]$ 的最优值 | 矩阵链乘（Chapter 29）|
| **状态机型** $dp[i][k]$ | 到 $i$ 为止、处于状态 $k$ 的最优值 | 股票问题、背包变体 |

**精确定义的要求**：

1. 说清楚是"前 $i$ 个"还是"以 $i$ 结尾"（两者不同！）；
2. 说清楚是"最长/最大/最小/计数；
3. 说清楚子问题的约束（如"恰好装满"vs"最多装 W"）。

---

### 28.4.3 滚动数组（Rolling Array）：空间压缩

很多 DP 的转移方程中，$dp[i][j]$ 只依赖 $dp[i-1][...]$，即只需上一行。这时可把二维数组压缩为一维，节省空间。

**LCS 空间优化**（从 $O(mn)$ 到 $O(\min(m, n))$）：

```python
def lcs_space_opt(X: str, Y: str) -> int:
    """
    LCS 滚动数组优化：O(min(m,n)) 空间

    【核心思路】
    dp[i][j] 只用到 dp[i-1][j] 和 dp[i-1][j-1]
    用两行交替，或一行 + 一个临时变量记录左上角

    【内存易错点】
    - 覆盖 dp[j-1] 之前必须先把它存到 prev（它是下一步的"左上角"）
    """
    m, n = len(X), len(Y)
    # 保证 Y 更短（优化空间）
    if m < n:
        X, Y = Y, X
        m, n = n, m

    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0  # prev = dp[i-1][j-1]，即左上角
        for j in range(1, n + 1):
            temp = dp[j]  # 保存旧值（将成为下一轮的 prev）
            if X[i-1] == Y[j-1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = temp
    return dp[n]
```

```cpp
int lcs_space_opt(const string& X, const string& Y) {
    int m = X.size(), n = Y.size();
    if (m < n) { return lcs_space_opt(Y, X); }
    vector<int> dp(n + 1, 0);
    for (int i = 1; i <= m; i++) {
        int prev = 0;
        for (int j = 1; j <= n; j++) {
            int temp = dp[j];
            if (X[i-1] == Y[j-1]) dp[j] = prev + 1;
            else dp[j] = max(dp[j], dp[j-1]);
            prev = temp;
        }
    }
    return dp[n];
}
```

---

### 28.4.4 DP 枚举顺序的重要性

**枚举顺序决定了"计算 $dp[i][j]$ 时依赖的格子是否已经有正确值"。**

**LCS**：外层 $i$ 从 1 到 $m$，内层 $j$ 从 1 到 $n$——因为 $dp[i][j]$ 依赖 $dp[i-1][j-1]$、$dp[i-1][j]$、$dp[i][j-1]$，三者都在 $i$ 行（已算）或 $i$ 列之前（已算）。

**0/1 背包一维滚动**：必须 **$j$ 从大到小**（$W$ 到 $w_i$）。若从小到大，$dp[j - w_i]$ 会是本轮（已经"选过"第 $i$ 件）的结果，导致同一件物品可能被反复选取。

**完全背包一维滚动**：必须 **$j$ 从小到大**。这样 $dp[j - w_i]$ 是本轮更新后的值，允许同一物品多次取。

| 问题 | 枚举顺序 | 原因 |
|---|---|---|
| 0/1 背包（一维） | 容量从大到小 | 防止同一件物品被重复选 |
| 完全背包（一维） | 容量从小到大 | 允许同一件物品重复选 |
| LCS（标准二维） | $i$ 正序，$j$ 正序 | 依赖左上、上方、左方 |
| 区间 DP | 先枚举区间长 $len$，再枚举左端点 | 短区间先算完 |

---

### 28.4.5 常见错误汇总

> ⚠️ 把这些错误都踩一遍，你的 DP 就入门了。

| 错误类型 | 描述 | 典型后果 |
|---|---|---|
| **初始化错误** | `dp[0][j]` 没初始化为 $j$（编辑距离） | 所有结果偏小 |
| **答案取错** | LIS 取 `dp[-1]` 而非 `max(dp)` | 只有末尾结尾的子序列被计入 |
| **枚举方向错** | 0/1 背包从小到大枚举容量 | 同一物品被无限次选取 |
| **边界遗漏** | 转移时没判断 `j - w[i] >= 0` | 数组越界 |
| **状态定义模糊** | "前 $i$ 个"和"以 $i$ 结尾"混用 | 转移方程自相矛盾 |
| **答案所在位置** | 搞错答案在哪个 $dp[i][j]$ | 返回错误格子 |

---

## 28.5 本章复杂度与总结

| 问题 | 状态数 | 转移代价 | 总时间 | 空间（优化后） |
|---|---|---|---|---|
| Fibonacci DP | $O(n)$ | $O(1)$ | $O(n)$ | $O(1)$ |
| Fibonacci 矩阵快速幂 | — | — | $O(\log n)$ | $O(1)$ |
| LCS | $O(mn)$ | $O(1)$ | $O(mn)$ | $O(\min(m,n))$ |
| 编辑距离 | $O(mn)$ | $O(1)$ | $O(mn)$ | $O(\min(m,n))$ |
| LIS ($O(n^2)$) | $O(n)$ | $O(n)$ | $O(n^2)$ | $O(n)$ |
| LIS ($O(n \log n)$) | $O(n)$ | $O(\log n)$ | $O(n \log n)$ | $O(n)$ |
| 0/1 背包 | $O(nW)$ | $O(1)$ | $O(nW)$ | $O(W)$ |
| 完全背包 | $O(nW)$ | $O(1)$ | $O(nW)$ | $O(W)$ |

---

## 本章常见错误与调试技巧

1. **打印 DP 表找规律**：写出一个小例子（字符串长度 4-5），手工填一遍 DP 表，然后用代码打印，对比是否一致。

2. **对拍**：用暴力递归（无缓存）对拍 DP 的输出。两者不一致，必有 Bug。

3. **先写二维再优化**：不要一开始就写滚动数组版本。先用二维数组写正确，再做空间压缩。

4. **初始化全检查**：把 Base Cases 单独列出来，逐一确认是否符合状态定义。

---

## 面试高频问题

1. **编辑距离的状态转移方程**（LeetCode #72）

2. **LIS 的 $O(n \log n)$ 解法**：描述 `tails` 数组的含义，解释为什么它始终有序，以及为什么它的长度等于 LIS 长度。

3. **0/1 背包与完全背包的主要区别**：枚举顺序（倒序 vs 正序）的原因能说清楚。

4. **什么是最优子结构？举一个没有最优子结构的例子**（最长简单路径）。

5. **Top-Down 和 Bottom-Up 各有什么优劣**？什么情况下只能用 Bottom-Up？

---

## 练习与思考题

1. LCS 与编辑距离（仅含插入/删除）的关系：若 $\text{LCS}(s, t) = l$，仅含插入/删除的编辑距离是多少？（提示：用 $|s| + |t| - 2l$ 推导）

2. 实现 `coin_change_ways`（LeetCode #518 零钱兑换 II）：求凑成 `amount` 的硬币组合方式总数。与 #322 的区别是什么？

3. 试证明：`tails` 数组在每次更新后仍然严格递增。（提示：反证法）

4. 将编辑距离代码修改为可以输出最短编辑序列（哪些字符在哪些位置被插入/删除/替换）。

5. 给定 `k` 种面值的硬币 `[c_1, c_2, ..., c_k]`，判断该系统是否"规范"（即贪心算法总是最优）。（提示：参考 Pearson 1994 的充要条件）

---

## 扩展阅读

- CLRS 第4版 Chapter 14（Dynamic Programming）
- MIT 6.006 Lecture 18–20：DP 入门、LCS、最短路 DP
- Sedgewick《Algorithms》3.5 节
- [LeetCode 题解：DP 专题](https://leetcode.com/tag/dynamic-programming/)
- [OI Wiki：动态规划](https://oi-wiki.org/dp/)
- Skiena《The Algorithm Design Manual》第8章

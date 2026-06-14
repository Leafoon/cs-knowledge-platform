# Chapter 34: 高级字符串技术

> **学习目标**：掌握 Manacher 算法的 O(n) 线性最长回文子串求解；熟练运用多项式滚动哈希与双模哈希进行 O(1) 子串比较；理解 Z 函数的 [l, r] 窗口复用技巧及其与 KMP π 数组的等价关系；综合运用上述工具解决回文计数、最长重复子串、字符串周期检测等高频竞赛与面试题。

---

## 章节导读

经过 Chapter 31（KMP / Rabin-Karp）、Chapter 32（Trie / AC 自动机）、Chapter 33（后缀数组 / SAM），你已经掌握了字符串世界的大部分"重武器"。本章是字符串专题的**最后一站**，聚焦三个精巧而实用的工具：

| 工具 | 解决的核心问题 | 时间复杂度 |
|---|---|---|
| **Manacher 算法** | 最长回文子串 | $O(n)$ |
| **字符串哈希** | 任意子串的 O(1) 比较 | 预处理 $O(n)$，查询 $O(1)$ |
| **Z 函数** | 字符串匹配、周期检测 | 构造 $O(n)$，查询 $O(1)$ |

这三种技术有一个共同的哲学：**"用已计算的信息，跳过不必要的重复计算"**，这与 KMP 的核心思想一脉相承。

---

## 34.1 Manacher 算法（最长回文子串 $O(n)$）

### 34.1.1 问题引入与朴素解法回顾

**回文串**（Palindrome）是正读反读都相同的字符串，例如 `"racecar"`、`"abba"`。

**最长回文子串**（Longest Palindromic Substring）是 LeetCode #5，是每位工程师必须熟悉的经典问题。

**朴素解法——中心扩散（Center Expansion）**：对每个位置 $i$ 作为回文中心，向两侧扩展，单次扩展最坏 $O(n)$，共 $n$ 个中心，总复杂度 $O(n^2)$（需要分奇偶讨论）。

```python
def longest_palindrome_naive(s: str) -> str:
    n = len(s)
    best_start, best_len = 0, 1

    def expand(l: int, r: int) -> tuple[int, int]:
        # 从中心 [l, r] 向外扩展，返回最终左右边界
        while l >= 0 and r < n and s[l] == s[r]:
            l -= 1
            r += 1
        # 退出时越界，回退一步
        return l + 1, r - 1

    for i in range(n):
        # 奇数长度回文（单个字符为中心）
        l, r = expand(i, i)
        if r - l + 1 > best_len:
            best_start, best_len = l, r - l + 1
        # 偶数长度回文（两个字符之间为中心）
        if i + 1 < n:
            l, r = expand(i, i + 1)
            if r - l + 1 > best_len:
                best_start, best_len = l, r - l + 1

    return s[best_start : best_start + best_len]
```

```cpp
#include <string>
#include <algorithm>
using namespace std;

string longestPalindrome_naive(const string& s) {
    int n = s.size();
    int best_start = 0, best_len = 1;

    auto expand = [&](int l, int r) -> pair<int,int> {
        while (l >= 0 && r < n && s[l] == s[r]) { l--; r++; }
        return {l + 1, r - 1};  // 退出后越界，回退
    };

    for (int i = 0; i < n; i++) {
        // 奇数长度
        auto [l1, r1] = expand(i, i);
        if (r1 - l1 + 1 > best_len) { best_start = l1; best_len = r1 - l1 + 1; }
        // 偶数长度
        if (i + 1 < n) {
            auto [l2, r2] = expand(i, i + 1);
            if (r2 - l2 + 1 > best_len) { best_start = l2; best_len = r2 - l2 + 1; }
        }
    }
    return s.substr(best_start, best_len);
}
```

朴素解法思路清晰，但 $O(n^2)$ 在 $n = 10^5$ 时会超时。Manacher 算法将其优化到 $O(n)$。

### 34.1.2 奇偶统一：插入 `#` 字符

Manacher 的第一个关键技巧是**通过插入分隔符 `#`，将所有回文统一为奇数长度**（以某个位置为中心）。

**转换规则**：
- 在字符串首尾和每两个字符之间插入 `#`
- 还可以在最外层加 `^` 和 `$` 防止越界（某些实现）

```
原始字符串：  a  b  b  a
              0  1  2  3

插入 # 后：   # a # b # b # a #
              0 1 2 3 4 5 6 7 8
```

一般形式：若原始字符串长度为 $n$，新字符串长度为 $2n+1$。

新旧字符串之间的下标关系：
- 新字符串中位置 $i$ 对应原始字符串位置 $\lfloor i/2 \rfloor$
- 奇数位置（`#` 字符）是"虚拟中心"，对应偶数长度回文
- 偶数位置是原始字符，对应奇数长度回文

**关键优势**：插入后，每个回文子串在新字符串中都有一个**实际的字符中心**，无需区分奇偶。

### 34.1.3 回文半径数组 $P[i]$

对新字符串 $t$（长度 $m = 2n+1$），定义**回文半径数组 $P[i]$**：

$$P[i] = \text{以 } t[i] \text{ 为中心的最长回文子串的半径}$$

即 $t[i-P[i]..i+P[i]]$ 是以 $t[i]$ 为中心的最长回文。

**示例**：

```
原始字符串  s = "abba"
扩充字符串  t = #a#b#b#a#

i:    0  1  2  3  4  5  6  7  8
t[i]: #  a  #  b  #  b  #  a  #
P[i]: 0  1  0  3  0  3  0  1  0
```

验证 $P[3] = 3$：以 $t[3] = \text{`b'}$ 为中心，向外扩展 3 格，得到 `t[0..6] = #a#b#b#a#`，去掉 `#` 是 `"abba"`，确实回文。

**从 $P[i]$ 还原原始答案**：
- 原始最长回文长度 = $\max(P[i])$（半径值直接等于原始中心扩展加 `#` 后的长度。因为扩展 1 格 = 原始 1 个字符 + 1 个 `#`。）
- 原始起始位置（0-indexed）= $(i - P[i]) / 2$

直觉验证：上面例子中 $\max(P) = 3$（最长回文 "abba" 长度 4，但半径 3 是新字符串的半径，对应原始长度确实是 $P[i] = 3$，因为每步 `#` 是虚拟字符）。

> 实际上更精确：$P[i]$ 恰好等于以该中心为中心的**原始回文长度**（奇数长度 = `#` 不计；偶数长度同理）。这是 `#` 插入的优美之处。

### 34.1.4 核心技巧：center 与 max\_right 的对称复用

Manacher 的 $O(n)$ 关键在于维护两个指针：
- `max_right`：当前所有已知回文中，**最右延伸端点**的位置
- `center`：取得上述 `max_right` 的回文中心

**关键观察**：当我们计算 $P[i]$（$i < \text{max\_right}$）时，$i$ 关于 `center` 的**对称点** $j = 2 \cdot \text{center} - i$ 的 $P[j]$ 已知。

利用回文对称性：

$$P[i] \geq \min(P[j],\ \text{max\_right} - i)$$

- 如果 $P[j] < \text{max\_right} - i$：$P[j]$ 范围完全在当前最大回文内 → $P[i] = P[j]$，无需扩展
- 如果 $P[j] \geq \text{max\_right} - i$：提供下界，但 `max_right` 之外未知，**需要继续扩展**

每次扩展都将 `max_right` 右移，而 `max_right` 最多到 $m$（整个字符串末尾），因此**整个算法的扩展操作总次数** $O(m) = O(n)$。

用一句话描述算法流程：
> 从左到右扫描新字符串，用（center, max_right）提供的对称信息给出 $P[i]$ 的初始值，然后尝试继续扩展；一旦扩展成功，更新 center 和 max_right。

<div data-component="ManacherPalindromeViz"></div>

### 34.1.5 Manacher 完整实现

```python
def manacher(s: str) -> list[int]:
    """
    构建 Manacher 回文半径数组。
    输入：原始字符串 s
    输出：扩充字符串 t 的回文半径数组 P

    关键思路：
    1. 将 s 扩充为 t = #s[0]#s[1]#...#s[n-1]#
    2. 维护 (center, max_right)，利用对称性复用已知值
    3. max_right 单调右移，确保 O(n)
    """
    # 构建扩充字符串，例如 "abc" -> "#a#b#c#"
    t = '#' + '#'.join(s) + '#'
    m = len(t)
    P = [0] * m
    center = 0    # 当前最大回文的中心
    max_right = 0 # 当前最大回文的右边界（不含）

    for i in range(m):
        if i < max_right:
            # 利用对称性：j 是 i 关于 center 的镜像
            j = 2 * center - i
            # 取对称点 P[j] 与到 max_right 距离的最小值作为初始半径
            P[i] = min(P[j], max_right - i)
        # 尝试继续扩展（越界时 s[?] != s[?] 保证退出）
        while i - P[i] - 1 >= 0 and i + P[i] + 1 < m and t[i - P[i] - 1] == t[i + P[i] + 1]:
            P[i] += 1
        # 更新 center 和 max_right
        if i + P[i] > max_right:
            max_right = i + P[i]
            center = i

    return P

def longest_palindrome_manacher(s: str) -> str:
    """
    利用上面的 manacher() 函数求最长回文子串。
    时间复杂度：O(n)
    空间复杂度：O(n)
    """
    if not s:
        return ""
    P = manacher(s)
    # 找最大半径及其位置
    # max_idx 是扩充字符串中的下标
    max_len, max_idx = max((v, i) for i, v in enumerate(P))
    # 原始字符串中的起点
    start = (max_idx - max_len) // 2
    return s[start : start + max_len]

# ---- 测试 ----
# 示例 1
print(longest_palindrome_manacher("babad"))    # "bab" 或 "aba"
# 示例 2
print(longest_palindrome_manacher("cbbd"))     # "bb"
# 示例 3
print(longest_palindrome_manacher("racecar"))  # "racecar"
# 示例 4
print(longest_palindrome_manacher("abba"))     # "abba"
```

```cpp
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
using namespace std;

// 构建 Manacher 回文半径数组
// 返回值：扩充字符串 t 的回文半径数组 P（长度为 2n+1）
vector<int> manacher(const string& s) {
    // 构建扩充字符串 t = "#s[0]#s[1]#...#s[n-1]#"
    string t = "#";
    for (char c : s) { t += c; t += '#'; }
    int m = t.size();
    vector<int> P(m, 0);
    int center = 0, max_right = 0;

    for (int i = 0; i < m; i++) {
        if (i < max_right) {
            int j = 2 * center - i;           // 关于 center 的对称点
            P[i] = min(P[j], max_right - i);  // 对称复用，取最小防止越界
        }
        // 尝试继续扩展
        while (i - P[i] - 1 >= 0 && i + P[i] + 1 < m &&
               t[i - P[i] - 1] == t[i + P[i] + 1]) {
            P[i]++;
        }
        // 更新最优中心
        if (i + P[i] > max_right) {
            max_right = i + P[i];
            center = i;
        }
    }
    return P;
}

// 求最长回文子串
string longestPalindrome(const string& s) {
    if (s.empty()) return "";
    vector<int> P = manacher(s);
    // 找最大半径
    int max_idx = max_element(P.begin(), P.end()) - P.begin();
    int max_len = P[max_idx];
    int start = (max_idx - max_len) / 2;
    return s.substr(start, max_len);
}

// ---- 扩展：统计所有回文子串数量 (LeetCode #647) ----
// 每个 P[i] > 0 的位置，以 t[i] 为中心的回文子串数量 = (P[i] + 1) / 2
// （因为 P[i] 每增加 2，原始字符串里多一个回文；P[i]=1 贡献 1 个）
int countSubstrings(const string& s) {
    vector<int> P = manacher(s);
    int count = 0;
    for (int p : P) count += (p + 1) / 2;
    return count;
}
```

**预期输出**：
```
bab
bb
racecar
abba
```

**时间与空间复杂度**：
- 时间：$O(n)$。`max_right` 指针单调向右移动，总扩展次数 $\leq m = 2n+1$。
- 空间：$O(n)$，用于存储扩充字符串和 $P$ 数组。

### 34.1.6 线性时间证明（势能分析）

定义势能 $\Phi = \text{max\_right}$（当前最大回文右端点坐标）。

- 每次**扩展操作**，`max_right` 至少右移 1；
- 每次**复用操作**（$P[i] = \min(P[j], \text{max\_right} - i)$），无需扩展，时间 $O(1)$；
- `max_right` 从 0 增长到最多 $m = 2n+1$，因此**扩展操作总次数** $\leq 2n+1 = O(n)$；
- 每个位置 $i$ 的非扩展部分代价 $O(1)$，共 $m = O(n)$ 个位置。

综合：算法总时间 $O(n)$。

### 34.1.7 常见陷阱与调试技巧

**陷阱 1：下标转换出错**

插入 `#` 后，扩充字符串的下标 $i$ 对应原始字符串的位置：
- 原始起点 = $(i - P[i]) / 2$（整数除法）
- 原始长度 = $P[i]$（等于回文在原始字符串中的长度）

**陷阱 2：max_right 的边界定义**

不同实现中 `max_right` 含义不同——有的是**闭区间**（最右字符的下标），有的是**开区间**（最右字符的下标 + 1）。  
本文使用**闭区间**：`max_right = i + P[i]`（中心 + 半径 = 最右端点包括在内）。

如果你看到别人代码里 `i < max_right + 1` 或 `i <= max_right`，说明的是同一件事，只是定义稍有不同。

**陷阱 3：超短字符串**

$n=0$ 时需要特判；$n=1$ 时 $P = [0]$，结果是 `s[0]`，长度 1。

---

## 34.2 字符串哈希（String Hashing）

### 34.2.1 动机：为什么需要字符串哈希？

字符串比较的朴素做法是逐字符对比，时间 $O(m)$（$m$ 为字符串长度）。对于很多字符串题，我们需要频繁比较**任意区间的子串是否相等**——如果每次都 $O(m)$，总复杂度会爆炸。

**字符串哈希的目标**：通过 $O(n)$ 预处理，实现后续每次子串相等性判断为 $O(1)$。

**生活类比**：给每本书计算一个"指纹码"（ISBN）。如果两本书的 ISBN 不同，它们一定不同；如果 ISBN 相同，大概率相同（极小概率碰撞）。字符串哈希就是给每个子串计算"指纹码"。

### 34.2.2 多项式哈希的定义

**多项式哈希（Polynomial Rolling Hash）** 将字符串视为一个以 `base` 为底的多项式：

$$H(s) = s[0] \cdot \text{base}^{n-1} + s[1] \cdot \text{base}^{n-2} + \cdots + s[n-1] \cdot \text{base}^0 \pmod{q}$$

其中 `base` 和 `q` 是事先选定的参数（`base` 常用 31、131、1e9+7；`q` 常用大质数如 $10^9+7$，$10^9+9$）。

**前缀哈希数组** $H$：定义 $H[i] = H(s[0..i-1])$（前 $i$ 个字符的哈希值），$H[0] = 0$。

递推关系：

$$H[i] = H[i-1] \cdot \text{base} + s[i-1] \pmod{q}$$

**O(1) 区间哈希查询**：设子串 $s[l..r]$（0-indexed），其哈希值为：

$$H(l, r) = H[r+1] - H[l] \cdot \text{base}^{r-l+1} \pmod{q}$$

这与前缀和的思想完全一致：$H[r+1]$ 包含了 $l$ 之前的"高位贡献"，减去后只剩 $s[l..r]$ 的贡献。

### 34.2.3 单模哈希的实现

```python
class StringHasher:
    """
    多项式哈希预处理与 O(1) 区间查询。
    设计选择：
    - base = 131（常用于大小写字母 + 数字场景）
    - mod = 10^9 + 7（大质数，足够随机）
    - 字符映射：直接使用 ASCII 值 ord(c)
    """
    def __init__(self, s: str, base: int = 131, mod: int = 10**9 + 7):
        self.n = len(s)
        self.mod = mod
        self.base = base

        # 预计算前缀哈希：h[i] = hash(s[0..i-1])
        # h[0] = 0（空字符串）
        self.h = [0] * (self.n + 1)
        for i, c in enumerate(s):
            self.h[i + 1] = (self.h[i] * base + ord(c)) % mod

        # 预计算 base 的幂次：pw[i] = base^i mod q
        self.pw = [1] * (self.n + 1)
        for i in range(1, self.n + 1):
            self.pw[i] = self.pw[i - 1] * base % mod

    def query(self, l: int, r: int) -> int:
        """
        查询 s[l..r]（0-indexed，闭区间）的哈希值。
        公式：H[r+1] - H[l] * base^(r-l+1)
        注意：结果可能为负（Python 取模自动处理；C++ 需要加 mod）
        """
        length = r - l + 1
        return (self.h[r + 1] - self.h[l] * self.pw[length]) % self.mod

    def equal(self, l1: int, r1: int, l2: int, r2: int) -> bool:
        """
        O(1) 判断 s[l1..r1] 与 s[l2..r2] 是否相等。
        注意：哈希相等不一定字符串相等（有碰撞概率），但实践中足够可靠。
        """
        return self.query(l1, r1) == self.query(l2, r2)

# ---- 测试：最长重复子串（LeetCode #1044，哈希 + 二分） ----
def longestDupSubstring(s: str) -> str:
    """
    思路：二分答案字符串长度 mid，用哈希 O(n) 检查是否存在长度为 mid 的重复子串。
    总复杂度：O(n log n)
    """
    n = len(s)
    hasher = StringHasher(s)

    def check(length: int) -> int:
        """返回第一个出现重复的子串起始下标，若无返回 -1"""
        seen = {}  # hash -> 起始下标
        for i in range(n - length + 1):
            h = hasher.query(i, i + length - 1)
            if h in seen:
                return seen[h]  # 找到重复
            seen[h] = i
        return -1

    lo, hi = 1, n - 1
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        idx = check(mid)
        if idx != -1:
            best = idx
            lo = mid + 1
        else:
            hi = mid - 1

    length = lo - 1
    return s[best : best + length] if length > 0 else ""

# 测试
print(longestDupSubstring("banana"))     # "ana"
print(longestDupSubstring("abcd"))       # ""（无重复）
```

```cpp
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
using namespace std;

class StringHasher {
    long long mod, base;
    vector<long long> h, pw;

public:
    StringHasher(const string& s, long long base = 131, long long mod = 1e9 + 7)
        : mod(mod), base(base), h(s.size() + 1, 0), pw(s.size() + 1, 1) {
        for (int i = 0; i < (int)s.size(); i++) {
            // 前缀哈希递推：h[i+1] = h[i] * base + s[i]
            h[i + 1] = (h[i] * base + s[i]) % mod;
        }
        for (int i = 1; i <= (int)s.size(); i++) {
            pw[i] = pw[i - 1] * base % mod;
        }
    }

    // 查询 s[l..r]（0-indexed 闭区间）的哈希值
    long long query(int l, int r) const {
        int len = r - l + 1;
        // (h[r+1] - h[l] * pw[len]) 可能为负，需要加 mod
        return (h[r + 1] - h[l] * pw[len] % mod + mod) % mod;
    }

    // O(1) 判断 s[l1..r1] == s[l2..r2]
    bool equal(int l1, int r1, int l2, int r2) const {
        return (r1 - l1 == r2 - l2) && (query(l1, r1) == query(l2, r2));
    }
};

// 最长重复子串（LeetCode #1044），时间 O(n log n)
string longestDupSubstring(const string& s) {
    int n = s.size();
    StringHasher hasher(s);

    auto check = [&](int len) -> int {
        unordered_map<long long, int> seen;
        for (int i = 0; i + len <= n; i++) {
            long long h = hasher.query(i, i + len - 1);
            if (seen.count(h)) return seen[h];  // 碰撞则认为找到
            seen[h] = i;
        }
        return -1;
    };

    int lo = 1, hi = n - 1, best = 0;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int idx = check(mid);
        if (idx != -1) { best = idx; lo = mid + 1; }
        else hi = mid - 1;
    }
    return s.substr(best, lo - 1);
}
```

### 34.2.4 双模哈希：大幅降低碰撞概率

单模哈希在随机数据下碰撞概率约 $1/q \approx 10^{-9}$。但在 Hack 测试或数据量 $n = 10^6$ 时，期望碰撞次数 $\approx n^2 / q = 10^3$（构造攻击可让所有哈希碰撞）。

**双模哈希（Double Hashing）**：同时维护两个不同模数的哈希值，用 `(h1, h2)` 作为"复合指纹"。碰撞概率降至 $\approx 1/(q_1 \cdot q_2) \approx 10^{-18}$，实践中几乎不可能碰撞。

```python
class DoubleHasher:
    """双模哈希：同时用两个大质数，碰撞概率 ≈ 10^-18"""
    def __init__(self, s: str):
        self.h1 = StringHasher(s, base=131,  mod=10**9 + 7)
        self.h2 = StringHasher(s, base=137,  mod=10**9 + 9)

    def query(self, l: int, r: int) -> tuple[int, int]:
        return self.h1.query(l, r), self.h2.query(l, r)

    def equal(self, l1: int, r1: int, l2: int, r2: int) -> bool:
        return self.query(l1, r1) == self.query(l2, r2)
```

```cpp
struct DoubleHasher {
    StringHasher h1, h2;
    DoubleHasher(const string& s)
        : h1(s, 131, 1000000007LL),
          h2(s, 137, 1000000009LL) {}

    pair<long long, long long> query(int l, int r) const {
        return {h1.query(l, r), h2.query(l, r)};
    }

    bool equal(int l1, int r1, int l2, int r2) const {
        return h1.equal(l1, r1, l2, r2) && h2.equal(l1, r1, l2, r2);
    }
};
```

**竞赛建议**：
- 题目 $n \leq 10^5$：单模哈希通常够用。
- 题目 $n \geq 10^6$ 或有 Hack 机制：使用双模哈希。
- 参数选择：`base` 尽量避免 2 的幂（容易与特殊字符串碰撞），质数更安全。

### 34.2.5 哈希应用：O(1) 回文判断

巧妙地为字符串 $s$ 和其**反转** $\hat{s}$ 各建一个哈希器，则：

$$s[l..r] \text{ 是回文} \iff H_s(l, r) = H_{\hat{s}}(n-1-r,\ n-1-l)$$

```python
def count_palindromes_hash(s: str) -> int:
    """
    统计所有回文子串数量（LeetCode #647）。
    哈希 + 中心扩散：对每个中心枚举，用二分找最大回文半径，再用哈希 O(1) 验证。
    总复杂度：O(n log n)（vs Manacher O(n)）
    """
    n = len(s)
    h_fwd = StringHasher(s)
    h_rev = StringHasher(s[::-1])

    def is_pal(l: int, r: int) -> bool:
        # s[l..r] 是否回文 = 正向哈希 == 反向哈希
        rev_l = n - 1 - r
        rev_r = n - 1 - l
        return h_fwd.query(l, r) == h_rev.query(rev_l, rev_r)

    count = 0
    for center in range(2 * n - 1):       # 枚举每个中心
        l0 = center // 2
        r0 = l0 + (center % 2)             # 奇 center -> center/2; 偶 -> 两字符
        # 二分：找最大半径 R 使 [l0-R, r0+R] 是回文
        lo, hi = 0, min(l0, n - 1 - r0)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if is_pal(l0 - mid, r0 + mid):
                lo = mid
            else:
                hi = mid - 1
        count += lo + 1                    # 半径 0..lo 每个都是一个回文
    return count
```

<div data-component="StringHashingDemo"></div>

---

## 34.3 Z 函数（Z-Function）

### 34.3.1 Z 数组的定义

**Z 函数**（Z-function, also called Z-array）是一个长度为 $n$ 的数组，其中：

$$Z[i] = \text{从位置 } i \text{ 开始的后缀 } s[i..n-1] \text{ 与整个字符串 } s \text{ 的最长公共前缀（LCP）长度}$$

通俗地说：$Z[i]$ 等于"从 $i$ 号位置开始，往右扫，能和字符串开头匹配多少个字符"。

**示例**：

```
s    =  a  a  b  a  a  b  a
下标:   0  1  2  3  4  5  6

计算方式（从 1 开始，Z[0] 通常置 0 或 n）：
Z[0] = 0（定义为 0 或 n，避免平凡）
Z[1] = 1  → "a", "a" 匹配 1 位（s[1]='a'=s[0], s[2]='b'≠s[1]）
Z[2] = 0  → "b", "a" 第 1 位不同
Z[3] = 4  → "aaba", "aab" → 匹配 "aab" 共 3 位... 再比 s[6]='a'=s[3]='a' 继续
              实际：s[3..6]="aaba", s[0..3]="aaba" → 匹配 4 位！
Z[4] = 1  → "a", "a" 匹配 1 位（s[5]='b'≠s[1]）
Z[5] = 0  → "b"≠"a"
Z[6] = 1  → "a", "a" 匹配 1 位（越界）

Z = [0, 1, 0, 4, 1, 0, 1]
```

**与 KMP π 数组的对比**：

| 特性 | KMP π 数组 $\pi[i]$ | Z 数组 $Z[i]$ |
|---|---|---|
| 定义 | $s[0..\pi[i]-1] = s[i-\pi[i]+1..i]$（最长真前缀=真后缀） | $s[0..Z[i]-1] = s[i..i+Z[i]-1]$（最长公共前缀） |
| 计算方向 | 从右往左（后缀角度） | 从左往右（前缀角度） |
| 字符串匹配 | 构造 P#T，查 π | 构造 P#T，查 Z |
| 等价性 | 两者可 O(n) 互相转化 | |

### 34.3.2 O(n) 构造：维护 [l, r] 最右匹配窗口

Z 算法的关键也是维护一对指针 `[l, r]`，含义是：**当前为止所有已计算的 Z[i] 中，最右延伸端点最远的那个区间 $[l, r]$**，满足 $s[l..r] = s[0..r-l]$。

**计算 $Z[i]$（$i > 0$）的步骤**：

1. 如果 $i \leq r$（$i$ 在当前已知窗口内）：
   - 令 $j = i - l$（$i$ 对应的"公共前缀"中的映射位置）
   - $Z[i] \geq \min(Z[j], r - i + 1)$（来自对称性）
   - 然后以此为初始值，继续向右扩展

2. 如果 $i > r$（$i$ 在窗口外）：
   - 直接从 0 开始扩展

3. 每次扩展成功时，若 $i + Z[i] - 1 > r$，更新 `l = i`，`r = i + Z[i] - 1`

`r` 单调右移（最多到 $n-1$），故扩展次数总计 $O(n)$。

**Manacher 与 Z 算法的共同之处**：

两者都用一个"**已知的最右端点**"来为新位置提供初始值，从而避免重复扩展，实现线性时间。这是一种被称为"势函数分析 / 摊销线性"的经典思想。

```python
def z_function(s: str) -> list[int]:
    """
    O(n) 构造 Z 函数数组。
    Z[i] = s[i:] 与 s 的最长公共前缀长度。
    Z[0] 定义为 0（也可以定义为 n，取决于应用场景）。

    关键思路：
    - 维护 [l, r]：当前最右延伸的匹配窗口，满足 s[l..r] = s[0..r-l]
    - 对 i 在窗口内：利用对称性 Z[i] >= min(Z[i-l], r-i+1)，然后继续扩
    - 每次扩展都将 r 向右推进，总扩展次数 O(n)
    """
    n = len(s)
    Z = [0] * n
    Z[0] = 0             # 按定义，Z[0] 置 0（避免与全串匹配造成歧义）
    l, r = 0, 0          # 当前最右匹配窗口 [l, r]（s[l..r] = s[0..r-l]）

    for i in range(1, n):
        if i <= r:
            # i 在窗口内，利用对称复用 Z[i - l]
            Z[i] = min(Z[i - l], r - i + 1)
        # 尝试继续扩展
        while i + Z[i] < n and s[Z[i]] == s[i + Z[i]]:
            Z[i] += 1
        # 如果扩展超出右边界，更新 [l, r]
        if i + Z[i] - 1 > r:
            l, r = i, i + Z[i] - 1

    return Z

# ---- 测试 ----
print(z_function("aabxaa"))    # [0, 1, 0, 0, 2, 1]  → 注意 Z[4]=2（"aa"=="aa"）
print(z_function("aaaaaa"))    # [0, 5, 4, 3, 2, 1]
print(z_function("abcabc"))    # [0, 0, 0, 3, 0, 0]
```

```cpp
#include <string>
#include <vector>
using namespace std;

// O(n) 构造 Z 函数数组
// Z[i] = s[i:] 与 s 的最长公共前缀长度，Z[0] = 0
vector<int> z_function(const string& s) {
    int n = s.size();
    vector<int> Z(n, 0);
    // [l, r]：当前最右匹配窗口，满足 s[l..r] = s[0..r-l]
    int l = 0, r = 0;

    for (int i = 1; i < n; i++) {
        if (i <= r) {
            // 利用对称性给初始值
            Z[i] = min(Z[i - l], r - i + 1);
        }
        // 尝试继续向右扩展
        while (i + Z[i] < n && s[Z[i]] == s[i + Z[i]]) {
            Z[i]++;
        }
        // 更新最右窗口
        if (i + Z[i] - 1 > r) {
            l = i;
            r = i + Z[i] - 1;
        }
    }
    return Z;
}
```

<div data-component="ZFunctionBuildViz"></div>

### 34.3.3 Z 函数与 KMP π 数组的等价性

**结论**：Z 数组和 KMP 的 π 数组（失败函数）携带等价的信息，可以 $O(n)$ 互相转化。

**从 Z 数组计算 π 数组**：

$$\pi[i + Z[i] - 1] = \max(\pi[i + Z[i] - 1],\ Z[i])$$

（对所有满足 $Z[i] > 0$ 的 $i$，更新 $s[i+Z[i]-1]$ 位置的 π 值）

```python
def z_to_pi(Z: list[int]) -> list[int]:
    """从 Z 数组 O(n) 计算 KMP π 数组"""
    n = len(Z)
    pi = [0] * n
    for i in range(1, n):
        if Z[i] > 0:
            # Z[i] > 0 表示 s[i..i+Z[i]-1] = s[0..Z[i]-1]
            # 等价于 s[i+Z[i]-1] 位置的最长公共前后缀长度 >= Z[i]
            pi[i + Z[i] - 1] = max(pi[i + Z[i] - 1], Z[i])
    # 从右往左传播（π 需要取最大可能值）
    for i in range(n - 2, 0, -1):
        pi[i] = max(pi[i], max(pi[i + 1] - 1, 0))
    return pi
```

**从 π 数组计算 Z 数组**（略，思路类似，逆向传播）。

在实际竞赛中，Z 函数和 KMP π 数组可以互相替代使用，选择哪个取决于个人习惯和题目语境。

### 34.3.4 Z 函数实现字符串匹配

**思路**：构建 `t = P + "#" + S`，其中 `#` 是一个不在 $P$ 和 $S$ 中出现的分隔符。

对 `t` 计算 Z 函数，对 `t` 的位置 $i > |P|$ 处：
- 若 $Z[i] = |P|$，说明 $S$ 在位置 $i - |P| - 1$（相对于 $S$）有一个完整匹配。

```python
def z_string_search(pattern: str, text: str) -> list[int]:
    """
    Z 函数实现字符串匹配。
    返回所有匹配起始位置（text 中，0-indexed）。
    时间复杂度：O(|P| + |T|)
    """
    m, n = len(pattern), len(text)
    if m == 0 or n < m:
        return []

    # 拼接：t = P + "#" + T
    # "#" 保证 Z[i] <= m（不会"穿越"分隔符）
    combined = pattern + "#" + text
    Z = z_function(combined)

    result = []
    for i in range(m + 1, len(combined)):
        if Z[i] == m:
            # 在 text 中的起始位置 = i - (m + 1)
            result.append(i - m - 1)
    return result

# ---- 测试 ----
print(z_string_search("ana", "banana"))      # [1, 3]
print(z_string_search("ab", "ababab"))       # [0, 2, 4]
print(z_string_search("xyz", "abcdef"))      # []
```

```cpp
#include <string>
#include <vector>
using namespace std;

// Z 函数字符串匹配，返回 pattern 在 text 中所有匹配起始位置
vector<int> z_string_search(const string& pattern, const string& text) {
    int m = pattern.size(), n = text.size();
    if (m == 0 || n < m) return {};

    // 拼接 P + "#" + T，分隔符不属于字符集
    string combined = pattern + "#" + text;
    vector<int> Z = z_function(combined);

    vector<int> result;
    for (int i = m + 1; i < (int)combined.size(); i++) {
        if (Z[i] == m) {
            result.push_back(i - m - 1);  // text 中的起点
        }
    }
    return result;
}
```

### 34.3.5 Z 函数确定字符串最小周期

**字符串周期**：若 $s[i] = s[i \bmod p]$ 对所有 $i$ 成立（即字符串可以由长度为 $p$ 的前缀重复（或部分重复）而得），则称 $p$ 是 $s$ 的一个**周期**。

**最小周期 $p$ 的 Z 函数判断条件**：

$$p \text{ 是 } s \text{ 的最小周期} \iff Z[p] = n - p \text{ 且 } p \text{ 是最小满足条件的正整数}$$

换句话说：从位置 $p$ 开始的后缀与前缀能匹配 $n - p$ 位（即一直到末尾），说明 $s[p..n-1] = s[0..n-p-1]$，整个字符串就是前 $p$ 个字符的重复延伸。

```python
def min_period_z(s: str) -> int:
    """
    利用 Z 函数求字符串的最小周期 p。
    p 是最小正整数使得 s[i] = s[i mod p] 对所有 i 成立。

    等价条件：Z[p] = n - p。

    复杂度：O(n)
    """
    n = len(s)
    Z = z_function(s)
    for p in range(1, n):
        # Z[p] = n - p 表示 s[p..n-1] = s[0..n-p-1]
        # 且 p 整除 n（如果要求完整重复）
        if n % p == 0 and Z[p] == n - p:
            return p
    return n  # 最坏情况：本身就是最小周期

# ---- 测试 ----
print(min_period_z("abcabc"))    # 3 → "abc" 重复 2 次
print(min_period_z("aaaa"))      # 1 → "a" 重复 4 次
print(min_period_z("abab"))      # 2 → "ab" 重复 2 次
print(min_period_z("abcd"))      # 4 → 无重复，最小周期是自身
```

```cpp
// 求字符串最小周期（O(n)）
int min_period_z(const string& s) {
    int n = s.size();
    vector<int> Z = z_function(s);
    for (int p = 1; p < n; p++) {
        // Z[p] == n - p：从 p 开始的后缀能匹配到末尾
        // n % p == 0：p 整除 n，保证完整重复
        if (n % p == 0 && Z[p] == n - p) return p;
    }
    return n;
}
```

**对比 KMP 周期性质**：

| 方法 | 条件 |
|---|---|
| KMP π | 最小周期 $p = n - \pi[n-1]$（若 $p \mid n$，则完整重复；否则不完整） |
| Z 函数 | 最小完整周期 $p$ = 最小满足 $Z[p] = n - p$ 且 $p \mid n$ 的正整数 |

两者本质等价，选择任意一种来解题都可以。

---

## 34.4 字符串综合应用

### 34.4.1 KMP 与 Z 函数的周期应用（LeetCode #459）

**题目**：给定字符串 $s$，判断能否由其某个子串重复多次构成。

**KMP 解法**：$p = n - \pi[n-1]$，若 $p \mid n$ 则答案为"是"。

**Z 函数解法**：如上，找最小周期 $p$ 使得 $Z[p] = n - p$ 且 $p \mid n$。

```python
def repeated_substring_pattern(s: str) -> bool:
    """
    LeetCode #459：重复子串模式。
    方法 1：KMP π 数组
    如果 π[n-1] > 0 且 n % (n - π[n-1]) == 0，则存在重复子串。
    """
    n = len(s)
    pi = [0] * n
    for i in range(1, n):
        j = pi[i - 1]
        while j > 0 and s[i] != s[j]:
            j = pi[j - 1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j

    period = n - pi[n - 1]
    return period < n and n % period == 0

def repeated_substring_pattern_z(s: str) -> bool:
    """
    LeetCode #459：重复子串模式。
    方法 2：Z 函数
    找最小整除周期 p 使 Z[p] = n - p。
    """
    n = len(s)
    Z = z_function(s)
    for p in range(1, n):
        if n % p == 0 and Z[p] == n - p:
            return True
    return False

# 测试
print(repeated_substring_pattern("abab"))    # True  (ab × 2)
print(repeated_substring_pattern("aba"))     # False
print(repeated_substring_pattern("abcabcabcabc")) # True (abc × 4)
```

### 34.4.2 最小表示法（Booth's Algorithm）—— 字典序最小循环移位

**问题**：给定字符串 $s$，在其所有**循环移位**中找字典序最小的那个。

**循环移位**：例如 `"abcde"` 的所有移位为 `"abcde"`, `"bcdea"`, `"cdeab"`, `"deabc"`, `"eabcd"`。

**朴素方法**：对所有 $n$ 个循环移位排序，$O(n^2 \log n)$。

**Booth 算法**：$O(n)$，通过双指针技巧在字符串的"双倍字符串" $ss = s + s$ 上操作。

```python
def min_rotation(s: str) -> int:
    """
    Booth 算法：求字符串所有循环移位中字典序最小的起始位置。
    时间复杂度：O(n)
    核心思路：在 s+s 上用双指针比较，维护候选最小起始位置。
    """
    n = len(s)
    ss = s + s              # 双倍字符串
    i, j, k = 0, 1, 0      # i: 当前最优起始; j: 候选起始; k: 已匹配长度
    while i < n and j < n and k < n:
        if ss[i + k] == ss[j + k]:
            k += 1          # 当前位置相等，继续比较下一位
        elif ss[i + k] > ss[j + k]:
            # i 开头的移位 > j 开头的移位
            i = max(j, i + k + 1)  # 跳过所有已知劣于 j 的起始位置
            k = 0
        else:
            # j 开头的移位 > i 开头的移位（i 仍然更优）
            j = max(i, j + k + 1)
            k = 0
        if i == j:
            j += 1          # 避免 i == j 死循环
    best = min(i, j)
    return best

# ---- 测试 ----
s = "cbabc"
idx = min_rotation(s)
print(f"最小循环移位起始位置：{idx}")
print(f"最小循环移位：{(s + s)[idx:idx + len(s)]}")
# idx = 3，移位 = "bccba"... 验证所有：cbabc, babcc, abccb, bccba, ccbab
# 最小：abccb（idx=2） ← 实际结果取决于字符串

s2 = "dcba"
idx2 = min_rotation(s2)
print(f"{s2} 最小循环移位：{(s2 * 2)[idx2:idx2 + len(s2)]}")
# 移位：dcba, cbad, badc, adcb → 最小为 "adcb"（idx=3）
```

```cpp
// Booth 算法：O(n) 最小循环移位
int min_rotation(const string& s) {
    int n = s.size();
    string ss = s + s;
    int i = 0, j = 1, k = 0;
    while (i < n && j < n && k < n) {
        if (ss[i + k] == ss[j + k]) {
            k++;
        } else if (ss[i + k] > ss[j + k]) {
            i = max(j, i + k + 1);
            k = 0;
        } else {
            j = max(i, j + k + 1);
            k = 0;
        }
        if (i == j) j++;
    }
    return min(i, j);
}
```

**应用场景**：
- 判断两个字符串是否**互为循环移位**（先规范化到最小表示，再比较）
- 字符串的"规范形式"表示（如旋转等价类代表元）

### 34.4.3 LZ77 压缩算法简介

**LZ77**（Lempel-Ziv 1977）是现代压缩算法（gzip、zlib、PNG、deflate）的基础，其核心思想是**后向引用**（Back Reference）：

> 将已见过的内容用"(偏移, 长度)"来表示，而无需重复存储。

**编码方式**：输出为一系列 `(offset, length, next_char)` 三元组：
- `offset`：向后偏移多少位找到匹配的开始
- `length`：匹配的长度
- `next_char`：匹配之后的下一个新字符

**示例**：

```
原始：  a  b  c  a  b  c  a  b  c  d
位置：  0  1  2  3  4  5  6  7  8  9

输出：
(0, 0, 'a')   → 无历史可用，输出新字符 'a'
(0, 0, 'b')   → 输出新字符 'b'
(0, 0, 'c')   → 输出新字符 'c'
(3, 6, 'd')   → 从当前位置往前 3，长度 6 = "abcabc"，后跟新字符 'd'

压缩比：10 个字符 → 4 个三元组，约节省 60%
```

**与后缀结构的关系**：
- LZ77 的"最长后向匹配"在理论上等价于在后缀数组或后缀树上的查找。
- 最优 LZ77 编码可以在 $O(n \log \sigma)$ 时间内完成（其中 $\sigma$ 是字母表大小），需要用到后缀数组或 SA-IS。

**LZ78**（Lempel-Ziv 1978）则维护一个动态字典（类似 Trie），不需要滑动窗口，是 LZW（GIF/TIFF 格式）的基础。

**实用意义**：LZ 系列算法让你理解为什么重复内容多的文件（如代码、日志、PNG）压缩比高，而随机数据（如视频帧、加密数据）几乎不可压缩。

### 34.4.4 字符串综合对比小结：什么时候用哪个？

理解了 KMP、Rabin-Karp、Z 函数、Manacher、哈希、SA/SAM 后，面对一道字符串题，如何选择工具？

<div data-component="StringAlgoComparison"></div>

**快速参考指南**：

| 场景 | 推荐工具 | 复杂度 |
|---|---|---|
| 单模式串匹配 | KMP 或 Z 函数 | $O(n + m)$ |
| 多模式串匹配 | Aho-Corasick | $O(\text{总长} + \text{匹配数})$ |
| 子串相等性判断（频繁） | 字符串哈希 | 预处理 $O(n)$，查询 $O(1)$ |
| 最长回文子串 | Manacher | $O(n)$ |
| 回文子串计数 | Manacher / 哈希+枚举 | $O(n)$ / $O(n \log n)$ |
| 最长重复子串 | 哈希 + 二分 / SA + max(LCP) | $O(n \log n)$ / $O(n)$ |
| 字符串中不同子串数 | SA + LCP / SAM | $O(n)$ |
| 两字符串最长公共子串 | SA / SAM / 广义后缀树 | $O(n)$ |
| 字符串最小周期 | KMP π / Z 函数 | $O(n)$ |
| 最小循环移位 | Booth's Algorithm | $O(n)$ |
| 数据压缩 | LZ77/LZ78 概念 | $O(n \log \sigma)$ |

---

## 34.5 工程陷阱与调试技巧

### 陷阱 1：Manacher 的 `#` 下标转换

插入 `#` 后，扩充字符串的下标与原始字符串间的映射是：
- 原始起点 = $(i - P[i]) / 2$，但这是**整数除法**，且必须向下取整
- $i$ 为偶数 → 对应原始字符；$i$ 为奇数 → 对应 `#`（两字符之间的虚拟中心）

**常见 bug**：把 `//` 写成 `/`（Python 中会得到浮点数），或者用错了 `P` 的值和 `i` 的对应关系。

### 陷阱 2：哈希的负数取模（C++ 必读）

在 C++ 中，`(a - b) % mod` 当 `a < b` 时得到负数（C++ 对负数取模结果是负的）。

**正确写法**：必须写成 `((a - b) % mod + mod) % mod`。

Python 中取模总是非负（Python 的 `%` 运算符对负数自动加 mod），所以 Python 不需要特殊处理。

### 陷阱 3：Z[0] 的定义不统一

不同教材、不同实现中，`Z[0]` 的值不一致：
- **本文定义**：`Z[0] = 0`（避免与整个字符串自身匹配造成歧义）
- **部分实现**：`Z[0] = n`（整串与自身完整匹配）

在利用 `Z[p] = n - p` 判断周期时，如果 `Z[0] = n`，则 `p = 0` 总是满足但无意义（周期为 0 没有意义）。务必确认你的模板对 `Z[0]` 的处理。

### 陷阱 4：单模哈希在大数据或 Hack 场景中会碰撞

如果你在竞赛中遇到"哈希碰撞导致 WA"，立刻切换为双模哈希或换更大的质数。

记住：`base = 131, mod = 1e9+7` 是非常常见的选择，但竞赛出题方可能针对这个组合构造 Hack 数据。换用 `base = 131, mod = 1e18 + 9`（大质数）或双模可以大幅提高安全性。

### 陷阱 5：Z 函数 `[l, r]` 的边界——开区间还是闭区间？

本文使用**闭区间**：`l, r` 满足 $s[l..r] = s[0..r-l]$，初始 `l = r = 0`。

更新条件：`if i + Z[i] - 1 > r: l, r = i, i + Z[i] - 1`

有的实现用**开区间** `r` = 最右端 + 1，写法略有不同。无论哪种，只要前后一致即可，但读别人代码时务必确认。

---

## 34.6 小练习

**练习 1**：给定字符串 `s = "aabaa"`，手动计算 Manacher 的扩充字符串 `t` 和回文半径数组 `P`。最长回文子串是什么？

**练习 2**：字符串哈希 + 二分搜索解决 LeetCode #718（最长公共子数组）：将问题转化为"固定长度，判断两个字符串是否有公共子串"，用哈希 $O(1)$ 判断。写出完整代码并分析复杂度。

**练习 3**：对于字符串 `s = "abcabcabc"`，用 Z 函数和 KMP π 数组分别验证其最小周期为 3。

**练习 4**：扩展 LZ77 编码器，实现对字符串 `"aabaab"` 的完整编码输出（逐个三元组列出）。

---

## 34.7 章节小结

本章完成了字符串算法专题的最后一块拼图：

- **Manacher 算法**：用 `#` 插入统一奇偶，center/max\_right 提供初始值，$O(n)$ 求解最长回文子串；`P[i]` 直接等于原始回文长度，下标转换是 $(i - P[i]) / 2$。

- **字符串哈希**：多项式前缀哈希实现 $O(1)$ 子串相等判断；双模哈希将碰撞概率降至 $10^{-18}$；哈希 + 二分是"最长重复子串"的标准解法（$O(n \log n)$）。

- **Z 函数**：$Z[i]$ = 后缀 $s[i:]$ 与 $s$ 的最长公共前缀；[l, r] 窗口保证 $O(n)$ 构造；`P#T` 拼接实现字符串匹配；$Z[p] = n - p$ 且 $p \mid n$ 判断最小完整周期。

- **综合应用**：Booth 最小表示法 $O(n)$、LZ77/LZ78 压缩原理、各算法场景对比表。

至此，字符串专题（Chapter 31–34）全部收尾。你已掌握了从简单匹配到高级后缀结构的完整工具链，足以应对 LeetCode 困难字符串题和大多数竞赛字符串考点。

---

## 参考资料

- **Manacher 算法**：cp-algorithms.com/string/manacher.html；Manacher, G. 1975 原始论文
- **字符串哈希**：cp-algorithms.com/string/string-hashing.html
- **Z 函数**：cp-algorithms.com/string/z-function.html；与 KMP 等价性参考 cpalgorithms
- **Booth 最小表示法**：Booth 1980 原始论文
- **LZ77/LZ78**：Ziv & Lempel 1977/1978 两篇原始论文；Sedgewick《算法》第 5.5 节
- **综合练习**：LeetCode #5, #647, #459, #1044, #718

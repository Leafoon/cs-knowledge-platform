# Chapter 33: 后缀数组与后缀树

> **学习目标**：掌握后缀数组（Suffix Array, SA）的朴素构造、倍增法构造与 LCP（Longest Common Prefix）数组计算；理解后缀树（Suffix Tree）与后缀自动机（Suffix Automaton, SAM）的核心思想；能够用 SA / LCP / SAM 解决最长重复子串、不同子串计数、子串搜索、两串最长公共子串（LCS）等高频问题。

---

## 章节导读

如果说 Chapter 31 的 KMP、Rabin-Karp 解决的是“**一个模式串**在文本中如何高效匹配”，那么本章要解决的是更强的问题：

- 文本固定，**模式串很多次变化**，如何快速查询？
- 如何快速回答“最长重复子串”“有多少不同子串”等结构性问题？
- 两个字符串之间的最长公共子串（LCS）如何高效求？

一个非常生活化的比喻：

- 你把一本字典按“词条”排序，查词很快；
- 你把一本书的**所有后缀**都排序，就能快速回答大量子串问题。

这就是后缀数组的核心直觉。

---

## 33.1 后缀数组（Suffix Array）

### 33.1.1 定义与直觉

给定字符串 $s$（长度 $n$），把所有后缀列出来：

- 后缀 0：$s[0..n-1]$
- 后缀 1：$s[1..n-1]$
- ...
- 后缀 $n-1$：$s[n-1..n-1]$

然后按字典序排序，排序后第 $i$ 小后缀的起点下标记为 `SA[i]`。

这就是后缀数组：

$$
SA[i] = \text{第 }i\text{ 小后缀的起始位置}
$$

示例：`s = "banana"`

后缀列表：

- 0: banana
- 1: anana
- 2: nana
- 3: ana
- 4: na
- 5: a

排序后：

- a (5)
- ana (3)
- anana (1)
- banana (0)
- na (4)
- nana (2)

所以 `SA = [5, 3, 1, 0, 4, 2]`。

> 关键价值：把“所有后缀”排序后，很多子串问题都可以转化为“在 SA 上做二分”或“看相邻后缀关系（LCP）”。

<div data-component="SuffixArrayNaive"></div>

---

### 33.1.2 朴素构造：$O(n^2 \log n)$

朴素法很直接：

1. 把所有下标 `0..n-1` 放进数组；
2. 按 `s[i:]` 排序；
3. 排序结果就是 SA。

时间复杂度为什么高？

- 排序有 $O(n\log n)$ 次比较；
- 每次比较两个后缀最坏要看 $O(n)$ 字符；
- 总计 $O(n^2\log n)$。

适合教学与小数据，不适合大规模。

```python
def build_sa_naive(s: str) -> list[int]:
    """
    朴素后缀数组构造。

    时间复杂度: O(n^2 log n)
      - n log n 次比较
      - 每次比较最坏 O(n)
    空间复杂度: O(n)

    Edge Cases:
    - 空串: 返回 []
    - 单字符: 返回 [0]
    """
    n = len(s)
    # Python 的 sort 是稳定排序，key=s[i:] 可直接用
    # 但会构造大量切片字符串，常数较大
    return sorted(range(n), key=lambda i: s[i:])
```

```cpp
#include <bits/stdc++.h>
using namespace std;

vector<int> build_sa_naive(const string& s) {
    int n = (int)s.size();
    vector<int> sa(n);
    iota(sa.begin(), sa.end(), 0);

    // 注意：这里比较时会逐字符比较两个后缀
    // 最坏每次 O(n)，所以整体 O(n^2 log n)
    sort(sa.begin(), sa.end(), [&](int i, int j) {
        // string::substr 会分配新内存，性能不佳。
        // 这里演示教学写法，实际可用指针比较减少常数。
        return s.substr(i) < s.substr(j);
    });
    return sa;
}
```

---

### 33.1.3 倍增法（Prefix Doubling）：$O(n\log n)$

#### 1）核心思想

如果我们已经知道每个后缀长度为 $2^k$ 前缀的排名，就可以用它来构造长度 $2^{k+1}$ 的排名。

长度 $2^{k+1}$ 的键值可以写成二元组：

$$
( rank_k[i],\ rank_k[i + 2^k] )
$$

即“前一半排名 + 后一半排名”。

每轮将比较长度翻倍：$1,2,4,8,...$，总轮数 $\log n$。

#### 2）流程概览（小白友好）

把后缀看成“身份证号码”，每轮给每个位置发新编号：

- 第 0 轮：按单字符排名（a < b < c ...）；
- 第 1 轮：按 2 个字符排名；
- 第 2 轮：按 4 个字符排名；
- ...
- 当 $2^k \ge n$，排名唯一，结束。

#### 3）复杂度

- 每轮排序 $O(n\log n)$（若用比较排序）；
- 共 $\log n$ 轮：$O(n\log^2 n)$；
- 结合基数排序可做到每轮 $O(n)$，总 $O(n\log n)$。

下面给出“工程常用、易读”的 $O(n\log n)$ 级模板（比较排序版，常数可接受）：

```python
def build_sa_doubling(s: str) -> list[int]:
    """
    后缀数组倍增法（教学友好的实现）。

    时间复杂度: O(n log^2 n) （每轮 sort O(n log n)，共 O(log n) 轮）
    在竞赛中若用基数排序可优化到 O(n log n)。

    Edge Cases:
    - n == 0 -> []
    - n == 1 -> [0]
    """
    n = len(s)
    if n == 0:
        return []

    sa = list(range(n))
    rank = [ord(c) for c in s]      # 初始按字符 ASCII 排名
    tmp = [0] * n
    k = 1

    while k < n:
        # 二元键：(当前 rank, 右半段 rank)
        sa.sort(key=lambda i: (rank[i], rank[i + k] if i + k < n else -1))

        # 重新压缩排名
        tmp[sa[0]] = 0
        for i in range(1, n):
            a, b = sa[i - 1], sa[i]
            prev_key = (rank[a], rank[a + k] if a + k < n else -1)
            curr_key = (rank[b], rank[b + k] if b + k < n else -1)
            tmp[b] = tmp[a] + (1 if curr_key != prev_key else 0)

        rank, tmp = tmp, rank

        # 若排名已全唯一，可提前结束
        if rank[sa[-1]] == n - 1:
            break
        k <<= 1

    return sa
```

```cpp
#include <bits/stdc++.h>
using namespace std;

vector<int> build_sa_doubling(const string& s) {
    int n = (int)s.size();
    if (n == 0) return {};

    vector<int> sa(n), rank(n), tmp(n);
    iota(sa.begin(), sa.end(), 0);
    for (int i = 0; i < n; ++i) rank[i] = (unsigned char)s[i];

    for (int k = 1; k < n; k <<= 1) {
        auto key = [&](int i) {
            return pair<int,int>{rank[i], (i + k < n ? rank[i + k] : -1)};
        };

        sort(sa.begin(), sa.end(), [&](int i, int j) {
            return key(i) < key(j);
        });

        tmp[sa[0]] = 0;
        for (int i = 1; i < n; ++i) {
            tmp[sa[i]] = tmp[sa[i - 1]] + (key(sa[i - 1]) != key(sa[i]));
        }

        rank = tmp;
        if (rank[sa.back()] == n - 1) break; // 排名唯一，提前结束
    }

    return sa;
}
```

> 设计考量：先掌握“比较排序版倍增法”，理解框架后再升级到“基数排序版”。

<div data-component="SuffixArrayDoubling"></div>

---

### 33.1.4 SA-IS 线性构造（高层概述）

SA-IS 是经典的 $O(n)$ 后缀数组构造算法，核心是“诱导排序（Induced Sorting）”。

你可以先把它理解成三步：

1. 把位置分成 L-type / S-type；
2. 先排好关键的 LMS 子串；
3. 由 LMS 结果诱导出全部后缀顺序。

为什么竞赛里经常提 SA-IS？

- 理论上线性；
- 实战性能好；
- 但代码实现复杂、调试成本高。

本教程建议学习路线：

- 面试/工程：先精通倍增法 + Kasai；
- 竞赛高阶：再学 SA-IS 模板化实现。

---

### 33.1.5 LCP 数组定义与意义

`LCP[i]` 表示 `SA[i]` 和 `SA[i-1]` 两个相邻后缀的最长公共前缀长度。

约定：`LCP[0] = 0`（因为没有前一个后缀）。

为什么 LCP 很重要？

- `max(LCP)` = 最长重复子串长度；
- 不同子串个数可由 SA + LCP 直接计算：

$$
\text{distinct} = \frac{n(n+1)}{2} - \sum_{i=0}^{n-1} LCP[i]
$$

直觉：

- 总子串数是 $n(n+1)/2$；
- 相邻后缀公共前缀部分代表“重复贡献”，减掉即可。

---

### 33.1.6 Kasai 算法：$O(n)$ 构造 LCP

#### 1）为什么能线性？

Kasai 维护一个变量 `k`：当前后缀与其前驱后缀的公共前缀长度。

当我们从位置 `i` 走到 `i+1` 时，公共前缀最多减少 1，所以可先 `k -= 1`，再继续比较。

这个“减少不超过 1”的性质让总比较次数是线性的。

#### 2）实现步骤

1. 构造 `rank`：`rank[pos] = 该后缀在 SA 中的名次`；
2. 按原串下标 `i=0..n-1` 遍历；
3. 找到其前驱后缀 `j = SA[rank[i]-1]`；
4. 从已有的 `k` 开始扩展匹配；
5. 写入 `LCP[rank[i]] = k`。

```python
def build_lcp_kasai(s: str, sa: list[int]) -> list[int]:
    """
    Kasai 算法构造 LCP。

    时间复杂度: O(n)
    空间复杂度: O(n)

    Edge Cases:
    - n == 0 -> []
    - n == 1 -> [0]
    """
    n = len(s)
    if n == 0:
        return []

    rank = [0] * n
    for i, pos in enumerate(sa):
        rank[pos] = i

    lcp = [0] * n
    k = 0

    for i in range(n):
        r = rank[i]
        if r == 0:
            # SA 第一个后缀没有前驱
            k = 0
            continue

        j = sa[r - 1]
        # 从已有的 k 开始扩展，减少重复比较
        while i + k < n and j + k < n and s[i + k] == s[j + k]:
            k += 1

        lcp[r] = k

        # 下一个 i 会至少少 1（若 k > 0）
        if k > 0:
            k -= 1

    return lcp
```

```cpp
#include <bits/stdc++.h>
using namespace std;

vector<int> build_lcp_kasai(const string& s, const vector<int>& sa) {
    int n = (int)s.size();
    if (n == 0) return {};

    vector<int> rank(n), lcp(n, 0);
    for (int i = 0; i < n; ++i) rank[sa[i]] = i;

    int k = 0;
    for (int i = 0; i < n; ++i) {
        int r = rank[i];
        if (r == 0) {
            k = 0;
            continue;
        }

        int j = sa[r - 1];
        while (i + k < n && j + k < n && s[i + k] == s[j + k]) {
            ++k;
        }
        lcp[r] = k;

        if (k > 0) --k;
    }
    return lcp;
}
```

<div data-component="LCPArrayKasai"></div>

---

### 33.1.7 SA + LCP 的经典应用

#### A. 子串搜索（Binary Search on SA）

把模式串 `p` 与 SA 中中间后缀比较，二分找到匹配区间。

- 复杂度：$O(m\log n)$，$m$ 为模式串长度。

```python
def contains_pattern(s: str, sa: list[int], p: str) -> bool:
    """
    在后缀数组中二分查找模式串 p 是否出现。

    时间复杂度: O(m log n)
    """
    n = len(s)
    m = len(p)
    lo, hi = 0, n - 1

    while lo <= hi:
        mid = (lo + hi) // 2
        start = sa[mid]
        suffix_prefix = s[start:start + m]

        if suffix_prefix == p:
            return True
        if suffix_prefix < p:
            lo = mid + 1
        else:
            hi = mid - 1

    return False
```

```cpp
bool contains_pattern(const string& s, const vector<int>& sa, const string& p) {
    int n = (int)s.size();
    int m = (int)p.size();
    int lo = 0, hi = n - 1;

    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        int start = sa[mid];

        // compare 后缀前 m 个字符与 p
        string t = s.substr(start, m);
        if (t == p) return true;
        if (t < p) lo = mid + 1;
        else hi = mid - 1;
    }
    return false;
}
```

<div data-component="SuffixSearchDemo"></div>

#### B. 最长重复子串（Longest Repeated Substring）

- 长度就是 `max(LCP)`；
- 子串可从对应的 `SA[argmax]` 取出。

```python
def longest_repeated_substring(s: str) -> str:
    if not s:
        return ""
    sa = build_sa_doubling(s)
    lcp = build_lcp_kasai(s, sa)

    best_len = 0
    best_pos = 0
    for i in range(1, len(s)):
        if lcp[i] > best_len:
            best_len = lcp[i]
            best_pos = sa[i]

    return s[best_pos: best_pos + best_len]
```

```cpp
string longest_repeated_substring(const string& s) {
    if (s.empty()) return "";

    vector<int> sa = build_sa_doubling(s);
    vector<int> lcp = build_lcp_kasai(s, sa);

    int best_len = 0, best_pos = 0;
    for (int i = 1; i < (int)s.size(); ++i) {
        if (lcp[i] > best_len) {
            best_len = lcp[i];
            best_pos = sa[i];
        }
    }
    return s.substr(best_pos, best_len);
}
```

#### C. 不同子串个数

```python
def count_distinct_substrings(s: str) -> int:
    n = len(s)
    if n == 0:
        return 0
    sa = build_sa_doubling(s)
    lcp = build_lcp_kasai(s, sa)
    total = n * (n + 1) // 2
    return total - sum(lcp)
```

```cpp
long long count_distinct_substrings(const string& s) {
    int n = (int)s.size();
    if (n == 0) return 0;

    vector<int> sa = build_sa_doubling(s);
    vector<int> lcp = build_lcp_kasai(s, sa);

    long long total = 1LL * n * (n + 1) / 2;
    long long dup = 0;
    for (int x : lcp) dup += x;
    return total - dup;
}
```

---

## 33.2 后缀树（Suffix Tree）

### 33.2.1 从后缀 Trie 到后缀树：压缩的力量

后缀 Trie 是把所有后缀插进 Trie，理论上直观，但节点太多。

后缀树（Suffix Tree）= 后缀 Trie 的压缩版本：

- 把单分支链压成一条边；
- 边上标记原串区间 `[l, r]` 而不是复制字符串。

典型性质：

- 字符串结尾加唯一终止符 `$`；
- 节点数线性（$O(n)$）；
- 许多查询可在线性时间完成。

<div data-component="SuffixTreeVisualization"></div>

---

### 33.2.2 后缀树与后缀数组的关系

两者本质上表达的是同一批后缀顺序信息：

- 对后缀树做 DFS（按边字典序），叶子访问顺序就是 SA；
- 相邻叶子的最近公共祖先深度，对应 LCP。

所以很多工程场景会选择：

- 实现更简单：SA + LCP；
- 查询类型极丰富、需在线更新：才考虑后缀树或 SAM。

---

### 33.2.3 Ukkonen 算法（线性在线构造）高层理解

Ukkonen 是后缀树经典线性算法。你不必一上来背代码，可先掌握这三个核心概念：

1. **Active Point（活跃点）**：当前扩展从哪里继续；
2. **Suffix Link（后缀链接）**：失败时快速跳到“次长后缀上下文”；
3. **Implicit Tree（隐式树）**：每一步不强制把所有叶子显式展开。

三条扩展规则（概念版）：

- Rule 1：叶边自动延长；
- Rule 2：需要时分裂边并创建新叶；
- Rule 3：遇到已存在路径可提前结束本轮。

复杂度为什么是 $O(n)$？

- 通过 suffix link 避免回退到根重做；
- 活跃点单调推进，摊销线性。

> 工程建议：后缀树实现复杂、坑点多；若你目标是面试或业务，优先 SA / SAM。

---

### 33.2.4 最长公共子串（LCS）与后缀结构

经典做法之一：

- 构造 `S = A + '#' + B + '$'` 的后缀结构；
- 在相邻后缀中找一个来自 A、一个来自 B 的 pair；
- 最大这样的 LCP 即 LCS 长度。

下面给出更易实现的 SA + LCP 版本：

```python
def lcs_two_strings(a: str, b: str) -> str:
    """
    用 SA + LCP 求两字符串最长公共子串。

    技巧：拼接串 s = a + '#' + b + '$'
    只在相邻后缀来自不同源串时更新答案。
    """
    sep1, sep2 = '#', '$'
    assert sep1 not in a and sep1 not in b
    assert sep2 not in a and sep2 not in b

    s = a + sep1 + b + sep2
    sa = build_sa_doubling(s)
    lcp = build_lcp_kasai(s, sa)

    split = len(a)
    best_len = 0
    best_pos = 0

    def from_a(pos: int) -> bool:
        return pos < split

    for i in range(1, len(s)):
        x, y = sa[i], sa[i - 1]
        if from_a(x) != from_a(y):
            if lcp[i] > best_len:
                best_len = lcp[i]
                best_pos = sa[i]

    return s[best_pos: best_pos + best_len]
```

```cpp
string lcs_two_strings(const string& a, const string& b) {
    char sep1 = '#', sep2 = '$';
    // 真实工程可选择未出现的分隔符，或用整型映射避免冲突
    string s = a + sep1 + b + sep2;

    vector<int> sa = build_sa_doubling(s);
    vector<int> lcp = build_lcp_kasai(s, sa);

    int split = (int)a.size();
    int best_len = 0, best_pos = 0;

    auto fromA = [&](int pos) {
        return pos < split;
    };

    for (int i = 1; i < (int)s.size(); ++i) {
        int x = sa[i], y = sa[i - 1];
        if (fromA(x) != fromA(y)) {
            if (lcp[i] > best_len) {
                best_len = lcp[i];
                best_pos = sa[i];
            }
        }
    }

    return s.substr(best_pos, best_len);
}
```

---

### 33.2.5 广义后缀树（Generalized Suffix Tree）

广义后缀树就是“多个字符串共建后缀树”。

常见用途：

- 多文档公共片段检测；
- 生物序列多样本比较；
- 代码克隆检测。

工程注意点：

- 每个字符串必须有不同终止符，避免后缀混淆；
- 节点需维护“来自哪些字符串”的标记信息。

---

## 33.3 后缀自动机（SAM, Suffix Automaton）

### 33.3.1 SAM 的状态到底表示什么？

SAM 不是按“具体字符串”建节点，而是按 **endpos 等价类** 建状态。

`endpos(x)`：子串 `x` 在原串中“作为结尾”的所有位置集合。

若两个子串 endpos 集合相同，它们落在同一个 SAM 状态里。

直觉上，SAM 把大量重复子串压缩成了少量状态，所以整体状态数是线性的（最多 `2n-1`）。

---

### 33.3.2 后缀链接（Suffix Link）与 KMP 的联系

后缀链接 `link[v]`：

- 从状态 `v` 指向“最长真后缀对应状态”。

这和 KMP 失配跳转有共同思想：

- 当前上下文失败，跳到“次长可行上下文”，避免从头开始。

---

### 33.3.3 SAM 线性构造（在线）

每加入一个字符 `c`：

1. 新建状态 `cur`；
2. 从 `last` 沿 suffix link 回溯，补转移；
3. 若遇到冲突（`len[p] + 1 != len[q]`），创建 `clone` 拆分；
4. 更新相关 suffix link。

这是 SAM 的精髓，也是最容易写错的地方。

```python
class SAMState:
    __slots__ = ("next", "link", "length", "occ")

    def __init__(self):
        # next: 转移边，字符 -> 状态编号
        self.next: dict[str, int] = {}
        # link: 后缀链接
        self.link: int = -1
        # length: 本状态可表示字符串的最大长度
        self.length: int = 0
        # occ: 该状态的出现次数（需后续拓扑传递）
        self.occ: int = 0


class SuffixAutomaton:
    def __init__(self):
        self.st: list[SAMState] = [SAMState()]
        self.last: int = 0

    def extend(self, c: str) -> None:
        """
        插入一个字符 c 到 SAM。

        关键陷阱:
        - clone 节点不能继承 occ（它不是新出现的终点）
        - 回溯改 link 时要小心终止条件 p == -1
        """
        cur = len(self.st)
        self.st.append(SAMState())
        self.st[cur].length = self.st[self.last].length + 1
        self.st[cur].occ = 1

        p = self.last
        while p != -1 and c not in self.st[p].next:
            self.st[p].next[c] = cur
            p = self.st[p].link

        if p == -1:
            self.st[cur].link = 0
        else:
            q = self.st[p].next[c]
            if self.st[p].length + 1 == self.st[q].length:
                self.st[cur].link = q
            else:
                clone = len(self.st)
                self.st.append(SAMState())
                self.st[clone].next = self.st[q].next.copy()
                self.st[clone].link = self.st[q].link
                self.st[clone].length = self.st[p].length + 1
                self.st[clone].occ = 0

                while p != -1 and self.st[p].next.get(c, -1) == q:
                    self.st[p].next[c] = clone
                    p = self.st[p].link

                self.st[q].link = clone
                self.st[cur].link = clone

        self.last = cur

    def build(self, s: str) -> None:
        for ch in s:
            self.extend(ch)

    def count_distinct_substrings(self) -> int:
        """
        不同子串数 = sum(len[v] - len[link[v]])，v != root
        """
        ans = 0
        for v in range(1, len(self.st)):
            ans += self.st[v].length - self.st[self.st[v].link].length
        return ans
```

```cpp
#include <bits/stdc++.h>
using namespace std;

struct SAMState {
    unordered_map<char, int> next;
    int link = -1;
    int len = 0;
    int occ = 0;
};

struct SuffixAutomaton {
    vector<SAMState> st;
    int last;

    SuffixAutomaton() {
        st.reserve(200000);
        st.push_back(SAMState()); // root
        last = 0;
    }

    void extend(char c) {
        int cur = (int)st.size();
        st.push_back(SAMState());
        st[cur].len = st[last].len + 1;
        st[cur].occ = 1;

        int p = last;
        while (p != -1 && !st[p].next.count(c)) {
            st[p].next[c] = cur;
            p = st[p].link;
        }

        if (p == -1) {
            st[cur].link = 0;
        } else {
            int q = st[p].next[c];
            if (st[p].len + 1 == st[q].len) {
                st[cur].link = q;
            } else {
                int clone = (int)st.size();
                st.push_back(st[q]); // 拷贝 q 的转移与 link
                st[clone].len = st[p].len + 1;
                st[clone].occ = 0;   // clone 不是新终止状态

                while (p != -1 && st[p].next[c] == q) {
                    st[p].next[c] = clone;
                    p = st[p].link;
                }

                st[q].link = clone;
                st[cur].link = clone;
            }
        }

        last = cur;
    }

    void build(const string& s) {
        for (char c : s) extend(c);
    }

    long long count_distinct_substrings() const {
        long long ans = 0;
        for (int v = 1; v < (int)st.size(); ++v) {
            ans += st[v].len - st[st[v].link].len;
        }
        return ans;
    }
};
```

<div data-component="SAMStateTransition"></div>

---

### 33.3.4 SAM 的高频应用

#### A. 不同子串个数

公式和上面实现一致：

$$
\sum_{v \neq root}(len[v] - len[link[v]])
$$

#### B. 子串出现次数

构建 SAM 后，对每个状态按 `len` 降序把 `occ` 向 `link` 汇总，可以得到每个状态代表子串族的出现次数。

#### C. 两字符串 LCS（SAM 版本）

- 先对 `A` 建 SAM；
- 用 `B` 在 SAM 上行走，失配沿 link 回退；
- 维护当前匹配长度最大值。

时间复杂度 $O(|A| + |B|)$。

```python
def lcs_with_sam(a: str, b: str) -> int:
    sam = SuffixAutomaton()
    sam.build(a)

    v = 0      # 当前状态
    l = 0      # 当前匹配长度
    best = 0

    for ch in b:
        if ch in sam.st[v].next:
            v = sam.st[v].next[ch]
            l += 1
        else:
            while v != -1 and ch not in sam.st[v].next:
                v = sam.st[v].link
            if v == -1:
                v = 0
                l = 0
                continue
            l = sam.st[v].length + 1
            v = sam.st[v].next[ch]

        if l > best:
            best = l

    return best
```

```cpp
int lcs_with_sam(const string& a, const string& b) {
    SuffixAutomaton sam;
    sam.build(a);

    int v = 0, l = 0, best = 0;

    for (char ch : b) {
        if (sam.st[v].next.count(ch)) {
            v = sam.st[v].next[ch];
            ++l;
        } else {
            while (v != -1 && !sam.st[v].next.count(ch)) {
                v = sam.st[v].link;
            }
            if (v == -1) {
                v = 0;
                l = 0;
                continue;
            }
            l = sam.st[v].len + 1;
            v = sam.st[v].next[ch];
        }
        best = max(best, l);
    }

    return best;
}
```

---

### 33.3.5 SAM vs SA vs 后缀树：如何选型？

| 结构 | 构建复杂度 | 常用能力 | 实现难度 | 空间特征 | 典型场景 |
|---|---:|---|---|---|---|
| SA + LCP | $O(n\log n)$（倍增） | 子串搜索、重复子串、统计类 | 中 | 低 | 面试/工程通用 |
| 后缀树 | $O(n)$ | 理论上最全，查询强 | 很高 | 中高 | 学术/特定引擎 |
| SAM | $O(n)$ | 不同子串计数、LCS、出现次数 | 中高 | 中 | 竞赛高频 |

选型建议（务实版）：

1. **先学 SA + Kasai**：最稳、最好讲、最好调试；
2. 需要线性在线构造时学 **SAM**；
3. 后缀树主要理解思想与能力边界。

<div data-component="StringStructureComparison"></div>

---

## 33.4 工程与面试中的常见坑

### 33.4.1 分隔符冲突

拼接两串做 LCS 时，`#`、`$` 必须不在原串内，否则结果错误。

### 33.4.2 LCP 下标错位

`LCP[i]` 对应 `SA[i]` 与 `SA[i-1]`，不是与 `SA[i+1]`。

### 33.4.3 SAM clone 细节

- clone 要复制 `next` 与 `link`；
- clone 的 `occ = 0`；
- 回溯重定向条件要严格。

### 33.4.4 复杂度误判

- SA 二分搜索是 $O(m\log n)$，不是 $O(\log n)$；
- 朴素 SA 在 Python 切片下常数非常大。

---

## 33.5 小练习（建议手算 + 编码）

1. 手算 `s = "mississippi"` 的 SA 与部分 LCP，验证 `max(LCP)`。
2. 实现 `count_distinct_substrings` 并与暴力去重结果对拍。
3. 用 SA + LCP 解决 LeetCode #1044（最长重复子串）。
4. 用 SAM 实现两串 LCS，并与 SA 版本比较时间和内存。

---

## 33.6 章节小结

- 后缀数组是“后缀排序”的标准化结构，配合 LCP 能解决大量子串问题；
- Kasai 算法是 LCP 的核心，线性且实用；
- 后缀树提供更强表达但实现复杂；
- SAM 在线线性、竞赛高频，尤其适合计数类与 LCS 类问题。

> 到这里，你已经具备进入 Chapter 34（Manacher / Z-Function / 高级字符串哈希）的完整前置能力。

---

## 参考资料

- CLRS（第4版）字符串相关章节
- Sedgewick《Algorithms 4th》字符串章节
- Stanford CS166（String Data Structures）
- Ukkonen, 1995, On-line construction of suffix trees

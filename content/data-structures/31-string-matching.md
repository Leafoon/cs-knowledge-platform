# Chapter 31: 字符串匹配——KMP 与 Rabin-Karp

## 章节导读

**字符串匹配（String Matching / Pattern Searching）**是计算机科学中最基础、也最广泛应用的问题之一：给定一个**文本串 T**（Text，长度 $n$）和一个**模式串 P**（Pattern，长度 $m$），找出 P 在 T 中所有出现位置。

这个问题无处不在：

- **IDE 中的"找到"功能**：在整个代码文件中查找某个变量名；
- **搜索引擎**：关键词在数百万网页文本中的定位；
- **生物信息学**：在人类基因组（约 30 亿碱基）中找一段基因序列；
- **网络安全**：深度包检测（DPI）在数据流中匹配恶意特征码；
- **编译器**：词法分析器在源代码中识别关键字、标识符。

**最简单的想法是对的吗？**

直觉上，我们可以在文本的每一个位置尝试与模式串比较——这就是朴素算法（Naive / Brute Force）。然而，当模式串是 $\underbrace{\texttt{aaa...a}}_{m-1}\texttt{b}$ 而文本是 $\underbrace{\texttt{aaa...a}}_{n}$ 时，朴素算法每次都要比较 $m-1$ 次才发现失配，总计 $O(nm)$ 次比较——这对于大文本是灾难性的。

本章介绍三种经典改进算法，它们从不同角度切入，都将最坏情况降到了 $O(n + m)$（或更好）：

| 算法 | 核心思想 | 构建时间 | 匹配时间 | 最坏情况 | 实践性能 |
|---|---|---|---|---|---|
| **朴素** | 逐位窗口比较 | $O(1)$ | $O(nm)$ | $O(nm)$ | 日常文本尚可 |
| **KMP** | 失配后利用前缀函数跳跃 | $O(m)$ | $O(n)$ | $O(n+m)$ | 稳定线性 |
| **Rabin-Karp** | 滚动哈希比较窗口 | $O(m)$ | $O(n)$ 期望 | $O(nm)$ 极罕见 | 多模式场景极优 |
| **Boyer-Moore** | 从右到左比较，大步跳跃 | $O(m + \sigma)$ | $O(n/m)$ 典型 | $O(nm)$ | 实践最快 |

> 📖 **参考来源**：CLRS 第4版 Chapter 32；MIT 6.006 Lecture 9（String Matching）

---

## 31.1 朴素字符串匹配回顾

### 31.1.1 朴素算法的完整分析

**算法描述**：

在文本 $T[0..n-1]$ 中的每一个可能的对齐位置 $s$（$s$ 从 $0$ 到 $n-m$），将模式串 $P[0..m-1]$ 与 $T[s..s+m-1]$ 逐字符比较，若全部匹配则记录匹配位置。

```
对于 s = 0, 1, ..., n-m:
    若 T[s..s+m-1] == P[0..m-1]:
        输出匹配位置 s
```

用一幅图来直观理解——模式 `ABCAB` 在文本 `ABCAABCAB` 中的匹配过程：

```
文本:   A B C A A B C A B
模式:   A B C A B
           ↕ ↕ ↕ ↕ ↕         s=0: 前4位匹配，第5位 'A'≠'B'，失配！
           A B C A B         → 直接 s=1

文本:   A B C A A B C A B
              A B C A B      s=1: 'B'≠'A'，立刻失配
                A B C A B    s=2: 'A'≠'A' 匹配，'A'≠'B' 失配
                  A B C A B  s=3: 匹配，... ，失配
                    A B C A B  s=4: 全部匹配！ ✓
```

**朴素算法的 Python 实现**：

```python
from typing import List

def naive_string_match(T: str, P: str) -> List[int]:
    """
    朴素字符串匹配。
    T: 文本串，长度 n
    P: 模式串，长度 m
    返回所有匹配起始位置的列表（0-indexed）
    
    时间复杂度: O(nm)  —— 每个对齐位置最多比较 m 次
    空间复杂度: O(1)
    """
    n, m = len(T), len(P)
    results = []

    # s: 对齐位置（shift），文本指针的起点
    for s in range(n - m + 1):       # n-m+1 个候选对齐位置
        matched = True
        for j in range(m):           # 逐字符比较
            if T[s + j] != P[j]:
                matched = False
                break                # 早停：任意一位失配就放弃
        if matched:
            results.append(s)

    return results

# ----- 测试 -----
T = "ABCAABCAB"
P = "ABCAB"
print(naive_string_match(T, P))  # 输出: [4]

T2 = "aabaaa"
P2 = "aa"
print(naive_string_match(T2, P2))  # 输出: [0, 3, 4]
```

```cpp
#include <bits/stdc++.h>
using namespace std;

/*
 * 朴素字符串匹配（C++）
 * T: 文本串（长度 n），P: 模式串（长度 m）
 * 返回所有匹配起始位置（0-indexed）
 *
 * 时间: O(nm)  空间: O(1)
 */
vector<int> naiveStringMatch(const string& T, const string& P) {
    int n = (int)T.size(), m = (int)P.size();
    vector<int> results;

    for (int s = 0; s <= n - m; ++s) {       // 枚举每个对齐位置
        bool matched = true;
        for (int j = 0; j < m; ++j) {        // 逐字符比较
            if (T[s + j] != P[j]) {
                matched = false;
                break;                         // 失配即终止当前位置
            }
        }
        if (matched) results.push_back(s);
    }
    return results;
}

// 测试
// naiveStringMatch("ABCAABCAB", "ABCAB") → {4}
// naiveStringMatch("aabaaa", "aa")       → {0, 3, 4}
```

### 31.1.2 $O(nm)$ 的根本原因：有效信息被丢弃

朴素算法低效的根本原因是**每次失配后都将两个指针归位到初始状态，完全抛弃了已经完成的比较信息**。

让我们来看一个典型的"信息浪费"场景：

**场景**：模式 P = `ABCAB`，文本 T = `ABCAABCAB`

```
步骤1（s=0）: 文本 T[0..4] = "ABCAA" vs 模式 P = "ABCAB"
              A B C A A       (文本窗口)
              A B C A B       (模式)
                      ↑ 第 4 位：'A' ≠ 'B'，失配！
```

朴素算法此时：**s → 1，j → 0**，重新从 P[0] 比较文本 T[1]。

**但我们已经知道**：在刚才的比较中，`T[0..3] == P[0..3] == "ABCA"`。所以：
- T[1] = `B`、T[2] = `C`、T[3] = `A` —— **这三个字符的值我们已经知道了！**
- 当 s=1 时，我们要比较 T[1]='B' 与 P[0]='A'——不同，必然失配。这次比较**完全是多余的**。

**真正的问题**：在已知前缀 `P[0..3] = "ABCA"` 匹配了 `T[0..3]` 后，我们能否直接推断出：下一个有可能成功的对齐位置在哪里？

答案是肯定的——这正是 KMP 算法的核心思想。KMP 通过预处理模式串，为每一种"已匹配前缀"计算出失配时的**下一个最佳对齐位置**，从而让文本指针永不回退。

### 31.1.3 改进目标：文本指针只进不退

高效字符串匹配算法的核心约束：**文本指针 `i` 永远只向右移动**。

这意味着整个匹配过程文本指针最多走 $n$ 步，总时间为 $O(n)$（加上预处理 $O(m)$）。要做到这一点，算法需要在失配时**仅移动模式指针**（或等价地，通过一个预计算的表格大幅右移模式窗口）。

> 💡 **直觉比喻**：想象你在用一把尺子（模式串）在一段文字（文本）上滑动寻找匹配。朴素算法每次滑动 1 格；聪明的算法根据失配情况，直接跳过明显不可能的对齐位置，一次可以滑动多格。

---

## 31.2 KMP 算法（Knuth-Morris-Pratt）

KMP 算法由 Donald Knuth、Vaughan Pratt 和 James Morris 于 1977 年发表，是字符串匹配领域最经典的算法之一。它的总时间复杂度为 $O(n + m)$，且在最坏情形下仍能保证这一界限。

### 31.2.1 失效函数（Failure Function / π 数组）的定义

KMP 的精髓在于预处理模式串，构建一个**失效函数（Failure Function）**，也称 **π 数组（pi array）** 或**部分匹配表（partial match table）**。

**定义**：对于模式串 $P[0..m-1]$，定义 $\pi[i]$ 为子串 $P[0..i]$ 的**最长真前缀后缀（Longest Proper Prefix which is also a Suffix，LPS）的长度**。

其中"**真前缀**"指不等于整个字符串本身的前缀，"**真后缀**"同理。

**用字典的类比**：想象模式串是一串有特定节律的音节，$\pi[i]$ 告诉你：如果从头听到第 $i$ 个音节后突然停止（失配），从哪里"接上"节奏继续听最有效——因为结尾的一段节奏和开头完全一样！

**逐字符计算 π 数组的例子**：

模式串 P = `A B C A B C D A B`（下标 0~8）

| $i$ | P[0..i] | 所有真前缀 | 所有真后缀 | 最长公共（LPS） | $\pi[i]$ |
|---|---|---|---|---|---|
| 0 | `A` | `{ε}` | `{ε}` | ε（空） | **0** |
| 1 | `AB` | `ε, A` | `ε, B` | ε | **0** |
| 2 | `ABC` | `ε, A, AB` | `ε, C, BC` | ε | **0** |
| 3 | `ABCA` | ...`A`... | ...`A`... | `A` | **1** |
| 4 | `ABCAB` | ...`AB`... | ...`AB`... | `AB` | **2** |
| 5 | `ABCABC` | ...`ABC`... | ...`ABC`... | `ABC` | **3** |
| 6 | `ABCABCD` | ...`ABCD`... | ...`ABCD`... 无 | ε | **0** |
| 7 | `ABCABCDA` | ...`A`... | ...`A`... | `A` | **1** |
| 8 | `ABCABCDAB` | ...`AB`... | ...`AB`... | `AB` | **2** |

所以 `π = [0, 0, 0, 1, 2, 3, 0, 1, 2]`。

**π[i] 的直觉意义**：若在文本中从 $P[0]$ 一路匹配到 $P[i]$ 后，下一个字符失配了，则模式指针不必归零，而是跳回 $\pi[i]$——因为 $P[0..\pi[i]-1]$ 与 $P[i-\pi[i]+1..i]$ 完全相同，前者已经与文本的对应部分匹配过了！

> **数学表达**：$\pi[i] = \max\{k : k < i+1 \text{ 且 } P[0..k-1] = P[i-k+1..i]\}$，若不存在则为 0。

### 31.2.2 π 数组的线性时间构造

**朴素 O(m²) 构造**：对每个 $i$，枚举所有 $k < i+1$ 检验是否满足条件——太慢。

**线性 O(m) 构造（双指针法）**：

核心观察：利用 π 数组的自相似性——$\pi$ 数组本身就是"在模式串上做字符串匹配"的结果。

**算法（COMPUTE-PREFIX-FUNCTION）**：

使用两个指针：
- `i`：慢指针，遍历 $P[1..m-1]$（我们已知 $\pi[0] = 0$）；
- `k`：快指针，追踪当前"匹配前缀"的长度（也是下一个待比较的位置）。

```
k = 0           # 初始时，已匹配的前缀长度为 0
π[0] = 0

for i = 1 to m-1:
    while k > 0 and P[k] ≠ P[i]:
        k = π[k-1]      # 失配时，k 利用 π 本身回跳（不是归零！）
    if P[k] == P[i]:
        k = k + 1        # 匹配：已匹配前缀增长 1
    π[i] = k            # 记录当前的 LPS 长度
```

**手工模拟**（P = `ABCAB`，$m = 5$）：

| 步骤 | i | P[i] | k | P[k] | 操作 | π |
|---|---|---|---|---|---|---|
| 初始 | - | - | 0 | - | π[0]=0 | [0] |
| 1 | 1 | B | 0 | A | B≠A，k=0不回跳；B≠A，不增；π[1]=0 | [0,0] |
| 2 | 2 | C | 0 | A | C≠A；π[2]=0 | [0,0,0] |
| 3 | 3 | A | 0 | A | A=A，k=1；π[3]=1 | [0,0,0,1] |
| 4 | 4 | B | 1 | B | B=B，k=2；π[4]=2 | [0,0,0,1,2] |

结果：`π = [0, 0, 0, 1, 2]` ✓

**摊销复杂度分析**：

`k` 最多增加 $m-1$ 次（每次 `i` 迭代最多 +1），而 `k` 只能通过 `k = π[k-1]` 递减，且每次递减都使 `k` 严格减小。因此 `k` 的总递减次数不超过总递增次数 $m-1$，整个循环执行 $O(m)$ 次。

<div data-component="KMPFailureFunctionBuild"></div>

### 31.2.3 π 数组的直觉理解：失配时的"最优跳跃"

**为什么失配时跳到 π[q-1]？**

设当前已匹配 P[0..q-1] 与 T[i-q..i-1]，然后 T[i] ≠ P[q] 失配。

```
文本:  ... T[i-q] ... T[i-1]  T[i] ...
           |=======P[0..q-1]=======|  ← 这段已匹配
模式:      P[0]   ... P[q-1]  P[q] ← 在此失配

π[q-1] = k
含义: P[0..k-1] = P[q-k..q-1]
即:   T[i-k..i-1] = P[0..k-1]   （因为 T[i-q..i-1]已知=P[0..q-1]）
```

所以，我们跳到 P[k] 处继续比较，**相当于把模式向右滑动 q-k 个位置**，同时文本指针不动。

```
滑动前: 文本 [T[i-q] ... T[i-1]] T[i]
        模式 [P[0]  ... P[q-1] ] P[q]  ← 在此失配

滑动后: 文本  ... [T[i-k] ... T[i-1]] T[i]
               模式 [P[0]  ... P[k-1]] P[k]  ← 从 P[k] 继续比较 T[i]
                     ↑已知相等，无需再比较↑
```

这就是 KMP 的精妙之处：**已匹配的部分信息被充分利用，文本指针从不后退**。

### 31.2.4 KMP-MATCHER 主循环

有了 π 数组，匹配过程就很自然了：

```
KMP-MATCHER(T, P):
    n = len(T), m = len(P)
    π = COMPUTE-PREFIX-FUNCTION(P)
    q = 0           # 当前已匹配的字符数（模式指针）
    
    for i = 0 to n-1:          # i: 文本指针，永远只向右
        while q > 0 and T[i] ≠ P[q]:
            q = π[q-1]         # 利用失效函数跳跃
        if T[i] == P[q]:
            q = q + 1          # 匹配成功，模式指针前进
        if q == m:             # 完整模式已匹配！
            输出匹配位置 i - m + 1
            q = π[q-1]         # 继续寻找下一个匹配
```

**关键点**：
1. `i` 在外层 for 循环中：**严格递增，永不回退**；
2. `q` 在失配时通过 `π[q-1]` 跳回，但不归零（除非已经是 0）；
3. 每当 `q == m`，立即输出，并用 `π[m-1]` 跳回（处理重叠匹配）。

**完整 Python 实现**（含 π 数组构建）：

```python
from typing import List

def compute_pi(P: str) -> List[int]:
    """
    构建 KMP 失效函数（π 数组）。
    π[i] = P[0..i] 的最长真前缀后缀的长度
    
    时间: O(m)；空间: O(m)
    """
    m = len(P)
    pi = [0] * m
    k = 0                          # 已匹配前缀长度

    for i in range(1, m):
        # 失配：利用 π 自身回跳（核心！不是归零）
        while k > 0 and P[k] != P[i]:
            k = pi[k - 1]

        if P[k] == P[i]:
            k += 1                 # 前缀延伸一位

        pi[i] = k                  # 记录当前 LPS 长度

    return pi


def kmp_matcher(T: str, P: str) -> List[int]:
    """
    KMP 字符串匹配。
    
    时间: O(n + m)；空间: O(m)（仅存储 π 数组）
    """
    n, m = len(T), len(P)
    if m == 0:
        return list(range(n + 1))  # 空模式：所有位置均匹配（边界处理）
    
    pi = compute_pi(P)
    results = []
    q = 0                          # 当前已匹配字符数（模式指针）

    for i in range(n):             # i: 文本指针，只增不减
        # 失配：模式指针利用 π 跳回
        while q > 0 and T[i] != P[q]:
            q = pi[q - 1]

        if T[i] == P[q]:
            q += 1                 # 匹配：模式指针前进

        if q == m:                 # 完整匹配！
            results.append(i - m + 1)
            q = pi[q - 1]         # 处理重叠匹配（继续右移寻找下一个）

    return results


# ----- 测试 -----
T = "ABABCABCABABABD"
P = "ABABD"
print(f"π 数组: {compute_pi(P)}")          # [0, 0, 1, 2, 0]
print(f"匹配位置: {kmp_matcher(T, P)}")    # [10]

T2 = "aababaa"
P2 = "aba"
print(f"π 数组: {compute_pi(P2)}")         # [0, 0, 1]
print(f"匹配位置: {kmp_matcher(T2, P2)}")  # [1, 3]  （注意重叠！）
```

```cpp
#include <bits/stdc++.h>
using namespace std;

/*
 * 构建 KMP 失效函数（π 数组）
 * 时间: O(m)；空间: O(m)
 */
vector<int> computePi(const string& P) {
    int m = (int)P.size();
    vector<int> pi(m, 0);
    int k = 0;                        // 已匹配前缀长度

    for (int i = 1; i < m; ++i) {
        // 失配时利用 π 自身回跳（摊销 O(m)）
        while (k > 0 && P[k] != P[i])
            k = pi[k - 1];

        if (P[k] == P[i])
            ++k;                      // 前缀延伸

        pi[i] = k;                    // 记录 LPS 长度
    }
    return pi;
}

/*
 * KMP 字符串匹配
 * 时间: O(n + m)；空间: O(m)
 */
vector<int> kmpMatcher(const string& T, const string& P) {
    int n = (int)T.size(), m = (int)P.size();
    vector<int> results;
    if (m == 0) {
        for (int i = 0; i <= n; ++i) results.push_back(i);
        return results;
    }

    vector<int> pi = computePi(P);
    int q = 0;                        // 当前已匹配字符数（模式指针）

    for (int i = 0; i < n; ++i) {    // i: 文本指针，严格递增
        while (q > 0 && T[i] != P[q])
            q = pi[q - 1];           // 利用失效函数跳跃

        if (T[i] == P[q])
            ++q;                      // 匹配：模式前进

        if (q == m) {                 // 完整匹配！
            results.push_back(i - m + 1);
            q = pi[q - 1];           // 允许重叠匹配
        }
    }
    return results;
}

/*
int main() {
    // kmpMatcher("ABABCABCABABABD", "ABABD") → {10}
    // kmpMatcher("aababaa", "aba")           → {1, 3}
    auto res = kmpMatcher("aababaa", "aba");
    for (int x : res) cout << x << " ";  // 1 3
    return 0;
}
*/
```

<div data-component="KMPMatcherPointerJump"></div>

### 31.2.5 总体 $O(n + m)$ 证明：势能/摊销分析

**定理**：KMP 算法的总时间复杂度为 $O(n + m)$。

**证明（势能法）**：

分别对 `compute_pi`（$O(m)$）和 `kmp_matcher`（$O(n)$）进行分析，两者独立。

**对 `kmp_matcher` 的摊销分析**：

定义势能函数 $\Phi = q$（当前已匹配字符数）。

- 每次外层 `for` 迭代（即每次 `i` 递增），`q` **最多增加 1**（`if T[i] == P[q]: q += 1`）；
- 每次 `while` 内的 `q = pi[q-1]` 使 `q` **严格减小**（因为 $\pi[k-1] < k$），且 `while` 循环继续进行需要 `q > 0`；
- 初始 $q = 0$，始终 $q \geq 0$；
- 因此，`q` 的**总增量** ≤ $n$（每次 `i` 递增最多 +1），`q` 的**总减量**也 ≤ $n$（不能减出负数）；
- `while` 循环的总执行次数 = `q` 的总减量 ≤ $n$。

**结论**：外层 `for` 循环体中的总操作次数（包括所有 `while` 迭代）为 $O(n)$。

同理，`compute_pi` 中 `k` 的总增量 ≤ $m-1$，总减量 ≤ $m-1$，故整个函数为 $O(m)$。

**总复杂度**：$O(m) + O(n) = O(n + m)$。$\square$

### 31.2.6 KMP 的应用：周期子串检测

KMP 不只是用来匹配，π 数组本身就蕴含了模式串的**周期结构**。

**定理（字符串周期性）**：若 $m - \pi[m-1]$ 整除 $m$，则 $P$ 的最小正周期为 $m - \pi[m-1]$。

**例子**：

- P = `ABABABAB`（$m=8$），$\pi[7] = 6$，最小周期 = $8 - 6 = 2$：`AB`（重复4次）✓
- P = `ABAAB`（$m=5$），$\pi[4] = 2$，$5 - 2 = 3$，但 3 不整除 5，所以无整数周期。

**LeetCode #459 应用**：

> 给定字符串 s，判断 s 是否可以由其某个子串的重复构成。

```python
def repeatedSubstringPattern(s: str) -> bool:
    """
    利用 KMP π 数组判断字符串是否为某子串的重复。
    
    原理：s 可被整除当且仅当 m - π[m-1] 整除 m。
    等价技巧：(s + s)[1:-1] 中若包含 s 则返回 True（经典双倍串技巧）。
    
    时间: O(m)
    """
    m = len(s)
    pi = compute_pi(s)              # 利用上面定义的函数
    k = m - pi[m - 1]               # 候选最小周期长度
    return k < m and m % k == 0    # 必须整除

# 测试
print(repeatedSubstringPattern("abab"))    # True  (周期 "ab")
print(repeatedSubstringPattern("aba"))     # False
print(repeatedSubstringPattern("abcabcabcabc"))  # True (周期 "abc")
```

```cpp
bool repeatedSubstringPattern(string s) {
    int m = (int)s.size();
    vector<int> pi = computePi(s);
    int k = m - pi[m - 1];           // 候选最小周期
    return k < m && m % k == 0;
}
```

<div data-component="KMPPeriodDetector"></div>

---

## 31.3 Rabin-Karp 算法

KMP 通过预处理模式串来避免重复比较；Rabin-Karp 则采用了完全不同的哲学：**用哈希值代替字符串比较**。比较两个哈希值只需 $O(1)$，而通过**滚动哈希（Rolling Hash）**，每次窗口滑动也只需 $O(1)$。

### 31.3.1 多项式滚动哈希的数学原理

**基本思想**：将字符串视为以 $d$（通常取 $d = 31$ 或 $256$）为底数、$q$（大质数）为模的多项式。

对于字符串 $s = s_0 s_1 \ldots s_{m-1}$，其哈希值定义为：

$$H(s) = \left(\sum_{i=0}^{m-1} s_i \cdot d^{m-1-i}\right) \bmod q$$

即最高次项对应最左字符（类似十进制数的表示方式）。

**例子**：取 $d = 31$，$q = 10^9 + 7$，字母 $a=1, b=2, \ldots, z=26$。

字符串 `abc` 的哈希值：

$$H(\texttt{abc}) = (1 \cdot 31^2 + 2 \cdot 31^1 + 3 \cdot 31^0) \bmod q = (961 + 62 + 3) \bmod q = 1026$$

**为什么用大质数取模**？防止哈希值溢出，并降低碰撞概率（模大质数后哈希值均匀分布在 $[0, q)$）。

### 31.3.2 滚动更新：$O(1)$ 每步滑动

**核心公式**：

设文本 T 中位置 $s$ 到 $s+m-1$ 的窗口哈希值为 $H_s$，则位置 $s+1$ 到 $s+m$ 的哈希值：

$$H_{s+1} = \left(H_s - T[s] \cdot d^{m-1}\right) \cdot d + T[s+m]）\bmod q$$

**直觉**：去掉最高位字符 $T[s]$（减法），整体左移一位（乘 $d$），加入新字符 $T[s+m]$（加法）。

就像数字 `1234` 去掉首位 `1`、左移、加入新尾数字 `5`：`234 → 2340 → 2345`。

$$d^{m-1} \bmod q \text{ 可预先计算好（高次幂）}$$

**注意**：减法后可能出现负数，需加 $q$ 再取模：

$$H_{s+1} = \left((H_s - T[s] \cdot d^{m-1} \bmod q + q) \cdot d + T[s+m]\right) \bmod q$$

### 31.3.3 哈希冲突（Spurious Hit）与验证

**哈希冲突（Spurious Hit）**：两个不同的字符串哈希值相同。这是概率问题，无法完全避免。

**处理方案**：当哈希值匹配时，**额外进行 $O(m)$ 的字符串比较**确认真实匹配。

**冲突期望次数**：若模数 $q$ 是随机选取的质数，对长度 $n$ 的文本，期望只有 $O(n/q)$ 次伪匹配（Spurious Hit）。当 $q \geq n$（常见做法是 $q \sim 10^9$），期望冲突次数接近 0。

**最坏情况**：若恶意构造数据，所有 $n-m+1$ 个窗口都产生哈希冲突，则每次都需要 $O(m)$ 验证，退化为 $O(nm)$。实践中极罕见。

<div data-component="RabinKarpCollisionDemo"></div>

### 31.3.4 期望 $O(n + m)$ 完整实现

```python
from typing import List

def rabin_karp(T: str, P: str, d: int = 31, q: int = 10**9 + 7) -> List[int]:
    """
    Rabin-Karp 字符串匹配算法。
    
    d: 进制基数（通常取 31 或 256）
    q: 模数（大质数，防溢出 + 降低碰撞率）
    
    时间（期望）: O(n + m)
    时间（最坏）: O(nm)（极罕见，恶意碰撞时）
    空间: O(1)
    
    字母映射: 'a'→1, 'b'→2, ..., 'z'→26（避免 'a' 映射为 0 导致 "a" 和 "" 哈希相同）
    """
    n, m = len(T), len(P)
    results = []
    if m > n:
        return results

    # 字符到数字的映射（'a'=1,...,'z'=26；若处理任意字符则用 ord(c)）
    def char_val(c: str) -> int:
        return ord(c) - ord('a') + 1

    # 预计算 d^(m-1) mod q（最高次幂，去掉窗口首字符时用）
    h = 1
    for _ in range(m - 1):
        h = (h * d) % q

    # 计算模式串哈希值 + 文本第一个窗口的哈希值
    p_hash = 0   # 模式串哈希
    t_hash = 0   # 文本窗口哈希
    for i in range(m):
        p_hash = (p_hash * d + char_val(P[i])) % q
        t_hash = (t_hash * d + char_val(T[i])) % q

    # 滑动窗口
    for s in range(n - m + 1):
        # 哈希值匹配：进一步验证（避免伪匹配）
        if p_hash == t_hash:
            if T[s:s + m] == P:    # O(m) 验证，期望很少触发
                results.append(s)

        # 滚动更新下一个窗口的哈希值
        if s < n - m:
            t_hash = (
                (t_hash - char_val(T[s]) * h % q + q) * d    # 去掉左端字符，左移
                + char_val(T[s + m])                           # 加入右端字符
            ) % q

    return results


# ----- 测试 -----
print(rabin_karp("GEEKS FOR GEEKS", "GEEKS"))
# 输出: [0, 10]

print(rabin_karp("aababaa", "aba"))
# 输出: [1, 3]

print(rabin_karp("aaaa", "aaa"))
# 输出: [0, 1]  （重叠匹配）
```

```cpp
#include <bits/stdc++.h>
using namespace std;

/*
 * Rabin-Karp 字符串匹配
 *
 * d: 进制基数；q: 大质数取模
 * 时间（期望）: O(n + m)；最坏: O(nm)
 * 空间: O(1)
 */
vector<int> rabinKarp(const string& T, const string& P,
                      long long d = 31, long long q = 1e9 + 7) {
    int n = (int)T.size(), m = (int)P.size();
    vector<int> results;
    if (m > n) return results;

    // 字符映射 'a'→1,...,'z'→26
    auto val = [](char c) -> long long { return c - 'a' + 1; };

    // 预计算 d^(m-1) mod q
    long long h = 1;
    for (int i = 0; i < m - 1; ++i)
        h = h * d % q;

    // 计算初始哈希
    long long pHash = 0, tHash = 0;
    for (int i = 0; i < m; ++i) {
        pHash = (pHash * d + val(P[i])) % q;
        tHash = (tHash * d + val(T[i])) % q;
    }

    // 滑动窗口匹配
    for (int s = 0; s <= n - m; ++s) {
        if (pHash == tHash) {
            // 哈希值相同时，O(m) 验证防止伪匹配
            if (T.substr(s, m) == P)
                results.push_back(s);
        }
        // 滚动更新下一个窗口
        if (s < n - m) {
            tHash = ((tHash - val(T[s]) * h % q + q) * d
                     + val(T[s + m])) % q;
        }
    }
    return results;
}
```

<div data-component="RabinKarpRollingHash"></div>

### 31.3.5 多模式匹配扩展

Rabin-Karp 的一大优势：**$k$ 个模式串的同时匹配**，只需将所有模式串的哈希值存入一个哈希集合，每次窗口滑动后查询集合（$O(1)$），整体为 $O(n + \sum |P_i|)$—— 比对每个模式串分别运行 KMP 快 $k$ 倍。

```python
def rabin_karp_multi(T: str, patterns: List[str], d: int = 31, q: int = 10**9 + 7) -> dict:
    """
    多模式 Rabin-Karp：一次扫描文本，匹配所有模式串。
    假设所有模式串等长（若不等长，分组处理）。
    
    时间: O(n + k·m)（k 个模式串，每个长 m）
    """
    def char_val(c):
        return ord(c) - ord('a') + 1
    
    # 这里简化假设所有 pattern 等长
    if not patterns:
        return {}
    m = len(patterns[0])
    n = len(T)
    
    # 将所有模式哈希存入字典（哈希→模式串列表）
    pat_hashes: dict = {}
    for p in patterns:
        ph = 0
        for c in p:
            ph = (ph * d + char_val(c)) % q
        pat_hashes.setdefault(ph, []).append(p)
    
    results = {p: [] for p in patterns}
    h = pow(d, m - 1, q)  # d^(m-1) mod q，用内置快速幂
    
    t_hash = 0
    for i in range(m):
        t_hash = (t_hash * d + char_val(T[i])) % q
    
    for s in range(n - m + 1):
        if t_hash in pat_hashes:
            window = T[s:s + m]
            for p in pat_hashes[t_hash]:
                if window == p:
                    results[p].append(s)
        if s < n - m:
            t_hash = ((t_hash - char_val(T[s]) * h % q + q) * d + char_val(T[s + m])) % q
    
    return results

# 示例
T = "ahishers"
patterns = ["he", "she", "his", "hers"]
# 注意：等长假设不满足，实际需改为分组。此处仅示意多模式思路。
```

### 31.3.6 双模哈希降低碰撞概率

使用**两个不同的质数**分别取模，只有当两个哈希值都相同才认为"可能匹配"（再验证），碰撞概率从 $O(1/q)$ 降至 $O(1/q_1 q_2)$，对于 $q_1, q_2 \sim 10^9$ 碰撞概率约为 $10^{-18}$，实践中几乎不可能碰撞。

```python
def rabin_karp_double_hash(T: str, P: str) -> List[int]:
    """
    双模哈希 Rabin-Karp，碰撞概率 ≈ 10^{-18}，实践中无需验证步骤。
    """
    d = 31
    q1, q2 = 10**9 + 7, 10**9 + 9    # 两个大质数

    def char_val(c): return ord(c) - ord('a') + 1

    n, m = len(T), len(P)
    results = []
    if m > n: return results

    h1, h2 = pow(d, m - 1, q1), pow(d, m - 1, q2)
    ph1 = ph2 = th1 = th2 = 0

    for i in range(m):
        ph1 = (ph1 * d + char_val(P[i])) % q1
        ph2 = (ph2 * d + char_val(P[i])) % q2
        th1 = (th1 * d + char_val(T[i])) % q1
        th2 = (th2 * d + char_val(T[i])) % q2

    for s in range(n - m + 1):
        if ph1 == th1 and ph2 == th2:       # 双重哈希匹配，几乎确定真匹配
            results.append(s)               # 不再需要 O(m) 验证
        if s < n - m:
            th1 = ((th1 - char_val(T[s]) * h1 % q1 + q1) * d + char_val(T[s + m])) % q1
            th2 = ((th2 - char_val(T[s]) * h2 % q2 + q2) * d + char_val(T[s + m])) % q2

    return results
```

```cpp
vector<int> rabinKarpDoubleHash(const string& T, const string& P) {
    const long long d = 31, q1 = 1e9 + 7, q2 = 1e9 + 9;
    auto val = [](char c) -> long long { return c - 'a' + 1; };

    int n = (int)T.size(), m = (int)P.size();
    vector<int> results;
    if (m > n) return results;

    auto pw = [&](long long base, int exp, long long mod) {
        long long r = 1;
        for (; exp > 0; exp >>= 1) {
            if (exp & 1) r = r * base % mod;
            base = base * base % mod;
        }
        return r;
    };
    long long h1 = pw(d, m-1, q1), h2 = pw(d, m-1, q2);
    long long ph1=0, ph2=0, th1=0, th2=0;

    for (int i = 0; i < m; ++i) {
        ph1 = (ph1*d + val(P[i])) % q1;
        ph2 = (ph2*d + val(P[i])) % q2;
        th1 = (th1*d + val(T[i])) % q1;
        th2 = (th2*d + val(T[i])) % q2;
    }
    for (int s = 0; s <= n-m; ++s) {
        if (ph1 == th1 && ph2 == th2)
            results.push_back(s);
        if (s < n-m) {
            th1 = ((th1 - val(T[s])*h1%q1 + q1)*d + val(T[s+m])) % q1;
            th2 = ((th2 - val(T[s])*h2%q2 + q2)*d + val(T[s+m])) % q2;
        }
    }
    return results;
}
```

---

## 31.4 Boyer-Moore 算法

Boyer-Moore（BM）算法由 Boyer 和 Moore 于 1977 年提出，是**实践性能最强**的字符串匹配算法。其核心思想有两点反直觉之处：

1. **从模式串右端开始比较**（而不是从左到右）；
2. **一次可以跳过多个字符**（在最优情况下可跳过整个 $m$ 个字符）。

这使得 BM 在随机文本中的平均比较次数约为 $O(n/m)$——模式串越长，速度越快！

### 31.4.1 坏字符规则（Bad Character Heuristic）

**定义**：当模式串 P[j] 与文本 T[i] 失配时，T[i] 称为**坏字符（Bad Character）**。

**规则**：将模式串右移，使模式串中最靠右出现的 T[i] 与文本中的 T[i] 对齐。

**预处理**：构建坏字符表 `bad_char[c]`，记录字符 `c` 在模式串中**最后一次出现**的位置。

```
bad_char[c] = max{k : P[k] == c, 0 ≤ k < m}，若 c 不在 P 中则为 -1
```

**右移量计算**（当前对齐起点为 $s$，失配位置为 $j$）：

$$\text{shift\_bc} = j - \text{bad\_char}[T[s+j]]$$

若此值 ≤ 0（坏字符在模式串更右的位置），取至少移动 1 步：$\max(1, j - \text{bad\_char}[T[s+j]])$

**图示**（P = `ABCDA`，文本在 s=2 处失配）：

```
文本: ... X C D A B ...
模式:     A B C D A    ← 在 j=2 时，T[i]='D'，P[2]='C' 失配
                       坏字符 'D' 在 P 中最后出现在 j=3（bad_char['D']=3）
                       移动量 = 2（失配位置）- 3（bad_char['D']）= -1 → 取 max(1,-1)=1
```

坏字符规则有时无效（移动量为负），这时好后缀规则往往能给出更大的跳跃。

### 31.4.2 好后缀规则（Good Suffix Heuristic）

**定义**：当 P[j+1..m-1] 已经与文本匹配（称为**好后缀 t**），但 P[j] 失配时，利用好后缀在模式串中的重现位置来决定移动量。

好后缀规则由两个子情形组成：

**情形 1**：好后缀 $t$ 在 P 的其他位置也出现，向右移动使第二个出现的 $t$ 与文本对齐。

```
文本: ... C A B A B ...
模式:     A B A B      ← 好后缀为 "AB"（j=1时失配）
           A B A B    ← "AB" 在 P[0..1] 也出现，移动 2 位对齐
```

**情形 2**：好后缀 $t$ 在 P 中不再完整出现，但 $t$ 的某个**后缀**与 **P 的前缀**相同，则移动使该前缀与文本对齐。

**情形 3**：以上两种都不适用，直接移动整个 $m$ 位。

**预处理**：构建好后缀表 `good_suffix[j]`，表示当在位置 $j$ 失配时，模式串应移动的步数。构建过程较为复杂，需要 $O(m)$ 时间（通过两个辅助数组 `suffix` 和 `prefix`）。

### 31.4.3 两规则取最大移动量

BM 算法每次取坏字符规则和好后缀规则给出的移动量中**较大的那个**：

$$\text{shift} = \max(\text{shift\_bc}, \text{shift\_gs})$$

这保证了模式串每次都跳过尽可能多的无用位置。

### 31.4.4 Boyer-Moore 简化版（Bad Character Only）实现

好后缀规则实现较复杂，以下提供**仅含坏字符规则**的简化版（Horspool 变体），已能在实践中获得很好性能：

```python
from typing import List

def build_bad_char_table(P: str, alphabet_size: int = 256) -> List[int]:
    """
    构建坏字符表。
    bad_char[ord(c)] = P 中字符 c 最后出现的下标，不存在则为 -1。
    时间: O(m + σ)；σ 为字母表大小
    """
    m = len(P)
    bad_char = [-1] * alphabet_size
    for i in range(m):
        bad_char[ord(P[i])] = i     # 记录每个字符最后出现的位置
    return bad_char


def boyer_moore_bad_char(T: str, P: str) -> List[int]:
    """
    Boyer-Moore 简化版（仅坏字符规则）。
    
    从模式串右端开始比较，失配时利用坏字符表大步右移。
    
    时间（平均）: O(n/m)（最佳情形，随机文本）
    时间（最坏）: O(nm)（如模式 'aab' 在全 'a' 文本中）
    空间: O(σ)（σ = 字母表大小）
    
    注意：仅坏字符规则的最坏情况较差，完整 BM 需加好后缀规则。
    """
    n, m = len(T), len(P)
    bad_char = build_bad_char_table(P)
    results = []

    s = 0                           # s: 当前对齐起点
    while s <= n - m:
        j = m - 1                   # j: 从模式串最右端开始比较

        # 从右到左比较
        while j >= 0 and P[j] == T[s + j]:
            j -= 1

        if j < 0:
            # j < 0 说明整个模式串匹配成功！
            results.append(s)
            # 移动：使下一个可能匹配的位置对齐
            # 若 s+m < n，利用 bad_char 移动；否则移动 1
            if s + m < n:
                s += m - bad_char[ord(T[s + m])]
            else:
                s += 1
        else:
            # 失配：坏字符右移
            shift = j - bad_char[ord(T[s + j])]
            s += max(1, shift)       # 至少移动 1 步（防止负移动）

    return results


# ----- 测试 -----
print(boyer_moore_bad_char("HERE IS A SIMPLE EXAMPLE", "EXAMPLE"))
# 输出: [17]

print(boyer_moore_bad_char("AAAAAA", "AAA"))
# 输出: [0, 1, 2, 3]

print(boyer_moore_bad_char("ABAAABCD", "ABC"))
# 输出: [4]
```

```cpp
#include <bits/stdc++.h>
using namespace std;

/*
 * Boyer-Moore 简化版（仅坏字符规则）
 *
 * 时间（平均）: O(n/m)；最坏: O(nm)
 * 空间: O(σ)
 */
vector<int> boyerMooreBadChar(const string& T, const string& P) {
    const int SIGMA = 256;
    int n = (int)T.size(), m = (int)P.size();
    vector<int> results;
    if (m > n) return results;

    // 构建坏字符表
    vector<int> badChar(SIGMA, -1);
    for (int i = 0; i < m; ++i)
        badChar[(unsigned char)P[i]] = i;     // 记录最后出现位置

    int s = 0;                                 // 当前对齐起点
    while (s <= n - m) {
        int j = m - 1;                         // 从右端开始

        // 从右向左比较
        while (j >= 0 && P[j] == T[s + j])
            --j;

        if (j < 0) {
            // 完整匹配
            results.push_back(s);
            s += (s + m < n) ? m - badChar[(unsigned char)T[s + m]] : 1;
        } else {
            // 坏字符移动
            int shift = j - badChar[(unsigned char)T[s + j]];
            s += max(1, shift);
        }
    }
    return results;
}

/*
 * 好后缀规则预处理（辅助函数）
 * 完整 Boyd-Moore 需要此函数，这里给出核心逻辑
 */
vector<int> buildGoodSuffix(const string& P) {
    int m = (int)P.size();
    vector<int> gs(m, 0);               // gs[j] = 在位置 j 失配时应移动的步数
    vector<int> suffix(m, 0);          // suffix[i] = P 以 P[i] 结尾的后缀与 P 的某后缀的最长公共后缀长度

    // Step 1: 计算 suffix 数组
    suffix[m - 1] = m;
    int g = m - 1;
    for (int i = m - 2; i >= 0; --i) {
        if (i > g && suffix[i + m - 1 - (m-1)] < i - g)
            suffix[i] = suffix[i + m - 1 - (m-1)];
        else {
            g = min(g, i);
            while (g >= 0 && P[g] == P[g + m - 1 - i]) --g;
            suffix[i] = i - g;
        }
    }

    // Step 2: 由 suffix 数组构建 gs 表（简化版，非完整 BM）
    // 实际完整实现需处理情形1和情形2，此处略
    for (int j = 0; j < m; ++j) gs[j] = m;   // 默认移动 m 步

    return gs;
}
```

<div data-component="BoyerMooreShiftDemo"></div>

### 31.4.5 KMP vs Rabin-Karp vs Boyer-Moore 综合对比

| 维度 | KMP | Rabin-Karp | Boyer-Moore（完整版） |
|---|---|---|---|
| **预处理时间** | $O(m)$ | $O(m)$ | $O(m + \sigma)$ |
| **匹配时间（最坏）** | $O(n)$ ✓ | $O(nm)$ 极罕见 | $O(nm)$（仅坏字符版）|
| **匹配时间（平均）** | $O(n)$ | $O(n)$ 期望 | $O(n/m)$ 🏆 |
| **额外空间** | $O(m)$（π 数组） | $O(1)$ | $O(m + \sigma)$ |
| **多模式匹配** | 需改用 AC 自动机 | ✓ 天然支持（哈希集合） | ✗ 复杂 |
| **实现复杂度** | ⭐⭐⭐ 中等 | ⭐⭐ 简单 | ⭐⭐⭐⭐⭐ 最复杂（好后缀） |
| **适用场景** | 通用；最坏情形稳定 | 多模式；哈希查找 | 实践性能极优；长模式串 |

**选型建议**：
- **面试 / 竞赛**：KMP（π 数组是考点，时间复杂度稳定）；
- **多模式匹配**：Rabin-Karp（配合哈希集合）或 Aho-Corasick（见 Chapter 32）；
- **工程实践**：Boyer-Moore 或其变体（如 bmh, Boyer-Moore-Horspool，libcxx 的 string::find 底层用 BM-Horspool）；
- **生物信息学（超长文本 + 超长模式）**：后缀阵列（见 Chapter 33）+ 各种在线工具（如 BWA、Bowtie）。

<div data-component="StringMatchComparison"></div>

---

## 31.5 经典 LeetCode 题解析

### 31.5.1 #28. 找出字符串中第一个匹配项的下标（KMP 模板）

> 给定字符串 haystack 和 needle，返回 needle 在 haystack 中第一次出现的下标，若不存在返回 -1。

这是 KMP 的标准模板题，也是面试中最高频考查的字符串匹配题。

```python
def strStr(haystack: str, needle: str) -> int:
    """
    LeetCode #28 - KMP 解法
    时间: O(n + m)；空间: O(m)
    """
    n, m = len(haystack), len(needle)
    if m == 0: return 0
    if m > n: return -1

    # 构建模式串的 π 数组
    pi = [0] * m
    k = 0
    for i in range(1, m):
        while k > 0 and needle[k] != needle[i]:
            k = pi[k - 1]
        if needle[k] == needle[i]:
            k += 1
        pi[i] = k

    # KMP 匹配
    q = 0
    for i in range(n):
        while q > 0 and haystack[i] != needle[q]:
            q = pi[q - 1]
        if haystack[i] == needle[q]:
            q += 1
        if q == m:
            return i - m + 1     # 返回第一个匹配位置

    return -1

# 测试
print(strStr("hello", "ll"))     # 2
print(strStr("aaaaa", "bba"))    # -1
print(strStr("mississippi", "issip"))  # 4
```

```cpp
int strStr(string haystack, string needle) {
    int n = haystack.size(), m = needle.size();
    if (m == 0) return 0;
    if (m > n) return -1;

    vector<int> pi(m, 0);
    for (int i = 1, k = 0; i < m; ++i) {
        while (k > 0 && needle[k] != needle[i]) k = pi[k-1];
        if (needle[k] == needle[i]) ++k;
        pi[i] = k;
    }

    for (int i = 0, q = 0; i < n; ++i) {
        while (q > 0 && haystack[i] != needle[q]) q = pi[q-1];
        if (haystack[i] == needle[q]) ++q;
        if (q == m) return i - m + 1;
    }
    return -1;
}
```

### 31.5.2 #187. 重复的 DNA 序列（Rabin-Karp / 哈希）

> 找出所有在 DNA 字符串 s 中出现超过一次的长度为 10 的子串。

这是 Rabin-Karp 滚动哈希的典型应用——固定窗口大小 $m=10$，滑动统计所有哈希。

```python
def findRepeatedDnaSequences(s: str) -> List[str]:
    """
    LeetCode #187 - 滚动哈希解法
    时间: O(n)；空间: O(n)
    
    字母表 {A, C, G, T}，映射为 {0, 1, 2, 3}，用4进制滚动哈希。
    """
    n, m = len(s), 10
    if n <= m: return []

    # {'A':0, 'C':1, 'G':2, 'T':3}
    val = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    d, q = 4, (1 << 20) - 1    # 模: 2^20-1（位运算友好的质数近似）

    # 计算初始窗口哈希（前10个字符）
    h = 0
    for c in s[:m]:
        h = (h * d + val[c]) & q    # 使用 & 代替 % 加速（仅当 q=2^k-1 时等价）

    seen = {h}
    result_set = set()

    high = pow(d, m - 1, q + 1)    # d^(m-1)，用于去除最高位

    for i in range(m, n):
        h = ((h - val[s[i - m]] * high) * d + val[s[i]]) & q
        if h in seen:
            result_set.add(s[i - m + 1:i + 1])    # 哈希匹配，记录子串（哈希可能有碰撞，用字符串去重）
        seen.add(h)

    return list(result_set)
```

```cpp
vector<string> findRepeatedDnaSequences(string s) {
    int n = s.size(), m = 10;
    if (n <= m) return {};

    unordered_map<char, int> val{{'A',0},{'C',1},{'G',2},{'T',3}};
    const int d = 4, q = (1 << 20) - 1;  // 2^20-1 作为掩码

    long long h = 0, high = 1;
    for (int i = 0; i < m - 1; ++i) high = (high * d) & q;

    for (int i = 0; i < m; ++i)
        h = (h * d + val[s[i]]) & q;

    unordered_set<long long> seen{h};
    unordered_set<string> resultSet;

    for (int i = m; i < n; ++i) {
        h = ((h - val[s[i - m]] * high) * d + val[s[i]]) & q;
        if (seen.count(h))
            resultSet.insert(s.substr(i - m + 1, m));
        seen.insert(h);
    }
    return vector<string>(resultSet.begin(), resultSet.end());
}
```

### 31.5.3 #459. 重复的子字符串（KMP 周期性质）

```python
def repeatedSubstringPattern(s: str) -> bool:
    """
    利用 KMP π 数组判断字符串是否为某子串的重复。
    
    核心定理：s 可由某子串重复构成 ⟺ len(s) % (len(s) - π[len(s)-1]) == 0
                                    且 len(s) - π[len(s)-1] < len(s)
    (即最小周期整除字符串长度)
    
    时间: O(n)；空间: O(n)
    """
    n = len(s)
    pi = [0] * n
    k = 0
    for i in range(1, n):
        while k > 0 and s[k] != s[i]:
            k = pi[k - 1]
        if s[k] == s[i]:
            k += 1
        pi[i] = k

    period = n - pi[n - 1]     # 候选最小周期长度
    return period < n and n % period == 0

# 测试
print(repeatedSubstringPattern("abab"))       # True  ("ab" 重复 2 次)
print(repeatedSubstringPattern("abcabcabcabc"))  # True ("abc" 重复 4 次)
print(repeatedSubstringPattern("aba"))        # False
print(repeatedSubstringPattern("aa"))         # True  ("a" 重复 2 次)
```

```cpp
bool repeatedSubstringPattern(string s) {
    int n = s.size();
    vector<int> pi(n, 0);
    for (int i = 1, k = 0; i < n; ++i) {
        while (k > 0 && s[k] != s[i]) k = pi[k-1];
        if (s[k] == s[i]) ++k;
        pi[i] = k;
    }
    int period = n - pi[n-1];
    return period < n && n % period == 0;
}
```

### 31.5.4 #1044. 最长重复子串（Rabin-Karp + 二分）

> 给定字符串 s，找出最长的重复子串（至少出现两次）。

**思路**：二分枚举子串长度 $L$（范围 $[1, n-1]$），对每个 $L$ 用 Rabin-Karp 在 $O(n)$ 内判断长度为 $L$ 的子串是否有重复——总时间 $O(n \log n)$。

```python
def longestDupSubstring(s: str) -> str:
    """
    LeetCode #1044 - 二分 + Rabin-Karp 滚动哈希
    时间: O(n log n)；空间: O(n)
    """
    n = len(s)
    d, q = 31, 10**9 + 7
    nums = [ord(c) - ord('a') + 1 for c in s]  # 字符映射

    def has_dup(length: int) -> int:
        """
        判断长度为 length 的子串是否有重复出现。
        有则返回起始位置，无则返回 -1。
        """
        h = 0
        high = 1
        for i in range(length - 1):
            high = high * d % q
        for i in range(length):
            h = (h * d + nums[i]) % q

        seen = {h: [0]}            # 哈希值 → 起始位置列表
        for i in range(1, n - length + 1):
            h = (h - nums[i - 1] * high % q + q) * d % q
            h = (h + nums[i + length - 1]) % q
            if h in seen:
                # 哈希碰撞验证
                for prev in seen[h]:
                    if s[prev:prev + length] == s[i:i + length]:
                        return i    # 真正的重复
            seen.setdefault(h, []).append(i)
        return -1

    # 二分查找最大的有重复子串的长度
    lo, hi = 1, n - 1
    ans_pos, ans_len = -1, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        pos = has_dup(mid)
        if pos != -1:
            ans_pos, ans_len = pos, mid
            lo = mid + 1
        else:
            hi = mid - 1

    return s[ans_pos:ans_pos + ans_len] if ans_pos != -1 else ""

# 测试
print(longestDupSubstring("banana"))   # "ana"
print(longestDupSubstring("abcd"))     # ""（无重复）
```

---

## 本章总结

### 核心知识点回顾

**朴素算法**：
- 时间 $O(nm)$，根本原因是失配后丢弃所有历史信息；
- 理解其低效后，才能更深刻地欣赏 KMP 的精妙。

**KMP 算法**（最重要！）：
- **π 数组**：$\pi[i]$ = P[0..i] 的最长真前缀后缀长度，线性 $O(m)$ 构建；
- **失配时**：模式指针跳到 $\pi[q-1]$，文本指针不动；
- **总时间** $O(n + m)$，摊销分析保证文本指针永不回退；
- **额外应用**：$\pi[m-1]$ 可检测字符串的最小周期。

**Rabin-Karp 算法**：
- **滚动哈希**：$O(1)$ 更新窗口哈希值，避免每次 $O(m)$ 比较；
- **期望** $O(n + m)$，但存在（极低概率的）$O(nm)$ 最坏情况；
- **多模式匹配**：天然优势，将模式哈希值存入集合一次扫描全部匹配；
- **双模哈希**：两个质数取模，碰撞概率趋近于 0。

**Boyer-Moore 算法**：
- **从右向左**比较模式串；
- **坏字符规则**：失配时查表大步右移；
- **好后缀规则**：利用已匹配后缀的重现位置跳跃（实现复杂）；
- **平均** $O(n/m)$，实践性能最强，但最坏仍可达 $O(nm)$。

### 面试高频考点速查

| 题目 | 核心技巧 | 难度 |
|---|---|---|
| LeetCode #28 | KMP 完整模板 | Medium |
| LeetCode #459 | π[m-1] 判断周期性 | Easy |
| LeetCode #187 | 固定窗口滚动哈希 | Medium |
| LeetCode #1044 | 二分 + 滚动哈希 | Hard |
| LeetCode #686 | KMP 拼接匹配 | Medium |
| LeetCode #214 | 最短回文串（KMP）| Hard |

### 常见错误与陷阱

> ⚠️ **π 数组索引混淆**：$\pi[0] = 0$ 是因为单字符无"真前缀"；许多教材用 1-indexed，与 0-indexed 代码互相转换时容易出错。本章统一使用 **0-indexed**。

> ⚠️ **滚动哈希的负数问题**：`(h - x + q) % q` 中必须先加 `q` 再取模，否则 Python 中虽然负数取模有保护（`-1 % 7 == 6`），但在 C++ 中会出错。

> ⚠️ **Rabin-Karp 的伪匹配**：哈希值匹配 ≠ 字符串匹配，必须在哈希命中后做 $O(m)$ 验证（或使用双模哈希省去验证）。

> ⚠️ **BM 坏字符移动量为负**：当坏字符在模式串中出现在失配位置右侧时，右移量为负，必须取 $\max(1, j - \text{bad\_char}[T[s+j]])$，否则模式串会向左移动！

> ⚠️ **重叠匹配**：KMP 在找到一个完整匹配后，通过 `q = π[q-1]` 继续寻找下一个（可能重叠的）匹配位置，而不是将 `q` 归零——这是 #1044 和 #686 等题的关键细节。

### 思考题

1. **KMP 与 Z 函数的关系**：Z 函数（Chapter 34）定义 $z[i]$ 为以 $s[i]$ 开头的子串与 $s$ 的最长公共前缀长度。π 数组与 Z 函数本质上等价——能否写出用 Z 函数替代 π 数组实现 KMP 的代码？

2. **Rabin-Karp 的最坏情况构造**：如何构造一个使 Rabin-Karp（单模哈希）退化到 $O(nm)$ 的输入？（提示：让所有窗口哈希值都等于模式哈希值）

3. **多个不同长度的模式串**：若 $k$ 个模式串长度各不相同，Rabin-Karp 的多模式方案需要如何修改？（提示：按长度分组）

4. **后缀数组的关系**：如果用后缀数组（Chapter 33）和 LCP 数组来解决 #1044，时间复杂度是多少？与二分+Rabin-Karp 相比有何优劣？

---

> 📖 **扩展阅读**：
> - CLRS 第4版 Chapter 32（String Matching，详尽的 KMP 和 Rabin-Karp 证明）
> - Dan Gusfield, *Algorithms on Strings, Trees and Sequences*（字符串算法的权威教材）
> - MIT 6.006 Lecture 9、Lecture 10（字符串匹配与 Trie）
> - Sedgewick《算法》第四版 Chapter 5.3（Boyer-Moore 详细实现）

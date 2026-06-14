# Chapter 6: 字符串基础（String Fundamentals）

> **前置知识**：Chapter 3 数组基础、Chapter 5 栈与队列
>
> **难度**：🟢 初级
>
> **核心目标**：掌握字符串的内存表示与常见操作的时间复杂度；熟练运用滑动窗口与哈希计数处理子串/异位词问题；理解中心扩展法检测回文；为后续 KMP 与 Rabin-Karp 高级算法打下坚实基础。

---

## 6.1 字符串内部表示

### 6.1.1 ASCII 与 Unicode（UTF-8 / UTF-16 编码差异）

**字符集（Character Set）** 是字符与整数编码之间的映射表；**编码（Encoding）** 是将这个整数写入内存的方式。两者是完全不同的概念，混淆二者是初学者最常犯的错误。

**ASCII**（American Standard Code for Information Interchange）：
- 7位编码，范围 `0–127`，共 128 个字符
- `'A'` = 65，`'a'` = 97，`'0'` = 48，`' '` = 32
- 仅覆盖英文字母、数字、常用标点与控制字符
- 每个字符**恰好 1 字节**，操作简单，是绝大多数面试题的字母集假设

**Unicode**：
- 覆盖全球所有书写系统，目前已分配超过 14 万个码点（code point）
- `U+0041` = `'A'`，`U+4E2D` = `'中'`，`U+1F600` = `'😀'`（Emoji）
- Unicode 字符集本身不规定存储方式，由编码方案决定

**UTF-8**（最常用，Web 标准）：
- 变长编码：ASCII 字符 1 字节，汉字 3 字节，Emoji 4 字节
- 向后兼容 ASCII（码点 `0–127` 编码方式完全相同）
- 优点：空间高效（英文文本与 ASCII 相同大小）
- 缺点：无法 $O(1)$ 随机访问第 $n$ 个字符（字符宽度不固定）

**UTF-16**：
- 基本多语言平面（BMP，`U+0000–U+FFFF`）用 2 字节，其余用 4 字节代理对
- Java / JavaScript / C# 的内部字符串表示（历史原因）
- 同样是变长编码，但多数常用字符恰好 2 字节

```python
# Python 3 字符串默认 Unicode，str 是码点序列
s = "hello"
print(len(s))          # 5，码点数（不是字节数）
print(s[0])            # 'h'，O(1) 随机访问（CPython 内部细节）

s2 = "中文"
print(len(s2))         # 2，两个 Unicode 字符
print(s2.encode('utf-8'))    # b'\xe4\xb8\xad\xe6\x96\x87'  → 6字节
print(s2.encode('utf-16'))   # b'\xff\xfe-N\x87e'  → 含BOM的4字节+BOM

# ord/chr 转换
print(ord('A'))        # 65
print(chr(65))         # 'A'
print(ord('中'))        # 20013
```

```cpp
// C++ 字符串
#include <string>

std::string ascii = "hello";     // 底层 char[]，每字符 1 字节（ASCII/UTF-8）
char ch = ascii[0];              // O(1) 随机访问
std::size_t bytes = ascii.size();  // 字节数（不是码点数！）

// 处理 UTF-8 中文（字节层面操作）
std::string utf8 = u8"中文";      // UTF-8 编码的字面量（C++11）
// utf8.size() = 6（字节），不是 2（字符数）

// C++20 的 char8_t 明确区分字节与码点
```

> ⚠️ **陷阱**：在 C/C++ 中，`strlen("中文")` 返回字节数 6，而不是字符数 2。面试题中若题目保证只含 ASCII 字符，则 `s.size()` = 字符个数，可安全使用。

### 6.1.2 Python 字符串的不可变性与 intern 机制

Python 中 `str` 是**不可变对象（immutable）**：一旦创建就不能修改任何字符。任何"修改"操作都会返回新字符串，原字符串不变。

```python
s = "hello"
# s[0] = 'H'    # ❌ TypeError: 'str' object does not support item assignment
s2 = 'H' + s[1:]   # ✅ 创建新字符串 "Hello"，原 s 不变

# 为什么不可变性对性能有影响？
# 不可变对象可以安全地作为字典键（hashable）
d = {"hello": 1}

# Intern 机制（字符串驻留）：
# 对于满足特定条件的字符串（短字符串、仅含字母数字下划线的标识符），
# Python 会缓存并复用对象，多个变量可能指向同一内存地址
a = "hello"
b = "hello"
print(a is b)          # True（被 intern，同一对象）

c = "hello world"      # 含空格，不会被自动 intern
d = "hello world"
print(c is d)          # 可能 False（CPython 实现细节，不应依赖）

# 正确的字符串相等比较
print(c == d)          # True ← 始终用 == 比较值，not "is"

# 字符串拼接的陷阱（见 6.1.4）
result = ""
for word in ["a"] * 10000:
    result += word     # 每次创建新对象，O(n²) 总开销
# 正确做法：
result = "".join(["a"] * 10000)  # O(n)
```

### 6.1.3 C 风格字符串与 C++ `std::string`

**C 风格字符串（null-terminated string）**：
- 本质是 `char[]` 数组，以 `'\0'`（ASCII 0）结尾作为终止标志
- `strlen(s)` 需要 $O(n)$ 遍历找 `'\0'`（不是 $O(1)$！）
- 没有边界检查，是历史上大量缓冲区溢出漏洞的根源

```c
// C 风格字符串
#include <string.h>
char s[] = "hello";     // 实际存储 ['h','e','l','l','o','\0']（6 字节）
int len = strlen(s);    // O(n) 遍历 → len = 5

char buf[10];
strcpy(buf, "hello");   // O(n) 拷贝，⚠️ 无越界检查（危险！）
strcat(buf, " world");  // O(m+n) 拼接，⚠️ 同样危险
```

**C++ `std::string`**：
- 封装了长度字段，`size()` 是 $O(1)$（不像 `strlen`）
- 小字符串优化（SSO, Small String Optimization）：长度 ≤ 15 字节的字符串直接存在栈对象内部，避免堆分配
- 支持 `+` 拼接（会创建新对象）、`+=` 拼接（均摊 $O(1)$）、`substr`（$O(k)$，$k$ 为子串长度）

```cpp
#include <string>
std::string s = "hello";
s.size();                  // O(1)，5
s.length();                // 同 size()
s += " world";             // O(k)，均摊 O(1) 追加
s.substr(0, 5);            // O(5) 拷贝子串 "hello"
s.find("world");           // O(nm) 朴素查找
s[0] = 'H';                // ✅ C++ std::string 可变！
s.at(0);                   // 同 []，但有越界检查（抛 std::out_of_range）

// build string efficiently
std::string result;
result.reserve(100);       // 预分配容量，避免多次 realloc
for (char c : "hello") result += c;  // 高效追加
```

### 6.1.4 字符串拼接的陷阱：循环拼接 O(n²) vs `join` O(n)

这是最经典的字符串性能陷阱，面试中常被考察：

**Python 循环拼接的复杂度分析**（以拼接 $n$ 个长度为 $l$ 的字符串为例）：

```
result = ""
result = "" + s1         → 拷贝 l 字节（新字符串长 l）
result = s1 + s2         → 拷贝 2l 字节（新字符串长 2l）
result = s1+s2 + s3      → 拷贝 3l 字节
...
result = s1+...+sn       → 拷贝 nl 字节
```

总拷贝 = $l + 2l + 3l + \ldots + nl = l \cdot \frac{n(n+1)}{2} = O(n^2 l)$

```python
# ❌ 低效：O(n²)
def slow_join(words):
    result = ""
    for w in words:
        result += w    # 每次创建新字符串
    return result

# ✅ 高效：O(n)
def fast_join(words):
    return "".join(words)   # 一次性扫描总长度，分配一次内存，拷贝一次

# 实测（n = 10000 个 "a"）：
# slow_join → ~3.5ms
# fast_join → ~0.1ms

# 通用高效拼接模式：用 list 收集，最后 join
parts = []
for item in data:
    parts.append(process(item))   # O(1) 均摊
result = "".join(parts)           # O(total_length)
```

```cpp
// C++ 中，std::string 的 += 是均摊 O(1)（类似 vector 的 push_back）
// 但 + 运算符会创建临时对象
std::string result = "";
for (const auto& s : words) {
    result += s;    // ✅ 均摊 O(1)，推荐
    // result = result + s;  // ❌ 每次创建临时对象，O(n²) 总开销
}

// 更高效：stringstream 或预分配
#include <sstream>
std::ostringstream oss;
for (const auto& s : words) oss << s;
std::string result = oss.str();   // O(n)
```

---

## 6.2 字符串基本操作复杂度

### 6.2.1 拼接、切片、反转的时间复杂度分析

**拼接（Concatenation）**：

| 语言/方式 | 复杂度 | 说明 |
|----------|--------|------|
| Python `a + b` | $O(m + n)$ | 创建新对象，单次拼接 |
| Python `a += b`（循环）| $O(n^2)$ | 每次均创建新字符串 |
| Python `"".join(list)` | $O(\sum len_i)$ | 最优，一次分配 |
| C++ `s1 + s2` | $O(m + n)$ | 创建临时对象 |
| C++ `s1 += s2` | 均摊 $O(n)$ | 类似 vector::push_back |

**切片（Slicing）**：

```python
s = "hello world"
sub = s[2:7]       # O(k)，k = 切片长度（拷贝 k 个字符）
# Python 不支持 O(1) 子串视图（不同于 C++ 的 std::string_view）

# C++17 std::string_view 实现 O(1) 子串（无拷贝）
```

```cpp
#include <string_view>
std::string s = "hello world";
std::string_view sv = s;           // O(1)，只保存指针+长度
std::string_view sub = sv.substr(2, 5);  // O(1)！不拷贝
// ⚠️ sv 依赖 s 的生命周期，s 销毁后 sv 变悬空引用
```

**反转**：

| 语言/方式 | 复杂度 | 写法 |
|----------|--------|------|
| Python `s[::-1]` | $O(n)$ | 创建新字符串 |
| Python `reversed(s)` | $O(n)$ | 惰性迭代器，join 后 $O(n)$ |
| C++ `std::reverse` | $O(n)$ | 原地修改（`std::string` 可变）|

```python
s = "hello"
rev = s[::-1]           # "olleh"，O(n) 拷贝
rev = "".join(reversed(s))  # 等价

# 反转单词顺序（面试常考）
sentence = "the sky is blue"
words = sentence.split()         # O(n)
words.reverse()                  # O(k)，k 为单词数，原地
result = " ".join(words)         # O(n)
# 总体 O(n)
```

### 6.2.2 字符串比较的字典序（逐字符 O(min(m,n))）

两字符串按**字典序（Lexicographic Order）** 比较：逐字符比较 Unicode 码点：

```python
# Python 字符串比较是字典序
print("abc" < "abd")    # True（第3位 c < d）
print("abc" < "abcd")   # True（前缀关系）
print("b" < "abc")      # False（'b' > 'a'）

# 复杂度：O(min(m, n))，找到第一个不同字符即停止
# 最坏情况（完全相同的字符串）：O(n)

# 注意大小写：大写字母 ASCII 值更小
print("A" < "a")        # True（65 < 97）
print("Z" < "a")        # True（90 < 97）
print(sorted(["banana", "Apple", "cherry"]))
# ['Apple', 'banana', 'cherry']（大写字母排在小写之前）

# 大小写无关比较
print("Hello".lower() == "hello".lower())   # True
```

```cpp
// C++ 字符串比较（字典序）
std::string a = "abc", b = "abd";
bool lt = (a < b);         // True，O(min(m,n))
int cmp = a.compare(b);    // <0 / 0 / >0，类似 strcmp
```

### 6.2.3 常见字符串操作复杂度汇总表

<div data-component="StringComplexityTable"></div>

以下是主要字符串操作的时间和空间复杂度汇总：

| 操作 | Python | C++ `std::string` | 说明 |
|------|--------|-------------------|------|
| 随机访问 `s[i]` | $O(1)$ | $O(1)$ | 直接索引 |
| 长度 `len(s)` / `s.size()` | $O(1)$ | $O(1)$ | 内部存储长度字段 |
| 拼接 `a + b` | $O(m+n)$ | $O(m+n)$ | 创建新对象 |
| 追加 `s += c` | $O(n)$（每次新建）| 均摊 $O(1)$ | Python 不可变，C++ 可变 |
| `join(list)` | $O(\Sigma len)$ | — | 最优批量拼接 |
| 切片 `s[i:j]` | $O(k)$ | $O(k)$（`substr`）| 拷贝 k 个字符 |
| 查找 `find(t)` | $O(nm)$ | $O(nm)$ | 朴素实现（最坏） |
| 反转 `s[::-1]` | $O(n)$ | $O(n)$（原地）| C++ 原地修改 |
| 比较 `s == t` | $O(n)$ | $O(n)$ | 逐字符比较 |
| 替换 `replace(a,b)` | $O(n)$ | $O(n)$ | 扫描 + 拷贝 |
| 分割 `split()` | $O(n)$ | $O(n)$ | 返回子串列表 |
| 哈希 `hash(s)` | $O(n)$ | 未缓存 $O(n)$ | Python 缓存哈希值 |

---

## 6.3 字符串匹配基础（暴力法）

### 6.3.1 朴素字符串匹配：O(nm) 最坏情况

**问题**：给定文本串 $T$（长度 $n$）和模式串 $P$（长度 $m$），找出 $P$ 在 $T$ 中所有出现位置。

**朴素算法（Naive / Brute-Force）**：对文本每个可能的起始位置 $i$（$0 \leq i \leq n-m$），逐字符检查 $T[i..i+m-1]$ 是否等于 $P$。

```python
def naive_search(text: str, pattern: str) -> list[int]:
    """
    朴素字符串匹配，返回所有匹配起始位置
    时间：O(nm) 最坏，O(n) 最好（第一个字符就失配）
    空间：O(1)（不含结果列表）
    """
    n, m = len(text), len(pattern)
    results = []

    for i in range(n - m + 1):        # 外循环：文本起始位置
        j = 0
        while j < m and text[i + j] == pattern[j]:   # 内循环：逐字符验证
            j += 1
        if j == m:                     # 全部匹配
            results.append(i)

    return results

# 示例
print(naive_search("abababab", "abab"))   # [0, 2, 4]
print(naive_search("aaaaaa", "aa"))       # [0, 1, 2, 3, 4]

# Python 内置 in / find 在 CPython 中使用优化过的 Boyer-Moore-Horspool：
"abab" in "abababab"          # True
"abababab".find("abab")       # 0（第一个位置）
"abababab".index("abab")      # 0（同 find，失败时抛异常）
```

```cpp
// C++ 朴素字符串匹配
#include <vector>
#include <string>

std::vector<int> naiveSearch(const std::string& text, const std::string& pattern) {
    int n = text.size(), m = pattern.size();
    std::vector<int> results;

    for (int i = 0; i <= n - m; i++) {
        int j = 0;
        while (j < m && text[i + j] == pattern[j]) j++;
        if (j == m) results.push_back(i);
    }
    return results;
}

// C++ std::string::find 使用更优算法
size_t pos = text.find(pattern);          // 第一个匹配位置，O(nm) 最坏
while (pos != std::string::npos) {
    results.push_back(pos);
    pos = text.find(pattern, pos + 1);    // 从下一位置继续
}
```

**复杂度分析**：

| 场景 | 时间复杂度 | 示例 |
|------|-----------|------|
| 最好情况 | $O(n)$ | 模式每次第一字符就失配（如 `T="aaa..."`, `P="b..."`）|
| 平均情况（随机字符串）| $O(n)$ | 平均每次只比较少量字符 |
| 最坏情况 | $O(nm)$ | `T="aaa...a"`, `P="aaa...ab"`（失配总在最后一位）|

### 6.3.2 最坏情况示例：`"aaa...a"` 匹配 `"aaa...ab"`

<div data-component="NaiveStringMatchTrace"></div>

```python
# 最坏情况构造
n = 10
text    = "a" * n              # "aaaaaaaaaa"
pattern = "a" * (n // 2) + "b" # "aaaaab"

# 外循环执行 n - m + 1 = 5 次
# 每次内循环都走到 m - 1 = 4 步才发现失配（第5位 'a' ≠ 'b'）
# 总比较次数 = 5 × 5 = 25 ≈ O(nm)

# 验证：
matches = naive_search(text, pattern)   # []（无匹配）
```

**图示说明**（$T = $ `"aaaaaaa"`, $P = $ `"aab"`）：

```
i=0: a a a a a a a
     a a b            "aa" 匹配，第3位失配
i=1:   a a a a a a
       a a b          "aa" 匹配，第3位失配
i=2:     a a a a a
         a a b        "aa" 匹配，第3位失配
i=3:       a a a a
           a a b      "aa" 匹配，第3位失配
i=4:         a a a
             a a b    "aa" 匹配，第3位失配
```

每次外循环都需要走完模式串长度（$m-1$）步才发现失配，造成大量**重复比较**。

### 6.3.3 引出高级算法（KMP / Rabin-Karp）的必要性

朴素算法的核心问题：**失配后回退过多**。

- **KMP 算法**：利用模式串的自身结构（前缀函数/失败函数），失配后不从头重来，而是跳转到已知的最长有效前缀处，保证 $O(n + m)$。
- **Rabin-Karp 算法**：用**滚动哈希**将内循环的逐字符比较变为 $O(1)$ 的哈希比较，均摊 $O(n + m)$，偶发哈希碰撞后再验证。
- **Boyer-Moore 算法**：从模式串末尾向前比较，利用"坏字符"和"好后缀"规则大幅跳过，实践中最快，$O(n/m)$ 最佳。

这些高级算法将在后续章节（字符串进阶）详细介绍。本章打好基础：理解为何朴素法低效，以及掌握朴素法用于正确性验证（对高级算法做对比测试）。

---

## 6.4 常见字符串技巧

### 6.4.1 字符频次统计（哈希表 / 26 桶 / Counter）

字符频次统计是字符串题目中最基础的工具，应熟练掌握各种实现方式：

**方法 1：哈希表（通用，支持任意字符集）**

```python
from collections import Counter

s = "abracadabra"

# 方法1：Counter（最简洁）
cnt = Counter(s)
print(cnt)   # Counter({'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1})
print(cnt['a'])    # 5
print(cnt['z'])    # 0（不存在时返回 0，不抛异常）

# 方法2：defaultdict
from collections import defaultdict
cnt = defaultdict(int)
for ch in s:
    cnt[ch] += 1

# 方法3：普通 dict
cnt = {}
for ch in s:
    cnt[ch] = cnt.get(ch, 0) + 1
```

**方法 2：26 桶数组（仅含小写字母时最高效）**

```python
def char_freq_array(s: str) -> list[int]:
    """O(n) 时间，O(26) = O(1) 空间（只含小写字母）"""
    freq = [0] * 26
    for ch in s:
        freq[ord(ch) - ord('a')] += 1
    return freq

# 适用场景：题目保证仅含小写字母（a-z），数组下标 0-25 对应 a-z
# 好处：比字典快 2-3 倍（数组访问 vs 哈希计算）
```

```cpp
// C++ 字符频次（ASCII，仅小写字母）
int freq[26] = {};
for (char c : s) freq[c - 'a']++;

// 通用（支持全 ASCII）
int freq[128] = {};
for (char c : s) freq[(unsigned char)c]++;

// 使用 unordered_map 处理 Unicode
#include <unordered_map>
std::unordered_map<char32_t, int> freq;
// 需配合 UTF-32 解码
```

**频次统计应用：最高频字符、唯一字符**

```python
s = "aabbccddee"
cnt = Counter(s)

# 出现最多的 k 个字符
print(cnt.most_common(2))    # [('a',2), ('b',2)]（前 2 个）

# 只出现一次的字符（面试 #387 第一个唯一字符）
def first_unique_char(s: str) -> int:
    cnt = Counter(s)
    for i, ch in enumerate(s):
        if cnt[ch] == 1:
            return i
    return -1

print(first_unique_char("leetcode"))    # 0（'l' 是第一个唯一字符）
print(first_unique_char("aabb"))        # -1
```

### 6.4.2 异位词（Anagram）检测与分组

**异位词（Anagram）**：两个字符串互为异位词，当且仅当它们包含完全相同的字符（频次也相同），仅是排列顺序不同。

```
"silent" 和 "listen" 互为异位词
"anagram" 和 "nagaram" 互为异位词
"rat" 和 "car" 互为异位词
```

<div data-component="AnagramHasher"></div>

**方法 1：排序法**（简单但 $O(n \log n)$）

```python
def is_anagram_sort(s: str, t: str) -> bool:
    """O(n log n) 时间，O(n) 空间"""
    if len(s) != len(t):
        return False
    return sorted(s) == sorted(t)
    # sorted("listen") = ['e','i','l','n','s','t']
    # sorted("silent") = ['e','i','l','n','s','t']
    # 相等 → True
```

**方法 2：频次计数法**（最优 $O(n)$）

```python
def is_anagram(s: str, t: str) -> bool:
    """O(n) 时间，O(1) 空间（仅含 26 个字母）"""
    if len(s) != len(t):
        return False
    cnt = [0] * 26
    for a, b in zip(s, t):
        cnt[ord(a) - ord('a')] += 1
        cnt[ord(b) - ord('a')] -= 1
    return all(c == 0 for c in cnt)
    # 先为 s 的字符加 1，再为 t 的字符减 1
    # 最终全 0 → 频次完全相同 → 互为异位词

# 等价写法（更 Pythonic）：
def is_anagram_v2(s: str, t: str) -> bool:
    return Counter(s) == Counter(t)
```

```cpp
// C++ 异位词检测
bool isAnagram(const std::string& s, const std::string& t) {
    if (s.size() != t.size()) return false;
    int cnt[26] = {};
    for (int i = 0; i < (int)s.size(); i++) {
        cnt[s[i] - 'a']++;
        cnt[t[i] - 'a']--;
    }
    for (int c : cnt)
        if (c != 0) return false;
    return true;
}
```

**异位词分组**（LeetCode #49）：

```python
from collections import defaultdict

def group_anagrams(strs: list[str]) -> list[list[str]]:
    """
    核心思路：异位词排序后得到相同的"规范形式"，作为哈希键
    O(nk log k) 时间，n = 单词数，k = 平均长度
    """
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))    # "eat" → ('a','e','t')，"tea" → 同
        groups[key].append(s)
    return list(groups.values())

# 示例：
# strs = ["eat","tea","tan","ate","nat","bat"]
# 输出：[["bat"],["nat","tan"],["ate","eat","tea"]]

# 优化版：用频次元组作为键（避免排序，O(nk)）
def group_anagrams_v2(strs: list[str]) -> list[list[str]]:
    """O(nk) 时间——用 26 维频次向量作键，避免 O(k log k) 排序"""
    groups = defaultdict(list)
    for s in strs:
        cnt = [0] * 26
        for c in s:
            cnt[ord(c) - ord('a')] += 1
        key = tuple(cnt)          # 如 (1,0,0,...,1,0,...) 唯一标识频次分布
        groups[key].append(s)
    return list(groups.values())
```

### 6.4.3 双指针处理子串问题（最小覆盖子串、无重复最长子串）

**滑动窗口（Sliding Window）** 是处理子串/子数组问题的核心框架：维护左右指针 `left` 和 `right`，右指针扩张窗口，左指针在条件不满足时收缩窗口。

**题型 1：无重复字符的最长子串（LeetCode #3）**

```python
def length_of_longest_substring(s: str) -> int:
    """
    滑动窗口：O(n) 时间，O(min(m,n)) 空间（m 为字符集大小）
    维护窗口 [left, right)，保证窗口内无重复字符
    """
    char_idx = {}    # 字符 → 上次出现位置
    left = 0
    max_len = 0

    for right, ch in enumerate(s):
        # 若 ch 在窗口内出现过（上次位置 >= left），收缩左边界
        if ch in char_idx and char_idx[ch] >= left:
            left = char_idx[ch] + 1   # 跳过重复字符

        char_idx[ch] = right           # 更新字符位置
        max_len = max(max_len, right - left + 1)

    return max_len

# 示例：
# "abcabcbb" → 3（"abc"）
# "bbbbb"    → 1（"b"）
# "pwwkew"   → 3（"wke"）

# 替代写法（用 set 维护窗口字符集）：
def length_of_longest_substring_v2(s: str) -> int:
    window = set()
    left = 0
    max_len = 0
    for right, ch in enumerate(s):
        while ch in window:          # 不断收缩左边界直到无重复
            window.remove(s[left])
            left += 1
        window.add(ch)
        max_len = max(max_len, right - left + 1)
    return max_len
```

```cpp
// C++ 无重复最长子串
int lengthOfLongestSubstring(const std::string& s) {
    std::unordered_map<char, int> charIdx;
    int left = 0, maxLen = 0;
    for (int right = 0; right < (int)s.size(); right++) {
        char ch = s[right];
        if (charIdx.count(ch) && charIdx[ch] >= left)
            left = charIdx[ch] + 1;
        charIdx[ch] = right;
        maxLen = std::max(maxLen, right - left + 1);
    }
    return maxLen;
}
```

**题型 2：最小覆盖子串（LeetCode #76）**

```python
from collections import Counter

def min_window(s: str, t: str) -> str:
    """
    经典双指针 + 计数，O(|s| + |t|) 时间
    窗口内必须包含 t 的所有字符（频次 ≥ t 中的频次）
    """
    if not t or not s:
        return ""

    need = Counter(t)           # 还需要的字符数量
    missing = len(t)            # 还缺少的字符总数
    left = 0
    best = ""                   # 最短覆盖子串（空表示未找到）
    best_left = 0

    for right, ch in enumerate(s):
        # 右边界扩张
        if need[ch] > 0:        # 当前字符还在"缺少"状态
            missing -= 1
        need[ch] -= 1           # 无论是否需要，计数减 1

        # 窗口已满足覆盖条件（missing == 0）
        while missing == 0:
            window_len = right - left + 1
            if not best or window_len < len(best):
                best = s[left:right + 1]

            # 收缩左边界
            left_ch = s[left]
            need[left_ch] += 1
            if need[left_ch] > 0:   # 真正"缺失"了一个字符
                missing += 1
            left += 1

    return best

# 示例：
# min_window("ADOBECODEBANC", "ABC") → "BANC"
# min_window("a", "a")               → "a"
# min_window("a", "aa")              → ""

# 双指针框架总结：
# 扩张条件：right 每步向右移动
# 满足条件：missing == 0（窗口覆盖了 t 的所有字符）
# 收缩条件：在满足时，尽量缩小窗口（左移 left）
```

```cpp
// C++ 最小覆盖子串
std::string minWindow(const std::string& s, const std::string& t) {
    std::unordered_map<char, int> need;
    for (char c : t) need[c]++;
    int missing = t.size(), left = 0;
    int bestLeft = 0, bestLen = INT_MAX;

    for (int right = 0; right < (int)s.size(); right++) {
        if (need[s[right]]-- > 0) missing--;
        while (missing == 0) {
            if (right - left + 1 < bestLen) {
                bestLeft = left; bestLen = right - left + 1;
            }
            if (++need[s[left++]] > 0) missing++;
        }
    }
    return bestLen == INT_MAX ? "" : s.substr(bestLeft, bestLen);
}
```

**滑动窗口框架总结**：

```python
def sliding_window_template(s: str, t: str):
    # 初始化
    need = Counter(t)       # 需求计数（或其他条件）
    window = {}             # 窗口字符计数
    left = right = 0
    valid = 0               # 窗口中满足条件的字符种数

    while right < len(s):
        ch = s[right]
        right += 1
        # ① 扩张：更新窗口数据
        if ch in need:
            window[ch] = window.get(ch, 0) + 1
            if window[ch] == need[ch]:
                valid += 1      # 该字符满足需求

        # ② 判断是否应收缩
        while valid == len(need):
            # 记录答案（或者做其他操作）
            # ...

            # ③ 收缩：移除左边字符
            d = s[left]
            left += 1
            if d in need:
                if window[d] == need[d]:
                    valid -= 1
                window[d] -= 1
```

### 6.4.4 回文串检测：中心扩展法 O(n²)

**回文串（Palindrome）**：正向和反向读取完全相同的字符串，如 `"racecar"`、`"abba"`。

**暴力法**（$O(n^3)$）：枚举所有 $O(n^2)$ 个子串，每次检测 $O(n)$。不实用。

**中心扩展法**（$O(n^2)$，实践最常用）：

关键洞察：回文串一定有一个中心，从中心向两侧扩展。
- **奇数长度**回文：中心是某个字符 `s[i]`，从 `(i, i)` 向外扩展
- **偶数长度**回文：中心是某两相邻字符之间，从 `(i, i+1)` 向外扩展

```python
def expand_around_center(s: str, left: int, right: int) -> int:
    """从 (left, right) 出发向外扩展，返回最长回文子串长度"""
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
    # 退出时 s[left] != s[right] 或到达边界
    return right - left - 1   # 有效回文长度

def longest_palindrome(s: str) -> str:
    """
    中心扩展法：O(n²) 时间，O(1) 空间（不含结果字符串）
    """
    if not s:
        return ""
    start, max_len = 0, 1

    for i in range(len(s)):
        # 奇数长度：中心是字符 s[i]
        odd_len = expand_around_center(s, i, i)
        # 偶数长度：中心是 s[i] 和 s[i+1] 之间
        even_len = expand_around_center(s, i, i + 1)

        best = max(odd_len, even_len)
        if best > max_len:
            max_len = best
            # 反推子串起始位置
            start = i - (best - 1) // 2

    return s[start:start + max_len]

# 示例：
# longest_palindrome("babad")    → "bab" 或 "aba"
# longest_palindrome("cbbd")     → "bb"
# longest_palindrome("a")        → "a"
# longest_palindrome("racecar")  → "racecar"
```

```cpp
// C++ 中心扩展
int expandAroundCenter(const std::string& s, int l, int r) {
    while (l >= 0 && r < (int)s.size() && s[l] == s[r]) { l--; r++; }
    return r - l - 1;
}

std::string longestPalindrome(const std::string& s) {
    int start = 0, maxLen = 1;
    for (int i = 0; i < (int)s.size(); i++) {
        int odd = expandAroundCenter(s, i, i);
        int even = expandAroundCenter(s, i, i + 1);
        int best = std::max(odd, even);
        if (best > maxLen) {
            maxLen = best;
            start = i - (best - 1) / 2;
        }
    }
    return s.substr(start, maxLen);
}
```

<div data-component="PalindromeExpander"></div>

**Manacher 算法（$O(n)$）预告**：

Manacher 算法利用已知回文的对称性，避免重复扩展，将中心扩展法从 $O(n^2)$ 优化到 $O(n)$。核心思想是维护一个"最右回文边界 $R$"和对应中心 $C$：  
对于新的中心 $i$，若 $i < R$，则利用 $i$ 关于 $C$ 的镜像位置 $i'$，用 $P[i'] $初始化 $P[i]$，跳过已知匹配的部分。具体实现将在「字符串进阶」章节展开。

**判断回文多种方法对比**：

```python
def is_palindrome_naive(s: str) -> bool:
    return s == s[::-1]         # O(n)，最简洁

def is_palindrome_two_ptr(s: str) -> bool:
    l, r = 0, len(s) - 1
    while l < r:
        if s[l] != s[r]:
            return False
        l += 1; r -= 1
    return True                 # O(n)，早期退出更快

# 链表回文判断（LeetCode #234）：
# 找中点 → 反转后半 → 双指针比较
```

### 6.4.5 字符串转整数（atoi）的边界处理

**LeetCode #8（`myAtoi`）** 是考察边界处理能力的经典题，需要处理：

1. **前导空格**：跳过
2. **符号位**：判断 `+` / `-`
3. **非数字字符**：遇到立即停止
4. **整数溢出**：结果超出 `[-2³¹, 2³¹-1]` 时夹取边界值

```python
import math

def my_atoi(s: str) -> int:
    """
    字符串转整数，O(n) 时间，O(1) 空间
    """
    INT_MIN, INT_MAX = -2**31, 2**31 - 1

    i = 0
    n = len(s)

    # 步骤1：跳过前导空格
    while i < n and s[i] == ' ':
        i += 1

    if i == n:
        return 0

    # 步骤2：处理符号
    sign = 1
    if s[i] in ('+', '-'):
        if s[i] == '-':
            sign = -1
        i += 1

    # 步骤3：读取数字（遇到非数字立即停止）
    result = 0
    while i < n and s[i].isdigit():
        digit = ord(s[i]) - ord('0')

        # 步骤4：溢出检测（在乘以10并加digit前检测）
        if result > (INT_MAX - digit) // 10:
            return INT_MAX if sign == 1 else INT_MIN

        result = result * 10 + digit
        i += 1

    return sign * result

# 测试边界：
print(my_atoi("42"))            # 42
print(my_atoi("   -42"))        # -42
print(my_atoi("4193 with words"))  # 4193
print(my_atoi("words and 987"))    # 0
print(my_atoi("-91283472332"))     # -2147483648（INT_MIN）
print(my_atoi("2147483648"))       # 2147483647（INT_MAX，溢出夹取）
```

```cpp
// C++ atoi 实现
int myAtoi(const std::string& s) {
    int i = 0, n = s.size();
    while (i < n && s[i] == ' ') i++;   // 跳过空格
    if (i == n) return 0;

    int sign = 1;
    if (s[i] == '+' || s[i] == '-') {
        if (s[i] == '-') sign = -1;
        i++;
    }

    long result = 0;  // 使用 long 防止中间溢出
    while (i < n && isdigit(s[i])) {
        result = result * 10 + (s[i++] - '0');
        if (result * sign > INT_MAX) return INT_MAX;
        if (result * sign < INT_MIN) return INT_MIN;
    }
    return (int)(sign * result);
}
```

> ⚠️ **溢出检测顺序**：必须在 `result = result * 10 + digit` **之前**检测溢出，否则 `result * 10` 本身可能已经溢出（Python 自动大整数不溢出，但 C/C++ 中会 UB）。

---

## 6.5 经典题精讲

### 题 1：最长回文子串（LeetCode #5）

已在 6.4.4 节详细讲解中心扩展法（$O(n^2)$）。

**关键要点**：
- 奇偶两种中心必须都考虑（`expand(i, i)` 和 `expand(i, i+1)`）
- 从长度反推起始位置：`start = i - (best_len - 1) // 2`

### 题 2：无重复字符的最长子串（LeetCode #3）

已在 6.4.3 节（滑动窗口）详细讲解。这是滑动窗口的入门题，模板可直接套用。

### 题 3：字母异位词分组（LeetCode #49）

已在 6.4.2 节详细讲解（排序键 vs 频次元组键）。

**追问**：若字符串含 Unicode 字符怎么办？
- 用 `Counter(s)` 作哈希键：`key = tuple(sorted(Counter(s).items()))`，对任意字符集都适用

### 题 4：验证回文串（LeetCode #125）

只考虑字母和数字，忽略大小写：

```python
def is_palindrome_125(s: str) -> bool:
    """O(n) 时间，O(1) 空间（双指针原地）"""
    left, right = 0, len(s) - 1
    while left < right:
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() != s[right].lower():
            return False
        left += 1; right -= 1
    return True

# "A man, a plan, a canal: Panama" → True
# "race a car" → False
```

### 题 5：最小覆盖子串（LeetCode #76）

已在 6.4.3 节详细讲解（双指针 + `missing` 计数器）。

**复杂度**：$O(|s| + |t|)$ 时间，$O(|t|)$ 空间（哈希表存 `need`）。

---

## 6.6 本章小结

| 知识点 | 核心结论 |
|--------|---------|
| Python 字符串不可变 | 任何"修改"均创建新对象；用 `list` 收集再 `join` |
| 循环拼接复杂度 | `+=` 在循环中是 $O(n^2)$；`"".join(list)` 是 $O(n)$ |
| ASCII vs Unicode | `len(s)` 是码点数；`s.encode('utf-8')` 才是字节数 |
| 朴素匹配 | $O(nm)$ 最坏；`"aaa...a"` 配 `"aaa...ab"` 是典型最坏 |
| 字符频次统计 | 通用用 `Counter`；仅小写字母时用 `int[26]` 更快 |
| 异位词判断 | 排序 $O(n \log n)$ 或频次计数 $O(n)$；分组时用排序/频次元组作哈希键 |
| 滑动窗口模板 | 右扩张 + 左收缩；`missing` / `valid` 变量跟踪窗口满足程度 |
| 回文检测 | 中心扩展 $O(n^2)$，奇偶各一次；Manacher $O(n)$（见进阶章节）|
| atoi 边界 | 空格→符号→数字→溢出，四步不能颠倒；溢出检测在乘法**之前** |

**🎯 面试高频题单**：
- `#3` 无重复字符最长子串（滑动窗口入门，必会）
- `#5` 最长回文子串（中心扩展 $O(n^2)$，Manacher $O(n)$ 加分）
- `#49` 字母异位词分组（哈希键设计）
- `#76` 最小覆盖子串（双指针进阶，高频）
- `#125` 验证回文串（双指针 + isalnum）
- `#242` 有效的字母异位词（基础频次统计）
- `#8` 字符串转整数（边界处理能力）

**💡 思考题**：
1. Python 的 `str.find()` 底层用的是什么算法？比朴素法快在哪里？（提示：Boyer-Moore-Horspool 变体）
2. 为什么 Rabin-Karp 使用"滚动哈希"而不是每次重新计算哈希？这和差分数组的"增量更新"有什么相似之处？
3. 如何在 $O(n)$ 时间内找出字符串中**所有**回文子串的总数？（提示：Manacher + 中心扩展计数）

> **参考资料**：Sedgewick Chapter 5.1（Substring Search）；CLRS 第4版 Chapter 32（字符串匹配）；LeetCode 题解 #3、#5、#76

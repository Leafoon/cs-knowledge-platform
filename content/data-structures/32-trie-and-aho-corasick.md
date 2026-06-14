# Chapter 32: Trie 树与 AC 自动机

> **学习目标**：掌握 Trie（前缀树）的插入/查找操作、空间优化与压缩变体；理解 Aho-Corasick 自动机的失败链接构建与 KMP 失效函数的本质联系；掌握二进制 Trie 解决 XOR 最大值等位运算问题；能解决 #208、#212、#421 等高频题。

---

## 32.1 Trie 树（前缀树）

### 32.1.1 为什么需要 Trie？——从暴力搜索说起

想象你是一个搜索引擎，用户输入关键词 `appl`，你需要从 10 万个词汇中找出所有以 `appl` 开头的词（自动补全功能）。

**暴力做法**：遍历所有词汇，对每个词用 `startswith("appl")` 检查 → 时间复杂度 $O(N \cdot L)$（$N$ 为词汇数，$L$ 为平均词长）。词库越大越慢。

**Trie 的做法**：将所有词汇预先组织成一棵树，**公共前缀共享节点**。查找 `appl` 时，只需沿树走 4 步，找到对应节点，然后用 DFS 枚举所有后缀 → 预处理时间 $O(N \cdot L)$，查找时间 $O(L + 匹配数)$，与词库大小无关！

**Trie** 的名字来自 re**trie**val（检索），读音通常为 "try"（也有人读 "tree"，约定俗成都可以接受）。

**Trie 的核心思想**：将字符串集合组织成一棵多叉树（字母 Trie 则是最多 26 叉）。从根节点到某个节点的路径拼接起来就是一个字符串（或其前缀）。**公共前缀在树中对应公共路径**，因此极大节约了存储和查找开销。

```
词汇集合：{"apple", "app", "apply", "apt", "banana"}

Trie 结构：
           root
          /    \
         a      b
         |      |
         p      a
        / \     |
       p   t    n
       |   |    |
       l   *    a
      / \       |
     e   y      n
     |   |      |
     *   *      a
                |
                *
（* 表示 is_end = True）
```

可以看到 `apple`、`app`、`apply` 共享了 `a → p → p` 这段前缀路径，节点被复用了。

### 32.1.2 节点结构设计

Trie 节点需要两个核心字段：

1. **`children`**：指向子节点的指针数组（或哈希表）。对于只含小写字母的题目，`children[26]` 足够；若字母表更大，用 `HashMap<char, Node>` 代替。
2. **`is_end`**：布尔标志，标记该节点是否是某个字符串的**结尾**。

**关键区分**："节点存在"和"节点是某词的结尾"是两件事。

- `app` 是词汇 → 节点 $p_2$ 的 `is_end = True`
- `appl` 不是词汇 → 节点 $l$ 的 `is_end = False`
- 但 `appl` 节点存在（因为 `apple` 和 `apply` 都经过它）

```python
class TrieNode:
    def __init__(self):
        # children: 字母表大小为26的指针数组（None 表示无子节点）
        # 也可用 dict[str, TrieNode] 代替，节省空间但查找略慢
        self.children: list[TrieNode | None] = [None] * 26
        
        # is_end: 标记从根到当前节点的路径是否构成一个完整的单词
        # 注意：is_end = True 的节点可以有子节点（如 "app" 和 "apple" 同时存在）
        self.is_end: bool = False
```

```cpp
struct TrieNode {
    // children[i]: 第 i 个字母（'a'+i）对应的子节点
    // nullptr 表示该字母方向无子节点
    TrieNode* children[26];
    
    // is_end: 是否为某个单词的终止节点
    bool is_end;
    
    TrieNode() : is_end(false) {
        fill(children, children + 26, nullptr);
    }
};
```

### 32.1.3 INSERT 操作——逐字符建路

插入字符串 $w$（长度为 $m$） 的过程：

1. 从根节点出发
2. 对 $w$ 的每个字符 $c$，检查当前节点在 $c$ 方向是否有子节点
   - **有**：沿该子节点继续
   - **没有**：创建新节点，连接到当前节点的 $c$ 方向
3. 处理完所有字符后，将当前节点的 `is_end` 置为 `True`

**时间复杂度**：$O(m)$（$m$ 为字符串长度），无论字典中已有多少词。
**空间复杂度**：每个字符最多新建一个节点，共 $O(m)$。

```python
class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        """
        插入单词 word 到 Trie 中。
        
        时间: O(m)，m = len(word)
        空间: O(m)（最坏：word 完全没有公共前缀时需新建 m 个节点）
        """
        node = self.root
        for ch in word:
            idx = ord(ch) - ord('a')     # 将字符映射到 0~25 的索引
            if node.children[idx] is None:
                node.children[idx] = TrieNode()   # 按需创建节点
            node = node.children[idx]    # 沿已有/新建节点继续
        
        # 循环结束后，node 指向 word 最后一个字符对应的节点
        node.is_end = True               # 标记单词终止位置
```

```cpp
void insert(const string& word) {
    TrieNode* node = root;
    for (char ch : word) {
        int idx = ch - 'a';
        if (!node->children[idx]) {
            node->children[idx] = new TrieNode();
        }
        node = node->children[idx];
    }
    node->is_end = true;
}
```

**逐步演示**（插入 `"app"`）：

```
初始：root → 空节点

处理 'a'：root['a'] = null → 新建节点 A；走到 A
处理 'p'：A['p'] = null → 新建节点 P1；走到 P1
处理 'p'：P1['p'] = null → 新建节点 P2；走到 P2
结束：P2.is_end = True ✓

root → A → P1 → P2(is_end=True)
```

**再插入 `"apple"`**（公共前缀 `app` 节点被复用）：

```
处理 'a'：root['a'] = A（已存在）→ 走到 A
处理 'p'：A['p'] = P1（已存在）→ 走到 P1
处理 'p'：P1['p'] = P2（已存在）→ 走到 P2
处理 'l'：P2['l'] = null → 新建节点 L；走到 L
处理 'e'：L['e'] = null → 新建节点 E；走到 E
结束：E.is_end = True ✓
```

没有修改 `P2.is_end`（它仍是 `"app"` 的结尾），只是在它下面延伸了新路径。

### 32.1.4 SEARCH 与 PREFIX 操作

```python
    def search(self, word: str) -> bool:
        """
        查询 word 是否在 Trie 中（精确匹配，整个单词必须存在）。
        
        时间: O(m)
        边界条件：
        - 路径不完整（某字符无子节点） → False
        - 路径完整但 is_end = False → False（说明这只是某个长词的前缀）
        """
        node = self._traverse(word)
        # 节点存在 AND 是某个单词的终止位置，才算找到
        return node is not None and node.is_end
    
    def startsWith(self, prefix: str) -> bool:
        """
        查询是否有任何单词以 prefix 开头（前缀查询）。
        
        时间: O(m)，m = len(prefix)
        与 search 的区别：不要求 is_end = True，只要前缀路径存在即可
        """
        return self._traverse(prefix) is not None
    
    def _traverse(self, s: str) -> 'TrieNode | None':
        """辅助函数：沿字符串 s 走 Trie，返回最后节点（失败则返回 None）"""
        node = self.root
        for ch in s:
            idx = ord(ch) - ord('a')
            if node.children[idx] is None:
                return None    # 路径断开，s 连前缀都不存在
            node = node.children[idx]
        return node
```

```cpp
TrieNode* traverse(const string& s) {
    TrieNode* node = root;
    for (char ch : s) {
        int idx = ch - 'a';
        if (!node->children[idx]) return nullptr;
        node = node->children[idx];
    }
    return node;
}

bool search(const string& word) {
    TrieNode* node = traverse(word);
    return node && node->is_end;
}

bool startsWith(const string& prefix) {
    return traverse(prefix) != nullptr;
}
```

**`search` vs `startsWith` 的核心区别**：

| 操作 | 路径存在？ | `is_end = True`？ | 结果 |
|---|---|---|---|
| `search("app")` | ✓ | ✓ | `True` |
| `search("appl")` | ✓ | ✗ | `False` |
| `startsWith("appl")` | ✓ | 无所谓 | `True` |
| `search("xyz")` | ✗ | — | `False` |

### 32.1.5 前缀枚举与自动补全

获得某前缀对应节点后，可以用 DFS 枚举所有以该前缀开头的单词——这就是搜索引擎自动补全的底层逻辑。

```python
def autocomplete(self, prefix: str) -> list[str]:
    """
    返回所有以 prefix 开头的单词（自动补全）。
    
    实现：找到 prefix 对应节点，从该节点出发 DFS 枚举所有路径
    时间: O(m + 匹配词总字符数)
    空间: O(匹配词总字符数)（结果列表）
    """
    node = self._traverse(prefix)
    if node is None:
        return []    # 前缀本身都不存在，无结果
    
    results = []
    self._dfs(node, list(prefix), results)    # 从前缀结尾节点出发 DFS
    return results

def _dfs(self, node: TrieNode, path: list[str], results: list[str]) -> None:
    """
    DFS 遍历从 node 开始的所有路径，收集完整单词。
    
    path: 当前已走过的字符路径（用 list 而非 str，避免字符串拼接的 O(n) 开销）
    """
    if node.is_end:
        results.append(''.join(path))    # 找到完整单词，加入结果
    
    for i in range(26):
        if node.children[i] is not None:
            ch = chr(ord('a') + i)
            path.append(ch)                          # 向下延伸一个字符
            self._dfs(node.children[i], path, results)
            path.pop()                               # 回溯（撤销上一步）
```

```cpp
void dfs(TrieNode* node, string& path, vector<string>& results) {
    if (node->is_end) results.push_back(path);
    for (int i = 0; i < 26; i++) {
        if (node->children[i]) {
            path.push_back('a' + i);
            dfs(node->children[i], path, results);
            path.pop_back();    // 回溯
        }
    }
}

vector<string> autocomplete(const string& prefix) {
    TrieNode* node = traverse(prefix);
    if (!node) return {};
    string path = prefix;
    vector<string> results;
    dfs(node, path, results);
    return results;
}
```

<div data-component="TrieInsertSearch"></div>

### 32.1.6 空间消耗分析与 HashMap 优化

**标准数组实现的空间问题**：

每个节点存储 `children[26]`，即每个节点固定占用 26 个指针槽位，不管实际用到多少。

- 若词汇表中有 $N$ 个字符（所有字符串的字符总数），则最多有 $N$ 个节点
- 每个节点 26 个槽位 → 总空间 $O(N \cdot \Sigma)$，$\Sigma = 26$

**问题**：如果字母表很大（如 Unicode，$\Sigma$ 可达 65536），或字符分布稀疏（每个节点平均只有 1~2 个子节点），这 26 个槽位大部分都是 `None`，造成严重内存浪费。

**HashMap 优化**：

```python
class TrieNodeMap:
    def __init__(self):
        # 用字典代替固定数组：键为字符，值为子节点
        # 空间占用：O(实际子节点个数)，而非 O(Σ)
        # 代价：访问速度从 O(1) 变为 O(1) 均摊（哈希开销）
        self.children: dict[str, 'TrieNodeMap'] = {}
        self.is_end: bool = False
```

```cpp
struct TrieNodeMap {
    unordered_map<char, TrieNodeMap*> children;
    bool is_end = false;
};
```

**两种实现的对比**：

| 特性 | 数组 `children[Σ]` | HashMap `children` |
|---|---|---|
| 字母表大小 $\Sigma$ | 固定（通常 26） | 任意 |
| 每节点空间 | $O(\Sigma)$ | $O(k)$，$k$ 为实际子节点数 |
| 子节点访问 | $O(1)$ | $O(1)$ 均摊 |
| 适用场景 | 字母表小，较密集 | 字母表大或稀疏 |

**LeetCode #208 实战**：通常使用 `children[26]` 实现（题目明确仅含小写字母）。

### 32.1.7 压缩 Trie（Patricia Tree / Radix Tree）

标准 Trie 的另一个问题：**单孩子链**浪费节点。

若词典中只有 `["apple", "apply"]`，从根到 `p` 的路径是 `a → p → p → l`，中间这些只有一个孩子的节点（`a`、第一个 `p`、第二个 `p`）不携带任何"分叉信息"。

**压缩 Trie（Compressed Trie / Radix Tree）**：合并连续的单孩子链为一条边，边上标注对应的字符串而非单个字符。

```
标准 Trie（{apple, apply}）：        压缩后（Patricia Tree）：
root → a → p → p → l → e*             root
                      → y*               |
                                      "appl" （一条边，代表4个字符）
                                        |
                                     ┌──┴──┐
                                    "e"*  "y"*
```

压缩后节点数从 7 减少到 3，节省了大量空间。

- **节点数**：$O(N)$（$N$ 为字符串个数），而非 $O(\text{总字符数})$
- **代价**：边上存储字符串而非单字符，比较时需匹配整段字符串
- **应用**：Linux 内核的 Radix tree、IP 路由表（IP Routing Tables 通常用 Trie），Redis 的压缩列表等

### 32.1.8 DELETE 操作（补充）

删除 Trie 中的某个单词，需要小心处理两种情况：

1. **该词是另一个词的前缀**（如删除 `"app"`，但 `"apple"` 仍在）：只需将目标节点的 `is_end` 置为 `False`，**不能删节点本身**。
2. **该词删完后，其路径上无其他词**：可以递归删除叶节点，释放内存（可选，不影响正确性）。

```python
def delete(self, word: str) -> bool:
    """
    从 Trie 中删除 word。返回 True 表示删除成功，False 表示 word 不存在。
    
    策略：递归回溯，从叶节点向上清理空节点（彻底回收内存）
    """
    def _delete(node: TrieNode, word: str, depth: int) -> bool:
        """返回 True 表示当前节点可以被删除（无用空节点）"""
        if depth == len(word):
            if not node.is_end:
                return False      # word 不在 Trie 中
            node.is_end = False   # 取消终止标记
            # 若无子节点，当前节点变成孤立空节点，可以删除
            return all(c is None for c in node.children)
        
        idx = ord(word[depth]) - ord('a')
        child = node.children[idx]
        if child is None:
            return False          # 路径断开，word 不存在
        
        should_delete_child = _delete(child, word, depth + 1)
        if should_delete_child:
            node.children[idx] = None    # 删除子节点引用
            # 若当前节点也无用（非终止且无其他子节点）
            return not node.is_end and all(c is None for c in node.children)
        return False
    
    return _delete(self.root, word, 0)
```

```cpp
// 返回 true 表示 node 可以被安全删除
bool deleteHelper(TrieNode* node, const string& word, int depth) {
    if (depth == (int)word.size()) {
        if (!node->is_end) return false;
        node->is_end = false;
        // 若无子节点，此节点可以释放
        for (auto c : node->children) if (c) return false;
        return true;
    }
    int idx = word[depth] - 'a';
    if (!node->children[idx]) return false;
    
    bool shouldDelete = deleteHelper(node->children[idx], word, depth + 1);
    if (shouldDelete) {
        delete node->children[idx];
        node->children[idx] = nullptr;
        // 检查当前节点是否也可以清理
        if (node->is_end) return false;
        for (auto c : node->children) if (c) return false;
        return true;
    }
    return false;
}
```

---

## 32.2 Aho-Corasick 自动机（AC Automaton）

### 32.2.1 多模式匹配问题——为什么单靠 KMP 不够

**问题**：给定一段文本 $T$（长度 $n$）和一组模式串 $\{P_1, P_2, \ldots, P_k\}$，找出所有模式串在文本中的所有出现位置。

**朴素做法**：对每个 $P_i$ 单独运行 KMP → 总时间 $O(k \cdot n + \sum |P_i|)$。

- 如果 $k = 1000$ 个关键词，$n = 10^6$ 长度的文章 → $10^9$ 次操作，太慢！

**Aho-Corasick（AC 自动机）的思路**：

把所有模式串合并到一个 Trie 中，然后像 KMP 一样在 Trie 上添加**失败链接**（Failure Link）。文本指针在 Trie 上滑动——遇到匹配的字符向下走；遇到不匹配时，通过失败链接跳到其他节点重试，**而不是从 Trie 根重新开始**。

**时间复杂度**：预处理 $O(\sum |P_i| \cdot \Sigma)$，搜索 $O(n + 匹配总数)$——与模式串数量无关！

这与 KMP 的思想完全相同，只不过 KMP 处理单模式串，AC 自动机处理多模式串。事实上，AC 自动机就是 **"KMP 在 Trie 上的推广"**。

**现实应用**：

- 微博/抖音的**敏感词过滤系统**：词库有数万个敏感词，需要实时检测用户发布的内容
- 生物信息学：同时在 DNA 序列（超长文本, $n \sim 10^9$）中查找多个基因片段
- 网络入侵检测（NIDS）：Snort 等系统用 AC 自动机做多模式规则匹配

### 32.2.2 构建 Trie——多模式串的公共前缀树

第一步将所有模式串插入同一棵 Trie，这与普通 Trie 的 INSERT 完全相同：

```python
class AhoCorasick:
    def __init__(self):
        # AC 自动机的节点需要额外存储：
        # 1. fail：失败链接（跳到 Trie 中最长真后缀对应节点）
        # 2. output：该节点匹配的所有模式串集合（含通过 output link 传递的）
        self.children = [{}]   # 每个节点用 dict 表示子节点（索引 = 节点编号）
        self.fail = [0]        # 失败链接，初始根的 fail 指向自身（或 -1）
        self.output = [[]]     # 每个节点匹配的 pattern 下标列表
        self.size = 1          # 当前节点总数（0 = 根节点）
    
    def _new_node(self) -> int:
        """分配新节点，返回节点编号"""
        self.children.append({})
        self.fail.append(0)
        self.output.append([])
        self.size += 1
        return self.size - 1
    
    def insert(self, pattern: str, pattern_id: int) -> None:
        """插入第 pattern_id 号模式串"""
        node = 0    # 从根节点（0）出发
        for ch in pattern:
            if ch not in self.children[node]:
                self.children[node][ch] = self._new_node()
            node = self.children[node][ch]
        self.output[node].append(pattern_id)    # 标记该节点是模式串的结尾
```

### 32.2.3 在 Trie 上建失败链接——BFS 逐层计算

**失败链接（Fail Link）**的定义：节点 $u$ 的失败链接指向 Trie 中与 $u$ 表示的字符串具有**最长真后缀**匹配的另一个节点。

**类比 KMP**：

- KMP 的 π[i] 表示 $P[0..i]$ 的最长真前缀后缀长度
- AC 自动机的 fail[u] 表示：$u$ 对应字符串的最长真后缀，在 Trie 中对应的节点

例如，若 Trie 中有 `"he"`, `"she"`, `"his"`, `"hers"`：

```
Trie 结构：
               root
             /    \
            h      s
            |      |
            e*     h
           / \     |
          r   ...  e*
          |
          s*

节点 "she" 的字符串是 "she"
最长真后缀 = "he"（"he" 节点在 Trie 中存在！）
所以 fail["she"] → "he" 节点
```

**BFS 构建算法**：

```python
from collections import deque

def build(self) -> None:
    """
    BFS 逐层构建失败链接（自顶向下，保证计算 fail[u] 时 fail[parent] 已计算好）。
    
    核心规则（类比 KMP）：
    对于节点 u（由父节点 parent 经字符 c 到达）：
        v = fail[parent]        # 先跳到父的失败链接
        while v != root and v 没有字符 c 的子节点:
            v = fail[v]         # 继续跳（类比 KMP 的 while k > 0 and P[k] != P[i]）
        if v 有字符 c 的子节点 and 该子节点 != u:
            fail[u] = children[v][c]
        else:
            fail[u] = root      # 无匹配，回到根
        
    优化（类 AC 自动机标准构建）：
    用 "goto 函数"（自动跳转函数 delta）而非 fail + while 循环
    """
    q = deque()
    root = 0
    
    # 第一层：根节点的直接子节点的 fail 都指向 root（没有更短的真后缀）
    for ch, child in self.children[root].items():
        self.fail[child] = root    # 深度为 1 的节点，fail → root
        q.append(child)
    
    # BFS 逐层处理
    while q:
        u = q.popleft()
        
        # u 的 output 需要合并其 fail 节点的 output（输出链接的语义）
        self.output[u] = self.output[u] + self.output[self.fail[u]]
        
        for ch, v in self.children[u].items():
            # 为子节点 v 计算 fail 链接
            # v 是由 u 经字符 ch 到达的节点
            f = self.fail[u]    # 先跳到 u 的 fail 节点
            
            # 从 f 出发，找第一个有字符 ch 子节点的祖先
            while f != root and ch not in self.children[f]:
                f = self.fail[f]
            
            if ch in self.children[f] and self.children[f][ch] != v:
                # f 有 ch 子节点且不是 v 自身 → fail[v] = children[f][ch]
                self.fail[v] = self.children[f][ch]
            else:
                # 否则 fail[v] = root
                self.fail[v] = root
            
            q.append(v)
```

```cpp
// AC 自动机标准 C++ 实现（基于数组，竞赛常用模板）
const int MAXN = 100005;    // 模式串字符总数上限

struct AhoCorasick {
    int ch[MAXN][26], fail[MAXN], tot = 1;
    vector<int> output[MAXN];   // 每个节点匹配的模式串 ID 列表
    
    void insert(const string& s, int id) {
        int u = 1;  // 根为 1（0 作为 "空" 节点使用）
        for (char c : s) {
            int x = c - 'a';
            if (!ch[u][x]) ch[u][x] = ++tot;
            u = ch[u][x];
        }
        output[u].push_back(id);
    }
    
    void build() {
        queue<int> q;
        // 根的孩子：fail → 根（这里根 = 1，不存在的子节点 → 回到根）
        for (int c = 0; c < 26; c++) {
            if (ch[1][c]) {
                fail[ch[1][c]] = 1;
                q.push(ch[1][c]);
            } else {
                ch[1][c] = 1;   // 自循环：根没有字符 c → 直接回根（goto 函数优化）
            }
        }
        while (!q.empty()) {
            int u = q.front(); q.pop();
            // 合并 fail 节点的输出（output link）
            for (int id : output[fail[u]]) output[u].push_back(id);
            for (int c = 0; c < 26; c++) {
                if (ch[u][c]) {
                    fail[ch[u][c]] = ch[fail[u]][c];   // 关键：fail 指向 fail[u] 的 c-孩子
                    q.push(ch[u][c]);
                } else {
                    ch[u][c] = ch[fail[u]][c];  // goto 函数：无 c 孩子直接跳
                }
            }
        }
    }
};
```

<div data-component="AhoCorasickFailureLinks"></div>

### 32.2.4 输出链接（Output Link / Dictionary Link）

**单纯的失败链接不够！**

考虑文本 `"hers"`，模式串为 `{"he", "her", "hers"}`：

当我们匹配到节点 `"hers"` 时，该节点的 `output` 包含 `"hers"` 这个模式串。但 `"he"` 和 `"her"` 也在文本中出现了！它们是 `"hers"` 的后缀，分别对应 `fail` 链上的节点。

**输出链接（Output Link）**：在报告某节点匹配时，不仅报告该节点自身的模式串，还要沿失败链依次跳转，报告所有途经节点的模式串。

在上面的代码中，`self.output[u] = self.output[u] + self.output[self.fail[u]]` 这一步已经将 fail 节点的 output 合并进来，所以搜索时只需查看 `self.output[u]` 即可——无需在搜索时再跳 fail 链。这是**构建阶段提前处理**的标准优化。

### 32.2.5 文本搜索——O(n + 匹配数)

```python
def search(self, text: str, patterns: list[str]) -> dict[str, list[int]]:
    """
    在 text 中搜索所有模式串的所有出现位置。
    
    返回：{pattern: [出现起始位置列表]}
    时间: O(n + 匹配总数)，n = len(text)
    """
    results = {p: [] for p in patterns}
    node = 0    # 从根出发
    
    for i, ch in enumerate(text):
        # 类 KMP：current node 无字符 ch 的子节点时，跳 fail
        while node != 0 and ch not in self.children[node]:
            node = self.fail[node]
        
        if ch in self.children[node]:
            node = self.children[node][ch]    # 正常往下走
        # else: node = 0（根也没有），node 保持 0
        
        # 报告所有在位置 i 结束的模式串匹配（含通过 output link 传递的）
        for pid in self.output[node]:
            pattern = patterns[pid]
            start = i - len(pattern) + 1
            results[pattern].append(start)
    
    return results
```

```cpp
vector<pair<int,int>> search(const string& text, const vector<string>& patterns) {
    vector<pair<int,int>> results;  // (起始位置, pattern_id)
    int u = 1;  // 从根出发
    for (int i = 0; i < (int)text.size(); i++) {
        u = ch[u][text[i] - 'a'];  // goto 函数：直接跳（已经处理了 fail）
        // 报告所有在 i 位置结束的匹配（含 output link 合并后的输出）
        for (int pid : output[u]) {
            int start = i - (int)patterns[pid].size() + 1;
            results.emplace_back(start, pid);
        }
    }
    return results;
}
```

**完整演示**：

```
模式串：{"he", "she", "his", "hers"}
文本：  "ushers"

构建 AC 自动机后，扫描文本：
位置 0 'u'：root → root
位置 1 's'：root → s
位置 2 'h'：s → sh
位置 3 'e'：sh → she（she 的 fail = he 节点，output = ["she", "he"]）
             → 报告匹配："she" 在位置 1，"he" 在位置 2
位置 4 'r'：she → her
位置 5 's'：her → hers（output = ["hers"]）
             → 报告匹配："hers" 在位置 2

最终：she: [1], he: [2], hers: [2]
```

### 32.2.6 与 KMP 的本质联系

| 维度 | KMP | AC 自动机 |
|---|---|---|
| **处理规模** | 1 个模式串 | $k$ 个模式串 |
| **数据结构** | 线性 π 数组 | Trie + fail 数组 |
| **"失配"处理** | π 跳转 | fail 跳转 |
| **时间复杂度** | $O(n + m)$ | $O(n + \sum m_i + 匹配数)$ |
| **KMP 退化** | — | $k=1$ 时 AC 自动机 = KMP |

**核心等价性**：KMP 的 π 数组计算等价于在一个只有一条链（路径）的 Trie 上构建 fail 链接。AC 自动机把 KMP 推广到了任意 Trie 结构上。

想直观理解这个等价性：将 KMP 的模式串 `P` 视为一棵退化的 Trie（只有一条链），在这条链上建 fail 链接（BFS），就会得到和 KMP π 数组完全相同的结果。

### 32.2.7 应用实战——敏感词过滤系统

```python
class ContentFilter:
    """
    基于 AC 自动机的敏感词过滤系统。
    支持：敏感词检测、替换（用 * 号掩码）
    
    工厂流程：
    1. 加载敏感词库（一次性预处理 O(Σ|patterns|)）
    2. 对每篇内容进行过滤（O(n + 匹配数) 每篇）
    """
    def __init__(self, banned_words: list[str]):
        self.ac = AhoCorasick()
        self.patterns = banned_words
        for i, w in enumerate(banned_words):
            self.ac.insert(w, i)
        self.ac.build()
    
    def contains_sensitive(self, text: str) -> bool:
        """快速判断文本是否含敏感词"""
        matches = self.ac.search(text, self.patterns)
        return any(len(v) > 0 for v in matches.values())
    
    def mask(self, text: str) -> str:
        """将文本中所有敏感词替换为等长的 * 号"""
        text = list(text)
        matches = self.ac.search(''.join(text), self.patterns)
        for pattern, positions in matches.items():
            for start in positions:
                for j in range(start, start + len(pattern)):
                    text[j] = '*'
        return ''.join(text)

# 使用示例
filter = ContentFilter(["bad", "evil", "hack"])
print(filter.mask("this is a bad hack attempt"))
# 输出: "this is a *** **** attempt"
```

---

## 32.3 其他 Trie 变体

### 32.3.1 二进制 Trie（Bitwise Trie）——XOR 最大值

**问题**：给定整数数组 $A$，找两个数 $A[i]$ 和 $A[j]$（$i \ne j$），使得 $A[i] \oplus A[j]$（XOR）最大。

**暴力**：枚举所有对 → $O(n^2)$。

**二进制 Trie 思路**：

将每个整数的**二进制表示**从高位（第 30 位或第 31 位）到低位（第 0 位）依次插入 Trie，每个节点只有 `0` 和 `1` 两个子节点。

对于每个数 $x$，在 Trie 中查询与之 XOR 最大的数的方向：

- 若当前位为 $b$，XOR 后希望该位为 $1$（最大化），则贪心选择 $1-b$ 方向的子节点
- 若 $1-b$ 方向不存在，只能选 $b$ 方向（该位 XOR 结果为 0）

**时间复杂度**：插入 $O(n \cdot L)$，查询 $O(n \cdot L)$，总 $O(n \cdot L)$，其中 $L = 30$（整数的位数）。

```python
class BitwiseTrie:
    """
    二进制 Trie，按整数二进制位（从最高位到最低位）组织。
    
    用于 XOR 最大值查询：O(L) 每次查询，L = 位数（通常 30 或 31）
    """
    def __init__(self, max_bits: int = 30):
        # children[node][bit]: 从 node 出发，经过 bit（0 或 1）到达的子节点
        # 用列表模拟，0 表示空节点
        self.children = [[0, 0]]   # 根节点（索引 0）
        self.max_bits = max_bits
    
    def insert(self, num: int) -> None:
        """将整数 num 的二进制表示插入 Trie（从第 max_bits-1 位到第 0 位）"""
        node = 0
        for bit in range(self.max_bits - 1, -1, -1):
            b = (num >> bit) & 1    # 取第 bit 位的值（0 或 1）
            if not self.children[node][b]:
                # 创建新节点
                self.children.append([0, 0])
                self.children[node][b] = len(self.children) - 1
            node = self.children[node][b]
    
    def max_xor(self, num: int) -> int:
        """
        查询 Trie 中与 num 异或值最大的数，返回最大 XOR 值。
        
        贪心策略：逐位选择使 XOR 结果当前位为 1 的方向
        """
        node = 0
        result = 0
        for bit in range(self.max_bits - 1, -1, -1):
            b = (num >> bit) & 1           # num 第 bit 位
            target = 1 - b                  # 希望遇到 1-b（使 XOR 该位为 1）
            if self.children[node][target]:
                result |= (1 << bit)        # 这一位 XOR 结果为 1
                node = self.children[node][target]
            else:
                node = self.children[node][b]   # 只能走 b 方向，该位 XOR 为 0
        return result


def findMaximumXOR(nums: list[int]) -> int:
    """
    LeetCode #421：数组中两个数的最大异或值
    时间: O(n log V)，V 为最大值；空间: O(n log V)
    """
    trie = BitwiseTrie()
    max_val = 0
    for num in nums:
        trie.insert(num)
        max_val = max(max_val, trie.max_xor(num))
    return max_val

# 测试
print(findMaximumXOR([3, 10, 5, 25, 2, 8]))   # 输出: 28（5 XOR 25 = 28）
print(findMaximumXOR([14, 70, 53, 83, 49, 91, 36, 80, 92, 51, 66, 70]))  # 127
```

```cpp
struct BitwiseTrie {
    int ch[32 * 100005][2];   // 每个数最多 32 位，最多 100005 个数
    int tot = 1;
    
    void insert(int num) {
        int u = 1;
        for (int bit = 29; bit >= 0; bit--) {
            int b = (num >> bit) & 1;
            if (!ch[u][b]) ch[u][b] = ++tot;
            u = ch[u][b];
        }
    }
    
    int maxXor(int num) {
        int u = 1, result = 0;
        for (int bit = 29; bit >= 0; bit--) {
            int b = (num >> bit) & 1;
            int target = 1 - b;
            if (ch[u][target]) {
                result |= (1 << bit);
                u = ch[u][target];
            } else {
                u = ch[u][b];
            }
        }
        return result;
    }
};

int findMaximumXOR(vector<int>& nums) {
    BitwiseTrie trie;
    int ans = 0;
    for (int num : nums) {
        trie.insert(num);
        ans = max(ans, trie.maxXor(num));
    }
    return ans;
}
```

**图示（数值 5 = `...000101` 和 25 = `...011001`，XOR = 28 = `...011100`）**：

```
在 Trie 中查询与 5（二进制 00101）异或最大的数：
位 4：5 的该位 = 0，目标 = 1 → Trie 有 1 方向（25 的该位 = 1）→ XOR 该位 = 1 ✓
位 3：5 的该位 = 0，目标 = 1 → Trie 有 1 方向 → XOR 该位 = 1 ✓
位 2：5 的该位 = 1，目标 = 0 → Trie 有 0 方向 → XOR 该位 = 1 ✓
位 1：五的该位 = 0，目标 = 1 → Trie 无 1 方向 → 走 0 → XOR 该位 = 0
位 0：5 的该位 = 1，目标 = 0 → Trie 有 0 方向 → XOR 该位 = 1 ✓

结果 XOR = 11100（二进制）= 28 ✓
```

<div data-component="BitwiseTrieXOR"></div>

### 32.3.2 Trie 实现 LCP 查询

**LCP（Longest Common Prefix，最长公共前缀）**：两个字符串 $s$ 和 $t$ 的 LCP 是它们共同的最长前缀长度。

**Trie 方法**：将所有字符串插入 Trie，两个字符串的 LCP 长度 = 它们在 Trie 中**从根出发分叉前走过的边数**（即公共路径长度）。

```python
def lcp_trie(s: str, t: str, trie: Trie) -> int:
    """
    利用已建好的 Trie 计算 s 和 t 的 LCP 长度。
    
    原理：从根出发，同时在 s 和 t 的路径上行走，
    只要两者走同一方向，LCP +1；第一次分歧时停止。
    
    时间: O(min(|s|, |t|))
    """
    node = trie.root
    length = 0
    for cs, ct in zip(s, t):
        if cs != ct:
            break    # 两者选择了不同方向，LCP 到此结束
        idx = ord(cs) - ord('a')
        if node.children[idx] is None:
            break    # 路径不存在
        node = node.children[idx]
        length += 1
    return length
```

**更实用的场景**：LeetCode #14"最长公共前缀"——求数组中所有字符串的公共最长前缀。

```python
def longestCommonPrefix(strs: list[str]) -> str:
    """
    方法一：逐字符扫描（最直接，无需 Trie）
    时间: O(Σ|strs[i]|)

    方法二（Trie）：插入所有字符串，从根出发走，
    直到遇到 is_end 或分叉点为止
    """
    if not strs:
        return ""
    
    # 方法一：以第一个串为基准逐字符比对
    prefix = []
    for i, ch in enumerate(strs[0]):
        for s in strs[1:]:
            if i >= len(s) or s[i] != ch:
                return ''.join(prefix)
        prefix.append(ch)
    return ''.join(prefix)
```

### 32.3.3 持久化 Trie（Persistent Trie）——历史版本查询

**场景**：有一个数组 $A$，支持两种操作：
1. `ADD(x)`：将 $x$ 加入集合  
2. `QUERY(l, r, k)`：查询历史版本 $l$ 到 $r$ 中，与 $k$ 异或最大的数

**持久化 Trie**（也称**可持久化 Trie**）：每次插入新值时，**不修改旧节点**，而是**复用旧路径上的公共节点**，新建沿途的节点，保留历史版本的根节点。

这个思想与**持久化线段树（主席树）** 完全相同。

```python
class PersistentTrieNode:
    def __init__(self):
        self.children = [None, None]   # 只有 0 和 1 两个子节点（二进制 Trie）
        self.count = 0                  # 该节点对应的值域中有多少数

class PersistentBitwiseTrie:
    """
    持久化二进制 Trie：每次 insert 不修改旧版本，返回新根节点。
    
    版本号即 versions 数组的索引：
    versions[0] = 初始（空）根
    versions[i] = 插入第 i 个元素后的根
    
    支持操作：
    - insert(root, x): 返回插入 x 后的新根
    - query(root_lo, root_hi, x): 查询版本区间 [lo+1, hi] 中与 x 异或最大的值
    """
    def __init__(self, max_bits: int = 30):
        self.max_bits = max_bits
    
    def insert(self, old_root: PersistentTrieNode, num: int) -> PersistentTrieNode:
        """
        插入数 num，返回新版本的根节点。
        
        路径复制（Path Copying）：只复制从根到新值所在叶的路径上的节点
        （log V 个节点），其余节点与旧版本共享。
        """
        new_root = PersistentTrieNode()
        cur_new = new_root
        cur_old = old_root
        
        for bit in range(self.max_bits - 1, -1, -1):
            b = (num >> bit) & 1
            # 复制另一侧的子节点（直接引用旧节点，共享）
            cur_new.children[1 - b] = cur_old.children[1 - b] if cur_old else None
            # 新建当前位方向的节点
            cur_new.children[b] = PersistentTrieNode()
            cur_new.children[b].count = (cur_old.children[b].count if cur_old and cur_old.children[b] else 0) + 1
            cur_new = cur_new.children[b]
            cur_old = cur_old.children[b] if cur_old else None
        
        return new_root
```

持久化 Trie 是竞赛中的进阶技巧，LeetCode 暂无直接对应题目，但在 Codeforces、洛谷等 OJ 上有经典应用，例如**区间 XOR 最大值**问题。

<div data-component="PatriciaTreeCompression"></div>

---

## 32.4 经典 LeetCode 题解析

### 32.4.1 #208. 实现 Trie（前缀树）——标准模板

> 实现 Trie 类，包含 `insert`、`search`、`startsWith` 三个操作。

这是 Trie 最核心的模板题，直接使用 §32.1.2~32.1.4 的实现即可。

**完整 Python 实现**：

```python
class Trie:
    def __init__(self):
        # 根节点初始化，28 字节的节点（26 个指针 + is_end）
        self.root = self._new_node()
    
    def _new_node(self) -> list:
        # 用 list 表示节点：[children_0, ..., children_25, is_end]
        # 长度 27，前 26 位为子节点引用，最后一位为 is_end 标志
        return [None] * 26 + [False]
    
    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            idx = ord(ch) - ord('a')
            if node[idx] is None:
                node[idx] = self._new_node()
            node = node[idx]
        node[26] = True     # is_end = True
    
    def search(self, word: str) -> bool:
        node = self._find(word)
        return node is not None and node[26]
    
    def startsWith(self, prefix: str) -> bool:
        return self._find(prefix) is not None
    
    def _find(self, s: str):
        node = self.root
        for ch in s:
            idx = ord(ch) - ord('a')
            if node[idx] is None:
                return None
            node = node[idx]
        return node
```

```cpp
class Trie {
    int ch[300005][26], tot = 1;
    bool is_end[300005];
public:
    Trie() { memset(ch, 0, sizeof(ch)); memset(is_end, 0, sizeof(is_end)); }
    
    void insert(string word) {
        int u = 1;
        for (char c : word) {
            int x = c - 'a';
            if (!ch[u][x]) ch[u][x] = ++tot;
            u = ch[u][x];
        }
        is_end[u] = true;
    }
    
    bool search(string word) {
        int u = 1;
        for (char c : word) {
            int x = c - 'a';
            if (!ch[u][x]) return false;
            u = ch[u][x];
        }
        return is_end[u];
    }
    
    bool startsWith(string prefix) {
        int u = 1;
        for (char c : prefix) {
            int x = c - 'a';
            if (!ch[u][x]) return false;
            u = ch[u][x];
        }
        return true;
    }
};
```

**时间复杂度**：三个操作均为 $O(m)$（$m$ = 操作字符串长度）。

**空间复杂度**：$O(\text{总字符数} \cdot 26)$。

### 32.4.2 #212. 单词搜索 II——Trie 剪枝 DFS

> 给定二维字符棋盘和单词列表，找出所有在棋盘中存在的单词（棋盘中字母相邻连接，不重复使用同一格子）。

**暴力做法**：对每个单词单独做 DFS → $O(k \cdot m \cdot n \cdot 4^L)$，$L$ 为单词长度，$k$ 为单词数，太慢。

**Trie 优化**：将所有单词构建成 Trie，DFS 时沿 Trie 走（而非对每个单词单独走）。当 DFS 到达 Trie 中的某个 `is_end` 节点时，说明找到了一个单词。当 DFS 走到 Trie 中不存在的路径时，直接剪枝，无需继续。

**关键剪枝**：一旦某个单词被找到，可以将 Trie 中该单词的结尾节点的 `is_end` 清除（避免重复加入结果），并向上清除空节点，防止无效搜索。

```python
def findWords(board: list[list[str]], words: list[str]) -> list[str]:
    """
    LeetCode #212: 单词搜索 II
    
    思路：Trie + DFS + 在 Trie 中删除已找到的单词（关键剪枝）
    时间: O(M * N * 4^L + Σ|words|)，L 为最长单词长度
    空间: O(Σ|words| * 26)（Trie）+ O(L)（DFS 栈）
    """
    ROWS, COLS = len(board), len(board[0])
    
    # 构建 Trie，每个叶节点存单词（而非单词 ID）
    root = {}
    for word in words:
        node = root
        for ch in word:
            node = node.setdefault(ch, {})
        node['#'] = word    # '#' 作为结束标记，值为单词本身
    
    results = []
    
    def dfs(r: int, c: int, node: dict) -> None:
        ch = board[r][c]
        if ch not in node:
            return    # 当前字符在 Trie 中无对应路径，剪枝
        
        next_node = node[ch]
        
        if '#' in next_node:
            results.append(next_node['#'])    # 找到单词！
            del next_node['#']               # 删除已找到的单词（避免重复）
        
        board[r][c] = '#'    # 标记当前格子已访问（防止重复使用）
        
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and board[nr][nc] != '#':
                dfs(nr, nc, next_node)
        
        board[r][c] = ch     # 回溯：恢复格子字符
        
        # 关键剪枝：若 next_node 变为空（所有子路径都找完了），删除父节点中的引用
        if not next_node:
            del node[ch]
    
    for r in range(ROWS):
        for c in range(COLS):
            dfs(r, c, root)
    
    return results
```

```cpp
struct TrieNode {
    TrieNode* children[26] = {};
    string word;   // 非空表示此处是某单词的结尾
};

class Solution {
    TrieNode* root;
    vector<string> results;
    
    void dfs(vector<vector<char>>& board, int r, int c, TrieNode* node) {
        if (r < 0 || r >= (int)board.size() || c < 0 || c >= (int)board[0].size()) return;
        char ch = board[r][c];
        if (ch == '#' || !node->children[ch - 'a']) return;
        
        node = node->children[ch - 'a'];
        if (!node->word.empty()) {
            results.push_back(node->word);
            node->word.clear();    // 防止重复
        }
        
        board[r][c] = '#';
        dfs(board, r+1, c, node);
        dfs(board, r-1, c, node);
        dfs(board, r, c+1, node);
        dfs(board, r, c-1, node);
        board[r][c] = ch;
    }
    
public:
    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        root = new TrieNode();
        for (auto& w : words) {
            TrieNode* cur = root;
            for (char c : w) {
                if (!cur->children[c-'a']) cur->children[c-'a'] = new TrieNode();
                cur = cur->children[c-'a'];
            }
            cur->word = w;
        }
        for (int r = 0; r < (int)board.size(); r++)
            for (int c = 0; c < (int)board[0].size(); c++)
                dfs(board, r, c, root);
        return results;
    }
};
```

**为什么要删除已找到的单词**？因为同一个单词可能从多个起始位置匹配到，但题目要求每个单词只报告一次。删除 `is_end` 标记后，Trie 搜索不会再次触发该单词的报告。

**为什么删除空节点**？当某条路径上所有单词都找完后，该路径的 Trie 节点变为空。删除空节点后，DFS 到达此处时会直接剪枝，节省大量无效搜索时间。这是 #212 优化的关键。

### 32.4.3 #421. 数组中两个数的最大异或值

> 给你一个整数数组 nums，返回 `nums[i] XOR nums[j]` 的最大结果。

直接使用 §32.3.1 的 `findMaximumXOR` 实现，时间复杂度 $O(n \cdot L)$，$L = 30$。

**为什么 XOR 要用 Trie 而不是数学方法？**

XOR 的"最大化"在十进制世界没有规律，但在二进制世界有清晰的贪心策略——从最高位贪心选择，每位独立决策，这正是 Trie 逐位走的天然契机。

**另一种思路**：前缀异或 + 位操作（不需要 Trie）：

```python
def findMaximumXOR_bit(nums: list[int]) -> int:
    """
    逐位确定答案的最高到最低位（掩码逼近法）。
    
    核心思路：假设已确定答案的高 bit 位，尝试通过前缀集合
    判断答案第 (bit-1) 位是否能为 1。
    
    时间: O(n log V)；空间: O(n)
    """
    ans = 0
    mask = 0
    for bit in range(29, -1, -1):
        mask |= (1 << bit)
        prefixes = set(num & mask for num in nums)    # 当前高位前缀集合
        
        # 尝试让第 bit 位为 1（即答案 |= (1 << bit)）
        candidate = ans | (1 << bit)
        
        # 若存在两个前缀 a, b 使得 a XOR b = candidate
        # 等价于 b = a XOR candidate 在 prefixes 中
        if any((candidate ^ p) in prefixes for p in prefixes):
            ans = candidate    # 第 bit 位可以是 1！
    
    return ans
```

<div data-component="BitwiseTrieXOR"></div>

### 32.4.4 #745. 前缀和后缀搜索（双 Trie 组合）

> 给定一组单词，设计一个数据结构，对于 `f(prefix, suffix)` 查询，返回具有给定前缀和后缀且下标最大的单词的下标，若不存在返回 -1。

**思路**：同时需要**前缀**和**后缀**匹配，两个 Trie 各自负责一半，然后取交集。

**更优技巧**：构造键 `suffix + '#' + prefix`，插入一个 Trie 中，同时支持前缀（即对 `suffix_prefix` 的前缀搜索）和后缀查询。

```python
class WordFilter:
    """
    LeetCode #745: 前缀和后缀搜索
    
    技巧：将 "suffix#prefix" 形式的所有组合插入 Trie，
    查询时搜 "suffix#prefix" 的前缀，返回路径上最大权重（单词索引）。
    
    时间：预处理 O(Σ |word|²)；查询 O(|prefix| + |suffix|)
    空间：O(Σ |word|²)
    
    注意：若词汇量很大（单词很长），空间消耗可能巨大，竞赛中需权衡。
    """
    def __init__(self, words: list[str]):
        self.trie = {}
        for weight, word in enumerate(words):
            # 将所有 "suffix#prefix" 组合插入 Trie
            n = len(word)
            for k in range(n + 1):
                suffix = word[n - k:]    # word 的后 k 个字符（k=0 时为空串）
                key = suffix + '#' + word
                node = self.trie
                node['weight'] = weight    # 根节点也更新权重（会被后续覆盖）
                for ch in key:
                    node = node.setdefault(ch, {})
                    node['weight'] = max(node.get('weight', -1), weight)
    
    def f(self, pref: str, suff: str) -> int:
        node = self.trie
        for ch in suff + '#' + pref:
            if ch not in node:
                return -1
            node = node[ch]
        return node.get('weight', -1)
```

### 32.4.5 #820. 单词的压缩编码

> 给定单词列表，对列表编码（每个单词可以是另一个单词的后缀），找最短编码串的长度。

**关键观察**：如果单词 $A$ 是单词 $B$ 的后缀，则 $A$ 不需要单独编码（$B$ 的编码已经包含了 $A$）。所以问题等价于：找字符串集合中，哪些词**不是任何其他词的后缀**，这些词的总长度 + 它们的个数（每个词后面需要一个 `#`）就是答案。

**Trie 方法**：将所有单词**反转**后插入 Trie，不是任何其他反转串前缀的终端节点（叶节点）对应的单词就是需要单独编码的词。

```python
def minimumLengthEncoding(words: list[str]) -> int:
    """
    LeetCode #820：单词的压缩编码
    
    思路：将反转后的单词插入 Trie，统计所有叶节点（无孩子的 is_end 节点）
    对应的单词长度之和，再加上叶节点个数（每个末尾的 '#'）。
    
    时间: O(Σ|words[i]|)；空间: O(Σ|words[i]| × 26)
    """
    root = {}
    nodes = {}   # 记录每个终端节点对应的单词长度
    
    for word in set(words):    # 去重
        node = root
        for ch in reversed(word):    # 反转后插入
            node = node.setdefault(ch, {})
        nodes[id(node)] = len(word)    # 记录叶节点（或中间节点）对应的原单词
    
    # 统计叶节点（无子节点）对应的单词长度 + 1（'#' 分隔符）
    return sum(length + 1 for nid, length in nodes.items()
               if id(root) != nid and not any(root is not None for _ in []))
```

更简洁的实现：

```python
def minimumLengthEncoding(words: list[str]) -> int:
    good = set(words)
    for word in words:
        for k in range(1, len(word)):
            good.discard(word[k:])    # 去掉 word 的所有后缀（它们不需要单独编码）
    return sum(len(word) + 1 for word in good)
```

---

## 32.5 复杂度总结与选型指南

### 32.5.1 各操作复杂度汇总

| 数据结构 | 构建时间 | 查询/搜索 | 空间 | 特点 |
|---|---|---|---|---|
| **Trie（数组）** | $O(\Sigma \|P\|)$ | $O(m)$ | $O(N \cdot \Sigma)$ | 查找快，字母表小时优 |
| **Trie（HashMap）** | $O(\Sigma \|P\|)$ | $O(m)$ 均摊 | $O(N)$ | 通用字母表，节省空间 |
| **压缩 Trie** | $O(\Sigma \|P\|)$ | $O(m)$ | $O(N)$（节点数） | 空间更优 |
| **AC 自动机** | $O(\Sigma \|P\| \cdot \Sigma)$ | $O(n + 匹配数)$ | $O(\Sigma \|P\| \cdot \Sigma)$ | 多模式匹配神器 |
| **二进制 Trie** | $O(n \cdot L)$ | $O(L)$ | $O(n \cdot L)$ | XOR 问题，$L = 30$ |

### 32.5.2 场景选型决策

```
需要字符串匹配？
├─ 单模式匹配 → KMP（Chapter 31）
├─ 多模式匹配 → AC 自动机
├─ 前缀查询 / 自动补全 → Trie
└─ 位运算（XOR、AND）最优化 → 二进制 Trie

实际字母表大小？
├─ 小（26个字母）→ children[26] 数组
└─ 大（Unicode、任意字符）→ HashMap 子节点

空间受限？
├─ 词有大量公共前缀 → 压缩 Trie（Radix Tree）
└─ 需要历史版本 → 持久化 Trie
```

<div data-component="TrieComplexityComparison"></div>

---

## 32.6 常见错误与陷阱

**1. `is_end` 与节点存在混淆**

```python
# ❌ 错误：以为到达节点就是找到单词
def search_wrong(self, word: str) -> bool:
    node = self._traverse(word)
    return node is not None    # 漏了 is_end 检查！

# ✓ 正确：必须同时检查路径完整且 is_end = True
def search_correct(self, word: str) -> bool:
    node = self._traverse(word)
    return node is not None and node.is_end
```

**2. AC 自动机漏掉输出链接**

```python
# ❌ 错误：只报告当前节点的模式，漏了 fail 链上的模式
if node.is_end:
    results.append(node.pattern)    # 只检查自身！

# ✓ 正确：需要沿 output link 报告所有匹配的模式
for pid in self.output[node]:    # output 已在构建时合并 fail 节点输出
    results.append(patterns[pid])
```

**3. Rabin-Karp 与 Trie 的混用错误**：Trie 只支持**前缀**匹配，不支持任意位置子串匹配。若要在 $O(1)$ 内查找任意子串，需要后缀数组或哈希（Chapter 33）。

**4. 二进制 Trie 处理负数**：

```python
# ❌ 负数直接插入：补码最高位为符号位，比较逻辑会混乱
# ✓ 统一用无符号位（对 Python 直接操作，对 C++ 用 unsigned int）

# Python 中处理负数（取 32 位二进制的低 30 位）
num &= (1 << 30) - 1    # 保留低 30 位，忽略符号
```

**5. 竞赛 AC 自动机的数组越界**：

```cpp
// ❌ 数组大小不够：模式串字符总数 × 2（clone 节点）
int ch[MAXN][26];    // MAXN 必须足够大

// ✓ 预先计算总字符数，留出足够余量
const int MAXN = 1e6 + 5;    // 建议至少是总字符数的 2 倍
```

---

## 32.7 思考题与扩展

**💡 思考题 1**：AC 自动机的失败链接构建为什么必须用 BFS（层序遍历），而不能用 DFS？

> 提示：计算 `fail[u]` 时需要用到 `fail[parent(u)]`，而 `fail[parent(u)]` 必须已经计算完毕。BFS 保证了父节点在子节点之前处理；DFS 则无法保证这个顺序。

**💡 思考题 2**：如果只有一个模式串，AC 自动机退化为什么结构？它的 fail 链接与 KMP 的 π 数组有何对应关系？

> 提示：退化为一条链式 Trie。fail[i] = π[i-1]（0-indexed）。AC 自动机的 fail 链接本质上就是 KMP π 数组在树状结构上的推广。

**💡 思考题 3**：Trie 和哈希表都能做 $O(1)$ 的字符串查找（哈希表对整个字符串哈希），什么情况下 Trie 优于哈希表？

> 提示：①前缀查询（哈希表无法高效支持）；②内存局部性（Trie 节点有空间局部性，缓存友好）；③最坏情况稳定性（哈希碰撞退化）；④无需哈希函数设计。

**💡 思考题 4**：在 #212 单词搜索 II 中，为什么使用 Trie 而不是依次对每个单词做 DFS？两者的时间复杂度差异在哪里？

> 分析：朴素做法：$k$ 个单词 × 棋盘格子数 × DFS 深度 = $O(k \cdot m \cdot n \cdot 4^L)$；Trie 方法：$O(m \cdot n \cdot 4^L + \sum |word_i|)$，DFS 只做一次，Trie 公共前缀剪枝大量减少了搜索空间。

---

## 本章总结

**Trie 的核心价值**：将字符串集合的**公共前缀结构**显式化，支持 $O(m)$ 的插入/查找/前缀查询，是自动补全、单词搜索等场景的基础数据结构。

**AC 自动机的核心价值**：将 KMP 的"失败链接"机制推广到 Trie 上，实现 $O(n + 匹配数)$ 的多模式匹配，是构建高性能文本过滤、网络检测系统的标准工具。

**二进制 Trie 的价值**：为位运算相关的最优化问题（尤其是 XOR 最优）提供 $O(L)$ 查询的贪心框架，$L = 30$ 或 32。

**记忆口诀**：
- Trie = 前缀树，公共前缀共路径
- AC 自动机 = Trie + KMP（fail 链），多模式一次扫
- 二进制 Trie = 位从高到低，贪心选对应位

**⚠️ 实战备忘**：
- 字母表大 → HashMap 节点；空间紧 → 压缩 Trie；历史版本 → 持久化 Trie
- AC 自动机的 output 链接必须在构建时合并；搜索时不再跳 fail 链
- 二进制 Trie 的位数 = $\lceil \log_2(\max(A)) \rceil$，默认取 30

**参考资料**：
- CLRS 第4版 Chapter 32（字符串匹配）
- Sedgewick《算法》Chapter 5.2（Tries）
- Aho, Corasick 1975 经典论文：*Efficient String Matching: An Aid to Bibliographic Search*
- 竞赛资料：OI-Wiki AC 自动机章节（https://oi-wiki.org/string/ac-automaton/）
- LeetCode 题目：#208、#211、#212、#421、#745、#820、#1032、#1065

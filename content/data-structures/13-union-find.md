# Chapter 13: 并查集（Union-Find / Disjoint Set Union）

> **学习目标**：  
> 理解并查集（Disjoint Set Union，DSU）如何高效地解决"动态连通性"问题；掌握路径压缩与按秩合并两大核心优化的实现原理及其协同效果；理解 Ackermann 反函数 $\alpha(n)$ 的直觉含义；能将并查集应用于 Kruskal 最小生成树、图连通分量、网格岛屿等经典问题中。

---

## 13.1 并查集的动机与定义

### 13.1.1 动态连通性问题——生活中的"朋友圈"

在深入代码之前，先建立直觉。

想象你负责管理一个**网络互联问题**：你有 $n$ 台服务器（编号 $0$ 到 $n-1$），工程师会陆续报告"服务器 $A$ 和服务器 $B$ 已经通过专线连通了"。与此同时，用户会不断询问"服务器 $X$ 和服务器 $Y$ 能互相通信吗？"

等价地，这是一个**社交网络**问题：有 $n$ 个人，不断有"$A$ 和 $B$ 成为朋友"的事件发生，用户随时问"$X$ 和 $Y$ 是否在同一个朋友圈里？"

这就是**动态连通性问题**（Dynamic Connectivity Problem）的核心：

给定 $n$ 个**元素**（节点、对象），支持以下两类操作，且操作**在线交替出现**：

| 操作 | 含义 |
|------|------|
| $\text{UNION}(x, y)$ | 将 $x$ 所在集合与 $y$ 所在集合**合并**为同一个集合 |
| $\text{FIND}(x)$ | 查询 $x$ 属于哪个集合（返回该集合的**代表元素**）|

通过判断 $\text{FIND}(x) = \text{FIND}(y)$ 是否成立，即可回答"$x$ 和 $y$ 是否连通"。

**为什么说"动态"？** 因为元素之间的关系随着 UNION 操作不断演化。若所有 UNION 提前已知，可以用 BFS/DFS 离线处理；但若操作是流式、在线到来的，就需要能**实时响应**的特殊数据结构——并查集正是为此而生。

**现实中的并查集应用场景**：

- 🌐 **网络连通**：Ethernet 或 TCP/IP 网络，动态加入链路后判断连通性
- 🗺️ **地图岛屿**：离线游戏世界中每放置一块地，判断大陆是否合并
- 📊 **Kruskal MST**：构建最小生成树时判断加边是否构成回路
- 🧬 **基因聚类**：将相似基因序列逐步合并为同一类别
- 🏦 **账户合并**：相同邮箱/手机号的账户视为同一用户

**算法竞赛中的"银弹"**：并查集代码极短（~10 行），在竞赛中能解决大量图连通性问题，几乎是必须烂熟于心的模板之一。

### 13.1.2 三个基本操作的语义

并查集维护一个"**不相交集合族**"（Disjoint Set Family / Disjoint Partition）：$n$ 个元素被划分成若干**两两不相交**的集合。每个集合都有一个**代表元素**（Representative）用于标识该集合。

三个核心操作：

**① MAKE-SET(x)**：将元素 $x$ 单独创建为一个只含自己的集合。初始化时对所有元素调用一次。

$$\{x\}$$

**② FIND(x)**：返回元素 $x$ 所在集合的代表元素。若 $\text{FIND}(x) = \text{FIND}(y)$，则 $x$ 与 $y$ **在同一集合**（连通）。

$$\text{FIND}(x) = \text{representative of } x\text{'s set}$$

**③ UNION(x, y)**（有时写作 MERGE/LINK）：将 $x$ 所在集合和 $y$ 所在集合合并，合并后两者共享一个代表元素。

$$\{S_x \cup S_y\}$$

> ⚠️ **重要约定**：UNION 的参数是**元素**，不是集合。且 UNION 之后原来两个集合消失，出现一个新的合并集合。每次成功的 UNION 操作都会让集合数减少 1（从 $n$ 个到最终可能的 1 个）。

### 13.1.3 "代表元素"的语义：任意性与一致性

FIND 返回的代表元素（Representative / Root）是由数据结构**内部自行选择**的，调用方只需保证：

- **一致性**：只要两次 FIND 之间没有发生 UNION，对同一集合内任意元素的 FIND 结果都相同
- **任意性**：代表元素不必是最小值、最大值或任何"特殊"元素，内部实现可自由选择

这意味着代码中**不能依赖代表元素的具体值**，只能用它做**相等性比较**：

```python
# ✅ 正确用法
if find(x) == find(y):
    print("x 和 y 在同一集合")

# ❌ 错误用法
r = find(x)
print(f"代表元素编号是 {r}，一定是最小的那个")  # 并查集不保证这一点
```

**集合数量的计数技巧**：初始时有 $n$ 个集合，每次成功的 UNION（两个元素之前不同集合）使集合数 $-1$。维护一个 `count` 变量即可：

```python
if find(x) != find(y):
    union(x, y)
    count -= 1
```

---

## 13.2 链表表示——朴素实现与加权合并

### 13.2.1 链表实现的基本结构

并查集的第一种实现方法是用**链表**表示每个集合：

- 每个集合对应一条**单向链表**，链表头（Head）节点作为代表元素
- 每个节点维护：`value`（元素值）、`next`（链表下一个节点）、`head`（指向链表头节点，即代表元素）

```
集合 {a, b, c, d}（代表元素为 a）：

┌───┬────┐   ┌───┬────┐   ┌───┬────┐   ┌───┬──────┐
│ a │ ──→│ ──→ b │ ──→│ ──→ c │ ──→│ ──→ d │ NULL │
└───┴────┘   └───┴────┘   └───┴────┘   └───┴──────┘
  ↑             ↑             ↑             ↑
 head          head          head          head
（所有节点的 head 指针都指向 a）
```

这样设计的操作复杂度：

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| MAKE-SET(x) | $O(1)$ | 创建单节点链表 |
| FIND(x) | $O(1)$ | 直接返回 `x.head`（代表元素） |
| UNION(x, y) | $O(\|S_y\|)$ | 将 $S_y$ 链表接到 $S_x$ 尾部，并更新 $S_y$ 中所有节点的 head 指针 |

**UNION 的代价瓶颈**：合并时需要遍历被"接入"的那个链表，把每个节点的 `head` 指针改为新的代表元素。最坏情况下，若每次都将大链表接到小链表后面，每次 UNION 代价 $O(n)$，$n$ 次 UNION 总代价 $O(n^2)$——非常差。

**具体的最坏情形**：对 $n$ 个元素依次执行 UNION(1,2)、UNION(2,3)、……、UNION(n-1,n)，如果每次都把长链表接到短链表后（不够智能），第 $k$ 次 UNION 代价 $\Theta(k)$，总代价 $\sum_{k=1}^{n-1} k = \Theta(n^2)$。

### 13.2.2 加权合并启发（Weighted-Union Heuristic）

一个简单的改进：**每次将规模较小的集合合并到规模较大的集合中**（"以小并大"），这样大的链表 head 指针不需要更新，只需更新小链表的 head 指针。

**实现方法**：每个链表维护一个 `size` 字段（集合大小）。UNION 时：

1. 若 $|S_x| \geq |S_y|$：将 $S_y$ 接到 $S_x$ 尾部，更新 $S_y$ 内所有节点的 head → 代价 $O(|S_y|)$
2. 若 $|S_x| < |S_y|$：将 $S_x$ 接到 $S_y$ 尾部，更新 $S_x$ 内所有节点的 head → 代价 $O(|S_x|)$

**关键引理**：采用加权合并后，每个元素的 head 指针被更新的次数**至多 $\lfloor \log_2 n \rfloor$ 次**。

**证明思路**：每次某元素的 head 被更新，说明它原来所在集合是"较小的那个"（大小为 $s$），合并后新集合大小至少是 $2s$（因为合并的两个集合中，它所在的是较小的，所以另一个集合大小 $\geq s$，合并后总大小 $\geq 2s$）。每次被更新，所在集合大小至少翻倍，从 1 开始最多翻倍 $\log_2 n$ 次就超过 $n$。因此每个元素至多更新 $\log_2 n$ 次。

### 13.2.3 加权合并的摊销复杂度

**定理**：$n$ 个元素，采用加权合并启发，执行 $m$ 次 MAKE-SET、UNION、FIND 操作，总时间代价（所有 head 指针更新次数）：

$$O(m + n \log n)$$

- $m$ 次 FIND 操作各 $O(1)$，共 $O(m)$
- 所有 UNION 操作中指针更新总次数：每个元素最多更新 $\log n$ 次，$n$ 个元素共 $O(n \log n)$ 次

这已经是相当好的结果，但链表实现有一个根本缺陷：**FIND 是 $O(1)$，UNION 代价在被合并那一侧是 $O(|S|)$**。

能否找到一种实现，让 FIND 和 UNION 都接近 $O(1)$？这引出了**树形表示（森林）**。

---

## 13.3 树形表示——森林结构的巧妙设计

### 13.3.1 用有根树表示每个集合

换一种数据结构：用**有向树**来表示集合：

- 每棵树对应一个集合，**根节点**是该集合的代表元素
- 每个非根节点有一个 `parent` 指针指向父节点
- 根节点的 `parent` 指向自身（`parent[root] = root`）

所有集合的树合在一起，形成一片**森林（Forest）**。

对于 $n$ 个元素，only 需要一个数组 `parent[0..n-1]`，`parent[i]` 表示元素 $i$ 的父节点编号：

```
初始（n=6，每个元素自成一个集合）：
parent = [0, 1, 2, 3, 4, 5]
（所有元素都是自己的父节点，即各自是树根）

执行 UNION(0, 1)、UNION(1, 2)、UNION(3, 4) 后的一种可能结构：

    0       3
   /       / \
  1       4
  |
  2
```

**三个操作的实现**：

- **MAKE-SET(x)**：`parent[x] = x`，$O(1)$
- **FIND(x)**：沿 `parent` 指针向上爬，直到找到根（`parent[root] == root`），$O(\text{树高})$
- **UNION(x, y)**：调用 `FIND(x)` 和 `FIND(y)` 找到两根，将一个根的 parent 指向另一个根，$O(\text{树高})$

**空间**：仅需一个长度为 $n$ 的整数数组，极其省内存。

### 13.3.2 朴素实现的最坏情况

朴素树形并查集（不做任何优化）的最坏情形十分糟糕：

对 $n$ 个元素，若每次 UNION 都把新元素接到当前树的最深节点的下面，会退化为**一条链**：

```
0 → 1 → 2 → 3 → 4 → 5（根）
```

此时 FIND(0) 需要遍历 $n$ 步才能找到根，FIND 代价 $O(n)$，完全没有改善。

朴素实现下，$m$ 次操作（其中 $n$ 次 MAKE-SET）的总代价最坏 $O(n + m \cdot n) = O(mn)$。

**核心问题**：树退化得太高（最坏高度 $O(n)$）。两种正交的优化思路将分别从"**UNION时控制树高**"和"**FIND时主动压缩树高**"两个方向解决这个问题。

### 13.3.3 按秩合并（Union by Rank）——控制树高

**核心思想**：UNION 时，让**矮树的根**的 parent 指向**高树的根**，保持树尽量矮。

**Rank（秩）**是对树高的上界估计：
- 初始时每个节点的 rank 为 0
- 只有两棵 rank 相等的树合并时，新树根的 rank 才 $+1$
- 若 rank 不相等，矮树根挂在高树根下，rank 不变

```python
# 按秩合并的 UNION
def union_by_rank(x, y, parent, rank):
    rx, ry = find(x, parent), find(y, parent)
    if rx == ry:
        return  # 已在同一集合
    if rank[rx] < rank[ry]:
        rx, ry = ry, rx   # 保证 rx 是秩更大的根
    parent[ry] = rx        # 矮树根 ry 挂到高树根 rx 下
    if rank[rx] == rank[ry]:
        rank[rx] += 1      # 相等时新树更高一层
```

**为什么 rank 不是实际树高？** 引入路径压缩后，树的实际高度可能降低，但 rank 不随之更新（更新会增加算法复杂度）。因此 rank 是树高的**上界**，而非精确值。

**关键定理**：单独使用按秩合并（不含路径压缩），对 $n$ 个元素执行 $m$ 次操作，总代价 $O(m \log n)$，且任意树的高度不超过 $\lfloor \log_2 n \rfloor$。

**证明要点**：按秩合并保证：若某节点 rank 为 $k$，则以它为根的子树中**至少有 $2^k$ 个节点**（数学归纳可证）。因为整棵树节点总数最多 $n$，所以最大 rank（即树高上界）是 $\lfloor \log_2 n \rfloor$。

```
按秩合并示例（n=8）：

初始 8 个节点，rank 均为 0

UNION(0,1): rank 相等 → 0 作根，rank[0]=1
  0(1)
  |
  1(0)

UNION(2,3): 同理
  2(1)
  |
  3(0)

UNION(0,2): rank 相等(均为1) → 0 作根，rank[0]=2
  0(2)
 / \
1(0) 2(1)
     |
    3(0)

UNION(4,5), UNION(6,7), UNION(4,6):
  4(2)
 / \
5(0) 6(1)
     |
    7(0)

UNION(0,4): rank 相等(均为2) → 任一作根，rank→3
  0(3)
 / | \
1  2  4(2)
   |  / \
   3 5   6
         |
         7
树高 = 3 = log₂8 ✓
```

<div data-component="UnionByRankTree"></div>

### 13.3.4 路径压缩（Path Compression）——FIND 时压缩树高

**核心思想**：执行 FIND(x) 时，沿 parent 指针找到根节点，然后**将路径上所有节点的 parent 直接指向根**，使后续对这些节点的 FIND 都能 $O(1)$ 完成。

**路径压缩的直觉**：就像整理文件夹——你找到某个深埋在子目录里的文件，找到之后索性给它建一个桌面快捷方式，下次直接访问。

```
路径压缩前（FIND(c) 要走 c→b→a→root 四步）：

        root
        |
        a
        |
        b
        |
        c

FIND(c) 执行后（路径压缩）：找到 root 后，把 a、b、c 都直接挂到 root 下

        root
      / | | \
     a  b  c ...

此后 FIND(a)、FIND(b)、FIND(c) 均只需 1 步。
```

**实现**（递归版本）：

```python
def find(x: int, parent: list[int]) -> int:
    # 递归找根，回溯时把路径上所有节点直接挂到根
    # 边界：x 是根节点（parent[x] == x），直接返回 x
    if parent[x] != x:
        parent[x] = find(parent[x], parent)   # ← 这一行完成路径压缩
        # 关键：先递归把 parent[x] 指向根，再将 x 的 parent 设为那个根
    return parent[x]
```

**实现**（迭代版本，适合竞赛/递归栈有限场景）：

```python
def find_iterative(x: int, parent: list[int]) -> int:
    # 第一步：找根
    root = x
    while parent[root] != root:
        root = parent[root]
    # 第二步：路径压缩（把路径上所有节点直接挂到根）
    while parent[x] != root:
        nxt = parent[x]
        parent[x] = root   # 压缩
        x = nxt
    return root
```

**注意点**：
- 递归版本更简洁，但 $n$ 很大时可能栈溢出（Python 默认递归深度约 1000）
- 迭代版本更安全，两次遍历同一路径，无栈溢出风险
- 两种版本的渐进复杂度相同

<div data-component="UnionFindPathCompression"></div>

### 13.3.5 路径分裂与路径减半（变体）

路径压缩的两种"轻量级"变体，在工程中也很常用：

**路径分裂（Path Splitting）**：将路径上每个节点的 parent 改为**祖父节点**（跳过一层），半压缩：

```python
def find_path_splitting(x: int, parent: list[int]) -> int:
    while parent[x] != x:
        parent[x], x = parent[parent[x]], parent[x]
        # 同时：x 的 parent 改为祖父；x 移动到原来的父节点位置
    return x
```

**路径减半（Path Halving）**：将路径上**每隔一个节点**的 parent 改为祖父节点：

```python
def find_path_halving(x: int, parent: list[int]) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]  # x 的 parent 跳过一层（指向祖父）
        x = parent[x]
    return x
```

| 变体 | 效果 | 代码复杂度 | 理论复杂度 |
|------|------|-----------|-----------|
| 完全路径压缩 | 最彻底，所有节点直接挂根 | 较复杂（两次遍历或递归） | $O(m\alpha(n))$ |
| 路径分裂 | 节点 parent 改为祖父，迭代 | 简洁（一次迭代） | $O(m\alpha(n))$ |
| 路径减半 | 隔行跳跃，迭代 | 最简洁（一次迭代） | $O(m\alpha(n))$ |

三种变体的渐进复杂度**完全相同**，在实践中路径减半因代码简洁而最受欢迎。

---

## 13.4 综合优化与摊销分析

### 13.4.1 组合两种优化的效果

将**按秩合并** + **路径压缩**同时使用，是并查集实践中的标准模板。二者互补：

- 按秩合并控制树的"初始高度"不过高（$O(\log n)$ 高度上界）
- 路径压缩在 FIND 时"主动压平"树，使再次访问极快

**完整标准模板代码**：

```python
class UnionFind:
    def __init__(self, n: int):
        """
        初始化：n 个元素，各自成一个集合
        parent[i] = i 表示 i 是自己的根
        rank[i] = 0 表示初始树高为 0
        count = n 初始有 n 个集合
        """
        self.parent: list[int] = list(range(n))  # parent[i] = i (自身为根)
        self.rank: list[int] = [0] * n            # 按秩合并所需的秩数组
        self.count: int = n                        # 当前集合数量

    def find(self, x: int) -> int:
        """
        路径压缩（递归版）：找 x 的根，并将路径上所有节点直接挂到根
        时间：均摊接近 O(1)（精确为 O(α(n))）
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # ← 递归压缩
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """
        按秩合并：将 x 和 y 所在集合合并
        返回 True 表示发生了合并（之前不同集合），False 表示已在同一集合
        边界：若 rx == ry，说明 x、y 已连通，不做操作
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False  # 已在同一集合，若用于判环则说明"成环了"
        # 将秩小的树并入秩大的树
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx     # 保证 rx 是秩更大（或相等）的根
        self.parent[ry] = rx    # ry 的根改为 rx（矮树挂到高树下）
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1  # 只有秩相等时，合并后秩才+1
        self.count -= 1         # 集合数减少 1
        return True

    def connected(self, x: int, y: int) -> bool:
        """判断 x 和 y 是否在同一集合（是否连通）"""
        return self.find(x) == self.find(y)

# === 使用示例 ===
uf = UnionFind(6)  # 6 个元素：0, 1, 2, 3, 4, 5

uf.union(0, 1)     # 合并 0 和 1
uf.union(1, 2)     # 合并 1 和 2（现在 0,1,2 连通）
uf.union(3, 4)     # 合并 3 和 4

print(uf.connected(0, 2))   # True（0 和 2 连通）
print(uf.connected(0, 3))   # False（0 和 3 不连通）
print(uf.count)              # 3（有 3 个集合：{0,1,2}、{3,4}、{5}）
```
```cpp
#include <vector>
#include <numeric>  // for iota
using namespace std;

class UnionFind {
public:
    vector<int> parent;
    vector<int> rank_;  // 注意：rank 是 C++ 关键词，用 rank_ 代替
    int count;          // 当前集合数量

    // 初始化：n 个元素，各自成一个集合
    explicit UnionFind(int n) : parent(n), rank_(n, 0), count(n) {
        iota(parent.begin(), parent.end(), 0);  // parent[i] = i
    }

    // FIND：路径压缩（递归版）
    // 注意：C++ 中深度过大可能栈溢出，大数据量请用迭代版
    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);  // 路径压缩：直接挂到根
        }
        return parent[x];
    }

    // FIND 迭代版（路径减半，更安全）
    int findIterative(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];  // 路径减半：跳过一层
            x = parent[x];
        }
        return x;
    }

    // UNION：按秩合并
    // 返回 true 表示成功合并（之前不同集合）
    bool unite(int x, int y) {
        int rx = find(x), ry = find(y);
        if (rx == ry) return false;  // 已连通

        // 矮树挂到高树下
        if (rank_[rx] < rank_[ry]) swap(rx, ry);
        parent[ry] = rx;
        if (rank_[rx] == rank_[ry]) rank_[rx]++;
        count--;
        return true;
    }

    // 判断连通性
    bool connected(int x, int y) {
        return find(x) == find(y);
    }
};

// 使用示例
/*
int main() {
    UnionFind uf(6);  // 6 个元素

    uf.unite(0, 1);
    uf.unite(1, 2);
    uf.unite(3, 4);

    cout << uf.connected(0, 2) << "\n";  // 1 (true)
    cout << uf.connected(0, 3) << "\n";  // 0 (false)
    cout << uf.count << "\n";            // 3
    return 0;
}
*/
```

### 13.4.2 Ackermann 函数与反函数 α(n)

单独用**按秩合并**：$m$ 次操作 $O(m \log n)$  
单独用**路径压缩**：$m$ 次操作 $O(m \log n)$（这没有量级改善）  
两者同时使用：$m$ 次操作 $O(m \cdot \alpha(n))$——**惊人地接近线性！**

这里 $\alpha(n)$ 是 **Ackermann 函数**的反函数。先来直觉理解 Ackermann 函数的增长有多"快"：

$$A(1, n) = 2n, \quad A(2, n) = 2^n, \quad A(3, n) \approx 2^{2^{2^{\cdots}}}\ (n \text{ 层塔}), \quad A(4, n) \approx \underbrace{2^{2^{2^{\cdots}}}}_{A(3,n)\text{层}}$$

Ackermann 函数增长极快（超过任何有限次嵌套的指数），因此其反函数 $\alpha(n)$ 增长**极其缓慢**：

$$\alpha(n) \leq 4 \quad \text{对所有实际上的 } n$$

具体地：
- $\alpha(4) = 1$
- $\alpha(16) = 2$  
- $\alpha(2^{16}) = 3$
- $\alpha(2^{65536}) = 4$（此时 $n$ 已远超宇宙中原子数目 $10^{80}$！）

**结论**：对任何你在程序中可能接触到的 $n$ 值，$\alpha(n) \leq 4$，实践中完全可以视为**常数**。

$$\text{总时间复杂度} = O(m \cdot \alpha(n)) \approx O(m) \quad \text{（工程上即线性）}$$

### 13.4.3 摊销分析直觉（CLRS Theorem 21.14 概述）

CLRS 第21章给出了严格证明，证明方法用到了**势能（Potential）函数**分析。我们只看直觉：

**为什么两个优化的组合比任意单一优化更好？**

- 按秩合并保证：没有被路径压缩过的树，高度 $\leq \log_2 n$
- 路径压缩保证：FIND 过的路径上的节点，下次 FIND 费用极低
- 两者组合：路径压缩后树实际高度极低，FIND 本已很快；而按秩合并保证压缩前树也不会太深

这两种优化形成**良性循环**：合并时树不高（按秩），查找时顺便压平（路径压缩），压平后更矮，合并时代价更低……

**实用工程结论**：在所有实际应用中，并查集的 $m$ 次 UNION/FIND 操作总时间**可以视作 $O(m)$**，即线性时间内完成所有操作。

---

## 13.5 带权并查集

### 13.5.1 什么是带权并查集？

基础并查集只能回答"$x$ 和 $y$ 是否在同一集合"，**带权并查集（Weighted Union-Find）** 则在此基础上额外维护元素之间的**相对关系/权值**，能回答"$x$ 相对于 $y$ 有什么属性？"

**经典场景：食物链**

有三种生物：兔（Rabbit）、狐（Fox）、草（Grass）。关系：
- 草被兔吃（Grass → Rabbit）
- 兔被狐吃（Rabbit → Fox）
- 狐被草吸收（Fox → Grass，形成循环）

若告诉你"$A$ 和 $B$ 属于同种/捕食关系"，如何高效判断"$C$ 和 $D$ 的关系"是否与之前陈述矛盾？

**另一经典：通货兑换汇率**

给定若干"$X$ 元 = $r$ 个 $Y$ 元"的汇率关系，查询两种货币之间的换算系数。

### 13.5.2 带权并查集的设计

在每个节点 $x$ 上，维护一个**权值 `dist[x]`**，表示 $x$ 与其父节点 `parent[x]` 之间的关系（具体语义由问题决定，如倍数关系、差值关系等）。

以"**差值关系**"为例：`dist[x]` = $x$ 相对于 `parent[x]` 的偏移量，且规定集合代表元素（根）的 `dist` 为 0。

**FIND 时的路径压缩 + 权值复合**：

FIND 过程中，将 $x$ 直接挂到根节点，同时需要把路径上的权值**累加（复合）**，得到 $x$ 相对于根的总偏移量：

```python
def find_weighted(x: int, parent: list[int], dist: list[float]) -> int:
    """
    带权並查集的 FIND（差值语义：dist[x] = x 相对于 parent[x] 的差值）
    路径压缩时，需要把沿途 dist 累加（复合）最终 dist[x] = x 相对于根的差值
    """
    if parent[x] == x:
        return x
    root = find_weighted(parent[x], parent, dist)
    # 关键：dist[x] += dist[parent[x]]（在 parent[x] 被压缩到根之后）
    dist[x] += dist[parent[x]]   # 路径上的权值要累积！
    parent[x] = root
    return root
```

> ⚠️ **最常见错误**：路径压缩做了，但忘记更新 `dist`，导致 `dist[x]` 仍指向原来的父节点，而父节点已经指向根了，这时 `dist[x]` 表示的关系不再正确。

**UNION 时的权值赋值**：

已知 $x$ 相对于根 $rx$ 的权值为 $d_x = \text{dist}[x]$（通过 FIND），$y$ 相对于根 $ry$ 的权值为 $d_y$，且题目告诉我们 $x$ 与 $y$ 的关系（如差值为 $w$）：

$$\text{dist}[ry] = d_x + w - d_y$$（差值语义，将 $ry$ 挂到 $rx$ 下，让 $ry$ 与 $rx$ 的差值如此设定）

具体计算因语义（相加/相乘/异或等）而异，**必须根据题目关系仔细推导**，这是带权并查集最易出错的地方。

**完整示例：相对差值**

```python
class WeightedUnionFind:
    """
    带权并查集（差值语义）
    dist[x] = x 相对于其集合根的"距离"（差值）
    语义：若 find(x) == find(y)，则 x - y = dist[x] - dist[y]
    """
    def __init__(self, n: int):
        self.parent: list[int] = list(range(n))
        self.dist: list[float] = [0.0] * n   # dist[i] = i - root_of_i

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            root = self.find(self.parent[x])
            # 路径压缩时累积权值
            # dist[x] 原本是 x - parent[x]
            # 压缩后 dist[x] 应变为 x - root
            # 因为 parent[x] 已被压缩，dist[parent[x]] 现在是 parent[x] - root
            # 所以 x - root = (x - parent[x]) + (parent[x] - root) = old_dist[x] + dist[parent[x]]
            self.dist[x] += self.dist[self.parent[x]]
            self.parent[x] = root
        return self.parent[x]

    def union(self, x: int, y: int, w: float) -> bool:
        """
        断言 x - y = w（即 x 比 y 大 w）
        返回 True 若合并成功（之前不同集合），False 若已同集合（可验证是否矛盾）
        """
        rx, ry = self.find(x), self.find(y)
        dx, dy = self.dist[x], self.dist[y]
        if rx == ry:
            # 验证是否矛盾：已知 x - y = dx - dy，与 w 是否一致？
            return abs((dx - dy) - w) < 1e-9  # True=一致，False=矛盾
        # 将 ry 挂到 rx 下，权值推导：
        # 需要 x - y = w
        # x = rx + dx（dist 的语义：x - root = dist[x]，所以 x = root + dist[x]）
        # Wait: dist[x] = x - rx（x 相对于根的差值）
        # 合并条件：x - y = w
        # → (dx + rx) - (dy + ry) = w（在同一参考系下）
        # 合并后 ry 的父节点为 rx，需要确定 dist[ry] = ry - rx
        # From: dist[x] = x - rx, dist[y] = y - ry
        # x - y = w → (rx + dx) - (ry + dy) = w
        # → ry - rx = dx - dy - w
        # → dist[ry] = ry - rx = dx - dy - w
        self.parent[ry] = rx
        self.dist[ry] = dx - dy - w
        return True

    def query(self, x: int, y: int) -> float | None:
        """
        查询 x - y 的值，若 x 和 y 不连通返回 None
        """
        if self.find(x) != self.find(y):
            return None
        return self.dist[x] - self.dist[y]
```
```cpp
#include <vector>
#include <numeric>
#include <cmath>
#include <optional>
using namespace std;

// 带权并查集（差值语义：dist[x] = x - root_of_set(x)）
class WeightedUnionFind {
public:
    vector<int> parent;
    vector<double> dist;  // dist[x] = x - root

    explicit WeightedUnionFind(int n) : parent(n), dist(n, 0.0) {
        iota(parent.begin(), parent.end(), 0);
    }

    // FIND：路径压缩 + 权值累积
    int find(int x) {
        if (parent[x] != x) {
            int root = find(parent[x]);
            dist[x] += dist[parent[x]];  // ← 权值复合（累加）
            parent[x] = root;
        }
        return parent[x];
    }

    // UNION：合并 x 和 y 所在集合，附带 x - y = w 的约束
    bool unite(int x, int y, double w) {
        int rx = find(x), ry = find(y);
        double dx = dist[x], dy = dist[y];
        if (rx == ry) {
            // 验证 x - y = dx - dy 是否等于 w
            return fabs((dx - dy) - w) < 1e-9;
        }
        parent[ry] = rx;
        dist[ry] = dx - dy - w;  // 推导见注释
        return true;
    }

    // QUERY：查询 x - y，不连通返回 nullopt
    optional<double> query(int x, int y) {
        if (find(x) != find(y)) return nullopt;
        return dist[x] - dist[y];
    }
};
```

### 13.5.3 应用：食物链关系（LeetCode #399 变形）

**食物链问题**：给定 $n$ 种生物，输入两类信息：
- `D x y`：$x$ 和 $y$ 是同类
- `A x y`：$x$ 吃 $y$

询问有多少句话是"谎言"（与之前已知信息矛盾，或吃自己）。

**建模**：将每种生物 $i$ 拆成三个虚拟节点：
- $i$：生物 $i$ 本身
- $i+n$：被 $i$ 捕食的物种
- $i+2n$：捕食 $i$ 的物种

这种**节点拆分技巧**将"三元关系"转化为普通并查集。具体关系：
- "同类"：合并 $(x, y)$、$(x+n, y+n)$、$(x+2n, y+2n)$
- "捕食"（$x$ 吃 $y$）：合并 $(x, y+n)$、$(x+n, y+2n)$、$(x+2n, y)$

判断语句是否是谎言：
- "$x$ 和 $y$ 同类"但 $x$ 在 $y$ 的"被吃区"或"食者区"
- "$x$ 吃 $y$"但 $y$ 在 $x$ 的"同类区"或 $x$ 在 $y$ 的"被吃区"
- 自己吃自己（$x = y$ 时"$A$ 类"操作）

<div data-component="WeightedUnionFind"></div>

---

## 13.6 典型应用

### 13.6.1 Kruskal 最小生成树——并查集的"本命用途"

**最小生成树（MST）问题**：给定带权无向连通图 $G(V, E)$，找一棵包含所有顶点、总边权最小的生成树。

**Kruskal 算法**流程：
1. 将所有边按权值**从小到大排序**
2. 依次考虑每条边 $(u, v, w)$：
   - 若 $u$ 和 $v$ **不连通**（$\text{FIND}(u) \neq \text{FIND}(v)$）：将此边加入 MST，执行 $\text{UNION}(u, v)$
   - 若 $u$ 和 $v$ **已连通**：跳过（加入此边会形成环）
3. 重复直到 MST 包含 $|V|-1$ 条边

**并查集在其中的核心作用**：判断两顶点是否已连通（是否成环）。这正是 FIND 操作的典型场景。

```python
from typing import NamedTuple

class Edge(NamedTuple):
    weight: int
    u: int
    v: int

def kruskal(n: int, edges: list[Edge]) -> list[Edge]:
    """
    Kruskal 最小生成树算法
    参数：n = 顶点数，edges = 边列表
    返回：MST 包含的边
    时间复杂度：O(E log E)（排序主导）+ O(E·α(V))（并查集操作）= O(E log E)
    """
    uf = UnionFind(n)
    mst: list[Edge] = []

    # Step 1: 按权值排序所有边 O(E log E)
    for edge in sorted(edges):  # NamedTuple 自动按字段顺序比较，weight 在前
        w, u, v = edge
        # Step 2: 判断是否成环（两端点是否已连通）
        if uf.union(u, v):      # UNION 返回 True 表示之前不连通，成功合并
            mst.append(edge)    # 加入 MST
            if len(mst) == n - 1:
                break           # MST 已有 n-1 条边，提前结束

    return mst

# 示例：5 个顶点，7 条边
n = 5
edges = [
    Edge(10, 0, 1), Edge(6, 0, 2), Edge(5, 0, 3),
    Edge(15, 1, 4), Edge(4, 2, 3), Edge(14, 2, 4),
    Edge(9, 3, 4),
]
mst = kruskal(n, edges)
print("MST 边：", mst)
# 输出：Edge(4,2,3) Edge(5,0,3) Edge(6,0,2) Edge(9,3,4) → 总权值 24
```
```cpp
#include <vector>
#include <algorithm>
#include <numeric>
using namespace std;

struct Edge {
    int weight, u, v;
    bool operator<(const Edge& o) const { return weight < o.weight; }
};

class UnionFind {
public:
    vector<int> parent, rank_;
    explicit UnionFind(int n) : parent(n), rank_(n, 0) {
        iota(parent.begin(), parent.end(), 0);
    }
    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }
    bool unite(int x, int y) {
        int rx = find(x), ry = find(y);
        if (rx == ry) return false;
        if (rank_[rx] < rank_[ry]) swap(rx, ry);
        parent[ry] = rx;
        if (rank_[rx] == rank_[ry]) rank_[rx]++;
        return true;
    }
};

// Kruskal MST
vector<Edge> kruskal(int n, vector<Edge> edges) {
    // Step 1: 按权值排序  O(E log E)
    sort(edges.begin(), edges.end());

    UnionFind uf(n);
    vector<Edge> mst;

    for (const auto& e : edges) {
        // Step 2: 不成环则加入 MST
        if (uf.unite(e.u, e.v)) {
            mst.push_back(e);
            if ((int)mst.size() == n - 1) break;  // MST 完成
        }
    }
    return mst;
}

// main 示例
// int main() {
//     int n = 5;
//     vector<Edge> edges = {{10,0,1},{6,0,2},{5,0,3},{15,1,4},{4,2,3},{14,2,4},{9,3,4}};
//     auto mst = kruskal(n, edges);
//     for (auto& e : mst) cout << e.u << "-" << e.v << "(" << e.weight << ") ";
// }
```

**复杂度分析**：
- 排序：$O(E \log E)$
- 并查集操作：$O(E \cdot \alpha(V))$，实践中约 $O(E)$
- 总体：$O(E \log E)$，排序主导
- 稀疏图（$E \approx V$）时 Kruskal 优于 Prim；稠密图（$E \approx V^2$）时 Prim + 优先队列 $O(E \log V)$ 或 Prim + Fibonacci 堆 $O(E + V \log V)$ 更优

<div data-component="KruskalUnionFindSim"></div>

### 13.6.2 图连通分量动态计算

**问题**：给定 $n$ 个节点，边一条一条地动态加入，每加一条边后，查询当前有多少个连通分量。

```python
def count_components_dynamic(n: int, edges: list[tuple[int, int]]) -> list[int]:
    """
    动态添加边，每次记录当前连通分量数量
    时间：O(m·α(n))
    """
    uf = UnionFind(n)
    results: list[int] = []

    for u, v in edges:
        uf.union(u, v)        # 若合并成功，uf.count 自动 -1
        results.append(uf.count)

    return results

# 示例
edges = [(0,1), (1,2), (3,4), (2,3)]
print(count_components_dynamic(5, edges))
# 初始 5 个分量
# 加 (0,1): 4 个
# 加 (1,2): 3 个
# 加 (3,4): 2 个
# 加 (2,3): 1 个（全部连通）
# 输出：[4, 3, 2, 1]
```
```cpp
#include <vector>
#include <numeric>
using namespace std;

// 使用同样的 UnionFind 类（见 13.4.1）
vector<int> countComponentsDynamic(int n, vector<pair<int,int>>& edges) {
    UnionFind uf(n);
    vector<int> result;
    for (auto [u, v] : edges) {
        uf.unite(u, v);
        result.push_back(uf.count);
    }
    return result;
}
```

**LeetCode 原题**：#323「无向图中连通分量的数目」——建图后直接统计 `count` 即可。

### 13.6.3 网格连通分量——岛屿数量的动态版

**经典问题**：给定 $m \times n$ 的网格，每次将一格从"水"变为"陆地"，每次操作后查询当前陆地的连通分量数（岛屿数）。

**关键技巧**：将二维坐标 $(i, j)$ 映射为一维编号 $i \times n + j$，然后正常做并查集。加入一块陆地时，与其上下左右的已有陆地尝试 UNION。

```python
DIRS = [(0,1),(0,-1),(1,0),(-1,0)]  # 上下左右四个方向

def num_islands_dynamic(m: int, n: int, positions: list[tuple[int,int]]) -> list[int]:
    """
    离线版岛屿数量（动态添加陆地）
    positions: 依次添加的陆地坐标 (r, c)
    返回：每次添加后的岛屿数量
    """
    uf = UnionFind(m * n)
    grid_state = [[False] * n for _ in range(m)]
    island_count = 0
    results = []

    for r, c in positions:
        if grid_state[r][c]:
            results.append(island_count)  # 重复添加，不变
            continue

        grid_state[r][c] = True
        island_count += 1          # 新加一块陆地，先当作新岛屿

        cur = r * n + c
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and grid_state[nr][nc]:
                neighbor = nr * n + nc
                if uf.union(cur, neighbor):
                    island_count -= 1  # 成功合并两个岛，岛屿数 -1

        results.append(island_count)

    return results

# 示例：3×3 网格，依次添加陆地
positions = [(0,0),(0,1),(1,2),(2,1),(1,1)]
print(num_islands_dynamic(3, 3, positions))
# 输出：[1, 1, 2, 3, 1]
# 最后加 (1,1) 把所有相邻陆地合并成 1 个岛
```
```cpp
#include <vector>
#include <numeric>
using namespace std;

// 使用同样的 UnionFind 类
vector<int> numIslandsDynamic(int m, int n, vector<pair<int,int>>& positions) {
    UnionFind uf(m * n);
    vector<vector<bool>> grid(m, vector<bool>(n, false));
    int islandCount = 0;
    vector<int> results;

    int dirs[4][2] = {{0,1},{0,-1},{1,0},{-1,0}};

    for (auto [r, c] : positions) {
        if (grid[r][c]) {
            results.push_back(islandCount);
            continue;
        }
        grid[r][c] = true;
        islandCount++;

        int cur = r * n + c;
        for (auto& d : dirs) {
            int nr = r + d[0], nc = c + d[1];
            if (nr >= 0 && nr < m && nc >= 0 && nc < n && grid[nr][nc]) {
                if (uf.unite(cur, nr * n + nc))
                    islandCount--;
            }
        }
        results.push_back(islandCount);
    }
    return results;
}
```

**进阶变形**：
- 上下左右外加**对角线**连通（8 方向）：把 `DIRS` 扩展为 8 个方向
- "逆向"处理：先把所有陆地加进去，再倒序删除，转化为添加问题

### 13.6.4 账户合并（LeetCode #721）

**问题**：给定账户列表，每个账户格式为 `[用户名, 邮箱1, 邮箱2, ...]`。若两个账户有公共邮箱，则认为是同一人。合并属于同一人的所有账户，输出合并结果。

**核心思路**：
1. 将每个邮箱映射为一个整数 ID（用 `dict`）
2. 同一账户内的所有邮箱 UNION 到一起（视作连通）
3. 利用 FIND 找出每个邮箱所属的集合（代表邮箱）
4. 将同集合邮箱收集、排序、加上用户名，输出

```python
from collections import defaultdict

def account_merge(accounts: list[list[str]]) -> list[list[str]]:
    """
    LeetCode #721 账户合并
    时间复杂度：O(N·K·α(N·K))，N=账户数，K=每账户最多邮箱数
    """
    email_to_id: dict[str, int] = {}   # 邮箱 → 唯一ID（并查集节点编号）
    email_to_name: dict[str, str] = {} # 邮箱 → 账户名

    # 第一步：为每个邮箱分配 ID，记录账户名
    for account in accounts:
        name = account[0]
        for email in account[1:]:
            if email not in email_to_id:
                email_to_id[email] = len(email_to_id)
            email_to_name[email] = name

    # 第二步：并查集初始化（节点数 = 唯一邮箱总数）
    uf = UnionFind(len(email_to_id))

    # 第三步：同账户内的邮箱全部 UNION 到一起
    for account in accounts:
        first_email_id = email_to_id[account[1]]
        for email in account[2:]:
            uf.union(first_email_id, email_to_id[email])

    # 第四步：按连通分量分组
    groups: dict[int, list[str]] = defaultdict(list)
    for email, eid in email_to_id.items():
        root = uf.find(eid)
        groups[root].append(email)

    # 第五步：构造输出（组内邮箱排序）
    result: list[list[str]] = []
    for root, emails in groups.items():
        # 用该组任意邮箱查用户名（同组用户名相同）
        name = email_to_name[emails[0]]
        result.append([name] + sorted(emails))

    return result

# 示例
accounts = [
    ["Alice", "a@x.com", "b@x.com"],
    ["Bob",   "c@x.com"],
    ["Alice", "b@x.com", "d@x.com"],  # b@x.com 与第1个账户重叠 → 合并
]
print(account_merge(accounts))
# 输出：
# [["Alice", "a@x.com", "b@x.com", "d@x.com"],
#  ["Bob", "c@x.com"]]
```
```cpp
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <set>
#include <algorithm>
#include <numeric>
using namespace std;

// 使用同样的 UnionFind 类
vector<vector<string>> accountsMerge(vector<vector<string>>& accounts) {
    unordered_map<string, int> emailToId;
    unordered_map<string, string> emailToName;

    // 第一步：为每个邮箱分配唯一 ID
    for (auto& account : accounts) {
        const string& name = account[0];
        for (int i = 1; i < (int)account.size(); i++) {
            const string& email = account[i];
            if (!emailToId.count(email)) {
                emailToId[email] = emailToId.size();
            }
            emailToName[email] = name;
        }
    }

    // 第二步：并查集，节点数 = 唯一邮箱总数
    UnionFind uf(emailToId.size());

    // 第三步：同账户内的邮箱 UNION
    for (auto& account : accounts) {
        int firstId = emailToId[account[1]];
        for (int i = 2; i < (int)account.size(); i++) {
            uf.unite(firstId, emailToId[account[i]]);
        }
    }

    // 第四步：按连通分量分组
    map<int, set<string>> groups;  // root → sorted email set
    for (auto& [email, id] : emailToId) {
        groups[uf.find(id)].insert(email);
    }

    // 第五步：构造输出
    vector<vector<string>> result;
    for (auto& [root, emails] : groups) {
        string name = emailToName[*emails.begin()];
        vector<string> merged = {name};
        merged.insert(merged.end(), emails.begin(), emails.end());
        result.push_back(merged);
    }
    return result;
}
```

---

## 本章总结

### 知识点全景图

```
并查集 (Disjoint Set Union)
├── 问题：动态连通性（在线 UNION/FIND 查询）
├── 实现 1：链表表示
│   ├── FIND O(1)，UNION O(|小集合|)
│   └── 加权合并 → O(m + n log n) 总代价
└── 实现 2：树形表示（森林）★ 主流选择
    ├── 朴素：FIND O(n) 最坏
    ├── 优化 A：按秩合并 → 树高 O(log n) → O(m log n)
    ├── 优化 B：路径压缩 → 摊销优化
    └── A+B 同时 → O(m·α(n)) ≈ O(m) 实践线性
        ├── 变体：路径分裂、路径减半（理论等价）
        └── 扩展：带权并查集（维护相对关系）
```

### 代码模板（竞赛/面试速记版）

并查集的核心代码极短，下面是最精炼的竞赛版本：

```python
# 竞赛速用模板（路径减半 + 按秩合并）
parent = list(range(n))
rank = [0] * n

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]  # 路径减半
        x = parent[x]
    return x

def union(x, y):
    rx, ry = find(x), find(y)
    if rx == ry: return False
    if rank[rx] < rank[ry]: rx, ry = ry, rx
    parent[ry] = rx
    if rank[rx] == rank[ry]: rank[rx] += 1
    return True
```
```cpp
// 竞赛速用模板
int parent[MAXN], rnk[MAXN];

int find(int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];  // 路径减半
        x = parent[x];
    }
    return x;
}

bool unite(int x, int y) {
    x = find(x); y = find(y);
    if (x == y) return false;
    if (rnk[x] < rnk[y]) swap(x, y);
    parent[y] = x;
    if (rnk[x] == rnk[y]) rnk[x]++;
    return true;
}

// init: for(int i=0;i<n;i++) parent[i]=i, rnk[i]=0;
```

### 复杂度与实现对比

| 实现方式 | FIND | UNION | $m$ 次总计 | 空间 |
|---------|------|-------|----------|------|
| 朴素链表 | $O(1)$ | $O(n)$ | $O(mn)$ | $O(n)$ |
| 加权合并链表 | $O(1)$ | $O(\log n)$（均摊） | $O(m + n\log n)$ | $O(n)$ |
| 朴素树形 | $O(n)$ | $O(n)$ | $O(mn)$ | $O(n)$ |
| 按秩合并 | $O(\log n)$ | $O(\log n)$ | $O(m\log n)$ | $O(n)$ |
| 路径压缩 | $O(\log n)$（均摊） | $O(1)$ | $O(m\log n)$ | $O(n)$ |
| **按秩 + 路径压缩** ★ | $O(\alpha(n))$ | $O(\alpha(n))$ | $O(m\cdot\alpha(n))$ | $O(n)$ |

### 常见错误与调试技巧

> ❌ **错误 1：路径压缩遗忘更新 parent**
> ```python
> def find_wrong(x):
>     if parent[x] == x: return x
>     return find_wrong(parent[x])  # ← 没有修改 parent[x]！每次还是要走完整路径
> ```
> ✅ 正确：`parent[x] = find(parent[x]); return parent[x]`

> ❌ **错误 2：带权并查集路径压缩忘记更新 dist**
> 路径压缩时只把 `parent[x] = root`，但没有把 `dist[x]` 从"相对父节点的权"更新为"相对根的权"。

> ❌ **错误 3：rank 与 size 混用**
> `rank` 是秩（树高上界），不是集合大小！按大小合并是另一种优化（`parent[smaller_root] = larger_root`，维护 `size` 数组），两者不能混用。

> ❌ **错误 4：UNION 的返回值含义**
> 并查集解决"判断是否成环"时，逻辑是：若 UNION 返回 `False`（已在同集合），说明这条边**形成了环**。不少代码反而把两种情况搞反了。

### 经典 LeetCode 练习题

| 题号 | 难度 | 关键技巧 |
|------|------|---------|
| #200 岛屿数量 | 🟡 中 | 并查集建模（二维→一维）|
| #323 连通分量数 | 🟢 易 | 直接并查集，输出 `count` |
| #684 冗余连接 | 🟡 中 | UNION 时检测环（返回 False 的边）|
| #547 省份数量 | 🟢 易 | 同 #323，矩阵输入 |
| #990 等式方程可满足性 | 🟡 中 | 先 UNION 所有 `==`，再验证所有 `!=` |
| #721 账户合并 | 🔴 难 | 邮箱→ID，字符串映射 + 并查集 |
| #1202 交换字符串中的元素 | 🟡 中 | 同集合字母可任意排列 |
| #1584 连接所有点的最小费用 | 🟡 中 | Kruskal + 曼哈顿距离建边 |

### 思考题

1. **Kruskal vs Prim**：Kruskal 依赖所有边排序 $O(E\log E)$，Prim 用堆 $O(E\log V)$。对于**稀疏图**（$E \approx V$）和**稠密图**（$E \approx V^2$），各自更适合哪种算法？为什么？

2. **路径压缩单独使用的复杂度**：只使用路径压缩（不用按秩合并），最坏情况 m 次操作是否能达到 $O(m\log n)$？你能构造一个最坏情形验证吗？

3. **α(n) ≤ 4 的实用性**：宇宙中原子约 $10^{80}$，$\alpha(10^{80})$ 等于多少？这对我们用并查集分析算法有何工程意义？

4. **并查集的局限**：并查集只支持"合并"，不支持"拆分"（将一个集合拆为两个）。如果需要支持拆分，应该用什么数据结构？（提示：Link-Cut Tree）

---

**参考资料**：
- CLRS 第4版 Chapter 21（不相交集合数据结构）
- Sedgewick 《算法》第1.5节（Union-Find）
- MIT 6.006 Lecture 16（Union-Find & Kruskal）
- [OI Wiki 并查集](https://oi-wiki.org/ds/dsu/)
- LeetCode 并查集专题：[Union Find](https://leetcode.com/tag/union-find/)

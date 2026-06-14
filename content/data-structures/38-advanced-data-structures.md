---
title: "Chapter 38: 其他高级数据结构"
description: "跳表、van Emde Boas 树、分块与莫队算法、KD-Tree——工程与竞赛中的高阶利器"
tags: ["skip-list", "van-emde-boas", "sqrt-decomposition", "mo-algorithm", "kd-tree"]
difficulty: "hard"
updated: "2026-03-12"
---

# Chapter 38: 其他高级数据结构

> **Part X · 高级数据结构与摊销分析**

本章我们来学习四种各具特色的高级数据结构：**跳表（Skip List）**、**van Emde Boas 树（vEB Tree）**、**分块与莫队（Sqrt Decomposition / Mo's Algorithm）**、**KD-Tree**。  
它们的共同点是：在某些场景下能在理论或工程上远胜常规方案。  
学完本章，你将能够在面试和竞赛中灵活选择"恰好合适"的工具。

---

## 38.1 跳表（Skip List）

### 38.1.1 为什么需要跳表？——从"书的目录"说起

想象你有一本 1000 页的字典，想查"Xenon"（氙）这个词。如果从第 1 页开始翻，要翻到第 900 多页才能找到，太慢了。  
但如果字典有**多个级别的目录**：

- 第 0 层：所有词（完整正文）
- 第 1 层：每 4 个词取 1 个（粗目录）
- 第 2 层：每 16 个词取 1 个（更粗目录）
- 第 3 层：每 64 个词取 1 个（章节目录）

查找时：先在最高层目录快速跳过大块内容，再逐层降到更细粒度，最终锁定目标位置。这就是跳表（**Skip List**）的核心思想。

> **技术动机**：有序链表的查找是 $O(n)$，而 BST/红黑树实现复杂。跳表通过**多层链表**，让有序集合的查找期望变成 $O(\log n)$，代码却比红黑树简单几倍。

### 38.1.2 跳表的物理结构

```
Level 3: 1 ─────────────────────────── 51 ──→ NIL
Level 2: 1 ─────────── 17 ─────────── 51 ──→ NIL
Level 1: 1 ─── 6 ───── 17 ──── 31 ─── 51 ──→ NIL
Level 0: 1 ─ 3 ─ 6 ─ 9 ─ 17 ─ 27 ─ 31 ─ 43 ─ 51 ──→ NIL
```

**核心规则**：
- **Level 0**（底层）：包含所有元素，按升序排列的单向链表。
- **Level $k$**（高层）：概率 $p$（通常 $p = 1/2$ 或 $p = 1/4$）包含 Level $k-1$ 的元素。
- 每个节点有多个**前向指针（forward pointer）**，`forward[k]` 指向该节点在第 $k$ 层的下一个节点。

**每个节点**（Node）的结构的伪代码：

```
Node {
    key         // 键
    value       // 值
    forward[]   // forward[0..level] 前向指针数组
}
```

<div data-component="SkipListStructureViz"></div>

### 38.1.3 期望层数与空间分析

**期望最大层数**：对于 $n$ 个元素和晋升概率 $p$，期望最大层数为：

$$L(n) = \log_{1/p}(n)$$

- $p=1/2$：期望 $\log_2 n$ 层（16 层可容纳 65536 个元素）
- $p=1/4$：期望 $\log_4 n$ 层（即 $\frac{\log_2 n}{2}$ 层）

**期望总指针数**（即原始空间开销）：

每个元素期望拥有的指针数：
$$\sum_{k=0}^{\infty} p^k = \frac{1}{1-p}$$

因此总空间期望为 $O\!\left(\dfrac{n}{1-p}\right)$，即 $O(n)$。

| 概率 $p$ | 期望层数 | 每节点指针数 | 空间 |
|----------|----------|-------------|------|
| 1/2 | $\log_2 n$ | 2 | $O(2n)$ |
| 1/4 | $\log_4 n$ | 4/3 | $O(1.33n)$ |
| 1/e | $\ln n$ | $e/(e-1)\approx1.58$ | $O(1.58n)$ |

### 38.1.4 查找（SEARCH）：从高层开始向右向下

**直觉**：从最高层开始，尽量向右跳（跳过比目标小的元素），当无法再右跳时下一层，直到第 0 层找到（或确认不存在）。

**伪代码（SKIP-LIST-SEARCH）**：
```
SKIP-LIST-SEARCH(list, target):
    curr = list.header            // 从哨兵头节点开始
    for k = list.maxLevel down to 0:
        while curr.forward[k] != NIL and curr.forward[k].key < target:
            curr = curr.forward[k]  // 同层右跳
        // 无法再右跳：在当前层找到了目标的前驱，下降一层
    if curr.forward[0] != NIL and curr.forward[0].key == target:
        return curr.forward[0].value  // 找到
    return NOT_FOUND
```

**期望时间复杂度**：$O(\log n)$（每层期望只跳 $1/p$ 步，共 $\log_{1/p} n$ 层）。

<div data-component="SkipListSearchViz"></div>

### 38.1.5 插入（INSERT）：随机层高 + 更新前驱指针

**插入步骤**：
1. 和查找类似，逐层找到目标的**前驱节点数组** `update[0..maxLevel]`（每层在哪里插入）。
2. 随机生成新节点的**层高** `newLevel`（掷硬币法：每次 $\text{random()} < p$ 则层高 +1）。
3. 如果 `newLevel > list.currentLevel`，更新 `update[currentLevel+1..newLevel] = header`。
4. 创建新节点，逐层将其插入 `update[k]` 和 `update[k].forward[k]` 之间。

**随机层高生成（RANDOM-LEVEL）**：
```
RANDOM-LEVEL(maxLevel, p):
    level = 0
    while random() < p and level < maxLevel - 1:
        level += 1
    return level
```

**伪代码（SKIP-LIST-INSERT）**：
```
SKIP-LIST-INSERT(list, key, value):
    update = array of NIL, size maxLevel
    curr = list.header
    for k = list.maxLevel down to 0:
        while curr.forward[k] != NIL and curr.forward[k].key < key:
            curr = curr.forward[k]
        update[k] = curr      // 记录每层的前驱节点

    newLevel = RANDOM-LEVEL()
    if newLevel > list.currentLevel:
        for k = list.currentLevel+1 to newLevel:
            update[k] = list.header    // 新层的前驱是 header
        list.currentLevel = newLevel

    newNode = Node(key, value, newLevel)
    for k = 0 to newLevel:
        newNode.forward[k] = update[k].forward[k]
        update[k].forward[k] = newNode   // 插入新节点
```

**期望时间复杂度**：$O(\log n)$（查找前驱 + 插入各层共计期望 $O(\log n)$ 步）。

<div data-component="SkipListInsertRandom"></div>

### 38.1.6 删除（DELETE）：找前驱 + 逐层摘除

**删除步骤**：与插入第 1 步相同，先找 `update[0..maxLevel]`，然后从第 0 层到 `target.level` 层逐层将 target 从链表中移除。

**伪代码（SKIP-LIST-DELETE）**：
```
SKIP-LIST-DELETE(list, key):
    update = array of NIL, size maxLevel
    curr = list.header
    for k = list.maxLevel down to 0:
        while curr.forward[k] != NIL and curr.forward[k].key < key:
            curr = curr.forward[k]
        update[k] = curr

    target = curr.forward[0]
    if target == NIL or target.key != key:
        return  // 不存在，直接返回

    for k = 0 to target.level:
        if update[k].forward[k] != target:
            break
        update[k].forward[k] = target.forward[k]  // 逐层摘除

    // 可能需要更新 currentLevel（若顶层空了）
    while list.currentLevel > 0 and list.header.forward[list.currentLevel] == NIL:
        list.currentLevel -= 1
```

**期望时间复杂度**：$O(\log n)$。

### 38.1.7 完整代码实现

```python
import random

class SkipListNode:
    """跳表节点：每个节点有一个 key、一个 value，以及一组前向指针"""
    def __init__(self, key, value, level):
        self.key = key
        self.value = value
        # forward[k] 指向第 k 层的下一个节点
        self.forward = [None] * (level + 1)

class SkipList:
    """
    跳表（Skip List）实现
    - p: 晋升概率（默认 0.5）
    - MAX_LEVEL: 最大层数（默认 16，支持 n ≤ 65536）
    """
    MAX_LEVEL = 16
    P = 0.5

    def __init__(self):
        # 哨兵头节点：key = -∞，level = MAX_LEVEL
        self.header = SkipListNode(float('-inf'), None, self.MAX_LEVEL)
        self.current_level = 0  # 当前实际最高层（从 0 开始）

    def _random_level(self):
        """掷硬币决定新节点的层高：每次 < P 则继续增层"""
        level = 0
        while random.random() < self.P and level < self.MAX_LEVEL:
            level += 1
        return level

    def search(self, key):
        """查找 key，返回 value，不存在则返回 None"""
        curr = self.header
        # 从最高层向下逐层跳跃
        for k in range(self.current_level, -1, -1):
            while curr.forward[k] and curr.forward[k].key < key:
                curr = curr.forward[k]  # 同层右跳
        # 此时 curr.forward[0] 是第 0 层中 ≥ key 的第一个节点
        curr = curr.forward[0]
        if curr and curr.key == key:
            return curr.value
        return None

    def insert(self, key, value):
        """插入 (key, value)，若 key 已存在则更新 value"""
        update = [None] * (self.MAX_LEVEL + 1)
        curr = self.header

        # 记录每层的前驱节点
        for k in range(self.current_level, -1, -1):
            while curr.forward[k] and curr.forward[k].key < key:
                curr = curr.forward[k]
            update[k] = curr  # 第 k 层中 key 的前驱

        curr = curr.forward[0]
        # 若 key 已存在，直接更新值
        if curr and curr.key == key:
            curr.value = value
            return

        new_level = self._random_level()
        # 若新节点层高超过当前最高层，补全 update
        if new_level > self.current_level:
            for k in range(self.current_level + 1, new_level + 1):
                update[k] = self.header  # 超出部分的前驱是 header
            self.current_level = new_level

        new_node = SkipListNode(key, value, new_level)
        # 逐层插入：new_node 插到 update[k] 和 update[k].forward[k] 之间
        for k in range(new_level + 1):
            new_node.forward[k] = update[k].forward[k]
            update[k].forward[k] = new_node

    def delete(self, key):
        """删除 key，不存在则不操作"""
        update = [None] * (self.MAX_LEVEL + 1)
        curr = self.header

        for k in range(self.current_level, -1, -1):
            while curr.forward[k] and curr.forward[k].key < key:
                curr = curr.forward[k]
            update[k] = curr

        target = curr.forward[0]
        if target is None or target.key != key:
            return  # key 不存在

        # 逐层摘除 target
        for k in range(self.current_level + 1):
            if update[k].forward[k] != target:
                break  # target 不在这层，提前结束
            update[k].forward[k] = target.forward[k]

        # 若顶层空了，降低 current_level
        while self.current_level > 0 and self.header.forward[self.current_level] is None:
            self.current_level -= 1
```

```cpp
#include <bits/stdc++.h>
using namespace std;

// 跳表节点
struct SkipNode {
    int key, val;
    vector<SkipNode*> forward;  // forward[k] 指向第 k 层下一节点
    SkipNode(int k, int v, int level)
        : key(k), val(v), forward(level + 1, nullptr) {}
};

class SkipList {
    static constexpr int MAX_LEVEL = 16;
    static constexpr double P = 0.5;

    SkipNode* header;  // 哨兵头节点，key = INT_MIN
    int cur_level;     // 当前实际最高层

    int random_level() {
        int level = 0;
        // 每次随机 < P 则继续增层
        while ((double)rand() / RAND_MAX < P && level < MAX_LEVEL)
            ++level;
        return level;
    }

public:
    SkipList() : cur_level(0) {
        header = new SkipNode(INT_MIN, 0, MAX_LEVEL);
    }

    // 查找 key，返回值；不存在返回 -1
    int search(int key) {
        SkipNode* curr = header;
        for (int k = cur_level; k >= 0; --k)
            while (curr->forward[k] && curr->forward[k]->key < key)
                curr = curr->forward[k];
        curr = curr->forward[0];
        if (curr && curr->key == key) return curr->val;
        return -1;
    }

    // 插入 (key, val)；key 存在则更新
    void insert(int key, int val) {
        vector<SkipNode*> update(MAX_LEVEL + 1, nullptr);
        SkipNode* curr = header;
        for (int k = cur_level; k >= 0; --k) {
            while (curr->forward[k] && curr->forward[k]->key < key)
                curr = curr->forward[k];
            update[k] = curr;
        }
        curr = curr->forward[0];
        if (curr && curr->key == key) { curr->val = val; return; }

        int new_level = random_level();
        if (new_level > cur_level) {
            for (int k = cur_level + 1; k <= new_level; ++k)
                update[k] = header;
            cur_level = new_level;
        }
        SkipNode* node = new SkipNode(key, val, new_level);
        for (int k = 0; k <= new_level; ++k) {
            node->forward[k] = update[k]->forward[k];
            update[k]->forward[k] = node;
        }
    }

    // 删除 key
    void remove(int key) {
        vector<SkipNode*> update(MAX_LEVEL + 1, nullptr);
        SkipNode* curr = header;
        for (int k = cur_level; k >= 0; --k) {
            while (curr->forward[k] && curr->forward[k]->key < key)
                curr = curr->forward[k];
            update[k] = curr;
        }
        SkipNode* target = curr->forward[0];
        if (!target || target->key != key) return;
        for (int k = 0; k <= cur_level; ++k) {
            if (update[k]->forward[k] != target) break;
            update[k]->forward[k] = target->forward[k];
        }
        delete target;
        while (cur_level > 0 && !header->forward[cur_level])
            --cur_level;
    }
};
```

### 38.1.8 跳表 vs 红黑树

| 特性 | 跳表 | 红黑树 |
|------|------|--------|
| **查找** | 期望 $O(\log n)$ | 最坏 $O(\log n)$ |
| **插入** | 期望 $O(\log n)$ | 最坏 $O(\log n)$ |
| **删除** | 期望 $O(\log n)$ | 最坏 $O(\log n)$ |
| **实现复杂度** | ⭐ 简单 | ⭐⭐⭐ 复杂（旋转+染色） |
| **有序遍历** | $O(n)$，底层链表天然有序 | $O(n)$，中序遍历 |
| **范围查询** | 极简：找到起点后沿底层链表遍历 | 需要中序遍历管理 |
| **并发友好性** | ✅ 更容易实现 Lock-Free | ❌ 旋转操作难并发 |
| **随机性依赖** | 期望性能依赖 RNG 质量 | 确定性保证 |
| **空间** | $O(n)$ 期望，常数 ≈ 2 | $O(n)$ 确定 |

> **Redis ZSet 为什么用跳表**：Redis 的有序集合（Sorted Set，ZSet）内部同时使用**哈希表**（O(1) 查 score）和**跳表**（O(log n) 按 score 范围查询 + 排名查询）。跳表的范围查询和排名（ZRANK/ZRANGE）实现比红黑树更直观，Redis 作者 Antirez 在博客中专门说明了这一考量。

<div data-component="SkipListVsRBTree"></div>

---

## 38.2 van Emde Boas 树（vEB Tree）

### 38.2.1 问题背景：整数全集上的优先队列

**场景**：有一个全集 $U = \{0, 1, 2, \ldots, u-1\}$（如所有 0~65535 的整数），我们需要维护一个动态子集 $S \subseteq U$，支持以下操作：

| 操作 | 含义 |
|------|------|
| `MIN()` / `MAX()` | 最小/最大元素 |
| `MEMBER(x)` | 判断 $x$ 是否在 $S$ 中 |
| `SUCCESSOR(x)` | $x$ 在 $S$ 中的后继（大于 $x$ 的最小元素） |
| `PREDECESSOR(x)` | $x$ 在 $S$ 中的前驱 |
| `INSERT(x)` | 插入 $x$ |
| `DELETE(x)` | 删除 $x$ |

**普通方法的复杂度**：

| 方法 | 所有操作 |
|-----------------------|------|
| 比较排序（BST/RBT） | $O(\log n)$ |
| 位图（Bitset） | $O(1)$ MIN/MAX 困难 |
| **vEB树** | $O(\log \log u)$ |

> **直觉**：$\log \log u$ 有多快？如果 $u = 2^{32}$（约 40 亿），$\log u = 32$，$\log \log u = 5$！也就是说，5 步内完成所有操作。

### 38.2.2 O(log log u) 的来源：子问题大小每次变为 sqrt

**递归分析**：vEB 树每次递归将子问题从大小 $u$ 缩小到 $\sqrt{u}$（子全集的大小）。

设 $T(u)$ 为全集大小 $u$ 时操作的时间复杂度：

$$T(u) = T(\sqrt{u}) + O(1)$$

令 $s = \log u$（即 $u = 2^s$），则 $\sqrt{u} = 2^{s/2}$，$\log(\sqrt{u}) = s/2$：

$$T(2^s) = T(2^{s/2}) + O(1)$$

令 $F(s) = T(2^s)$，则 $F(s) = F(s/2) + O(1)$，解得 $F(s) = O(\log s) = O(\log \log u)$。

### 38.2.3 vEB 树的结构设计

**基础思路**：将全集 $U = \{0, \ldots, u-1\}$ 劈成 $\sqrt{u}$ 个**簇（cluster）**，每个簇大小为 $\sqrt{u}$：

- `high(x) = ⌊x / √u⌋`：元素 $x$ 属于第几个簇（高位部分）
- `low(x) = x mod √u`：元素 $x$ 在所在簇内的偏移（低位部分）
- `index(h, l) = h × √u + l`：从高位和低位复原元素

**vEB 树（记 $\text{vEB}(u)$）的结构**：

```
vEB(u):
    min       ← 最小值（直接存储，不放入 cluster）
    max       ← 最大值（直接存储，不放入 cluster）
    cluster[] ← 共 √u 个子 vEB，每个是 vEB(√u)
    summary   ← 一个 vEB(√u)，记录哪些 cluster 非空
```

**最关键设计**：`min` 和 `max` 直接存在节点里，**不放进 cluster**——这是所有 $O(1)$ 特殊情形的来源。

<div data-component="vEBTreeStructure"></div>

### 38.2.4 核心操作伪代码

**MINIMUM / MAXIMUM**（$O(1)$）：直接返回节点的 `min` / `max`。

**MEMBER(V, x)**（$O(\log \log u)$）：
```
vEB-MEMBER(V, x):
    if x == V.min or x == V.max:
        return TRUE
    if V.u == 2:         // 基础情形：u=2 只有 0 和 1
        return FALSE
    return vEB-MEMBER(V.cluster[high(x)], low(x))
```

**SUCCESSOR(V, x)**（$O(\log \log u)$，核心代码）：
```
vEB-SUCCESSOR(V, x):
    if V.u == 2:
        if x == 0 and V.max == 1: return 1
        else: return ∞

    if V.min != NIL and x < V.min:
        return V.min  // x 比 min 还小，后继就是 min

    maxLow = vEB-MAX(V.cluster[high(x)])
    if maxLow != NIL and low(x) < maxLow:
        // 后继在同一个 cluster 内
        offset = vEB-SUCCESSOR(V.cluster[high(x)], low(x))
        return index(high(x), offset)
    else:
        // 后继在下一个非空 cluster 里
        succCluster = vEB-SUCCESSOR(V.summary, high(x))
        if succCluster == ∞: return ∞
        offset = vEB-MIN(V.cluster[succCluster])
        return index(succCluster, offset)
```

**INSERT(V, x)**（$O(\log \log u)$）：
```
vEB-INSERT(V, x):
    if V.min == NIL:   // V 为空，直接存
        V.min = V.max = x
        return

    if x < V.min:
        swap(x, V.min)  // 新 min 直接存，原 min 递归插入

    if V.u > 2:
        if vEB-MIN(V.cluster[high(x)]) == NIL:
            // 该 cluster 原来为空，需更新 summary
            vEB-INSERT(V.summary, high(x))
            V.cluster[high(x)].min = V.cluster[high(x)].max = low(x)
        else:
            vEB-INSERT(V.cluster[high(x)], low(x))

    if x > V.max:
        V.max = x
```

**DELETE(V, x)**（$O(\log \log u)$，最复杂）：
```
vEB-DELETE(V, x):
    if V.min == V.max:
        V.min = V.max = NIL
        return

    if V.u == 2:
        V.min = V.max = (1 - x)
        return

    if x == V.min:
        // 替换 min：找下一个最小值
        firstCluster = vEB-MIN(V.summary)
        x = index(firstCluster, vEB-MIN(V.cluster[firstCluster]))
        V.min = x  // 更新 min

    vEB-DELETE(V.cluster[high(x)], low(x))
    if vEB-MIN(V.cluster[high(x)]) == NIL:
        // 该 cluster 变空，更新 summary
        vEB-DELETE(V.summary, high(x))
        if x == V.max:
            summaryMax = vEB-MAX(V.summary)
            if summaryMax == NIL:
                V.max = V.min
            else:
                V.max = index(summaryMax, vEB-MAX(V.cluster[summaryMax]))
    elif x == V.max:
        V.max = index(high(x), vEB-MAX(V.cluster[high(x)]))
```

### 38.2.5 空间分析与哈希版本

**朴素版**空间：$O(u)$，所有 cluster 提前分配——当 $u = 2^{32}$ 时需要 $4 \times 10^9$ 个对象，不现实。

**哈希版（Proto-vEB）**：用哈希表代替 cluster 数组，只存储非空簇。空间降为 $O(n)$，但操作期望复杂度仍 $O(\log \log u)$（期望，不是最坏）。

**实际使用**：vEB 树在学术竞赛题和 IP 路由（PATRICIA trie 的变体）中出现，工程上因实现复杂和空间问题较少直接使用。

<div data-component="vEBRecursionTree"></div>

### 38.2.6 完整代码（简化版，u 为 2 的幂次）

```python
class VEB:
    """
    van Emde Boas 树（简化版）
    u 必须是 2 的幂次。空间 O(u)，操作 O(log log u)。
    """
    def __init__(self, u: int):
        assert u >= 2 and (u & (u-1)) == 0, "u 必须是 2 的幂次"
        self.u = u
        self.min_val = None  # 最小值（不放入 cluster）
        self.max_val = None  # 最大值（不放入 cluster）
        if u > 2:
            # 子簇大小：sqrt(u)
            self._su = int(u ** 0.5)
            # cluster[i] 是第 i 个子 vEB(sqrt(u))
            self.cluster = [VEB(self._su) for _ in range(self._su)]
            # summary 记录哪些 cluster 非空
            self.summary = VEB(self._su)

    def _high(self, x): return x // self._su
    def _low(self, x):  return x % self._su
    def _index(self, h, l): return h * self._su + l

    def minimum(self): return self.min_val
    def maximum(self): return self.max_val

    def member(self, x) -> bool:
        if x == self.min_val or x == self.max_val:
            return True
        if self.u == 2:
            return False
        return self.cluster[self._high(x)].member(self._low(x))

    def successor(self, x):
        if self.u == 2:
            if x == 0 and self.max_val == 1: return 1
            return None
        if self.min_val is not None and x < self.min_val:
            return self.min_val
        h = self._high(x)
        maxLow = self.cluster[h].maximum()
        if maxLow is not None and self._low(x) < maxLow:
            offset = self.cluster[h].successor(self._low(x))
            return self._index(h, offset)
        succCluster = self.summary.successor(h)
        if succCluster is None: return None
        return self._index(succCluster, self.cluster[succCluster].minimum())

    def insert(self, x):
        if self.min_val is None:
            self.min_val = self.max_val = x
            return
        if x < self.min_val:
            x, self.min_val = self.min_val, x  # 新 min 直接存
        if self.u > 2:
            h, l = self._high(x), self._low(x)
            if self.cluster[h].minimum() is None:
                self.summary.insert(h)  # cluster h 从空变非空，更新 summary
                self.cluster[h].min_val = self.cluster[h].max_val = l
            else:
                self.cluster[h].insert(l)
        if x > self.max_val:
            self.max_val = x

    def delete(self, x):
        if self.min_val == self.max_val:
            self.min_val = self.max_val = None
            return
        if self.u == 2:
            self.min_val = self.max_val = 1 - x
            return
        if x == self.min_val:
            first = self.summary.minimum()
            x = self._index(first, self.cluster[first].minimum())
            self.min_val = x
        h, l = self._high(x), self._low(x)
        self.cluster[h].delete(l)
        if self.cluster[h].minimum() is None:
            self.summary.delete(h)  # cluster h 变空，更新 summary
            if x == self.max_val:
                sm = self.summary.maximum()
                self.max_val = self.min_val if sm is None else self._index(sm, self.cluster[sm].maximum())
        elif x == self.max_val:
            self.max_val = self._index(h, self.cluster[h].maximum())
```

```cpp
#include <bits/stdc++.h>
using namespace std;

// 注意：此为教学用简化版，u 须为 2 的幂次，空间 O(u)
struct VEB {
    int u, su;            // 全集大小，子全集大小 sqrt(u)
    int mn, mx;           // -1 表示空
    vector<VEB*> cluster;
    VEB* summary;

    VEB(int _u) : u(_u), su(0), mn(-1), mx(-1), summary(nullptr) {
        if (u > 2) {
            su = (int)sqrt((double)u);
            cluster.resize(su);
            for (int i = 0; i < su; i++) cluster[i] = new VEB(su);
            summary = new VEB(su);
        }
    }

    int high(int x) { return x / su; }
    int low(int x)  { return x % su; }
    int idx(int h, int l) { return h * su + l; }

    int minimum() { return mn; }
    int maximum() { return mx; }

    bool member(int x) {
        if (x == mn || x == mx) return true;
        if (u == 2) return false;
        return cluster[high(x)]->member(low(x));
    }

    // 插入 x
    void insert(int x) {
        if (mn == -1) { mn = mx = x; return; }
        if (x < mn) swap(x, mn);  // 新 min 直接存
        if (u > 2) {
            int h = high(x), l = low(x);
            if (cluster[h]->minimum() == -1) {
                summary->insert(h);
                cluster[h]->mn = cluster[h]->mx = l;
            } else {
                cluster[h]->insert(l);
            }
        }
        if (x > mx) mx = x;
    }

    // 后继
    int successor(int x) {
        if (u == 2) { return (x == 0 && mx == 1) ? 1 : -1; }
        if (mn != -1 && x < mn) return mn;
        int h = high(x), l = low(x);
        int maxLow = cluster[h]->maximum();
        if (maxLow != -1 && l < maxLow) {
            int off = cluster[h]->successor(l);
            return idx(h, off);
        }
        int sc = summary->successor(h);
        if (sc == -1) return -1;
        return idx(sc, cluster[sc]->minimum());
    }
};
```

---

## 38.3 分块（Sqrt Decomposition）与莫队算法（Mo's Algorithm）

### 38.3.1 "根号万能"：无需高级树结构的权衡方案

**核心思想**：将大小为 $n$ 的数组平均分成 $B = \sqrt{n}$ 个块，每块大小约 $\sqrt{n}$。对于操作：

- **完整块**：作为整体处理，共 $\sqrt{n}$ 个，每个 $O(1)$ 或 $O(B)$，合计 $O(\sqrt{n})$
- **边缘块**（两端不完整的块）：逐元素处理，最多 $2B = 2\sqrt{n}$ 个元素，$O(\sqrt{n})$

> **直觉比喻**：把一条 100 米的路分成 10 段，每段 10 米。要查询第 35~72 米有多少棵树：完整段（40~69 米，3 段）直接查块统计；边缘（35~39 和 70~72 米）逐棵数。共需 3 + 5 + 3 = 11 次，远少于 38 次。

**为什么选 $B = \sqrt{n}$**：

若每块大小为 $B$，共 $n/B$ 块：
- 更新一个元素：最多修改 1 个块统计，$O(1)$
- 查询区间 $[l, r]$：最多 $n/B$ 个完整块 + $2B$ 个边缘元素

查询复杂度 $= O(n/B + B)$，对 $B$ 求导令等于 0：$-n/B^2 + 1 = 0$，得 $B = \sqrt{n}$，此时复杂度为 $O(\sqrt{n})$。

### 38.3.2 区间求和：分块实现

```python
import math

class SqrtDecomposition:
    """
    分块数据结构：支持单点更新 O(1) + 区间查询 O(√n)
    也可扩展为区间修改 O(√n) + 区间查询 O(√n)
    """
    def __init__(self, arr: list[int]):
        self.n = len(arr)
        self.arr = arr[:]
        self.block_size = int(math.isqrt(self.n)) + 1  # 块大小 ≈ √n
        # 块的数量
        self.num_blocks = (self.n + self.block_size - 1) // self.block_size
        # block_sum[i] = 第 i 个块的所有元素之和
        self.block_sum = [0] * self.num_blocks

        # 预处理：计算每个块的和
        for i, x in enumerate(arr):
            self.block_sum[i // self.block_size] += x

    def update(self, pos: int, val: int):
        """单点更新：O(1)"""
        self.block_sum[pos // self.block_size] += val - self.arr[pos]
        self.arr[pos] = val

    def query(self, l: int, r: int) -> int:
        """区间求和 arr[l..r]（含两端）：O(√n)"""
        result = 0
        bl = l // self.block_size  # l 所在块号
        br = r // self.block_size  # r 所在块号

        if bl == br:
            # 同一块内：逐元素求和
            result = sum(self.arr[l:r + 1])
        else:
            # 左边缘（不完整块）：逐元素
            result += sum(self.arr[l: (bl + 1) * self.block_size])
            # 完整块：直接累加块和
            for b in range(bl + 1, br):
                result += self.block_sum[b]
            # 右边缘（不完整块）：逐元素
            result += sum(self.arr[br * self.block_size: r + 1])

        return result
```

```cpp
#include <bits/stdc++.h>
using namespace std;

class SqrtDecompose {
    vector<int> arr;
    vector<long long> blk;  // 块和
    int n, B;               // 数组长度、块大小

public:
    SqrtDecompose(vector<int>& a) : arr(a), n(a.size()) {
        B = (int)sqrt((double)n) + 1;
        blk.assign((n + B - 1) / B, 0);
        for (int i = 0; i < n; i++)
            blk[i / B] += a[i];
    }

    // 单点更新 O(1)
    void update(int pos, int val) {
        blk[pos / B] += val - arr[pos];
        arr[pos] = val;
    }

    // 区间求和 [l, r]，O(√n)
    long long query(int l, int r) {
        long long res = 0;
        int bl = l / B, br = r / B;
        if (bl == br) {
            for (int i = l; i <= r; i++) res += arr[i];
        } else {
            // 左边缘
            for (int i = l; i < (bl + 1) * B; i++) res += arr[i];
            // 完整块
            for (int b = bl + 1; b < br; b++) res += blk[b];
            // 右边缘
            for (int i = br * B; i <= r; i++) res += arr[i];
        }
        return res;
    }
};
```

### 38.3.3 区间修改 + 区间查询（懒标记分块）

当需要「对区间 $[l, r]$ 加 $v$」时，可以给**完整块**打懒标记，边缘块展开再修改：

```python
class SqrtDecomposeWithLazy:
    """
    分块：支持区间加法修改 O(√n) + 区间求和 O(√n)
    """
    def __init__(self, arr: list[int]):
        self.n = len(arr)
        self.arr = arr[:]
        self.B = int(math.isqrt(self.n)) + 1
        self.num_blocks = (self.n + self.B - 1) // self.B
        self.block_sum = [0] * self.num_blocks
        self.lazy = [0] * self.num_blocks  # 每个块的懒标记
        for i, x in enumerate(arr):
            self.block_sum[i // self.B] += x

    def range_add(self, l: int, r: int, v: int):
        """区间 [l, r] 每个元素加 v：O(√n)"""
        bl, br = l // self.B, r // self.B
        if bl == br:
            # 同块：展开修改，更新块和
            for i in range(l, r + 1):
                self.arr[i] += v
            self.block_sum[bl] += v * (r - l + 1)
        else:
            # 左边缘：展开
            for i in range(l, (bl + 1) * self.B):
                self.arr[i] += v
            self.block_sum[bl] += v * ((bl + 1) * self.B - l)
            # 完整块：打懒标记
            for b in range(bl + 1, br):
                self.lazy[b] += v
                self.block_sum[b] += v * self.B
            # 右边缘：展开
            for i in range(br * self.B, r + 1):
                self.arr[i] += v
            self.block_sum[br] += v * (r - br * self.B + 1)

    def range_sum(self, l: int, r: int) -> int:
        """区间 [l, r] 求和：O(√n)"""
        bl, br = l // self.B, r // self.B
        res = 0
        if bl == br:
            res = sum(self.arr[l:r + 1]) + self.lazy[bl] * (r - l + 1)
        else:
            for i in range(l, (bl + 1) * self.B):
                res += self.arr[i] + self.lazy[bl]
            for b in range(bl + 1, br):
                res += self.block_sum[b]  # block_sum 已包含 lazy
            for i in range(br * self.B, r + 1):
                res += self.arr[i] + self.lazy[br]
        return res
```

<div data-component="SqrtDecompositionBlock"></div>

### 38.3.4 分块的适用场景

分块（根号分治）的优势：
- **$O(\sqrt{n})$ 够用**：$n = 10^6$ 时 $\sqrt{n} \approx 1000$，总操作 $O(q\sqrt{n})$ 约 $10^9$，可能刚好通过，适合没有线段树的情形（如区间众数、区间 $k$ 小值）。
- **实现简单干净**：不需要理解 segment tree 的懒标记细节。
- **区间众数**：分块可以做到 $O(n\sqrt{n})$ 预处理 + $O(\sqrt{n})$ 查询（离线），而线段树无法高效维护众数。

### 38.3.5 莫队算法（Mo's Algorithm）：离线区间查询的 O(n√n) 技巧

**问题背景**：给定 $q$ 个区间查询 $(l_i, r_i)$，每次暴力遍历是 $O(qn)$，太慢。

**核心思想**：将所有查询**离线排序**，使得查询区间的端点移动总次数最小。

**排序规则**：
- 按左端点所在**块号**为第一键排序（块号 = $l_i / \sqrt{n}$）
- 同一块内，按右端点 $r_i$ 排序（奇数块递增，偶数块递减，可节省一半移动）

**维护双指针**：用 `[curL, curR]` 表示当前区间，每次查询通过逐步扩展/收缩端点来转移：

```python
import math

def mo_algorithm(arr: list[int], queries: list[tuple]) -> list:
    """
    莫队算法：离线处理区间查询。
    queries: [(l, r, idx)] 列表，idx 是原始查询的编号
    返回：每个查询的结果列表
    """
    n = len(arr)
    B = max(1, int(math.isqrt(n)))  # 块大小

    # 按块号+�奇偶优化排序
    def key(q):
        l, r, _ = q
        bl = l // B
        return (bl, r if bl % 2 == 0 else -r)

    queries_sorted = sorted(enumerate(queries), key=lambda x: key(x[1]))

    # 当前维护的区间 [curL, curR]
    curL, curR = 0, -1
    cnt = {}      # 统计数组（视查询类型而定，这里统计频率）
    cur_ans = 0   # 当前答案（视具体操作定义）
    answers = [0] * len(queries)

    def add(pos):
        nonlocal cur_ans
        val = arr[pos]
        cnt[val] = cnt.get(val, 0) + 1
        # 例：统计不同元素个数
        if cnt[val] == 1:
            cur_ans += 1

    def remove(pos):
        nonlocal cur_ans
        val = arr[pos]
        cnt[val] -= 1
        if cnt[val] == 0:
            cur_ans -= 1
            del cnt[val]

    for orig_idx, (l, r, _) in queries_sorted:
        # 扩展/收缩右端点
        while curR < r: curR += 1; add(curR)
        while curR > r: remove(curR); curR -= 1
        # 扩展/收缩左端点
        while curL > l: curL -= 1; add(curL)
        while curL < l: remove(curL); curL += 1

        answers[orig_idx] = cur_ans

    return answers
```

```cpp
#include <bits/stdc++.h>
using namespace std;

// 莫队算法：统计区间不同元素个数
struct Query { int l, r, idx; };

int arr[100005], cnt[100005];
int cur_ans = 0, n, B;

void add(int pos) { if (++cnt[arr[pos]] == 1) cur_ans++; }
void rem(int pos) { if (--cnt[arr[pos]] == 0) cur_ans--; }

vector<int> mo_algorithm(vector<int>& a, vector<pair<int,int>>& qs) {
    n = a.size();
    B = max(1, (int)sqrt(n));
    for (int i = 0; i < n; i++) arr[i] = a[i];

    int q = qs.size();
    vector<Query> queries(q);
    for (int i = 0; i < q; i++) queries[i] = {qs[i].first, qs[i].second, i};

    // 排序：块号为主键，奇偶块决定右端点排序方向
    sort(queries.begin(), queries.end(), [](const Query& a, const Query& b) {
        int ba = a.l / B, bb = b.l / B;
        if (ba != bb) return ba < bb;
        return (ba & 1) ? (a.r > b.r) : (a.r < b.r);
    });

    vector<int> ans(q);
    int curL = 0, curR = -1;
    cur_ans = 0;

    for (auto& query : queries) {
        while (curR < query.r) add(++curR);
        while (curR > query.r) rem(curR--);
        while (curL > query.l) add(--curL);
        while (curL < query.l) rem(curL++);
        ans[query.idx] = cur_ans;
    }
    return ans;
}
```

**复杂度分析**：

- 共 $n/B$ 个块，每块内右端点单调，右端点总移动次数 $O(n)$ per 块 × $n/B$ 块 $= O(n^2/B)$。
- 每个块内左端点移动最多 $2B$，$q$ 个查询共 $O(qB)$。
- 总移动：$O(n^2/B + qB)$，令 $B = n/\sqrt{q}$，得 $O(n\sqrt{q})$；若 $q \approx n$，即 $O(n\sqrt{n})$。

<div data-component="MoAlgorithmTrace"></div>

---

## 38.4 K-D 树（KD-Tree）

### 38.4.1 从一维延伸到多维：空间分割的直觉

一维查找很简单：将数轴在中位数处一分为二，左边放小值，右边放大值，得到二叉搜索树，查找 $O(\log n)$。

**KD-Tree（K-Dimensional Tree）** 将这一思想推广到 $d$ 维空间：每一层选定一个维度，按该维度的**中位数**将点集一分为二，递归建树。

> **生活比喻**：把一张地图（2D）不断对折——第一刀横切（按纬度），第二刀竖切（按经度），第三刀横切……切到最后每个格子里只有 1 个城市。找附近城市时，沿着这张折叠地图展开搜索，很多"明显太远"的区域根本不用展开。

### 38.4.2 建树（BUILD-KD-TREE）

**策略**：每层轮流按第 $k \mod d$ 维（$k$ 为树高）选中位数分割。

```
BUILD-KD-TREE(points, depth):
    if points is empty: return NIL
    axis = depth % d                    // 轮换维度
    sorted = sort(points by axis)
    mid = len(sorted) // 2             // 中位数下标
    node = KDNode(sorted[mid], axis)   // 当前节点
    node.left  = BUILD-KD-TREE(sorted[:mid],   depth + 1)
    node.right = BUILD-KD-TREE(sorted[mid+1:], depth + 1)
    return node
```

**时间复杂度**：$O(n \log n)$（每层用线性时间选中位数，或排序后取中位，共 $O(\log n)$ 层）。  
**空间**：$O(n)$（每个点存一次）。

### 38.4.3 最近邻搜索（Nearest Neighbor Search）

**算法**：从根出发，先访问"可能最近"的子树，再回溯检查是否需要访问另一子树（用超矩形与当前最近距离对比来剪枝）。

**步骤**：
1. 计算当前节点到查询点 $q$ 的距离，更新最近距离 `best`。
2. 按分割轴比较：$q$ 在分割线的哪侧，先递归"近侧"子树。
3. 回溯：若"远侧"超矩形与以 $q$ 为圆心、`best` 为半径的球**有交**，则也递归"远侧"。

```python
import heapq
from typing import Optional

class KDNode:
    def __init__(self, point: list[float], axis: int):
        self.point = point
        self.axis = axis
        self.left: Optional['KDNode'] = None
        self.right: Optional['KDNode'] = None

class KDTree:
    def __init__(self, points: list[list[float]]):
        self.d = len(points[0]) if points else 0
        self.root = self._build(points, depth=0)

    def _build(self, points: list, depth: int) -> Optional[KDNode]:
        if not points:
            return None
        axis = depth % self.d
        # 按当前维度排序，取中位数
        points.sort(key=lambda p: p[axis])
        mid = len(points) // 2
        node = KDNode(points[mid], axis)
        node.left  = self._build(points[:mid],     depth + 1)
        node.right = self._build(points[mid + 1:], depth + 1)
        return node

    def _dist_sq(self, a: list, b: list) -> float:
        """欧几里得距离平方（避免开方，提升性能）"""
        return sum((x - y) ** 2 for x, y in zip(a, b))

    def nearest_neighbor(self, query: list[float]) -> tuple:
        """返回 (最近距离平方, 最近点)"""
        best = [float('inf'), None]

        def search(node: Optional[KDNode]):
            if node is None:
                return
            d_sq = self._dist_sq(node.point, query)
            if d_sq < best[0]:
                best[0], best[1] = d_sq, node.point

            axis = node.axis
            diff = query[axis] - node.point[axis]

            # 先递归"近侧"
            near, far = (node.left, node.right) if diff <= 0 else (node.right, node.left)
            search(near)
            # 仅当"远侧"超矩形可能更近，才递归
            if diff * diff < best[0]:
                search(far)

        search(self.root)
        return best[0] ** 0.5, best[1]

    def range_search(self, query: list[float], radius: float) -> list:
        """返回距 query 距离 ≤ radius 的所有点"""
        result = []
        r2 = radius * radius

        def search(node: Optional[KDNode]):
            if node is None:
                return
            if self._dist_sq(node.point, query) <= r2:
                result.append(node.point)
            axis = node.axis
            diff = query[axis] - node.point[axis]
            # 两侧都可能有结果（diff*diff < r2 才有必要查远侧）
            if diff <= 0 or diff * diff <= r2:
                search(node.left)
            if diff >= 0 or diff * diff <= r2:
                search(node.right)

        search(self.root)
        return result
```

```cpp
#include <bits/stdc++.h>
using namespace std;

struct KDNode {
    vector<double> point;
    int axis;
    KDNode *left = nullptr, *right = nullptr;
    KDNode(vector<double> p, int a) : point(p), axis(a) {}
};

class KDTree {
    int d;
    KDNode* root;

    double distSq(const vector<double>& a, const vector<double>& b) {
        double s = 0;
        for (int i = 0; i < d; i++) s += (a[i] - b[i]) * (a[i] - b[i]);
        return s;
    }

    KDNode* build(vector<vector<double>>& pts, int l, int r, int depth) {
        if (l > r) return nullptr;
        int axis = depth % d;
        sort(pts.begin() + l, pts.begin() + r + 1,
             [axis](auto& a, auto& b) { return a[axis] < b[axis]; });
        int mid = (l + r) / 2;
        auto* node = new KDNode(pts[mid], axis);
        node->left  = build(pts, l, mid - 1, depth + 1);
        node->right = build(pts, mid + 1, r, depth + 1);
        return node;
    }

    void searchNN(KDNode* node, const vector<double>& q,
                  double& bestDist, vector<double>& bestPt) {
        if (!node) return;
        double d2 = distSq(node->point, q);
        if (d2 < bestDist) { bestDist = d2; bestPt = node->point; }

        double diff = q[node->axis] - node->point[node->axis];
        KDNode *near = diff <= 0 ? node->left : node->right;
        KDNode *far  = diff <= 0 ? node->right : node->left;
        searchNN(near, q, bestDist, bestPt);
        if (diff * diff < bestDist)
            searchNN(far, q, bestDist, bestPt);
    }

public:
    KDTree(vector<vector<double>> pts) : d(pts[0].size()) {
        root = build(pts, 0, pts.size() - 1, 0);
    }

    pair<double, vector<double>> nearestNeighbor(const vector<double>& q) {
        double bd = 1e18; vector<double> bp;
        searchNN(root, q, bd, bp);
        return {sqrt(bd), bp};
    }
};
```

<div data-component="KDTreeBuildQuery"></div>

### 38.4.4 范围查询（Range Search）

查找半径 $r$ 内所有点（球形范围）：

- 时间复杂度：$O(n^{1-1/d} + k)$，其中 $d$ 是维度，$k$ 是结果数量。
- 对 $d=2$：期望 $O(\sqrt{n} + k)$，实践中非常高效。
- 对 $d=3, 4, \ldots$：效果逐渐退化（高维诅咒）。

### 38.4.5 高维诅咒与 KD-Tree 的适用范围

**高维诅咒（Curse of Dimensionality）**：当维度 $d$ 很大时，KD-Tree 几乎退化为线性扫描，原因如下：

> 在高维空间中，以任意点为中心的球，其体积相对整个超矩形趋近于 0。经过几次分割后，几乎所有叶子节点对应的超矩形都与查询球有交，导致剪枝几乎无效。

**经验法则**：维度 $d > 10$\~$20$ 时，KD-Tree 效果急剧退化。实践建议：
- $d \leq 3$：KD-Tree 表现优秀。
- $4 \leq d \leq 15$：需要 benchmark，考虑用 Ball-Tree 或 Cover-Tree。
- $d > 20$：ANN（近似最近邻）方法，如 HNSW、IVF、LSH（局部敏感哈希）。

<div data-component="KDTreeCurseOfDimensionality"></div>

### 38.4.6 工程应用

| 应用场景 | 使用的维度 | 用途 |
|----------|------------|------|
| GIS 地图近邻（Google Maps） | 2D | 找周边 POI |
| 游戏碰撞检测 | 2D / 3D | 快速找附近物体 |
| 点云处理（自动驾驶 LiDAR） | 3D | 地面分割、目标检测 |
| 图像压缩（K-均值） | 高维 | 颜色量化 |
| 机器人路径规划（RRT） | 6D 关节空间 | 最近节点查找 |

---

## 38.5 本章总结：场景选型指南

| 数据结构 | 最佳适用场景 | 时间复杂度 | 实现难度 |
|----------|------------|------------|----------|
| **跳表** | 有序动态集合、并发场景（Redis ZSet） | 期望 $O(\log n)$ | ⭐⭐ |
| **vEB Tree** | 有界整数全集上的优先队列，$u$ 不太大 | $O(\log \log u)$ | ⭐⭐⭐⭐ |
| **分块** | 线段树太复杂（区间众数、区间第 k 小离线）、$O(\sqrt{n})$ 够用 | $O(\sqrt{n})$ | ⭐ |
| **莫队** | 离线区间查询，转移容易定义 | $O(n\sqrt{q})$ | ⭐⭐ |
| **KD-Tree** | 低维（$d \leq 10$）近邻搜索或范围查询 | 期望 $O(\log n)$（低维）| ⭐⭐⭐ |

> **选型口诀**：有序集合用跳表，整数快查上 vEB，复杂操作根号打，多维近邻 KD-Tree 拉。

---

## 常见错误与陷阱

> ⚠️ **跳表随机性陷阱**：使用默认的 `random.random()` 在某些语言中性能较差，竞赛中可考虑用位运算加速。另外，跳表的最坏情况是 $O(n)$（所有元素晋升到最高层），不像红黑树有确定性的最坏保证。

> ⚠️ **vEB 树下标运算**：`high(x) = x // sqrt(u)` 和 `low(x) = x % sqrt(u)` 要确保 $u$ 是完全平方数（或用 $u = 2^{2k}$ 保证 $\sqrt{u} = 2^k$ 是整数）。

> ⚠️ **分块边界 off-by-one**：左边缘是 `[l, (bl+1)*B - 1]`，右边缘是 `[br*B, r]`，务必确认端点闭合/开放是否与实现一致。

> ⚠️ **KD-Tree 剪枝条件**：应比较的是**分割轴上的距离平方** `diff * diff` 与当前**最近距离平方** `bestDist`，而不是直接用欧几里得距离，避免浮点开方的性能开销和精度问题。

> ⚠️ **莫队离线限制**：莫队需要提前知道所有查询，不支持在线（带修改的版本称为"带修莫队"，复杂度升至 $O(n^{5/3})$）。

---

## 面试高频考点

1. **跳表实现（LeetCode #1206）**：手写跳表的 `search`、`add`、`erase` 三个操作。
2. **跳表 vs 红黑树**：随机性 vs 确定性、并发友好性、Redis 的选择原因。
3. **vEB 树的递归关系**：解释为什么操作是 $O(\log \log u)$（子问题大小变为 $\sqrt{u}$）。
4. **分块与线段树的权衡**：什么时候用分块（区间众数、多操作复杂、时限宽松）。
5. **莫队排序规则**：为什么按块号+右端点排序，时间复杂度如何推导。
6. **KD-Tree 剪枝**：最近邻搜索的回溯条件（超矩形到查询点的最短距离 < 当前最近距离）。
7. **高维诅咒**：举例说明 $d=10$ 时 KD-Tree 为何退化，应该用什么替代方案（HNSW/LSH）。

---

## 思考题

💡 **思考 1（跳表确定性类比）**：如果将跳表的"随机晋升"改为"每隔 $2^k$ 个元素在第 $k$ 层放一个节点"（确定性晋升），你会得到什么结构？它与 B-Tree 有什么相似之处？（提示：想想 B-Tree 叶子层到根层的结构。）

💡 **思考 2（vEB 树空间优化）**：朴素 vEB 树空间是 $O(u)$，若 $u = 2^{32}$ 则内存爆炸。如何使用哈希表（`unordered_map`）只存储非空子簇，将空间降到 $O(n)$？这会带来什么副作用？

💡 **思考 3（莫队带修改版）**：标准莫队不支持在线修改。若允许单点修改，可以引入第三个维度（时间轴），对块大小取 $n^{2/3}$，得到"带修莫队"。你能分析其时间复杂度吗？

💡 **思考 4（KD-Tree 动态插入）**：KD-Tree 经典实现是静态的（一次性建树），若频繁插入新点，树会退化。你会怎么维护树的平衡？（提示：考虑"部分重建"策略，或使用 kd-tree 的自平衡变体如 scapegoat KD-tree。）

---

## 扩展阅读

- **CLRS 第4版** Chapter 20：van Emde Boas 树（完整证明）
- **Skiena 第2版** Chapter 3：基本数据结构，跳表分析
- **Pugh 1990 原始论文**：*Skip Lists: A Probabilistic Alternative to Balanced Trees*
- **LeetCode #1206**：设计跳表（推荐动手实现）
- **莫队算法详解**：[CF 官方 Educational blog](https://codeforces.com/blog/entry/61203)
- **近似最近邻（ANN）**：了解 HNSW、FAISS 等工程级工具

---

*Chapter 38 完 · 下一章：Chapter 39 计算几何核心算法*

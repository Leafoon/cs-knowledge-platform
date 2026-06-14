# Chapter 11: B 树与 B+ 树

> **学习目标**：理解磁盘 I/O 是数据库性能的真正瓶颈，掌握 B 树和 B+ 树如何通过增大树的"扇出"显著减少磁盘访问次数；掌握 B 树的分裂/合并操作机制；深刻理解 B+ 树叶链表对范围查询的关键作用；能从 MySQL InnoDB 的视角理解聚簇索引与辅助索引。

---

## 11.1 B 树的动机：向磁盘 I/O 要效率

### 11.1.1 一个让人头疼的事实：内存与磁盘的速度鸿沟

在前两章，我们讨论了 AVL 树和红黑树，它们将树高控制在 $O(\log n)$，让查找速度达到了理论上的最优。然而，当数据量达到数亿条、无法全部放入内存时，树节点需要从**磁盘**中读取，情况就发生了根本变化：

| 存储层级 | 访问延迟 | 相对速度 |
|---------|---------|---------|
| CPU 寄存器 | 0.3 ns | 基准 |
| L1 Cache | 1 ns | ×3 |
| L2 Cache | 4 ns | ×13 |
| DRAM 内存 | 100 ns | ×300 |
| NVMe SSD | 100 μs | ×300,000 |
| HDD 磁盘 | 10 ms | **×30,000,000** |

这意味着：**从磁盘读取 1 次的时间，CPU 能执行 3000 万条指令**。内存中访问 1000 个节点的 AVL 树，其时间代价还不如一次磁盘 I/O 的零头。

因此，在磁盘存储场景下，**树的高度（= 磁盘 I/O 次数）才是真正的性能瓶颈**，而树节点内的比较次数反而是次要的。

### 11.1.2 磁盘 I/O 的工作方式

磁盘以**页（Page）**或**块（Block）**为单位进行读写，典型大小为 4KB 或 16KB（MySQL InnoDB 默认 16KB）。一次磁盘 I/O 无论你读 1 字节还是 16KB，消耗的时间几乎相同。

这给了我们一个关键洞察：**既然一次 I/O 可以读整页，为什么不把这 16KB 塞满有用的信息？**

对于一棵二叉树：
- 每个节点: 8 字节 key + 2 × 8 字节指针 = 24 字节
- 一页 16KB 能装 **682 个二叉节点**，但查找一个节点只用其中一个，**利用率约 1/682 ≈ 0.15%**

对于多路树（每节点 1000 个键）：
- 每个节点恰好占一页，**利用率接近 100%**
- 1000 个子指针 → **树高从 $\log_2 n$ 降到 $\log_{1000} n$**

对于 $n = 10^9$ 条记录：
- 平衡 BST 高度 $\approx \log_2(10^9) \approx 30$ 层
- 多路树（1000 路）高度 $\approx \log_{1000}(10^9) = 3$ 层

**从 30 次磁盘 I/O 降到 3 次——这就是 B 树被发明的动力！**

下面的可视化展示了不同数据量和度数 t 下磁盘 I/O 次数的对比：

<div data-component="DiskIOComparison"></div>

### 11.1.3 B 树的最小度数参数 t

**B 树的最小度数（Minimum Degree）** $t$（$t \geq 2$）是 B 树的核心参数，决定了节点键数的范围：

| 属性 | 约束 |
|------|------|
| 每个非根节点的键数 | $[t-1, \ 2t-1]$ |
| 根节点的键数 | $[1, \ 2t-1]$（树非空时） |
| 每个内部节点的子节点数 | $[\,t, \ 2t\,]$ |
| 根节点的子节点数 | $[2, \ 2t]$（非叶时） |

**直觉推导**：
- 下界 $t-1$ 保证节点不会"太空"（太空则不如合并到兄弟）
- 上界 $2t-1$ 是下界的 2 倍（保证插入分裂后两半均合法）
- 满节点有 $2t-1$ 个键和 $2t$ 个子指针

**工程中 t 的选取**：磁盘块大小 $B$ 字节，键大小 $k$ 字节，指针大小 $p$ 字节：

$$t \approx \left\lfloor \frac{B - p}{k + p} \right\rfloor + 1$$

例如 MySQL InnoDB 16KB 页，整数主键 8B，指针 8B：$t \approx \lfloor (16384 - 8) / (8 + 8) \rfloor + 1 \approx 1024$。树高仅 2–3 层即可支撑数亿记录！

### 11.1.4 B 树高度上界证明

**定理**：含 $n$ 个键（内部节点）的 B 树，高度 $h$ 满足：

$$h \leq \log_t \left\lceil \frac{n+1}{2} \right\rceil$$

**证明**：考虑高度为 $h$ 的 B 树中键数的下界。
- 根至少 1 个键 → 至少 2 个孩子
- 第 2 层：每个节点至少 $t-1$ 个键，至少 $t$ 个孩子
- 第 $i$ 层（$i \geq 2$）：至少 $2t^{i-2}$ 个节点，每节点至少 $t-1$ 个键

$$n \geq 1 + (t-1)\sum_{i=1}^{h} 2t^{i-1} = 1 + 2(t-1) \cdot \frac{t^h - 1}{t - 1} = 2t^h - 1$$

解得 $t^h \leq (n+1)/2$，即 $h \leq \log_t \lceil (n+1)/2 \rceil$。

对 $n = 10^6, t = 1000$：$h \leq \log_{1000}(500001) \approx 2.0$。B 树仅需 **2 层**！

---

## 11.2 B 树操作

B 树的节点结构：每个节点包含若干键和子指针，键将子树的键值范围分隔。设节点有 $k$ 个键 $K_1 < K_2 < \cdots < K_k$，则有 $k+1$ 个子树，第 $i$ 个子树包含所有介于 $K_{i-1}$ 和 $K_i$ 之间的键（边界约定为开/闭区间）。

### 11.2.1 SEARCH（查找）

与 BST 查找类似，但每层需要在节点内二分查找目标键：

```
B-TREE-SEARCH(node, key):
    i = 1
    while i ≤ node.n and key > node.keys[i]:
        i++
    if i ≤ node.n and key == node.keys[i]:
        return (node, i)          // 找到
    if node.is_leaf:
        return NOT_FOUND          // 叶节点未找到
    DISK_READ(node.children[i])   // 读磁盘
    return B-TREE-SEARCH(node.children[i], key)
```

**复杂度**：$O(t \log_t n)$
- 磁盘 I/O：$O(\log_t n)$（每层一次）
- CPU 比较：每层 $O(t)$ 次（线性查找）或 $O(\log t)$（二分）

```python
class BNode:
    def __init__(self, leaf=False):
        self.keys = []      # 有序键列表
        self.children = []  # 子节点引用（内部节点）
        self.leaf = leaf

class BTree:
    def __init__(self, t: int):
        self.t = t          # 最小度数
        self.root = BNode(leaf=True)

    def search(self, node: BNode, key: int):
        """在以 node 为根的子树中查找 key，返回 (节点, 索引) 或 None"""
        i = 0
        # 找到第一个 >= key 的位置
        while i < len(node.keys) and key > node.keys[i]:
            i += 1
        # 检查是否精确匹配
        if i < len(node.keys) and key == node.keys[i]:
            return (node, i)
        # 叶节点未找到
        if node.leaf:
            return None
        # 下降到第 i 个孩子（磁盘中模拟为直接访问）
        return self.search(node.children[i], key)
```
```cpp
#include <vector>
#include <optional>
using namespace std;

struct BNode {
    vector<int> keys;
    vector<BNode*> children;
    bool leaf;
    BNode(bool l = false) : leaf(l) {}
};

class BTree {
public:
    int t;          // 最小度数
    BNode* root;

    BTree(int t) : t(t) { root = new BNode(true); }

    // 返回 {节点指针, 键在节点中的索引}，未找到返回 {nullptr, -1}
    pair<BNode*, int> search(BNode* node, int key) {
        int i = 0;
        // 找第一个 >= key 的位置
        while (i < (int)node->keys.size() && key > node->keys[i]) i++;
        // 精确匹配
        if (i < (int)node->keys.size() && key == node->keys[i])
            return {node, i};
        // 叶节点未找到
        if (node->leaf) return {nullptr, -1};
        // 递归进入子节点（实际中这里是磁盘 I/O）
        return search(node->children[i], key);
    }
};
```

### 11.2.2 INSERT（插入）：提前分裂策略

B 树插入的关键挑战：从根向下搜索到叶子时，如果叶子已满（$2t-1$ 个键），需要分裂，分裂会向父节点上浮一个键，父节点可能也满……这种向上传播最坏需要 $O(h)$ 次磁盘写入，且得先找到分裂位置再返回，实现复杂。

**CLRS 的解决方案：提前分裂（Proactive Splitting）**

在向下搜索路径时，**每遇到满节点就立即分裂**，确保到达叶子时路径上所有节点都有插入空间，从而只需一趟下行即可完成插入。

**SPLIT-CHILD 操作**：将满孩子 $y$（$2t-1$ 个键）分裂为两个各含 $t-1$ 个键的节点，中间键上浮到父节点 $x$：

```
SPLIT-CHILD(x, i):
    // x = 父节点，i = 满孩子的下标
    y = x.children[i]        // 满孩子（2t-1 个键）
    z = new BNode()           // 新节点接收 y 的右半部分
    z.leaf = y.leaf
    z.keys = y.keys[t:]       // y 的后半 t-1 个键给 z
    y.keys = y.keys[:t-1]     // y 保留前半 t-1 个键
    if not y.leaf:
        z.children = y.children[t:]   // 孩子指针也要分
        y.children = y.children[:t]
    x.keys.insert(i, y.keys[t-1])    // y 的第 t 个键（中间键）上浮
    x.children.insert(i+1, z)         // z 成为 x 的第 i+1 个孩子
    DISK_WRITE(y); DISK_WRITE(z); DISK_WRITE(x)
```

下面的动画展示了分裂、合并与借键旋转三种操作的完整过程：

<div data-component="BTreeSplitMerge"></div>

```python
def _split_child(self, parent: BNode, i: int):
    """将 parent.children[i]（满节点）从中间分裂，中间键上浮到 parent"""
    t = self.t
    y = parent.children[i]     # 满孩子，含 2t-1 个键
    z = BNode(leaf=y.leaf)     # 新节点接收右半部分

    mid = y.keys[t - 1]        # 中间键将上浮
    z.keys = y.keys[t:]        # z 得到右半 t-1 个键
    y.keys = y.keys[:t - 1]    # y 保留左半 t-1 个键

    if not y.leaf:
        z.children = y.children[t:]   # z 得到右半 t 个孩子
        y.children = y.children[:t]   # y 保留左半 t 个孩子

    # 中间键上浮，插入 parent 的第 i 位
    parent.keys.insert(i, mid)
    parent.children.insert(i + 1, z)

def _insert_nonfull(self, node: BNode, key: int):
    """向非满节点为根的子树插入 key（保证路径上不经过满节点）"""
    i = len(node.keys) - 1
    if node.leaf:
        # 叶节点：直接插入保持有序
        while i >= 0 and key < node.keys[i]:
            i -= 1
        node.keys.insert(i + 1, key)
    else:
        # 找到应该下降的孩子下标
        while i >= 0 and key < node.keys[i]:
            i -= 1
        i += 1      # i 是子节点下标
        if len(node.children[i].keys) == 2 * self.t - 1:
            # 孩子满了，提前分裂
            self._split_child(node, i)
            # 分裂后 node.keys[i] 是新上浮的中间键
            if key > node.keys[i]:
                i += 1   # 去右半子树
        self._insert_nonfull(node.children[i], key)

def insert(self, key: int):
    """B 树对外插入接口"""
    r = self.root
    if len(r.keys) == 2 * self.t - 1:
        # 根节点满了：创建新根，将旧根作为孩子分裂
        s = BNode(leaf=False)
        self.root = s
        s.children.append(r)
        self._split_child(s, 0)   # 分裂旧根
        self._insert_nonfull(s, key)
    else:
        self._insert_nonfull(r, key)
```
```cpp
void splitChild(BNode* parent, int i) {
    int t = this->t;
    BNode* y = parent->children[i];  // 满孩子
    BNode* z = new BNode(y->leaf);   // 新节点

    // z 得到 y 的右半 t-1 个键
    z->keys.assign(y->keys.begin() + t, y->keys.end());
    int midKey = y->keys[t - 1];
    y->keys.resize(t - 1);          // y 保留左半

    if (!y->leaf) {
        z->children.assign(y->children.begin() + t, y->children.end());
        y->children.resize(t);
    }

    // 中间键上浮到 parent
    parent->keys.insert(parent->keys.begin() + i, midKey);
    parent->children.insert(parent->children.begin() + i + 1, z);
}

void insertNonFull(BNode* node, int key) {
    int i = (int)node->keys.size() - 1;
    if (node->leaf) {
        // 叶节点：找插入位置
        node->keys.push_back(0);    // 先扩容
        while (i >= 0 && key < node->keys[i]) {
            node->keys[i + 1] = node->keys[i];
            i--;
        }
        node->keys[i + 1] = key;
    } else {
        while (i >= 0 && key < node->keys[i]) i--;
        i++;    // 子节点下标
        if ((int)node->children[i]->keys.size() == 2 * t - 1) {
            splitChild(node, i);
            if (key > node->keys[i]) i++;
        }
        insertNonFull(node->children[i], key);
    }
}

void insert(int key) {
    BNode* r = root;
    if ((int)r->keys.size() == 2 * t - 1) {
        BNode* s = new BNode(false);
        root = s;
        s->children.push_back(r);
        splitChild(s, 0);
        insertNonFull(s, key);
    } else {
        insertNonFull(r, key);
    }
}
```

**插入复杂度**：
- 磁盘 I/O：$O(h) = O(\log_t n)$（每层最多一次分裂 = 3 次写）
- CPU：$O(t \log_t n)$（每层 $O(t)$ 比较）

### 11.2.3–11.2.4 DELETE（删除）：三种情形与向下处理

B 树删除是三种基本操作中最复杂的。CLRS 采用**向下预处理**策略：在路径上确保**每个经过的节点至少有 $t$ 个键**，这样在需要合并或借键时父节点一定有"余量"。

**三种删除情形**：

- **情形 1（键在叶节点，叶节点有足够键）**：直接删除，最简单

- **情形 2（键在内部节点 $x$ 的键 $k$）**：
  - **2a**：若 $k$ 的左孩子 $y$ 的键数 $\geq t$，用 $y$ 中的**前驱键**替换 $k$，再在 $y$ 中删前驱
  - **2b**：若右孩子 $z$ 的键数 $\geq t$，用 $z$ 的**后继键**替换 $k$，再在 $z$ 中删后继
  - **2c**：两个孩子都只有 $t-1$ 个键，把 $k$、$y$、$z$ **合并**（$k$ 下沉），在合并节点中删 $k$

- **情形 3（目标键不在当前节点，路径上的孩子只有 $t-1$ 个键）**：
  - **3a**：若孩子 $x$ 的兄弟（相邻子节点）有 $\geq t$ 个键，**借键旋转**（父节点中分隔键下降到 $x$，兄弟的边界键上升到父）
  - **3b**：兄弟也只有 $t-1$ 个键，**合并**（与兄弟 + 父的分隔键合并为新节点，父节点键减少 1）

```python
def delete(self, node: BNode, key: int):
    """从以 node 为根的子树中删除 key"""
    t = self.t
    i = 0
    while i < len(node.keys) and key > node.keys[i]:
        i += 1

    if i < len(node.keys) and key == node.keys[i]:
        # ── 情形 1 / 2：key 在当前节点 ──
        if node.leaf:
            # 情形 1：叶节点，直接删
            node.keys.pop(i)
        else:
            # 情形 2a / 2b / 2c
            left = node.children[i]
            right = node.children[i + 1]
            if len(left.keys) >= t:
                # 情形 2a：用前驱替换
                pred = self._get_max(left)     # 左子树最大键
                node.keys[i] = pred
                self.delete(left, pred)
            elif len(right.keys) >= t:
                # 情形 2b：用后继替换
                succ = self._get_min(right)    # 右子树最小键
                node.keys[i] = succ
                self.delete(right, succ)
            else:
                # 情形 2c：合并 left + key + right
                self._merge(node, i)           # left 吸收 key 和 right
                self.delete(left, key)         # 在合并后的节点中删
    else:
        # ── 情形 3：key 不在当前节点，向下 ──
        if node.leaf:
            return   # 键不存在
        child = node.children[i]
        if len(child.keys) < t:
            # 预处理：确保 child 至少 t 个键
            self._fill(node, i)    # 借键或合并
            # fill 后节点结构可能变化，重新在节点中查
            self.delete(node, key)
        else:
            self.delete(child, key)

def _merge(self, parent: BNode, i: int):
    """将 parent.children[i] 和 parent.children[i+1] 合并，parent.keys[i] 下沉"""
    t = self.t
    left = parent.children[i]
    right = parent.children[i + 1]
    mid = parent.keys.pop(i)            # 父节点键下沉
    parent.children.pop(i + 1)         # 删除 right 指针
    left.keys.append(mid)
    left.keys.extend(right.keys)
    if not left.leaf:
        left.children.extend(right.children)

def _fill(self, parent: BNode, i: int):
    """确保 parent.children[i] 至少有 t 个键：借键或合并"""
    t = self.t
    # 尝试从左兄弟借
    if i > 0 and len(parent.children[i - 1].keys) >= t:
        self._borrow_from_left(parent, i)
    # 尝试从右兄弟借
    elif i < len(parent.children) - 1 and len(parent.children[i + 1].keys) >= t:
        self._borrow_from_right(parent, i)
    # 合并
    elif i < len(parent.children) - 1:
        self._merge(parent, i)
    else:
        self._merge(parent, i - 1)

def _get_max(self, node: BNode) -> int:
    while not node.leaf:
        node = node.children[-1]
    return node.keys[-1]

def _get_min(self, node: BNode) -> int:
    while not node.leaf:
        node = node.children[0]
    return node.keys[0]
```
```cpp
// 获取子树最大键（前驱）
int getMax(BNode* node) {
    while (!node->leaf) node = node->children.back();
    return node->keys.back();
}
// 获取子树最小键（后继）
int getMin(BNode* node) {
    while (!node->leaf) node = node->children.front();
    return node->keys.front();
}

// 合并 children[i] + keys[i] + children[i+1]
void merge(BNode* parent, int i) {
    BNode* left  = parent->children[i];
    BNode* right = parent->children[i + 1];
    int mid = parent->keys[i];

    // 从 parent 移除 keys[i] 和 children[i+1]
    parent->keys.erase(parent->keys.begin() + i);
    parent->children.erase(parent->children.begin() + i + 1);

    // 合并到 left
    left->keys.push_back(mid);
    for (int k : right->keys) left->keys.push_back(k);
    if (!left->leaf)
        for (BNode* c : right->children) left->children.push_back(c);
    delete right;
}

void deleteKey(BNode* node, int key) {
    int t = this->t;
    int i = 0;
    while (i < (int)node->keys.size() && key > node->keys[i]) i++;

    if (i < (int)node->keys.size() && key == node->keys[i]) {
        if (node->leaf) {
            node->keys.erase(node->keys.begin() + i);  // 情形 1
        } else {
            BNode* left  = node->children[i];
            BNode* right = node->children[i + 1];
            if ((int)left->keys.size() >= t) {          // 情形 2a
                int pred = getMax(left);
                node->keys[i] = pred;
                deleteKey(left, pred);
            } else if ((int)right->keys.size() >= t) { // 情形 2b
                int succ = getMin(right);
                node->keys[i] = succ;
                deleteKey(right, succ);
            } else {                                     // 情形 2c：合并
                merge(node, i);
                deleteKey(left, key);
            }
        }
    } else {
        if (node->leaf) return;  // 键不存在
        BNode* child = node->children[i];
        if ((int)child->keys.size() < t) {
            fill(node, i);       // 情形 3：预填充
            deleteKey(node, key); // 重新在 node 中搜索（fill 可能改变结构）
        } else {
            deleteKey(child, key);
        }
    }
}
```

---

## 11.3 B+ 树：为范围查询而生

### 11.3.1 B+ 树与 B 树的核心区别

B 树和 B+ 树在结构上的本质区别只有一点，但引发了所有性质的不同：

| 特性 | B 树 | B+ 树 |
|------|------|-------|
| 数据存储位置 | 内部节点和叶子都存完整数据 | **只有叶节点存数据**，内部节点仅作路由索引 |
| 内部节点键 | 同时是数据 + 路由键 | 仅作路由，可能在叶子中有副本 |
| 叶节点链接 | 无 | **双向链表**相连 |
| 范围查询 | 需回溯到祖先节点 | 找到起点后沿链表 $O(k)$ 线性扫描 |
| 顺序遍历 | 中序遍历（涉及整棵树） | 直接遍历叶链表 |
| 内部节点存储效率 | 内部节点占用真实记录空间 | 内部节点只存路由键，扇出更大 |

**形象比喻**：B 树是一本目录，翻到目录就能直接看内容；B+ 树是一本索引+正文的书，目录页（内部节点）只写页码，所有内容（数据）都在正文页（叶节点），正文页还用书签串联（叶链表）。

**扇出更大的效果**：B+ 树的内部节点只存路由键和指针（不存完整记录），相同磁盘页内能放 **更多的键**，树高进一步降低。

### 11.3.2 叶节点双向链表与范围查询

B+ 树的每个叶节点维护指向下一个（和上一个）叶节点的指针，所有叶节点形成一个**有序的双向链表**。

**范围查询 [lo, hi] 的算法**：
1. 从根节点向下找到第一个 key ≥ lo 的叶节点（$O(\log_t n)$ 次磁盘 I/O）
2. 从该叶节点开始，沿叶链表顺序扫描，收集所有 key ≤ hi 的记录（$O(k/B)$ 次 I/O，$k$ 为匹配记录数，$B$ 为每页记录数）

**总复杂度**：$O(\log_t n + k/B)$，其中 $k/B$ 是**顺序 I/O**（比随机 I/O 快 100 倍以上）。

这使得 B+ 树特别适合：
- **范围查询**（`WHERE age BETWEEN 20 AND 30`）
- **ORDER BY 查询**（叶链表天然有序）
- **全表扫描**（只需扫描叶链表层）

下面的动画演示了三种查询类型（范围查询、点查询、全表扫描）的执行过程：

<div data-component="BPlusTreeRangeQuery"></div>

### 11.3.3 B+ 树插入与删除

B+ 树的插入/删除与 B 树类似，但有以下关键区别：

1. **插入**：新键**始终插入叶节点**。分裂叶节点时，右半叶节点的**最小键上浮到父节点**（注意：这个键同时保留在叶节点中，作为叶层数据的一部分 —— 这与 B 树不同！B 树分裂时中间键不保留在原位置）

2. **删除**：叶节点键不足时，可从兄弟借键 —— 借键时需要更新父节点的路由键（因为路由键是叶层的键的副本）

3. **叶链表维护**：分裂/合并时需要维护双向链表指针

```python
class BPlusNode:
    def __init__(self, leaf=False):
        self.keys = []
        self.children = []   # 内部节点：子节点引用列表
        self.records = []    # 叶节点：记录数据列表（与 keys 一一对应）
        self.leaf = leaf
        self.next = None     # 叶链表：指向下一个叶节点
        self.prev = None     # 叶链表：指向上一个叶节点

class BPlusTree:
    def __init__(self, t: int):
        self.t = t
        self.root = BPlusNode(leaf=True)
        self.leftmost_leaf = self.root  # 最左叶节点（全表扫描起点）

    def range_search(self, lo: int, hi: int):
        """范围查询 [lo, hi]，充分利用叶链表"""
        # 步骤 1：定位起始叶节点（二分查找路径）
        leaf = self._find_leaf(self.root, lo)
        result = []
        # 步骤 2：沿叶链表顺序扫描
        while leaf is not None:
            for i, k in enumerate(leaf.keys):
                if lo <= k <= hi:
                    result.append(leaf.records[i])
                elif k > hi:
                    return result   # 超出范围，提前退出
            leaf = leaf.next       # 下一个叶节点（一次磁盘 I/O）
        return result

    def _find_leaf(self, node: BPlusNode, key: int) -> BPlusNode:
        """找到 key 应在的叶节点"""
        if node.leaf:
            return node
        i = 0
        while i < len(node.keys) and key >= node.keys[i]:
            i += 1
        return self._find_leaf(node.children[i], key)
```
```cpp
struct BPlusNode {
    vector<int> keys;
    vector<BPlusNode*> children; // 内部节点子树
    vector<string> records;     // 叶节点数据记录（简化为 string）
    bool leaf;
    BPlusNode* next = nullptr;  // 叶链表
    BPlusNode* prev = nullptr;

    BPlusNode(bool l = false) : leaf(l) {}
};

// 范围查询 [lo, hi]
vector<string> rangeSearch(BPlusNode* root, int lo, int hi) {
    // 1. 找到起始叶节点
    BPlusNode* leaf = root;
    while (!leaf->leaf) {
        int i = 0;
        while (i < (int)leaf->keys.size() && lo >= leaf->keys[i]) i++;
        leaf = leaf->children[i];
    }
    // 2. 沿叶链表扫描
    vector<string> result;
    while (leaf != nullptr) {
        for (int i = 0; i < (int)leaf->keys.size(); i++) {
            if (leaf->keys[i] < lo) continue;
            if (leaf->keys[i] > hi) return result;
            result.push_back(leaf->records[i]);
        }
        leaf = leaf->next;  // 顺序 I/O，极快
    }
    return result;
}
```

### 11.3.4 MySQL InnoDB：聚簇索引 vs 辅助索引

MySQL InnoDB 存储引擎使用 B+ 树作为索引结构，但有两种不同的形态：

**聚簇索引（Clustered Index）**：
- 以**主键**为索引键构建的 B+ 树
- **叶节点直接存储整行数据**（而非记录的磁盘地址）
- 因为数据就在叶节点，主键查询不需要额外的 I/O
- 表数据的物理存储顺序与主键顺序一致（数据文件 = 索引文件）
- 每张表**只能有一个**聚簇索引（InnoDB 强制要求有主键）

**辅助索引（Secondary Index / Non-Clustered Index）**：
- 以非主键列（如 `age`）为键构建的 B+ 树  
- **叶节点存储 (索引列值, 主键ID)**，而非完整行数据
- 查询时先从辅助索引找到主键 ID，再到聚簇索引中找完整行 → **"回表"（Double Lookup）**
- 每个非主键列可以单独建辅助索引，一张表可以有多个辅助索引

**覆盖索引（Covering Index）**：
- 若查询所需的列**全部在辅助索引中**（索引列 + 主键），则无需回表
- `SELECT age, id FROM users WHERE age = 25` 对建有 age 索引的表是覆盖索引查询
- 这是高性能查询优化的关键手段，`EXPLAIN` 输出 `Using index` 即表示使用了覆盖索引

以下可视化展示了三种查询场景下索引的工作方式：

<div data-component="MySQLBTreeIndex"></div>

---

## 11.4 实际应用与生态

### 11.4.1 文件系统中的 B 树

**ext4（Linux）**：
- 目录项使用哈希树（HTree），底层是 B 树变体，支持大目录的快速文件查找
- extent 结构将相邻块聚合，减少 B 树节点开销，4 个 extent 可以直接内联在 inode 中

**NTFS（Windows）**：
- 主文件表（MFT）的索引属性（`$INDEX_ALLOCATION`）使用 B+ 树
- 目录文件名索引 (`$I30`) 是一棵以文件名为键的 B+ 树，支持快速目录项查找

**ZFS/Btrfs**：
- 使用 Copy-on-Write B 树，所有修改写新位置而不覆盖原数据，天然支持快照

### 11.4.2 数据库索引优化实践

**索引选择性（Selectivity）**：
$$\text{Selectivity} = \frac{\text{distinct values}}{\text{total rows}}$$

选择性越高，索引越有效（1.0 = 唯一值 = 最高选择性）。性别列（2 个不同值）建索引意义不大；用户 ID（唯一）建索引极有价值。

**经典优化场景**：

```sql
-- 低效：gender 选择性低，全表扫描可能更快
SELECT * FROM users WHERE gender = 'M';

-- 高效：id 唯一，聚簇索引一次命中
SELECT * FROM users WHERE id = 12345;

-- 范围查询 + 覆盖索引（假设有 (age, name) 联合索引）
SELECT age, name FROM users WHERE age BETWEEN 20 AND 30;
-- → 不回表，仅扫描辅助索引叶节点
```

**联合索引（Composite Index）**的最左前缀原则：
- 索引 (a, b, c) 可以用于查询 `WHERE a=1`、`WHERE a=1 AND b=2`、`WHERE a=1 AND b=2 AND c=3`
- 不能用于 `WHERE b=2`（跳过了最左列）
- 原因：联合索引按 (a, b, c) 排序，只有固定 a 的前提下 b 才有序

### 11.4.3 B+ 树 vs LSM-Tree：读优先 vs 写优先

| 维度 | B+ 树（InnoDB / PostgreSQL） | LSM-Tree（RocksDB / LevelDB / Cassandra） |
|------|---------------------|---------------------|
| **写入方式** | 随机写（原地更新，可能引发页分裂） | 追加写（写 MemTable 后 Sequential I/O） |
| **写放大** | 1–2 倍（最坏情形分裂级联） | 10–30 倍（多级 Compaction） |
| **读性能** | 极优（$O(\log n)$ 磁盘 I/O） | 较差（需查多个 SSTable Level） |
| **读放大** | $O(\log_t n)$ | $O(\log n \cdot \text{level})$ |
| **适合场景** | OLTP（高读取、事务型） | 高写入吞吐（日志、时序数据、KV 存储） |
| **典型系统** | MySQL、PostgreSQL、SQLite | RocksDB、Cassandra、HBase、TiKV |

**LSM-Tree（Log-Structured Merge Tree）简介**：
- 所有写入先进内存 MemTable（通常是红黑树或跳表）
- MemTable 满后整体 flush 到磁盘形成 Level-0 SSTable（有序文件）
- 后台 Compaction 将多级 SSTable 合并排序，消除重复和删除标记
- 写入即 Append，极高写吞吐；读取需查 MemTable + 多层 SSTable

### 11.4.4 时序数据库与列式存储中的树结构

**时序数据库（InfluxDB, TimescaleDB）**：
- 数据天然按时间有序 → 写入几乎全是顺序追加 → LSM-Tree 是更好选择
- 对最新 $k$ 条数据的范围查询极其频繁 → B+ 树的叶链表扫描也很高效
- TimescaleDB 基于 PostgreSQL（B+ 树）并添加表分区（hypertable），按时间分区后各分区高度极低

**列式存储（Parquet / ORC 文件）**：
- 不使用传统 B+ 树索引
- 使用**列 Block Statistics**（min/max/count）实现类似 B 树路由的功能（Predicate Pushdown）
- Zone Map / Sparse Index：每 N 行记录一次最小/最大键值，查询时跳过不相关的 Block

---

## 11.5 总结与练习

### 11.5.1 核心知识点回顾

| 知识点 | 关键点 |
|--------|--------|
| B 树动机 | 磁盘 I/O 代价远大于 CPU，需减少树高（= I/O 次数） |
| B 树度数 t | 每节点 [t-1, 2t-1] 键，[t, 2t] 子节点 |
| B 树高度 | $h \leq \log_t \lceil (n+1)/2 \rceil$ |
| 插入策略 | 提前分裂，保证一趟下行完成 |
| 删除策略 | 向下预填充，保证路径节点 ≥ t 键 |
| B+ 树关键区别 | 数据仅在叶节点 + 叶链表 → 高效范围查询 |
| 聚簇 vs 辅助 | 聚簇=数据即索引；辅助需回表（除非覆盖索引） |
| B+ 树 vs LSM | 读优先 vs 写优先，场景决定选型 |

### 11.5.2 常见陷阱总结

1. **B 树 vs B+ 树的内部节点**：B 树内部节点存完整数据，B+ 树内部节点只存路由键。写代码时一定区分，否则范围查询实现会有 bug。

2. **删除时的连锁合并**：合并会使父节点失去一个键，父节点可能也低于下界，引发向上的连锁合并，最终树高可能减 1（类似向下连锁分裂）。

3. **B+ 树分裂时路由键的选取**：叶节点分裂时，右半的**最小键**复制（不是移走！）到父节点作为路由键。而 B 树分裂时中间键是**移走**的（从原节点删除）。

4. **InnoDB 主键选取**：主键应该尽量**单调递增**（如自增 ID）。随机 UUID 作主键会导致聚簇索引频繁**页分裂**（因为新记录要插入中间），引发大量随机 I/O 和索引碎片。

5. **t 参数与磁盘页大小**：不是 t 越大越好。t 太大意味着每节点键数多，节点内线性/二分查找时间增加，且节点不能完整放入 CPU Cache 行，实际性能可能下降。

### 11.5.3 面试高频考点

| 题目 | 要点 |
|------|------|
| 为什么数据库用 B+ 树不用 B 树？ | 叶链表支持 $O(k)$ 范围查询；内部节点只存路由键，扇出更大（树更矮）；全表扫描只需枚举叶层 |
| 为什么不用 AVL 树 / 红黑树做数据库索引？ | 树高 $O(\log_2 n) \approx 30$（$n=10^9$），每次操作费 30 次磁盘 I/O；B+ 树高度 2–4 层 |
| 聚簇索引和辅助索引的区别？ | 见 11.3.4，重点：回表 / 覆盖索引 |
| 如何设计联合索引？ | 最左前缀原则；高选择性列放左；覆盖查询常见列组合 |
| LSM-Tree 和 B+ 树怎么选？ | 读多写少 → B+ 树；写多读少/追加为主 → LSM-Tree |

> **💡 思考题**：假设磁盘块大小从 4KB 变为 64KB（NVMe 优化），$t$ 如何变化？树高如何变化？每次磁盘读取后的比较次数如何变化？这对查找性能的综合影响是好是坏？

---

> **参考资料**：  
> - 《算法导论》（CLRS）第 4 版 Chapter 18（B-Trees）  
> - MySQL 官方文档：InnoDB Storage Engine / Index types  
> - Bayer & McCreight, *Organization and Maintenance of Large Ordered Indexes* (1972)  
> - Comer, *The Ubiquitous B-Tree* (1979), ACM Computing Surveys  
> - RocksDB Wiki: https://github.com/facebook/rocksdb/wiki  

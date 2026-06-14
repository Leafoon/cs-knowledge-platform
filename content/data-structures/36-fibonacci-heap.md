# Chapter 36: Fibonacci 堆与高级优先队列（Fibonacci Heap & Advanced Priority Queues）

> **学习目标**：理解 Fibonacci 堆的懒惰合并策略与 CONSOLIDATE 过程，掌握各操作的摊销复杂度；理解 DECREASE-KEY 的 O(1) 摊销如何实现；掌握二项堆的"二进制加法"合并类比；明确工程中为何不用 Fibonacci 堆。

---

## 章节导读

在所有通用优先队列中，**Fibonacci 堆**的理论复杂度最优：

| 操作 | 二叉堆 | 二项堆 | **Fibonacci 堆** |
|---|---|---|---|
| INSERT | $O(\log n)$ | $O(\log n)$ | **$O(1)$** 摊销 |
| MINIMUM | $O(1)$ | $O(\log n)$ | **$O(1)$** |
| UNION | $O(n)$ | $O(\log n)$ | **$O(1)$** 摊销 |
| EXTRACT-MIN | $O(\log n)$ | $O(\log n)$ | **$O(\log n)$** 摊销 |
| DECREASE-KEY | $O(\log n)$ | $O(\log n)$ | **$O(1)$** 摊销 |
| DELETE | $O(\log n)$ | $O(\log n)$ | **$O(\log n)$** 摊销 |

核心突破是 **DECREASE-KEY 降到 $O(1)$ 摊销**，这使 Dijkstra 算法从 $O((V+E)\log V)$ 降至 $O(E + V \log V)$（当 $E \gg V$ 时显著更优）。

**代价**：实现极其复杂，常数因子大，缓存行为差，工程中几乎不用。学习 Fibonacci 堆的意义更多在于：体验"精心设计的数据结构如何将理论摊销到极致"。

> **比喻**：Fibonacci 堆就像一个"懒惰的归档员"：
> - 收到新文件时（INSERT）：直接扔进桌面上的一堆（不整理）
> - 需要最小值时（EXTRACT-MIN）：才整理一次，把所有文件归档到恰好合适的文件夹（CONSOLIDATE）
> - 降低某个文件的优先级（DECREASE-KEY）：直接撕掉原来的标签贴新标签，卡片从文件夹里飞出来放到桌面上

---

## 36.1 Fibonacci 堆的结构

### 36.1.1 多树结构：多棵最小堆有序树

Fibonacci 堆不是一棵树，而是**一组最小堆有序树（Min-Heap-Ordered Trees）的集合**。

所谓"最小堆有序"：每棵树中，父节点的 key ≤ 所有子节点的 key（不要求是完全二叉树，形状完全任意）。

**为什么要多棵树？**  
单棵最小堆做 UNION 需要 $O(n)$。多棵树的集合做 UNION 只需将两个集合合并（$O(1)$），代价在 EXTRACT-MIN 时再统一支付（懒惰策略！）

### 36.1.2 根列表（Root List）：双向循环链表

所有树的根节点通过**双向循环链表**连接，称为**根列表（Root List）**：

```
min → [3] ↔ [7] ↔ [17] ↔ [24] ↔ (回到 [3])
       |         |          |
      [8]       [18]       [26]
       |
      [30]
```

- `H.min` 始终指向根列表中 key 最小的节点（任意遍历即可找到）
- INSERT 只是在根列表末尾插入新节点（$O(1)$）
- UNION 只是将两个根列表首尾相连（$O(1)$）

### 36.1.3 节点结构：key、degree、mark、parent、child

每个节点包含以下字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `key` | 数值 | 优先级（越小越优先） |
| `degree` | 整数 | 子节点数量 |
| `mark` | 布尔 | 是否已经失去过一个子节点（DECREASE-KEY 关键！） |
| `parent` | 指针 | 父节点（根节点的 parent = None） |
| `child` | 指针 | 任意一个子节点的指针 |
| `left`, `right` | 指针 | 兄弟链表（双向循环） |

`mark` 字段是 DECREASE-KEY O(1) 摊销的核心：它记录"这个节点是否已经失去过一个孩子"，用于触发级联裁剪（CASCADING-CUT）。

### 36.1.4 完整结构体定义

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math

@dataclass
class FibNode:
    """Fibonacci 堆节点"""
    key: float
    # 以下字段在插入时初始化
    degree: int = 0                     # 子节点数量
    mark: bool = False                  # 是否已失去一个子节点
    parent: Optional['FibNode'] = None  # 父节点，根节点为 None
    child: Optional['FibNode'] = None   # 任意一个子节点
    left: Optional['FibNode'] = None    # 兄弟链表（双向循环，初始指向自己）
    right: Optional['FibNode'] = None   # 兄弟链表

    def __post_init__(self):
        # 初始时，节点自身构成一个只有自己的循环链表
        self.left = self
        self.right = self

    def __repr__(self):
        return f"FibNode(key={self.key}, degree={self.degree}, mark={self.mark})"

class FibonacciHeap:
    """Fibonacci 堆实现（完整版）"""
    def __init__(self):
        self.min: Optional[FibNode] = None  # 最小节点指针
        self.n: int = 0                      # 节点总数
```

```cpp
#include <cstddef>
#include <cmath>
#include <vector>
#include <limits>
#include <cassert>

struct FibNode {
    double key;
    int    degree = 0;
    bool   mark   = false;
    FibNode* parent = nullptr;
    FibNode* child  = nullptr;
    FibNode* left   = nullptr;  // 兄弟双向循环链表
    FibNode* right  = nullptr;

    explicit FibNode(double k) : key(k) {
        left = right = this;   // 初始时自身成环
    }
};

class FibHeap {
public:
    FibNode* minNode = nullptr;
    int n = 0;   // 节点总数

    FibHeap() = default;
    ~FibHeap();  // 需要递归释放所有节点

    // ... 各操作见下文
};
```

---

## 36.2 Fibonacci 堆的各操作

### 36.2.1 INSERT：将节点加入根列表

**步骤**：
1. 创建新节点 $x$（degree=0, mark=false）
2. 将 $x$ 插入根列表（双向循环链表的 $O(1)$ 插入）
3. 若 $x.key < H.min.key$，更新 $H.min = x$

**实际代价**：$O(1)$  
**摊销代价**：$O(1)$（势能 $\Phi$ 增加 1，见 35.5 节）

```python
def insert(self, key: float) -> FibNode:
    """
    插入新节点。
    实际代价 O(1)，摊销代价 O(1)。
    不整理堆结构——完全懒惰！
    """
    node = FibNode(key)
    # 将 node 插入根列表（在 min 的左侧）
    self._root_list_insert(node)
    # 更新 min 指针
    if self.min is None or node.key < self.min.key:
        self.min = node
    self.n += 1
    return node  # 返回节点引用，后续 decrease_key 时需要

def _root_list_insert(self, node: FibNode):
    """将 node 插入根双向循环链表（在 self.min 左侧）"""
    if self.min is None:
        node.left = node
        node.right = node
    else:
        # 在 min 和 min.left 之间插入 node
        node.right = self.min
        node.left = self.min.left
        self.min.left.right = node
        self.min.left = node
    node.parent = None
```

```cpp
FibNode* insert(double key) {
    FibNode* x = new FibNode(key);
    // 将 x 接入根列表（在 minNode 左侧）
    if (minNode == nullptr) {
        minNode = x;
    } else {
        // x 插入 minNode 与 minNode->left 之间
        x->right = minNode;
        x->left  = minNode->left;
        minNode->left->right = x;
        minNode->left = x;
        if (x->key < minNode->key) minNode = x;
    }
    n++;
    return x;   // 返回句柄，decrease_key 时使用
}
```

### 36.2.2 UNION：两堆合并

**步骤**：将两个堆的根列表首尾相连，更新 min 为两者中较小的那个。

**实际代价**：$O(1)$  
**摊销代价**：$O(1)$

这是 Fibonacci 堆最显眼的优势：二叉堆 UNION 需要 $O(n)$ 重建，而 Fibonacci 堆只需合并链表。

```python
def union(self, other: 'FibonacciHeap') -> 'FibonacciHeap':
    """
    合并两个 Fibonacci 堆，返回新堆（破坏性合并）。
    只是将两个根列表首尾相连，O(1)。
    """
    result = FibonacciHeap()
    result.min = self.min

    # 连接两个根列表（双向循环链表的首尾连接）
    if self.min is not None and other.min is not None:
        # 将 other 的链表插入到 self.min 和 self.min.left 之间
        self_last = self.min.left
        other_last = other.min.left

        self_last.right = other.min
        other.min.left = self_last
        other_last.right = self.min
        self.min.left = other_last

    # 更新 min
    if self.min is None or (other.min is not None and other.min.key < self.min.key):
        result.min = other.min

    result.n = self.n + other.n
    return result
```

```cpp
static FibHeap* unite(FibHeap* h1, FibHeap* h2) {
    FibHeap* result = new FibHeap();
    result->minNode = h1->minNode;

    if (h1->minNode != nullptr && h2->minNode != nullptr) {
        // 将两个双向循环链表合并
        FibNode* last1 = h1->minNode->left;
        FibNode* last2 = h2->minNode->left;
        last1->right = h2->minNode;
        h2->minNode->left = last1;
        last2->right = h1->minNode;
        h1->minNode->left = last2;
    }

    if (h1->minNode == nullptr ||
        (h2->minNode != nullptr && h2->minNode->key < h1->minNode->key))
        result->minNode = h2->minNode;

    result->n = h1->n + h2->n;
    return result;
}
```

### 36.2.3 EXTRACT-MIN：取出最小节点（触发 CONSOLIDATE）

这是 Fibonacci 堆最复杂的操作。步骤如下：

1. 将 $H.min$ 的所有子节点加入根列表（子节点的 parent 置 None）
2. 从根列表中移除 $H.min$
3. 若根列表为空，$H.min = \text{None}$；否则运行 **CONSOLIDATE**
4. CONSOLIDATE 后，更新 $H.min$ 为根列表中的最小节点

**实际代价**：$O(D(n) + t(H))$（其中 $D(n) = O(\log n)$ 是最大度数，$t(H)$ 是 EXTRACT-MIN 前的树数）  
**摊销代价**：$O(\log n)$（CONSOLIDATE 使树数从 $t + D$ 降至 $D(n)$，势能大幅下降，弥补了实际代价）

```python
def extract_min(self) -> Optional[float]:
    """
    取出并返回最小 key。
    触发 CONSOLIDATE 整理根列表。
    摊销 O(log n)。
    """
    z = self.min
    if z is None:
        return None

    # Step 1: 将 z 的所有子节点加入根列表
    if z.child is not None:
        # 遍历 z 的子链表，全部移入根列表
        child = z.child
        children = []
        cur = child
        while True:
            children.append(cur)
            cur = cur.right
            if cur is child:
                break
        for ch in children:
            self._root_list_insert(ch)
            ch.parent = None

    # Step 2: 从根列表移除 z
    self._root_list_remove(z)

    # Step 3: 更新 min 并整理
    if z.right == z:
        # 根列表只有 z，移除后为空
        self.min = None
    else:
        # 暂时将 min 指向任意一个根（consolidate 会找到真正最小值）
        self.min = z.right
        self._consolidate()

    self.n -= 1
    return z.key

def _root_list_remove(self, node: FibNode):
    """从根双向循环链表中移除 node"""
    node.left.right = node.right
    node.right.left = node.left

def _consolidate(self):
    """
    整理根列表：将所有度相同的树两两合并，
    直到每个度数至多一棵树为止。
    类似二进制加法的进位过程。
    """
    max_degree = int(math.log(self.n, 1.618)) + 2 if self.n > 0 else 1
    A = [None] * (max_degree + 2)   # A[d] = 度为 d 的唯一根节点

    # 收集根列表中所有根节点
    roots = []
    if self.min:
        cur = self.min
        while True:
            roots.append(cur)
            cur = cur.right
            if cur is self.min:
                break

    for w in roots:
        x = w
        d = x.degree
        # 当存在同度的树时，合并（堆化：将较大根变为较小根的子节点）
        while d < len(A) and A[d] is not None:
            y = A[d]
            if x.key > y.key:
                x, y = y, x   # 保持 x.key <= y.key
            # 将 y 链接为 x 的子节点（FIB-HEAP-LINK）
            self._fib_heap_link(y, x)
            A[d] = None
            d += 1
        if d >= len(A):
            A.extend([None] * (d - len(A) + 2))
        A[d] = x

    # 重建根列表，找到最小值
    self.min = None
    for node in A:
        if node is None:
            continue
        node.left = node
        node.right = node
        self._root_list_insert(node)
        if self.min is None or node.key < self.min.key:
            self.min = node

def _fib_heap_link(self, y: FibNode, x: FibNode):
    """
    将 y 变为 x 的子节点（前提：x.key <= y.key）。
    y 从根列表移除，添加到 x 的子链表，x.degree += 1。
    """
    # y 已从根列表移除（由 consolidate 保证）
    y.left = y
    y.right = y
    # 将 y 插入 x 的子双向循环链表
    if x.child is None:
        x.child = y
    else:
        y.right = x.child
        y.left = x.child.left
        x.child.left.right = y
        x.child.left = y
    y.parent = x
    x.degree += 1
    y.mark = False   # 新加入子节点，mark 清零
```

```cpp
double extract_min() {
    FibNode* z = minNode;
    assert(z != nullptr);

    // Step 1: 将 z 的所有孩子加入根列表
    if (z->child != nullptr) {
        std::vector<FibNode*> children;
        FibNode* c = z->child;
        do { children.push_back(c); c = c->right; } while (c != z->child);
        for (FibNode* ch : children) {
            // 插入根列表
            ch->left = minNode->left;
            ch->right = minNode;
            minNode->left->right = ch;
            minNode->left = ch;
            ch->parent = nullptr;
        }
    }

    // Step 2: 从根列表移除 z
    z->left->right = z->right;
    z->right->left = z->left;

    double result = z->key;
    n--;

    if (z == z->right && z->child == nullptr) {
        minNode = nullptr;   // 堆变空
    } else {
        minNode = z->right;
        consolidate();
    }

    delete z;
    return result;
}

void fib_heap_link(FibNode* y, FibNode* x) {
    // 从根列表移除 y（已在 consolidate 外部完成）
    // 将 y 添加为 x 的孩子
    y->parent = x;
    if (x->child == nullptr) {
        x->child = y;
        y->left = y->right = y;
    } else {
        y->right = x->child;
        y->left  = x->child->left;
        x->child->left->right = y;
        x->child->left = y;
    }
    x->degree++;
    y->mark = false;
}

void consolidate() {
    int max_deg = (int)(std::log((double)n) / std::log(1.618)) + 2;
    std::vector<FibNode*> A(max_deg + 2, nullptr);

    // 收集根节点
    std::vector<FibNode*> roots;
    FibNode* cur = minNode;
    do { roots.push_back(cur); cur = cur->right; } while (cur != minNode);

    for (FibNode* w : roots) {
        FibNode* x = w;
        int d = x->degree;
        while (d < (int)A.size() && A[d] != nullptr) {
            FibNode* y = A[d];
            if (x->key > y->key) std::swap(x, y);
            fib_heap_link(y, x);
            A[d] = nullptr;
            d++;
        }
        if (d >= (int)A.size()) A.resize(d + 2, nullptr);
        A[d] = x;
    }

    // 重建根列表
    minNode = nullptr;
    for (FibNode* nd : A) {
        if (!nd) continue;
        nd->left = nd->right = nd;
        if (!minNode) { minNode = nd; }
        else {
            nd->right = minNode; nd->left = minNode->left;
            minNode->left->right = nd; minNode->left = nd;
            if (nd->key < minNode->key) minNode = nd;
        }
    }
}
```

<div data-component="FibHeapConsolidate"></div>

### 36.2.4 DECREASE-KEY：降低 key 并触发级联裁剪

这是 Fibonacci 堆最精妙的操作，也是 $O(1)$ 摊销的核心。

**步骤**：
1. 将 $x.key$ 降低为新值 $k$
2. 若 $x$ 不是根，且 $x.key < x.parent.key$（违反堆性质）：
   - **CUT**（裁剪）：将 $x$ 从父节点的子链表中切出，加入根列表；$x.mark = \text{false}$
   - **CASCADING-CUT**（级联裁剪）：对 $x$ 的原父节点 $y$ 递归检查：
     - 若 $y$ 是根：停止
     - 若 $y.mark = \text{false}$：将 $y.mark$ 置 true，停止（$y$ 第一次失去孩子，做标记）
     - 若 $y.mark = \text{true}$：对 $y$ 执行 CUT + 继续向上级联裁剪
3. 更新 $H.min$

**`mark` 的语义**：一个节点 $y$ 的 `mark = true` 意味着"$y$ 已经失去了一个孩子"。如果再次失去孩子，$y$ 本身也会被裁剪出来。这确保了每个节点的子节点数量不会无限减少，保证 Fibonacci 性质（度数界的证明依赖于此）。

> **直觉**：`mark` 像一个"黄牌"。第一次失去孩子 → 亮黄牌（mark=true）。第二次失去孩子 → 红牌出局（自己也被切出来）。切出来后黄牌清零（mark=false）。

```python
def decrease_key(self, x: FibNode, new_key: float):
    """
    将 x 的 key 降低为 new_key。
    摊销代价 O(1)。
    
    前提：new_key <= x.key (只能降低不能升高)
    关键：CUT + CASCADING-CUT 保证势能函数每步变化有界
    """
    assert new_key <= x.key, "new_key must be ≤ current key"
    x.key = new_key

    parent = x.parent
    if parent is not None and x.key < parent.key:
        # 违反堆序性质：将 x 切出
        self._cut(x, parent)
        self._cascading_cut(parent)

    if x.key < self.min.key:
        self.min = x

def _cut(self, x: FibNode, y: FibNode):
    """
    将 x 从 y 的子链表中切出，加入根列表。
    x.mark 清零（成为新树的根后重置）。
    """
    # 从 y 的子链表中移除 x
    if x.right == x:
        # x 是 y 的唯一孩子
        y.child = None
    else:
        if y.child is x:
            y.child = x.right   # 更新 child 指针
        x.left.right = x.right
        x.right.left = x.left
    y.degree -= 1
    # 将 x 加入根列表
    x.left = x
    x.right = x
    self._root_list_insert(x)
    x.parent = None
    x.mark = False   # 成为根，mark 清零

def _cascading_cut(self, y: FibNode):
    """
    级联裁剪：沿着 parent 链向上，对已标记的节点递归裁剪。
    摊销 O(1)：每次裁剪消耗 1 个 mark（势能 -2），补偿实际代价。
    """
    z = y.parent
    if z is not None:
        if not y.mark:
            # y 第一次失去孩子：设置 mark，停止级联
            y.mark = True
        else:
            # y 已有 mark（第二次失去孩子）：切出 y，继续向上
            self._cut(y, z)
            self._cascading_cut(z)
```

```cpp
void cut(FibNode* x, FibNode* y) {
    // 从 y 的子链表中删除 x
    if (x->right == x) {
        y->child = nullptr;
    } else {
        if (y->child == x) y->child = x->right;
        x->left->right = x->right;
        x->right->left = x->left;
    }
    y->degree--;
    // 将 x 插入根列表
    x->left = x->right = x;
    x->right = minNode;
    x->left  = minNode->left;
    minNode->left->right = x;
    minNode->left = x;
    x->parent = nullptr;
    x->mark = false;   // 成为根后清除标记
}

void cascading_cut(FibNode* y) {
    FibNode* z = y->parent;
    if (z != nullptr) {
        if (!y->mark) {
            y->mark = true;   // 第一次失去孩子，打标记
        } else {
            cut(y, z);        // 第二次失去孩子，切出
            cascading_cut(z); // 继续向上
        }
    }
}

void decrease_key(FibNode* x, double new_key) {
    assert(new_key <= x->key);
    x->key = new_key;
    FibNode* y = x->parent;
    if (y != nullptr && x->key < y->key) {
        cut(x, y);
        cascading_cut(y);
    }
    if (x->key < minNode->key) minNode = x;
}
```

<div data-component="FibHeapDecreaseKey"></div>

### 36.2.5 DELETE：删除任意节点

利用 DECREASE-KEY + EXTRACT-MIN 组合实现：

```python
def delete(self, x: FibNode):
    """
    删除节点 x。
    先将 x.key 降为 -∞（它必然成为 min），再 extract_min。
    摊销代价 O(log n)（EXTRACT-MIN 主导）。
    """
    self.decrease_key(x, float('-inf'))  # x 成为全局最小
    self.extract_min()                   # 取出并删除
```

```cpp
void delete_node(FibNode* x) {
    decrease_key(x, -std::numeric_limits<double>::infinity());
    extract_min();
}
```

---

## 36.3 Fibonacci 堆的理论意义

### 36.3.1 度数界 $D(n) \leq \log_\phi n$

**为什么叫 Fibonacci 堆？**  
Fibonacci 堆的名字来源于：CONSOLIDATE 后，**度数为 $k$ 的节点至少有 $F_{k+2}$ 个后代**（$F$ 是 Fibonacci 数列）。

**证明思路**（关键引理）：  
设 $y_1, y_2, \ldots, y_k$ 是节点 $x$ 的孩子（按加入顺序排列）。当 $y_i$ 被链接到 $x$ 时，$x.degree \geq i-1$（因为已有 $i-1$ 个孩子）。由 FIB-HEAP-LINK 规则（度相同才合并），此时 $y_i.degree \geq i-1$。之后 $y_i$ 最多失去 1 个孩子（否则会被级联裁剪切出），所以：

$$y_i.\text{degree} \geq i - 2 \quad (i \geq 2)$$

设 $s_k$ = 度为 $k$ 的节点的最小子树大小，则：

$$s_k \geq F_{k+2} = \begin{cases} 1 & k=0 \\ 1 & k=1 \\ F_k + F_{k+1} & k \geq 2 \end{cases}$$

由 $n \geq s_k \geq \phi^k$（$\phi = \frac{1+\sqrt{5}}{2} \approx 1.618$），得：

$$k \leq \log_\phi n \approx 1.44 \log_2 n$$

这就是为什么 CONSOLIDATE 后至多 $D(n) = \lfloor \log_\phi n \rfloor + 1$ 棵不同度的树，EXTRACT-MIN 的代价是 $O(\log n)$。

### 36.3.2 Dijkstra 使用 Fibonacci 堆：$O(V \log V + E)$

**标准 Dijkstra**（使用二叉堆）：
- $V$ 次 EXTRACT-MIN：$O(V \log V)$
- $E$ 次 DECREASE-KEY（松弛操作）：$O(E \log V)$
- **总计**：$O((V + E) \log V)$

**Fibonacci 堆 Dijkstra**：
- $V$ 次 EXTRACT-MIN 摊销：$O(V \log V)$
- $E$ 次 DECREASE-KEY 摊销：$O(E \times 1) = O(E)$
- **总计**：$O(V \log V + E)$

**优势显现的场景**：当 $E \gg V$（密集图）时，$O(E \log V)$ vs $O(E)$ 差距明显。例如 $E = V^2$（完全图）：

| 方法 | 时间复杂度 | 当 $V=10^4, E=10^8$ |
|---|---|---|
| 二叉堆 Dijkstra | $O(E \log V)$ | $\approx 10^8 \times 14 \approx 1.4 \times 10^9$ 步 |
| Fibonacci 堆 Dijkstra | $O(V \log V + E)$ | $\approx 10^8$ 步 |

```python
import heapq
from collections import defaultdict

# 使用标准二叉堆（Python heapq）的 Dijkstra
def dijkstra_binary_heap(graph: dict, src: int) -> dict:
    """
    graph: {u: [(v, w), ...]}（邻接表）
    返回 dist[v] = src 到 v 的最短路径长度
    
    时间复杂度：O((V + E) log V)（用二叉堆）
    """
    dist = defaultdict(lambda: float('inf'))
    dist[src] = 0
    pq = [(0, src)]   # (距离, 节点)

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue   # 过期条目，跳过（懒删除）
        for v, w in graph.get(u, []):
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))   # O(log V)

    return dict(dist)

# 使用 Fibonacci 堆的 Dijkstra（理论最优）
def dijkstra_fibonacci_heap(graph: dict, src: int) -> dict:
    """
    使用 Fibonacci 堆实现 Dijkstra。
    时间复杂度：O(V log V + E)
    
    注意：实际工程中 Python heapq 更快（缓存友好，常数小）；
    Fibonacci 堆仅理论上更优。
    """
    fib = FibonacciHeap()
    dist = {}
    node_map = {}   # vertex -> FibNode

    # 初始化：所有节点插入堆
    vertices = set()
    for u in graph:
        vertices.add(u)
        for v, _ in graph[u]:
            vertices.add(v)

    for v in vertices:
        d = 0.0 if v == src else float('inf')
        node_map[v] = fib.insert(d)
        dist[v] = d

    while fib.min is not None:
        u_node = fib.min
        u_dist = u_node.key
        fib.extract_min()

        # 找到对应的顶点（此处简化：实际需要 key -> vertex 映射）
        u = next(v for v, nd in node_map.items() if nd is u_node)

        for v, w in graph.get(u, []):
            if u_dist + w < dist[v]:
                dist[v] = u_dist + w
                fib.decrease_key(node_map[v], dist[v])  # O(1) 摊销！

    return dist
```

```cpp
#include <queue>
#include <vector>
#include <limits>
#include <unordered_map>

// 标准二叉堆 Dijkstra：O((V+E) log V)
std::vector<double> dijkstra_binary(
    int V,
    const std::vector<std::vector<std::pair<int,double>>>& adj,
    int src)
{
    std::vector<double> dist(V, std::numeric_limits<double>::infinity());
    dist[src] = 0;
    // (距离, 顶点) 最小堆
    std::priority_queue<std::pair<double,int>,
                        std::vector<std::pair<double,int>>,
                        std::greater<>> pq;
    pq.push({0.0, src});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;   // 过期条目（懒删除）
        for (auto [v, w] : adj[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});   // O(log V)
            }
        }
    }
    return dist;
}

// Fibonacci 堆 Dijkstra：O(V log V + E)
std::vector<double> dijkstra_fibonacci(
    int V,
    const std::vector<std::vector<std::pair<int,double>>>& adj,
    int src)
{
    FibHeap fib;
    std::vector<double> dist(V, std::numeric_limits<double>::infinity());
    std::vector<FibNode*> nodes(V);

    for (int v = 0; v < V; v++) {
        nodes[v] = fib.insert(v == src ? 0.0 : dist[v]);
        dist[v] = v == src ? 0.0 : dist[v];
    }

    // 需要额外的 vertex 到节点的映射（此处简化）
    while (fib.minNode != nullptr) {
        double d = fib.minNode->key;
        // 通过 tag/用户数据找到顶点 u（略）
        int u = 0; // 省略 tag 映射
        fib.extract_min();

        for (auto [v, w] : adj[u]) {
            if (d + w < dist[v]) {
                dist[v] = d + w;
                fib.decrease_key(nodes[v], dist[v]);  // O(1) 摊销！
            }
        }
    }
    return dist;
}
```

### 36.3.3 MST（Prim 算法）理论最优：$O(E + V \log V)$

与 Dijkstra 类似，Prim 算法本质是"贪心 + 优先队列"：每次取距离最近的非树节点，松弛其邻居。使用 Fibonacci 堆：

- $V$ 次 EXTRACT-MIN：$O(V \log V)$
- $E$ 次 DECREASE-KEY：$O(E)$
- **总计**：$O(E + V \log V)$

这是 Prim 算法目前已知的最优理论复杂度（对比使用二叉堆的 $O((V+E) \log V)$）。

### 36.3.4 工程障碍：为什么实际不用？

尽管理论最优，Fibonacci 堆在工业代码中几乎见不到，原因如图：

| 劣势 | 具体说明 |
|---|---|
| **大常数因子** | 每次 INSERT/DECREASE-KEY 涉及多次指针操作（left/right/parent/child），同级操作约为二叉堆的 5~10× |
| **缓存不友好** | 多棵树的节点分散在内存堆中，频繁访问随机地址，L1/L2 缓存命中率极低 |
| **实现复杂** | 需要正确维护 6 个指针字段 + 2 套双向循环链表（根列表 + 子链表）+ mark 机制，bug 极多 |
| **实际图规模限制** | 实践中 $E \sim O(V)$ 的稀疏图，$O(E \log V) \approx O(V \log V)$，二叉堆够用 |

**实测数据**（参考文献 Larkin et al. 2014）：在典型 Dijkstra 基准测试中，精心实现的二叉堆或斐波那契堆，后者运行时间往往是前者的 2~5 倍。

---

## 36.4 二项堆（Binomial Heap）

二项堆是 Fibonacci 堆的"严格版"前身：所有操作都是严格 $O(\log n)$（不需要摊销），但没有 $O(1)$ DECREASE-KEY。

### 36.4.1 二项树 $B_k$：递归定义

**$B_0$**：单个节点。

**$B_k$**：由两棵 $B_{k-1}$ 合并而成 —— 将其中一棵 $B_{k-1}$ 的根变为另一棵 $B_{k-1}$ 根的最左孩子。

```
B_0:  [x]

B_1:  [x]
        |
       [y]

B_2:  [x]
       / \
     [y] [z]
           |
          [w]

B_3:  [x]
      /|\ \
    [.][.][.][.]
    （B2, B1, B0 分别作为 B3 的孩子）
```

**二项树的性质**：
- $B_k$ 有 $2^k$ 个节点
- $B_k$ 的根有度数 $k$（恰好 $k$ 个子节点）
- $B_k$ 的高度为 $k$
- $B_k$ 在深度 $d$ 处有 $\binom{k}{d}$ 个节点（二项系数！这正是名字的由来）

### 36.4.2 二项堆结构：对应二进制表示

一个有 $n$ 个节点的二项堆，由若干不重复度数的二项树组成。对应 $n$ 的二进制表示：

$$n = \sum_{i} b_i \cdot 2^i \implies \text{二项堆包含} B_i \text{（对每个 } b_i = 1 \text{ 的 } i \text{）}$$

**例**：$n = 13 = (1101)_2 = 8 + 4 + 1$，二项堆包含 $B_3, B_2, B_0$（合计 $8+4+1=13$ 个节点）。

```python
@dataclass
class BinomialNode:
    """二项堆节点"""
    key: float
    degree: int = 0
    parent: Optional['BinomialNode'] = None
    child: Optional['BinomialNode'] = None   # 最左孩子
    sibling: Optional['BinomialNode'] = None  # 右兄弟

class BinomialHeap:
    """
    二项堆实现。
    根列表：各二项树的根，按 degree 升序排列（可含重复度，合并时整理）。
    所有操作严格 O(log n)（不需要摊销！）
    """
    def __init__(self):
        self.head: Optional[BinomialNode] = None   # 根列表头

    def minimum(self) -> float:
        """找到最小 key，O(log n)（遍历至多 log n 个根）"""
        if self.head is None:
            raise ValueError("empty heap")
        x = self.head
        min_key = x.key
        while x is not None:
            if x.key < min_key:
                min_key = x.key
            x = x.sibling
        return min_key
```

```cpp
struct BinomialNode {
    double key;
    int degree = 0;
    BinomialNode* parent  = nullptr;
    BinomialNode* child   = nullptr;   // 最左孩子
    BinomialNode* sibling = nullptr;   // 右兄弟
    explicit BinomialNode(double k) : key(k) {}
};

class BinomialHeap {
public:
    BinomialNode* head = nullptr;  // 根列表头（degree 升序）
    // ...
};
```

### 36.4.3 UNION/MERGE：类比二进制加法

二项堆的合并是其最优雅的操作。将两个二项堆的根列表按度数排序合并，然后"进位"（两个度数相同的 $B_k$ 合并为一个 $B_{k+1}$），完全类比二进制加法。

**例**：合并含 $B_1, B_3$ 的堆（$n_1 = 10 = (1010)_2$）和含 $B_0, B_1$ 的堆（$n_2 = 3 = (011)_2$）：

```
  1 0 1 0   (B3, B1)
+ 0 0 1 1   (B1, B0)
-----------
  1 1 0 1   (B3, B2, B0)  n = 13
```

```python
def _binomial_link(self, y: BinomialNode, z: BinomialNode):
    """
    将 y 链接为 z 的最左孩子（前提：y.key >= z.key）。
    类似二进制加法中将两个同位数字相加并进位。
    """
    y.parent = z
    y.sibling = z.child   # y 成为 z 的最左孩子
    z.child = y
    z.degree += 1

def union(self, other: 'BinomialHeap') -> 'BinomialHeap':
    """
    合并两个二项堆。O(log n) 严格。
    
    步骤：
    1. 将两个根列表按 degree 升序合并（类似归并排序）
    2. 从低度遍历，两个同度的树合并（进位），直到无重复度数
    """
    result = BinomialHeap()
    result.head = self._merge_roots(self.head, other.head)

    if result.head is None:
        return result

    prev = None
    x = result.head
    next_x = x.sibling

    while next_x is not None:
        if x.degree != next_x.degree or \
           (next_x.sibling is not None and next_x.sibling.degree == x.degree):
            # 无需合并，前进
            prev = x
            x = next_x
        else:
            # x 和 next_x 度数相同，合并（"进位"）
            if x.key <= next_x.key:
                x.sibling = next_x.sibling
                self._binomial_link(next_x, x)
            else:
                if prev is None:
                    result.head = next_x
                else:
                    prev.sibling = next_x
                self._binomial_link(x, next_x)
                x = next_x
        next_x = x.sibling
    return result

def _merge_roots(self, h1: Optional[BinomialNode],
                 h2: Optional[BinomialNode]) -> Optional[BinomialNode]:
    """
    将两个按 degree 升序的根列表归并。
    类似归并排序的 merge 步骤，O(log n)。
    """
    if h1 is None: return h2
    if h2 is None: return h1

    if h1.degree <= h2.degree:
        h1.sibling = self._merge_roots(h1.sibling, h2)
        return h1
    else:
        h2.sibling = self._merge_roots(h1, h2.sibling)
        return h2
```

```cpp
// 将两个升序根列表归并（O(log n)）
BinomialNode* merge_roots(BinomialNode* h1, BinomialNode* h2) {
    if (!h1) return h2;
    if (!h2) return h1;
    BinomialNode* head;
    if (h1->degree <= h2->degree) {
        head = h1; head->sibling = merge_roots(h1->sibling, h2);
    } else {
        head = h2; head->sibling = merge_roots(h1, h2->sibling);
    }
    return head;
}

void binomial_link(BinomialNode* y, BinomialNode* z) {
    // y 成为 z 的最左孩子（z.key <= y.key）
    y->parent = z;
    y->sibling = z->child;
    z->child = y;
    z->degree++;
}

BinomialNode* heap_union(BinomialNode* h1, BinomialNode* h2) {
    BinomialNode* head = merge_roots(h1, h2);
    if (!head) return nullptr;

    BinomialNode* prev = nullptr;
    BinomialNode* x    = head;
    BinomialNode* nx   = x->sibling;

    while (nx != nullptr) {
        if (x->degree != nx->degree ||
            (nx->sibling && nx->sibling->degree == x->degree)) {
            prev = x; x = nx;
        } else {
            if (x->key <= nx->key) {
                x->sibling = nx->sibling;
                binomial_link(nx, x);
            } else {
                if (!prev) head = nx;
                else prev->sibling = nx;
                binomial_link(x, nx);
                x = nx;
            }
        }
        nx = x->sibling;
    }
    return head;
}
```

<div data-component="BinomialHeapMerge"></div>

### 36.4.4 INSERT、EXTRACT-MIN、DECREASE-KEY

| 操作 | 实现方式 | 时间复杂度 |
|---|---|---|
| **INSERT** | 创建 $B_0$，UNION 入堆 | $O(\log n)$ 严格 |
| **MINIMUM** | 遍历至多 $\log n$ 个根 | $O(\log n)$ 严格 |
| **EXTRACT-MIN** | 找到最小根 $x$，移除 $x$，将其子树反转后 UNION | $O(\log n)$ 严格 |
| **DECREASE-KEY** | 减小 key，向上冒泡（类似二叉堆） | $O(\log n)$ 严格 |
| **DELETE** | DECREASE-KEY 到 $-\infty$ + EXTRACT-MIN | $O(\log n)$ 严格 |

```python
def insert(self, key: float) -> BinomialNode:
    """
    插入新节点：创建 B_0 后 UNION。
    O(log n) 严格（摊销为 O(1)，类似二进制加法中很少进多位位）
    """
    single = BinomialHeap()
    single.head = BinomialNode(key)
    merged = self.union(single)
    self.head = merged.head
    return self.head  # 简化，实际需要跟踪节点

def extract_min(self) -> float:
    """
    找到最小根 x，从根列表移除，
    将 x 的子节点（反转后）union 回堆。
    O(log n)。
    """
    if self.head is None:
        raise ValueError("empty")

    # 找最小根节点及其前驱
    prev_min = None
    min_node = self.head
    prev = None
    x = self.head.sibling
    while x is not None:
        if x.key < min_node.key:
            prev_min = prev
            min_node = x
        prev = x
        x = x.sibling

    # 从根列表移除 min_node
    if prev_min is None:
        self.head = min_node.sibling
    else:
        prev_min.sibling = min_node.sibling

    # 将 min_node 的子树反转（使其按 degree 升序）
    child_heap = BinomialHeap()
    child = min_node.child
    prev_child = None
    while child:
        nxt = child.sibling
        child.sibling = prev_child
        child.parent = None
        prev_child = child
        child = nxt
    child_heap.head = prev_child

    # UNION
    merged = self.union(child_heap)
    self.head = merged.head
    return min_node.key
```

```cpp
double extract_min_bin(BinomialNode*& head) {
    // 找最小根及其前驱
    BinomialNode* prevMin = nullptr, *minNode = head;
    BinomialNode* prev = nullptr, *x = head->sibling;
    while (x) {
        if (x->key < minNode->key) { prevMin = prev; minNode = x; }
        prev = x; x = x->sibling;
    }
    // 从根列表移除
    if (!prevMin) head = minNode->sibling;
    else prevMin->sibling = minNode->sibling;

    // 反转 minNode 的子链表（使 degree 升序）
    BinomialNode* childHead = nullptr, *ch = minNode->child;
    while (ch) {
        BinomialNode* nxt = ch->sibling;
        ch->sibling = childHead;
        ch->parent = nullptr;
        childHead = ch;
        ch = nxt;
    }
    head = heap_union(head, childHead);
    double result = minNode->key;
    delete minNode;
    return result;
}
```

### 36.4.5 二项堆 vs Fibonacci 堆对比

<div data-component="FibVsBinaryHeapPerf"></div>

| 维度 | 二项堆 | Fibonacci 堆 |
|---|---|---|
| INSERT | $O(\log n)$ 严格 | $O(1)$ 摊销 |
| EXTRACT-MIN | $O(\log n)$ 严格 | $O(\log n)$ 摊销 |
| DECREASE-KEY | $O(\log n)$ 严格 | **$O(1)$ 摊销** |
| UNION | $O(\log n)$ 严格 | $O(1)$ 摊销 |
| 实现复杂度 | 中等 | 极高 |
| 常数因子 | 较小 | 较大（多指针） |
| 缓存行为 | 较好（树形结构相对集中） | 较差（多树分散） |
| 工程应用 | 少量使用（理论教学） | 几乎不用 |
| 适用场景 | 需要严格最坏保证时 | 理论算法研究 |

---

## 36.5 摊销分析回顾（结合第 35 章）

### 36.5.1 Fibonacci 堆的势能函数与各操作分析

**势能函数**（第 35.5 节详述）：
$$\Phi(H) = t(H) + 2 \times m(H)$$

**各操作的摊销代价推导**：

**INSERT**（实际代价 1，$t$ 增 1，$m$ 不变）：
$$\hat{c} = 1 + \Delta\Phi = 1 + 1 = 2 = O(1)$$

**EXTRACT-MIN**（实际代价 $O(t(H) + D(n))$，CONSOLIDATE 后 $t$ 降至 $\leq D(n)+1$）：
$$\hat{c} = O(t + D) + (D(n)+1) - t - 2m = O(D(n)) = O(\log n)$$

**DECREASE-KEY**（级联裁剪 $c$ 次）：
- 实际代价：$c+1$（$c$ 次 CUT + 1 次改 key）
- 每次 CUT 后：$t$ 增 1，被裁剪的已标记节点 $m$ 减 1；共 $c$ 次：$\Delta t = c$，$\Delta m = -c+1$（最后一次变 true +1）
- $\Delta\Phi = c + 2(-c+1) = 2 - c$
- $\hat{c} = (c+1) + (2-c) = 3 = O(1)$ ✅

> **关键洞察**：级联裁剪中每次切掉一个已标记节点，$m$ 减少 1，势能减少 2，恰好补偿了实际的 1 步代价。这就是 mark 机制为何能使 DECREASE-KEY 摊销 $O(1)$。

### 36.5.2 若取消 mark 机制会怎样？

**思考题答案**：若取消 `mark`，DECREASE-KEY 中不做级联裁剪（只做一次 CUT）：

优点：操作更简单，每次 DECREASE-KEY 确实是 $O(1)$ 实际代价。

**问题**：节点可以失去任意多个孩子！度为 $k$ 的节点子树大小可能只有 $k+1$（不再是 Fibonacci 下界），导致 $D(n)$ 可以是 $O(n)$ 而非 $O(\log n)$。CONSOLIDATE 代价退化为 $O(n)$，EXTRACT-MIN 摊销变为 $O(n)$。

**结论**：mark + 级联裁剪 **必不可少**，它保证了子树大小的 Fibonacci 下界，进而保证了 $D(n) = O(\log n)$。

---

## 36.6 工程实践建议

在实际工作中选择优先队列：

| 场景 | 推荐实现 | 原因 |
|---|---|---|
| 通用用途 | Python `heapq` / C++ `priority_queue` | 最快，自带语言支持 |
| 需要任意元素删除 | `heapq` + 懒删除标记 | 简单高效 |
| Dijkstra（稀疏图） | Python `heapq` / C++ `priority_queue` | 缓存友好，常数小 |
| Dijkstra（超密集图，$E \sim V^2$） | Fibonacci 堆（如需理论最优） | 仅在 $E \gg V$ 时有优势 |
| 需要执行次序确定性保证 | 二项堆 / D-堆（D-ary Heap） | 严格最坏 vs 摊销 |
| 竞赛 DECREASE-KEY | 使用`indexed priority queue`（索引堆） | 实现简单，常数小 |

**索引堆（Indexed Priority Queue）**：二叉堆 + 位置数组，支持 $O(\log n)$ 的 decrease_key，工程中是 Fibonacci 堆的实用替代品：

```python
class IndexedMinHeap:
    """
    索引最小堆：支持 O(log n) 的 decrease_key。
    工程中最常用的 Dijkstra 优先队列。
    适合元素数量固定（如 Dijkstra 的 V 个顶点）。
    """
    def __init__(self, capacity: int):
        self.cap = capacity
        self.heap = []                   # [(key, idx), ...]
        self.pos = [-1] * capacity       # pos[idx] = 该 idx 在 heap 中的位置

    def push(self, idx: int, key: float):
        """插入 (key, idx)"""
        import heapq
        self.pos[idx] = len(self.heap)
        self.heap.append((key, idx))
        self._sift_up(len(self.heap) - 1)

    def decrease_key(self, idx: int, new_key: float):
        """将 idx 的 key 降低为 new_key，维护堆性质"""
        p = self.pos[idx]
        if p == -1 or self.heap[p][0] <= new_key:
            return   # 没有改变或 idx 不在堆中
        self.heap[p] = (new_key, idx)
        self._sift_up(p)

    def pop_min(self) -> tuple[float, int]:
        """弹出最小 (key, idx)"""
        if not self.heap:
            raise IndexError("empty")
        # 与最后元素交换，弹出顶部
        self._swap(0, len(self.heap) - 1)
        key, idx = self.heap.pop()
        self.pos[idx] = -1
        if self.heap:
            self._sift_down(0)
        return key, idx

    def _sift_up(self, i: int):
        while i > 0:
            parent = (i - 1) // 2
            if self.heap[parent][0] > self.heap[i][0]:
                self._swap(parent, i)
                i = parent
            else:
                break

    def _sift_down(self, i: int):
        n = len(self.heap)
        while True:
            smallest = i
            for child in [2*i+1, 2*i+2]:
                if child < n and self.heap[child][0] < self.heap[smallest][0]:
                    smallest = child
            if smallest == i:
                break
            self._swap(i, smallest)
            i = smallest

    def _swap(self, i: int, j: int):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        self.pos[self.heap[i][1]] = i
        self.pos[self.heap[j][1]] = j
```

```cpp
#include <vector>
#include <cassert>
#include <limits>

// 索引最小堆：O(log n) decrease_key，工程上替代 Fibonacci 堆
class IndexedMinHeap {
    int cap;
    std::vector<std::pair<double,int>> heap;   // (key, idx)
    std::vector<int> pos;                       // pos[idx] = heap 中位置，-1 表示不在堆中

    void swap_(int i, int j) {
        std::swap(heap[i], heap[j]);
        pos[heap[i].second] = i;
        pos[heap[j].second] = j;
    }

    void sift_up(int i) {
        while (i > 0) {
            int p = (i-1)/2;
            if (heap[p].first > heap[i].first) { swap_(p, i); i = p; }
            else break;
        }
    }

    void sift_down(int i) {
        int n = heap.size();
        while (true) {
            int s = i;
            for (int c : {2*i+1, 2*i+2})
                if (c < n && heap[c].first < heap[s].first) s = c;
            if (s == i) break;
            swap_(i, s); i = s;
        }
    }

public:
    explicit IndexedMinHeap(int cap) : cap(cap), pos(cap, -1) {}

    void push(int idx, double key) {
        pos[idx] = heap.size();
        heap.push_back({key, idx});
        sift_up(pos[idx]);
    }

    void decrease_key(int idx, double new_key) {
        int p = pos[idx];
        if (p == -1 || heap[p].first <= new_key) return;
        heap[p].first = new_key;
        sift_up(p);
    }

    std::pair<double,int> pop_min() {
        assert(!heap.empty());
        swap_(0, heap.size()-1);
        auto [key, idx] = heap.back();
        heap.pop_back();
        pos[idx] = -1;
        if (!heap.empty()) sift_down(0);
        return {key, idx};
    }

    bool empty() const { return heap.empty(); }
};
```

---

## 36.7 常见错误与陷阱

### 陷阱 1：CONSOLIDATE 中 D(n) 的估计不够大

`A` 数组的大小必须 $\geq D(n) + 1 = \lfloor \log_\phi n \rfloor + 2$。若分配太小，CONSOLIDATE 时会数组越界。常见错误：用 $\log_2 n$ 而非 $\log_\phi n \approx 1.44 \log_2 n$ 来估计上界。

**正确做法**：`max_degree = int(log(n) / log(1.618)) + 2`（留 2 的余量）。

### 陷阱 2：mark 机制误用

`mark` 只在节点失去孩子（DECREASE-KEY 触发 CUT）时置 true；节点作为根时 `mark` 始终为 false；节点被 CUT 后 `mark` 清零。错误：在 EXTRACT-MIN 后忘记对加入根列表的子节点清零 mark。

**发现规律**：只有三处会修改 `mark`：
- `_cut()`：$x.mark = \text{False}$（新根清零）
- `_cascading_cut()`：$y.mark = \text{True}$（第一次失去孩子）
- `_fib_heap_link()`：$y.mark = \text{False}$（CONSOLIDATE 链接时清零）

### 陷阱 3：二项树的 EXTRACT-MIN 中忘记反转子链表

二项树 $B_k$ 的子节点列表是按 degree 降序排列的（$B_{k-1}, B_{k-2}, \ldots, B_0$）。EXTRACT-MIN 后要将这些子节点加入二项堆根列表，但根列表要求 degree **升序**。必须**反转**子链表，否则 UNION 的归并步骤会出错。

### 陷阱 4：用 DECREASE-KEY 实现某元素的 INSERT

有人误以为 Fibonacci 堆中可以"先 INSERT 一个占位节点，再 DECREASE-KEY 到真实 key"。这虽然可行，但 INSERT 本身就是 $O(1)$，无需此迂回。更重要的是，`decrease_key` 要求新 key 必须 ≤ 原 key，方向不能错。

---

## 36.8 面试高频考点与思考题

### 面试考点

1. **Fibonacci 堆各操作摊销复杂度表**（必背）
2. **为什么 DECREASE-KEY 是 $O(1)$ 摊销而非 $O(1)$ 严格**：级联裁剪最坏 $O(n)$ 实际，但势能减少补偿
3. **mark 字段的作用**：防止任意节点失去过多孩子，保证 Fibonacci 子树大小下界
4. **Dijkstra 时间复杂度表**：二叉堆 $O((V+E)\log V)$ vs Fibonacci 堆 $O(V \log V + E)$
5. **二项堆合并的"二进制加法"类比**

### 思考题

**思考题 1**：Fibonacci 堆的 DECREASE-KEY 摊销 $O(1)$ 的关键是"级联裁剪保持每个节点子节点数量受限"。如果去掉 `mark` 机制（即不做级联裁剪），DECREASE-KEY 还能保证 $O(1)$ 摊销吗？（分析 EXTRACT-MIN 会变成什么）

**参考答案**：不能。没有 mark 机制，节点可以失去任意多个孩子，子树大小不再有 Fibonacci 下界。节点度数上界 $D(n)$ 可以是 $O(n)$，CONSOLIDATE 循环次数变为 $O(n)$，EXTRACT-MIN 摊销代价退化为 $O(n)$。

**思考题 2**：为 Fibonacci 堆添加 INCREASE-KEY 操作（增大某节点的 key），能否在 $O(\log n)$ 摊销时间内实现？

**参考答案**：可以。删除该节点（$O(\log n)$），插入新 key 的节点（$O(1)$），总计 $O(\log n)$。不能比这更快，因为 INCREASE-KEY 可能违反堆性质，需要将节点押送到正确位置。

**思考题 3**：D-堆（D-ary Heap，d 叉堆）中 DECREASE-KEY 时间为 $O(\log_d n)$；INSERT 是 $O(\log_d n)$；EXTRACT-MIN 是 $O(d \log_d n)$（从 D 个孩子中选最小）。对 Dijkstra，最优的 $d$ 是多少？

**参考答案**：令 EXTRACT-MIN（V 次）与 DECREASE-KEY（E 次）代价相等：$V \cdot d \log_d n = E \cdot \log_d n$，得 $d = E/V$（图的平均出度）。最优 D-堆 Dijkstra 的总复杂度 $O(E \log_{E/V} V)$，对稀疏图约为 $O(E \log V)$，对密集图趋近 $O(E)$——但实现远比 Fibonacci 堆简单！

---

## 章节小结

| 知识点 | 一句话要点 |
|---|---|
| Fibonacci 堆结构 | 多棵最小堆有序树 + 根双向循环链表，维护 min 指针 |
| 懒惰策略 | INSERT/UNION 只加入根列表，拖到 EXTRACT-MIN 才整理 |
| CONSOLIDATE | 合并同度树，类比二进制加法进位，确保至多 $D(n)$ 棵树 |
| DECREASE-KEY | CUT + CASCADING-CUT；mark 防止节点孩子数量无限减少 |
| 度数界 $D(n)$ | $\leq \log_\phi n$，来源于 Fibonacci 子树大小下界 |
| 理论优势 | DECREASE-KEY O(1) 使 Dijkstra 降至 $O(V\log V + E)$ |
| 工程劣势 | 大常数、缓存差、实现复杂，实战不如二叉堆/索引堆 |
| 二项堆 | 所有操作严格 $O(\log n)$，UNION 类比二进制加法 |
| 索引堆 | 工程上最实用的 DECREASE-KEY 数据结构，$O(\log n)$ 严格 |

---

## 参考资料

- CLRS 第4版 Chapter 19（Fibonacci Heaps）
- Fredman & Tarjan 1987：*Fibonacci Heaps and Their Uses in Improved Network Optimization Algorithms*（原始论文，定义了 Fibonacci 堆）
- CLRS 第4版 Chapter 19.4（Bounding the maximum degree）
- MIT 6.046 Lecture 4：Fibonacci Heaps（Jonathan Kelner，网络公开）
- Larkin, Sen & Tarjan 2014：*A Back-to-Basics Empirical Study of Priority Queues*（实测 Fibonacci 堆 vs 二叉堆性能）
- CP-algorithms.com：Fibonacci Heap 实现参考

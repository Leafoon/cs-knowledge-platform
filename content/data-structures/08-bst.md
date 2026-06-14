# Chapter 8: 二叉搜索树（Binary Search Tree）

## 学习目标

- 掌握 BST 的核心不变量及所有基本操作（查找 / 插入 / 删除 / 前驱 / 后继）的正确实现
- 理解随机 BST 期望高度 O(log n)，认识退化问题的根本原因
- 能写出"传递范围约束"验证 BST 合法性的正确算法
- 为平衡 BST（AVL / 红黑树，Chapter 10）打下坚实基础

> **典型 LeetCode**：#98 验证BST、#230 BST第K小、#450 删除BST节点、#700 BST搜索、#701 BST插入、#538 BST转累加树

---

## 8.1 BST 定义与核心性质

### 8.1.1 BST 核心不变量

一棵**二叉搜索树**（Binary Search Tree）是满足如下条件的二叉树：

$$\forall x \in \text{Node}: \quad \text{max}(\text{left}(x)) < \text{key}(x) < \text{min}(\text{right}(x))$$

这条不变量是**全局性**的：节点 $x$ 左子树中**所有**节点均严格小于 $\text{key}(x)$，右子树中**所有**节点均严格大于 $\text{key}(x)$。

> ⚠️ **常见误解**：仅比较节点与其直接子节点是不够的！
>
> ```
>       10
>      /  \\
>     5   15
>    / \\
>   3   12    ← 12 > 10，违反全局不变量！
> ```

### 8.1.2 中序遍历输出有序序列

**定理**：对 BST 做中序遍历（左→根→右），输出序列严格有序递增。

**证明（数学归纳法）**：

- **基例**：空树或单节点，显然有序。
- **归纳步骤**：设左子树中序输出 $L$，右子树中序输出 $R$，根键值 $k$。由 BST 不变量，$L$ 中所有元素 $< k <$ $R$ 中所有元素，归纳假设 $L$、$R$ 各自递增，故 $L + k + R$ 严格递增。$\square$

**核心价值**：BST 是排好序的动态结构——中序遍历 = 有序输出，插入/删除均 O(h)（有序数组需 O(n) 移位）。

### 8.1.3 BST 高度与操作复杂度

| 情形 | 高度 h | 操作时间 |
|------|--------|----------|
| 完全平衡 | $\Theta(\log n)$ | $\Theta(\log n)$ |
| 随机输入期望 | $\approx 2.77 \log n$ | $\Theta(\log n)$ 期望 |
| 有序输入（最坏） | $n-1$ | $\Theta(n)$ |

---

## 8.2 BST 基本操作

### 8.2.1 查找 TREE-SEARCH

**CLRS 伪代码**：

```
TREE-SEARCH(x, k):
    if x == NIL or k == x.key:  return x
    if k < x.key:  return TREE-SEARCH(x.left, k)
    else:          return TREE-SEARCH(x.right, k)
```

**双语实现（迭代版本，O(1) 额外空间）**：

```python
class BSTNode:
    def __init__(self, key):
        self.key = key
        self.left = self.right = self.parent = None

def search(root, k):
    while root and root.key != k:
        root = root.left if k < root.key else root.right
    return root   # None = 未找到
```

```cpp
struct BSTNode {
    int key;
    BSTNode *left, *right, *parent;
    BSTNode(int k) : key(k), left(nullptr), right(nullptr), parent(nullptr) {}
};

BSTNode* search(BSTNode* root, int k) {
    while (root && root->key != k)
        root = (k < root->key) ? root->left : root->right;
    return root;  // nullptr = 未找到
}
```

**复杂度**：时间 O(h)，迭代版本 O(1) 额外空间（递归版本 O(h) 栈深）。

### 8.2.2 插入 TREE-INSERT

**核心原则**：新节点总是作为**叶子**插入，不旋转，不调整。

**CLRS 伪代码**（含 parent 指针）：

```
TREE-INSERT(T, z):
    y = NIL;  x = T.root
    while x != NIL:
        y = x
        if z.key < x.key: x = x.left
        else:              x = x.right
    z.parent = y
    if y == NIL:        T.root = z
    elif z.key < y.key: y.left = z
    else:               y.right = z
```

**双语实现**：

```python
def insert(root, key):
    if root is None:
        return BSTNode(key)
    parent, cur = None, root
    while cur:
        parent = cur
        cur = cur.left if key < cur.key else cur.right
    node = BSTNode(key)
    node.parent = parent
    if key < parent.key:
        parent.left = node
    else:
        parent.right = node
    return root
```

```cpp
BSTNode* insert(BSTNode* root, int key) {
    BSTNode* node = new BSTNode(key);
    if (!root) return node;
    BSTNode *parent = nullptr, *cur = root;
    while (cur) {
        parent = cur;
        cur = (key < cur->key) ? cur->left : cur->right;
    }
    node->parent = parent;
    if (key < parent->key) parent->left = node;
    else                   parent->right = node;
    return root;
}
```

### 8.2.3 最小值与最大值

BST 最小值在最左路径末端，最大值在最右路径末端，复杂度均 O(h)。

```python
def minimum(node):
    while node.left:
        node = node.left
    return node

def maximum(node):
    while node.right:
        node = node.right
    return node
```

```cpp
BSTNode* minimum(BSTNode* node) {
    while (node->left) node = node->left;
    return node;
}

BSTNode* maximum(BSTNode* node) {
    while (node->right) node = node->right;
    return node;
}
```

### 8.2.4 前驱与后继

**后继（Successor）**：中序顺序中紧随其后的节点，分两种情形：

- **情形 1**：$x$ 有右子树 → 后继 = 右子树的最小值
- **情形 2**：$x$ 无右子树 → 向上走，直到"从左侧进入祖先"

**前驱（Predecessor）**：完全对称。

```python
def successor(x):
    if x.right:
        return minimum(x.right)      # 情形 1
    y = x.parent
    while y and x == y.right:        # 情形 2
        x, y = y, y.parent
    return y

def predecessor(x):
    if x.left:
        return maximum(x.left)       # 情形 1
    y = x.parent
    while y and x == y.left:         # 情形 2
        x, y = y, y.parent
    return y
```

```cpp
BSTNode* successor(BSTNode* x) {
    if (x->right) return minimum(x->right);
    BSTNode* y = x->parent;
    while (y && x == y->right) { x = y; y = y->parent; }
    return y;
}

BSTNode* predecessor(BSTNode* x) {
    if (x->left) return maximum(x->left);
    BSTNode* y = x->parent;
    while (y && x == y->left) { x = y; y = y->parent; }
    return y;
}
```

<div data-component="BSTPredSuccFinder"></div>

### 8.2.5 删除 TREE-DELETE：三种情形

| 情形 | 描述 | 处理方式 |
|-----|------|---------|
| 情形 1 | 叶节点（无子树） | 直接删除 |
| 情形 2 | 只有一棵子树 | 用该子树替换 $z$ |
| 情形 3 | 两棵子树均存在 | 用**中序后继**替换键值，再删后继 |

**TRANSPLANT 辅助函数**：用子树 $v$ 替换子树 $u$ 的位置。

```python
def transplant(T, u, v):
    if u.parent is None:        T['root'] = v
    elif u == u.parent.left:    u.parent.left = v
    else:                       u.parent.right = v
    if v: v.parent = u.parent

def tree_delete(T, z):
    if z.left is None:
        transplant(T, z, z.right)
    elif z.right is None:
        transplant(T, z, z.left)
    else:
        y = minimum(z.right)
        if y.parent != z:
            transplant(T, y, y.right)
            y.right = z.right
            y.right.parent = y
        transplant(T, z, y)
        y.left = z.left
        y.left.parent = y
```

```cpp
void transplant(BSTNode*& root, BSTNode* u, BSTNode* v) {
    if (!u->parent)                 root = v;
    else if (u == u->parent->left)  u->parent->left = v;
    else                            u->parent->right = v;
    if (v) v->parent = u->parent;
}

void tree_delete(BSTNode*& root, BSTNode* z) {
    if (!z->left) {
        transplant(root, z, z->right);
    } else if (!z->right) {
        transplant(root, z, z->left);
    } else {
        BSTNode* y = minimum(z->right);
        if (y->parent != z) {
            transplant(root, y, y->right);
            y->right = z->right;
            y->right->parent = y;
        }
        transplant(root, z, y);
        y->left = z->left;
        y->left->parent = y;
    }
    delete z;
}
```

<div data-component="BSTOperationsVisualizer"></div>

---

## 8.3 随机 BST 的期望高度分析

### 8.3.1 期望高度 ≈ 2.77 log n

**定理**（Reed 2003）：将 $n$ 个不同键以随机顺序插入 BST，期望高度为：

$$E[h] \leq 4.311 \ln n$$

实验上通常接近 $2.77 \ln n$。

**简化论证**：令 $X_n = 2^{h_n}$，对高度指数化以便递推：

- 第一个插入节点成为根，等可能为第 $k$ 大键（$1 \le k \le n$）
- $E[X_n] = \frac{2}{n}\sum_{j=0}^{n-1}E[X_j]$
- 用替换法证明 $E[X_n] = O(n^c)$（$c \approx 2.41$），从而 $E[h] = O(\log n)$

### 8.3.2 快速排序与随机 BST 的等价性

> 将 $n$ 个随机键插入 BST 的过程，与对同一序列做快速排序，在比较次数上**完全等价**。

**论证**：第一个插入的键成为根（= 快排轴），小于根的进入左子树（= 左分区），大于的进入右子树（= 右分区），递归同理。因此随机 BST 期望比较总次数 $= 2n \ln n$（等于随机快排期望）。

**Treap** 正是利用了该等价性——给每个节点赋随机优先级并维护堆性质，保证期望 O(log n)。

### 8.3.3 最坏情况退化

<div data-component="BSTDegenerationDemo"></div>

| 输入模式 | BST 形态 | 高度 |
|---------|---------|------|
| 严格升序 1,2,…,n | 右斜链 | $n-1$ |
| 严格降序 n,…,1 | 左斜链 | $n-1$ |
| 随机顺序 | 近似平衡 | $\approx 2.77 \log n$ |

---

## 8.4 BST 验证（LeetCode #98）

验证 BST 的**正确方法**是传递 $(lo, hi)$ 范围约束：

```python
def isValidBST(root, lo=float('-inf'), hi=float('inf')):
    if not root:
        return True
    if not (lo < root.val < hi):
        return False
    return (isValidBST(root.left, lo, root.val) and
            isValidBST(root.right, root.val, hi))
```

```cpp
bool isValidBST(TreeNode* root,
                long long lo = LLONG_MIN, long long hi = LLONG_MAX) {
    if (!root) return true;
    if (root->val <= lo || root->val >= hi) return false;
    return isValidBST(root->left,  lo, root->val) &&
           isValidBST(root->right, root->val, hi);
}
```

**为什么只比较父子节点是错的**：节点在更深层可能违反祖先的约束，只有从根传递范围才能检测。

另一种等价方法——**中序遍历验证严格递增**：

```python
def isValidBST_inorder(root):
    prev = [float('-inf')]
    def inorder(node):
        if not node: return True
        if not inorder(node.left): return False
        if node.val <= prev[0]: return False
        prev[0] = node.val
        return inorder(node.right)
    return inorder(root)
```

```cpp
bool isValidBST_inorder(TreeNode* root) {
    long long prev = LLONG_MIN;
    function<bool(TreeNode*)> inorder = [&](TreeNode* node) -> bool {
        if (!node) return true;
        if (!inorder(node->left)) return false;
        if (node->val <= prev) return false;
        prev = node->val;
        return inorder(node->right);
    };
    return inorder(root);
}
```

---

## 8.5 扩展：顺序统计树

### 8.5.1 动态有序集合对比

| 数据结构 | 查找 | 插入 | 删除 | 有序输出 |
|---------|------|------|------|---------|
| 有序数组 | O(log n) 二分 | O(n) 移位 | O(n) 移位 | O(n) |
| 链表 | O(n) | O(1) | O(1) | O(n) |
| BST（平衡） | O(log n) | O(log n) | O(log n) | O(n) |

### 8.5.2 OS-SELECT 与 OS-RANK

**顺序统计树** = BST + 每节点 `size` 字段，维护不变量：

$$\text{size}(x) = \text{size}(x.\text{left}) + \text{size}(x.\text{right}) + 1$$

```python
def os_select(node, k):
    if node is None:
        return None
    ls = node.left.size if node.left else 0
    r = ls + 1
    if k == r:   return node
    elif k < r:  return os_select(node.left, k)
    else:        return os_select(node.right, k - r)

def os_rank(node):
    rank = (node.left.size if node.left else 0) + 1
    cur = node
    while cur.parent:
        if cur == cur.parent.right:
            rank += (cur.parent.left.size if cur.parent.left else 0) + 1
        cur = cur.parent
    return rank
```

```cpp
int os_select(OSTNode* node, int k) {
    int ls = node->left ? node->left->size : 0;
    int r  = ls + 1;
    if (k == r)  return node->key;
    if (k <  r)  return os_select(node->left,  k);
    return       os_select(node->right, k - r);
}

int os_rank(OSTNode* node) {
    int rank = (node->left ? node->left->size : 0) + 1;
    OSTNode* cur = node;
    while (cur->parent) {
        if (cur == cur->parent->right)
            rank += (cur->parent->left ? cur->parent->left->size : 0) + 1;
        cur = cur->parent;
    }
    return rank;
}
```

<div data-component="AugmentedBSTDemo"></div>

---

## 8.6 经典 LeetCode 精讲

### #700 BST 搜索

```python
def searchBST(root, val):
    while root and root.val != val:
        root = root.left if val < root.val else root.right
    return root
```

```cpp
TreeNode* searchBST(TreeNode* root, int val) {
    while (root && root->val != val)
        root = (val < root->val) ? root->left : root->right;
    return root;
}
```

### #701 BST 插入

```python
def insertIntoBST(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insertIntoBST(root.left, val)
    else:
        root.right = insertIntoBST(root.right, val)
    return root
```

```cpp
TreeNode* insertIntoBST(TreeNode* root, int val) {
    if (!root) return new TreeNode(val);
    if (val < root->val) root->left  = insertIntoBST(root->left,  val);
    else                 root->right = insertIntoBST(root->right, val);
    return root;
}
```

### #450 删除 BST 节点

```python
def deleteNode(root, key):
    if not root:
        return None
    if key < root.val:
        root.left = deleteNode(root.left, key)
    elif key > root.val:
        root.right = deleteNode(root.right, key)
    else:
        if not root.left:  return root.right
        if not root.right: return root.left
        succ = root.right
        while succ.left: succ = succ.left
        root.val = succ.val
        root.right = deleteNode(root.right, succ.val)
    return root
```

```cpp
TreeNode* deleteNode(TreeNode* root, int key) {
    if (!root) return nullptr;
    if      (key < root->val) root->left  = deleteNode(root->left,  key);
    else if (key > root->val) root->right = deleteNode(root->right, key);
    else {
        if (!root->left)  return root->right;
        if (!root->right) return root->left;
        TreeNode* succ = root->right;
        while (succ->left) succ = succ->left;
        root->val = succ->val;
        root->right = deleteNode(root->right, succ->val);
    }
    return root;
}
```

### #98 验证二叉搜索树

```python
def isValidBST(root):
    def check(node, lo, hi):
        if not node: return True
        if not (lo < node.val < hi): return False
        return check(node.left, lo, node.val) and check(node.right, node.val, hi)
    return check(root, float('-inf'), float('inf'))
```

```cpp
bool isValidBST(TreeNode* root,
                long long lo = LLONG_MIN, long long hi = LLONG_MAX) {
    if (!root) return true;
    if (root->val <= lo || root->val >= hi) return false;
    return isValidBST(root->left,  lo, root->val) &&
           isValidBST(root->right, root->val, hi);
}
```

### #230 BST 第 K 小

```python
def kthSmallest(root, k):
    stack, node, count = [], root, 0
    while stack or node:
        while node:
            stack.append(node)
            node = node.left
        node = stack.pop()
        count += 1
        if count == k: return node.val
        node = node.right
```

```cpp
int kthSmallest(TreeNode* root, int k) {
    stack<TreeNode*> st;
    int count = 0;
    while (root || !st.empty()) {
        while (root) { st.push(root); root = root->left; }
        root = st.top(); st.pop();
        if (++count == k) return root->val;
        root = root->right;
    }
    return -1;
}
```

### #538 BST 转累加树

**思路**：反向中序遍历（右→根→左），维护累积和。

```python
def convertBST(root):
    acc = [0]
    def reverse_inorder(node):
        if not node: return
        reverse_inorder(node.right)
        acc[0] += node.val
        node.val = acc[0]
        reverse_inorder(node.left)
    reverse_inorder(root)
    return root
```

```cpp
class Solution {
    int acc = 0;
public:
    TreeNode* convertBST(TreeNode* root) {
        if (!root) return nullptr;
        convertBST(root->right);
        acc += root->val;
        root->val = acc;
        convertBST(root->left);
        return root;
    }
};
```

---

## 8.7 总结与复习

### 操作时间复杂度汇总

| 操作 | 最坏（退化链） | 平均（随机） | 平衡 BST |
|------|-------------|------------|---------|
| SEARCH | O(n) | O(log n) | O(log n) |
| INSERT | O(n) | O(log n) | O(log n) |
| DELETE | O(n) | O(log n) | O(log n) |
| MIN / MAX | O(n) | O(log n) | O(log n) |
| PRED / SUCC | O(n) | O(log n) | O(log n) |
| INORDER | O(n) | O(n) | O(n) |

### 常见错误汇总

| 错误 | 正确做法 |
|------|---------|
| 验证 BST 只比较父子节点 | 传递 (lo, hi) 范围约束，或中序验证严格递增 |
| 删除双子节点时直接删根 | 用中序后继键值覆盖，再递归删后继 |
| 顺序统计树插入后忘记更新 size | 沿插入路径回溯，逐一更新祖先 size |

### 💡 思考题

1. **快速排序等价**：为什么随机 BST 的构建与快速排序等价？从"第一个键成为根/轴"的角度推理。
2. **Treap 本质**：Treap 给每个节点赋随机优先级并维护堆性质，为何等价于随机插入的 BST？
3. **顺序统计树代价**：更新所有祖先 size 的额外开销是多少？最坏情况下是否影响整体复杂度？
4. **有序数组建最矮 BST**：给排好序的数组，如何构建高度最小的 BST？（LeetCode #108）

**参考资料**：CLRS 第 4 版 Chapter 12（BST）、Chapter 14（顺序统计树）；MIT 6.006 Lecture 5；LeetCode #98 #230 #450 #700 #701 #538

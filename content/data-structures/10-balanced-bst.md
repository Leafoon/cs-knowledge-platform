# Chapter 10: 平衡二叉搜索树

> **学习目标**：理解 BST 退化的根本原因；掌握 AVL 树与红黑树的完整操作逻辑；了解 Treap 与 Splay 树的设计哲学；能在具体工程场景中做出合理的数据结构选型。

---

## 10.1 平衡的必要性

### 10.1.1 BST 退化回顾

在第 8 章我们学习了**二叉搜索树（BST）**，它的查找、插入、删除操作的时间复杂度均为 $O(h)$，其中 $h$ 是树的高度。对于随机插入序列，期望高度为 $O(\log n)$，性能优秀。然而，若插入序列**有序**（如 $1, 2, 3, \ldots, n$），BST 会退化为一条右斜链，高度变为 $O(n)$，与线性表无异。

这种退化不是小概率事件——数据库查询返回的结果往往是有序的；用户按时间顺序插入日志记录；竞赛题目的特殊构造数据都可能触发此情形。

**退化示例**：依次插入 $1, 2, 3, 4, 5$ 后的 BST：

```
1
 \
  2
   \
    3
     \
      4
       \
        5
```

高度 $h = n - 1 = 4$，此时 `search(5)` 需要比较 5 次，与顺序查找相同。

### 10.1.2 高度上界的形式化定义

一棵含 $n$ 个节点的**平衡二叉搜索树**保证高度满足：

$$h = O(\log n)$$

更精确地，不同结构有不同的上界常数：

- **AVL 树**：$h \leq 1.44 \log_2(n+2) - 0.328$
- **红黑树**：$h \leq 2 \log_2(n+1)$
- **Treap**：$E[h] = O(\log n)$（期望高度）
- **Splay 树**：$h$ 最坏 $O(n)$，但任意 $m$ 次操作摊销 $O(m \log n)$

所有这些结构都通过不同机制维护"高度受控"这一核心不变量，从而保证所有操作的**最坏情况**（或摊销）时间复杂度为 $O(\log n)$。

### 10.1.3 各类平衡 BST 对比

下表总结了六种常见平衡搜索结构的关键指标。点击行可查看详细说明：

<div data-component="BalancedBSTComparison"></div>

**选型经验法则**：
- **频繁查询、少量更新** → AVL 树（高度最矮，查询最快）
- **读写均衡的通用场景** → 红黑树（insert/delete 旋转次数少）
- **竞赛、需要 Split/Merge** → Treap（实现简单，随机化保证期望性能）
- **时间局部性明显（最近访问的元素频繁再访问）** → Splay 树
- **磁盘存储、数据库索引** → B/B+ 树（更高扇出，减少 I/O）
- **高并发环境** → 跳表（无锁友好）

---

## 10.2 AVL 树

AVL 树由 Adelson-Velsky 和 Landis 在 1962 年提出，是**最早的自平衡 BST**。它通过维护每个节点的平衡因子来保证树高不超过 $1.44 \log_2 n$。

### 10.2.1 平衡因子

**定义**：节点 $x$ 的**平衡因子（Balance Factor, BF）**定义为其左子树高度与右子树高度之差：

$$\text{BF}(x) = h(\text{left}(x)) - h(\text{right}(x))$$

其中空子树的高度定义为 $-1$，叶节点高度为 $0$。

**AVL 性质（AVL Property）**：对于树中**每一个**节点 $x$，必须满足：

$$\text{BF}(x) \in \{-1, 0, +1\}$$

若某次插入或删除导致某节点 $\text{BF} \in \{-2, +2\}$，则称该节点**失衡**，需通过**旋转**恢复。

### 10.2.2 节点结构与高度更新

```python
class AVLNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 0  # 叶节点高度为 0

def get_height(node):
    """获取节点高度（None 节点高度为 -1）"""
    return node.height if node else -1

def update_height(node):
    """更新节点高度"""
    node.height = 1 + max(get_height(node.left), get_height(node.right))

def get_bf(node):
    """获取平衡因子"""
    if node is None:
        return 0
    return get_height(node.left) - get_height(node.right)
```
```cpp
struct AVLNode {
    int key;
    AVLNode* left;
    AVLNode* right;
    int height;
    AVLNode(int k) : key(k), left(nullptr), right(nullptr), height(0) {}
};

int getHeight(AVLNode* node) {
    return node ? node->height : -1;
}

void updateHeight(AVLNode* node) {
    if (node)
        node->height = 1 + max(getHeight(node->left), getHeight(node->right));
}

int getBF(AVLNode* node) {
    if (!node) return 0;
    return getHeight(node->left) - getHeight(node->right);
}
```

### 10.2.3 四种旋转

旋转是 AVL 树（以及红黑树）恢复平衡的核心操作。一次旋转的时间复杂度为 $O(1)$，且**不改变中序遍历顺序**（即 BST 性质保持不变）。

根据失衡节点 $z$（BF = ±2）和其失衡子树的位置，有四种情形：

| 失衡类型 | 触发条件 | 解决方案 |
|---------|---------|---------|
| **LL（左左）** | $z$ 的 BF = +2，$z$ 的左孩子 BF ≥ 0 | 对 $z$ 做一次右旋 |
| **RR（右右）** | $z$ 的 BF = -2，$z$ 的右孩子 BF ≤ 0 | 对 $z$ 做一次左旋 |
| **LR（左右）** | $z$ 的 BF = +2，$z$ 的左孩子 BF < 0 | 先对左孩子左旋，再对 $z$ 右旋 |
| **RL（右左）** | $z$ 的 BF = -2，$z$ 的右孩子 BF > 0 | 先对右孩子右旋，再对 $z$ 左旋 |

**单旋转（右旋，LL 情形）示意**：

```
    z (+2)               y (0)
   / \                  / \
  y (+1)  T4   →      x    z
 / \                  /\   /\
x   T3              T1 T2 T3 T4
/\
T1 T2
```

旋转后 $y$ 成为新根，$z$ 成为 $y$ 的右孩子；BST 中序顺序 $T_1 \leq x \leq T_2 \leq y \leq T_3 \leq z \leq T_4$ 保持不变。

```python
def rotate_right(z):
    """右旋（LL 情形）：z 失衡，左孩子 y 上移"""
    y = z.left
    T3 = y.right
    # 执行旋转
    y.right = z
    z.left = T3
    # 先更新 z 的高度（z 在 y 之下），再更新 y
    update_height(z)
    update_height(y)
    return y  # y 成为新子树根

def rotate_left(z):
    """左旋（RR 情形）：z 失衡，右孩子 y 上移"""
    y = z.right
    T2 = y.left
    y.left = z
    z.right = T2
    update_height(z)
    update_height(y)
    return y
```
```cpp
AVLNode* rotateRight(AVLNode* z) {
    AVLNode* y = z->left;
    AVLNode* T3 = y->right;
    y->right = z;
    z->left = T3;
    updateHeight(z);
    updateHeight(y);
    return y;
}

AVLNode* rotateLeft(AVLNode* z) {
    AVLNode* y = z->right;
    AVLNode* T2 = y->left;
    y->left = z;
    z->right = T2;
    updateHeight(z);
    updateHeight(y);
    return y;
}
```

**双旋转（LR 情形）**：失衡节点 $z$ 的 BF = +2，但左孩子 $y$ 的 BF = -1（意味着新节点插在 $y$ 的右子树）。此时单一右旋无法解决，需要**先对 $y$ 左旋，再对 $z$ 右旋**：

```python
def balance(node):
    """通用平衡恢复函数，调用时 node 的高度已更新"""
    bf = get_bf(node)
    # LL 情形
    if bf == 2 and get_bf(node.left) >= 0:
        return rotate_right(node)
    # LR 情形
    if bf == 2 and get_bf(node.left) < 0:
        node.left = rotate_left(node.left)
        return rotate_right(node)
    # RR 情形
    if bf == -2 and get_bf(node.right) <= 0:
        return rotate_left(node)
    # RL 情形
    if bf == -2 and get_bf(node.right) > 0:
        node.right = rotate_right(node.right)
        return rotate_left(node)
    return node  # 平衡，无需旋转
```
```cpp
AVLNode* balance(AVLNode* node) {
    updateHeight(node);
    int bf = getBF(node);
    // LL
    if (bf == 2 && getBF(node->left) >= 0)
        return rotateRight(node);
    // LR
    if (bf == 2 && getBF(node->left) < 0) {
        node->left = rotateLeft(node->left);
        return rotateRight(node);
    }
    // RR
    if (bf == -2 && getBF(node->right) <= 0)
        return rotateLeft(node);
    // RL
    if (bf == -2 && getBF(node->right) > 0) {
        node->right = rotateRight(node->right);
        return rotateLeft(node);
    }
    return node;
}
```

下面的动画展示了四种旋转情形从失衡到恢复的完整过程，节点上的徽章显示实时平衡因子：

<div data-component="AVLRotationAnimator"></div>

### 10.2.4 插入操作

AVL 树的插入与普通 BST 插入相同，只是在**递归回溯路径上**对每个祖先节点调用 `balance()`。由于插入最多只会导致一个节点失衡，且修复后整棵树高度不变或缩短，因此**最多执行 1 次（单旋）或 2 次（双旋）旋转**。

```python
def avl_insert(node, key):
    """AVL 树插入，返回新子树根"""
    # 1. 普通 BST 插入
    if node is None:
        return AVLNode(key)
    if key < node.key:
        node.left = avl_insert(node.left, key)
    elif key > node.key:
        node.right = avl_insert(node.right, key)
    else:
        return node  # 重复键，忽略
    # 2. 更新高度并恢复平衡
    update_height(node)
    return balance(node)

class AVLTree:
    def __init__(self):
        self.root = None

    def insert(self, key):
        self.root = avl_insert(self.root, key)
```
```cpp
AVLNode* avlInsert(AVLNode* node, int key) {
    // 1. 普通 BST 插入
    if (!node) return new AVLNode(key);
    if (key < node->key)
        node->left = avlInsert(node->left, key);
    else if (key > node->key)
        node->right = avlInsert(node->right, key);
    else
        return node; // 重复键
    // 2. 更新高度并恢复平衡
    return balance(node);
}

class AVLTree {
public:
    AVLNode* root = nullptr;
    void insert(int key) { root = avlInsert(root, key); }
};
```

**示例过程**：向空 AVL 树依次插入 $30, 20, 40, 10, 25$：

1. 插入 30：树 = `30(0)`，BF = 0，平衡
2. 插入 20：`30(1)` 左孩子为 `20(0)`，BF = 1，平衡
3. 插入 40：`30(0)` 左 `20`，右 `40`，BF = 0，平衡
4. 插入 10：`30(1)` 左子树高 2，BF = 1，`20(1)` 的 BF = 1（LL）→ 对 30 右旋 → 新根 `20`，子树 `{10, 20, 30-40}`
5. 插入 25：无失衡，高度正常

### 10.2.5 删除操作

删除比插入复杂：AVL 树删除后，可能需要从被删节点开始向上**逐层检查**并修复，最坏情形需旋转 $O(\log n)$ 次（插入只需 $O(1)$ 次）。

删除策略与 BST 相同（无子、单子、双子），双子情形用**中序后继**替换后删除后继节点：

```python
def avl_min_node(node):
    """找子树中的最小节点"""
    while node.left:
        node = node.left
    return node

def avl_delete(node, key):
    """AVL 树删除，返回新子树根"""
    if node is None:
        return None
    if key < node.key:
        node.left = avl_delete(node.left, key)
    elif key > node.key:
        node.right = avl_delete(node.right, key)
    else:
        # 找到目标节点
        if node.left is None:
            return node.right
        elif node.right is None:
            return node.left
        else:
            # 双子节点：用中序后继替换
            successor = avl_min_node(node.right)
            node.key = successor.key
            node.right = avl_delete(node.right, successor.key)
    # 回溯时更新高度并恢复平衡
    update_height(node)
    return balance(node)
```
```cpp
AVLNode* avlMinNode(AVLNode* node) {
    while (node->left) node = node->left;
    return node;
}

AVLNode* avlDelete(AVLNode* node, int key) {
    if (!node) return nullptr;
    if (key < node->key)
        node->left = avlDelete(node->left, key);
    else if (key > node->key)
        node->right = avlDelete(node->right, key);
    else {
        if (!node->left) return node->right;
        if (!node->right) return node->left;
        // 双子节点
        AVLNode* succ = avlMinNode(node->right);
        node->key = succ->key;
        node->right = avlDelete(node->right, succ->key);
    }
    return balance(node);
}
```

### 10.2.6 高度上界证明：$h \leq 1.44 \log_2(n+2)$

**定义**：$N(h)$ 为高度恰好为 $h$ 的 AVL 树所含节点数的**最小值**（即最稀疏的 AVL 树）。

**递推关系**：高度为 $h$ 的最稀疏 AVL 树，其两棵子树高度分别为 $h-1$ 和 $h-2$（恰好满足 BF = ±1），因此：

$$N(h) = 1 + N(h-1) + N(h-2), \quad N(0) = 1,\ N(1) = 2$$

这与 Fibonacci 数列的递推形式相同！令 $F_k$ 为第 $k$ 个 Fibonacci 数，可以证明：

$$N(h) = F_{h+3} - 1$$

由 Fibonacci 数的封闭形式 $F_k \approx \varphi^k / \sqrt{5}$（$\varphi = (1+\sqrt{5})/2 \approx 1.618$），有：

$$n \geq N(h) = F_{h+3} - 1 \approx \frac{\varphi^{h+3}}{\sqrt{5}} - 1$$

取对数解出 $h$：

$$h \leq \log_\varphi(\sqrt{5}(n+1)) - 3 = \frac{\log_2(\sqrt{5}(n+1))}{\log_2 \varphi} - 3 \approx 1.44 \log_2(n+2) - 0.328$$

**直觉**：AVL 树高度上界约为完全二叉树高度（$\log_2 n$）的 **1.44 倍**，代价极小。

---

## 10.3 红黑树

红黑树（Red-Black Tree）是另一类重要的自平衡 BST，被广泛用于：
- C++ STL 的 `std::map`、`std::set`、`std::multimap`
- Java 的 `TreeMap`、`TreeSet`
- Linux 内核的完全公平调度器（CFS）和虚拟内存管理

与 AVL 树相比，红黑树的**高度略高**（上界 $2 \log_2(n+1)$ vs $1.44 \log_2 n$），因此查询稍慢；但**插入和删除涉及的旋转次数更少**（INSERT 最多 2 次旋转，DELETE 最多 3 次），适合读写均衡的场景。

### 10.3.1 五条性质

红黑树在每个节点上新增一个**颜色属性**（红或黑），并满足以下五条性质：

1. **颜色性**：每个节点要么是红色，要么是黑色。
2. **根性**：根节点是黑色。
3. **叶节点性**：所有叶节点（NIL 节点）是黑色。**注意**：红黑树的"叶节点"是外部空节点 NIL，不是普通意义上的叶节点。
4. **红-红禁止性**：若一个节点是红色，则它的两个子节点均为黑色（即不存在两个相邻的红节点）。
5. **黑高一致性**：对于任意节点，从该节点到其每个后代叶节点（NIL）的所有简单路径，均包含相同数目的黑色节点。

**定义**：从节点 $x$（不含 $x$）到其后代叶节点的任意路径上黑色节点的数目，称为 $x$ 的**黑高（Black-Height）**，记为 $\text{bh}(x)$。由性质 5，黑高对同一节点的所有后代路径是相同的，因此定义良好。

### 10.3.2 黑高与高度关系

**引理**：以节点 $x$ 为根的子树，至少含有 $2^{\text{bh}(x)} - 1$ 个内部节点（非 NIL 节点）。

**证明**（数学归纳）：若 $x$ 是叶（NIL），子树含 $0$ 个节点，$2^0 - 1 = 0$，成立。若 $x$ 不是叶，其两个子节点的黑高至少为 $\text{bh}(x) - 1$（若子节点红色则黑高与 $x$ 相同，若黑则减 1）。由归纳假设，每个子树含至少 $2^{\text{bh}(x)-1}-1$ 个节点，故 $x$ 的子树含至少 $1 + 2(2^{\text{bh}(x)-1}-1) = 2^{\text{bh}(x)}-1$ 个节点。

**推论**：含 $n$ 个内部节点的红黑树，其高度 $h \leq 2 \log_2(n+1)$。

**证明**：由性质 4，从根到叶的任意路径上，红节点数不超过黑节点数，故 $\text{bh}(\text{root}) \geq h/2$。由引理 $n \geq 2^{h/2} - 1$，解得 $h \leq 2\log_2(n+1)$。

### 10.3.3 旋转操作

红黑树的旋转与 AVL 树相同，但不需要更新高度字段（红黑树不存储高度）：

```python
class RBNode:
    RED = True
    BLACK = False

    def __init__(self, key):
        self.key = key
        self.color = RBNode.RED   # 新节点默认红色
        self.left = None
        self.right = None
        self.parent = None

class RBTree:
    def __init__(self):
        # 使用哨兵节点代替 None，简化边界处理
        self.NIL = RBNode(0)
        self.NIL.color = RBNode.BLACK
        self.root = self.NIL

    def left_rotate(self, x):
        """左旋：x 的右孩子 y 上移，x 成为 y 的左孩子"""
        y = x.right
        x.right = y.left
        if y.left != self.NIL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent == self.NIL:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def right_rotate(self, y):
        """右旋：y 的左孩子 x 上移，y 成为 x 的右孩子"""
        x = y.left
        y.left = x.right
        if x.right != self.NIL:
            x.right.parent = y
        x.parent = y.parent
        if y.parent == self.NIL:
            self.root = x
        elif y == y.parent.right:
            y.parent.right = x
        else:
            y.parent.left = x
        x.right = y
        y.parent = x
```
```cpp
enum Color { RED, BLACK };

struct RBNode {
    int key;
    Color color;
    RBNode *left, *right, *parent;
    RBNode(int k) : key(k), color(RED), left(nullptr), right(nullptr), parent(nullptr) {}
};

class RBTree {
public:
    RBNode* NIL;  // 哨兵节点
    RBNode* root;

    RBTree() {
        NIL = new RBNode(0);
        NIL->color = BLACK;
        root = NIL;
    }

    void leftRotate(RBNode* x) {
        RBNode* y = x->right;
        x->right = y->left;
        if (y->left != NIL) y->left->parent = x;
        y->parent = x->parent;
        if (x->parent == NIL) root = y;
        else if (x == x->parent->left) x->parent->left = y;
        else x->parent->right = y;
        y->left = x;
        x->parent = y;
    }

    void rightRotate(RBNode* y) {
        RBNode* x = y->left;
        y->left = x->right;
        if (x->right != NIL) x->right->parent = y;
        x->parent = y->parent;
        if (y->parent == NIL) root = x;
        else if (y == y->parent->right) y->parent->right = x;
        else y->parent->left = x;
        x->right = y;
        y->parent = x;
    }
};
```

### 10.3.4 RB-INSERT 框架

红黑树插入分两步：
1. 按 BST 规则插入，将新节点着为**红色**（着红不影响黑高一致性，但可能违反性质 4）
2. 调用 `rb_insert_fixup` 修复可能的红-红冲突

着红插入的理由：若着黑，则必然违反性质 5（增加了某些路径的黑高），修复更复杂。

```python
def rb_insert(tree, key):
    z = RBNode(key)
    z.left = tree.NIL
    z.right = tree.NIL

    # 标准 BST 插入
    y = tree.NIL
    x = tree.root
    while x != tree.NIL:
        y = x
        if z.key < x.key:
            x = x.left
        else:
            x = x.right
    z.parent = y
    if y == tree.NIL:
        tree.root = z
    elif z.key < y.key:
        y.left = z
    else:
        y.right = z

    # 修复红黑性质
    rb_insert_fixup(tree, z)
```
```cpp
void rbInsert(RBTree& tree, int key) {
    RBNode* z = new RBNode(key);
    z->left = z->right = tree.NIL;

    RBNode* y = tree.NIL;
    RBNode* x = tree.root;
    while (x != tree.NIL) {
        y = x;
        if (z->key < x->key) x = x->left;
        else x = x->right;
    }
    z->parent = y;
    if (y == tree.NIL) tree.root = z;
    else if (z->key < y->key) y->left = z;
    else y->right = z;

    rbInsertFixup(tree, z);
}
```

### 10.3.5 插入修复：三种情形

设新插入节点为 $z$，$z$ 的父节点为 $p = z.\text{parent}$（红色，否则无需修复），祖父节点为 $g = p.\text{parent}$（必为黑色，因为插入前树合法），叔节点为 $u$。

> 以下以 $z$ 在 $g$ 的左子树中（$p$ 是 $g$ 的左孩子）为例，右子树情形对称。

**Case 1：叔节点 $u$ 为红色**
- 违规原因：$z$（红）→ $p$（红），两个相邻红节点
- 修复：将 $p$ 和 $u$ 染黑，$g$ 染红，将 $z$ 上移至 $g$ 继续检查
- 分析：此操作不改变 $g$ 及以下子树的黑高，但 $g$ 变红可能与 $g$ 的父节点产生新冲突，需继续向上修复

**Case 2：$u$ 为黑色，$z$ 是 $p$ 的右孩子（内侧）**
- 修复：对 $p$ 做左旋，将 $p$ 作为新的 $z$，转为 Case 3
- 分析：旋转后 $z$ 原来的父节点 $p$ 现在处于 $z$ 的左孩子位置，构型变为外侧

**Case 3：$u$ 为黑色，$z$ 是 $p$ 的左孩子（外侧）**
- 修复：将 $p$ 染黑，$g$ 染红，对 $g$ 做右旋
- 分析：旋转后 $p$ 成为新子树根（黑色），$g$（红色）成为 $p$ 右孩子，$z$ 仍是 $p$ 左孩子（红色），满足所有性质，修复完毕！

```python
def rb_insert_fixup(tree, z):
    while z.parent.color == RBNode.RED:
        if z.parent == z.parent.parent.left:  # 父节点是祖父的左孩子
            u = z.parent.parent.right          # 叔节点
            if u.color == RBNode.RED:          # Case 1：叔红
                z.parent.color = RBNode.BLACK
                u.color = RBNode.BLACK
                z.parent.parent.color = RBNode.RED
                z = z.parent.parent            # z 上移
            else:
                if z == z.parent.right:        # Case 2：叔黑，内侧
                    z = z.parent
                    tree.left_rotate(z)
                # Case 3：叔黑，外侧
                z.parent.color = RBNode.BLACK
                z.parent.parent.color = RBNode.RED
                tree.right_rotate(z.parent.parent)
        else:                                   # 对称情形（父是祖父右孩子）
            u = z.parent.parent.left
            if u.color == RBNode.RED:           # Case 1 对称
                z.parent.color = RBNode.BLACK
                u.color = RBNode.BLACK
                z.parent.parent.color = RBNode.RED
                z = z.parent.parent
            else:
                if z == z.parent.left:          # Case 2 对称
                    z = z.parent
                    tree.right_rotate(z)
                z.parent.color = RBNode.BLACK   # Case 3 对称
                z.parent.parent.color = RBNode.RED
                tree.left_rotate(z.parent.parent)
    tree.root.color = RBNode.BLACK              # 确保根为黑（Case 1 可能将根染红）
```
```cpp
void rbInsertFixup(RBTree& tree, RBNode* z) {
    while (z->parent->color == RED) {
        if (z->parent == z->parent->parent->left) {
            RBNode* u = z->parent->parent->right;  // 叔节点
            if (u->color == RED) {                  // Case 1
                z->parent->color = BLACK;
                u->color = BLACK;
                z->parent->parent->color = RED;
                z = z->parent->parent;
            } else {
                if (z == z->parent->right) {        // Case 2
                    z = z->parent;
                    tree.leftRotate(z);
                }
                z->parent->color = BLACK;           // Case 3
                z->parent->parent->color = RED;
                tree.rightRotate(z->parent->parent);
            }
        } else {                                    // 对称
            RBNode* u = z->parent->parent->left;
            if (u->color == RED) {
                z->parent->color = BLACK;
                u->color = BLACK;
                z->parent->parent->color = RED;
                z = z->parent->parent;
            } else {
                if (z == z->parent->left) {
                    z = z->parent;
                    tree.rightRotate(z);
                }
                z->parent->color = BLACK;
                z->parent->parent->color = RED;
                tree.leftRotate(z->parent->parent);
            }
        }
    }
    tree.root->color = BLACK;
}
```

以下动画展示了三种插入情形的颜色修复过程：

<div data-component="RedBlackTreeColorFix"></div>

**性质分析**：
- Case 1 可能反复触发，但每次 $z$ 上移 2 层，因此循环次数 $\leq h/2 = O(\log n)$
- Case 2 只执行一次左旋后转为 Case 3
- Case 3 执行一次右旋后修复完成，循环终止
- **总旋转次数 $\leq 2$**（Case 2 的左旋 + Case 3 的右旋），这是红黑树插入效率高的关键

### 10.3.6 RB-DELETE 与双黑处理

删除是红黑树最复杂的部分。整体分两阶段：
1. **BST 删除**（与普通 BST 相同，双子节点用后继替换）
2. **颜色修复**：若被删节点为**黑色**，则该节点所在路径黑高减 1，产生"**双黑（Double Black）**"冲突，需修复

被删节点为红色时，不影响任何性质，无需修复。

> 以被修复节点 $x$ 是其父节点 $p$ 的左孩子为例（右孩子对称）。$w$ 为 $x$ 的兄弟节点。

**Case 1：兄弟 $w$ 为红色**
- 将 $w$ 染黑，$p$ 染红，对 $p$ 左旋 → 转为 Case 2/3/4

**Case 2：兄弟 $w$ 黑色，$w$ 的两个子节点均为黑色**
- 将 $w$ 染红，双黑上移至 $p$（若 $p$ 原为红色直接变黑解决）

**Case 3：兄弟 $w$ 黑色，$w$ 的左子红、右子黑（近子红）**
- 将 $w$ 左孩子染黑，$w$ 染红，对 $w$ 右旋 → 转为 Case 4

**Case 4：兄弟 $w$ 黑色，$w$ 的右子为红色（远子红）**
- $w$ 继承 $p$ 的颜色，$p$ 染黑，$w$ 右孩子染黑，对 $p$ 左旋 → 修复完成！

```python
def rb_delete_fixup(tree, x):
    """x 为"双黑"节点（可以是 NIL）"""
    while x != tree.root and x.color == RBNode.BLACK:
        if x == x.parent.left:
            w = x.parent.right  # 兄弟节点
            if w.color == RBNode.RED:              # Case 1
                w.color = RBNode.BLACK
                x.parent.color = RBNode.RED
                tree.left_rotate(x.parent)
                w = x.parent.right
            if (w.left.color == RBNode.BLACK and
                    w.right.color == RBNode.BLACK): # Case 2
                w.color = RBNode.RED
                x = x.parent
            else:
                if w.right.color == RBNode.BLACK:  # Case 3
                    w.left.color = RBNode.BLACK
                    w.color = RBNode.RED
                    tree.right_rotate(w)
                    w = x.parent.right
                # Case 4
                w.color = x.parent.color
                x.parent.color = RBNode.BLACK
                w.right.color = RBNode.BLACK
                tree.left_rotate(x.parent)
                x = tree.root  # 结束循环
        else:                                      # 对称情形
            w = x.parent.left
            if w.color == RBNode.RED:
                w.color = RBNode.BLACK
                x.parent.color = RBNode.RED
                tree.right_rotate(x.parent)
                w = x.parent.left
            if (w.right.color == RBNode.BLACK and
                    w.left.color == RBNode.BLACK):
                w.color = RBNode.RED
                x = x.parent
            else:
                if w.left.color == RBNode.BLACK:
                    w.right.color = RBNode.BLACK
                    w.color = RBNode.RED
                    tree.left_rotate(w)
                    w = x.parent.left
                w.color = x.parent.color
                x.parent.color = RBNode.BLACK
                w.left.color = RBNode.BLACK
                tree.right_rotate(x.parent)
                x = tree.root
    x.color = RBNode.BLACK
```
```cpp
void rbDeleteFixup(RBTree& tree, RBNode* x) {
    while (x != tree.root && x->color == BLACK) {
        if (x == x->parent->left) {
            RBNode* w = x->parent->right;
            if (w->color == RED) {              // Case 1
                w->color = BLACK;
                x->parent->color = RED;
                tree.leftRotate(x->parent);
                w = x->parent->right;
            }
            if (w->left->color == BLACK && w->right->color == BLACK) { // Case 2
                w->color = RED;
                x = x->parent;
            } else {
                if (w->right->color == BLACK) { // Case 3
                    w->left->color = BLACK;
                    w->color = RED;
                    tree.rightRotate(w);
                    w = x->parent->right;
                }
                // Case 4
                w->color = x->parent->color;
                x->parent->color = BLACK;
                w->right->color = BLACK;
                tree.leftRotate(x->parent);
                x = tree.root;
            }
        } else {                               // 对称
            RBNode* w = x->parent->left;
            if (w->color == RED) {
                w->color = BLACK;
                x->parent->color = RED;
                tree.rightRotate(x->parent);
                w = x->parent->left;
            }
            if (w->right->color == BLACK && w->left->color == BLACK) {
                w->color = RED;
                x = x->parent;
            } else {
                if (w->left->color == BLACK) {
                    w->right->color = BLACK;
                    w->color = RED;
                    tree.leftRotate(w);
                    w = x->parent->left;
                }
                w->color = x->parent->color;
                x->parent->color = BLACK;
                w->left->color = BLACK;
                tree.rightRotate(x->parent);
                x = tree.root;
            }
        }
    }
    x->color = BLACK;
}
```

下面的动画展示了四种双黑删除情形，紫色虚线圆环代表双黑节点：

<div data-component="RBDeleteDoubleBlack"></div>

**删除旋转次数分析**：
- Case 1 → 进入 Case 2/3/4（1 次旋转）
- Case 2 → 可能上移（0 次旋转），最多 $O(\log n)$ 次
- Case 3 → Case 4（1 次旋转）
- Case 4 → 结束（1 次旋转）
- **总旋转次数 $\leq 3$**

### 10.3.7 AVL 树 vs 红黑树

| 维度 | AVL 树 | 红黑树 |
|------|--------|--------|
| 高度上界 | $1.44 \log_2 n$ | $2 \log_2 n$ |
| 查询性能 | 略优（树高更矮） | 略差 |
| 插入旋转次数 | 最多 2 次 | 最多 2 次 |
| 删除旋转次数 | 最多 $O(\log n)$ 次 | **最多 3 次** |
| 额外存储 | 每节点 1 个整数（高度） | 每节点 1 位（颜色） |
| 实现复杂度 | 中等 | 较复杂 |
| 适用场景 | 查询密集型 | 读写均衡型 |

> **陷阱**：很多教材描述 AVL delete 需 $O(\log n)$ 次旋转，实际上这是最坏情形。平均而言两者差距不大，实际性能差异往往来自缓存效果、内存分配等工程因素。

---

## 10.4 Treap

**Treap** = **Tree** + **Heap**，是一种以随机化为核心的平衡 BST。它的设计思路简洁优雅：给每个节点额外分配一个**随机优先级（priority）**，在维护 BST 键序的**同时**，使优先级满足**最小堆（或最大堆）性质**。

经过随机化，Treap 以极高概率保持 $O(\log n)$ 的树高，且实现远比 AVL 树和红黑树简单。

### 10.4.1 核心性质

- **BST 性质**：对键 key，满足 BST 中序性质（左子 key < 当前 key < 右子 key）
- **堆性质**：对随机优先级 priority，满足最大堆性质（父节点 priority > 子节点 priority）

**定理（Treap 唯一性）**：给定 $n$ 个键值对 $(k_1, p_1), (k_2, p_2), \ldots, (k_n, p_n)$，若所有 key 各不相同且所有 priority 各不相同，则满足上述双重性质的 Treap 是**唯一的**。

这是因为：priority 最大的节点必为根（堆性质），根确定后，左右子树的键由 BST 性质确定，递归地确定整棵树。

### 10.4.2 随机优先级与期望高度

若 priority 由均匀随机数生成，则 Treap 的结构等价于**随机插入序列的 BST**（以 priority 排名作为插入顺序），其期望高度为 $E[h] = O(\log n)$，方差也很小。

具体地，期望高度约 $2 \ln n \approx 1.386 \log_2 n$，与随机 BST 相同，且比红黑树的最坏界 $2 \log_2 n$ 更优。

### 10.4.3 Split 与 Merge：函数式 Treap 实现

现代 Treap 通常不用传统的旋转插入，而是用**Split**（分裂）和**Merge**（合并）两个原子操作构建所有功能：

**Split(T, k)**：将 Treap $T$ 分裂为两棵 Treap $L$ 和 $R$，其中 $L$ 包含所有 key $\leq k$ 的节点，$R$ 包含所有 key $> k$ 的节点。

**Merge(L, R)**：将两棵 Treap $L$（所有 key $\leq$ $R$ 中所有 key）合并为一棵 Treap，同时维护堆性质。

```python
import random

class TreapNode:
    def __init__(self, key):
        self.key = key
        self.priority = random.random()   # 均匀随机优先级
        self.left = None
        self.right = None
        self.size = 1                     # 子树大小（可选，支持顺序统计）

def update_size(node):
    if node:
        left_size = node.left.size if node.left else 0
        right_size = node.right.size if node.right else 0
        node.size = 1 + left_size + right_size

def split(node, key):
    """分裂：返回 (L, R)，L 含 key <= k，R 含 key > k"""
    if node is None:
        return None, None
    if node.key <= key:
        # 当前节点归 L；继续从右子树分裂
        l_right, r = split(node.right, key)
        node.right = l_right
        update_size(node)
        return node, r
    else:
        # 当前节点归 R；继续从左子树分裂
        l, r_left = split(node.left, key)
        node.left = r_left
        update_size(node)
        return l, node

def merge(left, right):
    """合并两棵 Treap（left 的所有 key < right 的所有 key）"""
    if left is None:
        return right
    if right is None:
        return left
    if left.priority > right.priority:
        # left 优先级更高，成为新根；将 right 合并到 left 的右子树
        left.right = merge(left.right, right)
        update_size(left)
        return left
    else:
        right.left = merge(left, right.left)
        update_size(right)
        return right

def treap_insert(root, key):
    """插入：分裂后插入新节点再合并"""
    l, r = split(root, key - 1)   # key-1 确保新节点在 r 的最小位置
    # 实际使用 key 分裂后再处理去重更严谨，此处简化
    l2, r2 = split(r, key)        # 分离已有的 key（去重）
    new_node = TreapNode(key)
    return merge(merge(l, new_node), r2)

def treap_delete(root, key):
    """删除：找到节点后合并左右子树"""
    l, r = split(root, key - 1)
    _, r2 = split(r, key)         # r 只含 key，丢弃
    return merge(l, r2)
```
```cpp
#include <cstdlib>
#include <ctime>

struct TreapNode {
    int key, priority, size;
    TreapNode *left, *right;
    TreapNode(int k) : key(k), priority(rand()), size(1),
                       left(nullptr), right(nullptr) {}
};

int sz(TreapNode* t) { return t ? t->size : 0; }

void pushup(TreapNode* t) {
    if (t) t->size = 1 + sz(t->left) + sz(t->right);
}

// 按 key 分裂：key <= k 归 l，key > k 归 r
void split(TreapNode* t, int k, TreapNode*& l, TreapNode*& r) {
    if (!t) { l = r = nullptr; return; }
    if (t->key <= k) {
        l = t;
        split(t->right, k, l->right, r);
        pushup(l);
    } else {
        r = t;
        split(t->left, k, l, r->left);
        pushup(r);
    }
}

TreapNode* merge(TreapNode* l, TreapNode* r) {
    if (!l) return r;
    if (!r) return l;
    if (l->priority > r->priority) {
        l->right = merge(l->right, r);
        pushup(l);
        return l;
    } else {
        r->left = merge(l, r->left);
        pushup(r);
        return r;
    }
}

TreapNode* insert(TreapNode* root, int key) {
    TreapNode *l, *r;
    split(root, key - 1, l, r);
    TreapNode *l2, *r2;
    split(r, key, l2, r2);   // 处理去重
    return merge(merge(l, new TreapNode(key)), r2);
}

TreapNode* del(TreapNode* root, int key) {
    TreapNode *l, *m, *r;
    split(root, key - 1, l, m);
    split(m, key, m, r);     // m 只含 key 节点，丢弃
    return merge(l, r);
}
```

下面的动画展示了 Split 和 Merge 的递归执行过程，节点上显示 key/priority 对：

<div data-component="TreapSplitMerge"></div>

### 10.4.4 Treap 的期望复杂度分析

**定理**：对于 $n$ 个节点的 Treap，以下操作的期望时间复杂度均为 $O(\log n)$：
- `Split(T, k)`
- `Merge(L, R)`
- `Insert(T, k)`（= 1 次 split + 1 次 merge）
- `Delete(T, k)`（= 2 次 split + 1 次 merge）

**证明思路**：Split 和 Merge 的递归深度等于路径长度，而 Treap 的期望高度为 $O(\log n)$，故复杂度为 $O(\log n)$。

**竞赛优势**：Treap 通过 Split/Merge 可优雅地实现**区间操作**（区间翻转、区间赋值等），配合"懒标记（lazy tag）"可以实现功能强大的**Fhq Treap（非旋转 Treap）**，在算法竞赛中极为常用。

**Treap 应用场景**：
- 算法竞赛中替代线段树处理动态序列
- 路由表的 IP 前缀查找（Treap with LPM）
- 实时系统中替代红黑树（实现简单、随机性保证期望性能）

---

## 10.5 Splay 树

**Splay 树**（伸展树）由 Tarjan 和 Sleator 在 1985 年提出。它是一种**自适应**的 BST：每次访问一个节点，就通过一系列旋转将其"提升"到根，这个过程称为 **Splay 操作**。

Splay 树没有存储额外的平衡信息（无颜色、无高度），结构最简洁。它的关键特性是：**访问局部性**——若某些元素被频繁访问，它们会持续处于树顶，访问速度趋近于 $O(1)$。

### 10.5.1 Splay 操作

Splay(x) 将节点 $x$ 旋转到根，分三种情形（以 $x$ 的父节点 $p$ 和祖父节点 $g$ 为基础）：

**Zig**（$x$ 的父节点是根）：
- 对 $p$ 做一次旋转（左旋或右旋），$x$ 成为新根

**Zig-Zig**（$x, p, g$ 同侧，即 $x$ 是 $p$ 的左孩子且 $p$ 是 $g$ 的左孩子，或都是右孩子）：
- 先对 $g$ 旋转（将 $p$ 提升），再对 $p$ 旋转（将 $x$ 提升）
- **注意**：与 AVL 树 LL 情形的区别是先旋转祖父而非父节点！这是摊销分析成立的关键

**Zig-Zag**（$x, p, g$ 不同侧，即 $x$ 是 $p$ 的右孩子而 $p$ 是 $g$ 的左孩子，或相反）：
- 先对 $p$ 旋转，再对 $g$ 旋转（与 AVL 树 LR 情形相同）

```python
class SplayNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.parent = None

class SplayTree:
    def __init__(self):
        self.root = None

    def _rotate(self, x):
        """将 x 旋转到其父节点位置"""
        p = x.parent
        g = p.parent if p else None

        if x == p.left:  # x 是左孩子，右旋 p
            p.left = x.right
            if x.right:
                x.right.parent = p
            x.right = p
        else:            # x 是右孩子，左旋 p
            p.right = x.left
            if x.left:
                x.left.parent = p
            x.left = p

        x.parent = g
        p.parent = x
        if g:
            if p == g.left:
                g.left = x
            elif p == g.right:
                g.right = x
        else:
            self.root = x

    def splay(self, x):
        """将 x 提升到根"""
        while x.parent:
            p = x.parent
            g = p.parent
            if g is None:           # Zig
                self._rotate(x)
            elif ((x == p.left) == (p == g.left)):  # Zig-Zig（同侧）
                self._rotate(p)     # 先旋转父节点
                self._rotate(x)
            else:                   # Zig-Zag（异侧）
                self._rotate(x)     # 先旋转 x
                self._rotate(x)

    def find(self, key):
        """查找并 splay 到根"""
        x = self.root
        last = None
        while x:
            last = x
            if key == x.key:
                break
            elif key < x.key:
                x = x.left
            else:
                x = x.right
        if last:
            self.splay(last)
        return last and last.key == key

    def insert(self, key):
        """插入键"""
        if self.root is None:
            self.root = SplayNode(key)
            return

        x = self.root
        while True:
            if key == x.key:
                self.splay(x)
                return           # 已存在，直接 splay
            elif key < x.key:
                if x.left is None:
                    node = SplayNode(key)
                    node.parent = x
                    x.left = node
                    self.splay(node)
                    return
                x = x.left
            else:
                if x.right is None:
                    node = SplayNode(key)
                    node.parent = x
                    x.right = node
                    self.splay(node)
                    return
                x = x.right
```
```cpp
struct SplayNode {
    int key;
    SplayNode *left, *right, *parent;
    SplayNode(int k) : key(k), left(nullptr), right(nullptr), parent(nullptr) {}
};

class SplayTree {
public:
    SplayNode* root = nullptr;

    bool isLeft(SplayNode* x) { return x->parent && x == x->parent->left; }

    void rotate(SplayNode* x) {
        SplayNode* p = x->parent;
        SplayNode* g = p ? p->parent : nullptr;

        if (isLeft(x)) {   // 右旋 p
            p->left = x->right;
            if (x->right) x->right->parent = p;
            x->right = p;
        } else {           // 左旋 p
            p->right = x->left;
            if (x->left) x->left->parent = p;
            x->left = p;
        }
        x->parent = g;
        p->parent = x;
        if (g) {
            if (p == g->left) g->left = x;
            else if (p == g->right) g->right = x;
        } else {
            root = x;
        }
    }

    void splay(SplayNode* x) {
        while (x->parent) {
            SplayNode* p = x->parent;
            SplayNode* g = p->parent;
            if (!g) {                     // Zig
                rotate(x);
            } else if (isLeft(x) == isLeft(p)) {  // Zig-Zig
                rotate(p);
                rotate(x);
            } else {                      // Zig-Zag
                rotate(x);
                rotate(x);
            }
        }
    }

    void insert(int key) {
        if (!root) { root = new SplayNode(key); return; }
        SplayNode* x = root;
        while (true) {
            if (key == x->key) { splay(x); return; }
            if (key < x->key) {
                if (!x->left) {
                    SplayNode* n = new SplayNode(key);
                    n->parent = x;
                    x->left = n;
                    splay(n); return;
                }
                x = x->left;
            } else {
                if (!x->right) {
                    SplayNode* n = new SplayNode(key);
                    n->parent = x;
                    x->right = n;
                    splay(n); return;
                }
                x = x->right;
            }
        }
    }
};
```

### 10.5.2 摊销分析：势能法

Splay 树的每次操作**最坏情形**为 $O(n)$（退化为链），但通过势能法可以证明**摊销复杂度**为 $O(\log n)$。

**定义**：节点 $x$ 的**秩（rank）**：$r(x) = \log_2 s(x)$，其中 $s(x)$ 为以 $x$ 为根的子树大小。整棵树的**势函数**：

$$\Phi(T) = \sum_{x \in T} r(x) = \sum_{x \in T} \log_2 s(x)$$

**访问引理（Access Lemma）**：将节点 $x$ splay 到根的摊销时间为 $O(1 + r(\text{root}) - r(x))$。

由于 $r(\text{root}) = \log_2 n$，所有操作的摊销时间 $\leq O(1 + \log_2 n) = O(\log n)$。

### 10.5.3 访问局部性（Locality）

Splay 树的核心优势在于**时间局部性**：若某个元素在过去 $k$ 次操作中被访问了 $t$ 次，则之后访问它的代价约为 $O(\log(k/t))$。

**静态最优性**：若元素 $i$ 被访问 $n_i$ 次，总操作次数 $N = \sum n_i$，则 Splay 树的总访问时间为：

$$O\left(N + \sum_i n_i \log \frac{N}{n_i}\right)$$

这与任何静态 BST 的最优信息熵下界一致（Splay 树是渐进最优的静态结构之一）。

**实际应用**：
- TCP/IP 路由（部分路由表实现使用 Splay 树，因路由局部性强）
- 缓存替换（高频访问元素自动靠近树根）
- 文本编辑器的光标位置管理

---

## 10.6 总结与练习

### 10.6.1 核心知识点回顾

| 结构 | 不变量 | 关键操作 | 最坏高度 | 旋转上界（插入/删除） |
|------|--------|---------|---------|------------------|
| AVL 树 | BF ∈ {-1,0,+1} | 插入/删除 + 向上恢复 | $1.44 \log n$ | 2 / $O(\log n)$ |
| 红黑树 | 5条性质 | 插入修复3Case/删除修复4Case | $2 \log n$ | 2 / 3 |
| Treap | BST + 堆（随机优先级） | Split + Merge | 期望 $1.39 \log n$ | — |
| Splay | 无（摊销保证） | Zig/Zig-Zig/Zig-Zag | $O(n)$（摊销 $O(\log n)$） | — |

### 10.6.2 常见陷阱

1. **AVL 删除的旋转次数**：最坏 $O(\log n)$ 次，每次在回溯路径上可能触发新的旋转——虽然单次旋转常数小，但次数多于红黑树。
2. **红黑树 NIL 节点**：CLRS 标准实现用哨兵节点 `T.nil`（黑色），所有本应是 nullptr 的指针都指向它，极大简化边界处理。处理红黑树时务必区分"null"和"NIL（哨兵）"。
3. **Treap 的 Split 键值**：`split(T, k)` 的语义需要明确是"$\leq k$"还是"$< k$"，两种定义都可以用，但在 insert 中处理去重时需特别小心。
4. **Splay 的 Zig-Zig 顺序**：Zig-Zig 要先旋转**祖父**，再旋转**父节点**，与朴素递归的顺序相反！若反过来，摊销分析不成立，实际退化到 $O(n^2)$。

### 10.6.3 LeetCode 相关练习

虽然 LeetCode 不直接要求手写平衡 BST 实现，但以下题目与本章内容密切相关：

| 题号 | 题目 | 关联知识点 |
|------|------|---------|
| 108 | 将有序数组转换为二叉搜索树 | BST 构建，完全平衡 |
| 109 | 有序链表转换二叉搜索树 | 中序遍历反向构建，模拟 AVL 平衡建树 |
| 230 | 二叉搜索树中第K小的元素 | 顺序统计树（增广 BST），类似 Treap size 字段 |
| 1382 | 将二叉搜索树变平衡 | 平衡化思路，DSW 算法或重建 |
| 315 | 计算右侧小于当前元素的个数 | 增广平衡 BST 或归并排序 |
| 327 | 区间和的个数 | 离散化 + 平衡 BST / 树状数组 |

> **进阶思考**：为什么 C++ `std::map` 使用红黑树而不是 AVL 树？提示：考虑 `insert` 和 `erase` 的实际调用频率，以及 iterator 失效的要求。

---

> **参考资料**：
> - 《算法导论》（CLRS）第 12-13 章（BST 与红黑树）
> - Adelson-Velsky & Landis, *An Algorithm for the Organization of Information* (1962)
> - Sleator & Tarjan, *Self-Adjusting Binary Search Trees* (1985)
> - OI-Wiki 平衡树章节：https://oi-wiki.org/ds/balanced/

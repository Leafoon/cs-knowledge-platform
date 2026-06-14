# Chapter 7: 二叉树基础（Binary Tree Fundamentals）

> **难度**：🟢 初级 ｜ **前置知识**：递归基础（Chapter 2）、链表指针操作（Chapter 4）、队列（Chapter 5）
>
> **核心口诀**：树的问题 = 递归分解 + 明确返回值 + 清楚访问时机

---

## 7.1 树的基本概念

### 7.1.1 根、节点、叶子、深度与高度的精确定义

树（Tree）是计算机科学中最重要的非线性数据结构之一。在进入算法之前，必须对核心术语有**精确而无歧义**的定义——很多初学者在这里就埋下了混淆的祸根。

**基本术语表**：

| 术语 | 精确定义 | 示例（以下图为例） |
|------|----------|-------------------|
| **根（Root）** | 树中唯一没有父节点的节点 | 节点 A |
| **子节点（Child）** | 直接连接在某节点下方的节点 | B、C 是 A 的子节点 |
| **父节点（Parent）** | 直接连接在某节点上方的节点 | A 是 B 的父节点 |
| **兄弟节点（Sibling）** | 共享同一父节点的节点 | B 和 C 互为兄弟 |
| **叶子节点（Leaf）** | 没有子节点的节点 | D、E、F |
| **内部节点（Internal）** | 至少有一个子节点的节点 | A、B、C |
| **深度（Depth）** | 从根到该节点的边数（根深度 = 0） | depth(A)=0, depth(B)=1, depth(D)=2 |
| **高度（Height）** | 从该节点到最远叶子的边数（叶高度 = 0） | height(D)=0, height(B)=1, height(A)=2 |
| **层（Level）** | 深度为 k 的所有节点所在的层（根在第 0 层或第 1 层，视约定） | A 在第 0 层，B/C 在第 1 层 |

```
        A          ← 根，depth=0, height=2
       / \
      B   C        ← depth=1, height(B)=1, height(C)=0
     / \
    D   E          ← 叶子，depth=2, height=0
```

> ⚠️ **高频混淆点**：**depth 从根算**（向下），**height 从叶算**（向上）。不同教材对"层号从0还是从1"有不同约定，面试时要先确认对方约定。LeetCode 通常以根为 depth=0。

**树的深度（整棵树的高度）**：所有节点深度的最大值，等于树根的高度。

```python
# 计算节点深度（从根出发 BFS/DFS 均可）
def tree_depth(root) -> int:
    if root is None:
        return -1  # 空树高度为 -1（约定之一）
    return 1 + max(tree_depth(root.left), tree_depth(root.right))
```

---

### 7.1.2 树的递归定义

树（以及二叉树）最优雅的定义方式是**递归定义**，这也是树上算法大多用递归的根本原因：

> 一棵**二叉树**要么是空树（`None`），要么由一个根节点加上**左子树**和**右子树**组成，其中左/右子树也是二叉树。

$$\text{BinaryTree} = \emptyset \;\bigg|\; (\text{Node}, \;\text{BinaryTree}_\text{left}, \;\text{BinaryTree}_\text{right})$$

这个定义天然支持**结构归纳**：若某性质对空树成立，且对"若左右子树满足性质则根也满足性质"这一归纳步骤成立，则整棵树满足该性质。

**Python 实现**：

```python
from __future__ import annotations
from typing import Optional

class TreeNode:
    def __init__(self, val: int = 0,
                 left: Optional[TreeNode] = None,
                 right: Optional[TreeNode] = None):
        self.val = val
        self.left = left
        self.right = right

# 构造示例：
#       1
#      / \
#     2   3
#    / \
#   4   5
root = TreeNode(1,
    TreeNode(2, TreeNode(4), TreeNode(5)),
    TreeNode(3)
)
```

**C++ 实现**：

```cpp
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode* l, TreeNode* r) : val(x), left(l), right(r) {}
};

// 构造同一棵树
TreeNode* build() {
    return new TreeNode(1,
        new TreeNode(2, new TreeNode(4), new TreeNode(5)),
        new TreeNode(3)
    );
}
```

---

### 7.1.3 路径、子树、祖先与后代

- **路径（Path）**：由边序列连接的节点序列。树中任意两节点之间**恰好存在唯一路径**（树是无环连通图）。
- **简单路径**：不重复经过任何节点。树中所有路径均为简单路径。
- **子树（Subtree）**：某节点 v 与其所有后代节点（含 v 本身）构成的子集，也是一棵合法的树。
- **祖先（Ancestor）**：从根到节点 v 的路径上的所有节点（含根，含 v 本身，视约定）。
- **后代（Descendant）**：以 v 为根的子树中的所有节点（含 v 本身）。

> **关键性质**：祖先/后代关系具有传递性，即若 u 是 v 的祖先，v 是 w 的祖先，则 u 也是 w 的祖先。

---

### 7.1.4 树的基本性质：n 节点树有 n-1 条边（证明）

**定理**：具有 n 个节点的树恰好有 **n−1** 条边。

**证明（数学归纳法）**：
- **基础情形**：n=1（单根节点），边数 = 0 = 1−1 ✓
- **归纳步骤**：设具有 k 个节点的树有 k−1 条边（归纳假设）。考虑 n=k+1 个节点的树，取任意一片叶子节点 v，删去 v 及其与父节点的边，得到一棵 k 节点树，由归纳假设有 k−1 条边，加回叶子 v 时添加 1 条边，总边数 = k = (k+1)−1 ✓

**直觉理解**：每个非根节点恰好有**一条边**连接到其父节点，共 n−1 个非根节点，故 n−1 条边。

**推论**：若 $E$ 条边、$V$ 个节点组成的连通无向图具有 $V = E + 1$，则它是一棵树。

---

### 7.1.5 有序树 vs 无序树；二叉树 vs 多叉树

| 分类 | 说明 | 典型例子 |
|------|------|----------|
| **有序树（Ordered Tree）** | 每个节点的子节点有明确左右（或第1/2/3…）之分 | 二叉树、表达式树 |
| **无序树（Unordered Tree）** | 子节点无顺序区别 | 文件系统目录树、组织架构图 |
| **二叉树（Binary Tree）** | 每个节点至多 2 个子节点（left/right） | BST、堆、红黑树 |
| **多叉树（k-ary / N-ary Tree）** | 每个节点可有任意多子节点 | B 树（B-Tree）、字典树（Trie） |
| **森林（Forest）** | 若干棵互不相交的树的集合 | 并查集 |

**二叉树 vs 多叉树的统一视角**：任意多叉树可通过"**左子右兄弟**（Left-Child Right-Sibling）"表示法转化为二叉树，实现存储统一化。这是很多面试题的切入点。

---

## 7.2 二叉树的存储方式

### 7.2.1 满二叉树、完全二叉树、完美二叉树的区别

这三个概念是面试高频混淆点，必须仔细区分：

| 类型 | 定义 | 节点数 | 典型用途 |
|------|------|--------|----------|
| **完美二叉树（Perfect Binary Tree）** | 所有内部节点有 2 个子节点，且所有叶子在同一层 | $2^{h+1} - 1$ | 理论分析 |
| **完全二叉树（Complete Binary Tree）** | 除最后一层外全满，最后一层从左向右填充 | n（1 到 $2^{h+1}-1$） | 堆（Heap）的存储基础 |
| **满二叉树（Full Binary Tree）** | 每个节点要么是叶子，要么有 2 个子节点（无度=1的节点） | 奇数个 | 表达式树 |

```
完美二叉树（4层）       完全二叉树              满二叉树
        1                   1                   1
      /   \               /   \               /   \
     2     3             2     3             2     3
    / \ . / \           / \   /            /     / \
   4  5 6   7          4  5  6            4     6   7
```

**节点数与高度关系**：
- 高度为 h 的完美二叉树：节点数 = $2^{h+1} - 1$，叶子数 = $2^h$
- 高度为 h 的完全二叉树：节点数 $n \in [2^h, 2^{h+1}-1]$，即 $h = \lfloor\log_2 n\rfloor$
- **满二叉树性质**：若叶子数为 $L$，内部节点数为 $I$，则 $L = I + 1$（证明：边数 = $2I = n-1 = L+I-1$，化简得 $I = L-1$，即 $L = I+1$）

---

### 7.2.2 数组表示（完全二叉树）

对于**完全二叉树**，可以用一维数组按**层序**紧凑存储，利用下标关系隐式表达父子关系，无需额外指针：

| 约定 | 根下标 | 左子 $i$ | 右子 $i$ | 父节点 $i$ |
|------|--------|---------|---------|-----------|
| 1-indexed（堆常用） | 1 | $2i$ | $2i+1$ | $\lfloor i/2 \rfloor$ |
| 0-indexed（代码常用） | 0 | $2i+1$ | $2i+2$ | $\lfloor (i-1)/2 \rfloor$ |

```python
# 0-indexed 完全二叉树数组表示
tree = [1, 2, 3, 4, 5, 6, 7]
#         0  1  2  3  4  5  6
# 树形：
#       1(0)
#      /    \
#    2(1)   3(2)
#   /  \   /  \
# 4(3) 5(4) 6(5) 7(6)

def left(i):  return 2 * i + 1
def right(i): return 2 * i + 2
def parent(i): return (i - 1) // 2

# 验证
print(left(0))    # 1（即 tree[1]=2，根的左孩子）
print(right(0))   # 2（即 tree[2]=3，根的右孩子）
print(parent(3))  # 1（即 tree[1]=2，索引3的父节点）
```

**优点**：随机访问 O(1)，缓存友好，无指针开销。  
**缺点**：仅适合完全二叉树；稀疏树（如退化链表）浪费 $O(2^n)$ 空间。

---

### 7.2.3 链表表示（指针表示）

通用二叉树使用**链式节点**，每个节点包含值域、左右子节点指针，可选父指针：

```python
class TreeNode:
    __slots__ = ('val', 'left', 'right')  # 节省内存
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

```cpp
struct TreeNode {
    int val;
    TreeNode *left, *right, *parent;  // parent 可选
    TreeNode(int v) : val(v), left(nullptr), right(nullptr), parent(nullptr) {}
};
```

**父指针的代价与收益**：
- **收益**：可以 O(h) 求后继/前驱节点（向上爬），简化 LCA 查询
- **代价**：每个节点多 8 字节（64位指针），插入/删除需维护父指针一致性

---

### 7.2.4 两种存储方式的权衡

| 维度 | 数组（完全二叉树）| 链表（通用二叉树）|
|------|------------------|------------------|
| 空间效率 | O(n)（完全树）/ O(2^n)（最坏） | O(n)（始终） |
| 随机访问父子 | O(1)（下标计算）| O(1)（指针解引用） |
| 插入/删除 | 末尾 O(1)，中间 O(n) | O(1)（已知父节点）|
| 缓存友好性 | ✅ 连续内存，L1 hit 高 | ❌ 指针跳转，L1 miss |
| 适用场景 | 堆（Heap）、完全二叉树 | BST、红黑树、通用树 |

---

## 7.3 二叉树遍历（核心）

遍历是二叉树算法的**基础骨架**，所有复杂树问题最终都能被分解为某种遍历 + 在每个节点执行的操作。理解四种遍历及其递归与迭代的等价性是本章最重要的目标。

<div data-component="BinaryTreeTraversalAnimator"></div>

---

### 7.3.1 前序遍历（Preorder）：根 → 左 → 右

**访问时机**：在递归进入左右子树**之前**访问根节点。  
**应用场景**：序列化树（记录根-左-右结构）、构建树的拷贝、打印目录结构。

```python
# ── 递归版（最直观）──────────────────────────────────
def preorder_recursive(root: Optional[TreeNode]) -> list[int]:
    if root is None:
        return []
    res = []
    def dfs(node):
        if not node:
            return
        res.append(node.val)   # ① 访问根
        dfs(node.left)          # ② 递归左
        dfs(node.right)         # ③ 递归右
    dfs(root)
    return res

# ── 迭代版（显式栈模拟调用栈）──────────────────────
def preorder_iterative(root: Optional[TreeNode]) -> list[int]:
    if not root:
        return []
    res, stack = [], [root]
    while stack:
        node = stack.pop()
        res.append(node.val)       # 弹出即访问
        if node.right:
            stack.append(node.right)  # 右先入栈（后处理）
        if node.left:
            stack.append(node.left)   # 左后入栈（先处理）
    return res
```

```cpp
// C++ 迭代版前序遍历
vector<int> preorder(TreeNode* root) {
    vector<int> res;
    stack<TreeNode*> st;
    if (root) st.push(root);
    while (!st.empty()) {
        auto node = st.top(); st.pop();
        res.push_back(node->val);
        if (node->right) st.push(node->right);  // 右先入
        if (node->left)  st.push(node->left);   // 左后入
    }
    return res;
}
```

---

### 7.3.2 中序遍历（Inorder）：左 → 根 → 右

**访问时机**：在递归进入左子树**之后**、右子树**之前**访问根节点。  
**应用场景**：BST 的有序输出（中序遍历 BST 产生升序序列）、表达式求值（中缀表达式）。

```python
# ── 递归版 ──────────────────────────────────────────
def inorder_recursive(root: Optional[TreeNode]) -> list[int]:
    res = []
    def dfs(node):
        if not node:
            return
        dfs(node.left)          # ① 递归左
        res.append(node.val)   # ② 访问根
        dfs(node.right)         # ③ 递归右
    dfs(root)
    return res

# ── 迭代版（需显式维护"回溯点"）────────────────────
def inorder_iterative(root: Optional[TreeNode]) -> list[int]:
    res, stack = [], []
    curr = root
    while curr or stack:
        # 一路向左走到底
        while curr:
            stack.append(curr)
            curr = curr.left
        # 回溯：弹出并访问
        curr = stack.pop()
        res.append(curr.val)
        # 转向右子树
        curr = curr.right
    return res
```

**迭代中序的核心思维**：模拟"左侧路径压栈 → 弹出访问 → 转右继续"，循环不变量：`stack` 中存储的是"已遍历左子树、等待访问的节点"。

---

### 7.3.3 后序遍历（Postorder）：左 → 右 → 根

**访问时机**：在递归完成左右子树**之后**访问根节点。  
**应用场景**：计算树的高度/大小（需要子树信息）、释放内存（先释放子节点）、后缀表达式求值。

```python
# ── 递归版 ──────────────────────────────────────────
def postorder_recursive(root: Optional[TreeNode]) -> list[int]:
    res = []
    def dfs(node):
        if not node:
            return
        dfs(node.left)          # ① 递归左
        dfs(node.right)         # ② 递归右
        res.append(node.val)   # ③ 访问根
    dfs(root)
    return res

# ── 迭代版技巧：前序改写 + 反转 ─────────────────────
# 前序：根→左→右  改为  根→右→左  再反转  =  左→右→根
def postorder_iterative(root: Optional[TreeNode]) -> list[int]:
    if not root:
        return []
    res, stack = [], [root]
    while stack:
        node = stack.pop()
        res.append(node.val)
        if node.left:  stack.append(node.left)   # 注意：左先入
        if node.right: stack.append(node.right)  # 右后入（先弹出）
    return res[::-1]  # 反转得到后序
```

```cpp
// C++ 后序（两栈法，更直观）
vector<int> postorder(TreeNode* root) {
    vector<int> res;
    if (!root) return res;
    stack<TreeNode*> st1, st2;
    st1.push(root);
    while (!st1.empty()) {
        auto n = st1.top(); st1.pop();
        st2.push(n);
        if (n->left)  st1.push(n->left);
        if (n->right) st1.push(n->right);
    }
    while (!st2.empty()) {
        res.push_back(st2.top()->val);
        st2.pop();
    }
    return res;
}
```

---

### 7.3.4 层序遍历（Level-order / BFS）：队列实现

层序遍历按层从上到下、从左到右访问所有节点，是**广度优先搜索（BFS）**在树上的体现。

```python
from collections import deque

def levelorder(root: Optional[TreeNode]) -> list[list[int]]:
    """按层返回，每层一个子列表（LeetCode #102 格式）"""
    if not root:
        return []
    res, queue = [], deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):   # 关键：记录当前层节点数
            node = queue.popleft()
            level.append(node.val)
            if node.left:  queue.append(node.left)
            if node.right: queue.append(node.right)
        res.append(level)
    return res
```

```cpp
vector<vector<int>> levelOrder(TreeNode* root) {
    vector<vector<int>> res;
    if (!root) return res;
    queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        int sz = q.size();   // 记录当前层大小
        vector<int> level;
        while (sz--) {
            auto node = q.front(); q.pop();
            level.push_back(node->val);
            if (node->left)  q.push(node->left);
            if (node->right) q.push(node->right);
        }
        res.push_back(level);
    }
    return res;
}
```

**关键细节**：在 `for _ in range(len(queue))` 进入内层循环时，`len(queue)` 冻结了当前层的节点数，避免在循环过程中新加入节点（下一层）被当作当前层处理。这是层序遍历最容易犯错的地方。

---

### 7.3.5 迭代遍历的统一框架（颜色标记法）

前/中/后序的迭代写法风格不一，难以记忆。**颜色标记法**（染色法）提供了一个统一的框架：

**核心思想**：用 `(node, color)` 元组入栈，`WHITE`（未处理）和 `GRAY`（已处理左子树，待访问）。仅改变节点入栈的顺序就能切换三种遍历。

```python
WHITE, GRAY = 0, 1

def unified_traversal(root, order='inorder'):
    res, stack = [], [(WHITE, root)]
    while stack:
        color, node = stack.pop()
        if node is None:
            continue
        if color == GRAY:
            res.append(node.val)   # 灰色节点直接访问
        else:
            # 根据遍历顺序安排入栈顺序（注意栈 LIFO，逆序入栈）
            if order == 'inorder':
                stack.extend([
                    (WHITE, node.right),  # ③ 右（最后处理）
                    (GRAY,  node),        # ② 根（中间处理）
                    (WHITE, node.left),   # ① 左（最先处理）
                ])
            elif order == 'preorder':
                stack.extend([
                    (WHITE, node.right),
                    (WHITE, node.left),
                    (GRAY,  node),        # 根最先出栈
                ])
            elif order == 'postorder':
                stack.extend([
                    (GRAY,  node),        # 根最后出栈
                    (WHITE, node.right),
                    (WHITE, node.left),
                ])
    return res
```

这个框架的优势：**只需改三行顺序**，就能实现三种遍历，无需死记三套不同的迭代写法。

---

### 7.3.6 Morris 遍历：O(1) 额外空间的中序遍历

递归遍历隐式使用调用栈（O(h) 空间），显式栈同样 O(h) 空间。**Morris 遍历**利用**线索二叉树（Threaded Binary Tree）**思想，将空指针临时借用为"线索"，实现 O(1) 额外空间的 O(n) 遍历。

<div data-component="MorrisTraversalStep"></div>

**核心思想**：对于当前节点 `curr`，
1. 若 `curr.left == None`：访问 `curr`，向右移动 `curr = curr.right`
2. 若 `curr.left != None`：
   - 找到中序前驱（左子树最右节点）`pred`
   - 若 `pred.right == None`：建立线索 `pred.right = curr`，向左移动
   - 若 `pred.right == curr`：说明左子树已遍历完，断开线索，**访问 curr**，向右移动

```python
def morris_inorder(root: Optional[TreeNode]) -> list[int]:
    res = []
    curr = root
    while curr:
        if curr.left is None:
            res.append(curr.val)   # 访问
            curr = curr.right
        else:
            # 找左子树最右节点（中序前驱）
            pred = curr.left
            while pred.right and pred.right is not curr:
                pred = pred.right
            
            if pred.right is None:
                pred.right = curr  # 建立线索
                curr = curr.left
            else:
                pred.right = None  # 断开线索
                res.append(curr.val)  # 访问
                curr = curr.right
    return res
```

**时间复杂度分析**：每条边最多被遍历 2 次（建线索 + 断线索），总 O(n)。  
**空间复杂度**：O(1)（不计输出数组）。

> ⚠️ **使用场景**：Morris 遍历修改了树的指针，虽然最终会恢复，但在多线程环境中不安全。面试中提出 Morris 遍历会让考官眼前一亮，但需同时说明其局限性。

---

## 7.4 二叉树基本算法

### 7.4.1 树的高度与节点数计算

这是递归分解思想的最佳入门示例：**每个问题 = 当前节点的贡献 + 递归处理子树的结果**。

```python
def height(root: Optional[TreeNode]) -> int:
    """空树高度 = -1，单节点高度 = 0"""
    if root is None:
        return -1
    return 1 + max(height(root.left), height(root.right))

def count_nodes(root: Optional[TreeNode]) -> int:
    """节点总数"""
    if root is None:
        return 0
    return 1 + count_nodes(root.left) + count_nodes(root.right)

# 对于完全二叉树可以优化到 O(log² n)
def count_complete_tree(root: Optional[TreeNode]) -> int:
    """利用完全二叉树性质，O(log²n)"""
    if not root:
        return 0
    # 最左路径长度
    left_h = 0
    node = root
    while node.left:
        left_h += 1
        node = node.left
    # 最右路径长度
    right_h = 0
    node = root
    while node.right:
        right_h += 1
        node = node.right
    if left_h == right_h:
        return (1 << (left_h + 1)) - 1  # 完美二叉树：2^(h+1) - 1
    return 1 + count_complete_tree(root.left) + count_complete_tree(root.right)
```

<div data-component="TreeHeightCalculator"></div>

**复杂度**：普通树 O(n)，完全二叉树优化版 O(log²n)（每次递归深度 O(log n)，共递归 O(log n) 次）。

---

### 7.4.2 最低公共祖先（LCA）：朴素版本

**问题定义**（LeetCode #236）：给定二叉树中两个节点 p 和 q，找到它们的**最低公共祖先（Lowest Common Ancestor, LCA）**，即同时是 p 和 q 的祖先中深度最大的节点。

**递归思路**（三种情况）：
1. 若 `root == None` 或 `root == p` 或 `root == q`：返回 `root`
2. 分别递归左右子树：
   - 若左子树找到、右子树也找到：说明 p、q 分居两侧，当前 root 就是 LCA
   - 若只有左子树找到：LCA 在左子树中
   - 若只有右子树找到：LCA 在右子树中

```python
def lowestCommonAncestor(root: Optional[TreeNode],
                          p: TreeNode, q: TreeNode) -> Optional[TreeNode]:
    if root is None or root is p or root is q:
        return root
    
    left  = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    
    if left and right:
        return root   # p、q 各在左右子树，当前节点是 LCA
    return left if left else right  # 只在一侧找到

# 时间：O(n)，空间：O(h)（递归栈）
```

<div data-component="LCAFinder"></div>

**正确性关键**：后序遍历（先处理子树再判断根），保证"归并"信息时子树已处理完毕。

**扩展**：若树有父指针，可将两节点到根的路径存入 set，然后从 q 向上找第一个在 set 中的节点，O(h) 时间。

---

### 7.4.3 二叉树序列化与反序列化

**问题定义**（LeetCode #297）：设计一种算法，将二叉树序列化为字符串，并能将该字符串反序列化还原为原始树结构。

<div data-component="TreeSerializeDemo"></div>

**前序 + null 标记方案**（最常用）：遇到空节点记录为 `"#"`，这样可以唯一确定树的结构。

```python
class Codec:
    def serialize(self, root: Optional[TreeNode]) -> str:
        """前序遍历，None 记为 '#'"""
        parts = []
        def dfs(node):
            if node is None:
                parts.append('#')
                return
            parts.append(str(node.val))
            dfs(node.left)
            dfs(node.right)
        dfs(root)
        return ','.join(parts)
    
    def deserialize(self, data: str) -> Optional[TreeNode]:
        """前序恢复：每次消费一个 token"""
        tokens = iter(data.split(','))
        def build():
            val = next(tokens)
            if val == '#':
                return None
            node = TreeNode(int(val))
            node.left  = build()
            node.right = build()
            return node
        return build()

# 示例
#       1
#      / \
#     2   3
#        / \
#       4   5
# serialize → "1,2,#,#,3,4,#,#,5,#,#"
# 共 11 个 token，5 个节点 + 6 个 null（每个节点贡献 2 个 null - 1 个 non-null，叶贡献 2 个 null）
```

**为什么需要 null 标记**？纯前序序列（如 `1,2,3`）无法区分以下两棵树：
```
  1      1
 /        \
2           2
 \           \
  3           3
```
加上 null 标记后（`1,2,#,3,#,#,#` vs `1,#,2,#,3,#,#`），两者序列化结果不同，唯一确定树结构。

**层序 BFS 序列化**（LeetCode 标准格式）：

```python
def serialize_bfs(root) -> str:
    if not root:
        return "[]"
    from collections import deque
    result, queue = [], deque([root])
    while queue:
        node = queue.popleft()
        if node:
            result.append(str(node.val))
            queue.append(node.left)
            queue.append(node.right)
        else:
            result.append("null")
    # 去掉末尾多余的 null
    while result and result[-1] == "null":
        result.pop()
    return "[" + ",".join(result) + "]"
```

---

### 7.4.4 镜像翻转与对称判断

**翻转二叉树**（LeetCode #226）：交换每个节点的左右子节点：

```python
def invertTree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    if not root:
        return None
    root.left, root.right = invertTree(root.right), invertTree(root.left)
    return root

# C++ 版本
TreeNode* invertTree(TreeNode* root) {
    if (!root) return nullptr;
    swap(root->left, root->right);
    invertTree(root->left);
    invertTree(root->right);
    return root;
}
```

**对称二叉树**（LeetCode #101）：判断树是否关于根轴对称：

```python
def isSymmetric(root: Optional[TreeNode]) -> bool:
    def isMirror(left, right) -> bool:
        if not left and not right:
            return True
        if not left or not right:
            return False
        return (left.val == right.val and
                isMirror(left.left, right.right) and   # 外侧对称
                isMirror(left.right, right.left))       # 内侧对称
    return isMirror(root.left, root.right)

# 迭代版（BFS，双端队列）
from collections import deque
def isSymmetric_iterative(root: Optional[TreeNode]) -> bool:
    queue = deque([root.left, root.right])
    while queue:
        left, right = queue.popleft(), queue.popleft()
        if not left and not right:
            continue
        if not left or not right or left.val != right.val:
            return False
        queue.extend([left.left, right.right, left.right, right.left])
    return True
```

---

### 7.4.5 路径和问题（三类经典变形）

路径和系列是**后序遍历收集子树信息**的典型应用，有三种递增难度的变形：

**变形 1：根到叶路径和为目标值**（LeetCode #112）

```python
def hasPathSum(root: Optional[TreeNode], target: int) -> bool:
    if not root:
        return False
    if not root.left and not root.right:   # 叶节点
        return root.val == target
    return (hasPathSum(root.left,  target - root.val) or
            hasPathSum(root.right, target - root.val))
```

**变形 2：任意节点到任意节点的最大路径和**（LeetCode #124，最难）

关键洞察：经过某节点的最长路径 = 左侧最优贡献 + 根值 + 右侧最优贡献；但**向上返回时只能选一侧**（路径不能分叉）。

```python
def maxPathSum(root: Optional[TreeNode]) -> int:
    res = float('-inf')
    
    def max_gain(node) -> int:
        """返回以 node 为端点（向上延伸）的最大路径和"""
        nonlocal res
        if not node:
            return 0
        
        left_gain  = max(max_gain(node.left),  0)  # 负增益舍弃
        right_gain = max(max_gain(node.right), 0)
        
        # 经过当前节点的最大路径（不向上延伸）
        path_sum = node.val + left_gain + right_gain
        res = max(res, path_sum)
        
        # 向上只能选一侧
        return node.val + max(left_gain, right_gain)
    
    max_gain(root)
    return res
```

**变形 3：二叉树直径**（LeetCode #543）：最长路径的**长度**（边数），思路与变形 2 完全一致：

```python
def diameterOfBinaryTree(root: Optional[TreeNode]) -> int:
    res = 0
    def depth(node) -> int:
        nonlocal res
        if not node:
            return 0
        l, r = depth(node.left), depth(node.right)
        res = max(res, l + r)  # 经过当前节点的路径长度
        return 1 + max(l, r)   # 向上贡献的深度
    depth(root)
    return res
```

---

### 7.4.6 二叉树的"分解问题 vs 遍历框架"

这是理解所有二叉树算法的**元思想**，掌握它可以系统性地解决绝大多数树问题。

**遍历框架（Traversal Framework）**：用一个外部变量（或全局状态）收集结果，遍历每个节点并更新状态。适合"每个节点都需要被访问一次"的统计类问题。

```python
# 遍历框架示例：统计节点数
count = 0
def traverse(node):
    global count
    if not node: return
    count += 1  # 在这里做操作
    traverse(node.left)
    traverse(node.right)
```

**分解框架（Decompose Framework）**：每个节点通过**返回值**向上传递信息，当前节点利用子树的返回值完成自己的计算。适合"需要子树信息才能得出结论"的问题。

```python
# 分解框架示例：计算高度
def height(node) -> int:
    if not node: return -1
    left_h  = height(node.left)   # 利用子树返回值
    right_h = height(node.right)
    return 1 + max(left_h, right_h)  # 向上返回
```

**选择指南**：
- 若问题可以"自顶向下"在遍历时得出答案（如打印所有路径）→ **遍历框架**
- 若当前节点的答案依赖于子树的结果（如高度、路径和、LCA）→ **分解框架**
- 许多复杂问题需要**两者结合**（如序列化、BST 范围求和）

**思考题**：为什么"任意路径最大和"（#124）适合分解框架而非遍历框架？因为每个节点需要知道左右子树各自能贡献多少增益，这是子树信息，只能用返回值传递。若用全局变量，则无法区分"左侧贡献"和"右侧贡献"。

---

## 7.5 经典 LeetCode 题解

### 题目 1：#94 二叉树的中序遍历

三种写法对比：

| 方法 | 代码量 | 空间 | 掌握优先级 |
|------|--------|------|-----------|
| 递归 | 5 行 | O(h) | ⭐⭐⭐必须会 |
| 显式栈迭代 | 10 行 | O(h) | ⭐⭐⭐必须会 |
| Morris 遍历 | 15 行 | O(1) | ⭐⭐进阶 |

---

### 题目 2：#102 二叉树的层序遍历

关键是记录每层节点数（`for _ in range(len(queue))`），将 BFS 改造为"按层批处理"。扩展题型：
- **#107**：从底部到顶部层序（结果 `reverse()`）
- **#103**：锯齿形层序（奇数层正向，偶数层反向）
- **#199**：右视图（每层取最后一个）

---

### 题目 3：#236 二叉树的最低公共祖先

模板解法如 7.4.2 所示。注意：BST 的 LCA 有更高效的 O(h) 解法，见 Chapter 8。

---

### 题目 4：#297 二叉树的序列化与反序列化

如 7.4.3 所示。面试追问：若树很大，如何流式序列化/节省内存？→ 用迭代 DFS + yield 生成器。

---

### 题目 5：#543 二叉树的直径

如 7.4.5 变形 3。注意：最长直径**不一定经过根节点**，必须用 `nonlocal res` 记录全局最大值。

---

### 题目 6：#124 二叉树中的最大路径和

最难的路径和变形，如 7.4.5 变形 2。三个关键点：
1. 负增益舍弃：`max(gain, 0)`
2. 每个节点计算"经过自己的完整路径"并更新全局最大
3. 返回值只能带一侧增益（路径不能分叉）

---

## 7.6 总结与面试考点

### 核心知识图谱

```
二叉树
├── 概念定义（深度/高度/完全/完美/满）
├── 存储方式（数组 vs 链表）
├── 遍历（前/中/后/层序）
│   ├── 递归版（3行核心）
│   ├── 迭代版（显式栈）
│   ├── 统一框架（颜色标记法）
│   └── Morris 遍历（O(1)空间）
└── 基本算法
    ├── 高度 / 节点数
    ├── LCA
    ├── 序列化 / 反序列化
    ├── 翻转 / 对称
    └── 路径和（根到叶 / 任意路径 / 直径）
```

### 遍历时机与访问顺序

| 遍历 | 访问时机 | 典型题型 | 核心 LeetCode |
|------|----------|----------|---------------|
| 前序 | 进入前 | 序列化、拷贝、构建 | #144, #297 |
| 中序 | 左后右前 | BST 有序输出、排名 | #94, #230 |
| 后序 | 离开后 | 高度、LCA、路径和 | #145, #124, #543 |
| 层序 | 逐层 | 最大宽度、锯齿、右视图 | #102, #103, #199 |

### 面试高频考点

1. **手写中序遍历的迭代版**（95% 的前中后序迭代题目均考查中序）
2. **LCA 后序递归模板**（#236，背熟分析框架）
3. **最大路径和的负增益舍弃技巧**（#124，最高频 Hard）
4. **层序遍历记录每层边界**（`for _ in range(len(queue))`）
5. **序列化 + null 标记的必要性**（#297）

### 常见错误汇总

| 错误 | 典型场景 | 修正 |
|------|----------|------|
| 深度/高度混淆 | 递归返回值理解偏差 | 明确：高度从叶子数，深度从根数 |
| 全局变量替代返回值 | 分解框架问题 | 子树信息必须通过返回值传递 |
| 层序不记录层大小 | `#102` 层序遍历 | `for _ in range(len(queue))` |
| LCA 忽略节点本身也是祖先 | `#236` | `if root == p or root == q: return root` |
| 路径和忘记舍弃负增益 | `#124` | `max(gain, 0)` |

### 思考题

1. **分解 vs 遍历**：路径打印（根到叶所有路径）适合哪种框架？为什么？（答：遍历框架，因为需要在每个叶节点触发输出，而不需要子树返回值）

2. **空间优化**：如何将递归中序遍历改为 O(1) 空间？（答：Morris 遍历，临时修改空指针为线索）

3. **完全二叉树节点数**：为什么统计完全二叉树节点数可以优化到 O(log²n)？能否进一步优化到 O(log n)？（答：可以用二分 + 比较最左最右路径长度判断满二叉树层；O(log n) 理论上需要 O(1) 验证是否满，目前标准方法仍是 O(log²n)）

---

**参考资料**：
- CLRS 第4版 Chapter 12 前导部分
- Sedgewick《算法》第4版 3.1–3.2
- MIT 6.006 Spring 2020 Lecture 6（树的遍历与分析）
- LeetCode《labuladong 的算法笔记》二叉树专题

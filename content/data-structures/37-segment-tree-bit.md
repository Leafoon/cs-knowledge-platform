# Chapter 37: 线段树与树状数组（Segment Tree & Binary Indexed Tree）

> **学习目标**：熟练实现树状数组（Fenwick Tree）的点更新与前缀查询，掌握 lowbit 的本质；掌握线段树的建树、点更新、区间查询及懒惰传播（区间修改），能处理复合操作；了解持久化线段树和稀疏表 RMQ 的使用场景。

---

## 章节导读

在竞赛和工程中，有一类极其常见的问题称为**区间查询 + 单点/区间更新**：

- "把第 $i$ 个元素加上 $\delta$，然后查询 $[l, r]$ 内所有元素的和是多少？"
- "把区间 $[l, r]$ 内所有元素都加上 $v$，然后查询 $[l', r']$ 的最大值？"

如果用普通数组：更新 $O(1)$，查询 $O(n)$。
如果用前缀和数组：查询 $O(1)$，但每次更新要重建前缀和，$O(n)$。

两种结构都存在严重的短板，**没能同时支持快速更新和快速查询**。本章介绍的三种数据结构，正是为了打破这一困境：

| 结构 | 更新 | 查询 | 区间修改 | 空间 | 实现难度 |
|---|---|---|---|---|---|
| 树状数组（BIT） | $O(\log n)$ | $O(\log n)$ | 需要技巧 | $O(n)$ | ⭐⭐ |
| 线段树 | $O(\log n)$ | $O(\log n)$ | $O(\log n)$（懒传播）| $O(n)$ | ⭐⭐⭐ |
| 稀疏表（Sparse Table） | 不支持 | $O(1)$ | 不支持 | $O(n \log n)$ | ⭐⭐ |

> **核心直觉**：  
> - **BIT / 线段树**：用树形结构把"区间信息"分层存储，把线性扫描变成"树上跳跃"，$O(n)$ 变 $O(\log n)$。  
> - **稀疏表**：预处理所有长度为 $2^k$ 的区间，查询时用两个相互重叠的区间覆盖目标区间，实现 $O(1)$，代价是只支持静态数据。

---

## 37.1 树状数组（Binary Indexed Tree / Fenwick Tree）

### 37.1.1 设计动机：从前缀和的困境说起

先回顾朴素前缀和数组的工作方式。给定原数组 $A[1..n]$，定义前缀和 $P[i] = A[1] + A[2] + \cdots + A[i]$。

查询 $[l, r]$ 的和：$P[r] - P[l-1]$，$O(1)$。  
单点更新 $A[i]$ 加 $\delta$：必须修改 $P[i], P[i+1], \ldots, P[n]$，$O(n)$。

当数组频繁被修改时，$O(n)$ 的更新代价无法接受。

**树状数组（Binary Indexed Tree，BIT）**，又称 **Fenwick Tree**（由 Peter Fenwick 于 1994 年提出），巧妙地利用了二进制表示的规律，使得**更新和查询都达到 $O(\log n)$**，并且代码极其简洁，常数因子小。

### 37.1.2 核心思想：让每个位置"负责"一段区间

BIT 的精髓在于：**数组 `tree[i]` 不存储 $A[i]$ 本身，而存储 $A[i]$ 所在的某一段区间的和**。

具体地，`tree[i]` 存储的区间是从 $i - \text{lowbit}(i) + 1$ 到 $i$ 的元素之和，即：

$$\text{tree}[i] = \sum_{k = i - \text{lowbit}(i) + 1}^{i} A[k]$$

其中 $\text{lowbit}(i)$ 是 $i$ 的二进制表示中**最低位的 1 所代表的数值**。

举个例子，$n = 8$：

```
 i  | 二进制 | lowbit(i) | tree[i] 负责的区间
 ---|--------|-----------|-------------------
 1  | 0001   |     1     | A[1..1]
 2  | 0010   |     2     | A[1..2]
 3  | 0011   |     1     | A[3..3]
 4  | 0100   |     4     | A[1..4]
 5  | 0101   |     1     | A[5..5]
 6  | 0110   |     2     | A[5..6]
 7  | 0111   |     1     | A[7..7]
 8  | 1000   |     8     | A[1..8]
```

可以看到，`tree[i]` 管辖的区间长度恰好等于 `lowbit(i)`，这不是巧合，而是二进制数结构的自然产物。

### 37.1.3 lowbit(i) = i & (-i)：二进制补码的魔法

`lowbit(i)` 的计算只需一行代码：

$$\text{lowbit}(i) = i \;\&\; (-i)$$

**为什么这样算？**

以 $i = 12 = (1100)_2$ 为例：

```
 i  = 12 =  0000 1100   （原码）
-i  = -12 = 1111 0100   （补码：按位取反再加1）
 i & (-i) = 0000 0100 = 4
```

通过补码运算，$-i$ 恰好将 $i$ 最低位的 1 以上的所有位取反，以下的所有位置 0，而最低位的 1 本身不变，因此 $i \;\&\; (-i)$ 精确提取了最低有效位（LSB，Least Significant Bit）。

这个技巧也是整个 BIT 能以 $O(\log n)$ 运行的根本原因。

### 37.1.4 前缀查询 QUERY(i)：逆向跳跃

**目标**：计算 $\text{prefix\_sum}(i) = A[1] + A[2] + \cdots + A[i]$。

**过程**：从 $i$ 出发，每次跳到 $i - \text{lowbit}(i)$，累加 `tree[i]`，直到 $i = 0$。

```
Query(6):
  i=6  (0110): tree[6] = A[5]+A[6]; 跳到 6 - lowbit(6) = 6 - 2 = 4
  i=4  (0100): tree[4] = A[1]+A[2]+A[3]+A[4]; 跳到 4 - lowbit(4) = 0
  i=0: 停止
  总和 = tree[6] + tree[4] = (A[5]+A[6]) + (A[1]+A[2]+A[3]+A[4]) = A[1..6] ✓
```

**为什么最多 $O(\log n)$ 步？**  
每次跳跃 $i \to i - \text{lowbit}(i)$ 都会将 $i$ 的二进制表示中最低位的 1 清零，$i$ 最多有 $\lfloor \log_2 n \rfloor + 1$ 个 1，所以最多跳 $O(\log n)$ 次。

### 37.1.5 点更新 UPDATE(i, delta)：向上传播

**目标**：将 $A[i]$ 增加 $\delta$，同时维护所有受影响的 `tree[]` 位置。

**过程**：从 $i$ 出发，每次跳到 $i + \text{lowbit}(i)$，更新 `tree[i] += delta`，直到超出范围。

```
Update(3, +5):
  i=3  (0011): tree[3] += 5; 跳到 3 + lowbit(3) = 3 + 1 = 4
  i=4  (0100): tree[4] += 5; 跳到 4 + lowbit(4) = 4 + 4 = 8
  i=8  (1000): tree[8] += 5; 跳到 8 + lowbit(8) = 16 > n
  停止
```

**正确性直觉**：`tree[i]` 管辖 $A[i]$ 当且仅当 $j - \text{lowbit}(j) < i \leq j$，即"$i$ 在 $j$ 管辖区间内"。恰好是 UPDATE 跳跃路径上经过的所有位置。

<Callout type="info" title="🔍 BIT 与 QUERY/UPDATE 的方向性">
**QUERY**（查询前缀和）：$i \to i - \text{lowbit}(i)$，方向向左（向小）。  
**UPDATE**（单点更新）：$i \to i + \text{lowbit}(i)$，方向向右（向大）。  
两个方向恰好相反，体现了 BIT 结构的对称美。
</Callout>

<div data-component="FenwickTreeUpdate"></div>

<div data-component="FenwickTreeQuery"></div>

### 37.1.6 完整代码实现

```python
class BIT:
    """
    树状数组（Binary Indexed Tree / Fenwick Tree）
    1-indexed：下标从 1 到 n
    支持：单点更新 O(log n)，前缀查询 O(log n)
    """
    def __init__(self, n: int):
        self.n = n
        self.tree = [0] * (n + 1)   # tree[0] 不使用，1-indexed

    def update(self, i: int, delta: int) -> None:
        """
        在位置 i 加上 delta（单点更新）
        每次跳到 i + lowbit(i)，直到超界
        时间复杂度：O(log n)
        """
        # ⚠️ 边界：i 必须 >= 1，否则 lowbit(0) = 0 死循环
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)           # 跳跃：i += lowbit(i)

    def query(self, i: int) -> int:
        """
        计算前缀和 A[1] + A[2] + ... + A[i]
        每次跳到 i - lowbit(i)，直到 i = 0
        时间复杂度：O(log n)
        """
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)           # 跳跃：i -= lowbit(i)
        return s

    def range_query(self, l: int, r: int) -> int:
        """区间查询 A[l..r] 的和，O(log n)"""
        return self.query(r) - self.query(l - 1)

    @classmethod
    def build(cls, arr: list) -> 'BIT':
        """
        从已有数组 arr（0-indexed）构建 BIT
        朴素方式：依次 update 每个元素，O(n log n)
        也可用 O(n) 构建（见下方）
        """
        n = len(arr)
        bit = cls(n)
        for i, val in enumerate(arr):
            bit.update(i + 1, val)  # 转为 1-indexed
        return bit


# ── 使用示例 ──
if __name__ == "__main__":
    A = [3, 2, -1, 6, 5, 4, -3, 3, 7, 2, 3]
    n = len(A)
    bit = BIT.build(A)

    # 查询 A[1..5] 的和
    # 期望：3 + 2 + (-1) + 6 + 5 = 15
    print(bit.range_query(1, 5))     # 输出：15

    # 单点更新：A[3] += 10，则 A[3] = 9
    bit.update(3, 10)

    # 再次查询 A[1..5]
    # 期望：3 + 2 + 9 + 6 + 5 = 25
    print(bit.range_query(1, 5))     # 输出：25
```

```cpp
#include <vector>
#include <iostream>
using namespace std;

class BIT {
    int n;
    vector<int> tree;  // 1-indexed，tree[0] 不使用
public:
    explicit BIT(int n) : n(n), tree(n + 1, 0) {}

    // 单点更新：在位置 i 加上 delta
    // ⚠️ i 必须 >= 1
    void update(int i, int delta) {
        for (; i <= n; i += i & (-i))  // i += lowbit(i)
            tree[i] += delta;
    }

    // 前缀查询：A[1] + ... + A[i]
    int query(int i) {
        int s = 0;
        for (; i > 0; i -= i & (-i))   // i -= lowbit(i)
            s += tree[i];
        return s;
    }

    // 区间查询：A[l..r]
    int range_query(int l, int r) {
        return query(r) - query(l - 1);
    }

    // 从 0-indexed 数组构建（O(n log n)）
    static BIT build(const vector<int>& arr) {
        BIT bit(arr.size());
        for (int i = 0; i < (int)arr.size(); ++i)
            bit.update(i + 1, arr[i]);
        return bit;
    }
};

int main() {
    vector<int> A = {3, 2, -1, 6, 5, 4, -3, 3, 7, 2, 3};
    BIT bit = BIT::build(A);

    // 查询 A[1..5] = 3+2-1+6+5 = 15
    cout << bit.range_query(1, 5) << "\n";  // 15

    // 单点更新 A[3] += 10
    bit.update(3, 10);
    cout << bit.range_query(1, 5) << "\n";  // 25
}
```

<Callout type="warning" title="⚠️ BIT 的经典陷阱">
**BIT 从 1 开始！** 如果 $i = 0$，则 `lowbit(0) = 0`，`i += lowbit(i)` 永远不会增大，变成死循环。  
所以在 `update` 中的条件必须是 `i <= n`，不能是 `i < n`；在 `query` 中停止条件是 `i > 0`，永远不会访问 `tree[0]`。
</Callout>

### 37.1.7 变体一：区间修改 + 点查询（差分 BIT）

**问题**：将区间 $[l, r]$ 内的每个元素加上 $v$，然后查询 $A[i]$ 的当前值。

**思路**：利用**差分数组**。定义差分数组 $D[i] = A[i] - A[i-1]$（$D[1] = A[1]$），则：

$$A[i] = D[1] + D[2] + \cdots + D[i] = \text{prefix\_sum}(D, i)$$

区间 $[l, r]$ 加 $v$ 在差分数组上等价于：$D[l] += v$，$D[r+1] -= v$（仅两次单点修改）。

因此：用 BIT 维护差分数组，可以做到：
- **区间修改**：两次 BIT 的 `update`，$O(\log n)$
- **点查询**：一次 BIT 的 `query`（前缀和），$O(\log n)$

```python
class DiffBIT:
    """差分 BIT：支持区间修改 + 点查询"""
    def __init__(self, n: int):
        self.bit = BIT(n)
        self.n = n

    def range_add(self, l: int, r: int, v: int) -> None:
        """区间 [l, r] 所有元素加 v，O(log n)"""
        self.bit.update(l, v)
        if r + 1 <= self.n:
            self.bit.update(r + 1, -v)

    def point_query(self, i: int) -> int:
        """查询位置 i 的当前值（前缀和 = A[i]），O(log n)"""
        return self.bit.query(i)
```

### 37.1.8 变体二：区间修改 + 区间查询（双 BIT）

这是 BIT 最难的变体，需要数学推导。

**推导**：设差分数组 $D[i] = A[i] - A[i-1]$，则：

$$\sum_{i=1}^{k} A[i] = \sum_{i=1}^{k} \sum_{j=1}^{i} D[j] = \sum_{j=1}^{k} D[j] \cdot (k - j + 1) = (k+1)\sum_{j=1}^{k} D[j] - \sum_{j=1}^{k} D[j] \cdot j$$

因此，维护两个 BIT：
- $\text{BIT}_1$ 维护 $D[j]$（用于计算 $\sum D[j]$）
- $\text{BIT}_2$ 维护 $D[j] \cdot j$（用于计算 $\sum D[j] \cdot j$）

区间 $[l, r]$ 的和 $= \text{prefix}(r) - \text{prefix}(l-1)$，其中：

$$\text{prefix}(k) = (k+1) \cdot \text{BIT}_1.\text{query}(k) - \text{BIT}_2.\text{query}(k)$$

```python
class RangeUpdateRangeQueryBIT:
    """双 BIT：支持区间修改 + 区间查询，均 O(log n)"""
    def __init__(self, n: int):
        self.n = n
        self.bit1 = BIT(n)   # 存储 D[j]
        self.bit2 = BIT(n)   # 存储 D[j] * j

    def _add(self, bit1: BIT, bit2: BIT, i: int, v: int) -> None:
        bit1.update(i, v)
        bit2.update(i, v * i)

    def range_add(self, l: int, r: int, v: int) -> None:
        """区间 [l, r] 全体加 v，O(log n)"""
        self._add(self.bit1, self.bit2, l, v)
        if r + 1 <= self.n:
            self._add(self.bit1, self.bit2, r + 1, -v)

    def _prefix_sum(self, k: int) -> int:
        """前缀和 A[1] + ... + A[k]，O(log n)"""
        return (k + 1) * self.bit1.query(k) - self.bit2.query(k)

    def range_query(self, l: int, r: int) -> int:
        """区间求和 A[l..r]，O(log n)"""
        return self._prefix_sum(r) - self._prefix_sum(l - 1)
```

```cpp
class RangeUpdateRangeQueryBIT {
    int n;
    BIT bit1, bit2;  // 维护 D[j] 和 D[j]*j

    void add(int i, long long v) {
        bit1.update(i, v);
        bit2.update(i, v * i);
    }

    long long prefixSum(int k) {
        return (long long)(k + 1) * bit1.query(k) - bit2.query(k);
    }
public:
    explicit RangeUpdateRangeQueryBIT(int n) : n(n), bit1(n), bit2(n) {}

    void range_add(int l, int r, long long v) {
        add(l, v);
        if (r + 1 <= n) add(r + 1, -v);
    }

    long long range_query(int l, int r) {
        return prefixSum(r) - prefixSum(l - 1);
    }
};
```

### 37.1.9 二维树状数组

将 BIT 扩展到二维矩阵：`tree2d[i][j]` 管辖矩形 $[i - \text{lowbit}(i) + 1, i] \times [j - \text{lowbit}(j) + 1, j]$。

点更新：双重跳跃，$O(\log n \cdot \log m)$。  
前缀查询（矩形 $[1,i] \times [1,j]$）：双重前缀求和，$O(\log n \cdot \log m)$。

```python
class BIT2D:
    """二维树状数组，n×m 矩阵"""
    def __init__(self, n: int, m: int):
        self.n, self.m = n, m
        self.tree = [[0] * (m + 1) for _ in range(n + 1)]

    def update(self, x: int, y: int, delta: int) -> None:
        """位置 (x, y) 加上 delta，O(log n · log m)"""
        i = x
        while i <= self.n:
            j = y
            while j <= self.m:
                self.tree[i][j] += delta
                j += j & (-j)
            i += i & (-i)

    def query(self, x: int, y: int) -> int:
        """前缀矩形 [1,x]×[1,y] 的和，O(log n · log m)"""
        s = 0
        i = x
        while i > 0:
            j = y
            while j > 0:
                s += self.tree[i][j]
                j -= j & (-j)
            i -= i & (-i)
        return s

    def range_query(self, x1: int, y1: int, x2: int, y2: int) -> int:
        """矩形 [x1,y1]×[x2,y2] 的和（容斥）"""
        return (self.query(x2, y2)
              - self.query(x1 - 1, y2)
              - self.query(x2, y1 - 1)
              + self.query(x1 - 1, y1 - 1))
```

```cpp
class BIT2D {
    int n, m;
    vector<vector<int>> tree;
public:
    BIT2D(int n, int m) : n(n), m(m), tree(n + 1, vector<int>(m + 1, 0)) {}

    void update(int x, int y, int delta) {
        for (int i = x; i <= n; i += i & (-i))
            for (int j = y; j <= m; j += j & (-j))
                tree[i][j] += delta;
    }

    int query(int x, int y) {
        int s = 0;
        for (int i = x; i > 0; i -= i & (-i))
            for (int j = y; j > 0; j -= j & (-j))
                s += tree[i][j];
        return s;
    }

    int range_query(int x1, int y1, int x2, int y2) {
        return query(x2, y2) - query(x1-1, y2)
             - query(x2, y1-1) + query(x1-1, y1-1);
    }
};
```

---

## 37.2 线段树（Segment Tree）

### 37.2.1 线段树的直觉：分而治之的存储

BIT 虽然简洁，但它有局限：BIT 的查询和更新依赖**"前缀和"可以被差分**这一性质，对于求和以外的操作（如区间最大值），BIT 需要非常特殊的技巧才能处理。

**线段树（Segment Tree）**通过显式的区间分治，更加通用。

> **生活比喻**：想象一场足球联赛的积分汇总体系。把全国20支球队分成两组（北区10队，南区10队），每组再分成更小的组……最终每个叶节点负责一支球队。查询任意区间的积分，从根节点"分包"下去，只查询相关的子区间，汇总结果。

线段树的核心设计：
- 根节点管辖整个数组 $[1, n]$
- 每个节点 $v$ 管辖区间 $[l, r]$，其左子管辖 $[l, \text{mid}]$，右子管辖 $[\text{mid}+1, r]$
- 叶节点管辖单个元素 $[i, i]$，存储 $A[i]$

树的深度：$\lceil \log_2 n \rceil$，节点总数：$O(n)$（完全二叉树最多 $4n$ 个节点，所以通常开 $4n$ 大小的数组）。

### 37.2.2 结构表示：数组 vs 指针

**数组表示（推荐用于竞赛）**：节点 $v$ 的左子是 $2v$，右子是 $2v+1$，根是 $1$。开 $4n$ 大小的数组即可。

**指针结构（动态开点）**：适合大值域/稀疏情况，按需申请节点（见 37.3.3）。

竞赛中一般用数组版，代码更简洁，访问更快（无需 `malloc`）。

### 37.2.3 建树 BUILD：$O(n)$

从叶节点出发，逐层向上合并。

```python
class SegTree:
    """
    线段树（数组存储，1-indexed）
    以区间和为例，支持：
      - 点更新 O(log n)
      - 区间查询 O(log n)
    注意：合并函数可替换为 max/min/gcd 等
    """
    def __init__(self, arr: list):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)  # 4n 保证节点足够
        if self.n > 0:
            self._build(arr, 1, 1, self.n)

    def _build(self, arr: list, node: int, l: int, r: int) -> None:
        """
        递归建树
        node: 当前节点编号（1开始）
        [l, r]: 当前节点管辖的区间（1-indexed）
        """
        if l == r:
            # 叶节点：直接赋值
            self.tree[node] = arr[l - 1]  # arr 是 0-indexed
            return
        mid = (l + r) // 2
        self._build(arr, 2 * node, l, mid)          # 递归建左子树
        self._build(arr, 2 * node + 1, mid + 1, r)  # 递归建右子树
        # 向上合并：父节点 = 左子 + 右子
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]
```

```cpp
class SegTree {
    int n;
    vector<long long> tree;  // 4n 大小

    void build(const vector<int>& arr, int node, int l, int r) {
        if (l == r) {
            tree[node] = arr[l - 1];  // arr 0-indexed
            return;
        }
        int mid = (l + r) / 2;
        build(arr, 2*node, l, mid);
        build(arr, 2*node+1, mid+1, r);
        tree[node] = tree[2*node] + tree[2*node+1];  // 合并
    }
public:
    explicit SegTree(const vector<int>& arr) : n(arr.size()), tree(4*arr.size(), 0) {
        build(arr, 1, 1, n);
    }
```

### 37.2.4 点更新 UPDATE：从叶到根，$O(\log n)$

```python
    def update(self, node: int, l: int, r: int, pos: int, val: int) -> None:
        """
        将 A[pos] 设置为 val（单点赋值）
        从根向下找到 pos，然后向上更新路径上的所有节点
        """
        if l == r:
            # 找到叶节点，直接赋值
            self.tree[node] = val
            return
        mid = (l + r) // 2
        if pos <= mid:
            self.update(2 * node, l, mid, pos, val)
        else:
            self.update(2 * node + 1, mid + 1, r, pos, val)
        # ⚠️ 回溯时重新合并：向上更新是关键
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def point_update(self, pos: int, val: int) -> None:
        """外部接口：将 A[pos] 设为 val"""
        self.update(1, 1, self.n, pos, val)
```

```cpp
    void update(int node, int l, int r, int pos, long long val) {
        if (l == r) {
            tree[node] = val;
            return;
        }
        int mid = (l + r) / 2;
        if (pos <= mid) update(2*node, l, mid, pos, val);
        else            update(2*node+1, mid+1, r, pos, val);
        tree[node] = tree[2*node] + tree[2*node+1];  // 向上合并
    }
    void point_update(int pos, long long val) { update(1, 1, n, pos, val); }
```

### 37.2.5 区间查询 QUERY：拆分为 $O(\log n)$ 个节点，$O(\log n)$

**核心思想**：查询 $[ql, qr]$ 时，将其拆分为至多 $O(\log n)$ 个在线段树中"完整对齐"的节点，分别取值合并。

判断规则：
- 若当前节点管辖 $[l, r]$ 完全在查询区间 $[ql, qr]$ 内（$ql \leq l \leq r \leq qr$）：直接返回该节点的值
- 若当前节点和查询区间无交集：返回"不影响结果"的初始值（$0$ 对于求和，$-\infty$ 对于求最大）
- 否则递归拆分

```python
    def query(self, node: int, l: int, r: int, ql: int, qr: int) -> int:
        """
        查询区间 [ql, qr] 的和
        node 管辖 [l, r]
        """
        if ql <= l and r <= qr:
            # 当前节点完全被查询区间覆盖，直接返回
            return self.tree[node]
        if r < ql or l > qr:
            # 完全不相交，返回不影响结果的初始值
            return 0
        # 部分重叠：分两半递归
        mid = (l + r) // 2
        left_val  = self.query(2 * node, l, mid, ql, qr)
        right_val = self.query(2 * node + 1, mid + 1, r, ql, qr)
        return left_val + right_val

    def range_query(self, ql: int, qr: int) -> int:
        """外部接口：查询 A[ql..qr] 的和"""
        return self.query(1, 1, self.n, ql, qr)
```

```cpp
    long long query(int node, int l, int r, int ql, int qr) {
        if (ql <= l && r <= qr) return tree[node];       // 完全覆盖
        if (r < ql || l > qr)   return 0;                // 不相交
        int mid = (l + r) / 2;
        return query(2*node, l, mid, ql, qr)
             + query(2*node+1, mid+1, r, ql, qr);
    }
    long long range_query(int ql, int qr) { return query(1, 1, n, ql, qr); }
```

### 37.2.6 懒惰传播（Lazy Propagation）：区间修改的利器

以上实现只支持**单点更新**。若要**区间修改**（比如把 $[3, 7]$ 的所有元素都加上 5），朴素方法需要 5 次单点更新，$O(5 \log n)$。对于 $n$ 次区间修改，代价是 $O(n^2 \log n)$，无法接受。

**懒惰传播**的思想：
- 当需要修改区间 $[l, r]$ 时，如果某个节点管辖的区间完全被 $[l, r]$ 覆盖，**不立刻递归修改所有子节点，而是在该节点上打一个"懒标记（lazy tag）"**，记录这个区间还有待施加的修改。
- 只有当后续访问需要下探到子节点时，才将懒标记**下推（PUSH-DOWN）**给子节点。

<div data-component="SegmentTreeLazyProp"></div>

```python
class LazySegTree:
    """
    线段树（懒惰传播版）
    支持：区间加法修改 + 区间求和查询，均 O(log n)
    """
    def __init__(self, arr: list):
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)   # 每个节点存区间和
        self.lazy = [0] * (4 * self.n)   # 懒标记：待累加的值
        if self.n > 0:
            self._build(arr, 1, 1, self.n)

    def _build(self, arr: list, node: int, l: int, r: int) -> None:
        if l == r:
            self.tree[node] = arr[l - 1]
            return
        mid = (l + r) // 2
        self._build(arr, 2*node, l, mid)
        self._build(arr, 2*node+1, mid+1, r)
        self.tree[node] = self.tree[2*node] + self.tree[2*node+1]

    def _push_down(self, node: int, l: int, r: int) -> None:
        """
        下推懒标记：将父节点的 lazy 传到两个子节点
        必须在访问子节点之前调用！
        """
        if self.lazy[node] == 0:
            return  # 没有待传播的标记，直接返回
        mid = (l + r) // 2
        left, right = 2 * node, 2 * node + 1

        # 左子树：区间长度 = mid - l + 1
        self.tree[left]  += self.lazy[node] * (mid - l + 1)
        self.lazy[left]  += self.lazy[node]

        # 右子树：区间长度 = r - mid
        self.tree[right] += self.lazy[node] * (r - mid)
        self.lazy[right] += self.lazy[node]

        # 清除自身的懒标记（已传播完毕）
        self.lazy[node] = 0

    def _range_add(self, node: int, l: int, r: int, ql: int, qr: int, val: int) -> None:
        """区间 [ql, qr] 所有元素加 val"""
        if ql <= l and r <= qr:
            # 完全覆盖：直接更新该节点，打懒标记
            self.tree[node] += val * (r - l + 1)
            self.lazy[node] += val
            return
        if r < ql or l > qr:
            return  # 不相交，忽略
        # 部分重叠：先下推，再递归
        self._push_down(node, l, r)  # ⚠️ 在分裂前必须下推！
        mid = (l + r) // 2
        self._range_add(2*node, l, mid, ql, qr, val)
        self._range_add(2*node+1, mid+1, r, ql, qr, val)
        self.tree[node] = self.tree[2*node] + self.tree[2*node+1]

    def _query(self, node: int, l: int, r: int, ql: int, qr: int) -> int:
        if ql <= l and r <= qr:
            return self.tree[node]
        if r < ql or l > qr:
            return 0
        self._push_down(node, l, r)  # ⚠️ 查询时也需要下推！
        mid = (l + r) // 2
        return (self._query(2*node, l, mid, ql, qr)
              + self._query(2*node+1, mid+1, r, ql, qr))

    def range_add(self, ql: int, qr: int, val: int) -> None:
        self._range_add(1, 1, self.n, ql, qr, val)

    def range_query(self, ql: int, qr: int) -> int:
        return self._query(1, 1, self.n, ql, qr)
```

```cpp
class LazySegTree {
    int n;
    vector<long long> tree, lazy;

    void build(const vector<int>& arr, int node, int l, int r) {
        lazy[node] = 0;
        if (l == r) { tree[node] = arr[l-1]; return; }
        int mid = (l+r)/2;
        build(arr, 2*node, l, mid);
        build(arr, 2*node+1, mid+1, r);
        tree[node] = tree[2*node] + tree[2*node+1];
    }

    void pushDown(int node, int l, int r) {
        if (!lazy[node]) return;
        int mid = (l+r)/2;
        // 左子
        tree[2*node]  += lazy[node] * (mid - l + 1);
        lazy[2*node]  += lazy[node];
        // 右子
        tree[2*node+1] += lazy[node] * (r - mid);
        lazy[2*node+1] += lazy[node];
        lazy[node] = 0;
    }

    void rangeAdd(int node, int l, int r, int ql, int qr, long long val) {
        if (ql <= l && r <= qr) {
            tree[node] += val * (r - l + 1);
            lazy[node] += val;
            return;
        }
        if (r < ql || l > qr) return;
        pushDown(node, l, r);   // ⚠️ 递归前先下推
        int mid = (l+r)/2;
        rangeAdd(2*node, l, mid, ql, qr, val);
        rangeAdd(2*node+1, mid+1, r, ql, qr, val);
        tree[node] = tree[2*node] + tree[2*node+1];
    }

    long long query(int node, int l, int r, int ql, int qr) {
        if (ql <= l && r <= qr) return tree[node];
        if (r < ql || l > qr)   return 0;
        pushDown(node, l, r);   // ⚠️ 查询也需要下推
        int mid = (l+r)/2;
        return query(2*node, l, mid, ql, qr)
             + query(2*node+1, mid+1, r, ql, qr);
    }
public:
    explicit LazySegTree(const vector<int>& arr)
        : n(arr.size()), tree(4*arr.size(), 0), lazy(4*arr.size(), 0)
    { build(arr, 1, 1, n); }

    void range_add(int ql, int qr, long long val) { rangeAdd(1,1,n,ql,qr,val); }
    long long range_query(int ql, int qr)          { return query(1,1,n,ql,qr); }
};
```

<Callout type="warning" title="⚠️ 懒惰传播的核心原则">
**PUSH-DOWN 必须在访问子节点之前进行**，不管是在 `update` 还是在 `query` 路径上。一旦遗漏，会导致子节点看到的是"旧数据"，查询结果错误且难以调试。
</Callout>

### 37.2.7 合并函数的设计：求和 / 最大 / 最小 / GCD / 异或

线段树的通用性体现在：只需改变**合并函数（`merge`）和初始值**，就能支持不同的区间操作：

| 操作 | 合并函数 | 初始值（不相交时返回） | 懒标记含义 |
|---|---|---|---|
| 区间求和 | `a + b` | `0` | 区间内每个元素加 $v$ |
| 区间最大值 | `max(a, b)` | $-\infty$ | 区间内每个元素加 $v$（最大值同样加 $v$） |
| 区间最小值 | `min(a, b)` | $+\infty$ | 区间内每个元素加 $v$ |
| 区间 GCD | `gcd(a, b)` | `0` | ——（GCD 不支持区间加法懒传播；单点修改可以） |
| 区间异或 | `a XOR b` | `0` | ——（异或有自己的懒传播方式） |

<Callout type="info" title="💡 懒标记的复合">
当操作是"区间乘 $a$，区间加 $b$"的复合时，懒标记需要存储一个 $(a, b)$ 对，且合并懒标记时要注意顺序：后来的乘 $a'$ 应作用在前的懒标记 $(a, b)$ 上，变为 $(a \cdot a', b \cdot a' + b')$。这类问题称为"等差/等比复合懒标记"。
</Callout>

---

## 37.3 线段树进阶

### 37.3.1 持久化线段树（Persistent Segment Tree / 主席树）

**问题**：需要维护数组的**历史版本**，即：在第 $t$ 次修改后，能查询当时任意区间的状态。

若每次修改都完整复制一棵线段树，空间 $O(n)$ 每次，$k$ 次修改后 $O(kn)$，不可接受。

**关键观察**：每次单点修改只改变从根到叶的路径上的 $O(\log n)$ 个节点，树中其余 $n - O(\log n)$ 个节点与修改前版本**完全相同**。

**持久化策略（路径复制，Path Copying）**：
- 修改时，不修改旧节点，而是**新建**路径上的 $O(\log n)$ 个节点
- 新版本根节点指向新路径，未修改的子树继续指向旧节点（**节点共享**）

每次修改额外空间 $O(\log n)$，$k$ 次修改后总空间 $O(n + k \log n)$。

```
版本 0: root_0 → ... → 旧叶子
            └─大部分节点共享─┘
版本 1: root_1 → 新节点1 → 新节点2 → 新叶子
                  ↓ 共享        ↓ 共享
             旧右子树       旧右子树
```

<div data-component="PersistentSegTreeViz"></div>

**应用**：区间第 $k$ 小值（离线/在线）、历史版本查询、可持久化数据结构族。

**区间第 $k$ 小值思路**：
1. 将数组离散化，值域 $[1, m]$
2. 前缀建持久化线段树：第 $i$ 次插入 $A[i]$ 时建生一个新版本，线段树节点记录"该值域范围内有多少个数"
3. 查询 $[l, r]$ 中第 $k$ 小：两棵版本树同步查询，差值即为区间内的计数，二分在线段树上"走"

```python
class PersistentSegTree:
    """
    持久化线段树：动态指针版（不用数组下标，用节点对象）
    每次修改返回新的根节点，旧版本完整保留
    """
    class Node:
        __slots__ = ['val', 'left', 'right']
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right

    def __init__(self, n: int):
        """n：值域大小（离散化后）"""
        self.n = n
        # 建一个空根（全零）
        self.roots = [self._build(1, n)]

    def _build(self, l: int, r: int) -> 'PersistentSegTree.Node':
        node = self.Node()
        if l == r:
            return node
        mid = (l + r) // 2
        node.left  = self._build(l, mid)
        node.right = self._build(mid + 1, r)
        return node

    def _update(self, old: 'PersistentSegTree.Node',
                l: int, r: int, pos: int, delta: int) -> 'PersistentSegTree.Node':
        """
        新建路径上的节点，未修改部分共享旧节点
        返回新版本的根节点
        """
        new_node = self.Node(old.val + delta, old.left, old.right)
        if l == r:
            return new_node
        mid = (l + r) // 2
        if pos <= mid:
            new_node.left = self._update(old.left, l, mid, pos, delta)
        else:
            new_node.right = self._update(old.right, mid + 1, r, pos, delta)
        new_node.val = new_node.left.val + new_node.right.val
        return new_node

    def insert(self, val: int) -> None:
        """插入值 val（离散化后的索引），产生新版本"""
        new_root = self._update(self.roots[-1], 1, self.n, val, 1)
        self.roots.append(new_root)

    def kth_smallest(self, l_ver: int, r_ver: int, k: int) -> int:
        """
        查询：前缀版本 r_ver 和 l_ver-1 的差，即 A[l..r] 区间中的第 k 小
        l_ver, r_ver 是 roots 的下标
        """
        return self._kth(self.roots[l_ver - 1], self.roots[r_ver], 1, self.n, k)

    def _kth(self, u: 'PersistentSegTree.Node',
             v: 'PersistentSegTree.Node',
             l: int, r: int, k: int) -> int:
        if l == r:
            return l
        mid = (l + r) // 2
        left_cnt = v.left.val - u.left.val  # 左子树在区间 [l, r] 内的计数
        if k <= left_cnt:
            return self._kth(u.left, v.left, l, mid, k)
        else:
            return self._kth(u.right, v.right, mid + 1, r, k - left_cnt)
```

```cpp
struct PersistNode {
    int val, left, right;
    PersistNode() : val(0), left(0), right(0) {}
};

class PersistSegTree {
    int n;
    vector<PersistNode> nodes;
    vector<int> roots;

    int _build(int l, int r) {
        int id = nodes.size();
        nodes.emplace_back();
        if (l == r) return id;
        int mid = (l+r)/2;
        nodes[id].left  = _build(l, mid);
        nodes[id].right = _build(mid+1, r);
        return id;
    }

    int _update(int old_id, int l, int r, int pos, int delta) {
        int id = nodes.size();
        nodes.push_back(nodes[old_id]);  // 复制旧节点
        nodes[id].val += delta;
        if (l == r) return id;
        int mid = (l+r)/2;
        if (pos <= mid)
            nodes[id].left  = _update(nodes[old_id].left,  l, mid,   pos, delta);
        else
            nodes[id].right = _update(nodes[old_id].right, mid+1, r, pos, delta);
        return id;
    }

    int _kth(int u, int v, int l, int r, int k) {
        if (l == r) return l;
        int mid = (l+r)/2;
        int left_cnt = nodes[nodes[v].left].val - nodes[nodes[u].left].val;
        if (k <= left_cnt)
            return _kth(nodes[u].left, nodes[v].left, l, mid, k);
        else
            return _kth(nodes[u].right, nodes[v].right, mid+1, r, k - left_cnt);
    }
public:
    explicit PersistSegTree(int n) : n(n) {
        nodes.reserve(n * 40);  // 预分配足够空间
        roots.push_back(_build(1, n));
    }

    void insert(int val) {
        roots.push_back(_update(roots.back(), 1, n, val, 1));
    }

    int kth_smallest(int l_ver, int r_ver, int k) {
        return _kth(roots[l_ver-1], roots[r_ver], 1, n, k);
    }
};
```

### 37.3.2 线段树合并（Merge Segment Trees）

**问题**：树形 DP 中，每个节点 $u$ 维护一棵线段树（代表 $u$ 子树内的某些信息）。当 DFS 完成 $u$ 的所有子树后，需要将各子树的线段树**合并**成 $u$ 的线段树。

**合并操作**：两棵动态开点线段树按节点位置对应相加（或取 max 等）。

**复杂度**：单次合并 $O(k \log n)$，其中 $k$ 是两棵树中有公共节点的数量。均摊下来，所有合并总时间 $O(n \log n)$（每个节点最多被合并一次）。

```python
def merge_trees(u_id: int, v_id: int, nodes: list) -> int:
    """
    合并两棵动态开点线段树（节点用列表存），返回合并后的根节点 id
    不修改原树节点（可持久化），或修改（非持久化节省空间）
    """
    if u_id == 0: return v_id  # u 为空，直接返回 v
    if v_id == 0: return u_id  # v 为空，直接返回 u
    # 两者都不空：新建合并节点（或修改 u）
    nodes[u_id].val += nodes[v_id].val
    nodes[u_id].left  = merge_trees(nodes[u_id].left,  nodes[v_id].left,  nodes)
    nodes[u_id].right = merge_trees(nodes[u_id].right, nodes[v_id].right, nodes)
    return u_id
```

### 37.3.3 动态开点线段树（节点按需创建）

当值域很大（比如 $[1, 10^9]$）但实际插入元素不多（$\leq n$ 个）时，静态开 $4 \times 10^9$ 的数组不现实。

**动态开点**：只有真正被访问到的节点才被创建，未访问的部分视为初始值（如 $0$）。

每次 `update` 从根到叶新建 $O(\log n)$ 个节点，$n$ 次更新后空间 $O(n \log n)$。

```python
class DynamicSegTree:
    """动态开点线段树：适用于大值域"""
    class Node:
        __slots__ = ['val', 'left', 'right']
        def __init__(self):
            self.val = 0
            self.left = self.right = None

    def __init__(self, lo: int, hi: int):
        self.lo, self.hi = lo, hi   # 全局值域 [lo, hi]
        self.root = None

    def _ensure(self, node: 'DynamicSegTree.Node | None') -> 'DynamicSegTree.Node':
        """懒惰创建节点：只有真正需要时才创建"""
        if node is None:
            return self.Node()
        return node

    def update(self, node: 'DynamicSegTree.Node | None',
               l: int, r: int, pos: int, delta: int) -> 'DynamicSegTree.Node':
        node = self._ensure(node)
        node.val += delta
        if l == r:
            return node
        mid = (l + r) // 2
        if pos <= mid:
            node.left  = self.update(node.left,  l, mid,   pos, delta)
        else:
            node.right = self.update(node.right, mid+1, r, pos, delta)
        return node

    def query(self, node: 'DynamicSegTree.Node | None',
              l: int, r: int, ql: int, qr: int) -> int:
        if node is None: return 0           # 未创建的节点视为 0
        if ql <= l and r <= qr: return node.val
        if r < ql or l > qr: return 0
        mid = (l + r) // 2
        return self.query(node.left, l, mid, ql, qr) + self.query(node.right, mid+1, r, ql, qr)

    def add(self, pos: int, delta: int) -> None:
        self.root = self.update(self.root, self.lo, self.hi, pos, delta)

    def range_sum(self, ql: int, qr: int) -> int:
        return self.query(self.root, self.lo, self.hi, ql, qr)
```

### 37.3.4 李超线段树（Li Chao Tree）：斜率优化 DP 的好帮手

**问题**：动态维护一组直线 $\{y = k_i x + b_i\}$，支持：
- 插入一条新直线
- 查询给定 $x$ 处所有直线的最大值（或最小值）

**直觉**：在一条直线"占优"的 $x$ 区间上，它是最佳直线。将 $x$ 轴用线段树管理，每个节点存"在该节点中点处占优的直线"（即"优势直线"）。

插入直线时，若新直线在节点中点优于当前直线，交换后将"落败"的旧直线下推；反之下推新直线一侧。可以证明：每次插入 $O(\log V)$（$V$ 为 $x$ 值域）。

**应用**：斜率优化（Convex Hull Trick）的在线版本，支持非单调斜率插入/查询。

<Callout type="info" title="💡 李超树 vs 凸包 Trick">
- **凸包 Trick（CHT）**：斜率单调时，用单调队列维护上凸壳，查询 $O(1)$ 均摊
- **李超树**：斜率非单调时，用线段树 $O(\log V)$ 插入与查询，更通用
</Callout>

---

## 37.4 稀疏表（Sparse Table）for RMQ

### 37.4.1 区间最小/最大值查询（Range Minimum Query，RMQ）

**RMQ 问题**：给定静态数组 $A[1..n]$，多次查询区间 $[l, r]$ 上的最小值（或最大值），不做修改。

用线段树解决 RMQ：预处理 $O(n)$，查询 $O(\log n)$，无疑正确。但如果 $n, q$ 极大，还能做到 **$O(1)$ 查询**吗？答案是：可以，这就是**稀疏表（Sparse Table）**。

代价：只适合**静态**数组（建表后不能修改），且需要 $O(n \log n)$ 的预处理时间和空间。

### 37.4.2 $O(n \log n)$ 预处理：覆盖所有 $2^k$ 长的区间

定义 $\text{ST}[i][j]$ = $A[i..i+2^j-1]$ 的最小值，即从位置 $i$ 开始长度为 $2^j$ 的区间的最小值。

$$\text{ST}[i][j] = \min(\text{ST}[i][j-1], \text{ST}[i + 2^{j-1}][j-1])$$

递推关系直觉：长度 $2^j$ 的区间 = 左半（长 $2^{j-1}$）和右半（从 $i + 2^{j-1}$ 开始，长 $2^{j-1}$）的最小值。

$j$ 从 $1$ 递推到 $\lfloor \log_2 n \rfloor$，总共计算 $O(n \log n)$ 个格子，每格 $O(1)$，总预处理 $O(n \log n)$。

<div data-component="SparseTableRMQ"></div>

### 37.4.3 $O(1)$ 查询：重叠区间的奥秘

查询 $[l, r]$ 的最小值时：

1. 令 $k = \lfloor \log_2(r - l + 1) \rfloor$（即最大的满足 $2^k \leq r - l + 1$ 的整数）
2. 返回 $\min(\text{ST}[l][k], \text{ST}[r - 2^k + 1][k])$

**为什么正确？**  
两个区间 $[l, l+2^k-1]$ 和 $[r-2^k+1, r]$ 都是长度 $2^k$ 的区间，**它们合并一定覆盖了整个 $[l, r]$**（因为 $2^k \leq r - l + 1$，所以两端有重叠）。由于 $\min$ 操作是幂等的（可重叠），重叠部分不会影响正确性。

```python
import math

class SparseTable:
    """
    稀疏表，O(n log n) 预处理，O(1) 查询区间最小值
    ⚠️ 仅适合静态数组（无修改）
    """
    def __init__(self, arr: list):
        n = len(arr)
        # LOG[i] = floor(log2(i))，预处理加速查询时计算 k
        LOG = [0] * (n + 1)
        for i in range(2, n + 1):
            LOG[i] = LOG[i // 2] + 1

        k_max = LOG[n] + 1
        # ST[j][i] = min(A[i..i+2^j-1])，注意 j 在外层
        self.ST = [[float('inf')] * n for _ in range(k_max)]
        self.LOG = LOG
        self.n = n

        # 初始化：j=0，每个区间长度为 1
        for i in range(n):
            self.ST[0][i] = arr[i]

        # 递推：j 从 1 到 k_max-1
        for j in range(1, k_max):
            for i in range(n - (1 << j) + 1):
                self.ST[j][i] = min(
                    self.ST[j-1][i],
                    self.ST[j-1][i + (1 << (j-1))]
                )

    def query(self, l: int, r: int) -> int:
        """
        查询 A[l..r] 的最小值（0-indexed）
        时间复杂度：O(1)
        """
        j = self.LOG[r - l + 1]
        return min(self.ST[j][l], self.ST[j][r - (1 << j) + 1])
```

```cpp
class SparseTable {
    int n;
    vector<vector<int>> st;
    vector<int> log2_;  // log2_[i] = floor(log2(i))

public:
    explicit SparseTable(const vector<int>& arr) : n(arr.size()) {
        log2_.resize(n + 1);
        log2_[1] = 0;
        for (int i = 2; i <= n; i++)
            log2_[i] = log2_[i/2] + 1;

        int k = log2_[n] + 1;
        st.assign(k, vector<int>(n, INT_MAX));
        for (int i = 0; i < n; i++) st[0][i] = arr[i];

        for (int j = 1; j < k; j++)
            for (int i = 0; i + (1 << j) <= n; i++)
                st[j][i] = min(st[j-1][i], st[j-1][i + (1<<(j-1))]);
    }

    // O(1) 查询，0-indexed [l, r]
    int query(int l, int r) {
        int j = log2_[r - l + 1];
        return min(st[j][l], st[j][r - (1<<j) + 1]);
    }
};
```

### 37.4.4 Sparse Table vs 线段树：如何选择？

| 特性 | 稀疏表 | 线段树 |
|---|---|---|
| 预处理时间 | $O(n \log n)$ | $O(n)$ |
| 预处理空间 | $O(n \log n)$ | $O(n)$ |
| 查询时间 | $O(1)$ | $O(\log n)$ |
| 支持单点修改 | ❌ 不支持（重建需 $O(n \log n)$） | ✅ $O(\log n)$ |
| 支持区间修改 | ❌ 不支持 | ✅ $O(\log n)$（懒传播）|
| 操作类型 | 仅 RMQ（最小/最大/GCD）| 任意满足结合律的操作 |
| 实现复杂度 | 简单 | 较复杂（懒传播） |

**选择原则**：
- 数组**静态**（不修改）且只查询区间 min/max/GCD → **Sparse Table**（$O(1)$ 查询更快）
- 需要**修改**（单点或区间），或操作复杂（如区间和 + 区间乘） → **线段树**

---

## 37.5 复杂度总结与应用选择

### 37.5.1 各结构复杂度一览

| 结构 | 建表 | 点更新 | 区间更新 | 点查询 | 区间查询 | 空间 |
|---|---|---|---|---|---|---|
| 朴素数组 | $O(1)$ | $O(1)$ | $O(n)$ | $O(1)$ | $O(n)$ | $O(n)$ |
| 前缀和 | $O(n)$ | $O(n)$ | $O(n)$ | — | $O(1)$ | $O(n)$ |
| 差分数组 | $O(n)$ | $O(1)$ | $O(1)$ | $O(n)$ | $O(n)$ | $O(n)$ |
| BIT（基础） | $O(n \log n)$ | $O(\log n)$ | 需双BIT技巧 | — | $O(\log n)$ | $O(n)$ |
| BIT（双BIT） | $O(n \log n)$ | — | $O(\log n)$ | — | $O(\log n)$ | $O(n)$ |
| 线段树 | $O(n)$ | $O(\log n)$ | $O(\log n)$（懒传播）| $O(\log n)$ | $O(\log n)$ | $O(n)$ |
| 稀疏表 | $O(n \log n)$ | ❌ | ❌ | $O(1)$ | $O(1)$ | $O(n \log n)$ |
| 持久化线段树 | $O(n)$ | $O(\log n)$/版本 | — | $O(\log n)$ | $O(\log n)$ | $O(n + k\log n)$ |

### 37.5.2 经典题目对应策略

| 题目类型 | 推荐结构 | 说明 |
|---|---|---|
| 单点修改 + 前缀和 | BIT | 最简洁，常数最小 |
| 区间修改 + 点查询 | 差分 BIT | 差分转化 |
| 区间修改 + 区间求和 | 双 BIT / 懒传播线段树 | 双 BIT 常数小，线段树更通用 |
| 区间 max/min 查询（静态）| Sparse Table | $O(1)$ 查询 |
| 区间 max/min 查询（动态）| 线段树 | 支持修改 |
| 历史版本查询 / 区间第k小 | 持久化线段树 | 路径复制 |
| 树上 DP 合并 | 线段树合并（动态开点） | 均摊 $O(n \log n)$ |
| 大值域稀疏更新 | 动态开点线段树 | 按需创建节点 |
| DP 斜率优化（非单调斜率）| 李超线段树 | 直线插入 + 最值查询 |

---

## 37.6 经典题解精讲

### 37.6.1 LeetCode #307：区间求和（Range Sum Query - Mutable）

**题意**：给定数组，支持单点修改和区间求和查询。

**BIT 解法**（推荐，常数最小）：

```python
class NumArray:
    """LeetCode 307 - BIT 解法"""
    def __init__(self, nums: list) -> None:
        self.n = len(nums)
        self.bit = [0] * (self.n + 1)
        self.nums = [0] * self.n
        for i, v in enumerate(nums):
            self._update_bit(i + 1, v)
            self.nums[i] = v

    def _update_bit(self, i: int, delta: int) -> None:
        while i <= self.n:
            self.bit[i] += delta
            i += i & (-i)

    def _query_bit(self, i: int) -> int:
        s = 0
        while i > 0:
            s += self.bit[i]
            i -= i & (-i)
        return s

    def update(self, index: int, val: int) -> None:
        delta = val - self.nums[index]
        self.nums[index] = val
        self._update_bit(index + 1, delta)  # BIT 是 1-indexed

    def sumRange(self, left: int, right: int) -> int:
        return self._query_bit(right + 1) - self._query_bit(left)
```

```cpp
class NumArray {
    int n;
    vector<int> bit, nums;
    void _update(int i, int delta) {
        for (; i <= n; i += i & (-i)) bit[i] += delta;
    }
    int _query(int i) {
        int s = 0;
        for (; i > 0; i -= i & (-i)) s += bit[i];
        return s;
    }
public:
    NumArray(vector<int>& nums_) : n(nums_.size()), bit(n+1,0), nums(nums_) {
        for (int i = 0; i < n; i++) _update(i+1, nums[i]);
    }
    void update(int index, int val) {
        _update(index + 1, val - nums[index]);
        nums[index] = val;
    }
    int sumRange(int left, int right) {
        return _query(right + 1) - _query(left);
    }
};
```

### 37.6.2 LeetCode #315：计算右侧小于当前元素的个数

**题意**：给定数组 $A$，对每个 $A[i]$，统计 $A[i+1..n-1]$ 中严格小于 $A[i]$ 的元素个数。

**思路**：从右向左扫描，用 BIT 统计"已插入元素中小于当前元素的数量"。

1. **离散化**：将值映射到 $[1, m]$（$m$ 为不同值的数量），保序
2. **从右往左插入**：每次处理 $A[i]$，先查询 BIT 中有多少个元素 $< A[i]$（即前缀和 $[1, A[i]-1]$），然后将 $A[i]$ 插入 BIT
3. 结果就是每次查询的答案

```python
from sortedcontainers import SortedList

def countSmaller_BIT(nums: list) -> list:
    """BIT + 离散化解法，O(n log n)"""
    # 离散化
    sorted_unique = sorted(set(nums))
    rank = {v: i + 1 for i, v in enumerate(sorted_unique)}  # 1-indexed
    m = len(sorted_unique)

    bit = [0] * (m + 1)
    def update(i):
        while i <= m:
            bit[i] += 1
            i += i & (-i)
    def query(i):
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & (-i)
        return s

    result = []
    for num in reversed(nums):
        r = rank[num]
        result.append(query(r - 1))   # 统计已插入的比 num 小的数
        update(r)

    return result[::-1]

# 测试
print(countSmaller_BIT([5, 2, 6, 1]))  # [2, 1, 1, 0]
```

```cpp
vector<int> countSmaller(vector<int>& nums) {
    int n = nums.size();
    // 离散化
    vector<int> sorted_nums = nums;
    sort(sorted_nums.begin(), sorted_nums.end());
    sorted_nums.erase(unique(sorted_nums.begin(), sorted_nums.end()), sorted_nums.end());
    auto rank = [&](int v) {
        return (int)(lower_bound(sorted_nums.begin(), sorted_nums.end(), v)
                     - sorted_nums.begin()) + 1;  // 1-indexed
    };

    int m = sorted_nums.size();
    vector<int> bit(m + 1, 0);
    auto update = [&](int i) { for(; i<=m; i+=i&(-i)) bit[i]++; };
    auto query  = [&](int i) { int s=0; for(; i>0; i-=i&(-i)) s+=bit[i]; return s; };

    vector<int> result(n);
    for (int i = n-1; i >= 0; i--) {
        int r = rank(nums[i]);
        result[i] = query(r - 1);
        update(r);
    }
    return result;
}
```

### 37.6.3 逆序对计数（LeetCode #493 / 归并排序对比）

**题意**：统计数组中逆序对的数量，即 $i < j$ 且 $A[i] > A[j]$ 的 $(i, j)$ 对数。

**BIT 解法**（与 #315 极为相似）：

```python
def mergeCount_BIT(nums: list) -> int:
    """BIT 解法统计逆序对，O(n log n)"""
    # 离散化
    sorted_unique = sorted(set(nums))
    rank = {v: i + 1 for i, v in enumerate(sorted_unique)}
    m = len(sorted_unique)
    bit = [0] * (m + 1)

    def update(i):
        while i <= m:
            bit[i] += 1
            i += i & (-i)

    def query(i):  # 前缀：有多少元素 <= i
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & (-i)
        return s

    count = 0
    for num in reversed(nums):
        r = rank[num]
        # 右边已有的比 num 小的元素数量（即 rank 更小的）
        # 从右往左处理，bit 中存的是 "右边" 已经见过的
        count += query(r - 1)
        update(r)

    return count

print(mergeCount_BIT([5, 2, 6, 1]))   # 4
print(mergeCount_BIT([1, 3, 2, 3, 1]))  # 3
```

---

## 37.7 常见错误与调试技巧

<Callout type="error" title="❌ 错误一：BIT 从 0 开始索引">
BIT **必须从 1 开始**。如果传入 `i=0`，`lowbit(0) = 0`，循环永远不会终止。
始终保持一个转换层：外部使用 0-indexed，内部传入 BIT 时 `+1`。
</Callout>

<Callout type="error" title="❌ 错误二：线段树数组开太小">
线段树节点数 = $4n$（保守估计），不能只开 $2n$。因为满二叉树最底层可能补齐到 $2^{\lceil \log n \rceil}$，最多 $4n$ 个节点。遇到奇怪的越界错误，优先检查数组大小。
</Callout>

<Callout type="error" title="❌ 错误三：忘记 PUSH-DOWN">
懒传播中，**每次递归访问子节点前都必须先 push_down**！在 `query` 函数中遗漏 push_down 是最常见的错误，表现为查询结果不正确，且难以复现（只有区间修改后才会出错）。
</Callout>

<Callout type="error" title="❌ 错误四：Sparse Table 下标越界">
$\text{ST}[j][i]$ 要求 $i + 2^j - 1 \leq n$，即 $i \leq n - 2^j$。在预处理时务必检查右端点不越界：`for i in range(n - (1 << j) + 1)`（注意 `+1`）。
</Callout>

---

## 37.8 面试与竞赛考点速查

**面试高频题型**：
1. **区间求和 + 单点更新**：BIT（$O(\log n)$ 两种操作，代码极简）
2. **区间最大/最小值，静态**：Sparse Table（$O(1)$ 查询）
3. **区间修改 + 区间求和**：懒传播线段树 或 双 BIT
4. **计算逆序对 / 右侧小于当前元素**：BIT + 离散化，从右往左扫描

**竞赛高频技巧**：
1. **离散化 + BIT**：将任意值域映射到 $[1, n]$，BIT 计数
2. **线段树合并**：树形 DP 中每个节点维护独立数值集合
3. **持久化线段树**：区间第 $k$ 小，历史版本差分
4. **李超树**：DP 斜率优化（CHT online 版）

**思考题**：

> 💡 **思考 1**：BIT 能直接支持"区间 max 查询"吗？为什么？（提示：考虑 QUERY 的跳跃路径是否能正确合并 max）
>
> 💡 **思考 2**：线段树和 BIT 解决区间求和问题时，复杂度相同（均 $O(\log n)$），但 BIT 常数因子更小，原因是什么？
>
> 💡 **思考 3**：持久化线段树使用路径复制策略，平均每次 $O(\log n)$ 新节点。如果修改是区间修改（配合懒传播），平均新节点数是多少？会有什么问题？

---

**参考资料**：
- Sedgewick & Wayne, *Algorithms* 4th ed., §3.2（Binary Search Trees）
- Fenwick, P.M. (1994). "A new data structure for cumulative frequency tables". *Software: Practice and Experience*, 24(3)：327–336
- CP-algorithms.com：[Fenwick Tree](https://cp-algorithms.com/data_structures/fenwick.html)、[Segment Tree](https://cp-algorithms.com/data_structures/segment_tree.html)（最权威的实现参考）
- OI-wiki：[树状数组](https://oi-wiki.org/ds/bit/)、[线段树](https://oi-wiki.org/ds/seg/)

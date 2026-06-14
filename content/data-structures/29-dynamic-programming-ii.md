# Chapter 29: 动态规划进阶（Dynamic Programming II）

## 章节导读

Chapter 28 建立了 DP 的基础认知——重叠子问题、最优子结构、Top-Down 与 Bottom-Up。这一章我们进入 **DP 的进阶领域**，解锁五类强大的 DP 模式：

| DP 类型 | 代表题目 | 难点 |
|---|---|---|
| 区间 DP | 矩阵链乘、气球爆炸 | 枚举顺序（先长度再左端点） |
| 树形 DP | 二叉树最大路径、打家劫舍 III | 状态随树结构传递 |
| DAG DP | 最长递增路径 | 拓扑序保证无后效性 |
| 数位 DP | 各位数字之和为 S | tight 约束管理 |
| 状压 DP | 旅行商问题（TSP） | 位运算枚举子集 |

本章还会介绍三种 **DP 优化技术**（单调队列、斜率优化、Knuth 优化），将某些 $O(n^3)$ 的 DP 压缩到 $O(n^2)$ 甚至 $O(n^2 \log n)$。

学完本章，你将能够：

1. 用区间 DP 解决"分治 + 合并代价"类问题，不再被枚举顺序搞混；
2. 在任意树结构上设计父子方向的 DP 状态；
3. 用数位 DP 解决"[1, n] 内满足某数字性质的计数"问题；
4. 用状压 DP 处理 $n \leq 20$ 的集合枚举问题；
5. 理解并正确应用单调队列优化与斜率优化。

---

## 29.1 区间 DP（Interval DP）

### 29.1.1 什么是区间 DP？框架与直觉

**生活比喻**：假设你有一排书，要把它们从左到右合并成一大本。每两本相邻书合并时需要付出成本。你希望选择一种合并顺序，使总成本最小。

这就是区间 DP 的核心场景：**把一段区间 $[l, r]$ 的问题，分解为两个更短的区间之和**，枚举所有可能的"分割点"，取最优。

**通用定义**：

$$dp[l][r] = \min_{k=l}^{r-1} \bigl( dp[l][k] + dp[k+1][r] + \text{cost}(l, r, k) \bigr)$$

其中：
- $dp[l][r]$ 表示处理区间 $[l, r]$ 的最优代价；
- $k$ 是区间的"分割点"或"最后操作的位置"；
- $\text{cost}(l, r, k)$ 是在当前层做的额外代价。

**与其他 DP 的对比**：

| 维度 | 线性 DP（第28章） | 区间 DP |
|---|---|---|
| 状态 | $dp[i]$，表示"前 i 个" | $dp[l][r]$，表示"区间 [l,r]" |
| 转移来源 | 从左往右，依赖 $dp[i-1]$ | 依赖所有更短的子区间 |
| 枚举顺序 | 从小到大直接枚举 $i$ | 先枚举区间长度，再枚举左端点 |
| 典型题目 | LCS、背包 | 矩阵链乘、气球爆炸 |

### 29.1.2 枚举顺序：先枚举区间长度，再枚举左端点

这是区间 DP 最容易出错的地方。我们必须保证：**计算 $dp[l][r]$ 时，所有更短的子区间已经计算完毕**。

正确的枚举顺序：

```
for len in range(2, n+1):        # 先枚举区间长度（从 2 到 n）
    for l in range(0, n-len+1):  # 再枚举左端点
        r = l + len - 1          # 右端点由左端点和长度确定
        for k in range(l, r):    # 枚举分割点
            dp[l][r] = min(dp[l][r], dp[l][k] + dp[k+1][r] + cost)
```

> ⚠️ **常见错误**：写成 `for l in ...: for r in ...: for k in ...`——如果直接枚举左右端点，无法保证短区间先被算出来。

<div data-component="MatrixChainAnimation"></div>

### 29.1.3 矩阵链乘法（Matrix Chain Multiplication）

#### 背景与直觉

计算矩阵乘积 $A_1 \times A_2 \times \cdots \times A_n$ 时，由于矩阵乘法满足结合律，不同的括号方案所需的乘法次数差异极大。

**例子**：$A_{10 \times 100},\ B_{100 \times 5},\ C_{5 \times 50}$

- 方案 1：$(A \times B) \times C$ → $10 \times 100 \times 5 + 10 \times 5 \times 50 = 7500$ 次
- 方案 2：$A \times (B \times C)$ → $100 \times 5 \times 50 + 10 \times 100 \times 50 = 75000$ 次

同样的计算，代价相差 **10 倍**！因此选择最优括号方案至关重要。

#### 状态定义

设矩阵 $A_i$ 的维度为 $p_{i-1} \times p_i$（即输入 $p$ 数组，长度为 $n+1$）。

$dp[i][j]$ = 计算 $A_i \times A_{i+1} \times \cdots \times A_j$ 所需的**最少标量乘法次数**。

#### 状态转移

$$dp[i][j] = \min_{k=i}^{j-1} \bigl( dp[i][k] + dp[k+1][j] + p_{i-1} \times p_k \times p_j \bigr)$$

**直觉**：枚举最后一次"大合并"发生在哪里（在 $A_k$ 和 $A_{k+1}$ 之间分开），左右两段分别递归求最优后，再加上这次合并的代价 $p_{i-1} \times p_k \times p_j$。

#### 初始化

$dp[i][i] = 0$（单个矩阵，不需要乘）。

#### 完整代码

```python
def matrix_chain_order(p: list[int]) -> tuple[int, list[list[int]]]:
    """
    矩阵链乘法最优括号化

    参数：p[i-1] × p[i] 是第 i 个矩阵的维度（共 n 个矩阵，p 长度为 n+1）

    返回：
        - cost: 最小乘法次数
        - s[i][j]: 区间 [i,j] 的最优分割点，用于重建括号方案

    时间复杂度：O(n³)
    空间复杂度：O(n²)

    【枚举顺序关键】：先枚举区间长度 len，再枚举左端点 i
    dp[i][j] 依赖于所有 dp[i][k] 和 dp[k+1][j]（均为更短区间）
    """
    n = len(p) - 1  # 矩阵数量
    INF = float('inf')

    dp = [[INF] * (n + 1) for _ in range(n + 1)]
    s  = [[0]   * (n + 1) for _ in range(n + 1)]   # 记录最优分割点（用于回溯）

    # 单个矩阵：代价为 0
    for i in range(1, n + 1):
        dp[i][i] = 0

    # 枚举区间长度（len = 2 到 n）
    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            for k in range(i, j):          # 枚举分割点
                cost = dp[i][k] + dp[k+1][j] + p[i-1] * p[k] * p[j]
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    s[i][j] = k            # 记录最优分割点

    return dp[1][n], s


def print_optimal_parens(s: list[list[int]], i: int, j: int) -> str:
    """根据 s 数组重建最优括号方案"""
    if i == j:
        return f"A{i}"
    k = s[i][j]
    left  = print_optimal_parens(s, i, k)
    right = print_optimal_parens(s, k + 1, j)
    return f"({left} × {right})"


# 示例：4 个矩阵 A1(30×35), A2(35×15), A3(15×5), A4(5×10)
p = [30, 35, 15, 5, 10]
cost, s = matrix_chain_order(p)
print(f"最优代价：{cost}")                       # 7125
print(f"括号方案：{print_optimal_parens(s, 1, 4)}")  # ((A1 × (A2 × A3)) × A4)
```

```cpp
#include <bits/stdc++.h>
using namespace std;

// 矩阵链乘法最优括号化
// p[i-1] * p[i] = 第 i 个矩阵的维度
// 返回最少标量乘法次数
pair<int, vector<vector<int>>> matrixChainOrder(vector<int>& p) {
    int n = p.size() - 1;  // 矩阵数量
    // dp[i][j] = 将 A_i ... A_j 相乘的最少次数（1-indexed）
    vector<vector<int>> dp(n + 1, vector<int>(n + 1, INT_MAX));
    vector<vector<int>> s(n + 1, vector<int>(n + 1, 0));  // 最优分割点

    for (int i = 1; i <= n; i++) dp[i][i] = 0;

    // 先枚举区间长度
    for (int len = 2; len <= n; len++) {
        for (int i = 1; i <= n - len + 1; i++) {
            int j = i + len - 1;
            for (int k = i; k < j; k++) {
                long long cost = (long long)dp[i][k] + dp[k+1][j]
                               + (long long)p[i-1] * p[k] * p[j];
                if (cost < dp[i][j]) {
                    dp[i][j] = (int)cost;
                    s[i][j]  = k;
                }
            }
        }
    }
    return {dp[1][n], s};
}

string printParens(vector<vector<int>>& s, int i, int j) {
    if (i == j) return "A" + to_string(i);
    int k = s[i][j];
    return "(" + printParens(s, i, k) + " × " + printParens(s, k+1, j) + ")";
}

int main() {
    vector<int> p = {30, 35, 15, 5, 10};
    auto [cost, s] = matrixChainOrder(p);
    cout << "最优代价: " << cost << endl;  // 7125
    cout << "括号方案: " << printParens(s, 1, 4) << endl;
    return 0;
}
```

**复杂度**：时间 $O(n^3)$，空间 $O(n^2)$（含回溯数组 $s$）。

---

### 29.1.4 石子合并（区间 DP 经典变体）

**问题描述**：一排 $n$ 堆石子，每次只能合并相邻两堆，合并代价等于两堆石子数之和。求合并成一堆的最小（或最大）代价。

**状态**：$dp[l][r]$ = 合并 $[l, r]$ 区间所有石子的最小代价。

**转移**：

$$dp[l][r] = \min_{k=l}^{r-1} \bigl( dp[l][k] + dp[k+1][r] + \text{sum}(l, r) \bigr)$$

注意 $\text{sum}(l, r)$ 是区间 $[l, r]$ 的石子总数，可以用前缀和 $O(1)$ 查询。

```python
def stone_merge_min(stones: list[int]) -> int:
    """
    石子合并 —— 最小代价

    时间复杂度：O(n³)
    空间复杂度：O(n²)

    使用前缀和 prefix[i] = stones[0] + ... + stones[i-1]
    sum(l, r) = prefix[r+1] - prefix[l]
    """
    n = len(stones)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + stones[i]

    INF = float('inf')
    dp = [[INF] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 0  # 单堆，代价 0

    for length in range(2, n + 1):
        for l in range(n - length + 1):
            r = l + length - 1
            cost_merge = prefix[r + 1] - prefix[l]  # 每次合并都要付出这个总代价
            for k in range(l, r):
                dp[l][r] = min(dp[l][r], dp[l][k] + dp[k+1][r] + cost_merge)

    return dp[0][n - 1]


# 示例
print(stone_merge_min([3, 1, 4, 1, 5]))  # 33
```

```cpp
int stoneMergeMin(vector<int>& stones) {
    int n = stones.size();
    vector<int> prefix(n + 1, 0);
    for (int i = 0; i < n; i++) prefix[i+1] = prefix[i] + stones[i];

    vector<vector<int>> dp(n, vector<int>(n, INT_MAX));
    for (int i = 0; i < n; i++) dp[i][i] = 0;

    for (int len = 2; len <= n; len++) {
        for (int l = 0; l <= n - len; l++) {
            int r = l + len - 1;
            int sumCost = prefix[r+1] - prefix[l];
            for (int k = l; k < r; k++) {
                if (dp[l][k] != INT_MAX && dp[k+1][r] != INT_MAX) {
                    dp[l][r] = min(dp[l][r], dp[l][k] + dp[k+1][r] + sumCost);
                }
            }
        }
    }
    return dp[0][n-1];
}
```

---

### 29.1.5 气球爆炸（LeetCode #312）："最后爆炸"的反向思维

#### 问题

给定 $n$ 个气球，第 $i$ 个有价值 $\text{nums}[i]$。当你戳爆第 $i$ 个气球时，获得 $\text{nums}[i-1] \times \text{nums}[i] \times \text{nums}[i+1]$ 的金币（边界外视为 1）。求能获得的最大金币。

#### 为什么常规区间 DP 失效？

直觉上，我们想定义 $dp[l][r]$ = 戳爆 $[l, r]$ 内所有气球获得的最大金币。但如果枚举第一个戳破的气球 $k$，那戳破 $k$ 后左右两边的边界会变化，导致子问题不独立——左侧区间的右边界会随 $k$ 的消失而改变。

#### 巧妙转化："最后一个"被戳破的思维

**关键洞见**：改为枚举区间 $[l, r]$ 中**最后一个**被戳破的气球 $k$。

当 $k$ 是最后一个被戳破的，$k$ 的左右两侧的气球此时是 $l-1$ 和 $r+1$（因为 $[l, k-1]$ 和 $[k+1, r]$ 的气球都已消失）。这样左右子问题就独立了！

#### 状态

为方便处理边界，在 $\text{nums}$ 首尾各填充一个 1：$\text{nums} = [1, \text{nums}, 1]$，下标从 0 到 $n+1$。

$dp[l][r]$ = 戳爆开区间 $(l, r)$ 内的所有气球能获得的最大金币（$l$ 和 $r$ 本身不在此范围内，作为边界保留）。

#### 状态转移

$$dp[l][r] = \max_{k=l+1}^{r-1} \bigl( dp[l][k] + dp[k][r] + \text{nums}[l] \times \text{nums}[k] \times \text{nums}[r] \bigr)$$

```python
def maxCoins(nums: list[int]) -> int:
    """
    气球爆炸（LeetCode #312）

    【转化思路】：枚举"最后一个"被戳破的气球 k
    - dp[l][r] = 开区间 (l, r) 内所有气球全部戳完获得的最大金币
    - k 是区间 (l, r) 内最后一个被戳的，此时两侧邻居是 nums[l] 和 nums[r]
    - 所以 k 被戳时的金币 = nums[l] * nums[k] * nums[r]

    时间复杂度：O(n³)   空间复杂度：O(n²)

    【边界处理】：在 nums 首尾填充 1，统一处理边界
    """
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0] * n for _ in range(n)]

    # 枚举区间长度（开区间 (l,r)，区间内至少有 1 个气球，故 r - l >= 2）
    for length in range(2, n):
        for l in range(0, n - length):
            r = l + length
            for k in range(l + 1, r):   # k 是区间 (l,r) 内最后被戳的
                val = (dp[l][k] + dp[k][r]
                       + nums[l] * nums[k] * nums[r])
                dp[l][r] = max(dp[l][r], val)

    return dp[0][n - 1]


print(maxCoins([3, 1, 5, 8]))   # 167
print(maxCoins([1, 5]))         # 10
```

```cpp
int maxCoins(vector<int>& nums) {
    nums.insert(nums.begin(), 1);
    nums.push_back(1);
    int n = nums.size();
    vector<vector<int>> dp(n, vector<int>(n, 0));

    for (int len = 2; len < n; len++) {
        for (int l = 0; l < n - len; l++) {
            int r = l + len;
            for (int k = l + 1; k < r; k++) {
                int val = dp[l][k] + dp[k][r]
                        + nums[l] * nums[k] * nums[r];
                dp[l][r] = max(dp[l][r], val);
            }
        }
    }
    return dp[0][n - 1];
}
```

> **易错总结**：
> - $dp[l][r]$ 是**开区间**，枚举 $k \in (l, r)$（不含边界）；
> - 转移是 $dp[l][k] + dp[k][r]$（**不是** $dp[l][k-1] + dp[k+1][r]$）。

---

## 29.2 树形 DP（Tree DP）

### 29.2.1 框架：状态随树结构传递

**生活比喻**：你是一个公司老板，要决定哪些员工可以参加派对（但直属上下级不能同时参加）。每个员工有一个"快乐值"。这就是树形 DP 的经典场景——在树的父子关系约束下，进行最优决策。

**核心思路**：

1. 对每个节点 $u$，定义与 $u$ 有关的 DP 状态，通常是 `dp[u][0]` 和 `dp[u][1]`（不选/选 $u$）；
2. 采用**后序遍历**（Post-order DFS）：先递归处理子节点，再用子节点的结果更新父节点；
3. 根节点处取最终答案。

**通用模板**：

```python
def dfs(node, parent):
    dp[node][0] = 0   # 不含 node 的最优值（基准为 0）
    dp[node][1] = val[node]  # 含 node 的最优值（基准为 node 自身的贡献）

    for child in children[node]:
        if child == parent:
            continue
        dfs(child, node)

        # 如果不选 node：子节点可选可不选
        dp[node][0] += max(dp[child][0], dp[child][1])
        # 如果选 node：子节点不能选
        dp[node][1] += dp[child][0]
```

<div data-component="TreeDPRerooting"></div>

### 29.2.2 二叉树最大路径和（LeetCode #124）

#### 问题

路径可以以任何节点为起点和终点，且路径中至少含一个节点。找出二叉树中所有路径中节点值之和最大的路径。

#### 关键状态设计

这题设计**两个互斥的 DP 值**：

- `max_gain(node)`：以 `node` 为**端点**（只向一个子方向延伸）能贡献给父节点的最大值 = 单侧最大贡献；
- `max_path`（全局变量）：经过 `node` 并**拐弯**的最大路径和（`node` 是拐点）。

拐弯不能上报给父节点（因为路径不能两头都朝下延伸再向上），所以分开维护。

```python
class Solution:
    def maxPathSum(self, root) -> int:
        """
        二叉树最大路径和（#124）

        【核心区分】：
        - gain(node) = 以 node 为端点，向下延伸的最大单侧贡献
          （用于上报给父节点，只能选左或右中的一边）
        - ans 全局记录：以 node 为拐点时的路径和
          = node.val + max(0, gain(left)) + max(0, gain(right))

        【边界】：负贡献不如不取，所以 gain 下限为 0

        时间复杂度：O(n)   空间复杂度：O(h)（递归栈）
        """
        self.ans = float('-inf')

        def gain(node) -> int:
            if not node:
                return 0
            left  = max(0, gain(node.left))   # 负数贡献舍弃
            right = max(0, gain(node.right))

            # 以 node 为拐点的路径和（不向上传递）
            self.ans = max(self.ans, node.val + left + right)

            # 向父节点传递时，只能选一个方向
            return node.val + max(left, right)

        gain(root)
        return self.ans
```

```cpp
struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    int ans = INT_MIN;

    int gain(TreeNode* node) {
        if (!node) return 0;
        int left  = max(0, gain(node->left));
        int right = max(0, gain(node->right));
        // 以 node 为拐点的路径和
        ans = max(ans, node->val + left + right);
        // 向上传递：只能选一侧
        return node->val + max(left, right);
    }

    int maxPathSum(TreeNode* root) {
        gain(root);
        return ans;
    }
};
```

**复杂度**：时间 $O(n)$（每个节点访问一次），空间 $O(h)$（递归栈深度）。

---

### 29.2.3 打家劫舍 III（LeetCode #337）：树形 DP 的 0/1 状态

#### 问题

树形版的打家劫舍：不能同时抢直接相连的两个节点，求最大总金额。

#### 状态设计

`dp(node) -> (rob, skip)`：

- `rob` = 选择抢该节点时，以该节点为根的子树最大收益；
- `skip` = 不抢该节点时，以该节点为根的子树最大收益。

```python
def rob(root) -> int:
    """
    打家劫舍 III（#337）

    返回从 root 开始的子树：(选root的最大值, 不选root的最大值)

    【转移】：
    - 选 root：左右子节点均不能选
      rob = root.val + skip(left) + skip(right)
    - 不选 root：左右子节点可选可不选
      skip = max(rob(left), skip(left)) + max(rob(right), skip(right))

    时间复杂度：O(n)   空间复杂度：O(h)
    """
    def dfs(node):
        if not node:
            return (0, 0)   # (rob, skip)
        left_rob,  left_skip  = dfs(node.left)
        right_rob, right_skip = dfs(node.right)

        # 选当前节点：子节点不能选
        rob  = node.val + left_skip + right_skip
        # 不选当前节点：子节点取两种状态的最大值
        skip = max(left_rob, left_skip) + max(right_rob, right_skip)

        return (rob, skip)

    rob_val, skip_val = dfs(root)
    return max(rob_val, skip_val)
```

```cpp
pair<int, int> dfs(TreeNode* node) {
    if (!node) return {0, 0};
    auto [lr, ls] = dfs(node->left);
    auto [rr, rs] = dfs(node->right);

    int rob  = node->val + ls + rs;                // 选当前节点
    int skip = max(lr, ls) + max(rr, rs);          // 不选
    return {rob, skip};
}

int rob(TreeNode* root) {
    auto [r, s] = dfs(root);
    return max(r, s);
}
```

---

### 29.2.4 树的直径（Tree Diameter）

树的直径是树中距离最长的两个节点之间的路径长度（以边数或节点数计）。

**方法 1：两次 BFS/DFS**（简单但不适合权重为负的情况）

- 从任意节点 $s$ 出发 BFS，找到最远节点 $u$；
- 从 $u$ 出发 BFS，找到最远节点 $v$；
- $u$ 到 $v$ 的距离即为直径。

**方法 2：树形 DP**（更通用，支持带权树）

对每个节点 $u$：`depth(u)` = 以 $u$ 为根，往下延伸的最大链长。

过 $u$ 的最长路径 = 最长两条链之和。遍历所有节点取最大。

```python
def tree_diameter(n: int, edges: list[tuple[int,int]]) -> int:
    """
    树的直径（树形 DP，以边数计）

    时间复杂度：O(n)
    """
    from collections import defaultdict
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    diameter = 0

    def dfs(node: int, parent: int) -> int:
        nonlocal diameter
        # 记录最长和次长链
        best1 = best2 = 0
        for child in graph[node]:
            if child == parent:
                continue
            child_depth = dfs(child, node) + 1  # 向下延伸
            if child_depth >= best1:
                best2 = best1
                best1 = child_depth
            elif child_depth > best2:
                best2 = child_depth

        diameter = max(diameter, best1 + best2)  # 过 node 的最长路
        return best1  # 向父节点上报最长链

    dfs(0, -1)
    return diameter


# 示例：路径 0-1-2-3-4，直径为 4
edges = [(0,1),(1,2),(2,3),(3,4)]
print(tree_diameter(5, edges))  # 4
```

```cpp
int diameter = 0;

int dfs(int node, int parent, vector<vector<int>>& graph) {
    int best1 = 0, best2 = 0;  // 最长、次长子链
    for (int child : graph[node]) {
        if (child == parent) continue;
        int d = dfs(child, node, graph) + 1;
        if (d >= best1) { best2 = best1; best1 = d; }
        else if (d > best2) best2 = d;
    }
    diameter = max(diameter, best1 + best2);
    return best1;
}

int treeDiameter(int n, vector<pair<int,int>>& edges) {
    vector<vector<int>> graph(n);
    for (auto [u, v] : edges) {
        graph[u].push_back(v);
        graph[v].push_back(u);
    }
    diameter = 0;
    dfs(0, -1, graph);
    return diameter;
}
```

---

### 29.2.5 换根 DP（Rerooting / Re-rooting DP）

**场景**：某些问题中，最终答案是以**任意节点为根**时的某个子树 DP 值的最大/最小值。如果对每个节点分别 DFS 会是 $O(n^2)$，换根 DP 可以在 $O(n)$ 完成。

**经典题**：每个节点到树中所有其他节点的距离之和（LeetCode #834）。

**两趟 DFS**：

1. **第一趟**（自底向上）：以任意节点（如节点 0）为根，计算每个节点的子树 DP 值；
2. **第二趟**（自顶向下）：将根从父节点"移动"到子节点，更新 DP 值。

```python
def sum_of_distances(n: int, edges: list[list[int]]) -> list[int]:
    """
    换根 DP：每个节点到所有其他节点的距离之和（类似 LeetCode #834）

    【两趟 DFS】：
    1. 第一趟：以节点 0 为根，计算：
       - subtree_size[u]：子树节点数（含 u）
       - ans[0]：以 0 为根时，0 到所有节点的距离和
    2. 第二趟：换根从 u 到子节点 v 时：
       - 子树中 subtree_size[v] 个节点，距离 -1
       - 其余 n - subtree_size[v] 个节点，距离 +1
       - ans[v] = ans[u] - subtree_size[v] + (n - subtree_size[v])

    时间复杂度：O(n)
    """
    from collections import defaultdict
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    subtree_size = [1] * n
    ans = [0] * n

    # 第一趟：自底向上，计算 subtree_size 和 ans[0]
    def dfs1(node, parent):
        for child in graph[node]:
            if child == parent:
                continue
            dfs1(child, node)
            subtree_size[node] += subtree_size[child]
            ans[node] += ans[child] + subtree_size[child]

    # 第二趟：自顶向下，换根
    def dfs2(node, parent):
        for child in graph[node]:
            if child == parent:
                continue
            # 换根：从 node 换到 child
            ans[child] = ans[node] - subtree_size[child] + (n - subtree_size[child])
            dfs2(child, node)

    dfs1(0, -1)
    dfs2(0, -1)
    return ans


# 示例：线型树 0-1-2
print(sum_of_distances(3, [[0,1],[1,2]]))  # [3, 2, 3]
```

```cpp
vector<int> subtree_size, ans;
vector<vector<int>> graph;

void dfs1(int node, int parent) {
    for (int child : graph[node]) {
        if (child == parent) continue;
        dfs1(child, node);
        subtree_size[node] += subtree_size[child];
        ans[node] += ans[child] + subtree_size[child];
    }
}

void dfs2(int node, int parent, int n) {
    for (int child : graph[node]) {
        if (child == parent) continue;
        ans[child] = ans[node] - subtree_size[child] + (n - subtree_size[child]);
        dfs2(child, node, n);
    }
}

vector<int> sumOfDistances(int n, vector<vector<int>>& edges) {
    subtree_size.assign(n, 1);
    ans.assign(n, 0);
    graph.assign(n, {});
    for (auto& e : edges) { graph[e[0]].push_back(e[1]); graph[e[1]].push_back(e[0]); }
    dfs1(0, -1);
    dfs2(0, -1, n);
    return ans;
}
```

---

## 29.3 DAG 上的 DP

### 29.3.1 DAG 最长路径：拓扑排序 + DP

**直觉**：DAG（有向无环图）天然满足 DP 的**无后效性**——拓扑序早的节点先被处理，后面的节点依赖前面已算出的值。

**状态**：$dp[u]$ = 从某起点到 $u$ 的最长路径长度。

**转移**：对有向边 $u \to v$，$dp[v] = \max(dp[v], dp[u] + w(u, v))$。

**枚举顺序**：按拓扑序处理节点（用 Kahn 算法或 DFS 逆后序）。

```python
from collections import defaultdict, deque

def dag_longest_path(n: int, edges: list[tuple[int, int, int]]) -> int:
    """
    DAG 最长路径

    参数：
        n: 节点数（0-indexed）
        edges: (u, v, w) 有向边

    时间复杂度：O(V + E)

    【设计考量】：
    - 不同于普通图（最长路 NP-hard），DAG 上因为无环，
      拓扑序保证了计算 dp[v] 时 dp[u] 已确定
    - 如果需要从"任意节点"出发，添加超级源点 s，s->每个节点的边权为 0
    """
    graph = defaultdict(list)
    indegree = [0] * n
    for u, v, w in edges:
        graph[u].append((v, w))
        indegree[v] += 1

    # Kahn 算法：拓扑排序
    queue = deque(i for i in range(n) if indegree[i] == 0)
    dp = [0] * n
    topo_order = []

    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v, w in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)

    # 按拓扑序更新 dp
    for u in topo_order:
        for v, w in graph[u]:
            if dp[u] + w > dp[v]:
                dp[v] = dp[u] + w

    return max(dp)


# 示例
edges = [(0,1,3), (0,2,2), (1,3,4), (2,3,1)]
print(dag_longest_path(4, edges))  # 7（路径 0→1→3，3+4=7）
```

```cpp
int dagLongestPath(int n, vector<tuple<int,int,int>>& edges) {
    vector<vector<pair<int,int>>> graph(n);
    vector<int> indegree(n, 0);
    for (auto [u, v, w] : edges) {
        graph[u].push_back({v, w});
        indegree[v]++;
    }

    queue<int> q;
    for (int i = 0; i < n; i++)
        if (indegree[i] == 0) q.push(i);

    vector<int> dp(n, 0);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (auto [v, w] : graph[u]) {
            dp[v] = max(dp[v], dp[u] + w);
            if (--indegree[v] == 0) q.push(v);
        }
    }
    return *max_element(dp.begin(), dp.end());
}
```

### 29.3.2 矩阵中最长递增路径（LeetCode #329）

这是 DAG DP 的一个隐含形式：把矩阵中每个格子视为 DAG 的一个节点，只有值严格递增的格子之间才有边（四个方向）。

```python
def longestIncreasingPath(matrix: list[list[int]]) -> int:
    """
    矩阵最长递增路径（#329）

    【隐式 DAG】：节点是格子，边是"值严格递增"的相邻关系
    【记忆化 DFS】：等价于 DAG 上的拓扑 DP
    时间复杂度：O(m * n)  每个格子只被计算一次
    """
    if not matrix or not matrix[0]:
        return 0
    m, n = len(matrix), len(matrix[0])
    memo = {}

    def dfs(i: int, j: int) -> int:
        if (i, j) in memo:
            return memo[(i, j)]
        best = 1
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and matrix[ni][nj] > matrix[i][j]:
                best = max(best, 1 + dfs(ni, nj))
        memo[(i, j)] = best
        return best

    return max(dfs(i, j) for i in range(m) for j in range(n))
```

```cpp
class Solution {
    int m, n;
    vector<vector<int>> memo;
    int dirs[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};

    int dfs(vector<vector<int>>& mat, int i, int j) {
        if (memo[i][j]) return memo[i][j];
        memo[i][j] = 1;
        for (auto& d : dirs) {
            int ni = i+d[0], nj = j+d[1];
            if (ni>=0 && ni<m && nj>=0 && nj<n && mat[ni][nj] > mat[i][j])
                memo[i][j] = max(memo[i][j], 1 + dfs(mat, ni, nj));
        }
        return memo[i][j];
    }
public:
    int longestIncreasingPath(vector<vector<int>>& matrix) {
        m = matrix.size(); n = matrix[0].size();
        memo.assign(m, vector<int>(n, 0));
        int ans = 0;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                ans = max(ans, dfs(matrix, i, j));
        return ans;
    }
};
```

---

## 29.4 数位 DP（Digit DP）

### 29.4.1 数位 DP 的框架与直觉

**适用场景**：统计 $[1, n]$ 内满足某种**数字性质**的整数个数。

**生活比喻**：你要统计 1 到 10000 中，各位数字之和等于 10 的数有多少个。逐一枚举太慢，数位 DP 让你按位从高到低"构造"数字，同时追踪"当前状态"（已选了哪些位，前缀大小关系等）。

**关键概念**：

- **tight（紧约束）**：当前是否每一位都和数字 $n$ 的对应位相同。如果 `tight = True`，下一位的上限是 $n$ 对应位；如果 `tight = False`，下一位可以取 0–9；
- **pos**：当前处理的位数（从高位到低位）；
- **leading_zero**：是否还在前导零阶段（处理如 "007" 被识别为 7 的问题）；
- 其他状态：根据具体问题，如数字之和 `sum`、是否含特定数字等。

**通用模板**：

```python
def digit_dp(n: int) -> int:
    """
    数位 DP 通用模板

    以"统计 [0, n] 内各位数字之和 = target 的数个数"为例

    【状态参数】：
    - pos: 当前处理的位（从高到低）
    - current_sum: 当前已选位的数字之和
    - tight: 是否还受上界约束
    - leading_zero: 是否还是前导零状态
    """
    digits = [int(d) for d in str(n)]
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dp(pos: int, current_sum: int, tight: bool, leading_zero: bool) -> int:
        # 到达末位：判断是否满足条件（此处以 sum == target 为例）
        if pos == len(digits):
            return 1 if (not leading_zero and current_sum == target) else 0

        limit = digits[pos] if tight else 9
        result = 0

        for digit in range(0, limit + 1):
            new_tight = tight and (digit == limit)
            new_leading = leading_zero and (digit == 0)
            new_sum = current_sum if new_leading else current_sum + digit
            result += dp(pos + 1, new_sum, new_tight, new_leading)

        return result

    target = 10  # 示例：数字之和为 10
    return dp(0, 0, True, True)
```

<div data-component="DigitDPViz"></div>

### 29.4.2 具体问题：[1, n] 内不含数字 4 的整数个数

```python
def count_without_4(n: int) -> int:
    """
    统计 [1, n] 内不含数字 4 的整数个数

    状态：(pos, tight, leading_zero)
    这里不需要 sum，只需要判断"是否出现了 4"
    """
    digits = [int(d) for d in str(n)]
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dp(pos: int, tight: bool, has4: bool) -> int:
        if pos == len(digits):
            return 0 if has4 else 1   # 不含 4 则计数
        limit = digits[pos] if tight else 9
        count = 0
        for d in range(0, limit + 1):
            new_has4 = has4 or (d == 4)
            count += dp(pos + 1, tight and (d == limit), new_has4)
        return count

    return dp(0, True, False)


print(count_without_4(100))   # 72
print(count_without_4(1000))  # 648
```

```cpp
string digits_str;

// 数位 DP：统计 [0, n] 中不含数字 4 的整数个数
// memo[pos][tight][has4]
int memo[20][2][2];

int dp(int pos, bool tight, bool has4) {
    if (pos == (int)digits_str.size())
        return has4 ? 0 : 1;
    if (memo[pos][tight][has4] != -1)
        return memo[pos][tight][has4];

    int limit = tight ? (digits_str[pos] - '0') : 9;
    int count = 0;
    for (int d = 0; d <= limit; d++) {
        count += dp(pos + 1, tight && (d == limit), has4 || (d == 4));
    }
    return memo[pos][tight][has4] = count;
}

int countWithout4(int n) {
    digits_str = to_string(n);
    memset(memo, -1, sizeof(memo));
    return dp(0, true, false);
}
```

### 29.4.3 数位 DP 的状态压缩技巧

**核心要点**：

| 原则 | 说明 |
|---|---|
| `lru_cache` 参数全部可哈希 | Python 中 bool 可以，list 不行，改用 tuple 或 int 压缩 |
| C++ 中用多维数组 | `memo[pos][tight][state]`，先清空为 -1 |
| `tight` 只会是 True/False | 一旦某一位选了 < limit，后续所有位 tight = False |
| 重置 cache | 每次调用前记得 `dp.cache_clear()`（Python LRU） |

---

## 29.5 状压 DP（Bitmask DP）

### 29.5.1 集合压缩为整数

当集合元素数量 $n \leq 20$ 时，可以用一个整数的二进制位来表示集合的子集状态：

- 第 $i$ 位为 1，表示元素 $i$ 在集合中；
- 全集：$(1 << n) - 1$；
- 判断 $i$ 是否在集合 $S$：`S >> i & 1`；
- 将 $i$ 加入集合 $S$：`S | (1 << i)`；
- 将 $i$ 从集合 $S$ 移除：`S & ~(1 << i)`；
- 枚举 $S$ 的所有子集：`sub = S; while sub: ... sub = (sub-1) & S`。

**复杂度**：状态数 $O(2^n)$，若每个状态需要枚举转移则为 $O(2^n \times n)$；枚举所有子集的子集共 $O(3^n)$（每个元素有在、不在当前子集、不在原集合三种状态）。

```python
# 位运算基础示例
n = 4  # 4 个元素（编号 0-3）

S = 0b1010  # = 集合 {1, 3}
print(f"集合 S = {{{', '.join(str(i) for i in range(n) if S >> i & 1)}}}")  # {1, 3}

# 加入元素 2
S2 = S | (1 << 2)   # 0b1110 = {1, 2, 3}
# 删除元素 3
S3 = S & ~(1 << 3)  # 0b0010 = {1}

# 枚举 S 的所有非空子集
sub = S
while sub:
    print(sub, end=' ')  # 1010, 1000, 0010
    sub = (sub - 1) & S
```

```cpp
// 枚举 S 的所有子集（含空集）
void enumSubsets(int S) {
    for (int sub = S; sub > 0; sub = (sub - 1) & S) {
        // 处理子集 sub
        cout << sub << " ";
    }
    // 注意：上面的循环不包含 sub=0（空集）
}
```

### 29.5.2 旅行商问题（TSP）：状压 DP 经典

**问题**：$n$ 个城市，已知任意两城市之间的距离 $\text{dist}[i][j]$。从城市 0 出发，恰好经过每个城市一次，最终回到城市 0，求最短总路程。

**状态**：$dp[S][i]$ = 从起点出发，**已经经过集合 $S$ 中所有城市**，**当前在城市 $i$** 时的最短路程。

**转移**：从状态 $(S, i)$ 出发，下一步去城市 $j$（$j \notin S$）：

$$dp[S | (1 << j)][j] = \min\bigl(dp[S | (1 << j)][j],\ dp[S][i] + \text{dist}[i][j]\bigr)$$

**初始化**：$dp[1 << 0][0] = 0$（从城市 0 出发，只经过了城市 0，代价为 0），其余为 $\infty$。

**最终答案**：$\min_{i=1}^{n-1} \bigl( dp[(1<<n)-1][i] + \text{dist}[i][0] \bigr)$。

**复杂度**：状态数 $O(2^n \times n)$，每个状态枚举下一城市 $O(n)$，总计 $O(2^n \times n^2)$。$n = 20$ 时约 $4 \times 10^7$，可接受。

<div data-component="BitmaskDPTSP"></div>

```python
def tsp(n: int, dist: list[list[int]]) -> int:
    """
    旅行商问题（TSP）—— 状压 DP

    参数：
        n: 城市数量
        dist[i][j]: 城市 i 到 j 的距离（若不可达设 INF）

    返回：从城市 0 出发经过所有城市返回 0 的最短路

    时间复杂度：O(2^n × n²)  适用 n ≤ 20
    空间复杂度：O(2^n × n)

    【初始化】：dp[1<<0][0] = 0，表示从 0 出发只经过 0
    【终止】：访问过全部 n 个城市（状态 = (1<<n)-1）后回到 0
    """
    INF = float('inf')
    FULL = (1 << n) - 1  # 全集

    # dp[S][i] = 已访问集合 S，当前在 i，的最短路
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1 << 0][0] = 0   # 从城市 0 出发

    for S in range(1 << n):
        for i in range(n):
            if dp[S][i] == INF:
                continue
            if not (S >> i & 1):    # 城市 i 不在集合 S 中，状态无效
                continue
            for j in range(n):
                if S >> j & 1:      # 城市 j 已访问，跳过
                    continue
                nS = S | (1 << j)
                if dp[S][i] + dist[i][j] < dp[nS][j]:
                    dp[nS][j] = dp[S][i] + dist[i][j]

    # 所有城市都访问后（S = FULL），回到城市 0
    return min(
        dp[FULL][i] + dist[i][0]
        for i in range(1, n)
        if dp[FULL][i] < INF
    )


# 示例：4 个城市
dist = [
    [0, 10, 15, 20],
    [10,  0, 35, 25],
    [15, 35,  0, 30],
    [20, 25, 30,  0],
]
print(tsp(4, dist))  # 80（路径：0→1→3→2→0）
```

```cpp
int tsp(int n, vector<vector<int>>& dist) {
    const int INF = 1e9;
    int FULL = (1 << n) - 1;
    // dp[S][i]：已经过集合 S，当前在 i 的最短距离
    vector<vector<int>> dp(1 << n, vector<int>(n, INF));
    dp[1][0] = 0;  // 从城市 0 出发（1 = 1<<0）

    for (int S = 0; S < (1 << n); S++) {
        for (int i = 0; i < n; i++) {
            if (dp[S][i] == INF) continue;
            if (!(S >> i & 1)) continue;  // i 不在 S 中，无效
            for (int j = 0; j < n; j++) {
                if (S >> j & 1) continue;  // j 已访问
                int nS = S | (1 << j);
                dp[nS][j] = min(dp[nS][j], dp[S][i] + dist[i][j]);
            }
        }
    }
    // 返回：所有城市都访问完（状态 FULL），从某城市回到 0
    int ans = INF;
    for (int i = 1; i < n; i++)
        if (dp[FULL][i] != INF)
            ans = min(ans, dp[FULL][i] + dist[i][0]);
    return ans;
}
```

> **面试高频问题**：TSP 的状压 DP 复杂度是多少？$n = 25$ 时是否可行？（$2^{25} \times 25^2 \approx 2 \times 10^{10}$，不可行；一般 $n \leq 20$ 才能接受。）

---

## 29.6 DP 优化技术

### 29.6.1 单调队列优化（Deque DP）

**适用场景**：DP 中出现形如下面的转移，其中 $j$ 的范围是一个固定长度为 $k$ 的滑动窗口：

$$dp[i] = \min_{i-k \leq j < i} \bigl( dp[j] + \text{cost}(j, i) \bigr)$$

朴素枚举 $j$ 是 $O(n)$，总计 $O(n^2)$。用单调双端队列可以降到 $O(n)$。

**核心思路**：维护队列中的 $j$ 按 $dp[j]$（或某个单调量）单调递增，队头始终是当前可用的最优 $j$。

**经典题：LeetCode #239（滑动窗口最大值）是队列的基础；更典型的 DP 问题是"跳跃游戏 VI"（#1696）**：

$$dp[i] = \max_{i-k \leq j \leq i-1} dp[j] + \text{nums}[i]$$

```python
from collections import deque

def maxResult(nums: list[int], k: int) -> int:
    """
    跳跃游戏 VI（LeetCode #1696）

    dp[i] = 从 nums[0] 跳到 nums[i] 能获得的最大分数
    dp[i] = max(dp[i-k], ..., dp[i-1]) + nums[i]

    【单调队列优化】：维护 dp 值的单调递减队列
    队头始终是 [i-k, i-1] 范围内 dp 最大的下标

    时间复杂度：O(n)   空间复杂度：O(n + k)
    """
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    dq = deque([0])  # 存下标，队头是 dp 最大的那个

    for i in range(1, n):
        # 1. 移除队头中超出窗口范围 [i-k, i-1] 的下标
        while dq and dq[0] < i - k:
            dq.popleft()
        # 2. 用队头（最大 dp 值）更新 dp[i]
        dp[i] = dp[dq[0]] + nums[i]
        # 3. 维护队列单调性：移除队尾中 dp 值 ≤ dp[i] 的
        while dq and dp[dq[-1]] <= dp[i]:
            dq.pop()
        dq.append(i)

    return dp[n - 1]


print(maxResult([1,-1,-2,4,-7,3], 2))  # 7（路径 1 + 4 + 3 - 1 = 7）
print(maxResult([10,-5,-2,4,0,3], 3))  # 17
```

```cpp
int maxResult(vector<int>& nums, int k) {
    int n = nums.size();
    vector<int> dp(n, 0);
    dp[0] = nums[0];
    deque<int> dq = {0};  // 存下标，队头是 dp 最大的

    for (int i = 1; i < n; i++) {
        // 移除超出窗口的队头
        while (!dq.empty() && dq.front() < i - k) dq.pop_front();
        // 用队头更新
        dp[i] = dp[dq.front()] + nums[i];
        // 维护单调递减
        while (!dq.empty() && dp[dq.back()] <= dp[i]) dq.pop_back();
        dq.push_back(i);
    }
    return dp[n - 1];
}
```

---

### 29.6.2 斜率优化（Convex Hull Trick / CHT）

**适用场景**：DP 转移方程可以变形为：

$$dp[i] = \min_{j < i} \bigl( f(j) \cdot g(i) + h(j) \bigr)$$

其中 $f(j) \cdot g(i)$ 是两项之积，可以理解为"直线 $y = f(j) \cdot x + h(j)$ 在 $x = g(i)$ 处的值"。

**核心思路**：维护一组直线的"下凸包"，查询特定 $x$ 时的最小值，每次操作均摊 $O(1)$（若 $g(i)$ 单调），总计 $O(n)$。

**经典变形**：假设 DP 为：

$$dp[i] = \min_{j < i} \bigl( dp[j] + (s_i - s_j)^2 \bigr)$$

其中 $s_i$ 是前缀和。展开后：

$$dp[i] = dp[j] + s_i^2 - 2s_i s_j + s_j^2 = (-2s_j) \cdot s_i + (dp[j] + s_j^2) + s_i^2$$

含 $s_i^2$ 的部分对 $j$ 无关，可以单独处理。令直线斜率 $m = -2s_j$，截距 $b = dp[j] + s_j^2$，查询点 $x = s_i$，则：

$$dp[i] = \min_j (m_j \cdot x + b_j) + s_i^2$$

<div data-component="ConvexHullTrickViz"></div>

```python
from typing import NamedTuple

class Line(NamedTuple):
    slope: float  # 斜率 m
    intercept: float  # 截距 b

def eval_line(line: Line, x: float) -> float:
    return line.slope * x + line.intercept

def bad(l1: Line, l2: Line, l3: Line) -> bool:
    """
    判断直线 l2 是否永远不会是最优答案（查询 min 时）
    即 l2 与 l1 的交点 x 坐标 >= l2 与 l3 的交点 x 坐标
    """
    return ((l3.intercept - l1.intercept) * (l1.slope - l2.slope)
          <= (l2.intercept - l1.intercept) * (l1.slope - l3.slope))


def cht_example(n: int, s: list[int]) -> list[int]:
    """
    斜率优化（CHT）示例：dp[i] = min_{j<i}(dp[j] + (s[i] - s[j])²)

    单调 CHT：当 s[i]（查询点 x）单调递增时，队头指针只往右走

    时间复杂度：O(n)  空间复杂度：O(n)
    """
    from collections import deque
    dp = [float('inf')] * (n + 1)
    dp[0] = 0
    hull: deque[Line] = deque()   # 凸包（斜率单调递增）

    def add_line(j: int) -> None:
        """将 j 对应的直线加入凸包"""
        new_line = Line(slope=-2 * s[j], intercept=dp[j] + s[j] ** 2)
        while len(hull) >= 2 and bad(hull[-2], hull[-1], new_line):
            hull.pop()
        hull.append(new_line)

    add_line(0)   # j = 0 的直线

    for i in range(1, n + 1):
        # 当 s[i] 单调递增时，维护队头（移除不再是最小的直线）
        while len(hull) >= 2 and eval_line(hull[0], s[i]) >= eval_line(hull[1], s[i]):
            hull.popleft()
        dp[i] = eval_line(hull[0], s[i]) + s[i] ** 2
        add_line(i)

    return dp
```

```cpp
// 斜率优化（CHT）框架
struct Line {
    long long slope, intercept;
    long long eval(long long x) const { return slope * x + intercept; }
};

// 判断 l2 是否在 l1 和 l3 之间永远不是最优
bool bad(Line l1, Line l2, Line l3) {
    return (__int128)(l3.intercept - l1.intercept) * (l1.slope - l2.slope)
        <= (__int128)(l2.intercept - l1.intercept) * (l1.slope - l3.slope);
}

long long cht_min(int n, vector<long long>& s) {
    vector<Line> hull;
    vector<long long> dp(n + 1, LLONG_MAX);
    dp[0] = 0;

    auto addLine = [&](int j) {
        Line nl = {-2 * s[j], dp[j] + s[j] * s[j]};
        while (hull.size() >= 2 && bad(hull[hull.size()-2], hull.back(), nl))
            hull.pop_back();
        hull.push_back(nl);
    };

    int ptr = 0;
    addLine(0);

    for (int i = 1; i <= n; i++) {
        while (ptr + 1 < (int)hull.size()
               && hull[ptr].eval(s[i]) >= hull[ptr+1].eval(s[i]))
            ptr++;
        dp[i] = hull[ptr].eval(s[i]) + s[i] * s[i];
        addLine(i);
    }
    return dp[n];
}
```

> ⚠️ **重要限制**：
> - 若斜率不单调，不能用双端队列指针，需改用二分查找（$O(n \log n)$）；
> - 若斜率和查询点都不单调，需要用 **李超线段树（Li Chao Tree）**，复杂度 $O(n \log V)$。

---

### 29.6.3 Knuth 优化：区间 DP 的 $O(n^3) \to O(n^2)$

**条件**：区间 DP $dp[i][j] = \min_{i \leq k < j}(dp[i][k] + dp[k+1][j]) + w(i, j)$ 满足**四边形不等式（Monge condition）**：

$$w(a, c) + w(b, d) \leq w(a, d) + w(b, c), \quad \forall a \leq b \leq c \leq d$$

等价地，$w(i, j)$ 满足"区间越大代价越大"且"区域不重叠的代价之和 ≤ 区域有重叠时的代价"。

**结论**：设 $\text{opt}[i][j]$ 为 $dp[i][j]$ 的最优分割点，则满足四边形不等式时：

$$\text{opt}[i][j-1] \leq \text{opt}[i][j] \leq \text{opt}[i+1][j]$$

即最优分割点具有**单调性**。利用此性质，枚举 $k$ 的范围从 $[i, j-1]$ 缩减到 $[\text{opt}[i][j-1], \text{opt}[i+1][j]]$，总枚举量从 $O(n^3)$ 降至 $O(n^2)$。

```python
def knuth_optimized_interval_dp(n: int, w: callable) -> int:
    """
    Knuth 优化的区间 DP

    前提：w(i,j) 满足四边形不等式（Monge 条件）
    复杂度：O(n²)

    【正确性说明】：
    - opt[i][j-1] ≤ opt[i][j] ≤ opt[i+1][j] 保证了折半枚举总量是 O(n²)
    - 每次枚举 k 的范围 [opt[i][j-1], opt[i+1][j]] 的总和是 O(n²)
    """
    INF = float('inf')
    dp  = [[INF] * (n + 1) for _ in range(n + 1)]
    opt = [[0]   * (n + 1) for _ in range(n + 1)]  # 最优分割点

    for i in range(1, n + 1):
        dp[i][i] = 0
        opt[i][i] = i

    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            lo = opt[i][j - 1]        # Knuth 优化：缩小 k 的枚举范围
            hi = opt[i + 1][j] if i + 1 <= j else j

            for k in range(lo, hi + 1):
                cost = dp[i][k] + dp[k + 1][j] + w(i, j)
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    opt[i][j] = k

    return dp[1][n]
```

```cpp
// Knuth 优化区间 DP
// 要求 w(i,j) 满足四边形不等式
void knuthDP(int n, vector<vector<int>>& dp, vector<vector<int>>& opt,
             function<int(int,int)> w) {
    const int INF = 1e9;
    for (int i = 1; i <= n; i++) { dp[i][i] = 0; opt[i][i] = i; }

    for (int len = 2; len <= n; len++) {
        for (int i = 1; i <= n - len + 1; i++) {
            int j = i + len - 1;
            dp[i][j] = INF;
            int lo = opt[i][j-1];
            int hi = (i+1 <= j) ? opt[i+1][j] : j;
            for (int k = lo; k <= hi; k++) {
                int cost = dp[i][k] + dp[k+1][j] + w(i, j);
                if (cost < dp[i][j]) {
                    dp[i][j] = cost;
                    opt[i][j] = k;
                }
            }
        }
    }
}
```

---

## 29.7 综合对比与面试指南

### 29.7.1 各类高级 DP 总结

| DP 类型 | 状态维度 | 枚举顺序 | 典型题目 | 复杂度 |
|---|---|---|---|---|
| 区间 DP | $dp[l][r]$ | 先长度再左端点 | 矩阵链乘、气球 | $O(n^3)$ |
| 树形 DP | $dp[u][state]$ | 后序 DFS | #124、#337 | $O(n)$ |
| 换根 DP | $dp[u]$（两趟） | 先下后上，再上往下 | #834 | $O(n)$ |
| DAG DP | $dp[u]$ | 拓扑排序 | 最长路、#329 | $O(V+E)$ |
| 数位 DP | $dp[pos][...][tight]$ | 逐位从高到低 | 区间计数 | $O(n \log N)$ |
| 状压 DP | $dp[S][i]$ | 枚举状态 $S$（从小到大） | TSP、#847 | $O(2^n \cdot n^2)$ |
| 单调队列优化 | 线性 DP | 从左到右 | #1696 | $O(n)$ |
| 斜率优化 | 线性 DP | 从左到右 | 制造业、邮局 | $O(n)$ |
| Knuth 优化 | 区间 DP | 先长度 | 最优 BST | $O(n^2)$ |

### 29.7.2 调试技巧

1. **打印 DP 表格**：区间 DP 可以打印 $dp[l][r]$ 的二维矩阵，直观看出填充顺序是否正确；
2. **小样例手动验证**：矩阵链乘 $n=3$，石子合并 $n=4$，手算 DP 表；
3. **分离状态验证**：先确认初始化，再验证单步转移；
4. **数位 DP 用 brute force 对拍**：对 $n \leq 10000$ 先暴力枚举，和 DP 结果对比。

### 29.7.3 面试高频题单

| 题号 | 题目 | 类型 | 难点 |
|---|---|---|---|
| #312 | 戳气球 | 区间 DP | 开区间 + "最后一个"思维 |
| #664 | 奇怪打印机 | 区间 DP | 状态定义需要思考 |
| #124 | 二叉树最大路径和 | 树形 DP | 拐点 vs 单侧贡献 |
| #337 | 打家劫舍 III | 树形 DP | 0/1 状态传递 |
| #329 | 矩阵中最长递增路径 | DAG DP | 隐式 DAG + 记忆化 |
| #847 | 访问所有节点最短路 | 状压 DP | BFS + 状压结合 |
| #1696 | 跳跃游戏 VI | 单调队列 DP | 队列维护方向 |

### 29.7.4 练习题

1. **LeetCode #664**（奇怪打印机）：设计区间 DP 的状态，要小心"相同字符可以合并"的条件；
2. **LeetCode #375**（猜数字大小 II）：经典区间 DP，dp[i][j] = 在 [i,j] 中猜数字的最差代价最优；
3. **USACO 2011**（Fence Repair）：等价于石子合并，注意变种；
4. 实现矩阵链乘法的 **Knuth 优化版本**并验证结果与 $O(n^3)$ 版本一致；
5. **LeetCode #233**（数字 1 的个数）：数位 DP 的经典题，也可以用公式法。

---

> **💡 思考题**：
> 1. 气球 #312 中，为什么枚举"最后一个"被戳的气球能使子问题独立？如果枚举"第一个"被戳的会怎样？
> 2. TSP 的 $O(2^n n^2)$ 算法和暴力枚举 $O(n!)$ 相比，$n=20$ 时各自的时间大约是多少？
> 3. 斜率优化（CHT）在斜率不单调时，为什么不能用双端队列指针，而要用二分？

**参考资料**：
- CLRS 第4版 Chapter 14（最优 BST、矩阵链乘）
- MIT 6.046 Lecture 8–9（DP 进阶）
- cp-algorithms.com（CHT、Knuth 优化、数位 DP）
- 竞赛参考：[OI-wiki.org](https://oi-wiki.org/dp/)

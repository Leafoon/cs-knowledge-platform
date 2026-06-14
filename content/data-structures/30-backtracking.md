# Chapter 30: 回溯与剪枝（Backtracking）

## 章节导读

回溯法（Backtracking）是一种通过**系统性穷举**来寻找满足约束条件的所有解（或存在性）的算法范式。它本质上是一棵隐式的**决策树上的 DFS 遍历**——沿某条路径深入，发现走不通时退回上一步，换另一条路再走。

与暴力枚举不同，回溯的精华在于**剪枝**：当当前部分解已经不可能延伸为有效完整解时，立即截断该分支，避免无效计算。这使得回溯在最坏情况下仍是指数级，但在实际问题中往往表现出色。

| 范式 | 核心思路 | 最坏复杂度 | 典型题型 |
|---|---|---|---|
| 回溯 | 枚举所有（满足约束的）解路径 | $O(k^n)$ — $k$ 个选择 $n$ 层 | N皇后、全排列、子集、数独 |
| 动态规划 | 重叠子问题记忆化，求最优解 | 多项式（通常 $O(n^2)$ 或 $O(n^3)$） | LCS、背包、区间 DP |
| 贪心 | 每步局部最优，全局最优 | $O(n \log n)$ | 活动选择、Huffman |

本章的核心收获：

1. **理解回溯的隐式搜索树**：把抽象问题转化为"每层作一个决策"的树结构；
2. **熟练书写通用回溯模板**："选择 → 递归 → 撤销"三步心法；
3. **掌握四类剪枝技术**：可行性、最优性（Branch and Bound）、对称性、排序预处理；
4. **能辨别回溯 vs DP vs 贪心**，不再"见题上 DP"或"乱用贪心"；
5. **理解分支限界法**与 DFS 回溯的本质区别，及其在优化问题中的应用。

---

## 30.1 回溯法框架

### 30.1.1 隐式搜索树（State Space Search）

**什么是"隐式搜索树"？**

解空间搜索树（State Space Tree）是回溯算法的核心抽象工具。它是一棵**在运行时动态展开**而非预先存储的树：

- **根节点**：空路径（初始状态，还未作任何决策）；
- **每个内部节点**：对应一个"部分解"（partial solution），即已经做了若干步决策后的状态；
- **每条边**：从当前状态出发，"选择一个候选元素"这一决策动作；
- **叶节点**：要么是一个完整解（满足所有约束），要么是一个死胡同（已证明不可能成为合法解）。

**直觉示例：3 元素的子集枚举**

设元素集合为 $\{1, 2, 3\}$，我们逐层决策每个元素"选"（✓）或"不选"（✗）：

```
               []               ← 第0层：空子集
         /          \
      [✗1]          [✓1]        ← 第1层：不选1 or 选1
      /    \        /    \
  [✗2]  [✓2]  [✗2]   [✓2]     ← 第2层
   / \   / \   / \    / \
  ✗3 ✓3 ✗3 ✓3 ✗3 ✓3 ✗3 ✓3    ← 第3层：叶节点即完整子集
  {}  {3} {2} {2,3} {1} {1,3} {1,2} {1,2,3}
```

这棵树有 $2^3 = 8$ 个叶节点，恰好对应 $\{1,2,3\}$ 的所有子集。**关键在于这棵树的节点不需要提前构造，只在 DFS 递归时按需展开**，这就是"隐式"的含义。

**树的深度与分叉**

不同问题对应不同形状的树：

| 问题 | 树的深度 | 每层分叉数 | 叶节点数（无剪枝） |
|---|---|---|---|
| 子集枚举 $n$ 元素 | $n$ | 2（选/不选） | $2^n$ |
| 全排列 $n$ 元素 | $n$ | 最多 $n$，逐减 | $n!$ |
| N 皇后 | $n$（每行） | 最多 $n$（每列） | $n!$（无剪枝） |
| 组合 $C(n,k)$ | $k$（选 k 个） | 递减（避免重复） | $C(n,k)$ |

### 30.1.2 DFS 回溯模板："选择 → 递归 → 撤销"

回溯的通用骨架可以用以下伪代码概括：

```
procedure backtrack(path, choices):
    if path 满足终止条件:
        record(path)      // 找到一个解，记录之
        return

    for each choice in choices:
        if choice 满足约束条件:     // 可行性检查（剪枝入口）
            apply(choice, path)     // ① 做出选择
            backtrack(path + choice, updated_choices)  // ② 递归
            undo(choice, path)      // ③ 撤销选择（回溯）
```

**核心三步的工程实现**：

1. **选择（apply）**：将当前候选项加入路径，更新全局状态（如 `used[i] = True`、在棋盘上落子）；
2. **递归（recurse）**：进入下一层决策；
3. **撤销（undo）**：递归返回后，将状态恢复到进入此层之前（如 `used[i] = False`、从棋盘上移子）。

**撤销的必要性**：回溯本质上是在同一棵状态树上共享路径，必须保证"探索完一个子树后，状态干净地还原"，否则不同分支之间会互相污染。

**全局变量 vs 参数传递**：

两种风格都正确，但有细微差异：

```python
# 风格一：path 是全局列表（in-place 修改），效率更高
result = []
path = []

def backtrack(start):
    result.append(path[:])  # 拷贝当前路径（关键！不能直接 append path）
    for i in range(start, n):
        path.append(nums[i])    # ① 选择
        backtrack(i + 1)        # ② 递归
        path.pop()              # ③ 撤销

# 风格二：path 通过参数传递（新建列表），更安全但略慢
def backtrack(path, start):
    result.append(path)        # path 已是不可变副本，无需拷贝
    for i in range(start, n):
        backtrack(path + [nums[i]], i + 1)  # 创建新列表，自动"撤销"
```

工程上优先选**风格一**（in-place + 手动 append/pop），避免频繁创建临时列表，在 $n$ 大时性能差异明显。

```python
# 通用回溯模板（Python）- 以子集枚举为例说明

def backtrack_template(nums):
    """
    通用回溯模板示例：枚举 nums 的所有子集
    ─── 时间复杂度 O(2^n · n)（约 n 次拷贝 × 2^n 个子集）
    ─── 空间复杂度 O(n)（栈深度）
    """
    result = []
    path = []       # 当前路径（共享状态）

    def backtrack(start: int) -> None:
        # ── 终止条件：到达叶节点 ──────────────────────────
        result.append(path[:])  # 收集当前路径的副本

        # ── 枚举候选项 ────────────────────────────────────
        for i in range(start, len(nums)):
            # ① 做出选择
            path.append(nums[i])

            # ② 递归（传入 i+1 避免重复选同一位置）
            backtrack(i + 1)

            # ③ 撤销选择（恢复状态）
            path.pop()

    backtrack(0)
    return result

# 验证
print(backtrack_template([1, 2, 3]))
# 输出: [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]
```

```cpp
#include <bits/stdc++.h>
using namespace std;

// 通用回溯模板（C++）- 子集枚举示例
vector<vector<int>> result;
vector<int>         path;

void backtrack(vector<int>& nums, int start) {
    // ── 终止/记录 ────────────────────────────────────────
    result.push_back(path);  // 收集当前路径

    // ── 枚举候选项 ────────────────────────────────────────
    for (int i = start; i < (int)nums.size(); i++) {
        // ① 做出选择
        path.push_back(nums[i]);

        // ② 递归
        backtrack(nums, i + 1);

        // ③ 撤销
        path.pop_back();
    }
}

vector<vector<int>> subsets(vector<int>& nums) {
    result.clear(); path.clear();
    backtrack(nums, 0);
    return result;
}
// 调用: subsets({1,2,3}) → 8 个子集
```

### 30.1.3 回溯 vs DP vs 贪心的决策框架

这是 DSA 学习者最容易困惑的选择题。三者并非对立，而是有清晰的适用边界：

**决策流程**：

```
问题
├── 求所有解/计数所有解/判断解的存在性？
│   ├── 是 → 回溯（DFS 穷举，配合剪枝）
│   └── 否（求最优解）
│       ├── 是否存在「贪心选择性质」：局部最优 → 全局最优？
│       │   ├── 是 → 贪心（线性或对数时间）
│       │   └── 否
│       │       ├── 是否有「最优子结构」 + 「重叠子问题」？
│       │       │   ├── 是 → 动态规划（多项式时间）
│       │       │   └── 否 → 回溯 + 最优性剪枝（Branch and Bound）
```

**具体示例辨析**：

| 问题 | 选择 | 理由 |
|---|---|---|
| 找出所有满足条件的子序列 | 回溯 | 需要"所有解" |
| 0/1 背包最大价值 | DP | 有重叠子问题，局部选择不满足贪心 |
| 分数背包最大价值 | 贪心 | 每次选价值/重量比最大的，局部→全局 |
| N 皇后有多少种放法 | 回溯 | 计数所有合法布局 |
| 最长递增子序列长度 | DP 或贪心+二分 | 最优子结构存在 |
| 能否拼出目标金额（硬币）| 两者皆可 | 恰好凑成→DP（计数/判断）；贪心在某些情形不对 |
| 图的着色（判断k-colorable）| 回溯 | NP 完全，无多项式解法 |

<div data-component="BacktrackVsDPChoice"></div>

**关键判断规则**：
1. **"找所有解"** → 几乎必然是回溯；
2. **"求最优值"** → 先试贪心（证明贪心选择性质），否则试 DP（验证最优子结构），都不对则用 Branch and Bound；
3. **解空间是排列/组合/子集** → 回溯的天然领域；
4. **状态空间可以紧凑表达（如字符串、整数、矩形）** → 考虑 DP。

### 30.1.4 时间复杂度分析

回溯的时间复杂度难以精确刻画——它高度依赖**剪枝的有效性**。以下是几个经典结论：

**上界（最坏情况，无剪枝）**：

- **子集枚举**：搜索树有 $2^n$ 个叶节点，总节点数为 $2^{n+1} - 1$，每节点 $O(1)$ 工作 → $O(2^n)$（不含拷贝；含拷贝则 $O(n \cdot 2^n)$）；
- **全排列**：搜索树第 0 层 1 个节点，第 1 层 $n$ 个，第 2 层 $n(n-1)$ 个……叶节点 $n!$ 个 → $O(n \cdot n!)$（含输出）；
- **N 皇后**：理论上 $O(n!)$，但实际有效分支远小于 $n!$，有精确解 $O(n! / 2^n)$ 量级。

**剪枝后的实际效果**：

| 问题($n$) | 无剪枝节点数 | 有效剪枝节点数 | 加速比 |
|---|---|---|---|
| N皇后 ($n=8$) | $8! = 40320$ | $~2000$ | ~20× |
| 数独（空格 $k$ 个） | $9^k$ | 通常 $<10000$ | 指数级 |
| 组合总和（经排序剪枝）| $O(2^n)$ | 取决于目标值 | 2-10× |

**分摊分析思路**：对于回溯问题，分析"每个候选项被尝试几次"——若每个候选仅在固定深度层被选一次，总工作量是有界的。

---

## 30.2 经典回溯问题

### 30.2.1 N 皇后问题（N-Queens，LeetCode #51 / #52）

#### 问题描述

在 $n \times n$ 棋盘上放置 $n$ 个皇后，使得任意两个皇后之间**不同行、不同列、不在同一对角线**。求所有合法方案（#51）或方案总数（#52）。

#### 搜索树框架

我们**逐行**放置皇后（第 0 行到第 $n-1$ 行），每行只需决定把皇后放在**哪一列**，这样"不同行"天然满足，只需检查"不同列"和"不同对角线"。

- **搜索树深度**：$n$（每一层对应一行）；
- **每层分叉**：最多 $n$ 列，但受约束后远小于 $n$；
- **状态**：`cols` 集合（已使用的列）+ `d1` 集合（主对角线 `row-col` 值的集合）+ `d2` 集合（副对角线 `row+col` 值的集合）。

#### 对角线冲突的数学本质

- **主对角线**（从左上到右下）：同一对角线上的格子满足 $row - col = \text{常数}$；
- **副对角线**（从右上到左下）：同一对角线上的格子满足 $row + col = \text{常数}$。

因此，只需用两个集合分别记录已使用的 $row - col$ 和 $row + col$ 值，$O(1)$ 完成冲突检测。

#### 位运算加速版（#52 最优解）

对于求方案总数（#52），可以用**三个整数的位掩码**代替集合，将冲突检测压缩到 $O(1)$ 位运算：

- `cols`：已占用列的掩码；
- `d1`：已占用的主对角线（向右传播，每层右移一位）；
- `d2`：已占用的副对角线（向左传播，每层左移一位）。

每层可用列为：`available = ((1 << n) - 1) & ~(cols | d1 | d2)`，用 `lowbit = available & (-available)` 依次取出每个可用位。

```python
from typing import List

class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        """
        N 皇后 —— 逐行放置，位掩码剪枝
        时间复杂度：O(n!)  空间复杂度：O(n)（栈 + 当前路径）
        """
        result: List[List[str]] = []
        queens = [-1] * n          # queens[row] = col，记录每行皇后所在列
        cols = set()               # 已占用的列
        d1   = set()               # 占用的 row - col 值（主对角线）
        d2   = set()               # 占用的 row + col 值（副对角线）

        def backtrack(row: int) -> None:
            # ── 叶节点：所有行都放好了 ─────────────────────────────────
            if row == n:
                board = []
                for r in range(n):
                    board.append('.' * queens[r] + 'Q' + '.' * (n - queens[r] - 1))
                result.append(board)
                return

            # ── 枚举当前行的每一列 ────────────────────────────────────
            for col in range(n):
                # 冲突检查：列、主对角线、副对角线
                if col in cols or (row - col) in d1 or (row + col) in d2:
                    continue  # 剪枝：跳过冲突列

                # ① 做出选择
                queens[row] = col
                cols.add(col)
                d1.add(row - col)
                d2.add(row + col)

                # ② 递归到下一行
                backtrack(row + 1)

                # ③ 撤销选择
                queens[row] = -1
                cols.discard(col)
                d1.discard(row - col)
                d2.discard(row + col)

        backtrack(0)
        return result

    def totalNQueens(self, n: int) -> int:
        """
        N 皇后计数（#52）—— 位掩码极速版
        用三个整型掩码取代 set，减少内存操作，实际快约 5-10×
        """
        full = (1 << n) - 1   # n 位全为1，表示 n 列都可用
        self.count = 0

        def bt(cols: int, d1: int, d2: int) -> None:
            # cols: 已占列掩码; d1: 主对角掩码; d2: 副对角掩码
            if cols == full:
                self.count += 1
                return
            # 当前行所有可用列（NOT 已占用的）
            available = full & ~(cols | d1 | d2)
            while available:
                # 取最低可用位（lowest set bit）
                bit = available & (-available)
                available -= bit
                # 下一行时，对角线掩码分别右移/左移1位传播
                bt(cols | bit, (d1 | bit) >> 1, (d2 | bit) << 1)

        bt(0, 0, 0)
        return self.count


# 测试
sol = Solution()
boards = sol.solveNQueens(4)
for b in boards:
    print('\n'.join(b))
    print('---')
# n=4 的解：
# .Q..      ..Q.
# ...Q      Q...
# Q...      ...Q
# ..Q.      .Q..

print(sol.totalNQueens(8))  # 92
```

```cpp
#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    /* ── N 皇后（#51）── 逐行放置 + 集合冲突检测 ──────────────── */
    vector<vector<string>> solveNQueens(int n) {
        vector<vector<string>> result;
        vector<int> queens(n, -1);   // queens[row] = col
        set<int>    usedCols, d1, d2;

        function<void(int)> bt = [&](int row) {
            if (row == n) {
                // 构造棋盘字符串
                vector<string> board;
                for (int r = 0; r < n; r++) {
                    string row_str(n, '.');
                    row_str[queens[r]] = 'Q';
                    board.push_back(row_str);
                }
                result.push_back(board);
                return;
            }
            for (int col = 0; col < n; col++) {
                if (usedCols.count(col)        ||
                    d1.count(row - col)        ||
                    d2.count(row + col)) continue;  // 剪枝

                // ① 选择
                queens[row] = col;
                usedCols.insert(col);
                d1.insert(row - col);
                d2.insert(row + col);

                // ② 递归
                bt(row + 1);

                // ③ 撤销
                queens[row] = -1;
                usedCols.erase(col);
                d1.erase(row - col);
                d2.erase(row + col);
            }
        };

        bt(0);
        return result;
    }

    /* ── N 皇后计数（#52）── 位掩码极速版 ─────────────────────── */
    int totalNQueens(int n) {
        int full = (1 << n) - 1;
        int cnt = 0;

        function<void(int, int, int)> bt = [&](int cols, int d1, int d2) {
            if (cols == full) { cnt++; return; }
            int avail = full & ~(cols | d1 | d2);
            while (avail) {
                int bit = avail & (-avail);   // lowest set bit
                avail  -= bit;
                bt(cols | bit, (d1 | bit) << 1, (d2 | bit) >> 1);
                // 注意：C++ 版本对角线传播方向与 Python 版本对称
            }
        };

        bt(0, 0, 0);
        return cnt;
    }
};

int main() {
    Solution sol;
    cout << sol.totalNQueens(8) << endl;  // 92
    auto boards = sol.solveNQueens(4);
    for (auto& b : boards) {
        for (auto& row : b) cout << row << '\n';
        cout << "---\n";
    }
}
```

<div data-component="NQueensBacktrackTree"></div>

**复杂度分析**：
- 时间：$O(n!)$（最坏，无剪枝）；但实际上第 $r$ 行的有效分支数随 $r$ 增大而急速减少，真实运行远快于 $n!$；
- 空间：$O(n)$（递归栈深 $n$ + `queens` 数组 $n$）。
- 位掩码版可比集合版快 3-5 倍（$n=15$ 以上差异明显）。

### 30.2.2 数独求解（Sudoku Solver，LeetCode #37）

#### 问题描述

给定一个 $9 \times 9$ 的数独谜题（空格用 `'.'` 表示），用 1-9 填满所有空格，使得：
- 每行包含 1-9 各一次；
- 每列包含 1-9 各一次；
- 每个 $3 \times 3$ 子方格包含 1-9 各一次。

#### 算法设计

**状态压缩**：用三个数组的位掩码记录已使用的数字：
- `row_used[r]`：第 $r$ 行已用数字的掩码（bit $k$ = 数字 $k+1$ 已用）；
- `col_used[c]`：第 $c$ 列的掩码；
- `box_used[b]`：第 $b$ 个 $3 \times 3$ 方格（编号 $b = (r//3) \times 3 + c//3$）的掩码。

枚举策略：
- **最简单的枚举**：按行列顺序逐格填空，每格尝试 1-9（已被约束掩码排除的跳过）；
- **改进版（最少候选值优先，MRV）**：每次找候选数字最少的空格先填——这能大幅减少搜索树规模，是 Solver 竞赛常见优化。

> 注：本题目保证有唯一解，所以找到第一个解后直接返回即可。

```python
class Solution:
    def solveSudoku(self, board: list[list[str]]) -> None:
        """
        数独求解：原地修改 board
        ── 策略：从左到右、从上到下枚举空格，位掩码快速过滤候选数
        ── 时间复杂度：最坏 O(9^(空格数))，但实际因约束极少能超 10^5 操作
        ── 空间复杂度：O(81)（栈深最多 81 层，对应 81 个空格）
        """
        # ── 初始化位掩码（用 9 位二进制表示 1-9 的使用情况）────────────
        row_used = [0] * 9   # row_used[r]: 第r行已用数字掩码（bit k = 数字 k+1）
        col_used = [0] * 9
        box_used = [0] * 9

        def set_bit(r, c, d, val):  # d: 数字 1-9, val: True=标记已用, False=撤销
            mask = 1 << (d - 1)
            b = (r // 3) * 3 + (c // 3)
            if val:
                row_used[r] |= mask
                col_used[c] |= mask
                box_used[b] |= mask
            else:
                row_used[r] &= ~mask
                col_used[c] &= ~mask
                box_used[b] &= ~mask

        def get_candidates(r, c) -> int:
            """返回 (r,c) 处合法候选数字的位掩码（bit k 置位 = 数字 k+1 合法）"""
            b = (r // 3) * 3 + (c // 3)
            used = row_used[r] | col_used[c] | box_used[b]
            return ((1 << 9) - 1) & ~used  # 9位全1 去掉已用位 = 候选位

        # ── 扫描初始局面，填入已知数字的掩码 ─────────────────────────
        empties = []  # 空格位置列表
        for r in range(9):
            for c in range(9):
                if board[r][c] == '.':
                    empties.append((r, c))
                else:
                    set_bit(r, c, int(board[r][c]), True)

        def backtrack(idx: int) -> bool:
            if idx == len(empties):
                return True  # 所有空格填完，找到解！

            r, c = empties[idx]
            candidates = get_candidates(r, c)

            while candidates:
                # 取最低候选位（任意选一个）
                bit = candidates & (-candidates)
                d = bit.bit_length()           # 对应数字 d（1-indexed）
                candidates -= bit

                # ① 填入数字 d
                board[r][c] = str(d)
                set_bit(r, c, d, True)

                # ② 递归填充下一个空格
                if backtrack(idx + 1):
                    return True               # 找到解，立即返回（唯一解题意）

                # ③ 撤销
                board[r][c] = '.'
                set_bit(r, c, d, False)

            return False  # 所有候选都失败 → 回溯

        backtrack(0)


# 测试
board = [
    ["5","3",".",".","7",".",".",".","."],
    ["6",".",".","1","9","5",".",".","."],
    [".","9","8",".",".",".",".","6","."],
    ["8",".",".",".","6",".",".",".","3"],
    ["4",".",".","8",".","3",".",".","1"],
    ["7",".",".",".","2",".",".",".","6"],
    [".","6",".",".",".",".","2","8","."],
    [".",".",".","4","1","9",".",".","5"],
    [".",".",".",".","8",".",".","7","9"]
]
Solution().solveSudoku(board)
for row in board:
    print(row)
```

```cpp
#include <bits/stdc++.h>
using namespace std;

class Solution {
    int rowUsed[9] = {}, colUsed[9] = {}, boxUsed[9] = {};

    void setBit(int r, int c, int d, bool on) {
        int mask = 1 << (d - 1);
        int b = (r / 3) * 3 + (c / 3);
        if (on) { rowUsed[r] |= mask; colUsed[c] |= mask; boxUsed[b] |= mask; }
        else    { rowUsed[r] &= ~mask; colUsed[c] &= ~mask; boxUsed[b] &= ~mask; }
    }

    int getCandidates(int r, int c) {
        int b = (r / 3) * 3 + (c / 3);
        return ((1 << 9) - 1) & ~(rowUsed[r] | colUsed[c] | boxUsed[b]);
    }

    bool backtrack(vector<vector<char>>& board, vector<pair<int,int>>& empties, int idx) {
        if (idx == (int)empties.size()) return true;
        auto [r, c] = empties[idx];
        int cands = getCandidates(r, c);
        while (cands) {
            int bit = cands & (-cands);
            int d = __builtin_ctz(bit) + 1;   // 数字 1-9
            cands -= bit;

            board[r][c] = '0' + d;
            setBit(r, c, d, true);

            if (backtrack(board, empties, idx + 1)) return true;

            board[r][c] = '.';
            setBit(r, c, d, false);
        }
        return false;
    }

public:
    void solveSudoku(vector<vector<char>>& board) {
        vector<pair<int,int>> empties;
        for (int r = 0; r < 9; r++)
            for (int c = 0; c < 9; c++) {
                if (board[r][c] != '.') {
                    setBit(r, c, board[r][c] - '0', true);
                } else {
                    empties.push_back({r, c});
                }
            }
        backtrack(board, empties, 0);
    }
};
```

<div data-component="SudokuConstraintProp"></div>

**关键工程细节**：
- **前向检查（Forward Checking）**：若某空格候选数为 0（即 `getCandidates(r,c) == 0`），可以提前返回 `false` 不必等到递归到该格——这是"可行性剪枝"的具体应用；
- **MRV（最少剩余值）启发式**：每次从候选数最少的空格开始填，而非顺序遍历。这是 Peter Norvig 数独求解器的核心加速手段，可让约搜索空间减少 100-1000 倍。

### 30.2.3 子集枚举（Subset Generation，LeetCode #78）

#### 两种等价视角

**视角一：逐元素"选/不选"（DFS 树）**

每一层对应一个元素，分两条分支：选（加入 path）或不选（跳过）。树的深度 $n$，恰好 $2^n$ 个叶节点。

**视角二：固定起点 `start` 的"递增枚举"**

通用回溯模板的典型实现：`backtrack(start)` 从 `start` 开始枚举，每次选一个元素加入后，`start = i+1`（确保不选已选元素之前的元素，避免重复子集）。

两种视角等价，但"视角二"更容易扩展到组合总和（允许重复），因此推荐掌握"视角二"。

```python
from typing import List

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        """
        子集枚举（无重复元素），LeetCode #78
        ── 每个子集恰好输出一次，时间 O(n · 2^n)（n 为每次拷贝），空间 O(n)
        """
        result: List[List[int]] = []
        path: List[int] = []

        def backtrack(start: int) -> None:
            result.append(path[:])   # 每进入一个节点都收集（包括空子集和 nums 本身）

            for i in range(start, len(nums)):
                path.append(nums[i])    # ① 选择 nums[i]
                backtrack(i + 1)        # ② 递归（i+1 保证不重选）
                path.pop()              # ③ 撤销

        backtrack(0)
        return result

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        """
        含重复元素的子集枚举（LeetCode #90）
        ── 关键：先排序，同层中跳过相邻相同元素
        """
        nums.sort()    # ← 排序是去重的前提
        result: List[List[int]] = []
        path: List[int] = []

        def backtrack(start: int) -> None:
            result.append(path[:])
            for i in range(start, len(nums)):
                # 去重：同一层中，如果当前元素与前一个相同且前一个未被使用，跳过
                if i > start and nums[i] == nums[i - 1]:
                    continue
                path.append(nums[i])
                backtrack(i + 1)
                path.pop()

        backtrack(0)
        return result


# 测试
sol = Solution()
print(sol.subsets([1, 2, 3]))
# [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]

print(sol.subsetsWithDup([1, 2, 2]))
# [[], [1], [1, 2], [1, 2, 2], [2], [2, 2]]  ← 只有 6 个而非 8 个
```

```cpp
#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    /* ── #78 子集（无重复）────────────────────────────────────── */
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> result;
        vector<int>         path;

        function<void(int)> bt = [&](int start) {
            result.push_back(path);          // 每个节点都是合法子集
            for (int i = start; i < (int)nums.size(); i++) {
                path.push_back(nums[i]);
                bt(i + 1);
                path.pop_back();
            }
        };

        bt(0);
        return result;
    }

    /* ── #90 含重复元素子集（先排序，同层去重）──────────────────── */
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        vector<vector<int>> result;
        vector<int>         path;

        function<void(int)> bt = [&](int start) {
            result.push_back(path);
            for (int i = start; i < (int)nums.size(); i++) {
                // 同层跳过相邻重复元素
                if (i > start && nums[i] == nums[i - 1]) continue;
                path.push_back(nums[i]);
                bt(i + 1);
                path.pop_back();
            }
        };

        bt(0);
        return result;
    }
};
```

<div data-component="SubsetEnumerationTree"></div>

**子集枚举的"选/不选"等价写法**（适合作为脑图参考）：

```python
def subsets_choose(nums):
    """逐元素决策：选 or 不选"""
    result, path = [], []

    def dfs(i):
        if i == len(nums):
            result.append(path[:])
            return
        # 路径一：选 nums[i]
        path.append(nums[i])
        dfs(i + 1)
        path.pop()
        # 路径二：不选 nums[i]
        dfs(i + 1)

    dfs(0)
    return result
```

### 30.2.4 全排列（Permutation，LeetCode #46 / #47）

#### 问题与搜索树结构

全排列问题：把 $n$ 个元素的所有顺序排列全部输出。树的深度为 $n$，第 $k$ 层分叉数为 $n-k$（已选了 $k$ 个，剩余 $n-k$ 个候选），叶节点数为 $n!$。

**两种实现风格**：

1. **`used[]` 数组标记**：全局 `used[i]` 记录第 $i$ 个元素是否已在当前路径中，$O(n)$ 检查。
2. **交换法（in-place）**：不使用额外 `used[]`，通过在当前位置与后续位置交换实现 $O(1)$ 放置——但去重时需额外处理。

#### 含重复元素的去重（#47）

关键技巧：**先对输入排序，同层枚举时跳过相邻值相同且前一个 `used[i-1]==False` 的元素**。

这个条件的含义：若 `nums[i] == nums[i-1]` 且 `nums[i-1]` 的当前层未被使用（即上一层已经通过 `nums[i-1]` 生成了一棵完整的子树），那么再用 `nums[i]`（同值）生成的子树会与之完全重复 → 跳过。

```python
from typing import List

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        """
        全排列（无重复元素），LeetCode #46
        ── used[] 标记已选元素
        ── 时间 O(n · n!)，空间 O(n)
        """
        n = len(nums)
        result: List[List[int]] = []
        path: List[int] = []
        used = [False] * n

        def backtrack() -> None:
            if len(path) == n:
                result.append(path[:])
                return

            for i in range(n):
                if used[i]:
                    continue        # 已选过，跳过

                # ① 选择
                used[i] = True
                path.append(nums[i])

                # ② 递归
                backtrack()

                # ③ 撤销
                path.pop()
                used[i] = False

        backtrack()
        return result

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        """
        全排列（含重复元素），LeetCode #47
        ── 先排序 + 同层跳过相邻重复（used[i-1]==False 时语义：同层已枚举过）
        """
        nums.sort()    # ← 必须先排序！
        n = len(nums)
        result: List[List[int]] = []
        path: List[int] = []
        used = [False] * n

        def backtrack() -> None:
            if len(path) == n:
                result.append(path[:])
                return

            for i in range(n):
                if used[i]:
                    continue
                # 去重核心：nums[i] == nums[i-1] 且 used[i-1] == False
                # 含义：同一层，相同值的元素 nums[i-1] 已经被作为这一层的起点
                # 处理完整棵子树了 → 再处理 nums[i] 会产生完全相同的子树
                if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                    continue

                used[i] = True
                path.append(nums[i])
                backtrack()
                path.pop()
                used[i] = False

        backtrack()
        return result


# 测试
sol = Solution()
print(sol.permute([1, 2, 3]))
# [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

print(sol.permuteUnique([1, 1, 2]))
# [[1,1,2],[1,2,1],[2,1,1]]  ← 只有 3 种而非 3! = 6 种
```

```cpp
#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    /* ── #46 全排列（无重复）────────────────────────────────────── */
    vector<vector<int>> permute(vector<int>& nums) {
        int n = nums.size();
        vector<vector<int>> result;
        vector<int>         path;
        vector<bool>        used(n, false);

        function<void()> bt = [&]() {
            if ((int)path.size() == n) {
                result.push_back(path);
                return;
            }
            for (int i = 0; i < n; i++) {
                if (used[i]) continue;
                used[i] = true;
                path.push_back(nums[i]);
                bt();
                path.pop_back();
                used[i] = false;
            }
        };

        bt();
        return result;
    }

    /* ── #47 全排列（含重复，去重）──────────────────────────────── */
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int n = nums.size();
        vector<vector<int>> result;
        vector<int>         path;
        vector<bool>        used(n, false);

        function<void()> bt = [&]() {
            if ((int)path.size() == n) {
                result.push_back(path);
                return;
            }
            for (int i = 0; i < n; i++) {
                if (used[i]) continue;
                // 关键去重条件：同值且前一个未被使用（即在同一层已被处理过）
                if (i > 0 && nums[i] == nums[i-1] && !used[i-1]) continue;
                used[i] = true;
                path.push_back(nums[i]);
                bt();
                path.pop_back();
                used[i] = false;
            }
        };

        bt();
        return result;
    }
};
```

<div data-component="PermutationBacktrack"></div>

**交换法（更快，但不稳定序）**：

```python
def permute_swap(nums: list) -> list:
    """
    交换法：nums[start] 与 nums[i] 交换，并行推进 start
    优势：O(1) 额外空间（不需要 used[]）
    劣势：输出顺序与排序无关，且对含重复元素的去重需要额外 set 保证同层元素唯一
    """
    result = []

    def dfs(start):
        if start == len(nums):
            result.append(nums[:])
            return
        seen = set()              # 同层去重（处理含重复情形）
        for i in range(start, len(nums)):
            if nums[i] in seen:
                continue
            seen.add(nums[i])
            nums[start], nums[i] = nums[i], nums[start]   # ① 交换
            dfs(start + 1)                                 # ② 递归
            nums[start], nums[i] = nums[i], nums[start]   # ③ 还原

    dfs(0)
    return result
```

### 30.2.5 组合总和（Combination Sum，LeetCode #39 / #40）

#### 问题描述

- **#39（Combination Sum）**：候选数组 `candidates`（无重复），每个元素**可以无限次**使用，找所有和为 `target` 的组合；
- **#40（Combination Sum II）**：候选数组有重复，每个元素**最多使用一次**，找所有和为 `target` 的不重复组合。

#### 剪枝分析

两道题的核心剪枝：**先对 `candidates` 排序**，则当 `canidates[i] > remaining`时，后续更大的元素也一定大于 `remaining`，可以直接 `break`（不是 `continue`！）终止当前层的枚举——这将搜索树的节点数大幅削减。

```python
from typing import List

class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        """
        组合总和（#39）：每个元素可重复使用
        ── 重复使用：递归时 start = i（不是 i+1）
        ── 排序 + 剩余量 > 0 剪枝
        ── 时间：O(n^(T/M)) 其中 T=target，M=min(candidates)；空间 O(T/M)（栈）
        """
        candidates.sort()   # ← 排序，用于剪枝（发现当前元素太大可直接break）
        result: List[List[int]] = []
        path: List[int] = []

        def backtrack(start: int, remaining: int) -> None:
            if remaining == 0:
                result.append(path[:])   # 找到一个合法组合
                return

            for i in range(start, len(candidates)):
                if candidates[i] > remaining:
                    break    # ← 排序后，后续元素更大，直接终止此层
                path.append(candidates[i])
                backtrack(i, remaining - candidates[i])  # i 不是 i+1（允许重复使用）
                path.pop()

        backtrack(0, target)
        return result

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        """
        组合总和 II（#40）：每个元素最多使用一次，结果不重复
        ── 与子集去重（#90）完全一样的去重手法：i > start and nums[i] == nums[i-1] → skip
        """
        candidates.sort()
        result: List[List[int]] = []
        path: List[int] = []

        def backtrack(start: int, remaining: int) -> None:
            if remaining == 0:
                result.append(path[:])
                return
            for i in range(start, len(candidates)):
                if candidates[i] > remaining:
                    break
                # 同层去重：跳过与前一个相同的元素
                if i > start and candidates[i] == candidates[i - 1]:
                    continue
                path.append(candidates[i])
                backtrack(i + 1, remaining - candidates[i])  # i+1：每个元素最多用一次
                path.pop()

        backtrack(0, target)
        return result


# 测试
sol = Solution()
print(sol.combinationSum([2, 3, 6, 7], 7))
# [[2, 2, 3], [7]]

print(sol.combinationSum2([10, 1, 2, 7, 6, 1, 5], 8))
# [[1,1,6],[1,2,5],[1,7],[2,6]]
```

```cpp
#include <bits/stdc++.h>
using namespace std;

class Solution {
public:
    /* ── #39 组合总和（每元素可重复使用）────────────────────────── */
    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        vector<vector<int>> result;
        vector<int>         path;

        function<void(int, int)> bt = [&](int start, int rem) {
            if (rem == 0) { result.push_back(path); return; }
            for (int i = start; i < (int)candidates.size(); i++) {
                if (candidates[i] > rem) break;   // 排序后剪枝
                path.push_back(candidates[i]);
                bt(i, rem - candidates[i]);       // i 非 i+1（允许重复）
                path.pop_back();
            }
        };

        bt(0, target);
        return result;
    }

    /* ── #40 组合总和 II（有重复，每元素最多一次）──────────────── */
    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        sort(candidates.begin(), candidates.end());
        vector<vector<int>> result;
        vector<int>         path;

        function<void(int, int)> bt = [&](int start, int rem) {
            if (rem == 0) { result.push_back(path); return; }
            for (int i = start; i < (int)candidates.size(); i++) {
                if (candidates[i] > rem) break;
                if (i > start && candidates[i] == candidates[i-1]) continue;  // 去重
                path.push_back(candidates[i]);
                bt(i + 1, rem - candidates[i]);
                path.pop_back();
            }
        };

        bt(0, target);
        return result;
    }
};
```

**关键去重对比表**（#39 / #40 / #46 / #47 / #78 / #90）：

| 题目 | 重复元素？| 重复使用？| 去重手段 |
|---|---|---|---|
| #78 子集 | 否 | 否 | 无需去重 |
| #90 子集 II | 是 | 否 | 排序 + `i > start and nums[i] == nums[i-1]` |
| #39 组合总和 | 否 | **是** | 无需去重，`start = i`（非 `i+1`） |
| #40 组合总和 II | 是 | 否 | 排序 + `i > start and ...` |
| #46 全排列 | 否 | 否 | `used[]` 标记 |
| #47 全排列 II | 是 | 否 | 排序 + `i>0 and nums[i]==nums[i-1] and not used[i-1]` |

### 30.2.6 单词搜索（Word Search，LeetCode #79）

#### 问题描述

给定一个 $m \times n$ 的字母网格 `board` 和一个单词 `word`，判断 `word` 是否可以由网格中上下左右相邻的字母组成（每个格子只能使用一次）。

#### 算法设计

这是一道典型的**二维网格 DFS + 回溯**题：

1. 枚举所有起始位置 $(r, c)$（与 `word[0]` 匹配的格子）；
2. 从 $(r, c)$ 出发，DFS 探索四个方向，逐字符匹配；
3. 进入某格前，将该格标记为已访问（原地修改，如改为 `'#'`）；
4. 回溯时，将该格还原。

**剪枝**：若当前格字符与 `word[k]` 不匹配，立即返回 `False`（可行性剪枝）。

```python
class Solution:
    def exist(self, board: list[list[str]], word: str) -> bool:
        """
        单词搜索（LeetCode #79）
        ── 原地标记 visited（改为 '#'），避免额外 visited 数组
        ── 时间：O(m·n · 4^L)，L = len(word)；空间：O(L)（栈）
        """
        m, n, L = len(board), len(board[0]), len(word)
        DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        def dfs(r: int, c: int, k: int) -> bool:
            """从 board[r][c] 开始，尝试匹配 word[k:]"""
            if k == L:
                return True          # 所有字符匹配完毕

            if r < 0 or r >= m or c < 0 or c >= n:
                return False         # 越界
            if board[r][c] != word[k]:
                return False         # 字符不匹配（剪枝）

            # ① 标记已访问（原地修改）
            tmp = board[r][c]
            board[r][c] = '#'

            # ② 递归四个方向
            for dr, dc in DIRS:
                if dfs(r + dr, c + dc, k + 1):
                    board[r][c] = tmp  # 恢复后也要返回
                    return True

            # ③ 撤销标记
            board[r][c] = tmp
            return False

        for r in range(m):
            for c in range(n):
                if dfs(r, c, 0):
                    return True

        return False


# 测试
board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
print(Solution().exist(board, "ABCCED"))   # True
print(Solution().exist(board, "SEE"))      # True
print(Solution().exist(board, "ABCB"))     # False（B 不能重复使用）
```

```cpp
#include <bits/stdc++.h>
using namespace std;

class Solution {
    int m, n;
    vector<pair<int,int>> DIRS = {{0,1},{0,-1},{1,0},{-1,0}};

    bool dfs(vector<vector<char>>& board, const string& word, int r, int c, int k) {
        if (k == (int)word.size()) return true;
        if (r < 0 || r >= m || c < 0 || c >= n) return false;
        if (board[r][c] != word[k]) return false;

        char tmp = board[r][c];
        board[r][c] = '#';           // ① 标记

        for (auto [dr, dc] : DIRS) {
            if (dfs(board, word, r+dr, c+dc, k+1)) {
                board[r][c] = tmp;   // 恢复
                return true;
            }
        }

        board[r][c] = tmp;           // ③ 撤销
        return false;
    }

public:
    bool exist(vector<vector<char>>& board, string word) {
        m = board.size(); n = board[0].size();
        for (int r = 0; r < m; r++)
            for (int c = 0; c < n; c++)
                if (dfs(board, word, r, c, 0)) return true;
        return false;
    }
};
```

**优化思路**：
- **前缀 + 后缀过滤**：统计 `word` 首字符在 `board` 中的出现次数，如果末字符更少，可以反转 `word`（从频率低的一端开始搜索），平均减少一半搜索量；
- **Trie + 多词搜索**：若同时查找多个单词（#212 Word Search II），将所有单词插入 Trie，一次 DFS 同时匹配所有，比逐词搜索高效 $O(L)$ 倍。

---

## 30.3 剪枝技术

剪枝是回溯的灵魂——正确的剪枝可以将指数级的搜索压缩到多项式级别。以下介绍四种系统性剪枝思路：

### 30.3.1 可行性剪枝（Feasibility Pruning）

**核心思想**：在树的某个节点处，若能够**证明当前分支的路径不可能延伸为任何合法完整解**，则立即终止该分支，不再递归。

可行性剪枝的判断条件必须满足：
1. **充分性**：满足剪枝条件的分支确实不含解（不能误剪）；
2. **计算高效**：判断条件本身的代价不能超过搜索节省的代价（通常要求 $O(1)$ 或 $O(k)$）。

**经典例子**：

**例1：N 皇后的冲突检测**

```
if col in cols or (row-col) in d1 or (row+col) in d2:
    continue  ← 当前列/对角线已有皇后，此列分支不可能成功，剪掉
```

**例2：组合总和的排序剪枝**

```python
candidates.sort()
for i in range(start, len(candidates)):
    if candidates[i] > remaining:
        break   ← 当前及后续所有元素均大于剩余目标，整棵右子树都剪掉
```

**例3：数独的前向检查（Forward Checking）**

在选择下一个空格之前，检查其候选数字集合：

```python
def has_candidates(r, c):
    return get_candidates(r, c) != 0  # 还有候选 → 继续；候选为0 → 提前剪枝
```

若发现某空格已无候选数（`candidates == 0`），当前分支必然无解，立即回溯。这比不检查直接进入该格再发现问题，节省了递归层数。

**例4：分割回文串（LeetCode #131）**

判断 `s[start:end+1]` 是否是回文串，是才递归，不是直接跳过——这是对每个候选划分的可行性检测。

```python
def partition(s: str) -> list[list[str]]:
    result, path = [], []

    # 预计算回文子串表（O(n^2) 预处理，O(1) 查询）
    n = len(s)
    is_pal = [[False]*n for _ in range(n)]
    for i in range(n-1, -1, -1):
        for j in range(i, n):
            if s[i] == s[j] and (j - i <= 2 or is_pal[i+1][j-1]):
                is_pal[i][j] = True

    def backtrack(start):
        if start == n:
            result.append(path[:])
            return
        for end in range(start, n):
            if not is_pal[start][end]:  # ← 可行性剪枝
                continue
            path.append(s[start:end+1])
            backtrack(end + 1)
            path.pop()

    backtrack(0)
    return result
```

### 30.3.2 最优性剪枝（Branch and Bound / Pruning）

**核心思想**：在求最优解（最大值/最小值）的回溯中，维护一个当前已知的最优解 `best`。若当前部分解 + **理论最优延伸**（即"下界"或"上界"的估计）已经不优于 `best`，则剪枝该分支。

$$\text{当前代价} + \text{估计剩余最小代价} \geq \text{当前最优} \Rightarrow \text{剪枝}$$

（对于最大化问题，改为：当前 + 估计剩余最大 ≤ best → 剪枝）

**下界/上界函数的质量决定剪枝效果**：
- **越紧（tight）越好**：理想情况下下界等于实际最优 → 只保留最优路径；
- **越快越好**：$O(1)$ 或 $O(k)$ 级别的估算，避免比递归本身还慢。

**经典应用：分数背包作为整数背包的上界**

对于 0/1 背包的回溯求解，用**贪心计算分数背包价值**作为上界（分数背包允许取物品的一部分，所以其价值 $\geq$ 0/1 背包的最优价值）：

```python
def knapsack_branch_bound(weights: list, values: list, W: int) -> int:
    """
    0/1 背包 Branch and Bound（最优性剪枝）
    上界：剩余容量用贪心（按价值密度）填充的分数背包值

    注意：这主要用于竞赛/研究场景。通常 DP 解决背包更简洁高效。
    """
    n = len(weights)
    # 按价值密度排序（贪心上界需要）
    items = sorted(zip(values, weights), key=lambda x: -x[0]/x[1])

    best = [0]  # 全局最优值（用列表以便内嵌函数修改）

    def upper_bound(idx: int, current_val: int, remaining_w: int) -> float:
        """从 idx 开始，用分数背包估算最大可得价值"""
        ub = current_val
        for i in range(idx, n):
            v, w = items[i]
            if w <= remaining_w:
                ub += v
                remaining_w -= w
            else:
                ub += v * (remaining_w / w)  # 分数取用
                break
        return ub

    def backtrack(idx: int, current_val: int, current_w: int) -> None:
        if idx == n or current_w == W:
            best[0] = max(best[0], current_val)
            return

        # ── 最优性剪枝：上界小于等于当前最优 → 剪枝 ─────────────────
        if upper_bound(idx, current_val, W - current_w) <= best[0]:
            return

        v, w = items[idx]
        # 路径一：选取物品 idx
        if current_w + w <= W:
            backtrack(idx + 1, current_val + v, current_w + w)
        # 路径二：不选取物品 idx
        backtrack(idx + 1, current_val, current_w)

    backtrack(0, 0, 0)
    return best[0]
```

<div data-component="PruningEffectDemo"></div>

### 30.3.3 对称性剪枝

**核心思想**：若问题具有对称性，可以将搜索空间减半（或更多）。

**N 皇后的镜像对称**：$n \times n$ 棋盘关于中央列对称，第一行皇后放在列 $j$ 的解与放在列 $n-1-j$ 的解互为镜像。因此，只需枚举第一行皇后放在左半部分（列 $0$ 到 $\lfloor n/2 \rfloor - 1$，中间列的方案数除以 2），将结果乘 2（中间列单独计算）。

```python
def total_n_queens_sym(n: int) -> int:
    """利用左右对称性，只搜索前半列，效率提升约 2×"""
    cnt = [0]
    full = (1 << n) - 1

    def bt(cols, d1, d2):
        if cols == full:
            cnt[0] += 1
            return
        avail = full & ~(cols | d1 | d2)
        while avail:
            bit = avail & (-avail)
            avail -= bit
            bt(cols | bit, (d1 | bit) >> 1, (d2 | bit) << 1)

    # 第一行只枚举左半（含中间列）
    for col in range((n + 1) // 2):
        bit = 1 << col
        bt(bit, bit >> 1, bit << 1)  # DFS（注：对角线传播方向视实现而定）

    # 若 n 为奇数，中间列对应的方案数不能翻倍，需要单独统计
    # 此处简化展示逻辑，完整实现见竞赛资料
    return cnt[0]  # 注：此处结果为利用对称性的半数统计，具体翻倍逻辑视奇偶而定
```

**全排列的对称消除**：若需要避免旋转等价的情况（如圆排列），可以固定第一个元素，只枚举剩余 $n-1$ 个元素的全排列，将解空间从 $n!$ 减到 $(n-1)!$。

### 30.3.4 排序预处理（Sort + Early Termination）

**核心价值**：许多回溯问题在枚举前对候选集合排序，可以使得循环内部提前终止（`break` 而非 `continue`），将线性扫描的剪枝退化为常数时间的终止。

**为什么排序后能 `break`？**

以组合总和为例：候选数组排序后，若 `candidates[i] > remaining`，则 `candidates[i+1], candidates[i+2], ...` 也必然大于 `remaining`（单调递增），整条尾部可以一次性剪掉。

没有排序时，只能用 `continue` 跳过当前不满足条件的候选，无法推断后续候选的情况。

**排序预处理的适用规律**：
1. **候选元素有序时可剪右子树**：组合总和、分割回文串（早期段起点）等；
2. **去重需要相邻元素比较**：#90、#40、#47 的去重都依赖 `nums[i] == nums[i-1]`，这只在排序后才有意义；
3. **贪心上界的计算**：Branch and Bound 中估算上界时，按价值密度排序才能快速贪心填充。

---

## 30.4 分支限界法（Branch and Bound）

分支限界法是求解**组合优化问题**（最优化问题的组合搜索）的一门系统方法，与回溯法同根（都是对搜索树的遍历），但有本质区别：**回溯用 DFS，分支限界用 BFS + 优先队列（Best-First Search）**。

### 30.4.1 BFS + 优先队列框架

**基本思路**：

1. 将初始状态插入优先队列（按**下界**排序，最小下界优先出队）；
2. 每次取出下界最小的节点（最"有希望"的部分解）；
3. 对该节点进行"分支"（展开其所有子节点）；
4. 对每个子节点计算下界：若下界已超过当前最优则剪掉（限界）；
5. 否则将子节点入队，继续；
6. 直到队列为空。

```python
import heapq
from typing import Any

def branch_and_bound_generic(initial_state: Any, expand, lower_bound, is_complete, goal_value):
    """
    通用分支限界框架（最小化问题）

    参数：
        initial_state: 初始状态
        expand(state): 返回所有子状态列表
        lower_bound(state): 当前状态的下界估计（越紧越好）
        is_complete(state): 判断是否为完整解
        goal_value(state): 完整解的目标值（用于更新 best）
    """
    best = float('inf')          # 当前最优解
    best_state = None

    # 堆：(下界, 状态)
    heap = [(lower_bound(initial_state), initial_state)]
    heapq.heapify(heap)

    while heap:
        lb, state = heapq.heappop(heap)

        # ── 限界剪枝 ─────────────────────────────────────────────
        if lb >= best:
            continue   # 即使最优情况也不比当前 best 好 → 丢弃

        # ── 完整解处理 ────────────────────────────────────────────
        if is_complete(state):
            val = goal_value(state)
            if val < best:
                best = val
                best_state = state
            continue

        # ── 分支展开 ──────────────────────────────────────────────
        for child in expand(state):
            child_lb = lower_bound(child)
            if child_lb < best:   # 只有有希望的子节点才入队
                heapq.heappush(heap, (child_lb, child))

    return best, best_state
```

```cpp
#include <bits/stdc++.h>
using namespace std;

// 通用分支限界框架（C++ 示意，需根据具体问题特化）
struct Node {
    double lb;     // 下界
    int    level;  // 搜索树层级（已决策的变量数）
    double value;  // 当前已累积的目标值
    // ... 其他状态字段

    // 优先队列：小下界优先
    bool operator>(const Node& o) const { return lb > o.lb; }
};

double branchAndBound(Node root) {
    double best = 1e18;
    priority_queue<Node, vector<Node>, greater<Node>> pq;
    pq.push(root);

    while (!pq.empty()) {
        Node cur = pq.top(); pq.pop();

        if (cur.lb >= best) continue;   // 限界剪枝

        // 若为叶节点（完整解）
        // if (isLeaf(cur)) { best = min(best, cur.value); continue; }

        // 分支：展开左右子节点
        // for (Node child : expand(cur)) {
        //     child.lb = computeLB(child);
        //     if (child.lb < best) pq.push(child);
        // }
    }
    return best;
}
```

### 30.4.2 上界/下界函数的设计质量

分支限界法的效率几乎完全取决于**界函数（bounding function）的质量**：

| 界函数质量 | 节点展开数 | 效果描述 |
|---|---|---|
| 完美界（等于真实最优） | $O(n)$ 或 $O(\text{问题规模})$ | 一路直达最优解，无冗余展开 |
| 紧界（接近真实最优） | 指数但系数极小 | 大量分支被提前剪掉 |
| 松界（远大于真实最优） | 退化为暴力搜索 | 几乎无剪枝效果 |

**设计下界的通用技巧**：
1. **松弛约束**：去掉整数性约束（LP 松弛），用线性规划的最优值作为下界（0/1 整数规划中常见）；
2. **贪心估算**：在背包问题中，用分数背包的最优值作为当前状态的上界（可得价值上界）；
3. **已知最优解的前缀固定**：若某子问题有已知下界公式（如最短路），直接使用。

**设计上界的通用技巧**（最大化问题）：
1. 当前已获得的价值 + 剩余物品（或任务）的贪心估计；
2. 若当前上界 ≤ 全局最优 → 剪枝。

### 30.4.3 应用：TSP 分支限界解法

旅行商问题（TSP）的分支限界法是教科书上的经典案例，核心是设计一个**较紧的下界**。

**常用下界：最小生成树（MST）下界**

对于当前路径已访问城市集合 $S$ 和当前位于城市 $v$，下界估计为：
$$\text{lb}(S, v) = \text{当前已走路径长度} + \text{remaining}(V \setminus S \cup \{v\}) + \text{minimum edge back to start}$$

其中 $\text{remaining}$ 是对未访问城市（加上当前城市 $v$）的子图做 MST 的最优代价——MST 是所有哈密顿路径代价的下界，因为哈密顿路径是生成树的一种特殊情形。

```python
import heapq
from itertools import combinations

def tsp_branch_bound(dist: list[list[int]]) -> int:
    """
    TSP 分支限界（仅演示框架，下界用简化版：剩余城市最小入边 + 出边和的一半）
    编注：精确的 MST 下界实现较复杂，此处用简化的减少矩阵下界（Assignment 松弛）

    注意：TSP Branch and Bound 在面试中基本不考，竞赛中 n<=20 使用状压 DP，
          n 更大时用 LKH 等启发式。此处仅作算法理解。
    """
    n = len(dist)
    INF = float('inf')

    def reduced_cost_matrix(d: list[list[float]]) -> tuple[float, list[list[float]]]:
        """行列归约：计算归约后的矩阵和归约成本（下界增量）"""
        d = [row[:] for row in d]  # 深拷贝
        cost = 0.0
        # 行归约
        for i in range(n):
            mn = min(d[i])
            if mn > 0 and mn != INF:
                cost += mn
                d[i] = [x - mn for x in d[i]]
        # 列归约
        for j in range(n):
            mn = min(d[i][j] for i in range(n))
            if mn > 0 and mn != INF:
                cost += mn
                for i in range(n): d[i][j] -= mn
        return cost, d

    # 初始状态：归约矩阵
    init_cost, init_mat = reduced_cost_matrix([[float(x) for x in row] for row in dist])

    # 状态：(下界, 当前节点, 路径, 归约矩阵)
    best = INF
    heap = [(init_cost, 0, [0], init_mat)]

    while heap:
        lb, node, path, mat = heapq.heappop(heap)

        if lb >= best:
            continue

        if len(path) == n:
            # 回到起点，完整哈密顿回路
            total = lb + mat[node][0]
            if total < best:
                best = total
            continue

        visited = set(path)
        for next_city in range(n):
            if next_city in visited:
                continue
            # 扩展：计算新的归约矩阵
            new_mat = [row[:] for row in mat]
            # 封闭 node 行和 next_city 列
            for j in range(n): new_mat[node][j] = INF
            for i in range(n): new_mat[i][next_city] = INF
            new_mat[next_city][0] = INF  # 防止提前回到起点（若路径未完成）

            edge_cost = mat[node][next_city]
            add_cost, new_mat = reduced_cost_matrix(new_mat)
            new_lb = lb + edge_cost + add_cost

            if new_lb < best:
                heapq.heappush(heap, (new_lb, next_city, path + [next_city], new_mat))

    return int(best)


# 测试（4 城市）
dist4 = [
    [0,  10, 15, 20],
    [10,  0, 35, 25],
    [15, 35,  0, 30],
    [20, 25, 30,  0]
]
print(tsp_branch_bound(dist4))  # 80 (A→B→D→C→A)
```

```cpp
// TSP 分支限界（C++ 精简框架）
#include <bits/stdc++.h>
using namespace std;

const int INF = 1e9;
int n;

struct State {
    double lb;             // 当前下界
    int    node;           // 当前城市
    vector<int> path;      // 已走路径
    vector<vector<double>> mat;  // 归约矩阵

    bool operator>(const State& o) const { return lb > o.lb; }
};

// 归约矩阵并返回下界增量
double reduce(vector<vector<double>>& mat, int n) {
    double cost = 0;
    for (auto& row : mat) {
        double mn = *min_element(row.begin(), row.end());
        if (mn > 0 && mn < INF) {
            cost += mn;
            for (auto& x : row) x -= mn;
        }
    }
    for (int j = 0; j < n; j++) {
        double mn = INF;
        for (int i = 0; i < n; i++) mn = min(mn, mat[i][j]);
        if (mn > 0 && mn < INF) {
            cost += mn;
            for (int i = 0; i < n; i++) mat[i][j] -= mn;
        }
    }
    return cost;
}

int tspBnB(vector<vector<int>>& dist) {
    int n = dist.size();
    // 初始化归约矩阵
    vector<vector<double>> initMat(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            initMat[i][j] = (i == j) ? INF : dist[i][j];

    double initCost = reduce(initMat, n);
    double best = INF;

    priority_queue<State, vector<State>, greater<State>> pq;
    pq.push({initCost, 0, {0}, initMat});

    while (!pq.empty()) {
        auto [lb, node, path, mat] = pq.top(); pq.pop();
        if (lb >= best) continue;
        if ((int)path.size() == n) {
            best = min(best, lb + mat[node][0]);
            continue;
        }
        set<int> visited(path.begin(), path.end());
        for (int next = 0; next < n; next++) {
            if (visited.count(next)) continue;
            auto newMat = mat;
            for (int j = 0; j < n; j++) newMat[node][j] = INF;
            for (int i = 0; i < n; i++) newMat[i][next] = INF;
            newMat[next][0] = INF;
            double edgeCost = mat[node][next];
            double addCost  = reduce(newMat, n);
            double newLb    = lb + edgeCost + addCost;
            if (newLb < best) {
                auto newPath = path;
                newPath.push_back(next);
                pq.push({newLb, next, newPath, newMat});
            }
        }
    }
    return (int)best;
}
```

### 30.4.4 回溯（DFS）vs 分支限界（BFS + 优先队列）

理解两者的本质区别，是选择算法框架的关键：

| 维度 | 回溯（Backtracking DFS） | 分支限界（Branch and Bound BFS+PQ） |
|---|---|---|
| **遍历方式** | DFS（深度优先） | Best-First BFS（最优优先） |
| **适用目标** | 找所有解 / 判断存在性 | 找最优解（最大化/最小化） |
| **内存消耗** | $O(\text{depth})$（栈） | $O(\text{活跃节点数})$（可能指数级） |
| **找到第一个解** | 快（DFS 快速深入） | 慢（需要维护优先队列） |
| **保证找最优** | 需遍历完（或记录最优再剪枝） | 一旦出队即为最优（若界函数正确） |
| **实现复杂度** | 简单（递归模板） | 复杂（需要设计界函数 + 优先队列） |
| **典型题型** | N皇后、全排列、子集、数独 | TSP 精确解、Job Scheduling 最优 |

**何时选哪个？**

```
需要找「所有」满足约束的解（或计数），或问题是判断性的（Yes/No）
    → 回溯（DFS）

需要找「最优」解，且状态空间太大无法用 DP 处理（如 TSP，n>20 时无法状压）
    → 分支限界（BFS + 优先队列）

能用 DP 的情形（状态可以紧凑表达、子问题重叠）
    → 动态规划（不用搜索）
```

**内存注意事项**：分支限界的优先队列在最坏情况下会同时持有指数级节点，对内存压力远大于回溯。在 $n$ 较大时，回溯+最优性剪枝（DFS Branch and Bound）是更实用的选择——牺牲部分"最优先展开"的效率，换取 $O(n)$ 的栈空间。

---

## 章节总结

本章系统介绍了回溯与剪枝的完整框架。以下是核心要点的提炼：

### 核心模板（必须熟记）

```python
def backtrack(path, choices, ...):
    if 满足终止条件:
        result.append(path[:])   # 或直接返回 True
        return

    for each choice in filtered_choices:  # 可行性剪枝在这里
        apply(choice)       # 做选择
        backtrack(...)      # 递归
        undo(choice)        # 撤销
```

### 经典题目速查

| 题目 | 难度 | 核心技巧 |
|---|---|---|
| #78 子集 | 🟢 简单 | 基础回溯，`start` 避免重复 |
| #46 全排列 | 🟡 中等 | `used[]` 标记 |
| #39 组合总和 | 🟡 中等 | 排序 + `break`，`start = i`（可重用） |
| #79 单词搜索 | 🟡 中等 | 二维 DFS，原地标记 `'#'` |
| #90 含重复子集 | 🟡 中等 | 排序 + `i > start and nums[i]==nums[i-1]` |
| #40 组合总和 II | 🟡 中等 | 同上去重手法 |
| #47 全排列 II | 🟡 中等 | 排序 + `not used[i-1]` 去重 |
| #51 N 皇后 | 🔴 困难 | 集合/位掩码冲突检测 |
| #37 数独 | 🔴 困难 | 位掩码前向检查，约束传播 |
| #131 分割回文串 | 🟡 中等 | 预处理回文表 + 回溯 |
| #212 单词搜索 II | 🔴 困难 | Trie + 网格 DFS |

### 剪枝选择指南

| 情景 | 推荐剪枝 |
|---|---|
| 候选有下界限制（如总和范围） | 排序 + `break` |
| 候选有重复 | 排序 + 同层跳过相邻相同元素 |
| 结构约束（行/列/对角线） | 集合/位掩码 $O(1)$ 冲突检测 |
| 最优化问题 | 贪心上界/下界 → 最优性剪枝 |
| 问题有对称性 | 固定部分决策（如排列固定首元素） |

---

**⚠️ 常见错误与陷阱**：

1. **忘记撤销（undo）**：选择后若不撤销，不同分支之间的状态会互相污染，产生错误结果；
2. **`result.append(path)` 而非 `result.append(path[:])`**：直接追加 `path` 引用，最终所有结果都是同一个对象（通常是空列表）；
3. **全排列去重条件写错**：`not used[i-1]`（父层未选）而非 `used[i-1]`（父层已选）——两种都能跑，但含义不同，后者相当于另一种等价去重方式；
4. **组合问题混淆 `start=i` 和 `start=i+1`**：**可以重复使用** → `i`；**不可重复** → `i+1`；
5. **N 皇后对角线方向弄反**：主对角线 `row - col` 相同；副对角线 `row + col` 相同（经常混淆）。

**🎯 面试高频考点**：
- 手写 #46（全排列）、#39（组合总和）、#51（N 皇后）的完整代码；
- 解释 #47（全排列 II）去重条件 `i>0 and nums[i]==nums[i-1] and not used[i-1]` 的含义；
- 分析回溯的时间复杂度（最坏指数，但解释剪枝如何改善）；
- 回溯 vs DP 的选择标准。

**💡 思考题**：
1. 数独的约束传播（Arc Consistency / AC-3 算法）能在多大程度上减少回溯次数？有没有方法在不回溯的情况下求解大多数数独谜题？
2. N 皇后问题有没有非搜索的多项式时间解法？（提示：对于 $n \geq 1$，已知构造性解法，可以 $O(n)$ 时间给出一个合法布局）
3. 在分支限界中，如果界函数永远返回 $-\infty$（对最大化问题），退化成什么算法？反过来，如果界函数极度紧（等于真实最优值），退化成什么？

**参考资料**：Skiena Chapter 7；《算法》第 4 版 Chapter 6；LeetCode 官方题解；Peter Norvig 的《数独求解器》博文。

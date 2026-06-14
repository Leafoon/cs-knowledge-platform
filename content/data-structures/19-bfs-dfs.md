# Chapter 19: 图遍历——BFS 与 DFS（Graph Traversal: BFS & DFS）

> **学习目标**：  
> 掌握广度优先搜索（BFS）与深度优先搜索（DFS）的完整实现，理解顶点染色、时间戳、BFS 最短路径定理及括号定理的精确含义；熟练分辨 DFS 中四类边（树边、前向边、后向边、横跨边）及其在有向图与无向图中的异同；能将 BFS/DFS 框架应用于二部图检测、环检测、连通分量、多源扩展等经典问题；为后续拓扑排序、强连通分量、最短路等高级图算法打下核心基础。

---

## 19.1 广度优先搜索（BFS，Breadth-First Search）

### 19.1.1 BFS 的直觉——像波纹一样向外扩散

**生活比喻——投石入水**：想象你站在一片平静的湖边，把一块石头扔进湖中心。石头落点处首先产生涟漪（第 0 层，起点），涟漪一圈圈向外扩散——第 1 圈比第 0 圈远，第 2 圈比第 1 圈远，以此类推。整个波纹扩散的顺序就是 **BFS（Breadth-First Search，广度优先搜索）** 的遍历顺序：**先把所有"近处"的顶点访问完，再去访问"远处"的顶点**。

```
起点 s = 节点 0，图为无向图：

    0 ─── 1 ─── 4
    │     │
    2 ─── 3

BFS 遍历顺序（按层）：
  第 0 层：{0}           —— 入水点
  第 1 层：{1, 2}        —— 距离 s 为 1 的所有节点
  第 2 层：{4, 3}        —— 距离 s 为 2 的所有节点
```

**BFS 的关键特性**：
- 极自然地给出**最短路径**（在**无权图**中，BFS 到达某节点时所经步数就是从起点到该节点的最短距离）。
- 使用一个**队列（Queue，先进先出 FIFO）** 来保证"近的先出、远的后出"。
- 访问顺序与图的邻接表存储顺序有关；对同一张图，邻接表顺序不同则 BFS 序也可能不同，但**分层结构**是唯一确定的。

---

### 19.1.2 顶点染色——三色标记法

BFS 使用三种颜色来追踪每个顶点的状态，这是 CLRS 书中最经典的描述方式：

| 颜色 | 状态 | 含义 |
|---|---|---|
| 🤍 **白色（White）** | 尚未发现 | 还没有被访问过，完全陌生 |
| 🩶 **灰色（Gray）** | **已发现但未完成** | 已入队，正在处理（边界前沿，Frontier） |
| 🖤 **黑色（Black）** | **完全处理完毕** | 已出队，所有邻居都已入队 |

**技术上**，实际代码中我们只需要一个 `visited[]` 布尔数组（白=False，灰/黑=True）即可，但理解三色状态有助于分析 BFS 的正确性与终止性：

- **入队时立即变灰**（而非出队时才变色！）——这是避免重复入队的关键：若等到出队才标记，同一节点会被多次推入队列
- **出队时变黑**，之后不再关心该节点

> ⚠️ **新手最易犯错的地方**：把 `visited[v] = True` 放在出队后而不是入队前，导致同一节点被重复入队，时间复杂度从 $O(V+E)$ 退化为 $O(V \cdot E)$！

---

### 19.1.3 BFS 算法完整实现

**算法步骤（CLRS 风格）**：

```
BFS(G, s):
  对所有 u ∈ V[G] - {s}：
    color[u] = WHITE
    d[u] = ∞
    π[u] = NIL
  color[s] = GRAY
  d[s] = 0
  π[s] = NIL
  Q = ∅（空队列）
  ENQUEUE(Q, s)
  while Q ≠ ∅:
    u = DEQUEUE(Q)
    for each v ∈ Adj[u]:
      if color[v] == WHITE:
        color[v] = GRAY
        d[v] = d[u] + 1
        π[v] = u
        ENQUEUE(Q, v)
    color[u] = BLACK
```

其中 `d[v]` 是 BFS 计算出的最短距离，`π[v]` 是 BFS 树中 `v` 的父节点（用于路径重建）。

**Python 实现**：

```python
from collections import deque
from typing import Optional

def bfs(
    graph: dict[int, list[int]],
    start: int
) -> tuple[dict[int, int], dict[int, Optional[int]]]:
    """
    广度优先搜索（BFS）
    
    Args:
        graph: 邻接表，graph[u] = [v1, v2, ...] 表示 u 的所有邻居
        start: 起点节点编号
    
    Returns:
        dist:   dist[v] = 从 start 到 v 的最短跳数（无权图），未可达则为 -1
        parent: parent[v] = BFS 树中 v 的父节点，起点的父节点为 None
    
    时间复杂度：O(V + E)
    空间复杂度：O(V)（visited 数组 + 队列最多存 V 个顶点）
    """
    dist: dict[int, int] = {}
    parent: dict[int, Optional[int]] = {}

    # 初始化所有节点为未访问
    for u in graph:
        dist[u] = -1
        parent[u] = None

    # 初始化起点
    dist[start] = 0           # 起点到自身距离为 0
    parent[start] = None      # 起点无父节点

    # 队列中存放"已发现但未完成处理"的顶点（灰色节点）
    queue: deque[int] = deque([start])

    while queue:
        u = queue.popleft()   # 出队：u 变为"黑色"（完全处理）

        for v in graph.get(u, []):
            if dist[v] == -1:     # v 还是白色（未访问）
                dist[v] = dist[u] + 1   # 层数 +1 即为最短距离
                parent[v] = u            # 记录 BFS 树中的父节点
                queue.append(v)          # 入队：v 变为"灰色"

    return dist, parent


def reconstruct_path(
    parent: dict[int, Optional[int]],
    start: int,
    end: int
) -> list[int]:
    """
    根据 BFS 树的 parent 数组重建从 start 到 end 的最短路径
    时间复杂度：O(V)（路径长度最长为 V-1）
    """
    if parent.get(end) is None and end != start:
        return []  # end 不可达

    path = []
    cur: Optional[int] = end
    while cur is not None:         # 从终点沿 parent 指针一直回溯到起点
        path.append(cur)
        cur = parent[cur]
    path.reverse()                 # 反转得到从起点到终点的路径
    return path


# ── 演示 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 构造无向图（邻接表）
    graph = {
        0: [1, 2],
        1: [0, 3, 4],
        2: [0, 3],
        3: [1, 2],
        4: [1],
    }

    dist, parent = bfs(graph, start=0)
    print("最短距离:", dist)
    # 输出：最短距离: {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}

    print("最短路径 0→4:", reconstruct_path(parent, 0, 4))
    # 输出：最短路径 0→4: [0, 1, 4]
```

**C++ 实现**：

```cpp
#include <vector>
#include <queue>
#include <algorithm>
#include <climits>
using namespace std;

/**
 * BFS（广度优先搜索）
 * @param graph  邻接表（graph[u] = {v1, v2, ...}）
 * @param n      顶点数（顶点编号 0 ~ n-1）
 * @param start  起点
 * @param dist   输出参数：dist[v] = 最短距离，-1 表示不可达
 * @param parent 输出参数：BFS 树父节点，-1 表示无父节点
 * 
 * 时间复杂度：O(V + E)
 * 空间复杂度：O(V)
 */
void bfs(
    const vector<vector<int>>& graph,
    int n, int start,
    vector<int>& dist,
    vector<int>& parent
) {
    // 初始化：-1 表示未访问（白色）
    dist.assign(n, -1);
    parent.assign(n, -1);

    dist[start] = 0;
    queue<int> q;
    q.push(start);    // 入队 = 变灰色

    while (!q.empty()) {
        int u = q.front(); q.pop();   // 出队 = 变黑色

        for (int v : graph[u]) {
            if (dist[v] == -1) {      // 白色：尚未访问
                dist[v] = dist[u] + 1;
                parent[v] = u;
                q.push(v);            // 入队 = 变灰色
            }
        }
    }
}

// 重建最短路径（从 start 到 end）
vector<int> reconstruct_path(
    const vector<int>& parent,
    int start, int end
) {
    if (parent[end] == -1 && end != start) return {}; // 不可达
    vector<int> path;
    for (int v = end; v != -1; v = parent[v])
        path.push_back(v);
    reverse(path.begin(), path.end());
    return path;
}

// ── 演示 ─────────────────────────────────────────────────────────────────
// int main() {
//     int n = 5;
//     vector<vector<int>> graph = {
//         {1, 2},    // 0 的邻居
//         {0, 3, 4}, // 1 的邻居
//         {0, 3},    // 2 的邻居
//         {1, 2},    // 3 的邻居
//         {1},       // 4 的邻居
//     };
//     vector<int> dist, parent;
//     bfs(graph, n, 0, dist, parent);
//     // dist: [0, 1, 1, 2, 2]
// }
```

---

### 19.1.4 BFS 最短路径定理——严格证明

> **定理（BFS 正确性）**：设 $G=(V,E)$ 为有向图或无向图，从源点 $s \in V$ 运行 BFS 后，对任意可达顶点 $v$，有：
> $$d[v] = \delta(s, v)$$
> 其中 $\delta(s,v)$ 是图中 $s$ 到 $v$ 的最短路径跳数（无权图）。

**证明思路（归纳法）**：

1. **基础**：$d[s] = 0 = \delta(s, s)$，成立。

2. **归纳步**：设所有距离小于 $k$ 的顶点均已正确计算（$d[u] = \delta(s, u)$ 对所有 $d[u] < k$ 成立）。  
   对于距离恰为 $k$ 的顶点 $v$（即 $\delta(s,v) = k$），必存在边 $(u,v)$ 使得 $\delta(s,u) = k-1$。  
   由归纳假设 $d[u] = k-1$，BFS 处理 $u$ 时发现 $v$ 尚未访问，设 $d[v] = d[u]+1 = k$。  
   另一方面，$d[v] \geq \delta(s,v) = k$（BFS 不会高估距离），故 $d[v] = k$。$\square$

3. **关键引理**：BFS 处理过程中，队列中的元素距离呈**非递减**顺序（$d_1 \leq d_2 \leq \ldots$）—— 这保证了先出队的顶点距离更小，从而先被"发现"的顶点距离不会更大。

**BFS 树（BFS Tree）**：由 BFS 过程中的 `parent[v]` 指针构成的树。BFS 树上从 $s$ 到 $v$ 的唯一路径就是图中 $s$ 到 $v$ 的**一条**最短路径（可能不唯一）。

---

### 19.1.5 BFS 时间复杂度分析

**时间复杂度：$\Theta(V + E)$**

详细分析：
- 初始化：$O(V)$（初始化所有顶点的颜色、距离、父节点）
- 每个顶点最多入队一次、出队一次：$O(V)$
- 处理每个顶点的邻居列表：所有顶点邻居列表长度之和 = 边数 $E$（有向图）或 $2E$（无向图），均为 $O(E)$
- **总计**：$O(V) + O(V) + O(E) = O(V + E)$

**空间复杂度：$O(V)$**（队列 + `dist` 数组 + `parent` 数组）

> **注意**：如果图用邻接矩阵存储，则遍历邻居需要 $O(V)$ 每个顶点，总时间 $O(V^2)$。所以 BFS/DFS 配合邻接链表才能达到 $O(V+E)$。

---

### 19.1.6 层序遍历与 BFS 的等价性

**层序遍历（Level-Order Traversal）** 是树上的 BFS——每层从左到右处理节点。BFS 是其对一般图的推广。

```
二叉树的层序遍历：
        1
       / \
      2   3
     / \   \
    4   5   6

BFS 队列变化：
  初始：[1]
  处理 1 → 入队 2, 3 → [2, 3]
  处理 2 → 入队 4, 5 → [3, 4, 5]
  处理 3 → 入队 6   → [4, 5, 6]
  处理 4,5,6（叶子，无子节点入队）

层序结果：[1, 2, 3, 4, 5, 6]
```

---

### 19.1.7 多源 BFS（Multi-Source BFS）

**场景**：有多个起点，求所有节点到**最近起点**的最短距离。  

**朴素做法**：对每个起点单独 BFS，时间 $O(K \cdot (V+E))$（$K$ 为起点数）。

**最优做法**：将所有起点**同时**加入队列（距离初始化为 0），然后统一 BFS 扩展。本质是建立一个**虚拟超级源点** $s_0$，向所有真实起点连接一条权重为 0 的边，时间 $O(V+E)$。

**经典题目**：[LeetCode #994 腐烂的橘子](https://leetcode.cn/problems/rotting-oranges/)——

```python
def oranges_rotting(grid: list[list[int]]) -> int:
    """
    多源 BFS：所有腐烂橘子同时扩散
    grid[i][j]: 0=空格, 1=新鲜橘子, 2=腐烂橘子
    返回使所有新鲜橘子腐烂所需的最少分钟数，若不可能则返回 -1
    
    时间复杂度：O(m*n)
    空间复杂度：O(m*n)（队列最多存所有格子）
    """
    from collections import deque
    m, n = len(grid), len(grid[0])
    queue: deque = deque()
    fresh_count = 0

    # 多源初始化：所有腐烂橘子同时入队（距离 = 0）
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 2:
                queue.append((i, j, 0))  # (行, 列, 当前时间)
            elif grid[i][j] == 1:
                fresh_count += 1

    if fresh_count == 0:
        return 0  # 没有新鲜橘子，立即返回

    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    max_time = 0

    while queue:
        r, c, t = queue.popleft()
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 1:
                grid[nr][nc] = 2          # 腐烂
                fresh_count -= 1
                max_time = max(max_time, t + 1)
                queue.append((nr, nc, t + 1))

    return max_time if fresh_count == 0 else -1
```

```cpp
#include <vector>
#include <queue>
using namespace std;

int orangesRotting(vector<vector<int>>& grid) {
    int m = grid.size(), n = grid[0].size();
    queue<tuple<int,int,int>> q;  // (row, col, time)
    int fresh = 0;

    // 多源初始化
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            if (grid[i][j] == 2) q.emplace(i, j, 0);
            else if (grid[i][j] == 1) fresh++;
        }

    if (fresh == 0) return 0;

    int dirs[][2] = {{-1,0},{1,0},{0,-1},{0,1}};
    int max_time = 0;

    while (!q.empty()) {
        auto [r, c, t] = q.front(); q.pop();
        for (auto& d : dirs) {
            int nr = r + d[0], nc = c + d[1];
            if (nr >= 0 && nr < m && nc >= 0 && nc < n && grid[nr][nc] == 1) {
                grid[nr][nc] = 2;
                fresh--;
                max_time = max(max_time, t + 1);
                q.emplace(nr, nc, t + 1);
            }
        }
    }
    return fresh == 0 ? max_time : -1;
}
```

<div data-component="BFSLevelExpansion"></div>

---

## 19.2 深度优先搜索（DFS，Depth-First Search）

### 19.2.1 DFS 的直觉——走迷宫时"一路走到底"

**生活比喻——走迷宫**：想象你在一个迷宫里，你的策略是"每次选择一个方向就一直走，直到走到死路，然后退回来换另一个方向"。这就是 **DFS（Depth-First Search，深度优先搜索）** 的核心思想：**沿一条路径走到尽头，再回溯探索其他路径**。

```
给定图（以邻接表为例）：
    0 → 1 → 2
            ↓
            3 → 4

DFS 从 0 开始：
  访问 0 → 进入 1 → 进入 2 → 进入 3 → 进入 4
  → 4 无新邻居，回溯到 3
  → 3 无新邻居，回溯到 2
  → 2 无新邻居，回溯到 1
  → 1 无新邻居，回溯到 0
  → 0 无新邻居，结束

DFS 访问顺序：0, 1, 2, 3, 4
```

**DFS 使用栈**：递归调用本质上是通过**系统调用栈**来维护回溯记录；也可以用**显式栈**来实现迭代版本。

---

### 19.2.2 时间戳——d[u] 与 f[u]

**DFS 的核心特色**：每个顶点被赋予两个时间戳：

| 时间戳 | 含义 | 触发时机 |
|---|---|---|
| $d[u]$（Discovery Time，发现时间）| DFS **第一次访问** $u$ 时的全局计数器值 | 进入 $u$（递归调用开始）|
| $f[u]$（Finish Time，完成时间）| DFS **完成** $u$ 的所有邻居处理后的计数器值 | 退出 $u$（递归调用结束）|

计数器从 1 开始，每操作一次（发现或完成一个顶点）加 1，因此最终计数器为 $2|V|$。

**时间戳示例**（有向图）：
```
图：0 → 1 → 3
         ↓
         2

DFS 顺序（从 0 开始）：
  发现 0: d[0]=1   → 递归进 1
  发现 1: d[1]=2   → 递归进 2
  发现 2: d[2]=3   → 2 无新邻居
  完成 2: f[2]=4   ← 回溯到 1
         → 继续进 3
  发现 3: d[3]=5
  完成 3: f[3]=6   ← 回溯到 1
  完成 1: f[1]=7   ← 回溯到 0
  完成 0: f[0]=8

时间戳汇总：
  节点 0: [1, 8]
  节点 1: [2, 7]
  节点 2: [3, 4]
  节点 3: [5, 6]
```

注意区间 $[d[u], f[u]]$ 和 $[d[v], f[v]]$ 要么**嵌套**（一个是另一个的祖先），要么**完全不相交**——这就是括号定理。

---

### 19.2.3 DFS 递归实现 + 显式栈迭代实现

**Python 实现（递归版 + 时间戳）**：

```python
from typing import Optional

def dfs_full(
    graph: dict[int, list[int]],
    vertices: list[int]
) -> tuple[dict[int, int], dict[int, int], dict[int, Optional[int]]]:
    """
    完整 DFS（含时间戳 d, f 与 parent 记录）
    支持非连通图：对所有白色顶点逐一触发 DFS
    
    Args:
        graph:    邻接表（有向图）
        vertices: 所有顶点列表（定义遍历顺序）
    
    Returns:
        discover: discover[u] = DFS 发现时间 d[u]
        finish:   finish[u]   = DFS 完成时间 f[u]
        parent:   parent[u]   = DFS 树中 u 的父节点
    
    时间复杂度：Θ(V + E)
    空间复杂度：O(V)（递归栈深度最坏为 V，即一条链的情况）
    """
    color  = {u: "WHITE" for u in vertices}  # 白/灰/黑
    discover: dict[int, int] = {}             # d[u]
    finish:   dict[int, int] = {}             # f[u]
    parent: dict[int, Optional[int]] = {u: None for u in vertices}
    time = [0]  # 用列表封装，使内嵌函数可修改（Python 闭包变量）

    def dfs_visit(u: int):
        """DFS 访问以 u 为根的子树"""
        time[0] += 1
        discover[u] = time[0]     # 记录发现时间
        color[u] = "GRAY"         # 变灰：已发现，未完成

        for v in graph.get(u, []):
            if color[v] == "WHITE":
                parent[v] = u
                dfs_visit(v)      # 递归深入（隐式使用系统栈）

        # ⚠️ 边界：color[u] = "GRAY" 时若有回向边 (u→v) 且 color[v] == "GRAY"
        #          说明存在环！（见 19.3.3 环检测）

        color[u] = "BLACK"        # 变黑：完全处理完毕
        time[0] += 1
        finish[u] = time[0]       # 记录完成时间

    # 处理非连通图：对每个白色顶点发起独立 DFS
    for u in vertices:
        if color[u] == "WHITE":
            dfs_visit(u)

    return discover, finish, parent


# ── 演示 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    graph = {0: [1], 1: [2, 3], 2: [], 3: []}
    vertices = [0, 1, 2, 3]
    d, f, par = dfs_full(graph, vertices)
    print("发现时间:", d)  # {0:1, 1:2, 2:3, 3:5}
    print("完成时间:", f)  # {0:8, 1:7, 2:4, 3:6}
```

**Python 实现（迭代版，使用显式栈）**：

```python
def dfs_iterative(graph: dict[int, list[int]], start: int) -> list[int]:
    """
    迭代版 DFS（使用显式栈，避免 Python 递归深度限制）
    注意：迭代版不直接产生时间戳，但访问顺序与递归版一致
    
    ⚠️ 技巧：若需与递归版完全一致的访问顺序，入栈时需逆序压入邻居
             （先压最后一个邻居，后压第一个邻居，使第一个邻居先出栈）
    
    时间复杂度：O(V + E)
    空间复杂度：O(V)
    """
    visited: set[int] = set()
    stack: list[int] = [start]
    order: list[int] = []

    while stack:
        u = stack.pop()           # 栈顶出栈
        if u in visited:
            continue              # 已访问则跳过（之前已入栈但尚未处理）
        visited.add(u)
        order.append(u)

        # 逆序入栈，保证邻居按原顺序被访问
        for v in reversed(graph.get(u, [])):
            if v not in visited:
                stack.append(v)

    return order
```

**C++ 实现（递归版 + 时间戳）**：

```cpp
#include <vector>
#include <functional>
using namespace std;

struct DFSResult {
    vector<int> discover;   // d[u]
    vector<int> finish;     // f[u]
    vector<int> parent;     // -1 表示无父节点
};

DFSResult dfs_full(const vector<vector<int>>& graph, int n) {
    DFSResult res;
    res.discover.resize(n, -1);
    res.finish.resize(n, -1);
    res.parent.resize(n, -1);

    vector<int> color(n, 0);  // 0=WHITE, 1=GRAY, 2=BLACK
    int timer = 0;

    // 使用 std::function 定义递归 lambda（C++14+）
    function<void(int)> dfs_visit = [&](int u) {
        color[u] = 1;                    // 变灰
        res.discover[u] = ++timer;       // 记录发现时间

        for (int v : graph[u]) {
            if (color[v] == 0) {         // WHITE：未访问
                res.parent[v] = u;
                dfs_visit(v);
            }
            // 若 color[v] == 1 (GRAY)：后向边，说明有环
        }

        color[u] = 2;                    // 变黑
        res.finish[u] = ++timer;         // 记录完成时间
    };

    // 处理非连通图
    for (int u = 0; u < n; u++)
        if (color[u] == 0)
            dfs_visit(u);

    return res;
}
```

> **Python 递归深度限制**：Python 默认最大递归深度为 1000。若图是一条长度 > 1000 的链，递归版 DFS 会触发 `RecursionError`。解决方案：① 用迭代版 DFS；② `sys.setrecursionlimit(200000)`（竞赛中常见但不推荐生产环境）。

<div data-component="DFSTimestamp"></div>

---

### 19.2.4 DFS 边的分类——四大边类型

DFS 的一个核心能力：**根据时间戳对图中所有边进行分类**，这四类边在有向图算法（拓扑排序、SCC、环检测）中极为重要。

**当 DFS 处理边 $(u, v)$ 时，根据 $v$ 的颜色进行分类**：

| 颜色 | 类型 | 含义 | 时间戳关系 |
|---|---|---|---|
| 🤍 **白色** | **树边（Tree Edge）** | DFS 树的边，$v$ 通过 $u$ 被发现 | $d[u] < d[v] < f[v] < f[u]$（嵌套）|
| 🖤 **黑色**，且 $d[u] < d[v]$ | **前向边（Forward Edge）** | 有向图中，$u$ 是 $v$ 的祖先，但不走树边 | $d[u] < d[v] < f[v] < f[u]$（嵌套）|
| 🩶 **灰色** | **后向边（Back Edge）** | $v$ 是 $u$ 的祖先（DFS 栈中），**表示有向环** | $d[v] < d[u] < f[u] < f[v]$（$u$ 在 $v$ 区间内）|
| 🖤 **黑色**，且 $d[u] > d[v]$ | **横跨边（Cross Edge）** | $u$ 与 $v$ 没有祖先-后代关系 | $d[v] < f[v] < d[u] < f[u]$（完全不相交）|

**形象理解**：

```
DFS 树（以下符号中「─ 」代表 DFS 树路径）

   s ────── a ────── b
   │              ⟋（前向边：s→b，越过树边）
   │        ⟵（后向边：b→a，指向祖先）
   └── c ────── d
          ←─────（横跨边：d→a，穿越不同子树）
```

**关键结论（仅适用于有向图 DFS）**：
- **后向边存在** $\Leftrightarrow$ 图中**有环**
- 无向图 DFS **只有树边和后向边**（无前向边、无横跨边）——详见 19.2.5

<div data-component="DFSEdgeClassifier"></div>

---

### 19.2.5 括号定理（Parenthesis Theorem）

> **括号定理（CLRS 定理 20.7）**：  
> 在对有向图或无向图 $G$ 进行 DFS 后，对于任何两个顶点 $u$ 和 $v$，以下三者**恰好**满足其一：  
> 1. $[d[u], f[u]]$ 与 $[d[v], f[v]]$ **完全不相交**（$u$ 和 $v$ 无祖先-后代关系）  
> 2. $[d[u], f[u]] \subseteq [d[v], f[v]]$（$u$ 是 $v$ 的后代）  
> 3. $[d[v], f[v]] \subseteq [d[u], f[u]]$（$v$ 是 $u$ 的后代）

**括号类比**：若把每个顶点 $u$ 的发现时间视为"左括号 $($"，完成时间视为"右括号 $)$"，则 DFS 遍历产生的括号序列是**合法匹配的括号序列**！

```
示例（上面时间戳例子）：
  节点 0: [1, 8] → ( ... )
  节点 1: [2, 7] →   ( ... )
  节点 2: [3, 4] →     ( )
  节点 3: [5, 6] →       ( )

括号序列：( ( ( ) ( ) ) )
           0 1 2 2 3 3 1 0
```

**白色路径定理（White-Path Theorem）**：  
> $v$ 是 DFS 树中 $u$ 的后代，当且仅当：在 DFS 发现 $u$ 的时刻（$d[u]$），存在一条从 $u$ 到 $v$ 的**全为白色顶点**（未访问顶点）的路径。

---

### 19.2.6 无向图 DFS 为何没有前向边和横跨边？

> **思考题**：无向图 DFS 中只有"树边"和"后向边"，为什么没有"前向边"和"横跨边"？

**论证**：

- 在**无向图**中，边 $\{u, v\}$ 可以从 $u$ 遍历到 $v$，也可以从 $v$ 遍历到 $u$。

- **不存在横跨边**：假设 DFS 处理 $u$ 时发现边 $\{u, v\}$，此时 $v$ 是黑色（已完成）。  
  由于无向图中 $v$ 有边指向 $u$，当 DFS 在 $v$ 的子树中时，一定会沿边 $\{v, u\}$ 到达 $u$——  
  但 $u$ 当时是白色（因为 DFS 还没访问 $u$），所以 $\{v, u\}$ 是树边，$u$ 在 $v$ 的子树中，即 $[d[u],f[u]] \subset [d[v],f[v]]$。  
  那 DFS 处理 $u$ 时发现 $v$ 是黑色，意味着 $f[v] < d[u]$，矛盾！故不存在横跨边。

- **不存在前向边**：前向边 $(u, v)$ 要求 $v$ 是 $u$ 的后代且已变黑，但无向图中边 $\{v, u\}$ 在处理 $v$ 时就已经被视为后向边（$u$ 是 $v$ 的祖先），所以从 $u$ 看 $v$ 的方向不会以"前向边"分类。

**结论**：无向图 DFS 中：
- 树边：DFS 树路径上的边
- 后向边：指向祖先的边（形成环）
- **没有前向边，没有横跨边**

---

## 19.3 BFS 与 DFS 的经典应用

### 19.3.1 二部图检测（Bipartite Graph Detection）

**什么是二部图（Bipartite Graph）？**  
顶点集 $V$ 可以被分成两个独立子集 $U$ 和 $W$（$U \cup W = V$，$U \cap W = \emptyset$），使得**所有边都跨越** $U$ 和 $W$（即同一子集内没有边）。也称**二分图**。

**直觉**：能否用**两种颜色**给图着色，使得任意相邻两节点颜色不同？能 $\Leftrightarrow$ 是二部图。

**等价判断**：$G$ 是二部图 $\Leftrightarrow$ $G$ 中**不存在奇数长度的环**（奇环）。

**算法（BFS 2-着色法）**：

```
对未访问的每个顶点 u，从 u 发起 BFS，给 u 着色 0：
  队列处理顶点 v 时：
    对 v 的每个邻居 w：
      若 w 未着色：给 w 着色 (1 - color[v])，入队
      若 w 已着色且 color[w] == color[v]：同色相邻，不是二部图！
若所有顶点处理完未发现冲突：是二部图
```

**Python 实现**：

```python
from collections import deque

def is_bipartite(graph: dict[int, list[int]], vertices: list[int]) -> bool:
    """
    BFS 2-着色法检测二部图
    支持非连通图
    
    时间复杂度：O(V + E)
    空间复杂度：O(V)
    
    原理：二部图 ⟺ 无奇数环 ⟺ 可 2-着色（BFS 层间交替着色）
    """
    color: dict[int, int] = {}  # -1 表示未着色, 0/1 表示两种颜色

    for start in vertices:
        if start in color:
            continue  # 已处理过该连通分量

        color[start] = 0
        queue: deque = deque([start])

        while queue:
            u = queue.popleft()
            for v in graph.get(u, []):
                if v not in color:
                    # v 未着色：给对面颜色
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    # 同色相邻 → 存在奇环 → 不是二部图
                    return False

    return True


# ── 演示 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 二部图示例（左边: 0,1 右边: 2,3）
    g_bipartite = {0: [2, 3], 1: [2, 3], 2: [0, 1], 3: [0, 1]}
    print(is_bipartite(g_bipartite, list(g_bipartite)))  # True

    # 非二部图（含奇环 0-1-2-0）
    g_odd_cycle = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    print(is_bipartite(g_odd_cycle, list(g_odd_cycle)))  # False
```

**C++ 实现**：

```cpp
#include <vector>
#include <queue>
using namespace std;

bool isBipartite(const vector<vector<int>>& graph, int n) {
    /*
     * BFS 2-着色法
     * color[u] = -1: 未着色
     * color[u] = 0/1: 两种颜色
     * 时间复杂度：O(V + E)
     */
    vector<int> color(n, -1);

    for (int start = 0; start < n; start++) {
        if (color[start] != -1) continue;  // 已处理

        color[start] = 0;
        queue<int> q;
        q.push(start);

        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v : graph[u]) {
                if (color[v] == -1) {
                    color[v] = 1 - color[u];
                    q.push(v);
                } else if (color[v] == color[u]) {
                    return false;  // 同色相邻 → 不是二部图
                }
            }
        }
    }
    return true;
}
```

> **应用**：二部图判断在**稳定匹配**（如求职-岗位匹配）、**任务分配**、**图的染色**等问题中极为重要。LeetCode #785 是该算法的标准练习题。

---

### 19.3.2 无向图连通分量（Connected Components）

**连通分量**：无向图中极大连通子图。求连通分量数等价于：从每个未访问顶点发起一次 DFS/BFS，每次发起都对应一个新的连通分量。

```python
def count_components(
    graph: dict[int, list[int]],
    vertices: list[int]
) -> tuple[int, dict[int, int]]:
    """
    求无向图的连通分量数与各顶点所属分量编号
    
    时间复杂度：O(V + E)
    空间复杂度：O(V)
    
    Returns:
        count:      连通分量总数
        component:  component[u] = u 所属分量编号（从 0 开始）
    """
    visited: set[int] = set()
    component: dict[int, int] = {}
    comp_id = 0

    def dfs(u: int):
        """DFS 标记当前连通分量的所有顶点"""
        visited.add(u)
        component[u] = comp_id
        for v in graph.get(u, []):
            if v not in visited:
                dfs(v)

    for u in vertices:
        if u not in visited:
            dfs(u)
            comp_id += 1

    return comp_id, component


# ── LeetCode #200 岛屿数量（经典网格版连通分量）──────────────────────────
def num_islands(grid: list[list[str]]) -> int:
    """
    将网格视为图，'1' 为陆地节点，相邻陆地之间有边
    统计连通陆地块的数量（即连通分量数）
    
    时间复杂度：O(m * n)
    空间复杂度：O(m * n)（最坏情况：全是陆地，DFS 栈深度 m*n）
    """
    if not grid: return 0
    m, n = len(grid), len(grid[0])
    count = 0

    def dfs(r: int, c: int):
        # 越界、水域、已访问（置为'0'的陆地）均直接返回
        if r < 0 or r >= m or c < 0 or c >= n or grid[r][c] != '1':
            return
        grid[r][c] = '0'   # 将访问过的陆地"淹没"，避免重复计数
        dfs(r-1, c)
        dfs(r+1, c)
        dfs(r, c-1)
        dfs(r, c+1)

    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                dfs(i, j)   # 发现新岛屿
                count += 1

    return count
```

```cpp
#include <vector>
using namespace std;

// 岛屿数量（C++ 版）
int numIslands(vector<vector<char>>& grid) {
    int m = grid.size(), n = grid[0].size(), count = 0;

    function<void(int,int)> dfs = [&](int r, int c) {
        if (r < 0 || r >= m || c < 0 || c >= n || grid[r][c] != '1') return;
        grid[r][c] = '0';
        dfs(r-1,c); dfs(r+1,c); dfs(r,c-1); dfs(r,c+1);
    };

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (grid[i][j] == '1') { dfs(i, j); count++; }

    return count;
}
```

---

### 19.3.3 有向图环检测——DFS 三色标记法

**问题**：给定有向图，判断是否存在环（有向环）。

**核心定理**：有向图 $G$ 有环 $\Leftrightarrow$ DFS 中存在**后向边**（即处理边 $(u, v)$ 时 $v$ 是灰色，$v$ 仍在 DFS 栈中）。

**三色标记法**：
- **白色（0）**：未访问
- **灰色（1）**：DFS 栈中（正在处理）
- **黑色（2）**：已完全处理（安全节点）

处理边 $(u, v)$ 时：
- $v$ 是白色：树边，继续 DFS
- $v$ 是灰色：**后向边 → 有环！**
- $v$ 是黑色：前向边或横跨边，无环风险

```python
def has_cycle_directed(
    graph: dict[int, list[int]],
    vertices: list[int]
) -> bool:
    """
    有向图环检测（DFS 三色标记法）
    
    颜色状态：
      0 = WHITE（未访问）
      1 = GRAY（DFS 栈中，当前正在访问的路径上）
      2 = BLACK（已完成，确认为安全节点）
    
    时间复杂度：O(V + E)
    空间复杂度：O(V)
    """
    color: dict[int, int] = {u: 0 for u in vertices}

    def dfs(u: int) -> bool:
        """返回 True 表示发现环"""
        color[u] = 1  # 进入 DFS 栈 → 变灰

        for v in graph.get(u, []):
            if color[v] == 1:
                return True   # 遇到灰色邻居 → 后向边 → 有环！
            if color[v] == 0:
                if dfs(v):    # 递归发现环
                    return True

        color[u] = 2  # 完全处理完 → 变黑（确认从 u 出发无环）
        return False

    for u in vertices:
        if color[u] == 0:
            if dfs(u):
                return True

    return False


# ── LeetCode #207 课程表（经典环检测应用）────────────────────────────────
def can_finish(num_courses: int, prerequisites: list[list[int]]) -> bool:
    """
    给定 n 门课程和先修关系列表，判断能否完成所有课程
    能完成 ⟺ 先修关系图（有向图）无环（即是 DAG）
    
    时间复杂度：O(V + E)，V = numCourses，E = len(prerequisites)
    """
    graph: dict[int, list[int]] = {i: [] for i in range(num_courses)}
    for a, b in prerequisites:
        graph[b].append(a)  # b 是 a 的先修课 → b → a

    color = [0] * num_courses  # 0=白, 1=灰, 2=黑

    def dfs(u: int) -> bool:
        color[u] = 1
        for v in graph[u]:
            if color[v] == 1: return True   # 有环
            if color[v] == 0 and dfs(v): return True
        color[u] = 2
        return False

    return not any(dfs(u) for u in range(num_courses) if color[u] == 0)
```

```cpp
#include <vector>
#include <functional>
using namespace std;

bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
    /*
     * LeetCode #207 有向图环检测
     * 使用三色标记：0=白, 1=灰, 2=黑
     * 
     * 时间复杂度：O(V + E)
     * 空间复杂度：O(V + E)（图的存储 + 递归栈）
     */
    vector<vector<int>> graph(numCourses);
    for (auto& p : prerequisites)
        graph[p[1]].push_back(p[0]);   // b → a

    vector<int> color(numCourses, 0);
    bool has_cycle = false;

    function<void(int)> dfs = [&](int u) {
        if (has_cycle) return;
        color[u] = 1;  // 变灰
        for (int v : graph[u]) {
            if (color[v] == 1) { has_cycle = true; return; }
            if (color[v] == 0) dfs(v);
        }
        color[u] = 2;  // 变黑
    };

    for (int u = 0; u < numCourses; u++)
        if (color[u] == 0) dfs(u);

    return !has_cycle;
}
```

<div data-component="GraphColoringBipartite"></div>

---

### 19.3.4 最大岛屿面积（BFS/DFS 变体）

**问题**：[LeetCode #695](https://leetcode.cn/problems/max-area-of-island/)——找面积最大的岛屿。

```python
def max_area_of_island(grid: list[list[int]]) -> int:
    """
    DFS 求最大连通分量大小（网格图中面积最大岛屿）
    
    技巧：DFS 返回"当前连通分量的顶点数"
    时间复杂度：O(m * n)
    """
    m, n = len(grid), len(grid[0])
    max_area = 0

    def dfs(r: int, c: int) -> int:
        """返回以 (r,c) 为根的岛屿面积"""
        if r < 0 or r >= m or c < 0 or c >= n or grid[r][c] != 1:
            return 0
        grid[r][c] = 0  # 标记已访问
        return 1 + dfs(r-1,c) + dfs(r+1,c) + dfs(r,c-1) + dfs(r,c+1)

    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                max_area = max(max_area, dfs(i, j))

    return max_area
```

```cpp
int maxAreaOfIsland(vector<vector<int>>& grid) {
    int m = grid.size(), n = grid[0].size(), ans = 0;

    function<int(int,int)> dfs = [&](int r, int c) -> int {
        if (r < 0 || r >= m || c < 0 || c >= n || grid[r][c] == 0)
            return 0;
        grid[r][c] = 0;
        return 1 + dfs(r-1,c) + dfs(r+1,c) + dfs(r,c-1) + dfs(r,c+1);
    };

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (grid[i][j] == 1)
                ans = max(ans, dfs(i, j));

    return ans;
}
```

---

### 19.3.5 BFS 与 DFS 全面对比

<div data-component="BFSvsDFSComparison"></div>

| 维度 | BFS | DFS |
|---|---|---|
| **数据结构** | 队列（FIFO） | 栈（LIFO，可递归/显式）|
| **遍历特征** | 按层扩展（波纹状）| 沿路径深入（先远后退）|
| **最短路径** | ✅ 无权图最短路 | ❌ 不保证最短路 |
| **空间复杂度** | $O(V)$（宽图时队列很大）| $O(V)$（深图时栈很深）|
| **时间复杂度** | $\Theta(V+E)$ | $\Theta(V+E)$ |
| **时间戳** | ❌ 无 | ✅ d[u], f[u] |
| **边分类** | ❌ 无法分类 | ✅ 树/前向/后向/横跨 |
| **适合问题** | 最短路、层序、多源扩散 | 拓扑排序、SCC、环检测、回溯 |
| **实现复杂度** | 简单（队列模板）| 中等（递归/显式栈）|

---

## 19.4 DFS 时间复杂度分析与复杂度证明

### 19.4.1 DFS 时间复杂度为 $\Theta(V+E)$

**完整分析**：

- **初始化**：设置所有顶点颜色为白色，$O(V)$
- **每个顶点**被发现一次（变灰）、完成一次（变黑），`dfs_visit` 对每个顶点恰好调用一次：$O(V)$
- **每条边**在处理其起点的邻居列表时被检查一次（有向图）或两次（无向图）：$O(E)$
- **总计**：$O(V) + O(V) + O(E) = O(V+E)$

**为什么是 $\Theta(V+E)$ 而不仅仅是 $O(V+E)$？**  
任何图遍历算法都必须至少访问每个顶点和每条边一次（否则可能漏掉某些连通分量或边），因此 $\Omega(V+E)$ 的下界也成立，从而得到 $\Theta(V+E)$。

### 19.4.2 空间复杂度分析

- **递归版 DFS**：调用栈深度 = DFS 树的高度。最坏情况（路径图/链）：$O(V)$；最好情况（完全平衡树）：$O(\log V)$。
- **BFS**：队列最多存 $O(V)$ 个元素（当所有顶点在同一层时）。

**数组/矩阵图的特殊性**：网格图（$m \times n$）的 DFS 递归深度可达 $O(m \cdot n)$，Python 递归限制需特别注意。

---

## 19.5 综合练习——典型题目解析

### 19.5.1 [LeetCode #994] 腐烂的橘子（多源 BFS）

> 已在 19.1.7 中完整给出，此处补充**时间复杂度**分析：  
> 时间 $O(mn)$（每个格子最多入队/出队一次），空间 $O(mn)$（队列）。

### 19.5.2 [LeetCode #785] 二分图（二部图 BFS 染色）

> 已在 19.3.1 中完整给出。  
> 注意：[LeetCode #785](https://leetcode.cn/problems/is-graph-bipartite/) 的图用邻接表存储，无需预处理，直接 BFS 染色。

### 19.5.3 [LeetCode #207] 课程表（有向图环检测）

> 已在 19.3.3 中给出 DFS 三色标记实现。  
> **扩展**：若要输出实际的一种课程顺序（拓扑序），见 Chapter 20 Kahn 算法。

### 19.5.4 [LeetCode #200] 岛屿数量（连通分量计数）

> 已在 19.3.2 中给出 DFS "淹没"实现。  
> **变体**：若不能修改原图，可使用额外 `visited` 集合或并查集（Union-Find）实现。

<div data-component="MultiSourceBFSDemo"></div>

---

## 本章小结

| 算法 | 核心数据结构 | 时间复杂度 | 空间复杂度 | 关键用途 |
|---|---|---|---|---|
| BFS | 队列（deque）| $\Theta(V+E)$ | $O(V)$ | 最短路（无权）、层序、多源扩散 |
| DFS（递归）| 系统调用栈 | $\Theta(V+E)$ | $O(V)$（栈深）| 时间戳、边分类、拓扑排序、SCC |
| DFS（迭代）| 显式栈 | $\Theta(V+E)$ | $O(V)$ | Python 深图无递归限制 |

**本章核心要点**：

1. **BFS 用队列**，天然给出无权图最短距离；**DFS 用栈**（递归），天然给出时间戳和边分类。
2. **三色标记**（白/灰/黑）不仅是 BFS/DFS 的正确性保证，也是环检测的基础。
3. **括号定理**：DFS 时间戳区间 $[d[u], f[u]]$ 要么嵌套要么不相交——这是 DFS 几乎所有深层性质的根基。
4. **边分类只有 DFS 能做**：树边、前向边、后向边（→有环）、横跨边；无向图只有前两种。
5. **多源 BFS** 是竞赛/面试中的高频技巧，本质是虚拟超级源点。
6. **无向图 DFS 无前向边和横跨边**——从括号定理可严格论证。

**下一章预告**：Chapter 20 将利用 DFS 完成时间 $f[u]$ 实现**拓扑排序**（Kahn 算法 + DFS 逆后序），并深入探讨**强连通分量**（Kosaraju 与 Tarjan 算法）。

---

> **扩展阅读**：  
> - CLRS 第4版 Chapter 20（广度优先搜索与深度优先搜索）  
> - Sedgewick《算法（第4版）》4.1–4.2 节  
> - MIT 6.006 Lecture 13–14（Graph Search, BFS, DFS）  
> - [LeetCode 图论题目合集](https://leetcode.cn/tag/graph/problemset/)

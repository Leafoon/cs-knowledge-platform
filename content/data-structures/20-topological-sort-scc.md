# Chapter 20: 拓扑排序与强连通分量（Topological Sort & Strongly Connected Components）

> **学习目标**：  
> 掌握 Kahn 算法（BFS + 入度队列）和 DFS 逆后序两种拓扑排序的思路与实现，理解拓扑排序与 DAG 判定的等价性；深刻理解强连通分量（SCC）的定义，掌握 Kosaraju 双趟 DFS 和 Tarjan 单趟 DFS+栈两种 SCC 算法的原理与工程细节；在 Tarjan 基础上扩展到桥与关节点（Articulation Point / Bridge）的检测；能将课程依赖、包管理、任务调度、2-SAT 等实际问题建模并求解。

---

## 20.1 有向无环图（DAG，Directed Acyclic Graph）

### 20.1.1 从「选课」理解 DAG —— 你必须先学 A，才能学 B

**生活比喻——大学选课**：你想修「机器学习」，但学校规定必须先修「线性代数」和「概率论」；修「概率论」之前又要先修「高等数学」。这些「先修关系」构成了一张**有向图**：箭头从前置课程指向后续课程。  

关键点：**这张图不能有环**——你不可能出现「要学 A 先得学 B，要学 B 先得学 A」的死循环。没有环的有向图，就叫做 **DAG（Directed Acyclic Graph，有向无环图）**。

```
高等数学 ──→ 概率论 ──→ 机器学习
                 ↗
线性代数 ───────

高等数学 ──→ 离散数学 ──→ 算法课
                              ↑
               数据结构 ──────┘

注意：以上所有箭头都朝同一个「方向」流动，不存在成环回路。
```

### 20.1.2 DAG 的严格定义与核心性质

**定义**：有向图 $G = (V, E)$ 是 DAG，当且仅当图中**不存在有向环（Directed Cycle）**，即不存在从某个顶点 $u$ 出发、经过若干有向边、又回到 $u$ 的路径。

**三条核心性质**：

| 性质 | 描述 | 为什么重要 |
|---|---|---|
| **拓扑序存在** | DAG 一定存在拓扑排序 | 反之，有拓扑排序 ⟺ 是 DAG（等价关系） |
| **源点与汇点** | DAG 至少有一个入度为 0 的顶点（源点）和一个出度为 0 的顶点（汇点） | Kahn 算法的基础 |
| **DAG 上的 DP** | DAG 是动态规划的天然载体（按拓扑序 DP） | 最长路径、关键路径、 DAG 最短路等 |

> **证明「DAG 至少一个入度为 0 的节点」**：反证法——若每个节点入度均 ≥ 1，则从任意节点出发，沿某条入边的反方向不断追溯前驱，由于顶点有限，必然构成一个有向环，与 DAG 的定义矛盾。∎

### 20.1.3 DAG 边数范围与最长路径

**边数上限**：$n$ 个节点的 DAG 最多有 $\binom{n}{2} = \frac{n(n-1)}{2}$ 条边（完全有向图的上三角，若节点有固定拓扑序 $v_1, v_2, \ldots, v_n$，所有边都从编号小的指向编号大的）。

**DAG 上的最长路径（关键路径，Critical Path）**：

工程项目管理中，每条边有「任务时长」权重，最长路径就是项目的最短完工时间（无法压缩）——这就是「关键路径法（CPM）」。在 DAG 上，按拓扑序做 DP 就可以在 $O(V+E)$ 内求出最长路：

```
dp[v] = max(dp[u] + w(u,v)) for all (u,v) ∈ E
```

**DAG 上的 DP 思路预热**：DAG 上做 DP 的顺序必须是拓扑序——处理节点 $v$ 时，所有「前驱节点」$u$（即有边 $u \to v$）的 $dp[u]$ 必须已经计算完毕。这就是为什么拓扑排序是图算法中最基础的工具之一。

---

## 20.2 拓扑排序（Topological Sort）

### 20.2.1 拓扑排序的定义——给节点排队

**定义**：给定 DAG $G = (V, E)$，拓扑排序是顶点集 $V$ 的一个**线性序列** $v_1, v_2, \ldots, v_n$，使得对图中**每一条有向边** $(u, v)$，$u$ 在序列中**排在** $v$ 的**前面**。

换句话说：所有「必须先做的事」都排在前面，所有「依赖前面任务的事」都排在后面。

```
图示（课程依赖）：

  数学(0) ──→ 概率(1) ──→ 统计(3)
     │                      ↑
     ↓                      │
  线代(2) ──────────────────┘

合法的拓扑序（不唯一）：
  [0, 1, 2, 3]    ✅（0在1前；0在2前；1在3前；2在3前）
  [0, 2, 1, 3]    ✅（同样满足所有有向边约束）
  [1, 0, 2, 3]    ❌（边 0→1 要求 0 在 1 前，违反）
```

> 🔑 **关键结论**：拓扑排序**存在** ⟺ 图是 **DAG**（无环有向图）。这是一个重要的等价关系，后面我们会证明它。

### 20.2.2 Kahn 算法（BFS + 入度队列）

#### 核心思想

**直觉**：入度为 0 的节点「没有任何前置依赖」，可以立即开始处理。把它输出到拓扑序后，删除它以及它的所有出边，这可能使得某些节点的入度降为 0，再把这些新的「可处理节点」加入队列……如此循环直至队列为空。

- 若最终输出的节点数 = $|V|$，说明图是 DAG，拓扑排序成功。
- 若最终输出的节点数 < $|V|$，说明存在**环**（环中所有节点的入度永远不会降为 0），可以检测出来。

#### 算法步骤

```
Kahn(G):
  1. 计算每个顶点 u 的入度 indegree[u]
  2. 将所有 indegree[u] == 0 的顶点入队 Q
  3. result = []
  4. while Q 非空:
       u = Q.dequeue()
       result.append(u)
       for each 邻居 v in Adj[u]:
           indegree[v] -= 1           ← 删除边 (u, v)
           if indegree[v] == 0:
               Q.enqueue(v)
  5. if len(result) == |V|:
       return result   ← 合法拓扑序
     else:
       return [] (或抛出异常)   ← 图中有环
```

**一步步手工追踪**（配合下方动画）：

```
图（4个节点，有向边：0→2, 1→2, 1→3, 2→3）：

  0 ──→ 2 ──→ 3
  1 ──→ 2
  1 ──→ 3

初始入度: indegree = {0:0, 1:0, 2:2, 3:2}
初始队列: Q = [0, 1]（所有入度=0的节点，顺序可任意）

第1步: 出队 0，result=[0]
  - 处理边 0→2：indegree[2] = 2-1 = 1（未降为0，不入队）
  Q = [1]

第2步: 出队 1，result=[0,1]
  - 处理边 1→2：indegree[2] = 1-1 = 0 → 2 入队
  - 处理边 1→3：indegree[3] = 2-1 = 1（未降为0）
  Q = [2]

第3步: 出队 2，result=[0,1,2]
  - 处理边 2→3：indegree[3] = 1-1 = 0 → 3 入队
  Q = [3]

第4步: 出队 3，result=[0,1,2,3]
  Q = []（空）

len(result)=4 == |V|=4，输出拓扑序 [0,1,2,3] ✅
```

<div data-component="TopologicalSortKahn"></div>

#### Python 实现

**Python 实现**：

```python
from collections import deque

def topological_sort_kahn(
    graph: dict[int, list[int]],
    n: int
) -> list[int]:
    """
    Kahn 算法（BFS + 入度队列）求拓扑排序
    
    Args:
        graph: 邻接表，graph[u] = [v1, v2, ...] 表示有向边 u→v1, u→v2,...
        n:     顶点数（顶点编号 0 ~ n-1）
    
    Returns:
        拓扑序列（列表），若图中有环则返回空列表 []
    
    时间复杂度：O(V + E)——每个顶点和每条边各处理一次
    空间复杂度：O(V)——入度数组 + 队列
    """
    # Step 1: 统计每个节点的入度
    indegree = [0] * n
    for u in range(n):
        for v in graph.get(u, []):
            indegree[v] += 1   # 边 u→v 使得 v 的入度 +1

    # Step 2: 将所有入度为 0 的节点入队（可以立即处理的节点）
    queue: deque[int] = deque()
    for u in range(n):
        if indegree[u] == 0:
            queue.append(u)

    result: list[int] = []

    # Step 3: BFS 循环
    while queue:
        u = queue.popleft()   # 取出一个「无前置依赖」的节点
        result.append(u)      # 输出到拓扑序

        # 删除 u 的所有出边，相当于「完成任务 u」
        for v in graph.get(u, []):
            indegree[v] -= 1          # v 少了一个前置依赖
            if indegree[v] == 0:      # v 的所有前置都已完成
                queue.append(v)       # v 现在可以开始处理了

    # 若所有节点都被输出，说明无环（DAG）
    if len(result) == n:
        return result
    else:
        # 有环的节点永远不会入度降为0，不会被输出
        return []   # 返回空列表表示存在有向环


# ── 演示 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 课程依赖图：0=数学，1=线代，2=概率，3=统计，4=机器学习
    # 边：必须先学 u，才能学 v
    graph = {
        0: [2, 3],   # 数学 → 概率, 数学 → 统计
        1: [2],      # 线代 → 概率
        2: [4],      # 概率 → 机器学习
        3: [4],      # 统计 → 机器学习
        4: [],       # 机器学习（终点）
    }
    order = topological_sort_kahn(graph, n=5)
    print("拓扑序:", order)
    # 可能输出: [0, 1, 2, 3, 4] 或 [0, 1, 3, 2, 4] 等（不唯一）

    # 有环的图（0→1→2→0）
    cyclic_graph = {0: [1], 1: [2], 2: [0]}
    result = topological_sort_kahn(cyclic_graph, n=3)
    print("有环图的结果:", result)  # 输出: []（表示有环）
```

**C++ 实现**：

```cpp
#include <vector>
#include <queue>
using namespace std;

/**
 * Kahn 算法求拓扑排序
 * @param graph   邻接表（有向图）
 * @param n       顶点数
 * @return 拓扑序列；若有环则返回空 vector
 *
 * 时间复杂度：O(V + E)
 * 空间复杂度：O(V)
 */
vector<int> topologicalSortKahn(
    const vector<vector<int>>& graph, int n
) {
    // Step 1: 统计每个节点的入度
    vector<int> indegree(n, 0);
    for (int u = 0; u < n; u++)
        for (int v : graph[u])
            indegree[v]++;   // 边 u→v 让 v 的入度 +1

    // Step 2: 所有入度为 0 的节点入队
    queue<int> q;
    for (int u = 0; u < n; u++)
        if (indegree[u] == 0)
            q.push(u);

    vector<int> result;

    // Step 3: BFS 主循环
    while (!q.empty()) {
        int u = q.front(); q.pop();
        result.push_back(u);

        for (int v : graph[u]) {
            if (--indegree[v] == 0)   // 入度减到 0，可以入队了
                q.push(v);
        }
    }

    // 所有节点都入队了才是合法拓扑序
    return (result.size() == (size_t)n) ? result : vector<int>{};
}

// ── 使用示例 ──────────────────────────────────────────────────────────────
// int main() {
//     int n = 5;
//     // 课程依赖图：0→2,0→3,1→2,2→4,3→4
//     vector<vector<int>> graph = {{2,3},{2},{4},{4},{}};
//     auto order = topologicalSortKahn(graph, n);
//     // order: [0,1,2,3,4] 或 [0,1,3,2,4] 等
// }
```

### 20.2.3 DFS 逆后序算法

#### 核心思想——「完成时间」揭示拓扑顺序

回顾上一章（DFS 时间戳）：DFS 会给每个节点记录「完成时间 $f[u]$」（所有后代都访问完后，$u$ 才「关闭」）。

**关键观察**：在 DAG 中，若存在边 $u \to v$，则 DFS 完成时 $f[u] > f[v]$，即 $u$ 比 $v$ 更晚结束。原因：
- 若在访问 $u$ 时 $v$ 还未被访问，DFS 会先递归进入 $v$ 并让 $v$ 先完成（$f[v] < f[u]$）。
- 若 $v$ 在访问 $u$ 之前已经完成（跨树边或前向边场景），则 $f[v] < f[u]$ 显然成立。

因此，**按 $f[u]$ 降序排列所有节点**——即「逆 DFS 完成顺序」——就是一个合法拓扑排序。

实现上，最简单的做法是：在 DFS 中，**当节点 $u$ 完成时（回溯时）将 $u$ 压入一个栈**，最后栈从顶到底读出就是拓扑序。

```
DFS-拓扑排序算法：

TopSortDFS(G):
  color[u] = WHITE for all u
  stack = []    ← 存放完成顺序（逆序）
  
  for each u in V:
    if color[u] == WHITE:
      DFS-Visit(G, u, color, stack)
  
  return reverse(stack)   ← 或者直接从栈顶读出即为拓扑序

DFS-Visit(G, u, color, stack):
  color[u] = GRAY    ← 进入（发现）
  for each v in Adj[u]:
    if color[v] == WHITE:
      DFS-Visit(G, v, color, stack)
    elif color[v] == GRAY:   ← 发现后向边！图中有环，抛出异常
      raise CycleError
  color[u] = BLACK   ← 完成
  stack.append(u)    ← 关键：完成时入栈
```

**手工追踪（同一图：0→2,1→2,1→3,2→3）**：

```
从节点 0 开始 DFS：
  进入 0（灰）→ 进入 2（灰）→ 进入 3（灰）→ 3 无未访问邻居 → 完成 3（黑），栈: [3]
  → 回到 2，2 无其他未访问邻居 → 完成 2（黑），栈: [3, 2]
  → 回到 0，0 无其他未访问邻居 → 完成 0（黑），栈: [3, 2, 0]

从节点 1 开始 DFS（0 已处理）：
  进入 1（灰）→ 邻居 2 是黑色（已完成），跳过 → 邻居 3 是黑色，跳过
  → 完成 1（黑），栈: [3, 2, 0, 1]

逆序读栈: [1, 0, 2, 3]  ✅（也是合法拓扑序）
```

<div data-component="TopologicalSortDFS"></div>

**Python 实现**：

```python
from enum import Enum

class Color(Enum):
    WHITE = 0  # 未访问
    GRAY  = 1  # 访问中（在当前 DFS 路径上）
    BLACK = 2  # 已完成

def topological_sort_dfs(
    graph: dict[int, list[int]],
    n: int
) -> list[int]:
    """
    DFS 逆后序算法求拓扑排序
    
    时间复杂度：O(V + E)
    空间复杂度：O(V)（颜色数组 + 递归栈 + 结果栈）
    
    ⚠️ 边界注意：Python 默认递归深度限制约 1000。
       如果图的节点数非常大（V > 1000），建议将递归改写为显式栈，
       或使用 sys.setrecursionlimit() 提前调大限制。
    """
    color = [Color.WHITE] * n
    stack: list[int] = []   # 存放「按完成顺序」排列的节点（逆拓扑序）
    has_cycle = [False]      # 用列表包装，方便在嵌套函数中修改

    def dfs(u: int) -> None:
        if has_cycle[0]:
            return  # 已发现环，提前终止
        
        color[u] = Color.GRAY  # 进入 u（标记为"访问中"）
        
        for v in graph.get(u, []):
            if color[v] == Color.WHITE:
                dfs(v)              # 继续深入探索
            elif color[v] == Color.GRAY:
                has_cycle[0] = True  # 后向边 (u→v)！发现有向环
                return
            # color[v] == BLACK：已完成的节点，安全跳过（前向边或横跨边）
        
        color[u] = Color.BLACK  # 完成 u
        stack.append(u)          # 完成时入栈（顺序是逆拓扑序）

    for u in range(n):
        if color[u] == Color.WHITE:
            dfs(u)

    if has_cycle[0]:
        return []

    stack.reverse()   # 逆序 = 拓扑序
    return stack


# ── 演示 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    graph = {
        0: [2],
        1: [2, 3],
        2: [3],
        3: [],
    }
    order = topological_sort_dfs(graph, n=4)
    print("DFS 拓扑序:", order)
    # 输出：DFS 拓扑序: [1, 0, 2, 3] 或类似合法序列

    # 含环的图
    cyclic = {0: [1], 1: [2], 2: [0]}
    print("有环结果:", topological_sort_dfs(cyclic, n=3))  # []
```

**C++ 实现**：

```cpp
#include <vector>
#include <algorithm>
using namespace std;

// 颜色状态
enum Color { WHITE, GRAY, BLACK };

bool dfsVisit(
    const vector<vector<int>>& graph,
    int u,
    vector<Color>& color,
    vector<int>& stack
) {
    color[u] = GRAY;  // 进入 u

    for (int v : graph[u]) {
        if (color[v] == WHITE) {
            if (!dfsVisit(graph, v, color, stack))
                return false;  // 有环，提前返回
        } else if (color[v] == GRAY) {
            return false;  // 后向边：发现有向环
        }
        // BLACK：已完成，跳过（前向边/横跨边）
    }

    color[u] = BLACK;
    stack.push_back(u);  // 完成时入栈（逆拓扑序）
    return true;
}

/**
 * DFS 逆后序求拓扑排序
 * 时间复杂度：O(V + E)
 * 空间复杂度：O(V)
 */
vector<int> topologicalSortDFS(
    const vector<vector<int>>& graph, int n
) {
    vector<Color> color(n, WHITE);
    vector<int> stack;
    stack.reserve(n);

    for (int u = 0; u < n; u++) {
        if (color[u] == WHITE) {
            if (!dfsVisit(graph, u, color, stack))
                return {};  // 有环
        }
    }

    reverse(stack.begin(), stack.end());  // 逆序 = 拓扑序
    return stack;
}
```

### 20.2.4 拓扑排序与 DAG 判定的等价性

以下是一个重要定理，理解它能让你在 DAG、拓扑排序、环检测三个问题之间自由转换：

> **定理**：有向图 $G$ 是 DAG（无有向环）**当且仅当** $G$ 存在拓扑排序。

**证明（双向）**：

**（⟹）若 $G$ 是 DAG，则存在拓扑排序**：  
用 Kahn 算法。DAG 至少有一个入度为 0 的节点（前面证明过），可以不断取出入度为 0 的节点输出并删除，因为 DAG 无环，这个过程一定能把所有节点都输出完毕。∎

**（⟸）若 $G$ 存在拓扑排序，则 $G$ 是 DAG**：  
反证法——设 $G$ 有拓扑序 $v_1, v_2, \ldots, v_n$，同时存在有向环 $u_1 \to u_2 \to \cdots \to u_k \to u_1$。在拓扑序中，$u_1$ 在 $u_2$ 前（因为边 $u_1 \to u_2$），$u_2$ 在 $u_3$ 前……$u_k$ 在 $u_1$ 前（因为边 $u_k \to u_1$）——这要求 $u_1$ 同时在 $u_1$ 的前面，矛盾。∎

**实际推论**：
- 「检测有向图是否有环」= 「尝试做拓扑排序，若失败则有环」
- Kahn 算法天然支持环检测：`len(result) < n` 时表示有环
- DFS 中灰色节点的出现（后向边）也能实时检测环

### 20.2.5 拓扑序的唯一性条件

**什么时候拓扑序唯一？**  

当且仅当在 Kahn 算法执行过程中，**队列里每次只有恰好 1 个节点**——即不存在任何两个节点可以互换顺序的情况。

等价条件：图中存在**哈密顿路径**（经过所有节点恰好一次的有向路径），即节点构成一条链 $v_1 \to v_2 \to \cdots \to v_n$。

```
唯一拓扑序：  0 → 1 → 2 → 3   （每个时刻只有一个入度=0的节点）

非唯一拓扑序（同时有 0 和 1 入度为 0）：
  0 ──→ 2
  1 ──→ 2        合法序: [0,1,2] 或 [1,0,2]
```

> 面试 Tips：「判断拓扑序是否唯一」等价于「Kahn 算法每一步队列大小是否始终为 1」，只需在 Kahn 循环中加一行判断即可。

### 20.2.6 拓扑排序的应用场景

拓扑排序是图算法中极其重要的基础工具，几乎半数图算法题都与它有关：

| 应用场景 | 具体描述 | 经典题目 |
|---|---|---|
| **课程依赖** | 判断选课顺序是否合法，输出一个可行修课计划 | LeetCode #207, #210 |
| **软件包管理** | npm/pip 依赖解析，确保安装顺序正确 | 现实工程问题 |
| **构建系统** | Makefile / Bazel 构建依赖，按拓扑序编译 | CI/CD 系统 |
| **任务调度** | 有依赖的任务流（工厂流水线、工程项目 CPM） | 项目管理 |
| **编译器** | 变量定义顺序检查、符号解析依赖分析 | 编译原理 |
| **DAG 上的 DP** | 按拓扑序做 DP（最长路径、DAG 最短路、记忆化搜索） | LeetCode #329, #1976 |
| **安全节点** | 所有出边都不在环中的节点（反向建图 + 拓扑序） | LeetCode #802 |

**经典应用精讲——课程表 II（LeetCode #210）**：

给定 $n$ 门课（$0$ 到 $n-1$），以及若干先修关系 $[a, b]$（意为「学 $a$ 之前必须先学 $b$」），输出任意一个可以完成所有课程的学习顺序，若不可能则返回空数组。

这就是一道「标准拓扑排序」题——建图（$b \to a$，$b$ 是 $a$ 的前置），跑 Kahn 即可。

---

## 20.3 强连通分量（SCC，Strongly Connected Components）

### 20.3.1 强连通的定义——「双向可达」

**定义**：在有向图 $G = (V, E)$ 中，两个节点 $u$ 和 $v$ **强连通（Strongly Connected）**，当且仅当：
- 存在从 $u$ 到 $v$ 的有向路径，**且**
- 存在从 $v$ 到 $u$ 的有向路径。

**强连通分量（SCC）**是满足「内部任意两点互相可达」的**极大**顶点子集。「极大」意味着再加入任何其他节点，该性质就不再成立。

**生活比喻——单行道城市**：把城市道路看作有向图，强连通分量就是「从其中任意一个路口出发，都可以沿单行道到达该区域内所有其他路口」的区域。不同的 SCC 之间可能有单向道路，但不能双向到达。

```
有向图示例（7个节点）：

  0 ──→ 1 ──→ 2
  ↑     │     │
  │     ↓     ↓
  └──── 3     4 ──→ 5 ──→ 6
              ↑           │
              └───────────┘

SCC 分析：
  SCC 1: {0, 1, 2, 3}   ← 0→1→3→0 形成环；2→4 无法回到 {0,1,2,3}，但 2∈{0→1→2→4}...
  
  实际：0→1→2（无回路）；0→1→3→0（有环！）
  
  重新分析：
  SCC {0,1,3}: 0→1→3→0（三元环）
  SCC {2}: 1→2，但没有 2→1 的边，所以 {2} 独立
  SCC {4,5,6}: 4→5→6→4（三元环）

最终 SCC 划分：{0,1,3}，{2}，{4,5,6}
```

### 20.3.2 SCC 压缩图是 DAG

将每个 SCC 收缩为一个超级节点（Super Node），超级节点之间的边保留，形成一张新图，称为 **SCC 压缩图（Component Graph / Condensation）**。

**定理**：SCC 压缩图是 DAG。

**证明**：若压缩图中有环 $C_1 \to C_2 \to \cdots \to C_k \to C_1$，则 $C_1$ 到 $C_k$ 有路径，$C_k$ 到 $C_1$ 也有路径，那么 $C_1 \cup C_2 \cup \cdots \cup C_k$ 内部任意两点互达，应该是同一个 SCC——与 $C_1, C_2, \ldots$ 是不同 SCC 矛盾。∎

**为什么这很重要**？许多图问题（如找出「影响整个图的最小起点集」）先求 SCC、在 DAG 上分析，比在原图上直接处理简单得多。

### 20.3.3 Kosaraju 算法——两次 DFS

#### 算法思想

Kosaraju 算法的核心洞见来自一个关于完成时间的性质：

> **关键引理**：在 DFS 树中，若 $C_1$ 和 $C_2$ 是两个不同的 SCC，且 $C_1$ 到 $C_2$ 有边，则 $C_1$ 中「最晚完成」的节点的 $f$ 值 > $C_2$ 中所有节点的 $f$ 值。

换句话说：完成时间最大的节点所在的 SCC，一定是 SCC 压缩图中的「源点 SCC」（无入边的 SCC）。

Kosaraju 算法利用这一洞见：
1. **第一次 DFS（正向图）**：计算所有节点的完成时间 $f[u]$，记录完成顺序。
2. **转置图 $G^T$**（把所有边反向）：$G^T$ 的 SCC 与 $G$ 完全相同，但 SCC 之间的方向反转。
3. **第二次 DFS（转置图）**：按 $f[u]$ **降序**处理节点（即从完成最晚的节点开始）。每次从一个未访问节点出发，能访问到的所有节点就是一个 SCC。

#### 直觉解释

在转置图 $G^T$ 中，原来的「源点 SCC」变成了「汇点 SCC」。按 $f[u]$ 降序，第一个处理的节点来自原图的「源点 SCC」——在 $G^T$ 中，从这个节点出发，只能到达同一个 SCC 内的其他节点（出不去，因为在 $G^T$ 中指向外面的边已经翻转成外面指进来的了）。

```
Kosaraju 算法步骤：

Step 1: 在正向图 G 上跑 DFS，按完成时间存入栈 S
        （完成越晚，排在栈顶）

Step 2: 构建转置图 G^T（所有边方向反转）

Step 3: 从 S 的栈顶开始，依次从未访问的节点在 G^T 上跑 DFS
        每次 DFS 能访问到的节点集合 = 一个 SCC
```

<div data-component="SCCKosarajuTwoPass"></div>

**Python 实现**：

```python
from collections import defaultdict

def kosaraju_scc(
    graph: dict[int, list[int]],
    n: int
) -> list[list[int]]:
    """
    Kosaraju 算法求所有强连通分量（SCC）
    
    Args:
        graph: 邻接表（有向图），graph[u] = [v1, v2, ...]
        n:     顶点数（0 ~ n-1）
    
    Returns:
        所有 SCC 的列表，每个 SCC 是一个节点列表
    
    时间复杂度：O(V + E)
    空间复杂度：O(V + E)（存储转置图）
    """
    # ── Step 1: 第一次 DFS（正向图），记录完成顺序 ─────────────────────────
    visited1 = [False] * n
    finish_order: list[int] = []  # 按完成顺序存放（栈底=最早完成，栈顶=最晚完成）

    def dfs1(u: int) -> None:
        """非递归版本，避免大图的 Python 递归栈溢出"""
        stack = [(u, 0)]        # (节点, 当前要处理的邻居索引)
        visited1[u] = True
        
        while stack:
            node, idx = stack[-1]
            nbrs = graph.get(node, [])
            
            if idx < len(nbrs):
                stack[-1] = (node, idx + 1)   # 更新邻居索引
                v = nbrs[idx]
                if not visited1[v]:
                    visited1[v] = True
                    stack.append((v, 0))
            else:
                # 该节点的所有邻居都处理完了（"完成"）
                stack.pop()
                finish_order.append(node)     # 完成时记录

    for u in range(n):
        if not visited1[u]:
            dfs1(u)

    # ── Step 2: 构建转置图 G^T ──────────────────────────────────────────
    trans: dict[int, list[int]] = defaultdict(list)
    for u in range(n):
        for v in graph.get(u, []):
            trans[v].append(u)   # 边 u→v 反转为 v→u

    # ── Step 3: 第二次 DFS（转置图），按 f[] 降序处理 ─────────────────────
    visited2 = [False] * n
    sccs: list[list[int]] = []

    def dfs2(start: int) -> list[int]:
        """在转置图上 DFS，收集一个 SCC 的所有节点"""
        component: list[int] = []
        stack = [start]
        visited2[start] = True
        while stack:
            u = stack.pop()
            component.append(u)
            for v in trans[u]:
                if not visited2[v]:
                    visited2[v] = True
                    stack.append(v)
        return component

    # 按完成时间逆序（栈顶 = 最晚完成 = 从这里开始）
    for u in reversed(finish_order):
        if not visited2[u]:
            scc = dfs2(u)
            sccs.append(scc)

    return sccs


# ── 演示 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 有向图：SCC = {0,1,3}, {2}, {4,5,6}
    n = 7
    graph = {
        0: [1],
        1: [2, 3],
        2: [4],
        3: [0],       # 0→1→3→0 构成环
        4: [5],
        5: [6],
        6: [4],       # 4→5→6→4 构成环
    }
    sccs = kosaraju_scc(graph, n)
    print("SCC 列表:")
    for i, scc in enumerate(sccs):
        print(f"  SCC {i}: {sorted(scc)}")
    # 输出:
    #   SCC 0: [0, 1, 3]
    #   SCC 1: [2]
    #   SCC 2: [4, 5, 6]
```

**C++ 实现**：

```cpp
#include <vector>
#include <stack>
#include <algorithm>
using namespace std;

/**
 * Kosaraju 算法求 SCC
 * 时间复杂度：O(V + E)
 * 空间复杂度：O(V + E)
 */

// Step 1: 正向图 DFS，记录完成顺序
void dfs1(
    const vector<vector<int>>& graph, int u,
    vector<bool>& visited, stack<int>& finishStack
) {
    visited[u] = true;
    for (int v : graph[u])
        if (!visited[v])
            dfs1(graph, v, visited, finishStack);
    finishStack.push(u);   // 完成时入栈
}

// Step 3: 转置图 DFS，收集一个 SCC
void dfs2(
    const vector<vector<int>>& trans, int u,
    vector<bool>& visited, vector<int>& component
) {
    visited[u] = true;
    component.push_back(u);
    for (int v : trans[u])
        if (!visited[v])
            dfs2(trans, v, visited, component);
}

vector<vector<int>> kosarajuSCC(
    const vector<vector<int>>& graph, int n
) {
    // Step 1: 正向图 DFS，获取完成顺序（栈）
    vector<bool> visited(n, false);
    stack<int> finishStack;
    for (int u = 0; u < n; u++)
        if (!visited[u])
            dfs1(graph, u, visited, finishStack);

    // Step 2: 构建转置图
    vector<vector<int>> trans(n);
    for (int u = 0; u < n; u++)
        for (int v : graph[u])
            trans[v].push_back(u);   // 边反向

    // Step 3: 按完成顺序的逆序，在转置图上 DFS
    fill(visited.begin(), visited.end(), false);
    vector<vector<int>> sccs;

    while (!finishStack.empty()) {
        int u = finishStack.top(); finishStack.pop();
        if (!visited[u]) {
            vector<int> component;
            dfs2(trans, u, visited, component);
            sccs.push_back(component);
        }
    }

    return sccs;
}
```

### 20.3.4 Tarjan 算法——单趟 DFS 找 SCC

#### 核心思想

Kosaraju 需要两次 DFS + 一张转置图，Tarjan 算法（Robert Tarjan, 1972）只用**一次 DFS** 就能找出所有 SCC，是更加优雅的方案。

**三个核心数组**：
- `disc[u]`：DFS 发现 $u$ 的时间戳（Discovery time）
- `low[u]`：从 $u$ 的子树（DFS 子树）出发，能到达的**最小 `disc` 值**（通过后向边或树边）
- `on_stack[u]`：节点 $u$ 是否当前在 SCC 栈中

**SCC 栈**：维护一个辅助栈，每个被发现的节点入栈，当一个 SCC 的「根」被找到时，弹出栈顶直到根节点，这批节点就是一个 SCC。

**low[u] 的直觉**：`low[u]` 代表「从 $u$ 出发，不越过已完成的 SCC，最远能回溯到多早的节点（最小 disc 值）」。若 `low[u] == disc[u]`，说明 $u$ 无法回溯到 DFS 树中更早的节点，$u$ 就是某个 SCC 的「根」。

#### low[u] 的更新规则（精确版）

```
DFS-Visit-Tarjan(u):
  disc[u] = low[u] = timer++   ← 发现时初始化
  stack.push(u)
  on_stack[u] = True
  
  for each v in Adj[u]:
    if disc[v] == -1:            ← v 未访问（树边）
      DFS-Visit-Tarjan(v)
      low[u] = min(low[u], low[v])   ← 通过树边更新（子节点可达更早节点）
    elif on_stack[v]:            ← v 在栈中（后向边）
      low[u] = min(low[u], disc[v])  ← 直接用 disc[v]（注意：不用 low[v]！）
    # else: v 已被访问但不在栈（属于已完成的 SCC）→ 不更新 low[u]
  
  if low[u] == disc[u]:          ← u 是 SCC 的根
    弹出栈顶直到 u，这批节点构成一个 SCC
```

> ⚠️ **关键细节**：更新 `low[u]` 时，对后向边 $(u \to v)$（$v$ 在栈中），使用 `disc[v]` 而非 `low[v]`。若使用 `low[v]`，会错误地把不属于当前 SCC 的节点纳入进来。

<div data-component="SCCTarjanStack"></div>

**Python 实现**：

```python
def tarjan_scc(
    graph: dict[int, list[int]],
    n: int
) -> list[list[int]]:
    """
    Tarjan 算法求所有强连通分量（SCC）
    单趟 DFS，时间复杂度 O(V + E)
    
    ⚠️ low[u] 更新规则：
       - 树边 (u→v)：low[u] = min(low[u], low[v])
       - 后向边 (u→v，v 在栈中)：low[u] = min(low[u], disc[v])
       - 跨边/v 不在栈：不更新（v 属于已完成的 SCC）
    """
    disc = [-1] * n          # -1 表示未发现
    low  = [0]  * n
    on_stack = [False] * n   # 节点是否在 SCC 辅助栈中
    
    scc_stack: list[int] = []   # SCC 辅助栈
    sccs: list[list[int]] = []
    timer = [0]              # 用列表包装，方便嵌套函数修改

    def dfs(u: int) -> None:
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        scc_stack.append(u)
        on_stack[u] = True

        for v in graph.get(u, []):
            if disc[v] == -1:
                # 树边：先递归探索 v
                dfs(v)
                low[u] = min(low[u], low[v])    # 通过子节点更新
            elif on_stack[v]:
                # 后向边：v 在栈中，u 可以"回溯"到 v
                low[u] = min(low[u], disc[v])   # 注意用 disc[v] 而非 low[v]

        # 判断 u 是否为某个 SCC 的根
        if low[u] == disc[u]:
            # 从栈顶弹出直到 u，这批节点是一个完整 SCC
            scc: list[int] = []
            while True:
                w = scc_stack.pop()
                on_stack[w] = False
                scc.append(w)
                if w == u:
                    break
            sccs.append(scc)

    for u in range(n):
        if disc[u] == -1:
            dfs(u)

    return sccs


# ── 演示 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    n = 7
    graph = {
        0: [1],
        1: [2, 3],
        2: [4],
        3: [0],
        4: [5],
        5: [6],
        6: [4],
    }
    sccs = tarjan_scc(graph, n)
    print("Tarjan SCC:")
    for i, scc in enumerate(sccs):
        print(f"  SCC {i}: {sorted(scc)}")
    # 输出（Tarjan 以逆拓扑序输出 SCC）:
    #   SCC 0: [4, 5, 6]
    #   SCC 1: [2]
    #   SCC 2: [0, 1, 3]
```

**C++ 实现**：

```cpp
#include <vector>
#include <stack>
#include <algorithm>
using namespace std;

struct TarjanSCC {
    int n;
    vector<vector<int>>& graph;
    vector<int> disc, low;
    vector<bool> onStack;
    stack<int> sccStack;
    vector<vector<int>> sccs;
    int timer;

    TarjanSCC(vector<vector<int>>& g, int n)
        : n(n), graph(g), disc(n, -1), low(n), onStack(n, false), timer(0) {}

    void dfs(int u) {
        disc[u] = low[u] = timer++;
        sccStack.push(u);
        onStack[u] = true;

        for (int v : graph[u]) {
            if (disc[v] == -1) {
                // 树边
                dfs(v);
                low[u] = min(low[u], low[v]);
            } else if (onStack[v]) {
                // 后向边（v 在栈中）
                low[u] = min(low[u], disc[v]);  // 用 disc[v] 而非 low[v]
            }
            // v 不在栈：属于已完成SCC，跳过
        }

        // u 是 SCC 根：弹栈收集 SCC
        if (low[u] == disc[u]) {
            vector<int> scc;
            while (true) {
                int w = sccStack.top(); sccStack.pop();
                onStack[w] = false;
                scc.push_back(w);
                if (w == u) break;
            }
            sccs.push_back(scc);
        }
    }

    vector<vector<int>> solve() {
        for (int u = 0; u < n; u++)
            if (disc[u] == -1)
                dfs(u);
        return sccs;
    }
};

// 使用示例：
// vector<vector<int>> graph = {...};
// TarjanSCC solver(graph, n);
// auto sccs = solver.solve();
```

### 20.3.5 Tarjan low-link 值的精确计算——边界情况剖析

#### 为什么后向边用 disc[v] 而不是 low[v]？

这是 Tarjan 算法中最容易出错的地方，值得深入理解。

**场景**：假设 $u \to v$ 是一条后向边（$v$ 在栈中，是 $u$ 的祖先）。

- 若用 `low[v]`：`low[v]` 可能已经被更新为某个「更早」的祖先，这会让 `low[u]` 也指向那个更早的祖先，**错误地将本不属于同一 SCC 的节点合并**。
- 若用 `disc[v]`：`disc[v]` 精确地代表 $v$ 自身被发现的时间，$u$ 通过后向边只能回到 $v$，而不是 $v$ 经由其他边能到达的更早节点——这才是「$u$ 的 SCC 能覆盖的范围」。

```
反例（若错误地用 low[v]）：

  图: 0 → 1 → 2 → 1  和  0 → 3（简化模型）

  disc: [0, 1, 2, ...]
  DFS 顺序: 0 → 1 → 2（发现后向边 2→1，low[2] = disc[1] = 1）
  
  若错误地让 low[0] = low[1]（因为有另一条路径），会误判为 {0,1,2} 同 SCC。
  但实际上 0 进不来 {1,2} 的循环。
```

#### low-link 值手工追踪示例

```
图: 0→1, 1→2, 2→0, 1→3, 3→4, 4→3

DFS 从 0 开始：

进入 0: disc[0]=0, low[0]=0, stack=[0]
  进入 1: disc[1]=1, low[1]=1, stack=[0,1]
    进入 2: disc[2]=2, low[2]=2, stack=[0,1,2]
      邻居 0 在栈中（后向边）: low[2]=min(2,disc[0])=0
    完成 2: low[2]=0 ≠ disc[2]=2, 不是 SCC 根
  回到 1: low[1]=min(1,low[2])=0
    进入 3: disc[3]=3, low[3]=3, stack=[0,1,2,3]
      进入 4: disc[4]=4, low[4]=4, stack=[0,1,2,3,4]
        邻居 3 在栈中（后向边）: low[4]=min(4,disc[3])=3
      完成 4: low[4]=3 ≠ disc[4]=4, 不是 SCC 根
    回到 3: low[3]=min(3,low[4])=3
    完成 3: low[3]=3 == disc[3]=3, 是 SCC 根！
      弹栈直到 3: 弹出 4, 3 → SCC {3,4}
      stack=[0,1,2]
  回到 1: （3 已处理，不在栈中，跳过）
  完成 1: low[1]=0 ≠ disc[1]=1, 不是 SCC 根
回到 0: low[0]=min(0,low[1])=0
完成 0: low[0]=0 == disc[0]=0, 是 SCC 根！
  弹栈直到 0: 弹出 2, 1, 0 → SCC {0,1,2}

最终 SCC: [{3,4}, {0,1,2}] ✅
```

### 20.3.6 Tarjan 的扩展：桥（Bridge）与关节点（Articulation Point）

Tarjan 算法的思想可以扩展到**无向图**，用于检测让图「断开」的关键元素：

#### 桥（Bridge / Cut Edge）

**定义**：在无向图中，删除边 $(u, v)$ 后，图的连通分量数增加，则 $(u, v)$ 是一条**桥**。

**Tarjan 桥检测**：DFS 时，若存在树边 $(u, v)$（即 $v$ 是 $u$ 的 DFS 子节点），且：
$$\text{low}[v] > \text{disc}[u]$$
则 $(u, v)$ 是桥。

直觉：`low[v] > disc[u]` 意味着从 $v$ 的子树出发，**没有任何后向边**能绕过 $(u, v)$ 到达 $u$ 或 $u$ 的祖先——那么 $(u, v)$ 就是唯一连接两侧的边，删掉它图就断了。

> ⚠️ **无向图处理重边**：若 $u$-$v$ 之间有多条边（重边），删除其中一条时另一条还存在，此时该边不是桥。实现时需要通过边的 ID 或记录父边来避免把来时的那条边当作回边更新 `low`。

#### 关节点（Articulation Point / Cut Vertex）

**定义**：在无向图中，删除顶点 $u$（及其所有关联边）后，连通分量数增加，则 $u$ 是**关节点**。

**Tarjan 关节点检测**：DFS 中，顶点 $u$ 是关节点，当且仅当满足以下**任一**条件：
1. $u$ 是 DFS 树的**根节点**，且 $u$ 在 DFS 树中有 **≥ 2 个子节点**。
2. $u$ 不是根节点，且存在 $u$ 的某个子节点 $v$，使得：$\text{low}[v] \geq \text{disc}[u]$。

```
条件2 的直觉：low[v] >= disc[u] 说明 v 的子树无法绕过 u 到达 u 的祖先，
              删除 u 后，v 的子树就与 u 的父部分断开了。
```

<div data-component="BridgeCutPointDemo"></div>

**Python 实现（桥 + 关节点）**：

```python
def find_bridges_and_cut_points(
    graph: dict[int, list[int]],
    n: int
) -> tuple[list[tuple[int, int]], list[int]]:
    """
    Tarjan 算法同时求无向图中的桥和关节点
    
    Args:
        graph: 无向图的邻接表（每条边存两次：u→v 和 v→u）
        n:     顶点数（0 ~ n-1）
    
    Returns:
        bridges:      桥的列表，每个桥为 (u, v) 元组（u < v）
        cut_points:   关节点的列表
    
    时间复杂度：O(V + E)
    """
    disc = [-1] * n
    low  = [0]  * n
    parent = [-1] * n
    timer = [0]
    
    bridges: list[tuple[int, int]] = []
    cut_points: set[int] = set()

    def dfs(u: int) -> None:
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        child_count = 0   # u 在 DFS 树中的孩子数（用于判断根节点）

        for v in graph.get(u, []):
            if disc[v] == -1:
                # 树边
                child_count += 1
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])

                # 判断桥：从 v 的子树无法到达 u 或 u 的祖先
                if low[v] > disc[u]:
                    bridges.append((min(u, v), max(u, v)))

                # 判断关节点（非根节点）：v 无法绕过 u
                if parent[u] != -1 and low[v] >= disc[u]:
                    cut_points.add(u)

            elif v != parent[u]:
                # 回边（非父节点方向）
                # ⚠️ 注意：无向图中要排除「回到父节点」的边
                low[u] = min(low[u], disc[v])

        # 关节点判断（根节点：超过1个DFS子节点）
        if parent[u] == -1 and child_count >= 2:
            cut_points.add(u)

    for u in range(n):
        if disc[u] == -1:
            dfs(u)

    return bridges, sorted(cut_points)


# ── 演示 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 无向图：
    #  0 - 1 - 2
    #  |   |
    #  3   4 - 5
    # 边：(0,1),(0,3),(1,2),(1,4),(4,5)
    graph = {
        0: [1, 3],
        1: [0, 2, 4],
        2: [1],
        3: [0],
        4: [1, 5],
        5: [4],
    }
    bridges, cut_points = find_bridges_and_cut_points(graph, n=6)
    print("桥:", bridges)
    # 所有边都只有一条路径经过，全是桥：
    # [(0,1), (0,3), (1,2), (1,4), (4,5)]
    print("关节点:", cut_points)
    # 输出: [0, 1, 4]（删除任一个，图就分裂）
```

**C++ 实现**：

```cpp
#include <vector>
#include <set>
#include <algorithm>
using namespace std;

struct BridgeCutFinder {
    int n, timer;
    vector<vector<int>>& graph;
    vector<int> disc, low, parent;
    vector<tuple<int,int>> bridges;
    set<int> cutPoints;

    BridgeCutFinder(vector<vector<int>>& g, int n)
        : n(n), timer(0), graph(g), disc(n,-1), low(n), parent(n,-1) {}

    void dfs(int u) {
        disc[u] = low[u] = timer++;
        int childCount = 0;

        for (int v : graph[u]) {
            if (disc[v] == -1) {
                childCount++;
                parent[v] = u;
                dfs(v);
                low[u] = min(low[u], low[v]);

                // 桥检测
                if (low[v] > disc[u])
                    bridges.emplace_back(min(u,v), max(u,v));

                // 关节点检测（非根节点）
                if (parent[u] != -1 && low[v] >= disc[u])
                    cutPoints.insert(u);

            } else if (v != parent[u]) {
                // 回边（去掉父节点方向）
                low[u] = min(low[u], disc[v]);
            }
        }

        // 根节点关节点判断
        if (parent[u] == -1 && childCount >= 2)
            cutPoints.insert(u);
    }

    void solve() {
        for (int u = 0; u < n; u++)
            if (disc[u] == -1) dfs(u);
    }
};
```

### 20.3.7 SCC 的应用——2-SAT 问题简介

**2-SAT** 是一类逻辑可满足性问题：给定 $m$ 个子句，每个子句是两个**字面量**（literal，变量或其否定）的析取（OR），判断是否存在对所有变量的赋值使得所有子句为真。

**建模方式**：
- 每个变量 $x_i$ 有两个节点：$x_i$（真） 和 $\neg x_i$（假）
- 对于子句 $(a \lor b)$，添加两条蕴含边：
  - $\neg a \to b$（若 $a$ 为假则 $b$ 必须为真）
  - $\neg b \to a$（若 $b$ 为假则 $a$ 必须为真）

**求解**：在 2-SAT 图上求 SCC。若存在变量 $x_i$ 使得 $x_i$ 和 $\neg x_i$ 在同一个 SCC 中，则无解。否则，根据 Tarjan 输出的 SCC 逆拓扑序给变量赋值（若 $x_i$ 所在 SCC 在拓扑序上晚于 $\neg x_i$ 所在 SCC，则 $x_i$ = 真）。

---

## 20.4 Kosaraju vs Tarjan——算法对比

| 特性 | Kosaraju | Tarjan |
|---|---|---|
| **DFS 次数** | 2 次 | 1 次 |
| **需要转置图** | ✅ 是 | ❌ 否（无需构建 $G^T$） |
| **时间复杂度** | $O(V + E)$ | $O(V + E)$ |
| **空间复杂度** | $O(V + E)$（存转置图） | $O(V)$（只用一个辅助栈） |
| **常数因子** | 约 2× 常数 | 更小 |
| **实现难度** | 相对简单（思路直观） | 略复杂（low-link 细节多） |
| **可扩展性** | SCC 专用 | 可扩展为桥/关节点检测 |
| **适合场景** | 需要理解原理时 | 工程/竞赛首选 |

> **工程实践建议**：竞赛和工程中优先使用 Tarjan（更少内存，更快常数）。理解 SCC 原理时，先学 Kosaraju（直觉更清晰），再实现 Tarjan。

---

## 20.5 经典例题精讲

### 例题 1：LeetCode #207 课程表（环检测）

**题意**：给定 $n$ 门课和先修关系 $[a, b]$（先学 $b$ 再学 $a$），判断是否可以修完所有课。

**思路**：建图 $b \to a$，跑 Kahn 判断是否有环。

```python
from collections import deque

def canFinish(numCourses: int, prerequisites: list[list[int]]) -> bool:
    graph = [[] for _ in range(numCourses)]
    indegree = [0] * numCourses
    
    for a, b in prerequisites:
        graph[b].append(a)   # b → a（先学 b）
        indegree[a] += 1
    
    queue = deque(u for u in range(numCourses) if indegree[u] == 0)
    count = 0
    
    while queue:
        u = queue.popleft()
        count += 1
        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)
    
    return count == numCourses  # 所有课都能修 = 无环
```

### 例题 2：LeetCode #1192 关键连接（Tarjan 求桥）

**题意**：给定 $n$ 个服务器和 $m$ 条网络连接（无向图），找出所有「删去后网络断开」的边（桥）。

**思路**：直接使用 Tarjan 桥检测算法。

```python
def criticalConnections(n: int, connections: list[list[int]]) -> list[list[int]]:
    graph = [[] for _ in range(n)]
    for u, v in connections:
        graph[u].append(v)
        graph[v].append(u)
    
    disc = [-1] * n
    low  = [0] * n
    parent = [-1] * n
    timer = [0]
    result = []
    
    def dfs(u):
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        for v in graph[u]:
            if disc[v] == -1:
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:        # 桥条件
                    result.append([u, v])
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])
    
    for u in range(n):
        if disc[u] == -1:
            dfs(u)
    
    return result
```

### 例题 3：LeetCode #802 找到最终安全状态

**题意**：有向图中，若从节点 $u$ 出发的**所有路径**最终都到达某个出度为 0 的节点（终端节点），则 $u$ 是「安全节点」。

**思路 1（Kahn + 反向图）**：在**反向图**中，终端节点的入度（原图出度）为 0，正是 Kahn 的「起点」。Kahn 能处理的节点就是安全的。

```python
from collections import deque

def eventualSafeNodes(graph: list[list[int]]) -> list[int]:
    n = len(graph)
    # 构建反向图，并记录原图的出度
    reverse = [[] for _ in range(n)]
    out_degree = [0] * n
    
    for u in range(n):
        for v in graph[u]:
            reverse[v].append(u)
            out_degree[u] += 1  # 原图出度（= 反向图入度）
    
    # Kahn：从原图出度=0（反向图入度=0）的节点开始
    queue = deque(u for u in range(n) if out_degree[u] == 0)
    safe = [False] * n
    
    while queue:
        u = queue.popleft()
        safe[u] = True
        for v in reverse[u]:
            out_degree[v] -= 1
            if out_degree[v] == 0:
                queue.append(v)
    
    return [u for u in range(n) if safe[u]]
```

---

## 20.6 常见错误与调试技巧

| 错误 | 场景 | 正确做法 |
|---|---|---|
| **Kahn 漏计入度** | 重边、自环 | 建图时仔细逐边统计 `indegree[v]++` |
| **忘记更新所有邻居入度** | Kahn 出队时处理邻居 | 遍历 `graph[u]` 的每一条边 |
| **Tarjan 后向边用 low[v]** | 计算 low[u] 时 | 后向边用 `disc[v]`，树边用 `low[v]` |
| **Tarjan 错误处理不在栈中的 v** | 已完成 SCC 的邻居 | 检查 `on_stack[v]`，不在栈则跳过 |
| **无向图 Tarjan 双向边** | 父节点方向的边误当回边 | 用 `parent` 数组或 `edge_id` 跳过父边 |
| **Kosaraju 第二次 DFS 忘记用转置图** | 误在原图做第二次 DFS | 第二次 DFS 必须在 $G^T$ 上 |
| **有向图中`low[v] >= disc[u]`** | 关节点检测（无向图中的规则） | 关节点只用于**无向图**！ |

---

## 20.7 复杂度总结

| 算法 | 时间 | 空间 | 备注 |
|---|---|---|---|
| Kahn 拓扑排序 | $O(V + E)$ | $O(V)$ | 可检测环 |
| DFS 拓扑排序 | $O(V + E)$ | $O(V)$ | 可检测环 |
| Kosaraju SCC | $O(V + E)$ | $O(V + E)$ | 需存转置图 |
| Tarjan SCC | $O(V + E)$ | $O(V)$ | 单趟 DFS |
| Tarjan 桥 | $O(V + E)$ | $O(V)$ | 无向图 |
| Tarjan 关节点 | $O(V + E)$ | $O(V)$ | 无向图 |

---

## 20.8 综合练习

**🟢 基础**：
1. 手工对图 $0 \to 1, 0 \to 2, 1 \to 3, 2 \to 3$ 用 Kahn 算法做拓扑排序，追踪每步的 `indegree` 数组和队列状态。
2. 什么样的 DAG 拓扑序唯一？给出充要条件。

**🟡 中级**：
3. 证明：若有向图 $G$ 的 DFS 中出现后向边，则 $G$ 有环。（提示：利用「灰色节点」的定义）
4. 编写 Kahn 算法的迭代版本，使其**同时输出所有合法拓扑序**（提示：回溯+队列状态复制）。
5. 在 Tarjan SCC 中，若将后向边的更新改为 `low[u] = min(low[u], low[v])`，给出一个反例说明会产生错误结果。

**🔴 高级**：
6. 实现 2-SAT 求解器：输入 $n$ 个变量和 $m$ 个 OR 子句，输出是否有解，若有解给出一组赋值。（参考：LeetCode #1. 并使用 Tarjan SCC）
7. 「最小路径覆盖」问题：在 DAG $G$ 中，用最少数量的不相交路径覆盖所有节点，证明答案 = $n - \text{最大匹配数}$（DAG 二分图模型 + 匈牙利算法）。

**💡 思考题**：
- Kosaraju 为什么要在**转置图** $G^T$ 上做第二次 DFS，而不是继续在原图上？
- Tarjan 算法的 SCC 输出顺序是**逆拓扑序**，能否在 $O(V+E)$ 内将其转为正拓扑序？如何做？
- 无向图的「双连通分量（2-edge-connected components）」与 SCC 有什么关系和类比？

---

**参考资料**：
- CLRS 第4版 Chapter 20（SCC、拓扑排序）
- Sedgewick《算法》第4版 4.1–4.2（有向图）
- MIT 6.006 Lecture 14
- Robert Tarjan, "Depth-first search and linear graph algorithms", SIAM J. Comput., 1972
- LeetCode 题目：#207、#210、#802、#1192

<div data-component="TopologicalOrderCompare"></div>

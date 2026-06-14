# Chapter 21: 最短路径基础（Shortest Path Fundamentals）

> **学习目标**：  
> 理解「松弛操作（RELAX）」是所有最短路径算法的统一核心，能用一行伪代码描述任何最短路算法的本质；掌握 Bellman-Ford 算法的完整轮次证明与负权环检测机制；理解 SPFA 队列优化的思路与局限；能在 DAG 上利用拓扑序在 $O(V+E)$ 内求最短/最长路径，并将其应用于工程中的关键路径（CPM/PERT）分析。

---

## 21.1 最短路径问题定义

### 21.1.1 用「地图导航」理解最短路径——生活中的算法

**你打开手机地图，输入「从家到公司」，导航 App 秒速给出最优路线。** 它做了什么？

本质上，它把城市的道路网络建模为一张**带权有向图**：  
- **顶点（节点）**：路口、地点  
- **有向边**：单行道或双行道（双向分解为两条有向边）  
- **边的权值**：行驶时间、路程、油费……

然后在这张图上，找到从出发点到目的地权值之和最小的路径——这就是**最短路径问题（Shortest Path Problem）**。

> 注意：「最短」指的是**边权之和最小**，不一定是边数最少（边数最少是 BFS 解决的，只要所有边权为 1）。

---

### 21.1.2 问题的两种形式：SSSP vs APSP

最短路径有两大经典问法：

| 问题类型 | 英文全称 | 中文 | 提问方式 | 代表算法 |
|---|---|---|---|---|
| **SSSP** | Single-Source Shortest Path | 单源最短路径 | 从**一个**出发点 $s$ 到所有其他顶点的最短距离 | Dijkstra、Bellman-Ford |
| **APSP** | All-Pairs Shortest Path | 全对最短路径 | 图中**所有顶点对**之间的最短距离 | Floyd-Warshall、Johnson |

**本章专注 SSSP（单源最短路径）**。

我们定义：
- $s$：源点（起始顶点）
- $d[v]$（也写作 $\text{dist}[v]$）：算法运行中对「$s$ 到 $v$ 最短距离」的**当前估计值**，初始时 $d[s] = 0$，$d[v] = +\infty$（其余节点）
- $\delta(s, v)$：$s$ 到 $v$ 的**真实最短路径长度**（算法目标，最终 $d[v] = \delta(s, v)$）
- $\pi[v]$：$v$ 的前驱节点（用于还原路径）

**初始化模板**（所有最短路算法都用此初始化）：

```
INITIALIZE-SINGLE-SOURCE(G, s):
    for each v ∈ G.V:
        d[v] = ∞
        π[v] = NIL
    d[s] = 0
```

---

### 21.1.3 最短路径的「灵魂操作」——松弛（RELAX）

**这是本章最重要的概念，请反复理解。**

> **松弛操作（Relax / Edge Relaxation）**：对于一条边 $(u, v)$，权值为 $w(u,v)$，如果当前估计 $d[v] > d[u] + w(u,v)$，就更新 $d[v] = d[u] + w(u,v)$，并记录 $\pi[v] = u$。

**生活比喻——「绕路更近」**：想象你知道从家（$s$）到超市（$u$）的距离是 3 公里，而从超市到学校（$v$）只有 1 公里。那么「家 → 超市 → 学校」这条路只要 4 公里。如果你之前以为家到学校要 7 公里，那现在你「松弛」了这个估计——更新为 4 公里。

```
RELAX(u, v, w):
    if d[v] > d[u] + w(u, v):
        d[v] = d[u] + w(u, v)
        π[v] = u
```

**这一行判断 + 更新，就是所有最短路算法的唯一核心操作。** 不同算法的差别仅在于：**以什么顺序、对哪些边、调用多少次 RELAX**。

<div data-component="RelaxOperationDemo"></div>

**松弛操作的两个关键性质**（CLRS 引理）：

1. **三角不等式**：对所有边 $(u,v) \in E$，有 $\delta(s,v) \leq \delta(s,u) + w(u,v)$。真实最短路不可能「绕远路」。

2. **上界性质（Upper-bound property）**：松弛操作执行的과정中，$d[v]$ 始终满足 $d[v] \geq \delta(s,v)$；一旦 $d[v] = \delta(s,v)$，后续松弛不会再改变它。

---

### 21.1.4 最短路径的最优子结构——「最优路径的子路径也是最优的」

这是动态规划的灵魂性质，最短路径也满足这一点，使得我们可以用 DP 思想求解。

> **最优子结构定理**：设 $p = v_0 \to v_1 \to \cdots \to v_k$ 是从 $v_0$ 到 $v_k$ 的一条最短路径，则对任意 $0 \leq i \leq j \leq k$，子路径 $p_{ij} = v_i \to v_{i+1} \to \cdots \to v_j$ 也是从 $v_i$ 到 $v_j$ 的最短路径。

**证明（反证法）**：若存在一条从 $v_i$ 到 $v_j$ 的路径 $p'_{ij}$ 比 $p_{ij}$ 更短，则将 $p$ 中的 $p_{ij}$ 替换为 $p'_{ij}$，可得一条从 $v_0$ 到 $v_k$ 更短的路径，与 $p$ 是最短路径矛盾。∎

---

### 21.1.5 负权边与负权环——小心陷阱！

**负权边（Negative-weight edge）合法吗？** 合法！现实中有很多场景会产生负权边：
- 航班积分、现金返还（经过某路段「负代价」）
- 货币汇率套利（对数变换后权值为负）

**负权边之所以麻烦，是因为它会让某些贪心策略失效**（这是为什么 Dijkstra 不能处理负权边，留到第 22 章详解）。

**负权环（Negative-weight cycle）是真正的噩梦**：若路径上存在一个环 $v_0 \to v_1 \to \cdots \to v_k \to v_0$，且环的总权值 $< 0$，那么你每绕一圈，路径总长就减少一次——可以无限绕下去，路径长度趋向 $-\infty$！

```
例子：
  A ──(1)──→ B
  ↑          │
  └──(-3)────┘

A→B 权值 1，B→A 权值 -3，环的总权值 = 1 + (-3) = -2 < 0
这是一个负权环！从 A 出发，A→B→A→B→... 路径代价 -∞
```

> **结论**：当图中包含**目标顶点可达的负权环**时，最短路径**无意义**（不存在定义良好的最短路径）。所有正确的最短路算法都需要检测这种情况。

<div data-component="NegativeCycleDetect"></div>

---

### 21.1.6 收敛性质（Convergence Property）

> 若 $s \to u \to v$ 是一条最短路径，且在某次 RELAX(u, v, w) 调用之前，$d[u] = \delta(s, u)$ 已经收敛到真实值，那么 RELAX 之后，$d[v] = \delta(s, v)$ 也会永久收敛。

这个性质告诉我们：**只要我们能保证按照正确的顺序松弛，每条边只需要松弛一次就够**。Dijkstra 正是利用这个性质（先松弛「已经确认最短的节点」的出边）将复杂度降低到 $O((V+E)\log V)$。而 Bellman-Ford 退而求其次，采用「不管顺序，全部边松弛 $V-1$ 次」的暴力保证正确性。

---

## 21.2 Bellman-Ford 算法

### 21.2.1 算法核心思想——「暴力松弛，轮次保证」

**先抛出一个问题**：如何在可能有负权边的图上找最短路？

Dijkstra（下一章）的贪心策略在负权边下会失效。Bellman-Ford 采用了一种更朴素但更鲁棒的策略：

> **核心思想**：最短路径最多经过 $V-1$ 条边（因为在无负权环的简单路径中，最多经过 $V$ 个顶点、$V-1$ 条边）。那么，如果我们把所有边松弛 $V-1$ 轮，每轮松弛所有 $E$ 条边，就**一定能找到所有最短路径**。

**完整算法伪代码**：

```
BELLMAN-FORD(G, w, s):
    INITIALIZE-SINGLE-SOURCE(G, s)          // d[s]=0, d[其他]=∞
    
    for i = 1 to |V| - 1:                   // 执行 V-1 轮
        for each edge (u, v) ∈ G.E:         // 遍历所有边
            RELAX(u, v, w)                   // 尝试通过 u 更新 v
    
    // 第 V 轮：负权环检测
    for each edge (u, v) ∈ G.E:
        if d[v] > d[u] + w(u, v):
            return FALSE                     // 存在负权环！
    
    return TRUE                              // 无负权环，d[] 即为最短路
```

**手工追踪示例**：

```
图结构（有向加权图）：
  顶点：A, B, C, D, E
  边：A→B(6), A→D(7), B→C(5), B→D(8), B→E(-4), 
      C→B(-2), D→E(9), D→C(-3), E→A(2), E→C(7)
  源点：A

初始化：
  d[A]=0, d[B]=∞, d[C]=∞, d[D]=∞, d[E]=∞

第 1 轮松弛（结束时至多确认「经 1 条边」的最短路）：
  松弛 A→B(6)：d[B] > d[A]+6 = 6  → d[B] = 6
  松弛 A→D(7)：d[D] > d[A]+7 = 7  → d[D] = 7
  松弛 B→C(5)：d[C] > d[B]+5 = 11 → d[C] = 11
  ... （其余类推）

第 2 轮（确认「经 ≤2 条边」的最短路）：
  ...

第 4 轮（V-1=4 轮后，结果收敛）：
  d[A]=0, d[B]=2, d[C]=4, d[D]=7, d[E]=-2
```

---

### 21.2.2 正确性证明——为什么 V-1 轮就够？

**定理**：对于不含负权环的图，BELLMAN-FORD 执行完 $V-1$ 轮后，对所有顶点 $v$ 有 $d[v] = \delta(s, v)$。

**直觉理解**：  
- 最短路径是一条简单路径（无重复顶点），最多经过 $V$ 个顶点，即最多 $V-1$ 条边。
- 设最短路径为 $s = v_0 \to v_1 \to v_2 \to \cdots \to v_k$（$k \leq V-1$）。
- **第 1 轮**松弛后，$d[v_1]$ 一定 = $\delta(s, v_1)$（因为第 1 轮会松弛边 $v_0 \to v_1$）。
- **第 2 轮**松弛后，$d[v_2] = \delta(s, v_2)$（因为第 2 轮会松弛 $v_1 \to v_2$，且 $d[v_1]$ 已是真实值）。
- **归纳得**：第 $k$ 轮结束后，$d[v_k] = \delta(s, v_k)$。
- 因为 $k \leq V-1$，所以 $V-1$ 轮必定足够。

> 严格地说：每一轮松弛**所有边**，相当于把「使用 ≤ i 条边的最短路」从 i-1 扩展到 i——这是一个归纳论证，与 DP 的状态转移完全类似！

---

### 21.2.3 负权环检测——第 V 轮仍能松弛

如果执行完 $V-1$ 轮之后，再做**第 $V$ 轮**时发现仍有边 $(u,v)$ 满足 $d[v] > d[u] + w(u,v)$，说明什么？

> **结论**：图中存在从源点 $s$ 可达的**负权环**！

**直觉**：若无负权环，$V-1$ 轮后所有距离已经收敛（真实最短路最多 $V-1$ 条边）。若第 $V$ 轮还能继续「更新」某些节点，意味着可以找到「经 $V$ 条边更短的路」——而长度为 $V$ 的路必然有重复顶点（环），且该环是负权环（否则绕环不会更短）。

**如何还原负权环**：在第 $V$ 轮被更新的节点 $v$，顺着 $\pi[v] \to \pi[\pi[v]] \to \cdots$ 追踪前驱，直到遇到重复节点，该重复节点在环上。

<div data-component="BellmanFordRelaxation"></div>

---

### 21.2.4 时间与空间复杂度

| 指标 | 值 | 说明 |
|---|---|---|
| **时间复杂度** | $O(VE)$ | 外层 $V-1$ 轮 × 内层遍历 $E$ 条边 |
| **空间复杂度** | $O(V)$ | 只需 $d[]$、$\pi[]$ 数组（排除输入图本身） |
| **适用场景** | 负权边图、图中可能存在负权环（需检测） | 没有负权边时优先用 Dijkstra（$O((V+E)\log V)$） |

**与 Dijkstra 的比较**（先剧透，第 22 章详解）：

| 特性 | Bellman-Ford | Dijkstra |
|---|---|---|
| 时间复杂度 | $O(VE)$ | $O((V+E)\log V)$（堆） |
| 负权边支持 | ✅ 支持 | ❌ 不支持 |
| 负权环检测 | ✅ 支持 | ❌ 不支持 |
| 实现复杂度 | ⭐ 简单 | ⭐⭐ 稍复杂 |

---

### 21.2.5 代码实现——Python & C++

```python
# ═══════════════════════════════════════════════════════════════
# Bellman-Ford 最短路径算法 - Python 实现
# 时间复杂度：O(V × E)  空间复杂度：O(V)
# ═══════════════════════════════════════════════════════════════
from typing import List, Dict, Tuple, Optional

def bellman_ford(
    n: int,                          # 顶点数（顶点编号 0 ~ n-1）
    edges: List[Tuple[int,int,int]], # 边列表：(u, v, weight)
    src: int                         # 源点
) -> Tuple[Optional[List[float]], Optional[List[int]]]:
    """
    执行 Bellman-Ford 算法。
    返回 (dist, prev)：
        dist[v] = 源点到 v 的最短距离（若含负权环则返回 None）
        prev[v] = 最短路径中 v 的前驱节点（无则 -1）
    """
    INF = float('inf')
    
    # ── 初始化 ──────────────────────────────────────────────────
    dist = [INF] * n          # dist[v] = 当前对 δ(src, v) 的估计
    prev = [-1] * n           # prev[v] = 最短路径中 v 的前驱
    dist[src] = 0             # 源点到自身距离为 0
    
    # ── V-1 轮松弛 ──────────────────────────────────────────────
    for round_i in range(n - 1):        # 共 n-1 = V-1 轮
        updated = False                  # 优化：本轮若无更新则提前退出
        for u, v, w in edges:           # 遍历所有边
            # ⚠️ 边界条件：若 dist[u] 仍为 INF，跳过
            # （u 从源点不可达时，不应通过 u 松弛 v）
            if dist[u] != INF and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w   # RELAX 操作核心
                prev[v] = u
                updated = True
        if not updated:                  # 提前退出优化
            break
    
    # ── 第 V 轮：检测负权环 ───────────────────────────────────
    # 若还能松弛，说明有可达负权环
    negative_cycle_nodes = []
    for u, v, w in edges:
        if dist[u] != INF and dist[u] + w < dist[v]:
            negative_cycle_nodes.append(v)
    
    if negative_cycle_nodes:
        # 存在负权环，最短路径无意义
        return None, None
    
    return dist, prev


def reconstruct_path(prev: List[int], src: int, dst: int) -> List[int]:
    """沿前驱数组还原最短路径（从 src 到 dst）"""
    path = []
    cur = dst
    while cur != -1:
        path.append(cur)
        if cur == src:
            break
        cur = prev[cur]
        if len(path) > len(prev):    # 防止无限循环（路径不存在时）
            return []
    return path[::-1] if path and path[-1] == src else []


# ── 使用示例 ────────────────────────────────────────────────────
if __name__ == "__main__":
    # 经典教材例子（CLRS 图 22.4）
    # 顶点: 0=s, 1=t, 2=x, 3=y, 4=z
    n = 5
    edges = [
        (0, 1, 6),   # s→t
        (0, 3, 7),   # s→y
        (1, 2, 5),   # t→x
        (1, 3, 8),   # t→y
        (1, 4, -4),  # t→z ← 负权边！
        (2, 1, -2),  # x→t ← 负权边！
        (3, 2, -3),  # y→x ← 负权边！
        (3, 4, 9),   # y→z
        (4, 0, 2),   # z→s
        (4, 2, 7),   # z→x
    ]
    
    dist, prev = bellman_ford(n, edges, src=0)
    
    if dist is None:
        print("❌ 检测到负权环！最短路径不存在。")
    else:
        node_name = ['s', 't', 'x', 'y', 'z']
        print("最短路径结果（源点 s）：")
        for v in range(n):
            path = reconstruct_path(prev, 0, v)
            path_str = " → ".join(node_name[x] for x in path)
            print(f"  s → {node_name[v]}: 距离 = {dist[v]:4}, 路径 = {path_str}")

# 预期输出：
#   s → s: 距离 =    0, 路径 = s
#   s → t: 距离 =    2, 路径 = s → y → x → t
#   s → x: 距离 =    4, 路径 = s → y → x
#   s → y: 距离 =    7, 路径 = s → y
#   s → z: 距离 =   -2, 路径 = s → t → z   （实际为 s→y→x→t→z）
```

```cpp
// ═══════════════════════════════════════════════════════════════
// Bellman-Ford 最短路径算法 - C++ 实现
// 时间复杂度：O(V × E)  空间复杂度：O(V)
// ═══════════════════════════════════════════════════════════════
#include <bits/stdc++.h>
using namespace std;

const long long INF = 1e18;  // 使用 long long 防止加法溢出

struct Edge {
    int u, v;
    long long w;
};

/**
 * Bellman-Ford 最短路径算法
 * @param n     顶点数（编号 0 ~ n-1）
 * @param edges 边列表
 * @param src   源点
 * @param dist  输出：dist[v] = 源点到 v 的最短距离
 * @param prev  输出：prev[v] = 路径中 v 的前驱（-1 表示不可达或无前驱）
 * @return      true = 无负权环；false = 检测到可达负权环
 */
bool bellmanFord(
    int n,
    const vector<Edge>& edges,
    int src,
    vector<long long>& dist,
    vector<int>& prev
) {
    // ── 初始化 ─────────────────────────────────────────────────
    dist.assign(n, INF);
    prev.assign(n, -1);
    dist[src] = 0;

    // ── V-1 轮松弛 ─────────────────────────────────────────────
    for (int round = 0; round < n - 1; ++round) {
        bool updated = false;
        for (const auto& [u, v, w] : edges) {
            // ⚠️ 边界：u 不可达时跳过，防止 INF + w 溢出
            if (dist[u] == INF) continue;
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;   // RELAX
                prev[v] = u;
                updated = true;
            }
        }
        if (!updated) break;             // 提前退出优化
    }

    // ── 第 V 轮：检测负权环 ────────────────────────────────────
    for (const auto& [u, v, w] : edges) {
        if (dist[u] == INF) continue;
        if (dist[u] + w < dist[v]) {
            // 还能松弛 → 存在可达负权环
            return false;
        }
    }
    return true;  // 无负权环
}

/** 还原最短路径（从 src 到 dst）*/
vector<int> reconstructPath(
    const vector<int>& prev, int src, int dst
) {
    vector<int> path;
    for (int cur = dst; cur != -1; cur = prev[cur]) {
        path.push_back(cur);
        if (cur == src) break;
        // ⚠️ 防止无路径时的死循环
        if ((int)path.size() > (int)prev.size()) return {};
    }
    if (path.empty() || path.back() != src) return {};
    reverse(path.begin(), path.end());
    return path;
}

int main() {
    // CLRS 教材例子（图 22.4）
    int n = 5;  // s=0, t=1, x=2, y=3, z=4
    vector<Edge> edges = {
        {0, 1,  6},   // s→t
        {0, 3,  7},   // s→y
        {1, 2,  5},   // t→x
        {1, 3,  8},   // t→y
        {1, 4, -4},   // t→z ← 负权边！
        {2, 1, -2},   // x→t ← 负权边！
        {3, 2, -3},   // y→x ← 负权边！
        {3, 4,  9},   // y→z
        {4, 0,  2},   // z→s
        {4, 2,  7},   // z→x
    };

    vector<long long> dist;
    vector<int> prev;
    string name[] = {"s", "t", "x", "y", "z"};

    if (!bellmanFord(n, edges, 0, dist, prev)) {
        cout << "❌ 检测到可达负权环！最短路径不存在。\n";
        return 1;
    }

    cout << "最短路径结果（源点 s）：\n";
    for (int v = 0; v < n; ++v) {
        auto path = reconstructPath(prev, 0, v);
        cout << "  s → " << name[v] << ": 距离 = " << setw(4) << dist[v] << "，路径 = ";
        for (int i = 0; i < (int)path.size(); ++i) {
            if (i) cout << " → ";
            cout << name[path[i]];
        }
        cout << "\n";
    }
    return 0;
}
```

---

### 21.2.6 Bellman-Ford 的 DP 视角——状态转移方程

**思考题**：Bellman-Ford 和 DP 有什么本质联系？

其实，Bellman-Ford 正是一种 DP！

定义状态：$dp[k][v]$ = 从源点 $s$ 出发，**恰好经过 ≤ k 条边**到达 $v$ 的最短距离。

**状态转移（松弛=DP转移）**：

$$dp[k][v] = \min\left( dp[k-1][v],\ \min_{(u,v)\in E} \{ dp[k-1][u] + w(u,v) \} \right)$$

**边界条件**：$dp[0][s] = 0$，$dp[0][v] = +\infty$（$v \neq s$）

**目标**：$dp[V-1][v] = \delta(s, v)$（最终答案）

Bellman-Ford 的每一轮，正是对 $k$ 从 $1$ 到 $V-1$ 的逐步展开，只不过实际实现用**原地更新**而非二维数组（节省空间）。

---

### 21.2.7 SPFA——队列优化的 Bellman-Ford

**动机**：Bellman-Ford 每轮遍历所有 $E$ 条边，但实际上，只有 $d[u]$ 在上一轮**被更新过**的节点 $u$，其出边才有松弛的可能。大多数情况下，每轮被更新的节点数远小于 $V$。

**SPFA（Shortest Path Faster Algorithm）** 就是利用**队列**来跟踪「哪些节点上轮被更新」，只对这些节点的出边做松弛：

```
SPFA(G, w, s):
    INITIALIZE-SINGLE-SOURCE(G, s)
    queue Q
    Q.enqueue(s)
    in_queue[s] = true         // 标记是否在队列中（防止重复加入）
    
    while Q 不为空:
        u = Q.dequeue()
        in_queue[u] = false
        for each (u, v) ∈ G.adj[u]:
            if RELAX(u, v, w) 成功更新了 d[v]:
                if not in_queue[v]:
                    Q.enqueue(v)
                    in_queue[v] = true
```

**SPFA 复杂度分析**：

| 场景 | 复杂度 | 说明 |
|---|---|---|
| **平均情况** | $O(kE)$（$k \approx 2 \sim 3$） | 稀疏图、随机权值时极快 |
| **最坏情况** | $O(VE)$ | 构造特殊图可使其退化（不保证优于 Bellman-Ford） |

> **重要提醒**：SPFA 在竞赛中一度很流行，但其最坏情况与 Bellman-Ford 一致。现代算法题通常会有专门 hack SPFA 的数据（卡 SPFA 的技巧是构造 SLF 退化图），因此**生产环境请用 Dijkstra（正权图）或 Bellman-Ford（负权图，正确性优先）**，而非 SPFA。

**SPFA 同样能检测负权环**：若某个节点被入队超过 $V$ 次，则存在负权环。

<div data-component="SPFAQueueTrace"></div>

#### SPFA 代码实现

```python
# ═══════════════════════════════════════════════════════════════
# SPFA（队列优化 Bellman-Ford）- Python 实现
# 平均 O(kE)，最坏 O(VE)
# ═══════════════════════════════════════════════════════════════
from collections import deque
from typing import List, Tuple, Optional

def spfa(
    n: int,
    adj: List[List[Tuple[int, int]]],  # adj[u] = [(v, w), ...]
    src: int
) -> Optional[List[float]]:
    """
    SPFA 单源最短路。
    返回 dist 列表；若检测到负权环则返回 None。
    """
    INF = float('inf')
    dist = [INF] * n
    dist[src] = 0
    
    in_queue = [False] * n
    enqueue_count = [0] * n   # 记录每个节点入队次数（用于负权环检测）
    
    q = deque([src])
    in_queue[src] = True
    enqueue_count[src] = 1
    
    while q:
        u = q.popleft()
        in_queue[u] = False
        
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if not in_queue[v]:
                    q.append(v)
                    in_queue[v] = True
                    enqueue_count[v] += 1
                    # ⚠️ 负权环检测：若某节点入队超过 n 次
                    if enqueue_count[v] >= n:
                        return None   # 负权环存在！
    
    return dist


# 使用示例
if __name__ == "__main__":
    n = 5
    # 构建邻接表（同 Bellman-Ford 例子）
    adj: List[List[Tuple[int,int]]] = [[] for _ in range(n)]
    edge_list = [
        (0,1,6),(0,3,7),(1,2,5),(1,3,8),(1,4,-4),
        (2,1,-2),(3,2,-3),(3,4,9),(4,0,2),(4,2,7),
    ]
    for u, v, w in edge_list:
        adj[u].append((v, w))
    
    result = spfa(n, adj, src=0)
    if result is None:
        print("❌ 负权环检测！")
    else:
        names = ['s','t','x','y','z']
        for v in range(n):
            print(f"  dist[{names[v]}] = {result[v]}")
```

```cpp
// ═══════════════════════════════════════════════════════════════
// SPFA（队列优化 Bellman-Ford）- C++ 实现
// ═══════════════════════════════════════════════════════════════
#include <bits/stdc++.h>
using namespace std;

const long long INF = 1e18;

/**
 * SPFA 单源最短路。
 * @param n   顶点数（编号 0~n-1）
 * @param adj 邻接表：adj[u] = {(v, w), ...}
 * @param src 源点
 * @param dist 输出：最短距离数组
 * @return true=无负权环，false=存在可达负权环
 */
bool spfa(
    int n,
    const vector<vector<pair<int,long long>>>& adj,
    int src,
    vector<long long>& dist
) {
    dist.assign(n, INF);
    dist[src] = 0;

    vector<bool> inQueue(n, false);
    vector<int> enqueueCnt(n, 0);  // 负权环检测计数

    queue<int> q;
    q.push(src);
    inQueue[src] = true;
    enqueueCnt[src] = 1;

    while (!q.empty()) {
        int u = q.front(); q.pop();
        inQueue[u] = false;

        for (auto& [v, w] : adj[u]) {
            if (dist[u] == INF) continue;   // u 不可达则跳过
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                if (!inQueue[v]) {
                    q.push(v);
                    inQueue[v] = true;
                    // ⚠️ 某节点入队 >= n 次 → 负权环
                    if (++enqueueCnt[v] >= n) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

int main() {
    int n = 5;
    vector<vector<pair<int,long long>>> adj(n);
    vector<tuple<int,int,long long>> edges = {
        {0,1,6},{0,3,7},{1,2,5},{1,3,8},{1,4,-4},
        {2,1,-2},{3,2,-3},{3,4,9},{4,0,2},{4,2,7}
    };
    for (auto& [u,v,w] : edges) adj[u].push_back({v, w});

    vector<long long> dist;
    string name[] = {"s","t","x","y","z"};

    if (!spfa(n, adj, 0, dist)) {
        cout << "❌ 检测到可达负权环！\n";
        return 1;
    }
    for (int v = 0; v < n; ++v)
        cout << "dist[" << name[v] << "] = " << dist[v] << "\n";
    return 0;
}
```

---

### 21.2.8 竞赛常见应用与变体

**可以用 Bellman-Ford / SPFA 解决的经典问题**：

| 场景 | 建模方式 |
|---|---|
| 含负权边的单源最短路 | 直接套用 Bellman-Ford |
| 套汇检测（Currency Arbitrage） | 将汇率取对数后转化，负权环 = 套利机会 |
| 限制边数的最短路（经过恰好 k 条边） | Bellman-Ford DP 第 k 轮的 $dp[k][v]$ |
| 差分约束系统（Difference Constraint） | 将约束 $x_j - x_i \leq w_{ij}$ 转为边 $i \to j$，权 $w_{ij}$，求最短路 |

**LeetCode 典型题目**：
- [#743 网络延迟时间](https://leetcode.cn/problems/network-delay-time/)（Bellman-Ford 或 Dijkstra）
- [#787 K 站中转内最便宜的航班](https://leetcode.cn/problems/cheapest-flights-within-k-stops/)（Bellman-Ford 限定轮次）

---

## 21.3 DAG 上的最短路径

### 21.3.1 生活比喻——「装配流水线」的最优路线

**场景**：工厂的装配流水线是一个有向无环图（DAG）——你必须先完成工序 A，才能进行工序 B。每条「流水线路段」有一个时间/成本权值（可以是负数，比如某道工序利用了上一道的流水节省了时间）。求从投料口到成品口的最短（最省时）路径。

关键洞察：**DAG 保证不存在环路**，所以我们可以在图上做 DP——按照**拓扑顺序**处理每个节点，就能保证处理节点 $v$ 时，所有前驱 $u$（即 $u \to v$ 的 $u$）已处理完毕。

---

### 21.3.2 DAG 最短路算法——拓扑序松弛

**算法思路**（比 Bellman-Ford 快得多！）：

1. 对 DAG 做拓扑排序，得到顶点的拓扑序
2. 初始化 $d[s] = 0$，其余 $d[v] = +\infty$
3. **按拓扑序逐一处理每个顶点 $u$**：对 $u$ 的每条出边 $(u, v)$，执行 RELAX(u, v, w)

**伪代码**：

```
DAG-SHORTEST-PATH(G, w, s):
    topological_order = TOPOLOGICAL-SORT(G)   // O(V+E)
    INITIALIZE-SINGLE-SOURCE(G, s)             // d[s]=0, rest ∞
    for each u in topological_order:           // 按拓扑序逐一处理
        for each (u, v) ∈ G.adj[u]:
            RELAX(u, v, w)
```

**时间复杂度**：$O(V+E)$（拓扑排序 $O(V+E)$ + 每条边恰好松弛一次 $O(E)$）。

**为什么每条边只需松弛一次就够？**  
因为拓扑序保证了：当我们处理 $u$ 时，$d[u]$ 已经是 $\delta(s,u)$（真实最短距离）——因为所有能更新 $d[u]$ 的前驱都已处理过。所以对 $u$ 的出边松弛一次就是最终结果！

**手工追踪示例**：

```
DAG（有向无环图）：
  r  s  x  y  z    ← 顶点
  拓扑序: r → s → x → y → z

边及权值:
  r→x(3), r→s(5)
  s→x(2), s→y(6)
  x→y(7), x→z(4), x→z(4)（取其一）
  y→z(−1)

源点 s，初始化：
  d[r]=∞, d[s]=0, d[x]=∞, d[y]=∞, d[z]=∞

拓扑序处理 r（d[r]=∞，从 s 不可达，松弛无效）
拓扑序处理 s（d[s]=0）：
  松弛 s→x(2): d[x] = min(∞, 0+2) = 2
  松弛 s→y(6): d[y] = min(∞, 0+6) = 6
拓扑序处理 x（d[x]=2）：
  松弛 x→y(7): d[y] = min(6, 2+7) = 6 （未更新）
  松弛 x→z(4): d[z] = min(∞, 2+4) = 6
拓扑序处理 y（d[y]=6）：
  松弛 y→z(-1): d[z] = min(6, 6+(-1)) = 5 ← 利用了负权边！
拓扑序处理 z（叶节点，无出边）

最终结果：d[s]=0, d[x]=2, d[y]=6, d[z]=5, d[r]=∞（不可达）
```

<div data-component="DAGShortestPath"></div>

---

### 21.3.3 代码实现——Python & C++

```python
# ═══════════════════════════════════════════════════════════════
# DAG 最短路径（拓扑序松弛）- Python 实现
# 时间复杂度：O(V + E)   空间复杂度：O(V)
# ⚠️ 仅适用于 DAG（有向无环图）
# ═══════════════════════════════════════════════════════════════
from typing import List, Tuple
from collections import deque

def dag_shortest_path(
    n: int,
    adj: List[List[Tuple[int, int]]],  # adj[u] = [(v, w), ...]
    src: int
) -> List[float]:
    """
    在 DAG 上用拓扑序松弛求单源最短路。
    若 src 不可达某节点 v，则 dist[v] = float('inf')。
    """
    INF = float('inf')
    
    # ── Step 1: Kahn 算法求拓扑序 ────────────────────────────────
    indegree = [0] * n
    for u in range(n):
        for v, _ in adj[u]:
            indegree[v] += 1
    
    queue = deque(v for v in range(n) if indegree[v] == 0)
    topo_order = []
    
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v, _ in adj[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)
    
    # 若 topo_order 不包含所有节点，则图有环（理论上 DAG 不会触发）
    assert len(topo_order) == n, "输入的图不是 DAG！"
    
    # ── Step 2: 初始化 ────────────────────────────────────────────
    dist = [INF] * n
    dist[src] = 0
    
    # ── Step 3: 按拓扑序松弛 ─────────────────────────────────────
    for u in topo_order:
        if dist[u] == INF:
            continue    # u 从 src 不可达，其出边松弛无意义
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w   # RELAX
    
    return dist


# 使用示例
if __name__ == "__main__":
    # r=0, s=1, x=2, y=3, z=4
    n = 5
    adj = [[] for _ in range(n)]
    edges = [(0,2,3),(0,1,5),(1,2,2),(1,3,6),(2,3,7),(2,4,4),(3,4,-1)]
    for u, v, w in edges:
        adj[u].append((v, w))
    
    dist = dag_shortest_path(n, adj, src=1)  # 以 s(=1) 为源点
    names = ['r','s','x','y','z']
    print("DAG 最短路径（源点 s）：")
    for v in range(n):
        d = dist[v] if dist[v] != float('inf') else "∞（不可达）"
        print(f"  s → {names[v]}: {d}")

# 预期输出：
#   s → r: ∞（不可达）
#   s → s: 0
#   s → x: 2
#   s → y: 6
#   s → z: 5
```

```cpp
// ═══════════════════════════════════════════════════════════════
// DAG 最短路径（拓扑序松弛）- C++ 实现
// 时间复杂度：O(V + E)
// ═══════════════════════════════════════════════════════════════
#include <bits/stdc++.h>
using namespace std;

const long long INF = 1e18;

/**
 * 在 DAG 上用拓扑序松弛求单源最短路。
 * @param n   顶点数（编号 0~n-1）
 * @param adj 邻接表：adj[u] = {(v, w), ...}
 * @param src 源点
 * @return    dist[v] = 源点到 v 的最短距离（不可达时为 INF）
 */
vector<long long> dagShortestPath(
    int n,
    const vector<vector<pair<int,long long>>>& adj,
    int src
) {
    // ── Step 1: Kahn 算法求拓扑序 ────────────────────────────────
    vector<int> indegree(n, 0);
    for (int u = 0; u < n; ++u)
        for (auto& [v, w] : adj[u])
            ++indegree[v];

    queue<int> q;
    for (int v = 0; v < n; ++v)
        if (indegree[v] == 0) q.push(v);

    vector<int> topoOrder;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        topoOrder.push_back(u);
        for (auto& [v, w] : adj[u])
            if (--indegree[v] == 0) q.push(v);
    }

    // ⚠️ 若 topoOrder.size() != n，则输入图有环
    assert((int)topoOrder.size() == n);

    // ── Step 2: 初始化 ────────────────────────────────────────────
    vector<long long> dist(n, INF);
    dist[src] = 0;

    // ── Step 3: 按拓扑序松弛 ─────────────────────────────────────
    for (int u : topoOrder) {
        if (dist[u] == INF) continue;   // u 不可达，跳过
        for (auto& [v, w] : adj[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;  // RELAX
            }
        }
    }
    return dist;
}

int main() {
    // r=0, s=1, x=2, y=3, z=4
    int n = 5;
    vector<vector<pair<int,long long>>> adj(n);
    vector<tuple<int,int,long long>> edges = {
        {0,2,3},{0,1,5},{1,2,2},{1,3,6},{2,3,7},{2,4,4},{3,4,-1}
    };
    for (auto& [u,v,w] : edges) adj[u].push_back({v, w});

    auto dist = dagShortestPath(n, adj, 1);  // 源点 s=1
    string name[] = {"r","s","x","y","z"};
    cout << "DAG 最短路径（源点 s）：\n";
    for (int v = 0; v < n; ++v) {
        cout << "  s → " << name[v] << ": ";
        if (dist[v] == INF) cout << "∞（不可达）\n";
        else cout << dist[v] << "\n";
    }
    return 0;
}
```

---

### 21.3.4 DAG 最长路径——「翻转权值」的优雅技巧

**思路**：将所有边权**取反**（$w \to -w$），然后求最短路径，得到的结果取反即为最长路径。

$$\text{最长路径} = -\text{在 }(-w)\text{ 权值图上的最短路径}$$

**为什么这个技巧能奏效？**  
- 一般图上，最长路径是 NP-Hard 问题（等价于哈密顿路径）。  
- 但 **DAG 上不存在正权环**，取负后不存在负权环，DAG 最短路算法依然正确。  
- DAG 上最长路径可以在 $O(V+E)$ 内解决。

---

### 21.3.5 PERT / CPM——工程项目中的关键路径

**背景**：大型工程（建筑、软件开发、电影制作）中，每个「工序」有依赖关系和完成时间。整个项目的最短完工时间由**最长路径**（关键路径）决定。

**CPM（Critical Path Method）关键路径法**：
- 建图：每个工序是一个节点，依赖关系是有向边，边权为工序时长
- 求 DAG 上的**最长路径**（即对权值取负后求最短路）
- 最长路径上的所有工序 = **关键工序**（任何一个延误都会导致整个项目延误）
- 关键路径的长度 = 项目最短完工时间

**最早/最晚完成时间**：
- $ES[v]$（Earliest Start）= 从起点到 $v$ 的最长路径 = $v$ 最早能开始的时间
- $LS[v]$（Latest Start）= 总工期 $-$ 从 $v$ 到终点的最长路径 = $v$ 最晚开始时间
- **浮动时间（Float/Slack）**= $LS[v] - ES[v]$：该工序可延迟多久不影响工期
- 关键工序的浮动时间为 **0**

```
典型工程 DAG 示例：
  工序 A(3天) → 工序 C(2天) → 工序 E(4天) → 竣工
  工序 A(3天) → 工序 D(5天) → 竣工
  工序 B(6天) → 工序 C(2天)
  工序 B(6天) → 工序 E(4天)

关键路径: B → D → 竣工（6+5=11天）或 B → C → E（6+2+4=12天）
项目最短完工时间 = 12 天
```

<div data-component="CriticalPathCPM"></div>

---

### 21.3.6 综合代码：关键路径计算——Python & C++

```python
# ═══════════════════════════════════════════════════════════════
# PERT/CPM 关键路径分析 - Python 实现
# 复杂度：O(V + E)
# ═══════════════════════════════════════════════════════════════
from typing import List, Tuple
from collections import deque

def critical_path(
    n: int,
    adj: List[List[Tuple[int, int]]],   # adj[u] = [(v, duration), ...]
    source: int,
    sink: int
) -> Tuple[int, List[int]]:
    """
    计算关键路径。
    返回（项目总工期, 关键路径节点列表）
    """
    INF = float('inf')
    
    # ── Step 1: 拓扑排序（Kahn） ────────────────────────────────
    indegree = [0] * n
    for u in range(n):
        for v, _ in adj[u]:
            indegree[v] += 1
    q = deque(v for v in range(n) if indegree[v] == 0)
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v, _ in adj[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                q.append(v)
    
    # ── Step 2: 前向传播——计算 ES（最早开始时间） ─────────────
    ES = [-INF] * n
    ES[source] = 0
    prev = [-1] * n           # 路径追踪
    
    for u in topo:
        for v, d in adj[u]:
            if ES[u] + d > ES[v]:
                ES[v] = ES[u] + d   # 这里是"最长路"DP
                prev[v] = u
    
    total_duration = ES[sink]
    
    # ── Step 3: 还原关键路径 ─────────────────────────────────────
    path = []
    cur = sink
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    
    return total_duration, path


if __name__ == "__main__":
    # 工序: 0=开始, 1=A, 2=B, 3=C, 4=D, 5=E, 6=竣工
    # 工序时长体现在边权上
    n = 7
    adj = [[] for _ in range(n)]
    edges = [
        (0, 1, 0), (0, 2, 0),   # 开始 → A, B（虚边，时长 0）
        (1, 3, 3), (1, 4, 3),   # A(3天) → C, D
        (2, 3, 6), (2, 5, 6),   # B(6天) → C, E
        (3, 5, 2),               # C(2天) → E
        (4, 6, 5),               # D(5天) → 竣工
        (5, 6, 4),               # E(4天) → 竣工
    ]
    for u, v, w in edges:
        adj[u].append((v, w))
    
    duration, path = critical_path(n, adj, source=0, sink=6)
    names = ["开始","A","B","C","D","E","竣工"]
    print(f"项目总工期: {duration} 天")
    print(f"关键路径: {' → '.join(names[v] for v in path)}")
```

```cpp
// ═══════════════════════════════════════════════════════════════
// PERT/CPM 关键路径分析 - C++ 实现
// 复杂度：O(V + E)
// ═══════════════════════════════════════════════════════════════
#include <bits/stdc++.h>
using namespace std;

pair<long long, vector<int>> criticalPath(
    int n,
    const vector<vector<pair<int,long long>>>& adj,
    int source, int sink
) {
    const long long NEG_INF = -1e18;

    // ── Step 1: 拓扑排序 ─────────────────────────────────────────
    vector<int> indegree(n, 0);
    for (int u = 0; u < n; ++u)
        for (auto& [v,d] : adj[u]) ++indegree[v];

    queue<int> q;
    for (int v = 0; v < n; ++v) if (!indegree[v]) q.push(v);
    vector<int> topo;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        topo.push_back(u);
        for (auto& [v,d] : adj[u]) if (!--indegree[v]) q.push(v);
    }

    // ── Step 2: 最长路 DP ─────────────────────────────────────────
    vector<long long> ES(n, NEG_INF);
    vector<int> prev(n, -1);
    ES[source] = 0;

    for (int u : topo) {
        if (ES[u] == NEG_INF) continue;
        for (auto& [v, d] : adj[u]) {
            if (ES[u] + d > ES[v]) {
                ES[v] = ES[u] + d;
                prev[v] = u;
            }
        }
    }

    // ── Step 3: 还原路径 ──────────────────────────────────────────
    vector<int> path;
    for (int cur = sink; cur != -1; cur = prev[cur])
        path.push_back(cur);
    reverse(path.begin(), path.end());

    return {ES[sink], path};
}

int main() {
    // 工序: 0=开始, 1=A, 2=B, 3=C, 4=D, 5=E, 6=竣工
    int n = 7;
    vector<vector<pair<int,long long>>> adj(n);
    vector<tuple<int,int,long long>> edges = {
        {0,1,0},{0,2,0},
        {1,3,3},{1,4,3},
        {2,3,6},{2,5,6},
        {3,5,2},
        {4,6,5},
        {5,6,4}
    };
    for (auto& [u,v,w] : edges) adj[u].push_back({v,w});
    string name[] = {"开始","A","B","C","D","E","竣工"};

    auto [duration, path] = criticalPath(n, adj, 0, 6);
    cout << "项目总工期: " << duration << " 天\n";
    cout << "关键路径: ";
    for (int i = 0; i < (int)path.size(); ++i) {
        if (i) cout << " → ";
        cout << name[path[i]];
    }
    cout << "\n";
    return 0;
}
```

---

## 21.4 本章总结与算法对比

### 21.4.1 本章三大算法对比

| 算法 | 适用图类型 | 时间复杂度 | 负权边 | 负权环检测 | 核心思想 |
|---|---|---|---|---|---|
| **Bellman-Ford** | 任意有向图 | $O(VE)$ | ✅ | ✅ | 暴力松弛 V-1 轮 |
| **SPFA** | 任意有向图 | 平均 $O(kE)$，最坏 $O(VE)$ | ✅ | ✅（入队次数） | 队列优化 B-F |
| **DAG 最短路** | 仅 DAG | $O(V+E)$ | ✅ | N/A（DAG无环） | 拓扑序松弛 |
| **Dijkstra**（预告） | 非负权有向/无向图 | $O((V+E)\log V)$ | ❌ | ❌ | 贪心 + 优先队列 |

### 21.4.2 决策树——选哪个算法？

```
图中有负权边？
├── 否 → 优先用 Dijkstra（见第 22 章）
└── 是 → 图是 DAG？
        ├── 是 → DAG 最短路（O(V+E)，最快！）
        └── 否 → 需要检测负权环？
                ├── 是 → Bellman-Ford（正确性有保证）
                └── 否 → SPFA（实践中更快，但最坏同 B-F）
```

### 21.4.3 思考题

1. **Bellman-Ford 与 DP 的联系**：请写出「从源点出发，经恰好 $k$ 条边到达 $v$ 的最短路径」的 DP 状态转移方程，并说明为什么 Bellman-Ford 的每一轮松弛恰好对应一次 DP 转移。

2. **SPFA 的最坏情况**：构造一个使 SPFA 退化到 $O(VE)$ 的图，并说明为什么在这个图上 SPFA 会遍历大量重复的节点。

3. **DAG 关键路径的应用**：在软件项目管理中，如果你发现某工序的浮动时间（Slack）为 0，意味着什么？如果你想缩短项目总工期，应该优先优化哪些工序？

4. **差分约束**：约束 $x_B - x_A \leq 5$，$x_C - x_B \leq 3$，$x_A - x_C \leq -4$ 构成一个差分约束系统。将其建模为最短路问题，并用 Bellman-Ford 判断是否有解（即不含负权环）。

---

## 21.5 面试与竞赛高频考点

### 21.5.1 高频问题精讲

**Q1：为什么 Dijkstra 不能处理负权边？**

> 答：Dijkstra 的贪心假设是「当节点 $u$ 被从优先队列中弹出时，$d[u]$ 就确定是最短距离」。这个假设基于：如果边权非负，后续节点只会让路径更长，所以已弹出的节点不可能被「绕路更新」为更短的距离。但若存在负权边，通过一条负权边绕路反而可以让路径更短，破坏了这个贪心不变量。

**反例**：

```
  A ──(5)──→ B
  A ──(2)──→ C ──(-10)──→ B

Dijkstra 会先弹出 C（d[C]=2），再弹出 B（d[B]=5）。
但通过 A→C→B：距离 = 2+(-10) = -8 < 5，Dijkstra 遗漏了这条最短路！
```

**Q2：Bellman-Ford 每轮必须遍历所有边吗？**

> 答：是的！Bellman-Ford 的正确性依赖于「每轮至少让每条边都有机会松弛一次」。若只遍历部分边，可能错过正确的松弛顺序，导致结果错误。这也是 SPFA 在最坏情况下不弱于 Bellman-Ford 的原因——它的队列保证每条边迟早会被松弛。

**Q3：如何还原 Bellman-Ford 中检测到的负权环？**

> 答：在第 $V$ 轮中，记所有仍能被松弛的节点集合 $S$。从 $S$ 中任取一节点 $v$，沿 $\pi[v]$ 不断追踪前驱，最多 $V$ 步必然重复访问某节点——该节点就在负权环上。从该节点沿前驱追踪直到再次回到该节点，即可还原环。

### 21.5.2 LeetCode 练习题推荐

| 题号 | 题目 | 知识点 | 难度 |
|---|---|---|---|
| #743 | 网络延迟时间 | SSSP（Dijkstra/Bellman-Ford） | 🟡 Medium |
| #787 | K 站中转内最便宜的航班 | Bellman-Ford 限制轮次 | 🟡 Medium |
| #1514 | 概率最大的路径 | DAG/最短路变体（取对数） | 🟡 Medium |
| #444 | 序列重建（进阶） | DAG + 拓扑排序约束 | 🔴 Hard |
| #1631 | 最小体力消耗路径 | 最短路 + 二分/Dijkstra | 🟡 Medium |

---

> **参考资料**：  
> - CLRS 第4版 Chapter 22（单源最短路径）
> - MIT 6.006 Lecture 15（Bellman-Ford）  
> - Sedgewick《算法》第4版 4.4（带权有向图、最短路径）
> - [CP-Algorithms: Bellman-Ford](https://cp-algorithms.com/graph/bellman_ford.html)
> - [VizAlgo 最短路径可视化](https://visualgo.net/en/sssp)

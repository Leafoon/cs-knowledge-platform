# Chapter 22: Dijkstra 与最小生成树

> **学习目标**：掌握 Dijkstra 贪心选择的本质与正确性证明；理解 A\* 启发式搜索如何"提前知情"地剪枝；能用切割性质证明 MST 算法的正确性；比较 Kruskal 与 Prim 的适用场景、复杂度与实现细节。

---

## 22.0 从"逐轮松弛"到"聪明的贪心"

在上一章，我们学习了 **Bellman-Ford**：它对所有 $|V|-1$ 轮每轮松弛全部 $|E|$ 条边，时间复杂度 $O(VE)$。
这个方式非常**保守**——它处理每一条边，哪怕大多数边当前轮不可能被更新。

有没有更聪明的方法？

> **关键洞见**：如果图中所有边权 $w \ge 0$，当我们第一次"确定"某个节点 $u$ 的最短路径时，这条路径就永久固定了，不可能被后来发现的新路径改进。

这就是 **Dijkstra 算法**的核心贪心思想：每次从"尚未确定"的节点中选取 $d[u]$ 最小者，宣告其最短路径已确定，然后用它去松弛邻居。

---

## 22.1 Dijkstra 算法

### 22.1.1 贪心直觉：地铁扩张比喻

> 想象你在规划地铁线路延伸方案。你从起点站 $s$ 出发，手头有一份"当前已知最短票价"列表。每一步，你都选择**票价最低的未开通车站**先开通，然后更新从该站出发能到达的所有邻站的票价。

这就是 Dijkstra：它像是一个**贪心的工程师**，永远优先处理"离起点最近"的节点。

> **为什么贪心有效？**
> 因为所有边权非负，所以一旦节点 $u$ 被选出（$d[u]$ 最小），任何经过其他未确定节点再到 $u$ 的路径，都会比当前 $d[u]$ 更长（因为路上每一条边都 $\ge 0$）。

### 22.1.2 算法伪代码（CLRS 风格）

$$
\text{DIJKSTRA}(G, w, s):
$$

1. 对所有 $v \in V$：$d[v] \leftarrow \infty$，$\pi[v] \leftarrow \text{NIL}$
2. $d[s] \leftarrow 0$
3. 优先队列 $Q \leftarrow V$（按 $d$ 值排序）
4. **while** $Q \neq \emptyset$:
   - $u \leftarrow \text{EXTRACT-MIN}(Q)$（提取 $d$ 值最小的节点）
   - **for each** 邻居 $v$ of $u$：
     - **RELAX**$(u, v, w)$：若 $d[u] + w(u,v) < d[v]$，则更新 $d[v]$ 并在 $Q$ 中 DECREASE-KEY

### 22.1.3 具体示例追踪（CLRS 经典 5 节点图）

使用以下有向图（源点 $s$）：

```
节点: s, t, x, y, z
有向边（u→v，权w）：
  s→t: 10   s→y: 5
  t→x: 1    t→y: 2
  x→z: 4
  y→t: 3    y→x: 9    y→z: 2
  z→x: 6    z→s: 7
```

**初始状态**：$d = [s:0,\ t:\infty,\ x:\infty,\ y:\infty,\ z:\infty]$，$Q = \{s,t,x,y,z\}$

| 步骤 | EXTRACT-MIN | 松弛操作 | 更新后 $d[]$ |
|------|------------|---------|------------|
| 1 | $s\ (d=0)$ | s→t: 0+10=10 < ∞ ✓; s→y: 0+5=5 < ∞ ✓ | t:10, y:5 |
| 2 | $y\ (d=5)$ | y→t: 5+3=8 < 10 ✓; y→x: 5+9=14 < ∞ ✓; y→z: 5+2=7 < ∞ ✓ | t:8, x:14, z:7 |
| 3 | $z\ (d=7)$ | z→x: 7+6=13 < 14 ✓; z→s: 7+7=14（s已确定，跳过） | x:13 |
| 4 | $t\ (d=8)$ | t→x: 8+1=9 < 13 ✓; t→y: 8+2=10（y已确定，跳过） | x:9 |
| 5 | $x\ (d=9)$ | x→z: 9+4=13（z已确定，跳过） | — |

**最终最短路径**：$d[s]=0,\ d[t]=8,\ d[x]=9,\ d[y]=5,\ d[z]=7$

<div data-component="DijkstraPriorityQueueViz"></div>

### 22.1.4 正确性证明（以归纳法为核心）

**定理**：Dijkstra 结束时，对每个节点 $u$，$d[u] = \delta(s,u)$（真实最短路径距离）。

**证明**（循环不变式）：

> **不变式**：在每次 EXTRACT-MIN 前，已确定集合 $S$ 中所有节点 $u$ 的 $d[u] = \delta(s,u)$。

**归纳基础**：第一次 EXTRACT-MIN 取出 $s$，$d[s]=0=\delta(s,s)$。✓

**归纳步骤**：设归纳假设已对 $S$ 中所有节点成立。现在 EXTRACT-MIN 取出节点 $u$（$d[u]$ 最小）。

反设 $d[u] > \delta(s,u)$，即存在更短路径 $p: s \leadsto u$。

设 $p$ 第一次离开 $S$ 时经过边 $(x, y)$（$x \in S$，$y \notin S$）：

$$\delta(s,u) = \delta(s,x) + w(x,y) + (\text{路径} y \leadsto u \text{ 的权})$$

- 由归纳假设：$d[x] = \delta(s,x)$
- 因为之前处理 $x$ 时已松弛 $(x,y)$：$d[y] \le d[x] + w(x,y) = \delta(s,x) + w(x,y)$
- 因为边权非负：$\delta(s,y) \le \delta(s,u)$
- 因此：$d[y] \le \delta(s,y) \le \delta(s,u)$

但由于 Dijkstra 选 $u$ 而非 $y$，有 $d[u] \le d[y]$，故 $d[u] \le \delta(s,u)$，与反设矛盾。✓

> ⚠️ **关键约束**：证明中用到"边权非负"保证路径 $y \leadsto u$ 不增长路径长度。如果存在负权边，此推理失效！

### 22.1.5 时间复杂度分析

| 实现方式 | EXTRACT-MIN | DECREASE-KEY | 总复杂度 |
|---------|-------------|--------------|---------|
| **朴素数组**（遍历找最小） | $O(V)$ × $V$次 | $O(1)$ × $E$次 | $O(V^2 + E) = O(V^2)$ |
| **二叉堆**（`heapq`） | $O(\log V)$ × $V$次 | $O(\log V)$ × $E$次 | $O((V+E)\log V)$ |
| **Fibonacci 堆**（理论最优） | $O(\log V)$ 均摊 × $V$次 | $O(1)$ 均摊 × $E$次 | $O(V\log V + E)$ |

**实践中**：

- **稀疏图**（$E = O(V)$）：二叉堆表现最好，$O(V\log V)$
- **稠密图**（$E = O(V^2)$）：朴素数组 $O(V^2)$ 反而优于二叉堆的 $O(V^2 \log V)$
- Fibonacci 堆常数因子大，工程中很少使用

### 22.1.6 代码实现

```python
import heapq
from collections import defaultdict
from typing import Dict, List, Tuple

def dijkstra(graph: Dict[int, List[Tuple[int, int]]], src: int, n: int) -> List[int]:
    """
    Dijkstra 最短路径算法（二叉堆实现）
    
    参数:
      graph: 邻接表，graph[u] = [(v, w), ...] 表示 u→v 权重为 w
      src:   源点
      n:     节点总数
    返回:
      dist[]: dist[v] = s 到 v 的最短距离（不可达则为 float('inf')）
    
    时间复杂度: O((V + E) log V)
    空间复杂度: O(V + E)
    """
    # 初始化：全部设为无穷大
    dist = [float('inf')] * n
    dist[src] = 0

    # 优先队列：(距离, 节点)
    # Python heapq 是最小堆，直接满足 EXTRACT-MIN 需求
    pq = [(0, src)]  # (d[s]=0, s)

    while pq:
        # EXTRACT-MIN：取出当前距离最小的节点
        d_u, u = heapq.heappop(pq)

        # 「懒删除」：若 d_u 已经过时（被更新过），跳过
        # 这是 Python heapq 无法 DECREASE-KEY 的替代方案
        if d_u > dist[u]:
            continue  # 边界条件：过期的堆项，不处理

        # 对 u 的每条出边进行松弛
        for v, w in graph[u]:
            # RELAX(u, v, w)
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))  # 推入堆（旧值成为过期项）

    return dist


# ── 示例：CLRS 5 节点图 ──────────────────────────────────
if __name__ == "__main__":
    # 构建邻接表
    g = defaultdict(list)
    edges = [
        (0, 1, 10), (0, 3, 5),   # s→t:10, s→y:5
        (1, 2, 1),  (1, 3, 2),   # t→x:1,  t→y:2
        (2, 4, 4),               # x→z:4
        (3, 1, 3),  (3, 2, 9), (3, 4, 2),  # y→t:3, y→x:9, y→z:2
        (4, 2, 6),  (4, 0, 7),   # z→x:6,  z→s:7
    ]
    for u, v, w in edges:
        g[u].append((v, w))

    dist = dijkstra(g, src=0, n=5)
    names = ['s', 't', 'x', 'y', 'z']
    for i, name in enumerate(names):
        print(f"d[{name}] = {dist[i]}")
    # 预期输出：d[s]=0, d[t]=8, d[x]=9, d[y]=5, d[z]=7
```

```cpp
#include <bits/stdc++.h>
using namespace std;

// 节点数 N，边权非负
vector<int> dijkstra(int n, vector<vector<pair<int,int>>>& adj, int src) {
    /*
     * Dijkstra 最短路径算法（二叉堆 + 懒删除）
     * adj[u] = {(v, w), ...}：u→v 权重 w
     * 返回 dist[]：dist[v] = src 到 v 的最短距离
     * 
     * 时间: O((V+E) log V)
     * 空间: O(V+E)
     */
    vector<int> dist(n, INT_MAX);
    dist[src] = 0;

    // priority_queue 默认是最大堆，用负权 or pair<int,int> + greater<>
    priority_queue<pair<int,int>,
                   vector<pair<int,int>>,
                   greater<pair<int,int>>> pq;   // 最小堆
    pq.push({0, src});  // (d[src]=0, src)

    while (!pq.empty()) {
        auto [d_u, u] = pq.top(); pq.pop();

        // 懒删除：已过期的堆项，跳过
        // 边界条件：可能同一节点被多次 push
        if (d_u > dist[u]) continue;

        for (auto [v, w] : adj[u]) {
            // RELAX(u, v, w)
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});  // 旧 dist[v] 的堆项变为过期项
            }
        }
    }
    return dist;
}

int main() {
    // CLRS 5 节点图（0=s, 1=t, 2=x, 3=y, 4=z）
    int n = 5;
    vector<vector<pair<int,int>>> adj(n);
    vector<tuple<int,int,int>> edges = {
        {0,1,10}, {0,3,5},
        {1,2,1},  {1,3,2},
        {2,4,4},
        {3,1,3},  {3,2,9}, {3,4,2},
        {4,2,6},  {4,0,7}
    };
    for (auto [u, v, w] : edges) adj[u].emplace_back(v, w);

    auto dist = dijkstra(n, adj, 0);
    string names[] = {"s","t","x","y","z"};
    for (int i = 0; i < n; i++)
        cout << "d[" << names[i] << "] = " << dist[i] << "\n";
    // 输出: d[s]=0, d[t]=8, d[x]=9, d[y]=5, d[z]=7
    return 0;
}
```

> **「懒删除」技巧**：Python 的 `heapq` 不支持 DECREASE-KEY，我们改为直接 `heappush` 新值到堆，旧值留在堆中等待出堆时检测（若 `d_u > dist[u]` 则为过期项，直接跳过）。这会导致堆内最多 $O(E)$ 个元素，但复杂度仍是 $O(E\log E) = O(E\log V)$。

---

### 22.1.7 为什么 Dijkstra 不能处理负权边？

<div data-component="DijkstraVsBellmanFord"></div>

**负权边反例**：

```
图:  s → a（权 2）
     s → b（权 4）
     b → a（权 -3）
真实最短路径: s→b→a = 4+(-3) = 1
```

Dijkstra 执行过程：
1. EXTRACT-MIN: $s$，$d[s]=0$。松弛后 $d[a]=2, d[b]=4$。
2. EXTRACT-MIN: $a$ （$d[a]=2$ 最小），**宣告 $a$ 已确定**，$d[a]=2$。
3. EXTRACT-MIN: $b$（$d[b]=4$），松弛 $b\to a$：$d[b]+w=-3+4=1$，但 $a$ **已被标记为完成** → 无法更新！

**Dijkstra 给出 $d[a]=2$，正确答案是 $1$**。❌

> **本质原因**：负权边使"贪心已确定"的论断失效——可以绕一条"暂时更长"的路再走一条负权边，反而更短。

---

## 22.2 A\* 搜索算法

### 22.2.1 动机：Dijkstra 的盲目搜索

Dijkstra 像一个没有地图的探险家：它盲目地向四面八方扩展，不管目标在哪个方向。

> **A\*** ：给探险家一个"预估地"。它用一个**启发式函数 $h(v)$**（对从 $v$ 到目标 $t$ 的真实距离的估计）来引导搜索朝正确方向快速推进。

### 22.2.2 核心公式

每个节点 $v$ 的综合评分：

$$f(v) = g(v) + h(v)$$

其中：
- $g(v)$：从源点 $s$ 到 $v$ 的**已知实际代价**（即 Dijkstra 中的 $d[v]$）
- $h(v)$：从 $v$ 到目标 $t$ 的**估计剩余代价**（启发式函数）
- $f(v)$：经过 $v$ 到达 $t$ 的**预估总代价**

A\* 每次从优先队列中取出 $f$ 值最小的节点（而非 $g$ 值最小）。

### 22.2.3 可采纳性（Admissibility）

**定义**：若对所有节点 $v$，$h(v) \le \delta(v, t)$（$h$ 从不高估真实代价），则称 $h$ 是**可采纳的（Admissible）**。

> **定理（A\* 正确性）**：当 $h$ 可采纳时，A\* 找到的是从 $s$ 到 $t$ 的最优路径。

**常用启发式函数**：

| 场景 | 启发式函数 | 说明 |
|------|-----------|------|
| 网格图（四方向移动） | $h = \|x_1 - x_2\| + \|y_1 - y_2\|$（曼哈顿距离） | 可采纳 |
| 网格图（八方向移动） | $h = \max(\|x_1-x_2\|, \|y_1-y_2\|)$（棋盘距离） | 可采纳 |
| 任意图（欧氏空间） | $h = \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}$（欧氏距离） | 可采纳（若边权≥实际距离） |
| $h \equiv 0$ | — | 退化为 Dijkstra |

**一致性（Consistency）**：更强的条件，要求对每条边 $(u,v)$ 有 $h(u) \le w(u,v) + h(v)$（三角不等式）。满足一致性的 $h$ 一定可采纳。

### 22.2.4 A\* vs Dijkstra

| 对比维度 | Dijkstra | A\* |
|---------|---------|-----|
| 搜索方向 | 无方向，均匀扩展 | 有引导，偏向目标方向 |
| 展开节点数 | 更多（最坏全图） | 更少（$h$ 越准越少） |
| 保证最优 | ✅（非负权图） | ✅（需 $h$ 可采纳） |
| 适用场景 | 单源所有终点 | 单源单目标 |
| 当 $h \equiv 0$ | 退化为 Dijkstra | — |

<div data-component="AStarPathfinding"></div>

> **实际应用**：游戏 AI 路径规划（Unity、Unreal Engine 中的寻路组件）、Google Maps 导航（结合地理信息预估剩余距离）。

---

## 22.3 最小生成树（MST）理论基础

### 22.3.1 问题定义

**输入**：无向连通加权图 $G=(V, E, w)$，边权 $w: E \to \mathbb{R}$

**输出**：一棵生成树 $T \subseteq E$（连通所有 $|V|$ 个节点，恰好 $|V|-1$ 条边），使得总权值 $\sum_{e \in T} w(e)$ **最小**。

> **生活类比**：你是一位网络工程师，需要用最少的电缆费用把 $n$ 个城市全部连通（每个城市直接或间接可达）。MST 告诉你最优的方案。

### 22.3.2 切割性质（Cut Property）——MST 正确性的基石

**定义**：图的一个**切割（Cut）** $(S, V \setminus S)$ 是把节点集 $V$ 分成两组 $S$ 和 $V \setminus S$。**跨越切割的边（Crossing Edges）** 是一端在 $S$、另一端在 $V \setminus S$ 的边。

**切割性质（*）**：设 $(S, V \setminus S)$ 是图 $G$ 的任意一个切割，且切割的跨越边中 $e^* = (u,v)$ 是**权值最小**者（若权值唯一则唯一确定），则 $e^*$ 属于某棵 MST。

**证明**（反证法）：

设反设 $e^*$ 不在任何 MST 中，取任意一棵 MST $T$。

将 $e^*$ 加入 $T$，形成图 $T \cup \{e^*\}$，它包含一个环 $C$（因为 $T$ 是树，加边成环）。

由于 $e^*$ 跨越切割 $(S, V \setminus S)$，环 $C$ 也必经过该切割，故 $C$ 中至少还有另一条跨越同一切割的边 $e'$（$e' \neq e^*$）。

由于 $e^*$ 是切割中权值最小的跨越边：$w(e^*) \le w(e')$（且由 $e^* \neq e'$，可取严格不等号若权唯一）。

构造 $T' = T \cup \{e^*\} \setminus \{e'\}$，则 $T'$ 仍是生成树，且：

$$w(T') = w(T) - w(e') + w(e^*) \le w(T)$$

若 $w(e^*) < w(e')$，则 $w(T') < w(T)$，与 $T$ 是 MST 矛盾。✓

若 $w(e^*) = w(e')$，则 $T'$ 也是 MST 且包含 $e^*$，与反设矛盾。✓

### 22.3.3 环路性质（Cycle Property）

**环路性质**：设 $C$ 是图中某个环，$e^* = (u,v)$ 是 $C$ 中**权值最大**者（唯一），则 $e^*$ 不属于任何 MST。

**证明**：反设 $e^*$ 在某棵 MST $T$ 中。从 $T$ 中删去 $e^*$ 得两棵子树，对应一个切割 $(S, V \setminus S)$。

$C$ 中除 $e^*$ 外至少还有一条跨越该切割的边 $e'$（$e' \in C$，$e' \neq e^*$，因为环也必须跨越该切割）。

$w(e') < w(e^*)$（$e^*$ 是环中最大），故替换 $e^*$ 为 $e'$ 可得更小代价的生成树，与 MST 矛盾。✓

### 22.3.4 MST 的唯一性

**定理**：若图中所有边权**互不相同**，则 MST 唯一。

**证明**：设存在两棵不同 MST $T_1 \neq T_2$，设 $e$ 是 $T_1 \setminus T_2$ 中权值最小的边。

$e \in T_1$，$e \notin T_2$。将 $e$ 加入 $T_2$，产生一个环 $C$。$C$ 中必存在边 $e' \in C \setminus T_1$（否则 $C \subseteq T_1$，与 $T_1$ 是树矛盾）。

由 $e$ 的选取方式，$w(e) < w(e')$（$e$ 是 $T_1 \setminus T_2$ 的最小权边，$e'$ 在 $T_2 \setminus T_1$ 中）。

故用 $e$ 替换 $e'$：$T_2' = T_2 \cup \{e\} \setminus \{e'\}$，$w(T_2') < w(T_2)$，与 $T_2$ 是 MST 矛盾。✓

---

## 22.4 Kruskal 算法

### 22.4.1 贪心策略：按边权从小到大贪心筛选

> **类比**：你面前有一堆按价格从低到高排好的电缆，每次拿起最便宜的一根，如果它能连接两个尚未互通的城市，就使用它；否则丢弃（它会形成环路，浪费资源）。

**算法**：

1. 将所有边按权值从小到大排序
2. 初始化并查集（每个节点独立）
3. 依次处理每条边 $(u,v,w)$：
   - 若 $\text{FIND}(u) \neq \text{FIND}(v)$（不在同一分量）：加入 MST，执行 $\text{UNION}(u,v)$
   - 否则（成环）：跳过
4. 重复直到 MST 拥有 $|V|-1$ 条边

### 22.4.2 具体示例追踪

使用 6 节点图（节点 A~F）：

```
边集（u, v, 权w）：
  A-B:4   A-C:2
  B-C:1   B-D:5
  C-D:8   C-E:10
  D-E:2   D-F:6
  E-F:3
```

排序后的处理顺序：

| 步骤 | 边 | 权重 | FIND(u) vs FIND(v) | 决策 | MST 边集 |
|------|------|------|---------------------|------|---------|
| 1 | B-C | 1 | B≠C | ✅ 加入 | {B-C} |
| 2 | A-C | 2 | A≠C | ✅ 加入 | {B-C, A-C} |
| 3 | D-E | 2 | D≠E | ✅ 加入 | {B-C, A-C, D-E} |
| 4 | E-F | 3 | E≠F | ✅ 加入 | {B-C, A-C, D-E, E-F} |
| 5 | A-B | 4 | **A=B**（均属{A,B,C}） | ❌ 成环，跳过 | — |
| 6 | B-D | 5 | B≠D（{A,B,C}连接{D,E,F}） | ✅ 加入 | {B-C, A-C, D-E, E-F, B-D} |

MST 总权值：$1+2+2+3+5 = 13$（5条边 = 6节点-1 ✓）

<div data-component="KruskalUnionFindTrace"></div>

### 22.4.3 代码实现（路径压缩 + 按秩合并）

```python
from typing import List, Tuple

class UnionFind:
    """
    并查集（路径压缩 + 按秩合并）
    单次操作均摊复杂度 O(α(n))，其中 α 为 Ackermann 反函数（近乎 O(1)）
    """
    def __init__(self, n: int):
        self.parent = list(range(n))  # parent[i] = i（自身为祖先）
        self.rank   = [0] * n         # 秩：子树高度上界，用于合并时平衡
    
    def find(self, x: int) -> int:
        # 路径压缩：将 x 到根路径上的所有节点直接挂到根
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 递归压缩
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """合并 x 和 y 所在集合，返回是否成功合并（False 表示已在同一集合）"""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False  # 已在同一集合，加边会成环
        # 按秩合并：小树挂到大树下，避免退化为链表
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx  # 保证 rx 是较大的树
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1  # 等秩时树高增加 1
        return True


def kruskal(n: int, edges: List[Tuple[int, int, int]]) -> Tuple[List, int]:
    """
    Kruskal 最小生成树算法
    
    参数:
      n:     节点数（0-indexed）
      edges: 边列表 [(u, v, w), ...]
    返回:
      (mst_edges, total_weight): MST 边集与总权值
    
    时间复杂度: O(E log E)（排序主导）
    空间复杂度: O(V + E)
    """
    # 步骤 1：按权值排序所有边
    edges_sorted = sorted(edges, key=lambda e: e[2])  # 按第三项（权重）排序

    uf = UnionFind(n)
    mst_edges = []
    total_weight = 0

    for u, v, w in edges_sorted:
        # FIND：检查是否在同一连通分量
        if uf.union(u, v):             # UNION 成功 → 不成环 → 加入 MST
            mst_edges.append((u, v, w))
            total_weight += w
            if len(mst_edges) == n - 1:  # MST 恰好有 n-1 条边，提前结束
                break

    return mst_edges, total_weight


# ── 示例：6 节点图 ─────────────────────────────────────────────
if __name__ == "__main__":
    # A=0, B=1, C=2, D=3, E=4, F=5
    edges = [
        (0,1,4), (0,2,2),  # A-B:4, A-C:2
        (1,2,1), (1,3,5),  # B-C:1, B-D:5
        (2,3,8), (2,4,10), # C-D:8, C-E:10
        (3,4,2), (3,5,6),  # D-E:2, D-F:6
        (4,5,3),           # E-F:3
    ]
    n = 6
    mst, total = kruskal(n, edges)
    names = ['A','B','C','D','E','F']
    print("MST 边集：")
    for u, v, w in mst:
        print(f"  {names[u]}-{names[v]}: {w}")
    print(f"总权值：{total}")  # 预期：13
```

```cpp
#include <bits/stdc++.h>
using namespace std;

// ── 并查集 ────────────────────────────────────────────────────
struct UnionFind {
    vector<int> parent, rank_;
    
    UnionFind(int n) : parent(n), rank_(n, 0) {
        iota(parent.begin(), parent.end(), 0);  // parent[i] = i
    }
    
    // 路径压缩
    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }
    
    // 按秩合并，返回是否成功合并
    bool unite(int x, int y) {
        int rx = find(x), ry = find(y);
        if (rx == ry) return false;  // 已连通，成环
        if (rank_[rx] < rank_[ry]) swap(rx, ry);
        parent[ry] = rx;
        if (rank_[rx] == rank_[ry]) rank_[rx]++;
        return true;  // 合并成功
    }
};

// ── Kruskal MST ───────────────────────────────────────────────
struct Edge { int u, v, w; };

pair<vector<Edge>, int> kruskal(int n, vector<Edge>& edges) {
    /*
     * 时间: O(E log E)（排序主导）
     * 空间: O(V)（并查集）
     */
    // 按权值升序排序
    sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        return a.w < b.w;
    });

    UnionFind uf(n);
    vector<Edge> mst;
    int total = 0;

    for (auto& [u, v, w] : edges) {
        if (uf.unite(u, v)) {      // 不成环 → 加入 MST
            mst.push_back({u, v, w});
            total += w;
            if ((int)mst.size() == n - 1) break;  // MST 完成
        }
    }
    return {mst, total};
}

int main() {
    // A=0, B=1, C=2, D=3, E=4, F=5
    vector<Edge> edges = {
        {0,1,4}, {0,2,2}, {1,2,1}, {1,3,5},
        {2,3,8}, {2,4,10},{3,4,2}, {3,5,6}, {4,5,3}
    };
    auto [mst, total] = kruskal(6, edges);
    string names[] = {"A","B","C","D","E","F"};
    cout << "MST 边集：\n";
    for (auto [u,v,w] : mst)
        cout << "  " << names[u] << "-" << names[v] << ": " << w << "\n";
    cout << "总权值：" << total << "\n";  // 13
    return 0;
}
```

### 22.4.4 正确性证明（切割性质应用）

**定理**：Kruskal 选择的每条边都属于某棵 MST。

**证明**：设 Kruskal 选择边 $e = (u,v,w)$。此时 $u$ 和 $v$ 不在同一连通分量，设 $u$ 所在分量为 $S$，$v$ 所在分量为 $V \setminus S$。

考虑切割 $(S, V \setminus S)$。由于 Kruskal 按升序处理边，$e$ 是此时还未加入的所有跨越该切割的边中权值最小者（否则之前就有更小的跨越边被加入，$u,v$ 早就连通了）。

由**切割性质**：$e$ 属于某棵 MST。✓

### 22.4.5 时间复杂度

- **排序**：$O(E \log E)$
- **并查集操作**：$O(E \cdot \alpha(V))$（近乎 $O(E)$）
- **总计**：$O(E \log E) = O(E \log V)$（因为 $E \le V^2$，故 $\log E \le 2\log V$）

**适用场景**：稀疏图（$E \ll V^2$），因为排序代价是 $O(E \log E)$。

---

## 22.5 Prim 算法

### 22.5.1 贪心策略：MST 从一点"生长"

> **类比**：你在建村子供水网络。从水泵（源点）开始，每次将已有水管能连通的所有未建管道中**最便宜的一条**铺设出去。MST 像一棵从源点不断向外生长的树。

**算法**（优先队列版）：

1. 初始化：选任意节点 $r$ 为起点，$\text{key}[r]=0$，$\text{key}[v]=\infty$（其余节点）
2. 优先队列 $Q \leftarrow V$（按 $\text{key}$ 排序）
3. **while** $Q \neq \emptyset$:
   - $u \leftarrow \text{EXTRACT-MIN}(Q)$（key 值最小的未加入 MST 的节点）
   - 将 $u$ 加入 MST，连接边 $(\pi[u], u)$（$\pi[u]$ 是 $u$ 的前驱节点）
   - **for each** 邻居 $v$ of $u$（$v$ 尚在 $Q$ 中）：
     - 若 $w(u,v) < \text{key}[v]$：$\text{key}[v] \leftarrow w(u,v)$，$\pi[v] \leftarrow u$，更新 $Q$ 中 $v$ 的 key

**关键区别**：

- **Dijkstra** 的 key 是"从源点到该节点的最短路径"（$d[v] = d[u] + w$）
- **Prim** 的 key 是"连接 MST 与该节点的最小边权"（$\text{key}[v] = w(u,v)$，只需边本身的权值）

### 22.5.2 具体示例追踪（同一 6 节点图，从 A 出发）

初始：$\text{key}=[A:0, B:\infty, C:\infty, D:\infty, E:\infty, F:\infty]$，$Q = \{A,B,C,D,E,F\}$

| 步骤 | EXTRACT-MIN | 更新邻居 key | 当前 MST 边 |
|------|------------|------------|------------|
| 1 | A (0) | B:min(∞,4)=4; C:min(∞,2)=2 | — |
| 2 | C (2) | B:min(4,1)=1; D:min(∞,8)=8; E:min(∞,10)=10 | A-C(2) |
| 3 | B (1) | D:min(8,5)=5 | A-C(2), C-B(1) |
| 4 | D (5) | E:min(10,2)=2; F:min(∞,6)=6 | A-C(2),C-B(1),B-D(5) |
| 5 | E (2) | F:min(6,3)=3 | +D-E(2) |
| 6 | F (3) | — | +E-F(3) |

MST 边：A-C(2), C-B(1), B-D(5), D-E(2), E-F(3)，总权值 $= 1+2+2+3+5 = 13$ ✓

<div data-component="PrimMSTGrowth"></div>

### 22.5.3 代码实现

```python
import heapq
from collections import defaultdict
from typing import List, Tuple, Dict

def prim(n: int, adj: Dict[int, List[Tuple[int, int]]], start: int = 0) -> Tuple[List, int]:
    """
    Prim 最小生成树算法（二叉堆 + 懒删除）
    
    参数:
      n:     节点数（0-indexed）
      adj:   邻接表，adj[u] = [(v, w), ...]（无向图，双向存储）
      start: 起始节点
    返回:
      (mst_edges, total_weight)
    
    时间复杂度: O((V+E) log V)
    空间复杂度: O(V+E)
    """
    in_mst = [False] * n         # 标记节点是否已加入 MST
    key     = [float('inf')] * n  # 当前已知的"连接 MST 的最小边权"
    parent  = [-1] * n            # MST 中的前驱节点

    key[start] = 0
    # 堆中存 (key值, 节点, 前驱节点)
    pq = [(0, start, -1)]

    mst_edges  = []
    total_weight = 0

    while pq:
        k_u, u, par = heapq.heappop(pq)

        # 懒删除：若已加入 MST，跳过
        if in_mst[u]:
            continue

        in_mst[u] = True
        if par != -1:  # 起点没有父边
            mst_edges.append((par, u, k_u))
            total_weight += k_u

        # 更新所有未加入 MST 的邻居
        for v, w in adj[u]:
            if not in_mst[v] and w < key[v]:
                key[v] = w
                parent[v] = u
                heapq.heappush(pq, (w, v, u))  # 旧值成为过期项（懒删除）

    return mst_edges, total_weight


# ── 示例 ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # A=0, B=1, C=2, D=3, E=4, F=5
    raw_edges = [
        (0,1,4),(0,2,2),(1,2,1),(1,3,5),
        (2,3,8),(2,4,10),(3,4,2),(3,5,6),(4,5,3)
    ]
    adj = defaultdict(list)
    for u, v, w in raw_edges:
        adj[u].append((v, w))  # 无向图：双向存储
        adj[v].append((u, w))

    mst, total = prim(n=6, adj=adj, start=0)
    names = ['A','B','C','D','E','F']
    print("MST 边集（Prim from A）：")
    for u, v, w in mst:
        print(f"  {names[u]}-{names[v]}: {w}")
    print(f"总权值：{total}")  # 13
```

```cpp
#include <bits/stdc++.h>
using namespace std;

pair<vector<tuple<int,int,int>>, int> prim(
    int n, vector<vector<pair<int,int>>>& adj, int start = 0) {
    /*
     * Prim MST（二叉堆 + 懒删除）
     * 时间: O((V+E) log V)
     * 空间: O(V+E)
     */
    vector<bool> in_mst(n, false);
    vector<int> key(n, INT_MAX), par(n, -1);
    key[start] = 0;

    // 最小堆：(key值, 节点, 父节点)
    priority_queue<tuple<int,int,int>,
                   vector<tuple<int,int,int>>,
                   greater<>> pq;
    pq.push({0, start, -1});

    vector<tuple<int,int,int>> mst_edges;
    int total = 0;

    while (!pq.empty()) {
        auto [k_u, u, p] = pq.top(); pq.pop();

        if (in_mst[u]) continue;  // 懒删除
        in_mst[u] = true;

        if (p != -1) {
            mst_edges.push_back({p, u, k_u});
            total += k_u;
        }

        for (auto [v, w] : adj[u]) {
            if (!in_mst[v] && w < key[v]) {
                key[v] = w;
                par[v] = u;
                pq.push({w, v, u});  // 旧堆项成为过期
            }
        }
    }
    return {mst_edges, total};
}

int main() {
    int n = 6;
    // A=0,B=1,C=2,D=3,E=4,F=5
    vector<tuple<int,int,int>> raw = {
        {0,1,4},{0,2,2},{1,2,1},{1,3,5},
        {2,3,8},{2,4,10},{3,4,2},{3,5,6},{4,5,3}
    };
    vector<vector<pair<int,int>>> adj(n);
    for (auto [u,v,w] : raw) {
        adj[u].emplace_back(v,w);
        adj[v].emplace_back(u,w);
    }
    auto [mst, total] = prim(n, adj, 0);
    string names[] = {"A","B","C","D","E","F"};
    cout << "MST 边集：\n";
    for (auto [u,v,w] : mst)
        cout << "  " << names[u] << "-" << names[v] << ": " << w << "\n";
    cout << "总权值：" << total << "\n";  // 13
    return 0;
}
```

### 22.5.4 Kruskal vs Prim 综合对比

| 对比维度 | Kruskal | Prim |
|---------|---------|------|
| **核心数据结构** | 排序 + 并查集 | 优先队列（最小堆） |
| **时间复杂度** | $O(E\log E)$ | $O((V+E)\log V)$ 堆；$O(V^2)$ 矩阵 |
| **稀疏图**（$E \approx V$） | ✅ 更优（排序少） | 相当 |
| **稠密图**（$E \approx V^2$） | ❌ 排序 $O(V^2\log V)$ | ✅ 矩阵版 $O(V^2)$ 更优 |
| **实现难度** | 低（排序+并查集） | 中（优先队列+key更新） |
| **负权边** | ✅ 支持（MST算法均支持） | ✅ 支持 |
| **适用场景** | 竞赛/稀疏图首选 | 稠密图（如邻接矩阵存储时） |
| **起点要求** | 无 | 需指定起点（结果不变） |
| **并行化** | 困难（排序顺序依赖） | 相对容易（类似Dijkstra的BFS扩展） |

> **实践选择建议**：
> - 竞赛/算法题（大多数稀疏图）→ **Kruskal**（实现简单，且通常 $E$ 不大）
> - 稠密图（如 $E=O(V^2)$）→ **Prim 的矩阵版**（$O(V^2)$ 优于 Kruskal 的 $O(V^2\log V)$）

---

## 22.6 算法选择指南

| 问题类型 | 推荐算法 | 复杂度 | 关键限制 |
|---------|---------|--------|---------|
| 单源最短路（无负权） | **Dijkstra** | $O((V+E)\log V)$ | 必须无负权边 |
| 单源最短路（有负权） | **Bellman-Ford** | $O(VE)$ | 可检测负权环 |
| 单源最短路（DAG） | **拓扑序 DP** | $O(V+E)$ | 必须是 DAG |
| 单源单目标（有地理信息） | **A\*** | 实践远优于 Dijkstra | 需设计可采纳 $h$ |
| 最小生成树（稀疏图） | **Kruskal** | $O(E\log E)$ | 无向图，可有负权 |
| 最小生成树（稠密图） | **Prim（矩阵版）** | $O(V^2)$ | 无向图 |
| 全对最短路 | **Floyd-Warshall** | $O(V^3)$ | 下一章介绍 |

---

## 22.7 经典 LeetCode 题解思路

### LeetCode 743：网络延迟时间（Dijkstra）

> 给定 $n$ 个节点和 $k$ 条有向带权边，从节点 $k$ 出发，求信号到达所有节点的最小时间。

**思路**：单源最短路 = Dijkstra。答案 = $\max_{v \in V} d[v]$（若某节点不可达，返回 -1）。

```python
def networkDelayTime(times, n, k):
    import heapq
    from collections import defaultdict
    g = defaultdict(list)
    for u, v, w in times:
        g[u].append((v, w))
    dist = [float('inf')] * (n + 1)
    dist[k] = 0
    pq = [(0, k)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        for v, w in g[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    ans = max(dist[1:])
    return ans if ans < float('inf') else -1
```

### LeetCode 1584：连接所有点的最小费用（MST）

> $n$ 个点在二维平面上，连接点 $i$ 和 $j$ 的费用为曼哈顿距离 $|x_i-x_j|+|y_i-y_j|$，求连通所有点的最小费用（即 MST）。

**思路**：建完全图（$O(n^2)$ 条边），用 Kruskal 或 Prim（由于节点数不大，二者均可）。

```python
def minCostConnectPoints(points):
    import heapq
    n = len(points)
    visited = [False] * n
    # Prim 从节点 0 出发
    pq = [(0, 0)]  # (代价, 节点)
    total, cnt = 0, 0
    while cnt < n:
        cost, u = heapq.heappop(pq)
        if visited[u]: continue
        visited[u] = True
        total += cost
        cnt += 1
        for v in range(n):
            if not visited[v]:
                d = abs(points[u][0]-points[v][0]) + abs(points[u][1]-points[v][1])
                heapq.heappush(pq, (d, v))
    return total
```

---

## 22.8 常见错误与调试技巧

### ⚠️ Dijkstra 的四大陷阱

1. **负权边**：Dijkstra 对负权图给出错误结果，切记改用 Bellman-Ford。
2. **过期堆项未判断**：忘记 `if d_u > dist[u]: continue`，导致同一节点被多次错误处理。
3. **有向图 vs 无向图**：建图时，无向图需双向存边；Dijkstra 只用于有向图（无向图可视为双向有向图）。
4. **节点编号**：LeetCode 题目节点通常 1-indexed，注意数组大小和初始化。

### ⚠️ MST 的三大陷阱

1. **Kruskal 判环用 DFS**：正确做法是并查集（$O(\alpha)$），DFS 判环是 $O(V)$，效率低且实现复杂。
2. **Prim 邻居已在 MST 中仍更新**：一定要检查 `not in_mst[v]`，否则产生环路。
3. **平行边**：多条相同端点的边，Kruskal 只取最小的那条（因为另一条权值更大一定会成环）；Prim 自然处理（只保留 key 最小的）。

---

## 22.9 思考题与练习

1. **平行边问题**：图中有两条边 A-B，权值分别为 3 和 7。Kruskal 和 Prim 各如何处理？MST 还是唯一的吗？

2. **Dijkstra 与 BFS**：如果图中**所有边权都相同**，Dijkstra 退化为什么算法？

3. **次小生成树**：在 MST 的基础上，如何用 $O(V^2)$ 的时间找到**次小生成树**（总权值第二小的生成树）？

4. **带负权的 MST**：MST 的 Kruskal/Prim 算法是否支持负权边？（提示：思考切割性质的证明中是否需要边权非负）

5. **瓶颈路径问题**：定义 $s$ 到 $t$ 的"瓶颈路径"为使路径上最大边权最小化的路径。这与 MST 有什么关系？

---

## 22.10 本章总结

```
本章知识图谱：

  Ch21 的 RELAX 操作
       ↓
  Bellman-Ford（通用，O(VE)）
       ↓ "如果边权非负，贪心加速"
  ┌─────────────────────────────────────┐
  │  Dijkstra（O((V+E)logV)）           │
  │  + 启发式 → A*（实践更快）          │
  └─────────────────────────────────────┘
  
  最小生成树（MST）
  ├── 理论基础：切割性质 & 环路性质
  ├── Kruskal: 排序+并查集，O(E logE)，稀疏图首选
  └── Prim:    优先队列，O((V+E)logV)，可退化为O(V²)

  下一章：Floyd-Warshall 全对最短路
```

**核心收获**：

- Dijkstra 的贪心能成立的**唯一前提**是边权非负——非负保证"先到先确定"的单调性。
- MST 的两大性质（切割 + 环路）提供了统一的证明框架，Kruskal 和 Prim 不过是从两个不同角度贪心地利用这两条性质。
- 算法选择的核心是**读懂约束**：有无负权、图的稠密程度、是单源还是全对、是否知道目标节点。

---

**参考资料**

- CLRS 第4版 Chapter 22（Dijkstra）、Chapter 23（MST）
- MIT 6.006 Lecture 15（Dijkstra）、Lecture 16（MST）
- Sedgewick & Wayne, *Algorithms* (4th ed.) 章节 4.3（MST）、4.4（最短路径）
- [LeetCode 743](https://leetcode.com/problems/network-delay-time/)
- [LeetCode 1584](https://leetcode.com/problems/min-cost-to-connect-all-points/)
- [LeetCode 778](https://leetcode.com/problems/swim-in-rising-water/)

# Chapter 18: 图的基本概念与表示（Graph Fundamentals & Representations）

> **学习目标**：  
> 彻底掌握图的所有基本术语，建立严格的形式化定义；能清晰辨别有向/无向图、加权/无权图、稠密/稀疏图等六大维度的分类；深刻理解邻接矩阵与邻接表两大存储结构的空间/时间权衡，并能根据图的特性与算法需求做出合理选择；掌握握手定理、入/出度等核心性质，为后续图遍历（BFS/DFS）、最短路、最小生成树等章节打下坚实基础。

---

## 18.1 图的定义与分类

### 18.1.1 为什么要研究图？——现实世界的连接关系

**生活比喻——城市与道路**：想象一张城市地图：城市是**节点（顶点，Vertex）**，城市之间的道路是**边（Edge）**。从北京到上海可以走，但从上海到北京也可以走（无向边）；但如果是单行道，方向就重要了（有向边）。如果路上有收费站，我们关心费用（加权边）。

图（Graph）就是这样一种**用于表示对象之间关系的数学结构**。它无处不在：

| 现实场景 | 顶点 | 边 | 边的类型 |
|---|---|---|---|
| 社交网络（微信好友） | 用户 | 好友关系 | 无向、无权 |
| 微博关注 | 用户 | 关注关系 | **有向**、无权 |
| 导航地图 | 地点 | 道路 | 无向/有向、**加权**（距离/时间）|
| 互联网 | 网页 | 超链接 | 有向、无权 |
| 电路图 | 元器件 | 导线 | 无向、加权（电阻）|
| 项目依赖 | 软件包 | 依赖关系 | 有向、无权 |
| 航班网络 | 城市 | 航线 | 有向、加权（时间/价格）|
| 学科知识图谱 | 概念 | 先修关系 | 有向、无权 |

**形式化定义**：一个图 $G = (V, E)$ 由两个集合组成：
- $V$：**顶点集（Vertex Set）**，也写作 $V(G)$，表示所有节点的集合，大小记为 $|V|$ 或 $n$
- $E$：**边集（Edge Set）**，也写作 $E(G)$，表示所有边的集合，大小记为 $|E|$ 或 $m$

> **约定**：本课程后续章节统一用 $V$（或 $n$）表示顶点数，$E$（或 $m$）表示边数。算法的时间复杂度通常用 $O(V + E)$ 表示。

---

### 18.1.2 有向图（Digraph）vs 无向图

**无向图（Undirected Graph）**：边没有方向，$(u, v)$ 与 $(v, u)$ 表示同一条边。  
- 记法：边用 $\{u, v\}$ 或 $(u, v)$（无秩序）表示
- 例：好友关系（A 是 B 的好友 ⟺ B 是 A 的好友）

**有向图（Directed Graph，Digraph）**：每条边有方向，从起点（尾，Tail）指向终点（头，Head）。  
- 记法：边用有序对 $(u, v)$ 表示（$u \to v$，但 $v \not\to u$）
- 例：关注关系（A 关注了 B，但 B 未必关注 A）

```
无向图示例（好友网络）        有向图示例（关注关系）

    A ――― B                   A ——→ B
    |       |                   ↑       |
    C ――― D                   C ←—— D
```

**关键区别**：
- 无向图中，若有 $n$ 个顶点，最多有 $\binom{n}{2} = \frac{n(n-1)}{2}$ 条边（完全图）。
- 有向图中，每对顶点 $(u, v)$ 可以有 $u\to v$ 和 $v\to u$ 两条独立的边，最多有 $n(n-1)$ 条边（允许两个方向）。
- 存储时，无向图的邻接表需要**每条边存两次**（$u \to v$ 和 $v \to u$），这是初学者最常见的遗漏！

**代码实现（图的基础类）**：

```python
# ── 无向图的邻接表表示 ─────────────────────────────────────────────
from collections import defaultdict

class Graph:
    """无向图，基于邻接表实现。"""

    def __init__(self, directed: bool = False):
        # 顶点 → 邻居列表的映射（可以是任意可哈希类型作为顶点 ID）
        self.adj: dict[int, list[int]] = defaultdict(list)
        self.directed = directed
        self.num_vertices = 0
        self.num_edges = 0

    def add_edge(self, u: int, v: int) -> None:
        """添加一条边 u-v（无向）或 u→v（有向）。"""
        # 边界情况：需要追踪新顶点
        if u not in self.adj:
            self.num_vertices += 1
        if v not in self.adj:
            self.num_vertices += 1

        self.adj[u].append(v)
        self.num_edges += 1

        if not self.directed:
            # 无向图：每条边存两次，反方向也需存储
            # 易错点：忘记存反方向 → BFS/DFS 无法从 v 侧发现 u
            self.adj[v].append(u)

    def neighbors(self, u: int) -> list[int]:
        """返回顶点 u 的所有邻居列表。"""
        return self.adj[u]

    def __repr__(self) -> str:
        lines = [f"{'有向' if self.directed else '无向'}图 |V|={self.num_vertices}, |E|={self.num_edges}"]
        for u, neighbors in sorted(self.adj.items()):
            lines.append(f"  {u}: {neighbors}")
        return "\n".join(lines)

# ── 演示 ───────────────────────────────────────────────────────────
g = Graph(directed=False)
g.add_edge(0, 1)  # 无向图：自动存储 0→1 和 1→0
g.add_edge(0, 2)
g.add_edge(1, 3)
print(g)
# 输出：
# 无向图 |V|=4, |E|=3
#   0: [1, 2]
#   1: [0, 3]
#   2: [0]
#   3: [1]

dg = Graph(directed=True)
dg.add_edge(0, 1)  # 有向图：只存 0→1
dg.add_edge(1, 2)
dg.add_edge(2, 0)
print(dg)
# 输出：
# 有向图 |V|=3, |E|=3
#   0: [1]
#   1: [2]
#   2: [0]
```
```cpp
// ── 无向图 / 有向图的邻接表表示 ───────────────────────────────────
#include <iostream>
#include <vector>
#include <algorithm>

class Graph {
public:
    int n;          // 顶点数（顶点编号 0 .. n-1）
    int m;          // 边数
    bool directed;
    std::vector<std::vector<int>> adj;   // adj[u] = u 的邻居列表

    explicit Graph(int n, bool directed = false)
        : n(n), m(0), directed(directed), adj(n) {}

    void add_edge(int u, int v) {
        // 前置条件：0 <= u, v < n
        adj[u].push_back(v);
        ++m;
        if (!directed) {
            // 无向图：反方向也要存
            // 易错点：遗漏此行会导致遍历缺失
            adj[v].push_back(u);
        }
    }

    const std::vector<int>& neighbors(int u) const { return adj[u]; }

    void print() const {
        std::cout << (directed ? "有向" : "无向") << "图 |V|=" << n
                  << ", |E|=" << m << "\n";
        for (int u = 0; u < n; ++u) {
            std::cout << "  " << u << ": [";
            for (int i = 0; i < (int)adj[u].size(); ++i) {
                if (i) std::cout << ", ";
                std::cout << adj[u][i];
            }
            std::cout << "]\n";
        }
    }
};

int main() {
    Graph g(4, false);    // 无向图，4 个顶点
    g.add_edge(0, 1);
    g.add_edge(0, 2);
    g.add_edge(1, 3);
    g.print();
    // 输出：
    // 无向图 |V|=4, |E|=3
    //   0: [1, 2]
    //   1: [0, 3]
    //   2: [0]
    //   3: [1]
    return 0;
}
```

---

### 18.1.3 加权图（Weighted Graph）vs 无权图

**无权图**：边只有"存在"与"不存在"之分，没有数值信息。在邻接表中，只需存储邻居 ID。

**加权图（Weighted Graph）**：每条边附带一个**权重（Weight）**，可以表示距离、费用、容量、时间等。在邻接表中，需要同时存储邻居 ID 和对应权重。

**权重可以是负数**：在货币套利、差分约束等问题中，负权边是合法且必要的（Bellman-Ford 算法专门处理此类情况）。

```
加权无向图示例（城市路网）：

    北京 ——100km—— 天津
      |                   \
   800km              120km
      |                     \
   上海 ——200km—— 杭州
```

**代码实现（加权图的邻接表）**：

```python
# ── 加权图：邻接表存 (邻居, 权重) 二元组 ──────────────────────────
from collections import defaultdict

class WeightedGraph:
    """加权无向图，adj[u] 存储 (v, w) 列表。"""

    def __init__(self, directed: bool = False):
        # adj[u] = [(v, weight_of_u_v), ...]
        # 使用元组而非单独的矩阵，节省空间（稀疏图时尤为明显）
        self.adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
        self.directed = directed

    def add_edge(self, u: int, v: int, w: float = 1.0) -> None:
        """添加权重为 w 的边 u-v（无向）或 u→v（有向）。"""
        self.adj[u].append((v, w))
        if not self.directed:
            self.adj[v].append((u, w))   # 反方向存相同权重

    def neighbors(self, u: int) -> list[tuple[int, float]]:
        """返回 [(邻居节点, 边权重), ...] 列表。"""
        return self.adj[u]

# ── 演示：路网 ─────────────────────────────────────────────────────
road = WeightedGraph()
road.add_edge(0, 1, 100)   # 北京(0) - 天津(1), 100km
road.add_edge(0, 2, 800)   # 北京(0) - 上海(2), 800km
road.add_edge(1, 3, 120)   # 天津(1) - 杭州(3), 120km（假设）
road.add_edge(2, 3, 200)   # 上海(2) - 杭州(3), 200km

# Dijkstra 等算法会这样遍历邻居
for v, w in road.neighbors(0):
    print(f"北京 → 顶点{v}，距离 {w}km")
# 输出：
# 北京 → 顶点1，距离 100km
# 北京 → 顶点2，距离 800km
```
```cpp
// ── 加权图：邻接表存 pair<int,int> 或自定义 Edge 结构体 ──────────
#include <iostream>
#include <vector>
#include <utility>

using Edge = std::pair<int, double>;  // {邻居ID, 权重}

class WeightedGraph {
public:
    int n;
    bool directed;
    std::vector<std::vector<Edge>> adj;  // adj[u] = {(v, w), ...}

    WeightedGraph(int n, bool directed = false)
        : n(n), directed(directed), adj(n) {}

    void add_edge(int u, int v, double w = 1.0) {
        adj[u].emplace_back(v, w);
        if (!directed) {
            // 无向加权图：反方向存相同权重
            adj[v].emplace_back(u, w);
        }
    }

    // 遍历邻居（用于 Dijkstra 等算法）
    const std::vector<Edge>& neighbors(int u) const { return adj[u]; }
};

int main() {
    WeightedGraph g(4);
    g.add_edge(0, 1, 100);  // 北京-天津 100km
    g.add_edge(0, 2, 800);  // 北京-上海 800km
    g.add_edge(1, 3, 120);
    g.add_edge(2, 3, 200);

    // 遍历北京(0)的邻居
    for (auto [v, w] : g.neighbors(0)) {
        std::cout << "北京 → " << v << "，距离 " << w << "km\n";
    }
    // 输出：
    // 北京 → 1，距离 100km
    // 北京 → 2，距离 800km
    return 0;
}
```

---

### 18.1.4 简单图、多重图、完全图 $K_n$

**简单图（Simple Graph）**：满足以下两个条件：
1. **无自环（No Self-loop）**：不存在 $(v, v)$ 形式的边（顶点连接到自身）
2. **无重边（No Multi-edge）**：任意两顶点之间至多一条边

> 本课程中，若无特别说明，所有图均为**简单图**。

**多重图（Multigraph）**：允许两顶点之间有多条平行边（如两城市之间有多条高铁线路）；**伪图（Pseudograph）** 还额外允许自环。

**完全图 $K_n$**：$n$ 个顶点的简单无向图，任意两个顶点之间都恰好有一条边。  
- 边数：$|E| = \binom{n}{2} = \frac{n(n-1)}{2}$
- 每个顶点的度：$n - 1$
- 特点：$K_4$ 以上的完全图中存在 $K_3$（三角形）

$$K_4 \text{ 示意：} \quad \binom{4}{2} = 6 \text{ 条边}$$

```
K₄（4个顶点的完全图）：

  1 ——— 2
  |\   /|
  | \ / |
  |  X  |
  | / \ |
  |/   \|
  3 ——— 4
  （对角线也有边）
```

**稀疏图与稠密图的边数范围**：

| 图的类型 | 最少边数 | 最多边数（无向）|
|---|---|---|
| 空图 | 0 | — |
| 树（连通无环）| $n - 1$ | $n - 1$ |
| 连通图 | $n - 1$ | $\frac{n(n-1)}{2}$ |
| 完全图 $K_n$ | $\frac{n(n-1)}{2}$ | — |

---

### 18.1.5 连通图、强连通图、弱连通图

图的**连通性（Connectivity）**描述了顶点之间是否存在路径。这是图中最基本也最重要的性质之一。

#### 无向图的连通性

**连通（Connected）**：无向图中，顶点 $u$ 和 $v$ 连通，当且仅当存在从 $u$ 到 $v$ 的路径（允许经过其他顶点）。

**连通图（Connected Graph）**：图中**任意两**顶点都互相连通。换言之，从任一顶点出发，可以到达图中所有其他顶点。

**连通分量（Connected Component）**：图中极大的连通子图。若一个图由 3 个相互独立的连通块组成，则它有 3 个连通分量。

```
连通图（1个连通分量）：       非连通图（2个连通分量）：

  A — B — C                  A — B     D — E
      |                                |
      D                                F
```

#### 有向图的连通性（更复杂！）

在有向图中，边的方向使连通性概念分裂为两种：

**强连通（Strongly Connected）**：有向图中，顶点 $u$ 和 $v$ 强连通，当且仅当 $u \to v$ **且** $v \to u$ 都有有向路径。

**强连通图（Strongly Connected Digraph）**：图中任意两顶点都强连通。

**弱连通图（Weakly Connected Digraph）**：忽略边的方向（把有向边当无向边），所有顶点连通。即有向图的"底层无向图"是连通的。

**强连通分量（Strongly Connected Component，SCC）**：有向图中极大的强连通子图。每个有向图可以唯一分解为若干 SCC（将在 Chapter 20 中深入讲解）。

```
有向图示例：

  A →→ B            这里：
  ↑    ↓            - {A, B, C} 构成一个 SCC（A→B→C→A 形成环）
  C ←← D            - {D} 单独是一个 SCC
                    - D → B，但 B ↛ D，故 D 不在同一 SCC

强连通：A↔B↔C
弱连通：A、B、C、D 都弱连通（忽略方向后是一个整体）
```

**为什么强连通分量重要？**：SCC 是有向图结构分析的核心工具，用于编译器优化（变量活跃分析）、网页排名（PageRank 中的"爬虫陷阱"检测）、2-SAT 问题求解等。

---

### 18.1.6 稀疏图（Sparse）vs 稠密图（Dense）与 DAG

#### 稀疏图 vs 稠密图

这是一个**影响算法与数据结构选型的关键概念**：

| | 稀疏图（Sparse Graph）| 稠密图（Dense Graph）|
|---|---|---|
| **边数量** | $E \ll V^2$，通常 $E = O(V)$ 或 $E = O(V \log V)$ | $E \approx V^2$，接近完全图 |
| **现实例子** | 路网、社交图（好友数有限）、依赖关系 | 完全图、平面图的对偶图 |
| **推荐存储** | **邻接表**（空间 $O(V + E)$）| **邻接矩阵**（空间 $O(V^2)$，但操作快）|
| **典型算法** | BFS/DFS $O(V+E)$ | Floyd-Warshall $O(V^3)$ |

> **工程直觉**：现实中绝大多数大规模图都是稀疏的。Google 的网页图有数千亿顶点，但每个网页平均只有约几十个链接（而非数千亿个）。

**判断图是稀疏还是稠密的经验规则**：  
- 如果边数 $E < 10V$（每个顶点平均邻居少于 10 个），通常视为稀疏图。  
- 如果 $E > \frac{V^2}{10}$，通常视为稠密图。

#### 有向无环图 DAG

**有向无环图（Directed Acyclic Graph，DAG）**：既是有向图，又不含任何有向环（不能从任一顶点出发沿有向边回到自身）。

```
DAG 示例（软件包依赖）：

  numpy ←── pandas ←── scikit-learn
    ↑                        ↑
  scipy ────────────────────┘
```

**DAG 的特殊性质**：
1. 必然存在**拓扑排序（Topological Sort）**：能找到一个顶点的线性顺序，使得所有有向边 $(u, v)$ 中 $u$ 都在 $v$ 前面（详见 Chapter 20）
2. 适合**动态规划**：因为无环，可以按拓扑序处理，每个子问题只被计算一次。
3. 常见于：任务调度、软件依赖管理、表达式求值树、有限状态自动机等。

**如何判断一个有向图是否是 DAG？**：运行 DFS，若存在任何**后向边（Back Edge）**，则有环，否则无环（详见 Chapter 19）。

---

<!-- 图类型分类器组件：让用户根据图的属性交互式判断 -->
<div data-component="GraphTypeClassifier"></div>

---

## 18.2 图的表示方式

### 18.2.1 邻接矩阵（Adjacency Matrix）

**核心思想**：用一个 $V \times V$ 的二维矩阵 $A$ 来表示图，其中：

$$A[u][v] = \begin{cases} 1 & \text{若存在边 } (u, v) \\ 0 & \text{否则} \end{cases} \quad \text{（无权图）}$$

$$A[u][v] = \begin{cases} w_{uv} & \text{若存在权重为 } w_{uv} \text{ 的边} \\ \infty & \text{否则（无穷大表示不可达）} \end{cases} \quad \text{（加权图）}$$

**无向图的对称性**：无向图的邻接矩阵是**对称矩阵**（$A[u][v] = A[v][u]$），可以只存储上三角部分节省一半空间，但实际工程中通常存完整矩阵以方便访问。

**有向图的不对称性**：有向图中 $A[u][v] \neq A[v][u]$ 是可能的。$A[u][v] = 1$ 表示有从 $u$ 到 $v$ 的边，但 $v$ 到 $u$ 未必有边。

**矩阵的幂有深刻含义**：$A^k[u][v]$ 等于从 $u$ 到 $v$ 长度恰好为 $k$ 的路径数目（线性代数方法，用于查找特定长度路径）！

**示例**：

```
图示（有向）：                邻接矩阵（A[行=起点][列=终点]）：

  0 ——→ 1                    [0][1][2][3]
  ↓       \                0: [0, 1, 1, 0]
  2 ←——— 3               1: [0, 0, 0, 1]  
                          2: [0, 0, 0, 0]
边：0→1, 0→2, 1→3, 3→2   3: [0, 0, 1, 0]
```

**邻接矩阵的核心操作复杂度**：

| 操作 | 时间复杂度 | 说明 |
|---|---|---|
| 初始化 | $O(V^2)$ | 分配并清零矩阵 |
| 添加/删除一条边 | $O(1)$ | 直接设置 `A[u][v]` |
| **判断边是否存在** | $O(1)$ | 直接读取 `A[u][v]`，这是矩阵最大优势 |
| 遍历顶点 u 的所有邻居 | $O(V)$ | 必须扫描整行 |
| 遍历所有边 | $O(V^2)$ | 扫描整个矩阵 |
| 空间占用 | $O(V^2)$ | 与边数无关，对稀疏图浪费严重 |

```python
# ── 邻接矩阵实现（无权无向图）─────────────────────────────────────
import math
from typing import Optional

class AdjMatrix:
    """邻接矩阵表示的图。
    
    设计考量：
    - 顶点编号必须是 0..n-1 的连续整数（非连续顶点请预先映射）
    - 无权图：A[u][v] ∈ {0, 1}
    - 加权图：A[u][v] = 权重（0 或 INF 表示无边）
    """

    INF = math.inf   # 加权图中"无边"的标记值

    def __init__(self, n: int, directed: bool = False, weighted: bool = False):
        self.n = n
        self.directed = directed
        self.weighted = weighted
        # 初始化为全 0（无权）或 INF（加权但无边）
        no_edge = 0 if not weighted else self.INF
        self.mat = [[no_edge] * n for _ in range(n)]
        if weighted:
            # 自身到自身的距离为 0（常用于 Floyd-Warshall）
            for i in range(n):
                self.mat[i][i] = 0

    def add_edge(self, u: int, v: int, w: float = 1) -> None:
        """添加边 u-v（无向）或 u→v（有向），权重为 w。"""
        # 边界检查：确保顶点索引有效
        assert 0 <= u < self.n and 0 <= v < self.n, "顶点越界"
        self.mat[u][v] = w
        if not self.directed:
            self.mat[v][u] = w  # 无向图：对称设置

    def has_edge(self, u: int, v: int) -> bool:
        """O(1) 判断边是否存在——邻接矩阵最大优势！"""
        if self.weighted:
            return self.mat[u][v] not in (0, self.INF)
        return self.mat[u][v] == 1

    def neighbors(self, u: int) -> list[int]:
        """遍历顶点 u 的所有邻居，O(V)——比邻接表慢！"""
        if self.weighted:
            return [v for v in range(self.n) if self.mat[u][v] not in (0, self.INF)]
        return [v for v in range(self.n) if self.mat[u][v] == 1]

    def print_matrix(self) -> None:
        """打印邻接矩阵（调试用）。"""
        header = "   " + " ".join(f"{j:3}" for j in range(self.n))
        print(header)
        for i, row in enumerate(self.mat):
            vals = " ".join(f"{(int(v) if v != self.INF else '∞'):3}" for v in row)
            print(f"{i:2} [{vals}]")

# ── 演示 ───────────────────────────────────────────────────────────
g = AdjMatrix(4, directed=False)
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 3)
g.print_matrix()
print(f"\n0-1 之间有边？{g.has_edge(0, 1)}")   # True, O(1)
print(f"0-3 之间有边？{g.has_edge(0, 3)}")   # False, O(1)
print(f"顶点 0 的邻居：{g.neighbors(0)}")      # [1, 2]，O(V)
```
```cpp
// ── 邻接矩阵实现（支持无权/加权、无向/有向）─────────────────────
#include <iostream>
#include <vector>
#include <limits>
#include <cassert>

class AdjMatrix {
public:
    static constexpr double INF = std::numeric_limits<double>::infinity();

    int n;
    bool directed, weighted;
    std::vector<std::vector<double>> mat;  // mat[u][v] = 权重或 0/INF

    AdjMatrix(int n, bool directed = false, bool weighted = false)
        : n(n), directed(directed), weighted(weighted)
    {
        double no_edge = weighted ? INF : 0.0;
        mat.assign(n, std::vector<double>(n, no_edge));
        if (weighted)
            for (int i = 0; i < n; ++i) mat[i][i] = 0.0;  // 自身距离为 0
    }

    void add_edge(int u, int v, double w = 1.0) {
        // 前置条件：0 <= u, v < n
        assert(u >= 0 && u < n && v >= 0 && v < n);
        mat[u][v] = w;
        if (!directed) mat[v][u] = w;  // 无向：对称
    }

    // O(1) 判断边是否存在
    bool has_edge(int u, int v) const {
        return weighted ? (mat[u][v] != INF && mat[u][v] != 0.0)
                        : (mat[u][v] != 0.0);
    }

    // O(V) 遍历邻居（必须扫整行）
    std::vector<int> neighbors(int u) const {
        std::vector<int> res;
        for (int v = 0; v < n; ++v)
            if (has_edge(u, v)) res.push_back(v);
        return res;
    }

    void print() const {
        std::cout << "   ";
        for (int j = 0; j < n; ++j) std::cout << "  " << j;
        std::cout << "\n";
        for (int i = 0; i < n; ++i) {
            std::cout << i << " [";
            for (int j = 0; j < n; ++j) {
                if (mat[i][j] == INF) std::cout << "  ∞";
                else std::cout << "  " << (int)mat[i][j];
            }
            std::cout << " ]\n";
        }
    }
};

int main() {
    AdjMatrix g(4, false, false);
    g.add_edge(0, 1);
    g.add_edge(0, 2);
    g.add_edge(1, 3);
    g.print();
    std::cout << "\n0-1 有边？" << g.has_edge(0, 1) << "\n";   // 1
    std::cout << "0-3 有边？" << g.has_edge(0, 3) << "\n";   // 0

    auto nb = g.neighbors(0);
    std::cout << "顶点0的邻居: ";
    for (int v : nb) std::cout << v << " ";   // 1 2
    std::cout << "\n";
    return 0;
}
```

---

### 18.2.2 邻接表（Adjacency List）

**核心思想**：为每个顶点 $u$ 维护一个列表（链表/动态数组），记录所有从 $u$ 出发的边的终点（以及可选的边权重）。整体结构是"顶点数组 + 各自的邻居列表"。

**邻接表的核心特性**：
- **空间效率极高**：总空间为 $O(V + E)$，仅存储实际存在的边。对于稀疏图，比邻接矩阵的 $O(V^2)$ 节省数百倍空间。
- **遍历邻居高效**：遍历顶点 $u$ 的所有邻居只需 $O(\deg(u))$ 时间（$\deg(u)$ 为 $u$ 的度数），而不是 $O(V)$。
- **判断边慢**：判断 $(u, v)$ 是否存在需要扫描 $u$ 的邻居列表，最坏 $O(\deg(u))$。

**两种常见实现方式**：

| 实现 | 优点 | 缺点 |
|---|---|---|
| `vector<vector<int>>` / `dict[list]` | 随机访问邻居 $O(1)$，缓存友好 | 删边慢（需线性查找）|
| 链表（`linked list`）| 删边 $O(1)$（已知指针时）| 随机访问慢，指针开销大 |
| `unordered_set` per vertex | 判边 $O(1)$，删边 $O(1)$ | 哈希开销，顺序不确定 |

**实际工程中**，`vector<vector<int>>` 是竞赛与面试的标准选择；Python 中用 `defaultdict(list)` 或 `dict[set]`。

```python
# ── 邻接表：三种 Python 风格实现 ──────────────────────────────────

# 方式1：defaultdict(list)——最常用，顶点可以是任意可哈希类型
from collections import defaultdict

adj1: dict = defaultdict(list)
# 添加有向边 0→1, 0→2, 1→3
for u, v in [(0, 1), (0, 2), (1, 3)]:
    adj1[u].append(v)
print(dict(adj1))   # {0: [1, 2], 1: [3]}

# 方式2：固定大小的列表索引——顶点必须是 0..n-1 的整数，竞赛常用
n = 4
adj2 = [[] for _ in range(n)]
for u, v in [(0, 1), (0, 2), (1, 3)]:
    adj2[u].append(v)
print(adj2)         # [[1, 2], [3], [], []]

# 方式3：set 表示——快速判断边是否存在，但遍历顺序不稳定
adj3: dict = defaultdict(set)
for u, v in [(0, 1), (0, 2), (1, 3)]:
    adj3[u].add(v)
print(0 in adj3[1])       # False（没有 1→0 的边）
print(1 in adj3[0])       # True（有 0→1 的边）

# ── 加权邻接表 ─────────────────────────────────────────────────────
# adj_w[u] = [(v, weight), ...]
adj_w = defaultdict(list)
adj_w[0].append((1, 5))   # 0→1，权重 5
adj_w[0].append((2, 3))   # 0→2，权重 3
adj_w[1].append((3, 7))   # 1→3，权重 7

# 典型用法（如 Dijkstra）：
for v, w in adj_w[0]:
    print(f"  0 → {v}，权重 {w}")
# 输出：0 → 1，权重 5
#       0 → 2，权重 3
```
```cpp
// ── C++ 邻接表：三种常见写法 ─────────────────────────────────────
#include <iostream>
#include <vector>
#include <list>
#include <unordered_set>
using namespace std;

int main() {
    int n = 4;
    vector<pair<int,int>> edges = {{0,1},{0,2},{1,3}};  // 有向边列表

    // ── 方式1：vector<vector<int>>（竞赛/面试首选）───────────────
    vector<vector<int>> adj1(n);
    for (auto [u, v] : edges) adj1[u].push_back(v);
    // adj1[0] = {1, 2}, adj1[1] = {3}

    // ── 方式2：vector<list<int>>（链表，支持 O(1) 删边）─────────
    vector<list<int>> adj2(n);
    for (auto [u, v] : edges) adj2[u].push_back(v);

    // ── 方式3：vector<unordered_set<int>>（O(1) 判边）──────────
    vector<unordered_set<int>> adj3(n);
    for (auto [u, v] : edges) adj3[u].insert(v);
    cout << "0→1 存在？" << adj3[0].count(1) << "\n";  // O(1), 输出 1

    // ── 加权邻接表：pair<int,int> = {邻居, 权重} ─────────────────
    using Edge = pair<int, int>;
    vector<vector<Edge>> adj_w(n);
    adj_w[0].emplace_back(1, 5);   // 0→1，权重 5
    adj_w[0].emplace_back(2, 3);   // 0→2，权重 3
    adj_w[1].emplace_back(3, 7);   // 1→3，权重 7

    // 遍历 0 的邻居（Dijkstra 的核心操作）
    for (auto [v, w] : adj_w[0])
        cout << "0 → " << v << ", 权重 " << w << "\n";
    // 输出：0 → 1, 权重 5
    //       0 → 2, 权重 3

    return 0;
}
```

---

### 18.2.3 两种表示的操作复杂度全面对比

这是选择存储结构的核心依据，务必深刻理解每一行的原因：

| 操作 | 邻接矩阵 | 邻接表（向量）| 邻接表（哈希集）| 解释 |
|---|:---:|:---:|:---:|---|
| **初始化（空图）** | $O(V^2)$ | $O(V)$ | $O(V)$ | 矩阵需要初始化整个 $V^2$ 区域 |
| **添加一条边** | $O(1)$ | $O(1)$ 均摊 | $O(1)$ 均摊 | 矩阵直接写格子；列表追加或插入哈希 |
| **删除一条边** | $O(1)$ | $O(\deg(u))$ | $O(1)$ 均摊 | 矩阵直接清零；列表需线性查找 |
| **判断边 $(u,v)$ 是否存在** | $O(1)$ ⭐ | $O(\deg(u))$ | $O(1)$ ⭐ | 矩阵直接读；列表需扫描 |
| **遍历顶点 $u$ 的所有邻居** | $O(V)$ | $O(\deg(u))$ ⭐ | $O(\deg(u))$ ⭐ | 矩阵需扫整行；列表只看实际邻居 |
| **遍历所有边** | $O(V^2)$ | $O(V + E)$ ⭐ | $O(V + E)$ ⭐ | BFS/DFS 的时间复杂度来源于此 |
| **空间占用** | $O(V^2)$ | $O(V + E)$ ⭐ | $O(V + E)$ ⭐ | 稀疏图时矩阵浪费 $V^2 - E$ 个格子 |
| **获取顶点 $u$ 的度数** | $O(V)$（扫行）| $O(1)$（存 size）| $O(1)$（存 size）| 列表长度就是度数 |
| **转置图（反转所有边）** | $O(V^2)$（转置矩阵）| $O(V + E)$ | $O(V + E)$ | Kosaraju 算法需要计算反转图 |

> ⭐ 表示该操作中该结构有优势。

**直觉总结**：
- 邻接矩阵适合**"稠密图 + 频繁判断某两顶点是否直接相连"**的场景，如 Floyd-Warshall 全源最短路。
- 邻接表适合**"稀疏图 + BFS/DFS/Dijkstra 等遍历算法"**，这是绝大多数竞赛和工业图算法的场景。

---

<!-- 图表示切换组件：同一图在三种表示之间互动切换 -->
<div data-component="GraphRepresentationToggle"></div>

---

### 18.2.4 稀疏图选邻接表、稠密图选邻接矩阵——工程决策指南

**量化判断标准**：

给定 $V$ 个顶点和 $E$ 条边：
- 若 $E < V \cdot \log V$：**强烈推荐邻接表**（稀疏，矩阵浪费严重）
- 若 $E \approx V^2 / 4$ 以上：**可以考虑邻接矩阵**（稠密，查边 $O(1)$ 的优势更明显）

**实际场景建议**：

| 场景 | V 典型值 | E 典型值 | 推荐 | 原因 |
|---|---|---|---|---|
| 社交网络 | $10^8$ | $10^9$（平均 10 个朋友）| 邻接表 | $E / V^2 \approx 10^{-7}$，极稀疏 |
| 路网 | $10^6$（路口）| $3 \times 10^6$（路段）| 邻接表 | 平均每路口 3 条路，稀疏 |
| 密码哈西图 | 100 | 4950（近完全图）| 邻接矩阵 | 接近 $V^2/2$ |
| 竞赛题，$V \le 10^3$ | $\le 10^3$ | 未知 | 邻接矩阵 | $10^6$ 空间可接受，写法简单 |
| 竞赛题，$V \le 10^5$ | $\le 10^5$ | $\le 10^6$ | 邻接表 | 矩阵 $10^{10}$ 会 MLE |

**⚠️ 竞赛常见错误**：当顶点数 $V = 10^5$ 时申请 `int adj[100005][100005]`，会占用约 $4 \times 10^{10}$ 字节（40 GB！），直接运行时内存错误（MLE）。

---

<!-- 性能对比组件：邻接矩阵 vs 邻接表随 V/E 变化的性能曲线 -->
<div data-component="AdjMatrixVsListPerf"></div>

---

### 18.2.5 边集（Edge List）表示

**核心思想**：最简洁的图表示——只存储一个**边的列表**，每个元素为 $(u, v)$ 或 $(u, v, w)$（加权）：

```
边集示例：{ (0,1), (0,2), (1,3), (2,3) }  →  代表 4 条边
```

**空间复杂度**：$O(E)$（不存储顶点信息，只存边）

**优点**：
- 实现极简单，边的枚举直接遍历列表
- 对于某些以"枚举所有边"为核心的算法非常合适

**缺点**：
- 无法 $O(1)$ 判断边是否存在（需要 $O(E)$ 扫描）
- 无法快速获取某顶点的所有邻居（需要 $O(E)$ 过滤）

**典型应用——Kruskal 最小生成树算法**：Kruskal 算法核心步骤是"按权重**排序**所有边，然后依次处理"，正好需要把所有边放在一个列表里排序，邻接表/矩阵反而不方便！

```python
# ── 边集表示 + Kruskal 最小生成树（预热）─────────────────────────
from typing import NamedTuple

class Edge(NamedTuple):
    w: float  # 权重（排序依据）
    u: int    # 起点
    v: int    # 终点

# 边集：直接保存所有边
# 设计考量：使用 NamedTuple 让 w < w' 的比较自动按权重排序
edges: list[Edge] = [
    Edge(4, 0, 1),    # 边 0-1，权重 4
    Edge(2, 0, 2),    # 边 0-2，权重 2
    Edge(5, 1, 2),    # 边 1-2，权重 5
    Edge(1, 1, 3),    # 边 1-3，权重 1
    Edge(3, 2, 3),    # 边 2-3，权重 3
]

# Kruskal 第一步：按权重排序（得益于边集格式）
edges.sort()  # 结果：权重 1, 2, 3, 4, 5 的顺序
for e in edges:
    print(f"  边 {e.u}-{e.v}，权重 {e.w}")
# 输出：
#   边 1-3，权重 1
#   边 0-2，权重 2
#   边 2-3，权重 3
#   边 0-1，权重 4
#   边 1-2，权重 5
```
```cpp
// ── 边集表示 + Kruskal 最小生成树（预热）────────────────────────
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct Edge {
    double w;   // 权重（排序依据）
    int u, v;   // 端点
    // 重载 < 运算符：按权重升序排序
    bool operator<(const Edge& o) const { return w < o.w; }
};

int main() {
    // 边集：直接存储所有边（无需知道顶点总数）
    vector<Edge> edges = {
        {4, 0, 1}, {2, 0, 2}, {5, 1, 2}, {1, 1, 3}, {3, 2, 3}
    };

    // Kruskal 第一步：按权重排序（边集格式的天然优势）
    sort(edges.begin(), edges.end());

    for (auto [w, u, v] : edges)
        cout << "边 " << u << "-" << v << "，权重 " << w << "\n";
    // 输出：
    // 边 1-3，权重 1
    // 边 0-2，权重 2
    // 边 2-3，权重 3
    // 边 0-1，权重 4
    // 边 1-2，权重 5
    return 0;
}
```

**三种表示方式对比速查表**：

| 比较维度 | 邻接矩阵 | 邻接表（向量）| 边集 |
|---|:---:|:---:|:---:|
| 空间复杂度 | $O(V^2)$ | $O(V + E)$ | $O(E)$ |
| 判断边是否存在 | $O(1)$ | $O(\deg(u))$ | $O(E)$ |
| 遍历某顶点邻居 | $O(V)$ | $O(\deg(u))$ | $O(E)$ |
| 枚举所有边 | $O(V^2)$ | $O(V + E)$ | $O(E)$ ⭐ |
| 适用算法 | Floyd-Warshall | BFS/DFS/Dijkstra/... | Kruskal |
| 代码简洁性 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

---

## 18.3 图的基本性质

### 18.3.1 握手定理（Handshaking Lemma）

**直觉比喻——聚会握手**：在一次聚会上，每当两个人握手一次，双方各自的"握手次数"都增加 1。聚会结束时，所有人的握手次数之和等于握手总次数的 2 倍（因为每次握手被两个人各计了一次）。

这就是图论中著名的**握手定理（Handshaking Lemma）**：

**定理**（无向图）：图 $G = (V, E)$ 中所有顶点的度数之和等于边数的两倍：

$$\sum_{v \in V} \deg(v) = 2|E|$$

**证明**：对每条边 $(u, v) \in E$，它各自为 $u$ 和 $v$ 的度数贡献 1，因此所有顶点的度数之和 $= \sum_{(u,v) \in E} 2 = 2|E|$。$\square$

**重要推论**：

1. **任意图中，度数为奇数的顶点个数必为偶数**（因为奇数个奇数之和是奇数，与 $2|E|$ 为偶数矛盾）。这在图论证明中常被用到！

2. 给定 $n$ 个顶点，如果每个顶点的度数都是 $d$，则边数 $|E| = \frac{nd}{2}$（$n \cdot d$ 必须是偶数——$n$ 和 $d$ 中至少有一个是偶数）。

**应用示例**：

```
验证"朋友圈"的握手定理

   A - B   度数序列：A=2, B=3, C=2, D=3, E=2
   |\ |    度数之和 = 2+3+2+3+2 = 12
   | \|    边数 = 6
   C - D   2|E| = 12 ✓
   |
   E

度数之和 = 2 × 边数   握手定理成立！
```

```python
# ── 验证握手定理 ─────────────────────────────────────────────────
def verify_handshaking(adj: dict) -> None:
    """验证无向图的握手定理：∑deg(v) = 2|E|。"""
    # 计算每个顶点的度数
    degree_sum = sum(len(neighbors) for neighbors in adj.values())
    # 无向图中，每条边被计了两次（正确！）
    num_edges = degree_sum // 2
    print(f"所有顶点度数之和：{degree_sum}")
    print(f"实际边数：{num_edges}")
    print(f"握手定理验证：{degree_sum} == 2 × {num_edges}？→ {degree_sum == 2 * num_edges}")

# 构造示例图
adj = {
    'A': ['B', 'C'],
    'B': ['A', 'C', 'D'],
    'C': ['A', 'B', 'D'],
    'D': ['B', 'C', 'E'],
    'E': ['D'],
}
verify_handshaking(adj)
# 输出：
# 所有顶点度数之和：10
# 实际边数：5
# 握手定理验证：10 == 2 × 5？→ True
```
```cpp
// ── 验证握手定理 ─────────────────────────────────────────────────
#include <iostream>
#include <vector>
using namespace std;

void verify_handshaking(const vector<vector<int>>& adj) {
    int degree_sum = 0;
    for (const auto& neighbors : adj)
        degree_sum += (int)neighbors.size();
    // 无向图：每条边贡献两次度数（正确！）
    int num_edges = degree_sum / 2;
    cout << "度数之和：" << degree_sum << "\n";
    cout << "边数：" << num_edges << "\n";
    cout << "握手定理：" << degree_sum << " == 2 × "
         << num_edges << " → " << (degree_sum == 2 * num_edges ? "✓" : "✗") << "\n";
}

int main() {
    // 5 个顶点的无向图 {A,B,C,D,E}，编号 0-4
    vector<vector<int>> adj(5);
    auto add = [&](int u, int v) { adj[u].push_back(v); adj[v].push_back(u); };
    add(0, 1); add(0, 2); add(1, 2); add(1, 3); add(3, 4);
    verify_handshaking(adj);
    // 输出：
    // 度数之和：10
    // 边数：5
    // 握手定理：10 == 2 × 5 → ✓
    return 0;
}
```

---

### 18.3.2 有向图的入度与出度

在有向图中，"度数"概念分裂为两个方向：

**出度（Out-degree，$\deg^+(v)$）**：顶点 $v$ 发出的有向边数目（你关注了多少人）。

**入度（In-degree，$\deg^-(v)$）**：指向顶点 $v$ 的有向边数目（有多少人关注了你）。

**有向图的握手定理**：

$$\sum_{v \in V} \deg^+(v) = \sum_{v \in V} \deg^-(v) = |E|$$

直觉：每条有向边 $(u \to v)$ 恰好贡献 $u$ 的出度 1 次，贡献 $v$ 的入度 1 次。因此所有出度之和 $=$ 所有入度之和 $=$ 边数。

**入度/出度在算法中的重要性**：

| 应用 | 使用 | 原因 |
|---|---|---|
| **拓扑排序（Kahn 算法）** | 入度 | 入度为 0 的顶点是"起点"（无前驱依赖）|
| **PageRank** | 入度 | 入度高的网页更"权威" |
| **欧拉回路（有向图）** | 入度 = 出度 | 欧拉回路存在的充要条件之一 |
| **源点/汇点识别** | 入度/出度为 0 | DAG 中源点（入度0）和汇点（出度0）|

```python
# ── 计算有向图的入度和出度 ────────────────────────────────────────
from collections import defaultdict

def compute_degrees(n: int, edges: list[tuple[int, int]]):
    """计算有向图中每个顶点的入度和出度。"""
    out_deg = [0] * n   # 出度数组
    in_deg  = [0] * n   # 入度数组

    for u, v in edges:
        out_deg[u] += 1  # u 多发出一条边
        in_deg[v]  += 1  # v 多收到一条边

    print(f"{'顶点':>4} {'出度':>6} {'入度':>6}")
    print("-" * 20)
    for i in range(n):
        marker = " ← 起点（入度为0）" if in_deg[i] == 0 else ""
        marker += " ← 汇点（出度为0）" if out_deg[i] == 0 else ""
        print(f"{i:>4} {out_deg[i]:>6} {in_deg[i]:>6}{marker}")

    print(f"\n出度之和 = {sum(out_deg)}，入度之和 = {sum(in_deg)}，边数 = {len(edges)}")

# 示例：DAG（依赖关系）
# 编译顺序：0→1, 0→2, 1→3, 2→3, 3→4
compute_degrees(5, [(0,1),(0,2),(1,3),(2,3),(3,4)])
# 输出：
# 顶点   出度   入度
# --------------------
#    0      2      0 ← 起点（入度为0）
#    1      1      1
#    2      1      1
#    3      1      2
#    4      0      1 ← 汇点（出度为0）
#
# 出度之和 = 5，入度之和 = 5，边数 = 5
```
```cpp
// ── 计算有向图的入度和出度 ─────────────────────────────────────
#include <iostream>
#include <vector>
using namespace std;

void compute_degrees(int n, const vector<pair<int,int>>& edges) {
    vector<int> out_deg(n, 0), in_deg(n, 0);
    for (auto [u, v] : edges) {
        ++out_deg[u];
        ++in_deg[v];
    }
    cout << "顶点  出度  入度\n";
    cout << "---------------\n";
    for (int i = 0; i < n; ++i) {
        cout << "  " << i << "     " << out_deg[i] << "     " << in_deg[i];
        if (in_deg[i] == 0)  cout << "  ← 起点";
        if (out_deg[i] == 0) cout << "  ← 汇点";
        cout << "\n";
    }
    int sum_out = 0, sum_in = 0;
    for (int i = 0; i < n; ++i) { sum_out += out_deg[i]; sum_in += in_deg[i]; }
    cout << "\n出度之和=" << sum_out << ", 入度之和=" << sum_in
         << ", 边数=" << (int)edges.size() << "\n";
}

int main() {
    compute_degrees(5, {{0,1},{0,2},{1,3},{2,3},{3,4}});
    return 0;
}
```

---

### 18.3.3 路径、简单路径、环（圈）、树与森林的精确定义

图论中的这些概念有**严格的数学定义**，初学者容易混淆，务必逐一区分：

#### 路径（Walk vs Path vs Trail）

| 术语 | 定义 | 允许重复顶点？| 允许重复边？|
|---|---|:---:|:---:|
| **游走（Walk）** | 顶点与边的交替序列 $v_0, e_1, v_1, e_2, \ldots, v_k$ | ✅ | ✅ |
| **迹（Trail）** | 不重复**边**的游走 | ✅ | ❌ |
| **路径 / 简单路径（Path）** | 不重复**顶点**（也不重复边）的游走 | ❌ | ❌ |

> **CLRS 约定**：在大多数算法教材中，"路径（path）"默认指**简单路径**（不重复顶点），后续章节均使用此约定。

**路径长度（Path Length）**：
- 无权图：路径中的**边数**
- 加权图：路径上所有**边权之和**

#### 环（Cycle）vs 回路（Circuit）

| 术语 | 定义 |
|---|---|
| **回路（Closed Walk）** | 起点 = 终点的游走（顶点和边均可重复）|
| **环 / 简单环（Cycle）** | 起点 = 终点，且中间所有顶点互不相同（长度 ≥ 3）|
| **奇圈 / 偶圈** | 边数为奇数/偶数的简单环（与二部图有关）|

**有向环（Directed Cycle）**：有向图中，按边方向行进能回到起点的环。DAG（有向无环图）中不存在有向环。

> **易混淆点**：2条边的"来回"（$u \to v \to u$）在有向图中形成一个长度为2的有向环；在**无向图**中，$(u,v)$ 和 $(v,u)$ 是同一条边，不构成环（简单图中无重边，所以无向图中环的长度至少为3）。

#### 树（Tree）与森林（Forest）

| 术语 | 定义 | 边数 | 连通性 |
|---|---|:---:|:---:|
| **树（Tree）** | 连通且无环的无向图 | $n-1$ | 连通 |
| **森林（Forest）** | 无环的无向图（可以不连通）| $\le n-1$ | 可不连通 |
| **根树（Rooted Tree）** | 一个顶点指定为根的树 | $n-1$ | 连通 |

**树的等价定义**（以下四条互相等价，证任意一条即可推出其余三条）：
1. 连通且无环
2. 连通且恰好有 $n-1$ 条边
3. 无环且恰好有 $n-1$ 条边
4. 任意两顶点之间恰好有一条简单路径

**图论中的树 vs 数据结构中的树**：

| | 图论的树 | CS 数据结构的树 |
|---|---|---|
| 方向 | 无向 | 通常有"父→子"方向 |
| 根 | 不必须（可以后指定）| 必须有根节点 |
| 度数约束 | 无（非根的树叶度数为1）| 子节点数受限（如二叉树最多2个子节点）|

---

### 18.3.4 图的同构（Graph Isomorphism）基础

**直觉问题**：下面两个图"长得一样"吗？

```
图 G1：                 图 G2：
  1 — 2                   A — B
  |       |                   |       |
  3 — 4               C — D
  （正方形）           （正方形）
```

直觉上"一样"——只是顶点换了名字（$1 \leftrightarrow A, 2 \leftrightarrow B$, ...）。这就是**图同构**的核心思想。

**形式化定义**：图 $G_1 = (V_1, E_1)$ 与 $G_2 = (V_2, E_2)$ **同构（Isomorphic）**，记作 $G_1 \cong G_2$，当且仅当存在一个**双射（bijection）** $f: V_1 \to V_2$，使得：

$$\forall u, v \in V_1: (u, v) \in E_1 \iff (f(u), f(v)) \in E_2$$

即：$f$ 保持了所有的"邻接关系"（边对应到边，非边对应到非边）。

**同构的必要条件（快速筛查）**：
1. 顶点数相同：$|V_1| = |V_2|$
2. 边数相同：$|E_1| = |E_2|$
3. **度序列（Degree Sequence）相同**：将两图的度数序列排序后相等（这是最有用的筛查条件！）
4. 对应连通分量数相同，且对应分量的度序列也相同

> ⚠️ **注意**：上述条件只是必要条件，不是充分条件——满足这些条件的两个图**不一定**同构！

**图同构判定问题（Graph Isomorphism Problem，GI）**：  
给定两个图，判断它们是否同构。这是理论计算机科学中著名的"既不知道是 P，也不知道是 NP-Complete"的问题，目前已知有拟多项式时间算法（Babai，2015）。在实际应用中（如化学分子指纹识别、模式匹配），图同构问题非常重要。

```python
# ── 图同构的必要条件检验 ──────────────────────────────────────────
from collections import Counter

def degree_sequence(adj: dict) -> list[int]:
    """返回排序后的度序列（快速检验同构必要条件）。"""
    return sorted(len(neighbors) for neighbors in adj.values())

def might_be_isomorphic(adj1: dict, adj2: dict) -> bool:
    """快速排除不可能同构的图对（必要条件检验）。"""
    ds1 = degree_sequence(adj1)
    ds2 = degree_sequence(adj2)

    if len(adj1) != len(adj2):
        print("顶点数不同 → 一定不同构")
        return False

    total_edges1 = sum(ds1) // 2
    total_edges2 = sum(ds2) // 2
    if total_edges1 != total_edges2:
        print("边数不同 → 一定不同构")
        return False

    if ds1 != ds2:
        print(f"度序列不同：{ds1} vs {ds2} → 一定不同构")
        return False

    print(f"度序列相同：{ds1} → 可能同构（需进一步验证）")
    return True  # 仅是必要条件，真正同构需暴力枚举或专用算法

# ── 测试 ───────────────────────────────────────────────────────────

# 两个 4-环图（实际上同构，都是正方形）
G1 = {1: [2,4], 2: [1,3], 3: [2,4], 4: [3,1]}
G2 = {'A': ['B','D'], 'B': ['A','C'], 'C': ['B','D'], 'D': ['C','A']}
might_be_isomorphic(G1, G2)
# 输出：度序列相同：[2, 2, 2, 2] → 可能同构（需进一步验证）

# 一个 4-环 vs 一个完全图 K4
K4 = {0:[1,2,3], 1:[0,2,3], 2:[0,1,3], 3:[0,1,2]}
might_be_isomorphic(G1, K4)
# 输出：边数不同 → 一定不同构
```
```cpp
// ── 图同构的必要条件检验 ──────────────────────────────────────────
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

// 计算度序列（排序后）
vector<int> degree_sequence(const vector<vector<int>>& adj) {
    vector<int> deg;
    for (const auto& nb : adj) deg.push_back((int)nb.size());
    sort(deg.begin(), deg.end());
    return deg;
}

bool might_be_isomorphic(const vector<vector<int>>& g1,
                          const vector<vector<int>>& g2) {
    if (g1.size() != g2.size()) {
        cout << "顶点数不同 → 一定不同构\n";
        return false;
    }
    auto ds1 = degree_sequence(g1);
    auto ds2 = degree_sequence(g2);
    // 边数（度数之和的一半）
    int e1 = 0, e2 = 0;
    for (int d : ds1) e1 += d; e1 /= 2;
    for (int d : ds2) e2 += d; e2 /= 2;
    if (e1 != e2) {
        cout << "边数不同 → 一定不同构\n";
        return false;
    }
    if (ds1 != ds2) {
        cout << "度序列不同 → 一定不同构\n";
        return false;
    }
    cout << "度序列相同 → 可能同构（需进一步验证）\n";
    return true;
}

int main() {
    // G1：4-环  0-1-2-3-0
    vector<vector<int>> G1 = {{1,3},{0,2},{1,3},{2,0}};
    // G2：4-环（顶点顺序不同）  0-2-1-3-0
    vector<vector<int>> G2 = {{2,3},{2,3},{0,1},{0,1}};
    // K4：完全图
    vector<vector<int>> K4 = {{1,2,3},{0,2,3},{0,1,3},{0,1,2}};

    might_be_isomorphic(G1, G2);   // 可能同构
    might_be_isomorphic(G1, K4);   // 边数不同
    return 0;
}
```

---

## 小结：Chapter 18 知识图谱

### 图的分类速查表

```
图 G = (V, E)
│
├── 按方向性
│   ├── 无向图（Undirected）：边 = {u,v}，邻接表每条边存两次
│   └── 有向图（Digraph）  ：边 = (u→v)，有入度和出度之分
│
├── 按权重
│   ├── 无权图（Unweighted）：只关心边的存在性
│   └── 加权图（Weighted）  ：边附带数值权重 w
│
├── 按结构
│   ├── 简单图：无自环，无重边（默认）
│   ├── 多重图：允许重边
│   ├── 完全图 K_n：任意两顶点有边，|E| = n(n-1)/2
│   ├── 树 / 森林：无环，|E| = n-1（连通）或 < n-1（森林）
│   └── DAG：有向 + 无环，必有拓扑排序
│
└── 按连通性
    ├── 无向图：连通图 / 非连通（有多个连通分量）
    └── 有向图：强连通 / 弱连通（忽略方向后连通）
```

### 存储结构选择决策树

```
需要选择图的存储结构？
│
├── 顶点数量 > 10⁴ 且边较少？
│   └── ✅ 邻接表（节省空间）
│
├── 边数接近 V² / 图较小（V ≤ 10³）？
│   └── ✅ 邻接矩阵（判边快，代码简洁）
│
├── 算法核心是"排序所有边"（如 Kruskal）？
│   └── ✅ 边集（Edge List）
│
└── 默认首选：邻接表（适应大多数情况）
```

### 核心公式与定理速查

| 定理/性质 | 公式 | 适用范围 |
|---|---|---|
| 握手定理 | $\sum \deg(v) = 2\|E\|$ | 无向图 |
| 有向握手定理 | $\sum \deg^+(v) = \sum \deg^-(v) = \|E\|$ | 有向图 |
| 完全图边数 | $\|E\| = \binom{n}{2} = \frac{n(n-1)}{2}$ | 无向完全图 $K_n$ |
| 树的边数 | $\|E\| = \|V\| - 1$ | 连通无环图 |
| 有向图入度=出度 | 充要条件之一 | 欧拉回路 |

### 常见陷阱与注意事项

1. **⚠️ 无向图邻接表忘记存反向边**：添加 $(u, v)$ 时必须同时添加 $(v, u)$，否则 BFS/DFS 等算法会遗漏路径。

2. **⚠️ 大规模图申请邻接矩阵 MLE**：当 $V = 10^5$ 时，`int adj[100005][100005]` 需要约 $4 \times 10^{10}$ 字节，必定超内存。

3. **⚠️ 混淆"有向图环"与"无向图环"**：无向图中环的最小长度是 3（3条边）；有向图中两条边 $u \to v \to u$ 就构成长度为 2 的有向环。

4. **⚠️ 握手定理的推论**：任何图（含有向图）中，奇度顶点的个数必为偶数。这经常用于判断欧拉路径是否存在。

5. **⚠️ 同构 ≠ 相同**：两个图同构意味着它们的"结构"相同，但顶点有不同名字。度序列相同是同构的必要不充分条件。

6. **⚠️ 稀疏图用 BFS/DFS 的复杂度**：$O(V + E)$，与边数严格相关（而不是 $O(V^2)$）——这正是为什么邻接表在遍历算法中比矩阵更高效的原因。

---

### 经典 LeetCode 导引

| 题号 | 题名 | 核心知识点 | 难度 |
|---|---|---|:---:|
| #547 | 省份数量 | 无向图连通分量（DFS/BFS/并查集）| 🟡 |
| #200 | 岛屿数量 | 网格图 DFS，等价于无向图连通分量 | 🟡 |
| #997 | 找到小镇的法官 | 入度/出度分析 | 🟢 |
| #1971 | 寻找图中是否存在路径 | 连通性判断（DFS/BFS/并查集）| 🟢 |
| #841 | 钥匙和房间 | 有向图可达性 | 🟡 |
| #207 | 课程表 | 有向图环检测，DAG 判定 | 🟡 |
| #684 | 冗余连接 | 树 + 多一条边 = 一个环（并查集）| 🟡 |

---

### 思考题

1. **握手定理的应用**：一个 10 人聚会中，每人与奇数个人握了手，这是否可能？为什么？（提示：结合握手定理的推论）

2. **存储结构选择**：对 Dijkstra 最短路算法，当图稀疏（$E = O(V \log V)$）时，邻接表的优势在这个算法中体现在哪一步？

3. **DAG 的性质**：一个有向图中所有顶点的入度都大于 0，这样的图一定有环吗？请给出证明或反例。（提示：考虑有最小入度顶点的路径会发生什么）

# Chapter 23: 全对最短路径（All-Pairs Shortest Paths）

## 章节导读

在 Chapter 21 和 Chapter 22 中，我们系统学习了**单源最短路径（SSSP）**问题——从一个固定源点出发，求到所有其他顶点的最短路径。但在很多实际场景中，我们需要的不是从某一个点出发的最短路，而是**图中任意两点之间的最短路径**：

- **交通导航系统**：城市 A 到城市 B、城市 C 到城市 D……每对城市之间都需要最优路线。
- **网络路由**：路由器需要知道从任意节点到任意节点的最短跳数，以构建完整路由表。
- **关系网络分析**：社交网络中任意两人的"最短关系距离"（六度分隔理论）。
- **矩阵运算与图论研究**：闭包计算、关系传递性判断等。

这就是**全对最短路径（All-Pairs Shortest Paths，APSP）**问题。

**一个朴素方案**：对每个顶点分别跑一遍 SSSP。如果用 Bellman-Ford，总时间为 $O(V^2 E)$；如果图无负权且用 Dijkstra + 二叉堆，总时间为 $O(VE \log V)$。这两种方法都不够优雅，而且无法充分利用"所有对"这一整体性质。本章介绍三种专门针对 APSP 的经典算法，各有其设计哲学和适用场景。

| 算法 | 思路 | 时间复杂度 | 适用条件 |
|:---:|:---:|:---:|:---:|
| Floyd-Warshall | 动态规划（DP） | $\Theta(V^3)$ | 允许负权，检测负权环 |
| Johnson | 重新定权 + V次Dijkstra | $O(VE + V^2 \log V)$ | 允许负权（无负权环），稀疏图有优势 |
| 重复平方 | (min,+) 矩阵乘法 | $O(V^3 \log V)$ | 理论意义，实践较少 |

---

## 23.1 Floyd-Warshall 算法

### 23.1.1 直觉与 DP 状态设计

**快递中转站类比**：想象有 $n$ 座城市，城市之间有公路（可能有收费，费用可正可负，例如优惠券）。现在要计算任意两城市间的最低费用路线。

最朴素的思路是：枚举所有可能的中转城市。如果我们只能通过城市 1 作为中转，能不能减少 A→B 的费用？如果再加入城市 2 可以中转，又能不能更短？……逐步引入每一座城市作为潜在中转站，直到引入所有城市。

这正是 Floyd-Warshall 的核心思想：**动态规划**。

**DP 状态定义**：

$$d_{ij}^{(k)} = \text{从顶点 } i \text{ 到顶点 } j \text{，中间节点只从集合 } \{1, 2, \ldots, k\} \text{ 中选取时的最短路径长度}$$

注意几个关键点：
- "中间节点"指路径 $i \to \cdots \to j$ 上除端点 $i$ 和 $j$ 之外的所有顶点。
- 端点 $i$ 和 $j$ 本身可以是任意编号，不受 $k$ 限制。
- 当 $k = 0$ 时，不允许任何中间节点，所以 $d_{ij}^{(0)}$ 就是原图中 $i$ 到 $j$ 的直接边权（若无直接边则为 $+\infty$，若 $i = j$ 则为 $0$）。

**初始化（$k=0$）**：

$$d_{ij}^{(0)} = \begin{cases} 0 & \text{若 } i = j \\ w(i, j) & \text{若 } (i,j) \in E \\ +\infty & \text{若 } (i,j) \notin E \end{cases}$$

**目标**：求出 $d_{ij}^{(V)}$，即中间节点可以是任意顶点时的最短路径——这正是我们想要的 APSP 答案。

### 23.1.2 状态转移方程

在计算 $d_{ij}^{(k)}$ 时，可以问一个关键问题：**从 $i$ 到 $j$，且中间节点 $\subseteq \{1, \ldots, k\}$ 的最短路径，是否经过顶点 $k$？**

**情况一：不经过顶点 $k$**
路径的中间节点仍然 $\subseteq \{1, \ldots, k-1\}$，所以：
$$d_{ij}^{(k)} = d_{ij}^{(k-1)}$$

**情况二：经过顶点 $k$**
路径形如 $i \to \cdots \to k \to \cdots \to j$，其中 $i$ 到 $k$ 的一段和 $k$ 到 $j$ 的一段，中间节点均 $\subseteq \{1, \ldots, k-1\}$（因为 $k$ 本身已经作为端点，不算中间节点）。所以：
$$d_{ij}^{(k)} = d_{ik}^{(k-1)} + d_{kj}^{(k-1)}$$

综合两种情况，取最小值：

$$\boxed{d_{ij}^{(k)} = \min\!\left(d_{ij}^{(k-1)},\; d_{ik}^{(k-1)} + d_{kj}^{(k-1)}\right)}$$

这个方程极其简洁，每次引入一个新的"可用中转站" $k$，检查是否能缩短某对 $(i,j)$ 之间的距离。

**手推四节点例子**：

设有向图，顶点 $\{1,2,3,4\}$，边及权值：

$$1\to2: 3,\quad 1\to4: 7,\quad 2\to3: 1,\quad 3\to1: 2,\quad 4\to3: -3,\quad 2\to4: 5,\quad 3\to4: 2$$

初始矩阵 $D^{(0)}$（$\infty$ 表示无直接边）：

|   | 1 | 2 | 3 | 4 |
|:---:|:---:|:---:|:---:|:---:|
| **1** | 0 | 3 | ∞ | 7 |
| **2** | ∞ | 0 | 1 | 5 |
| **3** | 2 | ∞ | 0 | 2 |
| **4** | ∞ | ∞ | -3 | 0 |

**k=1**（引入顶点1作为中转站）：检查 $d_{ij}^{(1)} = \min(d_{ij}^{(0)}, d_{i1}^{(0)} + d_{1j}^{(0)})$

关键变化：$d_{32}^{(1)} = \min(\infty, 2+3)=5$；$d_{34}^{(1)} = \min(2, 2+7)=2$（不变）……

**k=4**（引入顶点4作为中转）：$d_{13}^{(4)} = \min(d_{13}^{(3)}, d_{14}^{(3)} + d_{43}^{(3)})$。

完整推算会得到最终矩阵，从而得到所有对的最短路径。

<div data-component="FloydWarshallDP"></div>

### 23.1.3 三重循环实现与原地更新的正确性

Floyd-Warshall 的实现极其简洁——三重循环，外层枚举中转站 $k$，内层枚举起点 $i$ 和终点 $j$：

```python
import math

def floyd_warshall(n, edges):
    """
    Floyd-Warshall 全对最短路径
    
    Args:
        n: 顶点数（顶点编号 0 到 n-1）
        edges: 边列表 [(u, v, w), ...]
    
    Returns:
        dist: n×n 距离矩阵，dist[i][j] 为 i 到 j 的最短路径长度
        pred: n×n 前驱矩阵，pred[i][j] 为 i 到 j 最短路中 j 的前驱
    """
    INF = math.inf
    
    # 初始化距离矩阵
    dist = [[INF] * n for _ in range(n)]
    pred = [[None] * n for _ in range(n)]
    
    for i in range(n):
        dist[i][i] = 0
    
    for u, v, w in edges:
        if w < dist[u][v]:      # 处理重边取最小
            dist[u][v] = w
            pred[u][v] = u
    
    # 三重循环：外层枚举中转站 k
    for k in range(n):
        for i in range(n):
            # 剪枝：若 i 到 k 不可达，跳过（优化常数）
            if dist[i][k] == INF:
                continue
            for j in range(n):
                new_dist = dist[i][k] + dist[k][j]
                if new_dist < dist[i][j]:
                    dist[i][j] = new_dist
                    pred[i][j] = pred[k][j]  # j 的前驱继承自 k→j 路径
    
    return dist, pred


def reconstruct_path(pred, src, dst):
    """从前驱矩阵重建路径"""
    if pred[src][dst] is None:
        return None  # 不可达
    
    path = []
    cur = dst
    while cur != src:
        path.append(cur)
        cur = pred[src][cur]
        if cur is None:
            return None  # 不存在路径
    path.append(src)
    path.reverse()
    return path


# ========== 示例 ==========
if __name__ == "__main__":
    # 4 个顶点，顶点编号 0-3（对应手推例题中的 1-4）
    n = 4
    edges = [
        (0, 1, 3),   # 1→2: 3
        (0, 3, 7),   # 1→4: 7
        (1, 2, 1),   # 2→3: 1
        (2, 0, 2),   # 3→1: 2
        (3, 2, -3),  # 4→3: -3
        (1, 3, 5),   # 2→4: 5
        (2, 3, 2),   # 3→4: 2
    ]
    
    dist, pred = floyd_warshall(n, edges)
    
    print("最短路径距离矩阵：")
    for i in range(n):
        row = []
        for j in range(n):
            if dist[i][j] == math.inf:
                row.append(" ∞")
            else:
                row.append(f"{dist[i][j]:2d}")
        print(f"  顶点{i+1}: {row}")
    
    # 检测负权环：若任意 dist[i][i] < 0，存在负权环
    has_neg_cycle = any(dist[i][i] < 0 for i in range(n))
    print(f"\n是否存在负权环：{has_neg_cycle}")
    
    # 重建路径：顶点 0 → 顶点 3
    path = reconstruct_path(pred, 0, 3)
    print(f"\n顶点1 到 顶点4 的最短路径：{[p+1 for p in path]}")
    print(f"路径长度：{dist[0][3]}")

# 预期输出：
# 最短路径距离矩阵：
#   顶点1: [' 0', ' 3', ' 4', ' 6']
#   顶点2: [' 3', ' 0', ' 1', ' 3']
#   顶点3: [' 2', ' 5', ' 0', ' 2']
#   顶点4: ['-1', ' 2', '-3', ' 0']
# 是否存在负权环：False
# 顶点1 到 顶点4 的最短路径：[1, 2, 3, 4]
# 路径长度：6
```

```cpp
#include <bits/stdc++.h>
using namespace std;

const long long INF = 1e18;

struct FloydWarshall {
    int n;
    vector<vector<long long>> dist;
    vector<vector<int>>  pred;
    
    FloydWarshall(int n, vector<tuple<int,int,long long>>& edges) : n(n) {
        dist.assign(n, vector<long long>(n, INF));
        pred.assign(n, vector<int>(n, -1));
        
        for (int i = 0; i < n; i++) dist[i][i] = 0;
        
        for (auto& [u, v, w] : edges) {
            if (w < dist[u][v]) {
                dist[u][v] = w;
                pred[u][v] = u;
            }
        }
        
        // 三重循环：外层枚举中转站 k
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                if (dist[i][k] == INF) continue;  // 剪枝
                for (int j = 0; j < n; j++) {
                    if (dist[k][j] == INF) continue;
                    long long newDist = dist[i][k] + dist[k][j];
                    if (newDist < dist[i][j]) {
                        dist[i][j] = newDist;
                        pred[i][j] = pred[k][j];
                    }
                }
            }
        }
    }
    
    // 负权环检测：检查对角线
    bool hasNegCycle() const {
        for (int i = 0; i < n; i++)
            if (dist[i][i] < 0) return true;
        return false;
    }
    
    // 重建路径（返回顶点序列，若不可达返回空向量）
    vector<int> getPath(int src, int dst) const {
        if (pred[src][dst] == -1) return {};
        vector<int> path;
        int cur = dst;
        // 防止负权环导致的无限循环
        int limit = n + 1;
        while (cur != src && limit-- > 0) {
            path.push_back(cur);
            cur = pred[src][cur];
        }
        if (limit <= 0) return {};  // 检测到环
        path.push_back(src);
        reverse(path.begin(), path.end());
        return path;
    }
};

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int n = 4;
    // 边：(u, v, w)，顶点编号 0-based
    vector<tuple<int,int,long long>> edges = {
        {0, 1, 3},  // 1→2: 3
        {0, 3, 7},  // 1→4: 7
        {1, 2, 1},  // 2→3: 1
        {2, 0, 2},  // 3→1: 2
        {3, 2, -3}, // 4→3: -3
        {1, 3, 5},  // 2→4: 5
        {2, 3, 2},  // 3→4: 2
    };
    
    FloydWarshall fw(n, edges);
    
    cout << "最短路径距离矩阵：\n";
    for (int i = 0; i < n; i++) {
        cout << "  顶点" << (i+1) << ": [";
        for (int j = 0; j < n; j++) {
            if (j > 0) cout << ", ";
            if (fw.dist[i][j] == INF) cout << "∞";
            else cout << fw.dist[i][j];
        }
        cout << "]\n";
    }
    
    cout << "\n是否存在负权环：" << (fw.hasNegCycle() ? "是" : "否") << "\n";
    
    auto path = fw.getPath(0, 3);
    cout << "\n顶点1 到 顶点4 的最短路径：";
    for (int v : path) cout << (v+1) << " ";
    cout << "\n路径长度：" << fw.dist[0][3] << "\n";
    
    return 0;
}

// 预期输出：
// 最短路径距离矩阵：
//   顶点1: [0, 3, 4, 6]
//   顶点2: [3, 0, 1, 3]
//   顶点3: [2, 5, 0, 2]
//   顶点4: [-1, 2, -3, 0]
// 是否存在负权环：否
// 顶点1 到 顶点4 的最短路径：1 2 3 4
// 路径长度：6
```

**原地更新的正确性**（为什么不需要两个矩阵？）

这是 Floyd-Warshall 一个常见疑问：我们在同一个矩阵 $D$ 上原地修改，第 $k$ 轮中使用的 $d_{ik}$ 和 $d_{kj}$ 可能已经被本轮更新过了，会不会出错？

**答案：不会出错。** 关键观察：

- 在第 $k$ 轮（外层循环 $k$）中，我们检查的是 $d_{ik}$ 和 $d_{kj}$。
- $d_{ik}^{(k)}$ 和 $d_{ik}^{(k-1)}$ 相等！因为 $d_{ik}^{(k)} = \min(d_{ik}^{(k-1)}, d_{ik}^{(k-1)} + d_{kk}^{(k-1)}) = \min(d_{ik}^{(k-1)}, d_{ik}^{(k-1)} + 0) = d_{ik}^{(k-1)}$（因为 $d_{kk}^{(k-1)} \geq 0$，若无负权环则 $d_{kk}=0$）。
- 同理 $d_{kj}^{(k)} = d_{kj}^{(k-1)}$。

因此，即使在第 $k$ 轮先更新了 $d_{ij}$，之后用 $d_{ik}$ 或 $d_{kj}$ 时，它们的值也不受影响，原地更新是安全的。

> ⚡ **注意**：若图中存在负权环，则 $d_{kk}$ 可能变为负数，上述论证失效，但此时无论如何结果都是无意义的（负权环上的最短路为 $-\infty$）。

### 23.1.4 负权环检测

**何时存在负权环？** 若图中存在一个负权环，那么该环上任意顶点 $v$ 的"最短路径" $v \to \cdots \to v$（经过负权环）的长度为负数。

Floyd-Warshall 结束后，只需检查**对角线元素**：

$$\exists\, i:\; d_{ii}^{(V)} < 0 \implies \text{存在负权环}$$

若存在负权环，则某些点对之间的"最短路"为 $-\infty$（经过负权环绕无穷多圈），Floyd-Warshall 的结果对这些点对无意义。

实际工程中，可以在运行完后扫描对角线，也可以在运行中发现对角线变负就提前终止并报错。

**可达性传播**：若存在负权环，不仅环上的顶点的 $d_{ii} < 0$，还需要注意：能到达该负权环、且能从负权环到达的顶点对，其"最短路"也为 $-\infty$。实际工程中处理这种情况需要额外标记。

### 23.1.5 路径重建：前驱矩阵

距离矩阵 $D$ 只给了最短路径的**长度**，若要知道**路径本身**，需要维护**前驱矩阵** $\Pi$：

$$\pi_{ij} = \text{从 } i \text{ 到 } j \text{ 的最短路中，顶点 } j \text{ 的前驱顶点}$$

**初始化**（$k=0$ 时）：
$$\pi_{ij}^{(0)} = \begin{cases} \text{None} & \text{若 } i = j \text{ 或 } (i,j) \notin E \\ i & \text{若 } (i,j) \in E \end{cases}$$

**更新规则**（当 $d_{ij}$ 被通过 $k$ 松弛时）：
$$\pi_{ij} \leftarrow \pi_{kj}$$

含义：$i \to j$ 的最短路现在经过 $k$，而 $k \to j$ 这段的前驱信息已在 $\pi_{kj}$ 中，直接继承即可。

**路径重建**（从 $s$ 到 $t$）：

```
path ← [t]
cur ← t
while cur ≠ s:
    cur ← pred[s][cur]
    path.prepend(cur)
return path
```

时间复杂度：$O(V)$（路径长度最多 $V$）。

<div data-component="FloydPathReconstruct"></div>

### 23.1.6 复杂度分析与 DP 的自然联系

**时间复杂度**：三重循环，每层 $\Theta(V)$，总时间 $\Theta(V^3)$。注意这与 $E$ 无关——不论稀疏还是稠密图，始终是 $\Theta(V^3)$。

**空间复杂度**：距离矩阵和前驱矩阵各占 $O(V^2)$，且由于原地更新，不需要额外空间来存储上一轮的矩阵。

**与 Bellman-Ford 的 DP 视角对比**：

| 维度 | Floyd-Warshall | Bellman-Ford（固定源 $s$） |
|:---:|:---:|:---:|
| 状态变量 | $d_{ij}^{(k)}$：中间节点 $\subseteq\{1,\ldots,k\}$ | $d_v^{(m)}$：最多经过 $m$ 条边 |
| 转移逻辑 | 要不要经过节点 $k$ | 最后一条边是哪条 |
| 轮数 | $V$ 轮（每轮引入一个新中转站） | $V-1$ 轮（每轮多允许一条边） |
| 结果 | 全对 $(V^2$ 个值$)$ | 单源（$V$ 个值） |

两者都是 DP，只是状态定义不同：Bellman-Ford 的"允许 $m$ 条边"从**边的数量**维度扩展，Floyd-Warshall 的"中转节点 $\subseteq \{1,\ldots,k\}$"从**可用节点集合**维度扩展。

**Floyd 的局限性**：
- 不能处理**负权环**（结果无意义）。
- 稀疏图（$E \ll V^2$）时，$\Theta(V^3)$ 比 Johnson 算法慢（后者对稀疏图是 $O(VE + V^2 \log V)$）。
- 对于超大规模图（$V > 10^4$），$V^3$ 计算量太大。

**Floyd-Warshall 的最佳适用场景**：
- 中小规模稠密图（$V \leq 500$）。
- 需要负权边但确认无负权环。
- 代码极简，只需 10 行核心逻辑。

---

## 23.2 Johnson 算法

### 23.2.1 动机：如何在稀疏图中超越 Floyd？

对于稀疏图（$E = O(V)$ 或 $E = O(V \log V)$），Floyd-Warshall 的 $\Theta(V^3)$ 实在太慢。我们能不能对每个顶点都跑一遍高效的 Dijkstra——只需 $O(V \cdot E \log V)$ 呢？

**问题**：Dijkstra 要求所有边权**非负**，但图中可能有负权边。

**最直接想法**：给所有边权加上同一个常数 $c$（把最小边权变为0），让所有边变非负，再跑 Dijkstra。但这样做**破坏了路径权重的比较**：

边数多的路径被"惩罚"更多（加了更多个 $c$），最优路径可能改变。例如：
- 原图：$A \to B \to C$（权值 $-1 + 5 = 4$）优于 $A \to C$（权值 $6$）。
- 加常数 $c=1$ 后：$A \to B \to C$（$(0+6=6$）与 $A \to C$（$7$），虽然关系不变，但一般情况下可能颠倒。

**正确的重新定权**需要：每条边 $(u,v)$ 加上的常数依赖于端点，而不是全局统一的常数。这正是 Johnson 算法的关键创新。

### 23.2.2 重新定权（Reweighting）原理

**Johnson 的方案**：为每个顶点 $v$ 找一个"势能函数" $h(v)$，定义新权值：

$$w'(u, v) = w(u, v) + h(u) - h(v)$$

**为什么这样定义？**

设 $p = v_0, v_1, \ldots, v_k$ 是一条路径，则路径的新总权重为：
$$\sum_{i=0}^{k-1} w'(v_i, v_{i+1}) = \sum_{i=0}^{k-1}\bigl[w(v_i, v_{i+1}) + h(v_i) - h(v_{i+1})\bigr]$$

展开后大量项消去（**望远镜求和**）：
$$= \sum_{i=0}^{k-1} w(v_i, v_{i+1}) + h(v_0) - h(v_k) = \hat{d}(s, t) + h(s) - h(t)$$

其中 $\hat{d}(s,t)$ 是原图中 $s$ 到 $t$ 的路径长度。

**关键结论**：
1. **路径相对大小不变**：从 $s$ 到 $t$ 的所有路径，新权重都比原权重多 $h(s) - h(t)$（这是常量），所以最短路径的相对顺序保持不变。
2. **结果还原**：若 Dijkstra 在新图中求得 $d'(s, t)$，则原图中真实最短路 $d(s, t) = d'(s, t) - h(s) + h(t)$。

**如何保证 $w'(u,v) \geq 0$？**

需要：$w(u,v) + h(u) - h(v) \geq 0$，即：$h(v) - h(u) \leq w(u,v)$。

这正是**三角不等式**！若 $h(v)$ 是从某个**超级源点 $s'$** 到每个顶点的最短路距离，则三角不等式自动满足：

$$\delta(s', v) \leq \delta(s', u) + w(u, v)$$
$$\Rightarrow h(v) - h(u) \leq w(u, v)$$
$$\Rightarrow w(u, v) + h(u) - h(v) \geq 0 \checkmark$$

### 23.2.3 超级源点与 Bellman-Ford

**步骤：**

1. **添加超级源点 $s'$**：新增一个顶点 $s'$，从 $s'$ 向所有原图顶点 $v$ 连一条权值为 $0$ 的边。

2. **对 $s'$ 跑 Bellman-Ford**：求 $s'$ 到所有顶点的最短路 $h(v) = \delta(s', v)$。
   - 若图中存在**负权环**，Bellman-Ford 会检测到，报告"图含负权环，Johnson 无法处理"。
   - 由于 $s'$ 到所有顶点的边权为 $0$，且图无负权环，$h(v) \leq 0$ 对所有 $v$ 成立（可以直接到达，或通过负权（非负权环）路径到达）。

3. **重新计算边权**：$w'(u,v) = w(u,v) + h(u) - h(v)$，此时所有边权 $\geq 0$。

4. **移除 $s'$**，对原图的每个顶点 $u$，在新权图上跑 Dijkstra，得到 $d'(u, v)$。

5. **还原答案**：$d(u, v) = d'(u, v) - h(u) + h(v)$。

**完整时间复杂度**：
- Bellman-Ford：$O(VE)$（$V$ 个顶点，$E + V$ 条边，近似 $O(VE)$）。
- $V$ 次 Dijkstra（使用优先队列）：每次 $O(E \log V)$，共 $O(VE \log V)$。
- 整体：$O(VE + V^2 \log V)$（使用二叉堆 Dijkstra）。

若使用 Fibonacci 堆，Dijkstra 单次 $O(E + V \log V)$，Johnson 总体变为 $O(VE + V^2 \log V)$（稀疏图时：$E = O(V)$，则为 $O(V^2 \log V)$），优于 Floyd 的 $O(V^3)$。

**稀疏图对比**（$E = O(V)$）：
- Floyd：$O(V^3)$
- Johnson：$O(V^2 \log V)$，显著更快

**稠密图**（$E = O(V^2)$）：
- Floyd：$O(V^3)$
- Johnson：$O(V^3 + V^2 \log V) = O(V^3)$，相当

<div data-component="JohnsonReweighting"></div>

### 23.2.4 Johnson 算法完整实现

```python
import heapq
import math
from collections import defaultdict

def bellman_ford(graph, src, n):
    """
    Bellman-Ford 单源最短路径
    返回距离列表和是否存在负权环
    graph: 邻接表 {u: [(v, w), ...]}
    """
    INF = math.inf
    dist = [INF] * n
    dist[src] = 0
    
    # V-1 轮松弛（此处 n 包含超级源点，故 n-1 次）
    for _ in range(n - 1):
        updated = False
        for u in range(n):
            if dist[u] == INF:
                continue
            for v, w in graph[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    updated = True
        if not updated:
            break
    
    # 检测负权环（第 V 轮仍能松弛 → 存在负权环）
    for u in range(n):
        if dist[u] == INF:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                return None, True  # 存在负权环
    
    return dist, False


def dijkstra(graph, src, n):
    """
    使用优先队列的 Dijkstra（用于非负权图）
    返回从 src 到所有顶点的距离
    """
    INF = math.inf
    dist = [INF] * n
    dist[src] = 0
    pq = [(0, src)]  # (距离, 顶点)
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue  # 过期条目，跳过
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    
    return dist


def johnson(n, edges):
    """
    Johnson 全对最短路径算法
    
    Args:
        n: 原图顶点数（顶点编号 0 到 n-1）
        edges: 边列表 [(u, v, w), ...]
    
    Returns:
        D: n×n 最短路径矩阵
        如图中含负权环，返回 None
    """
    INF = math.inf
    
    # ---- 步骤 1: 添加超级源点 s' = n ----
    # 超级源点编号为 n（原图顶点 0..n-1，超级源点 n）
    total = n + 1
    
    # 构建扩展图的邻接表
    ext_graph = defaultdict(list)
    for u, v, w in edges:
        ext_graph[u].append((v, w))
    
    super_src = n
    for v in range(n):
        ext_graph[super_src].append((v, 0))  # s' → v，权值 0
    
    # ---- 步骤 2: 对超级源点跑 Bellman-Ford ----
    h, has_neg_cycle = bellman_ford(ext_graph, super_src, total)
    if has_neg_cycle:
        print("图中存在负权环，Johnson 算法无法处理！")
        return None
    
    # h[v] = δ(s', v)（势能函数）
    
    # ---- 步骤 3: 重新定权，构建非负权图 ----
    new_graph = defaultdict(list)
    for u, v, w in edges:
        new_w = w + h[u] - h[v]  # 保证 new_w >= 0
        assert new_w >= -1e-9, f"重新定权后边 ({u},{v}) 权值 {new_w:.4f} 仍为负！"
        new_graph[u].append((v, max(new_w, 0)))  # 浮点误差容忍
    
    # ---- 步骤 4: 对每个顶点跑 Dijkstra ----
    D = [[INF] * n for _ in range(n)]
    
    for u in range(n):
        d_prime = dijkstra(new_graph, u, n)
        for v in range(n):
            if d_prime[v] < INF:
                # 步骤 5: 还原真实最短路（减去势能差）
                D[u][v] = d_prime[v] - h[u] + h[v]
    
    return D


# ========== 示例 ==========
if __name__ == "__main__":
    # 含负权边的稀疏图
    n = 5
    edges = [
        (0, 1, 3),
        (0, 2, 8),
        (0, 4, -4),
        (1, 3, 1),
        (1, 4, 7),
        (2, 1, 4),
        (3, 0, 2),
        (3, 2, -5),
        (4, 3, 6),
    ]
    
    D = johnson(n, edges)
    
    if D:
        print("Johnson 算法 - 全对最短路径矩阵：")
        for i in range(n):
            row = []
            for j in range(n):
                if D[i][j] == math.inf:
                    row.append(" ∞")
                else:
                    row.append(f"{D[i][j]:3.0f}")
            print(f"  顶点{i}: {row}")

# 预期输出（CLRS 第4版 Figure 25.1 例子）：
# Johnson 算法 - 全对最短路径矩阵：
#   顶点0: ['  0', '  1', ' -3', '  2', ' -4']
#   顶点1: ['  3', '  0', ' -4', '  1', ' -1']
#   顶点2: ['  7', '  4', '  0', '  5', '  3']
#   顶点3: ['  2', ' -1', ' -5', '  0', ' -2']
#   顶点4: ['  8', '  5', '  1', '  6', '  0']
```

```cpp
#include <bits/stdc++.h>
using namespace std;

const long long INF = 1e18;

// -------- Bellman-Ford --------
// 返回 false 表示存在负权环
bool bellmanFord(int n, vector<tuple<int,int,long long>>& edges, 
                 int src, vector<long long>& dist) {
    dist.assign(n, INF);
    dist[src] = 0;
    
    for (int iter = 0; iter < n - 1; iter++) {
        bool updated = false;
        for (auto& [u, v, w] : edges) {
            if (dist[u] != INF && dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                updated = true;
            }
        }
        if (!updated) break;
    }
    
    // 检测负权环
    for (auto& [u, v, w] : edges) {
        if (dist[u] != INF && dist[u] + w < dist[v])
            return false; // 存在负权环
    }
    return true;
}

// -------- Dijkstra（二叉堆）--------
vector<long long> dijkstra(int n, vector<vector<pair<int,long long>>>& adj, int src) {
    vector<long long> dist(n, INF);
    priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<>> pq;
    dist[src] = 0;
    pq.push({0, src});
    
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto [v, w] : adj[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}

// -------- Johnson 算法 --------
// 返回 D[i][j] = i 到 j 的最短路，若存在负权环返回空矩阵
vector<vector<long long>> johnson(int n, vector<tuple<int,int,long long>> edges) {
    // 步骤 1: 添加超级源点 n
    int total = n + 1;
    int superSrc = n;
    
    // 超级源点到所有原图顶点加零权边
    for (int v = 0; v < n; v++)
        edges.push_back({superSrc, v, 0});
    
    // 步骤 2: Bellman-Ford 求势能 h[]
    vector<long long> h;
    if (!bellmanFord(total, edges, superSrc, h)) {
        cerr << "图中存在负权环，Johnson 算法无法处理！\n";
        return {};
    }
    
    // 步骤 3: 重新定权，构建非负权邻接表
    // 去掉超级源点相关的边
    vector<vector<pair<int,long long>>> adj(n);
    for (auto& [u, v, w] : edges) {
        if (u == superSrc || v == superSrc) continue; // 跳过超级源点边
        long long newW = w + h[u] - h[v];
        adj[u].push_back({v, newW});
    }
    
    // 步骤 4 & 5: 对每个顶点跑 Dijkstra，还原真实最短路
    vector<vector<long long>> D(n, vector<long long>(n, INF));
    
    for (int u = 0; u < n; u++) {
        auto dPrime = dijkstra(n, adj, u);
        for (int v = 0; v < n; v++) {
            if (dPrime[v] != INF) {
                // 还原：d(u,v) = d'(u,v) - h[u] + h[v]
                D[u][v] = dPrime[v] - h[u] + h[v];
            }
        }
    }
    
    return D;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    
    int n = 5;
    // CLRS 经典例题（顶点 0-based）
    vector<tuple<int,int,long long>> edges = {
        {0, 1, 3},  {0, 2, 8},  {0, 4, -4},
        {1, 3, 1},  {1, 4, 7},  {2, 1, 4},
        {3, 0, 2},  {3, 2, -5}, {4, 3, 6},
    };
    
    auto D = johnson(n, edges);
    
    if (!D.empty()) {
        cout << "Johnson 算法 - 全对最短路径矩阵：\n";
        for (int i = 0; i < n; i++) {
            cout << "  顶点" << i << ": [";
            for (int j = 0; j < n; j++) {
                if (j > 0) cout << ", ";
                if (D[i][j] == INF) cout << "∞";
                else cout << D[i][j];
            }
            cout << "]\n";
        }
    }
    
    return 0;
}
// 预期输出同上
```

<div data-component="JohnsonDijkstraPhase"></div>

### 23.2.5 Johnson 与 Floyd 的比较总结

| 维度 | Floyd-Warshall | Johnson |
|:---:|:---:|:---:|
| 时间复杂度 | $\Theta(V^3)$ | $O(VE + V^2 \log V)$ |
| 稀疏图（$E=O(V)$）| $O(V^3)$ | $O(V^2 \log V)$（快） |
| 稠密图（$E=O(V^2)$）| $O(V^3)$ | $O(V^3)$（持平） |
| 负权边 | ✅ 支持 | ✅ 支持（需跑 BF 检测负权环） |
| 负权环 | 检测到但结果无意义 | 检测到并报错 |
| 实现复杂度 | 极简（10行核心） | 较复杂（BF + 重新定权 + V次Dijkstra） |
| 适用场景 | 中小规模，代码简单 | 大规模稀疏图（$V$ 大，$E$ 小） |

---

## 23.3 矩阵乘法视角下的最短路径

### 23.3.1 (min, +) 半环与矩阵最短路

通常的矩阵乘法定义为：
$$(A \cdot B)_{ij} = \sum_k A_{ik} \cdot B_{kj}$$

将其中的 $\sum$（求和）替换为 $\min$（取最小），将 $\cdot$（乘法）替换为 $+$（加法），得到 **(min, +) 矩阵乘法**（也称**热带矩阵乘法**，Tropical Matrix Multiplication）：

$$(A \otimes B)_{ij} = \min_k \left( A_{ik} + B_{kj} \right)$$

这个运算在**(min, +) 半环**（Tropical semiring）上定义：
- "加法"是 $\min$，"乘法"是 $+$
- 加法单位元：$+\infty$（$\min(a, +\infty) = a$）
- 乘法单位元：$0$（$a + 0 = a$）

**为什么这能求最短路？**

设权重矩阵 $W$：$W_{ij} = w(i,j)$（若 $(i,j) \notin E$ 则 $+\infty$，$W_{ii} = 0$）。

定义 $L^{(m)}$ 为"恰好经过 $m$ 条边的最短路径"矩阵：$L_{ij}^{(m)}$ = 从 $i$ 到 $j$ 至多经过 $m$ 条边的最短路径权重。

则：
$$L^{(m)} = L^{(m-1)} \otimes W$$

扩展：$L^{(m)}_{ij} = \min_k\left(L^{(m-1)}_{ik} + W_{kj}\right)$

含义：从 $i$ 到 $j$ 经过 $m$ 步，枚举最后一跳的前置顶点 $k$：前 $m-1$ 步从 $i$ 到 $k$，第 $m$ 步从 $k$ 到 $j$。

**初始化**：$L^{(1)} = W$（只有直接边）；$L^{(0)}$ 为单位矩阵（对角线 $0$，其余 $+\infty$）。

**目标**：求 $L^{(V-1)} = W^{\otimes(V-1)}$（经过最多 $V-1$ 条边即可包含所有简单路径）。

### 23.3.2 重复平方：$O(V^3 \log V)$

朴素方法：逐步计算 $L^{(1)}, L^{(2)}, \ldots, L^{(V-1)}$，需要 $V-2$ 次矩阵乘法，每次 $O(V^3)$，总时间 $O(V^4)$。

**重复平方加速**：利用矩阵乘法的结合律（(min,+) 矩阵乘法满足结合律）：

$$L^{(2m)} = L^{(m)} \otimes L^{(m)}$$

因此可以通过以下步骤加速：
$$L^{(1)} \to L^{(2)} \to L^{(4)} \to \cdots \to L^{(2^{\lceil\log V\rceil})}$$

只需 $\lceil \log V \rceil$ 次矩阵乘法，每次 $O(V^3)$，总时间 $O(V^3 \log V)$。

**注意**：$L^{(2^k)}$ 当 $2^k \geq V-1$ 时就是最终答案（之后再乘也不会改变，因路径长度上限是 $V-1$）。

```python
import math

INF = math.inf

def min_plus_multiply(A, B, n):
    """(min, +) 矩阵乘法：C = A ⊗ B"""
    C = [[INF] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            if A[i][k] == INF:
                continue
            for j in range(n):
                if B[k][j] == INF:
                    continue
                C[i][j] = min(C[i][j], A[i][k] + B[k][j])
    return C


def apsp_repeated_squaring(n, edges):
    """
    使用 (min, +) 矩阵重复平方求 APSP
    时间：O(V^3 log V)
    """
    # 初始化 L = W（1条边的最短路矩阵）
    L = [[INF] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 0
    for u, v, w in edges:
        L[u][v] = min(L[u][v], w)
    
    # 重复平方：L^1 → L^2 → L^4 → ... → L^{≥V-1}
    m = 1
    while m < n - 1:
        L = min_plus_multiply(L, L, n)
        m *= 2
    
    return L


# 示例（同 Floyd-Warshall 例子）
n = 4
edges_ex = [
    (0, 1, 3), (0, 3, 7), (1, 2, 1), (2, 0, 2),
    (3, 2, -3), (1, 3, 5), (2, 3, 2),
]

L = apsp_repeated_squaring(n, edges_ex)
print("(min, +) 重复平方结果：")
for i in range(n):
    print([L[i][j] if L[i][j] != INF else '∞' for j in range(n)])

# 注意：由于有负权边，重复平方法需谨慎（可能需要更多轮）
# 对于无负权环图，最终结果与 Floyd-Warshall 一致
```

```cpp
#include <bits/stdc++.h>
using namespace std;

const long long INF = 1e18;

// (min, +) 矩阵乘法
vector<vector<long long>> minPlusMultiply(
    const vector<vector<long long>>& A,
    const vector<vector<long long>>& B, int n) 
{
    vector<vector<long long>> C(n, vector<long long>(n, INF));
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            if (A[i][k] == INF) continue;
            for (int j = 0; j < n; j++) {
                if (B[k][j] == INF) continue;
                C[i][j] = min(C[i][j], A[i][k] + B[k][j]);
            }
        }
    }
    return C;
}

// 重复平方法 APSP
vector<vector<long long>> apspRepeatedSquaring(
    int n, vector<tuple<int,int,long long>>& edges)
{
    // 初始化权重矩阵 L = W
    vector<vector<long long>> L(n, vector<long long>(n, INF));
    for (int i = 0; i < n; i++) L[i][i] = 0;
    for (auto& [u, v, w] : edges)
        L[u][v] = min(L[u][v], w);
    
    // 重复平方
    int m = 1;
    while (m < n - 1) {
        L = minPlusMultiply(L, L, n);
        m *= 2;
    }
    
    return L;
}

int main() {
    int n = 4;
    vector<tuple<int,int,long long>> edges = {
        {0,1,3},{0,3,7},{1,2,1},{2,0,2},{3,2,-3},{1,3,5},{2,3,2}
    };
    
    auto L = apspRepeatedSquaring(n, edges);
    
    cout << "(min,+) 重复平方 APSP 结果：\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (L[i][j] == INF) cout << " ∞";
            else cout << " " << L[i][j];
        }
        cout << "\n";
    }
    return 0;
}
```

<div data-component="MinPlusMatrixMult"></div>

### 23.3.3 Floyd-Warshall 与矩阵乘法视角的关系

**两种 DP，两种"扩展"维度**：

| 算法 | DP 维度 | 如何扩展 | 每步操作 |
|:---:|:---:|:---:|:---:|
| (min,+) 重复平方 | 允许的**边数** $m$ | $m \to 2m$（允许的边数翻倍） | $L^{(2m)} = L^{(m)} \otimes L^{(m)}$ |
| Floyd-Warshall | 可用**中转节点集** $\{1,\ldots,k\}$ | $k \to k+1$（加入一个新节点） | $d_{ij}^{(k)} = \min(d_{ij}^{(k-1)}, d_{ik}^{(k-1)}+d_{kj}^{(k-1)})$ |

**为什么 Floyd 更快？**

重复平方的总时间是 $O(V^3 \log V)$，比 Floyd 的 $O(V^3)$ 多一个 $\log V$ 因子。原因是重复平方每次矩阵乘法都是完整的 $V \times V \times V$ 操作，而 Floyd 的每轮 $k$ 只需要 $O(V^2)$ 操作（更新整个矩阵，但每个 $(i,j)$ 只做一次比较）。

**理论价值**：矩阵乘法视角揭示了 APSP 与代数结构的深层联系。在**(min,+) 半环**上，APSP 等价于矩阵幂运算，这开启了用矩阵算法工具（如 Strassen 变体）加速 APSP 的研究方向。

---

## 23.4 三种算法复杂度综合对比

<div data-component="APSPComplexityComparison"></div>

用量化的方式感受三种算法的差异。设 $V$ 个顶点，$E$ 条边：

**稀疏图**（$E \approx 2V$，例如树或稀疏路网）：

| 算法 | 计算量 | $V=1000$ 时 |
|:---:|:---:|:---:|
| Floyd-Warshall | $V^3$ | $10^9$ 次操作 |
| Johnson（二叉堆 Dijkstra） | $VE + V^2\log V$ | $\approx 2\times10^6 + 10^7 \approx 10^7$ |
| V次 Bellman-Ford | $V^2 E$ | $2\times10^9$ |
| (min,+) 重复平方 | $V^3 \log V$ | $\approx 10^{10}$ |

**稠密图**（$E \approx V^2/2$）：

| 算法 | 计算量 | $V=1000$ 时 |
|:---:|:---:|:---:|
| Floyd-Warshall | $V^3$ | $10^9$ |
| Johnson（二叉堆 Dijkstra） | $V^3 + V^2\log V$ | $\approx 10^9$（相当） |

**实践建议**：
- 稠密图（$E = O(V^2)$）且 $V \leq 500$：**Floyd-Warshall**（代码最简，常数最小）
- 稀疏图（$E = O(V)$ 或 $E = O(V \log V)$）含负权边：**Johnson 算法**
- 仅无负权边的大规模稀疏图：**V 次 Dijkstra**（不需要 Bellman-Ford 预处理）
- 理论研究或特殊代数结构（半环上的矩阵乘法推广）：**(min,+) 矩阵方法**

---

## 23.5 典型 LeetCode 题目与解题模式

### 23.5.1 直接 Floyd 模板题

**LeetCode 1334. 阈值距离内邻居最少的城市**

```python
def findTheCity(n: int, edges: list, distanceThreshold: int) -> int:
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]
    
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        dist[u][v] = min(dist[u][v], w)
        dist[v][u] = min(dist[v][u], w)  # 无向图
    
    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    # 找到距离 ≤ distanceThreshold 的邻居最少的城市（取编号最大的）
    ans = -1
    min_count = n + 1
    for i in range(n):
        count = sum(1 for j in range(n) if i != j and dist[i][j] <= distanceThreshold)
        if count <= min_count:
            min_count = count
            ans = i
    
    return ans
```

```cpp
int findTheCity(int n, vector<vector<int>>& edges, int distanceThreshold) {
    const long long INF = 1e9;
    vector<vector<long long>> dist(n, vector<long long>(n, INF));
    for (int i = 0; i < n; i++) dist[i][i] = 0;
    for (auto& e : edges) {
        dist[e[0]][e[1]] = min(dist[e[0]][e[1]], (long long)e[2]);
        dist[e[1]][e[0]] = min(dist[e[1]][e[0]], (long long)e[2]);
    }
    for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
    
    int ans = -1, minCnt = n + 1;
    for (int i = 0; i < n; i++) {
        int cnt = 0;
        for (int j = 0; j < n; j++)
            if (i != j && dist[i][j] <= distanceThreshold) cnt++;
        if (cnt <= minCnt) { minCnt = cnt; ans = i; }
    }
    return ans;
}
```

### 23.5.2 传递闭包（Floyd 变体）

**Floyd 也可用于计算传递闭包**（节点 $i$ 是否可达节点 $j$）：将"最短路"运算替换为"可达性"的布尔运算：

$$r_{ij}^{(k)} = r_{ij}^{(k-1)} \lor \left(r_{ik}^{(k-1)} \land r_{kj}^{(k-1)}\right)$$

```python
def transitive_closure(n: int, edges: list) -> list:
    reach = [[False] * n for _ in range(n)]
    for i in range(n):
        reach[i][i] = True
    for u, v in edges:
        reach[u][v] = True
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                reach[i][j] = reach[i][j] or (reach[i][k] and reach[k][j])
    
    return reach
```

### 23.5.3 LeetCode 743. 网络延迟时间（SSSP 变体）

虽然这道题是单源最短路，但可以用 Floyd 解决（$n \leq 100$）：

```python
import math

def networkDelayTime(times: list, n: int, k: int) -> int:
    INF = math.inf
    dist = [[INF] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dist[i][i] = 0
    for u, v, w in times:
        dist[u][v] = w
    
    for mid in range(1, n + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if dist[i][mid] + dist[mid][j] < dist[i][j]:
                    dist[i][j] = dist[i][mid] + dist[mid][j]
    
    ans = max(dist[k][v] for v in range(1, n + 1))
    return ans if ans < INF else -1
```

---

## 23.6 常见错误与调试技巧

### 错误 1：Floyd 循环顺序错误

```python
# ❌ 错误：内外层循环顺序弄反
for i in range(n):
    for j in range(n):
        for k in range(n):  # k 在最内层，WRONG!
            dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

# ✅ 正确：k 在最外层
for k in range(n):
    for i in range(n):
        for j in range(n):
            dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
```

**后果**：k 在内层时，计算 $d_{ij}$ 时可能用到了同轮已经更新的 $d_{ik}$、$d_{kj}$，但这些值是用"允许通过比 $k$ 更大编号节点的路径"计算的，破坏了 DP 不变式。

### 错误 2：溢出问题

当 `INF = 1e9` 时，`dist[i][k] + dist[k][j]` 可能溢出 `int` 范围（两个 $10^9$ 相加超过 `int` 上限 $2.1 \times 10^9$）：

```cpp
// ❌ 可能溢出
int dist[N][N];
// 若两者均 = 1e9，相加 = 2e9，超过 int 最大值 ~2.1e9（临界）

// ✅ 加保护
if (dist[i][k] != INF && dist[k][j] != INF) {
    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
}
// 或者使用 long long
```

### 错误 3：Johnson 算法还原时忘记减去势能

```python
# ❌ 忘记还原
D[u][v] = d_prime[u][v]  # WRONG，这是新权图中的距离

# ✅ 正确还原
D[u][v] = d_prime[u][v] - h[u] + h[v]
```

### 错误 4：浮点精度问题（负权边重新定权后）

若边权是浮点数，重新定权后本应为 $0$ 的值可能变成 $-\epsilon$（极小负数），导致 Dijkstra 出错。解决：

```python
new_w = max(0.0, w + h[u] - h[v])  # 容忍浮点误差
```

### 错误 5：self-loop 初始化

如果原图有自环，初始化时 $d_{ii}$ 应取 $\min(0, w_{\text{self-loop}})$。若自环权为负，则直接是负权环。

---

## 23.7 扩展阅读与参考资料

- **CLRS 第4版**：Chapter 23（All-Pairs Shortest Paths）—— 权威算法教材，Floyd-Warshall 与 Johnson 的完整证明与伪代码
- **Aho, Hopcroft & Ullman**：图算法经典参考书，矩阵乘法视角的早期探讨
- **(min, +) 半环**：Wikipedia "Tropical semiring"；Ryan Williams 等人的 APSP 下界研究（O(n³/log²n) 的微小改进，理论前沿）
- **LeetCode 专题**：#743、#1334、#787（最多 K 站中转），#1368 等
- **竞赛技巧**：CF 上许多图论题使用 Floyd 进行传递闭包或小规模 APSP；Johnson 算法在大稀疏图题目中出现较少，更多是考察理解

---

> 💡 **本章思考题**
>
> 1. Floyd-Warshall 需要 $\Theta(V^3)$ 时间，能否在 $O(V^3 / \log V)$ 内解决 APSP？（提示：研究前沿，目前最优有微小改进但尚未突破 $V^3$）
>
> 2. 若图是有向无环图（DAG），APSP 有更快的算法吗？（提示：利用拓扑序，DP，$O(V + E)$ 单源，APSP 为 $O(V(V + E))$）
>
> 3. Johnson 算法中，为什么超级源点的出边权值必须是 $0$ 而不是其他值？若设为 $1$ 会发生什么？
>
> 4. 在 (min, +) 矩阵重复平方中，若图中有负权环，$L^{(m)}$ 的对角线会发生什么变化？如何利用这一性质检测负权环？
>
> 5. 如何将 Floyd-Warshall 推广到有向图的"最大最小路径"（最大化路径上最小边权，即"最宽路径"）？状态转移方程如何修改？

---
title: "Chapter 42: 近似算法与处理 NP 难问题"
description: "当 NP-hard 问题无法精确求解时，如何在可接受时间内找到质量有保证的近似解？本章系统讲解近似比、PTAS/FPTAS、贪心近似（顶点覆盖 2-近似、集合覆盖 ln n 近似）、TSP 近似（Christofides 1.5）、子集和 FPTAS、随机化算法（Monte Carlo vs Las Vegas、MAX-CUT）以及启发式算法（模拟退火、遗传算法），构建面对计算难题的完整工程工具箱"
tags: ["approximation-algorithms", "NP-hard", "vertex-cover", "set-cover", "TSP", "Christofides", "FPTAS", "randomized-algorithms", "simulated-annealing", "genetic-algorithm"]
difficulty: "hard"
updated: "2026-03-12"
---

# Chapter 42: 近似算法与处理 NP 难问题

> **Part XII · 计算复杂性与 NP**

上一章我们认识了 NP 难（NP-hard）问题的本质：除非 P=NP，它们没有多项式时间的**精确**算法。然而，现实工程中这些问题每天都在出现——路线规划、资源调度、网络设计……我们不可能因为问题是 NP-hard 就放弃。

**那怎么办？** 计算机科学家给出了三条出路：

1. **近似算法（Approximation Algorithms）**：在多项式时间内找到"足够好"的解，并严格证明它与最优解的差距至多是某个倍数。
2. **随机化算法（Randomized Algorithms）**：引入随机性，以高概率得到好的结果，或以随机时间给出精确结果。
3. **启发式算法（Heuristic Algorithms）**：放弃理论保证，用问题特有的"直觉规则"快速找到实用上好的解。

> **本章学习路径**：近似比定义 → 顶点覆盖/集合覆盖的贪心近似 → TSP 的 Christofides 算法 → FPTAS → 随机化算法 → 启发式 SA/GA

---

## 42.1 近似算法基础

### 42.1.1 为什么需要近似算法？

面对 NP-hard 问题，通常有以下几种精确求解策略：

| 策略 | 代表算法 | 时间复杂度 | 适用场景 |
|---|---|---|---|
| 暴力枚举 | 枚举所有解 | O(2ⁿ·poly(n)) | n ≤ 20 |
| 分支定界 | Branch & Bound | 最坏指数，实践中有效 | n ≤ 50–100 |
| 动态规划（伪多项式） | Held-Karp TSP | O(2ⁿ · n²) | 特定结构问题 |
| 整数规划（IP Solver） | Gurobi, CPLEX | 指数最坏 | 工业级小规模 |

对于大规模实例（如 n = 10⁶ 的路由优化），精确算法根本无法在线上应用。**近似算法**用可接受的精度损失换取多项式时间——这是工程实践中处理 NP-hard 问题的首选理论工具。

### 42.1.2 近似比（Approximation Ratio）的严格定义

设 $I$ 为问题实例，$A(I)$ 为近似算法给出的解的值，$\text{OPT}(I)$ 为最优解的值。

**最小化问题**（如 TSP、顶点覆盖）：

$$
\rho_A = \sup_{I} \frac{A(I)}{\text{OPT}(I)} \leq \alpha
$$

称 $A$ 是 $\alpha$-近似算法（$\alpha \geq 1$）。$\alpha$ 越接近 1，近似质量越高。

**最大化问题**（如 MAX-CUT、最大独立集）：

$$
\rho_A = \sup_{I} \frac{\text{OPT}(I)}{A(I)} \leq \alpha
$$

或等价地，$A(I) \geq \frac{1}{\alpha} \cdot \text{OPT}(I)$。此时 $\alpha \geq 1$，$A(I)/\text{OPT}(I) \geq 1/\alpha$。

> **直觉理解**：对于最小化问题，2-近似意味着我们的解**最坏情况下只是最优解的 2 倍大**。如果最优 TSP 路程是 1000 km，我们保证找到不超过 2000 km 的路线。

**为什么不直接说"解比最优差多少"？** 因为"差"是绝对值，与问题规模相关；"比"是相对值，与规模无关，更有理论意义。

### 42.1.3 PTAS 与 FPTAS

近似能力的层次从强到弱：

**完全多项式时间近似方案（FPTAS, Fully Polynomial-Time Approximation Scheme）**：
- 对任意 $\varepsilon > 0$，给出 $(1+\varepsilon)$-近似解
- **运行时间**是 $n$ 和 $1/\varepsilon$ 的**多项式函数**，如 $O(n^2/\varepsilon)$ 或 $O(n^3/\varepsilon^2)$
- 这是最强的近似方案，代表问题："子集和"（§42.2.4 将详细讲解）

**多项式时间近似方案（PTAS, Polynomial-Time Approximation Scheme）**：
- 对任意 $\varepsilon > 0$，给出 $(1+\varepsilon)$-近似解
- **运行时间** 关于 $n$ 是多项式，但关于 $1/\varepsilon$ 可以是指数，如 $O(n^{1/\varepsilon})$ 或 $O(2^{1/\varepsilon} \cdot n)$
- 代表问题：欧氏 TSP（Arora's scheme）、背包问题
- 注意：FPTAS ⊂ PTAS（FPTAS 是 PTAS 的子集，添加了对 $1/\varepsilon$ 也是多项式的要求）

**固定近似比（Constant Approximation）**：
- 比如 2-近似、1.5-近似，与 $\varepsilon$ 无关
- 代表问题：顶点覆盖（2）、度量 TSP (Christofides 1.5)、集合覆盖（ln n，非常数！）

**APX-hard**：
- 不存在 PTAS 的问题类（除非 P=NP）
- 即使对任意固定 $\varepsilon > 0$，$(1+\varepsilon)$-近似也是 NP-hard
- 代表问题：一般 TSP（无三角不等式时甚至无任何常数近似）

<div data-component="ApproximationHierarchy"></div>

### 42.1.4 APX-hard 与"近似壁垒"

为什么一般 TSP 在没有三角不等式时**无任何常数近似**（除非 P=NP）？

**定理**（Sahni & Gonzalez 1976）：若 P≠NP，则一般 TSP（边权可以任意大）不存在任何多项式时间 $\rho$-近似算法（$\rho$ 是任意正整数）。

**证明思路（反证）**：
假设存在 $\rho$-近似算法 $A$。给定哈密顿回路判定问题实例 $G$（判断是否有哈密顿回路），构造完全图 $G'$：
- 原 $G$ 的边权 = 1
- $G$ 中不存在的边权 = $\rho \cdot n + 1$

若 $G$ 有哈密顿回路，则 $\text{OPT}(G') = n$，$A$ 输出 $\leq \rho n$，且必须只走权重为 1 的边（因为走一条权 $>\rho n$ 的边总和 $> \rho n$），即 $A$ 找到了哈密顿回路。若 $G$ 无哈密顿回路，则 $\text{OPT}(G') > \rho n$，$A$ 输出可能无法区分。

这将哈密顿回路问题（NP-hard）规约为 TSP $\rho$-近似，矛盾。$\square$

> **实践启示**：对一般 TSP（如非对称、不满足三角不等式的情形），工程上只能用启发式，没有保证近似比的多项式算法。

### 42.1.5 近似算法的设计方法论

近似算法的设计有几种常见范式：

1. **贪心近似**：局部最优的贪心策略往往带来全局的有界近似
   - 集合覆盖：每次选覆盖未覆盖元素最多的集合
   - 顶点覆盖：每次取一条边的两个端点

2. **LP 松弛 + 取整**：
   - 将整数规划（NP-hard）放松为线性规划（多项式）
   - LP 最优解 $\leq$ 整数规划最优（对最小化）
   - 对 LP 解取整，分析近似比

3. **组合结构利用**：
   - Christofides 算法利用 MST 的权重与 TSP 最优解的关系
   - 欧拉回路的存在性 → 哈密顿路径提取

4. **对偶理论**：
   - LP 对偶约束提供下界，原始解提供上界，分析两者的比值

---

## 42.2 经典近似算法

### 42.2.1 顶点覆盖的 2-近似算法

#### 问题回顾

**顶点覆盖（Vertex Cover）**：给定无向图 $G=(V,E)$，找到最小的顶点子集 $C \subseteq V$，使得每条边至少有一个端点在 $C$ 中。

这是 NP-complete 问题（Ch41 已证明），但有一个非常优雅的 2-近似算法。

#### 算法描述

```
APPROX-VERTEX-COVER(G):
  C ← ∅
  E' ← G.E（所有边的副本）
  while E' ≠ ∅:
    任取 E' 中一条边 (u, v)
    C ← C ∪ {u, v}      // 将两端点都加入覆盖
    从 E' 中删除所有与 u 或 v 相关联的边
  return C
```

这就是**最大匹配贪心**：每迭代一步，我们选出一条边加入匹配，并将匹配两端点都加入 $C$。

#### 正确性证明

- **正确性（C 确实是覆盖）**：每条原始边$(u,v)$，要么在算法中被直接选中，要么被某条关联边选中后删除。如果$(u,v)$在某次迭代中被删除（因为 $u$ 或 $v$ 已进入 $C$），则该端点已在 $C$ 中，即$(u,v)$被覆盖。
- **近似比（$\leq 2 \cdot \text{OPT}$）**：

设算法选出的边集（匹配）为 $M = \{(u_1,v_1), (u_2,v_2), \ldots, (u_k,v_k)\}$。

1. $M$ 是一个**匹配**（所选边两两不相邻）：因为每次选边后删除所有关联边，后续选出的边不与已选边共享端点。

2. $|C| = 2k$（每条匹配边贡献 2 个顶点）。

3. 任何顶点覆盖（包括 $\text{OPT}$）必须覆盖匹配 $M$ 的每条边，即**对匹配中每条边，至少选它的一个端点**。由于匹配中的边两两不共享端点，$\text{OPT}$ 必须包含 $k$ 个不同顶点（对每条匹配边至少一个）。

4. 因此 $\text{OPT} \geq k$，从而 $|C| = 2k \leq 2 \cdot \text{OPT}$。$\square$

> **注意**：2-近似是紧的（tight）。存在实例使得算法输出恰好是 OPT 的 2 倍（如二部图的完美匹配情形）。著名猜想（Unique Games Conjecture）表明，若某猜想成立则顶点覆盖不存在 $(2-\varepsilon)$-近似。

#### 代码实现

```python
from typing import List, Tuple, Set

def vertex_cover_2approx(n: int, edges: List[Tuple[int, int]]) -> Set[int]:
    """
    顶点覆盖的 2-近似算法（最大匹配贪心）
    
    参数:
        n: 顶点数（顶点编号 0 到 n-1）
        edges: 边列表，每条边为 (u, v)
    
    返回:
        顶点覆盖集合 C（大小 ≤ 2 × OPT）
    
    时间复杂度：O(V + E)
    近似比：2
    """
    cover = set()
    # 记录每个节点是否已覆盖（已在 cover 中或关联边已被删除）
    covered = [False] * n
    
    for u, v in edges:
        # 边界条件：跳过已被覆盖的边
        # 如果 u 或 v 已在覆盖中，这条边已被覆盖，无需处理
        if covered[u] or covered[v]:
            continue
        
        # 选择这条边：将两端点加入覆盖
        # 设计考量：贪心地同时加入两个端点，而非"选更好的那个"
        # 这样才能保证是最大匹配，从而保证 2-近似
        cover.add(u)
        cover.add(v)
        covered[u] = True
        covered[v] = True
    
    return cover


def verify_cover(n: int, edges: List[Tuple[int, int]], cover: Set[int]) -> bool:
    """验证给定集合是否是合法的顶点覆盖"""
    for u, v in edges:
        if u not in cover and v not in cover:
            return False  # 边 (u,v) 未被覆盖
    return True


# === 示例 ===
if __name__ == "__main__":
    # G: 6-cycle (0-1-2-3-4-5-0)
    n = 6
    edges = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,0)]
    
    cover = vertex_cover_2approx(n, edges)
    print(f"顶点覆盖: {cover}")       # e.g., {0, 1, 2, 3} — 4 个顶点
    print(f"大小: {len(cover)}")      # OPT = 3（交替选取），我们得到 ≤ 6
    print(f"合法性: {verify_cover(n, edges, cover)}")
    
    # 注意：最优覆盖是 {1,3,5} 或 {0,2,4}，大小为 3
    # 算法可能给出大小为 4 的覆盖（2-近似保证 ≤ 2×3 = 6）
```

```cpp
#include <bits/stdc++.h>
using namespace std;

// 顶点覆盖的 2-近似算法（最大匹配贪心）
// 时间复杂度：O(V + E)  近似比：2
set<int> vertexCover2Approx(int n, vector<pair<int,int>>& edges) {
    set<int> cover;
    vector<bool> covered(n, false); // 标记节点是否已在覆盖中
    
    for (auto& [u, v] : edges) {
        // 边界条件：如果 u 或 v 已在覆盖中，这条边已被覆盖
        if (covered[u] || covered[v]) continue;
        
        // 关键设计：同时将 u 和 v 加入覆盖（保持最大匹配结构）
        cover.insert(u);
        cover.insert(v);
        covered[u] = covered[v] = true;
    }
    return cover;
}

// 验证合法覆盖
bool verifyCover(vector<pair<int,int>>& edges, set<int>& cover) {
    for (auto& [u, v] : edges) {
        if (!cover.count(u) && !cover.count(v)) return false;
    }
    return true;
}

int main() {
    // 6-环图
    int n = 6;
    vector<pair<int,int>> edges = {{0,1},{1,2},{2,3},{3,4},{4,5},{5,0}};
    
    auto cover = vertexCover2Approx(n, edges);
    cout << "顶点覆盖: {";
    for (int v : cover) cout << v << " ";
    cout << "}\n";
    cout << "大小: " << cover.size() << "\n";          // ≤ 6（OPT=3）
    cout << "合法: " << verifyCover(edges, cover) << "\n";
    return 0;
}
```

#### 与最优解的对比

对于稠密图，贪心匹配可能涉及 O(E) 的边，每次操作 O(1)，总复杂度 O(V+E)。这个算法在实践中非常快，且对随机图表现往往优于 2 倍。

<div data-component="VertexCoverApprox"></div>

### 42.2.2 TSP 的近似算法

旅行商问题（TSP）在满足**三角不等式**（$d(u,w) \leq d(u,v) + d(v,w)$）时，称为**度量 TSP（Metric TSP）**，可以用以下近似算法处理。

#### 2-近似：MST 双重遍历

**关键观察**：
- MST 的权重 $w(\text{MST}) \leq w(\text{OPT})$（最优 TSP 回路去掉一条边就是一个生成树，权重 $\geq w(\text{MST})$）
- DFS 遍历 MST 形成的欧拉回路（每条 MST 边走两次），总路程 $= 2 \cdot w(\text{MST}) \leq 2 \cdot w(\text{OPT})$
- 利用三角不等式"跳过"重复节点，路程只会变短

```
APPROX-TSP-2(G, d):
  (1) 找 MST T（Prim 或 Kruskal）
  (2) 对 T 做 DFS，得到节点访问序列（每条边走两次 = 欧拉游走）
  (3) 按访问序列输出，跳过已访问节点（利用三角不等式）
  return 哈密顿回路 H
```

**近似比证明**：
- $w(H) \leq w(\text{欧拉游走}) = 2 \cdot w(\text{MST}) \leq 2 \cdot w(\text{OPT})$
- 每次"跳过"（用直达边替代绕路）因三角不等式不增加代价

#### Christofides 算法：1.5-近似

**Christofides（1976）** 将近似比改进到 $3/2$，这是目前度量 TSP 的最佳多项式近似比（直到 2020 年 Karlin-Klein-Oveis Gharan 才将其改进到 $3/2 - \varepsilon$）。

**算法步骤**：

```
CHRISTOFIDES(G, d):
  (1) 找最小生成树 T
  (2) 找 T 中所有奇度顶点的集合 O（|O| 必为偶数）
  (3) 在 O 的导出完全子图上，找最小权完美匹配 M
  (4) 合并 T 和 M 得到欧拉图 H
  (5) 在 H 上找欧拉回路
  (6) 跳过重复节点，提取哈密顿回路
```

**为什么是 1.5-近似？**

设最优 TSP 路程为 $\text{OPT}$。

1. $w(T) \leq \text{OPT}$（MST 权 $\leq$ TSP 最优）

2. $w(M) \leq \frac{1}{2} \cdot \text{OPT}$：
   - 考虑最优 TSP 哈密顿回路限制在奇度顶点集 $O$ 上的子路径
   - 这些子路径形成两个完美匹配（奇偶交替选取）
   - 两个匹配之一的权重 $\leq \frac{1}{2} \cdot \text{OPT}$
   - 最小完美匹配 $\leq$ 上述匹配权

3. $w(\text{欧拉回路}) = w(T) + w(M) \leq \text{OPT} + \frac{1}{2} \cdot \text{OPT} = \frac{3}{2} \cdot \text{OPT}$

4. 跳过重复节点（三角不等式）不增加代价

$$w(\text{Christofides 输出}) \leq \frac{3}{2} \cdot \text{OPT} \quad \square$$

```python
import heapq
from typing import List, Tuple, Dict
import itertools

def prim_mst(n: int, dist: List[List[float]]) -> List[Tuple[int, int]]:
    """Prim 算法求最小生成树，返回 MST 边集"""
    in_mst = [False] * n
    min_edge = [float('inf')] * n
    parent = [-1] * n
    min_edge[0] = 0
    
    # 最小堆：(权重, 节点)
    heap = [(0, 0)]
    mst_edges = []
    
    while heap:
        w, u = heapq.heappop(heap)
        if in_mst[u]:
            continue
        in_mst[u] = True
        if parent[u] != -1:
            mst_edges.append((parent[u], u))
        
        for v in range(n):
            if not in_mst[v] and dist[u][v] < min_edge[v]:
                min_edge[v] = dist[u][v]
                parent[v] = u
                heapq.heappush(heap, (dist[u][v], v))
    
    return mst_edges


def christofides_approx(n: int, dist: List[List[float]]) -> List[int]:
    """
    Christofides 1.5-近似 TSP 算法
    
    参数:
        n: 城市数
        dist: dist[i][j] = 城市 i 到城市 j 的距离（满足三角不等式）
    
    返回:
        哈密顿回路的顶点序列（已去重）
    
    时间复杂度：O(n³)（最小完美匹配占主导）
    近似比：1.5
    
    注意：此处的最小完美匹配用暴力枚举（仅供教学），
          工业实现应使用 Blossom V 等高效算法
    """
    # ── 步骤 1: 求最小生成树 ──
    mst_edges = prim_mst(n, dist)
    
    # ── 步骤 2: 找奇度顶点 ──
    degree = [0] * n
    adj = [[] for _ in range(n)]  # MST 邻接表
    for u, v in mst_edges:
        degree[u] += 1
        degree[v] += 1
        adj[u].append(v)
        adj[v].append(u)
    
    # 奇度顶点集合 O（|O| 保证为偶数）
    odd_vertices = [v for v in range(n) if degree[v] % 2 == 1]
    
    # ── 步骤 3: 奇度顶点上的最小权完美匹配（暴力，教学用）──
    # 对 |O| 个顶点，枚举所有完美匹配
    def min_perfect_matching(vertices: List[int]) -> List[Tuple[int, int]]:
        """暴力枚举所有完美匹配，返回最小权匹配"""
        if not vertices:
            return []
        # 固定第一个顶点，枚举它的匹配伙伴
        best_cost = float('inf')
        best_matching = []
        first = vertices[0]
        rest = vertices[1:]
        for i, partner in enumerate(rest):
            # first 与 partner 匹配
            remaining = rest[:i] + rest[i+1:]
            sub_matching = min_perfect_matching(remaining)
            cost = dist[first][partner] + sum(dist[a][b] for a, b in sub_matching)
            if cost < best_cost:
                best_cost = cost
                best_matching = [(first, partner)] + sub_matching
        return best_matching
    
    matching = min_perfect_matching(odd_vertices)
    
    # ── 步骤 4: 构造欧拉图（MST + 匹配边）──
    euler_adj = [[] for _ in range(n)]
    for u, v in mst_edges:
        euler_adj[u].append(v)
        euler_adj[v].append(u)
    for u, v in matching:
        euler_adj[u].append(v)
        euler_adj[v].append(u)
    
    # ── 步骤 5: 找欧拉回路（Hierholzer 算法）──
    def euler_tour(start: int) -> List[int]:
        """Hierholzer 算法求欧拉回路"""
        adj_copy = [list(lst) for lst in euler_adj]
        stack = [start]
        circuit = []
        while stack:
            v = stack[-1]
            if adj_copy[v]:
                u = adj_copy[v].pop()
                # 删除反向边（无向图）
                adj_copy[u].remove(v)
                stack.append(u)
            else:
                circuit.append(stack.pop())
        return circuit
    
    euler = euler_tour(0)
    
    # ── 步骤 6: 跳过重复节点，提取哈密顿回路 ──
    visited = set()
    hamiltonian = []
    for v in euler:
        if v not in visited:
            visited.add(v)
            hamiltonian.append(v)
    
    return hamiltonian


def tour_cost(tour: List[int], dist: List[List[float]]) -> float:
    """计算路线总距离"""
    n = len(tour)
    return sum(dist[tour[i]][tour[(i+1) % n]] for i in range(n))


# === 示例 ===
if __name__ == "__main__":
    import math
    
    # 5 个城市的坐标（欧氏距离满足三角不等式）
    coords = [(0,0), (1,0), (2,1), (1,2), (0,1)]
    n = len(coords)
    dist = [[math.sqrt((x1-x2)**2 + (y1-y2)**2)
             for x2, y2 in coords] for x1, y1 in coords]
    
    tour = christofides_approx(n, dist)
    print(f"路线: {tour}")
    print(f"总距离: {tour_cost(tour, dist):.4f}")
```

```cpp
#include <bits/stdc++.h>
using namespace std;
using pii = pair<int,int>;
using pdi = pair<double,int>;

// ── Prim MST ──
vector<pii> primMST(int n, vector<vector<double>>& dist) {
    vector<bool> inMST(n, false);
    vector<double> minEdge(n, 1e18);
    vector<int> parent(n, -1);
    priority_queue<pdi, vector<pdi>, greater<pdi>> pq;
    minEdge[0] = 0;
    pq.push({0.0, 0});
    vector<pii> mstEdges;
    
    while (!pq.empty()) {
        auto [w, u] = pq.top(); pq.pop();
        if (inMST[u]) continue;
        inMST[u] = true;
        if (parent[u] != -1) mstEdges.push_back({parent[u], u});
        for (int v = 0; v < n; v++) {
            if (!inMST[v] && dist[u][v] < minEdge[v]) {
                minEdge[v] = dist[u][v];
                parent[v] = u;
                pq.push({dist[u][v], v});
            }
        }
    }
    return mstEdges;
}

// ── Hierholzer 欧拉回路 ──
vector<int> eulerTour(int n, vector<vector<int>>& adj, int start) {
    vector<int> idx(n, 0); // 当前邻接表指针
    vector<int> stk = {start}, circuit;
    while (!stk.empty()) {
        int v = stk.back();
        if (idx[v] < (int)adj[v].size()) {
            stk.push_back(adj[v][idx[v]++]);
        } else {
            circuit.push_back(v);
            stk.pop_back();
        }
    }
    return circuit;
}

// ── Christofides 1.5-近似 TSP（教学版：暴力匹配）──
vector<int> christofidesApprox(int n, vector<vector<double>>& dist) {
    // 步骤 1: MST
    auto mstEdges = primMST(n, dist);
    vector<int> degree(n, 0);
    vector<vector<int>> adj(n);
    for (auto [u, v] : mstEdges) {
        degree[u]++; degree[v]++;
        adj[u].push_back(v); adj[v].push_back(u);
    }
    
    // 步骤 2: 奇度顶点
    vector<int> odd;
    for (int i = 0; i < n; i++) if (degree[i] & 1) odd.push_back(i);
    
    // 步骤 3: 暴力最小完美匹配（仅教学用）
    // 在奇度顶点集上枚举所有匹配
    int m = odd.size();
    vector<bool> used(m, false);
    function<pair<double, vector<pii>>(int)> matchDP = [&](int start) -> pair<double, vector<pii>> {
        while (start < m && used[start]) start++;
        if (start >= m) return {0.0, {}};
        used[start] = true;
        double best = 1e18;
        vector<pii> bestM;
        for (int j = start+1; j < m; j++) {
            if (used[j]) continue;
            used[j] = true;
            double d = dist[odd[start]][odd[j]];
            auto [sub, subM] = matchDP(start+1);
            if (d + sub < best) {
                best = d + sub;
                bestM = subM;
                bestM.push_back({odd[start], odd[j]});
            }
            used[j] = false;
        }
        used[start] = false;
        return {best, bestM};
    };
    auto [_, matching] = matchDP(0);
    
    // 步骤 4: 合并 MST 和匹配边
    vector<vector<int>> euAdj(n);
    for (auto [u, v] : mstEdges) { euAdj[u].push_back(v); euAdj[v].push_back(u); }
    for (auto [u, v] : matching) { euAdj[u].push_back(v); euAdj[v].push_back(u); }
    
    // 步骤 5: 欧拉回路
    auto euler = eulerTour(n, euAdj, 0);
    
    // 步骤 6: 去重提取哈密顿回路
    vector<bool> visited(n, false);
    vector<int> tour;
    for (int v : euler) {
        if (!visited[v]) { visited[v] = true; tour.push_back(v); }
    }
    return tour;
}

int main() {
    int n = 5;
    // 城市坐标（欧氏距离）
    vector<pair<double,double>> coords = {{0,0},{1,0},{2,1},{1,2},{0,1}};
    vector<vector<double>> dist(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double dx = coords[i].first - coords[j].first;
            double dy = coords[i].second - coords[j].second;
            dist[i][j] = sqrt(dx*dx + dy*dy);
        }
    
    auto tour = christofidesApprox(n, dist);
    double cost = 0;
    for (int i = 0; i < n; i++) cost += dist[tour[i]][tour[(i+1)%n]];
    
    cout << "路线: ";
    for (int v : tour) cout << v << " ";
    cout << "\n总距离: " << fixed << setprecision(4) << cost << "\n";
    return 0;
}
```

<div data-component="ChristofidesViz"></div>

### 42.2.3 集合覆盖的贪心 ln n 近似

#### 问题定义

**集合覆盖（Set Cover）**：给定全集 $U$（$|U|=n$），以及集合族 $\mathcal{F} = \{S_1, S_2, \ldots, S_m\}$（每个 $S_i \subseteq U$），找到**最少**的集合选取，使其并集覆盖 $U$。

这是 NP-complete 问题，但贪心算法有严格的对数近似保证。

#### 贪心算法

```
GREEDY-SET-COVER(U, F):
  C ← ∅           // 已覆盖的元素
  selected ← []   // 已选集合
  while C ≠ U:
    选取 F 中使 |S \ C| 最大的集合 S*（最多新元素）
    C ← C ∪ S*
    selected.append(S*)
    F ← F \ {S*}
  return selected
```

时间复杂度：$O(n \cdot m \cdot \log n)$（每轮最多 $n$ 次，每轮扫描 $m$ 个集合，每集合操作 $O(n)$）。

#### ln n 近似的证明

设最优覆盖选取了 $k^* = \text{OPT}$ 个集合。

我们跟踪**每次贪心迭代后的未覆盖元素减少情况**：

设在第 $t$ 次迭代前，尚有 $r$ 个元素未被覆盖。OPT 的 $k^*$ 个集合能覆盖这 $r$ 个元素，因此**至少存在一个集合 $S^*$，覆盖 $\geq r/k^*$ 个未覆盖元素**（鸽巢原理）。

贪心选最多的，故 $|S_{\text{greedy}} \setminus C| \geq r/k^*$，未覆盖数减少至少 $r/k^*$。

这意味着每轮后，未覆盖数 $r$ 乘以因子 $(1 - 1/k^*)$：

$$r_t \leq n \cdot \left(1 - \frac{1}{k^*}\right)^t \leq n \cdot e^{-t/k^*}$$

当 $t = k^* \ln n$ 时：$r_t \leq n \cdot e^{-\ln n} = 1$，即未覆盖元素 $< 1$，算法已终止。

因此贪心算法至多选 $k^* \cdot \ln n$ 个集合，**近似比 $= H_n = \ln n + O(1)$**（其中 $H_n$ 是调和级数）。$\square$

> **紧性**：集合覆盖的 $(1-\varepsilon) \ln n$-近似在 P≠NP 下不可能（Dinur & Steurer 2014）。贪心是最优的！

```python
from typing import List, Set, Tuple
import heapq

def greedy_set_cover(universe: Set[int], sets: List[Set[int]]) -> List[int]:
    """
    集合覆盖贪心 ln(n)-近似算法
    
    参数:
        universe: 全集 U（所有需覆盖的元素）
        sets: 集合族，每个元素是 U 的子集
    
    返回:
        selected_indices: 选取的集合在 sets 中的下标列表
    
    时间复杂度：O(|U| × |F| × log|U|)
    近似比：H_{|U|} ≈ ln(|U|) + 0.577（调和级数）
    
    设计考量：每次贪心选覆盖"新元素最多"的集合，
    不是"总元素最多"——注意区别！
    """
    covered = set()          # 已覆盖元素
    selected = []            # 被选取集合的下标
    remaining = set(universe) # 待覆盖元素
    
    while remaining:
        # 找覆盖最多未覆盖元素的集合
        # 边界条件：如果某集合为空或已被选过，覆盖数为 0
        best_idx = -1
        best_count = 0
        
        for i, s in enumerate(sets):
            # 新覆盖数 = 集合 s 中，尚未覆盖的元素数
            new_cover = len(s & remaining)
            if new_cover > best_count:
                best_count = new_cover
                best_idx = i
        
        if best_idx == -1 or best_count == 0:
            # 无法继续覆盖（理论上不应发生，若全集可覆盖）
            break
        
        # 选取最优集合
        selected.append(best_idx)
        # 更新已覆盖集合
        remaining -= sets[best_idx]
    
    return selected


def greedy_set_cover_priority(universe: Set[int], sets: List[Set[int]]) -> List[int]:
    """
    使用最大堆的优化版本（效率更高，适合大规模）
    注意：懒惰删除（lazy deletion）避免重新计算所有集合的覆盖数
    """
    remaining = set(universe)
    element_to_sets = {e: set() for e in universe}
    
    # 建立元素 → 包含该元素的集合字典
    for i, s in enumerate(sets):
        for e in s:
            if e in element_to_sets:
                element_to_sets[e].add(i)
    
    # 最大堆：(-覆盖数, 集合下标)，Python heapq 是最小堆所以取负
    heap = [(-len(s & remaining), i) for i, s in enumerate(sets)]
    heapq.heapify(heap)
    
    selected = []
    covered_count = [0] * len(sets)  # 每个集合实际覆盖的新元素数（懒惰更新）
    
    while remaining:
        # 懒惰删除：弹出时重新验证
        while heap:
            neg_cnt, i = heapq.heappop(heap)
            actual = len(sets[i] & remaining)
            if actual == 0:
                continue
            # 重新推入如果不是当前最优
            if actual < -neg_cnt:
                heapq.heappush(heap, (-actual, i))
                continue
            # 确认是当前最优
            selected.append(i)
            remaining -= sets[i]
            break
    
    return selected


# === 示例 ===
if __name__ == "__main__":
    U = {1, 2, 3, 4, 5, 6, 7, 8}
    F = [
        {1, 2, 3, 4},   # S0
        {2, 4, 6, 8},   # S1
        {1, 3, 5, 7},   # S2
        {5, 6, 7, 8},   # S3
        {1, 5},         # S4
    ]
    result = greedy_set_cover(U, F)
    print(f"选取集合下标: {result}")
    print(f"选取集合: {[F[i] for i in result]}")
    covered = set()
    for i in result: covered |= F[i]
    print(f"覆盖全集: {covered == U}")
    # OPT 可能是 2 个集合（如 {S0, S3} 或 {S1, S2}）
    # 贪心近似比 ≤ ln(8) ≈ 2.08，所以选 ≤ 3 个
```

```cpp
#include <bits/stdc++.h>
using namespace std;

// 集合覆盖贪心 ln(n) 近似
// 时间复杂度：O(|U| × |F| × log|U|)
vector<int> greedySetCover(int n, vector<set<int>>& sets) {
    set<int> remaining;
    for (int i = 0; i < n; i++) remaining.insert(i); // 全集 {0..n-1}
    
    vector<int> selected;
    int m = sets.size();
    
    while (!remaining.empty()) {
        int bestIdx = -1, bestCount = 0;
        
        for (int i = 0; i < m; i++) {
            // 计算该集合覆盖的新元素数
            int cnt = 0;
            for (int e : sets[i])
                if (remaining.count(e)) cnt++;
            if (cnt > bestCount) {
                bestCount = cnt;
                bestIdx = i;
            }
        }
        
        if (bestIdx == -1) break; // 无法继续覆盖
        
        selected.push_back(bestIdx);
        for (int e : sets[bestIdx])
            remaining.erase(e);
    }
    return selected;
}

int main() {
    int n = 8;
    vector<set<int>> F = {
        {0,1,2,3}, {1,3,5,7}, {0,2,4,6}, {4,5,6,7}, {0,4}
    };
    
    auto result = greedySetCover(n, F);
    cout << "选取集合: ";
    for (int i : result) cout << i << " ";
    cout << "\n";
    
    set<int> covered;
    for (int i : result)
        for (int e : F[i]) covered.insert(e);
    cout << "覆盖全集: " << (covered.size() == n ? "是" : "否") << "\n";
    return 0;
}
```

<div data-component="SetCoverGreedy"></div>

### 42.2.4 子集和的 FPTAS

#### 问题回顾

**子集和（Subset Sum）**：给定正整数集合 $S = \{a_1, a_2, \ldots, a_n\}$ 和目标 $t$，是否存在子集其和恰好为 $t$？

这是 NP-complete 问题。但对于其**最大化版本**（找和 $\leq t$ 的子集，使和尽可能大），存在 FPTAS。

#### 精确 DP（伪多项式时间）

先回顾精确算法：设 $L_i$ 为考虑前 $i$ 个数能凑出的所有可能和的集合。

$$L_0 = \{0\}, \quad L_i = L_{i-1} \cup \{x + a_i : x \in L_{i-1}\}$$

最终答案 = $\max\{x \in L_n : x \leq t\}$。

时间复杂度：$O(n \cdot t)$（依赖 $t$ 的大小，当 $t$ 指数大时不是多项式）。

#### FPTAS：缩放 + 舍入

**核心思想**：如果数值太大，就按比例缩小（宽松精度），使 DP 的状态数变为多项式级别。

设 $K = \varepsilon \cdot \max(a_i) / n$（缩放因子）：
1. 将每个 $a_i$ 替换为 $\hat{a}_i = \lfloor a_i / K \rfloor$（向下取整）
2. 对缩放后的集合 $\hat{S}$ 运行精确 DP，目标 $\hat{t} = \lfloor t / K \rfloor$
3. FPTAS 输出的解 $\hat{x}$ 满足：$t / (1+\varepsilon) \leq K \cdot \hat{x} \leq t$

**为什么 $(1+\varepsilon)$-近似？**

设最优解 $x^*$，对应子集 $S^*$，有 $\sum_{a_i \in S^*} a_i = x^* \leq t$。

在缩放后：$\sum_{a_i \in S^*} \hat{a}_i \geq \sum_{a_i \in S^*} (a_i/K - 1) \geq x^*/K - n$

FPTAS 找到缩放后的最优 $\hat{x}^*$，故 $\hat{x}^* \geq x^*/K - n$。

FPTAS 实际输出 $K \hat{x}^* \geq x^* - nK = x^* - \varepsilon \cdot \max(a_i) \geq x^*(1 - \varepsilon)$

（最后一步：$\max(a_i) \leq x^*$，因为 $x^*$ 包含至少一个元素）

**时间复杂度**：缩放后所有 $\hat{a}_i \leq \max(a_i)/K = n/\varepsilon$，故 $\hat{t} \leq t/K \leq nmax(a_i)/(\varepsilon \cdot max(a_i)) = n/\varepsilon$。

DP 状态数 $O(n/\varepsilon)$，每个元素 $O(n/\varepsilon)$，总时间 $O(n^2/\varepsilon)$。这是 $n$ 和 $1/\varepsilon$ 的**多项式**，满足 FPTAS 定义！

```python
from typing import List, Tuple

def subset_sum_fptas(items: List[int], t: int, eps: float) -> Tuple[int, List[int]]:
    """
    子集和最大化问题的 FPTAS
    
    找到和 ≤ t 的子集，使和最接近 t（值 ≥ OPT · 1/(1+ε)）
    
    参数:
        items: 正整数列表
        t: 目标上界
        eps: 近似参数（0 < eps ≤ 1）
    
    返回:
        (近似最优和, 选取元素的下标列表)
    
    时间复杂度：O(n²/ε)
    近似比：1/(1+ε)（即：输出 ≥ OPT/(1+ε)）
    
    注意：n=0 时返回 (0, [])；eps 越小精度越高但越慢
    """
    n = len(items)
    if n == 0:
        return 0, []
    
    # ── 步骤 1: 计算缩放因子 K ──
    # K = ε × max(a) / n
    # 设计考量：使缩放后最大值不超过 n/ε，从而控制 DP 状态数
    max_val = max(items)
    K = eps * max_val / n
    
    # ── 步骤 2: 缩放各元素（向下取整）──
    scaled = [int(a // K) for a in items]  # â_i = ⌊a_i / K⌋
    t_scaled = int(t // K)                 # t̂ = ⌊t / K⌋
    
    # ── 步骤 3: 精确 DP（在缩放值上）──
    # dp[s] = 能凑出缩放后的和 s 所需的最少个数（或用可达性 + 记录路径）
    # 这里用可达集合 + 路径记录
    
    # 节省空间：用字典 {scaled_sum: last_item_index}
    # reachable[s] = -1 表示 s 可用空集达到
    #              = i 表示通过加入第 i 个元素从某前驱状态到达
    INF_SUM = t_scaled + 1
    # dp_set: 当前可达的缩放和
    from_item = {}  # {scaled_sum: 到达该和时最后加入的 item 下标}
    prev_sum = {}   # {scaled_sum: 到达该和时的前驱缩放和}
    reachable = {0}
    from_item[0] = -1
    prev_sum[0] = -1
    
    for i, sv in enumerate(scaled):
        new_reachable = set()
        for s in reachable:
            ns = s + sv
            if ns <= t_scaled and ns not in reachable and ns not in new_reachable:
                new_reachable.add(ns)
                from_item[ns] = i
                prev_sum[ns] = s
        reachable |= new_reachable
    
    # ── 步骤 4: 找最大可达缩放和 ──
    best_scaled = max(reachable)
    
    # ── 步骤 5: 回溯找选取的元素 ──
    chosen = []
    cur = best_scaled
    while cur != 0:
        idx = from_item[cur]
        chosen.append(idx)
        cur = prev_sum[cur]
    
    # 实际输出：选取元素的真实值之和
    actual_sum = sum(items[i] for i in chosen)
    
    return actual_sum, chosen


# === 示例 ===
if __name__ == "__main__":
    items = [100, 200, 300, 400, 500]
    t = 950
    
    for eps in [0.5, 0.2, 0.1, 0.01]:
        approx, chosen = subset_sum_fptas(items, t, eps)
        print(f"ε={eps:.2f}: 近似值={approx}，选取={[items[i] for i in chosen]}")
    
    # 精确最优：100+200+300+350 → 但 350 不在集合中
    # 实际 OPT ≤ 950：选 [200,300,400] = 900，或 [100,400,500]=1000>950 不行
    # OPT = 900  (200+300+400)
    # ε=0.1: 输出 ≥ 900/1.1 ≈ 818，实际往往更好
```

```cpp
#include <bits/stdc++.h>
using namespace std;

// 子集和 FPTAS：O(n²/ε)
// 返回 (近似和, 选取下标列表)
pair<long long, vector<int>> subsetSumFPTAS(vector<int>& items, long long t, double eps) {
    int n = items.size();
    if (n == 0) return {0, {}};
    
    // 步骤 1: 缩放因子
    double maxVal = *max_element(items.begin(), items.end());
    double K = eps * maxVal / n;
    
    // 步骤 2: 缩放
    vector<long long> scaled(n);
    for (int i = 0; i < n; i++) scaled[i] = (long long)(items[i] / K);
    long long tScaled = (long long)(t / K);
    
    // 步骤 3: DP（可达集合 + 路径记录）
    // prevState[s] = {前驱和, 最后加入的 item 下标}
    map<long long, pair<long long,int>> fromState; // sum -> (prev_sum, item_idx)
    fromState[0] = {-1, -1};
    set<long long> reachable = {0};
    
    for (int i = 0; i < n; i++) {
        vector<pair<long long,pair<long long,int>>> toAdd;
        for (long long s : reachable) {
            long long ns = s + scaled[i];
            if (ns <= tScaled && !fromState.count(ns)) {
                toAdd.push_back({ns, {s, i}});
            }
        }
        for (auto& [ns, info] : toAdd) {
            reachable.insert(ns);
            fromState[ns] = info;
        }
    }
    
    // 步骤 4: 最大可达和
    long long bestScaled = *reachable.rbegin();
    
    // 步骤 5: 回溯
    vector<int> chosen;
    long long cur = bestScaled;
    while (cur != 0) {
        auto [prev, idx] = fromState[cur];
        chosen.push_back(idx);
        cur = prev;
    }
    
    long long actualSum = 0;
    for (int i : chosen) actualSum += items[i];
    return {actualSum, chosen};
}

int main() {
    vector<int> items = {100, 200, 300, 400, 500};
    long long t = 950;
    
    for (double eps : {0.5, 0.2, 0.1, 0.01}) {
        auto [approx, chosen] = subsetSumFPTAS(items, t, eps);
        cout << "ε=" << eps << ": 近似值=" << approx << ", 选取=[";
        for (int i : chosen) cout << items[i] << " ";
        cout << "]\n";
    }
    return 0;
}
```

<div data-component="FPTASSubsetSum"></div>

---

## 42.3 随机化算法

### 42.3.1 随机化的动机与分类

**为什么引入随机性？**

对于某些 NP-hard 问题，随机化能带来：
1. **更快的期望时间**（即使不改善最坏情况）：随机快排期望 O(n log n)，最坏 O(n²) 但极罕见
2. **更简单的算法**：随机 2-近似 MAX-CUT 只需对每个顶点抛硬币
3. **更高的近似质量**：某些问题的最优确定性近似比比随机算法更差
4. **避免对抗性输入**：随机化使攻击者无法构造最坏情况

**算法分类（重要概念！）**：

| 类型 | 正确性 | 时间 | 代表算法 |
|---|---|---|---|
| **Las Vegas** | 始终正确 | 随机（期望有限） | 随机快排、随机选择 |
| **Monte Carlo** | 以高概率正确（可能出错） | 确定性多项式 | Miller-Rabin 素性测试、随机 MAX-CUT |
| **Atlantic City** | 以高概率正确 | 以高概率快速 | 某些交互式证明协议 |

> **记忆口诀**：Las Vegas 赌场始终付**钱**（正确），但不保证什么时候付（时间随机）；Monte Carlo 也是赌场，有时可能被**骗**（出错），但每次赌多久是固定的（时间确定）。

### 42.3.2 Las Vegas：随机快排与随机选择

#### 随机快排（Randomized QuickSort）

**确定性快排的问题**：若每次选最大/最小元素作为枢轴（比如对有序输入选第一个元素），退化为 O(n²)。

**随机化**：每次随机选枢轴。

**期望分析**：设 $X_{ij}$（$i < j$）为第 $i$ 小和第 $j$ 小元素在排序过程中被比较的指示变量。

$$E[\text{总比较次数}] = \sum_{i < j} \Pr[X_{ij} = 1]$$

$X_{ij} = 1$ 当且仅当在元素 $\{a_i, a_{i+1}, \ldots, a_j\}$ 中，$a_i$ 或 $a_j$ **最先**被选为枢轴（共 $j-i+1$ 个元素中，$a_i$ 或 $a_j$ 先被选中，概率 $\frac{2}{j-i+1}$）。

$$E[\text{总比较次数}] = \sum_{i<j} \frac{2}{j-i+1} = \sum_{k=1}^{n-1} \sum_{i=1}^{n-k} \frac{2}{k+1} \approx 2n\ln n = O(n \log n)$$

```python
import random
from typing import List, TypeVar, Callable

T = TypeVar('T')

def randomized_quicksort(arr: List[int]) -> List[int]:
    """
    随机快速排序（Las Vegas 算法）
    
    始终正确，期望时间 O(n log n)，最坏 O(n²)（概率极小）
    """
    if len(arr) <= 1:
        return arr
    
    # 核心：随机选枢轴，打破对抗性输入的最坏情况
    pivot_idx = random.randint(0, len(arr) - 1)
    pivot = arr[pivot_idx]
    
    left  = [x for i, x in enumerate(arr) if x < pivot]
    mid   = [x for x in arr if x == pivot]
    right = [x for i, x in enumerate(arr) if x > pivot]
    
    return randomized_quicksort(left) + mid + randomized_quicksort(right)


def randomized_select(arr: List[int], k: int) -> int:
    """
    随机选择第 k 小元素（QuickSelect，Las Vegas 算法）
    
    期望时间 O(n)，最坏 O(n²)
    期望分析：类似随机快排，每次枢轴平均把问题规模减半
    """
    if len(arr) == 1:
        return arr[0]
    
    pivot_idx = random.randint(0, len(arr) - 1)
    pivot = arr[pivot_idx]
    
    left  = [x for x in arr if x < pivot]
    mid   = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    if k <= len(left):
        return randomized_select(left, k)
    elif k <= len(left) + len(mid):
        return pivot
    else:
        return randomized_select(right, k - len(left) - len(mid))
```

```cpp
#include <bits/stdc++.h>
using namespace std;

mt19937 rng(42); // 固定种子便于调试，实际用 random_device 种子

// 随机快速排序（原地版）
// Las Vegas 算法：始终正确，期望 O(n log n)
void randomizedQuicksort(vector<int>& arr, int lo, int hi) {
    if (lo >= hi) return;
    
    // 随机选枢轴（打破对抗性输入）
    int pivotIdx = lo + rng() % (hi - lo + 1);
    swap(arr[pivotIdx], arr[hi]); // 将枢轴移到末尾
    
    int pivot = arr[hi];
    int i = lo - 1;
    for (int j = lo; j < hi; j++) {
        if (arr[j] <= pivot) {
            ++i;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i+1], arr[hi]);
    int pi = i + 1;
    
    randomizedQuicksort(arr, lo, pi - 1);
    randomizedQuicksort(arr, pi + 1, hi);
}

// 随机选择第 k 小（QuickSelect）
// 期望 O(n)，最坏 O(n²)
int randomizedSelect(vector<int>& arr, int lo, int hi, int k) {
    if (lo == hi) return arr[lo];
    
    int pivotIdx = lo + rng() % (hi - lo + 1);
    swap(arr[pivotIdx], arr[hi]);
    
    int pivot = arr[hi], i = lo - 1;
    for (int j = lo; j < hi; j++)
        if (arr[j] <= pivot) swap(arr[++i], arr[j]);
    swap(arr[i+1], arr[hi]);
    int pi = i + 1;
    
    int rankPivot = pi - lo + 1;
    if (k == rankPivot) return arr[pi];
    else if (k < rankPivot) return randomizedSelect(arr, lo, pi-1, k);
    else return randomizedSelect(arr, pi+1, hi, k - rankPivot);
}

int main() {
    vector<int> a = {3,1,4,1,5,9,2,6,5,3};
    randomizedQuicksort(a, 0, a.size()-1);
    cout << "排序后: ";
    for (int x : a) cout << x << " ";
    cout << "\n";
    return 0;
}
```

### 42.3.3 Monte Carlo：随机化 MAX-CUT

**MAX-CUT 问题**：给定图 $G=(V,E)$，将 $V$ 分为 $S$ 和 $T = V \setminus S$ 两部分，最大化切割边数（$u \in S, v \in T$ 的边的数量）。

MAX-CUT 是 NP-hard 问题（实际上是 APX-hard），但有一个极简单的随机算法。

#### 随机 1/2-近似算法

```
RANDOM-MAX-CUT(G):
  独立地将每个顶点以 1/2 概率分到 S，以 1/2 概率分到 T
  输出 cut (S, V\S)
```

**期望分析**：对于每条边 $(u,v)$，它被切割当且仅当 $u,v$ 分属两侧：

$$\Pr[e \text{ 被切割}] = \Pr[u \in S, v \in T] + \Pr[u \in T, v \in S] = \frac{1}{2}$$

（因为 $u,v$ 独立赋值）

$$E[\text{切割边数}] = \sum_{e \in E} \Pr[e \text{ 被切割}] = \frac{|E|}{2}$$

由于 $\text{OPT} \leq |E|$，有 $E[\text{输出}] = |E|/2 \geq \text{OPT}/2$。

**这是 Monte Carlo 2-近似**（最大化问题，输出 $\geq \text{OPT}/2$）！

> **有趣事实**：Goemans-Williamson（1995）用半定规划（SDP）实现了 0.878-近似，这是目前 MAX-CUT 最好的多项式近似比，对应于 Unique Games Conjecture 的近似硬度壁垒。

```python
import random
from typing import List, Tuple, Set

def random_max_cut(n: int, edges: List[Tuple[int,int]], 
                   trials: int = 10) -> Tuple[Set[int], Set[int], int]:
    """
    MAX-CUT 随机 1/2-近似算法（Monte Carlo）
    
    参数:
        n: 顶点数
        edges: 边列表
        trials: 重复次数（增加可靠性，期望切割数随 trials 提高趋近 |E|/2）
    
    返回:
        (S, T, 切割边数)
    
    期望切割数 ≥ |E|/2 ≥ OPT/2
    """
    best_cut = 0
    best_S = set()
    
    for _ in range(trials):
        # 每个顶点以 1/2 概率进入 S
        S = {v for v in range(n) if random.random() < 0.5}
        T = set(range(n)) - S
        
        # 计算切割边数
        cut = sum(1 for u, v in edges if (u in S) != (v in S))
        
        if cut > best_cut:
            best_cut = cut
            best_S = S
    
    return best_S, set(range(n)) - best_S, best_cut


def deterministic_local_search_max_cut(n: int, edges: List[Tuple[int,int]]) -> Tuple[Set[int], int]:
    """
    局部搜索近似 MAX-CUT（确定性 1/2-近似）
    
    初始随机分割，迭代地将能增加切割数的顶点移到另一侧
    最终局部最优解保证 ≥ |E|/2（每条边若在同一侧，移动一端可增加切割）
    """
    S = {v for v in range(n) if random.random() < 0.5}
    
    improved = True
    while improved:
        improved = False
        for v in range(n):
            # 计算 v 在当前侧的切割贡献 vs 移到另一侧的收益
            in_cut = sum(1 for u in range(n) if (u, v) in set(edges) or (v, u) in set(edges)
                        if (u in S) != (v in S))
            # 如果移动 v 能增加切割数（即 v 当前侧的邻居比另一侧多），则移动
            neighbors = [u for u, w in edges if w == v] + [w for u, w in edges if u == v]
            same_side = sum(1 for u in neighbors if (u in S) == (v in S))
            other_side = len(neighbors) - same_side
            if same_side > other_side:
                if v in S: S.remove(v)
                else: S.add(v)
                improved = True
    
    cut = sum(1 for u, v in edges if (u in S) != (v in S))
    return S, cut
```

```cpp
#include <bits/stdc++.h>
using namespace std;
mt19937 rng(random_device{}());

// 随机 MAX-CUT（Monte Carlo 1/2-近似）
// 期望切割边数 ≥ |E|/2 ≥ OPT/2
pair<vector<bool>, int> randomMaxCut(int n, vector<pair<int,int>>& edges, int trials=10) {
    int bestCut = 0;
    vector<bool> bestAssign(n);
    
    for (int t = 0; t < trials; t++) {
        // 每个顶点以 1/2 概率进入 S（assign=true）
        vector<bool> assign(n);
        for (int v = 0; v < n; v++) assign[v] = rng() & 1;
        
        int cut = 0;
        for (auto& [u, v] : edges)
            if (assign[u] != assign[v]) cut++;
        
        if (cut > bestCut) {
            bestCut = cut;
            bestAssign = assign;
        }
    }
    return {bestAssign, bestCut};
}

int main() {
    int n = 6;
    vector<pair<int,int>> edges = {{0,1},{0,2},{1,3},{2,3},{3,4},{4,5},{0,5}};
    
    auto [assign, cut] = randomMaxCut(n, edges, 20);
    cout << "切割边数: " << cut << " / " << edges.size() << "\n";
    cout << "S: ";
    for (int v = 0; v < n; v++) if (assign[v]) cout << v << " ";
    cout << "\n";
    return 0;
}
```

<div data-component="MaxCutRandomized"></div>

### 42.3.4 Chernoff 界简介

在分析随机算法时，我们经常需要估计"一系列独立随机事件的总和大/小于期望多少"——这正是 **Chernoff 界（Chernoff Bound）** 的用武之地。

#### 基本形式

设 $X_1, X_2, \ldots, X_n$ 是 n 个独立的 0/1 随机变量，$X = \sum X_i$，$\mu = E[X]$。对任意 $\delta > 0$：

$$\Pr[X \geq (1+\delta)\mu] \leq \left(\frac{e^\delta}{(1+\delta)^{(1+\delta)}}\right)^\mu \leq e^{-\mu\delta^2/3}$$

$$\Pr[X \leq (1-\delta)\mu] \leq e^{-\mu\delta^2/2}$$

**直觉**：随机变量和偏离期望超过 $\delta\mu$ 的概率，随 $n$ 指数级降低——大数定律的定量精确版。

#### 典型应用

1. **负载均衡（哈希）**：将 $n$ 个元素随机哈希到 $n$ 个桶。每个桶期望 $\mu=1$ 个元素。用 Chernoff：最大桶的负载 $\leq O(\log n / \log\log n)$，以高概率成立。

2. **随机快排分析**：好枢轴（划分比在 25%~75% 之间）的概率 $\geq 1/2$。K 轮中至少 K/3 轮为好枢轴的概率 $\geq 1 - e^{-K/72}$（Chernoff），从而期望深度 $O(\log n)$。

3. **集合覆盖的随机算法**：LP 分数解取整后，每个元素被覆盖的概率 $\geq 1/2$。独立重复 $O(\log n)$ 轮，用 Chernoff 证明以高概率每个元素至少被覆盖一次。

### 42.3.5 Miller-Rabin 素性测试（Monte Carlo 经典案例）

尽管素性测试（AKS 算法）在 2002 年已被证明在 P 中，工业中广泛使用的仍是更快的随机化 Miller-Rabin 测试。

**核心思路**：若 $n$ 是奇素数，对任意 $a$ 不被 $n$ 整除，Fermat 定理给出约束；Miller-Rabin 用更强的条件（二次余数的性质）检测。若 $n$ 是合数，至少 $3/4$ 的 $a$ 会"暴露"合数性质（强伪证人，strong witness）。

每次独立选 $a$，合数逃脱检测的概率 $\leq 1/4$。重复 $k$ 次后，合数逃脱的概率 $\leq (1/4)^k$。取 $k=40$，错误概率 $< 10^{-24}$，实践中可忽略不计。

```python
import random

def miller_rabin(n: int, k: int = 40) -> bool:
    """
    Miller-Rabin 素性测试（Monte Carlo 算法）
    
    参数:
        n: 待测整数
        k: 重复次数（误判概率 ≤ (1/4)^k）
    
    返回:
        True = 极可能是素数；False = 确定是合数
    
    时间复杂度：O(k × log²n × log(log n))（使用快速幂取模）
    
    陷阱：返回 True 是"概率性"的，不是"确定性"的。
          处理密钥生成等安全敏感场景，k 至少取 40。
    """
    if n < 2: return False
    if n == 2 or n == 3: return True
    if n % 2 == 0: return False
    
    # 写 n-1 = 2^r × d（d 为奇数）
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    
    def is_composite_witness(a: int) -> bool:
        """检查 a 是否是 n 为合数的见证（若 True = 确定是合数）"""
        x = pow(a, d, n)  # 快速幂取模
        if x == 1 or x == n - 1:
            return False  # 通过测试（可能是素数）
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                return False
        return True  # a 是合数见证
    
    for _ in range(k):
        a = random.randrange(2, n - 1)
        if is_composite_witness(a):
            return False  # 确定是合数
    
    return True  # 以极高概率是素数


# === 示例 ===
if __name__ == "__main__":
    primes = [2, 17, 101, 997, 7919, 104729]
    composites = [4, 15, 100, 561, 1729]  # 561, 1729 是 Carmichael 数（欺骗 Fermat 测试）
    
    print("素数检测:")
    for p in primes:
        print(f"  {p}: {miller_rabin(p)}")  # 应全为 True
    
    print("合数检测:")
    for c in composites:
        print(f"  {c}: {miller_rabin(c)}")  # 应全为 False
```

---

## 42.4 启发式算法简介

当精确算法太慢，近似算法保证不够好（或难以设计），启发式算法是工程实践中最后的武器。启发式没有理论保证，但在实践中往往表现出色。

### 42.4.1 局部搜索（Local Search）

**基本框架**：

```
LOCAL-SEARCH(problem):
  s ← 初始可行解（可以随机生成，或贪心构造）
  while 存在 s 的邻域 N(s) 中某个解 s' 比 s 更好:
    s ← s'              // 移动到更好的邻居
  return s               // 局部最优解
```

**邻域的定义**因问题而异，这是局部搜索设计的核心：
- **TSP**：2-opt 邻域（删掉两条边，重新连接，若缩短则接受）、3-opt、Lin-Kernighan
- **图着色**：交换一个节点的颜色
- **调度问题**：交换两个任务的分配

**局部最优 vs 全局最优**：

局部搜索找到的是**局部最优**（local optimum）——无法通过单步改进的解。对于非凸问题，可能大量存在局部最优，离全局最优差距很大。

```
         全局最优
            ↓
    ████████████████
   ██            ████
  ██  局部最优    ████    ← 卡在这里！
 ████     ↓      ████
████████████████████
```

**2-opt TSP**：

```python
def two_opt_tsp(tour: List[int], dist: List[List[float]]) -> List[int]:
    """
    TSP 2-opt 局部搜索
    
    反复尝试：删去两条不相邻的边，以另一种方式重连（翻转中间段）
    若总路程减小则接受改变。直到无2-opt改进为止。
    
    时间复杂度：每轮 O(n²)，轮数不确定（无多项式上界）
    实践中：通常很快收敛，质量远好于随机初始解
    """
    n = len(tour)
    improved = True
    best = tour[:]
    best_cost = sum(dist[best[i]][best[(i+1)%n]] for i in range(n))
    
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                # 尝试反转 tour[i+1..j] 段
                # 原来: ... → tour[i] → tour[i+1] → ... → tour[j] → tour[j+1] → ...
                # 变成: ... → tour[i] → tour[j]  → ... → tour[i+1] → tour[j+1] → ...
                new_tour = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                new_cost = sum(dist[new_tour[k]][new_tour[(k+1)%n]] for k in range(n))
                if new_cost < best_cost - 1e-9:
                    best = new_tour
                    best_cost = new_cost
                    improved = True
    return best
```

### 42.4.2 模拟退火（Simulated Annealing, SA）

局部搜索的最大缺陷是**陷入局部最优**。**模拟退火**通过引入"以一定概率接受劣解"来逃脱局部最优，灵感来自冶金学中的退火过程。

#### 直觉类比

想象你在一片丘陵地形中寻找最低点（全局最小值）。普通局部搜索只会一路向下走，最终被困在某个山谷（局部最小值）。

模拟退火则像一个**疯狂但逐渐冷静的探险者**：开始时不论上山还是下山都愿意走；随着时间流逝，他逐渐"冷静"，越来越不愿意上山；最终完全不上山，安稳地停在某个山谷。关键是：初期的探险让他有机会翻过小山发现更低的深谷。

#### 算法描述

```
SIMULATED-ANNEALING(problem, T_init, T_min, cooling_rate):
  s ← 初始解（随机或贪心）
  T ← T_init                // 初始「温度」（高 = 更愿意接受劣解）
  best ← s
  while T > T_min:
    s' ← 从 s 的邻域随机选一个邻居
    ΔE ← cost(s') - cost(s) // ΔE > 0 表示 s' 比 s 差（最小化问题）
    if ΔE < 0:
      s ← s'                // 总是接受改进
    else:
      以概率 e^{-ΔE/T} 接受 s'  // 以小概率接受劣解（温度越高，概率越大）
    if cost(s) < cost(best):
      best ← s              // 记录历史最好解
    T ← T × cooling_rate    // 降温（cooling）
  return best
```

#### 关键参数

- **初始温度 $T_0$**：应足够高使得初期大多数劣解都被接受（约 $e^{-\Delta E_{\max}/T_0} \approx 0.9$）
- **终止温度 $T_{\min}$**：接近 0，此时几乎不接受劣解
- **冷却速率 $\alpha \in (0,1)$**：典型值 $0.95$ 到 $0.999$，越接近 1 冷却越慢，探索越充分
- **每温度步数（Markov 链长度）**：在每个温度下做多少次随机游走

```python
import random
import math
from typing import List, Callable, TypeVar

T_sol = TypeVar('T_sol')

def simulated_annealing(
    initial_solution,
    neighbor_fn: Callable,          # 生成邻居的函数
    cost_fn: Callable,              # 计算解的代价（越小越好）
    T_init: float = 1000.0,
    T_min: float = 1.0,
    alpha: float = 0.99,            # 冷却速率
    steps_per_temp: int = 100,      # 每温度步数
):
    """
    通用模拟退火框架
    
    以温度 T 控制接受劣解的概率：P(接受) = e^{-ΔE/T}
    T 逐步降低，使算法从全局探索逐渐收敛到局部精炼
    
    时间复杂度：O(steps_per_temp × log(T_init/T_min) / log(1/alpha))
    无理论近似比保证，实践中效果优异
    """
    current = initial_solution
    current_cost = cost_fn(current)
    best = current
    best_cost = current_cost
    
    T = T_init
    
    while T > T_min:
        for _ in range(steps_per_temp):
            # 生成随机邻居
            neighbor = neighbor_fn(current)
            neighbor_cost = cost_fn(neighbor)
            delta = neighbor_cost - current_cost  # 代价差
            
            # Metropolis 准则
            if delta < 0:
                # 接受更好的解
                current = neighbor
                current_cost = neighbor_cost
            else:
                # 以概率 e^{-ΔE/T} 接受劣解（逃离局部最优）
                # 温度越高，接受概率越大；ΔE 越小，接受概率越大
                accept_prob = math.exp(-delta / T)
                if random.random() < accept_prob:
                    current = neighbor
                    current_cost = neighbor_cost
            
            # 更新最优解
            if current_cost < best_cost:
                best = current
                best_cost = current_cost
        
        T *= alpha  # 降温
    
    return best, best_cost


# === 应用于 TSP ===
def sa_tsp(n: int, dist: List[List[float]]) -> List[int]:
    """使用模拟退火求解 TSP"""
    import random
    
    def tour_cost(tour: List[int]) -> float:
        return sum(dist[tour[i]][tour[(i+1)%n]] for i in range(n))
    
    def random_neighbor(tour: List[int]) -> List[int]:
        """2-opt 邻居：随机选两个位置，翻转中间段"""
        i, j = sorted(random.sample(range(n), 2))
        new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
        return new_tour
    
    # 初始解：随机排列
    initial = list(range(n))
    random.shuffle(initial)
    
    best_tour, best_cost = simulated_annealing(
        initial_solution=initial,
        neighbor_fn=random_neighbor,
        cost_fn=tour_cost,
        T_init=max(dist[i][j] for i in range(n) for j in range(n)) * n,
        T_min=0.01,
        alpha=0.995,
        steps_per_temp=n * 10,
    )
    return best_tour
```

```cpp
#include <bits/stdc++.h>
using namespace std;
mt19937 rng(42);

// 模拟退火 TSP
// 参数：T0=初温，Tmin=终温，alpha=冷却率，stepsPerTemp=每温度步数
vector<int> simulatedAnnealingTSP(int n, vector<vector<double>>& dist,
    double T0 = 1000, double Tmin = 1.0, double alpha = 0.995, int stepsPerTemp = 300) {
    
    auto tourCost = [&](vector<int>& t) {
        double c = 0;
        for (int i = 0; i < n; i++) c += dist[t[i]][t[(i+1)%n]];
        return c;
    };
    
    // 初始解：0,1,2,...,n-1
    vector<int> cur(n); iota(cur.begin(), cur.end(), 0);
    shuffle(cur.begin(), cur.end(), rng);
    double curCost = tourCost(cur);
    
    vector<int> best = cur;
    double bestCost = curCost;
    
    for (double T = T0; T > Tmin; T *= alpha) {
        for (int step = 0; step < stepsPerTemp; step++) {
            // 2-opt 随机邻居
            int i = rng() % n, j = rng() % n;
            if (i > j) swap(i, j);
            if (i == j) continue;
            
            // 翻转 cur[i+1..j]
            vector<int> nb = cur;
            reverse(nb.begin() + i, nb.begin() + j + 1);
            double nbCost = tourCost(nb);
            double delta = nbCost - curCost;
            
            // Metropolis 准则
            if (delta < 0 || (double)rng() / rng.max() < exp(-delta / T)) {
                cur = nb;
                curCost = nbCost;
                if (curCost < bestCost) { best = cur; bestCost = curCost; }
            }
        }
    }
    return best;
}

int main() {
    int n = 8;
    vector<pair<double,double>> coords = {{0,0},{2,0},{4,1},{5,3},{4,5},{2,6},{0,5},{-1,3}};
    vector<vector<double>> dist(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double dx = coords[i].first - coords[j].first;
            double dy = coords[i].second - coords[j].second;
            dist[i][j] = sqrt(dx*dx + dy*dy);
        }
    
    auto tour = simulatedAnnealingTSP(n, dist);
    double cost = 0;
    for (int i = 0; i < n; i++) cost += dist[tour[i]][tour[(i+1)%n]];
    cout << "SA-TSP 路程: " << fixed << setprecision(4) << cost << "\n";
    return 0;
}
```

<div data-component="SimulatedAnnealingTSP"></div>

### 42.4.3 遗传算法（Genetic Algorithms, GA）

遗传算法是一类**基于种群的**元启发式算法，模拟自然界的进化过程：适者生存（选择）、基因交叉（交叉/杂交）、基因突变（变异）。

#### 算法框架

```
GENETIC-ALGORITHM(population_size, max_generations):
  P ← 随机初始化 population_size 个解（个体）
  evaluate(P)                              // 计算每个个体的适应度
  for gen = 1 to max_generations:
    parents ← select(P)                   // 选择：倾向于选适应度高的个体
    offspring ← crossover(parents)        // 交叉：两个亲本产生后代
    offspring ← mutate(offspring)         // 变异：随机小改动
    P ← survive(P ∪ offspring)            // 淘汰低适应度个体
  return best individual in P
```

#### 关键算子（以 TSP 为例）

**选择（Selection）**：
- **轮盘赌**：每个个体被选中的概率正比于其适应度
- **锦标赛**：随机选 $k$ 个个体，取其中最优者进入下一代

**交叉（Crossover）**专为 TSP 设计：
- **部分映射交叉（PMX）**：保留亲本1 的一个子序列，其余按亲本2 的顺序填入，避免重复。
- **顺序交叉（OX）**：类似，但以顺序为核心保留。

**变异（Mutation）**：
- **互换变异（Swap Mutation）**：随机交换两个城市
- **逆转变异（Inversion Mutation）**：即 2-opt 翻转，随机翻转一个子区间

```python
import random
from typing import List, Callable, Tuple

def genetic_algorithm_tsp(
    n: int,
    dist: List[List[float]],
    pop_size: int = 100,
    max_gen: int = 500,
    mutation_rate: float = 0.02,
    tournament_size: int = 5,
) -> List[int]:
    """
    遗传算法求解 TSP（教学版）
    
    个体表示：城市访问顺序（排列）
    适应度：路程的倒数（越短越好）
    交叉：顺序交叉（OX）
    变异：随机逆转
    选择：锦标赛选择
    
    无理论近似保证，实践中效果良好，尤其对中型 TSP 实例
    """
    
    def tour_cost(tour: List[int]) -> float:
        return sum(dist[tour[i]][tour[(i+1) % n]] for i in range(n))
    
    def random_tour() -> List[int]:
        t = list(range(n))
        random.shuffle(t)
        return t
    
    def tournament_select(pop: List[List[int]]) -> List[int]:
        """锦标赛选择：k 个候选中取最短路线"""
        candidates = random.sample(pop, tournament_size)
        return min(candidates, key=tour_cost)
    
    def ox_crossover(p1: List[int], p2: List[int]) -> List[int]:
        """顺序交叉（Order Crossover, OX）"""
        size = len(p1)
        start, end = sorted(random.sample(range(size), 2))
        
        # 保留 p1 的 [start, end] 段
        child = [-1] * size
        child[start:end+1] = p1[start:end+1]
        inherited = set(p1[start:end+1])
        
        # 按 p2 的顺序填入剩余
        fill_pos = (end + 1) % size
        for city in p2[end+1:] + p2[:end+1]:
            if city not in inherited:
                child[fill_pos] = city
                fill_pos = (fill_pos + 1) % size
        return child
    
    def mutate(tour: List[int]) -> List[int]:
        """逆转变异：以 mutation_rate 概率随机翻转一段"""
        if random.random() < mutation_rate:
            i, j = sorted(random.sample(range(n), 2))
            tour[i:j+1] = tour[i:j+1][::-1]
        return tour
    
    # 初始化种群
    population = [random_tour() for _ in range(pop_size)]
    best_tour = min(population, key=tour_cost)
    best_cost = tour_cost(best_tour)
    
    for gen in range(max_gen):
        new_population = []
        
        # 精英保留（Elitism）：保留最好的 2 个个体
        population.sort(key=tour_cost)
        new_population.extend([population[0][:], population[1][:]])
        
        # 交叉 + 变异生成新个体
        while len(new_population) < pop_size:
            p1 = tournament_select(population)
            p2 = tournament_select(population)
            child = ox_crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)
        
        population = new_population
        
        # 更新全局最优
        gen_best = min(population, key=tour_cost)
        gen_cost = tour_cost(gen_best)
        if gen_cost < best_cost:
            best_cost = gen_cost
            best_tour = gen_best[:]
        
        # 早停：如果连续 50 代无改进可停止
    
    return best_tour


# === 对比实验 ===
if __name__ == "__main__":
    import math
    
    # 生成随机城市
    n = 20
    random.seed(42)
    coords = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]
    dist = [[math.sqrt((x1-x2)**2 + (y1-y2)**2) for x2,y2 in coords] for x1,y1 in coords]
    
    # 贪心初始（最近邻）
    def nearest_neighbor(start: int) -> List[int]:
        visited = {start}
        tour = [start]
        for _ in range(n - 1):
            last = tour[-1]
            nxt = min((v for v in range(n) if v not in visited), key=lambda v: dist[last][v])
            visited.add(nxt)
            tour.append(nxt)
        return tour
    
    nn_tour = nearest_neighbor(0)
    nn_cost = sum(dist[nn_tour[i]][nn_tour[(i+1)%n]] for i in range(n))
    
    ga_tour = genetic_algorithm_tsp(n, dist, pop_size=50, max_gen=200)
    ga_cost = sum(dist[ga_tour[i]][ga_tour[(i+1)%n]] for i in range(n))
    
    print(f"最近邻贪心: {nn_cost:.2f}")
    print(f"遗传算法  : {ga_cost:.2f}")
    print(f"改进率    : {(nn_cost - ga_cost) / nn_cost * 100:.1f}%")
```

<div data-component="GeneticAlgorithmViz"></div>

### 42.4.4 启发式 vs 近似算法：如何选择？

面对 NP-hard 问题，工程师需要做出策略决策：

| 维度 | 近似算法 | 启发式算法 |
|---|---|---|
| **理论保证** | ✅ 严格近似比（如 2x、ln n x OPT） | ❌ 无保证 |
| **适用场景** | 需要可证明质量下界 | 只需实用效果好 |
| **算法复杂度** | 通常较复杂（LP松弛、组合论证） | 通常简单易实现 |
| **调参需求** | 少（参数少，鲁棒性好） | 多（温度、种群大小、迭代次数） |
| **代表工具** | 顶点覆盖2x、Christofides1.5x | SA、GA、LKH for TSP |
| **工业场景** | 法律/合同保证、安全关键系统 | 大规模路由、调度、游戏AI |

**实践建议**：

1. **首先尝试近似算法**：若有多项式时间近似算法（如顶点覆盖、集合覆盖），优先使用，因为有保证。

2. **若近似比不够好（如 ln n 对 n=10⁶ 约 14 倍），考虑启发式**：实践中启发式往往比最坏近似比好得多。

3. **问题规模是关键**：$n \leq 50$ 时，精确算法（分支定界、整数规划）可行；$n \leq 10^4$ 时，LKH 等高质量启发式；$n > 10^5$ 时，必须用简单启发式或问题特有结构。

4. **工业界的真实做法**：将近似算法作为启发式的**初始解**，再用局部搜索精炼。例如：Christofides 1.5x 输出作为 2-opt/3-opt 的初始解，往往能找到非常接近最优的解。

<div data-component="ApproxVsHeuristic"></div>

---

## 42.5 本章总结与面试高频考点

### 知识体系全图

```
处理 NP-hard 问题
├── 近似算法（多项式时间 + 可证明近似比）
│   ├── 贪心近似：顶点覆盖 2x、集合覆盖 ln(n)x
│   ├── 组合近似：Christofides TSP 1.5x（MST + 奇度匹配）
│   ├── DP 缩放：子集和 FPTAS（ε-近似，poly(n, 1/ε)）
│   └── 质量层次：FPTAS ⊂ PTAS ⊂ 固定近似比 ⊂ APX-hard → 无近似
├── 随机化算法
│   ├── Las Vegas（始终正确，时间随机）：随机快排、随机选择
│   ├── Monte Carlo（时间确定，概率正确）：Miller-Rabin、随机 MAX-CUT
│   └── 分析工具：Chernoff 界（尾界概率估计）
└── 启发式算法（实践效果好，无理论保证）
    ├── 局部搜索：2-opt TSP、爬山法
    ├── 模拟退火（SA）：以概率接受劣解，逃离局部最优
    └── 遗传算法（GA）：选择 + 交叉 + 变异，基于种群进化
```

### 面试高频考点

**Q1：顶点覆盖 2-近似为什么是正确的？**

> 算法选出的边集 $M$ 是一个匹配（因为每条边选后删去关联边，后续不共享端点）。$|C| = 2|M|$。任何覆盖（包括 OPT）必须对匹配 $M$ 的每条边选一个端点，由于 $M$ 是匹配，OPT $\geq |M|$。因此 $|C| = 2|M| \leq 2 \cdot \text{OPT}$。✓

**Q2：Monte Carlo 与 Las Vegas 的区别？**

> Las Vegas：**总是正确**，运行时间随机（随机快排总给出正确排序，但时间因随机枢轴而变）。Monte Carlo：**固定多项式时间**，以小概率出错（Miller-Rabin 以 $(1/4)^k$ 概率误判合数为素数）。

**Q3：为什么一般 TSP（无三角不等式）无任何常数近似比（除非 P=NP）？**

> 因为若存在 $\rho$-近似（$\rho$ 任意大整数），可将哈密顿回路问题规约至其：构造完全图，存在边权 1，不存在的边权 $\rho n + 1$。若有哈密顿回路，$\text{OPT} = n$，近似输出 $\leq \rho n$ 即只走权 1 的边 = 找到哈密顿回路；若无，任何回路代价 $> \rho n$。从而区分哈密顿问题（NPC），矛盾。

**Q4：FPTAS 和 PTAS 的区别？**

> PTAS：$(1+\varepsilon)$-近似，时间是 $n$ 的多项式（但可以是 $\varepsilon$ 的指数），如 $O(n^{1/\varepsilon})$。FPTAS：时间同时是 $n$ 和 $1/\varepsilon$ 的多项式，如 $O(n^2/\varepsilon)$。FPTAS 更强：令 $\varepsilon$ 很小也能高效运行。背包/子集和有 FPTAS，某些问题只有 PTAS，APX-hard 问题连 PTAS 都没有。

---

> **⚠️ 常见错误与陷阱**
>
> - Christofides 1.5 近似**只对满足三角不等式的 TSP 有效**（如欧氏平面上的城市、道路网络）；对一般 TSP，只要不满足三角不等式，即无常数近似
> - 集合覆盖的 $\ln n$ 近似比是**紧的**（tight）：存在实例使贪心恰好选 $\ln n \times \text{OPT}$ 个集合
> - 模拟退火和遗传算法**没有近似比保证**，只有实践经验。在面试中被问"近似算法"时，请回答有理论保证的算法（顶点覆盖、集合覆盖等），而非 SA/GA
> - 随机 MAX-CUT 是 Monte Carlo（可能选到很差的分割，但期望好），而非 Las Vegas

**参考资料**：
- CLRS 第 4 版 Chapter 35（近似算法）
- Vazirani《Approximation Algorithms》教材（深入系统）
- Dasgupta, Papadimitriou, Vazirani《Algorithms》Chapter 9
- Christofides 1976 原始技术报告
- Motwani & Raghavan《Randomized Algorithms》（随机化算法经典教材）

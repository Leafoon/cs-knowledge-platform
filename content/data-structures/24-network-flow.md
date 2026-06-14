# Chapter 24: 网络流（Network Flow）

## 章节导读

想象一张错综复杂的水管网络——水从一个总水源（source）流出，穿过若干中间节点（水泵站、分配站），最终汇聚到一个总汇点（sink）。每段管道都有**最大承载流量上限（容量 capacity）**，问：在不违反任何管道上限的前提下，**从水源到汇点，最多能输送多少水**？

这就是**最大流（Maximum Flow）**问题。在计算机科学与运筹学中，它有着极为广泛的应用：

- **物流调度**：货物从仓库出发，经由若干中转站，最多能运多少到超市？
- **通信网络**：路由器之间的链路带宽有限，从服务器到客户端最大吞吐量是多少？
- **二部图匹配**：招聘岗位与候选人之间如何实现最多的一对一匹配？
- **图像分割（Image Segmentation）**：把图像像素分割为"前景"和"背景"两类，最小代价的分割方案是什么？
- **项目选择（Project Selection）**：哪些项目应该做、哪些放弃，以最大化收益减去成本？

本章将系统介绍网络流理论的核心：从流网络的基本定义出发，引出**最大流最小割定理**（整个理论的皇冠），然后逐步深入三种重要算法：**Ford-Fulkerson 方法**（理论框架）、**Edmonds-Karp 算法**（多项式保证）和 **Dinic 算法**（工程实践中的最优解）。最后，通过经典应用问题理解最大流如何统一描述众多看似不同的问题。

> **前置知识**：Chapter 18（图的表示）、Chapter 19（BFS/DFS）

---

## 24.1 网络流基础

### 24.1.1 流网络的严格定义

**流网络（Flow Network）** 是一个有向图 $G = (V, E)$，带有以下附加结构：

1. **容量函数 $c: V \times V \to \mathbb{R}_{\geq 0}$**：对每条有向边 $(u, v) \in E$，$c(u, v) \geq 0$ 表示该边的最大允许流量（容量）。若 $(u, v) \notin E$，则 $c(u, v) = 0$。

2. **源点（Source）$s \in V$**：只有流出，没有流入（外部注入点）。

3. **汇点（Sink）$t \in V$，$t \neq s$**：只有流入，没有流出（外部收集点）。

4. **特殊假设**（简化理论分析，CLRS 约定）：
   - 图中没有自环（self-loop）
   - 对任意 $(u, v) \in E$，若 $(v, u)$ 也存在，则需要特殊处理（后文给出方案）
   - 每个顶点都位于某条 $s \to t$ 路径上（孤立节点不影响最大流）

**流函数（Flow Function）** $f: V \times V \to \mathbb{R}$ 必须满足以下两个约束：

**① 容量约束（Capacity Constraint）**：每条边上的流量不能超过容量上限，且不能为负。

$$0 \leq f(u,v) \leq c(u,v), \quad \forall\, u,v \in V$$

**② 流守恒（Flow Conservation）**：对每个中间节点，流入量等于流出量（不积累、不凭空产生）。

$$\sum_{v \in V} f(v, u) = \sum_{v \in V} f(u, v), \quad \forall\, u \in V,\; u \neq s,\; u \neq t$$

就像水管网络中的分流节点：进来多少水，就必须出去多少水。

**网络的总流量（Flow Value）**：

$$|f| = \sum_{v \in V} f(s, v) - \sum_{v \in V} f(v, s)$$

即从源点 $s$ 净流出的总量。目标：**最大化 $|f|$**。

> **小技巧**：如果有向图中同时存在 $(u, v)$ 和 $(v, u)$ 两条边，可以引入一个虚拟中间节点 $x$，把 $(u, v)$ 拆分为 $(u, x)$ 和 $(x, v)$，容量不变，从而消除"反向平行边"。

<div data-component="FlowNetworkBasics"></div>

### 24.1.2 一个具体例子

考虑以下流网络（本章贯穿始终的例子）：

```
节点: s, A, B, C, D, t
有向边与容量:
  s → A: 10
  s → C: 10
  A → B: 4
  A → C: 2
  A → D: 8
  C → D: 9
  B → t: 10
  D → t: 10
```

这个网络的最大流是 **14**，后续我们将通过增广路径算法一步步求出。可以先直观地验证：

- $t$ 的入边总容量：$B \to t: 10$ + $D \to t: 10 = 20$，但 $D$ 的入边来自 $A$ 和 $C$，形成瓶颈。
- $C$ 的出边只有 $C \to D: 9$；$A$ 的出边总量受限于 $s \to A: 10$。

**最终合法的最大流分配**（三条增广路径）：

| 增广路径 | 流量 |
|:---|:---:|
| $s \to A \to B \to t$ | 4 |
| $s \to A \to D \to t$ | 6 |
| $s \to C \to D \to t$ | 4 |
| **总计** | **14** |

**逐边流量验证（流量 / 容量）**：

| 边 | 流量 / 容量 | 说明 |
|:---|:---:|:---|
| $s \to A$ | 10 / 10 | 满载（4 去 B，6 去 D）|
| $s \to C$ | 4 / 10 | 未满 |
| $A \to B$ | 4 / 4 | 饱和 |
| $A \to C$ | 0 / 2 | 未使用 |
| $A \to D$ | 6 / 8 | 未满 |
| $C \to D$ | 4 / 9 | 未满 |
| $B \to t$ | 4 / 10 | 未满 |
| $D \to t$ | 10 / 10 | 饱和 |

流守恒验证：$A$ 流入 10 = 流出 $4+0+6=10$ ✓；$C$ 流入 4 = 流出 4 ✓；$D$ 流入 $6+4=10$ = 流出 10 ✓。

### 24.1.3 s-t 割的定义与容量

**s-t 割（s-t Cut）** 是对顶点集 $V$ 的一个划分 $(S, T)$，满足 $s \in S$，$t \in T$。

**割的容量（Cut Capacity）**：

$$c(S, T) = \sum_{u \in S, v \in T} c(u, v)$$

注意：只计算从 $S$ 到 $T$ 方向的边，不计算从 $T$ 到 $S$ 的反向边。

**直觉**：割容量就是"切断哪些管道能彻底阻断水流"的代价总和。割掉 $S$ 到 $T$ 的所有边，网络就无流可通了。

**引理（流量 ≤ 割容量）**：对任意流 $f$ 和任意 s-t 割 $(S, T)$：

$$|f| \leq c(S, T)$$

**证明**：
$$|f| = \sum_{u \in S, v \in T} f(u, v) - \sum_{u \in T, v \in S} f(u, v) \leq \sum_{u \in S, v \in T} f(u, v) \leq \sum_{u \in S, v \in T} c(u, v) = c(S, T)$$

这给出了最大流的一个**上界**：任何 s-t 割容量都是最大流的上界。因此：

$$\text{最大流} \leq \min \text{割容量}$$

**最大流最小割定理**将证明等号成立，这是第 24.1.5 节的内容。

### 24.1.4 最大流最小割定理

> **Max-Flow Min-Cut Theorem（最大流最小割定理）**：
> 在任意流网络中，以下三个陈述等价：
> 1. $f$ 是一个最大流
> 2. 残差网络 $G_f$ 中不存在增广路径
> 3. 存在某个 s-t 割 $(S, T)$ 使得 $c(S, T) = |f|$

该定理是整个网络流理论的核心。它告诉我们：**"最大流"问题和"最小割"问题是同一枚硬币的两面**。这个对偶性（duality）在算法设计和理论分析中极为有用。

**证明思路**：

$(1 \Rightarrow 2)$：反证法。若存在增广路径 $p$，则沿 $p$ 可以增加流量，矛盾于 $f$ 最大。

$(2 \Rightarrow 3)$：设 $S = \{v \mid G_f \text{ 中 } s \text{ 可达 } v\}$，$T = V \setminus S$。由于 $G_f$ 中 $s$ 无法到达 $t$，故 $t \in T$，$(S,T)$ 是合法割。对每条 $(u,v)$（$u \in S, v \in T$），有 $f(u,v) = c(u,v)$（否则残差边 $(u,v)$ 存在，$v$ 本应在 $S$ 中）。对每条 $(v,u)$（$v \in T, u \in S$），有 $f(v,u) = 0$（否则残差边 $(u,v)$ 存在）。因此：

$$|f| = c(S, T)$$

$(3 \Rightarrow 1)$：由引理 $|f| \leq c(S, T)$，若 $|f| = c(S, T)$，则 $f$ 达到了上界，故为最大流。$\blacksquare$

<div data-component="MaxFlowMinCutHighlight"></div>

---

## 24.2 Ford-Fulkerson 方法

### 24.2.1 残差网络（Residual Graph）

**残差网络（Residual Network）** $G_f = (V, E_f)$ 是一个"虚拟图"，表示在当前流 $f$ 之上，还有多少"调整空间"：

- **正向残差边**：若 $(u, v) \in E$ 且 $f(u, v) < c(u, v)$，则 $G_f$ 中有正向边 $(u, v)$，**剩余容量 $c_f(u, v) = c(u, v) - f(u, v)$**（还能继续往这个方向多流多少）。

- **反向残差边**：若 $(u, v) \in E$ 且 $f(u, v) > 0$，则 $G_f$ 中有反向边 $(v, u)$，**剩余容量 $c_f(v, u) = f(u, v)$**（可以"撤销"已经从 $u$ 流向 $v$ 的流量，等价于反向流动）。

**反向边的直觉**：反向边是整个算法的精妙之处。它允许算法"悔棋"——如果之前走错了路，可以把流量"退还"回去，重新走更好的路径。

> **例**：若边 $A \to B$ 容量为 10，当前流量为 6，则：
> - 正向残差 $A \to B$：容量 $10 - 6 = 4$（还能再流 4 单位）
> - 反向残差 $B \to A$：容量 $6$（可以撤销 6 单位的已有流量）

<div data-component="ResidualNetworkBuilder"></div>

### 24.2.2 增广路径（Augmenting Path）

**增广路径（Augmenting Path）** 是残差网络 $G_f$ 中从 $s$ 到 $t$ 的任意简单路径。

沿增广路径可以增加的流量称为路径的**瓶颈（Bottleneck）**：

$$\Delta = \min_{(u,v) \in p} c_f(u, v)$$

即路径上所有残差边中，最小的那个容量。

**增广操作**：沿路径 $p$ 增加 $\Delta$ 单位的流：
- 对路径上每条正向边 $(u,v)$：$f(u,v) \leftarrow f(u,v) + \Delta$
- 对路径上每条反向边 $(v,u)$（对应原有边 $(u,v)$）：$f(u,v) \leftarrow f(u,v) - \Delta$

更新后，总流量 $|f|$ 增加 $\Delta$。

### 24.2.3 Ford-Fulkerson 算法框架

```
FORD-FULKERSON(G, s, t):
  for each edge (u, v) in E:
    f[u][v] = 0
    f[v][u] = 0
  
  while 残差网络 G_f 中存在 s→t 的增广路径 p:
    Δ = min c_f(u,v) for (u,v) in p    ▷ 瓶颈容量
    for each edge (u, v) in p:
      if (u,v) is a forward edge:
        f[u][v] += Δ
      else:                              ▷ (u,v) is reverse edge, 原边为 (v,u)
        f[v][u] -= Δ
  
  return f
```

**关键问题**：如何找增广路径？不同的搜索策略导致不同的算法：

| 搜索策略 | 算法名称 | 时间复杂度 |
|:---|:---|:---|
| 任意路径（DFS 或其他） | Ford-Fulkerson | $O(E \cdot \|f^*\|)$，可能不终止 |
| 最短路径（BFS） | Edmonds-Karp | $O(VE^2)$，多项式 |
| 层次图阻塞流 | Dinic | $O(V^2 E)$，工程最优 |

### 24.2.4 正确性与复杂度分析

**正确性**：由最大流最小割定理，算法终止时（无增广路径）流即为最大流。

**终止性（整数容量的情况）**：若所有容量均为正整数，则每次增广至少使 $|f|$ 增加 1（$\Delta \geq 1$）。而 $|f|$ 有上界 $\sum_{v} c(s, v)$，故算法必然在有限步后终止。

**复杂度**：
- 整数容量时，最多进行 $|f^*|$ 次增广，每次找路径 $O(E)$，总时间 $O(E \cdot |f^*|)$。
- 若 $|f^*|$ 很大（例如 $10^9$），算法会很慢。
- 若容量为无理数，算法可能永不收敛！（Zwick 1995 年给出反例）

**这就是为什么我们需要 Edmonds-Karp 和 Dinic 算法——给出与流量 $|f^*|$ 无关的多项式时间上界。**

### 24.2.5 为什么反向边如此重要——一个反例

考虑以下图（没有使用 BFS，而是每次找到任意增广路径）：

```
s → A, 容量 1000
s → B, 容量 1000  
A → B, 容量 1
A → t, 容量 1000
B → t, 容量 1000
```

最大流显然是 2000（走 s→A→t 和 s→B→t）。但如果算法糟糕地选择了路径：

1. 找到 $s \to A \to B \to t$，$\Delta = 1$，流量变 1
2. 找到 $s \to B \to A \to t$（利用 $B \to A$ 的反向边），$\Delta = 1$，流量变 2  
3. 如此循环 1000 次才能达到最大流 2000！

无反向边时，步骤 2 无法执行，算法可能一直走 $s \to A \to B \to t$ 和 $s \to B \to A \to t$（无终止或极慢）。**反向边让算法能够"撤销错误决策"**。

---

## 24.3 Edmonds-Karp 算法

### 24.3.1 用 BFS 寻找最短增广路

**Edmonds-Karp 算法**是 Ford-Fulkerson 方法的一种具体实现：**每次用 BFS（广度优先搜索）在残差网络中寻找最短（边数最少）的增广路径**。

这个看似简单的改动，彻底消除了算法对流量大小的依赖，使时间复杂度从可能的指数级降到了多项式级 $O(VE^2)$。

**直觉**：为什么 BFS 最短路更好？BFS 找到的路径边数最少，这意味着"瓶颈边"被饱和的速度最均匀——每次增广都让某条边完全饱和，整个算法在结构上更有规律性，从而可以证明增广次数的上界。

### 24.3.2 关键引理：增广路径长度单调不降

**引理（Edmonds-Karp 单调性）**：设 $\delta_f(s, v)$ 为残差网络 $G_f$ 中 $s$ 到 $v$ 的最短路径边数（hop count）。随着增广操作的进行，对所有 $v \in V$，$\delta_f(s, v)$ 单调不降（不会减小）。

**证明思路**：每次增广只修改与当前增广路径相关的残差边。反向边的加入可能提供新路径，但经过仔细分析，这些新路径的长度不会比已有路径更短。（CLRS 引理 24.7，完整证明见教材 P.726）

这个引理是分析复杂度的关键工具。

### 24.3.3 增广次数的上界推导

**定理**：Edmonds-Karp 算法至多进行 $O(VE)$ 次增广。

**证明**（思路）：每次增广，至少有一条边"饱和"（即正向边流量达到容量，或反向边流量降为 0，对应的残差边消失）。

称被饱和的边为**关键边（Critical Edge）**。每条边 $(u, v)$ 成为关键边的次数有上界：

- $(u, v)$ 成为关键边时，$\delta_f(s, v) = \delta_f(s, u) + 1$（它在最短路上）
- 之后若 $(u, v)$ 要再次出现在增广路中（即 $(v, u)$ 先被饱和，再变回 $(u, v)$），需要 $\delta_f(s, u) \geq \delta_f(s, v) + 1$
- 由单调性，$\delta_f(s, u)$ 只增不减，故 $\delta_f(s, u)$ 至少增加 2
- 路径长度上界为 $V-1$，故每条边至多成为关键边 $O(V)$ 次

共 $O(E)$ 条边，故总增广次数 $O(VE)$。每次增广需要 BFS $O(V+E)$，总时间复杂度：

$$\boxed{O(VE^2)}$$

### 24.3.4 Python 实现

下面是 Edmonds-Karp 的完整 Python 实现，使用邻接矩阵存储残差网络：

```python
from collections import deque

def edmonds_karp(graph: list[list[int]], source: int, sink: int) -> int:
    """
    Edmonds-Karp 最大流算法（BFS 增广路版 Ford-Fulkerson）
    
    Parameters:
        graph: 邻接矩阵，graph[u][v] 表示边 u→v 的容量
               【注意】算法会直接修改 graph 为残差网络
        source: 源点编号
        sink:   汇点编号
    
    Returns:
        最大流值
    
    Time:  O(V * E^2)
    Space: O(V) 用于 BFS 路径追踪
    """
    n = len(graph)
    max_flow = 0
    
    while True:
        # ── BFS 寻找最短增广路 ──────────────────────────────
        parent = [-1] * n          # parent[v] = v 在增广路中的前驱节点
        parent[source] = source    # 标记 source 已访问（特殊值）
        queue = deque([source])
        
        while queue and parent[sink] == -1:
            u = queue.popleft()
            for v in range(n):
                # 未访问 且 残差容量 > 0
                if parent[v] == -1 and graph[u][v] > 0:
                    parent[v] = u
                    queue.append(v)
        
        # ── 若汇点不可达，则已找到最大流 ──────────────────
        if parent[sink] == -1:
            break
        
        # ── 沿增广路回溯，找瓶颈容量 Δ ────────────────────
        # 边界条件：从 sink 回溯到 source
        delta = float('inf')
        v = sink
        while v != source:
            u = parent[v]
            delta = min(delta, graph[u][v])
            v = u
        
        # ── 更新残差网络 ─────────────────────────────────────
        # 设计考量：graph[u][v] 同时编码正向容量和反向容量
        # 正向边容量减少 Δ，反向边容量增加 Δ（"撤销"能力）
        v = sink
        while v != source:
            u = parent[v]
            graph[u][v] -= delta   # 正向：剩余容量减少
            graph[v][u] += delta   # 反向：撤销额度增加
            v = u
        
        max_flow += delta
    
    return max_flow


# ── 使用示例 ─────────────────────────────────────────────────
if __name__ == "__main__":
    # 节点编号: 0=s, 1=A, 2=B, 3=C, 4=D, 5=t
    # 容量矩阵（0 表示无边）
    INF = 0  # 无边用 0 表示（而非无穷）
    cap = [
        #  s   A   B   C   D   t
        [  0, 10,  0, 10,  0,  0],  # s
        [  0,  0,  4,  2,  8,  0],  # A
        [  0,  0,  0,  0,  0, 10],  # B
        [  0,  0,  0,  0,  9,  0],  # C
        [  0,  0,  0,  0,  0, 10],  # D
        [  0,  0,  0,  0,  0,  0],  # t
    ]
    
    result = edmonds_karp(cap, source=0, sink=5)
    print(f"最大流 = {result}")   # 输出: 最大流 = 14
```

**预期输出**：
```
最大流 = 14
```

### 24.3.5 C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

class EdmondsKarp {
public:
    int n;
    vector<vector<int>> cap;  // 残差容量矩阵（正向+反向合并）
    
    // 构造函数：n 个顶点
    EdmondsKarp(int n) : n(n), cap(n, vector<int>(n, 0)) {}
    
    // 添加有向边 u→v，容量 c
    // 注意：同时为反向边预留空间（初始容量 0）
    void add_edge(int u, int v, int c) {
        cap[u][v] += c;
        // cap[v][u] 已由构造函数初始化为 0（反向边）
    }
    
    // BFS 寻找增广路，返回是否找到
    // parent[v] = v 在增广路中的前驱，用于回溯路径
    bool bfs(int s, int t, vector<int>& parent) {
        fill(parent.begin(), parent.end(), -1);
        parent[s] = s;
        queue<int> q;
        q.push(s);
        
        while (!q.empty() && parent[t] == -1) {
            int u = q.front(); q.pop();
            for (int v = 0; v < n; v++) {
                // 未访问 且 有剩余容量
                if (parent[v] == -1 && cap[u][v] > 0) {
                    parent[v] = u;
                    q.push(v);
                }
            }
        }
        return parent[t] != -1;  // t 是否可达
    }
    
    // 主函数：求 s→t 最大流
    int max_flow(int s, int t) {
        int flow = 0;
        vector<int> parent(n);
        
        while (bfs(s, t, parent)) {
            // 找瓶颈容量
            int delta = INT_MAX;
            for (int v = t; v != s; v = parent[v]) {
                int u = parent[v];
                delta = min(delta, cap[u][v]);
            }
            
            // 更新残差网络
            for (int v = t; v != s; v = parent[v]) {
                int u = parent[v];
                cap[u][v] -= delta;  // 正向减
                cap[v][u] += delta;  // 反向增（撤销额度）
            }
            
            flow += delta;
        }
        return flow;
    }
};

int main() {
    // 节点: 0=s, 1=A, 2=B, 3=C, 4=D, 5=t
    EdmondsKarp ek(6);
    ek.add_edge(0, 1, 10);  // s→A
    ek.add_edge(0, 3, 10);  // s→C
    ek.add_edge(1, 2, 4);   // A→B
    ek.add_edge(1, 3, 2);   // A→C
    ek.add_edge(1, 4, 8);   // A→D
    ek.add_edge(3, 4, 9);   // C→D
    ek.add_edge(2, 5, 10);  // B→t
    ek.add_edge(4, 5, 10);  // D→t
    
    cout << "最大流 = " << ek.max_flow(0, 5) << endl;  // 14
    return 0;
}
```

<div data-component="FordFulkersonAugPath"></div>

---

## 24.4 Dinic 算法

### 24.4.1 为什么需要 Dinic？

Edmonds-Karp 的 $O(VE^2)$ 在稠密图中并不理想。对于 $V = 1000$，$E = 10^6$ 的图，复杂度约为 $10^{18}$，完全不可接受。

**Dinic 算法**（1970 年）通过**层次图（Layered Graph）+ 阻塞流（Blocking Flow）**将复杂度降到 $O(V^2 E)$，在实践中速度极快（通常比理论上界快得多）。

### 24.4.2 层次图（Layered Graph）的构建

**层次图 $L_f$**：从残差网络 $G_f$ 中，用 BFS 从 $s$ 出发，计算每个节点的**BFS 层级（level）**$\ell(v) = $ 残差网络中 $s$ 到 $v$ 的最短距离（边数）。

层次图只保留满足以下条件的边（仅"向前一层"推进的边）：

$$\forall (u, v) \in G_f,\quad \ell(v) = \ell(u) + 1$$

即只保留"向前一层"推进的边，去掉所有横向边和后向边。

**直觉**：层次图去除了"绕远路"的可能，所有路径都是从 $s$ 到 $t$ 的最短路（在残差图中边数最少的路径）。

```
BFS 层级示例（以上面的图为例）：
  ℓ(s) = 0
  ℓ(A) = 1,  ℓ(C) = 1
  ℓ(B) = 2,  ℓ(D) = 2
  ℓ(t) = 3

层次图仅保留：
  s→A（0→1）, s→C（0→1）
  A→B（1→2）, A→D（1→2）, C→D（1→2）
  B→t（2→3）, D→t（2→3）
（去掉 A→C，因为 ℓ(C)=1=ℓ(A)=1，不满足层次条件）
```

### 24.4.3 阻塞流（Blocking Flow）

**阻塞流（Blocking Flow）** 是层次图 $L_f$ 中的一个流 $f'$，满足：$L_f$ 中每条从 $s$ 到 $t$ 的路径上，**至少有一条边被饱和**（即流量 = 容量）。

直觉：阻塞流"堵死"了层次图中所有的 $s \to t$ 路径。之后需要重新 BFS 构建新的层次图。

**如何找阻塞流？DFS + 当前弧优化**

朴素 DFS 每次找一条增广路是 $O(V)$ 的，若有 $O(E)$ 条路，则找阻塞流需要 $O(VE)$，总时间仍然可接受。

**当前弧（Current Arc）优化**：记录每个节点当前"正在考察"的边（用数组 `iter[u]` 记录），避免重复扫描已饱和的边。每条边最多被扫描常数次，使得找一个阻塞流的时间从 $O(VE)$ 降到 $O(VE)$（但常数更小，实践更快）。精确来说，**找单个阻塞流用 $O(VE)$，Dinic 需要 $O(V)$ 轮，总 $O(V^2 E)$**。

### 24.4.4 Dinic 算法框架

```
DINIC(G, s, t):
  初始化流 f = 0
  
  while BFS 构建层次图 L_f 时 t 可达:
    重置当前弧指针 iter[v] = 0, ∀v
    
    while 存在阻塞流增量 Δ > 0:  ▷ 反复找增广路直至阻塞
      Δ = DFS_BLOCKING_FLOW(L_f, s, t, ∞)
      f += Δ
  
  return f

DFS_BLOCKING_FLOW(L_f, u, t, pushed):
  if u == t: return pushed
  for iter[u] 到 u 的最后一条边:
    v = 邻居
    if ℓ(v) == ℓ(u) + 1 且 c_f(u,v) > 0:
      d = DFS_BLOCKING_FLOW(L_f, v, t, min(pushed, c_f(u,v)))
      if d > 0:
        更新 f(u,v) 和残差
        return d
    iter[u]++  ▷ 当前弧前进（死路或饱和边跳过）
  return 0    ▷ 死路，回溯
```

### 24.4.5 时间复杂度分析

**引理 1**：每次 BFS 后，$s$ 到 $t$ 的最短路距离至少增加 1。
- 证明：阻塞流饱和了层次图中所有最短路（长度为 $\ell(t)$）。重新 BFS 后，新的最短路必须绕过这些饱和边，长度至少为 $\ell(t) + 1$。

**引理 2**：最短路距离上界为 $V - 1$（路径最多 $V - 1$ 条边）。

**推论**：Dinic 最多进行 **$O(V)$** 轮 BFS（每轮最短路长度增加至少 1，上限 $V-1$）。

**每轮 BFS 的代价**：
- 构建层次图：$O(V + E)$（BFS）
- 找阻塞流：$O(VE)$（DFS + 当前弧优化，每条边最多被处理常数次）

**总时间复杂度**：

$$\boxed{O(V^2 E)}$$

**特殊情况**：对单位容量图（所有容量为 1），每次阻塞流后最短路增加，且总流量有界，可以证明时间为 $O(E \sqrt{V})$（对二部图匹配极其重要！）。

### 24.4.6 Python 实现（邻接表 + 当前弧优化）

```python
from collections import deque

class Dinic:
    """
    Dinic 最大流算法，邻接表实现（链式前向星风格）
    
    Time:  O(V^2 * E)，单位容量图 O(E * sqrt(V))
    Space: O(V + E)
    """
    
    def __init__(self, n: int):
        self.n = n
        # 邻接表：graph[u] = [(v, cap, rev_idx)]
        # rev_idx: 边 u→v 在 graph[v] 中对应反向边的索引
        self.graph: list[list[list]] = [[] for _ in range(n)]
    
    def add_edge(self, u: int, v: int, cap: int) -> None:
        """添加有向边 u→v，容量 cap，同时添加容量为 0 的反向边"""
        # u→v 正向边，graph[u][-1] 将指向 graph[v] 中的反向边索引
        self.graph[u].append([v, cap, len(self.graph[v])])
        # v→u 反向边，容量 0；rev 指回 u 的正向边索引
        self.graph[v].append([u, 0,   len(self.graph[u]) - 1])
    
    def _bfs(self, s: int, t: int) -> bool:
        """BFS 构建层次图，返回 t 是否可达"""
        self.level = [-1] * self.n  # level[v] = BFS 深度（层级）
        self.level[s] = 0
        q = deque([s])
        
        while q:
            u = q.popleft()
            for v, cap, _ in self.graph[u]:
                # cap > 0：有剩余容量；level[v] == -1：未访问
                if cap > 0 and self.level[v] == -1:
                    self.level[v] = self.level[u] + 1
                    q.append(v)
        
        return self.level[t] != -1  # t 是否在层次图中
    
    def _dfs(self, u: int, t: int, pushed: int) -> int:
        """
        DFS 在层次图中寻找增广路，返回增广量
        pushed: 当前路径上的瓶颈容量（从上层传递而来）
        
        当前弧优化：iter[u] 记录 u 已处理到第几条边，
        避免重复扫描已饱和的死路
        """
        if u == t:
            return pushed
        
        # 从当前弧 iter[u] 开始扫描（不从头重复扫已死路）
        while self.iter[u] < len(self.graph[u]):
            v, cap, rev = self.graph[u][self.iter[u]]
            
            # 满足层次条件 且 有剩余容量
            if cap > 0 and self.level[v] == self.level[u] + 1:
                d = self._dfs(v, t, min(pushed, cap))
                if d > 0:
                    # 找到增广路：更新正向边和反向边
                    self.graph[u][self.iter[u]][1] -= d  # 正向容量减
                    self.graph[v][rev][1] += d            # 反向容量增
                    return d
            
            self.iter[u] += 1  # 当前弧前进（当前边已死或饱和）
        
        return 0  # 死路，回溯
    
    def max_flow(self, s: int, t: int) -> int:
        """求 s→t 的最大流"""
        flow = 0
        
        while self._bfs(s, t):              # 每轮：构建层次图
            self.iter = [0] * self.n        # 重置当前弧指针
            
            while True:                     # 在层次图中找阻塞流
                delta = self._dfs(s, t, float('inf'))
                if delta == 0:
                    break
                flow += delta
        
        return flow


# ── 使用示例 ─────────────────────────────────────────────────
if __name__ == "__main__":
    # 节点: 0=s, 1=A, 2=B, 3=C, 4=D, 5=t
    g = Dinic(6)
    g.add_edge(0, 1, 10)  # s→A: 10
    g.add_edge(0, 3, 10)  # s→C: 10
    g.add_edge(1, 2, 4)   # A→B: 4
    g.add_edge(1, 3, 2)   # A→C: 2
    g.add_edge(1, 4, 8)   # A→D: 8
    g.add_edge(3, 4, 9)   # C→D: 9
    g.add_edge(2, 5, 10)  # B→t: 10
    g.add_edge(4, 5, 10)  # D→t: 10
    
    print(f"最大流 = {g.max_flow(0, 5)}")  # 输出: 14
```

**预期输出**：
```
最大流 = 14
```

### 24.4.7 C++ 实现（标准竞赛模板）

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Edge {
    int to, rev;  // 终点、反向边在 graph[to] 中的索引
    int cap;      // 当前剩余容量
};

class Dinic {
public:
    int n;
    vector<vector<Edge>> graph;  // 邻接表（包含反向边）
    vector<int> level, iter;    // 层级、当前弧
    
    Dinic(int n) : n(n), graph(n), level(n), iter(n) {}
    
    // 添加有向边 u→v 容量 c，以及反向边（容量 0）
    void add_edge(int u, int v, int c) {
        graph[u].push_back({v, (int)graph[v].size(), c});
        graph[v].push_back({u, (int)graph[u].size() - 1, 0});
    }
    
    // BFS 构建层次图，返回 t 是否可达
    bool bfs(int s, int t) {
        fill(level.begin(), level.end(), -1);
        queue<int> q;
        level[s] = 0;
        q.push(s);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (auto& e : graph[u]) {
                if (e.cap > 0 && level[e.to] < 0) {
                    level[e.to] = level[u] + 1;
                    q.push(e.to);
                }
            }
        }
        return level[t] >= 0;
    }
    
    // DFS 找增广路（当前弧优化）
    int dfs(int u, int t, int f) {
        if (u == t) return f;
        for (int& i = iter[u]; i < (int)graph[u].size(); i++) {
            Edge& e = graph[u][i];
            int v = e.to;  // 明确声明变量 v
            if (e.cap > 0 && level[v] == level[u] + 1) {
                int d = dfs(v, t, min(f, e.cap));
                if (d > 0) {
                    e.cap -= d;                   // 正向容量减
                    graph[v][e.rev].cap += d;     // 反向容量增
                    return d;
                }
            }
        }
        return 0;
    }
    
    // 求 s→t 最大流
    int max_flow(int s, int t) {
        int flow = 0;
        while (bfs(s, t)) {
            fill(iter.begin(), iter.end(), 0);
            int d;
            while ((d = dfs(s, t, INT_MAX)) > 0)
                flow += d;
        }
        return flow;
    }
};

int main() {
    Dinic dinic(6);  // 0=s,1=A,2=B,3=C,4=D,5=t
    dinic.add_edge(0, 1, 10);
    dinic.add_edge(0, 3, 10);
    dinic.add_edge(1, 2, 4);
    dinic.add_edge(1, 3, 2);
    dinic.add_edge(1, 4, 8);
    dinic.add_edge(3, 4, 9);
    dinic.add_edge(2, 5, 10);
    dinic.add_edge(4, 5, 10);
    
    cout << "最大流 = " << dinic.max_flow(0, 5) << endl;  // 14
    return 0;
}
```

<div data-component="DinicLayeredGraph"></div>

---

## 24.5 最大流应用

### 24.5.1 二部图最大匹配

**问题**：给定二部图 $G = (U \cup V, E)$（$U$ 和 $V$ 是两个不相交的顶点集，边只存在于 $U$ 与 $V$ 之间），求最大匹配——即最多能配对多少对 $(u, v)$，使得每个顶点只出现在一对中。

**典型场景**：
- 招聘：$U$ = 候选人，$V$ = 岗位，边 $(u, v)$ 表示候选人 $u$ 胜任岗位 $v$，最多能匹配多少人？
- 任务分配：$U$ = 工人，$V$ = 任务，最多能完成多少任务？
- 稳定婚姻的存在性：最大匹配是否能覆盖所有人？

**建图方式（将二部图匹配归约为最大流）**：

1. 新增超级源点 $s$ 和超级汇点 $t$
2. $s$ 向 $U$ 中每个顶点连一条容量为 **1** 的边：$s \to u, c = 1$
3. $V$ 中每个顶点向 $t$ 连一条容量为 **1** 的边：$v \to t, c = 1$
4. $U$ 到 $V$ 的原有边：$u \to v, c = 1$（容量均为 1）
5. 最大流 = 最大匹配数

**为什么正确**？
- 每个 $u \in U$ 只能匹配一个 $v$（因为 $s \to u$ 容量为 1，限制了最多 1 单位流量）
- 每个 $v \in V$ 只能被一个 $u$ 匹配（因为 $v \to t$ 容量为 1）
- 每单位流量对应一条匹配边

**时间复杂度**：使用 Dinic，对单位容量图为 $O(E\sqrt{V})$（其中 $V = |U| + |V|$）。

```python
def bipartite_matching(left: int, right: int, edges: list[tuple[int,int]]) -> int:
    """
    二部图最大匹配 via 最大流
    
    Parameters:
        left:  左侧顶点数（编号 0 ~ left-1）
        right: 右侧顶点数（编号 0 ~ right-1）
        edges: [(u, v)] 表示左侧 u 与右侧 v 有边
    
    Returns:
        最大匹配数
    """
    # 节点编号：0=s, 1~left=左侧, left+1~left+right=右侧, left+right+1=t
    total = left + right + 2
    s = 0
    t = left + right + 1
    
    g = Dinic(total)
    
    # s → 每个左侧节点，容量 1
    for u in range(left):
        g.add_edge(s, u + 1, 1)
    
    # 每个右侧节点 → t，容量 1
    for v in range(right):
        g.add_edge(left + 1 + v, t, 1)
    
    # 原有匹配边，容量 1
    for u, v in edges:
        g.add_edge(u + 1, left + 1 + v, 1)
    
    return g.max_flow(s, t)

# 示例：4 名候选人，3 个岗位
# 候选人 0 可胜任岗位 0, 1
# 候选人 1 可胜任岗位 0, 2
# 候选人 2 可胜任岗位 1
# 候选人 3 可胜任岗位 2
edges = [(0,0),(0,1),(1,0),(1,2),(2,1),(3,2)]
print(bipartite_matching(4, 3, edges))  # 输出: 3 (完美匹配岗位)
```

**预期输出**：
```
3
```

<div data-component="BipartiteMatchingFlow"></div>

### 24.5.2 König 定理

**König 定理**（二部图）：

$$\text{最大匹配大小} = \text{最小顶点覆盖大小}$$

其中，**顶点覆盖（Vertex Cover）** 是一个顶点子集 $C$，使得每条边至少有一个端点在 $C$ 中（即 $C$ "覆盖"了所有边）。**最小顶点覆盖**就是最小的这样一个集合。

该定理在一般图中不成立（一般图的最小顶点覆盖是 NP-hard 问题！），但在二部图中可由最大流最小割定理直接推导。

**证明方向**：
- 最大匹配 ≤ 最小顶点覆盖：匹配的每条边需要至少一个端点在覆盖中，且匹配边之间不共享端点，故覆盖大小 ≥ 匹配大小
- 最大流最小割 → 最大匹配 = 最小割容量 = 最小顶点覆盖

**工程应用**：编译器中的寄存器分配、稀疏矩阵乘法的优化等场景，都用到了二部图覆盖的概念。

### 24.5.3 项目选择问题（Project Selection）

**问题**：有 $n$ 个项目和 $m$ 台设备。每个项目有收益/成本（正数为收益，负数为成本）。部分项目需要某些设备（设备有购置成本），且项目之间可能有"项目 A 依赖项目 B 完成"的依赖关系。问：选哪些项目能最大化总收益减总成本？

**建模为最小割**：

1. 源点 $s$ 代表"选择"，汇点 $t$ 代表"不选择"
2. 若项目/设备的净收益为正：$s \to v$，容量 = 收益（选它则不切割此边，获收益；放弃则割此边，代价 = 收益损失）
3. 若项目/设备净收益为负：$v \to t$，容量 = |成本|
4. 若项目 $u$ 依赖项目 $v$（做 $u$ 必须做 $v$）：$u \to v$，容量 = $\infty$

**总收益 = 所有正收益之和 - 最小割容量**

最小割对应"哪些选择我们放弃"，其容量即为放弃收益加上必须承担的成本。

### 24.5.4 图像分割（最小 s-t 割应用）

**问题**：将图片中每个像素分类为"前景"（对象）或"背景"。给定：
- 每个像素 $p$ 被标记为前景的代价 $\alpha_p$、标记为背景的代价 $\beta_p$
- 相邻像素 $(p, q)$ 标记不同类别的"不一致代价" $w_{pq}$

**最小化总代价**可建模为最小割：
- $s \to p$：容量 $\beta_p$（割此边 = $p$ 归入背景，代价 $\beta_p$）
- $p \to t$：容量 $\alpha_p$（割此边 = $p$ 归入前景，代价 $\alpha_p$）
- 相邻像素 $p \leftrightarrow q$：双向边，容量 $w_{pq}$（割此边 = 两者标签不同，代价 $w_{pq}$）

最小割即为最小代价分割方案。这是计算机视觉中经典的图割方法（Graph Cut），在医学图像分析、物体识别等领域广泛应用。

---

## 24.6 算法对比与选择指南

### 24.6.1 三种算法横向对比

| 指标 | Ford-Fulkerson | Edmonds-Karp | Dinic |
|:---|:---:|:---:|:---:|
| 增广策略 | 任意 DFS 路径 | BFS 最短路 | 层次图 + 阻塞流 |
| 时间复杂度 | $O(E \cdot \|f^*\|)$ | $O(VE^2)$ | $O(V^2 E)$ |
| 单位容量图 | $O(E \cdot \|f^*\|)$ | $O(VE^2)$ | $O(E\sqrt{V})$ |
| 稠密图表现 | 可能极差 | 一般 | 良好 |
| 稀疏图表现 | 取决于流量 | 尚可 | 优秀 |
| 实现复杂度 | 低 | 中 | 中偏高 |
| 工程推荐度 | ❌ | ⚠️ | ✅ |

**选择建议**：

- **竞赛/工程中首选 Dinic**：实际运行时间通常远小于理论上界，对大多数图都能高效处理。
- **Edmonds-Karp**：理论课代码示例或者小规模问题，实现简单。
- **Ford-Fulkerson**：仅用于整数容量且流量不大的教学演示；有理数/无理数容量场景下禁用。

### 24.6.2 当前弧优化的重要性

在 Dinic 算法中，**当前弧优化**是使算法高效的关键：

- **没有当前弧优化**：每次 DFS 从头扫描邻接表，找阻塞流需 $O(V \cdot E)$，总共 $O(V^2 E)$（但是常数更大）
- **有当前弧优化**：`iter[u]` 跳过已失效的边，每条边最多被"扫过放弃" 1 次，找阻塞流 $O(V + E)$（去掉死路后近似），实际上快很多

**实现细节**：必须在每轮 BFS（构建新层次图）后重置 `iter` 数组为 0，但在同一轮层次图内多次 DFS 时不重置（这正是优化的来源）。

---

## 24.7 常见错误与调试技巧

### 24.7.1 建图错误

**错误 1：忘记添加反向边（容量为 0 的反向边）**

```python
# ❌ 错误：只加了正向边，无法"撤销"
graph[u][v] = capacity

# ✅ 正确：正向 + 反向（初始容量 0）
graph[u][v] += capacity
graph[v][u] += 0  # 或初始化时已赋值 0
```

**错误 2：双向边处理不当**

若同时有 $u \to v$ 和 $v \to u$ 两条有向边：

```python
# ❌ 错误：graph[u][v] 和 graph[v][u] 会互相干扰
graph[u][v] = cap1
graph[v][u] = cap2

# ✅ 正确（邻接链表法）：显式区分正向边和对应的反向边
# 每条有向边都有独立索引，反向边索引通过 rev 字段关联
add_edge(u, v, cap1)  # 自动添加 v→u 容量为 0 的反向边
add_edge(v, u, cap2)  # 自动添加 u→v 容量为 0 的反向边（不同条目！）
```

### 24.7.2 Dinic 调试技巧

**Bug：DFS 在层次图外扩展**

`_dfs` 中必须严格检查 `level[v] == level[u] + 1`，否则会走层次条件外的边，导致结果偏大或死循环。

**Bug：当前弧重置时机错误**

`iter` 数组应在每轮 BFS 后重置，NOT 在每次 DFS 后重置：

```python
while bfs(s, t):       # 每轮：BFS 重建层次图
    self.iter = [0]*n  # ← 正确：每轮 BFS 后重置一次
    while True:
        d = dfs(s, t, INF)  # 多次 DFS 共享同一 iter
        if d == 0: break
        flow += d
```

**Bug：流量溢出**

使用 `int` 而非 `long long` 时，容量之和可能超出 32 位范围，应使用 64 位整数：

```cpp
// C++：使用 long long
long long dfs(int u, int t, long long f) { ... }
long long max_flow(int s, int t) { ... }
```

### 24.7.3 如何验证最大流正确性

1. **流守恒验证**：对每个中间节点，检查入流 = 出流
2. **容量约束验证**：检查每条边 $f(u,v) \leq c(u,v)$ 且 $f(u,v) \geq 0$
3. **找对应最小割**：BFS 残差网络，从 $s$ 可达的顶点集合 $S$ 和其补集 $T$ 构成最小割，验证 $c(S, T) = |f|$

---

## 本章小结

| 知识点 | 核心内容 |
|:---|:---|
| 流网络 | 容量约束 + 流守恒，源点汇点 |
| 最大流最小割定理 | 三等价命题，理论核心 |
| 残差网络 | 正向剩余 + 反向撤销，允许"悔棋" |
| Ford-Fulkerson | 框架方法，整数容量终止，依赖流量 |
| Edmonds-Karp | BFS 最短路增广，$O(VE^2)$，多项式 |
| Dinic | 层次图 + 阻塞流 + 当前弧，$O(V^2E)$，实践最优 |
| 二部图匹配 | 归约为最大流，Dinic 单位容量 $O(E\sqrt{V})$ |
| König 定理 | 最大匹配 = 最小顶点覆盖（仅二部图） |
| 项目选择/图像分割 | 最小割的实际应用 |

**思考题**：

1. 为什么反向边至关重要？请构造一个没有反向边时 Ford-Fulkerson 无法找到最大流的例子。
2. Dinic 算法每轮 BFS 后层次图的最短路长度为什么严格递增？若某轮没有增广到任何流量，算法还会终止吗？
3. 二部图最大匹配的 $O(E\sqrt{V})$ 复杂度来自哪里？尝试证明 Dinic 对单位容量图最多进行 $O(\sqrt{V})$ 轮 BFS。
4. König 定理在一般图中为何不成立？举出一个反例（提示：奇数环）。

**参考资料**：
- CLRS 第4版 Chapter 24（网络流）
- MIT 6.046 Lecture 12：Network Flow
- Sedgewick 《算法（第4版）》第 4.4 节
- [CP-Algorithms: Maximum Flow](https://cp-algorithms.com/graph/max_flow.html)
- [Codeforces Blog: Dinic's Algorithm](https://codeforces.com/blog/entry/104960)

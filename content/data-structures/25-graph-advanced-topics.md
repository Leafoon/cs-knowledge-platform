# Chapter 25: 关键路径、二部图与其他图问题

## 章节导读

经过前几章对图算法的系统学习，我们已经掌握了最短路径（Dijkstra、Bellman-Ford、Floyd-Warshall）和网络流（Ford-Fulkerson、Dinic）这两座高峰。本章是 Part VII "图高级算法"的收官之作，将探讨三类在网络分析、任务调度与系统设计中极其实用的图问题：

- **欧拉路径与欧拉回路**：能否一笔画遍所有边？这是数学史上最早的图论问题之一（1736年，欧拉解决了柯尼斯堡七桥问题）。
- **二部图匹配**：任务分配、课程排课、网络路由……这类"左右两侧的最优配对"问题无处不在。匈牙利算法（Hungarian Algorithm）和 Hopcroft-Karp 算法提供了从 $O(VE)$ 到 $O(E\sqrt{V})$ 的提升。
- **桥与割点（Tarjan 算法）**：网络中哪些连接是"命脉"？删除哪个节点会导致整个网络瘫痪？Tarjan 的 low-link 值在 $O(V+E)$ 时间内给出答案。
- **平面图与图着色**：地图着色、频率分配、编译器寄存器分配……图着色的 NP-完全性揭示了算法世界中"难问题"的本质。

> **前置知识**：Chapter 18（图的基本表示）、Chapter 19（BFS/DFS 框架）、Chapter 24（网络流 · 二部图匹配的流模型视角）

---

## 25.1 欧拉路径与欧拉回路

### 25.1.1 从柯尼斯堡七桥说起

**历史背景**：1736年，瑞士数学家莱昂哈德·欧拉（Leonhard Euler）被问到一个有趣的难题：普鲁士柯尼斯堡城有一条河，将城市分为四块陆地，陆地之间架有七座桥。问：**能否从任意起点出发，不重复地走遍所有七座桥，最后回到出发点？**

欧拉用抽象的方法研究这个问题：把每块陆地看成一个**节点**，每座桥看成一条**边**，将地图转化为图。然后他发现，这道题的关键不在于桥的具体位置，而在于每个节点的**度数（Degree）**——与该节点相连的边数。

**欧拉的洞察**：
- 若从某节点出发经过一条边，离开时消耗 1 度，必须有另一条边回来——这要求该节点度数为**偶数**。
- 只有起点（多出发一次）和终点（多到达一次）可以有奇数度数。
- 若要回路（回到起点），则起点也要度数为偶数。
- 柯尼斯堡四个陆地的度数分别为 3, 3, 3, 5，全部为奇数——因此**不可能**。

这个思想就诞生了图论中的第一个重要定理。

### 25.1.2 欧拉路径/回路的严格定义

**欧拉路径（Eulerian Path / Eulerian Trail）**：遍历图中**每条边恰好一次**的路径（起点和终点可以不同）。

**欧拉回路（Eulerian Circuit / Eulerian Cycle）**：遍历图中**每条边恰好一次**且**回到起点**的路径。

**存在条件**（前提：图必须连通，孤立节点除外）：

**无向图**：

$$\text{欧拉回路存在} \iff \text{所有顶点的度数均为偶数}$$

$$\text{欧拉路径存在（不成回路）} \iff \text{恰好有 2 个顶点的度数为奇数（起点与终点）}$$

**有向图**（设 $\deg^+(v)$ 为出度，$\deg^-(v)$ 为入度）：

$$\text{欧拉回路存在} \iff \forall v:\; \deg^+(v) = \deg^-(v)$$

$$\text{欧拉路径存在} \iff \exists!\, s:\;\deg^+(s) = \deg^-(s)+1,\quad \exists!\, t:\;\deg^-(t) = \deg^+(t)+1,\quad \forall v \neq s,t:\;\deg^+(v) = \deg^-(v)$$

即：有向图的欧拉路径要求除了起点（多一个出度）和终点（多一个入度）外，所有节点出入度相等。

> **例子**：一张有向图，边为 $A \to B, B \to C, C \to A, A \to D, D \to B$。
> - $\deg^+(A)=2, \deg^-(A)=1$：$A$ 的出度多 1，是起点
> - $\deg^-(B)=2, \deg^+(B)=1$：$B$ 的入度多 1，是终点
> - 其余节点 $C, D$ 出入度相等
> - 因此存在欧拉路径，从 $A$ 到 $B$

<div data-component="EulerianPathFinder"></div>

### 25.1.3 Hierholzer 算法：$O(V+E)$ 找欧拉回路

**朴素思路的问题**：最简单的想法是 DFS——从起点出发，遇到未访问的边就走。但朴素 DFS 会走进"死胡同"：走完一小段子回路后回到起点，但此时还有未访问的边被"跳过"了，如何合并回来？

**Hierholzer 算法**的核心思想：**找到子回路，然后把子回路"嵌入"到主路径中**。

算法步骤（以有向图欧拉回路为例）：

1. 找任意一个度数满足条件的起点（有向图中找出度大于入度的节点，或任意节点）
2. 沿任意未访问边进行 DFS，直到无路可走（到达死胡同）——这条路径形成一个**子回路**
3. 从当前节点回溯，找到子路径上**还有未访问出边**的节点
4. 从该节点出发重新 DFS，找到另一条子回路
5. 将新子回路"插入"原路径中该节点的位置
6. 重复直到所有边都被访问

**实现技巧**：用栈（stack）加后序记录，天然得到逆序欧拉路径，翻转即可：

```python
# Python 实现
from collections import defaultdict

def hierholzer(n: int, edges: list[tuple[int, int]]) -> list[int]:
    """
    Hierholzer 算法求欧拉路径/回路（有向图版本）
    
    参数:
        n: 顶点数（0-indexed）
        edges: 有向边列表 [(u, v), ...]
    
    返回:
        欧拉路径（节点列表），若不存在则返回空列表
    
    时间复杂度: O(V + E)
    空间复杂度: O(V + E)
    """
    # 构建邻接表，使用指针（索引）避免删除边的 O(E) 开销
    adj = defaultdict(list)
    in_degree  = [0] * n
    out_degree = [0] * n
    
    for u, v in edges:
        adj[u].append(v)
        out_degree[u] += 1
        in_degree[v]  += 1
    
    # 检验欧拉路径/回路是否存在
    start_nodes = []  # 出度 = 入度 + 1 的节点（欧拉路径起点）
    end_nodes   = []  # 入度 = 出度 + 1 的节点（欧拉路径终点）
    
    for v in range(n):
        diff = out_degree[v] - in_degree[v]
        if diff == 1:
            start_nodes.append(v)
        elif diff == -1:
            end_nodes.append(v)
        elif diff != 0:
            return []   # 差值超过 ±1，不存在欧拉路径
    
    # 判断是回路还是路径
    if len(start_nodes) == 0 and len(end_nodes) == 0:
        # 欧拉回路：从任意有出边的节点出发
        start = next((v for v in range(n) if out_degree[v] > 0), 0)
    elif len(start_nodes) == 1 and len(end_nodes) == 1:
        # 欧拉路径：必须从 start_node 出发
        start = start_nodes[0]
    else:
        return []   # 奇度节点超过 2 个，不存在欧拉路径
    
    # Hierholzer 核心：DFS + 栈记录后序（逆序路径）
    ptr = {v: 0 for v in range(n)}  # 当前弧指针（避免重复扫描已用边）
    stack  = [start]
    result = []
    
    while stack:
        v = stack[-1]
        if ptr[v] < len(adj[v]):
            # 还有未访问的出边：继续走
            next_v = adj[v][ptr[v]]
            ptr[v] += 1
            stack.append(next_v)
        else:
            # 死胡同（或子回路封闭）：加入结果
            result.append(stack.pop())
    
    # DFS 结束时 result 是逆序的欧拉路径
    result.reverse()
    
    # 验证：欧拉路径应包含所有边（共 E+1 个节点）
    if len(result) != len(edges) + 1:
        return []  # 图不连通，无法形成完整欧拉路径
    
    return result


# ══ 示例：LeetCode 332 重新安排行程 ══
def find_itinerary(tickets: list[list[str]]) -> list[str]:
    """
    将机场名映射为整数再运行 Hierholzer
    按字典序排列出边，保证字典序最小的欧拉路径
    """
    adj: dict[str, list[str]] = defaultdict(list)
    for src, dst in sorted(tickets, reverse=True):   # 逆序排，栈弹出时字典升序
        adj[src].append(dst)
    
    stack  = ["JFK"]
    result = []
    
    while stack:
        v = stack[-1]
        if adj[v]:
            stack.append(adj[v].pop())
        else:
            result.append(stack.pop())
    
    return result[::-1]
```

```cpp
// C++ 实现
#include <bits/stdc++.h>
using namespace std;

class EulerianPath {
public:
    /**
     * Hierholzer 算法求有向图欧拉路径/回路
     *
     * @param n     顶点数（0-indexed）
     * @param edges 有向边列表 {u, v}
     * @return      欧拉路径节点序列，不存在则为空向量
     *
     * 时间复杂度: O(V + E)
     * 空间复杂度: O(V + E)
     */
    static vector<int> solve(int n, vector<pair<int,int>>& edges) {
        vector<vector<int>> adj(n);
        vector<int> indeg(n, 0), outdeg(n, 0);
        
        for (auto& [u, v] : edges) {
            adj[u].push_back(v);
            outdeg[u]++;
            indeg[v]++;
        }
        
        // ── 检验欧拉路径/回路存在条件 ──────────────────────────
        int start = 0, start_cnt = 0, end_cnt = 0;
        for (int v = 0; v < n; v++) {
            int diff = outdeg[v] - indeg[v];
            if (diff == 1)       { start = v; start_cnt++; }
            else if (diff == -1) { end_cnt++; }
            else if (diff != 0)  return {};  // 绝对不存在
        }
        
        if (start_cnt == 0 && end_cnt == 0) {
            // 欧拉回路：从有出边的任意节点出发
            start = 0;
            while (start < n && outdeg[start] == 0) start++;
        } else if (start_cnt != 1 || end_cnt != 1) {
            return {};  // 奇度节点多于 2 个
        }
        
        // ── Hierholzer 核心（当前弧优化）──────────────────────────
        // ptr[v]: adj[v] 中下一条待访问边的索引（避免重复扫描已删除边）
        vector<int> ptr(n, 0);
        stack<int>  stk;
        vector<int> result;
        
        stk.push(start);
        while (!stk.empty()) {
            int v = stk.top();
            if (ptr[v] < (int)adj[v].size()) {
                // 还有未访问出边：继续向前
                stk.push(adj[v][ptr[v]++]);
            } else {
                // 死胡同：记录到结果
                result.push_back(v);
                stk.pop();
            }
        }
        
        reverse(result.begin(), result.end());
        
        // 验证完整性
        if ((int)result.size() != (int)edges.size() + 1) return {};
        return result;
    }
};

// ══ 示例驱动 ══
int main() {
    // 例图：0→1, 1→2, 2→0, 0→3, 3→1  (欧拉路径: 3→1→2→0→3→1? No...)
    // 重新使用：0→1, 1→2, 2→3, 3→0   (欧拉回路)
    int n = 4;
    vector<pair<int,int>> edges = {{0,1},{1,2},{2,3},{3,0}};
    
    auto path = EulerianPath::solve(n, edges);
    cout << "欧拉回路: ";
    for (int v : path) cout << v << " ";
    cout << "\n";
    // 输出: 0 1 2 3 0
    
    return 0;
}
```

**算法正确性直觉**：
- 每次 DFS 沿着出边走，走到死胡同意味着形成了一个子回路（每次离开一个节点都有对应的"回来"边，除终点外）。
- 将死胡同节点压入结果（后序记录），自然地将子回路拼接到主回路中。
- 当前弧指针 `ptr[v]` 保证每条边被访问恰好一次，总时间 $O(V+E)$。

### 25.1.4 欧拉 vs 哈密顿：为什么难度天壤之别？

**哈密顿路径（Hamiltonian Path）**：遍历图中每个**顶点**恰好一次的路径。
**哈密顿问题**：判断一个图是否存在哈密顿路径/回路。

| 维度 | 欧拉路径 | 哈密顿路径 |
|---|---|---|
| 约束对象 | **边**（每条边走一次） | **顶点**（每个顶点访问一次） |
| 存在条件判断 | 度数条件 $O(V)$ 可检查 | 无简单充要条件 |
| 求解算法 | Hierholzer $O(V+E)$ | 无多项式算法（NP-hard） |
| 复杂度类别 | **P**（多项式可解） | **NP-完全**（指数级搜索） |
| 局部结构 | 度数是局部信息 | 哈密顿性是全局信息，无法局部验证 |

**为什么欧拉问题竟然比哈密顿容易这么多？**

这是图论中最令人惊叹的不对称性之一。关键在于：

- 欧拉路径的存在性只取决于**每个节点的度数**——这是纯粹的**局部信息**，与边如何连接无关，$O(V)$ 即可检查。
- 哈密顿路径的存在性取决于图的**全局拓扑结构**——没有任何简单的局部条件能够确保哈密顿性。即使图看起来"连通且密集"，哈密顿路径也可能不存在；反之，一个稀疏图也可能有哈密顿回路。

这个反差完美体现了"问题结构"对算法复杂度的决定性影响：**同样是"遍历图元素"的问题，遍历边（局部约束）易，遍历点（全局约束）难**。

### 25.1.5 应用：中国邮差问题与骨牌排列

**中国邮差问题（Chinese Postman Problem）**：邮差需要走遍所有街道（边）并返回出发点，**总路程最短**。这是欧拉回路的"推广"——如果图本来就有欧拉回路，直接走即可；若有奇度节点，需要**重复走若干边**补全度数，用最小权匹配求重复边的最优选择（$O(n^3)$）。

**骨牌排列问题**：将一组多米诺骨牌 $[a|b]$ 排成一排，使得相邻骨牌接触的数字相同（如 $[1|3][3|5][5|2]...$）。这等价于把每种数字看成节点，每张骨牌看成边，问题变成**求欧拉路径**。

**LeetCode 332. 重新安排行程**：给定的机票列表，每张票 $[\text{src, dst}]$ 代表一条有向边，要求找出字典序最小的欧拉路径（从 JFK 出发）。

---

## 25.2 二部图匹配（Bipartite Matching）

### 25.2.1 什么是二部图匹配？

**二部图（Bipartite Graph）** 是指顶点可以分为两个不相交的集合 $L$（左部）和 $R$（右部），所有边都跨越两个集合（即没有 $L$ 内部或 $R$ 内部的边）的图。

**直觉比喻**——任务分配问题：
- $L$ = 员工集合（$W_1, W_2, W_3...$）
- $R$ = 任务集合（$J_1, J_2, J_3...$）
- 边 $(W_i, J_j)$ 表示员工 $W_i$ 能胜任任务 $J_j$
- **匹配（Matching）**：选取边的子集，使得每个员工最多匹配一个任务，每个任务最多由一个员工负责
- **最大匹配（Maximum Matching）**：匹配中包含边数最多的方案

**二部图的判定**：BFS 2-染色，$O(V+E)$——从任意节点出发，将相邻节点染成不同颜色。若出现同色相邻节点，则图不是二部图（含奇数环）。

### 25.2.2 增广路径思想（匈牙利算法基础）

**增广路径（Augmenting Path）**：在当前匹配 $M$ 下，从一个**未匹配**的左部节点出发，交替走未匹配边和匹配边，最终到达一个**未匹配**的右部节点的路径。

**关键定理（Berge 定理）**：匹配 $M$ 是最大匹配，当且仅当图中不存在 $M$-增广路径。

**匈牙利算法（Hungarian Algorithm）流程**：
对每个未匹配的左节点 $u$：
1. DFS 尝试为 $u$ 寻找增广路径
2. 若找到增广路径，沿路径翻转匹配边/非匹配边（匹配大小 +1）
3. 若找不到，跳过

每次 DFS 最多走 $O(V+E)$，最多进行 $|L|$ 次，总时间复杂度 **$O(VE)$**。

```python
# Python 实现：匈牙利算法（增广路径 DFS）
def hungarian(n_left: int, n_right: int, edges: list[tuple[int, int]]) -> int:
    """
    匈牙利算法求二部图最大匹配
    
    参数:
        n_left:  左部节点数（0-indexed: 0..n_left-1）
        n_right: 右部节点数（0-indexed: 0..n_right-1）
        edges:   边列表 [(left_node, right_node), ...]
    
    返回:
        最大匹配大小
    
    时间复杂度: O(V * E)，其中 V = n_left
    空间复杂度: O(V + E)
    """
    # 构建左部到右部的邻接表
    adj = [[] for _ in range(n_left)]
    for u, v in edges:
        adj[u].append(v)
    
    # match_right[v] = 右部节点 v 当前匹配的左部节点（-1 表示未匹配）
    match_right = [-1] * n_right
    
    def try_augment(u: int, visited: list[bool]) -> bool:
        """
        从左节点 u 出发尝试寻找增广路径（DFS）
        
        visited: 记录本次 DFS 已经访问过的右部节点（避免死循环）
        返回: 是否找到增广路径
        """
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True   # 标记 v 已在本次增广路径中
                
                # 若 v 未匹配，或者 v 的当前匹配者能找到新路径（递归腾位）
                if match_right[v] == -1 or try_augment(match_right[v], visited):
                    match_right[v] = u   # 将 v 匹配给 u
                    return True
        
        return False   # 找不到增广路径
    
    matching = 0
    for u in range(n_left):
        # 每次为一个新的左节点尝试匹配，重置 visited（保证每次 DFS 独立）
        visited = [False] * n_right
        
        # 关键易错点：visited 必须每轮重置，否则会阻止合法的增广路径
        if try_augment(u, visited):
            matching += 1
    
    return matching


# ══ 示例 ══
if __name__ == "__main__":
    # 4 个工人，4 个任务
    # W0->J0, W0->J1, W1->J1, W1->J2, W2->J2, W3->J2, W3->J3
    edges = [(0,0),(0,1),(1,1),(1,2),(2,2),(3,2),(3,3)]
    print(hungarian(4, 4, edges))  # 输出: 3 或 4，取决于匹配过程
```

```cpp
// C++ 实现：匈牙利算法
#include <bits/stdc++.h>
using namespace std;

class Hungarian {
    int n_left, n_right;
    vector<vector<int>> adj;   // 左部节点的邻接表（右部索引）
    vector<int> match_right;   // match_right[v] = 与右部v匹配的左部节点，-1表示未匹配
    vector<bool> visited;      // 当前增广DFS中已访问的右部节点

public:
    Hungarian(int nl, int nr) : n_left(nl), n_right(nr),
        adj(nl), match_right(nr, -1) {}
    
    void add_edge(int u, int v) {
        adj[u].push_back(v);
    }
    
    /**
     * 从左节点 u 出发 DFS 寻找增广路径
     * 
     * 核心逻辑：
     *   1. 遍历 u 的所有右邻居 v
     *   2. 若 v 未被本次 DFS 访问过：
     *      a. 若 v 未匹配 → 直接匹配，成功
     *      b. 若 v 已匹配给 w → 递归尝试给 w 找别的匹配（腾出 v）
     */
    bool try_augment(int u) {
        for (int v : adj[u]) {
            if (!visited[v]) {
                visited[v] = true;
                if (match_right[v] == -1 || try_augment(match_right[v])) {
                    match_right[v] = u;
                    return true;
                }
            }
        }
        return false;
    }
    
    /**
     * 求最大匹配
     * 
     * 时间复杂度: O(V_L * (V + E))，通常简写 O(VE)
     */
    int max_matching() {
        int result = 0;
        for (int u = 0; u < n_left; u++) {
            // 每次为一个左节点尝试时，重置 visited（避免不同左节点的搜索互相干扰）
            visited.assign(n_right, false);
            if (try_augment(u)) result++;
        }
        return result;
    }
    
    // 获取匹配结果（右部节点 v 匹配的左部节点）
    vector<int> get_matching() const { return match_right; }
};

int main() {
    Hungarian h(4, 4);
    h.add_edge(0,0); h.add_edge(0,1);
    h.add_edge(1,1); h.add_edge(1,2);
    h.add_edge(2,2);
    h.add_edge(3,2); h.add_edge(3,3);
    
    cout << "最大匹配: " << h.max_matching() << "\n";
    return 0;
}
```

### 25.2.3 Hopcroft-Karp 算法：$O(E\sqrt{V})$

匈牙利算法的瓶颈在于：每轮只找一条增广路径（最多 $|L|$ 轮），而每条增广路径搜索都是 $O(E)$ 的 DFS。Hopcroft-Karp 的突破在于：**每轮同时找出所有最短的增广路径（并行增广）**，将轮数从 $O(V)$ 降到 $O(\sqrt{V})$。

**算法流程**：
1. **BFS 构建层次图**：从所有未匹配左节点出发，交替走非匹配边（左→右）和匹配边（右→左），到达未匹配右节点时停止。记录每个节点的 BFS 层次 $\text{dist}[v]$。
2. **DFS 批量增广**：沿层次图做 DFS，找出所有相互不共享节点的最短增广路径（最大无交叉增广路径集——MICS）。
3. 每轮增广后匹配大小增加至少 1，而最短增广路径在每轮之后至少增长 2（因为已找到最短增广路径），所以最多 $O(\sqrt{V})$ 轮后，剩余最短增广路径长度超过 $\sqrt{V} \cdot 2$，而此时最多剩余 $\sqrt{V}$ 条增广路径（未匹配节点数上界），共 $O(\sqrt{V})$ 轮。
4. 每轮 BFS+DFS 是 $O(E)$，总计 $O(E\sqrt{V})$。

```python
# Python 实现：Hopcroft-Karp 算法
from collections import deque

def hopcroft_karp(n_left: int, n_right: int, edges: list[tuple[int, int]]) -> int:
    """
    Hopcroft-Karp 算法：O(E * sqrt(V)) 最大二部图匹配
    
    思路：
    - 每轮 BFS 构造所有最短增广路径的层次图
    - 之后 DFS 在层次图中批量增广（找出所有不相交的最短增广路径）
    - 重复直到不存在增广路径
    """
    INF = float('inf')
    
    # adj[u] = 左部节点 u 的右部邻居列表
    adj = [[] for _ in range(n_left)]
    for u, v in edges:
        adj[u].append(v)
    
    # 匹配数组（-1 表示未匹配）
    match_l = [-1] * n_left    # match_l[u] = 与左部 u 匹配的右部节点
    match_r = [-1] * n_right   # match_r[v] = 与右部 v 匹配的左部节点
    
    # dist[u] = 左部节点 u 在层次图中的 BFS 距离（用于只沿层次图向下）
    dist_l = [0] * n_left
    
    def bfs() -> bool:
        """
        BFS 构建增广路径层次图
        返回 True 表示存在增广路径
        """
        queue = deque()
        for u in range(n_left):
            if match_l[u] == -1:
                dist_l[u] = 0      # 未匹配左节点，层次为 0
                queue.append(u)
            else:
                dist_l[u] = INF    # 已匹配左节点，暂不可达
        
        found = False   # 是否找到增广路径（到达未匹配右节点）
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                # 经过右部节点 v，再跳回到其匹配的左节点 w
                w = match_r[v]
                if w == -1:
                    # v 是未匹配右节点：增广路径存在！
                    found = True
                    # 不立即增广，继续 BFS 直到找完所有同层的增广路径起点
                elif dist_l[w] == INF:
                    # 新发现的左节点 w（通过匹配边到达）
                    dist_l[w] = dist_l[u] + 1
                    queue.append(w)
        
        return found
    
    def dfs(u: int) -> bool:
        """
        DFS 沿层次图增广，找从左节点 u 出发的增广路径
        """
        for v in adj[u]:
            w = match_r[v]
            if w == -1 or (dist_l[w] == dist_l[u] + 1 and dfs(w)):
                # 找到增广路径：更新匹配
                match_l[u] = v
                match_r[v] = u
                return True
        
        dist_l[u] = INF   # 标记 u 在本轮层次图中已无法增广（剪枝）
        return False
    
    matching = 0
    while bfs():             # O(sqrt(V)) 轮
        for u in range(n_left):
            if match_l[u] == -1:
                if dfs(u):   # 每轮 DFS 总计 O(E)
                    matching += 1
    
    return matching
```

```cpp
// C++ 实现：Hopcroft-Karp
#include <bits/stdc++.h>
using namespace std;

class HopcroftKarp {
    int n_left, n_right;
    vector<vector<int>> adj;
    vector<int> match_l, match_r, dist_l;
    const int INF = INT_MAX;
    
    bool bfs() {
        queue<int> q;
        for (int u = 0; u < n_left; u++) {
            if (match_l[u] == -1) { dist_l[u] = 0; q.push(u); }
            else dist_l[u] = INF;
        }
        bool found = false;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v : adj[u]) {
                int w = match_r[v];
                if (w == -1) {
                    found = true;          // 存在增广路径
                } else if (dist_l[w] == INF) {
                    dist_l[w] = dist_l[u] + 1;
                    q.push(w);
                }
            }
        }
        return found;
    }
    
    bool dfs(int u) {
        for (int v : adj[u]) {
            int w = match_r[v];
            if (w == -1 || (dist_l[w] == dist_l[u] + 1 && dfs(w))) {
                match_l[u] = v;
                match_r[v] = u;
                return true;
            }
        }
        dist_l[u] = INF;   // 本轮无法增广，剪枝
        return false;
    }

public:
    HopcroftKarp(int nl, int nr) : n_left(nl), n_right(nr),
        adj(nl), match_l(nl, -1), match_r(nr, -1), dist_l(nl) {}
    
    void add_edge(int u, int v) { adj[u].push_back(v); }
    
    /**
     * 求最大匹配
     * 时间复杂度: O(E * sqrt(V))
     */
    int max_matching() {
        int result = 0;
        while (bfs()) {         // 最多 O(sqrt(V)) 轮
            for (int u = 0; u < n_left; u++)
                if (match_l[u] == -1 && dfs(u))
                    result++;
        }
        return result;
    }
};
```

<div data-component="BipartiteHopcroftKarp"></div>

### 25.2.4 转化为最大流：统一视角

Chapter 24 中我们已经看到，二部图匹配等价于最大流问题，构造方式如下：

1. 添加**超级源点** $s$，对 $L$ 中每个节点 $u$ 添加边 $s \to u$，容量为 1
2. 添加**超级汇点** $t$，对 $R$ 中每个节点 $v$ 添加边 $v \to t$，容量为 1
3. 二部图中的每条边 $(u, v)$ 在流网络中容量也设为 1

最大流 = 最大匹配。Dinic 算法（$O(V^2 E)$）作用在此图上，因为容量全为 1，可进一步化简为 $O(E\sqrt{V})$，与 Hopcroft-Karp 复杂度一致。

### 25.2.5 König 定理：最大匹配 = 最小顶点覆盖

**顶点覆盖（Vertex Cover）**：一个节点集合 $S \subseteq V$，使得图中每条边至少有一个端点在 $S$ 中。

**最小顶点覆盖**：基数最小的顶点覆盖。

**König 定理（König's Theorem）**（仅对二部图成立）：

$$\text{最大匹配} = \text{最小顶点覆盖}$$

这是一个深刻的对偶性定理（线性规划中的强对偶原理在组合数学中的体现）。在一般图中，最大匹配和最小顶点覆盖的关系由 **Gallai 定理** 给出：$|\text{最大匹配}| + |\text{最小顶点覆盖}| = |V|$（二部图）。

**从最大匹配构造最小顶点覆盖**：
1. 从所有未匹配的左节点出发，做 BFS/DFS（交替走非匹配边和匹配边）
2. 设 $Z$ 为所有可达节点的集合
3. 最小顶点覆盖 = $L \setminus Z \cup R \cap Z$（左部中不可达的节点 + 右部中可达的节点）

**最大独立集（Maximum Independent Set）** = $V \setminus$ 最小顶点覆盖，其大小为 $|V| - |\text{最大匹配}|$。

<div data-component="KonigTheoremViz"></div>

---

## 25.3 桥与割点（Bridges & Articulation Points）

### 25.3.1 定义与直觉

**网络中的薄弱环节**：想象一个互联网络，某些路由器或光缆如果故障，会直接导致网络分裂为两个无法通信的部分。在图论中，这对应两个核心概念：

**桥（Bridge）**（又称割边 / Cut Edge）：在无向连通图中，删除边 $e$ 后，图变得不连通，则 $e$ 是一条**桥**。

**割点（Articulation Point）**（又称关节点 / Cut Vertex）：在无向连通图中，删除顶点 $v$（及其所有关联边）后，图变得不连通，则 $v$ 是一个**割点**。

**暴力检测的局限性**：
- 对每条边，删除后做 BFS/DFS 检查连通性：$O(E \cdot (V+E))$——太慢
- 对每个点，删除后做 BFS/DFS：$O(V \cdot (V+E))$——同样太慢

**Tarjan 算法在一次 DFS 中找出所有桥和割点：$O(V+E)$**

### 25.3.2 Tarjan 算法：disc[] 与 low[] 的妙用

Tarjan 为每个节点维护两个值：

- $\text{disc}[v]$：**发现时间（Discovery Time）**，DFS 访问节点 $v$ 时的时间戳（全局递增）
- $\text{low}[v]$：**low-link 值**，从以 $v$ 为根的 DFS 子树中，通过**至多一条后向边**（Back Edge）能到达的最小 $\text{disc}$ 值

$$\text{low}[v] = \min\left(\text{disc}[v],\; \min_{(v,w) \text{ 后向边}} \text{disc}[w],\; \min_{(v,u) \text{ 树边}} \text{low}[u]\right)$$

**判断桥**：树边 $(u, v)$（$u$ 是 $v$ 的父节点）是桥，当且仅当 $v$ 的子树无法通过后向边"爬回" $u$ 的祖先：

$$\text{low}[v] > \text{disc}[u] \quad \Rightarrow \quad (u, v) \text{ 是桥}$$

**判断割点**：
- 对于**非根节点** $u$：若 $u$ 有一个子节点 $v$ 满足 $\text{low}[v] \geq \text{disc}[u]$，则 $u$ 是割点（$v$ 的子树无法绕过 $u$ 到达 $u$ 的上方）
- 对于**根节点** $u$：若 $u$ 在 DFS 树中有**至少 2 个子节点**，则 $u$ 是割点（删除 $u$ 后，两棵子树失去连接）

$$\begin{cases} u \text{ 非根} & \Rightarrow\; u \text{ 是割点} \iff \exists v \in \text{children}(u): \text{low}[v] \geq \text{disc}[u] \\ u \text{ 是根} & \Rightarrow\; u \text{ 是割点} \iff |\text{children}(u)| \geq 2 \end{cases}$$

**⚠️ 关键易错点**：无向图中，每条无向边 $(u, v)$ 在邻接表中以两条有向边 $(u \to v)$ 和 $(v \to u)$ 存储。DFS 中从 $u$ 访问 $v$ 之后，不能将 $(v, u)$ 视为后向边来更新 $\text{low}[u]$（否则每条树边都会因"父节点"的存在而看似不是桥）。**解决方式**：记录每条边的编号，避免通过"父边"更新 $\text{low}$。

```python
# Python 实现：Tarjan 桥与割点算法
def find_bridges_and_articulation_points(
    n: int, 
    edges: list[tuple[int, int]]
) -> tuple[list[tuple[int,int]], list[int]]:
    """
    Tarjan 算法求无向图的所有桥和割点
    
    参数:
        n:     节点数（0-indexed）
        edges: 无向边列表 [(u, v), ...]
    
    返回:
        (bridges, articulation_points)
        bridges: 桥边列表（以 (u,v) 形式，u < v）
        articulation_points: 割点列表（节点编号）
    
    时间复杂度: O(V + E)
    空间复杂度: O(V + E)
    """
    # 构建邻接表，记录边编号（避免通过父边误更新 low 值）
    adj = [[] for _ in range(n)]  # adj[u] = [(v, edge_id), ...]
    for eid, (u, v) in enumerate(edges):
        adj[u].append((v, eid))
        adj[v].append((u, eid))
    
    disc   = [-1] * n   # 发现时间（-1 表示未访问）
    low    = [0]  * n   # low-link 值
    timer  = [0]        # 全局时间戳（用列表以支持嵌套函数修改）
    
    bridges = []
    art_pts = set()
    
    def dfs(u: int, parent_eid: int) -> None:
        """
        DFS 计算 disc 和 low，识别桥与割点
        
        u:          当前节点
        parent_eid: 到达 u 时所用的边编号（-1 表示起始节点）
        """
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        child_count = 0   # u 在 DFS 树中的子节点数（用于判断根节点割点条件）
        
        for v, eid in adj[u]:
            if disc[v] == -1:
                # v 未访问：树边，递归进入
                child_count += 1
                dfs(v, eid)
                
                # 从子树返回后，用子节点的 low 值更新当前节点
                low[u] = min(low[u], low[v])
                
                # ── 判断桥 ──────────────────────────────────────
                # 树边 (u,v) 是桥：v 子树无法通过后向边回到 u 或 u 的祖先
                if low[v] > disc[u]:
                    bridges.append((min(u,v), max(u,v)))
                
                # ── 判断割点 ────────────────────────────────────
                # 非根：low[v] >= disc[u] → 删除 u 后 v 无法到达 u 以上部分
                # 根：有 2+ 个 DFS 子节点（子树之间通过 u 才相连）
                if parent_eid == -1:
                    if child_count >= 2:
                        art_pts.add(u)
                else:
                    if low[v] >= disc[u]:
                        art_pts.add(u)
            
            elif eid != parent_eid:
                # v 已访问且不是父节点：后向边，更新 low[u]
                # 注意：通过 eid != parent_eid 而非 v != parent_v
                # 是因为可能有平行边（多条同端节点的边，只有通过边编号才能正确区分）
                low[u] = min(low[u], disc[v])
    
    # 对每个连通分量分别运行 DFS
    for i in range(n):
        if disc[i] == -1:
            dfs(i, -1)
    
    return bridges, sorted(art_pts)


# ══ 示例 ══
if __name__ == "__main__":
    # 经典桥/割点例图：
    # 0-1-2-0（三角形，无桥无割点）+ 2-3-4（这条链中 2 是割点，3-4 桥）
    n = 5
    edges = [(0,1),(1,2),(2,0),(2,3),(3,4)]
    bridges, ap = find_bridges_and_articulation_points(n, edges)
    print("桥:", bridges)                  # [(2,3), (3,4)]
    print("割点:", ap)                      # [2, 3]
```

```cpp
// C++ 实现：Tarjan 桥与割点
#include <bits/stdc++.h>
using namespace std;

class TarjanBridgeAP {
    int n;
    vector<vector<pair<int,int>>> adj;  // adj[u] = {(v, edge_id)}
    vector<int> disc, low;
    int timer = 0;
    vector<pair<int,int>> bridges;
    set<int> art_pts;
    
    void dfs(int u, int parent_eid) {
        disc[u] = low[u] = timer++;
        int children = 0;
        
        for (auto [v, eid] : adj[u]) {
            if (disc[v] == -1) {
                children++;
                dfs(v, eid);
                low[u] = min(low[u], low[v]);
                
                // 判断桥：低链接值 > 父节点发现时间
                if (low[v] > disc[u])
                    bridges.push_back({min(u,v), max(u,v)});
                
                // 判断割点
                if (parent_eid == -1) {
                    if (children >= 2) art_pts.insert(u);
                } else {
                    if (low[v] >= disc[u]) art_pts.insert(u);
                }
            } else if (eid != parent_eid) {
                // 后向边（通过边ID排除父边，处理平行边）
                low[u] = min(low[u], disc[v]);
            }
        }
    }

public:
    TarjanBridgeAP(int n) : n(n), adj(n), disc(n,-1), low(n,0) {}
    
    void add_edge(int u, int v, int eid) {
        adj[u].push_back({v, eid});
        adj[v].push_back({u, eid});
    }
    
    /**
     * 执行 Tarjan 算法
     * 时间复杂度: O(V + E)
     */
    void solve() {
        for (int i = 0; i < n; i++)
            if (disc[i] == -1) dfs(i, -1);
    }
    
    const vector<pair<int,int>>& get_bridges() const { return bridges; }
    vector<int> get_articulation_points() const {
        return vector<int>(art_pts.begin(), art_pts.end());
    }
};

int main() {
    // 例图：0-1-2-0 三角 + 链 2-3-4
    int n = 5;
    vector<pair<int,int>> edges = {{0,1},{1,2},{2,0},{2,3},{3,4}};
    
    TarjanBridgeAP t(n);
    for (int i = 0; i < (int)edges.size(); i++)
        t.add_edge(edges[i].first, edges[i].second, i);
    
    t.solve();
    
    cout << "桥: ";
    for (auto [u,v] : t.get_bridges()) cout << "(" << u << "," << v << ") ";
    cout << "\n割点: ";
    for (int v : t.get_articulation_points()) cout << v << " ";
    cout << "\n";
    // 输出: 桥: (2,3) (3,4)  割点: 2 3
    
    return 0;
}
```

<div data-component="ArticulationPointHighlight"></div>

### 25.3.3 算法正确性与复杂度分析

**正确性直觉**（以桥为例）：

树边 $(u, v)$（$u$ 是父节点）是桥的条件：$v$ 的子树中没有任何后向边能"跳回" $u$ 的祖先（包括 $u$ 自身）。

- 若 $\text{low}[v] > \text{disc}[u]$：$v$ 子树中能达到的最低时间戳比 $u$ 还新，说明子树与 $u$ 之外的部分没有除 $(u,v)$ 以外的连接——$\text{删除 } (u,v)$ 后，$v$ 的子树孤立了下来，$(u,v)$ 是桥。
- 若 $\text{low}[v] \leq \text{disc}[u]$：$v$ 子树通过某条后向边连接到 $u$ 的祖先——即使删除 $(u,v)$，$v$ 的子树也能绕路到达 $u$ 以上，$(u,v)$ 不是桥。

**时间复杂度**：$O(V+E)$——每条边被访问恰好 2 次（$(u,v)$ 和 $(v,u)$），每个节点的 DFS 处理是 $O(1)$（不含邻边遍历）。

### 25.3.4 双连通分量与块-割点树

**双连通分量（Biconnected Component，BCC）**（基于顶点）：极大子图，满足删除任意单个节点后仍连通（等价于：任意两点之间至少有 2 条点不相交的路径）。

**2-边连通分量（2-Edge-Connected Component，2ECC）**（基于边）：极大子图，满足删除任意单条边后仍连通（即子图中无桥）。

**块-割点树（Block-Cut Tree）**：将每个双连通分量（"块"）和每个割点都看成一个超级节点，若某割点属于某个块，则连一条边。这样得到的树结构对许多问题大有裨益：

- 查询两点 $u, v$ 的"关键节点"（删除后 $u, v$ 不连通）：在块-割点树中找 $u$ 和 $v$ 对应节点的路径上的所有割点节点。
- 最少加几条边使图变成 2-连通：叶子块数量的一半（向上取整）。

<div data-component="BlockCutTreeBuilder"></div>

### 25.3.5 经典应用

**LeetCode 1192. 查找集群内的关键连接**：求无向图中的所有桥（关键连接），直接应用 Tarjan 算法。

**网络可靠性分析**：数据中心网络设计时，要确保没有单点故障（SPOF）——即不能有桥和割点。若存在，工程师需要增加冗余链路。

**增加最少边使图 2-连通**（无桥）：在块-割点树中，将所有叶节点"连接"起来，最少需要 $\lceil (\text{叶节点数} + 1) / 2 \rceil$ 条边。

---

## 25.4 平面图与图着色（简介）

### 25.4.1 平面图与 Euler 公式

**平面图（Planar Graph）**：可以在平面上画出，使得边只在端点处相交（不相互穿越）的图。

> **例子**：$K_4$（4个完全图）是平面图；$K_5$（5个完全图）和 $K_{3,3}$（完全二部图）不是平面图（Kuratowski 定理）。

**Euler 平面图公式**（欧拉公式）：对连通平面图，设 $V$ 为顶点数，$E$ 为边数，$F$ 为面数（包括无界外部面），则：

$$V - E + F = 2$$

**推论**：
- $E \leq 3V - 6$（三角化时等号成立）
- $E \leq 2V - 4$（无三角形时；二部图 $K_{3,3}$ 恰好满足 $E = 2V-2 < 2V-4$，因此不是平面图）
- 每个顶点的平均度数 $= 2E/V < 2(3V-6)/V = 6$，即平面图中必存在度数 $\leq 5$ 的顶点

**Euler 公式证明**（归纳法）：
- 基础：树（$F=1, E=V-1$）：$V - (V-1) + 1 = 2$ ✓
- 归纳步骤：对平面图添加一条边时（连接同一面内的两个节点），$E$ 增加 1，$F$ 增加 1（原来一个面被分成两个），$V-E+F$ 不变。

### 25.4.2 图着色

**图着色（Graph Coloring）**：给每个顶点分配一种颜色，使得相邻顶点颜色不同。**k-着色**：使用至多 $k$ 种颜色的合法着色方案。**色数（Chromatic Number）** $\chi(G)$：使图合法着色所需的最少颜色数。

**经典结论**：

| 图类型 | 色数 $\chi(G)$ |
|---|---|
| 树（无环连通图） | 2 |
| 奇数环 $C_{2k+1}$ | 3 |
| 偶数环 $C_{2k}$ | 2 |
| 二部图 | $\leq 2$ |
| 平面图 | $\leq 4$（四色定理，1976年计算机辅助证明） |
| $K_n$（完全图） | $n$ |

**四色定理**：任何平面图的色数 $\leq 4$。这是图论史上最著名的结论之一，1976年由 Appel 和 Haken 借助计算机穷举约 1936 种约化情形证明——这是数学史上第一个"依赖计算机"的重要证明。

**贪心着色算法**：按任意顺序遍历节点，为每个节点分配当前可用的最小编号颜色（不与邻居冲突）。贪心最多使用 $\Delta + 1$ 种颜色（$\Delta$ 为图的最大度数），但不保证最优。

```python
# Python 实现：贪心图着色
def greedy_coloring(n: int, edges: list[tuple[int, int]]) -> list[int]:
    """
    贪心图着色
    
    返回: color[v] = v 的颜色编号（0-indexed），保证相邻节点颜色不同
    使用颜色数 <= Delta + 1（不保证最优）
    
    时间复杂度: O(V + E)
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    color = [-1] * n   # -1 表示未着色
    
    for v in range(n):
        # 收集所有邻居已使用的颜色
        neighbor_colors = set(color[u] for u in adj[v] if color[u] != -1)
        
        # 分配最小的可用颜色
        c = 0
        while c in neighbor_colors:
            c += 1
        color[v] = c
    
    return color
```

```cpp
// C++ 实现：贪心图着色
#include <bits/stdc++.h>
using namespace std;

vector<int> greedy_coloring(int n, vector<pair<int,int>>& edges) {
    vector<vector<int>> adj(n);
    for (auto [u, v] : edges) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    
    vector<int> color(n, -1);
    
    for (int v = 0; v < n; v++) {
        // 收集邻居颜色
        set<int> used;
        for (int u : adj[v])
            if (color[u] != -1) used.insert(color[u]);
        
        // 分配最小可用颜色
        int c = 0;
        while (used.count(c)) c++;
        color[v] = c;
    }
    
    return color;
}
```

### 25.4.3 图着色的 NP-完全性

对于一般图，判断是否存在 $k$-着色方案（$k \geq 3$）是 **NP-完全**的。这意味着不存在已知的多项式时间算法来精确判断。

**为什么 $k=2$ 是完全可以的？**：2-着色 等价于 判断图是否是二部图（BFS 2-染色，$O(V+E)$）。

**为什么 $k=3$ 就 NP-难了？**：3-SAT 问题可以在多项式时间内规约到 3-着色问题（每个变量对应一对节点，每个子句对应一个三角形结构），而 3-SAT 是 NP-完全的。

**实际应用中的近似和启发式**：
- **寄存器分配（Register Allocation）**：编译器将变量分配给有限个寄存器，等价于对"干涉图"（两个变量同时活跃则有边）做图着色。
- **频率分配**：无线基站的频率分配，相邻基站不能用同一频率，等价于图着色。
- **地图着色**：相邻国家颜色不同，等价于对"邻接图"（相邻国家有边）做图着色。

---

## 25.5 算法复杂度对比与总结

| 算法 | 问题 | 时间复杂度 | 空间复杂度 | 关键思想 |
|---|---|---|---|---|
| Hierholzer | 欧拉路径 / 回路 | $O(V+E)$ | $O(V+E)$ | DFS 后序压栈 + 子回路合并 |
| 匈牙利算法 | 二部图最大匹配 | $O(VE)$ | $O(V+E)$ | 逐一找增广路径（DFS） |
| Hopcroft-Karp | 二部图最大匹配 | $O(E\sqrt{V})$ | $O(V+E)$ | BFS 分层 + DFS 批量增广 |
| 网络流 (Dinic) | 二部图最大匹配 | $O(E\sqrt{V})$ | $O(V+E)$ | 统一最大流框架 |
| Tarjan | 桥 / 割点 | $O(V+E)$ | $O(V)$ | disc[] & low[] 单次 DFS |
| 贪心着色 | 图着色（近似） | $O(V+E)$ | $O(V+E)$ | 贪心分配最小可用色 |
| Euler 公式 | 平面图判断（推论） | $O(1)$（检查边数） | — | $E \leq 3V-6$ |

**本章知识串联**：
- 欧拉路径：局部（度数）约束 → 多项式可解
- 哈密顿路径：全局约束 → NP-hard（第 42 章详述）
- 二部图匹配：从匈牙利的 $O(VE)$ 到 Hopcroft-Karp 的 $O(E\sqrt{V})$，可系统类比网络流
- Tarjan 桥/割点：单次 DFS 同时求解两类问题，low-link 技术的核心在于"能否绕路"
- 图着色：2-着色简单（二部图判断），3-着色 NP-完全——此不对称性是计算复杂性的缩影

---

## 本章常见错误与调试技巧

> ⚠️ **Hierholzer 算法**：使用前必须先验证欧拉路径/回路的度数条件。若直接运行但图不满足条件，算法会"走不完所有边"，需通过 `len(result) != E + 1` 来捕捉错误。

> ⚠️ **无向图中的 Tarjan 算法**：必须通过**边编号**（而非父节点编号）来排除父边的影响。若用 `v != parent` 的方式，遇到平行边（两个节点间多条边）会错误地将第二条平行边排除掉，误判桥/割点。

> ⚠️ **匈牙利算法 visited 数组**：`visited` 必须在每次为新左节点调用 `try_augment` 时**重置**。如果跨多次调用共用 visited，右节点被过早"封锁"，会减少实际匹配数。

> ⚠️ **有向图欧拉路径起点**：起点必须是出度比入度多 1 的节点，而不是随意选取。若从错节点出发，可能按 DFS 顺序拼出错误的路径（遗漏部分边）。

---

## 扩展阅读

- **Sedgewick 4th Ed.** 第 4.1–4.4 章：无向图与有向图的深度优先搜索，桥、割点、强连通分量
- **Skiena "Algorithm Design Manual"** 第 7 章：图算法实现细节与应用场景
- **LeetCode 高频题**：
  - #332 重新安排行程（Hierholzer）
  - #753 破解保险箱（欧拉回路）
  - #785 判断二部图（BFS 2-染色）
  - #1192 查找集群内的关键连接（桥 / Tarjan）
  - #1568 使陆地分离的最少天数（割点变体）
- **算法竞赛资源**：CP-Algorithms (cp-algorithms.com) 的 "Bridges and Articulation Points" 和 "Maximum Bipartite Matching" 章节，含完整代码和证明。

# Chapter 9: 网络层 — 路由算法与协议

> **学习目标**：
> - 理解路由的基本概念（静态/动态、全局/分布式路由）
> - 掌握距离向量算法（Bellman-Ford）及其问题（无穷计数、毒性逆转、水平分割）
> - 掌握链路状态算法（Dijkstra）及泛洪机制
> - 深入理解 OSPF 协议（区域、DR/BDR、LSA 类型、邻居状态机）
> - 理解 BGP 协议（eBGP/iBGP、AS-PATH、决策过程）
> - 区分域内路由与域间路由的设计哲学
> - 深入分析路由表、OSPF 邻居状态机、BGP 决策引擎和 LSDB 的内部实现

---

## 9.1 路由基础

### 9.1.1 什么是路由

路由（Routing）是网络层的核心功能，决定了数据报从源到目的所经过的路径。路由协议的目标是：
1. **正确性**：数据报能到达目的地
2. **最优性**：选择"最佳"路径（最短延迟、最大带宽、最小开销等）
3. **稳定性**：网络变化时快速收敛，避免路由振荡
4. **可扩展性**：支持大规模网络（互联网有超过 100 万条路由）

### 9.1.2 路由分类

| 维度 | 类型 | 说明 |
|------|------|------|
| 配置方式 | 静态路由 | 管理员手动配置 |
| | 动态路由 | 路由协议自动计算 |
| 信息来源 | 全局路由 | 每个节点知道完整拓扑（如 OSPF） |
| | 分布式路由 | 每个节点只知道邻居信息（如 RIP） |
| 作用范围 | 域内路由（IGP） | AS 内部使用（RIP、OSPF、IS-IS） |
| | 域间路由（EGP） | AS 之间使用（BGP） |

### 9.1.3 自治系统（AS）

**自治系统（Autonomous System, AS）** 是一个由单一组织管理的网络，使用统一的路由策略。每个 AS 有一个全球唯一的 AS 号（ASN）。

```
互联网路由层次:

┌─────────────────────────────────────────────┐
│                 互联网                       │
│                                             │
│  ┌─────┐  BGP   ┌─────┐  BGP   ┌─────┐   │
│  │AS 1 │◄──────►│AS 2 │◄──────►│AS 3 │   │
│  │     │        │     │        │     │   │
│  │OSPF │        │OSPF │        │IS-IS│   │
│  │     │        │     │        │     │   │
│  └─────┘        └─────┘        └─────┘   │
│  域内路由        域内路由        域内路由   │
└─────────────────────────────────────────────┘
```

---

## 9.2 距离向量算法（Distance Vector）

### 9.2.1 Bellman-Ford 方程

距离向量算法基于 Bellman-Ford 方程。设 $d(x, y)$ 为节点 $x$ 到节点 $y$ 的最短距离，$c(x, v)$ 为 $x$ 到邻居 $v$ 的链路开销：

$$d(x, y) = \min_{v \in \text{neighbors}(x)} \{c(x, v) + d(v, y)\}$$

每个节点维护一个**距离向量表**，记录到所有目的地的距离和下一跳。

### 9.2.2 算法流程

```
初始化:
  对于每个目的地 y:
    如果 y 是直接邻居:
      D(y) = c(x, y), next_hop(y) = y
    否则:
      D(y) = ∞

周期性执行（每 30 秒）:
  1. 向所有邻居发送自己的距离向量
  2. 收到邻居 v 的距离向量后:
     对于每个目的地 y:
       如果 D(v, y) + c(x, v) < D(x, y):
         D(x, y) = D(v, y) + c(x, v)
         next_hop(y) = v
```

```python
class DistanceVectorRouter:
    """距离向量路由器"""

    def __init__(self, router_id):
        self.router_id = router_id
        self.neighbors = {}     # neighbor_id -> cost
        self.distance_table = {}  # dest -> (cost, next_hop)
        self.distance_table[router_id] = (0, router_id)

    def add_neighbor(self, neighbor_id, cost):
        """添加邻居"""
        self.neighbors[neighbor_id] = cost
        self.distance_table[neighbor_id] = (cost, neighbor_id)

    def receive_dv(self, neighbor_id, neighbor_dv):
        """收到邻居的距离向量"""
        updated = False
        cost_to_neighbor = self.neighbors[neighbor_id]

        for dest, (neighbor_cost, _) in neighbor_dv.items():
            new_cost = cost_to_neighbor + neighbor_cost
            current_cost = self.distance_table.get(dest, (float('inf'), None))[0]

            if new_cost < current_cost:
                self.distance_table[dest] = (new_cost, neighbor_id)
                updated = True

        return updated

    def send_dv(self):
        """发送自己的距离向量给所有邻居"""
        return dict(self.distance_table)

    def get_routing_table(self):
        """获取路由表"""
        return {dest: (cost, nh) for dest, (cost, nh) in self.distance_table.items()}
```

### 9.2.3 无穷计数问题

当链路开销增加或链路断开时，距离向量算法可能出现**无穷计数（Count-to-Infinity）** 问题。

```
示例:

初始状态: A --1-- B --1-- C (C 到目标 X)

A 的路由表: X via B, cost=2
B 的路由表: X via C, cost=1
C 的路由表: X 直连, cost=0

场景: C 到 X 的链路断开

时间 1: C 将到 X 的距离设为 ∞
        C 告诉 B: "我到 X 的距离是 ∞"
        但 B 可能还没收到，B 告诉 A: "我到 X 的距离是 1"

时间 2: A 更新: X via B, cost=1+1=2 (不变)
        B 收到 C 的更新: X via C, cost=∞
        但 A 可能先告诉 B: "我到 X 的距离是 2"
        B 更新: X via A, cost=2+1=3

时间 3: B 告诉 A: "我到 X 的距离是 3"
        A 更新: X via B, cost=3+1=4
        ...

如此循环，直到开销增加到 ∞（通常定义为 16，即"不可达"）
```

### 9.2.4 解决方案

**1. 水平分割（Split Horizon）**：
- 不要把路由信息发回给它来的方向
- 如果路由是从邻居 N 学到的，就不再发回给 N

**2. 毒性逆转（Poison Reverse）**：
- 比水平分割更激进：把从邻居 N 学到的路由以开销 ∞ 发回给 N
- 确保邻居不会使用这条路由

```python
class SplitHorizonRouter(DistanceVectorRouter):
    """带水平分割和毒性逆转的距离向量路由器"""

    def send_dv_to_neighbor(self, neighbor_id):
        """发送距离向量给特定邻居（带水平分割）"""
        dv = {}
        for dest, (cost, next_hop) in self.distance_table.items():
            if next_hop == neighbor_id and dest != neighbor_id:
                # 水平分割: 不发回给来源方向
                # 毒性逆转: 发回 ∞
                dv[dest] = (float('inf'), self.router_id)
            else:
                dv[dest] = (cost, self.router_id)
        return dv
```

### 9.2.5 RIP 协议

RIP（Routing Information Protocol）是最著名的距离向量协议：

| 参数 | 值 |
|------|-----|
| 度量 | 跳数（最大 15，16=不可达） |
| 更新周期 | 30 秒 |
| 超时时间 | 180 秒 |
| 垃圾回收时间 | 120 秒 |
| 最大网络直径 | 15 跳 |

<div data-component="DistanceVectorDemo"></div>

---

## 9.3 链路状态算法（Link State）

### 9.3.1 链路状态路由的基本思想

链路状态算法的核心思想：
1. 每个节点发现其邻居及链路开销
2. 每个节点将链路状态信息**泛洪（Flooding）** 给所有其他节点
3. 每个节点拥有完整的网络拓扑图（链路状态数据库，LSDB）
4. 每个节点独立运行 Dijkstra 算法计算到所有目的地的最短路径

### 9.3.2 Dijkstra 算法

设 $N$ 为所有节点集合，$l(i, j)$ 为节点 $i$ 到节点 $j$ 的链路开销（$\infty$ 表示不直接相连）。

```
Dijkstra 算法:

初始化:
  N' = {源节点 u}
  对于所有节点 v:
    如果 v 是 u 的邻居:
      D(v) = l(u, v), p(v) = u
    否则:
      D(v) = ∞

循环 N-1 次:
  1. 找到不在 N' 中且 D(w) 最小的节点 w
  2. 将 w 加入 N'
  3. 更新所有 w 的邻居 v 的距离:
     如果 D(w) + l(w, v) < D(v):
       D(v) = D(w) + l(w, v)
       p(v) = w
```

```python
import heapq

def dijkstra(graph, source):
    """
    Dijkstra 最短路径算法
    graph: {node: {neighbor: cost, ...}, ...}
    返回: {node: (dist, path), ...}
    """
    dist = {source: 0}
    prev = {source: None}
    visited = set()
    pq = [(0, source)]  # (距离, 节点)

    while pq:
        d, u = heapq.heappop(pq)

        if u in visited:
            continue
        visited.add(u)

        for v, cost in graph.get(u, {}).items():
            if v in visited:
                continue
            new_dist = d + cost
            if v not in dist or new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    # 构建路径
    paths = {}
    for node in dist:
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = prev.get(current)
        paths[node] = (dist[node], list(reversed(path)))

    return paths

# 示例网络
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1},
}

result = dijkstra(graph, 'A')
for node, (dist, path) in sorted(result.items()):
    print(f"A → {node}: cost={dist}, path={' → '.join(path)}")
```

**输出**：
```
A → A: cost=0, path=A
A → B: cost=1, path=A → B
A → C: cost=3, path=A → B → C
A → D: cost=4, path=A → B → C → D
```

### 9.3.3 泛洪机制

链路状态信息通过泛洪传播到整个网络：

```
泛洪过程:

1. 节点 A 检测到链路变化（如 A-B 链路开销变化）
2. A 构造链路状态通告（LSA）
3. A 将 LSA 发送给所有邻居
4. 每个收到 LSA 的节点:
   a. 检查 LSA 是否是新的（通过序列号判断）
   b. 如果是新的:
      - 更新自己的 LSDB
      - 将 LSA 转发给除来源外的所有邻居
   c. 如果是旧的/重复的:
      - 丢弃，不转发
5. 所有节点收到 LSA 后，重新运行 Dijkstra 算法
```

**泛洪的可靠性保证**：
- 每个 LSA 有**序列号**，递增，用于去重和判断新旧
- 每个 LSA 有**生存时间**，超时后从 LSDB 中删除
- 确认机制：收到 LSA 后发送确认

<div data-component="DijkstraDemo"></div>

---

## 9.4 OSPF 协议

### 9.4.1 OSPF 概述

OSPF（Open Shortest Path First）是最重要的链路状态路由协议，属于 IGP（内部网关协议）。

**OSPF 的特点**：
- 开放标准（RFC 2328）
- 基于链路状态（使用 Dijkstra 算法）
- 支持 VLSM 和 CIDR
- 支持多区域层次化设计
- 快速收敛（事件触发更新）
- 支持认证

### 9.4.2 OSPF 区域

为了支持大规模网络，OSPF 引入了**区域（Area）** 的概念：

```
OSPF 多区域设计:

            ┌───────────────┐
            │   Area 0      │
            │  (骨干区域)    │
            │               │
            │  ABR1    ABR2 │
            └───┬───────┬───┘
                │       │
        ┌───────┴──┐ ┌──┴───────┐
        │  Area 1  │ │  Area 2  │
        │          │ │          │
        │ ASBR     │ │          │
        └──────────┘ └──────────┘

路由器角色:
- IR (Internal Router): 所有接口在同一区域
- ABR (Area Border Router): 连接多个区域
- ASBR (AS Boundary Router): 连接其他 AS
- Backbone Router: 至少有一个接口在 Area 0
```

**区域的作用**：
1. 减少 LSA 泛洪范围
2. 减小 LSDB 和路由表大小
3. 加速收敛
4. 提高可扩展性

### 9.4.3 OSPF LSA 类型

| LSA 类型 | 名称 | 产生者 | 范围 | 内容 |
|---------|------|--------|------|------|
| Type 1 | Router LSA | 每个路由器 | 区域内 | 路由器的链路和开销 |
| Type 2 | Network LSA | DR | 区域内 | 多路接入网络的所有路由器 |
| Type 3 | Summary LSA | ABR | 区域间 | 到其他区域的路由汇总 |
| Type 4 | ASBR Summary LSA | ABR | 区域间 | 到 ASBR 的路由 |
| Type 5 | AS External LSA | ASBR | 整个 AS | 到 AS 外部的路由 |
| Type 7 | NSSA External LSA | ASBR | NSSA 区域 | NSSA 的外部路由 |

### 9.4.4 DR/BDR 选举

在多路接入网络（如以太网）中，为减少 OSPF 邻居关系数量和 LSA 泛洪，选举**指定路由器（DR）** 和**备份指定路由器（BDR）**。

**选举规则**：
1. 接口 OSPF 优先级最高的路由器成为 DR（默认优先级=1）
2. 优先级相同则比较 Router ID，大的优先
3. 优先级=0 的路由器不参与选举
4. DR/BDR 选举是非抢占的（一旦选出，除非 DR 故障才重新选举）

```
以太网上的 OSPF 邻居关系:

无 DR/BDR (全网格):  N 个路由器需要 N(N-1)/2 个邻居关系
有 DR/BDR (星型):    N 个路由器只需要 2(N-2)+1 个邻居关系

例: 6 个路由器
无 DR: 6×5/2 = 15 个邻居关系
有 DR: 2×4+1 = 9 个邻居关系
```

### 9.4.5 OSPF 邻居状态机

OSPF 路由器之间的邻居关系经历 7 个状态：

```
┌──────────┐
│  Down    │  初始状态，未收到任何 Hello
└────┬─────┘
     │ 收到 Hello
     ▼
┌──────────┐
│  Init    │  收到邻居的 Hello，但 Hello 中没有自己的 Router ID
└────┬─────┘
     │ 收到包含自己 Router ID 的 Hello
     ▼
┌──────────┐
│  2-Way   │  双向通信建立，决定是否继续（DR/BDR 选举在此状态）
└────┬─────┘
     │ 决定建立邻接关系
     ▼
┌──────────┐
│ ExStart  │  协商主从关系和初始序列号
└────┬─────┘
     │ 协商完成
     ▼
┌──────────┐
│ Exchange │  交换 DBD（Database Description）包
└────┬─────┘
     │ 发现需要更新的 LSA
     ▼
┌──────────┐
│ Loading  │  交换 LSR/LSU/LSAck，同步 LSDB
└────┬─────┘
     │ LSDB 同步完成
     ▼
┌──────────┐
│  Full    │  邻接关系完全建立，LSDB 一致
└──────────┘
```

```python
class OSPFNeighborStateMachine:
    """OSPF 邻居状态机"""

    def __init__(self, router_id, neighbor_id):
        self.router_id = router_id
        self.neighbor_id = neighbor_id
        self.state = 'Down'
        self.master = False
        self.dd_seq = 0
        self.lsdb_request_list = []
        self.lsdb_retransmit_list = []

    def receive_hello(self, hello):
        """收到 Hello 包"""
        if self.state == 'Down':
            self.state = 'Init'

        if self.state == 'Init':
            if self.router_id in hello.neighbors:
                self.state = '2-Way'
                return '2-Way_reached'

        if self.state in ('2-Way', 'ExStart', 'Exchange', 'Loading', 'Full'):
            # 更新活跃定时器
            self.reset_hello_timer()
            return 'hello_received'

    def start_adjacency(self):
        """决定建立邻接关系"""
        if self.state == '2-Way':
            self.state = 'ExStart'
            self.dd_seq = self.router_id  # 使用 Router ID 作为初始序列号
            return 'negotiation_started'

    def receive_dbd(self, dbd):
        """收到 Database Description 包"""
        if self.state == 'ExStart':
            # 协商主从关系
            if dbd.initial and dbd.more and dbd.master:
                if self.router_id > self.neighbor_id:
                    self.master = True
                self.state = 'Exchange'
                return 'exchange_started'

        if self.state == 'Exchange':
            # 比较 LSDB，构建请求列表
            for lsa_header in dbd.lsa_headers:
                if self._need_update(lsa_header):
                    self.lsdb_request_list.append(lsa_header)

            if not dbd.more and self._all_dbd_received():
                if self.lsdb_request_list:
                    self.state = 'Loading'
                    return 'loading_started'
                else:
                    self.state = 'Full'
                    return 'adjacency_complete'

    def receive_lsu(self, lsu):
        """收到 Link State Update"""
        if self.state == 'Loading':
            # 从请求列表中移除已收到的 LSA
            for lsa in lsu.lsas:
                if lsa in self.lsdb_request_list:
                    self.lsdb_request_list.remove(lsa)

            if not self.lsdb_request_list:
                self.state = 'Full'
                return 'adjacency_complete'
```

<div dataComponent="OSPFNeighborDemo"></div>

### 9.4.6 OSPF 包类型

| 包类型 | 说明 | 用途 |
|--------|------|------|
| Hello | Hello 包 | 发现和维护邻居关系 |
| DBD | Database Description | 描述 LSDB 摘要 |
| LSR | Link State Request | 请求特定 LSA |
| LSU | Link State Update | 发送 LSA |
| LSAck | Link State Acknowledgment | 确认收到 LSA |

---

## 9.5 BGP 协议

### 9.5.1 BGP 概述

BGP（Border Gateway Protocol）是互联网的域间路由协议，也称为**路径向量（Path Vector）** 协议。

**BGP 的特点**：
- 用于 AS 之间的路由（也可用于 AS 内部）
- 基于策略的路由（不仅仅是最短路径）
- 使用 TCP 连接（端口 179）
- 增量更新（只发送变化）
- 丰富的路由属性和策略控制

### 9.5.2 eBGP 与 iBGP

```
eBGP: 不同 AS 之间的 BGP 对等
iBGP: 同一 AS 内部的 BGP 对等

         AS 100                AS 200
┌───────────────────┐  ┌───────────────────┐
│                   │  │                   │
│  R1 ──iBGP── R2  │  │  R4 ──iBGP── R5  │
│   │          │    │  │    │          │    │
│   └────eBGP──┘────│──│────┘          │    │
│                   │  │               │    │
│       R3 ─────────│──│───eBGP───────R6    │
│                   │  │                   │
└───────────────────┘  └───────────────────┘
```

**iBGP 的全网格要求**：
- iBGP 路由器不会将从 iBGP 邻居学到的路由再传给其他 iBGP 邻居
- 因此同一 AS 内所有 iBGP 路由器需要全网格连接
- 大型 AS 使用路由反射器（Route Reflector）或联盟（Confederation）减少 iBGP 连接数

### 9.5.3 BGP 路由属性

| 属性 | 类型 | 说明 |
|------|------|------|
| AS-PATH | 公认必遵 | 路由经过的 AS 列表，用于环路检测和路径选择 |
| NEXT-HOP | 公认必遵 | 下一跳 IP 地址 |
| LOCAL_PREF | 公认任意 | 本地优先级（值越大越优先），用于 iBGP |
| MED | 可选非传递 | 多出口鉴别器，影响入站流量 |
| COMMUNITY | 可选传递 | 路由标记，用于策略匹配 |
| ORIGIN | 公认必遵 | 路由来源（IGP/EGP/Incomplete） |

### 9.5.4 AS-PATH 与环路检测

AS-PATH 是 BGP 最重要的属性之一：

```
AS-PATH 示例:

AS 100 产生路由 10.0.0.0/8:
  R1(AS100): AS-PATH = 100

AS 200 从 AS 100 学到:
  R4(AS200): AS-PATH = 200 100

AS 300 从 AS 200 学到:
  R7(AS300): AS-PATH = 300 200 100

环路检测:
如果 AS-PATH 中已包含自己的 AS 号，则丢弃该路由
```

### 9.5.5 BGP 决策过程

BGP 选择最佳路由的决策过程有 13 步（按优先级顺序）：

```
BGP 最佳路由选择顺序:

1. 最高 LOCAL_PREF (本地优先级)
2. 最短 AS-PATH 长度
3. 最低 ORIGIN 类型 (IGP < EGP < Incomplete)
4. 最低 MED (多出口鉴别器)
5. eBGP 路由优先于 iBGP 路由
6. 最近的 IGP 下一跳（到下一跳的 IGP 度量最小）
7. 最旧的 eBGP 路由（稳定性考虑）
8. 最低 BGP Router ID
9. 最低的邻居 IP 地址
```

```python
class BGPDecisionEngine:
    """BGP 决策引擎"""

    def select_best_route(self, routes):
        """选择最佳路由"""
        if not routes:
            return None

        best = routes[0]
        for route in routes[1:]:
            best = self._compare(best, route)
        return best

    def _compare(self, route1, route2):
        """比较两条路由，返回更优的"""

        # 1. 最高 LOCAL_PREF
        if route1.local_pref != route2.local_pref:
            return route1 if route1.local_pref > route2.local_pref else route2

        # 2. 最短 AS-PATH
        len1 = len(route1.as_path)
        len2 = len(route2.as_path)
        if len1 != len2:
            return route1 if len1 < len2 else route2

        # 3. 最低 ORIGIN
        origin_order = {'IGP': 0, 'EGP': 1, 'Incomplete': 2}
        if origin_order[route1.origin] != origin_order[route2.origin]:
            o1 = origin_order[route1.origin]
            o2 = origin_order[route2.origin]
            return route1 if o1 < o2 else route2

        # 4. 最低 MED
        if route1.med != route2.med:
            return route1 if route1.med < route2.med else route2

        # 5. eBGP 优先于 iBGP
        if route1.type != route2.type:
            return route1 if route1.type == 'eBGP' else route2

        # 6. 最近的 IGP 下一跳
        if route1.igp_metric != route2.igp_metric:
            return route1 if route1.igp_metric < route2.igp_metric else route2

        # 7. 最旧的 eBGP 路由
        if route1.type == 'eBGP' and route2.type == 'eBGP':
            return route1 if route1.age > route2.age else route2

        # 8. 最低 Router ID
        return route1 if route1.router_id < route2.router_id else route2
```

<div data-component="BGPDecisionDemo"></div>

---

## 9.6 域内路由 vs 域间路由

### 9.6.1 设计哲学对比

| 特性 | 域内路由（IGP） | 域间路由（EGP/BGP） |
|------|----------------|---------------------|
| 目标 | 找到最短路径 | 实现策略控制 |
| 度量 | 链路开销、延迟 | AS-PATH、策略、商业关系 |
| 信息 | 完整拓扑 | 仅可达性信息 |
| 收敛 | 快速（秒级） | 较慢（分钟级） |
| 规模 | 单个 AS 内（数千路由器） | 全互联网（百万路由） |
| 信任 | 同一组织，可信 | 不同组织，不可信 |

### 9.6.2 BGP 与 IGP 的协同

```
数据包转发路径:

源主机 → AS 100 (OSPF) → AS 200 (OSPF) → AS 300 (OSPF) → 目的主机

路由计算路径:
- BGP: 计算 AS 级路径 (AS100 → AS200 → AS300)
- IGP: 计算 AS 内部路径 (具体路由器到路由器)

协同工作:
1. BGP 学到外部路由，设置 NEXT-HOP 为对端 AS 的 IP
2. IGP 负责找到到达 NEXT-HIP 的内部路径
3. 转发时：查 BGP 路由表得到下一跳 AS 的 IP，再查 IGP 路由表得到内部下一跳
```

---

## 9.7 核心组件深入分析

### 9.7.1 路由表的数据结构与查找算法

**路由表的高效数据结构**：

```c
/* 路由表条目 */
struct route_entry {
    uint32_t    prefix;         /* 网络前缀（主机字节序） */
    uint8_t     prefix_len;     /* 前缀长度 (0-32) */
    uint32_t    next_hop;       /* 下一跳 IP */
    uint8_t     out_interface;  /* 出接口编号 */
    uint32_t    metric;         /* 路由度量 */
    uint8_t     protocol;       /* 来源协议 (OSPF/BGP/STATIC) */
    uint8_t     type;           /* 路由类型 (internal/external) */
    uint32_t    flags;          /* 标志位 */
    time_t      last_update;    /* 最后更新时间 */
    uint32_t    ref_count;      /* 引用计数 */
};

/* Patricia Trie 节点 */
struct patricia_node {
    uint32_t    prefix;
    uint8_t     prefix_len;
    uint8_t     bit_position;   /* 分歧位位置 */
    struct route_entry *route;  /* 路由条目（可能为 NULL） */
    struct patricia_node *left;
    struct patricia_node *right;
};

/* Patricia Trie 路由查找 */
struct route_entry* patricia_lookup(struct patricia_node *root,
                                      uint32_t dest_ip) {
    struct patricia_node *node = root;
    struct route_entry *best_match = NULL;

    while (node != NULL) {
        /* 记录当前节点的路由（如果有） */
        if (node->route != NULL) {
            best_match = node->route;
        }

        /* 根据分歧位决定走向 */
        int bit = (dest_ip >> (31 - node->bit_position)) & 1;
        if (bit == 0) {
            node = node->left;
        } else {
            node = node->right;
        }
    }

    return best_match;
}
```

### 9.7.2 OSPF 邻居状态机与 LSDB 同步过程

```c
/* OSPF 邻居状态机实现 */

enum ospf_neighbor_state {
    OSPF_DOWN,
    OSPF_INIT,
    OSPF_2WAY,
    OSPF_EXSTART,
    OSPF_EXCHANGE,
    OSPF_LOADING,
    OSPF_FULL
};

struct ospf_neighbor {
    uint32_t                neighbor_id;
    uint32_t                neighbor_ip;
    enum ospf_neighbor_state state;
    uint8_t                 priority;
    uint32_t                dr;
    uint32_t                bdr;
    uint32_t                dd_seq_number;
    uint8_t                 master;
    struct ospf_lsa_header  *db_summary;     /* DBD 中的 LSA 摘要列表 */
    struct ospf_lsa_header  *link_state_req; /* 待请求的 LSA 列表 */
    struct ospf_lsa_header  *retransmit_list; /* 待重传的 LSA 列表 */
    timer_t                 hello_timer;
    timer_t                 dead_timer;
    timer_t                 retransmit_timer;
};

void ospf_neighbor_state_machine(struct ospf_neighbor *nbr,
                                  struct ospf_interface *intf,
                                  int event,
                                  void *data) {
    switch (nbr->state) {
    case OSPF_DOWN:
        if (event == OSPF_EVT_HELLO_RECEIVED) {
            nbr->state = OSPF_INIT;
            reset_timer(&nbr->dead_timer, intf->dead_interval);
        }
        break;

    case OSPF_INIT:
        if (event == OSPF_EVT_2WAY_RECEIVED) {
            /* 检查是否需要建立邻接关系 */
            if (intf->type == OSPF_IF_BROADCAST ||
                intf->type == OSPF_IF_NBMA) {
                /* 在多路接入网络中，只与 DR/BDR 建立邻接 */
                if (nbr->neighbor_id == intf->dr ||
                    nbr->neighbor_id == intf->bdr ||
                    intf->router_id == intf->dr ||
                    intf->router_id == intf->bdr) {
                    nbr->state = OSPF_EXSTART;
                    nbr->dd_seq_number = rand();
                    send_dbd(nbr, intf, DBD_INIT | DBD_MORE | DBD_MASTER);
                }
            } else {
                /* 点对点网络，直接建立邻接 */
                nbr->state = OSPF_EXSTART;
            }
        }
        break;

    case OSPF_EXSTART:
        if (event == OSPF_EVT_NEGOTIATION_DONE) {
            nbr->state = OSPF_EXCHANGE;
            nbr->db_summary = build_db_summary(intf->lsdb);
        }
        break;

    case OSPF_EXCHANGE:
        if (event == OSPF_EVT_EXCHANGE_DONE) {
            if (nbr->link_state_req != NULL) {
                nbr->state = OSPF_LOADING;
                send_lsr(nbr, intf);
            } else {
                nbr->state = OSPF_FULL;
                trigger_spf_calculation(intf);
            }
        }
        break;

    case OSPF_LOADING:
        if (event == OSPF_EVT_LOADING_DONE) {
            nbr->state = OSPF_FULL;
            trigger_spf_calculation(intf);
        }
        break;

    case OSPF_FULL:
        /* 邻接完全建立，维护状态 */
        if (event == OSPF_EVT_HELLO_RECEIVED) {
            reset_timer(&nbr->dead_timer, intf->dead_interval);
        }
        if (event == OSPF_EVT_ADJACENCY_OK) {
            /* 链路变化，需要重新同步 */
            nbr->state = OSPF_EXSTART;
        }
        break;
    }
}
```

### 9.7.3 BGP 决策引擎的选路规则

```c
/* BGP 决策引擎 - 13 步选路规则 */

struct bgp_route {
    uint32_t    prefix;
    uint8_t     prefix_len;
    uint32_t    next_hop;
    uint32_t    local_pref;     /* LOCAL_PREF 属性 */
    uint32_t    as_path_len;    /* AS-PATH 长度 */
    uint32_t   *as_path;        /* AS-PATH 数组 */
    uint8_t     origin;         /* IGP=0, EGP=1, Incomplete=2 */
    uint32_t    med;            /* MED 属性 */
    uint8_t     route_type;     /* eBGP=1, iBGP=2 */
    uint32_t    igp_metric;    /* 到下一跳的 IGP 度量 */
    uint32_t    router_id;     /* BGP Router ID */
    time_t      age;           /* 路由年龄 */
    uint32_t    peer_ip;       /* 邻居 IP */
};

struct bgp_route* bgp_best_path_select(struct bgp_route **routes,
                                         int count) {
    if (count == 0) return NULL;

    struct bgp_route *best = routes[0];

    for (int i = 1; i < count; i++) {
        struct bgp_route *candidate = routes[i];

        /* 规则 1: 最高 LOCAL_PREF */
        if (candidate->local_pref != best->local_pref) {
            if (candidate->local_pref > best->local_pref)
                best = candidate;
            continue;
        }

        /* 规则 2: 最短 AS-PATH */
        if (candidate->as_path_len != best->as_path_len) {
            if (candidate->as_path_len < best->as_path_len)
                best = candidate;
            continue;
        }

        /* 规则 3: 最低 ORIGIN */
        if (candidate->origin != best->origin) {
            if (candidate->origin < best->origin)
                best = candidate;
            continue;
        }

        /* 规则 4: 最低 MED */
        if (candidate->med != best->med) {
            if (candidate->med < best->med)
                best = candidate;
            continue;
        }

        /* 规则 5: eBGP 优先于 iBGP */
        if (candidate->route_type != best->route_type) {
            if (candidate->route_type == 1)  /* eBGP */
                best = candidate;
            continue;
        }

        /* 规则 6: 最近的 IGP 下一跳 */
        if (candidate->igp_metric != best->igp_metric) {
            if (candidate->igp_metric < best->igp_metric)
                best = candidate;
            continue;
        }

        /* 规则 7: 最旧的 eBGP 路由 */
        if (candidate->route_type == 1 && best->route_type == 1) {
            if (candidate->age > best->age)
                best = candidate;
            continue;
        }

        /* 规则 8: 最低 Router ID */
        if (candidate->router_id < best->router_id) {
            best = candidate;
        }
    }

    return best;
}
```

### 9.7.4 链路状态数据库（LSDB）的存储结构

```c
/* LSDB 存储结构 */

#define MAX_LSA_AGE    3600  /* LSA 最大存活时间（秒） */
#define LS_REFRESH_TIME 1800 /* LSA 刷新时间（秒） */

struct lsa_key {
    uint32_t    link_state_id;   /* LSA 标识 */
    uint32_t    adv_router;      /* 通告路由器 */
    uint8_t     lsa_type;        /* LSA 类型 */
};

struct ospf_lsa {
    struct lsa_key  key;
    uint32_t        sequence;    /* 序列号 */
    uint16_t        age;         /* 当前年龄（秒） */
    uint16_t        checksum;    /* 校验和 */
    uint8_t        *data;        /* LSA 数据 */
    uint16_t        data_len;    /* 数据长度 */
    time_t          install_time; /* 安装时间 */
};

/* LSDB 使用哈希表存储 */
#define LSDB_HASH_SIZE  1024

struct lsdb {
    struct ospf_lsa *hash_table[LSDB_HASH_SIZE];
    int             lsa_count;
    pthread_rwlock_t lock;
    time_t          last_spf_run;
};

/* LSDB 查找 */
struct ospf_lsa* lsdb_lookup(struct lsdb *db, struct lsa_key *key) {
    uint32_t hash = lsa_hash(key, LSDB_HASH_SIZE);
    struct ospf_lsa *lsa = db->hash_table[hash];

    while (lsa) {
        if (lsa_key_equal(&lsa->key, key)) {
            return lsa;
        }
        lsa = lsa->next;
    }
    return NULL;
}

/* LSDB 安装 LSA */
int lsdb_install(struct lsdb *db, struct ospf_lsa *new_lsa) {
    struct ospf_lsa *existing = lsdb_lookup(db, &new_lsa->key);

    if (existing != NULL) {
        /* 比较序列号 */
        if (new_lsa->sequence <= existing->sequence) {
            return 0;  /* 旧的或重复的，忽略 */
        }
        /* 更新现有 LSA */
        replace_lsa(db, existing, new_lsa);
    } else {
        /* 插入新 LSA */
        insert_lsa(db, new_lsa);
    }

    /* 触发 SPF 重新计算 */
    trigger_spf(db);

    return 1;
}
```

<div data-component="LSDBDemo"></div>

---

## 9.8 路由协议性能对比

| 协议 | 类型 | 算法 | 度量 | 收敛时间 | 扩展性 |
|------|------|------|------|---------|--------|
| RIP | IGP | Bellman-Ford | 跳数 | 慢(分钟级) | 差(15跳限制) |
| OSPF | IGP | Dijkstra | 开销 | 快(秒级) | 好(多区域) |
| IS-IS | IGP | Dijkstra | 开销 | 快(秒级) | 很好 |
| BGP | EGP | 策略 | AS-PATH等 | 慢(分钟级) | 极好(互联网级) |

---

## 9.9 本章总结

| 概念 | 核心要点 |
|------|---------|
| 距离向量 | Bellman-Ford 方程，分布式计算，无穷计数问题 |
| 水平分割/毒性逆转 | 解决无穷计数的两种方法 |
| 链路状态 | Dijkstra 算法，全局拓扑，泛洪分发 LSA |
| OSPF | 多区域设计，DR/BDR，7 种 LSA 类型，邻居状态机 |
| BGP | AS-PATH 属性，eBGP/iBGP，13 步决策过程 |
| 域内 vs 域间 | IGP 选最短路径，BGP 选策略路径 |

<div data-component="ChapterSummaryQuiz"></div>

---

## 练习题

**1. 算法题**：对以下网络运行 Dijkstra 算法，求从节点 A 到所有其他节点的最短路径：
- A-B: 2, A-C: 5, B-D: 3, B-E: 1, C-D: 2, D-F: 4, E-F: 3

**2. 分析题**：解释为什么 OSPF 在多路接入网络中需要选举 DR/BDR，而不选举会出现什么问题。

**3. 比较题**：对比距离向量算法和链路状态算法的优缺点。

**4. BGP 题**：某 AS 从两个 eBGP 邻居收到同一前缀的路由，LOCAL_PREF 分别为 100 和 200，AS-PATH 长度分别为 3 和 5。根据 BGP 决策规则，哪条路由会被选为最佳？

**5. 设计题**：设计一个简化的 OSPF 实现，支持点对点网络上的邻居发现和 LSDB 同步。

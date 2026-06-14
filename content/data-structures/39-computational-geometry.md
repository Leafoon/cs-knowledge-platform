---
title: "Chapter 39: 计算几何核心算法"
description: "叉积与点积原语、线段交叉判断、凸包（Graham/Jarvis/Andrew）、最近点对分治——几何算法的基石与工程实践"
tags: ["computational-geometry", "convex-hull", "closest-pair", "cross-product", "graham-scan", "segment-intersection"]
difficulty: "hard"
updated: "2026-03-12"
---

# Chapter 39: 计算几何核心算法

> **Part XI · 计算几何基础**

计算几何（**Computational Geometry**）研究的是如何用算法高效地解决几何问题——点、线、多边形、凸包……这些听起来很"数学"，但它们却深藏在日常应用的底层：

- **导航/GIS**：地图中判断某点是否在某区域内、两条道路是否相交
- **游戏开发**：碰撞检测（物体是否相撞？）、视野遮挡（敌人是否可见？）
- **机器人规划**：路径规避障碍物
- **计算机视觉**：人脸关键点三角剖分（Delaunay）、图像轮廓提取
- **竞赛算法**：凸包、最近点对、扫描线——ACM/ICPC 的高频考点

本章从最基础的几何原语（叉积/点积）出发，逐步掌握线段交叉判断、凸包的多种构建算法，以及经典的「最近点对」分治算法，为后续计算几何进阶打下坚实基础。

---

## 39.1 几何基础与原语

### 39.1.1 点与向量的表示

#### 通俗比喻：坐标系就是城市地图

把平面想象成一张城市地图，每个路口用一对坐标 $(x, y)$ 来定位。从路口 $A$ 走到路口 $B$，这段"行程"就是一个向量：方向是从 $A$ 指向 $B$，大小是走了多远。

> **点（Point）**：空间中的一个位置，用坐标 $(x, y)$ 表示（2D）或 $(x, y, z)$（3D）。  
> **向量（Vector）**：有方向、有大小的量。从 $A$ 到 $B$ 的向量 $\vec{AB} = B - A = (B.x - A.x,\ B.y - A.y)$。

**关键区分**：点是位置，向量是位移。  
虽然它们的数学表示形式相同（都是一对数），但含义不同。在代码实现中，通常用同一个结构体 `Point` 来同时表达这两个概念，通过上下文区分。

**数据结构定义**：

```python
class Point:
    """二维点/向量：支持加减法、数乘、叉积、点积等运算"""
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    # 向量加法：A + B
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    # 向量减法：A - B（从 B 指向 A 的向量）
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    # 数乘：k * A
    def __mul__(self, k: float):
        return Point(self.x * k, self.y * k)

    # 向量的欧氏长度（模长）|A|
    def norm(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def __repr__(self):
        return f"({self.x}, {self.y})"

# 使用示例
A = Point(1, 2)
B = Point(4, 6)
AB = B - A          # 向量 AB = (3, 4)
print(AB.norm())    # 5.0（勾股定理：3-4-5 三角形）
```

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef double ld;

// ---- 几何点/向量结构体 ----
// 【注意】竞赛中常用 long long 避免浮点误差（仅限整数坐标输入）
//         若坐标为浮点数，改用 double/long double
struct Point {
    ll x, y;

    Point(ll x = 0, ll y = 0) : x(x), y(y) {}

    // 向量加减
    Point operator+(const Point& o) const { return {x + o.x, y + o.y}; }
    Point operator-(const Point& o) const { return {x - o.x, y - o.y}; }

    // 数乘（缩放）
    Point operator*(ll k) const { return {x * k, y * k}; }

    // 比较（排序用）：先按 x，再按 y
    bool operator<(const Point& o) const {
        return x != o.x ? x < o.x : y < o.y;
    }
    bool operator==(const Point& o) const {
        return x == o.x && y == o.y;
    }

    // 欧氏距离（注意返回 double）
    double norm() const {
        return sqrt((double)(x * x + y * y));
    }
};

// 两点之间距离
double dist(Point a, Point b) {
    return (b - a).norm();
}
```

---

### 39.1.2 叉积（Cross Product）：判断转向的核心工具

叉积是计算几何中**最重要**的工具，没有之一。它能帮我们回答这样的问题：

> 站在 $O$ 点看向 $A$ 点，再从 $O$ 看向 $B$ 点，是向**左转**了，还是向**右转**，还是**正对着**（共线）？

#### 叉积的公式

对于两个向量 $\vec{OA} = (a_1, a_2)$ 和 $\vec{OB} = (b_1, b_2)$，它们的**二维叉积**（也叫"外积"的 z 分量）定义为：

$$\vec{OA} \times \vec{OB} = a_1 \cdot b_2 - a_2 \cdot b_1$$

**几何意义**：

| 叉积值 | 几何含义 | 方向 |
|--------|---------|------|
| $> 0$ | $B$ 在 $A$ 的**左侧**（$\vec{OA}$ 到 $\vec{OB}$ 逆时针） | 左转 ↺ |
| $< 0$ | $B$ 在 $A$ 的**右侧**（$\vec{OA}$ 到 $\vec{OB}$ 顺时针） | 右转 ↻ |
| $= 0$ | $O$、$A$、$B$ **共线** | 无转向 |

还有一个重要的几何意义：**$|\vec{OA} \times \vec{OB}|$ 等于以 $\vec{OA}$ 和 $\vec{OB}$ 为两边构成的平行四边形的面积**。

$$\text{平行四边形面积} = |\vec{OA} \times \vec{OB}|$$
$$\text{三角形面积} = \frac{1}{2}|\vec{OA} \times \vec{OB}|$$

**直觉记忆法**：想象自己站在原点，手指指向 $A$，然后转到指向 $B$。如果是逆时针（左转，往 $A$ 的左边），叉积为正；顺时针（右转），叉积为负。

<div data-component="CrossProductViz"></div>

```python
def cross(O: Point, A: Point, B: Point) -> float:
    """
    计算向量 OA 和 OB 的叉积。
    返回值含义：
      > 0 → B 在 OA 的左侧（逆时针）
      < 0 → B 在 OA 的右侧（顺时针）
      = 0 → O、A、B 三点共线
    【注意】整数坐标下返回整数，不会有浮点误差
    """
    oa = A - O  # 向量 OA
    ob = B - O  # 向量 OB
    return oa.x * ob.y - oa.y * ob.x
```

```cpp
// 【精度要点】整数坐标用 long long，绝对避免浮点比较
// cross(O, A, B) = (A-O) × (B-O)
ll cross(Point O, Point A, Point B) {
    // 展开：(A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x)
    // 注意：中间计算可能溢出 int！坐标最大约 1e9 时，乘积最大约 1e18，需要 long long
    return (ll)(A.x - O.x) * (B.y - O.y) - (ll)(A.y - O.y) * (B.x - O.x);
}

// 两向量叉积（以原点为基准）
ll cross2(Point a, Point b) {
    return a.x * b.y - a.y * b.x;
}
```

#### 叉积的三种经典应用场景

1. **判断三点的转向（方向测试）**：`cross(P1, P2, P3)` 的符号。
2. **判断线段是否相交**：见 39.2 节。
3. **计算多边形面积**：Shoelace 公式（见 Chapter 40）。

> ⚠️ **常见错误**：叉积计算时忘记用 `long long`。当坐标范围在 $10^5$ 量级时，乘积就会达到 $10^{10}$，超出 32 位整数范围，导致溢出错误。务必将中间结果转换为 `long long`（C++）或直接使用 Python（Python 自动大整数）。

---

### 39.1.3 点积（Dot Product）：判断夹角与投影

点积回答的是另一个问题：**两个向量有多"对齐"？**

$$\vec{OA} \cdot \vec{OB} = a_1 \cdot b_1 + a_2 \cdot b_2 = |\vec{OA}| \cdot |\vec{OB}| \cdot \cos\theta$$

其中 $\theta$ 是两向量的夹角。

| 点积值 | 几何含义 |
|--------|---------|
| $> 0$ | 两向量夹角 $\theta < 90°$（同侧、锐角） |
| $= 0$ | 两向量垂直（$\theta = 90°$） |
| $< 0$ | 两向量夹角 $\theta > 90°$（反向侧、钝角） |

**关键应用**：判断点 $P$ 是否是线段 $AB$ 上某点到 $A$ 或 $B$ 端点的最近点（即投影是否在线段内）。

$$\text{if } \vec{AP} \cdot \vec{AB} < 0 \Rightarrow P \text{ 的投影在 } A \text{ 的外侧}$$
$$\text{if } \vec{BP} \cdot \vec{BA} < 0 \Rightarrow P \text{ 的投影在 } B \text{ 的外侧}$$

```python
def dot(O: Point, A: Point, B: Point) -> float:
    """计算向量 OA 和 OB 的点积"""
    oa = A - O
    ob = B - O
    return oa.x * ob.x + oa.y * ob.y
```

```cpp
ll dot(Point O, Point A, Point B) {
    // 向量 OA · 向量 OB
    return (ll)(A.x - O.x) * (B.x - O.x) + (ll)(A.y - O.y) * (B.y - O.y);
}
```

---

### 39.1.4 面积公式

利用叉积，可以 $O(1)$ 计算任意三角形面积：

$$S_{\triangle ABC} = \frac{1}{2}|\text{cross}(A, B, C)| = \frac{1}{2}|(\vec{AB} \times \vec{AC})|$$

这是因为叉积的绝对值等于平行四边形面积，三角形取一半。

```python
def triangle_area(A: Point, B: Point, C: Point) -> float:
    """利用叉积计算三角形 ABC 的面积（恒为正）"""
    return abs(cross(A, B, C)) / 2.0
```

```cpp
// 三角形面积（返回 double，始终 ≥ 0）
double triangle_area(Point A, Point B, Point C) {
    return abs(cross(A, B, C)) / 2.0;
}
```

---

### 39.1.5 数值精度问题与处理策略

计算几何中最棘手的问题往往不是算法本身，而是**浮点精度**。

#### 整数坐标（整数运算）— 最推荐！

如果输入点的坐标都是整数，那么叉积/点积的计算结果也是整数，可以**完全精确比较**。这是竞赛和工程中的最优选择：

- 判断共线：`cross(O, A, B) == 0`（完全精确）
- 判断左转：`cross(O, A, B) > 0`（完全精确）
- 注意：使用 `long long` 而不是 `int`！

#### 浮点坐标（引入 EPS）

当坐标是浮点数时，必须引入误差阈值 $\varepsilon$（EPS）：

```python
EPS = 1e-9

def sign(x: float) -> int:
    """带 EPS 的符号函数：避免浮点噪声误判"""
    if x > EPS:  return 1
    if x < -EPS: return -1
    return 0       # 在 EPS 范围内视为 0（共线/垂直/等于）

def eq(a: float, b: float) -> bool:
    """浮点相等判断：|a - b| < EPS"""
    return abs(a - b) < EPS
```

```cpp
const double EPS = 1e-9;

int sign(double x) {
    if (x > EPS)  return 1;
    if (x < -EPS) return -1;
    return 0;  // 视为 0
}

bool feq(double a, double b) {
    return fabs(a - b) < EPS;
}
```

> **教训**：永远不要直接用 `==` 比较两个浮点数！用 `sign()` 和 `feq()` 封装所有比较逻辑，这是计算几何代码健壮性的基本保障。

> **最佳实践**：输入坐标如果是整数（即使题目给浮点，有时可乘以公倍数化为整数），优先使用整数运算+`long long`，彻底避免精度问题。

---

## 39.2 线段交叉判断

### 39.2.1 方向测试（Direction Test）

**核心问题**：给定三点 $p_i, p_j, p_k$，$p_k$ 相对于有向直线 $\overrightarrow{p_i p_j}$ 是在左边、右边，还是在直线上？

用叉积即可回答：

$$\text{DIRECTION}(p_i, p_j, p_k) = \text{cross}(p_i, p_j, p_k) = (\vec{p_j - p_i}) \times (\vec{p_k - p_i})$$

| 结果 | 含义 |
|------|------|
| $> 0$ | $p_k$ 在有向直线 $\overrightarrow{p_i p_j}$ 的**左侧** |
| $< 0$ | $p_k$ 在有向直线 $\overrightarrow{p_i p_j}$ 的**右侧** |
| $= 0$ | $p_i, p_j, p_k$ **共线** |

```python
def direction(pi: Point, pj: Point, pk: Point) -> int:
    """
    方向测试：返回叉积值
    > 0 → pk 在 pi->pj 左侧
    < 0 → pk 在 pi->pj 右侧
    = 0 → 共线
    """
    return cross(pi, pj, pk)
```

```cpp
ll direction(Point pi, Point pj, Point pk) {
    // (pj - pi) × (pk - pi)
    return cross(pi, pj, pk);
}
```

---

### 39.2.2 线段相交的充要条件

**问题**：线段 $\overline{AB}$ 和线段 $\overline{CD}$ 是否相交？

**充要条件（一般情形，排除共线）**：

1. $C$ 和 $D$ 在直线 $AB$ 的两侧：$\text{dir}(A, B, C) \times \text{dir}(A, B, D) < 0$
2. $A$ 和 $B$ 在直线 $CD$ 的两侧：$\text{dir}(C, D, A) \times \text{dir}(C, D, B) < 0$

**直觉理解**：

想象 $\overline{AB}$ 是一堵墙。$C$ 和 $D$ 分别在墙的两侧（一左一右），说明连接 $C$ 和 $D$ 的线段 $\overline{CD}$ 必然穿过了这堵墙（即相交）。但仅这一条件不够——还需要 $\overline{CD}$ 这堵"另一面墙"把 $A$ 和 $B$ 也分隔到两侧。两个条件同时满足，才能确认两线段真正相交。

$$\text{相交} \iff \underbrace{\text{dir}(A,B,C) \cdot \text{dir}(A,B,D) < 0}_{\text{C,D 在 AB 两侧}} \quad \text{AND} \quad \underbrace{\text{dir}(C,D,A) \cdot \text{dir}(C,D,B) < 0}_{\text{A,B 在 CD 两侧}}$$

---

### 39.2.3 退化情形：共线与端点重叠

当某个叉积恰好为 0（共线），需要特殊处理：端点是否落在另一条线段上？

用 `ON-SEGMENT` 函数判断：若三点共线，判断 $p_k$ 的坐标是否在 $p_i$ 和 $p_j$ 的范围内（即是否在线段上则需x,y坐标均处于区间内）。

```python
def on_segment(pi: Point, pj: Point, pk: Point) -> bool:
    """
    前提：pi, pj, pk 已经三点共线（叉积 = 0）
    判断 pk 是否在线段 pi-pj 上（含端点）
    """
    # pk 的 x 坐标需在 pi.x 和 pj.x 之间（inclusive）
    # pk 的 y 坐标需在 pi.y 和 pj.y 之间（inclusive）
    return (min(pi.x, pj.x) <= pk.x <= max(pi.x, pj.x) and
            min(pi.y, pj.y) <= pk.y <= max(pi.y, pj.y))
```

```cpp
bool on_segment(Point pi, Point pj, Point pk) {
    // 前提：三点共线
    // 判断 pk 是否在线段 pi-pj 上（含端点）
    return min(pi.x, pj.x) <= pk.x && pk.x <= max(pi.x, pj.x)
        && min(pi.y, pj.y) <= pk.y && pk.y <= max(pi.y, pj.y);
}
```

---

### 39.2.4 SEGMENTS-INTERSECT 完整实现

综合上述条件，实现完整的线段相交判断：

```python
def segments_intersect(A: Point, B: Point, C: Point, D: Point) -> bool:
    """
    判断线段 AB 和线段 CD 是否相交（包括端点相交和共线重叠）。
    时间复杂度：O(1)

    算法思路：
    1. 正常情形（非共线）：用叉积乘积符号判断
    2. 退化情形（某叉积=0，共线）：用 on_segment 判断端点是否在对方线段上
    """
    d1 = direction(C, D, A)  # A 相对于直线 CD 的方向
    d2 = direction(C, D, B)  # B 相对于直线 CD 的方向
    d3 = direction(A, B, C)  # C 相对于直线 AB 的方向
    d4 = direction(A, B, D)  # D 相对于直线 AB 的方向

    # 正常相交：A、B 在 CD 两侧，且 C、D 在 AB 两侧
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True

    # 退化：A 在直线 CD 上，且 A 在线段 CD 内
    if d1 == 0 and on_segment(C, D, A): return True
    # 退化：B 在直线 CD 上，且 B 在线段 CD 内
    if d2 == 0 and on_segment(C, D, B): return True
    # 退化：C 在直线 AB 上，且 C 在线段 AB 内
    if d3 == 0 and on_segment(A, B, C): return True
    # 退化：D 在直线 AB 上，且 D 在线段 AB 内
    if d4 == 0 and on_segment(A, B, D): return True

    return False
```

```cpp
bool segments_intersect(Point A, Point B, Point C, Point D) {
    ll d1 = direction(C, D, A);
    ll d2 = direction(C, D, B);
    ll d3 = direction(A, B, C);
    ll d4 = direction(A, B, D);

    // 正常情形：AB 与 CD 真正交叉
    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
        ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0)))
        return true;

    // 退化情形：共线端点
    if (d1 == 0 && on_segment(C, D, A)) return true;
    if (d2 == 0 && on_segment(C, D, B)) return true;
    if (d3 == 0 && on_segment(A, B, C)) return true;
    if (d4 == 0 && on_segment(A, B, D)) return true;

    return false;
}
```

**时间复杂度**：$O(1)$ — 仅 4 次叉积计算和若干比较。

**LeetCode 应用**：[#149 直线上最多的点数](https://leetcode.cn/problems/max-points-on-a-line/)（判断三点共线即叉积为 0）

<div data-component="SegmentIntersectionTest"></div>

> ⚠️ **常见错误**：忽略退化情形！只判断 `d1 * d2 < 0 && d3 * d4 < 0` 会漏掉端点正好落在线段上（如 T 形相交或端点重合）的情形。完整实现必须包含 4 个 `on_segment` 检查。

---

## 39.3 凸包（Convex Hull）

### 39.3.1 什么是凸包？

#### 通俗比喻：橡皮筋套住钉子

把一堆钉子（点）插在木板上，拉一根橡皮筋套住所有钉子，然后松手。橡皮筋会绷紧，形成一个多边形——这就是**凸包（Convex Hull）**。

> **正式定义**：点集 $P$ 的凸包是包含 $P$ 中所有点的**最小凸多边形**。凸多边形的任意两点连线都在多边形内部（即没有"凹进去"的部分）。

为什么计算凸包？
- **最远点对**：最远的两个点一定都在凸包上（否则可以替换为凸包顶点获得更大距离）
- **碰撞检测**：用凸包近似复杂形状
- **点在多边形内**：凸包可以用 $O(\log n)$ 二分判断
- **竞赛题型**：LeetCode #587 安装围栏

**凸包顶点数**：设 $n$ 个点，凸包有 $h$ 个顶点（$h \leq n$）。所有算法的下界是 $\Omega(n \log n)$（与排序等价，Chapter 39 末有思考题）。

---

### 39.3.2 Graham 扫描法：O(n log n)

Graham 扫描法是最经典的凸包算法，思路清晰：**先排序，再扫描**。

**算法步骤**：

**Step 1**：找基准点 $P_0$——选 $y$ 坐标最小的点（相同 $y$ 则选 $x$ 最小的），保证 $P_0$ 一定在凸包上（最低左点）。

**Step 2**：以 $P_0$ 为原点，对其他点按**极角**排序（$P_0$ 到该点的方向角）。极角相同时按距离升序排列。

$$\theta_i = \text{atan2}(P_i.y - P_0.y,\ P_i.x - P_0.x)$$

**Step 3**：用一个**栈**维护当前凸包的候选点，遍历排好序的点列：
- 若当前点 $P_k$ 使得栈顶两点构成**右转**（叉积 $\leq 0$），则弹出栈顶（该点不在凸包上）。
- 重复弹出，直到栈中最近两点到 $P_k$ 是**左转**（叉积 $> 0$）。
- 将 $P_k$ 压栈。

**结束**：栈中从底到顶就是凸包顶点（逆时针顺序）。

**为什么弹出右转点？** 凸包要求所有顶点处是左转（逆时针）。如果出现右转，说明该点在凸包内侧，不是凸包顶点。

<div data-component="GrahamScanAnimation"></div>

```python
def graham_scan(points: list) -> list:
    """
    Graham 扫描法求凸包。
    返回：凸包顶点列表，逆时针顺序（不含重复的起始点）。
    注意：若多点共线，此实现不保留共线点（严格凸包）。
          若需要保留共线点（如 LeetCode #587），需调整比较逻辑。
    """
    n = len(points)
    if n <= 1:
        return points

    # Step 1：找基准点 P0（y 最小，y 相同则 x 最小）
    p0 = min(points, key=lambda p: (p.y, p.x))

    def cmp_key(p):
        """极角排序的 key：用叉积替代 atan2，精度更高"""
        if p == p0:
            return (0, 0)  # p0 自己排最前
        # 叉积方向（逆时针为正）；距离作为次关键字
        c = cross(p0, p, Point(p0.x + 1, p0.y))  # 相对于 x 轴正方向的方向
        return (-cross(p0, p, p), (p.x - p0.x)**2 + (p.y - p0.y)**2)

    # 实际竞赛中常用：以叉积比较代替 atan2（避免浮点）
    def polar_cmp(p, q):
        """
        比较 p0->p 和 p0->q 的极角大小
        叉积 > 0 → p 在 q 的右侧（极角更小）
        """
        c = cross(p0, p, q)
        if c != 0:
            return -1 if c > 0 else 1  # c>0: p 极角 < q 极角
        # 极角相同：按距离排序（近的先处理）
        dp = (p.x - p0.x)**2 + (p.y - p0.y)**2
        dq = (q.x - p0.x)**2 + (q.y - p0.y)**2
        return -1 if dp < dq else 1

    import functools
    pts = [p for p in points if p != p0]
    pts.sort(key=functools.cmp_to_key(polar_cmp))
    pts = [p0] + pts

    # Step 3：扫描 + 维护凸包栈
    stack = []
    for p in pts:
        # 弹出不构成左转的栈顶元素
        # cross(倒数第二, 栈顶, p) <= 0 说明右转或共线
        while len(stack) >= 2 and cross(stack[-2], stack[-1], p) <= 0:
            stack.pop()
        stack.append(p)

    return stack
```

```cpp
// Graham 扫描法（整数坐标，long long 精度）
// 注意：cross <= 0 时弹出（严格凸包，不含共线点）
// 若需含共线点（LeetCode #587），改为 cross < 0

vector<Point> graham_scan(vector<Point> pts) {
    int n = pts.size();
    if (n <= 2) return pts;

    // Step 1：找基准点（y 最小，y 相同取 x 最小）
    int p0_idx = 0;
    for (int i = 1; i < n; i++)
        if (pts[i].y < pts[p0_idx].y ||
           (pts[i].y == pts[p0_idx].y && pts[i].x < pts[p0_idx].x))
            p0_idx = i;
    swap(pts[0], pts[p0_idx]);
    Point p0 = pts[0];

    // Step 2：极角排序（以 p0 为原点，用叉积比较，避免 atan2 浮点）
    sort(pts.begin() + 1, pts.end(), [&](const Point& a, const Point& b) {
        ll c = cross(p0, a, b);
        if (c != 0) return c > 0;  // c > 0 → a 的极角更小，排在前面
        // 极角相同：按距离排序（近的先）
        ll da = (a.x-p0.x)*(a.x-p0.x) + (a.y-p0.y)*(a.y-p0.y);
        ll db = (b.x-p0.x)*(b.x-p0.x) + (b.y-p0.y)*(b.y-p0.y);
        return da < db;
    });

    // Step 3：扫描建栈
    vector<Point> hull;
    for (int i = 0; i < n; i++) {
        // cross <= 0 → 右转或共线，弹出（严格凸包）
        while (hull.size() >= 2 &&
               cross(hull[hull.size()-2], hull.back(), pts[i]) <= 0)
            hull.pop_back();
        hull.push_back(pts[i]);
    }
    return hull;
}
```

**时间复杂度分析**：
- Step 1 找最低点：$O(n)$
- Step 2 极角排序：$O(n \log n)$（主导）
- Step 3 遍历扫描：$O(n)$（每个点至多入栈一次、出栈一次）
- **总计：$O(n \log n)$**

**空间复杂度**：$O(n)$（栈最大 $n$ 个元素）

---

### 39.3.3 Jarvis March（礼品包装法）：O(nh)

Jarvis March 的灵感来自**包礼品**：找到一张包装纸，从最左点开始，不断往最外侧延伸，直到绕一圈。

**算法步骤**：
1. 从最左（最低）点 $p_0$ 出发。
2. 每次找到相对于当前点极角最小的点（即使用叉积，找到"最右转"的点），作为凸包下一个顶点。
3. 重复直到回到起点。

**时间复杂度**：每轮找下一个点需要 $O(n)$，共 $h$ 轮。总计 $O(nh)$。

```python
def jarvis_march(points: list) -> list:
    """
    礼品包装法（Jarvis March）求凸包。
    时间复杂度：O(n * h)，h 为凸包顶点数。
    适用场景：h 很小时（如 h = O(log n)）优于 Graham，否则逊色。
    """
    n = len(points)
    if n <= 1:
        return points

    # 起点：最左点（x 最小，相同则 y 最小）
    start = min(range(n), key=lambda i: (points[i].x, points[i].y))

    hull = []
    cur = start
    while True:
        hull.append(points[cur])
        # 找到极角最逆时针的下一个点
        nxt = (cur + 1) % n
        for i in range(n):
            c = cross(points[cur], points[nxt], points[i])
            if c > 0:
                # points[i] 在 cur->nxt 的左侧，更逆时针
                nxt = i
            elif c == 0:
                # 共线：选距离更远的点（保留共线点时用，否则忽略）
                pass
        cur = nxt
        if cur == start:  # 回到起点，凸包完成
            break

    return hull
```

```cpp
vector<Point> jarvis_march(vector<Point>& pts) {
    int n = pts.size();
    if (n < 3) return pts;

    // 起点：最左下点
    int start = 0;
    for (int i = 1; i < n; i++)
        if (pts[i].x < pts[start].x ||
           (pts[i].x == pts[start].x && pts[i].y < pts[start].y))
            start = i;

    vector<Point> hull;
    int cur = start;
    do {
        hull.push_back(pts[cur]);
        int nxt = (cur + 1) % n;
        for (int i = 0; i < n; i++) {
            // 若 pts[i] 在 cur->nxt 的左侧（逆时针），更新 nxt
            if (cross(pts[cur], pts[nxt], pts[i]) > 0)
                nxt = i;
        }
        cur = nxt;
    } while (cur != start);

    return hull;
}
```

> **注意**：Jarvis March 中的 `cross > 0` 取的是"最逆时针"的方向；而 Graham 扫描中 `cross <= 0` 是弹出"右转"点——两者逻辑相反，容易写错，请仔细区分。

---

### 39.3.4 两算法的对比与选择

<div data-component="ConvexHullCompare"></div>

| 维度 | Graham 扫描 | Jarvis March |
|------|-------------|--------------|
| 时间复杂度 | $O(n \log n)$ | $O(nh)$ |
| $h = \Theta(n)$ 时 | $O(n \log n)$ ✅ | $O(n^2)$ ❌ |
| $h = O(1)$ 时 | $O(n \log n)$ | $O(n)$ ✅ |
| 实现难度 | 中等（极角排序较复杂） | 简单（每轮一次遍历）  |
| 数值稳定性 | 需处理极角相同的共线点 | 相对简单 |
| 适用场景 | 通用首选；$h$ 未知时用 | 已知 $h$ 远小于 $n$ |

**实际选择**：在不知道 $h$ 的情况下，**优先选择 Graham 扫描**，因为 $O(n \log n)$ 是有保证的最坏情形下界。

---

### 39.3.5 Andrew's Monotone Chain（单调链）

Andrew's 单调链是 Graham 的一个变种，**实现更简洁**，避免了极角排序的复杂性，用 `x` 坐标排序替代：

**算法**：
1. 将点按 $x$ 坐标排序（$x$ 相同按 $y$）
2. **构建下凸包**：从左到右扫描，维护逆时针序列（遇到右转则弹出）
3. **构建上凸包**：从右到左扫描，同样维护逆时针序列
4. 合并上下凸包（去除重复的两端点）

<div data-component="AndrewMonotoneChain"></div>

```python
def andrew_monotone_chain(points: list) -> list:
    """
    Andrew's 单调链法求凸包。
    时间复杂度：O(n log n)（排序主导）
    优点：实现简单，不需要极角排序，只需普通坐标排序。
    返回：凸包顶点，逆时针顺序（从最左下点开始）
    """
    pts = sorted(set((p.x, p.y) for p in points))  # 去重 + 排序
    pts = [Point(x, y) for x, y in pts]
    n = len(pts)
    if n <= 1:
        return pts

    # 构建下凸包（从左到右）
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # 构建上凸包（从右到左）
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # 合并：去除两端重复点（lower 的最后一个 = upper 的第一个，反之亦然）
    return lower[:-1] + upper[:-1]
```

```cpp
vector<Point> monotone_chain(vector<Point> pts) {
    int n = pts.size();
    // 按 x 坐标排序（x 相同按 y）
    sort(pts.begin(), pts.end());
    pts.erase(unique(pts.begin(), pts.end()), pts.end());
    n = pts.size();
    if (n < 2) return pts;

    vector<Point> hull;

    // 构建下凸包（从左到右）
    for (int i = 0; i < n; i++) {
        while (hull.size() >= 2 &&
               cross(hull[hull.size()-2], hull.back(), pts[i]) <= 0)
            hull.pop_back();
        hull.push_back(pts[i]);
    }

    // 构建上凸包（从右到左），跳过已加入下凸包的两端点
    int lower_size = hull.size() + 1;  // 下凸包结束位置
    for (int i = n - 2; i >= 0; i--) {
        while ((int)hull.size() >= lower_size &&
               cross(hull[hull.size()-2], hull.back(), pts[i]) <= 0)
            hull.pop_back();
        hull.push_back(pts[i]);
    }

    hull.pop_back();  // 去掉重复的起始点（它被上下凸包各加了一次）
    return hull;
}
```

**LeetCode #587 安装围栏**：这道题的本质就是求凸包，但要求**保留共线点**。只需将 `cross <= 0` 改为 `cross < 0`（即共线不弹出），并特别处理单点/双点情形。

---

### 39.3.6 凸包的进阶应用

#### 旋转卡壳（Rotating Calipers）：最远点对 O(n)

给定凸包（$h$ 个顶点）后，求凸包上最远的两个点（即直径），可以用旋转卡壳在 $O(h)$ 时间内完成：

1. 凸包最远点对一定是一对**对径点**（antipodal pair）
2. 用双指针 $i, j$ 维护"卡尺"，每次移动显然增大距离的那侧
3. 一共旋转一圈，$O(h)$ 比较

结合凸包构建 $O(n \log n)$，整体为 $O(n \log n)$。

#### 点在凸多边形内：O(log n)

将凸包按顶点 $p_0$ 分成若干扇形（$p_0 p_1 p_2, p_0 p_2 p_3, \ldots$），对查询点 $q$：
1. 二分确定 $q$ 在哪个扇形中
2. 用叉积验证 $q$ 是否在该三角形内（$O(1)$）

整体 $O(\log h) = O(\log n)$。

---

## 39.4 最近点对（Closest Pair of Points）

### 39.4.1 问题描述与暴力解法

**问题**：给定平面上 $n$ 个点，找出距离最近的两个点，返回它们之间的距离。

**暴力枚举**：枚举所有 $\binom{n}{2}$ 对点，求距离，取最小。时间复杂度 $O(n^2)$，当 $n = 10^5$ 时需要 $10^{10}$ 次运算，显然太慢。

**目标**：$O(n \log n)$ 的分治算法。

---

### 39.4.2 分治算法框架

**核心思路**：分治（Divide and Conquer）。

**预处理**：将所有点按 $x$ 坐标排序（只排一次！）。

**递归结构**：
1. **分（Divide）**：以 $x$ 中位数将点集分为左半 $L$、右半 $R$，各 $n/2$ 个点。
2. **治（Conquer）**：递归求 $L$ 的最近点对距离 $d_L$ 和 $R$ 的最近点对距离 $d_R$，取 $d = \min(d_L, d_R)$。
3. **合并（Combine）**：检查是否存在跨越中线、距离更小的点对。

**关键洞察**：跨越中线的最近点对，两点必须都在距中线 $d$ 距离的带状区域 $S$ 内（否则距离 $\geq d$）。

$$S = \{p \mid |p.x - \text{mid}| \leq d\}$$

<div data-component="ClosestPairDivide"></div>

---

### 39.4.3 带状区域的 7 点引理

**为什么带状区域内只需比较 7 个点？**

这就是著名的 **7 点引理（7-point bound）**：

**引理**：设 $d$ 是当前已知最近距离。在带状区域 $S$ 中，将点按 $y$ 坐标排序为 $q_1, q_2, \ldots$，对任意点 $q_i$，只需要与之后的 **最多 7 个点**（$q_{i+1}, \ldots, q_{i+7}$）比较距离即可找到潜在的最近点对。

**几何证明**：画一个以分界线为中轴线、高度为 $d$、宽度为 $2d$ 的矩形，分成左右各 $d \times d$ 的正方形，每个正方形可以被再分为 4 个 $d/2 \times d/2$ 的小方格。

由于 $L$ 内任意两点距离 $\geq d$（递归保证），同一 $d/2 \times d/2$ 方格内至多有 $2$ 个点（极端情形是一个角的两个端点，距离 $= d/2\sqrt{2} < d$，但这样两个点实际上不会在已知 $d$ 最优解的带状区域里共存）——实际上每个 $d/2 \times d/2$ 格子内同侧至多 $1$ 个点。

右侧 $d \times d$ 正方形分 4 格，每格至多 $1$ 点 → 右侧至多 $4$ 点。左侧同理至多 $4$ 点。去除 $q_i$ 本身，$q_i$ 右侧的格子最多有 $4 + 4 - 1 = 7$ 个点需要检查。

<div data-component="SevenPointLemma"></div>

---

### 39.4.4 算法实现（完整细节）

实现细节的关键：合并步骤中需要**按 $y$ 坐标排好序**的带状区域点列。

有两种方式：
- **方式 A（简单）**：每次合并时从带状区域重新排序。$O(n \log n)$ 总时间但递推为 $T(n) = 2T(n/2) + O(n \log n)$，实际是 $O(n \log^2 n)$。
- **方式 B（最优）**：预先按 $y$ 排序好，合并时用归并保持 $y$ 顺序。$T(n) = 2T(n/2) + O(n) = O(n \log n)$。

```python
from math import inf, sqrt

def dist_sq(a: Point, b: Point) -> int:
    """欧氏距离的平方（避免 sqrt，整数精确比较）"""
    return (a.x - b.x) ** 2 + (a.y - b.y) ** 2

def closest_pair(points: list) -> float:
    """
    最近点对分治算法。
    返回最近两点的欧氏距离（float）。
    时间复杂度：O(n log n)（若合并时用简单排序，实为 O(n log^2 n)）
    """
    def rec(pts_by_x: list, pts_by_y: list) -> float:
        """
        pts_by_x：当前点集，按 x 坐标排序
        pts_by_y：当前点集，按 y 坐标排序
        """
        n = len(pts_by_x)
        # 基准情形：点数 ≤ 3，直接暴力
        if n <= 3:
            best = inf
            for i in range(n):
                for j in range(i + 1, n):
                    d = sqrt(dist_sq(pts_by_x[i], pts_by_x[j]))
                    best = min(best, d)
            return best

        mid = n // 2
        mid_x = pts_by_x[mid].x  # 中线的 x 坐标

        # 分：将 pts_by_y 也分成左右两份（保持 y 顺序！）
        left_by_x  = pts_by_x[:mid]
        right_by_x = pts_by_x[mid:]
        left_by_y  = [p for p in pts_by_y if p.x <= mid_x or
                      (p.x == mid_x and p in set(id(q) for q in left_by_x))]
        # 简化版：重新排序（O(n log^2 n) 但代码简洁）
        left_by_y  = sorted(left_by_x,  key=lambda p: p.y)
        right_by_y = sorted(right_by_x, key=lambda p: p.y)

        # 治：递归
        d_l = rec(left_by_x,  left_by_y)
        d_r = rec(right_by_x, right_by_y)
        d = min(d_l, d_r)

        # 合并：提取带状区域（宽 2d）
        strip = [p for p in pts_by_y if abs(p.x - mid_x) < d]
        # 在 strip 中检查每个点与后 7 个点的距离
        for i in range(len(strip)):
            for j in range(i + 1, min(i + 8, len(strip))):
                if strip[j].y - strip[i].y >= d:
                    break  # y 差已 ≥ d，不可能更近
                d = min(d, sqrt(dist_sq(strip[i], strip[j])))

        return d

    # 预排序（全局只做一次）
    by_x = sorted(points, key=lambda p: (p.x, p.y))
    by_y = sorted(points, key=lambda p: (p.y, p.x))
    return rec(by_x, by_y)
```

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
struct Point {
    ll x, y;
    bool operator<(const Point& o) const {
        return x != o.x ? x < o.x : y < o.y;
    }
};

ll dist_sq(Point a, Point b) {
    return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}

// 带状区域扫描器：pts 已按 y 排序，mid_x 为中线，d 为当前最近平方距离
ll strip_closest(vector<Point>& strip, ll d) {
    for (int i = 0; i < (int)strip.size(); i++) {
        for (int j = i + 1; j < (int)strip.size(); j++) {
            ll dy = strip[j].y - strip[i].y;
            if (dy * dy >= d) break;  // y 差过大，提前终止（7点引理保证最多循环7次）
            d = min(d, dist_sq(strip[i], strip[j]));
        }
    }
    return d;
}

// 分治主体（pts_x 按 x 排序）
ll closest_rec(vector<Point>& pts, int l, int r) {
    if (r - l <= 3) {
        // 基础情形：暴力
        ll best = LLONG_MAX;
        for (int i = l; i < r; i++)
            for (int j = i + 1; j < r; j++)
                best = min(best, dist_sq(pts[i], pts[j]));
        sort(pts.begin() + l, pts.begin() + r,
             [](const Point& a, const Point& b){ return a.y < b.y; });
        return best;
    }

    int mid = (l + r) / 2;
    ll mid_x = pts[mid].x;

    ll d = min(closest_rec(pts, l, mid),
               closest_rec(pts, mid, r));

    // 合并：归并按 y 排序（关键！保持 O(n log n)）
    inplace_merge(pts.begin() + l, pts.begin() + mid, pts.begin() + r,
                  [](const Point& a, const Point& b){ return a.y < b.y; });

    // 提取带状区域（宽 2*sqrt(d)）
    vector<Point> strip;
    for (int i = l; i < r; i++)
        if ((pts[i].x - mid_x) * (pts[i].x - mid_x) < d)
            strip.push_back(pts[i]);

    return strip_closest(strip, d);
}

double closest_pair(vector<Point>& pts) {
    sort(pts.begin(), pts.end());  // 按 x 排序
    ll d_sq = closest_rec(pts, 0, pts.size());
    return sqrt((double)d_sq);
}
```

---

### 39.4.5 时间复杂度完整分析

递推关系（使用 inplace_merge 归并保持 y 顺序）：

$$T(n) = 2T\!\left(\frac{n}{2}\right) + O(n)$$

由主定理（$a=2, b=2, f(n)=O(n)$，情形 II / III）：

$$T(n) = O(n \log n)$$

各步骤分解：
| 步骤 | 时间 |
|------|------|
| 预排序（x 坐标） | $O(n \log n)$（全局一次） |
| 递归求左右半 | $2T(n/2)$ |
| 归并保持 y 顺序 | $O(n)$（每层） |
| 带状区域检查（7点引理） | $O(n)$（每层，最多 7 次比较/点） |
| **总计** | $\boldsymbol{O(n \log n)}$ |

若合并时重新排序（naive 方式）：每层额外 $O(n \log n)$，总计 $O(n \log^2 n)$。

---

### 39.4.6 实现中的常见陷阱

> ⚠️ **陷阱 1：带状区域未按 y 排序**  
> 如果带状区域内的点未按 $y$ 坐标排序，7 点引理的早停条件（`y 差 >= d`）就无法生效，最坏情形退化为 $O(n^2)$。  
> **解决方案**：分治时用归并维护 $y$ 排序，或在合并前对带状区域重新排序（接受 $O(n \log^2 n)$）。

> ⚠️ **陷阱 2：用 sqrt 比距离导致浮点误差**  
> 在递归内部频繁调用 `sqrt` 损耗性能且引入精度问题。  
> **解决方案**：全程用**距离平方**比较，只在最后返回结果时调用一次 `sqrt`。

> ⚠️ **陷阱 3：基准情形点数设置过小**  
> 若 $n=2$ 时继续分治可能产生大量小递归，效率低。  
> **解决方案**：$n \leq 3$ 时直接暴力（最多 3 次比较）。

---

## 39.5 其他几何算法（概述）

### 39.5.1 扫描线算法（Sweep Line）

**核心思想**：想象一条竖线从左向右"扫过"平面，每当遇到事件（线段起点/终点/交点）时触发处理。  
借助**优先队列**（事件堆）和**平衡BST**（维护当前活跃线段的 $y$ 顺序），将 $O(n^2)$ 问题降到 $O(n \log n)$。

**经典问题**：
- **判断 $n$ 条线段是否有交**：Shamos-Hoey 算法，$O(n \log n)$
- **区间重叠计数**：事件为区间端点，用扫描线 $O(n \log n)$

<div data-component="SweepLineDemo"></div>

**事件类型**：
1. **插入事件**（线段左端点）：将线段插入活跃集，检查与上下邻居的交叉。
2. **删除事件**（线段右端点）：从活跃集删除线段，检查其上下邻居是否交叉。
3. **交叉事件**（线段交点）：在交叉处交换两线段顺序，检查新邻居关系。

```python
from sortedcontainers import SortedList
import heapq

def shamos_hoey(segments: list) -> bool:
    """
    Shamos-Hoey 算法：判断 n 条线段中是否存在任意交叉。
    时间复杂度：O(n log n)
    参数：segments = [(A, B), (C, D), ...] 表示线段端点对

    【注意】此为简化示意版本：完整实现需精确处理数值精度和端点共线情形。
    """
    events = []
    for i, (a, b) in enumerate(segments):
        # 确保左端点在前
        if a.x > b.x or (a.x == b.x and a.y > b.y):
            a, b = b, a
        # 左端点事件（优先处理）
        heapq.heappush(events, (a.x, 0, i, a, b))  # type=0: 插入
        # 右端点事件
        heapq.heappush(events, (b.x, 1, i, a, b))  # type=1: 删除

    active = SortedList(key=lambda seg: seg[0].y)  # 按 y 坐标维护活跃线段

    while events:
        x, etype, idx, a, b = heapq.heappop(events)

        if etype == 0:  # 插入线段
            active.add((a, b))
            pos = active.index((a, b))
            # 检查与上下邻居的交叉
            if pos > 0 and segments_intersect(*active[pos-1], a, b):
                return True
            if pos < len(active) - 1 and segments_intersect(a, b, *active[pos+1]):
                return True
        else:           # 删除线段
            pos = active.index((a, b))
            # 删除前检查其上下邻居是否相交
            if (0 < pos < len(active) - 1 and
                    segments_intersect(*active[pos-1], *active[pos+1])):
                return True
            active.remove((a, b))

    return False
```

---

### 39.5.2 Shamos-Hoey 算法核心原理

Shamos-Hoey 的关键洞察：**如果存在交叉，那么在某个扫描线位置，会出现两条相邻线段（在活跃集BST中）恰好交叉的情形**。

因此，只需在每次插入/删除时检查与新邻居的交叉即可，不需要枚举所有对。这将 $n$ 条线段的 $O(n^2)$ 暴力检查降低到 $O(n \log n)$。

若需枚举**所有交点**（而非仅判断存在性），需要用更复杂的 **Bentley-Ottmann 算法**：

- 时间复杂度：$O((n + k) \log n)$，$k$ 为总交点数
- 核心：交叉事件也加入优先队列；交叉时交换两线段的顺序并检查新邻居
- 最坏情形（所有 $\binom{n}{2}$ 对都相交）：$O(n^2 \log n)$

---

### 39.5.3 Voronoi 图（沃罗诺伊图）

**定义**：给定平面上 $n$ 个"站点"（site），Voronoi 图将平面划分为 $n$ 个区域，每个区域包含距对应站点最近的所有点。

$$\text{Voronoi}(s_i) = \{p \mid \text{dist}(p, s_i) \leq \text{dist}(p, s_j), \forall j \neq i\}$$

**构建算法**：Fortune's Sweep Line 算法，$O(n \log n)$：
- 维护"抛物线海滩线"（Parabolic Beach Line）和事件队列
- 每遇到站点事件（site event）：新增弧；每遇到圆事件（circle event）：删除弧

**应用**：最近邻近似（ANN）预处理、GIS 区域划分、机器人路径规划、聚类分析

---

### 39.5.4 Delaunay 三角剖分

**定义**：Voronoi 图的**对偶图**（dual graph）。连接每对共享 Voronoi 边的站点，形成三角剖分。

**核心性质**：所有三角形中，**最小角最大化**（maximizes the minimum angle），避免了狭长三角形，是有限元分析、网格生成的理想基础。

**外接圆条件**：任意三角形的外接圆内**不包含**其他点（Empty Circumcircle Property）。

**构建复杂度**：$O(n \log n)$（Fortune 算法及其对偶变换）

**应用**：
- 图像插值与网格生成（FEM 有限元）
- 人脸关键点三角网格（图形学变形）
- 点云处理与 3D 重建

---

### 39.5.5 计算几何在工程中的应用总结

| 应用领域 | 使用算法 | 典型场景 |
|---------|---------|---------|
| 游戏开发 | 凸包、GJK 算法 | 碰撞检测（两凸体是否重叠）|
| 地图 / GIS | 点在多边形内（射线法） | 判断坐标是否在某区域内 |
| 自动驾驶 | Voronoi 图、凸包 | 障碍物描述、路径规划 |
| 3D 打印 / CAD | Delaunay 三角剖分 | 网格细化、切片算法 |
| 竞赛算法 | 凸包 + 旋转卡壳 | 最远点对、最小覆盖圆 |
| 图像处理 | 凸包、最近点对 | 轮廓检测、关键点匹配 |

---

## 39.6 经典题型与练习

### 39.6.1 LeetCode 精选

| 题目 | 核心知识点 | 难度 |
|------|-----------|------|
| #149 直线上最多的点数 | 叉积判断共线（精度处理） | Hard |
| #587 安装围栏 | 凸包（含共线点的变种） | Hard |
| #1956 感染了蛋糕的牧场 | 最近点对变形 | Hard |
| #939 最小面积矩形 | 哈希 + 几何枚举 | Medium |
| #223 矩形面积 | 基础几何判断 | Medium |

### 39.6.2 思考题

> 💡 **思考题 1**：凸包问题有 $\Omega(n \log n)$ 的**下界**（即不存在 $o(n \log n)$ 的算法），能够证明为什么吗？
>
> **提示**：构造一个从排序问题到凸包问题的**归约**（reduction）：给定 $n$ 个实数 $x_1, \ldots, x_n$，构造点集 $P = \{(x_i, x_i^2)\}$（所有点都在抛物线 $y = x^2$ 上），则所有点都在凸包上，且凸包顶点的逆时针顺序就是 $x_i$ 的排序结果。若存在 $O(f(n))$ 的凸包算法，则可以 $O(f(n))$ 完成排序，与排序下界 $\Omega(n \log n)$ 矛盾。

> 💡 **思考题 2**：如果给定点集已经按 $x$ 坐标排序，能否将凸包计算降低到 $O(n)$？
>
> **答**：可以！Andrew's 单调链的排序步骤 $O(n \log n)$ 是主导。如果点已按 $x$ 排序，则 Step 2 和 Step 3 各只需 $O(n)$，**整体降为 $O(n)$**。

> 💡 **思考题 3**：最近点对算法中，若用 `dist_sq`（距离平方）代替 `dist`（欧氏距离）做比较，整个算法能避免浮点运算吗？整数坐标下如何保证正确性？

---

## 39.7 常见错误与调试建议

| 错误类型 | 描述 | 解决方案 |
|---------|------|---------|
| 叉积溢出 | C++ 中坐标 $\leq 10^5$ 时乘积可达 $10^{10}$，超出 `int` | 改用 `long long` 存储叉积结果 |
| 直接比较浮点 | `cross(A,B,C) == 0` 浮点坐标不可靠 | 使用 EPS 的 `sign()` 函数 |
| Graham 极角共线未排序 | 多点共线时极角排序后弃掉了需要保留的点 | 共线时按距离排序，明确是否需要含共线点 |
| 最近点对未保持 $y$ 排序 | 带状区域未排序导致 7 点引理失效，$O(n^2)$ 退化 | 归并时维护 $y$ 排序，或合并时重排（接受 $O(n \log^2 n)$）|
| Jarvis 死循环 | 共线多点未处理，`nxt` 不更新回到起点 | 明确共线时的选择策略（一般取最远点）|
| 凸包点数为 1/2 | 未处理退化输入（所有点共线 / 单点） | 在 `n <= 2` 时提前返回 |

---

## 39.8 参考资料

- **CLRS 第4版** Chapter 33：Computational Geometry（中文版第三章）
- **Skiena 第2版** Chapter 17：Computational Geometry
- **Preparata & Shamos**：《Computational Geometry: An Introduction》（经典参考书）
- **cp-algorithms.com**：[Geometry](https://cp-algorithms.com/geometry/basic-geometry.html)（英文竞赛模板）
- **OI-wiki**：[计算几何基础](https://oi-wiki.org/geometry/2d/)（中文竞赛参考）
- **LeetCode**：#587（安装围栏）、#149（直线上最多的点数）

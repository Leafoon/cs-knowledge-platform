---
title: "Chapter 40: 计算几何进阶与综合"
description: "Shoelace 多边形面积公式、射线法/绕数法点包含判断、旋转卡壳最远点对、Shamos-Hoey 扫描线求交、SAT 碰撞检测——计算几何的工程级综合实战"
tags: ["computational-geometry", "shoelace", "point-in-polygon", "rotating-calipers", "sweep-line", "SAT", "convex-polygon"]
difficulty: "hard"
updated: "2026-03-12"
---

# Chapter 40: 计算几何进阶与综合

> **Part XI · 计算几何进阶**

在 Chapter 39 中，我们奠定了计算几何的基石：叉积原语、线段交叉判断、凸包算法与最近点对分治。本章在此基础上向前一步，聚焦于几何算法在**工程与竞赛**中最常遇到的四类进阶问题：

| 问题类别 | 典型问题 | 核心技术 |
|---------|---------|---------|
| 多边形度量 | 计算任意多边形面积 | Shoelace 公式 $O(n)$ |
| 点包含判断 | 点是否在多边形内？ | 射线法 / 绕数法 |
| 线段交 | n 条线段是否有交？所有交点？ | Shamos-Hoey / Bentley-Ottmann |
| 凸包直径 | 多边形最远点对？最小外接矩形？ | 旋转卡壳 $O(n)$ |

这四类问题在 **GIS 地图分析、游戏碰撞检测、机器人路径规划、计算机视觉** 中无处不在。让我们逐一深入。

---

## 40.1 多边形面积：Shoelace 公式

### 40.1.1 直觉建立：从三角形到多边形

**计算三角形面积** 你一定会：底 × 高 ÷ 2。但如果三角形三个顶点全是坐标，没有明显的"底"和"高"怎么办？

回忆 Chapter 39 的叉积：对于以原点为顶点、$A$ 和 $B$ 为另两顶点的三角形，其**有向面积**正好是：

$$S_{\triangle OAB} = \frac{1}{2} \times \text{cross}(A, B) = \frac{1}{2}(A.x \cdot B.y - A.y \cdot B.x)$$

这个公式天才般地把"面积"和"叉积"联系起来了。符号为正表示 $O \to A \to B$ 是逆时针，为负则是顺时针。

**推广到多边形**：把任意多边形切割成若干个以**原点 $O$** 为顶点的三角形，每个三角形的有向面积（带符号）加起来，恰好等于多边形的有向总面积！多余的区域会正负相消，神奇地自动处理凹多边形。

$$S = \frac{1}{2} \left| \sum_{i=0}^{n-1} (x_i \cdot y_{i+1} - x_{i+1} \cdot y_i) \right|$$

其中下标对 $n$ 取模（最后一条边回到第一个顶点）。这就是鼎鼎大名的 **Shoelace 公式**（也叫高斯面积公式，或 Surveyor's Formula，测量师公式）。

> **为什么叫 Shoelace（鞋带）公式？**  
> 把顶点坐标写成两列：
> $$\begin{pmatrix} x_0 & y_0 \\ x_1 & y_1 \\ \vdots \\ x_{n-1} & y_{n-1} \end{pmatrix}$$
> 计算时像交叉系鞋带一样，正对角线乘积之和减去负对角线乘积之和，视觉上恰好是交叉的"鞋带"形状。

<div data-component="ShoelacePolygonArea"></div>

### 40.1.2 公式推导的完整过程

设多边形顶点按序为 $P_0, P_1, \ldots, P_{n-1}$（顺时针或逆时针均可）。选定平面上任意一点 $O$（通常取原点），将多边形分解为 $n$ 个三角形：$\triangle O P_0 P_1, \triangle O P_1 P_2, \ldots, \triangle O P_{n-1} P_0$。

每个三角形的**有向面积**（Signed Area）：

$$S_{\triangle O P_i P_{i+1}} = \frac{1}{2}(x_i \cdot y_{i+1} - x_{i+1} \cdot y_i)$$

多边形有向总面积：

$$S = \sum_{i=0}^{n-1} S_{\triangle O P_i P_{i+1}} = \frac{1}{2} \sum_{i=0}^{n-1} (x_i \cdot y_{i+1} - x_{i+1} \cdot y_i)$$

- 若顶点为**逆时针（CCW）**排列，$S > 0$；
- 若为**顺时针（CW）**排列，$S < 0$；
- 取绝对值得到实际面积。

**凹多边形也适用**：凹入部分对应的三角形符号相反，自然相消，无需额外处理。

<div data-component="PolygonOrientationDemo"></div>

### 40.1.3 代码实现

```python
from typing import List, Tuple

# 用 (x, y) 元组表示点，也可配合 Chapter 39 的 Point 类
Polygon = List[Tuple[float, float]]

def polygon_area(pts: Polygon) -> float:
    """
    Shoelace 公式计算多边形面积
    
    参数：
        pts: 多边形顶点列表（顺时针或逆时针均可，但必须按序）
    返回：
        浮点数面积（始终为正值）
    
    时间复杂度：O(n)
    空间复杂度：O(1)
    
    【边界条件】
    - pts 少于 3 个点：面积为 0
    - 顶点不需要显式重复第一个点（函数自动闭合）
    """
    n = len(pts)
    if n < 3:
        return 0.0
    
    signed_area = 0.0
    for i in range(n):
        j = (i + 1) % n  # 下一个顶点（自动闭合）
        # 每一步累加 (x_i * y_{i+1} - x_{i+1} * y_i)
        signed_area += pts[i][0] * pts[j][1]
        signed_area -= pts[j][0] * pts[i][1]
    
    # 除以 2，取绝对值（有向面积的绝对值 = 实际面积）
    return abs(signed_area) / 2.0

def signed_polygon_area(pts: Polygon) -> float:
    """
    有向面积版本：逆时针为正，顺时针为负。
    可用来判断顶点排列方向。
    """
    n = len(pts)
    signed_area = 0.0
    for i in range(n):
        j = (i + 1) % n
        signed_area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return signed_area / 2.0

# ---- 示例 ----
# 一个简单的 4×3 矩形（逆时针）
rect_ccw = [(0,0), (4,0), (4,3), (0,3)]
print(polygon_area(rect_ccw))         # 12.0 ✓

# 随意的凸六边形
hexagon = [(3,0),(6,2),(6,5),(3,7),(0,5),(0,2)]
print(polygon_area(hexagon))          # 24.0

# 凹多边形（L 形）
L_shape = [(0,0),(2,0),(2,1),(1,1),(1,2),(0,2)]
print(polygon_area(L_shape))          # 3.0 ✓（凹多边形同样正确）

# 判断方向
print("逆时针?" , signed_polygon_area(rect_ccw) > 0)   # True
rect_cw = list(reversed(rect_ccw))
print("顺时针?" , signed_polygon_area(rect_cw) < 0)    # True
```

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef pair<ll, ll> Point;  // {x, y}

// Shoelace 公式：返回有向面积的 2 倍（避免除法，整数精确计算）
// 【设计考量】乘以 2 保留整数，最后判断时再 /2 或直接用 2 倍面积比较
ll signed_area2(const vector<Point>& pts) {
    int n = pts.size();
    if (n < 3) return 0;
    
    ll area2 = 0;
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        // cross(pts[i], pts[j]) with origin at (0,0)
        area2 += pts[i].first  * pts[j].second;
        area2 -= pts[j].first  * pts[i].second;
    }
    return area2;  // 逆时针为正，顺时针为负
}

// 实际面积（double）
double polygon_area(const vector<Point>& pts) {
    return abs(signed_area2(pts)) / 2.0;
}

// 【浮点版本】坐标为 double 时使用
typedef pair<double, double> PointF;

double polygon_area_f(const vector<PointF>& pts) {
    int n = pts.size();
    if (n < 3) return 0;
    double area2 = 0;
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        area2 += pts[i].first  * pts[j].second
               - pts[j].first  * pts[i].second;
    }
    return fabs(area2) / 2.0;
}

int main() {
    // 4×3 矩形（逆时针）
    vector<Point> rect = {{0,0},{4,0},{4,3},{0,3}};
    cout << polygon_area(rect) << "\n";  // 12

    // 有向面积判断方向
    ll s2 = signed_area2(rect);
    cout << (s2 > 0 ? "逆时针" : "顺时针") << "\n";  // 逆时针

    // L 形凹多边形
    vector<Point> L = {{0,0},{2,0},{2,1},{1,1},{1,2},{0,2}};
    cout << polygon_area(L) << "\n";  // 3
    return 0;
}
```

### 40.1.4 整数坐标下的精确计算

计算几何的一大痛点是**浮点误差**。当多边形顶点均为整数坐标时，Shoelace 公式的每步运算都是整数乘法，结果（乘以 2 后）也是整数，完全精确，无需任何 epsilon 比较：

- 使用 `long long` 存储 $2S$（面积的 2 倍）
- 若需比较面积大小，直接比较 $2S^2$ 即可，无需开方
- 只有最终需要输出实际面积时才除以 2（若为整数坐标，$2S$ 必为整数，若为奇数则面积含 $0.5$）

**实际竞赛例子 — LeetCode #812 最大三角形面积**：

给定若干点，找三点构成最大三角形。暴力枚举所有三点组合 $O(n^3)$，用 Shoelace 的三点版本 $|x_A(y_B-y_C) + x_B(y_C-y_A) + x_C(y_A-y_B)| / 2$ 计算面积，$O(1)$ 单次计算。

---

## 40.2 点在多边形内：射线法与绕数法

### 40.2.1 问题定义与直觉

**问题**：给定二维平面上的一个简单多边形（不自交）和一个查询点 $Q$，判断 $Q$ 是否在多边形内部。

**生活比喻**：你站在一片奇形怪状的农田边界外，闭上眼睛向任意方向走一条直线，穿越围栏偶数次说明你在外面，穿越奇数次说明你在里面。这就是射线法的基本直觉。

射线法（Ray Casting）的核心思想：
1. 从查询点 $Q$ 向**任意方向**（通常取正 $x$ 轴方向）发射一条射线；
2. 统计射线与多边形各边的**交点数**；
3. 奇数个交点 → 在内部；偶数个交点 → 在外部。

**为什么这个方法成立？** 拓扑学告诉我们，平面上任意一条从内部出发的射线，每次穿过边界都会改变"内/外"状态。从内部出发一定先穿越一次。

<div data-component="PointInPolygonRayCast"></div>

### 40.2.2 退化情形的处理

射线法最麻烦的部分是处理**射线恰好经过多边形顶点或与边共线**的情况：

**情形一：射线经过顶点**  
若射线过顶点 $V$，需要确定只计数一次，而不是两次（属于两条边的端点）。解决方法：**规定只有当顶点 $V$ 是其所属边中 $y$ 坐标"下方"的端点时，才计数**。即：

- 边的一个端点 $y > Q.y$（上方），另一端点 $y \leq Q.y$（下方或同高），视为"正常穿越"，计数 1；
- 两端点都在 $Q.y$ 以上或以下，不作贡献。

**情形二：射线与边共线（水平边）**  
若多边形有一条水平边与射线同高，跳过该边（不计数）。通过上述端点规约（$y > Q.y$ 才算上端点）可以自动处理。

```python
def point_in_polygon_ray(qx: float, qy: float, poly: List[Tuple[float, float]]) -> bool:
    """
    射线法：判断点 (qx, qy) 是否在多边形 poly 内部。
    
    约定：
    - 边界上的点（在多边形边上）：本函数返回 True（需根据题意调整）
    - 射线方向：向正 x 轴方向（水平向右）
    
    【边界条件处理】
    - 使用 y > qy 而非 y >= qy，确保顶点只被计数一次
    - 水平边（两端点 y 相同）自动被跳过
    
    时间复杂度：O(n)，n = 多边形顶点数
    """
    n = len(poly)
    inside = False
    
    j = n - 1  # 前一个顶点
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        
        # 关键条件：边的两端点分别在射线的上下两侧
        # (yi > qy) != (yj > qy) 确保恰好一个端点在射线上方
        if (yi > qy) != (yj > qy):
            # 计算射线与边的交点的 x 坐标
            # 利用线性插值：t = (qy - yj) / (yi - yj)
            # x_intersect = xj + t * (xi - xj)
            x_intersect = xj + (qy - yj) / (yi - yj) * (xi - xj)
            
            # 只有交点在查询点右侧才计数
            if qx < x_intersect:
                inside = not inside  # 奇偶翻转
        
        j = i  # 更新前一个顶点
    
    return inside

# ---- 验证 ----
# 正方形 [0,4]×[0,4]
square = [(0,0),(4,0),(4,4),(0,4)]
print(point_in_polygon_ray(2, 2, square))    # True  （内部）
print(point_in_polygon_ray(5, 2, square))    # False （外部）
print(point_in_polygon_ray(2, 0, square))    # 边界点，取决于实现约定

# L 形多边形
L_shape = [(0,0),(4,0),(4,2),(2,2),(2,4),(0,4)]
print(point_in_polygon_ray(1, 3, L_shape))   # True  （L 左上角内部）
print(point_in_polygon_ray(3, 3, L_shape))   # False （L 缺口处外部）
```

```cpp
#include <bits/stdc++.h>
using namespace std;

typedef double ld;
const ld EPS = 1e-9;

struct Point { ld x, y; };

// 判断点 (qx, qy) 是否在多边形 poly 内（射线法）
// 返回 true = 内部（包含边界由 EPS 处理）
bool point_in_polygon_ray(ld qx, ld qy, const vector<Point>& poly) {
    int n = poly.size();
    bool inside = false;
    
    for (int i = 0, j = n - 1; i < n; j = i++) {
        ld xi = poly[i].x, yi = poly[i].y;
        ld xj = poly[j].x, yj = poly[j].y;
        
        // 两端点分别在射线的上下两侧
        // 用 > 而非 >= 确保端点只计数一次
        if ((yi > qy) != (yj > qy)) {
            // 线性插值计算交点 x
            ld x_intersect = xj + (qy - yj) / (yi - yj) * (xi - xj);
            if (qx < x_intersect - EPS) {
                inside = !inside;
            }
        }
    }
    return inside;
}

int main() {
    vector<Point> square = {{0,0},{4,0},{4,4},{0,4}};
    cout << point_in_polygon_ray(2, 2, square) << "\n";  // 1 (true)
    cout << point_in_polygon_ray(5, 2, square) << "\n";  // 0 (false)
    
    // L 形
    vector<Point> L = {{0,0},{4,0},{4,2},{2,2},{2,4},{0,4}};
    cout << point_in_polygon_ray(1, 3, L) << "\n";  // 1
    cout << point_in_polygon_ray(3, 3, L) << "\n";  // 0
}
```

### 40.2.3 绕数法（Winding Number）

射线法对**简单多边形**（不自交）非常高效。但当多边形**自交**（如五角星）时，射线法无法正确判断哪些区域是"内部"。此时需要**绕数法（Winding Number）**。

**绕数** $w(Q, P)$ 定义为：从点 $Q$ 观察多边形 $P$ 时，$P$ 的边界绕 $Q$ 旋转的**圈数**（正为逆时针，负为顺时针）。

- $w \neq 0$：点在多边形"内部"（按绕数规则）
- $w = 0$：点在多边形"外部"

计算方法：遍历每条边 $(P_i, P_{i+1})$，根据边相对于 $Q$ 的位置（上穿或下穿 $Q$ 的水平线）累加或减少绕数：

$$w = \sum_{\text{每条边}} \begin{cases} +1 & \text{边从下向上穿越 }Q\text{ 的水平线，且 }Q\text{ 在边的左侧} \\ -1 & \text{边从上向下穿越 }Q\text{ 的水平线，且 }Q\text{ 在边的右侧} \\ 0 & \text{其他} \end{cases}$$

```python
def winding_number(qx: float, qy: float, poly: List[Tuple[float, float]]) -> int:
    """
    绕数法：处理自交多边形（如五角星）。
    返回绕数 w，w != 0 表示点在多边形内。
    
    相比射线法，绕数法更鲁棒（可处理非简单多边形），
    但实现稍复杂，常数因子略大。
    
    时间复杂度：O(n)
    """
    def cross_2d(ax, ay, bx, by) -> float:
        """二维叉积"""
        return ax * by - ay * bx
    
    n = len(poly)
    w = 0
    
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        
        if y1 <= qy:
            # 边从下往上（y1 <= qy < y2）
            if y2 > qy:
                # Q 是否在边的左侧？用叉积判断
                c = cross_2d(x2 - x1, y2 - y1, qx - x1, qy - y1)
                if c > 0:
                    w += 1   # 逆时针穿越：+1
        else:
            # 边从上往下（y1 > qy >= y2）
            if y2 <= qy:
                c = cross_2d(x2 - x1, y2 - y1, qx - x1, qy - y1)
                if c < 0:
                    w -= 1   # 顺时针穿越：-1
    
    return w  # w != 0 表示在内部

# ---- 五角星测试 ----
# 一个自交的五角星顶点（外圈五点）
import math
star = [(math.cos(math.radians(90 + i*144)), 
         math.sin(math.radians(90 + i*144))) for i in range(5)]
# 中心点的绕数为 2（五角星内部绕两圈）
print(winding_number(0, 0, star))   # 2 或 -2（取决于顶点顺序）
```

<div data-component="WindingNumberDemo"></div>

### 40.2.4 凸多边形的 $O(\log n)$ 点包含判断

对于**凸多边形**，可以利用其单调性将查询速度从 $O(n)$ 提升到 $O(\log n)$：

**算法**：以凸多边形的第一个顶点 $P_0$ 为"扇形顶点"，将凸多边形分成 $n-2$ 个三角形扇形。对于查询点 $Q$：
1. 先判断 $Q$ 是否在整个扇形的角度范围内（与 $P_0P_1$ 和 $P_0P_{n-1}$ 的叉积符号）；
2. 二分查找 $Q$ 落在哪个扇形扇面（哪两条从 $P_0$ 出发的对角线之间）；
3. 在找到的那个三角形 $P_0 P_i P_{i+1}$ 中，用叉积判断 $Q$ 是否在三角形内。

总时间 $O(\log n)$，适合在线多次查询同一个凸多边形。

<div data-component="ConvexPolygonBinarySearch"></div>

```python
def point_in_convex_polygon(q: Tuple[float, float], poly: List[Tuple[float, float]]) -> bool:
    """
    凸多边形点包含判断：O(log n) 二分法。
    
    前提：poly 顶点按逆时针排列，不含重复点。
    
    【算法步骤】
    1. O(1) 判断 q 与扇形起始/终止边的方向关系
    2. O(log n) 二分定位所在三角形扇面
    3. O(1) 判断 q 是否在该三角形内
    """
    def cross(ox, oy, ax, ay, bx, by):
        return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)
    
    n = len(poly)
    if n < 3:
        return False
    
    # P0 为扇形基点
    p0x, p0y = poly[0]
    qx, qy = q
    
    # 步骤1：快速排除：q 必须在 P0->P1 的左侧，且在 P0->P_{n-1} 的右侧
    if cross(p0x, p0y, poly[1][0], poly[1][1], qx, qy) < 0:
        return False
    if cross(p0x, p0y, poly[n-1][0], poly[n-1][1], qx, qy) > 0:
        return False
    
    # 步骤2：二分查找 q 在哪个扇面
    lo, hi = 1, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cross(p0x, p0y, poly[mid][0], poly[mid][1], qx, qy) >= 0:
            lo = mid
        else:
            hi = mid
    
    # 步骤3：检查 q 是否在三角形 P0, poly[lo], poly[lo+1] 内
    return cross(poly[lo][0], poly[lo][1], poly[lo+1][0], poly[lo+1][1], qx, qy) >= 0
```

---

## 40.3 线段求交进阶：Shamos-Hoey 与 Bentley-Ottmann

### 40.3.1 问题回顾与分级

给定 $n$ 条线段，存在两类问题：

| 问题 | 算法 | 时间复杂度 |
|------|------|-----------|
| **判断是否存在**任意一对相交线段 | Shamos-Hoey | $O(n \log n)$ |
| **枚举所有** $k$ 个交点 | Bentley-Ottmann | $O((n+k) \log n)$ |

最朴素的暴力法是 $O(n^2)$ 枚举所有对，然后对每对线段用 $O(1)$ 的叉积判断相交。当 $n$ 很大时，$O(n \log n)$ 的扫描线算法就很必要了。

### 40.3.2 Shamos-Hoey 算法：有则返回，无则确认

**核心思想**：扫描线从左到右移动，维护一个按 $y$ 坐标排序的**活跃线段集合**（Active Set），当一条新线段被插入时，只需检查它与**上、下两个最近邻线段**是否相交——如果 $n$ 条线段之间存在交点，那么**在扫描线某时刻**必然存在两条在活跃集中相邻的线段相交。

**事件队列（Event Queue）**：按 $x$ 坐标排序的端点列表（包含每条线段的左端点和右端点）。

**活跃集（Status Structure）**：通常用平衡 BST（如红黑树）维护，按当前扫描线位置处各线段的 $y$ 坐标排序。

**算法流程**：

```
事件队列 EQ ← 所有线段的左端点和右端点，按 x 排序
活跃集 S ← 空的有序集合（按 y 坐标）

for 每个事件点 e in EQ（按 x 从小到大）:
    if e 是线段 s 的左端点（INSERT 事件）:
        将 s 插入活跃集 S
        检查 s 与其在 S 中直接上邻 above(s) 是否相交 → 若是，返回 True
        检查 s 与其在 S 中直接下邻 below(s) 是否相交 → 若是，返回 True
    
    if e 是线段 s 的右端点（DELETE 事件）:
        检查 s 的上邻 t1 = above(s) 与下邻 t2 = below(s) 是否相交 → 若是，返回 True
        从活跃集 S 中删除 s
        
返回 False（无任何两线段相交）
```

**正确性关键**：Bentley-Ottmann 的正确性依赖如下几何事实——若两条线段 $s_1, s_2$ 相交，则在扫描线到达交点**略左前**的某个时刻，它们在活跃集中必然相邻。

<div data-component="ShamosHoeyDemo"></div>

```python
from sortedcontainers import SortedList

def shamos_hoey(segments):
    """
    Shamos-Hoey 算法：判断 n 条线段中是否存在任意两条相交。
    
    参数：
        segments: 线段列表，每条线段为 ((x1,y1), (x2,y2))
                  x1 <= x2（左端点在前，右端点在后）
    返回：
        True 如果存在相交，否则 False
    
    时间复杂度：O(n log n)
    空间复杂度：O(n)
    
    【注意】这是教学简化版，实际实现需要处理：
    1. 垂直线段（x1 == x2）
    2. 共端点线段（不视为相交还是视为相交，取决于题意）
    3. 活跃集的 y 坐标会随扫描线位置变化（动态 BST key）
    
    生产环境推荐使用成熟计算几何库（如 CGAL、Shapely）
    """
    
    def cross(o, a, b):
        """叉积 cross(OA, OB)"""
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    
    def on_segment(p, q, r):
        """判断点 r 是否在线段 pq 上"""
        return (min(p[0],q[0]) <= r[0] <= max(p[0],q[0]) and
                min(p[1],q[1]) <= r[1] <= max(p[1],q[1]))
    
    def segments_intersect(s1, s2):
        """O(1) 判断两线段是否相交"""
        (p1,p2), (p3,p4) = s1, s2
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1>0 and d2<0) or (d1<0 and d2>0)) and \
           ((d3>0 and d4<0) or (d3<0 and d4>0)):
            return True
        if d1==0 and on_segment(p3,p4,p1): return True
        if d2==0 and on_segment(p3,p4,p2): return True
        if d3==0 and on_segment(p1,p2,p3): return True
        if d4==0 and on_segment(p1,p2,p4): return True
        return False
    
    # 建立事件队列：(x, 0=左/1=右, 线段索引)
    events = []
    for i, ((x1,y1),(x2,y2)) in enumerate(segments):
        if x1 > x2:  # 确保左端点在前
            x1,y1,x2,y2 = x2,y2,x1,y1
            segments[i] = ((x1,y1),(x2,y2))
        events.append((x1, 0, i))  # 左端点事件
        events.append((x2, 1, i))  # 右端点事件
    events.sort()
    
    # 简化版：用列表模拟活跃集（可换作真正的平衡 BST）
    active = []
    
    for x, typ, idx in events:
        seg = segments[idx]
        if typ == 0:  # INSERT
            # 找插入位置（按该 x 处的 y 坐标）
            pos = len(active)
            for k, j in enumerate(active):
                # 检查与所有当前活跃线段（教学简化，实际只需邻居）
                if segments_intersect(seg, segments[j]):
                    return True
            active.append(idx)
        else:  # DELETE
            if idx in active:
                active.remove(idx)
    
    return False

# ---- 测试 ----
segs = [((0,2),(4,2)), ((1,0),(1,4)), ((2,0),(5,3))]
print(shamos_hoey(segs))  # True（前两条线段相交）

segs2 = [((0,0),(2,0)), ((0,2),(2,2)), ((0,4),(2,4))]
print(shamos_hoey(segs2))  # False（所有线段平行，无交点）
```

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

struct Seg {
    ll x1, y1, x2, y2;
    int id;
    // 确保左端点在前
    Seg(ll x1, ll y1, ll x2, ll y2, int id)
        : x1(x1), y1(y1), x2(x2), y2(y2), id(id) {
        if (this->x1 > this->x2) {
            swap(this->x1, this->x2);
            swap(this->y1, this->y2);
        }
    }
};

ll cross(ll ox, ll oy, ll ax, ll ay, ll bx, ll by) {
    return (ax-ox)*(ll)(by-oy) - (ay-oy)*(ll)(bx-ox);
}

bool on_seg(ll px, ll py, ll qx, ll qy, ll rx, ll ry) {
    return min(px,qx) <= rx && rx <= max(px,qx) &&
           min(py,qy) <= ry && ry <= max(py,qy);
}

bool intersect(const Seg& a, const Seg& b) {
    ll d1 = cross(b.x1,b.y1, b.x2,b.y2, a.x1,a.y1);
    ll d2 = cross(b.x1,b.y1, b.x2,b.y2, a.x2,a.y2);
    ll d3 = cross(a.x1,a.y1, a.x2,a.y2, b.x1,b.y1);
    ll d4 = cross(a.x1,a.y1, a.x2,a.y2, b.x2,b.y2);
    
    if (((d1>0&&d2<0)||(d1<0&&d2>0)) && ((d3>0&&d4<0)||(d3<0&&d4>0))) return true;
    if (!d1 && on_seg(b.x1,b.y1,b.x2,b.y2,a.x1,a.y1)) return true;
    if (!d2 && on_seg(b.x1,b.y1,b.x2,b.y2,a.x2,a.y2)) return true;
    if (!d3 && on_seg(a.x1,a.y1,a.x2,a.y2,b.x1,b.y1)) return true;
    if (!d4 && on_seg(a.x1,a.y1,a.x2,a.y2,b.x2,b.y2)) return true;
    return false;
}

// 教学简化版：O(n^2) 验证用
bool shamos_hoey_simple(vector<Seg>& segs) {
    int n = segs.size();
    for (int i = 0; i < n; i++)
        for (int j = i+1; j < n; j++)
            if (intersect(segs[i], segs[j])) return true;
    return false;
}

int main() {
    vector<Seg> s = {
        {0,2,4,2,0}, {1,0,1,4,1}, {2,0,5,3,2}
    };
    cout << shamos_hoey_simple(s) << "\n";  // 1 (true)
}
```

> **⚠️ 实现注意**：在竞赛中，完整实现 Shamos-Hoey 需要细心处理活跃集的 y 坐标随扫描线变化的问题（BST 的 key 动态变化）。生产级实现推荐使用 CGAL。

### 40.3.3 Bentley-Ottmann 算法：枚举所有交点

若需要枚举所有 $k$ 个交点，Bentley-Ottmann 在 Shamos-Hoey 基础上新增了**交点事件**：

**三种事件类型**：
1. **左端点（Insert）**：线段开始，插入活跃集，检查上/下邻居是否有新交点
2. **右端点（Delete）**：线段结束，删除线段，检查其原上/下邻居是否有新交点（因为删除后它们成为新邻居）
3. **交点事件（Swap）**：两条活跃线段 $s_i, s_j$ 在交点处互换 y 顺序，将来各自检查新邻居

**时间复杂度**：$O((n + k) \log n)$，$k$ 为交点数。当 $k = 0$ 时退化为 $O(n \log n)$（同 Shamos-Hoey）。

**适用场景**：
- GIS 中的多边形叠加分析（Polygon Overlay）
- 地图缩放时的边界相交处理
- 数值分析中函数图像的交点查找

---

## 40.4 旋转卡壳（Rotating Calipers）

### 40.4.1 灵感与核心思想

想象你用**两把平行的卡尺**夹住一个凸多边形。卡尺贴着多边形的两个"对径点"（Antipodal Pair）。现在缓慢地旋转卡尺——每一时刻卡尺夹住的两点就是当前方向上的"最宽点对"。旋转一整圈后，我们就遍历了所有可能的对径点对，其中距离最大的就是**多边形的直径**（最远点对）。

这就是**旋转卡壳（Rotating Calipers）**的核心思想，由 Shamos 在 1978 年提出。

**关键洞察**：凸包上的点对满足单调性——当一把卡尺沿凸包顺时针旋转时，另一个对径点也单调地沿凸包顺时针移动。这消除了暴力枚举 $O(n^2)$ 的必要性，使总体时间降为 $O(n)$（在凸包已知的前提下）。

<div data-component="RotatingCalipersDiameter"></div>

### 40.4.2 最远点对（多边形直径）

**算法**：
1. 求凸包（$O(n \log n)$）
2. 在凸包上用旋转卡壳枚举所有对径点对（$O(n)$）
3. 返回最大距离

**双指针实现**：维护两个指针 $i, j$（初始 $j$ 为距离 $i$ 最远的点），每步推进叉积更小的那个指针：

```python
import math
from typing import List, Tuple

def graham_convex_hull(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Graham 扫描法求凸包（逆时针排列）"""
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    
    pts = sorted(set(pts))
    n = len(pts)
    if n <= 2:
        return pts
    # 构建下凸包 + 上凸包（Andrew Monotone Chain）
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def farthest_pair_rotating_calipers(pts: List[Tuple[float, float]]) -> float:
    """
    旋转卡壳：求点集中最远点对的距离。
    
    步骤：
    1. 计算凸包 H（O(n log n)）
    2. 在凸包上旋转卡壳 O(|H|) 枚举对径点
    3. 记录最大距离
    
    时间复杂度：O(n log n)（凸包决定）
    空间复杂度：O(n)
    """
    hull = graham_convex_hull(pts)
    n = len(hull)
    if n == 1:
        return 0.0
    if n == 2:
        dx = hull[1][0]-hull[0][0]; dy = hull[1][1]-hull[0][1]
        return math.sqrt(dx*dx + dy*dy)
    
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    def dist2(p, q):
        return (p[0]-q[0])**2 + (p[1]-q[1])**2
    
    max_d2 = 0
    j = 1  # 第二个指针从凸包第 1 个点开始
    
    for i in range(n):
        ni = (i + 1) % n
        # 推进 j，直到无法再增大三角形面积
        # cross(hull[i], hull[ni], hull[(j+1)%n]) > cross(hull[i], hull[ni], hull[j])
        # 等价于 cross(hull[i]->hull[ni], hull[j]->hull[j+1]) > 0
        while cross(hull[i], hull[ni], hull[(j+1)%n]) > cross(hull[i], hull[ni], hull[j]):
            j = (j + 1) % n
        # 此时 hull[i] 和 hull[j] 是当前方向下的对径点对
        max_d2 = max(max_d2, dist2(hull[i], hull[j]))
        max_d2 = max(max_d2, dist2(hull[ni], hull[j]))
    
    return math.sqrt(max_d2)

# ---- 测试 ----
points = [(0,0),(3,0),(3,4),(0,4)]  # 4×3 的矩形，直径是对角线 = 5
print(f"{farthest_pair_rotating_calipers(points):.4f}")  # 5.0000

points2 = [(0,0),(6,0),(3,4)]  # 三角形
print(f"{farthest_pair_rotating_calipers(points2):.4f}")  # 约 6.4031
```

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<ll,ll> P;

ll cross(P o, P a, P b) {
    return (ll)(a.first-o.first)*(b.second-o.second)
          -(ll)(a.second-o.second)*(b.first-o.first);
}
ll dist2(P a, P b) {
    return (a.first-b.first)*(a.first-b.first)
          +(a.second-b.second)*(a.second-b.second);
}

// Andrew 单调链凸包
vector<P> convex_hull(vector<P> pts) {
    int n = pts.size();
    sort(pts.begin(), pts.end());
    pts.erase(unique(pts.begin(), pts.end()), pts.end());
    n = pts.size();
    if (n < 2) return pts;
    
    vector<P> hull;
    // 下凸包
    for (auto& p : pts) {
        while (hull.size() >= 2 && cross(hull[hull.size()-2], hull.back(), p) <= 0)
            hull.pop_back();
        hull.push_back(p);
    }
    // 上凸包
    int lower_size = hull.size();
    for (int i = n-2; i >= 0; i--) {
        while ((int)hull.size() > lower_size && cross(hull[hull.size()-2], hull.back(), pts[i]) <= 0)
            hull.pop_back();
        hull.push_back(pts[i]);
    }
    hull.pop_back();
    return hull;
}

// 旋转卡壳：最远点对的距离平方
ll farthest_pair(vector<P> pts) {
    auto hull = convex_hull(pts);
    int n = hull.size();
    if (n == 1) return 0;
    if (n == 2) return dist2(hull[0], hull[1]);
    
    ll ans = 0;
    int j = 1;
    for (int i = 0; i < n; i++) {
        int ni = (i + 1) % n;
        // cross(hull[i]->hull[ni], hull[j]->hull[j+1])
        // 若可以推进 j，则推进
        while (cross(hull[i], hull[ni], hull[(j+1)%n]) >
               cross(hull[i], hull[ni], hull[j]))
            j = (j + 1) % n;
        ans = max(ans, dist2(hull[i], hull[j]));
        ans = max(ans, dist2(hull[ni], hull[j]));
    }
    return ans;
}

int main() {
    vector<P> pts = {{0,0},{3,0},{3,4},{0,4}};
    cout << farthest_pair(pts) << "\n";  // 25 (5²)
    cout << sqrt(farthest_pair(pts)) << "\n";  // 5.0
}
```

### 40.4.3 最小面积外接矩形（Minimum Bounding Rectangle）

**问题**：找到包围给定点集（或凸多边形）的**面积最小**的矩形，矩形可以任意旋转。

**关键引理**：最小面积外接矩形至少有一条边与凸包的某条边重合。

**算法**（O(n)，基于凸包）：
1. 计算凸包 $H$；
2. 对凸包的每条边，用旋转卡壳确定三个支撑点（与该边最远的对面点、最左点、最右点）；
3. 当前边确定外接矩形的一条边，三个支撑点确定另三条边；
4. 计算矩形面积，记录最小值。

**三个支撑点的推进**：当凸包旋转（从一条边的方向转到下一条边的方向）时，三个支撑点均单调地在凸包上推进（总推进次数 $O(n)$）。

<div data-component="MinAreaBoundingRect"></div>

```python
def min_area_bounding_rect(pts: List[Tuple[float, float]]) -> float:
    """
    最小面积外接矩形。
    
    返回：最小外接矩形的面积
    
    算法核心：
    对凸包每条边，计算矩形（高 = 最远点距离，宽 = 投影宽度）。
    
    时间复杂度：O(n log n)（由凸包决定）
    """
    hull = graham_convex_hull(pts)
    n = len(hull)
    if n <= 2:
        return 0.0
    
    def dot(ax, ay, bx, by):
        return ax * bx + ay * by
    def cross_val(ax, ay, bx, by):
        return ax * by - ay * bx
    
    min_area = float('inf')
    
    # 三个卡壳指针：对面（最远点）、最右点、最左点
    j = 1  # 对面（高度）
    rp = 1  # 最右点（宽度右）
    lp = n - 1  # 最左点（宽度左）
    
    for i in range(n):
        ni = (i + 1) % n
        # 当前边的向量
        ex = hull[ni][0] - hull[i][0]
        ey = hull[ni][1] - hull[i][1]
        
        # 推进 j：最远点（叉积最大化 = 高度最大化）
        while cross_val(ex, ey,
                        hull[(j+1)%n][0]-hull[i][0],
                        hull[(j+1)%n][1]-hull[i][1]) > \
              cross_val(ex, ey,
                        hull[j][0]-hull[i][0],
                        hull[j][1]-hull[i][1]):
            j = (j + 1) % n
        
        # 推进 rp：最右点（点积最大化）
        while dot(ex, ey,
                  hull[(rp+1)%n][0]-hull[i][0],
                  hull[(rp+1)%n][1]-hull[i][1]) > \
              dot(ex, ey,
                  hull[rp][0]-hull[i][0],
                  hull[rp][1]-hull[i][1]):
            rp = (rp + 1) % n
        
        # 推进 lp：最左点（点积最小化）
        if i == 0:
            lp = rp
        while dot(ex, ey,
                  hull[(lp+1)%n][0]-hull[i][0],
                  hull[(lp+1)%n][1]-hull[i][1]) < \
              dot(ex, ey,
                  hull[lp][0]-hull[i][0],
                  hull[lp][1]-hull[i][1]):
            lp = (lp + 1) % n
        
        edge_len2 = ex*ex + ey*ey
        
        # 高度（法向距离）
        h = cross_val(ex, ey,
                      hull[j][0]-hull[i][0],
                      hull[j][1]-hull[i][1])
        # 宽度（切向投影差）
        r = dot(ex, ey, hull[rp][0]-hull[i][0], hull[rp][1]-hull[i][1])
        l = dot(ex, ey, hull[lp][0]-hull[i][0], hull[lp][1]-hull[i][1])
        w = r - l
        
        area = (h * w) / edge_len2  # 除以边长平方做归一化
        min_area = min(min_area, area)
    
    return min_area

# ---- 测试 ----
# 轴对齐矩形（最小外接矩形就是自身）
pts = [(0,0),(4,0),(4,3),(0,3)]
print(f"{min_area_bounding_rect(pts):.2f}")  # 12.00

# 旋转 45° 的正方形
import math
sq45 = [(2*math.cos(math.radians(45+90*i)), 2*math.sin(math.radians(45+90*i))) for i in range(4)]
print(f"{min_area_bounding_rect(sq45):.2f}")  # 约 8.00（边长 2√2 的正方形）
```

### 40.4.4 两凸多边形最小距离

若两个**凸多边形**不相交，旋转卡壳也可以 $O(n+m)$ 求最小距离：同时在两个凸多边形上各维护一个指针，比较各边法向量夹角来推进，找到最近点对。

这在机器人避障（两个凸形障碍物间的最窄间隙）和游戏碰撞响应（推开两个凸形物体所需的最小位移）中非常实用。

---

## 40.5 计算几何综合应用

### 40.5.1 SAT 分离轴定理（碰撞检测）

**SAT（Separating Axis Theorem，分离轴定理）** 是游戏开发中最常用的凸形体碰撞检测算法，其核心定理：

> **定理**：两个凸集合不相交，当且仅当存在一条**分离轴**（Separating Axis），使得两集合在该轴上的投影区间**不重叠**。

对于两个凸多边形，只需检查**所有边的法向量**作为候选分离轴——若任意一个方向上的投影不重叠，则两多边形不相交；若所有方向均重叠，则相交。

**算法复杂度**：两个凸多边形分别有 $m, n$ 条边，总共检查 $m+n$ 个候选轴，每次检查 $O(m+n)$，总时间 $O(m+n)$。

<div data-component="SATCollisionDetection"></div>

```python
def sat_convex_polygons(poly_a: List[Tuple[float, float]],
                        poly_b: List[Tuple[float, float]]) -> bool:
    """
    SAT 分离轴定理：判断两个凸多边形是否相交（碰撞检测）。
    
    返回：True = 相交（碰撞），False = 不相交
    
    算法：
    1. 枚举 poly_a 和 poly_b 所有边的法向量作为候选分离轴
    2. 对每条轴，计算两个多边形在轴上的投影区间 [minA, maxA] 和 [minB, maxB]
    3. 若存在任意轴使得两投影不重叠，则不碰撞（提前返回 False）
    4. 所有轴都重叠 → 碰撞（返回 True）
    
    时间复杂度：O(n + m)，n = len(poly_a)，m = len(poly_b)
    """
    def get_axes(poly):
        """获取多边形所有边的法向量（未归一化）"""
        axes = []
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            # 边向量
            ex = poly[j][0] - poly[i][0]
            ey = poly[j][1] - poly[i][1]
            # 法向量（旋转 90°）
            axes.append((-ey, ex))
        return axes
    
    def project(poly, ax, ay):
        """将多边形投影到轴 (ax, ay) 上，返回 [min, max]"""
        dots = [p[0] * ax + p[1] * ay for p in poly]
        return min(dots), max(dots)
    
    # 候选轴 = poly_a 的所有边法向量 + poly_b 的所有边法向量
    axes = get_axes(poly_a) + get_axes(poly_b)
    
    for ax, ay in axes:
        min_a, max_a = project(poly_a, ax, ay)
        min_b, max_b = project(poly_b, ax, ay)
        # 投影不重叠 → 存在分离轴 → 不碰撞
        if max_a < min_b or max_b < min_a:
            return False
    
    return True  # 所有轴均重叠 → 碰撞

# ---- 测试 ----
# 两个正方形，分开放
sq_a = [(0,0),(2,0),(2,2),(0,2)]
sq_b = [(3,0),(5,0),(5,2),(3,2)]
print("相交?", sat_convex_polygons(sq_a, sq_b))  # False（不相交）

# 两个正方形，部分重叠
sq_c = [(1,0),(3,0),(3,2),(1,2)]
print("相交?", sat_convex_polygons(sq_a, sq_c))  # True（重叠）

# 三角形与矩形
tri = [(4,1),(6,1),(5,3)]
rect = [(3.5,0.5),(6.5,0.5),(6.5,2),(3.5,2)]
print("相交?", sat_convex_polygons(tri, rect))   # True（重叠）
```

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef double ld;

struct P { ld x, y; };

// 将多边形投影到轴 (ax, ay) 上，返回 {min, max}
pair<ld,ld> project(const vector<P>& poly, ld ax, ld ay) {
    ld mn = LLONG_MAX, mx = LLONG_MIN;
    for (auto& p : poly) {
        ld d = p.x * ax + p.y * ay;
        mn = min(mn, d); mx = max(mx, d);
    }
    return {mn, mx};
}

// SAT 碰撞检测
bool sat_intersect(const vector<P>& a, const vector<P>& b) {
    // 枚举两多边形的所有边法向量
    auto check = [&](const vector<P>& poly) -> bool {
        int n = poly.size();
        for (int i = 0; i < n; i++) {
            int j = (i + 1) % n;
            ld ex = poly[j].x - poly[i].x;
            ld ey = poly[j].y - poly[i].y;
            // 法向量
            ld nx = -ey, ny = ex;
            
            auto [minA, maxA] = project(a, nx, ny);
            auto [minB, maxB] = project(b, nx, ny);
            if (maxA < minB || maxB < minA) return true;  // 找到分离轴
        }
        return false;
    };
    
    // 存在分离轴 → 不碰撞
    if (check(a) || check(b)) return false;
    return true;  // 碰撞
}

int main() {
    vector<P> sq_a = {{0,0},{2,0},{2,2},{0,2}};
    vector<P> sq_b = {{3,0},{5,0},{5,2},{3,2}};
    cout << sat_intersect(sq_a, sq_b) << "\n";  // 0 (false)
    
    vector<P> sq_c = {{1,0},{3,0},{3,2},{1,2}};
    cout << sat_intersect(sq_a, sq_c) << "\n";  // 1 (true)
}
```

### 40.5.2 可见性图（Visibility Graph）与机器人路径规划

**场景**：机器人在二维平面中从 $S$ 移动到 $T$，中间有若干多边形障碍物（凸）。机器人不能穿越障碍物，要找最短路径。

**可见性图**的节点包含：$S, T$ 以及所有障碍物的顶点。若两点之间的连线不穿越任意障碍物（只经过障碍物边界的切线或顶点），则在图中连一条边，权重为两点欧氏距离。

**算法**：
1. 构建可见性图（Visibility Graph），$O(n^2 \log n)$（扫描线可见性算法）；
2. 在可见性图上运行 Dijkstra，$O(n^2 \log n)$；
3. 返回最短路径。

**关键引理**：最短路径必然沿可见性图的边走（最短路径不会随意绕过凸角内部）。

### 40.5.3 GIS 地理空间查询：多边形叠加

GIS（地理信息系统）中的**多边形叠加分析**是将两个地理图层（如行政区划和土地利用）叠加，计算交、并、差等布尔运算：

- **Polygon Overlay** 的核心算法：先求两个多边形集合所有边之间的交点（Bentley-Ottmann），然后按交点重建多边形拓扑。
- **真实应用**：OpenStreetMap、GDAL、PostGIS 等 GIS 软件的底层几何运算库均依赖类似算法。

### 40.5.4 经典 LeetCode 题解精讲

**LeetCode #587 安装围栏（凸包应用）**

> 在花园中有若干棵树，给出这些树的坐标，求至少需要多长的围栏把所有树包起来（即凸包周长/顶点）。

```python
def outer_trees(trees: List[List[int]]) -> List[List[int]]:
    """
    Andrew Monotone Chain 凸包，边界点全部保留
    
    【注意】本题要求边界上的点也包含在内（即共线点保留），
    而标准凸包实现通常丢弃共线点（cross <= 0 时弹出）。
    本题应改为 cross < 0 时才弹出（严格右转才弹出）。
    """
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    
    pts = sorted([tuple(t) for t in trees])
    n = len(pts)
    if n <= 1:
        return trees
    
    lower = []
    for p in pts:
        # 改为 cross < 0 才弹出（保留共线点）
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) < 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) < 0:
            upper.pop()
        upper.append(p)
    
    hull = list(set(lower[:-1] + upper[:-1]))
    return [[x, y] for x, y in hull]
```

**LeetCode #963 最小面积外接矩形 II（旋转矩形）**

> 给定点集，求最小面积外接矩形（可旋转）。

核心思路：先求凸包，再用旋转卡壳遍历凸包每条边，对每条边枚举外接矩形，时间 $O(n \log n)$。

**LeetCode #812 最大三角形面积**

> 给定点集，找三点组成最大三角形面积。

```python
def largest_triangle_area(points: List[List[int]]) -> float:
    """暴力枚举三点，Shoelace 计算面积：O(n^3)"""
    def area(a, b, c):
        return abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])) / 2
    
    n = len(points)
    ans = 0.0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                ans = max(ans, area(points[i], points[j], points[k]))
    return ans
    
    # 优化版（n 较大时先取凸包）：
    # 凸包 O(n log n) → 凸包上三点枚举 O(h^3)（h = 凸包顶点数，可用旋转卡壳降至 O(h^2)）
```

---

## 40.6 常见错误、调试技巧与精度处理

### 40.6.1 浮点精度的"地雷"

计算几何中 90% 的 bug 都与浮点精度有关：

| 错误类型 | 典型表现 | 解决方法 |
|---------|---------|---------|
| 叉积符号误判 | 接近共线的三点被错判为左/右转 | 整数坐标用 `long long`，浮点用 epsilon 比较 |
| 射线恰好过顶点 | 点在多边形内/外误判 | 使用严格 `>` 而非 `>=` 的端点约定 |
| 旋转卡壳死循环 | 指针推进条件写反 | 确保推进条件包含等号时的行为正确 |
| Shoelace 整数溢出 | 坐标值大（如 $10^9$），面积超 `int` 范围 | 用 `long long`，坐标乘积可达 $10^{18}$ |

### 40.6.2 统一 Epsilon 管理

```python
EPS = 1e-9

def sign(x: float) -> int:
    """返回 x 的符号（+1, 0, -1），考虑 epsilon"""
    if x > EPS:  return 1
    if x < -EPS: return -1
    return 0

def eq(a: float, b: float) -> bool:
    return abs(a - b) < EPS

def le(a: float, b: float) -> bool:
    return a < b + EPS

def ge(a: float, b: float) -> bool:
    return a > b - EPS
```

### 40.6.3 调试工具：可视化输出

在竞赛中遇到几何 bug，最有效的调试方式是将中间状态可视化（输出 SVG 或调用 matplotlib）：

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_polygon(poly, color='blue', fill=True, label=''):
    xs = [p[0] for p in poly] + [poly[0][0]]
    ys = [p[1] for p in poly] + [poly[0][1]]
    plt.plot(xs, ys, color=color, label=label)
    if fill:
        plt.fill(xs[:-1], ys[:-1], color=color, alpha=0.15)

def debug_point_in_polygon(q, poly):
    """可视化调试：射线法点在多边形内"""
    fig, ax = plt.subplots()
    plot_polygon(poly, 'steelblue', label='多边形')
    ax.plot(*q, 'ro', markersize=10, label=f'查询点 {q}')
    # 画射线
    ax.annotate('', xy=(max(p[0] for p in poly)+1, q[1]), xytext=q,
                arrowprops=dict(arrowstyle='->', color='red'))
    result = point_in_polygon_ray(*q, poly)
    ax.set_title(f'点 {q} 在多边形{"内" if result else "外"}')
    ax.legend(); plt.axis('equal'); plt.show()
```

---

## 40.7 复杂度总结与选择指南

| 算法 | 时间复杂度 | 空间复杂度 | 适用场景 |
|------|-----------|-----------|---------|
| Shoelace 公式 | $O(n)$ | $O(1)$ | 多边形面积（任意简单多边形） |
| 射线法（点在多边形内） | $O(n)$ | $O(1)$ | 简单多边形（无自交）的点包含查询 |
| 绕数法 | $O(n)$ | $O(1)$ | 自交多边形（如五角星）的点包含判断 |
| 凸多边形二分法 | $O(\log n)$ | $O(n)$ | 在线多次查询同一凸多边形 |
| Shamos-Hoey | $O(n \log n)$ | $O(n)$ | 判断 n 条线段是否有交 |
| Bentley-Ottmann | $O((n+k) \log n)$ | $O(n+k)$ | 枚举所有 k 个交点 |
| 旋转卡壳（直径） | $O(n)$（+凸包 $O(n \log n)$）| $O(n)$ | 凸多边形最远点对、最小外接矩形 |
| SAT 碰撞检测 | $O(n+m)$ | $O(1)$ | 游戏引擎实时凸形体碰撞检测 |

> **选择原则**：
> - 面积/判断类问题优先考虑 $O(n)$ 的 Shoelace 和射线法；
> - 线段交问题：是否存在交 → Shamos-Hoey；枚举全部 → Bentley-Ottmann；
> - 凸包直径/外接矩形 → 先求凸包，再旋转卡壳；
> - 游戏实时碰撞（≤ 几十条边的凸形体）→ SAT；
> - 精度敏感场景 → 整数坐标 + `long long` 叉积，避免浮点。

---

## 本章小结

本章从多边形面积（Shoelace）出发，系统覆盖了计算几何在工程和竞赛中最核心的进阶技术：

1. **Shoelace 公式**：$O(n)$ 计算任意简单多边形面积，整数坐标下精确无误差；
2. **点在多边形内**：射线法（简单多边形 $O(n)$）、绕数法（自交多边形）、凸多边形二分（$O(\log n)$）；
3. **线段求交**：Shamos-Hoey 存在性检测（$O(n \log n)$）、Bentley-Ottmann 枚举所有交点（$O((n+k) \log n)$）；
4. **旋转卡壳**：$O(n)$ 求凸多边形直径、最小外接矩形、两凸多边形最小距离；
5. **SAT 碰撞检测**：$O(n+m)$ 判断两凸多边形是否碰撞，游戏引擎级别的实时性能。

> **💡 思考题**  
> 旋转卡壳与两指针法本质上类似——都是利用凸包上的单调性避免 $O(n^2)$ 暴力枚举。能否将旋转卡壳推广到三维凸多面体的直径计算？若可以，复杂度如何？（提示：三维凸包有 $O(n)$ 个面，每个面需枚举对极点对……）

> **扩展阅读**  
> - CLRS 第4版 Chapter 33（计算几何）
> - Shamos & Hoey (1976) 论文《Geometric Intersection Problems》
> - Preparata & Shamos《Computational Geometry: An Introduction》
> - LeetCode #587, #963, #812, #149

---

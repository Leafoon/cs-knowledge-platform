---
title: "Chapter 41: 计算复杂性理论"
description: "从图灵机计算模型出发，深入理解 P、NP、NP-hard、NP-complete 的精确定义，掌握多项式时间规约的证明框架，通过 Cook-Levin 定理与经典 NPC 规约链（SAT→3-SAT→顶点覆盖→子集和）建立对「计算难度」的系统认知"
tags: ["complexity-theory", "P-NP", "NP-complete", "reduction", "SAT", "Cook-Levin", "vertex-cover", "decision-problems"]
difficulty: "hard"
updated: "2026-03-12"
---

# Chapter 41: 计算复杂性理论

> **Part XII · 计算复杂性与 NP**

经过前面 40 个章节，我们积累了大量具体的算法工具：排序、图搜索、动态规划、计算几何……但有一类深刻的问题一直潜伏在背后：**有些问题，是否本质上就没有高效算法？**

这不是在说"我们暂时还没发现好算法"，而是在问："就算穷尽人类智慧，这类问题是否注定无法在合理时间内解决？"

**计算复杂性理论**（Computational Complexity Theory）正是研究这个问题的数学分支。它给出了一套严格的语言来讨论"问题难度"，并揭示了一个惊人的事实：数以千计看似毫不相关的问题，其实在难度上是**等价的**——要么全都有高效算法，要么全都没有。而这个问题（P vs NP）至今悬而未决，被列为七大**千禧年数学大奖赛**难题之一。

> **本章学习路径**：计算模型（图灵机）→ 问题分类（P/NP）→ 规约工具 → Cook-Levin 定理 → 经典 NPC 问题链

---

## 41.1 计算模型与问题分类

### 41.1.1 图灵机：计算的正式模型

#### 为什么需要一个"计算的定义"？

在日常编程中，我们直觉上知道什么是"算法"——一系列操作步骤。但要严格讨论哪些问题"可以被计算"、"需要多少步可以计算"，我们必须先有一个**精确的计算模型**。否则，类似于"这个问题需要指数时间"这样的结论就无法严格证明。

**图灵机**（Turing Machine，TM）由艾伦·图灵（Alan Turing）于 1936 年提出，是对"计算"最简洁、最通用的数学抽象。尽管看起来非常原始，理论上它与现代计算机等价——任何现代计算机能干的事，图灵机都能干（反之亦然）。

#### 图灵机的组成

想象一台非常简单的机器：

```
┌─────────────────────────────────────────────────────────┐
│  无限长纸带：  ... [0][1][0][1][ ][ ][ ] ...            │
│                              ↑                          │
│                         读写头（当前位置）              │
│                                                         │
│  有限状态控制器：（状态集合 Q，当前状态 q）            │
│                                                         │
│  转移规则：δ(q, 读到的符号) → (新状态, 写的符号, 移动方向) │
└─────────────────────────────────────────────────────────┘
```

- **纸带（Tape）**：双向无限，每格存储一个符号（如 0、1 或空格）
- **读写头（Head）**：每步只能读/写当前格，然后左移或右移一格
- **状态机（State Control）**：有限个内部状态，包含起始状态 $q_0$、接受状态 $q_{accept}$、拒绝状态 $q_{reject}$
- **转移函数 $\delta$**：当前状态 + 读到的符号 → 下一状态 + 写入符号 + 移动方向

**执行流程**：
1. 输入写在纸带上，读写头指向第一个符号，机器处于 $q_0$
2. 每步根据（当前状态，当前符号）查转移函数，执行写入 + 移动
3. 进入 $q_{accept}$ → **接受**（Yes）；进入 $q_{reject}$ → **拒绝**（No）；永不停止 → 死循环

<div data-component="TuringMachineSimulator"></div>

#### 确定型 vs 非确定型图灵机

**确定型图灵机（DTM）**：每个（状态，符号）组合只有**唯一**一个后继步骤。这就是真实计算机的模型——每步操作是确定的。

**非确定型图灵机（NTM）**：转移函数是**多值**的——同一个（状态，符号）可以有多个合法后继步骤。机器可以"神奇地"同时沿**所有**分支探索，只要**存在**一条接受路径，就算接受。

这并不是真实存在的物理设备，而是一个数学工具，用来定义 NP 类。可以想象成：NTM 能够在多个选择中"猜到"正确答案，然后验证它。

> **类比理解**：想象你在迷宫里找出口。DTM = 顺序尝试每条路径；NTM = 同时派出无数复制人分别走每条路，只要有一个复制人找到出口就成功。NTM 不是"并行计算机"，而是一种"完美猜测"的理论工具。

#### 多带图灵机与随机访问机器

实际上，复杂性理论用的计算模型有多种变体（多纸带 TM、随机访问机器 RAM 等），但它们彼此之间只有多项式的时间差异，因此对"多项式时间可解"这个核心概念的讨论毫无影响。**Church-Turing 论题**断言：一切合理的计算模型在能力上等价，差别只在效率细节，不影响"什么是可计算的"这一根本问题。

---

### 41.1.2 决策问题：统一的"是/否"框架

#### 为什么复杂性理论专注于决策问题？

复杂性理论的研究对象主要是**决策问题**（Decision Problem）——输出只有"是（Yes）"或"否（No）"的问题。

这一限制看似很强，但实际上**不失一般性**。任何优化问题都可以"降级"为一系列决策问题：

**例：最短路问题**
- **优化版**：图 $G$ 中，$s$ 到 $t$ 的最短路是多长？
- **决策版**：图 $G$ 中，$s$ 到 $t$ 是否存在长度 $\leq k$ 的路？

如果我们能高效解决决策版，通过二分搜索 $k$ 就能找到最优 $k$（多条调用，多项式代价）。因此，讨论决策问题的难度等价于讨论对应优化问题的难度。

#### 语言（Language）的概念

在形式语言理论中，决策问题等价于"语言识别"：

$$L = \{ \text{编码}(x) \mid x \text{ 是 Yes 实例} \}$$

例如，"图 $G$ 是否是二部图"这个决策问题对应语言：

$$L_{\text{BIPARTITE}} = \{ \langle G \rangle \mid G \text{ 是二部图} \}$$

其中 $\langle G \rangle$ 是图 $G$ 的二进制编码。图灵机"解决"这个问题，就等于它能识别（接受/拒绝）这个语言的所有字符串。

#### 常见经典问题的决策版对应

| 优化问题 | 决策版本 | 是否在 P 中 |
|---------|---------|------------|
| 图的最小生成树（权重）| MST 权重是否 $\leq k$? | ✅ 是（Kruskal/Prim） |
| 最短路 | $s$ 到 $t$ 距离 $\leq k$? | ✅ 是（Dijkstra） |
| 最大独立集大小 | 大小 $\geq k$ 的独立集是否存在? | ❌ 未知（NPC） |
| TSP 最短回路 | 哈密顿回路权重 $\leq k$? | ❌ 未知（NPC） |
| 整数规划最优解 | 目标值 $\geq k$ 的可行解存在? | ❌ 未知（NP-hard） |

---

### 41.1.3 多项式时间：高效算法的标准线

#### 为什么"多项式时间"是高效的分界线？

我们说一个算法是**高效的（Efficient）**，如果其运行时间是输入规模 $n$ 的**多项式函数**，即 $O(n^c)$（$c$ 为某个固定常数）。

这个定义看起来有些宽泛——$O(n^{100})$ 在实践中根本不可用，但理论上仍算多项式时间。然而，历史经验表明，**只要一个问题被证明有多项式算法，人们通常很快就能找到更实用的低次多项式算法**。因此多项式时间是一条合理的理论分界线。

对比来看：

| 类型 | 函数增长示例 | 意义 |
|------|------------|------|
| 对数 | $O(\log n)$ | 最理想，二分查找等级 |
| 多项式 | $O(n^2), O(n^3)$ | 可接受，被认为"高效" |
| 准多项式 | $O(n^{\log n})$ | 理论上低于指数，但实际极慢 |
| 指数 | $O(2^n)$ | 仅适用于极小规模（$n \leq 40$） |
| 阶乘 | $O(n!)$ | 完全不可用，TSP 暴力搜索 |

**为什么指数是绝境**：当 $n = 100$ 时，$2^{100} \approx 10^{30}$，宇宙年龄约 $4 \times 10^{17}$ 秒，每秒 $10^{10}$ 步的计算机需要 $10^{13}$ 年……因此指数时间算法对规模稍大的问题毫无实用价值。

#### 输入规模的精确定义

**输入规模 $n$** 是输入的**二进制编码位数**（bits）。这在大多数情况下与直觉一致：

- 整数 $N$：编码需要 $n = \lceil \log_2 N \rceil$ 位
- 图 $G = (V, E)$：编码需要 $O(|V|^2)$ 位（邻接矩阵）或 $O(|V| + |E|)$ 位（邻接表）
- 长度为 $m$ 的字符串：编码需要 $O(m)$ 位

> **⚠️ 陷阱：伪多项式时间（Pseudo-Polynomial）**
>
> 背包问题的 DP 算法时间复杂度是 $O(nW)$，其中 $n$ 是物品数，$W$ 是容量上限。
> 表面看是"多项式"，但 $W$ 的编码长度是 $O(\log W)$，所以输入规模 $n_{\text{input}} \approx n + \log W$，而 $W = 2^{O(\log W)} = 2^{O(n_{\text{input}})}$ 是指数量级！
> 因此背包 DP 是**伪多项式时间**，不是真正的多项式时间，背包问题仍然是 NP-hard 的。

---

### 41.1.4 问题的编码：一切从比特开始

图灵机的输入是**符号串**，因此任何问题实例都必须编码为有限长的字符串。合理的编码方案应满足：

1. **紧凑性**：编码长度是实例"自然大小"的多项式倍（如图的顶点数和边数）
2. **可解码性**：可在多项式时间内解码为原始实例

不合理的编码（如将整数 $N$ 编码为 $N$ 个 1 的一元表示）会人为膨胀输入，使本来是指数时间的算法看起来是"多项式"（**一元编码骗局**）。

复杂性理论中，所有讨论默认使用**合理的二进制编码**，且不同合理编码之间只差多项式因子，不影响多项式时间/指数时间的分类。

---

## 41.2 P 类与 NP 类

### 41.2.1 P 类：高效可解的问题集

$$\boxed{P = \text{存在多项式时间确定型算法可以决策的语言集合}}$$

直觉上，$P$ 类包含所有"能高效求解"的决策问题。

**P 中的经典例子**：

| 问题 | 算法 | 时间复杂度 |
|------|------|-----------|
| 排序（判断是否已排序）| 扫描 | $O(n)$ |
| 图的连通性 | BFS/DFS | $O(V + E)$ |
| 最短路 | Dijkstra | $O(E \log V)$ |
| 最小生成树 | Prim/Kruskal | $O(E \log V)$ |
| 最大流/最小割 | Ford-Fulkerson改进版 | 多项式 |
| 线性规划 | 内点法/椭球法 | 多项式 |
| 素性测试 | AKS 算法（2002年）| $O(n^6)$ 内 |
| 2-SAT | 强连通分量 | $O(V + E)$ |

注意素性测试（判断一个数是否是质数）直到 **2002 年**才被 Agrawal、Kayal、Saxena 证明在 P 中，这是一个令人惊喜的里程碑（AKS 算法）。

---

### 41.2.2 NP 类：高效可验证的问题集

$$\boxed{NP = \text{存在多项式时间确定型验证器（Verifier）的语言集合}}$$

**NP 的等价定义**：问题 $L \in NP$，当且仅当存在：
- 多项式时间算法 $V$（验证器）
- 多项式 $p$

使得对于任意输入 $x$：
$$x \in L \iff \exists c, |c| \leq p(|x|) \quad \text{且} \quad V(x, c) = \text{接受}$$

这里 $c$ 称为**证书（Certificate）**或**见证（Witness）**。

> **通俗理解**：NP = "如果给你一个候选答案，你能在多项式时间内验证它是否正确"的问题集。

#### 经典 NP 问题及其证书

| 问题 | 证书 | 验证方法 | 验证时间 |
|------|------|---------|---------|
| 哈密顿回路 | 一条路径（顶点序列）| 检查是否经过所有顶点各一次 | $O(V)$ |
| 子集和问题 | 目标子集 | 求和检验 | $O(n)$ |
| 图着色（3色）| 每顶点的颜色 | 检查相邻顶点颜色不同 | $O(E)$ |
| SAT（可满足性）| 变量的赋值（0/1 序列）| 代入公式求值 | $O(|\phi|)$ |
| 团（CLIQUE）| 顶点子集 | 检查是否两两相邻 | $O(k^2)$ |
| TSP 决策版 | 顶点访问顺序 | 计算路径总权重 | $O(V)$ |

**关键字：验证容易，求解不知道**

NP 的本质是一种**非对称性**：给你一个解，你能快速验证它对不对；但从零开始**找**一个解，你可能需要指数时间穷举。

---

### 41.2.3 P ⊆ NP：验证比求解更容易

**定理**：$P \subseteq NP$。

**证明**：若 $L \in P$，则存在多项式时间算法 $A$ 解决 $L$。对任意 $x$，我们构造验证器 $V(x, c) = A(x)$（直接忽略证书 $c$，用 $A$ 求解）。这是一个多项式时间验证器，故 $L \in NP$。$\square$

换言之：**能高效求解的问题，一定也能高效验证**（直接运行算法得答案就是最好的证明）。

**$P = NP$？这是千古谜题。**

如果 $P = NP$，则任何能"高效验证"的问题也能"高效求解"——这将颠覆现代密码学、人工智能规划、药物设计等众多领域：

- 密码学：公钥密码（RSA、椭圆曲线）依赖"因式分解难"，$P = NP$ 意味着 $n$ 位整数的因子分解有多项式算法，这些密码系统立刻崩溃
- 蛋白质折叠：能量最低构型的求解将高效化，蛋白质设计革命
- 最优规划：NLP、调度、GPS 路径规划等优化问题全部可高效求解

目前学术界普遍认为 $P \neq NP$，但这至今**没有数学证明**。

<div data-component="PvsNPVennDiagram"></div>

---

### 41.2.4 co-NP：反面也可验证

**co-NP** 是 NP 的"补"：

$$\boxed{co\text{-}NP = \{ L \mid \overline{L} \in NP \}}$$

其中 $\overline{L}$ 是 $L$ 的补语言（所有 No 实例组成的集合在 NP 中）。

co-NP 又等价于：存在多项式时间**反驳器**——当答案是"否"时，能提供一个多项式长度的"否证"（disqualifier）。

**例子**：

- "给定图 $G$，它是否**没有**哈密顿回路？"——这是哈密顿回路问题的补，属于 co-NP（但究竟是否在 NP 中则未知）。
- **素性测试**（PRIMES）：判断一个数是否是质数。有趣的是，PRIMES 既在 NP 中（证书是"原根"，验证用费马小定理变形）也在 co-NP 中（合数的证书是因子），后来被证明在 P 中（AKS2002）。

**关键关系（假设 P≠NP）**：

$$P \subsetneq NP \cap co\text{-}NP \subsetneq NP, \quad NP \neq co\text{-}NP$$

若 NP = co-NP，则 NP 问题都有短的否证，这比 P = NP 弱，但仍是重大突破。

---

### 41.2.5 NP-hard：至少与 NP 中最难问题一样难

$$\boxed{H \text{ 是 NP-hard} \iff \forall L \in NP, L \leq_p H}$$

读作："对 NP 中所有问题 $L$，$L$ 都能多项式时间规约到 $H$。"

也就是说，$H$ **至少和 NP 中任何问题一样难**。注意：NP-hard **不要求** $H$ 本身在 NP 中——NP-hard 问题可以比 NP "还难"！

**例子**：
- **停机问题（Halting Problem）**：给定程序和输入，判断程序是否停止。停机问题是 NP-hard（以及 co-NP-hard），但它甚至**不是可判定的**（无法被任何图灵机正确判断），更谈不上在 NP 中。
- **PSPACE-complete 问题**（如定量 2-SAT）：在 PSPACE 中但不在 NP 中（条件 NP ≠ PSPACE 成立时），却是 NP-hard。

---

### 41.2.6 NP-complete：NP 中最难的问题

$$\boxed{NPC = NP \cap NP\text{-hard}}$$

NP-complete（简称 NPC）是 **NP 中最难**的一类问题——它们同时满足：
1. **在 NP 中**：有多项式时间验证器
2. **NP-hard**：NP 中所有问题都能规约到它

**NPC 的关键性质**：
若任意一个 NPC 问题有多项式时间算法，则 **P = NP**（所有 NPC 问题都能高效解决）。这是为什么 NPC 是复杂性理论的核心！

**为什么 NPC 是"最难"的？**：

想象 NP 是一个王国，里面有各种问题。NPC 问题是这个王国中"领袖"级别的问题：征服任何一个领袖，整个王国都被征服。但在 P≠NP 的假设下，这些领袖全都坚不可摧。

<div data-component="PNPDecisionTree"></div>

---

## 41.3 多项式时间规约

### 41.3.1 规约的定义：把一个问题"变身"成另一个

**多项式时间多对一规约（Polynomial-time Many-one Reduction）**，记作 $A \leq_p B$：

$$A \leq_p B \iff \exists \text{ 多项式时间函数 } f, \forall x: \quad x \in A \iff f(x) \in B$$

可以理解为：存在一台在多项式时间内运行的**转换机（Transducer）**，它把 $A$ 的任意输入 $x$ 转化为 $B$ 的一个输入 $f(x)$，使得"$x$ 是 $A$ 的 yes 实例"当且仅当"$f(x)$ 是 $B$ 的 yes 实例"。

**图示**：

```
┌─────────────────────────────────────────────────────────┐
│  问题 A 的实例 x                                         │
│       │                                                  │
│       ▼ 多项式时间转换函数 f（规约算法）                │
│       │                                                  │
│  问题 B 的实例 f(x)                                     │
│       │                                                  │
│       ▼ B 的求解器                                      │
│                                                          │
│  f(x) ∈ B  →  x ∈ A（Yes）                             │
│  f(x) ∉ B  →  x ∉ A（No）                              │
└─────────────────────────────────────────────────────────┘
```

**规约的直觉**：$A \leq_p B$ 表示"$A$ 不比 $B$ 难"——有了 $B$ 的求解器，我就能先把 $A$ 的实例转换成 $B$ 的实例，然后用 $B$ 的求解器解决，整个过程仍在多项式时间内。

> **⚠️ 方向陷阱（极其高频面试考点！）**
>
> $A \leq_p B$ 读作"**$A$ 规约到 $B$**"或"**$A$ 不比 $B$ 难**"。
>
> - 方向：**箭头指向更难的问题**：$A \to B$ 意味着 $B$ 至少和 $A$ 一样难
> - **如果 $B$ 有多项式算法，则 $A$ 也有**（反推）
> - **如果 $A$ 没有多项式算法，则 $B$ 也没有**（正推同理）
>
> 很多人搞错方向：$A \leq_p B$ **不是**"$A$ 比 $B$ 难"，而是"$A$ 的难度不超过 $B$"，即 **B 至少和 A 一样难**。

---

### 41.3.2 规约的传递性

**定理**：若 $A \leq_p B$ 且 $B \leq_p C$，则 $A \leq_p C$。

**证明**：设 $f$ 是 $A \leq_p B$ 的规约函数，$g$ 是 $B \leq_p C$ 的规约函数，两者均多项式时间可计算。构造 $h(x) = g(f(x))$：

- $h$ 多项式时间可计算（多项式的复合仍是多项式）
- $x \in A \iff f(x) \in B \iff g(f(x)) \in C \iff h(x) \in C$

因此 $h$ 是 $A \leq_p C$ 的规约函数。$\square$

**意义**：规约的传递性使我们能建立**规约链**。只需证明 $A \leq_p B_1 \leq_p B_2 \leq_p \cdots \leq_p B_k$，就能得出 $A \leq_p B_k$。这是构建 NPC 版图的核心工具。

---

### 41.3.3 NPC 证明的标准两步框架

证明问题 $X$ 是 NP-complete，需要：

**步骤一：证明 $X \in NP$**

构造一个多项式时间验证器 $V(x, c)$：
- 指定"证书"$c$ 的形式（通常是候选解的紧凑描述）
- 证明 $V$ 在多项式时间内运行
- 证明：$x$ 是 yes 实例 ⟺ 存在 $c$ 使 $V(x, c)$ 接受

**步骤二：证明 $X$ 是 NP-hard（选一个已知 NPC 问题 $Y$，证明 $Y \leq_p X$）**

- 选择合适的"跳板"问题 $Y$（已知是 NPC 的）
- 构造多项式时间转换函数 $f$，将 $Y$ 的实例映射为 $X$ 的实例
- 证明**正确性**：$y \in Y \iff f(y) \in X$（双向：Yes 侧和 No 侧）

> **常见策略**：正确性证明往往是最难的部分。技巧是：先设计出"直觉上正确"的 $f$，再严格证明两个方向。

---

### 41.3.4 NPC 规约链的全局图景

几十年来，理论计算机科学家构建了一张庞大的 NPC 问题规约图，数以千计的问题都被证明是 NPC 的。以下是核心骨干链：

```
SAT (Cook-Levin 1971)
 │
 ▼
3-SAT
 │        ╔══════════════╗
 ├────────▶  INDEPENDENT SET ──── CLIQUE
 │        ╚══════════════╝
 │               │
 │               ▼
 ├──────── VERTEX COVER ──────── DOMINATING SET
 │               │
 │               ▼
 │           HAM-CYCLE ──────── TSP
 │
 ├────────▶ SUBSET SUM ──────── PARTITION ──── 0/1 KNAPSACK
 │
 └────────▶ GRAPH COLORING (k ≥ 3)
```

---

## 41.4 Cook-Levin 定理：SAT 是 NPC 的

### 41.4.1 SAT 问题的定义

**布尔可满足性问题（SAT -- Boolean Satisfiability Problem）**：

**输入**：一个布尔公式 $\phi$，变量为 $x_1, x_2, \ldots, x_n$，由 $\land$（AND）、$\lor$（OR）、$\lnot$（NOT）组成，通常写成**合取范式（CNF）**——若干"子句"的合取，每个子句是若干"文字"的析取：

$$\phi = (x_1 \lor \lnot x_2 \lor x_3) \land (\lnot x_1 \lor x_4) \land (x_2 \lor \lnot x_3 \lor \lnot x_4)$$

**输出**：是否存在变量赋值（每个 $x_i \in \{0, 1\}$）使 $\phi$ 为真（即每个子句都有至少一个满足的文字）？

**为什么 SAT 重要**：SAT 是第一个在电路设计、形式验证、人工智能规划、密码分析中广泛应用的 NP 问题。现代 **SAT solver**（如 MiniSat、Z3）能在实践中解决数百万变量的工业规模 SAT 实例（尽管最坏情况指数）。

#### 代码实现：暴力 SAT 求解（指数时间，仅用于理解）

```python
from itertools import product
from typing import List, Tuple

# CNF 表示：子句列表，每个子句是文字列表
# 文字用正数 k 表示 x_k，负数 -k 表示 ¬x_k
CNF = List[List[int]]

def sat_brute_force(n: int, clauses: CNF) -> List[int] | None:
    """
    暴力枚举所有 2^n 个变量赋值，找到满足所有子句的赋值。
    
    参数：
        n: 变量数量（变量编号 1 到 n）
        clauses: CNF 子句列表，文字用有符号整数表示
    返回：
        满足赋值（长度 n+1 的列表，assignment[i] 为 x_i 的值）
        或 None（不可满足）
    
    时间复杂度：O(2^n * |φ|)  ← 指数！仅用于理解原理
    
    【边界条件】：空公式（无子句）→ 可满足（返回全 0 赋值）
    """
    def evaluate_clause(clause: List[int], assignment: List[int]) -> bool:
        """评估单个子句：至少有一个文字为真即可"""
        for lit in clause:
            var = abs(lit)
            val = assignment[var]
            # 正文字 lit > 0 时值为 val；负文字 lit < 0 时值为 1-val
            if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                return True
        return False
    
    def evaluate_formula(clauses: CNF, assignment: List[int]) -> bool:
        """评估整个公式：所有子句都为真才满足"""
        return all(evaluate_clause(c, assignment) for c in clauses)
    
    # 枚举所有 2^n 个赋值（0 位用来对齐，变量下标从 1 开始）
    for bits in product([0, 1], repeat=n):
        assignment = [0] + list(bits)  # assignment[1..n]
        if evaluate_formula(clauses, assignment):
            return assignment
    
    return None  # UNSAT

# ---- 示例 ----
# φ = (x1 ∨ ¬x2) ∧ (¬x1 ∨ x2) ∧ (x1 ∨ x2)
clauses = [[1, -2], [-1, 2], [1, 2]]
result = sat_brute_force(2, clauses)
print(result)   # [0, 1, 1]  → x1=1, x2=1 满足
# 即 (1 ∨ 0) ∧ (0 ∨ 1) ∧ (1 ∨ 1) = True ✓

# 不可满足的公式：(x1) ∧ (¬x1)
unsat = sat_brute_force(1, [[1], [-1]])
print(unsat)    # None（UNSAT）
```

```cpp
#include <bits/stdc++.h>
using namespace std;

// CNF 公式：子句列表，每个子句是文字列表
// 文字：正整数 k 表示 x_k，负整数 -k 表示 ¬x_k
// 变量编号 1-indexed
using CNF = vector<vector<int>>;

// 评估单个子句
bool evalClause(const vector<int>& clause, const vector<int>& assign) {
    for (int lit : clause) {
        int var = abs(lit);
        // assign[var] == 1 表示 x_var = true
        bool val = (lit > 0) ? (assign[var] == 1) : (assign[var] == 0);
        if (val) return true;  // 子句中任一文字为真即可
    }
    return false;  // 所有文字均为假
}

// 评估整个公式：所有子句都必须为真
bool evalFormula(const CNF& clauses, const vector<int>& assign) {
    for (const auto& c : clauses)
        if (!evalClause(c, assign)) return false;
    return true;
}

/**
 * 暴力 SAT 求解：枚举所有 2^n 种赋值
 * 
 * @param n       变量数（1 到 n）
 * @param clauses CNF 公式
 * @return        满足赋值（1-indexed 向量）或空向量（UNSAT）
 * 
 * 时间复杂度：O(2^n × |φ|)  — 仅用于理解，实际不可用于大规模 SAT
 * 
 * 【边界条件】：n = 0 或 clauses 为空时直接返回全 0 赋值（可满足）
 */
vector<int> satBruteForce(int n, const CNF& clauses) {
    if (clauses.empty()) return vector<int>(n + 1, 0);
    
    // 枚举 2^n 种赋值（位操作）
    for (int mask = 0; mask < (1 << n); mask++) {
        vector<int> assign(n + 1);
        for (int i = 1; i <= n; i++)
            assign[i] = (mask >> (i - 1)) & 1;  // 第 i 位
        
        if (evalFormula(clauses, assign))
            return assign;  // 找到满足赋值
    }
    return {};  // UNSAT
}

int main() {
    // φ = (x1 ∨ ¬x2) ∧ (¬x1 ∨ x2) ∧ (x1 ∨ x2)
    CNF phi = {{1, -2}, {-1, 2}, {1, 2}};
    auto res = satBruteForce(2, phi);
    
    if (res.empty()) {
        cout << "UNSAT" << endl;
    } else {
        for (int i = 1; i <= 2; i++)
            cout << "x" << i << " = " << res[i] << "\n";
        // x1 = 1, x2 = 1
    }
    
    // UNSAT 示例：(x1) ∧ (¬x1)
    auto unsat = satBruteForce(1, {{1}, {-1}});
    cout << (unsat.empty() ? "UNSAT" : "SAT") << endl;  // UNSAT
    return 0;
}
```

---

### 41.4.2 Cook-Levin 定理：SAT 是 NPC 的第一个证明

**定理（Cook, 1971；Levin, 1973）**：SAT 是 NP-complete 的。

这是理论计算机科学史上最重要的定理之一。它建立了 NPC 类的"诞生证明"——在 SAT 之前，我们甚至不知道 NPC 类是否非空！

#### 第一步：SAT ∈ NP（容易）

给定公式 $\phi$ 和赋值 $a = (a_1, a_2, \ldots, a_n) \in \{0,1\}^n$，在 $O(|\phi|)$ 时间内代入求值即可验证。证书长度 $= n \leq |\phi|$，故 SAT ∈ NP。$\square$

#### 第二步：SAT 是 NP-hard（核心，困难！）

**证明思路**：任取 NP 中的语言 $L$，需证 $L \leq_p$ SAT。

$L \in NP$ 意味着存在多项式时间验证器 $V$ 和多项式 $p$。对任意输入 $x$（长度 $n = |x|$），我们构造一个 CNF 公式 $\phi$，使得：

$$\exists c, |c| \leq p(n): V(x, c) \text{ 接受} \iff \phi \text{ 可满足}$$

**编码思路**：将验证器 $V$ 在输入 $(x, c)$ 上的整个计算过程**编码为 CNF**：

- $V$ 是一台 DTM，在 $p(n)$ 步内完成计算
- 计算过程可以用一个 $p(n) \times p(n)$ 大小的"计算表格"表示：行 = 时间步，列 = 纸带格，每格存储一个符号
- 布尔变量用来描述每个时间步、每个位置、每种可能状态/符号的组合（$O(p(n)^2)$ 个变量）
- 图灵机转移规则、边界条件、接受条件，全部编码成 CNF 子句（每条规则变成子句，每个子句常数长度）

最终 $\phi$ 的大小是 $O(p(n)^2)$（多项式），构造时间也是多项式。$\square$

**意义的深刻性**：
- Cook-Levin 定理说明：任意 NP 问题都能"被 SAT 吸收"，SAT 本质上就是在模拟通用 NTM 的一步计算
- 从那之后，证明其他问题是 NPC，只需证明 $\text{SAT} \leq_p X$（或通过链式规约），远比重新证明 NP-hard 容易

---

### 41.4.3 从 SAT 到 3-SAT：规约示例

**3-SAT** 是 SAT 的特殊形式，要求每个子句**恰好有 3 个文字**。

**定理**：SAT $\leq_p$ 3-SAT（即 3-SAT 也是 NPC 的）。

**规约方案**：将 SAT 的每个子句转化为等价的若干 3-文字子句，通过引入辅助变量。

设子句 $C = (\ell_1 \lor \ell_2 \lor \cdots \lor \ell_k)$：

**Case 1（$k = 1$）**：$C = (\ell_1)$

引入两个新变量 $y, z$，替换为：
$$(\ell_1 \lor y \lor z) \land (\ell_1 \lor y \lor \lnot z) \land (\ell_1 \lor \lnot y \lor z) \land (\ell_1 \lor \lnot y \lor \lnot z)$$

**分析**：4 个子句，$y, z$ 任意取值，4 个子句有一个满足 ⟺ $\ell_1 = 1$。✓

**Case 2（$k = 2$）**：$C = (\ell_1 \lor \ell_2)$

引入一个新变量 $y$，替换为：
$$(\ell_1 \lor \ell_2 \lor y) \land (\ell_1 \lor \ell_2 \lor \lnot y)$$

**分析**：$y$ 取任何值，两个子句都满足 ⟺ $\ell_1 \lor \ell_2 = 1$。✓

**Case 3（$k = 3$）**：$C = (\ell_1 \lor \ell_2 \lor \ell_3)$

直接保留，无需修改。✓

**Case 4（$k \geq 4$）**：$C = (\ell_1 \lor \ell_2 \lor \cdots \lor \ell_k)$

引入 $k-3$ 个新变量 $y_1, y_2, \ldots, y_{k-3}$，分裂为：

$$(\ell_1 \lor \ell_2 \lor y_1)$$
$$(\lnot y_1 \lor \ell_3 \lor y_2)$$
$$(\lnot y_2 \lor \ell_4 \lor y_3)$$
$$\vdots$$
$$(\lnot y_{k-4} \lor \ell_{k-2} \lor y_{k-3})$$
$$(\lnot y_{k-3} \lor \ell_{k-1} \lor \ell_k)$$

**正确性证明（双向）**：

→ 若原子句满足（$\exists \ell_i = 1$），则从左向右，在 $\ell_i$ 所在子句处让 $y_{i-2} = 0$（此后所有 $y$ 令其使余下子句满足）。

← 若所有分裂子句都满足但原子句不满足，则所有 $\ell_i = 0$。第一个子句强制 $y_1 = 1$；传播下去每个子句强制 $y_j = 1$；但最后一个子句 $(\lnot y_{k-3} \lor \ell_{k-1} \lor \ell_k)$ 在所有 $\ell_i = 0$ 且 $y_{k-3} = 1$ 时为假，矛盾！

<div data-component="SATto3SATReduction"></div>

#### 代码实现：SAT → 3-SAT 规约

```python
from typing import List, Tuple

CNF = List[List[int]]

def sat_to_3sat(n: int, clauses: CNF) -> Tuple[int, CNF]:
    """
    SAT → 3-SAT 规约：将任意 CNF 转化为每子句恰好 3 个文字的 CNF。
    
    参数：
        n: 原始变量数（变量 1 到 n）
        clauses: 原始 CNF 子句列表
    返回：
        (新变量数, 新 CNF 子句列表)
    
    时间与空间复杂度：O(|φ|)（每个子句最多 4 个新子句，新增变量数 ≤ |子句数|）
    
    【设计考量】：变量编号从 n+1 开始，避免与原变量冲突
    """
    new_clauses: CNF = []
    next_var = n + 1  # 新增辅助变量编号从 n+1 开始
    
    for clause in clauses:
        k = len(clause)
        
        if k == 1:
            # Case 1: 单文字子句，用 y, z 填充
            y, z = next_var, next_var + 1
            next_var += 2
            new_clauses += [
                [clause[0],  y,  z],
                [clause[0],  y, -z],
                [clause[0], -y,  z],
                [clause[0], -y, -z],
            ]
        
        elif k == 2:
            # Case 2: 两文字子句，引入一个 y
            y = next_var
            next_var += 1
            new_clauses += [
                [clause[0], clause[1],  y],
                [clause[0], clause[1], -y],
            ]
        
        elif k == 3:
            # Case 3: 恰好 3 个文字，直接保留
            new_clauses.append(list(clause))
        
        else:
            # Case 4: k >= 4，链式分裂
            # 引入 k-3 个辅助变量 y[1..k-3]
            y = list(range(next_var, next_var + k - 3))
            next_var += k - 3
            
            # 第一个子句：(l1, l2, y1)
            new_clauses.append([clause[0], clause[1], y[0]])
            
            # 中间子句：(-y[i], l[i+2], y[i+1])，共 k-4 个
            for i in range(len(y) - 1):
                new_clauses.append([-y[i], clause[i + 2], y[i + 1]])
            
            # 最后一个子句：(-y[k-4], l[k-1], l[k])
            new_clauses.append([-y[-1], clause[-2], clause[-1]])
    
    return next_var - 1, new_clauses

# ---- 示例 ----
# 原始子句：(x1 ∨ x2 ∨ x3 ∨ x4)（4 个文字需要拆分）
n_vars = 4
original = [[1, 2, 3, 4]]  

new_n, new_clauses = sat_to_3sat(n_vars, original)
print(f"新变量数: {new_n}")      # 5（原4 + 新辅助变量 y1）
print(f"新子句数: {len(new_clauses)}")  # 3
for c in new_clauses:
    print(c)
# [1, 2, 5]
# [-5, 3, 4]    ← 链式结构
```

```cpp
#include <bits/stdc++.h>
using namespace std;

using CNF = vector<vector<int>>;

/**
 * SAT → 3-SAT 规约
 * 
 * @param n       原始变量数（变量 1 到 n）
 * @param clauses 原始 CNF 子句
 * @return        {新变量数, 新 CNF} — 每个子句恰好 3 个文字
 * 
 * 时间/空间：O(|φ|)
 * 新变量从 n+1 开始编号，保证无冲突
 * 
 * 【内存注意】：new_clauses 预留空间避免频繁扩容
 */
pair<int, CNF> satTo3SAT(int n, const CNF& clauses) {
    CNF newClauses;
    newClauses.reserve(clauses.size() * 4);  // 最坏每子句 4 个新子句
    int nextVar = n + 1;
    
    for (const auto& clause : clauses) {
        int k = clause.size();
        
        if (k == 1) {
            // 单文字：填充两个辅助变量 y, z
            int y = nextVar++, z = nextVar++;
            newClauses.push_back({clause[0],  y,  z});
            newClauses.push_back({clause[0],  y, -z});
            newClauses.push_back({clause[0], -y,  z});
            newClauses.push_back({clause[0], -y, -z});
        }
        else if (k == 2) {
            // 双文字：填充一个辅助变量 y
            int y = nextVar++;
            newClauses.push_back({clause[0], clause[1],  y});
            newClauses.push_back({clause[0], clause[1], -y});
        }
        else if (k == 3) {
            // 三文字：直接保留
            newClauses.push_back(clause);
        }
        else {
            // k >= 4：链式分裂，引入 k-3 个辅助变量
            // 【核心逻辑】：用辅助变量传递"是否有文字为真"的信息
            int startVar = nextVar;
            nextVar += k - 3;  // 一次性分配 k-3 个新变量
            
            // 第一个 3-文字子句：(l[0], l[1], y[startVar])
            newClauses.push_back({clause[0], clause[1], startVar});
            
            // 中间子句：(-y[j], l[j+2], y[j+1])
            for (int j = startVar; j < nextVar - 1; j++) {
                // j 对应 y 变量编号，clause[j - startVar + 2] 是对应原文字
                newClauses.push_back({-j, clause[j - startVar + 2], j + 1});
            }
            
            // 最后一个子句：(-y[nextVar-1], l[k-2], l[k-1])
            newClauses.push_back({-(nextVar - 1), clause[k - 2], clause[k - 1]});
        }
    }
    
    return {nextVar - 1, newClauses};
}

int main() {
    // (x1 ∨ x2 ∨ x3 ∨ x4) 的规约
    CNF phi = {{1, 2, 3, 4}};
    auto [newN, newClauses] = satTo3SAT(4, phi);
    
    cout << "新变量数: " << newN << "\n";  // 5
    cout << "新子句:\n";
    for (const auto& c : newClauses) {
        for (int lit : c) cout << lit << " ";
        cout << "\n";
    }
    // 1 2 5
    // -5 3 4
    return 0;
}
```

---

## 41.5 经典 NPC 问题与规约链

### 41.5.1 独立集、顶点覆盖与团：三兄弟的互补关系

#### 独立集（Independent Set，IS）

**定义**：给定无向图 $G = (V, E)$ 和整数 $k$，问是否存在大小 $\geq k$ 的**独立集**（任意两顶点间无边的顶点子集）？

**在现实中的意义**：
- 无线网信道分配：相互干扰的发射器（连边）中，选出最多互不干扰的子集
- 竞争关系建模：给定"竞争对手图"，选最大的"和平团体"

#### 顶点覆盖（Vertex Cover，VC）

**定义**：给定图 $G = (V, E)$ 和整数 $k$，问是否存在大小 $\leq k$ 的**顶点覆盖**（每条边至少有一个端点在覆盖集中）？

#### 团（Clique）

**定义**：给定图 $G = (V, E)$ 和整数 $k$，问是否存在大小 $\geq k$ 的**完全子图**（团中所有顶点两两相连）？

#### 三者的互补关系

**定理 1**：$S$ 是 $G$ 的独立集 $\iff$ $V \setminus S$ 是 $G$ 的顶点覆盖。

**证明**：
→ 设 $S$ 是独立集，任取边 $(u, v)$。若 $u, v$ 都在 $S$ 中，则 $S$ 内有边，矛盾！故 $u \in V \setminus S$ 或 $v \in V \setminus S$，即 $V \setminus S$ 覆盖了这条边。

← 设 $C = V \setminus S$ 是顶点覆盖，取 $u, v \in S$（即 $u, v \notin C$）。若 $(u, v) \in E$，则 $C$ 未覆盖该边，矛盾！故 $S$ 中无边，是独立集。$\square$

**推论**：$G$ 有大小 $k$ 的独立集 $\iff$ $G$ 有大小 $|V| - k$ 的顶点覆盖。

因此：$\text{IS} \leq_p \text{VC}$ 且 $\text{VC} \leq_p \text{IS}$（两者多项式等价！）

**定理 2**：$S$ 是 $G$ 的团 $\iff$ $S$ 是 $\bar{G}$（$G$ 的补图）的独立集。

**证明**：$G$ 的团 = $S$ 中两两有边 ↔ $\bar{G}$ 中两两无边 = $\bar{G}$ 的独立集。$\square$

<div data-component="VertexCoverNPCProof"></div>

#### 3-SAT 规约到独立集

**定理**：3-SAT $\leq_p$ 独立集。

**构造**：给定 3-SAT 公式 $\phi$，含 $m$ 个子句 $C_1, \ldots, C_m$，每个子句 3 个文字：

1. **节点**：为每个子句的每个文字创建一个节点——共 $3m$ 个节点
2. **边（两类）**：
   - **子句内边**：同一子句的 3 个节点两两相连（每子句是一个三角形）
   - **矛盾边**：若两个节点对应的文字**互补**（如 $x_i$ 和 $\lnot x_i$），则连一条边

3. **目标**：图中是否存在大小 $= m$ 的独立集？

**正确性**（$\phi$ 可满足 $\iff$ 图有 $m$ 大小的独立集）：

→ 给定满足赋值，每个子句至少有一个文字为真，从每个子句三角形中选一个为真的文字对应节点。选出的 $m$ 个节点：
  - 不含同一子句内两节点（因为每子句只选一个）
  - 不含互补文字节点（赋值一致，同一变量不能同时是 0 和 1）

故形成独立集。✓

← 给定大小 $m$ 的独立集 $S$，由于每个子句三角形内没有两个节点可以同时选（三角形内两两相连），每个子句恰好有一个节点在 $S$ 中。令这 $m$ 个节点对应的文字均为真（若变量赋值矛盾则无矛盾边保证不会出现），得到满足赋值。✓

---

### 41.5.2 顶点覆盖到哈密顿回路到 TSP

#### 哈密顿回路（Hamiltonian Cycle，HAM-CYCLE）

**定义**：给定图 $G$，是否存在经过每个顶点**恰好一次**的回路？

**日常比喻**：邮递员需要访问城市里的每条街道（欧拉回路问题）——这在多项式时间内可解（Fleury 算法）。但旅行商需要访问每个**城市**（顶点）恰好一次，这就是哈密顿问题——NPC！

**哈密顿路径 vs 哈密顿回路**：两者都是 NPC，可互相规约。

#### 这里省略完整的 VC → HAM-CYCLE 规约（极其技术性）

VC → HAM-CYCLE 的规约设计极为精巧（Karp 1972 规约），思路是：
- 为每条边 $(u, v) \in E$ 构造一个"14节点小工具"，工具的进出口对应顶点是否在覆盖中
- 把所有工具串联起来形成哈密顿路径的候选骨架
- 覆盖集中的顶点连接对应工具，确保整条路径存在

完整规约的细节见 CLRS 第4版 §34.5.3。

#### TSP（旅行商问题）决策版

**定义**：给定完全加权图 $G$（城市间两两有距离）和整数 $B$，是否存在总权重 $\leq B$ 的哈密顿回路（访问所有城市各一次的闭合路径）？

**HAM-CYCLE $\leq_p$ TSP 决策版**（Karp 1972，容易）：

**规约**：给定图 $G = (V, E)$（可能不完全），构造完全图 $G'$：
- 顶点集相同
- 若 $(u, v) \in E$：权重 = 0
- 若 $(u, v) \notin E$：权重 = 1

取 $B = 0$，询问是否存在总权重 $\leq 0$ 的哈密顿回路。

**正确性**：$G$ 有哈密顿回路 $\iff$ $G'$ 有总权重为 0 的哈密顿回路（只走原有的 0 权重边）。$\square$

---

### 41.5.3 3-SAT 规约到子集和（Subset Sum）

**子集和问题（SUBSET-SUM）**：给定正整数集合 $S = \{a_1, a_2, \ldots, a_n\}$ 和目标值 $T$，问是否存在子集使其元素之和等于恰好 $T$？

**定理**：3-SAT $\leq_p$ SUBSET-SUM。

**规约思路**（直觉）：

设 3-SAT 公式有 $n$ 个变量 $x_1, \ldots, x_n$ 和 $m$ 个子句 $C_1, \ldots, C_m$。构造 $2n + 2m$ 个整数（$2n$ 个对应变量赋值，$2m$ 个"松弛变量"对应子句）：

- 数字以 $(n + m)$ 位表示（十进制中每位区分变量和子句）
- 对变量 $x_i$：
  - $v_i$（赋值 1 时选）：第 $i$ 位为 1，第 $n+j$ 位为 1 当且仅当 $x_i$ 出现在 $C_j$ 的正文字中
  - $v'_i$（赋值 0 时选）：第 $i$ 位为 1，第 $n+j$ 位为 1 当且仅当 $\lnot x_i$ 出现在 $C_j$ 中
- 松弛变量 $s_j, s'_j$（1 和 2）：用于在子句贡献和为 1、2、3 时"补齐"到 4
- 目标 $T$：每个变量位为 1，每个子句位为 4

**直觉**：凑出目标 $T$ = 每个变量恰好选 $v_i$ 或 $v'_i$（对应赋值）+ 每个子句恰好有 1~3 个满足文字（用松弛补至 4）。若公式可满足 ↔ 子集和有解。

> 细节证明见 CLRS 第4版 §34.5.5；此处展示的核心思路是通过"位操作+十进制不进位"将布尔逻辑编码为整数加法。

<div data-component="SubsetSumReductionDemo"></div>

### 41.5.4 图着色（Graph Coloring）

**$k$-图着色问题**：给定图 $G$ 和整数 $k$，是否可以用 $k$ 种颜色对顶点着色，使得相邻顶点颜色不同？

- **$k = 1$**：当且仅当图中无边（平凡，P 中）
- **$k = 2$**：当且仅当图是二部图（P 中，BFS/DFS 验证）
- **$k \geq 3$**：NPC（3-SAT $\leq_p$ 3-COLORING）

**应用**：
- 考试时间表排列：科目 = 顶点，同一考生修的两门课 = 边，$k$ 色着色对应 $k$ 个时间段
- 编译器寄存器分配：变量 = 顶点，同时存活的两个变量 = 边，$k$ 色着色对应 $k$ 个寄存器
- 地图着色（四色定理：任何平面图均可 4 色着色——但找最优着色是 NPC 的

**3-SAT $\leq_p$ 3-COLORING（简略）**：

为每个变量 $x_i$ 建"真/假"节点，连一条"非边"表示互斥。加一个基准节点 B 与所有真/假节点相连（确保真/假节点颜色不同于 B 色）。为每个子句构造一个小工具（约 5 个节点），确保子句可满足 ↔ 工具可 3 着色。

---

### 41.5.5 NPC 规约关系图与整体结构

<div data-component="NPCompleteReductionMap"></div>

**规约链中各问题的复杂度总结**：

| 问题 | 是否在 NP | 是否 NPC | 最优近似 | 真实算法实践 |
|------|----------|----------|---------|------------|
| SAT | ✅ NP | ✅ | - | 现代 SAT solver（实践极快）|
| 3-SAT | ✅ NP | ✅ | - | 同上 |
| INDEPENDENT SET | ✅ NP | ✅ | $O(n/\log^2 n)$ 近似 | 仍困难 |
| VERTEX COVER | ✅ NP | ✅ | 2-近似（见 Ch42） | 2-近似实用 |
| CLIQUE | ✅ NP | ✅ | $n^{1-\varepsilon}$ 难以近似 | 困难 |
| HAM-CYCLE | ✅ NP | ✅ | - | 回溯 + 剪枝 |
| TSP（一般）| ✅ NP | ✅ | 不可近似（P≠NP） | Christofides 1.5（度量型）|
| SUBSET-SUM | ✅ NP | ✅ | FPTAS（见 Ch42） | DP 伪多项式 |
| 0/1 KNAPSACK | ✅ NP | NP-hard* | FPTAS | DP 伪多项式 |
| 3-COLORING | ✅ NP | ✅ | 近似难（APX-hard）| 回溯 |

*背包优化版是 NP-hard，但技术上决策版是 NPC

---

## 41.6 处理 NPC 问题的实践策略

即使面对 NPC 问题，工程实践中也有多种有效应对策略，这些策略将在 Chapter 42 详细展开，这里先做概述：

### 策略一：将输入限制在多项式子情形

许多 NPC 问题在特殊结构上变得高效可解：

| NPC 问题（一般）| 多项式时间特例 |
|--------------|-------------|
| 图着色（$k \geq 3$）| 平面图 → 4 色（可验证但找最优仍难）；树图 → 贪心 $O(V)$ |
| 独立集 | 树 → DP in $O(V)$；区间图 → 贪心 in $O(n \log n)$ |
| 0/1 背包 | 权重/价值均较小 → DP 伪多项式 |
| TSP | 度量 TSP（三角不等式）→ 1.5-近似 |
| SAT | 2-SAT → 强连通分量线性时间 |

### 策略二：近似算法（近似比保证）

接受"不最优但不太差"的解：顶点覆盖 2-近似、集合覆盖 $\ln n$-近似、Christofides 1.5-近似 TSP……

### 策略三：随机化算法

通过随机性在期望或高概率意义下得到好解：随机化 MAX-CUT 2-近似期望（$\geq \frac{1}{2}|E|$）。

### 策略四：FPT（固定参数可解）

当某个参数 $k$ 很小时，$O(f(k) \times \text{poly}(n))$ 是可接受的：如顶点覆盖有 $O(2^k \cdot k \cdot n)$ FPT 算法（分支界限思想）。

### 策略五：精确指数算法（枝剪）

接受指数时间，但通过精细剪枝使常数极小：如 TSP 的 Held-Karp DP 算法 $O(2^n n^2)$（当 $n \leq 25$ 时可接受）。

#### Held-Karp TSP 算法（经典 DP）

```python
import sys
from functools import lru_cache

def tsp_held_karp(dist: list[list[int]]) -> int:
    """
    Held-Karp 算法：精确求解 TSP（从节点 0 出发，访问所有节点各一次后返回 0）
    
    状态：dp[S][v] = 访问了顶点集合 S 且当前在 v 的最短路径
    
    参数：
        dist: n×n 距离矩阵（dist[i][j] 为 i 到 j 的距离，INF 表示不可达）
    返回：
        最短哈密顿回路长度（不存在时返回 INF）
    
    时间复杂度：O(2^n × n^2)
    空间复杂度：O(2^n × n)
    
    【边界条件】：n=1 返回 0（自身回路）；n=2 返回 2×dist[0][1]
    【实用限制】：n ≤ 20~25（超过此规模内存和时间都会爆炸）
    """
    n = len(dist)
    INF = float('inf')
    
    # dp[mask][v] = 已访问集合为 mask（按位表示），当前在顶点 v 时的最小路径长
    # 初始状态：从顶点 0 出发，已访问集合 = {0}，当前在 0
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0  # 集合 {0}，当前在 0，代价 0
    
    # 按集合大小从小到大枚举（确保子问题已解）
    for mask in range(1, 1 << n):
        for u in range(n):
            if not (mask >> u & 1): continue        # u 不在当前集合中，跳过
            if dp[mask][u] == INF: continue         # 此状态不可达
            
            # 从 u 扩展到下一个未访问的顶点 v
            for v in range(n):
                if mask >> v & 1: continue          # v 已访问，跳过
                new_mask = mask | (1 << v)          # 新集合加入 v
                new_dist = dp[mask][u] + dist[u][v]
                if new_dist < dp[new_mask][v]:
                    dp[new_mask][v] = new_dist      # 更新最短路
    
    # 所有点都已访问（mask = 全集），从任意 v 回到 0 的最短路
    full = (1 << n) - 1
    return min(dp[full][v] + dist[v][0] for v in range(n) if dp[full][v] < INF)

# ---- 示例 ----
INF = float('inf')
# 4 个城市的距离矩阵
d = [
    [0,  10, 15, 20],
    [10,  0, 35, 25],
    [15, 35,  0, 30],
    [20, 25, 30,  0],
]
print(tsp_held_karp(d))  # 80（路径 0→1→3→2→0: 10+25+30+15=80）
```

```cpp
#include <bits/stdc++.h>
using namespace std;

/**
 * Held-Karp 算法：精确求解 TSP
 * 
 * @param dist n×n 距离矩阵
 * @return 最短哈密顿回路长度
 * 
 * 时间：O(2^n × n^2)，空间：O(2^n × n)
 * 实用范围：n ≤ 20（内存约 20 MB，n=25 时约 12 GB 不可行）
 * 
 * 【实现细节】：
 * - 用 BIT MASK 表示顶点子集（第 i 位为 1 表示 i 已访问）
 * - dp 表初始化为 INF，用 bitwise 操作快速判断成员关系
 * - 【内存优化】：可以用滚动数组优化空间（层序 DP），但代码复杂度上升
 */
int tspHeldKarp(const vector<vector<int>>& dist) {
    int n = dist.size();
    const int INF = 1e9;
    int fullMask = (1 << n) - 1;
    
    // dp[mask][v]：已访问集合 mask，当前在 v 的最短路
    vector<vector<int>> dp(1 << n, vector<int>(n, INF));
    dp[1][0] = 0;  // 从顶点 0 出发，集合为 {0}，代价 0
    
    // 按集合大小枚举（BFS 顺序）
    for (int mask = 1; mask <= fullMask; mask++) {
        for (int u = 0; u < n; u++) {
            if (!(mask >> u & 1) || dp[mask][u] == INF) continue;
            
            // 向未访问的顶点 v 扩展
            for (int v = 0; v < n; v++) {
                if (mask >> v & 1) continue;  // 已访问，跳过
                int newMask = mask | (1 << v);
                int newDist = dp[mask][u] + dist[u][v];
                // 【易错点】：确保 dist[u][v] != INF 时再更新（避免溢出）
                if (dist[u][v] < INF && newDist < dp[newMask][v])
                    dp[newMask][v] = newDist;
            }
        }
    }
    
    // 回到起点 0 的最短回路
    int ans = INF;
    for (int v = 1; v < n; v++) {  // v=0 不需要（自环）
        if (dp[fullMask][v] < INF && dist[v][0] < INF)
            ans = min(ans, dp[fullMask][v] + dist[v][0]);
    }
    return ans;
}

int main() {
    vector<vector<int>> d = {
        {0,  10, 15, 20},
        {10,  0, 35, 25},
        {15, 35,  0, 30},
        {20, 25, 30,  0},
    };
    cout << tspHeldKarp(d) << "\n";  // 80
    return 0;
}
```

---

## 本章小结

| 概念 | 定义 | 关键点 |
|------|------|-------|
| 图灵机 | 计算的正式模型，移动读写头 + 有限状态转移 | DTM（确定）vs NTM（非确定）|
| 决策问题 | 输出 Yes/No 的问题 | 等价于语言识别 |
| P | 多项式时间确定性可解 | 包含 BFS/DFS/Dijkstra/LP |
| NP | 多项式时间可验证（证书存在）| ≠ "Non-Polynomial"！|
| co-NP | NP 的补 | 素性测试在 co-NP 也在 NP |
| NP-hard | NP 中所有问题可规约到此 | 不要求自身在 NP |
| NPC | NP ∩ NP-hard | SAT 是第一个（Cook-Levin）|
| $A \leq_p B$ | A 多项式规约到 B | B 至少和 A 一样难 |
| 3-SAT | SAT 每子句 3 文字 | 规约链起点 |
| Held-Karp | TSP 精确 DP | $O(2^n n^2)$，$n \leq 20$ |

> **思考题**：
> 1. 如果 P = NP，为什么 RSA 公钥加密会"立刻"不安全？（提示：整数分解 ∈ NP，若 P=NP，它也 ∈ P）
> 2. 一元编码下，背包问题的暴力算法看起来是"多项式时间"的——为什么这不意味着背包问题 ∈ P？
> 3. 停机问题是 NP-hard 的，但它不在 NP 中。你能解释为什么吗？
> 4. 为什么 3-SAT → IS 的规约构造中，矩阵内用"三角形"（子句内三两相连），而不是只用链接相邻文字的边？

**扩展阅读**：
- CLRS 第4版 Chapter 34（计算复杂性）
- Dasgupta, Papadimitriou, Vazirani《Algorithms》Chapter 8（在线免费）
- Sipser《Introduction to the Theory of Computation》第3版（最清晰的复杂性理论教材）
- Cook 1971 原始论文："The Complexity of Theorem Proving Procedures"
- Arora & Barak《Computational Complexity: A Modern Approach》（进阶研究生参考）

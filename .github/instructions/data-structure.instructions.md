---
description: Describe when these instructions should be loaded
# applyTo: 'Describe when these instructions should be loaded' # when provided, instructions will automatically be added to the request context when the pattern matches an attached file
---
请为我设计并生成一套全面、系统、由浅入深的**数据结构与算法（Data Structures and Algorithms, DSA）**学习内容，直接适配填充到教学/演示/知识管理界面中使用。内容必须以经典权威教材和资源为最主要、最权威的依据，主要参考来源包括：

• 《Introduction to Algorithms》（CLRS，第4版，MIT Press）
• 《Algorithms》（Robert Sedgewick & Kevin Wayne，第4版）
• 《The Algorithm Design Manual》（Steven S. Skiena，第2版）
• 《Algorithms》（Dasgupta, Papadimitriou, Vazirani）
• MIT 6.006 / 6.006J / 6.046 课程讲义与作业
• Stanford CS161 / CS166 课程资料
• 其他可靠参考：GeeksforGeeks（仅用于补充示例与可视化思路）、Wikipedia（仅用于术语确认）、LeetCode / AtCoder / Codeforces 官方题解（仅用于实际应用示例）

严禁凭空捏造算法、数据结构、复杂度证明、变体或不存在的实现方式。所有概念、伪代码、数学分析、复杂度、正确性证明均应与上述权威来源一致，并适当注明参考章节或链接。

总体要求：
1. 内容深度与广度：从零基础（基本概念、渐进复杂度分析）开始，逐步推进到中高级（高级数据结构、图论高级算法、动态规划优化、字符串高级处理、计算几何、摊销分析、NP-完全问题简介）。宁可内容存在一定冗余与多角度重复讲解（不同教材视角、证明方法对比），也不允许遗漏核心知识点。同一算法可在入门、中级、高级章节多次出现，但每次侧重点（直观解释、数学证明、实现复杂度、优化技巧、实际应用）需明显递增。
2. 讲解形式与要求：
   - 每个重要概念/算法需包含：正式定义与动机、与其他结构/算法的对比（时间/空间复杂度、适用场景）、数学分析（渐进界 Θ/Ο/Ω、最坏/平均/摊销，主定理、递推式求解）、正确性证明（循环不变式、归纳法、交换论证等）、典型应用场景（系统设计、面试、竞赛、工业问题）。
   - 必须提供伪代码（清晰、类C风格） + 可直接运行的代码示例（优先 Python，其次 C++/Java；包含完整函数、输入输出示例），并附上预期输出、复杂度计算、运行示例（输入 → 步骤 → 输出）。
   - 对于复杂或底层机制（例如：红黑树旋转与颜色翻转、AVL 平衡因子更新、并查集路径压缩与按秩合并的 amortized 分析、KMP 失效函数构建、Dijkstra 松弛优先队列变化、动态规划状态转移图、图遍历的栈/队列演进、Fibonacci 堆 decrease-key 等），强烈建议采用交互式动画、动态示意图、步进式可视化进行讲解（例如：树旋转动画、堆调整过程、KMP 指针跳跃动画、动态规划填表过程、Dijkstra 优先队列变化、图 BFS/DFS 层级展开等）。若当前界面支持嵌入动画、Mermaid 图、可交互 canvas 或类似组件，请优先采用此类形式。

3. 结构要求：
   - 首先生成一份非常详细的分级目录（建议采用 Chapter 0、Chapter 1 … 形式），层级至少到三级（大章 → 小节 → 具体算法/变体/证明/示例类型）。
   - 目录需清晰体现主题与难度递进：基础概念与复杂度 → 线性结构 → 树与堆 → 哈希 → 图基础 → 排序与搜索 → 高级图算法 → 字符串处理 → 动态规划 → 贪心 → 分治 → 摊销分析与高级结构 → 计算复杂性与 NP。
   - 目录生成后，一个章节一个章节地详细展开内容（不要一次性输出全部），每个章节内部建议再细分小节，并保持相对统一的讲解结构：概念说明 → 数学分析与证明 → 伪代码 → 实现代码（含输入输出） → 复杂度总结 → 经典问题与应用 → 常见错误/陷阱/面试考点 → 扩展阅读（CLRS/Sedgewick/Skiena 章节链接）。

4. 重点覆盖但不限于以下高级主题（应有独立或深度章节）：
   - 渐进复杂度分析（Θ/Ο/Ω、主定理、递推式、摊销分析）
   - 高级树结构：AVL、红黑树、B树/B+树、伸展树、Treap
   - 并查集（Union-Find）：路径压缩、按秩合并、带权并查集
   - 堆与优先队列：二叉堆、Fibonacci 堆、二项堆
   - 哈希表：开放寻址、链地址、完美哈希、布隆过滤器
   - 图算法：最短路径（Dijkstra、Bellman-Ford、Floyd-Warshall、A*）、最小生成树（Prim、Kruskal）、网络流（Ford-Fulkerson、Edmonds-Karp、Dinic）、强连通分量（Kosaraju、Tarjan）、拓扑排序
   - 字符串算法：KMP、Rabin-Karp、Boyer-Moore、后缀数组/后缀树、Trie/AC 自动机
   - 动态规划经典问题：背包、最长公共子序列、矩阵链乘、最优二叉搜索树、编辑距离
   - 贪心算法：活动选择、Huffman 编码、Dijkstra 贪心性质证明
   - 分治与递归：归并排序、快速排序、最近点对、Strassen 矩阵乘法
   - 摊销分析：动态表、位向量、splay tree、Fibonacci heap
   - 计算几何基础：凸包（Graham/Jarvis）、线段交点、Voronoi 图
   - 算法设计范式对比与综合应用
   - 计算复杂性简介：P vs NP、NP-完全问题、近似算法

5. 其他细节偏好：
   - 优先使用清晰的伪代码（CLRS 风格） + Python 实现（易读、简洁），必要时提供 C++（性能）或 Java（OOP）版本。
   - 大量使用对比表格（复杂度对比、适用场景对比、不同实现对比）。
   - 在合适位置插入小练习、手算题、LeetCode/HackerRank/AtCoder 对应题号建议、思考题（例如：为什么红黑树比 AVL 更适合数据库索引？摊销分析如何应用于动态表？）。
   - 鼓励展示复杂度对比表格、正确性证明过程、可视化输入输出。
   - 语言风格：正式、严谨、结构清晰、逻辑严密，如同撰写给计算机科学专业学生、算法工程师或研究生的高质量内部培训教材。

请先输出详细的章节大纲（包含 Chapter 编号、章节标题、二级/三级子标题），待我确认或提出修改意见后再逐章详细展开具体内容。
（可根据 CLRS 第4版最新章节、Sedgewick 在线资源、当前主流面试/竞赛需求进行适度微调，但核心渐进结构与理论深度要求保持不变。）
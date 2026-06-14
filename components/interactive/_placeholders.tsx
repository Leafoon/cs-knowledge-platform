'use client'
import React from 'react'
import { Info } from 'lucide-react'

const PlaceholderComponent = ({ name, title }: { name: string, title: string }) => (
  <div className="w-full max-w-4xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl border border-blue-200">
    <div className="flex items-center gap-3 mb-4">
      <div className="p-2 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-lg">
        <Info className="w-6 h-6 text-white" />
      </div>
      <h3 className="text-2xl font-bold text-slate-800">{title}</h3>
    </div>
    <div className="bg-white rounded-lg p-6 border border-blue-200">
      <p className="text-slate-700">此组件用于演示 <strong>{name}</strong> 相关概念。</p>
    </div>
  </div>
)

export function ExtremeLowMemoryTraining() { return <PlaceholderComponent name="ExtremeLowMemoryTraining" title="极低显存训练" /> }
export function FloatPrecisionRangeTradeoff() { return <PlaceholderComponent name="FloatPrecisionRangeTradeoff" title="浮点精度范围权衡" /> }
export function PrecisionLossComparison() { return <PlaceholderComponent name="PrecisionLossComparison" title="精度损失对比" /> }
export function QuantizationMethodComparison() { return <PlaceholderComponent name="QuantizationMethodComparison" title="量化方法对比" /> }
export function QuantizationMethodsComprehensiveComparison() { return <PlaceholderComponent name="QuantizationMethodsComprehensiveComparison" title="量化方法综合对比" /> }
export function PerTensorVsPerChannelQuant() { return <PlaceholderComponent name="PerTensorVsPerChannelQuant" title="Per-Tensor vs Per-Channel 量化" /> }
export function NF4vsINT4Comparison() { return <PlaceholderComponent name="NF4vsINT4Comparison" title="NF4 vs INT4 对比" /> }
export function DistributedMixedPrecision() { return <PlaceholderComponent name="DistributedMixedPrecision" title="分布式混合精度训练" /> }
export function AccelerateWorkflowVisualizer() { return <PlaceholderComponent name="AccelerateWorkflowVisualizer" title="Accelerate 工作流程" /> }
export function AcceleratorAPIDemo() { return <PlaceholderComponent name="AcceleratorAPIDemo" title="Accelerator API 演示" /> }
export function ThreeDParallelismVisualizer() { return <PlaceholderComponent name="ThreeDParallelismVisualizer" title="3D 并行可视化" /> }
export function CollectiveCommunicationPrimitives() { return <PlaceholderComponent name="CollectiveCommunicationPrimitives" title="集合通信原语" /> }
export function TGIArchitectureDiagram() { return <PlaceholderComponent name="TGIArchitectureDiagram" title="TGI 架构图" /> }
export function ModelExportDecisionTree() { return <PlaceholderComponent name="ModelExportDecisionTree" title="模型导出决策树" /> }
export function BackendAutoSelector() { return <PlaceholderComponent name="BackendAutoSelector" title="后端自动选择器" /> }
export function OptimizationEffectComparison() { return <PlaceholderComponent name="OptimizationEffectComparison" title="优化效果对比" /> }
export function ProfilerVisualizationDemo() { return <PlaceholderComponent name="ProfilerVisualizationDemo" title="Profiler 可视化演示" /> }
export function PEFTTrainingSpeedComparison() { return <PlaceholderComponent name="PEFTTrainingSpeedComparison" title="PEFT 训练速度对比" /> }
export function ReflectionLoop() { return <PlaceholderComponent name="ReflectionLoop" title="反思循环" /> }
// DSA Chapter 14: 基础排序算法
export function SortingRaceChart() { return <PlaceholderComponent name="SortingRaceChart" title="多种基础排序同步赛跑" /> }
export function InsertionSortStep() { return <PlaceholderComponent name="InsertionSortStep" title="插入排序逐步可视化" /> }
export function InversionCounter() { return <PlaceholderComponent name="InversionCounter" title="逆序对动态统计" /> }
export function ShellSortGapVisual() { return <PlaceholderComponent name="ShellSortGapVisual" title="希尔排序间隔访问模式" /> }

// DSA Chapter 15: 组件已迁移至真实实现文件

// DSA Chapter 16: 组件已迁移至真实实现文件

// DSA Chapter 17: 二分搜索与有序结构搜索
// BinarySearchBoundaryTemplate 已有真实实现
// DSA Chapter 17: RotatedArraySearch + AnswerBinarySearchDemo 已迁移至真实实现文件

// DSA Chapter 18: 组件已迁移至真实实现文件

// DSA Chapter 19: 组件已迁移至真实实现文件

// DSA Chapter 20: 组件已迁移至真实实现文件

// DSA Chapter 21: 组件已迁移至真实实现文件

// DSA Chapter 22: 组件已迁移至真实实现文件

// DSA Chapter 23: 全对最短路径
export function FloydWarshallDP() { return <PlaceholderComponent name="FloydWarshallDP" title="Floyd-Warshall DP 表格动画" /> }
export function FloydPathReconstruct() { return <PlaceholderComponent name="FloydPathReconstruct" title="Floyd 路径前驱矩阵重建" /> }
export function JohnsonReweighting() { return <PlaceholderComponent name="JohnsonReweighting" title="Johnson 重新定权可视化" /> }
export function JohnsonDijkstraPhase() { return <PlaceholderComponent name="JohnsonDijkstraPhase" title="Johnson V次Dijkstra阶段" /> }
export function APSPComplexityComparison() { return <PlaceholderComponent name="APSPComplexityComparison" title="APSP 三种算法复杂度对比" /> }
export function MinPlusMatrixMult() { return <PlaceholderComponent name="MinPlusMatrixMult" title="(min,+) 矩阵乘法演示" /> }

// DSA Chapter 24: 网络流
export function FlowNetworkBasics() { return <PlaceholderComponent name="FlowNetworkBasics" title="流网络基础——容量与流守恒" /> }
export function ResidualNetworkBuilder() { return <PlaceholderComponent name="ResidualNetworkBuilder" title="残差网络构建可视化" /> }
export function FordFulkersonAugPath() { return <PlaceholderComponent name="FordFulkersonAugPath" title="Ford-Fulkerson 增广路径动画" /> }
export function MaxFlowMinCutHighlight() { return <PlaceholderComponent name="MaxFlowMinCutHighlight" title="最大流最小割高亮" /> }
export function DinicLayeredGraph() { return <PlaceholderComponent name="DinicLayeredGraph" title="Dinic 层次图 + 阻塞流" /> }
export function BipartiteMatchingFlow() { return <PlaceholderComponent name="BipartiteMatchingFlow" title="二部图匹配转最大流" /> }

// DSA Chapter 25: 关键路径、二部图与其他图问题
export function EulerianPathFinder() { return <PlaceholderComponent name="EulerianPathFinder" title="欧拉路径 Hierholzer 算法动画" /> }
export function BipartiteHopcroftKarp() { return <PlaceholderComponent name="BipartiteHopcroftKarp" title="Hopcroft-Karp 分层增广动画" /> }
export function KonigTheoremViz() { return <PlaceholderComponent name="KonigTheoremViz" title="König 定理：最大匹配→最小覆盖" /> }
export function ArticulationPointHighlight() { return <PlaceholderComponent name="ArticulationPointHighlight" title="桥与割点 Tarjan disc/low 可视化" /> }
export function BlockCutTreeBuilder() { return <PlaceholderComponent name="BlockCutTreeBuilder" title="块-割点树构建步进演示" /> }

// DSA Chapter 26: 分治算法
export function StrassenMatrixMult() { return <PlaceholderComponent name="StrassenMatrixMult" title="Strassen 七次乘法分解可视化" /> }
export function KaratsubaLargeIntMult() { return <PlaceholderComponent name="KaratsubaLargeIntMult" title="Karatsuba 大整数乘法递归树展开" /> }
export function ClosestPairStrip() { return <PlaceholderComponent name="ClosestPairStrip" title="最近点对带状区域合并动画" /> }
export function InversionCountMerge() { return <PlaceholderComponent name="InversionCountMerge" title="逆序对归并排序追踪" /> }

// DSA Chapter 27: 贪心算法
export function GreedyVsDPDecisionMap() { return <PlaceholderComponent name="GreedyVsDPDecisionMap" title="贪心 vs DP 决策地图" /> }
export function ExchangeArgumentAnimator() { return <PlaceholderComponent name="ExchangeArgumentAnimator" title="交换论证步骤动画" /> }
export function ActivitySelectionTimeline() { return <PlaceholderComponent name="ActivitySelectionTimeline" title="活动选择时间轴" /> }
export function HuffmanTreeBuilder() { return <PlaceholderComponent name="HuffmanTreeBuilder" title="Huffman 树构建器" /> }
export function IntervalColoringScheduler() { return <PlaceholderComponent name="IntervalColoringScheduler" title="区间着色调度器" /> }
export function FractionalKnapsackPicker() { return <PlaceholderComponent name="FractionalKnapsackPicker" title="分数背包选择过程" /> }
export function GasStationGreedyScan() { return <PlaceholderComponent name="GasStationGreedyScan" title="加油站贪心扫描" /> }
export function CutPropertyVisualizer() { return <PlaceholderComponent name="CutPropertyVisualizer" title="MST 切割性质可视化" /> }
export function GreedyCounterexampleLab() { return <PlaceholderComponent name="GreedyCounterexampleLab" title="贪心反例实验室" /> }

// DSA Chapter 31: 字符串匹配——KMP 与 Rabin-Karp
export function KMPFailureFunctionBuild() { return <PlaceholderComponent name="KMPFailureFunctionBuild" title="KMP π 数组线性构造步进动画" /> }
export function KMPMatcherPointerJump() { return <PlaceholderComponent name="KMPMatcherPointerJump" title="KMP 匹配时文本/模式指针跳跃可视化" /> }
export function KMPPeriodDetector() { return <PlaceholderComponent name="KMPPeriodDetector" title="KMP 周期子串检测（π 数组应用）" /> }
export function RabinKarpRollingHash() { return <PlaceholderComponent name="RabinKarpRollingHash" title="Rabin-Karp 滚动哈希窗口滑动动画" /> }
export function RabinKarpCollisionDemo() { return <PlaceholderComponent name="RabinKarpCollisionDemo" title="Rabin-Karp 伪匹配与双模哈希对比" /> }
export function BoyerMooreShiftDemo() { return <PlaceholderComponent name="BoyerMooreShiftDemo" title="Boyer-Moore 坏字符与好后缀移动量对比" /> }
export function StringMatchComparison() { return <PlaceholderComponent name="StringMatchComparison" title="四种字符串匹配算法综合性能对比" /> }

// DSA Chapter 32: Trie 树与 AC 自动机
export function TrieInsertSearch() { return <PlaceholderComponent name="TrieInsertSearch" title="Trie 插入与搜索动画（逐字符高亮节点路径）" /> }
export function AhoCorasickFailureLinks() { return <PlaceholderComponent name="AhoCorasickFailureLinks" title="AC 自动机失败链接 BFS 构建可视化" /> }
export function BitwiseTrieXOR() { return <PlaceholderComponent name="BitwiseTrieXOR" title="二进制 Trie XOR 最大值贪心路径动画" /> }
export function PatriciaTreeCompression() { return <PlaceholderComponent name="PatriciaTreeCompression" title="普通 Trie vs 压缩 Trie 空间对比" /> }
export function TrieComplexityComparison() { return <PlaceholderComponent name="TrieComplexityComparison" title="Trie 变体与 AC 自动机复杂度对比表" /> }

// DSA Chapter 33: 后缀数组与后缀树
export function SuffixArrayNaive() { return <PlaceholderComponent name="SuffixArrayNaive" title="后缀数组——朴素排序可视化（所有后缀的字典序展示）" /> }
export function SuffixArrayDoubling() { return <PlaceholderComponent name="SuffixArrayDoubling" title="后缀数组倍增法构造步进（每轮排名更新动画）" /> }
export function LCPArrayKasai() { return <PlaceholderComponent name="LCPArrayKasai" title="Kasai 算法 LCP 数组计算动画（指针移动与 LCP 值更新）" /> }
export function SuffixSearchDemo() { return <PlaceholderComponent name="SuffixSearchDemo" title="SA + 二分搜索子串演示（模式串匹配范围高亮）" /> }
export function SuffixTreeVisualization() { return <PlaceholderComponent name="SuffixTreeVisualization" title="后缀树结构可视化（压缩后缀 Trie，节点/边标注）" /> }
export function SAMStateTransition() { return <PlaceholderComponent name="SAMStateTransition" title="后缀自动机状态转移图与后缀链接可视化" /> }
export function StringStructureComparison() { return <PlaceholderComponent name="StringStructureComparison" title="后缀数组 vs 后缀树 vs SAM 综合对比决策" /> }

// DSA Chapter 34: 高级字符串技术
export function ManacherPalindromeViz() { return <PlaceholderComponent name="ManacherPalindromeViz" title="Manacher 回文半径数组构建步进（center/max_right 指针动态展示）" /> }
export function StringHashingDemo() { return <PlaceholderComponent name="StringHashingDemo" title="字符串前缀哈希预处理与 O(1) 区间哈希查询演示" /> }
export function ZFunctionBuildViz() { return <PlaceholderComponent name="ZFunctionBuildViz" title="Z 函数 O(n) 构造动画（[l, r] 窗口扩展与复用过程）" /> }
export function StringAlgoComparison() { return <PlaceholderComponent name="StringAlgoComparison" title="四种字符串算法对比表（复杂度/场景/实现难度）" /> }

// DSA Chapter 35: 摊销分析
export function DynamicTableGrowthAmortized() { return <PlaceholderComponent name="DynamicTableGrowthAmortized" title="动态表 n 次追加：实际代价 vs 摊销代价曲线" /> }
export function PotentialMethodVisualizer() { return <PlaceholderComponent name="PotentialMethodVisualizer" title="势能函数 Φ 演变动画（动态表扩容时 Φ 骤降曲线）" /> }
export function ThreeAmortizedMethodsCompare() { return <PlaceholderComponent name="ThreeAmortizedMethodsCompare" title="聚合/记账/势能三种摊销证明并排对比" /> }
export function SplayAmortizedTrace() { return <PlaceholderComponent name="SplayAmortizedTrace" title="Splay 旋转时势能函数逐步变化追踪" /> }

// DSA Chapter 36: Fibonacci 堆与高级优先队列
export function FibHeapConsolidate() { return <PlaceholderComponent name="FibHeapConsolidate" title="Fibonacci 堆 CONSOLIDATE 过程动画（同度树合并步骤）" /> }
export function FibHeapDecreaseKey() { return <PlaceholderComponent name="FibHeapDecreaseKey" title="DECREASE-KEY 级联裁剪（CUT 与 CASCADING-CUT 路径可视化）" /> }
export function BinomialHeapMerge() { return <PlaceholderComponent name="BinomialHeapMerge" title="二项堆合并（二进制加法过程动画）" /> }
export function FibVsBinaryHeapPerf() { return <PlaceholderComponent name="FibVsBinaryHeapPerf" title="Fibonacci 堆 vs 二叉堆操作复杂度对比表与图" /> }

// DSA Chapter 37: 线段树与树状数组
export function FenwickTreeUpdate() { return <PlaceholderComponent name="FenwickTreeUpdate" title="树状数组点更新路径可视化（lowbit 跳转动画）" /> }
export function FenwickTreeQuery() { return <PlaceholderComponent name="FenwickTreeQuery" title="树状数组前缀查询路径可视化（逆 lowbit 跳转）" /> }
export function SegmentTreeLazyProp() { return <PlaceholderComponent name="SegmentTreeLazyProp" title="线段树懒惰传播步进动画（lazy 标记向下传播时机）" /> }
export function SparseTableRMQ() { return <PlaceholderComponent name="SparseTableRMQ" title="Sparse Table 预处理与 O(1) 区间最小值查询演示" /> }
export function PersistentSegTreeViz() { return <PlaceholderComponent name="PersistentSegTreeViz" title="持久化线段树版本共享节点可视化" /> }

// DSA Chapter 38: 其他高级数据结构
export function SkipListStructureViz() { return <PlaceholderComponent name="SkipListStructureViz" title="跳表多层链表结构全景图（各层节点与前向指针）" /> }
export function SkipListSearchViz() { return <PlaceholderComponent name="SkipListSearchViz" title="跳表查找步进动画（从顶层逐层向右向下跳跃）" /> }
export function SkipListInsertRandom() { return <PlaceholderComponent name="SkipListInsertRandom" title="跳表插入：随机层高生成过程 + 多层前向指针更新" /> }
export function SkipListVsRBTree() { return <PlaceholderComponent name="SkipListVsRBTree" title="跳表 vs 红黑树：期望/最坏复杂度、实现难度、Redis 选型对比" /> }
export function vEBTreeStructure() { return <PlaceholderComponent name="vEBTreeStructure" title="vEB 树递归结构：cluster、summary 与 min/max 直接存储演示" /> }
export function vEBRecursionTree() { return <PlaceholderComponent name="vEBRecursionTree" title="vEB 操作递推树：展示子问题从 u 缩小到 √u 的路径" /> }
export function SqrtDecompositionBlock() { return <PlaceholderComponent name="SqrtDecompositionBlock" title="分块区间修改 & 查询步进：完整块打标记 vs 边缘块展开对比" /> }
export function MoAlgorithmTrace() { return <PlaceholderComponent name="MoAlgorithmTrace" title="莫队算法：查询排序与双指针移动路径可视化（按块号+右端点排序）" /> }
export function KDTreeBuildQuery() { return <PlaceholderComponent name="KDTreeBuildQuery" title="KD-Tree：建树过程（维度轮流切分）+ 最近邻搜索剪枝动画" /> }
export function KDTreeCurseOfDimensionality() { return <PlaceholderComponent name="KDTreeCurseOfDimensionality" title="高维诅咒演示：维度增加时 KD-Tree 剪枝效果衰减图" /> }

// DSA Chapter 39: 计算几何核心算法
export function CrossProductViz() { return <PlaceholderComponent name="CrossProductViz" title="叉积方向可视化：正/负/零对应左转/右转/共线，动态调整三点位置" /> }
export function SegmentIntersectionTest() { return <PlaceholderComponent name="SegmentIntersectionTest" title="线段交叉判断：叉积方向测试 + 退化共线情形 ON-SEGMENT 可视化" /> }
export function GrahamScanAnimation() { return <PlaceholderComponent name="GrahamScanAnimation" title="Graham 扫描法：极角排序 + 栈弹出步进动画，实时展示凸包构建过程" /> }
export function ConvexHullCompare() { return <PlaceholderComponent name="ConvexHullCompare" title="Graham vs Jarvis March：两种凸包算法步进对比，展示时间复杂度差异" /> }
export function AndrewMonotoneChain() { return <PlaceholderComponent name="AndrewMonotoneChain" title="Andrew's 单调链：上下凸包分别构建动画，展示比 Graham 更简洁的实现" /> }
export function ClosestPairDivide() { return <PlaceholderComponent name="ClosestPairDivide" title="最近点对分治：左右递归 + 带状区域合并可视化，展示中线两侧的距离比较" /> }
export function SevenPointLemma() { return <PlaceholderComponent name="SevenPointLemma" title="7 点引理：带状区域 d×2d 网格分区证明，展示每点最多需比较 7 个邻居" /> }
export function SweepLineDemo() { return <PlaceholderComponent name="SweepLineDemo" title="扫描线算法：事件驱动的竖线从左到右扫过，展示线段插入/删除与邻居检查" /> }

// DSA Chapter 40: 计算几何进阶与综合
export function ShoelacePolygonArea() { return <PlaceholderComponent name="ShoelacePolygonArea" title="Shoelace 公式：逐边累加有向三角形面积，可视化每步贡献（正/负），最终汇总为多边形总面积" /> }
export function PolygonOrientationDemo() { return <PlaceholderComponent name="PolygonOrientationDemo" title="多边形顶点方向：顺时针（面积为负）vs 逆时针（面积为正），有向面积符号与面积值对比" /> }
export function PointInPolygonRayCast() { return <PlaceholderComponent name="PointInPolygonRayCast" title="射线法点在多边形内：拖动查询点，查看水平射线与各边交点计数（奇数=内部，偶数=外部）" /> }
export function WindingNumberDemo() { return <PlaceholderComponent name="WindingNumberDemo" title="绕数法：从查询点观察多边形边界，计算逆时针穿越次数（适用于自交多边形如五角星）" /> }
export function ConvexPolygonBinarySearch() { return <PlaceholderComponent name="ConvexPolygonBinarySearch" title="凸多边形二分点包含：以 P0 为扇形顶点，二分定位所在扇面，O(log n) 查询" /> }
export function ShamosHoeyDemo() { return <PlaceholderComponent name="ShamosHoeyDemo" title="Shamos-Hoey 扫描线：竖线从左到右扫过，BST 活跃集插入/删除时检查上下邻居是否相交" /> }
export function RotatingCalipersDiameter() { return <PlaceholderComponent name="RotatingCalipersDiameter" title="旋转卡壳最远点对：两把平行支撑线旋转夹住凸包，记录旋转过程中对径点的最大距离" /> }
export function MinAreaBoundingRect() { return <PlaceholderComponent name="MinAreaBoundingRect" title="最小面积外接矩形：凸包每条边对齐旋转，旋转卡壳枚举三个支撑点，计算并记录最小外接矩形面积" /> }
export function SATCollisionDetection() { return <PlaceholderComponent name="SATCollisionDetection" title="SAT 分离轴定理：枚举两凸多边形所有边法向量作为候选轴，若任意轴投影不重叠则不碰撞" /> }

// DSA Chapter 41: 计算复杂性理论
export function TuringMachineSimulator() { return <PlaceholderComponent name="TuringMachineSimulator" title="图灵机模拟器：纸带 + 读写头 + 状态转移，步进演示确定型图灵机识别语言的过程" /> }
export function PvsNPVennDiagram() { return <PlaceholderComponent name="PvsNPVennDiagram" title="P vs NP 集合关系图：P⊆NP⊆PSPACE 韦恩图，含 NPC / NP-hard / co-NP，支持 P=NP 与 P≠NP 两种情形切换" /> }
export function PNPDecisionTree() { return <PlaceholderComponent name="PNPDecisionTree" title="P=NP 决策树：若 P=NP 成立/不成立时，各类问题地位如何变化（交互式假设切换）" /> }
export function SATto3SATReduction() { return <PlaceholderComponent name="SATto3SATReduction" title="SAT → 3-SAT 规约动画：逐子句演示1/2/≥4文字子句的引入辅助变量拆分过程，步进高亮每个新子句" /> }
export function VertexCoverNPCProof() { return <PlaceholderComponent name="VertexCoverNPCProof" title="独立集 ↔ 顶点覆盖互补定理：可视化图中 IS 与 VC 的互补关系，并展示 3-SAT → IS 的变量/子句→图构造" /> }
export function SubsetSumReductionDemo() { return <PlaceholderComponent name="SubsetSumReductionDemo" title="Subset Sum 规约演示：3-SAT 实例→整数集合的构造过程，按位展示变量行与子句列" /> }
export function NPCompleteReductionMap() { return <PlaceholderComponent name="NPCompleteReductionMap" title="NPC 规约关系有向图：节点为经典 NPC 问题，箭头为规约方向，点击显示规约思路" /> }

// DSA Chapter 42: 近似算法与处理 NP 难问题
export function ApproximationHierarchy() { return <PlaceholderComponent name="ApproximationHierarchy" title="近似能力层次图：FPTAS ⊂ PTAS ⊂ 固定常数近似 ⊂ APX-hard，直观展示各层定义与代表问题" /> }
export function VertexCoverApprox() { return <PlaceholderComponent name="VertexCoverApprox" title="顶点覆盖 2-近似步进：逐边选取 u,v 并删除关联边，最终覆盖集 C 与 OPT 的对比可视化" /> }
export function ChristofidesViz() { return <PlaceholderComponent name="ChristofidesViz" title="Christofides 1.5-近似 TSP：MST → 奇度顶点完美匹配 → 欧拉回路 → 哈密顿路径提取的步骤动画" /> }
export function SetCoverGreedy() { return <PlaceholderComponent name="SetCoverGreedy" title="集合覆盖贪心 ln(n) 近似：每步选覆盖未覆盖元素最多的集合，步进展示覆盖进度与 ln(n) 近似比推导" /> }
export function FPTASSubsetSum() { return <PlaceholderComponent name="FPTASSubsetSum" title="子集和 FPTAS 交互演示：调节 ε 参数，观察缩放后 DP 状态数与近似精度的权衡变化" /> }
export function MaxCutRandomized() { return <PlaceholderComponent name="MaxCutRandomized" title="随机 MAX-CUT 2-近似：随机二着色演示，展示期望切割数 ≥ |E|/2 的概率证明与多次试验统计" /> }
export function SimulatedAnnealingTSP() { return <PlaceholderComponent name="SimulatedAnnealingTSP" title="模拟退火 TSP：温度曲线 + 路径总长随迭代步骤的实时演化，可调节 T₀、冷却率、每温度步数" /> }
export function GeneticAlgorithmViz() { return <PlaceholderComponent name="GeneticAlgorithmViz" title="遗传算法 TSP：种群适应度分布 + 最优个体路径随代数演化，展示选择、交叉、变异三算子效果" /> }
export function ApproxVsHeuristic() { return <PlaceholderComponent name="ApproxVsHeuristic" title="近似算法 vs 启发式算法：理论保证、实际质量、调参难度、适用规模四维对比表与决策流程图" /> }
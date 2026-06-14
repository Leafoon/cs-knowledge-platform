"use client";

import React, { useState } from "react";

// ── 数据定义 ─────────────────────────────────────────────────────────────────
interface Problem {
  name: string;
  approach: string; // greedy | divide | dp | backtrack
  why: string;
}

interface Paradigm {
  id: string;
  name: string;
  nameEn: string;
  emoji: string;
  tagline: string;
  analogy: string;
  analogyIcon: string;
  when: string[];
  notWhen: string[];
  steps: string[];
  complexity: string;
  proof: string;
  color: string;        // Tailwind gradient classes
  borderColor: string;
  textColor: string;
  bgLight: string;
  problems: { name: string; note: string }[];
}

const PARADIGMS: Paradigm[] = [
  {
    id: "greedy",
    name: "贪心",
    nameEn: "Greedy",
    emoji: "🏃",
    tagline: "局部最优 → 全局最优（不回头）",
    analogy: "找零钱：总是先用最大面额的硬币",
    analogyIcon: "💰",
    when: [
      "具有贪心选择性质（局部最优不妨碍全局）",
      "通常需要先排序后处理",
      "问题没有重叠子问题，不需要记忆化",
    ],
    notWhen: [
      "局部最优不蕴含全局最优（如 0/1 背包）",
      "需要考虑历史决策的后果（有后效性）",
    ],
    steps: ["制定贪心策略（往往是排序标准）", "扫描一遍做选择", "用交换论证证明正确性"],
    complexity: "通常 O(n log n)（排序主导）",
    proof: "交换论证（Exchange Argument）",
    color: "from-amber-500/20 to-orange-500/10",
    borderColor: "border-amber-400/40",
    textColor: "text-amber-300",
    bgLight: "bg-amber-500/10",
    problems: [
      { name: "活动选择", note: "选结束最早的" },
      { name: "哈夫曼编码", note: "每次合并最小的两棵树" },
      { name: "Dijkstra 最短路", note: "每次选距离最小的节点" },
      { name: "最小生成树 Prim/Kruskal", note: "选最小权边" },
      { name: "分数背包", note: "按单位价值排序" },
    ],
  },
  {
    id: "divide",
    name: "分治",
    nameEn: "Divide & Conquer",
    emoji: "✂️",
    tagline: "分解 → 解决 → 合并（子问题独立）",
    analogy: "查字典：先翻中间，再在一半里查找",
    analogyIcon: "📖",
    when: [
      "问题可以被分成若干规模更小的同类子问题",
      "子问题之间完全独立（无重叠！）",
      "有高效的合并步骤（O(n) 或 O(n log n)）",
    ],
    notWhen: [
      "子问题大量重叠（此时应用 DP + 记忆化）",
      "合并代价过高（O(n²)）导致整体仍然很慢",
    ],
    steps: ["Divide：将问题一分为二（或多份）", "Conquer：递归解决每个子问题", "Combine：合并子问题的解"],
    complexity: "T(n) = aT(n/b) + O(f(n))，主定理求解",
    proof: "递归归纳（对问题规模做归纳）",
    color: "from-blue-500/20 to-cyan-500/10",
    borderColor: "border-blue-400/40",
    textColor: "text-blue-500 dark:text-blue-300",
    bgLight: "bg-blue-500/10",
    problems: [
      { name: "归并排序", note: "O(n log n)" },
      { name: "快速排序", note: "期望 O(n log n)" },
      { name: "二分查找", note: "O(log n)" },
      { name: "卡拉苏巴乘法", note: "O(n^1.585)" },
      { name: "最近点对问题", note: "O(n log n)" },
    ],
  },
  {
    id: "dp",
    name: "动态规划",
    nameEn: "Dynamic Programming",
    emoji: "🧱",
    tagline: "重叠子问题 + 记忆化（避免重复计算）",
    analogy: "爬楼梯：记下到每一阶的方法数，下次直接查",
    analogyIcon: "🪜",
    when: [
      "问题满足最优子结构（最优解由子问题最优解构成）",
      "子问题大量重叠（分治会重复计算）",
      "子问题数量有限（通常多项式级别）",
    ],
    notWhen: [
      "状态空间过大（维度爆炸）",
      "无法明确定义状态转移方程",
      "子问题无重叠（直接用分治即可）",
    ],
    steps: ["定义状态 dp[i][j]（什么意思？）", "写出状态转移方程", "确定初始值和计算顺序", "从答案状态读取结果"],
    complexity: "状态数 × 转移代价（通常多项式）",
    proof: "归纳证明最优子结构",
    color: "from-violet-500/20 to-purple-500/10",
    borderColor: "border-violet-400/40",
    textColor: "text-violet-500 dark:text-violet-300",
    bgLight: "bg-violet-500/10",
    problems: [
      { name: "斐波那契数列", note: "O(n)" },
      { name: "最长公共子序列 (LCS)", note: "O(n²)" },
      { name: "0/1 背包", note: "O(nW)" },
      { name: "矩阵链乘积", note: "O(n³)" },
      { name: "Bellman-Ford 最短路", note: "O(VE)" },
    ],
  },
  {
    id: "backtrack",
    name: "回溯",
    nameEn: "Backtracking",
    emoji: "🌳",
    tagline: "DFS 枚举 + 剪枝（系统性搜索）",
    analogy: "走迷宫：碰到死路就退回上一个岔口换路",
    analogyIcon: "🗺️",
    when: [
      "需要枚举所有可能的解",
      "可以提前判断某条路径不可能成功（剪枝）",
      "解空间是树形或图形结构",
    ],
    notWhen: [
      "解空间太大（n!）且剪枝效果差",
      "问题有最优子结构（此时 DP 更高效）",
    ],
    steps: ["定义选择空间和状态", "在每步做一个选择", "判断是否可以剪枝（终止当前路径）", "到达终态则记录，否则回溯（撤销选择）"],
    complexity: "最坏 O(n!) 或 O(2^n)，剪枝后大幅降低",
    proof: "搜索树完整覆盖所有状态",
    color: "from-emerald-500/20 to-teal-500/10",
    borderColor: "border-emerald-400/40",
    textColor: "text-emerald-300",
    bgLight: "bg-emerald-500/10",
    problems: [
      { name: "N 皇后", note: "剪枝后远快于 O(n!)" },
      { name: "全排列", note: "O(n! × n)" },
      { name: "数独", note: "约束传播 + 剪枝" },
      { name: "子集枚举", note: "O(2^n)" },
      { name: "图着色问题", note: "NP-hard，剪枝关键" },
    ],
  },
];

// ── 问题判断提示 ──────────────────────────────────────────────────────────────
const QUIZ_PROBLEMS: Problem[] = [
  { name: "活动选择（最多参加几场不冲突的活动）", approach: "greedy", why: "贪心选结束最早的活动，具有贪心选择性质" },
  { name: "汉诺塔（将 n 个圆盘从 A 柱移到 C 柱）", approach: "divide", why: "三步分治：先把 n-1 个移到 B，再移最大盘，再把 n-1 个从 B 移到 C" },
  { name: "最长递增子序列（LIS）", approach: "dp", why: "设 dp[i]=以第i个元素结尾的LIS长度，有明显的最优子结构" },
  { name: "N 皇后（n×n 棋盘放 n 个互不攻击的皇后）", approach: "backtrack", why: "需要枚举所有可能位置，可用列/对角线约束剪枝" },
  { name: "分数背包（物品可以切割）", approach: "greedy", why: "按单位价值从高到低贪心选取，可证明局部最优=全局最优" },
  { name: "归并排序（稳定排序 O(n log n)）", approach: "divide", why: "分为两半分别排序，再合并 — 经典分治三步骤" },
  { name: "0/1 背包（物品不可切割）", approach: "dp", why: "分数背包的贪心不适用于此，必须DP：dp[i][w]=前i件重量不超w的最大价值" },
  { name: "数独求解", approach: "backtrack", why: "对每个空格枚举1-9，用行/列/格约束剪枝" },
];

// ── 主组件 ────────────────────────────────────────────────────────────────────
export default function AlgorithmParadigmExplorer() {
  const [activeTab, setActiveTab] = useState<string>("greedy");
  const [quizIdx, setQuizIdx] = useState(0);
  const [quizAnswer, setQuizAnswer] = useState<string | null>(null);
  const [showQuizResult, setShowQuizResult] = useState(false);
  const [showView, setShowView] = useState<"detail" | "compare" | "quiz">("detail");

  const active = PARADIGMS.find((p) => p.id === activeTab)!;
  const quiz = QUIZ_PROBLEMS[quizIdx];

  const handleQuizAnswer = (id: string) => {
    setQuizAnswer(id);
    setShowQuizResult(true);
  };

  const nextQuiz = () => {
    setQuizIdx((i) => (i + 1) % QUIZ_PROBLEMS.length);
    setQuizAnswer(null);
    setShowQuizResult(false);
  };

  return (
    <div className="rounded-2xl border border-border-subtle bg-bg-secondary p-6 my-6 shadow-sm">
      {/* 标题 */}
      <div className="flex items-center gap-3 mb-5">
        <div className="w-9 h-9 rounded-xl bg-purple-500/20 flex items-center justify-center text-xl">
          🧭
        </div>
        <div>
          <h3 className="font-bold text-text-primary text-base">四大算法范式探索器</h3>
          <p className="text-xs text-text-secondary">贪心 · 分治 · 动态规划 · 回溯——深入理解适用场景</p>
        </div>
        {/* 视图切换 */}
        <div className="ml-auto flex rounded-lg overflow-hidden border border-border-subtle text-xs">
          {(["detail", "compare", "quiz"] as const).map((v) => (
            <button
              key={v}
              onClick={() => setShowView(v)}
              className={`px-3 py-1.5 font-medium transition-colors ${showView === v ? "bg-purple-500/30 text-purple-700 dark:text-purple-200" : "bg-bg-tertiary text-text-secondary hover:text-text-secondary"}`}
            >
              {v === "detail" ? "详情" : v === "compare" ? "对比" : "测验"}
            </button>
          ))}
        </div>
      </div>

      {/* ──── 详情视图 ──── */}
      {showView === "detail" && (
        <div>
          {/* 范式选择 Tab */}
          <div className="flex gap-2 mb-5 flex-wrap">
            {PARADIGMS.map((p) => (
              <button
                key={p.id}
                onClick={() => setActiveTab(p.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl border text-sm font-medium transition-all ${
                  activeTab === p.id
                    ? `bg-gradient-to-r ${p.color} ${p.borderColor} ${p.textColor}`
                    : "border-border-subtle bg-bg-tertiary text-text-secondary hover:text-text-secondary"
                }`}
              >
                <span>{p.emoji}</span>
                <span>{p.name}</span>
                <span className="text-xs opacity-60 hidden sm:inline">{p.nameEn}</span>
              </button>
            ))}
          </div>

          {/* 活动范式详情 */}
          <div className={`rounded-xl border bg-gradient-to-br ${active.color} ${active.borderColor} p-5`}>
            {/* 标语 + 类比 */}
            <div className="flex flex-wrap gap-3 mb-4">
              <div className="flex-1 min-w-48">
                <div className={`text-lg font-bold ${active.textColor} mb-1`}>
                  {active.emoji} {active.name} <span className="text-sm font-normal opacity-70">({active.nameEn})</span>
                </div>
                <div className="text-sm text-text-secondary">{active.tagline}</div>
              </div>
              <div className="rounded-lg bg-bg-tertiary px-4 py-2 text-sm">
                <div className="text-xs text-text-tertiary mb-0.5">生活类比</div>
                <div className="text-text-primary">{active.analogyIcon} {active.analogy}</div>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* 左：适用/不适用 */}
              <div className="space-y-3">
                <div>
                  <div className="text-xs font-semibold text-emerald-400 mb-1.5 flex items-center gap-1">
                    <span>✅</span> 何时使用
                  </div>
                  <ul className="space-y-1">
                    {active.when.map((w, i) => (
                      <li key={i} className="flex items-start gap-1.5 text-xs text-text-secondary">
                        <span className="text-emerald-400 mt-0.5 shrink-0">●</span>
                        {w}
                      </li>
                    ))}
                  </ul>
                </div>
                <div>
                  <div className="text-xs font-semibold text-rose-400 mb-1.5 flex items-center gap-1">
                    <span>❌</span> 不适用时
                  </div>
                  <ul className="space-y-1">
                    {active.notWhen.map((w, i) => (
                      <li key={i} className="flex items-start gap-1.5 text-xs text-text-secondary">
                        <span className="text-rose-400 mt-0.5 shrink-0">●</span>
                        {w}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* 右：步骤 + 典型问题 */}
              <div className="space-y-3">
                <div>
                  <div className="text-xs font-semibold text-text-secondary mb-1.5">🪜 通用步骤</div>
                  <ol className="space-y-1">
                    {active.steps.map((s, i) => (
                      <li key={i} className="flex items-start gap-2 text-xs text-text-secondary">
                        <span className={`${active.textColor} font-bold shrink-0`}>{i + 1}.</span>
                        {s}
                      </li>
                    ))}
                  </ol>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className={`rounded-lg ${active.bgLight} p-2`}>
                    <div className="text-text-secondary mb-0.5">典型复杂度</div>
                    <div className={`font-mono ${active.textColor}`}>{active.complexity}</div>
                  </div>
                  <div className={`rounded-lg ${active.bgLight} p-2`}>
                    <div className="text-text-secondary mb-0.5">正确性证明</div>
                    <div className={`font-mono ${active.textColor} text-[10px]`}>{active.proof}</div>
                  </div>
                </div>
                <div>
                  <div className="text-xs font-semibold text-text-secondary mb-1.5">📌 典型问题</div>
                  <div className="flex flex-wrap gap-1.5">
                    {active.problems.map((p) => (
                      <span key={p.name} title={p.note}
                        className={`text-[11px] px-2 py-0.5 rounded-full ${active.bgLight} ${active.textColor} cursor-help`}>
                        {p.name}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ──── 对比视图 ──── */}
      {showView === "compare" && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr>
                <th className="text-left text-text-secondary font-normal p-2 border-b border-border-subtle w-24">维度</th>
                {PARADIGMS.map((p) => (
                  <th key={p.id} className="p-2 border-b border-border-subtle text-center">
                    <span className={`${p.textColor} font-semibold`}>{p.emoji} {p.name}</span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="text-text-secondary">
              {[
                { dim: "核心策略", keys: PARADIGMS.map((p) => p.tagline) },
                { dim: "子问题", keys: ["无重叠，直接做选择", "独立、无重叠", "重叠、需记忆化", "枚举所有路径"] },
                { dim: "回头？", keys: ["❌ 不回头", "❌ 不回头", "❌ 顺序推导", "✅ 会回溯"] },
                { dim: "典型复杂度", keys: PARADIGMS.map((p) => p.complexity.split("（")[0]) },
                { dim: "适用规模", keys: ["n ≤ 10⁶", "n ≤ 10⁷", "n ≤ 10³～10⁴", "n ≤ 20（精确）"] },
                { dim: "正确性证明", keys: PARADIGMS.map((p) => p.proof) },
              ].map((row, ri) => (
                <tr key={ri} className={ri % 2 === 0 ? "bg-bg-tertiary" : ""}>
                  <td className="p-2 text-text-secondary font-medium">{row.dim}</td>
                  {row.keys.map((v, ci) => (
                    <td key={ci} className={`p-2 text-center ${PARADIGMS[ci].textColor} text-[11px]`}>{v}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ──── 测验视图 ──── */}
      {showView === "quiz" && (
        <div>
          <div className="rounded-xl border border-border-subtle bg-bg-tertiary p-5">
            <div className="text-xs text-text-secondary mb-1">
              题目 {quizIdx + 1} / {QUIZ_PROBLEMS.length}
            </div>
            <div className="text-base font-semibold text-text-primary mb-4">
              以下问题最适合用哪种范式？
            </div>
            <div className="text-sm text-text-primary mb-5 p-3 rounded-lg bg-bg-tertiary border border-border-subtle">
              📝 {quiz.name}
            </div>

            <div className="grid grid-cols-2 gap-3">
              {PARADIGMS.map((p) => {
                const isCorrect = p.id === quiz.approach;
                const isSelected = quizAnswer === p.id;
                let cls = `rounded-xl border p-3 text-sm font-medium transition-all cursor-pointer `;
                if (!showQuizResult) {
                  cls += `bg-gradient-to-r ${p.color} ${p.borderColor} ${p.textColor} hover:opacity-90`;
                } else if (isCorrect) {
                  cls += "bg-emerald-500/20 border-emerald-400/60 text-emerald-700 dark:text-emerald-200";
                } else if (isSelected) {
                  cls += "bg-rose-500/20 border-rose-400/60 text-rose-300";
                } else {
                  cls += "bg-bg-tertiary border-border-subtle text-text-tertiary";
                }
                return (
                  <button
                    key={p.id}
                    onClick={() => !showQuizResult && handleQuizAnswer(p.id)}
                    disabled={showQuizResult}
                    className={cls}
                  >
                    <span className="mr-2">{p.emoji}</span>
                    {p.name}
                    {showQuizResult && isCorrect && <span className="ml-2">✓</span>}
                    {showQuizResult && isSelected && !isCorrect && <span className="ml-2">✗</span>}
                  </button>
                );
              })}
            </div>

            {showQuizResult && (
              <div className={`mt-4 p-3 rounded-lg border text-sm ${quizAnswer === quiz.approach ? "bg-emerald-500/10 border-emerald-400/30 text-emerald-700 dark:text-emerald-200" : "bg-rose-500/10 border-rose-400/30 text-rose-700 dark:text-rose-200"}`}>
                <div className="font-semibold mb-1">
                  {quizAnswer === quiz.approach ? "✅ 正确！" : `❌ 错误，正确答案是【${PARADIGMS.find((p) => p.id === quiz.approach)?.name}】`}
                </div>
                <div className="text-xs opacity-80">{quiz.why}</div>
              </div>
            )}

            {showQuizResult && (
              <button
                onClick={nextQuiz}
                className="mt-3 w-full py-2 rounded-lg bg-purple-500/20 hover:bg-purple-500/30 text-purple-700 dark:text-purple-200 text-sm font-medium transition-colors"
              >
                下一题 →
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

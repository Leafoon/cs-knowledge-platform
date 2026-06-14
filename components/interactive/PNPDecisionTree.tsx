"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CheckCircle2, XCircle, HelpCircle, ExternalLink } from "lucide-react";

type Complexity = "P" | "NP-only" | "NPC" | "NP-hard" | "UNDEC";

interface Problem {
  name: string;
  cn: string;
  complexity: Complexity;
  verifiable: boolean | "N/A";
  polyAlgo: boolean | "unknown" | "N/A";
  isNPhard: boolean;
  cert: string;
  algo: string;
  note: string;
  reduction?: string;
}

const PROBLEMS: Problem[] = [
  {
    name: "Sorting", cn: "排序",
    complexity: "P", verifiable: true, polyAlgo: true, isNPhard: false,
    cert: "给出排好序的序列，O(n) 验证 ✓",
    algo: "归并排序 O(n log n)，确定多项式时间可解",
    note: "最经典的 P 类问题",
  },
  {
    name: "Shortest Path", cn: "单源最短路",
    complexity: "P", verifiable: true, polyAlgo: true, isNPhard: false,
    cert: "给出路径，O(E) 验证权重之和 ✓",
    algo: "Dijkstra O(E + V log V)，Bellman-Ford O(VE)",
    note: "图算法中 P 类的代表",
  },
  {
    name: "2-SAT", cn: "2-SAT 可满足性",
    complexity: "P", verifiable: true, polyAlgo: true, isNPhard: false,
    cert: "给出赋值，O(n) 验证每个二文字子句 ✓",
    algo: "SCC（Kosaraju / Tarjan）O(V+E)",
    note: "与 3-SAT 的一字之差，复杂度天壤之别！",
  },
  {
    name: "Primality", cn: "素数判定",
    complexity: "P", verifiable: true, polyAlgo: true, isNPhard: false,
    cert: "NP ∩ co-NP（证明素或合数均有高效证书）",
    algo: "AKS 算法（2002）O(log⁶ n)，确定多项式",
    note: "2002 年前被认为可能不在 P 中，是复杂性理论的里程碑",
  },
  {
    name: "Graph Isomorphism (GI)", cn: "图同构",
    complexity: "NP-only", verifiable: true, polyAlgo: "unknown", isNPhard: false,
    cert: "给出节点映射，O(V+E) 验证 ✓",
    algo: "无已知多项式算法；Babai 2016 准多项式算法 n^{polylog n}",
    note: "疑似介于 P 和 NPC 之间，是复杂性理论最神秘问题之一",
  },
  {
    name: "3-SAT", cn: "3-SAT 可满足性",
    complexity: "NPC", verifiable: true, polyAlgo: false, isNPhard: true,
    cert: "给出变量赋值，O(子句数) 验证每个3文字子句 ✓",
    algo: "无多项式算法（除非 P=NP）；暴力 O(2ⁿ·m)",
    note: "Cook-Levin 定理：第一个被证明的 NPC 问题",
    reduction: "SAT ≤p 3-SAT（Cook-Levin）",
  },
  {
    name: "Independent Set", cn: "k-独立集",
    complexity: "NPC", verifiable: true, polyAlgo: false, isNPhard: true,
    cert: "给出 k 个节点集合，验证两两不相邻 O(k²) ✓",
    algo: "无多项式算法；FPT 在 k 固定时可行",
    note: "经典图论 NPC 问题，与顶点覆盖互补",
    reduction: "3-SAT ≤p IS（变量与子句小工件构造）",
  },
  {
    name: "Vertex Cover", cn: "k-顶点覆盖",
    complexity: "NPC", verifiable: true, polyAlgo: false, isNPhard: true,
    cert: "给出 k 个节点集合，验证每条边至少一个节点在集内 O(E) ✓",
    algo: "无多项式算法（一般情况）；2-近似算法存在",
    note: "IS ↔ VC 互补：V\\IS 是 VC，|IS|+|VC|=|V|",
    reduction: "IS ≤p VC（互补规约，即 IS → 取补集 → VC）",
  },
  {
    name: "Hamiltonian Cycle", cn: "哈密顿回路",
    complexity: "NPC", verifiable: true, polyAlgo: false, isNPhard: true,
    cert: "给出节点排列，验证相邻节点间均有边且恰好覆盖所有节点 O(V) ✓",
    algo: "无多项式算法；Held-Karp O(2ⁿ·n²) 精确算法",
    note: "与欧拉回路（P 类）易混淆，但 HAM-CYCLE 是 NPC",
    reduction: "3-SAT ≤p HAM-CYCLE（信号 gadget 构造）",
  },
  {
    name: "TSP (optimization)", cn: "旅行商（最优化版）",
    complexity: "NP-hard", verifiable: false, polyAlgo: false, isNPhard: true,
    cert: "最优化版本不是判定问题，无法直接验证「最优」性",
    algo: "TSP 判定版是 NPC；最优化版是 NP-hard（比 NPC 更难）",
    note: "「找最短哈密顿回路」的最优化版，超出 NP 范畴",
    reduction: "HAM-CYCLE ≤p TSP（判定版），优化版需额外论证",
  },
  {
    name: "Halting Problem", cn: "停机问题",
    complexity: "UNDEC", verifiable: "N/A", polyAlgo: "N/A", isNPhard: true,
    cert: "不可计算：不存在任何图灵机能正确判定所有程序是否停机",
    algo: "图灵 1936 年用对角线法证明不可判定，且无法枚举",
    note: "NP-hard 且不在 NP 中；计算理论的基石结论",
  },
];

const COMPLEXITY_CONFIG: Record<Complexity, { label: string; color: string; bg: string; ring: string }> = {
  "P":        { label: "P 类",       color: "text-indigo-700 dark:text-indigo-300", bg: "bg-indigo-50 dark:bg-indigo-900/40", ring: "ring-indigo-400" },
  "NP-only":  { label: "NP（疑似）", color: "text-blue-700 dark:text-blue-300",   bg: "bg-blue-50 dark:bg-blue-900/40",   ring: "ring-blue-400" },
  "NPC":      { label: "NP 完全",    color: "text-rose-700 dark:text-rose-300",    bg: "bg-rose-50 dark:bg-rose-900/40",    ring: "ring-rose-400" },
  "NP-hard":  { label: "NP-hard",   color: "text-orange-700 dark:text-orange-300", bg: "bg-orange-50 dark:bg-orange-900/40", ring: "ring-orange-400" },
  "UNDEC":    { label: "不可判定",   color: "text-slate-600 dark:text-slate-400",  bg: "bg-slate-100 dark:bg-slate-800",   ring: "ring-slate-400" },
};

function BoolIcon({ val }: { val: boolean | "unknown" | "N/A" }) {
  if (val === true) return <CheckCircle2 className="w-4 h-4 text-emerald-500 flex-shrink-0" />;
  if (val === false) return <XCircle className="w-4 h-4 text-red-500 flex-shrink-0" />;
  return <HelpCircle className="w-4 h-4 text-amber-500 flex-shrink-0" />;
}

export function PNPDecisionTree() {
  const [selected, setSelected] = useState<Problem>(PROBLEMS[5]);
  const [filter, setFilter] = useState<Complexity | "all">("all");

  const filtered = filter === "all" ? PROBLEMS : PROBLEMS.filter(p => p.complexity === filter);
  const cfg = COMPLEXITY_CONFIG[selected.complexity];

  return (
    <div className="w-full max-w-5xl mx-auto my-6 rounded-2xl overflow-hidden border border-emerald-200 dark:border-emerald-800 shadow-lg">
      {/* Header */}
      <div className="bg-gradient-to-r from-emerald-600 to-teal-600 px-6 py-4 text-white">
        <h3 className="text-lg font-bold">经典问题复杂度分类手册</h3>
        <p className="text-sm text-emerald-100 mt-0.5">选择问题，查看它在 P / NP / NPC / NP-hard 中的归属与依据</p>
      </div>

      <div className="bg-white dark:bg-slate-900">
        {/* Filter tabs */}
        <div className="flex gap-2 px-6 pt-4 flex-wrap">
          {(["all", "P", "NP-only", "NPC", "NP-hard", "UNDEC"] as const).map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1 text-xs rounded-lg border font-medium transition-all ${
                filter === f
                  ? f === "all"
                    ? "bg-slate-700 text-white border-slate-700"
                    : `${COMPLEXITY_CONFIG[f as Complexity]?.bg} ${COMPLEXITY_CONFIG[f as Complexity]?.color} border-current ring-1 ${COMPLEXITY_CONFIG[f as Complexity]?.ring}`
                  : "border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400 hover:border-slate-300"
              }`}
            >
              {f === "all" ? "全部" : COMPLEXITY_CONFIG[f as Complexity]?.label}
            </button>
          ))}
        </div>

        <div className="flex flex-col md:flex-row p-4 gap-4">
          {/* Problem list */}
          <div className="md:w-52 flex-shrink-0 space-y-1.5">
            {filtered.map(p => {
              const c = COMPLEXITY_CONFIG[p.complexity];
              const isSelected = selected.name === p.name;
              return (
                <button
                  key={p.name}
                  onClick={() => setSelected(p)}
                  className={`w-full text-left px-3 py-2.5 rounded-xl border transition-all ${
                    isSelected
                      ? `${c.bg} ${c.color} border-current font-semibold shadow ring-1 ${c.ring}`
                      : "border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300"
                  }`}
                >
                  <div className="text-sm">{p.cn}</div>
                  <div className={`text-xs mt-0.5 ${isSelected ? "opacity-80" : "text-slate-400 dark:text-slate-500"}`}>
                    {COMPLEXITY_CONFIG[p.complexity].label}
                  </div>
                </button>
              );
            })}
          </div>

          {/* Detail card */}
          <div className="flex-1 min-w-0">
            <AnimatePresence mode="wait">
              <motion.div
                key={selected.name}
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                transition={{ duration: 0.18 }}
                className="space-y-4"
              >
                {/* Title */}
                <div className={`p-4 rounded-2xl border ${cfg.bg} ${cfg.color} border-current/30`}>
                  <div className="flex items-center justify-between flex-wrap gap-2">
                    <div>
                      <h4 className="text-xl font-bold">{selected.cn}</h4>
                      <p className="text-sm opacity-70 font-mono">{selected.name}</p>
                    </div>
                    <span className={`px-4 py-1.5 rounded-xl text-sm font-bold border border-current/30 ${cfg.bg} ${cfg.color}`}>
                      {cfg.label}
                    </span>
                  </div>
                  <p className="text-sm mt-2 opacity-80 font-medium">{selected.note}</p>
                </div>

                {/* Three questions */}
                <div className="grid grid-cols-1 gap-3">
                  {[
                    {
                      q: "① 多项式时间可验证证书？（∈ NP 的条件）",
                      val: selected.verifiable,
                      detail: selected.cert,
                    },
                    {
                      q: "② 存在已知多项式时间算法？（∈ P 的条件）",
                      val: selected.polyAlgo,
                      detail: selected.algo,
                    },
                    {
                      q: "③ NP-hard？（所有 NP 问题可规约至此）",
                      val: selected.isNPhard,
                      detail: selected.reduction || (selected.isNPhard ? "是 NP-hard，存在从任意 NP 问题的多项式时间规约" : "无已知 NP-hard 证明"),
                    },
                  ].map((item, i) => (
                    <div key={i} className="flex gap-3 p-3.5 bg-slate-50 dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700">
                      <BoolIcon val={item.val} />
                      <div className="min-w-0">
                        <p className="text-xs font-semibold text-slate-700 dark:text-slate-300">{item.q}</p>
                        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">{item.detail}</p>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Classification reasoning */}
                <div className={`p-4 rounded-xl border border-current/20 ${cfg.bg}`}>
                  <p className="text-xs font-bold uppercase tracking-wider mb-2 opacity-70">复杂度结论</p>
                  <p className={`text-sm font-semibold ${cfg.color}`}>
                    {selected.complexity === "P" && "可验证 ✓ + 有多项式算法 ✓ → 在 P 类中。P ⊆ NP，也是 NP 问题。"}
                    {selected.complexity === "NP-only" && "可验证 ✓，但无已知多项式算法，也无 NP-hard 证明 → 疑似 NP-hard 之间的「灰色地带」。"}
                    {selected.complexity === "NPC" && "可验证 ✓ + NP-hard ✓ = NP 完全（NPC）。是 NP 中最难的一批。"}
                    {selected.complexity === "NP-hard" && "是 NP-hard（困难性同 NPC），但本问题是最优化版，不直接是判定问题，可能不在 NP 中。"}
                    {selected.complexity === "UNDEC" && "不在任何合理复杂度类中，超越图灵可计算范畴——连枚举都做不到。"}
                  </p>
                </div>
              </motion.div>
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
}

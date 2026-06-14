"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Shield, Zap, HelpCircle, CheckCircle, XCircle, Minus } from "lucide-react";

type Rating = "yes" | "no" | "partial";

interface Dimension {
  id: string;
  label: string;
  icon: string;
  approx: { rating: Rating; text: string };
  heuristic: { rating: Rating; text: string };
  note?: string;
}

const DIMENSIONS: Dimension[] = [
  {
    id: "guarantee",
    label: "理论近似保证",
    icon: "🛡️",
    approx: { rating: "yes", text: "有严格的近似比 ρ，可数学证明 OPT ≤ ALG ≤ ρ·OPT" },
    heuristic: { rating: "no", text: "无保证，输出质量依赖实例结构，最坏情况可能任意差" },
    note: "近似算法的核心优势：可证明的质量下界"
  },
  {
    id: "worstcase",
    label: "最坏情况复杂度",
    icon: "⏱️",
    approx: { rating: "yes", text: "多项式时间，如 O(n log n)、O(n²)" },
    heuristic: { rating: "partial", text: "通常快速，但最坏情况收敛性无保证（SA/GA 可能陷入局部最优）" },
    note: "二者都能在实践中快速运行，但保证程度不同"
  },
  {
    id: "tuning",
    label: "参数调整需求",
    icon: "🔧",
    approx: { rating: "partial", text: "少量参数（如 Christofides 无参数），FPTAS 有 ε 控制精度" },
    heuristic: { rating: "no", text: "大量超参数（SA：T₀、α；GA：交叉率、变异率、种群大小）" },
    note: "启发式算法的调参成本在实际工程中不可忽略"
  },
  {
    id: "design",
    label: "设计难度",
    icon: "🧠",
    approx: { rating: "no", text: "通常需要非平凡的算法设计技巧（原始对偶、LP 舍入、随机化）" },
    heuristic: { rating: "yes", text: "直觉性强，易于实现（爬山法、模拟退火均可快速原型）" },
    note: "近似算法的分析往往需要深厚的理论背景"
  },
  {
    id: "practical",
    label: "实际效果",
    icon: "🎯",
    approx: { rating: "partial", text: "实际解质量有时保守（近似比可能远离 OPT 的真实差距）" },
    heuristic: { rating: "yes", text: "实践中常常找到接近最优的解，尤其是精心设计的元启发式" },
    note: "启发式在实践中有时优于近似算法，但缺乏理论支撑"
  },
  {
    id: "applicability",
    label: "适用问题类型",
    icon: "📋",
    approx: { rating: "partial", text: "并非所有 NP-hard 问题都存在已知近似算法（APX-hard 问题很难近似）" },
    heuristic: { rating: "yes", text: "通用框架适用于几乎所有优化问题，无需深入了解问题结构" },
    note: "某些问题（如 Max-Clique）APX-hard，近似算法无法实现常数比"
  },
];

const EXAMPLES = [
  { problem: "顶点覆盖", approx: "2-近似（匹配贪心）", heuristic: "爬山法、局部搜索", winner: "both" },
  { problem: "TSP（度量）", approx: "Christofides 3/2", heuristic: "SA、GA、LKH", winner: "heuristic" },
  { problem: "集合覆盖", approx: "ln n-近似（贪心）", heuristic: "遗传算法", winner: "approx" },
  { problem: "背包/子集和", approx: "FPTAS (1±ε)", heuristic: "DP 启发剪枝", winner: "approx" },
  { problem: "Max-Cut", approx: "0.878（SDP，GW）", heuristic: "随机染色，差分进化", winner: "approx" },
  { problem: "调度问题", approx: "LPT 4/3-近似", heuristic: "模拟退火", winner: "both" },
];

const RATING_ICON: Record<Rating, React.ReactNode> = {
  yes: <CheckCircle size={14} className="text-emerald-500" />,
  no: <XCircle size={14} className="text-rose-500" />,
  partial: <Minus size={14} className="text-amber-500" />,
};
const RATING_BG: Record<Rating, string> = {
  yes: "bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800",
  no: "bg-rose-50 dark:bg-rose-900/20 border-rose-200 dark:border-rose-800",
  partial: "bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800",
};

export function ApproxVsHeuristic() {
  const [activeRow, setActiveRow] = useState<string | null>(null);
  const [tab, setTab] = useState<"compare" | "examples" | "decide">("compare");

  return (
    <div className="w-full max-w-4xl mx-auto my-6 rounded-2xl overflow-hidden border border-slate-200 dark:border-slate-700 shadow-xl bg-white dark:bg-slate-900">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-700 via-slate-600 to-slate-800 px-6 py-4">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <Shield size={16} className="text-blue-300" /> 近似算法
          <span className="text-slate-400 mx-2">vs</span>
          <Zap size={16} className="text-amber-300" /> 启发式算法
        </h3>
        <p className="text-xs text-slate-300 mt-0.5">
          两类方法各有权衡，选择取决于问题性质与工程需求
        </p>
      </div>

      {/* Tab bar */}
      <div className="flex border-b border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800">
        {(["compare", "examples", "decide"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`flex-1 text-xs font-semibold py-2.5 transition-colors ${
              tab === t
                ? "border-b-2 border-indigo-500 text-indigo-600 dark:text-indigo-400 bg-white dark:bg-slate-900"
                : "text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
            }`}
          >
            {t === "compare" ? "📊 多维对比" : t === "examples" ? "🗂️ 经典问题" : "🧭 如何选择"}
          </button>
        ))}
      </div>

      <div className="p-5">
        <AnimatePresence mode="wait">
          {tab === "compare" && (
            <motion.div key="compare" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              {/* Column headers */}
              <div className="grid grid-cols-[120px_1fr_1fr] gap-3 mb-3">
                <div />
                <div className="flex items-center justify-center gap-2 text-sm font-bold text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20 rounded-xl py-2">
                  <Shield size={14} /> 近似算法
                </div>
                <div className="flex items-center justify-center gap-2 text-sm font-bold text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/20 rounded-xl py-2">
                  <Zap size={14} /> 启发式算法
                </div>
              </div>

              {/* Dimension rows */}
              <div className="space-y-2">
                {DIMENSIONS.map((dim) => (
                  <div key={dim.id}>
                    <div
                      className={`grid grid-cols-[120px_1fr_1fr] gap-3 cursor-pointer rounded-xl transition-colors ${activeRow === dim.id ? "bg-slate-50 dark:bg-slate-800" : "hover:bg-slate-50 dark:hover:bg-slate-800/50"}`}
                      onClick={() => setActiveRow(dim.id === activeRow ? null : dim.id)}
                    >
                      {/* Dimension label */}
                      <div className="flex flex-col justify-center py-2 px-2">
                        <span className="text-lg leading-none">{dim.icon}</span>
                        <span className="text-[11px] font-semibold text-slate-600 dark:text-slate-300 mt-1 leading-tight">{dim.label}</span>
                      </div>

                      {/* Approx cell */}
                      <div className={`p-2.5 rounded-xl border text-xs ${RATING_BG[dim.approx.rating]}`}>
                        <div className="flex items-center gap-1.5 mb-1">
                          {RATING_ICON[dim.approx.rating]}
                          <span className="font-semibold text-slate-700 dark:text-slate-200">
                            {dim.approx.rating === "yes" ? "优" : dim.approx.rating === "partial" ? "部分" : "劣"}
                          </span>
                        </div>
                        <span className="text-slate-600 dark:text-slate-300 leading-tight">{dim.approx.text}</span>
                      </div>

                      {/* Heuristic cell */}
                      <div className={`p-2.5 rounded-xl border text-xs ${RATING_BG[dim.heuristic.rating]}`}>
                        <div className="flex items-center gap-1.5 mb-1">
                          {RATING_ICON[dim.heuristic.rating]}
                          <span className="font-semibold text-slate-700 dark:text-slate-200">
                            {dim.heuristic.rating === "yes" ? "优" : dim.heuristic.rating === "partial" ? "部分" : "劣"}
                          </span>
                        </div>
                        <span className="text-slate-600 dark:text-slate-300 leading-tight">{dim.heuristic.text}</span>
                      </div>
                    </div>

                    <AnimatePresence>
                      {activeRow === dim.id && dim.note && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: "auto", opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          className="overflow-hidden"
                        >
                          <div className="mx-2 mb-2 px-3 py-2 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg border border-indigo-100 dark:border-indigo-800 text-xs text-indigo-700 dark:text-indigo-300 flex items-start gap-2">
                            <HelpCircle size={13} className="flex-shrink-0 mt-0.5" />
                            {dim.note}
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                ))}
              </div>

              <div className="mt-3 text-[10px] text-slate-400 text-center">点击行查看补充说明</div>
            </motion.div>
          )}

          {tab === "examples" && (
            <motion.div key="examples" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <div className="overflow-x-auto">
                <table className="w-full text-xs border-collapse">
                  <thead>
                    <tr className="bg-slate-50 dark:bg-slate-800">
                      <th className="text-left p-3 font-semibold text-slate-600 dark:text-slate-300 rounded-tl-xl">问题</th>
                      <th className="text-left p-3 font-semibold text-blue-600 dark:text-blue-400">
                        <div className="flex items-center gap-1"><Shield size={12} /> 最佳近似算法</div>
                      </th>
                      <th className="text-left p-3 font-semibold text-amber-600 dark:text-amber-400">
                        <div className="flex items-center gap-1"><Zap size={12} /> 常用启发式</div>
                      </th>
                      <th className="text-center p-3 font-semibold text-slate-600 dark:text-slate-300 rounded-tr-xl">实践推荐</th>
                    </tr>
                  </thead>
                  <tbody>
                    {EXAMPLES.map((ex, i) => (
                      <motion.tr
                        key={ex.problem}
                        initial={{ opacity: 0, y: 5 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.05 }}
                        className={`border-t border-slate-100 dark:border-slate-800 ${i % 2 === 0 ? "" : "bg-slate-50/50 dark:bg-slate-800/30"}`}
                      >
                        <td className="p-3 font-semibold text-slate-700 dark:text-slate-200">{ex.problem}</td>
                        <td className="p-3 text-blue-700 dark:text-blue-300 font-mono">{ex.approx}</td>
                        <td className="p-3 text-amber-700 dark:text-amber-300">{ex.heuristic}</td>
                        <td className="p-3 text-center">
                          {ex.winner === "approx" ? (
                            <span className="px-2 py-0.5 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 font-semibold text-[10px]">近似算法</span>
                          ) : ex.winner === "heuristic" ? (
                            <span className="px-2 py-0.5 rounded-full bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 font-semibold text-[10px]">启发式</span>
                          ) : (
                            <span className="px-2 py-0.5 rounded-full bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 font-semibold text-[10px]">均可</span>
                          )}
                        </td>
                      </motion.tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </motion.div>
          )}

          {tab === "decide" && (
            <motion.div key="decide" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <div className="text-xs font-bold text-slate-600 dark:text-slate-300 mb-4 text-center">
                决策流程图 — 面对 NP-hard 问题时如何选择方法？
              </div>
              <div className="space-y-3">
                {[
                  {
                    q: "需要理论质量保证？",
                    yes: "→ 寻找近似算法",
                    no: "→ 启发式也可接受",
                    color: "border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20",
                  },
                  {
                    q: "问题是否存在已知近似方案？（查阅 Approximation Algorithms 文献）",
                    yes: "→ 使用近似算法，获得 ρ-近似保证",
                    no: "→ 问题可能是 APX-hard，转向启发式",
                    color: "border-indigo-200 dark:border-indigo-800 bg-indigo-50 dark:bg-indigo-900/20",
                  },
                  {
                    q: "实例规模是否超过近似算法的实际限制？",
                    yes: "→ 考虑更快速的启发式（SA/GA/局部搜索）",
                    no: "→ 理论近似算法通常足够",
                    color: "border-violet-200 dark:border-violet-800 bg-violet-50 dark:bg-violet-900/20",
                  },
                  {
                    q: "是否有足够时间进行参数调优？",
                    yes: "→ 元启发式（SA/GA）可获得更好实际效果",
                    no: "→ 使用近似算法（无需调参，行为可预测）",
                    color: "border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-900/20",
                  },
                ].map((item, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.1 }}
                    className={`rounded-xl border p-3 ${item.color}`}
                  >
                    <div className="text-xs font-bold text-slate-700 dark:text-slate-200 mb-2">
                      Q{i + 1}. {item.q}
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="text-[11px] text-emerald-700 dark:text-emerald-400 flex items-start gap-1">
                        <CheckCircle size={12} className="flex-shrink-0 mt-0.5" />
                        <span><strong>是</strong>：{item.yes}</span>
                      </div>
                      <div className="text-[11px] text-rose-700 dark:text-rose-400 flex items-start gap-1">
                        <XCircle size={12} className="flex-shrink-0 mt-0.5" />
                        <span><strong>否</strong>：{item.no}</span>
                      </div>
                    </div>
                  </motion.div>
                ))}

                {/* Bottom rule */}
                <div className="bg-slate-50 dark:bg-slate-800 rounded-xl p-4 border border-slate-200 dark:border-slate-700 text-center">
                  <div className="text-xs font-bold text-slate-600 dark:text-slate-300 mb-2">
                    🏆 工程实践黄金法则
                  </div>
                  <div className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                    若存在近似算法：<span className="text-blue-600 dark:text-blue-400 font-semibold">先用近似算法建立基线（有界）</span>，<br/>
                    再用启发式尝试提升实际性能，<br/>
                    最终基于实验数据做出决定。
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

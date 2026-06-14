"use client";
import React, { useState, useCallback } from "react";

/* ─── Types ──────────────────────────────────────────────────────────────── */

interface Algorithm {
  id: "sort" | "heap" | "quickselect";
  name: string;
  complexity: string;
  worstCase: string;
  spaceComplexity: string;
  colorClass: string;       // Tailwind gradient (light)
  darkColorClass: string;   // Tailwind gradient (dark)
  barColor: string;
  textColor: string;
  bgSoft: string;
  borderColor: string;
  pros: string[];
  cons: string[];
  bestFor: string;
}

/* ─── Config ─────────────────────────────────────────────────────────────── */

const ALGORITHMS: Algorithm[] = [
  {
    id: "sort",
    name: "完全排序",
    complexity: "O(n log n)",
    worstCase: "O(n log n)",
    spaceComplexity: "O(1) ~ O(n)",
    colorClass: "from-orange-400 to-amber-400",
    darkColorClass: "dark:from-orange-600 dark:to-amber-600",
    barColor: "bg-orange-400 dark:bg-orange-500",
    textColor: "text-orange-600 dark:text-orange-400",
    bgSoft: "bg-orange-50 dark:bg-orange-900/20",
    borderColor: "border-orange-200 dark:border-orange-800",
    pros: ["实现最简单", "结果全局有序", "适合小数据"],
    cons: ["排序了大量不需要的元素", "n 大时效率低"],
    bestFor: "n 较小 / 需要完整排序 / 频繁查询",
  },
  {
    id: "heap",
    name: "大小为 k 的堆",
    complexity: "O(n log k)",
    worstCase: "O(n log k)",
    spaceComplexity: "O(k)",
    colorClass: "from-sky-400 to-blue-500",
    darkColorClass: "dark:from-sky-600 dark:to-blue-600",
    barColor: "bg-sky-400 dark:bg-sky-500",
    textColor: "text-sky-600 dark:text-sky-400",
    bgSoft: "bg-sky-50 dark:bg-sky-900/20",
    borderColor: "border-sky-200 dark:border-sky-800",
    pros: ["流式处理友好，O(k) 空间", "k ≪ n 时远快于排序", "在线算法"],
    cons: ["不保证 Top-k 内部有序", "k 接近 n 时退化为 O(n log n)"],
    bestFor: "流式数据 / k ≪ n / 内存受限",
  },
  {
    id: "quickselect",
    name: "快速选择 (QuickSelect)",
    complexity: "O(n) 期望",
    worstCase: "O(n²) 最坏",
    spaceComplexity: "O(log n) 递归",
    colorClass: "from-emerald-400 to-teal-500",
    darkColorClass: "dark:from-emerald-600 dark:to-teal-600",
    barColor: "bg-emerald-400 dark:bg-emerald-500",
    textColor: "text-emerald-600 dark:text-emerald-400",
    bgSoft: "bg-emerald-50 dark:bg-emerald-900/20",
    borderColor: "border-emerald-200 dark:border-emerald-800",
    pros: ["平均 O(n)，线性时间最快", "in-place，O(log n) 空间"],
    cons: ["不稳定，最坏 O(n²)", "不支持流式，需全部数据"],
    bestFor: "大批量一次性查询 / 对绝对第 k 小感兴趣",
  },
];

/* ─── Math helpers ───────────────────────────────────────────────────────── */

function opCount(algoId: Algorithm["id"], n: number, k: number): number {
  switch (algoId) {
    case "sort":         return Math.round(n * Math.log2(n));
    case "heap":         return Math.round(n * Math.log2(Math.max(k, 2)));
    case "quickselect":  return Math.round(2 * n); // avg ~2n
  }
}

const N_OPTIONS = [100, 1_000, 10_000, 100_000];
const K_FRACS  = [
  { label: "k = 10",     frac: null, abs: 10 },
  { label: "k = √n",     frac: "sqrt", abs: null },
  { label: "k = n/10",   frac: "tenth", abs: null },
  { label: "k = n/2",    frac: "half", abs: null },
];

function resolveK(n: number, kFrac: typeof K_FRACS[number]) {
  if (kFrac.abs !== null) return Math.min(kFrac.abs, n);
  if (kFrac.frac === "sqrt") return Math.max(1, Math.round(Math.sqrt(n)));
  if (kFrac.frac === "tenth") return Math.max(1, Math.round(n / 10));
  if (kFrac.frac === "half") return Math.max(1, Math.round(n / 2));
  return 10;
}

function formatNum(x: number) {
  if (x >= 1_000_000) return (x / 1_000_000).toFixed(1) + "M";
  if (x >= 1_000)     return (x / 1_000).toFixed(1) + "K";
  return x.toString();
}

/* ─── Scenario Recommendation ───────────────────────────────────────────── */

function recommend(n: number, k: number): Algorithm["id"] {
  const ratio = k / n;
  if (ratio > 0.3) return "sort";
  if (k === 1 || n > 10_000) return "quickselect";
  return "heap";
}

/* ─── Main Component ─────────────────────────────────────────────────────── */

export default function TopKComparison() {
  const [nIdx, setNIdx]     = useState(2);   // n = 10,000
  const [kFracIdx, setKFracIdx] = useState(0);
  const [activeTab, setActiveTab] = useState<"visual" | "table" | "guide">("visual");

  const n = N_OPTIONS[nIdx];
  const kFrac = K_FRACS[kFracIdx];
  const k = resolveK(n, kFrac);

  const counts = ALGORITHMS.map(a => ({ ...a, ops: opCount(a.id, n, k) }));
  const maxOps = Math.max(...counts.map(c => c.ops));
  const recId  = recommend(n, k);

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">

      {/* ── Header ── */}
      <div className="bg-gradient-to-r from-violet-500 via-purple-500 to-fuchsia-500 p-5">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-white font-bold text-lg">Top-K 算法横向对比</h3>
            <p className="text-purple-100 text-sm mt-0.5">调节 n / k，实时比较三种算法的操作数量与适用场景</p>
          </div>
          <div className="text-3xl text-white/80 font-bold">K</div>
        </div>
      </div>

      <div className="p-5 space-y-5">

        {/* ── Controls ── */}
        <div className="grid grid-cols-2 gap-4">
          {/* n */}
          <div className="space-y-2">
            <label className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
              数组大小 n = {n.toLocaleString()}
            </label>
            <div className="flex gap-1">
              {N_OPTIONS.map((v, i) => (
                <button key={i} onClick={() => setNIdx(i)}
                  className={`flex-1 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
                    i === nIdx
                      ? "bg-purple-500 border-purple-500 text-white"
                      : "border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800"
                  }`}>
                  {v >= 1000 ? v / 1000 + "K" : v}
                </button>
              ))}
            </div>
          </div>

          {/* k */}
          <div className="space-y-2">
            <label className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">
              目标 k = {k.toLocaleString()} ({((k / n) * 100).toFixed(1)}%)
            </label>
            <div className="flex gap-1 flex-wrap">
              {K_FRACS.map((f, i) => (
                <button key={i} onClick={() => setKFracIdx(i)}
                  className={`flex-1 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
                    i === kFracIdx
                      ? "bg-purple-500 border-purple-500 text-white"
                      : "border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800"
                  }`}>
                  {f.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* ── Tabs ── */}
        <div className="flex gap-1 border-b border-slate-200 dark:border-slate-700">
          {(["visual", "table", "guide"] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors -mb-px ${
                activeTab === tab
                  ? "border-purple-500 text-purple-600 dark:text-purple-400"
                  : "border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
              }`}>
              {tab === "visual" ? "📊 操作数对比" : tab === "table" ? "📋 复杂度表格" : "🧭 场景推荐"}
            </button>
          ))}
        </div>

        {/* ── Tab: Visual bars ── */}
        {activeTab === "visual" && (
          <div className="space-y-3">
            {counts.map(algo => {
              const pct = maxOps > 0 ? (algo.ops / maxOps) * 100 : 0;
              const isRec = algo.id === recId;
              return (
                <div key={algo.id}
                  className={`rounded-xl border-2 transition-all ${
                    isRec
                      ? `${algo.bgSoft} ${algo.borderColor} shadow-md`
                      : "border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-800/40"
                  } p-4`}>
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <div className={`text-sm font-bold ${isRec ? algo.textColor : "text-slate-700 dark:text-slate-300"}`}>
                        {algo.name}
                      </div>
                      {isRec && (
                        <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${algo.bgSoft} ${algo.textColor} border ${algo.borderColor}`}>
                          推荐 ✓
                        </span>
                      )}
                    </div>
                    <div className={`font-mono text-sm font-semibold ${isRec ? algo.textColor : "text-slate-500 dark:text-slate-400"}`}>
                      {formatNum(algo.ops)} 次操作
                    </div>
                  </div>
                  <div className="h-6 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${algo.barColor} rounded-full transition-all duration-700 flex items-center justify-end pr-2`}
                      style={{ width: `${Math.max(pct, 2)}%` }}>
                      {pct > 15 && (
                        <span className="text-white text-xs font-bold">{algo.complexity}</span>
                      )}
                    </div>
                  </div>
                  {pct <= 15 && (
                    <div className={`text-xs font-mono mt-1 ${algo.textColor}`}>{algo.complexity}</div>
                  )}
                </div>
              );
            })}

            <p className="text-xs text-slate-400 dark:text-slate-500">
              操作数按近似公式估算：排序≈n·log₂n，堆≈n·log₂k，快速选择≈2n（期望）
            </p>
          </div>
        )}

        {/* ── Tab: Complexity table ── */}
        {activeTab === "table" && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-200 dark:border-slate-700">
                  <th className="text-left py-2 pr-4 text-slate-500 dark:text-slate-400 font-semibold">算法</th>
                  <th className="text-center py-2 px-2 text-slate-500 dark:text-slate-400 font-semibold">平均时间</th>
                  <th className="text-center py-2 px-2 text-slate-500 dark:text-slate-400 font-semibold">最坏时间</th>
                  <th className="text-center py-2 px-2 text-slate-500 dark:text-slate-400 font-semibold">空间</th>
                  <th className="text-center py-2 px-2 text-slate-500 dark:text-slate-400 font-semibold">流式</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {ALGORITHMS.map(a => (
                  <tr key={a.id} className={a.id === recId ? a.bgSoft : ""}>
                    <td className="py-3 pr-4">
                      <div className={`font-semibold ${a.textColor}`}>{a.name}</div>
                      {a.id === recId && (
                        <div className="text-[10px] text-slate-400 dark:text-slate-500 mt-0.5">← 当前推荐</div>
                      )}
                    </td>
                    <td className={`text-center py-3 px-2 font-mono font-semibold ${a.textColor}`}>
                      {a.complexity}
                    </td>
                    <td className="text-center py-3 px-2 font-mono text-slate-500 dark:text-slate-400">
                      {a.worstCase}
                    </td>
                    <td className="text-center py-3 px-2 font-mono text-slate-500 dark:text-slate-400">
                      {a.spaceComplexity}
                    </td>
                    <td className="text-center py-3 px-2">
                      {a.id === "heap" ? "✅" : "❌"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            <div className="mt-4 rounded-xl border border-slate-200 dark:border-slate-700 p-3 space-y-2">
              <div className="text-xs font-semibold text-slate-600 dark:text-slate-300">k/n 比率与算法性能规律</div>
              <div className="space-y-1 text-xs text-slate-500 dark:text-slate-400">
                <div>• k ≪ n（如 k = O(1) 或 k = √n）：堆 ≈ O(n)，与快速选择持平</div>
                <div>• k ≈ n/2：快速选择 O(n) 占优；堆退化到 O(n log n)</div>
                <div>• k ≈ n：三者相近，排序代码最简单</div>
                <div>• 当 n → ∞ 且 k 固定：堆每次 push/pop O(log k)，实践中最快</div>
              </div>
            </div>
          </div>
        )}

        {/* ── Tab: Scenario guide ── */}
        {activeTab === "guide" && (
          <div className="space-y-3">
            {ALGORITHMS.map(a => (
              <div key={a.id}
                className={`rounded-xl border-2 p-4 ${a.id === recId ? `${a.bgSoft} ${a.borderColor}` : "border-slate-100 dark:border-slate-800"}`}>
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <div className={`font-bold ${a.textColor}`}>{a.name}</div>
                    <div className="text-xs text-slate-500 dark:text-slate-400 mt-0.5 font-mono">{a.complexity}</div>
                  </div>
                  {a.id === recId && (
                    <span className={`text-xs font-bold px-2 py-0.5 rounded ${a.bgSoft} ${a.textColor} border ${a.borderColor} flex-none`}>
                      当前推荐
                    </span>
                  )}
                </div>
                <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <div className="font-semibold text-slate-600 dark:text-slate-300 mb-1">✅ 优点</div>
                    <ul className="space-y-0.5 text-slate-500 dark:text-slate-400">
                      {a.pros.map((p, i) => <li key={i}>· {p}</li>)}
                    </ul>
                  </div>
                  <div>
                    <div className="font-semibold text-slate-600 dark:text-slate-300 mb-1">⚠️ 缺点</div>
                    <ul className="space-y-0.5 text-slate-500 dark:text-slate-400">
                      {a.cons.map((c, i) => <li key={i}>· {c}</li>)}
                    </ul>
                  </div>
                </div>
                <div className={`mt-3 text-xs font-medium px-2 py-1 rounded ${a.bgSoft} ${a.textColor} border ${a.borderColor}`}>
                  🎯 最佳场景：{a.bestFor}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* ── Recommendation banner ── */}
        {(() => {
          const rec = ALGORITHMS.find(a => a.id === recId)!;
          return (
            <div className={`rounded-xl ${rec.bgSoft} border ${rec.borderColor} p-3 flex items-center gap-3`}>
              <div className={`w-8 h-8 rounded-lg bg-gradient-to-br ${rec.colorClass} ${rec.darkColorClass} flex items-center justify-center text-white font-bold text-sm flex-none`}>
                ✓
              </div>
              <div>
                <div className={`text-sm font-bold ${rec.textColor}`}>
                  当前配置推荐：{rec.name}
                </div>
                <div className="text-xs text-slate-500 dark:text-slate-400">
                  n = {n.toLocaleString()}, k = {k.toLocaleString()} ({((k / n) * 100).toFixed(1)}%) → {rec.bestFor}
                </div>
              </div>
            </div>
          );
        })()}
      </div>
    </div>
  );
}

"use client";
import React, { useState } from "react";

/* ─── Data ───────────────────────────────────────────────────────────────── */

type Scenario = "general" | "small" | "nearly_sorted" | "duplicates" | "stable" | "memory";

interface SortAlgo {
  name: string;
  best: string;
  avg: string;
  worst: string;
  space: string;
  stable: boolean;
  inPlace: boolean;          // O(1) extra space (not counting stack)
  cacheRating: 1 | 2 | 3;   // 1=poor, 2=ok, 3=good
  practiceRating: 1 | 2 | 3;
  bestScenario: Scenario[];  // where this algo shines
  worstScenario: string[];   // where it struggles
  notes: string;
  tag: "comparison" | "linear";
  color: string;             // Tailwind accent color class
}

const ALGOS: SortAlgo[] = [
  {
    name: "插入排序", best: "O(n)", avg: "O(n²)", worst: "O(n²)",
    space: "O(1)", stable: true, inPlace: true, cacheRating: 3, practiceRating: 1,
    bestScenario: ["small", "nearly_sorted"], worstScenario: ["逆序输入", "大规模随机"],
    notes: "几乎有序时接近线性；n<20 时常数因子最小，std::sort 对小子数组使用插入排序",
    tag: "comparison", color: "indigo",
  },
  {
    name: "归并排序", best: "Θ(n log n)", avg: "Θ(n log n)", worst: "Θ(n log n)",
    space: "O(n)", stable: true, inPlace: false, cacheRating: 2, practiceRating: 2,
    bestScenario: ["stable", "general"], worstScenario: ["内存受限场景"],
    notes: "最坏情况无退化；链表排序首选；外排序（多路归并）的核心",
    tag: "comparison", color: "sky",
  },
  {
    name: "快速排序", best: "O(n log n)", avg: "O(n log n)", worst: "O(n²)",
    space: "O(log n)", stable: false, inPlace: true, cacheRating: 3, practiceRating: 3,
    bestScenario: ["general", "duplicates"], worstScenario: ["极端有序输入（未随机化）"],
    notes: "实践最快（缓存命中高、常数小）；随机化后退化概率极低；std::sort 基础",
    tag: "comparison", color: "amber",
  },
  {
    name: "堆排序", best: "Θ(n log n)", avg: "Θ(n log n)", worst: "Θ(n log n)",
    space: "O(1)", stable: false, inPlace: true, cacheRating: 1, practiceRating: 1,
    bestScenario: ["memory"], worstScenario: ["缓存敏感场景", "大规模数据"],
    notes: "严格 O(1) 空间且最坏保证；但缓存失效严重，实际比快排慢 2-3×",
    tag: "comparison", color: "rose",
  },
  {
    name: "TimSort", best: "O(n)", avg: "O(n log n)", worst: "O(n log n)",
    space: "O(n)", stable: true, inPlace: false, cacheRating: 3, practiceRating: 3,
    bestScenario: ["nearly_sorted", "stable", "general"], worstScenario: ["纯随机无结构数据"],
    notes: "Python / Java 标准库；利用自然 Run，几乎有序时接近 O(n)；工程最佳稳定排序",
    tag: "comparison", color: "violet",
  },
  {
    name: "计数排序", best: "Θ(n+k)", avg: "Θ(n+k)", worst: "Θ(n+k)",
    space: "O(k)", stable: true, inPlace: false, cacheRating: 3, practiceRating: 2,
    bestScenario: ["duplicates", "general"], worstScenario: ["k >> n（键值范围太大）"],
    notes: "不基于比较；需要 k=O(n)；年龄/评级/字符等有界整数场景完美",
    tag: "linear", color: "emerald",
  },
  {
    name: "基数排序", best: "Θ(d(n+b))", avg: "Θ(d(n+b))", worst: "Θ(d(n+b))",
    space: "O(n+b)", stable: true, inPlace: false, cacheRating: 2, practiceRating: 2,
    bestScenario: ["general", "stable"], worstScenario: ["可变长度字符串", "d 很大"],
    notes: "32 位整数按字节 d=4 轮；字符串排序常用；依赖稳定子排序（计数排序）",
    tag: "linear", color: "teal",
  },
  {
    name: "桶排序", best: "O(n)", avg: "O(n)", worst: "O(n²)",
    space: "O(n)", stable: true, inPlace: false, cacheRating: 2, practiceRating: 1,
    bestScenario: ["general"], worstScenario: ["数据分布不均（集中到少数桶）"],
    notes: "要求输入均匀分布；实践中需谨慎验证分布假设；浮点/概率值场景合适",
    tag: "linear", color: "lime",
  },
];

type SortKey = "name" | "avg" | "space" | "stable" | "cacheRating" | "practiceRating";

const SCENARIO_INFO: Record<Scenario, { label: string; icon: string; tooltip: string }> = {
  general:       { label: "通用场景", icon: "⚡", tooltip: "随机数据，无特殊假设" },
  small:         { label: "小数组", icon: "📦", tooltip: "n < 20 的小规模数据" },
  nearly_sorted: { label: "几乎有序", icon: "📈", tooltip: "大多数元素已在正确位置" },
  duplicates:    { label: "大量重复", icon: "🔁", tooltip: "相同元素占比 > 30%" },
  stable:        { label: "需稳定排序", icon: "🔒", tooltip: "相同键值的元素必须保持原始顺序" },
  memory:        { label: "内存受限", icon: "💾", tooltip: "额外空间必须 O(1)" },
};

/* ─── Star rating ─────────────────────────────────────────────────────────── */

function Stars({ count, max = 3 }: { count: number; max?: number }) {
  return (
    <div className="flex gap-0.5">
      {Array.from({ length: max }, (_, i) => (
        <span key={i} className={`text-xs ${i < count ? "text-amber-400" : "text-slate-200 dark:text-slate-700"}`}>★</span>
      ))}
    </div>
  );
}

/* ─── Complexity badge ────────────────────────────────────────────────────── */

function CmplxBadge({ text, type }: { text: string; type: "best" | "avg" | "worst" }) {
  const colors: Record<string, string> = {
    best:  "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300",
    avg:   "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400",
    worst: "bg-rose-100 dark:bg-rose-900/30 text-rose-700 dark:text-rose-300",
  };
  // Highlight linear-time complexities
  const isLinear = text.includes("n+k") || text.includes("d(n") || (text.match(/O\(n\)/) && !text.includes("log"));
  return (
    <span className={`inline-block px-1.5 py-0.5 rounded text-[11px] font-mono font-semibold ${colors[type]} ${isLinear ? "ring-1 ring-emerald-400 dark:ring-emerald-600" : ""}`}>
      {text}
    </span>
  );
}

/* ─── Main component ─────────────────────────────────────────────────────── */

export default function SortingComplexityTable() {
  const [scenario, setScenario] = useState<Scenario | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("practiceRating");
  const [sortDesc, setSortDesc] = useState(true);
  const [showLinear, setShowLinear] = useState(true);
  const [showComparison, setShowComparison] = useState(true);
  const [hovered, setHovered] = useState<string | null>(null);

  // Filter and sort
  let displayed = ALGOS.filter(a => {
    if (!showLinear && a.tag === "linear") return false;
    if (!showComparison && a.tag === "comparison") return false;
    if (scenario && !a.bestScenario.includes(scenario)) return false;
    return true;
  });

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortDesc(d => !d);
    else { setSortKey(key); setSortDesc(true); }
  };

  // Sort
  displayed = [...displayed].sort((a, b) => {
    let av: number | string, bv: number | string;
    switch (sortKey) {
      case "stable":        av = +a.stable; bv = +b.stable; break;
      case "cacheRating":   av = a.cacheRating; bv = b.cacheRating; break;
      case "practiceRating": av = a.practiceRating; bv = b.practiceRating; break;
      default: return 0;
    }
    if (typeof av === "number" && typeof bv === "number") {
      return sortDesc ? bv - av : av - bv;
    }
    return 0;
  });

  const ACCENT_RINGS: Record<string, string> = {
    indigo: "ring-indigo-400", sky: "ring-sky-400", amber: "ring-amber-400",
    rose: "ring-rose-400", violet: "ring-violet-400", emerald: "ring-emerald-400",
    teal: "ring-teal-400", lime: "ring-lime-400",
  };

  const ACCENT_BG: Record<string, string> = {
    indigo: "bg-indigo-50 dark:bg-indigo-900/20", sky: "bg-sky-50 dark:bg-sky-900/20",
    amber: "bg-amber-50 dark:bg-amber-900/20", rose: "bg-rose-50 dark:bg-rose-900/20",
    violet: "bg-violet-50 dark:bg-violet-900/20", emerald: "bg-emerald-50 dark:bg-emerald-900/20",
    teal: "bg-teal-50 dark:bg-teal-900/20", lime: "bg-lime-50 dark:bg-lime-900/20",
  };

  const SortHeaderBtn = ({ label, k }: { label: string; k: SortKey }) => (
    <button onClick={() => handleSort(k)} className={`flex items-center gap-1 text-xs font-semibold transition-colors ${sortKey === k ? "text-indigo-600 dark:text-indigo-400" : "text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"}`}>
      {label}
      <span className="text-[9px]">{sortKey === k ? (sortDesc ? "▼" : "▲") : "⇅"}</span>
    </button>
  );

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 overflow-hidden shadow-sm">
      {/* Header */}
      <div className="px-5 py-4 border-b border-slate-200 dark:border-slate-700">
        <h3 className="font-bold text-slate-800 dark:text-slate-100 text-base">排序算法复杂度全览</h3>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">按场景筛选，点击表头可排序</p>
      </div>

      <div className="p-5 space-y-4">
        {/* Scenario filter */}
        <div>
          <p className="text-xs font-medium text-slate-600 dark:text-slate-400 mb-2">按场景筛选（高亮最优算法）：</p>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setScenario(null)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all ${scenario === null ? "bg-slate-700 dark:bg-slate-300 text-white dark:text-slate-900 border-slate-700 dark:border-slate-300" : "border-slate-200 dark:border-slate-700 text-slate-500 dark:text-slate-400 bg-white dark:bg-slate-800 hover:border-slate-400 dark:hover:border-slate-500"}`}
            >
              全部显示
            </button>
            {(Object.entries(SCENARIO_INFO) as [Scenario, typeof SCENARIO_INFO[Scenario]][]).map(([key, info]) => (
              <button
                key={key}
                onClick={() => setScenario(scenario === key ? null : key)}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all ${scenario === key ? "bg-indigo-600 text-white border-indigo-600 shadow-sm" : "border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400 bg-white dark:bg-slate-800 hover:border-indigo-300 dark:hover:border-indigo-700"}`}
                title={info.tooltip}
              >
                {info.icon} {info.label}
              </button>
            ))}
          </div>
        </div>

        {/* Type filter */}
        <div className="flex items-center gap-4">
          <p className="text-xs font-medium text-slate-600 dark:text-slate-400">类型：</p>
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input type="checkbox" checked={showComparison} onChange={e => setShowComparison(e.target.checked)} className="accent-indigo-500" />
            <span className="text-xs text-slate-600 dark:text-slate-400">基于比较</span>
          </label>
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input type="checkbox" checked={showLinear} onChange={e => setShowLinear(e.target.checked)} className="accent-emerald-500" />
            <span className="text-xs text-slate-600 dark:text-slate-400 flex items-center gap-1">
              线性时间 <span className="text-[9px] text-emerald-600 dark:text-emerald-400 font-semibold ring-1 ring-emerald-400 rounded px-0.5">突破下界</span>
            </span>
          </label>
        </div>

        {/* Table */}
        <div className="overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-700">
          <table className="w-full text-sm min-w-[700px]">
            <thead>
              <tr className="border-b border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/70">
                <th className="text-left px-4 py-3 text-xs font-semibold text-slate-600 dark:text-slate-400 w-32">算法</th>
                <th className="text-center px-3 py-3 text-xs font-semibold text-emerald-600 dark:text-emerald-400">最好情况</th>
                <th className="text-center px-3 py-3 text-xs font-semibold text-slate-600 dark:text-slate-400">平均情况</th>
                <th className="text-center px-3 py-3 text-xs font-semibold text-rose-600 dark:text-rose-400">最坏情况</th>
                <th className="text-center px-3 py-3 text-xs font-semibold text-slate-600 dark:text-slate-400">空间</th>
                <th className="px-3 py-3"><SortHeaderBtn label="稳定" k="stable" /></th>
                <th className="px-3 py-3"><SortHeaderBtn label="缓存" k="cacheRating" /></th>
                <th className="px-3 py-3"><SortHeaderBtn label="实践" k="practiceRating" /></th>
              </tr>
            </thead>
            <tbody>
              {displayed.length === 0 && (
                <tr><td colSpan={8} className="text-center py-8 text-sm text-slate-400 dark:text-slate-600">无符合条件的算法</td></tr>
              )}
              {displayed.map(algo => {
                const isHighlighted = scenario && algo.bestScenario.includes(scenario);
                const isHovered = hovered === algo.name;
                return (
                  <tr
                    key={algo.name}
                    onMouseEnter={() => setHovered(algo.name)}
                    onMouseLeave={() => setHovered(null)}
                    className={`border-b border-slate-100 dark:border-slate-800 transition-colors cursor-default
                      ${isHighlighted ? ACCENT_BG[algo.color] : isHovered ? "bg-slate-50 dark:bg-slate-800/40" : ""}`}
                  >
                    {/* Name */}
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        {isHighlighted && <span className="w-1.5 h-6 rounded-full bg-emerald-400 flex-shrink-0" />}
                        <div>
                          <p className={`font-semibold text-xs ${isHighlighted || isHovered ? "text-slate-800 dark:text-slate-100" : "text-slate-700 dark:text-slate-300"}`}>{algo.name}</p>
                          <span className={`text-[9px] px-1 rounded font-semibold ${algo.tag === "linear" ? "bg-emerald-100 dark:bg-emerald-900/40 text-emerald-600 dark:text-emerald-400" : "bg-slate-100 dark:bg-slate-700 text-slate-500 dark:text-slate-400"}`}>
                            {algo.tag === "linear" ? "线性" : "比较"}
                          </span>
                        </div>
                      </div>
                    </td>
                    {/* Complexities */}
                    <td className="px-3 py-3 text-center"><CmplxBadge text={algo.best} type="best" /></td>
                    <td className="px-3 py-3 text-center"><CmplxBadge text={algo.avg} type="avg" /></td>
                    <td className="px-3 py-3 text-center"><CmplxBadge text={algo.worst} type="worst" /></td>
                    <td className="px-3 py-3 text-center">
                      <span className="text-xs font-mono text-slate-600 dark:text-slate-400">{algo.space}</span>
                    </td>
                    {/* Stable */}
                    <td className="px-3 py-3 text-center">
                      <span className={`text-sm ${algo.stable ? "text-emerald-500" : "text-rose-400"}`}>{algo.stable ? "✓" : "✗"}</span>
                    </td>
                    {/* Cache */}
                    <td className="px-3 py-3"><Stars count={algo.cacheRating} /></td>
                    {/* Practice */}
                    <td className="px-3 py-3"><Stars count={algo.practiceRating} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Notes panel for hovered algo */}
        {hovered && (
          <div className={`rounded-xl border p-4 transition-all text-xs ${ACCENT_BG[ALGOS.find(a => a.name === hovered)?.color ?? "indigo"]} border-slate-200 dark:border-slate-700`}>
            {(() => {
              const a = ALGOS.find(x => x.name === hovered)!;
              return (
                <div className="space-y-1">
                  <p className="font-semibold text-slate-800 dark:text-slate-100">{a.name} — 工程说明</p>
                  <p className="text-slate-600 dark:text-slate-400">{a.notes}</p>
                  <div className="flex gap-4 mt-2">
                    <div>
                      <span className="text-[10px] font-semibold text-emerald-600 dark:text-emerald-400">✓ 适合：</span>
                      <span className="text-slate-600 dark:text-slate-400">{a.bestScenario.map(s => SCENARIO_INFO[s].label).join("、")}</span>
                    </div>
                    <div>
                      <span className="text-[10px] font-semibold text-rose-600 dark:text-rose-400">✗ 慎用：</span>
                      <span className="text-slate-600 dark:text-slate-400">{a.worstScenario.join("、")}</span>
                    </div>
                  </div>
                </div>
              );
            })()}
          </div>
        )}

        {/* Legend */}
        <div className="flex flex-wrap gap-4 text-xs text-slate-500 dark:text-slate-400 pt-2 border-t border-slate-100 dark:border-slate-800">
          <div className="flex items-center gap-1">
            <span className="px-1 rounded ring-1 ring-emerald-400 bg-emerald-100 dark:bg-emerald-900/30 text-[10px] text-emerald-700 dark:text-emerald-300 font-mono">O(n)</span>
            <span>= 线性复杂度（突破比较排序下界）</span>
          </div>
          <div className="flex items-center gap-1">
            <Stars count={3} />
            <span>= 缓存命中率 / 实践推荐度</span>
          </div>
        </div>
      </div>
    </div>
  );
}

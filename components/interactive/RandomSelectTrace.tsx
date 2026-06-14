"use client";
import React, { useState, useCallback } from "react";

/* ─── Types ──────────────────────────────────────────────────────────────── */

type CellRole =
  | "default"    // 不在当前子数组中（已排除）
  | "inactive"   // 当前子数组中，未被标记
  | "pivot"      // 当前 pivot
  | "smaller"    // 确定 < pivot
  | "larger"     // 确定 > pivot
  | "target"     // 找到的目标
  | "boundary";  // 左右边界指示

interface Step {
  title: string;
  detail: string;
  left: number;
  right: number;
  pivotIdx: number | null;
  partitionIdx: number | null;  // pivot 划分后的最终位置
  foundIdx: number | null;
  k: number;                    // 当前寻找「第 k 小」（1-indexed，相对子数组起点）
  phase: "select" | "partition" | "recurse_left" | "recurse_right" | "found";
  highlightFormula?: string;
}

/* ─── Example ────────────────────────────────────────────────────────────── */
// 数组 A = [3, 6, 1, 8, 2, 9, 4, 7, 5]，找第 5 小（= 5）
// pivot 序列（Lomuto）：先选 arr[right]

const ORIGINAL = [3, 6, 1, 8, 2, 9, 4, 7, 5];
const TARGET_K = 5;  // 全局第 5 小

// 预计算 steps（确定性，不用真正跑 QuickSelect，直接写好演示路径）
const STEPS: Step[] = [
  {
    title: "初始：在整个数组中找第 5 小",
    detail: "数组 A = [3,6,1,8,2,9,4,7,5]，n = 9，目标 k = 5（即 5 是第5小）。随机选 pivot = A[8] = 5，开始 Lomuto 划分。",
    left: 0, right: 8, pivotIdx: 8, partitionIdx: null, foundIdx: null,
    k: 5, phase: "select",
    highlightFormula: "pivot = A[8] = 5",
  },
  {
    title: "PARTITION：以 5 为 pivot 划分 A[0..8]",
    detail: "遍历 A[0..7]，凡 ≤ 5 的放左侧，> 5 的放右侧。划分结果：[3,1,2,4,5,9,8,7,6]，pivot 落在下标 q = 4。",
    left: 0, right: 8, pivotIdx: null, partitionIdx: 4, foundIdx: null,
    k: 5, phase: "partition",
    highlightFormula: "k_local = q - left + 1 = 4 - 0 + 1 = 5",
  },
  {
    title: "命中！k_local = 5 = k，直接返回",
    detail: "pivot 5 恰好是当前子数组第 5 小（k_local = 5 = k）。无需继续递归，直接返回 A[4] = 5。",
    left: 0, right: 8, pivotIdx: null, partitionIdx: null, foundIdx: 4,
    k: 5, phase: "found",
    highlightFormula: "i == k_local → return A[q] = 5 ✓",
  },
];

// 演示另一条完整递归路径：找第 7 小
const STEPS_K7: Step[] = [
  {
    title: "初始：在整个数组中找第 7 小",
    detail: "数组 A = [3,6,1,8,2,9,4,7,5]，n = 9，目标 k = 7（即 7 是第7小）。随机选 pivot = A[8] = 5。",
    left: 0, right: 8, pivotIdx: 8, partitionIdx: null, foundIdx: null,
    k: 7, phase: "select",
    highlightFormula: "pivot = A[8] = 5",
  },
  {
    title: "PARTITION：以 5 为 pivot 划分 A[0..8]",
    detail: "划分结果：[3,1,2,4,5,9,8,7,6]，pivot 落在 q = 4，k_local = 4 - 0 + 1 = 5。",
    left: 0, right: 8, pivotIdx: null, partitionIdx: 4, foundIdx: null,
    k: 7, phase: "partition",
    highlightFormula: "k_local = 5，k = 7 > k_local → 递归右半",
  },
  {
    title: "递归右半：在 A[5..8] 找第 (7-5)=2 小",
    detail: "右半子数组 A[5..8] = [9,8,7,6]，新目标 k = 7 - 5 = 2（第2小）。随机选 pivot = A[8] = 6。",
    left: 5, right: 8, pivotIdx: 8, partitionIdx: null, foundIdx: null,
    k: 2, phase: "recurse_right",
    highlightFormula: "pivot = A[8] = 6，i = k - k_local = 7 - 5 = 2",
  },
  {
    title: "PARTITION：以 6 为 pivot 划分 A[5..8]",
    detail: "右半划分结果：[6,8,7,9]，pivot 6 落在 q = 5，k_local = 5 - 5 + 1 = 1。k = 2 > 1 → 再递归右半。",
    left: 5, right: 8, pivotIdx: null, partitionIdx: 5, foundIdx: null,
    k: 2, phase: "partition",
    highlightFormula: "k_local = 1，k = 2 > 1 → 递归右半 A[6..8]",
  },
  {
    title: "递归右半：在 A[6..8] 找第 (2-1)=1 小",
    detail: "子数组 A[6..8] = [8,7,9]，目标 k = 1（最小值）。随机选 pivot = A[8] = 9。",
    left: 6, right: 8, pivotIdx: 8, partitionIdx: null, foundIdx: null,
    k: 1, phase: "recurse_right",
    highlightFormula: "pivot = A[8] = 9，找第 1 小",
  },
  {
    title: "PARTITION：以 9 为 pivot 划分 A[6..8]",
    detail: "划分结果：[8,7,9]，pivot 9 落在 q = 8，k_local = 8 - 6 + 1 = 3。k = 1 < 3 → 递归左半。",
    left: 6, right: 8, pivotIdx: null, partitionIdx: 8, foundIdx: null,
    k: 1, phase: "partition",
    highlightFormula: "k_local = 3，k = 1 < 3 → 递归左半 A[6..7]",
  },
  {
    title: "递归左半：在 A[6..7] 找第 1 小",
    detail: "子数组 A[6..7] = [8,7]，目标 k = 1（最小值）。选 pivot = A[7] = 7。",
    left: 6, right: 7, pivotIdx: 7, partitionIdx: null, foundIdx: null,
    k: 1, phase: "recurse_left",
    highlightFormula: "pivot = A[7] = 7",
  },
  {
    title: "PARTITION：以 7 为 pivot 划分 A[6..7]",
    detail: "划分结果：[7,8]，pivot 7 落在 q = 6，k_local = 6 - 6 + 1 = 1。k = 1 == k_local，命中！",
    left: 6, right: 7, pivotIdx: null, partitionIdx: 6, foundIdx: null,
    k: 1, phase: "partition",
    highlightFormula: "k_local = 1 = k → return A[6] = 7 ✓",
  },
  {
    title: "命中！A[4] = 7 是全局第 7 小",
    detail: "经过 4 层递归，4 次 PARTITION，每次只递归一侧，最终在 A[6] 找到第 7 小的数 = 7。",
    left: 6, right: 7, pivotIdx: null, partitionIdx: null, foundIdx: 6,
    k: 1, phase: "found",
    highlightFormula: "全局第 7 小 = 7 ✓",
  },
];

// 划分后的实际数组状态（对应 STEPS_K7 的每一步）
const ARRAYS_K7: number[][][] = [
  [[3,6,1,8,2,9,4,7,5]],
  [[3,1,2,4,5,9,8,7,6]],
  [[3,1,2,4,5,9,8,7,6]],
  [[3,1,2,4,5,6,8,7,9]],
  [[3,1,2,4,5,6,8,7,9]],
  [[3,1,2,4,5,6,8,7,9]],
  [[3,1,2,4,5,6,7,8,9]],
  [[3,1,2,4,5,6,7,8,9]],
  [[3,1,2,4,5,6,7,8,9]],
];

const ARRAYS_K5: number[][][] = [
  [[3,6,1,8,2,9,4,7,5]],
  [[3,1,2,4,5,9,8,7,6]],
  [[3,1,2,4,5,9,8,7,6]],
];

/* ─── Helpers ────────────────────────────────────────────────────────────── */

function getCellRole(
  idx: number,
  step: Step,
  arr: number[],
  partitionedArr: number[],
): CellRole {
  const { left, right, pivotIdx, partitionIdx, foundIdx, phase } = step;

  if (foundIdx !== null && idx === foundIdx) return "target";
  if (idx < left || idx > right) return "default";

  if (phase === "found") {
    if (idx === foundIdx) return "target";
    return "inactive";
  }

  if (phase === "select" && pivotIdx !== null) {
    if (idx === pivotIdx) return "pivot";
    return "inactive";
  }

  if (phase === "partition" && partitionIdx !== null) {
    if (idx === partitionIdx) return "pivot";
    if (idx < partitionIdx) return "smaller";
    if (idx > partitionIdx) return "larger";
    return "inactive";
  }

  if ((phase === "recurse_left" || phase === "recurse_right") && pivotIdx !== null) {
    if (idx === pivotIdx) return "pivot";
    return "inactive";
  }

  return "inactive";
}

const ROLE_STYLE: Record<CellRole, string> = {
  default:   "bg-slate-100 dark:bg-slate-800/40 text-slate-300 dark:text-slate-600 border-slate-200 dark:border-slate-700/50 scale-90 opacity-50",
  inactive:  "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-300 dark:border-slate-600",
  pivot:     "bg-amber-100 dark:bg-amber-900/50 text-amber-800 dark:text-amber-200 border-amber-400 dark:border-amber-500 ring-2 ring-amber-300 dark:ring-amber-600 scale-110 shadow-lg",
  smaller:   "bg-blue-100 dark:bg-blue-900/40 text-blue-800 dark:text-blue-200 border-blue-400 dark:border-blue-500",
  larger:    "bg-rose-100 dark:bg-rose-900/40 text-rose-800 dark:text-rose-200 border-rose-400 dark:border-rose-500",
  target:    "bg-emerald-100 dark:bg-emerald-900/50 text-emerald-800 dark:text-emerald-200 border-emerald-500 dark:border-emerald-400 ring-2 ring-emerald-400 dark:ring-emerald-500 scale-110 shadow-lg",
  boundary:  "bg-violet-100 dark:bg-violet-900/40 text-violet-800 dark:text-violet-200 border-violet-400 dark:border-violet-500",
};

const PHASE_CONFIG: Record<Step["phase"], { label: string; color: string }> = {
  select:        { label: "选择 Pivot",  color: "bg-amber-500" },
  partition:     { label: "PARTITION",   color: "bg-sky-500" },
  recurse_left:  { label: "递归左半",   color: "bg-blue-500" },
  recurse_right: { label: "递归右半",   color: "bg-violet-500" },
  found:         { label: "找到目标 ✓", color: "bg-emerald-500" },
};

/* ─── Main Component ─────────────────────────────────────────────────────── */

export default function RandomSelectTrace() {
  const [mode, setMode] = useState<"k5" | "k7">("k5");
  const [stepIdx, setStepIdx] = useState(0);

  const steps  = mode === "k5" ? STEPS    : STEPS_K7;
  const arrays = mode === "k5" ? ARRAYS_K5 : ARRAYS_K7;

  const step     = steps[stepIdx];
  const curArray = arrays[Math.min(stepIdx, arrays.length - 1)][0];
  const totalK   = mode === "k5" ? 5 : 7;

  const changeMode = (m: "k5" | "k7") => { setMode(m); setStepIdx(0); };
  const prev = () => setStepIdx(i => Math.max(0, i - 1));
  const next = () => setStepIdx(i => Math.min(steps.length - 1, i + 1));
  const reset = () => setStepIdx(0);

  const phase = PHASE_CONFIG[step.phase];

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">

      {/* ── Header ── */}
      <div className="bg-gradient-to-r from-amber-500 via-orange-500 to-rose-500 p-5">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <h3 className="text-white font-bold text-lg tracking-tight">RANDOMIZED-SELECT 执行追踪</h3>
            <p className="text-amber-100 text-sm mt-0.5">观察每次如何只递归一侧，期望 O(n)</p>
          </div>
          {/* Mode tabs */}
          <div className="flex gap-1 bg-white/20 rounded-lg p-1">
            {(["k5", "k7"] as const).map(m => (
              <button key={m} onClick={() => changeMode(m)}
                className={`px-3 py-1.5 rounded-md text-sm font-semibold transition-all ${
                  mode === m
                    ? "bg-white text-orange-600 shadow"
                    : "text-white hover:bg-white/20"
                }`}>
                找第 {m === "k5" ? "5" : "7"} 小
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="p-5 space-y-5">

        {/* ── Legend ── */}
        <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-xs">
          {[
            { role: "pivot",   label: "当前 Pivot" },
            { role: "smaller", label: "≤ pivot（左侧）" },
            { role: "larger",  label: "> pivot（右侧）" },
            { role: "target",  label: "找到目标" },
            { role: "default", label: "已排除" },
          ].map(({ role, label }) => (
            <div key={role} className="flex items-center gap-1.5">
              <div className={`w-4 h-4 rounded border ${ROLE_STYLE[role as CellRole]} flex-none`} />
              <span className="text-slate-500 dark:text-slate-400">{label}</span>
            </div>
          ))}
        </div>

        {/* ── Array visualization ── */}
        <div className="space-y-2">
          <div className="text-xs font-medium text-slate-500 dark:text-slate-400 uppercase tracking-wider">
            当前数组状态
          </div>
          <div className="flex gap-1.5 flex-wrap">
            {curArray.map((val, idx) => {
              const role = getCellRole(idx, step, ORIGINAL, curArray);
              return (
                <div key={idx} className="flex flex-col items-center gap-0.5">
                  <span className="text-[10px] text-slate-400 dark:text-slate-500 font-mono">{idx}</span>
                  <div className={`
                    w-10 h-10 rounded-xl border-2 flex items-center justify-center
                    font-bold text-base font-mono transition-all duration-300
                    ${ROLE_STYLE[role]}
                  `}>
                    {val}
                  </div>
                  {/* Range indicator under cells */}
                  <div className={`w-2 h-1.5 rounded-full transition-all duration-300 ${
                    idx >= step.left && idx <= step.right
                      ? "bg-slate-400 dark:bg-slate-500"
                      : "bg-transparent"
                  }`} />
                </div>
              );
            })}
          </div>
          {/* Range bar */}
          <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
            <span className="font-mono bg-slate-100 dark:bg-slate-800 px-2 py-0.5 rounded">
              当前区间：A[{step.left}..{step.right}]（共 {step.right - step.left + 1} 个）
            </span>
            <span className="font-mono bg-slate-100 dark:bg-slate-800 px-2 py-0.5 rounded">
              目标：第 {step.k} 小（局部）/ 全局第 {totalK} 小
            </span>
          </div>
        </div>

        {/* ── Step info card ── */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
          <div className={`flex items-center gap-3 px-4 py-3 ${phase.color} bg-opacity-90`}>
            <span className="text-white font-bold text-sm">{phase.label}</span>
            <div className="ml-auto text-white/80 text-xs font-mono">
              步骤 {stepIdx + 1} / {steps.length}
            </div>
          </div>
          <div className="p-4 space-y-3">
            <p className="font-semibold text-slate-800 dark:text-slate-100 text-sm">{step.title}</p>
            <p className="text-slate-600 dark:text-slate-400 text-sm leading-relaxed">{step.detail}</p>
            {step.highlightFormula && (
              <div className="flex items-center gap-2 bg-slate-50 dark:bg-slate-800/60 rounded-lg px-3 py-2 border border-slate-200 dark:border-slate-700">
                <span className="text-amber-500 dark:text-amber-400 text-xs">▶</span>
                <code className="text-sm font-mono text-slate-700 dark:text-slate-200">{step.highlightFormula}</code>
              </div>
            )}
          </div>
        </div>

        {/* ── Recursion direction indicator ── */}
        {(step.phase === "recurse_left" || step.phase === "recurse_right" || step.phase === "partition") && (
          <div className="grid grid-cols-3 gap-2 text-center text-xs">
            <div className={`rounded-lg p-2 border transition-all ${
              step.phase === "recurse_left"
                ? "bg-blue-50 dark:bg-blue-900/30 border-blue-300 dark:border-blue-600 text-blue-700 dark:text-blue-300 font-bold"
                : "bg-slate-50 dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-400 dark:text-slate-500"
            }`}>
              <div className="text-base mb-0.5">◀</div>
              递归左半
            </div>
            <div className="rounded-lg p-2 border bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-700 text-amber-700 dark:text-amber-300">
              <div className="text-base mb-0.5">⊙</div>
              Pivot 位置
            </div>
            <div className={`rounded-lg p-2 border transition-all ${
              step.phase === "recurse_right"
                ? "bg-violet-50 dark:bg-violet-900/30 border-violet-300 dark:border-violet-600 text-violet-700 dark:text-violet-300 font-bold"
                : "bg-slate-50 dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-400 dark:text-slate-500"
            }`}>
              <div className="text-base mb-0.5">▶</div>
              递归右半
            </div>
          </div>
        )}

        {/* ── Progress bar ── */}
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs text-slate-400 dark:text-slate-500">
            <span>进度</span>
            <span>{Math.round((stepIdx / (steps.length - 1)) * 100)}%</span>
          </div>
          <div className="h-1.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-amber-400 to-orange-500 rounded-full transition-all duration-500"
              style={{ width: `${(stepIdx / (steps.length - 1)) * 100}%` }}
            />
          </div>
        </div>

        {/* ── Controls ── */}
        <div className="flex items-center justify-center gap-3">
          <button onClick={reset}
            className="px-3 py-2 rounded-lg text-sm border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors">
            ↺ 重置
          </button>
          <button onClick={prev} disabled={stepIdx === 0}
            className="px-4 py-2 rounded-lg text-sm border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
            ← 上一步
          </button>
          <button onClick={next} disabled={stepIdx === steps.length - 1}
            className="px-4 py-2 rounded-lg text-sm bg-orange-500 hover:bg-orange-600 text-white shadow-sm disabled:opacity-40 disabled:cursor-not-allowed transition-colors font-medium">
            下一步 →
          </button>
        </div>

        {/* ── Key insight ── */}
        <div className="rounded-xl bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 p-3 flex gap-3">
          <span className="text-amber-500 text-lg flex-none">💡</span>
          <p className="text-xs text-amber-800 dark:text-amber-300 leading-relaxed">
            <strong>核心优势</strong>：每次 PARTITION 后只递归<strong>一侧</strong>（而快速排序两侧都要递归）。
            期望每次将问题规模缩小约一半，总期望代价 ≈ n + n/2 + n/4 + … = 2n = <strong>O(n)</strong>。
          </p>
        </div>
      </div>
    </div>
  );
}

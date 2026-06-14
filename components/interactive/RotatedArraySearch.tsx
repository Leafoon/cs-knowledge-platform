"use client";
import React, { useState } from "react";

/* ─── Types ─────────────────────────────────────────────────────────────── */

type Action = "go_left" | "go_right" | "found" | "not_found";

interface SearchStep {
  l: number;
  r: number;
  mid: number;
  leftHalfSorted: boolean;   // true = 左半 [l,mid] 有序；false = 右半 [mid,r] 有序
  targetInSorted: boolean;   // target 是否落在有序那半段
  action: Action;
  explanation: string;
  detail: string;
}

/* ─── Pre-computed search scenarios ─────────────────────────────────────── */

// Array: [4, 5, 6, 7, 0, 1, 2]  rotated at index 4

const ARRAY = [4, 5, 6, 7, 0, 1, 2];

type Scenario = {
  target: number;
  result: number;           // -1 = not found
  steps: SearchStep[];
};

const SCENARIOS: Scenario[] = [
  {
    target: 0,
    result: 4,
    steps: [
      {
        l: 0, r: 6, mid: 3,
        leftHalfSorted: true, targetInSorted: false,
        action: "go_right",
        explanation: "左半段 [0..3] 有序：[4,5,6,7]，target=0 不在 [4,7] → 去右半",
        detail: "arr[l]=4 ≤ arr[mid]=7，左半段有序。\n判断 target=0 是否在 [arr[l], arr[mid]) = [4, 7) → 否 → l = mid+1 = 4",
      },
      {
        l: 4, r: 6, mid: 5,
        leftHalfSorted: true, targetInSorted: true,
        action: "go_left",
        explanation: "左半段 [4..5] 有序：[0,1]，target=0 在 [0,1) → 去左半",
        detail: "arr[l]=0 ≤ arr[mid]=1，左半段有序。\n判断 target=0 是否在 [arr[l], arr[mid]) = [0, 1) → 是 → r = mid-1 = 4",
      },
      {
        l: 4, r: 4, mid: 4,
        leftHalfSorted: true, targetInSorted: true,
        action: "found",
        explanation: "arr[mid]=0 == target=0 → 找到！下标 4",
        detail: "mid = l + (r-l)/2 = 4\narr[4] = 0 == target → return 4 ✓",
      },
    ],
  },
  {
    target: 5,
    result: 1,
    steps: [
      {
        l: 0, r: 6, mid: 3,
        leftHalfSorted: true, targetInSorted: true,
        action: "go_left",
        explanation: "左半段 [0..3] 有序：[4,5,6,7]，target=5 在 [4,7) → 去左半",
        detail: "arr[l]=4 ≤ arr[mid]=7，左半段有序。\ntarget=5 在 [arr[l], arr[mid]) = [4, 7) → 是 → r = mid-1 = 2",
      },
      {
        l: 0, r: 2, mid: 1,
        leftHalfSorted: true, targetInSorted: true,
        action: "found",
        explanation: "arr[mid]=5 == target=5 → 找到！下标 1",
        detail: "mid = 0 + (2-0)/2 = 1\narr[1] = 5 == target → return 1 ✓",
      },
    ],
  },
  {
    target: 3,
    result: -1,
    steps: [
      {
        l: 0, r: 6, mid: 3,
        leftHalfSorted: true, targetInSorted: false,
        action: "go_right",
        explanation: "左半段 [0..3] 有序：[4,5,6,7]，target=3 不在 [4,7) → 去右半",
        detail: "arr[l]=4 ≤ arr[mid]=7，左半段有序。\ntarget=3 在 [4, 7)？→ 否 → l = mid+1 = 4",
      },
      {
        l: 4, r: 6, mid: 5,
        leftHalfSorted: true, targetInSorted: false,
        action: "go_right",
        explanation: "左半段 [4..5] 有序：[0,1]，target=3 不在 [0,1) → 去右半",
        detail: "arr[l]=0 ≤ arr[mid]=1，左半段有序。\ntarget=3 在 [0, 1)？→ 否 → l = mid+1 = 6",
      },
      {
        l: 6, r: 6, mid: 6,
        leftHalfSorted: true, targetInSorted: false,
        action: "go_right",
        explanation: "左半段 [6..6] 有序：[2]，target=3 不在 → l = 7 > r = 6",
        detail: "arr[l]=2 ≤ arr[mid]=2，左半段有序。\ntarget=3 在 [2, 2)？→ 否 → l = mid+1 = 7\nl(7) > r(6) → 退出循环，return -1",
      },
      {
        l: 7, r: 6, mid: 6,   // sentinel: l > r = loop ended
        leftHalfSorted: false, targetInSorted: false,
        action: "not_found",
        explanation: "l(7) > r(6)，搜索空间耗尽 → target=3 不存在！",
        detail: "循环终止条件：l > r\n搜索空间为空，target 不在数组中，return -1",
      },
    ],
  },
];

/* ─── Helper ─────────────────────────────────────────────────────────────── */

type CellRole = "out" | "default" | "sorted" | "unsorted" | "mid" | "found" | "eliminated";

function getCellRole(
  idx: number,
  step: SearchStep,
  scenarioResult: number,
): CellRole {
  const { l, r, mid, leftHalfSorted, action } = step;

  if (action === "not_found") return "eliminated";
  if (idx < l || idx > r) return "out";
  if (action === "found" && idx === mid) return "found";
  if (idx === mid) return "mid";

  if (leftHalfSorted) {
    if (idx >= l && idx <= mid) return "sorted";
    return "unsorted";
  } else {
    if (idx >= mid && idx <= r) return "sorted";
    return "unsorted";
  }
}

const ACTION_META: Record<Action, { label: string; color: string; icon: string }> = {
  go_left:   { label: "→ 去左半",  color: "bg-sky-500 dark:bg-sky-600",     icon: "◀" },
  go_right:  { label: "→ 去右半",  color: "bg-violet-500 dark:bg-violet-600", icon: "▶" },
  found:     { label: "✓ 找到！",  color: "bg-emerald-500 dark:bg-emerald-600", icon: "✓" },
  not_found: { label: "✗ 不存在",  color: "bg-rose-500 dark:bg-rose-600",    icon: "✗" },
};

/* ─── Main Component ─────────────────────────────────────────────────────── */

export default function RotatedArraySearch() {
  const [scenarioIdx, setScenarioIdx] = useState(0);
  const [stepIdx, setStepIdx] = useState(0);

  const sc = SCENARIOS[scenarioIdx];
  const totalSteps = sc.steps.length;
  const step = sc.steps[Math.min(stepIdx, totalSteps - 1)];
  const meta = ACTION_META[step.action];
  const isLastStep = stepIdx >= totalSteps - 1;

  const handleScenario = (i: number) => {
    setScenarioIdx(i);
    setStepIdx(0);
  };

  return (
    <div className="w-full max-w-3xl mx-auto rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl overflow-hidden">

      {/* ── Header ── */}
      <div className="bg-gradient-to-r from-violet-500 via-indigo-500 to-blue-500 p-5">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-white font-bold text-lg tracking-tight">
              旋转有序数组二分搜索
            </h3>
            <p className="text-violet-100 text-sm mt-0.5">
              观察每步如何判断「哪半段有序」，再决定往哪边搜索
            </p>
          </div>
          <div className="text-right text-white/80">
            <div className="text-2xl font-bold font-mono">[4,5,6,7,0,1,2]</div>
            <div className="text-xs mt-0.5">旋转点在下标 4</div>
          </div>
        </div>

        {/* Step progress */}
        <div className="flex gap-1 mt-4">
          {sc.steps.map((_, i) => (
            <button key={i} onClick={() => setStepIdx(i)}
              className={`flex-1 h-1.5 rounded-full transition-all duration-300 ${
                i <= stepIdx ? "bg-white" : "bg-white/25"
              }`}
            />
          ))}
        </div>
      </div>

      <div className="p-5 space-y-4">

        {/* ── Scenario tabs ── */}
        <div className="flex gap-2">
          {SCENARIOS.map((s, i) => (
            <button key={i} onClick={() => handleScenario(i)}
              className={`flex-1 py-2 rounded-xl text-sm font-semibold border transition-all ${
                i === scenarioIdx
                  ? "bg-indigo-50 dark:bg-indigo-900/40 border-indigo-300 dark:border-indigo-600 text-indigo-700 dark:text-indigo-300"
                  : "border-slate-200 dark:border-slate-700 text-slate-500 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800"
              }`}>
              查找 <span className="font-mono">{s.target}</span>
              <span className="ml-1 text-[11px]">
                {s.result === -1 ? "（不存在）" : `（→#${s.result}）`}
              </span>
            </button>
          ))}
        </div>

        {/* ── Array visualization ── */}
        <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/60 p-4">

          {/* Index labels */}
          <div className="flex gap-1 mb-1 px-1">
            {ARRAY.map((_, i) => (
              <div key={i} className="flex-1 text-center text-[10px] text-slate-400 dark:text-slate-500">
                {i}
              </div>
            ))}
          </div>

          {/* Cells */}
          <div className="flex gap-1">
            {ARRAY.map((val, i) => {
              const role = getCellRole(i, step, sc.result);

              const cellStyle: Record<CellRole, string> = {
                out:       "bg-slate-100 dark:bg-slate-800 text-slate-300 dark:text-slate-600 border-slate-200 dark:border-slate-700 opacity-40",
                eliminated:"bg-slate-100 dark:bg-slate-800 text-slate-300 dark:text-slate-600 border-slate-200 dark:border-slate-700 opacity-30",
                default:   "bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-200 border-slate-300 dark:border-slate-600",
                sorted:    "bg-sky-100 dark:bg-sky-900/50 text-sky-800 dark:text-sky-200 border-sky-400 dark:border-sky-500 ring-1 ring-sky-300 dark:ring-sky-600",
                unsorted:  "bg-slate-100 dark:bg-slate-700/60 text-slate-500 dark:text-slate-400 border-slate-300 dark:border-slate-600",
                mid:       "bg-amber-100 dark:bg-amber-900/50 text-amber-800 dark:text-amber-200 border-amber-400 dark:border-amber-500 ring-2 ring-amber-300 dark:ring-amber-600 scale-110 shadow",
                found:     "bg-emerald-100 dark:bg-emerald-900/50 text-emerald-800 dark:text-emerald-200 border-emerald-500 dark:border-emerald-400 ring-2 ring-emerald-400 dark:ring-emerald-500 scale-110 shadow-lg",
              };

              const isL = i === step.l && step.action !== "not_found";
              const isR = i === step.r && step.action !== "not_found";
              const isMid = i === step.mid && step.action !== "not_found";

              return (
                <div key={i} className="flex-1 flex flex-col items-center gap-1">
                  {/* Top indicator: l / mid / r */}
                  <div className="h-5 flex items-center justify-center">
                    {isL && !isMid && <span className="text-[9px] font-bold text-indigo-500 dark:text-indigo-400">l</span>}
                    {isMid && <span className="text-[9px] font-bold text-amber-600 dark:text-amber-400">mid</span>}
                    {isR && !isMid && step.l !== step.r && <span className="text-[9px] font-bold text-indigo-500 dark:text-indigo-400">r</span>}
                    {isL && isR && !isMid && <span className="text-[9px] font-bold text-indigo-500 dark:text-indigo-400">l=r</span>}
                  </div>

                  {/* Cell */}
                  <div className={`
                    w-full aspect-square min-w-0 rounded-xl border-2 flex items-center justify-center
                    font-bold font-mono text-sm transition-all duration-300
                    ${isMid || role === "found" ? "" : ""}
                    ${cellStyle[role]}
                  `}>
                    {val}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Legend */}
          <div className="flex flex-wrap gap-3 mt-3 pt-3 border-t border-slate-200 dark:border-slate-700">
            {[
              { color: "bg-sky-200 dark:bg-sky-800 border-sky-400 dark:border-sky-500", label: "有序半段" },
              { color: "bg-amber-200 dark:bg-amber-800 border-amber-400 dark:border-amber-500", label: "mid" },
              { color: "bg-emerald-200 dark:bg-emerald-800 border-emerald-400 dark:border-emerald-500", label: "找到" },
              { color: "bg-slate-200 dark:bg-slate-700 border-slate-300 dark:border-slate-600 opacity-40", label: "已排除" },
            ].map(({ color, label }) => (
              <div key={label} className="flex items-center gap-1">
                <div className={`w-3.5 h-3.5 rounded border-2 ${color}`} />
                <span className="text-[11px] text-slate-500 dark:text-slate-400">{label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* ── Step explanation card ── */}
        <div className={`rounded-xl border-2 p-4 space-y-3 transition-all duration-300 ${
          step.action === "found"
            ? "border-emerald-300 dark:border-emerald-600 bg-emerald-50 dark:bg-emerald-900/20"
            : step.action === "not_found"
            ? "border-rose-300 dark:border-rose-600 bg-rose-50 dark:bg-rose-900/20"
            : "border-indigo-200 dark:border-indigo-700 bg-indigo-50 dark:bg-indigo-900/20"
        }`}>

          <div className="flex items-start gap-3">
            <div className={`px-2.5 py-1 rounded-lg text-white text-xs font-bold flex-none ${meta.color}`}>
              第 {stepIdx + 1} 步
            </div>
            <div className={`font-semibold text-sm ${
              step.action === "found" ? "text-emerald-700 dark:text-emerald-300"
              : step.action === "not_found" ? "text-rose-700 dark:text-rose-300"
              : "text-indigo-700 dark:text-indigo-300"
            }`}>
              {step.explanation}
            </div>
          </div>

          {/* Range info */}
          {step.action !== "not_found" && (
            <div className="grid grid-cols-3 gap-2 text-center">
              {[
                { label: "l", val: step.l, color: "text-indigo-600 dark:text-indigo-400" },
                { label: "mid", val: step.mid, color: "text-amber-600 dark:text-amber-400" },
                { label: "r", val: step.r, color: "text-indigo-600 dark:text-indigo-400" },
              ].map(({ label, val, color }) => (
                <div key={label} className="rounded-lg bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 py-2">
                  <div className="text-[10px] text-slate-400 dark:text-slate-500 font-medium">{label}</div>
                  <div className={`font-bold font-mono text-lg ${color}`}>{val}</div>
                  <div className="text-[10px] text-slate-400 dark:text-slate-500">arr={ARRAY[val] ?? "—"}</div>
                </div>
              ))}
            </div>
          )}

          {/* Sorted half info */}
          {step.action !== "found" && step.action !== "not_found" && (
            <div className="flex items-center gap-2 text-sm">
              <div className="flex-1 rounded-lg px-3 py-2 bg-sky-100 dark:bg-sky-900/30 border border-sky-200 dark:border-sky-700">
                <span className="text-sky-600 dark:text-sky-300 font-semibold">
                  {step.leftHalfSorted ? "左半" : "右半"}段有序
                </span>
                <span className="text-sky-500 dark:text-sky-400 text-xs ml-2">
                  [{step.leftHalfSorted ? `${step.l}..${step.mid}` : `${step.mid}..${step.r}`}]
                </span>
              </div>
              <div className={`flex-1 rounded-lg px-3 py-2 border ${
                step.targetInSorted
                  ? "bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-700 text-emerald-700 dark:text-emerald-300"
                  : "bg-rose-50 dark:bg-rose-900/20 border-rose-200 dark:border-rose-700 text-rose-700 dark:text-rose-300"
              } text-sm font-semibold`}>
                {sc.target} {step.targetInSorted ? "在其中 ✓" : "不在其中 ✗"}
              </div>
            </div>
          )}

          {/* Pseudo-code detail */}
          <pre className="text-xs font-mono text-slate-600 dark:text-slate-400 bg-white dark:bg-slate-800 rounded-lg p-3 border border-slate-200 dark:border-slate-700 whitespace-pre-wrap leading-relaxed">
            {step.detail}
          </pre>
        </div>

        {/* ── Action badge ── */}
        <div className={`rounded-xl py-3 text-center font-bold text-white text-sm ${meta.color}`}>
          {meta.icon} {meta.label}
          {step.action === "found" && (
            <span className="ml-2 font-mono">下标 {sc.result}，arr[{sc.result}] = {ARRAY[sc.result]}</span>
          )}
        </div>

        {/* ── Controls ── */}
        <div className="flex items-center justify-between pt-1">
          <button onClick={() => setStepIdx(s => Math.max(0, s - 1))} disabled={stepIdx === 0}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
            ← 上一步
          </button>

          <div className="flex gap-1.5">
            {sc.steps.map((_, i) => (
              <button key={i} onClick={() => setStepIdx(i)}
                className={`h-2 rounded-full transition-all ${
                  i === stepIdx ? "bg-indigo-500 w-6" : "bg-slate-300 dark:bg-slate-600 hover:bg-slate-400 dark:hover:bg-slate-500 w-2"
                }`}
              />
            ))}
          </div>

          <button onClick={() => setStepIdx(s => Math.min(totalSteps - 1, s + 1))} disabled={isLastStep}
            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm bg-indigo-500 hover:bg-indigo-600 text-white shadow-sm disabled:opacity-40 disabled:cursor-not-allowed transition-colors font-medium">
            下一步 →
          </button>
        </div>
      </div>
    </div>
  );
}

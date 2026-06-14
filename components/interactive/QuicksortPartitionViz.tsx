"use client";
import React, { useState } from "react";

/* ─── Types ─────────────────────────────────────────────────────────────── */

type CellRole =
  | "default"      // untouched
  | "pivot"        // the chosen pivot
  | "small"        // ≤ pivot, in left zone
  | "large"        // > pivot, in right zone
  | "active_i"     // pointer i is here
  | "active_j"     // pointer j is here
  | "active_ij"    // both i and j (Hoare when they meet)
  | "swapping"     // currently being swapped
  | "done";        // pivot in final position

interface PartitionStep {
  arr: number[];
  roles: CellRole[];   // per-element role
  i: number | null;    // i pointer (-1 = before array)
  j: number | null;    // j pointer
  pivotIdx: number;
  description: string;
  detail: string;
  action?: "swap" | "pivot_place" | "done";
  comparisons: number;
  swaps: number;
}

/* ─── Lomuto trace ──────────────────────────────────────────────────────── */
// arr = [3, 6, 8, 2, 7, 4], pivot = arr[5] = 4
function buildLomutoTrace(): PartitionStep[] {
  const base = [3, 6, 8, 2, 7, 4];
  const steps: PartitionStep[] = [];
  let arr = [...base], cmp = 0, sw = 0;
  const pi = arr.length - 1; // pivot index (last)
  const pivot = arr[pi];

  const makeRoles = (a: number[], i: number, j: number, pivotFinalIdx?: number): CellRole[] =>
    a.map((v, idx) => {
      if (pivotFinalIdx !== undefined && idx === pivotFinalIdx) return "done";
      if (idx === pi) return "pivot";
      if (idx === i && idx === j) return "active_ij";
      if (idx === i) return "active_i";
      if (idx === j) return "active_j";
      if (i !== -1 && idx <= i) return "small";
      if (j !== null && idx < j && idx > i) return "large";
      return "default";
    });

  steps.push({
    arr: [...arr], roles: makeRoles(arr, -1, 0), i: null, j: 0, pivotIdx: pi,
    description: `初始状态`, detail: `选 pivot = arr[${pi}] = ${pivot}（最后一个元素）；i = -1（小于区右边界，初始为"空"），j 从 0 扫描到 ${pi - 1}`,
    comparisons: 0, swaps: 0,
  });

  let i = -1;
  for (let j = 0; j < pi; j++) {
    cmp++;
    if (arr[j] <= pivot) {
      i++;
      if (i !== j) {
        steps.push({
          arr: [...arr], roles: makeRoles(arr, i - 1, j), i: i - 1, j,
          pivotIdx: pi,
          description: `j=${j}：arr[${j}]=${arr[j]} ≤ pivot=${pivot}，i++ → i=${i}，交换 arr[i]↔arr[j]`,
          detail: `arr[${i}]=${arr[i]} 与 arr[${j}]=${arr[j]} 交换，扩展左侧（小于等于区域）`,
          action: "swap", comparisons: cmp, swaps: sw,
        });
        [arr[i], arr[j]] = [arr[j], arr[i]]; sw++;
      } else {
        steps.push({
          arr: [...arr], roles: makeRoles(arr, i, j), i, j, pivotIdx: pi,
          description: `j=${j}：arr[${j}]=${arr[j]} ≤ pivot=${pivot}，i++ → i=${i}，i==j 无需交换`,
          detail: `当前元素就在边界上，直接合并到小于区，i=${i}`,
          comparisons: cmp, swaps: sw,
        });
      }
    } else {
      steps.push({
        arr: [...arr], roles: makeRoles(arr, i, j), i, j, pivotIdx: pi,
        description: `j=${j}：arr[${j}]=${arr[j]} > pivot=${pivot}，跳过（属于右侧）`,
        detail: `arr[${j}]=${arr[j]} 大于 pivot，保留在右侧区域，j 右移`,
        comparisons: cmp, swaps: sw,
      });
    }
  }

  // Place pivot
  const finalPos = i + 1;
  steps.push({
    arr: [...arr], roles: makeRoles(arr, i, pi), i, j: pi, pivotIdx: pi,
    description: `扫描结束，将 pivot 放到分界位置 i+1=${finalPos}`,
    detail: `交换 arr[${finalPos}]=${arr[finalPos]} 与 arr[${pi}]=pivot=${pivot}，pivot 归位`,
    action: "pivot_place", comparisons: cmp, swaps: sw,
  });
  [arr[finalPos], arr[pi]] = [arr[pi], arr[finalPos]]; sw++;
  steps.push({
    arr: [...arr], roles: arr.map((_, idx) => {
      if (idx === finalPos) return "done";
      if (idx < finalPos) return "small";
      return "large";
    }), i: finalPos, j: null, pivotIdx: finalPos,
    description: `✅ 划分完成！pivot=${pivot} 在下标 ${finalPos}`,
    detail: `左侧 [${arr.slice(0, finalPos).join(",")}] ≤ ${pivot} ≤ 右侧 [${arr.slice(finalPos + 1).join(",")}]；下一步递归对两侧各自排序`,
    action: "done", comparisons: cmp, swaps: sw,
  });
  return steps;
}

/* ─── Hoare trace ───────────────────────────────────────────────────────── */
// arr = [3, 6, 8, 2, 7, 4], pivot = arr[0] = 3
function buildHoareTrace(): PartitionStep[] {
  const base = [3, 6, 8, 2, 7, 4];
  const steps: PartitionStep[] = [];
  let arr = [...base], cmp = 0, sw = 0;
  const pivot = arr[0]; // Hoare uses first element

  const makeRoles = (a: number[], i: number, j: number, done = false): CellRole[] =>
    a.map((v, idx) => {
      if (done && idx <= j) return "small";
      if (done && idx > j) return "large";
      if (idx === 0 && i <= 0 && j >= 0) return "pivot"; // pivot starts at 0
      if (idx === i && idx === j) return "active_ij";
      if (idx === i) return "active_i";
      if (idx === j) return "active_j";
      return "default";
    });

  steps.push({
    arr: [...arr], roles: makeRoles(arr, -1, arr.length), i: null, j: null, pivotIdx: 0,
    description: `初始状态`, detail: `pivot = arr[0] = ${pivot}（第一个元素）。i 从左向右找 ≥ pivot 的元素，j 从右向左找 ≤ pivot 的元素，双向扫描`,
    comparisons: 0, swaps: 0,
  });

  let i = -1, j = arr.length;

  // Iteration 1
  i++; cmp++;
  while (arr[i] < pivot) { i++; cmp++; }
  j--; cmp++;
  while (arr[j] > pivot) { j--; cmp++; }

  steps.push({
    arr: [...arr], roles: makeRoles(arr, i, j), i, j, pivotIdx: 0,
    description: `i→${i}: arr[${i}]=${arr[i]}≥pivot=${pivot}；j←${j}: arr[${j}]=${arr[j]}≤pivot`,
    detail: `i=${i} < j=${j}，两指针未相遇，交换 arr[${i}]和arr[${j}]`,
    action: "swap", comparisons: cmp, swaps: sw,
  });
  [arr[i], arr[j]] = [arr[j], arr[i]]; sw++;

  steps.push({
    arr: [...arr], roles: makeRoles(arr, i, j), i, j, pivotIdx: 0,
    description: `交换后：arr=[${arr.join(",")}]`,
    detail: `arr[${i}]=${arr[i]} 和 arr[${j}]=${arr[j]} 交换完毕，继续扫描`,
    comparisons: cmp, swaps: sw,
  });

  // Iteration 2
  const prevI = i, prevJ = j;
  i++; cmp++;
  while (arr[i] < pivot) { i++; cmp++; }
  j--; cmp++;
  while (arr[j] > pivot) { j--; cmp++; }

  steps.push({
    arr: [...arr], roles: makeRoles(arr, i, j), i, j, pivotIdx: 0,
    description: `i→${i}: arr[${i}]=${arr[i]}≥pivot；j←${j}: arr[${j}]=${arr[j]}≤pivot`,
    detail: `i=${i} ≥ j=${j}，两指针相遇！划分结束，返回 j=${j}`,
    action: "done", comparisons: cmp, swaps: sw,
  });

  steps.push({
    arr: [...arr], roles: arr.map((_, idx) => {
      if (idx <= j) return "small";
      return "large";
    }), i, j, pivotIdx: 0,
    description: `✅ Hoare 划分完成！返回 j=${j}`,
    detail: `arr[0..${j}]=[${arr.slice(0, j + 1).join(",")}] 全 ≤ pivot；arr[${j + 1}..${arr.length - 1}]=[${arr.slice(j + 1).join(",")}] 全 ≥ pivot。注意：pivot 不一定在 j 位置，在左半某处`,
    comparisons: cmp, swaps: sw,
  });
  return steps;
}

/* ─── Cell rendering ────────────────────────────────────────────────────── */

const ROLE_STYLES: Record<CellRole, string> = {
  default:   "bg-slate-100 dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-300",
  pivot:     "bg-red-100 dark:bg-red-900/40 border-red-500 dark:border-red-400 text-red-700 dark:text-red-200 ring-2 ring-red-400 dark:ring-red-500",
  small:     "bg-emerald-100 dark:bg-emerald-900/30 border-emerald-400 dark:border-emerald-500 text-emerald-800 dark:text-emerald-200",
  large:     "bg-sky-100 dark:bg-sky-900/30 border-sky-400 dark:border-sky-500 text-sky-800 dark:text-sky-200",
  active_i:  "bg-indigo-200 dark:bg-indigo-900/60 border-indigo-500 dark:border-indigo-400 text-indigo-900 dark:text-indigo-100 ring-2 ring-indigo-500 scale-105",
  active_j:  "bg-amber-200 dark:bg-amber-900/60 border-amber-500 dark:border-amber-400 text-amber-900 dark:text-amber-100 ring-2 ring-amber-500 scale-105",
  active_ij: "bg-purple-200 dark:bg-purple-900/60 border-purple-500 dark:border-purple-400 text-purple-900 dark:text-purple-100 ring-2 ring-purple-500 scale-110",
  swapping:  "bg-orange-200 dark:bg-orange-900/60 border-orange-500 dark:border-orange-400 text-orange-900 dark:text-orange-100",
  done:      "bg-emerald-300 dark:bg-emerald-700/60 border-emerald-600 dark:border-emerald-400 text-emerald-900 dark:text-emerald-100 ring-2 ring-emerald-500 scale-105",
};

function ArrayCell({ value, role, label }: { value: number; role: CellRole; label?: string }) {
  return (
    <div className="flex flex-col items-center gap-1">
      {label && (
        <span className="text-[10px] font-bold px-1 rounded bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-400">{label}</span>
      )}
      <div className={`w-10 h-10 rounded-lg border-2 flex items-center justify-center font-bold text-sm font-mono transition-all duration-300 ${ROLE_STYLES[role]}`}>
        {value}
      </div>
    </div>
  );
}

/* ─── Main Component ─────────────────────────────────────────────────────── */

type Mode = "lomuto" | "hoare";

export default function QuicksortPartitionViz() {
  const [mode, setMode] = useState<Mode>("lomuto");
  const lomutoSteps = React.useMemo(buildLomutoTrace, []);
  const hoareSteps  = React.useMemo(buildHoareTrace, []);

  const trace = mode === "lomuto" ? lomutoSteps : hoareSteps;
  const [step, setStep] = useState(0);

  const handleMode = (m: Mode) => { setMode(m); setStep(0); };
  const cur = trace[step];
  const total = trace.length - 1;

  const Legend = () => (
    <div className="flex flex-wrap gap-2">
      {[
        { role: "pivot" as CellRole, label: "Pivot" },
        { role: "active_i" as CellRole, label: mode === "lomuto" ? "指针 i（小于区边界）" : "指针 i（左向右）" },
        { role: "active_j" as CellRole, label: mode === "lomuto" ? "指针 j（扫描）" : "指针 j（右向左）" },
        { role: "small" as CellRole, label: "≤ pivot（左侧）" },
        { role: "large" as CellRole, label: "> pivot（右侧）" },
        { role: "done" as CellRole, label: "Pivot 归位" },
      ].map(({ role, label }) => (
        <div key={label} className="flex items-center gap-1.5">
          <div className={`w-4 h-4 rounded border ${ROLE_STYLES[role].split(" ").slice(0, 2).join(" ")}`} />
          <span className="text-[11px] text-slate-500 dark:text-slate-400">{label}</span>
        </div>
      ))}
    </div>
  );

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 overflow-hidden shadow-sm">
      {/* Header */}
      <div className="px-5 py-4 border-b border-slate-200 dark:border-slate-700">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-bold text-slate-800 dark:text-slate-100 text-base">Quicksort PARTITION 双指针可视化</h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">示例数组 [3, 6, 8, 2, 7, 4]</p>
          </div>
          {/* Mode tabs */}
          <div className="flex gap-1 p-1 rounded-lg bg-slate-100 dark:bg-slate-800">
            {(["lomuto", "hoare"] as Mode[]).map(m => (
              <button
                key={m}
                onClick={() => handleMode(m)}
                className={`px-4 py-1.5 rounded-md text-xs font-semibold transition-all ${
                  mode === m
                    ? "bg-white dark:bg-slate-700 text-slate-800 dark:text-slate-100 shadow-sm"
                    : "text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
                }`}
              >
                {m === "lomuto" ? "Lomuto 方案" : "Hoare 方案"}
              </button>
            ))}
          </div>
        </div>

        {/* Scheme description */}
        <div className="mt-3 grid grid-cols-2 gap-3">
          <div className={`p-3 rounded-lg text-xs ${mode === "lomuto" ? "bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800" : "bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 opacity-60"}`}>
            <p className="font-semibold text-indigo-700 dark:text-indigo-300 mb-1">Lomuto 方案</p>
            <p className="text-slate-600 dark:text-slate-400">pivot = 最后一个元素；i 维护"小于区"边界，j 单向扫描。代码简洁，平均交换次数较多</p>
          </div>
          <div className={`p-3 rounded-lg text-xs ${mode === "hoare" ? "bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800" : "bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 opacity-60"}`}>
            <p className="font-semibold text-amber-700 dark:text-amber-300 mb-1">Hoare 方案</p>
            <p className="text-slate-600 dark:text-slate-400">pivot = 第一个元素；i、j 双向扫描相向而行。效率更高（交换次数约为 Lomuto 的 ⅓），但返回值≠pivot 位置</p>
          </div>
        </div>
      </div>

      <div className="p-5">
        {/* Stats */}
        <div className="grid grid-cols-3 gap-3 mb-5">
          {[
            { label: "比较次数", value: cur.comparisons, color: "text-indigo-600 dark:text-indigo-400" },
            { label: "交换次数", value: cur.swaps, color: "text-amber-600 dark:text-amber-400" },
            { label: "步骤", value: `${step + 1} / ${total + 1}`, color: "text-slate-600 dark:text-slate-400" },
          ].map(({ label, value, color }) => (
            <div key={label} className="rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 px-4 py-3 text-center">
              <p className={`text-xl font-bold font-mono ${color}`}>{value}</p>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{label}</p>
            </div>
          ))}
        </div>

        {/* Array visualization */}
        <div className="flex justify-center gap-2 py-4">
          {cur.arr.map((v, idx) => {
            const role = cur.roles[idx];
            const labels: string[] = [];
            if (idx === cur.pivotIdx && mode === "lomuto") labels.push("pivot");
            if (cur.i !== null && idx === cur.i) labels.push("i");
            if (cur.j !== null && idx === cur.j) labels.push("j");
            return (
              <div key={idx} className="flex flex-col items-center gap-0.5">
                <ArrayCell value={v} role={role} />
                <span className="text-[9px] font-mono text-slate-400 dark:text-slate-600">[{idx}]</span>
                {labels.length > 0 && (
                  <span className={`text-[10px] font-bold ${
                    labels.includes("pivot") ? "text-red-500 dark:text-red-400" :
                    labels.includes("i") && labels.includes("j") ? "text-purple-500 dark:text-purple-400" :
                    labels.includes("i") ? "text-indigo-500 dark:text-indigo-400" :
                    "text-amber-500 dark:text-amber-400"
                  }`}>
                    {labels.join("/")}
                  </span>
                )}
              </div>
            );
          })}
        </div>

        {/* Description */}
        <div className={`rounded-xl p-4 border ${
          cur.action === "done"
            ? "bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800"
            : cur.action === "swap" || cur.action === "pivot_place"
              ? "bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800"
              : "bg-slate-50 dark:bg-slate-800 border-slate-200 dark:border-slate-700"
        }`}>
          <p className="font-semibold text-sm text-slate-800 dark:text-slate-100">{cur.description}</p>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1.5 leading-relaxed">{cur.detail}</p>
        </div>

        {/* Legend */}
        <div className="mt-4">
          <Legend />
        </div>

        {/* Progress */}
        <div className="mt-4 h-1.5 rounded-full bg-slate-100 dark:bg-slate-800">
          <div
            className="h-full rounded-full bg-gradient-to-r from-indigo-500 via-amber-500 to-emerald-500 transition-all duration-300"
            style={{ width: `${(step / total) * 100}%` }}
          />
        </div>

        {/* Controls */}
        <div className="mt-4 flex items-center gap-2 justify-center">
          <button onClick={() => setStep(0)} className="px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors">
            重置
          </button>
          <button onClick={() => setStep(s => Math.max(0, s - 1))} disabled={step === 0} className="px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
            ← 上一步
          </button>
          <button onClick={() => setStep(s => Math.min(total, s + 1))} disabled={step >= total} className="px-4 py-1.5 text-xs rounded-lg bg-indigo-500 hover:bg-indigo-600 text-white font-medium disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
            下一步 →
          </button>
          <button onClick={() => setStep(total)} disabled={step >= total} className="px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">
            跳至结束
          </button>
        </div>

        {/* Key difference callout */}
        {step === total && (
          <div className="mt-4 p-3 rounded-xl bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 text-xs">
            <p className="font-semibold text-purple-700 dark:text-purple-300">
              {mode === "lomuto"
                ? "⚡ Lomuto 特点：返回值 = pivot 的最终位置，递归对 [l..p-1] 和 [p+1..r]"
                : "⚡ Hoare 特点：返回值 j ≠ pivot 位置！递归对 [l..j] 和 [j+1..r]（含 j）"}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

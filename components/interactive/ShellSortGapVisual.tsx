"use client";
import React, { useState, useCallback } from "react";

// ─── Gap sequences ────────────────────────────────────────────────────────────
type GapSeqName = "Knuth" | "Hibbard" | "Shell";

function knuthGaps(n: number): number[] {
  const gaps: number[] = [];
  let h = 1;
  while (h < Math.floor(n / 3)) h = 3 * h + 1;
  while (h >= 1) { gaps.push(h); h = Math.floor((h - 1) / 3); }
  return gaps;
}
function hibbardGaps(n: number): number[] {
  const gaps: number[] = [];
  let k = 1;
  while ((1 << k) - 1 < n) k++;
  for (let i = k - 1; i >= 1; i--) {
    const g = (1 << i) - 1;
    if (g < n) gaps.push(g);
  }
  return gaps.length ? gaps : [1];
}
function shellGaps(n: number): number[] {
  const gaps: number[] = [];
  let g = Math.floor(n / 2);
  while (g >= 1) { gaps.push(g); g = Math.floor(g / 2); }
  return gaps;
}

const GAP_BUILDERS: Record<GapSeqName, (n: number) => number[]> = {
  Knuth: knuthGaps,
  Hibbard: hibbardGaps,
  Shell: shellGaps,
};
const GAP_LABELS: Record<GapSeqName, string> = {
  Knuth: "Knuth（1, 4, 13, 40…）",
  Hibbard: "Hibbard（1, 3, 7, 15…）",
  Shell: "Shell（n/2, n/4, …）",
};
const GAP_FORMULA: Record<GapSeqName, string> = {
  Knuth: "$h = 3h+1$，从 $h=1$ 迭代",
  Hibbard: "$h_k = 2^k - 1$，时间复杂度 $O(n^{3/2})$",
  Shell: "每次 $h = \\lfloor h/2 \\rfloor$，最坏 $O(n^2)$",
};

// ─── Subgroup colors ──────────────────────────────────────────────────────────
const GROUP_COLORS = [
  "#6366f1", "#f59e0b", "#ef4444", "#10b981", "#3b82f6",
  "#ec4899", "#14b8a6", "#f97316", "#8b5cf6", "#84cc16",
];

// ─── Trace insertion sort steps within one h-pass ────────────────────────────
interface PassStep {
  arr: number[];
  comparing: [number, number] | null;
  shifting: [number, number] | null;
  placing: number | null;
  phase: string;
}

function buildPassSteps(initArr: number[], h: number): PassStep[] {
  const arr = [...initArr];
  const n = arr.length;
  const steps: PassStep[] = [{ arr: [...arr], comparing: null, shifting: null, placing: null, phase: `开始 h=${h} 的插入排序轮次` }];

  for (let i = h; i < n; i++) {
    const key = arr[i];
    let j = i - h;
    steps.push({ arr: [...arr], comparing: null, shifting: null, placing: i, phase: `取出 arr[${i}]=${key} 作为 key` });
    while (j >= 0 && arr[j] > key) {
      steps.push({ arr: [...arr], comparing: [j, i], shifting: null, placing: null, phase: `比较 arr[${j}]=${arr[j]} > key=${key}` });
      arr[j + h] = arr[j];
      steps.push({ arr: [...arr], comparing: null, shifting: [j, j + h], placing: null, phase: `右移 arr[${j}] → arr[${j + h}]` });
      j -= h;
    }
    arr[j + h] = key;
    steps.push({ arr: [...arr], comparing: null, shifting: null, placing: j + h, phase: `插入 key=${key} 到位置 ${j + h}` });
  }
  steps.push({ arr: [...arr], comparing: null, shifting: null, placing: null, phase: `h=${h} 轮次结束` });
  return steps;
}

// ─── INITIAL array ────────────────────────────────────────────────────────────
const ARRAY_PRESETS: Record<string, number[]> = {
  随机20: [68, 15, 43, 7, 92, 31, 85, 24, 56, 3, 77, 48, 12, 64, 29, 91, 38, 5, 72, 19],
  随机12: [42, 17, 73, 8, 55, 29, 66, 11, 88, 34, 47, 6],
  逆序: [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
};

// ─── Main Component ───────────────────────────────────────────────────────────
export default function ShellSortGapVisual() {
  const [arrKey, setArrKey] = useState("随机12");
  const [baseArr, setBaseArr] = useState<number[]>(ARRAY_PRESETS["随机12"]);
  const [seqName, setSeqName] = useState<GapSeqName>("Knuth");
  const [passIdx, setPassIdx] = useState(0); // which h in gaps[]
  const [subStep, setSubStep] = useState<number | null>(null); // null = overview mode
  const [showSubSteps, setShowSubSteps] = useState(false);

  // derive sorted state after each pass
  const gaps = GAP_BUILDERS[seqName](baseArr.length);

  // Compute arr after `passIdx` previous passes
  const arrAfterPasses = useCallback((upTo: number): number[] => {
    const arr = [...baseArr];
    for (let pi = 0; pi < upTo; pi++) {
      const h = gaps[pi];
      for (let i = h; i < arr.length; i++) {
        const key = arr[i]; let j = i - h;
        while (j >= 0 && arr[j] > key) { arr[j + h] = arr[j]; j -= h; }
        arr[j + h] = key;
      }
    }
    return arr;
  }, [baseArr, gaps]);

  const currentPassArr = arrAfterPasses(passIdx);
  const h = gaps[passIdx] ?? 1;

  const passSteps = buildPassSteps(currentPassArr, h);
  const curSubStep = subStep !== null ? passSteps[Math.min(subStep, passSteps.length - 1)] : null;

  const displayArr = curSubStep ? curSubStep.arr : currentPassArr;
  const n = displayArr.length;

  // Subgroup assignment: element at index i belongs to group i % h
  const groupOf = (i: number) => i % h;
  const maxVal = Math.max(...baseArr);

  // ─── SVG layout ──────────────────────────────────────────────────────────────
  const SVG_W = 560, BAR_H = 100, ARC_H = 44;
  const barW = Math.max(14, Math.floor((SVG_W - 24) / n));
  const barGap = 2;
  const totalW = (barW + barGap) * n + 20;
  const barX = (i: number) => 10 + i * (barW + barGap) + barW / 2;

  const selectPreset = (key: string) => {
    setArrKey(key);
    setBaseArr(ARRAY_PRESETS[key]);
    setPassIdx(0);
    setSubStep(null);
  };

  const handleSeq = (s: GapSeqName) => {
    setSeqName(s);
    setPassIdx(0);
    setSubStep(null);
  };

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5 my-6 shadow-sm">
      {/* Header */}
      <h3 className="text-base font-bold text-slate-800 dark:text-slate-100 mb-1">
        🐚 希尔排序间隔访问模式
      </h3>
      <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">
        同色柱 = 同一个 <em>h-子序列</em>（每隔 $h$ 个元素一组）；每轮对各子序列做插入排序，再缩小 $h$ 直到 $h=1$。
      </p>

      {/* Controls row */}
      <div className="flex flex-wrap gap-3 mb-4">
        <div>
          <div className="text-xs text-slate-500 dark:text-slate-400 mb-1 font-medium">间隔序列</div>
          <div className="flex flex-wrap gap-1.5">
            {(["Knuth", "Hibbard", "Shell"] as GapSeqName[]).map(s => (
              <button key={s} onClick={() => handleSeq(s)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                  seqName === s
                    ? "bg-indigo-600 text-white shadow"
                    : "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-indigo-50 dark:hover:bg-slate-700"
                }`}>
                {s}
              </button>
            ))}
          </div>
        </div>
        <div>
          <div className="text-xs text-slate-500 dark:text-slate-400 mb-1 font-medium">数组</div>
          <div className="flex flex-wrap gap-1.5">
            {Object.keys(ARRAY_PRESETS).map(k => (
              <button key={k} onClick={() => selectPreset(k)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                  arrKey === k
                    ? "bg-amber-500 text-white shadow"
                    : "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-amber-50 dark:hover:bg-slate-700"
                }`}>
                {k}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Gap sequence display */}
      <div className="rounded-lg bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800 px-4 py-3 mb-4">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs font-semibold text-indigo-700 dark:text-indigo-300">{GAP_LABELS[seqName]}</span>
          <span className="text-xs text-indigo-500 dark:text-indigo-400">·</span>
          <span className="text-xs text-indigo-600 dark:text-indigo-400">{GAP_FORMULA[seqName]}</span>
        </div>
        <div className="flex flex-wrap gap-1.5 mt-2">
          {gaps.map((g, pi) => (
            <button key={pi} onClick={() => { setPassIdx(pi); setSubStep(null); }}
              className={`px-2.5 py-1 rounded-lg text-xs font-mono font-bold transition-colors ${
                pi === passIdx
                  ? "bg-indigo-600 text-white ring-2 ring-indigo-300 dark:ring-indigo-700"
                  : pi < passIdx
                  ? "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400"
                  : "bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400"
              }`}>
              h={g}
            </button>
          ))}
        </div>
      </div>

      {/* Subgroup color legend */}
      <div className="flex flex-wrap gap-2 mb-3 items-center">
        <span className="text-xs text-slate-500 dark:text-slate-400">子序列（h={h}）：</span>
        {Array.from({ length: Math.min(h, 8) }, (_, g) => (
          <div key={g} className="flex items-center gap-1">
            <span className="w-3 h-3 rounded-sm inline-block" style={{ backgroundColor: GROUP_COLORS[g % GROUP_COLORS.length] }} />
            <span className="text-xs font-mono text-slate-600 dark:text-slate-400">组{g}</span>
          </div>
        ))}
        {h > 8 && <span className="text-xs text-slate-400 dark:text-slate-500">…共{h}组</span>}
      </div>

      {/* SVG */}
      <div className="bg-slate-50 dark:bg-slate-800/60 rounded-xl overflow-x-auto mb-4">
        <svg width={Math.max(totalW, SVG_W)} height={BAR_H + ARC_H + 10} className="block" viewBox={`0 0 ${Math.max(totalW, SVG_W)} ${BAR_H + ARC_H + 10}`}>
          {/* Group arcs beneath bars */}
          {Array.from({ length: n }, (_, i) => {
            const g = groupOf(i);
            const color = GROUP_COLORS[g % GROUP_COLORS.length];
            // Draw arc from i to i+h if both exist
            if (i + h < n) {
              const x1 = barX(i), x2 = barX(i + h);
              const cy = BAR_H + 6 + (g % 3) * 10;
              const cx = (x1 + x2) / 2;
              return (
                <path key={`arc-${i}`}
                  d={`M ${x1} ${BAR_H + 4} Q ${cx} ${cy + 14} ${x2} ${BAR_H + 4}`}
                  fill="none" stroke={color} strokeWidth={1.5} strokeOpacity={0.55} />
              );
            }
            return null;
          })}

          {/* Bars */}
          {displayArr.map((v, i) => {
            const g = groupOf(i);
            const color = GROUP_COLORS[g % GROUP_COLORS.length];
            const x = 10 + i * (barW + barGap);
            const bh = Math.max(4, (v / maxVal) * (BAR_H - 18));
            const isComparing = curSubStep?.comparing && (curSubStep.comparing[0] === i || curSubStep.comparing[1] === i);
            const isShifting = curSubStep?.shifting && (curSubStep.shifting[0] === i || curSubStep.shifting[1] === i);
            const isPlacing = curSubStep?.placing === i;
            const barColor = isComparing ? "#ef4444" : isShifting ? "#f59e0b" : isPlacing ? "#fbbf24" : color;
            return (
              <g key={i}>
                <rect x={x} y={BAR_H - bh} width={barW} height={bh}
                  fill={barColor} rx={2}
                  fillOpacity={curSubStep && !isComparing && !isShifting && !isPlacing ? 0.5 : 0.85} />
                <text x={x + barW / 2} y={BAR_H - bh - 2} textAnchor="middle"
                  fontSize={Math.min(10, barW - 2)} fontWeight="bold" fill={barColor}>
                  {v}
                </text>
                <text x={x + barW / 2} y={BAR_H + 3} textAnchor="middle"
                  fontSize={8} fill="#94a3b8">
                  {i}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Sub-step area */}
      <div className="mb-3">
        <div className="flex items-center gap-2 mb-2">
          <button onClick={() => { setShowSubSteps(v => !v); setSubStep(0); }}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              showSubSteps
                ? "bg-amber-500 text-white"
                : "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-amber-50 dark:hover:bg-slate-700"
            }`}>
            {showSubSteps ? "▼ 隐藏分步" : "▶ 展开 h=" + h + " 的逐步插入排序"}
          </button>
        </div>

        {showSubSteps && (
          <div className="rounded-lg border border-slate-200 dark:border-slate-700 p-3">
            <div className={`text-xs rounded-lg px-3 py-2 mb-3 ${
              curSubStep?.placing !== null && curSubStep?.phase.includes("结束")
                ? "bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-200 dark:border-emerald-800 text-emerald-700 dark:text-emerald-300"
                : "bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 text-amber-700 dark:text-amber-300"
            }`}>
              <span className="font-semibold">步 {(subStep ?? 0) + 1}/{passSteps.length}：</span>
              {curSubStep?.phase ?? passSteps[0].phase}
            </div>
            <div className="flex items-center gap-2">
              <button onClick={() => setSubStep(0)} disabled={(subStep ?? 0) === 0}
                className="text-xs px-2 py-1 rounded bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 disabled:opacity-30">⏮</button>
              <button onClick={() => setSubStep(s => Math.max(0, (s ?? 0) - 1))} disabled={(subStep ?? 0) === 0}
                className="text-xs px-2 py-1 rounded bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 disabled:opacity-30">◀</button>
              <div className="flex-1 bg-slate-200 dark:bg-slate-700 rounded-full h-1.5 overflow-hidden cursor-pointer"
                onClick={e => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  const ratio = (e.clientX - rect.left) / rect.width;
                  setSubStep(Math.round(ratio * (passSteps.length - 1)));
                }}>
                <div className="h-full bg-amber-500 rounded-full transition-all" style={{ width: `${((subStep ?? 0) / (passSteps.length - 1)) * 100}%` }} />
              </div>
              <button onClick={() => setSubStep(s => Math.min(passSteps.length - 1, (s ?? 0) + 1))} disabled={(subStep ?? 0) >= passSteps.length - 1}
                className="text-xs px-2 py-1 rounded bg-amber-500 text-white disabled:opacity-30">▶</button>
              <button onClick={() => setSubStep(passSteps.length - 1)} disabled={(subStep ?? 0) >= passSteps.length - 1}
                className="text-xs px-2 py-1 rounded bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 disabled:opacity-30">⏭</button>
            </div>
          </div>
        )}
      </div>

      {/* Pass navigation */}
      <div className="flex items-center justify-between gap-3">
        <button onClick={() => { setPassIdx(p => Math.max(0, p - 1)); setSubStep(null); }} disabled={passIdx === 0}
          className="px-4 py-2 rounded-lg text-xs font-medium bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300
            disabled:opacity-30 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">
          ◀ 上一轮 (h={gaps[passIdx - 1] ?? "—"})
        </button>
        <div className="text-xs text-slate-500 dark:text-slate-400 text-center">
          第 {passIdx + 1}/{gaps.length} 轮 · 当前 h = <span className="font-bold font-mono text-indigo-600 dark:text-indigo-400">{h}</span>
        </div>
        <button onClick={() => { setPassIdx(p => Math.min(gaps.length - 1, p + 1)); setSubStep(null); }} disabled={passIdx >= gaps.length - 1}
          className="px-4 py-2 rounded-lg text-xs font-medium bg-indigo-600 text-white
            disabled:opacity-30 hover:bg-indigo-700 transition-colors">
          下一轮 (h={gaps[passIdx + 1] ?? "—"}) ▶
        </button>
      </div>

      <div className="mt-3 text-xs text-slate-400 dark:text-slate-500">
        💡 Knuth 序列：时间复杂度 $O(n^{3/2})$，实践中接近 $O(n \log n)$；Shell 原始序列最坏 $O(n^2)$
      </div>
    </div>
  );
}

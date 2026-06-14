"use client";
import React, { useState, useCallback } from "react";

// ─── Types ─────────────────────────────────────────────────────────────────────
interface Step {
  arr: number[];
  sortedLen: number; // elements arr[0..sortedLen-1] are "placed"
  keyIdx: number | null;
  comparing: number | null; // index being compared with key
  phase: "pick" | "shift" | "place" | "done";
  description: string;
  comparisons: number;
  shifts: number;
}

// ─── Presets ───────────────────────────────────────────────────────────────────
const PRESETS: Record<string, number[]> = {
  随机: [38, 27, 43, 3, 9, 82, 10, 55, 18, 64],
  几乎有序: [2, 4, 6, 8, 3, 10, 12, 14, 1, 16],
  逆序: [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
  重复元素: [5, 3, 5, 1, 3, 2, 5, 4, 1, 2],
};

// ─── Build full trace ──────────────────────────────────────────────────────────
function buildTrace(init: number[]): Step[] {
  const steps: Step[] = [];
  const arr = [...init];
  const n = arr.length;
  let cmp = 0, shift = 0;

  steps.push({ arr: [...arr], sortedLen: 1, keyIdx: null, comparing: null, phase: "done", description: "初始数组（第 0 个元素自然有序）", comparisons: 0, shifts: 0 });

  for (let i = 1; i < n; i++) {
    const key = arr[i];
    steps.push({ arr: [...arr], sortedLen: i, keyIdx: i, comparing: null, phase: "pick", description: `第 ${i} 轮：取出 key = ${key}`, comparisons: cmp, shifts: shift });

    let j = i - 1;
    while (j >= 0 && arr[j] > key) {
      cmp++;
      steps.push({ arr: [...arr], sortedLen: i, keyIdx: i, comparing: j, phase: "shift", description: `arr[${j}]=${arr[j]} > key=${key}，右移 arr[${j}] → arr[${j + 1}]`, comparisons: cmp, shifts: shift });
      arr[j + 1] = arr[j];
      shift++;
      steps.push({ arr: [...arr], sortedLen: i, keyIdx: j + 1, comparing: null, phase: "shift", description: `右移完成，空位在 ${j + 1}`, comparisons: cmp, shifts: shift });
      j--;
    }
    if (j >= 0) cmp++; // final failed comparison
    arr[j + 1] = key;
    steps.push({ arr: [...arr], sortedLen: i + 1, keyIdx: j + 1, comparing: null, phase: "place", description: `将 key=${key} 放入位置 ${j + 1}`, comparisons: cmp, shifts: shift });
  }
  steps.push({ arr: [...arr], sortedLen: n, keyIdx: null, comparing: null, phase: "done", description: "🎉 排序完成！", comparisons: cmp, shifts: shift });
  return steps;
}

// ─── Bar chart ─────────────────────────────────────────────────────────────────
function Bar({ value, maxVal, role }: { value: number; maxVal: number; role: "sorted" | "key" | "comparing" | "unsorted" }) {
  const pct = (value / maxVal) * 100;
  const colors: Record<string, string> = {
    sorted: "#10b981",
    key: "#f59e0b",
    comparing: "#ef4444",
    unsorted: "#6366f1",
  };
  const labelColors: Record<string, string> = {
    sorted: "text-emerald-600 dark:text-emerald-400",
    key: "text-amber-600 dark:text-amber-400",
    comparing: "text-red-600 dark:text-red-400",
    unsorted: "text-indigo-600 dark:text-indigo-400",
  };
  return (
    <div className="flex flex-col items-center gap-1 flex-1">
      <span className={`text-xs font-mono font-bold ${labelColors[role]}`}>{value}</span>
      <div className="w-full flex items-end" style={{ height: 80 }}>
        <div
          className="w-full rounded-t-md transition-all duration-200"
          style={{ height: `${pct}%`, backgroundColor: colors[role], minHeight: 3 }}
        />
      </div>
    </div>
  );
}

// ─── Legend ────────────────────────────────────────────────────────────────────
function Legend() {
  const items = [
    { bg: "#10b981", label: "已排序" },
    { bg: "#f59e0b", label: "当前 key" },
    { bg: "#ef4444", label: "正在比较" },
    { bg: "#6366f1", label: "未排序" },
  ];
  return (
    <div className="flex flex-wrap gap-3">
      {items.map(({ bg, label }) => (
        <div key={label} className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-sm inline-block" style={{ backgroundColor: bg }} />
          <span className="text-xs text-slate-500 dark:text-slate-400">{label}</span>
        </div>
      ))}
    </div>
  );
}

// ─── Main Component ────────────────────────────────────────────────────────────
export default function InsertionSortStep() {
  const [presetKey, setPresetKey] = useState<string>("随机");
  const [trace, setTrace] = useState<Step[]>(() => buildTrace(PRESETS["随机"]));
  const [step, setStep] = useState(0);

  const selectPreset = (key: string) => {
    setPresetKey(key);
    setTrace(buildTrace(PRESETS[key]));
    setStep(0);
  };

  const cur = trace[step];
  const maxVal = Math.max(...cur.arr);
  const total = trace.length - 1;

  const roleFor = useCallback((idx: number): "sorted" | "key" | "comparing" | "unsorted" => {
    if (idx === cur.keyIdx) return "key";
    if (idx === cur.comparing) return "comparing";
    if (idx < cur.sortedLen && idx !== cur.keyIdx) return "sorted";
    return "unsorted";
  }, [cur]);

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5 my-6 shadow-sm">
      {/* Header */}
      <h3 className="text-base font-bold text-slate-800 dark:text-slate-100 mb-1">
        🃏 插入排序逐步可视化
      </h3>
      <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">
        模拟"摸牌"过程：每轮取出一张牌（<span className="text-amber-600 dark:text-amber-400 font-semibold">key</span>），与左侧已排好的牌逐一比较，右移腾出空位后插入。
      </p>

      {/* Preset selector */}
      <div className="flex flex-wrap gap-2 mb-5">
        {Object.keys(PRESETS).map(k => (
          <button key={k} onClick={() => selectPreset(k)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              presetKey === k
                ? "bg-indigo-600 text-white shadow"
                : "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-indigo-50 dark:hover:bg-slate-700"
            }`}>
            {k}
          </button>
        ))}
      </div>

      {/* Main visualization */}
      <div className="bg-slate-50 dark:bg-slate-800/60 rounded-xl p-4 mb-4">
        {/* Divider label */}
        <div className="flex items-end gap-0.5 mb-1">
          {cur.arr.map((v, i) => (
            <div key={i} className="flex-1" />
          ))}
        </div>

        {/* Bars */}
        <div className="flex items-end gap-1">
          {cur.arr.map((v, i) => (
            <React.Fragment key={i}>
              {/* Divider between sorted and unsorted (before sortedLen index) */}
              {i === cur.sortedLen && cur.phase !== "done" && (
                <div className="w-0.5 self-stretch flex flex-col justify-between">
                  <div className="h-full border-l-2 border-dashed border-slate-400 dark:border-slate-500" />
                </div>
              )}
              <Bar value={v} maxVal={maxVal} role={roleFor(i)} />
            </React.Fragment>
          ))}
        </div>

        {/* Index row */}
        <div className="flex gap-1 mt-1">
          {cur.arr.map((_, i) => (
            <div key={i} className={`flex-1 text-center text-xs font-mono ${
              i === cur.keyIdx ? "text-amber-500 dark:text-amber-400 font-bold" :
              i === cur.comparing ? "text-red-500 dark:text-red-400" :
              "text-slate-400 dark:text-slate-600"
            }`}>
              {i}
            </div>
          ))}
        </div>
      </div>

      {/* Phase info */}
      <div className={`rounded-lg px-4 py-3 mb-4 border text-sm transition-colors ${
        cur.phase === "done"
          ? "bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800 text-emerald-700 dark:text-emerald-300"
          : cur.phase === "pick"
          ? "bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800 text-amber-700 dark:text-amber-300"
          : "bg-indigo-50 dark:bg-indigo-900/20 border-indigo-200 dark:border-indigo-800 text-indigo-700 dark:text-indigo-300"
      }`}>
        <span className="font-semibold">步骤 {step}/{total}：</span>{cur.description}
      </div>

      {/* Stats row */}
      <div className="flex flex-wrap gap-4 mb-4">
        {[
          { label: "已排好", value: `${cur.sortedLen} / ${cur.arr.length}` },
          { label: "比较次数", value: cur.comparisons },
          { label: "右移次数", value: cur.shifts },
        ].map(({ label, value }) => (
          <div key={label} className="flex-1 min-w-24 rounded-lg bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 p-3 text-center">
            <div className="text-xl font-bold font-mono text-slate-800 dark:text-slate-100">{value}</div>
            <div className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">{label}</div>
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="mb-4"><Legend /></div>

      {/* Controls */}
      <div className="flex items-center gap-2">
        <button onClick={() => setStep(0)} disabled={step === 0}
          className="px-3 py-1.5 rounded-lg text-xs bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400
            disabled:opacity-30 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">⏮</button>
        <button onClick={() => setStep(s => Math.max(0, s - 1))} disabled={step === 0}
          className="px-3 py-1.5 rounded-lg text-xs bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300
            disabled:opacity-30 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">◀ 上一步</button>

        {/* Progress bar */}
        <div className="flex-1 bg-slate-200 dark:bg-slate-700 rounded-full h-1.5 mx-1 overflow-hidden cursor-pointer"
          onClick={e => {
            const rect = e.currentTarget.getBoundingClientRect();
            const ratio = (e.clientX - rect.left) / rect.width;
            setStep(Math.round(ratio * total));
          }}>
          <div className="h-full bg-indigo-500 rounded-full transition-all" style={{ width: `${(step / total) * 100}%` }} />
        </div>

        <button onClick={() => setStep(s => Math.min(total, s + 1))} disabled={step === total}
          className="px-3 py-1.5 rounded-lg text-xs bg-indigo-600 text-white disabled:opacity-30
            hover:bg-indigo-700 transition-colors">下一步 ▶</button>
        <button onClick={() => setStep(total)} disabled={step === total}
          className="px-3 py-1.5 rounded-lg text-xs bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400
            disabled:opacity-30 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">⏭</button>
      </div>

      {/* Tips */}
      <div className="mt-3 text-xs text-slate-400 dark:text-slate-500">
        💡 "几乎有序"数组让插入排序接近 $O(n)$，"逆序"数组则退化至 $O(n^2)$，右移次数最多
      </div>
    </div>
  );
}

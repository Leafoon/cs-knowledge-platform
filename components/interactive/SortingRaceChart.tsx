"use client";
import React, { useState, useCallback, useRef, useEffect } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────
type AlgoName = "Insertion" | "Selection" | "Bubble" | "Shell";

interface SortState {
  arr: number[];
  comparisons: number;
  swaps: number;
  done: boolean;
  // Iterator state for step-by-step
  i: number;
  j: number;
  phase: string; // descriptive label
  highlights: number[]; // indices currently being compared/swapped
}

// ─── Algo configs ─────────────────────────────────────────────────────────────
const ALGO_COLORS: Record<AlgoName, { bar: string; accent: string; label: string }> = {
  Insertion: { bar: "#6366f1", accent: "#818cf8", label: "插入排序" },
  Selection: { bar: "#f59e0b", accent: "#fbbf24", label: "选择排序" },
  Bubble:    { bar: "#ef4444", accent: "#f87171", label: "冒泡排序" },
  Shell:     { bar: "#10b981", accent: "#34d399", label: "希尔排序" },
};

const ALGO_NAMES: AlgoName[] = ["Insertion", "Selection", "Bubble", "Shell"];

// ─── Full sort trace builders ─────────────────────────────────────────────────
// Each returns an array of snapshots (one per "step")
type Snapshot = { arr: number[]; comparisons: number; swaps: number; highlights: number[]; phase: string };

function traceInsertion(init: number[]): Snapshot[] {
  const arr = [...init];
  const snaps: Snapshot[] = [{ arr: [...arr], comparisons: 0, swaps: 0, highlights: [], phase: "初始" }];
  let cmp = 0, sw = 0;
  const n = arr.length;
  for (let i = 1; i < n; i++) {
    const key = arr[i];
    let j = i - 1;
    while (j >= 0 && arr[j] > key) {
      cmp++;
      arr[j + 1] = arr[j];
      sw++;
      snaps.push({ arr: [...arr], comparisons: cmp, swaps: sw, highlights: [j, j + 1], phase: `插入 arr[${i}]=${key}` });
      j--;
    }
    if (j >= 0) cmp++; // the failed comparison
    arr[j + 1] = key;
    snaps.push({ arr: [...arr], comparisons: cmp, swaps: sw, highlights: [j + 1], phase: `放置 ${key} 到位置 ${j + 1}` });
  }
  return snaps;
}

function traceSelection(init: number[]): Snapshot[] {
  const arr = [...init];
  const snaps: Snapshot[] = [{ arr: [...arr], comparisons: 0, swaps: 0, highlights: [], phase: "初始" }];
  let cmp = 0, sw = 0;
  const n = arr.length;
  for (let i = 0; i < n - 1; i++) {
    let minIdx = i;
    for (let j = i + 1; j < n; j++) {
      cmp++;
      snaps.push({ arr: [...arr], comparisons: cmp, swaps: sw, highlights: [minIdx, j], phase: `第 ${i + 1} 轮：寻找最小值` });
      if (arr[j] < arr[minIdx]) minIdx = j;
    }
    if (minIdx !== i) {
      [arr[i], arr[minIdx]] = [arr[minIdx], arr[i]];
      sw++;
    }
    snaps.push({ arr: [...arr], comparisons: cmp, swaps: sw, highlights: [i, minIdx], phase: `交换: ${arr[i]} ↔ ${arr[minIdx]}` });
  }
  return snaps;
}

function traceBubble(init: number[]): Snapshot[] {
  const arr = [...init];
  const snaps: Snapshot[] = [{ arr: [...arr], comparisons: 0, swaps: 0, highlights: [], phase: "初始" }];
  let cmp = 0, sw = 0;
  const n = arr.length;
  for (let i = 0; i < n - 1; i++) {
    let swapped = false;
    for (let j = 0; j < n - 1 - i; j++) {
      cmp++;
      snaps.push({ arr: [...arr], comparisons: cmp, swaps: sw, highlights: [j, j + 1], phase: `第 ${i + 1} 趟` });
      if (arr[j] > arr[j + 1]) {
        [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
        sw++;
        swapped = true;
      }
    }
    if (!swapped) break;
  }
  return snaps;
}

function traceShell(init: number[]): Snapshot[] {
  const arr = [...init];
  const snaps: Snapshot[] = [{ arr: [...arr], comparisons: 0, swaps: 0, highlights: [], phase: "初始" }];
  let cmp = 0, sw = 0;
  const n = arr.length;
  let h = 1;
  while (h < Math.floor(n / 3)) h = 3 * h + 1;
  while (h >= 1) {
    for (let i = h; i < n; i++) {
      const key = arr[i];
      let j = i - h;
      while (j >= 0 && arr[j] > key) {
        cmp++;
        arr[j + h] = arr[j];
        sw++;
        snaps.push({ arr: [...arr], comparisons: cmp, swaps: sw, highlights: [j, j + h], phase: `h=${h}：移动` });
        j -= h;
      }
      if (j >= 0) cmp++;
      arr[j + h] = key;
      snaps.push({ arr: [...arr], comparisons: cmp, swaps: sw, highlights: [j + h], phase: `h=${h}：插入 ${key}` });
    }
    h = Math.floor((h - 1) / 3);
  }
  return snaps;
}

// ─── Generate a random array ──────────────────────────────────────────────────
function randomArray(n: number, max: number): number[] {
  return Array.from({ length: n }, () => Math.floor(Math.random() * max) + 1);
}

const N = 16;
const MAX_VAL = 60;

// ─── Bar visualization ────────────────────────────────────────────────────────
function BarChart({ snap, colors, sorted }: {
  snap: Snapshot; colors: { bar: string; accent: string }; sorted: boolean;
}) {
  const maxVal = MAX_VAL;
  const barW = Math.floor(100 / snap.arr.length);
  return (
    <div className="flex items-end gap-0.5 h-28 px-1">
      {snap.arr.map((v, i) => {
        const isHL = snap.highlights.includes(i);
        const pct = (v / maxVal) * 100;
        return (
          <div key={i} className="flex-1 flex flex-col justify-end items-center gap-0" style={{ height: "100%" }}>
            <div
              style={{
                height: `${pct}%`,
                backgroundColor: sorted ? "#10b981" : isHL ? colors.accent : colors.bar,
                transition: "height 0.1s ease, background-color 0.1s",
                minHeight: 2,
                borderRadius: "2px 2px 0 0",
              }}
              className="w-full"
            />
          </div>
        );
      })}
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────
export default function SortingRaceChart() {
  const [initArr] = useState<number[]>(() => randomArray(N, MAX_VAL));
  const [traces] = useState<Record<AlgoName, Snapshot[]>>(() => ({
    Insertion: traceInsertion([...initArr]),
    Selection: traceSelection([...initArr]),
    Bubble:    traceBubble([...initArr]),
    Shell:     traceShell([...initArr]),
  }));

  const maxSteps = Math.max(...ALGO_NAMES.map(a => traces[a].length - 1));
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [speed, setSpeed] = useState(120); // ms per step

  const currentSnap = useCallback((algo: AlgoName) => {
    const t = traces[algo];
    const idx = Math.min(step, t.length - 1);
    return t[idx];
  }, [step, traces]);

  const isSorted = (algo: AlgoName) => step >= traces[algo].length - 1;

  // Auto-play
  useEffect(() => {
    if (!playing) { if (timerRef.current) clearInterval(timerRef.current); return; }
    timerRef.current = setInterval(() => {
      setStep(s => {
        if (s >= maxSteps) { setPlaying(false); return s; }
        return s + 1;
      });
    }, speed);
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [playing, speed, maxSteps]);

  const handleReset = () => { setStep(0); setPlaying(false); };
  const progress = maxSteps > 0 ? (step / maxSteps) * 100 : 0;

  const finalStats = useCallback((algo: AlgoName) => {
    const t = traces[algo];
    return t[t.length - 1];
  }, [traces]);

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5 my-6 shadow-sm">
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-2 mb-1">
        <h3 className="text-base font-bold text-slate-800 dark:text-slate-100">
          🏁 四种排序算法同步赛跑
        </h3>
        <div className="flex items-center gap-2">
          <label className="text-xs text-slate-500 dark:text-slate-400">速度</label>
          <select value={speed} onChange={e => setSpeed(Number(e.target.value))}
            className="text-xs rounded-lg bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700
              text-slate-700 dark:text-slate-300 px-2 py-1 outline-none">
            <option value={300}>慢</option>
            <option value={120}>中</option>
            <option value={40}>快</option>
            <option value={10}>极速</option>
          </select>
        </div>
      </div>
      <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">
        对同一个随机数组同步模拟四种 $O(n^2)$ 排序，观察它们柱状图的变化速度、比较次数、交换次数差异。
        高亮柱 = 当前正在比较/移动的元素。
      </p>

      {/* 4 charts grid */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        {ALGO_NAMES.map(algo => {
          const snap = currentSnap(algo);
          const cfg = ALGO_COLORS[algo];
          const done = isSorted(algo);
          const final = finalStats(algo);
          return (
            <div key={algo} className={`rounded-xl border p-3 transition-colors ${
              done
                ? "border-emerald-200 dark:border-emerald-800 bg-emerald-50/50 dark:bg-emerald-900/10"
                : "border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/40"
            }`}>
              {/* Algo label */}
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: cfg.bar }} />
                  <span className={`text-xs font-bold ${done ? "text-emerald-600 dark:text-emerald-400" : "text-slate-700 dark:text-slate-200"}`}>
                    {cfg.label}
                  </span>
                  {done && <span className="text-emerald-500 text-xs">✓</span>}
                </div>
                <span className="text-xs font-mono text-slate-400 dark:text-slate-500">
                  {Math.min(step, traces[algo].length - 1)}/{traces[algo].length - 1}步
                </span>
              </div>

              {/* Bar chart */}
              <BarChart snap={snap} colors={cfg} sorted={done} />

              {/* Stats */}
              <div className="flex gap-3 mt-2 text-xs">
                <span className="text-slate-500 dark:text-slate-400">
                  比较 <span className="font-mono font-bold text-slate-700 dark:text-slate-200">{snap.comparisons}</span>
                  <span className="text-slate-400 dark:text-slate-600 ml-1">/ {final.comparisons}</span>
                </span>
                <span className="text-slate-500 dark:text-slate-400">
                  移动 <span className="font-mono font-bold text-slate-700 dark:text-slate-200">{snap.swaps}</span>
                  <span className="text-slate-400 dark:text-slate-600 ml-1">/ {final.swaps}</span>
                </span>
              </div>
              {snap.phase && (
                <div className="mt-1 text-xs text-slate-400 dark:text-slate-500 font-mono truncate">{snap.phase}</div>
              )}
            </div>
          );
        })}
      </div>

      {/* Progress + Controls */}
      <div className="flex items-center gap-2 mb-3">
        <button onClick={handleReset}
          className="px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400
            text-xs hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors shrink-0">
          ↺
        </button>
        <div className="flex-1 bg-slate-200 dark:bg-slate-700 rounded-full h-1.5 overflow-hidden cursor-pointer"
          onClick={e => {
            const rect = e.currentTarget.getBoundingClientRect();
            const ratio = (e.clientX - rect.left) / rect.width;
            setStep(Math.round(ratio * maxSteps));
          }}>
          <div className="h-full bg-indigo-500 rounded-full transition-all duration-100" style={{ width: `${progress}%` }} />
        </div>
        <button onClick={() => setStep(s => Math.max(0, s - 1))} disabled={step <= 0}
          className="px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300
            text-xs disabled:opacity-30 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">◀</button>
        <button onClick={() => setPlaying(p => !p)} disabled={step >= maxSteps}
          className="px-4 py-1.5 rounded-lg bg-indigo-600 text-white text-xs disabled:opacity-30
            hover:bg-indigo-700 transition-colors font-medium min-w-16 text-center">
          {playing ? "⏸ 暂停" : "▶ 播放"}
        </button>
        <button onClick={() => setStep(s => Math.min(maxSteps, s + 1))} disabled={step >= maxSteps}
          className="px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300
            text-xs disabled:opacity-30 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">▶</button>
      </div>

      {/* Final comparison table */}
      <div className="rounded-lg border border-slate-200 dark:border-slate-700 overflow-hidden">
        <div className="bg-slate-50 dark:bg-slate-800/60 px-3 py-1.5 text-xs font-semibold text-slate-600 dark:text-slate-300 border-b border-slate-200 dark:border-slate-700">
          最终统计（n = {N}）
        </div>
        <div className="divide-y divide-slate-100 dark:divide-slate-800">
          {ALGO_NAMES.map(algo => {
            const f = finalStats(algo);
            const cfg = ALGO_COLORS[algo];
            const maxCmp = Math.max(...ALGO_NAMES.map(a => finalStats(a).comparisons));
            return (
              <div key={algo} className="flex items-center gap-2 px-3 py-2">
                <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: cfg.bar }} />
                <span className="text-xs text-slate-700 dark:text-slate-200 w-24 shrink-0">{cfg.label}</span>
                <div className="flex-1">
                  <div className="flex items-center gap-1">
                    <div className="flex-1 bg-slate-100 dark:bg-slate-800 rounded-full h-1.5 overflow-hidden">
                      <div className="h-full rounded-full" style={{ backgroundColor: cfg.bar, width: `${(f.comparisons / maxCmp) * 100}%` }} />
                    </div>
                    <span className="text-xs font-mono text-slate-600 dark:text-slate-400 w-10 text-right">{f.comparisons}</span>
                  </div>
                </div>
                <span className="text-xs text-slate-400 dark:text-slate-500 w-20 text-right shrink-0">移动 {f.swaps} 次</span>
              </div>
            );
          })}
        </div>
      </div>

      <div className="mt-3 text-xs text-slate-400 dark:text-slate-500">
        💡 点击进度条可快速跳转；绿色高亮 = 排序已完成
      </div>
    </div>
  );
}

"use client";
import React, { useState, useCallback, useRef, useEffect } from "react";

// ─── Helpers ──────────────────────────────────────────────────────────────────
function countInversions(arr: number[]): [number, number][] {
  const pairs: [number, number][] = [];
  for (let i = 0; i < arr.length; i++)
    for (let j = i + 1; j < arr.length; j++)
      if (arr[i] > arr[j]) pairs.push([i, j]);
  return pairs;
}

// Bubble sort step list: returns list of swapped index pairs
function bubbleSteps(init: number[]): Array<{ arr: number[]; swapped: [number, number] | null }> {
  const arr = [...init];
  const steps: Array<{ arr: number[]; swapped: [number, number] | null }> = [{ arr: [...arr], swapped: null }];
  const n = arr.length;
  for (let i = 0; i < n - 1; i++) {
    let anySwap = false;
    for (let j = 0; j < n - 1 - i; j++) {
      if (arr[j] > arr[j + 1]) {
        steps.push({ arr: [...arr], swapped: [j, j + 1] });
        [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
        steps.push({ arr: [...arr], swapped: null });
        anySwap = true;
      }
    }
    if (!anySwap) break;
  }
  return steps;
}

// ─── Palette for arc colors (by distance) ─────────────────────────────────────
function arcColor(dist: number, max: number): string {
  if (max === 0) return "#ef4444";
  const t = dist / max;
  const r = Math.round(239 + t * (251 - 239));
  const g = Math.round(68 + t * (113 - 68));
  const b = Math.round(68 + t * 12);
  return `rgb(${r},${g},${b})`;
}

// ─── Presets ───────────────────────────────────────────────────────────────────
const PRESETS: Record<string, number[]> = {
  随机: [7, 3, 8, 1, 5, 2, 9, 4, 6],
  "逆序（最多逆序对）": [9, 8, 7, 6, 5, 4, 3, 2, 1],
  "几乎有序（少量逆序对）": [1, 2, 3, 5, 4, 6, 7, 9, 8],
  "有序（无逆序对）": [1, 2, 3, 4, 5, 6, 7, 8, 9],
};

// ─── Main Component ────────────────────────────────────────────────────────────
export default function InversionCounter() {
  const [presetKey, setPresetKey] = useState("随机");
  const [arr, setArr] = useState<number[]>(PRESETS["随机"]);
  const [mode, setMode] = useState<"view" | "sort">("view");
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [hoveredPair, setHoveredPair] = useState<[number, number] | null>(null);

  const bubbleTrace = useCallback(() => bubbleSteps(arr), [arr]);
  const trace = bubbleTrace();
  const curArr = mode === "sort" ? trace[Math.min(step, trace.length - 1)].arr : arr;
  const curSwapped = mode === "sort" ? trace[Math.min(step, trace.length - 1)].swapped : null;

  const inversions = countInversions(curArr);
  const initialInversions = countInversions(arr);
  const maxDist = arr.length - 1;

  // Is a pair still an inversion in current array?
  const isInversionInCurrent = useCallback(([i, j]: [number, number]): boolean => {
    return curArr[i] > curArr[j];
  }, [curArr]);

  const visiblePairs = mode === "view"
    ? initialInversions
    : initialInversions.filter(p => isInversionInCurrent([p[0], p[1]]));

  // Auto-play
  useEffect(() => {
    if (!playing) { if (timerRef.current) clearInterval(timerRef.current); return; }
    timerRef.current = setInterval(() => {
      setStep(s => {
        if (s >= trace.length - 1) { setPlaying(false); return s; }
        return s + 1;
      });
    }, 350);
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [playing, trace.length]);

  const handlePreset = (key: string) => {
    setPresetKey(key);
    setArr(PRESETS[key]);
    setStep(0);
    setMode("view");
    setPlaying(false);
  };

  const handleModeSort = () => {
    setMode("sort");
    setStep(0);
    setPlaying(false);
  };

  const n = curArr.length;
  const W = 480, BAR_H = 100, ARC_H = 90;
  const barW = Math.floor((W - 20) / n);
  const barGap = 2;
  const effectiveW = (barW + barGap) * n;

  const barX = (i: number) => 10 + i * (barW + barGap) + barW / 2;
  const maxVal = Math.max(...arr);

  return (
    <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-5 my-6 shadow-sm">
      {/* Header */}
      <h3 className="text-base font-bold text-slate-800 dark:text-slate-100 mb-1">
        🔄 逆序对动态统计
      </h3>
      <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">
        逆序对：$i &lt; j$ 但 $A[i] &gt; A[j]$。弧线连接每一个逆序对，冒泡排序的交换次数恰好等于逆序对总数。
      </p>

      {/* Preset selector */}
      <div className="flex flex-wrap gap-2 mb-4">
        {Object.keys(PRESETS).map(k => (
          <button key={k} onClick={() => handlePreset(k)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              presetKey === k
                ? "bg-indigo-600 text-white shadow"
                : "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-indigo-50 dark:hover:bg-slate-700"
            }`}>
            {k}
          </button>
        ))}
      </div>

      {/* Big inversion count */}
      <div className="flex items-center gap-4 mb-4">
        <div className="rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700 px-6 py-4 text-center min-w-36">
          <div className="text-4xl font-black font-mono text-red-500 dark:text-red-400 transition-all">
            {inversions.length}
          </div>
          <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">当前逆序对数</div>
        </div>
        <div className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
          <p><span className="font-semibold text-slate-700 dark:text-slate-300">初始逆序对：</span>{initialInversions.length}</p>
          <p><span className="font-semibold text-slate-700 dark:text-slate-300">已消除：</span>{initialInversions.length - inversions.length}</p>
          <p className="mt-1">逆序对数 = 冒泡排序最少交换次数</p>
          <p>对于逆序数组，最大逆序对数 = $n(n-1)/2$ = {Math.floor(n * (n - 1) / 2)}</p>
        </div>
      </div>

      {/* SVG visualization */}
      <div className="bg-slate-50 dark:bg-slate-800/60 rounded-xl overflow-x-auto mb-4">
        <svg width={W} height={BAR_H + ARC_H + 10} className="block mx-auto" viewBox={`0 0 ${W} ${BAR_H + ARC_H + 10}`}>
          {/* Arcs — below bar chart */}
          {visiblePairs.map(([i, j]) => {
            const x1 = barX(i), x2 = barX(j);
            const cx = (x1 + x2) / 2;
            const dist = j - i;
            const arcDepth = (dist / maxDist) * (ARC_H * 0.85) + 8;
            const baseY = BAR_H + 5;
            const color = arcColor(dist, maxDist);
            const isHovered = hoveredPair && hoveredPair[0] === i && hoveredPair[1] === j;
            const isSwapping = curSwapped && ((curSwapped[0] === i && curSwapped[1] === j) || (curSwapped[0] === j && curSwapped[1] === i));
            return (
              <g key={`${i}-${j}`}
                onMouseEnter={() => setHoveredPair([i, j])}
                onMouseLeave={() => setHoveredPair(null)}
                style={{ cursor: "pointer" }}>
                <path
                  d={`M ${x1} ${baseY} Q ${cx} ${baseY + arcDepth} ${x2} ${baseY}`}
                  fill="none"
                  stroke={isSwapping ? "#fbbf24" : color}
                  strokeWidth={isHovered || isSwapping ? 2.5 : 1.2}
                  strokeOpacity={isHovered ? 1 : 0.55}
                />
                {isHovered && (
                  <text x={cx} y={baseY + arcDepth + 10} textAnchor="middle" fontSize={9} fill={color} opacity={0.9}>
                    ({curArr[i]},{curArr[j]})
                  </text>
                )}
              </g>
            );
          })}

          {/* Bars */}
          {curArr.map((v, i) => {
            const x = 10 + i * (barW + barGap);
            const bh = Math.max(3, (v / maxVal) * (BAR_H - 20));
            const isLeft = inversions.some(([a, b]) => a === i);
            const isRight = inversions.some(([a, b]) => b === i);
            const isSwap = curSwapped && (curSwapped[0] === i || curSwapped[1] === i);
            const fill = isSwap ? "#f59e0b" : (isLeft || isRight) ? "#ef4444" : "#10b981";
            return (
              <g key={i}>
                <rect x={x} y={BAR_H - bh} width={barW} height={bh}
                  fill={fill} rx={2}
                  fillOpacity={isSwap ? 1 : (isLeft || isRight) ? 0.85 : 0.7} />
                <text x={x + barW / 2} y={BAR_H - bh - 3} textAnchor="middle"
                  fontSize={10} fontWeight="bold"
                  fill={isSwap ? "#f59e0b" : (isLeft || isRight) ? "#ef4444" : "#10b981"}>
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

      {/* Mode switch + controls */}
      <div className="flex flex-wrap gap-2 mb-3">
        <button onClick={() => { setMode("view"); setStep(0); setPlaying(false); }}
          className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
            mode === "view"
              ? "bg-indigo-600 text-white shadow"
              : "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-indigo-50 dark:hover:bg-slate-700"
          }`}>
          🔍 显示所有逆序对
        </button>
        <button onClick={handleModeSort}
          className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
            mode === "sort"
              ? "bg-red-600 text-white shadow"
              : "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 hover:bg-red-50 dark:hover:bg-slate-700"
          }`}>
          🫧 运行冒泡排序
        </button>
      </div>

      {mode === "sort" && (
        <div className="flex items-center gap-2">
          <button onClick={() => setStep(0)}
            className="px-2 py-1.5 rounded-lg text-xs bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400
              hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">⏮</button>
          <button onClick={() => setStep(s => Math.max(0, s - 1))} disabled={step === 0}
            className="px-2 py-1.5 rounded-lg text-xs bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300
              disabled:opacity-30 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">◀</button>
          <div className="flex-1 bg-slate-200 dark:bg-slate-700 rounded-full h-1.5 overflow-hidden cursor-pointer"
            onClick={e => {
              const rect = e.currentTarget.getBoundingClientRect();
              const ratio = (e.clientX - rect.left) / rect.width;
              setStep(Math.round(ratio * (trace.length - 1)));
            }}>
            <div className="h-full bg-red-500 rounded-full transition-all" style={{ width: `${(step / (trace.length - 1)) * 100}%` }} />
          </div>
          <button onClick={() => setPlaying(p => !p)} disabled={step >= trace.length - 1}
            className="px-3 py-1.5 rounded-lg text-xs bg-red-600 text-white disabled:opacity-30 hover:bg-red-700 transition-colors">
            {playing ? "⏸" : "▶"}
          </button>
          <button onClick={() => setStep(s => Math.min(trace.length - 1, s + 1))} disabled={step >= trace.length - 1}
            className="px-2 py-1.5 rounded-lg text-xs bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300
              disabled:opacity-30 hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">▶</button>
          <button onClick={() => setStep(trace.length - 1)}
            className="px-2 py-1.5 rounded-lg text-xs bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400
              hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors">⏭</button>
        </div>
      )}

      <div className="flex flex-wrap gap-3 mt-3 text-xs text-slate-500 dark:text-slate-400">
        <span><span className="inline-block w-2 h-2 rounded-full bg-red-400 mr-1" />参与逆序对的元素</span>
        <span><span className="inline-block w-2 h-2 rounded-full bg-emerald-400 mr-1" />不在任何逆序对中</span>
        <span><span className="inline-block w-2 h-2 rounded-full bg-amber-400 mr-1" />正在交换</span>
        <span className="ml-auto">💡 将鼠标移到弧线上查看具体逆序对</span>
      </div>
    </div>
  );
}

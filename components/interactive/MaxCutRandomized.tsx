"use client";

import { useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Shuffle, RotateCcw, Zap } from "lucide-react";

// Fixed graph: 8 nodes
const NODE_POS: [number, number][] = [
  [200, 80],  // 0
  [340, 80],  // 1
  [440, 190], // 2
  [380, 310], // 3
  [200, 340], // 4
  [100, 240], // 5
  [270, 190], // 6  (center-ish)
  [80, 120],  // 7
];

const EDGES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0],   // outer hexagon (skipping 6,7)
  [0, 6], [1, 6], [2, 6], [3, 6], [4, 6], [5, 6],   // spokes to center
  [7, 0], [7, 5], [7, 5],                             // extra edges
];

// Dedupe edges
const UNIQUE_EDGES = Array.from(
  new Map(EDGES.map((e) => [`${Math.min(...e)}-${Math.max(...e)}`, e as [number, number]])).values()
);

function randomColoring(n: number): number[] {
  return Array.from({ length: n }, () => Math.random() < 0.5 ? 0 : 1);
}

function cutSize(coloring: number[], edges: [number, number][]) {
  return edges.filter(([u, v]) => coloring[u] !== coloring[v]).length;
}

const NUM_NODES = NODE_POS.length;

export function MaxCutRandomized() {
  const [coloring, setColoring] = useState<number[]>(() => Array(NUM_NODES).fill(-1));
  const [trials, setTrials] = useState(0);
  const [bestCut, setBestCut] = useState(0);
  const [bestColoring, setBestColoring] = useState<number[]>(Array(NUM_NODES).fill(-1));
  const [cutHistory, setCutHistory] = useState<number[]>([]);
  const [running, setRunning] = useState(false);
  const [flash, setFlash] = useState(false);
  const runRef = useRef(false);

  const doOneTrial = useCallback((
    prevBest: { cut: number; coloring: number[]; count: number }
  ): { cut: number; coloring: number[]; count: number } => {
    const c = randomColoring(NUM_NODES);
    const cut = cutSize(c, UNIQUE_EDGES);
    if (cut > prevBest.cut) {
      return { cut, coloring: c, count: prevBest.count + 1 };
    }
    return { ...prevBest, count: prevBest.count + 1 };
  }, []);

  const runTrials = useCallback(async (n: number) => {
    if (running) return;
    setRunning(true);
    runRef.current = true;

    let state = { cut: bestCut, coloring: bestColoring, count: trials };
    const histBatch: number[] = [];

    for (let i = 0; i < n; i++) {
      if (!runRef.current) break;
      state = doOneTrial(state);
      histBatch.push(state.cut);
      if (i % 5 === 0 || i === n - 1) {
        const snap = { ...state };
        setColoring(snap.coloring);
        setBestCut(snap.cut);
        setBestColoring(snap.coloring);
        setTrials(snap.count);
        setCutHistory((h) => [...h, ...histBatch.splice(0)]);
        setFlash(true);
        setTimeout(() => setFlash(false), 200);
        await new Promise((r) => setTimeout(r, 30));
      }
    }

    runRef.current = false;
    setRunning(false);
  }, [bestCut, bestColoring, trials, running, doOneTrial]);

  const doSingle = useCallback(() => {
    const c = randomColoring(NUM_NODES);
    const cut = cutSize(c, UNIQUE_EDGES);
    setColoring(c);
    setTrials((t) => t + 1);
    if (cut > bestCut) {
      setBestCut(cut);
      setBestColoring(c);
      setFlash(true);
      setTimeout(() => setFlash(false), 300);
    }
    setCutHistory((h) => [...h.slice(-49), cut]);
  }, [bestCut]);

  const reset = useCallback(() => {
    runRef.current = false;
    setRunning(false);
    setColoring(Array(NUM_NODES).fill(-1));
    setTrials(0);
    setBestCut(0);
    setBestColoring(Array(NUM_NODES).fill(-1));
    setCutHistory([]);
  }, []);

  const displayColoring = trials === 0 ? Array(NUM_NODES).fill(-1) : coloring;
  const currentCut = trials === 0 ? 0 : cutSize(displayColoring, UNIQUE_EDGES);
  const expectedCut = UNIQUE_EDGES.length / 2;

  const nodeColor = (i: number) => {
    const c = displayColoring[i];
    if (c === -1) return { fill: "#94a3b8", stroke: "#64748b" };
    if (c === 0) return { fill: "#f97316", stroke: "#c2410c" }; // orange = S
    return { fill: "#3b82f6", stroke: "#1d4ed8" }; // blue = T
  };

  // Mini bar chart for history
  const histMax = UNIQUE_EDGES.length;
  const recentHist = cutHistory.slice(-40);

  return (
    <div className="w-full max-w-4xl mx-auto my-6 rounded-2xl overflow-hidden border border-amber-200 dark:border-amber-800 shadow-xl bg-white dark:bg-slate-900">
      {/* Header */}
      <div className="bg-gradient-to-r from-orange-500 via-amber-500 to-yellow-500 px-6 py-4">
        <h3 className="text-base font-bold text-white">Max-Cut 随机化算法 — 1/2 近似演示</h3>
        <p className="text-xs text-amber-100 mt-0.5">
          每次随机 2-染色期望割边数 ≥ |E|/2，点击"随机着色"或"连续实验"观察效果
        </p>
      </div>

      <div className="p-5 grid grid-cols-1 lg:grid-cols-[1fr_auto] gap-5">
        {/* Graph SVG */}
        <div className="relative">
          <div className="flex items-center gap-2 text-[11px] mb-2">
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-orange-400 inline-block border border-orange-600" /> S 集合</span>
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-blue-500 inline-block border border-blue-700" /> T 集合</span>
            <span className="flex items-center gap-1"><span className="w-8 h-0.5 bg-yellow-400 inline-block" /> 割边</span>
            <span className="flex items-center gap-1"><span className="w-8 h-0.5 bg-slate-300 dark:bg-slate-600 inline-block" /> 非割边</span>
          </div>
          <svg viewBox="0 0 540 400" className="w-full max-w-md mx-auto">
            {/* Edges */}
            {UNIQUE_EDGES.map(([u, v], i) => {
              const isCut = trials > 0 && displayColoring[u] !== displayColoring[v];
              const [x1, y1] = NODE_POS[u];
              const [x2, y2] = NODE_POS[v];
              return (
                <motion.line
                  key={i}
                  x1={x1} y1={y1} x2={x2} y2={y2}
                  stroke={isCut ? "#fbbf24" : "#cbd5e1"}
                  strokeWidth={isCut ? 3 : 1.5}
                  strokeOpacity={isCut ? 1 : 0.5}
                  className="dark:[&]:stroke-slate-600"
                  animate={{ stroke: isCut ? "#fbbf24" : "#94a3b8", strokeWidth: isCut ? 3 : 1.5 }}
                  transition={{ duration: 0.3 }}
                />
              );
            })}

            {/* Nodes */}
            {NODE_POS.map(([cx, cy], i) => {
              const { fill, stroke } = nodeColor(i);
              return (
                <g key={i}>
                  <motion.circle
                    cx={cx} cy={cy} r={22}
                    fill={fill} stroke={stroke} strokeWidth={2.5}
                    animate={{ fill, r: 22 }}
                    transition={{ duration: 0.25 }}
                  />
                  <text x={cx} y={cy} textAnchor="middle" dominantBaseline="middle"
                    fill="white" fontSize={12} fontWeight="bold">
                    v{i}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* Stats sidebar */}
        <div className="flex flex-col gap-3 min-w-[220px]">
          {/* Controls */}
          <div className="flex flex-col gap-2">
            <button
              onClick={doSingle}
              disabled={running}
              className="flex items-center justify-center gap-2 bg-amber-500 hover:bg-amber-600 disabled:opacity-50 text-white text-sm font-semibold px-4 py-2.5 rounded-xl transition-colors"
            >
              <Shuffle size={15} /> 随机着色一次
            </button>
            <button
              onClick={() => runTrials(50)}
              disabled={running}
              className="flex items-center justify-center gap-2 bg-orange-500 hover:bg-orange-600 disabled:opacity-50 text-white text-sm font-semibold px-4 py-2.5 rounded-xl transition-colors"
            >
              <Zap size={15} /> 连续实验 50 次
            </button>
            <button
              onClick={reset}
              className="flex items-center justify-center gap-2 bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-200 text-sm font-semibold px-4 py-2.5 rounded-xl transition-colors"
            >
              <RotateCcw size={15} /> 重置
            </button>
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-2 gap-2">
            {[
              { label: "当前割边", value: trials > 0 ? currentCut : "—", color: "text-amber-600 dark:text-amber-400" },
              { label: "最优割边", value: trials > 0 ? bestCut : "—", color: "text-emerald-600 dark:text-emerald-400" },
              { label: "期望 |E|/2", value: expectedCut.toFixed(1), color: "text-blue-600 dark:text-blue-400" },
              { label: "实验次数", value: trials, color: "text-violet-600 dark:text-violet-400" },
            ].map(({ label, value, color }) => (
              <div key={label} className="bg-slate-50 dark:bg-slate-800 rounded-xl p-2.5 text-center">
                <div className="text-[10px] text-slate-500 dark:text-slate-400">{label}</div>
                <motion.div
                  key={String(value)}
                  initial={{ scale: 0.9 }}
                  animate={{ scale: 1 }}
                  className={`text-xl font-black font-mono ${color}`}
                >
                  {value}
                </motion.div>
              </div>
            ))}
          </div>

          {/* Best ratio */}
          {trials > 0 && (
            <div className={`rounded-xl p-3 text-center transition-colors ${flash ? "bg-amber-100 dark:bg-amber-900/40" : "bg-slate-50 dark:bg-slate-800"}`}>
              <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">最优割边比</div>
              <div className={`text-2xl font-black font-mono ${bestCut / UNIQUE_EDGES.length >= 0.5 ? "text-emerald-600 dark:text-emerald-400" : "text-amber-600 dark:text-amber-400"}`}>
                {(bestCut / UNIQUE_EDGES.length * 100).toFixed(0)}%
              </div>
              <div className="text-[10px] text-slate-400 mt-0.5">
                {bestCut}/{UNIQUE_EDGES.length} 条边被割 ≥ 50%？{bestCut >= expectedCut ? "✓" : "…"}
              </div>
            </div>
          )}

          {/* Mini history chart */}
          {recentHist.length > 1 && (
            <div className="bg-slate-50 dark:bg-slate-800 rounded-xl p-3">
              <div className="text-[10px] text-slate-500 dark:text-slate-400 mb-2">历次割边数</div>
              <div className="flex items-end gap-0.5 h-12">
                {recentHist.map((c, i) => (
                  <div
                    key={i}
                    className="flex-1 rounded-t-sm bg-amber-400 dark:bg-amber-500 opacity-80 transition-all"
                    style={{ height: `${(c / histMax) * 100}%` }}
                  />
                ))}
              </div>
              <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
                <span>旧</span>
                <span>最新</span>
              </div>
            </div>
          )}

          {/* Theory note */}
          <div className="bg-amber-50 dark:bg-amber-900/20 rounded-xl p-3 border border-amber-100 dark:border-amber-800">
            <div className="text-[10px] font-mono text-amber-700 dark:text-amber-300 space-y-0.5">
              <div>E[cut] = Σ Pr[e被割] = |E|/2</div>
              <div>每条边被割 ⟺ 端点异色</div>
              <div>随机 ⟹ Pr = 1/2 per edge</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

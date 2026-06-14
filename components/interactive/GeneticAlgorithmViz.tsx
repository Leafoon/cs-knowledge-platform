"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, RotateCcw, SkipForward } from "lucide-react";

// 8-city TSP (fixed coordinates)
const CITIES: [number, number][] = [
  [30, 60], [130, 20], [220, 50], [260, 140],
  [210, 230], [110, 260], [20, 180], [130, 140],
];
const C = CITIES.length;

function dist(a: number, b: number): number {
  const [x1, y1] = CITIES[a], [x2, y2] = CITIES[b];
  return Math.hypot(x2 - x1, y2 - y1);
}

function tourCost(t: number[]): number {
  let s = 0;
  for (let i = 0; i < t.length; i++) s += dist(t[i], t[(i + 1) % t.length]);
  return s;
}

function fitness(t: number[]): number { return 1000 / tourCost(t); }

function randPerm(): number[] {
  const a = Array.from({ length: C }, (_, i) => i);
  for (let i = C - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// OX Crossover (Order Crossover)
function oxCrossover(p1: number[], p2: number[]): number[] {
  const a = Math.floor(Math.random() * C);
  const b = Math.floor(Math.random() * C);
  const [lo, hi] = [Math.min(a, b), Math.max(a, b)];
  const child = new Array(C).fill(-1);
  for (let i = lo; i <= hi; i++) child[i] = p1[i];
  let ci = (hi + 1) % C;
  for (let i = 0; i < C; i++) {
    const gene = p2[(hi + 1 + i) % C];
    if (!child.includes(gene)) {
      child[ci] = gene;
      ci = (ci + 1) % C;
    }
  }
  return child;
}

function mutate(t: number[]): number[] {
  const a = Math.floor(Math.random() * C);
  let b = Math.floor(Math.random() * (C - 1));
  if (b >= a) b++;
  const r = [...t];
  [r[a], r[b]] = [r[b], r[a]];
  return r;
}

interface Individual { tour: number[]; fit: number; cost: number; }
interface GenRecord { gen: number; population: Individual[]; bestFit: number; avgFit: number; worstFit: number; }

const POP_SIZE = 8;
const MAX_GEN = 30;

function runGA(): GenRecord[] {
  let pop: Individual[] = Array.from({ length: POP_SIZE }, () => {
    const t = randPerm();
    return { tour: t, fit: fitness(t), cost: tourCost(t) };
  });
  pop.sort((a, b) => b.fit - a.fit);

  const records: GenRecord[] = [];

  const makeRecord = (g: number, p: Individual[]): GenRecord => ({
    gen: g,
    population: [...p],
    bestFit: p[0].fit,
    avgFit: p.reduce((s, x) => s + x.fit, 0) / p.length,
    worstFit: p[p.length - 1].fit,
  });

  records.push(makeRecord(0, pop));

  for (let g = 1; g <= MAX_GEN; g++) {
    // Elitism: top 2 survive
    const survivors = pop.slice(0, 2);
    // Tournament selection + crossover
    const children: Individual[] = [...survivors];
    while (children.length < POP_SIZE) {
      const p1 = pop[Math.floor(Math.random() * 4)];
      const p2 = pop[Math.floor(Math.random() * 4)];
      let child = oxCrossover(p1.tour, p2.tour);
      if (Math.random() < 0.3) child = mutate(child);
      children.push({ tour: child, fit: fitness(child), cost: tourCost(child) });
    }
    children.sort((a, b) => b.fit - a.fit);
    pop = children.slice(0, POP_SIZE);
    records.push(makeRecord(g, pop));
  }
  return records;
}

const GENS = runGA();
const FITNESS_MAX = Math.max(...GENS.map((g) => g.bestFit));
const FITNESS_MIN = Math.min(...GENS.map((g) => g.worstFit));

const RANK_COLORS = [
  "from-emerald-500 to-teal-400",
  "from-teal-500 to-cyan-400",
  "from-blue-500 to-indigo-400",
  "from-indigo-500 to-violet-400",
  "from-violet-500 to-purple-400",
  "from-purple-500 to-rose-400",
  "from-rose-500 to-red-400",
  "from-red-500 to-orange-400",
];

export function GeneticAlgorithmViz() {
  const [genIdx, setGenIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [highlightRank, setHighlightRank] = useState<number | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const step = useCallback(() => {
    setGenIdx((i) => {
      if (i >= GENS.length - 1) { setPlaying(false); return i; }
      return i + 1;
    });
  }, []);

  useEffect(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (!playing) return;
    timerRef.current = setInterval(step, 400);
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [playing, step]);

  const reset = useCallback(() => { setPlaying(false); setGenIdx(0); }, []);

  const record = GENS[genIdx];
  const prevRecord = genIdx > 0 ? GENS[genIdx - 1] : null;

  // City map colors
  const allCosts = GENS.map((g) => g.population[0].cost);
  const bestEver = Math.min(...allCosts);
  const currentBest = record.population[0];

  return (
    <div className="w-full max-w-4xl mx-auto my-6 rounded-2xl overflow-hidden border border-violet-200 dark:border-violet-800 shadow-xl bg-white dark:bg-slate-900">
      {/* Header */}
      <div className="bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600 px-6 py-4">
        <h3 className="text-base font-bold text-white">遗传算法 (GA) — TSP 种群进化演示</h3>
        <p className="text-xs text-violet-100 mt-0.5">
          选择 × 交叉 × 变异驱动种群向更优路径进化，第 {genIdx}/{MAX_GEN} 代
        </p>
      </div>

      <div className="p-5 flex flex-col gap-4">
        {/* Top: Population bars */}
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto] gap-5">
          {/* Population display */}
          <div>
            <div className="text-xs text-slate-500 dark:text-slate-400 mb-2 flex items-center justify-between">
              <span>种群（第 {genIdx} 代，{POP_SIZE} 个体，按适应度↓排列）</span>
              <span className="text-[10px] text-emerald-600 dark:text-emerald-400">
                最优路径长度 = {currentBest.cost.toFixed(1)}
              </span>
            </div>
            <div className="space-y-1.5">
              {record.population.map((ind, rank) => {
                const fitNorm = (ind.fit - FITNESS_MIN) / (FITNESS_MAX - FITNESS_MIN + 0.001);
                const prevFit = prevRecord?.population[rank]?.fit;
                const improved = prevFit !== undefined && ind.fit > prevFit;
                return (
                  <motion.div
                    key={rank}
                    layout
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    onClick={() => setHighlightRank(rank === highlightRank ? null : rank)}
                    className={`flex items-center gap-3 p-2 rounded-xl cursor-pointer transition-colors ${
                      highlightRank === rank
                        ? "bg-violet-50 dark:bg-violet-900/30 ring-1 ring-violet-300 dark:ring-violet-700"
                        : "hover:bg-slate-50 dark:hover:bg-slate-800"
                    }`}
                  >
                    {/* Rank badge */}
                    <div className={`w-6 h-6 rounded-full text-[10px] font-black text-white flex items-center justify-center bg-gradient-to-br ${RANK_COLORS[rank]} flex-shrink-0`}>
                      {rank + 1}
                    </div>

                    {/* Fitness bar */}
                    <div className="flex-1 flex flex-col gap-0.5">
                      <div className="flex items-center justify-between text-[10px]">
                        <span className="font-mono text-slate-600 dark:text-slate-300">
                          {ind.tour.map((c) => `C${c}`).join("→")}
                        </span>
                        {improved && (
                          <span className="text-emerald-500 text-[9px] font-bold">↑改善</span>
                        )}
                      </div>
                      <div className="h-4 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
                        <motion.div
                          className={`h-full rounded-full bg-gradient-to-r ${RANK_COLORS[rank]}`}
                          animate={{ width: `${fitNorm * 100}%` }}
                          transition={{ duration: 0.4 }}
                        />
                      </div>
                    </div>

                    {/* Fitness value */}
                    <div className="text-right min-w-[60px]">
                      <div className="text-xs font-black font-mono text-violet-600 dark:text-violet-400">
                        {ind.fit.toFixed(2)}
                      </div>
                      <div className="text-[9px] text-slate-400">{ind.cost.toFixed(0)}km</div>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </div>

          {/* Right: city map of best + controls */}
          <div className="flex flex-col gap-3 min-w-[200px]">
            <div className="bg-slate-50 dark:bg-slate-800 rounded-xl p-2">
              <div className="text-[10px] text-slate-400 mb-1 text-center">最优个体路径</div>
              <svg viewBox="-10 -10 300 300" style={{ width: "100%", height: 140 }}>
                {currentBest.tour.map((ci, i) => {
                  const next = currentBest.tour[(i + 1) % C];
                  const [x1, y1] = CITIES[ci], [x2, y2] = CITIES[next];
                  return <motion.line key={i} x1={x1} y1={y1} x2={x2} y2={y2}
                    stroke="#7c3aed" strokeWidth={2} strokeOpacity={0.8}
                    animate={{ x1, y1, x2, y2 }} transition={{ duration: 0.4 }} />;
                })}
                {CITIES.map(([cx, cy], i) => (
                  <g key={i}>
                    <circle cx={cx} cy={cy} r={6} fill="#7c3aed" />
                    <text x={cx + 8} y={cy - 4} fill="#64748b" fontSize={8} fontWeight="600">C{i}</text>
                  </g>
                ))}
              </svg>
            </div>

            {/* Controls */}
            <div className="grid grid-cols-3 gap-1.5">
              <button onClick={() => { setPlaying(false); setGenIdx((i) => Math.max(0, i - 1)); }}
                className="flex items-center justify-center p-2 bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-600 dark:text-slate-300 rounded-xl text-xs">
                ‹
              </button>
              <button onClick={() => setPlaying((p) => !p)}
                className="flex items-center justify-center gap-1 bg-violet-600 hover:bg-violet-700 text-white text-xs font-semibold px-2 py-2 rounded-xl">
                {playing ? <Pause size={12} /> : <Play size={12} />}
                {playing ? "暂停" : "运行"}
              </button>
              <button onClick={step}
                className="flex items-center justify-center p-2 bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-600 dark:text-slate-300 rounded-xl text-xs">
                ›
              </button>
            </div>
            <button onClick={reset}
              className="flex items-center justify-center gap-1.5 bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 text-slate-600 dark:text-slate-300 text-xs font-semibold px-3 py-2 rounded-xl transition-colors">
              <RotateCcw size={12} /> 重置（重新初始化种群）
            </button>

            {/* Stats */}
            <div className="grid grid-cols-1 gap-2">
              {[
                { label: "最佳适应度", value: record.bestFit.toFixed(3), color: "text-emerald-600 dark:text-emerald-400" },
                { label: "平均适应度", value: record.avgFit.toFixed(3), color: "text-blue-600 dark:text-blue-400" },
                { label: "最差适应度", value: record.worstFit.toFixed(3), color: "text-rose-600 dark:text-rose-400" },
              ].map(({ label, value, color }) => (
                <div key={label} className="bg-slate-50 dark:bg-slate-800 rounded-lg p-2 flex items-center justify-between">
                  <span className="text-[10px] text-slate-500 dark:text-slate-400">{label}</span>
                  <span className={`text-xs font-black font-mono ${color}`}>{value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Generation slider */}
        <div>
          <input type="range" min={0} max={GENS.length - 1} value={genIdx}
            onChange={(e) => { setPlaying(false); setGenIdx(parseInt(e.target.value)); }}
            className="w-full accent-violet-600" />
          <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
            <span>第 0 代（随机初始种群）</span>
            <span>第 {MAX_GEN} 代</span>
          </div>
        </div>

        {/* Fitness chart */}
        <div className="bg-slate-50 dark:bg-slate-800 rounded-xl p-3">
          <div className="text-xs text-slate-500 dark:text-slate-400 mb-2">适应度进化曲线（{MAX_GEN} 代）</div>
          <svg viewBox={`0 0 400 80`} className="w-full" style={{ height: 90 }}>
            {[0, 0.25, 0.5, 0.75, 1].map((r) => (
              <line key={r} x1={0} y1={80 * (1 - r)} x2={400} y2={80 * (1 - r)}
                stroke="#e2e8f0" strokeWidth={0.5} className="dark:stroke-slate-700" />
            ))}
            {["bestFit", "avgFit", "worstFit"].map((key, ki) => {
              const colors = ["#10b981", "#3b82f6", "#f43f5e"];
              const data = GENS.slice(0, genIdx + 1).map((g, i) => {
                const x = (i / (MAX_GEN)) * 400;
                const v = g[key as keyof GenRecord] as number;
                const y = 75 - ((v - FITNESS_MIN) / (FITNESS_MAX - FITNESS_MIN + 0.001)) * 65;
                return `${x},${y}`;
              }).join(" ");
              return data.includes(" ") ? (
                <polyline key={key} points={data} fill="none" stroke={colors[ki]} strokeWidth={1.5} />
              ) : null;
            })}
            {/* Current gen line */}
            {(() => {
              const x = (genIdx / MAX_GEN) * 400;
              return <line x1={x} y1={0} x2={x} y2={80} stroke="#a78bfa" strokeWidth={1.5} strokeDasharray="3,2" />;
            })()}
          </svg>
          <div className="flex gap-4 mt-1 justify-center">
            {[["最佳", "#10b981"], ["平均", "#3b82f6"], ["最差", "#f43f5e"]].map(([l, c]) => (
              <span key={l} className="flex items-center gap-1 text-[10px] text-slate-500">
                <span className="w-4 h-0.5 inline-block" style={{ background: c }} /> {l}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

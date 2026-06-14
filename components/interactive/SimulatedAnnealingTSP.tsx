"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { motion } from "framer-motion";
import { Play, Pause, RotateCcw, FastForward } from "lucide-react";

// 10 fixed cities (x,y) in a 300×250 canvas
const CITIES: [number, number][] = [
  [40, 80], [140, 30], [240, 60], [290, 140], [250, 230],
  [160, 270], [60, 240], [20, 160], [130, 140], [200, 160],
];
const N = CITIES.length;

function dist(a: number, b: number) {
  const [x1, y1] = CITIES[a], [x2, y2] = CITIES[b];
  return Math.hypot(x2 - x1, y2 - y1);
}

function tourCost(tour: number[]) {
  let c = 0;
  for (let i = 0; i < tour.length; i++) c += dist(tour[i], tour[(i + 1) % tour.length]);
  return c;
}

function swap2opt(tour: number[], i: number, j: number): number[] {
  // Reverse the segment between i and j
  const t = [...tour];
  let a = i + 1, b = j;
  while (a < b) { [t[a], t[b]] = [t[b], t[a]]; a++; b--; }
  return t;
}

// Pre-compute SA run and record snapshots every K steps
interface Snapshot {
  step: number;
  tour: number[];
  cost: number;
  temp: number;
  accepted: boolean;
}

function runSA(): Snapshot[] {
  const T0 = 200, alpha = 0.97, STEPS = 600, SNAPSHOT_INTERVAL = 10;
  let tour = Array.from({ length: N }, (_, i) => i);
  // shuffle
  for (let i = N - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [tour[i], tour[j]] = [tour[j], tour[i]];
  }
  let cost = tourCost(tour);
  let T = T0;
  const snaps: Snapshot[] = [{ step: 0, tour: [...tour], cost, temp: T, accepted: true }];

  for (let s = 1; s <= STEPS; s++) {
    const i = Math.floor(Math.random() * N);
    let j = Math.floor(Math.random() * (N - 1));
    if (j >= i) j++;
    const [a, b] = [Math.min(i, j), Math.max(i, j)];
    const newTour = swap2opt(tour, a, b);
    const newCost = tourCost(newTour);
    const delta = newCost - cost;
    const accepted = delta < 0 || Math.random() < Math.exp(-delta / T);
    if (accepted) { tour = newTour; cost = newCost; }
    T *= alpha;
    if (s % SNAPSHOT_INTERVAL === 0)
      snaps.push({ step: s, tour: [...tour], cost, temp: T, accepted });
  }
  return snaps;
}

// Pre-generate once
const STATIC_SNAPS = runSA();
const BEST_COST = Math.min(...STATIC_SNAPS.map((s) => s.cost));
const INITIAL_COST = STATIC_SNAPS[0].cost;

export function SimulatedAnnealingTSP() {
  const [snapIdx, setSnapIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1); // 1=normal, 2=fast, 3=turbo
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const snap = STATIC_SNAPS[snapIdx];
  const totalSnaps = STATIC_SNAPS.length;

  const step = useCallback(() => {
    setSnapIdx((i) => {
      if (i >= totalSnaps - 1) { setPlaying(false); return i; }
      return i + 1;
    });
  }, [totalSnaps]);

  useEffect(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (!playing) return;
    const delay = speed === 3 ? 40 : speed === 2 ? 100 : 200;
    timerRef.current = setInterval(step, delay);
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [playing, speed, step]);

  const reset = useCallback(() => {
    setPlaying(false);
    setSnapIdx(0);
  }, []);

  const { tour, cost, temp } = snap;
  const progress = snapIdx / (totalSnaps - 1);
  const T0 = STATIC_SNAPS[0].temp;

  // Cost chart data (last 30 snaps)
  const chartData = STATIC_SNAPS.slice(0, snapIdx + 1);
  const costMin = Math.min(...STATIC_SNAPS.map((s) => s.cost));
  const costMax = INITIAL_COST;
  const chartW = 280, chartH = 70;

  const toChartX = (i: number) => (i / (totalSnaps - 1)) * chartW;
  const toChartY = (c: number) => chartH - ((c - costMin) / (costMax - costMin || 1)) * (chartH - 6) - 3;

  const tempRatio = temp / T0;
  const coldColor = tempRatio < 0.1 ? "#3b82f6" : tempRatio < 0.3 ? "#8b5cf6" : tempRatio < 0.6 ? "#f97316" : "#ef4444";

  return (
    <div className="w-full max-w-4xl mx-auto my-6 rounded-2xl overflow-hidden border border-red-200 dark:border-red-900 shadow-xl bg-white dark:bg-slate-900">
      {/* Header */}
      <div className="bg-gradient-to-r from-red-500 via-orange-500 to-amber-500 px-6 py-4">
        <h3 className="text-base font-bold text-white">模拟退火 (SA) — TSP 求解过程</h3>
        <p className="text-xs text-red-100 mt-0.5">
          温度参数 T 随步数指数衰减，高温时接受劣解以逃离局部最优
        </p>
      </div>

      <div className="p-5 flex flex-col gap-4">
        {/* Main split: city map + metrics */}
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto] gap-5">
          {/* City map */}
          <div>
            <div className="text-xs text-slate-500 dark:text-slate-400 mb-2 flex items-center gap-2">
              <span>TSP 路径（{N} 城市）</span>
              <span className="text-[10px] px-1.5 py-0.5 rounded-full font-mono" style={{ background: coldColor + "22", color: coldColor }}>
                T = {temp.toFixed(1)}
              </span>
            </div>
            <div className="bg-slate-50 dark:bg-slate-800 rounded-xl p-2 overflow-hidden">
              <svg viewBox="-20 -20 360 310" className="w-full max-h-64">
                {/* Tour path */}
                {tour.map((ci, i) => {
                  const next = tour[(i + 1) % N];
                  const [x1, y1] = CITIES[ci], [x2, y2] = CITIES[next];
                  return (
                    <motion.line
                      key={`${ci}-${next}-${i}`}
                      x1={x1} y1={y1} x2={x2} y2={y2}
                      stroke={coldColor}
                      strokeWidth={2}
                      strokeOpacity={0.8}
                      animate={{ x1, y1, x2, y2, stroke: coldColor }}
                      transition={{ duration: 0.3 }}
                    />
                  );
                })}

                {/* Cities */}
                {CITIES.map(([cx, cy], i) => (
                  <g key={i}>
                    <circle cx={cx} cy={cy} r={8} fill="#1e293b" className="dark:fill-white" strokeWidth={0} />
                    <circle cx={cx} cy={cy} r={5} fill={coldColor} />
                    <text x={cx + 10} y={cy - 5} fill="#64748b" fontSize={9} fontWeight="600">C{i}</text>
                  </g>
                ))}
              </svg>
            </div>
          </div>

          {/* Right info */}
          <div className="flex flex-col gap-3 min-w-[200px]">
            {/* Controls */}
            <div className="flex flex-col gap-2">
              <div className="flex gap-2">
                <button
                  onClick={() => setPlaying((p) => !p)}
                  className="flex-1 flex items-center justify-center gap-1.5 bg-red-500 hover:bg-red-600 text-white text-sm font-semibold px-3 py-2 rounded-xl transition-colors"
                >
                  {playing ? <Pause size={14} /> : <Play size={14} />}
                  {playing ? "暂停" : "播放"}
                </button>
                <button
                  onClick={reset}
                  className="flex items-center justify-center p-2 bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 text-slate-600 dark:text-slate-300 rounded-xl transition-colors"
                >
                  <RotateCcw size={15} />
                </button>
              </div>
              {/* Speed */}
              <div className="flex gap-1">
                {[1, 2, 3].map((s) => (
                  <button
                    key={s}
                    onClick={() => setSpeed(s)}
                    className={`flex-1 text-xs px-2 py-1.5 rounded-lg font-semibold transition-colors ${speed === s ? "bg-orange-500 text-white" : "bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600"}`}
                  >
                    {s === 1 ? "正常" : s === 2 ? "快速" : "极速"}
                  </button>
                ))}
              </div>
            </div>

            {/* Progress */}
            <div>
              <input
                type="range" min={0} max={totalSnaps - 1} value={snapIdx}
                onChange={(e) => { setPlaying(false); setSnapIdx(parseInt(e.target.value)); }}
                className="w-full accent-red-500"
              />
              <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
                <span>步骤 0</span>
                <span>步骤 {(snapIdx * 10).toLocaleString()}</span>
                <span>600</span>
              </div>
            </div>

            {/* Metrics */}
            {[
              { label: "当前路径长度", value: cost.toFixed(1), good: cost <= BEST_COST * 1.05, sub: "初始: " + INITIAL_COST.toFixed(1) },
              { label: "最优路径长度", value: BEST_COST.toFixed(1), good: true, sub: "历史最佳" },
              { label: "当前温度 T", value: temp.toFixed(1), good: false, sub: `T₀=${T0}，α=0.97` },
            ].map(({ label, value, good, sub }) => (
              <div key={label} className="bg-slate-50 dark:bg-slate-800 rounded-xl p-2.5 text-center">
                <div className="text-[10px] text-slate-500 dark:text-slate-400">{label}</div>
                <div className={`text-base font-black font-mono ${good ? "text-emerald-600 dark:text-emerald-400" : "text-orange-500 dark:text-orange-400"}`}>
                  {value}
                </div>
                <div className="text-[10px] text-slate-400">{sub}</div>
              </div>
            ))}

            {/* Accept prob */}
            <div className="bg-red-50 dark:bg-red-900/20 rounded-xl p-2.5 border border-red-100 dark:border-red-800">
              <div className="text-[10px] text-red-600 dark:text-red-400 font-semibold mb-1">接受劣解概率</div>
              <div className="text-xs font-mono text-red-700 dark:text-red-300">
                P = e<sup>−ΔE/T</sup>
              </div>
              <div className="text-[10px] text-slate-500 dark:text-slate-400 mt-0.5">
                T→0 时 P→0（不再接受劣解）
              </div>
            </div>
          </div>
        </div>

        {/* Cost chart */}
        <div className="bg-slate-50 dark:bg-slate-800 rounded-xl p-3">
          <div className="text-xs text-slate-500 dark:text-slate-400 mb-2 flex items-center justify-between">
            <span>路径总长度随步骤变化</span>
            <span className="text-[10px] font-mono text-emerald-600 dark:text-emerald-400">
              改善 {(((INITIAL_COST - cost) / INITIAL_COST) * 100).toFixed(1)}%
            </span>
          </div>
          <svg viewBox={`0 0 ${chartW} ${chartH}`} className="w-full" style={{ height: 80 }}>
            {/* Grid lines */}
            {[0, 0.25, 0.5, 0.75, 1].map((r) => (
              <line key={r} x1={0} y1={chartH * (1 - r)} x2={chartW} y2={chartH * (1 - r)}
                stroke="#e2e8f0" strokeWidth={0.5} className="dark:stroke-slate-700" />
            ))}
            {/* Cost line */}
            {chartData.length > 1 && (
              <polyline
                points={chartData.map((s, i) => `${toChartX(i)},${toChartY(s.cost)}`).join(" ")}
                fill="none" stroke="#ef4444" strokeWidth={1.5}
              />
            )}
            {/* Current indicator */}
            {chartData.length > 0 && (
              <circle
                cx={toChartX(snapIdx)}
                cy={toChartY(cost)}
                r={3} fill="#ef4444"
              />
            )}
          </svg>
          <div className="flex justify-between text-[10px] text-slate-400 mt-1">
            <span>高={costMax.toFixed(0)}</span>
            <span>低={costMin.toFixed(0)}</span>
          </div>
        </div>

        {/* Temperature bar */}
        <div className="flex items-center gap-3">
          <span className="text-xs text-slate-500 dark:text-slate-400 w-16">温度</span>
          <div className="flex-1 h-3 bg-slate-100 dark:bg-slate-700 rounded-full overflow-hidden">
            <motion.div
              className="h-full rounded-full"
              style={{ background: coldColor }}
              animate={{ width: `${tempRatio * 100}%` }}
              transition={{ duration: 0.2 }}
            />
          </div>
          <span className="text-xs font-mono" style={{ color: coldColor }}>{(tempRatio * 100).toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
}

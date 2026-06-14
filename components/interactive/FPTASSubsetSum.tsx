"use client";

import { useState, useMemo } from "react";
import { motion } from "framer-motion";

// Fixed example: items, target
const ITEMS = [100, 200, 300, 400, 500];
const TARGET = 950;
const OPT = 900; // {200, 300, 400}

function computeFPTAS(items: number[], t: number, eps: number) {
  const n = items.length;
  if (n === 0) return { approx: 0, states: 0, scaledItems: [], K: 0, tScaled: 0 };

  const maxVal = Math.max(...items);
  const K = (eps * maxVal) / n;                          // scaling factor
  const scaled = items.map((a) => Math.floor(a / K));    // â_i
  const tScaled = Math.floor(t / K);                     // t̂

  // DP on scaled items
  let reachable = new Set<number>([0]);
  for (const sv of scaled) {
    const toAdd: number[] = [];
    for (const s of reachable) {
      const ns = s + sv;
      if (ns <= tScaled && !reachable.has(ns)) toAdd.push(ns);
    }
    for (const ns of toAdd) reachable.add(ns);
  }

  const bestScaled = Math.max(...reachable);
  const approx = Math.round(bestScaled * K); // de-scale (approximate)

  return { approx, states: reachable.size, scaledItems: scaled, K, tScaled };
}

const EPS_PRESETS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0];

export function FPTASSubsetSum() {
  const [eps, setEps] = useState(0.2);

  const { approx, states, scaledItems, K, tScaled } = useMemo(
    () => computeFPTAS(ITEMS, TARGET, eps),
    [eps]
  );

  const accuracy = approx / OPT;
  const guaranteed = 1 / (1 + eps);
  const maxStates = Math.round(TARGET / K);

  return (
    <div className="w-full max-w-4xl mx-auto my-6 rounded-2xl overflow-hidden border border-violet-200 dark:border-violet-800 shadow-xl bg-white dark:bg-slate-900">
      {/* Header */}
      <div className="bg-gradient-to-r from-violet-600 via-purple-600 to-fuchsia-600 px-6 py-4">
        <h3 className="text-base font-bold text-white">子集和 FPTAS — ε 缩放交互演示</h3>
        <p className="text-xs text-violet-100 mt-0.5">
          调节 ε 观察精度与 DP 状态数的权衡（FPTAS 核心思想）
        </p>
      </div>

      <div className="p-5 flex flex-col gap-5">
        {/* Problem setup */}
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-slate-50 dark:bg-slate-800 rounded-xl p-3">
            <div className="text-xs text-slate-500 dark:text-slate-400 mb-1.5 font-semibold">原始物品</div>
            <div className="flex flex-wrap gap-1.5">
              {ITEMS.map((v, i) => (
                <span key={i} className="text-xs px-2 py-1 rounded-lg bg-violet-100 dark:bg-violet-900/50 text-violet-700 dark:text-violet-300 font-mono font-bold">
                  {v}
                </span>
              ))}
            </div>
            <div className="text-xs text-slate-400 dark:text-slate-500 mt-2">
              目标 t = {TARGET}，OPT = {OPT} ({"{"}200+300+400{"}"})
            </div>
          </div>
          <div className="bg-slate-50 dark:bg-slate-800 rounded-xl p-3">
            <div className="text-xs text-slate-500 dark:text-slate-400 mb-1.5 font-semibold">缩放后物品 (⌊aᵢ/K⌋)</div>
            <div className="flex flex-wrap gap-1.5">
              {scaledItems.map((v, i) => (
                <span key={i} className="text-xs px-2 py-1 rounded-lg bg-fuchsia-100 dark:bg-fuchsia-900/50 text-fuchsia-700 dark:text-fuchsia-300 font-mono font-bold">
                  {v}
                </span>
              ))}
            </div>
            <div className="text-xs text-slate-400 dark:text-slate-500 mt-2">
              缩放目标 t̂ = {tScaled}，K = {K.toFixed(2)}
            </div>
          </div>
        </div>

        {/* ε slider */}
        <div className="bg-gradient-to-r from-violet-50 to-fuchsia-50 dark:from-violet-900/20 dark:to-fuchsia-900/20 rounded-xl p-4 border border-violet-100 dark:border-violet-800">
          <div className="flex items-center justify-between mb-3">
            <div>
              <span className="text-sm font-bold text-violet-700 dark:text-violet-300">ε = {eps.toFixed(2)}</span>
              <span className="text-xs text-slate-500 dark:text-slate-400 ml-2">（近似参数）</span>
            </div>
            <span className="text-xs font-mono bg-violet-500 text-white px-2 py-0.5 rounded-full">
              保证 ≥ OPT/(1+ε) = {Math.round(OPT / (1 + eps))}
            </span>
          </div>

          <input
            type="range" min={0.05} max={1.0} step={0.01}
            value={eps}
            onChange={(e) => setEps(parseFloat(e.target.value))}
            className="w-full accent-violet-600 cursor-pointer"
          />

          <div className="flex justify-between text-[10px] text-slate-400 mt-1">
            <span>0.05（高精度）</span>
            <span>1.0（低精度，快速）</span>
          </div>

          {/* Preset buttons */}
          <div className="flex gap-2 mt-3 flex-wrap">
            <span className="text-[10px] text-slate-500 self-center">预设：</span>
            {EPS_PRESETS.map((e) => (
              <button
                key={e}
                onClick={() => setEps(e)}
                className={`text-xs px-2.5 py-1 rounded-full font-mono font-semibold transition-colors ${
                  Math.abs(eps - e) < 0.005
                    ? "bg-violet-600 text-white"
                    : "bg-white dark:bg-slate-700 border border-violet-200 dark:border-violet-700 text-violet-600 dark:text-violet-300 hover:bg-violet-50 dark:hover:bg-violet-900/30"
                }`}
              >
                {e}
              </button>
            ))}
          </div>
        </div>

        {/* Key metrics */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            {
              label: "近似结果",
              value: approx,
              sub: `OPT = ${OPT}`,
              color: accuracy >= 1 ? "text-emerald-600 dark:text-emerald-400" : accuracy >= 0.9 ? "text-amber-600 dark:text-amber-400" : "text-rose-600 dark:text-rose-400",
              bg: "bg-emerald-50 dark:bg-emerald-900/20",
            },
            {
              label: "实际精度",
              value: `${(accuracy * 100).toFixed(1)}%`,
              sub: `保证 ≥ ${(guaranteed * 100).toFixed(1)}%`,
              color: accuracy >= 0.95 ? "text-emerald-600 dark:text-emerald-400" : "text-amber-600 dark:text-amber-400",
              bg: "bg-amber-50 dark:bg-amber-900/20",
            },
            {
              label: "DP 状态数",
              value: states,
              sub: `上限 ≤ n/ε = ${ITEMS.length}/ε ≈ ${Math.round(ITEMS.length / eps)}`,
              color: "text-violet-600 dark:text-violet-400",
              bg: "bg-violet-50 dark:bg-violet-900/20",
            },
            {
              label: "缩放因子 K",
              value: K.toFixed(2),
              sub: `ε×max/n = ${eps.toFixed(2)}×500/${ITEMS.length}`,
              color: "text-fuchsia-600 dark:text-fuchsia-400",
              bg: "bg-fuchsia-50 dark:bg-fuchsia-900/20",
            },
          ].map(({ label, value, sub, color, bg }) => (
            <motion.div
              key={label}
              layout
              className={`rounded-xl p-3 ${bg} text-center`}
            >
              <div className="text-xs text-slate-500 dark:text-slate-400">{label}</div>
              <motion.div
                key={String(value)}
                initial={{ scale: 0.95, opacity: 0.7 }}
                animate={{ scale: 1, opacity: 1 }}
                className={`text-lg font-black font-mono ${color}`}
              >
                {value}
              </motion.div>
              <div className="text-[10px] text-slate-400 dark:text-slate-500 leading-tight mt-0.5">{sub}</div>
            </motion.div>
          ))}
        </div>

        {/* Trade-off visualization */}
        <div className="bg-slate-50 dark:bg-slate-800 rounded-xl p-4">
          <div className="text-xs font-bold text-slate-600 dark:text-slate-300 mb-3">精度 vs DP 状态数权衡（随 ε 变化）</div>
          <div className="relative h-24">
            {/* Accuracy curve */}
            <svg viewBox="0 0 400 80" className="w-full h-full" preserveAspectRatio="none">
              {/* Grid */}
              {[0, 20, 40, 60, 80].map((y) => (
                <line key={y} x1={0} y1={y} x2={400} y2={y} stroke="#e2e8f0" strokeWidth={0.5} className="dark:stroke-slate-700" />
              ))}

              {/* States curve (scaled, inverse of eps) */}
              <polyline
                points={EPS_PRESETS.map((e, i) => {
                  const r = computeFPTAS(ITEMS, TARGET, e);
                  const x = (e - 0.05) / 0.95 * 400;
                  const normalized = Math.min(1, r.states / 50);
                  const y = 75 - normalized * 65;
                  return `${x},${y}`;
                }).join(" ")}
                fill="none" stroke="#8b5cf6" strokeWidth={2} opacity={0.7}
              />
              {/* Accuracy curve */}
              <polyline
                points={EPS_PRESETS.map((e) => {
                  const r = computeFPTAS(ITEMS, TARGET, e);
                  const x = (e - 0.05) / 0.95 * 400;
                  const accuracy = r.approx / OPT;
                  const y = 75 - accuracy * 65;
                  return `${x},${y}`;
                }).join(" ")}
                fill="none" stroke="#10b981" strokeWidth={2}
              />

              {/* Current ε marker */}
              {(() => {
                const x = (eps - 0.05) / 0.95 * 400;
                return <line x1={x} y1={0} x2={x} y2={80} stroke="#f59e0b" strokeWidth={2} strokeDasharray="3,2" />;
              })()}
            </svg>

            {/* Legend */}
            <div className="absolute top-1 right-2 flex flex-col gap-1">
              <div className="flex items-center gap-1.5 text-[10px] text-slate-500">
                <div className="w-4 h-0.5 bg-emerald-500" /> 精度
              </div>
              <div className="flex items-center gap-1.5 text-[10px] text-slate-500">
                <div className="w-4 h-0.5 bg-violet-500" /> 状态数(归一化)
              </div>
              <div className="flex items-center gap-1.5 text-[10px] text-amber-500">
                <div className="w-4 h-0.5 bg-amber-500" style={{ borderTop: "2px dashed" }} /> 当前ε
              </div>
            </div>
          </div>
          <div className="flex justify-between text-[10px] text-slate-400 mt-1">
            <span>ε=0.05 (慢&精)</span>
            <span>ε=1.0 (快&粗)</span>
          </div>
        </div>

        {/* Formula reminder */}
        <div className="bg-violet-50 dark:bg-violet-900/20 rounded-xl p-3 border border-violet-100 dark:border-violet-800">
          <div className="text-[11px] font-mono text-violet-700 dark:text-violet-300 space-y-1">
            <div>K = ε × max(aᵢ) / n = {eps.toFixed(2)} × 500 / 5 = <strong>{K.toFixed(2)}</strong></div>
            <div>âᵢ = ⌊aᵢ / K⌋ → 最大缩放值 ≤ n/ε = {ITEMS.length}/{eps.toFixed(2)} ≈ {Math.round(ITEMS.length / eps)}</div>
            <div>时间 O(n²/ε) = O({ITEMS.length}²/{eps.toFixed(2)}) ≈ O({Math.round(ITEMS.length ** 2 / eps)})</div>
            <div className="text-emerald-600 dark:text-emerald-400">输出 ≥ OPT/(1+ε) = {OPT}/(1+{eps.toFixed(2)}) = {Math.round(OPT / (1 + eps))}</div>
          </div>
        </div>
      </div>
    </div>
  );
}

"use client";

import React, { useState, useMemo } from "react";
import { motion } from "framer-motion";
import { Calculator, TrendingUp, Cpu, Info } from "lucide-react";

function amdahlSpeedup(s: number, n: number): number {
  return 1 / (s + (1 - s) / n);
}

function generateCurve(serialFraction: number, maxN: number): { n: number; speedup: number }[] {
  const points: { n: number; speedup: number }[] = [];
  for (let n = 1; n <= maxN; n++) {
    points.push({ n, speedup: amdahlSpeedup(serialFraction, n) });
  }
  return points;
}

export default function AmdahlLawCalculator() {
  const [serialPercent, setSerialPercent] = useState(10);
  const [processorCount, setProcessorCount] = useState(16);
  const maxN = 128;

  const serialFraction = serialPercent / 100;
  const speedup = useMemo(() => amdahlSpeedup(serialFraction, processorCount), [serialFraction, processorCount]);
  const maxSpeedup = useMemo(() => 1 / serialFraction, [serialFraction]);
  const efficiency = useMemo(() => speedup / processorCount * 100, [speedup, processorCount]);
  const curve = useMemo(() => generateCurve(serialFraction, maxN), [serialFraction]);

  const chartWidth = 500;
  const chartHeight = 260;
  const padding = { top: 20, right: 30, bottom: 40, left: 60 };
  const plotW = chartWidth - padding.left - padding.right;
  const plotH = chartHeight - padding.top - padding.bottom;

  const maxY = Math.min(maxSpeedup * 1.1, maxN);
  const scaleX = (n: number) => padding.left + ((n - 1) / (maxN - 1)) * plotW;
  const scaleY = (s: number) => padding.top + plotH - (s / maxY) * plotH;

  const pathD = curve
    .map((p, i) => `${i === 0 ? "M" : "L"} ${scaleX(p.n).toFixed(1)} ${scaleY(p.speedup).toFixed(1)}`)
    .join(" ");

  const linearPathD = [1, maxN]
    .map((n, i) => `${i === 0 ? "M" : "L"} ${scaleX(n).toFixed(1)} ${scaleY(n).toFixed(1)}`)
    .join(" ");

  const theoreticalMaxD = `M ${scaleX(1).toFixed(1)} ${scaleY(maxSpeedup).toFixed(1)} L ${scaleX(maxN).toFixed(1)} ${scaleY(maxSpeedup).toFixed(1)}`;

  const gridLinesY = [0, 0.25, 0.5, 0.75, 1].map(f => scaleY(f * maxY));
  const gridLabelsY = [0, 0.25, 0.5, 0.75, 1].map(f => (f * maxY).toFixed(1));

  const yTicks = 5;
  const yTickValues = Array.from({ length: yTicks + 1 }, (_, i) => (i / yTicks) * maxY);

  return (
    <div className="w-full max-w-4xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-emerald-50 dark:from-slate-900 dark:to-emerald-950 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-2 text-center flex items-center justify-center gap-2">
        <Calculator className="w-6 h-6 text-emerald-600" />
        Amdahl 定律计算器
      </h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 text-center mb-6">
        加速比 = 1 / (S + P/N)，探索并行加速的理论上限
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div className="space-y-4">
          <div>
            <label className="flex items-center justify-between text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              <span>串行比例 S</span>
              <span className="font-mono text-emerald-600 dark:text-emerald-400">{serialPercent}%</span>
            </label>
            <input
              type="range"
              min={1}
              max={50}
              value={serialPercent}
              onChange={e => setSerialPercent(Number(e.target.value))}
              className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-600"
            />
            <div className="flex justify-between text-xs text-slate-400 mt-1">
              <span>1%</span><span>25%</span><span>50%</span>
            </div>
          </div>

          <div>
            <label className="flex items-center justify-between text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
              <span>处理器数量 N</span>
              <span className="font-mono text-emerald-600 dark:text-emerald-400">{processorCount}</span>
            </label>
            <input
              type="range"
              min={1}
              max={maxN}
              value={processorCount}
              onChange={e => setProcessorCount(Number(e.target.value))}
              className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-600"
            />
            <div className="flex justify-between text-xs text-slate-400 mt-1">
              <span>1</span><span>32</span><span>64</span><span>128</span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <motion.div
              key={speedup.toFixed(2)}
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              className="bg-white dark:bg-slate-800 rounded-lg p-3 border border-slate-200 dark:border-slate-700 text-center"
            >
              <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">加速比</div>
              <div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400 font-mono">
                {speedup.toFixed(2)}x
              </div>
            </motion.div>
            <div className="bg-white dark:bg-slate-800 rounded-lg p-3 border border-slate-200 dark:border-slate-700 text-center">
              <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">理论上限</div>
              <div className="text-2xl font-bold text-orange-500 font-mono">
                {maxSpeedup.toFixed(1)}x
              </div>
            </div>
            <div className="bg-white dark:bg-slate-800 rounded-lg p-3 border border-slate-200 dark:border-slate-700 text-center">
              <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">并行效率</div>
              <div className="text-2xl font-bold text-blue-600 dark:text-blue-400 font-mono">
                {efficiency.toFixed(1)}%
              </div>
            </div>
            <div className="bg-white dark:bg-slate-800 rounded-lg p-3 border border-slate-200 dark:border-slate-700 text-center">
              <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">浪费的CPU</div>
              <div className="text-2xl font-bold text-red-500 font-mono">
                {(processorCount - speedup).toFixed(1)}
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-lg p-4 border border-slate-200 dark:border-slate-700">
          <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="w-full h-auto">
            <defs>
              <linearGradient id="speedupGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.3" />
                <stop offset="100%" stopColor="#10b981" stopOpacity="0.02" />
              </linearGradient>
            </defs>

            {yTickValues.map((v, i) => {
              const y = scaleY(v);
              return (
                <g key={i}>
                  <line x1={padding.left} y1={y} x2={chartWidth - padding.right} y2={y}
                    stroke="currentColor" strokeOpacity={0.08} className="text-slate-400" />
                  <text x={padding.left - 8} y={y + 4} textAnchor="end"
                    className="fill-slate-400 text-[10px]">{v.toFixed(1)}</text>
                </g>
              );
            })}

            {[1, 2, 4, 8, 16, 32, 64, 128].filter(n => n <= maxN).map(n => {
              const x = scaleX(n);
              return (
                <g key={n}>
                  <line x1={x} y1={padding.top} x2={x} y2={padding.top + plotH}
                    stroke="currentColor" strokeOpacity={0.06} className="text-slate-400" />
                  <text x={x} y={chartHeight - 8} textAnchor="middle"
                    className="fill-slate-400 text-[10px]">{n}</text>
                </g>
              );
            })}

            <text x={chartWidth / 2} y={chartHeight - 0} textAnchor="middle"
              className="fill-slate-500 text-[11px]">处理器数量 N</text>
            <text x={12} y={chartHeight / 2} textAnchor="middle" transform={`rotate(-90, 12, ${chartHeight / 2})`}
              className="fill-slate-500 text-[11px]">加速比</text>

            <path d={linearPathD} fill="none" stroke="#94a3b8" strokeWidth="1.5" strokeDasharray="6 4" />
            <path d={theoreticalMaxD} fill="none" stroke="#f97316" strokeWidth="1.5" strokeDasharray="4 3" />

            <path d={`${pathD} L ${scaleX(maxN).toFixed(1)} ${scaleY(0).toFixed(1)} L ${scaleX(1).toFixed(1)} ${scaleY(0).toFixed(1)} Z`}
              fill="url(#speedupGrad)" />
            <path d={pathD} fill="none" stroke="#10b981" strokeWidth="2.5" />

            <circle cx={scaleX(processorCount)} cy={scaleY(speedup)} r="5"
              fill="#10b981" stroke="white" strokeWidth="2" />
            <text x={scaleX(processorCount) + 10} y={scaleY(speedup) - 8}
              className="fill-emerald-600 text-[11px] font-bold">{speedup.toFixed(2)}x</text>

            <g transform={`translate(${chartWidth - padding.right - 120}, ${padding.top + 8})`}>
              <line x1="0" y1="6" x2="20" y2="6" stroke="#10b981" strokeWidth="2.5" />
              <text x="25" y="10" className="fill-slate-500 text-[10px]">Amdahl</text>
              <line x1="0" y1="22" x2="20" y2="22" stroke="#94a3b8" strokeWidth="1.5" strokeDasharray="6 4" />
              <text x="25" y="26" className="fill-slate-500 text-[10px]">线性加速</text>
              <line x1="0" y1="38" x2="20" y2="38" stroke="#f97316" strokeWidth="1.5" strokeDasharray="4 3" />
              <text x="25" y="42" className="fill-slate-500 text-[10px]">理论上限</text>
            </g>
          </svg>
        </div>
      </div>

      <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-4">
        <div className="flex items-start gap-2">
          <Info className="w-5 h-5 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-amber-800 dark:text-amber-200">
            <p className="font-semibold mb-1">关键洞察</p>
            <ul className="list-disc list-inside space-y-1 text-xs">
              <li>串行比例 S={serialPercent}% 时，理论最大加速比为 {maxSpeedup.toFixed(1)}x</li>
              <li>使用 {processorCount} 个 CPU，效率仅为 {efficiency.toFixed(1)}%——{(processorCount - speedup).toFixed(1)} 个 CPU 被浪费</li>
              <li>加速曲线越早趋于平坦，说明增加 CPU 收益越小</li>
              <li>实际系统中，锁竞争、同步开销会进一步增加串行比例</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

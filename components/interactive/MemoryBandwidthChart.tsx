"use client";

import { useState } from "react";

interface BandwidthItem {
  label: string;
  before: number;
  after: number;
  tip: string;
}

const items: BandwidthItem[] = [
  { label: "DRAM 带宽利用", before: 60, after: 95, tip: "通过 Prefetch 和 Double Buffering 提升" },
  { label: "SRAM (共享内存)", before: 45, after: 92, tip: "合理 Tiling 使数据驻留 SRAM" },
  { label: "L2 Cache 命中率", before: 55, after: 88, tip: "数据布局优化减少 Cache Miss" },
  { label: "寄存器利用率", before: 70, after: 96, tip: "循环展开和寄存器分配优化" },
  { label: "PCIe 传输效率", before: 40, after: 85, tip: "批量传输 + 流水线重叠" },
];

export function MemoryBandwidthChart() {
  const [showAfter, setShowAfter] = useState(true);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <div className="flex items-center justify-between mb-5">
        <div>
          <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100">内存带宽优化</h3>
          <p className="text-sm text-slate-500 dark:text-slate-400">优化前 vs 优化后的带宽利用率对比</p>
        </div>
        <button
          onClick={() => setShowAfter(!showAfter)}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
            showAfter
              ? "bg-emerald-500 text-white shadow-lg"
              : "bg-slate-200 dark:bg-slate-700 text-slate-600 dark:text-slate-300"
          }`}
        >
          {showAfter ? "显示对比" : "仅优化后"}
        </button>
      </div>

      <div className="space-y-4">
        {items.map((item, i) => (
          <div key={i} className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-bold text-slate-700 dark:text-slate-200">{item.label}</span>
              <span className="text-xs text-slate-500 dark:text-slate-400">
                {showAfter ? `${item.before}% → ${item.after}%` : `${item.after}%`}
              </span>
            </div>

            {showAfter && (
              <div className="relative mb-2">
                <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-slate-400 dark:bg-slate-500 rounded-full transition-all duration-700"
                    style={{ width: `${item.before}%` }}
                  />
                </div>
                <span className="absolute right-2 top-0 text-[10px] text-slate-600 dark:text-slate-300 leading-4">
                  {item.before}%
                </span>
              </div>
            )}

            <div className="relative">
              <div className="h-5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full transition-all duration-1000"
                  style={{ width: `${item.after}%` }}
                />
              </div>
              <span className="absolute right-2 top-0 text-[11px] font-bold text-white leading-5">
                {item.after}%
              </span>
            </div>

            <p className="text-xs text-slate-400 dark:text-slate-500 mt-2">{item.tip}</p>
          </div>
        ))}
      </div>

      <div className="mt-5 grid grid-cols-2 gap-3">
        <div className="bg-white/60 dark:bg-slate-800/60 rounded-lg p-3 border border-slate-200 dark:border-slate-700">
          <div className="text-sm font-bold text-indigo-600 dark:text-indigo-400">优化前均值</div>
          <div className="text-2xl font-bold text-slate-800 dark:text-slate-100">54%</div>
        </div>
        <div className="bg-white/60 dark:bg-slate-800/60 rounded-lg p-3 border border-slate-200 dark:border-slate-700">
          <div className="text-sm font-bold text-purple-600 dark:text-purple-400">优化后均值</div>
          <div className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">91%</div>
        </div>
      </div>
    </div>
  );
}

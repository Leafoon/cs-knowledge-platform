"use client";

import { useState } from "react";

interface TimelineEvent {
  time: number;
  type: "alloc" | "free" | "peak";
  tensor: string;
  size: string;
  addr: string;
}

const events: TimelineEvent[] = [
  { time: 0, type: "alloc", tensor: "input_A", size: "64 KB", addr: "0x1000" },
  { time: 1, type: "alloc", tensor: "weight_W", size: "256 KB", addr: "0x2000" },
  { time: 2, type: "alloc", tensor: "hidden_1", size: "128 KB", addr: "0x8000" },
  { time: 3, type: "free", tensor: "input_A", size: "64 KB", addr: "0x1000" },
  { time: 4, type: "alloc", tensor: "hidden_2", size: "128 KB", addr: "0x1000" },
  { time: 5, type: "peak", tensor: "PEAK", size: "512 KB", addr: "-" },
  { time: 6, type: "free", tensor: "hidden_1", size: "128 KB", addr: "0x8000" },
  { time: 7, type: "free", tensor: "hidden_2", size: "128 KB", addr: "0x1000" },
  { time: 8, type: "free", tensor: "weight_W", size: "256 KB", addr: "0x2000" },
];

const liveAt = [64, 64 + 256, 64 + 256 + 128, 256 + 128, 256 + 128 + 128, 256 + 128 + 128, 256 + 128, 256, 0];

export function MemoryTimelineChart() {
  const [selected, setSelected] = useState<number | null>(null);

  const typeColors = {
    alloc: "bg-emerald-500",
    free: "bg-red-400",
    peak: "bg-amber-500",
  };

  const typeLabels = { alloc: "分配", free: "释放", peak: "峰值" };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-1">内存时间线</h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-5">张量分配和释放的时序图，追踪内存使用变化</p>

      <div className="relative bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700 mb-5">
        <div className="flex items-end gap-1 h-40">
          {liveAt.map((kb, i) => (
            <div key={i} className="flex-1 flex flex-col items-center justify-end h-full">
              <div
                className="w-full bg-gradient-to-t from-indigo-500 to-purple-400 rounded-t transition-all duration-500 relative"
                style={{ height: `${(kb / 600) * 100}%`, minHeight: kb > 0 ? 4 : 0 }}
              >
                {selected === i && (
                  <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-slate-800 text-white text-[10px] px-2 py-0.5 rounded whitespace-nowrap">
                    {kb} KB
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
        <div className="flex gap-1 mt-2">
          {events.map((e, i) => (
            <button
              key={i}
              onClick={() => setSelected(i)}
              className={`flex-1 h-6 rounded text-[9px] font-medium flex items-center justify-center transition-all ${
                selected === i
                  ? "ring-2 ring-indigo-500 ring-offset-1 dark:ring-offset-slate-800"
                  : ""
              } ${typeColors[e.type]} text-white`}
            >
              T{e.time}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
          <h4 className="text-sm font-bold text-slate-700 dark:text-slate-200 mb-3">事件列表</h4>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {events.map((e, i) => (
              <button
                key={i}
                onClick={() => setSelected(i)}
                className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-all ${
                  selected === i
                    ? "bg-indigo-100 dark:bg-indigo-900/40"
                    : "hover:bg-slate-50 dark:hover:bg-slate-700/50"
                }`}
              >
                <span className={`w-2 h-2 rounded-full ${typeColors[e.type]}`} />
                <span className="text-xs font-mono text-slate-500 dark:text-slate-400">T{e.time}</span>
                <span className="text-xs font-bold text-slate-700 dark:text-slate-200">
                  {typeLabels[e.type]}
                </span>
                <span className="text-xs text-slate-500 dark:text-slate-400 flex-1">{e.tensor}</span>
                <span className="text-xs text-indigo-500">{e.size}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
          <h4 className="text-sm font-bold text-slate-700 dark:text-slate-200 mb-3">统计摘要</h4>
          <div className="space-y-3">
            {[
              { label: "总分配次数", value: "5 次", icon: "📥" },
              { label: "总释放次数", value: "4 次", icon: "📤" },
              { label: "峰值内存", value: "512 KB", icon: "📊" },
              { label: "活跃张量", value: "1 个", icon: "🧮" },
            ].map((s, i) => (
              <div key={i} className="flex items-center gap-3">
                <span>{s.icon}</span>
                <span className="text-sm text-slate-600 dark:text-slate-300 flex-1">{s.label}</span>
                <span className="text-sm font-bold text-indigo-600 dark:text-indigo-400">{s.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

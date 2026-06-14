"use client";

import { useState } from "react";

interface Metric {
  label: string;
  value: number;
  unit: string;
  status: "good" | "warn" | "bad";
  desc: string;
}

const pools = [
  {
    name: "CPU 内存池",
    metrics: [
      { label: "使用率", value: 72, unit: "%", status: "warn" as const, desc: "已分配 2.88 GB / 4 GB" },
      { label: "碎片率", value: 12, unit: "%", status: "good" as const, desc: "14 个碎片块，最大连续 1.2 GB" },
      { label: "峰值使用", value: 89, unit: "%", status: "bad" as const, desc: "峰值 3.56 GB，接近 OOM 阈值" },
    ],
  },
  {
    name: "GPU 内存池",
    metrics: [
      { label: "使用率", value: 58, unit: "%", status: "good" as const, desc: "已分配 4.64 GB / 8 GB" },
      { label: "碎片率", value: 25, unit: "%", status: "warn" as const, desc: "频繁小块分配导致碎片化" },
      { label: "峰值使用", value: 76, unit: "%", status: "good" as const, desc: "峰值 6.08 GB，有充足余量" },
    ],
  },
];

const statusColors = {
  good: "bg-emerald-500",
  warn: "bg-amber-500",
  bad: "bg-red-500",
};

const statusText = {
  good: "text-emerald-600 dark:text-emerald-400",
  warn: "text-amber-600 dark:text-amber-400",
  bad: "text-red-600 dark:text-red-400",
};

export function MemoryPoolStatsDiagram() {
  const [activePool, setActivePool] = useState(0);

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-1">内存池统计</h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-5">使用率 / 碎片率 / 峰值三个核心指标监控</p>

      <div className="flex gap-3 mb-6">
        {pools.map((p, i) => (
          <button
            key={i}
            onClick={() => setActivePool(i)}
            className={`px-5 py-2.5 rounded-lg text-sm font-medium transition-all ${
              activePool === i
                ? "bg-indigo-500 text-white shadow-lg"
                : "bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700"
            }`}
          >
            {p.name}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-5">
        {pools[activePool].metrics.map((m, i) => (
          <div
            key={i}
            className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700"
          >
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-bold text-slate-700 dark:text-slate-200">{m.label}</span>
              <span className={`text-xs px-2 py-0.5 rounded-full ${statusColors[m.status]} text-white`}>
                {m.status === "good" ? "正常" : m.status === "warn" ? "警告" : "危险"}
              </span>
            </div>
            <div className="flex items-baseline gap-1 mb-3">
              <span className={`text-3xl font-bold ${statusText[m.status]}`}>{m.value}</span>
              <span className="text-sm text-slate-500 dark:text-slate-400">{m.unit}</span>
            </div>
            <div className="h-2.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden mb-2">
              <div
                className={`h-full rounded-full transition-all duration-700 ${statusColors[m.status]}`}
                style={{ width: `${m.value}%` }}
              />
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400">{m.desc}</p>
          </div>
        ))}
      </div>

      <div className="bg-white/60 dark:bg-slate-800/60 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
        <h4 className="text-sm font-bold text-slate-700 dark:text-slate-200 mb-2">优化建议</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {[
            { icon: "♻️", title: "启用内存复用", desc: "设置 reuse_memory=True 减少重复分配" },
            { icon: "📐", title: "调整池大小", desc: "根据峰值设置 pool_size 避免 OOM" },
            { icon: "🧹", title: "定期碎片整理", desc: "调用 defragment() 合并空闲块" },
          ].map((t, i) => (
            <div key={i} className="flex gap-2">
              <span className="text-lg">{t.icon}</span>
              <div>
                <div className="text-xs font-bold text-slate-700 dark:text-slate-200">{t.title}</div>
                <div className="text-xs text-slate-500 dark:text-slate-400">{t.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

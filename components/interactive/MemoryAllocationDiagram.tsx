"use client";

import { useState } from "react";

const steps = [
  { id: "request", label: "请求分配", desc: "调用 tvm.memory.allocate(shape, dtype) 请求内存", icon: "📥", detail: "tvm.nd.empty((128, 128), dtype='float32', dev=cpu)\n→ 发起内存分配请求\n→ 指定 shape、dtype、device" },
  { id: "search", label: "查找空闲块", desc: "在内存池中搜索合适的空闲块 (Best-Fit/First-Fit)", icon: "🔍", detail: "内存池空闲链表:\n[0x1000, 64KB] [0x8000, 128KB] [0x10000, 256KB]\n\n需要: 64KB → 命中 [0x1000, 64KB]\n策略: First-Fit / Best-Fit / Buddy" },
  { id: "allocate", label: "分配内存", desc: "标记内存块为已用，更新分配表", icon: "✅", detail: "分配结果:\n┌────────────────────────┐\n│ Block: 0x1000          │\n│ Size:  64 KB           │\n│ Status: ALLOCATED      │\n│ Tensor: A[128,128]f32  │\n└────────────────────────┘" },
  { id: "return", label: "返回指针", desc: "返回 DLTensor 指针，可直接读写数据", icon: "🎯", detail: "返回 DLTensor*:\n  - data: 0x1000\n  - shape: [128, 128]\n  - dtype: float32\n  - device: cpu:0\n\n可直接: A.copyfrom(np_data)" },
];

export function MemoryAllocationDiagram() {
  const [active, setActive] = useState(0);
  const [history, setHistory] = useState<number[]>([0]);

  const handleStep = (idx: number) => {
    setActive(idx);
    setHistory((prev) => (prev.includes(idx) ? prev : [...prev, idx]));
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-slate-900 dark:to-indigo-950 rounded-2xl shadow-xl">
      <h3 className="text-xl font-bold text-slate-800 dark:text-slate-100 mb-1">内存分配流程</h3>
      <p className="text-sm text-slate-500 dark:text-slate-400 mb-5">TVM 内存分配: 请求 → 查找空闲块 → 分配 → 返回指针</p>

      <div className="flex items-center justify-center gap-1 mb-6 flex-wrap">
        {steps.map((s, i) => (
          <div key={s.id} className="flex items-center">
            <button
              onClick={() => handleStep(i)}
              className={`flex flex-col items-center px-4 py-3 rounded-xl border-2 transition-all duration-300 min-w-[100px] ${
                active === i
                  ? "border-indigo-500 bg-indigo-100 dark:bg-indigo-900/40 shadow-lg scale-105"
                  : history.includes(i)
                  ? "border-emerald-400 bg-emerald-50 dark:bg-emerald-900/20"
                  : "border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:border-indigo-300"
              }`}
            >
              <span className="text-2xl mb-1">{s.icon}</span>
              <span className="text-xs font-bold text-slate-700 dark:text-slate-200">{s.label}</span>
              {history.includes(i) && !active && (
                <span className="text-[10px] text-emerald-500">✓</span>
              )}
            </button>
            {i < steps.length - 1 && (
              <div className="flex flex-col items-center mx-1">
                <span className={`text-lg font-bold ${active > i ? "text-emerald-400" : "text-indigo-400"}`}>→</span>
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white dark:bg-slate-800/80 rounded-xl p-4 border border-slate-200 dark:border-slate-700">
          <h4 className="font-bold text-slate-700 dark:text-slate-200 mb-1">
            {steps[active].icon} {steps[active].label}
          </h4>
          <p className="text-sm text-slate-500 dark:text-slate-400 mb-3">{steps[active].desc}</p>
          <div className="flex gap-2">
            {steps.map((_, i) => (
              <span
                key={i}
                className={`w-3 h-3 rounded-full ${
                  history.includes(i) ? "bg-emerald-400" : i === active ? "bg-indigo-500" : "bg-slate-300 dark:bg-slate-600"
                }`}
              />
            ))}
          </div>
        </div>
        <div className="bg-slate-900 dark:bg-slate-950 rounded-xl p-4 font-mono text-xs leading-relaxed text-green-400 overflow-x-auto">
          <pre>{steps[active].detail}</pre>
        </div>
      </div>

      <div className="mt-5 grid grid-cols-3 gap-3">
        {[
          { title: "Pool 分配器", desc: "减少系统调用开销" },
          { title: "内存对齐", desc: "按 512B/4KB 对齐" },
          { title: "零拷贝", desc: "共享底层 buffer" },
        ].map((f, i) => (
          <div key={i} className="bg-white/60 dark:bg-slate-800/60 rounded-lg p-3 border border-slate-200 dark:border-slate-700">
            <div className="text-sm font-bold text-indigo-600 dark:text-indigo-400">{f.title}</div>
            <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">{f.desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

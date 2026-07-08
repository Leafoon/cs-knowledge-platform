"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Box, Users, Zap } from "lucide-react";

const modes = [
  { id: "process", label: "进程", icon: Box, color: "#ef4444",
    description: "独立的餐厅 — 各自拥有厨房、食材、账本，互不干扰",
    items: [
      { name: "进程 A", sub: ["线程 1", "线程 2"], color: "#ef4444" },
      { name: "进程 B", sub: ["线程 1"], color: "#3b82f6" },
    ],
    pros: ["隔离性强", "可利用多核 CPU", "一个崩溃不影响其他"],
    cons: ["创建成本高", "内存占用多", "数据传递麻烦"],
  },
  { id: "thread", label: "线程", icon: Users, color: "#3b82f6",
    description: "同一个厨房里的多个厨师 — 共享厨房、食材、冰箱",
    items: [
      { name: "进程", sub: ["主线程", "工作线程 1", "工作线程 2"], color: "#3b82f6" },
    ],
    pros: ["共享内存", "创建成本较低", "数据交换方便"],
    cons: ["需要同步机制", "受 GIL 限制", "调试复杂"],
  },
  { id: "coroutine", label: "协程", icon: Zap, color: "#10b981",
    description: "一个厨师同时照看多口锅 — 等待时切换到其他任务",
    items: [
      { name: "线程", sub: ["协程 A", "协程 B", "协程 C"], color: "#10b981" },
    ],
    pros: ["切换成本最低", "单线程并发", "无需锁机制"],
    cons: ["不能真正并行", "需要异步接口", "一个阻塞全部卡住"],
  },
];

export function ProcessVsThreadVsCoroutine() {
  const [selected, setSelected] = useState(0);
  const mode = modes[selected];

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100">进程 vs 线程 vs 协程</h3>
      <div className="flex gap-3 mb-6">
        {modes.map((m, i) => (
          <button key={m.id} onClick={() => setSelected(i)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              selected === i ? "text-white shadow-md" : "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400"
            }`}
            style={selected === i ? { backgroundColor: m.color } : undefined}>
            <m.icon className="w-4 h-4" /> {m.label}
          </button>
        ))}
      </div>
      <AnimatePresence mode="wait">
        <motion.div key={selected} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
          className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6">
          <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">{mode.description}</p>
          <div className="flex gap-4 mb-4">
            {mode.items.map((item, i) => (
              <div key={i} className="flex-1 bg-slate-50 dark:bg-slate-900 rounded-lg p-3">
                <div className="text-sm font-bold mb-2 px-2 py-1 rounded text-white text-center" style={{ backgroundColor: item.color }}>{item.name}</div>
                {item.sub.map((s, j) => (
                  <motion.div key={j} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: j * 0.1 }}
                    className="text-xs px-2 py-1 ml-4 mt-1 rounded border-l-2" style={{ borderColor: item.color, color: item.color }}>
                    {s}
                  </motion.div>
                ))}
              </div>
            ))}
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <h5 className="text-xs font-bold text-green-600 dark:text-green-400 mb-1">优点</h5>
              {mode.pros.map((p, i) => (
                <div key={i} className="text-xs text-slate-600 dark:text-slate-400 flex items-center gap-1">
                  <span className="text-green-500">+</span> {p}
                </div>
              ))}
            </div>
            <div>
              <h5 className="text-xs font-bold text-red-600 dark:text-red-400 mb-1">缺点</h5>
              {mode.cons.map((c, i) => (
                <div key={i} className="text-xs text-slate-600 dark:text-slate-400 flex items-center gap-1">
                  <span className="text-red-500">-</span> {c}
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

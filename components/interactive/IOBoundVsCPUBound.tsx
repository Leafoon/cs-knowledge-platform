"use client";

import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Wifi, Cpu } from "lucide-react";

const taskTypes = [
  { id: "io", label: "I/O 密集型", icon: Wifi, color: "#3b82f6",
    description: "大部分时间在等待外部资源",
    tasks: [
      { name: "网络请求", compute: 5, wait: 500, total: 505 },
      { name: "数据库查询", compute: 3, wait: 200, total: 203 },
      { name: "文件读写", compute: 2, wait: 100, total: 102 },
      { name: "API 调用", compute: 4, wait: 300, total: 304 },
    ],
    recommendation: "协程 (asyncio) 或多线程",
    reason: "等待期间可以让其他任务执行",
  },
  { id: "cpu", label: "CPU 密集型", icon: Cpu, color: "#ef4444",
    description: "大部分时间在持续计算",
    tasks: [
      { name: "图像压缩", compute: 95, wait: 5, total: 100 },
      { name: "视频编码", compute: 98, wait: 2, total: 100 },
      { name: "数学计算", compute: 99, wait: 1, total: 100 },
      { name: "密码破解", compute: 97, wait: 3, total: 100 },
    ],
    recommendation: "多进程 (multiprocessing)",
    reason: "需要利用多个 CPU 核心真正并行计算",
  },
];

export function IOBoundVsCPUBound() {
  const [selected, setSelected] = useState(0);
  const type = taskTypes[selected];

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100">I/O 密集型 vs CPU 密集型</h3>
      <div className="flex gap-3 mb-6">
        {taskTypes.map((t, i) => (
          <button key={t.id} onClick={() => setSelected(i)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              selected === i ? "text-white shadow-md" : "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400"
            }`}
            style={selected === i ? { backgroundColor: t.color } : undefined}>
            <t.icon className="w-4 h-4" /> {t.label}
          </button>
        ))}
      </div>
      <AnimatePresence mode="wait">
        <motion.div key={selected} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
          className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5">
          <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">{type.description}</p>
          <div className="space-y-3">
            {type.tasks.map((task, i) => (
              <div key={i}>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="font-medium text-slate-700 dark:text-slate-300">{task.name}</span>
                  <span className="text-slate-500">计算 {task.compute}ms + 等待 {task.wait}ms</span>
                </div>
                <div className="h-4 bg-slate-100 dark:bg-slate-900 rounded-full overflow-hidden flex">
                  <motion.div initial={{ width: 0 }} animate={{ width: `${task.compute}%` }}
                    className="h-full" style={{ backgroundColor: selected === 0 ? "#3b82f6" : "#ef4444" }} />
                  <motion.div initial={{ width: 0 }} animate={{ width: `${task.wait}%` }}
                    className="h-full bg-slate-300 dark:bg-slate-600" />
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4 flex items-start gap-3 p-3 rounded-lg" style={{ backgroundColor: `${type.color}10` }}>
            <div className="text-2xl">{selected === 0 ? "⚡" : "⚙️"}</div>
            <div>
              <div className="text-sm font-bold" style={{ color: type.color }}>推荐方案：{type.recommendation}</div>
              <div className="text-xs text-slate-600 dark:text-slate-400">{type.reason}</div>
            </div>
          </div>
          <div className="mt-3 flex items-center gap-4 text-xs text-slate-500">
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded" style={{ backgroundColor: selected === 0 ? "#3b82f6" : "#ef4444" }} /> 计算时间</span>
            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-slate-300 dark:bg-slate-600" /> 等待时间</span>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

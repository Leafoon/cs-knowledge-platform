"use client";

import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { ArrowRight, Zap, Clock } from "lucide-react";

const COLORS = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"];

export function SyncVsAsyncTimeline() {
  const [taskCount, setTaskCount] = useState(3);
  const [delays, setDelays] = useState([2, 1, 3]);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);

  const syncTotal = delays.reduce((a, b) => a + b, 0);
  const asyncTotal = Math.max(...delays);

  useEffect(() => {
    if (!running) return;
    const interval = setInterval(() => {
      setProgress((p) => {
        if (p >= syncTotal * 10) {
          setRunning(false);
          return syncTotal * 10;
        }
        return p + 1;
      });
    }, 100);
    return () => clearInterval(interval);
  }, [running, syncTotal]);

  const handleCountChange = (n: number) => {
    setTaskCount(n);
    const newDelays = Array.from({ length: n }, (_, i) => delays[i] ?? Math.floor(Math.random() * 3) + 1);
    setDelays(newDelays);
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <Zap className="w-5 h-5" />
        同步 vs 异步时间线对比
      </h3>
      <div className="flex items-center gap-4 mb-6">
        <span className="text-sm text-slate-600 dark:text-slate-400">任务数量：</span>
        {[2, 3, 4, 5].map((n) => (
          <button key={n} onClick={() => handleCountChange(n)}
            className={`w-8 h-8 rounded-lg text-sm font-bold ${taskCount === n ? "bg-indigo-600 text-white" : "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400"}`}>
            {n}
          </button>
        ))}
        <button onClick={() => { setProgress(0); setRunning(true); }}
          className="ml-auto px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700">
          {running ? "运行中..." : "开始演示"}
        </button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4">
          <h4 className="font-bold text-slate-900 dark:text-slate-100 mb-3 flex items-center gap-2">
            <Clock className="w-4 h-4 text-red-500" /> 顺序执行（同步）
          </h4>
          <div className="relative bg-slate-50 dark:bg-slate-900 rounded-lg p-3 h-32">
            {delays.map((d, i) => {
              const start = delays.slice(0, i).reduce((a, b) => a + b, 0);
              return (
                <motion.div key={i} className="absolute h-6 rounded flex items-center justify-center text-[11px] text-white font-medium"
                  style={{ left: `${(start / syncTotal) * 100}%`, width: `${(d / syncTotal) * 100}%`, backgroundColor: COLORS[i], top: `${i * 28 + 8}px` }}
                  initial={{ scaleX: 0 }} animate={{ scaleX: progress / 10 >= start + d ? 1 : progress / 10 > start ? (progress / 10 - start) / d : 0 }}
                  transition={{ duration: 0.1 }}>
                  任务{String.fromCharCode(65 + i)}: {d}s
                </motion.div>
              );
            })}
          </div>
          <div className="mt-2 text-sm text-red-600 dark:text-red-400 font-medium">总耗时: {syncTotal} 秒</div>
        </div>
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4">
          <h4 className="font-bold text-slate-900 dark:text-slate-100 mb-3 flex items-center gap-2">
            <Zap className="w-4 h-4 text-green-500" /> 并发执行（异步）
          </h4>
          <div className="relative bg-slate-50 dark:bg-slate-900 rounded-lg p-3 h-32">
            {delays.map((d, i) => (
              <motion.div key={i} className="absolute h-6 rounded flex items-center justify-center text-[11px] text-white font-medium"
                style={{ left: "0%", width: `${(d / asyncTotal) * 100}%`, backgroundColor: COLORS[i], top: `${i * 28 + 8}px` }}
                initial={{ scaleX: 0 }} animate={{ scaleX: progress / 10 >= d ? 1 : progress / 10 > 0 ? Math.min(progress / 10 / d, 1) : 0 }}
                transition={{ duration: 0.1 }}>
                任务{String.fromCharCode(65 + i)}: {d}s
              </motion.div>
            ))}
          </div>
          <div className="mt-2 text-sm text-green-600 dark:text-green-400 font-medium">总耗时: {asyncTotal} 秒 <span className="text-slate-400 ml-2">加速比: {(syncTotal / asyncTotal).toFixed(1)}x</span></div>
        </div>
      </div>
    </div>
  );
}

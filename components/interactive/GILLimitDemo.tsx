"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Lock, Play, RotateCcw } from "lucide-react";

export function GILLimitDemo() {
  const [running, setRunning] = useState(false);
  const [thread1Progress, setThread1Progress] = useState(0);
  const [thread2Progress, setThread2Progress] = useState(0);
  const [activeThread, setActiveThread] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const start = () => {
    setRunning(true);
    setThread1Progress(0);
    setThread2Progress(0);
    setActiveThread(1);
    setElapsed(0);
    const startTime = Date.now();
    let t1 = 0, t2 = 0;
    let active = 1;

    timerRef.current = setInterval(() => {
      const e = (Date.now() - startTime) / 1000;
      setElapsed(e);

      // Simulate GIL: only one thread executes at a time
      if (active === 1) {
        t1 += 0.05;
        if (t1 >= 100) { t1 = 100; active = 2; }
      } else {
        t2 += 0.05;
        if (t2 >= 100) { t2 = 100; }
      }

      setThread1Progress(t1);
      setThread2Progress(t2);
      setActiveThread(active);

      if (t1 >= 100 && t2 >= 100) {
        setRunning(false);
        if (timerRef.current) clearInterval(timerRef.current);
      }
    }, 50);
  };

  const reset = () => {
    setRunning(false);
    setThread1Progress(0);
    setThread2Progress(0);
    setActiveThread(0);
    setElapsed(0);
    if (timerRef.current) clearInterval(timerRef.current);
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <Lock className="w-5 h-5" />
        GIL 限制演示
      </h3>
      <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
        同一时刻只有一个线程能执行 Python 字节码。注意观察两个线程的交替执行。
      </p>
      <div className="flex gap-3 mb-4">
        <button onClick={start} disabled={running}
          className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium disabled:opacity-50 flex items-center gap-2">
          <Play className="w-4 h-4" /> {running ? "运行中..." : "开始 CPU 计算"}
        </button>
        <button onClick={reset} className="px-3 py-2 bg-slate-200 dark:bg-slate-700 rounded-lg"><RotateCcw className="w-4 h-4" /></button>
        <span className="ml-auto text-sm text-slate-500 self-center">时间: {elapsed.toFixed(1)}s</span>
      </div>
      <div className="space-y-4">
        {[{ name: "线程 1", progress: thread1Progress, color: "#ef4444" }, { name: "线程 2", progress: thread2Progress, color: "#3b82f6" }].map((t, i) => (
          <div key={i} className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-slate-700 dark:text-slate-300">{t.name}</span>
              <span className="text-xs text-slate-500">{Math.round(t.progress)}%</span>
            </div>
            <div className="h-6 bg-slate-100 dark:bg-slate-900 rounded-full overflow-hidden relative">
              <motion.div className="h-full rounded-full" style={{ width: `${t.progress}%`, backgroundColor: t.color }} />
              {activeThread === i + 1 && running && (
                <motion.div className="absolute right-2 top-0.5 text-[10px] font-bold text-white"
                  animate={{ opacity: [0.3, 1, 0.3] }} transition={{ repeat: Infinity, duration: 0.5 }}>
                  执行中
                </motion.div>
              )}
            </div>
          </div>
        ))}
      </div>
      <div className="mt-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg p-3 text-xs text-amber-700 dark:text-amber-300">
        <strong>注意：</strong>两个线程交替执行，而不是同时执行。这就是 GIL 的影响 — 同一时刻只有一个线程在执行 Python 字节码。
      </div>
    </div>
  );
}

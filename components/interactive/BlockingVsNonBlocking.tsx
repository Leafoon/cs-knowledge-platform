"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Lock, Unlock, Play, RotateCcw } from "lucide-react";

export function BlockingVsNonBlocking() {
  const [blockingProgress, setBlockingProgress] = useState(0);
  const [blockingRunning, setBlockingRunning] = useState(false);
  const [blockingDone, setBlockingDone] = useState(false);
  const [nonBlockingProgress, setNonBlockingProgress] = useState(0);
  const [nonBlockingRunning, setNonBlockingRunning] = useState(false);
  const [nonBlockingChecks, setNonBlockingChecks] = useState<number[]>([]);
  const [clickCount, setClickCount] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const startBlocking = () => {
    if (blockingRunning) return;
    setBlockingRunning(true);
    setBlockingProgress(0);
    setBlockingDone(false);
    const start = Date.now();
    timerRef.current = setInterval(() => {
      const elapsed = (Date.now() - start) / 3000;
      if (elapsed >= 1) {
        setBlockingProgress(100);
        setBlockingRunning(false);
        setBlockingDone(true);
        if (timerRef.current) clearInterval(timerRef.current);
      } else {
        setBlockingProgress(elapsed * 100);
      }
    }, 50);
  };

  const startNonBlocking = () => {
    if (nonBlockingRunning) return;
    setNonBlockingRunning(true);
    setNonBlockingProgress(0);
    setNonBlockingChecks([]);
    const start = Date.now();
    const checkInterval = setInterval(() => {
      const elapsed = (Date.now() - start) / 3000;
      setNonBlockingChecks((prev) => [...prev, Date.now()]);
      if (elapsed >= 1) {
        setNonBlockingProgress(100);
        setNonBlockingRunning(false);
        clearInterval(checkInterval);
      }
    }, 800);
    timerRef.current = setInterval(() => {
      const elapsed = (Date.now() - start) / 3000;
      if (elapsed >= 1) {
        if (timerRef.current) clearInterval(timerRef.current);
      } else {
        setNonBlockingProgress(elapsed * 100);
      }
    }, 50);
  };

  const reset = () => {
    setBlockingProgress(0); setBlockingRunning(false); setBlockingDone(false);
    setNonBlockingProgress(0); setNonBlockingRunning(false); setNonBlockingChecks([]);
    setClickCount(0);
    if (timerRef.current) clearInterval(timerRef.current);
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <Lock className="w-5 h-5" />
        阻塞 vs 非阻塞
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-red-200 dark:border-red-800 p-5">
          <h4 className="font-bold text-red-600 dark:text-red-400 mb-3 flex items-center gap-2">
            <Lock className="w-4 h-4" /> 阻塞模式
          </h4>
          <p className="text-xs text-slate-500 mb-3">点击后线程被卡住，无法做其他事情</p>
          <button onClick={startBlocking} disabled={blockingRunning}
            className="w-full mb-3 px-4 py-2 bg-red-600 text-white rounded-lg text-sm font-medium disabled:opacity-50 hover:bg-red-700 flex items-center justify-center gap-2">
            <Play className="w-4 h-4" /> {blockingRunning ? "等待中... 线程被阻塞" : "启动任务（3秒）"}
          </button>
          <div className="h-4 bg-slate-100 dark:bg-slate-900 rounded-full overflow-hidden mb-2">
            <motion.div className="h-full bg-red-500 rounded-full" style={{ width: `${blockingProgress}%` }} />
          </div>
          <div className="flex gap-2 mb-2">
            <button onClick={() => setClickCount((c) => c + 1)} disabled={blockingRunning}
              className="flex-1 px-3 py-1.5 bg-slate-100 dark:bg-slate-700 rounded text-xs disabled:opacity-30">
              点击测试（{clickCount}次）
            </button>
          </div>
          <div className="text-xs text-slate-500">
            状态：<span className={blockingRunning ? "text-red-500 font-bold" : "text-green-500"}>{blockingRunning ? "线程阻塞中 - 无法响应" : blockingDone ? "完成" : "就绪"}</span>
          </div>
        </div>
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-green-200 dark:border-green-800 p-5">
          <h4 className="font-bold text-green-600 dark:text-green-400 mb-3 flex items-center gap-2">
            <Unlock className="w-4 h-4" /> 非阻塞模式
          </h4>
          <p className="text-xs text-slate-500 mb-3">点击后立即返回，线程可以继续做其他事情</p>
          <button onClick={startNonBlocking} disabled={nonBlockingRunning}
            className="w-full mb-3 px-4 py-2 bg-green-600 text-white rounded-lg text-sm font-medium disabled:opacity-50 hover:bg-green-700 flex items-center justify-center gap-2">
            <Play className="w-4 h-4" /> {nonBlockingRunning ? "任务进行中..." : "启动任务（3秒）"}
          </button>
          <div className="h-4 bg-slate-100 dark:bg-slate-900 rounded-full overflow-hidden mb-2">
            <motion.div className="h-full bg-green-500 rounded-full" style={{ width: `${nonBlockingProgress}%` }} />
          </div>
          <div className="flex gap-2 mb-2">
            <button onClick={() => setClickCount((c) => c + 1)}
              className="flex-1 px-3 py-1.5 bg-slate-100 dark:bg-slate-700 rounded text-xs">
              点击测试（{clickCount}次）
            </button>
          </div>
          <div className="text-xs text-slate-500">
            状态：<span className="text-green-500 font-bold">{nonBlockingRunning ? "任务进行中 - 线程可响应" : "就绪"}</span>
            {nonBlockingChecks.length > 0 && <span className="ml-2">轮询 {nonBlockingChecks.length} 次</span>}
          </div>
        </div>
      </div>
      <div className="mt-4 flex justify-center">
        <button onClick={reset} className="px-4 py-2 bg-slate-200 dark:bg-slate-700 rounded-lg text-sm flex items-center gap-2">
          <RotateCcw className="w-4 h-4" /> 重置
        </button>
      </div>
    </div>
  );
}

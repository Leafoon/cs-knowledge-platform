"use client";

import React, { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Ban, CheckCircle, Play, RotateCcw } from "lucide-react";

const TASK_NAMES = ["任务 A", "任务 B", "任务 C"];
const COLORS = ["#ef4444", "#3b82f6", "#10b981"];
const DELAY = 2;

export function TimeSleepVsAsyncioSleep() {
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [syncStates, setSyncStates] = useState([0, 0, 0]);
  const [asyncStates, setAsyncStates] = useState([0, 0, 0]);
  const [log, setLog] = useState<{ side: string; msg: string; time: number }[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const start = () => {
    setRunning(true);
    setElapsed(0);
    setSyncStates([0, 0, 0]);
    setAsyncStates([0, 0, 0]);
    setLog([]);
    const startTime = Date.now();

    timerRef.current = setInterval(() => {
      const t = (Date.now() - startTime) / 1000;
      setElapsed(t);

      // Sync: sequential
      setSyncStates(() => {
        const states = [0, 0, 0];
        for (let i = 0; i < 3; i++) {
          const start = i * DELAY;
          const end = start + DELAY;
          if (t <= start) states[i] = 0;
          else if (t >= end) states[i] = 100;
          else states[i] = ((t - start) / DELAY) * 100;
        }
        return states;
      });

      // Async: concurrent
      setAsyncStates(() => {
        return [0, 1, 2].map((i) => {
          if (t <= 0) return 0;
          if (t >= DELAY) return 100;
          return (t / DELAY) * 100;
        });
      });

      if (t >= DELAY * 3) {
        setRunning(false);
        if (timerRef.current) clearInterval(timerRef.current);
      }
    }, 50);
  };

  const reset = () => {
    setRunning(false);
    setElapsed(0);
    setSyncStates([0, 0, 0]);
    setAsyncStates([0, 0, 0]);
    setLog([]);
    if (timerRef.current) clearInterval(timerRef.current);
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h3 className="text-xl font-bold mb-4 text-slate-900 dark:text-slate-100 flex items-center gap-2">
        <Ban className="w-5 h-5" />
        time.sleep() vs asyncio.sleep()
      </h3>
      <div className="flex gap-3 mb-4">
        <button onClick={start} disabled={running} className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium disabled:opacity-50 flex items-center gap-2">
          <Play className="w-4 h-4" /> {running ? "运行中..." : "开始演示"}
        </button>
        <button onClick={reset} className="px-3 py-2 bg-slate-200 dark:bg-slate-700 rounded-lg"><RotateCcw className="w-4 h-4" /></button>
        <span className="ml-auto text-sm text-slate-500 self-center">时间: {elapsed.toFixed(1)}s</span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-red-200 dark:border-red-800 p-5">
          <h4 className="font-bold text-red-600 dark:text-red-400 mb-2 flex items-center gap-2">
            <Ban className="w-4 h-4" /> time.sleep() — 阻塞事件循环
          </h4>
          <p className="text-xs text-slate-500 mb-3">每个任务阻塞整个线程，其他协程无法运行</p>
          <div className="space-y-2">
            {TASK_NAMES.map((name, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="w-14 text-xs text-slate-600 dark:text-slate-400">{name}</span>
                <div className="flex-1 h-6 bg-slate-100 dark:bg-slate-900 rounded-full overflow-hidden">
                  <motion.div className="h-full rounded-full flex items-center justify-end pr-1" style={{ width: `${syncStates[i]}%`, backgroundColor: COLORS[i] }}>
                    {syncStates[i] > 50 && <span className="text-[9px] text-white">{Math.round(syncStates[i])}%</span>}
                  </motion.div>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-3 text-sm text-red-600 dark:text-red-400 font-medium">总耗时: {DELAY} × 3 = {DELAY * 3} 秒</div>
          <div className="mt-2 text-xs text-slate-500 bg-red-50 dark:bg-red-900/20 rounded p-2">
            事件循环状态：<span className="text-red-500 font-bold">被阻塞</span> — 其他协程全部暂停
          </div>
        </div>
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-green-200 dark:border-green-800 p-5">
          <h4 className="font-bold text-green-600 dark:text-green-400 mb-2 flex items-center gap-2">
            <CheckCircle className="w-4 h-4" /> await asyncio.sleep() — 非阻塞
          </h4>
          <p className="text-xs text-slate-500 mb-3">协程暂停并让出控制权，其他协程可以运行</p>
          <div className="space-y-2">
            {TASK_NAMES.map((name, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="w-14 text-xs text-slate-600 dark:text-slate-400">{name}</span>
                <div className="flex-1 h-6 bg-slate-100 dark:bg-slate-900 rounded-full overflow-hidden">
                  <motion.div className="h-full rounded-full flex items-center justify-end pr-1" style={{ width: `${asyncStates[i]}%`, backgroundColor: COLORS[i] }}>
                    {asyncStates[i] > 50 && <span className="text-[9px] text-white">{Math.round(asyncStates[i])}%</span>}
                  </motion.div>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-3 text-sm text-green-600 dark:text-green-400 font-medium">总耗时: max({DELAY}, {DELAY}, {DELAY}) = {DELAY} 秒</div>
          <div className="mt-2 text-xs text-slate-500 bg-green-50 dark:bg-green-900/20 rounded p-2">
            事件循环状态：<span className="text-green-500 font-bold">空闲可调度</span> — 其他协程正常运行
          </div>
        </div>
      </div>
    </div>
  );
}

"use client";

import React, { useState, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { Timer, Play, RotateCcw, AlertTriangle } from "lucide-react";

export function TimeoutDemo() {
  const [timeoutMs, setTimeoutMs] = useState(2000);
  const [taskDuration, setTaskDuration] = useState(3000);
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [result, setResult] = useState<"success" | "timeout" | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startRef = useRef(0);

  const startTask = () => {
    setRunning(true);
    setResult(null);
    setElapsed(0);
    startRef.current = Date.now();

    timerRef.current = setInterval(() => {
      const e = Date.now() - startRef.current;
      setElapsed(e);

      if (e >= timeoutMs) {
        clearInterval(timerRef.current!);
        setResult("timeout");
        setRunning(false);
      }
    }, 50);

    setTimeout(() => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (Date.now() - startRef.current < timeoutMs) {
        setElapsed(Date.now() - startRef.current);
        setResult("success");
        setRunning(false);
      }
    }, taskDuration);
  };

  const reset = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    setRunning(false);
    setElapsed(0);
    setResult(null);
  };

  useEffect(() => {
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, []);

  const progress = Math.min(100, (elapsed / timeoutMs) * 100);
  const timedOut = elapsed >= timeoutMs;

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-red-50 dark:from-slate-900 dark:to-red-950 rounded-xl shadow-lg">
      <h3 className="text-2xl font-bold text-slate-800 dark:text-slate-100 mb-4 text-center flex items-center justify-center gap-2">
        <Timer className="w-7 h-7 text-red-600 dark:text-red-400" />
        Timeout 演示
      </h3>

      <div className="grid grid-cols-2 gap-4 mb-4">
        <div className="bg-white dark:bg-slate-800 rounded-lg p-3 shadow">
          <label className="text-sm font-semibold text-slate-700 dark:text-slate-200">超时时间: {timeoutMs}ms</label>
          <input type="range" min={500} max={5000} step={100} value={timeoutMs} onChange={(e) => setTimeoutMs(+e.target.value)}
            className="w-full mt-1 accent-red-500" disabled={running} />
        </div>
        <div className="bg-white dark:bg-slate-800 rounded-lg p-3 shadow">
          <label className="text-sm font-semibold text-slate-700 dark:text-slate-200">任务耗时: {taskDuration}ms</label>
          <input type="range" min={500} max={6000} step={100} value={taskDuration} onChange={(e) => setTaskDuration(+e.target.value)}
            className="w-full mt-1 accent-blue-500" disabled={running} />
        </div>
      </div>

      <div className="flex justify-center gap-3 mb-5">
        <button onClick={startTask} disabled={running} className="px-5 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 dark:bg-red-500">
          <span className="flex items-center gap-1"><Play className="w-4 h-4" /> 运行任务</span>
        </button>
        <button onClick={reset} className="px-5 py-2 bg-slate-500 text-white rounded-lg hover:bg-slate-600">
          <span className="flex items-center gap-1"><RotateCcw className="w-4 h-4" /> 重置</span>
        </button>
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-lg p-4 shadow mb-4">
        <div className="flex justify-between text-sm text-slate-600 dark:text-slate-300 mb-2">
          <span>已用时间: {elapsed}ms</span>
          <span>超时阈值: {timeoutMs}ms</span>
        </div>
        <div className="relative w-full bg-slate-200 dark:bg-slate-700 rounded-full h-8 overflow-hidden">
          <motion.div className={`h-8 rounded-full ${timedOut ? "bg-red-500" : "bg-blue-500"}`} animate={{ width: `${progress}%` }} />
          <div className="absolute inset-0 flex items-center justify-center text-sm font-bold text-white">
            {running ? `${elapsed}ms / ${timeoutMs}ms` : result === "timeout" ? "超时！" : result === "success" ? "完成！" : "就绪"}
          </div>
        </div>

        {result === "timeout" && (
          <motion.div initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} className="mt-3 bg-red-50 dark:bg-red-900/30 border border-red-300 dark:border-red-700 rounded p-3 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-red-600 dark:text-red-400" />
            <span className="text-red-700 dark:text-red-300 font-semibold">asyncio.TimeoutError: 任务超时</span>
          </motion.div>
        )}
        {result === "success" && (
          <motion.div initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} className="mt-3 bg-emerald-50 dark:bg-emerald-900/30 border border-emerald-300 dark:border-emerald-700 rounded p-3">
            <span className="text-emerald-700 dark:text-emerald-300 font-semibold">任务在超时前完成 ✓</span>
          </motion.div>
        )}
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-lg p-3 text-xs font-mono text-slate-700 dark:text-slate-300 shadow">
        <p>{`try:`}</p>
        <p className="pl-4">{`async with asyncio.timeout({timeoutMs / 1000}):`}</p>
        <p className="pl-8">{`result = await long_task()`}</p>
        <p>{`except asyncio.TimeoutError:`}</p>
        <p className="pl-4">{`print("任务超时！")`}</p>
      </div>
    </div>
  );
}
